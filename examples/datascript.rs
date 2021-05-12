//! This example allows queries inspired by
//! [DataScript](https://github.com/tonsky/datascript) to be run on arbitrary
//! JSON objects.
//!
//! The program reads a JSON file from stdin and a set of clauses passed as
//! arguments, and outputs the result of running those clauses on the JSON
//! object.
//!
//! Note that the syntax of the clauses is inspired but not compatible with
//! [DataScript](https://github.com/tonsky/datascript), which is itself
//! inspired by [Datomic](https://www.datomic.com)'s.
use std::{collections::HashMap, convert::TryInto, error::Error, hash::Hash};

use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_while},
    character::{complete::alphanumeric1, streaming::char},
    combinator::{map, value},
    error::context,
    multi::{many_m_n, separated_list0},
    number::complete::double,
    sequence::{delimited, preceded},
    IResult,
};
use serde_json::Value;
use tuplequery::{
    tuple::{dense::DenseTupleBuilder, entity::MutableEntityKeys, Bitset, DenseTuple},
    Query, Tuple,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Parse clauses from command line.
    let clauses = std::env::args()
        .skip(1)
        .map(|x| clause(&x).map(|x| x.1).map_err(|x| x.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    if clauses.is_empty() {
        let name = std::env::args().next().unwrap();

        eprintln!(r#"
USAGE:
    command | {} [CLAUSES ...]

EXAMPLE:
    curl file.json | {} '?a "age" ?age' '?b "age" ?age'"#, name, name);

        std::process::exit(1);
    }

    // Parse input object from stdin.
    let value = serde_json::from_reader(std::io::stdin())?;

    // Convert input object into triples.
    let root = Val::Entity(&value);
    let root_prop = Val::Str("root".into());
    let mut triples = Vec::new();

    value_to_triples(root, root_prop, &value, &mut triples);

    // Build clauses.
    let mut relations = Vec::with_capacity(clauses.len());
    let mut variables = HashMap::<String, usize>::new();

    for clause in clauses {
        let relation = match clause {
            Clause::Assign(name, mut expr) => {
                let mut bitset = Bitset::new();

                expr.assign_variables(&mut variables, &mut bitset);

                let name_idx = variables.find_index_or_insert(name.clone());

                if bitset.has(name_idx) {
                    return Err(format!("name {:?} has already been assigned", name).into());
                }

                BuiltClause::Assign(name_idx, bitset, expr)
            }
            Clause::Triple(patterns) => {
                let mut bitset = None;
                let mut relation = Vec::new();

                for triple in &triples {
                    let mut tuple_builder = DenseTupleBuilder::new();
                    let mut is_match = true;

                    for (val, pat) in triple.iter().zip(&patterns) {
                        match (val, pat) {
                            (Val::Num(actual), Pat::EqNum(expected)) if actual == expected => {
                                // Match.
                            }
                            (Val::Str(actual), Pat::EqStr(expected)) if actual == expected => {
                                // Match.
                            }
                            (Val::Bool(actual), Pat::EqBool(expected)) if actual == expected => {
                                // Match.
                            }
                            (val, Pat::Var(var)) => {
                                // Assign to variable.
                                let index = variables.find_index_or_insert(var.clone());

                                tuple_builder.set(index, val.clone());
                            }
                            _ => {
                                is_match = false;

                                break;
                            }
                        }
                    }

                    if is_match {
                        let (tuple, tuple_bitset) = tuple_builder.build();

                        if bitset.is_none() {
                            bitset = Some(tuple_bitset);
                        }

                        relation.push(tuple);
                    }
                }

                BuiltClause::Triples(relation, bitset.unwrap())
            }
            Clause::Where(mut expr) => {
                let mut bitset = Bitset::new();

                expr.assign_variables(&mut variables, &mut bitset);

                BuiltClause::Where(expr, bitset)
            }
        };

        relations.push(relation);
    }

    // Build query.
    let query = Query::new::<BuiltClause, _, _>(relations.iter())?;

    // Run query.
    let results = query.run(relations, tuplequery::Hash::new())?;

    // Display query results.
    for mut result_tuple in results {
        // SAFETY: the fieldset is returned by the `query`, so it is valid to
        // access tuple values with it.
        let result = variables
            .iter()
            .map(|(name, idx)| (name, unsafe { result_tuple.get(*idx, query.fields()) }.unwrap()))
            .collect::<HashMap<_, _>>();

        println!("{:?}", result);

        result_tuple.clear(query.fields());
    }

    Ok(())
}

/// A built [`Clause`] with variables resolved to numbers, triples converted to
/// [`DenseTuple`]s, and clause [`Bitset`]s resolved.
enum BuiltClause<'a> {
    Assign(usize, Bitset, Expr),
    Triples(Vec<DenseTuple<Val<'a>, 3>>, Bitset),
    Where(Expr, Bitset),
}

impl<'a> tuplequery::Clause<DenseTuple<Val<'a>, 3>> for BuiltClause<'a> {
    fn input_variables(&self) -> Bitset {
        match self {
            BuiltClause::Assign(_, vars, _) => vars.clone(),
            BuiltClause::Triples(..) => Bitset::new(),
            BuiltClause::Where(_, vars) => vars.clone(),
        }
    }

    fn output_variables(&self) -> Bitset {
        match self {
            BuiltClause::Assign(idx, vars, _) => {
                let mut vars = vars.clone();

                vars.on(*idx);
                vars
            }
            BuiltClause::Triples(_, vars) => vars.clone(),
            BuiltClause::Where(_, vars) => vars.clone(),
        }
    }

    fn transform_boxed<'b>(
        self,
        input: Box<dyn Iterator<Item = DenseTuple<Val<'a>, 3>> + 'b>,
        all_variables: Bitset,
    ) -> Box<dyn Iterator<Item = DenseTuple<Val<'a>, 3>> + 'b>
    where
        Self: 'b,
        DenseTuple<Val<'a>, 3>: 'b,
        Bitset: 'b,
    {
        match self {
            BuiltClause::Assign(i, vars, expr) => Box::new(input.map(move |mut tuple| {
                let mut assigned_tuple_builder = DenseTupleBuilder::new();

                assigned_tuple_builder.set(i, expr.eval(&tuple, &vars));

                let (assigned_tuple, assigned_bitset) = assigned_tuple_builder.build();

                tuple.merge_owned(assigned_tuple, &all_variables, &assigned_bitset);
                tuple
            })),
            BuiltClause::Triples(triples, _) => Box::new(triples.into_iter()),
            BuiltClause::Where(expr, vars) => {
                Box::new(input.filter(move |tuple| expr.eval(&tuple, &vars).is_truthy()))
            }
        }
    }

    fn transform_empty<'b>(self) -> Box<dyn Iterator<Item = DenseTuple<Val<'a>, 3>> + 'b>
    where
        Self: 'b,
        DenseTuple<Val<'a>, 3>: 'b,
        Bitset: 'b,
    {
        if let BuiltClause::Triples(triples, _) = self {
            Box::new(triples.into_iter())
        } else {
            unreachable!()
        }
    }
}

/// A value of a [`Triple`].
#[derive(Debug, Clone, PartialEq)]
enum Val<'a> {
    Entity(&'a Value),
    Str(String),
    Num(f64),
    Null,
    Bool(bool),
}

impl<'a> Hash for Val<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Val::Entity(x) => {
                state.write_u8(1);
                state.write_usize(*x as *const Value as usize)
            }
            Val::Str(x) => {
                state.write_u8(2);
                x.hash(state)
            }
            Val::Num(x) => {
                state.write_u8(3);
                // This is bad because values come both from computations
                // and user input and so their bits may not be equal, but
                // heh...
                (*x as u64).hash(state)
            }
            Val::Null => {
                state.write_u8(4);
                ().hash(state)
            }
            Val::Bool(x) => {
                state.write_u8(5);
                x.hash(state)
            }
        }
    }
}

impl<'a> Eq for Val<'a> {}

impl<'a> Val<'a> {
    fn is_truthy(&self) -> bool {
        match self {
            Val::Entity(_) => true,
            Val::Str(s) => !s.is_empty(),
            Val::Num(v) => *v != 0. && *v != -0.,
            Val::Null => false,
            Val::Bool(v) => *v,
        }
    }
}

/// A triple of [`Val`]s.
type Triple<'a> = [Val<'a>; 3];

/// Converts a JSON [`Value`] into a collection of [`Triple`]s that represent
/// that object and that can be queried.
fn value_to_triples<'a>(
    entity: Val<'a>,
    prop: Val<'a>,
    value: &'a Value,
    triples: &mut Vec<Triple<'a>>,
) {
    match value {
        Value::Null => triples.push([entity, prop, Val::Null]),
        Value::Bool(bool) => triples.push([entity, prop, Val::Bool(*bool)]),
        Value::Number(number) => triples.push([entity, prop, Val::Num(number.as_f64().unwrap())]),
        Value::String(string) => triples.push([entity, prop, Val::Str(string.clone())]),
        Value::Array(array) => {
            let array_entity = Val::Entity(value);

            triples.push([entity, prop, array_entity.clone()]);

            for (i, v) in array.iter().enumerate() {
                value_to_triples(array_entity.clone(), Val::Num(i as f64), v, triples);
            }
        }
        Value::Object(object) => {
            let object_entity = Val::Entity(value);

            triples.push([entity, prop, object_entity.clone()]);

            for (k, v) in object {
                value_to_triples(object_entity.clone(), Val::Str(k.clone()), v, triples);
            }
        }
    }
}

/// An expression in a [`Clause`].
#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Eq(Box<Expr>, Box<Expr>),
    Var(String, usize),
    Bool(bool),
    Num(f64),
    Str(String),
}

impl Expr {
    fn assign_variables(&mut self, variables: &mut HashMap<String, usize>, bitset: &mut Bitset) {
        match self {
            Expr::Var(str, idx) => {
                *idx = variables.find_index_or_insert(str.clone());
                bitset.on(*idx);
            }
            Expr::Eq(lhs, rhs) => {
                lhs.assign_variables(variables, bitset);
                rhs.assign_variables(variables, bitset);
            }
            _ => (),
        }
    }

    fn eval<'a>(&self, tuple: &DenseTuple<Val<'a>, 3>, fields: &Bitset) -> Val<'a> {
        // SAFETY: we assume that the given fields correspond to the tuple.
        match self {
            Expr::Eq(a, b) => Val::Bool(a.eval(tuple, fields) == b.eval(tuple, fields)),
            Expr::Var(_, i) => unsafe { tuple.get(*i, fields) }.unwrap().clone(),
            Expr::Bool(v) => Val::Bool(*v),
            Expr::Num(v) => Val::Num(*v),
            Expr::Str(v) => Val::Str(v.clone()),
        }
    }
}

/// A pattern to match on in a [`Clause::Triple`].
#[derive(Debug)]
enum Pat {
    Var(String),
    EqBool(bool),
    EqNum(f64),
    EqStr(String),
}

/// A clause of a query.
#[derive(Debug)]
enum Clause {
    Triple([Pat; 3]),
    Assign(String, Expr),
    Where(Expr),
}

/// Parses whitespace characters.
///
/// ```peg
/// ws = [ \t\n\r]*
/// ```
fn ws(input: &str) -> IResult<&str, ()> {
    value((), take_while(|c| " \t\n\r".contains(c)))(input)
}

/// Parses a variable.
///
/// ```peg
/// var = "?" [a-zA-Z0-9]+
/// ```
fn var(input: &str) -> IResult<&str, &str> {
    let (input, _) = tag("?")(input)?;
    let (input, varname) = alphanumeric1(input)?;

    Ok((input, varname))
}

/// Parses a boolean literal.
///
/// ```peg
/// bool = "true"
///      / "false"
/// ```
fn bool(input: &str) -> IResult<&str, bool> {
    alt((value(true, tag("true")), value(false, tag("false"))))(input)
}

/// Parses a number literal.
///
/// ```peg
/// num = DOUBLE
/// ```
fn num(input: &str) -> IResult<&str, f64> {
    context("number", double)(input)
}

/// Parses a string literal.
///
/// ```peg
/// str = "\"" [^"]* "\""
/// ```
fn str(input: &str) -> IResult<&str, &str> {
    context("string", delimited(char('"'), is_not("\""), char('"')))(input)
}

/// Parses a named function [`Expr`].
///
/// ```peg
/// named_fn name = "<name>" "(" (ws expr ws ",")* ws expr ws ")"
/// ```
fn named_fn<const PARAMS: usize>(
    name: &'static str,
    ctor: impl Fn([Expr; PARAMS]) -> Expr,
) -> impl FnMut(&str) -> IResult<&str, Expr> {
    move |input: &str| {
        preceded(
            tag(name),
            delimited(
                tag("("),
                map(separated_list0(delimited(ws, tag(","), ws), expr), |args| {
                    ctor(args.try_into().unwrap())
                }),
                tag(")"),
            ),
        )(input)
    }
}

/// Parses an [`Expr`].
///
/// ```peg
/// expr = "eq(" ws expr ws "," ws expr ws ")"
///      / bool
///      / num
///      / str
///      / var
/// ```
fn expr(input: &str) -> IResult<&str, Expr> {
    alt((
        named_fn("eq", |[a, b]| Expr::Eq(Box::new(a), Box::new(b))),
        map(bool, Expr::Bool),
        map(num, Expr::Num),
        map(str, |x| Expr::Str(x.to_string())),
        map(var, |x| Expr::Var(x.to_string(), 0)),
    ))(input)
}

/// Parses a [`Pat`].
///
/// ```peg
/// pat = bool
///     / num
///     / str
///     / var
/// ```
fn pat(input: &str) -> IResult<&str, Pat> {
    alt((
        map(bool, Pat::EqBool),
        map(num, Pat::EqNum),
        map(str, |x| Pat::EqStr(x.to_string())),
        map(var, |x| Pat::Var(x.to_string())),
    ))(input)
}

/// Parses a triple of [`Pat`]s.
///
/// ```peg
/// triple = (ws pat){3}
/// ```
fn triple(input: &str) -> IResult<&str, [Pat; 3]> {
    map(many_m_n(3, 3, preceded(ws, pat)), |x| x.try_into().unwrap())(input)
}

/// Parses an assignment from an [`Expr`] to a variable.
///
/// ```peg
/// assign = ws var ws "=" ws expr
/// ```
fn assign(input: &str) -> IResult<&str, (&str, Expr)> {
    let (input, var) = preceded(ws, var)(input)?;
    let (input, _) = preceded(ws, tag("="))(input)?;
    let (input, expr) = preceded(ws, expr)(input)?;

    Ok((input, (var, expr)))
}

/// Parses a [`Clause`].
///
/// ```peg
/// clause = "where" ws expr
///        / assign
///        / triple
/// ```
fn clause(input: &str) -> IResult<&str, Clause> {
    let (input, _) = ws(input)?;

    alt((
        map(preceded(tag("where"), preceded(ws, expr)), Clause::Where),
        map(assign, |(name, expr)| {
            Clause::Assign(name.to_string(), expr)
        }),
        map(triple, Clause::Triple),
    ))(input)
}
