//! This example allows queries inspired by
//! [DataScript](https://github.com/tonsky/datascript) to be run on arbitrary
//! JSON objects.
//!
//! Note that the syntax of the clauses is inspired but not compatible with
//! [DataScript](https://github.com/tonsky/datascript), which is itself
//! inspired by [Datomic](https://www.datomic.com)'s.
use std::{
    collections::HashMap, convert::TryInto, error::Error, hash::Hash, iter::FromIterator, usize,
};

use crate::{
    tuple::{
        dense::DenseTupleBuilder, entity::MutableEntityKeys, Bitset, CloneTuple, DenseTuple,
        EqTuple, HashTuple, SparseTuple,
    },
    Query, Tuple,
};
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
use serde_json::{Number, Value};
use smallvec::SmallVec;

/// Parses a clause.
pub fn parse_clause(input: &str) -> Result<Clause, Box<dyn Error>> {
    clause(input).map(|x| x.1).map_err(|x| x.to_string().into())
}

/// Runs a query built on the given clauses using the given triples.
pub fn query<'a, T: SupportedTuple<'a>>(
    clauses: Vec<Clause<'a>>,
    triples: &[Triple<'a>],
) -> Result<(Vec<T>, Query<T>, HashMap<&'a str, usize>), Box<dyn Error>> {
    // Build clauses.
    let mut relations = Vec::with_capacity(clauses.len());
    let mut variables = HashMap::new();

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

                for triple in triples {
                    let mut tuple = SmallVec::<[Option<Val<'a>>; 3]>::with_capacity(3);
                    let mut tuple_bitset = Bitset::new();
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

                                if tuple.len() <= index {
                                    tuple.resize_with(index, || None);
                                    tuple.push(Some(val.clone()));
                                } else {
                                    tuple[index] = Some(val.clone());
                                }

                                tuple_bitset.on(index);
                            }
                            _ => {
                                is_match = false;

                                break;
                            }
                        }
                    }

                    if is_match {
                        if bitset.is_none() {
                            bitset = Some(tuple_bitset);
                        }

                        relation.push(tuple.into_iter().collect());
                    }
                }

                let bitset = if let Some(bitset) = bitset {
                    bitset
                } else {
                    continue;
                };

                BuiltClause::Triples(relation, bitset)
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
    let query = Query::new::<BuiltClause<T>, _, _>(relations.iter())?;

    // Run query.
    let results = query.run(relations, crate::Hash::new())?.collect();

    Ok((results, query, variables))
}

pub trait SupportedTuple<'a>:
    Tuple<FieldSet = Bitset> + CloneTuple + EqTuple + HashTuple + FromIterator<Option<Val<'a>>> + 'a
{
    fn assign_at(&mut self, index: usize, self_fields: &Bitset, value: Val<'a>);
    fn value_at(&self, index: usize, self_fields: &Bitset) -> Option<&Val<'a>>;

    fn to_json(mut self, self_fields: &Bitset, variables: &HashMap<&'a str, usize>) -> Value {
        let result = Value::Object(
            variables
                .iter()
                .map(|(name, idx)| {
                    (
                        name.to_string(),
                        self.value_at(*idx, self_fields).unwrap().clone().to_json(),
                    )
                })
                .collect(),
        );

        self.clear(self_fields);

        result
    }
}

impl<'a, const N: usize> SupportedTuple<'a> for DenseTuple<Val<'a>, N> {
    fn assign_at(&mut self, index: usize, self_fields: &Bitset, value: Val<'a>) {
        let mut assigned_tuple_builder = DenseTupleBuilder::new();

        assigned_tuple_builder.set(index, value);

        let (assigned_tuple, assigned_bitset) = assigned_tuple_builder.build();

        self.merge_owned(assigned_tuple, self_fields, &assigned_bitset);
    }

    fn value_at(&self, index: usize, self_fields: &Bitset) -> Option<&Val<'a>> {
        // SAFETY: we assume that the given fields correspond to the tuple.
        unsafe { self.get(index, self_fields) }
    }
}

impl<'a, const N: usize> SupportedTuple<'a> for SparseTuple<Val<'a>, N> {
    fn assign_at(&mut self, index: usize, self_fields: &Bitset, value: Val<'a>) {
        let mut assigned_tuple = SmallVec::<[_; 4]>::with_capacity(index + 1);
        let mut assigned_bitset = Bitset::new();

        assigned_tuple.resize_with(index, || None);
        assigned_tuple.push(Some(value));

        assigned_bitset.on(index);

        self.merge_owned(
            assigned_tuple.into_iter().collect(),
            self_fields,
            &assigned_bitset,
        );
    }

    fn value_at(&self, index: usize, self_fields: &Bitset) -> Option<&Val<'a>> {
        self.get(self_fields, index)
    }
}

impl<'a, const N: usize> SupportedTuple<'a> for SmallVec<[Option<Val<'a>>; N]> {
    fn assign_at(&mut self, index: usize, _self_fields: &Bitset, value: Val<'a>) {
        self[index] = Some(value);
    }

    fn value_at(&self, index: usize, _self_fields: &Bitset) -> Option<&Val<'a>> {
        self.get(index)?.as_ref()
    }
}

impl<'a> SupportedTuple<'a> for Vec<Option<Val<'a>>> {
    fn assign_at(&mut self, index: usize, _self_fields: &Bitset, value: Val<'a>) {
        self[index] = Some(value);
    }

    fn value_at(&self, index: usize, _self_fields: &Bitset) -> Option<&Val<'a>> {
        self.get(index)?.as_ref()
    }
}

/// A built [`Clause`] with variables resolved to numbers, triples converted to
/// [`DenseTuple`]s, and clause [`Bitset`]s resolved.
enum BuiltClause<'a, T> {
    Assign(usize, Bitset, Expr<'a>),
    Triples(Vec<T>, Bitset),
    Where(Expr<'a>, Bitset),
}

impl<'a, T: SupportedTuple<'a>> crate::Clause<T> for BuiltClause<'a, T> {
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
        input: Box<dyn Iterator<Item = T> + 'b>,
        all_variables: Bitset,
    ) -> Box<dyn Iterator<Item = T> + 'b>
    where
        Self: 'b,
        T: 'b,
        Bitset: 'b,
    {
        match self {
            BuiltClause::Assign(i, vars, expr) => Box::new(input.map(move |mut tuple| {
                tuple.assign_at(i, &all_variables, expr.eval(&tuple, &vars));
                tuple
            })),
            BuiltClause::Triples(triples, _) => Box::new(triples.into_iter()),
            BuiltClause::Where(expr, vars) => {
                Box::new(input.filter(move |tuple| expr.eval(tuple, &vars).is_truthy()))
            }
        }
    }

    fn transform_empty<'b>(self) -> Box<dyn Iterator<Item = T> + 'b>
    where
        Self: 'b,
        T: 'b,
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
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Val<'a> {
    Entity(usize),
    Str(&'a str),
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

    fn to_json(self) -> Value {
        match self {
            Val::Entity(n) => Value::String(format!("entity #{}", n)),
            Val::Str(s) => Value::String(s.to_string()),
            Val::Num(v) => Value::Number(Number::from_f64(v).unwrap()),
            Val::Null => Value::Null,
            Val::Bool(v) => Value::Bool(v),
        }
    }
}

/// A triple of [`Val`]s.
pub type Triple<'a> = [Val<'a>; 3];

fn write_value_to_triples<'a>(
    entity: Val<'a>,
    prop: Val<'a>,
    value: &'a Value,
    triples: &mut Vec<Triple<'a>>,
) {
    match value {
        Value::Null => triples.push([entity, prop, Val::Null]),
        Value::Bool(bool) => triples.push([entity, prop, Val::Bool(*bool)]),
        Value::Number(number) => triples.push([entity, prop, Val::Num(number.as_f64().unwrap())]),
        Value::String(string) => triples.push([entity, prop, Val::Str(string)]),
        Value::Array(array) => {
            let array_entity = Val::Entity(triples.len());

            triples.push([entity, prop, array_entity.clone()]);

            for (i, v) in array.iter().enumerate() {
                write_value_to_triples(array_entity.clone(), Val::Num(i as f64), v, triples);
            }
        }
        Value::Object(object) => {
            let object_entity = Val::Entity(triples.len());

            triples.push([entity, prop, object_entity.clone()]);

            for (k, v) in object {
                write_value_to_triples(object_entity.clone(), Val::Str(k), v, triples);
            }
        }
    }
}

/// Converts a JSON [`Value`] into a collection of [`Triple`]s that represent
/// that object and that can be queried, and writes them to the given vector.
pub fn value_to_triples<'a>(value: &'a Value) -> Vec<Triple<'a>> {
    let root = Val::Entity(0);
    let root_prop = Val::Str("root");

    let mut triples = Vec::new();

    write_value_to_triples(root, root_prop, value, &mut triples);

    triples
}

/// An expression in a [`Clause`].
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Eq(Box<Expr<'a>>, Box<Expr<'a>>),
    Var(&'a str, usize),
    Bool(bool),
    Num(f64),
    Str(&'a str),
}

impl<'a> Expr<'a> {
    fn assign_variables(&mut self, variables: &mut HashMap<&'a str, usize>, bitset: &mut Bitset) {
        match self {
            Expr::Var(str, idx) => {
                *idx = variables.find_index_or_insert(*str);
                bitset.on(*idx);
            }
            Expr::Eq(lhs, rhs) => {
                lhs.assign_variables(variables, bitset);
                rhs.assign_variables(variables, bitset);
            }
            _ => (),
        }
    }

    fn eval<T: SupportedTuple<'a>>(&self, tuple: &T, fields: &Bitset) -> Val<'a> {
        match self {
            Expr::Eq(a, b) => Val::Bool(a.eval(tuple, fields) == b.eval(tuple, fields)),
            Expr::Var(_, i) => tuple.value_at(*i, fields).unwrap().clone(),
            Expr::Bool(v) => Val::Bool(*v),
            Expr::Num(v) => Val::Num(*v),
            Expr::Str(v) => Val::Str(v),
        }
    }
}

/// A pattern to match on in a [`Clause::Triple`].
#[derive(Debug, Clone)]
pub enum Pat<'a> {
    Var(&'a str),
    EqBool(bool),
    EqNum(f64),
    EqStr(&'a str),
}

/// A clause of a query.
#[derive(Debug, Clone)]
pub enum Clause<'a> {
    Triple([Pat<'a>; 3]),
    Assign(&'a str, Expr<'a>),
    Where(Expr<'a>),
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
        map(str, |x| Expr::Str(x)),
        map(var, |x| Expr::Var(x, 0)),
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
        map(str, |x| Pat::EqStr(x)),
        map(var, |x| Pat::Var(x)),
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
        map(assign, |(name, expr)| Clause::Assign(name, expr)),
        map(triple, Clause::Triple),
    ))(input)
}
