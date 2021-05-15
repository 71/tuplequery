#![cfg_attr(
    not(feature = "use_hashbrown"),
    feature(hash_drain_filter, hash_raw_entry, hash_set_entry)
)]

pub mod clause;
pub(crate) mod helpers;
pub mod join;
pub mod tuple;

#[cfg(feature = "datascript")]
pub mod datascript;

use std::borrow::Borrow;

use crate::join::Join;
pub use crate::join::{Hash, Merge};
use crate::tuple::{CloneTuple, HasFieldSet};
pub use crate::tuple::{FieldSet, Tuple, TuplePool};
pub use clause::Clause;
use smallvec::SmallVec;

#[derive(Clone)]
struct PlannedJoin {
    left_relation_index: usize,
    right_relations_indices: SmallVec<[usize; 4]>,
}

#[derive(Clone)]
struct PlannedTransform<T: Tuple> {
    resulting_fieldset: T::FieldSet,
    relation_idx: usize,
}

#[derive(Clone)]
enum PlannedOperation<T: Tuple> {
    Join(PlannedJoin),
    Read(usize),
    Transform(PlannedTransform<T>),
}

type BoxedTupleIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

/// Shorthand for [`Query::run_with`] that implicitly creates a [`Query`]
/// using the given `relations`.
pub fn query_with<'a, T: 'a + Tuple, R: 'a + Clause<T>, P: 'a + Clone + TuplePool<T>>(
    relations: impl IntoIterator<Item = R> + 'a,
    joiner: impl Join<T, P, BoxedTupleIterator<'a, T>, BoxedTupleIterator<'a, T>> + 'a,
    pool: P,
) -> Result<impl Iterator<Item = T> + 'a, QueryError> {
    let relations = relations.into_iter().collect::<Vec<_>>();
    let query = Query::new::<R, _, std::slice::Iter<R>>(relations.iter())?;

    Ok(query.run_with(relations, joiner, pool)?)
}

/// Shorthand for [`Query::run`] that implicitly creates a [`Query`] using the
/// given `relations`.
pub fn query<'a, T: 'a + CloneTuple, R: 'a + Clause<T>>(
    relations: impl IntoIterator<Item = R> + 'a,
    joiner: impl Join<T, (), BoxedTupleIterator<'a, T>, BoxedTupleIterator<'a, T>> + 'a,
) -> Result<impl Iterator<Item = T> + 'a, QueryError> {
    query_with(relations, joiner, ())
}

/// A query planned for a specific set of [`Clause`]s.
#[derive(Clone)]
pub struct Query<T: Tuple> {
    all_fields: T::FieldSet,
    operations: Vec<PlannedOperation<T>>,
    relations_fields: Vec<T::FieldSet>,
    input_fields: Vec<T::FieldSet>,
}

impl<T: Tuple> Query<T> {
    /// Returns a reference to the [`FieldSet`] that describes what output
    /// tuples contain.
    pub fn fields(&self) -> &T::FieldSet {
        &self.all_fields
    }

    /// Returns a reference to a slice of [`FieldSet`]s where each fieldset
    /// corresponds to the variables that the corresponding [`Clause`] must
    /// output.
    pub fn relation_fields(&self) -> &[T::FieldSet] {
        &self.relations_fields
    }

    /// Creates a query. The query can then be run with [`Self::run`] or
    /// [`Self::run_with`].
    #[cfg_attr(feature = "trace", tracing::instrument(skip(clauses)))]
    pub fn new<'a, R: Clause<T>, BR: Borrow<R>, II: IntoIterator<Item = BR>>(
        clauses: II,
    ) -> Result<Self, QueryPrepareError>
    where
        R: 'a,
    {
        let clauses = clauses.into_iter();
        let clauses_min_len = clauses.size_hint().0;

        let mut all_fields = T::FieldSet::default();
        let mut unsorted_clauses = Vec::with_capacity(clauses_min_len);
        let mut clauses_fields = Vec::with_capacity(clauses_min_len);

        // Discover all clauses and their variables.
        for (i, relation) in clauses.into_iter().enumerate() {
            let input_variables = relation.borrow().input_variables();
            let mut output_variables = relation.borrow().output_variables();

            // Save the "raw" output fields to later ensure that all clauses
            // given to `run` correspond to the ones given here.
            clauses_fields.push(output_variables.clone());

            // Make sure that the output variables are a superset of the input
            // variables.
            output_variables.union_in_place(&input_variables);
            all_fields.union_in_place(&output_variables);

            // Compute the "produced" variables: the variables that are added
            // by this clause.
            let produced_fields = output_variables.difference(&input_variables);

            // Add clause to dependency graph.
            unsorted_clauses.push((i, produced_fields, input_variables, output_variables));
        }

        // Sort clauses topologically; if there is a cycle, we fail.
        let sorted_clauses = crate::helpers::toposort(unsorted_clauses, |from, to| {
            let (_, _produced_variables1, input_variables1, _) = from;
            let (_, produced_variables2, _input_variables2, _) = to;

            produced_variables2.intersects(input_variables1)
        })
        .map_err(|index| QueryPrepareError::RelationCycle { index })?;

        // Prepare operations to build the output iterator.
        let mut relations_fields = Vec::with_capacity(sorted_clauses.len());
        let mut operations = Vec::with_capacity(sorted_clauses.len());

        for (ri, _, input_variables, mut output_variables) in sorted_clauses {
            let has_input_variables = !input_variables.is_empty();

            // Find common output variables with other clauses.
            let mut i = 0;
            let mut product_relations = SmallVec::new();
            let mut product_relations_variables = T::FieldSet::default();

            while let Some(output_variables2) = relations_fields.get(i) {
                if output_variables.intersects(output_variables2) {
                    if has_input_variables {
                        product_relations_variables.union_in_place(output_variables2);
                    }

                    output_variables.union_in_place(output_variables2);
                    product_relations.push(i);

                    relations_fields.swap_remove(i);
                } else {
                    i += 1;
                }
            }

            let mut is_simple_relation = true;

            if !product_relations.is_empty()
                && !(has_input_variables && product_relations.len() == 1)
            {
                // Schedule join if multiple clauses share output variables.
                let join = PlannedJoin {
                    left_relation_index: ri,
                    right_relations_indices: product_relations,
                };
                let operation = PlannedOperation::Join(join);

                operations.push(operation);
                is_simple_relation = false;
            }

            if has_input_variables {
                // Schedule transform operation if clause takes inputs.
                debug_assert!(product_relations_variables.is_superset(&input_variables));

                let transform = PlannedTransform {
                    relation_idx: ri,
                    resulting_fieldset: output_variables.clone(),
                };
                let operation = PlannedOperation::Transform(transform);

                operations.push(operation);
                is_simple_relation = false;
            }

            if is_simple_relation {
                // If the operation neither takes inputs nor needs to be joined
                // with the results of an operation, we simply read the
                // relation.
                operations.push(PlannedOperation::Read(ri));
            }

            relations_fields.push(output_variables);
        }

        Ok(Self {
            all_fields,
            operations,
            relations_fields,
            input_fields: clauses_fields,
        })
    }

    /// Runs the query with the given input relations, returning an iterator
    /// whose items are the resulting tuples of the query.
    ///
    /// Relations in `input` must be given in the same order as the ones passed
    /// to [`Self::prepare`], and must have equivalent [`FieldSet`]s.
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(skip(self, input, joiner, pool))
    )]
    pub fn run_with<
        'a,
        R: Clause<T>,
        P: Clone + TuplePool<T>,
        J: Join<T, P, Box<dyn Iterator<Item = T> + 'a>, Box<dyn Iterator<Item = T> + 'a>>,
    >(
        &self,
        input: impl IntoIterator<Item = R>,
        joiner: J,
        pool: P,
    ) -> Result<impl Iterator<Item = T> + 'a, QueryRunError>
    where
        T: 'a,
        P: 'a,
        R: 'a,
        J::IterSame: 'a,
        J::ProductIter: 'a,
    {
        // Collect and check input relations.
        let mut relations = input
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let expected_fields = self
                    .input_fields
                    .get(i)
                    .ok_or(QueryRunError::InvalidRelation { index: i })?;
                let actual_fields = r.output_variables();

                if &actual_fields != expected_fields {
                    return Err(QueryRunError::InvalidRelation { index: i });
                }

                Ok(Some((actual_fields, r)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if relations.len() != self.input_fields.len() {
            return Err(QueryRunError::InvalidRelationCount {
                expected: self.relations_fields.len(),
                found: relations.len(),
            });
        }

        let mut iterators = Vec::new();

        // Prepare output iterators.
        for operation in &self.operations {
            match operation {
                PlannedOperation::Read(relation_idx) => {
                    let (fieldset, relation) =
                        std::mem::replace(&mut relations[*relation_idx], None).unwrap();

                    iterators.push((fieldset, relation.transform_empty()));
                }
                PlannedOperation::Transform(PlannedTransform {
                    relation_idx,
                    resulting_fieldset,
                }) => {
                    let (_, relation) =
                        std::mem::replace(&mut relations[*relation_idx], None).unwrap();
                    let (fieldset, iterator) = iterators.pop().unwrap();
                    let iter = relation.transform_boxed(iterator, fieldset);

                    iterators.push((resulting_fieldset.clone(), iter));
                }
                PlannedOperation::Join(join) => {
                    let (left_fieldset, left_relation) =
                        std::mem::replace(&mut relations[join.left_relation_index], None).unwrap();
                    let left_iter = left_relation.transform_empty();

                    let mut right_relations = join
                        .right_relations_indices
                        .iter()
                        .rev()
                        .map(|i| iterators.swap_remove(*i))
                        .collect::<SmallVec<[_; 4]>>();

                    let (fieldset, iter) = if right_relations.len() == 1 {
                        let (right_fieldset, right_iter) = right_relations.pop().unwrap();
                        let union_fieldset = left_fieldset.union(&right_fieldset);
                        let join = joiner.join_same(
                            left_iter,
                            right_iter,
                            left_fieldset,
                            right_fieldset,
                            pool.clone(),
                        );

                        (union_fieldset, Box::new(join) as BoxedTupleIterator<'a, T>)
                    } else {
                        let join = joiner.join_product(
                            left_iter,
                            right_relations,
                            left_fieldset,
                            pool.clone(),
                        );
                        let union_fieldset = join.fieldset();

                        (union_fieldset, Box::new(join) as BoxedTupleIterator<'a, T>)
                    };

                    iterators.push((fieldset, iter));
                }
            }
        }

        // Return product of all output iterators.
        Ok(crate::join::product::Product::with_pool(
            iterators
                .into_iter()
                .map(|(fields, iter)| (fields, iter.collect())),
            pool,
        ))
    }

    /// Same as [`Self::run_with`], but with an implicit `pool` argument.
    pub fn run<
        'a,
        R: Clause<T>,
        J: Join<T, (), Box<dyn Iterator<Item = T> + 'a>, Box<dyn Iterator<Item = T> + 'a>>,
    >(
        &self,
        input: impl IntoIterator<Item = R> + 'a,
        joiner: J,
    ) -> Result<impl Iterator<Item = T> + 'a, QueryRunError>
    where
        T: 'a + CloneTuple,
        R: 'a,
        J: 'a,
        J::Iter: 'a,
        J::ProductIter: 'a,
    {
        self.run_with(input, joiner, ())
    }
}

/// An error encountered when preparing a [`Query`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QueryPrepareError {
    #[error("dependency cycle in relations detected")]
    RelationCycle { index: usize },
}

/// An error encountered when running a [`Query`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QueryRunError {
    #[error("invalid relation given")]
    InvalidRelation { index: usize },
    #[error("invalid number of relations given")]
    InvalidRelationCount { expected: usize, found: usize },
}

/// An error encountered when preparing or running a [`Query`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QueryError {
    #[error("failed to prepare query")]
    Prepare(#[from] QueryPrepareError),
    #[error("failed to run query")]
    Run(#[from] QueryRunError),
}

#[cfg(test)]
mod tests {
    use smallvec::SmallVec;
    use std::collections::HashMap;
    use std::convert::TryInto;

    use crate::{query, tuple::entity::RelationBuilder};

    #[derive(Copy, Clone, Ord, Eq, PartialEq, Debug, Hash)]
    struct Person {
        id: u32,
        name: &'static str,
        age: u8,
    }

    impl PartialOrd for Person {
        fn partial_cmp(&self, other: &Person) -> Option<std::cmp::Ordering> {
            self.id.partial_cmp(&other.id)
        }
    }

    #[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
    enum Value<'a> {
        Person(&'a Person),
        String(&'a str),
        Int(u8),
    }

    #[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
    struct Triple<'a>(Value<'a>, Value<'a>, Value<'a>);

    enum ValuePattern<'a> {
        Var(&'static str),
        Cond(Box<dyn Fn(&Value<'a>) -> bool>),
    }

    struct TripleClause<'a>([ValuePattern<'a>; 3]);

    fn eq_str<'a>(x: &'static str) -> Box<dyn Fn(&Value<'a>) -> bool> {
        Box::new(move |v| v == &Value::String(x))
    }

    fn run_query<'a, const N: usize>(
        clauses: &[TripleClause<'a>],
        triples: &[Triple<'a>],
        select: &[&'static str; N],
    ) -> Vec<[Value<'a>; N]> {
        let mut relations = RelationBuilder::<_, _, Vec<Option<_>>, HashMap<_, _>>::default();

        for clause in clauses {
            for triple in triples {
                let mut pairs = SmallVec::<[_; 3]>::new();

                for (var, &val) in clause.0.iter().zip(&[&triple.0, &triple.1, &triple.2]) {
                    match var {
                        ValuePattern::Cond(cond) => {
                            if !cond(val) {
                                pairs.clear();
                                break;
                            }
                        }
                        ValuePattern::Var(var) => {
                            pairs.push((*var, val.clone()));
                        }
                    }
                }

                if !pairs.is_empty() {
                    relations.push(pairs);
                }
            }
        }

        let (relations, keys) = relations.build();
        let results = query(relations, crate::Hash::new()).unwrap();

        results
            .map(|mut tuple| {
                select
                    .iter()
                    .map(|v| std::mem::replace(&mut tuple[keys[v]], None).unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn test_people() {
        let alice = Person {
            id: 0,
            name: "Alice",
            age: 22,
        };
        let bob = Person {
            id: 1,
            name: "Bob",
            age: 50,
        };
        let cid = Person {
            id: 2,
            name: "Cid",
            age: 22,
        };
        let dave = Person {
            id: 3,
            name: "Dave",
            age: 79,
        };

        let triples = [
            Triple(
                Value::Person(&alice),
                Value::String(":name"),
                Value::String(alice.name),
            ),
            Triple(
                Value::Person(&bob),
                Value::String(":name"),
                Value::String(bob.name),
            ),
            Triple(
                Value::Person(&cid),
                Value::String(":name"),
                Value::String(cid.name),
            ),
            Triple(
                Value::Person(&dave),
                Value::String(":name"),
                Value::String(dave.name),
            ),
            Triple(
                Value::Person(&alice),
                Value::String(":age"),
                Value::Int(alice.age),
            ),
            Triple(
                Value::Person(&bob),
                Value::String(":age"),
                Value::Int(bob.age),
            ),
            Triple(
                Value::Person(&cid),
                Value::String(":age"),
                Value::Int(cid.age),
            ),
            Triple(
                Value::Person(&dave),
                Value::String(":age"),
                Value::Int(dave.age),
            ),
            Triple(
                Value::Person(&alice),
                Value::String(":friend"),
                Value::Person(&cid),
            ),
            Triple(
                Value::Person(&dave),
                Value::String(":friend"),
                Value::Person(&alice),
            ),
            Triple(
                Value::Person(&bob),
                Value::String(":friend"),
                Value::Person(&dave),
            ),
        ];

        // friends with the same age
        assert_eq!(
            run_query(
                &[
                    TripleClause([
                        ValuePattern::Var("a"),
                        ValuePattern::Cond(eq_str(":age")),
                        ValuePattern::Var("age")
                    ]),
                    TripleClause([
                        ValuePattern::Var("p"),
                        ValuePattern::Cond(eq_str(":age")),
                        ValuePattern::Var("age")
                    ]),
                    TripleClause([
                        ValuePattern::Var("a"),
                        ValuePattern::Cond(eq_str(":friend")),
                        ValuePattern::Var("p")
                    ]),
                ],
                &triples,
                &["a", "p"]
            ),
            vec![[Value::Person(&alice), Value::Person(&cid)],],
        );

        // friends with the same age (reverse)
        assert_eq!(
            run_query(
                &[
                    TripleClause([
                        ValuePattern::Var("a"),
                        ValuePattern::Cond(eq_str(":age")),
                        ValuePattern::Var("age")
                    ]),
                    TripleClause([
                        ValuePattern::Var("p"),
                        ValuePattern::Cond(eq_str(":age")),
                        ValuePattern::Var("age")
                    ]),
                    TripleClause([
                        ValuePattern::Var("a"),
                        ValuePattern::Cond(eq_str(":friend")),
                        ValuePattern::Var("p")
                    ]),
                ],
                &triples,
                &["p", "a"]
            ),
            vec![[Value::Person(&cid), Value::Person(&alice)],],
        );

        // friends of bob
        assert_eq!(
            run_query(
                &[
                    TripleClause([
                        ValuePattern::Var("bob"),
                        ValuePattern::Cond(eq_str(":name")),
                        ValuePattern::Cond(eq_str("Bob"))
                    ]),
                    TripleClause([
                        ValuePattern::Var("bob"),
                        ValuePattern::Cond(eq_str(":friend")),
                        ValuePattern::Var("fob")
                    ]),
                ],
                &triples,
                &["fob"]
            ),
            vec![[Value::Person(&dave)],],
        );

        // friends of friends of bob
        assert_eq!(
            run_query(
                &[
                    TripleClause([
                        ValuePattern::Var("bob"),
                        ValuePattern::Cond(eq_str(":name")),
                        ValuePattern::Cond(eq_str("Bob"))
                    ]),
                    TripleClause([
                        ValuePattern::Var("bob"),
                        ValuePattern::Cond(eq_str(":friend")),
                        ValuePattern::Var("fob")
                    ]),
                    TripleClause([
                        ValuePattern::Var("fob"),
                        ValuePattern::Cond(eq_str(":friend")),
                        ValuePattern::Var("fofob")
                    ]),
                ],
                &triples,
                &["fofob"]
            ),
            vec![[Value::Person(&alice)],],
        );
    }
}
