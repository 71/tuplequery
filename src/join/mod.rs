//! The [`Join`] trait.

pub mod hash;
pub mod merge;
pub mod product;

use crate::tuple::{HasFieldSet, Tuple, TuplePool};

/// An interface for iterators that can be joined based on some common values
/// in a [`Tuple`].
pub trait Join<T: Tuple, P: TuplePool<T>, I1: Iterator<Item = T>, I2: Iterator<Item = T>> {
    /// The type of the iterator returned by [`join`].
    type Iter: Iterator<Item = T> + HasFieldSet<FieldSet = T::FieldSet>;

    /// The type of the iterator returned by [`join_product`].
    type ProductIter: Iterator<Item = T> + HasFieldSet<FieldSet = T::FieldSet>;

    /// Returns an iterator that produces the inner join between the tuples
    /// in `iter1` and the ones in `iter2`; the tuples in `iter1` (respectively
    /// `iter2`) will contain the fields in `fields1` (respectively `fields2`).
    ///
    /// The join should be performed on the fields returned by the intersection
    /// of `fields1` and `fields2`, which is guaranteed to be non-empty.
    fn join(
        &self,
        iter1: I1,
        iter2: I2,
        fields1: T::FieldSet,
        fields2: T::FieldSet,
        pool: P,
    ) -> Self::Iter;

    /// Same as [`join`], but with a product of relations on the right side.
    fn join_product(
        &self,
        iter1: I1,
        iter2s: impl IntoIterator<Item = (T::FieldSet, impl Iterator<Item = T>)>,
        fields1: T::FieldSet,
        pool: P,
    ) -> Self::ProductIter;
}

pub use hash::Hash;
pub use merge::Merge;

#[cfg(test)]
mod tests {
    use crate::{
        clause::relation::Relation,
        join::{hash::HashJoin, merge::MergeJoin},
        tuple::Bitset,
    };

    struct TestCase {
        input1: Vec<Vec<Option<&'static str>>>,
        input2: Vec<Vec<Option<&'static str>>>,
        fields1: Bitset,
        fields2: Bitset,
        expected: Vec<Vec<Option<&'static str>>>,
    }

    impl TestCase {
        fn new<const N: usize>(
            input: Vec<[Option<&'static str>; N]>,
            expected: Vec<[&'static str; N]>,
        ) -> Self {
            let mut split_input = Relation::split_ord(input.into_iter().map(Vec::from));

            assert_eq!(split_input.len(), 2);

            let (fields2, iter2) = split_input.pop().unwrap().into_tuple();
            let (fields1, iter1) = split_input.pop().unwrap().into_tuple();

            TestCase {
                input1: iter1.collect(),
                input2: iter2.collect(),
                fields1,
                fields2,
                expected: expected
                    .into_iter()
                    .map(Vec::from)
                    .map(|xs| xs.into_iter().map(Some).collect())
                    .collect(),
            }
        }
    }

    fn get_test_cases() -> impl Iterator<Item = TestCase> {
        vec![TestCase::new(
            vec![
                [Some("foo"), Some("bar"), None],
                [Some("bar"), Some("foo"), None],
                [Some("foo"), None, Some("baz")],
                [Some("foo"), None, Some("quux")],
                [Some("baz"), None, Some("foo")],
            ],
            vec![["foo", "bar", "baz"], ["foo", "bar", "quux"]],
        )]
        .into_iter()
    }

    #[test]
    fn test_hash_join() {
        for TestCase {
            input1,
            input2,
            fields1,
            fields2,
            expected,
        } in get_test_cases()
        {
            let mut results = HashJoin::new(input1, input2, fields1, fields2).collect::<Vec<_>>();

            results.sort();

            assert_eq!(results, expected);
        }
    }

    #[test]
    fn test_merge_join() {
        for TestCase {
            mut input1,
            mut input2,
            fields1,
            fields2,
            expected,
        } in get_test_cases()
        {
            input1.sort();
            input2.sort();

            let results = MergeJoin::new(input1, input2, fields1, fields2).collect::<Vec<_>>();

            assert_eq!(results, expected);
        }
    }
}
