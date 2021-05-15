//! The [`Merge`] struct, which provides a merge-join based implementation of
//! [`Join`].

use std::cmp::Ordering;
use std::iter::Peekable;

use crate::tuple::{CloneTuple, FieldSet, HasFieldSet, OrdTuple, Tuple, TuplePool};

use super::product::Product;
use super::Join;

/// A [`Join`] implementation that uses a [`MergeJoin`].
#[derive(Clone, Debug, Default)]
pub struct Merge {}

impl Merge {
    /// Creates a new [`Merge`] struct.
    pub const fn new() -> Merge {
        Self {}
    }
}

impl<T: OrdTuple, P: Clone + TuplePool<T>, I1: Iterator<Item = T>, I2: Iterator<Item = T>>
    Join<T, P, I1, I2> for Merge
{
    type Iter = MergeJoin<T, P, I1, I2>;
    type IterSame = MergeJoin<T, P, I1, I1>;
    type ProductIter = MergeJoin<T, P, I1, Product<T, P>>;

    fn join(
        &self,
        iter1: I1,
        iter2: I2,
        fields1: T::FieldSet,
        fields2: T::FieldSet,
        pool: P,
    ) -> Self::Iter {
        MergeJoin::with_pool(iter1, iter2, fields1, fields2, pool)
    }

    fn join_same(
        &self,
        iter1: I1,
        iter2: I1,
        fields1: T::FieldSet,
        fields2: T::FieldSet,
        pool: P,
    ) -> Self::IterSame {
        MergeJoin::with_pool(iter1, iter2, fields1, fields2, pool)
    }

    fn join_product(
        &self,
        iter1: I1,
        iter2s: impl IntoIterator<Item = (T::FieldSet, impl Iterator<Item = T>)>,
        fields1: T::FieldSet,
        pool: P,
    ) -> Self::ProductIter {
        let iter2 = Product::with_pool(
            iter2s
                .into_iter()
                .map(|(fields, iter)| (fields, iter.collect())),
            pool.clone(),
        );
        let fields2 = iter2.fieldset();

        MergeJoin::with_pool(iter1, iter2, fields1, fields2, pool)
    }
}

/// An iterator over the the merge-join of two iterators.
///
/// This iterator requires that `T` is [`Ord`] and that its inputs are sorted,
/// but should be more efficient than
/// [`HashJoin`](crate::join::hash::HashJoin).
#[derive(Clone)]
pub struct MergeJoin<T: Tuple, P: TuplePool<T>, I1: Iterator<Item = T>, I2: Iterator<Item = T>> {
    // The iterators to join.
    iter1: Peekable<I1>,
    iter2: Peekable<I2>,

    // The fields
    fields1: T::FieldSet,
    fields2: T::FieldSet,
    shared_fields: T::FieldSet,

    pending1: Vec<T>,
    pending2: Vec<T>,
    pending2_index: usize,

    pool: P,
}

impl<T: OrdTuple + CloneTuple, I1: Iterator<Item = T>, I2: Iterator<Item = T>>
    MergeJoin<T, (), I1, I2>
{
    /// Creates a new [`MergeJoin`] iterator. The two given iterators must be
    /// sorted.
    pub fn new(
        iter1: impl IntoIterator<IntoIter = I1, Item = T>,
        iter2: impl IntoIterator<IntoIter = I2, Item = T>,
        fields1: T::FieldSet,
        fields2: T::FieldSet,
    ) -> Self {
        Self::with_pool(iter1, iter2, fields1, fields2, ())
    }
}

impl<T: OrdTuple, P: TuplePool<T>, I1: Iterator<Item = T>, I2: Iterator<Item = T>>
    MergeJoin<T, P, I1, I2>
{
    /// Same as [`Self::new`], but also allows an explicit [`TuplePool`] to be
    /// given.
    pub fn with_pool(
        iter1: impl IntoIterator<IntoIter = I1, Item = T>,
        iter2: impl IntoIterator<IntoIter = I2, Item = T>,
        fields1: T::FieldSet,
        fields2: T::FieldSet,
        pool: P,
    ) -> Self {
        MergeJoin {
            shared_fields: fields1.intersect(&fields2),
            iter1: iter1.into_iter().peekable(),
            iter2: iter2.into_iter().peekable(),
            fields1,
            fields2,
            pool,

            pending1: Vec::new(),
            pending2: Vec::new(),
            pending2_index: 0,
        }
    }
}

impl<T: OrdTuple, P: TuplePool<T>, I1: Iterator<Item = T>, I2: Iterator<Item = T>> Iterator
    for MergeJoin<T, P, I1, I2>
{
    type Item = T;

    #[cfg_attr(feature = "trace", tracing::instrument(skip(self)))]
    fn next(&mut self) -> Option<Self::Item> {
        // If some values are pending, return their cartesian product.
        if let Some(pending1) = self.pending1.last() {
            let next_pending = if self.pending2_index + 1 == self.pending2.len() {
                // This is the last item in `pending2`, so we switch to the
                // next value (if any) of `pending1` for the next iteration.
                self.pending2_index = 0;

                let mut pending1 = self.pending1.pop().unwrap();
                let pending2 = self.pending2.last().unwrap();

                pending1.merge(pending2, &self.fields1, &self.fields2);
                pending1
            } else {
                self.pending2_index += 1;

                let mut pending1 = self.pool.clone_tuple(pending1, &self.fields1);
                let pending2 = self.pending2.get(self.pending2_index - 1).unwrap();

                pending1.merge(pending2, &self.fields1, &self.fields2);
                pending1
            };

            return Some(next_pending);
        }

        loop {
            // Peek at the next values in each input iterator.
            let v1 = self.iter1.peek()?;
            let v2 = self.iter2.peek()?;

            // Compare the fields they share. The call below may return
            // `Equal` even if `v1 != v2`. What matters is that the fields
            // they have in common are indeed equal.
            match v1.cmp(v2, &self.shared_fields) {
                Ordering::Equal => {
                    // Both tuples have equal common fields, take them.
                    let mut v1 = self.iter1.next().unwrap();
                    let v2 = self.iter2.next().unwrap();

                    // Find all subsequent values that have equal keys, and put
                    // them in two `pending{1,2}` vectors.
                    // In subsequent loops, the cartesian product of pending
                    // tuples will be returned.
                    self.pending2.clear();

                    // (Take values in `iter1` whose relevant fields equal
                    // those of `v1`.)
                    while let Some(next_v1) = self.iter1.peek() {
                        if v1.cmp(next_v1, &self.shared_fields) != Ordering::Equal {
                            break;
                        }

                        self.pending1.push(self.iter1.next().unwrap());
                    }

                    // (Take values in `iter2` whose relevant fields equal
                    // those of `v2`.)
                    self.pending2.push(v2);

                    while let Some(next_v2) = self.iter2.peek() {
                        if self.pending2[0].cmp(next_v2, &self.shared_fields) != Ordering::Equal {
                            break;
                        }

                        self.pending2.push(self.iter2.next().unwrap());
                    }

                    // Reverse `pending1`, since it is iterated on from end to
                    // start in the cartesian product at the start of `next()`.
                    self.pending1.reverse();

                    if self.pending2.len() == 1 {
                        self.pending2_index = 0;

                        if self.pending1.len() == 0 {
                            v1.merge_owned(
                                self.pending2.pop().unwrap(),
                                &self.fields1,
                                &self.fields2,
                            );
                        } else {
                            v1.merge(&self.pending2[0], &self.fields1, &self.fields2);
                        }
                    } else {
                        self.pending1
                            .push(self.pool.clone_tuple(&v1, &self.fields1));
                        self.pending2_index = 1;

                        v1.merge(&self.pending2[0], &self.fields1, &self.fields2);
                    }

                    return Some(v1);
                }

                // `v1` and `v2` have different relevant fields. Advance
                // whichever iterator needs to "catch up" on the other.
                Ordering::Greater => {
                    self.iter2.next();
                }
                Ordering::Less => {
                    self.iter1.next();
                }
            }
        }
    }
}

impl<T: Tuple, P: TuplePool<T>, I1: Iterator<Item = T>, I2: Iterator<Item = T>> HasFieldSet
    for MergeJoin<T, P, I1, I2>
{
    type FieldSet = T::FieldSet;

    fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
        None
    }

    fn fieldset(&self) -> Self::FieldSet {
        self.fields1.union(&self.fields2)
    }
}
