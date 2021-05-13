//! The [`Product`] iterator.

use std::mem::MaybeUninit;

use smallvec::{smallvec, SmallVec};

use crate::tuple::{CloneTuple, FieldSet, HasFieldSet, Tuple, TuplePool};

/// [`SmallVec`]s used inside [`Product`]. Most users won't have to deal with
/// n-ary products where `n >= 8`. In those cases, an allocation for a vector
/// will be insignificant compared to the actual work of the product.
type ProductVec<T> = SmallVec<[T; 8]>;

/// Computes the n-ary cartesian product of a set of relations.
///
/// It is similar to [`itertools::structs::MultiProduct`](
/// https://docs.rs/itertools/*/itertools/structs/struct.MultiProduct.html)
/// but builds [`Tuple`]s instead of [`Vec`]s.
pub struct Product<T: Tuple, P: TuplePool<T>> {
    /// The input relations.
    relations: ProductVec<Box<[MaybeUninit<T>]>>,

    /// The current index in each input relation.
    indices: ProductVec<usize>,

    /// The [`FieldSet`]s for each "step" of the product. Each fieldset at
    /// index `n` is a pair with:
    /// 1. The union of all the fieldsets from `0` to `n - 1` (included).
    /// 2. The actual fieldset for the `n`th tuple.
    fieldsets: ProductVec<(T::FieldSet, T::FieldSet)>,

    /// The [`TuplePool`] to return consumed tuples to.
    pool: P,
}

impl<T: Tuple + CloneTuple> Product<T, ()> {
    /// Creates a new [`Product`] iterator that will return the cartesian
    /// product of all the tuples in all the given relations. Note that here,
    /// the product of two tuples `a` and `b` consists of creating a tuple `c`
    /// with all the fields of `a` and `b`.
    pub fn new(relations: impl IntoIterator<Item = (T::FieldSet, Vec<T>)>) -> Self {
        Self::with_pool(relations, ())
    }
}

impl<T: Tuple, P: TuplePool<T>> Product<T, P> {
    /// Same as [`Self::new`], but also allows an explicit [`TuplePool`] to be
    /// given.
    pub fn with_pool(relations: impl IntoIterator<Item = (T::FieldSet, Vec<T>)>, pool: P) -> Self {
        // Split input pairs into relations and their corresponding FieldSets.
        let pairs = relations.into_iter();
        let mut relations = ProductVec::with_capacity(pairs.size_hint().0);
        let mut fieldsets = ProductVec::with_capacity(relations.capacity());

        for (fieldset, vec) in pairs {
            fieldsets.push(fieldset);
            relations.push(Self::vec_to_relation(vec));
        }

        // Special cases: empty and identity iterators.
        match relations.len() {
            0 => return Self::empty(pool),
            1 => return Self::identity(fieldsets.pop().unwrap(), relations.pop().unwrap(), pool),
            _ => (),
        }

        // Prepare fieldsets: for each fieldset in fieldsets, we compute a pair
        // `(a, b)` where `a` is the union of all the fields up to that
        // fieldset (excluded), and `b` is fieldset itself. As such, for any
        // pair `(a, b)` at index `i`, `a.union(b)` is the fieldset that
        // represents a tuple merged from all the relations up to `i`
        // (included).
        let fieldsets = {
            let mut vec =
                ProductVec::<(T::FieldSet, T::FieldSet)>::with_capacity(fieldsets.len() - 1);

            let n_minus_1_fieldset = fieldsets.pop().unwrap();
            let n_minus_2_fieldset = fieldsets.pop().unwrap();

            vec.push((n_minus_1_fieldset, n_minus_2_fieldset));

            for fieldset in fieldsets.into_iter().rev() {
                let (i_minus_1_union_fieldset, i_minus_1_fieldset) = vec.last().unwrap();
                let union_fieldset = i_minus_1_fieldset.union(i_minus_1_union_fieldset);

                vec.push((union_fieldset, fieldset));
            }

            vec.reverse();
            vec
        };

        Product {
            indices: smallvec![0; relations.len()],
            relations,
            fieldsets,
            pool,
        }
    }

    fn empty(pool: P) -> Self {
        Product {
            relations: smallvec![vec![].into_boxed_slice()],
            indices: smallvec![0],
            fieldsets: smallvec![(T::FieldSet::default(), T::FieldSet::default())],
            pool,
        }
    }

    fn identity(fieldset: T::FieldSet, vec: Box<[MaybeUninit<T>]>, pool: P) -> Self {
        Product {
            relations: smallvec![vec],
            indices: smallvec![0],
            fieldsets: smallvec![(fieldset, T::FieldSet::default())],
            pool,
        }
    }

    fn vec_to_relation(vec: Vec<T>) -> Box<[MaybeUninit<T>]> {
        // In both stable 1.52.1 and nightly 1.54.0, Rust is unable to optimize
        //   v.into_iter().map(std::mem::MaybeUninit::new).collect::<Vec<_>>()
        // So for now we rely on good-ol `transmute`.

        // SAFETY: `MaybeUninit<T>` has the same size and layout as `T`.
        // Furthermore, we do not store `MaybeUninit<T>` into a struct -- we
        // lay it down in memory into a slice.
        unsafe { std::mem::transmute(vec.into_boxed_slice()) }
    }

    fn return_tuple(&mut self, tuple: T, ri: usize) {
        let fields = match self.fieldsets.get(ri) {
            Some((fieldset, _)) => fieldset,
            None => &self.fieldsets.last().unwrap().0,
        };

        self.pool.return_tuple(tuple, fields);
    }

    /// Returns the current tuple in the relation at index `ri`, and advances
    /// to the next tuple.
    #[inline]
    fn curr_tuple_in_and_advance(&mut self, ri: usize) -> Option<T> {
        let index = self.indices[ri];

        let tuple = self.relations[ri].get_mut(index)?;
        let tuple = std::mem::replace(tuple, MaybeUninit::uninit());

        // SAFETY: tuples in relation `ri` with index `>= indices[ri]` have not
        // been processed yet, and are therefore still initialized.
        let tuple = unsafe { tuple.assume_init() };

        self.indices[ri] = index + 1;

        Some(tuple)
    }

    /// Returns the current tuple in the relation at index `ri`.
    #[inline]
    fn curr_tuple_in(&self, ri: usize) -> &T {
        // SAFETY: tuples in relation `ri` with index `>= indices[ri]` have not
        // been processed yet, and are therefore still initialized.
        unsafe { &*self.relations[ri][self.indices[ri]].as_ptr() }
    }
}

impl<T: Tuple, P: TuplePool<T>> Iterator for Product<T, P> {
    type Item = T;

    /// Returns the next result of the product. For a guided explanation of how
    /// this function works, see [`tests::test_guided`].
    fn next(&mut self) -> Option<Self::Item> {
        debug_assert_eq!(self.relations.len(), self.indices.len());

        if self.relations.len() == 1 {
            // Special case: this is not a multi-product, but a regular
            // iterator. We could handle this case by returning the iterator
            // instead, but then whatever piece of codes that call `new()`
            // would have to deal with multiple potential iterator types.
            return self.curr_tuple_in_and_advance(0);
        }

        let product_len = self.relations.len();
        let rightmost_index = product_len - 1;

        // If at the end of the rightmost vector, start over from next element
        // of the previous vector.
        if self.indices[rightmost_index] == self.relations[rightmost_index].len() {
            let mut i = rightmost_index - 1;

            debug_assert!(!self.relations[i].is_empty());

            while self.indices[i] == self.relations[i].len() - 1 {
                if i == 0 {
                    // All items have been consumed in all relations.
                    return None;
                }

                i -= 1;
                debug_assert!(!self.relations[i].is_empty());
            }

            self.indices[rightmost_index] = 0;

            // Are we going through the last iteration of the `i`th relation?
            let is_last_i_iteration =
                (0..i).all(|i| self.indices[i] == self.relations[i].len() - 1);

            if is_last_i_iteration {
                // If so, we can consume the tuple and return it to the pool.
                let tuple = self.curr_tuple_in_and_advance(i).unwrap();

                self.return_tuple(tuple, i);
            }

            // Reset all indices of the relations that are not on their last
            // item yet.
            for i in i + 1..self.indices.len() {
                self.indices[i] = 0;
            }

            debug_assert_ne!(
                self.indices[rightmost_index],
                self.relations[rightmost_index].len()
            );
        }

        // Add tuple.
        let is_very_last_tuple =
            (0..rightmost_index).all(|i| self.indices[i] == self.relations[i].len() - 1);
        let mut tuple = if is_very_last_tuple {
            // This is the very last tuple, since all iterators are on
            // their last item. We can consume it.
            self.curr_tuple_in_and_advance(rightmost_index).unwrap()
        } else {
            // Extend the lifetime of the returned tuple below.
            // SAFETY: `pool` is not related to the reference to the tuple,
            // so the immutable reference to the tuple can safely be used
            // while we mutably borrow `pool`.
            let tuple_ref = unsafe { &*(self.curr_tuple_in(rightmost_index) as *const _) };
            let cloned_tuple = self
                .pool
                .clone_tuple(tuple_ref, &self.fieldsets[rightmost_index - 1].0);

            self.indices[rightmost_index] += 1;

            cloned_tuple
        };

        // We obtained the tuple in the rightmost relation. Now, we merge it
        // with all current tuples in the relations on its left.
        for i in (0..rightmost_index).rev() {
            let other = self.curr_tuple_in(i);
            let (tuple_fields, other_fields) = &self.fieldsets[i];

            tuple.merge_no_overlap(other, tuple_fields, other_fields);
        }

        Some(tuple)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut len = 1;
        let mut remove_len = 0;

        for (relation, &index) in self.relations.iter().zip(self.indices.iter()).rev() {
            remove_len += len * index;
            len *= relation.len();
        }

        (len - remove_len, Some(len - remove_len))
    }
}

impl<T: Tuple, P: TuplePool<T>> HasFieldSet for Product<T, P> {
    type FieldSet = T::FieldSet;

    fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
        None
    }

    fn fieldset(&self) -> Self::FieldSet {
        self.fieldsets[0].0.union(&self.fieldsets[0].1)
    }
}

impl<T: Tuple, P: TuplePool<T>> ExactSizeIterator for Product<T, P> {}

impl<T: Tuple, P: Clone + TuplePool<T>> Clone for Product<T, P> {
    fn clone(&self) -> Self {
        let mut pool = self.pool.clone();
        let mut relations = ProductVec::with_capacity(self.relations.len());
        let mut indices = ProductVec::with_capacity(self.indices.len());
        let fieldsets = self.fieldsets.clone();
        let mut is_iterating = true;

        for (i, (relation, index)) in self
            .relations
            .iter()
            .zip(self.indices.iter().copied())
            .enumerate()
        {
            let fields = match fieldsets.get(i) {
                Some((fieldset, _)) => fieldset,
                None => &fieldsets[fieldsets.len() - 1].0,
            };

            if is_iterating {
                let mut relation_builder = Vec::with_capacity(relation.len() - index);

                for tuple in &relation[index..] {
                    // SAFETY: tuples at index `indices[i]` and up still exist
                    // by design.
                    let tuple = unsafe { &*tuple.as_ptr() };

                    relation_builder.push(pool.clone_tuple(tuple, fields));
                }

                is_iterating = relation.len() > index;
                relations.push(Self::vec_to_relation(relation_builder));
                indices.push(0);
            } else {
                let mut relation_builder = Vec::with_capacity(relation.len());

                for tuple in relation.iter() {
                    // SAFETY: tuples in this relation have not been consumed
                    // at all yet.
                    let tuple = unsafe { &*tuple.as_ptr() };

                    relation_builder.push(pool.clone_tuple(tuple, fields));
                }

                relations.push(Self::vec_to_relation(relation_builder));
                indices.push(index);
            }
        }

        Self {
            fieldsets,
            relations,
            indices,
            pool,
        }
    }
}

impl<T: Tuple, P: TuplePool<T>> Drop for Product<T, P> {
    fn drop(&mut self) {
        let mut is_iterating = true;
        let self_ptr = self as *mut Self;

        for (ri, (tuples, mut start)) in self
            .relations
            .iter_mut()
            .zip(self.indices.iter().copied())
            .enumerate()
        {
            if !is_iterating {
                start = 0;
            } else if start + 1 < tuples.len() {
                is_iterating = false;
            }

            for i in start..tuples.len() {
                // SAFETY: tuples in relation `i` have not been consumed (i.e.
                // are still initialized) if their index in that relation is
                // `>= indices[i]` or if the relations on the left have not
                // been consumed until their last element.
                let tuple = unsafe {
                    std::mem::replace(&mut tuples[i], MaybeUninit::uninit()).assume_init()
                };

                // SAFETY: relations and the tuple pool are unrelated.
                unsafe { &mut *self_ptr }.return_tuple(tuple, ri);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::TryInto, mem::MaybeUninit};

    use crate::{
        bitset,
        clause::relation::Relation,
        join::product::Product,
        tuple::{CloneTuple, HasFieldSet},
    };

    fn make_cartesian_product<T: Clone + std::fmt::Debug, const N: usize>(
        tuples: &[[Option<T>; N]],
    ) -> super::Product<Vec<Option<T>>, ()> {
        let tuples = tuples
            .into_iter()
            .map(|x| x.iter().cloned().collect::<Vec<_>>());
        let relations = Relation::split_eq(tuples);

        super::Product::with_pool(
            relations.into_iter().map(|x| {
                let (bitset, iter) = x.into_tuple();

                (bitset, iter.collect())
            }),
            (),
        )
    }

    fn cartesian_product<T: Clone + std::fmt::Debug, const N: usize>(
        tuples: &[[Option<T>; N]],
    ) -> Vec<[Option<T>; N]> {
        let product = make_cartesian_product(tuples);

        product.map(move |tup| tup.try_into().unwrap()).collect()
    }

    #[test]
    fn test_empty() {
        assert!(cartesian_product::<i32, 0>(&[]).is_empty());
    }

    #[test]
    fn test_identity() {
        assert_eq!(
            cartesian_product(&[[Some(1), Some(2)]]),
            vec![[Some(1), Some(2)]],
        );
    }

    #[test]
    fn test_2x2() {
        assert_eq!(
            cartesian_product(&[
                [Some(1), None],
                [Some(2), None],
                [None, Some(3)],
                [None, Some(4)],
            ]),
            vec![
                [Some(1), Some(3)],
                [Some(1), Some(4)],
                [Some(2), Some(3)],
                [Some(2), Some(4)],
            ],
        );
    }

    #[test]
    fn test_2x2x0_incomplete() {
        assert_eq!(
            cartesian_product(&[
                [Some(1), None, None],
                [Some(2), None, None],
                [None, Some(3), None],
                [None, Some(4), None],
            ]),
            vec![
                [Some(1), Some(3), None],
                [Some(1), Some(4), None],
                [Some(2), Some(3), None],
                [Some(2), Some(4), None],
            ],
        );
    }

    #[test]
    fn test_2x2x0_incomplete_fieldset() {
        let cartesian_product = make_cartesian_product(&[
            [Some(1), None, None],
            [Some(2), None, None],
            [None, Some(3), None],
            [None, Some(4), None],
        ]);

        assert_eq!(cartesian_product.fieldset(), bitset![1 1 0]);
    }

    #[test]
    fn test_3x1x2() {
        assert_eq!(
            cartesian_product(&[
                [Some(1), None, None],
                [Some(2), None, None],
                [Some(3), None, None],
                [None, Some(30), None],
                [None, None, Some(500)],
                [None, None, Some(100)],
            ]),
            vec![
                [Some(1), Some(30), Some(500)],
                [Some(1), Some(30), Some(100)],
                [Some(2), Some(30), Some(500)],
                [Some(2), Some(30), Some(100)],
                [Some(3), Some(30), Some(500)],
                [Some(3), Some(30), Some(100)],
            ],
        );
    }

    #[test]
    fn test_3x1x2_len() {
        let mut cartesian_product = make_cartesian_product(&[
            [Some(1), None, None],
            [Some(2), None, None],
            [Some(3), None, None],
            [None, Some(30), None],
            [None, None, Some(500)],
            [None, None, Some(100)],
        ]);

        assert_eq!(cartesian_product.len(), 6);
        assert_eq!(
            cartesian_product.next(),
            Some(vec![Some(1), Some(30), Some(500)]),
        );

        assert_eq!(cartesian_product.len(), 5);
        assert_eq!(
            cartesian_product.next(),
            Some(vec![Some(1), Some(30), Some(100)]),
        );

        assert_eq!(cartesian_product.len(), 4);
        assert_eq!(
            cartesian_product.next(),
            Some(vec![Some(2), Some(30), Some(500)]),
        );

        assert_eq!(cartesian_product.len(), 3);
        assert_eq!(
            cartesian_product.next(),
            Some(vec![Some(2), Some(30), Some(100)]),
        );

        assert_eq!(cartesian_product.len(), 2);
        assert_eq!(
            cartesian_product.next(),
            Some(vec![Some(3), Some(30), Some(500)]),
        );

        assert_eq!(cartesian_product.len(), 1);
        assert_eq!(
            cartesian_product.next(),
            Some(vec![Some(3), Some(30), Some(100)]),
        );

        assert_eq!(cartesian_product.len(), 0);
        assert_eq!(cartesian_product.next(), None);
    }

    #[test]
    fn test_3x1x2_fieldset() {
        let cartesian_product = make_cartesian_product(&[
            [Some(1), None, None],
            [Some(2), None, None],
            [Some(3), None, None],
            [None, Some(30), None],
            [None, None, Some(500)],
            [None, None, Some(100)],
        ]);

        assert_eq!(cartesian_product.fieldset(), bitset![1 1 1]);
    }

    #[test]
    fn test_guided() {
        let mut product = make_cartesian_product(&[
            [Some("a1"), None, None],
            [Some("a2"), None, None],
            [None, Some("b1"), None],
            [None, None, Some("c1")],
            [None, None, Some("c2")],
        ]);

        // With start with all indices equal to 0; one index per relation.
        assert_eq!(product.indices.as_ref(), &[0, 0, 0]);

        // Values at the given index of a given relation are only dropped once
        // they are no longer useful. We consider that `relations[ri][i]` is no
        // longer useful if `indices[ri'] == relations[ri'].len() - 1` for all
        // `ri' < ri`, and if `i < indices[ri]`.
        // We thus define a function to easily compare the initialized values.
        fn initialized_at<T: CloneTuple>(product: &Product<T, ()>, ri: usize) -> &[T] {
            let is_consuming_relation =
                (0..ri).all(|i| product.indices[i] == product.relations[i].len() - 1);
            let uninit_values = if is_consuming_relation {
                &product.relations[ri][product.indices[ri]..]
            } else {
                &product.relations[ri]
            };

            // SAFETY: in a slice, MaybeUninit<T> preserves the same layout and
            // semantics as T. Furthermore, values starting at `indices[ri]`
            // are initialized.
            unsafe { std::mem::transmute::<&[MaybeUninit<T>], &[T]>(uninit_values) }
        }

        // To make it even easier to test, we define a macro here.
        macro_rules! assert_initialized_values_eq {
            ( $r0: expr, $r1: expr, $r2: expr, ) => {
                assert_eq!(initialized_at(&product, 0), $r0);
                assert_eq!(initialized_at(&product, 1), $r1);
                assert_eq!(initialized_at(&product, 2), $r2);
            };
        }

        assert_initialized_values_eq!(
            &[vec![Some("a1"), None, None], vec![Some("a2"), None, None]],
            &[vec![None, Some("b1"), None]],
            &[vec![None, None, Some("c1")], vec![None, None, Some("c2")]],
        );

        // Okay, finally. Let's start iterating.
        assert_eq!(
            product.next(),
            Some(vec![Some("a1"), Some("b1"), Some("c1")]),
        );

        // In this case, only the last index changes.
        assert_eq!(product.indices.as_ref(), &[0, 0, 1]);

        // We still need all theses tuples for further iterations, so they're
        // still here.
        assert_initialized_values_eq!(
            &[vec![Some("a1"), None, None], vec![Some("a2"), None, None]],
            &[vec![None, Some("b1"), None]],
            &[vec![None, None, Some("c1")], vec![None, None, Some("c2")]],
        );

        // We move on to the next value.
        assert_eq!(
            product.next(),
            Some(vec![Some("a1"), Some("b1"), Some("c2")]),
            // Notice the change to the last item.  ^^
        );

        // We reached the end of the rightmost relation, so we will iterate on
        // the left.
        // The second vector only has one item, so we will iterate over the
        // first vector.
        assert_eq!(product.indices.as_ref(), &[0, 0, 2]);

        // And we no longer need its first value, so it's gone.
        assert_initialized_values_eq!(
            &[vec![Some("a1"), None, None], vec![Some("a2"), None, None]],
            &[vec![None, Some("b1"), None]],
            &[vec![None, None, Some("c1")], vec![None, None, Some("c2")]],
        );

        // We continue...
        assert_eq!(
            product.next(),
            //              vv Now we see the next value of the first vector.
            Some(vec![Some("a2"), Some("b1"), Some("c1")]),
            // And the first value of the last one. ^^
        );

        // As stated above, we moved changed the index of the very first vector.
        assert_eq!(product.indices.as_ref(), &[1, 0, 1]);

        // We no longer need the first value of the last iterator.
        assert_initialized_values_eq!(
            &[vec![Some("a2"), None, None]],
            &[vec![None, Some("b1"), None]],
            &[vec![None, None, Some("c2")]],
        );

        // Again...
        assert_eq!(
            product.next(),
            Some(vec![Some("a2"), Some("b1"), Some("c2")]),
        );

        // Those were the last values of each iterator.
        assert_eq!(product.next(), None);
    }
}
