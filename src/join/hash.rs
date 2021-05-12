//! The [`Hash`] struct, which provides a hash-join based implementation of
//! [`Join`].

#[cfg(not(feature = "use_hashbrown"))]
use std::collections::hash_map::{HashMap, RandomState as DefaultHashBuilder, RawEntryMut};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::{
    hash::{BuildHasher, Hasher},
    ptr::NonNull,
};

#[cfg(feature = "use_hashbrown")]
use hashbrown::hash_map::{DefaultHashBuilder, HashMap, RawEntryMut};
use smallvec::{smallvec, SmallVec};

use crate::tuple::{CloneTuple, EqTuple, FieldSet, HasFieldSet, HashTuple, Tuple, TuplePool};

use super::product::Product;
use super::Join;

/// A [`Join`] implementation that uses a [`HashJoin`].
#[derive(Clone, Debug, Default)]
pub struct Hash {}

impl Hash {
    /// Creates a new [`Hash`] struct.
    pub const fn new() -> Hash {
        Self {}
    }
}

impl<
        T: EqTuple + HashTuple,
        P: Clone + TuplePool<T>,
        I1: Iterator<Item = T>,
        I2: Iterator<Item = T>,
    > Join<T, P, I1, I2> for Hash
{
    type Iter = HashJoin<T, P, I2>;
    type ProductIter = HashJoin<T, P, Product<T, P>>;

    fn join(
        &self,
        iter1: I1,
        iter2: I2,
        fields1: T::FieldSet,
        fields2: T::FieldSet,
        pool: P,
    ) -> Self::Iter {
        HashJoin::with_pool(iter1, iter2, fields1, fields2, pool)
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

        HashJoin::with_pool(iter1, iter2, fields1, fields2, pool)
    }
}

/// An iterator over the hash-join of two iterators.
#[derive(Debug)]
pub struct HashJoin<T: Tuple, P: TuplePool<T>, I: Iterator<Item = T>> {
    iter: I,
    hashmap: HashMap<HashJoinKey<T>, ()>,

    hash_curr: smallvec::IntoIter<[T; 2]>,
    iter_curr: std::mem::MaybeUninit<T>,

    shared_fields: Pin<Box<T::FieldSet>>,

    hash_fields: T::FieldSet,
    iter_fields: T::FieldSet,

    pool: P,
}

impl<T: EqTuple + HashTuple + CloneTuple, I: Iterator<Item = T>> HashJoin<T, (), I> {
    /// Creates a new [`HashJoin`] iterator. `iter1` will be read into the hash
    /// table used to perform the join, so it should ideally be smaller than
    /// `iter2`.
    pub fn new(
        hash_iter: impl IntoIterator<Item = T>,
        lazy_iter: impl IntoIterator<IntoIter = I>,
        hash_fields: T::FieldSet,
        lazy_fields: T::FieldSet,
    ) -> Self {
        Self::with_pool(hash_iter, lazy_iter, hash_fields, lazy_fields, ())
    }
}

impl<T: EqTuple + HashTuple, P: TuplePool<T>, I: Iterator<Item = T>> HashJoin<T, P, I> {
    /// Same as [`Self::new`], but also allows an explicit [`TuplePool`] to be
    /// given.
    pub fn with_pool(
        hash_iter: impl IntoIterator<Item = T>,
        lazy_iter: impl IntoIterator<IntoIter = I>,
        hash_fields: T::FieldSet,
        lazy_fields: T::FieldSet,
        pool: P,
    ) -> Self {
        Self::new_if_unempty(hash_iter, lazy_iter, hash_fields, lazy_fields, pool)
            .unwrap_or_else(|(iter, pool, f1, f2)| Self::empty(f1, f2, iter, pool))
    }

    /// Returns an [`HashJoin`] if none of the given iterators is empty.
    /// Otherwise, returns an error with all the input parameters.
    fn new_if_unempty(
        hash_iter: impl IntoIterator<Item = T>,
        lazy_iter: impl IntoIterator<IntoIter = I>,
        hash_fields: T::FieldSet,
        lazy_fields: T::FieldSet,
        mut pool: P,
    ) -> Result<Self, (I, P, T::FieldSet, T::FieldSet)> {
        // Get the first tuple of the hash iterator.
        let mut hash_iter = hash_iter.into_iter();
        let hash_val: T = match hash_iter.next() {
            Some(v) => v,
            None => return Err((lazy_iter.into_iter(), pool, hash_fields, lazy_fields)),
        };

        // Get the first tuple of the lazy iterator.
        let mut lazy_iter = lazy_iter.into_iter();
        let lazy_val: T = match lazy_iter.next() {
            Some(v) => v,
            None => return Err((lazy_iter, pool, hash_fields, lazy_fields)),
        };

        // Compute fields common to hash and lazy iterators.
        //
        // Hash and equality comparisons for the join will only consider these
        // fields.
        let shared_fields = hash_fields.intersect(&lazy_fields);

        // Pin those fields to guarantee that a pointer to them will stay valid
        // as long as the returned `HashJoin` is alive.
        let shared_fields = Box::pin(shared_fields);
        let shared_fields_ptr = NonNull::from(shared_fields.as_ref().get_ref());

        // Add all values from `hash_iter` into `hashmap`, grouping those with
        // equal `shared_fields` in a single `HashJoinKey`.
        let mut hashmap = HashMap::new();

        // (First value we obtained earlier.)
        Self::get_raw_entry(&mut hashmap, &shared_fields, &hash_val).or_insert(
            HashJoinKey {
                significant_fields: shared_fields_ptr,
                vec: smallvec![hash_val],
            },
            (),
        );

        // (All other values of the `hash_iter`.)
        for hash_val in hash_iter {
            match Self::get_raw_entry(&mut hashmap, &shared_fields, &hash_val) {
                RawEntryMut::Occupied(mut entry) => entry.key_mut().vec.push(hash_val),
                RawEntryMut::Vacant(entry) => {
                    entry.insert(
                        HashJoinKey {
                            significant_fields: shared_fields_ptr,
                            vec: smallvec![hash_val],
                        },
                        (),
                    );
                }
            }
        }

        // Kick-off the first iteration by finding the values in `hashmap` that
        // correspond to `lazy_val`.
        let vec = match Self::get_raw_entry(&mut hashmap, &shared_fields, &lazy_val) {
            RawEntryMut::Occupied(entry) => entry
                .key()
                .vec
                .iter()
                .map(|v| pool.clone_tuple(v, &hash_fields))
                .collect(),
            RawEntryMut::Vacant(_) => {
                // No match for the first value: we use an empty vector, and
                // `next()` will automatically move on to the next value.
                SmallVec::new()
            }
        };

        Ok(HashJoin {
            iter: lazy_iter,
            hashmap,
            hash_curr: vec.into_iter(),
            iter_curr: std::mem::MaybeUninit::new(lazy_val),
            hash_fields,
            iter_fields: lazy_fields,
            shared_fields,
            pool,
        })
    }

    /// Returns an empty [`HashJoin`] iterator that will yield no items.
    fn empty(hash_fields: T::FieldSet, lazy_fields: T::FieldSet, iter: I, pool: P) -> Self {
        // `hashmap` is empty, so there will never be a match with `iter` and
        // no result will ever be returned from `next()`.
        HashJoin {
            iter,
            hash_curr: SmallVec::new().into_iter(),
            iter_curr: std::mem::MaybeUninit::uninit(),
            hash_fields,
            iter_fields: lazy_fields,
            shared_fields: Box::pin(T::FieldSet::default()),
            hashmap: HashMap::new(),
            pool,
        }
    }

    /// Returns the [`RawEntryMut`] corresponding computed with the specific
    /// fields of `tuple` specified in `fieldset`.
    fn get_raw_entry<'a>(
        hashmap: &'a mut HashMap<HashJoinKey<T>, ()>,
        fieldset: &T::FieldSet,
        tuple: &T,
    ) -> RawEntryMut<'a, HashJoinKey<T>, (), DefaultHashBuilder> {
        // Compute hash using only the relevant fields.
        let mut hasher = hashmap.hasher().build_hasher();
        tuple.hash(fieldset, &mut hasher);
        let hash = hasher.finish();

        // Get corresponding entry using that hash.
        hashmap.raw_entry_mut().from_hash(hash, |potential_key| {
            // There is a hash equality. We now compare the relevant fields
            // to make sure they are indeed equal.
            tuple.eq(&potential_key.vec[0], fieldset)
        })
    }
}

impl<T: EqTuple + HashTuple, P: TuplePool<T>, I: Iterator<Item = T>> Iterator
    for HashJoin<T, P, I>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // (Explicit tail recursion.)
        loop {
            if let Some(mut hash_value) = self.hash_curr.next() {
                // There is a pending value in `hash_curr`. Merge it with the
                // next value given by the last value returned by the iterator
                // and return it.

                // SAFETY: if `hash_curr` has elements, `iter_curr` was
                // initialized.
                let iter_value = unsafe { &*self.iter_curr.as_ptr() };
                hash_value.merge(iter_value, &self.hash_fields, &self.iter_fields);
                return Some(hash_value);
            }

            // There are no pending values in `hash_curr`. We can go on to the
            // next value in `iter`.
            let iter_value = self.iter.next()?;

            if let RawEntryMut::Occupied(entry) =
                Self::get_raw_entry(&mut self.hashmap, &self.shared_fields, &iter_value)
            {
                // The value returned by `iter` has one or more corresponding
                // tuples in the `hashmap`. We need to merge them all with
                // `iter_value`, so we store it locally and mark the `hashmap`
                // values as pending by setting the new value of `hash_curr`.
                let pool = &mut self.pool;
                let hash_fields = &self.hash_fields;

                self.hash_curr = entry
                    .key()
                    .vec
                    .iter()
                    .map(move |v| pool.clone_tuple(v, hash_fields))
                    .collect::<SmallVec<_>>()
                    .into_iter();

                // SAFETY: if `iter` has elements, `iter_curr` was initialized.
                let iter_curr = unsafe { &mut *self.iter_curr.as_mut_ptr() };

                // Return whatever value had previously been assigned to
                // `iter_curr` to the pool.
                let previous_iter_curr = std::mem::replace(iter_curr, iter_value);

                self.pool
                    .return_tuple(previous_iter_curr, &self.iter_fields);
            }

            // Note: if there is no corresponding entry, we will simply loop
            // back to `iter_value = self.iter.next()?` until the iterator
            // completes or a value with a corresponding entry is found.
        }
    }
}

impl<T: Tuple, P: TuplePool<T>, I: Iterator<Item = T>> HasFieldSet for HashJoin<T, P, I> {
    type FieldSet = T::FieldSet;

    fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
        None
    }

    fn fieldset(&self) -> Self::FieldSet {
        self.hash_fields.union(&self.iter_fields)
    }
}

impl<T: EqTuple + HashTuple, P: Clone + TuplePool<T>, I: Clone + Iterator<Item = T>> Clone
    for HashJoin<T, P, I>
{
    fn clone(&self) -> Self {
        if self.hashmap.is_empty() {
            return Self::empty(
                self.hash_fields.clone(),
                self.iter_fields.clone(),
                self.iter.clone(),
                self.pool.clone(),
            );
        }

        let mut pool = self.pool.clone();
        let iter = self.iter.clone();
        let hash_fields = self.hash_fields.clone();
        let iter_fields = self.iter_fields.clone();
        let shared_fields = self.shared_fields.clone();

        Self {
            hashmap: self
                .hashmap
                .iter()
                .map(|(k, _)| (k.clone_in(&shared_fields, &mut pool), ()))
                .collect(),
            hash_curr: self
                .hash_curr
                .as_slice()
                .into_iter()
                .map(|v| pool.clone_tuple(v, &hash_fields))
                .collect::<SmallVec<_>>()
                .into_iter(),
            iter_curr: MaybeUninit::new(
                // SAFETY: `iter_curr` is only uninitialized in empty
                // `HashJoin`s. We handled that case at the start of the
                // function.
                pool.clone_tuple(unsafe { &*self.iter_curr.as_ptr() }, &iter_fields),
            ),

            iter,
            hash_fields,
            iter_fields,
            shared_fields,
            pool,
        }
    }
}

impl<T: Tuple, P: TuplePool<T>, I: Iterator<Item = T>> Drop for HashJoin<T, P, I> {
    fn drop(&mut self) {
        if !self.hashmap.is_empty() {
            // SAFETY: if the hashmap is non-empty, `iter_curr` was
            // initialized.
            let iter_curr = unsafe {
                std::mem::replace(&mut self.iter_curr, MaybeUninit::uninit()).assume_init()
            };

            self.pool.return_tuple(iter_curr, &self.iter_fields);

            for (HashJoinKey { vec, .. }, ()) in self.hashmap.drain() {
                for tuple in vec {
                    self.pool.return_tuple(tuple, &self.hash_fields);
                }
            }
        }
    }
}

/// A wrapper around a vector of equal [`Tuple`]s whose equality is
/// determined by a given [`FieldSet`].
struct HashJoinKey<T: Tuple> {
    significant_fields: NonNull<T::FieldSet>,
    vec: SmallVec<[T; 2]>,
}

impl<T: Tuple> HashJoinKey<T> {
    /// Returns the [`FieldSet`] that describes what parts of each tuple should
    /// be considered.
    ///
    /// Each `T` tuple may have more fields than `significant_fields`. However,
    /// comparisons should only use `significant_fields` in order to compare
    /// sub-parts of different vectors to see if they should be joined.
    fn significant_fields(&self) -> &T::FieldSet {
        // SAFETY: `HashJoinKey` only ever lives in `HashJoin`, which
        // ensures that `significant_fields` is allocated in a box and never
        // moved or mutated during its lifetime.
        unsafe { self.significant_fields.as_ref() }
    }

    /// Clones the contents of the [`HashJoinKey`] into a new [`HashJoinKey`],
    /// using the given `pool` for clones. A new reference to `fields` is also
    /// required, since the current fields may not live as long as the clone.
    fn clone_in(&self, fields: &T::FieldSet, mut pool: impl TuplePool<T>) -> Self {
        Self {
            vec: self
                .vec
                .iter()
                .map(|v| pool.clone_tuple(v, fields))
                .collect(),
            significant_fields: NonNull::from(fields),
        }
    }
}

impl<T: std::fmt::Debug + Tuple> std::fmt::Debug for HashJoinKey<T>
where
    T::FieldSet: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("HashJoinKey")
            .field("significant_fields", self.significant_fields())
            .field("vec", &self.vec)
            .finish()
    }
}

// Equality and hash functions using only the significant fields.
impl<T: EqTuple> PartialEq for HashJoinKey<T> {
    fn eq(&self, other: &Self) -> bool {
        self.vec[0].eq(&other.vec[0], self.significant_fields())
    }
}

impl<T: EqTuple> Eq for HashJoinKey<T> {}

impl<T: HashTuple> std::hash::Hash for HashJoinKey<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vec[0].hash(self.significant_fields(), state);
    }
}
