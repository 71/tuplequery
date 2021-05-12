//! The [`Relation`] struct and utilities to build it.

#[cfg(not(feature = "use_hashbrown"))]
use std::collections::hash_map::{HashMap, RawEntryMut};
use std::hash::Hash;
use std::iter::{FusedIterator, Peekable};
use std::vec::IntoIter;

#[cfg(feature = "use_hashbrown")]
use hashbrown::hash_map::{HashMap, RawEntryMut};
use smallvec::SmallVec;

use super::Clause;
use crate::tuple::{HasFieldSet, Tuple};

/// A relation of tuples where the fields set in each tuple is determined by a
/// single [`FieldSet`](crate::tuple::FieldSet).
#[derive(Clone, Debug)]
pub struct Relation<T: Tuple, I: IntoIterator<Item = T>> {
    fieldset: T::FieldSet,
    iter: I,
}

impl<T: Tuple, I: IntoIterator<Item = T>> Relation<T, I> {
    /// Creates a new [`Relation`], given its tuples and the fields set in each
    /// tuple.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that every single tuple in `tuples`
    /// has fields corresponding to the given `fieldset`.
    pub unsafe fn new(fieldset: T::FieldSet, tuples: I) -> Self {
        Relation {
            fieldset,
            iter: tuples,
        }
    }

    /// Splits the relation into a tuple of its input
    /// [`FieldSet`](crate::tuple::FieldSet) and tuples.
    pub fn into_tuple(self) -> (T::FieldSet, I) {
        (self.fieldset, self.iter)
    }

    /// Returns the [`FieldSet`](crate::tuple::FieldSet) of the relation.
    pub fn into_fieldset(self) -> T::FieldSet {
        self.fieldset
    }

    /// Returns the underlying iterator of the relation.
    pub fn into_iter(self) -> I {
        self.iter
    }

    /// Returns a reference to the [`FieldSet`](crate::tuple::FieldSet)
    /// describing what fields are set in each tuple of the relation.
    pub fn fields(&self) -> &T::FieldSet {
        &self.fieldset
    }

    /// Returns a reference to the underlying iterator.
    pub fn iter(&self) -> &I {
        &self.iter
    }

    /// Returns a mutable reference to the underlying iterator.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the iterator is never mutated
    /// in such a way that tuples resulting from it would no longer correspond
    /// to the [`FieldSet`](crate::tuple::FieldSet) of the relation.
    pub unsafe fn iter_mut(&mut self) -> &mut I {
        &mut self.iter
    }
}

impl<T: Tuple, I: IntoIterator<Item = T>> Clause<T> for Relation<T, I> {
    fn output_variables(&self) -> T::FieldSet {
        self.fieldset.clone()
    }

    #[cfg(feature = "gat")]
    type Iter<I: Iterator<Item = T>> = I::IntoIter;

    #[cfg(feature = "gat")]
    fn transform(self, _input: I) -> Self::Iter {
        self.1.into_iter()
    }

    #[cfg(not(feature = "gat"))]
    fn transform_empty<'a>(self) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: 'a,
    {
        Box::new(self.iter.into_iter())
    }
}

impl<T: Tuple, I: IntoIterator<Item = T>> HasFieldSet for Relation<T, I> {
    type FieldSet = T::FieldSet;

    fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
        Some(&self.fieldset)
    }

    fn fieldset(&self) -> Self::FieldSet {
        self.fields().clone()
    }
}

impl<T: Tuple + HasFieldSet<FieldSet = <T as Tuple>::FieldSet>, I: Iterator<Item = T>>
    Relation<T, Peekable<I>>
{
    /// Given an collection of tuples with embedded
    /// [`FieldSet`](crate::tuple::FieldSet)s, returns a [`Relation`] wrapping
    /// all these tuples.
    ///
    /// If `tuples` is empty, [`None`] is returned.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that every single tuple in `tuples`
    /// has the same [`FieldSet`](crate::tuple::FieldSet).
    pub unsafe fn from_tuples(tuples: impl IntoIterator<Item = T, IntoIter = I>) -> Option<Self> {
        let mut tuples = tuples.into_iter().peekable();
        let fieldset = tuples.peek()?.fieldset().to_owned();

        Some(Relation::new(fieldset, tuples))
    }
}

/// A struct that can be split into its [`FieldSet`](crate::tuple::FieldSet)
/// and [`Tuple`].
pub trait SplitTupleAndFieldSet {
    /// The type of the tuple.
    type Tuple: Tuple;

    /// Returns a reference to the [`FieldSet`] part of the pair.
    ///
    /// If a reference cannot be returned (i.e. the [`FieldSet`] must be
    /// constructed dynamically), `None` should be returned.
    ///
    /// [`FieldSet`]: crate::tuple::FieldSet
    fn fieldset(&self) -> Option<&<Self::Tuple as Tuple>::FieldSet>;

    /// Returns the [`Tuple`] part of the pair. This function will be called
    /// if [`Self::fieldset`] returned `Some`.
    fn tuple(self) -> Self::Tuple;

    /// Returns the owned pair as a tuple. This function will be called
    /// if [`Self::fieldset`] returned `None`.
    fn split(self) -> (<Self::Tuple as Tuple>::FieldSet, Self::Tuple);
}

/// A `(T::FieldSet, T)` pair (with `T` a [`Tuple`]) can indeed be split into
/// a tuple and a fieldset.
impl<T: Tuple> SplitTupleAndFieldSet for (T::FieldSet, T) {
    type Tuple = T;

    fn fieldset(&self) -> Option<&<Self::Tuple as Tuple>::FieldSet> {
        Some(&self.0)
    }

    fn tuple(self) -> Self::Tuple {
        self.1
    }

    fn split(self) -> (<Self::Tuple as Tuple>::FieldSet, Self::Tuple) {
        self
    }
}

/// Tuples with embedded `FieldSet`s can also be split into a tuple (by
/// identity) and a `FieldSet`.
impl<T: Tuple + HasFieldSet<FieldSet = <T as Tuple>::FieldSet>> SplitTupleAndFieldSet for T {
    type Tuple = T;

    fn fieldset(&self) -> Option<&<Self::Tuple as Tuple>::FieldSet> {
        self.fieldset_ref()
    }

    fn tuple(self) -> Self::Tuple {
        self
    }

    fn split(self) -> (<Self::Tuple as Tuple>::FieldSet, Self::Tuple) {
        (self.fieldset(), self)
    }
}

impl<T: Tuple> Relation<T, IntoIter<T>> {
    /// Splits an iterator of [`Tuple`]s into a set of [`Relation`]s based on
    /// the [`FieldSet`](crate::tuple::FieldSet) of each tuple.
    ///
    /// Each [`Tuple`] must implement [`SplitTupleAndFieldSet`], either by
    /// passing `(T::FieldSet, T)` pairs or by using [`Tuple`]s that have
    /// embedded [`FieldSet`](crate::tuple::FieldSet)s (with [`HasFieldSet`]).
    ///
    /// If `T::FieldSet` is [`Ord`], prefer [`Self::split_ord`].
    /// If `T::FieldSet` is [`Hash`], prefer [`Self::split_hash`].
    pub fn split_eq(
        tuples: impl IntoIterator<Item = impl SplitTupleAndFieldSet<Tuple = T>>,
    ) -> SmallVec<[Self; 4]>
    where
        T::FieldSet: Eq,
    {
        let mut relations = SmallVec::<[(T::FieldSet, Vec<T>); 4]>::new();

        for tuple in tuples {
            if let Some(fields) = tuple.fieldset() {
                match relations
                    .iter()
                    .position(|(fieldset, _)| fields == fieldset)
                {
                    Some(i) => {
                        relations[i].1.push(tuple.tuple());
                    }
                    None => {
                        let (fields, tuple) = tuple.split();

                        relations.push((fields, vec![tuple]));
                    }
                }
            } else {
                let (fields, tuple) = tuple.split();

                match relations
                    .iter()
                    .position(|(fieldset, _)| &fields == fieldset)
                {
                    Some(i) => {
                        relations[i].1.push(tuple);
                    }
                    None => {
                        relations.push((fields, vec![tuple]));
                    }
                }
            }
        }

        // SAFETY: tuples are grouped by fieldset; they therefore all have the
        // same fieldset and a `Relation` can safely be created from them.
        relations
            .into_iter()
            .map(|(fieldset, tuples)| unsafe { Relation::new(fieldset, tuples.into_iter()) })
            .collect()
    }

    /// Specialization of [`Self::split_eq`] for [`Ord`]
    /// [`FieldSet`](crate::tuple::FieldSet)s.
    pub fn split_ord(
        tuples: impl IntoIterator<Item = impl SplitTupleAndFieldSet<Tuple = T>>,
    ) -> SmallVec<[Self; 4]>
    where
        T::FieldSet: Ord,
    {
        let mut relations = SmallVec::<[(T::FieldSet, Vec<T>); 4]>::new();

        for tuple in tuples {
            if let Some(fields) = tuple.fieldset() {
                match relations.binary_search_by(|(fieldset, _)| fieldset.cmp(fields)) {
                    Ok(i) => relations[i].1.push(tuple.tuple()),
                    Err(i) => {
                        let (fields, tuple) = tuple.split();

                        relations.insert(i, (fields, vec![tuple]));
                    }
                }
            } else {
                let (fields, tuple) = tuple.split();

                match relations.binary_search_by(|(fieldset, _)| fieldset.cmp(&fields)) {
                    Ok(i) => relations[i].1.push(tuple),
                    Err(i) => relations.insert(i, (fields, vec![tuple])),
                }
            }
        }

        // SAFETY: tuples are grouped by fieldset; they therefore all have the
        // same fieldset and a `Relation` can safely be created from them.
        relations
            .into_iter()
            .map(|(fieldset, tuples)| unsafe { Relation::new(fieldset, tuples.into_iter()) })
            .collect()
    }

    /// Specialization of [`Self::split_eq`] for [`Hash`]
    /// [`FieldSet`](crate::tuple::FieldSet)s.
    pub fn split_hash(
        tuples: impl IntoIterator<Item = impl SplitTupleAndFieldSet<Tuple = T>>,
    ) -> SmallVec<[Self; 4]>
    where
        T::FieldSet: Eq + Hash,
    {
        let mut hashmap = HashMap::<T::FieldSet, Vec<T>>::new();

        for tuple in tuples {
            if let Some(fields) = tuple.fieldset() {
                match hashmap.raw_entry_mut().from_key(fields) {
                    RawEntryMut::Occupied(mut entry) => {
                        entry.get_mut().push(tuple.tuple());
                    }
                    RawEntryMut::Vacant(entry) => {
                        let (fields, tuple) = tuple.split();

                        entry.insert(fields, vec![tuple]);
                    }
                }
            } else {
                let (fields, tuple) = tuple.split();

                match hashmap.raw_entry_mut().from_key(&fields) {
                    RawEntryMut::Occupied(mut entry) => {
                        entry.get_mut().push(tuple);
                    }
                    RawEntryMut::Vacant(entry) => {
                        entry.insert(fields.to_owned(), vec![tuple]);
                    }
                }
            }
        }

        // SAFETY: tuples are grouped by fieldset; they therefore all have the
        // same fieldset and a `Relation` can safely be created from them.
        hashmap
            .into_iter()
            .map(|(fieldset, tuples)| unsafe { Relation::new(fieldset, tuples.into_iter()) })
            .collect()
    }
}

/// Proxy [`Iterator`] (and variants) implementations to `Relation.iter`.
impl<T: Tuple, I: Iterator<Item = T>> Iterator for Relation<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T: Tuple, I: DoubleEndedIterator<Item = T>> DoubleEndedIterator for Relation<T, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n)
    }
}

impl<T: Tuple, I: ExactSizeIterator<Item = T>> ExactSizeIterator for Relation<T, I> {}
impl<T: Tuple, I: FusedIterator<Item = T>> FusedIterator for Relation<T, I> {}
