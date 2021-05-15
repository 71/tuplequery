//! The [`Tuple`] trait, as well as some of its built-in implementations:
//! - The basic implementation of [`Tuple`] is [`Vec<Option<T>>`] (also
//!   available as [`SmallVec<[Option<T>; N]>`]). It is safe to use, and
//!   reasonably efficient. However, it stores `None`s for each missing value,
//!   which may be wasteful if tuples are expected to have a log of missing
//!   values. Furthermore, if `T` is not [non-zero], the size of the tuple be
//!   even greater.
//! - If the overhead introduced by not having [non-zero] `T`s is too large,
//!   [`DenseTuple`] is provided. It similarly allocates enough memory to hold
//!   all elements of a tuple ahead of time and wastes memory when many
//!   elements are missing, but it uses [`std::mem::MaybeUninit`] instead of
//!   [`Option`] internally, removing the overhead added by using [`Option`]s.
//! - If tuples are expected to have many missing values, [`SparseTuple`] is
//!   provided as well. It only allocates memory to store its values and does
//!   not waste any memory. However, merging operations are slower than with
//!   [`DenseTuple`]s and [`Vec`]s.
//!
//! [`HashMap`]s can also be used as [`Tuple`]s, though this is not recommended
//! as it is much less efficient than the methods listed above. Instead, use
//! the [`build`] module to convert entity-like objects to [`Tuple`]s.
//!
//! Additionally, this module defines other traits used throughout the crate,
//! the most important of which being [`FieldSet`].
//!
//! [non-zero]: https://rust-lang.github.io/rfcs/2307-concrete-nonzero-types.html

use std::cmp::Ordering;
#[cfg(not(feature = "use_hashbrown"))]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "use_hashbrown")]
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

pub mod bitset;
pub(crate) mod check_fields;
pub mod dense;
pub mod entity;
pub mod sparse;

pub use self::bitset::Bitset;
pub use self::dense::DenseTuple;
pub use self::sparse::SparseTuple;
use entity::{EntityKeys, FromEntity, MutableEntityKeys};

/// An interface for types that can be given to a [`crate::Query`].
///
/// Tuples represent _incomplete_ tuples whose values are filled by merging
/// with other tuples over time.
pub trait Tuple: Sized {
    /// The type that stores what fields are set on the tuple.
    type FieldSet: FieldSet;

    /// Merges the given tuple into the current tuple.
    fn merge(&mut self, other: &Self, self_fields: &Self::FieldSet, other_fields: &Self::FieldSet);

    /// Merges the given tuple into the current tuple.
    fn merge_owned(
        &mut self,
        other: Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        self.merge(&other, self_fields, other_fields)
    }

    /// Merges the given tuple into the current tuple with the guarantee that
    /// none of their values overlap.
    fn merge_no_overlap(
        &mut self,
        other: &Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        self.merge(other, self_fields, other_fields);
    }

    /// Merges the given tuple into the current tuple with the guarantee that
    /// none of their values overlap.
    fn merge_owned_no_overlap(
        &mut self,
        other: Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        self.merge_owned(other, self_fields, other_fields);
    }

    /// Clears the content of the tuple.
    ///
    /// It can be assumed that `fields` corresponds to the fields of the tuple.
    fn clear(&mut self, fields: &Self::FieldSet);
}

/// An interface for [`Tuple`]s that can be compared for equality according to
/// their [`FieldSet`].
pub trait EqTuple: Tuple {
    /// Compares for equality the two given tuples, only considering the given
    /// fields.
    fn eq(&self, other: &Self, fields: &Self::FieldSet) -> bool;
}

/// An interface for [`Tuple`]s that can be ordered totally according to their
/// [`FieldSet`].
pub trait OrdTuple: Tuple {
    /// Compares the two given tuples, only considering the given fields.
    fn cmp(&self, other: &Self, fields: &Self::FieldSet) -> Ordering;
}

/// An interface for [`Tuple`]s that can be hashed according to their
/// [`FieldSet`].
pub trait HashTuple: Tuple {
    /// Hashes the fields of the tuple described by the given set.
    fn hash<H: std::hash::Hasher>(&self, fields: &Self::FieldSet, hasher: &mut H);
}

/// An interface for [`Tuple`]s that can be cloned according to their
/// [`FieldSet`].
pub trait CloneTuple: Tuple {
    /// Clones the tuple.
    fn clone(&self, fields: &Self::FieldSet) -> Self;
}

/// An interface for types that can define whether or not a [`Tuple`] contains
/// a field.
pub trait FieldSet: Clone + Default + Eq {
    /// Returns whether the current fieldset is empty, i.e. it does not have
    /// any field.
    fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// Modifies the current [`FieldSet`] so that it also contains fields from
    /// the given set.
    fn union_in_place(&mut self, other: &Self);

    /// Modifies the current [`FieldSet`] so that it only contains fields that
    /// are in both itself and `other`.
    fn intersect_in_place(&mut self, other: &Self);

    /// Removes all fields in `other` from `self`.
    fn difference_in_place(&mut self, other: &Self);

    /// Returns whether the current [`FieldSet`] has at least one field that is
    /// also in `other`.
    fn intersects(&self, other: &Self) -> bool {
        !self.intersect(other).is_empty()
    }

    /// Returns a [`FieldSet`] that contains all the fields that are either in
    /// `self` or in `other`, or in both.
    fn union(&self, other: &Self) -> Self {
        let mut union = self.clone();
        union.union_in_place(other);
        union
    }

    /// Returns a [`FieldSet`] that contains the fields that are both in `self`
    /// and in `other`.
    fn intersect(&self, other: &Self) -> Self {
        let mut intersection = self.clone();
        intersection.intersect_in_place(other);
        intersection
    }

    /// Returns the all the fields in `self` that are not in `other`.
    fn difference(&self, other: &Self) -> Self {
        let mut difference = self.clone();
        difference.difference_in_place(other);
        difference
    }

    /// Returns whether the current fieldset is a subset of the given fieldset.
    fn is_subset(&self, superset: &Self) -> bool {
        &self.intersect(superset) == self
    }

    /// Returns whether the current fieldset is a subset of the given fieldset.
    fn is_superset(&self, subset: &Self) -> bool {
        subset.is_subset(self)
    }
}

/// An interface for structs that have a set of fields represented by a
/// [`FieldSet`].
pub trait HasFieldSet {
    /// The type of the fieldset.
    type FieldSet: FieldSet;

    /// Returns a reference to the owned [`FieldSet`], if it is possible to
    /// cheaply obtain such a reference.
    fn fieldset_ref(&self) -> Option<&Self::FieldSet>;

    /// Returns the actual [`FieldSet`].
    fn fieldset(&self) -> Self::FieldSet;
}

impl<F: FieldSet> HasFieldSet for F {
    type FieldSet = F;

    fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
        Some(self)
    }

    fn fieldset(&self) -> Self::FieldSet {
        self.clone()
    }
}

impl<F: FieldSet, T> HasFieldSet for (F, T) {
    type FieldSet = F;

    fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
        Some(&self.0)
    }

    fn fieldset(&self) -> Self::FieldSet {
        self.0.clone()
    }
}

/// An interface for creating and destroying [`Tuple`]s, possibly recycling
/// previously created values.
///
/// The simplest implementation of [`TuplePool`] used by default by functions
/// that accept a pool is the unit type `()`. It simply calls [`Clone::clone`]
/// in [`TuplePool::clone_tuple`], and drops tuples in
/// [`TuplePool::return_tuple`].
pub trait TuplePool<T: Tuple> {
    /// Clones an existing tuple.
    fn clone_tuple(&mut self, tuple: &T, fields: &T::FieldSet) -> T;

    /// Returns a tuple to the pool.
    fn return_tuple(&mut self, tuple: T, fields: &T::FieldSet);
}

impl<T: CloneTuple + Tuple> TuplePool<T> for () {
    fn clone_tuple(&mut self, tuple: &T, fields: &T::FieldSet) -> T {
        tuple.clone(fields)
    }

    fn return_tuple(&mut self, mut tuple: T, fields: &T::FieldSet) {
        tuple.clear(fields);

        drop(tuple)
    }
}

impl<'p, T: Tuple, P: TuplePool<T>> TuplePool<T> for &'p mut P {
    fn clone_tuple(&mut self, tuple: &T, fields: &T::FieldSet) -> T {
        (*self).clone_tuple(tuple, fields)
    }

    fn return_tuple(&mut self, tuple: T, fields: &T::FieldSet) {
        (*self).return_tuple(tuple, fields)
    }
}

/// An implementation of [`TuplePool`] which stores tuples in a [`Vec`] and
/// pushes or pops them when needed.
#[derive(Default)]
pub struct RecycleTuplePool<T: Tuple> {
    max_capacity: usize,
    pool: Vec<T>,
}

impl<T: Tuple> RecycleTuplePool<T> {
    /// Creates a new [`RecycleTuplePool`] that starts with a `0` capacity and
    /// can grow to have up to `max_capacity` items, after which it will no
    /// longer recycle tuples.
    ///
    /// This function does not allocate.
    pub fn with_max_capacity(max_capacity: usize) -> Self {
        Self {
            max_capacity,
            pool: Vec::new(),
        }
    }

    /// Creates a new [`RecycleTuplePool`] with no maximum capacity and an
    /// initial capacity of `initial_capacity`.
    pub fn with_initial_capacity(initial_capacity: usize) -> Self {
        Self {
            max_capacity: 0,
            pool: Vec::with_capacity(initial_capacity),
        }
    }

    /// Creates a new [`RecycleTuplePool`] with the given initial and maximum
    /// capacities. See [`Self::with_initial_capacity`] and
    /// [`Self::with_max_capacity`] for more information.
    pub fn with_capacities(initial_capacity: usize, max_capacity: usize) -> Self {
        assert!(max_capacity == 0 || max_capacity >= initial_capacity);

        Self {
            max_capacity,
            pool: Vec::with_capacity(initial_capacity),
        }
    }
}

impl<T: Clone + Tuple> TuplePool<T> for RecycleTuplePool<T> {
    fn clone_tuple(&mut self, tuple: &T, _fields: &T::FieldSet) -> T {
        match self.pool.pop() {
            Some(mut new_tuple) => {
                new_tuple.clone_from(tuple);
                new_tuple
            }
            None => tuple.clone(),
        }
    }

    fn return_tuple(&mut self, mut tuple: T, fields: &T::FieldSet) {
        tuple.clear(fields);

        if self.max_capacity == 0 || self.pool.len() < self.max_capacity {
            self.pool.push(tuple);
        }
    }
}

// Built-in implementations for `Vec`, `SmallVec`, `HashMap` and `HashSet`.
// ============================================================================

// `Vec`s (and `SmallVec`s) of `Option`s can represent `Tuple`s and
// `FieldSet`s.
macro_rules! impl_tuple_for {
    ( $Container: ident < $T: ident ( $CT: ty ) ( $($tt: tt)* )> ) => {
        impl<$T: Clone, $($tt)*> Tuple for $Container<$CT> {
            type FieldSet = Bitset;

            fn merge(&mut self, other: &Self, _self_fields: &Self::FieldSet, _other_fields: &Self::FieldSet) {
                for (i, v) in other.iter().enumerate().filter_map(|(i, x)| Some((i, x.as_ref()?))) {
                    if self.len() <= i {
                        self.reserve(i + 1);

                        while self.len() < i {
                            self.push(None);
                        }

                        self.push(Some(v.clone()));
                    } else if self[i].is_none() {
                        self[i] = Some(v.clone());
                    }
                }
            }

            fn merge_owned(&mut self, other: Self, _self_fields: &Self::FieldSet, _other_fields: &Self::FieldSet) {
                for (i, v) in other.into_iter().enumerate().filter_map(|(i, x)| Some((i, x?))) {
                    if self.len() <= i {
                        self.reserve(i + 1);

                        while self.len() < i {
                            self.push(None);
                        }

                        self.push(Some(v));
                    } else if self[i].is_none() {
                        self[i] = Some(v);
                    }
                }
            }

            fn clear(&mut self, _fields: &Self::FieldSet) {
                for v in self {
                    *v = None;
                }
            }
        }

        impl<$T: Clone, $($tt)*> HasFieldSet for $Container<$CT> {
            type FieldSet = <Self as Tuple>::FieldSet;

            fn fieldset_ref(&self) -> Option<&Self::FieldSet> {
                None
            }

            fn fieldset(&self) -> Self::FieldSet {
                self.iter().enumerate().filter(|(_, o)| o.is_some()).map(|(i, _)| i).collect()
            }
        }

        impl<$T: Clone, $($tt)*> CloneTuple for $Container<$CT> {
            fn clone(&self, fields: &Self::FieldSet) -> Self {
                debug_assert!(self.fieldset().is_superset(fields));

                self.iter().enumerate().map(|(i, v)| if fields.has(i) { v.clone() } else { None }).collect()
            }
        }

        impl<$T: Clone + Eq, $($tt)*> EqTuple for $Container<$CT> {
            fn eq(&self, other: &Self, fields: &Self::FieldSet) -> bool {
                debug_assert!(self.fieldset().is_superset(fields));
                debug_assert!(other.fieldset().is_superset(fields));

                fields.iter().all(|i| self[i] == other[i])
            }
        }

        impl<$T: Clone + Ord, $($tt)*> OrdTuple for $Container<$CT> {
            fn cmp(&self, other: &Self, fields: &Self::FieldSet) -> Ordering {
                debug_assert!(self.fieldset().is_superset(fields));
                debug_assert!(other.fieldset().is_superset(fields));

                for i in fields.iter() {
                    match self[i].cmp(&other[i]) {
                        Ordering::Equal => continue,
                        ordering => return ordering,
                    }
                }

                Ordering::Equal
            }
        }

        impl<$T: Clone + std::hash::Hash, $($tt)*> HashTuple for $Container<$CT> {
            fn hash<H: std::hash::Hasher>(&self, fields: &Self::FieldSet, hasher: &mut H) {
                debug_assert!(self.fieldset().is_superset(fields));

                fields.for_each(|i| { std::hash::Hash::hash(&self[i], hasher); true });
            }
        }

        impl<K, $T, $($tt)*> FromEntity<K, $T> for $Container<$CT> {
            fn from_entity(keys: impl EntityKeys<Key = K>, fields: impl IntoIterator<Item = (K, $T)>) -> Option<Self> {
                let mut vec = $Container::new();

                for (k, v) in fields {
                    let index = keys.find_index(&k)?;

                    if let Some(value) = vec.get_mut(index) {
                        *value = Some(v);
                    } else {
                        while vec.len() < index {
                            vec.push(None);
                        }

                        vec.push(Some(v));
                    }
                }

                Some(vec)
            }

            fn from_entity_with_mut_keys(mut keys: impl MutableEntityKeys<Key = K>, fields: impl IntoIterator<Item = (K, $T)>) -> Self {
                let mut vec = $Container::new();

                for (k, v) in fields {
                    let index = keys.find_index_or_insert(k);

                    if let Some(value) = vec.get_mut(index) {
                        *value = Some(v);
                    } else {
                        while vec.len() < index {
                            vec.push(None);
                        }

                        vec.push(Some(v));
                    }
                }

                vec.shrink_to_fit();
                vec
            }
        }
    };
}

impl_tuple_for!(Vec<T (Option<T>) ()>);
impl_tuple_for!(SmallVec<T ([Option<T>; N]) (const N: usize)>);

// A `HashMap` can act both as a `Tuple` (with its values) and as a `FieldSet`
// (with its keys).
impl<K: Clone + Eq + std::hash::Hash, V: Clone> Tuple for HashMap<K, V> {
    type FieldSet = HashSet<K>;

    fn merge(&mut self, other: &Self, self_fields: &Self::FieldSet, other_fields: &Self::FieldSet) {
        for field in other_fields {
            if !self_fields.contains(field) {
                self.insert(field.clone(), other.get(field).unwrap().clone());
            }
        }
    }

    fn clear(&mut self, fields: &Self::FieldSet) {
        for field in fields {
            self.remove(field);
        }
    }
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone + Eq> FieldSet for HashMap<K, V> {
    fn is_empty(&self) -> bool {
        HashMap::is_empty(self)
    }

    fn union_in_place(&mut self, other: &Self) {
        for (k, v) in other {
            self.raw_entry_mut()
                .from_key(k)
                .or_insert_with(|| (k.clone(), v.clone()));
        }
    }

    fn intersect_in_place(&mut self, other: &Self) {
        self.drain_filter(move |k, _| !other.contains_key(k));
    }

    fn difference_in_place(&mut self, other: &Self) {
        self.drain_filter(move |k, _| other.contains_key(k));
    }

    fn intersects(&self, other: &Self) -> bool {
        if self.len() < other.len() {
            self.keys().any(move |k| other.contains_key(k))
        } else {
            other.keys().any(move |k| self.contains_key(k))
        }
    }
}

// A `HashSet` can act as a `FieldSet`.
impl<K: Clone + Eq + std::hash::Hash> FieldSet for HashSet<K> {
    fn is_empty(&self) -> bool {
        HashSet::is_empty(self)
    }

    fn union_in_place(&mut self, other: &Self) {
        for k in other {
            self.get_or_insert_with(k, K::clone);
        }
    }

    fn intersect_in_place(&mut self, other: &Self) {
        self.drain_filter(move |k| !other.contains(k));
    }

    fn difference_in_place(&mut self, other: &Self) {
        self.drain_filter(move |k| other.contains(k));
    }

    fn intersects(&self, other: &Self) -> bool {
        if self.len() < other.len() {
            self.iter().any(move |k| other.contains(k))
        } else {
            other.iter().any(move |k| self.contains(k))
        }
    }
}
