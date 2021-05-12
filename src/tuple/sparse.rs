//! The [`SparseTuple`] struct and its utilities.
use std::{cmp::Ordering, hash::Hash, iter::FromIterator};

use smallvec::SmallVec;

use super::{
    check_fields::DebugFields,
    entity::{EntityKeys, FromEntity, MutableEntityKeys},
    CloneTuple, EqTuple, HashTuple, OrdTuple,
};
use super::{Bitset, HasFieldSet, Tuple};

/// A [`Tuple`] implementation that can store an arbitrary number of `T`
/// values. If that number is lower than `N`, no allocation will be needed.
///
/// This implementation stores values in a continuous chunk of memory, and is
/// recommended for tuples that have sparse values, i.e. tuples that have a
/// lot of missing values.
///
/// It is safe, and should perform well for small (< 100 fields) tuples. It
/// does not waste any space, but does require a linear reordering of its
/// fields in [`Tuple::merge`].
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub struct SparseTuple<T, const N: usize>(SmallVec<[T; N]>, DebugFields<Bitset>);

impl<T, const N: usize> From<&mut [Option<T>]> for SparseTuple<T, N> {
    fn from(values: &mut [Option<T>]) -> Self {
        let size = values.iter().filter(|x| x.is_some()).count();
        let mut smallvec = SmallVec::with_capacity(size);
        let mut bitset = Bitset::new();

        for (i, value) in values.iter_mut().enumerate() {
            if value.is_none() {
                continue;
            }

            let value = std::mem::replace(value, None).unwrap();

            smallvec.push(value);
            bitset.on(i);
        }

        Self(smallvec, DebugFields::from(bitset))
    }
}

impl<T, const N: usize> FromIterator<Option<T>> for SparseTuple<T, N> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let mut bitset = Bitset::new();
        let mut smallvec = SmallVec::new();

        for (i, value) in iter.into_iter().enumerate() {
            if let Some(value) = value {
                smallvec.push(value);
                bitset.on(i);
            }
        }

        smallvec.shrink_to_fit();

        Self(smallvec, DebugFields::from(bitset))
    }
}

impl<K, V, const N: usize> FromEntity<K, V> for SparseTuple<V, N> {
    fn from_entity(
        keys: impl EntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Option<Self> {
        Some(Self::from_iter::<SmallVec<[Option<V>; N]>>(
            SmallVec::from_entity(keys, fields)?,
        ))
    }

    fn from_entity_with_mut_keys(
        keys: impl MutableEntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        Self::from_iter::<SmallVec<[Option<V>; N]>>(SmallVec::from_entity_with_mut_keys(
            keys, fields,
        ))
    }
}

impl<K, V: Clone, const N: usize> FromEntity<K, V> for (Bitset, SparseTuple<V, N>) {
    fn from_entity(
        keys: impl EntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Option<Self> {
        let smallvec = SmallVec::<[Option<V>; N]>::from_entity(keys, fields)?;
        let fields = smallvec.fieldset();

        Some((fields, SparseTuple::from_iter(smallvec)))
    }

    fn from_entity_with_mut_keys(
        keys: impl MutableEntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        let smallvec = SmallVec::<[Option<V>; N]>::from_entity_with_mut_keys(keys, fields);
        let fields = smallvec.fieldset();

        (fields, SparseTuple::from_iter(smallvec))
    }
}

impl<T, const N: usize> SparseTuple<T, N> {
    fn index_in_self(fields: &Bitset, i: usize) -> Option<usize> {
        fields.iter().position(|x| x == i)
    }

    /// Returns a reference to the `i`th value of the tuple.
    pub fn get(&self, fields: &Bitset, i: usize) -> Option<&T> {
        self.1.check_fields(fields);

        self.0.get(Self::index_in_self(fields, i)?)
    }

    /// Returns a mutable reference to the `i`th value of the tuple.
    pub fn get_mut(&mut self, fields: &Bitset, i: usize) -> Option<&mut T> {
        self.1.check_fields(fields);

        self.0.get_mut(Self::index_in_self(fields, i)?)
    }
}

impl<T: Clone, const N: usize> Tuple for SparseTuple<T, N> {
    type FieldSet = Bitset;

    fn merge(&mut self, other: &Self, self_fields: &Self::FieldSet, other_fields: &Self::FieldSet) {
        self.1.check_fields(self_fields);
        other.1.check_fields(other_fields);

        let size = self_fields.count_ones_in_union(other_fields);

        let mut self_values_index = 0;
        let mut self_values = std::mem::replace(&mut self.0, SmallVec::with_capacity(size));
        let mut self_values = self_values.drain(..);

        for (index, in_self, in_other) in self_fields.iter_both(other_fields) {
            if in_self {
                let real_index = Self::index_in_self(self_fields, index).unwrap();

                while self_values_index < real_index {
                    self_values.next();
                    self_values_index += 1;
                }

                self.0.push(self_values.next().unwrap());
                self_values_index += 1;
            } else {
                debug_assert!(in_other);

                self.0
                    .push(other.0[Self::index_in_self(other_fields, index).unwrap()].clone());
            }
        }

        self.1.merge_fields(other_fields);
    }

    fn merge_owned(
        &mut self,
        mut other: Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        self.1.check_fields(self_fields);
        other.1.check_fields(other_fields);

        let size = self_fields.count_ones_in_union(other_fields);

        let mut self_values_index = 0;
        let mut self_values = std::mem::replace(&mut self.0, SmallVec::with_capacity(size));
        let mut self_values = self_values.drain(..);

        let mut other_values_index = 0;
        let mut other_values = other.0.drain(..);

        for (index, in_self, in_other) in self_fields.iter_both(other_fields) {
            if in_self {
                let real_index = Self::index_in_self(self_fields, index).unwrap();

                while self_values_index < real_index {
                    self_values.next();
                    self_values_index += 1;
                }

                self.0.push(self_values.next().unwrap());
                self_values_index += 1;
            } else {
                debug_assert!(in_other);

                let real_index = Self::index_in_self(other_fields, index).unwrap();

                while other_values_index < real_index {
                    other_values.next();
                    other_values_index += 1;
                }

                self.0.push(other_values.next().unwrap());
                other_values_index += 1;
            }
        }

        self.1.merge_fields(other_fields);
    }

    fn clear(&mut self, _fields: &Self::FieldSet) {
        self.0.clear();
        self.1.clear_fields();
    }
}

impl<T: Clone + Eq, const N: usize> EqTuple for SparseTuple<T, N> {
    fn eq(&self, other: &Self, fields: &Self::FieldSet) -> bool {
        self.1.check_fields_subset(fields);
        other.1.check_fields_subset(fields);

        fields
            .iter()
            .all(|i| self.get(fields, i) == other.get(fields, i))
    }
}

impl<T: Clone + Ord, const N: usize> OrdTuple for SparseTuple<T, N> {
    fn cmp(&self, other: &Self, fields: &Self::FieldSet) -> Ordering {
        self.1.check_fields_subset(fields);
        other.1.check_fields_subset(fields);

        for i in fields.iter() {
            match self.get(fields, i).cmp(&other.get(fields, i)) {
                Ordering::Equal => continue,
                ordering => return ordering,
            }
        }

        Ordering::Equal
    }
}

impl<T: Clone + std::hash::Hash, const N: usize> HashTuple for SparseTuple<T, N> {
    fn hash<H: std::hash::Hasher>(&self, fields: &Self::FieldSet, hasher: &mut H) {
        self.1.check_fields_subset(fields);

        fields.iter().for_each(|i| self.get(fields, i).hash(hasher));
    }
}

impl<T: Clone, const N: usize> CloneTuple for SparseTuple<T, N> {
    fn clone(&self, fields: &Self::FieldSet) -> Self {
        self.1.check_fields_subset(fields);

        Self(
            fields
                .iter()
                .map(|i| self.get(fields, i).unwrap().clone())
                .collect(),
            DebugFields::new(fields),
        )
    }
}
