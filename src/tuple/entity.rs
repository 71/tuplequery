//! Utilities for building [`Tuple`]s out of entity-like objects, where we
//! define "entity-like" as a list of key-value pairs.

use smallvec::SmallVec;
use std::collections::hash_map::{Entry, HashMap};
use std::marker::PhantomData;

use crate::{clause::relation::Relation, Tuple};

use super::HasFieldSet;

/// Represents the possible keys in an entity.
pub trait EntityKeys {
    /// The type of a key.
    type Key;

    /// Returns the index of the key in its container, or `None` if this key
    /// is unknown.
    fn find_index(&self, key: &Self::Key) -> Option<usize>;
}

impl<'a, EK: EntityKeys> EntityKeys for &'a EK {
    type Key = EK::Key;

    fn find_index(&self, key: &Self::Key) -> Option<usize> {
        (*self).find_index(key)
    }
}

impl<K: Eq + std::hash::Hash> EntityKeys for HashMap<K, usize> {
    type Key = K;

    fn find_index(&self, key: &Self::Key) -> Option<usize> {
        self.get(key).copied()
    }
}

impl<K: Ord> EntityKeys for [K] {
    type Key = K;

    fn find_index(&self, key: &Self::Key) -> Option<usize> {
        self.binary_search(key).ok()
    }
}

impl<'a, K: Eq> EntityKeys for &'a [K] {
    type Key = K;

    fn find_index(&self, key: &Self::Key) -> Option<usize> {
        self.iter().position(|i| i == key)
    }
}

/// Represents the possible keys in an entity, with the possibility of adding
/// new keys.
pub trait MutableEntityKeys {
    /// The type of a key.
    type Key;

    /// Returns the index of the key in its container. If it does not exist, it
    /// will be inserted without invalidating the result of previous calls to
    /// this function.
    fn find_index_or_insert(&mut self, key: Self::Key) -> usize;
}

impl<'a, KS: MutableEntityKeys> MutableEntityKeys for &'a mut KS {
    type Key = KS::Key;

    fn find_index_or_insert(&mut self, key: Self::Key) -> usize {
        (*self).find_index_or_insert(key)
    }
}

impl<K: Eq + std::hash::Hash> MutableEntityKeys for HashMap<K, usize> {
    type Key = K;

    fn find_index_or_insert(&mut self, key: Self::Key) -> usize {
        let value_if_vacant = self.len();

        match self.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => *entry.insert(value_if_vacant),
        }
    }
}

impl<K: Eq> MutableEntityKeys for Vec<K> {
    type Key = K;

    fn find_index_or_insert(&mut self, key: Self::Key) -> usize {
        for (i, k) in self.iter().enumerate().rev() {
            if k == &key {
                return i;
            }
        }

        self.push(key);
        self.len() - 1
    }
}

impl<K: Eq, const N: usize> MutableEntityKeys for SmallVec<[K; N]> {
    type Key = K;

    fn find_index_or_insert(&mut self, key: Self::Key) -> usize {
        for (i, k) in self.iter().enumerate().rev() {
            if k == &key {
                return i;
            }
        }

        self.push(key);
        self.len() - 1
    }
}

/// An interface for [`Tuple`]s that can be built from entity-like objects.
pub trait FromEntity<K, V>: Sized {
    /// Returns the value corresponding to the given entity.
    ///
    /// If one of its keys cannot be found in `keys`, `None` will be returned.
    /// If multiple values have the same key, an arbitrary value will be used.
    fn from_entity(
        keys: impl EntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Option<Self>;

    /// Returns the value corresponding to the given entity, with the
    /// possibility of registering new keys when an unknown key is uncountered.
    ///
    /// If multiple values have the same key, an arbitrary value will be used.
    fn from_entity_with_mut_keys(
        keys: impl MutableEntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Self;
}

impl<K, V, T: FromEntity<K, V> + HasFieldSet> FromEntity<K, V> for (T::FieldSet, T) {
    fn from_entity(
        keys: impl EntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Option<Self> {
        let entity = T::from_entity(keys, fields)?;
        let fields = entity.fieldset();

        Some((fields, entity))
    }

    fn from_entity_with_mut_keys(
        keys: impl MutableEntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        let entity = T::from_entity_with_mut_keys(keys, fields);
        let fields = entity.fieldset();

        (fields, entity)
    }
}

/// A builder for a set of [`Relation`]s built from entities.
pub struct RelationBuilder<K, V, T: FromEntity<K, V> + HasFieldSet, KS: MutableEntityKeys<Key = K>>
{
    relations: HashMap<T::FieldSet, Vec<T>>,
    keys: KS,

    _v: PhantomData<*const (K, V)>,
}

impl<
        K,
        V,
        T: Tuple + FromEntity<K, V> + HasFieldSet<FieldSet = <T as Tuple>::FieldSet>,
        KS: MutableEntityKeys<Key = K>,
    > RelationBuilder<K, V, T, KS>
where
    <T as HasFieldSet>::FieldSet: std::hash::Hash,
{
    /// Creates a new [`RelationBuilder`] with the given keys initially known
    /// to the builder.
    ///
    /// Note that other keys may be used in calls to [`Self::push`].
    pub fn with_initial_variables(keys: KS) -> Self {
        Self {
            relations: HashMap::new(),
            keys,
            _v: PhantomData,
        }
    }

    /// Pushes an entity described by the given key-value pairs into the
    /// corresponding relation.
    pub fn push(&mut self, fields: impl IntoIterator<Item = (K, V)>) {
        let tuple = T::from_entity_with_mut_keys(&mut self.keys, fields);

        self.relations
            .entry(tuple.fieldset())
            .or_insert_with(|| Vec::with_capacity(1))
            .push(tuple);
    }

    /// Returns the built relations and output key container.
    pub fn build(self) -> (Vec<Relation<T, Vec<T>>>, KS) {
        // SAFETY: internally, tuples are assigned relations based on their
        // fields; we can thus ensure that every vector in a relation has the
        // same fields.
        let built_relations = self
            .relations
            .into_iter()
            .map(|(fields, vec)| unsafe { Relation::new(fields, vec) })
            .collect();
        let keys = self.keys;

        (built_relations, keys)
    }

    /// Returns the built relations.
    pub fn into_relations(self) -> Vec<Relation<T, Vec<T>>> {
        self.build().0
    }
}

impl<K, V, T: FromEntity<K, V> + HasFieldSet, KS: MutableEntityKeys<Key = K> + Default> Default
    for RelationBuilder<K, V, T, KS>
{
    fn default() -> Self {
        Self {
            relations: HashMap::new(),
            keys: KS::default(),
            _v: PhantomData,
        }
    }
}
