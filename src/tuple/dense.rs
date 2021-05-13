//! The [`DenseTuple`] struct and its utilities.
use std::mem::MaybeUninit;
use std::{cmp::Ordering, iter::FromIterator};

use smallvec::SmallVec;

use super::{
    check_fields::DebugFields,
    entity::{EntityKeys, FromEntity, MutableEntityKeys},
    OrdTuple,
};
use super::{Bitset, CloneTuple, EqTuple, HasFieldSet, HashTuple, Tuple};

/// A [`Tuple`] implementation that can store an arbitrary number of `T`
/// values. If that number is lower than `N`, no allocation will be needed.
///
/// This implementation allocates memory to store any information ahead of
/// time, and is recommended for tuples that have dense values, i.e. tuples
/// that have few missing values.
///
///
/// # Safety
///
/// [`DenseTuple`] does not store what values it has set, and thus should be
/// carefully manipulated to avoid leaks and the use of uninitialized data.
/// Users who only want to use safe implementations should use
/// `SmallVec<[Option<T>; N]>`, which is safe but uses more memory if `T`
/// is not [non-zero](
/// https://rust-lang.github.io/rfcs/2307-concrete-nonzero-types.html).
///
/// Most operations related to this struct are `unsafe`, since passing wrong
/// fields will result in undefined behavior. Unless using a function whose
/// name ends with `_unchecked`, the usage of the right bitset is safe.
pub struct DenseTuple<T, const N: usize>(SmallVec<[MaybeUninit<T>; N]>, DebugFields<Bitset>);

impl<T, const N: usize> DenseTuple<T, N> {
    /// Creates a new empty [`DenseTuple`].
    pub fn new(size: usize) -> Self {
        DenseTuple(
            std::iter::repeat_with(MaybeUninit::uninit)
                .take(size)
                .collect(),
            DebugFields::new(&Bitset::new()),
        )
    }

    fn take(mut self) -> SmallVec<[MaybeUninit<T>; N]> {
        self.1.clear_fields();

        std::mem::replace(&mut self.0, SmallVec::new())
    }

    /// Extracts the values set in the [`DenseTuple`].
    pub unsafe fn extract(self, fields: &Bitset) -> SmallVec<[T; N]> {
        self.1.check_fields(fields);

        let underlying = self.take();
        let values_cnt = fields.count_ones();

        if values_cnt == underlying.len() {
            // SAFETY: if all elements are initialized,
            // `SmallVec<[MaybeUninit<T>; N]>` and `SmallVec<[T; N]>` are
            // equivalent.
            return underlying
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();
        }

        let mut values = SmallVec::with_capacity(values_cnt);

        for (i, value) in underlying.into_iter().enumerate() {
            if fields.has(i) {
                values.push(value.assume_init());
            }
        }

        values
    }

    /// Returns a reference to the value at the given index.
    pub unsafe fn get(&self, i: usize, fields: &Bitset) -> Option<&T> {
        self.1.check_fields_subset(fields);

        if fields.has(i) {
            Some(self.get_unchecked(i, fields))
        } else {
            None
        }
    }

    /// Returns a reference to the value at the given index (without runtime
    /// checks).
    pub unsafe fn get_unchecked(&self, i: usize, fields: &Bitset) -> &T {
        self.1.check_fields_subset(fields);

        debug_assert!(fields.has(i));
        debug_assert!(self.0.len() > i);

        &*self.0.get_unchecked(i).as_ptr()
    }

    /// Returns a mutable reference to the value at the given index.
    pub unsafe fn get_mut(&mut self, i: usize, fields: &Bitset) -> Option<&mut T> {
        self.1.check_fields_subset(fields);

        if fields.has(i) {
            Some(self.get_mut_unchecked(i, fields))
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value at the given index (without
    /// runtime checks).
    pub unsafe fn get_mut_unchecked(&mut self, i: usize, fields: &Bitset) -> &mut T {
        self.1.check_fields_subset(fields);

        debug_assert!(fields.has(i));
        debug_assert!(self.0.len() > i);

        &mut *self.0.get_unchecked_mut(i).as_mut_ptr()
    }

    /// Returns an iterator over all the values of the [`DenseTuple`].
    pub unsafe fn into_iter<'a, const M: usize>(
        self,
        fields: Bitset,
    ) -> impl Iterator<Item = Option<T>> {
        self.1.check_fields(&fields);

        self.take().into_iter().enumerate().map(move |(i, maybe)| {
            if fields.has(i) {
                Some(unsafe { maybe.assume_init() })
            } else {
                None
            }
        })
    }

    /// Returns an iterator over all the set values of the [`DenseTuple`].
    pub unsafe fn iter(&self, fields: Bitset) -> impl Iterator<Item = &T> + '_ {
        self.1.check_fields_subset(&fields);

        fields.iter().map(move |i| unsafe { &*self.0[i].as_ptr() })
    }

    /// Returns an iterator over all the set values of the [`DenseTuple`]
    /// (using mutable references).
    pub unsafe fn iter_mut(&mut self, fields: Bitset) -> impl Iterator<Item = &mut T> + '_ {
        self.1.check_fields_subset(&fields);

        fields
            .iter()
            .map(move |i| unsafe { &mut *self.0[i].as_mut_ptr() })
    }

    /// Returns a reference to the tuple that can be printed.
    pub unsafe fn as_ref(&self, fields: Bitset) -> DenseTupleRef<T> {
        self.1.check_fields_subset(&fields);

        DenseTupleRef {
            bitset: fields,
            tuple: self.0.as_ref(),
        }
    }
}

impl<T, const N: usize> Default for DenseTuple<T, N> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Clone, const N: usize> Tuple for DenseTuple<T, N> {
    type FieldSet = Bitset;

    fn merge(&mut self, other: &Self, self_fields: &Self::FieldSet, other_fields: &Self::FieldSet)
    where
        T: Clone,
    {
        self.1.check_fields(self_fields);
        other.1.check_fields(other_fields);

        for (i, value) in other.0.iter().enumerate() {
            if !other_fields.has(i) {
                continue;
            }

            if !self_fields.has(i) {
                // SAFETY: the field is initialized.
                self.0[i] = MaybeUninit::new(unsafe { &*value.as_ptr() }.clone());
            }
        }

        self.1.merge_fields(other_fields);
    }

    fn merge_owned(
        &mut self,
        other: Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        self.1.check_fields(self_fields);
        other.1.check_fields(other_fields);

        for (i, mut value) in other.take().into_iter().enumerate() {
            if !other_fields.has(i) {
                continue;
            }

            if self_fields.has(i) {
                // SAFETY: the field is initialized.
                unsafe { std::ptr::drop_in_place(value.as_mut_ptr()) };
            } else {
                while i >= self.0.len() {
                    self.0.push(MaybeUninit::uninit());
                }

                self.0[i] = value;
            }
        }

        self.1.merge_fields(other_fields);
    }

    fn merge_no_overlap(
        &mut self,
        other: &Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        DebugFields::check_no_overlap(self_fields, other_fields);

        self.1.check_fields(self_fields);
        other.1.check_fields(other_fields);

        for (i, value) in other.0.iter().enumerate() {
            if !other_fields.has(i) {
                continue;
            }

            while i >= self.0.len() {
                self.0.push(MaybeUninit::uninit());
            }

            // SAFETY: the field is initialized.
            self.0[i] = MaybeUninit::new(unsafe { &*value.as_ptr() }.clone());
        }

        self.1.merge_fields(other_fields);
    }

    fn merge_owned_no_overlap(
        &mut self,
        other: Self,
        self_fields: &Self::FieldSet,
        other_fields: &Self::FieldSet,
    ) {
        DebugFields::check_no_overlap(self_fields, other_fields);

        self.1.check_fields(self_fields);
        other.1.check_fields(other_fields);

        for (i, value) in other.take().into_iter().enumerate() {
            if !other_fields.has(i) {
                continue;
            }

            while i >= self.0.len() {
                self.0.push(MaybeUninit::uninit());
            }

            self.0[i] = value;
        }

        self.1.merge_fields(other_fields);
    }

    fn clear(&mut self, fields: &Self::FieldSet) {
        self.1.check_fields(fields);

        for index in fields.iter() {
            // SAFETY: we're only accessing initialized fields.
            unsafe { std::ptr::drop_in_place(self.0[index].as_mut_ptr()) };
        }

        self.1.clear_fields();
    }
}

impl<T: Clone + Eq, const N: usize> EqTuple for DenseTuple<T, N> {
    fn eq(&self, other: &Self, fields: &Self::FieldSet) -> bool {
        self.1.check_fields_subset(fields);
        other.1.check_fields_subset(fields);

        fields
            .iter()
            .all(|i| unsafe { self.get_unchecked(i, fields) == other.get_unchecked(i, fields) })
    }
}

impl<T: Clone + Ord, const N: usize> OrdTuple for DenseTuple<T, N> {
    fn cmp(&self, other: &Self, fields: &Self::FieldSet) -> Ordering {
        self.1.check_fields_subset(fields);
        other.1.check_fields_subset(fields);

        for i in fields.iter() {
            let (self_value, other_value) = unsafe {
                (
                    self.get_unchecked(i, fields),
                    other.get_unchecked(i, fields),
                )
            };

            match self_value.cmp(other_value) {
                Ordering::Equal => continue,
                ordering => return ordering,
            }
        }

        Ordering::Equal
    }
}

impl<T: Clone + std::hash::Hash, const N: usize> HashTuple for DenseTuple<T, N> {
    fn hash<H: std::hash::Hasher>(&self, fields: &Self::FieldSet, hasher: &mut H) {
        self.1.check_fields_subset(fields);

        unsafe {
            // SAFETY: the caller guarantees that these are the right fields.
            self.iter(fields.clone())
        }
        .for_each(|t| t.hash(hasher));
    }
}

impl<T: Clone, const N: usize> CloneTuple for DenseTuple<T, N> {
    fn clone(&self, fields: &Self::FieldSet) -> Self {
        self.1.check_fields_subset(fields);

        let mut values = SmallVec::with_capacity(self.0.len());

        for (i, value) in self.0.iter().enumerate() {
            values.push(if fields.has(i) {
                // SAFETY: the fields include this value, so it is initialized.
                MaybeUninit::new(unsafe { &*value.as_ptr() }.clone())
            } else {
                MaybeUninit::uninit()
            });
        }

        DenseTuple(values, DebugFields::new(fields))
    }
}

impl<T, const N: usize> From<&mut [Option<T>]> for DenseTuple<T, N> {
    fn from(values: &mut [Option<T>]) -> Self {
        let mut bitset = Bitset::new();
        let smallvec = values
            .iter_mut()
            .enumerate()
            .map(|(i, v)| {
                if v.is_some() {
                    bitset.on(i);

                    MaybeUninit::new(std::mem::replace(v, None).unwrap())
                } else {
                    MaybeUninit::uninit()
                }
            })
            .collect();

        DenseTuple(smallvec, DebugFields::from(bitset))
    }
}

impl<T, const N: usize> FromIterator<Option<T>> for DenseTuple<T, N> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let mut bitset = Bitset::new();
        let smallvec = iter
            .into_iter()
            .enumerate()
            .map(|(i, v)| match v {
                Some(v) => {
                    bitset.on(i);
                    MaybeUninit::new(v)
                }
                None => MaybeUninit::uninit(),
            })
            .collect();

        DenseTuple(smallvec, DebugFields::from(bitset))
    }
}

impl<K, V, const N: usize> FromEntity<K, V> for DenseTuple<V, N> {
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

impl<K, V: Clone, const N: usize> FromEntity<K, V> for (Bitset, DenseTuple<V, N>) {
    fn from_entity(
        keys: impl EntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Option<Self> {
        let smallvec = SmallVec::<[Option<V>; N]>::from_entity(keys, fields)?;
        let fields = smallvec.fieldset();

        Some((fields, DenseTuple::from_iter(smallvec)))
    }

    fn from_entity_with_mut_keys(
        keys: impl MutableEntityKeys<Key = K>,
        fields: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        let smallvec = SmallVec::<[Option<V>; N]>::from_entity_with_mut_keys(keys, fields);
        let fields = smallvec.fieldset();

        (fields, DenseTuple::from_iter(smallvec))
    }
}

/// A reference to a [`DenseTuple`] along with its fields.
pub struct DenseTupleRef<'a, T> {
    bitset: Bitset,
    tuple: &'a [MaybeUninit<T>],
}

impl<'a, T> DenseTupleRef<'a, T> {
    /// Returns a reference to the value at the given index.
    pub fn get(&self, i: usize) -> Option<&T> {
        if self.bitset.has(i) {
            // SAFETY: we only access the values in the fieldset.
            Some(unsafe { self.get_unchecked(i) })
        } else {
            None
        }
    }

    /// Returns a reference to the value at the given index (without runtiime
    /// checks).
    pub unsafe fn get_unchecked(&self, i: usize) -> &T {
        debug_assert!(self.bitset.has(i));

        &*self.tuple[i].as_ptr()
    }

    /// Returns an iterator over the values.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        // SAFETY: we only access the values in the fieldset.
        self.bitset
            .iter()
            .map(move |i| unsafe { &*self.tuple[i].as_ptr() })
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for DenseTupleRef<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut debug_list = f.debug_list();

        for i in 0..self.tuple.len() {
            if let Some(v) = self.get(i) {
                debug_list.entry(v);
            } else {
                debug_list.entry(&());
            }
        }

        debug_list.finish()
    }
}

/// A builder for creating a [`DenseTuple`] and its corresponding [`Bitset`].
pub struct DenseTupleBuilder<T, const N: usize> {
    bitset: Bitset,
    tuple: SmallVec<[MaybeUninit<T>; N]>,
}

impl<T, const N: usize> Default for DenseTupleBuilder<T, N> {
    fn default() -> Self {
        Self {
            bitset: Bitset::default(),
            tuple: SmallVec::default(),
        }
    }
}

impl<T, const N: usize> DenseTupleBuilder<T, N> {
    /// Creates a new empty [`DenseTupleBuilder<T, N>`].
    pub fn new() -> Self {
        DenseTupleBuilder::default()
    }

    /// Creates a new empty [`DenseTupleBuilder<T, N>`] with the given
    /// capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut tuple = SmallVec::with_capacity(capacity);
        tuple.resize_with(capacity, MaybeUninit::uninit);
        DenseTupleBuilder {
            bitset: Bitset::new(),
            tuple,
        }
    }

    /// Returns the resulting [`DenseTuple`] and its corresponding [`Bitset`].
    pub fn build(self) -> (DenseTuple<T, N>, Bitset) {
        let debug_fields = DebugFields::new(&self.bitset);

        (DenseTuple(self.tuple, debug_fields), self.bitset)
    }

    /// Same as [`Self::build`], but swaps its result.
    pub fn build_swapped(self) -> (Bitset, DenseTuple<T, N>) {
        let debug_fields = DebugFields::new(&self.bitset);

        (self.bitset, DenseTuple(self.tuple, debug_fields))
    }

    /// Returns the [`Bitset`] for the built tuple.
    pub fn bitset(&self) -> Bitset {
        self.bitset.clone()
    }

    /// Sets the value at the given index, and returns the previous value if
    /// there was any.
    pub fn set(&mut self, index: usize, value: T) -> Option<T> {
        let result = if self.bitset.has(index) {
            // SAFETY: bitset has index so value was assigned to.
            let value = unsafe {
                std::mem::replace(&mut self.tuple[index], MaybeUninit::uninit()).assume_init()
            };

            Some(value)
        } else {
            None
        };

        if index >= self.tuple.len() {
            self.tuple.resize_with(index + 1, MaybeUninit::uninit);
        }

        self.tuple[index] = MaybeUninit::new(value);
        self.bitset.on(index);

        result
    }
}

impl<T, const N: usize> Into<DenseTuple<T, N>> for DenseTupleBuilder<T, N> {
    fn into(self) -> DenseTuple<T, N> {
        DenseTuple(self.tuple, DebugFields::from(self.bitset))
    }
}

impl<T, const N: usize> Into<Bitset> for DenseTupleBuilder<T, N> {
    fn into(self) -> Bitset {
        self.bitset
    }
}

impl<T, const N: usize> From<&mut [Option<T>]> for DenseTupleBuilder<T, N> {
    fn from(values: &mut [Option<T>]) -> Self {
        let mut builder = Self::with_capacity(values.len());

        for (i, value) in values.iter_mut().enumerate() {
            if value.is_none() {
                continue;
            }

            builder.set(i, std::mem::replace(value, None).unwrap());
        }

        builder
    }
}

impl<T, const N: usize> FromIterator<Option<T>> for DenseTupleBuilder<T, N> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Self::with_capacity(iter.size_hint().0);

        for (i, value) in iter.enumerate() {
            if let Some(value) = value {
                builder.set(i, value);
            }
        }

        builder
    }
}
