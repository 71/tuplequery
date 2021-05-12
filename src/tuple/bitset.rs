//! The [`Bitset`] struct along with its iterators.
use std::{
    iter::{FromIterator, FusedIterator},
    usize,
};

use smallvec::SmallVec;

use super::FieldSet;

type Bits = u32;
type Container = SmallVec<[Bits; 2]>;

/// A lightweight dynamically-sized bitset.
///
/// For bitsets where the bits >= 64 are never set, this bitset is stored
/// inline.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bitset(Container);

// Internal implementation note: we always make sure that zeros at the end of
// the bitset are erased, i.e. the underlying `SmallVec` is truncated.
// This ensures that int-by-int comparisons as performed by `PartialEq`, `Ord`
// and `Hash` always behave the same.

impl Bitset {
    const BITS: usize = std::mem::size_of::<Bits>() * 8;

    /// Returns the index in a `Container` where the bit at the given index can
    /// be found.
    #[inline]
    const fn pos(index: usize) -> usize {
        let diff = if index != 0 && index % Self::BITS == 0 {
            1
        } else {
            0
        };
        index / Self::BITS + diff
    }

    /// Returns the bit in `container[Self::pos(index)]` where the bit at the
    /// given index can be found.
    #[inline]
    const fn bit(index: usize) -> Bits {
        1 << (index % Self::BITS)
    }

    /// Returns a reference to the underlying container of the bitset.
    pub fn bits(&self) -> &[Bits] {
        &self.0
    }

    /// Creates a new empty bitset.
    pub fn new() -> Bitset {
        Self(SmallVec::new())
    }

    /// Returns the index of the greatest index.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// assert_eq!(bitset![0 1 0 1].greatest(), Some(3));
    /// assert_eq!(bitset![0 1 0 0].greatest(), Some(1));
    /// assert_eq!(bitset![0].greatest(), None);
    /// ```
    pub fn greatest(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &x)| x != 0)
            .map(|(i, x)| {
                (i + 1) * std::mem::size_of::<Bits>() * 8 - (x.leading_zeros() as usize + 1)
            })
    }

    /// Ensures that the bitset can contain bits at the given index.
    #[inline]
    pub fn ensure_size(&mut self, index: usize) {
        let pos = Bitset::pos(index);

        while pos >= self.0.len() {
            self.0.push(0);
        }
    }

    /// Returns whether the bit at the given index is set.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// assert!(bitset![0 1 0 1].has(1));
    /// assert!(bitset![0 1 0 1].has(3));
    /// assert!(!bitset![0 1 0 1].has(0));
    /// assert!(!bitset![0 1 0 1].has(2));
    /// ```
    pub fn has(&self, index: usize) -> bool {
        let pos = Bitset::pos(index);

        pos < self.0.len() && (self.0[Bitset::pos(index)] & Bitset::bit(index)) != 0
    }

    /// Toggles on the bit at the given index.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::tuple::bitset::Bitset;
    /// let mut bitset = Bitset::new();
    ///
    /// bitset.on(1);
    /// bitset.on(2);
    /// bitset.on(2);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![1, 2]);
    /// ```
    pub fn on(&mut self, index: usize) {
        self.ensure_size(index);
        self.0[Bitset::pos(index)] |= Bitset::bit(index);
    }

    /// Toggles off the bit at the given index.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::tuple::bitset::Bitset;
    /// let mut bitset = Bitset::new();
    ///
    /// bitset.on(1);
    /// bitset.on(2);
    /// bitset.on(2);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![1, 2]);
    ///
    /// bitset.off(1);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![2]);
    /// ```
    pub fn off(&mut self, index: usize) {
        if self.0.len() > Bitset::pos(index) {
            self.0[Bitset::pos(index)] &= !Bitset::bit(index);
            self.trim_after_removing_at(index);
        }
    }

    /// Sets the value of the bit at the given index.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::tuple::bitset::Bitset;
    /// let mut bitset = Bitset::new();
    ///
    /// bitset.set(1, true);
    /// bitset.set(2, true);
    /// bitset.set(2, true);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![1, 2]);
    ///
    /// bitset.set(1, false);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![2]);
    /// ```
    pub fn set(&mut self, index: usize, value: bool) {
        if value {
            self.ensure_size(index);
            self.0[Bitset::pos(index)] |= Bitset::bit(index);
        } else if self.0.len() > Bitset::pos(index) {
            self.0[Bitset::pos(index)] &= !Bitset::bit(index);
            self.trim_after_removing_at(index);
        }
    }

    /// Toggles the value of the bit at the given index and returns its new
    /// value.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::tuple::bitset::Bitset;
    /// let mut bitset = Bitset::new();
    ///
    /// bitset.on(2);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![2]);
    ///
    /// bitset.toggle(1);
    /// bitset.toggle(2);
    ///
    /// assert_eq!(bitset.iter().collect::<Vec<_>>(), vec![1]);
    /// ```
    pub fn toggle(&mut self, index: usize) -> bool {
        let (pos, bit) = (Bitset::pos(index), Bitset::bit(index));

        if (self.0[pos] & bit) == 0 {
            self.ensure_size(index);
            self.0[pos] |= bit;
            true
        } else {
            if self.0.len() > pos {
                self.0[pos] &= !bit;
                self.trim_after_removing_at(index);
            }
            false
        }
    }

    /// Sets all values of the bitset to 0.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// let mut bitset = bitset![1 1 0];
    ///
    /// assert_eq!(bitset.count_ones(), 2);
    ///
    /// bitset.clear();
    ///
    /// assert_eq!(bitset.count_ones(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Shrinks the bitset so that it does not use unneeded memory.
    pub fn shrink_to_fit(&mut self) {
        let unneeded_values = self.0.iter().rev().take_while(|&&x| x == 0).count();

        if unneeded_values > 0 {
            self.0.truncate(self.0.len() - unneeded_values);
            self.0.shrink_to_fit();
        }
    }

    /// Returns the number of 1s in the bitset.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// assert_eq!(bitset![].count_ones(), 0);
    /// assert_eq!(bitset![1 1].count_ones(), 2);
    /// assert_eq!(bitset![0 1 0 1 0 1].count_ones(), 3);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|x| x.count_ones()).sum::<u32>() as usize
    }

    /// Returns whether the bitset is empty, i.e. all bits are 0s.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// assert!(bitset![].is_empty());
    /// assert!(bitset![0].is_empty());
    /// assert!(!bitset![0 1].is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&x| x == 0)
    }

    /// Returns whether this bitset is a subset of the given bitset.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// assert!(bitset![].is_subset(&bitset![]));
    /// assert!(bitset![].is_subset(&bitset![1]));
    /// assert!(bitset![1].is_subset(&bitset![1]));
    /// assert!(bitset![0 1].is_subset(&bitset![0 1 1]));
    /// assert!(!bitset![1 0].is_subset(&bitset![0 1]));
    /// ```
    pub fn is_subset(&self, superset: &Bitset) -> bool {
        self.iter().all(move |i| superset.has(i))
    }

    /// Returns whether this bitset is a superset of the given bitset.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// assert!(bitset![].is_superset(&bitset![]));
    /// assert!(bitset![1].is_superset(&bitset![]));
    /// assert!(bitset![1].is_superset(&bitset![1]));
    /// assert!(!bitset![0 1].is_superset(&bitset![0 1 1]));
    /// assert!(!bitset![1 0].is_superset(&bitset![0 1]));
    /// ```
    pub fn is_superset(&self, subset: &Bitset) -> bool {
        subset.is_subset(self)
    }

    /// Returns a mutable reference to the `idx`th integer in `self` used to
    /// represent bits.
    #[inline]
    fn get_mut(&mut self, idx: usize) -> &mut Bits {
        while idx >= self.0.len() {
            self.0.push(0);
        }

        // SAFETY: the loop above can only complete when `idx < self.0.len()`.
        unsafe { self.0.get_unchecked_mut(idx) }
    }

    #[inline]
    fn trim_after_removing_at(&mut self, index: usize) {
        let pos = Bitset::pos(index);

        if self.0.len() == pos + 1 && self.0[pos] == 0 {
            self.0.pop();
        }
    }

    /// Returns an iterator over the indices of the 1s in the bitset.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// let bitset = bitset![0 0 1 0 1];
    ///
    /// assert_eq!(bitset.into_iter().collect::<Vec<_>>(), vec![2, 4]);
    /// ```
    pub fn into_iter(self) -> Iter {
        Iter { bitset: self }
    }

    /// Returns an iterator over the indices of the 1s in the bitset. See
    /// [`into_iter`] for more details.
    pub fn iter(&self) -> Iter {
        self.clone().into_iter()
    }

    /// Returns an iterator where each item is a triple
    /// `(index, is_in_self, in_in_other)`.
    ///
    /// # Example
    /// ```
    /// # use tuplequery::bitset;
    /// let a = bitset![0 1 0 1];
    /// let b = bitset![1 1 0 0];
    ///
    /// assert_eq!(
    ///     a.iter_both(&b).collect::<Vec<_>>(),
    ///     vec![(0, false, true), (1, true, true), (3, true, false)],
    /// );
    /// ```
    pub fn iter_both(&self, other: &Self) -> IterBoth {
        IterBoth {
            a: self.iter(),
            b: other.iter(),
        }
    }

    /// More efficient version of `self.intersect(other).count_ones()`.
    pub fn count_ones_in_intersection(&self, other: &Self) -> usize {
        let mut ones = 0;

        for (&a, &b) in self.0.iter().zip(other.0.iter()) {
            ones += (a & b).count_ones() as usize;
        }

        ones
    }

    /// More efficient version of `self.union(other).count_ones()`.
    pub fn count_ones_in_union(&self, other: &Self) -> usize {
        let mut ones = 0;

        for (&a, &b) in self.0.iter().zip(other.0.iter()) {
            ones += (a | b).count_ones() as usize;
        }

        ones
    }
}

impl FieldSet for Bitset {
    fn is_empty(&self) -> bool {
        Bitset::is_empty(self)
    }

    fn union_in_place(&mut self, other: &Self) {
        self.ensure_size(other.0.len() * std::mem::size_of::<Bits>());

        for (v1, &v2) in self.0.iter_mut().zip(&other.0) {
            *v1 |= v2;
        }
    }

    fn intersect_in_place(&mut self, other: &Self) {
        self.ensure_size(other.0.len() * std::mem::size_of::<Bits>());

        for (v1, &v2) in self.0.iter_mut().zip(&other.0) {
            *v1 &= v2;
        }

        self.shrink_to_fit();
    }

    fn difference_in_place(&mut self, other: &Self) {
        for (v1, &v2) in self.0.iter_mut().zip(&other.0) {
            *v1 &= !v2;
        }

        self.shrink_to_fit();
    }

    fn intersects(&self, other: &Self) -> bool {
        self.0.iter().zip(&other.0).any(|(&x, &y)| (x & y) != 0)
    }

    fn union(&self, other: &Self) -> Self {
        let mut result = Bitset::new();

        for (i, v) in self.0.iter().copied().enumerate() {
            if v != 0 {
                *result.get_mut(i) = v;
            }
        }

        for (i, v) in other.0.iter().copied().enumerate() {
            if v != 0 {
                *result.get_mut(i) |= v;
            }
        }

        result
    }

    fn intersect(&self, other: &Self) -> Self {
        let mut result = Bitset::new();

        for (i, (u, v)) in self
            .0
            .iter()
            .copied()
            .zip(other.0.iter().copied())
            .enumerate()
        {
            let w = u & v;

            if w != 0 {
                *result.get_mut(i) = w;
            }
        }

        result
    }
}

impl std::ops::Index<usize> for Bitset {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        if self.has(index) {
            &true
        } else {
            &false
        }
    }
}

impl std::fmt::Debug for Bitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<usize> for Bitset {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut bitset = Bitset::new();

        for i in iter {
            bitset.on(i);
        }

        bitset
    }
}

impl FromIterator<bool> for Bitset {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut bitset = Bitset::new();

        for (i, value) in iter.into_iter().enumerate() {
            bitset.set(i, value);
        }

        bitset
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! write_bitset_bit {
    ( 0 $bitset: ident $i: ident ) => {
        $i += 1;
    };

    ( 1 $bitset: ident $i: ident ) => {
        $bitset.on($i);
        $i += 1;
    };
}

/// Creates a bitset with the given values.
///
/// # Example
/// ```
/// # use tuplequery::bitset;
/// let actual_indices = bitset![0 0 1 0 1].into_iter().collect::<Vec<_>>();
/// let expected_indices = vec![2, 4];
///
/// assert_eq!(actual_indices, expected_indices);
/// ```
#[macro_export]
macro_rules! bitset {
    [] => {
        $crate::tuple::Bitset::new()
    };

    [ $($l: tt)+ ] => {{
        let mut bitset = $crate::bitset![];
        let mut _i = 0;

        $($crate::write_bitset_bit!($l bitset _i);)*

        bitset
    }};
}

/// An iterator over a [`Bitset`]. See [`Bitset::iter`] for more information.
#[derive(Clone)]
pub struct Iter {
    bitset: Bitset,
}

impl Iter {
    /// Returns the next value returned by [`Self::next`] without advancing.
    pub fn peek_next(&self) -> Option<usize> {
        let mut index = 0;

        for bits in &self.bitset.0 {
            let v = *bits;

            if v == 0 {
                index += Bitset::BITS;
                continue;
            }

            index += v.trailing_zeros() as usize;

            return Some(index);
        }

        None
    }
}

impl Iterator for Iter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut index = 0;

        for bits in &mut self.bitset.0 {
            let v = *bits;

            if v == 0 {
                index += Bitset::BITS;
                continue;
            }

            let trailing = v.trailing_zeros();

            index += trailing as usize;
            *bits &= !(1 << trailing);

            return Some(index);
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bitset.count_ones();

        (len, Some(len))
    }
}

impl ExactSizeIterator for Iter {}
impl FusedIterator for Iter {}

/// An iterator over two [`Bitset`]s. See [`Bitset::iter_both`] for more
/// information.
#[derive(Clone)]
pub struct IterBoth {
    a: Iter,
    b: Iter,
}

impl Iterator for IterBoth {
    type Item = (usize, bool, bool);

    fn next(&mut self) -> Option<Self::Item> {
        let a_next = self.a.peek_next();
        let b_next = self.b.peek_next();

        match (a_next, b_next) {
            (Some(a), Some(b)) => {
                if a == b {
                    self.a.next();
                    self.b.next();

                    Some((a, true, true))
                } else if a < b {
                    self.a.next();

                    Some((a, true, false))
                } else {
                    self.b.next();

                    Some((b, false, true))
                }
            }

            (Some(a), None) => {
                self.a.next();

                Some((a, true, false))
            }
            (None, Some(b)) => {
                self.b.next();

                Some((b, false, true))
            }

            (None, None) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.a.bitset.count_ones_in_union(&self.b.bitset);

        (len, Some(len))
    }
}

impl ExactSizeIterator for IterBoth {}
impl FusedIterator for IterBoth {}
