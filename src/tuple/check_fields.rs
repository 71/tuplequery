//! The [`DebugFields`] struct, used internally to ensure the correct
//! execution of queries in debug builds with unsafe tuples.
use crate::{helpers::DebugOnly, FieldSet};

/// A wrapper around a [`Tuple`] that ensures in debug builds that it is
/// always called with the right fields.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub(crate) struct DebugFields<F: FieldSet> {
    pub(crate) fields: DebugOnly<F>,
}

impl<F: FieldSet> Drop for DebugFields<F> {
    fn drop(&mut self) {
        self.fields.debug_check(|fields| fields.is_empty());
    }
}

impl<F: FieldSet> From<F> for DebugFields<F> {
    fn from(fields: F) -> Self {
        Self {
            fields: DebugOnly::from(fields),
        }
    }
}

impl<F: FieldSet> DebugFields<F> {
    /// Creates a new [`DebugFields`] wrapping the given [`FieldSet`].
    pub(crate) fn new(fields: &F) -> Self {
        Self {
            fields: DebugOnly::new(move || fields.clone()),
        }
    }

    #[inline(always)]
    pub(crate) fn check_fields(&self, given_fields: &F) {
        self.fields.debug_check(|fields| fields == given_fields);
    }

    #[inline(always)]
    pub(crate) fn check_fields_subset(&self, subset_of_self: &F) {
        self.fields
            .debug_check(|fields| fields.is_superset(subset_of_self));
    }

    #[inline(always)]
    pub(crate) fn check_no_overlap(self_fields: &F, other_fields: &F) {
        debug_assert!(!self_fields.intersects(other_fields));
    }

    #[inline(always)]
    pub(crate) fn merge_fields(&mut self, other_fields: &F) {
        self.fields
            .debug_do_mut(|fields| fields.union_in_place(other_fields));
    }

    #[inline(always)]
    pub(crate) fn clear_fields(&mut self) {
        self.fields.debug_do_mut(|fields| *fields = F::default());
    }
}
