//! The [`Clause`] trait.

pub mod relation;

use crate::tuple::Tuple;

/// A [`Query`](crate::Query) clause.
pub trait Clause<T: Tuple>: Sized {
    /// Returns the set of all fields that must be set on [`Tuple`]s given
    /// to [`Self::transform_boxed`].
    ///
    /// If empty, [`Self::transform_empty`] will be called instead.
    fn input_variables(&self) -> T::FieldSet {
        T::FieldSet::default()
    }

    /// Returns the set of all fields set on [`Tuple`]s returned by
    /// [`Self::transform_boxed`] and [`Self::transform_empty`].
    fn output_variables(&self) -> T::FieldSet;

    /// Returns an iterator of tuples computed from the given tuples.
    ///
    /// These tuples will at least have the variables requested in
    /// [`Self::input_variables`].
    fn transform_boxed<'a>(
        self,
        input: Box<dyn Iterator<Item = T> + 'a>,
        variables: T::FieldSet,
    ) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: 'a,
        T: 'a,
        T::FieldSet: 'a,
    {
        let _ = (input, variables);

        self.transform_empty()
    }

    /// Returns an iterator of tuples.
    fn transform_empty<'a>(self) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: 'a,
        T: 'a,
        T::FieldSet: 'a;
}

/// A `(T::FieldSet, Iterator<Item = T>)` pair can be used as a simple
/// [`Clause`] with no input variables.
impl<'i, T: Tuple, I: IntoIterator<Item = T> + 'i> Clause<T> for (T::FieldSet, I) {
    fn output_variables(&self) -> T::FieldSet {
        self.0.clone()
    }

    fn transform_empty<'a>(self) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: 'a,
    {
        Box::new(self.1.into_iter())
    }
}
