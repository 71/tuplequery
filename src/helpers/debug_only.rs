/// A struct whose operations are only performed in debug builds.
#[cfg(debug_assertions)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub struct DebugOnly<T>(T);

/// A struct whose operations are only performed in debug builds.
#[cfg(not(debug_assertions))]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub struct DebugOnly<T>(std::marker::PhantomData<*const T>);

impl<T> From<T> for DebugOnly<T> {
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    fn from(value: T) -> Self {
        #[cfg(not(debug_assertions))]
        let value = std::marker::PhantomData;

        Self(value)
    }
}

#[allow(dead_code)]
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
impl<T> DebugOnly<T> {
    /// Creates a new [`DebugOnly`] wrapper. In release builds, this is
    /// equivalent to `drop(value)`.
    #[inline]
    pub fn new(value: impl Fn() -> T) -> Self {
        #[cfg(not(debug_assertions))]
        let value = || std::marker::PhantomData;

        Self(value())
    }

    /// In debug builds, returns the underlying value.
    /// In release builds, returns `None`.
    #[inline]
    pub fn take(self) -> Option<T> {
        #[cfg(debug_assertions)]
        return Some(self.0);
        #[cfg(not(debug_assertions))]
        return None;
    }

    /// In debug builds, executes the given function with the underlying value.
    /// In release builds, does nothing.
    #[inline]
    pub fn debug_do(&self, f: impl FnOnce(&T) -> ()) {
        #[cfg(debug_assertions)]
        f(&self.0);
    }

    /// In debug builds, executes the given function with the underlying value.
    /// In release builds, does nothing.
    #[inline]
    pub fn debug_do_mut(&mut self, f: impl FnOnce(&mut T) -> ()) {
        #[cfg(debug_assertions)]
        f(&mut self.0);
    }

    /// In debug builds, asserts that the given function returns true.
    /// In release builds, does nothing.
    #[inline]
    pub fn debug_check(&self, f: impl FnOnce(&T) -> bool) {
        #[cfg(debug_assertions)]
        debug_assert!(f(&self.0));
    }

    /// In debug builds, asserts that the given function returns true.
    /// In release builds, does nothing.
    #[inline]
    pub fn debug_check_mut(&mut self, f: impl FnOnce(&mut T) -> bool) {
        #[cfg(debug_assertions)]
        debug_assert!(f(&mut self.0));
    }

    /// In debug builds, returns the value returned by `f` invoked on the
    /// underlying value.
    /// In release builds, returns `None`.
    #[inline]
    pub fn debug_get<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        #[cfg(debug_assertions)]
        return Some(f(&self.0));
        #[cfg(not(debug_assertions))]
        return None;
    }

    /// In debug builds, returns the value returned by `f` invoked on the
    /// underlying value.
    /// In release builds, returns `None`.
    #[inline]
    pub fn debug_get_mut<R>(&mut self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        #[cfg(debug_assertions)]
        return Some(f(&mut self.0));
        #[cfg(not(debug_assertions))]
        return None;
    }
}
