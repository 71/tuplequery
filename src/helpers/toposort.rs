use std::mem::MaybeUninit;

/// Returns a vector containing all the items in `items` sorted topologically
/// such that if `is_edge_from(from, to)` returns `true`, then `from` appears
/// after `to` in the output vector.
///
/// If there is a cycle in the edges, returns the index of one of the items
/// that belongs to the cycle.
pub fn toposort<T>(items: Vec<T>, is_edge_from: impl Fn(&T, &T) -> bool) -> Result<Vec<T>, usize> {
    // Over the execution of `toposort_visit`, we move all items out of
    // `items`. We need to keep their indices consistent througout the
    // execution of the loop below, so we wrap them all in `MaybeUninit<T>`
    // values that we slowly move out of `items`.
    // SAFETY: in a slice of memory, `MaybeUninit<T>` is guaranteed to have
    // the same layout as `T`.
    let mut items = unsafe { std::mem::transmute::<_, Vec<_>>(items) };
    // When all dependencies of an item have been processed, we move that item
    // out of `items` into `result`. We also change its state to `Seen`,
    // ensuring that we don't look at its (now uninitialized) value in `items`
    // again.
    let mut result = Vec::with_capacity(items.len());
    // At first, all items are `Unseen`.
    let mut state = vec![State::Unseen; items.len()];

    for i in 0..items.len() {
        if !toposort_visit(i, &mut items, &mut state, &mut result, &is_edge_from) {
            // `toposort_visit` detected a cycle originating at `i`. Before
            // returning an error, manually drop all items that have not been
            // moved out of `items` yet.
            for (i, state) in state.into_iter().enumerate() {
                if state != State::Seen {
                    // SAFETY: items are only consumed (and therefore
                    // uninitialized) when their corresponding state is set to
                    // `Seen`. We ensure that `state != Seen` above.
                    unsafe { std::ptr::drop_in_place(items[i].as_mut_ptr()) };
                }
            }

            return Err(i);
        }
    }

    debug_assert_eq!(items.len(), result.len());

    Ok(result)
}

/// The state of an item in [`toposort_visit`].
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum State {
    Unseen,
    Visiting,
    Seen,
}

/// Recursively visits the node at the given index; returns `true` if the
/// visit completes successfully, or `false` if a cycle was detected.
fn toposort_visit<T>(
    index: usize,
    items: &mut [MaybeUninit<T>],
    state: &mut [State],
    result: &mut Vec<T>,
    is_edge_from: &dyn Fn(&T, &T) -> bool,
) -> bool {
    match state[index] {
        State::Seen => true,
        State::Visiting => false, // Cycle.
        State::Unseen => {
            // Indicate that the item is being visited to detect cycles and
            // ensure no recursive call to `toposort_visit` can move
            // `items[index]` out of `items`.
            state[index] = State::Visiting;

            // SAFETY: items are only consumed (and therefore uninitialized)
            // when their corresponding state is set to `Seen`. Here, the state
            // is `Unseen`.
            let value = unsafe { &*items[index].as_ptr() };

            // Iterate over all unseen items. If `items[index]` depends on
            // `items[i]`, we recursively call `toposort_visit` to add that
            // dependency to `result` before `items[index]` itself.
            let mut i = 0;

            while let Some(i_state) = state.get(i) {
                if i != index && *i_state != State::Seen {
                    // SAFETY: the `Unseen` and `Visiting` states indicate that
                    // the item has not been moved out of the array yet.
                    let i_value = unsafe { &*items[i].as_ptr() };

                    if is_edge_from(value, i_value) {
                        if !toposort_visit(i, items, state, result, is_edge_from) {
                            // Cycle encountered.
                            return false;
                        }
                    }
                }

                i += 1;
            }

            // SAFETY: item at `index` was `Unseen` and is guaranteed to still
            // have a value in `items[index]`.
            result.push(unsafe {
                std::mem::replace(&mut items[index], MaybeUninit::uninit()).assume_init()
            });
            state[index] = State::Seen;

            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::toposort;

    #[test]
    fn test_simple() {
        // `is_edge_from(a, b)` iff string `b` appears in `a`.
        assert_eq!(
            toposort(vec!["c", "bc", "a", "abcd", "bcd", "b", "d"], |a, b| a
                .contains(b)),
            Ok(vec!["c", "b", "bc", "a", "d", "bcd", "abcd"]),
        );
    }

    #[test]
    fn test_direct_cycle() {
        // `is_edge_from(a, b)` iff number `b` is divisible by `a`.
        assert_eq!(toposort(vec![1, 2, 1], |a, b| b % a == 0), Err(0),);
    }

    #[test]
    fn test_indirect_cycle() {
        // A depends on B, B depends on C, C depends on A.
        #[derive(Debug, PartialEq, Eq)]
        enum V {
            A,
            B,
            C,
            D,
        }

        use V::*;

        fn is_edge_from(from: &V, to: &V) -> bool {
            match (from, to) {
                (A, B) => true,
                (B, C) => true,
                (C, A) => true,
                _ => false,
            }
        }

        assert_eq!(toposort(vec![D, B, C, A], is_edge_from), Err(1),);
    }
}
