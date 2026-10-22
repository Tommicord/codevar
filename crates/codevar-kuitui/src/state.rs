//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Observable state that survives a recomposition.
//!
//! A [`MutableState`] is the Kuitui equivalent of Compose's
//! `mutableStateOf`. Writing to it bumps its revision and marks it
//! dirty; the composition notices the dirty state and recomposes the
//! subtree that read it.
//!
//! States are shared by handle: cloning a [`MutableState`] yields
//! another handle to the *same* value, which is what makes a state
//! usable from a widget and from the code that owns it.

use alloc::rc::Rc;
use core::cell::{Cell, RefCell};

/// The value shared by the handles of a state.
struct StateCell<T> {
    /// The current value.
    value: T,
    /// How many times the value was written.
    revision: u64,
    /// Whether the value was written since the last clean-up.
    dirty: bool,
}

/// A read-only handle to a piece of state.
///
/// A `State` observes the value without being able to change it, which
/// lets a widget that only reads a value say so.
#[derive(Clone)]
pub struct State<T> {
    /// The shared value.
    cell: Rc<RefCell<StateCell<T>>>,
}

impl<T> State<T> {
    /// Reads the value.
    #[must_use]
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.cell.borrow().value.clone()
    }

    /// Runs `f` on the value.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let cell = self.cell.borrow();
        f(&cell.value)
    }

    /// The number of writes the value received.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.cell.borrow().revision
    }

    /// Whether the value was written since the last clean-up.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.cell.borrow().dirty
    }
}

/// A writable handle to a piece of state.
///
/// Writing to the state marks it dirty, which asks the composition to
/// recompose the widgets that read it.
#[derive(Clone)]
pub struct MutableState<T> {
    /// The shared value.
    cell: Rc<RefCell<StateCell<T>>>,
}

impl<T> MutableState<T> {
    /// Creates a state holding `value`.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self {
            cell: Rc::new(RefCell::new(StateCell {
                value,
                revision: 0,
                dirty: false,
            })),
        }
    }

    /// Reads the value.
    #[must_use]
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.cell.borrow().value.clone()
    }

    /// Runs `f` on the value.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let cell = self.cell.borrow();
        f(&cell.value)
    }

    /// Runs `f` on the value and marks the state dirty.
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut cell = self.cell.borrow_mut();
        let result = f(&mut cell.value);
        cell.revision = cell.revision.wrapping_add(1);
        cell.dirty = true;
        result
    }

    /// Replaces the value and marks the state dirty.
    pub fn set(&self, value: T) {
        let mut cell = self.cell.borrow_mut();
        cell.value = value;
        cell.revision = cell.revision.wrapping_add(1);
        cell.dirty = true;
    }

    /// The number of writes the value received.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.cell.borrow().revision
    }

    /// Whether the value was written since the last clean-up.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.cell.borrow().dirty
    }

    /// Marks the state as clean again.
    pub fn clear_dirty(&self) {
        self.cell.borrow_mut().dirty = false;
    }

    /// A read-only handle to the same value.
    #[must_use]
    pub fn as_state(&self) -> State<T> {
        State {
            cell: Rc::clone(&self.cell),
        }
    }
}

impl<T: Clone> MutableState<T> {
    /// The value with the dirty flag cleared.
    ///
    /// The composition calls this once it has recomposed the widgets
    /// that read the state.
    pub fn take_dirty(&self) -> bool {
        let mut cell = self.cell.borrow_mut();
        let was_dirty = cell.dirty;
        cell.dirty = false;
        was_dirty
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for MutableState<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MutableState")
            .field("value", &self.cell.borrow().value)
            .field("revision", &self.cell.borrow().revision)
            .finish()
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for State<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("State")
            .field("value", &self.cell.borrow().value)
            .field("revision", &self.cell.borrow().revision)
            .finish()
    }
}

/// A counter of the writes a set of states received.
///
/// The composition uses it to decide whether a recomposition is
/// needed without walking every state of the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RevisionCounter {
    /// The revisions seen so far.
    revisions: u64,
    /// The number of states that were written.
    writes: u64,
}

impl RevisionCounter {
    /// Creates a counter that has seen nothing yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            revisions: 0,
            writes: 0,
        }
    }

    /// The number of writes that were seen.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }

    /// The sum of the revisions of the observed states.
    #[must_use]
    pub const fn revisions(&self) -> u64 {
        self.revisions
    }

    /// Records the revision of a state that was read.
    pub fn observe(&mut self, revision: u64) {
        self.revisions = self.revisions.wrapping_add(revision);
    }

    /// Records that a state was written.
    pub fn record_write(&mut self) {
        self.writes = self.writes.wrapping_add(1);
    }

    /// Whether any state was written since the last call.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.writes > 0
    }

    /// Forgets the writes seen so far.
    pub fn clear(&mut self) {
        self.writes = 0;
    }
}

/// A cell holding a value that is read often and written rarely.
///
/// The value is stored behind a [`Cell`], so reading it costs nothing
/// and writing it cannot fail.
#[derive(Debug, PartialEq, Eq)]
pub struct ValueCell<T: Copy> {
    /// The stored value.
    value: Cell<T>,
}

impl<T: Copy> Clone for ValueCell<T> {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl<T: Copy> ValueCell<T> {
    /// Creates a cell holding `value`.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self {
            value: Cell::new(value),
        }
    }

    /// The stored value.
    #[must_use]
    pub fn get(&self) -> T {
        self.value.get()
    }

    /// Replaces the stored value.
    pub fn set(&self, value: T) {
        self.value.set(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_state_starts_clean_and_becomes_dirty_on_a_write() {
        let state = MutableState::new(0_u32);
        assert!(!state.is_dirty());
        state.set(1);
        assert!(state.is_dirty());
        assert_eq!(state.get(), 1);
    }

    #[test]
    fn a_write_bumps_the_revision() {
        let state = MutableState::new(0_u32);
        let before = state.revision();
        state.set(1);
        state.set(2);
        assert_eq!(state.revision(), before + 2);
    }

    #[test]
    fn clones_of_a_state_share_the_value() {
        let state = MutableState::new(0_u32);
        let clone = state.clone();
        clone.set(7);
        assert_eq!(state.get(), 7);
    }

    #[test]
    fn a_read_only_handle_sees_the_writes_of_the_mutable_one() {
        let state = MutableState::new(String::from("a"));
        let read_only = state.as_state();
        state.set(String::from("b"));
        assert_eq!(read_only.get(), String::from("b"));
        assert!(read_only.is_dirty());
    }

    #[test]
    fn taking_the_dirty_flag_clears_it() {
        let state = MutableState::new(0_u32);
        state.set(1);
        assert!(state.take_dirty());
        assert!(!state.take_dirty());
    }

    #[test]
    fn a_revision_counter_reports_the_writes_it_saw() {
        let mut counter = RevisionCounter::new();
        assert!(!counter.is_dirty());
        counter.record_write();
        assert!(counter.is_dirty());
        assert_eq!(counter.writes(), 1);
        counter.clear();
        assert!(!counter.is_dirty());
    }

    #[test]
    fn a_value_cell_stores_a_copyable_value() {
        let cell = ValueCell::new(3_u32);
        assert_eq!(cell.get(), 3);
        cell.set(4);
        assert_eq!(cell.get(), 4);
    }
}
