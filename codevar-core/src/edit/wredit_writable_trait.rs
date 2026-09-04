//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use super::wredit_observer::{UserActionEvent, UserActionObserver};
use super::{Cursor, EventBridge, SharedGap};
use std::cell::UnsafeCell;
use std::sync::Arc;

/// Trait for writable data structures with cursor and gap buffer support.
///
/// This trait provides a polymorphic interface for writable data structures that
/// support multiple cursors, gap buffers for efficient insertions, and shared gap
/// regions for cursor proximity management.
pub trait Writable<Raw, Buf, const GAP_SIZE: usize> {
    /// Creates a new writable instance with default capacity.
    ///
    /// # Returns
    ///
    /// A new writable instance initialized with default settings.
    fn new() -> Self
    where
        Self: Sized;

    /// Creates a new writable instance with the specified name and capacity.
    ///
    /// # Arguments
    ///
    /// * `name` - The name identifier for this writable instance (e.g., file name).
    ///            The name is truncated to 255 characters if longer.
    /// * `capacity` - The initial capacity for the raw data buffer. If 0, uses the
    ///               default capacity of 8192 elements.
    ///
    /// # Returns
    ///
    /// A new writable instance initialized with the specified name and capacity.
    fn with_capacity(name: &str, capacity: usize) -> Self
    where
        Self: Sized;

    /// Adds a new cursor to the writable instance.
    ///
    /// The new cursor is cloned from the currently active cursor if one exists,
    /// otherwise it starts at the origin (position 0, 0). The cursor is assigned
    /// the next available ID.
    ///
    /// # Returns
    ///
    /// The ID of the newly created cursor, or the last cursor ID if the maximum
    /// cursor count (512) has been reached.
    fn append_cursor(&self) -> usize;

    /// Removes a cursor from the writable instance.
    ///
    /// This removes the cursor with the specified ID and updates the cursor-to-gap
    /// mapping. If the removed cursor was the active cursor, the active cursor ID
    /// is reset to 0. If the removed cursor ID was less than the active cursor ID,
    /// the active cursor ID is decremented.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to remove.
    ///
    /// # Returns
    ///
    /// `true` if the cursor was successfully removed, `false` if the cursor ID
    /// was invalid.
    fn delete_cursor(&self, cursor_id: usize) -> bool;

    /// Returns an immutable reference to the cursor with the specified ID.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to retrieve. If the ID is out of bounds,
    ///                the first cursor is returned instead.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the cursor, or `None` if no cursor
    /// exists at the specified position.
    fn cursor_from_id(&self, cursor_id: usize) -> Option<&Cursor<Raw, Buf, GAP_SIZE>>;

    /// Returns a mutable reference to the cursor with the specified ID.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to retrieve. If the ID is out of bounds,
    ///                the first cursor is returned instead.
    ///
    /// # Returns
    ///
    /// An `Option` containing a mutable reference to the cursor, or `None` if no
    /// cursor exists at the specified position.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it returns a mutable reference that could
    /// violate Rust's borrowing rules if used incorrectly.
    unsafe fn cursor_from_id_mut(
        &self,
        cursor_id: usize,
    ) -> Option<&mut Cursor<Raw, Buf, GAP_SIZE>>;

    /// Returns the vertical space (number of lines) in the writable.
    ///
    /// This tracks the number of newline characters encountered, which is used
    /// for cursor positioning and display calculations.
    ///
    /// # Returns
    ///
    /// The number of lines (vertical space) in the writable.
    fn v_space(&self) -> usize;

    /// Returns a reference to the shared gaps container.
    ///
    /// Shared gaps are gap regions that are shared among multiple cursors in
    /// proximity to each other, allowing efficient concurrent writes.
    ///
    /// # Returns
    ///
    /// A reference to the `UnsafeCell` containing the vector of shared gaps.
    fn shared_gaps(&self) -> &UnsafeCell<Vec<SharedGap<Raw, GAP_SIZE>>>;

    /// Returns the ID of the currently active cursor.
    ///
    /// The active cursor is the cursor that receives write operations by default.
    ///
    /// # Returns
    ///
    /// The ID of the active cursor.
    fn active_cursor_id(&self) -> usize;

    /// Sets the active cursor ID.
    ///
    /// The active cursor is the cursor that receives write operations by default.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to set as active. If the ID is invalid
    ///                (greater than or equal to the number of cursors), this
    ///                operation has no effect.
    fn set_active_cursor(&self, cursor_id: usize);

    /// Returns an immutable reference to the active cursor.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the active cursor, or `None` if no
    /// active cursor exists.
    fn active_cursor(&self) -> Option<&Cursor<Raw, Buf, GAP_SIZE>>;

    /// Returns an immutable pointer to the raw data buffer.
    ///
    /// # Returns
    ///
    /// A const pointer to the raw data buffer.
    fn raw_ptr(&self) -> *const Raw;

    /// Returns a mutable pointer to the raw data buffer.
    ///
    /// # Returns
    ///
    /// A mutable pointer to the raw data buffer.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it returns a mutable pointer that could
    /// violate Rust's borrowing rules if used incorrectly.
    unsafe fn raw_mut_ptr(&self) -> *mut Raw;

    /// Returns the length of the raw data buffer.
    ///
    /// # Returns
    ///
    /// The number of elements currently stored in the raw data buffer.
    fn raw_length(&self) -> usize;

    /// Sets the event bridge for this writable instance.
    ///
    /// The event bridge is used for dependency injection and allows external
    /// components to receive notifications about writable events.
    ///
    /// # Arguments
    ///
    /// * `bridge` - An `Arc` to an object implementing the `EventBridge` trait.
    ///              This bridge will receive notifications about cursor movements,
    ///              insertions, deletions, and other events.
    fn set_event_bridge(&self, bridge: Arc<dyn EventBridge>);

    /// Returns the event bridge for this writable instance.
    ///
    /// # Returns
    ///
    /// An `Option` containing an `Arc` to the event bridge, or `None` if no
    /// event bridge has been set.
    fn event_bridge(&self) -> Option<Arc<dyn EventBridge>>;

    /// Returns the name of this writable instance.
    ///
    /// # Returns
    ///
    /// A byte slice containing the name of the writable instance.
    fn name(&self) -> &[u8];

    /// Returns a mutable reference to the name of this writable instance.
    ///
    /// # Returns
    ///
    /// A mutable byte slice containing the name of the writable instance.
    fn name_mut(&mut self) -> &mut [u8];

    /// Returns the flags for this writable instance.
    ///
    /// Flags are used to store various state information about the writable.
    ///
    /// # Returns
    ///
    /// The flags value as a 32-bit unsigned integer.
    fn flags(&self) -> u32;

    /// Returns a mutable reference to the flags for this writable instance.
    ///
    /// # Returns
    ///
    /// A mutable reference to the flags value.
    fn flags_mut(&mut self) -> &mut u32;

    /// Checks if two cursors are within GAP_SIZE range of each other.
    ///
    /// This method calculates the absolute difference between the cursor positions
    /// and checks if it is within the GAP_SIZE threshold. Cursors in proximity can
    /// share a gap region for efficient concurrent writes.
    ///
    /// # Arguments
    ///
    /// * `cursor_id1` - The ID of the first cursor.
    /// * `cursor_id2` - The ID of the second cursor.
    ///
    /// # Returns
    ///
    /// `true` if the cursors are within GAP_SIZE range of each other, `false` otherwise.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it accesses cursor positions through unsafe
    /// interior mutability.
    unsafe fn proximity(&self, cursor_id1: usize, cursor_id2: usize) -> bool;

    /// Creates a new shared gap or adds the cursor to an existing shared gap.
    ///
    /// This method checks if the specified cursor is already in a shared gap.
    /// If not, it checks if the cursor is in proximity with any cursor that has
    /// a shared gap and adds it to that gap. If no existing shared gap is found,
    /// it creates a new shared gap with any nearby cursors.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to add to a shared gap.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it mutates shared gap structures through
    /// unsafe interior mutability.
    unsafe fn new_shared_gap(&self, cursor_id: usize);

    /// Returns the shared gap index for the specified cursor.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor.
    ///
    /// # Returns
    ///
    /// The index of the shared gap containing the cursor, or `usize::MAX` if
    /// the cursor is not in a shared gap.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it accesses the cursor-to-gap mapping through
    /// unsafe interior mutability.
    unsafe fn shared_gap_index(&self, cursor_id: usize) -> usize;

    /// Checks if a cursor is in a shared gap.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to check.
    ///
    /// # Returns
    ///
    /// `true` if the cursor is in a shared gap, `false` otherwise.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it accesses the cursor-to-gap mapping through
    /// unsafe interior mutability.
    unsafe fn is_cursor_in_shared_gap(&self, cursor_id: usize) -> bool;

    /// Returns a reference to the history for this writable instance.
    ///
    /// The history tracks all changes made to the buffer, enabling undo/redo
    /// functionality and change tracking.
    ///
    /// # Returns
    ///
    /// A reference to the `UnsafeCell` containing the history.
    fn history(&self) -> &UnsafeCell<crate::edit::wredit_history::History>;

    /// Returns the current history index.
    ///
    /// This indicates the current position in the history timeline.
    ///
    /// # Returns
    ///
    /// The current history index.
    fn history_current_index(&self) -> usize;

    /// Sets the current history index.
    ///
    /// This moves the history pointer to the specified position.
    ///
    /// # Arguments
    ///
    /// * `index` - The new history index.
    fn set_history_current_index(&self, index: usize);

    /// Records a change in the history.
    ///
    /// This method should be called whenever a modification is made to the buffer
    /// to track the change for undo/redo functionality.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it mutates the history through unsafe interior mutability.
    unsafe fn record_change(&self);

    /// Registers an observer for user actions performed on this writable.
    fn register_action_observer(&self, observer: Arc<dyn UserActionObserver>);

    /// Notifies registered observers of a user action.
    fn notify_action_observers(&self, event: &UserActionEvent);

    /// Selects all text in the writable buffer.
    ///
    /// This method sets the selection range to cover the entire content of the buffer.
    ///
    /// # Safety
    ///
    /// This method performs unsafe operations on the writable instance.
    unsafe fn select_all(&mut self);

    /// Copies the current selection to the clipboard.
    ///
    /// This method copies the currently selected text to the system clipboard.
    ///
    /// # Safety
    ///
    /// This method performs unsafe operations on the writable instance.
    unsafe fn copy_selection(&mut self);

    /// Pastes content from the clipboard at the current cursor position.
    ///
    /// This method inserts the clipboard content at the current cursor position.
    ///
    /// # Safety
    ///
    /// This method performs unsafe operations on the writable instance.
    unsafe fn paste_from_clipboard(&mut self);

    /// Cuts the current selection to the clipboard.
    ///
    /// This method removes the currently selected text and copies it to the system clipboard.
    ///
    /// # Safety
    ///
    /// This method performs unsafe operations on the writable instance.
    unsafe fn cut_selection(&mut self);
}
