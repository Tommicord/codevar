//! Copyright 2026 Codevar
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

//! Event handler for processing keyboard events into Writable actions.
//!
//! This module provides the `WritableEventHandler` struct which receives
//! keyboard events from the mediator and translates them into appropriate
//! actions on the Writable data structure, such as insertions, deletions,
//! cursor movements, and undo/redo operations.

use super::Writable;
use super::wredit_observer::{UserActionEvent, UserActionType};
use super::wredit_textedit_trait::{
    TextEditableCursor, TextEditableDelete, TextEditableHistory, TextEditablePut,
};
use std::marker::PhantomData;
use std::time::Duration;

/// Represents the type of action to be performed on the writable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritableAction {
    /// Insert a character
    InsertChar(u32),
    /// Delete character at cursor (forward)
    Delete,
    /// Delete character before cursor (backward)
    Backspace,
    /// Move cursor forward
    MoveForward(usize),
    /// Move cursor backward
    MoveBackward(usize),
    /// Move cursor to next line
    MoveNextLine,
    /// Move cursor to previous line
    MovePrevLine,
    /// Move cursor to line start
    MoveLineStart,
    /// Move cursor to line end
    MoveLineEnd,
    /// Move cursor forward by word
    MoveWordForward,
    /// Move cursor backward by word
    MoveWordBackward,
    /// Select all text
    SelectAll,
    /// Copy selected text to clipboard
    Copy,
    /// Paste text from clipboard
    Paste,
    /// Cut selected text to clipboard
    Cut,
    /// Undo last operation
    Undo,
    /// Redo last undone operation
    Redo,
    /// Begin edit group for history
    BeginEditGroup,
    /// End edit group for history
    EndEditGroup,
    /// No action (ignore event)
    None,
}

/// Event handler for processing keyboard events into Writable actions.
///
/// This struct is generic over the Writable type and handles the translation
/// of keyboard events into appropriate editing operations. It manages
/// cursor validation and applies actions to multiple cursors when needed.
pub struct WritableEventHandler<W, Raw, Buf, const GAP_SIZE: usize>
where
    W: Writable<Raw, Buf, GAP_SIZE>
        + TextEditablePut<Raw, Buf, GAP_SIZE>
        + TextEditableDelete<Raw, Buf, GAP_SIZE>
        + TextEditableCursor<Raw, Buf, GAP_SIZE>
        + TextEditableHistory<Raw, Buf, GAP_SIZE>,
{
    /// The writable instance to operate on
    writable: W,
    /// The timeout duration for grouping consecutive edits (in milliseconds)
    group_timeout: Duration,
    /// Whether an edit group is currently active
    group_active: bool,
    /// Marker for Raw type parameter
    _phantom_raw: PhantomData<Raw>,
    /// Marker for Buf type parameter
    _phantom_buf: PhantomData<Buf>,
}

impl<W, Raw, Buf, const GAP_SIZE: usize> WritableEventHandler<W, Raw, Buf, GAP_SIZE>
where
    W: Writable<Raw, Buf, GAP_SIZE>
        + TextEditablePut<Raw, Buf, GAP_SIZE>
        + TextEditableDelete<Raw, Buf, GAP_SIZE>
        + TextEditableCursor<Raw, Buf, GAP_SIZE>
        + TextEditableHistory<Raw, Buf, GAP_SIZE>,
{
    /// Creates a new event handler with the specified writable instance.
    ///
    /// # Arguments
    ///
    /// * `writable` - The writable instance to operate on.
    /// * `group_timeout_ms` - The timeout in milliseconds for grouping consecutive edits.
    ///
    /// # Returns
    ///
    /// A new WritableEventHandler instance.
    pub fn new(writable: W, group_timeout_ms: u64) -> Self {
        Self {
            writable,
            group_timeout: Duration::from_millis(group_timeout_ms),
            group_active: false,
            _phantom_raw: PhantomData,
            _phantom_buf: PhantomData,
        }
    }

    /// Processes a writable action on the specified cursor IDs.
    ///
    /// This method validates the cursor IDs and applies the action to each valid cursor.
    /// For actions that don't require cursor-specific handling (like undo/redo), it
    /// applies them once without cursor validation.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to perform.
    /// * `cursor_ids` - Slice of cursor IDs to apply the action to.
    ///
    /// # Safety
    ///
    /// This method performs unsafe operations on the writable instance.
    pub unsafe fn process_action(
        &mut self,
        action: WritableAction,
        cursor_ids: &[usize],
    ) {
        match action {
            WritableAction::InsertChar(ch) => {
                self.begin_edit_group();
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::InsertChar,
                            cursor_id,
                            Some(ch),
                            1,
                            0,
                        );
                        self.writable.put(ch, true, cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::Delete => {
                self.begin_edit_group();
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::Delete,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.delete(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::Backspace => {
                self.begin_edit_group();
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::Backspace,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.backspace(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveForward(count) => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveForward,
                            cursor_id,
                            None,
                            count,
                            0,
                        );
                        self.writable.move_cursor_forward(cursor_id, count);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveBackward(count) => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveBackward,
                            cursor_id,
                            None,
                            count,
                            0,
                        );
                        self.writable.move_cursor_backward(cursor_id, count);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveNextLine => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveNextLine,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.move_cursor_next_line(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MovePrevLine => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MovePrevLine,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.move_cursor_prev_line(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveLineStart => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveLineStart,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.move_cursor_line_start(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveLineEnd => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveLineEnd,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.move_cursor_line_end(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveWordForward => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveWordForward,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.move_cursor_word_forward(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::MoveWordBackward => {
                for &cursor_id in cursor_ids {
                    if self.is_cursor_valid(cursor_id) {
                        let event = UserActionEvent::new(
                            UserActionType::MoveWordBackward,
                            cursor_id,
                            None,
                            1,
                            0,
                        );
                        self.writable.move_cursor_word_backward(cursor_id);
                        self.notify_action_observers(&event);
                    }
                }
            }
            WritableAction::SelectAll => {
                let event =
                    UserActionEvent::new(UserActionType::SelectAll, 0, None, 1, 0);
                self.writable.select_all();
                self.notify_action_observers(&event);
            }
            WritableAction::Copy => {
                let event = UserActionEvent::new(UserActionType::Copy, 0, None, 1, 0);
                self.writable.copy_selection();
                self.notify_action_observers(&event);
            }
            WritableAction::Paste => {
                self.begin_edit_group();
                let event = UserActionEvent::new(UserActionType::Paste, 0, None, 1, 0);
                self.writable.paste_from_clipboard();
                self.notify_action_observers(&event);
            }
            WritableAction::Cut => {
                self.begin_edit_group();
                let event = UserActionEvent::new(UserActionType::Cut, 0, None, 1, 0);
                self.writable.cut_selection();
                self.notify_action_observers(&event);
            }
            WritableAction::Undo => {
                self.end_edit_group();
                let event = UserActionEvent::new(UserActionType::Undo, 0, None, 1, 0);
                self.writable.undo();
                self.notify_action_observers(&event);
            }
            WritableAction::Redo => {
                let event = UserActionEvent::new(UserActionType::Redo, 0, None, 1, 0);
                self.writable.redo();
                self.notify_action_observers(&event);
            }
            WritableAction::BeginEditGroup => {
                if !self.group_active {
                    let event = UserActionEvent::new(
                        UserActionType::BeginEditGroup,
                        0,
                        None,
                        1,
                        0,
                    );
                    self.writable.begin_edit_group();
                    self.group_active = true;
                    self.notify_action_observers(&event);
                }
            }
            WritableAction::EndEditGroup => {
                if self.group_active {
                    let event =
                        UserActionEvent::new(UserActionType::EndEditGroup, 0, None, 1, 0);
                    self.writable.end_edit_group();
                    self.group_active = false;
                    self.notify_action_observers(&event);
                }
            }
            WritableAction::None => {}
        }
    }

    /// Notifies action observers of a user action event.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event to notify observers about
    fn notify_action_observers(&self, event: &UserActionEvent) {
        self.writable.notify_action_observers(event);
    }

    /// Checks if a cursor ID is valid.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID to validate.
    ///
    /// # Returns
    ///
    /// `true` if the cursor is valid, `false` otherwise.
    fn is_cursor_valid(&self, cursor_id: usize) -> bool {
        self.writable.cursor_from_id(cursor_id).is_some()
    }

    /// Ensures an edit group is active before performing edits.
    ///
    /// This method starts a new edit group if one is not already active.
    pub fn begin_edit_group(&mut self) {
        if !self.group_active {
            unsafe {
                self.writable.begin_edit_group();
            }
            self.group_active = true;
        }
    }

    /// Ends the current edit group if one is active.
    ///
    /// This should be called after a period of inactivity to finalize
    /// the edit group for history tracking.
    pub fn end_edit_group(&mut self) {
        if self.group_active {
            unsafe {
                self.writable.end_edit_group();
            }
            self.group_active = false;
        }
    }

    /// Returns a reference to the writable instance.
    ///
    /// # Returns
    ///
    /// A reference to the writable instance.
    pub fn writable(&self) -> &W {
        &self.writable
    }

    /// Returns a mutable reference to the writable instance.
    ///
    /// # Returns
    ///
    /// A mutable reference to the writable instance.
    pub fn writable_mut(&mut self) -> &mut W {
        &mut self.writable
    }

    /// Returns the group timeout duration.
    ///
    /// # Returns
    ///
    /// The duration before an edit group is automatically ended.
    pub fn group_timeout(&self) -> Duration {
        self.group_timeout
    }

    /// Sets the group timeout duration.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The new timeout duration.
    pub fn set_group_timeout(&mut self, timeout: Duration) {
        self.group_timeout = timeout;
    }

    /// Returns whether an edit group is currently active.
    ///
    /// # Returns
    ///
    /// `true` if an edit group is active, `false` otherwise.
    pub fn is_group_active(&self) -> bool {
        self.group_active
    }
}

#[cfg(test)]
mod tests {
    use crate::edit::{StreamWritable, Writable, WritableAction, WritableEventHandler};
    use std::time::Duration;

    fn event_handler_creation() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let handler = WritableEventHandler::new(writable, 1000);
        assert_eq!(handler.group_timeout(), Duration::from_millis(1000));
        assert!(!handler.is_group_active());
    }

    fn event_handler_custom_timeout() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let handler = WritableEventHandler::new(writable, 500);
        assert_eq!(handler.group_timeout(), Duration::from_millis(500));
    }

    fn none_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::None, &[]);
        }
        assert!(!handler.is_group_active());
    }

    fn insert_char_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::InsertChar(0x41), &[cursor_id]);
        }
        // Edit group should be started after insert
        assert!(handler.is_group_active());
    }

    fn delete_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::Delete, &[cursor_id]);
        }
        assert!(handler.is_group_active());
    }

    fn backspace_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::Backspace, &[cursor_id]);
        }
        assert!(handler.is_group_active());
    }

    fn move_cursor_forward_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::MoveForward(5), &[cursor_id]);
        }
        // Cursor movement should not start edit group
        assert!(!handler.is_group_active());
    }

    fn move_cursor_backward_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::MoveBackward(3), &[cursor_id]);
        }
        assert!(!handler.is_group_active());
    }

    fn move_next_line_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::MoveNextLine, &[cursor_id]);
        }
        assert!(!handler.is_group_active());
    }

    fn move_prev_line_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::MovePrevLine, &[cursor_id]);
        }
        assert!(!handler.is_group_active());
    }

    fn move_line_start_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::MoveLineStart, &[cursor_id]);
        }
        assert!(!handler.is_group_active());
    }

    fn move_line_end_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::MoveLineEnd, &[cursor_id]);
        }
        assert!(!handler.is_group_active());
    }

    fn undo_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::Undo, &[]);
        }
        // Undo should end edit group if active
        assert!(!handler.is_group_active());
    }

    fn redo_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::Redo, &[]);
        }
        assert!(!handler.is_group_active());
    }

    fn begin_edit_group_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::BeginEditGroup, &[]);
        }
        assert!(handler.is_group_active());
    }

    fn end_edit_group_action() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::BeginEditGroup, &[]);
            handler.process_action(WritableAction::EndEditGroup, &[]);
        }
        assert!(!handler.is_group_active());
    }

    fn multiple_cursor_ids() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id1 = writable.append_cursor();
        let cursor_id2 = writable.append_cursor();
        let cursor_id3 = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(
                WritableAction::InsertChar(0x42),
                &[cursor_id1, cursor_id2, cursor_id3],
            );
        }
        assert!(handler.is_group_active());
    }

    fn invalid_cursor_id() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            // Process with valid and invalid cursor IDs
            handler.process_action(WritableAction::InsertChar(0x43), &[cursor_id, 999]);
        }
        // Should still start edit group
        assert!(handler.is_group_active());
    }

    fn end_edit_group_method() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::InsertChar(0x44), &[cursor_id]);
        }
        assert!(handler.is_group_active());
        handler.end_edit_group();
        assert!(!handler.is_group_active());
    }

    fn ensure_edit_group() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        assert!(!handler.is_group_active());
        // Insert char should start edit group
        unsafe {
            handler.process_action(WritableAction::InsertChar(0x45), &[cursor_id]);
        }
        assert!(handler.is_group_active());
    }

    fn set_group_timeout() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        assert_eq!(handler.group_timeout(), Duration::from_millis(1000));
        handler.set_group_timeout(Duration::from_millis(2000));
        assert_eq!(handler.group_timeout(), Duration::from_millis(2000));
    }

    fn writable_access() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let handler = WritableEventHandler::new(writable, 1000);
        // Test that we can access the writable
        let _writable_ref = handler.writable();
        let mut handler_mut =
            WritableEventHandler::new(StreamWritable::<u32, u8, 4096>::new(), 1000);
        let _writable_mut = handler_mut.writable_mut();
    }

    fn consecutive_edits_same_group() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::InsertChar(0x46), &[cursor_id]);
            handler.process_action(WritableAction::InsertChar(0x47), &[cursor_id]);
            handler.process_action(WritableAction::InsertChar(0x48), &[cursor_id]);
        }
        // All consecutive inserts should stay in same group
        assert!(handler.is_group_active());
    }

    fn edit_group_persistence() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_id = writable.append_cursor();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::BeginEditGroup, &[]);
            handler.process_action(WritableAction::InsertChar(0x49), &[cursor_id]);
        }
        assert!(handler.is_group_active());
        // Cursor movement should not end the group
        unsafe {
            handler.process_action(WritableAction::MoveForward(1), &[cursor_id]);
        }
        assert!(handler.is_group_active());
    }

    fn empty_cursor_ids() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let mut handler = WritableEventHandler::new(writable, 1000);
        unsafe {
            handler.process_action(WritableAction::InsertChar(0x4A), &[]);
        }
        // Should still start edit group even with no cursors
        assert!(handler.is_group_active());
    }
}
