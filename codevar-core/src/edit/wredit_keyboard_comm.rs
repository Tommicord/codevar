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

//! Keyboard communication mediator for Android Bluetooth keyboard input.
//!
//! This module provides the `WritableKeyboardComm` struct which acts as a mediator
//! between the Android Bluetooth keyboard input system and the Writable data structure.
//! It translates keyboard events into appropriate actions using the WritableEventHandler.

use super::wredit_event::{WritableAction, WritableEventHandler};
use super::{StreamWritable, Writable};
#[cfg(target_os = "android")]
use crate::android::bluetooth_keyboard::{KeyAction, KeyEvent};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// Keys that should be ignored (volume, power, camera, etc.)
#[cfg(target_os = "android")]
const IGNORED_KEYS: [i32; 5] = [
    crate::android::bluetooth_keyboard::keycode::VOLUME_UP,
    crate::android::bluetooth_keyboard::keycode::VOLUME_DOWN,
    crate::android::bluetooth_keyboard::keycode::POWER,
    crate::android::bluetooth_keyboard::keycode::CAMERA,
    crate::android::bluetooth_keyboard::keycode::CLEAR,
];

/// Keys that represent printable characters (A-Z, 0-9, space, punctuation)
#[cfg(target_os = "android")]
const PRINTABLE_KEYS_START: i32 = crate::android::bluetooth_keyboard::keycode::A;
#[cfg(target_os = "android")]
const PRINTABLE_KEYS_END: i32 = crate::android::bluetooth_keyboard::keycode::Z;
#[cfg(target_os = "android")]
const DIGIT_KEYS_START: i32 = 7;
#[cfg(target_os = "android")]
const DIGIT_KEYS_END: i32 = 16;

/// Keyboard communication mediator for Android Bluetooth keyboard input.
///
/// This struct acts as a mediator between the Android Bluetooth keyboard input
/// system and the Writable data structure. It translates keyboard events into
/// appropriate actions using the WritableEventHandler, handling key codes,
/// modifiers, and history integration.
pub struct WritableKeyboardComm<W, Raw, Buf, const GAP_SIZE: usize>
where
    W: Writable<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditablePut<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditableDelete<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditableCursor<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditableHistory<Raw, Buf, GAP_SIZE>,
{
    /// The event handler that processes actions on the writable
    event_handler: WritableEventHandler<W, Raw, Buf, GAP_SIZE>,
    /// The cursor IDs to apply actions to
    cursor_ids: Vec<usize>,
    /// Timestamp of the last key press for edit group timing
    last_key_time: Option<Instant>,
    /// Timeout for grouping consecutive edits (in milliseconds)
    group_timeout_ms: u64,
    /// Caps lock state
    caps_lock: bool,
    /// Marker for Raw type parameter
    _phantom_raw: PhantomData<Raw>,
    /// Marker for Buf type parameter
    _phantom_buf: PhantomData<Buf>,
}

impl<W, Raw, Buf, const GAP_SIZE: usize> WritableKeyboardComm<W, Raw, Buf, GAP_SIZE>
where
    W: Writable<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditablePut<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditableDelete<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditableCursor<Raw, Buf, GAP_SIZE>
        + super::wredit_textedit_trait::TextEditableHistory<Raw, Buf, GAP_SIZE>,
{
    /// Creates a new keyboard communication mediator.
    ///
    /// # Arguments
    ///
    /// * `writable` - The writable instance to operate on.
    /// * `cursor_ids` - The cursor IDs to apply actions to.
    /// * `group_timeout_ms` - The timeout in milliseconds for grouping consecutive edits.
    ///
    /// # Returns
    ///
    /// A new WritableKeyboardComm instance.
    pub fn new(writable: W, cursor_ids: Vec<usize>, group_timeout_ms: u64) -> Self {
        Self {
            event_handler: WritableEventHandler::new(writable, group_timeout_ms),
            cursor_ids,
            last_key_time: None,
            group_timeout_ms,
            caps_lock: false,
            _phantom_raw: PhantomData,
            _phantom_buf: PhantomData,
        }
    }

    /// Processes a keyboard event from the Android Bluetooth keyboard.
    ///
    /// This method translates the keyboard event into an appropriate action
    /// and processes it through the event handler. It handles key codes,
    /// modifiers, and history integration.
    ///
    /// # Arguments
    ///
    /// * `event` - The keyboard event to process.
    ///
    /// # Safety
    ///
    /// This method performs unsafe operations on the writable instance.
    #[cfg(target_os = "android")]
    pub unsafe fn process_key_event(&mut self, event: KeyEvent) {
        if IGNORED_KEYS.contains(&event.key_code) {
            return;
        }
        // Only process key press events
        if !event.is_press() {
            return;
        }

        // Handle modifier key state changes
        self.handle_modifier_keys(&event);

        // Check if we need to end the edit group due to timeout
        self.check_edit_group_timeout();

        // Translate the key event to an action
        let action = self.translate_key_event(&event);

        // Process the action
        self.event_handler.process_action(action, &self.cursor_ids);

        // Update last key time
        self.last_key_time = Some(Instant::now());
    }

    /// Handles modifier key state changes.
    ///
    /// # Arguments
    ///
    /// * `event` - The keyboard event to process.
    #[cfg(target_os = "android")]
    fn handle_modifier_keys(&mut self, event: &KeyEvent) {
        match event.key_code {
            crate::android::bluetooth_keyboard::keycode::CAPS_LOCK
                if event.is_press() =>
            {
                self.caps_lock = !self.caps_lock;
            }
            _ => {}
        }
    }

    /// Translates a keyboard event into a writable action.
    ///
    /// # Arguments
    ///
    /// * `event` - The keyboard event to translate.
    ///
    /// # Returns
    ///
    /// The corresponding WritableAction.
    #[cfg(target_os = "android")]
    fn translate_key_event(&self, event: &KeyEvent) -> WritableAction {
        // Handle Ctrl key combinations
        if event.has_ctrl() {
            return self.handle_ctrl_combinations(event);
        }

        // Handle Alt key combinations
        if event.has_alt() {
            return self.handle_alt_combinations(event);
        }

        // Handle special editing keys
        match event.key_code {
            crate::android::bluetooth_keyboard::keycode::ENTER => {
                WritableAction::InsertChar(0x0A)
            }
            crate::android::bluetooth_keyboard::keycode::BACK => WritableAction::None,
            crate::android::bluetooth_keyboard::keycode::DEL => WritableAction::Delete,
            crate::android::bluetooth_keyboard::keycode::TAB => {
                WritableAction::InsertChar(0x09)
            }
            crate::android::bluetooth_keyboard::keycode::ESCAPE => WritableAction::None,
            crate::android::bluetooth_keyboard::keycode::MENU => WritableAction::None,
            _ => self.translate_printable_character(event),
        }
    }

    /// Handles Ctrl key combinations.
    ///
    /// # Arguments
    ///
    /// * `event` - The keyboard event to process.
    ///
    /// # Returns
    ///
    /// The corresponding WritableAction.
    #[cfg(target_os = "android")]
    fn handle_ctrl_combinations(&self, event: &KeyEvent) -> WritableAction {
        match event.key_code {
            crate::android::bluetooth_keyboard::keycode::Z => WritableAction::Undo,
            crate::android::bluetooth_keyboard::keycode::Y => WritableAction::Redo,
            crate::android::bluetooth_keyboard::keycode::A => WritableAction::SelectAll,
            crate::android::bluetooth_keyboard::keycode::C => WritableAction::Copy,
            crate::android::bluetooth_keyboard::keycode::V => WritableAction::Paste,
            crate::android::bluetooth_keyboard::keycode::X => WritableAction::Cut,
            _ => WritableAction::None,
        }
    }

    /// Handles Alt key combinations.
    ///
    /// # Arguments
    ///
    /// * `event` - The keyboard event to process.
    ///
    /// # Returns
    ///
    /// The corresponding WritableAction.
    #[cfg(target_os = "android")]
    fn handle_alt_combinations(&self, event: &KeyEvent) -> WritableAction {
        match event.key_code {
            crate::android::bluetooth_keyboard::keycode::DPAD_RIGHT => {
                WritableAction::MoveWordForward
            }
            crate::android::bluetooth_keyboard::keycode::DPAD_LEFT => {
                WritableAction::MoveWordBackward
            }
            _ => WritableAction::None,
        }
    }

    /// Translates a printable character key event.
    ///
    /// # Arguments
    ///
    /// * `event` - The keyboard event to translate.
    ///
    /// # Returns
    ///
    /// The corresponding WritableAction with the character code.
    #[cfg(target_os = "android")]
    fn translate_printable_character(&self, event: &KeyEvent) -> WritableAction {
        let key_code = event.key_code;
        let has_shift = event.has_shift();
        let caps_lock = self.caps_lock;

        // Handle letters (A-Z)
        if key_code >= PRINTABLE_KEYS_START && key_code <= PRINTABLE_KEYS_END {
            let base_char = (key_code - PRINTABLE_KEYS_START + 0x41) as u32; // 'A' = 0x41
            let is_upper = has_shift ^ caps_lock;
            let char_code = if is_upper {
                base_char
            } else {
                base_char + 0x20
            };
            return WritableAction::InsertChar(char_code);
        }
        // Handle digits (0-9) - Android key codes 7-16
        if key_code >= DIGIT_KEYS_START && key_code <= DIGIT_KEYS_END {
            let base_char = (key_code - DIGIT_KEYS_START + 0x30) as u32; // '0' = 0x30
            if has_shift {
                // Handle shifted digits (symbols)
                return self.translate_shifted_digit(key_code);
            }
            return WritableAction::InsertChar(base_char);
        }
        if key_code == crate::android::bluetooth_keyboard::keycode::SPACE {
            return WritableAction::InsertChar(0x20);
        }

        // Handle punctuation
        match key_code {
            crate::android::bluetooth_keyboard::keycode::COMMA => {
                if has_shift {
                    WritableAction::InsertChar(0x3C) // '<'
                } else {
                    WritableAction::InsertChar(0x2C) // ','
                }
            }
            crate::android::bluetooth_keyboard::keycode::PERIOD => {
                if has_shift {
                    WritableAction::InsertChar(0x3E) // '>'
                } else {
                    WritableAction::InsertChar(0x2E) // '.'
                }
            }
            _ => WritableAction::None,
        }
    }

    /// Translates shifted digit keys to their corresponding symbols.
    ///
    /// # Arguments
    ///
    /// * `key_code` - The digit key code.
    ///
    /// # Returns
    ///
    /// The corresponding WritableAction with the symbol character code.
    #[cfg(target_os = "android")]
    fn translate_shifted_digit(&self, key_code: i32) -> WritableAction {
        // Android key codes 7-16 map to 0-9
        // Shifted digits map to symbols: )!@#$%^&*(
        match key_code - DIGIT_KEYS_START {
            0 => WritableAction::InsertChar(0x29), // ')'
            1 => WritableAction::InsertChar(0x21), // '!'
            2 => WritableAction::InsertChar(0x40), // '@'
            3 => WritableAction::InsertChar(0x23), // '#'
            4 => WritableAction::InsertChar(0x24), // '$'
            5 => WritableAction::InsertChar(0x25), // '%'
            6 => WritableAction::InsertChar(0x5E), // '^'
            7 => WritableAction::InsertChar(0x26), // '&'
            8 => WritableAction::InsertChar(0x2A), // '*'
            9 => WritableAction::InsertChar(0x28), // '('
            _ => WritableAction::None,
        }
    }

    /// Checks if the edit group timeout has been exceeded and ends the group if so.
    fn check_edit_group_timeout(&mut self) {
        if let Some(last_time) = self.last_key_time {
            if last_time.elapsed() > Duration::from_millis(self.group_timeout_ms) {
                self.event_handler.end_edit_group();
            }
        }
    }

    /// Manually ends the current edit group.
    ///
    /// This can be called explicitly to finalize an edit group before the timeout.
    pub fn end_edit_group(&mut self) {
        self.event_handler.end_edit_group();
        self.last_key_time = None;
    }

    /// Returns a reference to the event handler.
    ///
    /// # Returns
    ///
    /// A reference to the event handler.
    pub fn event_handler(&self) -> &WritableEventHandler<W, Raw, Buf, GAP_SIZE> {
        &self.event_handler
    }

    /// Returns a mutable reference to the event handler.
    ///
    /// # Returns
    ///
    /// A mutable reference to the event handler.
    pub fn event_handler_mut(
        &mut self,
    ) -> &mut WritableEventHandler<W, Raw, Buf, GAP_SIZE> {
        &mut self.event_handler
    }

    /// Returns the cursor IDs.
    ///
    /// # Returns
    ///
    /// A slice of the cursor IDs.
    pub fn cursor_ids(&self) -> &[usize] {
        &self.cursor_ids
    }

    /// Sets the cursor IDs.
    ///
    /// # Arguments
    ///
    /// * `cursor_ids` - The new cursor IDs.
    pub fn set_cursor_ids(&mut self, cursor_ids: Vec<usize>) {
        self.cursor_ids = cursor_ids;
    }

    /// Returns the group timeout in milliseconds.
    ///
    /// # Returns
    ///
    /// The group timeout duration in milliseconds.
    pub fn group_timeout_ms(&self) -> u64 {
        self.group_timeout_ms
    }

    /// Sets the group timeout in milliseconds.
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - The new timeout in milliseconds.
    pub fn set_group_timeout_ms(&mut self, timeout_ms: u64) {
        self.group_timeout_ms = timeout_ms;
        self.event_handler
            .set_group_timeout(Duration::from_millis(timeout_ms));
    }

    /// Returns the caps lock state.
    ///
    /// # Returns
    ///
    /// `true` if caps lock is on, `false` otherwise.
    pub fn caps_lock(&self) -> bool {
        self.caps_lock
    }

    /// Sets the caps lock state.
    ///
    /// # Arguments
    ///
    /// * `caps_lock` - The new caps lock state.
    pub fn set_caps_lock(&mut self, caps_lock: bool) {
        self.caps_lock = caps_lock;
    }
}

#[cfg(test)]
mod tests {
    use super::{StreamWritable, WritableKeyboardComm};
    use std::time::Duration;

    fn keyboard_comm_creation() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.group_timeout_ms(), 1000);
        assert_eq!(comm.cursor_ids(), &[0]);
        assert!(!comm.caps_lock());
    }

    fn keyboard_comm_custom_timeout() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 500);
        assert_eq!(comm.group_timeout_ms(), 500);
    }

    fn keyboard_comm_multiple_cursors() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0, 1, 2];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids(), &[0, 1, 2]);
    }

    fn keyboard_comm_empty_cursors() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids: Vec<usize> = vec![];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids(), &[]);
    }

    fn set_cursor_ids() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_cursor_ids(vec![0, 1, 2]);
        assert_eq!(comm.cursor_ids(), &[0, 1, 2]);
    }

    fn set_cursor_ids_empty() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0, 1];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_cursor_ids(vec![]);
        assert_eq!(comm.cursor_ids(), &[]);
    }

    fn caps_lock_initial_state() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert!(!comm.caps_lock());
    }

    fn caps_lock_toggle_on() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert!(!comm.caps_lock());
        comm.set_caps_lock(true);
        assert!(comm.caps_lock());
    }

    fn caps_lock_toggle_off() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_caps_lock(true);
        assert!(comm.caps_lock());
        comm.set_caps_lock(false);
        assert!(!comm.caps_lock());
    }

    fn group_timeout_initial() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.group_timeout_ms(), 1000);
    }

    fn set_group_timeout() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.group_timeout_ms(), 1000);
        comm.set_group_timeout_ms(500);
        assert_eq!(comm.group_timeout_ms(), 500);
    }

    fn set_group_timeout_zero() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_group_timeout_ms(0);
        assert_eq!(comm.group_timeout_ms(), 0);
    }

    fn end_edit_group() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.end_edit_group();
        assert!(!comm.event_handler().is_group_active());
    }

    fn event_handler_access() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        let _handler = comm.event_handler();
    }

    fn event_handler_mut_access() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        let _handler = comm.event_handler_mut();
    }

    fn writable_access_through_handler() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        let _writable = comm.event_handler().writable();
    }

    fn writable_mut_access_through_handler() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        let _writable = comm.event_handler_mut().writable_mut();
    }

    fn event_handler_timeout_sync() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_group_timeout_ms(2000);
        assert_eq!(
            comm.event_handler().group_timeout(),
            Duration::from_millis(2000)
        );
    }

    fn cursor_ids_slice() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![1, 2, 3, 4, 5];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids().len(), 5);
        assert_eq!(comm.cursor_ids()[0], 1);
        assert_eq!(comm.cursor_ids()[4], 5);
    }

    fn large_cursor_ids() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids: Vec<usize> = (0..100).collect();
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids().len(), 100);
    }

    fn single_cursor_id() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![42];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids(), &[42]);
    }

    fn duplicate_cursor_ids() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0, 0, 1, 1, 2];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids(), &[0, 0, 1, 1, 2]);
    }

    fn end_edit_group_multiple_times() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.end_edit_group();
        comm.end_edit_group();
        comm.end_edit_group();
        assert!(!comm.event_handler().is_group_active());
    }

    fn caps_lock_persistence() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_caps_lock(true);
        comm.set_group_timeout_ms(500);
        assert!(comm.caps_lock());
        assert_eq!(comm.group_timeout_ms(), 500);
    }

    fn cursor_ids_update_after_creation() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids(), &[0]);
        comm.set_cursor_ids(vec![5, 10, 15]);
        assert_eq!(comm.cursor_ids(), &[5, 10, 15]);
    }

    fn event_handler_initial_state() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert!(!comm.event_handler().is_group_active());
    }

    fn group_timeout_large_value() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![0];
        let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        comm.set_group_timeout_ms(10000);
        assert_eq!(comm.group_timeout_ms(), 10000);
    }

    fn cursor_ids_with_large_values() {
        let writable = StreamWritable::<u32, u8, 4096>::new();
        let cursor_ids = vec![1000, 2000, 3000];
        let comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);
        assert_eq!(comm.cursor_ids(), &[1000, 2000, 3000]);
    }

    #[cfg(target_os = "android")]

    mod android {
        use super::*;
        use crate::android::bluetooth_keyboard::{KeyAction, KeyEvent, keycode};
        use crate::edit::Writable;

        fn process_key_event_ignored_volume_up() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::VOLUME_UP, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            // Should not start edit group for ignored keys
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ignored_volume_down() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::VOLUME_DOWN, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ignored_power() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::POWER, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ignored_camera() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::CAMERA, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_key_release_ignored() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::A, KeyAction::Release, 0);
            unsafe {
                comm.process_key_event(event);
            }
            // Key release should not start edit group
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_enter() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::ENTER, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            // Enter should start edit group
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_space() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::SPACE, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_delete() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::DEL, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_tab() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::TAB, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_back_ignored() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::BACK, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            // Back key should be ignored (navigation key)
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_escape_ignored() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::ESCAPE, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_caps_lock_toggle() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            assert!(!comm.caps_lock());

            let event = KeyEvent::new(keycode::CAPS_LOCK, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }

            assert!(comm.caps_lock());
        }

        fn process_key_event_ctrl_z_undo() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            // Start an edit group first
            unsafe {
                comm.event_handler_mut().begin_edit_group();
            }
            assert!(comm.event_handler().is_group_active());

            let event = KeyEvent::new(keycode::Z, KeyAction::Press, 0x04); // Ctrl
            unsafe {
                comm.process_key_event(event);
            }

            // Undo should end edit group
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ctrl_y_redo() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::Y, KeyAction::Press, 0x04); // Ctrl
            unsafe {
                comm.process_key_event(event);
            }

            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_letter_a() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::A, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_letter_a_with_shift() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::A, KeyAction::Press, 0x01); // Shift
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_letter_a_with_caps_lock() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            comm.set_caps_lock(true);

            let event = KeyEvent::new(keycode::A, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_comma() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::COMMA, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_period() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::PERIOD, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_comma_with_shift() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::COMMA, KeyAction::Press, 0x01); // Shift
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_period_with_shift() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::PERIOD, KeyAction::Press, 0x01); // Shift
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_ctrl_a_none() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::A, KeyAction::Press, 0x04); // Ctrl
            unsafe {
                comm.process_key_event(event);
            }
            // Ctrl+A is not implemented yet, should return None
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ctrl_c_none() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::C, KeyAction::Press, 0x04); // Ctrl
            unsafe {
                comm.process_key_event(event);
            }
            // Ctrl+C is not implemented yet, should return None
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ctrl_v_none() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::V, KeyAction::Press, 0x04); // Ctrl
            unsafe {
                comm.process_key_event(event);
            }
            // Ctrl+V is not implemented yet, should return None
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_ctrl_x_none() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::X, KeyAction::Press, 0x04); // Ctrl
            unsafe {
                comm.process_key_event(event);
            }
            // Ctrl+X is not implemented yet, should return None
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_alt_combination_none() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::A, KeyAction::Press, 0x02); // Alt
            unsafe {
                comm.process_key_event(event);
            }
            // Alt combinations not implemented yet
            assert!(!comm.event_handler().is_group_active());
        }

        fn process_key_event_multiple_cursors() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id1 = writable.append_cursor();
            let cursor_id2 = writable.append_cursor();
            let cursor_ids = vec![cursor_id1, cursor_id2];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::A, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_consecutive_edits() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event1 = KeyEvent::new(keycode::A, KeyAction::Press, 0);
            let event2 = KeyEvent::new(keycode::B, KeyAction::Press, 0);
            let event3 = KeyEvent::new(keycode::C, KeyAction::Press, 0);

            unsafe {
                comm.process_key_event(event1);
                comm.process_key_event(event2);
                comm.process_key_event(event3);
            }

            assert!(comm.event_handler().is_group_active());
        }

        fn process_key_event_menu_ignored() {
            let writable = StreamWritable::<u32, u8, 4096>::new();
            let cursor_id = writable.append_cursor();
            let cursor_ids = vec![cursor_id];
            let mut comm = WritableKeyboardComm::new(writable, cursor_ids, 1000);

            let event = KeyEvent::new(keycode::MENU, KeyAction::Press, 0);
            unsafe {
                comm.process_key_event(event);
            }
            assert!(!comm.event_handler().is_group_active());
        }
    }
}
