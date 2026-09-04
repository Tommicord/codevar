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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Keyboard input event types for Android Bluetooth HID devices.
//!
//! This module provides types for representing keyboard input events from
//! Bluetooth HID devices. It includes the `KeyEvent` struct which encapsulates
//! key codes, actions (press/release), and modifier states.
//!
//! # Android Key Codes
//!
//! Key codes follow the Android KeyEvent constants. Common examples:
//! - `KEYCODE_A` = 29
//! - `KEYCODE_ENTER` = 66
//! - `KEYCODE_BACK` = 4
//! - `KEYCODE_SPACE` = 62
//!
//! # Modifier Flags
//!
//! Modifier flags are represented as a bitmask:
//! - Bit 0 (0x01): SHIFT
//! - Bit 1 (0x02): ALT
//! - Bit 2 (0x04): CTRL
//! - Bit 3 (0x08): META
//!
//! # Example
//!
//! ```rust,ignore
//! use codevar_core::android::bluetooth_keyboard::{KeyEvent, KeyAction};
//!
//! // Create a key press event
//! let event = KeyEvent::new(29, KeyAction::Press, 0b0001);
//! assert_eq!(event.key_code, 29);
//! assert!(event.has_shift());
//!
//! // Check all modifiers
//! let event = KeyEvent::new(29, KeyAction::Press, 0b1111);
//! assert!(event.has_shift());
//! assert!(event.has_alt());
//! assert!(event.has_ctrl());
//! assert!(event.has_meta());
//! ```

/// Represents a keyboard input event.
///
/// This struct encapsulates all information about a keyboard event from a
/// Bluetooth HID device, including the key code, action type, and modifier state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyEvent {
    /// The Android key code (e.g., Android KeyEvent.KEYCODE_A = 29).
    pub key_code: i32,
    /// The key action (press or release).
    pub action: KeyAction,
    /// Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
    pub modifiers: i32,
}

impl KeyEvent {
    /// Creates a new key event.
    ///
    /// # Arguments
    ///
    /// * `key_code` - The Android key code.
    /// * `action` - The key action (press or release).
    /// * `modifiers` - Modifier flags bitmask.
    pub fn new(key_code: i32, action: KeyAction, modifiers: i32) -> Self {
        Self {
            key_code,
            action,
            modifiers,
        }
    }

    /// Creates a new key press event with no modifiers.
    pub fn press(key_code: i32) -> Self {
        Self::new(key_code, KeyAction::Press, 0)
    }

    /// Creates a new key release event with no modifiers.
    pub fn release(key_code: i32) -> Self {
        Self::new(key_code, KeyAction::Release, 0)
    }

    /// Checks if the SHIFT modifier is set.
    pub fn has_shift(&self) -> bool {
        self.modifiers & 0x01 != 0
    }

    /// Checks if the ALT modifier is set.
    pub fn has_alt(&self) -> bool {
        self.modifiers & 0x02 != 0
    }

    /// Checks if the CTRL modifier is set.
    pub fn has_ctrl(&self) -> bool {
        self.modifiers & 0x04 != 0
    }

    /// Checks if the META modifier is set.
    pub fn has_meta(&self) -> bool {
        self.modifiers & 0x08 != 0
    }

    /// Returns the number of active modifiers.
    pub fn modifier_count(&self) -> u32 {
        self.modifiers.count_ones()
    }

    /// Returns true if this is a key press event.
    pub fn is_press(&self) -> bool {
        self.action == KeyAction::Press
    }

    /// Returns true if this is a key release event.
    pub fn is_release(&self) -> bool {
        self.action == KeyAction::Release
    }
}

/// Enumeration of key actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyAction {
    /// Key was pressed
    Press,
    /// Key was released
    Release,
}

impl KeyAction {
    /// Converts from Java int representation.
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value from Java.
    ///
    /// # Returns
    ///
    /// `Some(KeyAction)` if the value is valid, `None` otherwise.
    pub fn from(value: i32) -> Option<Self> {
        match value {
            0 => Some(KeyAction::Press),
            1 => Some(KeyAction::Release),
            _ => None,
        }
    }

    /// Converts to Java int representation.
    ///
    /// # Returns
    ///
    /// The integer value compatible with Java code.
    pub fn to(&self) -> i32 {
        match self {
            KeyAction::Press => 0,
            KeyAction::Release => 1,
        }
    }
}

/// Common Android key code constants.
/// These are provided for convenience when creating key events.
pub mod keycode {
    /// Back key
    pub const BACK: i32 = 4;
    /// Volume up
    pub const VOLUME_UP: i32 = 24;
    /// Volume down
    pub const VOLUME_DOWN: i32 = 25;
    /// Power key
    pub const POWER: i32 = 26;
    /// Camera key
    pub const CAMERA: i32 = 27;
    /// Clear key
    pub const CLEAR: i32 = 28;
    /// A key
    pub const A: i32 = 29;
    /// B key
    pub const B: i32 = 30;
    /// C key
    pub const C: i32 = 31;
    /// D key
    pub const D: i32 = 32;
    /// E key
    pub const E: i32 = 33;
    /// F key
    pub const F: i32 = 34;
    /// G key
    pub const G: i32 = 35;
    /// H key
    pub const H: i32 = 36;
    /// I key
    pub const I: i32 = 37;
    /// J key
    pub const J: i32 = 38;
    /// K key
    pub const K: i32 = 39;
    /// L key
    pub const L: i32 = 40;
    /// M key
    pub const M: i32 = 41;
    /// N key
    pub const N: i32 = 42;
    /// O key
    pub const O: i32 = 43;
    /// P key
    pub const P: i32 = 44;
    /// Q key
    pub const Q: i32 = 45;
    /// R key
    pub const R: i32 = 46;
    /// S key
    pub const S: i32 = 47;
    /// T key
    pub const T: i32 = 48;
    /// U key
    pub const U: i32 = 49;
    /// V key
    pub const V: i32 = 50;
    /// W key
    pub const W: i32 = 51;
    /// X key
    pub const X: i32 = 52;
    /// Y key
    pub const Y: i32 = 53;
    /// Z key
    pub const Z: i32 = 54;
    /// Comma key
    pub const COMMA: i32 = 55;
    /// Period key
    pub const PERIOD: i32 = 56;
    /// Alt left
    pub const ALT_LEFT: i32 = 57;
    /// Alt right
    pub const ALT_RIGHT: i32 = 58;
    /// Shift left
    pub const SHIFT_LEFT: i32 = 59;
    /// Shift right
    pub const SHIFT_RIGHT: i32 = 60;
    /// Tab key
    pub const TAB: i32 = 61;
    /// Space key
    pub const SPACE: i32 = 62;
    /// Enter key
    pub const ENTER: i32 = 66;
    /// Delete key
    pub const DEL: i32 = 67;
    /// Menu key
    pub const MENU: i32 = 82;
    /// Escape key
    pub const ESCAPE: i32 = 111;
    /// Ctrl left
    pub const CTRL_LEFT: i32 = 113;
    /// Ctrl right
    pub const CTRL_RIGHT: i32 = 114;
    /// Caps lock
    pub const CAPS_LOCK: i32 = 115;
    /// Meta left
    pub const META_LEFT: i32 = 117;
    /// Meta right
    pub const META_RIGHT: i32 = 118;
}

/// Modifier flag constants.
pub mod modifier {
    /// Shift modifier flag
    pub const SHIFT: i32 = 0x01;
    /// Alt modifier flag
    pub const ALT: i32 = 0x02;
    /// Ctrl modifier flag
    pub const CTRL: i32 = 0x04;
    /// Meta modifier flag
    pub const META: i32 = 0x08;
}

#[cfg(target_os = "android")]
#[cfg(test)]
mod tests {
    use super::{KeyAction, KeyEvent, keycode, modifier};

    fn key_event_creation() {
        let event = KeyEvent::new(29, KeyAction::Press, 0);
        assert_eq!(event.key_code, 29);
        assert_eq!(event.action, KeyAction::Press);
        assert_eq!(event.modifiers, 0);
    }

    fn key_event_shortcuts() {
        let press = KeyEvent::press(29);
        assert_eq!(press.key_code, 29);
        assert!(press.is_press());

        let release = KeyEvent::release(29);
        assert_eq!(release.key_code, 29);
        assert!(release.is_release());
    }

    fn modifiers() {
        let event = KeyEvent::new(29, KeyAction::Press, 0b0001);
        assert!(event.has_shift());
        assert!(!event.has_alt());
        assert!(!event.has_ctrl());
        assert!(!event.has_meta());

        let event = KeyEvent::new(29, KeyAction::Press, 0b1111);
        assert!(event.has_shift());
        assert!(event.has_alt());
        assert!(event.has_ctrl());
        assert!(event.has_meta());
    }

    fn modifier_count() {
        let event = KeyEvent::new(29, KeyAction::Press, 0b0101);
        assert_eq!(event.modifier_count(), 2);
    }

    fn modifier_constants() {
        assert_eq!(modifier::SHIFT, 0x01);
        assert_eq!(modifier::ALT, 0x02);
        assert_eq!(modifier::CTRL, 0x04);
        assert_eq!(modifier::META, 0x08);
    }
}
