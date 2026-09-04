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

//! Input event types for Android Bluetooth HID devices.
//!
//! This module provides the unified `InputEvent` enum that represents both
//! keyboard and mouse input events from Bluetooth HID devices. It serves as
//! the primary type for event handling in the Rust callback system.
//!
//! # Event Types
//!
//! - **Key events**: Represent keyboard key presses and releases
//! - **Mouse events**: Represent mouse movements, button presses/releases, and scrolling
//!
//! # Example
//!
//! ```rust,ignore
//! use codevar_core::android::bluetooth_input::{InputEvent, InputType};
//! use codevar_core::android::bluetooth_keyboard::{KeyEvent, KeyAction};
//! use codevar_core::android::bluetooth_mouse::{MouseEvent, MouseEventType};
//!
//! // Create a keyboard event
//! let key_event = InputEvent::Key(KeyEvent::new(29, KeyAction::Press, 0));
//! assert_eq!(key_event.input_type(), InputType::Keyboard);
//!
//! // Create a mouse event
//! let mouse_event = InputEvent::Mouse(MouseEvent::new(100, 200, 0, 0, MouseEventType::Move));
//! assert_eq!(mouse_event.input_type(), InputType::Mouse);
//!
//! // Pattern match on event type
//! match key_event {
//!     InputEvent::Key(key) => println!("Key: {}", key.key_code),
//!     InputEvent::Mouse(mouse) => println!("Mouse: ({}, {})", mouse.x, mouse.y),
//! }
//! ```

use crate::android::bluetooth_keyboard::KeyEvent;
use crate::android::bluetooth_mouse::MouseEvent;
use std::fmt;

/// Represents an input event from a Bluetooth HID device.
///
/// This enum unifies keyboard and mouse events into a single type for
/// easier handling in callback implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputEvent {
    /// Keyboard input event
    Key(KeyEvent),
    /// Mouse input event
    Mouse(MouseEvent),
}

impl InputEvent {
    /// Returns the input type of this event.
    ///
    /// # Returns
    ///
    /// The input type (Keyboard or Mouse).
    pub fn input_type(&self) -> InputType {
        match self {
            InputEvent::Key(_) => InputType::Keyboard,
            InputEvent::Mouse(_) => InputType::Mouse,
        }
    }

    /// Returns true if this is a keyboard event.
    pub fn is_keyboard(&self) -> bool {
        matches!(self, InputEvent::Key(_))
    }

    /// Returns true if this is a mouse event.
    pub fn is_mouse(&self) -> bool {
        matches!(self, InputEvent::Mouse(_))
    }

    /// Attempts to extract the keyboard event.
    ///
    /// # Returns
    ///
    /// `Some(KeyEvent)` if this is a keyboard event, `None` otherwise.
    pub fn as_key(&self) -> Option<&KeyEvent> {
        match self {
            InputEvent::Key(key) => Some(key),
            _ => None,
        }
    }

    /// Attempts to extract the mouse event.
    ///
    /// # Returns
    ///
    /// `Some(MouseEvent)` if this is a mouse event, `None` otherwise.
    pub fn as_mouse(&self) -> Option<&MouseEvent> {
        match self {
            InputEvent::Mouse(mouse) => Some(mouse),
            _ => None,
        }
    }
}

impl fmt::Display for InputEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputEvent::Key(key) => write!(
                f,
                "InputEvent::Key(key_code={}, action={:?}, modifiers={})",
                key.key_code, key.action, key.modifiers
            ),
            InputEvent::Mouse(mouse) => write!(
                f,
                "InputEvent::Mouse(x={}, y={}, buttons={}, event_type={:?})",
                mouse.x, mouse.y, mouse.buttons, mouse.event_type
            ),
        }
    }
}

/// Enumeration of input event types.
///
/// Used to categorize events without needing to pattern match on the full enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputType {
    /// Keyboard input event
    Keyboard,
    /// Mouse input event
    Mouse,
}

impl InputType {
    /// Returns the name of this input type as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            InputType::Keyboard => "keyboard",
            InputType::Mouse => "mouse",
        }
    }
}

impl fmt::Display for InputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(target_os = "android")]
#[cfg(test)]
mod tests {
    use super::{InputEvent, InputType, KeyAction, KeyEvent, MouseEvent, MouseEventType};

    fn input_type_classification() {
        let key_event = InputEvent::Key(KeyEvent::new(29, KeyAction::Press, 0));
        let mouse_event =
            InputEvent::Mouse(MouseEvent::new(100, 200, 0, 0, MouseEventType::Move));

        assert!(key_event.is_keyboard());
        assert!(!key_event.is_mouse());
        assert!(mouse_event.is_mouse());
        assert!(!mouse_event.is_keyboard());
    }

    fn as_key() {
        let key_event = InputEvent::Key(KeyEvent::new(29, KeyAction::Press, 0));
        let mouse_event =
            InputEvent::Mouse(MouseEvent::new(100, 200, 0, 0, MouseEventType::Move));

        assert!(key_event.as_key().is_some());
        assert!(key_event.as_mouse().is_none());
        assert!(mouse_event.as_mouse().is_some());
        assert!(mouse_event.as_key().is_none());
    }

    fn input_type_display() {
        assert_eq!(InputType::Keyboard.as_str(), "keyboard");
        assert_eq!(InputType::Mouse.as_str(), "mouse");
    }

    fn input_event_display() {
        let key_event = InputEvent::Key(KeyEvent::new(29, KeyAction::Press, 0));
        let display = format!("{}", key_event);
        assert!(display.contains("InputEvent::Key"));
        assert!(display.contains("29"));
    }
}
