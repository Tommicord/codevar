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

//! Mouse input event types for Android Bluetooth HID devices.
//!
//! This module provides types for representing mouse input events from
//! Bluetooth HID devices. It includes the `MouseEvent` struct which encapsulates
//! coordinates, button states, scroll deltas, and event types.
//!
//! # Button Identifiers
//!
//! Button identifiers follow standard conventions:
//! - 1: Left button
//! - 2: Middle button
//! - 3: Right button
//! - 4+: Additional buttons (e.g., side buttons)
//!
//! # Button State Bitmask
//!
//! The button state is represented as a bitmask:
//! - Bit 0 (0x01): Left button pressed
//! - Bit 1 (0x02): Right button pressed
//! - Bit 2 (0x04): Middle button pressed
//! - Bit 3+ (0x08+): Additional buttons
//!
//! # Event Types
//!
//! - **Move**: Mouse movement without button state change
//! - **ButtonPress**: A mouse button was pressed
//! - **ButtonRelease**: A mouse button was released
//! - **Scroll**: Mouse wheel scroll event
//!
//! # Example
//!
//! ```rust,ignore
//! use codevar_core::android::bluetooth_mouse::{MouseEvent, MouseEventType};
//!
//! // Create a mouse move event
//! let event = MouseEvent::new(100, 200, 0, 0, MouseEventType::Move);
//! assert_eq!(event.x, 100);
//! assert_eq!(event.y, 200);
//!
//! // Create a button press event
//! let event = MouseEvent::new(100, 200, 0x01, 0, MouseEventType::ButtonPress);
//! assert!(event.is_left_button_pressed());
//! ```

use std::fmt;

/// Represents a mouse input event.
///
/// This struct encapsulates all information about a mouse event from a
/// Bluetooth HID device, including coordinates, button states, scroll deltas,
/// and the event type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MouseEvent {
    /// The X coordinate (relative to screen origin).
    pub x: i32,
    /// The Y coordinate (relative to screen origin).
    pub y: i32,
    /// Button state (bitmask of button states).
    pub buttons: i32,
    /// Scroll delta (for scroll events; positive for scroll up, negative for scroll down).
    pub scroll_delta: i32,
    /// The mouse event type.
    pub event_type: MouseEventType,
}

impl MouseEvent {
    /// Creates a new mouse event.
    ///
    /// # Arguments
    ///
    /// * `x` - The X coordinate.
    /// * `y` - The Y coordinate.
    /// * `buttons` - Button state bitmask.
    /// * `scroll_delta` - Scroll delta (0 for non-scroll events).
    /// * `event_type` - The mouse event type.
    pub fn new(
        x: i32,
        y: i32,
        buttons: i32,
        scroll_delta: i32,
        event_type: MouseEventType,
    ) -> Self {
        Self {
            x,
            y,
            buttons,
            scroll_delta,
            event_type,
        }
    }

    /// Creates a mouse move event.
    pub fn move_event(x: i32, y: i32) -> Self {
        Self::new(x, y, 0, 0, MouseEventType::Move)
    }

    /// Creates a button press event.
    pub fn button_press(x: i32, y: i32, button: i32) -> Self {
        let buttons = 1 << (button - 1);
        Self::new(x, y, buttons, 0, MouseEventType::ButtonPress)
    }

    /// Creates a button release event.
    pub fn button_release(x: i32, y: i32, button: i32) -> Self {
        let buttons = 1 << (button - 1);
        Self::new(x, y, buttons, 0, MouseEventType::ButtonRelease)
    }

    /// Creates a scroll event.
    pub fn scroll(x: i32, y: i32, delta: i32) -> Self {
        Self::new(x, y, 0, delta, MouseEventType::Scroll)
    }

    /// Checks if the left button is pressed.
    pub fn is_left_button_pressed(&self) -> bool {
        self.buttons & 0x01 != 0
    }

    /// Checks if the right button is pressed.
    pub fn is_right_button_pressed(&self) -> bool {
        self.buttons & 0x02 != 0
    }

    /// Checks if the middle button is pressed.
    pub fn is_middle_button_pressed(&self) -> bool {
        self.buttons & 0x04 != 0
    }

    /// Checks if a specific button is pressed.
    ///
    /// # Arguments
    ///
    /// * `button` - The button identifier (1=left, 2=right, 3=middle, etc.).
    pub fn is_button_pressed(&self, button: i32) -> bool {
        if button < 1 || button > 32 {
            return false;
        }
        self.buttons & (1 << (button - 1)) != 0
    }

    /// Returns the number of pressed buttons.
    pub fn pressed_button_count(&self) -> u32 {
        self.buttons.count_ones()
    }

    /// Returns true if this is a move event.
    pub fn is_move(&self) -> bool {
        self.event_type == MouseEventType::Move
    }

    /// Returns true if this is a button press event.
    pub fn is_button_press(&self) -> bool {
        self.event_type == MouseEventType::ButtonPress
    }

    /// Returns true if this is a button release event.
    pub fn is_button_release(&self) -> bool {
        self.event_type == MouseEventType::ButtonRelease
    }

    /// Returns true if this is a scroll event.
    pub fn is_scroll(&self) -> bool {
        self.event_type == MouseEventType::Scroll
    }

    /// Returns true if scrolling up (positive delta).
    pub fn is_scrolling_up(&self) -> bool {
        self.is_scroll() && self.scroll_delta > 0
    }

    /// Returns true if scrolling down (negative delta).
    pub fn is_scrolling_down(&self) -> bool {
        self.is_scroll() && self.scroll_delta < 0
    }

    /// Returns the position as a tuple.
    pub fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }
}

/// Enumeration of mouse event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseEventType {
    /// Mouse moved
    Move,
    /// Mouse button pressed
    ButtonPress,
    /// Mouse button released
    ButtonRelease,
    /// Mouse wheel scrolled
    Scroll,
}

impl MouseEventType {
    /// Converts from Java int representation.
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value from Java.
    ///
    /// # Returns
    ///
    /// `Some(MouseEventType)` if the value is valid, `None` otherwise.
    pub fn from(value: i32) -> Option<Self> {
        match value {
            0 => Some(MouseEventType::Move),
            1 => Some(MouseEventType::ButtonPress),
            2 => Some(MouseEventType::ButtonRelease),
            3 => Some(MouseEventType::Scroll),
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
            MouseEventType::Move => 0,
            MouseEventType::ButtonPress => 1,
            MouseEventType::ButtonRelease => 2,
            MouseEventType::Scroll => 3,
        }
    }
}

/// Button identifier constants.
pub mod button {
    /// Left mouse button
    pub const LEFT: i32 = 1;
    /// Right mouse button
    pub const RIGHT: i32 = 2;
    /// Middle mouse button
    pub const MIDDLE: i32 = 3;
    /// Back button (side button 1)
    pub const BACK: i32 = 4;
    /// Forward button (side button 2)
    pub const FORWARD: i32 = 5;
}

/// Button state bitmask constants.
pub mod button_state {
    /// Left button state bit
    pub const LEFT: i32 = 0x01;
    /// Right button state bit
    pub const RIGHT: i32 = 0x02;
    /// Middle button state bit
    pub const MIDDLE: i32 = 0x04;
    /// Back button state bit
    pub const BACK: i32 = 0x08;
    /// Forward button state bit
    pub const FORWARD: i32 = 0x10;
}

#[cfg(target_os = "android")]
#[cfg(test)]
mod tests {
    use super::{MouseEvent, MouseEventType, button, button_state};

    fn mouse_event_creation() {
        let event = MouseEvent::new(100, 200, 0, 0, MouseEventType::Move);
        assert_eq!(event.x, 100);
        assert_eq!(event.y, 200);
        assert_eq!(event.buttons, 0);
        assert_eq!(event.scroll_delta, 0);
    }

    fn mouse_event_shortcuts() {
        let move_event = MouseEvent::move_event(100, 200);
        assert!(move_event.is_move());

        let press_event = MouseEvent::button_press(100, 200, button::LEFT);
        assert!(press_event.is_button_press());
        assert!(press_event.is_left_button_pressed());

        let release_event = MouseEvent::button_release(100, 200, button::LEFT);
        assert!(release_event.is_button_release());

        let scroll_event = MouseEvent::scroll(100, 200, 10);
        assert!(scroll_event.is_scroll());
        assert!(scroll_event.is_scrolling_up());
    }

    fn button_states() {
        let event = MouseEvent::new(100, 200, 0b001, 0, MouseEventType::ButtonPress);
        assert!(event.is_left_button_pressed());
        assert!(!event.is_right_button_pressed());
        assert!(!event.is_middle_button_pressed());

        let event = MouseEvent::new(100, 200, 0b111, 0, MouseEventType::ButtonPress);
        assert!(event.is_left_button_pressed());
        assert!(event.is_right_button_pressed());
        assert!(event.is_middle_button_pressed());
    }

    fn is_button_pressed() {
        let event = MouseEvent::new(100, 200, 0x01, 0, MouseEventType::ButtonPress);
        assert!(event.is_button_pressed(1));
        assert!(!event.is_button_pressed(2));

        let event = MouseEvent::new(100, 200, 0x10, 0, MouseEventType::ButtonPress);
        assert!(event.is_button_pressed(5));
    }

    fn pressed_button_count() {
        let event = MouseEvent::new(100, 200, 0b101, 0, MouseEventType::ButtonPress);
        assert_eq!(event.pressed_button_count(), 2);
    }

    fn scroll_direction() {
        let scroll_up = MouseEvent::scroll(100, 200, 10);
        assert!(scroll_up.is_scrolling_up());
        assert!(!scroll_up.is_scrolling_down());

        let scroll_down = MouseEvent::scroll(100, 200, -10);
        assert!(scroll_down.is_scrolling_down());
        assert!(!scroll_down.is_scrolling_up());
    }

    fn position() {
        let event = MouseEvent::new(100, 200, 0, 0, MouseEventType::Move);
        assert_eq!(event.position(), (100, 200));
    }

    fn mouse_event_type_conversion() {
        assert_eq!(MouseEventType::from(0), Some(MouseEventType::Move));
        assert_eq!(MouseEventType::from(1), Some(MouseEventType::ButtonPress));
        assert_eq!(MouseEventType::from(2), Some(MouseEventType::ButtonRelease));
        assert_eq!(MouseEventType::from(3), Some(MouseEventType::Scroll));
        assert_eq!(MouseEventType::from(4), None);

        assert_eq!(MouseEventType::Move.to(), 0);
        assert_eq!(MouseEventType::ButtonPress.to(), 1);
        assert_eq!(MouseEventType::ButtonRelease.to(), 2);
        assert_eq!(MouseEventType::Scroll.to(), 3);
    }

    fn button_constants() {
        assert_eq!(button::LEFT, 1);
        assert_eq!(button::RIGHT, 2);
        assert_eq!(button::MIDDLE, 3);
        assert_eq!(button::BACK, 4);
        assert_eq!(button::FORWARD, 5);
    }

    fn button_state_constants() {
        assert_eq!(button_state::LEFT, 0x01);
        assert_eq!(button_state::RIGHT, 0x02);
        assert_eq!(button_state::MIDDLE, 0x04);
        assert_eq!(button_state::BACK, 0x08);
        assert_eq!(button_state::FORWARD, 0x10);
    }
}
