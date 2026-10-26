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

//! The events a widget can receive.
//!
//! The event set is deliberately small: pointer, keyboard, focus,
//! scroll and window events. A widget reacts to an event by returning
//! [`EventResult::Consumed`], which stops the event from reaching the
//! widgets below it in the tree.

use crate::geometry::{Offset, Size};

/// A bit set of the modifier keys that are held down.
///
/// The modifiers are a bit set rather than a struct of booleans so
/// that a new modifier is a new constant and not a new field of every
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Modifiers(u8);

impl Modifiers {
    /// No modifier is held down.
    pub const NONE: Modifiers = Modifiers(0);
    /// The shift key is held down.
    pub const SHIFT: Modifiers = Modifiers(1 << 0);
    /// The control key is held down.
    pub const CONTROL: Modifiers = Modifiers(1 << 1);
    /// The alt key is held down.
    pub const ALT: Modifiers = Modifiers(1 << 2);
    /// The meta or super key is held down.
    pub const META: Modifiers = Modifiers(1 << 3);

    /// Whether every modifier of `other` is held down.
    #[must_use]
    pub const fn contains(self, other: Modifiers) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no modifier at all is held down.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for Modifiers {
    type Output = Modifiers;

    fn bitor(self, rhs: Modifiers) -> Modifiers {
        Modifiers(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for Modifiers {
    fn bitor_assign(&mut self, rhs: Modifiers) {
        self.0 |= rhs.0;
    }
}

/// A button of a pointing device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerButton {
    /// The primary button, usually the left one.
    Primary,
    /// The secondary button, usually the right one.
    Secondary,
    /// The middle button.
    Middle,
}

/// A bit set of the pointer buttons that are held down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PointerButtons(u8);

impl PointerButtons {
    /// No button is held down.
    pub const NONE: PointerButtons = PointerButtons(0);
    /// The primary button is held down.
    pub const PRIMARY: PointerButtons = PointerButtons(1 << 0);
    /// The secondary button is held down.
    pub const SECONDARY: PointerButtons = PointerButtons(1 << 1);
    /// The middle button is held down.
    pub const MIDDLE: PointerButtons = PointerButtons(1 << 2);

    /// Whether every button of `other` is held down.
    #[must_use]
    pub const fn contains(self, other: PointerButtons) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no button at all is held down.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for PointerButtons {
    type Output = PointerButtons;

    fn bitor(self, rhs: PointerButtons) -> PointerButtons {
        PointerButtons(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for PointerButtons {
    fn bitor_assign(&mut self, rhs: PointerButtons) {
        self.0 |= rhs.0;
    }
}

/// What a pointer event reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerEventKind {
    /// A button went down.
    Down,
    /// A button went up.
    Up,
    /// The pointer moved.
    Move,
    /// The pointer entered the widget.
    Enter,
    /// The pointer left the widget.
    Leave,
    /// The pointer was cancelled, for example by a palm rejection.
    Cancel,
}

/// A pointer event delivered to a widget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerEvent {
    /// What the pointer did.
    pub kind: PointerEventKind,
    /// The position of the pointer, in the coordinate space of the
    /// window.
    pub position: Offset,
    /// The button the event is about.
    pub button: PointerButton,
    /// The buttons that are held down.
    pub buttons: PointerButtons,
    /// The modifier keys that are held down.
    pub modifiers: Modifiers,
}

impl PointerEvent {
    /// Creates a pointer event of the given kind at `position`.
    #[must_use]
    pub const fn new(kind: PointerEventKind, position: Offset) -> Self {
        Self {
            kind,
            position,
            button: PointerButton::Primary,
            buttons: PointerButtons::NONE,
            modifiers: Modifiers::NONE,
        }
    }

    /// The event with a different set of held buttons.
    #[must_use]
    pub const fn with_buttons(mut self, buttons: PointerButtons) -> Self {
        self.buttons = buttons;
        self
    }

    /// The event with a different set of held modifiers.
    #[must_use]
    pub const fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }
}

/// A key of the keyboard.
///
/// The set covers the keys a user interface reacts to. A text input
/// widget uses the [`Event::Text`] variant instead, which carries the
/// character the key produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Key {
    /// The tab key.
    Tab,
    /// The enter key.
    Enter,
    /// The escape key.
    Escape,
    /// The backspace key.
    Backspace,
    /// The delete key.
    Delete,
    /// The left arrow key.
    ArrowLeft,
    /// The right arrow key.
    ArrowRight,
    /// The up arrow key.
    ArrowUp,
    /// The down arrow key.
    ArrowDown,
    /// The home key.
    Home,
    /// The end key.
    End,
    /// The page up key.
    PageUp,
    /// The page down key.
    PageDown,
    /// The space bar.
    Space,
}

/// A key press delivered to a widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyEvent {
    /// The key that was pressed.
    pub key: Key,
    /// Whether the key was pressed or released.
    pub pressed: bool,
    /// The modifier keys that were held down.
    pub modifiers: Modifiers,
}

impl KeyEvent {
    /// Creates a key event for `key`.
    #[must_use]
    pub const fn new(key: Key, pressed: bool) -> Self {
        Self {
            key,
            pressed,
            modifiers: Modifiers::NONE,
        }
    }

    /// The event with a different set of held modifiers.
    #[must_use]
    pub const fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }
}

/// A change of the state of the window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum WindowEvent {
    /// The window changed its size.
    Resized(Size),
    /// The window gained or lost the focus.
    FocusChanged(bool),
    /// The window is about to close.
    CloseRequested,
    /// The scale factor of the window changed.
    ScaleChanged(f32),
}

/// An event delivered to the widget tree.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// A pointer event.
    Pointer(PointerEvent),
    /// A key press or release.
    Key(KeyEvent),
    /// A character was typed.
    Text(char),
    /// The pointer wheel was scrolled.
    Scroll(Offset),
    /// The widget gained or lost the focus.
    Focus(bool),
    /// The window changed.
    Window(WindowEvent),
}

/// What a widget did with an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventResult {
    /// The event was not handled and keeps travelling down the tree.
    Ignored,
    /// The event was handled and stops travelling down the tree.
    Consumed,
}

impl EventResult {
    /// Whether the event was handled.
    #[must_use]
    pub const fn is_consumed(self) -> bool {
        matches!(self, Self::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifiers_combine_with_the_bit_or_operator() {
        let modifiers = Modifiers::SHIFT | Modifiers::CONTROL;
        assert!(modifiers.contains(Modifiers::SHIFT));
        assert!(modifiers.contains(Modifiers::CONTROL));
        assert!(!modifiers.contains(Modifiers::ALT));
        assert!(!modifiers.is_empty());
        assert!(Modifiers::NONE.is_empty());
    }

    #[test]
    fn pointer_buttons_combine_with_the_bit_or_operator() {
        let buttons = PointerButtons::PRIMARY | PointerButtons::MIDDLE;
        assert!(buttons.contains(PointerButtons::PRIMARY));
        assert!(buttons.contains(PointerButtons::MIDDLE));
        assert!(!buttons.contains(PointerButtons::SECONDARY));
    }

    #[test]
    fn an_event_result_reports_whether_it_was_consumed() {
        assert!(EventResult::Consumed.is_consumed());
        assert!(!EventResult::Ignored.is_consumed());
    }

    #[test]
    fn a_pointer_event_carries_its_position() {
        let event = PointerEvent::new(PointerEventKind::Down, Offset::new(3.0, 4.0));
        assert_eq!(event.position, Offset::new(3.0, 4.0));
        assert_eq!(event.kind, PointerEventKind::Down);
    }
}
