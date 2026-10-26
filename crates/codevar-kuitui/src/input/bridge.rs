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

//! The bridge between the pump's device events and the tree's events.
//!
//! The pump stages what the devices did — a [`KeyTransition`] with a
//! normalized [`KeyCode`], a [`PointerUpdate`] with the buttons held after the
//! event — and the widget tree is offered an [`Event`] instead, which names the
//! handful of keys a layout reacts to and folds a click into a
//! [`PointerEventKind::Down`] at a position. The two models are deliberately
//! different, so this module is the only place that converts between them.
//!
//! # What is not bridged
//!
//! A device event the tree cannot name produces no tree event. That is not a
//! loss: the pump applied it to the states all the same, and an application
//! that walks the tree with the [`InputFrame`] of the frame instead of the
//! bridge reaches a widget's [`KeyboardHandler::on_key`] and
//! [`MouseHandler::on_pointer`] hooks with the unconverted event.
//!
//! [`InputFrame`]: crate::input::InputFrame
//! [`KeyboardHandler::on_key`]: crate::input::KeyboardHandler::on_key
//! [`MouseHandler::on_pointer`]: crate::input::MouseHandler::on_pointer

use alloc::vec::Vec;

use codevar_kuitui_control::{InputEvent, KeyCode, KeyState, PointerButton};

use crate::event::{
    Event, Key, KeyEvent, Modifiers, PointerButton as TreeButton, PointerButtons as TreeButtons,
    PointerEvent, PointerEventKind,
};
use crate::geometry::Offset;
use crate::input::{
    InputFrame, KeyTransition, KeyboardModifiers, PointerButtons, PointerPhase, PointerUpdate,
};

/// The tree event a staged device event becomes.
///
/// `None` means the tree has no vocabulary for the event, not that the event
/// was dropped before it reached the states.
pub trait ToTreeEvent {
    /// The tree event for this device event.
    fn to_tree_event(&self) -> Option<Event>;
}

/// The tree key a device key code names, or `None`.
///
/// The tree reacts to the navigation keys; a letter or a digit reaches a
/// widget as [`Event::Text`], which carries what the keymap composed rather
/// than which key was pressed.
#[inline]
#[must_use]
pub fn key_of(code: KeyCode) -> Option<Key> {
    let key = match code {
        KeyCode::Escape => Key::Escape,
        KeyCode::Tab => Key::Tab,
        KeyCode::Enter => Key::Enter,
        KeyCode::Space => Key::Space,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Delete => Key::Delete,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::ArrowLeft => Key::ArrowLeft,
        KeyCode::ArrowRight => Key::ArrowRight,
        KeyCode::ArrowUp => Key::ArrowUp,
        KeyCode::ArrowDown => Key::ArrowDown,
        _ => return None,
    };
    Some(key)
}

/// The tree's modifier set for the device's modifier flags.
///
/// The lock bits are dropped: a tree event reports what was held while it
/// happened, and a widget reads caps lock from the keyboard state.
#[inline]
#[must_use]
pub fn modifiers_of(modifiers: KeyboardModifiers) -> Modifiers {
    let mut flags = Modifiers::NONE;
    if modifiers.contains(KeyboardModifiers::SHIFT) {
        flags |= Modifiers::SHIFT;
    }
    if modifiers.contains(KeyboardModifiers::CONTROL) {
        flags |= Modifiers::CONTROL;
    }
    if modifiers.contains(KeyboardModifiers::ALT) {
        flags |= Modifiers::ALT;
    }
    if modifiers.contains(KeyboardModifiers::META) {
        flags |= Modifiers::META;
    }
    flags
}

/// The tree's button set for the device's held buttons.
///
/// The tree names three buttons; a button held beyond them does not change
/// what a widget paints and stays visible in the pointer state.
#[inline]
#[must_use]
pub fn buttons_of(buttons: PointerButtons) -> TreeButtons {
    let mut held = TreeButtons::NONE;
    if buttons.contains(PointerButtons::PRIMARY) {
        held |= TreeButtons::PRIMARY;
    }
    if buttons.contains(PointerButtons::SECONDARY) {
        held |= TreeButtons::SECONDARY;
    }
    if buttons.contains(PointerButtons::MIDDLE) {
        held |= TreeButtons::MIDDLE;
    }
    held
}

/// The tree's button a device button names, or `None`.
#[inline]
#[must_use]
pub fn button_of(button: PointerButton) -> Option<TreeButton> {
    let tree = match button {
        PointerButton::Primary => TreeButton::Primary,
        PointerButton::Secondary => TreeButton::Secondary,
        PointerButton::Middle => TreeButton::Middle,
        _ => return None,
    };
    Some(tree)
}

impl ToTreeEvent for KeyTransition {
    fn to_tree_event(&self) -> Option<Event> {
        let key = key_of(self.code)?;
        let pressed = matches!(self.state, KeyState::Pressed | KeyState::Repeat);
        Some(Event::Key(
            KeyEvent::new(key, pressed).with_modifiers(modifiers_of(self.modifiers)),
        ))
    }
}

impl ToTreeEvent for PointerUpdate {
    fn to_tree_event(&self) -> Option<Event> {
        if self.phase == PointerPhase::Scroll {
            return Some(Event::Scroll(self.scroll));
        }
        let (kind, pressed) = match self.phase {
            PointerPhase::Motion => (PointerEventKind::Move, None),
            PointerPhase::Enter => (PointerEventKind::Enter, None),
            PointerPhase::Leave => (PointerEventKind::Leave, None),
            PointerPhase::Cancel => (PointerEventKind::Cancel, None),
            PointerPhase::Scroll => return Some(Event::Scroll(self.scroll)),
            PointerPhase::Unknown => return None,
            PointerPhase::Button => {
                let device = self.button?;
                let button = button_of(device)?;
                let held = PointerButtons::from_button(device).is_some_and(|bit| self.buttons.contains(bit));
                (
                    if held {
                        PointerEventKind::Down
                    } else {
                        PointerEventKind::Up
                    },
                    Some(button),
                )
            }
        };
        // The tree dispatches by position, so a backend that reports only
        // relative motion delivers the event at the origin instead of losing
        // the click it carried.
        let mut event = PointerEvent::new(kind, self.position.unwrap_or(Offset::ZERO))
            .with_buttons(buttons_of(self.buttons))
            .with_modifiers(modifiers_of(self.modifiers));
        if let Some(button) = pressed {
            event.button = button;
        }
        Some(Event::Pointer(event))
    }
}

/// Pushes the tree events of one device event that carries no state.
///
/// Text is the only one the tree understands: a composed string becomes one
/// [`Event::Text`] per character, in order. A gamepad, device or audio event
/// has no tree representation and pushes nothing.
pub fn push_input_event(event: &InputEvent, out: &mut Vec<Event>) {
    if let InputEvent::Text(text) = event {
        out.extend(text.text.chars().map(Event::Text));
    }
}

/// Collects the tree events of one staged frame into `out`.
///
/// `out` is appended to and not cleared, so a caller keeps one buffer across
/// frames and allocates only while it grows. The order follows
/// [`InputFrame::deliver_to`]: the key transitions first, so a shortcut that
/// consumes a key is seen before the text the same press produced, then the
/// pointer updates, then the text.
///
/// [`InputFrame::deliver_to`]: crate::input::InputFrame::deliver_to
pub fn bridge_frame(frame: InputFrame<'_>, out: &mut Vec<Event>) {
    for transition in frame.keys() {
        if let Some(event) = transition.to_tree_event() {
            out.push(event);
        }
    }
    for update in frame.pointers() {
        if let Some(event) = update.to_tree_event() {
            out.push(event);
        }
    }
    for event in frame.events() {
        push_input_event(&event.payload, out);
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use codevar_kuitui_control::{
        DeviceId, InputEvent as DeviceEvent, Modifiers as DeviceModifiers, Point,
        PointerEvent as DevicePointerEvent, TextEvent, Timestamp,
    };

    use super::*;
    use crate::input::PointerState;

    fn when(micros: u64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn transition(code: KeyCode, state: KeyState, flags: DeviceModifiers) -> KeyTransition {
        KeyTransition {
            timestamp: when(1),
            device: DeviceId::new(3),
            code,
            state,
            modifiers: KeyboardModifiers::from(flags),
        }
    }

    fn pointer(position: Option<Point>, delta: Option<Point>) -> DevicePointerEvent {
        DevicePointerEvent::Motion { position, delta }
    }

    fn press(button: PointerButton, state: KeyState, position: Option<Point>) -> DevicePointerEvent {
        DevicePointerEvent::Button {
            button,
            state,
            position,
            modifiers: DeviceModifiers::ALT,
        }
    }

    #[test]
    fn a_navigation_key_bridges_to_a_key_event_with_its_modifiers() {
        let event = transition(
            KeyCode::Tab,
            KeyState::Pressed,
            DeviceModifiers::SHIFT | DeviceModifiers::CONTROL,
        )
        .to_tree_event();
        assert_eq!(
            event,
            Some(Event::Key(
                KeyEvent::new(Key::Tab, true).with_modifiers(Modifiers::SHIFT | Modifiers::CONTROL)
            ))
        );
    }

    #[test]
    fn a_release_is_not_pressed_and_a_key_repeat_is() {
        assert_eq!(
            transition(KeyCode::Enter, KeyState::Released, DeviceModifiers::NONE).to_tree_event(),
            Some(Event::Key(KeyEvent::new(Key::Enter, false)))
        );
        assert_eq!(
            transition(KeyCode::Space, KeyState::Repeat, DeviceModifiers::NONE).to_tree_event(),
            Some(Event::Key(KeyEvent::new(Key::Space, true)))
        );
    }

    #[test]
    fn a_key_the_tree_does_not_name_bridges_to_nothing() {
        for code in [KeyCode::KeyA, KeyCode::NumpadEnter, KeyCode::Native(0x1234)] {
            assert_eq!(
                transition(code, KeyState::Pressed, DeviceModifiers::NONE).to_tree_event(),
                None,
                "{code:?}"
            );
        }
    }

    #[test]
    fn a_motion_bridges_to_a_move_at_the_last_reported_position() {
        let state = PointerState::new();
        state.apply(&pointer(Some(Point::new(6.0, 8.0)), None), when(1));
        let update = PointerUpdate::new(&state, &pointer(None, Some(Point::new(1.0, 1.0))), when(2));
        assert_eq!(
            update.to_tree_event(),
            Some(Event::Pointer(PointerEvent::new(
                PointerEventKind::Move,
                Offset::new(6.0, 8.0)
            )))
        );
    }

    #[test]
    fn a_press_and_a_release_of_one_button_become_down_and_up() {
        let state = PointerState::new();
        let down = PointerUpdate::new(
            &state,
            &press(PointerButton::Secondary, KeyState::Pressed, None),
            when(3),
        )
        .to_tree_event();
        assert!(matches!(
            down,
            Some(Event::Pointer(PointerEvent {
                kind: PointerEventKind::Down,
                button: TreeButton::Secondary,
                buttons,
                modifiers,
                ..
            })) if buttons.contains(TreeButtons::SECONDARY) && modifiers.contains(Modifiers::ALT)
        ));

        let up = PointerUpdate::new(
            &state,
            &press(
                PointerButton::Secondary,
                KeyState::Released,
                Some(Point::new(2.0, 3.0)),
            ),
            when(4),
        )
        .to_tree_event();
        assert!(matches!(
            up,
            Some(Event::Pointer(PointerEvent {
                kind: PointerEventKind::Up,
                position,
                buttons,
                ..
            })) if position == Offset::new(2.0, 3.0) && !buttons.contains(TreeButtons::SECONDARY)
        ));
    }

    #[test]
    fn a_button_and_a_phase_the_tree_cannot_name_bridge_to_nothing() {
        let state = PointerState::new();
        let back = PointerUpdate::new(
            &state,
            &press(PointerButton::Back, KeyState::Pressed, None),
            when(5),
        );
        assert_eq!(back.to_tree_event(), None);

        let unknown = PointerUpdate {
            timestamp: when(6),
            phase: PointerPhase::Unknown,
            ..PointerUpdate::new(&state, &pointer(None, None), when(6))
        };
        assert_eq!(unknown.to_tree_event(), None);
    }

    #[test]
    fn a_scroll_bridges_to_the_scroll_of_its_own_event() {
        let state = PointerState::new();
        let update = PointerUpdate::new(
            &state,
            &DevicePointerEvent::Scroll {
                delta: Point::new(0.0, 12.0),
                position: None,
                modifiers: DeviceModifiers::NONE,
            },
            when(6),
        );
        assert_eq!(
            update.to_tree_event(),
            Some(Event::Scroll(Offset::new(0.0, 12.0)))
        );
    }

    #[test]
    fn text_bridges_to_one_character_event_and_the_rest_of_the_stream_to_none() {
        let mut out = Vec::new();
        push_input_event(&text("hi"), &mut out);
        assert_eq!(out, [Event::Text('h'), Event::Text('i')]);

        out.clear();
        push_input_event(
            &DeviceEvent::Gamepad(codevar_kuitui_control::GamepadEvent::Button {
                device: DeviceId::new(9),
                button: 0,
                state: KeyState::Pressed,
            }),
            &mut out,
        );
        push_input_event(&DeviceEvent::DeviceRemoved(DeviceId::new(9)), &mut out);
        assert!(out.is_empty());
    }

    fn text(body: &str) -> DeviceEvent {
        DeviceEvent::Text(TextEvent {
            device: DeviceId::new(3),
            text: body.to_string(),
        })
    }

    // The frame-level ordering that `bridge_frame` guarantees is covered end
    // to end by the driver, which is the only caller that holds a real frame.
}
