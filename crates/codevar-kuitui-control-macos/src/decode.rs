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

//! Per-device decoding state.
//!
//! The event tap reports one interleaved stream for every device, so the
//! backend keeps the state a consumer expects per device: the modifier set for
//! the keyboard and the last absolute position for the pointer.

use codevar_kuitui_control::{
    DeviceId, InputEvent, KeyEvent, KeyState, Modifiers, Point, PointerButton, PointerEvent,
};

use crate::keymap;

/// What a keyboard event says happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyAction {
    /// The key went down.
    Press,
    /// The key is held and the system repeated it.
    Repeat,
    /// The key went up.
    Release,
    /// A modifier or lock key changed state.
    FlagsChanged,
}

/// Keyboard decoding state.
#[derive(Debug)]
pub(crate) struct KeyboardState {
    /// Identifier carried by every event.
    id: DeviceId,
    /// Modifiers held after the last event.
    modifiers: Modifiers,
}

impl KeyboardState {
    /// Creates the state of the keyboard that `id` identifies.
    pub(crate) const fn new(id: DeviceId) -> Self {
        Self {
            id,
            modifiers: Modifiers::NONE,
        }
    }

    /// Applies one keyboard event and emits its normalized form.
    ///
    /// A flags-changed event has no direction of its own: whether it pressed
    /// or released the key comes from the flag bit the key's family owns.
    pub(crate) fn key(
        &mut self,
        keycode: u16,
        action: KeyAction,
        flags: u64,
        emit: &mut impl FnMut(InputEvent),
    ) {
        let state = match action {
            KeyAction::Press => KeyState::Pressed,
            KeyAction::Repeat => KeyState::Repeat,
            KeyAction::Release => KeyState::Released,
            KeyAction::FlagsChanged => {
                let Some(family) = keymap::family(keycode) else {
                    return;
                };
                if flags & family.mask() != 0 {
                    KeyState::Pressed
                } else {
                    KeyState::Released
                }
            }
        };
        self.modifiers = keymap::modifiers(flags);
        emit(InputEvent::Key(KeyEvent {
            device: self.id,
            code: keymap::key_code(keycode),
            state,
            modifiers: self.modifiers,
        }));
    }
}

/// Pointer decoding state.
#[derive(Debug)]
pub(crate) struct PointerState {
    /// The last absolute position, kept for button and scroll events.
    position: Option<Point>,
}

impl PointerState {
    /// Creates a pointer that has reported no position yet.
    pub(crate) const fn new() -> Self {
        Self { position: None }
    }

    /// Records an absolute position and emits it with the delta since the
    /// previous sample.
    pub(crate) fn motion(&mut self, x: f32, y: f32, emit: &mut impl FnMut(InputEvent)) {
        let position = Point::new(x, y);
        let delta = self
            .position
            .map(|previous| Point::new(position.x - previous.x, position.y - previous.y));
        self.position = Some(position);
        emit(InputEvent::Pointer(PointerEvent::Motion {
            position: Some(position),
            delta,
        }));
    }

    /// Emits a button transition at the last known position.
    pub(crate) fn button(
        &self,
        button: PointerButton,
        state: KeyState,
        modifiers: Modifiers,
        emit: &mut impl FnMut(InputEvent),
    ) {
        emit(InputEvent::Pointer(PointerEvent::Button {
            button,
            state,
            position: self.position,
            modifiers,
        }));
    }

    /// Emits a scroll at the last known position.
    pub(crate) fn scroll(&self, delta: Point, modifiers: Modifiers, emit: &mut impl FnMut(InputEvent)) {
        emit(InputEvent::Pointer(PointerEvent::Scroll {
            delta,
            position: self.position,
            modifiers,
        }));
    }
}

/// The normalized button at a CoreGraphics button number.
///
/// Quartz numbers the buttons left, right, middle, then the side buttons,
/// which is also the order the other backends report.
#[must_use]
pub(crate) fn button(number: i64) -> PointerButton {
    match number {
        0 => PointerButton::Primary,
        1 => PointerButton::Secondary,
        2 => PointerButton::Middle,
        3 => PointerButton::Back,
        4 => PointerButton::Forward,
        other => PointerButton::Other(u16::try_from(other).unwrap_or(u16::MAX)),
    }
}

/// Converts a wheel reading into the backend's scroll units.
///
/// A discrete wheel reports one line per notch. A trackpad reports pixels, so
/// ten pixels are treated as one notch to keep both kinds in the same range.
#[must_use]
pub(crate) fn scroll_delta(continuous: bool, axis1: i64, axis2: i64) -> Point {
    let scale = if continuous { 0.1 } else { 1.0 };
    Point::new(axis2 as f32 * scale, axis1 as f32 * scale)
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::{
        DeviceId, InputEvent, KeyCode, KeyState, Modifiers, Point, PointerButton, PointerEvent,
    };

    use super::{KeyAction, KeyboardState, PointerState, button, scroll_delta};

    fn key_state(events: &[InputEvent]) -> KeyState {
        match events {
            [InputEvent::Key(event)] => event.state,
            _ => panic!("expected one key event"),
        }
    }

    #[test]
    fn a_flags_change_presses_and_releases_from_the_family_bit() {
        let mut keyboard = KeyboardState::new(DeviceId::new(1));
        let mut pressed = alloc::vec::Vec::new();
        keyboard.key(
            0x38,
            KeyAction::FlagsChanged,
            crate::keymap::EVENT_FLAG_SHIFT,
            &mut |event| pressed.push(event),
        );
        assert_eq!(key_state(&pressed), KeyState::Pressed);

        let mut released = alloc::vec::Vec::new();
        keyboard.key(0x38, KeyAction::FlagsChanged, 0, &mut |event| {
            released.push(event)
        });
        assert_eq!(key_state(&released), KeyState::Released);
    }

    #[test]
    fn caps_lock_reports_the_lock_bit_from_the_flags() {
        let mut keyboard = KeyboardState::new(DeviceId::new(1));
        let mut events = alloc::vec::Vec::new();
        keyboard.key(
            0x39,
            KeyAction::FlagsChanged,
            crate::keymap::EVENT_FLAG_CAPS_LOCK,
            &mut |event| events.push(event),
        );
        let [InputEvent::Key(event)] = events.as_slice() else {
            panic!("expected one key event");
        };
        assert_eq!(event.code, KeyCode::CapsLock);
        assert!(event.modifiers.contains(Modifiers::CAPS_LOCK));
    }

    #[test]
    fn motion_reports_a_delta_after_the_first_sample() {
        let mut pointer = PointerState::new();
        let mut events = alloc::vec::Vec::new();
        pointer.motion(10.0, 20.0, &mut |event| events.push(event));
        pointer.motion(13.0, 18.0, &mut |event| events.push(event));
        assert!(matches!(
            events.as_slice(),
            [
                InputEvent::Pointer(PointerEvent::Motion { delta: None, .. }),
                InputEvent::Pointer(PointerEvent::Motion {
                    position: Some(position),
                    delta: Some(delta),
                }),
            ] if *position == Point::new(13.0, 18.0) && *delta == Point::new(3.0, -2.0)
        ));
    }

    #[test]
    fn buttons_follow_the_quartz_numbering() {
        assert_eq!(button(0), PointerButton::Primary);
        assert_eq!(button(1), PointerButton::Secondary);
        assert_eq!(button(2), PointerButton::Middle);
        assert_eq!(button(3), PointerButton::Back);
        assert_eq!(button(4), PointerButton::Forward);
        assert_eq!(button(9), PointerButton::Other(9));
    }

    #[test]
    fn scroll_keeps_lines_and_scales_pixels() {
        assert_eq!(scroll_delta(false, 1, -1), Point::new(-1.0, 1.0));
        assert_eq!(scroll_delta(true, 20, 0), Point::new(0.0, 2.0));
    }
}
