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

//! Decoding of Win32 raw records into portable input events.
//!
//! The records are plain values copied out of the Win32 structures, so the
//! state machine that turns them into events can be exercised without a
//! message queue or a window.

use codevar_kuitui_control::{
    DeviceId, InputEvent, KeyEvent, KeyState, Modifiers, Point, PointerButton, PointerEvent,
};

use crate::keymap;

/// The record is a key release instead of a press.
pub(crate) const RI_KEY_BREAK: u16 = 0x0001;
/// The record arrived through the extended (E0) prefix.
pub(crate) const RI_KEY_E0: u16 = 0x0002;
/// The pointer position is absolute.
pub(crate) const MOUSE_MOVE_ABSOLUTE: u16 = 0x0001;

/// Left button transition to pressed.
const RI_MOUSE_LEFT_BUTTON_DOWN: u16 = 0x0001;
/// Left button transition to released.
const RI_MOUSE_LEFT_BUTTON_UP: u16 = 0x0002;
/// Right button transition to pressed.
const RI_MOUSE_RIGHT_BUTTON_DOWN: u16 = 0x0004;
/// Right button transition to released.
const RI_MOUSE_RIGHT_BUTTON_UP: u16 = 0x0008;
/// Middle button transition to pressed.
const RI_MOUSE_MIDDLE_BUTTON_DOWN: u16 = 0x0010;
/// Middle button transition to released.
const RI_MOUSE_MIDDLE_BUTTON_UP: u16 = 0x0020;
/// Fourth button transition to pressed.
const RI_MOUSE_BUTTON_4_DOWN: u16 = 0x0040;
/// Fourth button transition to released.
const RI_MOUSE_BUTTON_4_UP: u16 = 0x0080;
/// Fifth button transition to pressed.
const RI_MOUSE_BUTTON_5_DOWN: u16 = 0x0100;
/// Fifth button transition to released.
const RI_MOUSE_BUTTON_5_UP: u16 = 0x0200;
/// The record carries a vertical wheel turn in `button_data`.
const RI_MOUSE_WHEEL: u16 = 0x0400;
/// The record carries a horizontal wheel turn in `button_data`.
const RI_MOUSE_HWHEEL: u16 = 0x0800;

/// Native wheel units in one notch.
const WHEEL_DELTA: f32 = 120.0;

/// The caps lock virtual key.
const VK_CAPITAL: u16 = 0x14;
/// The num lock virtual key.
const VK_NUMLOCK: u16 = 0x90;

/// The button transitions, paired with the portable button.
const BUTTONS: [(u16, u16, PointerButton); 5] = [
    (
        RI_MOUSE_LEFT_BUTTON_DOWN,
        RI_MOUSE_LEFT_BUTTON_UP,
        PointerButton::Primary,
    ),
    (
        RI_MOUSE_RIGHT_BUTTON_DOWN,
        RI_MOUSE_RIGHT_BUTTON_UP,
        PointerButton::Secondary,
    ),
    (
        RI_MOUSE_MIDDLE_BUTTON_DOWN,
        RI_MOUSE_MIDDLE_BUTTON_UP,
        PointerButton::Middle,
    ),
    (RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_4_UP, PointerButton::Back),
    (
        RI_MOUSE_BUTTON_5_DOWN,
        RI_MOUSE_BUTTON_5_UP,
        PointerButton::Forward,
    ),
];

/// One keyboard record, copied out of `RAWKEYBOARD`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeyRecord {
    /// Virtual-key code.
    pub(crate) vk: u16,
    /// Scan code of the key.
    pub(crate) make_code: u16,
    /// Transition and prefix flags.
    pub(crate) flags: u16,
}

/// One pointer record, copied out of `RAWMOUSE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MouseRecord {
    /// Movement mode flags.
    pub(crate) flags: u16,
    /// Button and wheel transitions.
    pub(crate) button_flags: u16,
    /// Wheel distance when a wheel flag is set.
    pub(crate) button_data: u16,
    /// Horizontal movement or absolute horizontal position.
    pub(crate) x: i32,
    /// Vertical movement or absolute vertical position.
    pub(crate) y: i32,
}

/// The decoding state of one device.
#[derive(Debug, Clone)]
pub(crate) struct DeviceState {
    /// Identifier carried by every event from the device.
    id: DeviceId,
    /// Modifiers and locks held by the device.
    modifiers: Modifiers,
    /// Last absolute position reported by the device.
    position: Option<Point>,
}

impl DeviceState {
    /// Creates the state of a device that has reported nothing yet.
    pub(crate) fn new(id: DeviceId) -> Self {
        Self {
            id,
            modifiers: Modifiers::NONE,
            position: None,
        }
    }

    /// Applies one keyboard record and emits its event.
    pub(crate) fn key(&mut self, record: KeyRecord, emit: &mut impl FnMut(InputEvent)) {
        let state = if record.flags & RI_KEY_BREAK != 0 {
            KeyState::Released
        } else {
            KeyState::Pressed
        };
        if let Some(modifier) = keymap::modifier(record.vk) {
            if state == KeyState::Pressed {
                self.modifiers.insert(modifier);
            } else {
                self.modifiers.remove(modifier);
            }
        }
        if state == KeyState::Pressed {
            match record.vk {
                VK_CAPITAL => toggle(&mut self.modifiers, Modifiers::CAPS_LOCK),
                VK_NUMLOCK => toggle(&mut self.modifiers, Modifiers::NUM_LOCK),
                _ => {}
            }
        }
        emit(InputEvent::Key(KeyEvent {
            device: self.id,
            code: keymap::key_code(record.vk, record.make_code, record.flags & RI_KEY_E0 != 0),
            state,
            modifiers: self.modifiers,
        }));
    }

    /// Applies one pointer record and emits the events it describes.
    pub(crate) fn mouse(&mut self, record: MouseRecord, emit: &mut impl FnMut(InputEvent)) {
        if record.flags & MOUSE_MOVE_ABSOLUTE != 0 {
            let position = Point::new(record.x as f32, record.y as f32);
            self.position = Some(position);
            emit(InputEvent::Pointer(PointerEvent::Motion {
                position: Some(position),
                delta: None,
            }));
        } else if record.x != 0 || record.y != 0 {
            emit(InputEvent::Pointer(PointerEvent::Motion {
                position: None,
                delta: Some(Point::new(record.x as f32, record.y as f32)),
            }));
        }
        for (down, up, button) in BUTTONS {
            if record.button_flags & down != 0 {
                self.button(button, KeyState::Pressed, emit);
            }
            if record.button_flags & up != 0 {
                self.button(button, KeyState::Released, emit);
            }
        }
        let mut delta = Point::ORIGIN;
        if record.button_flags & RI_MOUSE_WHEEL != 0 {
            delta.y = wheel(record.button_data);
        }
        if record.button_flags & RI_MOUSE_HWHEEL != 0 {
            delta.x = wheel(record.button_data);
        }
        if delta != Point::ORIGIN {
            emit(InputEvent::Pointer(PointerEvent::Scroll {
                delta,
                position: self.position,
                modifiers: self.modifiers,
            }));
        }
    }

    /// Emits one button transition.
    fn button(&self, button: PointerButton, state: KeyState, emit: &mut impl FnMut(InputEvent)) {
        emit(InputEvent::Pointer(PointerEvent::Button {
            button,
            state,
            position: self.position,
            modifiers: self.modifiers,
        }));
    }
}

/// Flips one lock flag.
fn toggle(modifiers: &mut Modifiers, flag: Modifiers) {
    if modifiers.contains(flag) {
        modifiers.remove(flag);
    } else {
        modifiers.insert(flag);
    }
}

/// Converts raw wheel units to notches.
fn wheel(data: u16) -> f32 {
    f32::from(data as i16) / WHEEL_DELTA
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use codevar_kuitui_control::{DeviceId, InputEvent, KeyCode, KeyState, Modifiers, Point, PointerButton};

    use super::{
        DeviceState, KeyRecord, MouseRecord, RI_KEY_BREAK, RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_WHEEL,
    };

    fn key(vk: u16, make_code: u16, flags: u16) -> KeyRecord {
        KeyRecord { vk, make_code, flags }
    }

    fn mouse(button_flags: u16, button_data: u16, x: i32, y: i32) -> MouseRecord {
        MouseRecord {
            flags: 0,
            button_flags,
            button_data,
            x,
            y,
        }
    }

    #[test]
    fn a_modifier_key_changes_the_state_carried_by_later_keys() {
        let mut state = DeviceState::new(DeviceId::new(7));
        let mut events = Vec::new();
        state.key(key(0x10, 0x2A, 0), &mut |event| events.push(event));
        state.key(key(0x41, 0x1E, 0), &mut |event| events.push(event));
        state.key(key(0x10, 0x2A, RI_KEY_BREAK), &mut |event| events.push(event));
        state.key(key(0x41, 0x1E, RI_KEY_BREAK), &mut |event| events.push(event));

        let codes: Vec<(KeyCode, KeyState, Modifiers)> = events
            .iter()
            .filter_map(|event| match event {
                InputEvent::Key(key) => Some((key.code, key.state, key.modifiers)),
                _ => None,
            })
            .collect();
        assert_eq!(codes[0].0, KeyCode::ShiftLeft);
        assert!(codes[0].2.contains(Modifiers::SHIFT));
        assert!(codes[1].2.contains(Modifiers::SHIFT));
        assert!(!codes[2].2.contains(Modifiers::SHIFT));
        assert!(!codes[3].2.contains(Modifiers::SHIFT));
        assert_eq!(codes[3].0, KeyCode::KeyA);
    }

    #[test]
    fn a_lock_key_toggles_its_flag_once_per_press() {
        let mut state = DeviceState::new(DeviceId::new(1));
        let mut modifiers = Vec::new();
        for _ in 0..2 {
            state.key(key(0x14, 0x3A, 0), &mut |event| {
                if let InputEvent::Key(key) = event {
                    modifiers.push(key.modifiers);
                }
            });
        }
        assert!(modifiers[0].contains(Modifiers::CAPS_LOCK));
        assert!(!modifiers[1].contains(Modifiers::CAPS_LOCK));
    }

    #[test]
    fn pointer_buttons_keep_their_identity() {
        let mut state = DeviceState::new(DeviceId::new(2));
        let mut buttons = Vec::new();
        state.mouse(mouse(RI_MOUSE_LEFT_BUTTON_DOWN, 0, 3, -4), &mut |event| {
            if let InputEvent::Pointer(pointer) = event {
                buttons.push(pointer);
            }
        });
        assert_eq!(
            buttons[0],
            codevar_kuitui_control::PointerEvent::Motion {
                position: None,
                delta: Some(Point::new(3.0, -4.0)),
            }
        );
        assert_eq!(
            buttons[1],
            codevar_kuitui_control::PointerEvent::Button {
                button: PointerButton::Primary,
                state: KeyState::Pressed,
                position: None,
                modifiers: Modifiers::NONE,
            }
        );
    }

    #[test]
    fn a_wheel_notch_becomes_one_unit() {
        let mut state = DeviceState::new(DeviceId::new(2));
        let mut scroll = None;
        state.mouse(mouse(RI_MOUSE_WHEEL, 120, 0, 0), &mut |event| {
            if let InputEvent::Pointer(codevar_kuitui_control::PointerEvent::Scroll { delta, .. }) = event {
                scroll = Some(delta);
            }
        });
        assert_eq!(scroll, Some(Point::new(0.0, 1.0)));
    }
}
