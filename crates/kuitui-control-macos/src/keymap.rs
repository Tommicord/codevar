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

//! Translation from CoreGraphics virtual keycodes and event flags to the
//! normalized model.
//!
//! The keycode is the physical key position in the ANSI layout, so it stays
//! stable when the user switches keyboard layout. Keys the table does not
//! cover keep their native keycode in [`KeyCode::Native`].

use codevar_kuitui_control::{KeyCode, Modifiers};

/// Shift is held (`kCGEventFlagMaskShift`).
pub(crate) const EVENT_FLAG_SHIFT: u64 = 0x0002_0000;
/// Control is held (`kCGEventFlagMaskControl`).
pub(crate) const EVENT_FLAG_CONTROL: u64 = 0x0004_0000;
/// Option is held (`kCGEventFlagMaskAlternate`).
pub(crate) const EVENT_FLAG_ALT: u64 = 0x0008_0000;
/// Command is held (`kCGEventFlagMaskCommand`).
pub(crate) const EVENT_FLAG_META: u64 = 0x0010_0000;
/// Caps lock is on (`kCGEventFlagMaskAlphaShift`).
pub(crate) const EVENT_FLAG_CAPS_LOCK: u64 = 0x0001_0000;
/// The numeric keypad is active (`kCGEventFlagMaskNumericPad`).
pub(crate) const EVENT_FLAG_NUM_LOCK: u64 = 0x0020_0000;

/// The modifier family a key belongs to.
///
/// The family only says which side-independent bit the key controls; the
/// left and right keys of a family share it, and the keycode identifies which
/// one changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    /// Shift, mask [`EVENT_FLAG_SHIFT`].
    Shift,
    /// Control, mask [`EVENT_FLAG_CONTROL`].
    Control,
    /// Option, mask [`EVENT_FLAG_ALT`].
    Alt,
    /// Command, mask [`EVENT_FLAG_META`].
    Meta,
    /// Caps lock, mask [`EVENT_FLAG_CAPS_LOCK`].
    CapsLock,
}

impl Family {
    /// The event-flag bit the family reports.
    #[must_use]
    pub(crate) const fn mask(self) -> u64 {
        match self {
            Family::Shift => EVENT_FLAG_SHIFT,
            Family::Control => EVENT_FLAG_CONTROL,
            Family::Alt => EVENT_FLAG_ALT,
            Family::Meta => EVENT_FLAG_META,
            Family::CapsLock => EVENT_FLAG_CAPS_LOCK,
        }
    }
}

/// The modifier family controlled by a virtual keycode, if any.
///
/// A flags-changed event is sent instead of a key-down event for these keys,
/// so the backend needs the family to tell whether the change pressed or
/// released the key.
#[must_use]
pub(crate) const fn family(keycode: u16) -> Option<Family> {
    match keycode {
        // Left and right shift.
        0x38 | 0x3C => Some(Family::Shift),
        // Left and right control.
        0x3B | 0x3E => Some(Family::Control),
        // Left and right option.
        0x3A | 0x3D => Some(Family::Alt),
        // Left and right command.
        0x37 | 0x36 => Some(Family::Meta),
        0x39 => Some(Family::CapsLock),
        _ => None,
    }
}

/// The modifier set described by the event flags.
///
/// CoreGraphics reports the whole set with every event, so the backend does
/// not have to track which key changed to know the current state.
#[must_use]
pub(crate) fn modifiers(flags: u64) -> Modifiers {
    let mut modifiers = Modifiers::NONE;
    if flags & EVENT_FLAG_SHIFT != 0 {
        modifiers.insert(Modifiers::SHIFT);
    }
    if flags & EVENT_FLAG_CONTROL != 0 {
        modifiers.insert(Modifiers::CONTROL);
    }
    if flags & EVENT_FLAG_ALT != 0 {
        modifiers.insert(Modifiers::ALT);
    }
    if flags & EVENT_FLAG_META != 0 {
        modifiers.insert(Modifiers::META);
    }
    if flags & EVENT_FLAG_CAPS_LOCK != 0 {
        modifiers.insert(Modifiers::CAPS_LOCK);
    }
    if flags & EVENT_FLAG_NUM_LOCK != 0 {
        modifiers.insert(Modifiers::NUM_LOCK);
    }
    modifiers
}

/// The normalized key at a virtual keycode.
///
/// The values are the `kVK_*` constants from `Carbon/HIToolbox/Events.h`.
#[must_use]
pub(crate) const fn key_code(keycode: u16) -> KeyCode {
    match keycode {
        0x00 => KeyCode::KeyA,
        0x0B => KeyCode::KeyB,
        0x08 => KeyCode::KeyC,
        0x02 => KeyCode::KeyD,
        0x0E => KeyCode::KeyE,
        0x03 => KeyCode::KeyF,
        0x05 => KeyCode::KeyG,
        0x04 => KeyCode::KeyH,
        0x22 => KeyCode::KeyI,
        0x26 => KeyCode::KeyJ,
        0x28 => KeyCode::KeyK,
        0x25 => KeyCode::KeyL,
        0x2E => KeyCode::KeyM,
        0x2D => KeyCode::KeyN,
        0x1F => KeyCode::KeyO,
        0x23 => KeyCode::KeyP,
        0x0C => KeyCode::KeyQ,
        0x0F => KeyCode::KeyR,
        0x01 => KeyCode::KeyS,
        0x11 => KeyCode::KeyT,
        0x20 => KeyCode::KeyU,
        0x09 => KeyCode::KeyV,
        0x0D => KeyCode::KeyW,
        0x07 => KeyCode::KeyX,
        0x10 => KeyCode::KeyY,
        0x06 => KeyCode::KeyZ,
        0x12 => KeyCode::Digit1,
        0x13 => KeyCode::Digit2,
        0x14 => KeyCode::Digit3,
        0x15 => KeyCode::Digit4,
        0x17 => KeyCode::Digit5,
        0x16 => KeyCode::Digit6,
        0x1A => KeyCode::Digit7,
        0x1C => KeyCode::Digit8,
        0x19 => KeyCode::Digit9,
        0x1D => KeyCode::Digit0,
        0x32 => KeyCode::Backquote,
        0x1B => KeyCode::Minus,
        0x18 => KeyCode::Equal,
        0x21 => KeyCode::BracketLeft,
        0x1E => KeyCode::BracketRight,
        0x2A => KeyCode::Backslash,
        0x29 => KeyCode::Semicolon,
        0x27 => KeyCode::Quote,
        0x2B => KeyCode::Comma,
        0x2F => KeyCode::Period,
        0x2C => KeyCode::Slash,
        0x0A => KeyCode::IntlBackslash,
        0x35 => KeyCode::Escape,
        0x30 => KeyCode::Tab,
        0x39 => KeyCode::CapsLock,
        0x31 => KeyCode::Space,
        0x24 => KeyCode::Enter,
        0x33 => KeyCode::Backspace,
        0x75 => KeyCode::Delete,
        0x72 => KeyCode::Insert,
        0x73 => KeyCode::Home,
        0x77 => KeyCode::End,
        0x74 => KeyCode::PageUp,
        0x79 => KeyCode::PageDown,
        0x7E => KeyCode::ArrowUp,
        0x7D => KeyCode::ArrowDown,
        0x7B => KeyCode::ArrowLeft,
        0x7C => KeyCode::ArrowRight,
        0x38 => KeyCode::ShiftLeft,
        0x3C => KeyCode::ShiftRight,
        0x3B => KeyCode::ControlLeft,
        0x3E => KeyCode::ControlRight,
        0x3A => KeyCode::AltLeft,
        0x3D => KeyCode::AltRight,
        0x37 => KeyCode::MetaLeft,
        0x36 => KeyCode::MetaRight,
        0x3F => KeyCode::Fn,
        0x47 => KeyCode::NumLock,
        0x4B => KeyCode::NumpadDivide,
        0x43 => KeyCode::NumpadMultiply,
        0x4E => KeyCode::NumpadSubtract,
        0x45 => KeyCode::NumpadAdd,
        0x4C => KeyCode::NumpadEnter,
        0x41 => KeyCode::NumpadDecimal,
        0x52 => KeyCode::Numpad0,
        0x53 => KeyCode::Numpad1,
        0x54 => KeyCode::Numpad2,
        0x55 => KeyCode::Numpad3,
        0x56 => KeyCode::Numpad4,
        0x57 => KeyCode::Numpad5,
        0x58 => KeyCode::Numpad6,
        0x59 => KeyCode::Numpad7,
        0x5B => KeyCode::Numpad8,
        0x5C => KeyCode::Numpad9,
        0x7A => KeyCode::F1,
        0x78 => KeyCode::F2,
        0x63 => KeyCode::F3,
        0x76 => KeyCode::F4,
        0x60 => KeyCode::F5,
        0x61 => KeyCode::F6,
        0x62 => KeyCode::F7,
        0x64 => KeyCode::F8,
        0x65 => KeyCode::F9,
        0x6D => KeyCode::F10,
        0x67 => KeyCode::F11,
        0x6F => KeyCode::F12,
        0x5D => KeyCode::IntlYen,
        other => KeyCode::Native(other as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::{Family, key_code, modifiers};
    use codevar_kuitui_control::{KeyCode, Modifiers};

    #[test]
    fn letters_and_digits_follow_the_physical_layout() {
        assert_eq!(key_code(0x00), KeyCode::KeyA);
        assert_eq!(key_code(0x06), KeyCode::KeyZ);
        assert_eq!(key_code(0x12), KeyCode::Digit1);
        assert_eq!(key_code(0x1D), KeyCode::Digit0);
    }

    #[test]
    fn navigation_and_modifiers_are_identified() {
        assert_eq!(key_code(0x7E), KeyCode::ArrowUp);
        assert_eq!(key_code(0x3C), KeyCode::ShiftRight);
        assert_eq!(key_code(0x36), KeyCode::MetaRight);
        assert_eq!(key_code(0x4C), KeyCode::NumpadEnter);
    }

    #[test]
    fn unmapped_keys_keep_the_native_code() {
        assert_eq!(key_code(0x48), KeyCode::Native(0x48));
    }

    #[test]
    fn families_cover_both_sides() {
        assert_eq!(super::family(0x38), Some(Family::Shift));
        assert_eq!(super::family(0x3C), Some(Family::Shift));
        assert_eq!(super::family(0x39), Some(Family::CapsLock));
        assert_eq!(super::family(0x00), None);
    }

    #[test]
    fn flags_become_modifier_bits() {
        assert_eq!(modifiers(0), Modifiers::NONE);
        let set = modifiers(super::EVENT_FLAG_SHIFT | super::EVENT_FLAG_META);
        assert!(set.contains(Modifiers::SHIFT));
        assert!(set.contains(Modifiers::META));
        assert!(!set.contains(Modifiers::CONTROL));
        let locked = modifiers(super::EVENT_FLAG_CAPS_LOCK);
        assert!(locked.contains(Modifiers::CAPS_LOCK));
    }
}
