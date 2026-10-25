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

//! Translation from Win32 virtual-key codes to the portable key vocabulary.
//!
//! Raw Input reports a virtual-key code, a scan code, and whether the key
//! arrived through the extended (E0) prefix. The scan code and the prefix are
//! needed because the left and right variants of shift, control, and alt share
//! one virtual-key code. Anything outside the table keeps its virtual-key code
//! in [`KeyCode::Native`], so a consumer can still bind a vendor key.

use codevar_kuitui_control::{KeyCode, Modifiers};

/// The physical key that produced a virtual-key code.
///
/// `make_code` is the scan code from the raw record and `extended` is set when
/// the record carried the E0 prefix, which separates the right-hand modifiers
/// and the keypad enter from their main-block counterparts.
#[must_use]
pub(crate) fn key_code(vk: u16, make_code: u16, extended: bool) -> KeyCode {
    match vk {
        // Letters and digits are contiguous in the virtual-key encoding.
        0x41..=0x5A => letter(vk),
        0x30..=0x39 => digit(vk),
        0x60..=0x69 => number_pad(vk),
        0x70..=0x7B => function(vk),
        0x08 => KeyCode::Backspace,
        0x09 => KeyCode::Tab,
        0x0D if extended => KeyCode::NumpadEnter,
        0x0D => KeyCode::Enter,
        0x10 if make_code == 0x36 => KeyCode::ShiftRight,
        0x10 => KeyCode::ShiftLeft,
        0x11 if extended => KeyCode::ControlRight,
        0x11 => KeyCode::ControlLeft,
        0x12 if extended => KeyCode::AltRight,
        0x12 => KeyCode::AltLeft,
        0x13 => KeyCode::Pause,
        0x14 => KeyCode::CapsLock,
        0x1B => KeyCode::Escape,
        0x20 => KeyCode::Space,
        0x21 => KeyCode::PageUp,
        0x22 => KeyCode::PageDown,
        0x23 => KeyCode::End,
        0x24 => KeyCode::Home,
        0x25 => KeyCode::ArrowLeft,
        0x26 => KeyCode::ArrowUp,
        0x27 => KeyCode::ArrowRight,
        0x28 => KeyCode::ArrowDown,
        0x2C => KeyCode::PrintScreen,
        0x2D => KeyCode::Insert,
        0x2E => KeyCode::Delete,
        0x5B => KeyCode::MetaLeft,
        0x5C => KeyCode::MetaRight,
        0x5D => KeyCode::ContextMenu,
        0x6A => KeyCode::NumpadMultiply,
        0x6B => KeyCode::NumpadAdd,
        0x6D => KeyCode::NumpadSubtract,
        0x6E => KeyCode::NumpadDecimal,
        0x6F => KeyCode::NumpadDivide,
        0x90 => KeyCode::NumLock,
        0x91 => KeyCode::ScrollLock,
        0xA0 => KeyCode::ShiftLeft,
        0xA1 => KeyCode::ShiftRight,
        0xA2 => KeyCode::ControlLeft,
        0xA3 => KeyCode::ControlRight,
        0xA4 => KeyCode::AltLeft,
        0xA5 => KeyCode::AltRight,
        0xBA => KeyCode::Semicolon,
        0xBB => KeyCode::Equal,
        0xBC => KeyCode::Comma,
        0xBD => KeyCode::Minus,
        0xBE => KeyCode::Period,
        0xBF => KeyCode::Slash,
        0xC0 => KeyCode::Backquote,
        0xDB => KeyCode::BracketLeft,
        0xDC => KeyCode::Backslash,
        0xDD => KeyCode::BracketRight,
        0xDE => KeyCode::Quote,
        0xE2 => KeyCode::IntlBackslash,
        other => KeyCode::Native(u32::from(other)),
    }
}

/// The modifier held by a virtual-key code, if it is a modifier.
#[must_use]
pub(crate) fn modifier(vk: u16) -> Option<Modifiers> {
    match vk {
        0x10 | 0xA0 | 0xA1 => Some(Modifiers::SHIFT),
        0x11 | 0xA2 | 0xA3 => Some(Modifiers::CONTROL),
        0x12 | 0xA4 | 0xA5 => Some(Modifiers::ALT),
        0x5B | 0x5C => Some(Modifiers::META),
        _ => None,
    }
}

/// A letter from the `VK_A`..=`VK_Z` block.
fn letter(vk: u16) -> KeyCode {
    const LETTERS: [KeyCode; 26] = [
        KeyCode::KeyA,
        KeyCode::KeyB,
        KeyCode::KeyC,
        KeyCode::KeyD,
        KeyCode::KeyE,
        KeyCode::KeyF,
        KeyCode::KeyG,
        KeyCode::KeyH,
        KeyCode::KeyI,
        KeyCode::KeyJ,
        KeyCode::KeyK,
        KeyCode::KeyL,
        KeyCode::KeyM,
        KeyCode::KeyN,
        KeyCode::KeyO,
        KeyCode::KeyP,
        KeyCode::KeyQ,
        KeyCode::KeyR,
        KeyCode::KeyS,
        KeyCode::KeyT,
        KeyCode::KeyU,
        KeyCode::KeyV,
        KeyCode::KeyW,
        KeyCode::KeyX,
        KeyCode::KeyY,
        KeyCode::KeyZ,
    ];
    LETTERS
        .get(usize::from(vk - 0x41))
        .copied()
        .unwrap_or(KeyCode::Unidentified)
}

/// A digit from the `VK_0`..=`VK_9` block.
fn digit(vk: u16) -> KeyCode {
    const DIGITS: [KeyCode; 10] = [
        KeyCode::Digit0,
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    DIGITS
        .get(usize::from(vk - 0x30))
        .copied()
        .unwrap_or(KeyCode::Unidentified)
}

/// A digit from the `VK_NUMPAD0`..=`VK_NUMPAD9` block.
fn number_pad(vk: u16) -> KeyCode {
    const NUMPAD: [KeyCode; 10] = [
        KeyCode::Numpad0,
        KeyCode::Numpad1,
        KeyCode::Numpad2,
        KeyCode::Numpad3,
        KeyCode::Numpad4,
        KeyCode::Numpad5,
        KeyCode::Numpad6,
        KeyCode::Numpad7,
        KeyCode::Numpad8,
        KeyCode::Numpad9,
    ];
    NUMPAD
        .get(usize::from(vk - 0x60))
        .copied()
        .unwrap_or(KeyCode::Unidentified)
}

/// A function key from the `VK_F1`..=`VK_F12` block.
fn function(vk: u16) -> KeyCode {
    const FUNCTIONS: [KeyCode; 12] = [
        KeyCode::F1,
        KeyCode::F2,
        KeyCode::F3,
        KeyCode::F4,
        KeyCode::F5,
        KeyCode::F6,
        KeyCode::F7,
        KeyCode::F8,
        KeyCode::F9,
        KeyCode::F10,
        KeyCode::F11,
        KeyCode::F12,
    ];
    FUNCTIONS
        .get(usize::from(vk - 0x70))
        .copied()
        .unwrap_or(KeyCode::Unidentified)
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::{KeyCode, Modifiers};

    use super::{key_code, modifier};

    #[test]
    fn letters_and_digits_follow_their_blocks() {
        assert_eq!(key_code(0x41, 0x1E, false), KeyCode::KeyA);
        assert_eq!(key_code(0x5A, 0x2C, false), KeyCode::KeyZ);
        assert_eq!(key_code(0x30, 0x0B, false), KeyCode::Digit0);
        assert_eq!(key_code(0x39, 0x0A, false), KeyCode::Digit9);
    }

    #[test]
    fn punctuation_keeps_its_positional_name() {
        assert_eq!(key_code(0xC0, 0x29, false), KeyCode::Backquote);
        assert_eq!(key_code(0xBA, 0x27, false), KeyCode::Semicolon);
        assert_eq!(key_code(0xDB, 0x1A, false), KeyCode::BracketLeft);
        assert_eq!(key_code(0xDC, 0x2B, false), KeyCode::Backslash);
        assert_eq!(key_code(0xDE, 0x28, false), KeyCode::Quote);
        assert_eq!(key_code(0xBF, 0x35, false), KeyCode::Slash);
    }

    #[test]
    fn scan_code_and_prefix_split_the_shared_modifiers() {
        assert_eq!(key_code(0x10, 0x2A, false), KeyCode::ShiftLeft);
        assert_eq!(key_code(0x10, 0x36, false), KeyCode::ShiftRight);
        assert_eq!(key_code(0x11, 0x1D, false), KeyCode::ControlLeft);
        assert_eq!(key_code(0x11, 0x1D, true), KeyCode::ControlRight);
        assert_eq!(key_code(0x12, 0x38, false), KeyCode::AltLeft);
        assert_eq!(key_code(0x12, 0x38, true), KeyCode::AltRight);
        assert_eq!(key_code(0x0D, 0x1C, true), KeyCode::NumpadEnter);
        assert_eq!(key_code(0x0D, 0x1C, false), KeyCode::Enter);
    }

    #[test]
    fn keypad_and_function_keys_are_named() {
        assert_eq!(key_code(0x60, 0x52, false), KeyCode::Numpad0);
        assert_eq!(key_code(0x69, 0x49, false), KeyCode::Numpad9);
        assert_eq!(key_code(0x6B, 0x4E, false), KeyCode::NumpadAdd);
        assert_eq!(key_code(0x70, 0x3B, false), KeyCode::F1);
        assert_eq!(key_code(0x7B, 0x58, false), KeyCode::F12);
    }

    #[test]
    fn an_unknown_code_stays_available_as_a_native_key() {
        assert_eq!(key_code(0xFF, 0, false), KeyCode::Native(0xFF));
    }

    #[test]
    fn only_modifier_keys_report_a_modifier() {
        assert_eq!(modifier(0x10), Some(Modifiers::SHIFT));
        assert_eq!(modifier(0xA3), Some(Modifiers::CONTROL));
        assert_eq!(modifier(0x12), Some(Modifiers::ALT));
        assert_eq!(modifier(0x5C), Some(Modifiers::META));
        assert_eq!(modifier(0x41), None);
    }
}
