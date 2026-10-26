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

//! Translation from Linux key codes to the portable key vocabulary.
//!
//! The table covers the keys a desktop editor binds. Anything else keeps
//! its native code in [`KeyCode::Native`] instead of being discarded, so a
//! consumer can still implement a vendor-specific shortcut.

use codevar_kuitui_control::{KeyCode, Modifiers};

/// The portable code of a Linux key.
#[must_use]
pub(crate) fn key_code(code: u16) -> KeyCode {
    match code {
        1 => KeyCode::Escape,
        2 => KeyCode::Digit1,
        3 => KeyCode::Digit2,
        4 => KeyCode::Digit3,
        5 => KeyCode::Digit4,
        6 => KeyCode::Digit5,
        7 => KeyCode::Digit6,
        8 => KeyCode::Digit7,
        9 => KeyCode::Digit8,
        10 => KeyCode::Digit9,
        11 => KeyCode::Digit0,
        12 => KeyCode::Minus,
        13 => KeyCode::Equal,
        14 => KeyCode::Backspace,
        15 => KeyCode::Tab,
        16 => KeyCode::KeyQ,
        17 => KeyCode::KeyW,
        18 => KeyCode::KeyE,
        19 => KeyCode::KeyR,
        20 => KeyCode::KeyT,
        21 => KeyCode::KeyY,
        22 => KeyCode::KeyU,
        23 => KeyCode::KeyI,
        24 => KeyCode::KeyO,
        25 => KeyCode::KeyP,
        26 => KeyCode::BracketLeft,
        27 => KeyCode::BracketRight,
        28 => KeyCode::Enter,
        29 => KeyCode::ControlLeft,
        30 => KeyCode::KeyA,
        31 => KeyCode::KeyS,
        32 => KeyCode::KeyD,
        33 => KeyCode::KeyF,
        34 => KeyCode::KeyG,
        35 => KeyCode::KeyH,
        36 => KeyCode::KeyJ,
        37 => KeyCode::KeyK,
        38 => KeyCode::KeyL,
        39 => KeyCode::Semicolon,
        40 => KeyCode::Quote,
        41 => KeyCode::Backquote,
        42 => KeyCode::ShiftLeft,
        43 => KeyCode::Backslash,
        44 => KeyCode::KeyZ,
        45 => KeyCode::KeyX,
        46 => KeyCode::KeyC,
        47 => KeyCode::KeyV,
        48 => KeyCode::KeyB,
        49 => KeyCode::KeyN,
        50 => KeyCode::KeyM,
        51 => KeyCode::Comma,
        52 => KeyCode::Period,
        53 => KeyCode::Slash,
        54 => KeyCode::ShiftRight,
        55 => KeyCode::NumpadMultiply,
        56 => KeyCode::AltLeft,
        57 => KeyCode::Space,
        58 => KeyCode::CapsLock,
        59 => KeyCode::F1,
        60 => KeyCode::F2,
        61 => KeyCode::F3,
        62 => KeyCode::F4,
        63 => KeyCode::F5,
        64 => KeyCode::F6,
        65 => KeyCode::F7,
        66 => KeyCode::F8,
        67 => KeyCode::F9,
        68 => KeyCode::F10,
        69 => KeyCode::NumLock,
        70 => KeyCode::ScrollLock,
        71 => KeyCode::Numpad7,
        72 => KeyCode::Numpad8,
        73 => KeyCode::Numpad9,
        74 => KeyCode::NumpadSubtract,
        75 => KeyCode::Numpad4,
        76 => KeyCode::Numpad5,
        77 => KeyCode::Numpad6,
        78 => KeyCode::NumpadAdd,
        79 => KeyCode::Numpad1,
        80 => KeyCode::Numpad2,
        81 => KeyCode::Numpad3,
        82 => KeyCode::Numpad0,
        83 => KeyCode::NumpadDecimal,
        86 => KeyCode::IntlBackslash,
        87 => KeyCode::F11,
        88 => KeyCode::F12,
        89 => KeyCode::IntlRo,
        96 => KeyCode::NumpadEnter,
        97 => KeyCode::ControlRight,
        98 => KeyCode::NumpadDivide,
        99 => KeyCode::PrintScreen,
        100 => KeyCode::AltRight,
        102 => KeyCode::Home,
        103 => KeyCode::ArrowUp,
        104 => KeyCode::PageUp,
        105 => KeyCode::ArrowLeft,
        106 => KeyCode::ArrowRight,
        107 => KeyCode::End,
        108 => KeyCode::ArrowDown,
        109 => KeyCode::PageDown,
        110 => KeyCode::Insert,
        111 => KeyCode::Delete,
        119 => KeyCode::Pause,
        124 => KeyCode::IntlYen,
        125 => KeyCode::MetaLeft,
        126 => KeyCode::MetaRight,
        139 => KeyCode::ContextMenu,
        464 => KeyCode::Fn,
        other => KeyCode::Native(u32::from(other)),
    }
}

/// The modifier a Linux key represents, if any.
///
/// Caps lock and num lock are absent: they are lock state that the worker
/// toggles, not a modifier that follows the key's pressed state.
#[must_use]
pub(crate) fn modifier(code: u16) -> Option<Modifiers> {
    match code {
        42 | 54 => Some(Modifiers::SHIFT),
        29 | 97 => Some(Modifiers::CONTROL),
        56 | 100 => Some(Modifiers::ALT),
        125 | 126 => Some(Modifiers::META),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{key_code, modifier};
    use codevar_kuitui_control::{KeyCode, Modifiers};

    #[test]
    fn letters_and_digits_follow_the_linux_codes() {
        assert_eq!(key_code(30), KeyCode::KeyA);
        assert_eq!(key_code(50), KeyCode::KeyM);
        assert_eq!(key_code(2), KeyCode::Digit1);
        assert_eq!(key_code(11), KeyCode::Digit0);
    }

    #[test]
    fn punctuation_and_navigation_are_named() {
        assert_eq!(key_code(41), KeyCode::Backquote);
        assert_eq!(key_code(39), KeyCode::Semicolon);
        assert_eq!(key_code(103), KeyCode::ArrowUp);
        assert_eq!(key_code(111), KeyCode::Delete);
        assert_eq!(key_code(58), KeyCode::CapsLock);
    }

    #[test]
    fn function_and_keypad_keys_are_named() {
        assert_eq!(key_code(59), KeyCode::F1);
        assert_eq!(key_code(88), KeyCode::F12);
        assert_eq!(key_code(82), KeyCode::Numpad0);
        assert_eq!(key_code(96), KeyCode::NumpadEnter);
        assert_eq!(key_code(98), KeyCode::NumpadDivide);
    }

    #[test]
    fn unknown_keys_keep_their_native_code() {
        assert_eq!(key_code(113), KeyCode::Native(113));
        assert_eq!(key_code(240), KeyCode::Native(240));
    }

    #[test]
    fn side_specific_modifiers_are_distinguished() {
        assert_eq!(key_code(29), KeyCode::ControlLeft);
        assert_eq!(key_code(97), KeyCode::ControlRight);
        assert_eq!(modifier(42), Some(Modifiers::SHIFT));
        assert_eq!(modifier(54), Some(Modifiers::SHIFT));
        assert_eq!(modifier(125), Some(Modifiers::META));
    }

    #[test]
    fn lock_and_regular_keys_are_not_modifiers() {
        assert_eq!(modifier(58), None);
        assert_eq!(modifier(69), None);
        assert_eq!(modifier(30), None);
    }
}
