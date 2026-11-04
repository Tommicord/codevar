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

//! Translation of `KeyboardEvent.code` values into [`KeyCode`].
//!
//! The table is ordinary Rust rather than a Web API call, so it stays
//! portable and its tests run on every target, not only in a browser.

use codevar_kuitui_control::KeyCode;

/// Maps a DOM key code to the normalized key.
///
/// The value is the layout-independent `code` of the event, so the result
/// names the physical key rather than the character the active layout
/// produces. A code the table does not cover becomes
/// [`KeyCode::Unidentified`].
#[must_use]
pub(crate) fn key_code(code: &str) -> KeyCode {
    match code {
        "KeyA" => KeyCode::KeyA,
        "KeyB" => KeyCode::KeyB,
        "KeyC" => KeyCode::KeyC,
        "KeyD" => KeyCode::KeyD,
        "KeyE" => KeyCode::KeyE,
        "KeyF" => KeyCode::KeyF,
        "KeyG" => KeyCode::KeyG,
        "KeyH" => KeyCode::KeyH,
        "KeyI" => KeyCode::KeyI,
        "KeyJ" => KeyCode::KeyJ,
        "KeyK" => KeyCode::KeyK,
        "KeyL" => KeyCode::KeyL,
        "KeyM" => KeyCode::KeyM,
        "KeyN" => KeyCode::KeyN,
        "KeyO" => KeyCode::KeyO,
        "KeyP" => KeyCode::KeyP,
        "KeyQ" => KeyCode::KeyQ,
        "KeyR" => KeyCode::KeyR,
        "KeyS" => KeyCode::KeyS,
        "KeyT" => KeyCode::KeyT,
        "KeyU" => KeyCode::KeyU,
        "KeyV" => KeyCode::KeyV,
        "KeyW" => KeyCode::KeyW,
        "KeyX" => KeyCode::KeyX,
        "KeyY" => KeyCode::KeyY,
        "KeyZ" => KeyCode::KeyZ,
        "Digit0" => KeyCode::Digit0,
        "Digit1" => KeyCode::Digit1,
        "Digit2" => KeyCode::Digit2,
        "Digit3" => KeyCode::Digit3,
        "Digit4" => KeyCode::Digit4,
        "Digit5" => KeyCode::Digit5,
        "Digit6" => KeyCode::Digit6,
        "Digit7" => KeyCode::Digit7,
        "Digit8" => KeyCode::Digit8,
        "Digit9" => KeyCode::Digit9,
        "Backquote" => KeyCode::Backquote,
        "Minus" => KeyCode::Minus,
        "Equal" => KeyCode::Equal,
        "BracketLeft" => KeyCode::BracketLeft,
        "BracketRight" => KeyCode::BracketRight,
        "Backslash" => KeyCode::Backslash,
        "Semicolon" => KeyCode::Semicolon,
        "Quote" => KeyCode::Quote,
        "Comma" => KeyCode::Comma,
        "Period" => KeyCode::Period,
        "Slash" => KeyCode::Slash,
        "IntlBackslash" => KeyCode::IntlBackslash,
        "IntlRo" => KeyCode::IntlRo,
        "IntlYen" => KeyCode::IntlYen,
        "Escape" => KeyCode::Escape,
        "Tab" => KeyCode::Tab,
        "CapsLock" => KeyCode::CapsLock,
        "Space" => KeyCode::Space,
        "Enter" => KeyCode::Enter,
        "Backspace" => KeyCode::Backspace,
        "Delete" => KeyCode::Delete,
        "Insert" => KeyCode::Insert,
        "Home" => KeyCode::Home,
        "End" => KeyCode::End,
        "PageUp" => KeyCode::PageUp,
        "PageDown" => KeyCode::PageDown,
        "ArrowUp" => KeyCode::ArrowUp,
        "ArrowDown" => KeyCode::ArrowDown,
        "ArrowLeft" => KeyCode::ArrowLeft,
        "ArrowRight" => KeyCode::ArrowRight,
        "ShiftLeft" => KeyCode::ShiftLeft,
        "ShiftRight" => KeyCode::ShiftRight,
        "ControlLeft" => KeyCode::ControlLeft,
        "ControlRight" => KeyCode::ControlRight,
        "AltLeft" => KeyCode::AltLeft,
        "AltRight" => KeyCode::AltRight,
        "MetaLeft" => KeyCode::MetaLeft,
        "MetaRight" => KeyCode::MetaRight,
        "ContextMenu" => KeyCode::ContextMenu,
        "PrintScreen" => KeyCode::PrintScreen,
        "ScrollLock" => KeyCode::ScrollLock,
        "Pause" => KeyCode::Pause,
        "Fn" => KeyCode::Fn,
        "NumLock" => KeyCode::NumLock,
        "NumpadDivide" => KeyCode::NumpadDivide,
        "NumpadMultiply" => KeyCode::NumpadMultiply,
        "NumpadSubtract" => KeyCode::NumpadSubtract,
        "NumpadAdd" => KeyCode::NumpadAdd,
        "NumpadEnter" => KeyCode::NumpadEnter,
        "NumpadDecimal" => KeyCode::NumpadDecimal,
        "Numpad0" => KeyCode::Numpad0,
        "Numpad1" => KeyCode::Numpad1,
        "Numpad2" => KeyCode::Numpad2,
        "Numpad3" => KeyCode::Numpad3,
        "Numpad4" => KeyCode::Numpad4,
        "Numpad5" => KeyCode::Numpad5,
        "Numpad6" => KeyCode::Numpad6,
        "Numpad7" => KeyCode::Numpad7,
        "Numpad8" => KeyCode::Numpad8,
        "Numpad9" => KeyCode::Numpad9,
        "F1" => KeyCode::F1,
        "F2" => KeyCode::F2,
        "F3" => KeyCode::F3,
        "F4" => KeyCode::F4,
        "F5" => KeyCode::F5,
        "F6" => KeyCode::F6,
        "F7" => KeyCode::F7,
        "F8" => KeyCode::F8,
        "F9" => KeyCode::F9,
        "F10" => KeyCode::F10,
        "F11" => KeyCode::F11,
        "F12" => KeyCode::F12,
        _ => KeyCode::Unidentified,
    }
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::KeyCode;

    use super::key_code;

    #[test]
    fn letters_and_digits_keep_their_dom_names() {
        assert_eq!(key_code("KeyA"), KeyCode::KeyA);
        assert_eq!(key_code("KeyZ"), KeyCode::KeyZ);
        assert_eq!(key_code("Digit0"), KeyCode::Digit0);
        assert_eq!(key_code("Digit9"), KeyCode::Digit9);
    }

    #[test]
    fn modifiers_and_navigation_are_named() {
        assert_eq!(key_code("ShiftRight"), KeyCode::ShiftRight);
        assert_eq!(key_code("ControlLeft"), KeyCode::ControlLeft);
        assert_eq!(key_code("MetaLeft"), KeyCode::MetaLeft);
        assert_eq!(key_code("ArrowUp"), KeyCode::ArrowUp);
        assert_eq!(key_code("PageDown"), KeyCode::PageDown);
    }

    #[test]
    fn keypads_and_function_keys_are_named() {
        assert_eq!(key_code("NumpadEnter"), KeyCode::NumpadEnter);
        assert_eq!(key_code("Numpad5"), KeyCode::Numpad5);
        assert_eq!(key_code("F12"), KeyCode::F12);
        assert_eq!(key_code("IntlBackslash"), KeyCode::IntlBackslash);
    }

    #[test]
    fn an_unknown_code_is_reported_as_unidentified() {
        assert_eq!(key_code(""), KeyCode::Unidentified);
        assert_eq!(key_code("Unidentified"), KeyCode::Unidentified);
        assert_eq!(key_code("VolumeUp"), KeyCode::Unidentified);
    }
}
