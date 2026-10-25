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

//! The dense table of normalized key codes and the packed encoding used by
//! the "last key" atomics.
//!
//! A normalized key has a fixed slot in [`KEY_SLOTS`], so the pressed set is a
//! fixed-size bitset and a lookup is a binary search over a constant array.
//! A native code does not belong to the table; it is stored in its own set and
//! marked with [`NATIVE_FLAG`] when it is packed into one `u64`.

use codevar_kuitui_control::KeyCode;

/// The number of normalized key codes [`KeyboardState`] can track at once.
///
/// Every variant of [`KeyCode`] except [`KeyCode::Native`] has a dense slot;
/// the table is a constant so a lookup needs no allocation.
pub(super) const KEY_SLOTS: [KeyCode; 109] = [
    KeyCode::Unidentified,
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
    KeyCode::Backquote,
    KeyCode::Minus,
    KeyCode::Equal,
    KeyCode::BracketLeft,
    KeyCode::BracketRight,
    KeyCode::Backslash,
    KeyCode::Semicolon,
    KeyCode::Quote,
    KeyCode::Comma,
    KeyCode::Period,
    KeyCode::Slash,
    KeyCode::IntlBackslash,
    KeyCode::IntlRo,
    KeyCode::IntlYen,
    KeyCode::Escape,
    KeyCode::Tab,
    KeyCode::CapsLock,
    KeyCode::Space,
    KeyCode::Enter,
    KeyCode::Backspace,
    KeyCode::Delete,
    KeyCode::Insert,
    KeyCode::Home,
    KeyCode::End,
    KeyCode::PageUp,
    KeyCode::PageDown,
    KeyCode::ArrowUp,
    KeyCode::ArrowDown,
    KeyCode::ArrowLeft,
    KeyCode::ArrowRight,
    KeyCode::ShiftLeft,
    KeyCode::ShiftRight,
    KeyCode::ControlLeft,
    KeyCode::ControlRight,
    KeyCode::AltLeft,
    KeyCode::AltRight,
    KeyCode::MetaLeft,
    KeyCode::MetaRight,
    KeyCode::ContextMenu,
    KeyCode::PrintScreen,
    KeyCode::ScrollLock,
    KeyCode::Pause,
    KeyCode::Fn,
    KeyCode::NumLock,
    KeyCode::NumpadDivide,
    KeyCode::NumpadMultiply,
    KeyCode::NumpadSubtract,
    KeyCode::NumpadAdd,
    KeyCode::NumpadEnter,
    KeyCode::NumpadDecimal,
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

/// Marks an encoded last-key value as a native code rather than a slot.
const NATIVE_FLAG: u64 = 1 << 40;

/// The slot of a normalized key, or `None` for a native code.
#[inline]
pub(super) fn slot(code: KeyCode) -> Option<u8> {
    KEY_SLOTS
        .binary_search(&code)
        .ok()
        .and_then(|index| u8::try_from(index).ok())
}

/// The normalized key of `slot`.
#[inline]
pub(super) fn code_at(slot: u8) -> KeyCode {
    KEY_SLOTS
        .get(usize::from(slot))
        .copied()
        .unwrap_or(KeyCode::Unidentified)
}

/// Packs a key into the value of one of the "last" atomics.
#[inline]
pub(super) fn encode_key(code: KeyCode) -> u64 {
    match code {
        KeyCode::Native(value) => NATIVE_FLAG | u64::from(value),
        other => slot(other).map_or(0, |slot| u64::from(slot) + 1),
    }
}

/// Unpacks one of the "last" atomics.
#[inline]
pub(super) fn decode_key(value: u64) -> Option<KeyCode> {
    if value == 0 {
        return None;
    }
    if value & NATIVE_FLAG != 0 {
        return u32::try_from(value & (NATIVE_FLAG - 1))
            .ok()
            .map(KeyCode::Native);
    }
    u8::try_from(value - 1).ok().map(code_at)
}
