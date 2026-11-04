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

//! The modifier and lock flags a keyboard reports.
//!
//! [`KeyboardModifiers`] mirrors the control layer's four modifiers and adds
//! the two locks, because a shortcut may require Caps Lock off and a text
//! field needs to know whether the next letter is upper case.

use codevar_kuitui_control::{KeyCode, Modifiers};

/// The bits [`KeyboardModifiers`] calls a modifier.
const MODIFIER_MASK: u8 = 0b0000_1111;

/// The set of modifier keys and lock flags currently active.
///
/// This mirrors [`Modifiers`] from the control layer but keeps two additions
/// that matter to a widget: the flags are a plain `u8` a backend can report in
/// one atomic, and the lock state is separate from the held modifier keys, so
/// a snapshot can tell "shift is down" from "caps lock is on".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KeyboardModifiers(u8);

impl KeyboardModifiers {
    /// No modifier and no lock is active.
    pub const NONE: KeyboardModifiers = KeyboardModifiers(0);
    /// Shift is held.
    pub const SHIFT: KeyboardModifiers = KeyboardModifiers(1 << 0);
    /// Control is held.
    pub const CONTROL: KeyboardModifiers = KeyboardModifiers(1 << 1);
    /// Alt or option is held.
    pub const ALT: KeyboardModifiers = KeyboardModifiers(1 << 2);
    /// Meta, command or the Windows key is held.
    pub const META: KeyboardModifiers = KeyboardModifiers(1 << 3);
    /// Caps lock is on.
    pub const CAPS_LOCK: KeyboardModifiers = KeyboardModifiers(1 << 4);
    /// Num lock is on.
    pub const NUM_LOCK: KeyboardModifiers = KeyboardModifiers(1 << 5);

    /// Whether every flag of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: KeyboardModifiers) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no flag is set.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Adds every flag of `other`.
    #[inline]
    pub fn insert(&mut self, other: KeyboardModifiers) {
        self.0 |= other.0;
    }

    /// Removes every flag of `other`.
    #[inline]
    pub fn remove(&mut self, other: KeyboardModifiers) {
        self.0 &= !other.0;
    }

    /// The raw flag bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Rebuilds the flags from raw bits, dropping bits this version does not
    /// define.
    #[inline]
    #[must_use]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & 0b0011_1111)
    }

    /// The flags without the two lock bits.
    #[inline]
    #[must_use]
    pub const fn without_locks(self) -> Self {
        Self(self.0 & MODIFIER_MASK)
    }

    /// The same flags as the control layer's [`Modifiers`].
    #[must_use]
    pub fn as_control(self) -> Modifiers {
        let mut modifiers = Modifiers::NONE;
        if self.contains(Self::SHIFT) {
            modifiers.insert(Modifiers::SHIFT);
        }
        if self.contains(Self::CONTROL) {
            modifiers.insert(Modifiers::CONTROL);
        }
        if self.contains(Self::ALT) {
            modifiers.insert(Modifiers::ALT);
        }
        if self.contains(Self::META) {
            modifiers.insert(Modifiers::META);
        }
        if self.contains(Self::CAPS_LOCK) {
            modifiers.insert(Modifiers::CAPS_LOCK);
        }
        if self.contains(Self::NUM_LOCK) {
            modifiers.insert(Modifiers::NUM_LOCK);
        }
        modifiers
    }
}

impl core::ops::BitOr for KeyboardModifiers {
    type Output = KeyboardModifiers;

    #[inline]
    fn bitor(self, rhs: KeyboardModifiers) -> KeyboardModifiers {
        KeyboardModifiers(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for KeyboardModifiers {
    #[inline]
    fn bitor_assign(&mut self, rhs: KeyboardModifiers) {
        self.0 |= rhs.0;
    }
}

impl From<Modifiers> for KeyboardModifiers {
    fn from(modifiers: Modifiers) -> Self {
        let mut flags = Self::NONE;
        if modifiers.contains(Modifiers::SHIFT) {
            flags.insert(Self::SHIFT);
        }
        if modifiers.contains(Modifiers::CONTROL) {
            flags.insert(Self::CONTROL);
        }
        if modifiers.contains(Modifiers::ALT) {
            flags.insert(Self::ALT);
        }
        if modifiers.contains(Modifiers::META) {
            flags.insert(Self::META);
        }
        if modifiers.contains(Modifiers::CAPS_LOCK) {
            flags.insert(Self::CAPS_LOCK);
        }
        if modifiers.contains(Modifiers::NUM_LOCK) {
            flags.insert(Self::NUM_LOCK);
        }
        flags
    }
}

/// The modifier flag a key represents, if any.
#[inline]
pub(super) fn modifier_of(code: KeyCode) -> Option<KeyboardModifiers> {
    match code {
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Some(KeyboardModifiers::SHIFT),
        KeyCode::ControlLeft | KeyCode::ControlRight => Some(KeyboardModifiers::CONTROL),
        KeyCode::AltLeft | KeyCode::AltRight => Some(KeyboardModifiers::ALT),
        KeyCode::MetaLeft | KeyCode::MetaRight => Some(KeyboardModifiers::META),
        _ => None,
    }
}
