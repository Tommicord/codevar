//! Copyright 2026 Codevar
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

//! 8-Byte Bit-Packed Action Format
//!
//! Bit layout (64 bits total):
//! ```text
//!  63............56 55........................33 32 31....................0
//! +----------------+------------------------------+--+-----------------------+
//! |  extra_info    |           count              | f|      action_key       |
//! |   (8 bits)     |           (23 bits)          |  |      (32 bits)        |
//! +----------------+------------------------------+--+-----------------------+
//! ```
//!
//! - action_key (32 bits): Encodes UserActionType + character/code
//! - flag (1 bit): Composite/repeat/modifier flag
//! - count (23 bits): Repeat count (max 8,388,607)
//! - extra_info (8 bits): Cursor ID mod 256, modifier mask, or context

use super::super::wredit_observer::{UserActionEvent, UserActionType};
use std::fmt;

/// 32-bit action key encoding action type and character
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ActionKey(pub u32);

impl ActionKey {
    /// Maximum value for action type (5 bits)
    const MAX_ACTION_TYPE: u32 = 0x1F;
    /// Maximum value for character (27 bits, supports full Unicode)
    const MAX_CHARACTER: u32 = 0x07FF_FFFF;

    /// Create from action type and optional character
    pub fn new(action_type: UserActionType, character: Option<u32>) -> Self {
        let type_bits = (action_type as u32) & Self::MAX_ACTION_TYPE;
        let char_bits = character.unwrap_or(0) & Self::MAX_CHARACTER;
        Self((type_bits << 27) | char_bits)
    }

    /// Create from raw u32
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Get raw u32 value
    pub const fn as_raw(&self) -> u32 {
        self.0
    }

    /// Extract action type (5 bits)
    pub fn action_type(&self) -> UserActionType {
        let type_val = (self.0 >> 27) & Self::MAX_ACTION_TYPE;
        match type_val {
            0 => UserActionType::InsertChar,
            1 => UserActionType::Delete,
            2 => UserActionType::Backspace,
            3 => UserActionType::MoveForward,
            4 => UserActionType::MoveBackward,
            5 => UserActionType::MoveNextLine,
            6 => UserActionType::MovePrevLine,
            7 => UserActionType::MoveLineStart,
            8 => UserActionType::MoveLineEnd,
            9 => UserActionType::Undo,
            10 => UserActionType::Redo,
            11 => UserActionType::BeginEditGroup,
            12 => UserActionType::EndEditGroup,
            13 => UserActionType::FileOpened,
            14 => UserActionType::FileClosed,
            15 => UserActionType::SelectionChanged,
            _ => UserActionType::InsertChar,
        }
    }

    /// Extract character (27 bits)
    pub fn character(&self) -> Option<u32> {
        let char_val = self.0 & Self::MAX_CHARACTER;
        if char_val == 0 { None } else { Some(char_val) }
    }

    /// Check if this is a character insertion
    pub fn is_insert(&self) -> bool {
        self.action_type() == UserActionType::InsertChar
    }

    /// Check if this is a navigation action
    pub fn is_navigation(&self) -> bool {
        matches!(
            self.action_type(),
            UserActionType::MoveForward
                | UserActionType::MoveBackward
                | UserActionType::MoveNextLine
                | UserActionType::MovePrevLine
                | UserActionType::MoveLineStart
                | UserActionType::MoveLineEnd
        )
    }

    /// Check if this is a deletion action
    pub fn is_deletion(&self) -> bool {
        matches!(
            self.action_type(),
            UserActionType::Delete | UserActionType::Backspace
        )
    }

    /// Check if this is an edit group action
    pub fn is_edit_group(&self) -> bool {
        matches!(
            self.action_type(),
            UserActionType::BeginEditGroup | UserActionType::EndEditGroup
        )
    }

    /// Check if this is a file operation
    pub fn is_file_op(&self) -> bool {
        matches!(
            self.action_type(),
            UserActionType::FileOpened | UserActionType::FileClosed
        )
    }
}

impl From<UserActionEvent> for ActionKey {
    fn from(event: UserActionEvent) -> Self {
        Self::new(event.action_type, event.character)
    }
}

impl fmt::Display for ActionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ActionKey(type={:?}, char={:?})",
            self.action_type(),
            self.character()
        )
    }
}

/// 1-bit action flag
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionFlag {
    /// Normal action
    Normal = 0,
    /// Composite/repeated action
    Composite = 1,
}

impl ActionFlag {
    pub fn from_bool(composite: bool) -> Self {
        if composite {
            Self::Composite
        } else {
            Self::Normal
        }
    }

    pub fn is_composite(&self) -> bool {
        matches!(self, Self::Composite)
    }

    pub fn as_bit(&self) -> u64 {
        *self as u64
    }
}

/// 8-bit extra info field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ActionExtraInfo(pub u8);

impl ActionExtraInfo {
    /// Create from cursor ID (mod 256)
    pub fn from_cursor_id(cursor_id: usize) -> Self {
        Self((cursor_id & 0xFF) as u8)
    }

    /// Create from modifier mask
    pub fn from_modifiers(shift: bool, ctrl: bool, alt: bool, meta: bool) -> Self {
        let mut bits = 0u8;
        if shift {
            bits |= 0x01;
        }
        if ctrl {
            bits |= 0x02;
        }
        if alt {
            bits |= 0x04;
        }
        if meta {
            bits |= 0x08;
        }
        Self(bits)
    }

    /// Create from raw byte
    pub const fn from_raw(raw: u8) -> Self {
        Self(raw)
    }

    /// Get raw byte
    pub const fn as_raw(&self) -> u8 {
        self.0
    }

    /// Get cursor ID (lower 8 bits)
    pub fn cursor_id(&self) -> u8 {
        self.0
    }

    /// Check modifier flags
    pub fn has_shift(&self) -> bool {
        self.0 & 0x01 != 0
    }
    pub fn has_ctrl(&self) -> bool {
        self.0 & 0x02 != 0
    }
    pub fn has_alt(&self) -> bool {
        self.0 & 0x04 != 0
    }
    pub fn has_meta(&self) -> bool {
        self.0 & 0x08 != 0
    }

    /// Get modifier mask
    pub fn modifier_mask(&self) -> u8 {
        self.0 & 0x0F
    }
}

impl Default for ActionExtraInfo {
    fn default() -> Self {
        Self(0)
    }
}

/// Fully packed 8-byte action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct PackedAction {
    data: u64,
}

impl PackedAction {
    /// Bit positions
    const ACTION_KEY_SHIFT: u64 = 0;
    const FLAG_SHIFT: u64 = 32;
    const COUNT_SHIFT: u64 = 33;
    const EXTRA_INFO_SHIFT: u64 = 56;

    /// Bit masks
    const ACTION_KEY_MASK: u64 = 0xFFFF_FFFF;
    const FLAG_MASK: u64 = 0x1;
    const COUNT_MASK: u64 = 0x7F_FFFF; // 23 bits
    const EXTRA_INFO_MASK: u64 = 0xFF;

    /// Maximum count value (23 bits)
    pub const MAX_COUNT: u32 = 0x7F_FFFF;

    /// Create a new packed action
    pub fn new(
        action_key: ActionKey,
        flag: ActionFlag,
        count: u32,
        extra_info: ActionExtraInfo,
    ) -> Self {
        let count = count.min(Self::MAX_COUNT);
        let data = ((action_key.as_raw() as u64) & Self::ACTION_KEY_MASK)
            << Self::ACTION_KEY_SHIFT
            | (flag.as_bit() & Self::FLAG_MASK) << Self::FLAG_SHIFT
            | ((count as u64) & Self::COUNT_MASK) << Self::COUNT_SHIFT
            | ((extra_info.as_raw() as u64) & Self::EXTRA_INFO_MASK)
                << Self::EXTRA_INFO_SHIFT;
        Self { data }
    }

    /// Create from UserActionEvent with count=1
    pub fn from_event(event: &UserActionEvent, cursor_id: usize) -> Self {
        Self::new(
            ActionKey::from(event.clone()),
            ActionFlag::Normal,
            1,
            ActionExtraInfo::from_cursor_id(cursor_id),
        )
    }

    /// Create from UserActionEvent with custom count
    pub fn from_event_with_count(
        event: &UserActionEvent,
        count: u32,
        cursor_id: usize,
    ) -> Self {
        Self::new(
            ActionKey::from(event.clone()),
            ActionFlag::Normal,
            count,
            ActionExtraInfo::from_cursor_id(cursor_id),
        )
    }

    /// Create composite action (merged repeats)
    pub fn composite(
        action_key: ActionKey,
        count: u32,
        extra_info: ActionExtraInfo,
    ) -> Self {
        Self::new(action_key, ActionFlag::Composite, count, extra_info)
    }

    /// Create empty/invalid action
    pub const fn empty() -> Self {
        Self { data: 0 }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data == 0
    }

    /// Get raw u64
    pub const fn as_raw(&self) -> u64 {
        self.data
    }

    /// Extract action key
    pub fn action_key(&self) -> ActionKey {
        ActionKey::from_raw(
            ((self.data >> Self::ACTION_KEY_SHIFT) & Self::ACTION_KEY_MASK) as u32,
        )
    }

    /// Extract flag
    pub fn flag(&self) -> ActionFlag {
        if ((self.data >> Self::FLAG_SHIFT) & Self::FLAG_MASK) != 0 {
            ActionFlag::Composite
        } else {
            ActionFlag::Normal
        }
    }

    /// Extract count
    pub fn count(&self) -> u32 {
        ((self.data >> Self::COUNT_SHIFT) & Self::COUNT_MASK) as u32
    }

    /// Extract extra info
    pub fn extra_info(&self) -> ActionExtraInfo {
        ActionExtraInfo::from_raw(
            ((self.data >> Self::EXTRA_INFO_SHIFT) & Self::EXTRA_INFO_MASK) as u8,
        )
    }

    /// Increment count (saturating)
    pub fn increment_count(&mut self) -> bool {
        let current = self.count();
        if current < Self::MAX_COUNT {
            let new_data = (self.data & !(Self::COUNT_MASK << Self::COUNT_SHIFT))
                | (((current + 1) as u64) & Self::COUNT_MASK) << Self::COUNT_SHIFT;
            self.data = new_data;
            true
        } else {
            false
        }
    }

    /// Set count
    pub fn set_count(&mut self, count: u32) {
        let count = count.min(Self::MAX_COUNT);
        self.data = (self.data & !(Self::COUNT_MASK << Self::COUNT_SHIFT))
            | ((count as u64) & Self::COUNT_MASK) << Self::COUNT_SHIFT;
    }

    /// Set flag
    pub fn set_flag(&mut self, flag: ActionFlag) {
        self.data = (self.data & !(Self::FLAG_MASK << Self::FLAG_SHIFT))
            | (flag.as_bit() & Self::FLAG_MASK) << Self::FLAG_SHIFT;
    }

    /// Set extra info
    pub fn set_extra_info(&mut self, extra: ActionExtraInfo) {
        self.data = (self.data & !(Self::EXTRA_INFO_MASK << Self::EXTRA_INFO_SHIFT))
            | ((extra.as_raw() as u64) & Self::EXTRA_INFO_MASK) << Self::EXTRA_INFO_SHIFT;
    }

    /// Convert to byte array (little-endian)
    pub fn to_bytes(&self) -> [u8; 8] {
        self.data.to_le_bytes()
    }

    /// Create from byte array (little-endian)
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self {
            data: u64::from_le_bytes(bytes),
        }
    }

    /// Create from byte slice (must be 8 bytes)
    pub fn from_slice(slice: &[u8]) -> Option<Self> {
        if slice.len() == 8 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(slice);
            Some(Self::from_bytes(bytes))
        } else {
            None
        }
    }
}

impl Default for PackedAction {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for PackedAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "PackedAction(empty)")
        } else {
            write!(
                f,
                "PackedAction(key={}, flag={:?}, count={}, extra={:02X})",
                self.action_key(),
                self.flag(),
                self.count(),
                self.extra_info().as_raw()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action_key_roundtrip() {
        let key = ActionKey::new(UserActionType::InsertChar, Some('a' as u32));
        assert_eq!(key.action_type(), UserActionType::InsertChar);
        assert_eq!(key.character(), Some('a' as u32));
    }

    fn action_key_navigation() {
        let key = ActionKey::new(UserActionType::MoveForward, None);
        assert!(key.is_navigation());
        assert!(!key.is_insert());
    }

    fn packed_action_roundtrip() {
        let key = ActionKey::new(UserActionType::InsertChar, Some('x' as u32));
        let packed = PackedAction::new(
            key,
            ActionFlag::Normal,
            5,
            ActionExtraInfo::from_cursor_id(42),
        );

        assert_eq!(packed.action_key(), key);
        assert_eq!(packed.flag(), ActionFlag::Normal);
        assert_eq!(packed.count(), 5);
        assert_eq!(packed.extra_info().cursor_id(), 42);
    }

    fn packed_action_bytes() {
        let packed = PackedAction::new(
            ActionKey::new(UserActionType::Delete, None),
            ActionFlag::Composite,
            100,
            ActionExtraInfo::from_modifiers(true, false, true, false),
        );
        let bytes = packed.to_bytes();
        let restored = PackedAction::from_bytes(bytes);
        assert_eq!(packed, restored);
    }

    fn packed_action_increment() {
        let mut packed = PackedAction::new(
            ActionKey::new(UserActionType::MoveForward, None),
            ActionFlag::Normal,
            1,
            ActionExtraInfo::default(),
        );
        assert!(packed.increment_count());
        assert_eq!(packed.count(), 2);

        // Saturate at max
        packed.set_count(PackedAction::MAX_COUNT);
        assert!(!packed.increment_count());
        assert_eq!(packed.count(), PackedAction::MAX_COUNT);
    }

    fn composite_action() {
        let key = ActionKey::new(UserActionType::InsertChar, Some('a' as u32));
        let composite =
            PackedAction::composite(key, 50, ActionExtraInfo::from_cursor_id(1));
        assert_eq!(composite.flag(), ActionFlag::Composite);
        assert_eq!(composite.count(), 50);
    }
}
