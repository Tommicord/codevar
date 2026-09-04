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

//! Action caching system for user view tracking.
//!
//! This module implements an efficient 8-byte action cache that compresses
//! repeated user actions by storing:
//! - 32 bits: action key (encodes UserActionType + character/code)
//! - 1 bit: flag bit (composite/repeat/modifier flag)
//! - 23 bits: repetition count (max 8,388,607)
//! - 8 bits: additional metadata (cursor ID mod 256, modifier mask, or context)
//!
//! Bit layout (64 bits total):
//! ```text
//!  63............56 55........................33 32 31....................0
//! +----------------+------------------------------+--+-----------------------+
//! |  extra_info    |           count              | f|      action_key       |
//! |   (8 bits)     |           (23 bits)          |  |      (32 bits)        |
//! +----------------+------------------------------+--+-----------------------+
//! ```

use codevar_core::edit::{UserActionEvent, UserActionType};
use std::collections::HashMap;

/// Encodes action type and character into a 32-bit action key.
///
/// # Arguments
///
/// * `action_type` - The user action type
/// * `character` - Optional character value
///
/// # Returns
///
/// 32-bit action key
pub fn encode_action_key(action_type: UserActionType, character: Option<u32>) -> u32 {
    const MAX_ACTION_TYPE: u32 = 0x1F; // 5 bits
    const MAX_CHARACTER: u32 = 0x07FF_FFFF; // 27 bits

    let type_bits = (action_type as u32) & MAX_ACTION_TYPE;
    let char_bits = character.unwrap_or(0) & MAX_CHARACTER;
    (type_bits << 27) | char_bits
}

/// 8-byte packed action cache entry.
///
/// Format (64 bits total):
/// - Bits 0-31 (32 bits): Action key (encodes UserActionType + character)
/// - Bit 32 (1 bit): Flag bit (composite/repeat/modifier flag)
/// - Bits 33-55 (23 bits): Repetition count (max 8,388,607)
/// - Bits 56-63 (8 bits): Additional metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PackedAction(u64);

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

    /// Creates a new packed action from components.
    ///
    /// # Arguments
    ///
    /// * `action_key` - 32-bit action key (encodes UserActionType + character)
    /// * `flag` - 1-bit flag (composite/repeat/modifier flag)
    /// * `count` - 23-bit repetition count (must be <= MAX_COUNT)
    /// * `extra_info` - 8-bit additional metadata
    ///
    /// # Returns
    ///
    /// A new PackedAction instance.
    pub fn new(action_key: u32, flag: bool, count: u32, extra_info: u8) -> Self {
        let count = count.min(Self::MAX_COUNT);
        let data = ((action_key as u64) & Self::ACTION_KEY_MASK)
            << Self::ACTION_KEY_SHIFT
            | ((if flag { 1 } else { 0 }) & Self::FLAG_MASK) << Self::FLAG_SHIFT
            | ((count as u64) & Self::COUNT_MASK) << Self::COUNT_SHIFT
            | ((extra_info as u64) & Self::EXTRA_INFO_MASK) << Self::EXTRA_INFO_SHIFT;

        Self(data)
    }

    /// Extracts the action key (32 bits).
    pub fn action_key(&self) -> u32 {
        ((self.0 >> Self::ACTION_KEY_SHIFT) & Self::ACTION_KEY_MASK) as u32
    }

    /// Extracts the flag bit (1 bit).
    pub fn flag(&self) -> bool {
        ((self.0 >> Self::FLAG_SHIFT) & Self::FLAG_MASK) != 0
    }

    /// Extracts the repetition count (23 bits).
    pub fn count(&self) -> u32 {
        ((self.0 >> Self::COUNT_SHIFT) & Self::COUNT_MASK) as u32
    }

    /// Extracts the additional metadata (8 bits).
    pub fn extra_info(&self) -> u8 {
        ((self.0 >> Self::EXTRA_INFO_SHIFT) & Self::EXTRA_INFO_MASK) as u8
    }

    /// Returns the raw packed value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Creates a packed action from a raw u64 value.
    pub fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Creates a packed action from a UserActionEvent.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event
    /// * `cursor_id` - The cursor ID for extra_info
    ///
    /// # Returns
    ///
    /// A new PackedAction instance.
    pub fn from_event(event: &UserActionEvent, cursor_id: usize) -> Self {
        let action_key = encode_action_key(event.action_type, event.character);
        let extra_info = (cursor_id & 0xFF) as u8;
        Self::new(action_key, false, 1, extra_info)
    }

    /// Increments the count by 1, saturating at max 23-bit value.
    pub fn increment_count(&mut self) -> bool {
        let current = self.count();
        if current < Self::MAX_COUNT {
            let new_data = (self.0 & !(Self::COUNT_MASK << Self::COUNT_SHIFT))
                | (((current + 1) as u64) & Self::COUNT_MASK) << Self::COUNT_SHIFT;
            self.0 = new_data;
            true
        } else {
            false
        }
    }

    /// Sets the count to a specific value.
    pub fn set_count(&mut self, count: u32) {
        let count = count.min(Self::MAX_COUNT);
        self.0 = (self.0 & !(Self::COUNT_MASK << Self::COUNT_SHIFT))
            | ((count as u64) & Self::COUNT_MASK) << Self::COUNT_SHIFT;
    }

    /// Sets the flag bit.
    pub fn set_flag(&mut self, flag: bool) {
        self.0 = (self.0 & !(Self::FLAG_MASK << Self::FLAG_SHIFT))
            | ((if flag { 1 } else { 0 }) & Self::FLAG_MASK) << Self::FLAG_SHIFT;
    }

    /// Sets the extra info field.
    pub fn set_extra_info(&mut self, extra_info: u8) {
        self.0 = (self.0 & !(Self::EXTRA_INFO_MASK << Self::EXTRA_INFO_SHIFT))
            | ((extra_info as u64) & Self::EXTRA_INFO_MASK) << Self::EXTRA_INFO_SHIFT;
    }
}

/// Action cache that compresses repeated user actions.
///
/// This cache maintains a mapping of action signatures to packed action entries,
/// allowing efficient compression of repeated actions (e.g., typing the same
/// character multiple times).
#[derive(Clone, Debug)]
pub struct ActionCache {
    /// Map from action signature to packed action entry
    cache: HashMap<u64, PackedAction>,
    /// Maximum count before forced flush
    max_count: u32,
}

impl ActionCache {
    /// Creates a new action cache with default settings.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_count: PackedAction::MAX_COUNT,
        }
    }

    /// Creates a new action cache with specified max count.
    ///
    /// # Arguments
    ///
    /// * `max_count` - Maximum count before forced flush
    pub fn with_max_count(max_count: u32) -> Self {
        Self {
            cache: HashMap::new(),
            max_count: max_count.min(PackedAction::MAX_COUNT),
        }
    }

    /// Generates a signature for a user action event.
    ///
    /// The signature is used to identify similar actions for caching.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event
    ///
    /// # Returns
    ///
    /// A 64-bit signature representing the action.
    fn signature(&self, event: &UserActionEvent) -> u64 {
        let action_type = event.action_type as u64;
        let cursor_id = event.cursor_id as u64;
        let character = event.character.unwrap_or(0) as u64;
        let metadata = event.metadata as u64;

        // Combine action type, cursor ID, and character to create signature
        // For insert actions, character is important
        // For other actions, cursor ID and action type are more important
        match event.action_type {
            UserActionType::InsertChar => {
                (action_type << 32) | (cursor_id << 16) | (character & 0xFFFF)
            }
            _ => (action_type << 32) | (cursor_id << 16) | metadata,
        }
    }

    /// Extracts metadata from a user action event.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event
    ///
    /// # Returns
    ///
    /// 8-bit metadata value (cursor ID mod 256).
    fn extract_metadata(&self, event: &UserActionEvent) -> u8 {
        (event.cursor_id & 0xFF) as u8
    }

    /// Caches a user action event.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event to cache
    ///
    /// # Returns
    ///
    /// Option containing the packed action if it should be flushed (e.g., count exceeded),
    /// or None if the action was just cached.
    pub fn cache_action(&mut self, event: &UserActionEvent) -> Option<PackedAction> {
        let signature = self.signature(event);
        let flag = false; // Default to normal action
        let metadata = self.extract_metadata(event);

        if let Some(packed) = self.cache.get_mut(&signature) {
            // Increment count for existing action
            let _should_flush = !packed.increment_count();

            // Check if we should flush due to max count
            if packed.count() >= self.max_count {
                let to_flush = *packed;
                self.cache.remove(&signature);
                return Some(to_flush);
            }
        } else {
            // Create new packed action using the action key encoding
            let action_key = encode_action_key(event.action_type, event.character);
            let packed = PackedAction::new(action_key, flag, 1, metadata);
            self.cache.insert(signature, packed);
        }

        None
    }

    /// Flushes all cached actions and returns them.
    ///
    /// # Returns
    ///
    /// A vector of all packed actions in the cache.
    pub fn flush(&mut self) -> Vec<PackedAction> {
        let actions: Vec<PackedAction> = self.cache.values().copied().collect();
        self.cache.clear();
        actions
    }

    /// Returns the number of cached actions.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clears all cached actions.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for ActionCache {
    fn default() -> Self {
        Self::new()
    }
}
