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

//! The consistent read side of the keyboard state.
//!
//! A widget asks for a [`KeyboardSnapshot`] when it needs several fields to
//! agree. The snapshot copies every field between two equal sequence values,
//! so the combination it returns is one transition, never a mix of two.

use core::sync::atomic::Ordering;

use codevar_kuitui_control::{DeviceId, KeyCode, Timestamp};

use super::keys::{decode_key, slot};
use super::modifiers::KeyboardModifiers;
use super::state::{KeyboardState, NATIVE_EMPTY, NATIVE_SLOTS, NATIVE_TOMBSTONE, NO_DEVICE, PRESSED_WORDS};

impl KeyboardState {
    /// Reads every field as one consistent transition.
    ///
    /// The loop re-reads when a writer was active before or during the read.
    /// Writers hold the sequence for a handful of stores, so the retry is
    /// bounded by the writer's critical section, not by how often writers
    /// arrive.
    #[must_use]
    pub fn snapshot(&self) -> KeyboardSnapshot {
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            let snapshot = KeyboardSnapshot {
                last_pressed: decode_key(self.last_pressed.load(Ordering::Relaxed)),
                last_released: decode_key(self.last_released.load(Ordering::Relaxed)),
                last_repeat: decode_key(self.last_repeat.load(Ordering::Relaxed)),
                last_modifier: decode_key(self.last_modifier.load(Ordering::Relaxed)),
                modifiers: KeyboardModifiers::from_bits_truncate(
                    self.held.load(Ordering::Relaxed) | self.locks.load(Ordering::Relaxed),
                ),
                pressed: [
                    self.pressed[0].load(Ordering::Relaxed),
                    self.pressed[1].load(Ordering::Relaxed),
                ],
                native: self.read_native(),
                press_count: self.press_count.load(Ordering::Relaxed),
                release_count: self.release_count.load(Ordering::Relaxed),
                timestamp: Timestamp::from_micros(self.timestamp.load(Ordering::Relaxed)),
                device: match self.device.load(Ordering::Relaxed) {
                    NO_DEVICE => None,
                    value => Some(DeviceId::new(value)),
                },
            };
            if self.sequence.load(Ordering::Acquire) == before {
                return snapshot;
            }
            core::hint::spin_loop();
        }
    }

    /// Copies the native key set.
    fn read_native(&self) -> [u64; NATIVE_SLOTS] {
        core::array::from_fn(|index| self.native[index].load(Ordering::Relaxed))
    }
}

/// The value of every field [`KeyboardState`] had at one instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyboardSnapshot {
    /// The last key that went down.
    pub last_pressed: Option<KeyCode>,
    /// The last key that went up.
    pub last_released: Option<KeyCode>,
    /// The last key the platform repeated.
    pub last_repeat: Option<KeyCode>,
    /// The last modifier key that went down.
    pub last_modifier: Option<KeyCode>,
    /// The held modifiers and the lock flags that are on.
    pub modifiers: KeyboardModifiers,
    /// The normalized keys that were held, one bit per slot.
    pub pressed: [u64; PRESSED_WORDS],
    /// The native keys that were held, in the internal set layout.
    pub native: [u64; NATIVE_SLOTS],
    /// Number of key-down transitions recorded so far.
    pub press_count: u32,
    /// Number of key-up transitions recorded so far.
    pub release_count: u32,
    /// Time of the last transition.
    pub timestamp: Timestamp,
    /// The keyboard that reported the last transition.
    pub device: Option<DeviceId>,
}

impl KeyboardSnapshot {
    /// Whether `code` was held at the instant of the snapshot.
    #[must_use]
    pub fn is_pressed(&self, code: KeyCode) -> bool {
        match code {
            KeyCode::Native(value) => self.native_contains(value),
            other => slot(other).is_some_and(|slot| {
                let word = usize::from(slot) / 64;
                let mask = 1u64 << (usize::from(slot) % 64);
                self.pressed[word] & mask != 0
            }),
        }
    }

    /// Whether `code` was held with every flag of `required` active.
    #[must_use]
    pub fn is_shortcut(&self, code: KeyCode, required: KeyboardModifiers) -> bool {
        self.modifiers.contains(required) && self.is_pressed(code)
    }

    /// Number of keys that were held.
    #[must_use]
    pub fn held_count(&self) -> u32 {
        let normalized: u32 = self
            .pressed
            .iter()
            .map(|word| word.count_ones())
            .sum();
        let native: u32 = self
            .native
            .iter()
            .filter(|value| **value != NATIVE_EMPTY && **value != NATIVE_TOMBSTONE)
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        normalized + native
    }

    /// Whether one native code is in the snapshot's set.
    fn native_contains(&self, value: u32) -> bool {
        let stored = u64::from(value) + 1;
        let start = KeyboardState::native_start(value);
        for step in 0..NATIVE_SLOTS {
            let index = (start + step) % NATIVE_SLOTS;
            let current = self.native[index];
            if current == stored {
                return true;
            }
            if current == NATIVE_EMPTY {
                return false;
            }
        }
        false
    }
}
