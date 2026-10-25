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

//! The shared, lock-free record of what the keyboard is doing.
//!
//! The state is a sequence counter followed by plain atomics. The single
//! writer (the input pump) increments the counter, updates the fields and
//! increments it again; a reader copies the fields only between two equal,
//! even counter values, so it never observes a half-applied transition and
//! never takes a lock.

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use codevar_kuitui_control::{DeviceId, KeyCode, KeyEvent, KeyState, Timestamp};

use super::keys::{KEY_SLOTS, decode_key, encode_key, slot};
use super::modifiers::{KeyboardModifiers, modifier_of};

/// Number of 64-bit words the normalized pressed set occupies.
pub(super) const PRESSED_WORDS: usize = KEY_SLOTS.len().div_ceil(64);
/// Number of native key codes that may be held at the same time.
pub(super) const NATIVE_SLOTS: usize = 32;
/// Empty slot of the native key set.
pub(super) const NATIVE_EMPTY: u64 = 0;
/// Removed slot of the native key set.
pub(super) const NATIVE_TOMBSTONE: u64 = u64::MAX;
/// A device identifier that no event has set yet.
pub(super) const NO_DEVICE: u64 = u64::MAX;

/// The keyboard state shared by the pump and every [`KeyboardHandler`].
///
/// The layout is a sequence counter followed by plain atomics. A writer
/// increments the counter, updates the fields and increments it again; a
/// reader copies the fields only between two equal, even counter values. The
/// state is therefore consistent to a reader without a lock, and a backend
/// thread can never block the UI (or the reverse).
pub struct KeyboardState {
    /// Sequence counter; odd while a writer is active.
    pub(super) sequence: AtomicU64,
    /// Last key that went down, zero when none has.
    pub(super) last_pressed: AtomicU64,
    /// Last key that went up, zero when none has.
    pub(super) last_released: AtomicU64,
    /// Last key the platform repeated, zero when none has.
    pub(super) last_repeat: AtomicU64,
    /// Last modifier key that went down, zero when none has.
    pub(super) last_modifier: AtomicU64,
    /// Held modifiers, without the lock flags.
    pub(super) held: AtomicU8,
    /// Caps lock and num lock.
    pub(super) locks: AtomicU8,
    /// Normalized keys currently held, one bit per slot.
    pub(super) pressed: [AtomicU64; PRESSED_WORDS],
    /// Native keys currently held, an open-addressed set.
    pub(super) native: [AtomicU64; NATIVE_SLOTS],
    /// Number of key-down transitions since the state was created.
    pub(super) press_count: AtomicU32,
    /// Number of key-up transitions since the state was created.
    pub(super) release_count: AtomicU32,
    /// Time of the last transition.
    pub(super) timestamp: AtomicU64,
    /// Device that reported the last transition, or [`NO_DEVICE`].
    pub(super) device: AtomicU64,
}

impl KeyboardState {
    /// Creates an empty state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            last_pressed: AtomicU64::new(0),
            last_released: AtomicU64::new(0),
            last_repeat: AtomicU64::new(0),
            last_modifier: AtomicU64::new(0),
            held: AtomicU8::new(0),
            locks: AtomicU8::new(0),
            pressed: [const { AtomicU64::new(0) }; PRESSED_WORDS],
            native: [const { AtomicU64::new(NATIVE_EMPTY) }; NATIVE_SLOTS],
            press_count: AtomicU32::new(0),
            release_count: AtomicU32::new(0),
            timestamp: AtomicU64::new(0),
            device: AtomicU64::new(NO_DEVICE),
        }
    }

    /// Records one key transition.
    ///
    /// This is the write side of the state and is meant for the single
    /// `InputPump` that owns the stream.
    /// Calling it from two threads at once is safe for memory but leaves the
    /// counters and the "last" fields ordered by whichever write finished
    /// last.
    pub fn apply(&self, event: &KeyEvent, timestamp: Timestamp) {
        self.sequence.fetch_add(1, Ordering::AcqRel);
        let encoded = encode_key(event.code);
        match event.state {
            KeyState::Pressed => {
                if encoded != 0 {
                    self.last_pressed
                        .store(encoded, Ordering::Relaxed);
                }
                self.mark_pressed(event.code, true);
                self.press_count.fetch_add(1, Ordering::Relaxed);
                if let Some(flag) = modifier_of(event.code) {
                    self.last_modifier
                        .store(encoded, Ordering::Relaxed);
                    self.held.fetch_or(flag.bits(), Ordering::Relaxed);
                }
                match event.code {
                    KeyCode::CapsLock => {
                        self.locks
                            .fetch_xor(KeyboardModifiers::CAPS_LOCK.bits(), Ordering::Relaxed);
                    }
                    KeyCode::NumLock => {
                        self.locks
                            .fetch_xor(KeyboardModifiers::NUM_LOCK.bits(), Ordering::Relaxed);
                    }
                    _ => {}
                }
                // A backend that reports the modifiers of every event can know
                // about a modifier that went down before the stream opened.
                let reported = KeyboardModifiers::from(event.modifiers).without_locks();
                self.held
                    .fetch_or(reported.bits(), Ordering::Relaxed);
            }
            KeyState::Released => {
                if encoded != 0 {
                    self.last_released
                        .store(encoded, Ordering::Relaxed);
                }
                self.mark_pressed(event.code, false);
                self.release_count.fetch_add(1, Ordering::Relaxed);
                if let Some(flag) = modifier_of(event.code) {
                    self.refresh_modifier(flag);
                }
            }
            KeyState::Repeat => {
                if encoded != 0 {
                    self.last_repeat.store(encoded, Ordering::Relaxed);
                }
            }
        }
        self.timestamp
            .store(timestamp.micros(), Ordering::Relaxed);
        self.device
            .store(event.device.get(), Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
    }

    /// Whether `code` is currently held.
    #[must_use]
    pub fn is_pressed(&self, code: KeyCode) -> bool {
        match code {
            KeyCode::Native(value) => self.native_contains(value),
            other => slot(other).is_some_and(|slot| self.slot_pressed(slot)),
        }
    }

    /// The modifier keys that are held and the lock flags that are on.
    #[must_use]
    pub fn modifiers(&self) -> KeyboardModifiers {
        KeyboardModifiers::from_bits_truncate(
            self.held.load(Ordering::Relaxed) | self.locks.load(Ordering::Relaxed),
        )
    }

    /// The last key that went down.
    #[must_use]
    pub fn last_pressed(&self) -> Option<KeyCode> {
        decode_key(self.last_pressed.load(Ordering::Relaxed))
    }

    /// The last key that went up.
    #[must_use]
    pub fn last_released(&self) -> Option<KeyCode> {
        decode_key(self.last_released.load(Ordering::Relaxed))
    }

    /// The last key the platform repeated.
    #[must_use]
    pub fn last_repeat(&self) -> Option<KeyCode> {
        decode_key(self.last_repeat.load(Ordering::Relaxed))
    }

    /// The last modifier key that went down.
    #[must_use]
    pub fn last_modifier(&self) -> Option<KeyCode> {
        decode_key(self.last_modifier.load(Ordering::Relaxed))
    }

    /// Number of key-down transitions recorded so far.
    #[must_use]
    pub fn press_count(&self) -> u32 {
        self.press_count.load(Ordering::Relaxed)
    }

    /// Number of key-up transitions recorded so far.
    #[must_use]
    pub fn release_count(&self) -> u32 {
        self.release_count.load(Ordering::Relaxed)
    }

    /// Number of keys currently held.
    #[must_use]
    pub fn held_count(&self) -> u32 {
        let normalized: u32 = self
            .pressed
            .iter()
            .map(|word| word.load(Ordering::Relaxed).count_ones())
            .sum();
        normalized + self.native_count()
    }

    /// The current sequence value; even between transitions.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }

    /// The time of the last transition.
    #[must_use]
    pub fn timestamp(&self) -> Timestamp {
        Timestamp::from_micros(self.timestamp.load(Ordering::Relaxed))
    }

    /// The keyboard that reported the last transition.
    #[must_use]
    pub fn device(&self) -> Option<DeviceId> {
        match self.device.load(Ordering::Relaxed) {
            NO_DEVICE => None,
            value => Some(DeviceId::new(value)),
        }
    }

    /// Adds or removes one key from the pressed set.
    fn mark_pressed(&self, code: KeyCode, pressed: bool) {
        if let Some(slot) = slot(code) {
            let word = usize::from(slot) / 64;
            let mask = 1u64 << (usize::from(slot) % 64);
            if pressed {
                self.pressed[word].fetch_or(mask, Ordering::Relaxed);
            } else {
                self.pressed[word].fetch_and(!mask, Ordering::Relaxed);
            }
        } else if let KeyCode::Native(value) = code {
            if pressed {
                self.native_insert(value);
            } else {
                self.native_remove(value);
            }
        }
    }

    /// Recomputes one modifier family from the pressed set.
    ///
    /// Both sides of the keyboard map to the same flag, so releasing the left
    /// shift while the right one is still down must keep the flag set. The
    /// pressed set is the source of truth, not a counter.
    fn refresh_modifier(&self, flag: KeyboardModifiers) {
        self.held
            .fetch_and(!flag.bits(), Ordering::Relaxed);
        let family = if flag == KeyboardModifiers::SHIFT {
            [KeyCode::ShiftLeft, KeyCode::ShiftRight]
        } else if flag == KeyboardModifiers::CONTROL {
            [KeyCode::ControlLeft, KeyCode::ControlRight]
        } else if flag == KeyboardModifiers::ALT {
            [KeyCode::AltLeft, KeyCode::AltRight]
        } else if flag == KeyboardModifiers::META {
            [KeyCode::MetaLeft, KeyCode::MetaRight]
        } else {
            return;
        };
        if family
            .iter()
            .any(|code| slot(*code).is_some_and(|slot| self.slot_pressed(slot)))
        {
            self.held.fetch_or(flag.bits(), Ordering::Relaxed);
        }
    }

    /// Whether the bit of one normalized slot is set.
    fn slot_pressed(&self, slot: u8) -> bool {
        let word = usize::from(slot) / 64;
        let mask = 1u64 << (usize::from(slot) % 64);
        self.pressed[word].load(Ordering::Relaxed) & mask != 0
    }

    /// The slot a native code hashes to.
    #[inline]
    pub(super) fn native_start(value: u32) -> usize {
        let mixed = u64::from(value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        usize::try_from(mixed >> 59).unwrap_or(0)
    }

    /// Adds one native code to the open-addressed set.
    ///
    /// The set holds [`NATIVE_SLOTS`] codes. A press beyond that is still
    /// recorded in `last_pressed` but not in the set; native codes are
    /// vendor-specific extras, not the keys a shortcut is built from.
    fn native_insert(&self, value: u32) {
        let stored = u64::from(value) + 1;
        let start = Self::native_start(value);
        for step in 0..NATIVE_SLOTS {
            let index = (start + step) % NATIVE_SLOTS;
            let current = self.native[index].load(Ordering::Relaxed);
            if current == stored {
                return;
            }
            if current == NATIVE_EMPTY || current == NATIVE_TOMBSTONE {
                self.native[index].store(stored, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Removes one native code from the open-addressed set.
    fn native_remove(&self, value: u32) {
        let stored = u64::from(value) + 1;
        let start = Self::native_start(value);
        for step in 0..NATIVE_SLOTS {
            let index = (start + step) % NATIVE_SLOTS;
            let current = self.native[index].load(Ordering::Relaxed);
            if current == stored {
                self.native[index].store(NATIVE_TOMBSTONE, Ordering::Relaxed);
                return;
            }
            if current == NATIVE_EMPTY {
                return;
            }
        }
    }

    /// Whether one native code is in the set.
    fn native_contains(&self, value: u32) -> bool {
        let stored = u64::from(value) + 1;
        let start = Self::native_start(value);
        for step in 0..NATIVE_SLOTS {
            let index = (start + step) % NATIVE_SLOTS;
            let current = self.native[index].load(Ordering::Relaxed);
            if current == stored {
                return true;
            }
            if current == NATIVE_EMPTY {
                return false;
            }
        }
        false
    }

    /// Number of native codes in the set.
    fn native_count(&self) -> u32 {
        self.native
            .iter()
            .filter(|slot| {
                let value = slot.load(Ordering::Relaxed);
                value != NATIVE_EMPTY && value != NATIVE_TOMBSTONE
            })
            .count()
            .try_into()
            .unwrap_or(u32::MAX)
    }
}

impl Default for KeyboardState {
    fn default() -> Self {
        Self::new()
    }
}
