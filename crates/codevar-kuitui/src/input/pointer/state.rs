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

//! The shared, lock-free record of what the pointing device is doing.
//!
//! The state is a sequence counter followed by plain atomics, the same
//! protocol as the keyboard state: the single writer increments the counter,
//! updates the fields and increments it again, and a reader copies between
//! two equal, even counter values. Two `f32` pairs (position and accumulated
//! motion) are packed into one `u64` each, so a reader never sees half of a
//! coordinate.

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use codevar_kuitui_control::{KeyState, Point, PointerButton, PointerEvent, Timestamp};

use crate::geometry::Offset;
use crate::input::KeyboardModifiers;

/// Bit of [`PointerState::flags`] that marks a reported position.
pub(super) const HAS_POSITION: u8 = 1 << 0;
/// Bit of [`PointerState::flags`] that marks the pointer as inside the
/// application region.
pub(super) const INSIDE: u8 = 1 << 1;

/// Encoded value of "no button" in the last-button fields.
pub(super) const NO_BUTTON: u32 = 0;
/// Bit offset of the first [`PointerButton::Other`] button.
pub(super) const OTHER_BUTTON_BASE: u32 = 5;
/// Number of [`PointerButton::Other`] buttons a set can name.
pub(super) const OTHER_BUTTONS: u16 = 32 - OTHER_BUTTON_BASE as u16;

/// The set of pointer buttons currently held.
///
/// The five standard buttons have a fixed bit, and [`PointerButton::Other`]
/// gets the bits above them, so a backend that reports an unusual button
/// still fits in the same atomic word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PointerButtons(u32);

impl PointerButtons {
    /// No button is held.
    pub const NONE: PointerButtons = PointerButtons(0);
    /// The primary button is held.
    pub const PRIMARY: PointerButtons = PointerButtons(1 << 0);
    /// The secondary button is held.
    pub const SECONDARY: PointerButtons = PointerButtons(1 << 1);
    /// The middle button is held.
    pub const MIDDLE: PointerButtons = PointerButtons(1 << 2);
    /// The back button is held.
    pub const BACK: PointerButtons = PointerButtons(1 << 3);
    /// The forward button is held.
    pub const FORWARD: PointerButtons = PointerButtons(1 << 4);

    /// The bit of one button, or `None` for an `Other` index this set cannot
    /// name.
    #[must_use]
    pub fn from_button(button: PointerButton) -> Option<PointerButtons> {
        let word = match button {
            PointerButton::Primary => Self::PRIMARY.0,
            PointerButton::Secondary => Self::SECONDARY.0,
            PointerButton::Middle => Self::MIDDLE.0,
            PointerButton::Back => Self::BACK.0,
            PointerButton::Forward => Self::FORWARD.0,
            PointerButton::Other(value) if value < OTHER_BUTTONS => {
                1 << (OTHER_BUTTON_BASE + u32::from(value))
            }
            _ => return None,
        };
        Some(Self(word))
    }

    /// The raw bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Rebuilds the set from raw bits, dropping bits this version does not
    /// define.
    #[must_use]
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits)
    }

    /// Whether every button of `other` is held.
    #[must_use]
    pub const fn contains(self, other: PointerButtons) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no button is held.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Number of held buttons.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl core::ops::BitOr for PointerButtons {
    type Output = PointerButtons;

    #[inline]
    fn bitor(self, rhs: PointerButtons) -> PointerButtons {
        PointerButtons(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for PointerButtons {
    #[inline]
    fn bitor_assign(&mut self, rhs: PointerButtons) {
        self.0 |= rhs.0;
    }
}

/// The pointer state shared by the pump and every [`MouseHandler`].
///
/// [`MouseHandler`]: super::MouseHandler
///
/// The write side is [`PointerState::apply`] or [`PointerState::update`];
/// both are meant for the single `InputPump` that owns the stream. Readers use
/// [`PointerState::snapshot`] for a consistent copy or the individual
/// accessors when one field is enough.
pub struct PointerState {
    /// Sequence counter; odd while a writer is active.
    pub(super) sequence: AtomicU64,
    /// Packed position; valid when [`HAS_POSITION`] is set.
    pub(super) position: AtomicU64,
    /// Packed motion accumulated since the last [`PointerState::finish_frame`].
    pub(super) delta: AtomicU64,
    /// Packed scroll accumulated since the last frame.
    pub(super) scroll: AtomicU64,
    /// Held buttons.
    pub(super) buttons: AtomicU32,
    /// Modifiers reported by the last button or scroll event.
    pub(super) modifiers: AtomicU8,
    /// [`HAS_POSITION`] and [`INSIDE`].
    pub(super) flags: AtomicU8,
    /// Last button that went down, encoded, or [`NO_BUTTON`].
    pub(super) last_pressed: AtomicU32,
    /// Last button that went up, encoded, or [`NO_BUTTON`].
    pub(super) last_released: AtomicU32,
    /// Number of button-down transitions since the state was created.
    pub(super) press_count: AtomicU32,
    /// Number of button-up transitions since the state was created.
    pub(super) release_count: AtomicU32,
    /// Time of the last event.
    pub(super) timestamp: AtomicU64,
}

impl PointerState {
    /// Creates an empty state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            position: AtomicU64::new(pack(Offset::ZERO)),
            delta: AtomicU64::new(pack(Offset::ZERO)),
            scroll: AtomicU64::new(pack(Offset::ZERO)),
            buttons: AtomicU32::new(0),
            modifiers: AtomicU8::new(0),
            flags: AtomicU8::new(0),
            last_pressed: AtomicU32::new(NO_BUTTON),
            last_released: AtomicU32::new(NO_BUTTON),
            press_count: AtomicU32::new(0),
            release_count: AtomicU32::new(0),
            timestamp: AtomicU64::new(0),
        }
    }

    /// Records one pointer event.
    ///
    /// This is the write side of the state and is meant for the single
    /// `InputPump` that owns the stream.
    ///
    /// A motion event that reports a position but no delta gets the
    /// difference from the previous position, so a backend that only tracks
    /// absolute coordinates still produces a usable delta. A scroll event
    /// accumulates into the frame scroll instead of overwriting it, because a
    /// single frame may carry several notches.
    pub fn apply(&self, event: &PointerEvent, timestamp: Timestamp) {
        self.sequence.fetch_add(1, Ordering::AcqRel);
        match event {
            PointerEvent::Motion { position, delta } => {
                let mut motion = delta.map_or(Offset::ZERO, to_offset);
                if let Some(position) = position {
                    if delta.is_none() && self.flags.load(Ordering::Relaxed) & HAS_POSITION != 0 {
                        motion = to_offset(*position).sub(self.packed_position());
                    }
                    self.store_position(*position);
                }
                self.accumulate(&self.delta, motion);
            }
            PointerEvent::Button {
                button,
                state,
                position,
                modifiers,
            } => {
                if let Some(position) = position {
                    self.store_position(*position);
                }
                self.modifiers
                    .store(KeyboardModifiers::from(*modifiers).bits(), Ordering::Relaxed);
                match state {
                    KeyState::Pressed => {
                        if let Some(bit) = PointerButtons::from_button(*button) {
                            self.buttons
                                .fetch_or(bit.bits(), Ordering::Relaxed);
                        }
                        self.last_pressed
                            .store(encode_button(*button), Ordering::Relaxed);
                        self.press_count.fetch_add(1, Ordering::Relaxed);
                    }
                    KeyState::Released => {
                        if let Some(bit) = PointerButtons::from_button(*button) {
                            self.buttons
                                .fetch_and(!bit.bits(), Ordering::Relaxed);
                        }
                        self.last_released
                            .store(encode_button(*button), Ordering::Relaxed);
                        self.release_count.fetch_add(1, Ordering::Relaxed);
                    }
                    KeyState::Repeat => {}
                }
            }
            PointerEvent::Scroll {
                delta,
                position,
                modifiers,
            } => {
                if let Some(position) = position {
                    self.store_position(*position);
                }
                self.modifiers
                    .store(KeyboardModifiers::from(*modifiers).bits(), Ordering::Relaxed);
                self.accumulate(&self.scroll, to_offset(*delta));
            }
            PointerEvent::Enter { position } => {
                if let Some(position) = position {
                    self.store_position(*position);
                }
                self.flags.fetch_or(INSIDE, Ordering::Relaxed);
            }
            PointerEvent::Leave => {
                self.flags.fetch_and(!INSIDE, Ordering::Relaxed);
            }
            PointerEvent::Cancel => {
                self.buttons.store(0, Ordering::Relaxed);
                self.flags.fetch_and(!INSIDE, Ordering::Relaxed);
            }
            _ => {}
        }
        self.timestamp
            .store(timestamp.micros(), Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
    }

    /// The last reported position, or `None` before the first one.
    #[must_use]
    pub fn position(&self) -> Option<Offset> {
        (self.flags.load(Ordering::Relaxed) & HAS_POSITION != 0).then(|| self.packed_position())
    }

    /// Whether the pointer is inside the application's input region.
    #[must_use]
    pub fn is_inside(&self) -> bool {
        self.flags.load(Ordering::Relaxed) & INSIDE != 0
    }

    /// The buttons that are held.
    #[must_use]
    pub fn buttons(&self) -> PointerButtons {
        PointerButtons::from_bits_truncate(self.buttons.load(Ordering::Relaxed))
    }

    /// Whether `button` is held.
    #[must_use]
    pub fn is_pressed(&self, button: PointerButton) -> bool {
        PointerButtons::from_button(button).is_some_and(|bit| self.buttons().contains(bit))
    }

    /// The modifiers reported by the last button or scroll event.
    #[must_use]
    pub fn modifiers(&self) -> KeyboardModifiers {
        KeyboardModifiers::from_bits_truncate(self.modifiers.load(Ordering::Relaxed))
    }

    /// The motion accumulated since the last [`Self::finish_frame`].
    #[must_use]
    pub fn delta(&self) -> Offset {
        unpack(self.delta.load(Ordering::Relaxed))
    }

    /// The scroll accumulated since the last [`Self::finish_frame`].
    #[must_use]
    pub fn scroll(&self) -> Offset {
        unpack(self.scroll.load(Ordering::Relaxed))
    }

    /// The last button that went down.
    #[must_use]
    pub fn last_pressed(&self) -> Option<PointerButton> {
        decode_button(self.last_pressed.load(Ordering::Relaxed))
    }

    /// The last button that went up.
    #[must_use]
    pub fn last_released(&self) -> Option<PointerButton> {
        decode_button(self.last_released.load(Ordering::Relaxed))
    }

    /// Number of button-down transitions recorded so far.
    #[must_use]
    pub fn press_count(&self) -> u32 {
        self.press_count.load(Ordering::Relaxed)
    }

    /// Number of button-up transitions recorded so far.
    #[must_use]
    pub fn release_count(&self) -> u32 {
        self.release_count.load(Ordering::Relaxed)
    }

    /// The current sequence value; even between events.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }

    /// The time of the last event.
    #[must_use]
    pub fn timestamp(&self) -> Timestamp {
        Timestamp::from_micros(self.timestamp.load(Ordering::Relaxed))
    }

    /// Stores a position and marks it as reported.
    fn store_position(&self, position: Point) {
        self.position
            .store(pack(to_offset(position)), Ordering::Relaxed);
        self.flags
            .fetch_or(HAS_POSITION, Ordering::Relaxed);
    }

    /// Reads the packed position, which is the zero offset until one is
    /// reported.
    pub(super) fn packed_position(&self) -> Offset {
        unpack(self.position.load(Ordering::Relaxed))
    }

    /// Adds a delta to an accumulator.
    fn accumulate(&self, accumulator: &AtomicU64, delta: Offset) {
        if delta != Offset::ZERO {
            let current = unpack(accumulator.load(Ordering::Relaxed));
            accumulator.store(pack(current.add(delta)), Ordering::Relaxed);
        }
    }
}

impl Default for PointerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a control-layer point to a Kuitui offset.
#[inline]
pub(super) fn to_offset(point: Point) -> Offset {
    Offset::new(point.x, point.y)
}

/// Packs two coordinates into one atomic word.
#[inline]
pub(super) const fn pack(offset: Offset) -> u64 {
    ((offset.x.to_bits() as u64) << 32) | offset.y.to_bits() as u64
}

/// Unpacks an offset written by [`pack`].
#[inline]
pub(super) fn unpack(value: u64) -> Offset {
    Offset::new(f32::from_bits((value >> 32) as u32), f32::from_bits(value as u32))
}

/// Encodes a button for the last-button fields.
pub(super) fn encode_button(button: PointerButton) -> u32 {
    match button {
        PointerButton::Primary => 1,
        PointerButton::Secondary => 2,
        PointerButton::Middle => 3,
        PointerButton::Back => 4,
        PointerButton::Forward => 5,
        PointerButton::Other(value) => 6 + u32::from(value),
        _ => NO_BUTTON,
    }
}

/// Decodes a last-button value.
pub(super) fn decode_button(value: u32) -> Option<PointerButton> {
    match value {
        1 => Some(PointerButton::Primary),
        2 => Some(PointerButton::Secondary),
        3 => Some(PointerButton::Middle),
        4 => Some(PointerButton::Back),
        5 => Some(PointerButton::Forward),
        NO_BUTTON => None,
        other => Some(PointerButton::Other(u16::try_from(other - 6).unwrap_or(u16::MAX))),
    }
}
