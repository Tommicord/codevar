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

//! A consistent copy of the pointer state and the end-of-frame reset.
//!
//! [`PointerSnapshot`] is what a widget reads when it needs position, motion
//! and buttons to describe the same instant. [`PointerState::finish_frame`]
//! snapshots the state and then clears the accumulated motion and scroll, so
//! the pump can present one frame worth of movement without a second copy of
//! every field.

use core::sync::atomic::Ordering;

use codevar_kuitui_control::{PointerButton, Timestamp};

use crate::geometry::Offset;
use crate::input::KeyboardModifiers;

use super::state::{HAS_POSITION, INSIDE, PointerButtons, PointerState, decode_button, pack, unpack};

impl PointerState {
    /// Copies every field between two equal, even sequence values.
    ///
    /// The copy is consistent even when the pump is applying an event on
    /// another thread; the loop retries while a write is in progress.
    #[must_use]
    pub fn snapshot(&self) -> PointerSnapshot {
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            let flags = self.flags.load(Ordering::Relaxed);
            let snapshot = PointerSnapshot {
                timestamp: Timestamp::from_micros(self.timestamp.load(Ordering::Relaxed)),
                position: (flags & HAS_POSITION != 0).then(|| unpack(self.position.load(Ordering::Relaxed))),
                delta: unpack(self.delta.load(Ordering::Relaxed)),
                scroll: unpack(self.scroll.load(Ordering::Relaxed)),
                buttons: PointerButtons::from_bits_truncate(self.buttons.load(Ordering::Relaxed)),
                modifiers: KeyboardModifiers::from_bits_truncate(self.modifiers.load(Ordering::Relaxed)),
                last_pressed: decode_button(self.last_pressed.load(Ordering::Relaxed)),
                last_released: decode_button(self.last_released.load(Ordering::Relaxed)),
                press_count: self.press_count.load(Ordering::Relaxed),
                release_count: self.release_count.load(Ordering::Relaxed),
                inside: flags & INSIDE != 0,
                sequence: before,
            };
            if self.sequence.load(Ordering::Acquire) == before {
                return snapshot;
            }
            core::hint::spin_loop();
        }
    }

    /// Snapshots the state and clears the per-frame motion and scroll.
    ///
    /// The position and every counter survive; only [`PointerState::delta`]
    /// and [`PointerState::scroll`] restart from zero. This is the write side
    /// and belongs to the single pump.
    pub fn finish_frame(&self) -> PointerSnapshot {
        let snapshot = self.snapshot();
        self.sequence.fetch_add(1, Ordering::AcqRel);
        self.delta
            .store(pack(Offset::ZERO), Ordering::Relaxed);
        self.scroll
            .store(pack(Offset::ZERO), Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
        snapshot
    }
}

/// A consistent copy of the pointer state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerSnapshot {
    /// Time of the last event in the copied state.
    pub timestamp: Timestamp,
    /// The last reported position, or `None` before the first one.
    pub position: Option<Offset>,
    /// Motion accumulated during the frame.
    pub delta: Offset,
    /// Scroll accumulated during the frame.
    pub scroll: Offset,
    /// Buttons held at the copied sequence.
    pub buttons: PointerButtons,
    /// Modifiers reported by the last button or scroll event.
    pub modifiers: KeyboardModifiers,
    /// The last button that went down.
    pub last_pressed: Option<PointerButton>,
    /// The last button that went up.
    pub last_released: Option<PointerButton>,
    /// Number of button-down transitions recorded so far.
    pub press_count: u32,
    /// Number of button-up transitions recorded so far.
    pub release_count: u32,
    /// Whether the pointer was inside the application region.
    pub inside: bool,
    /// The sequence value the copy was taken at.
    pub sequence: u64,
}

impl PointerSnapshot {
    /// Whether `button` is held in this snapshot.
    #[must_use]
    pub fn is_pressed(&self, button: PointerButton) -> bool {
        PointerButtons::from_button(button).is_some_and(|bit| self.buttons.contains(bit))
    }
}
