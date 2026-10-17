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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! [`TextEditablePut`]: staged insertion at a cursor.

use codevar_logger::log_error;

use super::super::shared_gap::{SLOT_CURSOR_BITS, SLOT_MAX_BASE, SLOT_MAX_SIZE};
use super::super::traits::TextEditablePut;
use super::super::writable::{MAX_CURSORS, MAX_GAPS, StreamWritable};
use crate::error::{WreditError, WreditResult};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditablePut<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn put(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        let position = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        self.stage_private(id, value, position)?;
        Ok(self.move_caret_to(id, position.saturating_add(1)))
    }

    fn put_shared(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // The mask only addresses live cursors; ids at or above the
        // mask width can never join a shared gap.
        if id >= MAX_CURSORS || id >= SLOT_CURSOR_BITS {
            log_error!("wredit: cursor {id} cannot join a shared gap mask");
            return Err(WreditError::InvalidCursor { cursor: id });
        }
        let mut recovered = false;
        loop {
            let position = self
                .base
                .cursor(id)
                .map(super::super::cursor::Cursor::array_pos)
                .unwrap_or(0);
            let slot = self.ensure_shared_slot(id)?;
            let Some(descriptor) = self.base.shared_ref().slot(slot) else {
                log_error!("wredit: shared gap slot {slot} vanished");
                return Err(WreditError::InvalidOperation {
                    what: "shared gap slot missing",
                });
            };
            let size = descriptor.size();
            if size >= SLOT_MAX_SIZE {
                if recovered {
                    log_error!("wredit: shared gap region exhausted");
                    return Err(WreditError::CapacityExhausted {
                        requested: size + 1,
                        available: 0,
                    });
                }
                // Commit the full region and restage into a fresh one.
                let _ = self.flush_gap(MAX_GAPS + slot);
                recovered = true;
                continue;
            }
            // A fresh region commits where the caret stands; an
            // existing one appends after its members' text.
            let base = if size == 0 {
                position.min(SLOT_MAX_BASE)
            } else {
                descriptor.base()
            };
            let Some(offset) = base
                .checked_add(size)
                .and_then(|value| value.checked_add(1))
            else {
                log_error!("wredit: shared gap region offset overflow");
                return Err(WreditError::LimitExceeded {
                    what: "shared gap region offset",
                });
            };
            if self
                .base
                .buffer_mut()
                .ensure_capacity(offset)
                .is_err()
            {
                return Err(WreditError::CapacityExhausted {
                    requested: offset,
                    available: 0,
                });
            }
            let arena = self.base.buffer_mut();
            arena.set_count(offset);
            if let Some(slot_ref) = arena.slice_mut().get_mut(base + size) {
                *slot_ref = Buf::from(value);
            } else {
                log_error!("wredit: shared gap write escaped the staging arena");
                return Err(WreditError::InvalidOperation {
                    what: "shared gap write out of bounds",
                });
            }
            self.base
                .shared_mut()
                .set_region(slot, base, size + 1);
            return Ok(self.move_caret_to(id, position.saturating_add(1)));
        }
    }
}
