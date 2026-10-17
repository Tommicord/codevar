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

//! [`TextEditableGap`]: flushing and relocating staged gap runs.

use codevar_logger::log_error;

use super::super::traits::TextEditableGap;
use super::super::writable::{MAX_GAPS, SHARED_GAP_SLOTS, StreamWritable};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableGap<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn flush_gap(&mut self, gap_index: usize) -> usize {
        // The inherent flusher implements the commit path; the trait
        // method is the contract-facing alias of it.
        StreamWritable::flush_gap(self, gap_index)
    }

    fn flush_all(&mut self) -> usize {
        StreamWritable::flush_all(self)
    }

    unsafe fn shift_gap_block(&mut self, gap_index: usize, distance: isize) -> usize {
        let logical_length = self.base.logical_length();
        let current = if gap_index < MAX_GAPS {
            self.base
                .gaps_ref()
                .get(gap_index)
                .map(|gap| gap.base())
        } else if gap_index - MAX_GAPS < SHARED_GAP_SLOTS {
            let slot = gap_index - MAX_GAPS;
            self.base
                .shared_ref()
                .slot(slot)
                .filter(|descriptor| descriptor.is_allocated())
                .map(|descriptor| descriptor.base())
        } else {
            None
        };
        let Some(current) = current else {
            log_error!("wredit: shift of the unallocated gap {gap_index}");
            return 0;
        };
        let moved = if distance >= 0 {
            current
                .saturating_add(usize::try_from(distance).unwrap_or(usize::MAX))
                .min(logical_length)
        } else {
            current.saturating_sub(distance.unsigned_abs())
        };
        if gap_index < MAX_GAPS {
            if let Some(gap) = self.base.gaps_mut().get_mut(gap_index) {
                gap.set_base(moved);
            }
        } else {
            let slot = gap_index - MAX_GAPS;
            let size = self
                .base
                .shared_ref()
                .slot(slot)
                .map(|descriptor| descriptor.size())
                .unwrap_or(0);
            self.base
                .shared_mut()
                .set_region(slot, moved, size);
        }
        moved
    }

    unsafe fn shift_right_gap_block(&mut self, gap_index: usize, distance: usize) -> usize {
        // SAFETY: forwarded to `shift_gap_block`, which clamps the
        // base into the logical document.
        unsafe { self.shift_gap_block(gap_index, isize::try_from(distance).unwrap_or(isize::MAX)) }
    }

    unsafe fn shift_left_gap_block(&mut self, gap_index: usize, distance: usize) -> usize {
        let magnitude = isize::try_from(distance).unwrap_or(isize::MAX);
        // SAFETY: forwarded to `shift_gap_block`, which clamps the
        // base into the logical document.
        unsafe { self.shift_gap_block(gap_index, -magnitude) }
    }
}
