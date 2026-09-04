//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use super::{SharedGap, StreamWritable, TextEditableGap, TextEditablePut, Writable};

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE> {
    /// Handles newline character by moving cursor to next line
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It requires a valid cursor_id that corresponds to an existing cursor
    /// - It directly modifies vertical space counter without bounds checking
    /// - It performs mutable cursor access that could invalidate other references
    ///
    /// # Preconditions
    ///
    /// - cursor_id must be a valid cursor ID in the writable
    /// - The cursor must be properly initialized
    /// - Vertical space counter must not overflow (handled by saturating_add)
    unsafe fn handle_newline(&mut self, cursor_id: usize) {
        *self.base_mut().v_space_mut() = self.base().v_space().saturating_add(1);
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            cursor.move_next_line();
        }
    }

    /// Writes packed value to available space in shared gap region
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It performs raw pointer operations on the gap buffer
    /// - It assumes the gap buffer is properly allocated and aligned
    /// - It does not bounds check the write operation (relies on caller)
    /// - It uses pointer arithmetic that could overflow if indices are invalid
    ///
    /// # Preconditions
    ///
    /// - shared_gap must contain a valid, allocated gap buffer
    /// - region_pos must be within the valid range of the gap buffer
    /// - gap_siz must be a power of two for the masking operation to work correctly
    /// - The cursor must have sufficient remaining space in its region
    unsafe fn write_packed_value(
        &mut self,
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        region_pos: usize,
        ch: u32,
        points_new: bool,
        gap_siz: usize,
    ) {
        let gap_ptr = shared_gap.gap_mut();
        let buff_ptr = gap_ptr.gap_ptr_mut().buff_mut();

        // Check if buffer is allocated before writing
        if buff_ptr.is_null() {
            return;
        }

        let index = region_pos & (gap_siz - 1);

        let packed = Self::make_packed(
            if points_new {
                Self::T_MERGE
            } else {
                Self::T_COMMON
            },
            ch,
        );
        let packed_bytes = packed.to_ne_bytes();
        std::ptr::copy_nonoverlapping(
            packed_bytes.as_ptr(),
            buff_ptr.add(index) as *mut u8,
            std::mem::size_of::<u32>(),
        );
    }

    /// Updates region metadata after writing

    unsafe fn update_region_metadata(
        &mut self,
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        region_pos: usize,
        cursor_id: usize,
    ) {
        let current_gap_ptr = shared_gap.gap_mut().gap_ptr_mut().gap_ptr();
        if let Some(region) = shared_gap.cursor_region_mut(cursor_id) {
            region.decrement_remaining(1);
            region.set_cursor_position(region_pos + 1);
        }
        shared_gap
            .gap_mut()
            .gap_ptr_mut()
            .set_gap_ptr(current_gap_ptr + 1);
    }

    /// Writes to available space directly

    unsafe fn write_to_available_space_direct(
        &mut self,
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        region_pos: usize,
        ch: u32,
        points_new: bool,
        gap_siz: usize,
        cursor_id: usize,
    ) {
        self.write_packed_value(shared_gap, region_pos, ch, points_new, gap_siz);
        self.update_region_metadata(shared_gap, region_pos, cursor_id);
    }

    /// Writes to a full region by moving it right

    unsafe fn write_to_full_region_direct(
        &mut self,
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        region_pos: usize,
        ch: u32,
        points_new: bool,
        gap_siz: usize,
        cursor_id: usize,
    ) {
        if !shared_gap.move_cursor_region_right(cursor_id) {
            self.shift_right_gap_block(0, gap_siz);

            let regions = shared_gap.cursor_regions_mut();
            shared_gap.rebalance(regions);
            self.retry_write_after_move_direct(
                shared_gap, region_pos, ch, points_new, gap_siz, cursor_id,
            );
        } else {
            self.write_packed_value(shared_gap, region_pos, ch, points_new, gap_siz);
            self.update_region_metadata(shared_gap, region_pos, cursor_id);
        }
    }

    /// Retries writing after moving region

    unsafe fn retry_write_after_move_direct(
        &mut self,
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        region_pos: usize,
        ch: u32,
        points_new: bool,
        gap_siz: usize,
        cursor_id: usize,
    ) {
        if let Some(region) = shared_gap.cursor_region_mut(cursor_id) {
            let has_space = region.has_space();
            let new_region_pos = region.cursor_position();
            let _ = region;

            if has_space {
                self.write_packed_value(
                    shared_gap,
                    new_region_pos,
                    ch,
                    points_new,
                    gap_siz,
                );
                self.update_region_metadata(shared_gap, new_region_pos, cursor_id);
            }
        }
    }

    /// Checks if gap is full and flushes if necessary

    unsafe fn check_and_flush_gap(
        &mut self,
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        gap_siz: usize,
        cursor_id: usize,
    ) {
        let gap_ptr = shared_gap.gap().gap_ptr().gap_ptr();
        if gap_ptr >= gap_siz {
            self.flush_gap(cursor_id);
            shared_gap.gap_mut().gap_ptr_mut().set_gap_ptr(0);
        }
    }

    /// Advances the cursor forward
    unsafe fn advance_cursor(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            cursor.move_forward(1);
        }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditablePut<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn put(&mut self, ch: u32, points_new: bool, cursor_id: usize) {
        let ptr = self.base().cursor_from_id(cursor_id);
        if ptr.is_none() {
            return;
        }

        // Check if cursor is in a shared gap and use suballocation
        if self.base().is_cursor_in_shared_gap(cursor_id) {
            self.put_shared(ch, points_new, cursor_id);
            return;
        }

        let mut i = 0;
        let gap_count = self.base().gaps_mut().gap_count();
        let mut should_flush = false;
        let mut flush_gap_siz = 0;
        let mut flush_gap_start = std::ptr::null();
        let mut v_space_increment = 0;

        while i < gap_count {
            let gaps = self.base().gaps_mut();
            let gap = gaps.get_mut(i);
            if let Some(gap_value) = gap {
                let old_ptr = gap_value.gap_ptr().gap_ptr();
                let gap_start = gap_value.gap_start();

                if ch == 0x0A {
                    v_space_increment = 1;
                    if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
                        cursor.move_next_line();
                    }
                } else {
                    let gap_siz = gap_value.gap_size();
                    let gap_ptr = gap_value.gap_ptr_mut();

                    // Check if buffer is allocated before writing
                    let buff_ptr = gap_ptr.buff_mut();
                    if buff_ptr.is_null() {
                        i += 1;
                        continue;
                    }

                    gap_ptr.set_gap_ptr(old_ptr + 1);
                    let index = gap_ptr.gap_ptr() & (gap_siz - 1);
                    let packed = Self::make_packed(
                        if points_new {
                            Self::T_MERGE
                        } else {
                            Self::T_COMMON
                        },
                        ch,
                    );
                    // Write packed value as bytes to avoid type issues
                    let packed_bytes = packed.to_ne_bytes();
                    std::ptr::copy_nonoverlapping(
                        packed_bytes.as_ptr(),
                        buff_ptr.add(index) as *mut u8,
                        std::mem::size_of::<u32>(),
                    );

                    if gap_ptr.gap_ptr() >= gap_siz {
                        should_flush = true;
                        flush_gap_siz = gap_siz;
                        flush_gap_start = gap_start;
                    }

                    if gap_ptr.gap_ptr() <= (old_ptr / gap_siz) * gap_siz {
                        self.shift_gap_block(0, gap_ptr.gap_ptr(), gap_siz);
                    }
                }

                if points_new {
                    if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
                        cursor.move_forward(1);
                    }
                }

                if should_flush {
                    self.shift_right_gap_block(0, flush_gap_siz);
                    self.flush_gap(cursor_id);
                    let gaps = self.base().gaps_mut();
                    if let Some(gap) = gaps.get_mut(i) {
                        let gap_ptr = gap.gap_ptr_mut();
                        gap_ptr.set_gap_ptr(0);
                        let buff_ptr = gap_ptr.buff_mut();
                        std::ptr::write_bytes(
                            buff_ptr,
                            0,
                            flush_gap_siz * std::mem::size_of::<Raw>(),
                        );
                        gap.set_gap_bounds(flush_gap_start as *mut Raw, unsafe {
                            flush_gap_start.add(flush_gap_siz) as *mut Raw
                        });
                    }
                    should_flush = false;
                }
            }
            i += 1;
        }

        if v_space_increment > 0 {
            *self.base_mut().v_space_mut() =
                self.base().v_space().saturating_add(v_space_increment);
        }
    }

    unsafe fn put_shared(&mut self, ch: u32, points_new: bool, cursor_id: usize) {
        let gap_idx = self.base().shared_gap_index(cursor_id);
        let shared_gaps = &mut *self.base().shared_gaps().get();
        if let Some(shared_gap) =
            shared_gaps.get_mut(gap_idx.unwrap_or_else(|| usize::MAX))
        {
            let gap_size = shared_gap.gap().gap_size();
            if let Some(region) = shared_gap.cursor_region_mut(cursor_id) {
                let has_space = region.has_space();
                let region_pos = region.cursor_position();
                let _ = region;

                if ch == 0x0A {
                    self.handle_newline(cursor_id);
                } else {
                    if has_space {
                        self.write_to_available_space_direct(
                            shared_gap, region_pos, ch, points_new, gap_size, cursor_id,
                        );
                    } else {
                        self.write_to_full_region_direct(
                            shared_gap, region_pos, ch, points_new, gap_size, cursor_id,
                        );
                    }
                    self.check_and_flush_gap(shared_gap, gap_size, cursor_id);
                }

                if points_new {
                    self.advance_cursor(cursor_id);
                }
            }
        }
    }
}
