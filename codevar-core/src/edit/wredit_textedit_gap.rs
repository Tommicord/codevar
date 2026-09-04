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

use super::{StreamWritable, TextEditableGap};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableGap<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn flush_gap(&mut self, cursor_id: usize) {
        self.flush_gap(cursor_id);
    }

    unsafe fn flush_all(&mut self) -> usize {
        self.flush_all()
    }

    unsafe fn shift_gap_block(&mut self, gap_id: usize, offset: usize, size: usize) {
        let gaps = self.base().gaps_mut();
        let gap = gaps.get(gap_id);
        let ptr = self.base().active_cursor();

        if gap.is_none() || ptr.is_none() {
            return;
        }
        if let Some(unwrap_ptr) = ptr {
            let start = unwrap_ptr.array_pos();
            let end = self.base().raw_length();
            let raw_ptr = unsafe { self.base().raw_mut_ptr() };
            std::ptr::copy(
                raw_ptr.add(start),
                raw_ptr.add(start + offset),
                end - start - size,
            );
        }
    }

    unsafe fn shift_right_gap_block(&mut self, gap_id: usize, count: usize) {
        self.shift_gap_block(gap_id, count, 0);
    }

    unsafe fn shift_left_gap_block(&mut self, gap_id: usize, count: usize) {
        self.shift_gap_block(gap_id, 0, count);
    }

    unsafe fn move_gap_block(&mut self, gap_id: usize, offset: usize, size: usize) {
        self.shift_gap_block(gap_id, offset, size);
    }

    unsafe fn set_gap_bounds(&mut self, gap_id: usize, offset: usize, len: usize) {
        let gaps = self.base().gaps_mut();
        let gap = gaps.get_mut(gap_id);

        if gap.is_none() || self.base().raw_ptr().is_null() {
            return;
        }

        let raw_ptr = unsafe { self.base().raw_mut_ptr() };
        if let Some(gap_value) = gap {
            gap_value.set_gap_bounds(raw_ptr.add(offset), raw_ptr.add(offset + len));
        }
    }
}
