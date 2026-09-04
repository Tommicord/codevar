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

use super::{StreamWritable, TextEditableDelete};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableDelete<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn delete(&mut self, cursor_id: usize) {
        self.delete_raw_range(cursor_id, 1);
    }

    unsafe fn backspace(&mut self, cursor_id: usize) {
        let cursor_pos = self
            .base()
            .cursor_from_id(cursor_id)
            .map(|cursor| cursor.array_pos())
            .unwrap_or(0);
        if cursor_pos > 0 {
            self.delete_raw_range_at(cursor_pos - 1, 1);
        }
    }

    unsafe fn delete_range(&mut self, cursor_id: usize, count: usize) {
        let cursor_pos = self
            .base()
            .cursor_from_id(cursor_id)
            .map(|cursor| cursor.array_pos())
            .unwrap_or(0);
        self.delete_raw_range(cursor_id, count);
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE> {
    pub(crate) unsafe fn delete_raw_range(&mut self, cursor_id: usize, count: usize) {
        let start = self
            .base()
            .cursor_from_id(cursor_id)
            .map(|cursor| cursor.array_pos())
            .unwrap_or(0);
        self.delete_raw_range_at(start, count);
    }

    pub(crate) unsafe fn delete_raw_range_at(&mut self, start: usize, count: usize) {
        self.flush_all();
        let length = self.base().raw_length();
        let start = start.min(length);
        let end = start.saturating_add(count).min(length);
        if start == end {
            return;
        }

        let raw = self.base_mut().raw_mut_ptr() as *mut u32;
        std::ptr::copy(raw.add(end), raw.add(start), length - end);
        std::ptr::write_bytes(raw.add(length - (end - start)), 0, end - start);
        *self.base_mut().raw_length_mut() = length - (end - start);
        self.base_mut().adjust_cursors_after_delete(start, end);
        self.base_mut().reset_gap_at(start);
    }
}
