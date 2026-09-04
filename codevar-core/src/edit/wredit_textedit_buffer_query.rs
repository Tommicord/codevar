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

use super::{StreamWritable, TextEditableQuery, Writable};

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE> {
    unsafe fn byte_flush(&mut self) {
        let writable = self as *mut Self;
        let writable_mut = &mut *writable;
        let _ = writable_mut.flush_all();
    }

    pub unsafe fn snapshot_text(&mut self) -> Vec<u32> {
        self.byte_flush();
        let raw_length = self.base().raw_length();
        let raw_ptr = self.base().raw_ptr() as *const u32;
        let mut text = Vec::with_capacity(raw_length);
        for index in 0..raw_length {
            let packed = raw_ptr.add(index).read();
            text.push(Self::char(packed));
        }
        text
    }

    pub unsafe fn restore_text(&mut self, text: &[u32]) {
        let writable = self as *mut Self;
        let writable_mut = &mut *writable;
        let raw_ptr = writable_mut.base().raw_mut_ptr();
        if raw_ptr.is_null() {
            return;
        }
        let raw_u32 = raw_ptr as *mut u32;
        let old_len = writable_mut.base().raw_length();
        for (index, ch) in text.iter().enumerate() {
            raw_u32
                .add(index)
                .write(Self::make_packed(Self::T_COMMON, *ch));
        }
        if old_len > text.len() {
            let clear_len = old_len - text.len();
            std::ptr::write_bytes(raw_u32.add(text.len()), 0, clear_len);
        }
        let mut base_mut = writable_mut.base_mut();
        *base_mut.raw_length_mut() = text.len();
        *base_mut.v_space_mut() = text.iter().filter(|&&ch| ch == 0x0A).count();
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableQuery<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn char_at_cursor(&mut self, cursor_id: usize) -> Option<u32> {
        self.byte_flush();
        let cursor = self.base().cursor_from_id(cursor_id)?;
        let pos = cursor.array_pos();
        let text = self.snapshot_text();
        text.get(pos).copied()
    }

    unsafe fn line_width(&mut self, cursor_id: usize) -> usize {
        self.byte_flush();
        let cursor_value = self.base().cursor_from_id(cursor_id);
        let Some(cursor) = cursor_value else { return 0 };
        let row = cursor.row();
        let text = self.snapshot_text();

        if text.is_empty() {
            return 0;
        }

        let mut current_row = 0usize;
        let mut row_width = 0usize;

        for ch in text.iter().copied() {
            if ch == 0x0A {
                current_row += 1;
                if current_row > row {
                    break;
                }
                row_width = 0;
            } else if current_row == row {
                row_width += 1;
            }
        }

        row_width
    }

    fn line_count(&mut self) -> usize {
        let text = unsafe { self.snapshot_text() };
        if text.is_empty() {
            0
        } else {
            text.iter().filter(|&&ch| ch == 0x0A).count() + 1
        }
    }

    fn char_count(&mut self) -> usize {
        unsafe { self.snapshot_text().len() }
    }

    fn is_empty(&self) -> bool {
        self.base().raw_length() == 0
    }
}
