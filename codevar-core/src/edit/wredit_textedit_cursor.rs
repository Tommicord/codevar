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

use super::{StreamWritable, TextEditableCursor};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableCursor<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn move_cursor_forward(&mut self, cursor_id: usize, count: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            for _ in 0..count {
                cursor.move_forward(1);
            }
        }
    }

    unsafe fn move_cursor_backward(&mut self, cursor_id: usize, count: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            for _ in 0..count {
                cursor.move_backward(1);
            }
        }
    }

    unsafe fn move_cursor_next_line(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            cursor.move_next_line();
        }
    }

    unsafe fn move_cursor_prev_line(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            cursor.move_prev_line();
        }
    }

    unsafe fn move_cursor_line_start(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            cursor.move_to(cursor.row(), 0);
        }
    }

    unsafe fn move_cursor_line_end(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            let width = cursor.width();
            cursor.move_to(cursor.row(), width);
        }
    }

    unsafe fn move_cursor_word_forward(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base().cursor_from_id_mut(cursor_id) {
            let current_pos = cursor.array_pos();
            let raw_len = self.base().raw_length();
            let bytes =
                std::slice::from_raw_parts(self.base().raw_ptr() as *const u8, raw_len);
            let content = String::from_utf8_lossy(bytes);

            let mut pos = current_pos;
            while pos < content.len()
                && !content
                    .chars()
                    .nth(pos)
                    .map_or(false, |c| c.is_whitespace())
            {
                pos += 1;
            }
            while pos < content.len()
                && content
                    .chars()
                    .nth(pos)
                    .map_or(false, |c| c.is_whitespace())
            {
                pos += 1;
            }
            let target_row = self.base().cursor_from_id(cursor_id).map_or(0, |c| c.row());
            let target_col = pos;
            cursor.move_to(target_row, target_col);
        }
    }

    unsafe fn move_cursor_word_backward(&mut self, cursor_id: usize) {
        if let Some(cursor) = self.base().cursor_from_id_mut(cursor_id) {
            let current_pos = cursor.array_pos();
            let raw_len = self.base().raw_length();
            let bytes =
                std::slice::from_raw_parts(self.base().raw_ptr() as *const u8, raw_len);
            let content = String::from_utf8_lossy(bytes);

            if current_pos == 0 {
                return;
            }
            let mut pos = current_pos.saturating_sub(1);
            while pos > 0
                && content
                    .chars()
                    .nth(pos)
                    .map_or(false, |c| c.is_whitespace())
            {
                pos -= 1;
            }
            while pos > 0
                && !content
                    .chars()
                    .nth(pos)
                    .map_or(false, |c| c.is_whitespace())
            {
                pos -= 1;
            }
            if pos > 0 {
                pos += 1;
            }
            let target_row = self.base().cursor_from_id(cursor_id).map_or(0, |c| c.row());
            let target_col = pos;
            cursor.move_to(target_row, target_col);
        }
    }

    unsafe fn cursor_position(&self, cursor_id: usize) -> Option<(usize, usize)> {
        let cursor = self.base().cursor_from_id(cursor_id)?;
        Some((cursor.row(), cursor.col()))
    }
}
