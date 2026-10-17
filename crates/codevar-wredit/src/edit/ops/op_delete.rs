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

//! [`TextEditableDelete`]: removal of committed text.

use codevar_logger::log_error;

use super::super::traits::TextEditableDelete;
use super::super::writable::StreamWritable;
use crate::error::{WreditError, WreditResult};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableDelete<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn delete(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // Commit the cursor's own staged run first so the deletion
        // addresses a committed layout.
        self.flush_cursor_gap(id);
        let position = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position >= self.base.raw_length {
            return Ok(0);
        }
        Ok(self.commit_removal(position, 1))
    }

    fn backspace(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.flush_cursor_gap(id);
        let position = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position == 0 {
            return Ok(0);
        }
        let at = position - 1;
        Ok(self.commit_removal(at, 1))
    }

    fn delete_range(&mut self, start: usize, end: usize) -> WreditResult<usize> {
        let raw_length = self.base.raw_length;
        if start > end || end > raw_length {
            log_error!("wredit: delete_range {start}..{end} outside the document (length {raw_length})");
            return Err(WreditError::OutOfRange {
                start: start as u64,
                end: end as u64,
            });
        }
        let count = end - start;
        if count == 0 {
            return Ok(0);
        }
        Ok(self.commit_removal(start, count))
    }
}
