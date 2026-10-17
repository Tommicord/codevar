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

//! [`TextEditableReplace`]: swap committed text for new content.

use codevar_logger::log_error;

use super::super::traits::{TextEditablePut, TextEditableReplace};
use super::super::writable::StreamWritable;
use crate::error::{WreditError, WreditResult};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableReplace<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn replace(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // A staged run at the caret commits first, then the element
        // it landed on is removed and the replacement stages.
        self.flush_cursor_gap(id);
        let position = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position < self.base.raw_length {
            let removed = self.commit_removal(position, 1);
            if removed == 0 {
                return Err(WreditError::InvalidOperation {
                    what: "replace could not remove the covered element",
                });
            }
        }
        self.put(id, value)
    }

    fn replace_range(&mut self, start: usize, end: usize, values: &[Raw]) -> WreditResult<usize> {
        let raw_length = self.base.raw_length;
        if start > end || end > raw_length {
            log_error!("wredit: replace_range {start}..{end} outside the document (length {raw_length})");
            return Err(WreditError::OutOfRange {
                start: start as u64,
                end: end as u64,
            });
        }
        let removed = end - start;
        if removed > 0 {
            let done = self.commit_removal(start, removed);
            if done == 0 {
                return Err(WreditError::InvalidOperation {
                    what: "replace_range could not remove the old range",
                });
            }
        }
        if values.is_empty() {
            return Ok(0);
        }
        let inserted = self.commit_insertion(start, values);
        if inserted == 0 {
            // The removal already applied; the insertion was refused
            // (allocation) — report it without unwinding anything.
            return Err(WreditError::CapacityExhausted {
                requested: values.len(),
                available: 0,
            });
        }
        Ok(inserted)
    }
}
