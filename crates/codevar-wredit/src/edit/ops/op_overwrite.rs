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

//! [`TextEditableOverwrite`]: in-place overwriting of committed text.

use codevar_logger::log_error;

use super::super::base::{T_MERGE, make_packed, packed_char};
use super::super::traits::{TextEditableOverwrite, TextEditablePut};
use super::super::writable::{StreamWritable, WritableEvent};
use crate::error::{WreditError, WreditResult};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableOverwrite<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn overwrite(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        let position = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position >= self.base.raw_length {
            // Nothing to overwrite at (or past) the end: append.
            return self.put(id, value);
        }
        // The merge element consumes the committed element that
        // follows the insertion point when the run is flushed.
        let merge = Raw::from(make_packed(packed_char(Into::<u32>::into(value)), T_MERGE));
        self.stage_private(id, merge, position)?;
        Ok(self.move_caret_to(id, position.saturating_add(1)))
    }

    fn overwrite_slice(&mut self, start: usize, values: &[Raw]) -> WreditResult<usize> {
        if values.is_empty() {
            return Ok(0);
        }
        let raw_length = self.base.raw_length;
        if start > raw_length {
            log_error!("wredit: overwrite_slice at {start} exceeds length {raw_length}");
            return Err(WreditError::InvalidPosition {
                position: start as u64,
                len: raw_length as u64,
            });
        }
        // The overlapping prefix is overwritten in place (no cursor
        // moves, no tail shifts); anything past the end appends.
        let overwritten = values.len().min(raw_length - start);
        if overwritten > 0 {
            let raw = self.base.raw_mut();
            if let Some(target) = raw
                .slice_mut()
                .get_mut(start..start + overwritten)
            {
                target.copy_from_slice(&values[..overwritten]);
            } else {
                log_error!("wredit: overwrite_slice target escaped the raw buffer");
                return Err(WreditError::InvalidOperation {
                    what: "overwrite_slice target out of bounds",
                });
            }
        }
        let mut changed = overwritten;
        let appended = values.len() - overwritten;
        if appended > 0 {
            let inserted = self.commit_insertion(raw_length, &values[overwritten..]);
            if inserted == 0 {
                return Err(WreditError::CapacityExhausted {
                    requested: appended,
                    available: 0,
                });
            }
            changed += inserted;
        }
        if changed > 0 {
            self.base
                .emit(WritableEvent::buffer_changed(start, changed));
        }
        Ok(changed)
    }
}
