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

//! [`TextEditableCursor`]: caret movement and row/column queries.

use super::super::traits::TextEditableCursor;
use super::super::writable::{StreamWritable, WritableEvent};
use crate::error::WreditResult;

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableCursor<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn move_cursor_forward(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // Element-wise movement works against the logical document,
        // so staged runs need no flush: only the clamping bound has
        // to be current.
        self.base.refresh_cursor_views();
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(|cursor| cursor.move_forward(amount))
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_backward(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.base.refresh_cursor_views();
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(|cursor| cursor.move_backward(amount))
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_next_line(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // Line navigation reads the committed layout: commit whatever
        // this cursor has staged first.
        self.flush_cursor_gap(id);
        self.base.refresh_cursor_views();
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::move_next_line)
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_prev_line(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.flush_cursor_gap(id);
        self.base.refresh_cursor_views();
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::move_prev_line)
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_to(&mut self, cursor_id: usize, row: usize, col: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.flush_cursor_gap(id);
        self.base.refresh_cursor_views();
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(|cursor| cursor.move_to(row, col))
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn cursor_row(&self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        Ok(self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::row)
            .unwrap_or(0))
    }

    fn cursor_col(&self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        Ok(self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::col)
            .unwrap_or(0))
    }

    fn cursor_width(&self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        Ok(self
            .base
            .cursor(id)
            .map(super::super::cursor::Cursor::width)
            .unwrap_or(0))
    }
}
