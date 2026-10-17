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

//! [`TextEditableSelection`]: cursor selection management.

use codevar_logger::log_error;

use super::super::traits::TextEditableSelection;
use super::super::writable::{StreamWritable, WritableEvent};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableSelection<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn set_selection_anchor(&mut self, cursor_id: usize, position: usize) -> bool {
        let Ok(id) = self.validate_cursor(cursor_id) else {
            return false;
        };
        // The clamping bound must cover staged positions first.
        self.base.refresh_cursor_views();
        let Some(cursor) = self.base.cursor(id) else {
            return false;
        };
        cursor.set_selection_anchor(position);
        self.base
            .emit(WritableEvent::selection_changed(id));
        true
    }

    fn set_selection_end(&mut self, cursor_id: usize, position: usize) -> bool {
        let Ok(id) = self.validate_cursor(cursor_id) else {
            return false;
        };
        self.base.refresh_cursor_views();
        let Some(cursor) = self.base.cursor(id) else {
            return false;
        };
        cursor.set_selection_end(position);
        self.base
            .emit(WritableEvent::selection_changed(id));
        true
    }

    fn clear_selection(&mut self, cursor_id: usize) -> bool {
        let Ok(id) = self.validate_cursor(cursor_id) else {
            return false;
        };
        let Some(cursor) = self.base.cursor(id) else {
            return false;
        };
        cursor.clear_selection();
        self.base
            .emit(WritableEvent::selection_changed(id));
        true
    }

    fn selection_range(&self, cursor_id: usize) -> Option<(usize, usize)> {
        match self.base.cursor(cursor_id) {
            Some(cursor) => cursor.selection_range(),
            None => {
                log_error!("wredit: selection query for dead cursor {cursor_id}");
                None
            }
        }
    }
}
