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

use super::{StreamWritable, TextEditableSelection};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableSelection<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn start_selection(&mut self, cursor_id: usize) {
        if let Some(cursor) = unsafe { self.base_mut().cursor_from_id_mut(cursor_id) } {
            cursor.set_selection_anchor(cursor.array_pos());
            cursor.set_selection_end(cursor.array_pos());
        }
    }

    unsafe fn extend_selection(&mut self, cursor_id: usize) {
        if let Some(cursor) = unsafe { self.base_mut().cursor_from_id_mut(cursor_id) } {
            cursor.set_selection_end(cursor.array_pos());
        }
    }

    unsafe fn clear_selection(&mut self, cursor_id: usize) {
        if let Some(cursor) = unsafe { self.base_mut().cursor_from_id_mut(cursor_id) } {
            cursor.clear_selection();
        }
    }

    unsafe fn delete_selection(&mut self, cursor_id: usize) {
        let range = self
            .base()
            .cursor_from_id(cursor_id)
            .and_then(|cursor| cursor.selection_range());
        let Some((start, end)) = range else {
            return;
        };
        let length = self.base().raw_length();
        let start = start.min(length);
        let end = end.min(length);
        let delete_start = start.min(end);
        let delete_end = end.max(start);
        self.delete_raw_range_at(delete_start, delete_end - delete_start);
        if let Some(cursor) = unsafe { self.base_mut().cursor_from_id_mut(cursor_id) } {
            cursor.clear_selection();
            cursor.set_array_pos(delete_start.min(length - (delete_end - delete_start)));
        }
    }

    unsafe fn selection_range(&self, cursor_id: usize) -> Option<(usize, usize)> {
        self.base().cursor_from_id(cursor_id)?.selection_range()
    }
}
