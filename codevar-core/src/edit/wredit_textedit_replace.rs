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

use super::{StreamWritable, TextEditablePut, TextEditableReplace};

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableReplace<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn replace(&mut self, ch: u32, cursor_id: usize) {
        let cursor_pos = self
            .base()
            .cursor_from_id(cursor_id)
            .map(|cursor| cursor.array_pos())
            .unwrap_or(0);
        self.flush_all();
        if cursor_pos < self.base().raw_length() {
            let raw = self.base_mut().raw_mut_ptr() as *mut u32;
            raw.add(cursor_pos)
                .write(Self::make_packed(Self::T_COMMON, ch));
        } else {
            self.put(ch, false, cursor_id);
        }
    }

    unsafe fn replace_range(&mut self, chars: &[u32], cursor_id: usize, count: usize) {
        let cursor_pos = self
            .base()
            .cursor_from_id(cursor_id)
            .map(|cursor| cursor.array_pos())
            .unwrap_or(0);
        let start = cursor_pos.min(self.base().raw_length());
        self.delete_raw_range_at(start, count);
        if let Some(cursor) = self.base_mut().cursor_from_id_mut(cursor_id) {
            cursor.set_array_pos(start);
        }
        for &ch in chars {
            self.put(ch, true, cursor_id);
        }
    }
}
