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

//! [`TextEditableBufferQuery`]: read-only views of the raw document.

use super::super::base::{packed_char, packed_type};
use super::super::traits::TextEditableBufferQuery;
use super::super::writable::StreamWritable;

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableBufferQuery<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn raw_slice(&self, start: usize, end: usize) -> Option<&[Raw]> {
        if start > end {
            return None;
        }
        // The raw buffer's initialized count mirrors `raw_length`.
        let slice = self.base.raw_ref().slice();
        if end > slice.len() {
            return None;
        }
        slice.get(start..end)
    }

    fn packed_at(&self, position: usize) -> Option<Raw> {
        self.raw_slice(position, position.checked_add(1)?)
            .and_then(|elements| elements.first().copied())
    }

    fn type_at(&self, position: usize) -> Option<u32> {
        self.packed_at(position)
            .map(|element| packed_type(Into::<u32>::into(element)))
    }

    fn char_at(&self, position: usize) -> Option<u32> {
        self.packed_at(position)
            .map(|element| packed_char(Into::<u32>::into(element)))
    }

    fn line_width_at(&self, position: usize) -> Option<usize> {
        if position > self.base.raw_length {
            return None;
        }
        let elements = self.base.raw_ref().slice();
        let position = position.min(elements.len());
        let newline = u32::from(b'\n');
        let mut start = position;
        while start > 0 {
            let previous: u32 = elements[start - 1].into();
            if packed_char(previous) == newline {
                break;
            }
            start -= 1;
        }
        let mut end = position;
        while end < elements.len() {
            let current: u32 = elements[end].into();
            if packed_char(current) == newline {
                break;
            }
            end += 1;
        }
        Some(end - start)
    }
}
