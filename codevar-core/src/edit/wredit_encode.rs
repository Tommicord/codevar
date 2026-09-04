//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to
//! in writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use super::{BaseWritable, WritableGapSize};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EncodingType {
    Utf8,
    Utf16,
    Utf32,
}

/// `pub struct Encoder<Raw, Buf, const GAP_SIZE: usize>`
///
/// UTF encoder integrated with the Writable data structure.
///
/// This encoder provides UTF-8, UTF-16, and UTF-32 encoding capabilities
/// that work directly with the gap buffer-based Writable structure
///
/// # Type Parameters
///
/// * `Raw` - The raw data type stored in the buffer (typically u32 for packed data)
/// * `Buf` - The buffer type for output operations (e.g., u8 for UTF-8 output)
/// * `GAP_SIZE` - The gap size in bytes determining gap buffer capacity
#[repr(C)]
pub struct Encoder<Raw, Buf, const GAP_SIZE: usize> {
    _phantom: std::marker::PhantomData<(Raw, Buf)>,
}

impl<Raw, Buf, const GAP_SIZE: usize> Encoder<Raw, Buf, GAP_SIZE> {
    /// Creates a new encoder instance
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Encodes a writable buffer to UTF-8
    ///
    /// # Arguments
    ///
    /// * `writable` - The writable buffer to encode
    /// * `output` - Output buffer for UTF-8 data
    /// * `offset` - Offset in the raw buffer to start encoding
    /// * `raw_size` - Number of elements to encode
    ///
    /// # Safety
    ///
    /// The writable must be properly initialized and the output buffer must have sufficient capacity
    pub fn encode_utf8_writable(
        &self,
        writable: &BaseWritable<Raw, Buf, GAP_SIZE>,
        output: *mut Buf,
        output_capacity: usize,
        offset: usize,
        raw_size: usize,
    ) {
        let raw_ptr = writable.raw_ptr() as *const u32;
        let output_ptr = output as *mut u8;
        let raw_len = writable.raw_length().min(offset + raw_size);

        let mut output_offset = 0usize;
        for i in offset..raw_len {
            if output_offset >= output_capacity {
                break;
            }
            let codepoint = unsafe { *raw_ptr.add(i) };
            let bytes_written = self.encode_utf8_fallback(codepoint, unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr.add(output_offset),
                    output_capacity - output_offset,
                )
            });
            output_offset += bytes_written;
        }
    }

    pub fn encode_utf16le_writable(
        &self,
        writable: &BaseWritable<Raw, Buf, GAP_SIZE>,
        output: *mut Buf,
        output_capacity: usize,
        offset: usize,
        raw_size: usize,
    ) {
        let raw_ptr = writable.raw_ptr() as *const u32;
        let output_ptr = output as *mut u8;
        let raw_len = writable.raw_length().min(offset + raw_size);

        let mut output_offset = 0usize;
        for i in offset..raw_len {
            if output_offset + 2 > output_capacity {
                break;
            }
            let codepoint = unsafe { *raw_ptr.add(i) };
            let bytes_written = self.encode_utf16le_fallback(codepoint, unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr.add(output_offset),
                    output_capacity - output_offset,
                )
            });
            output_offset += bytes_written;
        }
    }

    pub fn encode_utf16be_writable(
        &self,
        writable: &BaseWritable<Raw, Buf, GAP_SIZE>,
        output: *mut Buf,
        output_capacity: usize,
        offset: usize,
        raw_size: usize,
    ) {
        let raw_ptr = writable.raw_ptr() as *const u32;
        let output_ptr = output as *mut u8;
        let raw_len = writable.raw_length().min(offset + raw_size);

        let mut output_offset = 0usize;
        for i in offset..raw_len {
            if output_offset + 2 > output_capacity {
                break;
            }
            let codepoint = unsafe { *raw_ptr.add(i) };
            let bytes_written = self.encode_utf16be_fallback(codepoint, unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr.add(output_offset),
                    output_capacity - output_offset,
                )
            });
            output_offset += bytes_written;
        }
    }

    pub fn encode_utf32le_writable(
        &self,
        writable: &BaseWritable<Raw, Buf, GAP_SIZE>,
        output: *mut Buf,
        output_capacity: usize,
        offset: usize,
        raw_size: usize,
    ) {
        let raw_ptr = writable.raw_ptr() as *const u32;
        let output_ptr = output as *mut u8;
        let raw_len = writable.raw_length().min(offset + raw_size);

        let mut output_offset = 0usize;
        for i in offset..raw_len {
            if output_offset + 4 > output_capacity {
                break;
            }
            let codepoint = unsafe { *raw_ptr.add(i) };
            let bytes_written = self.encode_utf32le_fallback(codepoint, unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr.add(output_offset),
                    output_capacity - output_offset,
                )
            });
            output_offset += bytes_written;
        }
    }

    pub fn encode_utf32be_writable(
        &self,
        writable: &BaseWritable<Raw, Buf, GAP_SIZE>,
        output: *mut Buf,
        output_capacity: usize,
        offset: usize,
        raw_size: usize,
    ) {
        let raw_ptr = writable.raw_ptr() as *const u32;
        let output_ptr = output as *mut u8;
        let raw_len = writable.raw_length().min(offset + raw_size);

        let mut output_offset = 0usize;
        for i in offset..raw_len {
            if output_offset + 4 > output_capacity {
                break;
            }
            let codepoint = unsafe { *raw_ptr.add(i) };
            let bytes_written = self.encode_utf32be_fallback(codepoint, unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr.add(output_offset),
                    output_capacity - output_offset,
                )
            });
            output_offset += bytes_written;
        }
    }

    fn encode_utf8_fallback(&self, codepoint: u32, buffer: &mut [u8]) -> usize {
        if codepoint <= 0x7F {
            if !buffer.is_empty() {
                buffer[0] = codepoint as u8;
            }
            1
        } else if codepoint <= 0x7FF {
            if buffer.len() >= 2 {
                buffer[0] = 0xC0 | ((codepoint >> 6) & 0x1F) as u8;
                buffer[1] = 0x80 | (codepoint & 0x3F) as u8;
            }
            2
        } else if codepoint <= 0xFFFF {
            if buffer.len() >= 3 {
                buffer[0] = 0xE0 | ((codepoint >> 12) & 0x0F) as u8;
                buffer[1] = 0x80 | ((codepoint >> 6) & 0x3F) as u8;
                buffer[2] = 0x80 | (codepoint & 0x3F) as u8;
            }
            3
        } else if codepoint <= 0x10FFFF {
            if buffer.len() >= 4 {
                buffer[0] = 0xF0 | ((codepoint >> 18) & 0x07) as u8;
                buffer[1] = 0x80 | ((codepoint >> 12) & 0x3F) as u8;
                buffer[2] = 0x80 | ((codepoint >> 6) & 0x3F) as u8;
                buffer[3] = 0x80 | (codepoint & 0x3F) as u8;
            }
            4
        } else {
            0
        }
    }

    fn encode_utf16le_fallback(&self, codepoint: u32, buffer: &mut [u8]) -> usize {
        if codepoint <= 0xFFFF {
            if buffer.len() >= 2 {
                let bytes = (codepoint as u16).to_le_bytes();
                buffer[0] = bytes[0];
                buffer[1] = bytes[1];
            }
            2
        } else if codepoint <= 0x10FFFF {
            if buffer.len() >= 4 {
                let surrogate = codepoint.saturating_sub(0x10000);
                let high_surrogate = 0xD800 + ((surrogate >> 10) & 0x3FF) as u16;
                let low_surrogate = 0xDC00 + (surrogate & 0x3FF) as u16;

                let high_bytes = high_surrogate.to_le_bytes();
                let low_bytes = low_surrogate.to_le_bytes();

                buffer[0] = high_bytes[0];
                buffer[1] = high_bytes[1];
                buffer[2] = low_bytes[0];
                buffer[3] = low_bytes[1];
            }
            4
        } else {
            0
        }
    }

    fn encode_utf16be_fallback(&self, codepoint: u32, buffer: &mut [u8]) -> usize {
        if codepoint <= 0xFFFF {
            if buffer.len() >= 2 {
                let bytes = (codepoint as u16).to_be_bytes();
                buffer[0] = bytes[0];
                buffer[1] = bytes[1];
            }
            2
        } else if codepoint <= 0x10FFFF {
            if buffer.len() >= 4 {
                let surrogate = codepoint.saturating_sub(0x10000);
                let high_surrogate = 0xD800 + ((surrogate >> 10) & 0x3FF) as u16;
                let low_surrogate = 0xDC00 + (surrogate & 0x3FF) as u16;

                let high_bytes = high_surrogate.to_be_bytes();
                let low_bytes = low_surrogate.to_be_bytes();

                buffer[0] = high_bytes[0];
                buffer[1] = high_bytes[1];
                buffer[2] = low_bytes[0];
                buffer[3] = low_bytes[1];
            }
            4
        } else {
            0
        }
    }

    fn encode_utf32le_fallback(&self, codepoint: u32, buffer: &mut [u8]) -> usize {
        if buffer.len() >= 4 {
            let bytes = codepoint.to_le_bytes();
            buffer[0] = bytes[0];
            buffer[1] = bytes[1];
            buffer[2] = bytes[2];
            buffer[3] = bytes[3];
        }
        4
    }

    fn encode_utf32be_fallback(&self, codepoint: u32, buffer: &mut [u8]) -> usize {
        if buffer.len() >= 4 {
            let bytes = codepoint.to_be_bytes();
            buffer[0] = bytes[0];
            buffer[1] = bytes[1];
            buffer[2] = bytes[2];
            buffer[3] = bytes[3];
        }
        4
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Default for Encoder<Raw, Buf, GAP_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {}
