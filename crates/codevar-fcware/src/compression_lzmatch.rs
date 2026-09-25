//! Copyright 2026 Codevar Project
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

use crate::compression_error::{CompressorError, CompressorResult};
use crate::compression_frame::{FrameKind, extend_bytes, frame_read_u16};
use alloc::vec::Vec;

const HEADER_SIZE: usize = 7;
/// Minimum match length encoded in an LZ-match frame.
pub const MIN_MATCH: usize = 4;

/// Decodes an LZ-match frame into the original byte stream.
pub fn lz_match_decode(frame: &[u8]) -> CompressorResult<Vec<u8>> {
    if frame.len() < HEADER_SIZE {
        return Err(CompressorError::InvalidFrame);
    }
    if FrameKind::from_magic(frame) != Some(FrameKind::LzMatch) {
        return Err(CompressorError::InvalidFrame);
    }
    // SAFETY: length checked above; bytes 3..7 are in-bounds.
    let expected = unsafe {
        let p = frame
            .as_ptr()
            .add(3);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    let mut output = Vec::with_capacity(expected);
    let mut position = HEADER_SIZE;

    while position < frame.len() {
        // SAFETY: `position < frame.len()`.
        let marker = unsafe {
            *frame
                .as_ptr()
                .add(position)
        };
        position += 1;
        match marker {
            0 => {
                let length = usize::from(frame_read_u16(frame, &mut position)?);
                let end = position
                    .checked_add(length)
                    .ok_or(CompressorError::TruncatedFrame)?;
                if end > frame.len() {
                    return Err(CompressorError::TruncatedFrame);
                }
                // SAFETY: `position..end` is within `frame`.
                let literals = unsafe {
                    core::slice::from_raw_parts(
                        frame
                            .as_ptr()
                            .add(position),
                        length,
                    )
                };
                extend_bytes(&mut output, literals);
                position = end;
            }
            1 => {
                let distance = usize::from(frame_read_u16(frame, &mut position)?);
                let length = usize::from(frame_read_u16(frame, &mut position)?);
                if distance == 0 || distance > output.len() || length < MIN_MATCH {
                    return Err(CompressorError::InvalidMatch);
                }
                append_match(&mut output, distance, length);
            }
            marker => return Err(CompressorError::invalid_token(marker)),
        }
        if output.len() > expected {
            return Err(CompressorError::length_mismatch(expected, output.len()));
        }
    }
    if output.len() == expected {
        Ok(output)
    } else {
        Err(CompressorError::length_mismatch(expected, output.len()))
    }
}

/// Appends a back-reference of `length` bytes at `distance`.
///
/// Caller must ensure `distance > 0`, `distance <= output.len()`, and capacity.
#[inline]
fn append_match(output: &mut Vec<u8>, distance: usize, length: usize) {
    let start = output.len() - distance;
    output.reserve(length);
    if distance >= length {
        // Non-overlapping: bulk copy.
        // SAFETY: `start + length <= output.len()` because `distance >= length`,
        // and reserved capacity covers the new length; source and dest do not overlap.
        unsafe {
            let src = output
                .as_ptr()
                .add(start);
            let dst = output
                .as_mut_ptr()
                .add(output.len());
            core::ptr::copy_nonoverlapping(src, dst, length);
            output.set_len(output.len() + length);
        }
    } else {
        // Overlapping RLE-style match: byte-by-byte via raw pointer.
        // SAFETY: each read index `start + i` is `< output.len() + i` and we grow by one,
        // so every read is of a previously written byte.
        unsafe {
            let base = output.as_mut_ptr();
            let mut out_len = output.len();
            for index in 0..length {
                let value = *base.add(start + index);
                *base.add(out_len) = value;
                out_len += 1;
            }
            output.set_len(out_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MIN_MATCH, lz_match_decode};
    use crate::compression::lz_match_encode;
    use crate::compression_error::CompressorError;
    use alloc::vec::Vec;

    #[test]
    fn min_match_constant() {
        assert_eq!(MIN_MATCH, 4);
    }

    #[test]
    fn roundtrip_and_invalid_frames() {
        let input = b"01234567890123456789_0123456789";
        let frame = lz_match_encode(input).expect("encode");
        assert_eq!(lz_match_decode(&frame).expect("decode"), input);

        assert_eq!(lz_match_decode(b"XX"), Err(CompressorError::InvalidFrame));
        assert_eq!(
            lz_match_decode(b"LM\x01\x00\x00\x00\x01"),
            Err(CompressorError::length_mismatch(1, 0))
        );
    }

    #[test]
    fn rejects_bad_match_and_token() {
        // LM + len=4 + match with distance 0
        let mut frame = Vec::from(&b"LM\x01\x00\x00\x00\x04"[..]);
        frame.extend_from_slice(&[1, 0, 0, 0, 4]);
        assert_eq!(lz_match_decode(&frame), Err(CompressorError::InvalidMatch));

        let mut frame = Vec::from(&b"LM\x01\x00\x00\x00\x00"[..]);
        frame.push(9);
        assert_eq!(lz_match_decode(&frame), Err(CompressorError::invalid_token(9)));
    }

    #[test]
    fn overlapping_match_roundtrip() {
        // Highly repetitive so encoder emits overlapping-style matches.
        let input = vec![b'A'; 64];
        let frame = lz_match_encode(&input).expect("encode");
        assert_eq!(lz_match_decode(&frame).expect("decode"), input);
    }
}
