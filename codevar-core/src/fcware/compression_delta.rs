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

use crate::fcware::compression_error::{Error, Result};
use crate::fcware::compression_frame::FrameKind;

/// Encodes `input` as a delta / residual frame.
pub fn delta_encode(input: &[u8]) -> Result<Vec<u8>> {
    let length = u32::try_from(input.len()).map_err(|_| Error::InputTooLarge)?;
    let mut frame = Vec::with_capacity(7 + input.len());
    frame.extend_from_slice(FrameKind::Delta.magic());
    frame.extend_from_slice(&length.to_be_bytes());
    let Some((&first, rest)) = input.split_first() else {
        return Ok(frame);
    };
    frame.push(first);
    let mut index = 0usize;
    while index < rest.len() {
        // `rest[i]` == `input[i + 1]`; previous byte is `input[i]`.
        // SAFETY: `index < rest.len()` so `index + 1 < input.len()`.
        let previous = unsafe { *input.as_ptr().add(index) };
        let current = unsafe { *rest.as_ptr().add(index) };
        let difference = current.wrapping_sub(previous);
        let mut count = 1usize;
        while index + count < rest.len() {
            let prior = unsafe { *input.as_ptr().add(index + count) };
            let next = unsafe { *rest.as_ptr().add(index + count) };
            if next.wrapping_sub(prior) != difference {
                break;
            }
            count += 1;
        }
        if difference == 0 || count > 1 {
            let mut remaining = count;
            while remaining > 0 {
                let chunk = remaining.min(usize::from(u8::MAX));
                frame.extend([0, difference, chunk as u8]);
                remaining -= chunk;
            }
        } else {
            frame.push(difference);
        }
        index += count;
    }
    Ok(frame)
}

/// Decodes a delta frame into the original byte stream.
pub fn delta_decode(frame: &[u8]) -> Result<Vec<u8>> {
    if frame.len() < 7 || FrameKind::from_magic(frame) != Some(FrameKind::Delta) {
        return Err(Error::InvalidFrame);
    }
    // SAFETY: length >= 7.
    let expected = unsafe {
        let p = frame.as_ptr().add(3);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    if expected == 0 {
        return if frame.len() == 7 {
            Ok(Vec::new())
        } else {
            Err(Error::length_mismatch(0, frame.len() - 7))
        };
    }
    let first = *frame.get(7).ok_or(Error::TruncatedFrame)?;
    let mut output = Vec::with_capacity(expected);
    output.push(first);
    let mut position = 8usize;
    while position < frame.len() {
        // SAFETY: `position < frame.len()`.
        let marker = unsafe { *frame.as_ptr().add(position) };
        position += 1;
        let (difference, count) = if marker == 0 {
            let difference = *frame.get(position).ok_or(Error::TruncatedFrame)?;
            let count = *frame.get(position + 1).ok_or(Error::TruncatedFrame)?;
            if count == 0 {
                return Err(Error::invalid_control(0));
            }
            position += 2;
            (difference, usize::from(count))
        } else {
            (marker, 1)
        };
        output.reserve(count);
        // SAFETY: we reserved `count`; each step reads the previous written byte.
        unsafe {
            let base = output.as_mut_ptr();
            let mut out_len = output.len();
            for _ in 0..count {
                let next = (*base.add(out_len - 1)).wrapping_add(difference);
                *base.add(out_len) = next;
                out_len += 1;
            }
            output.set_len(out_len);
        }
        if output.len() > expected {
            return Err(Error::length_mismatch(expected, output.len()));
        }
    }
    if output.len() == expected {
        Ok(output)
    } else {
        Err(Error::length_mismatch(expected, output.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::{delta_decode, delta_encode};
    use crate::fcware::compression_error::Error;

    #[test]
    fn roundtrip_patterns() {
        for sample in [
            &b""[..],
            &b"\x00"[..],
            &b"aaaaaaa"[..],
            &b"abcdefg"[..],
            &[0u8, 1, 2, 3, 4, 5][..],
            &[255u8, 0, 255, 0][..],
        ] {
            let frame = delta_encode(sample).expect("encode");
            assert_eq!(delta_decode(&frame).expect("decode"), sample);
        }
    }

    #[test]
    fn rejects_invalid_and_truncated() {
        assert_eq!(delta_decode(b"XX\x01"), Err(Error::InvalidFrame));
        assert_eq!(
            delta_decode(b"DL\x01\x00\x00\x00\x02\x01"),
            Err(Error::length_mismatch(2, 1))
        );
        // Truncated run record after control 0.
        let truncated = Vec::from(&b"DL\x01\x00\x00\x00\x02\x01\x00"[..]);
        assert_eq!(delta_decode(&truncated), Err(Error::TruncatedFrame));
        let mut bad = Vec::from(&b"DL\x01\x00\x00\x00\x02\x01"[..]);
        bad.extend([0, 1, 0]); // count == 0
        assert_eq!(delta_decode(&bad), Err(Error::invalid_control(0)));
    }

    #[test]
    fn long_zero_run_chunks() {
        let input = vec![7u8; 300];
        let frame = delta_encode(&input).expect("encode");
        assert_eq!(delta_decode(&frame).expect("decode"), input);
    }
}
