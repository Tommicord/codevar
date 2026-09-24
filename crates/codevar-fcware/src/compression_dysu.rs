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
use crate::compression_frame::{FrameKind, extend_bytes, frame_read_u16, frame_read_u32};
use alloc::vec::Vec;

/// Minimum substring match length (also the hash key width).
pub const SU_LENGTH: usize = 8;

fn substring_tokens(
    input: &[u8],
    block_end: usize,
    positions: &mut std::collections::HashMap<[u8; SU_LENGTH], Vec<usize>>,
) -> CompressorResult<Vec<u8>> {
    let mut tokens = Vec::new();
    let mut literals = Vec::new();
    let mut index = 0usize;
    while index < block_end {
        let mut best = None;
        if index + SU_LENGTH <= input.len() {
            let mut key = [0u8; SU_LENGTH];
            // SAFETY: `index + SU_LENGTH <= input.len()`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(index),
                    key.as_mut_ptr(),
                    SU_LENGTH,
                );
            }
            if let Some(starts) = positions.get(&key) {
                for &start in starts.iter().rev() {
                    let offset = index
                        .checked_sub(start)
                        .ok_or(CompressorError::InvalidMatch)?;
                    if offset == 0 || offset > usize::from(u16::MAX) {
                        continue;
                    }
                    let limit = (block_end - index).min(usize::from(u8::MAX));
                    let length = match_prefix(input, start, index, limit);
                    if length >= SU_LENGTH {
                        best = Some((offset, length));
                        break;
                    }
                }
            }
        }
        if let Some((offset, length)) = best {
            if !literals.is_empty() {
                flush_substring_literals(&mut tokens, &mut literals);
            }
            tokens.push(1);
            tokens.extend_from_slice(&(offset as u16).to_be_bytes());
            tokens.push(length as u8);
            index_substring_positions(input, index, length, positions);
            index += length;
        } else {
            // SAFETY: `index < block_end <= input.len()` for relative slices, or
            // `index < input.len()` when encoding a whole buffer.
            let byte = *input.get(index).ok_or(CompressorError::TruncatedFrame)?;
            literals.push(byte);
            if index + SU_LENGTH <= input.len() {
                let mut key = [0u8; SU_LENGTH];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        input.as_ptr().add(index),
                        key.as_mut_ptr(),
                        SU_LENGTH,
                    );
                }
                positions.entry(key).or_default().push(index);
            }
            index += 1;
        }
    }
    if !literals.is_empty() {
        flush_substring_literals(&mut tokens, &mut literals);
    }
    Ok(tokens)
}

#[inline]
fn match_prefix(input: &[u8], a: usize, b: usize, limit: usize) -> usize {
    let limit = limit
        .min(input.len().saturating_sub(a))
        .min(input.len().saturating_sub(b));
    let mut length = 0usize;
    while length + 8 <= limit {
        // SAFETY: `a + length + 8 <= input.len()` and same for `b`.
        let equal = unsafe {
            let left = input.as_ptr().add(a + length);
            let right = input.as_ptr().add(b + length);
            let la = u64::from_le_bytes([
                *left,
                *left.add(1),
                *left.add(2),
                *left.add(3),
                *left.add(4),
                *left.add(5),
                *left.add(6),
                *left.add(7),
            ]);
            let lb = u64::from_le_bytes([
                *right,
                *right.add(1),
                *right.add(2),
                *right.add(3),
                *right.add(4),
                *right.add(5),
                *right.add(6),
                *right.add(7),
            ]);
            if la == lb {
                8
            } else {
                (la ^ lb).trailing_zeros() as usize / 8
            }
        };
        if equal != 8 {
            return length + equal;
        }
        length += 8;
    }
    while length < limit {
        // SAFETY: `a + length < input.len()` and same for `b` by `limit` clamp.
        let (left, right) = unsafe {
            (
                *input.as_ptr().add(a + length),
                *input.as_ptr().add(b + length),
            )
        };
        if left != right {
            break;
        }
        length += 1;
    }
    length
}

fn flush_substring_literals(tokens: &mut Vec<u8>, literals: &mut Vec<u8>) {
    let mut offset = 0usize;
    while offset < literals.len() {
        let chunk = (literals.len() - offset).min(usize::from(u8::MAX));
        tokens.push(0);
        tokens.push(chunk as u8);
        tokens.extend_from_slice(&literals[offset..offset + chunk]);
        offset += chunk;
    }
    literals.clear();
}

fn index_substring_positions(
    input: &[u8],
    start: usize,
    length: usize,
    positions: &mut std::collections::HashMap<[u8; SU_LENGTH], Vec<usize>>,
) {
    let end = start.saturating_add(length).min(input.len());
    for index in start..end {
        if index + SU_LENGTH <= input.len() {
            let mut key = [0u8; SU_LENGTH];
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(index),
                    key.as_mut_ptr(),
                    SU_LENGTH,
                );
            }
            let chain = positions.entry(key).or_default();
            chain.push(index);
            if chain.len() > 32 {
                chain.drain(..chain.len() - 32);
            }
        }
    }
}

fn decode_substring_tokens(
    tokens: &[u8],
    expected: usize,
    output: &mut Vec<u8>,
) -> CompressorResult<()> {
    let mut position = 0usize;
    while position < tokens.len() {
        // SAFETY: `position < tokens.len()`.
        let marker = unsafe { *tokens.as_ptr().add(position) };
        match marker {
            0 => {
                position += 1;
                let length = usize::from(
                    *tokens
                        .get(position)
                        .ok_or(CompressorError::TruncatedFrame)?,
                );
                position += 1;
                let end = position
                    .checked_add(length)
                    .ok_or(CompressorError::TruncatedFrame)?;
                if end > tokens.len() {
                    return Err(CompressorError::TruncatedFrame);
                }
                let literals = unsafe {
                    core::slice::from_raw_parts(tokens.as_ptr().add(position), length)
                };
                extend_bytes(output, literals);
                position = end;
            }
            1 => {
                position += 1;
                let offset = usize::from(frame_read_u16(tokens, &mut position)?);
                let length = usize::from(
                    *tokens
                        .get(position)
                        .ok_or(CompressorError::TruncatedFrame)?,
                );
                position += 1;
                if offset == 0 || offset > output.len() || length < SU_LENGTH {
                    return Err(CompressorError::InvalidMatch);
                }
                let start = output.len() - offset;
                output.reserve(length);
                if offset >= length {
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            output.as_ptr().add(start),
                            output.as_mut_ptr().add(output.len()),
                            length,
                        );
                        output.set_len(output.len() + length);
                    }
                } else {
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
            marker => return Err(CompressorError::invalid_token(marker)),
        }
        if output.len() > expected {
            return Err(CompressorError::length_mismatch(expected, output.len()));
        }
    }
    Ok(())
}

/// Encodes `input` as a fixed-window substring frame.
pub fn substring_encode(input: &[u8]) -> CompressorResult<Vec<u8>> {
    let length =
        u32::try_from(input.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut positions = std::collections::HashMap::new();
    let tokens = substring_tokens(input, input.len(), &mut positions)?;
    let mut frame = Vec::with_capacity(7 + tokens.len());
    frame.extend_from_slice(FrameKind::Substring.magic());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&tokens);
    Ok(frame)
}

/// Decodes a substring frame.
pub fn substring_decode(frame: &[u8]) -> CompressorResult<Vec<u8>> {
    if frame.len() < 7 || FrameKind::from_magic(frame) != Some(FrameKind::Substring) {
        return Err(CompressorError::InvalidFrame);
    }
    let expected = unsafe {
        let p = frame.as_ptr().add(3);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    let mut output = Vec::with_capacity(expected);
    decode_substring_tokens(&frame[7..], expected, &mut output)?;
    if output.len() == expected {
        Ok(output)
    } else {
        Err(CompressorError::length_mismatch(expected, output.len()))
    }
}

/// Encodes `input` as blocked dynamic-substring (`Dysu`) frames.
pub fn dynamic_substring_encode(
    input: &[u8],
    block_size: usize,
) -> CompressorResult<Vec<u8>> {
    if block_size == 0 || block_size > usize::from(u16::MAX) {
        return Err(CompressorError::InputTooLarge);
    }
    let length =
        u32::try_from(input.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut blocks = Vec::new();
    let mut index = 0usize;
    while index < input.len() {
        let end = (index + block_size).min(input.len());
        let mut positions = std::collections::HashMap::new();
        blocks.push(substring_tokens(
            &input[index..end],
            end - index,
            &mut positions,
        )?);
        index = end;
    }
    let block_count =
        u32::try_from(blocks.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut frame = Vec::new();
    frame.extend_from_slice(FrameKind::Dysu.magic());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&(block_size as u16).to_be_bytes());
    frame.extend_from_slice(&block_count.to_be_bytes());
    for block in blocks {
        let block_length =
            u32::try_from(block.len()).map_err(|_| CompressorError::InputTooLarge)?;
        frame.extend_from_slice(&block_length.to_be_bytes());
        frame.extend_from_slice(&block);
    }
    Ok(frame)
}

/// Decodes a Dysu frame.
pub fn dynamic_substring_decode(frame: &[u8]) -> CompressorResult<Vec<u8>> {
    if frame.len() < 13 || FrameKind::from_magic(frame) != Some(FrameKind::Dysu) {
        return Err(CompressorError::InvalidFrame);
    }
    let expected = unsafe {
        let p = frame.as_ptr().add(3);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    let block_count = unsafe {
        let p = frame.as_ptr().add(9);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    let mut output = Vec::with_capacity(expected);
    let mut position = 13usize;
    for _ in 0..block_count {
        let block_length = frame_read_u32(frame, &mut position)? as usize;
        let end = position
            .checked_add(block_length)
            .ok_or(CompressorError::TruncatedFrame)?;
        let block = frame
            .get(position..end)
            .ok_or(CompressorError::TruncatedFrame)?;
        decode_substring_tokens(block, expected, &mut output)?;
        position = end;
    }
    if position != frame.len() || output.len() != expected {
        return Err(CompressorError::length_mismatch(expected, output.len()));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        SU_LENGTH, dynamic_substring_decode, dynamic_substring_encode, substring_decode,
        substring_encode,
    };
    use crate::compression_error::CompressorError;
    use alloc::vec::Vec;

    #[test]
    fn substring_roundtrip() {
        let samples: &[&[u8]] = &[
            b"",
            b"short",
            b"abcdefghijklmnopabcdefghijklmnop",
            &[0xAAu8; 64],
        ];
        for sample in samples {
            let frame = substring_encode(sample).expect("encode");
            assert_eq!(substring_decode(&frame).expect("decode"), *sample);
        }
    }

    #[test]
    fn dysu_roundtrip_and_limits() {
        let input: Vec<u8> = (0..200).map(|v| (v % 17) as u8).collect();
        let frame = dynamic_substring_encode(&input, 32).expect("encode");
        assert_eq!(dynamic_substring_decode(&frame).expect("decode"), input);
        assert_eq!(
            dynamic_substring_encode(&input, 0),
            Err(CompressorError::InputTooLarge)
        );
        assert_eq!(SU_LENGTH, 8);
    }

    #[test]
    fn long_literal_run_chunks() {
        // No repeats of 8+ identical windows — force long literal flush.
        let input: Vec<u8> = (0..400).map(|v| v as u8).collect();
        let frame = substring_encode(&input).expect("encode");
        assert_eq!(substring_decode(&frame).expect("decode"), input);
    }

    #[test]
    fn rejects_bad_magic() {
        assert_eq!(
            substring_decode(b"XX\x01"),
            Err(CompressorError::InvalidFrame)
        );
        assert_eq!(
            dynamic_substring_decode(b"SX\x01"),
            Err(CompressorError::InvalidFrame)
        );
    }
}
