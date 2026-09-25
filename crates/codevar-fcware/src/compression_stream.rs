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
use crate::compression_frame::{FrameKind, write_bytes};
use crate::compression_lzmatch::MIN_MATCH;

/// Sliding-window history length (also used as a bitmask modulus).
pub const HISTORY_LIMIT: usize = u16::MAX as usize;
/// Maximum hash-chain probes per position.
pub const HASH_CHAIN_LIMIT: usize = 32;
/// Number of hash-table buckets.
pub const HASH_TABLE_SIZE: usize = 1 << 16;
/// Sentinel for an empty hash / chain slot.
pub const NO_POSITION: u32 = u32::MAX;

/// Borrowed LZ match workspace (hash heads + previous-position chain).
pub struct LzWorkspace<'a> {
    /// Hash-table head positions (`HASH_TABLE_SIZE` entries).
    pub heads: &'a mut [u32],
    /// Previous-position chain (`HISTORY_LIMIT + 1` entries).
    pub previous: &'a mut [u32],
}

impl<'a> LzWorkspace<'a> {
    /// Validates workspace buffer sizes and wraps them.
    pub fn new(heads: &'a mut [u32], previous: &'a mut [u32]) -> CompressorResult<Self> {
        if heads.len() < HASH_TABLE_SIZE || previous.len() < HISTORY_LIMIT + 1 {
            return Err(CompressorError::OutputTooSmall);
        }
        Ok(Self { heads, previous })
    }

    fn reset(&mut self) {
        self.heads[..HASH_TABLE_SIZE].fill(NO_POSITION);
        self.previous[..HISTORY_LIMIT + 1].fill(NO_POSITION);
    }
}

/// Streaming LZ encoder that reuses an external workspace across blocks.
pub struct StreamingEncoder<'a> {
    workspace: LzWorkspace<'a>,
}

impl<'a> StreamingEncoder<'a> {
    /// Creates a streaming encoder over the given workspace buffers.
    pub fn new(heads: &'a mut [u32], previous: &'a mut [u32]) -> CompressorResult<Self> {
        Ok(Self {
            workspace: LzWorkspace::new(heads, previous)?,
        })
    }

    /// Encodes one input block into `output`, returning bytes written.
    pub fn encode_block(&mut self, input: &[u8], output: &mut [u8]) -> CompressorResult<usize> {
        compress_block_into(input, output, &mut self.workspace)
    }
}

fn hash4(bytes: &[u8]) -> usize {
    usize::from(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u16)
}

/// Compresses `input` into an LZ-match frame written into `output`.
pub fn compress_block_into(
    input: &[u8],
    output: &mut [u8],
    workspace: &mut LzWorkspace<'_>,
) -> CompressorResult<usize> {
    let input_length = u32::try_from(input.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut cursor = 0usize;
    write_bytes(output, &mut cursor, FrameKind::LzMatch.magic())?;
    write_bytes(output, &mut cursor, &input_length.to_be_bytes())?;
    workspace.reset();
    let mut literal_start = 0usize;
    let mut position = 0usize;
    while position < input.len() {
        let mut best_position = NO_POSITION;
        let mut best_length = 0usize;
        if position + MIN_MATCH <= input.len() {
            let key = hash4(&input[position..position + MIN_MATCH]);
            let mut candidate = workspace.heads[key];
            let mut checked = 0usize;
            while candidate != NO_POSITION && checked < HASH_CHAIN_LIMIT {
                let previous_position = candidate as usize;
                let distance = position.saturating_sub(previous_position);
                if distance == 0 || distance > HISTORY_LIMIT {
                    break;
                }
                let limit = (input.len() - position).min(usize::from(u16::MAX));
                let length = match_length(input, previous_position, position, limit);
                if length > best_length {
                    best_position = candidate;
                    best_length = length;
                }
                candidate = workspace.previous[previous_position & HISTORY_LIMIT];
                checked += 1;
            }
        }
        if best_length >= MIN_MATCH {
            flush_literal_slice(input, literal_start, position, output, &mut cursor)?;
            let distance = u16::try_from(position - best_position as usize)
                .map_err(|_| CompressorError::InvalidMatch)?;
            let length = u16::try_from(best_length).map_err(|_| CompressorError::InvalidMatch)?;
            write_bytes(output, &mut cursor, &[1])?;
            write_bytes(output, &mut cursor, &distance.to_be_bytes())?;
            write_bytes(output, &mut cursor, &length.to_be_bytes())?;
            for index in 0..best_length {
                insert_position_flat(input, position + index, workspace);
            }
            position += best_length;
            literal_start = position;
        } else {
            insert_position_flat(input, position, workspace);
            position += 1;
        }
    }
    flush_literal_slice(input, literal_start, input.len(), output, &mut cursor)?;
    Ok(cursor)
}

/// Compares two slices and returns the common prefix length, using SIMD when available.
#[inline]
fn match_length(input: &[u8], a: usize, b: usize, limit: usize) -> usize {
    let mut length = 0usize;
    while length + 16 <= limit {
        // SAFETY: `a + length + 16 <= a + limit` and both regions are within `input`
        // because `limit <= input.len() - b` and `a + limit <= input.len()` from caller.
        let equal = unsafe { load_cmp16(input.as_ptr().add(a + length), input.as_ptr().add(b + length)) };
        if equal != 16 {
            return length + equal;
        }
        length += 16;
    }
    while length < limit && input[a + length] == input[b + length] {
        length += 1;
    }
    length
}

/// Returns how many of the first 16 bytes are equal (0..=16).
///
/// # Safety
///
/// `a` and `b` must each point to at least 16 readable bytes.
#[inline]
unsafe fn load_cmp16(a: *const u8, b: *const u8) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: caller guarantees 16 readable bytes at `a` and `b`; SSE2 is detected.
            return unsafe { load_cmp16_sse2(a, b) };
        }
    }
    let mut equal = 0usize;
    while equal < 16 {
        // SAFETY: `equal < 16` and both pointers have 16 valid bytes.
        let (left, right) = unsafe { (*a.add(equal), *b.add(equal)) };
        if left != right {
            break;
        }
        equal += 1;
    }
    equal
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn load_cmp16_sse2(a: *const u8, b: *const u8) -> usize {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};

    // SAFETY: caller guarantees 16 readable bytes; SSE2 enabled via target_feature.
    unsafe {
        let va = _mm_loadu_si128(a.cast());
        let vb = _mm_loadu_si128(b.cast());
        let eq = _mm_cmpeq_epi8(va, vb);
        let mask = _mm_movemask_epi8(eq) as u32;
        mask.trailing_ones() as usize
    }
}

fn flush_literal_slice(
    input: &[u8],
    mut start: usize,
    end: usize,
    output: &mut [u8],
    cursor: &mut usize,
) -> CompressorResult<()> {
    while start < end {
        let length = (end - start).min(usize::from(u16::MAX));
        write_bytes(output, cursor, &[0])?;
        write_bytes(output, cursor, &(length as u16).to_be_bytes())?;
        write_bytes(output, cursor, &input[start..start + length])?;
        start += length;
    }
    Ok(())
}

fn insert_position_flat(input: &[u8], position: usize, workspace: &mut LzWorkspace<'_>) {
    if position + MIN_MATCH > input.len() {
        return;
    }
    let key = hash4(&input[position..position + MIN_MATCH]);
    let slot = position & HISTORY_LIMIT;
    workspace.previous[slot] = workspace.heads[key];
    workspace.heads[key] = position as u32;
}

#[cfg(test)]
mod tests {
    use super::{
        HASH_TABLE_SIZE, HISTORY_LIMIT, LzWorkspace, NO_POSITION, StreamingEncoder, compress_block_into,
    };
    use crate::compression::lz_match_encode;
    use crate::compression_error::CompressorError;
    use crate::compression_lzmatch::lz_match_decode;

    #[test]
    fn workspace_size_validation() {
        let mut heads = [NO_POSITION; 8];
        let mut previous = [NO_POSITION; 8];
        assert!(LzWorkspace::new(&mut heads, &mut previous).is_err());
    }

    #[test]
    fn streaming_encoder_roundtrip() {
        let mut heads = vec![NO_POSITION; HASH_TABLE_SIZE];
        let mut previous = vec![NO_POSITION; HISTORY_LIMIT + 1];
        let mut encoder = StreamingEncoder::new(&mut heads, &mut previous).expect("workspace");
        let input = b"stream-stream-stream-data-stream";
        let mut output = vec![0u8; input.len() * 2 + 64];
        let written = encoder
            .encode_block(input, &mut output)
            .expect("encode");
        assert_eq!(lz_match_decode(&output[..written]).expect("decode"), input);
    }

    #[test]
    fn compress_block_rejects_tiny_output() {
        let mut heads = vec![NO_POSITION; HASH_TABLE_SIZE];
        let mut previous = vec![NO_POSITION; HISTORY_LIMIT + 1];
        let mut workspace = LzWorkspace::new(&mut heads, &mut previous).expect("ws");
        let mut output = [0u8; 4];
        assert_eq!(
            compress_block_into(b"hello", &mut output, &mut workspace),
            Err(CompressorError::OutputTooSmall)
        );
    }

    #[test]
    fn matches_one_shot_api() {
        let input = b"aaaaaaaaaaaaaaaa";
        let one_shot = lz_match_encode(input).expect("one_shot");
        let mut heads = vec![NO_POSITION; HASH_TABLE_SIZE];
        let mut previous = vec![NO_POSITION; HISTORY_LIMIT + 1];
        let mut workspace = LzWorkspace::new(&mut heads, &mut previous).expect("ws");
        let mut output = vec![0u8; one_shot.len() + 16];
        let written = compress_block_into(input, &mut output, &mut workspace).expect("block");
        assert_eq!(&output[..written], one_shot.as_slice());
    }
}
