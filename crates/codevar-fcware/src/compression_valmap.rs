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
use crate::compression_frame::{FrameKind, frame_read_u16};
use alloc::vec;
use alloc::vec::Vec;

/// Bytes per compact-range table record (`delta_le16`, `count`).
#[allow(dead_code)] // public predictive-table surface for dictionary tuning
pub const COMPACT_RANGE_RECORD_SIZE: usize = 3;

/// Packed predictive candidate range table (little-endian deltas).
#[allow(dead_code)]
pub const COMPACT_RANGE_TABLE: &[u8] =
    b"\x00\x00\x10UU\r\x01\x00\x0c\xa9*\n\x01\x00\x08\xfd\x7f\x07\x01\x00\x06\x01\x00\x05";

/// A contiguous predictive candidate window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub struct CandidateRange {
    /// Inclusive start of the previous-value window.
    pub start: u16,
    /// Number of `(first, next)` multiplier pairs available.
    pub count: u8,
}

/// Iterates compact candidate ranges for dictionary prediction.
#[allow(dead_code)]
pub fn compact_candidate_ranges() -> impl Iterator<Item = CandidateRange> {
    COMPACT_RANGE_TABLE
        .chunks_exact(COMPACT_RANGE_RECORD_SIZE)
        .scan(0u16, |start, record| {
            let delta = u16::from_le_bytes([record[0], record[1]]);
            *start = start.saturating_add(delta);
            Some(CandidateRange {
                start: *start,
                count: record[2],
            })
        })
}

/// Returns how many candidates are available after `previous`.
#[must_use]
#[allow(dead_code)]
pub fn compact_candidate_count(previous: u16) -> u8 {
    compact_candidate_ranges()
        .take_while(|range| range.start <= previous)
        .last()
        .map_or(0, |range| range.count)
}

/// Yields `(candidate, first_multiplier, next_multiplier)` for `previous`.
#[allow(dead_code)]
pub fn compact_candidates(previous: u16) -> impl Iterator<Item = (u16, u8, u8)> {
    let count = usize::from(compact_candidate_count(previous));
    (0u8..4)
        .flat_map(|first| (0u8..4).map(move |next| (first, next)))
        .take(count)
        .filter_map(move |(first, next)| {
            let value = u32::from(previous)
                .saturating_mul(u32::from(first))
                .saturating_add(u32::from(next));
            u16::try_from(value)
                .ok()
                .map(|candidate| (candidate, first, next))
        })
}

/// Encodes values without a frame header.
pub fn valmap_encode(values: &[u16]) -> CompressorResult<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(values.len().saturating_mul(3));
    // SAFETY: `values` is non-empty.
    let first = unsafe { *values.as_ptr() };
    if first < 0x80 {
        output.push(0x80 | first as u8);
    } else {
        output.push(0);
        output.extend_from_slice(&first.to_be_bytes());
    }
    for index in 1..values.len() {
        // SAFETY: `index` and `index - 1` are in-bounds.
        let (previous, current) = unsafe { (*values.as_ptr().add(index - 1), *values.as_ptr().add(index)) };
        let mut best: Option<(u16, u8, u8, u32, u16)> = None;
        for first_multiplier in 0u8..4 {
            for next_multiplier in 0u8..4 {
                let candidate = u32::from(previous)
                    .saturating_mul(u32::from(first_multiplier))
                    .saturating_add(u32::from(next_multiplier));
                if let Ok(candidate) = u16::try_from(candidate) {
                    let distance = u32::from(candidate.abs_diff(current));
                    let order = (u16::from(first_multiplier) << 2) | u16::from(next_multiplier);
                    let replace = best.is_none_or(|(_, _, _, best_distance, best_order)| {
                        (distance, order) < (best_distance, best_order)
                    });
                    if replace {
                        best = Some((candidate, first_multiplier, next_multiplier, distance, order));
                    }
                }
            }
        }
        let (candidate, first_multiplier, next_multiplier, _, _) =
            best.ok_or(CompressorError::InvalidIndex)?;
        let similarity = (15u32.saturating_sub((previous ^ current).count_ones())).min(7) as u8;
        if candidate == current {
            output.push((similarity << 1) | (first_multiplier << 4) | (next_multiplier << 6));
        } else if current < 0x80 {
            output.push((similarity << 1) | 0x11);
            output.push(current as u8);
        } else {
            output.push((similarity << 1) | 1);
            output.extend_from_slice(&current.to_be_bytes());
        }
    }
    Ok(output)
}

/// Encodes values into a Dictionary frame (`DI\x01`).
pub fn valmap_encode_frame(values: &[u16]) -> CompressorResult<Vec<u8>> {
    let payload = valmap_encode(values)?;
    let mut frame = Vec::with_capacity(3 + payload.len());
    frame.extend_from_slice(FrameKind::Dictionary.magic());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes a header-less dictionary stream.
pub fn valmap_decode(stream: &[u8]) -> CompressorResult<Vec<u16>> {
    if stream.is_empty() {
        return Ok(Vec::new());
    }
    let (first, mut position) = if stream[0] & 0x80 != 0 {
        (u16::from(stream[0] & 0x7f), 1)
    } else {
        if stream.len() < 3 {
            return Err(CompressorError::TruncatedFrame);
        }
        (u16::from_be_bytes([stream[1], stream[2]]), 3)
    };
    let mut output = vec![first];
    while position < stream.len() {
        let header = stream[position];
        position += 1;
        let mode = header & 1;
        let first_multiplier = (header >> 4) & 3;
        let next_multiplier = (header >> 6) & 3;
        let current = if mode == 0 {
            let value = u32::from(
                *output
                    .last()
                    .ok_or(CompressorError::InvalidIndex)?,
            )
            .saturating_mul(u32::from(first_multiplier))
            .saturating_add(u32::from(next_multiplier));
            u16::try_from(value).map_err(|_| CompressorError::InvalidIndex)?
        } else if first_multiplier == 1 && next_multiplier == 0 {
            let value = *stream
                .get(position)
                .ok_or(CompressorError::TruncatedFrame)?;
            position += 1;
            u16::from(value)
        } else {
            frame_read_u16(stream, &mut position)?
        };
        output.push(current);
    }
    Ok(output)
}

/// Decodes a Dictionary frame (`DI\x01`).
pub fn valmap_decode_frame(frame: &[u8]) -> CompressorResult<Vec<u16>> {
    if frame.len() < 3 || FrameKind::from_magic(frame) != Some(FrameKind::Dictionary) {
        return Err(CompressorError::InvalidFrame);
    }
    valmap_decode(&frame[3..])
}

#[cfg(test)]
mod tests {
    use super::{
        COMPACT_RANGE_RECORD_SIZE, CandidateRange, compact_candidate_count, compact_candidate_ranges,
        compact_candidates, valmap_decode, valmap_decode_frame, valmap_encode, valmap_encode_frame,
    };
    use crate::compression_error::CompressorError;
    use alloc::vec::Vec;

    #[test]
    fn table_and_candidates_are_well_formed() {
        assert_eq!(COMPACT_RANGE_RECORD_SIZE, 3);
        const { assert!(COMPACT_RANGE_RECORD_SIZE > 0) };
        let ranges: Vec<CandidateRange> = compact_candidate_ranges().collect();
        assert!(!ranges.is_empty());
        assert!(
            ranges
                .windows(2)
                .all(|w| w[0].start <= w[1].start)
        );
        let _ = compact_candidate_count(0);
        let _ = compact_candidates(1).count();
    }

    #[test]
    fn encode_decode_roundtrip() {
        let values = [0u16, 1, 2, 3, 0x80, 0x0100, 0xffff, 7, 7, 8];
        let stream = valmap_encode(&values).expect("encode");
        assert_eq!(valmap_decode(&stream).expect("decode"), values);
        let frame = valmap_encode_frame(&values).expect("frame");
        assert_eq!(valmap_decode_frame(&frame).expect("decode_frame"), values);
        assert_eq!(valmap_encode(&[]).expect("empty"), Vec::<u8>::new());
        assert_eq!(valmap_decode(&[]).expect("empty"), Vec::<u16>::new());
    }

    #[test]
    fn rejects_bad_frames() {
        assert_eq!(valmap_decode_frame(b"XX"), Err(CompressorError::InvalidFrame));
        assert_eq!(valmap_decode(&[0]), Err(CompressorError::TruncatedFrame));
    }
}
