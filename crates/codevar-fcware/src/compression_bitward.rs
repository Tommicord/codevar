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
use crate::compression_frame::FrameKind;
use alloc::vec;
use alloc::vec::Vec;

const HEADER_SIZE: usize = 11;
const DUPLICATE_MARKER: u8 = 0x7f;
const INDEX_LIMIT: usize = 256;

/// Encodes `values` as a Bitward frame.
pub fn bitward_encode(values: &[u16]) -> CompressorResult<Vec<u8>> {
    let count = u32::try_from(values.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut frequencies = vec![0u32; 65_536];
    for &value in values {
        // SAFETY: `value` is a `u16`, so the index is always in `0..65536`.
        let slot = unsafe { frequencies.get_unchecked_mut(usize::from(value)) };
        *slot = slot.saturating_add(1);
    }

    let mut repeat_table = Vec::new();
    for (value, frequency) in frequencies
        .iter()
        .enumerate()
    {
        if *frequency > 1 && repeat_table.len() < INDEX_LIMIT {
            repeat_table.push(value as u16);
        }
    }
    let repeat_index: std::collections::HashMap<u16, u8> = repeat_table
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index as u8))
        .collect();

    let mut duplicate_bytes = Vec::new();
    for (value, frequency) in frequencies
        .iter()
        .enumerate()
    {
        if *frequency > 1 {
            let value = value as u16;
            let [high, low] = value.to_be_bytes();
            if !duplicate_bytes.contains(&high) {
                duplicate_bytes.push(high);
            }
            if !duplicate_bytes.contains(&low) {
                duplicate_bytes.push(low);
            }
        }
    }
    duplicate_bytes.sort_unstable();
    let duplicate_index: std::collections::HashMap<u8, u8> = duplicate_bytes
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index as u8))
        .collect();

    let mut records = Vec::with_capacity(
        values
            .len()
            .saturating_mul(2),
    );
    let mut index = 0usize;
    while index < values.len() {
        // SAFETY: `index < values.len()`.
        let value = unsafe {
            *values
                .as_ptr()
                .add(index)
        };
        let mut run_length = 1usize;
        while index + run_length < values.len()
            && unsafe {
                *values
                    .as_ptr()
                    .add(index + run_length)
            } == value
        {
            run_length += 1;
        }
        if let Some(&table_index) = repeat_index.get(&value)
            && table_index < 8
            && run_length >= 2
        {
            let mut remaining = run_length;
            while remaining > 0 {
                let chunk = remaining.min(16);
                records.push(0x80 | (table_index << 4) | (chunk as u8 - 1));
                remaining -= chunk;
            }
            index += run_length;
            continue;
        }
        let [high, low] = value.to_be_bytes();
        if high == low
            && let Some(&duplicate) = duplicate_index.get(&high)
        {
            records.extend([DUPLICATE_MARKER, duplicate]);
            index += 1;
            continue;
        }
        let mask = if high == low {
            3
        } else {
            u8::from(high == 0) | (u8::from(low == 0) << 1)
        };
        let payload = if mask == 3 {
            vec![high]
        } else {
            [high, low]
                .into_iter()
                .filter(|byte| *byte != 0)
                .collect::<Vec<_>>()
        };
        let control = (mask << 5) | (byte_exponent(high) << 3) | byte_exponent(low).saturating_mul(2);
        records.push(control);
        records.extend(payload);
        index += 1;
    }
    let repeat_len = u16::try_from(repeat_table.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let duplicate_len = u16::try_from(duplicate_bytes.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut frame =
        Vec::with_capacity(HEADER_SIZE + repeat_table.len() * 2 + duplicate_bytes.len() + records.len());
    frame.extend_from_slice(FrameKind::Bitward.magic());
    frame.extend_from_slice(&count.to_be_bytes());
    frame.extend_from_slice(&repeat_len.to_be_bytes());
    frame.extend_from_slice(&duplicate_len.to_be_bytes());
    for value in repeat_table {
        frame.extend_from_slice(&value.to_be_bytes());
    }
    frame.extend_from_slice(&duplicate_bytes);
    frame.extend_from_slice(&records);
    Ok(frame)
}

#[inline]
fn byte_exponent(value: u8) -> u8 {
    ((8 - value.leading_zeros()) / 2).min(3) as u8
}

/// Decodes a Bitward frame into `u16` values.
pub fn bitward_decode(frame: &[u8]) -> CompressorResult<Vec<u16>> {
    if frame.len() < HEADER_SIZE || FrameKind::from_magic(frame) != Some(FrameKind::Bitward) {
        return Err(CompressorError::InvalidFrame);
    }
    let expected = unsafe {
        let p = frame
            .as_ptr()
            .add(3);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    let repeat_len = u16::from_be_bytes([frame[7], frame[8]]) as usize;
    let duplicate_len = u16::from_be_bytes([frame[9], frame[10]]) as usize;
    let repeat_start = HEADER_SIZE;
    let duplicate_start = repeat_start
        .checked_add(
            repeat_len
                .checked_mul(2)
                .ok_or(CompressorError::TruncatedFrame)?,
        )
        .ok_or(CompressorError::TruncatedFrame)?;
    let records_start = duplicate_start
        .checked_add(duplicate_len)
        .ok_or(CompressorError::TruncatedFrame)?;
    if records_start > frame.len() {
        return Err(CompressorError::TruncatedFrame);
    }
    let repeat_table = frame[repeat_start..duplicate_start]
        .chunks_exact(2)
        .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    let duplicate_bytes = &frame[duplicate_start..records_start];
    let mut output = Vec::with_capacity(expected);
    let mut position = records_start;
    while output.len() < expected {
        let control = *frame
            .get(position)
            .ok_or(CompressorError::TruncatedFrame)?;
        position += 1;
        if control == DUPLICATE_MARKER {
            let duplicate = usize::from(
                *frame
                    .get(position)
                    .ok_or(CompressorError::TruncatedFrame)?,
            );
            position += 1;
            let byte = *duplicate_bytes
                .get(duplicate)
                .ok_or(CompressorError::InvalidDuplicateIndex)?;
            output.push(u16::from_be_bytes([byte, byte]));
            continue;
        }
        if control & 0x80 != 0 {
            let table_index = usize::from((control >> 4) & 7);
            let run_length = usize::from(control & 0x0f) + 1;
            let value = *repeat_table
                .get(table_index)
                .ok_or(CompressorError::InvalidIndex)?;
            for _ in 0..run_length {
                if output.len() >= expected {
                    return Err(CompressorError::length_mismatch(expected, output.len()));
                }
                output.push(value);
            }
            continue;
        }
        let mask = (control >> 5) & 3;
        if mask == 3 {
            let byte = *frame
                .get(position)
                .ok_or(CompressorError::TruncatedFrame)?;
            position += 1;
            output.push(u16::from_be_bytes([byte, byte]));
            continue;
        }
        let payload_size = 2usize - usize::from(mask & 1 != 0) - usize::from(mask & 2 != 0);
        let end = position
            .checked_add(payload_size)
            .ok_or(CompressorError::TruncatedFrame)?;
        let payload = frame
            .get(position..end)
            .ok_or(CompressorError::TruncatedFrame)?;
        position = end;
        let high = if mask & 1 != 0 { 0 } else { payload[0] };
        let low = if mask & 2 != 0 {
            0
        } else {
            payload[payload_size - 1]
        };
        output.push(u16::from_be_bytes([high, low]));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{bitward_decode, bitward_encode};
    use crate::compression_error::CompressorError;

    #[test]
    fn roundtrip_patterns() {
        let samples: &[&[u16]] = &[
            &[],
            &[0],
            &[0x0101, 0x0101, 0x0101],
            &[0, 0, 1, 1, 1, 2, 0xff00, 0x00ff, 0x7f7f],
            &[42; 40],
        ];
        for sample in samples {
            let frame = bitward_encode(sample).expect("encode");
            assert_eq!(bitward_decode(&frame).expect("decode"), *sample);
        }
    }

    #[test]
    fn rejects_invalid_frames() {
        assert_eq!(bitward_decode(b"XX"), Err(CompressorError::InvalidFrame));
        assert_eq!(
            bitward_decode(b"BW\x01\x00\x00\x00\x01\x00\x00\x00\x00"),
            Err(CompressorError::TruncatedFrame)
        );
    }
}
