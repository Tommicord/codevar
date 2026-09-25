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

//! Bulk bit extraction into SIMD lanes for entropy codecs.
//!
//! Instead of reading one bit at a time from the byte stream, [`BitLaneReader`]
//! refills an eight-lane buffer with a single unaligned load + SIMD (or scalar)
//! expand so consumers pull from packed lanes.

use crate::compression_error::{CompressorError, CompressorResult};

/// Number of parallel bit lanes filled per refill.
pub const BIT_LANES: usize = 8;

/// MSB-first bit reader that stores extracted bits in fixed SIMD lanes.
#[derive(Debug)]
pub struct BitLaneReader<'a> {
    data: &'a [u8],
    bit_index: usize,
    bit_end: usize,
    lanes: [u8; BIT_LANES],
    lane_len: u8,
    lane_pos: u8,
}

impl<'a> BitLaneReader<'a> {
    /// Creates a reader over `data` with `available_bits` valid MSB-first bits.
    #[inline]
    #[must_use]
    pub fn new(data: &'a [u8], available_bits: usize) -> Self {
        let max_bits = data
            .len()
            .saturating_mul(8);
        Self {
            data,
            bit_index: 0,
            bit_end: available_bits.min(max_bits),
            lanes: [0; BIT_LANES],
            lane_len: 0,
            lane_pos: 0,
        }
    }

    /// Absolute bit cursor within the stream.
    #[inline]
    #[must_use]
    pub const fn bit_index(&self) -> usize {
        self.bit_index
    }

    /// Remaining unconsumed bits in the stream.
    #[inline]
    #[must_use]
    pub fn remaining_bits(&self) -> usize {
        self.bit_end
            .saturating_sub(self.bit_index)
    }

    /// Returns the next bit (`0` or `1`), refilling lanes when empty.
    #[inline]
    pub fn next_bit(&mut self) -> CompressorResult<u8> {
        if self.lane_pos >= self.lane_len {
            self.refill_lanes()?;
        }
        let bit = self.lanes[usize::from(self.lane_pos)];
        self.lane_pos = self
            .lane_pos
            .saturating_add(1);
        self.bit_index = self
            .bit_index
            .saturating_add(1);
        Ok(bit)
    }

    /// Returns a view of the currently buffered lanes.
    #[inline]
    #[must_use]
    pub fn lanes(&self) -> &[u8] {
        &self.lanes[..usize::from(self.lane_len)]
    }

    /// Fills [`BIT_LANES`] (or fewer at EOF) bit lanes from the stream.
    ///
    /// # Safety invariants (internal)
    ///
    /// Unaligned loads only touch bytes still inside `self.data`, and lane
    /// stores write into the local `[u8; 8]` buffer.
    #[inline]
    pub fn refill_lanes(&mut self) -> CompressorResult<()> {
        if self.bit_index >= self.bit_end {
            return Err(CompressorError::TruncatedFrame);
        }
        let remaining = self.bit_end - self.bit_index;
        let count = remaining.min(BIT_LANES);
        self.lanes = extract_bit_lanes(self.data, self.bit_index, count);
        self.lane_len = count as u8;
        self.lane_pos = 0;
        Ok(())
    }
}

/// Extracts up to eight consecutive MSB-first bits into byte lanes.
#[inline]
pub(crate) fn extract_bit_lanes(data: &[u8], bit_index: usize, count: usize) -> [u8; BIT_LANES] {
    let count = count.min(BIT_LANES);
    if count == 0 {
        return [0; BIT_LANES];
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 feature is detected; data bounds are checked inside.
            return unsafe { extract_bit_lanes_sse2(data, bit_index, count) };
        }
    }
    extract_bit_lanes_scalar(data, bit_index, count)
}

#[inline]
fn extract_bit_lanes_scalar(data: &[u8], bit_index: usize, count: usize) -> [u8; BIT_LANES] {
    let mut lanes = [0u8; BIT_LANES];
    let window = load_bit_window(data, bit_index);
    for (lane, slot) in lanes
        .iter_mut()
        .enumerate()
        .take(count)
    {
        *slot = ((window >> (63 - lane)) & 1) as u8;
    }
    lanes
}

/// Loads enough bytes so the next bit at `bit_index` sits at bit 63 of the window.
#[inline]
fn load_bit_window(data: &[u8], bit_index: usize) -> u64 {
    let byte_index = bit_index / 8;
    let bit_off = bit_index % 8;
    if byte_index >= data.len() {
        return 0;
    }
    let mut tmp = [0u8; 8];
    let available = data.len() - byte_index;
    let copy_len = available.min(8);
    // SAFETY: `byte_index + copy_len <= data.len()` and `copy_len <= 8`.
    unsafe {
        core::ptr::copy_nonoverlapping(
            data.as_ptr()
                .add(byte_index),
            tmp.as_mut_ptr(),
            copy_len,
        );
    }
    let window = u64::from_be_bytes(tmp);
    window << bit_off
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn extract_bit_lanes_sse2(data: &[u8], bit_index: usize, count: usize) -> [u8; BIT_LANES] {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_set1_epi8, _mm_setr_epi8, _mm_storeu_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_set1_epi8, _mm_setr_epi8, _mm_storeu_si128,
    };

    let window = load_bit_window(data, bit_index);
    let top = (window >> 56) as i8;
    // Broadcast the aligned top byte, AND with MSB..LSB masks, compare to masks
    // so each lane becomes 0xFF (bit set) or 0x00, then mask to 0/1.
    // SAFETY: SSE2 is enabled via `target_feature`; store target is a local 16-byte buffer.
    unsafe {
        let broadcast = _mm_set1_epi8(top);
        let masks = _mm_setr_epi8(-128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
        let masked = _mm_and_si128(broadcast, masks);
        let eq = _mm_cmpeq_epi8(masked, masks);
        let ones = _mm_set1_epi8(1);
        let bits: __m128i = _mm_and_si128(eq, ones);
        let mut lanes = [0u8; 16];
        _mm_storeu_si128(
            lanes
                .as_mut_ptr()
                .cast::<__m128i>(),
            bits,
        );
        let mut out = [0u8; BIT_LANES];
        out[..count].copy_from_slice(&lanes[..count]);
        for slot in out
            .iter_mut()
            .skip(count)
        {
            *slot = 0;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{BIT_LANES, BitLaneReader, extract_bit_lanes};

    #[test]
    fn extracts_msb_first_bits_into_lanes() {
        // 0b1011_0001 => bits 1,0,1,1,0,0,0,1
        let data = [0b1011_0001];
        let lanes = extract_bit_lanes(&data, 0, 8);
        assert_eq!(lanes, [1, 0, 1, 1, 0, 0, 0, 1]);
        assert_eq!(BIT_LANES, 8);
    }

    #[test]
    fn extracts_across_byte_boundary() {
        let data = [0b0000_0001, 0b1000_0000];
        // bit 7 of first byte = 1, then MSB of second = 1
        let lanes = extract_bit_lanes(&data, 7, 2);
        assert_eq!(&lanes[..2], &[1, 1]);
    }

    #[test]
    fn reader_matches_sequential_bits() {
        let data = [0b1100_1010, 0b1111_0000];
        let mut reader = BitLaneReader::new(&data, 12);
        assert_eq!(reader.remaining_bits(), 12);
        let expected = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1];
        for bit in expected {
            assert_eq!(
                reader
                    .next_bit()
                    .expect("bit"),
                bit
            );
        }
        assert_eq!(reader.remaining_bits(), 0);
        assert!(
            reader
                .next_bit()
                .is_err()
        );
    }

    #[test]
    fn clamps_available_bits_to_data_size() {
        let data = [0xff];
        let mut reader = BitLaneReader::new(&data, 10_000);
        for _ in 0..8 {
            assert_eq!(
                reader
                    .next_bit()
                    .expect("bit"),
                1
            );
        }
        assert!(
            reader
                .next_bit()
                .is_err()
        );
    }

    #[test]
    fn empty_extract_is_zeroed() {
        assert_eq!(extract_bit_lanes(&[], 0, 8), [0; BIT_LANES]);
        assert_eq!(extract_bit_lanes(&[0xff], 0, 0), [0; BIT_LANES]);
    }
}
