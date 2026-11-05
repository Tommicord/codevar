// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! The carryless 32-bit binary arithmetic coder.
//!
//! One bit at a time: the model hands over the probability that the bit is a
//! one, the coder narrows `[x1, x2]` around it, and a byte leaves whenever
//! the two bounds agree in their top byte. The split point is chosen so a
//! carry can never leave the interval - the low bound moves to `mid + 1`, one
//! above the point the high bound takes - which is why there is no pending
//! cache byte and no carry propagation over the output. It is also why the
//! renorm test is one comparison on the hottest path in the codec, instead of
//! the three cases a carry-correct coder needs.
//!
//! The decoder reads four bytes ahead of itself and zero-pads past the end of
//! the payload, so a truncated stream decodes to a determinate - if wrong -
//! tail instead of failing mid-symbol. Telling the two apart is the
//! container's job, not the coder's.

use alloc::vec::Vec;

/// Bytes the encoder writes out when the stream ends.
const FLUSH_BYTES: usize = 4;

/// Mask of the byte the renorm test compares.
const TOP_BYTE: u32 = 0xFF00_0000;

/// The point `p/4096` of the way across `[x1, x2]`.
///
/// Shared by both directions so the two cannot drift apart: whatever the
/// encoder computes for a bit, the decoder computes identically before it
/// knows the bit.
#[inline]
const fn mid(x1: u32, x2: u32, p: u32) -> u32 {
    x1.wrapping_add((x2.wrapping_sub(x1) >> 12).wrapping_mul(p))
}

/// Byte `i` of the payload, zero past its end.
#[inline]
fn payload_byte(payload: &[u8], i: usize) -> u32 {
    payload.get(i).copied().unwrap_or(0) as u32
}

/// Turns the model's probabilities into bytes.
///
/// The output grows with the stream, so the caller owns no buffer: a document
/// is coded in one pass and the body is taken with [`Encoder::into_bytes`].
#[derive(Debug, Default)]
pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    /// A coder over the whole 32-bit interval, with no output yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            x1: 0,
            x2: u32::MAX,
            out: Vec::new(),
        }
    }

    /// Narrow the interval around `bit` and emit the bytes it settled.
    ///
    /// `p` is the 12-bit probability that the bit is a one, as the model
    /// returns it. The coder only moves the bounds, so a `p` of 0 or 4096
    /// would not narrow at all and would spin forever on a bit that costs
    /// infinite information: the model clamps to `1..=4095` for that reason.
    #[inline]
    pub fn encode_bit(&mut self, p: u32, bit: bool) {
        let split = mid(self.x1, self.x2, p);
        if bit {
            self.x2 = split;
        } else {
            self.x1 = split.wrapping_add(1);
        }
        while (self.x1 ^ self.x2) & TOP_BYTE == 0 {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }

    /// Write the bytes a decoder still needs to finish reading the interval.
    pub fn finish(&mut self) {
        for _ in 0..FLUSH_BYTES {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
    }

    /// Bytes written so far.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.out
    }

    /// How many bytes have been written.
    #[must_use]
    pub fn len(&self) -> usize {
        self.out.len()
    }

    /// Whether nothing has been written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    /// Take the coded stream.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.out
    }
}

/// Reads back what [`Encoder`] wrote, one bit at a time.
#[derive(Debug)]
pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    payload: &'a [u8],
    pos: usize,
}

impl Decoder<'_> {
    /// A coder over `payload`, positioned after the four bytes it looks ahead.
    #[must_use]
    pub fn new(payload: &[u8]) -> Decoder<'_> {
        let window = (0..FLUSH_BYTES).fold(0u32, |acc, i| (acc << 8) | payload_byte(payload, i));
        Decoder {
            x1: 0,
            x2: u32::MAX,
            x: window,
            payload,
            pos: FLUSH_BYTES,
        }
    }

    /// Return the bit the payload holds, narrowing the interval the encoder
    /// narrowed for the same `p`.
    #[inline]
    pub fn decode_bit(&mut self, p: u32) -> bool {
        let split = mid(self.x1, self.x2, p);
        let bit = self.x <= split;
        if bit {
            self.x2 = split;
        } else {
            self.x1 = split.wrapping_add(1);
        }
        while (self.x1 ^ self.x2) & TOP_BYTE == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | payload_byte(self.payload, self.pos);
            self.pos += 1;
        }
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::{Decoder, Encoder, mid};

    /// The bit stream and probability schedule the tests share with the
    /// Python reference, so a golden vector means the same thing on both
    /// sides: bit `i` comes from a xorshift and `p` sweeps the 12-bit domain.
    fn corpus(bits: usize) -> (Vec<bool>, Vec<u32>) {
        let mut state = 0x2545_F491_u32;
        let mut stream = Vec::with_capacity(bits);
        let mut probs = Vec::with_capacity(bits);
        for i in 0..bits {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            stream.push(state & 2 != 0);
            probs.push(1 + ((i as u32 * 977) % 4095));
        }
        (stream, probs)
    }

    #[test]
    fn round_trip_recovers_every_bit() {
        let (stream, probs) = corpus(600);
        let mut encoder = Encoder::new();
        for (i, &bit) in stream.iter().enumerate() {
            encoder.encode_bit(probs[i], bit);
        }
        encoder.finish();
        let coded = encoder.into_bytes();
        let mut decoder = Decoder::new(&coded);
        for (i, &bit) in stream.iter().enumerate() {
            assert_eq!(decoder.decode_bit(probs[i]), bit, "bit {i}");
        }
    }

    #[test]
    fn the_split_point_scales_the_interval_by_the_probability() {
        assert_eq!(mid(0, 0xFFFF_FFFF, 1), 0x000F_FFFF);
        assert_eq!(mid(0, 0xFFFF_FFFF, 2048), 0x7FFF_F800);
        assert_eq!(mid(0, 0xFFFF_FFFF, 4095), 0xFFEF_F001);
        assert!(
            mid(0, 0xFFFF_FFFF, 4095) < 0xFFFF_FFFF,
            "the bounds never collapse"
        );
    }

    #[test]
    fn a_confident_symbol_costs_almost_nothing() {
        // Forty zeros the model is nearly sure of, then the same forty at even
        // odds: the gap is what the coder charges for information, not for
        // bits.
        let mut cheap = Encoder::new();
        let mut dear = Encoder::new();
        for _ in 0..40 {
            cheap.encode_bit(50, false);
            dear.encode_bit(2048, false);
        }
        cheap.finish();
        dear.finish();
        assert_eq!((cheap.len(), dear.len()), (4, 8));
    }

    #[test]
    fn flushing_writes_the_remaining_interval_bytes() {
        let mut encoder = Encoder::new();
        assert_eq!(encoder.len(), 0);
        assert!(encoder.is_empty());
        encoder.encode_bit(2048, true);
        encoder.finish();
        assert_eq!(encoder.bytes().len(), 4, "no top byte has settled yet");
        let coded = encoder.into_bytes();
        let mut decoder = Decoder::new(&coded);
        assert!(decoder.decode_bit(2048));
    }

    #[test]
    fn a_short_payload_pads_instead_of_failing() {
        // Truncation is a container error, but the coder must still be a total
        // function: reading past the end yields zeros, and so a determinate
        // bit stream.
        let mut decoder = Decoder::new(&[0x00, 0x11]);
        for _ in 0..64 {
            let _ = decoder.decode_bit(2048);
        }
    }

    #[test]
    fn golden_stream_matches_the_python_reference() {
        let (stream, probs) = corpus(24);
        let mut encoder = Encoder::new();
        for (i, &bit) in stream.iter().enumerate() {
            encoder.encode_bit(probs[i], bit);
        }
        encoder.finish();
        assert_eq!(
            encoder.bytes(),
            [0x00, 0x01, 0xEF, 0xD2, 0xE5, 0x01, 0x8E, 0x9E, 0x00],
            "the coder diverged from the Python reference"
        );
    }
}
