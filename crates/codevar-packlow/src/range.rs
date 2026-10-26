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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Adaptive binary range coder.
//!
//! The Packlow stream is a sequence of binary decisions (is this token a
//! literal, is this byte `x`, which exponent, ...), so a binary range coder
//! is the natural entropy layer. The formulation is the one LZMA uses:
//! the encoder and the decoder perform the very same normalization steps,
//! so the byte streams line up exactly.
//!
//! Probabilities are 12-bit counts of a *zero* bit, in `1..=4095`.

use alloc::vec::Vec;

/// Width of the probability counters.
const PROB_BITS: u32 = 0x12;
/// One in the probability scale.
const PROB_ONE: u32 = 1 << PROB_BITS;
/// The neutral probability, used when a model starts out.
const PROB_INIT: u32 = PROB_ONE >> 1;
/// The range is renormalised once it drops below this value.
const TOP: u32 = 1 << 24;

/// The entropy coder's output.
#[derive(Debug, Default, Clone)]
pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u32,
    out: Vec<u8>,
}

impl RangeEncoder {
    /// Creates an empty encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    /// Shifts one byte out of the low register, propagating the carry.
    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            self.out.push(self.cache.wrapping_add(carry));
            for _ in 0..self.cache_size - 1 {
                self.out.push(0xFFu8.wrapping_add(carry));
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
            self.cache_size = 0;
        }
        self.cache_size += 1;
        self.low = (self.low << 8) & 0xFFFF_FFFF;
    }

    /// Codes one bit whose probability of being zero is `prob`.
    ///
    /// `prob` is clamped into the representable range so a caller can
    /// pass a raw count without checking it first.
    pub fn encode_bit(&mut self, prob: u32, bit: u32) {
        let prob = prob.clamp(1, PROB_ONE - 1);
        let bound = (self.range >> PROB_BITS) * prob;
        if bit == 0 {
            self.range = bound;
        } else {
            self.low += u64::from(bound);
            self.range -= bound;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Codes `count` raw bits of `value`, most significant first.
    pub fn encode_bits(&mut self, value: u32, count: u32) {
        for index in (0..count).rev() {
            self.encode_bit(PROB_INIT, (value >> index) & 1);
        }
    }

    /// Flushes the coder and returns the finished stream.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

/// The entropy coder's input.
#[derive(Debug, Clone)]
pub struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    /// Creates a decoder over `data`.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            pos: 0,
            range: 0xFFFF_FFFF,
            code: 0,
        };
        for _ in 0..5 {
            decoder.code = (decoder.code << 8) | decoder.next_byte();
        }
        decoder
    }

    /// Reads the next byte, returning zero past the end of the stream.
    ///
    /// The position advances even past the end so a caller can tell that
    /// the stream ran out and stop instead of decoding zeros forever.
    fn next_byte(&mut self) -> u32 {
        let byte = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        u32::from(byte)
    }

    /// Decodes one bit whose probability of being zero is `prob`.
    pub fn decode_bit(&mut self, prob: u32) -> u32 {
        let prob = prob.clamp(1, PROB_ONE - 1);
        let bound = (self.range >> PROB_BITS) * prob;
        let bit = if self.code < bound {
            self.range = bound;
            0
        } else {
            self.code -= bound;
            self.range -= bound;
            1
        };
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next_byte();
        }
        bit
    }

    /// Decodes `count` raw bits, most significant first.
    pub fn decode_bits(&mut self, count: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.decode_bit(PROB_INIT);
        }
        value
    }

    /// How many bytes of the input have been consumed.
    #[must_use]
    pub const fn consumed(&self) -> usize {
        self.pos
    }
}

/// One adaptive probability counter.
#[derive(Debug, Clone, Copy)]
pub struct Probability {
    prob: u32,
}

impl Probability {
    /// The neutral probability.
    pub const INIT: Self = Self { prob: PROB_INIT };

    /// The current probability of a zero bit.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.prob
    }

    /// Adapts the counter after a bit was coded.
    pub fn update(&mut self, bit: u32) {
        self.prob = if bit == 0 {
            self.prob + ((PROB_ONE - self.prob) >> 5)
        } else {
            self.prob - (self.prob >> 5)
        };
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self::INIT
    }
}

/// An order-zero model over a byte alphabet.
///
/// The model keeps one adaptive counter per *bit position*, which is
/// what an adaptive binary coder needs: the decoder knows the position
/// it is at even though it does not yet know the symbol.
#[derive(Debug, Clone)]
pub struct ByteModel {
    probs: [Probability; 8],
}

impl ByteModel {
    /// Creates a model over the byte alphabet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            probs: [Probability::INIT; 8],
        }
    }

    /// The probability of a zero bit at `position`.
    #[must_use]
    pub fn prob(&self, position: usize) -> u32 {
        self.probs
            .get(position)
            .map_or(PROB_INIT, |p| p.get())
    }

    /// Codes one symbol of the alphabet.
    pub fn encode(&mut self, encoder: &mut RangeEncoder, symbol: u8) {
        for position in (0..8).rev() {
            let bit = u32::from((symbol >> position) & 1);
            let prob = self.prob(position);
            encoder.encode_bit(prob, bit);
            if let Some(counter) = self.probs.get_mut(position) {
                counter.update(bit);
            }
        }
    }

    /// Decodes one symbol of the alphabet.
    pub fn decode(&mut self, decoder: &mut RangeDecoder<'_>) -> u8 {
        let mut value = 0u8;
        for position in 0..8 {
            let bit = decoder.decode_bit(self.prob(position));
            if let Some(counter) = self.probs.get_mut(position) {
                counter.update(bit);
            }
            value = (value << 1) | bit as u8;
        }
        value
    }
}

impl Default for ByteModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_stream_round_trips() {
        let bits: [u32; 64] = core::array::from_fn(|index| if index % 7 == 0 { 1 } else { 0 });
        let mut encoder = RangeEncoder::new();
        for &bit in &bits {
            encoder.encode_bit(PROB_INIT, bit);
        }
        let stream = encoder.finish();
        let mut decoder = RangeDecoder::new(&stream);
        for &expected in &bits {
            assert_eq!(decoder.decode_bit(PROB_INIT), expected);
        }
    }

    #[test]
    fn byte_model_round_trips() {
        let symbols: [u8; 256] = core::array::from_fn(|index| index as u8);
        let mut encoder = RangeEncoder::new();
        let mut model = ByteModel::new();
        for &symbol in &symbols {
            model.encode(&mut encoder, symbol);
        }
        let stream = encoder.finish();
        let mut decoder = RangeDecoder::new(&stream);
        let mut model = ByteModel::new();
        for &expected in &symbols {
            assert_eq!(model.decode(&mut decoder), expected);
        }
    }

    #[test]
    fn raw_bits_round_trip() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_bits(0xABCD, 16);
        let stream = encoder.finish();
        let mut decoder = RangeDecoder::new(&stream);
        assert_eq!(decoder.decode_bits(16), 0xABCD);
    }
}
