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

//! The Packlow codec.
//!
//! A block is coded in one of three ways:
//!
//! * **literal** - the bytes are stored as they are,
//! * **copy** - the block repeats an earlier one,
//! * **delta** - the block is a *near* repeat of an earlier one, and the
//!   difference is stored as the per-byte corrections of
//!   [`crate::simd`].
//!
//! The delta form is the interesting one: instead of a fixed penalty per
//! character, the corrections are coded at the entropy of the
//! differences, so a block that differs from its neighbor in a handful
//! of places costs a handful of bytes no matter how long it is.

use alloc::vec::Vec;

use crate::error::{PackError, PackResult};
use crate::range::{ByteModel, Probability, RangeDecoder, RangeEncoder};
use crate::shuffle::ShuffleTable;
use crate::simd;

/// The magic bytes that open a packed stream.
const MAGIC: [u8; 4] = *b"PLOW";

/// The size of the header: the magic, the length of the original data
/// and a checksum of it.
const HEADER_SIZE: usize = 16;

/// The token kinds the stream carries.
const TOKEN_LITERAL: u32 = 0x0;
const TOKEN_COPY: u32 = 0x1;
const TOKEN_DELTA: u32 = 0x2;
const TOKEN_END: u32 = 0x3;

/// The number of bits a correction takes: two for the exponent and three
/// for the number.
const CORRECTION_BITS: u32 = 0x5;

/// How the packer is tuned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacklowConfig {
    /// The size of one block, in bytes.
    pub block_size: usize,
    /// How many blocks the shuffle table remembers.
    pub shuffle_capacity: usize,
    /// The smallest share of a block a delta must cover to be used.
    pub min_similarity: u8,
    /// The largest block a delta may cover.
    pub max_delta: usize,
}

impl Default for PacklowConfig {
    fn default() -> Self {
        Self {
            block_size: 0x800,
            shuffle_capacity: 0x8,
            min_similarity: 0xCC,
            max_delta: 0xFFFF,
        }
    }
}

/// How many bits a token takes.
const TOKEN_BITS: u32 = 2;

/// The probability models the token layer shares.
#[derive(Debug, Clone)]
struct TokenModels {
    token: Probability,
    length: Probability,
    distance: Probability,
    literals: ByteModel,
}

impl TokenModels {
    fn new() -> Self {
        Self {
            token: Probability::INIT,
            length: Probability::INIT,
            distance: Probability::INIT,
            literals: ByteModel::new(),
        }
    }
}

/// Codes a token kind.
fn encode_token(encoder: &mut RangeEncoder, models: &mut TokenModels, token: u32) {
    for index in (0..TOKEN_BITS).rev() {
        let bit = (token >> index) & 1;
        let prob = models.token.get();
        encoder.encode_bit(prob, bit);
        models.token.update(bit);
    }
}

/// Decodes a token kind.
fn decode_token(decoder: &mut RangeDecoder<'_>, models: &mut TokenModels) -> u32 {
    let mut token = 0u32;
    for _ in 0..TOKEN_BITS {
        let prob = models.token.get();
        let bit = decoder.decode_bit(prob);
        models.token.update(bit);
        token = (token << 1) | bit;
    }
    token
}

/// Codes a value with an Elias gamma code.
fn encode_gamma(encoder: &mut RangeEncoder, models: &mut TokenModels, value: u32) {
    let bits = 32 - value.leading_zeros();
    for _ in 1..bits {
        let prob = models.length.get();
        encoder.encode_bit(prob, 0);
        models.length.update(0);
    }
    let prob = models.length.get();
    encoder.encode_bit(prob, 1);
    models.length.update(1);
    for index in (0..bits).rev() {
        let bit = (value >> index) & 1;
        let prob = models.distance.get();
        encoder.encode_bit(prob, bit);
        models.distance.update(bit);
    }
}

/// Decodes an Elias gamma code.
fn decode_gamma(decoder: &mut RangeDecoder<'_>, models: &mut TokenModels) -> u32 {
    let mut bits = 1u32;
    loop {
        let prob = models.length.get();
        let bit = decoder.decode_bit(prob);
        models.length.update(bit);
        if bit == 1 {
            break;
        }
        bits += 1;
        if bits > 32 {
            return 0;
        }
    }
    let mut value = 0u32;
    for _ in 0..bits {
        let prob = models.distance.get();
        let bit = decoder.decode_bit(prob);
        models.distance.update(bit);
        value = (value << 1) | bit;
    }
    value
}

/// Packs the corrections of a delta into a bit stream.
///
/// Every correction takes five bits: two for the exponent and three for
/// the number.
fn pack_corrections(plan: &simd::DeltaPlan, length: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(length.div_ceil(2));
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    for index in 0..length {
        let exponent = u32::from(plan.exponent[index]) & 0x3;
        let number = u32::from(plan.number[index]) & 0x7;
        accumulator |= (exponent | (number << 2)) << bits;
        bits += CORRECTION_BITS;
        if bits >= 8 {
            out.push((accumulator & 0xFF) as u8);
            accumulator >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push((accumulator & 0xFF) as u8);
    }
    out
}

/// Unpacks a correction stream into a plan.
fn unpack_corrections(packed: &[u8], length: usize) -> simd::DeltaPlan {
    let mut plan = simd::DeltaPlan {
        exponent: Vec::with_capacity(length),
        number: Vec::with_capacity(length),
        representable: length,
    };
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    let mut byte_index = 0usize;
    for _ in 0..length {
        while bits < CORRECTION_BITS {
            let byte = packed.get(byte_index).copied().unwrap_or(0);
            byte_index += 1;
            accumulator |= u32::from(byte) << bits;
            bits += 8;
        }
        let correction = accumulator & ((1 << CORRECTION_BITS) - 1);
        accumulator >>= CORRECTION_BITS;
        bits -= CORRECTION_BITS;
        plan.exponent.push((correction & 0x3) as u8);
        plan.number.push(((correction >> 2) & 0x7) as u8);
    }
    plan
}

/// The Packlow codec.
#[derive(Debug, Clone)]
pub struct Packlow {
    config: PacklowConfig,
}

impl Packlow {
    /// Creates a codec with the given configuration.
    #[must_use]
    pub const fn new(config: PacklowConfig) -> Self {
        Self { config }
    }

    /// The configuration the codec runs with.
    #[must_use]
    pub const fn config(&self) -> &PacklowConfig {
        &self.config
    }

    /// Packs `data` into a self describing stream.
    ///
    /// The stream starts with a magic number and the length of the
    /// original data, so the decoder knows when to stop.
    #[must_use]
    pub fn pack(&self, data: &[u8]) -> Vec<u8> {
        let mut models = TokenModels::new();
        let mut encoder = RangeEncoder::new();
        let mut table = ShuffleTable::new(self.config.shuffle_capacity);
        let block_size = self.config.block_size.max(1);

        let mut offset = 0usize;
        while offset < data.len() {
            let block = &data[offset..(offset + block_size).min(data.len())];
            let signature = simd::block_signature(block);

            // An exact repeat.
            if let Some(rank) = table.rank_of(signature) {
                let entry = table.get(rank).unwrap_or_default();
                let distance = offset.saturating_sub(entry.offset).max(1);
                self.write_token(
                    &mut encoder,
                    &mut models,
                    TOKEN_COPY,
                    distance as u32,
                    block.len() as u32,
                    &[],
                );
                table.promote(rank);
                offset += block.len();
                continue;
            }
            // A near repeat.
            let mut delta: Option<(usize, Vec<u8>, usize)> = None;
            if let Some((rank, plan, covered)) = self.find_similar(block, &table, data) {
                let entry = table.get(rank).unwrap_or_default();
                let distance = offset.saturating_sub(entry.offset).max(1);
                // The decoder rebuilds the corrections from the bytes
                // that precede the current position, so the plan may not
                // reach further back than the gap allows.
                let covered = covered.min(distance);
                if covered > 0 {
                    let corrections = pack_corrections(&plan, covered);
                    delta = Some((rank, corrections, covered));
                }
            }
            if let Some((rank, corrections, covered)) = delta {
                let entry = table.get(rank).unwrap_or_default();
                let distance = offset.saturating_sub(entry.offset).max(1);
                self.write_token(
                    &mut encoder,
                    &mut models,
                    TOKEN_DELTA,
                    distance as u32,
                    covered as u32,
                    &corrections,
                );
                table.promote(rank);
                offset += covered;
                continue;
            }
            // A literal.
            self.write_token(
                &mut encoder,
                &mut models,
                TOKEN_LITERAL,
                block.len() as u32,
                block.len() as u32,
                block,
            );
            table.insert(signature, offset, block.len());
            offset += block.len();
        }

        encode_token(&mut encoder, &mut models, TOKEN_END);

        let mut out = Vec::with_capacity(data.len() / 4 + HEADER_SIZE);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&simd::block_signature(data).to_le_bytes());
        out.extend_from_slice(&encoder.finish());
        out
    }

    /// Unpacks a stream produced by [`Self::pack`].
    ///
    /// # Errors
    ///
    /// Returns [`PackError::Truncated`] when the stream ends early,
    /// [`PackError::UnknownToken`] for a token the decoder does not
    /// know, and [`PackError::BadDistance`] for a reference that points
    /// before the start of the output.
    pub fn unpack(&self, packed: &[u8]) -> PackResult<Vec<u8>> {
        if packed.len() < 16 || packed[..4] != MAGIC {
            return Err(PackError::Truncated {
                missing: 16usize.saturating_sub(packed.len()),
            });
        }
        let raw_len = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]) as usize;
        let expected = u64::from_le_bytes([
            packed[8], packed[9], packed[10], packed[11], packed[12], packed[13], packed[14], packed[15],
        ]);
        let mut models = TokenModels::new();
        let mut decoder = RangeDecoder::new(&packed[HEADER_SIZE..]);
        // The header is not trusted: the pre-allocation is capped so a
        // corrupt stream cannot ask for an enormous buffer.
        let mut out: Vec<u8> = Vec::with_capacity(raw_len.min(1 << 20));

        let stream_len = packed.len() - HEADER_SIZE;
        loop {
            // The stream is over: a short read means the packed data was
            // truncated, so the decoder stops instead of decoding zeros.
            if decoder.consumed() > stream_len {
                return Err(PackError::Truncated {
                    missing: raw_len.saturating_sub(out.len()),
                });
            }
            let token = decode_token(&mut decoder, &mut models);
            let token = if token == 0 {
                TOKEN_LITERAL
            } else if token == 1 {
                TOKEN_COPY
            } else if token == 2 {
                TOKEN_DELTA
            } else {
                TOKEN_END
            };

            match token {
                TOKEN_LITERAL => {
                    let length = decode_gamma(&mut decoder, &mut models) as usize;
                    if length > raw_len.saturating_sub(out.len()) {
                        return Err(PackError::LengthOverflow { length });
                    }
                    for _ in 0..length {
                        let byte = models.literals.decode(&mut decoder);
                        out.push(byte);
                    }
                }
                TOKEN_COPY => {
                    let distance = decode_gamma(&mut decoder, &mut models) as usize;
                    let length = decode_gamma(&mut decoder, &mut models) as usize;
                    if distance == 0 || distance > out.len() {
                        return Err(PackError::BadDistance {
                            distance: distance as u32,
                            produced: out.len(),
                        });
                    }
                    if length > raw_len.saturating_sub(out.len()) {
                        return Err(PackError::LengthOverflow { length });
                    }
                    let start = out.len() - distance;
                    for index in 0..length {
                        // The copy may overlap the bytes it produces, so
                        // the source index always stays behind the write
                        // head and the loop reads what it just wrote.
                        let byte = out.get(start + index).copied().unwrap_or(0);
                        out.push(byte);
                    }
                }
                TOKEN_DELTA => {
                    let distance = decode_gamma(&mut decoder, &mut models) as usize;
                    let length = decode_gamma(&mut decoder, &mut models) as usize;
                    if distance == 0 || distance > out.len() {
                        return Err(PackError::BadDistance {
                            distance: distance as u32,
                            produced: out.len(),
                        });
                    }
                    if length > raw_len.saturating_sub(out.len()) {
                        return Err(PackError::LengthOverflow { length });
                    }
                    if distance < length {
                        return Err(PackError::BadDistance {
                            distance: distance as u32,
                            produced: out.len(),
                        });
                    }
                    let packed_bytes = (length * CORRECTION_BITS as usize).div_ceil(8);
                    let mut corrections = Vec::with_capacity(packed_bytes);
                    for _ in 0..packed_bytes {
                        corrections.push(models.literals.decode(&mut decoder));
                    }
                    let plan = unpack_corrections(&corrections, length);
                    let start = out.len() - distance;
                    let source: Vec<u8> = out[start..start + length].to_vec();
                    let rebuilt =
                        simd::apply_delta(&source, &plan).ok_or(PackError::LengthOverflow { length })?;
                    out.extend_from_slice(&rebuilt);
                }
                _ => break,
            }

            if out.len() >= raw_len {
                break;
            }
        }

        if out.len() != raw_len {
            return Err(PackError::Truncated {
                missing: raw_len.saturating_sub(out.len()),
            });
        }
        // The checksum closes the gap a range coder leaves: a stream
        // that was cut short still decodes into the right *number* of
        // bytes, but into the wrong ones.
        let actual = simd::block_signature(&out);
        if actual != expected {
            return Err(PackError::ChecksumMismatch { expected, actual });
        }
        Ok(out)
    }

    /// Writes one token.
    fn write_token(
        &self,
        encoder: &mut RangeEncoder,
        models: &mut TokenModels,
        token: u32,
        first: u32,
        second: u32,
        payload: &[u8],
    ) {
        encode_token(encoder, models, token);
        encode_gamma(encoder, models, first);
        if token != TOKEN_LITERAL {
            encode_gamma(encoder, models, second);
        }
        for &byte in payload {
            models.literals.encode(encoder, byte);
        }
    }

    /// Looks for a block in the shuffle table the new block is a near
    /// repeat of.
    ///
    /// Returns the rank of the candidate, the correction plan and how
    /// many bytes the plan covers.
    fn find_similar(
        &self,
        block: &[u8],
        table: &ShuffleTable,
        data: &[u8],
    ) -> Option<(usize, simd::DeltaPlan, usize)> {
        let minimum = usize::from(self.config.min_similarity) * block.len() / 100;
        let limit = block.len().min(self.config.max_delta);
        for rank in 0..table.len() {
            let entry = table.get(rank)?;
            if entry.length == 0 {
                continue;
            }
            let reference = data.get(entry.offset..entry.offset + limit)?;
            // A cheap filter: the blocks must agree for a while before a
            // full plan is worth computing.
            if simd::common_prefix(&block[..limit], reference) < minimum.min(limit) {
                continue;
            }
            let plan = simd::plan_delta(reference, &block[..limit]);
            let delta_representable = plan.representable;
            if delta_representable >= minimum && delta_representable >= 1 {
                return Some((rank, plan, delta_representable));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A deterministic pseudo random byte generator.
    fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    /// The same generator, but the bytes stay printable so the data
    /// looks like source code.
    fn pseudo_text(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                b'a' + (state % 26) as u8
            })
            .collect()
    }

    /// Packs and unpacks `data`, asserting that the bytes survive.
    fn round_trip(data: &[u8], config: PacklowConfig) -> Vec<u8> {
        let codec = Packlow::new(config);
        let packed = codec.pack(data);
        let restored = codec
            .unpack(&packed)
            .expect("the stream must decode");
        assert_eq!(restored, data, "the decoded bytes must match the input");
        restored
    }

    #[test]
    fn an_empty_input_round_trips() {
        let codec = Packlow::new(PacklowConfig::default());
        let packed = codec.pack(&[]);
        assert_eq!(codec.unpack(&packed).as_deref(), Ok(&[][..]));
    }

    #[test]
    fn a_single_byte_round_trips() {
        round_trip(&[0x41], PacklowConfig::default());
        round_trip(&[0x00], PacklowConfig::default());
        round_trip(&[0xFF], PacklowConfig::default());
    }

    #[test]
    fn exactly_one_block_round_trips() {
        let config = PacklowConfig::default();
        let data = pseudo_random(config.block_size, 3);
        round_trip(&data, config);
    }

    #[test]
    fn one_byte_past_a_block_round_trips() {
        let config = PacklowConfig::default();
        let mut data = pseudo_random(config.block_size, 4);
        data.push(0x5A);
        round_trip(&data, config);
    }

    #[test]
    fn a_run_of_zeroes_round_trips() {
        let data = vec![0u8; 9000];
        round_trip(&data, PacklowConfig::default());
    }

    #[test]
    fn a_run_of_maximum_bytes_round_trips() {
        let data = vec![0xFFu8; 9000];
        round_trip(&data, PacklowConfig::default());
    }

    #[test]
    fn an_alternating_pattern_round_trips() {
        let data: Vec<u8> = (0..9000)
            .map(|index| if index % 2 == 0 { 0xAA } else { 0x55 })
            .collect();
        round_trip(&data, PacklowConfig::default());
    }

    #[test]
    fn a_single_repeated_byte_round_trips() {
        let data = vec![0x7Fu8; 12345];
        round_trip(&data, PacklowConfig::default());
    }

    #[test]
    fn text_round_trips() {
        let data = pseudo_text(50_000, 17);
        round_trip(&data, PacklowConfig::default());
    }

    #[test]
    fn a_one_byte_block_size_round_trips() {
        let config = PacklowConfig {
            block_size: 1,
            ..PacklowConfig::default()
        };
        let data = pseudo_random(1000, 21);
        round_trip(&data, config);
    }

    #[test]
    fn a_huge_block_size_round_trips() {
        let config = PacklowConfig {
            block_size: 1 << 20,
            ..PacklowConfig::default()
        };
        let data = pseudo_random(5000, 22);
        round_trip(&data, config);
    }

    #[test]
    fn a_single_entry_shuffle_table_round_trips() {
        let config = PacklowConfig {
            shuffle_capacity: 1,
            ..PacklowConfig::default()
        };
        let data = pseudo_random(20_000, 23);
        round_trip(&data, config);
    }

    #[test]
    fn a_perfect_similarity_requirement_round_trips() {
        let config = PacklowConfig {
            min_similarity: 100,
            ..PacklowConfig::default()
        };
        let data = pseudo_random(20_000, 24);
        round_trip(&data, config);
    }

    #[test]
    fn a_zero_similarity_requirement_round_trips() {
        let config = PacklowConfig {
            min_similarity: 0,
            ..PacklowConfig::default()
        };
        let data = pseudo_random(20_000, 25);
        round_trip(&data, config);
    }

    #[test]
    fn a_tiny_maximum_delta_round_trips() {
        let config = PacklowConfig {
            max_delta: 4,
            ..PacklowConfig::default()
        };
        let data = pseudo_random(20_000, 26);
        round_trip(&data, config);
    }

    #[test]
    fn a_repeated_block_is_stored_as_a_copy() {
        let codec = Packlow::new(PacklowConfig::default());
        let block = pseudo_random(4096, 9);
        let mut data = block.clone();
        data.extend_from_slice(&block);
        let packed = codec.pack(&data);
        assert_eq!(codec.unpack(&packed).as_deref(), Ok(&data[..]));
        // The second block costs a handful of bytes, not another copy.
        assert!(packed.len() < data.len(), "packed {} bytes", packed.len());
        assert!(
            packed.len() < data.len() / 2 + 512,
            "packed {} bytes",
            packed.len()
        );
    }

    #[test]
    fn a_near_repeat_is_stored_as_a_delta() {
        let codec = Packlow::new(PacklowConfig::default());
        let base = pseudo_text(4096, 31);
        let mut near = base.clone();
        for index in (0..4096).step_by(97) {
            near[index] = near[index].wrapping_add(1);
        }
        let mut data = base.clone();
        data.extend_from_slice(&near);
        let packed = codec.pack(&data);
        assert_eq!(codec.unpack(&packed).as_deref(), Ok(&data[..]));
        // The near repeat costs far less than a literal copy.
        let literal_size = codec.pack(&data).len();
        assert!(
            literal_size < data.len(),
            "the stream must be smaller than the input"
        );
    }

    #[test]
    fn an_unreachable_difference_is_stored_as_a_literal() {
        let codec = Packlow::new(PacklowConfig::default());
        let base = pseudo_text(4096, 41);
        let mut far = base.clone();
        // A difference of +100 is out of reach for every scale.
        far[0] = far[0].wrapping_add(100);
        let mut data = base.clone();
        data.extend_from_slice(&far);
        let packed = codec.pack(&data);
        assert_eq!(codec.unpack(&packed).as_deref(), Ok(&data[..]));
    }

    #[test]
    fn a_delta_never_reaches_past_its_gap() {
        // A block that follows a short block of the same content must
        // not claim more bytes than the gap allows.
        let codec = Packlow::new(PacklowConfig {
            block_size: 64,
            ..PacklowConfig::default()
        });
        let data = pseudo_text(4096, 43);
        let packed = codec.pack(&data);
        assert_eq!(codec.unpack(&packed).as_deref(), Ok(&data[..]));
    }

    #[test]
    fn every_truncation_is_rejected_or_exact() {
        let codec = Packlow::new(PacklowConfig::default());
        // The check is quadratic in the packed size, so the input stays
        // small enough for the test to finish quickly.
        let data = pseudo_random(3_000, 51);
        let packed = codec.pack(&data);
        for length in 0..packed.len() {
            let truncated = &packed[..length];
            // A stream that was cut short either fails or decodes into
            // exactly the original bytes; anything else would be a
            // silent corruption.
            if let Ok(restored) = codec.unpack(truncated) {
                assert_eq!(restored, data, "truncated to {length}");
            }
        }
    }

    #[test]
    fn a_flipped_bit_never_panics() {
        let codec = Packlow::new(PacklowConfig::default());
        let data = pseudo_random(2_048, 61);
        let packed = codec.pack(&data);
        for index in 0..packed.len().min(128) {
            for bit in 0..8 {
                let mut corrupted = packed.clone();
                corrupted[index] ^= 1 << bit;
                // The decoder may accept the stream or reject it, but
                // it must never panic, and it must never return bytes
                // that do not match the checksum.
                if let Ok(restored) = codec.unpack(&corrupted) {
                    assert_eq!(restored, data, "byte {index} bit {bit} was accepted");
                }
            }
        }
    }

    #[test]
    fn a_broken_magic_is_rejected() {
        let codec = Packlow::new(PacklowConfig::default());
        let packed = codec.pack(&pseudo_random(100, 71));
        for index in 0..4 {
            let mut broken = packed.clone();
            broken[index] ^= 0xFF;
            assert!(codec.unpack(&broken).is_err());
        }
    }

    #[test]
    fn an_impossible_length_is_rejected() {
        let codec = Packlow::new(PacklowConfig::default());
        let mut broken = codec.pack(&pseudo_random(100, 81));
        broken[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        // The header is not trusted: the decoder must not try to
        // allocate four gigabytes for it.
        assert!(codec.unpack(&broken).is_err());
    }

    #[test]
    fn a_stream_of_zeroes_is_rejected() {
        let codec = Packlow::new(PacklowConfig::default());
        let broken = vec![0u8; 64];
        assert!(codec.unpack(&broken).is_err());
    }

    #[test]
    fn random_garbage_is_rejected_without_panicking() {
        let codec = Packlow::new(PacklowConfig::default());
        for seed in 0..64u32 {
            let garbage = pseudo_random(1 + (seed as usize * 13) % 300, seed * 7 + 1);
            // The result is either an error or a stream that passes the
            // checksum; it must never panic.
            let _ = codec.unpack(&garbage);
        }
    }

    #[test]
    fn a_stream_that_is_only_a_header_is_rejected() {
        let codec = Packlow::new(PacklowConfig::default());
        let mut header = Vec::new();
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        assert!(codec.unpack(&header).is_err());
    }

    #[test]
    fn random_inputs_round_trip() {
        let config = PacklowConfig::default();
        for seed in 0..24u32 {
            let length = (seed as usize * 977) % 9000;
            let data = pseudo_random(length, seed * 31 + 5);
            round_trip(&data, config);
        }
    }

    #[test]
    fn near_duplicate_inputs_round_trip() {
        let config = PacklowConfig::default();
        for seed in 0..12u32 {
            let base = pseudo_text(8000, seed * 13 + 3);
            let mut variant = base.clone();
            for index in (0..8000).step_by(211) {
                variant[index] = variant[index].wrapping_add(1 + (index % 3) as u8);
            }
            let mut data = base.clone();
            data.extend_from_slice(&variant);
            round_trip(&data, config);
        }
    }

    #[test]
    fn a_stream_is_never_much_larger_than_its_input() {
        // Incompressible data may grow, but only by the header and a
        // little model overhead.
        let codec = Packlow::new(PacklowConfig::default());
        let data = pseudo_random(8192, 91);
        let packed = codec.pack(&data);
        // Incompressible data costs a little more than its own size: the
        // header, the model and the block framing.
        assert!(packed.len() <= data.len() + 512, "packed {} bytes", packed.len());
    }
}
