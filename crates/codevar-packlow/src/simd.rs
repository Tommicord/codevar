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

//! Vectorized kernels for the Packlow codec.
//!
//! The codec reaches a neighbouring region by describing how it differs
//! from an earlier one. Every byte of that difference is expressed as a
//! small signed *number* together with an *exponent*, the number being
//! scaled by `2^exponent`:
//!
//! ```text
//! target = source + (number - 3) * 2^exponent
//! ```
//!
//! with `number` in `0..=7` (the signed offset `-3..=4`) and `exponent` in
//! `0..=3` (the scale `1`, `2`, `4` or `8`). A byte is representable
//! when its difference is one of
//!
//! ```text
//! scale 1: -3 -2 -1 0 1 2 3 4
//! scale 2: -6 -4 -2 0 2 4 6 8
//! scale 4: -12 -8 -4 0 4 8 12 16
//! scale 8: -24 -16 -8 0 8 16 24 32
//! ```
//!
//! The kernels below find the smallest scale that reaches each byte and
//! report the first byte that no scale can reach.
//!
//! Three techniques carry the implementation:
//!
//! * **Vectorized loads and stores** - the sixteen (or thirty-two) bytes
//!   of a run are loaded once and processed in registers.
//! * **Magic bitmasks** - the "no scale reaches this byte" test is
//!   reduced to a single `movemask`, whose trailing ones give the index
//!   of the first offending byte without any per-byte branching.
//! * **Shuffle tables** - the decode side turns a number into its signed
//!   offset with one `pshufb` over a sixteen entry table, instead of a
//!   table lookup per byte.
//!
//! Every vector path has a scalar twin in [`scalar`]; the two are
//! checked against each other by the tests, and the scalar path is the
//! oracle the vector paths must reproduce bit for bit.

use alloc::vec::Vec;

use codevar_base::cpuid::{Feature, has};

/// The number of scales a byte difference can use.
pub const EXPONENT_COUNT: u32 = 4;

/// The prime the block signature folds its words with.
const SIGNATURE_PRIME: u64 = 0x0000_0100_0000_01B3;

/// The seed the block signature starts from.
const SIGNATURE_SEED: u64 = 0xCBF2_9CE4_8422_2325;

/// Folds the eight byte words of `block` into a signature.
///
/// The signature is defined over *words*, not bytes, so the scalar and
/// the vector paths produce the very same value and a block packed on
/// one machine is recognised on another.
fn fold_words(mut hash: u64, mut index: u64, block: &[u8]) -> (u64, u64) {
    let mut offset = 0usize;
    while offset + 8 <= block.len() {
        let mut word = [0u8; 8];
        word.copy_from_slice(&block[offset..offset + 8]);
        hash = hash.rotate_left(7)
            ^ u64::from_le_bytes(word)
                .wrapping_mul(SIGNATURE_PRIME)
                .wrapping_add(index);
        index += 1;
        offset += 8;
    }
    (hash, index)
}

/// Folds the trailing bytes (fewer than eight) into a signature.
fn fold_tail(hash: u64, index: u64, tail: &[u8]) -> u64 {
    let mut word = 0u64;
    for (position, &byte) in tail.iter().enumerate() {
        word |= u64::from(byte) << (position * 8);
    }
    hash.rotate_left(7)
        ^ word
            .wrapping_mul(SIGNATURE_PRIME)
            .wrapping_add(index)
}

/// The signed offsets a number can stand for, biased by three.
pub const NUMBER_BIAS: i32 = 3;

/// The largest magnitude a scale can reach.
const MAX_MAGNITUDE: [u8; EXPONENT_COUNT as usize] = [4, 8, 16, 32];

/// The correction plan for a run of bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeltaPlan {
    /// The scale exponent of every byte, `0..=3`.
    pub exponent: Vec<u8>,
    /// The biased signed offset of every byte, `0..=7`.
    pub number: Vec<u8>,
    /// How many leading bytes are representable.
    pub representable: usize,
}

/// Computes the correction plan that turns `source` into `target`.
///
/// Only the leading [`DeltaPlan::representable`] bytes are reachable;
/// the rest of the plan is left zeroed.
///
/// # Examples
///
/// ```
/// use codevar_packlow::simd::plan_delta;
///
/// let plan = plan_delta(b"abcd", b"abed");
/// assert_eq!(plan.representable, 4);
/// assert_eq!(plan.number[2], 5); // 'c' -> 'e' is +2, biased by three
/// ```
#[must_use]
pub fn plan_delta(source: &[u8], target: &[u8]) -> DeltaPlan {
    #[cfg(target_arch = "x86_64")]
    {
        if has(Feature::Avx2) {
            // SAFETY: the AVX2 feature was proved by the CPUID probe.
            return unsafe { x86::avx2::plan_delta(source, target) };
        }
        if has(Feature::Ssse3) {
            // SAFETY: the SSSE3 feature was proved by the CPUID probe.
            return unsafe { x86::ssse3::plan_delta(source, target) };
        }
    }
    scalar::plan_delta(source, target)
}

/// Rebuilds `target` from `source` and a correction plan.
///
/// Returns [`None`] when the plan is shorter than the target or holds an
/// out-of-range exponent or number.
///
/// # Examples
///
/// ```
/// use codevar_packlow::simd::{apply_delta, plan_delta};
///
/// let source = b"abcd";
/// let target = b"abed";
/// let plan = plan_delta(source, target);
/// assert_eq!(apply_delta(source, &plan).as_deref(), Some(&target[..]));
/// ```
#[must_use]
pub fn apply_delta(source: &[u8], plan: &DeltaPlan) -> Option<Vec<u8>> {
    if plan.exponent.len() != plan.number.len() || source.len() < plan.exponent.len() {
        return None;
    }
    // Every path validates the plan the same way, so a malformed stream
    // is rejected whichever kernel would have run.
    for (exponent, number) in plan.exponent.iter().zip(plan.number.iter()) {
        if u32::from(*exponent) >= EXPONENT_COUNT || *number > 7 {
            return None;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has(Feature::Avx2) {
            // SAFETY: the AVX2 feature was proved by the CPUID probe.
            return unsafe { x86::avx2::apply_delta(source, plan) };
        }
        if has(Feature::Ssse3) {
            // SAFETY: the SSSE3 feature was proved by the CPUID probe.
            return unsafe { x86::ssse3::apply_delta(source, plan) };
        }
    }
    scalar::apply_delta(source, plan)
}

/// Returns the number of leading bytes where `left` and `right` agree.
#[must_use]
pub fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if has(Feature::Avx2) {
            // SAFETY: the AVX2 feature was proved by the CPUID probe.
            return unsafe { x86::avx2::common_prefix(left, right) };
        }
        if has(Feature::Ssse3) {
            // SAFETY: the SSSE3 feature was proved by the CPUID probe.
            return unsafe { x86::ssse3::common_prefix(left, right) };
        }
    }
    scalar::common_prefix(left, right)
}

/// A 64-bit signature of a block, used to look a block up in the
/// shuffle table.
#[must_use]
pub fn block_signature(block: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if has(Feature::Avx2) {
            // SAFETY: the AVX2 feature was proved by the CPUID probe.
            return unsafe { x86::avx2::block_signature(block) };
        }
        if has(Feature::Ssse3) {
            // SAFETY: the SSSE3 feature was proved by the CPUID probe.
            return unsafe { x86::ssse3::block_signature(block) };
        }
    }
    scalar::block_signature(block)
}

/// The scalar reference implementation of every kernel.
///
/// This module is the oracle the vector paths are tested against, and
/// the implementation used on targets without a vector unit.
pub mod scalar {
    use alloc::vec::Vec;

    use super::{DeltaPlan, EXPONENT_COUNT, MAX_MAGNITUDE, NUMBER_BIAS};

    /// Finds the smallest scale that reaches `delta`, if any.
    #[must_use]
    pub fn classify(delta: i16) -> Option<(u8, u8)> {
        let magnitude = delta.unsigned_abs();
        for exponent in 0..EXPONENT_COUNT {
            let scale = 1u32 << exponent;
            if u32::from(magnitude) % scale != 0 {
                continue;
            }
            if u32::from(magnitude) > u32::from(MAX_MAGNITUDE[exponent as usize]) {
                continue;
            }
            let quotient = i32::from(delta) / (scale as i32);
            let number = quotient + NUMBER_BIAS;
            if (0..=7).contains(&number) {
                return Some((exponent as u8, number as u8));
            }
        }
        None
    }

    /// Computes the correction plan that turns `source` into `target`.
    #[must_use]
    pub fn plan_delta(source: &[u8], target: &[u8]) -> DeltaPlan {
        let len = source.len().min(target.len());
        let mut plan = DeltaPlan {
            exponent: alloc::vec![0u8; len],
            number: alloc::vec![0u8; len],
            representable: 0,
        };
        for index in 0..len {
            let delta = i16::from(target[index]) - i16::from(source[index]);
            match classify(delta) {
                Some((exponent, number)) => {
                    plan.exponent[index] = exponent;
                    plan.number[index] = number;
                    plan.representable = index + 1;
                }
                None => break,
            }
        }
        plan
    }

    /// Rebuilds `target` from `source` and a correction plan.
    #[must_use]
    pub fn apply_delta(source: &[u8], plan: &DeltaPlan) -> Option<Vec<u8>> {
        let len = plan.exponent.len();
        if plan.number.len() != len || source.len() < len {
            return None;
        }
        let mut out = Vec::with_capacity(len);
        for (index, (&exponent, &number)) in plan
            .exponent
            .iter()
            .zip(plan.number.iter())
            .enumerate()
        {
            if u32::from(exponent) >= EXPONENT_COUNT || number > 7 {
                return None;
            }
            let scale = 1i32 << exponent;
            let offset = (i32::from(number) - NUMBER_BIAS) * scale;
            let value = i32::from(source[index]) + offset;
            out.push(value as u8);
        }
        Some(out)
    }

    /// Returns the number of leading bytes where `left` and `right` agree.
    #[must_use]
    pub fn common_prefix(left: &[u8], right: &[u8]) -> usize {
        let len = left.len().min(right.len());
        let mut index = 0;
        while index < len && left[index] == right[index] {
            index += 1;
        }
        index
    }

    /// A 64-bit signature of a block.
    ///
    /// The recurrence folds eight byte words, so the scalar and the
    /// vector paths compute the very same value and a block packed on
    /// one machine is recognised on another.
    #[must_use]
    pub fn block_signature(block: &[u8]) -> u64 {
        let (hash, index) = super::fold_words(super::SIGNATURE_SEED, 0, block);
        let tail = &block[block.len() & !7..];
        let hash = super::fold_tail(hash, index, tail);
        hash ^ (hash >> 29)
    }
}

/// The x86 vector implementations.
#[cfg(target_arch = "x86_64")]
mod x86 {
    /// The 128-bit implementations.
    pub mod ssse3 {
        use alloc::vec::Vec;
        use core::arch::x86_64::*;

        /// The shuffle table that turns a biased number into its signed
        /// offset. Only the first eight entries are ever used; the rest
        /// are zero so a malformed stream cannot smuggle a large offset
        /// into the reconstruction.
        pub(super) const NUMBER_SHUFFLE: [u8; 16] = {
            let mut table = [0u8; 16];
            let mut index = 0usize;
            while index < 8 {
                table[index] = (index as i8 - 3) as u8;
                index += 1;
            }
            table
        };

        /// Masks that clear the bits a logical byte shift would carry
        /// from the high byte of a lane into the low byte.
        pub(super) const LEAK_MASK: [[u8; 16]; 4] = {
            let mut masks = [[0u8; 16]; 4];
            let mut shift = 1usize;
            while shift < 4 {
                let mut index = 0usize;
                while index < 16 {
                    masks[shift][index] = if index.is_multiple_of(2) {
                        0xFFu8 >> shift
                    } else {
                        0xFF
                    };
                    index += 1;
                }
                shift += 1;
            }
            masks
        };

        /// Loads sixteen bytes without reading past the end of the slice.
        #[inline]
        fn load16(bytes: &[u8]) -> __m128i {
            // SAFETY: the caller guarantees at least sixteen readable bytes.
            unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
        }

        /// Computes the correction plan that turns `source` into `target`.
        ///
        /// # Safety
        ///
        /// Requires SSSE3, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "ssse3")]
        pub unsafe fn plan_delta(source: &[u8], target: &[u8]) -> super::super::DeltaPlan {
            let len = source.len().min(target.len());
            let mut plan = super::super::DeltaPlan {
                exponent: alloc::vec![0u8; len],
                number: alloc::vec![0u8; len],
                representable: 0,
            };
            let mut offset = 0usize;
            while offset < len {
                let block = (len - offset).min(16);
                let mut src = [0u8; 16];
                let mut dst = [0u8; 16];
                src[..block].copy_from_slice(&source[offset..offset + block]);
                dst[..block].copy_from_slice(&target[offset..offset + block]);
                // SAFETY: the arrays are sixteen bytes long and the
                // instruction is available under the target feature.
                let (exponents, numbers, invalid) = unsafe { classify16(src, dst) };
                // The mask has a one for every byte no scale reaches, so
                // the leading bytes that *are* reachable are the trailing
                // ones of its complement.
                let good = block.min((!invalid).trailing_ones() as usize);
                plan.exponent[offset..offset + good].copy_from_slice(&exponents[..good]);
                plan.number[offset..offset + good].copy_from_slice(&numbers[..good]);
                plan.representable = offset + good;
                if good < block {
                    return plan;
                }
                offset += block;
            }
            plan.representable = len;
            plan
        }

        /// Classifies sixteen byte pairs.
        ///
        /// # Safety
        ///
        /// Requires SSSE3.
        #[target_feature(enable = "ssse3")]
        unsafe fn classify16(src: [u8; 16], dst: [u8; 16]) -> ([u8; 16], [u8; 16], u32) {
            // SAFETY: the instruction is available under the target feature.
            unsafe {
                let src_v = _mm_loadu_si128(src.as_ptr().cast());
                let dst_v = _mm_loadu_si128(dst.as_ptr().cast());
                let delta = _mm_sub_epi8(dst_v, src_v);
                let signed = _mm_sub_epi8(delta, _mm_setzero_si128());
                // The magnitude of the signed difference.
                let magnitude = _mm_min_epu8(_mm_abs_epi8(signed), _mm_set1_epi8(127));

                // The scale ladder: the first scale whose magnitude limit
                // the difference fits under and whose scale divides it.
                let mut reachable = _mm_setzero_si128();
                let mut exponent = _mm_setzero_si128();
                // A negative difference may only reach -3, a positive one
                // may reach +4, so the magnitude limit depends on the
                // sign of the difference.
                let negative = _mm_cmpgt_epi8(_mm_setzero_si128(), signed);
                let mut limit = 4i8;
                let mut divisor = 1i8;
                let mut scale = 0i32;
                while scale < 4 {
                    let allowance =
                        _mm_sub_epi8(_mm_set1_epi8(limit), _mm_and_si128(negative, _mm_set1_epi8(1)));
                    let fits = _mm_cmplt_epi8(magnitude, _mm_add_epi8(allowance, _mm_set1_epi8(1)));
                    let divides = _mm_cmpeq_epi8(
                        _mm_and_si128(magnitude, _mm_set1_epi8(divisor - 1)),
                        _mm_setzero_si128(),
                    );
                    let ok = _mm_and_si128(fits, divides);
                    let fresh = _mm_andnot_si128(reachable, ok);
                    exponent = _mm_or_si128(exponent, _mm_and_si128(fresh, _mm_set1_epi8(scale as i8)));
                    reachable = _mm_or_si128(reachable, ok);
                    limit *= 2;
                    divisor *= 2;
                    scale += 1;
                }
                let invalid = _mm_cmpeq_epi8(reachable, _mm_setzero_si128());

                // The quotient is the magnitude shifted by the exponent,
                // with the sign of the difference applied afterward.
                let shifted1 = _mm_and_si128(_mm_srli_epi16(magnitude, 1), leak_mask(1));
                let shifted2 = _mm_and_si128(_mm_srli_epi16(magnitude, 2), leak_mask(2));
                let shifted3 = _mm_and_si128(_mm_srli_epi16(magnitude, 3), leak_mask(3));
                let is0 = _mm_cmpeq_epi8(exponent, _mm_setzero_si128());
                let is1 = _mm_cmpeq_epi8(exponent, _mm_set1_epi8(1));
                let is2 = _mm_cmpeq_epi8(exponent, _mm_set1_epi8(2));
                let quotient = _mm_or_si128(
                    _mm_or_si128(_mm_and_si128(magnitude, is0), _mm_and_si128(shifted1, is1)),
                    _mm_or_si128(
                        _mm_and_si128(shifted2, is2),
                        _mm_and_si128(shifted3, _mm_cmpeq_epi8(exponent, _mm_set1_epi8(3))),
                    ),
                );
                let signed_quotient = _mm_sign_epi8(quotient, signed);
                let number = _mm_add_epi8(signed_quotient, _mm_set1_epi8(3));

                let mut exponents = [0u8; 16];
                let mut numbers = [0u8; 16];
                _mm_storeu_si128(exponents.as_mut_ptr().cast(), exponent);
                _mm_storeu_si128(numbers.as_mut_ptr().cast(), number);
                let invalid_mask = _mm_movemask_epi8(invalid) as u32;
                (exponents, numbers, invalid_mask)
            }
        }

        /// The leak mask for a logical byte shift by `shift`.
        #[inline]
        fn leak_mask(shift: usize) -> __m128i {
            // SAFETY: the instruction is available under the target feature.
            unsafe { _mm_loadu_si128(LEAK_MASK[shift].as_ptr().cast()) }
        }

        /// Rebuilds the target from the source and a correction plan.
        ///
        /// # Safety
        ///
        /// Requires SSSE3, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "ssse3")]
        pub unsafe fn apply_delta(source: &[u8], plan: &super::super::DeltaPlan) -> Option<Vec<u8>> {
            let len = plan.exponent.len();
            if plan.number.len() != len || source.len() < len {
                return None;
            }
            let mut out = Vec::with_capacity(len);
            let mut offset = 0usize;
            while offset < len {
                let block = (len - offset).min(16);
                let mut src = [0u8; 16];
                let mut exp = [0u8; 16];
                let mut num = [0u8; 16];
                src[..block].copy_from_slice(&source[offset..offset + block]);
                exp[..block].copy_from_slice(&plan.exponent[offset..offset + block]);
                num[..block].copy_from_slice(&plan.number[offset..offset + block]);
                // SAFETY: the arrays are sixteen bytes long and the
                // instruction is available under the target feature.
                let bytes = unsafe { apply16(src, exp, num) };
                out.extend_from_slice(&bytes[..block]);
                offset += block;
            }
            Some(out)
        }

        /// Applies sixteen corrections.
        ///
        /// # Safety
        ///
        /// Requires SSSE3.
        #[target_feature(enable = "ssse3")]
        unsafe fn apply16(src: [u8; 16], exp: [u8; 16], num: [u8; 16]) -> [u8; 16] {
            // SAFETY: the instruction is available under the target feature.
            unsafe {
                let src_v = _mm_loadu_si128(src.as_ptr().cast());
                let exp_v = _mm_loadu_si128(exp.as_ptr().cast());
                let num_v = _mm_loadu_si128(num.as_ptr().cast());
                // The shuffle table turns the biased number into its
                // signed offset in one instruction.
                let offset = _mm_shuffle_epi8(_mm_loadu_si128(NUMBER_SHUFFLE.as_ptr().cast()), num_v);
                // The scale is a shift of the offset; the four candidates
                // are selected with the exponent.
                let scaled1 = _mm_slli_epi16(offset, 1);
                let scaled2 = _mm_slli_epi16(offset, 2);
                let scaled3 = _mm_slli_epi16(offset, 3);
                let is0 = _mm_cmpeq_epi8(exp_v, _mm_setzero_si128());
                let is1 = _mm_cmpeq_epi8(exp_v, _mm_set1_epi8(1));
                let is2 = _mm_cmpeq_epi8(exp_v, _mm_set1_epi8(2));
                let is3 = _mm_cmpeq_epi8(exp_v, _mm_set1_epi8(3));
                let scaled = _mm_or_si128(
                    _mm_or_si128(_mm_and_si128(offset, is0), _mm_and_si128(scaled1, is1)),
                    _mm_or_si128(_mm_and_si128(scaled2, is2), _mm_and_si128(scaled3, is3)),
                );
                let mut out = [0u8; 16];
                _mm_storeu_si128(out.as_mut_ptr().cast(), _mm_add_epi8(src_v, scaled));
                out
            }
        }

        /// Returns the number of leading bytes where `left` and `right` agree.
        ///
        /// # Safety
        ///
        /// Requires SSSE3, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "ssse3")]
        pub unsafe fn common_prefix(left: &[u8], right: &[u8]) -> usize {
            let len = left.len().min(right.len());
            let mut offset = 0usize;
            while offset < len {
                let block = (len - offset).min(16);
                let mut lhs = [0u8; 16];
                let mut rhs = [0u8; 16];
                lhs[..block].copy_from_slice(&left[offset..offset + block]);
                rhs[..block].copy_from_slice(&right[offset..offset + block]);
                // SAFETY: the arrays are sixteen bytes long and the
                // instruction is available under the target feature.
                let diff = _mm_cmpeq_epi8(load16(&lhs), load16(&rhs));
                // SAFETY: the instruction is available under the target feature.
                let mask = _mm_movemask_epi8(diff) as u32;
                let equal = mask.trailing_ones() as usize;
                if equal < block {
                    return offset + equal;
                }
                offset += block;
            }
            len
        }

        /// A 64-bit signature of a block.
        ///
        /// # Safety
        ///
        /// Requires SSSE3, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "ssse3")]
        pub unsafe fn block_signature(block: &[u8]) -> u64 {
            let mut hash = 0xCBF2_9CE4_8422_2325u64;
            let mut index = 0u64;
            let mut offset = 0usize;
            while offset + 16 <= block.len() {
                // SAFETY: the load stays inside the block and the
                // instruction is available under the target feature.
                let chunk = load16(&block[offset..]);
                // SAFETY: the instruction is available under the target feature.
                let low = _mm_cvtsi128_si64(chunk) as u64;
                // SAFETY: the instruction is available under the target feature.
                let high = _mm_cvtsi128_si64(_mm_srli_si128(chunk, 8)) as u64;
                hash = hash.rotate_left(7)
                    ^ low
                        .wrapping_mul(super::super::SIGNATURE_PRIME)
                        .wrapping_add(index);
                index += 1;
                hash = hash.rotate_left(7)
                    ^ high
                        .wrapping_mul(super::super::SIGNATURE_PRIME)
                        .wrapping_add(index);
                index += 1;
                offset += 16;
            }
            // The words that do not fill a vector are folded exactly as
            // the scalar path folds them, so both agree byte for byte.
            let (hash, index) = super::super::fold_words(hash, index, &block[offset..]);
            let tail = &block[block.len() & !7..];
            let hash = super::super::fold_tail(hash, index, tail);
            hash ^ (hash >> 29)
        }
    }

    /// The 256-bit implementations.
    pub mod avx2 {
        use alloc::vec::Vec;
        use core::arch::x86_64::*;

        use super::ssse3;

        /// Computes the correction plan that turns `source` into `target`.
        ///
        /// # Safety
        ///
        /// Requires AVX2, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "avx2")]
        pub unsafe fn plan_delta(source: &[u8], target: &[u8]) -> super::super::DeltaPlan {
            let len = source.len().min(target.len());
            let mut plan = super::super::DeltaPlan {
                exponent: alloc::vec![0u8; len],
                number: alloc::vec![0u8; len],
                representable: 0,
            };
            let mut offset = 0usize;
            while offset < len {
                let block = (len - offset).min(32);
                let mut src = [0u8; 32];
                let mut dst = [0u8; 32];
                src[..block].copy_from_slice(&source[offset..offset + block]);
                dst[..block].copy_from_slice(&target[offset..offset + block]);
                // SAFETY: the arrays are thirty-two bytes long and the
                // instruction is available under the target feature.
                let (exponents, numbers, invalid) = unsafe { classify32(src, dst) };
                let good = block.min((!invalid).trailing_ones() as usize);
                plan.exponent[offset..offset + good].copy_from_slice(&exponents[..good]);
                plan.number[offset..offset + good].copy_from_slice(&numbers[..good]);
                plan.representable = offset + good;
                if good < block {
                    return plan;
                }
                offset += block;
            }
            plan.representable = len;
            plan
        }

        /// Classifies thirty-two byte pairs.
        ///
        /// # Safety
        ///
        /// Requires AVX2.
        #[target_feature(enable = "avx2")]
        unsafe fn classify32(src: [u8; 32], dst: [u8; 32]) -> ([u8; 32], [u8; 32], u32) {
            // SAFETY: the instruction is available under the target feature.
            unsafe {
                let src_v = _mm256_loadu_si256(src.as_ptr().cast());
                let dst_v = _mm256_loadu_si256(dst.as_ptr().cast());
                let delta = _mm256_sub_epi8(dst_v, src_v);
                let signed = _mm256_sub_epi8(delta, _mm256_setzero_si256());
                let magnitude = _mm256_min_epu8(_mm256_abs_epi8(signed), _mm256_set1_epi8(127));

                let mut reachable = _mm256_setzero_si256();
                let mut exponent = _mm256_setzero_si256();
                // A negative difference may only reach -3, a positive one
                // may reach +4, so the magnitude limit depends on the
                // sign of the difference.
                let negative = _mm256_cmpgt_epi8(_mm256_setzero_si256(), signed);
                let mut limit = 4i8;
                let mut divisor = 1i8;
                let mut scale = 0i32;
                while scale < 4 {
                    let allowance = _mm256_sub_epi8(
                        _mm256_set1_epi8(limit),
                        _mm256_and_si256(negative, _mm256_set1_epi8(1)),
                    );
                    let fits = _mm256_cmpgt_epi8(_mm256_add_epi8(allowance, _mm256_set1_epi8(1)), magnitude);
                    let divides = _mm256_cmpeq_epi8(
                        _mm256_and_si256(magnitude, _mm256_set1_epi8(divisor - 1)),
                        _mm256_setzero_si256(),
                    );
                    let ok = _mm256_and_si256(fits, divides);
                    let fresh = _mm256_andnot_si256(reachable, ok);
                    exponent =
                        _mm256_or_si256(exponent, _mm256_and_si256(fresh, _mm256_set1_epi8(scale as i8)));
                    reachable = _mm256_or_si256(reachable, ok);
                    limit *= 2;
                    divisor *= 2;
                    scale += 1;
                }
                let invalid = _mm256_cmpeq_epi8(reachable, _mm256_setzero_si256());

                let shifted1 = _mm256_and_si256(_mm256_srli_epi16(magnitude, 1), leak_mask(1));
                let shifted2 = _mm256_and_si256(_mm256_srli_epi16(magnitude, 2), leak_mask(2));
                let shifted3 = _mm256_and_si256(_mm256_srli_epi16(magnitude, 3), leak_mask(3));
                let is0 = _mm256_cmpeq_epi8(exponent, _mm256_setzero_si256());
                let is1 = _mm256_cmpeq_epi8(exponent, _mm256_set1_epi8(1));
                let is2 = _mm256_cmpeq_epi8(exponent, _mm256_set1_epi8(2));
                let is3 = _mm256_cmpeq_epi8(exponent, _mm256_set1_epi8(3));
                let quotient = _mm256_or_si256(
                    _mm256_or_si256(_mm256_and_si256(magnitude, is0), _mm256_and_si256(shifted1, is1)),
                    _mm256_or_si256(_mm256_and_si256(shifted2, is2), _mm256_and_si256(shifted3, is3)),
                );
                let signed_quotient = _mm256_sign_epi8(quotient, signed);
                let number = _mm256_add_epi8(signed_quotient, _mm256_set1_epi8(3));

                let mut exponents = [0u8; 32];
                let mut numbers = [0u8; 32];
                _mm256_storeu_si256(exponents.as_mut_ptr().cast(), exponent);
                _mm256_storeu_si256(numbers.as_mut_ptr().cast(), number);
                let invalid_mask = _mm256_movemask_epi8(invalid) as u32;
                (exponents, numbers, invalid_mask)
            }
        }

        /// The leak mask for a logical byte shift by `shift`.
        #[inline]
        fn leak_mask(shift: usize) -> __m256i {
            // SAFETY: the instruction is available under the target feature.
            unsafe {
                let mask = ssse3::LEAK_MASK[shift];
                _mm256_setr_epi8(
                    mask[0] as i8,
                    mask[1] as i8,
                    mask[2] as i8,
                    mask[3] as i8,
                    mask[4] as i8,
                    mask[5] as i8,
                    mask[6] as i8,
                    mask[7] as i8,
                    mask[8] as i8,
                    mask[9] as i8,
                    mask[10] as i8,
                    mask[11] as i8,
                    mask[12] as i8,
                    mask[13] as i8,
                    mask[14] as i8,
                    mask[15] as i8,
                    mask[0] as i8,
                    mask[1] as i8,
                    mask[2] as i8,
                    mask[3] as i8,
                    mask[4] as i8,
                    mask[5] as i8,
                    mask[6] as i8,
                    mask[7] as i8,
                    mask[8] as i8,
                    mask[9] as i8,
                    mask[10] as i8,
                    mask[11] as i8,
                    mask[12] as i8,
                    mask[13] as i8,
                    mask[14] as i8,
                    mask[15] as i8,
                )
            }
        }

        /// Rebuilds the target from the source and a correction plan.
        ///
        /// # Safety
        ///
        /// Requires AVX2, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "avx2")]
        pub unsafe fn apply_delta(source: &[u8], plan: &super::super::DeltaPlan) -> Option<Vec<u8>> {
            let len = plan.exponent.len();
            if plan.number.len() != len || source.len() < len {
                return None;
            }
            let mut out = Vec::with_capacity(len);
            let mut offset = 0usize;
            while offset < len {
                let block = (len - offset).min(32);
                let mut src = [0u8; 32];
                let mut exp = [0u8; 32];
                let mut num = [0u8; 32];
                src[..block].copy_from_slice(&source[offset..offset + block]);
                exp[..block].copy_from_slice(&plan.exponent[offset..offset + block]);
                num[..block].copy_from_slice(&plan.number[offset..offset + block]);
                // SAFETY: the arrays are thirty-two bytes long and the
                // instruction is available under the target feature.
                let bytes = unsafe { apply32(src, exp, num) };
                out.extend_from_slice(&bytes[..block]);
                offset += block;
            }
            Some(out)
        }

        /// Applies thirty-two corrections.
        ///
        /// # Safety
        ///
        /// Requires AVX2.
        #[target_feature(enable = "avx2")]
        unsafe fn apply32(src: [u8; 32], exp: [u8; 32], num: [u8; 32]) -> [u8; 32] {
            // SAFETY: the instruction is available under the target feature.
            unsafe {
                let src_v = _mm256_loadu_si256(src.as_ptr().cast());
                let exp_v = _mm256_loadu_si256(exp.as_ptr().cast());
                let num_v = _mm256_loadu_si256(num.as_ptr().cast());
                let shuffle =
                    _mm256_broadcastsi128_si256(_mm_loadu_si128(ssse3::NUMBER_SHUFFLE.as_ptr().cast()));
                let offset = _mm256_shuffle_epi8(shuffle, num_v);
                let scaled1 = _mm256_slli_epi16(offset, 1);
                let scaled2 = _mm256_slli_epi16(offset, 2);
                let scaled3 = _mm256_slli_epi16(offset, 3);
                let is0 = _mm256_cmpeq_epi8(exp_v, _mm256_setzero_si256());
                let is1 = _mm256_cmpeq_epi8(exp_v, _mm256_set1_epi8(1));
                let is2 = _mm256_cmpeq_epi8(exp_v, _mm256_set1_epi8(2));
                let is3 = _mm256_cmpeq_epi8(exp_v, _mm256_set1_epi8(3));
                let scaled = _mm256_or_si256(
                    _mm256_or_si256(_mm256_and_si256(offset, is0), _mm256_and_si256(scaled1, is1)),
                    _mm256_or_si256(_mm256_and_si256(scaled2, is2), _mm256_and_si256(scaled3, is3)),
                );
                let mut out = [0u8; 32];
                _mm256_storeu_si256(out.as_mut_ptr().cast(), _mm256_add_epi8(src_v, scaled));
                out
            }
        }

        /// Returns the number of leading bytes where `left` and `right` agree.
        ///
        /// # Safety
        ///
        /// Requires AVX2, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "avx2")]
        pub unsafe fn common_prefix(left: &[u8], right: &[u8]) -> usize {
            let len = left.len().min(right.len());
            let mut offset = 0usize;
            while offset < len {
                let block = (len - offset).min(32);
                let mut lhs = [0u8; 32];
                let mut rhs = [0u8; 32];
                lhs[..block].copy_from_slice(&left[offset..offset + block]);
                rhs[..block].copy_from_slice(&right[offset..offset + block]);
                // SAFETY: the arrays are thirty-two bytes long and the
                // instruction is available under the target feature.
                let diff = unsafe {
                    _mm256_cmpeq_epi8(
                        _mm256_loadu_si256(lhs.as_ptr().cast()),
                        _mm256_loadu_si256(rhs.as_ptr().cast()),
                    )
                };
                // SAFETY: the instruction is available under the target feature.
                let mask = _mm256_movemask_epi8(diff) as u32;
                let equal = mask.trailing_ones() as usize;
                if equal < block {
                    return offset + equal;
                }
                offset += block;
            }
            len
        }

        /// A 64-bit signature of a block.
        ///
        /// # Safety
        ///
        /// Requires AVX2, which the caller must have proved with a CPUID
        /// probe.
        #[target_feature(enable = "avx2")]
        pub unsafe fn block_signature(block: &[u8]) -> u64 {
            const PRIME: u64 = 0x0000_0100_0000_01B3;
            let mut hash = super::super::SIGNATURE_SEED;
            let mut index = 0u64;
            let mut offset = 0usize;
            while offset + 32 <= block.len() {
                // SAFETY: the load stays inside the block and the
                // instruction is available under the target feature.
                let chunk = unsafe { _mm256_loadu_si256(block.as_ptr().add(offset).cast()) };
                // SAFETY: the instruction is available under the target feature.
                let lanes = [_mm256_castsi256_si128(chunk), _mm256_extracti128_si256(chunk, 1)];
                for lane in lanes {
                    // SAFETY: the instruction is available under the target feature.
                    let low = _mm_cvtsi128_si64(lane) as u64;
                    // SAFETY: the instruction is available under the target feature.
                    let high = _mm_cvtsi128_si64(_mm_srli_si128(lane, 8)) as u64;
                    hash = hash.rotate_left(7) ^ low.wrapping_mul(PRIME).wrapping_add(index);
                    index += 1;
                    hash = hash.rotate_left(7) ^ high.wrapping_mul(PRIME).wrapping_add(index);
                    index += 1;
                }
                offset += 32;
            }
            // The words that do not fill a vector are folded exactly as
            // the scalar path folds them, so both agree byte for byte.
            let (hash, index) = super::super::fold_words(hash, index, &block[offset..]);
            let tail = &block[block.len() & !7..];
            let hash = super::super::fold_tail(hash, index, tail);
            hash ^ (hash >> 29)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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

    /// Every byte difference that a scale can reach.
    fn reachable_deltas() -> Vec<(i16, u8, u8)> {
        let mut out = vec![];
        for exponent in 0..4u8 {
            let scale = 1i16 << exponent;
            for number in 0..8u8 {
                let offset = (i16::from(number) - 3) * scale;
                out.push((offset, exponent, number));
            }
        }
        out
    }

    #[test]
    fn classification_agrees_with_the_reference() {
        for (delta, _exponent, _number) in reachable_deltas() {
            let classified = scalar::classify(delta);
            let (exponent, number) = classified.unwrap_or_else(|| panic!("delta {delta}"));
            // The classification must rebuild the difference exactly, and
            // it must pick a scale the difference actually sits on.
            let rebuilt = (i16::from(number) - 3) * (1i16 << exponent);
            assert_eq!(rebuilt, delta, "delta {delta}");
            assert!(
                delta % (1i16 << exponent) == 0,
                "delta {delta} does not sit on the scale it claims"
            );
        }
    }

    #[test]
    fn unrepresentable_deltas_are_rejected() {
        for delta in [-128i16, -33, -25, -13, -11, -9, -7, -5, 5, 7, 9, 33, 100, 127] {
            assert_eq!(scalar::classify(delta), None, "delta {delta}");
        }
    }

    #[test]
    fn plan_reconstructs_the_target() {
        let source = pseudo_random(64, 7);
        let mut target = source.clone();
        for (index, byte) in target.iter_mut().enumerate().take(16) {
            *byte = byte.wrapping_add((index % 5) as u8);
        }
        let plan = plan_delta(&source, &target);
        // Only the first sixteen bytes differ; the rest match exactly.
        assert_eq!(plan.representable, 64);
        assert_eq!(apply_delta(&source, &plan).as_deref(), Some(&target[..]));
    }

    #[test]
    fn plan_stops_at_the_first_unreachable_byte() {
        let source = [0u8; 32];
        let mut target = source;
        target[10] = 100; // +100 is out of reach for every scale
        let plan = plan_delta(&source, &target);
        assert_eq!(plan.representable, 10);
    }

    #[test]
    fn vector_paths_match_the_scalar_reference() {
        let cases: [Vec<u8>; 6] = [
            vec![0u8; 0],
            vec![7u8; 1],
            vec![3u8; 15],
            vec![9u8; 16],
            vec![11u8; 17],
            pseudo_random(97, 31),
        ];
        for case in &cases {
            let mut target = case.clone();
            for (index, byte) in target.iter_mut().enumerate() {
                if index % 3 == 0 {
                    *byte = byte.wrapping_add(1);
                }
            }
            let reference = scalar::plan_delta(case, &target);
            let vector = plan_delta(case, &target);
            assert_eq!(reference, vector, "case of {} bytes", case.len());

            let rebuilt = scalar::apply_delta(case, &reference);
            assert_eq!(rebuilt.as_deref(), Some(&target[..reference.representable]));
            let rebuilt = apply_delta(case, &reference);
            assert_eq!(rebuilt.as_deref(), Some(&target[..reference.representable]));
        }
    }

    #[test]
    fn every_difference_is_classified_the_same_way() {
        // A negative difference may only reach -3 and a positive one +4,
        // so the scale a difference needs depends on its sign.  The
        // vector path has to agree with the reference for all of them,
        // and the biased number must always stay in 0..=7.
        for difference in -128i16..=127 {
            let source = vec![100u8; 16];
            let mut target = source.clone();
            for byte in target.iter_mut() {
                *byte = (100i16 + difference) as u8;
            }
            let reference = scalar::plan_delta(&source, &target);
            let vector = plan_delta(&source, &target);
            assert_eq!(
                reference.representable, vector.representable,
                "difference {difference}"
            );
            assert_eq!(reference.exponent, vector.exponent, "difference {difference}");
            assert_eq!(reference.number, vector.number, "difference {difference}");
            if vector.representable > 0 {
                assert!(
                    vector.number[0] <= 7,
                    "difference {difference} produced the number {}",
                    vector.number[0]
                );
            }
        }
    }

    #[test]
    fn the_scale_a_difference_needs_depends_on_its_sign() {
        // -4 needs the scale two, +4 is reached by the scale one, and
        // -32 is out of reach for every scale.
        assert_eq!(scalar::classify(-4), Some((1, 1)));
        assert_eq!(scalar::classify(4), Some((0, 7)));
        assert_eq!(scalar::classify(-32), None);
        assert_eq!(scalar::classify(32), Some((3, 7)));
        assert_eq!(scalar::classify(-24), Some((3, 0)));
        assert_eq!(scalar::classify(24), Some((3, 6)));
    }

    #[test]
    fn common_prefix_matches_the_reference() {
        let left = pseudo_random(80, 5);
        let mut right = left.clone();
        right[41] = right[41].wrapping_add(1);
        assert_eq!(common_prefix(&left, &right), 41);
        assert_eq!(scalar::common_prefix(&left, &right), 41);
    }

    #[test]
    fn apply_rejects_a_malformed_plan() {
        let source = [0u8; 4];
        let plan = DeltaPlan {
            exponent: vec![0, 0, 9, 0],
            number: vec![3, 3, 3, 3],
            representable: 4,
        };
        assert!(apply_delta(&source, &plan).is_none());
        let plan = DeltaPlan {
            exponent: vec![0, 0],
            number: vec![3, 3, 3],
            representable: 3,
        };
        assert!(apply_delta(&source, &plan).is_none());
    }
}
