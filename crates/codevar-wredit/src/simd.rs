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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Runtime-dispatched SIMD kernels for the Wredit engine.
//!
//! The engine's hot loops — counting code units, mapping indices to
//! byte offsets, and the position-weighted integrity checksum — are
//! written once per instruction-set tier and selected at run time
//! through [`codevar_base::cpuid`]. Every tier produces bit-identical
//! results; the tier only changes how fast the answer appears.
//!
//! # Tier ladder
//!
//! [`SimdTier`] lists the rungs a machine can reach,
//! [`active_tier`] reports the widest one available here, and each
//! kernel picks the widest tier *its own instructions* allow:
//!
//! | Kernel | Scalar | SSE2 | SSSE3 | SSE4.1 | AVX2 | AVX-512 | AVX-512-IFMA |
//! |--------|--------|------|-------|--------|------|---------|--------------|
//! | [`count_utf8_units`] | yes | yes | — | — | yes | yes | — |
//! | [`linear_checksum`] | yes | — | yes | — | yes | yes | reported |
//! | index lookups | yes | via count | — | — | via count | via count | — |
//!
//! Two rungs are reported but never selected by a kernel, and the
//! reason is deliberate rather than an omission:
//!
//! - **SSE4.1** adds `_mm_mullo_epi32` / `_mm_cvtepu8_epi32`, which
//!   compute the checksum's `byte × weight` product *slower* than the
//!   SSSE3 `_mm_maddubs_epi16` pair the SSE tier already uses.
//! - **AVX-512-IFMA**'s 52-bit multiply-add is the mathematically
//!   natural fit for the checksum (both operands are far below the
//!   26-bit limit, and the 52-bit accumulator reduces modulo 2^30
//!   exactly), but widening eight bytes into each 64-bit lane costs
//!   about ten times the instructions of the `vpmaddubsw` path that
//!   every IFMA-capable CPU also has. It stays available as
//!   [`SimdTier`] metadata so callers can see what the machine
//!   offers.
//!
//! Non-x86 targets (and x86 machines with no vector extensions) run
//! the scalar kernels, which are the reference every SIMD tier is
//! tested against.
//!
//! # Safety
//!
//! Every `#[target_feature]` kernel below is `unsafe` and is only
//! reached after a run-time CPUID probe proved the feature, with the
//! proof restated in the `SAFETY` comment at each call site. See
//! [`active_tier`] for how the probe is memoized.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::x86::{
    checksum_avx2, checksum_avx512, checksum_ssse3, count_avx2, count_avx512, count_sse2,
};
use alloc::vec::Vec;
use core::fmt;

/// Number of bytes one SIMD chunk of the UTF-8 scanners owns.
const UTF8_CHUNK: usize = 64;

/// Period of the integrity checksum's position weight, in bytes.
const WEIGHT_PERIOD: usize = 13;

/// Clear mask for the checksum result: the value is taken modulo 2^30.
const CHECKSUM_MASK: u32 = (1 << 30) - 1;

/// Position weights for the integrity checksum.
///
/// `WEIGHT_ROWS[r][j]` is the weight of byte `j` inside a SIMD block
/// that starts at a position congruent to `r` modulo
/// [`WEIGHT_PERIOD`], i.e. `((r + j) % 13) + 1`. Because the weight
/// only depends on the position modulo 13, one table of 13 rows
/// covers every block size (16, 32 and 64 bytes) and every offset.
///
/// Weights are stored as `i8` because the SSSE3/AVX2/AVX-512 tiers
/// feed them to `pmaddubsw`, which multiplies unsigned bytes by
/// *signed* bytes; every weight is in `1..=13`, so the sign is never
/// an issue.
const WEIGHT_ROWS: [[i8; UTF8_CHUNK]; WEIGHT_PERIOD] = {
    let mut rows = [[0i8; UTF8_CHUNK]; WEIGHT_PERIOD];
    let mut row = 0;
    while row < WEIGHT_PERIOD {
        let mut column = 0;
        while column < UTF8_CHUNK {
            rows[row][column] = ((row + column) % WEIGHT_PERIOD + 1) as i8;
            column += 1;
        }
        row += 1;
    }
    rows
};

/// The instruction-set tier a Wredit build can dispatch to.
///
/// Rungs are ordered by capability: [`Scalar`](Self::Scalar) is the
/// portable baseline and each later variant requires everything the
/// previous ones do plus more. See the [module documentation][self]
/// for which rung each kernel actually selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SimdTier {
    /// Portable scalar loops; always available.
    Scalar = 0,
    /// SSE2: 16-byte byte-mask scans.
    Sse2 = 1,
    /// SSSE3: `pmaddubsw` byte×weight products for the checksum.
    Ssse3 = 2,
    /// SSE4.1: capability rung, no dedicated kernel (see module docs).
    Sse41 = 3,
    /// AVX2: 32-byte scans and 256-bit checksum products.
    Avx2 = 4,
    /// AVX-512 (F/BW): 64-byte scans and 512-bit checksum products.
    Avx512 = 5,
    /// AVX-512-IFMA: 52-bit multiply-add capability rung.
    Avx512Ifma = 6,
}

impl SimdTier {
    /// Every rung of the ladder, in ascending capability order.
    pub const ALL: &'static [SimdTier] = &[
        Self::Scalar,
        Self::Sse2,
        Self::Ssse3,
        Self::Sse41,
        Self::Avx2,
        Self::Avx512,
        Self::Avx512Ifma,
    ];

    /// Canonical spelling of the tier, matching the `target_feature`
    /// name where one exists.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Sse2 => "sse2",
            Self::Ssse3 => "ssse3",
            Self::Sse41 => "sse4.1",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512Ifma => "avx512-ifma",
        }
    }
}

impl fmt::Display for SimdTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The widest tier this CPU reports as available.
///
/// Combines the compile-time answer (`cfg(target_feature)`) with a
/// memoized run-time probe: the first call pays for CPUID, later
/// calls are a single atomic load. The AVX-512-IFMA rung needs the
/// supplementary leaf-7 bit and is only reported when AVX-512F (and
/// therefore the OS-enabled ZMM state) is present too.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::{SimdTier, active_tier};
///
/// let tier = active_tier();
/// assert!(tier >= SimdTier::Scalar);
/// assert_eq!(SimdTier::ALL.last().copied(), Some(SimdTier::Avx512Ifma));
/// ```
#[must_use]
pub fn active_tier() -> SimdTier {
    let mut tier = SimdTier::Scalar;
    for rung in SimdTier::ALL.iter().skip(1) {
        if tier_available(*rung) {
            tier = *rung;
        } else {
            break;
        }
    }
    tier
}

/// Counts the UTF-8 code units in `bytes`.
///
/// A code unit starts at every byte that is not a continuation byte
/// (`10xxxxxx`), so a chunk may begin mid-sequence and the partial
/// counts still sum to exactly the number of code points — the
/// property the block storage layer relies on when it counts a block
/// in isolation.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::count_utf8_units;
///
/// assert_eq!(count_utf8_units(b"abc"), 3);
/// assert_eq!(count_utf8_units("héllo".as_bytes()), 5);
/// assert_eq!(count_utf8_units("🦀".as_bytes()), 1);
/// assert_eq!(count_utf8_units(b""), 0);
/// ```
#[inline]
#[must_use]
pub fn count_utf8_units(bytes: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if tier_available(SimdTier::Avx512) {
            // SAFETY: `tier_available(Avx512)` proves AVX-512F/BW and
            // the OS-enabled ZMM state; `count_avx512` only reads
            // `bytes` in full 64-byte chunks plus one scalar tail.
            return unsafe { count_avx512(bytes) };
        }
        if tier_available(SimdTier::Avx2) {
            // SAFETY: `tier_available(Avx2)` proves AVX2 (and its
            // AVX/SSE state enablement); chunking keeps every load
            // inside `bytes`.
            return unsafe { count_avx2(bytes) };
        }
        if tier_available(SimdTier::Sse2) {
            // SAFETY: `tier_available(Sse2)` proves SSE2, which every
            // 64-bit x86 and nearly every 32-bit x86 provides; loads
            // stay inside `bytes`.
            return unsafe { count_sse2(bytes) };
        }
    }
    count_scalar(bytes)
}

/// Maps a byte offset to the code unit that starts there (or the one
/// that spans it).
///
/// Counts the code-unit starts in `bytes[..byte_index]`; when
/// `byte_index` lands inside a multi-byte sequence the count drops by
/// one, so the answer is the index of the code unit containing
/// `byte_index`. Offsets past the end of the buffer clamp to the
/// number of code units the buffer holds.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::utf8_byte_index_to_unit;
///
/// let bytes = "aé".as_bytes();
/// assert_eq!(utf8_byte_index_to_unit(bytes, 0), 0);
/// assert_eq!(utf8_byte_index_to_unit(bytes, 1), 1);
/// // Byte 2 is a continuation byte: the unit index does not advance.
/// assert_eq!(utf8_byte_index_to_unit(bytes, 2), 1);
/// assert_eq!(utf8_byte_index_to_unit(bytes, 3), 2);
/// ```
#[inline]
#[must_use]
pub fn utf8_byte_index_to_unit(bytes: &[u8], byte_index: usize) -> u64 {
    let starts = count_utf8_units(bytes.get(..byte_index).unwrap_or(bytes));
    match bytes.get(byte_index) {
        // The prefix counted the unit that starts at `byte_index - 1`
        // (or earlier): back up to the unit that spans this byte.
        Some(&byte) if is_continuation(byte) => starts.saturating_sub(1),
        _ => starts,
    }
}

/// Maps a zero-based code unit index to its byte offset.
///
/// `unit_index == count` yields the end of the buffer, so a cursor
/// sitting one past the last code point still round-trips; anything
/// past that is [`None`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::utf8_unit_index_to_byte;
///
/// let bytes = "aé".as_bytes();
/// assert_eq!(utf8_unit_index_to_byte(bytes, 0), Some(0));
/// assert_eq!(utf8_unit_index_to_byte(bytes, 1), Some(1));
/// assert_eq!(utf8_unit_index_to_byte(bytes, 2), Some(3));
/// assert_eq!(utf8_unit_index_to_byte(bytes, 3), None);
/// ```
#[must_use]
pub fn utf8_unit_index_to_byte(bytes: &[u8], unit_index: u64) -> Option<usize> {
    let total = count_utf8_units(bytes);
    if unit_index > total {
        return None;
    }
    if unit_index == total {
        return Some(bytes.len());
    }
    let mut seen = 0u64;
    let mut chunk_number = 0usize;
    while seen <= unit_index {
        let base = chunk_number.checked_mul(UTF8_CHUNK)?;
        let end = base.saturating_add(UTF8_CHUNK).min(bytes.len());
        let chunk = bytes.get(base..end)?;
        let starts = count_utf8_units(chunk);
        if unit_index < seen + starts {
            let mut remaining = unit_index - seen;
            for (offset, &byte) in chunk.iter().enumerate() {
                if !is_continuation(byte) {
                    if remaining == 0 {
                        return Some(base + offset);
                    }
                    remaining -= 1;
                }
            }
            return None;
        }
        seen += starts;
        chunk_number += 1;
    }
    None
}

/// Maps many sorted code unit indices to byte offsets in one pass.
///
/// The indices must be sorted ascending (the cursor pool keeps them
/// that way); unsorted input or an index past the end of the buffer
/// is reported as [`None`] rather than silently reordered.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::utf8_char_indices_to_byte_offsets;
///
/// let bytes = "héllo".as_bytes();
/// let offsets = utf8_char_indices_to_byte_offsets(bytes, &[0, 1, 4]);
/// assert_eq!(offsets, Some(vec![0, 1, 5]));
/// assert_eq!(utf8_char_indices_to_byte_offsets(bytes, &[4, 1]), None);
/// ```
#[must_use]
pub fn utf8_char_indices_to_byte_offsets(bytes: &[u8], unit_indices: &[u64]) -> Option<Vec<usize>> {
    let mut offsets = Vec::with_capacity(unit_indices.len());
    if unit_indices.is_empty() {
        return Some(offsets);
    }
    let total = count_utf8_units(bytes);
    let mut previous = 0u64;
    for &index in unit_indices {
        if index < previous || index > total {
            return None;
        }
        previous = index;
    }
    let mut seen = 0u64;
    let mut chunk_number = 0usize;
    for &wanted in unit_indices {
        if wanted == total {
            offsets.push(bytes.len());
            continue;
        }
        'chunk: loop {
            let base = chunk_number.checked_mul(UTF8_CHUNK)?;
            let end = base.saturating_add(UTF8_CHUNK).min(bytes.len());
            let chunk = bytes.get(base..end)?;
            let starts = count_utf8_units(chunk);
            if wanted < seen + starts {
                let mut remaining = wanted - seen;
                for (offset, &byte) in chunk.iter().enumerate() {
                    if !is_continuation(byte) {
                        if remaining == 0 {
                            offsets.push(base + offset);
                            break 'chunk;
                        }
                        remaining -= 1;
                    }
                }
                return None;
            }
            seen += starts;
            chunk_number += 1;
        }
    }
    Some(offsets)
}

/// Maps a UTF-16 byte offset to a code unit index.
///
/// UTF-16 code units are fixed width, so no scan is needed: the index
/// is half the byte offset (odd offsets round down, matching the
/// block layer's "unit = two bytes" accounting).
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::utf16_byte_index_to_unit;
///
/// assert_eq!(utf16_byte_index_to_unit(0), 0);
/// assert_eq!(utf16_byte_index_to_unit(5), 2);
/// ```
#[inline]
#[must_use]
pub const fn utf16_byte_index_to_unit(byte_index: usize) -> u64 {
    (byte_index / 2) as u64
}

/// Maps a UTF-16 code unit index to its byte offset.
///
/// Returns [`None`] when the doubled offset no longer fits a
/// `usize`.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::utf16_unit_index_to_byte;
///
/// assert_eq!(utf16_unit_index_to_byte(0), Some(0));
/// assert_eq!(utf16_unit_index_to_byte(3), Some(6));
/// ```
#[inline]
#[must_use]
pub fn utf16_unit_index_to_byte(unit_index: u64) -> Option<usize> {
    match unit_index.checked_mul(2) {
        Some(bytes) => usize::try_from(bytes).ok(),
        None => None,
    }
}

/// Position-weighted integrity checksum of a byte range.
///
/// Computes
///
/// ```text
/// ( Σ bytes[i] * ((i % 13) + 1) ) mod 2^30
/// ```
///
/// The weight cycles with period 13 so a one-byte edit changes both
/// its own term and the weights that follow it inside the period,
/// while a multi-byte insertion shifts every later term. The result
/// is the value the `block_checksum` and
/// `history_checksum` kernels produce — tests assert the CPU
/// and GPU paths agree byte for byte, because a disagreement would
/// surface as a false "corrupted block" report.
///
/// All tiers accumulate with wrapping arithmetic and reduce modulo
/// 2^30 once at the end; since 2^30 divides 2^32 (and 2^52), the
/// wrap points cannot change the answer.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::linear_checksum;
///
/// assert_eq!(linear_checksum(b""), 0);
/// // 0x41 * 1 + 0x42 * 2 = 65 + 132 = 197.
/// assert_eq!(linear_checksum(b"AB"), 197);
/// ```
#[must_use]
pub fn linear_checksum(bytes: &[u8]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if tier_available(SimdTier::Avx512) {
            // SAFETY: `tier_available(Avx512)` proves AVX-512F/BW and
            // ZMM state; the kernel only reads `bytes` and writes to
            // stack-allocated vectors.
            return unsafe { checksum_avx512(bytes) };
        }
        if tier_available(SimdTier::Avx2) {
            // SAFETY: `tier_available(Avx2)` proves AVX2; loads stay
            // inside `bytes`.
            return unsafe { checksum_avx2(bytes) };
        }
        if tier_available(SimdTier::Ssse3) {
            // SAFETY: `tier_available(Ssse3)` proves SSSE3 (and with
            // it SSE3's `phaddd`); loads stay inside `bytes`.
            return unsafe { checksum_ssse3(bytes) };
        }
    }
    checksum_scalar(bytes)
}

/// Reference implementation of [`linear_checksum`].
///
/// Every SIMD tier is tested against this loop; it also runs on
/// targets with no vector extensions.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::checksum_scalar;
///
/// assert_eq!(checksum_scalar(b"AB"), 197);
/// ```
#[must_use]
pub fn checksum_scalar(bytes: &[u8]) -> u32 {
    checksum_scalar_at(bytes, 0)
}

/// Like [`checksum_scalar`], but the first byte of `bytes` sits at
/// absolute position `offset`, so byte `i` is weighted
/// `(offset + i) % 13 + 1`.
///
/// The SIMD tiers use this for the tail block that follows their
/// vectorised chunks: that block starts past the chunks, and
/// restarting the weights at zero would disagree with the reference.
#[inline]
fn checksum_scalar_at(bytes: &[u8], offset: usize) -> u32 {
    let mut sum = 0u64;
    for (index, &byte) in bytes.iter().enumerate() {
        let weight = (offset + index) % WEIGHT_PERIOD + 1;
        sum += u64::from(byte) * weight as u64;
    }
    (sum & u64::from(CHECKSUM_MASK)) as u32
}

/// `true` when `byte` is a UTF-8 continuation byte (`10xxxxxx`).
#[inline]
const fn is_continuation(byte: u8) -> bool {
    byte & 0xC0 == 0x80
}

/// Scalar UTF-8 unit count; the reference for the vector tiers.
#[inline]
fn count_scalar(bytes: &[u8]) -> u64 {
    let mut count = 0u64;
    for &byte in bytes {
        if !is_continuation(byte) {
            count += 1;
        }
    }
    count
}

/// `true` when every rung up to and including `tier` is available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn tier_available(tier: SimdTier) -> bool {
    use codevar_base::cpuid::{Feature, has};

    match tier {
        SimdTier::Scalar => true,
        SimdTier::Sse2 => has(Feature::Sse2),
        SimdTier::Ssse3 => has(Feature::Ssse3) && has(Feature::Sse2),
        SimdTier::Sse41 => has(Feature::Sse41) && has(Feature::Ssse3),
        SimdTier::Avx2 => has(Feature::Avx2) && has(Feature::Sse41),
        SimdTier::Avx512 => has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2),
        // The leaf-7 feature bit only means anything once AVX-512F
        // (and therefore XGETBV's ZMM enablement) is confirmed, so
        // the umbrella check comes first.
        SimdTier::Avx512Ifma => has(Feature::Avx512F) && has(Feature::Avx512Bw) && ifma_bit_set(),
    }
}

/// Reads CPUID leaf 7, sub-leaf 0, EBX bit 21 (AVX512-IFMA).
///
/// Memoized in a [`spin::Once`] so the leaf is executed at most once
/// per process; gated on AVX-512F by the caller.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn ifma_bit_set() -> bool {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::__cpuid_count;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::__cpuid_count;
    use core::sync::atomic::{AtomicBool, Ordering};

    static DONE: AtomicBool = AtomicBool::new(false);
    static mut IFMA: bool = false;

    // RelAcquire pairs with the Release in the initializing thread:
    // once `DONE` reads `true`, `IFMA` has been written.
    if DONE.load(Ordering::Acquire) {
        // SAFETY: `IFMA` is only written before `DONE` is set to
        // `true`, and the atomic load above synchronizes with that
        // store, so the read below observes a fully initialized
        // value. The statics are never written again.
        return unsafe { IFMA };
    }
    #[cfg(target_arch = "x86")]
    let bit = unsafe { __cpuid_count(7, 0) }.ebx & (1 << 21) != 0;
    #[cfg(target_arch = "x86_64")]
    let bit = unsafe { __cpuid_count(7, 0) }.ebx & (1 << 21) != 0;
    // SAFETY: single-writer pattern guarded by `DONE` above; the
    // value is a plain `bool`.
    unsafe { IFMA = bit };
    DONE.store(true, Ordering::Release);
    bit
}

/// `true` when `tier` is available on non-x86 targets (only
/// [`SimdTier::Scalar`]).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
const fn tier_available(tier: SimdTier) -> bool {
    matches!(tier, SimdTier::Scalar)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::{CHECKSUM_MASK, UTF8_CHUNK, WEIGHT_PERIOD, WEIGHT_ROWS, checksum_scalar_at, count_scalar};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Counts UTF-8 unit starts over 16-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires SSE2 (provided by every x86_64 and by `cpuid`'s
    /// `Sse2` feature on 32-bit x86).
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn count_sse2(bytes: &[u8]) -> u64 {
        // SAFETY: the caller proved SSE2; `chunks_exact(16)` yields
        // only full blocks, so every `loadu` stays inside `bytes`.
        unsafe {
            let and_mask = _mm_set1_epi8(0xC0u8 as i8);
            let eq_mask = _mm_set1_epi8(0x80u8 as i8);
            let mut total = 0u64;
            let mut chunks = bytes.chunks_exact(16);
            for chunk in &mut chunks {
                let data = _mm_loadu_si128(chunk.as_ptr().cast());
                let masked = _mm_and_si128(data, and_mask);
                let continuation = _mm_cmpeq_epi8(masked, eq_mask);
                let bits = _mm_movemask_epi8(continuation) as u32;
                total += 16 - u64::from(bits.count_ones());
            }
            total + count_scalar(chunks.remainder())
        }
    }

    /// Counts UTF-8 unit starts over 32-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn count_avx2(bytes: &[u8]) -> u64 {
        // SAFETY: the caller proved AVX2; `chunks_exact(32)` keeps
        // every load inside `bytes`.
        unsafe {
            let and_mask = _mm256_set1_epi8(0xC0u8 as i8);
            let eq_mask = _mm256_set1_epi8(0x80u8 as i8);
            let mut total = 0u64;
            let mut chunks = bytes.chunks_exact(32);
            for chunk in &mut chunks {
                let data = _mm256_loadu_si256(chunk.as_ptr().cast());
                let masked = _mm256_and_si256(data, and_mask);
                let continuation = _mm256_cmpeq_epi8(masked, eq_mask);
                let bits = _mm256_movemask_epi8(continuation) as u32;
                total += 32 - u64::from(bits.count_ones());
            }
            total + count_scalar(chunks.remainder())
        }
    }

    /// Counts UTF-8 unit starts over 64-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub(super) unsafe fn count_avx512(bytes: &[u8]) -> u64 {
        // SAFETY: the caller proved AVX-512F/BW; `chunks_exact(64)`
        // keeps every load inside `bytes`.
        unsafe {
            let and_mask = _mm512_set1_epi8(0xC0u8 as i8);
            let eq_mask = _mm512_set1_epi8(0x80u8 as i8);
            let mut total = 0u64;
            let mut chunks = bytes.chunks_exact(64);
            for chunk in &mut chunks {
                let data = _mm512_loadu_si512(chunk.as_ptr().cast());
                let masked = _mm512_and_si512(data, and_mask);
                let continuation = _mm512_cmpeq_epi8_mask(masked, eq_mask);
                total += 64 - u64::from(continuation.count_ones());
            }
            total + count_scalar(chunks.remainder())
        }
    }

    /// Returns the weight row for a block starting at `base`.
    #[inline]
    fn weight_row(base: usize) -> &'static [i8; UTF8_CHUNK] {
        // 13 is prime and the table has exactly 13 rows, so the
        // remainder is always a valid index.
        &WEIGHT_ROWS[base % WEIGHT_PERIOD]
    }

    /// Reduces four packed `i32` lanes to one `mod 2^30` value.
    ///
    /// # Safety
    ///
    /// Requires SSE3's `phaddd` (implied by SSSE3).
    #[target_feature(enable = "ssse3")]
    unsafe fn reduce_sse(lanes: __m128i) -> u32 {
        // The horizontal adds carry the `ssse3` feature this function
        // is compiled for, so both calls are safe here; the callers
        // proved the feature before entering this `unsafe fn`.
        let pairs = _mm_hadd_epi32(lanes, lanes);
        let total = _mm_hadd_epi32(pairs, pairs);
        (_mm_cvtsi128_si32(total) as u32) & CHECKSUM_MASK
    }

    /// Checksum over 16-byte blocks with `pmaddubsw`/`pmaddwd`.
    ///
    /// # Safety
    ///
    /// Requires SSSE3 (for `pmaddubsw` and `phaddd`).
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn checksum_ssse3(bytes: &[u8]) -> u32 {
        // SAFETY: the caller proved SSSE3; `chunks_exact(16)` keeps
        // every load inside `bytes`, and the products fit the `i16`
        // lanes because `255 * 13 <= 3315` and `pmaddubsw` sums pairs
        // (`6630 <= 32767`) before `pmaddwd` widens them.
        unsafe {
            let ones = _mm_set1_epi16(1);
            let mut accum = _mm_setzero_si128();
            let mut base = 0usize;
            let mut chunks = bytes.chunks_exact(16);
            for chunk in &mut chunks {
                let data = _mm_loadu_si128(chunk.as_ptr().cast());
                let weights = _mm_loadu_si128(weight_row(base).as_ptr().cast());
                let products = _mm_maddubs_epi16(data, weights);
                let summed = _mm_madd_epi16(products, ones);
                accum = _mm_add_epi32(accum, summed);
                base += 16;
            }
            let mut total = reduce_sse(accum);
            total = (total + (checksum_scalar_at(chunks.remainder(), base) & CHECKSUM_MASK)) & CHECKSUM_MASK;
            total
        }
    }

    /// Checksum over 32-byte blocks with the AVX2 `vpmaddubsw` pair.
    ///
    /// # Safety
    ///
    /// Requires AVX2 (which implies SSSE3).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn checksum_avx2(bytes: &[u8]) -> u32 {
        // SAFETY: the caller proved AVX2; `chunks_exact(32)` keeps
        // every load inside `bytes`; the `i16` lane bounds shown for
        // `checksum_ssse3` hold here too.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut accum = _mm256_setzero_si256();
            let mut base = 0usize;
            let mut chunks = bytes.chunks_exact(32);
            for chunk in &mut chunks {
                let data = _mm256_loadu_si256(chunk.as_ptr().cast());
                let weights = _mm256_loadu_si256(weight_row(base).as_ptr().cast());
                let products = _mm256_maddubs_epi16(data, weights);
                let summed = _mm256_madd_epi16(products, ones);
                accum = _mm256_add_epi32(accum, summed);
                base += 32;
            }
            let low = _mm256_castsi256_si128(accum);
            let high = _mm256_extracti128_si256(accum, 1);
            let lanes = _mm_add_epi32(low, high);
            let mut total = reduce_sse(lanes);
            total = (total + (checksum_scalar_at(chunks.remainder(), base) & CHECKSUM_MASK)) & CHECKSUM_MASK;
            total
        }
    }

    /// Checksum over 64-byte blocks with the AVX-512 `vpmaddubsw` pair.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub(super) unsafe fn checksum_avx512(bytes: &[u8]) -> u32 {
        // SAFETY: the caller proved AVX-512F/BW; `chunks_exact(64)`
        // keeps every load inside `bytes`; the `i16` lane bounds hold
        // as above.
        unsafe {
            let ones = _mm512_set1_epi16(1);
            let mut accum = _mm512_setzero_si512();
            let mut base = 0usize;
            let mut chunks = bytes.chunks_exact(64);
            for chunk in &mut chunks {
                let data = _mm512_loadu_si512(chunk.as_ptr().cast());
                let weights = _mm512_loadu_si512(weight_row(base).as_ptr().cast());
                let products = _mm512_maddubs_epi16(data, weights);
                let summed = _mm512_madd_epi16(products, ones);
                accum = _mm512_add_epi32(accum, summed);
                base += 64;
            }
            let mut total = (_mm512_reduce_add_epi32(accum) as u32) & CHECKSUM_MASK;
            total = (total + (checksum_scalar_at(chunks.remainder(), base) & CHECKSUM_MASK)) & CHECKSUM_MASK;
            total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes (xorshift), so the SIMD
    /// tiers meet inputs no hand-written case would cover.
    fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            out.push(state as u8);
        }
        out
    }

    /// Random *valid* UTF-8 (ASCII, 2-, 3- and 4-byte sequences).
    fn pseudo_random_utf8(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = state % 0x11_0000;
            let mut buffer = [0u8; 4];
            let text = char::from_u32(value).unwrap_or(char::REPLACEMENT_CHARACTER);
            let encoded = text.encode_utf8(&mut buffer);
            out.extend_from_slice(encoded.as_bytes());
        }
        out.truncate(len);
        // Never cut a multi-byte sequence: trim back to a boundary so
        // the buffer stays valid.
        while !out.is_empty() && (out.last().copied().unwrap_or(0) & 0xC0) == 0x80 {
            out.pop();
        }
        out
    }

    #[test]
    fn tier_ladder_is_ordered_and_named() {
        assert_eq!(SimdTier::ALL.first().copied(), Some(SimdTier::Scalar));
        assert_eq!(SimdTier::Scalar.name(), "scalar");
        assert_eq!(SimdTier::Avx512Ifma.name(), "avx512-ifma");
        let tier = active_tier();
        assert!(SimdTier::ALL.contains(&tier), "unknown active tier");
        assert_eq!(tier.to_string(), tier.name());
    }

    #[test]
    fn active_tier_is_monotonic_with_availability() {
        // Repeated calls must agree (the probe is memoized).
        assert_eq!(active_tier(), active_tier());
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            assert!(tier_available(SimdTier::Scalar));
            if active_tier() >= SimdTier::Sse2 {
                assert!(tier_available(SimdTier::Sse2));
            }
            if active_tier() >= SimdTier::Avx2 {
                assert!(tier_available(SimdTier::Avx2));
            }
        }
    }

    #[test]
    fn count_matches_reference_on_random_bytes() {
        for seed in 1..40u32 {
            let bytes = pseudo_random(1 + (seed as usize * 37) % 500, seed);
            assert_eq!(count_utf8_units(&bytes), count_scalar(&bytes), "seed {seed}");
        }
    }

    #[test]
    fn count_handles_boundaries() {
        assert_eq!(count_utf8_units(b""), 0);
        assert_eq!(count_utf8_units(&[0x80]), 0);
        assert_eq!(count_utf8_units(&[0xC3, 0xA9]), 1);
        assert_eq!(count_utf8_units(&[0xF0, 0x9F, 0xA6, 0x80]), 1);
        // A sequence straddling the 64-byte block boundary is still
        // counted once: the lead at index 63 pairs with the `0xA9`
        // continuation across the boundary and adds no extra unit,
        // while every other `0xC3` starts one — `UTF8_CHUNK` in all.
        let mut split = vec![0xC3; UTF8_CHUNK];
        split.push(0xA9);
        assert_eq!(count_utf8_units(&split), UTF8_CHUNK as u64);
    }

    #[test]
    fn unit_and_byte_indices_round_trip() {
        let bytes = pseudo_random_utf8(400, 7);
        let total = count_utf8_units(&bytes);
        let mut expected = 0u64;
        let mut offsets = Vec::new();
        for (index, &byte) in bytes.iter().enumerate() {
            if !is_continuation(byte) {
                assert_eq!(utf8_unit_index_to_byte(&bytes, expected), Some(index));
                assert_eq!(utf8_byte_index_to_unit(&bytes, index), expected);
                offsets.push(expected);
                expected += 1;
            }
        }
        assert_eq!(expected, total);
        assert_eq!(utf8_unit_index_to_byte(&bytes, total), Some(bytes.len()));
        assert_eq!(utf8_unit_index_to_byte(&bytes, total + 1), None);
        assert_eq!(
            utf8_char_indices_to_byte_offsets(&bytes, &offsets),
            Some(
                offsets
                    .iter()
                    .map(|&unit| utf8_unit_index_to_byte(&bytes, unit).unwrap_or(0))
                    .collect::<Vec<usize>>()
            )
        );
    }

    #[test]
    fn char_indices_reject_unsorted_and_out_of_range() {
        let bytes = "abc".as_bytes();
        assert_eq!(utf8_char_indices_to_byte_offsets(bytes, &[]), Some(vec![]));
        assert_eq!(
            utf8_char_indices_to_byte_offsets(bytes, &[0, 2]),
            Some(vec![0, 2])
        );
        assert_eq!(utf8_char_indices_to_byte_offsets(bytes, &[2, 0]), None);
        assert_eq!(utf8_char_indices_to_byte_offsets(bytes, &[4]), None);
    }

    #[test]
    fn utf16_index_mapping_is_fixed_width() {
        assert_eq!(utf16_byte_index_to_unit(0), 0);
        assert_eq!(utf16_byte_index_to_unit(6), 3);
        assert_eq!(utf16_unit_index_to_byte(3), Some(6));
        assert_eq!(utf16_unit_index_to_byte(u64::MAX), None);
    }

    #[test]
    fn checksum_matches_reference_on_random_bytes() {
        assert_eq!(linear_checksum(b""), 0);
        assert_eq!(linear_checksum(b"AB"), 197);
        for seed in 1..40u32 {
            let bytes = pseudo_random(1 + (seed as usize * 53) % 700, seed);
            assert_eq!(linear_checksum(&bytes), checksum_scalar(&bytes), "seed {seed}");
        }
    }

    #[test]
    fn checksum_detects_single_byte_edits() {
        let mut bytes = pseudo_random(300, 11);
        let original = linear_checksum(&bytes);
        for position in [0usize, 1, 12, 13, 14, 299] {
            let saved = bytes[position];
            bytes[position] ^= 0x5A;
            assert_ne!(linear_checksum(&bytes), original, "position {position}");
            bytes[position] = saved;
        }
        assert_eq!(linear_checksum(&bytes), original);
    }

    /// Each available SIMD tier must agree with the scalar reference,
    /// including on lengths that are not a multiple of the block size.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn simd_tiers_agree_with_scalar() {
        for seed in 1..25u32 {
            let bytes = pseudo_random(1 + (seed as usize * 61) % 800, seed);
            if tier_available(SimdTier::Sse2) {
                // SAFETY: `tier_available` proved SSE2.
                let got = unsafe { x86::count_sse2(&bytes) };
                assert_eq!(got, count_scalar(&bytes), "sse2 seed {seed}");
            }
            if tier_available(SimdTier::Avx2) {
                // SAFETY: `tier_available` proved AVX2.
                let got = unsafe { x86::count_avx2(&bytes) };
                assert_eq!(got, count_scalar(&bytes), "avx2 seed {seed}");
            }
            if tier_available(SimdTier::Avx512) {
                // SAFETY: `tier_available` proved AVX-512F/BW.
                let got = unsafe { x86::count_avx512(&bytes) };
                assert_eq!(got, count_scalar(&bytes), "avx512 seed {seed}");
            }
            if tier_available(SimdTier::Ssse3) {
                // SAFETY: `tier_available` proved SSSE3.
                let got = unsafe { x86::checksum_ssse3(&bytes) };
                assert_eq!(got, checksum_scalar(&bytes), "ssse3 seed {seed}");
            }
            if tier_available(SimdTier::Avx2) {
                // SAFETY: `tier_available` proved AVX2.
                let got = unsafe { x86::checksum_avx2(&bytes) };
                assert_eq!(got, checksum_scalar(&bytes), "avx2 seed {seed}");
            }
            if tier_available(SimdTier::Avx512) {
                // SAFETY: `tier_available` proved AVX-512F/BW.
                let got = unsafe { x86::checksum_avx512(&bytes) };
                assert_eq!(got, checksum_scalar(&bytes), "avx512 seed {seed}");
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn weight_rows_match_the_scalar_formula() {
        for (row, weights) in WEIGHT_ROWS.iter().enumerate() {
            for (column, &weight) in weights.iter().enumerate() {
                let expected = (row + column) % WEIGHT_PERIOD + 1;
                assert_eq!(weight, expected as i8, "row {row} column {column}");
            }
        }
    }
}
