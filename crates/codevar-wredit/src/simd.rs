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
//! implemented once per architecture in dedicated submodules and
//! selected at run time through [`codevar_base::cpuid`]. Every path
//! produces bit-identical results; only the speed differs.
//!
//! Non-x86 targets (and x86 machines with no vector extensions) run
//! the scalar kernels, which are the reference every SIMD path is
//! tested against.
//!
//! # Safety
//!
//! Every `#[target_feature]` kernel below is `unsafe` and is only
//! reached after a run-time CPUID probe proved the feature, with the
//! proof restated in the `SAFETY` comment at each call site.

use alloc::vec::Vec;

/// Number of bytes one SIMD chunk of the UTF-8 scanners owns.
const UTF8_CHUNK: usize = 64;

/// Prefix block size for index-to-offset mapping.
///
/// Counting whole 4 KiB blocks keeps the run-time dispatch and scalar
/// boundary work amortised; only the block that contains the target is
/// rescanned byte by byte.
const UTF8_SCAN_CHUNK: usize = 4096;

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
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
            // SAFETY: the CPUID probes above confirmed AVX-512F/BW (and
            // the OS-enabled ZMM state) plus AVX2; `x86::count_avx512`
            // only reads `bytes` in full 64-byte chunks plus one scalar
            // tail.
            return unsafe { x86::count_avx512(bytes) };
        }
        if has(Feature::Avx2) && has(Feature::Sse41) {
            // SAFETY: the probes above confirmed AVX2 (and AVX/SSE state
            // enablement); chunking keeps every load inside `bytes`.
            return unsafe { x86::count_avx2(bytes) };
        }
        if has(Feature::Sse2) {
            // SAFETY: the probe above confirmed SSE2, which every
            // 64-bit x86 and nearly every 32-bit x86 provides; loads
            // stay inside `bytes`.
            return unsafe { x86::count_sse2(bytes) };
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
    let mut seen = 0u64;
    let mut chunk_number = 0usize;
    loop {
        let base = chunk_number.checked_mul(UTF8_SCAN_CHUNK)?;
        if base >= bytes.len() {
            // Past the last unit: only the one-past-the-end index maps.
            return (unit_index == seen).then_some(bytes.len());
        }
        let end = base
            .saturating_add(UTF8_SCAN_CHUNK)
            .min(bytes.len());
        let chunk = bytes.get(base..end)?;
        let starts = count_utf8_units(chunk);
        if unit_index < seen.saturating_add(starts) {
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
        seen = seen.saturating_add(starts);
        chunk_number += 1;
    }
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
    // Validate ordering up front; range validation happens while the
    // prefix is scanned, so the buffer is never counted twice.
    let mut previous = 0u64;
    for &index in unit_indices {
        if index < previous {
            return None;
        }
        previous = index;
    }
    let mut seen = 0u64;
    let mut chunk_number = 0usize;
    for &wanted in unit_indices {
        'chunk: loop {
            let base = chunk_number.checked_mul(UTF8_SCAN_CHUNK)?;
            if base >= bytes.len() {
                if wanted == seen {
                    offsets.push(bytes.len());
                    break 'chunk;
                }
                return None;
            }
            let end = base
                .saturating_add(UTF8_SCAN_CHUNK)
                .min(bytes.len());
            let chunk = bytes.get(base..end)?;
            let starts = count_utf8_units(chunk);
            if wanted < seen.saturating_add(starts) {
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
            seen = seen.saturating_add(starts);
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
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
            // SAFETY: the probes above confirmed AVX-512F/BW and the
            // ZMM state; the kernel only reads `bytes` and writes to
            // stack-allocated vectors.
            return unsafe { x86::checksum_avx512(bytes) };
        }
        if has(Feature::Avx2) && has(Feature::Sse41) {
            // SAFETY: the probe above confirmed AVX2; loads stay
            // inside `bytes`.
            return unsafe { x86::checksum_avx2(bytes) };
        }
        if has(Feature::Ssse3) && has(Feature::Sse2) {
            // SAFETY: the probes above confirmed SSSE3 (and with it
            // SSE3's `phaddd`); loads stay inside `bytes`.
            return unsafe { x86::checksum_ssse3(bytes) };
        }
    }
    checksum_scalar(bytes)
}

/// Counts how many times the byte `needle` appears in `bytes`.
///
/// The engine uses this for the cheap, byte-oriented statistics of the
/// hot paths: line feeds and tabs while resolving line/column, and the
/// separator scans of the directory-name hashing. Every vector tier is
/// required to agree with [`count_byte_scalar`] bit for bit.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::count_byte;
///
/// assert_eq!(count_byte(b"a\nb\nc", b'\n'), 2);
/// assert_eq!(count_byte(b"", b'\n'), 0);
/// ```
#[must_use]
pub fn count_byte(bytes: &[u8], needle: u8) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
            // SAFETY: AVX-512F/BW and the OS-enabled ZMM state were
            // confirmed; the kernel reads only `bytes`.
            return unsafe { x86::count_byte_avx512(bytes, needle) };
        }
        if has(Feature::Avx2) && has(Feature::Sse41) {
            // SAFETY: AVX2 was confirmed; loads stay inside `bytes`.
            return unsafe { x86::count_byte_avx2(bytes, needle) };
        }
        if has(Feature::Sse2) {
            // SAFETY: SSE2 was confirmed; loads stay inside `bytes`.
            return unsafe { x86::count_byte_sse2(bytes, needle) };
        }
    }
    count_byte_scalar(bytes, needle)
}

/// Reference implementation of [`count_byte`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::count_byte_scalar;
///
/// assert_eq!(count_byte_scalar(b"aaba", b'a'), 3);
/// ```
#[must_use]
pub fn count_byte_scalar(bytes: &[u8], needle: u8) -> u64 {
    bytes
        .iter()
        .filter(|&&byte| byte == needle)
        .count() as u64
}

/// Index of the first occurrence of `needle` in `bytes`.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::find_byte;
///
/// assert_eq!(find_byte(b"hello", b'l'), Some(2));
/// assert_eq!(find_byte(b"hello", b'z'), None);
/// ```
#[must_use]
pub fn find_byte(bytes: &[u8], needle: u8) -> Option<usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
            // SAFETY: AVX-512F/BW were confirmed; the kernel reads
            // only `bytes`.
            return unsafe { x86::find_byte_avx512(bytes, needle) };
        }
        if has(Feature::Avx2) && has(Feature::Sse41) {
            // SAFETY: AVX2 was confirmed; loads stay inside `bytes`.
            return unsafe { x86::find_byte_avx2(bytes, needle) };
        }
        if has(Feature::Sse41) {
            // SAFETY: SSE4.1 was confirmed (its `ptest` skips blocks
            // that cannot hold the needle).
            return unsafe { x86::find_byte_sse41(bytes, needle) };
        }
    }
    find_byte_scalar(bytes, needle)
}

/// Reference implementation of [`find_byte`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::find_byte_scalar;
///
/// assert_eq!(find_byte_scalar(b"ab", b'b'), Some(1));
/// ```
#[must_use]
pub fn find_byte_scalar(bytes: &[u8], needle: u8) -> Option<usize> {
    bytes.iter().position(|&byte| byte == needle)
}

/// Returns `bytes` with every occurrence of `value` removed.
///
/// This is the engine's byte-compaction primitive: dropping the gap
/// bytes of a serialized buffer, stripping padding before hashing, and
/// squeezing the separators out of a directory name. The AVX-512
/// VBMI2 tier uses the hardware `vpcompressb` instructions (the
/// `_mm512_maskz_compress_epi8` / `_mm256_maskz_compress_epi8`
/// family); the SSSE3 tier emulates the same compaction with a
/// 256-entry shuffle LUT, and the scalar tier is the reference.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::filter_byte;
///
/// assert_eq!(filter_byte(b"a_b_c", b'_'), b"abc");
/// assert_eq!(filter_byte(b"____", b'_'), b"");
/// ```
#[must_use]
pub fn filter_byte(bytes: &[u8], value: u8) -> Vec<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx512Vbmi2) {
            // SAFETY: AVX-512F/BW/VBMI2 were confirmed; the kernel
            // reads only `bytes` and appends to a fresh `Vec`.
            return unsafe { x86::filter_byte_avx512(bytes, value) };
        }
        if has(Feature::Avx512Vl) && has(Feature::Avx512Vbmi2) {
            // SAFETY: AVX-512VL and VBMI2 were confirmed, enabling the
            // 256-bit `vpcompressb` form.
            return unsafe { x86::filter_byte_avx512vl(bytes, value) };
        }
        if has(Feature::Ssse3) {
            // SAFETY: SSSE3 was confirmed; the kernel only shuffles
            // registers and writes into a `Vec` reserved for the
            // compacted bytes plus one vector of slack.
            return unsafe { x86::filter_byte_ssse3(bytes, value) };
        }
    }
    filter_byte_scalar(bytes, value)
}

/// Reference implementation of [`filter_byte`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::filter_byte_scalar;
///
/// assert_eq!(filter_byte_scalar(b"a-a-a", b'-'), b"aaa");
/// ```
#[must_use]
pub fn filter_byte_scalar(bytes: &[u8], value: u8) -> Vec<u8> {
    bytes
        .iter()
        .copied()
        .filter(|&byte| byte != value)
        .collect()
}

/// Modulus of the polynomial name hash: `2^52`, the width of an
/// AVX-512 IFMA multiply-add.
const HASH_MASK: u64 = (1u64 << 52) - 1;

/// Multiplier of the polynomial name hash (the FNV-1a 64-bit prime,
/// reduced mod `2^52` at use).
const HASH_PRIME: u64 = 0x1_0000_01B3;

/// Polynomial hash of `bytes`, folded to 32 bits.
///
/// The byte stream is read as little-endian 64-bit words (the final
/// word zero-padded) and folded with `h = (h * P + word) mod 2^52`.
/// `2^52` is exactly the width of one AVX-512 IFMA multiply-add, so the
/// acceleration ([`x86::hash_bytes_ifma`]) computes the same value with
/// `_mm512_madd52lo_epu64` while the scalar path is the reference.
///
/// The result is what the directory-name layer stores in each entry:
/// comparing the 32-bit hash first rejects a non-match without touching
/// the name bytes.
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::{hash_bytes, hash_bytes_scalar};
///
/// assert_eq!(hash_bytes(b""), hash_bytes(b""));
/// assert_ne!(hash_bytes(b"src"), hash_bytes(b"lib"));
/// assert_eq!(hash_bytes(b"main.rs"), hash_bytes_scalar(b"main.rs"));
/// ```
#[must_use]
pub fn hash_bytes(bytes: &[u8]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Ifma) {
            // SAFETY: AVX-512F and IFMA were confirmed; the kernel
            // reads only `bytes`.
            return unsafe { x86::hash_bytes_ifma(bytes) };
        }
    }
    hash_bytes_scalar(bytes)
}

/// The full 52-bit state behind [`hash_bytes`], as a `u64` fingerprint.
///
/// [`hash_bytes`] folds this state down to 32 bits; this wider form
/// keeps all of it, so the directory-name layer can reject non-matches
/// with four more bytes of discrimination when it can afford them. Both
/// widths come from the same recurrence, so a 32-bit match is implied
/// by a 64-bit match (they can never disagree about equality).
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::hash_bytes_wide;
///
/// assert_eq!(hash_bytes_wide(b"src"), hash_bytes_wide(b"src"));
/// assert_ne!(hash_bytes_wide(b"src"), hash_bytes_wide(b"lib"));
/// ```
#[must_use]
pub fn hash_bytes_wide(bytes: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx512F) && has(Feature::Avx512Ifma) {
            // SAFETY: AVX-512F and IFMA were confirmed; the kernel
            // reads only `bytes`.
            return unsafe { x86::hash_state_ifma(bytes) };
        }
    }
    hash_state(0, bytes)
}

/// Reference implementation of [`hash_bytes`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::simd::hash_bytes_scalar;
///
/// assert_eq!(hash_bytes_scalar(b"abc"), hash_bytes_scalar(b"abc"));
/// ```
#[must_use]
pub fn hash_bytes_scalar(bytes: &[u8]) -> u32 {
    let hash = hash_state(0, bytes);
    let folded = hash ^ (hash >> 32);
    folded as u32
}

/// Continues the [`hash_bytes`] recurrence from a running `hash`.
///
/// `bytes` is consumed as little-endian 64-bit words (the final word
/// zero-padded); the returned state is still reduced modulo `2^52`.
/// The SIMD tiers call this for their trailing bytes after folding
/// whole blocks, and [`hash_bytes_scalar`] starts it from zero.
#[inline]
fn hash_state(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        hash = hash_step(hash, word_of(chunk));
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        hash = hash_step(hash, word_of(tail));
    }
    hash
}

/// One Horner step of the name hash, reduced mod `2^52`.
#[inline]
fn hash_step(hash: u64, word: u64) -> u64 {
    let product = (u128::from(hash) * u128::from(HASH_PRIME)) & u128::from(HASH_MASK);
    ((product as u64) + word) & HASH_MASK
}

/// Reads up to eight bytes of `chunk` as a little-endian, zero-padded
/// word, reduced mod `2^52`.
#[inline]
fn word_of(chunk: &[u8]) -> u64 {
    let mut buffer = [0u8; 8];
    let take = chunk.len().min(8);
    buffer[..take].copy_from_slice(&chunk[..take]);
    u64::from_le_bytes(buffer) & HASH_MASK
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
    // Walk the cyclic weight instead of taking a modulo per byte; the
    // SIMD tiers leave at most one short tail block here.
    let mut weight = offset % WEIGHT_PERIOD + 1;
    for &byte in bytes {
        sum += u64::from(byte) * weight as u64;
        weight += 1;
        if weight > WEIGHT_PERIOD {
            weight = 1;
        }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use alloc::vec::Vec;

    use super::{
        CHECKSUM_MASK, HASH_MASK, UTF8_CHUNK, WEIGHT_PERIOD, WEIGHT_ROWS, checksum_scalar_at,
        count_byte_scalar, count_scalar, find_byte_scalar, hash_state,
    };
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Powers of [`super::HASH_PRIME`] used to fold eight words at once.
    ///
    /// Lane `j` holds `P^(7 - j) mod 2^52`, so a single `vpmadd52luq`
    /// multiplies word `v_j` by the weight the scalar Horner recurrence
    /// would give it after eight steps (`h' = h*P^8 + Σ v_j*P^(7-j)`).
    pub(super) const LANE_POWERS: [u64; 8] = {
        // lane_powers[j] = P^(7-j) mod 2^52
        let mut lane_powers = [0u64; 8];
        let mut index = 0;
        while index < 8 {
            lane_powers[index] = pow_mod(super::HASH_PRIME, (7 - index) as u32);
            index += 1;
        }
        lane_powers
    };

    /// `P^8 mod 2^52`, the factor that carries the running hash across
    /// one eight-word block.
    pub(super) const HASH_PRIME_POW8: u64 = pow_mod(super::HASH_PRIME, 8);

    /// `base^exp mod 2^52`, evaluated with square-and-multiply in `u128`
    /// so no intermediate step can overflow.
    const fn pow_mod(base: u64, exp: u32) -> u64 {
        let mut result = 1u64;
        let mut factor = base & HASH_MASK;
        let mut power = exp;
        while power > 0 {
            if power & 1 == 1 {
                result = ((result as u128 * factor as u128) & HASH_MASK as u128) as u64;
            }
            factor = ((factor as u128 * factor as u128) & HASH_MASK as u128) as u64;
            power >>= 1;
        }
        result
    }

    /// Builds one byte-compaction shuffle table.
    ///
    /// `table[mask][k]` is the source index of the `k`-th byte to keep
    /// when `mask`'s set bits name the kept lanes of an eight-byte half;
    /// the unused slots are `0x80`, which `pshufb` turns into a zero.
    /// `offset` is 0 for the low half of a register and 8 for the high
    /// half, so the same table drives both lanes of a 128-bit shuffle.
    const fn compact_table(offset: u8) -> [[u8; 8]; 256] {
        let mut table = [[0x80u8; 8]; 256];
        let mut mask = 0usize;
        while mask < 256 {
            let mut slot = 0usize;
            let mut bit = 0usize;
            while bit < 8 {
                if mask & (1 << bit) != 0 {
                    table[mask][slot] = bit as u8 + offset;
                    slot += 1;
                }
                bit += 1;
            }
            mask += 1;
        }
        table
    }

    /// Shuffle table for the low eight bytes of a register.
    pub(super) const COMPACT_LOW: [[u8; 8]; 256] = compact_table(0);

    /// Shuffle table for the high eight bytes of a register.
    pub(super) const COMPACT_HIGH: [[u8; 8]; 256] = compact_table(8);

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

    /// Counts occurrences of `needle` over 16-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires SSE2.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn count_byte_sse2(bytes: &[u8], needle: u8) -> u64 {
        // SAFETY: the caller proved SSE2; `chunks_exact(16)` keeps every
        // load inside `bytes`.
        unsafe {
            let target = _mm_set1_epi8(needle as i8);
            let mut total = 0u64;
            let mut chunks = bytes.chunks_exact(16);
            for chunk in &mut chunks {
                let data = _mm_loadu_si128(chunk.as_ptr().cast());
                let equal = _mm_cmpeq_epi8(data, target);
                total += u64::from((_mm_movemask_epi8(equal) as u32).count_ones());
            }
            total + count_byte_scalar(chunks.remainder(), needle)
        }
    }

    /// Counts occurrences of `needle` over 32-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn count_byte_avx2(bytes: &[u8], needle: u8) -> u64 {
        // SAFETY: the caller proved AVX2; `chunks_exact(32)` keeps every
        // load inside `bytes`.
        unsafe {
            let target = _mm256_set1_epi8(needle as i8);
            let mut total = 0u64;
            let mut chunks = bytes.chunks_exact(32);
            for chunk in &mut chunks {
                let data = _mm256_loadu_si256(chunk.as_ptr().cast());
                let equal = _mm256_cmpeq_epi8(data, target);
                total += u64::from((_mm256_movemask_epi8(equal) as u32).count_ones());
            }
            total + count_byte_scalar(chunks.remainder(), needle)
        }
    }

    /// Counts occurrences of `needle` over 64-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub(super) unsafe fn count_byte_avx512(bytes: &[u8], needle: u8) -> u64 {
        // SAFETY: the caller proved AVX-512F/BW; `chunks_exact(64)` keeps
        // every load inside `bytes`.
        unsafe {
            let target = _mm512_set1_epi8(needle as i8);
            let mut total = 0u64;
            let mut chunks = bytes.chunks_exact(64);
            for chunk in &mut chunks {
                let data = _mm512_loadu_si512(chunk.as_ptr().cast());
                let equal = _mm512_cmpeq_epi8_mask(data, target);
                total += u64::from(equal.count_ones());
            }
            total + count_byte_scalar(chunks.remainder(), needle)
        }
    }

    /// Finds the first `needle` over 16-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires SSE4.1.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn find_byte_sse41(bytes: &[u8], needle: u8) -> Option<usize> {
        // SAFETY: the caller proved SSE4.1; `chunks_exact(16)` keeps every
        // load inside `bytes`. `ptest` answers "is any lane set?" without a
        // `movemask`, and the `movemask` runs only for the matching block.
        unsafe {
            let target = _mm_set1_epi8(needle as i8);
            let mut base = 0usize;
            let mut chunks = bytes.chunks_exact(16);
            for chunk in &mut chunks {
                let data = _mm_loadu_si128(chunk.as_ptr().cast());
                let equal = _mm_cmpeq_epi8(data, target);
                if _mm_testz_si128(equal, equal) == 0 {
                    let bits = _mm_movemask_epi8(equal) as u32;
                    return Some(base + bits.trailing_zeros() as usize);
                }
                base += 16;
            }
            find_byte_scalar(chunks.remainder(), needle).map(|index| base + index)
        }
    }

    /// Finds the first `needle` over 32-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_byte_avx2(bytes: &[u8], needle: u8) -> Option<usize> {
        // SAFETY: the caller proved AVX2; `chunks_exact(32)` keeps every
        // load inside `bytes`.
        unsafe {
            let target = _mm256_set1_epi8(needle as i8);
            let mut base = 0usize;
            let mut chunks = bytes.chunks_exact(32);
            for chunk in &mut chunks {
                let data = _mm256_loadu_si256(chunk.as_ptr().cast());
                let equal = _mm256_cmpeq_epi8(data, target);
                let bits = _mm256_movemask_epi8(equal) as u32;
                if bits != 0 {
                    return Some(base + bits.trailing_zeros() as usize);
                }
                base += 32;
            }
            find_byte_scalar(chunks.remainder(), needle).map(|index| base + index)
        }
    }

    /// Finds the first `needle` over 64-byte blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub(super) unsafe fn find_byte_avx512(bytes: &[u8], needle: u8) -> Option<usize> {
        // SAFETY: the caller proved AVX-512F/BW; `chunks_exact(64)` keeps
        // every load inside `bytes`.
        unsafe {
            let target = _mm512_set1_epi8(needle as i8);
            let mut base = 0usize;
            let mut chunks = bytes.chunks_exact(64);
            for chunk in &mut chunks {
                let data = _mm512_loadu_si512(chunk.as_ptr().cast());
                let mask = _mm512_cmpeq_epi8_mask(data, target);
                if mask != 0 {
                    return Some(base + mask.trailing_zeros() as usize);
                }
                base += 64;
            }
            find_byte_scalar(chunks.remainder(), needle).map(|index| base + index)
        }
    }

    /// Removes every `value` from `bytes` with the SSSE3 shuffle tables.
    ///
    /// Each 16-byte register is split into two eight-byte halves; the
    /// comparison mask selects a shuffle index vector from
    /// [`COMPACT_LOW`]/[`COMPACT_HIGH`], and one `pshufb` squeezes the
    /// kept bytes to the front of each half. The compacted halves are
    /// stored directly at the output cursor (each store writes eight
    /// bytes into reserved slack) instead of being copied through a
    /// stack scratch buffer.
    ///
    /// # Safety
    ///
    /// Requires SSSE3.
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn filter_byte_ssse3(bytes: &[u8], value: u8) -> Vec<u8> {
        // SAFETY: the caller proved SSSE3; `chunks_exact(16)` keeps every
        // load inside `bytes`, the shuffle indices are all within one
        // 128-bit lane, and the output reserves one extra 16-byte vector
        // so the eight-byte stores stay inside the allocation.
        unsafe {
            let target = _mm_set1_epi8(value as i8);
            let mut out = Vec::<u8>::with_capacity(bytes.len() + 16);
            let mut written = 0usize;
            let mut chunks = bytes.chunks_exact(16);
            for chunk in &mut chunks {
                let data = _mm_loadu_si128(chunk.as_ptr().cast());
                let equal = _mm_cmpeq_epi8(data, target);
                let keep = !(_mm_movemask_epi8(equal) as u32) & 0xFFFF;
                let low = (keep & 0xFF) as usize;
                let high = (keep >> 8) as usize;
                let low_index = _mm_loadl_epi64(COMPACT_LOW[low].as_ptr().cast());
                let high_index = _mm_loadl_epi64(COMPACT_HIGH[high].as_ptr().cast());
                let index = _mm_unpacklo_epi64(low_index, high_index);
                let compacted = _mm_shuffle_epi8(data, index);
                let cursor = out.as_mut_ptr().add(written);
                let low_count = low.count_ones() as usize;
                let high_count = high.count_ones() as usize;
                _mm_storel_epi64(cursor.cast(), compacted);
                _mm_storel_epi64(cursor.add(low_count).cast(), _mm_srli_si128(compacted, 8));
                written += low_count + high_count;
            }
            out.set_len(written);
            for &byte in chunks.remainder() {
                if byte != value {
                    out.push(byte);
                }
            }
            out
        }
    }

    /// Removes every `value` from `bytes` with 256-bit `vpcompressb`.
    ///
    /// # Safety
    ///
    /// Requires AVX-512VL, AVX-512BW and AVX-512 VBMI2.
    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx512vl",
        enable = "avx512vbmi2"
    )]
    pub(super) unsafe fn filter_byte_avx512vl(bytes: &[u8], value: u8) -> Vec<u8> {
        // SAFETY: the caller proved AVX-512VL/BW/VBMI2; `chunks_exact(32)`
        // keeps every load inside `bytes`, and `keep` names exactly the
        // lanes the store copies back.
        unsafe {
            let target = _mm256_set1_epi8(value as i8);
            let mut out = Vec::with_capacity(bytes.len());
            let mut scratch = [0u8; 32];
            let mut chunks = bytes.chunks_exact(32);
            for chunk in &mut chunks {
                let data = _mm256_loadu_si256(chunk.as_ptr().cast());
                let keep = !_mm256_cmpeq_epi8_mask(data, target);
                let compacted = _mm256_maskz_compress_epi8(keep, data);
                _mm256_storeu_si256(scratch.as_mut_ptr().cast(), compacted);
                out.extend_from_slice(&scratch[..keep.count_ones() as usize]);
            }
            for &byte in chunks.remainder() {
                if byte != value {
                    out.push(byte);
                }
            }
            out
        }
    }

    /// Removes every `value` from `bytes` with 512-bit `vpcompressb`.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F, AVX-512BW and AVX-512 VBMI2.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi2")]
    pub(super) unsafe fn filter_byte_avx512(bytes: &[u8], value: u8) -> Vec<u8> {
        // SAFETY: the caller proved AVX-512F/BW/VBMI2; `chunks_exact(64)`
        // keeps every load inside `bytes`, and `keep` names exactly the
        // lanes the store copies back.
        unsafe {
            let target = _mm512_set1_epi8(value as i8);
            let mut out = Vec::with_capacity(bytes.len());
            let mut scratch = [0u8; 64];
            let mut chunks = bytes.chunks_exact(64);
            for chunk in &mut chunks {
                let data = _mm512_loadu_si512(chunk.as_ptr().cast());
                let keep = !_mm512_cmpeq_epi8_mask(data, target);
                let compacted = _mm512_maskz_compress_epi8(keep, data);
                _mm512_storeu_si512(scratch.as_mut_ptr().cast(), compacted);
                out.extend_from_slice(&scratch[..keep.count_ones() as usize]);
            }
            for &byte in chunks.remainder() {
                if byte != value {
                    out.push(byte);
                }
            }
            out
        }
    }

    /// Polynomial name hash accelerated with AVX-512 IFMA.
    ///
    /// Eight little-endian words are folded in one `vpmadd52luq`:
    /// `h' = (h*P^8 + Σ v_j*P^(7-j)) mod 2^52`, which is exactly eight
    /// scalar Horner steps because the modulus is a multiply-add width.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F and AVX-512 IFMA.
    #[target_feature(enable = "avx512f", enable = "avx512ifma")]
    pub(super) unsafe fn hash_bytes_ifma(bytes: &[u8]) -> u32 {
        // SAFETY: the caller proved AVX-512F/IFMA.
        let hash = unsafe { hash_state_ifma(bytes) };
        let folded = hash ^ (hash >> 32);
        folded as u32
    }

    /// The unfolded `mod 2^52` name-hash state, accelerated with IFMA.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F and AVX-512 IFMA.
    #[target_feature(enable = "avx512f", enable = "avx512ifma")]
    pub(super) unsafe fn hash_state_ifma(bytes: &[u8]) -> u64 {
        // SAFETY: the caller proved AVX-512F/IFMA; `chunks_exact(64)`
        // keeps every load inside `bytes`, and every intermediate stays
        // below 2^55 so the reduce cannot overflow.
        unsafe {
            let powers = _mm512_loadu_si512(LANE_POWERS.as_ptr().cast());
            let prime8 = _mm512_set1_epi64(HASH_PRIME_POW8 as i64);
            let zero = _mm512_setzero_si512();
            let mut hash = 0u64;
            let mut chunks = bytes.chunks_exact(64);
            for chunk in &mut chunks {
                let words = _mm512_loadu_si512(chunk.as_ptr().cast());
                let products = _mm512_madd52lo_epu64(words, powers, zero);
                let lane_sum = (_mm512_reduce_add_epi64(products) as u64) & HASH_MASK;
                let carried = _mm512_madd52lo_epu64(
                    _mm512_set1_epi64(hash as i64),
                    prime8,
                    _mm512_set1_epi64(lane_sum as i64),
                );
                let mut lane = [0u64; 8];
                _mm512_storeu_si512(lane.as_mut_ptr().cast(), carried);
                hash = lane[0] & HASH_MASK;
            }
            hash_state(hash, chunks.remainder())
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
        use codevar_base::cpuid::{Feature, has};

        for seed in 1..25u32 {
            let bytes = pseudo_random(1 + (seed as usize * 61) % 800, seed);
            if has(Feature::Sse2) {
                // SAFETY: `has` proved SSE2.
                let got = unsafe { x86::count_sse2(&bytes) };
                assert_eq!(got, count_scalar(&bytes), "sse2 seed {seed}");
            }
            if has(Feature::Avx2) && has(Feature::Sse41) {
                // SAFETY: `has` proved AVX2.
                let got = unsafe { x86::count_avx2(&bytes) };
                assert_eq!(got, count_scalar(&bytes), "avx2 seed {seed}");
            }
            if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
                // SAFETY: `has` proved AVX-512F/BW.
                let got = unsafe { x86::count_avx512(&bytes) };
                assert_eq!(got, count_scalar(&bytes), "avx512 seed {seed}");
            }
            if has(Feature::Ssse3) && has(Feature::Sse2) {
                // SAFETY: `has` proved SSSE3.
                let got = unsafe { x86::checksum_ssse3(&bytes) };
                assert_eq!(got, checksum_scalar(&bytes), "ssse3 seed {seed}");
            }
            if has(Feature::Avx2) && has(Feature::Sse41) {
                // SAFETY: `has` proved AVX2.
                let got = unsafe { x86::checksum_avx2(&bytes) };
                assert_eq!(got, checksum_scalar(&bytes), "avx2 seed {seed}");
            }
            if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
                // SAFETY: `has` proved AVX-512F/BW.
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

    #[test]
    fn count_byte_matches_reference_on_random_bytes() {
        for seed in 1..40u32 {
            let bytes = pseudo_random(1 + (seed as usize * 43) % 600, seed);
            for needle in [0u8, 1, 0x61, 0xFF] {
                assert_eq!(
                    count_byte(&bytes, needle),
                    count_byte_scalar(&bytes, needle),
                    "seed {seed} needle {needle}"
                );
            }
        }
    }

    #[test]
    fn find_byte_matches_reference_on_random_bytes() {
        for seed in 1..40u32 {
            let bytes = pseudo_random(1 + (seed as usize * 47) % 600, seed);
            for needle in [0u8, 7, 0x7F, 0xFE] {
                assert_eq!(
                    find_byte(&bytes, needle),
                    find_byte_scalar(&bytes, needle),
                    "seed {seed} needle {needle}"
                );
            }
        }
    }

    #[test]
    fn filter_byte_matches_reference_on_random_bytes() {
        for seed in 1..40u32 {
            let bytes = pseudo_random(1 + (seed as usize * 71) % 900, seed);
            for value in [0u8, 0x20, 0x80, 0xFF] {
                assert_eq!(
                    filter_byte(&bytes, value),
                    filter_byte_scalar(&bytes, value),
                    "seed {seed} value {value}"
                );
            }
        }
        assert_eq!(filter_byte(b"", b'x'), Vec::<u8>::new());
        assert_eq!(filter_byte(b"xxxx", b'x'), Vec::<u8>::new());
        assert_eq!(filter_byte(b"abcd", b'x'), b"abcd".to_vec());
    }

    #[test]
    fn hash_matches_reference_on_random_bytes() {
        assert_eq!(hash_bytes(b""), hash_bytes_scalar(b""));
        for seed in 1..40u32 {
            let bytes = pseudo_random(1 + (seed as usize * 67) % 900, seed);
            assert_eq!(hash_bytes(&bytes), hash_bytes_scalar(&bytes), "seed {seed}");
        }
    }

    #[test]
    fn hash_reads_little_endian_words_including_a_partial_tail() {
        // A single trailing byte exercises the zero-padded final word.
        let mut bytes = vec![0u8; 16];
        for (index, slot) in bytes.iter_mut().enumerate() {
            *slot = index as u8 + 1;
        }
        bytes.truncate(9);
        assert_eq!(hash_bytes(&bytes), hash_bytes_scalar(&bytes));
    }

    /// Every vector tier must agree with the byte-oriented scalar
    /// references, including on lengths that straddle the block size.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn byte_tiers_agree_with_scalar() {
        use codevar_base::cpuid::{Feature, has};

        for seed in 1..20u32 {
            let bytes = pseudo_random(1 + (seed as usize * 91) % 700, seed);
            let needle = bytes[seed as usize % bytes.len()];
            if has(Feature::Sse2) {
                // SAFETY: `has` proved SSE2.
                unsafe {
                    assert_eq!(
                        x86::count_byte_sse2(&bytes, needle),
                        count_byte_scalar(&bytes, needle)
                    );
                }
            }
            if has(Feature::Avx2) {
                // SAFETY: `has` proved AVX2.
                unsafe {
                    assert_eq!(
                        x86::count_byte_avx2(&bytes, needle),
                        count_byte_scalar(&bytes, needle)
                    );
                }
            }
            if has(Feature::Avx512F) && has(Feature::Avx512Bw) {
                // SAFETY: `has` proved AVX-512F/BW.
                unsafe {
                    assert_eq!(
                        x86::count_byte_avx512(&bytes, needle),
                        count_byte_scalar(&bytes, needle)
                    );
                }
            }
            // The needle is present somewhere, so the find paths cannot
            // return `None`; also probe a byte that cannot be present.
            if has(Feature::Sse41) {
                // SAFETY: `has` proved SSE4.1.
                unsafe {
                    assert_eq!(
                        x86::find_byte_sse41(&bytes, needle),
                        find_byte_scalar(&bytes, needle)
                    );
                }
            }
            if has(Feature::Avx2) {
                // SAFETY: `has` proved AVX2.
                unsafe {
                    assert_eq!(
                        x86::find_byte_avx2(&bytes, needle),
                        find_byte_scalar(&bytes, needle)
                    );
                }
            }
            if has(Feature::Avx512F) && has(Feature::Avx512Bw) {
                // SAFETY: `has` proved AVX-512F/BW.
                unsafe {
                    assert_eq!(
                        x86::find_byte_avx512(&bytes, needle),
                        find_byte_scalar(&bytes, needle)
                    );
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn compaction_tiers_agree_with_scalar() {
        use codevar_base::cpuid::{Feature, has};

        for seed in 1..20u32 {
            let bytes = pseudo_random(1 + (seed as usize * 89) % 700, seed);
            let value = bytes[seed as usize % bytes.len()];
            if has(Feature::Ssse3) {
                // SAFETY: `has` proved SSSE3.
                unsafe {
                    assert_eq!(
                        x86::filter_byte_ssse3(&bytes, value),
                        filter_byte_scalar(&bytes, value)
                    );
                }
            }
            if has(Feature::Avx512Vl) && has(Feature::Avx512Bw) && has(Feature::Avx512Vbmi2) {
                // SAFETY: `has` proved AVX-512VL/BW/VBMI2.
                unsafe {
                    assert_eq!(
                        x86::filter_byte_avx512vl(&bytes, value),
                        filter_byte_scalar(&bytes, value)
                    );
                }
            }
            if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx512Vbmi2) {
                // SAFETY: `has` proved AVX-512F/BW/VBMI2.
                unsafe {
                    assert_eq!(
                        x86::filter_byte_avx512(&bytes, value),
                        filter_byte_scalar(&bytes, value)
                    );
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn ifma_hash_agrees_with_scalar() {
        use codevar_base::cpuid::{Feature, has};
        if !(has(Feature::Avx512F) && has(Feature::Avx512Ifma)) {
            return;
        }
        for seed in 1..30u32 {
            let bytes = pseudo_random(1 + (seed as usize * 83) % 900, seed);
            // SAFETY: `has` proved AVX-512F/IFMA.
            let got = unsafe { x86::hash_bytes_ifma(&bytes) };
            assert_eq!(got, hash_bytes_scalar(&bytes), "seed {seed}");
        }
    }

    /// The shuffle tables must keep the selected lanes in order and pad
    /// every unused slot with `0x80` (which `pshufb` zeroes).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn compaction_tables_are_sound() {
        for mask in 0usize..256 {
            for (offset, table) in [(0u8, &x86::COMPACT_LOW), (8u8, &x86::COMPACT_HIGH)] {
                let row = table[mask];
                let mut expected = 0usize;
                for bit in 0..8usize {
                    if mask & (1 << bit) != 0 {
                        assert_eq!(row[expected], bit as u8 + offset, "mask {mask}");
                        expected += 1;
                    }
                }
                for &pad in &row[expected..] {
                    assert_eq!(pad, 0x80, "mask {mask} padding");
                }
            }
        }
    }

    /// The IFMA lane weights must reproduce eight plain Horner steps.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn ifma_lane_weights_match_repeated_horner() {
        for lane in 0..8usize {
            let mut expected = 1u64;
            for _ in 0..(7 - lane) {
                expected = hash_step(expected, 0);
            }
            assert_eq!(x86::LANE_POWERS[lane], expected, "lane {lane}");
        }
        let mut prime8 = 1u64;
        for _ in 0..8 {
            prime8 = hash_step(prime8, 0);
        }
        assert_eq!(x86::HASH_PRIME_POW8, prime8);
    }
}
