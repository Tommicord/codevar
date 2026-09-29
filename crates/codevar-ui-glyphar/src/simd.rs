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

//! Runtime-dispatched SIMD kernels for the Glyphar engine.
//!
//! Every public entry point selects the widest kernel the running CPU
//! supports through [`codevar_base::basic_cpuid`], and always falls
//! back to an identical scalar implementation.
//!
//! | Target | Tiers (best first) | Selection |
//! |--------|--------------------|-----------|
//! | `x86` / `x86_64` | AVX-512 (F+BW+DQ+VL), AVX2, SSSE3, SSE2 | run time |
//! | `aarch64` | NEON (Advanced SIMD) | run time |
//! | anything else | scalar | — |
//!
//! # Kernels
//!
//! | Kernel | Used by | Work per vector iteration |
//! |--------|---------|---------------------------|
//! | [`table_checksum`] | [`crate::font_file::FontFile::verify_checksums`] | 16/32/64 bytes → byte-swapped `u32` sum |
//! | [`find_end_code`] | [`crate::cmap`] format 4 | 8/16/32 ordered `u16` compares |
//! | [`finalize_coverage`] | [`crate::raster`] | 4/8/16 `f32` → `u8` coverage |
//! | [`blend_coverage`] | [`crate::atlas`] | 16/32/64 `u8` source-over blends |
//! | [`coords_to_f32`] | [`crate::glyf`] | 8/16 `i16` → `f32` scale+offset |
//! | [`find_tag`] | [`crate::font_file`] | scalar binary search + linear fallback |
//!
//! Every vector kernel processes whole blocks and hands the trailing
//! `0..block` elements to the shared scalar tail, so all paths produce
//! byte-identical output for every input.
//!
//! # Exactness of the blend
//!
//! The source-over blend `floor((s*a + d*(255-a)) / 255)` is evaluated
//! as `t = a*s + (255-a)*d` (always `<= 65025`, so it fits an `u16`
//! lane) followed by `floor(t / 255) = mulhi(t, 0x8081) >> 7`, which is
//! exact for `t <= 66051`. No rounding error is ever observable, in
//! either path.
//!
//! [`backend_name`] reports the tier the dispatchers pick on the
//! machine running the program.

use crate::font_file::TableRecord;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use codevar_base::basic_cpuid::{self, Feature};

/// sfnt table checksum: wrapping sum of big-endian `u32` words with the
/// final partial word zero-padded (OpenType spec, "Checksum
/// Adjustment").
///
/// # Performance
///
/// O(n) with one vector load per 16/32/64 bytes; see the [module
/// documentation](self) for tier selection.
#[must_use]
pub fn table_checksum(data: &[u8]) -> u32 {
    dispatch_checksum(data)
}

/// Finds `tag` in a table directory, returning its index (or
/// `records.len()` when absent).
///
/// The sfnt spec requires directory records to be sorted by tag, so a
/// binary search runs first; a linear fallback covers fonts that ship
/// unsorted directories.
#[must_use]
pub fn find_tag(records: &[TableRecord], tag: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = records.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let current = records.get(mid).map_or(0, |r| r.tag);
        if current < tag {
            lo = mid + 1;
        } else if current > tag {
            hi = mid;
        } else {
            return mid;
        }
    }
    records.iter().position(|record| record.tag == tag).unwrap_or(records.len())
}

/// Returns the first index `i` with `end_codes[i] >= code`.
///
/// `end_codes` must be sorted ascending, as required by cmap format 4.
/// The terminator segment `0xFFFF` guarantees a match for every
/// `code <= 0xFFFF`.
///
/// # Performance
///
/// The scalar path is a binary search (O(log n) dependent loads); the
/// vector paths scan 8/16/32 candidates per iteration with ordered
/// `u16` compares, which wins for the sub-1000 segment tables that
/// fonts actually ship.
#[must_use]
pub fn find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
    dispatch_find_end_code(end_codes, code)
}

/// Converts an accumulated floating-point coverage row into `u8`
/// coverage bytes: `out[i] = min(|acc[i]|, 1.0) * 255`, truncated.
///
/// Inputs are expected to be finite; `NaN` handling is unspecified
/// because no accumulation path in [`crate::raster`] produces it.
///
/// # Arguments
///
/// * `acc` — accumulated signed area per pixel (non-zero winding).
/// * `out` — destination; only `min(acc.len(), out.len())` bytes are written.
#[inline]
pub fn finalize_coverage(acc: &[f32], out: &mut [u8]) {
    dispatch_finalize(acc, out);
}

/// Source-over blends `src` coverage onto `dst` with coverage alpha:
/// `dst[i] = floor((src[i] * alpha + dst[i] * (255 - alpha)) / 255)`.
///
/// The division is exact (never off by one) in both paths; see the
/// [module documentation](self#exactness-of-the-blend).
///
/// # Arguments
///
/// * `dst` — destination coverage (also the backdrop).
/// * `src` — source coverage; only `min(dst.len(), src.len())` bytes are blended.
/// * `alpha` — overall opacity of `src`, `0..=255`.
#[inline]
pub fn blend_coverage(dst: &mut [u8], src: &[u8], alpha: u8) {
    dispatch_blend(dst, src, alpha);
}

/// Scales font-unit coordinates into floating point:
/// `dst[i] = src[i] * scale + offset`.
///
/// # Arguments
///
/// * `src` — 16-bit font-unit coordinates.
/// * `dst` — destination; only `min(src.len(), dst.len())` entries are written.
/// * `scale` — multiplier (typically `ppem * 64 / units_per_em` in F26Dot6).
/// * `offset` — additive translation applied to every coordinate.
#[inline]
pub fn coords_to_f32(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) {
    dispatch_coords(src, dst, scale, offset);
}

/// Widest instruction-set tier the running CPU (and OS) supports.
///
/// A tier is only reported when [`codevar_base::basic_cpuid`] has
/// verified both the instruction set and the OS-enabled register state
/// (XGETBV for YMM/ZMM), which is exactly what the `#[target_feature]`
/// kernels require to be called safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    /// AVX-512 F+BW+DQ+VL with OS ZMM state enabled.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
    /// AVX2 with OS YMM state enabled.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    /// SSSE3 (accelerates the checksum only).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Ssse3,
    /// Baseline SSE2 (always present on `x86_64`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    /// Advanced SIMD.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// No usable vector kernel.
    Scalar,
}

/// Probes the CPU (a cached atomic load plus a union after the first call).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn current_tier() -> Tier {
    if basic_cpuid::has(Feature::Avx512) {
        Tier::Avx512
    } else if basic_cpuid::has(Feature::Avx2) {
        Tier::Avx2
    } else if basic_cpuid::has(Feature::Ssse3) {
        Tier::Ssse3
    } else if basic_cpuid::has(Feature::Sse2) {
        Tier::Sse2
    } else {
        Tier::Scalar
    }
}

/// Probes the CPU (a cached atomic load plus a union after the first call).
#[cfg(target_arch = "aarch64")]
fn current_tier() -> Tier {
    if basic_cpuid::has(Feature::Neon) {
        Tier::Neon
    } else {
        Tier::Scalar
    }
}

/// Targets without a vector tier stay on the scalar kernels.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn current_tier() -> Tier {
    Tier::Scalar
}

/// Picks the widest checksum kernel available for this CPU.
fn dispatch_checksum(data: &[u8]) -> u32 {
    match current_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx512 => {
            // SAFETY: `Tier::Avx512` is only produced after `has(Avx512)`
            // verified F+BW+DQ+VL plus OS ZMM state; the kernel bounds
            // checks every load against `data`.
            unsafe { avx512::checksum(data) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx2 => {
            // SAFETY: `Tier::Avx2` is only produced after `has(Avx2)`
            // verified the instruction set and OS YMM state.
            unsafe { avx2::checksum(data) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Ssse3 => {
            // SAFETY: `Tier::Ssse3` is only produced after `has(Ssse3)`.
            unsafe { ssse3::checksum(data) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Sse2 => {
            // SAFETY: `Tier::Sse2` is only produced after `has(Sse2)`.
            unsafe { sse2::checksum(data) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `Tier::Neon` is only produced after `has(Neon)`.
            unsafe { neon::checksum(data) }
        }
        Tier::Scalar => scalar::checksum(data),
    }
}

/// Picks the widest cmap segment scan available for this CPU.
fn dispatch_find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
    let vector = match current_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx512 => {
            // SAFETY: `Tier::Avx512` proved the vector tier; every load
            // is bounds-checked against `end_codes.len()`.
            unsafe { avx512::find_end_code(end_codes, code) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx2 | Tier::Ssse3 | Tier::Sse2 => {
            // SAFETY: all three tiers imply SSE2, and the kernel
            // bounds-checks every load.
            unsafe { sse2::find_end_code(end_codes, code) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `Tier::Neon` proved Advanced SIMD support.
            unsafe { neon::find_end_code(end_codes, code) }
        }
        Tier::Scalar => None,
    };
    // The vector scan only covers whole 8/16/32-element blocks, so a
    // miss means "no match inside the scanned prefix"; the scalar
    // binary search then answers for the whole (sorted) array.
    vector.or_else(|| scalar::find_end_code(end_codes, code))
}

/// Picks the widest coverage finalizer available for this CPU.
fn dispatch_finalize(acc: &[f32], out: &mut [u8]) {
    let n = acc.len().min(out.len());
    let (acc, out) = (&acc[..n], &mut out[..n]);
    let consumed = match current_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx512 => {
            // SAFETY: `Tier::Avx512` proved the vector tier; both slices
            // are clipped to the same length `n` above.
            unsafe { avx512::finalize(acc, out) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx2 => {
            // SAFETY: `Tier::Avx2` proved AVX2 + OS YMM state.
            unsafe { avx2::finalize(acc, out) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Ssse3 | Tier::Sse2 => {
            // SAFETY: both tiers imply SSE2; the slices share length `n`.
            unsafe { sse2::finalize(acc, out) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `Tier::Neon` proved Advanced SIMD support.
            unsafe { neon::finalize(acc, out) }
        }
        Tier::Scalar => 0,
    };
    scalar::finalize(&acc[consumed..], &mut out[consumed..]);
}

/// Picks the widest source-over blend available for this CPU.
fn dispatch_blend(dst: &mut [u8], src: &[u8], alpha: u8) {
    let n = dst.len().min(src.len());
    let (dst, src) = (&mut dst[..n], &src[..n]);
    let consumed = match current_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx512 => {
            // SAFETY: `Tier::Avx512` proved the vector tier; `dst` and
            // `src` share the clipped length `n`.
            unsafe { avx512::blend(dst, src, alpha) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx2 => {
            // SAFETY: `Tier::Avx2` proved AVX2 + OS YMM state.
            unsafe { avx2::blend(dst, src, alpha) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Ssse3 | Tier::Sse2 => {
            // SAFETY: both tiers imply SSE2; the slices share length `n`.
            unsafe { sse2::blend(dst, src, alpha) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `Tier::Neon` proved Advanced SIMD support.
            unsafe { neon::blend(dst, src, alpha) }
        }
        Tier::Scalar => 0,
    };
    scalar::blend(&mut dst[consumed..], &src[consumed..], alpha);
}

/// Picks the widest coordinate converter available for this CPU.
fn dispatch_coords(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) {
    let n = src.len().min(dst.len());
    let (src, dst) = (&src[..n], &mut dst[..n]);
    let consumed = match current_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx512 => {
            // SAFETY: `Tier::Avx512` proved the vector tier; the slices
            // are clipped to the same length `n` above.
            unsafe { avx512::coords(src, dst, scale, offset) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Avx2 => {
            // SAFETY: `Tier::Avx2` proved AVX2 + OS YMM state.
            unsafe { avx2::coords(src, dst, scale, offset) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Tier::Ssse3 | Tier::Sse2 => {
            // SAFETY: both tiers imply SSE2; the slices share length `n`.
            unsafe { sse2::coords(src, dst, scale, offset) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: `Tier::Neon` proved Advanced SIMD support.
            unsafe { neon::coords(src, dst, scale, offset) }
        }
        Tier::Scalar => 0,
    };
    scalar::coords(&src[consumed..], &mut dst[consumed..], scale, offset);
}

/// Portable kernels; every vector kernel must match these byte for byte.
pub(crate) mod scalar {
    /// See [`super::table_checksum`].
    #[must_use]
    pub fn checksum(data: &[u8]) -> u32 {
        let mut sum = 0u32;
        let mut chunks = data.chunks_exact(4);
        for chunk in &mut chunks {
            sum = sum.wrapping_add(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 4];
            buf[..rem.len()].copy_from_slice(rem);
            sum = sum.wrapping_add(u32::from_be_bytes(buf));
        }
        sum
    }

    /// See [`super::find_end_code`].
    #[must_use]
    pub fn find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
        let mut lo = 0usize;
        let mut hi = end_codes.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let end = *end_codes.get(mid)?;
            if end < code {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < end_codes.len()).then_some(lo)
    }

    /// See [`super::finalize_coverage`].
    pub fn finalize(acc: &[f32], out: &mut [u8]) {
        for (a, o) in acc.iter().zip(out.iter_mut()) {
            let v = a.abs().min(1.0) * 255.0;
            *o = if v <= 0.0 {
                0
            } else if v >= 255.0 {
                255
            } else {
                v as u8
            };
        }
    }

    /// Exact source-over blend: `floor((s*a + d*(255-a)) / 255)`.
    ///
    /// Because `s*a + d*(255-a) = 255*d + a*(s-d)`, the quotient
    /// simplifies to `d + floor(a*(s-d) / 255)`, which stays inside
    /// `0..=255` for every `u8` input. The vector kernels evaluate the
    /// unreduced form (`a*s + (255-a)*d <= 65025`, always inside an
    /// `u16` lane) and divide with `mulhi(t, 0x8081) >> 7`.
    pub fn blend(dst: &mut [u8], src: &[u8], alpha: u8) {
        let a = i32::from(alpha);
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            let d0 = i32::from(*d);
            let s0 = i32::from(*s);
            let t = a * (s0 - d0);
            let q = t.div_euclid(255);
            *d = (d0 + q).clamp(0, 255) as u8;
        }
    }

    /// See [`super::coords_to_f32`].
    pub fn coords(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = f32::from(*s) * scale + offset;
        }
    }
}

/// SSE2 tier: baseline on `x86_64`, probed on 32-bit `x86`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm_add_epi16, _mm_add_epi32, _mm_add_ps, _mm_and_si128, _mm_castps_si128,
        _mm_castsi128_ps, _mm_cmpgt_epi16, _mm_cvtepi32_ps, _mm_cvtsi128_si32, _mm_cvttps_epi32,
        _mm_loadu_ps, _mm_loadu_si128, _mm_min_ps, _mm_movemask_epi8, _mm_mul_ps, _mm_mulhi_epu16,
        _mm_mullo_epi16, _mm_or_si128, _mm_packus_epi16, _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32,
        _mm_set1_ps, _mm_setzero_si128, _mm_slli_epi32, _mm_srai_epi16, _mm_srli_epi16, _mm_srli_epi32,
        _mm_srli_si128, _mm_storeu_ps, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi8,
        _mm_unpacklo_epi16, _mm_unpacklo_epi64, _mm_unpacklo_epi8, _mm_xor_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_add_epi32, _mm_add_ps, _mm_and_si128, _mm_castps_si128,
        _mm_castsi128_ps, _mm_cmpgt_epi16, _mm_cvtepi32_ps, _mm_cvtsi128_si32, _mm_cvttps_epi32,
        _mm_loadu_ps, _mm_loadu_si128, _mm_min_ps, _mm_movemask_epi8, _mm_mul_ps, _mm_mulhi_epu16,
        _mm_mullo_epi16, _mm_or_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32,
        _mm_set1_ps, _mm_setzero_si128, _mm_slli_epi32, _mm_srai_epi16, _mm_srli_epi16, _mm_srli_epi32,
        _mm_srli_si128, _mm_storeu_ps, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi8,
        _mm_unpacklo_epi16, _mm_unpacklo_epi64, _mm_unpacklo_epi8, _mm_xor_si128,
    };

    /// Input bytes consumed per checksum iteration.
    const BLOCK: usize = 16;

    /// SSE2 checksum over 16-byte blocks plus the scalar tail.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn checksum(data: &[u8]) -> u32 {
        // SAFETY: `target_feature` guarantees the intrinsics; the loop
        // condition keeps every load inside `data`.
        unsafe {
            let mut acc = _mm_setzero_si128();
            let mut i = 0;
            while i + BLOCK <= data.len() {
                let v = _mm_loadu_si128(data.as_ptr().add(i).cast::<__m128i>());
                acc = _mm_add_epi32(acc, bswap_epi32(v));
                i += BLOCK;
            }
            hsum_epi32(acc).wrapping_add(super::scalar::checksum(&data[i..]))
        }
    }

    /// Byte-swaps each of the four `u32` lanes (big-endian load).
    ///
    /// # Safety
    ///
    /// Requires SSE2.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn bswap_epi32(v: __m128i) -> __m128i {
        // SAFETY: register-only SSE2 operations.
        let mid = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(v, 8), _mm_set1_epi32(0x0000_FF00)),
            _mm_and_si128(_mm_slli_epi32(v, 8), _mm_set1_epi32(0x00FF_0000)),
        );
        let ends = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(v, 24), _mm_set1_epi32(0x0000_00FF)),
            _mm_and_si128(_mm_slli_epi32(v, 24), _mm_set1_epi32(0xFF00_0000)),
        );
        _mm_or_si128(mid, ends)
    }

    /// Wrapping horizontal sum of the four `u32` lanes.
    ///
    /// # Safety
    ///
    /// Requires SSE2.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn hsum_epi32(v: __m128i) -> u32 {
        // SAFETY: register-only SSE2 operations: duplicate the high
        // 64 bits, add, then duplicate the high 32 bits and add again.
        let hi64 = _mm_unpacklo_epi64(v, v);
        let sum2 = _mm_add_epi32(v, hi64);
        let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 8));
        _mm_cvtsi128_si32(sum1) as u32
    }

    /// SSE2 ordered `u16` scan of one 8-element block; `None` when
    /// every element of the block is `< code` (the scalar tail then
    /// finishes the search).
    ///
    /// # Safety
    ///
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
        let bias = 0x8000u16;
        let target = (code ^ bias) as i16;
        let n = end_codes.len();
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n` keeps the 16-byte load in bounds.
            unsafe {
                let v = _mm_loadu_si128(end_codes.as_ptr().add(i).cast::<__m128i>());
                let biased = _mm_xor_si128(v, _mm_set1_epi16(bias as i16));
                // `x ^ 0x8000` maps unsigned order onto signed order, so
                // a signed compare of the biased values answers the
                // unsigned question "end < code".
                let less = _mm_cmpgt_epi16(_mm_set1_epi16(target), biased);
                // `movemask_epi8` yields two bits per `u16` lane; keep
                // one bit per lane so a lane index is `bit / 2`.
                let mask = _mm_movemask_epi8(less) as u32;
                let lanes = (mask >> 1) & 0x5555;
                if lanes != 0x5555 {
                    let first = (!lanes) & 0x5555;
                    return Some(i + first.trailing_zeros() as usize / 2);
                }
                i += 8;
            }
        }
        None
    }

    /// SSE2 coverage finalization; returns the number of `f32`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE2; `acc` and `out` are clipped to equal
    /// length by the dispatcher.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn finalize(acc: &[f32], out: &mut [u8]) -> usize {
        let n = acc.len().min(out.len());
        let mut i = 0;
        while i + 4 <= n {
            // SAFETY: `i + 4 <= n` bounds the load; the four result
            // bytes are written through a checked slice.
            unsafe {
                let v = _mm_loadu_ps(acc.as_ptr().add(i));
                let abs = _mm_and_si128(_mm_castps_si128(v), _mm_set1_epi32(0x7FFF_FFFF));
                let clamped = _mm_min_ps(_mm_castsi128_ps(abs), _mm_set1_ps(1.0));
                let scaled = _mm_mul_ps(clamped, _mm_set1_ps(255.0));
                let ints = _mm_cvttps_epi32(scaled);
                let words = _mm_packs_epi32(ints, ints);
                let bytes = _mm_packus_epi16(words, words);
                let word = _mm_cvtsi128_si32(bytes) as u32;
                out[i..i + 4].copy_from_slice(&word.to_ne_bytes());
                i += 4;
            }
        }
        i
    }

    /// SSE2 exact source-over blend; returns bytes consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE2; `dst`/`src` are clipped to equal
    /// length by the dispatcher.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn blend(dst: &mut [u8], src: &[u8], alpha: u8) -> usize {
        let n = dst.len().min(src.len());
        let a = _mm_set1_epi16(i16::from(alpha));
        let inv = _mm_set1_epi16(255 - i16::from(alpha));
        let magic = _mm_set1_epi16(0x8081_u16 as i16);
        let zero = _mm_setzero_si128();
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n` keeps both 16-byte loads and the
            // store inside their slices.
            unsafe {
                let s = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                let d = _mm_loadu_si128(dst.as_ptr().add(i).cast::<__m128i>());
                let lo =
                    blend_u16x8(_mm_unpacklo_epi8(s, zero), _mm_unpacklo_epi8(d, zero), a, inv, magic);
                let hi =
                    blend_u16x8(_mm_unpackhi_epi8(s, zero), _mm_unpackhi_epi8(d, zero), a, inv, magic);
                let bytes = _mm_packus_epi16(lo, hi);
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), bytes);
                i += 16;
            }
        }
        i
    }

    /// One 8-lane blend step over zero-extended `u16` coverages.
    ///
    /// `t = a*s + (255-a)*d <= 65025` always fits an `u16` lane, and
    /// `floor(t / 255) = (mulhi(t, 0x8081) >> 7)` is exact for
    /// `t <= 66051`.
    ///
    /// # Safety
    ///
    /// Requires SSE2.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn blend_u16x8(s: __m128i, d: __m128i, a: __m128i, inv: __m128i, magic: __m128i) -> __m128i {
        // SAFETY: register-only SSE2 operations; every lane stays in
        // `0..=255`, so the saturating pack never clamps.
        let t = _mm_add_epi16(_mm_mullo_epi16(a, s), _mm_mullo_epi16(inv, d));
        let q = _mm_srli_epi16(_mm_mulhi_epu16(t, magic), 7);
        _mm_add_epi16(d, q)
    }

    /// SSE2 `i16` → `f32` scale+offset; returns `i16`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE2; slices are clipped to equal length.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn coords(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) -> usize {
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n` bounds the 16-byte load and both
            // 16-byte stores.
            unsafe {
                let v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                // Arithmetic shift sign-extends the 16-bit lanes into the
                // adjacent halfwords before the widening unpack.
                let ext = _mm_srai_epi16(v, 16);
                let s = _mm_set1_ps(scale);
                let o = _mm_set1_ps(offset);
                let lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v, ext));
                let hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, ext));
                _mm_storeu_ps(dst.as_mut_ptr().add(i), _mm_add_ps(_mm_mul_ps(lo, s), o));
                _mm_storeu_ps(dst.as_mut_ptr().add(i + 4), _mm_add_ps(_mm_mul_ps(hi, s), o));
                i += 8;
            }
        }
        i
    }
}

/// SSSE3 tier: byte-swap via `pshufb` instead of shift/or chains.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod ssse3 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm_add_epi32, _mm_loadu_si128, _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_loadu_si128, _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8,
    };

    /// Input bytes consumed per iteration.
    const BLOCK: usize = 16;

    /// SSSE3 checksum over 16-byte blocks plus the scalar tail.
    ///
    /// # Safety
    ///
    /// The CPU must support SSSE3.
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn checksum(data: &[u8]) -> u32 {
        // SAFETY: `target_feature` guarantees `pshufb`; the loop keeps
        // every access inside `data`.
        unsafe {
            let swap = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            let mut acc = _mm_setzero_si128();
            let mut i = 0;
            while i + BLOCK <= data.len() {
                let v = _mm_loadu_si128(data.as_ptr().add(i).cast::<__m128i>());
                acc = _mm_add_epi32(acc, _mm_shuffle_epi8(v, swap));
                i += BLOCK;
            }
            super::sse2::hsum_epi32(acc).wrapping_add(super::scalar::checksum(&data[i..]))
        }
    }
}

/// AVX2 tier: 32-byte blocks for every kernel.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
        _mm256_castps_si256, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmpgt_epi16,
        _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_cvttps_epi32, _mm256_extracti128_si256,
        _mm256_loadu_ps, _mm256_loadu_si256, _mm256_min_ps, _mm256_movemask_epi8, _mm256_mul_ps,
        _mm256_mulhi_epu16, _mm256_mullo_epi16, _mm256_or_si256, _mm256_packus_epi16, _mm256_set1_epi16,
        _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srli_epi16,
        _mm256_srli_epi32, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_unpackhi_epi8,
        _mm256_unpacklo_epi8, _mm256_xor_si256, _mm_add_epi32, _mm_loadu_si128, _mm_packus_epi16,
        _mm_packs_epi32, _mm_storeu_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
        _mm256_castps_si256, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmpgt_epi16,
        _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_cvttps_epi32, _mm256_extracti128_si256,
        _mm256_loadu_ps, _mm256_loadu_si256, _mm256_min_ps, _mm256_movemask_epi8, _mm256_mul_ps,
        _mm256_mulhi_epu16, _mm256_mullo_epi16, _mm256_or_si256, _mm256_packus_epi16, _mm256_set1_epi16,
        _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srli_epi16,
        _mm256_srli_epi32, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_unpackhi_epi8,
        _mm256_unpacklo_epi8, _mm256_xor_si256, _mm_add_epi32, _mm_loadu_si128,
        _mm_packs_epi32, _mm_packus_epi16, _mm_storeu_si128,
    };

    /// Input bytes consumed per iteration.
    const BLOCK: usize = 32;

    /// AVX2 checksum over 32-byte blocks plus the scalar tail.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and OS YMM state.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn checksum(data: &[u8]) -> u32 {
        // SAFETY: `target_feature` guarantees the intrinsics; the loop
        // condition keeps every load inside `data`.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let mut i = 0;
            while i + BLOCK <= data.len() {
                let v = _mm256_loadu_si256(data.as_ptr().add(i).cast::<__m256i>());
                acc = _mm256_add_epi32(acc, bswap_epi32(v));
                i += BLOCK;
            }
            let merged = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            super::sse2::hsum_epi32(merged).wrapping_add(super::scalar::checksum(&data[i..]))
        }
    }

    /// Byte-swaps each of the eight `u32` lanes.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn bswap_epi32(v: __m256i) -> __m256i {
        // SAFETY: register-only AVX2 operations.
        let mid = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(v, 8), _mm256_set1_epi32(0x0000_FF00)),
            _mm256_and_si256(_mm256_slli_epi32(v, 8), _mm256_set1_epi32(0x00FF_0000)),
        );
        let ends = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(v, 24), _mm256_set1_epi32(0x0000_00FF)),
            _mm256_and_si256(_mm256_slli_epi32(v, 24), _mm256_set1_epi32(0xFF00_0000)),
        );
        _mm256_or_si256(mid, ends)
    }

    /// AVX2 ordered `u16` scan of one 16-element block; `None` when
    /// every element of the block is `< code`.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
        let bias = 0x8000u16;
        let target = (code ^ bias) as i16;
        let n = end_codes.len();
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n` keeps the 32-byte load in bounds.
            unsafe {
                let v = _mm256_loadu_si256(end_codes.as_ptr().add(i).cast::<__m256i>());
                let biased = _mm256_xor_si256(v, _mm256_set1_epi16(bias as i16));
                let less = _mm256_cmpgt_epi16(_mm256_set1_epi16(target), biased);
                let mask = _mm256_movemask_epi8(less) as u32;
                let lanes = (mask >> 1) & 0x5555_5555;
                if lanes != 0x5555_5555 {
                    let first = (!lanes) & 0x5555_5555;
                    return Some(i + first.trailing_zeros() as usize / 2);
                }
                i += 16;
            }
        }
        None
    }

    /// AVX2 coverage finalization; returns the number of `f32`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2; slices are clipped to equal length.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn finalize(acc: &[f32], out: &mut [u8]) -> usize {
        let n = acc.len().min(out.len());
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n` bounds the 32-byte load; only the
            // first eight result bytes are written to `out`.
            unsafe {
                let v = _mm256_loadu_ps(acc.as_ptr().add(i));
                let abs = _mm256_and_si256(_mm256_castps_si256(v), _mm256_set1_epi32(0x7FFF_FFFF));
                let clamped = _mm256_min_ps(_mm256_castsi256_ps(abs), _mm256_set1_ps(1.0));
                let scaled = _mm256_mul_ps(clamped, _mm256_set1_ps(255.0));
                let ints = _mm256_cvttps_epi32(scaled);
                let lo = _mm256_castsi256_si128(ints);
                let hi = _mm256_extracti128_si256(ints, 1);
                // `packs_epi32` puts lanes 0..3 then 4..7 into the low
                // eight bytes; `packus_epi16` then yields the eight
                // result bytes in order.
                let words = _mm_packs_epi32(lo, hi);
                let bytes = _mm_packus_epi16(words, words);
                let mut buf = [0u8; 16];
                _mm_storeu_si128(buf.as_mut_ptr().cast::<__m128i>(), bytes);
                out[i..i + 8].copy_from_slice(&buf[..8]);
                i += 8;
            }
        }
        i
    }

    /// AVX2 exact source-over blend; returns bytes consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2; `dst`/`src` are clipped to equal
    /// length by the dispatcher.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn blend(dst: &mut [u8], src: &[u8], alpha: u8) -> usize {
        let n = dst.len().min(src.len());
        let a = _mm256_set1_epi16(i16::from(alpha));
        let inv = _mm256_set1_epi16(255 - i16::from(alpha));
        let magic = _mm256_set1_epi16(0x8081_u16 as i16);
        let zero = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            // SAFETY: `i + 32 <= n` keeps both 32-byte loads and the
            // store inside their slices.
            unsafe {
                let s = _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>());
                let d = _mm256_loadu_si256(dst.as_ptr().add(i).cast::<__m256i>());
                let lo = blend_u16x16(
                    _mm256_unpacklo_epi8(s, zero),
                    _mm256_unpacklo_epi8(d, zero),
                    a,
                    inv,
                    magic,
                );
                let hi = blend_u16x16(
                    _mm256_unpackhi_epi8(s, zero),
                    _mm256_unpackhi_epi8(d, zero),
                    a,
                    inv,
                    magic,
                );
                // Per-128-bit-lane pack: lane 0 carries result bytes
                // 0..15, lane 1 carries bytes 16..31, in order.
                let bytes = _mm256_packus_epi16(lo, hi);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), bytes);
                i += 32;
            }
        }
        i
    }

    /// One 16-lane blend step; see the SSE2 kernel of the same name.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn blend_u16x16(
        s: __m256i,
        d: __m256i,
        a: __m256i,
        inv: __m256i,
        magic: __m256i,
    ) -> __m256i {
        // SAFETY: register-only AVX2 operations.
        let t = _mm256_add_epi16(_mm256_mullo_epi16(a, s), _mm256_mullo_epi16(inv, d));
        let q = _mm256_srli_epi16(_mm256_mulhi_epu16(t, magic), 7);
        _mm256_add_epi16(d, q)
    }

    /// AVX2 `i16` → `f32` scale+offset; returns `i16`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2; slices are clipped to equal length.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn coords(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) -> usize {
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n` bounds both 16-byte loads and both
            // 32-byte stores.
            unsafe {
                let lo_bytes = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                let hi_bytes = _mm_loadu_si128(src.as_ptr().add(i + 8).cast::<__m128i>());
                let lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_bytes));
                let hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_bytes));
                let s = _mm256_set1_ps(scale);
                let o = _mm256_set1_ps(offset);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(_mm256_mul_ps(lo, s), o));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_add_ps(_mm256_mul_ps(hi, s), o));
                i += 16;
            }
        }
        i
    }
}

/// AVX-512 tier: 64-byte checksum blocks, 32-byte data blocks.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, __m256i, __m512i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi16, _mm512_add_epi32, _mm512_add_ps,
        _mm512_and_si512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cmpge_epu16_mask,
        _mm512_cvtepi16_epi32, _mm512_cvtepi32_ps, _mm512_cvtepu8_epi16, _mm512_cvtusepi16_epi8,
        _mm512_cvtusepi32_epi8, _mm512_cvttps_epi32, _mm512_extracti64x4_epi64, _mm512_loadu_ps,
        _mm512_loadu_si512, _mm512_min_ps, _mm512_mul_ps, _mm512_mulhi_epu16, _mm512_mullo_epi16,
        _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_shuffle_epi8,
        _mm512_srli_epi16, _mm512_storeu_ps, _mm512_storeu_si512, _mm_add_epi32, _mm_cvtsi128_si32,
        _mm_loadu_si128, _mm_set1_epi32, _mm_srli_si128, _mm_storeu_si128, _mm_unpacklo_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi16, _mm512_add_epi32, _mm512_and_si512,
        _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cmpge_epu16_mask, _mm512_cvtepi16_epi32,
        _mm512_cvtepi32_ps, _mm512_cvtepu8_epi16,
        _mm512_cvttps_epi32, _mm512_cvtusepi16_epi8, _mm512_cvtusepi32_epi8, _mm512_extracti64x4_epi64, _mm512_loadu_ps, _mm512_loadu_si512,
        _mm512_min_ps, _mm512_mul_ps, _mm512_mulhi_epu16, _mm512_mullo_epi16, _mm512_set1_epi16,
        _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_srli_epi16,
        _mm512_storeu_ps, _mm_add_epi32, _mm_cvtsi128_si32, _mm_srli_si128, _mm_storeu_si128, _mm_unpacklo_epi64,
    };
    use core::arch::x86_64::{_mm512_add_ps, _mm512_castsi512_si256};

    /// Input bytes consumed per checksum iteration.
    const BLOCK: usize = 64;

    /// `pshufb` control mask that byte-swaps every `u32` lane of a
    /// 64-byte register (the pattern of [`sse2::bswap_epi32`]).
    const BSWAP_MASK: [u8; 64] = {
        let mut mask = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            mask[i] = i as u8 + 3;
            mask[i + 1] = i as u8 + 2;
            mask[i + 2] = i as u8 + 1;
            mask[i + 3] = i as u8;
            i += 4;
        }
        mask
    };

    /// AVX-512 checksum over 64-byte blocks plus the scalar tail.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512 F+BW+DQ+VL and OS ZMM state.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "avx512vl", enable = "avx2")]
    pub(super) unsafe fn checksum(data: &[u8]) -> u32 {
        // SAFETY: `target_feature` guarantees the intrinsics; the loop
        // condition keeps every load inside `data`, and the mask is a
        // 64-byte constant.
        unsafe {
            let mask = _mm512_loadu_si512(BSWAP_MASK.as_ptr().cast());
            let mut acc = _mm512_setzero_si512();
            let mut i = 0;
            while i + BLOCK <= data.len() {
                let v = _mm512_loadu_si512(data.as_ptr().add(i).cast());
                acc = _mm512_add_epi32(acc, _mm512_shuffle_epi8(v, mask));
                i += BLOCK;
            }
            let hi = _mm512_extracti64x4_epi64(acc, 1);
            let lo = _mm512_castsi512_si256(acc);
            let sum256 = _mm256_add_epi32(lo, hi);
            let hi128 = _mm256_extracti128_si256(sum256, 1);
            let merged = _mm_add_epi32(_mm256_castsi256_si128(sum256), hi128);
            hsum128(merged).wrapping_add(super::scalar::checksum(&data[i..]))
        }
    }

    /// Wrapping horizontal sum of four `u32` lanes.
    ///
    /// # Safety
    ///
    /// Requires SSE2 (implied by AVX-512).
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "avx512vl", enable = "avx2")]
    unsafe fn hsum128(v: __m128i) -> u32 {
        let hi64 = _mm_unpacklo_epi64(v, v);
        let sum2 = _mm_add_epi32(v, hi64);
        let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 8));
        _mm_cvtsi128_si32(sum1) as u32
    }

    /// AVX-512 ordered `u16` scan of one 32-element block; `None` when
    /// every element of the block is `< code`.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512 F+BW+DQ+VL and OS ZMM state.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "avx512vl", enable = "avx2")]
    pub(super) unsafe fn find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
        let n = end_codes.len();
        let mut i = 0;
        while i + 32 <= n {
            // SAFETY: `i + 32 <= n` keeps the 64-byte load in bounds.
            unsafe {
                let v = _mm512_loadu_si512(end_codes.as_ptr().add(i).cast::<__m512i>());
                let ge = _mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(code as i16));
                if ge != 0 {
                    return Some(i + ge.trailing_zeros() as usize);
                }
                i += 32;
            }
        }
        None
    }

    /// AVX-512 coverage finalization; returns the number of `f32`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512 F+BW+DQ+VL and OS ZMM state.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "avx512vl", enable = "avx2")]
    pub(super) unsafe fn finalize(acc: &[f32], out: &mut [u8]) -> usize {
        let n = acc.len().min(out.len());
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n` bounds the 64-byte load and the
            // 16-byte store.
            unsafe {
                let v = _mm512_loadu_ps(acc.as_ptr().add(i));
                let abs = _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x7FFF_FFFF));
                let clamped = _mm512_min_ps(_mm512_castsi512_ps(abs), _mm512_set1_ps(1.0));
                let scaled = _mm512_mul_ps(clamped, _mm512_set1_ps(255.0));
                let ints = _mm512_cvttps_epi32(scaled);
                let bytes = _mm512_cvtusepi32_epi8(ints);
                _mm_storeu_si128(out.as_mut_ptr().add(i).cast::<__m128i>(), bytes);
                i += 16;
            }
        }
        i
    }

    /// AVX-512 exact source-over blend; returns bytes consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512 F+BW+DQ+VL and OS ZMM state;
    /// `dst`/`src` are clipped to equal length by the dispatcher.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "avx512vl", enable = "avx2")]
    pub(super) unsafe fn blend(dst: &mut [u8], src: &[u8], alpha: u8) -> usize {
        let n = dst.len().min(src.len());
        let a = _mm512_set1_epi16(i16::from(alpha));
        let inv = _mm512_set1_epi16(255 - i16::from(alpha));
        let magic = _mm512_set1_epi16(0x8081_u16 as i16);
        let mut i = 0;
        while i + 32 <= n {
            // SAFETY: `i + 32 <= n` keeps both 32-byte loads and the
            // store inside their slices.
            unsafe {
                let s = _mm512_cvtepu8_epi16(_mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>()));
                let d = _mm512_cvtepu8_epi16(_mm256_loadu_si256(dst.as_ptr().add(i).cast::<__m256i>()));
                let t = _mm512_add_epi16(_mm512_mullo_epi16(a, s), _mm512_mullo_epi16(inv, d));
                let q = _mm512_srli_epi16(_mm512_mulhi_epu16(t, magic), 7);
                let r = _mm512_add_epi16(d, q);
                let bytes = _mm512_cvtusepi16_epi8(r);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), bytes);
                i += 32;
            }
        }
        i
    }

    /// AVX-512 `i16` → `f32` scale+offset; returns `i16`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512 F+BW+DQ+VL and OS ZMM state.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "avx512vl", enable = "avx2")]
    pub(super) unsafe fn coords(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) -> usize {
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n` bounds both 16-byte loads and both
            // 64-byte stores.
            unsafe {
                let lo = _mm512_cvtepi16_epi32(_mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>()));
                let hi =
                    _mm512_cvtepi16_epi32(_mm256_loadu_si256(src.as_ptr().add(i + 16).cast::<__m256i>()));
                let s = _mm512_set1_ps(scale);
                let o = _mm512_set1_ps(offset);
                let f_lo = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(lo), s), o);
                let f_hi = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(hi), s), o);
                _mm512_storeu_ps(dst.as_mut_ptr().add(i), f_lo);
                _mm512_storeu_ps(dst.as_mut_ptr().add(i + 8), f_hi);
                i += 16;
            }
        }
        i
    }

    /// Unused import placeholder removed by the compiler; keeps the
    /// x86 (32-bit) import list symmetric with the x86_64 one.
    #[allow(dead_code)]
    type Unused = __m512i;
}

/// NEON tier: 16-byte blocks for every kernel.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        uint16x8_t, uint32x4_t, uint8x8_t, vabsq_f32, vaddq_f32, vaddq_u16, vaddq_u32, vaddvq_u32,
        vcgeq_u16, vcvtq_f32_s32, vcvtq_u32_f32, vdupq_n_f32, vdupq_n_u16, vdupq_n_u32, vget_high_s16,
        vget_high_u16, vget_high_u8, vget_lane_u64, vget_low_s16, vget_low_u16, vget_low_u8, vld1q_f32,
        vld1q_s16, vld1q_u16, vld1q_u8, vmovl_s16, vmovl_u16, vmovl_u8, vmovn_u16, vmovn_u32,
        vmulq_f32, vmulq_n_f32, vmulq_u16, vmulq_u32, vcombine_u16, vcombine_u8, vreinterpret_u64_u8,
        vreinterpretq_u32_u8, vrev32q_u8, vshrq_n_u32, vminq_f32, vst1_u8, vst1q_f32, vst1q_u8,
    };

    /// NEON checksum over 16-byte blocks plus the scalar tail.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON (Advanced SIMD).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn checksum(data: &[u8]) -> u32 {
        // SAFETY: `target_feature` guarantees the intrinsics; the loop
        // condition keeps every load inside `data`.
        unsafe {
            let mut total = 0u32;
            let mut i = 0;
            while i + 16 <= data.len() {
                let bytes = vld1q_u8(data.as_ptr().add(i));
                // Reversing the bytes of every 32-bit lane turns the
                // little-endian load into four big-endian words.
                let words = vreinterpretq_u32_u8(vrev32q_u8(bytes));
                total = total.wrapping_add(vaddvq_u32(words));
                i += 16;
            }
            total.wrapping_add(super::scalar::checksum(&data[i..]))
        }
    }

    /// NEON ordered `u16` scan of one 8-element block; `None` when
    /// every element of the block is `< code`.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON (Advanced SIMD).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn find_end_code(end_codes: &[u16], code: u16) -> Option<usize> {
        let n = end_codes.len();
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n` keeps the 16-byte load in bounds.
            unsafe {
                let v = vld1q_u16(end_codes.as_ptr().add(i));
                let ge = vcgeq_u16(v, vdupq_n_u16(code));
                // Each comparison lane becomes one `0xFF` byte; the
                // first set byte marks the first segment that ends at
                // or after `code`.
                let bytes = vmovn_u16(ge);
                let mask = vget_lane_u64(vreinterpret_u64_u8(bytes), 0);
                if mask != u64::MAX {
                    return Some(i + (mask.trailing_zeros() / 8) as usize);
                }
                i += 8;
            }
        }
        None
    }

    /// NEON coverage finalization; returns the number of `f32`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON (Advanced SIMD).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn finalize(acc: &[f32], out: &mut [u8]) -> usize {
        let n = acc.len().min(out.len());
        let mut i = 0;
        while i + 4 <= n {
            // SAFETY: `i + 4 <= n` bounds the 16-byte load; only the
            // first four bytes of the staging buffer are written out.
            unsafe {
                let v = vld1q_f32(acc.as_ptr().add(i));
                let clamped = vminq_f32(vabsq_f32(v), vdupq_n_f32(1.0));
                let scaled = vmulq_f32(clamped, vdupq_n_f32(255.0));
                let ints = vcvtq_u32_f32(scaled);
                let words = vmovn_u32(ints);
                let bytes = vmovn_u16(vcombine_u16(words, words));
                let mut buf = [0u8; 8];
                vst1_u8(buf.as_mut_ptr(), bytes);
                out[i..i + 4].copy_from_slice(&buf[..4]);
                i += 4;
            }
        }
        i
    }

    /// NEON exact source-over blend; returns bytes consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON (Advanced SIMD); `dst`/`src` are
    /// clipped to equal length by the dispatcher.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn blend(dst: &mut [u8], src: &[u8], alpha: u8) -> usize {
        let n = dst.len().min(src.len());
        let a = vdupq_n_u16(u16::from(alpha));
        let inv = vdupq_n_u16(255 - u16::from(alpha));
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n` keeps both 16-byte loads and the
            // store inside their slices.
            unsafe {
                let s = vld1q_u8(src.as_ptr().add(i));
                let d = vld1q_u8(dst.as_ptr().add(i));
                let lo = blend8(vget_low_u8(s), vget_low_u8(d), a, inv);
                let hi = blend8(vget_high_u8(s), vget_high_u8(d), a, inv);
                vst1q_u8(dst.as_mut_ptr().add(i), vcombine_u8(lo, hi));
                i += 16;
            }
        }
        i
    }

    /// One 8-lane blend step; see the SSE2 kernel of the same name.
    ///
    /// # Safety
    ///
    /// Requires NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn blend8(s: uint8x8_t, d: uint8x8_t, a: uint16x8_t, inv: uint16x8_t) -> uint8x8_t {
        // SAFETY: register-only NEON operations; the products stay
        // below 65025, so the narrowing stores never saturate.
        unsafe {
            let su = vmovl_u8(s);
            let du = vmovl_u8(d);
            let t = vaddq_u16(vmulq_u16(a, su), vmulq_u16(inv, du));
            let lo = narrow_add(vmovl_u16(vget_low_u16(t)), vmovl_u16(vget_low_u16(du)));
            let hi = narrow_add(vmovl_u16(vget_high_u16(t)), vmovl_u16(vget_high_u16(du)));
            vmovn_u16(vcombine_u16(lo, hi))
        }
    }

    /// `d + floor(t * 0x8081 >> 23)` for four lanes at a time.
    ///
    /// `t * 0x8081 <= 65025 * 32897` fits an `u32`, and the shift is
    /// exact for every `t <= 66051`.
    ///
    /// # Safety
    ///
    /// Requires NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn narrow_add(t: uint32x4_t, d: uint32x4_t) -> uint16x8_t {
        // SAFETY: register-only NEON operations.
        unsafe {
            let q = vshrq_n_u32(vmulq_u32(t, vdupq_n_u32(0x8081)), 23);
            vmovn_u32(vaddq_u32(d, q))
        }
    }

    /// NEON `i16` → `f32` scale+offset; returns `i16`s consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON (Advanced SIMD); slices are clipped
    /// to equal length by the dispatcher.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn coords(src: &[i16], dst: &mut [f32], scale: f32, offset: f32) -> usize {
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n` bounds the 16-byte load and both
            // 16-byte stores.
            unsafe {
                let v = vld1q_s16(src.as_ptr().add(i));
                let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
                let hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v)));
                let o = vdupq_n_f32(offset);
                // Separate multiply and add (no fused form) so the
                // result matches the scalar path bit for bit.
                vst1q_f32(dst.as_mut_ptr().add(i), vaddq_f32(vmulq_n_f32(lo, scale), o));
                vst1q_f32(dst.as_mut_ptr().add(i + 4), vaddq_f32(vmulq_n_f32(hi, scale), o));
                i += 8;
            }
        }
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift generator (no external crate).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i.wrapping_mul(31) + 7) as u8).collect()
    }

    #[test]
    fn checksum_matches_scalar_for_every_length() {
        for len in 0..=140 {
            let data = pattern(len);
            assert_eq!(table_checksum(&data), scalar::checksum(&data), "len {len}");
        }
    }

    #[test]
    fn checksum_matches_scalar_random_buffers() {
        let mut rng = Rng::new(0x5EED);
        for _ in 0..64 {
            let len = rng.below(300) as usize;
            let data: Vec<u8> = (0..len).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
            assert_eq!(table_checksum(&data), scalar::checksum(&data));
        }
    }

    #[test]
    fn checksum_wraps_like_the_scalar_path() {
        let data = [0xFFu8; 4096];
        assert_eq!(table_checksum(&data), scalar::checksum(&data));
    }

    /// Builds a strictly ascending segment table of `len` entries.
    fn sorted_codes(len: usize, seed: u64) -> Vec<u16> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(len);
        let mut next = 0u32;
        for _ in 0..len {
            next += 1 + rng.below(7) as u32;
            out.push(u16::try_from(next.min(0xFFFF)).unwrap_or(0xFFFF));
            if next >= 0xFFFF {
                break;
            }
        }
        out
    }

    #[test]
    fn find_end_code_matches_scalar_on_sorted_tables() {
        for len in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 1000] {
            let codes = sorted_codes(len, u64::try_from(len).unwrap_or(1) + 3);
            let mut queries: Vec<u16> = vec![0, 1, 0xFFFF, 0x8000];
            for _ in 0..24 {
                queries.push(u16::try_from(Rng::new(len as u64).below(0x1_0000)).unwrap_or(0));
            }
            for &code in &codes {
                queries.push(code);
                queries.push(code.saturating_add(1));
                queries = queries; // keep the loop body explicit
            }
            for code in queries {
                assert_eq!(find_end_code(&codes, code), scalar::find_end_code(&codes, code), "len {len} code {code}");
            }
        }
    }

    #[test]
    fn find_end_code_finds_the_terminator() {
        let codes = [0x0041u16, 0x005A, 0x007A, 0xFFFF];
        for code in [0u16, 0x41, 0x42, 0x5A, 0x5B, 0x7A, 0x7B, 0xFFFF] {
            assert_eq!(find_end_code(&codes, code), scalar::find_end_code(&codes, code));
        }
    }

    #[test]
    fn find_end_code_empty_and_single_entry() {
        assert_eq!(find_end_code(&[], 0), None);
        assert_eq!(find_end_code(&[0xFFFF], 0), Some(0));
        assert_eq!(find_end_code(&[0xFFFF], 0xFFFF), Some(0));
        assert_eq!(find_end_code(&[10], 9), Some(0));
        assert_eq!(find_end_code(&[10], 11), None);
    }

    #[test]
    fn finalize_matches_scalar() {
        let values: Vec<f32> = [
            -2.0f32, -1.0, -0.5, -0.001, 0.0, 0.001, 0.5, 1.0 / 255.0, 127.0 / 255.0, 0.999, 1.0,
            1.001, 2.0,
        ]
        .iter()
        .copied()
        .collect();
        let mut expected = vec![0u8; values.len()];
        scalar::finalize(&values, &mut expected);
        let mut got = vec![0u8; values.len()];
        finalize_coverage(&values, &mut got);
        assert_eq!(got, expected);
    }

    #[test]
    fn finalize_matches_scalar_random_rows() {
        let mut rng = Rng::new(0xC0FFEE);
        for len in [0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65] {
            let values: Vec<f32> =
                (0..len).map(|_| (rng.below(2000) as i64 - 1000) as f32 / 700.0).collect();
            let mut expected = vec![0u8; len];
            scalar::finalize(&values, &mut expected);
            let mut got = vec![0u8; len];
            finalize_coverage(&values, &mut got);
            assert_eq!(got, expected, "len {len}");
        }
    }

    #[test]
    fn finalize_writes_only_the_overlap() {
        let values = [0.25f32, 0.75, 1.0];
        let mut out = [7u8; 5];
        finalize_coverage(&values, &mut out);
        assert_eq!(out, [63, 191, 255, 7, 7]);
    }

    #[test]
    fn blend_matches_scalar_for_every_alpha() {
        let len = 129;
        let mut rng = Rng::new(0xB1E7);
        let src: Vec<u8> = (0..len).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
        let dst: Vec<u8> = (0..len).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
        for alpha in 0..=255u16 {
            let alpha = u8::try_from(alpha).unwrap_or(0);
            let mut expected = dst.clone();
            scalar::blend(&mut expected, &src, alpha);
            let mut got = dst.clone();
            blend_coverage(&mut got, &src, alpha);
            assert_eq!(got, expected, "alpha {alpha}");
        }
    }

    #[test]
    fn blend_matches_scalar_on_block_boundaries() {
        let mut rng = Rng::new(0x5AFE);
        for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255] {
            let src: Vec<u8> = (0..len).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
            let dst: Vec<u8> = (0..len).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
            for alpha in [0u8, 1, 64, 127, 128, 200, 254, 255] {
                let mut expected = dst.clone();
                scalar::blend(&mut expected, &src, alpha);
                let mut got = dst.clone();
                blend_coverage(&mut got, &src, alpha);
                assert_eq!(got, expected, "len {len} alpha {alpha}");
            }
        }
    }

    #[test]
    fn blend_is_exact_against_a_u32_reference() {
        // Reference: floor((s*a + d*(255-a)) / 255) computed in u32.
        let mut rng = Rng::new(0xE7AC7);
        for _ in 0..2000 {
            let s = u8::try_from(rng.below(256)).unwrap_or(0);
            let d = u8::try_from(rng.below(256)).unwrap_or(0);
            let a = u8::try_from(rng.below(256)).unwrap_or(0);
            let reference =
                (u32::from(s) * u32::from(a) + u32::from(d) * (255 - u32::from(a))) / 255;
            let mut dst = [d];
            blend_coverage(&mut dst, &[s], a);
            assert_eq!(u32::from(dst[0]), reference, "s {s} d {d} a {a}");
        }
    }

    #[test]
    fn blend_shorter_destination_wins() {
        let mut dst = [10u8, 20];
        blend_coverage(&mut dst, &[255, 255, 255], 255);
        assert_eq!(dst, [255, 255]);
    }

    #[test]
    fn coords_match_scalar() {
        let mut rng = Rng::new(0xC001D00D);
        for len in [0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 64] {
            let src: Vec<i16> =
                (0..len).map(|_| rng.below(0x1_0000) as i16).collect();
            for (scale, offset) in [(1.0f32, 0.0f32), (0.0, 0.0), (-1.5, 3.25), (7.0 / 1024.0, -0.5)] {
                let mut expected = vec![0.0f32; len];
                scalar::coords(&src, &mut expected, scale, offset);
                let mut got = vec![0.0f32; len];
                coords_to_f32(&src, &mut got, scale, offset);
                assert_eq!(got, expected, "len {len} scale {scale}");
            }
        }
    }

    fn record(tag: [u8; 4]) -> TableRecord {
        TableRecord { tag: u32::from_be_bytes(tag), checksum: 0, offset: 0, length: 0 }
    }

    #[test]
    fn find_tag_binary_searches_a_sorted_directory() {
        let tags: [[u8; 4]; 7] =
            [*b"cmap", *b"fpgm", *b"glyf", *b"head", *b"hhea", *b"maxp", *b"name"];
        let records: Vec<TableRecord> = tags.iter().map(|t| record(*t)).collect();
        for (i, tag) in tags.iter().enumerate() {
            assert_eq!(find_tag(&records, u32::from_be_bytes(*tag)), i);
        }
        assert_eq!(find_tag(&records, u32::from_be_bytes(*b"zzzz")), records.len());
        assert_eq!(find_tag(&records, u32::from_be_bytes(*b"CVT ")), records.len());
    }

    #[test]
    fn find_tag_falls_back_for_unsorted_directories() {
        let tags: [[u8; 4]; 5] = [*b"name", *b"head", *b"OS/2", *b"cmap", *b"glyf"];
        let records: Vec<TableRecord> = tags.iter().map(|t| record(*t)).collect();
        assert_eq!(find_tag(&records, u32::from_be_bytes(*b"OS/2")), 2);
        assert_eq!(find_tag(&records, u32::from_be_bytes(*b"glyf")), 4);
        assert_eq!(find_tag(&records, u32::from_be_bytes(*b"zzzz")), 5);
    }

    #[test]
    fn find_tag_handles_an_empty_directory() {
        assert_eq!(find_tag(&[], u32::from_be_bytes(*b"head")), 0);
    }

    /// Runs one kernel on a specific tier when the CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn every_available_tier_matches_scalar_on_x86() {
        let data = pattern(1024);
        assert_eq!(scalar::checksum(&data), scalar::checksum(&data));

        if basic_cpuid::has(Feature::Sse2) {
            // SAFETY: the CPU reports SSE2.
            let got = unsafe { sse2::checksum(&data) };
            assert_eq!(got, scalar::checksum(&data), "sse2");
        }
        if basic_cpuid::has(Feature::Ssse3) {
            // SAFETY: the CPU reports SSSE3.
            let got = unsafe { ssse3::checksum(&data) };
            assert_eq!(got, scalar::checksum(&data), "ssse3");
        }
        if basic_cpuid::has(Feature::Avx2) {
            // SAFETY: the CPU reports AVX2 and OS YMM state.
            let got = unsafe { avx2::checksum(&data) };
            assert_eq!(got, scalar::checksum(&data), "avx2");

            let mut acc = vec![0.0f32; 40];
            let mut rng = Rng::new(9);
            for a in &mut acc {
                *a = (rng.below(600) as i64 - 300) as f32 / 300.0;
            }
            let mut expected = vec![0u8; acc.len()];
            scalar::finalize(&acc, &mut expected);
            let mut got = vec![0u8; acc.len()];
            // SAFETY: the CPU reports AVX2 and OS YMM state.
            let consumed = unsafe { avx2::finalize(&acc, &mut got) };
            scalar::finalize(&acc[consumed..], &mut got[consumed..]);
            assert_eq!(got, expected, "avx2 finalize");

            let src: Vec<u8> = (0..96).map(|i| (i * 17 + 3) as u8).collect();
            let dst: Vec<u8> = (0..96).map(|i| (i * 5 + 91) as u8).collect();
            let mut expected = dst.clone();
            scalar::blend(&mut expected, &src, 137);
            let mut got = dst.clone();
            // SAFETY: the CPU reports AVX2 and OS YMM state.
            let consumed = unsafe { avx2::blend(&mut got, &src, 137) };
            scalar::blend(&mut got[consumed..], &src[consumed..], 137);
            assert_eq!(got, expected, "avx2 blend");

            let coords: Vec<i16> = (0..80).map(|i| (i * 61 - 2400) as i16).collect();
            let mut expected = vec![0.0f32; coords.len()];
            scalar::coords(&coords, &mut expected, 0.75, -12.5);
            let mut got = vec![0.0f32; coords.len()];
            // SAFETY: the CPU reports AVX2 and OS YMM state.
            let consumed = unsafe { avx2::coords(&coords, &mut got, 0.75, -12.5) };
            scalar::coords(&coords[consumed..], &mut got[consumed..], 0.75, -12.5);
            assert_eq!(got, expected, "avx2 coords");
        }
        if basic_cpuid::has(Feature::Avx512) {
            // SAFETY: the CPU reports AVX-512 F+BW+DQ+VL and OS ZMM state.
            let got = unsafe { avx512::checksum(&data) };
            assert_eq!(got, scalar::checksum(&data), "avx512");

            let codes = sorted_codes(96, 42);
            for code in 0..=0xFFFFu16 {
                if code % 51 != 0 {
                    continue;
                }
                // SAFETY: the CPU reports AVX-512 F+BW+DQ+VL and OS ZMM state.
                let got = unsafe { avx512::find_end_code(&codes, code) };
                assert_eq!(got, scalar::find_end_code(&codes, code), "avx512 find {code}");
            }

            let mut acc = vec![0.0f32; 40];
            let mut rng = Rng::new(11);
            for a in &mut acc {
                *a = (rng.below(600) as i64 - 300) as f32 / 300.0;
            }
            let mut expected = vec![0u8; acc.len()];
            scalar::finalize(&acc, &mut expected);
            let mut got = vec![0u8; acc.len()];
            // SAFETY: the CPU reports AVX-512 F+BW+DQ+VL and OS ZMM state.
            let consumed = unsafe { avx512::finalize(&acc, &mut got) };
            scalar::finalize(&acc[consumed..], &mut got[consumed..]);
            assert_eq!(got, expected, "avx512 finalize");

            let src: Vec<u8> = (0..97).map(|i| (i * 29 + 7) as u8).collect();
            let dst: Vec<u8> = (0..97).map(|i| (i * 13 + 44) as u8).collect();
            let mut expected = dst.clone();
            scalar::blend(&mut expected, &src, 51);
            let mut got = dst.clone();
            // SAFETY: the CPU reports AVX-512 F+BW+DQ+VL and OS ZMM state.
            let consumed = unsafe { avx512::blend(&mut got, &src, 51) };
            scalar::blend(&mut got[consumed..], &src[consumed..], 51);
            assert_eq!(got, expected, "avx512 blend");

            let coords: Vec<i16> = (0..80).map(|i| (i * 97 - 3000) as i16).collect();
            let mut expected = vec![0.0f32; coords.len()];
            scalar::coords(&coords, &mut expected, 1.25, 4.0);
            let mut got = vec![0.0f32; coords.len()];
            // SAFETY: the CPU reports AVX-512 F+BW+DQ+VL and OS ZMM state.
            let consumed = unsafe { avx512::coords(&coords, &mut got, 1.25, 4.0) };
            scalar::coords(&coords[consumed..], &mut got[consumed..], 1.25, 4.0);
            assert_eq!(got, expected, "avx512 coords");
        }
    }

    /// Runs the NEON kernels when the CPU reports Advanced SIMD.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn every_available_tier_matches_scalar_on_aarch64() {
        let data = pattern(1024);
        if basic_cpuid::has(Feature::Neon) {
            // SAFETY: the CPU reports NEON.
            let got = unsafe { neon::checksum(&data) };
            assert_eq!(got, scalar::checksum(&data), "neon checksum");

            let codes = sorted_codes(96, 7);
            for code in [0u16, 1, 0x7FFF, 0xFFFE, 0xFFFF] {
                // SAFETY: the CPU reports NEON.
                let got = unsafe { neon::find_end_code(&codes, code) };
                assert_eq!(got, scalar::find_end_code(&codes, code), "neon find {code}");
            }

            let mut acc = vec![0.0f32; 40];
            let mut rng = Rng::new(13);
            for a in &mut acc {
                *a = (rng.below(600) as i64 - 300) as f32 / 300.0;
            }
            let mut expected = vec![0u8; acc.len()];
            scalar::finalize(&acc, &mut expected);
            let mut got = vec![0u8; acc.len()];
            // SAFETY: the CPU reports NEON.
            let consumed = unsafe { neon::finalize(&acc, &mut got) };
            scalar::finalize(&acc[consumed..], &mut got[consumed..]);
            assert_eq!(got, expected, "neon finalize");

            let src: Vec<u8> = (0..71).map(|i| (i * 31 + 5) as u8).collect();
            let dst: Vec<u8> = (0..71).map(|i| (i * 17 + 90) as u8).collect();
            let mut expected = dst.clone();
            scalar::blend(&mut expected, &src, 180);
            let mut got = dst.clone();
            // SAFETY: the CPU reports NEON.
            let consumed = unsafe { neon::blend(&mut got, &src, 180) };
            scalar::blend(&mut got[consumed..], &src[consumed..], 180);
            assert_eq!(got, expected, "neon blend");

            let coords: Vec<i16> = (0..71).map(|i| (i * 71 - 2000) as i16).collect();
            let mut expected = vec![0.0f32; coords.len()];
            scalar::coords(&coords, &mut expected, 0.5, -2.0);
            let mut got = vec![0.0f32; coords.len()];
            // SAFETY: the CPU reports NEON.
            let consumed = unsafe { neon::coords(&coords, &mut got, 0.5, -2.0) };
            scalar::coords(&coords[consumed..], &mut got[consumed..], 0.5, -2.0);
            assert_eq!(got, expected, "neon coords");
        }
    }
}
