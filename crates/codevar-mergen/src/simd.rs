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

//! SIMD rank-merge kernels for the Mergen merge algorithm.
//!
//! The kernels compute, for two streams of SHAKE256 change keys, the
//! position each element takes in the merged order:
//!
//! - [`merge_rank_u64`] fills both rank arrays in one pass of counts
//! - [`count_less_u64`] / [`count_le_u64`] count keys below a
//!   threshold
//! - [`first_mismatch_u8`] finds the first differing byte of two
//!   buffers (used to detect identical content)
//!
//! Backends are selected at run time: x86 / x86_64 picks the widest
//! of AVX-512F, AVX2, SSE4.2 (plus SSE2 for byte scans), aarch64 uses
//! Advanced SIMD (NEON), wasm32 with `simd128` uses v128; every
//! other target uses the scalar loop. All backends produce
//! bit-identical results — the tier only affects performance.
//!
//! All counting kernels are branchless in the key values (constant
//! time with respect to the keys being compared), matching the
//! constant-time ranking used by the merge algorithm itself.

/// Merges the ranks of two key streams into `ranks_a` and `ranks_b`.
///
/// For each `a[i]`, `ranks_a[i] = i + |{ b[j] : b[j] < a[i] }|` —
/// the index `a[i]` would occupy in the stable merge of `a` and `b`
/// where `a` wins ties. For each `b[j]`,
/// `ranks_b[j] = j + |{ a[i] : a[i] <= b[j] }|`.
///
/// Both streams must be sorted in non-decreasing order (the merge
/// algorithm feeds them SHAKE256 change keys in sorted order). The
/// rank arrays must have the same lengths as `a` and `b`, and
/// `a.len() + b.len()` must fit in `u32`; otherwise the call is a
/// no-op.
///
/// # Examples
///
/// ```
/// let a = [1u64, 3, 5];
/// let b = [2u64, 3, 4];
/// let mut ranks_a = [0u32; 3];
/// let mut ranks_b = [0u32; 3];
/// codevar_mergen::merge_rank_u64(&a, &b, &mut ranks_a, &mut ranks_b);
/// assert_eq!(ranks_a, [0, 2, 5]);
/// assert_eq!(ranks_b, [1, 3, 4]);
/// ```
pub fn merge_rank_u64(a: &[u64], b: &[u64], ranks_a: &mut [u32], ranks_b: &mut [u32]) {
    if ranks_a.len() != a.len() || ranks_b.len() != b.len() {
        return;
    }
    if a.len().saturating_add(b.len()) > u32::MAX as usize {
        return;
    }
    for (i, &key) in a.iter().enumerate() {
        ranks_a[i] = (i as u64 + count_less_u64(b, key)) as u32;
    }
    for (j, &key) in b.iter().enumerate() {
        ranks_b[j] = (j as u64 + count_le_u64(a, key)) as u32;
    }
}

/// Counts how many of `keys` are strictly less than `key`.
///
/// # Examples
///
/// ```
/// assert_eq!(codevar_mergen::count_less_u64(&[1, 3, 5, 7], 5), 2);
/// assert_eq!(codevar_mergen::count_less_u64(&[], 5), 0);
/// ```
#[must_use]
pub fn count_less_u64(keys: &[u64], key: u64) -> u64 {
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            x86::count_less_u64(keys, key)
        } else if #[cfg(target_arch = "aarch64")] {
            aarch64::count_less_u64(keys, key)
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            wasm::count_less_u64(keys, key)
        } else {
            scalar::count_less_u64(keys, key)
        }
    }
}

/// Counts how many of `keys` are less than or equal to `key`.
///
/// # Examples
///
/// ```
/// assert_eq!(codevar_mergen::count_le_u64(&[1, 3, 5, 7], 5), 3);
/// assert_eq!(codevar_mergen::count_le_u64(&[], 5), 0);
/// ```
#[must_use]
pub fn count_le_u64(keys: &[u64], key: u64) -> u64 {
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            x86::count_le_u64(keys, key)
        } else if #[cfg(target_arch = "aarch64")] {
            aarch64::count_le_u64(keys, key)
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            wasm::count_le_u64(keys, key)
        } else {
            scalar::count_le_u64(keys, key)
        }
    }
}

/// Returns the index of the first byte where `a` and `b` differ,
/// scanning only the first `min(a.len(), b.len())` bytes; if none
/// differ, returns that shared length.
///
/// # Examples
///
/// ```
/// assert_eq!(codevar_mergen::first_mismatch_u8(b"abcXde", b"abcYde"), 3);
/// assert_eq!(codevar_mergen::first_mismatch_u8(b"same", b"same"), 4);
/// ```
#[must_use]
pub fn first_mismatch_u8(a: &[u8], b: &[u8]) -> usize {
    let shared = core::cmp::min(a.len(), b.len());
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            x86::first_mismatch_u8(a, b, shared)
        } else if #[cfg(target_arch = "aarch64")] {
            aarch64::first_mismatch_u8(a, b, shared)
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            wasm::first_mismatch_u8(a, b, shared)
        } else {
            scalar::first_mismatch_u8(a, b, shared)
        }
    }
}

/// Scalar reference backend; also the tail handler for every SIMD
/// backend (tails are shorter than one vector, so they never
/// re-enter a SIMD kernel).
mod scalar {
    /// All-ones mask when `a < b` (unsigned), zero otherwise.
    ///
    /// Sign-bit flip turns unsigned order into signed order, then the
    /// Hacker's Delight signed comparison yields the answer in the
    /// sign bit with no branches.
    #[inline(always)]
    pub(super) fn lt_mask_u64(a: u64, b: u64) -> u64 {
        const TOP: u64 = 1 << 63;
        let x = a ^ TOP;
        let y = b ^ TOP;
        let diff = x.wrapping_sub(y);
        let t = (x ^ y) & (diff ^ x);
        (diff ^ t) >> 63
    }

    pub(super) fn count_less_u64(keys: &[u64], key: u64) -> u64 {
        let mut total = 0u64;
        for &other in keys {
            total += lt_mask_u64(other, key) & 1;
        }
        total
    }

    pub(super) fn count_le_u64(keys: &[u64], key: u64) -> u64 {
        let mut total = 0u64;
        for &other in keys {
            total += 1 ^ (lt_mask_u64(key, other) & 1);
        }
        total
    }

    pub(super) fn first_mismatch_u8(a: &[u8], b: &[u8], shared: usize) -> usize {
        for index in 0..shared {
            if a[index] != b[index] {
                return index;
            }
        }
        shared
    }
}

/// x86 / x86_64 backends.
///
/// Kernels are tiered by the instructions they actually need and
/// selected at run time through `codevar_base::cpuid`:
///
/// - **AVX-512F**: eight `u64` keys compared per instruction
///   (`_mm512_cmp_epi64_mask` straight to a bitmask)
/// - **AVX2**: four `u64` keys per instruction
/// - **SSE4.2**: two `u64` keys per instruction
///   (`_mm_cmpgt_epi64`; earlier SSE tiers have no 64-bit integer
///   compare, so they cannot accelerate the rank kernels)
/// - **SSE2**: 16-byte byte comparisons for the mismatch scan
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use super::scalar;

    #[must_use]
    pub(super) fn count_less_u64(keys: &[u64], key: u64) -> u64 {
        if avx512_available() {
            // SAFETY: `avx512_available` proves AVX-512F at run time.
            unsafe { count_less_avx512(keys, key) }
        } else if avx2_available() {
            // SAFETY: `avx2_available` proves AVX2 at run time.
            unsafe { count_less_avx2(keys, key) }
        } else if sse42_available() {
            // SAFETY: `sse42_available` proves SSE4.2 at run time.
            unsafe { count_less_sse42(keys, key) }
        } else {
            scalar::count_less_u64(keys, key)
        }
    }

    #[must_use]
    pub(super) fn count_le_u64(keys: &[u64], key: u64) -> u64 {
        if avx512_available() {
            // SAFETY: `avx512_available` proves AVX-512F at run time.
            unsafe { count_le_avx512(keys, key) }
        } else if avx2_available() {
            // SAFETY: `avx2_available` proves AVX2 at run time.
            unsafe { count_le_avx2(keys, key) }
        } else if sse42_available() {
            // SAFETY: `sse42_available` proves SSE4.2 at run time.
            unsafe { count_le_sse42(keys, key) }
        } else {
            scalar::count_le_u64(keys, key)
        }
    }

    #[must_use]
    pub(super) fn first_mismatch_u8(a: &[u8], b: &[u8], shared: usize) -> usize {
        if avx2_available() {
            // SAFETY: `avx2_available` proves AVX2 at run time.
            unsafe { first_mismatch_avx2(a, b, shared) }
        } else if sse2_available() {
            // SAFETY: `sse2_available` proves SSE2 at run time (always
            // true on x86_64).
            unsafe { first_mismatch_sse2(a, b, shared) }
        } else {
            scalar::first_mismatch_u8(a, b, shared)
        }
    }

    #[inline]
    fn avx512_available() -> bool {
        codevar_base::cpuid::has(codevar_base::cpuid::Feature::Avx512)
    }

    #[inline]
    fn avx2_available() -> bool {
        codevar_base::cpuid::has(codevar_base::cpuid::Feature::Avx2)
    }

    #[inline]
    fn sse42_available() -> bool {
        codevar_base::cpuid::has(codevar_base::cpuid::Feature::Sse42)
    }

    #[inline]
    fn sse2_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            true
        }
        #[cfg(target_arch = "x86")]
        {
            codevar_base::cpuid::has(codevar_base::cpuid::Feature::Sse2)
        }
    }

    /// AVX-512F count of lanes `< key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX-512F target feature.
    #[target_feature(enable = "avx512f")]
    unsafe fn count_less_avx512(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves AVX-512F;
        // the vector loads stay inside `keys` because
        // `chunks_exact(8)` guarantees full chunks; the scalar tail
        // never re-enters a SIMD kernel.
        unsafe {
            // Flipping the sign bit turns unsigned order into the
            // signed order `_mm512_cmp_epi64_mask` compares.
            let top = _mm512_set1_epi64(i64::MIN);
            let wanted = _mm512_set1_epi64((key ^ (1 << 63)) as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(8);
            for chunk in &mut chunks {
                let values = _mm512_xor_si512(_mm512_loadu_si512(chunk.as_ptr() as *const __m512i), top);
                let mask = _mm512_cmp_epi64_mask(values, wanted, _MM_CMPINT_LT);
                total += u64::from(mask.count_ones());
            }
            total + scalar::count_less_u64(chunks.remainder(), key)
        }
    }

    /// AVX-512F count of lanes `<= key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX-512F target feature.
    #[target_feature(enable = "avx512f")]
    unsafe fn count_le_avx512(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves AVX-512F;
        // the vector loads stay inside `keys` because
        // `chunks_exact(8)` guarantees full chunks; the scalar tail
        // never re-enters a SIMD kernel.
        unsafe {
            // Flipping the sign bit turns unsigned order into the
            // signed order `_mm512_cmp_epi64_mask` compares.
            let top = _mm512_set1_epi64(i64::MIN);
            let wanted = _mm512_set1_epi64((key ^ (1 << 63)) as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(8);
            for chunk in &mut chunks {
                let values = _mm512_xor_si512(_mm512_loadu_si512(chunk.as_ptr() as *const __m512i), top);
                let mask = _mm512_cmp_epi64_mask(values, wanted, _MM_CMPINT_LE);
                total += u64::from(mask.count_ones());
            }
            total + scalar::count_le_u64(chunks.remainder(), key)
        }
    }

    /// AVX2 count of lanes `< key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature.
    #[target_feature(enable = "avx2")]
    unsafe fn count_less_avx2(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves AVX2; the
        // vector loads stay inside `keys` because `chunks_exact(4)`
        // guarantees full chunks; the scalar tail never re-enters a
        // SIMD kernel.
        unsafe {
            // Flipping the sign bit turns unsigned order into the
            // signed order `_mm256_cmpgt_epi64` compares.
            let top = _mm256_set1_epi64x(i64::MIN);
            let wanted = _mm256_set1_epi64x((key ^ (1 << 63)) as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(4);
            for chunk in &mut chunks {
                let values = _mm256_xor_si256(_mm256_loadu_si256(chunk.as_ptr() as *const __m256i), top);
                // wanted > values  <=>  values < wanted; every true
                // lane is all-ones, so the byte movemask carries 8
                // bits per lane.
                let cmp = _mm256_cmpgt_epi64(wanted, values);
                total += u64::from(_mm256_movemask_epi8(cmp).count_ones() / 8);
            }
            total + scalar::count_less_u64(chunks.remainder(), key)
        }
    }

    /// AVX2 count of lanes `<= key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature.
    #[target_feature(enable = "avx2")]
    unsafe fn count_le_avx2(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves AVX2; the
        // vector loads stay inside `keys` because `chunks_exact(4)`
        // guarantees full chunks; the scalar tail never re-enters a
        // SIMD kernel.
        unsafe {
            // Flipping the sign bit turns unsigned order into the
            // signed order `_mm256_cmpgt_epi64` compares.
            let top = _mm256_set1_epi64x(i64::MIN);
            let wanted = _mm256_set1_epi64x((key ^ (1 << 63)) as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(4);
            for chunk in &mut chunks {
                let values = _mm256_xor_si256(_mm256_loadu_si256(chunk.as_ptr() as *const __m256i), top);
                let cmp = _mm256_cmpgt_epi64(values, wanted);
                // values > wanted is the complement of values <= wanted.
                let gt = _mm256_movemask_epi8(cmp).count_ones() / 8;
                total += u64::from(4 - gt);
            }
            total + scalar::count_le_u64(chunks.remainder(), key)
        }
    }

    /// SSE4.2 count of lanes `< key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the SSE4.2 target feature (which implies SSE2).
    #[target_feature(enable = "sse4.2")]
    unsafe fn count_less_sse42(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves SSE4.2;
        // the vector loads stay inside `keys` because
        // `chunks_exact(2)` guarantees full chunks; the scalar tail
        // never re-enters a SIMD kernel.
        unsafe {
            // Flipping the sign bit turns unsigned order into the
            // signed order `_mm_cmpgt_epi64` compares.
            let top = _mm_set1_epi64x(i64::MIN);
            let wanted = _mm_set1_epi64x((key ^ (1 << 63)) as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(2);
            for chunk in &mut chunks {
                let values = _mm_xor_si128(_mm_loadu_si128(chunk.as_ptr() as *const __m128i), top);
                let cmp = _mm_cmpgt_epi64(wanted, values);
                total += u64::from(_mm_movemask_epi8(cmp).count_ones() / 8);
            }
            total + scalar::count_less_u64(chunks.remainder(), key)
        }
    }

    /// SSE4.2 count of lanes `<= key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the SSE4.2 target feature (which implies SSE2).
    #[target_feature(enable = "sse4.2")]
    unsafe fn count_le_sse42(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves SSE4.2;
        // the vector loads stay inside `keys` because
        // `chunks_exact(2)` guarantees full chunks; the scalar tail
        // never re-enters a SIMD kernel.
        unsafe {
            // Flipping the sign bit turns unsigned order into the
            // signed order `_mm_cmpgt_epi64` compares.
            let top = _mm_set1_epi64x(i64::MIN);
            let wanted = _mm_set1_epi64x((key ^ (1 << 63)) as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(2);
            for chunk in &mut chunks {
                let values = _mm_xor_si128(_mm_loadu_si128(chunk.as_ptr() as *const __m128i), top);
                let cmp = _mm_cmpgt_epi64(values, wanted);
                let gt = _mm_movemask_epi8(cmp).count_ones() / 8;
                total += u64::from(2 - gt);
            }
            total + scalar::count_le_u64(chunks.remainder(), key)
        }
    }

    /// AVX2 32-byte block mismatch scan with an SSE2/scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature (which implies SSE2).
    #[target_feature(enable = "avx2")]
    unsafe fn first_mismatch_avx2(a: &[u8], b: &[u8], shared: usize) -> usize {
        // SAFETY: the `#[target_feature]` attribute proves AVX2 (and
        // therefore SSE2); `chunks_exact(32)` guarantees full blocks
        // and stays within `shared`; the tail scan never re-enters a
        // SIMD kernel.
        unsafe {
            let mut index = 0usize;
            let mut chunks_a = a[..shared].chunks_exact(32);
            let mut chunks_b = b[..shared].chunks_exact(32);
            for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
                let va = _mm256_loadu_si256(chunk_a.as_ptr() as *const __m256i);
                let vb = _mm256_loadu_si256(chunk_b.as_ptr() as *const __m256i);
                let eq = _mm256_cmpeq_epi8(va, vb);
                let bits = _mm256_movemask_epi8(eq) as u32;
                if bits != u32::MAX {
                    // One movemask bit per byte (set when equal); the
                    // first zero bit of `bits` is the mismatching byte
                    // index, found via the complement.
                    return index + ((!bits).trailing_zeros() as usize);
                }
                index += 32;
            }
            index + scalar::first_mismatch_u8(&a[index..shared], &b[index..shared], shared - index)
        }
    }

    /// SSE2 16-byte block mismatch scan with a scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the SSE2 target feature.
    #[target_feature(enable = "sse2")]
    unsafe fn first_mismatch_sse2(a: &[u8], b: &[u8], shared: usize) -> usize {
        // SAFETY: the `#[target_feature]` attribute proves SSE2; the
        // vector loads stay inside `a`/`b` because `chunks_exact(16)`
        // guarantees full blocks and stays within `shared`; the tail
        // scan never re-enters a SIMD kernel.
        unsafe {
            let mut index = 0usize;
            let mut chunks_a = a[..shared].chunks_exact(16);
            let mut chunks_b = b[..shared].chunks_exact(16);
            for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
                let va = _mm_loadu_si128(chunk_a.as_ptr() as *const __m128i);
                let vb = _mm_loadu_si128(chunk_b.as_ptr() as *const __m128i);
                let eq = _mm_cmpeq_epi8(va, vb);
                let bits = _mm_movemask_epi8(eq) as u32;
                if bits != 0xFFFF {
                    // One movemask bit per byte (set when equal); the
                    // first zero bit of the low 16 is the mismatching
                    // byte index, found via the complement.
                    return index + (((!bits) & 0xFFFF).trailing_zeros() as usize);
                }
                index += 16;
            }
            index + scalar::first_mismatch_u8(&a[index..shared], &b[index..shared], shared - index)
        }
    }
}

/// aarch64 Advanced SIMD (NEON) backend.
///
/// `vcltq_u64` compares two `u64` lanes per 128-bit vector; the lane
/// mask is narrowed to one bit per lane and widened back with
/// `vaddvq_u64`.
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::arch::aarch64::{
        vaddvq_u64, vceqq_u8, vcltq_u64, vdupq_n_u64, vld1q_u8, vld1q_u64, vminvq_u8, vshrq_n_u64,
    };

    use super::scalar;

    #[must_use]
    pub(super) fn count_less_u64(keys: &[u64], key: u64) -> u64 {
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return scalar::count_less_u64(keys, key);
        }
        // SAFETY: the probe above confirmed Advanced SIMD.
        unsafe { count_less_neon(keys, key) }
    }

    #[must_use]
    pub(super) fn count_le_u64(keys: &[u64], key: u64) -> u64 {
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return scalar::count_le_u64(keys, key);
        }
        // SAFETY: the probe above confirmed Advanced SIMD.
        unsafe { count_le_neon(keys, key) }
    }

    #[must_use]
    pub(super) fn first_mismatch_u8(a: &[u8], b: &[u8], shared: usize) -> usize {
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return scalar::first_mismatch_u8(a, b, shared);
        }
        // SAFETY: the probe above confirmed Advanced SIMD.
        unsafe { first_mismatch_neon(a, b, shared) }
    }

    /// NEON count of lanes `< key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the NEON target feature.
    #[target_feature(enable = "neon")]
    unsafe fn count_less_neon(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves NEON; the
        // vector loads stay inside `keys` because `chunks_exact(2)`
        // guarantees full chunks; the scalar tail never re-enters a
        // SIMD kernel.
        unsafe {
            let wanted = vdupq_n_u64(key);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(2);
            for chunk in &mut chunks {
                let values = vld1q_u64(chunk.as_ptr());
                // One bit per lane, widened to 0/1 and summed.
                let mask = vcltq_u64(values, wanted);
                let bits = vshrq_n_u64(mask, 63);
                total += vaddvq_u64(bits);
            }
            total + scalar::count_less_u64(chunks.remainder(), key)
        }
    }

    /// NEON count of lanes `<= key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the NEON target feature.
    #[target_feature(enable = "neon")]
    unsafe fn count_le_neon(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves NEON; the
        // vector loads stay inside `keys` because `chunks_exact(2)`
        // guarantees full chunks; the scalar tail never re-enters a
        // SIMD kernel.
        unsafe {
            let wanted = vdupq_n_u64(key);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(2);
            for chunk in &mut chunks {
                let values = vld1q_u64(chunk.as_ptr());
                // lanes <= key  <=>  !(key < lanes).
                let mask = vcltq_u64(wanted, values);
                let bits = vshrq_n_u64(mask, 63);
                total += 2 - vaddvq_u64(bits);
            }
            total + scalar::count_le_u64(chunks.remainder(), key)
        }
    }

    /// NEON 16-byte block mismatch scan with a scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the NEON target feature.
    #[target_feature(enable = "neon")]
    unsafe fn first_mismatch_neon(a: &[u8], b: &[u8], shared: usize) -> usize {
        // SAFETY: the `#[target_feature]` attribute proves NEON; the
        // vector loads stay inside `a`/`b` because `chunks_exact(16)`
        // guarantees full blocks and stays within `shared`; the tail
        // scan never re-enters a SIMD kernel.
        unsafe {
            let mut index = 0usize;
            let mut chunks_a = a[..shared].chunks_exact(16);
            let mut chunks_b = b[..shared].chunks_exact(16);
            for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
                let va = vld1q_u8(chunk_a.as_ptr());
                let vb = vld1q_u8(chunk_b.as_ptr());
                let eq = vceqq_u8(va, vb);
                // A fully equal block has no byte below 0xFF; the
                // reduction is branchless and the exact offset is
                // found by the scalar scan of just these 16 bytes.
                if vminvq_u8(eq) != 0xFF {
                    return index + scalar::first_mismatch_u8(&a[index..], &b[index..], 16);
                }
                index += 16;
            }
            index + scalar::first_mismatch_u8(&a[index..shared], &b[index..shared], shared - index)
        }
    }
}

/// wasm32 SIMD128 backend; compiled only when `simd128` is enabled
/// for the target (otherwise the scalar backend is used).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        i8x16_eq, i64x2_extract_lane, i64x2_lt_u, u8x16_bitmask, u64x2_splat, v128_load,
    };

    use super::scalar;

    pub(super) fn count_less_u64(keys: &[u64], key: u64) -> u64 {
        let wanted = u64x2_splat(key);
        let mut total = 0u64;
        let mut chunks = keys.chunks_exact(2);
        for chunk in &mut chunks {
            let values = v128_load(chunk.as_ptr() as *const v128);
            let mask = i64x2_lt_u(values, wanted);
            total += u64::from(i64x2_extract_lane::<0>(mask) != 0);
            total += u64::from(i64x2_extract_lane::<1>(mask) != 0);
        }
        total + scalar::count_less_u64(chunks.remainder(), key)
    }

    pub(super) fn count_le_u64(keys: &[u64], key: u64) -> u64 {
        let wanted = u64x2_splat(key);
        let mut total = 0u64;
        let mut chunks = keys.chunks_exact(2);
        for chunk in &mut chunks {
            let values = v128_load(chunk.as_ptr() as *const v128);
            // lanes <= key  <=>  !(key < lanes).
            let mask = i64x2_lt_u(wanted, values);
            let gt =
                u64::from(i64x2_extract_lane::<0>(mask) != 0) + u64::from(i64x2_extract_lane::<1>(mask) != 0);
            total += 2 - gt;
        }
        total + scalar::count_le_u64(chunks.remainder(), key)
    }

    pub(super) fn first_mismatch_u8(a: &[u8], b: &[u8], shared: usize) -> usize {
        let mut index = 0usize;
        let mut chunks_a = a[..shared].chunks_exact(16);
        let mut chunks_b = b[..shared].chunks_exact(16);
        for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
            let va = v128_load(chunk_a.as_ptr() as *const v128);
            let vb = v128_load(chunk_b.as_ptr() as *const v128);
            let bits = u8x16_bitmask(i8x16_eq(va, vb));
            if bits != 0xFFFF {
                return index + ((!bits) & 0xFFFF).trailing_zeros() as usize;
            }
            index += 16;
        }
        index + scalar::first_mismatch_u8(&a[index..shared], &b[index..shared], shared - index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: position in the stable merge sorted by
    /// `(key, side, index)` with `a` before `b` on equal keys.
    fn reference_ranks(a: &[u64], b: &[u64]) -> (Vec<u32>, Vec<u32>) {
        let mut items: Vec<(u64, u8, u32)> = Vec::new();
        for (i, &k) in a.iter().enumerate() {
            items.push((k, 0, i as u32));
        }
        for (j, &k) in b.iter().enumerate() {
            items.push((k, 1, j as u32));
        }
        items.sort_by(|x, y| {
            x.0.cmp(&y.0)
                .then(x.1.cmp(&y.1))
                .then(x.2.cmp(&y.2))
        });
        let mut ranks_a = vec![0u32; a.len()];
        let mut ranks_b = vec![0u32; b.len()];
        for (pos, &(_, side, idx)) in items.iter().enumerate() {
            if side == 0 {
                ranks_a[idx as usize] = pos as u32;
            } else {
                ranks_b[idx as usize] = pos as u32;
            }
        }
        (ranks_a, ranks_b)
    }

    fn check_merge(a: &[u64], b: &[u64]) {
        let mut ranks_a = vec![0u32; a.len()];
        let mut ranks_b = vec![0u32; b.len()];
        merge_rank_u64(a, b, &mut ranks_a, &mut ranks_b);
        let (want_a, want_b) = reference_ranks(a, b);
        assert_eq!(ranks_a, want_a, "ranks_a mismatch for a={a:?} b={b:?}");
        assert_eq!(ranks_b, want_b, "ranks_b mismatch for a={a:?} b={b:?}");
    }

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn edge_cases() {
        check_merge(&[], &[]);
        check_merge(&[1], &[]);
        check_merge(&[], &[2]);
        check_merge(&[1], &[1]);
        check_merge(&[1, 3, 5], &[]);
        check_merge(&[], &[2, 4, 6]);
        // All keys equal: a wins every tie.
        check_merge(&[7, 7, 7], &[7, 7]);
        // Sign-bit boundary values.
        check_merge(&[0, 1, 1 << 63, u64::MAX], &[0, 1 << 62, 1 << 63, (1 << 63) + 1]);
        // Small example from the doctest.
        check_merge(&[1, 3, 5], &[2, 3, 4]);
    }

    #[test]
    fn random_streams_with_ties() {
        let sizes = [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 257];
        let mut state = 0x243F_6A88_85A3_08D3u64;
        for &n in &sizes {
            for &m in &sizes {
                let mut a: Vec<u64> = (0..n)
                    .map(|_| xorshift64(&mut state) % 8)
                    .collect();
                let mut b: Vec<u64> = (0..m)
                    .map(|_| xorshift64(&mut state) % 8)
                    .collect();
                a.sort_unstable();
                b.sort_unstable();
                check_merge(&a, &b);
            }
        }
    }

    #[test]
    fn random_streams_full_range() {
        let mut state = 0x1319_8A2E_0370_7344u64;
        for _ in 0..32 {
            let n = (xorshift64(&mut state) % 200) as usize;
            let m = (xorshift64(&mut state) % 200) as usize;
            let mut a: Vec<u64> = (0..n).map(|_| xorshift64(&mut state)).collect();
            let mut b: Vec<u64> = (0..m).map(|_| xorshift64(&mut state)).collect();
            a.sort_unstable();
            b.sort_unstable();
            check_merge(&a, &b);
        }
    }

    #[test]
    fn length_mismatch_is_noop() {
        let a = [1u64, 2, 3];
        let b = [4u64, 5];
        let mut ranks_a = [9u32; 3];
        let mut ranks_b_short = [9u32; 1];
        merge_rank_u64(&a, &b, &mut ranks_a, &mut ranks_b_short);
        assert_eq!(ranks_a, [9, 9, 9]);
        assert_eq!(ranks_b_short, [9]);
    }

    #[test]
    fn count_less_matches_reference() {
        let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
        for _ in 0..64 {
            let n = (xorshift64(&mut state) % 100) as usize;
            let keys: Vec<u64> = (0..n)
                .map(|_| xorshift64(&mut state) % 16)
                .collect();
            let key = xorshift64(&mut state) % 16;
            let want = keys.iter().filter(|&&k| k < key).count() as u64;
            assert_eq!(count_less_u64(&keys, key), want, "n={n} key={key}");
        }
        assert_eq!(count_less_u64(&[1, 3, 5, 7], 5), 2);
        assert_eq!(count_less_u64(&[5, 5, 5], 5), 0);
        assert_eq!(count_less_u64(&[6, 7, 8], 5), 0);
        assert_eq!(count_less_u64(&[3, 4], 5), 2);
    }

    #[test]
    fn count_le_matches_reference() {
        let mut state = 0xA409_3822_299F_31D0u64;
        for _ in 0..64 {
            let n = (xorshift64(&mut state) % 100) as usize;
            let keys: Vec<u64> = (0..n)
                .map(|_| xorshift64(&mut state) % 16)
                .collect();
            let key = xorshift64(&mut state) % 16;
            let want = keys.iter().filter(|&&k| k <= key).count() as u64;
            assert_eq!(count_le_u64(&keys, key), want, "n={n} key={key}");
        }
        assert_eq!(count_le_u64(&[1, 3, 5, 7], 5), 3);
        assert_eq!(count_le_u64(&[5, 5, 5], 5), 3);
        assert_eq!(count_le_u64(&[6, 7, 8], 5), 0);
        assert_eq!(count_le_u64(&[3, 4], 5), 2);
    }

    #[test]
    fn first_mismatch_edge_cases() {
        assert_eq!(first_mismatch_u8(b"", b""), 0);
        assert_eq!(first_mismatch_u8(b"a", b""), 0);
        assert_eq!(first_mismatch_u8(b"", b"a"), 0);
        assert_eq!(first_mismatch_u8(b"same", b"same"), 4);
        assert_eq!(first_mismatch_u8(b"abcXde", b"abcYde"), 3);
        assert_eq!(first_mismatch_u8(b"prefix_longer", b"prefix"), 6);
        // Differences on block boundaries for 16/32-wide kernels.
        for offset in [1, 14, 15, 16, 17, 30, 31, 32, 33, 47, 48, 63, 64, 65] {
            let mut a = vec![0xAAu8; 96];
            let mut b = vec![0xAAu8; 96];
            b[offset] = 0x55;
            assert_eq!(first_mismatch_u8(&a, &b), offset, "offset={offset}");
            a[offset] = 0x11;
            assert_eq!(first_mismatch_u8(&a, &b), offset, "offset={offset}");
        }
    }

    #[test]
    fn first_mismatch_large_buffer() {
        let len = 4096 + 17;
        let a = vec![0x5Au8; len];
        let mut b = vec![0x5Au8; len];
        b[len - 1] = 0x77;
        assert_eq!(first_mismatch_u8(&a, &b), len - 1);
        let mut c = vec![0x5Au8; len];
        c[len - 2] = 0x00;
        assert_eq!(first_mismatch_u8(&a, &c), len - 2);
        assert_eq!(first_mismatch_u8(&a, &a), len);
        // Longer/shorter buffers only scan the shared prefix.
        assert_eq!(first_mismatch_u8(&a, &a[..100]), 100);
    }

    #[test]
    fn first_mismatch_random() {
        let mut state = 0x082E_FA98_EC4E_6C89u64;
        for _ in 0..64 {
            let len = (xorshift64(&mut state) % 300) as usize;
            let mut a: Vec<u8> = (0..len)
                .map(|_| (xorshift64(&mut state) % 251) as u8)
                .collect();
            let mut b = a.clone();
            let want = if len == 0 || xorshift64(&mut state).is_multiple_of(4) {
                len
            } else {
                let idx = (xorshift64(&mut state) as usize) % len;
                b[idx] = b[idx].wrapping_add(1);
                idx
            };
            assert_eq!(first_mismatch_u8(&a, &b), want, "len={len}");
            // Also verify against a differing prefix length.
            a.push(0);
            assert_eq!(first_mismatch_u8(&a, &b), want.min(len));
        }
    }
}
