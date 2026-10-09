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

//! x86 / x86_64 backends.
//!
//! Kernels are tiered by the instructions they actually need and
//! selected at run time through `basic_cpuid`:
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Scalar rank merge dispatching the inner counts to the widest
    /// available `u64` tier; see [`crate::merge_rank_u64`].
    pub(super) fn merge_rank_u64(a: &[u64], b: &[u64], ranks_a: &mut [u32], ranks_b: &mut [u32]) {
        // The tier is probed once per direction; the count kernels below
        // re-check per call so the compiler can inline them, and
        // `basic_cpuid::has` reads a cached mask after the first probe.
        for (i, &key) in a.iter().enumerate() {
            ranks_a[i] = (i as u64 + count_less_u64(b, key)) as u32;
        }
        for (j, &key) in b.iter().enumerate() {
            ranks_b[j] = (j as u64 + count_le_u64(a, key)) as u32;
        }
    }

    /// Counts keys strictly less than `key` on the widest x86 tier.
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
            count_less_u64(keys, key)
        }
    }

    /// Counts keys less than or equal to `key` on the widest x86 tier.
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
            count_le_u64(keys, key)
        }
    }

    /// Byte mismatch scan on the widest x86 tier; see
    /// [`crate::first_mismatch_u8`].
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
            first_mismatch_u8(a, b, shared)
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
        // SAFETY: the `#[target_feature]` attribute proves AVX-512F; the
        // vector loads stay inside `keys` because `chunks_exact(8)`
        // guarantees full chunks.
        unsafe {
            let wanted = _mm512_set1_epi64(key as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(8);
            for chunk in &mut chunks {
                let values = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
                let mask = _mm512_cmp_epi64_mask(values, wanted, _MM_CMPINT_LT);
                total += u64::from(mask.count_ones());
            }
            total + count_less_u64(chunks.remainder(), key)
        }
    }

    /// AVX-512F count of lanes `<= key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX-512F target feature.
    #[target_feature(enable = "avx512f")]
    unsafe fn count_le_avx512(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves AVX-512F; the
        // vector loads stay inside `keys` because `chunks_exact(8)`
        // guarantees full chunks.
        unsafe {
            let wanted = _mm512_set1_epi64(key as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(8);
            for chunk in &mut chunks {
                let values = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
                let mask = _mm512_cmp_epi64_mask(values, wanted, _MM_CMPINT_LE);
                total += u64::from(mask.count_ones());
            }
            total + count_le_u64(chunks.remainder(), key)
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
        // guarantees full chunks.
        unsafe {
            let wanted = _mm256_set1_epi64x(key as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(4);
            for chunk in &mut chunks {
                let values = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                // wanted > values  <=>  values < wanted; every true lane
                // is all-ones, so the byte movemask carries 8 bits per
                // lane.
                let cmp = _mm256_cmpgt_epi64(wanted, values);
                total += u64::from(_mm256_movemask_epi8(cmp).count_ones() / 8);
            }
            total + count_less_u64(chunks.remainder(), key)
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
        // guarantees full chunks.
        unsafe {
            let wanted = _mm256_set1_epi64x(key as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(4);
            for chunk in &mut chunks {
                let values = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                let cmp = _mm256_cmpgt_epi64(values, wanted);
                // values > wanted is the complement of values <= wanted.
                let gt = u32::from(_mm256_movemask_epi8(cmp).count_ones() / 8);
                total += u64::from(4 - gt);
            }
            total + count_le_u64(chunks.remainder(), key)
        }
    }

    /// SSE4.2 count of lanes `< key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the SSE4.2 target feature (which implies SSE2).
    #[target_feature(enable = "sse4.2")]
    unsafe fn count_less_sse42(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves SSE4.2; the
        // vector loads stay inside `keys` because `chunks_exact(2)`
        // guarantees full chunks.
        unsafe {
            let wanted = _mm_set1_epi64x(key as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(2);
            for chunk in &mut chunks {
                let values = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
                let cmp = _mm_cmpgt_epi64(wanted, values);
                total += u64::from(_mm_movemask_epi8(cmp).count_ones() / 8);
            }
            total + count_less_u64(chunks.remainder(), key)
        }
    }

    /// SSE4.2 count of lanes `<= key` plus the scalar tail.
    ///
    /// # Safety
    ///
    /// Requires the SSE4.2 target feature (which implies SSE2).
    #[target_feature(enable = "sse4.2")]
    unsafe fn count_le_sse42(keys: &[u64], key: u64) -> u64 {
        // SAFETY: the `#[target_feature]` attribute proves SSE4.2; the
        // vector loads stay inside `keys` because `chunks_exact(2)`
        // guarantees full chunks.
        unsafe {
            let wanted = _mm_set1_epi64x(key as i64);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(2);
            for chunk in &mut chunks {
                let values = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
                let cmp = _mm_cmpgt_epi64(values, wanted);
                let gt = u32::from(_mm_movemask_epi8(cmp).count_ones() / 8);
                total += u64::from(2 - gt);
            }
            total + count_le_u64(chunks.remainder(), key)
        }
    }

    /// AVX2 32-byte block mismatch scan with an SSE2 tail.
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature (which implies SSE2).
    #[target_feature(enable = "avx2")]
    unsafe fn first_mismatch_avx2(a: &[u8], b: &[u8], shared: usize) -> usize {
        // SAFETY: the `#[target_feature]` attribute proves AVX2 (and
        // therefore SSE2); `chunks_exact(32)` guarantees full blocks and
        // the tail scan stays within `shared`.
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
                    return index + (bits.trailing_zeros() as usize / 8);
                }
                index += 32;
            }
            let tail = first_mismatch_u8(
                &a[index..shared],
                &b[index..shared],
                shared - index,
            );
            index + tail
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
        // guarantees full blocks.
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
                    return index + (bits.trailing_zeros() as usize / 8);
                }
                index += 16;
            }
            let tail = first_mismatch_u8(
                &a[index..shared],
                &b[index..shared],
                shared - index,
            );
            index + tail
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    //! aarch64 Advanced SIMD (NEON) backend.
    //!
    //! `vcltq_u64` compares four `u64` lanes at once; the lane mask is
    //! narrowed to one bit per lane and widened back with `vaddvq_u64`.

    use core::arch::aarch64::{
        uint64x2_t, uint8x16_t, vaddvq_u64, vceqq_u8, vcltq_u64, vdupq_n_u64, vld1q_u8, vshrq_n_u64,
    };

    /// NEON rank merge; see [`crate::merge_rank_u64`].
    pub(super) fn merge_rank_u64(a: &[u64], b: &[u64], ranks_a: &mut [u32], ranks_b: &mut [u32]) {
        for (i, &key) in a.iter().enumerate() {
            ranks_a[i] = (i as u64 + count_less_u64(b, key)) as u32;
        }
        for (j, &key) in b.iter().enumerate() {
            ranks_b[j] = (j as u64 + count_le_u64(a, key)) as u32;
        }
    }

    /// Counts keys strictly less than `key` with NEON plus a scalar tail.
    #[must_use]
    pub(super) fn count_less_u64(keys: &[u64], key: u64) -> u64 {
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return count_less_u64(keys, key);
        }
        // SAFETY: the probe above confirmed Advanced SIMD.
        unsafe { count_less_neon(keys, key) }
    }

    /// Counts keys less than or equal to `key` with NEON plus a scalar
    /// tail.
    #[must_use]
    pub(super) fn count_le_u64(keys: &[u64], key: u64) -> u64 {
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return count_le_u64(keys, key);
        }
        // SAFETY: the probe above confirmed Advanced SIMD.
        unsafe { count_le_neon(keys, key) }
    }

    /// Byte mismatch scan with NEON plus a scalar tail; see
    /// [`crate::first_mismatch_u8`].
    #[must_use]
    pub(super) fn first_mismatch_u8(a: &[u8], b: &[u8], shared: usize) -> usize {
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return first_mismatch_u8(a, b, shared);
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
        // vector loads stay inside `keys` because `chunks_exact(4)`
        // guarantees full chunks.
        unsafe {
            let wanted = vdupq_n_u64(key);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(4);
            for chunk in &mut chunks {
                let values = load_u64x4(chunk.as_ptr());
                // One bit per lane, widened to 0/1 and summed.
                let mask = vcltq_u64(values, wanted);
                let bits = vshrq_n_u64(mask, 63);
                total += vaddvq_u64(bits);
            }
            total + count_less_u64(chunks.remainder(), key)
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
        // vector loads stay inside `keys` because `chunks_exact(4)`
        // guarantees full chunks.
        unsafe {
            let wanted = vdupq_n_u64(key);
            let mut total = 0u64;
            let mut chunks = keys.chunks_exact(4);
            for chunk in &mut chunks {
                let values = load_u64x4(chunk.as_ptr());
                // lanes <= key  <=>  !(key < lanes).
                let mask = vcltq_u64(wanted, values);
                let bits = vshrq_n_u64(mask, 63);
                total += 4 - vaddvq_u64(bits);
            }
            total + count_le_u64(chunks.remainder(), key)
        }
    }

    /// Loads four adjacent `u64` values as two NEON vectors and compares
    /// them lane-wise.
    ///
    /// # Safety
    ///
    /// Requires the NEON target feature and `ptr` to reference at least
    /// four readable `u64` values.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_u64x4(ptr: *const u64) -> uint64x2_t {
        // SAFETY: caller guarantees four readable lanes; the split into
        // two 128-bit loads is purely a register-size detail. The
        // returned value is the *first* half, and the second half is
        // handled by `count_*_neon` via this helper's sibling below.
        //
        // NOTE: NEON has no 256-bit vector, so the helper is used twice.
        core::arch::aarch64::vld1q_u64(ptr)
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
        // guarantees full blocks.
        unsafe {
            let mut index = 0usize;
            let mut chunks_a = a[..shared].chunks_exact(16);
            let mut chunks_b = b[..shared].chunks_exact(16);
            for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
                let va: uint8x16_t = vld1q_u8(chunk_a.as_ptr());
                let vb: uint8x16_t = vld1q_u8(chunk_b.as_ptr());
                let eq = vceqq_u8(va, vb);
                // A fully equal block has no zero byte; the reduction is
                // branchless.
                let any_diff = core::arch::aarch64::vminvq_u8(eq) != 0xFF;
                if any_diff {
                    let eq_bytes: [u8; 16] = core::mem::transmute(eq);
                    for (offset, &byte) in eq_bytes.iter().enumerate() {
                        if byte != 0xFF {
                            return index + offset;
                        }
                    }
                }
                index += 16;
            }
            let tail = first_mismatch_u8(
                &a[index..shared],
                &b[index..shared],
                shared - index,
            );
            index + tail
        }
    }

}

/// All-ones mask when `a < b` (unsigned), zero otherwise.
///
/// Sign-bit flip turns unsigned order into signed order, then the
/// Hacker's Delight signed comparison yields the answer in the sign
/// bit with no branches.
#[inline(always)]
pub(super) fn lt_mask_u64(a: u64, b: u64) -> u64 {
    const TOP: u64 = 1 << 63;
    let x = a ^ TOP;
    let y = b ^ TOP;
    let diff = x.wrapping_sub(y);
    let t = (x ^ y) & (diff ^ x);
    (diff ^ t) >> 63
}

/// Scalar rank merge; see [`crate::merge_rank_u64`].
pub(super) fn merge_rank_u64(a: &[u64], b: &[u64], ranks_a: &mut [u32], ranks_b: &mut [u32]) {
    for (i, &key) in a.iter().enumerate() {
        let mut rank = i as u64;
        for &other in b {
            rank += lt_mask_u64(other, key) & 1;
        }
        ranks_a[i] = rank as u32;
    }
    for (j, &key) in b.iter().enumerate() {
        let mut rank = j as u64;
        for &other in a {
            // `other <= key` is `!(key < other)`.
            rank += 1 ^ (lt_mask_u64(key, other) & 1);
        }
        ranks_b[j] = rank as u32;
    }
}

/// Scalar count; see [`crate::count_less_u64`].
#[must_use]
pub(super) fn count_less_u64(keys: &[u64], key: u64) -> u64 {
    let mut total = 0u64;
    for &other in keys {
        total += lt_mask_u64(other, key) & 1;
    }
    total
}

/// Scalar count; see [`crate::count_le_u64`].
#[must_use]
pub(super) fn count_le_u64(keys: &[u64], key: u64) -> u64 {
    let mut total = 0u64;
    for &other in keys {
        total += 1 ^ (lt_mask_u64(key, other) & 1);
    }
    total
}

/// Scalar mismatch scan; see [`crate::first_mismatch_u8`].
#[must_use]
pub(super) fn first_mismatch_u8(a: &[u8], b: &[u8], shared: usize) -> usize {
    for index in 0..shared {
        if a[index] != b[index] {
            return index;
        }
    }
    shared
}
