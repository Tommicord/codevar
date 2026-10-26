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

//! SIMD kernels backing the TLSF allocator.
//!
//! Two primitives are provided, both with an AVX2 backend (x86/x86_64, run
//! time dispatched through [`codevar_base::cpuid`]), a NEON backend
//! (aarch64) and a portable scalar fallback:
//!
//! - [`find_first_set`] / [`find_first_set_from`] — index of the first set
//!   bit in a multi-word bitmap. The allocator uses this to scan the
//!   first-level free-class bitmap in O(1) and to walk larger bitmaps in
//!   four (AVX2) or two (NEON) words per iteration.
//! - [`fill_bytes`] — unconditional byte fill, used for `alloc_zeroed`
//!   (and available for heap painting). AVX2 stores 32 bytes per iteration,
//!   NEON 16.
//!
//! Every backend is bit-identical to the scalar reference; the detected
//! tier only affects performance, never results.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Returns the index of the first set bit in `words`, treating the slice as
/// one wide little-endian bitmap (bit `i * 64 + j` of the slice is bit `j`
/// of `words[i]`), or `None` when every word is zero.
///
/// # Performance
///
/// The AVX2 backend tests four words per iteration with a single compare
/// and move-mask; the NEON backend two. The scalar fallback costs one
/// branch per word and is used when no SIMD backend is available.
#[inline]
pub(crate) fn find_first_set(words: &[u64]) -> Option<u32> {
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            if codevar_base::cpu_feature!(avx2) {
                // SAFETY: `avx2` was positively probed above, which is
                // exactly the precondition of the `avx2` target feature.
                return unsafe { find_first_set_avx2(words) };
            }
        } else if #[cfg(target_arch = "aarch64")] {
            if codevar_base::cpu_feature!(neon) {
                // SAFETY: `neon` was positively probed above; the function
                // only uses NEON loads and compares with no preconditions
                // beyond the feature itself.
                return unsafe { find_first_set_neon(words) };
            }
        }
    }
    find_first_set_scalar(words)
}

/// Returns the index of the first set bit whose position is greater than or
/// equal to `start_bit`, or `None` when no such bit exists.
#[inline]
pub(crate) fn find_first_set_from(words: &[u64], start_bit: u32) -> Option<u32> {
    let start_word = (start_bit / 64) as usize;
    if start_word >= words.len() {
        return None;
    }
    let bit = start_bit % 64;
    // Mask off every bit below `start_bit` in the first candidate word.
    let first = words[start_word] & (!0u64 << bit);
    if first != 0 {
        // `trailing_zeros` is absolute within the word, so re-base it on
        // the word index, never on `start_bit`.
        return Some(start_word as u32 * 64 + first.trailing_zeros());
    }
    let rest = words.get(start_word + 1..)?;
    find_first_set(rest).map(|index| (start_word as u32 + 1) * 64 + index)
}

/// Fills `len` bytes at `dst` with `value`.
///
/// # Safety
///
/// The caller must guarantee that `dst` is valid for writes of `len` bytes
/// and that no read of that region is required for the duration of the call.
#[inline]
pub(crate) unsafe fn fill_bytes(dst: *mut u8, len: usize, value: u8) {
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            if codevar_base::cpu_feature!(avx2) {
                // SAFETY: the caller guarantees `dst` is valid for `len`
                // writes; `avx2` was positively probed above.
                unsafe { fill_bytes_avx2(dst, len, value) };
                return;
            }
        } else if #[cfg(target_arch = "aarch64")] {
            if codevar_base::cpu_feature!(neon) {
                // SAFETY: the caller guarantees `dst` is valid for `len`
                // writes; `neon` was positively probed above.
                unsafe { fill_bytes_neon(dst, len, value) };
                return;
            }
        }
    }
    // SAFETY: the caller guarantees `dst` is valid for `len` writes.
    unsafe { core::ptr::write_bytes(dst, value, len) };
}

/// Scalar reference implementation of [`find_first_set`].
#[inline]
fn find_first_set_scalar(words: &[u64]) -> Option<u32> {
    for (i, &word) in words.iter().enumerate() {
        if word != 0 {
            return Some(i as u32 * 64 + word.trailing_zeros());
        }
    }
    None
}

/// AVX2 backend of [`find_first_set`]: four words per iteration.
///
/// # Safety
///
/// The caller must have verified (at run time or at compile time) that the
/// CPU supports AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn find_first_set_avx2(words: &[u64]) -> Option<u32> {
    let mut i = 0;
    // SAFETY: the loop condition keeps loads inside the slice; `loadu`
    // accepts any alignment. All memory is owned by the caller's slice.
    unsafe {
        let zero = _mm256_setzero_si256();
        while i + 4 <= words.len() {
            let v = _mm256_loadu_si256(words.as_ptr().add(i) as *const __m256i);
            // Each byte of a matching 64-bit lane is all-ones, so
            // `movemask_epi8` yields 8 consecutive bits per zero word:
            // the mask is all-ones iff all four words are zero.
            let is_zero = _mm256_cmpeq_epi64(v, zero);
            let mask = _mm256_movemask_epi8(is_zero) as u32;
            if mask != u32::MAX {
                // First clear byte of `mask` = first non-zero lane, and
                // each lane contributes exactly 8 mask bits.
                let lane = (!mask).trailing_zeros() as usize / 8;
                let word = *words.get_unchecked(i + lane);
                return Some((i + lane) as u32 * 64 + word.trailing_zeros());
            }
            i += 4;
        }
    }
    find_first_set_scalar(&words[i..]).map(|index| i as u32 * 64 + index)
}

/// NEON backend of [`find_first_set`]: two words per iteration.
///
/// # Safety
///
/// The caller must have verified (at run time or at compile time) that the
/// CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_first_set_neon(words: &[u64]) -> Option<u32> {
    let mut i = 0;
    // SAFETY: the loop condition keeps loads inside the slice; `vld1q_u64`
    // accepts any alignment. All memory is owned by the caller's slice.
    unsafe {
        let zero = vdupq_n_u64(0);
        while i + 2 <= words.len() {
            let v = vld1q_u64(words.as_ptr().add(i));
            // Lane is all-ones iff the corresponding word is zero.
            let is_zero = vceqq_u64(v, zero);
            if vminvq_u64(is_zero) != u64::MAX {
                if vgetq_lane_u64(is_zero, 0) != u64::MAX {
                    let word = *words.get_unchecked(i);
                    return Some(i as u32 * 64 + word.trailing_zeros());
                }
                let word = *words.get_unchecked(i + 1);
                return Some((i + 1) as u32 * 64 + word.trailing_zeros());
            }
            i += 2;
        }
    }
    find_first_set_scalar(&words[i..]).map(|index| i as u32 * 64 + index)
}

/// AVX2 backend of [`fill_bytes`]: 32 bytes per iteration.
///
/// # Safety
///
/// Same requirements as [`fill_bytes`], plus the caller must have verified
/// that the CPU supports AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_bytes_avx2(dst: *mut u8, len: usize, value: u8) {
    // SAFETY: the caller guarantees `dst` is valid for `len` writes; the
    // loop bounds keep every store inside that region; `storeu` accepts any
    // alignment.
    unsafe {
        let mut i = 0;
        if len >= 32 {
            let splat = _mm256_set1_epi8(value as i8);
            while i + 32 <= len {
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, splat);
                i += 32;
            }
        }
        core::ptr::write_bytes(dst.add(i), value, len - i);
    }
}

/// NEON backend of [`fill_bytes`]: 16 bytes per iteration.
///
/// # Safety
///
/// Same requirements as [`fill_bytes`], plus the caller must have verified
/// that the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_bytes_neon(dst: *mut u8, len: usize, value: u8) {
    // SAFETY: the caller guarantees `dst` is valid for `len` writes; the
    // loop bounds keep every store inside that region; `vst1q_u8` accepts
    // any alignment.
    unsafe {
        let mut i = 0;
        if len >= 16 {
            let splat = vdupq_n_u8(value);
            while i + 16 <= len {
                vst1q_u8(dst.add(i), splat);
                i += 16;
            }
        }
        core::ptr::write_bytes(dst.add(i), value, len - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar oracle for [`find_first_set_from`].
    fn reference(words: &[u64], start_bit: u32) -> Option<u32> {
        (start_bit..words.len() as u32 * 64)
            .find(|&bit| words[(bit / 64) as usize] & (1u64 << (bit % 64)) != 0)
    }

    #[test]
    fn first_set_matches_reference() {
        let cases: &[&[u64]] = &[
            &[],
            &[0],
            &[0, 0, 0, 0],
            &[0b1000],
            &[0, 0, 1 << 63],
            &[0, u64::MAX, 0],
            &[0; 9],
            &[0; 9],
            &[0, 0, 0b1010, 0, 1],
        ];
        for words in cases {
            for start in 0..=(words.len() as u32 * 64) {
                assert_eq!(
                    find_first_set_from(words, start),
                    reference(words, start),
                    "words={words:?} start={start}"
                );
            }
        }
    }

    #[test]
    fn first_set_finds_late_bits_after_long_zero_runs() {
        let mut words = [0u64; 40];
        words[37] = 1 << 9;
        assert_eq!(find_first_set(&words), Some(37 * 64 + 9));
        assert_eq!(find_first_set_from(&words, 37 * 64), Some(37 * 64 + 9));
        assert_eq!(find_first_set_from(&words, 37 * 64 + 10), None);
        assert_eq!(find_first_set_from(&words, 37 * 64 + 9), Some(37 * 64 + 9));
    }

    #[test]
    fn fill_matches_std_memset() {
        for &len in &[0usize, 1, 15, 16, 31, 32, 33, 63, 64, 65, 1000] {
            for &value in &[0u8, 0xAA, 0xFF] {
                let mut actual = vec![0u8; len];
                let mut expected = vec![0u8; len];
                // SAFETY: `actual` is valid for `len` writes.
                unsafe { fill_bytes(actual.as_mut_ptr(), len, value) };
                expected.fill(value);
                assert_eq!(actual, expected, "len={len} value={value:#x}");
            }
        }
    }
}
