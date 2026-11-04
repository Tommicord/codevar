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

//! Vectorised newline scans over the packed raw document.
//!
//! Cursor navigation ([`super::cursor`]) and the line-width query
//! ([`op_buffer_query`]) used to walk the document one
//! packed element at a time. A newline test is a mask plus a compare,
//! so eight or four elements fit in one SIMD comparison; the kernels
//! below dispatch at run time through [`codevar_base::cpuid`] and fall
//! back to a scalar loop on every other target.
//!
//! All functions take a raw pointer so they can be used from both the
//! cursor's pointer-based view and the raw query slice. The caller
//! guarantees the pointer is valid for `len` initialized elements.

use crate::edit::base::CHAR_MASK;

/// Packed character value of a newline.
const NEWLINE: u32 = b'\n' as u32;

/// Counts newline elements in `raw[..len]`.
///
/// # Safety
///
/// `raw` must be valid for reads of `len` initialized `Raw` elements
/// (or be null when `len == 0`).
#[inline]
pub(crate) unsafe fn count_newlines<Raw: Copy + Into<u32>>(raw: *const Raw, len: usize) -> usize {
    if len == 0 || raw.is_null() {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Spill-free hand-written kernels. They are assembled for
        // SSE2/AVX2/AVX-512 and dispatch through the cached CPU level.
        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            // SAFETY: the caller guarantees `raw[..len]`; the kernel
            // only reads inside that range.
            return unsafe { crate::arch::x86_64::newline_count(raw.cast::<u32>(), len) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        // A `Raw` of exactly one word has the packed layout the SIMD
        // kernels compare; other instantiations use the scalar path.
        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            if has(Feature::Avx2) {
                // SAFETY: the probe confirmed AVX2 and the caller
                // guarantees `raw` is valid for `len` elements; the
                // kernel only reads within that range.
                return unsafe { x86::count_avx2(raw.cast::<u32>(), len) };
            }
            if has(Feature::Sse2) {
                // SAFETY: the probe confirmed SSE2; loads stay inside
                // `raw[..len]`.
                return unsafe { x86::count_sse2(raw.cast::<u32>(), len) };
            }
        }
    }
    // The other assembled targets expose the same masked-word
    // primitives through `arch::rout`; use them instead of the scalar
    // oracle so navigation benefits from the hand-written kernels on
    // ARM and RISC-V too.
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv32",
        target_arch = "riscv64"
    ))]
    {
        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            // SAFETY: the caller guarantees `raw[..len]`.
            return unsafe {
                crate::arch::rout::codevar_wredit_k_count_eq_u32(raw.cast::<u32>(), NEWLINE, CHAR_MASK, len)
            };
        }
    }
    // SAFETY: the caller guarantees `raw` is valid for `len` elements.
    unsafe { count_scalar(raw, len) }
}

/// Index of the first newline at or after `from`, within `raw[..len]`.
///
/// `from` is clamped to `len`.
///
/// # Safety
///
/// Same contract as [`count_newlines`].
#[inline]
pub(crate) unsafe fn find_newline_forward<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    from: usize,
) -> Option<usize> {
    if len == 0 || raw.is_null() {
        return None;
    }
    let from = from.min(len);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            if has(Feature::Avx2) {
                // SAFETY: see `count_newlines`; every load is bounded
                // by `len`.
                return unsafe { x86::find_forward_avx2(raw.cast::<u32>(), len, from) };
            }
            if has(Feature::Sse2) {
                // SAFETY: see `count_newlines`.
                return unsafe { x86::find_forward_sse2(raw.cast::<u32>(), len, from) };
            }
        }
    }
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv32",
        target_arch = "riscv64"
    ))]
    {
        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            // SAFETY: the caller guarantees `raw[..len]` and
            // `from <= len`; the kernel only reads that range.
            let found = unsafe {
                crate::arch::rout::codevar_wredit_k_find_eq_u32(
                    raw.add(from).cast::<u32>(),
                    NEWLINE,
                    CHAR_MASK,
                    len - from,
                )
            };
            if found.is_null() {
                return None;
            }
            return Some(from + (found as usize - raw as usize) / core::mem::size_of::<u32>());
        }
    }
    // SAFETY: the caller guarantees `raw` is valid for `len` elements.
    unsafe { find_forward_scalar(raw, len, from) }
}

/// Index of the last newline before `before`, within `raw[..len]`.
///
/// `before` is clamped to `len`, so the newline element at index
/// `before` itself is not considered.
///
/// # Safety
///
/// Same contract as [`count_newlines`].
#[inline]
pub(crate) unsafe fn find_newline_backward<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    before: usize,
) -> Option<usize> {
    if len == 0 || raw.is_null() {
        return None;
    }
    let before = before.min(len);
    if before == 0 {
        return None;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            if has(Feature::Avx2) {
                // SAFETY: see `count_newlines`; the backward kernel
                // only loads complete blocks ending at or below
                // `before`.
                return unsafe { x86::find_backward_avx2(raw.cast::<u32>(), before) };
            }
            if has(Feature::Sse2) {
                // SAFETY: see `count_newlines`.
                return unsafe { x86::find_backward_sse2(raw.cast::<u32>(), before) };
            }
        }
    }
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv32",
        target_arch = "riscv64"
    ))]
    {
        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            // SAFETY: the caller guarantees `raw[..len]` and the
            // kernel is bounded by `before <= len`.
            let found = unsafe {
                crate::arch::rout::codevar_wredit_k_find_eq_u32_rev(
                    raw.cast::<u32>(),
                    NEWLINE,
                    CHAR_MASK,
                    before,
                )
            };
            if found.is_null() {
                return None;
            }
            return Some((found as usize - raw as usize) / core::mem::size_of::<u32>());
        }
    }
    // SAFETY: the caller guarantees `raw` is valid for `len` elements.
    unsafe { find_backward_scalar(raw, before) }
}

/// Collects up to `out.len()` newline indices at or after `from`, in
/// ascending order, and returns how many were written.
///
/// # Safety
///
/// `raw` must be valid for reads of `len` elements and `from <= len`.
#[inline]
pub(crate) unsafe fn collect_newlines_forward<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    from: usize,
    out: &mut [u32],
) -> usize {
    if out.is_empty() || len == 0 || raw.is_null() {
        return 0;
    }
    let from = from.min(len);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() && has(Feature::Avx2) {
            // SAFETY: the caller guarantees `raw[..len]` and
            // `from <= len`; the kernel only reads that range.
            return unsafe { x86::collect_forward_avx2(raw.cast::<u32>(), len, from, out) };
        }
    }
    let mut count = 0;
    let mut index = from;
    while index < len && count < out.len() {
        // SAFETY: `index < len` and the caller guarantees the range.
        let word = unsafe { core::ptr::read_unaligned(raw.add(index)) };
        if (word.into() & CHAR_MASK) == NEWLINE {
            out[count] = index as u32;
            count += 1;
        }
        index += 1;
    }
    count
}

/// Collects up to `out.len()` newline indices below `before`, in
/// descending order, and returns how many were written.
///
/// # Safety
///
/// `raw` must be valid for reads of `len` elements and `before <= len`.
#[inline]
pub(crate) unsafe fn collect_newlines_backward<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    before: usize,
    out: &mut [u32],
) -> usize {
    if out.is_empty() || len == 0 || raw.is_null() || before == 0 {
        return 0;
    }
    let before = before.min(len);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() && has(Feature::Avx2) {
            // SAFETY: the caller guarantees `raw[..len]` and
            // `before <= len`; the kernel only reads that range.
            return unsafe { x86::collect_backward_avx2(raw.cast::<u32>(), before, out) };
        }
    }
    let mut count = 0;
    let mut index = before;
    while index > 0 && count < out.len() {
        index -= 1;
        // SAFETY: `index < before <= len`.
        let word = unsafe { core::ptr::read_unaligned(raw.add(index)) };
        if (word.into() & CHAR_MASK) == NEWLINE {
            out[count] = index as u32;
            count += 1;
        }
    }
    count
}

/// Index of the `n`-th newline (one-based) in `raw[..len]`.
///
/// Returns [`None`] when fewer than `n` newlines exist. `n == 0` is
/// reported as [`None`].
///
/// # Safety
///
/// Same contract as [`count_newlines`].
#[inline]
pub(crate) unsafe fn find_nth_newline<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    n: usize,
) -> Option<usize> {
    if n == 0 || len == 0 || raw.is_null() {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            // SAFETY: see `count_newlines`; loads stay inside
            // `raw[..len]`.
            return unsafe { crate::arch::x86_64::newline_nth(raw.cast::<u32>(), len, n) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            if has(Feature::Avx2) {
                // SAFETY: see `count_newlines`; loads stay inside
                // `raw[..len]`.
                return unsafe { x86::nth_avx2(raw.cast::<u32>(), len, n) };
            }
            if has(Feature::Sse2) {
                // SAFETY: see `count_newlines`.
                return unsafe { x86::nth_sse2(raw.cast::<u32>(), len, n) };
            }
        }
    }
    // SAFETY: the caller guarantees `raw` is valid for `len` elements.
    unsafe { nth_scalar(raw, len, n) }
}

/// Start and end element of the line containing `position`.
///
/// Unlike two independent [`find_newline_backward`] /
/// [`find_newline_forward`] calls, this performs the CPUID dispatch
/// once and lets the two vector loops inline into a single function.
/// The end is exclusive and clamps to `len`.
///
/// # Safety
///
/// Same contract as [`count_newlines`].
#[inline]
pub(crate) unsafe fn line_bounds<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    position: usize,
) -> (usize, usize) {
    let position = position.min(len);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() {
            if has(Feature::Avx2) {
                // SAFETY: see `count_newlines`; both scans stay inside
                // `raw[..len]`.
                return unsafe { x86::line_bounds_avx2(raw.cast::<u32>(), len, position) };
            }
            if has(Feature::Sse2) {
                // SAFETY: see `count_newlines`.
                return unsafe { x86::line_bounds_sse2(raw.cast::<u32>(), len, position) };
            }
        }
    }
    let start = unsafe { find_newline_backward(raw, len, position) }.map_or(0, |newline| newline + 1);
    let end = unsafe { find_newline_forward(raw, len, position) }.unwrap_or(len);
    (start, end)
}

/// `true` when a packed element holds a newline.
#[inline]
fn is_newline_word(word: u32) -> bool {
    (word & CHAR_MASK) == NEWLINE
}

/// Scalar fallback for [`count_newlines`].
///
/// # Safety
///
/// `raw` must be valid for reads of `len` initialized `Raw` elements.
#[inline]
unsafe fn count_scalar<Raw: Copy + Into<u32>>(raw: *const Raw, len: usize) -> usize {
    let mut count = 0;
    let mut index = 0;
    while index < len {
        // SAFETY: `index < len` and the caller guarantees the range is
        // initialized.
        let word: u32 = unsafe { core::ptr::read_unaligned(raw.add(index)) }.into();
        if is_newline_word(word) {
            count += 1;
        }
        index += 1;
    }
    count
}

/// Scalar fallback for [`find_newline_forward`].
///
/// # Safety
///
/// `raw` must be valid for reads of `len` initialized `Raw` elements
/// and `from <= len`.
#[inline]
unsafe fn find_forward_scalar<Raw: Copy + Into<u32>>(
    raw: *const Raw,
    len: usize,
    from: usize,
) -> Option<usize> {
    let mut index = from;
    while index < len {
        // SAFETY: `index < len`.
        let word: u32 = unsafe { core::ptr::read_unaligned(raw.add(index)) }.into();
        if is_newline_word(word) {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Scalar fallback for [`find_newline_backward`].
///
/// # Safety
///
/// `raw` must be valid for reads of `before` initialized `Raw`
/// elements.
#[inline]
unsafe fn find_backward_scalar<Raw: Copy + Into<u32>>(raw: *const Raw, before: usize) -> Option<usize> {
    let mut index = before;
    while index > 0 {
        index -= 1;
        // SAFETY: `index < before`, which is within the caller's
        // valid range.
        let word: u32 = unsafe { core::ptr::read_unaligned(raw.add(index)) }.into();
        if is_newline_word(word) {
            return Some(index);
        }
    }
    None
}

/// Scalar fallback for [`find_nth_newline`].
///
/// # Safety
///
/// `raw` must be valid for reads of `len` initialized `Raw` elements.
#[inline]
unsafe fn nth_scalar<Raw: Copy + Into<u32>>(raw: *const Raw, len: usize, n: usize) -> Option<usize> {
    let mut remaining = n;
    let mut index = 0;
    while index < len {
        // SAFETY: `index < len`.
        let word: u32 = unsafe { core::ptr::read_unaligned(raw.add(index)) }.into();
        if is_newline_word(word) {
            remaining -= 1;
            if remaining == 0 {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

/// Index of the `n`-th set bit of `mask` (`n` is one-based).
#[inline]
fn nth_bit(mask: u32, n: usize) -> Option<usize> {
    let mut bits = mask;
    let mut remaining = n;
    while remaining > 1 {
        if bits == 0 {
            return None;
        }
        bits &= bits - 1;
        remaining -= 1;
    }
    if bits == 0 {
        None
    } else {
        Some(bits.trailing_zeros() as usize)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use super::{CHAR_MASK, NEWLINE, nth_bit};

    /// Masked compare of eight packed words against a newline.
    ///
    /// # Safety
    ///
    /// Requires AVX2. `raw` must be valid for reads of `len` words and
    /// `at + 8 <= len`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare8(raw: *const u32, at: usize) -> u32 {
        // SAFETY: the caller bounds the load.
        unsafe {
            let words = _mm256_loadu_si256(raw.add(at).cast());
            let chars = _mm256_and_si256(words, _mm256_set1_epi32(CHAR_MASK as i32));
            let newlines = _mm256_cmpeq_epi32(chars, _mm256_set1_epi32(NEWLINE as i32));
            _mm256_movemask_ps(_mm256_castsi256_ps(newlines)) as u32
        }
    }

    /// Masked compare of four packed words against a newline.
    ///
    /// # Safety
    ///
    /// Requires SSE2. `raw` must be valid for reads of `len` words and
    /// `at + 4 <= len`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn compare4(raw: *const u32, at: usize) -> u32 {
        // SAFETY: the caller bounds the load.
        unsafe {
            let words = _mm_loadu_si128(raw.add(at).cast());
            let chars = _mm_and_si128(words, _mm_set1_epi32(CHAR_MASK as i32));
            let newlines = _mm_cmpeq_epi32(chars, _mm_set1_epi32(NEWLINE as i32));
            _mm_movemask_ps(_mm_castsi128_ps(newlines)) as u32
        }
    }

    /// AVX2 collector for newline indices in ascending order.
    ///
    /// # Safety
    ///
    /// Requires AVX2, a valid `raw[..len]` and `from <= len`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn collect_forward_avx2(
        raw: *const u32,
        len: usize,
        from: usize,
        out: &mut [u32],
    ) -> usize {
        let capacity = out.len();
        let mut count = 0;
        let mut index = from;
        // SAFETY: every `compare8` call is bounded by `len`.
        unsafe {
            while index + 8 <= len && count < capacity {
                if index & 8 == 0 {
                    // SAFETY: the helper only prefetches in range.
                }
                let mut mask = compare8(raw, index);
                while mask != 0 && count < capacity {
                    let bit = mask.trailing_zeros() as usize;
                    out[count] = (index + bit) as u32;
                    count += 1;
                    mask &= mask - 1;
                }
                index += 8;
            }
            while index < len && count < capacity {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    out[count] = index as u32;
                    count += 1;
                }
                index += 1;
            }
        }
        count
    }

    /// AVX2 collector for newline indices in descending order.
    ///
    /// # Safety
    ///
    /// Requires AVX2, a valid `raw[..len]` and `before <= len`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn collect_backward_avx2(raw: *const u32, before: usize, out: &mut [u32]) -> usize {
        let capacity = out.len();
        let mut count = 0;
        let mut end = before;
        // SAFETY: every `compare8` call ends at or below `before`.
        unsafe {
            while end >= 8 && count < capacity {
                if end & 8 == 0 {
                    // SAFETY: the helper only prefetches in range.
                }
                let mut mask = compare8(raw, end - 8);
                while mask != 0 && count < capacity {
                    let bit = 31 - mask.leading_zeros() as usize;
                    out[count] = (end - 8 + bit) as u32;
                    count += 1;
                    mask &= !(1u32 << bit);
                }
                end -= 8;
            }
            while end > 0 && count < capacity {
                end -= 1;
                let word = core::ptr::read_unaligned(raw.add(end));
                if (word & CHAR_MASK) == NEWLINE {
                    out[count] = end as u32;
                    count += 1;
                }
            }
        }
        count
    }

    /// AVX2 newline counter.
    ///
    /// # Safety
    ///
    /// Requires AVX2 and a valid `raw[..len]`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn count_avx2(raw: *const u32, len: usize) -> usize {
        // SAFETY: every load is inside `raw[..len]`.
        unsafe {
            let mut total = 0;
            let mut index = 0;
            while index + 8 <= len {
                total += compare8(raw, index).count_ones() as usize;
                index += 8;
            }
            while index < len {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    total += 1;
                }
                index += 1;
            }
            total
        }
    }

    /// SSE2 newline counter.
    ///
    /// # Safety
    ///
    /// Requires SSE2 and a valid `raw[..len]`.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn count_sse2(raw: *const u32, len: usize) -> usize {
        // SAFETY: every load is inside `raw[..len]`.
        unsafe {
            let mut total = 0;
            let mut index = 0;
            while index + 4 <= len {
                total += compare4(raw, index).count_ones() as usize;
                index += 4;
            }
            while index < len {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    total += 1;
                }
                index += 1;
            }
            total
        }
    }

    /// AVX2 forward newline search.
    ///
    /// # Safety
    ///
    /// Requires AVX2, a valid `raw[..len]` and `from <= len`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_forward_avx2(raw: *const u32, len: usize, from: usize) -> Option<usize> {
        // SAFETY: every load is inside `raw[..len]`.
        unsafe {
            let mut index = from;
            while index + 8 <= len {
                if index & 8 == 0 {
                    // SAFETY: the helper only prefetches in range.
                }
                let mask = compare8(raw, index);
                if mask != 0 {
                    return Some(index + mask.trailing_zeros() as usize);
                }
                index += 8;
            }
            while index < len {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    return Some(index);
                }
                index += 1;
            }
            None
        }
    }

    /// SSE2 forward newline search.
    ///
    /// # Safety
    ///
    /// Requires SSE2, a valid `raw[..len]` and `from <= len`.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn find_forward_sse2(raw: *const u32, len: usize, from: usize) -> Option<usize> {
        // SAFETY: every load is inside `raw[..len]`.
        unsafe {
            let mut index = from;
            while index + 4 <= len {
                let mask = compare4(raw, index);
                if mask != 0 {
                    return Some(index + mask.trailing_zeros() as usize);
                }
                index += 4;
            }
            while index < len {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    return Some(index);
                }
                index += 1;
            }
            None
        }
    }

    /// AVX2 backward newline search over `raw[..before]`.
    ///
    /// # Safety
    ///
    /// Requires AVX2 and a valid `raw[..before]`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_backward_avx2(raw: *const u32, before: usize) -> Option<usize> {
        // SAFETY: every load ends at or below `before`.
        unsafe {
            let mut end = before;
            while end >= 8 {
                if end & 8 == 0 {
                    // SAFETY: the helper only prefetches in range.
                }
                let mask = compare8(raw, end - 8);
                if mask != 0 {
                    return Some(end - 8 + (31 - mask.leading_zeros() as usize));
                }
                end -= 8;
            }
            while end > 0 {
                end -= 1;
                let word = core::ptr::read_unaligned(raw.add(end));
                if (word & CHAR_MASK) == NEWLINE {
                    return Some(end);
                }
            }
            None
        }
    }

    /// SSE2 backward newline search over `raw[..before]`.
    ///
    /// # Safety
    ///
    /// Requires SSE2 and a valid `raw[..before]`.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn find_backward_sse2(raw: *const u32, before: usize) -> Option<usize> {
        // SAFETY: every load ends at or below `before`.
        unsafe {
            let mut end = before;
            while end >= 4 {
                let mask = compare4(raw, end - 4);
                if mask != 0 {
                    return Some(end - 4 + (31 - mask.leading_zeros() as usize));
                }
                end -= 4;
            }
            while end > 0 {
                end -= 1;
                let word = core::ptr::read_unaligned(raw.add(end));
                if (word & CHAR_MASK) == NEWLINE {
                    return Some(end);
                }
            }
            None
        }
    }

    /// AVX2 line bounds with a single feature-dispatch.
    ///
    /// # Safety
    ///
    /// Requires AVX2, a valid `raw[..len]` and `position <= len`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn line_bounds_avx2(raw: *const u32, len: usize, position: usize) -> (usize, usize) {
        // SAFETY: both scans stay inside `raw[..len]`.
        let start = unsafe { find_backward_avx2(raw, position) }.map_or(0, |newline| newline + 1);
        let end = unsafe { find_forward_avx2(raw, len, position) }.unwrap_or(len);
        (start, end)
    }

    /// SSE2 line bounds with a single feature-dispatch.
    ///
    /// # Safety
    ///
    /// Requires SSE2, a valid `raw[..len]` and `position <= len`.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn line_bounds_sse2(raw: *const u32, len: usize, position: usize) -> (usize, usize) {
        // SAFETY: both scans stay inside `raw[..len]`.
        let start = unsafe { find_backward_sse2(raw, position) }.map_or(0, |newline| newline + 1);
        let end = unsafe { find_forward_sse2(raw, len, position) }.unwrap_or(len);
        (start, end)
    }

    /// AVX2 search for the `n`-th newline.
    ///
    /// # Safety
    ///
    /// Requires AVX2 and a valid `raw[..len]`; `n > 0`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn nth_avx2(raw: *const u32, len: usize, n: usize) -> Option<usize> {
        // SAFETY: every load is inside `raw[..len]`.
        unsafe {
            let mut remaining = n;
            let mut index = 0;
            while index + 8 <= len {
                let mask = compare8(raw, index);
                let count = mask.count_ones() as usize;
                if count >= remaining {
                    return nth_bit(mask, remaining).map(|bit| index + bit);
                }
                remaining -= count;
                index += 8;
            }
            while index < len {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    remaining -= 1;
                    if remaining == 0 {
                        return Some(index);
                    }
                }
                index += 1;
            }
            None
        }
    }

    /// SSE2 search for the `n`-th newline.
    ///
    /// # Safety
    ///
    /// Requires SSE2 and a valid `raw[..len]`; `n > 0`.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn nth_sse2(raw: *const u32, len: usize, n: usize) -> Option<usize> {
        // SAFETY: every load is inside `raw[..len]`.
        unsafe {
            let mut remaining = n;
            let mut index = 0;
            while index + 4 <= len {
                let mask = compare4(raw, index);
                let count = mask.count_ones() as usize;
                if count >= remaining {
                    return nth_bit(mask, remaining).map(|bit| index + bit);
                }
                remaining -= count;
                index += 4;
            }
            while index < len {
                let word = core::ptr::read_unaligned(raw.add(index));
                if (word & CHAR_MASK) == NEWLINE {
                    remaining -= 1;
                    if remaining == 0 {
                        return Some(index);
                    }
                }
                index += 1;
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn pack(character: u8) -> u32 {
        (0xE << 28) | u32::from(character)
    }

    /// Builds a document with newlines at the requested positions.
    fn fixture(len: usize, newlines: &[usize]) -> Vec<u32> {
        let mut doc = vec![pack(b'a'); len];
        for &at in newlines {
            doc[at] = pack(b'\n');
        }
        doc
    }

    #[test]
    fn counts_and_finds_match_a_scalar_walk() {
        let doc = fixture(1029, &[0, 7, 8, 31, 32, 100, 511, 512, 1028]);
        // SAFETY: the slice is valid for its whole length.
        unsafe {
            assert_eq!(count_newlines(doc.as_ptr(), doc.len()), 9);
            assert_eq!(find_newline_forward(doc.as_ptr(), doc.len(), 0), Some(0));
            assert_eq!(find_newline_forward(doc.as_ptr(), doc.len(), 1), Some(7));
            assert_eq!(find_newline_forward(doc.as_ptr(), doc.len(), 9), Some(31));
            assert_eq!(find_newline_forward(doc.as_ptr(), doc.len(), 1029), None);
            assert_eq!(find_newline_backward(doc.as_ptr(), doc.len(), 1029), Some(1028));
            assert_eq!(find_newline_backward(doc.as_ptr(), doc.len(), 1028), Some(512));
            assert_eq!(find_newline_backward(doc.as_ptr(), doc.len(), 8), Some(7));
            assert_eq!(find_newline_backward(doc.as_ptr(), doc.len(), 0), None);
            assert_eq!(find_nth_newline(doc.as_ptr(), doc.len(), 1), Some(0));
            assert_eq!(find_nth_newline(doc.as_ptr(), doc.len(), 4), Some(31));
            assert_eq!(find_nth_newline(doc.as_ptr(), doc.len(), 9), Some(1028));
            assert_eq!(find_nth_newline(doc.as_ptr(), doc.len(), 10), None);
            assert_eq!(find_nth_newline(doc.as_ptr(), doc.len(), 0), None);
        }
    }

    #[test]
    fn empty_and_tail_only_documents_are_inert() {
        let empty: [u32; 0] = [];
        // SAFETY: zero-length input needs no readable memory.
        unsafe {
            assert_eq!(count_newlines(empty.as_ptr(), 0), 0);
            assert_eq!(find_newline_forward(empty.as_ptr(), 0, 0), None);
            assert_eq!(find_newline_backward(empty.as_ptr(), 0, 0), None);
            assert_eq!(find_nth_newline(empty.as_ptr(), 0, 1), None);
        }
        let doc = fixture(5, &[4]);
        // SAFETY: the slice is valid for its whole length.
        unsafe {
            assert_eq!(count_newlines(doc.as_ptr(), doc.len()), 1);
            assert_eq!(find_newline_forward(doc.as_ptr(), doc.len(), 4), Some(4));
            assert_eq!(find_newline_backward(doc.as_ptr(), doc.len(), 4), None);
        }
    }

    #[test]
    fn collectors_return_newlines_in_scan_order() {
        let doc = fixture(1029, &[0, 7, 8, 31, 32, 100, 511, 512, 1028]);
        let mut forward = [0u32; 5];
        let mut backward = [0u32; 4];
        // SAFETY: the slice is valid for its whole length and the scan
        // bounds are inside it.
        unsafe {
            let count = collect_newlines_forward(doc.as_ptr(), doc.len(), 8, &mut forward);
            assert_eq!(count, forward.len());
            assert_eq!(forward, [8, 31, 32, 100, 511]);

            let count = collect_newlines_backward(doc.as_ptr(), doc.len(), 513, &mut backward);
            assert_eq!(count, backward.len());
            assert_eq!(backward, [512, 511, 100, 32]);

            assert_eq!(
                collect_newlines_backward(doc.as_ptr(), doc.len(), 0, &mut backward),
                0
            );
            assert_eq!(collect_newlines_forward(doc.as_ptr(), doc.len(), 0, &mut []), 0);
            assert_eq!(
                collect_newlines_forward(doc.as_ptr(), doc.len(), 1029, &mut forward),
                0
            );
        }
    }

    #[test]
    fn type_bits_are_ignored_when_testing_for_newlines() {
        // A merge-type newline must still end a line.
        let doc = [(0xD << 28) | u32::from(b'\n')];
        // SAFETY: the slice is valid for its whole length.
        unsafe {
            assert_eq!(count_newlines(doc.as_ptr(), doc.len()), 1);
            assert_eq!(find_newline_forward(doc.as_ptr(), doc.len(), 0), Some(0));
        }
    }
}
