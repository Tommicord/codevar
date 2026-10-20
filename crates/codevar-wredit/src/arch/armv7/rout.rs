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

//! Scalar ARMv7-A (A32) kernels.
//!
//! Each function is hand-written assembly in `rout.S` and follows
//! AAPCS: the four argument registers r0-r3, results in r0 (r0:r1 for
//! `u64`), callee-saved registers preserved. The routines are baseline
//! A32 only; NEON is not required.
//!
//! Every entry point is `unsafe`: callers must uphold the buffer
//! contracts documented on each wrapper.

unsafe extern "C" {
    fn codevar_wredit_k_fill_bytes(dst: *mut u8, value: u8, len: usize) -> *mut u8;
    fn codevar_wredit_k_fill_words(dst: *mut u32, value: u32, count: usize) -> *mut u32;
    fn codevar_wredit_k_copy_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;
    fn codevar_wredit_k_move_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;
    fn codevar_wredit_k_count_byte(hay: *const u8, needle: u8, len: usize) -> usize;
    fn codevar_wredit_k_find_byte_rev(hay: *const u8, needle: u8, len: usize) -> *const u8;
    fn codevar_wredit_k_find_any2(hay: *const u8, a: u8, b: u8, len: usize) -> *const u8;
    fn codevar_wredit_k_find_not_byte(hay: *const u8, needle: u8, len: usize) -> *const u8;
    fn codevar_wredit_k_count_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> usize;
    fn codevar_wredit_k_find_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32;
    fn codevar_wredit_k_find_eq_u32_rev(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32;
    fn codevar_wredit_k_memxor(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;
    fn codevar_wredit_k_memcmp(a: *const u8, b: *const u8, len: usize) -> i32;
    fn codevar_wredit_k_sum_bytes(src: *const u8, len: usize) -> u64;
    fn codevar_wredit_k_weighted_checksum(src: *const u8, len: usize, seed: u32) -> u64;
    fn codevar_wredit_k_utf8_count_starts(src: *const u8, len: usize) -> usize;
    fn codevar_wredit_k_replace_byte(dst: *mut u8, from: u8, to: u8, len: usize) -> usize;
    fn codevar_wredit_k_reverse_bytes(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;
    fn codevar_wredit_k_fill_pattern8(dst: *mut u8, pattern: u64, len: usize) -> *mut u8;
    fn codevar_wredit_k_byte_histogram(counts: *mut u32, src: *const u8, len: usize);
}

/// Writes `value` to `dst[..len]` and returns `dst`.
///
/// # Safety
///
/// `dst` must be valid for writes of `len` bytes. A `len` of zero does not
/// access `dst`.
#[inline]
pub unsafe fn fill_bytes(dst: *mut u8, value: u8, len: usize) -> *mut u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_fill_bytes(dst, value, len) }
}

/// Writes `value` to `dst[..count]` and returns `dst`.
///
/// # Safety
///
/// `dst` must be valid for writes of `count` `u32`s and aligned for `u32`.
/// A `count` of zero does not access `dst`.
#[inline]
pub unsafe fn fill_words(dst: *mut u32, value: u32, count: usize) -> *mut u32 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_fill_words(dst, value, count) }
}

/// Copies `count` words from `src` to `dst` and returns `dst`.
///
/// # Safety
///
/// Both ranges must be valid for `count` `u32`s and must not overlap.
/// A `count` of zero does not access either pointer.
#[inline]
pub unsafe fn copy_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_copy_words(dst, src, count) }
}

/// Copies `count` words from `src` to `dst`, handling overlap, and returns
/// `dst`. Copies forward when `dst < src` and backward otherwise.
///
/// # Safety
///
/// Both ranges must be valid for `count` `u32`s. Overlap is allowed. A
/// `count` of zero does not access either pointer.
#[inline]
pub unsafe fn move_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_move_words(dst, src, count) }
}

/// Counts the bytes equal to `needle` in `hay[..len]`.
///
/// # Safety
///
/// `hay` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `hay`.
#[inline]
pub unsafe fn count_byte(hay: *const u8, needle: u8, len: usize) -> usize {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_count_byte(hay, needle, len) }
}

/// Returns a pointer to the last occurrence of `needle` in `hay[..len]`, or
/// null when it does not appear.
///
/// # Safety
///
/// `hay` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `hay`.
#[inline]
pub unsafe fn find_byte_rev(hay: *const u8, needle: u8, len: usize) -> *const u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_find_byte_rev(hay, needle, len) }
}

/// Returns a pointer to the first byte equal to `a` or `b` in `hay[..len]`,
/// or null when neither appears.
///
/// # Safety
///
/// `hay` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `hay`.
#[inline]
pub unsafe fn find_any2(hay: *const u8, a: u8, b: u8, len: usize) -> *const u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_find_any2(hay, a, b, len) }
}

/// Returns a pointer to the first byte in `hay[..len]` that is not `needle`,
/// or null when every byte equals `needle`.
///
/// # Safety
///
/// `hay` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `hay`.
#[inline]
pub unsafe fn find_not_byte(hay: *const u8, needle: u8, len: usize) -> *const u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_find_not_byte(hay, needle, len) }
}

/// Counts the words where `(word & mask) == value`.
///
/// # Safety
///
/// `src` must be valid for reads of `count` `u32`s and aligned for `u32`.
/// A `count` of zero does not access `src`.
#[inline]
pub unsafe fn count_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> usize {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_count_eq_u32(src, value, mask, count) }
}

/// Returns a pointer to the first word where `(word & mask) == value`, or
/// null when no word matches.
///
/// # Safety
///
/// `src` must be valid for reads of `count` `u32`s and aligned for `u32`.
/// A `count` of zero does not access `src`.
#[inline]
pub unsafe fn find_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_find_eq_u32(src, value, mask, count) }
}

/// Returns a pointer to the last word where `(word & mask) == value`, or
/// null when no word matches.
///
/// # Safety
///
/// `src` must be valid for reads of `count` `u32`s and aligned for `u32`.
/// A `count` of zero does not access `src`.
#[inline]
pub unsafe fn find_eq_u32_rev(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_find_eq_u32_rev(src, value, mask, count) }
}

/// Applies `dst[i] ^= src[i]` for every byte in `dst[..len]` and returns
/// `dst`.
///
/// # Safety
///
/// Both ranges must be valid for `len` bytes. They may overlap only when
/// `dst == src`. A `len` of zero does not access either pointer.
#[inline]
pub unsafe fn memxor(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_memxor(dst, src, len) }
}

/// Compares two byte ranges. Returns 0 when equal, -1 when `a < b` at the
/// first differing byte, and 1 when `a > b`.
///
/// # Safety
///
/// Both ranges must be valid for reads of `len` bytes. A `len` of zero does
/// not access either pointer.
#[inline]
pub unsafe fn memcmp(a: *const u8, b: *const u8, len: usize) -> i32 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_memcmp(a, b, len) }
}

/// Returns the sum of the zero-extended bytes in `src[..len]` as a `u64`.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `src` and returns zero.
#[inline]
pub unsafe fn sum_bytes(src: *const u8, len: usize) -> u64 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_sum_bytes(src, len) }
}

/// Returns the weighted checksum of `src[..len]`.
///
/// The first weight is `seed % 251 + 1`; each following byte uses the next
/// weight, wrapping back to 1 after 251. Every byte contributes
/// `byte * weight` to the 64-bit sum.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `src` and returns zero.
#[inline]
pub unsafe fn weighted_checksum(src: *const u8, len: usize, seed: u32) -> u64 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_weighted_checksum(src, len, seed) }
}

/// Counts UTF-8 lead/ASCII bytes: bytes where `(byte & 0xC0) != 0x80`.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes. A `len` of zero does not
/// access `src`.
#[inline]
pub unsafe fn utf8_count_starts(src: *const u8, len: usize) -> usize {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_utf8_count_starts(src, len) }
}

/// Replaces every `from` byte with `to` in `dst[..len]` and returns the
/// number of bytes replaced.
///
/// # Safety
///
/// `dst` must be valid for reads and writes of `len` bytes. A `len` of zero
/// does not access `dst`.
#[inline]
pub unsafe fn replace_byte(dst: *mut u8, from: u8, to: u8, len: usize) -> usize {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_replace_byte(dst, from, to, len) }
}

/// Copies `src[..len]` reversed into `dst[..len]` and returns `dst`.
///
/// # Safety
///
/// Both ranges must be valid for `len` bytes and must not overlap. A `len`
/// of zero does not access either pointer.
#[inline]
pub unsafe fn reverse_bytes(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_reverse_bytes(dst, src, len) }
}

/// Repeats the eight bytes of `pattern` across `dst[..len]` and returns
/// `dst`. The final partial block uses the first `len % 8` bytes of the
/// pattern in native memory order.
///
/// # Safety
///
/// `dst` must be valid for writes of `len` bytes. A `len` of zero does not
/// access `dst`.
#[inline]
pub unsafe fn fill_pattern8(dst: *mut u8, pattern: u64, len: usize) -> *mut u8 {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_fill_pattern8(dst, pattern, len) }
}

/// Clears the 256 `u32` counters at `counts`, then increments
/// `counts[byte]` for every byte in `src[..len]`.
///
/// # Safety
///
/// `counts` must be valid for writes of 256 `u32`s and aligned for `u32`.
/// `src` must be valid for reads of `len` bytes and must not overlap
/// `counts`. A `len` of zero does not access `src`.
#[inline]
pub unsafe fn byte_histogram(counts: *mut u32, src: *const u8, len: usize) {
    // Safety: forwarded to the caller.
    unsafe { codevar_wredit_k_byte_histogram(counts, src, len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    #[test]
    fn fill_bytes_writes_prefix_and_returns_dst() {
        let mut buf = [0xa5u8; 11];
        let ret = unsafe { fill_bytes(buf.as_mut_ptr(), 0x5a, 7) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert!(buf[..7].iter().all(|&b| b == 0x5a));
        assert_eq!(buf[7], 0xa5);

        let mut untouched = [0x33u8; 2];
        assert_eq!(
            unsafe { fill_bytes(untouched.as_mut_ptr(), 0x00, 0) },
            untouched.as_mut_ptr()
        );
        assert_eq!(untouched, [0x33, 0x33]);

        let mut one = [0u8; 1];
        assert_eq!(unsafe { fill_bytes(one.as_mut_ptr(), 0x7e, 1) }, one.as_mut_ptr());
        assert_eq!(one, [0x7e]);

        // An unaligned start exercises the alignment prologue.
        let mut unaligned = [0u8; 12];
        unsafe { fill_bytes(unaligned.as_mut_ptr().add(1), 0x3c, 9) };
        assert_eq!(unaligned[0], 0);
        assert!(unaligned[1..10].iter().all(|&b| b == 0x3c));
        assert_eq!(unaligned[10], 0);

        assert!(unsafe { fill_bytes(ptr::null_mut(), 0xff, 0) }.is_null());
    }

    #[test]
    fn fill_words_writes_count_words() {
        let mut buf = [0u32; 9];
        let ret = unsafe { fill_words(buf.as_mut_ptr(), 0xdead_beef, 5) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(&buf[..5], &[0xdead_beef; 5]);
        assert_eq!(buf[5], 0);

        let mut one = [0u32; 1];
        assert_eq!(unsafe { fill_words(one.as_mut_ptr(), 7, 1) }, one.as_mut_ptr());
        assert_eq!(one, [7]);

        assert!(unsafe { fill_words(ptr::null_mut(), 1, 0) }.is_null());
    }

    #[test]
    fn copy_words_copies_words() {
        let src = [1u32, 2, 3, 4, 5];
        let mut dst = [0u32; 5];
        let ret = unsafe { copy_words(dst.as_mut_ptr(), src.as_ptr(), 5) };
        assert_eq!(ret, dst.as_mut_ptr());
        assert_eq!(dst, src);

        let mut one = [0u32; 1];
        assert_eq!(
            unsafe { copy_words(one.as_mut_ptr(), src.as_ptr(), 1) },
            one.as_mut_ptr()
        );
        assert_eq!(one, [1]);

        assert!(unsafe { copy_words(ptr::null_mut(), ptr::null(), 0) }.is_null());
    }

    #[test]
    fn move_words_handles_both_overlap_directions() {
        // Shift right by two words.
        let mut right = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let ret = unsafe { move_words(right.as_mut_ptr().add(2), right.as_ptr(), 5) };
        assert_eq!(ret, unsafe { right.as_mut_ptr().add(2) });
        assert_eq!(right, [0, 1, 0, 1, 2, 3, 4, 7]);

        // Shift left by two words.
        let mut left = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let ret = unsafe { move_words(left.as_mut_ptr(), left.as_ptr().add(2), 5) };
        assert_eq!(ret, left.as_mut_ptr());
        assert_eq!(left, [2, 3, 4, 5, 6, 5, 6, 7]);

        // Identical ranges are a no-op.
        let mut same = [9u32, 8, 7];
        assert_eq!(
            unsafe { move_words(same.as_mut_ptr(), same.as_ptr(), 3) },
            same.as_mut_ptr()
        );
        assert_eq!(same, [9, 8, 7]);

        // One word, no overlap.
        let mut one = [0u32; 1];
        assert_eq!(
            unsafe { move_words(one.as_mut_ptr(), right.as_ptr(), 1) },
            one.as_mut_ptr()
        );
        assert_eq!(one, [0]);

        assert!(unsafe { move_words(ptr::null_mut(), ptr::null(), 0) }.is_null());
    }

    #[test]
    fn count_byte_counts_matches() {
        let hay = b"banana";
        assert_eq!(unsafe { count_byte(hay.as_ptr(), b'a', hay.len()) }, 3);
        assert_eq!(unsafe { count_byte(hay.as_ptr(), b'n', hay.len()) }, 2);
        assert_eq!(unsafe { count_byte(hay.as_ptr(), b'z', hay.len()) }, 0);
        assert_eq!(unsafe { count_byte(hay.as_ptr(), b'b', 1) }, 1);
        assert_eq!(unsafe { count_byte(ptr::null(), b'a', 0) }, 0);
    }

    #[test]
    fn find_byte_rev_finds_last_occurrence() {
        let hay = b"abcabc";
        assert_eq!(unsafe { find_byte_rev(hay.as_ptr(), b'a', 6) }, unsafe {
            hay.as_ptr().add(3)
        });
        assert_eq!(unsafe { find_byte_rev(hay.as_ptr(), b'a', 3) }, hay.as_ptr());
        assert!(unsafe { find_byte_rev(hay.as_ptr(), b'z', 6) }.is_null());
        assert!(unsafe { find_byte_rev(ptr::null(), b'a', 0) }.is_null());
    }

    #[test]
    fn find_any2_finds_either_byte() {
        let hay = b"hello";
        assert_eq!(unsafe { find_any2(hay.as_ptr(), b'z', b'l', 5) }, unsafe {
            hay.as_ptr().add(2)
        });
        assert_eq!(unsafe { find_any2(hay.as_ptr(), b'o', b'l', 5) }, unsafe {
            hay.as_ptr().add(2)
        });
        assert!(unsafe { find_any2(hay.as_ptr(), b'x', b'y', 5) }.is_null());
        assert!(unsafe { find_any2(ptr::null(), b'a', b'b', 0) }.is_null());
    }

    #[test]
    fn find_not_byte_finds_first_difference() {
        let hay = [b'a', b'a', b'b', b'a'];
        assert_eq!(unsafe { find_not_byte(hay.as_ptr(), b'a', 4) }, unsafe {
            hay.as_ptr().add(2)
        });
        assert!(unsafe { find_not_byte(hay.as_ptr(), b'a', 2) }.is_null());
        assert!(unsafe { find_not_byte(hay.as_ptr(), b'b', 0) }.is_null());
        assert_eq!(unsafe { find_not_byte(hay.as_ptr(), b'z', 1) }, hay.as_ptr());
    }

    #[test]
    fn count_eq_u32_respects_mask() {
        let src = [0x1234_5678u32, 0x0000_1234, 0xffff_1234, 0x1234_0000, 0x0000_1234];
        assert_eq!(unsafe { count_eq_u32(src.as_ptr(), 0x1234, 0xffff, 5) }, 3);
        assert_eq!(unsafe { count_eq_u32(src.as_ptr(), 0xffff_1234, 0xffff, 5) }, 3);
        assert_eq!(unsafe { count_eq_u32(src.as_ptr(), 0x1234, 0xffff, 1) }, 0);
        assert_eq!(unsafe { count_eq_u32(src.as_ptr(), 0x1234, 0xffff, 2) }, 1);
        assert_eq!(unsafe { count_eq_u32(ptr::null(), 1, 0xffff, 0) }, 0);
    }

    #[test]
    fn find_eq_u32_finds_first_match() {
        let src = [0u32, 0xaaaa_1111, 0xbbbb_2222, 0xaaaa_1111];
        assert_eq!(unsafe { find_eq_u32(src.as_ptr(), 0x1111, 0xffff, 4) }, unsafe {
            src.as_ptr().add(1)
        });
        assert!(unsafe { find_eq_u32(src.as_ptr(), 0x1111, 0xffff, 1) }.is_null());
        assert!(unsafe { find_eq_u32(src.as_ptr(), 0x9999, 0xffff, 4) }.is_null());
        assert!(unsafe { find_eq_u32(ptr::null(), 1, 0xffff, 0) }.is_null());
    }

    #[test]
    fn find_eq_u32_rev_finds_last_match() {
        let src = [0u32, 0xaaaa_1111, 0xbbbb_2222, 0xaaaa_1111];
        assert_eq!(
            unsafe { find_eq_u32_rev(src.as_ptr(), 0x1111, 0xffff, 4) },
            unsafe { src.as_ptr().add(3) }
        );
        assert_eq!(
            unsafe { find_eq_u32_rev(src.as_ptr(), 0x1111, 0xffff, 3) },
            unsafe { src.as_ptr().add(1) }
        );
        assert!(unsafe { find_eq_u32_rev(src.as_ptr(), 0x9999, 0xffff, 4) }.is_null());
        assert!(unsafe { find_eq_u32_rev(ptr::null(), 1, 0xffff, 0) }.is_null());
    }

    #[test]
    fn memxor_xors_word_and_tail() {
        let src = [0xffu8, 0x01, 0x0f, 0xaa, 0x01, 0x02, 0x03];
        let mut dst = [0xf0u8, 0x0f, 0xaa, 0x55, 0x12, 0x34, 0x56];
        let ret = unsafe { memxor(dst.as_mut_ptr(), src.as_ptr(), 7) };
        assert_eq!(ret, dst.as_mut_ptr());
        assert_eq!(dst, [0x0f, 0x0e, 0xa5, 0xff, 0x13, 0x36, 0x55]);

        // Exact overlap zeroes the buffer.
        let mut exact = [0x5au8; 7];
        assert_eq!(
            unsafe { memxor(exact.as_mut_ptr(), exact.as_ptr(), 7) },
            exact.as_mut_ptr()
        );
        assert_eq!(exact, [0u8; 7]);

        // Differently aligned operands take the byte path.
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut unaligned = [0u8; 10];
        unsafe { memxor(unaligned.as_mut_ptr().add(1), src.as_ptr().add(2), 6) };
        assert_eq!(&unaligned[1..7], &[3, 4, 5, 6, 7, 8]);
        assert_eq!(unaligned[0], 0);

        assert!(unsafe { memxor(ptr::null_mut(), ptr::null(), 0) }.is_null());
    }

    #[test]
    fn memcmp_classifies_first_difference() {
        let a = [1u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 5];
        let high = [1u8, 2, 3, 4, 6];
        let low = [0u8, 2, 3, 4, 5];
        assert_eq!(unsafe { memcmp(a.as_ptr(), b.as_ptr(), 5) }, 0);
        assert_eq!(unsafe { memcmp(a.as_ptr(), high.as_ptr(), 5) }, -1);
        assert_eq!(unsafe { memcmp(high.as_ptr(), a.as_ptr(), 5) }, 1);
        assert_eq!(unsafe { memcmp(a.as_ptr(), low.as_ptr(), 5) }, 1);
        assert_eq!(unsafe { memcmp(a.as_ptr(), high.as_ptr(), 4) }, 0);
        assert_eq!(unsafe { memcmp(a.as_ptr(), high.as_ptr(), 1) }, 0);
        assert_eq!(unsafe { memcmp(ptr::null(), ptr::null(), 0) }, 0);
    }

    #[test]
    fn sum_bytes_accumulates_bytes() {
        let src = [1u8, 2, 3, 250];
        assert_eq!(unsafe { sum_bytes(src.as_ptr(), 4) }, 256);
        let one = [7u8];
        assert_eq!(unsafe { sum_bytes(one.as_ptr(), 1) }, 7);
        let big = [0xffu8; 300];
        assert_eq!(unsafe { sum_bytes(big.as_ptr(), 300) }, 76_500);
        assert_eq!(unsafe { sum_bytes(ptr::null(), 0) }, 0);
    }

    #[test]
    fn weighted_checksum_matches_known_values() {
        let src = [1u8, 2, 3];
        assert_eq!(unsafe { weighted_checksum(src.as_ptr(), 3, 0) }, 14);
        assert_eq!(unsafe { weighted_checksum(src.as_ptr(), 3, 1) }, 20);
        // Weight starts at 251, then wraps to 1: 251 + 2 + 6.
        assert_eq!(unsafe { weighted_checksum(src.as_ptr(), 3, 250) }, 259);
        // 251 % 251 == 0, so this is the same as seed 0.
        assert_eq!(unsafe { weighted_checksum(src.as_ptr(), 3, 251) }, 14);

        // 260 bytes of one: 1 + ... + 251 + 1 + ... + 9.
        let ones = [1u8; 260];
        assert_eq!(unsafe { weighted_checksum(ones.as_ptr(), 260, 0) }, 31_671);

        // Large seeds exercise the multiply-high modulo reduction.
        let one = [1u8];
        assert_eq!(unsafe { weighted_checksum(one.as_ptr(), 1, u32::MAX) }, 123);
        assert_eq!(unsafe { weighted_checksum(one.as_ptr(), 1, 0x8000_0000) }, 188);

        assert_eq!(unsafe { weighted_checksum(ptr::null(), 0, 12345) }, 0);
    }

    #[test]
    fn utf8_count_starts_skips_continuations() {
        let text = [0x68u8, 0xc3, 0xa9, 0x6c, 0x6c, 0x6f];
        assert_eq!(unsafe { utf8_count_starts(text.as_ptr(), 6) }, 5);
        assert_eq!(unsafe { utf8_count_starts(text.as_ptr(), 1) }, 1);

        let continuations = [0x80u8, 0xbf, 0x81];
        assert_eq!(unsafe { utf8_count_starts(continuations.as_ptr(), 3) }, 0);
        assert_eq!(unsafe { utf8_count_starts(ptr::null(), 0) }, 0);
    }

    #[test]
    fn replace_byte_reports_replacements() {
        let mut buf = [b'a', b'b', b'a', b'c', b'a'];
        assert_eq!(unsafe { replace_byte(buf.as_mut_ptr(), b'a', b'z', 5) }, 3);
        assert_eq!(buf, [b'z', b'b', b'z', b'c', b'z']);

        let mut none = [b'x'; 3];
        assert_eq!(unsafe { replace_byte(none.as_mut_ptr(), b'a', b'z', 3) }, 0);
        assert_eq!(none, [b'x'; 3]);

        // Replacing a byte with itself still counts the occurrences.
        let mut same = [b'a', b'b', b'a'];
        assert_eq!(unsafe { replace_byte(same.as_mut_ptr(), b'a', b'a', 3) }, 2);
        assert_eq!(same, [b'a', b'b', b'a']);

        assert_eq!(unsafe { replace_byte(ptr::null_mut(), b'a', b'z', 0) }, 0);
    }

    #[test]
    fn reverse_bytes_handles_word_and_tail() {
        let src = [1u8, 2, 3, 4, 5, 6, 7];
        let mut dst = [0u8; 7];
        let ret = unsafe { reverse_bytes(dst.as_mut_ptr(), src.as_ptr(), 7) };
        assert_eq!(ret, dst.as_mut_ptr());
        assert_eq!(dst, [7, 6, 5, 4, 3, 2, 1]);

        let mut four = [0u8; 4];
        let ret = unsafe { reverse_bytes(four.as_mut_ptr(), src.as_ptr(), 4) };
        assert_eq!(ret, four.as_mut_ptr());
        assert_eq!(four, [4, 3, 2, 1]);

        let mut one = [0u8; 1];
        unsafe { reverse_bytes(one.as_mut_ptr(), src.as_ptr(), 1) };
        assert_eq!(one, [1]);

        assert!(unsafe { reverse_bytes(ptr::null_mut(), ptr::null(), 0) }.is_null());
    }

    #[test]
    fn fill_pattern8_repeats_native_bytes() {
        // ARMv7 Linux targets are little-endian, so the low word precedes
        // the high word in memory.
        let pattern: u64 = 0x0807_0605_0403_0201;
        let mut buf = [0u8; 12];
        let ret = unsafe { fill_pattern8(buf.as_mut_ptr(), pattern, 12) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]);

        let mut eight = [0u8; 8];
        unsafe { fill_pattern8(eight.as_mut_ptr(), pattern, 8) };
        assert_eq!(eight, [1, 2, 3, 4, 5, 6, 7, 8]);

        let mut five = [0u8; 5];
        unsafe { fill_pattern8(five.as_mut_ptr(), pattern, 5) };
        assert_eq!(five, [1, 2, 3, 4, 5]);

        let mut three = [0u8; 3];
        unsafe { fill_pattern8(three.as_mut_ptr(), pattern, 3) };
        assert_eq!(three, [1, 2, 3]);

        // An unaligned start exercises the rotation prologue.
        let mut unaligned = [0u8; 14];
        unsafe { fill_pattern8(unaligned.as_mut_ptr().add(1), pattern, 10) };
        assert_eq!(&unaligned[1..11], &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2]);
        assert_eq!(unaligned[0], 0);

        assert!(unsafe { fill_pattern8(ptr::null_mut(), pattern, 0) }.is_null());
    }

    #[test]
    fn byte_histogram_clears_and_counts() {
        let src = b"abracadabra";
        let mut counts = [0xdead_beefu32; 256];
        unsafe { byte_histogram(counts.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(counts[b'a' as usize], 5);
        assert_eq!(counts[b'b' as usize], 2);
        assert_eq!(counts[b'r' as usize], 2);
        assert_eq!(counts[b'c' as usize], 1);
        assert_eq!(counts[b'd' as usize], 1);
        assert_eq!(counts[b'e' as usize], 0);
        assert_eq!(counts[0xff], 0);

        let all: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut every = [0u32; 256];
        unsafe { byte_histogram(every.as_mut_ptr(), all.as_ptr(), all.len()) };
        assert!(every.iter().all(|&n| n == 1));

        let mut zeros = [7u32; 256];
        unsafe { byte_histogram(zeros.as_mut_ptr(), ptr::null(), 0) };
        assert!(zeros.iter().all(|&n| n == 0));
    }
}
