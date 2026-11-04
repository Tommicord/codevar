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

//! AArch64 auxiliary kernels.
//!
//! These bindings wrap the hand-written routines in `rout.S`. All of
//! them follow the AAPCS64 C ABI used by Rust `extern "C"`, are
//! position-independent leaf functions, and only clobber caller-saved
//! registers. A `len`/`count` of zero never dereferences a pointer.
//!
//! Callers should prefer the public wrappers under [`crate::arch`]; these
//! raw entry points exist so the kernels can be tested directly.

unsafe extern "C" {
    /// Writes `value` to every byte of `dst[..len]`.
    ///
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes. `len` may be zero,
    /// in which case `dst` is not accessed.
    pub fn codevar_wredit_k_fill_bytes(dst: *mut u8, value: u8, len: usize) -> *mut u8;

    /// Writes `value` to every word of `dst[..count]`.
    ///
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count` words. `count` may be
    /// zero, in which case `dst` is not accessed.
    pub fn codevar_wredit_k_fill_words(dst: *mut u32, value: u32, count: usize) -> *mut u32;

    /// Copies `count` words from `src` to `dst`.
    ///
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for writes of `count`
    /// words. The two ranges must not overlap. `count` may be zero.
    pub fn codevar_wredit_k_copy_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;

    /// Copies `count` words from `src` to `dst`, handling overlap.
    ///
    /// Copies forwards when `dst < src` and backwards otherwise, so the
    /// result is as if `src` were copied first.
    ///
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for writes of `count`
    /// words. The ranges may overlap. `count` may be zero.
    pub fn codevar_wredit_k_move_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;

    /// Counts the bytes equal to `needle` in `hay[..len]`.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_count_byte(hay: *const u8, needle: u8, len: usize) -> usize;

    /// Returns the last occurrence of `needle` in `hay[..len]`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_find_byte_rev(hay: *const u8, needle: u8, len: usize) -> *const u8;

    /// Returns the first byte in `hay[..len]` equal to `a` or `b`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_find_any2(hay: *const u8, a: u8, b: u8, len: usize) -> *const u8;

    /// Returns the first byte in `hay[..len]` different from `needle`, or
    /// null when every byte matches.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_find_not_byte(hay: *const u8, needle: u8, len: usize) -> *const u8;

    /// Counts the words of `src[..count]` where `(word & mask) == value`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` words. `count` may be zero.
    pub fn codevar_wredit_k_count_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> usize;

    /// Returns the first word of `src[..count]` where
    /// `(word & mask) == value`, or null.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` words. `count` may be zero.
    pub fn codevar_wredit_k_find_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32;

    /// Returns the last word of `src[..count]` where
    /// `(word & mask) == value`, or null.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` words. `count` may be zero.
    pub fn codevar_wredit_k_find_eq_u32_rev(
        src: *const u32,
        value: u32,
        mask: u32,
        count: usize,
    ) -> *const u32;

    /// XORs every byte of `dst[..len]` with the matching byte of
    /// `src[..len]`, in place.
    ///
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for reads and writes of
    /// `len` bytes. The ranges may overlap only exactly. `len` may be zero.
    pub fn codevar_wredit_k_memxor(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;

    /// Compares `a[..len]` with `b[..len]` lexicographically.
    ///
    /// Returns 0 when equal, -1 when the first differing byte has
    /// `a < b`, and 1 when `a > b`.
    ///
    /// # Safety
    ///
    /// Both pointers must be valid for reads of `len` bytes. `len` may be
    /// zero, in which case the pointers are not accessed.
    pub fn codevar_wredit_k_memcmp(a: *const u8, b: *const u8, len: usize) -> i32;

    /// Sums the zero-extended bytes of `src[..len]`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_sum_bytes(src: *const u8, len: usize) -> u64;

    /// Computes a position-weighted byte checksum.
    ///
    /// The first weight is `seed % 251 + 1`; each byte adds
    /// `byte * weight`, then the weight advances and wraps from 251 back
    /// to 1.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_weighted_checksum(src: *const u8, len: usize, seed: u32) -> u64;

    /// Counts the UTF-8 lead bytes (including ASCII) in `src[..len]`, that
    /// is, bytes where `(byte & 0xC0) != 0x80`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_utf8_count_starts(src: *const u8, len: usize) -> usize;

    /// Replaces every `from` byte with `to` in `dst[..len]`.
    ///
    /// Returns the number of bytes replaced.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for reads and writes of `len` bytes. `len` may
    /// be zero.
    pub fn codevar_wredit_k_replace_byte(dst: *mut u8, from: u8, to: u8, len: usize) -> usize;

    /// Copies `src[..len]` reversed into `dst[..len]`.
    ///
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for writes of `len` bytes.
    /// The ranges must not overlap. `len` may be zero.
    pub fn codevar_wredit_k_reverse_bytes(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;

    /// Repeats the eight native-order bytes of `pattern` across
    /// `dst[..len]`.
    ///
    /// A trailing partial block uses the first `len % 8` pattern bytes.
    /// Returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes. `len` may be zero.
    pub fn codevar_wredit_k_fill_pattern8(dst: *mut u8, pattern: u64, len: usize) -> *mut u8;

    /// Zeroes the 256 `u32` counters at `counts`, then increments
    /// `counts[byte]` for every byte of `src[..len]`.
    ///
    /// # Safety
    ///
    /// `counts` must be valid for writes of 256 `u32` values even when
    /// `len` is zero. `src` must be valid for reads of `len` bytes; it is
    /// not accessed when `len` is zero.
    pub fn codevar_wredit_k_byte_histogram(counts: *mut u32, src: *const u8, len: usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_bytes_handles_zero_length_and_tails() {
        let mut buf = [0u8; 40];
        let dst = buf.as_mut_ptr();
        let len = buf.len();
        // Safety: `buf` is valid for writes of `len` bytes.
        let ret = unsafe { codevar_wredit_k_fill_bytes(dst, 0xAB, len) };
        assert_eq!(ret, dst);
        assert!(buf.iter().all(|&b| b == 0xAB));

        // Zero length must not touch memory but still returns `dst`.
        // Safety: no bytes are accessed when `len == 0`.
        let ret0 = unsafe { codevar_wredit_k_fill_bytes(dst, 0xCD, 0) };
        assert_eq!(ret0, dst);
        assert!(buf.iter().all(|&b| b == 0xAB));

        // Safety: no bytes are accessed when `len == 0`.
        let null = unsafe { codevar_wredit_k_fill_bytes(core::ptr::null_mut(), 1, 0) };
        assert!(null.is_null());

        // Short length exercises the scalar tail.
        let mut small = [0u8; 7];
        let small_len = small.len();
        // Safety: `small` is valid for writes of `small_len` bytes.
        unsafe { codevar_wredit_k_fill_bytes(small.as_mut_ptr(), 0x11, small_len) };
        assert_eq!(small, [0x11; 7]);
    }

    #[test]
    fn fill_words_handles_zero_length_and_tails() {
        let mut buf = [0u32; 19];
        let dst = buf.as_mut_ptr();
        let count = buf.len();
        // Safety: `buf` is valid for writes of `count` words.
        let ret = unsafe { codevar_wredit_k_fill_words(dst, 0xDEAD_BEEF, count) };
        assert_eq!(ret, dst);
        assert!(buf.iter().all(|&w| w == 0xDEAD_BEEF));

        // Safety: no words are accessed when `count == 0`.
        let null = unsafe { codevar_wredit_k_fill_words(core::ptr::null_mut(), 7, 0) };
        assert!(null.is_null());

        // Tail of three words.
        let mut small = [0u32; 3];
        let small_count = small.len();
        // Safety: `small` is valid for writes of `small_count` words.
        unsafe { codevar_wredit_k_fill_words(small.as_mut_ptr(), 9, small_count) };
        assert_eq!(small, [9, 9, 9]);
    }

    #[test]
    fn copy_words_copies_and_returns_dst() {
        let src = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut dst = [0u32; 11];
        let dst_ptr = dst.as_mut_ptr();
        // Safety: distinct buffers; `dst` is valid for 11 words.
        let ret = unsafe { codevar_wredit_k_copy_words(dst_ptr, src.as_ptr(), src.len()) };
        assert_eq!(ret, dst_ptr);
        assert_eq!(dst, src);

        // Safety: no words are accessed when `count == 0`.
        let null = unsafe { codevar_wredit_k_copy_words(core::ptr::null_mut(), core::ptr::null(), 0) };
        assert!(null.is_null());
    }

    #[test]
    fn move_words_handles_both_overlap_directions() {
        // Shift right by two words (dst > src): copies backwards.
        let mut right = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let right_dst = unsafe { right.as_mut_ptr().add(2) };
        // Safety: `right` is valid for 8 words; the ranges overlap.
        let ret = unsafe { codevar_wredit_k_move_words(right_dst, right.as_ptr(), 6) };
        assert_eq!(ret, right_dst);
        assert_eq!(right, [0, 1, 0, 1, 2, 3, 4, 5]);

        // Shift left by two words (dst < src): copies forwards.
        let mut left = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let left_src = unsafe { left.as_ptr().add(2) };
        // Safety: `left` is valid for 8 words; the ranges overlap.
        let ret = unsafe { codevar_wredit_k_move_words(left.as_mut_ptr(), left_src, 6) };
        assert_eq!(ret, left.as_mut_ptr());
        assert_eq!(left, [2, 3, 4, 5, 6, 7, 6, 7]);

        // Safety: no words are accessed when `count == 0`.
        let null = unsafe { codevar_wredit_k_move_words(core::ptr::null_mut(), core::ptr::null(), 0) };
        assert!(null.is_null());
    }

    #[test]
    fn count_byte_counts_with_tail() {
        let hay = b"the quick brown fox jumps over the lazy dog";
        // Safety: `hay` is valid for its full length.
        let n = unsafe { codevar_wredit_k_count_byte(hay.as_ptr(), b'o', hay.len()) };
        let expected = hay.iter().filter(|&&b| b == b'o').count();
        assert_eq!(n, expected);

        // Vector path plus a non-multiple-of-16 tail.
        let mut buf = [0u8; 67];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = if i % 3 == 0 { 0x5A } else { 0 };
        }
        let expected = (0..buf.len()).filter(|i| i % 3 == 0).count();
        // Safety: `buf` is valid for its full length.
        let n = unsafe { codevar_wredit_k_count_byte(buf.as_ptr(), 0x5A, buf.len()) };
        assert_eq!(n, expected);

        // Safety: no bytes are read when `len == 0`.
        assert_eq!(unsafe { codevar_wredit_k_count_byte(core::ptr::null(), 1, 0) }, 0);
    }

    #[test]
    fn find_byte_rev_finds_last_occurrence() {
        let hay = b"abcabc";
        // Safety: `hay` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_byte_rev(hay.as_ptr(), b'a', hay.len()) };
        assert_eq!(found, unsafe { hay.as_ptr().add(3) });

        // Safety: `hay` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_byte_rev(hay.as_ptr(), b'c', hay.len()) };
        assert_eq!(found, unsafe { hay.as_ptr().add(5) });

        // Safety: `hay` is valid for its full length.
        assert!(unsafe { codevar_wredit_k_find_byte_rev(hay.as_ptr(), b'z', hay.len()) }.is_null());

        // Safety: no bytes are read when `len == 0`.
        assert!(unsafe { codevar_wredit_k_find_byte_rev(core::ptr::null(), b'a', 0) }.is_null());
    }

    #[test]
    fn find_any2_finds_either_byte() {
        let hay = b"hello";
        // Safety: `hay` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_any2(hay.as_ptr(), b'x', b'l', hay.len()) };
        assert_eq!(found, unsafe { hay.as_ptr().add(2) });

        // Safety: `hay` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_any2(hay.as_ptr(), b'h', b'z', hay.len()) };
        assert_eq!(found, hay.as_ptr());

        // Safety: `hay` is valid for its full length.
        assert!(unsafe { codevar_wredit_k_find_any2(hay.as_ptr(), b'q', b'z', hay.len()) }.is_null());

        // A match past the first 16-byte chunk exercises the vector path.
        let mut long = [0u8; 40];
        long[33] = b'b';
        long[35] = b'a';
        // Safety: `long` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_any2(long.as_ptr(), b'a', b'b', long.len()) };
        assert_eq!(found, unsafe { long.as_ptr().add(33) });

        // Safety: no bytes are read when `len == 0`.
        assert!(unsafe { codevar_wredit_k_find_any2(core::ptr::null(), 0, 1, 0) }.is_null());
    }

    #[test]
    fn find_not_byte_finds_first_difference() {
        let buf = [7u8, 7, 7, 8, 7];
        // Safety: `buf` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_not_byte(buf.as_ptr(), 7, buf.len()) };
        assert_eq!(found, unsafe { buf.as_ptr().add(3) });

        // Every byte matching returns null, including through the vector path.
        let all = [7u8; 20];
        // Safety: `all` is valid for its full length.
        assert!(unsafe { codevar_wredit_k_find_not_byte(all.as_ptr(), 7, all.len()) }.is_null());

        // Safety: no bytes are read when `len == 0`.
        assert!(unsafe { codevar_wredit_k_find_not_byte(core::ptr::null(), 7, 0) }.is_null());
    }

    #[test]
    fn count_eq_u32_masks_words() {
        let src = [1u32, 2, 3, 2, 5, 2];
        // Safety: `src` is valid for its full length.
        let n = unsafe { codevar_wredit_k_count_eq_u32(src.as_ptr(), 2, 0xFF, src.len()) };
        assert_eq!(n, 3);

        // Safety: `src` is valid for its full length.
        let n = unsafe { codevar_wredit_k_count_eq_u32(src.as_ptr(), 9, 0xFF, src.len()) };
        assert_eq!(n, 0);

        // Safety: no words are read when `count == 0`.
        assert_eq!(
            unsafe { codevar_wredit_k_count_eq_u32(core::ptr::null(), 0, 0, 0) },
            0
        );
    }

    #[test]
    fn find_eq_u32_finds_first_match() {
        let src = [1u32, 2, 3, 2, 5, 2];
        // Safety: `src` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_eq_u32(src.as_ptr(), 2, 0xFF, src.len()) };
        assert_eq!(found, unsafe { src.as_ptr().add(1) });

        // Safety: `src` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_eq_u32(src.as_ptr(), 5, 0xFF, src.len()) };
        assert_eq!(found, unsafe { src.as_ptr().add(4) });

        // Safety: `src` is valid for its full length.
        assert!(unsafe { codevar_wredit_k_find_eq_u32(src.as_ptr(), 9, 0xFF, src.len()) }.is_null());

        // Safety: no words are read when `count == 0`.
        assert!(unsafe { codevar_wredit_k_find_eq_u32(core::ptr::null(), 0, 0, 0) }.is_null());
    }

    #[test]
    fn find_eq_u32_rev_finds_last_match() {
        let src = [1u32, 2, 3, 2, 5, 2];
        // Safety: `src` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_eq_u32_rev(src.as_ptr(), 2, 0xFF, src.len()) };
        assert_eq!(found, unsafe { src.as_ptr().add(5) });

        // Safety: `src` is valid for its full length.
        let found = unsafe { codevar_wredit_k_find_eq_u32_rev(src.as_ptr(), 1, 0xFF, src.len()) };
        assert_eq!(found, src.as_ptr());

        // Safety: `src` is valid for its full length.
        assert!(unsafe { codevar_wredit_k_find_eq_u32_rev(src.as_ptr(), 9, 0xFF, src.len()) }.is_null());

        // Safety: no words are read when `count == 0`.
        assert!(unsafe { codevar_wredit_k_find_eq_u32_rev(core::ptr::null(), 0, 0, 0) }.is_null());
    }

    #[test]
    fn memxor_xors_in_place() {
        let mut dst = [0xFFu8; 20];
        let src = [0x0Fu8; 20];
        let dst_ptr = dst.as_mut_ptr();
        let len = dst.len();
        // Safety: distinct buffers, both valid for `len` bytes.
        let ret = unsafe { codevar_wredit_k_memxor(dst_ptr, src.as_ptr(), len) };
        assert_eq!(ret, dst_ptr);
        assert!(dst.iter().all(|&b| b == 0xF0));

        // Scalar tail.
        let mut small = [0x01u8, 0x02, 0x03];
        let mask = [0xFFu8, 0x0F, 0xF0];
        let small_len = small.len();
        // Safety: distinct buffers, both valid for `small_len` bytes.
        unsafe { codevar_wredit_k_memxor(small.as_mut_ptr(), mask.as_ptr(), small_len) };
        assert_eq!(small, [0xFE, 0x0D, 0xF3]);

        // Safety: no bytes are accessed when `len == 0`.
        let null = unsafe { codevar_wredit_k_memxor(core::ptr::null_mut(), core::ptr::null(), 0) };
        assert!(null.is_null());
    }

    #[test]
    fn memcmp_orders_first_difference() {
        let a = [1u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 5];
        // Safety: both buffers are valid for 5 bytes.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(a.as_ptr(), b.as_ptr(), a.len()) },
            0
        );

        let c = [1u8, 2, 3, 4, 6];
        // Safety: both buffers are valid for 5 bytes.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(a.as_ptr(), c.as_ptr(), a.len()) },
            -1
        );
        // Safety: both buffers are valid for 5 bytes.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(c.as_ptr(), a.as_ptr(), a.len()) },
            1
        );

        // Difference inside the 8-byte word path.
        let long_a = [0u8; 40];
        let mut long_b = [0u8; 40];
        long_b[33] = 9;
        // Safety: both buffers are valid for 40 bytes.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(long_a.as_ptr(), long_b.as_ptr(), 40) },
            -1
        );
        // Safety: both buffers are valid for 40 bytes.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(long_b.as_ptr(), long_a.as_ptr(), 40) },
            1
        );

        // Safety: no bytes are read when `len == 0`.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(core::ptr::null(), core::ptr::null(), 0) },
            0
        );
    }

    #[test]
    fn sum_bytes_sums_zero_extended_bytes() {
        let src = [1u8, 2, 3, 4, 5];
        // Safety: `src` is valid for its full length.
        assert_eq!(unsafe { codevar_wredit_k_sum_bytes(src.as_ptr(), src.len()) }, 15);

        let big = [0xFFu8; 100];
        // Safety: `big` is valid for its full length.
        assert_eq!(
            unsafe { codevar_wredit_k_sum_bytes(big.as_ptr(), big.len()) },
            255 * 100
        );

        // Safety: no bytes are read when `len == 0`.
        assert_eq!(unsafe { codevar_wredit_k_sum_bytes(core::ptr::null(), 0) }, 0);
    }

    #[test]
    fn weighted_checksum_matches_known_values() {
        let src = [1u8, 2, 3];
        // Seed 0 starts at weight 1: 1*1 + 2*2 + 3*3 == 14.
        // Safety: `src` is valid for its full length.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(src.as_ptr(), 3, 0) },
            14
        );

        // Seed 250 starts at weight 251, then wraps to 1: 2*251 + 3*1.
        let two = [2u8, 3];
        // Safety: `two` is valid for its full length.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(two.as_ptr(), 2, 250) },
            2 * 251 + 3
        );

        // seed % 251 == 0 behaves like seed 0.
        // Safety: `src` is valid for its full length.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(src.as_ptr(), 3, 251) },
            unsafe { codevar_wredit_k_weighted_checksum(src.as_ptr(), 3, 0) }
        );

        // Safety: no bytes are read when `len == 0`.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(core::ptr::null(), 0, 7) },
            0
        );
    }

    #[test]
    fn utf8_count_starts_counts_lead_bytes() {
        let text = "h\u{e9}llo";
        let bytes = text.as_bytes();
        // 'h', 0xC3, 'l', 'l', 'o' are starts; 0xA9 is a continuation.
        // Safety: `bytes` is valid for its full length.
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(bytes.as_ptr(), bytes.len()) },
            5
        );

        // 20 ASCII bytes followed by 20 continuations, through the vector path.
        let mut mixed = [0x80u8; 40];
        for b in &mut mixed[..20] {
            *b = b'a';
        }
        // Safety: `mixed` is valid for its full length.
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(mixed.as_ptr(), mixed.len()) },
            20
        );

        // Safety: no bytes are read when `len == 0`.
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(core::ptr::null(), 0) },
            0
        );
    }

    #[test]
    fn replace_byte_replaces_and_counts() {
        let mut buf = [1u8, 2, 1, 3, 1];
        let buf_ptr = buf.as_mut_ptr();
        let len = buf.len();
        // Safety: `buf` is valid for reads and writes of `len` bytes.
        let n = unsafe { codevar_wredit_k_replace_byte(buf_ptr, 1, 9, len) };
        assert_eq!(n, 3);
        assert_eq!(buf, [9, 2, 9, 3, 9]);

        // Safety: `buf` is valid for reads and writes of `len` bytes.
        let n = unsafe { codevar_wredit_k_replace_byte(buf_ptr, 4, 0, len) };
        assert_eq!(n, 0);

        // 40 bytes exercises the vector path; the 0x55 tail is untouched.
        let mut long = [1u8; 40];
        long[39] = 0x55;
        let long_ptr = long.as_mut_ptr();
        let long_len = long.len();
        // Safety: `long` is valid for reads and writes of `long_len` bytes.
        let n = unsafe { codevar_wredit_k_replace_byte(long_ptr, 1, 2, long_len) };
        assert_eq!(n, 39);
        assert_eq!(long[39], 0x55);
        assert!(long[..39].iter().all(|&b| b == 2));

        // Safety: no bytes are accessed when `len == 0`.
        assert_eq!(
            unsafe { codevar_wredit_k_replace_byte(core::ptr::null_mut(), 1, 2, 0) },
            0
        );
    }

    #[test]
    fn reverse_bytes_reverses_into_dst() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let mut dst = [0u8; 6];
        let dst_ptr = dst.as_mut_ptr();
        let len = dst.len();
        // Safety: distinct buffers, both valid for `len` bytes.
        let ret = unsafe { codevar_wredit_k_reverse_bytes(dst_ptr, src.as_ptr(), len) };
        assert_eq!(ret, dst_ptr);
        assert_eq!(dst, [6, 5, 4, 3, 2, 1]);

        // 17 bytes exercises the 8-byte path and the scalar tail.
        let mut src17 = [0u8; 17];
        for (i, b) in src17.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut dst17 = [0u8; 17];
        let dst17_ptr = dst17.as_mut_ptr();
        // Safety: distinct buffers, both valid for 17 bytes.
        unsafe { codevar_wredit_k_reverse_bytes(dst17_ptr, src17.as_ptr(), 17) };
        assert!(
            dst17
                .iter()
                .enumerate()
                .all(|(i, &b)| b == (16 - i) as u8)
        );

        // Safety: no bytes are accessed when `len == 0`.
        let null = unsafe { codevar_wredit_k_reverse_bytes(core::ptr::null_mut(), core::ptr::null(), 0) };
        assert!(null.is_null());
    }

    #[test]
    fn fill_pattern8_repeats_and_tapers() {
        let pattern = 0x0807_0605_0403_0201u64;
        let mut dst = [0u8; 20];
        let dst_ptr = dst.as_mut_ptr();
        let len = dst.len();
        // Safety: `dst` is valid for writes of `len` bytes.
        let ret = unsafe { codevar_wredit_k_fill_pattern8(dst_ptr, pattern, len) };
        assert_eq!(ret, dst_ptr);
        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]);

        // The partial block uses the first `len % 8` pattern bytes.
        let mut tail = [0u8; 3];
        let tail_ptr = tail.as_mut_ptr();
        // Safety: `tail` is valid for writes of 3 bytes.
        unsafe { codevar_wredit_k_fill_pattern8(tail_ptr, pattern, 3) };
        assert_eq!(tail, [1, 2, 3]);

        // Safety: no bytes are written when `len == 0`.
        let null = unsafe { codevar_wredit_k_fill_pattern8(core::ptr::null_mut(), pattern, 0) };
        assert!(null.is_null());
    }

    #[test]
    fn byte_histogram_zeroes_then_counts() {
        let mut counts = [0xFFFF_FFFFu32; 256];
        let src = [0u8, 1, 1, 2, 255];
        let counts_ptr = counts.as_mut_ptr();
        // Safety: `counts` holds 256 counters and `src` is valid for 5 bytes.
        unsafe { codevar_wredit_k_byte_histogram(counts_ptr, src.as_ptr(), src.len()) };
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 1);
        assert_eq!(counts[255], 1);
        assert!(counts[3..255].iter().all(|&c| c == 0));

        // Zero length still zeroes all counters and never reads `src`.
        let mut zeroed = [0xFFFF_FFFFu32; 256];
        let zeroed_ptr = zeroed.as_mut_ptr();
        // Safety: `counts` holds 256 counters; `src` is not accessed.
        unsafe { codevar_wredit_k_byte_histogram(zeroed_ptr, core::ptr::null(), 0) };
        assert!(zeroed.iter().all(|&c| c == 0));
    }
}
