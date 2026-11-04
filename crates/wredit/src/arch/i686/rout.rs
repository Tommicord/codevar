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

//! General-purpose i686 kernel entry points.
//!
//! The implementations live in `rout.S` and follow the i686 cdecl
//! ABI: arguments are passed on the stack, the caller removes them, and
//! 64-bit results are returned in `edx:eax`.
//!
//! Routines that take a `len` or `count` of zero never dereference the
//! corresponding pointer, so null and dangling pointers are accepted in
//! that case.

unsafe extern "C" {
    /// Writes `value` to `dst[..len]` and returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_fill_bytes(dst: *mut u8, value: u8, len: usize) -> *mut u8;

    /// Writes `value` to `dst[..count]` and returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count` words. When `count` is
    /// zero the pointer is not dereferenced.
    pub fn codevar_wredit_k_fill_words(dst: *mut u32, value: u32, count: usize) -> *mut u32;

    /// Copies `count` words from `src` to `dst` and returns `dst`.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `count` words and must not overlap.
    /// When `count` is zero the pointers are not dereferenced.
    pub fn codevar_wredit_k_copy_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;

    /// Copies `count` words from `src` to `dst`, handling overlap, and
    /// returns `dst`.
    ///
    /// The copy runs forward when `dst < src` and backward otherwise.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `count` words; they may overlap.
    /// When `count` is zero the pointers are not dereferenced.
    pub fn codevar_wredit_k_move_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;

    /// Counts the bytes in `hay[..len]` equal to `needle`.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_count_byte(hay: *const u8, needle: u8, len: usize) -> usize;

    /// Returns the last occurrence of `needle` in `hay[..len]`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_find_byte_rev(hay: *const u8, needle: u8, len: usize) -> *const u8;

    /// Returns the first byte in `hay[..len]` equal to `a` or `b`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_find_any2(hay: *const u8, a: u8, b: u8, len: usize) -> *const u8;

    /// Returns the first byte in `hay[..len]` not equal to `needle`, or
    /// null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_find_not_byte(hay: *const u8, needle: u8, len: usize) -> *const u8;

    /// Counts words in `src[..count]` where `(word & mask) == value`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` words. When `count` is
    /// zero the pointer is not dereferenced.
    pub fn codevar_wredit_k_count_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> usize;

    /// Returns the first word in `src[..count]` matching
    /// `(word & mask) == value`, or null.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` words. When `count` is
    /// zero the pointer is not dereferenced.
    pub fn codevar_wredit_k_find_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32;

    /// Returns the last word in `src[..count]` matching
    /// `(word & mask) == value`, or null.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` words. When `count` is
    /// zero the pointer is not dereferenced.
    pub fn codevar_wredit_k_find_eq_u32_rev(
        src: *const u32,
        value: u32,
        mask: u32,
        count: usize,
    ) -> *const u32;

    /// XORs `src[..len]` into `dst[..len]` and returns `dst`.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `len` bytes. They may overlap only
    /// when `dst == src`. When `len` is zero the pointers are not
    /// dereferenced.
    pub fn codevar_wredit_k_memxor(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;

    /// Compares `a[..len]` with `b[..len]`.
    ///
    /// Returns `0` when equal, `-1` when the first differing byte in `a`
    /// is smaller, and `1` when it is greater.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for reads of `len` bytes. When `len` is
    /// zero the pointers are not dereferenced.
    pub fn codevar_wredit_k_memcmp(a: *const u8, b: *const u8, len: usize) -> i32;

    /// Sums the zero-extended bytes of `src[..len]` into a 64-bit value.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_sum_bytes(src: *const u8, len: usize) -> u64;

    /// Returns the weighted byte checksum of `src[..len]`.
    ///
    /// The initial weight is `seed % 251 + 1`; each byte is multiplied by
    /// the current weight, which then advances and wraps back to `1`
    /// after `251`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_weighted_checksum(src: *const u8, len: usize, seed: u32) -> u64;

    /// Counts the UTF-8 lead/ASCII bytes in `src[..len]`, that is bytes
    /// where `(byte & 0xC0) != 0x80`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_utf8_count_starts(src: *const u8, len: usize) -> usize;

    /// Replaces every byte equal to `from` with `to` in `dst[..len]` and
    /// returns the number of replacements.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for reads and writes of `len` bytes. When
    /// `len` is zero the pointer is not dereferenced.
    pub fn codevar_wredit_k_replace_byte(dst: *mut u8, from: u8, to: u8, len: usize) -> usize;

    /// Copies `src[..len]` into `dst[..len]` in reverse order and returns
    /// `dst`.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `len` bytes and must not overlap.
    /// When `len` is zero the pointers are not dereferenced.
    pub fn codevar_wredit_k_reverse_bytes(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;

    /// Repeats the eight bytes of `pattern` in native memory order across
    /// `dst[..len]` and returns `dst`.
    ///
    /// A trailing partial block uses the first `len % 8` pattern bytes.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes. When `len` is zero
    /// the pointer is not dereferenced.
    pub fn codevar_wredit_k_fill_pattern8(dst: *mut u8, pattern: u64, len: usize) -> *mut u8;

    /// Zeroes the 256 `u32` counters at `counts`, then increments
    /// `counts[byte]` for every byte of `src[..len]`.
    ///
    /// # Safety
    ///
    /// `counts` must be valid for reads and writes of 256 `u32` values.
    /// `src` must be valid for reads of `len` bytes. When `len` is zero
    /// `src` is not dereferenced.
    pub fn codevar_wredit_k_byte_histogram(counts: *mut u32, src: *const u8, len: usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    #[test]
    fn fill_bytes_handles_zero_and_tail() {
        let mut buf = [0u8; 13];
        // Safety: zero length with a null pointer; the routine must not
        // dereference it.
        let empty = unsafe { codevar_wredit_k_fill_bytes(ptr::null_mut(), 0xAA, 0) };
        assert_eq!(empty, ptr::null_mut());

        // Safety: `buf` is valid for writes of its whole length.
        let ret = unsafe { codevar_wredit_k_fill_bytes(buf.as_mut_ptr(), 0x5A, buf.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, [0x5A; 13]);

        let mut one = [0u8; 1];
        // Safety: `one` is valid for one byte.
        unsafe { codevar_wredit_k_fill_bytes(one.as_mut_ptr(), 0x11, 1) };
        assert_eq!(one, [0x11]);
    }

    #[test]
    fn fill_words_handles_zero_and_count() {
        let mut buf = [0u32; 5];
        // Safety: zero count with a null pointer.
        let empty = unsafe { codevar_wredit_k_fill_words(ptr::null_mut(), 0xDEAD_BEEF, 0) };
        assert_eq!(empty, ptr::null_mut());

        // Safety: `buf` is valid for writes of its whole length.
        let ret = unsafe { codevar_wredit_k_fill_words(buf.as_mut_ptr(), 0xDEAD_BEEF, buf.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, [0xDEAD_BEEF; 5]);

        let mut one = [0u32; 1];
        // Safety: `one` is valid for one word.
        unsafe { codevar_wredit_k_fill_words(one.as_mut_ptr(), 7, 1) };
        assert_eq!(one, [7]);
    }

    #[test]
    fn copy_words_handles_zero_and_copy() {
        let src = [1u32, 2, 3, 4, 5];
        let mut dst = [0u32; 5];
        // Safety: zero count with null pointers.
        let empty = unsafe { codevar_wredit_k_copy_words(ptr::null_mut(), ptr::null(), 0) };
        assert_eq!(empty, ptr::null_mut());

        // Safety: disjoint ranges valid for their whole length.
        let ret = unsafe { codevar_wredit_k_copy_words(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(ret, dst.as_mut_ptr());
        assert_eq!(dst, src);

        let mut one = [0u32; 1];
        // Safety: disjoint one-word ranges.
        unsafe { codevar_wredit_k_copy_words(one.as_mut_ptr(), src.as_ptr(), 1) };
        assert_eq!(one, [1]);
    }

    #[test]
    fn move_words_handles_both_overlap_directions() {
        // Safety: zero count with null pointers.
        let empty = unsafe { codevar_wredit_k_move_words(ptr::null_mut(), ptr::null(), 0) };
        assert_eq!(empty, ptr::null_mut());

        // Forward: dst < src, so the words must be copied low to high.
        let mut forward = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let base = forward.as_mut_ptr();
        // Safety: destination [2..6] and source [0..4] overlap inside the
        // same array; codevar_wredit_k_move_words is overlap-safe.
        let ret = unsafe { codevar_wredit_k_move_words(base.add(2), base, 4) };
        assert_eq!(ret, base.add(2));
        assert_eq!(forward, [0, 1, 0, 1, 2, 3, 6, 7]);

        // Backward: dst > src, so the words must be copied high to low.
        let mut backward = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let base = backward.as_mut_ptr();
        // Safety: destination [0..4] and source [2..6] overlap.
        let ret = unsafe { codevar_wredit_k_move_words(base, base.add(2), 4) };
        assert_eq!(ret, base);
        assert_eq!(backward, [2, 3, 4, 5, 4, 5, 6, 7]);

        // Same pointer: copying in place must leave the data intact.
        let mut in_place = [9u32, 8, 7];
        let base = in_place.as_mut_ptr();
        // Safety: both ranges are the same valid allocation.
        unsafe { codevar_wredit_k_move_words(base, base, in_place.len()) };
        assert_eq!(in_place, [9, 8, 7]);
    }

    #[test]
    fn count_byte_counts_and_handles_zero() {
        let hay = b"banana";
        // Safety: zero length with a null pointer.
        assert_eq!(unsafe { codevar_wredit_k_count_byte(ptr::null(), b'a', 0) }, 0);
        // Safety: `hay` is a valid slice for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_count_byte(hay.as_ptr(), b'a', hay.len()) },
            3
        );
        assert_eq!(
            unsafe { codevar_wredit_k_count_byte(hay.as_ptr(), b'n', hay.len()) },
            2
        );
        assert_eq!(
            unsafe { codevar_wredit_k_count_byte(hay.as_ptr(), b'z', hay.len()) },
            0
        );

        let one = [0x42u8];
        // Safety: one-byte valid slice.
        assert_eq!(unsafe { codevar_wredit_k_count_byte(one.as_ptr(), 0x42, 1) }, 1);
    }

    #[test]
    fn find_byte_rev_finds_last_and_misses() {
        let hay = b"abcabc";
        // Safety: zero length with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_find_byte_rev(ptr::null(), b'a', 0) },
            ptr::null()
        );
        // Safety: `hay` is valid for its whole length.
        let last_b = unsafe { codevar_wredit_k_find_byte_rev(hay.as_ptr(), b'b', hay.len()) };
        assert_eq!(last_b, unsafe { hay.as_ptr().add(4) });
        let last_a = unsafe { codevar_wredit_k_find_byte_rev(hay.as_ptr(), b'a', hay.len()) };
        assert_eq!(last_a, unsafe { hay.as_ptr().add(3) });
        assert_eq!(
            unsafe { codevar_wredit_k_find_byte_rev(hay.as_ptr(), b'z', hay.len()) },
            ptr::null()
        );

        let one = [0x77u8];
        // Safety: one-byte valid slice.
        assert_eq!(
            unsafe { codevar_wredit_k_find_byte_rev(one.as_ptr(), 0x77, 1) },
            one.as_ptr()
        );
    }

    #[test]
    fn find_any2_matches_each_byte_and_misses() {
        let hay = b"hello";
        // Safety: zero length with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_find_any2(ptr::null(), b'a', b'b', 0) },
            ptr::null()
        );
        // Safety: `hay` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_find_any2(hay.as_ptr(), b'l', b'z', hay.len()) },
            unsafe { hay.as_ptr().add(2) }
        );
        assert_eq!(
            unsafe { codevar_wredit_k_find_any2(hay.as_ptr(), b'z', b'o', hay.len()) },
            unsafe { hay.as_ptr().add(4) }
        );
        assert_eq!(
            unsafe { codevar_wredit_k_find_any2(hay.as_ptr(), b'z', b'q', hay.len()) },
            ptr::null()
        );

        let one = [0xA5u8];
        // Safety: one-byte valid slice.
        assert_eq!(
            unsafe { codevar_wredit_k_find_any2(one.as_ptr(), 0xA5, 0x00, 1) },
            one.as_ptr()
        );
    }

    #[test]
    fn find_not_byte_finds_mismatch_and_misses() {
        let hay = [0xAAu8, 0xAA, 0xBB, 0xAA];
        // Safety: zero length with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_find_not_byte(ptr::null(), 0xAA, 0) },
            ptr::null()
        );
        // Safety: `hay` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_find_not_byte(hay.as_ptr(), 0xAA, hay.len()) },
            unsafe { hay.as_ptr().add(2) }
        );

        let all = [0xAAu8; 4];
        // Safety: `all` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_find_not_byte(all.as_ptr(), 0xAA, all.len()) },
            ptr::null()
        );

        let one = [0x01u8];
        // Safety: one-byte valid slice.
        assert_eq!(
            unsafe { codevar_wredit_k_find_not_byte(one.as_ptr(), 0x01, 1) },
            ptr::null()
        );
    }

    #[test]
    fn count_eq_u32_uses_mask_and_handles_zero() {
        let words = [0x1234_5601u32, 0x1234_5602, 0xABCD_5603, 0x1234_5701];
        // Safety: zero count with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_count_eq_u32(ptr::null(), 0x5600, 0x0000_FF00, 0) },
            0
        );
        // Safety: `words` is valid for its whole length.
        let count =
            unsafe { codevar_wredit_k_count_eq_u32(words.as_ptr(), 0x5600, 0x0000_FF00, words.len()) };
        assert_eq!(count, 3);
        let none = unsafe { codevar_wredit_k_count_eq_u32(words.as_ptr(), 0x9900, 0x0000_FF00, words.len()) };
        assert_eq!(none, 0);

        let one = [0xFFFF_FFFFu32];
        // Safety: one-word valid slice.
        assert_eq!(
            unsafe { codevar_wredit_k_count_eq_u32(one.as_ptr(), 0xFFFF_FFFF, 0xFFFF_FFFF, 1) },
            1
        );
    }

    #[test]
    fn find_eq_u32_finds_first_and_misses() {
        let words = [0xBBu32, 0xAA, 0xAA, 0xCC];
        // Safety: zero count with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32(ptr::null(), 0xAA, 0xFFFF_FFFF, 0) },
            ptr::null()
        );
        // Safety: `words` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32(words.as_ptr(), 0xAA, 0xFFFF_FFFF, words.len()) },
            unsafe { words.as_ptr().add(1) }
        );
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32(words.as_ptr(), 0xDD, 0xFFFF_FFFF, words.len()) },
            ptr::null()
        );

        // A mask lets the match ignore the high half.
        let masked = [0x1234_0009u32, 0x5678_0009];
        // Safety: `masked` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32(masked.as_ptr(), 0x0009, 0x0000_FFFF, masked.len()) },
            masked.as_ptr()
        );
    }

    #[test]
    fn find_eq_u32_rev_finds_last_and_misses() {
        let words = [0xBBu32, 0xAA, 0xAA, 0xCC];
        // Safety: zero count with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32_rev(ptr::null(), 0xAA, 0xFFFF_FFFF, 0) },
            ptr::null()
        );
        // Safety: `words` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32_rev(words.as_ptr(), 0xAA, 0xFFFF_FFFF, words.len()) },
            unsafe { words.as_ptr().add(2) }
        );
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32_rev(words.as_ptr(), 0xDD, 0xFFFF_FFFF, words.len()) },
            ptr::null()
        );

        let one = [0x55u32];
        // Safety: one-word valid slice.
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32_rev(one.as_ptr(), 0x55, 0xFFFF_FFFF, 1) },
            one.as_ptr()
        );
    }

    #[test]
    fn memxor_handles_zero_word_tail_and_exact_overlap() {
        let mut buf = [0u8; 13];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as u8;
        }
        let src = [0xFFu8; 13];
        // Safety: zero length with null pointers.
        let empty = unsafe { codevar_wredit_k_memxor(ptr::null_mut(), ptr::null(), 0) };
        assert_eq!(empty, ptr::null_mut());

        // Safety: `buf` and `src` are valid for 13 bytes and disjoint.
        let ret = unsafe { codevar_wredit_k_memxor(buf.as_mut_ptr(), src.as_ptr(), buf.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        let expected: [u8; 13] = core::array::from_fn(|i| (i as u8) ^ 0xFF);
        assert_eq!(buf, expected);

        // Exact overlap: xor a buffer with itself to zero it.
        let mut same = [1u8, 2, 3, 4, 5];
        // Safety: destination and source are the identical range.
        unsafe { codevar_wredit_k_memxor(same.as_mut_ptr(), same.as_ptr(), same.len()) };
        assert_eq!(same, [0; 5]);
    }

    #[test]
    fn memcmp_reports_order_and_equal() {
        // Safety: zero length with null pointers.
        assert_eq!(unsafe { codevar_wredit_k_memcmp(ptr::null(), ptr::null(), 0) }, 0);

        let a = b"abcdef";
        let b = b"abcdef";
        // Safety: both slices are valid for their whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(a.as_ptr(), b.as_ptr(), a.len()) },
            0
        );

        let greater = b"abcdeg";
        // Safety: both slices are valid for their whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(a.as_ptr(), greater.as_ptr(), a.len()) },
            -1
        );
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(greater.as_ptr(), a.as_ptr(), a.len()) },
            1
        );

        // Difference only in the final byte exercises the scalar tail.
        let tail_less = b"abcdec";
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(a.as_ptr(), tail_less.as_ptr(), a.len()) },
            1
        );
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(tail_less.as_ptr(), a.as_ptr(), a.len()) },
            -1
        );

        let one_a = [9u8];
        let one_b = [10u8];
        // Safety: one-byte valid slices.
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(one_a.as_ptr(), one_b.as_ptr(), 1) },
            -1
        );
    }

    #[test]
    fn sum_bytes_accumulates_into_64_bits() {
        // Safety: zero length with a null pointer.
        assert_eq!(unsafe { codevar_wredit_k_sum_bytes(ptr::null(), 0) }, 0);

        let small = [1u8, 2, 3, 4, 5];
        // Safety: `small` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_sum_bytes(small.as_ptr(), small.len()) },
            15
        );

        // A byte tail after the four-byte loop.
        let tail = [255u8, 255, 255, 255, 255, 255, 255];
        // Safety: `tail` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_sum_bytes(tail.as_ptr(), tail.len()) },
            1785
        );

        // Large enough to need the high 32 bits of the result.
        let large = vec![0xFFu8; 17_000_000];
        // Safety: `large` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_sum_bytes(large.as_ptr(), large.len()) },
            255u64 * 17_000_000
        );
    }

    #[test]
    fn weighted_checksum_matches_known_values() {
        // Safety: zero length with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(ptr::null(), 0, 12345) },
            0
        );

        // seed = 0 -> weight starts at 1.
        let data = [1u8, 2, 3, 4];
        // Safety: `data` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(data.as_ptr(), data.len(), 0) },
            1 + 4 + 9 + 16
        );

        // seed = 251 -> weight starts at 1.
        let pair = [10u8, 20];
        // Safety: `pair` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(pair.as_ptr(), pair.len(), 251) },
            10 + 40
        );

        // seed = 250 -> weight starts at 251 and wraps to 1 for byte two.
        let wrap = [1u8, 2, 3];
        // Safety: `wrap` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(wrap.as_ptr(), wrap.len(), 250) },
            1 * 251 + 2 + 6
        );
    }

    #[test]
    fn utf8_count_starts_counts_lead_bytes() {
        // Safety: zero length with a null pointer.
        assert_eq!(unsafe { codevar_wredit_k_utf8_count_starts(ptr::null(), 0) }, 0);

        let text = b"h\xC3\xA9llo";
        // Safety: `text` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(text.as_ptr(), text.len()) },
            5
        );

        let continuations = [0x80u8, 0xBF, 0x80, 0x9F];
        // Safety: `continuations` is valid for its whole length.
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(continuations.as_ptr(), continuations.len()) },
            0
        );

        let one = [b'x'];
        // Safety: one-byte valid slice.
        assert_eq!(unsafe { codevar_wredit_k_utf8_count_starts(one.as_ptr(), 1) }, 1);
    }

    #[test]
    fn replace_byte_replaces_and_counts() {
        // Safety: zero length with a null pointer.
        assert_eq!(
            unsafe { codevar_wredit_k_replace_byte(ptr::null_mut(), 1, 9, 0) },
            0
        );

        let mut buf = [1u8, 2, 1, 3, 1];
        // Safety: `buf` is valid for reads and writes of its whole length.
        let replaced = unsafe { codevar_wredit_k_replace_byte(buf.as_mut_ptr(), 1, 9, buf.len()) };
        assert_eq!(replaced, 3);
        assert_eq!(buf, [9, 2, 9, 3, 9]);

        // No match leaves the data and the count unchanged.
        let mut none = [4u8, 5, 6];
        // Safety: `none` is valid for reads and writes of its whole length.
        let replaced = unsafe { codevar_wredit_k_replace_byte(none.as_mut_ptr(), 1, 0, none.len()) };
        assert_eq!(replaced, 0);
        assert_eq!(none, [4, 5, 6]);
    }

    #[test]
    fn reverse_bytes_reverses_odd_and_even_lengths() {
        // Safety: zero length with null pointers.
        let empty = unsafe { codevar_wredit_k_reverse_bytes(ptr::null_mut(), ptr::null(), 0) };
        assert_eq!(empty, ptr::null_mut());

        let even = b"abcdef";
        let mut even_dst = [0u8; 6];
        // Safety: disjoint ranges valid for their whole length.
        let ret = unsafe { codevar_wredit_k_reverse_bytes(even_dst.as_mut_ptr(), even.as_ptr(), even.len()) };
        assert_eq!(ret, even_dst.as_mut_ptr());
        assert_eq!(&even_dst, b"fedcba");

        let odd = b"abcde";
        let mut odd_dst = [0u8; 5];
        // Safety: disjoint ranges valid for their whole length.
        unsafe { codevar_wredit_k_reverse_bytes(odd_dst.as_mut_ptr(), odd.as_ptr(), odd.len()) };
        assert_eq!(&odd_dst, b"edcba");

        let one = [0x42u8];
        let mut one_dst = [0u8; 1];
        // Safety: disjoint one-byte ranges.
        unsafe { codevar_wredit_k_reverse_bytes(one_dst.as_mut_ptr(), one.as_ptr(), 1) };
        assert_eq!(one_dst, [0x42]);
    }

    #[test]
    fn fill_pattern8_handles_blocks_and_tail() {
        // Safety: zero length with a null pointer.
        let empty = unsafe { codevar_wredit_k_fill_pattern8(ptr::null_mut(), 0x0102_0304_0506_0708, 0) };
        assert_eq!(empty, ptr::null_mut());

        let pattern: u64 = 0x0807_0605_0403_0201;
        let pattern_bytes = pattern.to_ne_bytes();
        let mut expected = [0u8; 13];
        for (i, byte) in expected.iter_mut().enumerate() {
            *byte = pattern_bytes[i % pattern_bytes.len()];
        }

        let mut buf = [0u8; 13];
        // Safety: `buf` is valid for writes of its whole length.
        let ret = unsafe { codevar_wredit_k_fill_pattern8(buf.as_mut_ptr(), pattern, buf.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, expected);

        let mut exact = [0u8; 8];
        // Safety: `exact` is valid for writes of its whole length.
        unsafe { codevar_wredit_k_fill_pattern8(exact.as_mut_ptr(), pattern, 8) };
        assert_eq!(exact, pattern_bytes);

        let mut one = [0u8; 1];
        // Safety: `one` is valid for writes of one byte.
        unsafe { codevar_wredit_k_fill_pattern8(one.as_mut_ptr(), pattern, 1) };
        assert_eq!(one, [pattern_bytes[0]]);
    }

    #[test]
    fn byte_histogram_zeroes_and_counts() {
        let mut counts = [7u32; 256];
        // Safety: `counts` is valid for 256 counters; zero length with a
        // null source must not dereference it.
        unsafe { codevar_wredit_k_byte_histogram(counts.as_mut_ptr(), ptr::null(), 0) };
        assert!(counts.iter().all(|&count| count == 0));

        let data = [0u8, 255, 0, 1, 255, 0, 7];
        // Safety: `counts` is valid for 256 counters and `data` for its
        // whole length.
        unsafe { codevar_wredit_k_byte_histogram(counts.as_mut_ptr(), data.as_ptr(), data.len()) };
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[7], 1);
        assert_eq!(counts[255], 2);
        assert_eq!(counts.iter().sum::<u32>(), data.len() as u32);
    }
}
