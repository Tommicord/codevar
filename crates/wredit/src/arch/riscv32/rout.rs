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

//! RISC-V RV32 (ILP32) kernel bindings for `rout.S`.
//!
//! Every routine below is a raw `extern "C"` entry point assembled from
//! `rout.S`. They follow the ILP32 ABI: `a0`-`a7` for arguments,
//! `a0` for a 32-bit return and the `a0`/`a1` pair for a 64-bit (`u64`)
//! return. Only caller-saved registers are touched, so no callee-saved
//! state is disturbed and the stack is never used.

unsafe extern "C" {
    /// Writes `value` to `dst[..len]`; returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_fill_bytes(dst: *mut u8, value: u8, len: usize) -> *mut u8;

    /// Writes `value` to `dst[..count]`; returns `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count` `u32`s (or `count == 0`)
    /// and 4-byte aligned.
    pub fn codevar_wredit_k_fill_words(dst: *mut u32, value: u32, count: usize) -> *mut u32;

    /// Copies `count` words; the ranges must not overlap; returns `dst`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for writes of `count`
    /// `u32`s, both 4-byte aligned, with no overlap (or `count == 0`).
    pub fn codevar_wredit_k_copy_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;

    /// Overlap-safe word copy, forward when `dst < src`, backward
    /// otherwise; returns `dst`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for writes of `count`
    /// `u32`s, both 4-byte aligned; overlap is allowed (or `count == 0`).
    pub fn codevar_wredit_k_move_words(dst: *mut u32, src: *const u32, count: usize) -> *mut u32;

    /// Counts bytes equal to `needle`.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_count_byte(hay: *const u8, needle: u8, len: usize) -> usize;

    /// Returns the last occurrence of `needle` in `hay[..len]`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_find_byte_rev(hay: *const u8, needle: u8, len: usize) -> *const u8;

    /// Returns the first byte equal to `a` or `b`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_find_any2(hay: *const u8, a: u8, b: u8, len: usize) -> *const u8;

    /// Returns the first byte not equal to `needle`, or null.
    ///
    /// # Safety
    ///
    /// `hay` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_find_not_byte(hay: *const u8, needle: u8, len: usize) -> *const u8;

    /// Counts words where `(word & mask) == value`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` `u32`s (or `count == 0`)
    /// and 4-byte aligned.
    pub fn codevar_wredit_k_count_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> usize;

    /// Returns the first word matching `(word & mask) == value`, or null.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` `u32`s (or `count == 0`)
    /// and 4-byte aligned.
    pub fn codevar_wredit_k_find_eq_u32(src: *const u32, value: u32, mask: u32, count: usize) -> *const u32;

    /// Returns the last word matching `(word & mask) == value`, or null.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `count` `u32`s (or `count == 0`)
    /// and 4-byte aligned.
    pub fn codevar_wredit_k_find_eq_u32_rev(
        src: *const u32,
        value: u32,
        mask: u32,
        count: usize,
    ) -> *const u32;

    /// `dst[i] ^= src[i]` for every byte; returns `dst`.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `len` bytes; distinct ranges must
    /// not overlap, or they may be exactly the same address (or
    /// `len == 0`).
    pub fn codevar_wredit_k_memxor(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;

    /// Compares `a[..len]` with `b[..len]`.
    ///
    /// Returns 0 when equal, -1 when the first differing byte has
    /// `a < b`, and 1 otherwise.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_memcmp(a: *const u8, b: *const u8, len: usize) -> i32;

    /// Sums zero-extended bytes as a `u64`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_sum_bytes(src: *const u8, len: usize) -> u64;

    /// Weighted byte sum, returning a `u64`.
    ///
    /// `weight = seed % 251 + 1`; for each byte `sum += byte * weight`,
    /// then `weight += 1`, wrapping to 1 once it exceeds 251.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_weighted_checksum(src: *const u8, len: usize, seed: u32) -> u64;

    /// Counts bytes whose `(byte & 0xC0) != 0x80` (UTF-8 lead/ASCII).
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_utf8_count_starts(src: *const u8, len: usize) -> usize;

    /// Replaces every `from` with `to` in place; returns the count.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for reads and writes of `len` bytes (or
    /// `len == 0`).
    pub fn codevar_wredit_k_replace_byte(dst: *mut u8, from: u8, to: u8, len: usize) -> usize;

    /// Copies `src[..len]` reversed into `dst[..len]`; returns `dst`.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `len` bytes and must not overlap
    /// (or `len == 0`).
    pub fn codevar_wredit_k_reverse_bytes(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;

    /// Repeats the 8 bytes of `pattern` in native memory order; the final
    /// partial block uses the first `len % 8` pattern bytes.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes (or `len == 0`).
    pub fn codevar_wredit_k_fill_pattern8(dst: *mut u8, pattern: u64, len: usize) -> *mut u8;

    /// Zeros all 256 `u32` counters, then increments `counts[byte]` for
    /// every byte in `src[..len]`.
    ///
    /// # Safety
    ///
    /// `counts` must be valid for writes of 256 `u32`s and 4-byte
    /// aligned; `src` must be valid for reads of `len` bytes (or
    /// `len == 0`).
    pub fn codevar_wredit_k_byte_histogram(counts: *mut u32, src: *const u8, len: usize);
}

#[cfg(test)]
mod tests {
    use core::ptr::NonNull;

    use super::*;

    /// A dangling-but-aligned pointer, used to prove that a zero length
    /// never dereferences the argument.
    fn dangling<T>() -> *mut T {
        NonNull::<T>::dangling().as_ptr()
    }

    #[test]
    fn fill_bytes_covers_zero_word_and_tail_lengths() {
        let dst = dangling::<u8>();
        assert_eq!(unsafe { codevar_wredit_k_fill_bytes(dst, 0xAB, 0) }, dst);

        let mut one = [0u8; 1];
        unsafe { codevar_wredit_k_fill_bytes(one.as_mut_ptr(), 0x7F, 1) };
        assert_eq!(one, [0x7F]);

        let mut buf = [0u8; 19];
        let ret = unsafe { codevar_wredit_k_fill_bytes(buf.as_mut_ptr(), 0x5A, buf.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert!(buf.iter().all(|&b| b == 0x5A));
    }

    #[test]
    fn fill_words_covers_zero_one_and_many() {
        let dst = dangling::<u32>();
        assert_eq!(unsafe { codevar_wredit_k_fill_words(dst, 0xDEADBEEF, 0) }, dst);

        let mut one = [0u32; 1];
        unsafe { codevar_wredit_k_fill_words(one.as_mut_ptr(), 0x11223344, 1) };
        assert_eq!(one, [0x11223344]);

        let mut buf = [0u32; 5];
        unsafe { codevar_wredit_k_fill_words(buf.as_mut_ptr(), 0xABCDEF01, buf.len()) };
        assert_eq!(buf, [0xABCDEF01; 5]);
    }

    #[test]
    fn copy_words_covers_zero_single_and_block() {
        let dst = dangling::<u32>();
        assert_eq!(
            unsafe { codevar_wredit_k_copy_words(dst, dangling::<u32>(), 0) },
            dst
        );

        let one_src = [42u32];
        let mut one_dst = [0u32];
        unsafe { codevar_wredit_k_copy_words(one_dst.as_mut_ptr(), one_src.as_ptr(), 1) };
        assert_eq!(one_dst, [42]);

        let src: [u32; 11] = core::array::from_fn(|i| i as u32 + 1);
        let mut buf = [0u32; 11];
        let ret = unsafe { codevar_wredit_k_copy_words(buf.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, src);
    }

    #[test]
    fn move_words_handles_both_overlap_directions() {
        let dst = dangling::<u32>();
        assert_eq!(
            unsafe { codevar_wredit_k_move_words(dst, dangling::<u32>(), 0) },
            dst
        );

        // Forward overlap: dst > src, so the copy must run backwards.
        let mut fwd = [1u32, 2, 3, 4, 5, 6, 7, 8];
        unsafe {
            let base = fwd.as_mut_ptr();
            codevar_wredit_k_move_words(base.add(2), base, 4);
        }
        assert_eq!(fwd, [1, 2, 1, 2, 3, 4, 7, 8]);

        // Backward overlap: dst < src, so the copy must run forwards.
        let mut bwd = [1u32, 2, 3, 4, 5, 6, 7, 8];
        unsafe {
            let base = bwd.as_mut_ptr();
            codevar_wredit_k_move_words(base, base.add(2), 4);
        }
        assert_eq!(bwd, [3, 4, 5, 6, 5, 6, 7, 8]);

        // Identical same-pointer path.
        let mut same = [9u32, 8, 7];
        unsafe {
            let base = same.as_mut_ptr();
            codevar_wredit_k_move_words(base, base, 3);
        }
        assert_eq!(same, [9, 8, 7]);
    }

    #[test]
    fn count_byte_counts_zero_and_hits() {
        let hay = dangling::<u8>();
        assert_eq!(unsafe { codevar_wredit_k_count_byte(hay, b's', 0) }, 0);

        let text = b"mississippi";
        assert_eq!(
            unsafe { codevar_wredit_k_count_byte(text.as_ptr(), b's', text.len()) },
            4
        );
        assert_eq!(
            unsafe { codevar_wredit_k_count_byte(text.as_ptr(), b'z', text.len()) },
            0
        );
    }

    #[test]
    fn find_byte_rev_finds_last_or_null() {
        let hay = dangling::<u8>();
        assert_eq!(
            unsafe { codevar_wredit_k_find_byte_rev(hay, b'x', 0) },
            core::ptr::null()
        );

        let text = b"abcabc";
        let hit = unsafe { codevar_wredit_k_find_byte_rev(text.as_ptr(), b'a', text.len()) };
        assert_eq!(hit, unsafe { text.as_ptr().add(3) });
        let miss = unsafe { codevar_wredit_k_find_byte_rev(text.as_ptr(), b'z', text.len()) };
        assert_eq!(miss, core::ptr::null());
    }

    #[test]
    fn find_any2_and_find_not_byte() {
        let hay = dangling::<u8>();
        assert_eq!(
            unsafe { codevar_wredit_k_find_any2(hay, b'a', b'b', 0) },
            core::ptr::null()
        );
        assert_eq!(
            unsafe { codevar_wredit_k_find_not_byte(hay, b'a', 0) },
            core::ptr::null()
        );

        let text = b"hello";
        let any = unsafe { codevar_wredit_k_find_any2(text.as_ptr(), b'l', b'o', text.len()) };
        assert_eq!(any, unsafe { text.as_ptr().add(2) });
        let none = unsafe { codevar_wredit_k_find_any2(text.as_ptr(), b'x', b'y', text.len()) };
        assert_eq!(none, core::ptr::null());

        let padded = b"  x";
        let first = unsafe { codevar_wredit_k_find_not_byte(padded.as_ptr(), b' ', padded.len()) };
        assert_eq!(first, unsafe { padded.as_ptr().add(2) });
        let all = b"     ";
        let miss = unsafe { codevar_wredit_k_find_not_byte(all.as_ptr(), b' ', all.len()) };
        assert_eq!(miss, core::ptr::null());
    }

    #[test]
    fn count_and_find_eq_u32_respect_the_mask() {
        let src = dangling::<u32>();
        assert_eq!(unsafe { codevar_wredit_k_count_eq_u32(src, 1, 0xF, 0) }, 0);
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32(src, 1, 0xF, 0) },
            core::ptr::null()
        );
        assert_eq!(
            unsafe { codevar_wredit_k_find_eq_u32_rev(src, 1, 0xF, 0) },
            core::ptr::null()
        );

        let words = [0x11u32, 0x21, 0x01, 0x1F];
        assert_eq!(
            unsafe { codevar_wredit_k_count_eq_u32(words.as_ptr(), 0x1, 0xF, words.len()) },
            3
        );
        let first = unsafe { codevar_wredit_k_find_eq_u32(words.as_ptr(), 0x1, 0xF, words.len()) };
        assert_eq!(first, words.as_ptr());
        let last = unsafe { codevar_wredit_k_find_eq_u32_rev(words.as_ptr(), 0x1, 0xF, words.len()) };
        assert_eq!(last, unsafe { words.as_ptr().add(2) });
    }

    #[test]
    fn memxor_word_path_tail_and_exact_overlap() {
        let dst = dangling::<u8>();
        assert_eq!(unsafe { codevar_wredit_k_memxor(dst, dangling::<u8>(), 0) }, dst);

        let mut buf = [0xFFu8, 0x0F, 0x00, 0xAA, 0x55, 0x33, 0x77];
        let src = [0x0Fu8, 0xF0, 0xFF, 0x55, 0xAA, 0xCC, 0x88];
        let ret = unsafe { codevar_wredit_k_memxor(buf.as_mut_ptr(), src.as_ptr(), buf.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, [0xF0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

        let mut same = [0x12u8, 0x34, 0x56, 0x78, 0x9A];
        unsafe { codevar_wredit_k_memxor(same.as_mut_ptr(), same.as_ptr(), same.len()) };
        assert_eq!(same, [0; 5]);
    }

    #[test]
    fn memcmp_orders_bytes() {
        let a = dangling::<u8>();
        assert_eq!(unsafe { codevar_wredit_k_memcmp(a, a, 0) }, 0);

        let x = [1u8, 2, 3];
        let y = [1u8, 2, 3];
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(x.as_ptr(), y.as_ptr(), x.len()) },
            0
        );

        let lo = [1u8, 2, 3];
        let hi = [1u8, 2, 4];
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(lo.as_ptr(), hi.as_ptr(), lo.len()) },
            -1
        );
        assert_eq!(
            unsafe { codevar_wredit_k_memcmp(hi.as_ptr(), lo.as_ptr(), lo.len()) },
            1
        );
    }

    #[test]
    fn sum_bytes_zero_and_known_values() {
        assert_eq!(unsafe { codevar_wredit_k_sum_bytes(dangling::<u8>(), 0) }, 0);
        let src = [1u8, 2, 3, 255];
        assert_eq!(
            unsafe { codevar_wredit_k_sum_bytes(src.as_ptr(), src.len()) },
            261
        );
    }

    #[test]
    fn weighted_checksum_known_values_and_wrap() {
        // seed 0 -> weight 1, so the sum is 1*1 + 2*2 + 3*3.
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(dangling::<u8>(), 0, 0) },
            0
        );
        let src = [1u8, 2, 3];
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(src.as_ptr(), src.len(), 0) },
            14
        );

        // seed 250 -> first weight 251, which wraps to 1 for the next byte.
        let pair = [1u8, 1];
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(pair.as_ptr(), pair.len(), 250) },
            252
        );

        // Weight cycles through exactly 251 distinct values.
        let ones = [1u8; 252];
        // 251 ones at weights 1..=251 then one at weight 1 again.
        let expected: u64 = (1..=251u64).sum::<u64>() + 1;
        assert_eq!(
            unsafe { codevar_wredit_k_weighted_checksum(ones.as_ptr(), ones.len(), 0) },
            expected
        );
    }

    #[test]
    fn utf8_count_starts_counts_lead_bytes() {
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(dangling::<u8>(), 0) },
            0
        );
        // 'A', continuation, 2-byte lead, continuation, continuation.
        let src = [0x41u8, 0x80, 0xC3, 0xA9, 0x80];
        assert_eq!(
            unsafe { codevar_wredit_k_utf8_count_starts(src.as_ptr(), src.len()) },
            2
        );
    }

    #[test]
    fn replace_byte_counts_replacements() {
        let dst = dangling::<u8>();
        assert_eq!(unsafe { codevar_wredit_k_replace_byte(dst, b'a', b'b', 0) }, 0);

        let mut buf = *b"banana";
        let n = unsafe { codevar_wredit_k_replace_byte(buf.as_mut_ptr(), b'a', b'o', buf.len()) };
        assert_eq!(n, 3);
        assert_eq!(&buf, b"bonono");
    }

    #[test]
    fn reverse_bytes_copies_reversed() {
        let dst = dangling::<u8>();
        assert_eq!(
            unsafe { codevar_wredit_k_reverse_bytes(dst, dangling::<u8>(), 0) },
            dst
        );

        let src = [1u8, 2, 3, 4, 5];
        let mut buf = [0u8; 5];
        let ret = unsafe { codevar_wredit_k_reverse_bytes(buf.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(ret, buf.as_mut_ptr());
        assert_eq!(buf, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn fill_pattern8_aligned_unaligned_and_tail() {
        let dst = dangling::<u8>();
        assert_eq!(unsafe { codevar_wredit_k_fill_pattern8(dst, 0, 0) }, dst);

        // Little-endian byte order 00 01 02 03 04 05 06 07.
        let pattern: u64 = 0x0706_0504_0302_0100;
        let mut aligned = [0u8; 10];
        unsafe { codevar_wredit_k_fill_pattern8(aligned.as_mut_ptr(), pattern, aligned.len()) };
        assert_eq!(aligned, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]);

        // Unaligned start exercises the byte-at-a-time path.
        let mut raw = [0u8; 12];
        unsafe {
            let p = raw.as_mut_ptr().add(1);
            codevar_wredit_k_fill_pattern8(p, pattern, 8);
        }
        assert_eq!(&raw[1..9], &[0, 1, 2, 3, 4, 5, 6, 7]);

        // Partial block shorter than 8 bytes.
        let mut small = [0u8; 5];
        unsafe { codevar_wredit_k_fill_pattern8(small.as_mut_ptr(), pattern, small.len()) };
        assert_eq!(small, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn byte_histogram_zero_and_known_counts() {
        let mut counts = [0xFFFF_FFFFu32; 256];
        unsafe { codevar_wredit_k_byte_histogram(counts.as_mut_ptr(), dangling::<u8>(), 0) };
        assert!(counts.iter().all(|&c| c == 0));

        let src = b"aabbc";
        unsafe { codevar_wredit_k_byte_histogram(counts.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(counts[b'a' as usize], 2);
        assert_eq!(counts[b'b' as usize], 2);
        assert_eq!(counts[b'c' as usize], 1);
        assert_eq!(counts[b'z' as usize], 0);
    }
}
