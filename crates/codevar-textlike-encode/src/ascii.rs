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

use cfg_if::cfg_if;

pub(crate) fn is_ascii(s: &[u8; STRIDE]) -> bool {
    s.iter().all(|b| *b < 0x80)
}

pub(crate) fn is_basic_latin(s: &[u16; STRIDE]) -> bool {
    s.iter().all(|b| *b < 0x80)
}

pub(crate) fn is_utf16_latin1(s: &[u16; STRIDE]) -> bool {
    s.iter().all(|b| *b < 0x100)
}

pub(crate) fn copy_stride(src_stride: &[u8; STRIDE], dst_stride: &mut [u8; STRIDE]) {
    *dst_stride = *src_stride;
}

pub(crate) fn unpack_stride(src_stride: &[u8; STRIDE], dst_stride: &mut [u16; STRIDE]) {
    src_stride
        .iter()
        .zip(dst_stride.iter_mut())
        .for_each(|(s, d)| *d = *s as u16);
}

pub(crate) fn pack_stride(src_stride: &[u16; STRIDE], dst_stride: &mut [u8; STRIDE]) {
    src_stride
        .iter()
        .zip(dst_stride.iter_mut())
        .for_each(|(s, d)| *d = *s as u8);
}

fn copy_stride_tail(src_stride: &[u8; 16], dst_stride: &mut [u8; 16]) -> (u8, usize) {
    for (i, (s, d)) in src_stride
        .iter()
        .zip(dst_stride.iter_mut())
        .enumerate()
    {
        let c = *s;
        if c >= 0x80 {
            return (c, i);
        }
        *d = c;
    }
    (0, 0)
}

fn unpack_stride_tail(src_stride: &[u8; 16], dst_stride: &mut [u16; 16]) -> (u8, usize) {
    for (i, (s, d)) in src_stride
        .iter()
        .zip(dst_stride.iter_mut())
        .enumerate()
    {
        let c = *s;
        if c >= 0x80 {
            return (c, i);
        }
        *d = c as u16;
    }
    (0, 0)
}

fn pack_stride_tail(src_stride: &[u16; 16], dst_stride: &mut [u8; 16]) -> (u16, usize) {
    for (i, (s, d)) in src_stride
        .iter()
        .zip(dst_stride.iter_mut())
        .enumerate()
    {
        let c = *s;
        if c >= 0x80 {
            return (c, i);
        }
        *d = c as u8;
    }
    (0, 0)
}

fn validate_ascii_stride_tail(stride: &[u8; 16]) -> (u8, usize) {
    for (i, s) in stride.iter().enumerate() {
        let b = *s;
        if b >= 0x80 {
            return (b, i);
        }
    }
    debug_assert!(false);
    (0, 0)
}

fn validate_basic_latin_stride_tail(stride: &[u16; 16]) -> usize {
    for (i, s) in stride.iter().enumerate() {
        if *s >= 0x80 {
            return i;
        }
    }
    debug_assert!(false);
    0
}

fn ascii_to_ascii_stride(src_stride: &[u8; STRIDE], dst_stride: &mut [u8; STRIDE]) -> Option<(u8, usize)> {
    if is_ascii(src_stride) {
        copy_stride(src_stride, dst_stride);
        return None;
    }
    Some(copy_stride_tail(src_stride, dst_stride))
}

fn ascii_to_basic_latin_stride(
    src_stride: &[u8; STRIDE],
    dst_stride: &mut [u16; STRIDE],
) -> Option<(u8, usize)> {
    if is_ascii(src_stride) {
        unpack_stride(src_stride, dst_stride);
        return None;
    }
    Some(unpack_stride_tail(src_stride, dst_stride))
}

fn basic_latin_to_ascii_stride(
    src_stride: &[u16; STRIDE],
    dst_stride: &mut [u8; STRIDE],
) -> Option<(u16, usize)> {
    if is_basic_latin(src_stride) {
        pack_stride(src_stride, dst_stride);
        return None;
    }
    Some(pack_stride_tail(src_stride, dst_stride))
}

fn validate_ascii_stride(stride: &[u8; STRIDE]) -> Option<(u8, usize)> {
    if is_ascii(stride) {
        return None;
    }
    Some(validate_ascii_stride_tail(stride))
}

fn validate_ascii_double_stride(strides: &[[u8; STRIDE]; 2]) -> Option<(u8, usize)> {
    if let Some((b, pos)) = validate_ascii_stride(&strides[0]) {
        return Some((b, pos));
    }
    if let Some((b, pos)) = validate_ascii_stride(&strides[1]) {
        return Some((b, STRIDE + pos));
    }
    None
}

fn validate_basic_latin_stride(stride: &[u16; STRIDE]) -> Option<usize> {
    if is_basic_latin(stride) {
        return None;
    }
    Some(validate_basic_latin_stride_tail(stride))
}

pub(crate) const STRIDE: usize = 16;

pub(crate) const MAX_STRIDE_SIZE: usize = STRIDE;

#[allow(unused_macros)]
macro_rules! ascii_copy_impl_double {
    ($name:ident, $stride:ident, $double_stride:ident, $src_unit:ty, $dst_unit:ty) => {
        #[inline(always)]
        pub(crate) fn $name(src: &[$src_unit], dst: &mut [$dst_unit]) -> Option<($src_unit, usize)> {
            // Make both the same length here to have the chunks and tail match
            let len = core::cmp::min(src.len(), dst.len());
            let mut consumed = 0usize;
            let (src_strides, src_tail) = src[..len].as_chunks::<STRIDE>();
            let (dst_strides, dst_tail) = dst[..len].as_chunks_mut::<STRIDE>();
            if let Some((src_first_stride, src_strides_tail)) = src_strides.split_first() {
                if let Some((dst_first_stride, dst_strides_tail)) = dst_strides.split_first_mut() {
                    if let Some(pos) = $stride(src_first_stride, dst_first_stride) {
                        return Some(pos);
                    }
                    consumed = STRIDE;

                    let (src_double_strides, src_single_stride) = src_strides_tail.as_chunks::<2>();
                    let (dst_double_strides, dst_single_stride) = dst_strides_tail.as_chunks_mut::<2>();
                    for (src_double_stride, dst_double_stride) in src_double_strides
                        .iter()
                        .zip(dst_double_strides.iter_mut())
                    {
                        if let Some((c, pos)) = $double_stride(src_double_stride, dst_double_stride) {
                            return Some((c, consumed + pos));
                        }
                        consumed += STRIDE << 1;
                    }
                    for (src_stride, dst_stride) in src_single_stride
                        .iter()
                        .zip(dst_single_stride.iter_mut())
                    {
                        if let Some((c, pos)) = $stride(src_stride, dst_stride) {
                            return Some((c, consumed + pos));
                        }
                        consumed += STRIDE;
                    }
                } else {
                    debug_assert!(false);
                }
            }
            for (src_slot, dst_slot) in src_tail.iter().zip(dst_tail.iter_mut()) {
                let c = *src_slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                *dst_slot = c as $dst_unit;
                consumed += 1;
            }
            None
        }
    };
}

#[allow(unused_macros)]
macro_rules! ascii_copy_impl_single {
    ($name:ident, $stride:ident, $double_stride:ident, $src_unit:ty, $dst_unit:ty) => {
        #[inline(always)]
        pub fn $name(src: &[$src_unit], dst: &mut [$dst_unit]) -> Option<($src_unit, usize)> {
            // Make both the same length here to have the chunks and tail match.
            let len = core::cmp::min(src.len(), dst.len());
            let mut consumed = 0usize;
            let (src_strides, src_tail) = src[..len].as_chunks::<STRIDE>();
            let (dst_strides, dst_tail) = dst[..len].as_chunks_mut::<STRIDE>();
            for (src_stride, dst_stride) in src_strides.iter().zip(dst_strides.iter_mut()) {
                if let Some((c, pos)) = $stride(src_stride, dst_stride) {
                    return Some((c, consumed + pos));
                }
                consumed += STRIDE;
            }
            for (src_slot, dst_slot) in src_tail.iter().zip(dst_tail.iter_mut()) {
                let c = *src_slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                *dst_slot = c as $dst_unit;
                consumed += 1;
            }
            None
        }
    };
}

cfg_if! {
    if #[cfg(target_arch = "arm")] {
        /// Mask with the high bit set in every byte of a `usize`, used to
        /// detect non-ASCII bytes with word-at-a-time scans.
        const ASCII_MASK: usize = usize::from_ne_bytes([0x80; core::mem::size_of::<usize>()]);

        #[inline(always)]
        fn ascii_valid_impl(bytes: &[u8]) -> Option<(u8, usize)> {
            let mut consumed = 0usize;
            #[allow(clippy::never_loop)]
            'outer: loop {
                let (strides, tail) = bytes.as_chunks::<{core::mem::size_of::<usize>()}>();
                if let Some((first_stride, strides_tail)) = strides.split_first() {
                    let first_usize = usize::from_ne_bytes(*first_stride);
                    if first_usize & ASCII_MASK != 0 {
                        break 'outer;
                    }
                    consumed = core::mem::size_of::<usize>();

                    let (double_strides, single_stride) = strides_tail.as_chunks::<2>();
                    for double_stride in double_strides.iter() {
                        let first_usize = usize::from_ne_bytes(double_stride[0]);
                        let second_usize = usize::from_ne_bytes(double_stride[1]);
                        if (first_usize | second_usize) & ASCII_MASK != 0 {
                            break 'outer;
                        }
                        consumed += core::mem::size_of::<usize>() * 2;
                    }
                    for stride in single_stride.iter() {
                        let last_usize = usize::from_ne_bytes(*stride);
                        if last_usize & ASCII_MASK != 0 {
                            break 'outer;
                        }
                        consumed += core::mem::size_of::<usize>();
                    }
                }
                for slot in tail.iter() {
                    let c = *slot;
                    if c >= 0x80 {
                        return Some((c, consumed));
                    }
                    consumed += 1;
                }
                return None;
            }
            let tail = &bytes[consumed..];
            for slot in tail.iter() {
                let c = *slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                consumed += 1;
            }
            debug_assert!(false);
            None
        }

    } else if #[cfg(target_endian = "little")] {
        #[inline(always)]
        fn ascii_valid_impl(bytes: &[u8]) -> Option<(u8, usize)> {
            let mut consumed = 0usize;
            let (strides, tail) = bytes.as_chunks::<STRIDE>();
            if let Some((first_stride, strides_tail)) = strides.split_first() {
                if let Some((c, pos)) = validate_ascii_stride(first_stride) {
                    return Some((c, pos));
                }
                consumed = STRIDE;

                let (double_strides, single_stride) = strides_tail.as_chunks::<2>();
                for double_stride in double_strides {
                    if let Some((c, pos)) = validate_ascii_double_stride(double_stride) {
                        return Some((c, consumed + pos));
                    }
                    consumed += STRIDE * 2;
                }
                for stride in single_stride.iter() {
                    if let Some((c, pos)) = validate_ascii_stride(stride) {
                        return Some((c, consumed + pos));
                    }
                    consumed += STRIDE;
                }
            }
            for slot in tail.iter() {
                let c = *slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                consumed += 1;
            }
            None
        }

    } else {
        #[inline(always)]
        fn ascii_valid_impl(bytes: &[u8]) -> Option<(u8, usize)> {
            let mut consumed = 0usize;
            let (strides, tail) = bytes.as_chunks::<STRIDE>();
            for stride in strides.iter() {
                if let Some((b, pos)) = validate_ascii_stride(stride) {
                    return Some((b, consumed + pos));
                }
                consumed += STRIDE;
            }
            for slot in tail.iter() {
                let b = *slot;
                if b >= 0x80 {
                    return Some((b, consumed));
                }
                consumed += 1;
            }
            None
        }

    }
}

cfg_if! {
    if #[cfg(all(feature = "simd-accel", target_endian = "little", not(target_arch = "arm")))] {
        ascii_copy_impl_double!(
            ascii_to_ascii_impl,
            ascii_to_ascii_stride,
            ascii_to_ascii_double_stride,
            u8,
            u8
        );
        ascii_copy_impl_double!(
            ascii_to_basic_latin_impl,
            ascii_to_basic_latin_stride,
            ascii_to_basic_latin_double_stride,
            u8,
            u16
        );
    } else {
        ascii_copy_impl_single!(
            ascii_to_ascii_impl,
            ascii_to_ascii_stride,
            ascii_to_ascii_double_stride,
            u8,
            u8
        );
        ascii_copy_impl_single!(
            ascii_to_basic_latin_impl,
            ascii_to_basic_latin_stride,
            ascii_to_basic_latin_double_stride,
            u8,
            u16
        );
    }
}

cfg_if! {
    if #[cfg(all(feature = "simd-accel", target_endian = "little"))] {
        ascii_copy_impl_double!(
            basic_latin_to_ascii_impl,
            basic_latin_to_ascii_stride,
            basic_latin_to_ascii_double_stride,
            u16,
            u8
        );
    } else {
        ascii_copy_impl_single!(
            basic_latin_to_ascii_impl,
            basic_latin_to_ascii_stride,
            basic_latin_to_ascii_double_stride,
            u16,
            u8
        );
    }
}

macro_rules! ascii_copy {
    ($name:ident, $impl:ident, $src_unit:ty, $dst_unit:ty) => {
        #[inline(always)]
        pub(crate) fn $name(src: &[$src_unit], dst: &mut [$dst_unit]) -> Option<($src_unit, usize)> {
            $impl(src, dst)
        }
    };
}
ascii_copy!(ascii_to_ascii, ascii_to_ascii_impl, u8, u8);
ascii_copy!(ascii_to_basic_latin, ascii_to_basic_latin_impl, u8, u16);
ascii_copy!(basic_latin_to_ascii, basic_latin_to_ascii_impl, u16, u8);

pub fn ascii_valid_up_to(bytes: &[u8]) -> usize {
    ascii_valid_impl(bytes).map_or(bytes.len(), |(_, pos)| pos)
}

pub fn validate_ascii(bytes: &[u8]) -> Option<(u8, usize)> {
    ascii_valid_impl(bytes)
}

pub(crate) fn iso_2022_jp_ascii_valid_up_to(bytes: &[u8]) -> usize {
    for (i, b_ref) in bytes.iter().enumerate() {
        let b = *b_ref;
        if b >= 0x80 || b == 0x1B || b == 0x0E || b == 0x0F {
            return i;
        }
    }
    bytes.len()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const SENT: u8 = 0xEE;
    const SENT16: u16 = 0xEEE;

    fn reference_first_non_ascii(bytes: &[u8]) -> Option<(u8, usize)> {
        bytes
            .iter()
            .enumerate()
            .find(|(_, b)| **b >= 0x80)
            .map(|(i, b)| (*b, i))
    }

    fn assert_scanners_agree(bytes: &[u8]) {
        let expected = reference_first_non_ascii(bytes);
        assert_eq!(validate_ascii(bytes), expected, "len={}", bytes.len());
        assert_eq!(
            ascii_valid_up_to(bytes),
            expected.map_or(bytes.len(), |(_, i)| i),
            "len={}",
            bytes.len()
        );
    }

    fn ascii_run(len: usize) -> Vec<u8> {
        vec![b'a'; len]
    }

    fn run_with_byte_at(len: usize, pos: usize, value: u8) -> Vec<u8> {
        assert!(pos < len, "test bug: pos {pos} >= len {len}");
        let mut bytes = ascii_run(len);
        bytes[pos] = value;
        bytes
    }

    fn patterned(len: usize, salt: usize) -> Vec<u8> {
        (0..len)
            .map(|i| ((i * 11 + salt * 5) % 256) as u8)
            .collect()
    }

    fn basic_latin_run(len: usize) -> Vec<u16> {
        (0..len).map(|i| (i % 0x80) as u16).collect()
    }

    fn run16_with_at(len: usize, pos: usize, value: u16) -> Vec<u16> {
        assert!(pos < len, "test bug: pos {pos} >= len {len}");
        let mut units = basic_latin_run(len);
        units[pos] = value;
        units
    }

    #[test]
    fn ascii_valid_up_to_empty_slice_is_zero() {
        assert_eq!(ascii_valid_up_to(&[]), 0);
        assert_eq!(validate_ascii(&[]), None);
    }

    #[test]
    fn ascii_valid_up_to_all_ascii_inputs_len_zero_to_forty() {
        for len in 0..=40usize {
            let bytes = ascii_run(len);
            assert_eq!(ascii_valid_up_to(&bytes), len, "len={len}");
        }
    }

    #[test]
    fn validate_ascii_returns_none_for_all_ascii_inputs_len_zero_to_forty() {
        for len in 0..=40usize {
            let bytes = ascii_run(len);
            assert_eq!(validate_ascii(&bytes), None, "len={len}");
        }
    }

    #[test]
    fn validate_ascii_and_valid_up_to_agree_on_last_byte_len_one_to_forty() {
        for len in 1..=40usize {
            let bytes = run_with_byte_at(len, len - 1, 0x80);
            assert_scanners_agree(&bytes);
            assert_eq!(validate_ascii(&bytes), Some((0x80, len - 1)), "len={len}");
            assert_eq!(ascii_valid_up_to(&bytes), len - 1, "len={len}");
        }
    }

    #[test]
    fn validate_ascii_and_valid_up_to_agree_on_first_byte_len_one_to_forty() {
        for len in 1..=40usize {
            let bytes = run_with_byte_at(len, 0, 0xFF);
            assert_scanners_agree(&bytes);
            assert_eq!(validate_ascii(&bytes), Some((0xFF, 0)), "len={len}");
            assert_eq!(ascii_valid_up_to(&bytes), 0, "len={len}");
        }
    }

    #[test]
    fn validate_ascii_and_valid_up_to_agree_at_every_position_of_sixty_five_bytes() {
        let mut bytes = ascii_run(65);
        for pos in 0..65usize {
            bytes[pos] = 0x80;
            assert_scanners_agree(&bytes);
            bytes[pos] = b'a';
        }
    }

    #[test]
    fn validate_ascii_matches_reference_for_patterned_inputs() {
        for len in 0..=80usize {
            for salt in 0..6usize {
                assert_scanners_agree(&patterned(len, salt));
            }
        }
    }

    #[test]
    fn validate_ascii_reports_first_of_two_high_bytes() {
        let mut bytes = ascii_run(40);
        bytes[20] = 0xC3;
        bytes[5] = 0xFF;
        assert_eq!(validate_ascii(&bytes), Some((0xFF, 5)));
        assert_eq!(ascii_valid_up_to(&bytes), 5);
    }

    #[test]
    fn ascii_valid_up_to_boundary_between_0x7f_and_0x80() {
        for len in 1..=40usize {
            let mut bytes = ascii_run(len);
            bytes[len - 1] = 0x7F;
            assert_eq!(ascii_valid_up_to(&bytes), len, "len={len}");

            let mut bytes = ascii_run(len);
            bytes[len - 1] = 0x80;
            assert_eq!(ascii_valid_up_to(&bytes), len - 1, "len={len}");
        }
    }

    #[test]
    fn ascii_valid_up_to_detects_non_ascii_values_across_high_range() {
        for value in [0x80u8, 0x81, 0xBF, 0xC3, 0xFE, 0xFF] {
            assert_eq!(validate_ascii(&[value]), Some((value, 0)));
            let bytes = run_with_byte_at(25, 12, value);
            assert_eq!(validate_ascii(&bytes), Some((value, 12)));
            assert_eq!(ascii_valid_up_to(&bytes), 12);
        }
    }

    #[test]
    fn ascii_valid_up_to_len16_non_ascii_at_last_lane_of_first_stride() {
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(16, 15, 0x80)), 15);
        assert_eq!(ascii_valid_up_to(&ascii_run(16)), 16);
    }

    #[test]
    fn ascii_valid_up_to_len32_non_ascii_at_second_stride_edges() {
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(32, 16, 0x80)), 16);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(32, 31, 0x81)), 31);
        assert_eq!(ascii_valid_up_to(&ascii_run(32)), 32);
    }

    #[test]
    fn ascii_valid_up_to_len33_non_ascii_hits_single_stride_remainder_and_tail() {
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(33, 16, 0x80)), 16);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(33, 32, 0x80)), 32);
        assert_eq!(ascii_valid_up_to(&ascii_run(33)), 33);
    }

    #[test]
    fn ascii_valid_up_to_len48_covers_double_stride_windows() {
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(48, 20, 0x80)), 20);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(48, 31, 0xFE)), 31);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(48, 32, 0x80)), 32);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(48, 47, 0xFF)), 47);
        assert_eq!(ascii_valid_up_to(&ascii_run(48)), 48);
    }

    #[test]
    fn ascii_valid_up_to_len64_covers_double_then_single_stride() {
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(64, 16, 0x80)), 16);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(64, 40, 0x80)), 40);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(64, 50, 0x90)), 50);
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(64, 63, 0xFF)), 63);
    }

    #[test]
    fn ascii_valid_up_to_len65_non_ascii_in_final_tail_byte() {
        assert_eq!(ascii_valid_up_to(&run_with_byte_at(65, 64, 0x80)), 64);
        assert_eq!(ascii_valid_up_to(&ascii_run(65)), 65);
    }

    #[test]
    fn ascii_valid_up_to_at_every_stride_crossing_position() {
        for pos in [15usize, 16, 31, 32, 47, 48, 63, 64] {
            let bytes = run_with_byte_at(65, pos, 0x80);
            assert_eq!(ascii_valid_up_to(&bytes), pos, "pos={pos}");
            assert_eq!(validate_ascii(&bytes), Some((0x80, pos)), "pos={pos}");
        }
    }

    #[test]
    fn ascii_valid_up_to_long_input_non_ascii_at_the_end() {
        let bytes = run_with_byte_at(1000, 999, 0x80);
        assert_eq!(ascii_valid_up_to(&bytes), 999);
        assert_eq!(validate_ascii(&bytes), Some((0x80, 999)));
    }

    #[test]
    fn validate_ascii_returns_none_for_large_ascii_buffer() {
        let bytes = ascii_run(8192);
        assert_eq!(validate_ascii(&bytes), None);
        assert_eq!(ascii_valid_up_to(&bytes), 8192);
    }

    #[test]
    fn stride_constants_match_documented_width() {
        let stride = STRIDE;
        let max = MAX_STRIDE_SIZE;
        assert_eq!(stride, 16);
        assert_eq!(max, stride);
    }

    #[test]
    fn is_ascii_accepts_0x7f_in_every_lane() {
        assert!(is_ascii(&[0x7F; STRIDE]));
        assert!(is_ascii(&[0x00; STRIDE]));
        assert!(is_ascii(&[b'\x1B'; STRIDE]));
    }

    #[test]
    fn is_ascii_rejects_0x80_in_any_lane() {
        for lane in 0..STRIDE {
            let mut stride = [b'a'; STRIDE];
            stride[lane] = 0x80;
            assert!(!is_ascii(&stride), "lane={lane}");
        }
    }

    #[test]
    fn is_basic_latin_uses_the_0x80_cut_not_the_0x100_cut() {
        assert!(is_basic_latin(&[0x007F; STRIDE]));
        assert!(!is_basic_latin(&[0x0080; STRIDE]));
        assert!(!is_basic_latin(&[0x00FF; STRIDE]));
        assert!(!is_basic_latin(&[0x0100; STRIDE]));
        for lane in 0..STRIDE {
            let mut stride = [b'a' as u16; STRIDE];
            stride[lane] = 0x0080;
            assert!(!is_basic_latin(&stride), "lane={lane}");
        }
    }

    #[test]
    fn is_utf16_latin1_accepts_0x00ff_and_rejects_0x0100() {
        assert!(is_utf16_latin1(&[0x0000; STRIDE]));
        assert!(is_utf16_latin1(&[0x007F; STRIDE]));
        assert!(is_utf16_latin1(&[0x0080; STRIDE]));
        assert!(is_utf16_latin1(&[0x00FF; STRIDE]));
        assert!(!is_utf16_latin1(&[0x0100; STRIDE]));
    }

    #[test]
    fn copy_stride_writes_all_sixteen_slots() {
        let src: [u8; STRIDE] = core::array::from_fn(|i| i as u8 + 1);
        let mut dst = [SENT; STRIDE];
        copy_stride(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn unpack_stride_zero_extends_each_byte() {
        let src: [u8; STRIDE] = core::array::from_fn(|i| (i * 17) as u8);
        let mut dst = [SENT16; STRIDE];
        unpack_stride(&src, &mut dst);
        for lane in 0..STRIDE {
            assert_eq!(dst[lane], src[lane] as u16, "lane={lane}");
        }
        let mut high_dst = [0u16; STRIDE];
        unpack_stride(&[0xFF; STRIDE], &mut high_dst);
        assert_eq!(high_dst, [0x00FFu16; STRIDE]);
    }

    #[test]
    fn pack_stride_truncates_above_0xff_by_raw_cast() {
        let src: [u16; STRIDE] = core::array::from_fn(|i| 0x0100 + i as u16);
        let mut dst = [SENT; STRIDE];
        pack_stride(&src, &mut dst);
        assert_eq!(dst[0], 0x00);
        assert_eq!(dst[1], 0x01);
        assert_eq!(dst[15], 0x0F);
    }

    #[test]
    fn ascii_to_ascii_empty_src_never_writes() {
        let mut dst = [SENT; 8];
        assert_eq!(ascii_to_ascii(&[], &mut dst), None);
        assert_eq!(dst, [SENT; 8]);
    }

    #[test]
    fn ascii_to_ascii_empty_dst_returns_none_even_for_non_ascii_src() {
        let mut dst: [u8; 0] = [];
        assert_eq!(ascii_to_ascii(&[0x80, 0x41], &mut dst), None);
    }

    #[test]
    fn ascii_to_ascii_full_overlap_lengths_zero_to_forty() {
        for len in 0..=40usize {
            let src: Vec<u8> = (0..len).map(|i| (i % 0x80) as u8).collect();
            let mut dst = vec![SENT; len];
            assert_eq!(ascii_to_ascii(&src, &mut dst), None, "len={len}");
            assert_eq!(dst, src, "len={len}");
        }
    }

    #[test]
    fn ascii_to_ascii_multiple_of_stride_all_ascii_returns_none() {
        for len in [16usize, 32, 48, 64] {
            let src = ascii_run(len);
            let mut dst = vec![SENT; len];
            assert_eq!(ascii_to_ascii(&src, &mut dst), None, "len={len}");
            assert_eq!(dst, src, "len={len}");
        }
    }

    #[test]
    fn ascii_to_ascii_non_ascii_first_byte_consumes_nothing() {
        let mut dst = [SENT; 16];
        assert_eq!(ascii_to_ascii(&[0xFF; 16], &mut dst), Some((0xFF, 0)));
        assert_eq!(dst, [SENT; 16]);
    }

    #[test]
    fn ascii_to_ascii_copies_prefix_then_reports_position_in_stride() {
        let src = run_with_byte_at(40, 5, 0x80);
        let mut dst = vec![SENT; 40];
        assert_eq!(ascii_to_ascii(&src, &mut dst), Some((0x80, 5)));
        assert_eq!(&dst[..5], &src[..5]);
        assert!(dst[5..].iter().all(|&b| b == SENT));
    }

    #[test]
    fn ascii_to_ascii_tail_byte_after_full_stride_consumes_sixteen() {
        let src = run_with_byte_at(17, 16, 0x90);
        let mut dst = vec![SENT; 17];
        assert_eq!(ascii_to_ascii(&src, &mut dst), Some((0x90, 16)));
        assert_eq!(&dst[..16], &src[..16]);
        assert_eq!(dst[16], SENT);
    }

    #[test]
    fn ascii_to_ascii_ignores_non_ascii_beyond_dst_len() {
        let src = run_with_byte_at(32, 20, 0x80);
        let mut dst = vec![SENT; 16];
        assert_eq!(ascii_to_ascii(&src, &mut dst), None);
        assert_eq!(&dst[..], &src[..16]);
    }

    #[test]
    fn ascii_to_ascii_dst_longer_than_src_leaves_surplus_untouched() {
        let src = ascii_run(20);
        let mut dst = vec![SENT; 40];
        assert_eq!(ascii_to_ascii(&src, &mut dst), None);
        assert_eq!(&dst[..20], &src[..]);
        assert!(dst[20..].iter().all(|&b| b == SENT));
    }

    #[test]
    fn ascii_to_ascii_boundary_0x7f_copied_0x80_stops() {
        let mut src = ascii_run(16);
        src[15] = 0x7F;
        let mut dst = [SENT; 16];
        assert_eq!(ascii_to_ascii(&src, &mut dst), None);
        assert_eq!(dst[15], 0x7F);

        src[15] = 0x80;
        let mut dst = [SENT; 16];
        assert_eq!(ascii_to_ascii(&src, &mut dst), Some((0x80, 15)));
        assert_eq!(&dst[..15], &src[..15]);
        assert_eq!(dst[15], SENT);
    }

    #[test]
    fn ascii_to_ascii_non_ascii_at_every_position_len_33() {
        for pos in 0..33usize {
            let src = run_with_byte_at(33, pos, 0x80);
            let mut dst = vec![SENT; 33];
            assert_eq!(ascii_to_ascii(&src, &mut dst), Some((0x80, pos)), "pos={pos}");
            assert_eq!(&dst[..pos], &src[..pos], "pos={pos}");
            assert!(dst[pos..].iter().all(|&b| b == SENT), "pos={pos}");
        }
    }

    #[test]
    fn ascii_to_ascii_non_ascii_in_second_stride_of_overlap_len_40() {
        let src = run_with_byte_at(40, 25, 0xAB);
        let mut dst = vec![SENT; 40];
        assert_eq!(ascii_to_ascii(&src, &mut dst), Some((0xAB, 25)));
        assert_eq!(&dst[..25], &src[..25]);
        assert!(dst[25..].iter().all(|&b| b == SENT));
    }

    #[test]
    fn ascii_to_ascii_matches_plain_copy_for_long_ascii_input() {
        let src = ascii_run(257);
        let mut dst = vec![SENT; 257];
        assert_eq!(ascii_to_ascii(&src, &mut dst), None);
        assert_eq!(dst, src);
    }

    #[test]
    fn ascii_to_basic_latin_empty_inputs_return_none() {
        let mut dst = [SENT16; 8];
        assert_eq!(ascii_to_basic_latin(&[], &mut dst), None);
        assert_eq!(dst, [SENT16; 8]);

        let mut narrow: [u16; 0] = [];
        assert_eq!(ascii_to_basic_latin(&[0x80, 0x41], &mut narrow), None);
    }

    #[test]
    fn ascii_to_basic_latin_widens_full_overlap_lengths_zero_to_forty() {
        for len in 0..=40usize {
            let src: Vec<u8> = (0..len).map(|i| (i % 0x80) as u8).collect();
            let mut dst = vec![SENT16; len];
            assert_eq!(ascii_to_basic_latin(&src, &mut dst), None, "len={len}");
            for (d, s) in dst.iter().zip(src.iter()) {
                assert_eq!(*d, *s as u16);
            }
        }
    }

    #[test]
    fn ascii_to_basic_latin_multiple_of_stride_all_ascii_returns_none() {
        for len in [16usize, 32, 48] {
            let src = ascii_run(len);
            let mut dst = vec![SENT16; len];
            assert_eq!(ascii_to_basic_latin(&src, &mut dst), None, "len={len}");
            assert!(dst.iter().all(|&w| w < 0x80), "len={len}");
        }
    }

    #[test]
    fn ascii_to_basic_latin_non_ascii_first_byte_reports_u8_value() {
        let mut dst = [SENT16; 16];
        assert_eq!(ascii_to_basic_latin(&[0xC3; 16], &mut dst), Some((0xC3, 0)));
        assert_eq!(dst, [SENT16; 16]);
    }

    #[test]
    fn ascii_to_basic_latin_reports_consumed_when_dst_is_shorter() {
        let src = run_with_byte_at(33, 20, 0x80);
        let mut dst = vec![SENT16; 24];
        assert_eq!(ascii_to_basic_latin(&src, &mut dst), Some((0x80, 20)));
        for lane in 0..20 {
            assert_eq!(dst[lane], src[lane] as u16, "lane={lane}");
        }
        assert!(dst[20..].iter().all(|&w| w == SENT16));
    }

    #[test]
    fn ascii_to_basic_latin_tail_position_right_after_stride() {
        let src = run_with_byte_at(20, 16, 0x80);
        let mut dst = vec![SENT16; 20];
        assert_eq!(ascii_to_basic_latin(&src, &mut dst), Some((0x80, 16)));
        for lane in 0..16 {
            assert_eq!(dst[lane], src[lane] as u16, "lane={lane}");
        }
        assert!(dst[16..].iter().all(|&w| w == SENT16));
    }

    #[test]
    fn ascii_to_basic_latin_ignores_high_byte_beyond_overlap() {
        let src = run_with_byte_at(40, 30, 0xFF);
        let mut dst = vec![SENT16; 20];
        assert_eq!(ascii_to_basic_latin(&src, &mut dst), None);
        assert!(dst.iter().all(|&w| w < 0x80));
    }

    #[test]
    fn ascii_to_basic_latin_non_ascii_at_every_position_len_32() {
        for pos in 0..32usize {
            let src = run_with_byte_at(32, pos, 0x81);
            let mut dst = vec![SENT16; 32];
            assert_eq!(
                ascii_to_basic_latin(&src, &mut dst),
                Some((0x81, pos)),
                "pos={pos}"
            );
            for lane in 0..pos {
                assert_eq!(dst[lane], src[lane] as u16, "pos={pos} lane={lane}");
            }
            assert!(dst[pos..].iter().all(|&w| w == SENT16), "pos={pos}");
        }
    }

    #[test]
    fn basic_latin_to_ascii_empty_inputs_return_none() {
        let mut dst = [SENT; 8];
        assert_eq!(basic_latin_to_ascii(&[], &mut dst), None);
        assert_eq!(dst, [SENT; 8]);

        let mut narrow: [u8; 0] = [];
        assert_eq!(basic_latin_to_ascii(&[0x0100], &mut narrow), None);
    }

    #[test]
    fn basic_latin_to_ascii_copies_full_overlap_lengths_zero_to_forty() {
        for len in 0..=40usize {
            let src = basic_latin_run(len);
            let mut dst = vec![SENT; len];
            assert_eq!(basic_latin_to_ascii(&src, &mut dst), None, "len={len}");
            for (d, s) in dst.iter().zip(src.iter()) {
                assert_eq!(*d, *s as u8);
            }
        }
    }

    #[test]
    fn basic_latin_to_ascii_boundary_0x7f_passes_0x80_stops() {
        let mut src = [b'a'; 16].map(|c| c as u16);
        src[15] = 0x007F;
        let mut dst = [SENT; 16];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), None);
        assert_eq!(dst[15], 0x7F);

        src[15] = 0x0080;
        let mut dst = [SENT; 16];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), Some((0x0080, 15)));
        for lane in 0..15 {
            assert_eq!(dst[lane], src[lane] as u8, "lane={lane}");
        }
        assert_eq!(dst[15], SENT);
    }

    #[test]
    fn basic_latin_to_ascii_reports_exact_u16_value_for_0xff_and_0x100() {
        let src = run16_with_at(20, 10, 0x00FF);
        let mut dst = vec![SENT; 20];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), Some((0x00FF, 10)));

        let src = run16_with_at(20, 18, 0x0100);
        let mut dst = vec![SENT; 20];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), Some((0x0100, 18)));
    }

    #[test]
    fn basic_latin_to_ascii_stride_hit_copies_prefix_only() {
        let src = run16_with_at(40, 20, 0x0080);
        let mut dst = vec![SENT; 40];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), Some((0x0080, 20)));
        for lane in 0..20 {
            assert_eq!(dst[lane], src[lane] as u8, "lane={lane}");
        }
        assert!(dst[20..].iter().all(|&b| b == SENT));
    }

    #[test]
    fn basic_latin_to_ascii_dst_shorter_stops_at_overlap() {
        let src = basic_latin_run(33);
        let mut dst = vec![SENT; 16];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), None);
        for lane in 0..16 {
            assert_eq!(dst[lane], src[lane] as u8, "lane={lane}");
        }
    }

    #[test]
    fn basic_latin_to_ascii_ignores_high_code_beyond_overlap() {
        let src = run16_with_at(33, 20, 0x0100);
        let mut dst = vec![SENT; 16];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), None);
    }

    #[test]
    fn basic_latin_to_ascii_tail_position_after_two_strides() {
        let src = run16_with_at(33, 32, 0x0080);
        let mut dst = vec![SENT; 33];
        assert_eq!(basic_latin_to_ascii(&src, &mut dst), Some((0x0080, 32)));
        assert!(dst[32] == SENT);
    }

    #[test]
    fn basic_latin_roundtrip_with_ascii_to_basic_latin() {
        for len in 1..=40usize {
            let src: Vec<u8> = (0..len).map(|i| (i % 0x80) as u8).collect();
            let mut wide = vec![SENT16; len];
            assert_eq!(ascii_to_basic_latin(&src, &mut wide), None, "len={len}");
            let mut back = vec![SENT; len];
            assert_eq!(basic_latin_to_ascii(&wide, &mut back), None, "len={len}");
            assert_eq!(back, src, "len={len}");
        }
    }

    #[test]
    fn iso_2022_empty_input_is_zero() {
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&[]), 0);
    }

    #[test]
    fn iso_2022_pure_ascii_matches_len_zero_to_forty() {
        for len in 0..=40usize {
            assert_eq!(iso_2022_jp_ascii_valid_up_to(&ascii_run(len)), len, "len={len}");
        }
    }

    #[test]
    fn iso_2022_stops_at_esc() {
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&[0x1B]), 0);
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&run_with_byte_at(20, 7, 0x1B)), 7);
    }

    #[test]
    fn iso_2022_stops_at_shift_out() {
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&[0x0E]), 0);
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&run_with_byte_at(20, 3, 0x0E)), 3);
    }

    #[test]
    fn iso_2022_stops_at_shift_in() {
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&[0x0F]), 0);
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&run_with_byte_at(20, 11, 0x0F)), 11);
    }

    #[test]
    fn iso_2022_stops_at_high_bytes_like_ascii_valid_up_to() {
        for value in [0x80u8, 0xA5, 0xFE, 0xFF] {
            let bytes = run_with_byte_at(24, 13, value);
            assert_eq!(iso_2022_jp_ascii_valid_up_to(&bytes), 13);
            assert_eq!(iso_2022_jp_ascii_valid_up_to(&bytes), ascii_valid_up_to(&bytes));
        }
    }

    #[test]
    fn iso_2022_allows_nul_bell_tab_cr_and_0x7f() {
        let bytes = [0x00u8, 0x01, 0x07, 0x09, 0x0D, 0x1A, 0x7F];
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&bytes), bytes.len());
    }

    #[test]
    fn iso_2022_returns_earliest_trigger() {
        let mut bytes = ascii_run(10);
        bytes[5] = 0x80;
        bytes[3] = 0x1B;
        bytes[1] = 0x0F;
        assert_eq!(iso_2022_jp_ascii_valid_up_to(&bytes), 1);
    }

    #[test]
    fn iso_2022_long_input_trigger_beyond_strides() {
        assert_eq!(
            iso_2022_jp_ascii_valid_up_to(&run_with_byte_at(100, 60, 0x1B)),
            60
        );
    }

    #[test]
    fn iso_2022_never_lags_ascii_valid_up_to() {
        for len in 0..=60usize {
            for salt in 0..4usize {
                let mut bytes = patterned(len, salt);
                if len > 3 {
                    bytes[len / 2] = 0x1B;
                }
                let iso = iso_2022_jp_ascii_valid_up_to(&bytes);
                let plain = ascii_valid_up_to(&bytes);
                assert!(iso <= plain, "len={len} salt={salt}");
            }
        }
    }

    #[test]
    fn ascii_valid_up_to_treats_low_control_and_escape_bytes_as_ascii() {
        let bytes = [0x00u8, 0x01, 0x07, 0x0E, 0x0F, 0x1B, 0x7F];
        assert_eq!(ascii_valid_up_to(&bytes), bytes.len());
        assert_eq!(validate_ascii(&bytes), None);
    }

    #[test]
    fn validate_ascii_single_byte_input_short_circuits_stride_path() {
        assert_eq!(validate_ascii(&[0x80]), Some((0x80, 0)));
        assert_eq!(validate_ascii(&[0x7F]), None);
        assert_eq!(ascii_valid_up_to(&[0x80]), 0);
        assert_eq!(ascii_valid_up_to(&[0x7F]), 1);
    }

    #[test]
    fn validate_ascii_len16_scans_every_lane_of_single_stride() {
        for pos in 0..16usize {
            let bytes = run_with_byte_at(16, pos, 0x80);
            assert_eq!(ascii_valid_up_to(&bytes), pos, "pos={pos}");
            assert_eq!(validate_ascii(&bytes), Some((0x80, pos)), "pos={pos}");
        }
    }
}
