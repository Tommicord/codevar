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

use crate::ascii::{ascii_to_basic_latin, basic_latin_to_ascii};
use crate::encoding::{DecoderResult, EncoderResult, VariantDecoder};
use crate::handles::{ByteSource, CopyAsciiResult, Space, Utf8Destination};

#[derive(Debug, Clone)]
pub struct SingleByteDecoder {
    pub(crate) table: &'static [u16; 128],
}

impl SingleByteDecoder {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(data: &'static [u16; 128]) -> VariantDecoder {
        VariantDecoder::SingleByte(SingleByteDecoder { table: data })
    }

    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        Some(byte_length)
    }

    pub fn max_utf8_buffer_length_without_replacement(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_mul(3)
    }

    pub fn max_utf8_buffer_length(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_mul(3)
    }

    pub fn decode_to_utf8_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        _last: bool,
    ) -> (DecoderResult, usize, usize) {
        let mut source = ByteSource::new(src);
        let mut dest = Utf8Destination::new(dst);
        'outermost: loop {
            match dest.copy_ascii_from_check_space_bmp(&mut source) {
                CopyAsciiResult::Stop(ret) => return ret,
                CopyAsciiResult::GoOn((mut non_ascii, mut handle)) => 'middle: loop {
                    // SAFETY: `non_ascii` is a u8 byte >=0x80, from the invariants
                    // on Utf8Destination::copy_ascii_from_check_space_bmp()
                    let mapped = unsafe {
                        *(self
                            .table
                            .get_unchecked(non_ascii as usize - 0x80usize))
                    };
                    // let mapped = self.table[non_ascii as usize - 0x80usize];
                    if mapped == 0u16 {
                        return (
                            DecoderResult::Malformed(1, 0),
                            source.consumed(),
                            handle.written(),
                        );
                    }
                    let dest_again = handle.write_bmp_excl_ascii(mapped);
                    match source.check_available() {
                        Space::Full(src_consumed) => {
                            return (DecoderResult::InputEmpty, src_consumed, dest_again.written());
                        }
                        Space::Available(source_handle) => {
                            match dest_again.check_space_bmp() {
                                Space::Full(dst_written) => {
                                    return (
                                        DecoderResult::OutputFull,
                                        source_handle.consumed(),
                                        dst_written,
                                    );
                                }
                                Space::Available(mut destination_handle) => {
                                    let (mut b, unread_handle) = source_handle.read();
                                    let source_again = unread_handle.commit();
                                    'innermost: loop {
                                        if b > 127 {
                                            non_ascii = b;
                                            handle = destination_handle;
                                            continue 'middle;
                                        }
                                        let dest_again_again = destination_handle.write_ascii(b);
                                        if b < 60 {
                                            // We've got punctuation
                                            match source_again.check_available() {
                                                Space::Full(src_consumed_again) => {
                                                    return (
                                                        DecoderResult::InputEmpty,
                                                        src_consumed_again,
                                                        dest_again_again.written(),
                                                    );
                                                }
                                                Space::Available(source_handle_again) => {
                                                    match dest_again_again.check_space_bmp() {
                                                        Space::Full(dst_written_again) => {
                                                            return (
                                                                DecoderResult::OutputFull,
                                                                source_handle_again.consumed(),
                                                                dst_written_again,
                                                            );
                                                        }
                                                        Space::Available(destination_handle_again) => {
                                                            let (b_again, unread_handle_again) =
                                                                source_handle_again.read();
                                                            unread_handle_again.commit();
                                                            b = b_again;
                                                            destination_handle = destination_handle_again;
                                                            continue 'innermost;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        // We've got markup or ASCII text
                                        continue 'outermost;
                                    }
                                }
                            }
                        }
                    }
                },
            }
        }
    }

    #[inline(always)]
    pub fn decode_to_utf16_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        _last: bool,
    ) -> (DecoderResult, usize, usize) {
        let (pending, length) = if dst.len() < src.len() {
            (DecoderResult::OutputFull, dst.len())
        } else {
            (DecoderResult::InputEmpty, src.len())
        };
        // Safety invariant: converted <= length. Quite often we have `converted < length`
        // which will be separately marked.
        let mut converted = 0usize;
        'outermost: loop {
            // SAFETY: length is the minimum length, `src/dst + x` will always be valid for reads/writes of `len - x`
            match ascii_to_basic_latin(&src[converted..], &mut dst[converted..]) {
                None => {
                    return (pending, length, length);
                }
                Some((mut non_ascii, consumed)) => {
                    // Safety invariant: `converted <= length` upheld, since this can only consume
                    // up to `length - converted` bytes.
                    //
                    // Furthermore, in this context,
                    // we can assume `converted < length` since this branch is only ever hit when
                    // ascii_to_basic_latin fails to consume the entire slice
                    converted += consumed;
                    'middle: loop {
                        // `converted` doesn't count the reading of `non_ascii` yet.
                        // Since the non-ASCIIness of `non_ascii` is hidden from
                        // the optimizer, it can't figure out that it's OK to
                        // statically omit the bound check when accessing
                        // `[u16; 128]` with an index
                        // `non_ascii as usize - 0x80usize`.
                        //
                        // SAFETY: We can rely on `non_ascii` being between `0x80` and `0xFF` due to
                        // the invariants of `ascii_to_basic_latin()`, and our table has enough space for that.
                        let mapped = unsafe {
                            *(self
                                .table
                                .get_unchecked(non_ascii as usize - 0x80usize))
                        };
                        // let mapped = self.table[non_ascii as usize - 0x80usize];
                        if mapped == 0u16 {
                            return (
                                DecoderResult::Malformed(1, 0),
                                converted + 1, // +1 `for non_ascii`
                                converted,
                            );
                        }
                        unsafe {
                            // SAFETY: As mentioned above, `converted < length`
                            *(dst.get_unchecked_mut(converted)) = mapped;
                        }
                        // SAFETY: `converted <= length` upheld, since `converted < length` before this
                        converted += 1;
                        // Next, handle ASCII punctuation and non-ASCII without
                        // going back to ASCII acceleration. Non-ASCII scripts
                        // use ASCII punctuation, so this avoids going to
                        // acceleration just for punctuation/space and then
                        // failing. This is a significant boost to non-ASCII
                        // scripts.
                        if converted == length {
                            return (pending, length, length);
                        }
                        // SAFETY: We are back to `converted < length` because of the == above
                        // and can perform this check.
                        let mut b = unsafe { *(src.get_unchecked(converted)) };
                        // SAFETY: `converted < length` is upheld for this loop
                        'innermost: loop {
                            if b > 127 {
                                non_ascii = b;
                                continue 'middle;
                            }
                            unsafe {
                                // SAFETY: `converted < length` is true for this loop
                                *(dst.get_unchecked_mut(converted)) = u16::from(b);
                            }
                            converted += 1;
                            if b < 60 {
                                // We've got punctuation
                                if converted == length {
                                    return (pending, length, length);
                                }
                                // SAFETY: we're back to `converted <= length` because of the == above
                                b = unsafe { *(src.get_unchecked(converted)) };
                                // SAFETY: The loop continues as `converted < length`
                                continue 'innermost;
                            }
                            // We've got markup or ASCII text
                            continue 'outermost;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SingleByteEncoder {
    table: &'static [u16; 128],
    run_bmp_offset: usize,
    run_byte_offset: usize,
    run_length: usize,
}

impl SingleByteEncoder {
    pub fn new(
        data: &'static [u16; 128],
        run_bmp_offset: u16,
        run_byte_offset: u8,
        run_length: u8,
    ) -> SingleByteEncoder {
        SingleByteEncoder {
            table: data,
            run_bmp_offset: run_bmp_offset as usize,
            run_byte_offset: run_byte_offset as usize,
            run_length: run_length as usize,
        }
    }

    pub fn max_buffer_length_from_utf16_without_replacement(&self, u16_length: usize) -> Option<usize> {
        Some(u16_length)
    }

    pub fn max_buffer_length_from_utf8_without_replacement(&self, byte_length: usize) -> Option<usize> {
        Some(byte_length)
    }

    #[inline(always)]
    fn encode_u16(&self, code_unit: u16) -> Option<u8> {
        // Run of consecutive units
        let unit_as_usize = code_unit as usize;
        let offset = unit_as_usize.wrapping_sub(self.run_bmp_offset);
        if offset < self.run_length {
            return Some((128 + self.run_byte_offset + offset) as u8);
        }
        // Search after the run
        let tail_start = self.run_byte_offset + self.run_length;
        if let Some(pos) = position(&self.table[tail_start..], code_unit) {
            return Some((128 + tail_start + pos) as u8);
        }

        if self.run_byte_offset >= 64 {
            // Search third quadrant before the run
            if let Some(pos) = position(&self.table[64..self.run_byte_offset], code_unit) {
                return Some(((128 + 64) + pos) as u8);
            }
            // Search second quadrant
            if let Some(pos) = position(&self.table[32..64], code_unit) {
                return Some(((128 + 32) + pos) as u8);
            }
        } else if let Some(pos) = position(&self.table[32..self.run_byte_offset.max(32)], code_unit) {
            // windows-1252, windows-874, ISO-8859-15 and ISO-8859-5
            // Search second quadrant before the run. When the run starts below the
            // second quadrant (the ISO-8859-x indexes run from pointer 0, which maps
            // U+0080) there is nothing before it in this quadrant, and the first
            // quadrant search below covers the whole region.
            return Some(((128 + 32) + pos) as u8);
        }
        // Search first quadrant
        if let Some(pos) = position(&self.table[..32], code_unit) {
            return Some((128 + pos) as u8);
        }

        None
    }

    pub fn encode_from_utf16_raw(
        &mut self,
        src: &[u16],
        dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let (pending, length) = if dst.len() < src.len() {
            (EncoderResult::OutputFull, dst.len())
        } else {
            (EncoderResult::InputEmpty, src.len())
        };
        // Safety invariant: converted <= length. Quite often we have `converted < length`
        // which will be separately marked.
        let mut converted = 0usize;
        'outermost: loop {
            // SAFETY: length is the minimum length, `src/dst + x` will always be valid for reads/writes of `len - x`
            match basic_latin_to_ascii(&src[converted..], &mut dst[converted..]) {
                None => {
                    return (pending, length, length);
                }
                Some((mut non_ascii, consumed)) => {
                    // Safety invariant: `converted <= length` upheld, since this can only consume
                    // up to `length - converted` bytes.
                    //
                    // Furthermore, in this context,
                    // we can assume `converted < length` since this branch is only ever hit when
                    // ascii_to_basic_latin fails to consume the entire slice
                    converted += consumed;
                    'middle: loop {
                        // `converted` doesn't count the reading of `non_ascii` yet.
                        match self.encode_u16(non_ascii) {
                            Some(byte) => {
                                unsafe {
                                    // SAFETY: we're allowed this access since `converted < length`
                                    *(dst.get_unchecked_mut(converted)) = byte;
                                }
                                converted += 1;
                                // `converted <= length` now
                            }
                            None => {
                                // At this point, we need to know if we
                                // have a surrogate.
                                let high_bits = non_ascii & 0xFC00u16;
                                if high_bits == 0xD800u16 {
                                    // high surrogate
                                    if converted + 1 == length {
                                        // End of buffer. This surrogate is unpaired.
                                        return (
                                            EncoderResult::Unmappable('\u{FFFD}'),
                                            converted + 1, // +1 `for non_ascii`
                                            converted,
                                        );
                                    }
                                    // SAFETY: convered < length from outside the match, and `converted + 1 != length`,
                                    // So `converted + 1 < length` as well. We're in bounds
                                    let second = u32::from(unsafe { *src.get_unchecked(converted + 1) });
                                    if second & 0xFC00u32 != 0xDC00u32 {
                                        return (
                                            EncoderResult::Unmappable('\u{FFFD}'),
                                            converted + 1, // +1 `for non_ascii`
                                            converted,
                                        );
                                    }
                                    // The next code unit is a low surrogate.
                                    let astral: char = unsafe {
                                        // SAFETY: We can rely on non_ascii being 0xD800-0xDBFF since the high bits are 0xD800
                                        // Then, (non_ascii << 10 - 0xD800 << 10) becomes between (0 to 0x3FF) << 10, which is between
                                        // 0x400 to 0xffc00. Adding the 0x10000 gives a range of 0x10400 to 0x10fc00. Subtracting the 0xDC00
                                        // gives 0x2800 to 0x102000
                                        // The second term is between 0xDC00 and 0xDFFF from the check above. This gives a maximum
                                        // possible range of (0x10400 + 0xDC00) to (0x102000 + 0xDFFF) which is 0x1E000 to 0x10ffff.
                                        // This is in range.
                                        //
                                        // From a Unicode principles perspective this can also be verified as we have checked that `non_ascii` is a high surrogate
                                        // (0xD800..=0xDBFF), and that `second` is a low surrogate (`0xDC00..=0xDFFF`), and we are applying reverse of the UTC16 transformation
                                        // algorithm <https://en.wikipedia.org/wiki/UTF-16#Code_points_from_U+010000_to_U+10FFFF>, by applying the high surrogate - 0xD800 to the
                                        // high ten bits, and the low surrogate - 0xDc00 to the low ten bits, and then adding 0x10000
                                        char::from_u32_unchecked(
                                            (u32::from(non_ascii) << 10) + second
                                                - (((0xD800u32 << 10) - 0x1_0000u32) + 0xDC00u32),
                                        )
                                    };
                                    return (
                                        EncoderResult::Unmappable(astral),
                                        converted + 2, // +2 `for non_ascii` and `second`
                                        converted,
                                    );
                                }
                                if high_bits == 0xDC00u16 {
                                    // Unpaired low surrogate
                                    return (
                                        EncoderResult::Unmappable('\u{FFFD}'),
                                        converted + 1, // +1 `for non_ascii`
                                        converted,
                                    );
                                }
                                return (
                                    EncoderResult::unmappable_from_bmp(non_ascii),
                                    converted + 1, // +1 `for non_ascii`
                                    converted,
                                );
                                // SAFETY: This branch diverges, so no need to uphold invariants on `converted`
                            }
                        }
                        // Next, handle ASCII punctuation and non-ASCII without
                        // going back to ASCII acceleration. Non-ASCII scripts
                        // use ASCII punctuation, so this avoid going to
                        // acceleration just for punctuation/space and then
                        // failing. This is a significant boost to non-ASCII
                        // scripts.
                        if converted == length {
                            return (pending, length, length);
                        }
                        // SAFETY: we're back to `converted < length` due to the == above and can perform
                        // the unchecked read
                        let mut unit = unsafe { *(src.get_unchecked(converted)) };
                        'innermost: loop {
                            // SAFETY: This loop always begins with `converted < length`, see
                            // the invariant outside and the comment on the continue below
                            if unit > 127 {
                                non_ascii = unit;
                                continue 'middle;
                            }
                            unsafe {
                                // SAFETY: Can rely on converted < length
                                *(dst.get_unchecked_mut(converted)) = unit as u8;
                            }
                            converted += 1;
                            // `converted <= length` here
                            if unit < 60 {
                                if converted == length {
                                    return (pending, length, length);
                                }
                                // SAFETY: `converted < length` due to the == above. The read is safe.
                                unit = unsafe { *(src.get_unchecked(converted)) };
                                // SAFETY: This only happens if `converted < length`, maintaining it
                                continue 'innermost;
                            }
                            // We've got markup or ASCII text
                            continue 'outermost;
                            // SAFETY: All other routes to here diverge so the continue is the only
                            // way to run the innermost loop.
                        }
                    }
                }
            }
        }
    }

    pub fn encode_from_utf8_raw(
        &mut self,
        src: &str,
        dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let mut read = 0;
        let mut written = 0;
        for (i, ch) in src.char_indices() {
            if written >= dst.len() {
                return (EncoderResult::OutputFull, i, written);
            }
            if ch.is_ascii() {
                dst[written] = ch as u8;
                written += 1;
                read = i + 1;
                continue;
            }
            let cp = ch as u32;
            // The unmappable character is reported as consumed, like the utf-16
            // path above, so `Encoder::encode_from_utf8` can write its numeric
            // character reference and move on instead of re-reading it forever.
            let consumed = i + ch.len_utf8();
            if cp > 0xFFFF {
                return (EncoderResult::Unmappable(ch), consumed, written);
            }
            match self.encode_u16(cp as u16) {
                Some(byte) => {
                    dst[written] = byte;
                    written += 1;
                    read = consumed;
                }
                None => return (EncoderResult::Unmappable(ch), consumed, written),
            }
        }
        (EncoderResult::InputEmpty, read, written)
    }
}

#[inline(always)]
fn position(slice: &[u16], needle: u16) -> Option<usize> {
    slice.iter().position(|&x| x == needle)
}

fn write_ncr(unmappable: char, dst: &mut [u8]) -> usize {
    // len is the number of decimal digits needed to represent unmappable plus
    // 3 (the length of "&#" and ";").
    let mut number = unmappable as u32;
    let len = if number >= 1_000_000u32 {
        10usize
    } else if number >= 100_000u32 {
        9usize
    } else if number >= 10_000u32 {
        8usize
    } else if number >= 1_000u32 {
        7usize
    } else if number >= 100u32 {
        6usize
    } else {
        // Review the outcome of https://github.com/whatwg/encoding/issues/15
        // to see if this case is possible
        5usize
    };
    debug_assert!(number >= 10u32);
    debug_assert!(len <= dst.len());
    let mut pos = len - 1;
    dst[pos] = b';';
    pos -= 1;
    loop {
        let rightmost = number % 10;
        dst[pos] = rightmost as u8 + b'0';
        pos -= 1;
        if number < 10 {
            break;
        }
        number /= 10;
    }
    dst[1] = b'#';
    dst[0] = b'&';
    len
}

#[inline(always)]
fn in_range16(i: u16, start: u16, end: u16) -> bool {
    i.wrapping_sub(start) < (end - start)
}

#[inline(always)]
fn in_range32(i: u32, start: u32, end: u32) -> bool {
    i.wrapping_sub(start) < (end - start)
}

#[inline(always)]
fn in_inclusive_range8(i: u8, start: u8, end: u8) -> bool {
    i.wrapping_sub(start) <= (end - start)
}

#[inline(always)]
fn in_inclusive_range16(i: u16, start: u16, end: u16) -> bool {
    i.wrapping_sub(start) <= (end - start)
}

#[inline(always)]
fn in_inclusive_range32(i: u32, start: u32, end: u32) -> bool {
    i.wrapping_sub(start) <= (end - start)
}

#[inline(always)]
fn in_inclusive_range(i: usize, start: usize, end: usize) -> bool {
    i.wrapping_sub(start) <= (end - start)
}

#[inline(always)]
fn checked_add(num: usize, opt: Option<usize>) -> Option<usize> {
    if let Some(n) = opt {
        n.checked_add(num)
    } else {
        None
    }
}

#[inline(always)]
fn checked_add_opt(one: Option<usize>, other: Option<usize>) -> Option<usize> {
    if let Some(n) = one {
        checked_add(n, other)
    } else {
        None
    }
}

#[inline(always)]
fn checked_mul(num: usize, opt: Option<usize>) -> Option<usize> {
    if let Some(n) = opt {
        n.checked_mul(num)
    } else {
        None
    }
}

#[inline(always)]
fn checked_div(opt: Option<usize>, num: usize) -> Option<usize> {
    if let Some(n) = opt {
        n.checked_div(num)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{CoderResult, Encoding};
    use crate::legacy;
    use crate::single_byte_tables;

    /// A table shaped like the ISO-8859-2 index: pointers 0..32 are the C1
    /// controls, pointer 32 is U+00A0 and the run therefore starts at byte
    /// offset 0, below the second quadrant.
    static C1_RUN_TABLE: [u16; 128] = {
        let mut table = [0u16; 128];
        let mut i = 0usize;
        while i < 33 {
            table[i] = 0x0080u16 + i as u16;
            i += 1;
        }
        table[33] = 0x0104;
        table[127] = 0x17C6;
        table
    };

    #[test]
    fn unmappable_below_second_quadrant_run_does_not_panic() {
        let mut encoder = SingleByteEncoder::new(&C1_RUN_TABLE, 0x0080, 0x00, 33);
        let mut dst = [0u8; 16];
        let (result, read, written) = encoder.encode_from_utf16_raw(&[0x20AC], &mut dst, true);
        assert_eq!(result, EncoderResult::Unmappable('\u{20AC}'));
        assert_eq!((read, written), (1, 0));
    }

    #[test]
    fn c1_run_maps_arithmetically() {
        let mut encoder = SingleByteEncoder::new(&C1_RUN_TABLE, 0x0080, 0x00, 33);
        let mut dst = [0u8; 16];
        let (result, read, written) =
            encoder.encode_from_utf16_raw(&[0x0080, 0x009F, 0x00A0], &mut dst, true);
        assert_eq!(result, EncoderResult::InputEmpty);
        assert_eq!((read, written), (3, 3));
        assert_eq!(&dst[..3], &[0x80, 0x9F, 0xA0]);
    }

    #[test]
    fn maps_slots_after_and_before_the_run() {
        let mut encoder = SingleByteEncoder::new(&C1_RUN_TABLE, 0x0080, 0x00, 33);
        let mut dst = [0u8; 16];
        let (result, read, written) = encoder.encode_from_utf16_raw(&[0x0104, 0x17C6], &mut dst, true);
        assert_eq!(result, EncoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
        assert_eq!(&dst[..2], &[0xA1, 0xFF]);
    }

    #[test]
    fn maps_slots_inside_the_second_quadrant_before_a_later_run() {
        // windows-1252 shape: the run starts at byte offset 0x20, so U+00A0..U+00BF
        // sit in the first quadrant and U+20AC sits at pointer 0 in the same slot.
        static WINDOWS_1252_LIKE: [u16; 128] = {
            let mut table = [0u16; 128];
            table[0] = 0x20AC;
            let mut i = 0usize;
            while i < 96 {
                table[0x20 + i] = 0x00A0 + i as u16;
                i += 1;
            }
            table
        };
        let mut encoder = SingleByteEncoder::new(&WINDOWS_1252_LIKE, 0x00A0, 0x20, 96);
        let mut dst = [0u8; 16];
        let (result, read, written) = encoder.encode_from_utf16_raw(&[0x20AC, 0x00A1], &mut dst, true);
        assert_eq!(result, EncoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
        assert_eq!(&dst[..2], &[0x80, 0xA1]);
    }

    #[test]
    fn undefined_pointer_decodes_as_malformed() {
        let mut decoder = SingleByteDecoder { table: &C1_RUN_TABLE };
        let mut dst = [0u8; 16];
        // Slot 40 (byte 0xA8) is undefined in the table above.
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0xA8], &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
    }

    // ---------------------------------------------------------------------
    // Fixtures for the tests below.
    //
    // The tables are written out here instead of importing
    // `crate::single_byte_tables` so that each case pins one behaviour and the
    // module stays self-contained. The `*_INDEX` tables are literal copies of
    // the normative WHATWG indexes and are used where realistic data matters
    // (round trips and markup decoding).
    // ---------------------------------------------------------------------

    /// Decoder fixture with pointers in every UTF-8 width plus holes:
    /// `0x80` is the C1 control U+0080, `0x81` U+00E9 (two UTF-8 bytes),
    /// `0x82` U+2020 and `0x83` U+FFFF (three bytes), `0x84` U+00A0 and
    /// `0x88` U+00B7. Every other pointer is undefined.
    static WIDTHS_TABLE: [u16; 128] = {
        let mut table = [0u16; 128];
        table[0] = 0x0080;
        table[1] = 0x00E9;
        table[2] = 0x2020;
        table[3] = 0xFFFF;
        table[4] = 0x00A0;
        table[8] = 0x00B7;
        table
    };

    /// Encoder fixture with one defined pointer per search region: first
    /// quadrant (`0x80`, `0x88`), second quadrant (`0xB0`, `0xB8`), third
    /// quadrant before the run (`0xC0`, `0xC6`), an eight-unit run starting at
    /// pointer `0xD0` and a tail after the run (`0xD8`, `0xFF`).
    static QUADRANT_TABLE: [u16; 128] = {
        let mut table = [0u16; 128];
        table[0] = 0x00A0;
        table[8] = 0x00B0;
        table[32] = 0x00C0;
        table[40] = 0x00D0;
        table[64] = 0x00E0;
        table[70] = 0x00F0;
        table[100] = 0x2000;
        table[127] = 0x3000;
        let mut i = 0usize;
        while i < 8 {
            table[80 + i] = 0x1000 + i as u16;
            i += 1;
        }
        table
    };

    /// Same shape as [`QUADRANT_TABLE`] but with code points deliberately
    /// present in more than one region, so that the search *order*
    /// (tail, third quadrant, second quadrant, first quadrant) is observable.
    static DUPLICATE_TABLE: [u16; 128] = {
        let mut table = [0u16; 128];
        table[80] = 0x1000;
        table[81] = 0x1001;
        table[5] = 0x00FF;
        table[40] = 0x00FF;
        table[70] = 0x00FF;
        table[6] = 0x01FF;
        table[100] = 0x01FF;
        table
    };

    /// Like [`DUPLICATE_TABLE`], but with the run starting below the third
    /// quadrant (pointer `0x30`), which is the `windows-1252` / `ISO-8859-15`
    /// shape: the region searched before the run is the second quadrant.
    static LOW_RUN_DUPLICATE_TABLE: [u16; 128] = {
        let mut table = [0u16; 128];
        table[48] = 0x1000;
        table[49] = 0x1001;
        table[5] = 0x00FF;
        table[40] = 0x00FF;
        table[6] = 0x01FF;
        table[90] = 0x01FF;
        table
    };

    /// The windows-1252 index (`index-windows-1252.txt`): 128 of 128 pointers
    /// defined, encoder run of 96 code points from U+00A0 at pointer 0x20.
    static WINDOWS_1252_INDEX: [u16; 128] = [
        0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039,
        0x0152, 0x008D, 0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
        0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178, 0x00A0, 0x00A1, 0x00A2, 0x00A3,
        0x00A4, 0x00A5, 0x00A6, 0x00A7, 0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
        0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7, 0x00B8, 0x00B9, 0x00BA, 0x00BB,
        0x00BC, 0x00BD, 0x00BE, 0x00BF, 0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7,
        0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3,
        0x00D4, 0x00D5, 0x00D6, 0x00D7, 0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
        0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB,
        0x00EC, 0x00ED, 0x00EE, 0x00EF, 0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7,
        0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF,
    ];

    /// The ISO-8859-15 index (`index-iso-8859-15.txt`): 128 of 128 pointers
    /// defined, encoder run of 65 code points from U+00BF at pointer 0x3F.
    /// The euro sign sits at pointer 0xA4, in the second quadrant *before* the
    /// run, which is what makes this index interesting for the encoder search.
    static ISO_8859_15_INDEX: [u16; 128] = [
        0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089, 0x008A, 0x008B,
        0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097,
        0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D, 0x009E, 0x009F, 0x00A0, 0x00A1, 0x00A2, 0x00A3,
        0x20AC, 0x00A5, 0x0160, 0x00A7, 0x0161, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
        0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x017D, 0x00B5, 0x00B6, 0x00B7, 0x017E, 0x00B9, 0x00BA, 0x00BB,
        0x0152, 0x0153, 0x0178, 0x00BF, 0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7,
        0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3,
        0x00D4, 0x00D5, 0x00D6, 0x00D7, 0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
        0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB,
        0x00EC, 0x00ED, 0x00EE, 0x00EF, 0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7,
        0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF,
    ];

    /// The KOI8-R index (`index-koi8-r.txt`): 128 of 128 pointers defined,
    /// encoder run of 15 code points from U+2553 at pointer 0x24. KOI8-R is
    /// neither quadrant-aligned nor Latin, so it exercises the tail search.
    static KOI8_R_INDEX: [u16; 128] = [
        0x2500, 0x2502, 0x250C, 0x2510, 0x2514, 0x2518, 0x251C, 0x2524, 0x252C, 0x2534, 0x253C, 0x2580,
        0x2584, 0x2588, 0x258C, 0x2590, 0x2591, 0x2592, 0x2593, 0x2320, 0x25A0, 0x2219, 0x221A, 0x2248,
        0x2264, 0x2265, 0x00A0, 0x2321, 0x00B0, 0x00B2, 0x00B7, 0x00F7, 0x2550, 0x2551, 0x2552, 0x0451,
        0x2553, 0x2554, 0x2555, 0x2556, 0x2557, 0x2558, 0x2559, 0x255A, 0x255B, 0x255C, 0x255D, 0x255E,
        0x255F, 0x2560, 0x2561, 0x0401, 0x2562, 0x2563, 0x2564, 0x2565, 0x2566, 0x2567, 0x2568, 0x2569,
        0x256A, 0x256B, 0x256C, 0x00A9, 0x044E, 0x0430, 0x0431, 0x0446, 0x0434, 0x0435, 0x0444, 0x0433,
        0x0445, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F, 0x044F, 0x0440, 0x0441,
        0x0442, 0x0443, 0x0436, 0x0432, 0x044C, 0x044B, 0x0437, 0x0448, 0x044D, 0x0449, 0x0447, 0x044A,
        0x042E, 0x0410, 0x0411, 0x0426, 0x0414, 0x0415, 0x0424, 0x0413, 0x0425, 0x0418, 0x0419, 0x041A,
        0x041B, 0x041C, 0x041D, 0x041E, 0x041F, 0x042F, 0x0420, 0x0421, 0x0422, 0x0423, 0x0416, 0x0412,
        0x042C, 0x042B, 0x0417, 0x0428, 0x042D, 0x0429, 0x0427, 0x042A,
    ];

    /// The windows-1251 index (`index-windows-1251.txt`): 128 of 128 pointers
    /// defined, encoder run of 64 code points from U+0410 at pointer 0x40. The
    /// run starts exactly at the third quadrant, so `run_byte_offset >= 64`.
    static WINDOWS_1251_INDEX: [u16; 128] = [
        0x0402, 0x0403, 0x201A, 0x0453, 0x201E, 0x2026, 0x2020, 0x2021, 0x20AC, 0x2030, 0x0409, 0x2039,
        0x040A, 0x040C, 0x040B, 0x040F, 0x0452, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
        0x0098, 0x2122, 0x0459, 0x203A, 0x045A, 0x045C, 0x045B, 0x045F, 0x00A0, 0x040E, 0x045E, 0x0408,
        0x00A4, 0x0490, 0x00A6, 0x00A7, 0x0401, 0x00A9, 0x0404, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x0407,
        0x00B0, 0x00B1, 0x0406, 0x0456, 0x0491, 0x00B5, 0x00B6, 0x00B7, 0x0451, 0x2116, 0x0454, 0x00BB,
        0x0458, 0x0405, 0x0455, 0x0457, 0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417,
        0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F, 0x0420, 0x0421, 0x0422, 0x0423,
        0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
        0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B,
        0x043C, 0x043D, 0x043E, 0x043F, 0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447,
        0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
    ];

    /// The ISO-8859-2 index (`index-iso-8859-2.txt`): 128 of 128 pointers
    /// defined, encoder run of 33 code points from U+0080 at pointer 0x00 —
    /// a run that starts before the second quadrant.
    static ISO_8859_2_INDEX: [u16; 128] = [
        0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089, 0x008A, 0x008B,
        0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097,
        0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D, 0x009E, 0x009F, 0x00A0, 0x0104, 0x02D8, 0x0141,
        0x00A4, 0x013D, 0x015A, 0x00A7, 0x00A8, 0x0160, 0x015E, 0x0164, 0x0179, 0x00AD, 0x017D, 0x017B,
        0x00B0, 0x0105, 0x02DB, 0x0142, 0x00B4, 0x013E, 0x015B, 0x02C7, 0x00B8, 0x0161, 0x015F, 0x0165,
        0x017A, 0x02DD, 0x017E, 0x017C, 0x0154, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0139, 0x0106, 0x00C7,
        0x010C, 0x00C9, 0x0118, 0x00CB, 0x011A, 0x00CD, 0x00CE, 0x010E, 0x0110, 0x0143, 0x0147, 0x00D3,
        0x00D4, 0x0150, 0x00D6, 0x00D7, 0x0158, 0x016E, 0x00DA, 0x0170, 0x00DC, 0x00DD, 0x0162, 0x00DF,
        0x0155, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x013A, 0x0107, 0x00E7, 0x010D, 0x00E9, 0x0119, 0x00EB,
        0x011B, 0x00ED, 0x00EE, 0x010F, 0x0111, 0x0144, 0x0148, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x00F7,
        0x0159, 0x016F, 0x00FA, 0x0171, 0x00FC, 0x00FD, 0x0163, 0x02D9,
    ];

    /// The ISO-8859-7 index (`index-iso-8859-7.txt`): 125 of 128 pointers
    /// defined (0xA0, 0xD4 and 0xFF are undefined), encoder run of 44 code
    /// points from U+03A3 at pointer 0x53.
    static ISO_8859_7_INDEX: [u16; 128] = [
        0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089, 0x008A, 0x008B,
        0x008C, 0x008D, 0x008E, 0x008F, 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097,
        0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D, 0x009E, 0x009F, 0x00A0, 0x2018, 0x2019, 0x00A3,
        0x20AC, 0x20AF, 0x00A6, 0x00A7, 0x00A8, 0x00A9, 0x037A, 0x00AB, 0x00AC, 0x00AD, 0x0000, 0x2015,
        0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x0384, 0x0385, 0x0386, 0x00B7, 0x0388, 0x0389, 0x038A, 0x00BB,
        0x038C, 0x00BD, 0x038E, 0x038F, 0x0390, 0x0391, 0x0392, 0x0393, 0x0394, 0x0395, 0x0396, 0x0397,
        0x0398, 0x0399, 0x039A, 0x039B, 0x039C, 0x039D, 0x039E, 0x039F, 0x03A0, 0x03A1, 0x0000, 0x03A3,
        0x03A4, 0x03A5, 0x03A6, 0x03A7, 0x03A8, 0x03A9, 0x03AA, 0x03AB, 0x03AC, 0x03AD, 0x03AE, 0x03AF,
        0x03B0, 0x03B1, 0x03B2, 0x03B3, 0x03B4, 0x03B5, 0x03B6, 0x03B7, 0x03B8, 0x03B9, 0x03BA, 0x03BB,
        0x03BC, 0x03BD, 0x03BE, 0x03BF, 0x03C0, 0x03C1, 0x03C2, 0x03C3, 0x03C4, 0x03C5, 0x03C6, 0x03C7,
        0x03C8, 0x03C9, 0x03CA, 0x03CB, 0x03CC, 0x03CD, 0x03CE, 0x0000,
    ];

    /// A table in which no pointer is defined: every non-ASCII byte is malformed.
    static UNDEFINED_TABLE: [u16; 128] = [0u16; 128];

    /// Encodes one code unit with `table` and the given run, into a buffer that
    /// is large enough for any single-byte result.
    fn encode_one(
        table: &'static [u16; 128],
        run_bmp_offset: u16,
        run_byte_offset: u8,
        run_length: u8,
        code_unit: u16,
    ) -> (EncoderResult, usize, usize, u8) {
        let mut encoder = SingleByteEncoder::new(table, run_bmp_offset, run_byte_offset, run_length);
        let mut dst = [0u8; 8];
        let (result, read, written) = encoder.encode_from_utf16_raw(&[code_unit], &mut dst, true);
        (result, read, written, dst[0])
    }

    #[test]
    fn ascii_only_input_decodes_byte_for_byte_to_utf8() {
        // Pure ASCII never reaches the innermost punctuation loop: the
        // accelerator in `copy_ascii_from_check_space_bmp` copies it all.
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let src = b"plain ascii text";
        let mut dst = [0u8; 32];
        let (result, read, written) = decoder.decode_to_utf8_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (src.len(), src.len()));
        assert_eq!(&dst[..written], src);
    }

    #[test]
    fn empty_utf8_input_with_room_left_is_input_empty() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn empty_utf8_input_with_empty_destination_is_input_empty() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst: [u8; 0] = [];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn ascii_input_into_empty_destination_reports_output_full() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst: [u8; 0] = [];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"abc", &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn ascii_input_one_byte_short_of_destination_reports_output_full() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 3];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"abcd", &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (3, 3));
        assert_eq!(&dst[..], b"abc");
    }

    #[test]
    fn single_non_ascii_byte_decodes_to_two_utf8_bytes() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        // 0x81 -> U+00E9, which is two bytes in UTF-8 (C3 A9).
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 2));
        assert_eq!(&dst[..2], &[0xC3, 0xA9]);
    }

    #[test]
    fn c1_pointer_decodes_to_two_utf8_bytes() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        // 0x80 -> U+0080, the C1 range, which is *not* ASCII and takes two bytes.
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x80], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 2));
        assert_eq!(&dst[..2], &[0xC2, 0x80]);
    }

    #[test]
    fn undefined_c1_pointer_reports_malformed() {
        // A table whose C1 slot is undefined must report the byte as malformed
        // rather than decoding it as the control character.
        let mut decoder = SingleByteDecoder {
            table: &UNDEFINED_TABLE,
        };
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x80], &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
        assert_eq!(dst[0], 0, "nothing may be written for a bad byte");
    }

    #[test]
    fn three_byte_pointer_decodes_when_three_bytes_are_reserved() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        // 0x82 -> U+2020 (E2 80 A0). The reservation check only asks for two
        // free bytes, but the strict `pos + 2 < len` comparison always leaves
        // room for the third.
        let mut dst = [0u8; 3];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x82], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 3));
        assert_eq!(&dst[..], &[0xE2, 0x80, 0xA0]);
    }

    #[test]
    fn three_byte_pointer_is_not_written_into_two_byte_destination() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 2];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x82], &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0), "the bad byte stays unread");
        assert_eq!(&dst[..], &[0, 0]);
    }

    #[test]
    fn non_ascii_byte_is_unread_when_only_one_destination_byte_is_free() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 1];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn ascii_before_non_ascii_decodes_to_utf8() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"a\x81", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 3));
        assert_eq!(&dst[..3], b"a\xC3\xA9");
    }

    #[test]
    fn punctuation_below_angle_bracket_stays_on_the_fast_path_to_utf8() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        // '!' is 0x21 < 60, so it is written by the innermost punctuation loop
        // instead of re-entering the ASCII accelerator.
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81, b'!'], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 3));
        assert_eq!(&dst[..3], &[0xC3, 0xA9, b'!']);
    }

    #[test]
    fn markup_at_angle_bracket_restarts_the_accelerator_to_utf8() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        // '<' is exactly 60, which is markup: the loop breaks out to the
        // accelerator for the rest of the run.
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81, b'<', b'a', b'>'], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 5));
        assert_eq!(&dst[..5], &[0xC3, 0xA9, b'<', b'a', b'>']);
    }

    #[test]
    fn uppercase_letter_counts_as_markup_in_utf8_input() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81, b'A', b'!'], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (3, 4));
        assert_eq!(&dst[..4], &[0xC3, 0xA9, b'A', b'!']);
    }

    #[test]
    fn alternating_non_ascii_and_punctuation_decodes_to_utf8() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81, b'!', 0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (3, 5));
        assert_eq!(&dst[..5], &[0xC3, 0xA9, b'!', 0xC3, 0xA9]);
    }

    #[test]
    fn trailing_non_ascii_ends_the_input_after_a_punctuation_run_to_utf8() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        // Two C1/upper-BMP pointers around one punctuation byte.
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x80, b'.', 0x83], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (3, 6));
        assert_eq!(&dst[..6], &[0xC2, 0x80, b'.', 0xEF, 0xBF, 0xBF]);
    }

    #[test]
    fn undefined_pointer_after_ascii_is_malformed_but_counts_ascii_to_utf8() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u8; 16];
        // 0x90 has no pointer in the fixture: read includes the bad byte,
        // written stops before it.
        let (result, read, written) = decoder.decode_to_utf8_raw(b"a\x90", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (2, 1));
        assert_eq!(&dst[..1], b"a");
    }

    #[test]
    fn only_the_first_undefined_pointer_is_reported_to_utf8() {
        let mut decoder = SingleByteDecoder {
            table: &UNDEFINED_TABLE,
        };
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x90, 0x91, 0x92], &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0), "one bad byte per call");
    }

    #[test]
    fn every_non_ascii_byte_is_malformed_for_a_fully_undefined_table() {
        for byte in 0x80u16..=0xFF {
            let mut decoder = SingleByteDecoder {
                table: &UNDEFINED_TABLE,
            };
            let mut dst = [0u8; 8];
            let (result, read, written) = decoder.decode_to_utf8_raw(&[byte as u8], &mut dst, true);
            assert_eq!(
                (result, read, written),
                (DecoderResult::Malformed(1, 0), 1, 0),
                "byte {byte:#04x}"
            );
        }
    }

    #[test]
    fn utf8_output_full_position_points_before_the_failed_write() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        // Two bytes fit, the second pointer's two-byte sequence does not.
        let mut dst = [0u8; 3];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81, 0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (1, 2));
        assert_eq!(&dst[..2], &[0xC3, 0xA9]);
    }

    #[test]
    fn two_free_utf8_bytes_are_still_output_full_before_a_punctuation_write() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        // After U+00E9 and '!' the destination has two bytes left, yet
        // `check_space_bmp` demands three so that the loop can keep writing
        // without re-checking: the remaining input byte is reported unread.
        let mut dst = [0u8; 5];
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x81, b'!', b'x'], &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (2, 3));
        assert_eq!(&dst[..3], &[0xC3, 0xA9, b'!']);
    }

    #[test]
    fn last_flag_does_not_change_the_utf8_decoding_result() {
        // Single-byte decoding is stateless: no partial sequence can be open.
        let src = [0x81u8, b'!', 0x83, b'-', 0x90];
        let mut first = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut second = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst_a = [0u8; 32];
        let mut dst_b = [0u8; 32];
        let (result_a, read_a, written_a) = first.decode_to_utf8_raw(&src, &mut dst_a, true);
        let (result_b, read_b, written_b) = second.decode_to_utf8_raw(&src, &mut dst_b, false);
        assert_eq!((result_a, read_a, written_a), (result_b, read_b, written_b));
        assert_eq!(&dst_a[..written_a], &dst_b[..written_b]);
    }

    #[test]
    fn windows_1252_markup_decodes_to_utf8() {
        let mut decoder = SingleByteDecoder {
            table: &WINDOWS_1252_INDEX,
        };
        // "U+201C U+00EB U+201D" quoted in ASCII markup.
        let src = b"<p>\x93\xEB\x94</p>";
        let mut dst = [0u8; 32];
        let (result, read, written) = decoder.decode_to_utf8_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (src.len(), 15));
        assert_eq!(&dst[..written], b"<p>\xE2\x80\x9C\xC3\xAB\xE2\x80\x9D</p>");
    }

    #[test]
    fn decoder_new_returns_a_single_byte_variant_decoder() {
        let mut variant = SingleByteDecoder::new(&WIDTHS_TABLE);
        assert!(matches!(variant, VariantDecoder::SingleByte(_)));
        assert_eq!(variant.max_utf16_buffer_length(7), Some(7));

        let mut dst = [0u16; 8];
        let (result, read, written) = variant.decode_to_utf16_raw(&[0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[0], 0x00E9);
    }

    #[test]
    fn ascii_only_input_decodes_byte_for_byte_to_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let src = b"abc!<>~ text";
        let mut dst = [0u16; 32];
        let (result, read, written) = decoder.decode_to_utf16_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (src.len(), src.len()));
        let expected: [u16; 12] = core::array::from_fn(|i| u16::from(src[i]));
        assert_eq!(&dst[..written], &expected[..]);
    }

    #[test]
    fn empty_utf16_input_is_input_empty() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn empty_utf16_destination_with_ascii_input_reports_output_full() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst: [u16; 0] = [];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"abc", &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn non_ascii_first_byte_decodes_to_one_utf16_unit() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[0], 0x00E9);
    }

    #[test]
    fn c1_pointer_decodes_to_the_control_unit_in_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x80], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[0], 0x0080);
    }

    #[test]
    fn undefined_c1_pointer_reports_malformed_in_utf16() {
        let mut decoder = SingleByteDecoder {
            table: &UNDEFINED_TABLE,
        };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x80], &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
        assert_eq!(dst[0], 0);
    }

    #[test]
    fn punctuation_below_angle_bracket_stays_on_the_fast_path_to_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x81, b'!'], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
        assert_eq!(&dst[..2], &[0x00E9, u16::from(b'!')]);
    }

    #[test]
    fn markup_at_angle_bracket_restarts_the_accelerator_to_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x81, b'<', b'a', b'>'], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 4));
        assert_eq!(
            &dst[..4],
            &[0x00E9, u16::from(b'<'), u16::from(b'a'), u16::from(b'>')]
        );
    }

    #[test]
    fn alternating_non_ascii_and_punctuation_decodes_to_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x81, b'!', 0x81], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (3, 3));
        assert_eq!(&dst[..3], &[0x00E9, u16::from(b'!'), 0x00E9]);
    }

    #[test]
    fn non_ascii_last_input_byte_decodes_to_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"xy\x83", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (3, 3));
        assert_eq!(&dst[..3], &[u16::from(b'x'), u16::from(b'y'), 0xFFFF]);
    }

    #[test]
    fn undefined_pointer_after_ascii_is_malformed_in_utf16() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"a\x90", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (2, 1));
        assert_eq!(&dst[..1], &[u16::from(b'a')]);
    }

    #[test]
    fn only_the_first_undefined_pointer_is_reported_to_utf16() {
        let mut decoder = SingleByteDecoder {
            table: &UNDEFINED_TABLE,
        };
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x90, 0x91, 0x92], &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
    }

    #[test]
    fn every_non_ascii_byte_is_malformed_for_a_fully_undefined_table_in_utf16() {
        for byte in 0x80u16..=0xFF {
            let mut decoder = SingleByteDecoder {
                table: &UNDEFINED_TABLE,
            };
            let mut dst = [0u16; 4];
            let (result, read, written) = decoder.decode_to_utf16_raw(&[byte as u8], &mut dst, true);
            assert_eq!(
                (result, read, written),
                (DecoderResult::Malformed(1, 0), 1, 0),
                "byte {byte:#04x}"
            );
        }
    }

    #[test]
    fn utf16_destination_one_unit_short_of_ascii_input_reports_output_full() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 3];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"abcd", &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (3, 3));
        assert_eq!(&dst[..], &[u16::from(b'a'), u16::from(b'b'), u16::from(b'c')]);
    }

    #[test]
    fn utf16_non_ascii_beyond_capacity_is_left_unread() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        // One unit of room: the ASCII byte is written, the pointer byte is not read.
        let mut dst = [0u16; 1];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"a\x81", &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[0], u16::from(b'a'));
    }

    #[test]
    fn utf16_output_full_stops_before_the_unread_unit_after_punctuation() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        // The punctuation byte fills the destination, so `length` is reached in
        // the innermost loop and the trailing byte stays unread.
        let mut dst = [0u16; 2];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x81, b'!', b'x'], &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (2, 2));
        assert_eq!(&dst[..], &[0x00E9, u16::from(b'!')]);
    }

    #[test]
    fn utf16_destination_exactly_the_input_size_reports_input_empty() {
        let mut decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst = [0u16; 2];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x81, b'A'], &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
    }

    #[test]
    fn last_flag_does_not_change_the_utf16_decoding_result() {
        let src = [0x81u8, b'!', 0x83, b'-', 0x90];
        let mut first = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut second = SingleByteDecoder { table: &WIDTHS_TABLE };
        let mut dst_a = [0u16; 32];
        let mut dst_b = [0u16; 32];
        let (result_a, read_a, written_a) = first.decode_to_utf16_raw(&src, &mut dst_a, true);
        let (result_b, read_b, written_b) = second.decode_to_utf16_raw(&src, &mut dst_b, false);
        assert_eq!((result_a, read_a, written_a), (result_b, read_b, written_b));
        assert_eq!(&dst_a[..written_a], &dst_b[..written_b]);
    }

    #[test]
    fn koi8_r_cyrillic_word_decodes_to_utf16() {
        let mut decoder = SingleByteDecoder { table: &KOI8_R_INDEX };
        // "привет" in KOI8-R.
        let src = [0xD0u8, 0xD2, 0xC9, 0xD7, 0xC5, 0xD4];
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (6, 6));
        assert_eq!(&dst[..6], &[0x043F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442]);
    }

    #[test]
    fn full_pointer_range_decodes_to_utf16_pointer_for_pointer() {
        // The normative contract: entry `i` of the index is the code point for
        // byte `0x80 + i`.
        let mut decoder = SingleByteDecoder { table: &KOI8_R_INDEX };
        let src: [u8; 128] = core::array::from_fn(|i| 0x80 + i as u8);
        let mut dst = [0u16; 128];
        let (result, read, written) = decoder.decode_to_utf16_raw(&src[..], &mut dst[..], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (128, 128));
        assert_eq!(&dst[..], &KOI8_R_INDEX[..]);
    }

    #[test]
    fn max_utf16_buffer_length_is_one_unit_per_input_byte() {
        let decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        assert_eq!(decoder.max_utf16_buffer_length(0), Some(0));
        assert_eq!(decoder.max_utf16_buffer_length(1), Some(1));
        assert_eq!(decoder.max_utf16_buffer_length(128), Some(128));
    }

    #[test]
    fn max_utf8_buffer_length_triples_the_input_length() {
        let decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        assert_eq!(decoder.max_utf8_buffer_length(0), Some(0));
        assert_eq!(decoder.max_utf8_buffer_length(5), Some(15));
        assert_eq!(decoder.max_utf8_buffer_length_without_replacement(5), Some(15));
        // Replacement code points are three bytes, so both reservations agree.
        assert_eq!(
            decoder.max_utf8_buffer_length(1024),
            decoder.max_utf8_buffer_length_without_replacement(1024)
        );
    }

    #[test]
    fn max_utf8_buffer_length_returns_none_on_overflow() {
        let decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        assert_eq!(decoder.max_utf8_buffer_length(usize::MAX), None);
        assert_eq!(
            decoder.max_utf8_buffer_length_without_replacement(usize::MAX),
            None
        );
        // The smallest length whose triple wraps.
        let just_over = usize::MAX / 3 + 1;
        assert_eq!(decoder.max_utf8_buffer_length(just_over), None);
        let largest_ok = usize::MAX / 3;
        assert_eq!(
            decoder.max_utf8_buffer_length(largest_ok),
            largest_ok.checked_mul(3)
        );
    }

    #[test]
    fn max_utf16_buffer_length_does_not_saturate_on_overflow() {
        // The current implementation is a plain identity, so `usize::MAX` is
        // echoed back rather than rejected. Pinned because the UTF-8 helpers
        // above do return `None` for the same input.
        let decoder = SingleByteDecoder { table: &WIDTHS_TABLE };
        assert_eq!(decoder.max_utf16_buffer_length(usize::MAX), Some(usize::MAX));
    }

    #[test]
    fn encoder_max_buffer_lengths_are_one_byte_per_input_unit() {
        let encoder = SingleByteEncoder::new(&WIDTHS_TABLE, 0x00E9, 1, 1);
        assert_eq!(
            encoder.max_buffer_length_from_utf16_without_replacement(0),
            Some(0)
        );
        assert_eq!(
            encoder.max_buffer_length_from_utf16_without_replacement(7),
            Some(7)
        );
        assert_eq!(
            encoder.max_buffer_length_from_utf8_without_replacement(0),
            Some(0)
        );
        assert_eq!(
            encoder.max_buffer_length_from_utf8_without_replacement(7),
            Some(7)
        );
    }

    #[test]
    fn encoder_max_buffer_lengths_echo_usize_max_instead_of_none() {
        // A surrogate pair is two units for one character, and the astral
        // character never encodes, so one byte per unit is still an upper
        // bound; the helpers are identities and therefore overflow-silent.
        let encoder = SingleByteEncoder::new(&WIDTHS_TABLE, 0x00E9, 1, 1);
        assert_eq!(
            encoder.max_buffer_length_from_utf16_without_replacement(usize::MAX),
            Some(usize::MAX)
        );
        assert_eq!(
            encoder.max_buffer_length_from_utf8_without_replacement(usize::MAX),
            Some(usize::MAX)
        );
    }

    // Registry-driven cases: these run against the real WHATWG index tables
    // rather than fixtures shaped like them.

    fn decode_units(encoding: &'static Encoding, src: &[u8]) -> Vec<u16> {
        let mut dst = [0u16; 64];
        let (result, read, written) = encoding
            .new_decoder()
            .decode_to_utf16_without_replacement(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!(read, src.len());
        assert!(written <= dst.len());
        dst[..written].to_vec()
    }

    #[test]
    fn windows_1252_round_trips_the_euro_through_its_first_pointer() {
        assert_eq!(decode_units(&legacy::WINDOWS_1252, &[0x80]), vec![0x20AC]);
        let mut dst = [0u8; 16];
        let (result, read, written, _unmappables) = legacy::WINDOWS_1252
            .new_encoder()
            .encode_from_utf8("\u{20AC}", &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!((read, written), (3, 1));
        assert_eq!(dst[0], 0x80);
    }

    // The WHATWG index for ISO 8859-7 deviates from the published ISO standard and
    // fills the same 0xA4 slot with the euro sign that ISO 8859-15 uses.
    #[test]
    fn greek_index_shares_the_euro_pointer_with_iso_8859_15() {
        assert_eq!(decode_units(&legacy::ISO_8859_7, &[0xA4]), vec![0x20AC]);
        let mut dst = [0u8; 16];
        let (result, read, written, _unmappables) = legacy::ISO_8859_7
            .new_encoder()
            .encode_from_utf8("\u{20AC}", &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!((read, written), (3, 1));
        assert_eq!(dst[0], 0xA4);
    }

    #[test]
    fn iso_8859_15_encodes_the_euro_at_a_different_pointer() {
        assert_eq!(decode_units(&legacy::ISO_8859_15, &[0xA4]), vec![0x20AC]);
        let mut dst = [0u8; 16];
        let (result, _read, written, _unmappables) = legacy::ISO_8859_15
            .new_encoder()
            .encode_from_utf8("\u{20AC}", &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!(written, 1);
        assert_eq!(dst[0], 0xA4);
    }

    #[test]
    fn koi8_r_round_trips_a_cyrillic_word() {
        let src = [0xD2u8, 0xCF, 0xC4];
        let units = decode_units(&legacy::KOI8_R, &src);
        assert_eq!(units.len(), 3);
        let mut dst = [0u8; 16];
        let (result, read, written, _unmappables) = legacy::KOI8_R
            .new_encoder()
            .encode_from_utf16(&units, &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!((read, written), (3, 3));
        assert_eq!(&dst[..written], &src[..]);
    }

    #[test]
    fn windows_1252_fills_the_cp1252_holes_with_control_characters() {
        // Microsoft's CP1252 leaves 0x81, 0x8D, 0x8F, 0x90 and 0x9D undefined, but
        // the WHATWG index maps them to the matching C1 control character, so none
        // of these bytes is ever malformed.
        for byte in [0x81u8, 0x8D, 0x8F, 0x90, 0x9D] {
            assert_eq!(
                decode_units(&legacy::WINDOWS_1252, &[byte]),
                vec![u16::from(byte)],
                "byte {byte:02X}"
            );
        }
    }

    #[test]
    fn undefined_pointers_are_malformed_and_their_code_points_unmappable() {
        let cases: [(&'static Encoding, u8); 5] = [
            (&legacy::ISO_8859_3, 0xA5),
            (&legacy::ISO_8859_3, 0xC3),
            (&legacy::ISO_8859_3, 0xF0),
            (&legacy::WINDOWS_1257, 0xA1),
            (&legacy::WINDOWS_1257, 0xA5),
        ];
        for (encoding, byte) in cases {
            let mut dst = [0u16; 4];
            let (result, read, written) = encoding
                .new_decoder()
                .decode_to_utf16_without_replacement(&[byte], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "byte {byte:02X}");
            assert_eq!((read, written), (1, 0), "byte {byte:02X}");

            let mut out = [0u8; 4];
            let (result, read, written) = encoding
                .new_encoder()
                .encode_from_utf16_without_replacement(&[u16::from(byte)], &mut out, true);
            let expected = match char::from_u32(u32::from(byte)) {
                Some(ch) => ch,
                None => panic!("every single-byte pointer is below the surrogates"),
            };
            assert_eq!(result, EncoderResult::Unmappable(expected), "byte {byte:02X}");
            assert_eq!((read, written), (1, 0), "byte {byte:02X}");
        }
    }

    #[test]
    fn iso_8859_2_round_trips_its_c1_run() {
        for byte in 0x80u8..=0x9F {
            let units = decode_units(&legacy::ISO_8859_2, &[byte]);
            assert_eq!(units, vec![u16::from(byte)], "byte {byte:02X}");
            let mut dst = [0u8; 16];
            let (result, _read, written, _unmappables) = legacy::ISO_8859_2
                .new_encoder()
                .encode_from_utf16(&units, &mut dst, true);
            assert_eq!(result, CoderResult::InputEmpty, "byte {byte:02X}");
            assert_eq!(dst[0], byte, "byte {byte:02X}");
            assert_eq!(written, 1);
        }
    }

    #[test]
    fn the_raw_encoder_reports_the_astral_scalar_it_cannot_map() {
        let mut encoder = SingleByteEncoder::new(&single_byte_tables::INDEX_WINDOWS_1252, 0x00A0, 0x20, 96);
        let mut dst = [0u8; 8];
        let (result, read, written) = encoder.encode_from_utf16_raw(&[0xD83D, 0xDE00], &mut dst, true);
        assert_eq!(result, EncoderResult::Unmappable('\u{1F600}'));
        assert_eq!((read, written), (2, 0));
    }

    #[test]
    fn the_raw_encoder_reports_an_unpaired_surrogate_as_unmappable() {
        let mut encoder = SingleByteEncoder::new(&single_byte_tables::INDEX_WINDOWS_1252, 0x00A0, 0x20, 96);
        let mut dst = [0u8; 8];
        let (high, _rh, _wh) = encoder.encode_from_utf16_raw(&[0xD83D], &mut dst, true);
        let (low, _rl, _wl) = encoder.encode_from_utf16_raw(&[0xDE00], &mut dst, true);
        assert!(matches!(high, EncoderResult::Unmappable(_)));
        assert!(matches!(low, EncoderResult::Unmappable(_)));
    }

    #[test]
    fn the_decoder_refills_a_destination_shorter_than_the_input() {
        let src = b"KOI8-R text that is longer than the destination";
        let mut decoder = legacy::KOI8_R.new_decoder_without_bom_handling();
        let mut chunk = [0u16; 16];
        let mut out = Vec::new();
        let mut read_total = 0usize;
        let mut rounds = 0usize;
        loop {
            rounds += 1;
            assert!(rounds < 16, "the decoder stopped making progress");
            let (result, read, written) =
                decoder.decode_to_utf16_without_replacement(&src[read_total..], &mut chunk, true);
            read_total += read;
            out.extend_from_slice(&chunk[..written]);
            match result {
                DecoderResult::OutputFull => assert!(written > 0, "stalled on a full destination"),
                DecoderResult::InputEmpty => break,
                DecoderResult::Malformed(bad, _second) => {
                    panic!("ascii input reported malformed({bad})")
                }
            }
        }
        assert_eq!(read_total, src.len());
        assert_eq!(out.len(), src.len());
        assert_eq!(
            out,
            src.iter()
                .map(|&b| u16::from(b))
                .collect::<Vec<u16>>()
        );
    }

    #[test]
    fn an_encoder_needs_room_for_the_replacement_before_it_starts() {
        // the replacing encoder reserves NCR_EXTRA bytes for a numeric character
        // reference, so a tiny destination reports OutputFull instead of writing
        let mut dst = [0u8; 4];
        let (result, read, written, _unmappables) = legacy::ISO_8859_7
            .new_encoder()
            .encode_from_utf8("\u{03A3}", &mut dst, true);
        assert_eq!(result, CoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
    }
}
