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

use crate::encoding::{DecoderResult, EncoderResult, Encoding, VariantDecoder};
use crate::encoding_ascii::{ascii_to_basic_latin, basic_latin_to_ascii};
use crate::encoding_handles::{ByteSource, CopyAsciiResult, Space, Utf8Destination};
#[derive(Debug, Clone)]
pub struct SingleByteDecoder {
    pub(crate) table: &'static [u16; 128],
}

impl SingleByteDecoder {
    pub fn new(data: &'static [u16; 128]) -> VariantDecoder {
        VariantDecoder::SingleByte(SingleByteDecoder { table: data })
    }

    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        Some(byte_length)
    }

    pub fn max_utf8_buffer_length_without_replacement(
        &self,
        byte_length: usize,
    ) -> Option<usize> {
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
                        *(self.table.get_unchecked(non_ascii as usize - 0x80usize))
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
                            return (
                                DecoderResult::InputEmpty,
                                src_consumed,
                                dest_again.written(),
                            );
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
                                        let dest_again_again =
                                            destination_handle.write_ascii(b);
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
                                                    match dest_again_again
                                                        .check_space_bmp()
                                                    {
                                                        Space::Full(
                                                            dst_written_again,
                                                        ) => {
                                                            return (
                                                                DecoderResult::OutputFull,
                                                                source_handle_again
                                                                    .consumed(),
                                                                dst_written_again,
                                                            );
                                                        }
                                                        Space::Available(
                                                            destination_handle_again,
                                                        ) => {
                                                            let (
                                                                b_again,
                                                                unread_handle_again,
                                                            ) = source_handle_again
                                                                .read();
                                                            unread_handle_again.commit();
                                                            b = b_again;
                                                            destination_handle =
                                                                destination_handle_again;
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
                            *(self.table.get_unchecked(non_ascii as usize - 0x80usize))
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
                        // use ASCII punctuation, so this avoid going to
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
        encoding: &'static Encoding,
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

    pub fn max_buffer_length_from_utf16_without_replacement(
        &self,
        u16_length: usize,
    ) -> Option<usize> {
        Some(u16_length)
    }

    pub fn max_buffer_length_from_utf8_without_replacement(
        &self,
        byte_length: usize,
    ) -> Option<usize> {
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
            if let Some(pos) = position(&self.table[64..self.run_byte_offset], code_unit)
            {
                return Some(((128 + 64) + pos) as u8);
            }
            // Search second quadrant
            if let Some(pos) = position(&self.table[32..64], code_unit) {
                return Some(((128 + 32) + pos) as u8);
            }
        } else if let Some(pos) =
            position(&self.table[32..self.run_byte_offset], code_unit)
        {
            // windows-1252, windows-874, ISO-8859-15 and ISO-8859-5
            // Search second quadrant before the run
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
                                    let second = u32::from(unsafe {
                                        *src.get_unchecked(converted + 1)
                                    });
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
                                                - (((0xD800u32 << 10) - 0x1_0000u32)
                                                    + 0xDC00u32),
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
            if cp > 0xFFFF {
                return (EncoderResult::Unmappable(ch), i, written);
            }
            match self.encode_u16(cp as u16) {
                Some(byte) => {
                    dst[written] = byte;
                    written += 1;
                    read = i + ch.len_utf8();
                }
                None => return (EncoderResult::Unmappable(ch), i, written),
            }
        }
        (EncoderResult::InputEmpty, src.len(), written)
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
