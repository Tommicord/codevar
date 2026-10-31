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

use crate::ascii::{ascii_to_basic_latin, basic_latin_to_ascii, validate_ascii};
use crate::encoding::{CoderResult, DecoderResult, EncoderResult, UTF_8, convert_utf16_to_utf8_partial};
use crate::encoding::{VariantDecoder, VariantEncoder};
use crate::handles::{ByteSource, Space, Utf8Destination, Utf16Destination};
use alloc::vec;
use alloc::vec::Vec;

#[repr(align(64))]
pub struct Utf8Data {
    pub table: [u8; 384],
}

pub static UTF8_DATA: Utf8Data = Utf8Data {
    table: [
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
        84, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32, 8, 8, 64,
        8, 8, 8, 128, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ],
};

pub fn utf8_valid_up_to(src: &[u8]) -> usize {
    let mut read = 0;
    'outer: loop {
        let mut byte = {
            let src_remaining = &src[read..];
            match validate_ascii(src_remaining) {
                None => return src.len(),
                Some((non_ascii, consumed)) => {
                    read += consumed;
                    non_ascii
                }
            }
        };
        if read + 4 <= src.len() {
            'inner: loop {
                if (0xC2..=0xDF).contains(&byte) {
                    let second = unsafe { *(src.get_unchecked(read + 1)) };
                    if !in_inclusive_range8(second, 0x80, 0xBF) {
                        break 'outer;
                    }
                    read += 2;
                    if read + 4 <= src.len() {
                        byte = unsafe { *(src.get_unchecked(read)) };
                        if byte < 0x80 {
                            read += 1;
                            continue 'outer;
                        }
                        continue 'inner;
                    }
                    break 'inner;
                }
                if byte < 0xF0 {
                    'three: loop {
                        let second = unsafe { *(src.get_unchecked(read + 1)) };
                        let third = unsafe { *(src.get_unchecked(read + 2)) };
                        if ((UTF8_DATA.table[usize::from(second)]
                            & unsafe {
                                *(UTF8_DATA
                                    .table
                                    .get_unchecked(byte as usize + 0x80))
                            })
                            | (third >> 6))
                            != 2
                        {
                            break 'outer;
                        }
                        read += 3;
                        if read + 4 <= src.len() {
                            byte = unsafe { *(src.get_unchecked(read)) };
                            if in_inclusive_range8(byte, 0xE0, 0xEF) {
                                continue 'three;
                            }
                            if byte < 0x80 {
                                read += 1;
                                continue 'outer;
                            }
                            continue 'inner;
                        }
                        break 'inner;
                    }
                }
                let second = unsafe { *(src.get_unchecked(read + 1)) };
                let third = unsafe { *(src.get_unchecked(read + 2)) };
                let fourth = unsafe { *(src.get_unchecked(read + 3)) };
                if (u16::from(
                    UTF8_DATA.table[usize::from(second)]
                        & unsafe {
                            *(UTF8_DATA
                                .table
                                .get_unchecked(byte as usize + 0x80))
                        },
                ) | u16::from(third >> 6)
                    | (u16::from(fourth & 0xC0) << 2))
                    != 0x202
                {
                    break 'outer;
                }
                read += 4;
                if read + 4 <= src.len() {
                    byte = unsafe { *(src.get_unchecked(read)) };
                    if byte < 0x80 {
                        read += 1;
                        continue 'outer;
                    }
                    continue 'inner;
                }
                break 'inner;
            }
        }
        'tail: loop {
            if read >= src.len() {
                break 'outer;
            }
            byte = src[read];
            if byte < 0x80 {
                read += 1;
                continue 'tail;
            }
            if in_inclusive_range8(byte, 0xC2, 0xDF) {
                let new_read = read + 2;
                if new_read > src.len() {
                    break 'outer;
                }
                let second = src[read + 1];
                if !in_inclusive_range8(second, 0x80, 0xBF) {
                    break 'outer;
                }
                read += 2;
                continue 'tail;
            }
            if byte < 0xF0 {
                let new_read = read + 3;
                if new_read > src.len() {
                    break 'outer;
                }
                let second = src[read + 1];
                let third = src[read + 2];
                if ((UTF8_DATA.table[usize::from(second)]
                    & unsafe {
                        *(UTF8_DATA
                            .table
                            .get_unchecked(byte as usize + 0x80))
                    })
                    | (third >> 6))
                    != 2
                {
                    break 'outer;
                }
                read += 3;
                break 'outer;
            }
            break 'outer;
        }
    }
    read
}

pub fn convert_utf8_to_utf16_up_to_invalid(src: &[u8], dst: &mut [u16]) -> (usize, usize) {
    let mut read = 0;
    let mut written = 0;
    'outer: loop {
        let mut byte = {
            let src_remaining = &src[read..];
            let dst_remaining = &mut dst[written..];
            let length = ::core::cmp::min(src_remaining.len(), dst_remaining.len());
            match ascii_to_basic_latin(src_remaining, dst_remaining) {
                None => {
                    read += length;
                    written += length;
                    break 'outer;
                }
                Some((non_ascii, consumed)) => {
                    read += consumed;
                    written += consumed;
                    non_ascii
                }
            }
        };
        if read + 4 <= src.len() {
            'inner: loop {
                if in_inclusive_range8(byte, 0xC2, 0xDF) {
                    let second = unsafe { *(src.get_unchecked(read + 1)) };
                    if !in_inclusive_range8(second, 0x80, 0xBF) {
                        break 'outer;
                    }
                    unsafe {
                        *(dst.get_unchecked_mut(written)) =
                            ((u16::from(byte) & 0x1F) << 6) | (u16::from(second) & 0x3F);
                    }
                    read += 2;
                    written += 1;
                    if written == dst.len() {
                        break 'outer;
                    }
                    if read + 4 <= src.len() {
                        byte = unsafe { *(src.get_unchecked(read)) };
                        if byte < 0x80 {
                            unsafe {
                                *(dst.get_unchecked_mut(written)) = u16::from(byte);
                            }
                            read += 1;
                            written += 1;
                            continue 'outer;
                        }
                        continue 'inner;
                    }
                    break 'inner;
                }
                if byte < 0xF0 {
                    'three: loop {
                        let second = unsafe { *(src.get_unchecked(read + 1)) };
                        let third = unsafe { *(src.get_unchecked(read + 2)) };
                        if ((UTF8_DATA.table[usize::from(second)]
                            & unsafe {
                                *(UTF8_DATA
                                    .table
                                    .get_unchecked(byte as usize + 0x80))
                            })
                            | (third >> 6))
                            != 2
                        {
                            break 'outer;
                        }
                        let point = ((u16::from(byte) & 0xF) << 12)
                            | ((u16::from(second) & 0x3F) << 6)
                            | (u16::from(third) & 0x3F);
                        unsafe {
                            *(dst.get_unchecked_mut(written)) = point;
                        }
                        read += 3;
                        written += 1;
                        if written == dst.len() {
                            break 'outer;
                        }
                        if read + 4 <= src.len() {
                            byte = unsafe { *(src.get_unchecked(read)) };
                            if in_inclusive_range8(byte, 0xE0, 0xEF) {
                                continue 'three;
                            }
                            if byte < 0x80 {
                                unsafe {
                                    *(dst.get_unchecked_mut(written)) = u16::from(byte);
                                }
                                read += 1;
                                written += 1;
                                continue 'outer;
                            }
                            continue 'inner;
                        }
                        break 'inner;
                    }
                }
                if written + 1 == dst.len() {
                    break 'outer;
                }
                let second = unsafe { *(src.get_unchecked(read + 1)) };
                let third = unsafe { *(src.get_unchecked(read + 2)) };
                let fourth = unsafe { *(src.get_unchecked(read + 3)) };
                if (u16::from(
                    UTF8_DATA.table[usize::from(second)]
                        & unsafe {
                            *(UTF8_DATA
                                .table
                                .get_unchecked(byte as usize + 0x80))
                        },
                ) | u16::from(third >> 6)
                    | (u16::from(fourth & 0xC0) << 2))
                    != 0x202
                {
                    break 'outer;
                }
                let point = ((u32::from(byte) & 0x7) << 18)
                    | ((u32::from(second) & 0x3F) << 12)
                    | ((u32::from(third) & 0x3F) << 6)
                    | (u32::from(fourth) & 0x3F);
                unsafe {
                    *(dst.get_unchecked_mut(written)) = (0xD7C0 + (point >> 10)) as u16;
                }
                unsafe {
                    *(dst.get_unchecked_mut(written + 1)) = (0xDC00 + (point & 0x3FF)) as u16;
                }
                read += 4;
                written += 2;
                if written == dst.len() {
                    break 'outer;
                }
                if read + 4 <= src.len() {
                    byte = unsafe { *(src.get_unchecked(read)) };
                    if byte < 0x80 {
                        unsafe {
                            *(dst.get_unchecked_mut(written)) = u16::from(byte);
                        }
                        read += 1;
                        written += 1;
                        continue 'outer;
                    }
                    continue 'inner;
                }
                break 'inner;
            }
        }
        'tail: loop {
            if read >= src.len() || written >= dst.len() {
                break 'outer;
            }
            byte = src[read];
            if byte < 0x80 {
                dst[written] = u16::from(byte);
                read += 1;
                written += 1;
                continue 'tail;
            }
            if in_inclusive_range8(byte, 0xC2, 0xDF) {
                let new_read = read + 2;
                if new_read > src.len() {
                    break 'outer;
                }
                let second = src[read + 1];
                if !in_inclusive_range8(second, 0x80, 0xBF) {
                    break 'outer;
                }
                dst[written] = ((u16::from(byte) & 0x1F) << 6) | (u16::from(second) & 0x3F);
                read += 2;
                written += 1;
                continue 'tail;
            }
            if byte < 0xF0 {
                let new_read = read + 3;
                if new_read > src.len() {
                    break 'outer;
                }
                let second = src[read + 1];
                let third = src[read + 2];
                if ((UTF8_DATA.table[usize::from(second)]
                    & unsafe {
                        *(UTF8_DATA
                            .table
                            .get_unchecked(byte as usize + 0x80))
                    })
                    | (third >> 6))
                    != 2
                {
                    break 'outer;
                }
                let point = ((u16::from(byte) & 0xF) << 12)
                    | ((u16::from(second) & 0x3F) << 6)
                    | (u16::from(third) & 0x3F);
                dst[written] = point;
                read += 3;
                written += 1;
                break 'outer;
            }
            break 'outer;
        }
    }
    (read, written)
}

pub fn convert_utf16_to_utf8_partial_inner(src: &[u16], dst: &mut [u8]) -> (usize, usize) {
    let mut read = 0;
    let mut written = 0;
    'outer: loop {
        let mut unit = {
            let src_remaining = &src[read..];
            let dst_remaining = &mut dst[written..];
            let length = if dst_remaining.len() < src_remaining.len() {
                dst_remaining.len()
            } else {
                src_remaining.len()
            };
            match basic_latin_to_ascii(src_remaining, dst_remaining) {
                None => {
                    read += length;
                    written += length;
                    return (read, written);
                }
                Some((non_ascii, consumed)) => {
                    read += consumed;
                    written += consumed;
                    non_ascii
                }
            }
        };
        'inner: loop {
            {
                if written.saturating_add(4) > dst.len() {
                    return (read, written);
                }
                read += 1;
                if unit < 0x800 {
                    unsafe {
                        *(dst.get_unchecked_mut(written)) = (unit >> 6) as u8 | 0xC0u8;
                        written += 1;
                        *(dst.get_unchecked_mut(written)) = (unit & 0x3F) as u8 | 0x80u8;
                        written += 1;
                    }
                    break;
                }
                let unit_minus_surrogate_start = unit.wrapping_sub(0xD800);
                if unit_minus_surrogate_start > (0xDFFF - 0xD800) {
                    unsafe {
                        *(dst.get_unchecked_mut(written)) = (unit >> 12) as u8 | 0xE0u8;
                        written += 1;
                        *(dst.get_unchecked_mut(written)) = ((unit & 0xFC0) >> 6) as u8 | 0x80u8;
                        written += 1;
                        *(dst.get_unchecked_mut(written)) = (unit & 0x3F) as u8 | 0x80u8;
                        written += 1;
                    }
                    break;
                }
                if unit_minus_surrogate_start <= (0xDBFF - 0xD800) {
                    if read >= src.len() {
                        unsafe {
                            *(dst.get_unchecked_mut(written)) = 0xEFu8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) = 0xBFu8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) = 0xBDu8;
                            written += 1;
                        }
                        return (read, written);
                    }
                    let second = src[read];
                    let second_minus_low_surrogate_start = second.wrapping_sub(0xDC00);
                    if second_minus_low_surrogate_start <= (0xDFFF - 0xDC00) {
                        read += 1;
                        let astral = (u32::from(unit) << 10) + u32::from(second)
                            - (((0xD800u32 << 10) - 0x10000u32) + 0xDC00u32);
                        unsafe {
                            *(dst.get_unchecked_mut(written)) = (astral >> 18) as u8 | 0xF0u8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) = ((astral & 0x3F000u32) >> 12) as u8 | 0x80u8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) = ((astral & 0xFC0u32) >> 6) as u8 | 0x80u8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) = (astral & 0x3F) as u8 | 0x80u8;
                            written += 1;
                        }
                        break;
                    }
                }
                unsafe {
                    *(dst.get_unchecked_mut(written)) = 0xEFu8;
                    written += 1;
                    *(dst.get_unchecked_mut(written)) = 0xBFu8;
                    written += 1;
                    *(dst.get_unchecked_mut(written)) = 0xBDu8;
                    written += 1;
                }
            }
            'punctuation: loop {
                if read >= src.len() {
                    return (read, written);
                }
                unit = src[read];
                if unit < 0x80 {
                    if written >= dst.len() {
                        return (read, written);
                    }
                    dst[written] = unit as u8;
                    read += 1;
                    written += 1;
                    if unit < 0x3C {
                        continue 'punctuation;
                    }
                    continue 'outer;
                }
                continue 'inner;
            }
        }
    }
}

pub fn convert_utf16_to_utf8_partial_tail(src: &[u16], dst: &mut [u8]) -> (usize, usize) {
    let mut read = 0;
    let mut written = 0;
    let mut unit = src[read];
    if unit < 0x800 {
        loop {
            if unit < 0x80 {
                if written >= dst.len() {
                    return (read, written);
                }
                read += 1;
                dst[written] = unit as u8;
                written += 1;
            } else if unit < 0x800 {
                if written + 2 > dst.len() {
                    return (read, written);
                }
                read += 1;
                dst[written] = (unit >> 6) as u8 | 0xC0u8;
                written += 1;
                dst[written] = (unit & 0x3F) as u8 | 0x80u8;
                written += 1;
            } else {
                return (read, written);
            }
            if read >= src.len() {
                return (read, written);
            }
            unit = src[read];
        }
    }
    if written + 3 > dst.len() {
        return (read, written);
    }
    read += 1;
    let unit_minus_surrogate_start = unit.wrapping_sub(0xD800);
    if unit_minus_surrogate_start <= (0xDFFF - 0xD800) {
        if unit_minus_surrogate_start <= (0xDBFF - 0xD800) {
            if read >= src.len() {
                unit = 0xFFFD;
            } else {
                let second = src[read];
                if in_inclusive_range16(second, 0xDC00, 0xDFFF) {
                    read -= 1;
                    return (read, written);
                }
                unit = 0xFFFD;
            }
        } else {
            unit = 0xFFFD;
        }
    }
    dst[written] = (unit >> 12) as u8 | 0xE0u8;
    written += 1;
    dst[written] = ((unit & 0xFC0) >> 6) as u8 | 0x80u8;
    written += 1;
    dst[written] = (unit & 0x3F) as u8 | 0x80u8;
    written += 1;
    debug_assert_eq!(written, dst.len());
    (read, written)
}

#[derive(Debug, Clone)]
pub struct Utf8Decoder {
    code_point: u32,
    bytes_seen: usize,
    bytes_needed: usize,
    lower_boundary: u8,
    upper_boundary: u8,
}

impl Utf8Decoder {
    pub fn new_inner() -> Utf8Decoder {
        Utf8Decoder {
            code_point: 0,
            bytes_seen: 0,
            bytes_needed: 0,
            lower_boundary: 0x80,
            upper_boundary: 0xBF,
        }
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> VariantDecoder {
        VariantDecoder::Utf8(Utf8Decoder::new_inner())
    }

    pub fn in_neutral_state(&self) -> bool {
        self.bytes_needed == 0
    }

    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_add(1 + self.extra_from_state())
    }

    pub fn max_utf8_buffer_length_without_replacement(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_add(3 + self.extra_from_state())
    }

    pub fn max_utf8_buffer_length(&self, byte_length: usize) -> Option<usize> {
        checked_add(
            3,
            checked_mul(3, byte_length.checked_add(self.extra_from_state())),
        )
    }

    fn extra_from_state(&self) -> usize {
        if self.bytes_needed == 0 {
            0
        } else {
            self.bytes_seen + 1
        }
    }

    pub fn decode_to_utf8_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        last: bool,
    ) -> (DecoderResult, usize, usize) {
        let mut source = ByteSource::new(src);
        let mut dest = Utf8Destination::new(dst);
        loop {
            if self.bytes_needed == 0 {
                dest.copy_utf8_up_to_invalid_from(&mut source);
            }
            match source.check_available() {
                Space::Full(src_consumed) => {
                    if last && self.bytes_needed != 0 {
                        let bad_bytes = (self.bytes_seen + 1) as u8;
                        self.code_point = 0;
                        self.bytes_needed = 0;
                        self.bytes_seen = 0;
                        return (
                            DecoderResult::Malformed(bad_bytes, 0),
                            src_consumed,
                            dest.written(),
                        );
                    }
                    return (DecoderResult::InputEmpty, src_consumed, dest.written());
                }
                Space::Available(source_handle) => match dest.check_space_astral() {
                    Space::Full(dst_written) => {
                        return (DecoderResult::OutputFull, source_handle.consumed(), dst_written);
                    }
                    Space::Available(destination_handle) => {
                        let (b, unread_handle) = source_handle.read();
                        if self.bytes_needed == 0 {
                            if b < 0x80u8 {
                                destination_handle.write_ascii(b);
                                continue;
                            }
                            if b < 0xC2u8 {
                                return (
                                    DecoderResult::Malformed(1, 0),
                                    unread_handle.consumed(),
                                    destination_handle.written(),
                                );
                            }
                            if b < 0xE0u8 {
                                self.bytes_needed = 1;
                                self.code_point = u32::from(b) & 0x1F;
                                continue;
                            }
                            if b < 0xF0u8 {
                                if b == 0xE0u8 {
                                    self.lower_boundary = 0xA0;
                                } else if b == 0xEDu8 {
                                    self.upper_boundary = 0x9F;
                                }
                                self.bytes_needed = 2;
                                self.code_point = u32::from(b) & 0xF;
                                continue;
                            }
                            if b < 0xF5u8 {
                                if b == 0xF0u8 {
                                    self.lower_boundary = 0x90;
                                } else if b == 0xF4u8 {
                                    self.upper_boundary = 0x8F;
                                }
                                self.bytes_needed = 3;
                                self.code_point = u32::from(b) & 0x7;
                                continue;
                            }
                            return (
                                DecoderResult::Malformed(1, 0),
                                unread_handle.consumed(),
                                destination_handle.written(),
                            );
                        }
                        if !(b >= self.lower_boundary && b <= self.upper_boundary) {
                            let bad_bytes = (self.bytes_seen + 1) as u8;
                            self.code_point = 0;
                            self.bytes_needed = 0;
                            self.bytes_seen = 0;
                            self.lower_boundary = 0x80;
                            self.upper_boundary = 0xBF;
                            unread_handle.unread();
                            return (
                                DecoderResult::Malformed(bad_bytes, 0),
                                source.consumed(),
                                destination_handle.written(),
                            );
                        }
                        self.lower_boundary = 0x80;
                        self.upper_boundary = 0xBF;
                        self.code_point = (self.code_point << 6) | (u32::from(b) & 0x3F);
                        self.bytes_seen += 1;
                        if self.bytes_seen != self.bytes_needed {
                            continue;
                        }
                        if self.bytes_needed == 3 {
                            destination_handle.write_astral(self.code_point);
                        } else {
                            destination_handle.write_bmp_excl_ascii(self.code_point as u16);
                        }
                        self.code_point = 0;
                        self.bytes_needed = 0;
                        self.bytes_seen = 0;
                        continue;
                    }
                },
            }
        }
    }

    pub fn decode_to_utf16_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        last: bool,
    ) -> (DecoderResult, usize, usize) {
        let mut source = ByteSource::new(src);
        let mut dest = Utf16Destination::new(dst);
        loop {
            if self.bytes_needed == 0 {
                dest.copy_utf8_up_to_invalid_from(&mut source);
            }
            match source.check_available() {
                Space::Full(src_consumed) => {
                    if last && self.bytes_needed != 0 {
                        let bad_bytes = (self.bytes_seen + 1) as u8;
                        self.code_point = 0;
                        self.bytes_needed = 0;
                        self.bytes_seen = 0;
                        return (
                            DecoderResult::Malformed(bad_bytes, 0),
                            src_consumed,
                            dest.written(),
                        );
                    }
                    return (DecoderResult::InputEmpty, src_consumed, dest.written());
                }
                Space::Available(source_handle) => match dest.check_space_astral() {
                    Space::Full(dst_written) => {
                        return (DecoderResult::OutputFull, source_handle.consumed(), dst_written);
                    }
                    Space::Available(destination_handle) => {
                        let (b, unread_handle) = source_handle.read();
                        if self.bytes_needed == 0 {
                            if b < 0x80u8 {
                                destination_handle.write_ascii(b);
                                continue;
                            }
                            if b < 0xC2u8 {
                                return (
                                    DecoderResult::Malformed(1, 0),
                                    unread_handle.consumed(),
                                    destination_handle.written(),
                                );
                            }
                            if b < 0xE0u8 {
                                self.bytes_needed = 1;
                                self.code_point = u32::from(b) & 0x1F;
                                continue;
                            }
                            if b < 0xF0u8 {
                                if b == 0xE0u8 {
                                    self.lower_boundary = 0xA0;
                                } else if b == 0xEDu8 {
                                    self.upper_boundary = 0x9F;
                                }
                                self.bytes_needed = 2;
                                self.code_point = u32::from(b) & 0xF;
                                continue;
                            }
                            if b < 0xF5u8 {
                                if b == 0xF0u8 {
                                    self.lower_boundary = 0x90;
                                } else if b == 0xF4u8 {
                                    self.upper_boundary = 0x8F;
                                }
                                self.bytes_needed = 3;
                                self.code_point = u32::from(b) & 0x7;
                                continue;
                            }
                            return (
                                DecoderResult::Malformed(1, 0),
                                unread_handle.consumed(),
                                destination_handle.written(),
                            );
                        }
                        if !(b >= self.lower_boundary && b <= self.upper_boundary) {
                            let bad_bytes = (self.bytes_seen + 1) as u8;
                            self.code_point = 0;
                            self.bytes_needed = 0;
                            self.bytes_seen = 0;
                            self.lower_boundary = 0x80;
                            self.upper_boundary = 0xBF;
                            unread_handle.unread();
                            return (
                                DecoderResult::Malformed(bad_bytes, 0),
                                source.consumed(),
                                destination_handle.written(),
                            );
                        }
                        self.lower_boundary = 0x80;
                        self.upper_boundary = 0xBF;
                        self.code_point = (self.code_point << 6) | (u32::from(b) & 0x3F);
                        self.bytes_seen += 1;
                        if self.bytes_seen != self.bytes_needed {
                            continue;
                        }
                        if self.bytes_needed == 3 {
                            destination_handle.write_astral(self.code_point);
                        } else {
                            destination_handle.write_bmp_excl_ascii(self.code_point as u16);
                        }
                        self.code_point = 0;
                        self.bytes_needed = 0;
                        self.bytes_seen = 0;
                        continue;
                    }
                },
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Utf8Encoder;

impl Utf8Encoder {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> crate::encoding::Encoder {
        crate::encoding::Encoder::new(&UTF_8, VariantEncoder::Utf8(Utf8Encoder))
    }

    pub fn max_buffer_length_from_utf16_without_replacement(&self, u16_length: usize) -> Option<usize> {
        u16_length.checked_mul(3)
    }

    pub fn max_buffer_length_from_utf8_without_replacement(&self, byte_length: usize) -> Option<usize> {
        Some(byte_length)
    }

    pub fn encode_from_utf16_raw(
        &mut self,
        src: &[u16],
        dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let (read, written) = convert_utf16_to_utf8_partial(src, dst);
        if read == src.len() {
            (EncoderResult::InputEmpty, read, written)
        } else {
            (EncoderResult::OutputFull, read, written)
        }
    }

    pub fn encode_from_utf8_raw(
        &mut self,
        src: &str,
        dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let bytes = src.as_bytes();
        let mut to_write = bytes.len();
        if to_write <= dst.len() {
            dst[..to_write].copy_from_slice(bytes);
            return (EncoderResult::InputEmpty, to_write, to_write);
        }
        to_write = dst.len();
        while (bytes[to_write] & 0xC0) == 0x80 {
            to_write -= 1;
        }
        dst[..to_write].copy_from_slice(&bytes[..to_write]);
        (EncoderResult::OutputFull, to_write, to_write)
    }
}

/// Encodes `text` as UTF-8 bytes through [`Utf8Encoder`].
///
/// The destination buffer is sized to `text.len()` and grown only if the
/// encoder reports [`CoderResult::OutputFull`] (never for pure UTF-8, but
/// keeps the control flow correct if the encoder ever changes).
///
/// # Examples
///
/// ```
/// use codevar_textlike_encode::utf8::encode_text;
///
/// assert_eq!(encode_text("héllo"), b"h\xc3\xa9llo");
/// assert_eq!(encode_text(""), Vec::<u8>::new());
/// ```
#[must_use]
pub fn encode_text(text: &str) -> Vec<u8> {
    let mut encoder = Utf8Encoder::new();
    let mut buf = vec![0u8; text.len()];
    let mut written = 0usize;
    let mut read = 0usize;
    loop {
        let (result, r, w, _) = encoder.encode_from_utf8(&text[read..], &mut buf[written..], true);
        read += r;
        written += w;
        match result {
            CoderResult::InputEmpty => break,
            CoderResult::OutputFull => {
                let needed = written + (text.len() - read);
                let new_len = buf.len().max(needed).saturating_mul(2).max(16);
                buf.resize(new_len, 0);
            }
        }
    }
    buf.truncate(written);
    buf
}

/// Encodes as much of `text` into `dst` as fits, returning
/// `(bytes_read, bytes_written)`.
///
/// Stops on a UTF-8 character boundary when `dst` fills up, so a caller can
/// drain the remaining input with a larger buffer. An empty `dst` reads and
/// writes nothing.
#[must_use]
pub fn encode_text_into(text: &str, dst: &mut [u8]) -> (usize, usize) {
    let mut encoder = Utf8Encoder::new();
    let (result, read, written, _) = encoder.encode_from_utf8(text, dst, true);
    debug_assert!(matches!(
        result,
        CoderResult::InputEmpty | CoderResult::OutputFull
    ));
    (read, written)
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
fn checked_mul(num: usize, opt: Option<usize>) -> Option<usize> {
    if let Some(n) = opt {
        n.checked_mul(num)
    } else {
        None
    }
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
fn in_range16(i: u16, start: u16, end: u16) -> bool {
    i.wrapping_sub(start) < (end - start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::DecoderResult;

    #[test]
    fn test_encode_text_roundtrip() {
        for text in ["", "ascii", "héllo wörld", "🙂☃€", "\u{10FFFF}"] {
            assert_eq!(encode_text(text), text.as_bytes(), "text={text:?}");
        }
    }

    #[test]
    fn test_encode_text_into_fills_buffer_on_char_boundary() {
        let text = "aé🙂b";
        let mut dst = [0u8; 4];
        let (read, written) = encode_text_into(text, &mut dst);
        // "a" + "é" (3 bytes) fits; the 4-byte emoji would not start.
        assert_eq!(read, 3);
        assert_eq!(written, 3);
        assert_eq!(&dst[..written], "aé".as_bytes());

        // Buffer exactly large enough for the emoji after the prefix.
        let mut dst2 = [0u8; 7];
        let (read, written) = encode_text_into(text, &mut dst2);
        assert_eq!(read, 7);
        assert_eq!(written, 7);
        assert_eq!(&dst2[..written], "aé🙂".as_bytes());

        let mut big = [0u8; 64];
        let (read, written) = encode_text_into(text, &mut big);
        assert_eq!(read, text.len());
        assert_eq!(written, text.len());
        assert_eq!(&big[..written], text.as_bytes());

        let mut empty: [u8; 0] = [];
        let (read, written) = encode_text_into("x", &mut empty);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn test_utf8_valid_up_to() {
        assert_eq!(utf8_valid_up_to(b"hello"), 5);
        assert_eq!(utf8_valid_up_to(b"hello\x80"), 5);
        assert_eq!(utf8_valid_up_to(b"\xC3\xA9"), 2);
        assert_eq!(utf8_valid_up_to(b"\xC3"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x9F\x92\xA9"), 4);
        // Three-byte sequences exercising the corrected table (lead 0xE0-0xEF)
        assert_eq!(utf8_valid_up_to(b"\xE2\x98\x83"), 3); // snowman U+2603
        assert_eq!(utf8_valid_up_to(b"\xEF\xBF\xBD"), 3); // replacement U+FFFD
        // Four-byte sequences (lead 0xF0-0xF4)
        assert_eq!(utf8_valid_up_to(b"\xF0\x90\x80\x80"), 4); // U+10000
        assert_eq!(utf8_valid_up_to(b"\xF4\x8F\xBF\xBF"), 4); // U+10FFFF
        // Invalid lead / overlong / out of range
        assert_eq!(utf8_valid_up_to(b"\xC0\x80"), 0); // overlong
        assert_eq!(utf8_valid_up_to(b"\xE0\x80\x80"), 0); // overlong 3-byte
        assert_eq!(utf8_valid_up_to(b"\xF0\x80\x80\x80"), 0); // overlong 4-byte
        assert_eq!(utf8_valid_up_to(b"\xF4\x90\x80\x80"), 0); // > U+10FFFF
        assert_eq!(utf8_valid_up_to(b"\xED\xA0\x80"), 0); // surrogate
        assert_eq!(utf8_valid_up_to(b"\xF5\x80\x80\x80"), 0); // invalid lead
        // Mixed ASCII + multi-byte
        assert_eq!(utf8_valid_up_to(b"ok \xE2\x98\x83 \xF0\x9F\x92\xA9!"), 12);
    }

    #[test]
    fn test_utf8_decode_valid() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"hello", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!(read, 5);
        assert_eq!(written, 5);
    }

    #[test]
    fn test_utf8_decode_malformed() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        // \xC3 is an incomplete lead; Z is not a continuation and is unread
        let (result, _read, _written) = decoder.decode_to_utf8_raw(b"a\xC3Z", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        // Incomplete sequence at EOF
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        let (result, _read, _written) = decoder.decode_to_utf8_raw(b"a\xC3", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        // Invalid lead byte
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        let (result, _read, _written) = decoder.decode_to_utf8_raw(b"\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
    }

    #[test]
    fn test_utf8_encode_from_utf16() {
        let mut encoder = Utf8Encoder;
        let src: Vec<u16> = "\u{1F4A9}".encode_utf16().collect();
        let mut dst = [0u8; 4];
        let (result, read, written) = encoder.encode_from_utf16_raw(&src, &mut dst, true);
        assert_eq!(result, EncoderResult::InputEmpty);
        assert_eq!(read, 2);
        assert_eq!(written, 4);
        assert_eq!(&dst[..4], "\u{1F4A9}".as_bytes());
    }

    #[test]
    fn test_convert_utf8_to_utf16() {
        let src = "abc\u{1F4A9}";
        let mut dst: Vec<u16> = vec![0; src.len() + 1];
        let (read, written) = convert_utf8_to_utf16_up_to_invalid(src.as_bytes(), &mut dst[..]);
        assert_eq!(read, src.len());
        assert_eq!(written, src.encode_utf16().count());
    }

    #[test]
    fn test_convert_utf16_to_utf8() {
        let src: Vec<u16> = "abc\u{1F4A9}".encode_utf16().collect();
        let mut dst = [0u8; 32];
        let (read, written) = convert_utf16_to_utf8_partial_inner(&src, &mut dst);
        assert_eq!(read, src.len());
        assert_eq!(&dst[..written], "abc\u{1F4A9}".as_bytes());
    }

    /// The eight scalars whose UTF-8 encodings sit on the length boundaries.
    const BOUNDARY_SCALARS: [char; 8] = [
        '\u{0}',
        '\u{7F}',
        '\u{80}',
        '\u{7FF}',
        '\u{800}',
        '\u{FFFF}',
        '\u{10000}',
        '\u{10FFFF}',
    ];

    /// One well-formed sequence of every length, in one source slice.
    const ONE_OF_EACH_LENGTH: &str = "a\u{80}\u{800}\u{10000}";

    fn scalar_bytes(scalar: char) -> ([u8; 4], usize) {
        let mut bytes = [0u8; 4];
        let len = scalar.encode_utf8(&mut bytes).len();
        (bytes, len)
    }

    fn decode_utf8_raw(src: &[u8], dst: &mut [u8], last: bool) -> (DecoderResult, usize, usize) {
        Utf8Decoder::new_inner().decode_to_utf8_raw(src, dst, last)
    }

    fn decode_utf16_raw(src: &[u8], dst: &mut [u16], last: bool) -> (DecoderResult, usize, usize) {
        Utf8Decoder::new_inner().decode_to_utf16_raw(src, dst, last)
    }

    /// Decodes with the replacing [`Decoder`] handle over the UTF-8 encoding.
    fn decode_utf8_replacing(src: &[u8], dst: &mut [u8]) -> (CoderResult, usize, usize, bool) {
        UTF_8.new_decoder().decode_to_utf8(src, dst, true)
    }

    fn encode_text_replacing(text: &str, dst: &mut [u8]) -> (CoderResult, usize, usize, bool) {
        UTF_8
            .new_encoder()
            .encode_from_utf8(text, dst, true)
    }

    // -- utf8_valid_up_to: well-formed input ----------------------------------

    #[test]
    fn test_valid_up_to_ascii_runs_at_stride_boundaries() {
        let buf = [b'a'; 64];
        for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64] {
            assert_eq!(utf8_valid_up_to(&buf[..len]), len, "ascii len {len}");
        }
    }

    #[test]
    fn test_valid_up_to_boundary_scalars() {
        for scalar in BOUNDARY_SCALARS {
            let (bytes, len) = scalar_bytes(scalar);
            assert_eq!(utf8_valid_up_to(&bytes[..len]), len, "scalar {scalar:?}");
        }
    }

    #[test]
    fn test_valid_up_to_every_assigned_scalar_alone() {
        let mut bytes = [0u8; 4];
        for cp in 0u32..0x11_0000 {
            let Some(scalar) = char::from_u32(cp) else {
                continue;
            };
            let len = scalar.encode_utf8(&mut bytes).len();
            assert_eq!(utf8_valid_up_to(&bytes[..len]), len, "cp {cp:04X}");
        }
    }

    #[test]
    fn test_valid_up_to_every_assigned_scalar_after_ascii_prefix() {
        let mut bytes = [0u8; 4];
        let mut buf = [0u8; 64];
        for prefix in 0usize..=20 {
            buf[..prefix].fill(b'x');
            for cp in [0x1, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x1_0000, 0x10_FFFF] {
                let scalar = char::from_u32(cp).unwrap();
                let len = scalar.encode_utf8(&mut bytes).len();
                buf[prefix..prefix + len].copy_from_slice(&bytes[..len]);
                let total = prefix + len;
                assert_eq!(
                    utf8_valid_up_to(&buf[..total]),
                    total,
                    "prefix {prefix}, cp {cp:04X}"
                );
            }
        }
    }

    #[test]
    fn test_valid_up_to_two_byte_second_byte_range_is_contiguous() {
        for lead in 0xC2u8..=0xDF {
            for second in 0x80u8..=0xBF {
                assert_eq!(utf8_valid_up_to(&[lead, second]), 2, "{lead:02X} {second:02X}");
            }
        }
    }

    #[test]
    fn test_valid_up_to_trailing_partial_two_byte_sequence() {
        assert_eq!(utf8_valid_up_to(b"\xC2"), 0);
        assert_eq!(utf8_valid_up_to(b"\xDF"), 0);
        assert_eq!(utf8_valid_up_to(b"ABC\xC2"), 3);
        assert_eq!(utf8_valid_up_to(b"ABC\xDF\xB2"), 5);
    }

    #[test]
    fn test_valid_up_to_trailing_partial_three_byte_sequence() {
        assert_eq!(utf8_valid_up_to(b"\xE1"), 0);
        assert_eq!(utf8_valid_up_to(b"\xE1\x80"), 0);
        assert_eq!(utf8_valid_up_to(b"xy\xE1\x80"), 2);
        assert_eq!(utf8_valid_up_to(b"xyz\xEF\xBF"), 3);
    }

    #[test]
    fn test_valid_up_to_trailing_partial_four_byte_sequence() {
        assert_eq!(utf8_valid_up_to(b"\xF0"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x90"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x90\x80"), 0);
        assert_eq!(utf8_valid_up_to(b"ab\xF4\x8F\xBF"), 2);
    }

    #[test]
    fn test_valid_up_to_three_byte_second_byte_boundaries() {
        // E0 rejects the overlong 80..=9F second bytes; E1..=EC and EE..=EF accept
        // the whole continuation range.
        for second in 0x80u8..0xA0 {
            assert_eq!(utf8_valid_up_to(&[0xE0, second, 0x80]), 0, "E0 {second:02X}");
        }
        for second in 0xA0u8..=0xBF {
            assert_eq!(utf8_valid_up_to(&[0xE0, second, 0x80]), 3, "E0 {second:02X}");
        }
        for second in 0x80u8..=0xBF {
            assert_eq!(utf8_valid_up_to(&[0xE2, second, 0xBF]), 3, "E2 {second:02X}");
        }
    }

    #[test]
    fn test_valid_up_to_four_byte_second_byte_boundaries() {
        // F0 rejects the overlong 80..=8F second bytes; F4 caps at 8F (U+10FFFF).
        for second in 0x80u8..0x90 {
            assert_eq!(
                utf8_valid_up_to(&[0xF0, second, 0x80, 0x80]),
                0,
                "F0 {second:02X}"
            );
        }
        for second in 0x90u8..=0xBF {
            assert_eq!(
                utf8_valid_up_to(&[0xF0, second, 0x80, 0x80]),
                4,
                "F0 {second:02X}"
            );
        }
        for second in 0x80u8..=0x8F {
            assert_eq!(
                utf8_valid_up_to(&[0xF4, second, 0xBF, 0xBF]),
                4,
                "F4 {second:02X}"
            );
        }
        for second in 0x90u8..=0xBF {
            assert_eq!(
                utf8_valid_up_to(&[0xF4, second, 0x80, 0x80]),
                0,
                "F4 {second:02X}"
            );
        }
        for second in 0x90u8..=0xBF {
            assert_eq!(
                utf8_valid_up_to(&[0xF1, second, 0x80, 0x80]),
                4,
                "F1 {second:02X}"
            );
        }
    }

    #[test]
    fn test_valid_up_to_rejects_all_invalid_leads() {
        for lead in 0x80u8..=0xFF {
            if matches!(lead, 0xC2..=0xF4) {
                continue;
            }
            assert_eq!(utf8_valid_up_to(&[lead, 0x80, 0x80, 0x80]), 0, "lead {lead:02X}");
        }
    }

    #[test]
    fn test_valid_up_to_rejects_overlong_forms() {
        assert_eq!(utf8_valid_up_to(b"\xC0\x80"), 0);
        assert_eq!(utf8_valid_up_to(b"\xC0\xAF"), 0);
        assert_eq!(utf8_valid_up_to(b"\xC1\xBF"), 0);
        assert_eq!(utf8_valid_up_to(b"\xE0\x80\x80"), 0);
        assert_eq!(utf8_valid_up_to(b"\xE0\x9F\xBF"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x80\x80\x80"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x8F\xBF\xBF"), 0);
        assert_eq!(utf8_valid_up_to(b"ab\xE0\x80\x80"), 2);
    }

    #[test]
    fn test_valid_up_to_rejects_surrogate_half_encodings() {
        for second in 0xA0u8..=0xBF {
            assert_eq!(utf8_valid_up_to(&[0xED, second, 0x80]), 0, "ED {second:02X} 80");
        }
        for second in 0x80u8..0xA0 {
            assert_eq!(utf8_valid_up_to(&[0xED, second, 0xBF]), 3, "ED {second:02X} BF");
        }
        assert_eq!(utf8_valid_up_to(b"\xED\xBF\xBF"), 0);
        assert_eq!(utf8_valid_up_to(b"\xED\xA0\x80\xC3\xA9"), 0);
    }

    #[test]
    fn test_valid_up_to_rejects_bad_continuation_bytes() {
        assert_eq!(utf8_valid_up_to(b"\xC3\x28"), 0);
        assert_eq!(utf8_valid_up_to(b"\xC3\xC3"), 0);
        assert_eq!(utf8_valid_up_to(b"\xC3\xFF"), 0);
        assert_eq!(utf8_valid_up_to(b"\xE1\x80\x28"), 0);
        assert_eq!(utf8_valid_up_to(b"\xE1\x80\xC0"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x9F\x9F\x28"), 0);
        assert_eq!(utf8_valid_up_to(b"\xF0\x9F\x9F\xC0"), 0);
        // An incomplete trailing sequence stops the scan at its lead byte.
        assert_eq!(utf8_valid_up_to(b"ok \xE2\x98\x28"), 3);
        assert_eq!(utf8_valid_up_to(b"\xE2\x98"), 0);
        assert_eq!(utf8_valid_up_to(b"\xE2\x98\x83"), 3);
    }

    #[test]
    fn test_valid_up_to_reports_first_maximal_subpart_offset() {
        assert_eq!(utf8_valid_up_to(b"AB\xC3\xA9\xFF"), 4);
        assert_eq!(utf8_valid_up_to(b"\x80\x80"), 0);
        assert_eq!(utf8_valid_up_to(b"ok\xE2\x98\x83\xF0\x9F\x92\xA9\x21\xF5"), 10);
        let mut buf = [b'a'; 16];
        assert_eq!(utf8_valid_up_to(&buf), 16);
        buf[15] = 0x80;
        assert_eq!(utf8_valid_up_to(&buf), 15);
    }

    #[test]
    fn test_valid_up_to_second_byte_bad_at_stride_boundary() {
        let mut buf = [b'a'; 20];
        buf[15] = 0xC3;
        buf[16] = 0x28;
        assert_eq!(utf8_valid_up_to(&buf), 15);
        let mut ok = [b'a'; 20];
        ok[15] = 0xC3;
        ok[16] = 0xA9;
        // The whole buffer is well formed: the trailing ASCII keeps going.
        assert_eq!(utf8_valid_up_to(&ok), 20);
        assert_eq!(utf8_valid_up_to(&ok[..16]), 15);
        assert_eq!(utf8_valid_up_to(&ok[..17]), 17);
    }

    #[test]
    fn test_valid_up_to_third_byte_bad_at_stride_boundary() {
        let mut buf = [b'a'; 20];
        buf[15] = 0xE2;
        buf[16] = 0x98;
        buf[17] = 0x28;
        assert_eq!(utf8_valid_up_to(&buf), 15);
        buf[17] = 0x83;
        assert_eq!(utf8_valid_up_to(&buf), 20);
        assert_eq!(utf8_valid_up_to(&buf[..17]), 15);
        assert_eq!(utf8_valid_up_to(&buf[..18]), 18);
    }

    #[test]
    fn test_valid_up_to_fourth_byte_bad_at_stride_boundary() {
        let mut buf = [b'a'; 20];
        buf[15] = 0xF0;
        buf[16] = 0x9F;
        buf[17] = 0x92;
        buf[18] = 0x28;
        assert_eq!(utf8_valid_up_to(&buf), 15);
        buf[18] = 0xA9;
        assert_eq!(utf8_valid_up_to(&buf), 20);
        assert_eq!(utf8_valid_up_to(&buf[..18]), 15);
        assert_eq!(utf8_valid_up_to(&buf[..19]), 19);
    }

    #[test]
    fn test_valid_up_to_multi_stride_ascii_then_multibyte() {
        let mut buf = [b'a'; 33];
        buf[32] = 0xE2;
        assert_eq!(utf8_valid_up_to(&buf[..32]), 32);
        assert_eq!(utf8_valid_up_to(&buf), 32);
        let mut full = [b'a'; 33];
        full[32] = 0x61;
        assert_eq!(utf8_valid_up_to(&full), 33);
        let mut tail = [0u8; 4];
        tail[..3].copy_from_slice(b"\xE2\x98\x83");
        assert_eq!(utf8_valid_up_to(&tail[..3]), 3);
        assert_eq!(utf8_valid_up_to(&tail), 4);
        tail[..3].copy_from_slice(b"\xE2\x98\x28");
        assert_eq!(utf8_valid_up_to(&tail), 0);
    }

    // -- decode_to_utf8_raw: well-formed --------------------------------------

    #[test]
    fn test_decode_utf8_boundary_scalars_round_trip() {
        let mut dst = [0u8; 8];
        for scalar in BOUNDARY_SCALARS {
            let (bytes, len) = scalar_bytes(scalar);
            let (result, read, written) = decode_utf8_raw(&bytes[..len], &mut dst, true);
            assert_eq!(result, DecoderResult::InputEmpty, "scalar {scalar:?}");
            assert_eq!((read, written), (len, len), "scalar {scalar:?}");
            assert_eq!(&dst[..written], &bytes[..len], "scalar {scalar:?}");
        }
    }

    #[test]
    fn test_decode_utf8_one_of_each_sequence_length() {
        let src = ONE_OF_EACH_LENGTH.as_bytes();
        let mut dst = [0u8; 16];
        let (result, read, written) = decode_utf8_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (src.len(), src.len()));
        assert_eq!(&dst[..written], src);
    }

    #[test]
    fn test_decode_utf8_long_ascii_runs_use_the_fast_loop() {
        let mut dst = [0u8; 64];
        for len in [1usize, 15, 16, 17, 31, 32, 33, 48, 63] {
            let src = [b'a'; 64];
            let (result, read, written) = decode_utf8_raw(&src[..len], &mut dst, true);
            assert_eq!(result, DecoderResult::InputEmpty, "ascii len {len}");
            assert_eq!((read, written), (len, len), "ascii len {len}");
            assert_eq!(&dst[..written], &src[..len], "ascii len {len}");
        }
    }

    #[test]
    fn test_decode_utf8_multibyte_after_stride_of_ascii() {
        let src = "aaaaaaaaaaaaaaaa\u{800}";
        assert_eq!(src.len(), 19);
        let mut dst = [0u8; 32];
        let (result, read, written) = decode_utf8_raw(src.as_bytes(), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (19, 19));
        assert_eq!(&dst[..written], src.as_bytes());
    }

    #[test]
    fn test_decode_utf8_last_false_with_complete_input() {
        let mut dst = [0u8; 16];
        let (result, read, written) = decode_utf8_raw(ONE_OF_EACH_LENGTH.as_bytes(), &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (10, 10));
    }

    #[test]
    fn test_decode_utf8_empty_input_last_true() {
        let mut dst = [0u8; 16];
        let (result, read, written) = decode_utf8_raw(b"", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn test_decode_utf8_empty_input_leaves_state_neutral() {
        let mut decoder = Utf8Decoder::new_inner();
        assert!(decoder.in_neutral_state());
        let (result, _read, _written) = decoder.decode_to_utf8_raw(b"a", &mut [0u8; 4], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert!(decoder.in_neutral_state());
    }

    #[test]
    fn test_decode_utf8_split_one_byte_at_a_time_matches_whole() {
        let src = ONE_OF_EACH_LENGTH.as_bytes();
        let mut whole = [0u8; 32];
        let (result, read, written) = decode_utf8_raw(src, &mut whole, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (src.len(), src.len()));

        let mut split = [0u8; 32];
        let mut decoder = Utf8Decoder::new_inner();
        let mut out_pos = 0usize;
        for (i, byte) in src.iter().enumerate() {
            let last = i + 1 == src.len();
            let (result, read, written) =
                decoder.decode_to_utf8_raw(core::slice::from_ref(byte), &mut split[out_pos..], last);
            assert_eq!(result, DecoderResult::InputEmpty, "byte offset {i}");
            assert_eq!(read, 1, "byte offset {i}");
            out_pos += written;
        }
        assert_eq!(out_pos, src.len());
        assert_eq!(&split[..out_pos], &whole[..written]);
    }

    #[test]
    fn test_decode_utf8_two_byte_sequence_spanning_two_calls() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 8];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\xC3", &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 0));
        assert!(!decoder.in_neutral_state());
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\xA9", &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 2));
        assert!(decoder.in_neutral_state());
        assert_eq!(&dst[..2], "é".as_bytes());
    }

    #[test]
    fn test_decode_utf8_astral_sequence_spanning_four_calls() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 8];
        for (i, byte) in [0xF0u8, 0x9F, 0x92, 0xA9].iter().enumerate() {
            let (result, read, written) =
                decoder.decode_to_utf8_raw(core::slice::from_ref(byte), &mut dst, i == 3);
            assert_eq!(result, DecoderResult::InputEmpty, "byte {i}");
            assert_eq!(read, 1, "byte {i}");
            let expected_written = if i == 3 { 4 } else { 0 };
            assert_eq!(written, expected_written, "byte {i}");
        }
        assert_eq!(&dst[..4], "\u{1F4A9}".as_bytes());
        assert!(decoder.in_neutral_state());
    }

    #[test]
    fn test_decode_utf8_pending_sequence_flushes_when_last() {
        for (pending, bad) in [(&b"\xE2\x98"[..], 2u8), (b"\xF0\x9F\x92", 3), (b"\xC3", 1)] {
            let mut decoder = Utf8Decoder::new_inner();
            let mut dst = [0u8; 16];
            let (result, _read, _written) = decoder.decode_to_utf8_raw(pending, &mut dst, false);
            assert_eq!(result, DecoderResult::InputEmpty, "pending {pending:?}");
            let (result, read, written) = decoder.decode_to_utf8_raw(b"", &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(bad, 0), "pending {pending:?}");
            assert_eq!((read, written), (0, 0), "pending {pending:?}");
            assert!(decoder.in_neutral_state(), "pending {pending:?}");
        }
    }

    #[test]
    fn test_decode_utf8_truncated_pending_flush_in_one_call() {
        let mut dst = [0u8; 16];
        let (result, read, written) = decode_utf8_raw(b"\xE2\x98", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (2, 0));

        let mut dst = [0u8; 16];
        let (result, read, written) = decode_utf8_raw(b"abc\xF0\x9F", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (5, 3));
        assert_eq!(&dst[..3], b"abc");
    }

    #[test]
    fn test_decode_utf8_resumes_after_flushed_pending_sequence() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decoder.decode_to_utf8_raw(b"\xE2", &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        let (result, _read, _written) = decoder.decode_to_utf8_raw(b"", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        let (result, read, written) = decoder.decode_to_utf8_raw("é".as_bytes(), &mut dst[3..], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
    }

    // -- decode_to_utf8_raw: malformed ----------------------------------------

    #[test]
    fn test_decode_utf8_unexpected_continuation_bytes() {
        for byte in [0x80u8, 0x81, 0x9F, 0xA0, 0xBE, 0xBF, 0xC0, 0xC1] {
            let mut dst = [0u8; 16];
            let (result, read, written) = decode_utf8_raw(&[byte], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "byte {byte:02X}");
            assert_eq!((read, written), (1, 0), "byte {byte:02X}");
        }
    }

    #[test]
    fn test_decode_utf8_invalid_lead_bytes() {
        for byte in [0xF5u8, 0xF8, 0xFC, 0xFD, 0xFE, 0xFF] {
            let mut dst = [0u8; 16];
            let (result, read, written) = decode_utf8_raw(&[byte, 0x80, 0x80, 0x80], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "byte {byte:02X}");
            assert_eq!(read, 1, "byte {byte:02X}");
            assert_eq!(written, 0, "byte {byte:02X}");
        }
    }

    #[test]
    fn test_decode_utf8_overlong_two_byte_forms() {
        for second in [0x80u8, 0xA0, 0xBF] {
            let mut dst = [0u8; 16];
            for lead in [0xC0u8, 0xC1] {
                let (result, _read, _written) = decode_utf8_raw(&[lead, second], &mut dst, true);
                assert_eq!(result, DecoderResult::Malformed(1, 0), "{lead:02X} {second:02X}");
            }
        }
    }

    #[test]
    fn test_decode_utf8_continuation_out_of_range_for_second_byte() {
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decode_utf8_raw(b"\xC3\x28", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decode_utf8_raw(b"\xDF\x41", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
    }

    #[test]
    fn test_decode_utf8_second_byte_out_of_range_for_three_byte_lead() {
        // E0 only accepts A0..=BF, ED only 80..=9F.
        for second in [0x80u8, 0x8F, 0x90, 0x9F] {
            let mut dst = [0u8; 16];
            let (result, _read, _written) = decode_utf8_raw(&[0xE0, second, 0x80], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "E0 {second:02X}");
        }
        for second in [0xA0u8, 0xB0, 0xBF] {
            let mut dst = [0u8; 16];
            let (result, _read, _written) = decode_utf8_raw(&[0xED, second, 0x80], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "ED {second:02X}");
        }
    }

    #[test]
    fn test_decode_utf8_second_byte_out_of_range_for_four_byte_lead() {
        // F0 requires 90..=BF, F4 requires 80..=8F.
        for second in [0x80u8, 0x88, 0x8F] {
            let mut dst = [0u8; 16];
            let (result, _read, _written) = decode_utf8_raw(&[0xF0, second, 0x80, 0x80], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "F0 {second:02X}");
        }
        for second in [0x90u8, 0xA0, 0xBF] {
            let mut dst = [0u8; 16];
            let (result, _read, _written) = decode_utf8_raw(&[0xF4, second, 0x80, 0x80], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "F4 {second:02X}");
        }
    }

    #[test]
    fn test_decode_utf8_third_byte_of_four_is_out_of_range() {
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decode_utf8_raw(b"\xF0\x9F\x28\x80", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decode_utf8_raw(b"\xF0\x9F\x9F\x28", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(3, 0));
    }

    #[test]
    fn test_decode_utf8_third_byte_of_three_is_out_of_range() {
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decode_utf8_raw(b"\xE2\x98\x28", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        let mut dst = [0u8; 16];
        let (result, _read, _written) = decode_utf8_raw(b"\xE2\x98\xC0", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
    }

    #[test]
    fn test_decode_utf8_surrogate_half_encodings_are_malformed() {
        for second in [0xA0u8, 0xAF, 0xBF] {
            let mut dst = [0u8; 16];
            let (result, _read, _written) = decode_utf8_raw(&[0xED, second, 0x80], &mut dst, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "ED {second:02X} 80");
            assert_eq!(dst, [0u8; 16], "ED {second:02X} 80");
        }
    }

    #[test]
    fn test_decode_utf8_malformed_consumes_nothing_of_the_destination() {
        let mut dst = [b'Z'; 16];
        let (result, _read, written) = decode_utf8_raw(b"\xE0\x80\x80", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!(written, 0);
        assert_eq!(dst, [b'Z'; 16]);
    }

    #[test]
    fn test_decode_utf8_valid_prefix_is_emitted_before_malformed() {
        let src = b"abc\xE0\x80";
        let mut dst = [0u8; 16];
        let (result, _read, written) = decode_utf8_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!(&dst[..written], b"abc");
    }

    #[test]
    fn test_decode_utf8_two_maximal_subparts_in_sequence() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\xFF", &mut dst, false);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
    }

    #[test]
    fn test_decode_utf8_last_false_does_not_flush_pending() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\xE2\x98", &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 0));
        let (result, read, written) = decoder.decode_to_utf8_raw(&[0x83], &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 3));
        assert_eq!(&dst[..3], "☃".as_bytes());
    }

    // -- decode_to_utf8_raw: destination space ------------------------------

    #[test]
    fn test_decode_utf8_output_full_needs_room_for_the_sequence() {
        // a two-byte sequence does not fit one free byte: nothing is consumed
        let mut dst = [0u8; 1];
        let (result, read, written) = decode_utf8_raw("é".as_bytes(), &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));

        // three free bytes admit it and the copy stays byte-identical
        let mut dst = [0u8; 3];
        let (result, read, written) = decode_utf8_raw("é".as_bytes(), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
        assert_eq!(&dst[..2], "é".as_bytes());

        // a four-byte sequence needs all four bytes of room
        let mut dst = [0u8; 3];
        let (result, read, written) = decode_utf8_raw("🙂".as_bytes(), &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn test_decode_utf8_fits_exactly_when_dst_matches_src() {
        let src = "é☃🙂".as_bytes();
        let mut dst = [0u8; 9];
        assert_eq!(src.len(), 9);
        let (result, read, written) = decode_utf8_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (9, 9));
        assert_eq!(&dst, src);
    }

    #[test]
    fn test_decode_utf8_refill_progress_and_final_output() {
        let src = "aé☃🙂".as_bytes();
        let mut out = [0u8; 32];
        let mut decoder = Utf8Decoder::new_inner();
        let (mut total_read, mut total_written) = (0usize, 0usize);
        let mut calls = 0usize;
        let mut result = DecoderResult::InputEmpty;
        while total_read < src.len() {
            calls += 1;
            assert!(calls < 10, "no forward progress at offset {total_read}");
            let (r, read, written) =
                decoder.decode_to_utf8_raw(&src[total_read..], &mut out[total_written..], true);
            assert!(read > 0 || written > 0, "stalled at offset {total_read}");
            result = r;
            total_read += read;
            total_written += written;
        }
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((total_read, total_written), (src.len(), src.len()));
        assert_eq!(&out[..total_written], src);
    }

    #[test]
    fn test_decode_utf8_ascii_fills_a_one_byte_destination() {
        let mut dst = [0u8; 1];
        let (result, read, written) = decode_utf8_raw(b"a", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[0], b'a');
    }

    #[test]
    fn test_decode_utf8_malformed_needs_four_free_bytes() {
        let mut dst = [0u8; 3];
        let (result, read, written) = decode_utf8_raw(b"\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
        let mut dst = [0u8; 4];
        let (result, read, written) = decode_utf8_raw(b"\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));
    }

    // -- decode_to_utf16_raw --------------------------------------------------

    #[test]
    fn test_decode_utf16_boundary_scalars() {
        let mut units = [0u16; 4];
        let mut dst = [0u16; 8];
        for scalar in BOUNDARY_SCALARS {
            let expected_len = scalar.encode_utf16(&mut units).len();
            let (bytes, len) = scalar_bytes(scalar);
            let (result, read, written) = decode_utf16_raw(&bytes[..len], &mut dst, true);
            assert_eq!(result, DecoderResult::InputEmpty, "scalar {scalar:?}");
            assert_eq!((read, written), (len, expected_len), "scalar {scalar:?}");
            assert_eq!(&dst[..written], &units[..expected_len], "scalar {scalar:?}");
        }
    }

    #[test]
    fn test_decode_utf16_max_one_unit_for_bmp_four_for_astral() {
        let mut dst = [0u16; 8];
        let src = "aé☃".as_bytes();
        let (result, read, written) = decode_utf16_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (src.len(), 3));
        assert_eq!(&dst[..3], &[0x61, 0xE9, 0x2603]);

        let src = "\u{10FFFF}".as_bytes();
        let (result, read, written) = decode_utf16_raw(src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..2], &[0xDBFF, 0xDFFF]);
    }

    #[test]
    fn test_decode_utf16_long_ascii_run() {
        let src = [b'b'; 40];
        let mut dst = [0u16; 48];
        let (result, read, written) = decode_utf16_raw(&src, &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (40, 40));
        assert!(dst[..40].iter().all(|&u| u == 0x62));
    }

    #[test]
    fn a_broken_sequence_reports_the_reprocessed_byte_position() {
        // WHATWG puts the non-continuation byte back so the caller sees it
        // again; the raw step must not count it as consumed
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\xF4!", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (1, 0));

        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 16];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"ok\xF4!", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (3, 2));
    }

    #[test]
    fn replacing_decode_reprocesses_the_byte_after_a_broken_sequence() {
        let mut dst = [0u8; 32];
        let (result, read, written, had_errors) = UTF_8
            .new_decoder()
            .decode_to_utf8(b"ok\xF4!", &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!((read, written), (4, 6));
        assert_eq!(&dst[..written], "ok\u{FFFD}!".as_bytes());
        assert!(had_errors);
    }
}
