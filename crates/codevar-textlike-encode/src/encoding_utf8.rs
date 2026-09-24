use crate::encoding::{
    CoderResult, DecoderResult, EncoderResult, UTF_8, convert_utf16_to_utf8_partial,
};
use crate::encoding::{VariantDecoder, VariantEncoder};
use crate::encoding_ascii::{ascii_to_basic_latin, basic_latin_to_ascii, validate_ascii};
use crate::encoding_handles::{ByteSource, Space, Utf8Destination, Utf16Destination};

#[repr(align(64))]
pub struct Utf8Data {
    pub table: [u8; 384],
}

pub static UTF8_DATA: Utf8Data = Utf8Data {
    table: [
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 148, 148, 148,
        148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32, 8, 8,
        64, 8, 8, 8, 128, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
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
                                *(UTF8_DATA.table.get_unchecked(byte as usize + 0x80))
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
                            *(UTF8_DATA.table.get_unchecked(byte as usize + 0x80))
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
                    & unsafe { *(UTF8_DATA.table.get_unchecked(byte as usize + 0x80)) })
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

pub fn convert_utf8_to_utf16_up_to_invalid(
    src: &[u8],
    dst: &mut [u16],
) -> (usize, usize) {
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
                                *(UTF8_DATA.table.get_unchecked(byte as usize + 0x80))
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
                            *(UTF8_DATA.table.get_unchecked(byte as usize + 0x80))
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
                    *(dst.get_unchecked_mut(written + 1)) =
                        (0xDC00 + (point & 0x3FF)) as u16;
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
                dst[written] =
                    ((u16::from(byte) & 0x1F) << 6) | (u16::from(second) & 0x3F);
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
                    & unsafe { *(UTF8_DATA.table.get_unchecked(byte as usize + 0x80)) })
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

pub fn convert_utf16_to_utf8_partial_inner(
    src: &[u16],
    dst: &mut [u8],
) -> (usize, usize) {
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
                        *(dst.get_unchecked_mut(written)) =
                            ((unit & 0xFC0) >> 6) as u8 | 0x80u8;
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
                            *(dst.get_unchecked_mut(written)) =
                                (astral >> 18) as u8 | 0xF0u8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) =
                                ((astral & 0x3F000u32) >> 12) as u8 | 0x80u8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) =
                                ((astral & 0xFC0u32) >> 6) as u8 | 0x80u8;
                            written += 1;
                            *(dst.get_unchecked_mut(written)) =
                                (astral & 0x3F) as u8 | 0x80u8;
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

    pub fn max_utf8_buffer_length_without_replacement(
        &self,
        byte_length: usize,
    ) -> Option<usize> {
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
                        return (
                            DecoderResult::OutputFull,
                            source_handle.consumed(),
                            dst_written,
                        );
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
                            return (
                                DecoderResult::Malformed(bad_bytes, 0),
                                unread_handle.unread(),
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
                            destination_handle
                                .write_bmp_excl_ascii(self.code_point as u16);
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
                        return (
                            DecoderResult::OutputFull,
                            source_handle.consumed(),
                            dst_written,
                        );
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
                            return (
                                DecoderResult::Malformed(bad_bytes, 0),
                                unread_handle.unread(),
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
                            destination_handle
                                .write_bmp_excl_ascii(self.code_point as u16);
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

    pub fn max_buffer_length_from_utf16_without_replacement(
        &self,
        u16_length: usize,
    ) -> Option<usize> {
        u16_length.checked_mul(3)
    }

    pub fn max_buffer_length_from_utf8_without_replacement(
        &self,
        byte_length: usize,
    ) -> Option<usize> {
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
/// use codevar_textlike_encode::encoding_utf8::encode_text;
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
        let (result, r, w, _) =
            encoder.encode_from_utf8(&text[read..], &mut buf[written..], true);
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
        let (result, read, written) =
            decoder.decode_to_utf8_raw(b"hello", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!(read, 5);
        assert_eq!(written, 5);
    }

    #[test]
    fn test_utf8_decode_malformed() {
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        // \xC3 is an incomplete lead; Z is not a continuation and is unread
        let (result, _read, _written) =
            decoder.decode_to_utf8_raw(b"a\xC3Z", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        // Incomplete sequence at EOF
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        let (result, _read, _written) =
            decoder.decode_to_utf8_raw(b"a\xC3", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        // Invalid lead byte
        let mut decoder = Utf8Decoder::new_inner();
        let mut dst = [0u8; 10];
        let (result, _read, _written) =
            decoder.decode_to_utf8_raw(b"\xFF", &mut dst, true);
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
        let (read, written) =
            convert_utf8_to_utf16_up_to_invalid(src.as_bytes(), &mut dst[..]);
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
}
