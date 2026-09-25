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

use crate::encoding::{DecoderResult, EncoderResult};
use crate::encoding_handles::{
    BigEndian, ByteSource, LittleEndian, Space, Utf8Destination, Utf16Destination,
};

#[derive(Debug, Clone)]
pub struct Utf16Decoder {
    lead_surrogate: u16,
    lead_byte: Option<u8>,
    be: bool,
    pending_bmp: bool,
}

impl Utf16Decoder {
    pub fn new(big_endian: bool) -> Utf16Decoder {
        Utf16Decoder {
            lead_surrogate: 0,
            lead_byte: None,
            be: big_endian,
            pending_bmp: false,
        }
    }

    pub fn additional_from_state(&self) -> usize {
        1 + if self.lead_byte.is_some() { 1 } else { 0 } + if self.lead_surrogate == 0 { 0 } else { 2 }
    }

    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        checked_add(
            1,
            checked_div(byte_length.checked_add(self.additional_from_state()), 2),
        )
    }

    pub fn max_utf8_buffer_length_without_replacement(&self, byte_length: usize) -> Option<usize> {
        checked_add(
            1,
            checked_mul(
                3,
                checked_div(byte_length.checked_add(self.additional_from_state()), 2),
            ),
        )
    }

    pub fn max_utf8_buffer_length(&self, byte_length: usize) -> Option<usize> {
        checked_add(
            1,
            checked_mul(
                3,
                checked_div(byte_length.checked_add(self.additional_from_state()), 2),
            ),
        )
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
            if self.pending_bmp {
                match dest.check_space_bmp() {
                    Space::Full(_) => return (DecoderResult::OutputFull, 0, 0),
                    Space::Available(dh) => {
                        dh.write_bmp(self.lead_surrogate);
                        self.pending_bmp = false;
                        self.lead_surrogate = 0;
                    }
                }
            }
            if self.lead_byte.is_none() && self.lead_surrogate == 0 {
                let result = if self.be {
                    dest.copy_utf16_from::<BigEndian>(&mut source)
                } else {
                    dest.copy_utf16_from::<LittleEndian>(&mut source)
                };
                if let Some((read, written)) = result {
                    return (DecoderResult::Malformed(2, 0), read, written);
                }
            }
            match source.check_available() {
                Space::Full(src_consumed) => {
                    if last && (self.lead_surrogate != 0 || self.lead_byte.is_some()) {
                        match dest.check_space_bmp() {
                            Space::Full(_) => {
                                return (DecoderResult::OutputFull, 0, 0);
                            }
                            Space::Available(_) => {
                                if self.lead_surrogate != 0 {
                                    self.lead_surrogate = 0;
                                    match self.lead_byte {
                                        None => {
                                            return (
                                                DecoderResult::Malformed(2, 0),
                                                src_consumed,
                                                dest.written(),
                                            );
                                        }
                                        Some(_) => {
                                            self.lead_byte = None;
                                            return (
                                                DecoderResult::Malformed(3, 0),
                                                src_consumed,
                                                dest.written(),
                                            );
                                        }
                                    }
                                }
                                debug_assert!(self.lead_byte.is_some());
                                self.lead_byte = None;
                                return (DecoderResult::Malformed(1, 0), src_consumed, dest.written());
                            }
                        }
                    }
                    return (DecoderResult::InputEmpty, src_consumed, dest.written());
                }
                Space::Available(source_handle) => match dest.check_space_astral() {
                    Space::Full(dst_written) => {
                        return (DecoderResult::OutputFull, source_handle.consumed(), dst_written);
                    }
                    Space::Available(destination_handle) => {
                        let (b, unread_handle) = source_handle.read();
                        match self.lead_byte {
                            None => {
                                self.lead_byte = Some(b);
                                continue;
                            }
                            Some(lead) => {
                                self.lead_byte = None;
                                let code_unit = if self.be {
                                    u16::from(lead) << 8 | u16::from(b)
                                } else {
                                    u16::from(b) << 8 | u16::from(lead)
                                };
                                let high_bits = code_unit & 0xFC00u16;
                                if high_bits == 0xD800u16 {
                                    if self.lead_surrogate != 0 {
                                        self.lead_surrogate = code_unit;
                                        return (
                                            DecoderResult::Malformed(2, 2),
                                            unread_handle.consumed(),
                                            destination_handle.written(),
                                        );
                                    }
                                    self.lead_surrogate = code_unit;
                                    continue;
                                }
                                if high_bits == 0xDC00u16 {
                                    if self.lead_surrogate == 0 {
                                        return (
                                            DecoderResult::Malformed(2, 0),
                                            unread_handle.consumed(),
                                            destination_handle.written(),
                                        );
                                    }
                                    destination_handle.write_surrogate_pair(self.lead_surrogate, code_unit);
                                    self.lead_surrogate = 0;
                                    continue;
                                }
                                if self.lead_surrogate != 0 {
                                    self.lead_surrogate = code_unit;
                                    self.pending_bmp = true;
                                    return (
                                        DecoderResult::Malformed(2, 2),
                                        unread_handle.consumed(),
                                        destination_handle.written(),
                                    );
                                }
                                destination_handle.write_bmp(code_unit);
                                continue;
                            }
                        }
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
            if self.pending_bmp {
                match dest.check_space_bmp() {
                    Space::Full(_) => return (DecoderResult::OutputFull, 0, 0),
                    Space::Available(dh) => {
                        dh.write_bmp(self.lead_surrogate);
                        self.pending_bmp = false;
                        self.lead_surrogate = 0;
                    }
                }
            }
            if self.lead_byte.is_none() && self.lead_surrogate == 0 {
                let result = if self.be {
                    dest.copy_utf16_from::<BigEndian>(&mut source)
                } else {
                    dest.copy_utf16_from::<LittleEndian>(&mut source)
                };
                if let Some((read, written)) = result {
                    return (DecoderResult::Malformed(2, 0), read, written);
                }
            }
            match source.check_available() {
                Space::Full(src_consumed) => {
                    if last && (self.lead_surrogate != 0 || self.lead_byte.is_some()) {
                        return match dest.check_space_bmp() {
                            Space::Full(_) => (DecoderResult::OutputFull, 0, 0),
                            Space::Available(_) => {
                                if self.lead_surrogate != 0 {
                                    self.lead_surrogate = 0;
                                    return match self.lead_byte {
                                        None => {
                                            (DecoderResult::Malformed(2, 0), src_consumed, dest.written())
                                        }
                                        Some(_) => {
                                            self.lead_byte = None;
                                            (DecoderResult::Malformed(3, 0), src_consumed, dest.written())
                                        }
                                    };
                                }
                                debug_assert!(self.lead_byte.is_some());
                                self.lead_byte = None;
                                (DecoderResult::Malformed(1, 0), src_consumed, dest.written())
                            }
                        };
                    }
                    return (DecoderResult::InputEmpty, src_consumed, dest.written());
                }
                Space::Available(source_handle) => match dest.check_space_astral() {
                    Space::Full(dst_written) => {
                        return (DecoderResult::OutputFull, source_handle.consumed(), dst_written);
                    }
                    Space::Available(destination_handle) => {
                        let (b, unread_handle) = source_handle.read();
                        match self.lead_byte {
                            None => {
                                self.lead_byte = Some(b);
                                continue;
                            }
                            Some(lead) => {
                                self.lead_byte = None;
                                let code_unit = if self.be {
                                    u16::from(lead) << 8 | u16::from(b)
                                } else {
                                    u16::from(b) << 8 | u16::from(lead)
                                };
                                let high_bits = code_unit & 0xFC00u16;
                                if high_bits == 0xD800u16 {
                                    if self.lead_surrogate != 0 {
                                        self.lead_surrogate = code_unit;
                                        return (
                                            DecoderResult::Malformed(2, 2),
                                            unread_handle.consumed(),
                                            destination_handle.written(),
                                        );
                                    }
                                    self.lead_surrogate = code_unit;
                                    continue;
                                }
                                if high_bits == 0xDC00u16 {
                                    if self.lead_surrogate == 0 {
                                        return (
                                            DecoderResult::Malformed(2, 0),
                                            unread_handle.consumed(),
                                            destination_handle.written(),
                                        );
                                    }
                                    destination_handle.write_surrogate_pair(self.lead_surrogate, code_unit);
                                    self.lead_surrogate = 0;
                                    continue;
                                }
                                if self.lead_surrogate != 0 {
                                    self.lead_surrogate = code_unit;
                                    self.pending_bmp = true;
                                    return (
                                        DecoderResult::Malformed(2, 2),
                                        unread_handle.consumed(),
                                        destination_handle.written(),
                                    );
                                }
                                destination_handle.write_bmp(code_unit);
                                continue;
                            }
                        }
                    }
                },
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Utf16Encoder {
    be: bool,
}

impl Utf16Encoder {
    pub fn new(big_endian: bool) -> Utf16Encoder {
        Utf16Encoder { be: big_endian }
    }

    pub fn max_buffer_length_from_utf16_without_replacement(&self, u16_length: usize) -> Option<usize> {
        u16_length.checked_mul(2)
    }

    pub fn max_buffer_length_from_utf8_without_replacement(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_mul(2)
    }

    pub fn encode_from_utf16_raw(
        &mut self,
        src: &[u16],
        dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let mut read = 0;
        let mut written = 0;
        while read < src.len() {
            if written + 2 > dst.len() {
                return (EncoderResult::OutputFull, read, written);
            }
            let unit = src[read];
            if self.be {
                dst[written] = (unit >> 8) as u8;
                dst[written + 1] = unit as u8;
            } else {
                dst[written] = unit as u8;
                dst[written + 1] = (unit >> 8) as u8;
            }
            written += 2;
            read += 1;
        }
        (EncoderResult::InputEmpty, read, written)
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
            let mut buf = [0u16; 2];
            let units = ch.encode_utf16(&mut buf);
            let unit_count = units.len();
            if written + unit_count * 2 > dst.len() {
                return (EncoderResult::OutputFull, i, written);
            }
            for &u in units.iter() {
                if self.be {
                    dst[written] = (u >> 8) as u8;
                    dst[written + 1] = u as u8;
                } else {
                    dst[written] = u as u8;
                    dst[written + 1] = (u >> 8) as u8;
                }
                written += 2;
            }
            read = i + ch.len_utf8();
        }
        (EncoderResult::InputEmpty, read, written)
    }
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
fn checked_div(opt: Option<usize>, num: usize) -> Option<usize> {
    if let Some(n) = opt {
        n.checked_div(num)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf16_decode_le() {
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 10];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x61\x00\x62\x00", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!(read, 4);
        assert_eq!(written, 2);
        assert_eq!(&dst[..written], &[0x0061, 0x0062]);
    }

    #[test]
    fn test_utf16_decode_be() {
        let mut decoder = Utf16Decoder::new(true);
        let mut dst = [0u16; 10];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x00\x61\x00\x62", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!(read, 4);
        assert_eq!(written, 2);
    }

    #[test]
    fn test_utf16_encode() {
        let mut encoder = Utf16Encoder::new(false);
        let src: &[u16] = &[0x00A9, 0x2603];
        let mut dst = [0u8; 8];
        let (result, read, written) = encoder.encode_from_utf16_raw(src, &mut dst, true);
        assert_eq!(result, EncoderResult::InputEmpty);
        assert_eq!(read, 2);
        assert_eq!(written, 4);
        assert_eq!(&dst[..4], &[0xA9, 0x00, 0x03, 0x26]);
    }
}
