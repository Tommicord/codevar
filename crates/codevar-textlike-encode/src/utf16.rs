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
use crate::handles::{BigEndian, ByteSource, LittleEndian, Space, Utf8Destination, Utf16Destination};

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
    use crate::encoding::{CoderResult, UTF_16BE, UTF_16LE};

    // -- tests inherited from the original module ------------------------------

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

    // -- helpers --------------------------------------------------------------

    /// Code units of `text`, exactly as a UTF-16 producer would emit them.
    fn units(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    /// The `text` scalars as wire bytes in the requested byte order.
    fn le(text: &str) -> Vec<u8> {
        to_bytes(&units(text), false)
    }

    fn be(text: &str) -> Vec<u8> {
        to_bytes(&units(text), true)
    }

    /// Surrogate code units as wire bytes in the requested byte order.
    fn pair_bytes(high: u16, low: u16, big_endian: bool) -> Vec<u8> {
        to_bytes(&[high, low], big_endian)
    }

    fn to_bytes(units: &[u16], big_endian: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(units.len() * 2);
        for &unit in units {
            let (first, second) = if big_endian {
                ((unit >> 8) as u8, unit as u8)
            } else {
                (unit as u8, (unit >> 8) as u8)
            };
            out.push(first);
            out.push(second);
        }
        out
    }

    fn decode_le_to_utf16(src: &[u8], dst: &mut [u16], last: bool) -> (DecoderResult, usize, usize) {
        Utf16Decoder::new(false).decode_to_utf16_raw(src, dst, last)
    }

    fn decode_be_to_utf16(src: &[u8], dst: &mut [u16], last: bool) -> (DecoderResult, usize, usize) {
        Utf16Decoder::new(true).decode_to_utf16_raw(src, dst, last)
    }

    fn decode_le_to_utf8(src: &[u8], dst: &mut [u8], last: bool) -> (DecoderResult, usize, usize) {
        Utf16Decoder::new(false).decode_to_utf8_raw(src, dst, last)
    }

    fn decode_be_to_utf8(src: &[u8], dst: &mut [u8], last: bool) -> (DecoderResult, usize, usize) {
        Utf16Decoder::new(true).decode_to_utf8_raw(src, dst, last)
    }

    /// Encodes `text` with the raw encoder of `encode_be` and decodes the wire
    /// bytes back with the raw decoder of `decode_be`.
    ///
    /// The returned buffer is truncated to the bytes the decoder reported, so
    /// nothing past the reported write index is ever compared.
    fn encode_then_decode(text: &str, encode_be: bool, decode_be: bool) -> (DecoderResult, Vec<u8>) {
        let src = units(text);
        let mut wire = vec![0u8; src.len() * 2];
        let mut encoder = Utf16Encoder::new(encode_be);
        let (result, read, written) = encoder.encode_from_utf16_raw(&src, &mut wire, true);
        assert_eq!(result, EncoderResult::InputEmpty, "encoding {text:?}");
        assert_eq!((read, written), (src.len(), src.len() * 2), "encoding {text:?}");
        let mut out = vec![0u8; written * 2 + 8];
        let mut decoder = Utf16Decoder::new(decode_be);
        let (result, reread, wrote) = decoder.decode_to_utf8_raw(&wire, &mut out, true);
        assert_eq!(reread, wire.len(), "decoding {text:?}");
        out.truncate(wrote);
        (result, out)
    }

    /// Drains `src` through a fixed-width UTF-8 destination window, the way the
    /// replacing [`Decoder`](crate::encoding::Decoder) handle refills its output.
    fn refill_to_utf8(
        decoder: &mut Utf16Decoder,
        src: &[u8],
        window: usize,
        out: &mut [u8],
    ) -> (DecoderResult, usize, usize) {
        let mut read = 0usize;
        let mut written = 0usize;
        let mut result = DecoderResult::InputEmpty;
        for _ in 0..64 {
            if read == src.len() {
                break;
            }
            let (step, r, w) =
                decoder.decode_to_utf8_raw(&src[read..], &mut out[written..written + window], true);
            assert!(
                r > 0 || w > 0,
                "stalled at byte {read} with a {window}-byte window: {step:?}"
            );
            assert!(
                !matches!(step, DecoderResult::Malformed(_, _)),
                "unexpected error at {read}"
            );
            result = step;
            read += r;
            written += w;
        }
        (result, read, written)
    }

    // -- well-formed decoding: endianness -------------------------------------

    #[test]
    fn le_and_be_decoders_agree_on_the_same_scalars() {
        let text = "Hi \u{00A9} \u{20AC}";
        let want = units(text);
        let mut le_dst = [0u16; 16];
        let (result, read, written) = decode_le_to_utf16(&le(text), &mut le_dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (12, 6));
        assert_eq!(&le_dst[..written], &want[..]);

        let mut be_dst = [0u16; 16];
        let (result, read, written) = decode_be_to_utf16(&be(text), &mut be_dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (12, 6));
        assert_eq!(&be_dst[..written], &want[..]);
    }

    #[test]
    fn a_be_stream_is_misread_by_the_le_decoder_and_the_reverse() {
        // Neither decoder sniffs a byte-order mark, so each reads the other
        // order's stream as byte-swapped BMP scalars.
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(b"\x00\x48\x00\x69", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0x4800, 0x6900]);

        let mut dst = [0u16; 8];
        let (result, read, written) = decode_be_to_utf16(b"\x48\x00\x69\x00", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0x4800, 0x6900]);
    }

    #[test]
    fn decoder_reads_basic_latin_scalar_boundaries_to_utf16() {
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(&le("\u{1}"), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0x0001);

        let (result, read, written) = decode_le_to_utf16(&le("\u{7F}"), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0x007F);
    }

    #[test]
    fn decoder_reads_non_ascii_bmp_scalar_boundaries_to_utf16() {
        for (text, want) in [("\u{80}", [0x0080u16]), ("\u{7FF}", [0x07FF])] {
            let mut dst = [0u16; 8];
            let (result, read, written) = decode_le_to_utf16(&le(text), &mut dst, true);
            assert_eq!(result, DecoderResult::InputEmpty, "text {text:?}");
            assert_eq!((read, written), (2, 1), "text {text:?}");
            assert_eq!(&dst[..written], &want[..], "text {text:?}");
        }
    }

    #[test]
    fn decoder_reads_upper_bmp_and_pre_surrogate_scalar_boundaries() {
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(&le("\u{800}\u{D7FE}"), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0x0800, 0xD7FE]);
    }

    #[test]
    fn decoder_reads_post_surrogate_and_last_unit_bmp_boundaries() {
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(&le("\u{E000}\u{FFFD}\u{FFFF}"), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (6, 3));
        assert_eq!(&dst[..written], &[0xE000, 0xFFFD, 0xFFFF]);
    }

    #[test]
    fn decoder_reads_the_nul_unit_as_data_not_a_terminator() {
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(b"\x00\x00A\x00", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0x0000, 0x0041]);

        let mut out = [0u8; 8];
        let (result, read, written) = decode_le_to_utf8(b"\x00\x00", &mut out, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(out[0], 0x00);
    }

    #[test]
    fn decoder_reads_a_multi_stride_run_of_basic_latin() {
        let text: String = "a".repeat(40);
        let mut dst = [0u16; 48];
        let (result, read, written) = decode_le_to_utf16(&le(&text), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (80, 40));
        assert!(dst[..40].iter().all(|&u| u == 0x61));

        let mut out = [0u8; 64];
        let (result, read, written) = decode_le_to_utf8(&le(&text), &mut out, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (80, 40));
        assert_eq!(&out[..written], text.as_bytes());
    }

    // -- byte-order marks -----------------------------------------------------

    #[test]
    fn le_decoder_emits_its_own_byte_order_mark_as_a_character() {
        // The raw decoder performs no BOM removal: 0xFEFF read in the
        // decoder's order is an ordinary BMP scalar.
        let mut dst = [0u16; 4];
        let (result, read, written) = decode_le_to_utf16(b"\xFF\xFE", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0xFEFF);
    }

    #[test]
    fn be_decoder_emits_its_own_byte_order_mark_as_a_character() {
        let mut dst = [0u16; 4];
        let (result, read, written) = decode_be_to_utf16(b"\xFE\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0xFEFF);
    }

    #[test]
    fn decoder_emits_the_opposite_order_mark_as_u_plus_fffe() {
        let mut dst = [0u16; 4];
        let (result, read, written) = decode_le_to_utf16(b"\xFE\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0xFFFE);

        let mut dst = [0u16; 4];
        let (result, _read, _written) = decode_be_to_utf16(b"\xFF\xFE", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!(dst[0], 0xFFFE);
    }

    #[test]
    fn every_bom_unit_of_a_stream_is_emitted() {
        // NOTE: the WHATWG decode algorithm removes one leading BOM in BOM
        // removal mode; nothing here does, so the second 0xFEFF is a character
        // just like the first.
        let mut dst = [0u16; 4];
        let (result, read, written) = decode_be_to_utf16(b"\xFE\xFF\xFE\xFF", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0xFEFF, 0xFEFF]);
    }

    #[test]
    fn bom_units_are_reencoded_to_utf8_like_any_bmp_scalar() {
        let mut out = [0u8; 8];
        let (result, read, written) = decode_be_to_utf8(b"\xFE\xFF\xFF\xFE", &mut out, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 6));
        assert_eq!(&out[..written], "\u{FEFF}\u{FFFE}".as_bytes());
    }

    #[test]
    fn replacing_decoder_handle_does_not_remove_a_utf16_byte_order_mark() {
        // NOTE: upstream `Decoder::decode` consumes the leading BOM when the
        // encoding was created with BOM removal; here the life cycle recorded
        // by `Decoder::new` is never consulted by `decode_to_utf8`, so the BOM
        // reaches the output as U+FEFF.
        let mut out = [0u8; 16];
        let (result, read, written, had_errors) = UTF_16LE
            .new_decoder_with_bom_removal()
            .decode_to_utf8(b"\xFF\xFE\x68\x00\x69\x00", &mut out, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!((read, written), (6, 5));
        assert!(!had_errors);
        assert_eq!(&out[..written], "\u{FEFF}hi".as_bytes());
    }

    #[test]
    fn bom_removal_flag_does_not_change_the_decoded_output() {
        // NOTE: `encoding.rs:548-562` records `AtUtf16LeStart` for the removal
        // mode, but neither `decode_to_utf8` (`encoding.rs:614`) nor
        // `decode_to_utf16` (`encoding.rs:657`) ever reads `life_cycle`, so the
        // two handles behave identically and the BOM is never consumed.
        let mut removing = [0u8; 16];
        let removal = UTF_16LE
            .new_decoder_with_bom_removal()
            .decode_to_utf8(b"\xFF\xFE\x41\x00", &mut removing, true);
        let mut plain = [0u8; 16];
        let no_removal = UTF_16LE
            .new_decoder_without_bom_handling()
            .decode_to_utf8(b"\xFF\xFE\x41\x00", &mut plain, true);
        assert_eq!(removal, no_removal);
        assert_eq!((removal.1, removal.2), (4, 4));
        assert!(!removal.3);
        assert_eq!(&removing[..4], "\u{FEFF}A".as_bytes());
        assert_eq!(plain, removing);
    }

    #[test]
    fn the_replacing_wrapper_appends_one_error_character_after_the_valid_output() {
        let mut out = [0u8; 16];
        let mut decoder = UTF_16LE.new_decoder_with_bom_removal();
        let (result, read, written, had_errors) = decoder.decode_to_utf8(b"\x41\x00\x42", &mut out, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!((read, written), (3, 4));
        assert!(had_errors);
        // NOTE: `encoding.rs:635-643` always writes a three-byte U+FFFD for a
        // `Malformed` report, whatever `bytes_in_error` says, so one trailing
        // half byte costs the same as a whole errored code unit.
        assert_eq!(&out[..written], [b'A', 0xEF, 0xBF, 0xBD]);
    }

    #[test]
    fn the_bom_start_life_cycle_is_never_left() {
        let mut decoder = UTF_16BE.new_decoder_with_bom_removal();
        assert_eq!(decoder.max_utf16_buffer_length(4), Some(3));
        let mut dst = [0u16; 8];
        assert_eq!(
            decoder.decode_to_utf16(b"\xFE\xFF\x00\x41", &mut dst, true),
            (CoderResult::InputEmpty, 4, 2, false)
        );
        assert_eq!(&dst[..2], &[0xFEFF, 0x0041]);
        // the decoder still answers as if it were at the start of the stream
        assert_eq!(decoder.max_utf16_buffer_length(4), Some(3));
        assert_eq!(decoder.max_utf8_buffer_length(4), Some(7));
        assert_eq!(decoder.max_utf8_buffer_length_without_replacement(4), Some(7));
    }

    // -- half code units ------------------------------------------------------

    #[test]
    fn trailing_half_unit_is_malformed_when_last() {
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(b"\x61\x00\x62", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (3, 1));
        assert_eq!(dst[0], 0x0061);
        assert_eq!(dst[1], 0);
    }

    #[test]
    fn trailing_half_unit_is_buffered_when_not_last() {
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x61\x00\x62", &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (3, 1));
        // the half unit already counted as read
        assert_eq!(decoder.additional_from_state(), 2);
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x00", &mut dst[1..], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[1], 0x0062);
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn odd_length_input_is_malformed_once_in_utf8_and_leaves_no_state() {
        let mut out = [0u8; 8];
        let mut decoder = Utf16Decoder::new(false);
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x61\x00\xE9", &mut out, true);
        assert_eq!(result, DecoderResult::Malformed(1, 0));
        assert_eq!((read, written), (3, 1));
        assert_eq!(out[0], b'a');
        assert_eq!(out[1], 0);
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn a_half_unit_can_arrive_before_its_partner_in_the_next_chunk() {
        // The reverse split: lead byte in chunk one, trail byte in chunk two.
        let mut decoder = Utf16Decoder::new(true);
        let mut out = [0u8; 8];
        let (result, read, written) = decoder.decode_to_utf8_raw(&be("\u{2603}")[..1], &mut out, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 0));
        // the half unit already counted as read
        assert_eq!(decoder.additional_from_state(), 2);
        let (result, read, written) = decoder.decode_to_utf8_raw(&be("\u{2603}")[1..], &mut out, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 3));
        assert_eq!(&out[..written], "\u{2603}".as_bytes());
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn empty_input_with_no_state_is_input_empty() {
        let mut dst = [0u16; 8];
        assert_eq!(
            decode_le_to_utf16(b"", &mut dst, true),
            (DecoderResult::InputEmpty, 0, 0)
        );
        assert_eq!(
            decode_be_to_utf16(b"", &mut dst, false),
            (DecoderResult::InputEmpty, 0, 0)
        );
    }

    #[test]
    fn single_byte_chunk_completes_only_on_the_second_call() {
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x41], &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 0));
        let (result, read, written) = decoder.decode_to_utf16_raw(&[0x00], &mut dst, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (1, 1));
        assert_eq!(dst[0], 0x0041);
    }

    // -- surrogate pairs ------------------------------------------------------

    #[test]
    fn valid_astral_pair_is_reencoded_to_four_utf8_bytes() {
        let mut out = [0u8; 8];
        let (result, read, written) = decode_le_to_utf8(&pair_bytes(0xD83D, 0xDE00, false), &mut out, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 4));
        assert_eq!(&out[..written], "\u{1F600}".as_bytes());
    }

    #[test]
    fn valid_astral_pair_is_written_verbatim_to_a_utf16_destination() {
        // The bulk path (`handles.rs:429-430`) stores both units of a well
        // formed pair, so the destination keeps the surrogates exactly as they
        // appeared on the wire.
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(&pair_bytes(0xD83D, 0xDE00, false), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0xD83D, 0xDE00]);
    }

    #[test]
    fn surrogate_pair_boundaries_are_written_verbatim_too() {
        // U+10000 = D800 DC00 and U+10FFFF = DBFF DFFF, big-endian order.
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_be_to_utf16(&pair_bytes(0xD800, 0xDC00, true), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0xD800, 0xDC00]);

        let mut dst = [0u16; 8];
        let (result, read, written) = decode_be_to_utf16(&pair_bytes(0xDBFF, 0xDFFF, true), &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 2));
        assert_eq!(&dst[..written], &[0xDBFF, 0xDFFF]);
    }

    #[test]
    fn highest_and_lowest_astral_scalars_reach_utf8() {
        for (high, low, want) in [
            (0xD800u16, 0xDC00u16, "\u{10000}"),
            (0xDBFFu16, 0xDFFFu16, "\u{10FFFF}"),
            (0xD83Du16, 0xDE00u16, "\u{1F600}"),
        ] {
            let mut out = [0u8; 8];
            for big_endian in [false, true] {
                let src = pair_bytes(high, low, big_endian);
                let (result, read, written) = if big_endian {
                    decode_be_to_utf8(&src, &mut out, true)
                } else {
                    decode_le_to_utf8(&src, &mut out, true)
                };
                assert_eq!(
                    result,
                    DecoderResult::InputEmpty,
                    "{high:04X} {low:04X} be={big_endian}"
                );
                assert_eq!(
                    (read, written),
                    (4, want.len()),
                    "{high:04X} {low:04X} be={big_endian}"
                );
                assert_eq!(
                    &out[..written],
                    want.as_bytes(),
                    "{high:04X} {low:04X} be={big_endian}"
                );
            }
        }
    }

    #[test]
    fn unpaired_high_surrogate_at_end_is_two_malformed_bytes() {
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(b"\x3D\xD8", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (2, 0));
        assert_eq!(dst[0], 0);
    }

    #[test]
    fn unpaired_high_surrogate_after_text_is_two_malformed_bytes_after_the_text() {
        let mut out = [0u8; 8];
        let (result, read, written) = decode_le_to_utf8(b"\x41\x00\x3D\xD8", &mut out, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (4, 1));
        assert_eq!(out[0], b'A');
    }

    #[test]
    fn unpaired_high_surrogate_followed_by_a_bmp_unit_defers_the_bmp_unit() {
        // NOTE: `utf16.rs:176` stashes the pending BMP unit in the
        // `lead_surrogate` field and raises `pending_bmp`, so
        // `additional_from_state` (utf16.rs:37) cannot tell "pending BMP" from
        // "pending lead surrogate": both report 3.
        let mut decoder = Utf16Decoder::new(false);
        let mut out = [0u8; 8];
        // the buffered lead surrogate forces the one-unit-at-a-time path
        assert_eq!(
            decoder.decode_to_utf8_raw(b"\x3D\xD8", &mut out, false),
            (DecoderResult::InputEmpty, 2, 0)
        );
        // NOTE: `Malformed(2, 2)` promises two reprocessable bytes, yet the
        // returned read index (2) already covers the whole chunk; the pending
        // BMP unit is kept in the decoder instead of being handed back.
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x41\x00", &mut out[1..], true);
        assert_eq!(result, DecoderResult::Malformed(2, 2));
        assert_eq!((read, written), (2, 0));
        assert_eq!(out[1], 0);
        assert_eq!(decoder.additional_from_state(), 3);
        // the deferred unit is still inside the decoder and needs three free
        // bytes, the widest BMP sequence
        let mut flushed = [0u8; 4];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"", &mut flushed, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (0, 1));
        assert_eq!(flushed[0], b'A');
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn an_errored_pair_in_the_bulk_path_stops_before_the_next_unit() {
        // With everything in one chunk the bulk path handles the pair: it
        // reports the lead surrogate as malformed at the point it found it,
        // keeps no state, and leaves the following unit for the next call.
        let mut decoder = Utf16Decoder::new(false);
        let mut out = [0u8; 8];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x41\x00\x3D\xD8\x41\x00", &mut out, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (4, 1));
        assert_eq!(out[0], b'A');
        assert_eq!(out[1], 0);
        assert_eq!(decoder.additional_from_state(), 1);
        // nothing was deferred inside the decoder, so the tail is replayed
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x41\x00", &mut out[1..], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(&out[..2], b"AA");
    }

    #[test]
    fn unpaired_high_surrogate_followed_by_a_bmp_unit_defers_it_to_utf16() {
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        assert_eq!(
            decoder.decode_to_utf16_raw(b"\x3D\xD8", &mut dst, false),
            (DecoderResult::InputEmpty, 2, 0)
        );
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x03\x26", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 2));
        assert_eq!((read, written), (2, 0));
        assert_eq!(dst[0], 0);
        assert_eq!(decoder.additional_from_state(), 3);
        let (result, read, written) = decoder.decode_to_utf16_raw(b"", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (0, 1));
        assert_eq!(dst[0], 0x2603);
    }

    #[test]
    fn an_errored_pair_in_the_utf16_bulk_path_leaves_the_next_unit_unread() {
        // NOTE: `handles.rs:415-428` stops at the non-trail unit and reports the
        // lead surrogate as `Malformed(2, 0)` with the read index already past
        // the errored unit, so the stream's remaining unit is simply not read.
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        let (result, read, written) =
            decoder.decode_to_utf16_raw(b"\x41\x00\x3D\xD8\x03\x26", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (4, 1));
        assert_eq!(dst[0], 0x0041);
        assert_eq!(dst[1], 0);
        assert_eq!(decoder.additional_from_state(), 1);
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x03\x26", &mut dst[1..], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[1], 0x2603);
    }

    #[test]
    fn pending_bmp_unit_is_written_exactly_once_across_calls() {
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 4];
        assert_eq!(
            decoder.decode_to_utf16_raw(b"\x3D\xD8", &mut dst, false),
            (DecoderResult::InputEmpty, 2, 0)
        );
        // a U+FEFF unit cannot pair with a lead surrogate: it becomes the
        // pending BMP character
        assert_eq!(
            decoder.decode_to_utf16_raw(b"\xFF\xFE", &mut dst, false),
            (DecoderResult::Malformed(2, 2), 2, 0)
        );
        assert_eq!(decoder.additional_from_state(), 3);
        assert_eq!(
            decoder.decode_to_utf16_raw(b"", &mut dst, true),
            (DecoderResult::InputEmpty, 0, 1)
        );
        assert_eq!(dst[0], 0xFEFF);
        // and is not replayed by the following call
        assert_eq!(
            decoder.decode_to_utf16_raw(b"", &mut dst[1..], true),
            (DecoderResult::InputEmpty, 0, 0)
        );
        assert_eq!(dst[1], 0);
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn two_lead_surrogates_in_a_row_keep_the_second_as_a_lead() {
        // NOTE: `utf16.rs:152-159` overwrites `lead_surrogate` with the second
        // lead surrogate but does not set `pending_bmp`, so the first lead
        // surrogate disappears with a single `Malformed(2, 2)` report and the
        // second one is paired with whatever follows it. WHATWG's algorithm
        // emits an error for the first lead and then treats the second as a
        // fresh lead, which costs the same output here, but the standard also
        // reprocesses the trailing unit; `read` already counts it.
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        assert_eq!(
            decoder.decode_to_utf16_raw(b"\x3D\xD8", &mut dst, false),
            (DecoderResult::InputEmpty, 2, 0)
        );
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x00\xD8", &mut dst, false);
        assert_eq!(result, DecoderResult::Malformed(2, 2));
        assert_eq!((read, written), (2, 0));
        assert_eq!(decoder.additional_from_state(), 3);
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x00\xDE", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
        assert_eq!(&dst[..written], &[0xD800, 0xDE00]);
    }

    #[test]
    fn lone_low_surrogate_is_two_malformed_bytes() {
        // Bulk path: the offending unit is counted as read and nothing is
        // written for it.
        let mut dst = [0u16; 8];
        let (result, read, written) = decode_le_to_utf16(b"\x00\xDE", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (2, 0));
    }

    #[test]
    fn lone_low_surrogate_after_a_pair_counts_as_consumed() {
        // The valid pair reaches the destination, then the third unit (a lone
        // low surrogate) is reported malformed with the read index already
        // past it, so a caller that trusts `read` loses that unit.
        // NOTE: WHATWG's algorithm emits an error for the offending unit and
        // then keeps decoding; here `Malformed(2, 0)` with `read == 6` means the
        // stream stops being decoded at that point.
        let mut dst = [0u16; 16];
        let (result, read, written) = decode_le_to_utf16(b"\x3D\xD8\x00\xDE\x01\xDD", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (6, 2));
        assert_eq!(&dst[..2], &[0xD83D, 0xDE00]);
        assert_eq!(dst[2], 0);
    }

    #[test]
    fn lone_low_surrogate_after_basic_latin_reports_the_stop_positions() {
        let mut dst = [0u16; 32];
        let text: String = "a".repeat(20);
        let mut src = le(&text);
        src.extend_from_slice(&[0x00, 0xDE]);
        let (result, read, written) = decode_le_to_utf16(&src, &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (42, 20));
        assert!(dst[..20].iter().all(|&u| u == 0x61));
        assert_eq!(dst[20], 0);
    }

    #[test]
    fn lone_low_surrogate_in_the_slow_path_reports_stop_positions() {
        // A half unit pending from an earlier chunk keeps this chunk on the
        // one-code-unit-at-a-time path, where a low surrogate with no lead is
        // malformed with the same (2, 0) shape but only the byte read here is
        // counted: its partner was already counted by the previous call.
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        assert_eq!(
            decoder.decode_to_utf16_raw(b"\x00", &mut dst, false),
            (DecoderResult::InputEmpty, 1, 0)
        );
        assert_eq!(decoder.additional_from_state(), 2);
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\xDE\x00\x41\x00", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(2, 0));
        assert_eq!((read, written), (1, 0));
        assert_eq!(dst, [0u16; 8]);
        // the errored lead byte was dropped, so the next chunk starts fresh
        assert_eq!(decoder.additional_from_state(), 1);
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x41\x00", &mut dst, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0x0041);
    }

    #[test]
    fn surrogate_pair_split_across_two_calls_is_assembled() {
        let mut decoder = Utf16Decoder::new(false);
        let mut out = [0u8; 8];
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x3D\xD8", &mut out, false);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 0));
        assert_eq!(decoder.additional_from_state(), 3);
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x00\xDE", &mut out, true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 4));
        assert_eq!(&out[..written], "\u{1F600}".as_bytes());
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn surrogate_pair_split_one_byte_at_a_time_is_assembled() {
        let mut decoder = Utf16Decoder::new(true);
        let mut out = [0u8; 8];
        let src = pair_bytes(0xD83D, 0xDE00, true);
        let mut written = 0usize;
        for (i, byte) in src.iter().enumerate() {
            let (result, read, w) =
                decoder.decode_to_utf8_raw(core::slice::from_ref(byte), &mut out[written..], false);
            assert_eq!(result, DecoderResult::InputEmpty, "byte {i}");
            assert_eq!(read, 1, "byte {i}");
            written += w;
        }
        let (result, read, w) = decoder.decode_to_utf8_raw(b"", &mut out[written..], true);
        assert_eq!(result, DecoderResult::InputEmpty);
        assert_eq!((read, written + w), (0, 4));
        assert_eq!(&out[..4], "\u{1F600}".as_bytes());
    }

    #[test]
    fn pending_lead_surrogate_plus_a_half_unit_flushes_three_malformed_bytes() {
        let mut decoder = Utf16Decoder::new(false);
        let mut dst = [0u16; 8];
        assert_eq!(
            decoder.decode_to_utf16_raw(b"\x3D\xD8", &mut dst, false),
            (DecoderResult::InputEmpty, 2, 0)
        );
        let (result, read, written) = decoder.decode_to_utf16_raw(b"\x41", &mut dst, true);
        assert_eq!(result, DecoderResult::Malformed(3, 0));
        assert_eq!((read, written), (1, 0));
        assert_eq!(decoder.additional_from_state(), 1);
    }

    #[test]
    fn pending_lead_surrogate_plus_a_half_unit_flushes_three_bytes_to_utf8() {
        let mut decoder = Utf16Decoder::new(true);
        let mut out = [0u8; 8];
        assert_eq!(
            decoder.decode_to_utf8_raw(b"\xD8\x3D", &mut out, false),
            (DecoderResult::InputEmpty, 2, 0)
        );
        let (result, read, written) = decoder.decode_to_utf8_raw(b"\x41", &mut out, true);
        assert_eq!(result, DecoderResult::Malformed(3, 0));
        assert_eq!((read, written), (1, 0));
        assert_eq!(decoder.additional_from_state(), 1);
    }
}
