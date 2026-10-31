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

use crate::ascii::{ascii_to_ascii, ascii_to_basic_latin, ascii_valid_up_to, validate_ascii};
use crate::single_byte::{SingleByteDecoder, SingleByteEncoder};
use crate::utf8::{
    Utf8Decoder, Utf8Encoder, convert_utf16_to_utf8_partial_inner, convert_utf16_to_utf8_partial_tail,
    utf8_valid_up_to,
};
use crate::utf16::{Utf16Decoder, Utf16Encoder};
use alloc::borrow::Cow;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::PartialEq;
use core::slice;

pub const NCR_EXTRA: usize = 10;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Encoding {
    pub name: &'static str,
    pub variant: VariantEncoding,
}

impl Encoding {
    #[inline]
    pub fn encoding(&'static self) -> &'static Encoding {
        self
    }

    #[inline]
    pub fn can_encode_everything(&self) -> bool {
        matches!(self.variant, VariantEncoding::Utf8)
    }

    #[inline]
    pub fn new_decoder(&'static self) -> Decoder {
        Decoder::new(self, self.variant.new_variant_decoder(), BomHandling::Off)
    }

    #[inline]
    pub fn new_decoder_with_bom_removal(&'static self) -> Decoder {
        Decoder::new(self, self.variant.new_variant_decoder(), BomHandling::Remove)
    }

    #[inline]
    pub fn new_decoder_without_bom_handling(&'static self) -> Decoder {
        Decoder::new(self, self.variant.new_variant_decoder(), BomHandling::Off)
    }

    #[inline]
    pub fn new_encoder(&'static self) -> Encoder {
        self.variant.new_encoder(self)
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum VariantEncoding {
    SingleByte(&'static [u16; 128], u16, u8, u8),
    Utf8,
    Utf16Be,
    Utf16Le,
}

impl VariantEncoding {
    pub fn new_variant_decoder(&self) -> VariantDecoder {
        match *self {
            VariantEncoding::SingleByte(table, _, _, _) => {
                VariantDecoder::SingleByte(SingleByteDecoder { table })
            }
            VariantEncoding::Utf8 => VariantDecoder::Utf8(Utf8Decoder::new_inner()),
            VariantEncoding::Utf16Be => VariantDecoder::Utf16(Utf16Decoder::new(true)),
            VariantEncoding::Utf16Le => VariantDecoder::Utf16(Utf16Decoder::new(false)),
        }
    }

    pub fn new_encoder(&self, encoding: &'static Encoding) -> Encoder {
        match *self {
            VariantEncoding::SingleByte(table, run_bmp_offset, run_byte_offset, run_length) => Encoder::new(
                encoding,
                VariantEncoder::SingleByte(SingleByteEncoder::new(
                    table,
                    run_bmp_offset,
                    run_byte_offset,
                    run_length,
                )),
            ),
            VariantEncoding::Utf8 => Encoder::new(encoding, VariantEncoder::Utf8(Utf8Encoder)),
            VariantEncoding::Utf16Be => {
                Encoder::new(encoding, VariantEncoder::Utf16(Utf16Encoder::new(true)))
            }
            VariantEncoding::Utf16Le => {
                Encoder::new(encoding, VariantEncoder::Utf16(Utf16Encoder::new(false)))
            }
        }
    }

    pub fn is_single_byte(&self) -> bool {
        matches!(*self, VariantEncoding::SingleByte(_, _, _, _))
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    encoding: &'static Encoding,
    variant: VariantDecoder,
    life_cycle: DecoderLifeCycle,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum DecoderLifeCycle {
    AtStart,
    AtUtf8Start,
    AtUtf16BeStart,
    AtUtf16LeStart,
    SeenUtf8First,
    SeenUtf8Second,
    SeenUtf16BeFirst,
    SeenUtf16LeFirst,
    ConvertingWithPendingBB,
    Converting,
    Finished,
}

#[derive(Debug, Copy, Clone)]
pub enum BomHandling {
    Off,
    Sniff,
    Remove,
}

#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub enum CoderResult {
    InputEmpty,
    OutputFull,
}

#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub enum DecoderResult {
    InputEmpty,
    OutputFull,
    Malformed(u8, u8),
}

#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub enum EncoderResult {
    InputEmpty,
    OutputFull,
    Unmappable(char),
}

impl EncoderResult {
    pub fn unmappable_from_bmp(bmp: u16) -> EncoderResult {
        EncoderResult::Unmappable(
            ::core::char::from_u32(u32::from(bmp)).unwrap_or(char::REPLACEMENT_CHARACTER),
        )
    }
}

/// Errors produced by the fallible conversion functions in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingError {
    /// The destination buffer does not meet the minimum size required by the
    /// conversion contract.
    DestinationTooSmall {
        /// Minimum number of elements the destination must hold.
        required: usize,
        /// Number of elements actually available in the destination.
        actual: usize,
    },
    /// The destination buffer filled up before the input was fully consumed.
    OutputFull,
}

impl core::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DestinationTooSmall { required, actual } => write!(
                f,
                "destination buffer too small: requires {required} elements, got {actual}"
            ),
            Self::OutputFull => {
                write!(f, "destination buffer filled before conversion completed")
            }
        }
    }
}

impl core::error::Error for EncodingError {}

#[derive(Debug, Clone)]
pub struct Encoder {
    encoding: &'static Encoding,
    variant: VariantEncoder,
}

#[derive(Debug, Clone)]
pub enum VariantDecoder {
    SingleByte(SingleByteDecoder),
    Utf8(Utf8Decoder),
    Utf16(Utf16Decoder),
}

#[derive(Debug, Clone)]
pub enum VariantEncoder {
    SingleByte(SingleByteEncoder),
    Utf8(Utf8Encoder),
    Utf16(Utf16Encoder),
}

impl VariantDecoder {
    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        match *self {
            VariantDecoder::SingleByte(ref v) => v.max_utf16_buffer_length(byte_length),
            VariantDecoder::Utf8(ref v) => v.max_utf16_buffer_length(byte_length),
            VariantDecoder::Utf16(ref v) => v.max_utf16_buffer_length(byte_length),
        }
    }

    pub fn max_utf8_buffer_length_without_replacement(&self, byte_length: usize) -> Option<usize> {
        match *self {
            VariantDecoder::SingleByte(ref v) => v.max_utf8_buffer_length_without_replacement(byte_length),
            VariantDecoder::Utf8(ref v) => v.max_utf8_buffer_length_without_replacement(byte_length),
            VariantDecoder::Utf16(ref v) => v.max_utf8_buffer_length_without_replacement(byte_length),
        }
    }

    pub fn max_utf8_buffer_length(&self, byte_length: usize) -> Option<usize> {
        match *self {
            VariantDecoder::SingleByte(ref v) => v.max_utf8_buffer_length(byte_length),
            VariantDecoder::Utf8(ref v) => v.max_utf8_buffer_length(byte_length),
            VariantDecoder::Utf16(ref v) => v.max_utf8_buffer_length(byte_length),
        }
    }

    pub fn decode_to_utf8_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        last: bool,
    ) -> (DecoderResult, usize, usize) {
        match *self {
            VariantDecoder::SingleByte(ref mut v) => v.decode_to_utf8_raw(src, dst, last),
            VariantDecoder::Utf8(ref mut v) => v.decode_to_utf8_raw(src, dst, last),
            VariantDecoder::Utf16(ref mut v) => v.decode_to_utf8_raw(src, dst, last),
        }
    }

    pub fn decode_to_utf16_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        last: bool,
    ) -> (DecoderResult, usize, usize) {
        match *self {
            VariantDecoder::SingleByte(ref mut v) => v.decode_to_utf16_raw(src, dst, last),
            VariantDecoder::Utf8(ref mut v) => v.decode_to_utf16_raw(src, dst, last),
            VariantDecoder::Utf16(ref mut v) => v.decode_to_utf16_raw(src, dst, last),
        }
    }
}

impl VariantEncoder {
    pub fn max_buffer_length_from_utf16_without_replacement(&self, u16_length: usize) -> Option<usize> {
        match *self {
            VariantEncoder::SingleByte(ref v) => {
                v.max_buffer_length_from_utf16_without_replacement(u16_length)
            }
            VariantEncoder::Utf8(ref v) => v.max_buffer_length_from_utf16_without_replacement(u16_length),
            VariantEncoder::Utf16(ref v) => v.max_buffer_length_from_utf16_without_replacement(u16_length),
        }
    }

    pub fn max_buffer_length_from_utf8_without_replacement(&self, byte_length: usize) -> Option<usize> {
        match *self {
            VariantEncoder::SingleByte(ref v) => {
                v.max_buffer_length_from_utf8_without_replacement(byte_length)
            }
            VariantEncoder::Utf8(ref v) => v.max_buffer_length_from_utf8_without_replacement(byte_length),
            VariantEncoder::Utf16(ref v) => v.max_buffer_length_from_utf8_without_replacement(byte_length),
        }
    }

    pub fn encode_from_utf16_raw(
        &mut self,
        src: &[u16],
        dst: &mut [u8],
        last: bool,
    ) -> (EncoderResult, usize, usize) {
        match *self {
            VariantEncoder::SingleByte(ref mut v) => v.encode_from_utf16_raw(src, dst, last),
            VariantEncoder::Utf8(ref mut v) => v.encode_from_utf16_raw(src, dst, last),
            VariantEncoder::Utf16(ref mut v) => v.encode_from_utf16_raw(src, dst, last),
        }
    }

    pub fn encode_from_utf8_raw(
        &mut self,
        src: &str,
        dst: &mut [u8],
        last: bool,
    ) -> (EncoderResult, usize, usize) {
        match *self {
            VariantEncoder::SingleByte(ref mut v) => v.encode_from_utf8_raw(src, dst, last),
            VariantEncoder::Utf8(ref mut v) => v.encode_from_utf8_raw(src, dst, last),
            VariantEncoder::Utf16(ref mut v) => v.encode_from_utf8_raw(src, dst, last),
        }
    }

    pub fn has_pending_state(&self) -> bool {
        false
    }
}

impl Encoder {
    pub fn new(enc: &'static Encoding, encoder: VariantEncoder) -> Encoder {
        Encoder {
            encoding: enc,
            variant: encoder,
        }
    }

    #[inline]
    pub fn encoding(&self) -> &'static Encoding {
        self.encoding
    }

    pub fn max_buffer_length_from_utf8_if_no_unmappables(&self, byte_length: usize) -> Option<usize> {
        checked_add(
            if self.encoding().can_encode_everything() {
                0
            } else {
                NCR_EXTRA
            },
            self.max_buffer_length_from_utf8_without_replacement(byte_length),
        )
    }

    pub fn max_buffer_length_from_utf8_without_replacement(&self, byte_length: usize) -> Option<usize> {
        self.variant
            .max_buffer_length_from_utf8_without_replacement(byte_length)
    }

    pub fn encode_from_utf8(
        &mut self,
        src: &str,
        dst: &mut [u8],
        last: bool,
    ) -> (CoderResult, usize, usize, bool) {
        let dst_len = dst.len();
        let effective_dst_len = if self.encoding().can_encode_everything() {
            dst_len
        } else {
            if dst_len < NCR_EXTRA {
                if src.is_empty() && !(last && self.has_pending_state()) {
                    return (CoderResult::InputEmpty, 0, 0, false);
                }
                return (CoderResult::OutputFull, 0, 0, false);
            }
            dst_len - NCR_EXTRA
        };
        let mut had_unmappables = false;
        let mut total_read = 0usize;
        let mut total_written = 0usize;
        loop {
            let (result, read, written) = self.encode_from_utf8_without_replacement(
                &src[total_read..],
                &mut dst[total_written..effective_dst_len],
                last,
            );
            total_read += read;
            total_written += written;
            match result {
                EncoderResult::InputEmpty => {
                    return (
                        CoderResult::InputEmpty,
                        total_read,
                        total_written,
                        had_unmappables,
                    );
                }
                EncoderResult::OutputFull => {
                    return (
                        CoderResult::OutputFull,
                        total_read,
                        total_written,
                        had_unmappables,
                    );
                }
                EncoderResult::Unmappable(unmappable) => {
                    had_unmappables = true;
                    total_written += write_ncr(unmappable, &mut dst[total_written..]);
                    if total_written >= effective_dst_len {
                        if total_read == src.len() && !(last && self.has_pending_state()) {
                            return (
                                CoderResult::InputEmpty,
                                total_read,
                                total_written,
                                had_unmappables,
                            );
                        }
                        return (
                            CoderResult::OutputFull,
                            total_read,
                            total_written,
                            had_unmappables,
                        );
                    }
                }
            }
        }
    }

    pub fn encode_from_utf8_without_replacement(
        &mut self,
        src: &str,
        dst: &mut [u8],
        last: bool,
    ) -> (EncoderResult, usize, usize) {
        self.variant.encode_from_utf8_raw(src, dst, last)
    }

    pub fn max_buffer_length_from_utf16_if_no_unmappables(&self, u16_length: usize) -> Option<usize> {
        checked_add(
            if self.encoding().can_encode_everything() {
                0
            } else {
                NCR_EXTRA
            },
            self.max_buffer_length_from_utf16_without_replacement(u16_length),
        )
    }

    pub fn max_buffer_length_from_utf16_without_replacement(&self, u16_length: usize) -> Option<usize> {
        self.variant
            .max_buffer_length_from_utf16_without_replacement(u16_length)
    }

    pub fn encode_from_utf16(
        &mut self,
        src: &[u16],
        dst: &mut [u8],
        last: bool,
    ) -> (CoderResult, usize, usize, bool) {
        let dst_len = dst.len();
        let effective_dst_len = if self.encoding().can_encode_everything() {
            dst_len
        } else {
            if dst_len < NCR_EXTRA {
                if src.is_empty() && !(last && self.has_pending_state()) {
                    return (CoderResult::InputEmpty, 0, 0, false);
                }
                return (CoderResult::OutputFull, 0, 0, false);
            }
            dst_len - NCR_EXTRA
        };
        let mut had_unmappables = false;
        let mut total_read = 0usize;
        let mut total_written = 0usize;
        loop {
            let (result, read, written) = self.encode_from_utf16_without_replacement(
                &src[total_read..],
                &mut dst[total_written..effective_dst_len],
                last,
            );
            total_read += read;
            total_written += written;
            match result {
                EncoderResult::InputEmpty => {
                    return (
                        CoderResult::InputEmpty,
                        total_read,
                        total_written,
                        had_unmappables,
                    );
                }
                EncoderResult::OutputFull => {
                    return (
                        CoderResult::OutputFull,
                        total_read,
                        total_written,
                        had_unmappables,
                    );
                }
                EncoderResult::Unmappable(unmappable) => {
                    had_unmappables = true;
                    total_written += write_ncr(unmappable, &mut dst[total_written..]);
                    if total_written >= effective_dst_len {
                        if total_read == src.len() && !(last && self.has_pending_state()) {
                            return (
                                CoderResult::InputEmpty,
                                total_read,
                                total_written,
                                had_unmappables,
                            );
                        }
                        return (
                            CoderResult::OutputFull,
                            total_read,
                            total_written,
                            had_unmappables,
                        );
                    }
                }
            }
        }
    }

    pub fn encode_from_utf16_without_replacement(
        &mut self,
        src: &[u16],
        dst: &mut [u8],
        last: bool,
    ) -> (EncoderResult, usize, usize) {
        self.variant.encode_from_utf16_raw(src, dst, last)
    }

    #[inline]
    pub fn has_pending_state(&self) -> bool {
        self.variant.has_pending_state()
    }
}

impl PartialEq<Encoding> for &Encoding {
    fn eq(&self, other: &Encoding) -> bool {
        self.variant == other.variant
    }
}

impl Decoder {
    fn new(enc: &'static Encoding, decoder: VariantDecoder, sniffing: BomHandling) -> Decoder {
        Decoder {
            encoding: enc,
            variant: decoder,
            life_cycle: match sniffing {
                BomHandling::Off => DecoderLifeCycle::Converting,
                BomHandling::Sniff => DecoderLifeCycle::AtStart,
                BomHandling::Remove => {
                    if enc == UTF_8 {
                        DecoderLifeCycle::AtUtf8Start
                    } else if enc == UTF_16BE {
                        DecoderLifeCycle::AtUtf16BeStart
                    } else if enc == UTF_16LE {
                        DecoderLifeCycle::AtUtf16LeStart
                    } else {
                        DecoderLifeCycle::Converting
                    }
                }
            },
        }
    }

    #[inline]
    pub fn encoding(&self) -> &'static Encoding {
        self.encoding
    }

    pub fn max_utf8_buffer_length(&self, byte_length: usize) -> Option<usize> {
        match self.life_cycle {
            DecoderLifeCycle::Converting
            | DecoderLifeCycle::AtUtf8Start
            | DecoderLifeCycle::AtUtf16LeStart
            | DecoderLifeCycle::AtUtf16BeStart => {
                return self.variant.max_utf8_buffer_length(byte_length);
            }
            _ => {}
        }
        None
    }

    pub fn max_utf8_buffer_length_without_replacement(&self, byte_length: usize) -> Option<usize> {
        match self.life_cycle {
            DecoderLifeCycle::Converting
            | DecoderLifeCycle::AtUtf8Start
            | DecoderLifeCycle::AtUtf16LeStart
            | DecoderLifeCycle::AtUtf16BeStart => {
                return self
                    .variant
                    .max_utf8_buffer_length_without_replacement(byte_length);
            }
            _ => {}
        }
        None
    }

    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        match self.life_cycle {
            DecoderLifeCycle::Converting
            | DecoderLifeCycle::AtUtf8Start
            | DecoderLifeCycle::AtUtf16LeStart
            | DecoderLifeCycle::AtUtf16BeStart => {
                return self.variant.max_utf16_buffer_length(byte_length);
            }
            _ => {}
        }
        None
    }

    pub fn decode_to_utf8(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        last: bool,
    ) -> (CoderResult, usize, usize, bool) {
        let mut had_errors = false;
        let mut total_read = 0usize;
        let mut total_written = 0usize;
        loop {
            let (result, read, written) =
                self.decode_to_utf8_without_replacement(&src[total_read..], &mut dst[total_written..], last);
            total_read += read;
            total_written += written;
            match result {
                DecoderResult::InputEmpty => {
                    return (CoderResult::InputEmpty, total_read, total_written, had_errors);
                }
                DecoderResult::OutputFull => {
                    return (CoderResult::OutputFull, total_read, total_written, had_errors);
                }
                DecoderResult::Malformed(_, _) => {
                    had_errors = true;
                    dst[total_written] = 0xEFu8;
                    total_written += 1;
                    dst[total_written] = 0xBFu8;
                    total_written += 1;
                    dst[total_written] = 0xBDu8;
                    total_written += 1;
                }
            }
        }
    }

    pub fn decode_to_utf8_without_replacement(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        last: bool,
    ) -> (DecoderResult, usize, usize) {
        self.variant.decode_to_utf8_raw(src, dst, last)
    }

    pub fn decode_to_utf16(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        last: bool,
    ) -> (CoderResult, usize, usize, bool) {
        let mut had_errors = false;
        let mut total_read = 0usize;
        let mut total_written = 0usize;
        loop {
            let (result, read, written) =
                self.decode_to_utf16_without_replacement(&src[total_read..], &mut dst[total_written..], last);
            total_read += read;
            total_written += written;
            match result {
                DecoderResult::InputEmpty => {
                    return (CoderResult::InputEmpty, total_read, total_written, had_errors);
                }
                DecoderResult::OutputFull => {
                    return (CoderResult::OutputFull, total_read, total_written, had_errors);
                }
                DecoderResult::Malformed(_, _) => {
                    had_errors = true;
                    dst[total_written] = 0xFFFD;
                    total_written += 1;
                }
            }
        }
    }

    pub fn decode_to_utf16_without_replacement(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        last: bool,
    ) -> (DecoderResult, usize, usize) {
        self.variant.decode_to_utf16_raw(src, dst, last)
    }
}

fn write_ncr(unmappable: char, dst: &mut [u8]) -> usize {
    let mut number = unmappable as u32;
    let len = if number >= 1_000_000u32 {
        10
    } else if number >= 100_000u32 {
        9
    } else if number >= 10_000u32 {
        8
    } else if number >= 1_000u32 {
        7
    } else if number >= 100u32 {
        6
    } else {
        5
    };
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
pub(crate) fn in_range16(i: u16, start: u16, end: u16) -> bool {
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
fn checked_div(opt: Option<usize>, num: usize) -> Option<usize> {
    if let Some(n) = opt {
        n.checked_div(num)
    } else {
        None
    }
}

pub static UTF_8: Encoding = Encoding {
    name: "UTF-8",
    variant: VariantEncoding::Utf8,
};
pub static UTF_16BE: Encoding = Encoding {
    name: "UTF-16BE",
    variant: VariantEncoding::Utf16Be,
};
pub static UTF_16LE: Encoding = Encoding {
    name: "UTF-16LE",
    variant: VariantEncoding::Utf16Le,
};

pub fn decode_latin1(bytes: &[u8]) -> Cow<'_, str> {
    unsafe {
        let up_to = ascii_valid_up_to(bytes);
        if up_to >= bytes.len() {
            return Cow::Borrowed(::core::str::from_utf8_unchecked(bytes));
        }
        let (head, tail) = bytes.split_at(up_to);
        let capacity = head.len() + tail.len() * 2;
        let mut vec = Vec::with_capacity(capacity);
        vec.extend_from_slice(head);
        let old_len = vec.len();
        let mut spare_temp = vec.clone();
        let spare_capacity = minimally_init(spare_temp.spare_capacity_mut());
        debug_assert_eq!(old_len, up_to);
        let written = convert_latin1_to_utf8(tail, spare_capacity.assume_init_mut());
        debug_assert!(written <= spare_capacity.len());
        let new_len = old_len + written;
        debug_assert!(new_len <= vec.capacity());
        let new_len = new_len.min(vec.capacity());
        vec.set_len(new_len);
        Cow::Owned(String::from_utf8_unchecked(vec))
    }
}

pub fn encode_latin1_lossy(string: &str) -> Cow<'_, [u8]> {
    unsafe {
        let bytes = string.as_bytes();
        let up_to = ascii_valid_up_to(bytes);
        if up_to >= bytes.len() {
            return Cow::Borrowed(bytes);
        }
        let (head, tail) = bytes.split_at(up_to);
        let capacity = bytes.len();
        let mut vec = Vec::with_capacity(capacity);
        vec.extend_from_slice(head);
        let old_len = vec.len();
        let mut spare_temp = vec.clone();
        let spare_capacity = minimally_init(spare_temp.spare_capacity_mut());
        debug_assert_eq!(old_len, up_to);
        let written = convert_utf8_to_latin1_lossy(tail, spare_capacity.assume_init_mut());
        debug_assert!(written <= spare_capacity.len());
        let new_len = old_len + written;
        debug_assert!(new_len <= vec.capacity());
        let new_len = new_len.min(vec.capacity());
        vec.set_len(new_len);
        Cow::Owned(vec)
    }
}

pub fn convert_utf8_to_latin1_lossy(src: &[u8], dst: &mut [u8]) -> usize {
    let src_len = src.len();
    let mut total_read = 0usize;
    let mut total_written = 0usize;
    loop {
        let src_left = src_len - total_read;
        let dst_left = dst.len() - total_written;
        let _min_left = ::core::cmp::min(src_left, dst_left);
        if let Some((non_ascii, consumed)) = { ascii_to_ascii(&src[total_read..], &mut dst[total_written..]) }
        {
            total_read += consumed + 1;
            total_written += consumed;
            if total_read == src_len {
                return total_written;
            }
            let trail = src[total_read];
            total_read += 1;
            dst[total_written] = ((non_ascii & 0x1F) << 6) | (trail & 0x3F);
            total_written += 1;
            continue;
        }
        return total_written + src_left;
    }
}

pub fn convert_latin1_to_utf8(src: &[u8], dst: &mut [u8]) -> usize {
    let (read, written) = convert_latin1_to_utf8_partial(src, dst);
    debug_assert_eq!(read, src.len());
    written
}

pub fn convert_latin1_to_utf8_partial(src: &[u8], dst: &mut [u8]) -> (usize, usize) {
    let src_len = src.len();
    let dst_len = dst.len();
    let mut total_read = 0usize;
    let mut total_written = 0usize;
    loop {
        let src_left = src_len - total_read;
        let dst_left = dst_len - total_written;
        let min_left = ::core::cmp::min(src_left, dst_left);
        if let Some((non_ascii, consumed)) = { ascii_to_ascii(&src[total_read..], &mut dst[total_written..]) }
        {
            total_read += consumed;
            total_written += consumed;
            if total_written.saturating_add(2) > dst_len {
                return (total_read, total_written);
            }
            total_read += 1;
            dst[total_written] = (non_ascii >> 6) | 0xC0;
            total_written += 1;
            dst[total_written] = (non_ascii & 0x3F) | 0x80;
            total_written += 1;
            continue;
        }
        return (total_read + min_left, total_written + min_left);
    }
}

pub fn convert_utf8_to_utf16(src: &[u8], dst: &mut [u16]) -> Result<usize, EncodingError> {
    if dst.len() <= src.len() {
        return Err(EncodingError::DestinationTooSmall {
            required: src.len().saturating_add(1),
            actual: dst.len(),
        });
    }
    let mut decoder = Utf8Decoder::new_inner();
    let mut total_read = 0usize;
    let mut total_written = 0usize;
    loop {
        let (result, read, written) =
            decoder.decode_to_utf16_raw(&src[total_read..], &mut dst[total_written..], true);
        total_read += read;
        total_written += written;
        match result {
            DecoderResult::InputEmpty => return Ok(total_written),
            DecoderResult::OutputFull => {
                // The precondition above guarantees enough room for every
                // input byte, so this is defensive only: report it as an
                // error instead of panicking.
                return Err(EncodingError::OutputFull);
            }
            DecoderResult::Malformed(_, _) => unsafe {
                *dst.as_mut_ptr()
                    .wrapping_add(total_written)
                    .cast::<u32>() = 0xFFFD;
                total_written += 1;
            },
        }
    }
}

pub fn convert_str_to_utf16(src: &str, dst: &mut [u16]) -> Result<usize, EncodingError> {
    if dst.len() < src.len() {
        return Err(EncodingError::DestinationTooSmall {
            required: src.len(),
            actual: dst.len(),
        });
    }
    let bytes = src.as_bytes();
    let mut read = 0;
    let mut written = 0;
    let mut byte = {
        let src_remaining = &bytes[read..];
        let dst_remaining = &mut dst[written..];
        let length = src_remaining.len();
        match ascii_to_basic_latin(src_remaining, dst_remaining) {
            None => {
                written += length;
                return Ok(written);
            }
            Some((non_ascii, consumed)) => {
                read += consumed;
                written += consumed;
                non_ascii
            }
        }
    };
    loop {
        if byte < 0xE0 {
            if byte >= 0x80 {
                let second = unsafe { *(bytes.get_unchecked(read + 1)) };
                let point = ((u16::from(byte) & 0x1F) << 6) | (u16::from(second) & 0x3F);
                unsafe { *(dst.get_unchecked_mut(written)) = point };
                read += 2;
                written += 1;
            } else {
                unsafe { *(dst.get_unchecked_mut(written)) = u16::from(byte) };
                read += 1;
                written += 1;
            }
        } else if byte < 0xF0 {
            let second = unsafe { *(bytes.get_unchecked(read + 1)) };
            let third = unsafe { *(bytes.get_unchecked(read + 2)) };
            let point = ((u16::from(byte) & 0xF) << 12)
                | ((u16::from(second) & 0x3F) << 6)
                | (u16::from(third) & 0x3F);
            unsafe { *(dst.get_unchecked_mut(written)) = point };
            read += 3;
            written += 1;
        } else {
            let second = unsafe { *(bytes.get_unchecked(read + 1)) };
            let third = unsafe { *(bytes.get_unchecked(read + 2)) };
            let fourth = unsafe { *(bytes.get_unchecked(read + 3)) };
            let point = ((u32::from(byte) & 0x7) << 18)
                | ((u32::from(second) & 0x3F) << 12)
                | ((u32::from(third) & 0x3F) << 6)
                | (u32::from(fourth) & 0x3F);
            unsafe { *(dst.get_unchecked_mut(written)) = (0xD7C0 + (point >> 10)) as u16 };
            unsafe { *(dst.get_unchecked_mut(written + 1)) = (0xDC00 + (point & 0x3FF)) as u16 };
            read += 4;
            written += 2;
        }
        if read >= bytes.len() {
            return Ok(written);
        }
        byte = bytes[read];
    }
}

pub fn convert_utf16_to_utf8_partial(src: &[u16], dst: &mut [u8]) -> (usize, usize) {
    let (read, written) = convert_utf16_to_utf8_partial_inner(src, dst);
    if likely(read == src.len()) {
        return (read, written);
    }
    let (tail_read, tail_written) = convert_utf16_to_utf8_partial_tail(&src[read..], &mut dst[written..]);
    (read + tail_read, written + tail_written)
}

pub fn convert_utf16_to_utf8(src: &[u16], dst: &mut [u8]) -> Result<usize, EncodingError> {
    let required = src.len().saturating_mul(3);
    if dst.len() < required {
        return Err(EncodingError::DestinationTooSmall {
            required,
            actual: dst.len(),
        });
    }
    let (read, written) = convert_utf16_to_utf8_partial(src, dst);
    debug_assert_eq!(read, src.len());
    Ok(written)
}

#[inline(always)]
fn likely(b: bool) -> bool {
    b
}

pub fn convert_latin1_to_str_partial(src: &[u8], dst: &mut str) -> (usize, usize) {
    let bytes: &mut [u8] = unsafe { dst.as_bytes_mut() };
    let (read, written) = convert_latin1_to_utf8_partial(src, bytes);
    let len = bytes.len();
    let mut trail = written;
    let max = ::core::cmp::min(len, trail + MAX_STRIDE_SIZE);
    while trail < max {
        bytes[trail] = 0;
        trail += 1;
    }
    while trail < len && ((bytes[trail] & 0xC0) == 0x80) {
        bytes[trail] = 0;
        trail += 1;
    }
    (read, written)
}

pub fn convert_latin1_to_str(src: &[u8], dst: &mut str) -> Result<usize, EncodingError> {
    let required = src.len().saturating_mul(2);
    if dst.len() < required {
        return Err(EncodingError::DestinationTooSmall {
            required,
            actual: dst.len(),
        });
    }
    let (read, written) = convert_latin1_to_str_partial(src, dst);
    debug_assert_eq!(read, src.len());
    Ok(written)
}

pub fn decode_utf8(bytes: &[u8]) -> Cow<'_, str> {
    unsafe {
        let up_to = utf8_valid_up_to(bytes);
        if up_to >= bytes.len() {
            return Cow::Borrowed(str::from_utf8_unchecked(bytes));
        }
        let (head, tail) = bytes.split_at(up_to);
        let capacity = head.len() + tail.len() * 3;
        let mut vec = Vec::with_capacity(capacity);
        vec.extend_from_slice(head);
        let old_len = vec.len();
        let mut spare_temp = vec.clone();
        let spare_capacity = minimally_init(spare_temp.spare_capacity_mut());
        debug_assert_eq!(old_len, up_to);
        let bytes_len = spare_capacity.len() / 2;
        let written = match convert_utf8_to_utf16(
            tail,
            slice::from_raw_parts_mut(spare_capacity.as_mut_ptr() as *mut u16, bytes_len),
        ) {
            Ok(written) => written,
            Err(_) => {
                debug_assert!(false, "decode_utf8 buffer sizing guarantees conversion succeeds");
                // Fail safe without panicking: return only the validated
                // prefix rather than risk inconsistent lengths below.
                return Cow::Borrowed(str::from_utf8_unchecked(head));
            }
        };
        debug_assert!(written <= spare_capacity.len());
        let new_len = old_len + written;
        debug_assert!(new_len <= vec.capacity());
        let new_len = new_len.min(vec.capacity());
        vec.set_len(new_len);
        let mut bytes = str::from_utf8_unchecked(spare_capacity[..new_len].assume_init_mut()).to_string();
        bytes.shrink_to_fit();
        Cow::Owned(bytes)
    }
}

pub fn ensure_utf16_validity(buffer: &mut [u16]) {
    let mut offset = 0;
    loop {
        offset += utf16_valid_up_to(&buffer[offset..]);
        if offset == buffer.len() {
            return;
        }
        buffer[offset] = 0xFFFD;
        offset += 1;
    }
}

pub fn utf16_valid_up_to(buffer: &[u16]) -> usize {
    let mut consumed = 0usize;
    'outer: loop {
        let (strides, tail) = &buffer[consumed..].as_chunks::<STRIDE>();
        let mut found_surrogate = false;
        for stride in strides.iter() {
            if let Some(pos) = validate_bmp_stride(stride) {
                consumed += pos;
                found_surrogate = true;
                break;
            }
            consumed += STRIDE;
        }
        if !found_surrogate {
            for slot in tail.iter() {
                if *slot & 0xF800 == 0xD800 {
                    found_surrogate = true;
                    break;
                }
                consumed += 1;
            }
            if !found_surrogate {
                debug_assert_eq!(consumed, buffer.len());
                return consumed;
            }
        }
        let mut unit = buffer[consumed];
        let mut unit_minus_surrogate_start = unit.wrapping_sub(0xD800);
        debug_assert!(unit_minus_surrogate_start <= (0xDFFF - 0xD800));
        'surrogate: loop {
            if unit_minus_surrogate_start > (0xDBFF - 0xD800) {
                return consumed;
            }
            let next = consumed + 1;
            if next == buffer.len() {
                return consumed;
            }
            let second = buffer[next];
            let second_minus_low_surrogate_start = second.wrapping_sub(0xDC00);
            if second_minus_low_surrogate_start > (0xDFFF - 0xDC00) {
                return consumed;
            }
            consumed = next + 1;
            loop {
                if consumed == buffer.len() {
                    return consumed;
                }
                unit = buffer[consumed];
                unit_minus_surrogate_start = unit.wrapping_sub(0xD800);
                if unit_minus_surrogate_start <= (0xDFFF - 0xD800) {
                    continue 'surrogate;
                }
                consumed += 1;
                if unit == 0x0020 {
                    continue;
                }
                continue 'outer;
            }
        }
    }
}

pub fn utf8_latin1_up_to(buffer: &[u8]) -> usize {
    is_utf8_latin1_impl(buffer).unwrap_or(buffer.len())
}
pub fn str_latin1_up_to(buffer: &str) -> usize {
    is_str_latin1_impl(buffer).unwrap_or(buffer.len())
}

fn is_utf8_latin1_impl(buffer: &[u8]) -> Option<usize> {
    let mut bytes = buffer;
    let mut total = 0;
    loop {
        if let Some((byte, offset)) = validate_ascii(bytes) {
            total += offset;
            if in_inclusive_range8(byte, 0xC2, 0xC3) {
                let next = offset + 1;
                if next == bytes.len() {
                    return Some(total);
                }
                if bytes[next] & 0xC0 != 0x80 {
                    return Some(total);
                }
                bytes = &bytes[offset + 2..];
                total += 2;
            } else {
                return Some(total);
            }
        } else {
            return None;
        }
    }
}

fn is_str_latin1_impl(buffer: &str) -> Option<usize> {
    let mut bytes = buffer.as_bytes();
    let mut total = 0;
    loop {
        if let Some((byte, offset)) = validate_ascii(bytes) {
            total += offset;
            if byte > 0xC3 {
                return Some(total);
            }
            bytes = &bytes[offset + 2..];
            total += 2;
        } else {
            return None;
        }
    }
}

const MAX_STRIDE_SIZE: usize = 16;
const STRIDE: usize = 16;

fn minimally_init<T>(slice: &mut [T]) -> &mut [T] {
    slice
}

fn validate_bmp_stride(stride: &[u16; STRIDE]) -> Option<usize> {
    if (stride[0] & 0xF800 != 0xD800)
        && (stride[1] & 0xF800 != 0xD800)
        && (stride[2] & 0xF800 != 0xD800)
        && (stride[3] & 0xF800 != 0xD800)
        && (stride[4] & 0xF800 != 0xD800)
        && (stride[5] & 0xF800 != 0xD800)
        && (stride[6] & 0xF800 != 0xD800)
        && (stride[7] & 0xF800 != 0xD800)
        && (stride[8] & 0xF800 != 0xD800)
        && (stride[9] & 0xF800 != 0xD800)
        && (stride[10] & 0xF800 != 0xD800)
        && (stride[11] & 0xF800 != 0xD800)
        && (stride[12] & 0xF800 != 0xD800)
        && (stride[13] & 0xF800 != 0xD800)
        && (stride[14] & 0xF800 != 0xD800)
        && (stride[15] & 0xF800 != 0xD800)
    {
        return None;
    }
    for (i, c) in stride.iter().enumerate() {
        if c & 0xF800 == 0xD800 {
            return Some(i);
        }
    }
    debug_assert!(false);
    None
}

fn validate_latin1_str_stride(stride: &[u8; STRIDE]) -> Option<usize> {
    if stride
        .iter()
        .all(|b| b & 0xC0 != 0x80 || b <= &0xC3)
    {
        return None;
    }
    for (i, b) in stride.iter().enumerate() {
        if *b >= 0x80 && (*b > 0xC3 || *b & 0xC0 == 0x80) {
            return Some(i);
        }
    }
    debug_assert!(false);
    None
}

pub fn convert_utf16_to_str_partial(src: &[u16], dst: &mut str) -> (usize, usize) {
    let bytes: &mut [u8] = unsafe { dst.as_bytes_mut() };
    let (read, written) = convert_utf16_to_utf8_partial(src, bytes);
    let len = bytes.len();
    let mut trail = written;
    while trail < len && ((bytes[trail] & 0xC0) == 0x80) {
        bytes[trail] = 0;
        trail += 1;
    }
    (read, written)
}

pub fn convert_utf16_to_str(src: &[u16], dst: &mut str) -> Result<usize, EncodingError> {
    let required = src.len().saturating_mul(3);
    if dst.len() < required {
        return Err(EncodingError::DestinationTooSmall {
            required,
            actual: dst.len(),
        });
    }
    let (read, written) = convert_utf16_to_str_partial(src, dst);
    debug_assert_eq!(read, src.len());
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    const fn latin1_table() -> [u16; 128] {
        let mut t = [0u16; 128];
        let mut i = 0usize;
        while i < 128 {
            t[i] = 0x80 + i as u16;
            i += 1;
        }
        t
    }

    const fn bad_latin1_table() -> [u16; 128] {
        let mut t = latin1_table();
        // Byte 0xC1 (= 0x80 + 0x41) has no mapping, so decoding it is malformed.
        t[0x41] = 0;
        t
    }

    static LATIN1_TABLE: [u16; 128] = latin1_table();
    static BAD_TABLE: [u16; 128] = bad_latin1_table();
    static ZERO_TABLE: [u16; 128] = [0u16; 128];

    // A latin-1-shaped identity encoding (byte N <-> U+00NN for N >= 0x80).
    // The run is deliberately empty (`run_length = 0`, byte offset 64): the
    // encoder then resolves every non-ASCII unit through the reverse table
    // search.
    static LATIN1: Encoding = Encoding {
        name: "latin1-test",
        variant: VariantEncoding::SingleByte(&LATIN1_TABLE, 0x80, 64, 0),
    };
    static LATIN1_DUP: Encoding = Encoding {
        name: "latin1-dup",
        variant: VariantEncoding::SingleByte(&LATIN1_TABLE, 0x80, 64, 0),
    };
    static BAD_LATIN1: Encoding = Encoding {
        name: "bad-latin1",
        variant: VariantEncoding::SingleByte(&BAD_TABLE, 0x80, 64, 0),
    };
    // A run-only encoding: the forward table is all zeroes (nothing decodes),
    // so any successful non-ASCII encode must come from the run map:
    // U+E000..U+E00F -> bytes 0xC0..0xCF.
    static RUN_ONLY: Encoding = Encoding {
        name: "run-only",
        variant: VariantEncoding::SingleByte(&ZERO_TABLE, 0xE000, 32, 16),
    };
    // Same table as LATIN1 but with the run covering the whole index from
    // `run_byte_offset = 0`, so a char outside the run must fall through every
    // quadrant search.
    static ZERO_RUN_LATIN1: Encoding = Encoding {
        name: "zero-run-latin1",
        variant: VariantEncoding::SingleByte(&LATIN1_TABLE, 0x80, 0, 0x80),
    };

    fn ok_size(res: Result<usize, EncodingError>) -> usize {
        match res {
            Ok(n) => n,
            Err(e) => panic!("conversion unexpectedly failed: {e:?}"),
        }
    }

    fn err_of(res: Result<usize, EncodingError>) -> EncodingError {
        match res {
            Err(e) => e,
            Ok(n) => panic!("conversion unexpectedly succeeded with {n}"),
        }
    }

    fn decode_all(d: &mut Decoder, src: &[u8]) -> (Vec<u8>, usize, bool) {
        let mut dst = vec![0u8; src.len() * 4 + 32];
        let (res, read, written, errors) = d.decode_to_utf8(src, &mut dst, true);
        assert!(
            matches!(res, CoderResult::InputEmpty),
            "decode did not finish: {src:?}"
        );
        dst.truncate(written);
        (dst, read, errors)
    }

    fn decode_all16(d: &mut Decoder, src: &[u8]) -> (Vec<u16>, bool) {
        let mut dst = vec![0u16; src.len() * 2 + 16];
        let (res, read, written, errors) = d.decode_to_utf16(src, &mut dst, true);
        assert!(
            matches!(res, CoderResult::InputEmpty),
            "decode16 did not finish: {src:?}"
        );
        assert_eq!(read, src.len());
        dst.truncate(written);
        (dst, errors)
    }

    fn decode_raw(d: &mut Decoder, src: &[u8], last: bool) -> (DecoderResult, usize, usize) {
        let mut dst = vec![0u8; src.len() * 4 + 32];
        d.decode_to_utf8_without_replacement(src, &mut dst, last)
    }

    fn encode_all(e: &mut Encoder, src: &str) -> (Vec<u8>, usize, bool) {
        let mut dst = vec![0u8; src.len() * 4 + 64];
        let (res, read, written, had) = e.encode_from_utf8(src, &mut dst, true);
        assert!(
            matches!(res, CoderResult::InputEmpty),
            "encode did not finish: {src:?}"
        );
        dst.truncate(written);
        (dst, read, had)
    }

    fn encode16_all(e: &mut Encoder, src: &[u16]) -> (Vec<u8>, bool) {
        let mut dst = vec![0u8; src.len() * 4 + 64];
        let (res, read, written, had) = e.encode_from_utf16(src, &mut dst, true);
        assert!(
            matches!(res, CoderResult::InputEmpty),
            "encode16 did not finish: {src:?}"
        );
        assert_eq!(read, src.len());
        dst.truncate(written);
        (dst, had)
    }

    fn utf16_units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn utf8_encoding_metadata() {
        assert_eq!(UTF_8.name, "UTF-8");
        assert_eq!(UTF_8.variant, VariantEncoding::Utf8);
        assert!(UTF_8.can_encode_everything());
    }

    #[test]
    fn utf16_encoding_metadata() {
        assert_eq!(UTF_16LE.name, "UTF-16LE");
        assert_eq!(UTF_16BE.name, "UTF-16BE");
        assert_eq!(UTF_16LE.variant, VariantEncoding::Utf16Le);
        assert_eq!(UTF_16BE.variant, VariantEncoding::Utf16Be);
        assert!(!UTF_16LE.can_encode_everything());
        assert!(!UTF_16BE.can_encode_everything());
    }

    #[test]
    fn single_byte_can_encode_everything_is_false() {
        assert!(!LATIN1.can_encode_everything());
        assert!(matches!(LATIN1.variant, VariantEncoding::SingleByte(..)));
    }

    #[test]
    fn encoding_returns_static_self() {
        assert!(core::ptr::eq(UTF_8.encoding(), &UTF_8));
        assert!(core::ptr::eq(LATIN1.encoding(), &LATIN1));
    }

    // The `PartialEq<Encoding> for &Encoding` impl requires the reference on
    // the left; clippy's op_ref suggestion (`LATIN1 == LATIN1_DUP`) would
    // switch to the derived impl, which also compares names, changing the
    // meaning of the assertions.
    #[allow(clippy::op_ref)]
    #[test]
    fn encoding_eq_compares_variant_not_name() {
        assert!(&LATIN1 == LATIN1_DUP);
        assert!(&UTF_8 == UTF_8);
        assert_ne!(&UTF_8, UTF_16BE);
        assert_ne!(&UTF_16LE, UTF_16BE);
        assert_ne!(&LATIN1, UTF_8);
    }

    #[test]
    fn new_decoder_starts_converting() {
        assert_eq!(UTF_8.new_decoder().life_cycle, DecoderLifeCycle::Converting);
        assert_eq!(
            UTF_8
                .new_decoder_without_bom_handling()
                .life_cycle,
            DecoderLifeCycle::Converting
        );
        assert_eq!(UTF_16LE.new_decoder().life_cycle, DecoderLifeCycle::Converting);
    }

    #[test]
    fn bom_removal_life_cycle_by_encoding() {
        assert_eq!(
            UTF_8.new_decoder_with_bom_removal().life_cycle,
            DecoderLifeCycle::AtUtf8Start
        );
        assert_eq!(
            UTF_16BE.new_decoder_with_bom_removal().life_cycle,
            DecoderLifeCycle::AtUtf16BeStart
        );
        assert_eq!(
            UTF_16LE.new_decoder_with_bom_removal().life_cycle,
            DecoderLifeCycle::AtUtf16LeStart
        );
        // Non-UTF encodings skip straight to Converting even with removal.
        assert_eq!(
            LATIN1.new_decoder_with_bom_removal().life_cycle,
            DecoderLifeCycle::Converting
        );
    }

    #[test]
    fn bom_sniff_life_cycle_is_at_start() {
        let d = Decoder::new(
            &UTF_8,
            VariantEncoding::Utf8.new_variant_decoder(),
            BomHandling::Sniff,
        );
        assert_eq!(d.life_cycle, DecoderLifeCycle::AtStart);
        assert!(d.max_utf8_buffer_length(16).is_none());
        assert!(
            d.max_utf8_buffer_length_without_replacement(16)
                .is_none()
        );
        assert!(d.max_utf16_buffer_length(16).is_none());
    }

    #[test]
    fn decoder_life_cycle_variants_are_distinct() {
        assert_eq!(DecoderLifeCycle::AtUtf8Start, DecoderLifeCycle::AtUtf8Start);
        assert_ne!(DecoderLifeCycle::SeenUtf8First, DecoderLifeCycle::Converting);
        assert_ne!(DecoderLifeCycle::Finished, DecoderLifeCycle::AtStart);
        assert_ne!(
            DecoderLifeCycle::ConvertingWithPendingBB,
            DecoderLifeCycle::Converting
        );
        assert_ne!(DecoderLifeCycle::AtUtf16BeStart, DecoderLifeCycle::AtUtf16LeStart);
        assert_ne!(
            DecoderLifeCycle::SeenUtf16BeFirst,
            DecoderLifeCycle::SeenUtf16LeFirst
        );
        assert_ne!(DecoderLifeCycle::AtStart, DecoderLifeCycle::SeenUtf8Second);
    }

    #[test]
    fn utf8_decoder_max_buffer_lengths_some_while_converting() {
        let d = UTF_8.new_decoder();
        assert!(d.max_utf8_buffer_length(10).is_some());
        assert!(
            d.max_utf8_buffer_length_without_replacement(10)
                .is_some()
        );
        assert!(d.max_utf16_buffer_length(10).is_some());
    }

    #[test]
    fn single_byte_decoder_max_lengths() {
        let d = LATIN1.new_decoder();
        assert_eq!(d.max_utf16_buffer_length(7), Some(7));
        assert_eq!(d.max_utf8_buffer_length(7), Some(21));
        assert_eq!(d.max_utf8_buffer_length_without_replacement(7), Some(21));
        // Multiplication overflow must be reported as `None`.
        assert_eq!(d.max_utf8_buffer_length(usize::MAX), None);
        assert_eq!(d.max_utf8_buffer_length(usize::MAX / 2,), None);
        assert_eq!(d.max_utf16_buffer_length(usize::MAX), Some(usize::MAX));
    }

    #[test]
    fn bom_removal_decoder_answers_max_lengths_like_plain() {
        let d = UTF_8.new_decoder_with_bom_removal();
        assert_eq!(
            d.max_utf8_buffer_length(4),
            UTF_8.new_decoder().max_utf8_buffer_length(4)
        );
        assert!(d.max_utf16_buffer_length(0).is_some());
    }

    #[test]
    fn decoder_encoding_accessor() {
        assert!(core::ptr::eq(UTF_16LE.new_decoder().encoding(), &UTF_16LE));
        assert!(core::ptr::eq(LATIN1.new_decoder().encoding(), &LATIN1));
    }

    #[test]
    fn utf8_bom_is_not_removed_by_current_impl() {
        // Documents actual behavior: `new_decoder_with_bom_removal` only moves
        // the life cycle; the decode path forwards to the variant decoder, so
        // the BOM survives in the output.
        let mut d = UTF_8.new_decoder_with_bom_removal();
        let src = [0xEFu8, 0xBB, 0xBF, b'a'];
        let (out, read, errors) = decode_all(&mut d, &src);
        assert_eq!(out, src);
        assert_eq!(read, 4);
        assert!(!errors);
    }

    #[test]
    fn utf8_bom_split_across_calls_is_preserved() {
        let mut d = UTF_8.new_decoder_with_bom_removal();
        let mut dst = vec![0u8; 16];
        let (res1, _, written1, _) = d.decode_to_utf8(&[0xEF], &mut dst, false);
        assert!(matches!(res1, CoderResult::InputEmpty));
        let rest = [0xBBu8, 0xBF, b'a'];
        let (res2, _, written2, _) = d.decode_to_utf8(&rest, &mut dst[written1..], true);
        assert!(matches!(res2, CoderResult::InputEmpty));
        assert_eq!(&dst[..written1 + written2], b"\xEF\xBB\xBFa");
    }

    #[test]
    fn utf16le_bom_survives_removal_decoder() {
        let mut d = UTF_16LE.new_decoder_with_bom_removal();
        // FF FE is U+FEFF in LE; decoded output is its UTF-8 form, so the BOM
        // survives (the current implementation does not strip it).
        let src = [0xFFu8, 0xFE, 0x61, 0x00];
        let (out, _, errors) = decode_all(&mut d, &src);
        assert_eq!(out, [0xEFu8, 0xBB, 0xBF, 0x61]);
        assert!(!errors);
    }

    #[test]
    fn utf16be_fe_ff_decodes_to_replacement_order_bom_char() {
        // `FE FF` read as UTF-16BE is U+FEFF (the BOM character), which the
        // decoder emits as its UTF-8 encoding rather than stripping.
        let mut d = UTF_16BE.new_decoder_with_bom_removal();
        let src = [0xFEu8, 0xFF, 0x00, 0x61];
        let (out, _, errors) = decode_all(&mut d, &src);
        assert_eq!(out, [0xEFu8, 0xBB, 0xBF, 0x61]);
        assert!(!errors);
    }

    #[test]
    fn utf8_decode_empty_input() {
        let mut d = UTF_8.new_decoder();
        let mut dst = [0u8; 8];
        let (res, read, written, errors) = d.decode_to_utf8(&[], &mut dst, true);
        assert_eq!(
            (res, read, written, errors),
            (CoderResult::InputEmpty, 0, 0, false)
        );
    }

    #[test]
    fn utf8_decode_empty_destination_reports_output_full() {
        let mut d = UTF_8.new_decoder();
        let mut dst: [u8; 0] = [];
        let (res, read, written) = d.decode_to_utf8_without_replacement(b"a", &mut dst, true);
        assert_eq!(res, DecoderResult::OutputFull);
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn utf8_decode_multibyte_identity() {
        let mut d = UTF_8.new_decoder();
        let src = "a\u{e9}\u{2603}\u{1F4A9}";
        let (out, read, errors) = decode_all(&mut d, src.as_bytes());
        assert_eq!(out, src.as_bytes());
        assert_eq!(read, src.len());
        assert!(!errors);
    }

    #[test]
    fn utf8_decode_truncated_two_byte_last_is_malformed() {
        let mut d = UTF_8.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0xC3], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf8_decode_truncated_two_byte_with_replacement() {
        let mut d = UTF_8.new_decoder();
        let (out, read, errors) = decode_all(&mut d, &[0x41u8, 0xC3]);
        assert!(errors);
        assert_eq!(out, [0x41u8, 0xEF, 0xBF, 0xBD]);
        assert_eq!(read, 2);
    }

    #[test]
    fn utf8_decode_truncated_two_byte_not_last_pends_until_final() {
        let mut d = UTF_8.new_decoder();
        let mut dst = vec![0u8; 16];
        let (res1, _, written1, errors1) = d.decode_to_utf8(&[0x41u8, 0xC3], &mut dst, false);
        assert!(matches!(res1, CoderResult::InputEmpty));
        assert!(!errors1);
        let (res2, _, written2, errors2) = d.decode_to_utf8(&[0xA9u8], &mut dst[written1..], true);
        assert!(matches!(res2, CoderResult::InputEmpty));
        assert!(!errors2);
        assert_eq!(&dst[..written1 + written2], b"A\xC3\xA9");
    }

    #[test]
    fn utf8_decode_lone_continuation_is_malformed() {
        let mut d = UTF_8.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0xA9], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf8_decode_overlong_sequence_is_malformed() {
        let mut d = UTF_8.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0xC0, 0xAF], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf8_decode_surrogate_encoding_is_malformed() {
        let mut d = UTF_8.new_decoder();
        // ED A0 80 would decode to U+D800, a surrogate: forbidden by the spec.
        let (res, _, _) = decode_raw(&mut d, &[0xED, 0xA0, 0x80], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf8_decode_d7ff_is_valid() {
        let mut d = UTF_8.new_decoder();
        // ED 9F BF is U+D7FF, the last valid non-surrogate three-byte point.
        let (res, _, _) = decode_raw(&mut d, &[0xED, 0x9F, 0xBF], true);
        assert_eq!(res, DecoderResult::InputEmpty);
    }

    #[test]
    fn utf8_decode_invalid_lead_f5_is_malformed() {
        let mut d = UTF_8.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0xF5, 0x80, 0x80, 0x80], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf8_decode_bad_second_byte_of_three_is_malformed() {
        let mut d = UTF_8.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0xE2, 0x41], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf8_decode_max_valid_point_succeeds() {
        let mut d = UTF_8.new_decoder();
        let (res, read, written) = decode_raw(&mut d, &[0xF4, 0x8F, 0xBF, 0xBF], true);
        assert_eq!(res, DecoderResult::InputEmpty);
        assert_eq!((read, written), (4, 4));
    }

    #[test]
    fn utf8_decode_malformed_position_between_ascii() {
        let mut d = UTF_8.new_decoder();
        let (out, read, errors) = decode_all(&mut d, &[0x41u8, 0xFF, 0x42]);
        assert!(errors);
        assert_eq!(out, [0x41u8, 0xEF, 0xBF, 0xBD, 0x42]);
        assert_eq!(read, 3);
    }

    #[test]
    fn utf8_decode_to_utf16_astral_pair() {
        let mut d = UTF_8.new_decoder();
        let src = "\u{1F4A9}".as_bytes();
        let (units, errors) = decode_all16(&mut d, src);
        assert!(!errors);
        assert_eq!(units, utf16_units("\u{1F4A9}"));
    }

    #[test]
    fn utf8_decode_to_utf16_malformed_replacement_unit() {
        let mut d = UTF_8.new_decoder();
        let (units, errors) = decode_all16(&mut d, &[0x41u8, 0xFF]);
        assert!(errors);
        assert_eq!(units, [0x0041, 0xFFFD]);
    }

    #[test]
    fn utf8_decode_to_utf16_output_full_on_tiny_destination() {
        let astral = "\u{1F4A9}";
        let mut d = UTF_8.new_decoder();
        let mut dst = [0u16; 1];
        let (res, _, _, _) = d.decode_to_utf16(astral.as_bytes(), &mut dst, true);
        assert_eq!(res, CoderResult::OutputFull);
    }

    #[test]
    fn utf16le_decode_ascii_to_utf8() {
        let mut d = UTF_16LE.new_decoder();
        let (out, read, errors) = decode_all(&mut d, b"a\x00b\x00");
        assert_eq!(out, *b"ab");
        assert_eq!(read, 4);
        assert!(!errors);
    }

    #[test]
    fn utf16be_decode_ascii_to_utf8() {
        let mut d = UTF_16BE.new_decoder();
        let (out, read, errors) = decode_all(&mut d, b"\x00a\x00b");
        assert_eq!(out, *b"ab");
        assert_eq!(read, 4);
        assert!(!errors);
    }

    #[test]
    fn utf16le_odd_byte_not_last_pends_then_completes() {
        let mut d = UTF_16LE.new_decoder();
        let mut dst = vec![0u8; 16];
        let (res1, written1, _) = {
            let (res, _, written, errors) = d.decode_to_utf8(&[0x61u8], &mut dst, false);
            (res, written, errors)
        };
        assert!(matches!(res1, CoderResult::InputEmpty));
        assert_eq!(written1, 0);
        let (res2, _, written2, _) = d.decode_to_utf8(&[0x00u8], &mut dst[written1..], true);
        assert!(matches!(res2, CoderResult::InputEmpty));
        assert_eq!(&dst[..written1 + written2], b"a");
    }

    #[test]
    fn utf16le_final_odd_byte_is_malformed() {
        let mut d = UTF_16LE.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0x61u8, 0x00, 0x62], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf16be_final_odd_byte_is_malformed() {
        let mut d = UTF_16BE.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0x00, 0x61u8, 0x00], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf16le_lone_high_surrogate_raw_is_malformed() {
        let mut d = UTF_16LE.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0x3D, 0xD8], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf16le_lone_low_surrogate_raw_is_malformed() {
        let mut d = UTF_16LE.new_decoder();
        let (res, _, _) = decode_raw(&mut d, &[0xA9, 0xDC], true);
        assert!(matches!(res, DecoderResult::Malformed(..)));
    }

    #[test]
    fn utf16le_lone_surrogate_with_replacement() {
        let mut d = UTF_16LE.new_decoder();
        let (out, read, errors) = decode_all(&mut d, &[0x3D, 0xD8, 0x61, 0x00]);
        assert!(errors);
        assert_eq!(out, [0xEFu8, 0xBF, 0xBD, 0x61]);
        assert_eq!(read, 4);
    }

    #[test]
    fn utf16be_lone_surrogate_with_replacement() {
        let mut d = UTF_16BE.new_decoder();
        let (out, _, errors) = decode_all(&mut d, &[0xD8, 0x3D, 0x00, 0x61]);
        assert!(errors);
        assert_eq!(out, [0xEFu8, 0xBF, 0xBD, 0x61]);
    }

    #[test]
    fn utf16le_astral_pair_decodes_to_utf8() {
        let mut d = UTF_16LE.new_decoder();
        let src: Vec<u8> = utf16_units("\u{1F4A9}")
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let (out, read, errors) = decode_all(&mut d, &src);
        assert!(!errors);
        assert_eq!(out, "\u{1F4A9}".as_bytes());
        assert_eq!(read, src.len());
    }
}
