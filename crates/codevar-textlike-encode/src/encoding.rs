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
            DecoderResult::Malformed(_, _) => {
                let Some(slot) = dst.get_mut(total_written) else {
                    return Err(EncodingError::OutputFull);
                };
                *slot = 0xFFFD;
                total_written += 1;
            }
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

    #[test]
    fn utf16be_astral_pair_decodes_to_utf8() {
        let mut d = UTF_16BE.new_decoder();
        let src: Vec<u8> = utf16_units("\u{1F4A9}")
            .iter()
            .flat_map(|u| u.to_be_bytes())
            .collect();
        let (out, _, errors) = decode_all(&mut d, &src);
        assert!(!errors);
        assert_eq!(out, "\u{1F4A9}".as_bytes());
    }

    #[test]
    fn utf16_endianness_changes_meaning() {
        let src = [0x41u8, 0x00];
        let mut le = UTF_16LE.new_decoder();
        let mut be = UTF_16BE.new_decoder();
        let (le_out, _, le_err) = decode_all(&mut le, &src);
        let (be_out, _, be_err) = decode_all(&mut be, &src);
        assert!(!le_err && !be_err);
        assert_eq!(le_out, *b"A");
        // The same bytes as BE are U+4100, a three-byte UTF-8 sequence.
        assert_eq!(be_out, "\u{4100}".as_bytes());
        assert_ne!(le_out, be_out);
    }

    #[test]
    fn utf16le_decode_to_utf16_bmp_units_preserved() {
        let mut d = UTF_16LE.new_decoder();
        let src: Vec<u8> = utf16_units("ab\u{e9}")
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let (units, errors) = decode_all16(&mut d, &src);
        assert!(!errors);
        assert_eq!(units, utf16_units("ab\u{e9}"));
    }

    #[test]
    fn utf16le_decode_to_utf16_keeps_both_units_of_an_astral_pair() {
        let mut d = UTF_16LE.new_decoder();
        let src: Vec<u8> = utf16_units("a\u{e9}\u{1F4A9}")
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let (units, errors) = decode_all16(&mut d, &src);
        assert!(!errors);
        assert_eq!(units, utf16_units("a\u{e9}\u{1F4A9}"));
    }

    #[test]
    fn utf16_decode_output_full_then_resume() {
        let mut d = UTF_16LE.new_decoder();
        let src = [0x61u8, 0x00, 0x62, 0x00];
        let mut dst = [0u16; 1];
        let (res, read, written, _) = d.decode_to_utf16(&src, &mut dst, true);
        assert_eq!(res, CoderResult::OutputFull);
        assert_eq!((read, written), (2, 1));
        assert_eq!(dst[0], 0x0061);
        let mut dst2 = [0u16; 2];
        let (res2, read2, written2, _) = d.decode_to_utf16(&src[read..], &mut dst2, true);
        assert_eq!(res2, CoderResult::InputEmpty);
        assert_eq!((read2, written2), (2, 1));
        assert_eq!(dst2[0], 0x0062);
    }

    #[test]
    fn utf16_exactly_fitting_destination_is_input_empty() {
        let mut d = UTF_16BE.new_decoder();
        let mut dst = [0u16; 1];
        let (res, read, written, errors) = d.decode_to_utf16(&[0x00u8, 0x61], &mut dst, true);
        assert_eq!(res, CoderResult::InputEmpty);
        assert_eq!((read, written, errors), (2, 1, false));
        assert_eq!(dst[0], 0x0061);
    }

    #[test]
    fn utf16_decode_empty_input_and_empty_destination() {
        let mut d = UTF_16LE.new_decoder();
        let mut dst = [0u16; 4];
        let (res, read, written, errors) = d.decode_to_utf16(&[], &mut dst, true);
        assert_eq!(
            (res, read, written, errors),
            (CoderResult::InputEmpty, 0, 0, false)
        );
        let mut empty: [u16; 0] = [];
        let (res2, _, _) = d.decode_to_utf16_without_replacement(&[0x61, 0x00], &mut empty, true);
        assert_eq!(res2, DecoderResult::OutputFull);
    }

    #[test]
    fn utf8_encoder_str_roundtrip_identity() {
        let mut e = UTF_8.new_encoder();
        let src = "a\u{e9}\u{2603}\u{1F4A9}";
        let (out, read, had) = encode_all(&mut e, src);
        assert_eq!(out, src.as_bytes());
        assert_eq!(read, src.len());
        assert!(!had);
    }

    #[test]
    fn utf8_encoder_empty_input() {
        let mut e = UTF_8.new_encoder();
        let mut dst = [0u8; 8];
        let (res, read, written, had) = e.encode_from_utf8("", &mut dst, true);
        assert_eq!((res, read, written, had), (CoderResult::InputEmpty, 0, 0, false));
    }

    #[test]
    fn utf8_encoder_from_utf16_matches_std_bytes() {
        let mut e = UTF_8.new_encoder();
        let src = utf16_units("h\u{e9}\u{1F4A9}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(!had);
        let reference: String = core::char::decode_utf16(src.clone())
            .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect();
        assert_eq!(out, reference.as_bytes());
    }

    #[test]
    fn utf8_encoder_small_destination_needs_no_ncr_reserve() {
        // UTF-8 can encode everything, so unlike other encodings it does not
        // require `NCR_EXTRA` spare bytes before attempting progress.
        let mut e = UTF_8.new_encoder();
        let mut dst = [0u8; 5];
        let (res, read, written, _) = e.encode_from_utf8("abcdefghij", &mut dst, true);
        assert_eq!(res, CoderResult::OutputFull);
        assert!(written > 0 && written <= 5);
        assert!(read >= written);
    }

    #[test]
    fn utf16le_encoder_ascii_bytes_are_little_endian() {
        let mut e = UTF_16LE.new_encoder();
        let (out, read, had) = encode_all(&mut e, "a");
        assert_eq!(out, [0x61u8, 0x00]);
        assert_eq!(read, 1);
        assert!(!had);
    }

    #[test]
    fn utf16be_encoder_ascii_bytes_are_big_endian() {
        let mut e = UTF_16BE.new_encoder();
        let (out, _, had) = encode_all(&mut e, "a");
        assert_eq!(out, [0x00u8, 0x61]);
        assert!(!had);
    }

    #[test]
    fn utf16le_encoder_from_utf8_astral_produces_pair() {
        let mut e = UTF_16LE.new_encoder();
        let (out, _, had) = encode_all(&mut e, "\u{1F4A9}");
        assert!(!had);
        let expected: Vec<u8> = utf16_units("\u{1F4A9}")
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn utf16be_encoder_from_utf16_astral_produces_pair() {
        let mut e = UTF_16BE.new_encoder();
        let src = utf16_units("\u{1F600}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(!had);
        let expected: Vec<u8> = src.iter().flat_map(|u| u.to_be_bytes()).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn encoder_encoding_accessor_and_pending_state() {
        let e = UTF_16BE.new_encoder();
        assert!(core::ptr::eq(e.encoding(), &UTF_16BE));
        assert!(!e.has_pending_state());
        let mut sb = LATIN1.new_encoder();
        let mut dst = vec![0u8; 4];
        let (res, _, _, _) = sb.encode_from_utf8("abcdefgh", &mut dst, false);
        assert!(matches!(res, CoderResult::OutputFull));
        assert!(!sb.has_pending_state());
    }

    #[test]
    fn utf8_if_no_unmappables_equals_without_replacement() {
        let e = UTF_8.new_encoder();
        assert_eq!(
            e.max_buffer_length_from_utf8_if_no_unmappables(10),
            e.max_buffer_length_from_utf8_without_replacement(10)
        );
        assert_eq!(
            e.max_buffer_length_from_utf16_if_no_unmappables(10),
            e.max_buffer_length_from_utf16_without_replacement(10)
        );
    }

    #[test]
    fn non_utf8_if_no_unmappables_adds_ncr_extra() {
        for enc in [&UTF_16LE, &UTF_16BE, &LATIN1] {
            let e = enc.new_encoder();
            let without = e.max_buffer_length_from_utf8_without_replacement(8);
            assert_eq!(
                e.max_buffer_length_from_utf8_if_no_unmappables(8),
                without.map(|v| v + NCR_EXTRA)
            );
            let without16 = e.max_buffer_length_from_utf16_without_replacement(8);
            assert_eq!(
                e.max_buffer_length_from_utf16_if_no_unmappables(8),
                without16.map(|v| v + NCR_EXTRA)
            );
        }
    }

    #[test]
    fn encoder_max_buffer_length_overflow_is_none() {
        let e = UTF_16LE.new_encoder();
        assert_eq!(
            e.max_buffer_length_from_utf8_without_replacement(usize::MAX),
            None
        );
        assert_eq!(e.max_buffer_length_from_utf8_if_no_unmappables(usize::MAX), None);
        let sb = LATIN1.new_encoder();
        assert_eq!(sb.max_buffer_length_from_utf8_if_no_unmappables(usize::MAX), None);
    }

    #[test]
    fn variant_encoder_and_decoder_passthrough() {
        let mut vd = VariantEncoding::Utf8.new_variant_decoder();
        let mut dst = [0u8; 8];
        let (res, read, written) = vd.decode_to_utf8_raw(b"hi", &mut dst, true);
        assert_eq!(res, DecoderResult::InputEmpty);
        assert_eq!((read, written), (2, 2));
        assert_eq!(
            vd.max_utf8_buffer_length(10),
            UTF_8.new_decoder().max_utf8_buffer_length(10)
        );
        let mut ve = VariantEncoder::Utf8(Utf8Encoder);
        assert_eq!(
            ve.max_buffer_length_from_utf8_without_replacement(10),
            UTF_8
                .new_encoder()
                .max_buffer_length_from_utf8_without_replacement(10)
        );
        assert!(!ve.has_pending_state());
        let (res, _, _) = ve.encode_from_utf8_raw("", &mut dst, true);
        assert_eq!(res, EncoderResult::InputEmpty);
    }

    #[test]
    fn ncr_extra_constant() {
        assert_eq!(NCR_EXTRA, 10);
    }

    #[test]
    fn single_byte_decode_ascii_passthrough() {
        let mut d = LATIN1.new_decoder();
        let (out, read, errors) = decode_all(&mut d, b"Hello");
        assert_eq!(out, *b"Hello");
        assert_eq!(read, 5);
        assert!(!errors);
    }

    #[test]
    fn single_byte_decode_full_mapped_range() {
        let src: Vec<u8> = (0x80u8..=0xFF).collect();
        let mut expected: Vec<u8> = Vec::new();
        for b in &src {
            let c = char::from_u32(u32::from(*b));
            let mut buf = [0u8; 4];
            expected.extend_from_slice(
                c.unwrap_or(char::REPLACEMENT_CHARACTER)
                    .encode_utf8(&mut buf)
                    .as_bytes(),
            );
        }
        let mut d = LATIN1.new_decoder();
        let (out, read, errors) = decode_all(&mut d, &src);
        assert!(!errors);
        assert_eq!(read, src.len());
        assert_eq!(out, expected);
    }

    #[test]
    fn single_byte_unmapped_table_entry_raw_is_malformed() {
        let mut d = BAD_LATIN1.new_decoder();
        // 0xC1 = 0x80 + 0x41, the entry zeroed in the table.
        let (res, _, _) = decode_raw(&mut d, &[0xC1], true);
        assert_eq!(res, DecoderResult::Malformed(1, 0));
    }

    #[test]
    fn single_byte_malformed_at_start_with_replacement() {
        let mut d = BAD_LATIN1.new_decoder();
        let (out, read, errors) = decode_all(&mut d, &[0xC1u8, 0x41]);
        assert!(errors);
        assert_eq!(out, [0xEFu8, 0xBF, 0xBD, 0x41]);
        assert_eq!(read, 2);
    }

    #[test]
    fn single_byte_malformed_at_end_with_replacement() {
        let mut d = BAD_LATIN1.new_decoder();
        let (out, read, errors) = decode_all(&mut d, &[0x41u8, 0xC1]);
        assert!(errors);
        assert_eq!(out, [0x41u8, 0xEF, 0xBF, 0xBD]);
        assert_eq!(read, 2);
    }

    #[test]
    fn run_only_encoder_uses_run_map_from_utf16() {
        // The forward table is all zeroes, so a successful non-ASCII encode
        // can only come from the run map (U+E000..U+E00F -> bytes
        // 128 + run_byte_offset(32) + offset = 0xA0..0xAF).
        let mut e = RUN_ONLY.new_encoder();
        let src = utf16_units("\u{E000}\u{E007}\u{E00F}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(!had);
        assert_eq!(out, [0xA0u8, 0xA7, 0xAF]);
    }

    #[test]
    fn run_only_encoder_char_past_run_end_is_unmappable() {
        // U+E00F is the last code point in the run; U+E010 sits one past it
        // and the all-zero table has no entry either.
        let mut e = RUN_ONLY.new_encoder();
        let src = utf16_units("\u{E010}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(had);
        assert_eq!(out, format!("&#{};", '\u{E010}' as u32).into_bytes());
    }

    #[test]
    fn single_byte_encoder_run_below_the_second_quadrant_stays_in_bounds() {
        // The second-quadrant reverse search starts at the run's byte offset, so
        // a run that begins below pointer 0x20 (the ISO-8859-x indexes run from
        // pointer 0) must clamp that range instead of slicing `table[32..0]`.
        let mut e = ZERO_RUN_LATIN1.new_encoder();
        let src = utf16_units("\u{4E2D}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(had);
        assert_eq!(out, format!("&#{};", '\u{4E2D}' as u32).into_bytes());

        let mut in_run = ZERO_RUN_LATIN1.new_encoder();
        let src = utf16_units("\u{00E9}");
        let (out, had) = encode16_all(&mut in_run, &src);
        assert!(!had);
        assert_eq!(out, [0xE9u8]);
    }

    #[test]
    fn single_byte_encode_ascii_passthrough() {
        let mut e = LATIN1.new_encoder();
        let (out, read, had) = encode_all(&mut e, "abc");
        assert_eq!(out, [0x61u8, 0x62, 0x63]);
        assert_eq!(read, 3);
        assert!(!had);
    }

    #[test]
    fn single_byte_encode_run_map_from_utf8() {
        let mut e = LATIN1.new_encoder();
        let (out, _, had) = encode_all(&mut e, "caf\u{e9}");
        assert!(!had);
        assert_eq!(out, [0x63u8, 0x61, 0x66, 0xE9]);
    }

    #[test]
    fn single_byte_encode_unmappable_bmp_produces_ncr() {
        // The UTF-16 input path consumes the unmappable unit and reports it,
        // so the NCR appears exactly once.
        let mut e = LATIN1.new_encoder();
        let src = utf16_units("\u{4E2D}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(had);
        let expected = format!("&#{};", '\u{4E2D}' as u32);
        assert_eq!(out, expected.as_bytes());
    }

    #[test]
    fn single_byte_encode_unmappable_astral_produces_six_digit_ncr() {
        let mut e = LATIN1.new_encoder();
        let src = utf16_units("\u{1D11E}");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(had);
        let expected = format!("&#{};", 0x1D11Eu32);
        assert_eq!(out, expected.as_bytes());
        assert_eq!(expected.len(), 9);
    }

    #[test]
    fn single_byte_encode_ncr_between_ascii_bytes() {
        let mut e = LATIN1.new_encoder();
        let src = utf16_units("a\u{4E2D}b");
        let (out, had) = encode16_all(&mut e, &src);
        assert!(had);
        let mut expected = vec![b'a'];
        expected.extend_from_slice(format!("&#{};", '\u{4E2D}' as u32).as_bytes());
        expected.push(b'b');
        assert_eq!(out, expected);
    }

    #[test]
    fn single_byte_unmappable_from_utf8_raw_does_not_consume_current_behavior() {
        // Documents actual behavior (reported suspect): the single-byte
        // encoder's UTF-8 path returns `Unmappable` with read = 0, never
        // advancing past the offending character, unlike the UTF-16 path.
        let mut e = LATIN1.new_encoder();
        let mut dst = [0u8; 40];
        let (res, read, written) = e.encode_from_utf8_without_replacement("\u{4E2D}", &mut dst, true);
        assert_eq!(res, EncoderResult::Unmappable('\u{4E2D}'));
        assert_eq!((read, written), (0, 0));
    }

    #[test]
    fn single_byte_encode_from_utf8_unmappable_stalls_to_output_full_current_behavior() {
        // Because the UTF-8 raw path above never consumes the unmappable
        // char, the replacement-char NCR loop in `encode_from_utf8` rewrites
        // the same NCR until the effective destination is exhausted.
        let mut e = LATIN1.new_encoder();
        let mut dst = [0u8; 64];
        let (res, _, written, had) = e.encode_from_utf8("\u{4E2D}", &mut dst, true);
        assert!(had);
        assert_eq!(res, CoderResult::OutputFull);
        assert!(written >= 7);
    }

    #[test]
    fn single_byte_encode_destination_below_ncr_reserve_is_output_full() {
        let mut e = LATIN1.new_encoder();
        let mut dst = [0u8; 9];
        let (res, read, written, had) = e.encode_from_utf8("a", &mut dst, true);
        assert_eq!(res, CoderResult::OutputFull);
        assert_eq!((read, written, had), (0, 0, false));
    }

    #[test]
    fn single_byte_encode_empty_src_small_dst_is_input_empty() {
        let mut e = LATIN1.new_encoder();
        let mut dst = [0u8; 4];
        let (res, read, written, had) = e.encode_from_utf8("", &mut dst, false);
        assert_eq!((res, read, written, had), (CoderResult::InputEmpty, 0, 0, false));
        let (res2, read2, written2, had2) = e.encode_from_utf16(&[], &mut dst, true);
        assert_eq!(
            (res2, read2, written2, had2),
            (CoderResult::InputEmpty, 0, 0, false)
        );
    }

    #[test]
    fn single_byte_decode_encode_roundtrip_all_mapped_bytes() {
        let mut d = LATIN1.new_decoder();
        let mut e = LATIN1.new_encoder();
        for b in 0x80u8..=0xFF {
            let (utf8, _, errors) = decode_all(&mut d, &[b]);
            assert!(!errors);
            let s = core::str::from_utf8(&utf8);
            let text = match s {
                Ok(t) => t,
                Err(_) => panic!("decoder emitted invalid UTF-8 for byte {b:02X}"),
            };
            let (back, _, had) = encode_all(&mut e, text);
            assert!(!had);
            assert_eq!(back, [b]);
        }
    }

    #[test]
    fn write_ncr_matches_decimal_reference_at_digit_boundaries() {
        let points = [10u32, 99, 100, 999, 1000, 9999, 10000, 99999, 100000, 1114111];
        for p in points {
            let c = char::from_u32(p);
            let Some(c) = c else {
                panic!("{p} is not a char");
            };
            let reference = format!("&#{p};");
            let mut dst = [0u8; 16];
            let len = write_ncr(c, &mut dst);
            assert_eq!(len, reference.len());
            assert_eq!(&dst[..len], reference.as_bytes());
        }
    }

    #[test]
    fn write_ncr_single_digit_leaves_interior_byte_current_behavior() {
        // Documents actual behavior: for a code point below 100 the fixed
        // 5-byte layout only fills two digit slots, so a one-digit value
        // leaves the interior byte untouched (suspect noted in `write_ncr`,
        // unreachable from the encoders because ASCII never unmaps).
        let mut dst = [0u8; 16];
        let len = write_ncr('\u{1}', &mut dst);
        assert_eq!(len, 5);
        assert_eq!(&dst[..len], &[b'&', b'#', 0, b'1', b';']);
    }

    #[test]
    fn encoder_result_unmappable_from_bmp_helpers() {
        assert_eq!(
            EncoderResult::unmappable_from_bmp(0x4E2D),
            EncoderResult::Unmappable('\u{4E2D}')
        );
        // Lone surrogates cannot become chars; they degrade to U+FFFD.
        assert_eq!(
            EncoderResult::unmappable_from_bmp(0xD800),
            EncoderResult::Unmappable(char::REPLACEMENT_CHARACTER)
        );
        assert_eq!(
            EncoderResult::unmappable_from_bmp(0xDFFF),
            EncoderResult::Unmappable(char::REPLACEMENT_CHARACTER)
        );
    }

    #[test]
    fn coder_enum_variants_equality() {
        assert_eq!(CoderResult::InputEmpty, CoderResult::InputEmpty);
        assert_ne!(CoderResult::InputEmpty, CoderResult::OutputFull);
        assert_ne!(DecoderResult::Malformed(1, 0), DecoderResult::Malformed(0, 1));
        assert_eq!(DecoderResult::Malformed(2, 2), DecoderResult::Malformed(2, 2));
        assert_ne!(EncoderResult::OutputFull, EncoderResult::InputEmpty);
        assert_eq!(EncoderResult::Unmappable('a'), EncoderResult::Unmappable('a'));
        assert_ne!(EncoderResult::Unmappable('a'), EncoderResult::Unmappable('b'));
    }

    #[test]
    fn encoding_error_display_and_equality() {
        let small = EncodingError::DestinationTooSmall {
            required: 3,
            actual: 2,
        };
        assert_eq!(
            format!("{small}"),
            "destination buffer too small: requires 3 elements, got 2"
        );
        assert_eq!(small, small.clone());
        assert_ne!(small, EncodingError::OutputFull);
        assert_eq!(
            format!("{}", EncodingError::OutputFull),
            "destination buffer filled before conversion completed"
        );
    }

    #[test]
    fn convert_utf8_to_utf16_valid_mixed_planes() {
        let src = "abc\u{1F4A9}\u{2603}";
        let mut dst = vec![0u16; src.len() + 1];
        let len = ok_size(convert_utf8_to_utf16(src.as_bytes(), &mut dst));
        dst.truncate(len);
        assert_eq!(dst, utf16_units(src));
    }

    #[test]
    fn convert_utf8_to_utf16_destination_must_exceed_source_len() {
        let src = b"abc";
        let mut dst = vec![0u16; 3];
        let err = err_of(convert_utf8_to_utf16(src, &mut dst));
        assert_eq!(
            err,
            EncodingError::DestinationTooSmall {
                required: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn convert_utf8_to_utf16_replaces_a_bad_lead_byte_without_clobbering_the_next_unit() {
        // A malformed sequence writes exactly one replacement unit: the unit
        // that follows it must still be the decoded next character, not the
        // zero half of a widened write.
        let src = b"\xFFa\xFFb";
        let mut dst = vec![0u16; src.len() + 1];
        let len = ok_size(convert_utf8_to_utf16(src, &mut dst));
        dst.truncate(len);
        assert_eq!(dst, [0xFFFDu16, 0x61, 0xFFFD, 0x62]);
    }

    #[test]
    fn convert_utf8_to_utf16_replaces_a_truncated_tail_once() {
        for truncated in [b"\xC3".as_slice(), b"\xE2\x98", b"\xF0\x9F\x92", b"ok\xED\xA0"] {
            let mut dst = vec![0u16; truncated.len() + 1];
            let len = ok_size(convert_utf8_to_utf16(truncated, &mut dst));
            assert!(len <= dst.len(), "{truncated:?} overran the destination");
            dst.truncate(len);
            let tail = &dst[dst.len() - 1..];
            assert_eq!(tail, [0xFFFDu16], "{truncated:?}");
        }
    }

    #[test]
    fn convert_utf8_to_utf16_reprocesses_a_bad_trail_byte_as_a_lead() {
        // The decoder reports the offending trail byte without consuming it, so
        // the conversion sees the 0x28 twice: once as the error and once as the
        // valid ascii character it is.
        let src = b"\xC3\x28";
        let mut dst = vec![0u16; src.len() + 1];
        let len = ok_size(convert_utf8_to_utf16(src, &mut dst));
        dst.truncate(len);
        assert_eq!(dst, [0xFFFDu16, 0x28]);
    }

    #[test]
    fn convert_utf8_to_utf16_stays_inside_the_destination_for_all_malformed_input() {
        let src = [0xEDu8, 0xA0, 0x80, 0xC0, 0x80, 0xF5, 0xFF];
        let mut dst = vec![0u16; src.len() + 1];
        let len = ok_size(convert_utf8_to_utf16(&src, &mut dst));
        assert!(len <= dst.len());
        dst.truncate(len);
        assert_eq!(dst, vec![0xFFFDu16; src.len()]);
        assert_eq!(len, src.len());
    }

    #[test]
    fn convert_str_to_utf16_valid_paths() {
        let src = "abc\u{1F4A9}";
        let mut dst = vec![0u16; src.len()];
        let len = ok_size(convert_str_to_utf16(src, &mut dst));
        dst.truncate(len);
        assert_eq!(dst, utf16_units(src));
        let mut ascii_dst = vec![0u16; 3];
        let len2 = ok_size(convert_str_to_utf16("abc", &mut ascii_dst));
        assert_eq!(len2, 3);
        assert_eq!(ascii_dst.as_slice(), &[0x61u16, 0x62, 0x63]);
    }

    #[test]
    fn convert_str_to_utf16_destination_too_small() {
        let mut dst: Vec<u16> = vec![0; 2];
        let err = err_of(convert_str_to_utf16("abc", &mut dst[..]));
        assert_eq!(
            err,
            EncodingError::DestinationTooSmall {
                required: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn convert_utf16_to_utf8_valid_and_too_small() {
        let src: Vec<u16> = "abc\u{1F4A9}".encode_utf16().collect();
        let mut dst: Vec<u8> = vec![0; src.len() * 3];
        let len = ok_size(convert_utf16_to_utf8(&src[..], &mut dst[..]));
        dst.truncate(len);
        assert_eq!(&dst[..], "abc\u{1F4A9}".as_bytes());
        let hello: Vec<u16> = "hello".encode_utf16().collect();
        let err = err_of(convert_utf16_to_utf8(&hello[..], &mut [0u8; 4]));
        assert_eq!(
            err,
            EncodingError::DestinationTooSmall {
                required: 15,
                actual: 4
            }
        );
    }

    #[test]
    fn convert_utf16_to_utf8_partial_stops_and_resumes() {
        let text = "ab\u{2603}\u{1F4A9}";
        let expected = text.as_bytes().to_vec();
        let src = utf16_units(text);
        let mut small = vec![0u8; 4];
        let (r1, w1) = convert_utf16_to_utf8_partial(&src, &mut small);
        assert!(w1 <= 4);
        assert!(r1 < src.len());
        let mut full = vec![0u8; 3 * src.len()];
        full[..w1].copy_from_slice(&small[..w1]);
        let (r2, w2) = convert_utf16_to_utf8_partial(&src[r1..], &mut full[w1..]);
        assert_eq!((r1 + r2, w1 + w2), (src.len(), expected.len()));
        assert_eq!(&full[..w1 + w2], &expected[..]);
    }

    #[test]
    fn convert_utf16_to_utf8_partial_empty() {
        let (read, written) = convert_utf16_to_utf8_partial(&[], &mut [0u8; 8]);
        assert_eq!((read, written), (0, 0));
        assert_eq!(ok_size(convert_utf16_to_utf8(&[], &mut [])), 0);
    }

    #[test]
    fn convert_utf16_to_str_writes_valid_utf8() {
        let text = "a\u{e9}\u{1F4A9}";
        let src = utf16_units(text);
        let required = src.len() * 3;
        let mut holder: String = "\u{0}".repeat(required);
        let len = ok_size(convert_utf16_to_str(&src, holder.as_mut_str()));
        assert_eq!(len, text.len());
        assert_eq!(&holder.as_bytes()[..len], text.as_bytes());
    }

    #[test]
    fn convert_utf16_to_str_destination_too_small() {
        let src = utf16_units("abc");
        let mut holder: String = "\u{0}".repeat(5);
        let err = err_of(convert_utf16_to_str(&src, holder.as_mut_str()));
        assert_eq!(
            err,
            EncodingError::DestinationTooSmall {
                required: 9,
                actual: 5
            }
        );
    }

    #[test]
    fn convert_latin1_to_utf8_matches_std() {
        let src = [0x41u8, 0xE9, 0xFF, 0x80];
        let mut expected: Vec<u8> = Vec::new();
        for b in &src {
            let c = char::from_u32(u32::from(*b));
            let Some(c) = c else {
                panic!("{b} has no char");
            };
            let mut buf = [0u8; 4];
            expected.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
        let mut dst = vec![0u8; src.len() * 2];
        let written = convert_latin1_to_utf8(&src, &mut dst);
        assert_eq!(written, expected.len());
        assert_eq!(&dst[..written], &expected[..]);
    }

    #[test]
    fn convert_latin1_to_utf8_partial_room_checks() {
        assert_eq!(convert_latin1_to_utf8_partial(&[0xE9u8], &mut [0u8; 1]), (0, 0));
        assert_eq!(convert_latin1_to_utf8_partial(&[0xE9u8], &mut [0u8; 2]), (1, 2));
        assert_eq!(
            convert_latin1_to_utf8_partial(&[0xE9u8, 0xE9], &mut [0u8; 3]),
            (1, 2)
        );
        assert_eq!(convert_latin1_to_utf8_partial(&[], &mut [0u8; 4]), (0, 0));
    }

    #[test]
    fn decode_latin1_borrows_for_ascii() {
        let ascii = decode_latin1(b"plain ascii");
        assert!(matches!(ascii, Cow::Borrowed(_)));
        assert_eq!(ascii.as_ref(), "plain ascii");
        let empty = decode_latin1(&[]);
        assert!(matches!(empty, Cow::Borrowed(_)));
    }

    #[test]
    #[should_panic]
    fn decode_latin1_non_ascii_panics_in_debug_current_behavior() {
        // Suspected production bug (reported, not fixed): `decode_latin1`
        // clones the head `vec` to reach its spare capacity (encoding.rs),
        // but `Vec::clone` allocates exactly `len` capacity, so the spare
        // slice is empty. The debug assertion in `convert_latin1_to_utf8`
        // fires; in release the non-ASCII tail would be silently dropped.
        let owned = decode_latin1(&[0x41u8, 0xE9]);
        assert_eq!(owned.as_ref(), "A\u{e9}");
    }

    #[test]
    fn encode_latin1_lossy_borrows_for_ascii() {
        let ascii = encode_latin1_lossy("abc");
        assert!(matches!(ascii, Cow::Borrowed(_)));
        assert_eq!(ascii.as_ref(), b"abc");
    }

    #[test]
    #[should_panic]
    fn encode_latin1_lossy_non_ascii_panics_in_debug_current_behavior() {
        // Same `Vec::clone` spare-capacity suspect as `decode_latin1`: the
        // debug assertion on `written <= spare_capacity.len()` fires because
        // the clone has no spare capacity.
        let owned = encode_latin1_lossy("A\u{e9}");
        assert_eq!(owned.as_ref(), &[0x41u8, 0xE9]);
    }

    #[test]
    fn decode_utf8_borrows_valid_input() {
        let text = "h\u{e9}llo \u{1F4A9}";
        let out = decode_utf8(text.as_bytes());
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), text);
    }

    #[test]
    fn convert_latin1_to_str_ok_and_too_small() {
        let mut holder: String = "\u{0}".repeat(2);
        let len = ok_size(convert_latin1_to_str(&[0xE9u8], holder.as_mut_str()));
        assert_eq!(len, 2);
        assert_eq!(&holder.as_bytes()[..2], "\u{e9}".as_bytes());
        let mut tiny: String = "\u{0}".to_string();
        let err = err_of(convert_latin1_to_str(&[0xE9u8], tiny.as_mut_str()));
        assert_eq!(
            err,
            EncodingError::DestinationTooSmall {
                required: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn convert_latin1_to_str_partial_zeroes_tail() {
        let mut holder: String = "\u{0}".repeat(4);
        let (read, written) = convert_latin1_to_str_partial(&[0xE9u8], holder.as_mut_str());
        assert_eq!((read, written), (1, 2));
        assert_eq!(holder.as_bytes(), [0xC3u8, 0xA9, 0x00, 0x00]);
    }

    #[test]
    fn utf16_valid_up_to_covers_pairs_and_strays() {
        let good = utf16_units("a\u{1F4A9}b");
        assert_eq!(utf16_valid_up_to(&good), good.len());
        let lone_high_end = [0x0061u16, 0xD83D];
        assert_eq!(utf16_valid_up_to(&lone_high_end), 1);
        let lone_low_first = [0xDEADu16, 0x0061];
        assert_eq!(utf16_valid_up_to(&lone_low_first), 0);
        assert_eq!(utf16_valid_up_to(&[]), 0);
    }

    #[test]
    fn ensure_utf16_validity_replaces_unpaired_keeps_pairs() {
        let mut buf = [0x0061u16, 0xD800, 0xD83D, 0xDCA9, 0xDEAD];
        ensure_utf16_validity(&mut buf);
        assert_eq!(buf[0], 0x0061);
        assert_eq!(buf[1], 0xFFFD);
        assert_eq!(&buf[2..4], &[0xD83D, 0xDCA9]);
        assert_eq!(buf[4], 0xFFFD);
    }

    #[test]
    fn utf8_latin1_up_to_stops_above_latin1_points() {
        assert_eq!(utf8_latin1_up_to(b"plain"), 5);
        assert_eq!(utf8_latin1_up_to("caf\u{e9}".as_bytes()), 5);
        assert_eq!(utf8_latin1_up_to("a\u{e9}\u{2603}".as_bytes()), 3);
    }

    #[test]
    fn str_latin1_up_to_reports_char_boundary_prefix() {
        assert_eq!(str_latin1_up_to("hello"), 5);
        assert_eq!(str_latin1_up_to("a\u{e9}\u{2603}"), 3);
        assert_eq!(str_latin1_up_to("\u{2603}"), 0);
    }

    #[test]
    fn variant_encoding_is_single_byte_flag() {
        assert!(LATIN1.variant.is_single_byte());
        assert!(RUN_ONLY.variant.is_single_byte());
        assert!(!UTF_8.variant.is_single_byte());
        assert!(!UTF_16LE.variant.is_single_byte());
        assert!(!UTF_16BE.variant.is_single_byte());
    }
}
