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

use crate::ascii::{ascii_to_ascii, ascii_to_basic_latin};
use crate::encoding::DecoderResult;
use core::cmp::min;

#[derive(PartialEq, Debug)]
pub enum Space<T> {
    Available(T),
    Full(usize),
}

#[derive(PartialEq, Debug)]
pub enum CopyAsciiResult<T, U> {
    Stop(T),
    GoOn(U),
}

#[derive(PartialEq, Debug)]
pub enum NonAscii {
    BmpExclAscii(u16),
    Astral(char),
}

#[derive(PartialEq, Debug)]
pub enum Unicode {
    Ascii(u8),
    NonAscii(NonAscii),
}

pub trait Endian {
    const OPPOSITE_ENDIAN: bool;
}

pub struct BigEndian;
impl Endian for BigEndian {
    #[cfg(target_endian = "little")]
    const OPPOSITE_ENDIAN: bool = true;
    #[cfg(target_endian = "big")]
    const OPPOSITE_ENDIAN: bool = false;
}

pub struct LittleEndian;
impl Endian for LittleEndian {
    #[cfg(target_endian = "little")]
    const OPPOSITE_ENDIAN: bool = false;
    #[cfg(target_endian = "big")]
    const OPPOSITE_ENDIAN: bool = true;
}

#[derive(Debug, Copy, Clone)]
pub struct UnalignedU16Slice {
    ptr: *const u8,
    len: usize,
}

impl UnalignedU16Slice {
    /// # Safety
    #[inline(always)]
    pub unsafe fn new(ptr: *const u8, len: usize) -> UnalignedU16Slice {
        UnalignedU16Slice { ptr, len }
    }
    #[inline(always)]
    pub fn trim_last(&mut self) {
        debug_assert!(self.len > 0);
        self.len = self.len.saturating_sub(1);
    }
    #[inline(always)]
    pub fn at(&self, i: usize) -> u16 {
        debug_assert!(i < self.len);
        if i >= self.len {
            return 0;
        }
        unsafe { core::ptr::read_unaligned(self.ptr.add(i * 2) as *const u16) }
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline(always)]
    pub fn tail(&self, from: usize) -> UnalignedU16Slice {
        debug_assert!(from <= self.len);
        let from = from.min(self.len);
        unsafe { UnalignedU16Slice::new(self.ptr.add(from * 2), self.len - from) }
    }
    #[inline(always)]
    pub fn copy_bmp_to<E: Endian>(&self, dst: &mut [u16]) -> Option<(u16, usize)> {
        let n = min(self.len, dst.len());
        for (i, unit) in dst.iter_mut().take(n).enumerate() {
            let value = if E::OPPOSITE_ENDIAN {
                self.at(i).swap_bytes()
            } else {
                self.at(i)
            };
            if value.wrapping_sub(0xD800) <= (0xDFFF - 0xD800) {
                return Some((value, i));
            }
            *unit = value;
        }
        None
    }
}

pub struct ByteSource<'a> {
    slice: &'a [u8],
    pos: usize,
}

impl<'a> ByteSource<'a> {
    #[inline(always)]
    pub fn new(src: &'a [u8]) -> ByteSource<'a> {
        ByteSource { slice: src, pos: 0 }
    }
    #[inline(always)]
    pub fn check_available<'b>(&'b mut self) -> Space<ByteReadHandle<'b, 'a>> {
        if self.pos < self.slice.len() {
            Space::Available(ByteReadHandle::new(self))
        } else {
            Space::Full(self.consumed())
        }
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.pos
    }
}

pub struct ByteReadHandle<'a, 'b>
where
    'b: 'a,
{
    source: &'a mut ByteSource<'b>,
}
impl<'a, 'b> ByteReadHandle<'a, 'b> {
    #[inline(always)]
    fn new(src: &'a mut ByteSource<'b>) -> ByteReadHandle<'a, 'b> {
        ByteReadHandle { source: src }
    }
    #[inline(always)]
    pub fn read(self) -> (u8, ByteUnreadHandle<'a, 'b>) {
        let source = self.source;
        let byte = source.slice[source.pos];
        source.pos += 1;
        let old_pos = source.pos - 1;
        (byte, ByteUnreadHandle { source, old_pos })
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.source.consumed()
    }
}

pub struct ByteUnreadHandle<'a, 'b>
where
    'b: 'a,
{
    source: &'a mut ByteSource<'b>,
    old_pos: usize,
}
impl<'a, 'b> ByteUnreadHandle<'a, 'b> {
    #[inline(always)]
    pub fn unread(self) -> usize {
        let pos = self.source.pos;
        self.source.pos = self.old_pos;
        pos
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.source.consumed()
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut ByteSource<'b> {
        self.source
    }
}

pub struct Utf16BmpHandle<'a, 'b>
where
    'b: 'a,
{
    dest: &'a mut Utf16Destination<'b>,
}
impl<'a, 'b> Utf16BmpHandle<'a, 'b> {
    #[inline(always)]
    fn new(dst: &'a mut Utf16Destination<'b>) -> Utf16BmpHandle<'a, 'b> {
        Utf16BmpHandle { dest: dst }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.dest.written()
    }
    #[inline(always)]
    pub fn write_ascii(self, ascii: u8) -> &'a mut Utf16Destination<'b> {
        self.dest.write_ascii(ascii);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp(self, bmp: u16) -> &'a mut Utf16Destination<'b> {
        self.dest.write_bmp(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp_excl_ascii(self, bmp: u16) -> &'a mut Utf16Destination<'b> {
        self.dest.write_bmp_excl_ascii(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_astral(self, astral: u32) -> &'a mut Utf16Destination<'b> {
        self.dest.write_astral(astral);
        self.dest
    }
    #[inline(always)]
    pub fn write_surrogate_pair(self, high: u16, low: u16) -> &'a mut Utf16Destination<'b> {
        self.dest.write_surrogate_pair(high, low);
        self.dest
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut Utf16Destination<'b> {
        self.dest
    }
}

pub struct Utf16AstralHandle<'a, 'b>
where
    'b: 'a,
{
    dest: &'a mut Utf16Destination<'b>,
}
impl<'a, 'b> Utf16AstralHandle<'a, 'b> {
    #[inline(always)]
    fn new(dst: &'a mut Utf16Destination<'b>) -> Utf16AstralHandle<'a, 'b> {
        Utf16AstralHandle { dest: dst }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.dest.written()
    }
    #[inline(always)]
    pub fn write_ascii(self, ascii: u8) -> &'a mut Utf16Destination<'b> {
        self.dest.write_ascii(ascii);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp(self, bmp: u16) -> &'a mut Utf16Destination<'b> {
        self.dest.write_bmp(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp_excl_ascii(self, bmp: u16) -> &'a mut Utf16Destination<'b> {
        self.dest.write_bmp_excl_ascii(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_astral(self, astral: u32) -> &'a mut Utf16Destination<'b> {
        self.dest.write_astral(astral);
        self.dest
    }
    #[inline(always)]
    pub fn write_surrogate_pair(self, high: u16, low: u16) -> &'a mut Utf16Destination<'b> {
        self.dest.write_surrogate_pair(high, low);
        self.dest
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut Utf16Destination<'b> {
        self.dest
    }
}

pub struct Utf16Destination<'a> {
    slice: &'a mut [u16],
    pos: usize,
}
impl<'a> Utf16Destination<'a> {
    #[inline(always)]
    pub fn new(dst: &mut [u16]) -> Utf16Destination<'_> {
        Utf16Destination { slice: dst, pos: 0 }
    }
    #[inline(always)]
    pub fn check_space_bmp<'b>(&'b mut self) -> Space<Utf16BmpHandle<'b, 'a>> {
        if self.pos < self.slice.len() {
            Space::Available(Utf16BmpHandle::new(self))
        } else {
            Space::Full(self.written())
        }
    }
    #[inline(always)]
    pub fn check_space_astral<'b>(&'b mut self) -> Space<Utf16AstralHandle<'b, 'a>> {
        if self.pos + 1 < self.slice.len() {
            Space::Available(Utf16AstralHandle::new(self))
        } else {
            Space::Full(self.written())
        }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.pos
    }
    #[inline(always)]
    fn write_code_unit(&mut self, u: u16) {
        unsafe {
            *(self.slice.get_unchecked_mut(self.pos)) = u;
        }
        self.pos += 1;
    }
    #[inline(always)]
    fn write_ascii(&mut self, ascii: u8) {
        debug_assert!(ascii < 0x80);
        self.write_code_unit(u16::from(ascii));
    }
    #[inline(always)]
    fn write_bmp(&mut self, bmp: u16) {
        self.write_code_unit(bmp);
    }
    #[inline(always)]
    fn write_bmp_excl_ascii(&mut self, bmp: u16) {
        debug_assert!(bmp >= 0x80);
        self.write_code_unit(bmp);
    }
    #[inline(always)]
    fn write_astral(&mut self, astral: u32) {
        debug_assert!(astral > 0xFFFF && astral <= 0x10_FFFF);
        self.write_code_unit((0xD7C0 + (astral >> 10)) as u16);
        self.write_code_unit((0xDC00 + (astral & 0x3FF)) as u16);
    }
    #[inline(always)]
    fn write_surrogate_pair(&mut self, high: u16, low: u16) {
        self.write_code_unit(high);
        self.write_code_unit(low);
    }
    #[inline(always)]
    pub fn copy_ascii_from_check_space_bmp<'b>(
        &'b mut self,
        source: &mut ByteSource,
    ) -> CopyAsciiResult<(DecoderResult, usize, usize), (u8, Utf16BmpHandle<'b, 'a>)> {
        let non_ascii_ret = {
            let src_remaining = &source.slice[source.pos..];
            let dst_remaining = &mut self.slice[self.pos..];
            let (pending, length) = if dst_remaining.len() < src_remaining.len() {
                (DecoderResult::OutputFull, dst_remaining.len())
            } else {
                (DecoderResult::InputEmpty, src_remaining.len())
            };
            match ascii_to_basic_latin(src_remaining, dst_remaining) {
                None => {
                    source.pos += length;
                    self.pos += length;
                    return CopyAsciiResult::Stop((pending, source.pos, self.pos));
                }
                Some((non_ascii, consumed)) => {
                    source.pos += consumed;
                    self.pos += consumed;
                    source.pos += 1;
                    non_ascii
                }
            }
        };
        CopyAsciiResult::GoOn((non_ascii_ret, Utf16BmpHandle::new(self)))
    }
    #[inline(always)]
    pub fn copy_utf8_up_to_invalid_from(&mut self, source: &mut ByteSource) {
        let src_remaining = &source.slice[source.pos..];
        let dst_remaining = &mut self.slice[self.pos..];
        let (read, written) = crate::utf8::convert_utf8_to_utf16_up_to_invalid(src_remaining, dst_remaining);
        source.pos += read;
        self.pos += written;
    }
    #[inline(always)]
    pub fn copy_utf16_from<E: Endian>(&mut self, source: &mut ByteSource) -> Option<(usize, usize)> {
        let src_remaining = &source.slice[source.pos..];
        let dst_remaining = &mut self.slice[self.pos..];
        let mut src_unaligned = unsafe {
            UnalignedU16Slice::new(
                src_remaining.as_ptr(),
                min(src_remaining.len() / 2, dst_remaining.len()),
            )
        };
        if src_unaligned.is_empty() {
            return None;
        }
        let last_unit = if E::OPPOSITE_ENDIAN {
            src_unaligned
                .at(src_unaligned.len() - 1)
                .swap_bytes()
        } else {
            src_unaligned.at(src_unaligned.len() - 1)
        };
        if crate::encoding::in_range16(last_unit, 0xD800, 0xDC00) {
            src_unaligned.trim_last();
        }
        let mut offset = 0usize;
        loop {
            if let Some((surrogate, bmp_len)) = {
                let src_left = src_unaligned.tail(offset);
                let dst_left = &mut dst_remaining[offset..src_unaligned.len()];
                src_left.copy_bmp_to::<E>(dst_left)
            } {
                offset += bmp_len;
                let second_pos = offset + 1;
                if surrogate > 0xDBFF || second_pos == src_unaligned.len() {
                    source.pos += second_pos * 2;
                    self.pos += offset;
                    return Some((source.pos, self.pos));
                }
                let second = if E::OPPOSITE_ENDIAN {
                    src_unaligned.at(second_pos).swap_bytes()
                } else {
                    src_unaligned.at(second_pos)
                };
                if !crate::encoding::in_range16(second, 0xDC00, 0xE000) {
                    source.pos += second_pos * 2;
                    self.pos += offset;
                    return Some((source.pos, self.pos));
                }
                dst_remaining[offset] = surrogate;
                dst_remaining[second_pos] = second;
                offset += 2;
                continue;
            } else {
                source.pos += src_unaligned.len() * 2;
                self.pos += src_unaligned.len();
                return None;
            }
        }
    }
}
pub struct Utf8BmpHandle<'a, 'b>
where
    'b: 'a,
{
    dest: &'a mut Utf8Destination<'b>,
}
impl<'a, 'b> Utf8BmpHandle<'a, 'b> {
    #[inline(always)]
    fn new(dst: &'a mut Utf8Destination<'b>) -> Utf8BmpHandle<'a, 'b> {
        Utf8BmpHandle { dest: dst }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.dest.written()
    }
    #[inline(always)]
    pub fn write_ascii(self, ascii: u8) -> &'a mut Utf8Destination<'b> {
        self.dest.write_ascii(ascii);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp(self, bmp: u16) -> &'a mut Utf8Destination<'b> {
        self.dest.write_bmp(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp_excl_ascii(self, bmp: u16) -> &'a mut Utf8Destination<'b> {
        self.dest.write_bmp_excl_ascii(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_astral(self, astral: u32) -> &'a mut Utf8Destination<'b> {
        self.dest.write_astral(astral);
        self.dest
    }
    #[inline(always)]
    pub fn write_surrogate_pair(self, high: u16, low: u16) -> &'a mut Utf8Destination<'b> {
        self.dest.write_surrogate_pair(high, low);
        self.dest
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut Utf8Destination<'b> {
        self.dest
    }
}

pub struct Utf8AstralHandle<'a, 'b>
where
    'b: 'a,
{
    dest: &'a mut Utf8Destination<'b>,
}
impl<'a, 'b> Utf8AstralHandle<'a, 'b> {
    #[inline(always)]
    fn new(dst: &'a mut Utf8Destination<'b>) -> Utf8AstralHandle<'a, 'b> {
        Utf8AstralHandle { dest: dst }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.dest.written()
    }
    #[inline(always)]
    pub fn write_ascii(self, ascii: u8) -> &'a mut Utf8Destination<'b> {
        self.dest.write_ascii(ascii);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp(self, bmp: u16) -> &'a mut Utf8Destination<'b> {
        self.dest.write_bmp(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_bmp_excl_ascii(self, bmp: u16) -> &'a mut Utf8Destination<'b> {
        self.dest.write_bmp_excl_ascii(bmp);
        self.dest
    }
    #[inline(always)]
    pub fn write_astral(self, astral: u32) -> &'a mut Utf8Destination<'b> {
        self.dest.write_astral(astral);
        self.dest
    }
    #[inline(always)]
    pub fn write_surrogate_pair(self, high: u16, low: u16) -> &'a mut Utf8Destination<'b> {
        self.dest.write_surrogate_pair(high, low);
        self.dest
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut Utf8Destination<'b> {
        self.dest
    }
}

pub struct Utf8Destination<'a> {
    slice: &'a mut [u8],
    pos: usize,
}
impl<'a> Utf8Destination<'a> {
    #[inline(always)]
    pub fn new(dst: &mut [u8]) -> Utf8Destination<'_> {
        Utf8Destination { slice: dst, pos: 0 }
    }
    #[inline(always)]
    pub fn check_space_bmp<'b>(&'b mut self) -> Space<Utf8BmpHandle<'b, 'a>> {
        if self.pos + 2 < self.slice.len() {
            Space::Available(Utf8BmpHandle::new(self))
        } else {
            Space::Full(self.written())
        }
    }
    #[inline(always)]
    pub fn check_space_astral<'b>(&'b mut self) -> Space<Utf8AstralHandle<'b, 'a>> {
        if self.pos + 3 < self.slice.len() {
            Space::Available(Utf8AstralHandle::new(self))
        } else {
            Space::Full(self.written())
        }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.pos
    }
    #[inline(always)]
    fn write_code_unit(&mut self, u: u8) {
        unsafe {
            *(self.slice.get_unchecked_mut(self.pos)) = u;
        }
        self.pos += 1;
    }
    #[inline(always)]
    fn write_ascii(&mut self, ascii: u8) {
        debug_assert!(ascii < 0x80);
        self.write_code_unit(ascii);
    }
    #[inline(always)]
    fn write_bmp(&mut self, bmp: u16) {
        if bmp < 0x80 {
            self.write_ascii(bmp as u8);
        } else if bmp < 0x800 {
            self.write_mid_bmp(bmp);
        } else {
            self.write_upper_bmp(bmp);
        }
    }
    #[inline(always)]
    fn write_mid_bmp(&mut self, mid_bmp: u16) {
        debug_assert!((0x80..0x800).contains(&mid_bmp));
        self.write_code_unit(((mid_bmp >> 6) | 0xC0) as u8);
        self.write_code_unit(((mid_bmp & 0x3F) | 0x80) as u8);
    }
    #[inline(always)]
    fn write_upper_bmp(&mut self, upper_bmp: u16) {
        debug_assert!(upper_bmp >= 0x800);
        self.write_code_unit(((upper_bmp >> 12) | 0xE0) as u8);
        self.write_code_unit((((upper_bmp & 0xFC0) >> 6) | 0x80) as u8);
        self.write_code_unit(((upper_bmp & 0x3F) | 0x80) as u8);
    }
    #[inline(always)]
    fn write_bmp_excl_ascii(&mut self, bmp: u16) {
        if bmp < 0x800 {
            self.write_mid_bmp(bmp);
        } else {
            self.write_upper_bmp(bmp);
        }
    }
    #[inline(always)]
    fn write_astral(&mut self, astral: u32) {
        debug_assert!(astral > 0xFFFF && astral <= 0x10_FFFF);
        self.write_code_unit(((astral >> 18) | 0xF0) as u8);
        self.write_code_unit((((astral & 0x3F000) >> 12) | 0x80) as u8);
        self.write_code_unit((((astral & 0xFC0) >> 6) | 0x80) as u8);
        self.write_code_unit(((astral & 0x3F) | 0x80) as u8);
    }
    #[inline(always)]
    pub fn write_surrogate_pair(&mut self, high: u16, low: u16) {
        self.write_astral(
            (u32::from(high) << 10) + u32::from(low) - (((0xD800u32 << 10) - 0x10000u32) + 0xDC00u32),
        );
    }
    #[inline(always)]
    pub fn copy_ascii_from_check_space_bmp<'b>(
        &'b mut self,
        source: &mut ByteSource,
    ) -> CopyAsciiResult<(DecoderResult, usize, usize), (u8, Utf8BmpHandle<'b, 'a>)> {
        let non_ascii_ret = {
            let dst_len = self.slice.len();
            let src_remaining = &source.slice[source.pos..];
            let dst_remaining = &mut self.slice[self.pos..];
            let (pending, length) = if dst_remaining.len() < src_remaining.len() {
                (DecoderResult::OutputFull, dst_remaining.len())
            } else {
                (DecoderResult::InputEmpty, src_remaining.len())
            };
            match ascii_to_ascii(src_remaining, dst_remaining) {
                None => {
                    source.pos += length;
                    self.pos += length;
                    return CopyAsciiResult::Stop((pending, source.pos, self.pos));
                }
                Some((non_ascii, consumed)) => {
                    source.pos += consumed;
                    self.pos += consumed;
                    if self.pos + 2 < dst_len {
                        source.pos += 1;
                        non_ascii
                    } else {
                        return CopyAsciiResult::Stop((DecoderResult::OutputFull, source.pos, self.pos));
                    }
                }
            }
        };
        CopyAsciiResult::GoOn((non_ascii_ret, Utf8BmpHandle::new(self)))
    }
    #[inline(always)]
    pub fn copy_ascii_from_check_space_astral<'b>(
        &'b mut self,
        source: &mut ByteSource,
    ) -> CopyAsciiResult<(DecoderResult, usize, usize), (u8, Utf8AstralHandle<'b, 'a>)> {
        let non_ascii_ret = {
            let dst_len = self.slice.len();
            let src_remaining = &source.slice[source.pos..];
            let dst_remaining = &mut self.slice[self.pos..];
            let (pending, length) = if dst_remaining.len() < src_remaining.len() {
                (DecoderResult::OutputFull, dst_remaining.len())
            } else {
                (DecoderResult::InputEmpty, src_remaining.len())
            };
            match ascii_to_ascii(src_remaining, dst_remaining) {
                None => {
                    source.pos += length;
                    self.pos += length;
                    return CopyAsciiResult::Stop((pending, source.pos, self.pos));
                }
                Some((non_ascii, consumed)) => {
                    source.pos += consumed;
                    self.pos += consumed;
                    if self.pos + 3 < dst_len {
                        source.pos += 1;
                        non_ascii
                    } else {
                        return CopyAsciiResult::Stop((DecoderResult::OutputFull, source.pos, self.pos));
                    }
                }
            }
        };
        CopyAsciiResult::GoOn((non_ascii_ret, Utf8AstralHandle::new(self)))
    }
    #[inline(always)]
    pub fn copy_utf8_up_to_invalid_from(&mut self, source: &mut ByteSource) {
        let src_remaining = &source.slice[source.pos..];
        let dst_remaining = &mut self.slice[self.pos..];
        let min_len = min(src_remaining.len(), dst_remaining.len());
        let valid_len = crate::utf8::utf8_valid_up_to(&src_remaining[..min_len]);
        dst_remaining[..valid_len].copy_from_slice(&src_remaining[..valid_len]);
        source.pos += valid_len;
        self.pos += valid_len;
    }
    #[inline(always)]
    pub fn copy_utf16_from<E: Endian>(&mut self, source: &mut ByteSource) -> Option<(usize, usize)> {
        let src_remaining = &source.slice[source.pos..];
        let dst_remaining = &mut self.slice[self.pos..];
        let mut src_unaligned =
            unsafe { UnalignedU16Slice::new(src_remaining.as_ptr(), src_remaining.len() / 2) };
        if src_unaligned.is_empty() {
            return None;
        }
        let mut last_unit = src_unaligned.at(src_unaligned.len() - 1);
        if E::OPPOSITE_ENDIAN {
            last_unit = last_unit.swap_bytes();
        }
        if crate::encoding::in_range16(last_unit, 0xD800, 0xDC00) {
            src_unaligned.trim_last();
        }
        let (read, written, had_error) = convert_unaligned_utf16_to_utf8::<E>(src_unaligned, dst_remaining);
        source.pos += read * 2;
        self.pos += written;
        if had_error {
            Some((source.pos, self.pos))
        } else {
            None
        }
    }
}

pub struct Utf16Source<'a> {
    slice: &'a [u16],
    pos: usize,
}
impl<'a> Utf16Source<'a> {
    #[inline(always)]
    pub fn new(src: &'a [u16]) -> Utf16Source<'a> {
        Utf16Source { slice: src, pos: 0 }
    }
    #[inline(always)]
    pub fn check_available<'b>(&'b mut self) -> Space<Utf16ReadHandle<'b, 'a>> {
        if self.pos < self.slice.len() {
            Space::Available(Utf16ReadHandle::new(self))
        } else {
            Space::Full(self.consumed())
        }
    }
    #[inline(always)]
    fn read(&mut self) -> char {
        let unit = self.slice[self.pos];
        self.pos += 1;
        let unit_minus_surrogate_start = unit.wrapping_sub(0xD800);
        if unit_minus_surrogate_start > (0xDFFF - 0xD800) {
            return unsafe { char::from_u32_unchecked(u32::from(unit)) };
        }
        if unit_minus_surrogate_start <= (0xDBFF - 0xD800) && self.pos < self.slice.len() {
            let second = self.slice[self.pos];
            let second_minus_low_surrogate_start = second.wrapping_sub(0xDC00);
            if second_minus_low_surrogate_start <= (0xDFFF - 0xDC00) {
                self.pos += 1;
                return unsafe {
                    char::from_u32_unchecked(
                        (u32::from(unit) << 10) + u32::from(second)
                            - (((0xD800u32 << 10) - 0x10000u32) + 0xDC00u32),
                    )
                };
            }
        }
        unsafe { char::from_u32_unchecked(0xFFFD) }
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.pos
    }
}

pub struct Utf16ReadHandle<'a, 'b>
where
    'b: 'a,
{
    source: &'a mut Utf16Source<'b>,
}
impl<'a, 'b> Utf16ReadHandle<'a, 'b> {
    #[inline(always)]
    fn new(src: &'a mut Utf16Source<'b>) -> Utf16ReadHandle<'a, 'b> {
        Utf16ReadHandle { source: src }
    }
    #[inline(always)]
    pub fn read(self) -> (char, Utf16UnreadHandle<'a, 'b>) {
        let source = self.source;
        let old_pos = source.pos;
        let c = source.read();
        (c, Utf16UnreadHandle { source, old_pos })
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.source.consumed()
    }
}

pub struct Utf16UnreadHandle<'a, 'b>
where
    'b: 'a,
{
    source: &'a mut Utf16Source<'b>,
    old_pos: usize,
}
impl<'a, 'b> Utf16UnreadHandle<'a, 'b> {
    #[inline(always)]
    pub fn unread(self) -> usize {
        let pos = self.source.pos;
        self.source.pos = self.old_pos;
        pos
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.source.consumed()
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut Utf16Source<'b> {
        self.source
    }
}

pub struct Utf8Source<'a> {
    slice: &'a [u8],
    pos: usize,
}
impl<'a> Utf8Source<'a> {
    #[inline(always)]
    pub fn new(src: &'a str) -> Utf8Source<'a> {
        Utf8Source {
            slice: src.as_bytes(),
            pos: 0,
        }
    }
    #[inline(always)]
    pub fn check_available<'b>(&'b mut self) -> Space<Utf8ReadHandle<'b, 'a>> {
        if self.pos < self.slice.len() {
            Space::Available(Utf8ReadHandle::new(self))
        } else {
            Space::Full(self.consumed())
        }
    }
    #[inline(always)]
    fn read(&mut self) -> char {
        let Some(remaining) = self.slice.get(self.pos..) else {
            return char::REPLACEMENT_CHARACTER;
        };
        if remaining.is_empty() {
            return char::REPLACEMENT_CHARACTER;
        }
        // Decode at most one scalar from the next up-to-4 bytes.
        let take = &remaining[..core::cmp::min(4, remaining.len())];
        match core::str::from_utf8(take) {
            Ok(s) => match s.chars().next() {
                Some(c) => {
                    self.pos += c.len_utf8();
                    c
                }
                None => char::REPLACEMENT_CHARACTER,
            },
            Err(e) if e.valid_up_to() > 0 => {
                // The leading bytes form a valid scalar; consume just that.
                match core::str::from_utf8(&take[..e.valid_up_to()]) {
                    Ok(s) => match s.chars().next() {
                        Some(c) => {
                            self.pos += c.len_utf8();
                            c
                        }
                        None => {
                            self.pos += 1;
                            char::REPLACEMENT_CHARACTER
                        }
                    },
                    Err(_) => {
                        self.pos += 1;
                        char::REPLACEMENT_CHARACTER
                    }
                }
            }
            Err(e) => {
                // Invalid or truncated sequence: emit U+FFFD and skip it.
                self.pos += match e.error_len() {
                    Some(n) => n,
                    None => remaining.len(),
                };
                char::REPLACEMENT_CHARACTER
            }
        }
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.pos
    }
}

pub struct Utf8ReadHandle<'a, 'b>
where
    'b: 'a,
{
    source: &'a mut Utf8Source<'b>,
}
impl<'a, 'b> Utf8ReadHandle<'a, 'b> {
    #[inline(always)]
    fn new(src: &'a mut Utf8Source<'b>) -> Utf8ReadHandle<'a, 'b> {
        Utf8ReadHandle { source: src }
    }
    #[inline(always)]
    pub fn read(self) -> (char, Utf8UnreadHandle<'a, 'b>) {
        let source = self.source;
        let old_pos = source.pos;
        let c = source.read();
        (c, Utf8UnreadHandle { source, old_pos })
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.source.consumed()
    }
}

pub struct Utf8UnreadHandle<'a, 'b>
where
    'b: 'a,
{
    source: &'a mut Utf8Source<'b>,
    old_pos: usize,
}
impl<'a, 'b> Utf8UnreadHandle<'a, 'b> {
    #[inline(always)]
    pub fn unread(self) -> usize {
        let pos = self.source.pos;
        self.source.pos = self.old_pos;
        pos
    }
    #[inline(always)]
    pub fn consumed(&self) -> usize {
        self.source.consumed()
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut Utf8Source<'b> {
        self.source
    }
}

pub struct ByteDestination<'a> {
    slice: &'a mut [u8],
    pos: usize,
}
impl<'a> ByteDestination<'a> {
    #[inline(always)]
    pub fn new(dst: &mut [u8]) -> ByteDestination<'_> {
        ByteDestination { slice: dst, pos: 0 }
    }
    #[inline(always)]
    pub fn remaining(&self) -> &[u8] {
        &self.slice[self.pos..]
    }
    #[inline(always)]
    pub fn advance(&mut self, n: usize) {
        self.pos += n;
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.pos
    }
    #[inline(always)]
    pub fn check_space_two<'b>(&'b mut self) -> Space<ByteTwoHandle<'b, 'a>> {
        if self.pos + 2 <= self.slice.len() {
            Space::Available(ByteTwoHandle::new(self))
        } else {
            Space::Full(self.written())
        }
    }
    #[inline(always)]
    pub fn check_space_four<'b>(&'b mut self) -> Space<ByteFourHandle<'b, 'a>> {
        if self.pos + 4 <= self.slice.len() {
            Space::Available(ByteFourHandle::new(self))
        } else {
            Space::Full(self.written())
        }
    }
    #[inline(always)]
    pub fn write_byte(&mut self, b: u8) {
        unsafe {
            *(self.slice.get_unchecked_mut(self.pos)) = b;
        }
        self.pos += 1;
    }
}

pub struct ByteTwoHandle<'a, 'b>
where
    'b: 'a,
{
    dest: &'a mut ByteDestination<'b>,
}
impl<'a, 'b> ByteTwoHandle<'a, 'b> {
    #[inline(always)]
    fn new(dest: &'a mut ByteDestination<'b>) -> ByteTwoHandle<'a, 'b> {
        ByteTwoHandle { dest }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.dest.written()
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut ByteDestination<'b> {
        self.dest
    }
}

pub struct ByteFourHandle<'a, 'b>
where
    'b: 'a,
{
    dest: &'a mut ByteDestination<'b>,
}
impl<'a, 'b> ByteFourHandle<'a, 'b> {
    #[inline(always)]
    fn new(dest: &'a mut ByteDestination<'b>) -> ByteFourHandle<'a, 'b> {
        ByteFourHandle { dest }
    }
    #[inline(always)]
    pub fn written(&self) -> usize {
        self.dest.written()
    }
    #[inline(always)]
    pub fn commit(self) -> &'a mut ByteDestination<'b> {
        self.dest
    }
}

pub fn copy_unaligned_basic_latin_to_ascii<E: Endian>(
    src: UnalignedU16Slice,
    dst: &mut [u8],
) -> CopyAsciiResult<usize, (u16, usize)> {
    let len = min(src.len(), dst.len());
    let mut i = 0usize;
    loop {
        if i == len {
            return CopyAsciiResult::Stop(i);
        }
        let unit = if E::OPPOSITE_ENDIAN {
            src.at(i).swap_bytes()
        } else {
            src.at(i)
        };
        if unit > 0x7F {
            return CopyAsciiResult::GoOn((unit, i));
        }
        dst[i] = unit as u8;
        i += 1;
    }
}

pub fn convert_unaligned_utf16_to_utf8<E: Endian>(
    src: UnalignedU16Slice,
    dst: &mut [u8],
) -> (usize, usize, bool) {
    if dst.len() < 4 {
        return (0, 0, false);
    }
    let mut src_pos = 0usize;
    let mut dst_pos = 0usize;
    let src_len = src.len();
    let dst_len_minus_three = dst.len() - 3;
    'outer: loop {
        let mut non_ascii =
            match copy_unaligned_basic_latin_to_ascii::<E>(src.tail(src_pos), &mut dst[dst_pos..]) {
                CopyAsciiResult::GoOn((unit, read_written)) => {
                    src_pos += read_written;
                    dst_pos += read_written;
                    unit
                }
                CopyAsciiResult::Stop(read_written) => {
                    return (src_pos + read_written, dst_pos + read_written, false);
                }
            };
        if dst_pos >= dst_len_minus_three {
            break 'outer;
        }
        src_pos += 1;
        'inner: loop {
            let non_ascii_minus_surrogate_start = non_ascii.wrapping_sub(0xD800);
            if non_ascii_minus_surrogate_start > (0xDFFF - 0xD800) {
                if non_ascii < 0x800 {
                    dst[dst_pos] = ((non_ascii >> 6) | 0xC0) as u8;
                    dst_pos += 1;
                    dst[dst_pos] = ((non_ascii & 0x3F) | 0x80) as u8;
                    dst_pos += 1;
                } else {
                    dst[dst_pos] = ((non_ascii >> 12) | 0xE0) as u8;
                    dst_pos += 1;
                    dst[dst_pos] = (((non_ascii & 0xFC0) >> 6) | 0x80) as u8;
                    dst_pos += 1;
                    dst[dst_pos] = ((non_ascii & 0x3F) | 0x80) as u8;
                    dst_pos += 1;
                }
            } else if non_ascii_minus_surrogate_start <= (0xDBFF - 0xD800) {
                if src_pos < src_len {
                    let second = if E::OPPOSITE_ENDIAN {
                        src.at(src_pos).swap_bytes()
                    } else {
                        src.at(src_pos)
                    };
                    let second_minus_low_surrogate_start = second.wrapping_sub(0xDC00);
                    if second_minus_low_surrogate_start <= (0xDFFF - 0xDC00) {
                        src_pos += 1;
                        let astral = (u32::from(non_ascii) << 10) + u32::from(second)
                            - (((0xD800u32 << 10) - 0x10000u32) + 0xDC00u32);
                        dst[dst_pos] = ((astral >> 18) | 0xF0) as u8;
                        dst_pos += 1;
                        dst[dst_pos] = (((astral & 0x3F000) >> 12) | 0x80) as u8;
                        dst_pos += 1;
                        dst[dst_pos] = (((astral & 0xFC0) >> 6) | 0x80) as u8;
                        dst_pos += 1;
                        dst[dst_pos] = (astral & 0x3F) as u8 | 0x80;
                        dst_pos += 1;
                    } else {
                        return (src_pos, dst_pos, true);
                    }
                } else {
                    return (src_pos, dst_pos, true);
                }
            } else {
                return (src_pos, dst_pos, true);
            }
            if dst_pos >= dst_len_minus_three || src_pos == src_len {
                break 'outer;
            }
            let unit = if E::OPPOSITE_ENDIAN {
                src.at(src_pos).swap_bytes()
            } else {
                src.at(src_pos)
            };
            src_pos += 1;
            if unit > 0x7F {
                non_ascii = unit;
                continue 'inner;
            }
            dst[dst_pos] = unit as u8;
            dst_pos += 1;
            continue 'outer;
        }
    }
    (src_pos, dst_pos, false)
}
const STRIDE: usize = 16;
pub fn validate_bmp_stride(stride: &[u16; STRIDE]) -> Option<usize> {
    if stride.iter().all(|&c| c & 0xF800 != 0xD800) {
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

pub fn validate_latin1_str_stride(stride: &[u8; STRIDE]) -> Option<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn read_one(src: &mut Utf8Source<'_>) -> Option<char> {
        match src.check_available() {
            Space::Available(handle) => {
                let (c, unread) = handle.read();
                unread.commit();
                Some(c)
            }
            Space::Full(_) => None,
        }
    }

    #[test]
    fn utf8_source_reads_ascii_and_multibyte_scalars() {
        let mut src = Utf8Source::new("a\u{00A9}\u{20AC}\u{1F600}");
        assert_eq!(read_one(&mut src), Some('a'));
        assert_eq!(read_one(&mut src), Some('\u{00A9}'));
        assert_eq!(read_one(&mut src), Some('\u{20AC}'));
        assert_eq!(read_one(&mut src), Some('\u{1F600}'));
        assert_eq!(read_one(&mut src), None);
        assert_eq!(src.consumed(), "a\u{00A9}\u{20AC}\u{1F600}".len());
    }

    #[test]
    fn utf8_read_handle_unread_restores_position() {
        let mut src = Utf8Source::new("ab");
        let before = src.consumed();
        match src.check_available() {
            Space::Available(handle) => {
                let (c, unread) = handle.read();
                assert_eq!(c, 'a');
                assert_eq!(unread.consumed(), 1);
                let pos_after_read = unread.unread();
                assert_eq!(pos_after_read, before + 1);
            }
            Space::Full(_) => panic!("expected available input"),
        }
        assert_eq!(src.consumed(), before);
        assert_eq!(read_one(&mut src), Some('a'));
        assert_eq!(read_one(&mut src), Some('b'));
        assert_eq!(read_one(&mut src), None);
    }

    fn read_one_u16(src: &mut Utf16Source<'_>) -> Option<char> {
        match src.check_available() {
            Space::Available(handle) => {
                let (c, unread) = handle.read();
                unread.commit();
                Some(c)
            }
            Space::Full(_) => None,
        }
    }

    fn read_one_byte(src: &mut ByteSource<'_>) -> Option<u8> {
        match src.check_available() {
            Space::Available(handle) => {
                let (b, unread) = handle.read();
                unread.unread();
                Some(b)
            }
            Space::Full(_) => None,
        }
    }

    fn unaligned_over(bytes: &[u8]) -> UnalignedU16Slice {
        // SAFETY: `ptr` and `len` both derive from the same live slice, so the
        // window of `bytes.len() / 2` native-order u16 units is in bounds.
        unsafe { UnalignedU16Slice::new(bytes.as_ptr(), bytes.len() / 2) }
    }

    // Space boundary tests for `Utf8Destination`: `check_space_bmp` grants a
    // handle when `pos + 2 < len` (3 free bytes), `check_space_astral` when
    // `pos + 3 < len` (4 free bytes).

    #[test]
    fn utf8_destination_bmp_check_is_full_on_one_byte_buffer() {
        let mut out = [0u8; 1];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(dest.check_space_bmp(), Space::Full(0)));
    }

    #[test]
    fn utf8_destination_bmp_check_is_full_on_two_byte_buffer() {
        let mut out = [0u8; 2];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(dest.check_space_bmp(), Space::Full(0)));
        assert!(matches!(dest.check_space_astral(), Space::Full(0)));
    }

    #[test]
    fn utf8_destination_bmp_check_admits_three_byte_buffer_and_fills_it() {
        let mut out = [0u8; 3];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x20AC);
        } else {
            panic!("three free bytes must admit the widest BMP sequence");
        }
        assert!(matches!(dest.check_space_bmp(), Space::Full(3)));
        assert_eq!(out, [0xE2, 0x82, 0xAC]);
    }

    #[test]
    fn utf8_destination_checks_are_full_after_two_of_three_bytes_written() {
        let mut out = [0u8; 3];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x00E9);
        } else {
            panic!("expected space for one 2-byte scalar");
        }
        // one slot left over: neither the 3-byte nor the 4-byte check passes
        assert!(matches!(dest.check_space_bmp(), Space::Full(2)));
        assert!(matches!(dest.check_space_astral(), Space::Full(2)));
    }

    #[test]
    fn utf8_destination_astral_check_admits_four_byte_buffer() {
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(dest.check_space_bmp(), Space::Available(_)));
        if let Space::Available(handle) = dest.check_space_astral() {
            handle.write_astral(0x10FFFF);
        } else {
            panic!("four free bytes must admit an astral sequence");
        }
        assert!(matches!(dest.check_space_astral(), Space::Full(4)));
        assert_eq!(out, [0xF4, 0x8F, 0xBF, 0xBF]);
    }

    #[test]
    fn utf8_destination_three_bytes_written_leaves_room_for_bmp_only() {
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x20AC);
        } else {
            panic!("expected space for a 3-byte sequence");
        }
        assert!(matches!(dest.check_space_astral(), Space::Full(3)));
        assert!(matches!(dest.check_space_bmp(), Space::Full(3)));
    }

    #[test]
    fn utf8_destination_two_written_leaves_both_space_checks_full() {
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_astral() {
            handle.write_bmp(0x00E9);
        } else {
            panic!("expected space for a 2-byte sequence");
        }
        // a bmp handle needs room for the widest bmp sequence, so two free
        // bytes leave both checks full even though an ascii byte would fit
        assert!(matches!(dest.check_space_astral(), Space::Full(2)));
        assert!(matches!(dest.check_space_bmp(), Space::Full(2)));
        assert_eq!(dest.written(), 2);
        assert_eq!(out, [0xC3, 0xA9, 0, 0]);
    }

    #[test]
    fn utf8_destination_astral_check_is_full_on_three_byte_buffer() {
        let mut out = [0u8; 3];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(dest.check_space_astral(), Space::Full(0)));
    }

    // Space boundary tests for `Utf16Destination`: bmp needs 1 free unit,
    // astral needs 2.

    #[test]
    fn utf16_destination_empty_buffer_reports_full() {
        let mut out: [u16; 0] = [];
        let mut dest = Utf16Destination::new(&mut out);
        assert!(matches!(dest.check_space_bmp(), Space::Full(0)));
        assert!(matches!(dest.check_space_astral(), Space::Full(0)));
    }

    #[test]
    fn utf16_destination_one_unit_admits_bmp_but_not_astral() {
        let mut out = [0u16; 1];
        let mut dest = Utf16Destination::new(&mut out);
        assert!(matches!(dest.check_space_astral(), Space::Full(0)));
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x20AC);
        } else {
            panic!("one free unit must admit the bmp handle");
        }
        assert!(matches!(dest.check_space_bmp(), Space::Full(1)));
        assert_eq!(out, [0x20AC]);
    }

    #[test]
    fn utf16_destination_second_unit_makes_astral_full() {
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_astral() {
            handle.write_ascii(b'A');
        } else {
            panic!("two free units must admit the astral handle");
        }
        assert!(matches!(dest.check_space_astral(), Space::Full(1)));
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x42);
        } else {
            panic!("expected the last unit to be available");
        }
        assert_eq!(out, [0x41, 0x42]);
    }

    #[test]
    fn utf16_destination_astral_check_reports_intermediate_written_count() {
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_ascii(b'A');
        } else {
            panic!("expected space");
        }
        assert!(matches!(dest.check_space_astral(), Space::Full(1)));
    }

    // Writing through the handles.

    #[test]
    fn utf8_bmp_handle_write_bmp_selects_sequence_width() {
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x41);
        } else {
            panic!("expected space");
        }
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x00E9);
        } else {
            panic!("expected space");
        }
        assert_eq!(&out[..3], [0x41, 0xC3, 0xA9]);
    }

    #[test]
    fn utf8_mid_bmp_encoding_boundaries() {
        let mut out = [0u8; 5];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x80);
        } else {
            panic!("expected space");
        }
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x7FF);
        } else {
            panic!("expected space");
        }
        assert_eq!(dest.written(), 4);
        assert_eq!(&out[..4], [0xC2, 0x80, 0xDF, 0xBF]);
    }

    #[test]
    fn utf8_upper_bmp_encoding_boundaries() {
        let mut out = [0u8; 6];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x800);
        } else {
            panic!("expected space");
        }
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0xFFFF);
        } else {
            panic!("expected space");
        }
        assert_eq!(out, [0xE0, 0xA0, 0x80, 0xEF, 0xBF, 0xBF]);
    }

    #[test]
    fn utf8_bmp_excl_ascii_handles_mid_and_upper_ranges() {
        let mut out = [0u8; 5];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp_excl_ascii(0x00FC);
        } else {
            panic!("expected space");
        }
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp_excl_ascii(0x0416);
        } else {
            panic!("expected space");
        }
        assert_eq!(dest.written(), 4);
        assert_eq!(out, [0xC3, 0xBC, 0xD0, 0x96, 0x00]);
    }

    #[test]
    fn utf8_astral_handle_pair_encoding_matches_astral_scalar() {
        let mut via_pair = [0u8; 4];
        {
            let mut dest = Utf8Destination::new(&mut via_pair);
            if let Space::Available(handle) = dest.check_space_astral() {
                handle.write_surrogate_pair(0xD83D, 0xDE00);
            } else {
                panic!("expected space for a pair");
            }
        }
        let mut via_astral = [0u8; 4];
        {
            let mut dest = Utf8Destination::new(&mut via_astral);
            if let Space::Available(handle) = dest.check_space_astral() {
                handle.write_astral(0x1F600);
            } else {
                panic!("expected space for an astral scalar");
            }
        }
        assert_eq!(via_pair, [0xF0, 0x9F, 0x98, 0x80]);
        assert_eq!(via_pair, via_astral);
    }

    #[test]
    fn utf8_astral_handle_writes_bmp_and_then_blocks_astral() {
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_astral() {
            assert_eq!(handle.written(), 0);
            handle.write_bmp(0x20AC);
        } else {
            panic!("expected astral space");
        }
        assert!(matches!(dest.check_space_astral(), Space::Full(3)));
        assert_eq!(&out[..3], [0xE2, 0x82, 0xAC]);
    }

    #[test]
    fn utf8_bmp_handle_write_astral_fills_four_byte_buffer() {
        // The bmp handle exposes `write_astral`, which only fits buffers with
        // 4 free bytes; `check_space_bmp` itself merely promises 3, so codecs
        // must not rely on this. A 4-byte destination happens to fit it.
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_astral(0x1F600);
        } else {
            panic!("expected space");
        }
        assert!(matches!(dest.check_space_bmp(), Space::Full(4)));
        assert_eq!(out, [0xF0, 0x9F, 0x98, 0x80]);
    }

    #[test]
    fn utf16_astral_handle_pair_writes_equal_units() {
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_astral() {
            handle.write_astral(0x1F600);
        } else {
            panic!("expected space");
        }
        assert_eq!(out, [0xD83D, 0xDE00]);
        let mut pair_out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut pair_out);
        if let Space::Available(handle) = dest.check_space_astral() {
            handle.write_surrogate_pair(0xD83D, 0xDE00);
        } else {
            panic!("expected space");
        }
        assert_eq!(pair_out, out);
    }

    #[test]
    fn utf16_handles_write_single_units() {
        let mut out = [0u16; 3];
        let mut dest = Utf16Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_ascii(b'A');
        } else {
            panic!("expected space");
        }
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp_excl_ascii(0x00E9);
        } else {
            panic!("expected space");
        }
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_bmp(0x20AC);
        } else {
            panic!("expected space");
        }
        assert!(matches!(dest.check_space_bmp(), Space::Full(3)));
        assert_eq!(out, [0x0041, 0x00E9, 0x20AC]);
    }

    #[test]
    fn utf16_bmp_handle_write_astral_fills_two_unit_buffer() {
        // Like the UTF-8 bmp handle, the UTF-16 bmp handle exposes
        // `write_astral` even though `check_space_bmp` only guarantees one
        // free unit; using it on a one-unit window would be out of bounds.
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        if let Space::Available(handle) = dest.check_space_bmp() {
            handle.write_astral(0x1F600);
        } else {
            panic!("expected space");
        }
        assert!(matches!(dest.check_space_bmp(), Space::Full(2)));
        assert_eq!(out, [0xD83D, 0xDE00]);
    }

    // ByteSource handle protocol.

    #[test]
    fn byte_source_check_available_on_empty_source_reports_consumed() {
        let mut src = ByteSource::new(b"");
        assert!(matches!(src.check_available(), Space::Full(0)));
        assert_eq!(src.consumed(), 0);
    }

    #[test]
    fn byte_source_read_handle_reports_position_before_reading() {
        let mut src = ByteSource::new(b"Hi");
        if let Space::Available(handle) = src.check_available() {
            assert_eq!(handle.consumed(), 0);
            let (b, unread) = handle.read();
            assert_eq!(b, b'H');
            assert_eq!(unread.consumed(), 1);
            assert_eq!(unread.commit().consumed(), 1);
        } else {
            panic!("expected one available byte");
        }
        assert_eq!(src.consumed(), 1);
    }

    #[test]
    fn byte_source_unread_restores_position_and_reports_rewound_from() {
        let mut src = ByteSource::new(b"ab");
        if let Space::Available(handle) = src.check_available() {
            let (b, unread) = handle.read();
            assert_eq!(b, b'a');
            assert_eq!(unread.unread(), 1);
        } else {
            panic!("expected available input");
        }
        assert_eq!(src.consumed(), 0);
        // peeking with read + unread never advances the source
        assert_eq!(read_one_byte(&mut src), Some(b'a'));
        assert_eq!(src.consumed(), 0);
    }

    #[test]
    fn byte_source_commit_leaves_position_advanced_for_next_handle() {
        let mut src = ByteSource::new(b"xy");
        if let Space::Available(handle) = src.check_available() {
            let (_b, unread) = handle.read();
            assert_eq!(unread.commit().consumed(), 1);
        } else {
            panic!("expected available input");
        }
        if let Space::Available(handle) = src.check_available() {
            let (b, unread) = handle.read();
            assert_eq!(b, b'y');
            assert_eq!(unread.commit().consumed(), 2);
        } else {
            panic!("expected available input");
        }
        assert!(matches!(src.check_available(), Space::Full(2)));
    }

    #[test]
    fn byte_source_consumed_is_monotonic_across_sequential_handles() {
        let mut src = ByteSource::new(b"abc");
        let mut marks = Vec::new();
        while let Space::Available(handle) = src.check_available() {
            let (_b, unread) = handle.read();
            // go through the reborrow from `commit`: the scrutinee borrow of
            // `src` stays live for the whole body in edition 2024
            let src_ref = unread.commit();
            marks.push(src_ref.consumed());
        }
        assert_eq!(marks, vec![1, 2, 3]);
    }

    // copy_ascii_from_check_space_bmp on a UTF-16 destination.

    #[test]
    fn utf16_copy_ascii_reports_input_empty_when_ascii_source_is_shorter() {
        let mut src = ByteSource::new(b"AB");
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::InputEmpty, 2, 2))
        ));
        assert_eq!(src.consumed(), 2);
        assert_eq!(dest.written(), 2);
        assert_eq!(&out[..2], [0x41, 0x42]);
    }

    #[test]
    fn utf16_copy_ascii_reports_output_full_when_destination_is_shorter() {
        let mut src = ByteSource::new(b"ABCD");
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::OutputFull, 2, 2))
        ));
        assert_eq!(out, [0x41, 0x42]);
    }

    #[test]
    fn utf16_copy_ascii_equal_lengths_count_as_input_empty() {
        let mut src = ByteSource::new(b"AB");
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::InputEmpty, 2, 2))
        ));
    }

    #[test]
    fn utf16_copy_ascii_empty_source_and_empty_destination_still_stop_differently() {
        let mut src = ByteSource::new(b"");
        let mut roomy = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut roomy);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::InputEmpty, 0, 0))
        ));
        let mut src2 = ByteSource::new(b"A");
        let mut empty: [u16; 0] = [];
        let mut dest2 = Utf16Destination::new(&mut empty);
        assert!(matches!(
            dest2.copy_ascii_from_check_space_bmp(&mut src2),
            CopyAsciiResult::Stop((DecoderResult::OutputFull, 0, 0))
        ));
    }

    #[test]
    fn utf16_copy_ascii_non_ascii_at_position_zero_goes_on_without_writing() {
        let mut src = ByteSource::new(b"\xC3\xA9");
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        if let CopyAsciiResult::GoOn((byte, handle)) = dest.copy_ascii_from_check_space_bmp(&mut src) {
            assert_eq!(byte, 0xC3);
            assert_eq!(handle.written(), 0);
            handle.commit();
        } else {
            panic!("expected the non-ASCII byte to stop the ascii run at zero");
        }
        // the non-ASCII byte is consumed (skipped) but nothing was written
        assert_eq!(src.consumed(), 1);
        assert_eq!(dest.written(), 0);
    }

    #[test]
    fn utf16_copy_ascii_non_ascii_last_byte_goes_on_after_prefix() {
        let mut src = ByteSource::new(b"AB\xE2");
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        if let CopyAsciiResult::GoOn((byte, handle)) = dest.copy_ascii_from_check_space_bmp(&mut src) {
            assert_eq!(byte, 0xE2);
            handle.write_bmp(0x20AC);
        } else {
            panic!("expected a non-ASCII tail byte");
        }
        assert_eq!(src.consumed(), 3);
        assert_eq!(dest.written(), 3);
        assert_eq!(&out[..3], [0x41, 0x42, 0x20AC]);
    }

    #[test]
    fn utf16_copy_ascii_non_ascii_in_middle_goes_on_and_drains_afterwards() {
        let mut src = ByteSource::new(b"AY\xC2B");
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        if let CopyAsciiResult::GoOn((byte, handle)) = dest.copy_ascii_from_check_space_bmp(&mut src) {
            assert_eq!(byte, 0xC2);
            handle.write_bmp(0x00A2);
        } else {
            panic!("expected a mid-stream non-ASCII byte");
        }
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::InputEmpty, 4, 4))
        ));
        assert_eq!(out, [0x41, 0x59, 0x00A2, 0x42]);
    }

    // copy_ascii_from_check_space_bmp on a UTF-8 destination.

    #[test]
    fn utf8_copy_ascii_reports_input_empty_when_ascii_source_is_shorter() {
        let mut src = ByteSource::new(b"AB");
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::InputEmpty, 2, 2))
        ));
        assert_eq!(&out[..2], b"AB");
    }

    #[test]
    fn utf8_copy_ascii_reports_output_full_when_destination_is_shorter() {
        let mut src = ByteSource::new(b"ABCDEFGHIJKL");
        let mut out = [0u8; 8];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::OutputFull, 8, 8))
        ));
        assert_eq!(out, *b"ABCDEFGH");
        assert_eq!(src.consumed(), 8);
    }

    #[test]
    fn utf8_copy_ascii_empty_source_or_empty_destination_stops_without_progress() {
        let mut src = ByteSource::new(b"");
        let mut out = [0u8; 2];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::InputEmpty, 0, 0))
        ));
        let mut src2 = ByteSource::new(b"A");
        let mut empty: [u8; 0] = [];
        let mut dest2 = Utf8Destination::new(&mut empty);
        assert!(matches!(
            dest2.copy_ascii_from_check_space_bmp(&mut src2),
            CopyAsciiResult::Stop((DecoderResult::OutputFull, 0, 0))
        ));
    }

    #[test]
    fn utf8_copy_ascii_non_ascii_at_zero_goes_on_when_room_remains() {
        let mut src = ByteSource::new(b"\xC3\xA9");
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let CopyAsciiResult::GoOn((byte, handle)) = dest.copy_ascii_from_check_space_bmp(&mut src) {
            assert_eq!(byte, 0xC3);
            handle.write_bmp(0x00E9);
        } else {
            panic!("expected a go-on at position zero");
        }
        // only the lead byte was consumed; the trail byte waits for the next round
        assert_eq!(src.consumed(), 1);
        assert_eq!(dest.written(), 2);
        assert_eq!(out, [0xC3, 0xA9, 0, 0]);
    }

    #[test]
    fn utf8_copy_ascii_non_ascii_without_three_free_bytes_stops_unconsumed() {
        let mut src = ByteSource::new(b"A\xC3");
        let mut out = [0u8; 2];
        let mut dest = Utf8Destination::new(&mut out);
        assert!(matches!(
            dest.copy_ascii_from_check_space_bmp(&mut src),
            CopyAsciiResult::Stop((DecoderResult::OutputFull, 1, 1))
        ));
        // unlike the UTF-16 variant, the non-ASCII lead byte stays unread
        assert_eq!(read_one_byte(&mut src), Some(0xC3));
        assert_eq!(src.consumed(), 1);
        assert_eq!(out, [0x41, 0]);
    }

    #[test]
    fn utf8_copy_ascii_non_ascii_last_byte_with_room_goes_on() {
        let mut src = ByteSource::new(b"A\xC3");
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        if let CopyAsciiResult::GoOn((byte, handle)) = dest.copy_ascii_from_check_space_bmp(&mut src) {
            assert_eq!(byte, 0xC3);
            assert_eq!(handle.written(), 1);
            handle.commit();
        } else {
            panic!("expected a go-on after the ascii prefix");
        }
        assert_eq!(src.consumed(), 2);
        assert_eq!(dest.written(), 1);
    }

    #[test]
    fn utf8_copy_ascii_astral_variant_requires_four_free_bytes() {
        let mut src = ByteSource::new(b"A\xF0");
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        // after copying 'A' only three bytes remain, so the astral handle is
        // withheld and the lead byte stays unread
        assert!(matches!(
            dest.copy_ascii_from_check_space_astral(&mut src),
            CopyAsciiResult::Stop((DecoderResult::OutputFull, 1, 1))
        ));
        assert_eq!(src.consumed(), 1);
        let mut src2 = ByteSource::new(b"A\xF0");
        let mut out2 = [0u8; 5];
        let mut dest2 = Utf8Destination::new(&mut out2);
        if let CopyAsciiResult::GoOn((byte, handle)) = dest2.copy_ascii_from_check_space_astral(&mut src2) {
            assert_eq!(byte, 0xF0);
            assert_eq!(handle.written(), 1);
            handle.commit();
        } else {
            panic!("four free bytes must admit the astral handle");
        }
        assert_eq!(src2.consumed(), 2);
    }

    // Utf16Destination::copy_utf16_from.

    #[test]
    fn utf16_dest_copy_utf16_le_copies_bmp_run_and_drains_source() {
        let bytes = [0x41u8, 0x00, 0x42, 0x00];
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(src.consumed(), 4);
        assert_eq!(dest.written(), 2);
        assert_eq!(&out[..2], [0x0041, 0x0042]);
    }

    #[test]
    fn utf16_dest_copy_utf16_be_swaps_units() {
        let bytes = [0x00u8, 0x41, 0x00, 0x42];
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<BigEndian>(&mut src), None);
        assert_eq!(out, [0x0041, 0x0042]);
        assert_eq!(src.consumed(), 4);
    }

    #[test]
    fn utf16_dest_copy_utf16_odd_length_source_leaves_stray_byte() {
        let bytes = [0x41u8, 0x00, 0x42];
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        // the half code unit is not consumed
        assert_eq!(src.consumed(), 2);
        assert_eq!(dest.written(), 1);
        assert_eq!(read_one_byte(&mut src), Some(0x42));
    }

    #[test]
    fn utf16_dest_copy_utf16_uses_destination_room_as_window() {
        let bytes = [0x41u8, 0x00, 0x42, 0x00, 0x43, 0x00];
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(out, [0x0041, 0x0042]);
        assert_eq!(src.consumed(), 4);
        // the rest lands in a fresh destination on the next round
        let mut out2 = [0u16; 1];
        let mut dest2 = Utf16Destination::new(&mut out2);
        assert_eq!(dest2.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(out2, [0x0043]);
        assert_eq!(src.consumed(), 6);
    }

    #[test]
    fn utf16_dest_copy_utf16_empty_window_returns_none_without_progress() {
        let mut src = ByteSource::new(b"");
        let mut out = [0u16; 2];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(dest.written(), 0);
        // also when the source has data but the destination is full
        let bytes = [0x41u8, 0x00];
        let mut src2 = ByteSource::new(&bytes);
        let mut full: [u16; 0] = [];
        let mut dest2 = Utf16Destination::new(&mut full);
        assert_eq!(dest2.copy_utf16_from::<LittleEndian>(&mut src2), None);
        assert_eq!(src2.consumed(), 0);
    }

    #[test]
    fn utf16_dest_copy_utf16_trailing_high_surrogate_is_trimmed_and_left_unread() {
        let bytes = [0x41u8, 0x00, 0x3D, 0xD8]; // 'A' then U+D83D in native order
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        // the lead surrogate may pair with the next chunk, so it is trimmed
        assert_eq!(src.consumed(), 2);
        assert_eq!(dest.written(), 1);
        assert_eq!(out[0], 0x0041);
    }

    #[test]
    fn utf16_dest_copy_utf16_valid_pair_copies_both_units() {
        let bytes = [0x41u8, 0x00, 0x3D, 0xD8, 0x00, 0xDE]; // 'A' U+D83D U+DE00
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(src.consumed(), 6);
        assert_eq!(dest.written(), 3);
        assert_eq!(out[0], 0x0041);
        // `copy_bmp_to` stops before the lead surrogate without writing it, so
        // the pair branch has to place both units itself.
        assert_eq!(out[1], 0xD83D);
        assert_eq!(out[2], 0xDE00);
    }

    #[test]
    fn utf16_dest_copy_utf16_swaps_both_units_of_a_big_endian_pair() {
        let bytes = [0x00u8, 0x41, 0xD8, 0x3D, 0xDE, 0x00]; // 'A' U+1F600, BE
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<BigEndian>(&mut src), None);
        assert_eq!(src.consumed(), 6);
        assert_eq!(dest.written(), 3);
        assert_eq!(&out[..3], [0x0041, 0xD83D, 0xDE00]);
    }

    #[test]
    fn utf16_dest_copy_utf16_low_surrogate_reports_malformed_positions() {
        let bytes = [0x41u8, 0x00, 0x00, 0xDE]; // 'A' then a lone low surrogate
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), Some((4, 1)));
        assert_eq!(src.consumed(), 4);
        assert_eq!(dest.written(), 1);
        assert_eq!(out[0], 0x0041);
    }

    #[test]
    fn utf16_dest_copy_utf16_unpaired_high_surrogate_reports_malformed_positions() {
        let bytes = [0x3Du8, 0xD8, 0x42, 0x00]; // lead surrogate then 'B' (not low)
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u16; 4];
        let mut dest = Utf16Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), Some((2, 0)));
        assert_eq!(src.consumed(), 2);
        assert_eq!(dest.written(), 0);
    }

    // Utf8Destination::copy_utf16_from.

    #[test]
    fn utf8_dest_copy_utf16_reencodes_bmp_run() {
        let bytes = [0x41u8, 0x00, 0xE9, 0x00]; // 'A' U+00E9 in native order
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u8; 8];
        let mut dest = Utf8Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(src.consumed(), 4);
        assert_eq!(dest.written(), 3);
        assert_eq!(&out[..3], [0x41, 0xC3, 0xA9]);
    }

    #[test]
    fn utf8_dest_copy_utf16_reencodes_astral_pair() {
        let bytes = [0x3Du8, 0xD8, 0x00, 0xDE]; // U+1F600 as a surrogate pair
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(src.consumed(), 4);
        assert_eq!(dest.written(), 4);
        assert_eq!(out, [0xF0, 0x9F, 0x98, 0x80]);
    }

    #[test]
    fn utf8_dest_copy_utf16_big_endian_source_swaps_before_encoding() {
        let bytes = [0x00u8, 0x41, 0xD8, 0x3D, 0xDE, 0x00]; // 'A' U+1F600, BE
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u8; 5];
        let mut dest = Utf8Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<BigEndian>(&mut src), None);
        assert_eq!(src.consumed(), 6);
        assert_eq!(dest.written(), 5);
        assert_eq!(out, [0x41, 0xF0, 0x9F, 0x98, 0x80]);
    }

    #[test]
    fn utf8_dest_copy_utf16_stops_when_non_ascii_cannot_fit_three_free_bytes() {
        let bytes = [0x41u8, 0x00, 0x42, 0x00, 0xE9, 0x00];
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u8; 4];
        let mut dest = Utf8Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), None);
        // exactly two units (and three bytes) were taken; the non-ASCII
        // unit stays unread until a roomier destination arrives
        assert_eq!(dest.written(), 2);
        assert_eq!(src.consumed(), 4);
        let mut out2 = [0u8; 8];
        let mut dest2 = Utf8Destination::new(&mut out2);
        assert_eq!(dest2.copy_utf16_from::<LittleEndian>(&mut src), None);
        assert_eq!(&out2[..2], [0xC3, 0xA9]);
        assert_eq!(src.consumed(), 6);
    }

    #[test]
    fn utf8_dest_copy_utf16_reports_malformed_unpaired_high_surrogate() {
        let bytes = [0x3Du8, 0xD8, 0x42, 0x00]; // lead surrogate then 'B'
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u8; 8];
        let mut dest = Utf8Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), Some((2, 0)));
        assert_eq!(src.consumed(), 2);
        assert_eq!(dest.written(), 0);
    }

    #[test]
    fn utf8_dest_copy_utf16_malformed_low_surrogate_counts_as_consumed() {
        // U+1F600 followed by a lone low surrogate: the error marker is
        // reported after the offending unit was counted as read
        let bytes = [0x3Du8, 0xD8, 0x00, 0xDE, 0x01, 0xDD];
        let mut src = ByteSource::new(&bytes);
        let mut out = [0u8; 8];
        let mut dest = Utf8Destination::new(&mut out);
        assert_eq!(dest.copy_utf16_from::<LittleEndian>(&mut src), Some((6, 4)));
        assert_eq!(src.consumed(), 6);
        assert_eq!(dest.written(), 4);
        assert_eq!(&out[..4], [0xF0, 0x9F, 0x98, 0x80]);
    }

    // UnalignedU16Slice.

    #[test]
    fn unaligned_slice_reads_units_and_reports_emptiness() {
        let empty = unaligned_over(&[]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        let units: [u16; 3] = [0x0041, 0x20AC, 0xD83D];
        // SAFETY: the pointer and length both come from the live `units` array.
        let slice = unsafe { UnalignedU16Slice::new(units.as_ptr().cast::<u8>(), units.len()) };
        assert!(!slice.is_empty());
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.at(0), 0x0041);
        assert_eq!(slice.at(2), 0xD83D);
    }

    #[test]
    fn unaligned_slice_tail_windows_from_offset() {
        let bytes = [0x41u8, 0x00, 0x42, 0x00];
        let slice = unaligned_over(&bytes);
        let tail = slice.tail(1);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail.at(0), 0x0042);
        assert!(slice.tail(2).is_empty());
    }

    #[test]
    fn unaligned_slice_trim_last_shortens_window() {
        let bytes = [0x41u8, 0x00, 0x3D, 0xD8];
        let mut slice = unaligned_over(&bytes);
        slice.trim_last();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice.at(0), 0x0041);
    }

    #[test]
    fn unaligned_slice_copy_bmp_writes_all_units_and_returns_none() {
        let bytes = [0x41u8, 0x00, 0xAC, 0x20];
        let slice = unaligned_over(&bytes);
        let mut out = [0u16; 2];
        assert_eq!(slice.copy_bmp_to::<LittleEndian>(&mut out), None);
        assert_eq!(out, [0x0041, 0x20AC]);
    }

    #[test]
    fn unaligned_slice_copy_bmp_byte_swaps_for_opposite_endian() {
        let bytes = [0x00u8, 0x41, 0x20, 0xAC]; // big-endian wire order
        let slice = unaligned_over(&bytes);
        let mut out = [0u16; 2];
        assert_eq!(slice.copy_bmp_to::<BigEndian>(&mut out), None);
        assert_eq!(out, [0x0041, 0x20AC]);
    }

    #[test]
    fn unaligned_slice_copy_bmp_stops_before_high_surrogate_without_writing_it() {
        let bytes = [0x41u8, 0x00, 0x3D, 0xD8, 0xAC, 0x20];
        let slice = unaligned_over(&bytes);
        let mut out = [0u16; 3];
        out[1] = 0xFFFF; // poison: unlike upstream, this copy never speculatively writes
        assert_eq!(slice.copy_bmp_to::<LittleEndian>(&mut out), Some((0xD83D, 1)));
        assert_eq!(out, [0x0041, 0xFFFF, 0x0000]);
        assert_eq!(
            slice
                .tail(1)
                .copy_bmp_to::<LittleEndian>(&mut [0u16; 1]),
            Some((0xD83D, 0))
        );
    }

    #[test]
    fn unaligned_slice_copy_bmp_stops_before_low_surrogate_too() {
        let bytes = [0x41u8, 0x00, 0x00, 0xDE];
        let slice = unaligned_over(&bytes);
        let mut out = [0u16; 2];
        assert_eq!(slice.copy_bmp_to::<LittleEndian>(&mut out), Some((0xDE00, 1)));
        assert_eq!(out[0], 0x0041);
    }

    #[test]
    fn unaligned_slice_copy_bmp_truncates_when_destination_is_shorter() {
        let bytes = [0x41u8, 0x00, 0x42, 0x00, 0x43, 0x00];
        let slice = unaligned_over(&bytes);
        let mut out = [0u16; 2];
        assert_eq!(slice.copy_bmp_to::<LittleEndian>(&mut out), None);
        assert_eq!(out, [0x0041, 0x0042]);
        // a surrogate beyond the destination window is not seen
        let with_surrogate = [0x41u8, 0x00, 0x3D, 0xD8];
        let slice2 = unaligned_over(&with_surrogate);
        let mut out2 = [0u16; 1];
        assert_eq!(slice2.copy_bmp_to::<LittleEndian>(&mut out2), None);
        assert_eq!(out2, [0x0041]);
    }

    // Unicode / NonAscii enums.

    #[test]
    fn unicode_ascii_variant_equality_and_matching() {
        let value = Unicode::Ascii(b'A');
        assert_eq!(value, Unicode::Ascii(b'A'));
        assert_ne!(value, Unicode::Ascii(b'B'));
        match value {
            Unicode::Ascii(byte) => assert_eq!(byte, b'A'),
            Unicode::NonAscii(_) => panic!("expected the ascii variant"),
        }
    }

    #[test]
    fn unicode_non_ascii_variants_carry_code_unit_and_char() {
        let bmp = Unicode::NonAscii(NonAscii::BmpExclAscii(0x20AC));
        let astral = Unicode::NonAscii(NonAscii::Astral('\u{1F600}'));
        assert_ne!(bmp, astral);
        match bmp {
            Unicode::NonAscii(NonAscii::BmpExclAscii(unit)) => assert_eq!(unit, 0x20AC),
            _ => panic!("expected a bmp-excluding-ascii code unit"),
        }
        match astral {
            Unicode::NonAscii(NonAscii::Astral(c)) => assert_eq!(c, '\u{1F600}'),
            _ => panic!("expected an astral character"),
        }
    }

    #[test]
    fn non_ascii_variants_are_not_equal_across_representations() {
        assert_ne!(NonAscii::BmpExclAscii(0x80), NonAscii::Astral('\u{0080}'));
        assert_eq!(NonAscii::Astral('\u{1F600}'), NonAscii::Astral('\u{1F600}'));
        assert_eq!(NonAscii::BmpExclAscii(0xD800), NonAscii::BmpExclAscii(0xD800));
    }

    #[test]
    fn unicode_variants_reproduce_the_codec_handoff_shape() {
        // codecs signal one non-ASCII scalar per round; both storage shapes
        // must survive a match round-trip without lossy conversion
        let cases = [
            Unicode::NonAscii(NonAscii::BmpExclAscii(0xA5)),
            Unicode::NonAscii(NonAscii::Astral('\u{10FFFF}')),
        ];
        for case in cases {
            match &case {
                Unicode::NonAscii(NonAscii::BmpExclAscii(unit)) => {
                    assert_eq!(Unicode::NonAscii(NonAscii::BmpExclAscii(*unit)), case);
                }
                Unicode::NonAscii(NonAscii::Astral(c)) => {
                    assert_eq!(Unicode::NonAscii(NonAscii::Astral(*c)), case);
                }
                Unicode::Ascii(_) => panic!("expected a non-ascii case"),
            }
        }
    }

    // Utf16Source reading.

    #[test]
    fn utf16_source_empty_reports_full_consumed() {
        let mut src = Utf16Source::new(&[]);
        assert!(matches!(src.check_available(), Space::Full(0)));
        assert_eq!(read_one_u16(&mut src), None);
    }

    #[test]
    fn utf16_source_reads_ascii_units_one_at_a_time() {
        let mut src = Utf16Source::new(&[0x0041, 0x0042]);
        assert_eq!(read_one_u16(&mut src), Some('A'));
        assert_eq!(read_one_u16(&mut src), Some('B'));
        assert!(matches!(src.check_available(), Space::Full(2)));
    }

    #[test]
    fn utf16_source_reads_bmp_scalars() {
        let mut src = Utf16Source::new(&[0x20AC, 0xD7FF]);
        assert_eq!(read_one_u16(&mut src), Some('\u{20AC}'));
        assert_eq!(read_one_u16(&mut src), Some('\u{D7FF}'));
        assert_eq!(src.consumed(), 2);
    }

    #[test]
    fn utf16_source_combines_surrogate_pairs_and_advances_two_units() {
        let mut src = Utf16Source::new(&[0xD83D, 0xDE00]);
        assert_eq!(read_one_u16(&mut src), Some('\u{1F600}'));
        assert_eq!(src.consumed(), 2);
        assert!(matches!(src.check_available(), Space::Full(2)));
    }

    #[test]
    fn utf16_source_unpaired_high_surrogate_yields_replacement_advancing_one() {
        let mut src = Utf16Source::new(&[0xD83D, 0x0041]);
        if let Space::Available(handle) = src.check_available() {
            let (c, unread) = handle.read();
            assert_eq!(c, core::char::REPLACEMENT_CHARACTER);
            // only the lead surrogate itself was stepped over, not its follower
            assert_eq!(unread.consumed(), 1);
            assert_eq!(unread.unread(), 1);
        } else {
            panic!("expected available input");
        }
        assert_eq!(src.consumed(), 0);
        assert_eq!(read_one_u16(&mut src), Some(core::char::REPLACEMENT_CHARACTER));
        assert_eq!(read_one_u16(&mut src), Some('A'));
    }

    #[test]
    fn utf16_source_high_surrogate_at_end_yields_replacement_advancing_one() {
        let mut src = Utf16Source::new(&[0xD83D]);
        assert_eq!(read_one_u16(&mut src), Some(core::char::REPLACEMENT_CHARACTER));
        assert_eq!(src.consumed(), 1);
    }

    #[test]
    fn utf16_source_lone_low_surrogate_yields_replacement() {
        let mut src = Utf16Source::new(&[0xDE00]);
        assert_eq!(read_one_u16(&mut src), Some(core::char::REPLACEMENT_CHARACTER));
        assert_eq!(src.consumed(), 1);
    }

    #[test]
    fn utf16_source_unread_restores_pair_position() {
        let mut src = Utf16Source::new(&[0xD83D, 0xDE00, 0x0041]);
        if let Space::Available(handle) = src.check_available() {
            let (c, unread) = handle.read();
            assert_eq!(c, '\u{1F600}');
            assert_eq!(unread.unread(), 2);
        } else {
            panic!("expected available input");
        }
        assert_eq!(src.consumed(), 0);
        assert_eq!(read_one_u16(&mut src), Some('\u{1F600}'));
        assert_eq!(read_one_u16(&mut src), Some('A'));
    }

    #[test]
    fn utf16_source_consumed_never_moves_back_without_unread() {
        let mut src = Utf16Source::new(&[0x0041, 0x0042, 0x0043]);
        let mut marks = Vec::new();
        while let Space::Available(handle) = src.check_available() {
            let (_c, unread) = handle.read();
            let src_ref = unread.commit();
            marks.push(src_ref.consumed());
        }
        assert_eq!(marks, vec![1, 2, 3]);
    }

    // Utf8Source reading (beyond the existing tests).

    #[test]
    fn utf8_source_check_available_reports_consumed_when_exhausted() {
        let mut src = Utf8Source::new("a\u{1F600}");
        while let Space::Available(handle) = src.check_available() {
            let (_c, unread) = handle.read();
            unread.commit();
        }
        let total = src.consumed();
        assert_eq!(total, "a\u{1F600}".len());
        assert!(matches!(src.check_available(), Space::Full(_)));
    }

    #[test]
    fn utf8_source_unread_of_multibyte_scalar_reports_end_position() {
        let mut src = Utf8Source::new("\u{1F600}A");
        if let Space::Available(handle) = src.check_available() {
            let (c, unread) = handle.read();
            assert_eq!(c, '\u{1F600}');
            assert_eq!(unread.consumed(), 4);
            assert_eq!(unread.unread(), 4);
        } else {
            panic!("expected available input");
        }
        assert_eq!(src.consumed(), 0);
        assert_eq!(read_one(&mut src), Some('\u{1F600}'));
        assert_eq!(read_one(&mut src), Some('A'));
    }

    #[test]
    fn utf8_source_rewound_handle_yields_the_same_scalar_again() {
        let mut src = Utf8Source::new("\u{20AC}");
        let (first, rewind_pos) = if let Space::Available(handle) = src.check_available() {
            let (c, unread) = handle.read();
            (c, unread.unread())
        } else {
            panic!("expected available input");
        };
        assert_eq!(first, '\u{20AC}');
        assert_eq!(rewind_pos, 3);
        assert_eq!(src.consumed(), 0);
        assert_eq!(read_one(&mut src), Some('\u{20AC}'));
        assert_eq!(src.consumed(), 3);
    }

    // copy_unaligned_basic_latin_to_ascii.

    #[test]
    fn copy_basic_latin_stops_after_copying_fully_ascii_window() {
        let units: [u16; 2] = [0x0041, 0x0042];
        // SAFETY: pointer and length come from the live `units` array.
        let src = unsafe { UnalignedU16Slice::new(units.as_ptr().cast::<u8>(), units.len()) };
        let mut out = [0u8; 2];
        assert_eq!(
            copy_unaligned_basic_latin_to_ascii::<LittleEndian>(src, &mut out),
            CopyAsciiResult::Stop(2)
        );
        assert_eq!(out, [0x41, 0x42]);
    }

    #[test]
    fn copy_basic_latin_goes_on_at_first_non_ascii() {
        let units: [u16; 3] = [0x0041, 0x00E9, 0x0042];
        // SAFETY: pointer and length come from the live `units` array.
        let src = unsafe { UnalignedU16Slice::new(units.as_ptr().cast::<u8>(), units.len()) };
        let mut out = [0u8; 3];
        assert_eq!(
            copy_unaligned_basic_latin_to_ascii::<LittleEndian>(src, &mut out),
            CopyAsciiResult::GoOn((0x00E9, 1))
        );
        assert_eq!(out[0], 0x41);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn copy_basic_latin_byte_swaps_for_opposite_endian() {
        let bytes = [0x00u8, 0x41, 0x00, 0x42]; // big-endian wire order
        let src = unaligned_over(&bytes);
        let mut out = [0u8; 2];
        assert_eq!(
            copy_unaligned_basic_latin_to_ascii::<BigEndian>(src, &mut out),
            CopyAsciiResult::Stop(2)
        );
        assert_eq!(out, [0x41, 0x42]);
    }

    #[test]
    fn copy_basic_latin_truncates_at_destination_window() {
        let units: [u16; 3] = [0x0041, 0x0042, 0x00E9];
        // SAFETY: pointer and length come from the live `units` array.
        let src = unsafe { UnalignedU16Slice::new(units.as_ptr().cast::<u8>(), units.len()) };
        let mut out = [0u8; 2];
        assert_eq!(
            copy_unaligned_basic_latin_to_ascii::<LittleEndian>(src, &mut out),
            CopyAsciiResult::Stop(2)
        );
        assert_eq!(out, [0x41, 0x42]);
    }

    #[test]
    fn copy_basic_latin_empty_source_or_destination_stops_at_zero() {
        let empty = unaligned_over(&[]);
        let mut out = [0u8; 2];
        assert_eq!(
            copy_unaligned_basic_latin_to_ascii::<LittleEndian>(empty, &mut out),
            CopyAsciiResult::Stop(0)
        );
        let units: [u16; 1] = [0x0041];
        // SAFETY: pointer and length come from the live `units` array.
        let src = unsafe { UnalignedU16Slice::new(units.as_ptr().cast::<u8>(), units.len()) };
        let mut no_room: [u8; 0] = [];
        assert_eq!(
            copy_unaligned_basic_latin_to_ascii::<LittleEndian>(src, &mut no_room),
            CopyAsciiResult::Stop(0)
        );
    }

    // convert_unaligned_utf16_to_utf8 boundaries.

    #[test]
    fn convert_unaligned_utf16_returns_zero_for_undersized_destination() {
        let bytes = [0x41u8, 0x00];
        let src = unaligned_over(&bytes);
        let mut out = [0u8; 3];
        assert_eq!(
            convert_unaligned_utf16_to_utf8::<LittleEndian>(src, &mut out),
            (0, 0, false)
        );
    }

    #[test]
    fn convert_unaligned_utf16_needs_four_bytes_to_attempt_non_ascii() {
        // with 5 bytes (dst_len - 3 == 2) only the first non-ASCII fits
        let bytes = [0xE9u8, 0x00, 0xE9, 0x00]; // two U+00E9 units
        let src = unaligned_over(&bytes);
        let mut out = [0u8; 5];
        let (read, written, err) = convert_unaligned_utf16_to_utf8::<LittleEndian>(src, &mut out);
        assert_eq!((read, written, err), (1, 2, false));
        assert_eq!(out, [0xC3, 0xA9, 0, 0, 0]);
        // the remainder is taken on the next round with a fresh destination
        let mut out2 = [0u8; 4];
        let (read2, written2, err2) = convert_unaligned_utf16_to_utf8::<LittleEndian>(src.tail(1), &mut out2);
        assert_eq!((read2, written2, err2), (1, 2, false));
        assert_eq!(&out2[..2], [0xC3, 0xA9]);
    }

    // validate_bmp_stride.

    #[test]
    fn validate_bmp_stride_accepts_surrogate_free_run() {
        let stride = [0x20ACu16; 16];
        assert_eq!(validate_bmp_stride(&stride), None);
    }

    #[test]
    fn validate_bmp_stride_finds_surrogates_at_edges_and_middle() {
        let mut stride = [0x0041u16; 16];
        stride[0] = 0xD800;
        assert_eq!(validate_bmp_stride(&stride), Some(0));
        let mut stride = [0x0041u16; 16];
        stride[15] = 0xDFFF;
        assert_eq!(validate_bmp_stride(&stride), Some(15));
        let mut stride = [0x0041u16; 16];
        stride[7] = 0xDC00;
        assert_eq!(validate_bmp_stride(&stride), Some(7));
    }

    #[test]
    fn validate_bmp_stride_rejects_values_just_outside_surrogate_range() {
        let mut stride = [0x0041u16; 16];
        stride[3] = 0xD7FF;
        assert_eq!(validate_bmp_stride(&stride), None);
        let mut stride = [0x0041u16; 16];
        stride[3] = 0xE000;
        assert_eq!(validate_bmp_stride(&stride), None);
    }

    #[test]
    fn validate_latin1_str_stride_never_flags_any_byte() {
        // The `all` guard (`b & 0xC0 != 0x80 || b <= 0xC3`) holds for every
        // possible u8: continuation bytes (0x80..=0xBF) pass via `<= 0xC3`,
        // everything else never matches `0x80`. So the function reports `None`
        // for every stride and its scan loop is dead code. Pinned as-is.
        let all_high = [0xFFu8; 16];
        assert_eq!(validate_latin1_str_stride(&all_high), None);
        let continuations = [0x80u8; 16];
        assert_eq!(validate_latin1_str_stride(&continuations), None);
        let ascii = [0x41u8; 16];
        assert_eq!(validate_latin1_str_stride(&ascii), None);
    }
}
