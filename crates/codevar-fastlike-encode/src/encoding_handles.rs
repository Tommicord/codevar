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

use crate::encoding::DecoderResult;
use crate::encoding_ascii::{ascii_to_ascii, ascii_to_basic_latin};
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
    #[inline(always)]
    pub fn tail(&self, from: usize) -> UnalignedU16Slice {
        debug_assert!(from <= self.len);
        let from = from.min(self.len);
        unsafe { UnalignedU16Slice::new(self.ptr.add(from * 2), self.len - from) }
    }
    #[inline(always)]
    pub fn copy_bmp_to<E: Endian>(&self, dst: &mut [u16]) -> Option<(u16, usize)> {
        let n = min(self.len, dst.len());
        for i in 0..n {
            let unit = if E::OPPOSITE_ENDIAN {
                self.at(i).swap_bytes()
            } else {
                self.at(i)
            };
            if unit.wrapping_sub(0xD800) <= (0xDFFF - 0xD800) {
                return Some((unit, i));
            }
            dst[i] = unit;
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
    pub fn write_surrogate_pair(
        self,
        high: u16,
        low: u16,
    ) -> &'a mut Utf16Destination<'b> {
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
    pub fn write_surrogate_pair(
        self,
        high: u16,
        low: u16,
    ) -> &'a mut Utf16Destination<'b> {
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
    ) -> CopyAsciiResult<(DecoderResult, usize, usize), (u8, Utf16BmpHandle<'b, 'a>)>
    {
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
        let (read, written) = crate::encoding_utf8::convert_utf8_to_utf16_up_to_invalid(
            src_remaining,
            dst_remaining,
        );
        source.pos += read;
        self.pos += written;
    }
    #[inline(always)]
    pub fn copy_utf16_from<E: Endian>(
        &mut self,
        source: &mut ByteSource,
    ) -> Option<(usize, usize)> {
        let src_remaining = &source.slice[source.pos..];
        let dst_remaining = &mut self.slice[self.pos..];
        let mut src_unaligned = unsafe {
            UnalignedU16Slice::new(
                src_remaining.as_ptr(),
                min(src_remaining.len() / 2, dst_remaining.len()),
            )
        };
        if src_unaligned.len() == 0 {
            return None;
        }
        let last_unit = if E::OPPOSITE_ENDIAN {
            src_unaligned.at(src_unaligned.len() - 1).swap_bytes()
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
    pub fn write_surrogate_pair(
        self,
        high: u16,
        low: u16,
    ) -> &'a mut Utf8Destination<'b> {
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
    pub fn write_surrogate_pair(
        self,
        high: u16,
        low: u16,
    ) -> &'a mut Utf8Destination<'b> {
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
        debug_assert!(mid_bmp >= 0x80 && mid_bmp < 0x800);
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
            (u32::from(high) << 10) + u32::from(low)
                - (((0xD800u32 << 10) - 0x10000u32) + 0xDC00u32),
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
                        return CopyAsciiResult::Stop((
                            DecoderResult::OutputFull,
                            source.pos,
                            self.pos,
                        ));
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
    ) -> CopyAsciiResult<(DecoderResult, usize, usize), (u8, Utf8AstralHandle<'b, 'a>)>
    {
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
                        return CopyAsciiResult::Stop((
                            DecoderResult::OutputFull,
                            source.pos,
                            self.pos,
                        ));
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
        let valid_len = crate::encoding_utf8::utf8_valid_up_to(&src_remaining[..min_len]);
        dst_remaining[..valid_len].copy_from_slice(&src_remaining[..valid_len]);
        source.pos += valid_len;
        self.pos += valid_len;
    }
    #[inline(always)]
    pub fn copy_utf16_from<E: Endian>(
        &mut self,
        source: &mut ByteSource,
    ) -> Option<(usize, usize)> {
        let src_remaining = &source.slice[source.pos..];
        let dst_remaining = &mut self.slice[self.pos..];
        let mut src_unaligned = unsafe {
            UnalignedU16Slice::new(src_remaining.as_ptr(), src_remaining.len() / 2)
        };
        if src_unaligned.len() == 0 {
            return None;
        }
        let mut last_unit = src_unaligned.at(src_unaligned.len() - 1);
        if E::OPPOSITE_ENDIAN {
            last_unit = last_unit.swap_bytes();
        }
        if crate::encoding::in_range16(last_unit, 0xD800, 0xDC00) {
            src_unaligned.trim_last();
        }
        let (read, written, had_error) =
            convert_unaligned_utf16_to_utf8::<E>(src_unaligned, dst_remaining);
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
        if unit_minus_surrogate_start <= (0xDBFF - 0xD800) {
            if self.pos < self.slice.len() {
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
        let mut non_ascii = match copy_unaligned_basic_latin_to_ascii::<E>(
            src.tail(src_pos),
            &mut dst[dst_pos..],
        ) {
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
    if stride.iter().all(|b| b & 0xC0 != 0x80 || b <= &0xC3) {
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
}
