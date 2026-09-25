//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use crate::compression_error::{CompressorError, CompressorResult};
use alloc::vec::Vec;

/// Three-byte LZ-match frame magic (`LM` + version).
pub const LZ_MATCH_MAGIC: &[u8; 3] = b"LM\x01";
/// Three-byte Bitward frame magic (`BW` + version).
pub const BITWARD_MAGIC: &[u8; 3] = b"BW\x01";
/// Three-byte dictionary / valmap frame magic (`DI` + version).
pub const DICTIONARY_MAGIC: &[u8; 3] = b"DI\x01";
/// Three-byte Huffman frame magic (`HF` + version).
pub const HUFFMAN_MAGIC: &[u8; 3] = b"HF\x01";
/// Three-byte substring frame magic (`SX` + version).
pub const SUBSTRING_MAGIC: &[u8; 3] = b"SX\x01";
/// Three-byte dynamic-substring frame magic (`DX` + version).
pub const DYNSU_MAGIC: &[u8; 3] = b"DX\x01";
/// Three-byte delta frame magic (`DL` + version).
pub const DELTA_MAGIC: &[u8; 3] = b"DL\x01";

/// Identifies an FcWare frame codec from its magic bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameKind {
    /// Byte LZ match / literal stream.
    LzMatch,
    /// Bitward u16 value packing.
    Bitward,
    /// Dictionary predictive u16 coding.
    Dictionary,
    /// Canonical Huffman entropy coding.
    Huffman,
    /// Fixed-window substring matches.
    Substring,
    /// Blocked dynamic substring matches.
    Dysu,
    /// Byte delta / RLE residuals.
    Delta,
}

impl FrameKind {
    /// Returns the three-byte magic for this frame kind.
    #[inline]
    #[must_use]
    pub const fn magic(self) -> &'static [u8; 3] {
        match self {
            Self::LzMatch => LZ_MATCH_MAGIC,
            Self::Bitward => BITWARD_MAGIC,
            Self::Dictionary => DICTIONARY_MAGIC,
            Self::Huffman => HUFFMAN_MAGIC,
            Self::Substring => SUBSTRING_MAGIC,
            Self::Dysu => DYNSU_MAGIC,
            Self::Delta => DELTA_MAGIC,
        }
    }

    /// Parses a frame kind from a three-byte magic prefix.
    #[inline]
    #[must_use]
    pub fn from_magic(magic: &[u8]) -> Option<Self> {
        const LM: u32 = b'L' as u32 | ((b'M' as u32) << 8) | (1u32 << 16);
        const BW: u32 = b'B' as u32 | ((b'W' as u32) << 8) | (1u32 << 16);
        const DI: u32 = b'D' as u32 | ((b'I' as u32) << 8) | (1u32 << 16);
        const HF: u32 = b'H' as u32 | ((b'F' as u32) << 8) | (1u32 << 16);
        const SX: u32 = b'S' as u32 | ((b'X' as u32) << 8) | (1u32 << 16);
        const DX: u32 = b'D' as u32 | ((b'X' as u32) << 8) | (1u32 << 16);
        const DL: u32 = b'D' as u32 | ((b'L' as u32) << 8) | (1u32 << 16);

        let prefix = magic.get(..3)?;
        // SAFETY: `prefix` is exactly three bytes.
        let tag = unsafe {
            let p = prefix.as_ptr();
            u32::from(*p) | (u32::from(*p.add(1)) << 8) | (u32::from(*p.add(2)) << 16)
        };
        match tag {
            LM => Some(Self::LzMatch),
            BW => Some(Self::Bitward),
            DI => Some(Self::Dictionary),
            HF => Some(Self::Huffman),
            SX => Some(Self::Substring),
            DX => Some(Self::Dysu),
            DL => Some(Self::Delta),
            _ => None,
        }
    }
}

/// Reads a big-endian `u16` from `frame` at `position` and advances the cursor.
#[inline]
pub fn frame_read_u16(frame: &[u8], position: &mut usize) -> CompressorResult<u16> {
    let start = *position;
    let end = start
        .checked_add(2)
        .ok_or(CompressorError::TruncatedFrame)?;
    if end > frame.len() {
        return Err(CompressorError::TruncatedFrame);
    }
    // SAFETY: `end <= frame.len()` and the two bytes at `start` are in-bounds.
    let value = unsafe {
        let p = frame.as_ptr().add(start);
        u16::from_be_bytes([*p, *p.add(1)])
    };
    *position = end;
    Ok(value)
}

/// Reads a big-endian `u32` from `frame` at `position` and advances the cursor.
#[inline]
pub fn frame_read_u32(frame: &[u8], position: &mut usize) -> CompressorResult<u32> {
    let start = *position;
    let end = start
        .checked_add(4)
        .ok_or(CompressorError::TruncatedFrame)?;
    if end > frame.len() {
        return Err(CompressorError::TruncatedFrame);
    }
    // SAFETY: `end <= frame.len()` and four bytes at `start` are in-bounds.
    let value = unsafe {
        let p = frame.as_ptr().add(start);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)])
    };
    *position = end;
    Ok(value)
}

/// Writes `bytes` into `output` at `cursor`, advancing the cursor on success.
#[inline]
pub fn write_bytes(output: &mut [u8], cursor: &mut usize, bytes: &[u8]) -> CompressorResult<()> {
    let start = *cursor;
    let end = start
        .checked_add(bytes.len())
        .ok_or(CompressorError::OutputTooSmall)?;
    if end > output.len() {
        return Err(CompressorError::OutputTooSmall);
    }
    // SAFETY: `start..end` fits in `output`; regions are non-overlapping with `bytes`
    // because `bytes` is a shared borrow of a different allocation (or disjoint slice).
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), output.as_mut_ptr().add(start), bytes.len());
    }
    *cursor = end;
    Ok(())
}

/// Appends `src` onto `dst`, reserving capacity first. Uses a raw copy on the hot path.
#[inline]
pub fn extend_bytes(dst: &mut Vec<u8>, src: &[u8]) {
    let old_len = dst.len();
    let new_len = old_len.saturating_add(src.len());
    dst.reserve(src.len());
    // SAFETY: reserved capacity covers `new_len`; `src` does not alias `dst`'s buffer
    // because `src` is a shared slice from a distinct allocation or a prior snapshot.
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(old_len), src.len());
        dst.set_len(new_len);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BITWARD_MAGIC, DELTA_MAGIC, DICTIONARY_MAGIC, DYNSU_MAGIC, FrameKind, HUFFMAN_MAGIC, LZ_MATCH_MAGIC,
        SUBSTRING_MAGIC, frame_read_u16, frame_read_u32, write_bytes,
    };
    use crate::compression_error::CompressorError;

    #[test]
    fn magic_round_trip() {
        for kind in [
            FrameKind::LzMatch,
            FrameKind::Bitward,
            FrameKind::Dictionary,
            FrameKind::Huffman,
            FrameKind::Substring,
            FrameKind::Dysu,
            FrameKind::Delta,
        ] {
            assert_eq!(FrameKind::from_magic(kind.magic()), Some(kind));
        }
        assert_eq!(FrameKind::from_magic(b"ZZ\x01"), None);
        assert_eq!(FrameKind::from_magic(b"LM"), None);
    }

    #[test]
    fn magic_constants_match_kinds() {
        assert_eq!(LZ_MATCH_MAGIC, FrameKind::LzMatch.magic());
        assert_eq!(BITWARD_MAGIC, FrameKind::Bitward.magic());
        assert_eq!(DICTIONARY_MAGIC, FrameKind::Dictionary.magic());
        assert_eq!(HUFFMAN_MAGIC, FrameKind::Huffman.magic());
        assert_eq!(SUBSTRING_MAGIC, FrameKind::Substring.magic());
        assert_eq!(DYNSU_MAGIC, FrameKind::Dysu.magic());
        assert_eq!(DELTA_MAGIC, FrameKind::Delta.magic());
    }

    #[test]
    fn read_write_integers_and_bytes() {
        let mut pos = 0usize;
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        assert_eq!(frame_read_u16(&data, &mut pos).expect("u16"), 0x1234);
        assert_eq!(frame_read_u32(&data, &mut pos).expect("u32"), 0x5678_9ABC);
        assert_eq!(
            frame_read_u16(&data, &mut pos),
            Err(CompressorError::TruncatedFrame)
        );

        let mut out = [0u8; 4];
        let mut cursor = 0usize;
        write_bytes(&mut out, &mut cursor, b"ab").expect("w1");
        write_bytes(&mut out, &mut cursor, b"cd").expect("w2");
        assert_eq!(&out, b"abcd");
        assert_eq!(
            write_bytes(&mut out, &mut cursor, b"x"),
            Err(CompressorError::OutputTooSmall)
        );
    }

    #[test]
    fn checked_add_overflow_paths() {
        let data = [1u8, 2];
        let mut pos = usize::MAX - 1;
        assert_eq!(
            frame_read_u16(&data, &mut pos),
            Err(CompressorError::TruncatedFrame)
        );
        let mut out = [0u8; 2];
        let mut cursor = usize::MAX - 1;
        assert_eq!(
            write_bytes(&mut out, &mut cursor, b"xx"),
            Err(CompressorError::OutputTooSmall)
        );
    }
}
