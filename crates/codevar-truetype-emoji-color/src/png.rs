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
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! A minimal PNG decoder for color font bitmaps.
//!
//! FreeType's `src/sfnt/pngshim.c` (`Load_SBit_Png`) delegates PNG
//! decoding to libpng and then applies one user transform: RGBA bytes
//! are converted to *premultiplied BGRA* (`premultiply_data`), plain
//! RGB becomes opaque BGRA (`convert_bytes_to_data`).  This module
//! reproduces that pipeline without libpng:
//!
//! 1. parse signature, `IHDR`, `PLTE`, `tRNS`, `IDAT`, `IEND`
//!    (critical-chunk CRCs are verified),
//! 2. inflate the concatenated `IDAT` payload with
//!    [`codevar_truetype_gzip::uncompress`] (zlib stream, as PNG
//!    requires),
//! 3. undo the per-row filters (None/Sub/Up/Average/Paeth), for both
//!    non-interlaced and Adam7 images,
//! 4. expand samples to 8-bit RGBA honoring bit depths 1/2/4/8/16 and
//!    color types 0/2/3/4/6 plus `tRNS`,
//! 5. apply the pngshim premultiply/swap transform.
//!
//! ## Deviations from FreeType 2.13
//!
//! * libpng's ancillary chunk handling (iCCP, sRGB, gAMA, ...) is
//!   ignored; unknown *critical* chunks are an error, unknown
//!   ancillary chunks are skipped (libpng's default for safe chunks
//!   is similar, but not identical).
//! * Error codes are [`TtError::INVALID_FILE_FORMAT`] for structural
//!   problems and [`TtError::INVALID_TABLE`] for corrupt pixel data;
//!   FreeType surfaces libpng's longjmp as `Invalid_File_Format`
//!   (or `Out_Of_Memory` for libpng hard errors).
//! * Output is always a premultiplied [`PixelMode::Bgra`] [`Bitmap`]
//!   with top-down rows, ready for [`crate::render`] compositing.

use alloc::vec;
use alloc::vec::Vec;
use codevar_truetype_core::{Bitmap, PixelMode, TtError, TtResult};

/// The 8-byte PNG signature.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
/// The maximum image dimension accepted (mirrors pngshim's `0x7FFF`
/// rasterizer bound).
const MAX_DIMENSION: u32 = 0x7FFF;
/// The maximum number of `IDAT` bytes accepted (guards decompression
/// bombs on corrupt input).
const MAX_IDAT: usize = 64 * 1024 * 1024;

/// A decoded PNG image: premultiplied BGRA pixels in a top-down
/// [`Bitmap`], ready for compositing.
///
/// # Errors
///
/// * [`TtError::INVALID_FILE_FORMAT`] — bad signature, malformed
///   chunk structure, unsupported header values, CRC mismatch, or
///   pixel data inconsistent with the header.
/// * [`TtError::INVALID_TABLE`] — corrupt inflated data (filters,
///   palette/tRNS indices) or a zlib stream error.
/// * [`TtError::OUT_OF_MEMORY`] / [`TtError::ARRAY_TOO_LARGE`] —
///   allocation failure or an image larger than the accepted bounds.
pub fn decode(data: &[u8]) -> TtResult<Bitmap> {
    let image = PngChunks::parse(data)?;
    let mut bitmap = image.expand()?;
    premultiply_and_swap(&mut bitmap)?;
    Ok(bitmap)
}

/// Applies `premultiply_data` of pngshim: RGBA bytes become
/// premultiplied BGRA (alpha 0 zeroes the pixel; opaque pixels only
/// swap channels).
fn premultiply_and_swap(bitmap: &mut Bitmap) -> TtResult<()> {
    let buffer = bitmap.buffer.as_mut_slice();
    for pixel in buffer.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        if alpha == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            pixel[3] = 0;
        } else {
            let red = u32::from(pixel[0]);
            let green = u32::from(pixel[1]);
            let blue = u32::from(pixel[2]);
            if alpha != 0xFF {
                pixel[2] = multiply_alpha(alpha, red) as u8;
                pixel[1] = multiply_alpha(alpha, green) as u8;
                pixel[0] = multiply_alpha(alpha, blue) as u8;
            } else {
                pixel[0] = blue as u8;
                pixel[1] = green as u8;
                pixel[2] = red as u8;
            }
            // `pixel[3]` (alpha) is already in place.
        }
    }
    Ok(())
}

/// `multiply_alpha` of pngshim: `(alpha * color + 0x80 +
/// ((alpha * color + 0x80) >> 8)) >> 8`, the rounded `alpha*color/255`.
#[inline]
fn multiply_alpha(alpha: u32, color: u32) -> u32 {
    let temp = alpha * color + 0x80;
    (temp + (temp >> 8)) >> 8
}

/// The parsed PNG file header (`IHDR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ihdr {
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// Bits per sample (1, 2, 4, 8 or 16).
    bit_depth: u8,
    /// Color type (0, 2, 3, 4 or 6).
    color_type: u8,
    /// Interlace method (0 = none, 1 = Adam7).
    interlace: u8,
}

impl Ihdr {
    /// Samples per pixel for the color type (palette counts as one
    /// index sample).
    const fn channels(self) -> u32 {
        match self.color_type {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        }
    }

    /// Validates the color type / bit depth combination of the PNG
    /// specification.
    fn validate(self) -> TtResult<()> {
        let ok = match self.color_type {
            0 => matches!(self.bit_depth, 1 | 2 | 4 | 8 | 16),
            3 => matches!(self.bit_depth, 1 | 2 | 4 | 8),
            2 | 4 | 6 => matches!(self.bit_depth, 8 | 16),
            _ => false,
        };
        if !ok || self.interlace > 1 {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        if self.width == 0 || self.height == 0 {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        if self.width > MAX_DIMENSION || self.height > MAX_DIMENSION {
            return Err(TtError::ARRAY_TOO_LARGE);
        }
        Ok(())
    }
}

/// A PNG file split into its chunks (headers kept for expansion).
struct PngChunks {
    /// The validated header.
    ihdr: Ihdr,
    /// The `PLTE` palette (`3 * n` bytes), if present.
    palette: Option<Vec<u8>>,
    /// The raw `tRNS` payload, if present.
    transparency: Option<Vec<u8>>,
    /// The concatenated `IDAT` payload.
    idat: Vec<u8>,
}

impl PngChunks {
    /// Walks the chunk structure, verifying CRCs.
    fn parse(data: &[u8]) -> TtResult<Self> {
        if data.len() < SIGNATURE.len() || data.get(..8) != Some(SIGNATURE.as_slice()) {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        let mut pos = SIGNATURE.len();
        let mut ihdr = None;
        let mut palette = None;
        let mut transparency = None;
        let mut idat: Vec<u8> = Vec::new();
        let mut seen_idat = false;
        let mut seen_iend = false;

        while pos + 12 <= data.len() && !seen_iend {
            let length = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap_or([0; 4])) as usize;
            let kind_pos = pos + 4;
            let kind_end = kind_pos
                .checked_add(4)
                .ok_or(TtError::INVALID_TABLE)?;
            let payload_end = kind_end
                .checked_add(length)
                .ok_or(TtError::INVALID_TABLE)?;
            let crc_end = payload_end
                .checked_add(4)
                .ok_or(TtError::INVALID_TABLE)?;
            if crc_end > data.len() {
                return Err(TtError::INVALID_FILE_FORMAT);
            }
            let kind: [u8; 4] = data[kind_pos..kind_end]
                .try_into()
                .unwrap_or([0; 4]);
            let payload = &data[kind_end..payload_end];
            let crc = u32::from_be_bytes(
                data[payload_end..crc_end]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            if crc32(&data[kind_pos..payload_end]) != crc {
                return Err(TtError::INVALID_FILE_FORMAT);
            }
            if ihdr.is_none() && &kind != b"IHDR" {
                // The PNG specification requires IHDR first.
                return Err(TtError::INVALID_FILE_FORMAT);
            }

            match &kind {
                b"IHDR" => {
                    if ihdr.is_some() || length != 13 {
                        return Err(TtError::INVALID_FILE_FORMAT);
                    }
                    if payload[10] != 0 || payload[11] != 0 {
                        return Err(TtError::INVALID_FILE_FORMAT);
                    }
                    let header = Ihdr {
                        width: u32::from_be_bytes(payload[0..4].try_into().unwrap_or([0; 4])),
                        height: u32::from_be_bytes(payload[4..8].try_into().unwrap_or([0; 4])),
                        bit_depth: payload[8],
                        color_type: payload[9],
                        interlace: payload[12],
                    };
                    header.validate()?;
                    ihdr = Some(header);
                }
                b"PLTE" => {
                    if seen_idat
                        || palette.is_some()
                        || length == 0
                        || !length.is_multiple_of(3)
                        || length / 3 > 256
                    {
                        return Err(TtError::INVALID_FILE_FORMAT);
                    }
                    palette = Some(payload.to_vec());
                }
                b"tRNS" => {
                    if seen_idat || transparency.is_some() {
                        return Err(TtError::INVALID_FILE_FORMAT);
                    }
                    let color_type = ihdr
                        .ok_or(TtError::INVALID_FILE_FORMAT)?
                        .color_type;
                    let ok = match color_type {
                        0 => payload.len() == 2,
                        2 => payload.len() == 6,
                        3 => payload.len() <= palette.as_ref().map_or(0, Vec::len),
                        _ => false,
                    };
                    if !ok {
                        return Err(TtError::INVALID_FILE_FORMAT);
                    }
                    transparency = Some(payload.to_vec());
                }
                b"IDAT" => {
                    seen_idat = true;
                    let new_len = idat
                        .len()
                        .checked_add(length)
                        .ok_or(TtError::ARRAY_TOO_LARGE)?;
                    if new_len > MAX_IDAT {
                        return Err(TtError::ARRAY_TOO_LARGE);
                    }
                    idat.extend_from_slice(payload);
                }
                b"IEND" => {
                    seen_iend = true;
                }
                [first, ..] if first.is_ascii_uppercase() => {
                    // Unknown critical chunk: libpng refuses the file.
                    return Err(TtError::INVALID_FILE_FORMAT);
                }
                _ => {} // unknown ancillary chunk: skip
            }
            pos = crc_end;
        }

        let ihdr = ihdr
            .filter(|_| seen_idat && seen_iend)
            .ok_or(TtError::INVALID_FILE_FORMAT)?;
        if palette.is_none() && ihdr.color_type == 3 {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        Ok(PngChunks {
            ihdr,
            palette,
            transparency,
            idat,
        })
    }

    /// Inflates the image data and expands it to RGBA8 rows inside a
    /// [`PixelMode::Bgra`] bitmap (before premultiplication the bytes
    /// are plain RGBA; [`decode`] applies the pngshim transform).
    fn expand(&self) -> TtResult<Bitmap> {
        let raw_len = self.raw_len()?;
        let mut raw = vec![0u8; raw_len];
        let written = match codevar_truetype_gzip::uncompress(&mut raw, &self.idat) {
            Ok(n) => n,
            Err(TtError::OUT_OF_MEMORY) | Err(TtError::ARRAY_TOO_LARGE) => {
                return Err(TtError::OUT_OF_MEMORY);
            }
            Err(_) => return Err(TtError::INVALID_TABLE),
        };
        if written != raw_len {
            return Err(TtError::INVALID_TABLE);
        }

        let width = self.ihdr.width as usize;
        let height = self.ihdr.height as usize;
        let buffer_len = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .ok_or(TtError::ARRAY_TOO_LARGE)?;
        let mut buffer = vec![0u8; buffer_len];
        self.unfilter_and_expand(&raw, &mut buffer)?;

        Ok(Bitmap {
            rows: self.ihdr.height,
            width: self.ihdr.width,
            pitch: (self.ihdr.width * 4) as i32,
            buffer,
            num_grays: 256,
            pixel_mode: PixelMode::Bgra,
        })
    }

    /// The exact inflated size: sum over all passes of
    /// `filter byte + row bytes`.
    fn raw_len(&self) -> TtResult<usize> {
        let h = self.ihdr.height;
        let w = self.ihdr.width;
        let mut total = 0usize;
        if self.ihdr.interlace == 0 {
            total = add_rows(&mut total, h, w, self.ihdr)?;
        } else {
            for (x0, y0, dx, dy) in ADAM7 {
                let pw = pass_size(w, x0, dx);
                let ph = pass_size(h, y0, dy);
                if pw == 0 || ph == 0 {
                    continue;
                }
                add_rows(&mut total, ph, pw, self.ihdr)?;
            }
        }
        Ok(total)
    }

    /// Undoes the row filters and expands every pixel to RGBA8.
    fn unfilter_and_expand(&self, raw: &[u8], out: &mut [u8]) -> TtResult<()> {
        let ihdr = self.ihdr;
        let width = ihdr.width as usize;
        let height = ihdr.height as usize;
        let bytes_per_pixel = ((ihdr.channels() as usize) * (ihdr.bit_depth as usize))
            .div_ceil(8)
            .max(1);

        let mut pos = 0usize;
        if ihdr.interlace == 0 {
            self.walk_pass(raw, &mut pos, out, 0, 0, 1, 1, width, height, bytes_per_pixel)?;
        } else {
            for (x0, y0, dx, dy) in ADAM7 {
                let pw = pass_size(ihdr.width, x0, dx) as usize;
                let ph = pass_size(ihdr.height, y0, dy) as usize;
                if pw == 0 || ph == 0 {
                    continue;
                }
                self.walk_pass(
                    raw,
                    &mut pos,
                    out,
                    x0 as usize,
                    y0 as usize,
                    dx as usize,
                    dy as usize,
                    pw,
                    ph,
                    bytes_per_pixel,
                )?;
            }
        }
        if pos != raw.len() {
            return Err(TtError::INVALID_TABLE);
        }
        Ok(())
    }

    /// Unfilters and expands one (sub-)image into `out`.
    ///
    /// `emit` writes pixel `(px, py)`'s RGBA bytes; `x0/y0/dx/dy`
    /// map pass-local coordinates to image coordinates.
    #[allow(clippy::too_many_arguments)]
    fn walk_pass(
        &self,
        raw: &[u8],
        pos: &mut usize,
        out: &mut [u8],
        x0: usize,
        y0: usize,
        dx: usize,
        dy: usize,
        pass_width: usize,
        pass_height: usize,
        bpp: usize,
    ) -> TtResult<()> {
        let ihdr = self.ihdr;
        let row_bytes = (pass_width * (ihdr.channels() as usize) * (ihdr.bit_depth as usize)).div_ceil(8);
        let mut prev = vec![0u8; row_bytes];
        let mut cur = vec![0u8; row_bytes];

        for row in 0..pass_height {
            let filter = *raw.get(*pos).ok_or(TtError::INVALID_TABLE)?;
            *pos += 1;
            let start = (*pos)
                .checked_add(row_bytes)
                .ok_or(TtError::INVALID_TABLE)?;
            let src = raw
                .get(*pos..start)
                .ok_or(TtError::INVALID_TABLE)?;
            cur.copy_from_slice(src);
            *pos = start;
            unfilter_row(filter, &mut cur, &prev, bpp)?;

            for col in 0..pass_width {
                let px = x0 + col * dx;
                let py = y0 + row * dy;
                let out_index = py
                    .checked_mul(ihdr.width as usize)
                    .and_then(|n| n.checked_add(px))
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(TtError::ARRAY_TOO_LARGE)?..;
                let pixel = out
                    .get_mut(out_index.start..out_index.start + 4)
                    .ok_or(TtError::INVALID_TABLE)?;
                self.expand_pixel(&cur, col, pixel)?;
            }
            core::mem::swap(&mut prev, &mut cur);
        }
        Ok(())
    }

    /// Expands the pixel at column `col` of an unfiltered row into
    /// four RGBA bytes.
    fn expand_pixel(&self, row: &[u8], col: usize, out: &mut [u8]) -> TtResult<()> {
        let ihdr = self.ihdr;
        let channels = ihdr.channels() as usize;
        let bd = ihdr.bit_depth as usize;

        let sample = |channel: usize| -> TtResult<u32> {
            let index = col * channels + channel;
            read_sample(row, index, bd)
        };

        match ihdr.color_type {
            0 => {
                let value = sample(0)?;
                let transparent = self
                    .transparency
                    .as_ref()
                    .is_some_and(|t| t.len() >= 2 && u16::from_be_bytes([t[0], t[1]]) as u32 == value);
                let gray = scale_to_8(value, bd);
                out[0] = gray;
                out[1] = gray;
                out[2] = gray;
                out[3] = if transparent { 0 } else { 255 };
            }
            2 => {
                let red = sample(0)?;
                let green = sample(1)?;
                let blue = sample(2)?;
                let transparent = self.transparency.as_ref().is_some_and(|t| {
                    t.len() >= 6
                        && u16::from_be_bytes([t[0], t[1]]) as u32 == red
                        && u16::from_be_bytes([t[2], t[3]]) as u32 == green
                        && u16::from_be_bytes([t[4], t[5]]) as u32 == blue
                });
                out[0] = scale_to_8(red, bd);
                out[1] = scale_to_8(green, bd);
                out[2] = scale_to_8(blue, bd);
                out[3] = if transparent { 0 } else { 255 };
            }
            3 => {
                let index = sample(0)? as usize;
                let palette = self
                    .palette
                    .as_ref()
                    .ok_or(TtError::INVALID_FILE_FORMAT)?;
                let entry = palette
                    .get(index * 3..index * 3 + 3)
                    .ok_or(TtError::INVALID_TABLE)?;
                out[0] = entry[0];
                out[1] = entry[1];
                out[2] = entry[2];
                let alpha = self
                    .transparency
                    .as_ref()
                    .and_then(|t| t.get(index).copied())
                    .unwrap_or(255);
                out[3] = alpha;
            }
            4 => {
                let value = sample(0)?;
                let alpha = sample(1)?;
                let gray = scale_to_8(value, bd);
                out[0] = gray;
                out[1] = gray;
                out[2] = gray;
                out[3] = scale_to_8(alpha, bd);
            }
            _ => {
                out[0] = scale_to_8(sample(0)?, bd);
                out[1] = scale_to_8(sample(1)?, bd);
                out[2] = scale_to_8(sample(2)?, bd);
                out[3] = scale_to_8(sample(3)?, bd);
            }
        }
        Ok(())
    }
}

/// The Adam7 pass origins and steps `(x0, y0, dx, dy)`.
const ADAM7: [(u32, u32, u32, u32); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// Adds `height * (1 + row_bytes)` to `total` with overflow checks.
fn add_rows(total: &mut usize, height: u32, width: u32, ihdr: Ihdr) -> TtResult<usize> {
    let row_bytes = (width as usize)
        .checked_mul(ihdr.channels() as usize)
        .and_then(|n| n.checked_mul(ihdr.bit_depth as usize))
        .map(|n| n.div_ceil(8))
        .ok_or(TtError::ARRAY_TOO_LARGE)?;
    let per_row = row_bytes
        .checked_add(1)
        .ok_or(TtError::ARRAY_TOO_LARGE)?;
    let block = (height as usize)
        .checked_mul(per_row)
        .ok_or(TtError::ARRAY_TOO_LARGE)?;
    *total = total
        .checked_add(block)
        .ok_or(TtError::ARRAY_TOO_LARGE)?;
    Ok(*total)
}

/// The number of samples an Adam7 pass covers along one axis
/// (`ceil((size - start) / step)`).
fn pass_size(size: u32, start: u32, step: u32) -> u32 {
    if size <= start {
        0
    } else {
        (size - start).div_ceil(step)
    }
}

/// Reads sample `index` of `row` at `bit_depth` bits (MSB-first
/// packing).
fn read_sample(row: &[u8], index: usize, bit_depth: usize) -> TtResult<u32> {
    match bit_depth {
        16 => {
            let hi = index
                .checked_mul(2)
                .ok_or(TtError::INVALID_TABLE)?;
            let bytes = row
                .get(hi..hi + 2)
                .ok_or(TtError::INVALID_TABLE)?;
            Ok(u16::from_be_bytes(bytes.try_into().unwrap_or([0; 2])) as u32)
        }
        8 => Ok(u32::from(*row.get(index).ok_or(TtError::INVALID_TABLE)?)),
        1 | 2 | 4 => {
            let bit_pos = index
                .checked_mul(bit_depth)
                .ok_or(TtError::INVALID_TABLE)?;
            let byte = row
                .get(bit_pos / 8)
                .ok_or(TtError::INVALID_TABLE)?;
            let shift = 8 - bit_depth - (bit_pos % 8);
            let mask = (1u32 << bit_depth) - 1;
            Ok(((*byte as u32) >> shift) & mask)
        }
        _ => Err(TtError::INVALID_FILE_FORMAT),
    }
}

/// Scales a sample of `bit_depth` bits to 8 bits, like libpng's
/// `png_set_expand_gray_1_2_4_to_8` (`v * 255 / max`) and
/// `png_set_strip_16` (keep the high byte).
#[inline]
fn scale_to_8(value: u32, bit_depth: usize) -> u8 {
    match bit_depth {
        8 => value as u8,
        16 => (value >> 8) as u8,
        1..=7 => ((value * 255) / ((1u32 << bit_depth) - 1)) as u8,
        _ => 0,
    }
}

/// Reverses one PNG row filter in place (`prev` is the previous
/// unfiltered row).
fn unfilter_row(filter: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> TtResult<()> {
    match filter {
        0 => {}
        1 => {
            for i in bpp..cur.len() {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        2 => {
            for (c, p) in cur.iter_mut().zip(prev.iter()) {
                *c = c.wrapping_add(*p);
            }
        }
        3 => {
            for i in 0..cur.len() {
                let left = if i >= bpp { u32::from(cur[i - bpp]) } else { 0 };
                let up = u32::from(prev[i]);
                cur[i] = cur[i].wrapping_add(((left + up) / 2) as u8);
            }
        }
        4 => {
            for i in 0..cur.len() {
                let left = if i >= bpp { i32::from(cur[i - bpp]) } else { 0 };
                let up = i32::from(prev[i]);
                let up_left = if i >= bpp { i32::from(prev[i - bpp]) } else { 0 };
                let p = left + up - up_left;
                let pa = p.abs_diff(left);
                let pb = p.abs_diff(up);
                let pc = p.abs_diff(up_left);
                let pred = if pa <= pb && pa <= pc {
                    left
                } else if pb <= pc {
                    up
                } else {
                    up_left
                };
                cur[i] = cur[i].wrapping_add(pred as u8);
            }
        }
        _ => return Err(TtError::INVALID_TABLE),
    }
    Ok(())
}

/// The CRC-32 (IEEE 802.3) of `data`, as used by PNG chunk trailers.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{chunk, zlib_stored};
    use alloc::vec::Vec;

    /// Builds a non-interlaced PNG from raw RGBA8 rows (filter 0).
    fn png_rgba(width: u32, height: u32, rows: &[&[u8]]) -> Vec<u8> {
        let mut raw = Vec::new();
        for row in rows {
            raw.push(0);
            raw.extend_from_slice(row);
        }
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, no interlace
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &zlib_stored(&raw));
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn decodes_rgba_rows_premultiplied_to_bgra() {
        // Red opaque next to half-transparent white.
        let row = [255, 0, 0, 255, 255, 255, 255, 128];
        let data = png_rgba(2, 1, &[&row]);
        let bitmap = decode(&data).unwrap();

        assert_eq!(bitmap.pixel_mode, PixelMode::Bgra);
        assert_eq!((bitmap.width, bitmap.rows), (2, 1));
        // Opaque red: BGRA swap, no premultiply.
        assert_eq!(&bitmap.buffer[0..4], &[0, 0, 255, 255]);
        // Alpha 128 white: premultiplied to 128 in every channel.
        assert_eq!(&bitmap.buffer[4..8], &[128, 128, 128, 128]);
    }

    #[test]
    fn zero_alpha_pixel_is_zeroed() {
        let data = png_rgba(1, 1, &[&[10, 20, 30, 0]]);
        let bitmap = decode(&data).unwrap();
        assert_eq!(&bitmap.buffer[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn palette_images_expand_and_apply_transparency() {
        // 4x1 image, bit depth 2, palette [red, lime, blue, white],
        // tRNS = [255, 255, 64] (entry 3 defaults to opaque).
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&4u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[2, 3, 0, 0, 0]); // 2-bit palette
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(
            &mut out,
            b"PLTE",
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
        );
        chunk(&mut out, b"tRNS", &[255, 255, 64]);
        // One packed row: indices 0b00_01_10_11 = 0x1B, filter 0.
        chunk(&mut out, b"IDAT", &zlib_stored(&[0, 0x1B]));
        chunk(&mut out, b"IEND", &[]);
        let bitmap = decode(&out).unwrap();

        assert_eq!(bitmap.width, 4);
        // index 0: opaque red -> BGRA (255,0,0,255) wait: red pixel
        // BGRA = [blue=0, green=0, red=255, alpha=255].
        assert_eq!(&bitmap.buffer[0..4], &[0, 0, 255, 255]);
        // index 1: opaque lime.
        assert_eq!(&bitmap.buffer[4..8], &[0, 255, 0, 255]);
        // index 2: alpha 64 blue, premultiplied: 64*255/255 rounded.
        let a = multiply_alpha(64, 255) as u8;
        assert_eq!(&bitmap.buffer[8..12], &[a, 0, 0, 64]);
        // index 3: default opaque white.
        assert_eq!(&bitmap.buffer[12..16], &[255, 255, 255, 255]);
    }

    #[test]
    fn sub_byte_gray_scales_and_trns_matches_raw_samples() {
        // 2x1, bit depth 4, gray values 0 and 15, tRNS gray = 15.
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[4, 0, 0, 0, 0]); // 4-bit gray
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"tRNS", &15u16.to_be_bytes());
        chunk(&mut out, b"IDAT", &zlib_stored(&[0, 0x0F]));
        chunk(&mut out, b"IEND", &[]);
        let bitmap = decode(&out).unwrap();

        // sample 0 -> black opaque; sample 15 -> transparent white.
        assert_eq!(&bitmap.buffer[0..4], &[0, 0, 0, 255]);
        assert_eq!(&bitmap.buffer[4..8], &[0, 0, 0, 0]);
    }

    #[test]
    fn sub_up_average_and_paeth_filters_roundtrip() {
        // 4 rows using filters 1..4 over gray8 data.
        // Unfiltered rows are [10, 20], [30, 40], [50, 60], [70, 80].
        let unfiltered: [[u8; 2]; 4] = [[10, 20], [30, 40], [50, 60], [70, 80]];
        let mut raw = Vec::new();
        // Encoded rows, one filter per row:
        // row 0 Sub, row 1 Up, row 2 Average, row 3 Paeth.
        let encoded: [(u8, [u8; 2]); 4] = [
            (1, [10, 10]),
            (2, [20, 20]),
            (3, [35, 15]),
            // Paeth against [50, 60]: predictors 50 (up) then 70
            // (left), so the residues are 70 - 50 and 80 - 70.
            (4, [20, 10]),
        ];
        for (filter, values) in encoded {
            raw.push(filter);
            raw.extend_from_slice(&values);
        }
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&4u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &zlib_stored(&raw));
        chunk(&mut out, b"IEND", &[]);
        let bitmap = decode(&out).unwrap();

        for (row, values) in unfiltered.iter().enumerate() {
            for (col, &value) in values.iter().enumerate() {
                let index = (row * 2 + col) * 4;
                assert_eq!(bitmap.buffer[index], value, "row {row} col {col}");
                assert_eq!(bitmap.buffer[index + 3], 255);
            }
        }
    }

    #[test]
    fn filters_use_average_and_paeth_predictors_correctly() {
        // Directly exercise the two predictor helpers.
        let prev = [100u8, 200];
        let mut row = [40u8, 10]; // encoded values
        // Average: pixel0 = 40 + (0 + 100)/2 = 90; pixel1 = 10 +
        // (90 + 200)/2 = 155.
        unfilter_row(3, &mut row, &prev, 1).unwrap();
        assert_eq!(row, [90, 155]);

        let mut row = [40u8, 10];
        // Paeth pixel 0: left=0, up=100, up-left=0 → p=100, the
        // closest predictor is `up` (100) → 40 + 100 = 140.
        unfilter_row(4, &mut row, &prev, 1).unwrap();
        assert_eq!(row[0], 140);
        // Paeth pixel 1: left=140, up=200, up-left=100 → p=240, the
        // closest predictor is `up` (200) → 10 + 200 = 210.
        assert_eq!(row[1], 210);
    }

    #[test]
    fn adam7_interlaced_images_place_pixels_correctly() {
        // 3x3 RGBA image; encode it interlaced (all passes).
        let mut pixels = [[0u8; 4]; 9];
        for (i, pixel) in pixels.iter_mut().enumerate() {
            pixel[0] = u8::try_from(i + 1).unwrap_or(0);
            pixel[3] = 255;
        }
        let mut raw = Vec::new();
        for (x0, y0, dx, dy) in ADAM7 {
            let pw = pass_size(3, x0, dx);
            let ph = pass_size(3, y0, dy);
            if pw == 0 || ph == 0 {
                continue;
            }
            for row in 0..ph {
                raw.push(0); // filter
                for col in 0..pw {
                    let x = x0 + col * dx;
                    let y = y0 + row * dy;
                    raw.extend_from_slice(&pixels[(y * 3 + x) as usize]);
                }
            }
        }
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&3u32.to_be_bytes());
        ihdr.extend_from_slice(&3u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 1]); // RGBA, Adam7
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &zlib_stored(&raw));
        chunk(&mut out, b"IEND", &[]);
        let bitmap = decode(&out).unwrap();

        for (i, pixel) in pixels.iter().enumerate() {
            let expect = [pixel[2], pixel[1], pixel[0], pixel[3]]; // BGRA
            assert_eq!(&bitmap.buffer[i * 4..i * 4 + 4], &expect, "pixel {i}");
        }
    }

    #[test]
    fn structural_errors_are_reported_never_panicking() {
        // Bad signature.
        assert!(matches!(decode(b"not a png"), Err(TtError::INVALID_FILE_FORMAT)));
        // Empty input.
        assert!(matches!(decode(&[]), Err(TtError::INVALID_FILE_FORMAT)));

        // Truncated chunk.
        let mut data = Vec::from(SIGNATURE);
        data.extend_from_slice(&4u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        assert!(matches!(decode(&data), Err(TtError::INVALID_FILE_FORMAT)));

        // Corrupt CRC.
        let mut data = png_rgba(1, 1, &[&[0, 0, 0, 255]]);
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        assert!(matches!(decode(&data), Err(TtError::INVALID_FILE_FORMAT)));

        // Missing IDAT (IHDR + IEND only).
        let mut data = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(&mut data, b"IHDR", &ihdr);
        chunk(&mut data, b"IEND", &[]);
        assert!(matches!(decode(&data), Err(TtError::INVALID_FILE_FORMAT)));

        // Unsupported color type / bit depth.
        let mut data = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 1, 0, 0, 0]); // color type 1 invalid
        chunk(&mut data, b"IHDR", &ihdr);
        chunk(&mut data, b"IDAT", &zlib_stored(&[0, 0, 0, 0, 255]));
        chunk(&mut data, b"IEND", &[]);
        assert!(matches!(decode(&data), Err(TtError::INVALID_FILE_FORMAT)));

        // Zero-sized image.
        let mut data = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&0u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(&mut data, b"IHDR", &ihdr);
        assert!(matches!(decode(&data), Err(TtError::INVALID_FILE_FORMAT)));
    }

    #[test]
    fn corrupt_idat_reports_invalid_table() {
        // Valid structure but the IDAT payload is not a zlib stream.
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &[0xDE, 0xAD, 0xBE, 0xEF]);
        chunk(&mut out, b"IEND", &[]);
        assert!(matches!(decode(&out), Err(TtError::INVALID_TABLE)));
    }

    #[test]
    fn sixteen_bit_samples_take_the_high_byte() {
        // 1x1 RGB16: 0x8000/0x1234/0xABCD strip to 0x80/0x12/0xAB
        // and arrive as BGRA.
        let mut out = Vec::from(SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[16, 2, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(
            &mut out,
            b"IDAT",
            &zlib_stored(&[0, 0x80, 0x00, 0x12, 0x34, 0xAB, 0xCD]),
        );
        chunk(&mut out, b"IEND", &[]);
        let bitmap = decode(&out).unwrap();
        assert_eq!(&bitmap.buffer[0..4], &[0xAB, 0x12, 0x80, 255]);
    }
}
