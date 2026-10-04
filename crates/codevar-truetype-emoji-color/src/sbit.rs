//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License.  You may obtain a copy of the
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

//! `CBLC`/`CBDT` embedded color bitmap strikes (`ttsbit.c`).
//!
//! FreeType splits this across `tt_face_load_sbit` (strike directory),
//! `tt_sbit_decoder_init`/`tt_sbit_decoder_load_image` (index formats)
//! and `tt_sbit_decoder_load_png` + `Load_SBit_Png` (image data).  This
//! module folds the three into [`SbitStrikes`], which parses the
//! strike array once and decodes individual glyph images on demand:
//!
//! * index formats 1, 2, 3, 4 and 5 (plus FreeType's alias 19),
//! * image formats 17 (small metrics + PNG), 18 (big metrics + PNG)
//!   and 19 (metrics from the index + PNG).
//!
//! # Deviations from FreeType 2.13
//!
//! * Only `CBLC`/`CBDT` pairs are recognized; `EBLC`/`EBDT`/`bloc`/
//!   `bdat` (monochrome strikes) and `sbix` are out of scope, and
//!   image formats other than 17/18/19 report
//!   [`TtError::UNIMPLEMENTED_FEATURE`] instead of being rendered as
//!   gray bitmaps.
//! * [`SbitStrikes::best_strike`] only considers 32-bit strikes and
//!   picks the *nearest* ppem (ties resolve to the larger strike);
//!   FreeType matches strikes exactly against the requested size and
//!   otherwise falls back to the outline.
//! * A PNG whose dimensions disagree with the strike metrics is an
//!   error here ([`TtError::INVALID_FILE_FORMAT`]); FreeType keeps the
//!   zeroed buffer and reports success.

use alloc::vec::Vec;

use codevar_truetype_core::{Bitmap, TtError, TtResult};

use crate::cursor::{Reader, read_u16_at, read_u32_at};
use crate::png;

/// Size of one `BitmapSize` record in `CBLC`.
const BITMAP_SIZE: usize = 48;
/// Size of one entry of the index subtable array (start/end gid +
/// offset).
const RANGE_SIZE: usize = 8;
/// Size of an index subtable header (formats/offset).
const INDEX_HEADER: usize = 8;
/// The table header of `CBLC` (version + strike count).
const HEADER_SIZE: usize = 8;

/// `true` for the `CBLC` versions accepted by `tt_face_load_sbit`
/// (0x00020000 / 0x00030000, byte-swapped variants included).
const fn valid_version(version: u32) -> bool {
    (version & 0xFFFF_0000) == 0x0002_0000
        || (version & 0x0000_FFFF) == 0x0000_0200
        || (version & 0xFFFF_0000) == 0x0003_0000
        || (version & 0x0000_FFFF) == 0x0000_0300
}

/// One `BitmapSize` record of the `CBLC` strike directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strike {
    /// Offset of the index subtable array inside `CBLC`.
    pub index_array: usize,
    /// `numberOfIndexSubTables` (the number of gid ranges).
    pub num_subtables: u32,
    /// First glyph index of the strike.
    pub start_gid: u16,
    /// Last glyph index of the strike.
    pub end_gid: u16,
    /// Horizontal ppem of the strike.
    pub ppem_x: u8,
    /// Vertical ppem of the strike.
    pub ppem_y: u8,
    /// Bits per pixel (32 for color strikes).
    pub bit_depth: u8,
    /// `flags` (`bit 0`: horizontal metrics).
    pub flags: u8,
}

/// `TT_SBit_MetricsRec`: the small/big metrics block that describes a
/// strike image, in strike pixels.
///
/// The vertical fields are parsed for fidelity with FreeType but are
/// unused by this crate's horizontal layout path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SbitMetrics {
    /// Image height in pixels.
    pub height: u8,
    /// Image width in pixels.
    pub width: u8,
    /// Horizontal left side bearing.
    pub bearing_x: i8,
    /// Horizontal top side bearing.
    pub bearing_y: i8,
    /// Horizontal advance width.
    pub advance: u8,
    /// Vertical left side bearing (unused).
    pub vert_bearing_x: i8,
    /// Vertical top side bearing (unused).
    pub vert_bearing_y: i8,
    /// Vertical advance height (unused).
    pub vert_advance: u8,
}

/// A decoded strike image: premultiplied BGRA pixels plus the metrics
/// that described them.
#[derive(Clone, Debug)]
pub struct SbitImage {
    /// The decoded PNG data as a top-down BGRA bitmap.
    pub bitmap: Bitmap,
    /// The strike metrics for this glyph.
    pub metrics: SbitMetrics,
    /// Horizontal ppem of the source strike.
    pub ppem_x: u8,
    /// Vertical ppem of the source strike.
    pub ppem_y: u8,
}

/// The parsed `CBLC` strike directory plus its `CBDT` payload.
///
/// Parsing validates only the table header and the `BitmapSize`
/// records; per-strike index bounds are checked when a glyph is
/// loaded, mirroring FreeType's lazy `tt_sbit_decoder_init` checks.
#[derive(Clone, Debug)]
pub struct SbitStrikes<'a> {
    /// The whole `CBLC` table.
    cblc: &'a [u8],
    /// The whole `CBDT` table.
    cbdt: &'a [u8],
    /// The `BitmapSize` records, in table order.
    strikes: Vec<Strike>,
}

impl<'a> SbitStrikes<'a> {
    /// Parses the `CBLC`/`CBDT` pair of a font.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_FILE_FORMAT`] — the table is shorter than
    ///   its header, the version is unknown, or `numStikes` exceeds
    ///   0xFFFF.
    /// * [`TtError::INVALID_TABLE`] — a `BitmapSize` record is
    ///   truncated (only possible when the strike count was clamped).
    pub fn parse(cblc: &'a [u8], cbdt: &'a [u8]) -> TtResult<Self> {
        let mut reader = Reader::new(cblc);
        let version = reader
            .u32()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)?;
        if !valid_version(version) {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        let num_strikes = reader
            .u32()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)?;
        if num_strikes >= 0x1_0000 {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        // `tt_face_load_sbit` does not trust the declared count: the
        // number of records that physically fit wins.
        let mut count = num_strikes as usize;
        let fitting = cblc.len().saturating_sub(HEADER_SIZE) / BITMAP_SIZE;
        if count > fitting {
            count = fitting;
        }

        let mut strikes = Vec::with_capacity(count);
        for index in 0..count {
            let base = HEADER_SIZE + index * BITMAP_SIZE;
            let record = cblc
                .get(base..base + BITMAP_SIZE)
                .ok_or(TtError::INVALID_TABLE)?;
            strikes.push(Strike {
                index_array: read_u32_at(record, 0)? as usize,
                num_subtables: read_u32_at(record, 8)?,
                start_gid: read_u16_at(record, 40)?,
                end_gid: read_u16_at(record, 42)?,
                bit_depth: record[46],
                flags: record[47],
                ppem_x: record[44],
                ppem_y: record[45],
            });
        }

        Ok(SbitStrikes { cblc, cbdt, strikes })
    }

    /// The parsed strike records.
    #[inline]
    pub fn strikes(&self) -> &[Strike] {
        &self.strikes
    }

    /// The strike closest to the requested `ppem`, considering only
    /// 32-bit (color) strikes.
    ///
    /// Distance is `max(|ppemX - ppem|, |ppemY - ppem|)`; ties resolve
    /// to the strike with the larger ppem (larger `ppem_x`, then
    /// larger `ppem_y`), so downscaling is preferred over upscaling.
    ///
    /// Returns `None` when the font has no color strike.
    pub fn best_strike(&self, ppem: u16) -> Option<usize> {
        let target = u32::from(ppem);
        let mut best: Option<(u32, usize)> = None;
        for (index, strike) in self.strikes.iter().enumerate() {
            if strike.bit_depth != 32 {
                continue;
            }
            let dx = u32::from(strike.ppem_x).abs_diff(target);
            let dy = u32::from(strike.ppem_y).abs_diff(target);
            let distance = dx.max(dy);
            let better = match best {
                None => true,
                Some((best_distance, best_index)) => {
                    if distance != best_distance {
                        distance < best_distance
                    } else {
                        let other = &self.strikes[best_index];
                        (strike.ppem_x, strike.ppem_y) > (other.ppem_x, other.ppem_y)
                    }
                }
            };
            if better {
                best = Some((distance, index));
            }
        }
        best.map(|(_, index)| index)
    }

    /// Decodes the image of `gindex` from strike `strike_index`.
    ///
    /// Returns `Ok(None)` when the strike's index arrays contain no
    /// (or an empty) entry for `gindex` — FreeType's
    /// `Missing_Bitmap`, which callers treat as "fall back to the
    /// outline".
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_FILE_FORMAT`] — index offsets leave the
    ///   table, the PNG length is truncated, or the PNG size differs
    ///   from the strike metrics.
    /// * [`TtError::INVALID_TABLE`] — an image offset points outside
    ///   the `CBLC` table.
    /// * [`TtError::INVALID_ARGUMENT`] — `strike_index` does not
    ///   exist, the image range leaves `CBDT`, the metrics block is
    ///   truncated, the strike is not 32-bit, or the format 19
    ///   metrics are missing.
    /// * [`TtError::UNIMPLEMENTED_FEATURE`] — the image format is not
    ///   17/18/19.
    /// * [`TtError::INVALID_TABLE`] / [`TtError::OUT_OF_MEMORY`] —
    ///   forwarded from [`png::decode`] for corrupt PNG payloads.
    pub fn load(&self, strike_index: usize, gindex: u32) -> TtResult<Option<SbitImage>> {
        let strike = self
            .strikes
            .get(strike_index)
            .ok_or(TtError::INVALID_ARGUMENT)?;
        // `tt_sbit_decoder_init` validates the index array before the
        // range walk, for every glyph.
        let array_end = strike
            .index_array
            .checked_add(
                (strike.num_subtables as usize)
                    .checked_mul(RANGE_SIZE)
                    .ok_or(TtError::INVALID_FILE_FORMAT)?,
            )
            .ok_or(TtError::INVALID_FILE_FORMAT)?;
        if strike.index_array > self.cblc.len() || array_end > self.cblc.len() {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        if gindex > u32::from(u16::MAX) {
            return Ok(None);
        }
        let gid = gindex as u16;

        // Find the gid range that holds `gindex` (`load_image`'s
        // range walk).  No matching range — including one outside the
        // strike's declared `[start_gid, end_gid]` window — is
        // FreeType's `Missing_Bitmap`; the index formats below
        // resolve offsets relative to the *range's* first glyph, not
        // the strike's.
        let mut covering = None;
        for range in 0..strike.num_subtables as usize {
            let base = strike.index_array + range * RANGE_SIZE;
            let start = read_u16_at(self.cblc, base)?;
            let end = read_u16_at(self.cblc, base + 2)?;
            if gid >= start && gid <= end {
                covering = Some((start, read_u32_at(self.cblc, base + 4)?));
                break;
            }
        }
        let Some((range_start, image_offset)) = covering else {
            return Ok(None);
        };
        // FreeType reports `Invalid_Table` when the range offset runs
        // past the index array.
        let image_offset = usize::try_from(image_offset).map_err(|_| TtError::INVALID_TABLE)?;
        if image_offset > self.cblc.len() - strike.index_array {
            return Err(TtError::INVALID_TABLE);
        }

        let header = strike.index_array + image_offset;
        // A truncated index subtable header (`p + 8 > p_limit` in
        // FreeType) is `Missing_Bitmap`.
        if header
            .checked_add(INDEX_HEADER)
            .is_none_or(|end| end > self.cblc.len())
        {
            return Ok(None);
        }
        let mut reader = Reader::at(self.cblc, header).ok_or(TtError::INVALID_FILE_FORMAT)?;
        let index_format = reader
            .u16()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)?;
        let image_format = reader
            .u16()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)?;
        let image_data_offset = reader
            .u32()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)? as usize;
        let entries = header + INDEX_HEADER;

        // Resolve `(image_start, image_end)` inside `CBDT`, with the
        // index formats of `tt_sbit_decoder_load_image`.  Every
        // truncated read here is FreeType's `Missing_Bitmap`.
        let (start, end, preloaded) = match index_format {
            1 => {
                let base = entries
                    .checked_add(4 * usize::from(gid - range_start))
                    .ok_or(TtError::INVALID_TABLE)?;
                let (Some(start), Some(end)) = (
                    read_u32_at(self.cblc, base).ok(),
                    read_u32_at(self.cblc, base + 4).ok(),
                ) else {
                    return Ok(None);
                };
                if start == end {
                    return Ok(None);
                }
                (u64::from(start), u64::from(end), None)
            }
            2 => {
                // `p + 12 > p_limit`: image size + big metrics.
                if entries
                    .checked_add(12)
                    .is_none_or(|limit| limit > self.cblc.len())
                {
                    return Ok(None);
                }
                let size = read_u32_at(self.cblc, entries).map_err(|_| TtError::INVALID_TABLE)?;
                let Ok(metrics) = read_metrics(self.cblc, entries + 4, true) else {
                    return Ok(None);
                };
                let start = u64::from(size) * u64::from(gid - range_start);
                (start, start + u64::from(size), Some(metrics))
            }
            3 => {
                let base = entries
                    .checked_add(2 * usize::from(gid - range_start))
                    .ok_or(TtError::INVALID_TABLE)?;
                let (Some(start), Some(end)) = (
                    read_u16_at(self.cblc, base).ok(),
                    read_u16_at(self.cblc, base + 2).ok(),
                ) else {
                    return Ok(None);
                };
                if start == end {
                    return Ok(None);
                }
                (u64::from(start), u64::from(end), None)
            }
            4 => {
                let Some(num_glyphs) = read_u32_at(self.cblc, entries).ok() else {
                    return Ok(None);
                };
                let pairs = entries + 4;
                // `(numGlyphs + 1) * 4` bytes of `(gid, offset)`
                // pairs must fit after the count: the end of the
                // last glyph's image lives in the sentinel pair.
                let available = self.cblc.len().saturating_sub(pairs) / 4;
                if u64::from(num_glyphs) + 1 > available as u64 {
                    return Ok(None);
                }
                let mut found = None;
                for slot in 0..num_glyphs as usize {
                    let base = pairs + slot * 4;
                    if read_u16_at(self.cblc, base).ok() == Some(gid) {
                        let (Some(start), Some(end)) = (
                            read_u16_at(self.cblc, base + 2).ok(),
                            read_u16_at(self.cblc, base + 6).ok(),
                        ) else {
                            return Ok(None);
                        };
                        found = Some((u64::from(start), u64::from(end)));
                        break;
                    }
                }
                let Some((start, end)) = found else {
                    return Ok(None);
                };
                if start == end {
                    return Ok(None);
                }
                (start, end, None)
            }
            5 | 19 => {
                // `p + 16 > p_limit`: image size + big metrics +
                // glyph count.
                if entries
                    .checked_add(16)
                    .is_none_or(|limit| limit > self.cblc.len())
                {
                    return Ok(None);
                }
                let size = read_u32_at(self.cblc, entries).map_err(|_| TtError::INVALID_TABLE)?;
                let Ok(metrics) = read_metrics(self.cblc, entries + 4, true) else {
                    return Ok(None);
                };
                let num_glyphs_at = entries + 12;
                let num_glyphs = read_u32_at(self.cblc, num_glyphs_at).map_err(|_| TtError::INVALID_TABLE)?;
                let list = num_glyphs_at + 4;
                if u64::from(num_glyphs) > self.cblc.len().saturating_sub(list) as u64 / 2 {
                    return Ok(None);
                }
                let mut position = None;
                for slot in 0..num_glyphs as usize {
                    if read_u16_at(self.cblc, list + slot * 2).ok() == Some(gid) {
                        position = Some(slot);
                        break;
                    }
                }
                let Some(slot) = position else {
                    return Ok(None);
                };
                let start = u64::from(size) * slot as u64;
                (start, start + u64::from(size), Some(metrics))
            }
            _ => return Ok(None),
        };

        if start > end {
            return Ok(None);
        }
        // `tt_sbit_decoder_load_bitmap`: `!glyph_size || glyph_start
        // + glyph_size > ebdt_size` is `Invalid_Argument`.
        let size = end - start;
        let begin = (image_data_offset as u64)
            .checked_add(start)
            .ok_or(TtError::INVALID_ARGUMENT)?;
        let begin = usize::try_from(begin).map_err(|_| TtError::INVALID_ARGUMENT)?;
        let size = usize::try_from(size).map_err(|_| TtError::INVALID_ARGUMENT)?;
        if size == 0 || begin > self.cbdt.len() || size > self.cbdt.len() - begin {
            return Err(TtError::INVALID_ARGUMENT);
        }
        let data = self
            .cbdt
            .get(begin..begin + size)
            .ok_or(TtError::INVALID_ARGUMENT)?;

        self.load_image(data, image_format, preloaded, strike.bit_depth, strike)
            .map(Some)
    }

    /// Reads the metrics, PNG length and payload of one `CBDT` image
    /// record (`tt_sbit_decoder_load_bitmap` + `load_png`).
    ///
    /// # Errors
    ///
    /// * [`TtError::UNIMPLEMENTED_FEATURE`] — the image format is not
    ///   17/18/19.
    /// * [`TtError::INVALID_ARGUMENT`] — the metrics block does not
    ///   fit, the format 19 metrics are missing, or the strike is not
    ///   32-bit.
    /// * [`TtError::INVALID_FILE_FORMAT`] — the PNG length is
    ///   truncated or the decoded PNG size differs from the metrics.
    fn load_image(
        &self,
        data: &'a [u8],
        image_format: u16,
        preloaded: Option<SbitMetrics>,
        bit_depth: u8,
        strike: &Strike,
    ) -> TtResult<SbitImage> {
        let metrics = match image_format {
            17 => read_metrics(data, 0, false)?,
            18 => read_metrics(data, 0, true)?,
            19 => preloaded.ok_or(TtError::INVALID_ARGUMENT)?,
            // FreeType renders formats 1/2/6/7/8/9 as gray bitmaps;
            // this crate is color-only (see the module docs).
            _ => return Err(TtError::UNIMPLEMENTED_FEATURE),
        };
        // `Load_SBit_Png`'s `pix_bits != 32` check.
        if bit_depth != 32 {
            return Err(TtError::INVALID_ARGUMENT);
        }
        let png_start = match image_format {
            17 => 5,
            18 => 8,
            _ => 0,
        };
        let mut reader = Reader::at(data, png_start).ok_or(TtError::INVALID_FILE_FORMAT)?;
        let png_len = reader
            .u32()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)? as usize;
        let payload = reader
            .take(png_len)
            .map_err(|_| TtError::INVALID_FILE_FORMAT)?;
        let bitmap = png::decode(payload)?;
        // FreeType leaves a silently zeroed buffer on a size
        // mismatch; this crate reports the corruption instead.
        if bitmap.width != u32::from(metrics.width) || bitmap.rows != u32::from(metrics.height) {
            return Err(TtError::INVALID_FILE_FORMAT);
        }
        Ok(SbitImage {
            bitmap,
            metrics,
            ppem_x: strike.ppem_x,
            ppem_y: strike.ppem_y,
        })
    }
}

/// `tt_sbit_decoder_load_metrics`: reads 5 (small) or 8 (big) metric
/// bytes at `offset`.
///
/// # Errors
///
/// * [`TtError::INVALID_ARGUMENT`] — the block does not fit.
fn read_metrics(data: &[u8], offset: usize, big: bool) -> TtResult<SbitMetrics> {
    let total = if big { 8 } else { 5 };
    let mut reader = Reader::at(data, offset).ok_or(TtError::INVALID_ARGUMENT)?;
    let block = reader
        .take(total)
        .map_err(|_| TtError::INVALID_ARGUMENT)?;
    let mut metrics = SbitMetrics {
        height: block[0],
        width: block[1],
        bearing_x: block[2] as i8,
        bearing_y: block[3] as i8,
        advance: block[4],
        ..SbitMetrics::default()
    };
    if big {
        metrics.vert_bearing_x = block[5] as i8;
        metrics.vert_bearing_y = block[6] as i8;
        metrics.vert_advance = block[7];
    }
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{SbitImageSpec, cblc_cbdt17, cblc_cbdt19, png_rgba, push_u16, push_u32};
    use codevar_truetype_core::PixelMode;

    /// A 2x1 opaque-red PNG, matching the metrics below.
    fn red_png() -> Vec<u8> {
        let row = [255, 0, 0, 255, 255, 0, 0, 255];
        png_rgba(2, 1, &[&row])
    }

    fn spec(gid: u16) -> SbitImageSpec {
        SbitImageSpec {
            gid,
            height: 1,
            width: 2,
            bearing_x: 1,
            bearing_y: 2,
            advance: 3,
            png: red_png(),
        }
    }

    #[test]
    fn parses_strike_directory_and_loads_png_image() {
        let (cblc, cbdt) = cblc_cbdt17(16, 32, &[spec(1), spec(2)]);
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        assert_eq!(strikes.strikes().len(), 1);
        assert_eq!(strikes.best_strike(16), Some(0));

        let image = strikes
            .load(0, 1)
            .unwrap()
            .expect("image present");
        assert_eq!((image.bitmap.width, image.bitmap.rows), (2, 1));
        assert_eq!(image.bitmap.pixel_mode, PixelMode::Bgra);
        // Opaque red premultiplied to BGRA.
        assert_eq!(&image.bitmap.buffer[0..4], &[0, 0, 255, 255]);
        assert_eq!(image.metrics.bearing_x, 1);
        assert_eq!(image.metrics.bearing_y, 2);
        assert_eq!(image.metrics.advance, 3);
        assert_eq!((image.ppem_x, image.ppem_y), (16, 16));
    }

    #[test]
    fn missing_glyphs_report_none() {
        let (cblc, cbdt) = cblc_cbdt17(16, 32, &[spec(4), spec(5)]);
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        // Before the range, after the range and above u16.
        assert!(strikes.load(0, 3).unwrap().is_none());
        assert!(strikes.load(0, 6).unwrap().is_none());
        assert!(strikes.load(0, 70_000).unwrap().is_none());
        assert!(strikes.load(1, 4).is_err());
    }

    #[test]
    fn format19_takes_metrics_from_the_index() {
        let (cblc, cbdt) = cblc_cbdt19(24, 32, &[spec(2), spec(3)]);
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        let image = strikes
            .load(0, 3)
            .unwrap()
            .expect("image present");
        assert_eq!((image.bitmap.width, image.bitmap.rows), (2, 1));
        assert_eq!(image.metrics.advance, 3);
        // No metrics block in CBDT: data starts at the PNG length.
        assert_eq!(&image.bitmap.buffer[0..4], &[0, 0, 255, 255]);
    }

    #[test]
    fn best_strike_skips_gray_and_prefers_nearer_then_larger() {
        // Two strikes at 16 and 32 ppem plus a gray strike that must
        // never be selected.
        let mut cblc = Vec::new();
        push_u32(&mut cblc, 0x0003_0000);
        push_u32(&mut cblc, 3);
        for (index, (ppem, depth)) in [(16u8, 32u8), (32, 32), (16, 8)]
            .iter()
            .enumerate()
        {
            push_u32(&mut cblc, 56 + 48 * index as u32);
            push_u32(&mut cblc, 0);
            push_u32(&mut cblc, 8);
            push_u32(&mut cblc, 0);
            cblc.extend_from_slice(&[0u8; 24]);
            push_u16(&mut cblc, 0);
            push_u16(&mut cblc, 9);
            cblc.extend_from_slice(&[*ppem, *ppem, *depth, 1]);
        }
        let cbdt = 0u32.to_be_bytes().to_vec();
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        assert_eq!(strikes.best_strike(10), Some(0));
        assert_eq!(strikes.best_strike(20), Some(0));
        assert_eq!(strikes.best_strike(30), Some(1));
        // Equidistant: the larger strike wins.
        assert_eq!(strikes.best_strike(24), Some(1));
        // Only gray strikes would give None; here both color strikes
        // exist, so the gray one is never reached.
        assert_ne!(strikes.best_strike(16), Some(2));
    }

    #[test]
    fn corrupt_structures_report_errors() {
        // Unknown version.
        let mut cblc = 0u32.to_be_bytes().to_vec();
        cblc.extend_from_slice(&1u32.to_be_bytes());
        assert!(matches!(
            SbitStrikes::parse(&cblc, &[]),
            Err(TtError::INVALID_FILE_FORMAT)
        ));

        // Declared strike count beyond 0xFFFF.
        let mut cblc = Vec::new();
        push_u32(&mut cblc, 0x0002_0000);
        push_u32(&mut cblc, 0x1_0000);
        assert!(matches!(
            SbitStrikes::parse(&cblc, &[]),
            Err(TtError::INVALID_FILE_FORMAT)
        ));

        // Strike whose index array points outside the table.
        let (mut cblc, cbdt) = cblc_cbdt17(16, 32, &[spec(1)]);
        let length = cblc.len() as u32;
        cblc[8..12].copy_from_slice(&length.to_be_bytes());
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        assert!(matches!(strikes.load(0, 1), Err(TtError::INVALID_FILE_FORMAT)));
    }

    #[test]
    fn png_dimensions_must_match_the_metrics() {
        let mut bad = spec(1);
        bad.png = png_rgba(1, 1, &[&[255, 0, 0, 255]]);
        let (cblc, cbdt) = cblc_cbdt17(16, 32, &[bad]);
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        assert!(matches!(strikes.load(0, 1), Err(TtError::INVALID_FILE_FORMAT)));
    }

    #[test]
    fn unsupported_image_formats_report_unimplemented() {
        let (mut cblc, cbdt) = cblc_cbdt17(16, 32, &[spec(1)]);
        // Index subtable header: index format 1 @64, image format 17 @66.
        cblc[66..68].copy_from_slice(&1u16.to_be_bytes());
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        assert!(matches!(strikes.load(0, 1), Err(TtError::UNIMPLEMENTED_FEATURE)));
    }

    #[test]
    fn gray_strikes_are_rejected_on_load() {
        let (cblc, cbdt) = cblc_cbdt17(16, 8, &[spec(1)]);
        let strikes = SbitStrikes::parse(&cblc, &cbdt).unwrap();
        assert_eq!(strikes.best_strike(16), None);
        assert!(matches!(strikes.load(0, 1), Err(TtError::INVALID_ARGUMENT)));
    }
}
