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
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The glyph rendering pipeline: turns a glyph index plus size into a
//! premultiplied bitmap, in the order FreeType 2.13.3 explores.
//!
//! This module ports the pixel-producing half of FreeType:
//!
//! | FreeType 2.13 | this module |
//! |---|---|
//! | `FT_Render_Glyph_Internal` layer loop (`ftobjs.c`) | [`ColorFont::render_glyph`] + [`ColorFont::render_layers`] |
//! | `tt_face_colr_blend_layer` (`ttcolr.c`) | [`Canvas`] |
//! | `ft_glyphslot_preset_bitmap` + `ft_smooth_render` (`ftsmooth.c`) | [`render_outline`] |
//! | `TT_Load_Glyph` sbit metrics scaling (`ttgload.c`) | [`ColorFont`]'s strike path |
//!
//! [`ColorFont::render_glyph`] resolves a glyph in three steps,
//! mirroring `FT_Load_Glyph(FT_LOAD_COLOR)` followed by
//! `FT_Render_Glyph`:
//!
//! 1. the nearest `CBLC`/`CBDT` strike ([`SbitStrikes`]),
//! 2. the `COLR` v0 layer list blended with `CPAL` colors,
//! 3. the plain `glyf` outline through the smooth raster.
//!
//! Every step falls through to the next one when it cannot produce a
//! bitmap, so the pipeline always renders *something*.  The
//! crate-level docs list the deviations from FreeType.
//!
//! [`SbitStrikes`]: crate::sbit::SbitStrikes

use alloc::sync::Arc;
use alloc::vec::Vec;

use codevar_logger::log_warn;
use codevar_truetype_core::{
    BBox, Bitmap, Generic, GlyphFormat, GlyphLoader, GlyphMetrics, GlyphSlot, Library, LoadFlags, Matrix,
    Outline, PixelMode, Pos, Raster, RasterFlags, RasterParams, SizeMetrics, SlotInternal, TtError, TtResult,
    Vector, div_fix, mul_fix,
};
use codevar_truetype_glyf::load_glyph;
use codevar_truetype_sfnt::{
    SfntFont,
    tags::{TAG_CBDT, TAG_CBLC, TAG_COLR, TAG_CPAL},
};
use codevar_truetype_smooth::grays::GrayRaster;

use crate::colr::{COLOR_FOREGROUND, ColrTable, LayerIter};
use crate::cpal::CpalTable;
use crate::sbit::{SbitImage, SbitStrikes};

/// The largest bitmap edge FreeType's smooth renderer accepts
/// (`ft_glyphslot_preset_bitmap`'s `0x7FFF` range check).
const MAX_DIMENSION: i64 = 0x7FFF;

/// A rendered glyph: top-down pixels plus placement and advance.
///
/// Colors are premultiplied BGRA, matching
/// [`PixelMode::Bgra`]; plain outlines are [`PixelMode::Gray`].
#[derive(Clone, Debug)]
pub struct RenderedGlyph {
    /// The pixel buffer with its geometry.
    pub bitmap: Bitmap,
    /// Left side bearing of the bitmap, in pixels (may be negative).
    pub bitmap_left: i32,
    /// Distance from the baseline to the bitmap's top row, in pixels.
    pub bitmap_top: i32,
    /// Horizontal advance in 26.6 pixels (sixty-fourths of a pixel).
    pub advance: Pos,
    /// The ppem the glyph was rendered at.
    pub source_ppem: u16,
}

/// A color font: the (optional) `CPAL`, `COLR` and `CBLC`+`CBDT`
/// tables of one [`SfntFont`], plus the pipeline that composites
/// them.
///
/// [`ColorFont::open`] is deliberately lenient: a table that fails to
/// parse is logged and dropped, so a broken color table degrades to
/// the plain outline instead of failing the whole font.
#[derive(Debug)]
pub struct ColorFont<'a> {
    /// The face the color tables belong to.
    font: SfntFont<'a>,
    /// The parsed `CPAL` palette, when valid.
    cpal: Option<CpalTable<'a>>,
    /// The parsed `COLR` layers, when a valid `CPAL` accompanies them.
    colr: Option<ColrTable<'a>>,
    /// The parsed `CBLC`/`CBDT` strikes, when both tables are valid.
    strikes: Option<SbitStrikes<'a>>,
}

impl<'a> ColorFont<'a> {
    /// Takes `font` and keeps every color table that parses.
    ///
    /// A malformed `CPAL` is dropped (which also drops `COLR`, since
    /// layer colors live in `CPAL`); `CBLC` without `CBDT` is
    /// dropped; each case logs a warning instead of failing.
    #[must_use]
    pub fn open(font: SfntFont<'a>) -> Self {
        let cpal = match font.table(TAG_CPAL) {
            Some(data) => match CpalTable::parse(data) {
                Ok(parsed) => Some(parsed),
                Err(error) => {
                    log_warn!("ColorFont: ignoring a malformed CPAL table: {error}");
                    None
                }
            },
            None => None,
        };
        let colr = match font.table(TAG_COLR) {
            Some(data) if cpal.is_some() => match ColrTable::parse(data) {
                Ok(parsed) => Some(parsed),
                Err(error) => {
                    log_warn!("ColorFont: ignoring a malformed COLR table: {error}");
                    None
                }
            },
            Some(_) => {
                log_warn!("ColorFont: COLR needs a valid CPAL table; using the outline instead");
                None
            }
            None => None,
        };
        let strikes = match (font.table(TAG_CBLC), font.table(TAG_CBDT)) {
            (Some(cblc), Some(cbdt)) => match SbitStrikes::parse(cblc, cbdt) {
                Ok(parsed) => Some(parsed),
                Err(error) => {
                    log_warn!("ColorFont: ignoring a malformed CBLC/CBDT pair: {error}");
                    None
                }
            },
            (Some(_), None) => {
                log_warn!("ColorFont: the font has CBLC but no CBDT; ignoring the strikes");
                None
            }
            _ => None,
        };
        ColorFont {
            font,
            cpal,
            colr,
            strikes,
        }
    }

    /// Whether the font carries any usable color table.
    #[inline]
    #[must_use]
    pub const fn has_color(&self) -> bool {
        self.colr.is_some() || self.strikes.is_some()
    }

    /// The number of `CPAL` palettes (0 without a valid `CPAL`).
    #[inline]
    #[must_use]
    pub fn num_palettes(&self) -> u16 {
        self.cpal
            .as_ref()
            .map_or(0, |cpal| cpal.num_palettes())
    }

    /// Renders the glyph `character` maps to via the font's `cmap`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_GLYPH_INDEX`] — the font has no `cmap`
    ///   entry for `character`.
    /// * Whatever [`ColorFont::render_glyph`] reports.
    pub fn render_char(&self, character: char, ppem: u16, palette: u16) -> TtResult<RenderedGlyph> {
        let gindex = self.font.char_index(u32::from(character));
        if gindex == 0 {
            return Err(TtError::INVALID_GLYPH_INDEX);
        }
        self.render_glyph(gindex, ppem, palette)
    }

    /// Renders `gindex` at `ppem` pixels per em with `palette`'s
    /// colors, trying in order: the nearest `CBLC`/`CBDT` strike, the
    /// `COLR` layer list, the plain outline.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_ARGUMENT`] — `ppem` is zero, `gindex` is
    ///   outside the face, or `palette` does not exist.
    /// * [`TtError::INVALID_FILE_FORMAT`] / [`TtError::INVALID_TABLE`] /
    ///   [`TtError::UNIMPLEMENTED_FEATURE`] — corrupt glyph, strike or
    ///   PNG data (forwarded from the loaders).
    /// * [`TtError::RASTER_OVERFLOW`] / [`TtError::ARRAY_TOO_LARGE`] —
    ///   the outline's box does not fit a bitmap.
    pub fn render_glyph(&self, gindex: u32, ppem: u16, palette: u16) -> TtResult<RenderedGlyph> {
        if ppem == 0 || gindex >= u32::from(self.font.num_glyphs()) {
            return Err(TtError::INVALID_ARGUMENT);
        }
        if let Some(cpal) = &self.cpal
            && palette >= cpal.num_palettes()
        {
            return Err(TtError::INVALID_ARGUMENT);
        }
        let size = scaled_size(self.font.units_per_em(), ppem);

        // 1. `TT_Load_Glyph` tries the embedded strike first; a
        //    missing glyph falls through silently, a broken one warns.
        if let Some(strikes) = &self.strikes
            && let Some(index) = strikes.best_strike(ppem)
        {
            match strikes.load(index, gindex) {
                Ok(Some(image)) => return self.render_strike(image, &size, gindex),
                Ok(None) => {}
                Err(error) => log_warn!(
                    "ColorFont: reading the sbit strike failed for glyph {gindex}; \
                     falling back to the outline: {error}"
                ),
            }
        }

        // 2. Load the base glyph: its advance belongs to the result,
        //    and a failed load is fatal (FreeType reports it too).
        let mut slot = new_slot();
        load_glyph(&self.font, &mut slot, &size, gindex, LoadFlags::DEFAULT)?;
        let advance = slot.metrics.hori_advance;

        // 3. `FT_Render_Glyph_Internal`'s layer loop.
        if let Some(colr) = &self.colr
            && let Some(cpal) = &self.cpal
            && let Some(mut layers) = colr.layers(gindex, self.font.num_glyphs(), cpal.num_palette_entries())
            && let Some(first) = layers.next()
        {
            match self.render_layers(first, layers, cpal, palette, &size, &mut slot) {
                Ok((bitmap, bitmap_left, bitmap_top)) => {
                    return Ok(RenderedGlyph {
                        bitmap,
                        bitmap_left,
                        bitmap_top,
                        advance,
                        source_ppem: ppem,
                    });
                }
                Err(error) => log_warn!(
                    "ColorFont: compositing the COLR layers failed for glyph {gindex}; \
                     falling back to the outline: {error}"
                ),
            }
        }

        // 4. The plain gray outline.
        let (bitmap, bitmap_left, bitmap_top) = render_outline(&mut slot)?;
        Ok(RenderedGlyph {
            bitmap,
            bitmap_left,
            bitmap_top,
            advance,
            source_ppem: ppem,
        })
    }

    /// Renders every `COLR` layer of one base glyph into a single
    /// BGRA canvas (`tt_face_colr_blend_layer`'s caller loop).
    ///
    /// Each layer loads like FreeType's `FT_LOAD_RENDER` inner call:
    /// any failure aborts the whole composite and lets
    /// [`ColorFont::render_glyph`] fall back to the outline.
    ///
    /// # Errors
    ///
    /// Forwarded from the glyph loader, [`render_outline`] or the
    /// `CPAL` color lookup.
    fn render_layers(
        &self,
        first: (u16, u16),
        rest: LayerIter<'_>,
        cpal: &CpalTable<'_>,
        palette: u16,
        size: &SizeMetrics,
        slot: &mut GlyphSlot,
    ) -> TtResult<(Bitmap, i32, i32)> {
        let mut canvas = Canvas::empty();
        for (gindex, entry) in core::iter::once(first).chain(rest) {
            load_glyph(&self.font, slot, size, u32::from(gindex), LoadFlags::DEFAULT)?;
            let (bitmap, left, top) = render_outline(slot)?;
            let color = resolve_color(cpal, palette, entry)?;
            canvas.resize(left, top, bitmap.width, bitmap.rows)?;
            canvas.blend(&bitmap, left, top, color)?;
        }
        let Canvas { bitmap, left, top } = canvas;
        Ok((bitmap, left, top))
    }

    /// Converts a decoded strike image to the requested size
    /// (`TT_Load_Glyph`'s sbit metrics scaling).
    ///
    /// # Errors
    ///
    /// * [`TtError::ARRAY_TOO_LARGE`] — a scaled metric or dimension
    ///   overflows the bitmap limits.
    /// * [`TtError::INVALID_FILE_FORMAT`] — the strike image is not
    ///   BGRA (impossible for `png::decode` output).
    fn render_strike(&self, image: SbitImage, size: &SizeMetrics, gindex: u32) -> TtResult<RenderedGlyph> {
        let ppem = size.x_ppem;
        let width = scale_dim(image.bitmap.width, ppem, u32::from(image.ppem_x))?;
        let rows = scale_dim(image.bitmap.rows, ppem, u32::from(image.ppem_y))?;
        let bitmap = if width == image.bitmap.width && rows == image.bitmap.rows {
            image.bitmap
        } else {
            resample_bgra(&image.bitmap, width, rows)?
        };
        let bitmap_left = i32::try_from(scale_round(
            i64::from(image.metrics.bearing_x),
            ppem,
            u32::from(image.ppem_x),
        ))
        .map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let bitmap_top = i32::try_from(scale_round(
            i64::from(image.metrics.bearing_y),
            ppem,
            u32::from(image.ppem_y),
        ))
        .map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let scaled_advance = scale_round(
            i64::from(image.metrics.advance) * 64,
            ppem,
            u32::from(image.ppem_x),
        );
        // FreeType's sbit sanity fallback: a strike advance that
        // scales to zero falls back to the `hmtx` value.
        let advance = if scaled_advance == 0 {
            mul_fix(i64::from(self.font.advance_for_glyph(gindex)), size.x_scale)
        } else {
            scaled_advance
        };
        Ok(RenderedGlyph {
            bitmap,
            bitmap_left,
            bitmap_top,
            advance,
            source_ppem: ppem,
        })
    }
}

/// The composited BGRA canvas of the layer loop
/// (`tt_face_colr_blend_layer`'s destination bitmap).
///
/// The canvas grows by union with each layer's box, exactly like the
/// C code, and re-initializes whenever its buffer is still empty —
/// which is how FreeType treats a zero-sized first layer.
struct Canvas {
    /// The premultiplied BGRA pixels (empty until the first layer).
    bitmap: Bitmap,
    /// Left edge of the canvas box, in pixels.
    left: i32,
    /// Top edge of the canvas box, in pixels.
    top: i32,
}

impl Canvas {
    /// A canvas with no pixels: the next [`Canvas::resize`]
    /// initializes it from its argument.
    fn empty() -> Self {
        Canvas {
            bitmap: Bitmap::new(),
            left: 0,
            top: 0,
        }
    }

    /// Grows the canvas so `width` x `rows` at (`left`, `top`) fits,
    /// preserving the old pixels (`tt_face_colr_blend_layer`'s
    /// resize branch; an empty canvas re-initializes like FreeType's
    /// `dst->bitmap.buffer == NULL` check).
    ///
    /// # Errors
    ///
    /// * [`TtError::ARRAY_TOO_LARGE`] — the union exceeds
    ///   [`MAX_DIMENSION`].
    /// * [`TtError::INVALID_ARGUMENT`] — an inconsistent old buffer
    ///   (an internal invariant; impossible for bitmaps this module
    ///   allocates itself).
    fn resize(&mut self, left: i32, top: i32, width: u32, rows: u32) -> TtResult<()> {
        if self.bitmap.buffer.is_empty() {
            self.left = left;
            self.top = top;
            self.bitmap = Bitmap::new_sized(rows, width, PixelMode::Bgra, 256)?;
            return Ok(());
        }

        let destination_x0 = i64::from(self.left);
        let destination_x1 = destination_x0 + i64::from(self.bitmap.width);
        let destination_y0 = i64::from(self.top) - i64::from(self.bitmap.rows);
        let destination_y1 = i64::from(self.top);
        let source_x0 = i64::from(left);
        let source_x1 = source_x0 + i64::from(width);
        let source_y0 = i64::from(top) - i64::from(rows);
        let source_y1 = i64::from(top);

        let x_min = destination_x0.min(source_x0);
        let x_max = destination_x1.max(source_x1);
        let y_min = destination_y0.min(source_y0);
        let y_max = destination_y1.max(source_y1);
        if x_min == destination_x0
            && x_max == destination_x1
            && y_min == destination_y0
            && y_max == destination_y1
        {
            return Ok(());
        }

        let new_width = x_max - x_min;
        let new_rows = y_max - y_min;
        if new_width < 0 || new_rows < 0 || new_width > MAX_DIMENSION || new_rows > MAX_DIMENSION {
            return Err(TtError::ARRAY_TOO_LARGE);
        }
        let new_width = u32::try_from(new_width).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let new_rows = u32::try_from(new_rows).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let new_left = i32::try_from(x_min).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let new_top = i32::try_from(y_max).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let column_offset = usize::try_from(destination_x0 - x_min).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let row_offset = y_max - destination_y1;

        let mut grown = Bitmap::new_sized(new_rows, new_width, PixelMode::Bgra, 256)?;
        for row in 0..self.bitmap.rows {
            let source_row = self
                .bitmap
                .row(row)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            let target_row =
                u32::try_from(row_offset + i64::from(row)).map_err(|_| TtError::INVALID_ARGUMENT)?;
            let target_row = grown
                .row_mut(target_row)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            let end = column_offset + source_row.len();
            let target_slice = target_row
                .get_mut(column_offset..end)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            target_slice.copy_from_slice(source_row);
        }

        self.bitmap = grown;
        self.left = new_left;
        self.top = new_top;
        Ok(())
    }

    /// Blends one gray layer over the canvas with `color`
    /// (`tt_face_colr_blend_layer`'s inner loop, verbatim): `src` is
    /// coverage (0..=255), `color` is an unpremultiplied BGRA tuple.
    ///
    /// Every channel stays premultiplied because
    /// `channel <= alpha` is preserved by `c * fa / 255 +
    /// old * (255 - fa) / 255`, so the sums never exceed 255.
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_ARGUMENT`] — a slice of either bitmap is
    /// missing (an internal invariant; the layer box is always inside
    /// the canvas after [`Canvas::resize`]).
    fn blend(&mut self, src: &Bitmap, src_left: i32, src_top: i32, color: [u8; 4]) -> TtResult<()> {
        let [blue, green, red, alpha] = color;
        let row_offset = i64::from(self.top) - i64::from(src_top);
        let column_offset = i64::from(src_left) - i64::from(self.left);

        for row in 0..src.rows {
            let source_row = src.row(row).ok_or(TtError::INVALID_ARGUMENT)?;
            let target_index =
                u32::try_from(row_offset + i64::from(row)).map_err(|_| TtError::INVALID_ARGUMENT)?;
            let target_row = self
                .bitmap
                .row_mut(target_index)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            for column in 0..src.width {
                let source_pixel = *source_row
                    .get(column as usize)
                    .ok_or(TtError::INVALID_ARGUMENT)?;
                let coverage = u32::from(source_pixel);
                if coverage == 0 {
                    // `dst * 255 / 255 == dst`, so a transparent
                    // source pixel would be a no-op anyway.
                    continue;
                }

                let factor = u32::from(alpha) * coverage / 255;
                let add_blue = u32::from(blue) * factor / 255;
                let add_green = u32::from(green) * factor / 255;
                let add_red = u32::from(red) * factor / 255;
                let base_factor = 255 - factor;

                let target_column = usize::try_from(column_offset + i64::from(column))
                    .map_err(|_| TtError::INVALID_ARGUMENT)?;
                let target_start = target_column * 4;
                let target_pixel = target_row
                    .get_mut(target_start..target_start + 4)
                    .ok_or(TtError::INVALID_ARGUMENT)?;
                // The premultiplied invariant keeps every result at
                // most 255, so the casts cannot truncate.
                target_pixel[0] = (u32::from(target_pixel[0]) * base_factor / 255 + add_blue) as u8;
                target_pixel[1] = (u32::from(target_pixel[1]) * base_factor / 255 + add_green) as u8;
                target_pixel[2] = (u32::from(target_pixel[2]) * base_factor / 255 + add_red) as u8;
                target_pixel[3] = (u32::from(target_pixel[3]) * base_factor / 255 + factor) as u8;
            }
        }
        Ok(())
    }
}

/// Resolves a layer's `CPAL` entry to an unpremultiplied BGRA tuple
/// (`tt_face_colr_blend_layer`'s color selection): the foreground
/// sentinel maps to white on a dark palette and black otherwise.
///
/// # Errors
///
/// Forwarded from [`CpalTable::color`] for an out-of-range entry.
fn resolve_color(cpal: &CpalTable<'_>, palette: u16, entry: u16) -> TtResult<[u8; 4]> {
    if entry == COLOR_FOREGROUND {
        let opaque_white_on_dark = cpal.is_dark_background(palette);
        return Ok(if opaque_white_on_dark {
            [255, 255, 255, 255]
        } else {
            [0, 0, 0, 255]
        });
    }
    let color = cpal.color(palette, entry)?;
    Ok([color.blue, color.green, color.red, color.alpha])
}

/// Rasterizes `slot`'s outline into a fresh gray bitmap and returns
/// it with the preset box (`ft_glyphslot_preset_bitmap` plus
/// `ft_smooth_render`'s translate/render/translate-back sequence).
///
/// The outline is shifted so the box's top-left corner sits at the
/// origin, rendered, and shifted back even when the raster fails.
///
/// # Errors
///
/// * [`TtError::RASTER_OVERFLOW`] — the box leaves FreeType's
///   `0x7FFF` range or is inverted.
/// * [`TtError::ARRAY_TOO_LARGE`] — the box exceeds this crate's
///   [`MAX_DIMENSION`] guard (a deviation from FreeType, which would
///   run out of memory instead).
/// * Forwarded from the raster for a corrupt outline.
fn render_outline(slot: &mut GlyphSlot) -> TtResult<(Bitmap, i32, i32)> {
    let cbox = slot.outline.get_cbox();
    let left = cbox.x_min >> 6;
    let right = cbox.x_max.saturating_add(63) >> 6;
    let bottom = cbox.y_min >> 6;
    let top = cbox.y_max.saturating_add(63) >> 6;
    if left < -0x8000 || right > 0x7FFF || bottom < -0x8000 || top > 0x7FFF || right < left || top < bottom {
        return Err(TtError::RASTER_OVERFLOW);
    }
    let width = right - left;
    let height = top - bottom;
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(TtError::ARRAY_TOO_LARGE);
    }

    let (pixel_left, pixel_top) = (left as i32, top as i32);
    if width == 0 || height == 0 {
        // FreeType leaves the buffer NULL for an empty preset.
        let empty = Bitmap::new_sized(0, 0, PixelMode::Gray, 256)?;
        return Ok((empty, pixel_left, pixel_top));
    }

    let mut target = Bitmap::new_sized(height as u32, width as u32, PixelMode::Gray, 256)?;
    let x_shift = -64 * left;
    let y_shift = -64 * top + 64 * height;
    if x_shift != 0 || y_shift != 0 {
        slot.outline.translate(x_shift, y_shift);
    }

    let mut raster = GrayRaster::new();
    let rendered = {
        let mut params = RasterParams {
            target: &mut target,
            source: &slot.outline,
            flags: RasterFlags::AA,
            gray_spans: None,
            user: None,
            clip_box: BBox::new(),
        };
        raster.render(&mut params)
    };

    if x_shift != 0 || y_shift != 0 {
        slot.outline.translate(-x_shift, -y_shift);
    }
    rendered?;
    Ok((target, pixel_left, pixel_top))
}

/// Rounds `value * target / base` half away from zero, in the style
/// of FreeType's `FT_MulFix` (used for strike metrics, where `base`
/// is the strike ppem and `target` the requested one).
#[inline]
fn scale_round(value: i64, target: u16, base: u32) -> i64 {
    if base == 0 {
        return value;
    }
    let scaled = value.saturating_mul(i64::from(target));
    let magnitude = scaled.unsigned_abs();
    let numerator = magnitude
        .saturating_mul(2)
        .saturating_add(u64::from(base));
    let rounded = numerator / (2 * u64::from(base));
    let rounded = i64::try_from(rounded).unwrap_or(i64::MAX);
    if scaled < 0 { -rounded } else { rounded }
}

/// Scales a bitmap dimension (`value * target / base`, rounded half
/// up) with a floor of one pixel for any non-empty input.
///
/// # Errors
///
/// [`TtError::ARRAY_TOO_LARGE`] — the result does not fit the
/// bitmap limits.
fn scale_dim(length: u32, target: u16, base: u32) -> TtResult<u32> {
    if base == 0 {
        return Ok(length);
    }
    if length == 0 {
        return Ok(0);
    }
    let numerator = 2 * u64::from(length) * u64::from(target) + u64::from(base);
    let scaled = numerator / (2 * u64::from(base));
    if scaled == 0 {
        return Ok(1);
    }
    let scaled = u32::try_from(scaled).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
    if i64::from(scaled) > MAX_DIMENSION {
        return Err(TtError::ARRAY_TOO_LARGE);
    }
    Ok(scaled)
}

/// Resamples a BGRA bitmap to `width` x `rows` with a box filter
/// (averaging source pixels; upscaling degenerates to nearest
/// neighbor, which is what the filter computes anyway).
///
/// # Errors
///
/// * [`TtError::INVALID_FILE_FORMAT`] — the source is not BGRA or a
///   row/pixel slice is missing.
fn resample_bgra(source: &Bitmap, width: u32, rows: u32) -> TtResult<Bitmap> {
    if source.pixel_mode != PixelMode::Bgra {
        return Err(TtError::INVALID_FILE_FORMAT);
    }
    let mut target = Bitmap::new_sized(rows, width, PixelMode::Bgra, 256)?;
    if source.rows == 0 || source.width == 0 || rows == 0 || width == 0 {
        return Ok(target);
    }

    for y in 0..rows {
        let y0 = u64::from(y) * u64::from(source.rows) / u64::from(rows);
        let y1 = ((u64::from(y) + 1) * u64::from(source.rows) / u64::from(rows))
            .max(y0 + 1)
            .min(u64::from(source.rows));
        let target_row = target
            .row_mut(y)
            .ok_or(TtError::INVALID_ARGUMENT)?;
        for x in 0..width {
            let x0 = u64::from(x) * u64::from(source.width) / u64::from(width);
            let x1 = ((u64::from(x) + 1) * u64::from(source.width) / u64::from(width))
                .max(x0 + 1)
                .min(u64::from(source.width));

            let mut sums = [0u64; 4];
            let mut count = 0u64;
            for row in y0..y1 {
                let source_row = source
                    .row(row as u32)
                    .ok_or(TtError::INVALID_FILE_FORMAT)?;
                for column in x0..x1 {
                    let start = column as usize * 4;
                    let pixel = source_row
                        .get(start..start + 4)
                        .ok_or(TtError::INVALID_FILE_FORMAT)?;
                    for (sum, channel) in sums.iter_mut().zip(pixel) {
                        *sum += u64::from(*channel);
                    }
                    count += 1;
                }
            }
            // `count` is at least one: both ranges are non-empty.
            let averages = [
                (sums[0] + count / 2) / count,
                (sums[1] + count / 2) / count,
                (sums[2] + count / 2) / count,
                (sums[3] + count / 2) / count,
            ];
            let start = usize::try_from(x * 4).map_err(|_| TtError::INVALID_ARGUMENT)?;
            let pixel = target_row
                .get_mut(start..start + 4)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            for (target, average) in pixel.iter_mut().zip(averages) {
                // Each average is of bytes, hence at most 255.
                *target = average as u8;
            }
        }
    }
    Ok(target)
}

/// `FT_DivFix(ppem << 6, units_per_em)` size metrics for `ppem`.
fn scaled_size(units_per_em: u16, ppem: u16) -> SizeMetrics {
    let scale = div_fix(i64::from(ppem) * 64, i64::from(units_per_em));
    SizeMetrics {
        x_ppem: ppem,
        y_ppem: ppem,
        x_scale: scale,
        y_scale: scale,
        ..SizeMetrics::default()
    }
}

/// A [`GlyphSlot`] shaped like the one an outline-capable driver
/// builds: `internal` present and holding a fresh [`GlyphLoader`].
fn new_slot() -> GlyphSlot {
    GlyphSlot {
        library: Arc::new(Library::new()),
        generic: Generic::default(),
        metrics: GlyphMetrics::default(),
        linear_hori_advance: 0,
        linear_vert_advance: 0,
        advance: Vector::new(0, 0),
        format: GlyphFormat::None,
        bitmap: Bitmap::new(),
        bitmap_left: 0,
        bitmap_top: 0,
        outline: Outline::new(),
        subglyphs: Vec::new(),
        control_data: None,
        lsb_delta: 0,
        rsb_delta: 0,
        other: None,
        internal: Some(SlotInternal {
            loader: Some(GlyphLoader::new()),
            flags: 0,
            glyph_transformed: false,
            glyph_matrix: Matrix::default(),
        }),
        driver_data: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{
        SbitImageSpec, cblc_cbdt17, color_font, colr_table, cpal_table, png_rgba, rect_glyph,
    };

    /// Opens `bytes` as a color font.
    fn open(bytes: &[u8]) -> ColorFont<'_> {
        ColorFont::open(SfntFont::open(bytes, 0).unwrap())
    }

    /// The gray coverage of the pixel at (`row`, `column`).
    fn gray(bitmap: &Bitmap, row: u32, column: u32) -> u8 {
        bitmap.row(row).unwrap()[column as usize]
    }

    /// The premultiplied BGRA pixel at (`row`, `column`).
    fn bgra(bitmap: &Bitmap, row: u32, column: u32) -> [u8; 4] {
        let start = column as usize * 4;
        bitmap.row(row).unwrap()[start..start + 4]
            .try_into()
            .unwrap()
    }

    #[test]
    fn outlined_glyphs_render_the_grid_aligned_preset_box() {
        let bytes = color_font(&[rect_glyph(0, 0, 500, 700)], &[(600, 0)], &[]);
        let font = open(&bytes);
        assert!(!font.has_color());
        assert_eq!(font.num_palettes(), 0);

        let glyph = font.render_glyph(0, 16, 0).unwrap();
        assert_eq!((glyph.bitmap.width, glyph.bitmap.rows), (8, 12));
        assert_eq!(glyph.bitmap.pixel_mode, PixelMode::Gray);
        // Hinting rounds the phantoms to the grid; core's `pix_round`
        // shifts the box one pixel left of the advance:
        // pix_round(614) - pix_round(0) = 704 - 64.
        assert_eq!((glyph.bitmap_left, glyph.bitmap_top), (-1, 12));
        assert_eq!(glyph.advance, 640);
        assert_eq!(glyph.source_ppem, 16);

        // The top scanline keeps 13/64 of the glyph; the rows below
        // are fully covered.
        for column in 0..8 {
            assert_eq!(gray(&glyph.bitmap, 0, column), 52, "column {column}");
        }
        assert_eq!(gray(&glyph.bitmap, 1, 3), 255);
        assert_eq!(gray(&glyph.bitmap, 11, 3), 255);
        assert_eq!(glyph.bitmap.buffer.len(), 8 * 12);
    }

    #[test]
    fn colr_layers_blend_into_one_bgra_canvas() {
        let glyphs = [
            Vec::new(),
            rect_glyph(0, 0, 500, 700),
            rect_glyph(0, 0, 500, 700),
            rect_glyph(100, 100, 400, 400),
            Vec::new(),
        ];
        let metrics = [(0, 0), (600, 0), (600, 0), (600, 100), (600, 0)];
        let extra = [
            (TAG_COLR, colr_table(4, &[(2, 0), (3, 1)])),
            (TAG_CPAL, cpal_table(&[(0, 0, 255, 255), (0, 255, 0, 255)], false)),
        ];
        let bytes = color_font(&glyphs, &metrics, &extra);
        let font = open(&bytes);
        assert!(font.has_color());
        assert_eq!(font.num_palettes(), 1);

        let glyph = font.render_glyph(4, 16, 0).unwrap();
        assert_eq!((glyph.bitmap.width, glyph.bitmap.rows), (8, 12));
        assert_eq!(glyph.bitmap.pixel_mode, PixelMode::Bgra);
        assert_eq!((glyph.bitmap_left, glyph.bitmap_top), (-1, 12));
        // The base glyph is empty, so the advance is the unrounded
        // hmtx value: mul_fix(600, scale) = 614.
        assert_eq!(glyph.advance, 614);

        // The red layer covers the whole canvas apart from its top
        // scanline, where it keeps 13/64 coverage.
        assert_eq!(bgra(&glyph.bitmap, 0, 0), [0, 0, 52, 52]);
        assert_eq!(bgra(&glyph.bitmap, 3, 1), [0, 0, 255, 255]);
        assert_eq!(bgra(&glyph.bitmap, 11, 7), [0, 0, 255, 255]);
        // The green layer sits inside the canvas and overwrites the
        // red one where it is opaque...
        assert_eq!(bgra(&glyph.bitmap, 6, 2), [0, 255, 0, 255]);
        // ...and blends with 104/255 coverage at its edges.
        assert_eq!(bgra(&glyph.bitmap, 6, 1), [0, 104, 151, 255]);
        // The second layer never reaches the rows above its box.
        assert_eq!(bgra(&glyph.bitmap, 1, 1), [0, 0, 255, 255]);
    }

    #[test]
    fn foreground_layers_follow_the_palette_background_hint() {
        let glyphs = [Vec::new(), rect_glyph(0, 0, 500, 700), Vec::new()];
        let metrics = [(600, 0), (600, 0), (600, 0)];
        let colr = colr_table(2, &[(1, COLOR_FOREGROUND)]);

        // A plain (light background) palette paints the foreground
        // layer black.
        let light = color_font(
            &glyphs,
            &metrics,
            &[
                (TAG_COLR, colr.clone()),
                (TAG_CPAL, cpal_table(&[(0, 0, 0, 255)], false)),
            ],
        );
        let glyph = open(&light).render_glyph(2, 16, 0).unwrap();
        assert_eq!(glyph.bitmap.pixel_mode, PixelMode::Bgra);
        assert_eq!(bgra(&glyph.bitmap, 3, 1), [0, 0, 0, 255]);

        // Version 1 palettes flagged for dark backgrounds paint it
        // white instead.
        let dark = color_font(
            &glyphs,
            &metrics,
            &[(TAG_COLR, colr), (TAG_CPAL, cpal_table(&[(0, 0, 0, 255)], true))],
        );
        let glyph = open(&dark).render_glyph(2, 16, 0).unwrap();
        assert_eq!(bgra(&glyph.bitmap, 3, 1), [255, 255, 255, 255]);
    }

    #[test]
    fn strike_images_render_at_the_strike_size_and_resample() {
        let row = [255, 0, 0, 255, 255, 0, 0, 255];
        let spec = SbitImageSpec {
            gid: 1,
            height: 2,
            width: 2,
            bearing_x: 1,
            bearing_y: 2,
            advance: 3,
            png: png_rgba(2, 2, &[&row, &row]),
        };
        let (cblc, cbdt) = cblc_cbdt17(16, 32, &[spec]);
        let bytes = color_font(
            &[rect_glyph(0, 0, 500, 700), Vec::new()],
            &[(600, 0), (600, 0)],
            &[(TAG_CBLC, cblc), (TAG_CBDT, cbdt)],
        );
        let font = open(&bytes);
        assert!(font.has_color());
        assert_eq!(font.num_palettes(), 0);

        // Exact strike size: no resampling.
        let exact = font.render_glyph(1, 16, 0).unwrap();
        assert_eq!((exact.bitmap.width, exact.bitmap.rows), (2, 2));
        assert_eq!(exact.bitmap.pixel_mode, PixelMode::Bgra);
        assert_eq!((exact.bitmap_left, exact.bitmap_top), (1, 2));
        assert_eq!(exact.advance, 192);
        for row in 0..2 {
            for column in 0..2 {
                assert_eq!(bgra(&exact.bitmap, row, column), [0, 0, 255, 255]);
            }
        }

        // Twice the ppem: the box filter doubles every edge.
        let doubled = font.render_glyph(1, 32, 0).unwrap();
        assert_eq!((doubled.bitmap.width, doubled.bitmap.rows), (4, 4));
        assert_eq!((doubled.bitmap_left, doubled.bitmap_top), (2, 4));
        assert_eq!(doubled.advance, 384);
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(bgra(&doubled.bitmap, row, column), [0, 0, 255, 255]);
            }
        }

        // A glyph outside the strike falls back to its outline.
        let missed = font.render_glyph(0, 16, 0).unwrap();
        assert_eq!((missed.bitmap.width, missed.bitmap.rows), (8, 12));
        assert_eq!(missed.bitmap.pixel_mode, PixelMode::Gray);
        assert_eq!((missed.bitmap_left, missed.bitmap_top), (-1, 12));
        assert_eq!(missed.advance, 640);
    }

    #[test]
    fn invalid_arguments_and_missing_cmap_entries_are_rejected() {
        let bytes = color_font(
            &[rect_glyph(0, 0, 500, 700)],
            &[(600, 0)],
            &[(TAG_CPAL, cpal_table(&[(0, 0, 0, 255)], false))],
        );
        let font = open(&bytes);
        // A palette without COLR still counts as no color.
        assert!(!font.has_color());
        assert_eq!(font.num_palettes(), 1);
        assert_eq!(
            font.render_glyph(0, 16, 0)
                .unwrap()
                .bitmap
                .pixel_mode,
            PixelMode::Gray
        );

        assert!(matches!(
            font.render_glyph(0, 0, 0),
            Err(TtError::INVALID_ARGUMENT)
        ));
        assert!(matches!(
            font.render_glyph(1, 16, 0),
            Err(TtError::INVALID_ARGUMENT)
        ));
        assert!(matches!(
            font.render_glyph(0, 16, 1),
            Err(TtError::INVALID_ARGUMENT)
        ));
        // The fixture font has no `cmap`.
        assert!(matches!(
            font.render_char('\u{1F600}', 16, 0),
            Err(TtError::INVALID_GLYPH_INDEX)
        ));
    }

    #[test]
    fn corrupt_color_tables_are_dropped_and_the_outline_still_renders() {
        let glyphs = [rect_glyph(0, 0, 500, 700)];
        let metrics = [(600, 0)];

        // Garbage CPAL drops both CPAL and COLR.
        let bytes = color_font(
            &glyphs,
            &metrics,
            &[(TAG_CPAL, vec![0xFF; 4]), (TAG_COLR, vec![0xFF; 8])],
        );
        let font = open(&bytes);
        assert!(!font.has_color());
        assert_eq!(font.num_palettes(), 0);
        let glyph = font.render_glyph(0, 16, 0).unwrap();
        assert_eq!(glyph.bitmap.pixel_mode, PixelMode::Gray);
        assert_eq!((glyph.bitmap.width, glyph.bitmap.rows), (8, 12));
        assert_eq!(glyph.advance, 640);

        // CBLC without CBDT: the strikes are ignored.
        let bytes = color_font(&glyphs, &metrics, &[(TAG_CBLC, vec![0xFF; 16])]);
        let font = open(&bytes);
        assert!(!font.has_color());
        let glyph = font.render_glyph(0, 16, 0).unwrap();
        assert_eq!(glyph.bitmap.pixel_mode, PixelMode::Gray);
        assert_eq!((glyph.bitmap.width, glyph.bitmap.rows), (8, 12));
    }
}
