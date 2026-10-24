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

#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::vec::Vec;

use codevar_logger::log_warn;
use codevar_truetype_core::{
    BBox, Bitmap, Generic, GlyphFormat, GlyphLoader, GlyphMetrics, GlyphSlot, Library, LoadFlags, Matrix,
    Outline, PixelMode, Pos, Raster, RasterFlags, RasterParams, SizeMetrics, SlotInternal, TtError, TtResult,
    Vector, atan2, div_fix, mul_fix,
};
use codevar_truetype_glyf::load_glyph;
use codevar_truetype_sfnt::{
    SfntFont,
    tags::{TAG_CBDT, TAG_CBLC, TAG_COLR, TAG_CPAL},
};
use codevar_truetype_smooth::grays::GrayRaster;

use crate::colr::{COLOR_FOREGROUND, ColrTable, CompositeMode, LayerIter, PaintExtend};
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
    /// Left side bearing of the bitmap, in pixels (maybe negative).
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

/// The deepest `COLR` v1 paint nesting this renderer follows.
///
/// `PaintColorGlyph` can reference any base glyph, including one that
/// (directly or indirectly) contains the reference itself; the bound
/// turns such a cycle into a recoverable error.
const MAX_PAINT_DEPTH: u16 = 64;

/// 16.16 fixed-point one.
const FIXED_ONE: i64 = 0x1_0000;

/// One full turn in 16.16 degrees ([`ANGLE_PI`] is 180 degrees).
const FIXED_TURN: i64 = 360 << 16;

/// Saturates an `i128` into `i64`.
#[inline]
fn clamp_i64(value: i128) -> i64 {
    if value > i128::from(i64::MAX) {
        i64::MAX
    } else if value < i128::from(i64::MIN) {
        i64::MIN
    } else {
        value as i64
    }
}

/// 16.16 x 16.16 -> 16.16 with a wide intermediate (unlike
/// [`mul_fix`], which truncates its operands to 32 bits first).
#[inline]
fn mul_wide(a: i64, b: i64) -> i64 {
    clamp_i64(((i128::from(a) * i128::from(b)) + 0x8000) >> 16)
}

/// A design-space to device-space affine transform.
///
/// The linear part is 16.16 and maps 16.16 design coordinates to 26.6
/// device coordinates: `x_device = (x_design * xx + y_design * xy) /
/// 2^32 + tx`.  The translation is already in 26.6 device units.
/// Font-unit (integer) positions use [`Affine::apply_int`], which is
/// the same mapping with the design coordinate shifted left 16 bits.
#[derive(Clone, Copy, Debug)]
struct Affine {
    /// Horizontal scale (16.16).
    xx: i64,
    /// Horizontal shear (16.16).
    xy: i64,
    /// Vertical shear (16.16).
    yx: i64,
    /// Vertical scale (16.16).
    yy: i64,
    /// Horizontal translation in 26.6 device pixels.
    tx: i64,
    /// Vertical translation in 26.6 device pixels.
    ty: i64,
}

impl Affine {
    /// The root transform for `size`: font design units to 26.6
    /// device pixels (`FT_Set_Transform`'s scale component).
    #[inline]
    fn root(size: &SizeMetrics) -> Self {
        Affine {
            xx: size.x_scale,
            xy: 0,
            yx: 0,
            yy: size.y_scale,
            tx: 0,
            ty: 0,
        }
    }

    /// `true` for the identity transform.
    #[inline]
    fn is_identity(self) -> bool {
        self.xx == FIXED_ONE
            && self.xy == 0
            && self.yx == 0
            && self.yy == FIXED_ONE
            && self.tx == 0
            && self.ty == 0
    }

    /// Composes a child transform in front of `self`.
    ///
    /// `matrix` is `[xx, xy, yx, yy, dx, dy]` with a 16.16 linear part
    /// and a 16.16 design-space translation.  The result maps the
    /// child's design coordinates through `matrix` and then through
    /// `self`.
    fn concat(self, matrix: [i64; 6]) -> Self {
        let [xx, xy, yx, yy, dx, dy] = matrix;
        // `apply_f16` already maps the child translation through this
        // transform and adds this transform's own translation.
        let (tx, ty) = self.apply_f16(dx, dy);
        Affine {
            xx: mul_wide(self.xx, xx) + mul_wide(self.xy, yx),
            xy: mul_wide(self.xx, xy) + mul_wide(self.xy, yy),
            yx: mul_wide(self.yx, xx) + mul_wide(self.yy, yx),
            yy: mul_wide(self.yx, xy) + mul_wide(self.yy, yy),
            tx,
            ty,
        }
    }

    /// Maps a 16.16 design point to 26.6 device coordinates.
    #[inline]
    fn apply_f16(self, x: i64, y: i64) -> (i64, i64) {
        let x = i128::from(x);
        let y = i128::from(y);
        (
            clamp_i64(((x * i128::from(self.xx) + y * i128::from(self.xy)) >> 32) + i128::from(self.tx)),
            clamp_i64(((x * i128::from(self.yx) + y * i128::from(self.yy)) >> 32) + i128::from(self.ty)),
        )
    }

    /// Maps an integer design-unit point to 26.6 device coordinates.
    #[inline]
    fn apply_int(self, x: i64, y: i64) -> (i64, i64) {
        (
            clamp_i64(
                i128::from(mul_wide(x, self.xx)) + i128::from(mul_wide(y, self.xy)) + i128::from(self.tx),
            ),
            clamp_i64(
                i128::from(mul_wide(x, self.yx)) + i128::from(mul_wide(y, self.yy)) + i128::from(self.ty),
            ),
        )
    }

    /// Applies the transform to every outline point.
    fn transform_outline(self, outline: &mut Outline) {
        if self.is_identity() {
            return;
        }
        for point in &mut outline.points {
            let (x, y) = self.apply_int(point.x, point.y);
            point.x = x;
            point.y = y;
        }
    }

    /// Maps a 26.6 device point back to 16.16 design coordinates.
    ///
    /// Returns `None` for a singular transform.
    fn invert(self, device_x: i64, device_y: i64) -> Option<(i64, i64)> {
        let det = i128::from(self.xx) * i128::from(self.yy) - i128::from(self.xy) * i128::from(self.yx);
        if det == 0 {
            return None;
        }
        let ux = i128::from(device_x) - i128::from(self.tx);
        let uy = i128::from(device_y) - i128::from(self.ty);
        let fx = ((ux * i128::from(self.yy) - uy * i128::from(self.xy)) << 32) / det;
        let fy = ((uy * i128::from(self.xx) - ux * i128::from(self.yx)) << 32) / det;
        Some((clamp_i64(fx), clamp_i64(fy)))
    }
}

/// A fill source resolved from a `PaintGlyph` child paint.
enum Fill {
    /// A premultiplied BGRA color.
    Solid([u8; 4]),
    /// A gradient evaluated in the fill's design space.
    Gradient(GradientFill),
}

/// One color stop with its color linearized and premultiplied
/// (per the CPAL "Interpolation of colors" requirements).
#[derive(Clone, Copy, Debug)]
struct ResolvedStop {
    /// Stop position in 16.16 of the color line.
    offset: i64,
    /// Stop alpha in `0..=255`.
    alpha: i64,
    /// Linear-light premultiplied channels in B, G, R order; each
    /// value is `linear(0..=65536) * alpha`.
    linear: [i64; 3],
}

/// The geometry of one gradient.
#[derive(Clone, Copy, Debug)]
enum GradientKind {
    /// `PaintLinearGradient`.
    Linear {
        /// Start point (16.16 design units).
        p0: (i64, i64),
        /// End point (16.16 design units).
        p1: (i64, i64),
        /// Rotation point (16.16 design units).
        p2: (i64, i64),
    },
    /// `PaintRadialGradient`.
    Radial {
        /// Start circle center (16.16 design units).
        c0: (i64, i64),
        /// Start radius (16.16 design units).
        r0: i64,
        /// End circle center (16.16 design units).
        c1: (i64, i64),
        /// End radius (16.16 design units).
        r1: i64,
    },
    /// `PaintSweepGradient`.
    Sweep {
        /// Center point (16.16 design units).
        center: (i64, i64),
        /// Start angle in 16.16 degrees.
        start: i64,
        /// End angle in 16.16 degrees.
        end: i64,
    },
}

/// A gradient fill with its color line resolved and its geometry
/// expressed in the design space of the paint graph.
struct GradientFill {
    /// The resolved color stops, in font order.
    stops: Vec<ResolvedStop>,
    /// How colors extend outside the stop interval.
    extend: PaintExtend,
    /// The gradient geometry.
    kind: GradientKind,
    /// Design-to-device mapping used to sample the gradient.
    transform: Affine,
    /// Offset of the first stop (16.16).
    first: i64,
    /// Offset of the last stop (16.16).
    last: i64,
}

impl GradientFill {
    /// Samples the gradient at one 26.6 device point.
    ///
    /// Returns `None` when the point is not painted (the geometry is
    /// degenerate or the radial cone does not cover it).
    fn sample(&self, device_x: i64, device_y: i64) -> Option<[u8; 4]> {
        let (x, y) = self.transform.invert(device_x, device_y)?;
        let t = match self.kind {
            GradientKind::Linear { p0, p1, p2 } => linear_parameter(p0, p1, p2, (x, y))?,
            GradientKind::Radial { c0, r0, c1, r1 } => radial_parameter(c0, r0, c1, r1, (x, y))?,
            GradientKind::Sweep { center, start, end } => {
                sweep_parameter(center, start, end, (x, y), self.extend)?
            }
        };
        self.color_at(t)
    }

    /// Resolves the color at color-line position `t` (16.16), applying
    /// the extend mode and interpolating between adjacent stops.
    fn color_at(&self, t: i64) -> Option<[u8; 4]> {
        let last = *self.stops.last()?;
        if self.last <= self.first {
            return Some(stop_color(&last));
        }
        let t = match self.extend {
            PaintExtend::Pad => t.clamp(self.first, self.last),
            PaintExtend::Repeat => {
                if t == self.last {
                    self.last
                } else {
                    let span = self.last - self.first;
                    let mut position = (t - self.first) % span;
                    if position < 0 {
                        position += span;
                    }
                    self.first + position
                }
            }
            PaintExtend::Reflect => {
                let span = self.last - self.first;
                let period = span.saturating_mul(2);
                let mut position = (t - self.first) % period;
                if position < 0 {
                    position += period;
                }
                if position > span {
                    position = period - position;
                }
                self.first + position
            }
        };

        if t <= self.first {
            // The spec resolves a duplicate offset to the last stop at
            // or above it.
            let mut chosen = 0usize;
            for (index, stop) in self.stops.iter().enumerate() {
                if stop.offset != self.first {
                    break;
                }
                chosen = index;
            }
            return self.stops.get(chosen).map(stop_color);
        }
        if t >= self.last {
            return Some(stop_color(&last));
        }
        for index in 1..self.stops.len() {
            let upper = &self.stops[index];
            if t < upper.offset {
                let lower = &self.stops[index - 1];
                let span = upper.offset - lower.offset;
                if span <= 0 {
                    continue;
                }
                let factor = clamp_i64((i128::from(t - lower.offset) << 16) / i128::from(span));
                return Some(interpolate_stop(lower, upper, factor));
            }
        }
        Some(stop_color(&last))
    }
}

/// Computes the linear-gradient parameter for design point `q`
/// (16.16 units): the position of the line through `q` parallel to
/// `p0p2` where it crosses `p0p1`.
fn linear_parameter(p0: (i64, i64), p1: (i64, i64), p2: (i64, i64), q: (i64, i64)) -> Option<i64> {
    if p0 == p1 || p0 == p2 {
        return None;
    }
    let dx = i128::from(p1.0) - i128::from(p0.0);
    let dy = i128::from(p1.1) - i128::from(p0.1);
    let rx = i128::from(p2.0) - i128::from(p0.0);
    let ry = i128::from(p2.1) - i128::from(p0.1);
    let denominator = dx * ry - dy * rx;
    if denominator == 0 {
        return None;
    }
    let qx = i128::from(q.0) - i128::from(p0.0);
    let qy = i128::from(q.1) - i128::from(p0.1);
    let numerator = qx * ry - qy * rx;
    Some(clamp_i64((numerator << 16) / denominator))
}

/// Integer square root of `value`.
fn isqrt(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut guess = value;
    let mut next = guess.div_ceil(2);
    while next < guess {
        guess = next;
        next = (guess + value / guess) / 2;
    }
    guess
}

/// Computes the radial-gradient parameter for design point `q`
/// following the WHATWG canvas algorithm: of the circles that pass
/// through `q`, the one with the largest `ω` and a positive radius
/// wins.
fn radial_parameter(c0: (i64, i64), r0: i64, c1: (i64, i64), r1: i64, q: (i64, i64)) -> Option<i64> {
    let dcx = i128::from(c1.0) - i128::from(c0.0);
    let dcy = i128::from(c1.1) - i128::from(c0.1);
    let dr = i128::from(r1) - i128::from(r0);
    let px = i128::from(q.0) - i128::from(c0.0);
    let py = i128::from(q.1) - i128::from(c0.1);

    let a = dcx * dcx + dcy * dcy - dr * dr;
    let b = -2 * (px * dcx + py * dcy + i128::from(r0) * dr);
    let c = px * px + py * py - i128::from(r0) * i128::from(r0);

    let (numerator, denominator) = if a == 0 {
        // |c1 - c0| == |r1 - r0|: the equation degenerates to
        // `b * ω + c = 0`.  `b == 0` leaves either no solution or
        // every ω a solution, neither of which names a color.
        if b == 0 {
            return None;
        }
        // The root must describe a circle with a positive radius:
        // `r(ω) * b == r0 * b - c * dr` must share the sign of `b`.
        let radius = i128::from(r0) * b - c * dr;
        if radius == 0 || (radius > 0) != (b > 0) {
            return None;
        }
        (-c, b)
    } else {
        let discriminant = b * b - 4 * a * c;
        if discriminant < 0 {
            return None;
        }
        let root = i128::try_from(isqrt(discriminant as u128)).unwrap_or(i128::MAX);
        let denominator = 2 * a;
        let lower = -b - root;
        let upper = -b + root;

        // `r(ω) = r0 + (n / denominator) * dr` is positive exactly
        // when `r0 * denominator + n * dr` has the sign of
        // `denominator`.
        let lower_radius = i128::from(r0) * denominator + lower * dr;
        let upper_radius = i128::from(r0) * denominator + upper * dr;
        let lower_valid = lower_radius != 0 && (lower_radius > 0) == (denominator > 0);
        let upper_valid = upper_radius != 0 && (upper_radius > 0) == (denominator > 0);

        match (lower_valid, upper_valid) {
            (true, true) => {
                // Largest ω = largest numerator over a positive
                // denominator, smallest over a negative one.
                if denominator > 0 {
                    (upper, denominator)
                } else {
                    (lower, denominator)
                }
            }
            (true, false) => (lower, denominator),
            (false, true) => (upper, denominator),
            (false, false) => return None,
        }
    };

    Some(clamp_i64((numerator << 16) / denominator))
}

/// Computes the sweep-gradient parameter for design point `q`.
///
/// Offsets grow from `start` to `end` in the direction implied by
/// their order; a point's angle is unwrapped into the same revolution.
fn sweep_parameter(
    center: (i64, i64),
    start: i64,
    end: i64,
    q: (i64, i64),
    extend: PaintExtend,
) -> Option<i64> {
    if start == end {
        return match extend {
            PaintExtend::Pad => Some(0),
            PaintExtend::Repeat | PaintExtend::Reflect => None,
        };
    }
    let angle = atan2(q.0 - center.0, q.1 - center.1);
    let span = end - start;
    let mut delta = angle - start;
    if span > 0 {
        delta = delta.rem_euclid(FIXED_TURN);
    } else {
        delta = -((-delta).rem_euclid(FIXED_TURN));
    }
    Some(clamp_i64((i128::from(delta) << 16) / i128::from(span)))
}

/// Converts a premultiplied linear-light pair of stops into a
/// premultiplied BGRA pixel, un-premultiplying before re-applying the
/// sRGB transfer function.
fn interpolate_stop(lower: &ResolvedStop, upper: &ResolvedStop, factor: i64) -> [u8; 4] {
    let inverse = FIXED_ONE - factor;
    let alpha = (lower.alpha * inverse + upper.alpha * factor) / FIXED_ONE;
    if alpha <= 0 {
        return [0, 0, 0, 0];
    }
    let mut color = [0u8; 4];
    for (channel, target) in color[..3].iter_mut().enumerate() {
        let linear = (lower.linear[channel] * inverse + upper.linear[channel] * factor) / FIXED_ONE;
        let linear = linear / alpha;
        let byte = linear_to_srgb(linear.clamp(0, 65536) as u32);
        *target = ((i64::from(byte) * alpha + 127) / 255) as u8;
    }
    color[3] = alpha as u8;
    color
}

/// Converts one resolved stop into a premultiplied BGRA pixel.
fn stop_color(stop: &ResolvedStop) -> [u8; 4] {
    if stop.alpha <= 0 {
        return [0, 0, 0, 0];
    }
    let mut color = [0u8; 4];
    for (channel, target) in color[..3].iter_mut().enumerate() {
        let linear = (stop.linear[channel] / stop.alpha).clamp(0, 65536) as u32;
        let byte = linear_to_srgb(linear);
        *target = ((i64::from(byte) * stop.alpha + 127) / 255) as u8;
    }
    color[3] = stop.alpha as u8;
    color
}

/// The sRGB electro-optical transfer function, sampled at every
/// 8-bit input and scaled to 16.16.
const SRGB_TO_LINEAR: [u32; 256] = [
    0, 20, 40, 60, 80, 99, 119, 139, 159, 179, 199, 219, 241, 264, 288, 313, 340, 367, 396, 427, 458, 491,
    526, 562, 599, 637, 677, 718, 761, 805, 851, 898, 947, 997, 1048, 1101, 1156, 1212, 1270, 1330, 1391,
    1453, 1517, 1583, 1651, 1720, 1791, 1863, 1937, 2013, 2090, 2170, 2250, 2333, 2418, 2504, 2592, 2681,
    2773, 2866, 2961, 3058, 3157, 3258, 3360, 3464, 3570, 3678, 3788, 3900, 4014, 4129, 4247, 4366, 4488,
    4611, 4736, 4864, 4993, 5124, 5257, 5392, 5530, 5669, 5810, 5953, 6099, 6246, 6395, 6547, 6701, 6856,
    7014, 7174, 7336, 7500, 7666, 7834, 8004, 8177, 8352, 8529, 8708, 8889, 9072, 9258, 9446, 9636, 9828,
    10022, 10219, 10418, 10619, 10822, 11028, 11236, 11446, 11658, 11873, 12090, 12309, 12531, 12754, 12981,
    13209, 13440, 13673, 13909, 14147, 14387, 14629, 14874, 15122, 15372, 15624, 15878, 16135, 16394, 16656,
    16920, 17187, 17456, 17727, 18001, 18278, 18556, 18838, 19121, 19408, 19696, 19988, 20281, 20578, 20876,
    21178, 21481, 21788, 22096, 22408, 22722, 23038, 23357, 23679, 24003, 24329, 24659, 24991, 25325, 25662,
    26002, 26344, 26689, 27036, 27387, 27739, 28095, 28453, 28813, 29177, 29543, 29911, 30283, 30657, 31033,
    31413, 31795, 32180, 32567, 32957, 33350, 33746, 34144, 34545, 34949, 35355, 35765, 36177, 36591, 37009,
    37429, 37852, 38278, 38707, 39138, 39572, 40009, 40449, 40892, 41337, 41786, 42237, 42691, 43147, 43607,
    44069, 44534, 45003, 45474, 45947, 46424, 46904, 47386, 47871, 48360, 48851, 49345, 49842, 50342, 50844,
    51350, 51859, 52370, 52884, 53402, 53922, 54445, 54972, 55501, 56033, 56568, 57106, 57647, 58191, 58738,
    59288, 59841, 60397, 60956, 61518, 62083, 62651, 63222, 63796, 64373, 64953, 65536,
];

/// Upper bounds for [`linear_to_srgb`]: `LINEAR_THRESHOLDS[i]` is the
/// smallest linear value that rounds to byte `i + 1`.
const LINEAR_THRESHOLDS: [u32; 255] = [
    10, 30, 50, 70, 89, 109, 129, 149, 169, 189, 209, 230, 252, 276, 300, 326, 353, 381, 411, 442, 474, 508,
    544, 580, 618, 657, 697, 739, 783, 828, 874, 922, 972, 1022, 1074, 1128, 1184, 1241, 1300, 1360, 1422,
    1485, 1550, 1617, 1685, 1755, 1827, 1900, 1975, 2051, 2130, 2210, 2291, 2375, 2461, 2548, 2636, 2727,
    2819, 2913, 3009, 3107, 3207, 3309, 3412, 3517, 3624, 3733, 3844, 3957, 4071, 4188, 4306, 4427, 4549,
    4673, 4800, 4928, 5058, 5190, 5324, 5461, 5599, 5739, 5881, 6026, 6172, 6320, 6471, 6624, 6778, 6935,
    7094, 7255, 7418, 7583, 7750, 7919, 8090, 8264, 8440, 8618, 8798, 8980, 9165, 9352, 9541, 9732, 9925,
    10120, 10318, 10518, 10720, 10925, 11132, 11341, 11552, 11765, 11981, 12199, 12420, 12642, 12867, 13095,
    13324, 13556, 13791, 14028, 14267, 14508, 14751, 14998, 15247, 15498, 15751, 16006, 16264, 16525, 16788,
    17053, 17321, 17591, 17864, 18139, 18417, 18697, 18979, 19264, 19552, 19842, 20134, 20429, 20727, 21027,
    21329, 21634, 21942, 22252, 22565, 22880, 23197, 23518, 23841, 24166, 24494, 24825, 25158, 25493, 25832,
    26173, 26516, 26862, 27211, 27563, 27917, 28274, 28633, 28995, 29360, 29727, 30097, 30470, 30845, 31223,
    31604, 31987, 32373, 32762, 33153, 33548, 33945, 34344, 34747, 35152, 35560, 35971, 36384, 36800, 37219,
    37640, 38065, 38492, 38922, 39355, 39790, 40229, 40670, 41114, 41561, 42011, 42464, 42919, 43377, 43838,
    44301, 44768, 45238, 45710, 46185, 46664, 47145, 47628, 48115, 48605, 49098, 49593, 50092, 50593, 51097,
    51604, 52114, 52627, 53143, 53662, 54183, 54708, 55236, 55767, 56300, 56837, 57376, 57919, 58464, 59013,
    59564, 60119, 60676, 61237, 61800, 62367, 62936, 63509, 64084, 64663, 65244,
];

/// Linearizes one 8-bit sRGB channel (result in 16.16).
#[inline]
fn srgb_to_linear(value: u8) -> u32 {
    SRGB_TO_LINEAR[value as usize]
}

/// Applies the inverse sRGB transfer function to a linear 16.16
/// channel (values above one clamp to 255).
fn linear_to_srgb(value: u32) -> u8 {
    if value >= FIXED_ONE as u32 {
        return 255;
    }
    let mut low = 0usize;
    let mut high = LINEAR_THRESHOLDS.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if value < LINEAR_THRESHOLDS[middle] {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low as u8
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

    /// `true` when the canvas holds no pixels.
    #[inline]
    fn is_empty(&self) -> bool {
        self.bitmap.buffer.is_empty()
    }

    /// The canvas box as `(x_min, x_max, y_min, y_max)` in pixels, or
    /// `None` when the canvas holds no pixels.
    #[inline]
    fn bounds(&self) -> Option<(i64, i64, i64, i64)> {
        if self.bitmap.width == 0 || self.bitmap.rows == 0 {
            return None;
        }
        let x_min = i64::from(self.left);
        let x_max = x_min + i64::from(self.bitmap.width);
        let y_max = i64::from(self.top);
        let y_min = y_max - i64::from(self.bitmap.rows);
        Some((x_min, x_max, y_min, y_max))
    }

    /// Hands the canvas's pixels and placement to the caller.
    ///
    /// An empty canvas becomes an empty BGRA bitmap so callers always
    /// see a color bitmap, even when the paint graph painted nothing.
    ///
    /// # Errors
    ///
    /// [`TtError::ARRAY_TOO_LARGE`] — allocating the empty bitmap
    /// overflowed (it cannot for a `0 x 0` bitmap).
    fn into_bgra(self) -> TtResult<(Bitmap, i32, i32)> {
        if self.is_empty() {
            let bitmap = Bitmap::new_sized(0, 0, PixelMode::Bgra, 256)?;
            return Ok((bitmap, 0, 0));
        }
        Ok((self.bitmap, self.left, self.top))
    }

    /// Reads the premultiplied BGRA pixel covering device pixel
    /// (`x`, `y`), or transparent black when it lies outside the
    /// canvas.
    #[inline]
    fn pixel_at(&self, x: i64, y: i64) -> [u8; 4] {
        let column = x - i64::from(self.left);
        let row = i64::from(self.top) - 1 - y;
        if column < 0 || row < 0 {
            return [0; 4];
        }
        let (Ok(column), Ok(row)) = (u32::try_from(column), u32::try_from(row)) else {
            return [0; 4];
        };
        let Some(row) = self.bitmap.row(row) else {
            return [0; 4];
        };
        let start = usize::try_from(column)
            .ok()
            .and_then(|column| column.checked_mul(4))
            .unwrap_or(usize::MAX);
        let Some(pixel) = row.get(start..start.saturating_add(4)) else {
            return [0; 4];
        };
        [pixel[0], pixel[1], pixel[2], pixel[3]]
    }

    /// Composites `source` over this canvas with `SrcOver`
    /// (`PaintColrLayers`' layer combination).
    ///
    /// # Errors
    ///
    /// Forwarded from [`Canvas::composite`].
    fn draw_over(&mut self, source: &Canvas) -> TtResult<()> {
        if source.is_empty() {
            return Ok(());
        }
        let backdrop = core::mem::replace(self, Canvas::empty());
        self.composite(source, &backdrop, CompositeMode::SrcOver)
    }

    /// Fills `mask`'s pixels with `fill` and composites them over the
    /// canvas with `SrcOver`.
    ///
    /// `mask` is a gray coverage bitmap whose box is (`left`, `top`);
    /// gradients are sampled at each pixel center in device space.
    ///
    /// # Errors
    ///
    /// * [`TtError::ARRAY_TOO_LARGE`] — the box exceeds
    ///   [`MAX_DIMENSION`].
    /// * [`TtError::INVALID_ARGUMENT`] — a row slice is missing (an
    ///   internal invariant).
    fn fill_mask(&mut self, mask: &Bitmap, left: i32, top: i32, fill: &Fill) -> TtResult<()> {
        self.resize(left, top, mask.width, mask.rows)?;
        if mask.width == 0 || mask.rows == 0 || self.bitmap.buffer.is_empty() {
            return Ok(());
        }
        let row_offset = i64::from(self.top) - i64::from(top);
        let column_offset = i64::from(left) - i64::from(self.left);
        for row in 0..mask.rows {
            let source_row = mask.row(row).ok_or(TtError::INVALID_ARGUMENT)?;
            let target_index =
                u32::try_from(row_offset + i64::from(row)).map_err(|_| TtError::INVALID_ARGUMENT)?;
            let target_row = self
                .bitmap
                .row_mut(target_index)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            let device_y = i64::from(top) - 1 - i64::from(row);
            for column in 0..mask.width {
                let coverage = u32::from(source_row[column as usize]);
                if coverage == 0 {
                    continue;
                }
                let color = match fill {
                    Fill::Solid(color) => *color,
                    Fill::Gradient(gradient) => {
                        let device_x = (i64::from(left) + i64::from(column)) * 64 + 32;
                        let center_y = device_y * 64 + 32;
                        match gradient.sample(device_x, center_y) {
                            Some(color) => color,
                            None => continue,
                        }
                    }
                };
                let source = [
                    (u32::from(color[0]) * coverage / 255) as u8,
                    (u32::from(color[1]) * coverage / 255) as u8,
                    (u32::from(color[2]) * coverage / 255) as u8,
                    (u32::from(color[3]) * coverage / 255) as u8,
                ];
                let target_column = usize::try_from(column_offset + i64::from(column))
                    .map_err(|_| TtError::INVALID_ARGUMENT)?;
                let target_start = target_column
                    .checked_mul(4)
                    .ok_or(TtError::INVALID_ARGUMENT)?;
                let target_pixel = target_row
                    .get_mut(target_start..target_start + 4)
                    .ok_or(TtError::INVALID_ARGUMENT)?;
                over_pixel(target_pixel, source);
            }
        }
        Ok(())
    }

    /// Composites `source` against `backdrop` with `mode`, storing the
    /// result in this (previously unused) canvas.
    ///
    /// The result covers the union of both boxes; a pixel outside one
    /// operand reads as transparent black, which is what the W3C
    /// compositing model specifies for an absent paint.
    ///
    /// # Errors
    ///
    /// * [`TtError::ARRAY_TOO_LARGE`] — the union exceeds
    ///   [`MAX_DIMENSION`] or does not fit the canvas coordinates.
    /// * [`TtError::INVALID_ARGUMENT`] — a row slice is missing (an
    ///   internal invariant).
    fn composite(&mut self, source: &Canvas, backdrop: &Canvas, mode: CompositeMode) -> TtResult<()> {
        let (x_min, x_max, y_min, y_max) = match (source.bounds(), backdrop.bounds()) {
            (Some(source_box), Some(backdrop_box)) => (
                source_box.0.min(backdrop_box.0),
                source_box.1.max(backdrop_box.1),
                source_box.2.min(backdrop_box.2),
                source_box.3.max(backdrop_box.3),
            ),
            (Some(bounds), None) | (None, Some(bounds)) => bounds,
            (None, None) => return Ok(()),
        };
        let width = x_max - x_min;
        let rows = y_max - y_min;
        if width <= 0 || rows <= 0 || width > MAX_DIMENSION || rows > MAX_DIMENSION {
            return Err(TtError::ARRAY_TOO_LARGE);
        }
        let width = u32::try_from(width).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let rows = u32::try_from(rows).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let left = i32::try_from(x_min).map_err(|_| TtError::ARRAY_TOO_LARGE)?;
        let top = i32::try_from(y_max).map_err(|_| TtError::ARRAY_TOO_LARGE)?;

        let mut bitmap = Bitmap::new_sized(rows, width, PixelMode::Bgra, 256)?;
        for row in 0..rows {
            let target_row = bitmap
                .row_mut(row)
                .ok_or(TtError::INVALID_ARGUMENT)?;
            let y = i64::from(top) - 1 - i64::from(row);
            for column in 0..width {
                let x = i64::from(left) + i64::from(column);
                let pixel = composite_pixel(source.pixel_at(x, y), backdrop.pixel_at(x, y), mode);
                let start = usize::try_from(column)
                    .map_err(|_| TtError::INVALID_ARGUMENT)?
                    .checked_mul(4)
                    .ok_or(TtError::INVALID_ARGUMENT)?;
                let target = target_row
                    .get_mut(start..start + 4)
                    .ok_or(TtError::INVALID_ARGUMENT)?;
                target.copy_from_slice(&pixel);
            }
        }
        self.bitmap = bitmap;
        self.left = left;
        self.top = top;
        Ok(())
    }
}

/// Blends `source` over `target` with `SrcOver`, both premultiplied
/// BGRA; `source` is already scaled by its coverage.
#[inline]
fn over_pixel(target: &mut [u8], source: [u8; 4]) {
    let inverse = 255 - u32::from(source[3]);
    for channel in 0..4 {
        target[channel] =
            ((u32::from(target[channel]) * inverse + 127) / 255 + u32::from(source[channel])) as u8;
    }
}

/// A separable blend function of the W3C Compositing and Blending
/// Level 1 model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlendOp {
    /// `B(Cb, Cs) = Cb * Cs`.
    Multiply,
    /// `B(Cb, Cs) = Cb + Cs - Cb * Cs`.
    Screen,
    /// `B(Cb, Cs) = HardLight(Cs, Cb)`.
    Overlay,
    /// `B(Cb, Cs) = min(Cb, Cs)`.
    Darken,
    /// `B(Cb, Cs) = max(Cb, Cs)`.
    Lighten,
    /// `B(Cb, Cs) = min(1, Cb / (1 - Cs))`.
    ColorDodge,
    /// `B(Cb, Cs) = 1 - min(1, (1 - Cb) / Cs)`.
    ColorBurn,
    /// `B(Cb, Cs) = Multiply(Cb, 2 Cs)` or `Screen(Cb, 2 Cs - 1)`.
    HardLight,
    /// The piecewise `D(Cb)` soft-light function.
    SoftLight,
    /// `B(Cb, Cs) = |Cb - Cs|`.
    Difference,
    /// `B(Cb, Cs) = Cb + Cs - 2 Cb Cs`.
    Exclusion,
}

/// A non-separable blend function of the W3C model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HslOp {
    /// `SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))`.
    Hue,
    /// `SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb))`.
    Saturation,
    /// `SetLum(Cs, Lum(Cb))`.
    Color,
    /// `SetLum(Cb, Lum(Cs))`.
    Luminosity,
}

/// Composites a premultiplied source pixel with a premultiplied
/// backdrop pixel, returning the premultiplied result.
fn composite_pixel(source: [u8; 4], backdrop: [u8; 4], mode: CompositeMode) -> [u8; 4] {
    let source_alpha = u32::from(source[3]);
    let backdrop_alpha = u32::from(backdrop[3]);
    match mode {
        CompositeMode::Clear => porter_duff(source, backdrop, 0, 0),
        CompositeMode::Src => porter_duff(source, backdrop, 255, 0),
        CompositeMode::Dest => porter_duff(source, backdrop, 0, 255),
        CompositeMode::SrcOver => porter_duff(source, backdrop, 255, 255 - source_alpha),
        CompositeMode::DestOver => porter_duff(source, backdrop, 255 - backdrop_alpha, 255),
        CompositeMode::SrcIn => porter_duff(source, backdrop, backdrop_alpha, 0),
        CompositeMode::DestIn => porter_duff(source, backdrop, 0, source_alpha),
        CompositeMode::SrcOut => porter_duff(source, backdrop, 255 - backdrop_alpha, 0),
        CompositeMode::DestOut => porter_duff(source, backdrop, 0, 255 - source_alpha),
        CompositeMode::SrcAtop => porter_duff(source, backdrop, backdrop_alpha, 255 - source_alpha),
        CompositeMode::DestAtop => porter_duff(source, backdrop, 255 - backdrop_alpha, source_alpha),
        CompositeMode::Xor => porter_duff(source, backdrop, 255 - backdrop_alpha, 255 - source_alpha),
        CompositeMode::Plus => plus(source, backdrop),
        CompositeMode::Screen => blend_pixel(source, backdrop, BlendOp::Screen),
        CompositeMode::Overlay => blend_pixel(source, backdrop, BlendOp::Overlay),
        CompositeMode::Darken => blend_pixel(source, backdrop, BlendOp::Darken),
        CompositeMode::Lighten => blend_pixel(source, backdrop, BlendOp::Lighten),
        CompositeMode::ColorDodge => blend_pixel(source, backdrop, BlendOp::ColorDodge),
        CompositeMode::ColorBurn => blend_pixel(source, backdrop, BlendOp::ColorBurn),
        CompositeMode::HardLight => blend_pixel(source, backdrop, BlendOp::HardLight),
        CompositeMode::SoftLight => blend_pixel(source, backdrop, BlendOp::SoftLight),
        CompositeMode::Difference => blend_pixel(source, backdrop, BlendOp::Difference),
        CompositeMode::Exclusion => blend_pixel(source, backdrop, BlendOp::Exclusion),
        CompositeMode::Multiply => blend_pixel(source, backdrop, BlendOp::Multiply),
        CompositeMode::HslHue => hsl_pixel(source, backdrop, HslOp::Hue),
        CompositeMode::HslSaturation => hsl_pixel(source, backdrop, HslOp::Saturation),
        CompositeMode::HslColor => hsl_pixel(source, backdrop, HslOp::Color),
        CompositeMode::HslLuminosity => hsl_pixel(source, backdrop, HslOp::Luminosity),
    }
}

/// The general Porter-Duff operator: `co = Cs * Fa + Cb * Fb` with
/// factors in `0..=255`.
fn porter_duff(source: [u8; 4], backdrop: [u8; 4], source_factor: u32, backdrop_factor: u32) -> [u8; 4] {
    let alpha = (u32::from(source[3]) * source_factor + u32::from(backdrop[3]) * backdrop_factor + 127) / 255;
    let mut pixel = [0u8; 4];
    for channel in 0..3 {
        let value = (u32::from(source[channel]) * source_factor
            + u32::from(backdrop[channel]) * backdrop_factor
            + 127)
            / 255;
        pixel[channel] = value.min(alpha) as u8;
    }
    pixel[3] = alpha as u8;
    pixel
}

/// `Plus`: every premultiplied channel (alpha included) saturates at
/// one.
fn plus(source: [u8; 4], backdrop: [u8; 4]) -> [u8; 4] {
    let mut pixel = [0u8; 4];
    for channel in 0..4 {
        pixel[channel] = source[channel].saturating_add(backdrop[channel]);
    }
    pixel
}

/// Applies a separable blend function and the W3C source-over blend
/// formula.
fn blend_pixel(source: [u8; 4], backdrop: [u8; 4], op: BlendOp) -> [u8; 4] {
    if backdrop[3] == 0 {
        return source;
    }
    if source[3] == 0 {
        return backdrop;
    }
    let source_rgb = unpremultiply(source);
    let backdrop_rgb = unpremultiply(backdrop);
    let blended = [
        blend_channel(backdrop_rgb[0], source_rgb[0], op),
        blend_channel(backdrop_rgb[1], source_rgb[1], op),
        blend_channel(backdrop_rgb[2], source_rgb[2], op),
    ];
    compose_blended(source, backdrop, blended)
}

/// Applies a non-separable (HSL) blend function and the W3C
/// source-over blend formula.
fn hsl_pixel(source: [u8; 4], backdrop: [u8; 4], op: HslOp) -> [u8; 4] {
    if backdrop[3] == 0 {
        return source;
    }
    if source[3] == 0 {
        return backdrop;
    }
    let source_rgb = unpremultiply(source);
    let backdrop_rgb = unpremultiply(backdrop);
    let blended = hsl_blend(backdrop_rgb, source_rgb, op);
    compose_blended(source, backdrop, blended)
}

/// The W3C blending step with its coefficients:
/// `co = (1 - ab) Cs + (1 - as) Cb + as ab B(Cb, Cs)`.
fn compose_blended(source: [u8; 4], backdrop: [u8; 4], blended: [u8; 3]) -> [u8; 4] {
    let source_alpha = u32::from(source[3]);
    let backdrop_alpha = u32::from(backdrop[3]);
    let alpha = source_alpha + backdrop_alpha - (source_alpha * backdrop_alpha + 127) / 255;
    let mut pixel = [0u8; 4];
    for channel in 0..3 {
        let blend = u32::from(blended[2 - channel]);
        let numerator = u32::from(source[channel]) * (255 - backdrop_alpha)
            + u32::from(backdrop[channel]) * (255 - source_alpha)
            + (source_alpha * backdrop_alpha * blend + 127) / 255;
        let value = (numerator + 127) / 255;
        pixel[channel] = value.min(alpha) as u8;
    }
    pixel[3] = alpha as u8;
    pixel
}

/// Evaluates one separable blend function on a single channel.
fn blend_channel(backdrop: u8, source: u8, op: BlendOp) -> u8 {
    let cb = u32::from(backdrop);
    let cs = u32::from(source);
    let value = match op {
        BlendOp::Multiply => multiply_channel(cb, cs),
        BlendOp::Screen => screen_channel(cb, cs),
        BlendOp::Overlay => hard_light_channel(cs, cb),
        BlendOp::Darken => cb.min(cs),
        BlendOp::Lighten => cb.max(cs),
        BlendOp::ColorDodge => {
            if cb == 0 {
                0
            } else if cs == 255 {
                255
            } else {
                (cb * 255 / (255 - cs)).min(255)
            }
        }
        BlendOp::ColorBurn => {
            if cb == 255 {
                255
            } else if cs == 0 {
                0
            } else {
                255 - ((255 - cb) * 255 / cs).min(255)
            }
        }
        BlendOp::HardLight => hard_light_channel(cb, cs),
        BlendOp::SoftLight => return soft_light_channel(backdrop, source),
        BlendOp::Difference => cb.abs_diff(cs),
        BlendOp::Exclusion => (cb + cs).saturating_sub(2 * multiply_channel(cb, cs)),
    };
    value.min(255) as u8
}

/// `Multiply(Cb, Cs) = Cb * Cs`.
#[inline]
fn multiply_channel(backdrop: u32, source: u32) -> u32 {
    (backdrop * source + 127) / 255
}

/// `Screen(Cb, Cs) = Cb + Cs - Cb * Cs`.
#[inline]
fn screen_channel(backdrop: u32, source: u32) -> u32 {
    backdrop + source - multiply_channel(backdrop, source)
}

/// `HardLight(Cb, Cs)`: multiply for a dark source, screen otherwise.
#[inline]
fn hard_light_channel(backdrop: u32, source: u32) -> u32 {
    if source <= 127 {
        multiply_channel(backdrop, 2 * source)
    } else {
        screen_channel(backdrop, 2 * source - 255)
    }
}

/// `SoftLight(Cb, Cs)` (W3C), including the `D(Cb)` helper.
fn soft_light_channel(backdrop: u8, source: u8) -> u8 {
    let cb = i64::from(backdrop);
    let cs = i64::from(source);
    let value = if cs <= 127 {
        // Cb - (1 - 2 Cs) Cb (1 - Cb)
        let scale = 255 - 2 * cs;
        cb - (scale * cb * (255 - cb) + 32512) / 65025
    } else {
        // Cb + (2 Cs - 1) (D(Cb) - Cb)
        let d = if cb <= 63 {
            // ((16 Cb - 12) Cb + 4) Cb
            (((16 * cb - 12 * 255) * cb + 4 * 255 * 255) * cb) / 65025
        } else {
            isqrt((255 * cb) as u128) as i64
        };
        cb + (2 * cs - 255) * (d - cb) / 255
    };
    value.clamp(0, 255) as u8
}

/// Converts a premultiplied BGRA pixel into unpremultiplied RGB.
fn unpremultiply(pixel: [u8; 4]) -> [u8; 3] {
    let alpha = u32::from(pixel[3]);
    if alpha == 0 {
        return [0; 3];
    }
    [
        ((u32::from(pixel[2]) * 255 + alpha / 2) / alpha).min(255) as u8,
        ((u32::from(pixel[1]) * 255 + alpha / 2) / alpha).min(255) as u8,
        ((u32::from(pixel[0]) * 255 + alpha / 2) / alpha).min(255) as u8,
    ]
}

/// Evaluates a non-separable blend function on unpremultiplied RGB.
fn hsl_blend(backdrop: [u8; 3], source: [u8; 3], op: HslOp) -> [u8; 3] {
    let backdrop = scale_rgb(backdrop);
    let source = scale_rgb(source);
    let blended = match op {
        HslOp::Hue => set_lum(set_sat(source, saturation(backdrop)), luminance(backdrop)),
        HslOp::Saturation => set_lum(set_sat(backdrop, saturation(source)), luminance(backdrop)),
        HslOp::Color => set_lum(source, luminance(backdrop)),
        HslOp::Luminosity => set_lum(backdrop, luminance(source)),
    };
    [
        ((blended[0] + 500) / 1000).clamp(0, 255) as u8,
        ((blended[1] + 500) / 1000).clamp(0, 255) as u8,
        ((blended[2] + 500) / 1000).clamp(0, 255) as u8,
    ]
}

/// Scales an RGB triplet to thousandths for the HSL float-free math.
#[inline]
fn scale_rgb(color: [u8; 3]) -> [i64; 3] {
    [
        i64::from(color[0]) * 1000,
        i64::from(color[1]) * 1000,
        i64::from(color[2]) * 1000,
    ]
}

/// `Lum(C) = 0.3 R + 0.59 G + 0.11 B`.
#[inline]
fn luminance(color: [i64; 3]) -> i64 {
    (30 * color[0] + 59 * color[1] + 11 * color[2] + 50) / 100
}

/// The largest channel minus the smallest (`Sat`).
#[inline]
fn saturation(color: [i64; 3]) -> i64 {
    color[0].max(color[1]).max(color[2]) - color[0].min(color[1]).min(color[2])
}

/// W3C `ClipColor`: pulls an out-of-gamut triplet back while keeping
/// its luminosity.
fn clip_color(mut color: [i64; 3]) -> [i64; 3] {
    let lum = luminance(color);
    let low = color[0].min(color[1]).min(color[2]);
    let high = color[0].max(color[1]).max(color[2]);
    if low < 0 && lum > low {
        for channel in &mut color {
            *channel = lum + (*channel - lum) * lum / (lum - low);
        }
    }
    if high > 255_000 && high > lum {
        for channel in &mut color {
            *channel = lum + (*channel - lum) * (255_000 - lum) / (high - lum);
        }
    }
    color
}

/// W3C `SetLum`: shifts every channel so the luminosity becomes `lum`.
#[inline]
fn set_lum(color: [i64; 3], lum: i64) -> [i64; 3] {
    let delta = lum - luminance(color);
    clip_color([color[0] + delta, color[1] + delta, color[2] + delta])
}

/// W3C `SetSat`: rescales the channels so the spread becomes `sat`.
fn set_sat(color: [i64; 3], sat: i64) -> [i64; 3] {
    let high = color[0].max(color[1]).max(color[2]);
    let low = color[0].min(color[1]).min(color[2]);
    if high == low {
        return [0; 3];
    }
    let middle = color[0] + color[1] + color[2] - high - low;
    let scaled_middle = (middle - low) * sat / (high - low);
    let mut result = [0i64; 3];
    for index in 0..3 {
        result[index] = if color[index] == high {
            sat
        } else if color[index] == low {
            0
        } else {
            scaled_middle
        };
    }
    result
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
