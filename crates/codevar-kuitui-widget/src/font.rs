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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! A real font, measured and rasterized with the `codevar-truetype-*`
//! stack.
//!
//! Kuitui measures text through the [`FontMetrics`] trait but never
//! rasterizes it: the backend owns the atlas. This module closes that
//! gap. [`Font`] owns the font bytes, implements [`FontMetrics`] over
//! them so a theme can measure exactly, and additionally walks the
//! glyph pipeline — `SfntFont::open` → `char_index` → `load_glyph` →
//! outline rasterization with `GrayRaster` — to produce the
//! single-channel coverage the `text` shader samples.
//!
//! The pipeline is a port of the repository's `gpu_font_raster` probe,
//! rewritten without its panicking error handling:
//!
//! 1. open the face and map a character to a glyph index;
//! 2. load and scale the glyph outline;
//! 3. crop a bitmap to the outline's control box and translate the
//!    outline into it;
//! 4. rasterize anti-aliased coverage into a gray [`Bitmap`];
//! 5. pack the coverage into a [`GlyphAtlas`] and keep a [`GlyphQuad`].
//!
//! [`Font::layout`] runs the whole loop for a run of text and returns a
//! [`GlyphRun`] whose vertices feed [`crate::shaders::TEXT_VERT_SPIRV`]
//! directly.

use alloc::sync::Arc;
use alloc::vec::Vec;

use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::Size;
use codevar_kuitui::text::{FontMetrics, TextStyle};
use codevar_truetype_core::{
    BBox, Bitmap, Generic, GlyphFormat, GlyphLoader, GlyphMetrics, GlyphSlot, Library, LoadFlags, Matrix,
    Outline, PixelMode, Raster, RasterFlags, RasterParams, SizeMetrics, SlotInternal, TtError, Vector,
    div_fix, mul_fix,
};
use codevar_truetype_glyf::load_glyph;
use codevar_truetype_sfnt::SfntFont;
use codevar_truetype_smooth::grays::GrayRaster;

/// The edge length of the coverage atlas [`Font::layout`] allocates.
const ATLAS_SIZE: u32 = 256;

/// The padding kept between two glyphs in a [`GlyphAtlas`], in pixels.
const ATLAS_PADDING: u32 = 1;

/// Everything that can go wrong while measuring or rasterizing a font.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontError {
    /// The font bytes could not be opened as an SFNT face.
    Open(TtError),
    /// The face carries no outlines (`glyf` or `CFF `).
    NoOutlines,
    /// A glyph outline could not be loaded or scaled.
    Load(TtError),
    /// A glyph outline could not be rasterized.
    Raster(TtError),
    /// A glyph bitmap could not be allocated.
    Bitmap(TtError),
    /// The coverage atlas ran out of room for a glyph.
    AtlasFull,
}

impl core::fmt::Display for FontError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Open(error) => write!(f, "could not open the font: {error:?}"),
            Self::NoOutlines => f.write_str("the font has no outlines"),
            Self::Load(error) => write!(f, "could not load a glyph: {error:?}"),
            Self::Raster(error) => write!(f, "could not rasterize a glyph: {error:?}"),
            Self::Bitmap(error) => write!(f, "could not allocate a glyph bitmap: {error:?}"),
            Self::AtlasFull => f.write_str("the glyph atlas is full"),
        }
    }
}

impl core::error::Error for FontError {}

/// A font face: its bytes, its size and the metrics read from `head`
/// and `hhea`.
///
/// The struct owns the bytes through an [`Arc`], so it is cheap to
/// share between widgets, and it does **not** cache the parsed
/// [`SfntFont`] — that view borrows the bytes and cannot be stored
/// beside them without a self-referential struct. Each measurement or
/// layout re-opens the face, which parses a handful of tables and is
/// far cheaper than rasterizing a glyph.
#[derive(Debug, Clone)]
pub struct Font {
    /// The font bytes.
    bytes: Arc<[u8]>,
    /// The sub-font index inside the container.
    face_index: i32,
    /// The requested pixels per em.
    ppem: u16,
    /// `head.unitsPerEm`.
    units_per_em: u16,
    /// `hhea.ascender`, in font units.
    ascender: i32,
    /// `hhea.descender`, in font units.
    descender: i32,
    /// `hhea.lineGap`, in font units.
    line_gap: i32,
    /// The 16.16 scale from font units to 26.6 pixels.
    scale: i64,
}

impl Font {
    /// Opens a font face from owned bytes at 16 pixels per em.
    ///
    /// # Errors
    ///
    /// * [`FontError::Open`] — the bytes are not a usable SFNT face.
    /// * [`FontError::NoOutlines`] — the face has no outlines.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, FontError> {
        Self::from_arc(Arc::from(bytes))
    }

    /// Opens a font face from an already shared byte buffer.
    ///
    /// # Errors
    ///
    /// Same as [`Font::from_bytes`].
    pub fn from_arc(bytes: Arc<[u8]>) -> Result<Self, FontError> {
        let (units_per_em, ascender, descender, line_gap) = read_face_metrics(&bytes, 0)?;
        let mut font = Self {
            bytes,
            face_index: 0,
            ppem: 16,
            units_per_em,
            ascender,
            descender,
            line_gap,
            scale: 0,
        };
        font.rescale();
        Ok(font)
    }

    /// The face with a different pixels-per-em size.
    #[must_use]
    pub fn with_ppem(mut self, ppem: u16) -> Self {
        self.ppem = ppem.max(1);
        self.rescale();
        self
    }

    /// The face with a different sub-font index.
    ///
    /// The `head` and `hhea` metrics cached by [`Font::from_arc`] are
    /// re-read from the new face, so measurement keeps matching the
    /// glyphs [`Font::sfnt`] will load.
    ///
    /// # Errors
    ///
    /// [`FontError::Open`] when the container has no face at
    /// `face_index`, or [`FontError::NoOutlines`] when that face has no
    /// outlines.
    pub fn with_face_index(mut self, face_index: i32) -> Result<Self, FontError> {
        let (units_per_em, ascender, descender, line_gap) = read_face_metrics(&self.bytes, face_index)?;
        self.face_index = face_index;
        self.units_per_em = units_per_em;
        self.ascender = ascender;
        self.descender = descender;
        self.line_gap = line_gap;
        self.rescale();
        Ok(self)
    }

    /// Recomputes the 16.16 scale from the current ppem and upem.
    fn rescale(&mut self) {
        self.scale = div_fix(i64::from(self.ppem) * 64, i64::from(self.units_per_em));
    }

    /// The requested pixels per em.
    #[inline]
    #[must_use]
    pub const fn ppem(&self) -> u16 {
        self.ppem
    }

    /// `head.unitsPerEm`.
    #[inline]
    #[must_use]
    pub const fn units_per_em(&self) -> u16 {
        self.units_per_em
    }

    /// The distance from the top of the line to the baseline, in pixels.
    #[inline]
    #[must_use]
    pub fn ascent(&self) -> f32 {
        fixed_pixels(mul_fix(i64::from(self.ascender), self.scale))
    }

    /// The distance from the baseline to the bottom of the line, in
    /// pixels (positive below the baseline).
    #[inline]
    #[must_use]
    pub fn descent(&self) -> f32 {
        fixed_pixels(-mul_fix(i64::from(self.descender), self.scale))
    }

    /// The distance between two baselines, in pixels.
    ///
    /// This is `ascender - descender + lineGap` of the `hhea` table.
    #[inline]
    #[must_use]
    pub fn line_height(&self) -> f32 {
        let span = self.ascender - self.descender + self.line_gap;
        fixed_pixels(mul_fix(i64::from(span), self.scale))
    }

    /// Opens the face as a borrowed [`SfntFont`].
    ///
    /// # Errors
    ///
    /// [`FontError::Open`] when the *cached* bytes no longer open,
    /// which can only happen if the buffer was externally mutated.
    pub fn sfnt(&self) -> Result<SfntFont<'_>, FontError> {
        SfntFont::open(&self.bytes, self.face_index).map_err(FontError::Open)
    }

    /// The `FT_Size_Metrics` of the current ppem.
    #[must_use]
    pub fn size_metrics(&self) -> SizeMetrics {
        SizeMetrics {
            x_ppem: self.ppem,
            y_ppem: self.ppem,
            x_scale: self.scale,
            y_scale: self.scale,
            ascender: mul_fix(i64::from(self.ascender), self.scale),
            descender: mul_fix(i64::from(self.descender), self.scale),
            height: mul_fix(
                i64::from(self.ascender - self.descender + self.line_gap),
                self.scale,
            ),
            max_advance: 0,
        }
    }

    /// The glyph index a character maps to, or `0` when it is missing.
    ///
    /// # Errors
    ///
    /// [`FontError::Open`] when the cached bytes no longer open.
    pub fn glyph_index(&self, character: char) -> Result<u32, FontError> {
        Ok(self.sfnt()?.char_index(u32::from(character)))
    }

    /// Loads and rasterizes the glyph for `character`.
    ///
    /// Returns `None` for a character with no glyph, a glyph with an
    /// empty outline (a space) or a glyph whose control box is
    /// degenerate. The returned coverage is row-major, top row first,
    /// one byte per pixel.
    ///
    /// # Errors
    ///
    /// [`FontError::Open`], [`FontError::Load`], [`FontError::Raster`]
    /// or [`FontError::Bitmap`] as the pipeline advances.
    pub fn rasterize(&self, character: char) -> Result<Option<RasterGlyph>, FontError> {
        let font = self.sfnt()?;
        let glyph_index = font.char_index(u32::from(character));
        if glyph_index == 0 {
            return Ok(None);
        }
        let metrics = self.size_metrics();
        let advance = fixed_pixels(mul_fix(
            i64::from(font.advance_for_glyph(glyph_index)),
            self.scale,
        ));
        let mut slot = new_slot();
        let mut raster = GrayRaster::new();
        let Some(bitmap) = rasterize_glyph(&font, &mut slot, &mut raster, &metrics, glyph_index)? else {
            return Ok(None);
        };
        let mut coverage = Vec::with_capacity((bitmap.bitmap.width * bitmap.bitmap.rows) as usize);
        for row in 0..bitmap.bitmap.rows {
            if let Some(slice) = bitmap.bitmap.row(row) {
                coverage.extend_from_slice(slice);
            }
        }
        Ok(Some(RasterGlyph {
            glyph_index,
            width: bitmap.bitmap.width,
            height: bitmap.bitmap.rows,
            left: bitmap.left as i32,
            top: bitmap.top as i32,
            advance,
            coverage,
        }))
    }

    /// Lays `text` out along one horizontal baseline.
    ///
    /// Every glyph is rasterized, packed into a fresh [`GlyphAtlas`] and
    /// recorded as a [`GlyphQuad`] in run space: `x` grows to the right
    /// from the pen start, `y` is the top of the glyph relative to the
    /// baseline (negative above it). A production application should
    /// cache the atlas across frames; the widget crate keeps one per run
    /// so a caller sees the whole pipeline in one value.
    ///
    /// `style` supplies the color and the letter spacing; the font size
    /// is the font's own ppem, not `style.font_size`, so measurements
    /// and pixels agree.
    ///
    /// # Errors
    ///
    /// [`FontError::Open`], [`FontError::Load`], [`FontError::Raster`],
    /// [`FontError::Bitmap`] or [`FontError::AtlasFull`].
    pub fn layout(&self, text: &str, style: &TextStyle) -> Result<GlyphRun, FontError> {
        let font = self.sfnt()?;
        let metrics = self.size_metrics();
        let mut slot = new_slot();
        let mut raster = GrayRaster::new();
        let mut atlas = GlyphAtlas::new(ATLAS_SIZE, ATLAS_SIZE);
        let mut quads = Vec::new();
        let mut pen: i64 = 0;
        let mut furthest: i64 = 0;
        let spacing = (style.letter_spacing * 64.0) as i64;
        let color = color_to_array(style.color);

        for character in text.chars() {
            if character == '\n' {
                pen += spacing;
                continue;
            }
            let glyph_index = font.char_index(u32::from(character));
            let advance = mul_fix(i64::from(font.advance_for_glyph(glyph_index)), self.scale);
            if glyph_index != 0
                && let Some(bitmap) = rasterize_glyph(&font, &mut slot, &mut raster, &metrics, glyph_index)?
            {
                let Some(region) = atlas.insert(&bitmap.bitmap) else {
                    return Err(FontError::AtlasFull);
                };
                let atlas_width = atlas.width() as f32;
                let atlas_height = atlas.height() as f32;
                quads.push(GlyphQuad {
                    glyph_index,
                    x: fixed_pixels(pen) + bitmap.left as f32,
                    y: -(bitmap.top as f32),
                    width: bitmap.bitmap.width as f32,
                    height: bitmap.bitmap.rows as f32,
                    uv_min: [region.x as f32 / atlas_width, region.y as f32 / atlas_height],
                    uv_max: [
                        (region.x + region.width) as f32 / atlas_width,
                        (region.y + region.height) as f32 / atlas_height,
                    ],
                    color,
                });
            }
            pen += advance + spacing;
            furthest = furthest.max(pen);
        }

        Ok(GlyphRun {
            quads,
            atlas,
            size: Size::new(fixed_pixels(furthest), self.line_height()),
            baseline: self.ascent(),
        })
    }
}

/// Reads the `head`/`hhea` metrics [`Font`] caches for one face.
///
/// # Errors
///
/// [`FontError::Open`] when the container has no face at `face_index`,
/// or [`FontError::NoOutlines`] when that face carries no outlines.
fn read_face_metrics(bytes: &[u8], face_index: i32) -> Result<(u16, i32, i32, i32), FontError> {
    let font = SfntFont::open(bytes, face_index).map_err(FontError::Open)?;
    if !font.has_outline() {
        return Err(FontError::NoOutlines);
    }
    let units_per_em = font.units_per_em();
    if units_per_em == 0 {
        return Err(FontError::Open(TtError::INVALID_TABLE));
    }
    let hhea = font.hhea();
    Ok((
        units_per_em,
        i32::from(hhea.ascender),
        i32::from(hhea.descender),
        i32::from(hhea.line_gap),
    ))
}

impl FontMetrics for Font {
    fn line_height(&self, _style: &TextStyle) -> f32 {
        Font::line_height(self)
    }

    fn advance(&self, text: &str, style: &TextStyle) -> f32 {
        let Ok(font) = self.sfnt() else {
            return 0.0;
        };
        let spacing = (style.letter_spacing * 64.0) as i64;
        let mut pen: i64 = 0;
        for character in text.chars() {
            if character == '\n' {
                pen += spacing;
                continue;
            }
            let glyph_index = font.char_index(u32::from(character));
            pen += mul_fix(i64::from(font.advance_for_glyph(glyph_index)), self.scale) + spacing;
        }
        fixed_pixels(pen)
    }
}

/// A rasterized glyph: its coverage and where it sits relative to the
/// pen and the baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterGlyph {
    /// The glyph index inside the font.
    pub glyph_index: u32,
    /// The width of the coverage bitmap in pixels.
    pub width: u32,
    /// The height of the coverage bitmap in pixels.
    pub height: u32,
    /// The horizontal bearing from the pen, in whole pixels.
    pub left: i32,
    /// The vertical bearing from the baseline to the bitmap top, in
    /// whole pixels.
    pub top: i32,
    /// The advance of the glyph from this pen, in pixels.
    pub advance: f32,
    /// The coverage, row-major with the top row first, one byte per
    /// pixel.
    pub coverage: Vec<u8>,
}

/// A single-channel coverage atlas.
///
/// The atlas packs glyph bitmaps with a one-pixel gap using a shelf
/// packer: glyphs are placed left to right, and a new shelf starts when
/// the current row is full. The `.r` channel of the pixels the `text`
/// fragment shader samples is exactly this buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlyphAtlas {
    /// The width of the atlas, in pixels.
    width: u32,
    /// The height of the atlas, in pixels.
    height: u32,
    /// The coverage of every pixel, row-major, one byte per pixel.
    pixels: Vec<u8>,
    /// The x position of the next free cell.
    pen_x: u32,
    /// The y position of the next free cell.
    pen_y: u32,
    /// The height of the shelf currently being filled.
    shelf_height: u32,
}

impl GlyphAtlas {
    /// Creates a zeroed atlas of `width` by `height` pixels.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: alloc::vec![0u8; width as usize * height as usize],
            pen_x: 0,
            pen_y: 0,
            shelf_height: 0,
        }
    }

    /// The width of the atlas, in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The height of the atlas, in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The coverage of every pixel, row-major.
    #[inline]
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Copies a glyph bitmap into the atlas, returning where it landed.
    ///
    /// The atlas stores one byte per pixel, so `bitmap` should be gray
    /// (one byte per pixel). A packed mode such as `Mono` contributes
    /// only the bytes its row actually holds; the rest of the region
    /// stays zeroed. Returns `None` when the glyph is larger than the
    /// atlas or the atlas has no room left; the caller can then grow the
    /// atlas or start a new one.
    pub fn insert(&mut self, bitmap: &Bitmap) -> Option<AtlasRegion> {
        let width = bitmap.width;
        let height = bitmap.rows;
        if width == 0 || height == 0 || width > self.width || height > self.height {
            return None;
        }
        if self.pen_x + width > self.width {
            self.pen_x = 0;
            self.pen_y += self.shelf_height + ATLAS_PADDING;
            self.shelf_height = 0;
        }
        if self.pen_y + height > self.height {
            return None;
        }
        let region = AtlasRegion {
            x: self.pen_x,
            y: self.pen_y,
            width,
            height,
        };
        for row in 0..height {
            let Some(source) = bitmap.row(row) else {
                continue;
            };
            let destination_start = ((region.y + row) * self.width + region.x) as usize;
            let destination_end = destination_start + width as usize;
            if let Some(destination) = self
                .pixels
                .get_mut(destination_start..destination_end)
            {
                // A packed pixel mode has a row stride shorter than `width`;
                // copy only what the row holds and leave the rest zeroed
                // rather than panicking on a length mismatch.
                let copy = (width as usize).min(source.len());
                destination[..copy].copy_from_slice(&source[..copy]);
            }
        }
        self.pen_x += width + ATLAS_PADDING;
        self.shelf_height = self.shelf_height.max(height);
        Some(region)
    }
}

/// Where a glyph bitmap landed inside a [`GlyphAtlas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AtlasRegion {
    /// The left edge of the region, in pixels.
    pub x: u32,
    /// The top edge of the region, in pixels.
    pub y: u32,
    /// The width of the region, in pixels.
    pub width: u32,
    /// The height of the region, in pixels.
    pub height: u32,
}

/// One glyph of a [`GlyphRun`], positioned relative to the baseline
/// origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphQuad {
    /// The glyph index inside the font.
    pub glyph_index: u32,
    /// The left edge of the quad, in pixels from the pen start.
    pub x: f32,
    /// The top edge of the quad, in pixels from the baseline (negative
    /// above it).
    pub y: f32,
    /// The width of the quad, in pixels.
    pub width: f32,
    /// The height of the quad, in pixels.
    pub height: f32,
    /// The normalized top-left atlas coordinate.
    pub uv_min: [f32; 2],
    /// The normalized bottom-right atlas coordinate.
    pub uv_max: [f32; 2],
    /// The non-premultiplied sRGB color of the run.
    pub color: [f32; 4],
}

/// A laid-out run of text: its glyph quads, its coverage atlas and its
/// measured extent.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphRun {
    /// The glyph quads, in reading order.
    pub quads: Vec<GlyphQuad>,
    /// The coverage atlas the quads sample.
    pub atlas: GlyphAtlas,
    /// The extent of the run; `width` is the advance, `height` the line
    /// height.
    pub size: Size,
    /// The distance from the top of the run to the baseline, in pixels.
    pub baseline: f32,
}

impl GlyphRun {
    /// Builds the six vertices of every glyph quad.
    ///
    /// The vertices use the [`crate::shaders::TextVertex`] layout the
    /// `text` shader program declares: two triangles per quad, in
    /// logical pixels relative to `origin` (the baseline start).
    #[must_use]
    pub fn vertices(&self, origin: codevar_kuitui::geometry::Offset) -> Vec<crate::shaders::TextVertex> {
        let mut vertices = Vec::with_capacity(self.quads.len() * 6);
        for quad in &self.quads {
            let left = origin.x + quad.x;
            let top = origin.y + quad.y;
            let right = left + quad.width;
            let bottom = top + quad.height;
            let [u0, v0] = quad.uv_min;
            let [u1, v1] = quad.uv_max;
            let corners = [
                ([left, top], [u0, v0]),
                ([right, top], [u1, v0]),
                ([right, bottom], [u1, v1]),
                ([left, top], [u0, v0]),
                ([right, bottom], [u1, v1]),
                ([left, bottom], [u0, v1]),
            ];
            for (position, uv) in corners {
                vertices.push(crate::shaders::TextVertex::new(position, uv, quad.color));
            }
        }
        vertices
    }
}

/// The result of rasterizing one glyph into a cropped bitmap.
struct GlyphBitmap {
    /// The coverage bitmap.
    bitmap: Bitmap,
    /// The horizontal bearing of the bitmap from the pen, in pixels.
    left: i64,
    /// The vertical bearing from the baseline to the bitmap top, in
    /// pixels.
    top: i64,
}

/// Loads, scales and rasterizes one glyph into a bitmap cropped to its
/// control box.
fn rasterize_glyph(
    font: &SfntFont<'_>,
    slot: &mut GlyphSlot,
    raster: &mut GrayRaster,
    metrics: &SizeMetrics,
    glyph_index: u32,
) -> Result<Option<GlyphBitmap>, FontError> {
    load_glyph(font, slot, metrics, glyph_index, LoadFlags::DEFAULT).map_err(FontError::Load)?;
    if slot.outline.is_empty() {
        return Ok(None);
    }
    let cbox = slot.outline.get_cbox();
    let left = cbox.x_min >> 6;
    let top = cbox.y_max.saturating_add(63) >> 6;
    let width = (cbox.x_max.saturating_add(63) >> 6) - left;
    let height = top - (cbox.y_min >> 6);
    if width <= 0 || height <= 0 {
        return Ok(None);
    }
    let mut bitmap =
        Bitmap::new_sized(height as u32, width as u32, PixelMode::Gray, 256).map_err(FontError::Bitmap)?;
    slot.outline
        .translate(-64 * left, -64 * top + 64 * height);
    let mut params = RasterParams {
        target: &mut bitmap,
        source: &slot.outline,
        flags: RasterFlags::AA,
        gray_spans: None,
        user: None,
        clip_box: BBox::new(),
    };
    raster
        .render(&mut params)
        .map_err(FontError::Raster)?;
    Ok(Some(GlyphBitmap { bitmap, left, top }))
}

/// Builds the driver-shaped glyph slot `load_glyph` requires.
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

/// Converts a 26.6 fixed-point value to whole pixels.
#[inline]
fn fixed_pixels(value: i64) -> f32 {
    value as f32 / 64.0
}

/// Converts a color to the `[r, g, b, a]` array a vertex carries.
#[inline]
fn color_to_array(color: Color) -> [f32; 4] {
    [color.red, color.green, color.blue, color.alpha]
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_kuitui::text::TextStyle;

    const FONT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/tf/codevar.tf.space-mono.regular.ttf"
    );

    fn test_font(ppem: u16) -> Font {
        let bytes = std::fs::read(FONT).unwrap();
        Font::from_bytes(bytes).unwrap().with_ppem(ppem)
    }

    #[test]
    fn a_font_reports_its_em_and_line_metrics() {
        let font = test_font(16);
        assert!(font.units_per_em() > 0);
        assert!(font.ascent() > 0.0);
        assert!(font.line_height() > font.ascent());
    }

    #[test]
    fn the_advance_grows_with_the_size() {
        let small = test_font(16);
        let large = test_font(32);
        let style = TextStyle::new(16.0);
        assert!(large.advance("codevar", &style) > small.advance("codevar", &style));
    }

    #[test]
    fn a_space_has_an_advance_but_no_quad() {
        let font = test_font(16);
        let style = TextStyle::new(16.0);
        assert!(font.advance(" ", &style) > 0.0);
        let run = font.layout(" ", &style).unwrap();
        assert!(run.quads.is_empty());
    }

    #[test]
    fn rasterizing_a_letter_produces_coverage() {
        let font = test_font(24);
        let glyph = font.rasterize('A').unwrap().unwrap();
        assert!(glyph.width > 0 && glyph.height > 0);
        assert_eq!(glyph.coverage.len(), (glyph.width * glyph.height) as usize);
        assert!(glyph.coverage.iter().any(|value| *value > 0));
        assert!(glyph.advance > 0.0);
    }

    #[test]
    fn a_missing_glyph_rasterizes_to_nothing() {
        let font = test_font(16);
        assert!(font.rasterize('\u{10FFFF}').unwrap().is_none());
    }

    #[test]
    fn a_layout_packs_every_glyph_of_a_monospaced_run() {
        let font = test_font(24);
        let style = TextStyle::new(24.0);
        let run = font.layout("Codevar", &style).unwrap();
        assert_eq!(run.quads.len(), 7);
        assert!(run.size.width > 0.0);
        assert!(run.size.height > 0.0);
        assert!(run.atlas.pixels().iter().any(|value| *value > 0));
    }

    #[test]
    fn a_layout_emits_six_vertices_per_quad() {
        let font = test_font(16);
        let style = TextStyle::new(16.0);
        let run = font.layout("hi", &style).unwrap();
        assert_eq!(
            run.vertices(codevar_kuitui::geometry::Offset::ZERO)
                .len(),
            run.quads.len() * 6
        );
    }

    #[test]
    fn the_atlas_runs_out_of_room_instead_of_wrapping() {
        let mut atlas = GlyphAtlas::new(4, 2);
        let mut bitmap = Bitmap::new_sized(2, 4, PixelMode::Gray, 256).unwrap();
        for row in 0..2u32 {
            if let Some(slice) = bitmap.row_mut(row) {
                slice.fill(255);
            }
        }
        assert!(atlas.insert(&bitmap).is_some());
        assert!(atlas.insert(&bitmap).is_none());
    }

    #[test]
    fn a_glyph_larger_than_the_atlas_is_rejected() {
        let mut atlas = GlyphAtlas::new(4, 4);
        let bitmap = Bitmap::new_sized(2, 8, PixelMode::Gray, 256).unwrap();
        assert!(atlas.insert(&bitmap).is_none());
    }

    #[test]
    fn a_packed_bitmap_is_copied_without_panicking() {
        let mut atlas = GlyphAtlas::new(8, 8);
        let bitmap = Bitmap::new_sized(4, 8, PixelMode::Mono, 2).unwrap();
        assert!(atlas.insert(&bitmap).is_some());
    }

    #[test]
    fn an_unknown_face_index_is_an_error() {
        let font = test_font(16);
        assert!(font.with_face_index(9).is_err());
    }
}
