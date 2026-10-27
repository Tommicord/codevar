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
//! The pipeline is a port of the standalone `gpu_font_raster` probe,
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
use codevar_kuitui::geometry::{Offset, Rect, Size};
use codevar_kuitui::text::{FontMetrics, TextStyle};
use codevar_kuitui_text::{GlyphKey, GlyphQuad as AtlasQuad, TextError, TextFrame};
use codevar_truetype_core::{
    BBox, Bitmap, Generic, GlyphFormat, GlyphLoader, GlyphMetrics, GlyphSlot, Library, LoadFlags, Matrix,
    Outline, PixelMode, Raster, RasterFlags, RasterParams, SizeMetrics, SlotInternal, TtError, Vector,
    div_fix, mul_fix,
};
use codevar_truetype_emoji_color::render::{ColorFont, RenderedGlyph};
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

/// Why a run could not be shaped into a [`TextFrame`].
///
/// Shaping advances the glyph pipeline and then asks the shared atlas for
/// a slot, so it has two failure sources. Both are kept distinct so a
/// caller can tell a bad font from a full atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    /// The glyph pipeline could not open, load or rasterize a glyph.
    Font(FontError),
    /// The shared atlas rejected or could not place a glyph.
    Atlas(TextError),
}

impl core::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Font(error) => write!(f, "could not shape the text: {error}"),
            Self::Atlas(error) => write!(f, "could not place the coverage: {error}"),
        }
    }
}

impl core::error::Error for ShapeError {}

impl From<FontError> for ShapeError {
    fn from(error: FontError) -> Self {
        Self::Font(error)
    }
}

impl From<TextError> for ShapeError {
    fn from(error: TextError) -> Self {
        Self::Atlas(error)
    }
}

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
    pub fn layout(&self, text: &str, style: &TextStyle) -> Result<GlyphRun<GlyphAtlas>, FontError> {
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

    /// Lays `text` out into a shared [`TextFrame`].
    ///
    /// This is [`Font::layout`]'s counterpart for the renderer text stack.
    /// Instead of building a private [`GlyphAtlas`], it rasterizes each
    /// glyph, inserts its coverage into the frame's shared atlas and pushes
    /// the quad that samples the slot the atlas returned. A backend that owns
    /// the `text` pipeline drains the same frame, so the CPU stays the only
    /// source of coverage and the GPU is never read back.
    ///
    /// `origin` is the baseline origin of the run and `clip` is the clip the
    /// run was recorded with. As in [`Font::layout`], the size is the font's
    /// own ppem, not `style.font_size`, so measurement and pixels agree.
    ///
    /// # Errors
    ///
    /// * [`ShapeError::Font`] as the glyph pipeline advances;
    /// * [`ShapeError::Atlas`] when the shared atlas cannot take a glyph.
    pub fn shape_into(
        &self,
        frame: &TextFrame,
        origin: Offset,
        text: &str,
        style: &TextStyle,
        clip: Option<Rect>,
    ) -> Result<(), ShapeError> {
        let font = self.sfnt().map_err(ShapeError::Font)?;
        let metrics = self.size_metrics();
        let mut slot = new_slot();
        let mut raster = GrayRaster::new();
        let spacing = (style.letter_spacing * 64.0) as i64;
        let color = color_to_array(style.color);
        let mut coverage = Vec::new();

        frame.with(|atlas, mesh| -> Result<(), ShapeError> {
            let mut pen: i64 = 0;
            for character in text.chars() {
                if character == '\n' {
                    pen += spacing;
                    continue;
                }
                let glyph_index = font.char_index(u32::from(character));
                let advance = mul_fix(i64::from(font.advance_for_glyph(glyph_index)), self.scale);
                if glyph_index != 0
                    && let Some(bitmap) =
                        rasterize_glyph(&font, &mut slot, &mut raster, &metrics, glyph_index)
                            .map_err(ShapeError::Font)?
                {
                    let width = bitmap.bitmap.width;
                    let height = bitmap.bitmap.rows;
                    coverage.clear();
                    for row in 0..height {
                        if let Some(slice) = bitmap.bitmap.row(row) {
                            coverage.extend_from_slice(slice);
                        }
                    }
                    let key = glyph_key(self.ppem, glyph_index);
                    if let Some(slot) = atlas
                        .insert(key, width, height, &coverage)
                        .map_err(ShapeError::Atlas)?
                    {
                        let position = [
                            origin.x + fixed_pixels(pen) + bitmap.left as f32,
                            origin.y - bitmap.top as f32,
                        ];
                        mesh.push(&AtlasQuad::from_slot(slot, position, color, clip));
                    }
                }
                pen += advance + spacing;
            }
            Ok(())
        })
    }

    /// Shapes a color text run into a shared frame.
    ///
    /// `text_font` is the face the run was laid out with and therefore
    /// the primary one: a character it covers is shaped into the
    /// frame's coverage atlas at its advance, whatever the color face
    /// holds. The color face is the fallback — a character the text
    /// face lacks (an emoji, when the text face is a text font) is
    /// rasterized by the color pipeline (strikes → COLR/CPAL → outline)
    /// as premultiplied BGRA, inserted into the frame's color atlas and
    /// pushed with white vertex color so the glyph keeps its own
    /// colors. A run that mixes both keeps both.
    ///
    /// The precedence is not cosmetic: an emoji face usually has its
    /// own, much wider space, so drawing the spaces of a run through it
    /// made the drawn run wider than the width the layout was measured
    /// with and slid the run out of the box it was centered in.
    ///
    /// # Errors
    ///
    /// * [`ShapeError::Font`] when the color or the text font pipeline fails;
    /// * [`ShapeError::Atlas`] when either shared atlas cannot take a glyph.
    pub fn shape_into_color(
        &self,
        frame: &TextFrame,
        text_font: &Font,
        origin: Offset,
        text: &str,
        style: &TextStyle,
        clip: Option<Rect>,
    ) -> Result<(), ShapeError> {
        let color_font = match self.color_font() {
            Some(cf) => cf,
            None => return Err(ShapeError::Font(FontError::Open(TtError::INVALID_TABLE))),
        };
        let spacing = (style.letter_spacing * 64.0) as i64;
        let color = [1.0, 1.0, 1.0, 1.0]; // white so emoji colors show
        let text_color = color_to_array(style.color);

        // Use the underlying sfnt for glyph index mapping (same as gray path).
        let sfnt = self.sfnt().map_err(ShapeError::Font)?;
        // The coverage pipeline of the text face, for the characters the
        // color face does not cover.
        let text_sfnt = text_font.sfnt().map_err(ShapeError::Font)?;
        let text_metrics = text_font.size_metrics();
        let mut slot = new_slot();
        let mut raster = GrayRaster::new();
        let mut coverage = Vec::new();

        frame.with_color(|atlas, mesh| -> Result<(), ShapeError> {
            let mut pen: i64 = 0;
            for character in text.chars() {
                if character == '\n' {
                    pen += spacing;
                    continue;
                }
                // The text face is primary: its advance is the one the
                // layout measured, so a character it covers stays in the
                // coverage atlas even when the color face also has a
                // glyph for it.
                let text_glyph = text_sfnt.char_index(u32::from(character));
                if text_glyph != 0 {
                    let advance = mul_fix(
                        i64::from(text_sfnt.advance_for_glyph(text_glyph)),
                        text_font.scale,
                    );
                    if let Some(bitmap) =
                        rasterize_glyph(&text_sfnt, &mut slot, &mut raster, &text_metrics, text_glyph)
                            .map_err(ShapeError::Font)?
                    {
                        coverage.clear();
                        for row in 0..bitmap.bitmap.rows {
                            if let Some(row) = bitmap.bitmap.row(row) {
                                coverage.extend_from_slice(row);
                            }
                        }
                        let key = glyph_key(text_font.ppem, text_glyph);
                        frame.with(|text_atlas, text_mesh| -> Result<(), ShapeError> {
                            if let Some(slot) = text_atlas
                                .insert(key, bitmap.bitmap.width, bitmap.bitmap.rows, &coverage)
                                .map_err(ShapeError::Atlas)?
                            {
                                let position = [
                                    origin.x + fixed_pixels(pen) + bitmap.left as f32,
                                    origin.y - bitmap.top as f32,
                                ];
                                text_mesh.push(&AtlasQuad::from_slot(slot, position, text_color, clip));
                            }
                            Ok(())
                        })?;
                    }
                    pen += advance + spacing;
                    continue;
                }
                let rendered = match color_font.render_char(character, self.ppem, 0) {
                    Ok(rendered) => rendered,
                    Err(TtError::INVALID_GLYPH_INDEX) => {
                        // Neither face covers the character: the text
                        // face's `.notdef` keeps the pen moving exactly
                        // as the layout measured it.
                        let advance = mul_fix(i64::from(text_sfnt.advance_for_glyph(0)), text_font.scale);
                        pen += advance + spacing;
                        continue;
                    }
                    Err(error) => return Err(ShapeError::Font(FontError::Open(error))),
                };
                let glyph_index = sfnt.char_index(u32::from(character));
                let bitmap = &rendered.bitmap;
                let width = bitmap.width;
                let height = bitmap.rows;
                let key = glyph_key(self.ppem, glyph_index);
                if let Some(slot) = atlas
                    .insert(key, width, height, bitmap.buffer.as_slice())
                    .map_err(ShapeError::Atlas)?
                {
                    let position = [
                        origin.x + fixed_pixels(pen) + rendered.bitmap_left as f32,
                        origin.y - rendered.bitmap_top as f32,
                    ];
                    mesh.push(&AtlasQuad::from_slot(slot, position, color, clip));
                }
                pen += rendered.advance + spacing;
            }
            Ok(())
        })
    }

    /// Opens a [`ColorFont`] for this face, if it has color tables.
    ///
    /// Returns `None` when the font has no `CPAL`/`COLR` or `CBLC`/`CBDT`
    /// tables, or when they fail to parse (the `ColorFont::open` call
    /// logs warnings and degrades to the outline).
    pub fn color_font(&self) -> Option<ColorFont<'_>> {
        let sfnt = self.sfnt().ok()?;
        let color_font = ColorFont::open(sfnt);
        if color_font.has_color() {
            Some(color_font)
        } else {
            None
        }
    }

    /// Rasterizes `character` using the color font pipeline.
    ///
    /// Returns `None` when the character has no glyph, the glyph is empty,
    /// or the font lacks color tables. The returned bitmap is premultiplied
    /// BGRA (`PixelMode::Bgra`), 4 bytes per pixel.
    ///
    /// # Errors
    ///
    /// Propagates errors from the color font pipeline.
    pub fn rasterize_color(&self, character: char) -> Result<Option<RenderedGlyph>, FontError> {
        let color_font = match self.color_font() {
            Some(cf) => cf,
            None => return Ok(None),
        };
        color_font
            .render_char(character, self.ppem, 0)
            .map(Some)
            .or_else(|e| {
                if matches!(e, TtError::INVALID_GLYPH_INDEX) {
                    Ok(None)
                } else {
                    Err(FontError::Open(e))
                }
            })
    }

    /// Lays `text` out using the color font, returning a run with a BGRA atlas.
    ///
    /// This is the color counterpart of [`Font::layout`]. Each glyph is
    /// rasterized through the color pipeline (strikes → COLR layers → outline)
    /// and packed into a [`ColorableGlyphAtlas`]. The run's vertices carry
    /// white as the vertex color so the emoji's own colors are reproduced.
    ///
    /// # Errors
    ///
    /// Same as [`Font::layout`], plus color font pipeline errors.
    pub fn layout_color(
        &self,
        text: &str,
        style: &TextStyle,
    ) -> Result<GlyphRun<ColorableGlyphAtlas>, FontError> {
        let color_font = match self.color_font() {
            Some(cf) => cf,
            None => return Err(FontError::Open(TtError::INVALID_TABLE)),
        };
        let mut atlas = ColorableGlyphAtlas::new(ATLAS_SIZE, ATLAS_SIZE);
        let mut quads = Vec::new();
        let mut pen: i64 = 0;
        let spacing = (style.letter_spacing * 64.0) as i64;
        let color = [1.0, 1.0, 1.0, 1.0]; // white so emoji colors show

        for character in text.chars() {
            if character == '\n' {
                pen += spacing;
                continue;
            }
            let rendered = match color_font.render_char(character, self.ppem, 0) {
                Ok(r) => r,
                Err(TtError::INVALID_GLYPH_INDEX) => {
                    // Character not in font; skip but advance by a default amount?
                    // For now, just skip.
                    continue;
                }
                Err(e) => return Err(FontError::Open(e)),
            };
            let advance = rendered.advance;
            let bitmap = &rendered.bitmap;
            let width = bitmap.width;
            let height = bitmap.rows;
            if let Some(region) = atlas.insert(bitmap) {
                let atlas_width = atlas.width() as f32;
                let atlas_height = atlas.height() as f32;
                quads.push(GlyphQuad {
                    glyph_index: 0, // Not tracked for color runs
                    x: fixed_pixels(pen) + rendered.bitmap_left as f32,
                    y: -(rendered.bitmap_top as f32),
                    width: width as f32,
                    height: height as f32,
                    uv_min: [region.x as f32 / atlas_width, region.y as f32 / atlas_height],
                    uv_max: [
                        (region.x + region.width) as f32 / atlas_width,
                        (region.y + region.height) as f32 / atlas_height,
                    ],
                    color,
                });
            }
            pen += advance + spacing;
        }

        Ok(GlyphRun {
            quads,
            atlas,
            size: Size::new(fixed_pixels(pen), self.line_height()),
            baseline: self.ascent(),
        })
    }
}

/// The atlas key of a glyph: its pixels-per-em and its glyph index.
///
/// The rasterizer lays every glyph out at an integer pen, so two uses of the
/// same glyph at the same size always share one bitmap and therefore one key.
#[inline]
fn glyph_key(ppem: u16, glyph_index: u32) -> GlyphKey {
    GlyphKey::from_raw((u64::from(ppem) << 32) | u64::from(glyph_index))
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

    fn ascent(&self, _style: &TextStyle) -> f32 {
        Font::ascent(self)
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

/// The metrics of a text face with a color face as its fallback.
///
/// [`Font::shape_into_color`] shapes a character through the text face
/// when the text face has a glyph for it and through the color face
/// otherwise, so the advance of a mixed run belongs to neither face
/// alone. This type answers [`FontMetrics`] with that same rule — the
/// line box from the text face, the advance of each character from the
/// face that will draw it — so a run is measured the way it is drawn.
///
/// Measuring a color run with the text face alone reserves the `.notdef`
/// width for every emoji and the run is drawn wider than its box, which
/// is what pushed a centered label off center. An application that
/// shapes color runs installs these metrics on its theme in place of
/// the text face alone.
#[derive(Debug, Clone)]
pub struct ColorFontMetrics {
    /// The face the layout is measured with, and the primary face of
    /// the shaping.
    text: Font,
    /// The face whose glyphs replace the characters `text` lacks.
    color: Font,
}

impl ColorFontMetrics {
    /// The metrics of `text`, falling back to `color`.
    #[must_use]
    pub const fn new(text: Font, color: Font) -> Self {
        Self { text, color }
    }
}

impl FontMetrics for ColorFontMetrics {
    fn line_height(&self, _style: &TextStyle) -> f32 {
        Font::line_height(&self.text)
    }

    fn ascent(&self, _style: &TextStyle) -> f32 {
        Font::ascent(&self.text)
    }

    fn advance(&self, text: &str, style: &TextStyle) -> f32 {
        let Ok(text_face) = self.text.sfnt() else {
            return 0.0;
        };
        let color_face = self.color.sfnt().ok();
        let notdef = mul_fix(i64::from(text_face.advance_for_glyph(0)), self.text.scale);
        let spacing = (style.letter_spacing * 64.0) as i64;
        let mut pen: i64 = 0;
        for character in text.chars() {
            if character == '\n' {
                pen += spacing;
                continue;
            }
            let text_glyph = text_face.char_index(u32::from(character));
            let advance = if text_glyph != 0 {
                mul_fix(
                    i64::from(text_face.advance_for_glyph(text_glyph)),
                    self.text.scale,
                )
            } else if let Some(face) = color_face.as_ref() {
                let color_glyph = face.char_index(u32::from(character));
                if color_glyph != 0 {
                    mul_fix(i64::from(face.advance_for_glyph(color_glyph)), self.color.scale)
                } else {
                    notdef
                }
            } else {
                notdef
            };
            pen += advance + spacing;
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

/// An atlas for glyph bitmaps: stores packed bitmaps and returns
/// the region where each glyph was placed.
///
/// Implementations differ in the pixel format they store (e.g. single-
/// channel coverage or premultiplied BGRA).
pub trait Atlas {
    /// The pixel format of this atlas.
    fn pixel_mode(&self) -> PixelMode;

    /// The width of the atlas in pixels.
    fn width(&self) -> u32;

    /// The height of the atlas in pixels.
    fn height(&self) -> u32;

    /// The raw pixel buffer.
    fn pixels(&self) -> &[u8];

    /// Inserts a glyph bitmap and returns its region, or `None` if it
    /// does not fit.
    fn insert(&mut self, bitmap: &Bitmap) -> Option<AtlasRegion>;
}

/// A single-channel coverage atlas (gray).
///
/// This is the original `GlyphAtlas`, now implementing [`Atlas`].
impl Atlas for GlyphAtlas {
    #[inline]
    fn pixel_mode(&self) -> PixelMode {
        PixelMode::Gray
    }

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    #[inline]
    fn insert(&mut self, bitmap: &Bitmap) -> Option<AtlasRegion> {
        self.insert(bitmap)
    }
}

/// A BGRA atlas for color glyphs (emoji).
///
/// Stores premultiplied BGRA bitmaps (4 bytes per pixel), matching
/// the output of [`codevar_truetype_emoji_color::render::ColorFont`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorableGlyphAtlas {
    /// The width of the atlas, in pixels.
    width: u32,
    /// The height of the atlas, in pixels.
    height: u32,
    /// The BGRA pixels, row-major, 4 bytes per pixel.
    pixels: Vec<u8>,
    /// The x position of the next free cell.
    pen_x: u32,
    /// The y position of the next free cell.
    pen_y: u32,
    /// The height of the shelf currently being filled.
    shelf_height: u32,
}

impl ColorableGlyphAtlas {
    /// Creates a zeroed atlas of `width` by `height` pixels.
    ///
    /// Each pixel occupies 4 bytes (BGRA), so the buffer size is
    /// `width * height * 4`.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let byte_len = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        Self {
            width,
            height,
            pixels: alloc::vec![0u8; byte_len],
            pen_x: 0,
            pen_y: 0,
            shelf_height: 0,
        }
    }
}

impl Atlas for ColorableGlyphAtlas {
    #[inline]
    fn pixel_mode(&self) -> PixelMode {
        PixelMode::Bgra
    }

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    fn insert(&mut self, bitmap: &Bitmap) -> Option<AtlasRegion> {
        debug_assert_eq!(bitmap.pixel_mode, PixelMode::Bgra);
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
        let pitch = bitmap.pitch;
        let src_stride = if pitch < 0 {
            (-pitch) as usize
        } else {
            pitch as usize
        };
        for row in 0..height {
            let src_offset = row as usize * src_stride;
            if src_offset >= bitmap.buffer.len() {
                break;
            }
            let src_end = (src_offset + width as usize * 4).min(bitmap.buffer.len());
            let source = &bitmap.buffer[src_offset..src_end];
            let dst_start = ((region.y + row) * self.width + region.x) as usize * 4;
            let dst_end = dst_start + width as usize * 4;
            if let Some(destination) = self.pixels.get_mut(dst_start..dst_end) {
                let copy = (width as usize * 4).min(source.len());
                destination[..copy].copy_from_slice(&source[..copy]);
            }
        }
        self.pen_x += width + ATLAS_PADDING;
        self.shelf_height = self.shelf_height.max(height);
        Some(region)
    }
}

/// Where a glyph bitmap landed inside a [`GlyphAtlas`] or [`ColorableGlyphAtlas`].
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

/// A laid-out run of text: its glyph quads, its atlas and its
/// measured extent.
///
/// The atlas type parameter allows the run to carry either a gray
/// [`GlyphAtlas`] or a color [`ColorableGlyphAtlas`], both implementing
/// [`Atlas`].
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphRun<A: Atlas = GlyphAtlas> {
    /// The glyph quads, in reading order.
    pub quads: Vec<GlyphQuad>,
    /// The atlas the quads sample.
    pub atlas: A,
    /// The extent of the run; `width` is the advance, `height` the line
    /// height.
    pub size: Size,
    /// The distance from the top of the run to the baseline, in pixels.
    pub baseline: f32,
}

impl<A: Atlas> GlyphRun<A> {
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
        flags: RasterFlags::AA | RasterFlags::HOT,
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

    const EMOJI_FONT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../codevar-truetype-emoji-color/tests/resources/NotoColorEmoji-colr-v1-subset.ttf"
    );

    fn test_font(ppem: u16) -> Font {
        let bytes = std::fs::read(FONT).unwrap();
        Font::from_bytes(bytes).unwrap().with_ppem(ppem)
    }

    fn emoji_font(ppem: u16) -> Font {
        let bytes = std::fs::read(EMOJI_FONT).unwrap();
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
    fn shaping_a_run_fills_the_shared_atlas_and_mesh() {
        let font = test_font(24);
        let style = TextStyle::new(24.0);
        let atlas = codevar_kuitui_text::TextAtlas::new(codevar_kuitui_text::AtlasConfig::DEFAULT)
            .expect("the default atlas config is valid");
        let frame = codevar_kuitui_text::TextFrame::new(atlas).expect("valid atlas config creates frame");
        frame.begin_frame();

        let clip =
            codevar_kuitui::geometry::Rect::from_min_size(codevar_kuitui::geometry::Offset::ZERO, 64.0, 32.0);
        font.shape_into(
            &frame,
            codevar_kuitui::geometry::Offset::ZERO,
            "Hello",
            &style,
            Some(clip),
        )
        .expect("the run shapes");

        assert!(!frame.is_empty(), "the run produced geometry");
        let runs = frame.runs();
        assert_eq!(runs.len(), 1, "one page, one clip, one run");
        assert_eq!(runs[0].quad_count(), 5, "one quad per non-space glyph");
        assert_eq!(runs[0].clip, Some(clip), "the run carries the command clip");
        drop(runs);
        assert_eq!(frame.atlas().glyph_count(), 4, "the repeated 'l' shares a glyph");
        assert!(
            frame.atlas().dirty_rects().len() >= 5,
            "each placed glyph is a dirty rectangle to upload"
        );
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

    #[test]
    fn the_color_metrics_advance_a_mixed_run_with_the_face_that_draws_it() {
        let text = test_font(24);
        let emoji = emoji_font(24);
        let metrics = ColorFontMetrics::new(text.clone(), emoji.clone());
        let style = TextStyle::new(24.0);

        let mixed = metrics.advance("Hi 😀", &style);
        let expected = text.advance("Hi ", &style) + emoji.advance("😀", &style);
        assert!((mixed - expected).abs() < 0.05, "{mixed} should be {expected}");
        assert!(
            mixed > text.advance("Hi 😀", &style),
            "the emoji advances with the color face, not with the text face's notdef"
        );
    }

    #[test]
    fn the_color_metrics_take_their_line_box_from_the_text_face() {
        let text = test_font(24);
        let emoji = emoji_font(24);
        let metrics = ColorFontMetrics::new(text.clone(), emoji);
        let style = TextStyle::new(24.0);
        assert_eq!(metrics.line_height(&style), Font::line_height(&text));
        assert_eq!(metrics.ascent(&style), Font::ascent(&text));
    }
}
