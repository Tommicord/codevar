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
//! 5. hand the coverage to the shared atlas the backend owns.
//!
//! [`Font::shape_into`] runs the whole loop for a run of text and pushes
//! the quads into a [`TextFrame`] from `codevar-kuitui-text`, the same
//! atlas and mesh the `text` shader draws. The font never owns an atlas:
//! caching, eviction and upload belong to the renderer, so a run that is
//! reshaped every frame only rasterizes the glyphs that are not resident
//! yet.

use alloc::sync::Arc;
use alloc::vec::Vec;

use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::{Offset, Rect};
use codevar_kuitui::text::{FontMetrics, TextStyle};
use codevar_kuitui_text::{GlyphKey, GlyphQuad, TextError, TextFrame};
use codevar_truetype_core::{
    BBox, Bitmap, Generic, GlyphFormat, GlyphLoader, GlyphMetrics, GlyphSlot, Library, LoadFlags, Matrix,
    Outline, PixelMode, Raster, RasterFlags, RasterParams, SizeMetrics, SlotInternal, TtError, Vector,
    div_fix, mul_fix,
};
use codevar_truetype_emoji_color::render::{ColorFont, RenderedGlyph};
use codevar_truetype_glyf::load_glyph;
use codevar_truetype_sfnt::SfntFont;
use codevar_truetype_smooth::grays::GrayRaster;

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

    /// Lays `text` out into a shared [`TextFrame`].
    ///
    /// It rasterizes each glyph that the frame's atlas does not hold yet,
    /// inserts its coverage into that shared atlas and pushes the quad that
    /// samples the slot the atlas returned. A backend that owns the `text`
    /// pipeline drains the same frame, so the CPU stays the only source of
    /// coverage and the GPU is never read back.
    ///
    /// `origin` is the baseline origin of the run and `clip` is the clip the
    /// run was recorded with. The raster size is the font's own ppem, not
    /// `style.font_size`, so measurement and pixels agree.
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
                if glyph_index != 0 {
                    let key = glyph_key(self.ppem, glyph_index);
                    // The atlas answers before the raster is asked. A glyph
                    // already resident has its coverage in a page that will not
                    // move, so reshaping a run that repeats it needs only the
                    // control box that placed it — never the GPU again. Without
                    // this order every frame of a scroll re-rasterizes every
                    // glyph it draws, which is a round trip per glyph.
                    if let Some(resident) = atlas.get(key) {
                        if let Some(position) =
                            resident_position(&font, &mut slot, &metrics, glyph_index, pen, origin)
                                .map_err(ShapeError::Font)?
                        {
                            mesh.push(&GlyphQuad::from_slot(resident, position, color, clip));
                        }
                    } else if let Some(bitmap) =
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
                        if let Some(atlased) = atlas
                            .insert(key, width, height, &coverage)
                            .map_err(ShapeError::Atlas)?
                        {
                            let position = [
                                origin.x + fixed_pixels(pen) + bitmap.left as f32,
                                origin.y - bitmap.top as f32,
                            ];
                            mesh.push(&GlyphQuad::from_slot(atlased, position, color, clip));
                        }
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
                                text_mesh.push(&GlyphQuad::from_slot(slot, position, text_color, clip));
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
                    mesh.push(&GlyphQuad::from_slot(slot, position, color, clip));
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

/// Where a glyph whose coverage is already resident in the atlas is drawn.
///
/// The outline is loaded for the control box — the same box [`rasterize_glyph`]
/// cropped its bitmap to, so a cached glyph lands exactly where a freshly
/// rasterized one would — and the raster is never reached. `None` is a glyph
/// with no outline, which is drawn as nothing and only moves the pen.
fn resident_position(
    font: &SfntFont<'_>,
    slot: &mut GlyphSlot,
    metrics: &SizeMetrics,
    glyph_index: u32,
    pen: i64,
    origin: Offset,
) -> Result<Option<[f32; 2]>, FontError> {
    load_glyph(font, slot, metrics, glyph_index, LoadFlags::DEFAULT).map_err(FontError::Load)?;
    if slot.outline.is_empty() {
        return Ok(None);
    }
    let cbox = slot.outline.get_cbox();
    let left = cbox.x_min >> 6;
    let top = cbox.y_max.saturating_add(63) >> 6;
    Ok(Some([
        origin.x + fixed_pixels(pen) + left as f32,
        origin.y - top as f32,
    ]))
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
        "/../../assets/ttf/space-mono.regular.ttf"
    );

    const EMOJI_FONT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../truetype-emoji-color/tests/resources/NotoColorEmoji-colr-v1-subset.ttf"
    );

    fn test_font(ppem: u16) -> Font {
        let bytes = std::fs::read(FONT).unwrap();
        Font::from_bytes(bytes).unwrap().with_ppem(ppem)
    }

    fn emoji_font(ppem: u16) -> Font {
        let bytes = std::fs::read(EMOJI_FONT).unwrap();
        Font::from_bytes(bytes).unwrap().with_ppem(ppem)
    }

    fn test_frame() -> TextFrame {
        let atlas = codevar_kuitui_text::TextAtlas::new(codevar_kuitui_text::AtlasConfig::DEFAULT)
            .expect("the default atlas config is valid");
        TextFrame::new(atlas).expect("valid atlas config creates frame")
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
    fn a_space_has_an_advance_but_produces_no_geometry() {
        let font = test_font(16);
        let style = TextStyle::new(16.0);
        assert!(font.advance(" ", &style) > 0.0);
        let frame = test_frame();
        frame.begin_frame();
        font.shape_into(&frame, Offset::ZERO, " ", &style, None)
            .expect("a space shapes to nothing");
        assert!(frame.is_empty(), "a space has no glyph to pack");
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
    fn reshaping_a_run_reuses_the_resident_glyphs() {
        let font = test_font(24);
        let style = TextStyle::new(24.0);
        let frame = test_frame();

        frame.begin_frame();
        font.shape_into(&frame, Offset::ZERO, "Hello", &style, None)
            .expect("the first shape rasterizes every glyph");
        let first = frame.atlas().dirty_rects().len();
        assert!(first > 0, "the first shape uploads coverage");

        frame.begin_frame();
        font.shape_into(&frame, Offset::ZERO, "Hello", &style, None)
            .expect("the second shape hits the cache");
        assert!(
            frame.atlas().dirty_rects().len() == first,
            "no new coverage is written for glyphs that are already resident"
        );
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
