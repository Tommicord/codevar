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

//! The atlas of the rasterized glyphs of a frame.
//!
//! Kuitui does not rasterize glyphs: it measures the text with the
//! [`FontMetrics`](codevar_kuitui::FontMetrics) of the theme and hands
//! the measured runs to the backend. The backend needs the pixels of
//! the glyphs, and this atlas holds them.
//!
//! An atlas is a shelf packer: the glyphs are laid out in rows, and a
//! new row is started when the next glyph does not fit in the current
//! one. The coverage of every glyph is stored as one byte per pixel,
//! which is what a fragment shader needs to blend a glyph.
//!
//! The atlas reaches the GPU as an [`ATLAS_FORMAT`] texture: the
//! coverage of a glyph is a single channel, so one byte per pixel is
//! enough. One pixel of empty space separates two glyphs, so that the
//! linear filtering of the sampler cannot bleed the coverage of one
//! glyph into the next one.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use std::collections::HashMap;

use crate::error::WgError;

/// The format of the texture of the atlas.
///
/// The coverage of a glyph is a single channel, so the atlas is an
/// `R8Unorm` texture with one byte per pixel.
pub const ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The number of empty pixels between two glyphs of the atlas.
///
/// The gap keeps the linear filtering of the sampler from bleeding the
/// coverage of a glyph into its neighbour.
pub const GLYPH_GAP: u32 = 1;

/// The identity of a glyph inside the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// The character the glyph draws.
    pub character: char,
    /// The size of the em square the glyph was rasterized at.
    pub size: u16,
    /// The index of the glyph inside the atlas.
    pub index: u16,
}

/// A rasterized glyph.
///
/// The coverage is stored in the glyph as well as in the atlas, so a
/// caller can keep the pixels of a glyph it drew itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Glyph {
    /// The identity of the glyph.
    pub key: GlyphKey,
    /// The width of the coverage bitmap.
    pub width: u32,
    /// The height of the coverage bitmap.
    pub height: u32,
    /// The horizontal distance from the origin of the text to the left
    /// edge of the glyph.
    pub bearing_x: i32,
    /// The vertical distance from the baseline to the top edge of the
    /// glyph.
    pub bearing_y: i32,
    /// The horizontal advance of the glyph.
    pub advance: f32,
    /// The coverage of the glyph, one byte per pixel.
    pub coverage: Vec<u8>,
    /// The x of the top-left corner of the glyph inside the atlas.
    pub atlas_x: u32,
    /// The y of the top-left corner of the glyph inside the atlas.
    pub atlas_y: u32,
}

impl Glyph {
    /// The number of pixels the glyph covers.
    #[inline]
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// A glyph that is about to be added to an atlas.
///
/// A draft holds the pixels of a glyph and the metrics of the text it
/// belongs to; the atlas packs it and gives it its place.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphDraft {
    /// The character the glyph draws.
    pub character: char,
    /// The size of the em square the glyph was rasterized at.
    pub size: u16,
    /// The horizontal distance from the origin of the text to the left
    /// edge of the glyph.
    pub bearing_x: i32,
    /// The vertical distance from the baseline to the top edge of the
    /// glyph.
    pub bearing_y: i32,
    /// The horizontal advance of the glyph.
    pub advance: f32,
    /// The width of the coverage bitmap.
    pub width: u32,
    /// The height of the coverage bitmap.
    pub height: u32,
    /// The coverage of the glyph, one byte per pixel.
    pub coverage: Vec<u8>,
}

impl GlyphDraft {
    /// Creates a draft of a glyph of `width` by `height` pixels.
    #[must_use]
    pub const fn new(character: char, size: u16, width: u32, height: u32) -> Self {
        Self {
            character,
            size,
            bearing_x: 0,
            bearing_y: 0,
            advance: 0.0,
            width,
            height,
            coverage: Vec::new(),
        }
    }

    /// The draft with the bearings of the glyph.
    #[must_use]
    pub const fn with_bearings(mut self, bearing_x: i32, bearing_y: i32) -> Self {
        self.bearing_x = bearing_x;
        self.bearing_y = bearing_y;
        self
    }

    /// The draft with the advance of the glyph.
    #[must_use]
    pub const fn with_advance(mut self, advance: f32) -> Self {
        self.advance = advance;
        self
    }

    /// The draft with the coverage of the glyph.
    #[must_use]
    pub fn with_coverage(mut self, coverage: Vec<u8>) -> Self {
        self.coverage = coverage;
        self
    }
}

/// An atlas of rasterized glyphs.
///
/// The atlas owns the coverage of every glyph it holds and packs it
/// into a single texture. The atlas has a fixed size: a glyph that
/// does not fit is refused, and the caller starts a new atlas.
#[derive(Debug, Clone)]
pub struct GlyphAtlas {
    /// The glyphs of the atlas.
    glyphs: Vec<Glyph>,
    /// The coverage of the whole atlas, one byte per pixel.
    coverage: Vec<u8>,
    /// The width of the atlas, in pixels.
    width: u32,
    /// The height of the atlas, in pixels.
    height: u32,
    /// The x of the next free column of the current row.
    cursor_x: u32,
    /// The y of the current row.
    cursor_y: u32,
    /// The height of the current row.
    row_height: u32,
    /// The index of the glyph of every character and size.
    lookup: HashMap<(char, u16), u16>,
    /// The number of glyphs the atlas has taken, used to know when the
    /// texture of the atlas has to be uploaded again.
    revision: u32,
}

impl GlyphAtlas {
    /// Creates an empty atlas of `width` by `height` pixels.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let pixels = width as usize * height as usize;
        Self {
            glyphs: Vec::new(),
            coverage: vec![0; pixels],
            width,
            height,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            lookup: HashMap::new(),
            revision: 0,
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

    /// The number of glyphs the atlas holds.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    /// Whether the atlas holds no glyph.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// The glyphs of the atlas.
    #[inline]
    #[must_use]
    pub fn glyphs(&self) -> &[Glyph] {
        &self.glyphs
    }

    /// The coverage of the whole atlas, one byte per pixel.
    #[inline]
    #[must_use]
    pub fn coverage(&self) -> &[u8] {
        &self.coverage
    }

    /// The number of times the content of the atlas changed.
    ///
    /// The renderer uses it to know when the texture of the atlas has
    /// to be uploaded again.
    #[inline]
    #[must_use]
    pub const fn revision(&self) -> u32 {
        self.revision
    }

    /// The glyph the key points at.
    #[must_use]
    pub fn glyph(&self, key: &GlyphKey) -> Option<&Glyph> {
        self.glyphs.get(key.index as usize)
    }

    /// The glyph of a character at a size, if the atlas holds it.
    #[must_use]
    pub fn lookup(&self, character: char, size: u16) -> Option<&Glyph> {
        let index = *self.lookup.get(&(character, size))?;
        self.glyphs.get(index as usize)
    }

    /// Whether a glyph of `width` by `height` pixels still fits.
    ///
    /// A glyph fits when it fits in the current row, or when a new row
    /// can be started for it.
    #[must_use]
    pub fn fits(&self, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return true;
        }
        if width > self.width {
            return false;
        }
        if self.row_height > 0
            && height <= self.row_height
            && self.cursor_x + width <= self.width
        {
            return true;
        }
        self.next_row_y() + height <= self.height
    }

    /// Whether the atlas is full.
    ///
    /// The atlas is full when not even the smallest glyph fits any
    /// more.
    #[must_use]
    pub fn is_full(&self) -> bool {
        !self.fits(1, 1)
    }

    /// The texture coordinates of a glyph inside the atlas.
    ///
    /// The coordinates are inset by half a texel, so that the linear
    /// filtering of the sampler reads the texels of the glyph and
    /// nothing else.
    #[must_use]
    pub fn uv_rect(&self, glyph: &Glyph) -> [f32; 4] {
        if glyph.width == 0 || glyph.height == 0 || self.width == 0 || self.height == 0 {
            return [0.0; 4];
        }
        let width = self.width as f32;
        let height = self.height as f32;
        [
            (glyph.atlas_x as f32 + 0.5) / width,
            (glyph.atlas_y as f32 + 0.5) / height,
            (glyph.atlas_x as f32 + glyph.width as f32 - 0.5) / width,
            (glyph.atlas_y as f32 + glyph.height as f32 - 0.5) / height,
        ]
    }

    /// A description of the atlas, for logs.
    #[must_use]
    pub fn describe(&self) -> String {
        alloc::format!(
            "{} glyphs in a {}x{} atlas",
            self.glyphs.len(),
            self.width,
            self.height
        )
    }

    /// Adds a glyph to the atlas.
    ///
    /// The glyph is packed into the first free spot of the atlas and
    /// its coverage is copied into the texture of the atlas. A glyph
    /// that is already in the atlas is not added twice: its key is
    /// returned instead.
    ///
    /// # Errors
    ///
    /// Returns a [`WgError`] when the atlas is full, when the glyph
    /// does not fit in it or when the atlas holds too many glyphs.
    pub fn insert(&mut self, draft: GlyphDraft) -> Result<GlyphKey, WgError> {
        let character = draft.character;
        let size = draft.size;
        if let Some(glyph) = self.lookup(character, size) {
            return Ok(glyph.key);
        }
        let width = draft.width;
        let height = draft.height;
        if !self.fits(width, height) {
            return Err(WgError::Atlas(String::from(
                "the glyph does not fit in the atlas",
            )));
        }
        let index = u16::try_from(self.glyphs.len())
            .map_err(|_| WgError::Atlas(String::from("the atlas holds too many glyphs")))?;
        let (x, y) = self.reserve(width, height);
        self.copy_coverage(x, y, &draft.coverage, width, height);
        let key = GlyphKey {
            character,
            size,
            index,
        };
        self.glyphs.push(Glyph {
            key,
            width,
            height,
            bearing_x: draft.bearing_x,
            bearing_y: draft.bearing_y,
            advance: draft.advance,
            coverage: draft.coverage,
            atlas_x: x,
            atlas_y: y,
        });
        self.lookup.insert((character, size), index);
        self.revision = self.revision.wrapping_add(1);
        Ok(key)
    }

    /// The y of the row that follows the current one.
    fn next_row_y(&self) -> u32 {
        if self.row_height == 0 {
            self.cursor_y
        } else {
            self.cursor_y + self.row_height + GLYPH_GAP
        }
    }

    /// Reserves the spot of a glyph and returns its top-left corner.
    fn reserve(&mut self, width: u32, height: u32) -> (u32, u32) {
        if width == 0 || height == 0 {
            return (self.cursor_x, self.cursor_y);
        }
        let fits_in_row = self.row_height > 0
            && height <= self.row_height
            && self.cursor_x + width <= self.width;
        if !fits_in_row && self.row_height > 0 {
            self.cursor_y = self.next_row_y();
            self.cursor_x = 0;
            self.row_height = 0;
        }
        let x = self.cursor_x;
        let y = self.cursor_y;
        self.cursor_x += width + GLYPH_GAP;
        self.row_height = self.row_height.max(height);
        (x, y)
    }

    /// Copies the coverage of a glyph into the coverage of the atlas.
    fn copy_coverage(&mut self, x: u32, y: u32, coverage: &[u8], width: u32, height: u32) {
        if width == 0 || height == 0 || self.width == 0 {
            return;
        }
        let row = width as usize;
        let stride = self.width as usize;
        for line in 0..height as usize {
            let source = line * row;
            let target = (y as usize + line) * stride + x as usize;
            let available = coverage.len().saturating_sub(source).min(row);
            if available == 0 {
                break;
            }
            self.coverage[target..target + available].copy_from_slice(&coverage[source..source + available]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A coverage bitmap of `width` by `height` filled with `value`.
    fn bitmap(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; width as usize * height as usize]
    }

    #[test]
    fn an_empty_atlas_holds_no_glyph() {
        let atlas = GlyphAtlas::new(64, 64);
        assert!(atlas.is_empty());
        assert_eq!(atlas.len(), 0);
        assert_eq!(atlas.coverage().len(), 64 * 64);
        assert!(!atlas.is_full());
    }

    #[test]
    fn a_glyph_is_packed_at_the_origin_of_the_atlas() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let key = atlas
            .insert(
                GlyphDraft::new('a', 16, 4, 6)
                    .with_bearings(1, 12)
                    .with_advance(8.0)
                    .with_coverage(bitmap(4, 6, 255)),
            )
            .unwrap_or(GlyphKey {
                character: 'a',
                size: 16,
                index: 0,
            });
        assert_eq!(key.index, 0);
        let glyph = &atlas.glyphs()[0];
        assert_eq!(glyph.atlas_x, 0);
        assert_eq!(glyph.atlas_y, 0);
        assert_eq!(glyph.pixel_count(), 24);
        assert!(atlas.glyph(&key).is_some());
    }

    #[test]
    fn the_coverage_of_a_glyph_reaches_the_atlas() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let _ = atlas.insert(
            GlyphDraft::new('a', 16, 2, 2)
                .with_advance(8.0)
                .with_coverage(bitmap(2, 2, 200)),
        );
        let glyph = &atlas.glyphs()[0];
        assert_eq!(glyph.coverage, vec![200, 200, 200, 200]);
        // The coverage of the glyph sits at its spot in the atlas, so
        // the second row of the glyph is a whole row of the atlas
        // further down.
        assert_eq!(atlas.coverage()[0], 200);
        assert_eq!(atlas.coverage()[1], 200);
        assert_eq!(atlas.coverage()[2], 0);
        assert_eq!(atlas.coverage()[64], 200);
        assert_eq!(atlas.coverage()[65], 200);
    }

    #[test]
    fn the_glyphs_of_an_atlas_do_not_overlap() {
        let mut atlas = GlyphAtlas::new(32, 32);
        for index in 0..8 {
            let character = char::from(b'a' + index as u8);
            let _ = atlas.insert(
                GlyphDraft::new(character, 16, 6, 6)
                    .with_advance(8.0)
                    .with_coverage(bitmap(6, 6, 128)),
            );
        }
        let glyphs = atlas.glyphs();
        for (index, first) in glyphs.iter().enumerate() {
            for second in glyphs.iter().skip(index + 1) {
                let separated = first.atlas_x + first.width + GLYPH_GAP <= second.atlas_x
                    || second.atlas_x + second.width + GLYPH_GAP <= first.atlas_x
                    || first.atlas_y + first.height + GLYPH_GAP <= second.atlas_y
                    || second.atlas_y + second.height + GLYPH_GAP <= first.atlas_y;
                assert!(separated, "the glyphs {index} and the next one overlap");
            }
        }
    }

    #[test]
    fn a_full_atlas_refuses_a_new_glyph() {
        let mut atlas = GlyphAtlas::new(8, 8);
        let _ = atlas.insert(
            GlyphDraft::new('a', 16, 8, 8)
                .with_advance(8.0)
                .with_coverage(bitmap(8, 8, 255)),
        );
        assert!(atlas.is_full());
        let refused = atlas.insert(
            GlyphDraft::new('b', 16, 8, 8)
                .with_advance(8.0)
                .with_coverage(bitmap(8, 8, 255)),
        );
        assert!(refused.is_err());
        assert_eq!(atlas.len(), 1);
    }

    #[test]
    fn a_glyph_is_looked_up_by_character_and_size() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let _ = atlas.insert(
            GlyphDraft::new('a', 16, 4, 4)
                .with_advance(8.0)
                .with_coverage(bitmap(4, 4, 255)),
        );
        assert!(atlas.lookup('a', 16).is_some());
        assert!(atlas.lookup('a', 12).is_none());
        assert!(atlas.lookup('b', 16).is_none());
    }

    #[test]
    fn a_glyph_is_not_added_twice() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let draft = || {
            GlyphDraft::new('a', 16, 4, 4)
                .with_advance(8.0)
                .with_coverage(bitmap(4, 4, 255))
        };
        let first = atlas.insert(draft());
        let second = atlas.insert(draft());
        assert_eq!(first, second);
        assert_eq!(atlas.len(), 1);
    }

    #[test]
    fn the_revision_of_the_atlas_grows_with_every_glyph() {
        let mut atlas = GlyphAtlas::new(64, 64);
        assert_eq!(atlas.revision(), 0);
        let _ = atlas.insert(
            GlyphDraft::new('a', 16, 4, 4)
                .with_advance(8.0)
                .with_coverage(bitmap(4, 4, 255)),
        );
        assert_eq!(atlas.revision(), 1);
    }

    #[test]
    fn the_texture_coordinates_of_a_glyph_stay_inside_the_atlas() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let _ = atlas.insert(
            GlyphDraft::new('a', 16, 4, 4)
                .with_advance(8.0)
                .with_coverage(bitmap(4, 4, 255)),
        );
        let glyph = &atlas.glyphs()[0];
        let uv = atlas.uv_rect(glyph);
        assert!(uv[0] >= 0.0 && uv[1] >= 0.0);
        assert!(uv[2] <= 1.0 && uv[3] <= 1.0);
        assert!(uv[2] > uv[0] && uv[3] > uv[1]);
    }
}

