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

//! The face abstraction the auto-hinter needs (`FT_Face` subset).
//!
//! FreeType hands the auto-hinter a full `FT_Face`, but only a handful
//! of its services are ever used: the face dimensions, charmap
//! selection and iteration, glyph loading in font units and the
//! advance widths.  [`Face`] captures exactly that surface so the
//! hinter stays independent of any particular font backend.

use codevar_truetype_core::{Matrix, Outline, Pos, TtResult, Vector};

/// `FT_STYLE_FLAG_ITALIC` (`ftimage.h`): the face is italic.
pub const STYLE_FLAG_ITALIC: u32 = 1;
/// `FT_STYLE_FLAG_BOLD` (`ftimage.h`): the face is bold.
pub const STYLE_FLAG_BOLD: u32 = 2;

/// `FT_Encoding` (`ftimage.h`): the charmap encodings the auto-hinter
/// selects while scanning a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// `FT_ENCODING_UNICODE`.
    Unicode,
    /// `FT_ENCODING_APPLE_ROMAN`.
    AppleRoman,
    /// `FT_ENCODING_ADOBE_STANDARD`.
    AdobeStandard,
    /// `FT_ENCODING_ADOBE_LATIN_1`.
    AdobeLatin1,
}

/// `FT_Glyph_Metrics` (`ftimage.h`): the scalable metrics of one
/// glyph, in font units while the hinter runs with `FT_LOAD_NO_SCALE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlyphMetrics {
    /// Horizontal advance width.
    pub hori_advance: Pos,
    /// Left side bearing, horizontal layout.
    pub hori_bearing_x: Pos,
    /// Top side bearing, horizontal layout.
    pub hori_bearing_y: Pos,
    /// Vertical advance height.
    pub vert_advance: Pos,
    /// Left side bearing, vertical layout.
    pub vert_bearing_x: Pos,
    /// Top side bearing, vertical layout.
    pub vert_bearing_y: Pos,
    /// Width of the black box, horizontal layout.
    pub width: Pos,
    /// Height of the black box, horizontal layout.
    pub height: Pos,
}

/// `FT_GlyphSlotRec` (`ftimage.h`) as read and written by
/// `af_loader_load_glyph` (afloader.c).
///
/// FreeType only ever hints outline glyphs, so
/// [`Face::load_glyph`] reports [`TtError::UNIMPLEMENTED_FEATURE`]
/// for any other format instead of carrying a format tag.
#[derive(Clone, Debug, Default)]
pub struct GlyphSlot {
    /// The glyph outline, already loaded in font units.
    pub outline: Outline,
    /// The glyph metrics in font units.
    pub metrics: GlyphMetrics,
    /// Horizontal shift of the left side bearing (`slot->lsb_delta`),
    /// written back by the hinter.
    pub lsb_delta: Pos,
    /// Horizontal shift of the right side bearing (`slot->rsb_delta`),
    /// written back by the hinter.
    pub rsb_delta: Pos,
    /// True when the loader applied a transformation
    /// (`internal->glyph_transformed`).
    pub transformed: bool,
    /// The transformation applied by the loader (`glyph_matrix`).
    pub matrix: Matrix,
    /// The translation applied by the loader (`glyph_delta`).
    pub delta: Vector,
}

/// The services the auto-hinter requires from a font face.
///
/// Every method mirrors one `FT_Face` access of FreeType's
/// `afglobal.c`, `aflatin.c`, `afcjk.c` and `afloader.c`.  The trait is
/// object safe so the writing system dispatch table can accept
/// `&mut dyn Face`.
pub trait Face {
    /// `face->units_per_EM`.
    fn units_per_em(&self) -> u16;

    /// `face->num_glyphs`.
    fn num_glyphs(&self) -> usize;

    /// `face->size->metrics.x_ppem`.
    fn x_ppem(&self) -> u32;

    /// `face->style_flags`.
    fn style_flags(&self) -> u32;

    /// `FT_IS_FIXED_WIDTH(face)`.
    fn is_fixed_width(&self) -> bool;

    /// The currently selected charmap (`face->charmap`), `None` when
    /// the face has none selected.
    fn charmap(&self) -> Option<Encoding>;

    /// `FT_Set_Charmap(face, charmap)`: selects a previously captured
    /// charmap.  Passing `None` restores the "no charmap selected"
    /// state of a face that had none.
    fn set_charmap(&mut self, charmap: Option<Encoding>) -> TtResult<()>;

    /// `FT_Select_Charmap(face, encoding)`: selects the charmap with
    /// the given encoding, leaving the selection untouched when the
    /// face has no such charmap.
    fn select_charmap(&mut self, encoding: Encoding) -> TtResult<()>;

    /// `FT_Get_Char_Index(face, charcode)`: maps a character code
    /// through the selected charmap, returning `0` when unmapped.
    fn char_index(&self, charcode: u32) -> u32;

    /// `FT_Get_Next_Char(face, charcode, &gindex)`: the first
    /// `(charcode, gindex)` pair strictly after `charcode`.
    ///
    /// Returns `(0, 0)` when the iteration is exhausted, matching the
    /// `gindex == 0` termination condition of `afglobal.c`.
    fn next_char(&self, after: u32) -> (u32, u32);

    /// `FT_Get_Advance(face, gindex, FT_LOAD_NO_SCALE | ...)`: the
    /// horizontal advance of `gindex` in font units.
    fn advance(&self, gindex: u32) -> TtResult<Pos>;

    /// `FT_Load_Glyph(face, gindex, FT_LOAD_NO_SCALE)`: loads the
    /// outline and metrics of `gindex` in font units.
    ///
    /// Returns [`TtError::UNIMPLEMENTED_FEATURE`] when the glyph is
    /// not an outline, exactly as `af_loader_load_g` does for foreign
    /// glyph formats.
    fn load_glyph(&self, gindex: u32) -> TtResult<GlyphSlot>;
}

#[cfg(test)]
pub(crate) use mock::MockFace;

#[cfg(test)]
mod mock {
    use codevar_truetype_core::TtError;

    use super::*;

    /// A minimal in-memory [`Face`] used by the hinter's tests.
    ///
    /// The charmap tables must be sorted by character code so that
    /// [`Face::next_char`] can walk them the way FreeType walks a
    /// Unicode cmap.
    pub(crate) struct MockFace {
        units_per_em: u16,
        x_ppem: u32,
        style_flags: u32,
        fixed_width: bool,
        charmap: Option<Encoding>,
        charmaps: Vec<(Encoding, Vec<(u32, u32)>)>,
        glyphs: Vec<GlyphSlot>,
        advances: Vec<Pos>,
    }

    impl MockFace {
        /// An empty face with FreeType's default `units_per_EM`.
        pub(crate) fn new() -> MockFace {
            MockFace {
                units_per_em: 2048,
                x_ppem: 0,
                style_flags: 0,
                fixed_width: false,
                charmap: None,
                charmaps: Vec::new(),
                glyphs: Vec::new(),
                advances: Vec::new(),
            }
        }

        /// Sets `face->units_per_EM`.
        pub(crate) fn with_units_per_em(mut self, upem: u16) -> MockFace {
            self.units_per_em = upem;
            self
        }

        /// Sets `face->size->metrics.x_ppem`.
        pub(crate) fn with_x_ppem(mut self, ppem: u32) -> MockFace {
            self.x_ppem = ppem;
            self
        }

        /// Sets `face->style_flags`.
        pub(crate) fn with_style_flags(mut self, flags: u32) -> MockFace {
            self.style_flags = flags;
            self
        }

        /// Marks the face as `FT_IS_FIXED_WIDTH`.
        pub(crate) fn with_fixed_width(mut self, fixed: bool) -> MockFace {
            self.fixed_width = fixed;
            self
        }

        /// Adds a charmap of `encoding` mapping `table`, sorted by
        /// character code.
        pub(crate) fn with_charmap(mut self, encoding: Encoding, table: &[(u32, u32)]) -> MockFace {
            self.charmaps.push((encoding, table.to_vec()));
            self
        }

        /// Selects `encoding` as the initial charmap.
        pub(crate) fn select(mut self, encoding: Encoding) -> MockFace {
            self.charmap = Some(encoding);
            self
        }

        /// Registers the outline of glyph `gindex`, growing the face
        /// as needed.
        pub(crate) fn with_glyph(mut self, gindex: usize, slot: GlyphSlot) -> MockFace {
            if self.glyphs.len() <= gindex {
                self.glyphs
                    .resize_with(gindex + 1, GlyphSlot::default);
            }
            if self.advances.len() <= gindex {
                self.advances.resize(gindex + 1, 0);
            }
            self.advances[gindex] = slot.metrics.hori_advance;
            self.glyphs[gindex] = slot;
            self
        }
    }

    impl Face for MockFace {
        fn units_per_em(&self) -> u16 {
            self.units_per_em
        }

        fn num_glyphs(&self) -> usize {
            self.glyphs.len()
        }

        fn x_ppem(&self) -> u32 {
            self.x_ppem
        }

        fn style_flags(&self) -> u32 {
            self.style_flags
        }

        fn is_fixed_width(&self) -> bool {
            self.fixed_width
        }

        fn charmap(&self) -> Option<Encoding> {
            self.charmap
        }

        fn set_charmap(&mut self, charmap: Option<Encoding>) -> TtResult<()> {
            self.charmap = charmap;
            Ok(())
        }

        fn select_charmap(&mut self, encoding: Encoding) -> TtResult<()> {
            if self.charmaps.iter().any(|(e, _)| *e == encoding) {
                self.charmap = Some(encoding);
                return Ok(());
            }
            Err(TtError::INVALID_ARGUMENT)
        }

        fn char_index(&self, charcode: u32) -> u32 {
            let Some(encoding) = self.charmap else {
                return 0;
            };
            for (e, table) in &self.charmaps {
                if *e == encoding {
                    return table
                        .iter()
                        .find(|(c, _)| *c == charcode)
                        .map_or(0, |(_, g)| *g);
                }
            }
            0
        }

        fn next_char(&self, after: u32) -> (u32, u32) {
            let Some(encoding) = self.charmap else {
                return (0, 0);
            };
            for (e, table) in &self.charmaps {
                if *e == encoding {
                    return table
                        .iter()
                        .find(|(c, _)| *c > after)
                        .map_or((0, 0), |(c, g)| (*c, *g));
                }
            }
            (0, 0)
        }

        fn advance(&self, gindex: u32) -> TtResult<Pos> {
            self.advances
                .get(gindex as usize)
                .copied()
                .ok_or(TtError::INVALID_GLYPH_INDEX)
        }

        fn load_glyph(&self, gindex: u32) -> TtResult<GlyphSlot> {
            self.glyphs
                .get(gindex as usize)
                .cloned()
                .ok_or(TtError::INVALID_GLYPH_INDEX)
        }
    }
}

#[cfg(test)]
mod tests {
    use codevar_truetype_core::TtError;

    use super::*;

    fn rectangle(x0: i64, y0: i64, x1: i64, y1: i64) -> GlyphSlot {
        let mut outline = Outline::with_capacity(4, 1);
        outline.points = vec![
            Vector::new(x0, y0),
            Vector::new(x1, y0),
            Vector::new(x1, y1),
            Vector::new(x0, y1),
        ];
        outline.tags = vec![0, 0, 0, 0];
        outline.contours = vec![3];
        outline.n_points = 4;
        outline.n_contours = 1;
        GlyphSlot {
            outline,
            metrics: GlyphMetrics {
                hori_advance: x1 - x0,
                ..GlyphMetrics::default()
            },
            ..GlyphSlot::default()
        }
    }

    #[test]
    fn encoding_values_match_free_types() {
        assert_eq!(STYLE_FLAG_ITALIC, 1);
        assert_eq!(STYLE_FLAG_BOLD, 2);
    }

    #[test]
    fn mock_face_maps_codes_through_the_selected_charmap() {
        let face = MockFace::new()
            .with_charmap(Encoding::Unicode, &[(0x41, 1), (0x42, 2), (0x7A, 3)])
            .with_charmap(Encoding::AppleRoman, &[(0x41, 7)])
            .select(Encoding::Unicode);

        assert_eq!(face.char_index(0x41), 1);
        assert_eq!(face.char_index(0x42), 2);
        assert_eq!(face.char_index(0x43), 0);
        assert_eq!(face.next_char(0), (0x41, 1));
        assert_eq!(face.next_char(0x41), (0x42, 2));
        assert_eq!(face.next_char(0x42), (0x7A, 3));
        assert_eq!(face.next_char(0x7A), (0, 0));

        let mut face = face;
        assert!(face.select_charmap(Encoding::AppleRoman).is_ok());
        assert_eq!(face.char_index(0x41), 7);
        assert!(
            face.select_charmap(Encoding::AdobeLatin1)
                .is_err()
        );
        assert_eq!(face.charmap(), Some(Encoding::AppleRoman));

        assert!(face.set_charmap(Some(Encoding::Unicode)).is_ok());
        assert_eq!(face.char_index(0x41), 1);
        assert!(face.set_charmap(None).is_ok());
        assert_eq!(face.charmap(), None);
        assert_eq!(face.char_index(0x41), 0);
        assert_eq!(face.next_char(0), (0, 0));
    }

    #[test]
    fn mock_face_loads_registered_glyphs_only() {
        let face = MockFace::new().with_glyph(1, rectangle(0, 0, 500, 700));

        assert_eq!(face.num_glyphs(), 2);
        assert_eq!(face.advance(1), Ok(500));
        assert_eq!(face.advance(2), Err(TtError::INVALID_GLYPH_INDEX));

        let slot = face.load_glyph(1).expect("glyph exists");
        assert_eq!(slot.outline.n_points, 4);
        assert_eq!(slot.metrics.hori_advance, 500);
        assert!(face.load_glyph(9).is_err());
    }

    #[test]
    fn face_defaults_are_reported_back() {
        let face = MockFace::new();
        assert_eq!(face.units_per_em(), 2048);
        assert_eq!(face.x_ppem(), 0);
        assert_eq!(face.style_flags(), 0);
        assert!(!face.is_fixed_width());
        assert_eq!(face.charmap(), None);

        let face = MockFace::new()
            .with_units_per_em(1000)
            .with_x_ppem(16)
            .with_style_flags(STYLE_FLAG_ITALIC)
            .with_fixed_width(true);
        assert_eq!(face.units_per_em(), 1000);
        assert_eq!(face.x_ppem(), 16);
        assert_eq!(face.style_flags(), STYLE_FLAG_ITALIC);
        assert!(face.is_fixed_width());
    }
}
