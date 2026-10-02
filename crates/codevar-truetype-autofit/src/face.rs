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
//!
//! On top of that trait [`FaceGlobals`] ports `AF_FaceGlobalsRec`
//! (`afglobal.c`): the per-glyph style map and the lazily created
//! per-style metrics of one face.

use alloc::rc::Rc;
use alloc::vec::Vec;

use codevar_truetype_core::{Matrix, Outline, Pos, TtError, TtResult, Vector};

use crate::metrics::{GlobalsShared, StyleMetrics};
use crate::ranges::{Coverage, DIGIT, SCRIPT_CLASSES, STYLE_CLASSES, STYLE_MAX, STYLE_UNASSIGNED, Style};

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

/// Module wide auto-hinter properties consumed by [`FaceGlobals::new`]
/// (`AF_ModuleRec`, `afmodule.c`).
///
/// `af_autofitter_init` installs the [`Default`] values; the
/// `warping` and `fallback-script` properties update them before a
/// face's globals are created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalsConfig {
    /// `AF_ModuleRec::warping` (`warping` property): enables the
    /// warper in light hinting mode.
    pub warping: bool,
    /// `AF_ModuleRec::fallback_style` (`fallback-script` property):
    /// style of every glyph no charmap covers; `None` corresponds to
    /// `AF_STYLE_UNASSIGNED` and disables the fallback.
    pub fallback_style: Option<Style>,
}

impl Default for GlobalsConfig {
    /// The values `af_autofitter_init` (`afmodule.c`) writes into a
    /// fresh `AF_ModuleRec`.
    #[inline]
    fn default() -> GlobalsConfig {
        GlobalsConfig {
            warping: false,
            fallback_style: Some(Style::HaniDflt),
        }
    }
}

/// `AF_FaceGlobalsRec` (`afglobal.h`): the global hinting values of
/// one face, computed once by `af_face_globals_new`.
///
/// The record owns the per-glyph style map (`glyph_styles[]`), the
/// per-style metrics (`metrics[]`, created on first use) and the face
/// wide shared properties ([`GlobalsShared`]).  The face itself is
/// not stored: FreeType only keeps `AF_FaceGlobalsRec::face` as a
/// borrowed pointer, so every entry point that needs it takes the
/// face as an argument, mirroring the separate `face` and `globals`
/// members of `AF_LoaderRec`.
#[derive(Debug)]
pub struct FaceGlobals {
    /// `AF_FaceGlobalsRec::glyph_styles`: style index (and [`DIGIT`]
    /// flag) of every glyph of the face.
    glyph_styles: Vec<u8>,
    /// `AF_FaceGlobalsRec::metrics`: per-style metrics, initialized
    /// on first access.
    metrics: [Option<StyleMetrics>; STYLE_MAX],
    /// Face wide shared properties (`AF_FaceGlobalsRec` fields read
    /// in C through `metrics->globals`).
    shared: Rc<GlobalsShared>,
}

impl FaceGlobals {
    /// `af_face_globals_new` (`afglobal.c`): computes the style map
    /// of `face` and prepares the shared face properties.
    ///
    /// Nothing here can fail port-side: a face without a Unicode
    /// charmap simply leaves every glyph on the fallback style, since
    /// the C code ignores the charmap selection error as well.
    pub fn new(face: &mut dyn Face, config: GlobalsConfig) -> FaceGlobals {
        let mut glyph_styles = alloc::vec![STYLE_UNASSIGNED; face.num_glyphs()];
        compute_style_coverage(face, &mut glyph_styles, config.fallback_style);

        let shared = Rc::new(GlobalsShared::new(face.units_per_em()));
        shared.set_warping(config.warping);

        FaceGlobals {
            glyph_styles,
            metrics: core::array::from_fn(|_| None),
            shared,
        }
    }

    /// `af_face_globals_get_metrics` (`afglobal.c`): returns the
    /// metrics of `gindex`, creating and initializing them on first
    /// use.
    ///
    /// `forced` plays the role of the C `options` argument: a style
    /// of [`Style::NoneDflt`] or one whose successor reaches
    /// [`STYLE_MAX`] looks the glyph's own style up in
    /// `glyph_styles[]`, exactly like `AF_STYLE_NONE_DFLT` and
    /// `style + 1 >= AF_STYLE_MAX` do in FreeType.  The face is only
    /// needed while the metrics are initialized, so the returned
    /// metrics no longer borrow it.
    ///
    /// Errors: `Invalid_Argument` if `gindex` is out of range or the
    /// glyph carries no known style (the latter only happens with the
    /// fallback style disabled); otherwise the writing system's
    /// `style_metrics_init` failed.
    pub fn get_metrics(
        &mut self,
        face: &mut dyn Face,
        gindex: u32,
        forced: Option<Style>,
    ) -> TtResult<&mut StyleMetrics> {
        let index = gindex as usize;
        if index >= self.glyph_styles.len() {
            return Err(TtError::INVALID_ARGUMENT);
        }

        let style_index = match forced {
            Some(style) if style != Style::NoneDflt && style.index() + 1 < STYLE_MAX => style.index(),
            _ => (self.glyph_styles[index] & STYLE_UNASSIGNED) as usize,
        };
        if style_index >= STYLE_MAX {
            return Err(TtError::INVALID_ARGUMENT);
        }

        if self.metrics[style_index].is_none() {
            let mut metrics = StyleMetrics::new(&STYLE_CLASSES[style_index], self.shared.clone());
            metrics_init(&mut metrics, face)?;
            self.metrics[style_index] = Some(metrics);
        }

        // The slot was just filled above when it was empty.
        self.metrics[style_index]
            .as_mut()
            .ok_or(TtError::INVALID_ARGUMENT)
    }

    /// `af_face_globals_is_digit` (`afglobal.c`): true when `gindex`
    /// is an ASCII digit glyph; out of range indices are never
    /// digits.
    #[inline]
    pub fn is_digit(&self, gindex: u32) -> bool {
        self.glyph_styles
            .get(gindex as usize)
            .is_some_and(|style| style & DIGIT != 0)
    }

    /// `AF_FaceGlobalsRec::glyph_styles` (exported by the
    /// `glyph-to-script-map` property of `afmodule.c`): the style
    /// index of every glyph, flagged with [`DIGIT`] for the ASCII
    /// digits.
    #[inline]
    pub fn glyph_styles(&self) -> &[u8] {
        &self.glyph_styles
    }

    /// The face wide shared properties (`AF_FaceGlobalsRec` via
    /// `metrics->globals`): `units_per_EM` plus the `warping` and
    /// `increase-x-height` cells.
    #[inline]
    pub fn shared(&self) -> &GlobalsShared {
        &self.shared
    }
}

/// `af_face_globals_compute_style_coverage` (`afglobal.c`): assigns
/// every glyph of `face` to the first style whose Unicode ranges
/// cover it, marks the ASCII digits and finally applies the fallback
/// style to everything left uncovered.
///
/// The `af_get_coverage` phases of the C function (`hbshim.c`, OpenType
/// features through HarfBuzz) are no-ops without HarfBuzz, both for
/// the non-default coverages skipped below and for the two extra
/// passes over the default styles that FreeType runs before the digit
/// scan.
fn compute_style_coverage(face: &mut dyn Face, glyph_styles: &mut [u8], fallback_style: Option<Style>) {
    glyph_styles.fill(STYLE_UNASSIGNED);

    let old_charmap = face.charmap();
    if face.select_charmap(Encoding::Unicode).is_ok() {
        for (ss, style_class) in STYLE_CLASSES.iter().enumerate() {
            let script_class = &SCRIPT_CLASSES[style_class.script.index()];
            if script_class.ranges.is_empty() {
                continue;
            }
            if style_class.coverage != Coverage::Default {
                continue;
            }

            for range in script_class.ranges {
                let mut charcode = range.first;
                let mut gindex = face.char_index(charcode);
                mark_style(glyph_styles, gindex, ss as u8);

                loop {
                    let (next_charcode, next_gindex) = face.next_char(charcode);
                    charcode = next_charcode;
                    gindex = next_gindex;
                    if gindex == 0 || charcode > range.last {
                        break;
                    }
                    mark_style(glyph_styles, gindex, ss as u8);
                }
            }
        }

        // mark ASCII digits (digit `0` is 0x30 in all charmaps)
        for charcode in 0x30..=0x39 {
            let gindex = face.char_index(charcode);
            if gindex != 0
                && let Some(style) = glyph_styles.get_mut(gindex as usize)
            {
                *style |= DIGIT;
            }
        }
    }

    // By default all uncovered glyphs get the fallback style; a
    // disabled fallback leaves them marked `AF_STYLE_UNASSIGNED`.
    if let Some(fallback) = fallback_style {
        let fallback = fallback.index() as u8;
        for style in glyph_styles.iter_mut() {
            if *style & !DIGIT == STYLE_UNASSIGNED {
                *style = (*style & !STYLE_UNASSIGNED) | fallback;
            }
        }
    }

    // `FT_Set_Charmap(face, old_charmap)` restores the charmap that
    // was selected before the Unicode scan; C ignores its error.
    let _ = face.set_charmap(old_charmap);
}

/// Records `style` for `gindex` while the glyph is still uncovered
/// (`gindex == 0` means "not mapped" and glyph indices beyond the
/// face are ignored).
#[inline]
fn mark_style(glyph_styles: &mut [u8], gindex: u32, style: u8) {
    if gindex != 0
        && let Some(slot) = glyph_styles.get_mut(gindex as usize)
        && *slot == STYLE_UNASSIGNED
    {
        *slot = style;
    }
}

/// Dispatches `writing_system_class->style_metrics_init`
/// (`af_face_globals_get_metrics` in `afglobal.c`); the dummy writing
/// system has no such entry in `afdummy.c`.  The face is consumed by
/// the Latin initializer (charmap switching and glyph lookups); the
/// CJK initializer is ported together with `afcjk.c`.
fn metrics_init(metrics: &mut StyleMetrics, face: &mut dyn Face) -> TtResult<()> {
    match metrics {
        StyleMetrics::Dummy(_) => Ok(()),
        StyleMetrics::Latin(latin) => crate::latin::metrics_init(latin, face),
        StyleMetrics::Cjk(_) => {
            // `af_cjk_metrics_init` arrives with the `afcjk.c`
            // section of the port.
            Err(TtError::UNIMPLEMENTED_FEATURE)
        }
    }
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

    /// A face with seven glyphs: five covered by Latin, Cyrillic and
    /// Devanagari ranges, two uncovered, plus a charmap entry whose
    /// glyph index lies beyond the face.
    fn coverage_face() -> MockFace {
        let mut face = MockFace::new()
            .with_charmap(
                Encoding::Unicode,
                &[
                    (0x30, 3),   // `0`  -> Latin digit
                    (0x41, 1),   // `A`  -> Latin
                    (0x42, 2),   // `B`  -> Latin
                    (0x44, 100), // `D`  -> beyond `num_glyphs`
                    (0x43E, 4),  // `о`  -> Cyrillic
                    (0x0915, 6), // `क` -> Devanagari
                ],
            )
            .with_charmap(Encoding::AppleRoman, &[(0x41, 1)])
            .select(Encoding::AppleRoman);
        for gindex in 0..=6 {
            face = face.with_glyph(gindex, GlyphSlot::default());
        }
        face
    }

    #[test]
    fn default_module_properties_match_af_autofitter_init() {
        let config = GlobalsConfig::default();
        assert!(!config.warping, "AF_ModuleRec::warping starts at 0");
        assert_eq!(
            config.fallback_style,
            Some(Style::HaniDflt),
            "AF_STYLE_FALLBACK is AF_STYLE_HANI_DFLT"
        );
        assert_eq!(
            Style::HaniDflt.index(),
            crate::ranges::STYLE_FALLBACK,
            "the module default equals AF_STYLE_FALLBACK"
        );
    }

    #[test]
    fn new_seeds_the_warping_property_into_the_shared_cells() {
        let mut face = coverage_face();
        let config = GlobalsConfig {
            warping: true,
            ..GlobalsConfig::default()
        };
        let globals = FaceGlobals::new(&mut face, config);

        assert!(globals.shared().warping());
        assert_eq!(globals.shared().units_per_em, face.units_per_em());
        assert_eq!(globals.shared().increase_x_height(), 0);
    }

    #[test]
    fn coverage_assigns_each_glyph_to_the_first_matching_style() {
        let mut face = coverage_face();
        let globals = FaceGlobals::new(&mut face, GlobalsConfig::default());

        //                    no cmap   A     B    `0`    о    no cmap  क
        assert_eq!(globals.glyph_styles(), [48, 31, 31, 159, 10, 48, 32]);
        assert_eq!(
            face.charmap(),
            Some(Encoding::AppleRoman),
            "the charmap selected before the scan is restored"
        );
    }

    #[test]
    fn coverage_marks_the_ascii_digit_glyphs() {
        let mut face = coverage_face();
        let globals = FaceGlobals::new(&mut face, GlobalsConfig::default());

        assert!(globals.is_digit(3));
        assert!(!globals.is_digit(1));
        assert!(!globals.is_digit(0));
        assert!(!globals.is_digit(7), "out of range is never a digit");
        assert_eq!(globals.glyph_styles()[3], 31 | DIGIT);
    }

    #[test]
    fn coverage_without_a_unicode_charmap_only_applies_the_fallback() {
        let mut face = MockFace::new()
            .with_charmap(Encoding::AppleRoman, &[(0x30, 1), (0x41, 2)])
            .select(Encoding::AppleRoman);
        for gindex in 0..=2 {
            face = face.with_glyph(gindex, GlyphSlot::default());
        }

        let globals = FaceGlobals::new(&mut face, GlobalsConfig::default());

        assert_eq!(globals.glyph_styles(), [48, 48, 48]);
        assert!(
            !globals.is_digit(1),
            "the digit scan never runs without a Unicode charmap"
        );
        assert_eq!(face.charmap(), Some(Encoding::AppleRoman));
    }

    #[test]
    fn an_empty_face_has_no_styles_at_all() {
        let mut face = MockFace::new();
        let mut globals = FaceGlobals::new(&mut face, GlobalsConfig::default());

        assert!(globals.glyph_styles().is_empty());
        assert!(!globals.is_digit(0));
        assert!(matches!(
            globals.get_metrics(&mut face, 0, None),
            Err(TtError::INVALID_ARGUMENT)
        ));
    }

    #[test]
    fn get_metrics_creates_the_metrics_of_the_fallback_style() {
        let mut face = coverage_face();
        let config = GlobalsConfig {
            fallback_style: Some(Style::NoneDflt),
            ..GlobalsConfig::default()
        };
        let mut globals = FaceGlobals::new(&mut face, config);

        let metrics = globals
            .get_metrics(&mut face, 0, None)
            .expect("the dummy writing system needs no initialization");
        assert_eq!(metrics.style(), Style::NoneDflt);
        assert!(matches!(metrics, StyleMetrics::Dummy(_)));

        let again = globals
            .get_metrics(&mut face, 0, None)
            .expect("the metrics are cached after the first call");
        assert_eq!(again.style(), Style::NoneDflt);

        // `AF_STYLE_HANI_DFLT + 1 >= AF_STYLE_MAX`, so the forced
        // style falls back to the glyph's own (dummy) style.
        let looked_up = globals
            .get_metrics(&mut face, 0, Some(Style::HaniDflt))
            .expect("the lookup still resolves to the dummy metrics");
        assert_eq!(looked_up.style(), Style::NoneDflt);
    }

    #[test]
    fn get_metrics_rejects_invalid_glyph_indices() {
        let mut face = coverage_face();
        let mut globals = FaceGlobals::new(&mut face, GlobalsConfig::default());

        assert!(matches!(
            globals.get_metrics(&mut face, 7, None),
            Err(TtError::INVALID_ARGUMENT)
        ));
        assert!(matches!(
            globals.get_metrics(&mut face, u32::MAX, None),
            Err(TtError::INVALID_ARGUMENT)
        ));
    }

    #[test]
    fn get_metrics_rejects_a_glyph_without_a_known_style() {
        let mut face = coverage_face();
        let config = GlobalsConfig {
            fallback_style: None,
            ..GlobalsConfig::default()
        };
        let mut globals = FaceGlobals::new(&mut face, config);

        assert_eq!(globals.glyph_styles()[0], STYLE_UNASSIGNED);
        assert!(matches!(
            globals.get_metrics(&mut face, 0, None),
            Err(TtError::INVALID_ARGUMENT)
        ));
    }
}
