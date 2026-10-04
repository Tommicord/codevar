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

//! The SFNT face facade: [`SfntFont`].
//!
//! Port of `sfobjs.c` (`sfnt_init_face`, `sfnt_load_face`) plus the
//! `loca` glue of `ttpload.c` (`tt_face_load_loca`, `tt_face_get_location`).
//! [`SfntFont::open`] selects a sub-font through
//! [`SfntContainer::open`], loads and validates the tables
//! `sfnt_load_face` requires (`head`, `maxp`, `hhea`, `hmtx`, `loca`,
//! `cmap`, `kern`) and keeps them as typed views over the input slice.
//!
//! | FreeType item | Item here |
//! |---------------|-----------|
//! | `sfnt_init_face` | [`SfntFont::open`] (via [`SfntContainer::open`]) |
//! | `sfnt_load_face` | [`SfntFont::open`] |
//! | `tt_face_load_loca` | the effective `loca` slice built by [`SfntFont::open`] |
//! | `tt_face_get_location` | [`SfntFont::glyph_location`] |
//! | `tt_face_get_metrics` | [`SfntFont::metrics_for_glyph`], [`SfntFont::advance_for_glyph`] |
//! | `FT_Get_Char_Index` | [`SfntFont::char_index`] |
//! | `find_unicode_charmap` (`ftobjs.c`) | [`SfntFont::active_charmap`] |
//! | `FT_FACE_FLAG_*` / `FT_STYLE_FLAG_*` | [`SfntFont::apply_face_info`] |
//! | `FT_CMap_New` (`ftobjs.c`) | [`SfntFont::install_charmaps`] |
//!
//! ## Deviations from FreeType 2.6
//!
//! * The `name`, `post`, `OS/2`, `gasp`, `eblc`, `pclt` and vertical
//!   metrics tables are not loaded: family/style names, underline
//!   metrics, bitmap strikes and vertical advances are left to the
//!   driver (`sfnt_load_face` loads them for the `FT_FaceRec` fields
//!   this facade does not own).
//! * The effective `loca` slice never grows past the next directory
//!   record.  `tt_face_load_loca` computes the gap to the next table
//!   but then overwrites it with the distance to the end of the stream,
//!   which would let `loca` read into the following table; this port
//!   keeps the computed gap (for a trailing `loca` both agree).
//! * `loca` is kept when present even for fonts without outlines;
//!   FreeType only loads it for scalable faces, and every location then
//!   clamps to the empty `glyf` anyway.
//! * Fonts without a Unicode charmap report `0` from
//!   [`SfntFont::char_index`] (`find_unicode_charmap` leaves
//!   `face->charmap` `NULL` in that case).
//! * `cmap` construction errors are swallowed during [`SfntFont::open`]
//!   (see the crate docs); [`parse_charmaps`] reports them standalone.

use alloc::vec::Vec;
use codevar_truetype_core::{
    Encoding, FACE_FLAG_HORIZONTAL, FACE_FLAG_KERNING, FACE_FLAG_SCALABLE, FACE_FLAG_SFNT, Face,
    STYLE_FLAG_BOLD, STYLE_FLAG_ITALIC, Tag, TtError, TtResult,
};

use crate::cmap::{OwnedCmapSubtable, SFNT_CMAP_CLASS, SfntCMap, parse_charmaps};
use crate::container::{SfntContainer, SfntDirectory};
use crate::tables::{HeadTable, HheaTable, HmtxTable, KernTable, LocaTable, MaxpTable};
use crate::tags::{
    MS_ID_UCS_4, PLATFORM_MICROSOFT, PLATFORM_UNICODE, TAG_CFF, TAG_CMAP, TAG_GLYF, TAG_HEAD, TAG_HHEA,
    TAG_HMTX, TAG_KERN, TAG_LOCA, TAG_MAXP, TAG_TRUE,
};

/// `TT_APPLE_ID_UNICODE_32` (`tttypes.h`): the `(0, 5)` platform/
/// encoding pair of an Apple UCS-4 charmap.
const APPLE_ID_UNICODE_32: u16 = 5;

/// The SFNT facade of one sub-font: the container plus every table
/// `sfnt_load_face` (`sfobjs.c`) reads.
///
/// Instances borrow the font bytes (`data`), so all table views are
/// zero-copy slices.  Create one with [`SfntFont::open`]:
///
/// ```ignore
/// let font = SfntFont::open(bytes, 0)?;
/// let glyph = font.char_index(u32::from(b'A'));
/// let (advance, lsb) = font.metrics_for_glyph(glyph);
/// ```
pub struct SfntFont<'a> {
    container: SfntContainer<'a>,
    head: HeadTable,
    maxp: MaxpTable,
    hhea: HheaTable,
    hmtx: HmtxTable<'a>,
    loca: Option<LocaTable<'a>>,
    kern: Option<KernTable<'a>>,
    charmaps: Vec<SfntCMap<'a>>,
    has_outline: bool,
}

impl<'a> SfntFont<'a> {
    /// `sfnt_init_face` + `sfnt_load_face` (`sfobjs.c`): opens
    /// sub-font `face_index` of the SFNT file in `data` and loads the
    /// tables the TrueType driver needs.
    ///
    /// # Errors
    ///
    /// * Whatever [`SfntContainer::open`] reports for the container
    ///   (truncated file, unknown version, bad face index, ...).
    /// * [`TtError::TABLE_MISSING`] — the `head` table is absent or
    ///   shorter than 54 bytes (`check_table_dir`), or `maxp` is
    ///   absent (`sfnt_load_face` overwrites `maxp`'s `Table_Missing`
    ///   with a fatal one at the end of the `LOAD_` chain).
    /// * [`TtError::INVALID_TABLE`] — `head.units_per_em` is `0`
    ///   (`sfnt_load_face`), a table cannot be parsed, or `loca` is
    ///   larger than the long/short format limit of
    ///   `tt_face_load_loca`.
    /// * [`TtError::HORIZ_HEADER_MISSING`] — `hhea` is absent (and the
    ///   file is not an SFNT Mac font, which FreeType accepts without
    ///   horizontal metrics).
    /// * [`TtError::HMTX_TABLE_MISSING`] — `hmtx` is absent while
    ///   `hhea` is present.
    /// * [`TtError::LOCATIONS_MISSING`] — `loca` is absent while the
    ///   font has outlines (`tt_face_load_loca`).
    ///
    /// A malformed `cmap` or `kern` table is ignored instead: the
    /// charmap list ends up empty and kerning reports `0`, matching
    /// the `LOAD_( cmap )` / `LOAD_( kern )` calls of
    /// `sfnt_load_face`.
    pub fn open(data: &'a [u8], face_index: i32) -> TtResult<Self> {
        let container = SfntContainer::open(data, face_index)?;

        let head_bytes = container
            .table(TAG_HEAD)
            .ok_or(TtError::TABLE_MISSING)?;
        if head_bytes.len() < 54 {
            return Err(TtError::TABLE_MISSING);
        }
        let head = HeadTable::parse(head_bytes)?;
        if head.units_per_em == 0 {
            return Err(TtError::INVALID_TABLE);
        }

        let maxp = MaxpTable::parse(
            container
                .table(TAG_MAXP)
                .ok_or(TtError::TABLE_MISSING)?,
        )?;

        let mac_true = container.sfnt_version() == TAG_TRUE;
        let hhea = match container.table(TAG_HHEA) {
            Some(bytes) => Some(HheaTable::parse(bytes)?),
            None if mac_true => None,
            None => return Err(TtError::HORIZ_HEADER_MISSING),
        };
        let hmtx = match &hhea {
            Some(hhea) => HmtxTable::new(
                container
                    .table(TAG_HMTX)
                    .ok_or(TtError::HMTX_TABLE_MISSING)?,
                hhea.number_of_h_metrics,
            ),
            None => HmtxTable::new(&[], 0),
        };
        let hhea = hhea.unwrap_or_default();

        let has_outline = container.table(TAG_GLYF).is_some() || container.table(TAG_CFF).is_some();
        let loca = match container.directory().table_range(TAG_LOCA) {
            Some((offset, length)) => {
                let bytes = effective_loca(
                    container.data(),
                    container.directory(),
                    offset,
                    length,
                    head.index_to_loc_format,
                    maxp.num_glyphs,
                )?;
                Some(LocaTable::new(bytes, head.index_to_loc_format))
            }
            None if has_outline => return Err(TtError::LOCATIONS_MISSING),
            None => None,
        };

        let charmaps = match container.table(TAG_CMAP) {
            Some(table) => parse_charmaps(table).unwrap_or_default(),
            None => Vec::new(),
        };
        let kern = container
            .table(TAG_KERN)
            .and_then(|table| KernTable::parse(table).ok());

        Ok(SfntFont {
            container,
            head,
            maxp,
            hhea,
            hmtx,
            loca,
            kern,
            charmaps,
            has_outline,
        })
    }

    /// `find_unicode_charmap` (`ftobjs.c`): the default character map —
    /// the last UCS-4 Unicode record (`(3, 10)` or `(0, 5)`), else the
    /// last Unicode record, else `None`.
    ///
    /// [`SfntFont::char_index`] maps through this record only, like
    /// `FT_Get_Char_Index` uses `face->charmap`.
    pub fn active_charmap(&self) -> Option<&SfntCMap<'a>> {
        self.charmaps
            .iter()
            .rev()
            .find(|cmap| {
                cmap.encoding == Encoding::UNICODE
                    && ((cmap.platform_id == PLATFORM_MICROSOFT && cmap.encoding_id == MS_ID_UCS_4)
                        || (cmap.platform_id == PLATFORM_UNICODE && cmap.encoding_id == APPLE_ID_UNICODE_32))
            })
            .or_else(|| {
                self.charmaps
                    .iter()
                    .rev()
                    .find(|cmap| cmap.encoding == Encoding::UNICODE)
            })
    }

    /// `FT_Get_Char_Index`: the glyph index of `char_code` in the
    /// [`Self::active_charmap`], or `0` when the code is not mapped (or
    /// the font has no Unicode charmap).
    #[inline]
    pub fn char_index(&self, char_code: u32) -> u32 {
        self.active_charmap()
            .map_or(0, |cmap| cmap.char_index(char_code))
    }

    /// `tt_face_get_metrics` (`ttmtx.c`): the `(advance, lsb)` pair of
    /// `gindex` in font units — `(0, 0)` without `hmtx` data, and the
    /// last advance with a `0` bearing once `gindex` passes the short
    /// side-bearing array (the `NoData` fallbacks of
    /// `tt_face_get_metrics`).
    #[inline]
    pub fn metrics_for_glyph(&self, gindex: u32) -> (u16, i16) {
        self.hmtx.metrics(gindex)
    }

    /// The advance width of `gindex` in font units
    /// (`tt_face_get_metrics` with the bearing discarded).
    #[inline]
    pub fn advance_for_glyph(&self, gindex: u32) -> u16 {
        self.hmtx.advance_width(gindex)
    }

    /// `tt_face_get_location` (`ttpload.c`): the byte range of glyph
    /// `gindex` inside the `glyf` table, or `None` when the glyph is
    /// empty or `gindex` is out of range.
    ///
    /// The start/end offsets come from `loca[gindex]`/`loca[gindex + 1]`
    /// (short entries doubled); an offset past the end of `glyf` is
    /// clamped, exactly like the "broken location data" handling of
    /// `tt_face_get_location`.
    pub fn glyph_location(&self, gindex: u32) -> Option<(usize, usize)> {
        if gindex >= u32::from(self.maxp.num_glyphs) {
            return None;
        }
        let loca = self.loca.as_ref()?;
        let index = gindex as usize;
        let start = loca.entry(index)? as usize;
        let end = loca.entry(index + 1)? as usize;
        let glyf_len = self
            .container
            .table(TAG_GLYF)
            .map_or(0, |table| table.len());
        if start > glyf_len {
            return None;
        }
        let end = if end > glyf_len || end < start {
            glyf_len
        } else {
            end
        };
        if end <= start {
            return None;
        }
        Some((start, end))
    }

    /// The bytes of the table named `tag`, or `None` when missing
    /// (`tt_face_goto_table` through the sanitized directory).
    #[inline]
    pub fn table(&self, tag: Tag) -> Option<&'a [u8]> {
        self.container.table(tag)
    }

    /// The table directory of the selected sub-font.
    #[inline]
    pub fn directory(&self) -> &SfntDirectory<'a> {
        self.container.directory()
    }

    /// `face->root.num_faces` (`sfnt_init_face`): the number of fonts
    /// in the file.
    #[inline]
    pub fn num_faces(&self) -> usize {
        self.container.num_faces()
    }

    /// `face->root.face_index` (`sfnt_init_face`): the index of the
    /// selected sub-font.
    #[inline]
    pub fn face_index(&self) -> i64 {
        self.container.face_index()
    }

    /// `face->root.num_glyphs` (`sfnt_load_face`): `maxp.numGlyphs`.
    #[inline]
    pub fn num_glyphs(&self) -> u16 {
        self.maxp.num_glyphs
    }

    /// `face->root.units_per_EM` (`sfnt_load_face`): `head.unitsPerEm`.
    #[inline]
    pub fn units_per_em(&self) -> u16 {
        self.head.units_per_em
    }

    /// The parsed `head` table.
    #[inline]
    pub fn head(&self) -> &HeadTable {
        &self.head
    }

    /// The parsed `maxp` table.
    #[inline]
    pub fn maxp(&self) -> &MaxpTable {
        &self.maxp
    }

    /// The parsed `hhea` table (all-zero for an SFNT Mac font without
    /// horizontal metrics).
    #[inline]
    pub fn hhea(&self) -> &HheaTable {
        &self.hhea
    }

    /// Every charmap record that validated (`tt_face_build_cmaps`), in
    /// table order.
    #[inline]
    pub fn charmaps(&self) -> &[SfntCMap<'a>] {
        &self.charmaps
    }

    /// `true` when the font carries outlines (`glyf` or `CFF `), the
    /// `has_outline` test of `sfnt_load_face`.
    #[inline]
    pub fn has_outline(&self) -> bool {
        self.has_outline
    }

    /// `true` when a usable horizontal format-0 `kern` subtable was
    /// found (`FT_FACE_FLAG_KERNING` predicate of `sfnt_load_face`).
    #[inline]
    pub fn has_kerning(&self) -> bool {
        self.kern
            .as_ref()
            .is_some_and(|kern| !kern.is_empty())
    }

    /// `tt_face_get_kerning` (`ttkern.c`): the adjustment of the
    /// `(left, right)` pair in font units (`0` without kerning data).
    #[inline]
    pub fn kerning(&self, left: u32, right: u32) -> i32 {
        self.kern
            .as_ref()
            .map_or(0, |kern| kern.kerning(left, right))
    }

    /// `sfnt_load_face`'s "set up root fields" block: copies the parsed
    /// tables into the `FT_FaceRec` fields of `face`.
    ///
    /// Sets `num_faces`/`face_index`, `num_glyphs`, `bbox`,
    /// `units_per_em`, the `hhea` ascender/descender/height and
    /// advance fields, the `FT_FACE_FLAG_*` bits (`SFNT`,
    /// `HORIZONTAL`, `SCALABLE`, `KERNING`) and the `FT_STYLE_FLAG_*`
    /// bits from `head.macStyle`.  Names, underline metrics, strikes
    /// and vertical metrics are not owned by this facade and are left
    /// untouched (see the module docs).
    pub fn apply_face_info(&self, face: &mut Face) {
        face.num_faces = self.container.num_faces() as i64;
        face.face_index = self.container.face_index();
        face.num_glyphs = i64::from(self.maxp.num_glyphs);
        face.bbox = self.head.bbox;
        face.units_per_em = self.head.units_per_em;
        face.ascender = self.hhea.ascender;
        face.descender = self.hhea.descender;
        face.height = (i32::from(self.hhea.ascender) - i32::from(self.hhea.descender)
            + i32::from(self.hhea.line_gap)) as i16;
        face.max_advance_width = self.hhea.advance_width_max as i16;
        face.max_advance_height = face.height;

        let mut flags = face.face_flags | FACE_FLAG_SFNT | FACE_FLAG_HORIZONTAL;
        if self.has_outline {
            flags |= FACE_FLAG_SCALABLE;
        }
        if face.available_sizes.is_empty() && flags & FACE_FLAG_SCALABLE == 0 {
            flags |= FACE_FLAG_SCALABLE;
        }
        if self.has_kerning() {
            flags |= FACE_FLAG_KERNING;
        }
        face.face_flags = flags;

        let mut style_flags = 0i64;
        if self.head.mac_style & 1 != 0 {
            style_flags |= STYLE_FLAG_BOLD;
        }
        if self.head.mac_style & 2 != 0 {
            style_flags |= STYLE_FLAG_ITALIC;
        }
        face.style_flags = style_flags;
    }

    /// `FT_CMap_New` (`ftobjs.c`) for every record of
    /// [`Self::charmaps`]: installs the charmaps into `face` with the
    /// format-agnostic [`SFNT_CMAP_CLASS`], copying each validated
    /// subtable into the charmap payload.
    ///
    /// # Errors
    ///
    /// Whatever [`codevar_truetype_core::Face::add_charmap`] reports
    /// (a conflicting charmap cell borrow, or an `init` failure).
    pub fn install_charmaps(&self, face: &Face) -> TtResult<()> {
        for cmap in &self.charmaps {
            let payload = OwnedCmapSubtable {
                format: cmap.subtable.format,
                flags: cmap.subtable.flags,
                data: cmap.subtable.data.to_vec(),
            };
            face.add_charmap(
                &SFNT_CMAP_CLASS,
                cmap.encoding,
                cmap.platform_id,
                cmap.encoding_id,
                Some(&payload),
            )?;
        }
        Ok(())
    }
}

/// `tt_face_load_loca` (`ttpload.c`): the effective `loca` slice of
/// `length` bytes at `offset`.
///
/// Rejects the format size limits of the C code, then applies the
/// `num_locations` adjustment: when the table holds fewer than
/// `numGlyphs + 1` entries, it is extended into the gap before the next
/// directory record (or the end of the file) if the complete table
/// fits — see the module docs for the gap/deviation note.
fn effective_loca<'a>(
    data: &'a [u8],
    directory: &SfntDirectory<'a>,
    offset: usize,
    length: usize,
    index_to_loc_format: i16,
    num_glyphs: u16,
) -> TtResult<&'a [u8]> {
    let long = index_to_loc_format != 0;
    let shift = if long { 2u32 } else { 1u32 };
    let limit = if long { 0x4_0000usize } else { 0x2_0000usize };
    if length >= limit {
        return Err(TtError::INVALID_TABLE);
    }
    let mut effective = length;
    let num_locations = length >> shift;
    let wanted = usize::from(num_glyphs) + 1;
    if num_locations != wanted && num_locations <= usize::from(num_glyphs) {
        let new_length = wanted << shift;
        let mut dist = data.len().saturating_sub(offset);
        for record in directory.records() {
            let gap = record.offset as i64 - offset as i64;
            if gap > 0 && (gap as usize) < dist {
                dist = gap as usize;
            }
        }
        if new_length <= dist {
            effective = new_length;
        }
    }
    let end = offset
        .checked_add(effective)
        .ok_or(TtError::INVALID_TABLE)?;
    data.get(offset..end)
        .ok_or(TtError::INVALID_TABLE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::{PLATFORM_MACINTOSH, TAG_MAXP};
    use crate::test_util::{
        build_font, cmap_table, default_font, default_tables, head_table, push_u16, push_u32,
    };
    use codevar_truetype_core::{BBox, FaceInternal, Generic, LibCell, Matrix, Stream, Vector, make_tag};
    use std::sync::Weak;

    fn table_with(tag: Tag, body: Vec<u8>) -> Vec<u8> {
        let tables: Vec<(Tag, Vec<u8>)> = default_tables()
            .into_iter()
            .map(|(current, bytes)| {
                if current == tag {
                    (current, body.clone())
                } else {
                    (current, bytes)
                }
            })
            .collect();
        build_font(&tables)
    }

    fn font_without(tags: &[Tag]) -> Vec<u8> {
        let tables: Vec<(Tag, Vec<u8>)> = default_tables()
            .into_iter()
            .filter(|(tag, _)| !tags.contains(tag))
            .collect();
        build_font(&tables)
    }

    fn test_face() -> Face {
        Face {
            num_faces: 0,
            face_index: 0,
            face_flags: 0,
            style_flags: 0,
            num_glyphs: 0,
            family_name: None,
            style_name: None,
            available_sizes: Vec::new(),
            charmaps: LibCell::new(Vec::new()),
            active_charmap: LibCell::new(None),
            generic: LibCell::new(Generic::default()),
            bbox: BBox::new(),
            units_per_em: 0,
            ascender: 0,
            descender: 0,
            height: 0,
            max_advance_width: 0,
            max_advance_height: 0,
            underline_position: 0,
            underline_thickness: 0,
            glyphs: LibCell::new(Vec::new()),
            active_glyph: LibCell::new(None),
            sizes: LibCell::new(Vec::new()),
            active_size: LibCell::new(None),
            autohint: LibCell::new(Generic::default()),
            driver: Weak::new(),
            stream: Stream::from_bytes(Vec::new()),
            internal: LibCell::new(FaceInternal {
                transform_matrix: Matrix::IDENTITY,
                transform_delta: Vector::ZERO,
                transform_flags: 0,
                ignore_unpatented_hinter: false,
            }),
            driver_data: None,
            this: Weak::new(),
        }
    }

    #[test]
    fn open_parses_every_loaded_table() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        assert_eq!(font.num_faces(), 1);
        assert_eq!(font.face_index(), 0);
        assert_eq!(font.num_glyphs(), 5);
        assert_eq!(font.units_per_em(), 1000);
        assert_eq!(font.directory().num_tables(), 8);
        assert_eq!(font.table(TAG_MAXP).map(|table| table.len()), Some(32));
        assert!(
            font.table(make_tag(b'n', b'o', b'p', b'e'))
                .is_none()
        );
        assert!(font.head().has_valid_magic());
        assert!(!font.head().uses_long_loca());
        assert_eq!(font.maxp().version, 0x0001_0000);
        assert_eq!(font.hhea().number_of_h_metrics, 3);
        assert_eq!(font.hhea().ascender, 800);
        assert_eq!(font.hhea().advance_width_max, 700);
        assert!(font.has_outline());
        assert_eq!(font.charmaps().len(), 2);
        assert_eq!(font.charmaps()[0].platform_id, 3);
        assert_eq!(font.charmaps()[0].encoding_id, 1);
        assert_eq!(font.charmaps()[0].subtable.format, 4);
        assert_eq!(font.charmaps()[1].platform_id, 3);
        assert_eq!(font.charmaps()[1].encoding_id, 10);
        assert_eq!(font.charmaps()[1].subtable.format, 12);
    }

    #[test]
    fn char_index_uses_the_default_unicode_charmap() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        let active = font.active_charmap().unwrap();
        assert_eq!(active.platform_id, 3);
        assert_eq!(active.encoding_id, 10);
        assert_eq!(font.char_index(u32::from(b'A')), 1);
        assert_eq!(font.char_index(u32::from(b'B')), 2);
        assert_eq!(font.char_index(u32::from(b'Z')), 0);
        assert_eq!(font.char_index(0x1F600), 0);
        assert_eq!(font.char_index(0), 0);
    }

    #[test]
    fn charmaps_without_unicode_select_no_default() {
        let mut cmap = Vec::new();
        push_u16(&mut cmap, 0);
        push_u16(&mut cmap, 1);
        push_u16(&mut cmap, PLATFORM_MACINTOSH);
        push_u16(&mut cmap, 0);
        push_u32(&mut cmap, 12);
        cmap.extend_from_slice(&crate::test_util::cmap_format4());
        let data = table_with(crate::tags::TAG_CMAP, cmap);
        let font = SfntFont::open(&data, 0).unwrap();
        assert_eq!(font.charmaps().len(), 1);
        assert_eq!(font.charmaps()[0].encoding, Encoding::APPLE_ROMAN);
        assert!(font.active_charmap().is_none());
        assert_eq!(font.char_index(u32::from(b'A')), 0);
    }

    #[test]
    fn metrics_and_advances_follow_hmtx() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        assert_eq!(font.metrics_for_glyph(0), (500, -10));
        assert_eq!(font.metrics_for_glyph(1), (600, 20));
        assert_eq!(font.metrics_for_glyph(2), (700, -30));
        assert_eq!(font.metrics_for_glyph(3), (700, 40));
        assert_eq!(font.metrics_for_glyph(4), (700, -50));
        assert_eq!(font.metrics_for_glyph(5), (700, 0));
        assert_eq!(font.advance_for_glyph(0), 500);
        assert_eq!(font.advance_for_glyph(4), 700);
        assert_eq!(font.advance_for_glyph(99), 700);
        assert_eq!(font.metrics_for_glyph(99), (700, 0));
    }

    #[test]
    fn glyph_locations_follow_loca() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        assert_eq!(font.glyph_location(0), None);
        assert_eq!(font.glyph_location(1), Some((0, 10)));
        assert_eq!(font.glyph_location(2), Some((10, 30)));
        assert_eq!(font.glyph_location(3), Some((30, 46)));
        assert_eq!(font.glyph_location(4), Some((46, 60)));
        assert_eq!(font.glyph_location(5), None);
        assert_eq!(font.glyph_location(6), None);
        assert_eq!(font.glyph_location(u32::MAX), None);
    }

    #[test]
    fn kerning_reaches_the_caller() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        assert!(font.has_kerning());
        assert_eq!(font.kerning(1, 2), -40);
        assert_eq!(font.kerning(3, 4), 25);
        assert_eq!(font.kerning(2, 1), 0);
    }

    #[test]
    fn truncated_loca_record_is_extended_into_the_gap() {
        let mut font = default_font();
        let num_tables = u16::from_be_bytes([font[4], font[5]]) as usize;
        for index in 0..num_tables {
            let record = 12 + index * 16;
            if &font[record..record + 4] == b"loca" {
                font[record + 12..record + 16].copy_from_slice(&4u32.to_be_bytes());
            }
        }
        let font = SfntFont::open(&font, 0).unwrap();
        assert_eq!(font.glyph_location(0), None);
        assert_eq!(font.glyph_location(1), Some((0, 10)));
        assert_eq!(font.glyph_location(4), Some((46, 60)));
    }

    #[test]
    fn loca_extension_stops_before_the_next_table() {
        let mut short_loca = Vec::new();
        push_u16(&mut short_loca, 0);
        push_u16(&mut short_loca, 5);
        let tables: Vec<(Tag, Vec<u8>)> = default_tables()
            .into_iter()
            .map(|(tag, body)| {
                if tag == crate::tags::TAG_LOCA {
                    (tag, short_loca.clone())
                } else {
                    (tag, body)
                }
            })
            .collect();
        let data = build_font(&tables);
        let font = SfntFont::open(&data, 0).unwrap();
        assert_eq!(font.glyph_location(0), Some((0, 10)));
        assert_eq!(font.glyph_location(1), None);
        assert_eq!(font.glyph_location(4), None);
    }

    #[test]
    fn missing_essential_tables_are_reported() {
        let missing_head = font_without(&[TAG_HEAD]);
        assert!(matches!(
            SfntFont::open(&missing_head, 0),
            Err(TtError::TABLE_MISSING)
        ));

        let missing_maxp = font_without(&[TAG_MAXP]);
        assert!(matches!(
            SfntFont::open(&missing_maxp, 0),
            Err(TtError::TABLE_MISSING)
        ));

        let missing_hhea = font_without(&[TAG_HHEA]);
        assert!(matches!(
            SfntFont::open(&missing_hhea, 0),
            Err(TtError::HORIZ_HEADER_MISSING)
        ));

        let missing_hmtx = font_without(&[TAG_HMTX]);
        assert!(matches!(
            SfntFont::open(&missing_hmtx, 0),
            Err(TtError::HMTX_TABLE_MISSING)
        ));

        let missing_loca = font_without(&[TAG_LOCA]);
        assert!(matches!(
            SfntFont::open(&missing_loca, 0),
            Err(TtError::LOCATIONS_MISSING)
        ));
    }

    #[test]
    fn short_head_and_zero_units_per_em_are_rejected() {
        let mut short_head = head_table();
        short_head.pop();
        assert_eq!(short_head.len(), 53);
        let short = table_with(TAG_HEAD, short_head);
        assert!(matches!(SfntFont::open(&short, 0), Err(TtError::TABLE_MISSING)));

        let mut zero_upem = head_table();
        zero_upem[18..20].copy_from_slice(&0u16.to_be_bytes());
        let zero = table_with(TAG_HEAD, zero_upem);
        assert!(matches!(SfntFont::open(&zero, 0), Err(TtError::INVALID_TABLE)));
    }

    #[test]
    fn oversized_loca_is_invalid() {
        let mut long_head = head_table();
        long_head[50..52].copy_from_slice(&1i16.to_be_bytes());
        let mut tables: Vec<(Tag, Vec<u8>)> = default_tables()
            .into_iter()
            .map(|(tag, body)| {
                if tag == TAG_HEAD {
                    (tag, long_head.clone())
                } else {
                    (tag, body)
                }
            })
            .collect();
        let huge = vec![0u8; 0x4_0000];
        for entry in tables.iter_mut() {
            if entry.0 == crate::tags::TAG_LOCA {
                entry.1 = huge.clone();
            }
        }
        let font = build_font(&tables);
        assert!(matches!(SfntFont::open(&font, 0), Err(TtError::INVALID_TABLE)));
    }

    #[test]
    fn broken_cmap_and_kern_are_swallowed() {
        let mut cmap = cmap_table();
        cmap[0..2].copy_from_slice(&1u16.to_be_bytes());
        let data = table_with(crate::tags::TAG_CMAP, cmap);
        let font = SfntFont::open(&data, 0).unwrap();
        assert!(font.charmaps().is_empty());
        assert!(font.active_charmap().is_none());
        assert_eq!(font.char_index(u32::from(b'A')), 0);

        let kern_data = table_with(TAG_KERN, vec![0u8; 2]);
        let kern = SfntFont::open(&kern_data, 0).unwrap();
        assert!(!kern.has_kerning());
        assert_eq!(kern.kerning(1, 2), 0);
    }

    #[test]
    fn container_errors_propagate() {
        let data = default_font();
        assert!(matches!(
            SfntFont::open(&data[..2], 0),
            Err(TtError::INVALID_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntFont::open(&data[..11], 0),
            Err(TtError::INVALID_TABLE)
        ));
        let mut bad_version = data.clone();
        bad_version[0..4].copy_from_slice(b"XXXX");
        assert!(matches!(
            SfntFont::open(&bad_version, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert!(matches!(SfntFont::open(&data, 4), Err(TtError::INVALID_ARGUMENT)));
    }

    #[test]
    fn sfnt_mac_font_without_hhea_is_accepted() {
        let tables: Vec<(Tag, Vec<u8>)> = default_tables()
            .into_iter()
            .filter(|(tag, _)| *tag != TAG_HHEA && *tag != TAG_HMTX)
            .collect();
        let mut font = build_font(&tables);
        font[0..4].copy_from_slice(b"true");
        let font = SfntFont::open(&font, 0).unwrap();
        assert_eq!(font.hhea().number_of_h_metrics, 0);
        assert_eq!(font.metrics_for_glyph(1), (0, 0));
        assert_eq!(font.advance_for_glyph(1), 0);
    }

    #[test]
    fn apply_face_info_fills_the_face() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        let mut face = test_face();
        font.apply_face_info(&mut face);
        assert_eq!(face.num_faces, 1);
        assert_eq!(face.face_index, 0);
        assert_eq!(face.num_glyphs, 5);
        assert_eq!(face.units_per_em, 1000);
        assert_eq!(face.bbox, BBox::from_edges(-100, -200, 900, 800));
        assert_eq!(face.ascender, 800);
        assert_eq!(face.descender, -200);
        assert_eq!(face.height, 1090);
        assert_eq!(face.max_advance_width, 700);
        assert_eq!(face.max_advance_height, 1090);
        assert!(face.is_sfnt());
        assert!(face.has_horizontal());
        assert!(face.has_kerning());
        assert!(face.is_scalable());
        assert_eq!(face.style_flags, STYLE_FLAG_BOLD);
        assert_eq!(face.family_name, None);
        assert_eq!(face.underline_position, 0);
    }

    #[test]
    fn install_charmaps_populates_the_face() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        let face = test_face();
        font.install_charmaps(&face).unwrap();
        let charmaps = face.charmaps.borrow().unwrap();
        assert_eq!(charmaps.len(), 2);
        let first = &charmaps[0];
        assert_eq!(first.platform_id, 3);
        assert_eq!(first.encoding_id, 1);
        assert_eq!(first.encoding, Encoding::UNICODE);
        assert_eq!(first.char_index(u32::from(b'A')), 1);
        assert_eq!(first.char_index(u32::from(b'Z')), 0);
        assert_eq!(charmaps[1].char_index(u32::from(b'B')), 2);
        drop(charmaps);
        assert_eq!(face.num_charmaps().unwrap(), 2);
        font.install_charmaps(&face).unwrap();
        assert_eq!(face.num_charmaps().unwrap(), 4);
    }

    #[test]
    fn apply_face_info_keeps_existing_driver_flags() {
        let data = default_font();
        let font = SfntFont::open(&data, 0).unwrap();
        let mut face = test_face();
        face.face_flags = codevar_truetype_core::FACE_FLAG_HINTER;
        font.apply_face_info(&mut face);
        assert!(face.has_hinter());
        assert!(face.is_sfnt());
        assert!(face.has_kerning());
    }
}
