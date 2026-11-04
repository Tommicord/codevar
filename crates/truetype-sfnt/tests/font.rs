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

//! SFNT facade tests driven through the crate's public API only.

extern crate alloc;

use alloc::sync::Weak;
use codevar_truetype_core::{
    BBox, Encoding, Face, FaceInternal, Generic, LibCell, Matrix, STYLE_FLAG_BOLD, Stream, TtError, Vector,
    make_tag,
};
use codevar_truetype_sfnt::SfntFont;
use codevar_truetype_sfnt::{container, tags};

#[allow(dead_code)]
#[path = "../src/test_util.rs"]
mod test_util;

use test_util::{build_font, default_font, default_tables, head_table};

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

fn font_without(removed: &[codevar_truetype_core::Tag]) -> Vec<u8> {
    let tables: Vec<(codevar_truetype_core::Tag, Vec<u8>)> = default_tables()
        .into_iter()
        .filter(|(tag, _)| !removed.contains(tag))
        .collect();
    build_font(&tables)
}

#[test]
fn loads_every_facade_view_of_the_default_font() {
    let data = default_font();
    let font = SfntFont::open(&data, 0).unwrap();
    assert_eq!(font.num_faces(), 1);
    assert_eq!(font.face_index(), 0);
    assert_eq!(font.num_glyphs(), 5);
    assert_eq!(font.units_per_em(), 1000);
    assert_eq!(font.directory().num_tables(), 8);
    assert!(font.has_outline());
    assert!(font.has_kerning());
    assert!(font.head().has_valid_magic());
    assert_eq!(font.maxp().version, 0x0001_0000);
    assert_eq!(font.hhea().number_of_h_metrics, 3);
    assert_eq!(font.hhea().ascender, 800);
    assert_eq!(font.hhea().descender, -200);
    assert_eq!(font.hhea().advance_width_max, 700);
    assert_eq!(
        font.table(tags::TAG_MAXP)
            .map(|table| table.len()),
        Some(32)
    );
    assert!(
        font.table(make_tag(b'n', b'o', b'p', b'e'))
            .is_none()
    );
}

#[test]
fn maps_characters_through_the_default_charmap() {
    let data = default_font();
    let font = SfntFont::open(&data, 0).unwrap();
    let active = font.active_charmap().unwrap();
    assert_eq!(active.platform_id, 3);
    assert_eq!(active.encoding_id, 10);
    assert_eq!(active.encoding, Encoding::UNICODE);
    assert_eq!(font.charmaps().len(), 2);
    assert_eq!(font.char_index(u32::from(b'A')), 1);
    assert_eq!(font.char_index(u32::from(b'B')), 2);
    assert_eq!(font.char_index(u32::from(b'Z')), 0);
    assert_eq!(font.char_index(0x1F600), 0);
}

#[test]
fn reports_metrics_locations_and_kerning() {
    let data = default_font();
    let font = SfntFont::open(&data, 0).unwrap();
    assert_eq!(font.metrics_for_glyph(0), (500, -10));
    assert_eq!(font.metrics_for_glyph(2), (700, -30));
    assert_eq!(font.metrics_for_glyph(4), (700, -50));
    assert_eq!(font.advance_for_glyph(1), 600);

    assert_eq!(font.glyph_location(0), None);
    assert_eq!(font.glyph_location(1), Some((0, 10)));
    assert_eq!(font.glyph_location(4), Some((46, 60)));
    assert_eq!(font.glyph_location(5), None);

    assert_eq!(font.kerning(1, 2), -40);
    assert_eq!(font.kerning(3, 4), 25);
    assert_eq!(font.kerning(4, 1), 0);
}

#[test]
fn fills_a_face_and_installs_its_charmaps() {
    let data = default_font();
    let font = SfntFont::open(&data, 0).unwrap();
    let mut face = test_face();
    font.apply_face_info(&mut face);
    font.install_charmaps(&face).unwrap();

    assert_eq!(face.num_faces, 1);
    assert_eq!(face.face_index, 0);
    assert_eq!(face.num_glyphs, 5);
    assert_eq!(face.units_per_em, 1000);
    assert_eq!(face.bbox, BBox::from_edges(-100, -200, 900, 800));
    assert_eq!(face.ascender, 800);
    assert_eq!(face.descender, -200);
    assert_eq!(face.height, 1090);
    assert_eq!(face.max_advance_width, 700);
    assert!(face.is_sfnt());
    assert!(face.has_horizontal());
    assert!(face.has_kerning());
    assert!(face.is_scalable());
    assert_eq!(face.style_flags, STYLE_FLAG_BOLD);
    assert_eq!(face.family_name, None);

    assert_eq!(face.num_charmaps().unwrap(), 2);
    let charmaps = face.charmaps.borrow().unwrap();
    assert_eq!(charmaps[0].platform_id, 3);
    assert_eq!(charmaps[0].encoding, Encoding::UNICODE);
    assert_eq!(charmaps[0].char_index(u32::from(b'A')), 1);
    assert_eq!(charmaps[1].char_index(u32::from(b'B')), 2);
}

#[test]
fn reports_missing_essential_tables() {
    assert!(matches!(
        SfntFont::open(&font_without(&[tags::TAG_HEAD]), 0),
        Err(TtError::TABLE_MISSING)
    ));
    assert!(matches!(
        SfntFont::open(&font_without(&[tags::TAG_MAXP]), 0),
        Err(TtError::TABLE_MISSING)
    ));
    assert!(matches!(
        SfntFont::open(&font_without(&[tags::TAG_HHEA]), 0),
        Err(TtError::HORIZ_HEADER_MISSING)
    ));
    assert!(matches!(
        SfntFont::open(&font_without(&[tags::TAG_HMTX]), 0),
        Err(TtError::HMTX_TABLE_MISSING)
    ));
    assert!(matches!(
        SfntFont::open(&font_without(&[tags::TAG_LOCA]), 0),
        Err(TtError::LOCATIONS_MISSING)
    ));
}

#[test]
fn rejects_broken_files() {
    let data = default_font();
    assert!(matches!(
        SfntFont::open(&data[..2], 0),
        Err(TtError::INVALID_FILE_FORMAT)
    ));
    assert!(matches!(SfntFont::open(&data, 4), Err(TtError::INVALID_ARGUMENT)));

    let mut short_head = head_table();
    short_head.pop();
    let tables: Vec<(codevar_truetype_core::Tag, Vec<u8>)> = default_tables()
        .into_iter()
        .map(|(tag, body)| {
            if tag == tags::TAG_HEAD {
                (tag, short_head.clone())
            } else {
                (tag, body)
            }
        })
        .collect();
    assert!(matches!(
        SfntFont::open(&build_font(&tables), 0),
        Err(TtError::TABLE_MISSING)
    ));
}

#[test]
fn accepts_a_mac_true_font_without_horizontal_metrics() {
    let tables: Vec<(codevar_truetype_core::Tag, Vec<u8>)> = default_tables()
        .into_iter()
        .filter(|(tag, _)| *tag != tags::TAG_HHEA && *tag != tags::TAG_HMTX)
        .collect();
    let mut data = build_font(&tables);
    data[0..4].copy_from_slice(b"true");
    let font = SfntFont::open(&data, 0).unwrap();
    assert_eq!(font.hhea().number_of_h_metrics, 0);
    assert_eq!(font.metrics_for_glyph(1), (0, 0));
    assert_eq!(font.advance_for_glyph(1), 0);

    let mut face = test_face();
    font.apply_face_info(&mut face);
    assert!(face.is_sfnt());
    assert!(face.has_horizontal());
}
