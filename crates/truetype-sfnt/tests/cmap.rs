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

//! Character-map tests driven through the crate's public API only.

extern crate alloc;

use alloc::sync::Weak;
use codevar_truetype_core::{CharMap, Encoding, LibCell, TtError};
use codevar_truetype_sfnt::cmap::{
    CMAP_FLAG_OVERLAPPING, CMAP_FLAG_UNSORTED, CmapSubtable, OwnedCmapSubtable, SFNT_CMAP_CLASS,
    parse_charmaps, sfnt_find_encoding,
};
use codevar_truetype_sfnt::{container, tags};

#[allow(dead_code)]
#[path = "../src/test_util.rs"]
mod test_util;

use test_util::{cmap_format4, cmap_table, push_u16, push_u32};

fn one_record(platform_id: u16, encoding_id: u16, subtable: &[u8]) -> Vec<u8> {
    let mut table = Vec::with_capacity(12 + subtable.len());
    push_u16(&mut table, 0);
    push_u16(&mut table, 1);
    push_u16(&mut table, platform_id);
    push_u16(&mut table, encoding_id);
    push_u32(&mut table, 12);
    table.extend_from_slice(subtable);
    table
}

fn test_charmap() -> CharMap {
    CharMap {
        face: Weak::new(),
        encoding: Encoding::UNICODE,
        platform_id: 3,
        encoding_id: 1,
        clazz: &SFNT_CMAP_CLASS,
        data: LibCell::new(None),
    }
}

#[test]
fn parses_the_default_cmap_table() {
    let table = cmap_table();
    let charmaps = parse_charmaps(&table).unwrap();
    assert_eq!(charmaps.len(), 2);

    let bmp = &charmaps[0];
    assert_eq!(bmp.platform_id, 3);
    assert_eq!(bmp.encoding_id, 1);
    assert_eq!(bmp.encoding, Encoding::UNICODE);
    assert_eq!(bmp.subtable.format, 4);
    assert_eq!(bmp.subtable.flags, 0);
    assert_eq!(bmp.char_index(0x41), 1);
    assert_eq!(bmp.char_index(0x42), 2);
    assert_eq!(bmp.char_index(0x43), 0);
    assert_eq!(bmp.char_next(0x41), Some((0x42, 2)));
    assert_eq!(bmp.char_next(0x42), None);

    let ucs4 = &charmaps[1];
    assert_eq!(ucs4.platform_id, 3);
    assert_eq!(ucs4.encoding_id, 10);
    assert_eq!(ucs4.encoding, Encoding::UNICODE);
    assert_eq!(ucs4.subtable.format, 12);
    assert_eq!(ucs4.char_index(0x40), 0);
    assert_eq!(ucs4.char_index(0x42), 2);
    assert_eq!(ucs4.char_next(0x40), Some((0x41, 1)));
    assert_eq!(ucs4.char_next(0x42), None);
}

#[test]
fn rejects_broken_headers() {
    assert!(matches!(parse_charmaps(&[0, 0]), Err(TtError::INVALID_TABLE)));
    assert!(matches!(
        parse_charmaps(&[1, 0, 0, 0]),
        Err(TtError::INVALID_TABLE)
    ));
    assert!(parse_charmaps(&[0, 0, 0, 0]).unwrap().is_empty());
}

#[test]
fn skips_records_that_do_not_validate() {
    let mut table = Vec::new();
    push_u16(&mut table, 0);
    push_u16(&mut table, 2);
    push_u16(&mut table, 3);
    push_u16(&mut table, 1);
    push_u32(&mut table, 0);
    push_u16(&mut table, 3);
    push_u16(&mut table, 1);
    push_u32(&mut table, 20);
    push_u16(&mut table, 13);
    push_u16(&mut table, 0);
    push_u32(&mut table, 0);
    assert!(parse_charmaps(&table).unwrap().is_empty());
}

#[test]
fn format4_flags_reach_the_caller() {
    let mut overlapping = cmap_format4();
    overlapping[14..16].copy_from_slice(&0xFFFFu16.to_be_bytes());
    let table = one_record(3, 1, &overlapping);
    let charmaps = parse_charmaps(&table).unwrap();
    assert_eq!(charmaps[0].subtable.flags, CMAP_FLAG_OVERLAPPING);
    assert_eq!(charmaps[0].char_index(0x41), 1);
    assert_eq!(charmaps[0].char_index(0x60), 32);

    let mut unsorted = cmap_format4();
    unsorted[14..16].copy_from_slice(&0x0061u16.to_be_bytes());
    unsorted[20..22].copy_from_slice(&0x0061u16.to_be_bytes());
    unsorted[22..24].copy_from_slice(&0x0041u16.to_be_bytes());
    let table = one_record(3, 1, &unsorted);
    let charmaps = parse_charmaps(&table).unwrap();
    assert_eq!(charmaps[0].subtable.flags, CMAP_FLAG_UNSORTED);
    assert_eq!(charmaps[0].char_index(0x61), 33);
    assert_eq!(charmaps[0].char_index(0x41), 66);

    assert_eq!(CMAP_FLAG_UNSORTED, 1);
    assert_eq!(CMAP_FLAG_OVERLAPPING, 2);
}

#[test]
fn encoding_records_map_to_encodings() {
    assert_eq!(sfnt_find_encoding(0, 4), Encoding::UNICODE);
    assert_eq!(sfnt_find_encoding(1, 0), Encoding::APPLE_ROMAN);
    assert_eq!(sfnt_find_encoding(2, 1), Encoding::UNICODE);
    assert_eq!(sfnt_find_encoding(3, 1), Encoding::UNICODE);
    assert_eq!(sfnt_find_encoding(3, 10), Encoding::UNICODE);
    assert_eq!(sfnt_find_encoding(3, 2), Encoding::SJIS);
    assert_eq!(sfnt_find_encoding(4, 4), Encoding::NONE);
}

#[test]
fn the_static_class_dispatches_through_an_owned_payload() {
    let payload = OwnedCmapSubtable {
        format: 4,
        flags: 0,
        data: cmap_format4(),
    };
    assert_eq!(payload.char_index(0x41), 1);
    assert_eq!(payload.char_next(0x41), Some((0x42, 2)));

    let charmap = test_charmap();
    (SFNT_CMAP_CLASS.init)(&charmap, Some(&payload)).unwrap();
    assert_eq!(charmap.char_index(0x41), 1);
    assert_eq!(charmap.char_index(0x43), 0);
    assert_eq!(charmap.char_next(0x41), Some((0x42, 2)));
    assert_eq!(charmap.char_next(0x42), None);
    (SFNT_CMAP_CLASS.done)(&charmap);
    assert_eq!(charmap.char_index(0x41), 0);
    assert_eq!(charmap.char_next(0x41), None);
}

#[test]
fn the_static_class_rejects_a_missing_payload() {
    let charmap = test_charmap();
    assert!(matches!(
        (SFNT_CMAP_CLASS.init)(&charmap, None),
        Err(TtError::INVALID_ARGUMENT)
    ));
    assert_eq!(charmap.char_index(0x41), 0);
    assert_eq!(charmap.char_next(0x41), None);
}

#[test]
fn truncated_subtables_stay_bounds_checked() {
    let empty = CmapSubtable {
        format: 4,
        flags: 0,
        data: &[],
    };
    assert_eq!(empty.char_index(0x41), 0);
    assert!(empty.char_next(0x41).is_none());

    let short = CmapSubtable {
        format: 6,
        flags: 0,
        data: &cmap_table()[12..16],
    };
    assert_eq!(short.char_index(0x41), 0);
    assert!(short.char_next(0x41).is_none());
}
