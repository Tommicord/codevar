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

//! Container tests driven through the crate's public API only.

extern crate alloc;

use codevar_truetype_core::TtError;
use codevar_truetype_sfnt::container::{SfntContainer, SfntDirectory};
use codevar_truetype_sfnt::{container, tags};

#[allow(dead_code)]
#[path = "../src/test_util.rs"]
mod test_util;

use test_util::{build_font, build_ttc, default_font, default_tables};

#[test]
fn opens_a_plain_font_and_serves_its_tables() {
    let data = default_font();
    let container = SfntContainer::open(&data, 0).unwrap();
    assert_eq!(container.sfnt_version(), 0x0001_0000);
    assert_eq!(container.num_tables(), 8);
    assert_eq!(container.records().len(), 8);
    assert_eq!(container.num_faces(), 1);
    assert_eq!(container.face_index(), 0);
    assert_eq!(container.data(), data.as_slice());

    let ttc = container.ttc_header();
    assert!(!ttc.is_collection());
    assert!(!ttc.is_empty());
    assert_eq!(ttc.num_faces(), 1);
    assert_eq!(ttc.offset(0), Some(0));
    assert_eq!(ttc.offset(1), None);

    let head = container.table(tags::TAG_HEAD).unwrap();
    assert_eq!(head.len(), 54);
    assert!(container.table(tags::TAG_NAME).is_none());
}

#[test]
fn directory_lookup_round_trips_the_file_bytes() {
    let data = default_font();
    let container = SfntContainer::open(&data, 0).unwrap();
    let directory = container.directory();
    assert_eq!(directory.num_tables(), 8);
    assert_eq!(directory.sfnt_version(), 0x0001_0000);
    assert_eq!(directory.data(), data.as_slice());
    assert_eq!(directory.search_range(), 128);
    assert_eq!(directory.entry_selector(), 3);
    assert_eq!(directory.range_shift(), 0);

    let record = directory.find(tags::TAG_MAXP).unwrap();
    assert_eq!(record.tag, tags::TAG_MAXP);
    assert_eq!(record.length, 32);

    let (offset, length) = directory.table_range(tags::TAG_MAXP).unwrap();
    assert_eq!(length, 32);
    let table = directory.table(tags::TAG_MAXP).unwrap();
    assert_eq!(&data[offset..offset + length], table);
    assert!(directory.find(tags::TAG_NAME).is_none());
    assert!(directory.table_range(tags::TAG_NAME).is_none());
}

#[test]
fn checksums_accept_the_untouched_font_and_report_a_tampered_one() {
    let mut data = default_font();
    let container = SfntContainer::open(&data, 0).unwrap();
    assert!(container.verify_checksums().is_empty());
    assert_eq!(container.verify_checksum(tags::TAG_MAXP), Some(true));
    assert_eq!(container.verify_checksum(tags::TAG_NAME), None);

    let record = container
        .directory()
        .find(tags::TAG_MAXP)
        .unwrap();
    let at = record.offset as usize;
    data[at + 8] ^= 0xFF;
    let tampered = SfntContainer::open(&data, 0).unwrap();
    assert_eq!(tampered.verify_checksums(), vec![tags::TAG_MAXP]);
    assert_eq!(tampered.verify_checksum(tags::TAG_MAXP), Some(false));
}

#[test]
fn opens_every_font_of_a_collection() {
    let first = default_font();
    let mut tables = default_tables();
    for (tag, body) in tables.iter_mut() {
        if *tag == tags::TAG_MAXP {
            body[4..6].copy_from_slice(&7u16.to_be_bytes());
        }
    }
    let second = build_font(&tables);
    let collection = build_ttc(&[first.clone(), second]);

    let face0 = SfntContainer::open(&collection, 0).unwrap();
    assert_eq!(face0.num_faces(), 2);
    assert_eq!(face0.face_index(), 0);
    assert!(face0.ttc_header().is_collection());
    assert_eq!(face0.ttc_header().num_faces(), 2);
    let maxp0 = face0.table(tags::TAG_MAXP).unwrap();
    assert_eq!(u16::from_be_bytes([maxp0[4], maxp0[5]]), 5);

    let face1 = SfntContainer::open(&collection, 1).unwrap();
    assert_eq!(face1.face_index(), 1);
    assert_eq!(face1.num_faces(), 2);
    let maxp1 = face1.table(tags::TAG_MAXP).unwrap();
    assert_eq!(u16::from_be_bytes([maxp1[4], maxp1[5]]), 7);

    let negative = SfntContainer::open(&collection, -1).unwrap();
    assert_eq!(negative.face_index(), 0);

    assert!(matches!(
        SfntContainer::open(&collection, 2),
        Err(TtError::INVALID_ARGUMENT)
    ));
}

#[test]
fn open_reports_container_failures() {
    let data = default_font();
    assert!(matches!(
        SfntContainer::open(&data[..2], 0),
        Err(TtError::INVALID_FILE_FORMAT)
    ));
    assert!(matches!(
        SfntContainer::open(b"wOFF....", 0),
        Err(TtError::UNKNOWN_FILE_FORMAT)
    ));
    assert!(matches!(
        SfntContainer::open(&data, 4),
        Err(TtError::INVALID_ARGUMENT)
    ));
    assert!(matches!(
        SfntContainer::open(b"XXXX\0\0\0\0\0\0\0\x08", 0),
        Err(TtError::UNKNOWN_FILE_FORMAT)
    ));
}

#[test]
fn directory_parse_rejects_a_truncated_header() {
    let data = default_font();
    assert!(matches!(
        SfntDirectory::parse(&data[..8], 0),
        Err(TtError::INVALID_TABLE)
    ));
    assert!(matches!(
        SfntDirectory::parse(&data[..12], 0),
        Err(TtError::INVALID_TABLE)
    ));
    assert!(SfntDirectory::parse(&data, 0).is_ok());
}

#[test]
fn records_are_sorted_by_tag() {
    let data = default_font();
    let container = SfntContainer::open(&data, 0).unwrap();
    let record_tags: Vec<_> = container
        .records()
        .iter()
        .map(|record| record.tag)
        .collect();
    assert!(record_tags.contains(&tags::TAG_CMAP));
    assert!(record_tags.contains(&tags::TAG_LOCA));
    let mut expected = record_tags.clone();
    expected.sort_unstable();
    assert_eq!(record_tags, expected);
}
