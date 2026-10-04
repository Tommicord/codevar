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

//! The full multi-stage pipeline this crate enables, as one flowing
//! scenario: raw font bytes -> [`SfntContainer::open`] ->
//! [`SfntFont::open`] -> table parsing (`head`/`maxp`/`hhea`/`hmtx`/
//! `loca`) -> `char_index` -> `glyph_location` -> metrics -> kerning.
//!
//! Only this crate's stages are exercised: the assertions check that
//! the pipeline runs end to end, that the same input always yields the
//! same collected results, and that corrupted bytes produce errors
//! instead of panics.  Stages beyond the SFNT layer (outline loading,
//! rasterization, autofitting) belong to later crates and are covered
//! by an umbrella test elsewhere.

extern crate alloc;

use codevar_truetype_core::TtError;
use codevar_truetype_sfnt::{SfntContainer, SfntFont, container, tags};

#[allow(dead_code)]
#[path = "../src/test_util.rs"]
mod test_util;

use test_util::default_font;

/// Everything the pipeline collects for one character of the walk.
#[derive(Debug, PartialEq, Eq)]
struct GlyphReport {
    char_code: u32,
    gindex: u32,
    location: Option<(usize, usize)>,
    metrics: (u16, i16),
    advance: u16,
}

/// Everything the pipeline collects for one pass over the font bytes.
#[derive(Debug, PartialEq, Eq)]
struct PipelineReport {
    num_tables: usize,
    num_faces: usize,
    face_index: i64,
    num_glyphs: u16,
    units_per_em: u16,
    num_charmaps: usize,
    has_kerning: bool,
    glyphs: Vec<GlyphReport>,
    kern_pairs: Vec<(u32, u32, i32)>,
    table_lengths: Vec<(u32, usize)>,
}

/// Runs the whole pipeline over `data`, collecting deterministic
/// results; `Err` for any corrupted stage along the way.
fn run_pipeline(data: &[u8]) -> Result<PipelineReport, TtError> {
    let container = SfntContainer::open(data, 0)?;
    let num_tables = container.num_tables();
    let num_faces = container.num_faces();
    let face_index = container.face_index();
    let table_lengths = container
        .records()
        .iter()
        .map(|record| (record.tag, record.length as usize))
        .collect();

    let font = SfntFont::open(data, 0)?;
    let num_glyphs = font.num_glyphs();
    let units_per_em = font.units_per_em();
    let num_charmaps = font.charmaps().len();
    let has_kerning = font.has_kerning();

    let head = font.head();
    let maxp = font.maxp();
    let hhea = font.hhea();
    if head.units_per_em == 0 || maxp.num_glyphs != num_glyphs || hhea.number_of_h_metrics == 0 {
        return Err(TtError::INVALID_TABLE);
    }

    let mut glyphs = Vec::new();
    for char_code in [u32::from(b'A'), u32::from(b'B'), u32::from(b'Z')] {
        let gindex = font.char_index(char_code);
        let location = if gindex == 0 {
            None
        } else {
            font.glyph_location(gindex)
        };
        glyphs.push(GlyphReport {
            char_code,
            gindex,
            location,
            metrics: font.metrics_for_glyph(gindex),
            advance: font.advance_for_glyph(gindex),
        });
    }

    let kern_pairs = [(1u32, 2u32), (3, 4), (2, 1)]
        .into_iter()
        .map(|(left, right)| (left, right, font.kerning(left, right)))
        .collect();

    Ok(PipelineReport {
        num_tables,
        num_faces,
        face_index,
        num_glyphs,
        units_per_em,
        num_charmaps,
        has_kerning,
        glyphs,
        kern_pairs,
        table_lengths,
    })
}

#[test]
fn the_full_pipeline_runs_end_to_end() {
    let data = default_font();
    let report = run_pipeline(&data).unwrap();

    assert_eq!(report.num_tables, 8);
    assert_eq!(report.num_faces, 1);
    assert_eq!(report.face_index, 0);
    assert_eq!(report.num_glyphs, 5);
    assert_eq!(report.units_per_em, 1000);
    assert_eq!(report.num_charmaps, 2);
    assert!(report.has_kerning);
    assert_eq!(report.table_lengths.len(), 8);

    assert_eq!(
        report.glyphs,
        vec![
            GlyphReport {
                char_code: 0x41,
                gindex: 1,
                location: Some((0, 10)),
                metrics: (600, 20),
                advance: 600,
            },
            GlyphReport {
                char_code: 0x42,
                gindex: 2,
                location: Some((10, 30)),
                metrics: (700, -30),
                advance: 700,
            },
            GlyphReport {
                char_code: 0x5A,
                gindex: 0,
                location: None,
                metrics: (500, -10),
                advance: 500,
            },
        ]
    );

    assert_eq!(report.kern_pairs, vec![(1, 2, -40), (3, 4, 25), (2, 1, 0)]);
}

#[test]
fn the_pipeline_is_deterministic_across_runs() {
    let data = default_font();
    let first = run_pipeline(&data).unwrap();
    let second = run_pipeline(&data).unwrap();
    assert_eq!(first, second);

    let container = SfntContainer::open(&data, 0).unwrap();
    assert!(container.verify_checksum(tags::TAG_MAXP).unwrap());
    assert_eq!(
        codevar_truetype_sfnt::cmap::parse_charmaps(container.table(tags::TAG_CMAP).unwrap())
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn corrupted_bytes_return_errors_and_never_panic() {
    let data = default_font();

    let truncated_header = &data[..2];
    assert!(matches!(
        run_pipeline(truncated_header),
        Err(TtError::INVALID_FILE_FORMAT)
    ));

    let truncated_directory = &data[..11];
    assert!(matches!(
        run_pipeline(truncated_directory),
        Err(TtError::INVALID_TABLE)
    ));

    let mut bad_version = data.clone();
    bad_version[0..4].copy_from_slice(b"XXXX");
    assert!(matches!(
        run_pipeline(&bad_version),
        Err(TtError::UNKNOWN_FILE_FORMAT)
    ));

    let mut truncated_records = data.clone();
    truncated_records.truncate(64);
    assert!(matches!(
        run_pipeline(&truncated_records),
        Err(TtError::INVALID_TABLE)
    ));

    let mut shifted = data.clone();
    let num_tables = u16::from_be_bytes([shifted[4], shifted[5]]) as usize;
    for index in 0..num_tables {
        let record = 12 + index * 16 + 8;
        shifted[record..record + 4].copy_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
    }
    assert!(matches!(
        run_pipeline(&shifted),
        Err(TtError::UNKNOWN_FILE_FORMAT)
    ));

    let mut no_maxp = data.clone();
    let num_tables = u16::from_be_bytes([no_maxp[4], no_maxp[5]]) as usize;
    for index in 0..num_tables {
        let record = 12 + index * 16;
        if &no_maxp[record..record + 4] == b"maxp" {
            no_maxp[record..record + 4].copy_from_slice(b"nopE");
        }
    }
    assert!(matches!(run_pipeline(&no_maxp), Err(TtError::TABLE_MISSING)));

    let mut bad_upem = data.clone();
    let num_tables = u16::from_be_bytes([bad_upem[4], bad_upem[5]]) as usize;
    for index in 0..num_tables {
        let record = 12 + index * 16;
        if &bad_upem[record..record + 4] == b"head" {
            let offset = u32::from_be_bytes([
                bad_upem[record + 8],
                bad_upem[record + 9],
                bad_upem[record + 10],
                bad_upem[record + 11],
            ]) as usize;
            bad_upem[offset + 18..offset + 20].copy_from_slice(&0u16.to_be_bytes());
        }
    }
    assert!(matches!(run_pipeline(&bad_upem), Err(TtError::INVALID_TABLE)));
}
