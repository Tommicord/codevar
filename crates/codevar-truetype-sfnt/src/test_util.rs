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

//! Canonical synthetic-font builders shared by this crate's unit tests.
//!
//! Every fixture below is hand-built from the OpenType field layouts
//! that FreeType 2.6 reads in `ttload.c`, `ttmtx.c`, `ttkern.c` and
//! `ttcmap.c`; [`default_font`] assembles a complete, checksummed
//! 8-table font so container, table and charmap tests all agree on the
//! same bytes.

use alloc::vec;
use alloc::vec::Vec;

use crate::container::SfntDirectory;
use crate::tags::{TAG_CMAP, TAG_GLYF, TAG_HEAD, TAG_HHEA, TAG_HMTX, TAG_KERN, TAG_LOCA, TAG_MAXP};
use codevar_truetype_core::Tag;

/// Appends `value` as big-endian bytes.
pub(crate) fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
pub(crate) fn push_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
pub(crate) fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
pub(crate) fn push_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// A 54-byte `'head'` table: `unitsPerEm` 1000, `macStyle` bold,
/// short `loca`, bbox `(-100, -200, 900, 800)`.
pub(crate) fn head_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(54);
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, 0x1234_5678);
    push_u32(&mut out, 0x5F0F_3CF5);
    push_u16(&mut out, 3);
    push_u16(&mut out, 1000);
    push_i64(&mut out, -1);
    push_i64(&mut out, 3_660_000_000);
    push_i16(&mut out, -100);
    push_i16(&mut out, -200);
    push_i16(&mut out, 900);
    push_i16(&mut out, 800);
    push_u16(&mut out, 1);
    push_u16(&mut out, 8);
    push_i16(&mut out, 2);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    out
}

/// A 32-byte `'maxp'` table: 5 glyphs, version 1.0, clamped fields.
pub(crate) fn maxp_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, 5);
    for field in [10u16, 3, 0, 0, 2, 16, 8, 100, 4, 64, 0, 0, 0] {
        push_u16(&mut out, field);
    }
    out
}

/// A 36-byte `'hhea'` table: ascender 800, descender -200, 3 metrics.
pub(crate) fn hhea_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(36);
    push_u32(&mut out, 0x0001_0000);
    push_i16(&mut out, 800);
    push_i16(&mut out, -200);
    push_i16(&mut out, 90);
    push_u16(&mut out, 700);
    push_i16(&mut out, -50);
    push_i16(&mut out, -40);
    push_i16(&mut out, 900);
    push_i16(&mut out, 1);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    push_u16(&mut out, 3);
    out
}

/// A 16-byte `'hmtx'` table: 3 long metrics `(500, -10)`, `(600, 20)`,
/// `(700, -30)` plus trailing side bearings `40` and `-50`.
pub(crate) fn hmtx_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    push_u16(&mut out, 500);
    push_i16(&mut out, -10);
    push_u16(&mut out, 600);
    push_i16(&mut out, 20);
    push_u16(&mut out, 700);
    push_i16(&mut out, -30);
    push_i16(&mut out, 40);
    push_i16(&mut out, -50);
    out
}

/// A short-format `'loca'` table of 6 entries encoding the byte
/// offsets `[0, 0, 10, 30, 46, 60]`.
pub(crate) fn loca_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    for half in [0u16, 0, 5, 15, 23, 30] {
        push_u16(&mut out, half);
    }
    out
}

/// A 60-byte `'glyf'` table matching [`loca_table`]: glyph 0 is empty,
/// the remaining glyphs hold distinct filler bytes.
pub(crate) fn glyf_table() -> Vec<u8> {
    (0..60)
        .map(|byte| (byte as u8).wrapping_add(1))
        .collect()
}

/// A format 4 `'cmap'` subtable (32 bytes) mapping `A` to glyph 1,
/// `B` to glyph 2 and `0xFFFF` to glyph 0, plus the mandatory
/// `0xFFFF` sentinel segment.
pub(crate) fn cmap_format4() -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    push_u16(&mut out, 4);
    push_u16(&mut out, 32);
    push_u16(&mut out, 0);
    push_u16(&mut out, 4);
    push_u16(&mut out, 4);
    push_u16(&mut out, 1);
    push_u16(&mut out, 0);
    for end in [0x0042u16, 0xFFFF] {
        push_u16(&mut out, end);
    }
    push_u16(&mut out, 0);
    for start in [0x0041u16, 0xFFFF] {
        push_u16(&mut out, start);
    }
    for delta in [0xFFC0u16, 1] {
        push_u16(&mut out, delta);
    }
    push_u16(&mut out, 0);
    push_u16(&mut out, 0);
    out
}

/// A format 12 `'cmap'` subtable (28 bytes) mapping `U+0041..U+0042`
/// to glyphs 1 and 2.
pub(crate) fn cmap_format12() -> Vec<u8> {
    let mut out = Vec::with_capacity(28);
    push_u16(&mut out, 12);
    push_u16(&mut out, 0);
    push_u32(&mut out, 28);
    push_u32(&mut out, 0);
    push_u32(&mut out, 1);
    push_u32(&mut out, 0x41);
    push_u32(&mut out, 0x42);
    push_u32(&mut out, 1);
    out
}

/// The complete 80-byte `'cmap'` table: a `(3, 1)` format 4 subtable
/// at offset 20 and a `(3, 10)` format 12 subtable at offset 52.
pub(crate) fn cmap_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(80);
    push_u16(&mut out, 0);
    push_u16(&mut out, 2);
    push_u16(&mut out, 3);
    push_u16(&mut out, 1);
    push_u32(&mut out, 20);
    push_u16(&mut out, 3);
    push_u16(&mut out, 10);
    push_u32(&mut out, 52);
    out.extend_from_slice(&cmap_format4());
    out.extend_from_slice(&cmap_format12());
    out
}

/// A 30-byte `'kern'` table with one horizontal format 0 subtable
/// carrying `(1, 2) -> -40` and `(3, 4) -> 25`.
pub(crate) fn kern_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(30);
    push_u16(&mut out, 0);
    push_u16(&mut out, 1);
    push_u16(&mut out, 0);
    push_u16(&mut out, 26);
    push_u16(&mut out, 0x0001);
    push_u16(&mut out, 2);
    push_u16(&mut out, 12);
    push_u16(&mut out, 1);
    push_u16(&mut out, 0);
    push_u32(&mut out, 0x0001_0002);
    push_i16(&mut out, -40);
    push_u32(&mut out, 0x0003_0004);
    push_i16(&mut out, 25);
    out
}

/// The tables of [`default_font`] in `(tag, bytes)` form.
pub(crate) fn default_tables() -> Vec<(Tag, Vec<u8>)> {
    vec![
        (TAG_CMAP, cmap_table()),
        (TAG_GLYF, glyf_table()),
        (TAG_HEAD, head_table()),
        (TAG_HHEA, hhea_table()),
        (TAG_HMTX, hmtx_table()),
        (TAG_KERN, kern_table()),
        (TAG_LOCA, loca_table()),
        (TAG_MAXP, maxp_table()),
    ]
}

/// Assembles `tables` into a complete SFNT file: sorted directory,
/// correct `searchRange`/`entrySelector`/`rangeShift`, 4-byte aligned
/// offsets and spec-conformant checksums (`'head'` with
/// `checkSumAdjustment` zeroed).
pub(crate) fn build_font(tables: &[(Tag, Vec<u8>)]) -> Vec<u8> {
    let mut sorted: Vec<(Tag, &[u8])> = tables
        .iter()
        .map(|(tag, body)| (*tag, body.as_slice()))
        .collect();
    sorted.sort_by_key(|(tag, _)| *tag);
    let count = sorted.len() as u16;
    let mut entry_selector = 0u16;
    let mut largest_pow2 = 1u16;
    while largest_pow2 * 2 <= count {
        largest_pow2 *= 2;
        entry_selector += 1;
    }
    let search_range = largest_pow2 * 16;
    let range_shift = count * 16 - search_range;
    let mut out = Vec::new();
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, count);
    push_u16(&mut out, search_range);
    push_u16(&mut out, entry_selector);
    push_u16(&mut out, range_shift);
    let dir_start = out.len();
    out.resize(dir_start + count as usize * 16, 0);
    for (index, (tag, body)) in sorted.iter().enumerate() {
        let table_offset = out.len() as u32;
        let checksum = if *tag == TAG_HEAD && body.len() >= 12 {
            let mut head = body.to_vec();
            head[8..12].copy_from_slice(&0u32.to_be_bytes());
            SfntDirectory::checksum(&head)
        } else {
            SfntDirectory::checksum(body)
        };
        let record = dir_start + index * 16;
        out[record..record + 4].copy_from_slice(&tag.to_be_bytes());
        out[record + 4..record + 8].copy_from_slice(&checksum.to_be_bytes());
        out[record + 8..record + 12].copy_from_slice(&table_offset.to_be_bytes());
        out[record + 12..record + 16].copy_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(body);
        let padding = (4 - (body.len() % 4)) % 4;
        out.resize(out.len() + padding, 0);
    }
    out
}

/// A complete, checksummed font with all 8 tables.
pub(crate) fn default_font() -> Vec<u8> {
    build_font(&default_tables())
}

/// Wraps each font of `fonts` in a `'ttcf'` collection header with
/// 4-byte aligned sub-font offsets.  Every embedded directory's record
/// offsets are relocated to absolute file positions, as required by
/// `TableRecord`'s contract (and by the OpenType collection spec).
pub(crate) fn build_ttc(fonts: &[Vec<u8>]) -> Vec<u8> {
    let count = fonts.len() as u32;
    let header_len = 12 + count as usize * 4;
    let mut bases = Vec::with_capacity(fonts.len());
    let mut running = header_len as u32;
    for font in fonts {
        bases.push(running);
        running += font.len() as u32;
        running += ((4 - (font.len() % 4)) % 4) as u32;
    }
    let mut out = Vec::with_capacity(running as usize);
    out.extend_from_slice(b"ttcf");
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, count);
    for base in &bases {
        push_u32(&mut out, *base);
    }
    for (font, base) in fonts.iter().zip(&bases) {
        let mut font = font.clone();
        let num_tables = u16::from_be_bytes([font[4], font[5]]) as usize;
        for index in 0..num_tables {
            let record = 12 + index * 16 + 8;
            let offset =
                u32::from_be_bytes([font[record], font[record + 1], font[record + 2], font[record + 3]]);
            font[record..record + 4].copy_from_slice(&(offset + *base).to_be_bytes());
        }
        out.extend_from_slice(&font);
        let padding = (4 - (font.len() % 4)) % 4;
        out.resize(out.len() + padding, 0);
    }
    out
}
