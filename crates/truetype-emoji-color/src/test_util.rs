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
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Synthetic color-font fixtures shared by this crate's unit tests:
//! minimal `glyf` fonts, `COLR`/`CPAL` tables, `CBLC`/`CBDT` strike
//! pairs and the PNG/zlib helpers that describe their images.
//!
//! The glyph side mirrors `codevar-truetype-glyf`'s test utilities
//! (a checksummed SFNT with long `loca`), narrowed to what the color
//! pipeline needs: rectangles, empty glyphs and one `hmtx` record per
//! glyph.

use alloc::vec;
use alloc::vec::Vec;

use codevar_truetype_core::Tag;
use codevar_truetype_sfnt::{
    SfntDirectory,
    tags::{TAG_GLYF, TAG_HEAD, TAG_HHEA, TAG_HMTX, TAG_LOCA, TAG_MAXP},
};

/// Appends `value` as big-endian bytes.
pub fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
fn push_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
pub fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// One point of a simple glyph contour (always on-curve in the
/// rectangles this module builds).
#[derive(Clone, Copy)]
struct Point {
    x: i64,
    y: i64,
    on_curve: bool,
}

/// A ten-byte `glyf` glyph header.
fn glyph_header(n_contours: i16, x_min: i16, y_min: i16, x_max: i16, y_max: i16) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    push_i16(&mut out, n_contours);
    push_i16(&mut out, x_min);
    push_i16(&mut out, y_min);
    push_i16(&mut out, x_max);
    push_i16(&mut out, y_max);
    out
}

/// Encodes a well-formed simple glyph with one contour per entry and
/// no instructions.
///
/// # Panics
///
/// Only when a fixture contour is empty — a programming error of the
/// test data, never of the loader under test.
fn simple_glyph(
    x_min: i16,
    y_min: i16,
    x_max: i16,
    y_max: i16,
    contours: &[Vec<Point>],
    instructions: &[u8],
) -> Vec<u8> {
    assert!(!contours.is_empty(), "a simple glyph needs one contour");
    let mut out = glyph_header(contours.len() as i16, x_min, y_min, x_max, y_max);

    let mut total = 0usize;
    for contour in contours {
        assert!(!contour.is_empty(), "an empty contour has no end point");
        total += contour.len();
        push_i16(&mut out, (total - 1) as i16);
    }
    push_u16(&mut out, instructions.len() as u16);
    out.extend_from_slice(instructions);

    let points: Vec<Point> = contours.iter().flatten().copied().collect();
    let (flags, x_data, y_data) = encode_coordinates(&points);
    write_repeated_flags(&mut out, &flags);
    out.extend_from_slice(&x_data);
    out.extend_from_slice(&y_data);
    out
}

/// Chooses the flag bits and payload bytes for every point.
fn encode_coordinates(points: &[Point]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut flags = Vec::with_capacity(points.len());
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    let mut previous = 0i64;
    for point in points {
        let mut flag = if point.on_curve { 1u8 } else { 0 };
        flag |= encode_delta(point.x - previous, 0x02, 0x10, &mut x_data);
        previous = point.x;
        flags.push(flag);
    }

    let mut previous = 0i64;
    for (index, point) in points.iter().enumerate() {
        let bits = encode_delta(point.y - previous, 0x04, 0x20, &mut y_data);
        flags[index] |= bits;
        previous = point.y;
    }

    (flags, x_data, y_data)
}

/// Encodes one delta: returns the flag bits and appends the payload.
fn encode_delta(delta: i64, short_bit: u8, same_bit: u8, out: &mut Vec<u8>) -> u8 {
    if delta == 0 {
        same_bit
    } else if (1..=255).contains(&delta) {
        out.push(delta as u8);
        short_bit | same_bit
    } else if (-255..=-1).contains(&delta) {
        out.push(delta.unsigned_abs() as u8);
        short_bit
    } else {
        push_i16(out, delta as i16);
        0
    }
}

/// Writes `flags` with the `REPEAT` (0x08) run-length encoding.
fn write_repeated_flags(out: &mut Vec<u8>, flags: &[u8]) {
    let mut index = 0usize;
    while index < flags.len() {
        let flag = flags[index];
        let mut run = 1usize;
        while index + run < flags.len() && flags[index + run] == flag && run < 255 {
            run += 1;
        }
        if run == 1 {
            out.push(flag);
        } else {
            out.push(flag | 0x08);
            out.push((run - 1) as u8);
        }
        index += run;
    }
}

/// A simple glyph filling the rectangle `(x_min, y_min)` to
/// `(x_max, y_max)`, with four on-curve points.
pub fn rect_glyph(x_min: i16, y_min: i16, x_max: i16, y_max: i16) -> Vec<u8> {
    let contour = vec![
        Point {
            x: i64::from(x_min),
            y: i64::from(y_min),
            on_curve: true,
        },
        Point {
            x: i64::from(x_max),
            y: i64::from(y_min),
            on_curve: true,
        },
        Point {
            x: i64::from(x_max),
            y: i64::from(y_max),
            on_curve: true,
        },
        Point {
            x: i64::from(x_min),
            y: i64::from(y_max),
            on_curve: true,
        },
    ];
    simple_glyph(x_min, y_min, x_max, y_max, &[contour], &[])
}

/// A 54-byte `'head'` table: `unitsPerEm` 1000, long `loca`.
fn head_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(54);
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, 0);
    push_u32(&mut out, 0x5F0F_3CF5);
    push_u16(&mut out, 3);
    push_u16(&mut out, 1000);
    out.extend_from_slice(&0i64.to_be_bytes());
    out.extend_from_slice(&0i64.to_be_bytes());
    push_i16(&mut out, -100);
    push_i16(&mut out, -200);
    push_i16(&mut out, 900);
    push_i16(&mut out, 800);
    push_u16(&mut out, 0);
    push_u16(&mut out, 8);
    push_i16(&mut out, 2);
    push_i16(&mut out, 1);
    push_i16(&mut out, 0);
    out
}

/// A 32-byte `'maxp'` table (version 1.0).
fn maxp_table(num_glyphs: u16, max_component_depth: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, num_glyphs);
    for field in [16u16, 4, 32, 8, 2, 64, 64, 16, 0, 256, 64, 8] {
        push_u16(&mut out, field);
    }
    push_u16(&mut out, max_component_depth);
    out
}

/// A 36-byte `'hhea'` table: ascender 800, descender -200.
fn hhea_table(number_of_h_metrics: u16) -> Vec<u8> {
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
    for _ in 0..4 {
        push_i16(&mut out, 0);
    }
    push_i16(&mut out, 0);
    push_u16(&mut out, number_of_h_metrics);
    out
}

/// An `'hmtx'` table with one `(advance, left_side_bearing)` long
/// metric per glyph.
fn hmtx_table(metrics: &[(u16, i16)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(metrics.len() * 4);
    for &(advance, left_side_bearing) in metrics {
        push_u16(&mut out, advance);
        push_i16(&mut out, left_side_bearing);
    }
    out
}

/// The `(glyf, loca)` pair for `glyphs`, using the long `loca` format.
fn glyf_and_loca(glyphs: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>) {
    let mut glyf = Vec::new();
    let mut loca = Vec::new();
    push_u32(&mut loca, 0);
    for glyph in glyphs {
        glyf.extend_from_slice(glyph);
        push_u32(&mut loca, glyf.len() as u32);
    }
    (glyf, loca)
}

/// Assembles `tables` into a complete SFNT file: sorted directory,
/// correct `searchRange`/`entrySelector`/`rangeShift`, 4-byte aligned
/// offsets and spec-conformant checksums (`'head'` with
/// `checkSumAdjustment` zeroed).
fn build_font(tables: &[(Tag, Vec<u8>)]) -> Vec<u8> {
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

/// Wraps `glyphs`, `metrics` (one pair per glyph) and the `extra`
/// tables (e.g. `COLR`/`CPAL`/`CBLC`/`CBDT`) into a checksummed
/// font.
///
/// # Panics
///
/// Only when `metrics` and `glyphs` disagree in length — a fixture
/// programming error.
pub fn color_font(glyphs: &[Vec<u8>], metrics: &[(u16, i16)], extra: &[(Tag, Vec<u8>)]) -> Vec<u8> {
    assert_eq!(metrics.len(), glyphs.len(), "one hmtx record per glyph");
    let (glyf, loca) = glyf_and_loca(glyphs);
    let num_glyphs = glyphs.len() as u16;
    let mut tables = vec![
        (TAG_GLYF, glyf),
        (TAG_HEAD, head_table()),
        (TAG_HHEA, hhea_table(num_glyphs)),
        (TAG_HMTX, hmtx_table(metrics)),
        (TAG_LOCA, loca),
        (TAG_MAXP, maxp_table(num_glyphs, 0)),
    ];
    tables.extend(extra.iter().cloned());
    build_font(&tables)
}

/// Builds a `COLR` v0 table with one base glyph pointing at
/// `layers` (`(glyph_index, color_index)` pairs; `0xFFFF` is the
/// foreground sentinel).
pub fn colr_table(base_glyph: u16, layers: &[(u16, u16)]) -> Vec<u8> {
    let mut out = Vec::new();
    push_u16(&mut out, 0); // version
    push_u16(&mut out, 1); // numBaseGlyphRecords
    push_u32(&mut out, 14); // baseGlyphRecordsOffset
    // The layer array follows the 6-byte base record; an empty array
    // may reuse the base offset because only `offset < len` is
    // validated.
    let layer_offset = if layers.is_empty() { 14 } else { 20 };
    push_u32(&mut out, layer_offset);
    push_u16(&mut out, layers.len() as u16); // numLayerRecords
    push_u16(&mut out, base_glyph);
    push_u16(&mut out, 0); // firstLayerIndex
    push_u16(&mut out, layers.len() as u16); // numLayers
    for &(glyph, color) in layers {
        push_u16(&mut out, glyph);
        push_u16(&mut out, color);
    }
    out
}

/// Builds a one-palette `CPAL` table; `colors` are
/// `(blue, green, red, alpha)` records.  `dark_background` selects
/// version 1 with the palette flagged
/// [`PALETTE_FOR_DARK_BACKGROUND`](crate::cpal::PALETTE_FOR_DARK_BACKGROUND).
pub fn cpal_table(colors: &[(u8, u8, u8, u8)], dark_background: bool) -> Vec<u8> {
    let mut out = Vec::new();
    push_u16(&mut out, u16::from(dark_background)); // version
    push_u16(&mut out, colors.len() as u16); // numPaletteEntries
    push_u16(&mut out, 1); // numPalettes
    push_u16(&mut out, colors.len() as u16); // numColorRecords
    if dark_background {
        push_u32(&mut out, 28); // colorRecordsArrayOffset
        push_u16(&mut out, 0); // colorRecordIndices[0]
        push_u32(&mut out, 26); // v1 paletteTypesArrayOffset
        push_u32(&mut out, 0); // paletteLabelsArrayOffset (absent)
        push_u32(&mut out, 0); // paletteEntryLabelsArrayOffset
        push_u16(&mut out, 0x0001); // paletteTypes[0]
    } else {
        push_u32(&mut out, 14); // colorRecordsArrayOffset
        push_u16(&mut out, 0); // colorRecordIndices[0]
    }
    for &(blue, green, red, alpha) in colors {
        out.extend_from_slice(&[blue, green, red, alpha]);
    }
    out
}

/// One embedded bitmap image of a strike fixture.
pub struct SbitImageSpec {
    /// The glyph the image belongs to (ascending and contiguous
    /// within one strike).
    pub gid: u16,
    /// Image height in pixels.
    pub height: u8,
    /// Image width in pixels.
    pub width: u8,
    /// Horizontal left side bearing.
    pub bearing_x: i8,
    /// Horizontal top side bearing.
    pub bearing_y: i8,
    /// Horizontal advance.
    pub advance: u8,
    /// The PNG payload describing the pixels.
    pub png: Vec<u8>,
}

/// The `CBLC` directory of one strike: version 3.0, a single
/// `BitmapSize` record at offset 8, one gid range, and an
/// `IndexSubHeader` at 64 followed by `entries`.
fn cblc_skeleton(
    ppem: u8,
    bit_depth: u8,
    start_gid: u16,
    end_gid: u16,
    index_format: u16,
    image_format: u16,
    entries: &[u8],
) -> Vec<u8> {
    let mut cblc = Vec::new();
    push_u32(&mut cblc, 0x0003_0000); // version
    push_u32(&mut cblc, 1); // numSizes
    // `BitmapSize` @8.
    push_u32(&mut cblc, 56); // indexSubTableArrayOffset
    push_u32(&mut cblc, 16 + entries.len() as u32); // indexTablesSize
    push_u32(&mut cblc, 1); // numberOfIndexSubTables
    push_u32(&mut cblc, 0); // colorRef
    cblc.extend_from_slice(&[0u8; 24]); // hori/vert sbitLineMetrics
    push_u16(&mut cblc, start_gid);
    push_u16(&mut cblc, end_gid);
    cblc.extend_from_slice(&[ppem, ppem, bit_depth, 1]); // ppemX/Y, depth, flags
    // `IndexSubTableArray` @56.
    push_u16(&mut cblc, start_gid);
    push_u16(&mut cblc, end_gid);
    push_u32(&mut cblc, 8); // relative offset to the header @64
    // `IndexSubHeader` @64.
    push_u16(&mut cblc, index_format);
    push_u16(&mut cblc, image_format);
    push_u32(&mut cblc, 4); // imageDataOffset: CBDT records start at 4
    cblc.extend_from_slice(entries);
    cblc
}

/// A `CBLC`/`CBDT` pair with index format 1 and image format 17
/// (small metrics plus PNG), for one strike at `ppem` and
/// `bit_depth`.  `specs` must be in ascending, contiguous glyph
/// order.
///
/// # Panics
///
/// Only for an empty `specs` slice — a fixture programming error.
pub fn cblc_cbdt17(ppem: u8, bit_depth: u8, specs: &[SbitImageSpec]) -> (Vec<u8>, Vec<u8>) {
    assert!(!specs.is_empty(), "a strike needs at least one glyph");
    let start_gid = specs[0].gid;
    let end_gid = specs[specs.len() - 1].gid;

    let mut cbdt = Vec::new();
    push_u32(&mut cbdt, 0x0003_0000); // CBDT version
    // Index format 1: `specs.len() + 1` big-endian u32 offsets
    // relative to the record after the version.
    let mut entries = Vec::with_capacity((specs.len() + 1) * 4);
    push_u32(&mut entries, 0);
    for spec in specs {
        cbdt.push(spec.height);
        cbdt.push(spec.width);
        cbdt.push(spec.bearing_x as u8);
        cbdt.push(spec.bearing_y as u8);
        cbdt.push(spec.advance);
        push_u32(&mut cbdt, spec.png.len() as u32);
        cbdt.extend_from_slice(&spec.png);
        push_u32(&mut entries, cbdt.len() as u32 - 4);
    }

    let cblc = cblc_skeleton(ppem, bit_depth, start_gid, end_gid, 1, 17, &entries);
    (cblc, cbdt)
}

/// A `CBLC`/`CBDT` pair with index format 19 and image format 19
/// (uniform big metrics in the index, PNG payloads in `CBDT`).
///
/// # Panics
///
/// Only for an empty `specs` slice or images whose record sizes
/// differ — fixture programming errors, since format 19 requires one
/// size for the whole strike.
pub fn cblc_cbdt19(ppem: u8, bit_depth: u8, specs: &[SbitImageSpec]) -> (Vec<u8>, Vec<u8>) {
    assert!(!specs.is_empty(), "a strike needs at least one glyph");
    let image_size = 4 + specs[0].png.len();
    assert!(
        specs
            .iter()
            .all(|spec| 4 + spec.png.len() == image_size),
        "format 19 needs a uniform image size"
    );
    let start_gid = specs[0].gid;
    let end_gid = specs[specs.len() - 1].gid;

    let mut cbdt = Vec::new();
    push_u32(&mut cbdt, 0x0003_0000); // CBDT version
    for spec in specs {
        push_u32(&mut cbdt, spec.png.len() as u32);
        cbdt.extend_from_slice(&spec.png);
    }

    let first = &specs[0];
    let mut entries = Vec::new();
    push_u32(&mut entries, image_size as u32);
    entries.push(first.height);
    entries.push(first.width);
    entries.push(first.bearing_x as u8);
    entries.push(first.bearing_y as u8);
    entries.push(first.advance);
    entries.extend_from_slice(&[0u8; 3]); // vertical metrics
    push_u32(&mut entries, specs.len() as u32);
    for spec in specs {
        push_u16(&mut entries, spec.gid);
    }

    let cblc = cblc_skeleton(ppem, bit_depth, start_gid, end_gid, 19, 19, &entries);
    (cblc, cbdt)
}

/// Appends one PNG chunk (length, type, data, CRC) to `out`.
pub fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    push_u32(out, data.len() as u32);
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    push_u32(out, crate::png::crc32(&crc_input));
}

/// Wraps `raw` in a zlib stream of stored (uncompressed) deflate
/// blocks with an Adler-32 trailer.
pub fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::from([0x78u8, 0x01]); // deflate, 32K window
    if raw.is_empty() {
        out.push(0x01); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0xFFFFu16.to_le_bytes());
    } else {
        let mut offset = 0usize;
        while offset < raw.len() {
            let end = (offset + usize::from(u16::MAX)).min(raw.len());
            let last = end == raw.len();
            out.push(if last { 0x01 } else { 0x00 });
            let length = (end - offset) as u16;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&(!length).to_le_bytes());
            out.extend_from_slice(&raw[offset..end]);
            offset = end;
        }
    }
    out.extend_from_slice(&codevar_truetype_gzip::adler32(1, raw).to_be_bytes());
    out
}

/// Builds a non-interlaced RGBA8 PNG from raw rows (filter 0).
pub fn png_rgba(width: u32, height: u32, rows: &[&[u8]]) -> Vec<u8> {
    let mut raw = Vec::new();
    for row in rows {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut out = Vec::from([0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}
