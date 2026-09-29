//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Deterministic in-memory TrueType font used by the engine tests.
//!
//! [`SyntheticFont::build`] assembles a complete, checksum-correct sfnt
//! file containing four glyphs:
//!
//! | Id | Shape | Purpose |
//! |----|-------|---------|
//! | 0 | quadratic blob | off-curve point handling, composite source |
//! | 1 | axis-aligned square with a hinting program | rasterizer and interpreter tests |
//! | 2 | empty (space) | empty-glyph paths |
//! | 3 | composite of 1 + 0 | composite outline resolution |
//!
//! The font also carries `cvt`, `fpgm`, `prep`, `gasp`, `name`, `OS/2`
//! and (optionally) legacy `kern` tables so parser coverage spans the
//! whole surface this crate implements.

use alloc::vec::Vec;

/// Builder for the deterministic test font.
#[derive(Debug, Clone)]
pub struct SyntheticFont {
    with_kern: bool,
    with_cmap12: bool,
    without_cmap: bool,
}

impl Default for SyntheticFont {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticFont {
    /// Creates a builder with the standard configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            with_kern: false,
            with_cmap12: false,
            without_cmap: false,
        }
    }

    /// Adds a legacy `kern` format-0 table with pair (1, 1) = −40.
    #[must_use]
    pub fn with_kern(mut self) -> Self {
        self.with_kern = true;
        self
    }

    /// Adds a `cmap` format-12 subtable alongside the format-4 one.
    #[must_use]
    pub fn with_cmap12(mut self) -> Self {
        self.with_cmap12 = true;
        self
    }

    /// Omits the `cmap` table entirely.
    #[must_use]
    pub fn without_cmap(mut self) -> Self {
        self.without_cmap = true;
        self
    }

    /// Builds the sfnt file bytes, including a correct
    /// `checkSumAdjustment`.
    pub fn build(&self) -> Vec<u8> {
        let head = build_head();
        let hhea = build_hhea();
        let maxp = build_maxp();
        let hmtx = build_hmtx();
        let loca_glyf = build_loca_glyf();
        let (loca, glyf) = loca_glyf;
        let cvt = build_cvt();
        let fpgm = build_fpgm();
        let prep = build_prep();
        let os2 = build_os2();
        let gasp = build_gasp();
        let name = build_name();
        let cmap = if self.without_cmap {
            None
        } else {
            Some(build_cmap(self.with_cmap12))
        };
        let kern = if self.with_kern { Some(build_kern()) } else { None };

        let mut tables: Vec<(&[u8; 4], Vec<u8>)> = vec![
            (b"cvt ", cvt),
            (b"fpgm", fpgm),
            (b"glyf", glyf),
            (b"gasp", gasp),
            (b"head", head),
            (b"hhea", hhea),
            (b"hmtx", hmtx),
            (b"loca", loca),
            (b"maxp", maxp),
            (b"name", name),
            (b"OS/2", os2),
            (b"prep", prep),
        ];
        if let Some(cmap) = cmap {
            tables.push((b"cmap", cmap));
        }
        if let Some(kern) = kern {
            tables.push((b"kern", kern));
        }
        tables.sort_by_key(|(tag, _)| *tag);

        assemble_sfnt(&tables)
    }
}

/// Appends a big-endian `u16`.
fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends a big-endian `i16`.
fn push_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends a big-endian `u32`.
fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// sfnt table checksum: wrapping sum of big-endian `u32`s, zero-padded.
fn table_checksum(data: &[u8]) -> u32 {
    let mut sum = 0u32;
    let mut chunks = data.chunks_exact(4);
    for chunk in &mut chunks {
        sum = sum.wrapping_add(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut buf = [0u8; 4];
        buf[..rem.len()].copy_from_slice(rem);
        sum = sum.wrapping_add(u32::from_be_bytes(buf));
    }
    sum
}

/// Assembles the offset table + directory around `tables`.
fn assemble_sfnt(tables: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let num = u16::try_from(tables.len()).unwrap_or(0);
    let entry_selector = if num == 0 {
        0
    } else {
        15 - num.leading_zeros() as u16
    };
    let search_range = 16u16 << entry_selector;
    let range_shift = num * 16 - search_range;

    let mut out = Vec::new();
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, num);
    push_u16(&mut out, search_range);
    push_u16(&mut out, entry_selector);
    push_u16(&mut out, range_shift);

    let dir_start = out.len();
    let dir_size = tables.len() * 16;

    // Pre-compute padded payload offsets (directory records themselves
    // must stay contiguous).
    let mut starts = Vec::with_capacity(tables.len());
    let mut offset = dir_start + dir_size;
    let mut head_offset = 0usize;
    for (tag, payload) in tables {
        while offset % 4 != 0 {
            offset += 1;
        }
        if **tag == *b"head" {
            head_offset = offset;
        }
        starts.push(offset);
        offset += payload.len();
    }

    for ((tag, payload), start) in tables.iter().zip(&starts) {
        push_u32(&mut out, u32::from_be_bytes(**tag));
        push_u32(&mut out, table_checksum(payload));
        push_u32(&mut out, u32::try_from(*start).unwrap_or(0));
        push_u32(&mut out, u32::try_from(payload.len()).unwrap_or(0));
    }

    for ((_, payload), start) in tables.iter().zip(&starts) {
        while out.len() < *start {
            out.push(0);
        }
        out.extend_from_slice(payload);
    }
    while out.len() % 4 != 0 {
        out.push(0);
    }

    // Fill in head.checkSumAdjustment so the whole file hashes to
    // 0xB1B0AFBA (sfnt requirement).
    if head_offset != 0 {
        let adj_pos = head_offset + 8;
        out[adj_pos..adj_pos + 4].copy_from_slice(&0u32.to_be_bytes());
        let sum = table_checksum(&out);
        let adjustment = 0xB1B0_AFBA_u32.wrapping_sub(sum);
        out[adj_pos..adj_pos + 4].copy_from_slice(&adjustment.to_be_bytes());
    }
    out
}

/// `head` table: upem 1000, long loca = 0, valid magic.
fn build_head() -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x0001_0000); // version
    push_u32(&mut out, 0x0001_0000); // fontRevision 1.0
    push_u32(&mut out, 0); // checkSumAdjustment (patched later)
    push_u32(&mut out, 0x5F0F_3CF5); // magicNumber
    push_u16(&mut out, 0x000B); // flags: baseline y0, lsb x0, advance x-max
    push_u16(&mut out, 1000); // unitsPerEm
    push_u32(&mut out, 0); // created hi
    push_u32(&mut out, 0); // created lo
    push_u32(&mut out, 0); // modified hi
    push_u32(&mut out, 0); // modified lo
    push_i16(&mut out, 50); // xMin
    push_i16(&mut out, -200); // yMin
    push_i16(&mut out, 1200); // xMax
    push_i16(&mut out, 700); // yMax
    push_u16(&mut out, 0); // macStyle
    push_i16(&mut out, 8); // lowestRecPPEM
    push_i16(&mut out, 2); // fontDirectionHint
    push_i16(&mut out, 0); // indexToLocFormat: short
    push_i16(&mut out, 0); // glyphDataFormat
    out
}

/// `hhea` table: ascender 800, three long metrics.
fn build_hhea() -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x0001_0000);
    push_i16(&mut out, 800); // ascender
    push_i16(&mut out, -200); // descender
    push_i16(&mut out, 90); // lineGap
    push_u16(&mut out, 1200); // advanceWidthMax
    push_i16(&mut out, 0); // minLSB
    push_i16(&mut out, 0); // minRSB
    push_i16(&mut out, 1200); // xMaxExtent
    push_i16(&mut out, 1); // caretSlopeRise
    push_i16(&mut out, 0); // caretSlopeRun
    push_i16(&mut out, 0); // caretOffset
    for _ in 0..4 {
        push_i16(&mut out, 0); // reserved
    }
    push_i16(&mut out, 0); // metricDataFormat
    push_u16(&mut out, 4); // numberOfHMetrics
    out
}

/// `maxp` table version 1.0 with interpreter budgets.
fn build_maxp() -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, 4); // numGlyphs
    push_u16(&mut out, 16); // maxPoints
    push_u16(&mut out, 4); // maxContours
    push_u16(&mut out, 32); // maxCompositePoints
    push_u16(&mut out, 8); // maxCompositeContours
    push_u16(&mut out, 2); // maxZones
    push_u16(&mut out, 16); // maxTwilightPoints
    push_u16(&mut out, 32); // maxStorage
    push_u16(&mut out, 8); // maxFunctionDefs
    push_u16(&mut out, 2); // maxInstructionDefs
    push_u16(&mut out, 64); // maxStackElements
    push_u16(&mut out, 128); // maxSizeOfInstructions
    push_u16(&mut out, 8); // maxComponentElements
    push_u16(&mut out, 2); // maxComponentDepth
    out
}

/// `hmtx`: four long metrics (glyphs 0..=3).
fn build_hmtx() -> Vec<u8> {
    let mut out = Vec::new();
    for (advance, lsb) in [(600u16, 0i16), (600, 50), (250, 0), (900, 0)] {
        push_u16(&mut out, advance);
        push_i16(&mut out, lsb);
    }
    out
}

/// Encodes one simple glyph; `points` are `(x, y, on_curve)`.
fn encode_simple_glyph(
    points: &[(i16, i16, bool)],
    end_pts: &[u16],
    instructions: &[u8],
    bbox: (i16, i16, i16, i16),
) -> Vec<u8> {
    let mut out = Vec::new();
    push_i16(&mut out, i16::try_from(end_pts.len()).unwrap_or(0));
    push_i16(&mut out, bbox.0);
    push_i16(&mut out, bbox.1);
    push_i16(&mut out, bbox.2);
    push_i16(&mut out, bbox.3);
    for &e in end_pts {
        push_u16(&mut out, e);
    }
    push_u16(&mut out, u16::try_from(instructions.len()).unwrap_or(0));
    out.extend_from_slice(instructions);

    // Flags: on-curve bit only; explicit x/y deltas (no short/same
    // encoding) to keep the fixture readable.
    let mut flags = Vec::with_capacity(points.len());
    for &(_, _, on) in points {
        flags.push(if on { 0x01 } else { 0x00 });
    }
    out.extend_from_slice(&flags);

    let mut prev = 0i16;
    for &(x, _, _) in points {
        push_i16(&mut out, x.wrapping_sub(prev));
        prev = x;
    }
    let mut prev = 0i16;
    for &(_, y, _) in points {
        push_i16(&mut out, y.wrapping_sub(prev));
        prev = y;
    }
    out
}

/// Encodes a composite glyph from `(glyph_id, dx, dy, more)` parts.
fn encode_composite_glyph(
    parts: &[(u16, i16, i16, bool)],
    bbox: (i16, i16, i16, i16),
    instructions: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    push_i16(&mut out, -1);
    push_i16(&mut out, bbox.0);
    push_i16(&mut out, bbox.1);
    push_i16(&mut out, bbox.2);
    push_i16(&mut out, bbox.3);
    for (i, &(glyph, dx, dy, more)) in parts.iter().enumerate() {
        let mut flags = 0x0001u16 | 0x0002; // words + xy values
        if more {
            flags |= 0x0020;
        }
        if i + 1 == parts.len() && !instructions.is_empty() {
            flags |= 0x0100; // WE_HAVE_INSTRUCTIONS
        }
        push_u16(&mut out, flags);
        push_u16(&mut out, glyph);
        push_i16(&mut out, dx);
        push_i16(&mut out, dy);
    }
    if !instructions.is_empty() {
        push_u16(&mut out, u16::try_from(instructions.len()).unwrap_or(0));
        out.extend_from_slice(instructions);
    }
    out
}

/// Builds `glyf` plus its short-format `loca` (even offsets only).
fn build_loca_glyf() -> (Vec<u8>, Vec<u8>) {
    // Glyph 0: quadratic blob — two on-curve, two off-curve points.
    let g0 = encode_simple_glyph(
        &[
            (100, 0, true),
            (300, 600, false),
            (500, 0, true),
            (300, -200, false),
        ],
        &[3],
        &[],
        (100, -200, 500, 600),
    );
    // Glyph 1: axis-aligned square with a tiny hinting program that
    // rounds point 0 along x (SVTCA[x]; PUSHB 0; MDAP[rnd]).
    let g1 = encode_simple_glyph(
        &[(50, 0, true), (550, 0, true), (550, 700, true), (50, 700, true)],
        &[3],
        &[0x01, 0xB0, 0x00, 0x2F],
        (50, 0, 550, 700),
    );
    // Glyph 2: empty (space).
    let g2: Vec<u8> = Vec::new();
    // Glyph 3: composite = square + blob shifted right by 700.
    let g3 = encode_composite_glyph(&[(1, 0, 0, true), (0, 700, 0, false)], (50, -200, 1200, 700), &[]);

    let mut glyf = Vec::new();
    let mut offsets = Vec::new();
    for glyph in [&g0[..], &g1[..], &g2[..], &g3[..]] {
        while glyf.len() % 4 != 0 {
            glyf.push(0);
        }
        offsets.push(u16::try_from(glyf.len() / 2).unwrap_or(0));
        glyf.extend_from_slice(glyph);
    }
    while glyf.len() % 4 != 0 {
        glyf.push(0);
    }
    offsets.push(u16::try_from(glyf.len() / 2).unwrap_or(0));

    let mut loca = Vec::new();
    for offset in offsets {
        push_u16(&mut loca, offset);
    }
    (loca, glyf)
}

/// `cvt ` with three font-unit entries.
fn build_cvt() -> Vec<u8> {
    let mut out = Vec::new();
    for value in [20i16, 40, 60] {
        push_i16(&mut out, value);
    }
    out
}

/// Font program defining function 0 (stores 42 into storage[0]).
fn build_fpgm() -> Vec<u8> {
    // PUSHB[0] 0; FDEF; PUSHB[1] 42 0; WS; ENDF
    vec![0xB0, 0x00, 0x2C, 0xB1, 0x2A, 0x00, 0x42, 0x2D]
}

/// Pre-program: round to grid.
fn build_prep() -> Vec<u8> {
    vec![0x18]
}

/// Minimal `OS/2` version 2 table.
fn build_os2() -> Vec<u8> {
    let mut out = Vec::new();
    push_u16(&mut out, 2); // version
    push_i16(&mut out, 500); // xAvgCharWidth
    push_u16(&mut out, 400); // usWeightClass
    push_u16(&mut out, 5); // usWidthClass
    push_u16(&mut out, 0); // fsType
    out.extend_from_slice(&[0; 10]); // ySubscript*
    out.extend_from_slice(&[0; 10]); // ySuperscript*
    out.extend_from_slice(&[0; 10]); // yStrikeout*
    push_i16(&mut out, 0); // sFamilyClass
    out.extend_from_slice(&[0; 10]); // panose
    out.extend_from_slice(&[0; 16]); // ulUnicodeRange
    out.extend_from_slice(b"CVAR"); // achVendID
    push_u16(&mut out, 0x0040); // fsSelection REGULAR
    push_u16(&mut out, 0x0020); // usFirstCharIndex
    push_u16(&mut out, 0x0042); // usLastCharIndex
    push_i16(&mut out, 800); // sTypoAscender
    push_i16(&mut out, -200); // sTypoDescender
    push_i16(&mut out, 90); // sTypoLineGap
    push_u16(&mut out, 800); // usWinAscent
    push_u16(&mut out, 200); // usWinDescent
    out.extend_from_slice(&[0; 8]); // ulCodePageRange (v1)
    push_i16(&mut out, 520); // sxHeight (v2)
    push_i16(&mut out, 700); // sCapHeight (v2)
    push_u16(&mut out, 0x0020); // usDefaultChar
    push_u16(&mut out, 0x0020); // usBreakChar
    push_u16(&mut out, 2); // usMaxContext
    out
}

/// `gasp`: smoothing enabled up to 40 ppem.
fn build_gasp() -> Vec<u8> {
    let mut out = Vec::new();
    push_u16(&mut out, 1); // version
    push_u16(&mut out, 1); // numRanges
    push_u16(&mut out, 40); // maxRangePPEM
    push_u16(&mut out, 0x0003); // grayscale + symmetric grid-fit
    out
}

/// `name` table with Windows English family/full/PostScript names.
fn build_name() -> Vec<u8> {
    let strings: [(u16, &str); 4] = [
        (1, "Glyphar Test"),
        (2, "Regular"),
        (4, "Glyphar Test Regular"),
        (6, "GlypharTest-Regular"),
    ];
    let mut storage = Vec::new();
    let mut records = Vec::new();
    for (id, text) in strings {
        let start = u16::try_from(storage.len()).unwrap_or(0);
        for unit in text.encode_utf16() {
            push_u16(&mut storage, unit);
        }
        let len = u16::try_from(storage.len()).unwrap_or(0) - start;
        // platform 3 (Windows), encoding 1 (BMP), language 0x0409.
        push_u16(&mut records, 3);
        push_u16(&mut records, 1);
        push_u16(&mut records, 0x0409);
        push_u16(&mut records, id);
        push_u16(&mut records, start);
        push_u16(&mut records, len);
    }
    let mut out = Vec::new();
    push_u16(&mut out, 0); // format
    push_u16(&mut out, u16::try_from(strings.len()).unwrap_or(0));
    push_u16(&mut out, u16::try_from(6 + records.len()).unwrap_or(0));
    out.extend_from_slice(&records);
    out.extend_from_slice(&storage);
    out
}

/// `cmap` with Windows BMP format 4 (and optional format 12).
fn build_cmap(with_format12: bool) -> Vec<u8> {
    let sub4 = build_cmap_format4();
    let sub12 = if with_format12 {
        Some(build_cmap_format12())
    } else {
        None
    };

    let mut encoding_records = Vec::new();
    // platform 3 / encoding 1 (BMP) -> format 4.
    push_u16(&mut encoding_records, 3);
    push_u16(&mut encoding_records, 1);
    let mut offset = 4 + 8 + if sub12.is_some() { 8 } else { 0 };
    push_u32(&mut encoding_records, offset as u32);
    if sub12.is_some() {
        // platform 3 / encoding 10 (full Unicode) -> format 12.
        push_u16(&mut encoding_records, 3);
        push_u16(&mut encoding_records, 10);
        offset += sub4.len();
        push_u32(&mut encoding_records, offset as u32);
    }

    let count = 1 + usize::from(sub12.is_some());
    let mut out = Vec::new();
    push_u16(&mut out, 0); // version
    push_u16(&mut out, u16::try_from(count).unwrap_or(0));
    out.extend_from_slice(&encoding_records);
    out.extend_from_slice(&sub4);
    if let Some(sub12) = sub12 {
        out.extend_from_slice(&sub12);
    }
    out
}

/// Format 4 subtable mapping U+0020→2, U+0041→1, U+0042→0.
fn build_cmap_format4() -> Vec<u8> {
    let segments: [(u16, u16, u16); 4] = [
        (0x0020, 0x0020, (2u16.wrapping_sub(0x0020))),
        (0x0041, 0x0041, 1u16.wrapping_sub(0x0041)),
        (0x0042, 0x0042, 0u16.wrapping_sub(0x0042)),
        (0xFFFF, 0xFFFF, 1),
    ];
    let seg_count = segments.len() as u16;
    let entry_selector = 15 - seg_count.leading_zeros() as u16;
    let search_range = 2 * (1u16 << entry_selector);
    let range_shift = 2 * seg_count - search_range;

    let mut out = Vec::new();
    push_u16(&mut out, 4); // format
    push_u16(&mut out, 0); // length (patched)
    push_u16(&mut out, 0); // language
    push_u16(&mut out, seg_count * 2);
    push_u16(&mut out, search_range);
    push_u16(&mut out, entry_selector);
    push_u16(&mut out, range_shift);
    for &(_, end, _) in &segments {
        push_u16(&mut out, end);
    }
    push_u16(&mut out, 0); // reservedPad
    for &(start, _, _) in &segments {
        push_u16(&mut out, start);
    }
    for &(_, _, delta) in &segments {
        push_u16(&mut out, delta);
    }
    for _ in 0..seg_count {
        push_u16(&mut out, 0); // idRangeOffset
    }
    let len = u16::try_from(out.len()).unwrap_or(0);
    out[2..4].copy_from_slice(&len.to_be_bytes());
    out
}

/// Format 12 subtable mapping U+0020..U+0042 to glyph 1 (sparse groups).
fn build_cmap_format12() -> Vec<u8> {
    let groups: [(u32, u32, u32); 3] = [(0x20, 0x20, 2), (0x41, 0x41, 1), (0x42, 0x42, 0)];
    let mut out = Vec::new();
    push_u16(&mut out, 12); // format
    push_u16(&mut out, 0); // reserved
    push_u32(&mut out, 16 + 12 * groups.len() as u32); // length
    push_u32(&mut out, 0); // language
    push_u32(&mut out, u32::try_from(groups.len()).unwrap_or(0));
    for (start, end, gid) in groups {
        push_u32(&mut out, start);
        push_u32(&mut out, end);
        push_u32(&mut out, gid);
    }
    out
}

/// Legacy `kern` format-0 table with the single pair (1, 1) = −40.
fn build_kern() -> Vec<u8> {
    let mut subtable = Vec::new();
    push_u16(&mut subtable, 0); // subtable version
    push_u16(&mut subtable, 0); // length (patched)
    push_u16(&mut subtable, 0x0001); // coverage: horizontal, format 0
    push_u16(&mut subtable, 1); // nPairs
    push_u16(&mut subtable, 6); // searchRange
    push_u16(&mut subtable, 0); // entrySelector
    push_u16(&mut subtable, 0); // rangeShift
    push_u16(&mut subtable, 1); // left
    push_u16(&mut subtable, 1); // right
    push_i16(&mut subtable, -40); // value
    let len = u16::try_from(subtable.len()).unwrap_or(0);
    subtable[2..4].copy_from_slice(&len.to_be_bytes());

    let mut out = Vec::new();
    push_u16(&mut out, 0); // table version
    push_u16(&mut out, 1); // nTables
    out.extend_from_slice(&subtable);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_file::FontFile;

    #[test]
    fn synthetic_font_parses_and_verifies() {
        let bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&bytes).expect("synthetic font");
        assert_eq!(font.num_glyphs(), 4);
        font.verify_checksums().expect("table checksums");
        font.verify_font_checksum()
            .expect("file checksum");
        assert_eq!(font.family_name().as_deref(), Some("Glyphar Test"));
        assert_eq!(font.advance_width(2), 250);
        assert!(font.glyph_range(2).expect("id 2").is_none());
        assert!(font.glyph_range(3).expect("id 3").is_some());
    }

    #[test]
    fn kern_variant_adds_pairs() {
        let bytes = SyntheticFont::new().with_kern().build();
        let font = FontFile::parse(&bytes).expect("synthetic font");
        assert_eq!(font.kerning(1, 1), -40);
        assert_eq!(font.kerning(0, 3), 0);
    }

    #[test]
    fn cmap12_variant_has_two_subtables() {
        let bytes = SyntheticFont::new().with_cmap12().build();
        let font = FontFile::parse(&bytes).expect("synthetic font");
        let cmap = font.table(*b"cmap").expect("cmap payload");
        let count = u16::from_be_bytes([cmap[2], cmap[3]]);
        assert_eq!(count, 2);
    }
}
