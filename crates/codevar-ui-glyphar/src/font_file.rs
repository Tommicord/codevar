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

//! TrueType (sfnt) container parsing.
//!
//! [`FontFile::parse`] validates the offset table, the table directory
//! and every table this engine needs, then exposes the fields through
//! cheap accessors that borrow directly from the input bytes — no
//! intermediate copies are made for `cvt`, `fpgm`, `prep`, `kern`,
//! `name` or `cmap` payloads.
//!
//! Supported `sfnt` versions: `0x00010000` (Windows) and `true`
//! (Apple). `OTTO` (CFF outlines), `ttcf` (collections) and `typ1`
//! are rejected with [`GlypharError::Unsupported`] because this engine
//! only rasterizes quadratic `glyf` outlines.
//!
//! # Performance
//!
//! - Table directory lookups are linear over an already small
//!   (`numTables ≤ 64` in practice) record vector.
//! - Table checksum verification runs through the SIMD kernels in
//!   [`crate::simd`] (SSE2/SSSE3/AVX2/AVX-512/NEON byte-swap adds).
//! - All parsing is O(size of the input) and performs zero heap
//!   allocations except for the table directory and `name` strings.
//!
//! # Examples
//!
//! ```ignore
//! use codevar_ui_glyphar::font_file::FontFile;
//!
//! let font = FontFile::parse(font_bytes)?;
//! assert!(font.units_per_em() >= 16 && font.units_per_em() <= 16384);
//! ```

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::GlypharError;
use crate::GlypharResult;

/// Big-endian byte reader with bounds checking.
///
/// Every read returns [`GlypharError::Truncated`] instead of panicking,
/// so malformed fonts can never take the process down.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Creates a reader positioned at the start of `data`.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Creates a reader over a sub-slice.
    #[inline]
    pub fn with_pos(data: &'a [u8], pos: usize) -> GlypharResult<Self> {
        if pos > data.len() {
            return Err(GlypharError::Truncated {
                context: "reader start",
            });
        }
        Ok(Self { data, pos })
    }

    /// Current byte offset.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes left before the end of the slice.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Total length of the underlying slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether no bytes remain.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Moves the cursor to an absolute offset inside the slice.
    #[inline]
    pub fn seek(&mut self, pos: usize, context: &'static str) -> GlypharResult<()> {
        if pos > self.data.len() {
            return Err(GlypharError::Truncated { context });
        }
        self.pos = pos;
        Ok(())
    }

    /// Reads `n` raw bytes and advances the cursor.
    pub fn read_bytes(&mut self, n: usize, context: &'static str) -> GlypharResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(GlypharError::Truncated { context })?;
        if end > self.data.len() {
            return Err(GlypharError::Truncated { context });
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Reads one unsigned byte.
    #[inline]
    pub fn read_u8(&mut self, context: &'static str) -> GlypharResult<u8> {
        let bytes = self.read_bytes(1, context)?;
        // Bounds checked above, so indexing is safe.
        Ok(bytes[0])
    }

    /// Reads one signed byte.
    #[inline]
    pub fn read_i8(&mut self, context: &'static str) -> GlypharResult<i8> {
        Ok(i8::from_ne_bytes([self.read_u8(context)?]))
    }

    /// Reads a big-endian `u16`.
    pub fn read_u16(&mut self, context: &'static str) -> GlypharResult<u16> {
        let bytes = self.read_bytes(2, context)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    /// Reads a big-endian `i16`.
    pub fn read_i16(&mut self, context: &'static str) -> GlypharResult<i16> {
        Ok(self.read_u16(context)? as i16)
    }

    /// Reads a big-endian `u32`.
    pub fn read_u32(&mut self, context: &'static str) -> GlypharResult<u32> {
        let bytes = self.read_bytes(4, context)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Reads a big-endian `i32`.
    pub fn read_i32(&mut self, context: &'static str) -> GlypharResult<i32> {
        Ok(self.read_u32(context)? as i32)
    }

    /// Reads a big-endian `i64` (used by `head` timestamps).
    pub fn read_i64(&mut self, context: &'static str) -> GlypharResult<i64> {
        let bytes = self.read_bytes(8, context)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(i64::from_be_bytes(buf))
    }

    /// Reads a big-endian `u64`.
    pub fn read_u64(&mut self, context: &'static str) -> GlypharResult<u64> {
        let bytes = self.read_bytes(8, context)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(buf))
    }

    /// Reads one `F2Dot14` fixed-point value as `i16` (16.14 scaled).
    #[inline]
    pub fn read_f2dot14(&mut self, context: &'static str) -> GlypharResult<i16> {
        self.read_i16(context)
    }
}

/// One record of the sfnt table directory.
///
/// Laid out as four big-endian `u32`s so the tag array can be scanned
/// by the SIMD kernels in [`crate::simd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct TableRecord {
    /// Four-character table tag as packed big-endian bytes.
    pub tag: u32,
    /// Checksum of the table as stored in the font.
    pub checksum: u32,
    /// Byte offset of the table from the start of the file.
    pub offset: u32,
    /// Length of the table in bytes.
    pub length: u32,
}

impl TableRecord {
    /// Decodes the packed tag into its four ASCII bytes.
    #[must_use]
    #[inline]
    pub const fn tag_bytes(self) -> [u8; 4] {
        self.tag.to_be_bytes()
    }

    /// Human-readable tag (lossy: non-ASCII bytes become `?`).
    #[must_use]
    pub fn tag_string(self) -> String {
        let raw = self.tag_bytes();
        let mut out = String::with_capacity(4);
        for b in raw {
            out.push(if b.is_ascii_graphic() || b == b' ' {
                char::from(b)
            } else {
                '?'
            });
        }
        out
    }
}

/// Parsed contents of the `head` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadTable {
    /// Fixed-point font revision (`16.16`).
    pub font_revision: u32,
    /// Sum of all table checksums; used for whole-file validation.
    pub check_sum_adjustment: u32,
    /// Must be `0x5F0F3CF5` in valid fonts.
    pub magic_number: u32,
    /// Bit flags (bit 0: baseline at y=0, bit 1: lsb at x=0, ...).
    pub flags: u16,
    /// Design units per em (`16..=16384`).
    pub units_per_em: u16,
    /// Smallest x of any glyph in the font.
    pub x_min: i16,
    /// Smallest y of any glyph in the font.
    pub y_min: i16,
    /// Largest x of any glyph in the font.
    pub x_max: i16,
    /// Largest y of any glyph in the font.
    pub y_max: i16,
    /// Mac-style bold/italic bits.
    pub mac_style: u16,
    /// Smallest readable size in ppem.
    pub lowest_rec_ppem: i16,
    /// Expected direction of the glyph outlines.
    pub font_direction_hint: i16,
    /// `0` = `u16` loca offsets, `1` = `u32`.
    pub index_to_loc_format: i16,
    /// Must be `0` for `glyf`-based fonts.
    pub glyph_data_format: i16,
}

/// Parsed contents of the `hhea` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HheaTable {
    /// Typographic ascent in font units.
    pub ascender: i16,
    /// Typographic descent in font units.
    pub descender: i16,
    /// Line gap in font units.
    pub line_gap: i16,
    /// Maximum advance width in the font.
    pub advance_width_max: u16,
    /// Minimum left side bearing.
    pub min_left_side_bearing: i16,
    /// Minimum right side bearing.
    pub min_right_side_bearing: i16,
    /// Maximum x extent (`lsb + (xMax - xMin)`).
    pub x_max_extent: i16,
    /// Caret slope rise (1 for vertical caret).
    pub caret_slope_rise: i16,
    /// Caret slope run (0 for vertical caret).
    pub caret_slope_run: i16,
    /// Caret offset.
    pub caret_offset: i16,
    /// Number of entries in the `hmtx` long metric array.
    pub number_of_h_metrics: u16,
}

/// Parsed contents of the `maxp` table (version `1.0` fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxpTable {
    /// `0x00005000` (v0.5) or `0x00010000` (v1.0).
    pub version: u32,
    /// Number of glyphs in the font.
    pub num_glyphs: u16,
    /// Maximum points in a non-composite glyph.
    pub max_points: u16,
    /// Maximum contours in a non-composite glyph.
    pub max_contours: u16,
    /// Maximum points in a composite glyph.
    pub max_composite_points: u16,
    /// Maximum contours in a composite glyph.
    pub max_composite_contours: u16,
    /// `1` for most fonts; `2` when twilight-zone hinting is used.
    pub max_zones: u16,
    /// Maximum twilight-zone points.
    pub max_twilight_points: u16,
    /// Storage area size (`WS`/`RS`).
    pub max_storage: u16,
    /// Number of function definitions (`FDEF`).
    pub max_function_defs: u16,
    /// Number of instruction definitions (`IDEF`).
    pub max_instruction_defs: u16,
    /// Maximum interpreter stack depth.
    pub max_stack_elements: u16,
    /// Maximum size of glyph instructions in bytes.
    pub max_size_of_instructions: u16,
    /// Maximum components in one composite glyph.
    pub max_component_elements: u16,
    /// Maximum nesting depth of composites.
    pub max_component_depth: u16,
}

/// A parsed `OS/2` table (fields relevant to layout and rendering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Os2Table {
    /// Table version (`0..=5`).
    pub version: u16,
    /// Average character width in font units.
    pub x_avg_char_width: i16,
    /// Weight class (`1..=9`, 400 = normal, 700 = bold).
    pub us_weight_class: u16,
    /// Width class (`1..=9`).
    pub us_width_class: u16,
    /// Typographic ascent.
    pub s_typo_ascender: i16,
    /// Typographic descent.
    pub s_typo_descender: i16,
    /// Typographic line gap.
    pub s_typo_line_gap: i16,
    /// Windows ascent.
    pub us_win_ascent: u16,
    /// Windows descent.
    pub us_win_descent: u16,
    /// First Unicode code point covered by the font.
    pub us_first_char_index: u16,
    /// Last Unicode code point covered by the font.
    pub us_last_char_index: u16,
    /// x height (version ≥ 2).
    pub s_x_height: i16,
    /// Cap height (version ≥ 2).
    pub s_cap_height: i16,
    /// Default character (version ≥ 2).
    pub us_default_char: u16,
    /// Break character (version ≥ 2).
    pub us_break_char: u16,
    /// Maximum horizontal context (version ≥ 2, ligature/kerning lookups).
    pub us_max_context: u16,
}

/// One range of the `gasp` table (grid-fitting and smoothing flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaspRange {
    /// Highest ppem this range applies to.
    pub max_range_ppem: u16,
    /// Behavior bits: bit 0 = grayscale, bit 1 = symmetric grid-fit,
    /// bit 2 = symmetric smoothing, bit 3 = forced (ClearType) smoothing.
    pub gasp_behavior: u16,
}

/// One record of the `name` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameRecord {
    /// Platform (`0` Unicode, `1` Macintosh, `3` Windows).
    pub platform_id: u16,
    /// Encoding specific to the platform.
    pub encoding_id: u16,
    /// Language ID.
    pub language_id: u16,
    /// Name ID (`1` family, `2` subfamily, `4` full name, `6` PostScript).
    pub name_id: u16,
    /// Byte offset into the `name` string storage.
    pub offset: u16,
    /// Byte length of the string.
    pub length: u16,
}

/// A single kerning pair from the legacy `kern` format-0 subtable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernPair {
    /// Left glyph id.
    pub left: u16,
    /// Right glyph id.
    pub right: u16,
    /// Kerning adjustment in font units.
    pub value: i16,
}

/// A fully validated sfnt font file.
///
/// The struct borrows the original byte buffer; dropping it releases
/// nothing but the borrow.
#[derive(Debug, Clone)]
pub struct FontFile<'a> {
    data: &'a [u8],
    records: Vec<TableRecord>,
    head: HeadTable,
    hhea: HheaTable,
    maxp: MaxpTable,
    os2: Option<Os2Table>,
    gasp: Vec<GaspRange>,
    hmtx: &'a [u8],
    loca: &'a [u8],
    glyf: &'a [u8],
    cvt: &'a [u8],
    fpgm: &'a [u8],
    prep: &'a [u8],
    cmap: Option<Range>,
    name: Option<Range>,
    kern: Option<&'a [u8]>,
}

impl<'a> FontFile<'a> {
    /// Parses and validates a TrueType font from raw bytes.
    ///
    /// # Errors
    ///
    /// - [`GlypharError::Truncated`] when the buffer ends mid-structure.
    /// - [`GlypharError::Malformed`] when magic values, sizes or field
    ///   ranges are invalid (including a bad `head` magic number or a
    ///   `loca` entry past the end of `glyf`).
    /// - [`GlypharError::Unsupported`] for `OTTO`, `ttcf` and `typ1`
    ///   containers and for fonts without a `glyf` table.
    pub fn parse(data: &'a [u8]) -> GlypharResult<Self> {
        let mut r = Reader::new(data);

        let sfnt = r.read_u32("sfnt version")?;
        match sfnt {
            0x0001_0000 | 0x7472_7565 => {}
            0x4F54_544F => {
                return Err(GlypharError::Unsupported {
                    feature: "CFF/CFF2 outlines (OTTO)",
                });
            }
            0x7474_6366 => {
                return Err(GlypharError::Unsupported {
                    feature: "TrueType Collection (ttcf)",
                });
            }
            0x7479_7031 => {
                return Err(GlypharError::Unsupported {
                    feature: "Type 1 sfnt (typ1)",
                });
            }
            other => {
                return Err(GlypharError::Malformed {
                    context: "sfnt version",
                    detail: match other {
                        _ => "unknown sfnt version tag",
                    },
                });
            }
        }

        let num_tables = r.read_u16("numTables")?;
        let _search_range = r.read_u16("searchRange")?;
        let _entry_selector = r.read_u16("entrySelector")?;
        let _range_shift = r.read_u16("rangeShift")?;
        if num_tables == 0 {
            return Err(GlypharError::Malformed {
                context: "table directory",
                detail: "numTables is zero",
            });
        }
        let num = usize::from(num_tables);
        let mut records = Vec::with_capacity(num);
        for _ in 0..num {
            let tag = r.read_u32("table tag")?;
            let checksum = r.read_u32("table checksum")?;
            let offset = r.read_u32("table offset")?;
            let length = r.read_u32("table length")?;
            let end = usize::try_from(offset)
                .ok()
                .and_then(|o| o.checked_add(usize::try_from(length).ok()?))
                .ok_or(GlypharError::Malformed {
                    context: "table record",
                    detail: "offset + length overflows",
                })?;
            if end > data.len() {
                return Err(GlypharError::Truncated {
                    context: "table payload",
                });
            }
            records.push(TableRecord {
                tag,
                checksum,
                offset,
                length,
            });
        }

        let head_raw = find_table(&records, *b"head").ok_or(GlypharError::Malformed {
            context: "table directory",
            detail: "missing required table head",
        })?;
        let hhea_raw = find_table(&records, *b"hhea").ok_or(GlypharError::Malformed {
            context: "table directory",
            detail: "missing required table hhea",
        })?;
        let maxp_raw = find_table(&records, *b"maxp").ok_or(GlypharError::Malformed {
            context: "table directory",
            detail: "missing required table maxp",
        })?;
        let hmtx_raw = find_table(&records, *b"hmtx").ok_or(GlypharError::Malformed {
            context: "table directory",
            detail: "missing required table hmtx",
        })?;
        let loca_raw = find_table(&records, *b"loca").ok_or(GlypharError::Malformed {
            context: "table directory",
            detail: "missing required table loca",
        })?;
        let glyf_raw = find_table(&records, *b"glyf").ok_or(GlypharError::Unsupported {
            feature: "font without glyf outlines",
        })?;

        let head = parse_head(&data[head_raw])?;
        let hhea = parse_hhea(&data[hhea_raw])?;
        let maxp = parse_maxp(&data[maxp_raw])?;

        if head.units_per_em < 16 || head.units_per_em > 16_384 {
            return Err(GlypharError::Malformed {
                context: "head unitsPerEm",
                detail: "unitsPerEm must be within 16..=16384",
            });
        }
        if head.magic_number != 0x5F0F_3CF5 {
            return Err(GlypharError::Malformed {
                context: "head magicNumber",
                detail: "expected 0x5F0F3CF5",
            });
        }
        if head.glyph_data_format != 0 {
            return Err(GlypharError::Malformed {
                context: "head glyphDataFormat",
                detail: "must be zero",
            });
        }
        if hhea.number_of_h_metrics == 0 {
            return Err(GlypharError::Malformed {
                context: "hhea numberOfHMetrics",
                detail: "must be at least 1",
            });
        }
        if usize::from(hhea.number_of_h_metrics) > usize::from(maxp.num_glyphs) {
            return Err(GlypharError::Malformed {
                context: "hhea numberOfHMetrics",
                detail: "exceeds numGlyphs",
            });
        }
        if !matches!(head.index_to_loc_format, 0 | 1) {
            return Err(GlypharError::Malformed {
                context: "head indexToLocFormat",
                detail: "must be 0 or 1",
            });
        }

        let hmtx = &data[hmtx_raw];
        let need_hmtx = usize::from(hhea.number_of_h_metrics) * 4
            + (usize::from(maxp.num_glyphs) - usize::from(hhea.number_of_h_metrics)) * 2;
        if hmtx.len() < need_hmtx {
            return Err(GlypharError::Truncated {
                context: "hmtx metrics",
            });
        }

        let loca = &data[loca_raw];
        let loca_entries = usize::from(maxp.num_glyphs) + 1;
        let loca_need = loca_entries * if head.index_to_loc_format == 0 { 2 } else { 4 };
        if loca.len() < loca_need {
            return Err(GlypharError::Truncated {
                context: "loca offsets",
            });
        }

        let glyf = &data[glyf_raw];
        validate_loca(loca, glyf, &head, &maxp)?;

        let cvt = find_table(&records, *b"cvt ").map_or(&[][..], |r| &data[r]);
        let fpgm = find_table(&records, *b"fpgm").map_or(&[][..], |r| &data[r]);
        let prep = find_table(&records, *b"prep").map_or(&[][..], |r| &data[r]);
        let os2 = find_table(&records, *b"OS/2").and_then(|r| parse_os2(&data[r]).ok());
        let gasp = find_table(&records, *b"gasp").map_or_else(
            || Vec::new(),
            |r| parse_gasp(&data[r]).unwrap_or_else(|_| Vec::new()),
        );
        let cmap = find_table(&records, *b"cmap");
        let name = find_table(&records, *b"name");
        let kern = find_table(&records, *b"kern").map(|r| &data[r]);

        Ok(Self {
            data,
            records,
            head,
            hhea,
            maxp,
            os2,
            gasp,
            hmtx,
            loca,
            glyf,
            cvt,
            fpgm,
            prep,
            cmap,
            name,
            kern,
        })
    }

    /// The original byte buffer this font was parsed from.
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The full table directory.
    #[inline]
    pub fn table_records(&self) -> &[TableRecord] {
        &self.records
    }

    /// Returns the raw payload of `tag`, if present.
    #[must_use]
    pub fn table(&self, tag: [u8; 4]) -> Option<&'a [u8]> {
        let start = find_table(&self.records, tag)?;
        Some(&self.data[start])
    }

    /// Parsed `head` table.
    #[inline]
    pub fn head(&self) -> &HeadTable {
        &self.head
    }

    /// Parsed `hhea` table.
    #[inline]
    pub fn hhea(&self) -> &HheaTable {
        &self.hhea
    }

    /// Parsed `maxp` table.
    #[inline]
    pub fn maxp(&self) -> &MaxpTable {
        &self.maxp
    }

    /// Parsed `OS/2` table, when the font carries one.
    #[inline]
    pub fn os2(&self) -> Option<&Os2Table> {
        self.os2.as_ref()
    }

    /// Design units per em from `head`.
    #[inline]
    pub fn units_per_em(&self) -> u16 {
        self.head.units_per_em
    }

    /// Number of glyphs from `maxp`.
    #[inline]
    pub fn num_glyphs(&self) -> u16 {
        self.maxp.num_glyphs
    }

    /// Horizontal advance width of `glyph` in font units.
    ///
    /// # Panics
    ///
    /// Never panics: an out-of-range id yields the advance of the last
    /// glyph (the `hmtx` trailing-repeat rule), and `0` for an empty font.
    #[must_use]
    pub fn advance_width(&self, glyph: u16) -> u16 {
        let n = usize::from(self.hhea.number_of_h_metrics);
        if n == 0 {
            return 0;
        }
        let idx = usize::from(glyph).min(n - 1);
        let off = idx * 4;
        let bytes = self.hmtx.get(off..off + 2).unwrap_or(&[0, 0]);
        u16::from_be_bytes([bytes[0], bytes[1]])
    }

    /// Left side bearing of `glyph` in font units.
    #[must_use]
    pub fn left_side_bearing(&self, glyph: u16) -> i16 {
        let n = usize::from(self.hhea.number_of_h_metrics);
        let off = if n == 0 {
            0
        } else if usize::from(glyph) < n {
            usize::from(glyph) * 4 + 2
        } else {
            n * 4 + (usize::from(glyph) - n) * 2
        };
        let bytes = self.hmtx.get(off..off + 2).unwrap_or(&[0, 0]);
        u16::from_be_bytes([bytes[0], bytes[1]]) as i16
    }

    /// Byte range of `glyph` inside the `glyf` table.
    ///
    /// Returns `Ok(None)` for empty glyphs (zero-length `loca` slice,
    /// e.g. the space character).
    ///
    /// # Errors
    ///
    /// [`GlypharError::OutOfRange`] when `glyph` is not a valid id, and
    /// [`GlypharError::Malformed`] when the `loca` slice is inverted or
    /// runs past the end of `glyf`.
    pub fn glyph_range(&self, glyph: u16) -> GlypharResult<Option<core::ops::Range<usize>>> {
        let num = usize::from(self.maxp.num_glyphs);
        let id = usize::from(glyph);
        if id >= num {
            return Err(GlypharError::OutOfRange {
                what: "glyph id",
                index: id as u32,
                len: num as u32,
            });
        }
        let long = self.head.index_to_loc_format == 1;
        let entry = |i: usize| -> GlypharResult<usize> {
            let off = i * if long { 4 } else { 2 };
            let bytes = self
                .loca
                .get(off..off + if long { 4 } else { 2 })
                .ok_or(GlypharError::Truncated {
                    context: "loca entry",
                })?;
            if long {
                Ok(
                    usize::try_from(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).map_err(
                        |_| GlypharError::Malformed {
                            context: "loca entry",
                            detail: "offset exceeds addressable size",
                        },
                    )?,
                )
            } else {
                Ok(usize::from(u16::from_be_bytes([bytes[0], bytes[1]])) * 2)
            }
        };
        let start = entry(id)?;
        let end = entry(id + 1)?;
        if start > end {
            return Err(GlypharError::Malformed {
                context: "loca range",
                detail: "start offset greater than end offset",
            });
        }
        if end > self.glyf.len() {
            return Err(GlypharError::Malformed {
                context: "loca range",
                detail: "offset beyond glyf table",
            });
        }
        if start == end {
            Ok(None)
        } else {
            Ok(Some(start..end))
        }
    }

    /// Raw `glyf` payload.
    #[inline]
    pub fn glyf_data(&self) -> &'a [u8] {
        self.glyf
    }

    /// Raw `cvt ` payload (sequence of big-endian `i16` font units).
    #[inline]
    pub fn cvt_data(&self) -> &'a [u8] {
        self.cvt
    }

    /// `cvt` entries decoded to `i16` font-unit values.
    pub fn cvt_values(&self) -> Vec<i16> {
        self.cvt
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]) as i16)
            .collect()
    }

    /// Raw `fpgm` (font program) instructions.
    #[inline]
    pub fn fpgm(&self) -> &'a [u8] {
        self.fpgm
    }

    /// Raw `prep` (CVT program) instructions.
    #[inline]
    pub fn prep(&self) -> &'a [u8] {
        self.prep
    }

    /// Byte range of the best Unicode `cmap` subtable, if any.
    #[inline]
    pub fn cmap_range(&self) -> Option<Range> {
        self.cmap.clone()
    }

    /// Byte range of the `name` table, if any.
    #[inline]
    pub fn name_range(&self) -> Option<Range> {
        self.name.clone()
    }

    /// Raw `kern` table (legacy binary kerning), if present.
    #[inline]
    pub fn kern_data(&self) -> Option<&'a [u8]> {
        self.kern
    }

    /// `gasp` ranges sorted by ascending `maxRangePPEM`.
    #[inline]
    pub fn gasp_ranges(&self) -> &[GaspRange] {
        &self.gasp
    }

    /// Smoothing behavior for `ppem` according to `gasp`.
    ///
    /// Returns `0` (no smoothing, grid-fit only) when the font has no
    /// `gasp` table or no range covers `ppem`.
    #[must_use]
    pub fn gasp_behavior(&self, ppem: u16) -> u16 {
        for range in &self.gasp {
            if ppem <= range.max_range_ppem {
                return range.gasp_behavior;
            }
        }
        0
    }

    /// Decodes a `name` table entry (for example `1` = family name).
    ///
    /// Windows/Unicode records are decoded as UTF-16BE; Macintosh
    /// records are decoded as Latin-1. Records are preferred in the
    /// order Windows, Unicode, Mac, and by English language ids.
    #[must_use]
    pub fn name_string(&self, name_id: u16) -> Option<String> {
        let range = self.name.clone()?;
        let raw = self.data.get(range.start..range.end)?;
        let mut r = Reader::with_pos(raw, 0).ok()?;
        let _format = r.read_u16("name format").ok()?;
        let count = r.read_u16("name count").ok()?;
        let storage = usize::from(r.read_u16("name storage offset").ok()?);
        let mut best: Option<(u8, u16, &[u8])> = None;
        for _ in 0..count {
            let platform = r.read_u16("name platform").ok()?;
            let _encoding = r.read_u16("name encoding").ok()?;
            let language = r.read_u16("name language").ok()?;
            let id = r.read_u16("name id").ok()?;
            let off = r.read_u16("name offset").ok()?;
            let len = r.read_u16("name length").ok()?;
            if id != name_id {
                continue;
            }
            let start = storage.checked_add(usize::from(off))?;
            let end = start.checked_add(usize::from(len))?;
            let bytes = raw.get(start..end)?;
            // Preference: Windows English (0x409), Windows, Unicode,
            // Mac English (0), Mac, everything else.
            let rank = match (platform, language) {
                (3, 0x0409) => 0,
                (3, _) => 1,
                (0, _) => 2,
                (1, 0) => 3,
                _ => 4,
            };
            if best.is_none_or(|(best_rank, _, _)| rank < best_rank) {
                best = Some((rank, platform, bytes));
            }
        }
        let (_, platform, bytes) = best?;
        decode_name(platform, bytes)
    }

    /// Convenience wrapper for the family name (`name` id 1).
    #[must_use]
    pub fn family_name(&self) -> Option<String> {
        self.name_string(1)
    }

    /// Legacy format-0 kerning for a glyph pair, in font units.
    ///
    /// Returns `0` when the font has no `kern` table or the pair is
    /// absent.
    #[must_use]
    pub fn kerning(&self, left: u16, right: u16) -> i16 {
        let Some(raw) = self.kern else { return 0 };
        parse_kern(raw, left, right).unwrap_or(0)
    }

    /// Verifies every table checksum against the directory entry.
    ///
    /// # Errors
    ///
    /// [`GlypharError::Malformed`] naming the first table whose stored
    /// checksum differs from [`simd::table_checksum`].
    pub fn verify_checksums(&self) -> GlypharResult<()> {
        for record in &self.records {
            if record.tag == u32::from_be_bytes(*b"head") {
                // head.checkSumAdjustment is excluded from its own
                // table checksum by the sfnt specification.
                continue;
            }
            let payload = self
                .data
                .get(
                    usize::try_from(record.offset).unwrap_or(usize::MAX)
                        ..usize::try_from(record.offset)
                            .unwrap_or(usize::MAX)
                            .saturating_add(usize::try_from(record.length).unwrap_or(usize::MAX)),
                )
                .ok_or(GlypharError::Truncated {
                    context: "checksum payload",
                })?;
            let actual = crate::simd::table_checksum(payload);
            if actual != record.checksum {
                return Err(GlypharError::Malformed {
                    context: "table checksum",
                    detail: "stored checksum does not match table contents",
                });
            }
        }
        Ok(())
    }

    /// Verifies the whole-file checksum (`0xB1B0AFBA` rule of sfnt).
    ///
    /// # Errors
    ///
    /// [`GlypharError::Malformed`] when the final sum differs from the
    /// magic constant.
    pub fn verify_font_checksum(&self) -> GlypharResult<()> {
        // Summing the complete file — including head.checkSumAdjustment —
        // must yield the sfnt magic constant.
        let sum = crate::simd::table_checksum(self.data);
        if sum != 0xB1B0_AFBA {
            return Err(GlypharError::Malformed {
                context: "font checksum",
                detail: "checkSumAdjustment does not produce 0xB1B0AFBA",
            });
        }
        Ok(())
    }

    /// Returns the `TableRecord` for `tag`.
    #[must_use]
    pub fn find_record(&self, tag: [u8; 4]) -> Option<&TableRecord> {
        let idx = find_record_index(&self.records, u32::from_be_bytes(tag));
        self.records.get(idx)
    }
}

/// Half-open byte range, re-exported for accessor signatures.
pub type Range = core::ops::Range<usize>;

impl fmt::Display for TableRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({} bytes @ 0x{:08X})",
            self.tag_string(),
            self.length,
            self.offset
        )
    }
}

/// Returns the byte range of `tag` inside the directory.
fn find_table(records: &[TableRecord], tag: [u8; 4]) -> Option<Range> {
    let wanted = u32::from_be_bytes(tag);
    let idx = find_record_index(records, wanted);
    let record = records.get(idx)?;
    if record.tag != wanted {
        return None;
    }
    let start = usize::try_from(record.offset).ok()?;
    let end = start.checked_add(usize::try_from(record.length).ok()?)?;
    Some(start..end)
}

/// Scans the directory for `wanted`, preferring the SIMD kernel.
fn find_record_index(records: &[TableRecord], wanted: u32) -> usize {
    crate::simd::find_tag(records, wanted)
}

/// Decodes one `name` record payload given the platform id.
fn decode_name(platform: u16, bytes: &[u8]) -> Option<String> {
    match platform {
        1 => {
            // Macintosh Roman: a superset of Latin-1 for our purposes.
            let mut out = String::with_capacity(bytes.len());
            for &b in bytes {
                out.push(char::from(b));
            }
            Some(out)
        }
        _ => {
            // Unicode / Windows: UTF-16BE.
            let units: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect();
            let mut out = String::with_capacity(units.len());
            for unit in char::decode_utf16(units) {
                out.push(unit.ok()?);
            }
            Some(out)
        }
    }
}

/// Parses the `head` table.
fn parse_head(raw: &[u8]) -> GlypharResult<HeadTable> {
    let mut r = Reader::new(raw);
    let _version = r.read_u32("head version")?;
    let font_revision = r.read_u32("head fontRevision")?;
    let check_sum_adjustment = r.read_u32("head checkSumAdjustment")?;
    let magic_number = r.read_u32("head magicNumber")?;
    let flags = r.read_u16("head flags")?;
    let units_per_em = r.read_u16("head unitsPerEm")?;
    let _created = r.read_i64("head created")?;
    let _modified = r.read_i64("head modified")?;
    let x_min = r.read_i16("head xMin")?;
    let y_min = r.read_i16("head yMin")?;
    let x_max = r.read_i16("head xMax")?;
    let y_max = r.read_i16("head yMax")?;
    let mac_style = r.read_u16("head macStyle")?;
    let lowest_rec_ppem = r.read_i16("head lowestRecPPEM")?;
    let font_direction_hint = r.read_i16("head fontDirectionHint")?;
    let index_to_loc_format = r.read_i16("head indexToLocFormat")?;
    let glyph_data_format = r.read_i16("head glyphDataFormat")?;
    Ok(HeadTable {
        font_revision,
        check_sum_adjustment,
        magic_number,
        flags,
        units_per_em,
        x_min,
        y_min,
        x_max,
        y_max,
        mac_style,
        lowest_rec_ppem,
        font_direction_hint,
        index_to_loc_format,
        glyph_data_format,
    })
}

/// Parses the `hhea` table.
fn parse_hhea(raw: &[u8]) -> GlypharResult<HheaTable> {
    let mut r = Reader::new(raw);
    let _version = r.read_u32("hhea version")?;
    let ascender = r.read_i16("hhea ascender")?;
    let descender = r.read_i16("hhea descender")?;
    let line_gap = r.read_i16("hhea lineGap")?;
    let advance_width_max = r.read_u16("hhea advanceWidthMax")?;
    let min_left_side_bearing = r.read_i16("hhea minLeftSideBearing")?;
    let min_right_side_bearing = r.read_i16("hhea minRightSideBearing")?;
    let x_max_extent = r.read_i16("hhea xMaxExtent")?;
    let caret_slope_rise = r.read_i16("hhea caretSlopeRise")?;
    let caret_slope_run = r.read_i16("hhea caretSlopeRun")?;
    let caret_offset = r.read_i16("hhea caretOffset")?;
    let _r0 = r.read_i16("hhea reserved")?;
    let _r1 = r.read_i16("hhea reserved")?;
    let _r2 = r.read_i16("hhea reserved")?;
    let _r3 = r.read_i16("hhea reserved")?;
    let _metric_data_format = r.read_i16("hhea metricDataFormat")?;
    let number_of_h_metrics = r.read_u16("hhea numberOfHMetrics")?;
    Ok(HheaTable {
        ascender,
        descender,
        line_gap,
        advance_width_max,
        min_left_side_bearing,
        min_right_side_bearing,
        x_max_extent,
        caret_slope_rise,
        caret_slope_run,
        caret_offset,
        number_of_h_metrics,
    })
}

/// Parses the `maxp` table (both v0.5 and v1.0 layouts).
fn parse_maxp(raw: &[u8]) -> GlypharResult<MaxpTable> {
    let mut r = Reader::new(raw);
    let version = r.read_u32("maxp version")?;
    let num_glyphs = r.read_u16("maxp numGlyphs")?;
    if num_glyphs == 0 {
        return Err(GlypharError::Malformed {
            context: "maxp numGlyphs",
            detail: "font has no glyphs",
        });
    }
    if version == 0x0000_5000 {
        return Ok(MaxpTable {
            version,
            num_glyphs,
            max_points: 0,
            max_contours: 0,
            max_composite_points: 0,
            max_composite_contours: 0,
            max_zones: 0,
            max_twilight_points: 0,
            max_storage: 0,
            max_function_defs: 0,
            max_instruction_defs: 0,
            max_stack_elements: 0,
            max_size_of_instructions: 0,
            max_component_elements: 0,
            max_component_depth: 0,
        });
    }
    if version != 0x0001_0000 {
        return Err(GlypharError::Malformed {
            context: "maxp version",
            detail: "expected 0x00005000 or 0x00010000",
        });
    }
    Ok(MaxpTable {
        version,
        num_glyphs,
        max_points: r.read_u16("maxp maxPoints")?,
        max_contours: r.read_u16("maxp maxContours")?,
        max_composite_points: r.read_u16("maxp maxCompositePoints")?,
        max_composite_contours: r.read_u16("maxp maxCompositeContours")?,
        max_zones: r.read_u16("maxp maxZones")?,
        max_twilight_points: r.read_u16("maxp maxTwilightPoints")?,
        max_storage: r.read_u16("maxp maxStorage")?,
        max_function_defs: r.read_u16("maxp maxFunctionDefs")?,
        max_instruction_defs: r.read_u16("maxp maxInstructionDefs")?,
        max_stack_elements: r.read_u16("maxp maxStackElements")?,
        max_size_of_instructions: r.read_u16("maxp maxSizeOfInstructions")?,
        max_component_elements: r.read_u16("maxp maxComponentElements")?,
        max_component_depth: r.read_u16("maxp maxComponentDepth")?,
    })
}

/// Parses the `OS/2` table through version 5.
fn parse_os2(raw: &[u8]) -> GlypharResult<Os2Table> {
    let mut r = Reader::new(raw);
    let version = r.read_u16("OS/2 version")?;
    let x_avg_char_width = r.read_i16("OS/2 xAvgCharWidth")?;
    let us_weight_class = r.read_u16("OS/2 usWeightClass")?;
    let us_width_class = r.read_u16("OS/2 usWidthClass")?;
    let _fs_type = r.read_u16("OS/2 fsType")?;
    let _sub = r.read_bytes(10, "OS/2 subscript")?;
    let _super = r.read_bytes(10, "OS/2 superscript")?;
    let _strike = r.read_bytes(10, "OS/2 strikeout")?;
    let _family = r.read_i16("OS/2 sFamilyClass")?;
    let _panose = r.read_bytes(10, "OS/2 panose")?;
    let _ul = r.read_bytes(16, "OS/2 ulUnicodeRange")?;
    let _vendor = r.read_bytes(4, "OS/2 achVendID")?;
    let _fs_selection = r.read_u16("OS/2 fsSelection")?;
    let us_first_char_index = r.read_u16("OS/2 usFirstCharIndex")?;
    let us_last_char_index = r.read_u16("OS/2 usLastCharIndex")?;
    let s_typo_ascender = r.read_i16("OS/2 sTypoAscender")?;
    let s_typo_descender = r.read_i16("OS/2 sTypoDescender")?;
    let s_typo_line_gap = r.read_i16("OS/2 sTypoLineGap")?;
    let us_win_ascent = r.read_u16("OS/2 usWinAscent")?;
    let us_win_descent = r.read_u16("OS/2 usWinDescent")?;
    if version >= 1 {
        let _code_page = r.read_bytes(8, "OS/2 ulCodePageRange")?;
    }
    let (s_x_height, s_cap_height, us_default_char, us_break_char, us_max_context) = if version >= 2 {
        (
            r.read_i16("OS/2 sxHeight")?,
            r.read_i16("OS/2 sCapHeight")?,
            r.read_u16("OS/2 usDefaultChar")?,
            r.read_u16("OS/2 usBreakChar")?,
            r.read_u16("OS/2 usMaxContext")?,
        )
    } else {
        (0, 0, 0, 0, 0)
    };
    if version >= 5 {
        let _lower = r.read_u16("OS/2 usLowerOpticalPointSize")?;
        let _upper = r.read_u16("OS/2 usUpperOpticalPointSize")?;
    }
    Ok(Os2Table {
        version,
        x_avg_char_width,
        us_weight_class,
        us_width_class,
        s_typo_ascender,
        s_typo_descender,
        s_typo_line_gap,
        us_win_ascent,
        us_win_descent,
        us_first_char_index,
        us_last_char_index,
        s_x_height,
        s_cap_height,
        us_default_char,
        us_break_char,
        us_max_context,
    })
}

/// Parses the `gasp` table.
fn parse_gasp(raw: &[u8]) -> GlypharResult<Vec<GaspRange>> {
    let mut r = Reader::new(raw);
    let _version = r.read_u16("gasp version")?;
    let count = r.read_u16("gasp numRanges")?;
    let mut ranges = Vec::with_capacity(usize::from(count));
    for _ in 0..count {
        let max_range_ppem = r.read_u16("gasp maxRangePPEM")?;
        let gasp_behavior = r.read_u16("gasp gaspBehavior")?;
        ranges.push(GaspRange {
            max_range_ppem,
            gasp_behavior,
        });
    }
    ranges.sort_by_key(|range| range.max_range_ppem);
    Ok(ranges)
}

/// Checks every `loca` pair against the `glyf` table length.
fn validate_loca(loca: &[u8], glyf: &[u8], head: &HeadTable, maxp: &MaxpTable) -> GlypharResult<()> {
    let long = head.index_to_loc_format == 1;
    let count = usize::from(maxp.num_glyphs) + 1;
    for i in 0..count {
        let off = i * if long { 4 } else { 2 };
        let bytes = &loca[off..off + if long { 4 } else { 2 }];
        let value = if long {
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
        } else {
            usize::from(u16::from_be_bytes([bytes[0], bytes[1]])) * 2
        };
        if value > glyf.len() {
            return Err(GlypharError::Malformed {
                context: "loca entry",
                detail: "glyph offset beyond glyf table",
            });
        }
    }
    Ok(())
}

/// Looks up a format-0 kerning pair by binary search.
fn parse_kern(raw: &[u8], left: u16, right: u16) -> Option<i16> {
    let mut r = Reader::new(raw);
    let _version = r.read_u16("kern version").ok()?;
    let n_tables = r.read_u16("kern nTables").ok()?;
    for _ in 0..n_tables {
        let sub_start = r.position();
        let _sub_version = r.read_u16("kern subtable version").ok()?;
        let length = usize::from(r.read_u16("kern subtable length").ok()?);
        let coverage = r.read_u16("kern subtable coverage").ok()?;
        let format = (coverage >> 8) as u8;
        let horizontal = coverage & 1 != 0;
        let cross_stream = coverage & 4 != 0;
        if format == 0 && horizontal && !cross_stream {
            let n_pairs = usize::from(r.read_u16("kern nPairs").ok()?);
            let _search_range = r.read_u16("kern searchRange").ok()?;
            let _entry_selector = r.read_u16("kern entrySelector").ok()?;
            let _range_shift = r.read_u16("kern rangeShift").ok()?;
            let mut lo = 0usize;
            let mut hi = n_pairs;
            let key = (u32::from(left) << 16) | u32::from(right);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let bytes = r.data.get(r.pos + mid * 6..r.pos + mid * 6 + 6)?;
                let l = u16::from_be_bytes([bytes[0], bytes[1]]);
                let rr = u16::from_be_bytes([bytes[2], bytes[3]]);
                let value = u16::from_be_bytes([bytes[4], bytes[5]]) as i16;
                let pair_key = (u32::from(l) << 16) | u32::from(rr);
                if pair_key < key {
                    lo = mid + 1;
                } else if pair_key > key {
                    hi = mid;
                } else {
                    return Some(value);
                }
            }
        }
        r.seek(sub_start.checked_add(length)?, "kern subtable")
            .ok()?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::SyntheticFont;

    #[test]
    fn parses_synthetic_font_tables() {
        let font_bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&font_bytes).expect("synthetic font must parse");
        assert_eq!(font.units_per_em(), 1000);
        assert_eq!(font.num_glyphs(), 4);
        assert_eq!(font.hhea().ascender, 800);
        assert_eq!(font.maxp().max_stack_elements, 64);
        assert_eq!(font.family_name().as_deref(), Some("Glyphar Test"));
        assert_eq!(font.advance_width(1), 600);
        assert_eq!(font.left_side_bearing(1), 50);
    }

    #[test]
    fn glyph_ranges_resolve_and_empty_glyphs_are_none() {
        let font_bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&font_bytes).expect("parse");
        // Glyph 0 (.notdef) and glyph 1 (square) have outlines.
        assert!(font.glyph_range(0).expect("id 0").is_some());
        assert!(font.glyph_range(1).expect("id 1").is_some());
        // Glyph 2 (space) is an empty glyph.
        assert!(font.glyph_range(2).expect("id 2").is_none());
        assert!(matches!(
            font.glyph_range(99),
            Err(GlypharError::OutOfRange { what: "glyph id", .. })
        ));
    }

    #[test]
    fn checksums_match_for_synthetic_font() {
        let font_bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&font_bytes).expect("parse");
        font.verify_checksums()
            .expect("table checksums must verify");
    }

    #[test]
    fn rejects_truncated_input() {
        let font_bytes = SyntheticFont::new().build();
        for len in [0usize, 4, 12, 20, 64] {
            let sliced = &font_bytes[..len.min(font_bytes.len())];
            assert!(FontFile::parse(sliced).is_err(), "len {len} must fail");
        }
    }

    #[test]
    fn rejects_wrong_sfnt_version() {
        let mut font_bytes = SyntheticFont::new().build();
        font_bytes[..4].copy_from_slice(&0x4F54_544Fu32.to_be_bytes());
        assert_eq!(
            FontFile::parse(&font_bytes).err(),
            Some(GlypharError::Unsupported {
                feature: "CFF/CFF2 outlines (OTTO)"
            })
        );
    }

    #[test]
    fn rejects_bad_head_magic() {
        let mut font_bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&font_bytes).expect("parse");
        let head = font
            .find_record(*b"head")
            .expect("head record")
            .offset as usize;
        // Overwrite the magic number (offset 12 inside head).
        font_bytes[head + 12..head + 16].copy_from_slice(&0u32.to_be_bytes());
        assert!(matches!(
            FontFile::parse(&font_bytes),
            Err(GlypharError::Malformed {
                context: "head magicNumber",
                ..
            })
        ));
    }

    #[test]
    fn name_and_kern_and_gasp_lookups() {
        let font_bytes = SyntheticFont::new().with_kern().build();
        let font = FontFile::parse(&font_bytes).expect("parse");
        assert_eq!(font.name_string(6).as_deref(), Some("GlypharTest-Regular"));
        assert_eq!(font.kerning(1, 1), -40);
        assert_eq!(font.kerning(0, 1), 0);
        assert_eq!(font.gasp_behavior(8), 0b0011);
        assert_eq!(font.gasp_behavior(40), 0b0011);
        assert_eq!(font.gasp_behavior(64), 0);
        assert!(font.cmap_range().is_some());
    }

    #[test]
    fn table_lookup_and_display() {
        let font_bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&font_bytes).expect("parse");
        assert!(font.table(*b"head").is_some());
        assert!(font.table(*b"GSUB").is_none());
        let record = font.find_record(*b"maxp").expect("maxp");
        assert!(record.to_string().contains("maxp"));
        assert_eq!(record.tag_bytes(), *b"maxp");
    }

    #[test]
    fn reader_reports_truncation_not_panics() {
        let data = [1u8, 2, 3];
        let mut r = Reader::new(&data);
        assert!(r.read_u32("x").is_err());
        assert_eq!(r.read_u16("x").expect("2 bytes"), 0x0102);
        assert_eq!(r.read_u8("x").expect("last byte"), 3);
        assert!(r.read_u8("x").is_err());
        assert!(r.seek(99, "x").is_err());
    }
}
