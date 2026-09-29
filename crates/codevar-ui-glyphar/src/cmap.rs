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

//! Character-to-glyph mapping (`cmap`).
//!
//! [`Cmap::parse`] walks the `cmap` encoding-record directory, ranks
//! every subtable by how well its (platform, encoding) pair suits
//! Unicode text, decodes the winner, and then answers
//! [`Cmap::glyph_index`] from a cache-friendly lookup that never
//! allocates.
//!
//! # Supported subtable formats
//!
//! | Format | Kind | Lookup |
//! |--------|------|--------|
//! | 0 | byte array (256 entries) | `O(1)` |
//! | 2 | high-byte mapping (mixed 8/16-bit) | `O(1)` after sub-header select |
//! | 4 | segment mapping with delta (BMP) | binary search via [`crate::simd::find_end_code`] |
//! | 6 | trimmed table | `O(1)` |
//! | 12 | segmented coverage (full Unicode) | binary search over `u32` groups |
//! | 13 | many-to-one range mapping | binary search over `u32` groups |
//! | 14 | Unicode variation sequences | binary search over selectors |
//!
//! Formats 8 and 10 (the deprecated "mixed coverage" formats) are not
//! implemented: they are skipped during selection, so a font whose only
//! subtable uses one of them fails [`Cmap::parse`] with
//! [`GlypharError::Malformed`] instead of silently mapping everything
//! to glyph 0.
//!
//! # Subtable selection
//!
//! Encoding records are ranked by platform/encoding first and by
//! format second (see [`subtable_rank`]); the highest-ranked *supported*
//! record becomes the primary subtable. Ties are resolved in favour of
//! the record that appears first in the table, which is what the spec
//! recommends for directories built in preference order.
//!
//! Format 14 subtables are never selected as the primary mapping; the
//! first one found is kept aside for [`Cmap::variation_action`].
//!
//! # Example
//!
//! ```
//! use codevar_ui_glyphar::cmap::Cmap;
//!
//! // A format 6 subtable: U+0041..U+0042 -> glyphs 3, 4.
//! let mut sub = Vec::new();
//! sub.extend_from_slice(&6u16.to_be_bytes());    // format
//! sub.extend_from_slice(&14u16.to_be_bytes());   // length
//! sub.extend_from_slice(&0u16.to_be_bytes());    // language
//! sub.extend_from_slice(&0x41u16.to_be_bytes()); // firstCode
//! sub.extend_from_slice(&2u16.to_be_bytes());    // entryCount
//! sub.extend_from_slice(&3u16.to_be_bytes());    // glyphIdArray
//! sub.extend_from_slice(&4u16.to_be_bytes());
//!
//! // One encoding record (platform 3, encoding 1) at offset 12.
//! let mut table = Vec::new();
//! table.extend_from_slice(&0u16.to_be_bytes());  // version
//! table.extend_from_slice(&1u16.to_be_bytes());  // numTables
//! table.extend_from_slice(&3u16.to_be_bytes());  // platformID
//! table.extend_from_slice(&1u16.to_be_bytes());  // encodingID
//! table.extend_from_slice(&12u32.to_be_bytes()); // subtableOffset
//! table.extend_from_slice(&sub);
//!
//! let cmap = Cmap::parse(&table)?;
//! assert_eq!(cmap.glyph_index('A'), 3);
//! assert_eq!(cmap.glyph_index('B'), 4);
//! assert_eq!(cmap.glyph_index('C'), 0);
//! # Ok::<(), codevar_ui_glyphar::GlypharError>(())
//! ```
//!
//! # Performance
//!
//! Format 4 decodes its segment arrays once at parse time into
//! contiguous `Vec<u16>`s. The segment search then runs through
//! [`crate::simd::find_end_code`], which vectorises the ordered scan of
//! the `endCode` array (8 lanes with SSE2, 16 with AVX2, 32 with
//! AVX-512, 8 with NEON). Every other format binary-searches the raw
//! big-endian bytes without touching the heap.

use alloc::vec::Vec;
use core::fmt;

use crate::font_file::{FontFile, Reader};
use crate::simd;
use crate::{GlypharError, GlypharResult};

/// Result of a format 14 variation-sequence lookup.
///
/// The selector covers a *base character* pair such as
/// `U+4E00 U+FE00`; the action says which glyph the pair resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvsAction {
    /// The (base, selector) pair is not present in the font.
    NotCovered,
    /// Use the default glyph for `base` (the pair appears in the
    /// *Default UVS* table).
    UseDefault,
    /// Use the explicit glyph from the *Non-Default UVS* table.
    UseGlyph(u16),
}

/// One entry of the `cmap` encoding-record directory.
///
/// The four fields mirror the on-disk record exactly; [`Cmap`]
/// exposes the whole directory through [`Cmap::subtables`] so callers
/// can inspect what the font offers even when they only ever look up
/// characters through the primary subtable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubtableInfo {
    /// Platform identifier (`0` Unicode, `1` Macintosh, `3` Windows).
    pub platform_id: u16,
    /// Encoding identifier, interpreted relative to `platform_id`.
    pub encoding_id: u16,
    /// Subtable format read from the subtable header (`0`, `2`, `4`,
    /// `6`, `12`, `13`, `14`, …).
    pub format: u16,
    /// Byte offset of the subtable inside the `cmap` table.
    pub offset: u32,
}

/// Decoded format 4 segment arrays.
///
/// The arrays are stored separately (structure-of-arrays) so the
/// segment scan touches only `end_codes`, which is what
/// [`crate::simd::find_end_code`] searches.
#[derive(Debug, Clone, Default)]
struct Format4 {
    /// `endCode` of each segment, ascending, native endian.
    end_codes: Vec<u16>,
    /// `startCode` of each segment.
    start_codes: Vec<u16>,
    /// `idDelta` of each segment (added modulo 65536).
    id_deltas: Vec<i16>,
    /// `idRangeOffset` of each segment, in bytes, relative to the
    /// element itself.
    id_range_offsets: Vec<u16>,
    /// The `glyphIdArray` tail the range offsets point into.
    glyph_id_array: Vec<u16>,
}

impl Format4 {
    /// Decodes a format 4 subtable.
    ///
    /// The declared `length` field bounds the read, so a subtable
    /// that claims to be longer than the bytes actually available is
    /// reported as [`GlypharError::Truncated`].
    fn parse(raw: &[u8]) -> GlypharResult<Self> {
        let mut r = Reader::new(raw);
        let format = r.read_u16("cmap format 4 format")?;
        if format != 4 {
            return Err(GlypharError::Malformed {
                context: "cmap format 4",
                detail: "subtable does not declare format 4",
            });
        }
        let length = usize::from(r.read_u16("cmap format 4 length")?);
        if length < 16 || length > raw.len() {
            return Err(GlypharError::Malformed {
                context: "cmap format 4",
                detail: "declared length is impossible for a segment map",
            });
        }
        let _language = r.read_u16("cmap format 4 language")?;
        let seg_count_x2 = r.read_u16("cmap format 4 segCountX2")?;
        if seg_count_x2 == 0 || seg_count_x2 % 2 != 0 {
            return Err(GlypharError::Malformed {
                context: "cmap format 4",
                detail: "segCountX2 must be a positive even number",
            });
        }
        let seg_count = usize::from(seg_count_x2) / 2;
        // searchRange / entrySelector / rangeShift are hints for a
        // binary search; the arrays below are authoritative.
        let _search_range = r.read_u16("cmap format 4 searchRange")?;
        let _entry_selector = r.read_u16("cmap format 4 entrySelector")?;
        let _range_shift = r.read_u16("cmap format 4 rangeShift")?;

        let mut end_codes = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            end_codes.push(r.read_u16("cmap format 4 endCode")?);
        }
        let _reserved_pad = r.read_u16("cmap format 4 reservedPad")?;
        let mut start_codes = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            start_codes.push(r.read_u16("cmap format 4 startCode")?);
        }
        let mut id_deltas = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            id_deltas.push(r.read_i16("cmap format 4 idDelta")?);
        }
        let mut id_range_offsets = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            id_range_offsets.push(r.read_u16("cmap format 4 idRangeOffset")?);
        }

        // Anything left between here and the declared length is the
        // glyphIdArray the idRangeOffset fields point into.
        let end = length.min(raw.len());
        let mut glyph_id_array = Vec::new();
        while r.position() + 2 <= end {
            glyph_id_array.push(r.read_u16("cmap format 4 glyphIdArray")?);
        }

        Ok(Self {
            end_codes,
            start_codes,
            id_deltas,
            id_range_offsets,
            glyph_id_array,
        })
    }

    /// Maps a BMP code point to a glyph id (0 = notdef).
    #[inline]
    fn glyph(&self, code: u16) -> u16 {
        let Some(i) = simd::find_end_code(&self.end_codes, code) else {
            return 0;
        };
        if code < self.start_codes[i] {
            return 0;
        }
        if self.id_range_offsets[i] == 0 {
            return code.wrapping_add(self.id_deltas[i] as u16);
        }
        // `idRangeOffset[i]` is the byte distance from the element
        // itself to the glyphIdArray entry, so in glyphIdArray units
        // the index is `offset/2 + i - segCount`.
        let seg_count = self.end_codes.len() as i64;
        let index = i64::from(self.id_range_offsets[i]) / 2 + i as i64 - seg_count;
        let index = if index < 0 { return 0 } else { index as usize };
        let Some(&glyph) = self.glyph_id_array.get(index) else {
            return 0;
        };
        if glyph == 0 {
            return 0;
        }
        glyph.wrapping_add(self.id_deltas[i] as u16)
    }

    /// Appends every mapping of this subtable to `out`.
    ///
    /// The mandatory `U+FFFF` terminator segment is skipped: it exists
    /// only so the segment array is well-formed, and `U+FFFF` is not a
    /// Unicode scalar value.
    fn collect(&self, out: &mut Vec<(u32, u16)>) {
        for i in 0..self.end_codes.len() {
            let (start, end) = (self.start_codes[i], self.end_codes[i]);
            for code in start..=end {
                if code == 0xFFFF {
                    break;
                }
                let glyph = self.glyph(code);
                out.push((u32::from(code), glyph));
            }
        }
    }
}

/// The primary (best-ranked) subtable, decoded for fast lookup.
enum Primary<'a> {
    /// Format 0: `subtable[6 + code]` is the glyph id.
    Format0(&'a [u8]),
    /// Format 2: high-byte mapping over raw bytes.
    Format2(&'a [u8]),
    /// Format 4: segment map with delta, fully decoded.
    Format4(Format4),
    /// Format 6: trimmed `firstCode` + glyph array.
    Format6(&'a [u8]),
    /// Format 12: 12-byte groups of `(start, end, firstGlyph)`.
    Format12(&'a [u8]),
    /// Format 13: 12-byte groups of `(start, end, constantGlyph)`.
    Format13(&'a [u8]),
}

impl Primary<'_> {
    /// On-disk format number of this subtable.
    #[must_use]
    const fn format(&self) -> u16 {
        match self {
            Self::Format0(_) => 0,
            Self::Format2(_) => 2,
            Self::Format4(_) => 4,
            Self::Format6(_) => 6,
            Self::Format12(_) => 12,
            Self::Format13(_) => 13,
        }
    }

    /// Maps a Unicode scalar value (or legacy code) to a glyph id.
    #[inline]
    fn glyph(&self, code: u32) -> u16 {
        match self {
            Self::Format0(raw) => lookup_format0(raw, code),
            Self::Format2(raw) => lookup_format2(raw, code),
            Self::Format4(decoded) => {
                let Ok(code) = u16::try_from(code) else {
                    return 0;
                };
                decoded.glyph(code)
            }
            Self::Format6(raw) => lookup_format6(raw, code),
            Self::Format12(raw) => lookup_groups(raw, code, false),
            Self::Format13(raw) => lookup_groups(raw, code, true),
        }
    }

    /// Appends every mapping of this subtable to `out`.
    fn collect(&self, out: &mut Vec<(u32, u16)>) {
        match self {
            Self::Format0(raw) => {
                for code in 0..256usize {
                    let glyph = u16::from(raw.get(6 + code).copied().unwrap_or(0));
                    out.push((code as u32, glyph));
                }
            }
            Self::Format2(raw) => collect_format2(raw, out),
            Self::Format4(decoded) => decoded.collect(out),
            Self::Format6(raw) => collect_format6(raw, out),
            Self::Format12(raw) => collect_groups(raw, out, false),
            Self::Format13(raw) => collect_groups(raw, out, true),
        }
    }
}

/// A parsed `cmap` table.
///
/// The struct borrows the table bytes: nothing but the format 4
/// segment arrays (and the directory listing) is copied.
pub struct Cmap<'a> {
    subtables: Vec<SubtableInfo>,
    primary: Option<Primary<'a>>,
    uvs: Option<&'a [u8]>,
}

impl<'a> Cmap<'a> {
    /// Parses a complete `cmap` table.
    ///
    /// # Errors
    ///
    /// * [`GlypharError::Truncated`] — the header or an encoding
    ///   record does not fit, or a subtable offset points outside the
    ///   table.
    /// * [`GlypharError::Malformed`] — no subtable is both supported
    ///   and well-formed, or the chosen subtable fails validation.
    pub fn parse(data: &'a [u8]) -> GlypharResult<Self> {
        let mut r = Reader::new(data);
        let _version = r.read_u16("cmap version")?;
        let count = r.read_u16("cmap subtable count")?;
        let mut subtables = Vec::with_capacity(usize::from(count));
        for _ in 0..count {
            let platform_id = r.read_u16("cmap platform id")?;
            let encoding_id = r.read_u16("cmap encoding id")?;
            let offset = r.read_u32("cmap subtable offset")?;
            let off = usize::try_from(offset).map_err(|_| GlypharError::OutOfRange {
                what: "cmap subtable offset",
                index: offset,
                len: u32::try_from(data.len()).unwrap_or(u32::MAX),
            })?;
            let mut sub = Reader::with_pos(data, off)?;
            let format = sub.read_u16("cmap subtable format")?;
            subtables.push(SubtableInfo {
                platform_id,
                encoding_id,
                format,
                offset,
            });
        }

        let primary = select_primary(data, &subtables)?;
        let uvs = select_uvs(data, &subtables)?;
        Ok(Self {
            subtables,
            primary: Some(primary),
            uvs,
        })
    }

    /// Parses the `cmap` of an already-parsed font.
    ///
    /// Returns `Ok(None)` when the font carries no `cmap` table at
    /// all, which is legal for symbol-only fonts.
    ///
    /// # Errors
    ///
    /// Propagates [`Cmap::parse`] failures.
    pub fn from_font(font: &FontFile<'a>) -> GlypharResult<Option<Self>> {
        let Some(data) = font.table([b'c', b'm', b'a', b'p']) else {
            return Ok(None);
        };
        Self::parse(data).map(Some)
    }

    /// Parses raw bytes, returning `None` when they are not a usable
    /// `cmap` table.
    ///
    /// This convenience wrapper exists for callers that only need a
    /// best-effort mapping (a cache, a preview) and would rather skip
    /// a broken font than handle an error.
    #[must_use]
    pub fn from_font_bytes(data: &'a [u8]) -> Option<Self> {
        Self::parse(data).ok()
    }

    /// Maps a Unicode scalar value to a glyph id.
    ///
    /// Code points outside the primary subtable's coverage (for
    /// example a supplementary-plane character in a format 4 font)
    /// map to glyph 0, the `.notdef` glyph.
    #[inline]
    #[must_use]
    pub fn glyph_index(&self, c: char) -> u16 {
        self.glyph_index_u32(c as u32)
    }

    /// Maps a raw code point to a glyph id.
    ///
    /// # Panics
    ///
    /// Never panics; out-of-range or unmapped codes return 0.
    #[inline]
    #[must_use]
    pub fn glyph_index_u32(&self, code: u32) -> u16 {
        match self.primary.as_ref() {
            Some(primary) => primary.glyph(code),
            None => 0,
        }
    }

    /// Format number of the primary subtable, if one was selected.
    #[must_use]
    pub fn format(&self) -> Option<u16> {
        self.primary.as_ref().map(Primary::format)
    }

    /// The encoding-record directory, in file order.
    #[must_use]
    pub fn subtables(&self) -> &[SubtableInfo] {
        &self.subtables
    }

    /// Resolves a format 14 variation sequence.
    ///
    /// Without a format 14 subtable every pair resolves to
    /// [`UvsAction::NotCovered`].
    #[must_use]
    pub fn variation_action(&self, base: char, selector: char) -> UvsAction {
        self.variation_action_u32(base as u32, selector as u32)
    }

    /// Resolves a format 14 variation sequence from raw code points.
    ///
    /// When both a default and a non-default mapping exist for the
    /// pair the explicit (non-default) mapping wins, because it names
    /// the exact glyph the font designer assigned to the sequence.
    #[must_use]
    pub fn variation_action_u32(&self, base: u32, selector: u32) -> UvsAction {
        match self.uvs {
            Some(data) => lookup_uvs(data, base, selector),
            None => UvsAction::NotCovered,
        }
    }

    /// Collects every mapping of the primary subtable.
    ///
    /// Mappings that resolve to glyph 0 are included: they are real
    /// entries of the font and callers that rebuild reverse maps want
    /// to see them.
    #[must_use]
    pub fn collect_mappings(&self) -> Vec<(u32, u16)> {
        let mut out = Vec::new();
        if let Some(primary) = self.primary.as_ref() {
            primary.collect(&mut out);
        }
        out
    }
}

impl fmt::Debug for Cmap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cmap")
            .field("format", &self.format())
            .field("subtables", &self.subtables)
            .field("has_uvs", &self.uvs.is_some())
            .finish()
    }
}

/// Ranks a subtable by platform/encoding, then by format.
///
/// Higher is better. Windows UCS-4 (`3/10`) and Unicode full
/// (`0/4`–`0/6`) records beat the BMP-only pairs because they are the
/// only ones that can cover the astral planes; a format bonus then
/// breaks ties in favour of the formats the ranking would expect to
/// find behind that encoding.
#[must_use]
pub fn subtable_rank(platform_id: u16, encoding_id: u16, format: u16) -> u16 {
    let encoding = match (platform_id, encoding_id) {
        (3, 10) => 60,                    // Windows, UCS-4
        (0, 6) | (0, 4) => 55,            // Unicode, full repertoire
        (3, 1) => 50,                     // Windows, BMP
        (0, 3) => 45,                     // Unicode, BMP
        (0, 5) | (3, 12) | (3, 13) => 44, // variation selectors / UCS-4 variants
        (0, 0) | (0, 1) | (0, 2) => 40,   // Unicode 1.0 / 1.1 / 2.0
        (3, 0) => 35,                     // Windows symbol
        (1, 0) => 20,                     // Macintosh Roman
        _ => 10,                          // anything else
    };
    let bonus = match format {
        12 | 13 => 2,
        4 => 1,
        _ => 0,
    };
    encoding + bonus
}

/// Picks the best *supported* subtable, or fails when none exists.
fn select_primary<'a>(data: &'a [u8], subtables: &[SubtableInfo]) -> GlypharResult<Primary<'a>> {
    let mut best: Option<(u16, usize)> = None;
    for (i, info) in subtables.iter().enumerate() {
        if info.format == 14 || !is_supported(info.format) {
            continue;
        }
        let rank = subtable_rank(info.platform_id, info.encoding_id, info.format);
        if best.is_none_or(|(best_rank, _)| rank > best_rank) {
            best = Some((rank, i));
        }
    }
    let (_, index) = best.ok_or(GlypharError::Malformed {
        context: "cmap",
        detail: "no supported subtable (formats 0, 2, 4, 6, 12, 13)",
    })?;
    decode_primary(data, &subtables[index])
}

/// Whether this engine implements the given subtable format.
#[must_use]
const fn is_supported(format: u16) -> bool {
    matches!(format, 0 | 2 | 4 | 6 | 12 | 13)
}

/// Validates and borrows one subtable for lookup.
fn decode_primary<'a>(data: &'a [u8], info: &SubtableInfo) -> GlypharResult<Primary<'a>> {
    let off = usize::try_from(info.offset).map_err(|_| GlypharError::OutOfRange {
        what: "cmap subtable offset",
        index: info.offset,
        len: u32::try_from(data.len()).unwrap_or(u32::MAX),
    })?;
    let raw = data.get(off..).ok_or(GlypharError::Truncated {
        context: "cmap subtable",
    })?;
    match info.format {
        0 => {
            // format, length, language, glyphIdArray[256]
            let declared = usize::from(read_be16(raw, 2).unwrap_or(0));
            if declared < 262 {
                return Err(GlypharError::Malformed {
                    context: "cmap format 0",
                    detail: "declared length is shorter than the 256-entry array",
                });
            }
            let slice = raw.get(..262).ok_or(GlypharError::Truncated {
                context: "cmap format 0",
            })?;
            Ok(Primary::Format0(slice))
        }
        2 => {
            // format, length, language, subHeaderKeys[256], subheaders.
            // The glyphIdArray that the idRangeOffset fields point
            // into sits past the header, so the whole declared
            // subtable has to stay visible to the lookup.
            let header_end = 6 + 512 + 8;
            if raw.len() < header_end {
                return Err(GlypharError::Malformed {
                    context: "cmap format 2",
                    detail: "subtable has no sub-header key array",
                });
            }
            let declared = usize::from(read_be16(raw, 2).unwrap_or(0));
            let end = if declared >= header_end && declared <= raw.len() {
                declared
            } else {
                raw.len()
            };
            Ok(Primary::Format2(&raw[..end]))
        }
        4 => Ok(Primary::Format4(Format4::parse(raw)?)),
        6 => {
            // format, length, language are all `u16` in this format.
            let length = usize::from(read_be16(raw, 2).unwrap_or(0));
            if length < 10 || length > raw.len() {
                return Err(GlypharError::Malformed {
                    context: "cmap format 6",
                    detail: "declared length is impossible for a trimmed table",
                });
            }
            Ok(Primary::Format6(&raw[..length]))
        }
        12 | 13 => {
            let length = declared_length_u32(raw).ok_or(GlypharError::Malformed {
                context: "cmap format 12/13",
                detail: "missing length field",
            })?;
            if length < 16 || length > raw.len() {
                return Err(GlypharError::Malformed {
                    context: "cmap format 12/13",
                    detail: "declared length is impossible for a group table",
                });
            }
            let groups = read_be32(raw, 12).unwrap_or(0);
            let needed = 16u64 + u64::from(groups) * 12;
            if needed > u64::try_from(length).unwrap_or(u64::MAX) {
                return Err(GlypharError::Truncated {
                    context: "cmap format 12/13 groups",
                });
            }
            let slice = &raw[..length];
            if info.format == 12 {
                Ok(Primary::Format12(slice))
            } else {
                Ok(Primary::Format13(slice))
            }
        }
        other => Err(GlypharError::Unsupported {
            feature: match other {
                8 => "cmap format 8",
                10 => "cmap format 10",
                14 => "cmap format 14 as primary subtable",
                _ => "unknown cmap format",
            },
        }),
    }
}

/// Selects and bounds the format 14 subtable, when the font has one.
///
/// The header is `format u16`, `length u32`, `numVarSelectorRecords
/// u32`; each record then spans 11 bytes (`uint24` selector plus two
/// `Offset32`s). The returned slice is trimmed to the declared length
/// so every later read is bounds-checked against the table itself.
fn select_uvs<'a>(data: &'a [u8], subtables: &[SubtableInfo]) -> GlypharResult<Option<&'a [u8]>> {
    for info in subtables {
        if info.format != 14 {
            continue;
        }
        let off = usize::try_from(info.offset).map_err(|_| GlypharError::OutOfRange {
            what: "cmap format 14 offset",
            index: info.offset,
            len: u32::try_from(data.len()).unwrap_or(u32::MAX),
        })?;
        let raw = data.get(off..).ok_or(GlypharError::Truncated {
            context: "cmap format 14",
        })?;
        if raw.len() < 10 {
            return Err(GlypharError::Truncated {
                context: "cmap format 14 header",
            });
        }
        let declared = read_be32(raw, 2).unwrap_or(0) as usize;
        if declared < 10 || declared > raw.len() {
            return Err(GlypharError::Malformed {
                context: "cmap format 14",
                detail: "declared length is impossible for a variation table",
            });
        }
        let records = read_be32(raw, 6).unwrap_or(0);
        let needed = 10u64 + u64::from(records) * 11;
        if needed > u64::try_from(declared).unwrap_or(u64::MAX) {
            return Err(GlypharError::Truncated {
                context: "cmap format 14 selectors",
            });
        }
        return Ok(Some(&raw[..declared]));
    }
    Ok(None)
}

/// Format 0 lookup: the glyph id *is* the array entry.
#[inline]
fn lookup_format0(raw: &[u8], code: u32) -> u16 {
    if code > 0xFF {
        return 0;
    }
    u16::from(raw.get(6 + code as usize).copied().unwrap_or(0))
}

/// Format 6 lookup: `glyphIdArray[code - firstCode]`.
#[inline]
fn lookup_format6(raw: &[u8], code: u32) -> u16 {
    if code > 0xFFFF {
        return 0;
    }
    let code = code as u16;
    let Some(first) = read_be16(raw, 6) else {
        return 0;
    };
    let Some(count) = read_be16(raw, 8) else {
        return 0;
    };
    if code < first || code - first >= count {
        return 0;
    }
    let index = 10 + usize::from(code - first) * 2;
    read_be16(raw, index).unwrap_or(0)
}

/// Format 6 enumeration.
fn collect_format6(raw: &[u8], out: &mut Vec<(u32, u16)>) {
    let (Some(first), Some(count)) = (read_be16(raw, 6), read_be16(raw, 8)) else {
        return;
    };
    for i in 0..u32::from(count) {
        let code = u32::from(first) + i;
        let index = 10 + i as usize * 2;
        out.push((code, read_be16(raw, index).unwrap_or(0)));
    }
}

/// Formats 12 and 13 share the same group layout; only the glyph
/// computation differs (`startGlyph + delta` vs `startGlyph`).
#[inline]
fn lookup_groups(raw: &[u8], code: u32, constant: bool) -> u16 {
    let Some(count) = read_be32(raw, 12) else {
        return 0;
    };
    let mut lo = 0u32;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let at = 16 + mid as usize * 12;
        let (Some(start), Some(end), Some(glyph)) =
            (read_be32(raw, at), read_be32(raw, at + 4), read_be32(raw, at + 8))
        else {
            return 0;
        };
        if code < start {
            hi = mid;
        } else if code > end {
            lo = mid + 1;
        } else {
            let glyph = if constant {
                glyph
            } else {
                glyph.wrapping_add(code - start)
            };
            return u16::try_from(glyph).unwrap_or(0);
        }
    }
    0
}

/// Format 12/13 enumeration.
fn collect_groups(raw: &[u8], out: &mut Vec<(u32, u16)>, constant: bool) {
    let Some(count) = read_be32(raw, 12) else {
        return;
    };
    for i in 0..count {
        let at = 16 + i as usize * 12;
        let (Some(start), Some(end), Some(glyph)) =
            (read_be32(raw, at), read_be32(raw, at + 4), read_be32(raw, at + 8))
        else {
            return;
        };
        for code in start..=end {
            let glyph = if constant {
                glyph
            } else {
                glyph.wrapping_add(code - start)
            };
            out.push((code, u16::try_from(glyph).unwrap_or(0)));
        }
    }
}

/// Format 2 lookup over the raw sub-header key array.
///
/// `subHeaderKeys[high] / 8` selects the sub-header used for the byte
/// pair. A key of 0 means the high byte has no byte-pair mapping, so
/// only single-byte codes (high byte 0) are looked up through
/// sub-header 0.
fn lookup_format2(raw: &[u8], code: u32) -> u16 {
    if code > 0xFFFF {
        return 0;
    }
    let high = (code >> 8) as u16;
    let low = (code & 0xFF) as u16;
    let Some(key) = read_be16(raw, 6 + usize::from(high) * 2) else {
        return 0;
    };
    let index = usize::from(key / 8);
    if index == 0 && high != 0 {
        // Sub-header 0 is reserved for single-byte codes; a high byte
        // pointing at it declares no byte-pair mapping.
        return 0;
    }
    let sub = 6 + 512 + index * 8;
    let (Some(first), Some(count), Some(delta), Some(range)) = (
        read_be16(raw, sub),
        read_be16(raw, sub + 2),
        read_be16(raw, sub + 4),
        read_be16(raw, sub + 6),
    ) else {
        return 0;
    };
    if low < first || low - first >= count {
        return 0;
    }
    if range == 0 {
        return 0;
    }
    let entry = sub + 6 + usize::from(range) + usize::from(low - first) * 2;
    let Some(glyph) = read_be16(raw, entry) else {
        return 0;
    };
    if glyph == 0 {
        return 0;
    }
    (i32::from(glyph) + i32::from(delta)) as u16
}

/// Format 2 enumeration over every distinct (high, low) pair.
fn collect_format2(raw: &[u8], out: &mut Vec<(u32, u16)>) {
    for high in 0..256u32 {
        let Some(key) = read_be16(raw, 6 + high as usize * 2) else {
            return;
        };
        if key / 8 == 0 && high != 0 {
            continue;
        }
        let sub = 6 + 512 + usize::from(key / 8) * 8;
        let (Some(first), Some(count)) = (read_be16(raw, sub), read_be16(raw, sub + 2)) else {
            continue;
        };
        for i in 0..u32::from(count) {
            let low = u32::from(first) + i;
            if low > 0xFF {
                break;
            }
            let code = (high << 8) | low;
            out.push((code, lookup_format2(raw, code)));
        }
    }
}

/// Format 14 variation-sequence lookup.
fn lookup_uvs(raw: &[u8], base: u32, selector: u32) -> UvsAction {
    let Some(count) = read_be32(raw, 6) else {
        return UvsAction::NotCovered;
    };
    let selector = selector & 0xFF_FFFF;
    let mut lo = 0u32;
    let mut hi = count;
    let mut found: Option<(u32, u32)> = None;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let at = 10 + mid as usize * 11;
        let Some(record) = raw.get(at..at + 11) else {
            return UvsAction::NotCovered;
        };
        let value = u32::from(record[0]) << 16 | u32::from(record[1]) << 8 | u32::from(record[2]);
        let default = read_be32(record, 3).unwrap_or(0);
        let non_default = read_be32(record, 7).unwrap_or(0);
        if selector < value {
            hi = mid;
        } else if selector > value {
            lo = mid + 1;
        } else {
            found = Some((default, non_default));
            break;
        }
    }
    let Some((default, non_default)) = found else {
        return UvsAction::NotCovered;
    };

    if non_default != 0 {
        if let Some(glyph) = find_uvs_glyph(raw, non_default, base) {
            return UvsAction::UseGlyph(glyph);
        }
    }
    if default != 0 && uvs_in_default_range(raw, default, base) {
        return UvsAction::UseDefault;
    }
    UvsAction::NotCovered
}

/// Looks `base` up in the non-default UVS table at `offset`.
fn find_uvs_glyph(raw: &[u8], offset: u32, base: u32) -> Option<u16> {
    let at = usize::try_from(offset).ok()?;
    let count = read_be32(raw, at)?;
    let mut lo = 0u32;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let entry = at.checked_add(4 + mid as usize * 5)?;
        let record = raw.get(entry..entry + 5)?;
        let value = u32::from(record[0]) << 16 | u32::from(record[1]) << 8 | u32::from(record[2]);
        let glyph = u16::from_be_bytes([record[3], record[4]]);
        if base < value {
            hi = mid;
        } else if base > value {
            lo = mid + 1;
        } else {
            return Some(glyph);
        }
    }
    None
}

/// Whether `base` falls inside a default UVS range at `offset`.
fn uvs_in_default_range(raw: &[u8], offset: u32, base: u32) -> bool {
    let Some(at) = usize::try_from(offset).ok() else {
        return false;
    };
    let Some(count) = read_be32(raw, at) else {
        return false;
    };
    let mut lo = 0u32;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let entry = at.checked_add(4 + mid as usize * 4);
        let Some(entry) = entry else {
            return false;
        };
        let Some(range) = raw.get(entry..entry + 4) else {
            return false;
        };
        let start = u32::from(range[0]) << 16 | u32::from(range[1]) << 8 | u32::from(range[2]);
        let end = start + u32::from(range[3]);
        if base < start {
            hi = mid;
        } else if base > end {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// Reads a big-endian `u16` with bounds checking.
#[inline]
fn read_be16(data: &[u8], off: usize) -> Option<u16> {
    let end = off.checked_add(2)?;
    let bytes = data.get(off..end)?;
    Some(u16::from_be_bytes([bytes[0], bytes[1]]))
}

/// Reads a big-endian `u32` with bounds checking.
#[inline]
fn read_be32(data: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let bytes = data.get(off..end)?;
    Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// The `length` field of formats 12 and 13 (`u32` at offset 4,
/// after the `u16` reserved word).
#[inline]
fn declared_length_u32(raw: &[u8]) -> Option<usize> {
    read_be32(raw, 4).map(|length| length as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_file::FontFile;
    use crate::test_support::SyntheticFont;

    fn push_u16(out: &mut Vec<u8>, v: u16) {
        out.extend_from_slice(&v.to_be_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_be_bytes());
    }

    /// Wraps `subtables` (paired with their encoding records) in a
    /// `cmap` header.
    fn cmap_table(records: &[(u16, u16)], subtables: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 0); // version
        push_u16(&mut out, u16::try_from(records.len()).unwrap_or(0));
        let header = 4 + 8 * records.len();
        let mut offset = header as u32;
        for ((platform, encoding), sub) in records.iter().zip(subtables) {
            push_u16(&mut out, *platform);
            push_u16(&mut out, *encoding);
            push_u32(&mut out, offset);
            offset += sub.len() as u32;
        }
        for sub in subtables {
            out.extend_from_slice(sub);
        }
        out
    }

    /// Format 6: `U+0041..U+0043 → 5, 6, 7`.
    fn format6() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 6);
        push_u16(&mut out, 0); // length, patched
        push_u16(&mut out, 0); // language
        push_u16(&mut out, 0x41); // firstCode
        push_u16(&mut out, 3); // entryCount
        for glyph in [5u16, 6, 7] {
            push_u16(&mut out, glyph);
        }
        let len = u16::try_from(out.len()).unwrap_or(0);
        out[2..4].copy_from_slice(&len.to_be_bytes());
        out
    }

    /// Format 0 with `U+0041 → 9`.
    fn format0() -> Vec<u8> {
        let mut out = vec![0u8; 262];
        out[0..2].copy_from_slice(&0u16.to_be_bytes()); // format
        let len = 262u16;
        out[2..4].copy_from_slice(&len.to_be_bytes());
        out[6 + 0x41] = 9;
        out
    }

    /// Format 12 with three single-code groups.
    fn format12(groups: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 12);
        push_u16(&mut out, 0); // reserved
        push_u32(&mut out, 16 + 12 * groups.len() as u32);
        push_u32(&mut out, 0); // language
        push_u32(&mut out, groups.len() as u32);
        for (start, end, glyph) in groups {
            push_u32(&mut out, *start);
            push_u32(&mut out, *end);
            push_u32(&mut out, *glyph);
        }
        out
    }

    /// Format 2: single-byte mapping `0x41..0x43 → 5, 6, 7` plus a
    /// byte-pair mapping for high byte 0x02 (`0x0241 → 11`).
    fn format2() -> Vec<u8> {
        let subheader_start = 6 + 512;
        let glyph_start = subheader_start + 16; // two sub-headers
        let mut out = Vec::new();
        push_u16(&mut out, 2);
        push_u16(&mut out, 0); // length, patched
        push_u16(&mut out, 0); // language
        for high in 0..256u16 {
            // high byte 0 uses sub-header 0, high byte 2 uses 1.
            let key = match high {
                0 => 0,
                2 => 8,
                _ => 0,
            };
            push_u16(&mut out, key);
        }
        // Sub-header 0: single bytes 0x41..0x43.
        push_u16(&mut out, 0x41); // firstCode
        push_u16(&mut out, 3); // entryCount
        push_u16(&mut out, 0); // idDelta
        push_u16(&mut out, 2); // idRangeOffset → first array slot
        // Sub-header 1: byte pair (0x02, 0x41).
        push_u16(&mut out, 0x41); // firstCode
        push_u16(&mut out, 1); // entryCount
        push_u16(&mut out, 0); // idDelta
        push_u16(&mut out, 2); // idRangeOffset → first array slot
        // glyphIdArray: sub-header 0 entries, then sub-header 1.
        for glyph in [5u16, 6, 7, 11] {
            push_u16(&mut out, glyph);
        }
        let len = u16::try_from(out.len()).unwrap_or(0);
        out[2..4].copy_from_slice(&len.to_be_bytes());
        // Both idRangeOffsets are byte distances from their own field
        // to the first array element they own.
        let range0 = glyph_start - (subheader_start + 6);
        let range1 = glyph_start + 6 - (subheader_start + 8 + 6);
        out[subheader_start + 6..subheader_start + 8].copy_from_slice(&(range0 as u16).to_be_bytes());
        let second = subheader_start + 8;
        out[second + 6..second + 8].copy_from_slice(&(range1 as u16).to_be_bytes());
        out
    }

    /// Format 14 with one selector: U+0042..U+0043 use the default
    /// glyph, U+0041 → 7 and U+0044 → 8 are explicit mappings.
    ///
    /// Layout: header (10) + one 11-byte record, then the default
    /// table, then the non-default table.
    fn format14() -> Vec<u8> {
        let default_off = (10 + 11) as u32;
        let default_len = 4 + 4; // count + one 4-byte range
        let non_default_off = default_off + default_len;
        let non_default_len = 4 + 5 * 2; // count + two 5-byte mappings

        let mut out = Vec::new();
        push_u16(&mut out, 14); // format
        push_u32(&mut out, non_default_off + non_default_len); // length
        push_u32(&mut out, 1); // numVarSelectorRecords
        // VariationSelector record for U+FE00.
        out.extend_from_slice(&[0x00, 0xFE, 0x00]);
        push_u32(&mut out, default_off);
        push_u32(&mut out, non_default_off);
        // Default UVS: U+0042..U+0043 (start + additionalCount).
        push_u32(&mut out, 1);
        out.extend_from_slice(&[0x00, 0x00, 0x42, 0x01]);
        // Non-default UVS: U+0041 → glyph 7, U+0044 → glyph 8.
        push_u32(&mut out, 2);
        for (code, glyph) in [(0x41u32, 7u16), (0x44, 8)] {
            out.extend_from_slice(&[
                ((code >> 16) & 0xFF) as u8,
                ((code >> 8) & 0xFF) as u8,
                (code & 0xFF) as u8,
            ]);
            push_u16(&mut out, glyph);
        }
        out
    }

    /// Format 13: every code of a range maps to one glyph.
    fn format13() -> Vec<u8> {
        let mut out = format12(&[(0x10_0000, 0x10_0003, 42)]);
        out[0..2].copy_from_slice(&13u16.to_be_bytes());
        out
    }

    /// Format 4 with `segCountX2` odd, which the spec forbids.
    fn broken_format4() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 4);
        push_u16(&mut out, 32); // length (larger than the bytes below)
        push_u16(&mut out, 0); // language
        push_u16(&mut out, 3); // segCountX2 — odd
        push_u16(&mut out, 4);
        push_u16(&mut out, 1);
        push_u16(&mut out, 2);
        for _ in 0..6 {
            push_u16(&mut out, 0);
        }
        out
    }

    #[test]
    fn synthetic_font_maps_bmp_characters() {
        let bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&bytes).expect("font");
        let cmap = Cmap::from_font(&font)
            .expect("cmap")
            .expect("cmap present");
        assert_eq!(cmap.format(), Some(4));
        assert_eq!(cmap.glyph_index(' '), 2);
        assert_eq!(cmap.glyph_index('A'), 1);
        assert_eq!(cmap.glyph_index('B'), 0);
        assert_eq!(cmap.glyph_index('é'), 0);
        assert_eq!(cmap.glyph_index_u32(0x1_F600), 0);
    }

    #[test]
    fn format12_is_preferred_when_present() {
        let bytes = SyntheticFont::new().with_cmap12().build();
        let font = FontFile::parse(&bytes).expect("font");
        let cmap = Cmap::from_font(&font)
            .expect("cmap")
            .expect("cmap present");
        assert_eq!(cmap.format(), Some(12));
        assert_eq!(cmap.glyph_index('A'), 1);
        assert_eq!(cmap.glyph_index(' '), 2);
        assert_eq!(cmap.glyph_index('B'), 0);
        assert_eq!(cmap.subtables().len(), 2);
        assert_eq!(cmap.subtables()[0].format, 4);
        assert_eq!(cmap.subtables()[1].format, 12);
    }

    #[test]
    fn format4_mappings_are_enumerable() {
        let bytes = SyntheticFont::new().build();
        let font = FontFile::parse(&bytes).expect("font");
        let cmap = Cmap::from_font(&font)
            .expect("cmap")
            .expect("cmap present");
        let mappings = cmap.collect_mappings();
        assert_eq!(mappings, [(0x20, 2), (0x41, 1), (0x42, 0)]);
    }

    #[test]
    fn format6_trims_out_of_range_codes() {
        let sub = format6();
        let table = cmap_table(&[(3, 1)], &[sub]);
        let cmap = Cmap::parse(&table).expect("parse");
        assert_eq!(cmap.format(), Some(6));
        assert_eq!(cmap.glyph_index('A'), 5);
        assert_eq!(cmap.glyph_index('C'), 7);
        assert_eq!(cmap.glyph_index('@'), 0);
        assert_eq!(cmap.glyph_index('D'), 0);
        assert_eq!(cmap.collect_mappings(), [(0x41, 5), (0x42, 6), (0x43, 7)]);
    }

    #[test]
    fn format0_indexes_a_byte_array() {
        let table = cmap_table(&[(1, 0)], &[format0()]);
        let cmap = Cmap::parse(&table).expect("parse");
        assert_eq!(cmap.format(), Some(0));
        assert_eq!(cmap.glyph_index('A'), 9);
        assert_eq!(cmap.glyph_index('B'), 0);
        assert_eq!(cmap.glyph_index('€'), 0);
        assert_eq!(cmap.collect_mappings().len(), 256);
    }

    #[test]
    fn format2_handles_byte_pairs() {
        let table = cmap_table(&[(1, 0)], &[format2()]);
        let cmap = Cmap::parse(&table).expect("parse");
        assert_eq!(cmap.format(), Some(2));
        assert_eq!(cmap.glyph_index('A'), 5);
        assert_eq!(cmap.glyph_index('B'), 6);
        assert_eq!(cmap.glyph_index('C'), 7);
        assert_eq!(cmap.glyph_index_u32(0x0241), 11);
        assert_eq!(cmap.glyph_index_u32(0x0141), 0);
        assert_eq!(cmap.glyph_index('@'), 0);
        let mappings = cmap.collect_mappings();
        assert!(mappings.contains(&(0x41, 5)));
        assert!(mappings.contains(&(0x0241, 11)));
        assert!(!mappings.contains(&(0x0141, 11)));
    }

    #[test]
    fn format13_maps_a_whole_range_to_one_glyph() {
        let table = cmap_table(&[(3, 10)], &[format13()]);
        let cmap = Cmap::parse(&table).expect("parse");
        assert_eq!(cmap.format(), Some(13));
        assert_eq!(cmap.glyph_index_u32(0x10_0000), 42);
        assert_eq!(cmap.glyph_index_u32(0x10_0003), 42);
        assert_eq!(cmap.glyph_index_u32(0x10_0004), 0);
        assert_eq!(cmap.collect_mappings().len(), 4);
    }

    #[test]
    fn format14_alone_is_not_a_usable_directory() {
        let table = cmap_table(&[(0, 5)], &[format14()]);
        assert!(matches!(Cmap::parse(&table), Err(GlypharError::Malformed { .. })));
    }

    #[test]
    fn format14_resolves_variation_sequences() {
        let table = cmap_table(&[(3, 1), (0, 5)], &[format6(), format14()]);
        let cmap = Cmap::parse(&table).expect("parse");
        assert_eq!(cmap.format(), Some(6));
        assert_eq!(cmap.glyph_index('A'), 5);
        assert_eq!(
            cmap.variation_action('\u{4E00}', '\u{FE00}'),
            UvsAction::NotCovered
        );
        assert_eq!(cmap.variation_action('A', '\u{FE00}'), UvsAction::UseGlyph(7));
        assert_eq!(cmap.variation_action('D', '\u{FE00}'), UvsAction::UseGlyph(8));
        assert_eq!(cmap.variation_action('B', '\u{FE00}'), UvsAction::UseDefault);
        assert_eq!(cmap.variation_action('B', '\u{FE01}'), UvsAction::NotCovered);
        // The primary subtable is untouched by the variation data.
        assert_eq!(cmap.glyph_index('B'), 6);
    }

    #[test]
    fn records_are_ranked_before_formats() {
        assert!(subtable_rank(3, 10, 12) > subtable_rank(3, 1, 12));
        assert!(subtable_rank(3, 1, 12) > subtable_rank(3, 1, 4));
        assert!(subtable_rank(0, 6, 4) > subtable_rank(1, 0, 12));
        assert!(subtable_rank(3, 0, 12) < subtable_rank(0, 0, 4));
    }

    #[test]
    fn empty_input_is_truncated() {
        assert!(matches!(Cmap::parse(&[]), Err(GlypharError::Truncated { .. })));
    }

    #[test]
    fn out_of_range_offset_is_rejected() {
        let mut table = Vec::new();
        push_u16(&mut table, 0);
        push_u16(&mut table, 1);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 0x00FF_FFFF);
        assert!(matches!(Cmap::parse(&table), Err(GlypharError::Truncated { .. })));
    }

    #[test]
    fn unsupported_only_directory_fails() {
        let sub = format12(&[(0x41, 0x41, 1)]);
        let mut patched = sub;
        patched[0..2].copy_from_slice(&8u16.to_be_bytes()); // format 8
        let table = cmap_table(&[(0, 4)], &[patched]);
        assert!(matches!(Cmap::parse(&table), Err(GlypharError::Malformed { .. })));
    }

    #[test]
    fn malformed_format4_is_rejected() {
        let table = cmap_table(&[(3, 1)], &[broken_format4()]);
        assert!(matches!(Cmap::parse(&table), Err(GlypharError::Malformed { .. })));
    }

    #[test]
    fn short_format6_is_rejected() {
        let mut sub = format6();
        sub.truncate(8);
        let table = cmap_table(&[(3, 1)], &[sub]);
        assert!(matches!(Cmap::parse(&table), Err(GlypharError::Malformed { .. })));
    }

    #[test]
    fn from_font_bytes_swallows_broken_tables() {
        assert!(Cmap::from_font_bytes(&[]).is_none());
        assert!(Cmap::from_font_bytes(&[0, 0, 0, 0]).is_none());
        let table = cmap_table(&[(1, 0)], &[format0()]);
        assert!(Cmap::from_font_bytes(&table).is_some());
    }

    #[test]
    fn font_without_cmap_yields_none() {
        let bytes = SyntheticFont::new().without_cmap().build();
        let font = FontFile::parse(&bytes).expect("font");
        assert!(Cmap::from_font(&font).expect("result").is_none());
    }

    #[test]
    fn missing_glyph_id_array_is_mapped_to_zero() {
        // A range offset that points past the glyphIdArray tail must
        // degrade to glyph 0 rather than reading out of bounds.
        let mut sub = Vec::new();
        push_u16(&mut sub, 4);
        push_u16(&mut sub, 0); // length, patched
        push_u16(&mut sub, 0);
        push_u16(&mut sub, 4); // segCountX2 = 2 segments
        push_u16(&mut sub, 4);
        push_u16(&mut sub, 1);
        push_u16(&mut sub, 0);
        for end in [0x0041u16, 0xFFFF] {
            push_u16(&mut sub, end);
        }
        push_u16(&mut sub, 0); // reservedPad
        for start in [0x0041u16, 0xFFFF] {
            push_u16(&mut sub, start);
        }
        push_u16(&mut sub, 0); // idDelta: segment 0
        push_u16(&mut sub, 1); // idDelta: segment 1
        // idRangeOffset: segment 0 skips past the (empty) glyphIdArray
        // and segment 1 has no array at all.
        push_u16(&mut sub, 4);
        push_u16(&mut sub, 0);
        let len = u16::try_from(sub.len()).unwrap_or(0);
        sub[2..4].copy_from_slice(&len.to_be_bytes());
        let table = cmap_table(&[(3, 1)], &[sub]);
        let cmap = Cmap::parse(&table).expect("parse");
        assert_eq!(cmap.glyph_index('A'), 0);
        assert_eq!(cmap.glyph_index('\u{FFFE}'), 0);
    }
}
