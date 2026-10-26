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

//! The core SFNT table readers: `head`, `maxp`, `hhea`, `hmtx`, `loca`
//! and `kern`.
//!
//! | FreeType file | Item |
//! |---------------|------|
//! | `ttload.c`    | [`HeadTable::parse`], [`MaxpTable::parse`] |
//! | `ttmtx.c`     | [`HheaTable::parse`], [`HmtxTable`] |
//! | `ttpload.c`   | [`LocaTable`] |
//! | `ttkern.c`    | [`KernTable`] |
//!
//! Every parser validates its input length up front and reads
//! big-endian fields through the fallible `bytes::Buf` peeks, so a
//! truncated or corrupted table produces a [`TtError`] instead of a
//! panic.

use alloc::vec::Vec;
use bytes::Buf;
use codevar_truetype_core::{BBox, TtError, TtResult};

/// The `head` table (`TT_Header`, `tt_face_load_generic_header`).
///
/// Parsed from the 54-byte layout of `ttload.c`; `Created`/`Modified`
/// are the 64-bit `LONGDATETIME` values seconds since 1904-01-01.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeadTable {
    /// `Table_Version` (normally `0x00010000`).
    pub table_version: u32,
    /// `Font_Revision` (16.16 fixed).
    pub font_revision: u32,
    /// `CheckSum_Adjustment` of the whole font.
    pub check_sum_adjustment: u32,
    /// `Magic_Number`, [`crate::tags::HEAD_MAGIC_NUMBER`] when intact.
    pub magic_number: u32,
    /// `Flags` (bit 0: baseline at y=0, bit 1: lsb at x=0, ...).
    pub flags: u16,
    /// `Units_Per_EM` (16..16384; must be non-zero).
    pub units_per_em: u16,
    /// `Created` timestamp (seconds since 1904-01-01).
    pub created: i64,
    /// `Modified` timestamp (seconds since 1904-01-01).
    pub modified: i64,
    /// The font bounding box in font units (`xMin`/`yMin`/`xMax`/`yMax`).
    pub bbox: BBox,
    /// `Mac_Style` bits (bit 0 bold, bit 1 italic).
    pub mac_style: u16,
    /// `Lowest_Rec_PPEM`.
    pub lowest_rec_ppem: u16,
    /// `Font_Direction` (2 = left-to-right, 1 = right-to-left).
    pub font_direction: i16,
    /// `Index_To_Loc_Format`: `0` = short (`u16 * 2`), anything else =
    /// long (`u32`), following `tt_face_load_loca`'s `!= 0` test.
    pub index_to_loc_format: i16,
    /// `Glyph_Data_Format` (must be 0).
    pub glyph_data_format: i16,
}

impl HeadTable {
    /// Parses the `head` table (or `bhed`, same layout) from `data`.
    ///
    /// `check_table_dir` (`ttload.c`) rejects tables shorter than
    /// `0x36` bytes with `Table_Missing`; this reader reports the
    /// truncation as [`TtError::INVALID_TABLE`] and lets the caller
    /// decide (see [`crate::SfntFont::open`]).
    pub fn parse(data: &[u8]) -> TtResult<Self> {
        if data.len() < 54 {
            return Err(TtError::INVALID_TABLE);
        }
        let mut buf: &[u8] = data;
        Ok(HeadTable {
            table_version: buf
                .try_get_u32()
                .map_err(|_| TtError::INVALID_TABLE)?,
            font_revision: buf
                .try_get_u32()
                .map_err(|_| TtError::INVALID_TABLE)?,
            check_sum_adjustment: buf
                .try_get_u32()
                .map_err(|_| TtError::INVALID_TABLE)?,
            magic_number: buf
                .try_get_u32()
                .map_err(|_| TtError::INVALID_TABLE)?,
            flags: buf
                .try_get_u16()
                .map_err(|_| TtError::INVALID_TABLE)?,
            units_per_em: buf
                .try_get_u16()
                .map_err(|_| TtError::INVALID_TABLE)?,
            created: buf
                .try_get_i64()
                .map_err(|_| TtError::INVALID_TABLE)?,
            modified: buf
                .try_get_i64()
                .map_err(|_| TtError::INVALID_TABLE)?,
            bbox: BBox::from_edges(
                i64::from(
                    buf.try_get_i16()
                        .map_err(|_| TtError::INVALID_TABLE)?,
                ),
                i64::from(
                    buf.try_get_i16()
                        .map_err(|_| TtError::INVALID_TABLE)?,
                ),
                i64::from(
                    buf.try_get_i16()
                        .map_err(|_| TtError::INVALID_TABLE)?,
                ),
                i64::from(
                    buf.try_get_i16()
                        .map_err(|_| TtError::INVALID_TABLE)?,
                ),
            ),
            mac_style: buf
                .try_get_u16()
                .map_err(|_| TtError::INVALID_TABLE)?,
            lowest_rec_ppem: buf
                .try_get_u16()
                .map_err(|_| TtError::INVALID_TABLE)?,
            font_direction: buf
                .try_get_i16()
                .map_err(|_| TtError::INVALID_TABLE)?,
            index_to_loc_format: buf
                .try_get_i16()
                .map_err(|_| TtError::INVALID_TABLE)?,
            glyph_data_format: buf
                .try_get_i16()
                .map_err(|_| TtError::INVALID_TABLE)?,
        })
    }

    /// `true` when the `head.magic_number` field holds
    /// [`crate::tags::HEAD_MAGIC_NUMBER`].
    ///
    /// `check_table_dir` only traces a wrong magic number (it is not a
    /// fatal error in FreeType), so this is exposed for callers instead
    /// of being enforced.
    #[inline]
    pub fn has_valid_magic(&self) -> bool {
        self.magic_number == crate::tags::HEAD_MAGIC_NUMBER
    }

    /// `true` when `loca` stores long (`u32`) offsets
    /// (`Index_To_Loc_Format != 0`, the test of `tt_face_load_loca`).
    #[inline]
    pub fn uses_long_loca(&self) -> bool {
        self.index_to_loc_format != 0
    }
}

/// The `maxp` table (`TT_MaxProfile`, `tt_face_load_max_profile`).
///
/// Version `0x5000` (CFF) fonts only carry `numGlyphs`; the 13
/// additional fields are parsed when `version >= 0x10000` and the
/// table is long enough.  The three defensive clamps of `ttload.c`
/// (`maxFunctionDefs >= 64`, `maxTwilightPoints <= 0xFFFB`,
/// `maxComponentDepth <= 100`) are applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaxpTable {
    /// `version` (`0x00005000` or `0x00010000`).
    pub version: u32,
    /// `numGlyphs`.
    pub num_glyphs: u16,
    /// `maxPoints`.
    pub max_points: u16,
    /// `maxContours`.
    pub max_contours: u16,
    /// `maxCompositePoints`.
    pub max_composite_points: u16,
    /// `maxCompositeContours`.
    pub max_composite_contours: u16,
    /// `maxZones`.
    pub max_zones: u16,
    /// `maxTwilightPoints`
    pub max_twilight_points: u16,
    /// `maxStorage`.
    pub max_storage: u16,
    /// `maxFunctionDefs`
    pub max_function_defs: u16,
    /// `maxInstructionDefs`.
    pub max_instruction_defs: u16,
    /// `maxStackElements`.
    pub max_stack_elements: u16,
    /// `maxSizeOfInstructions`.
    pub max_size_of_instructions: u16,
    /// `maxComponentElements`.
    pub max_component_elements: u16,
    /// `maxComponentDepth`
    pub max_component_depth: u16,
}

impl MaxpTable {
    /// Parses the `maxp` table from `data`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than 6 bytes, or a version
    ///   `>= 0x10000` table shorter than the 32 bytes the 13 extra
    ///   fields occupy (`tt_face_load_max_profile` fails on the short
    ///   stream read).
    pub fn parse(data: &[u8]) -> TtResult<Self> {
        if data.len() < 6 {
            return Err(TtError::INVALID_TABLE);
        }
        let mut buf: &[u8] = data;
        let invalid = |_| TtError::INVALID_TABLE;
        let version = buf.try_get_u32().map_err(invalid)?;
        let mut table = MaxpTable {
            version,
            num_glyphs: buf.try_get_u16().map_err(invalid)?,
            ..MaxpTable::default()
        };
        if version < 0x1_0000 {
            return Ok(table);
        }
        if data.len() < 32 {
            return Err(TtError::INVALID_TABLE);
        }
        table.max_points = buf.try_get_u16().map_err(invalid)?;
        table.max_contours = buf.try_get_u16().map_err(invalid)?;
        table.max_composite_points = buf.try_get_u16().map_err(invalid)?;
        table.max_composite_contours = buf.try_get_u16().map_err(invalid)?;
        table.max_zones = buf.try_get_u16().map_err(invalid)?;
        table.max_twilight_points = buf.try_get_u16().map_err(invalid)?.min(0xFFFB);
        table.max_storage = buf.try_get_u16().map_err(invalid)?;
        table.max_function_defs = buf.try_get_u16().map_err(invalid)?.max(64);
        table.max_instruction_defs = buf.try_get_u16().map_err(invalid)?;
        table.max_stack_elements = buf.try_get_u16().map_err(invalid)?;
        table.max_size_of_instructions = buf.try_get_u16().map_err(invalid)?;
        table.max_component_elements = buf.try_get_u16().map_err(invalid)?;
        table.max_component_depth = buf.try_get_u16().map_err(invalid)?.min(100);
        Ok(table)
    }
}

/// The `hhea` table (`TT_HoriHeader`, `tt_face_load_hhea`).
///
/// The vertical `vhea` table has the same layout; this port does not
/// load vertical metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HheaTable {
    /// `Version` (normally `0x00010000`).
    pub version: u32,
    /// `Ascender` (font units, may be negative).
    pub ascender: i16,
    /// `Descender` (font units, normally negative).
    pub descender: i16,
    /// `Line_Gap`.
    pub line_gap: i16,
    /// `advance_Width_Max`.
    pub advance_width_max: u16,
    /// `min_Left_Side_Bearing`.
    pub min_left_side_bearing: i16,
    /// `min_Right_Side_Bearing`.
    pub min_right_side_bearing: i16,
    /// `xMax_Extent`.
    pub x_max_extent: i16,
    /// `caret_Slope_Rise` (1 for a vertical caret).
    pub caret_slope_rise: i16,
    /// `caret_Slope_Run`.
    pub caret_slope_run: i16,
    /// `caret_Offset`.
    pub caret_offset: i16,
    /// The four `Reserved` slots of the header.
    pub reserved: [i16; 4],
    /// `metric_Data_Format` (0 for current fonts).
    pub metric_data_format: i16,
    /// `number_Of_HMetrics`: the number of full (`advance`, `lsb`)
    /// records at the start of `hmtx`.
    pub number_of_h_metrics: u16,
}

impl HheaTable {
    /// Parses the `hhea`/`vhea` header from `data`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than 36 bytes.
    pub fn parse(data: &[u8]) -> TtResult<Self> {
        if data.len() < 36 {
            return Err(TtError::INVALID_TABLE);
        }
        let mut buf: &[u8] = data;
        let invalid = |_| TtError::INVALID_TABLE;
        Ok(HheaTable {
            version: buf.try_get_u32().map_err(invalid)?,
            ascender: buf.try_get_i16().map_err(invalid)?,
            descender: buf.try_get_i16().map_err(invalid)?,
            line_gap: buf.try_get_i16().map_err(invalid)?,
            advance_width_max: buf.try_get_u16().map_err(invalid)?,
            min_left_side_bearing: buf.try_get_i16().map_err(invalid)?,
            min_right_side_bearing: buf.try_get_i16().map_err(invalid)?,
            x_max_extent: buf.try_get_i16().map_err(invalid)?,
            caret_slope_rise: buf.try_get_i16().map_err(invalid)?,
            caret_slope_run: buf.try_get_i16().map_err(invalid)?,
            caret_offset: buf.try_get_i16().map_err(invalid)?,
            reserved: [
                buf.try_get_i16().map_err(invalid)?,
                buf.try_get_i16().map_err(invalid)?,
                buf.try_get_i16().map_err(invalid)?,
                buf.try_get_i16().map_err(invalid)?,
            ],
            metric_data_format: buf.try_get_i16().map_err(invalid)?,
            number_of_h_metrics: buf.try_get_u16().map_err(invalid)?,
        })
    }
}

/// The `hmtx` table reader (`tt_face_get_metrics`, `ttmtx.c`).
///
/// `hmtx` starts with `number_of_h_metrics` four-byte records
/// (`advance: u16`, `lsb: i16`); every following glyph only stores a
/// left side bearing and reuses the *last* advance width — the
/// "short-array monospace rule" that lets monospaced (and many CJK)
/// fonts shrink the table.
#[derive(Clone, Copy, Debug)]
pub struct HmtxTable<'a> {
    data: &'a [u8],
    number_of_h_metrics: u16,
}

impl<'a> HmtxTable<'a> {
    /// Wraps the `hmtx` bytes with the `number_Of_HMetrics` value from
    /// `hhea`.  The slice is the (possibly sanitized) table from the
    /// directory; short slices simply yield `(0, 0)` metrics, matching
    /// the `NoData` fallback of `tt_face_get_metrics`.
    #[inline]
    pub const fn new(data: &'a [u8], number_of_h_metrics: u16) -> Self {
        HmtxTable {
            data,
            number_of_h_metrics,
        }
    }

    /// `hhea.numberOfHMetrics`.
    #[inline]
    pub const fn number_of_h_metrics(&self) -> u16 {
        self.number_of_h_metrics
    }

    /// The raw table bytes.
    #[inline]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// `tt_face_get_metrics` (`ttmtx.c`): the `(advance, lsb)` pair of
    /// `gindex` in font units.
    ///
    /// For `gindex >= numberOfHMetrics` the advance comes from the last
    /// full record and the bearing is read from the trailing short
    /// array; a bearing past the end of the table falls back to `0`
    /// (`tt_face_get_metrics`'s `*abearing = 0` path).  Every access is
    /// bounds-checked against the table slice.
    pub fn metrics(&self, gindex: u32) -> (u16, i16) {
        let k = u32::from(self.number_of_h_metrics);
        if k == 0 {
            return (0, 0);
        }
        if gindex < k {
            let pos = (gindex as usize) * 4;
            let Some(mut record) = self.data.get(pos..pos + 4) else {
                return (0, 0);
            };
            let Some(advance) = record.try_get_u16().ok() else {
                return (0, 0);
            };
            let Some(lsb) = record.try_get_i16().ok() else {
                return (0, 0);
            };
            return (advance, lsb);
        }
        let last = ((k - 1) as usize) * 4;
        let Some(mut record) = self.data.get(last..last + 4) else {
            return (0, 0);
        };
        let Some(advance) = record.try_get_u16().ok() else {
            return (0, 0);
        };
        let lsb_pos = u64::from(k) * 4 + u64::from(gindex - k) * 2;
        let lsb = usize::try_from(lsb_pos)
            .ok()
            .and_then(|pos| self.data.get(pos..))
            .and_then(|mut at| at.try_get_i16().ok())
            .unwrap_or(0);
        (advance, lsb)
    }

    /// The advance width of `gindex` in font units
    /// (`tt_face_get_metrics`, `abearing` discarded).
    #[inline]
    pub fn advance_width(&self, gindex: u32) -> u16 {
        self.metrics(gindex).0
    }

    /// The left side bearing of `gindex` in font units
    /// (`tt_face_get_metrics`, `aadvance` discarded).
    #[inline]
    pub fn left_side_bearing(&self, gindex: u32) -> i16 {
        self.metrics(gindex).1
    }
}

/// The `loca` table reader (`tt_face_load_loca`, `ttpload.c`).
///
/// The slice handed to [`LocaTable::new`] is the *effective* table:
/// [`crate::SfntFont::open`] extends a too-short table into the gap up
/// to the next SFNT table when there is room, exactly like FreeType's
/// `num_locations` adjustment.
#[derive(Clone, Copy, Debug)]
pub struct LocaTable<'a> {
    data: &'a [u8],
    long: bool,
}

impl<'a> LocaTable<'a> {
    /// Wraps the `loca` bytes according to `head.indexToLocFormat`.
    ///
    /// The entry size follows `tt_face_load_loca`: format `0` selects
    /// 2-byte entries scaled by two, every other value selects 4-byte
    /// entries.  The number of entries is `data.len() / entry_size`.
    #[inline]
    pub const fn new(data: &'a [u8], index_to_loc_format: i16) -> Self {
        LocaTable {
            data,
            long: index_to_loc_format != 0,
        }
    }

    /// `true` when entries are 4-byte offsets (`Index_To_Loc_Format`
    /// other than 0).
    #[inline]
    pub const fn is_long(&self) -> bool {
        self.long
    }

    /// The entry size in bytes (2 or 4).
    #[inline]
    pub const fn entry_size(&self) -> usize {
        if self.long { 4 } else { 2 }
    }

    /// The number of stored locations (normally `numGlyphs + 1`).
    #[inline]
    pub fn num_entries(&self) -> usize {
        self.data.len() / self.entry_size()
    }

    /// The raw bytes of the (effective) table.
    #[inline]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The byte offset stored at `index`; short entries are doubled.
    ///
    /// Returns `None` when `index` leaves the table.
    #[inline]
    pub fn entry(&self, index: usize) -> Option<u32> {
        if index >= self.num_entries() {
            return None;
        }
        let pos = index.checked_mul(self.entry_size())?;
        let mut at = self.data.get(pos..)?;
        if self.long {
            at.try_get_u32().ok()
        } else {
            at.try_get_u16()
                .ok()
                .map(|value| u32::from(value) * 2)
        }
    }
}

/// A usable horizontal format-0 subtable of `kern`
/// (`tt_face_load_kern`'s per-subtable state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KernSub {
    /// Offset of the first `(key, value)` pair inside the table.
    pairs: usize,
    /// Number of pairs after the broken-count clamp.
    num_pairs: u32,
    /// The raw `coverage` field (`bit 3` = override).
    coverage: u16,
    /// `true` when the pair keys are strictly increasing (binary
    /// search is then possible, see `tt_face_load_kern`).
    ordered: bool,
}

/// The `kern` table reader (`tt_face_load_kern` / `tt_face_get_kerning`,
/// `tttkern.c`).
///
/// Only horizontal subtables with format 0 are retained; Apple format 2
/// subtables are unsupported in FreeType 2.6 as well ("we haven't seen
/// a single font using it").  Subtables whose declared `length` leaves
/// the table, or whose pair count exceeds the bytes present, are
/// clamped exactly like the C code.
#[derive(Clone, Debug)]
pub struct KernTable<'a> {
    data: &'a [u8],
    subtables: Vec<KernSub>,
}

impl<'a> KernTable<'a> {
    /// Parses the `kern` table from `data`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than 4 bytes
    ///   (`tt_face_load_kern`: "kerning table is too small").
    ///
    /// A malformed subtable is skipped or clamped rather than failing
    /// the whole table, matching `tt_face_load_kern`.
    pub fn parse(data: &'a [u8]) -> TtResult<Self> {
        if data.len() < 4 {
            return Err(TtError::INVALID_TABLE);
        }
        let invalid = |_| TtError::INVALID_TABLE;
        let mut header: &[u8] = data;
        let _version = header.try_get_u16().map_err(invalid)?;
        let declared = header.try_get_u16().map_err(invalid)?;
        let num_tables = declared.min(32);
        let mut subtables = Vec::new();
        let mut pos = 4usize;
        for _ in 0..num_tables {
            if pos + 6 > data.len() {
                break;
            }
            let Some(mut sub_header) = data.get(pos..pos + 6) else {
                break;
            };
            let _sub_version = sub_header.try_get_u16().unwrap_or(0);
            let length = usize::from(sub_header.try_get_u16().unwrap_or(0));
            let coverage = sub_header.try_get_u16().unwrap_or(0);
            if length <= 14 {
                break;
            }
            let end = pos.saturating_add(length).min(data.len());
            if (coverage & !8) == 1 && pos + 14 <= end && (coverage >> 8) == 0 {
                let pairs = pos + 14;
                let num_pairs = data
                    .get(pos + 6..pos + 8)
                    .and_then(|mut at| at.try_get_u16().ok())
                    .unwrap_or(0);
                let available = ((end - pairs) / 6) as u32;
                let num_pairs = u32::from(num_pairs).min(available);
                let ordered = is_ordered(data, pairs, num_pairs);
                subtables.push(KernSub {
                    pairs,
                    num_pairs,
                    coverage,
                    ordered,
                });
            }
            pos += length;
        }
        Ok(KernTable { data, subtables })
    }

    /// `true` when the table contains no usable horizontal subtable
    /// (the `FT_FACE_FLAG_KERNING` predicate of `sfnt_load_face`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.subtables.is_empty()
    }

    /// The number of usable horizontal format-0 subtables.
    #[inline]
    pub fn len(&self) -> usize {
        self.subtables.len()
    }

    /// `tt_face_get_kerning` (`ttkern.c`): the accumulated kerning
    /// adjustment of the `(left, right)` glyph pair in font units.
    ///
    /// Every usable subtable is scanned: a format-0 pair hit *adds* its
    /// value, or *replaces* the result when `coverage & 8` (override)
    /// is set; subtables without the pair contribute nothing.
    pub fn kerning(&self, left: u32, right: u32) -> i32 {
        let key = (left << 16) | right;
        let mut result = 0i32;
        for sub in &self.subtables {
            let Some(value) = self.lookup(sub, key) else {
                continue;
            };
            if sub.coverage & 8 != 0 {
                result = i32::from(value);
            } else {
                result += i32::from(value);
            }
        }
        result
    }

    /// Binary search (`sub.ordered`) or linear scan of the pairs of one
    /// subtable; `None` when the key is absent.
    fn lookup(&self, sub: &KernSub, key: u32) -> Option<i16> {
        let count = sub.num_pairs as usize;
        if sub.ordered {
            let mut min = 0usize;
            let mut max = count;
            while min < max {
                let mid = (min + max) >> 1;
                let pos = sub.pairs + mid * 6;
                let mut record = self.data.get(pos..pos + 6)?;
                let current = record.try_get_u32().ok()?;
                if current == key {
                    return record.try_get_i16().ok();
                }
                if current < key {
                    min = mid + 1;
                } else {
                    max = mid;
                }
            }
            None
        } else {
            for index in 0..count {
                let pos = sub.pairs + index * 6;
                let mut record = self.data.get(pos..pos + 6)?;
                let Ok(current) = record.try_get_u32() else {
                    return None;
                };
                if current == key {
                    return record.try_get_i16().ok();
                }
            }
            None
        }
    }
}

/// `tt_face_load_kern`'s ordering scan: `true` when the `num_pairs`
/// keys starting at `pairs` are strictly increasing.
fn is_ordered(data: &[u8], pairs: usize, num_pairs: u32) -> bool {
    if num_pairs == 0 {
        return false;
    }
    let Some(mut first) = data.get(pairs..pairs + 6) else {
        return false;
    };
    let Some(mut previous) = first.try_get_u32().ok() else {
        return false;
    };
    for index in 1..num_pairs {
        let pos = pairs + (index as usize) * 6;
        let Some(mut record) = data.get(pos..pos + 6) else {
            return false;
        };
        let Some(current) = record.try_get_u32().ok() else {
            return false;
        };
        if current <= previous {
            return false;
        }
        previous = current;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    type KernFixture<'a> = (u16, u16, &'a [(u32, i16)]);

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn push_i16(out: &mut Vec<u8>, value: i16) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn push_i64(out: &mut Vec<u8>, value: i64) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn head_fixture() -> Vec<u8> {
        head_with(crate::tags::HEAD_MAGIC_NUMBER, 1)
    }

    fn head_with(magic_number: u32, index_to_loc_format: i16) -> Vec<u8> {
        let mut out = Vec::with_capacity(54);
        push_u32(&mut out, 0x0001_0000);
        push_u32(&mut out, 0x0002_0003);
        push_u32(&mut out, 0xDEAD_BEEF);
        push_u32(&mut out, magic_number);
        push_u16(&mut out, 0x000B);
        push_u16(&mut out, 2048);
        push_i64(&mut out, -5);
        push_i64(&mut out, 3_600_000_000);
        push_i16(&mut out, -100);
        push_i16(&mut out, -200);
        push_i16(&mut out, 900);
        push_i16(&mut out, 800);
        push_u16(&mut out, 0x0003);
        push_u16(&mut out, 7);
        push_i16(&mut out, -1);
        push_i16(&mut out, index_to_loc_format);
        push_i16(&mut out, 0);
        out
    }

    fn maxp_fixture(version: u32, num_glyphs: u16, extra: Option<&[u16; 13]>) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        push_u32(&mut out, version);
        push_u16(&mut out, num_glyphs);
        if let Some(fields) = extra {
            for field in fields {
                push_u16(&mut out, *field);
            }
        }
        out
    }

    fn hhea_fixture() -> Vec<u8> {
        let mut out = Vec::with_capacity(36);
        push_u32(&mut out, 0x0001_0000);
        push_i16(&mut out, 800);
        push_i16(&mut out, -200);
        push_i16(&mut out, 90);
        push_u16(&mut out, 1000);
        push_i16(&mut out, -50);
        push_i16(&mut out, -40);
        push_i16(&mut out, 950);
        push_i16(&mut out, 1);
        push_i16(&mut out, 0);
        push_i16(&mut out, 5);
        push_i16(&mut out, 7);
        push_i16(&mut out, -8);
        push_i16(&mut out, 9);
        push_i16(&mut out, -10);
        push_i16(&mut out, -3);
        push_u16(&mut out, 42);
        out
    }

    fn hmtx_fixture() -> Vec<u8> {
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

    fn kern_subtable(coverage: u16, declared_pairs: u16, pairs: &[(u32, i16)]) -> Vec<u8> {
        let declared_length = 14 + usize::from(declared_pairs) * 6;
        let mut out = Vec::with_capacity(14 + pairs.len() * 6);
        push_u16(&mut out, 0);
        push_u16(&mut out, declared_length as u16);
        push_u16(&mut out, coverage);
        push_u16(&mut out, declared_pairs);
        push_u16(&mut out, 12);
        push_u16(&mut out, 1);
        push_u16(&mut out, 0);
        for (key, value) in pairs {
            push_u32(&mut out, *key);
            push_i16(&mut out, *value);
        }
        out
    }

    fn kern_fixture(subtables: &[KernFixture]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 0);
        push_u16(&mut out, subtables.len() as u16);
        for (coverage, declared, pairs) in subtables {
            out.extend_from_slice(&kern_subtable(*coverage, *declared, pairs));
        }
        out
    }

    #[test]
    fn head_parses_every_field() {
        let head = HeadTable::parse(&head_fixture()).unwrap();
        assert_eq!(head.table_version, 0x0001_0000);
        assert_eq!(head.font_revision, 0x0002_0003);
        assert_eq!(head.check_sum_adjustment, 0xDEAD_BEEF);
        assert_eq!(head.magic_number, 0x5F0F_3CF5);
        assert_eq!(head.flags, 0x000B);
        assert_eq!(head.units_per_em, 2048);
        assert_eq!(head.created, -5);
        assert_eq!(head.modified, 3_600_000_000);
        assert_eq!(head.bbox, BBox::from_edges(-100, -200, 900, 800));
        assert_eq!(head.mac_style, 0x0003);
        assert_eq!(head.lowest_rec_ppem, 7);
        assert_eq!(head.font_direction, -1);
        assert_eq!(head.index_to_loc_format, 1);
        assert_eq!(head.glyph_data_format, 0);
    }

    #[test]
    fn head_rejects_truncated_table() {
        let fixture = head_fixture();
        for len in [0usize, 1, 12, 53] {
            let result = HeadTable::parse(&fixture[..len]);
            assert_eq!(result, Err(TtError::INVALID_TABLE));
        }
        assert!(HeadTable::parse(&fixture[..54]).is_ok());
    }

    #[test]
    fn head_magic_and_loca_format_helpers() {
        let valid = HeadTable::parse(&head_fixture()).unwrap();
        assert!(valid.has_valid_magic());
        assert!(valid.uses_long_loca());

        let broken = HeadTable::parse(&head_with(0x1234_5678, 1)).unwrap();
        assert!(!broken.has_valid_magic());

        let short_loca = HeadTable::parse(&head_with(crate::tags::HEAD_MAGIC_NUMBER, 0)).unwrap();
        assert!(!short_loca.uses_long_loca());
        assert!(short_loca.has_valid_magic());
    }

    #[test]
    fn maxp_parses_short_version() {
        let table = MaxpTable::parse(&maxp_fixture(0x0000_5000, 7, None)).unwrap();
        assert_eq!(table.version, 0x0000_5000);
        assert_eq!(table.num_glyphs, 7);
        assert_eq!(table.max_points, 0);
        assert_eq!(table.max_component_depth, 0);
    }

    #[test]
    fn maxp_short_version_ignores_extra_bytes() {
        let extra = [1u16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let table = MaxpTable::parse(&maxp_fixture(0x0000_5000, 9, Some(&extra))).unwrap();
        assert_eq!(table.num_glyphs, 9);
        assert_eq!(table.max_points, 0);
        assert_eq!(table.max_instruction_defs, 0);
        assert_eq!(table.max_component_elements, 0);
    }

    #[test]
    fn maxp_parses_full_version() {
        let extra = [11u16, 22, 33, 44, 2, 65531, 55, 100, 66, 77, 88, 99, 100];
        let table = MaxpTable::parse(&maxp_fixture(0x0001_0000, 0x1234, Some(&extra))).unwrap();
        assert_eq!(table.version, 0x0001_0000);
        assert_eq!(table.num_glyphs, 0x1234);
        assert_eq!(table.max_points, 11);
        assert_eq!(table.max_contours, 22);
        assert_eq!(table.max_composite_points, 33);
        assert_eq!(table.max_composite_contours, 44);
        assert_eq!(table.max_zones, 2);
        assert_eq!(table.max_twilight_points, 65531);
        assert_eq!(table.max_storage, 55);
        assert_eq!(table.max_function_defs, 100);
        assert_eq!(table.max_instruction_defs, 66);
        assert_eq!(table.max_stack_elements, 77);
        assert_eq!(table.max_size_of_instructions, 88);
        assert_eq!(table.max_component_elements, 99);
        assert_eq!(table.max_component_depth, 100);
    }

    #[test]
    fn maxp_applies_ttload_clamps() {
        let mut extra = [1u16; 13];
        extra[5] = 0xFFFF;
        extra[7] = 10;
        extra[12] = 500;
        let table = MaxpTable::parse(&maxp_fixture(0x0001_0000, 5, Some(&extra))).unwrap();
        assert_eq!(table.max_twilight_points, 0xFFFB);
        assert_eq!(table.max_function_defs, 64);
        assert_eq!(table.max_instruction_defs, 1);
        assert_eq!(table.max_component_depth, 100);

        let mut extra = [0u16; 13];
        extra[5] = 0xFFFB;
        extra[7] = 63;
        extra[12] = 101;
        let table = MaxpTable::parse(&maxp_fixture(0x0001_0000, 5, Some(&extra))).unwrap();
        assert_eq!(table.max_twilight_points, 0xFFFB);
        assert_eq!(table.max_function_defs, 64);
        assert_eq!(table.max_component_depth, 100);

        let mut extra = [0u16; 13];
        extra[5] = 0xFFFA;
        extra[7] = 65;
        extra[12] = 99;
        let table = MaxpTable::parse(&maxp_fixture(0x0001_0000, 5, Some(&extra))).unwrap();
        assert_eq!(table.max_twilight_points, 0xFFFA);
        assert_eq!(table.max_function_defs, 65);
        assert_eq!(table.max_component_depth, 99);
    }

    #[test]
    fn maxp_rejects_bad_lengths() {
        let full = maxp_fixture(0x0001_0000, 5, Some(&[0u16; 13]));
        for len in [0usize, 1, 5] {
            assert_eq!(
                MaxpTable::parse(&full[..len]),
                Err(TtError::INVALID_TABLE),
                "len {len}"
            );
        }
        for len in [6usize, 7, 8, 30, 31] {
            assert_eq!(
                MaxpTable::parse(&full[..len]),
                Err(TtError::INVALID_TABLE),
                "len {len}"
            );
        }
        assert!(MaxpTable::parse(&full[..32]).is_ok());
        let short = maxp_fixture(0x0000_5000, 5, None);
        assert!(MaxpTable::parse(&short[..6]).is_ok());
        assert_eq!(MaxpTable::parse(&short[..5]), Err(TtError::INVALID_TABLE));
    }

    #[test]
    fn hhea_parses_every_field() {
        let hhea = HheaTable::parse(&hhea_fixture()).unwrap();
        assert_eq!(hhea.version, 0x0001_0000);
        assert_eq!(hhea.ascender, 800);
        assert_eq!(hhea.descender, -200);
        assert_eq!(hhea.line_gap, 90);
        assert_eq!(hhea.advance_width_max, 1000);
        assert_eq!(hhea.min_left_side_bearing, -50);
        assert_eq!(hhea.min_right_side_bearing, -40);
        assert_eq!(hhea.x_max_extent, 950);
        assert_eq!(hhea.caret_slope_rise, 1);
        assert_eq!(hhea.caret_slope_run, 0);
        assert_eq!(hhea.caret_offset, 5);
        assert_eq!(hhea.reserved, [7, -8, 9, -10]);
        assert_eq!(hhea.metric_data_format, -3);
        assert_eq!(hhea.number_of_h_metrics, 42);
    }

    #[test]
    fn hhea_rejects_truncated_table() {
        let fixture = hhea_fixture();
        assert_eq!(fixture.len(), 36);
        assert_eq!(HheaTable::parse(&fixture[..35]), Err(TtError::INVALID_TABLE));
        assert_eq!(HheaTable::parse(&fixture[..0]), Err(TtError::INVALID_TABLE));
        assert!(HheaTable::parse(&fixture[..36]).is_ok());
    }

    #[test]
    fn hmtx_full_records() {
        let fixture = hmtx_fixture();
        let hmtx = HmtxTable::new(&fixture, 3);
        assert_eq!(hmtx.number_of_h_metrics(), 3);
        assert_eq!(hmtx.data().len(), 16);
        assert_eq!(hmtx.metrics(0), (500, -10));
        assert_eq!(hmtx.metrics(1), (600, 20));
        assert_eq!(hmtx.metrics(2), (700, -30));
        assert_eq!(hmtx.advance_width(1), 600);
        assert_eq!(hmtx.left_side_bearing(1), 20);
    }

    #[test]
    fn hmtx_short_array_reuses_last_advance() {
        let fixture = hmtx_fixture();
        let hmtx = HmtxTable::new(&fixture, 3);
        assert_eq!(hmtx.metrics(3), (700, 40));
        assert_eq!(hmtx.metrics(4), (700, -50));
        assert_eq!(hmtx.metrics(5), (700, 0));
    }

    #[test]
    fn hmtx_zero_metrics_reports_no_data() {
        let fixture = hmtx_fixture();
        let hmtx = HmtxTable::new(&fixture, 0);
        assert_eq!(hmtx.metrics(0), (0, 0));
        assert_eq!(hmtx.metrics(99), (0, 0));
    }

    #[test]
    fn hmtx_out_of_range_reports_no_data() {
        let fixture = hmtx_fixture();
        let truncated = HmtxTable::new(&fixture[..2], 3);
        assert_eq!(truncated.metrics(0), (0, 0));
        assert_eq!(truncated.metrics(2), (0, 0));
        assert_eq!(truncated.metrics(3), (0, 0));

        let no_records = HmtxTable::new(&[], 3);
        assert_eq!(no_records.metrics(0), (0, 0));

        let missing_last = HmtxTable::new(&fixture[..6], 3);
        assert_eq!(missing_last.metrics(2), (0, 0));
        assert_eq!(missing_last.metrics(3), (0, 0));
    }

    #[test]
    fn loca_short_entries_are_doubled() {
        let mut data = Vec::new();
        for value in [0u16, 5, 15, 23, 30] {
            push_u16(&mut data, value);
        }
        let loca = LocaTable::new(&data, 0);
        assert!(!loca.is_long());
        assert_eq!(loca.entry_size(), 2);
        assert_eq!(loca.num_entries(), 5);
        assert_eq!(loca.entry(0), Some(0));
        assert_eq!(loca.entry(1), Some(10));
        assert_eq!(loca.entry(2), Some(30));
        assert_eq!(loca.entry(3), Some(46));
        assert_eq!(loca.entry(4), Some(60));
        assert_eq!(loca.entry(5), None);
        assert_eq!(loca.data(), data.as_slice());
    }

    #[test]
    fn loca_long_entries_are_verbatim() {
        let mut data = Vec::new();
        for value in [0u32, 10, 30, 100_000] {
            push_u32(&mut data, value);
        }
        let loca = LocaTable::new(&data, 1);
        assert!(loca.is_long());
        assert_eq!(loca.entry_size(), 4);
        assert_eq!(loca.num_entries(), 4);
        assert_eq!(loca.entry(0), Some(0));
        assert_eq!(loca.entry(2), Some(30));
        assert_eq!(loca.entry(3), Some(100_000));
        assert_eq!(loca.entry(4), None);
        assert_eq!(LocaTable::new(&data, -2).entry(3), Some(100_000));
    }

    #[test]
    fn loca_odd_length_keeps_whole_entries_only() {
        let data = [0x00u8, 0x01, 0x00, 0x02, 0xFF];
        let loca = LocaTable::new(&data, 0);
        assert_eq!(loca.num_entries(), 2);
        assert_eq!(loca.entry(0), Some(2));
        assert_eq!(loca.entry(1), Some(4));
        assert_eq!(loca.entry(2), None);
    }

    #[test]
    fn kern_rejects_tiny_table() {
        for len in [0usize, 1, 2, 3] {
            assert!(matches!(
                KernTable::parse(&[0u8; 4][..len]),
                Err(TtError::INVALID_TABLE)
            ));
        }
    }

    #[test]
    fn kern_ordered_binary_lookup() {
        let data = kern_fixture(&[(0x0001, 2, &[(0x0001_0002, -40), (0x0003_0004, 25)])]);
        let kern = KernTable::parse(&data).unwrap();
        assert!(!kern.is_empty());
        assert_eq!(kern.len(), 1);
        assert_eq!(kern.kerning(1, 2), -40);
        assert_eq!(kern.kerning(3, 4), 25);
        assert_eq!(kern.kerning(1, 4), 0);
        assert_eq!(kern.kerning(0, 0), 0);
        assert_eq!(kern.kerning(4, 3), 0);
    }

    #[test]
    fn kern_unordered_falls_back_to_linear_scan() {
        let data = kern_fixture(&[(0x0001, 2, &[(0x0003_0004, 25), (0x0001_0002, -40)])]);
        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.len(), 1);
        assert_eq!(kern.kerning(1, 2), -40);
        assert_eq!(kern.kerning(3, 4), 25);
        assert_eq!(kern.kerning(2, 2), 0);
    }

    #[test]
    fn kern_accumulates_and_overrides() {
        let data = kern_fixture(&[
            (0x0001, 1, &[(0x0001_0002, -40)]),
            (0x0001, 1, &[(0x0001_0002, 5)]),
        ]);
        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.len(), 2);
        assert_eq!(kern.kerning(1, 2), -35);

        let data = kern_fixture(&[
            (0x0001, 1, &[(0x0001_0002, -40)]),
            (0x0009, 1, &[(0x0001_0002, 70)]),
        ]);
        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.kerning(1, 2), 70);

        let data = kern_fixture(&[
            (0x0009, 1, &[(0x0001_0002, 70)]),
            (0x0001, 1, &[(0x0001_0002, -40)]),
        ]);
        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.kerning(1, 2), 30);
    }

    #[test]
    fn kern_skips_non_horizontal_and_non_format_zero() {
        let data = kern_fixture(&[
            (0x0002, 1, &[(0x0001_0002, 100)]),
            (0x0101, 1, &[(0x0001_0002, 200)]),
        ]);
        let kern = KernTable::parse(&data).unwrap();
        assert!(kern.is_empty());
        assert_eq!(kern.len(), 0);
        assert_eq!(kern.kerning(1, 2), 0);

        let data = kern_fixture(&[(0x0001, 1, &[(0x0001_0002, -40)])]);
        let kern = KernTable::parse(&data).unwrap();
        assert!(!kern.is_empty());
    }

    #[test]
    fn kern_clamps_broken_pair_counts() {
        let mut data = kern_fixture(&[(0x0001, 2, &[(0x0001_0002, -40), (0x0003_0004, 25)])]);
        let declared_pairs_pos = 4 + 6;
        data[declared_pairs_pos..declared_pairs_pos + 2].copy_from_slice(&100u16.to_be_bytes());
        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.len(), 1);
        assert_eq!(kern.kerning(1, 2), -40);
        assert_eq!(kern.kerning(3, 4), 25);
        assert_eq!(kern.kerning(5, 6), 0);
    }

    #[test]
    fn kern_empty_and_broken_subtables_yield_no_kerning() {
        let data = kern_fixture(&[]);
        let kern = KernTable::parse(&data).unwrap();
        assert!(kern.is_empty());
        assert_eq!(kern.kerning(1, 2), 0);

        let data = kern_fixture(&[(0x0001, 1, &[])]);
        let kern = KernTable::parse(&data).unwrap();
        assert!(!kern.is_empty());
        assert_eq!(kern.kerning(1, 2), 0);
    }

    #[test]
    fn kern_header_clamps_declared_subtable_count() {
        let mut data = kern_fixture(&[(0x0001, 1, &[(0x0001_0002, -40)])]);
        data[2..4].copy_from_slice(&64u16.to_be_bytes());
        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.len(), 1);
        assert_eq!(kern.kerning(1, 2), -40);
    }
}
