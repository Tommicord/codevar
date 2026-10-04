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

//! `COLR`: color glyph layers.
//!
//! A faithful port of the table validation and the v0 layer iterator of
//! FreeType 2.13's `src/sfnt/ttcolr.c`
//! (`tt_face_load_colr`, `find_base_glyph_record`,
//! `tt_face_get_colr_layer`).  The table is validated once in
//! [`ColrTable::parse`]; after that [`ColrTable::layers`] walks the
//! base glyph's layer records exactly like `FT_Get_Color_Glyph_Layer`.
//!
//! ## Scope
//!
//! Only `COLR` version 0 layers are *rendered* (see
//! [`crate::render`]).  The version 1 header (BaseGlyphV1List,
//! LayerList, ClipList, VarIdxMap, VarStore) is parsed and validated so
//! that v1 fonts keep their v0 compatibility layers usable, but v1
//! paint graphs, clip boxes and item variation stores are **not**
//! implemented ([`TtError::UNIMPLEMENTED_FEATURE`]).

use codevar_truetype_core::{TtError, TtResult};

use crate::cursor::{Reader, read_u16_at, read_u32_at};

/// `BASE_GLYPH_SIZE`: one v0 `BaseGlyphRecord` (gid, firstLayerIndex,
/// numLayers) is six bytes.
const BASE_GLYPH_SIZE: usize = 6;
/// `LAYER_SIZE`: one v0 `LayerRecord` (gid, paletteIndex) is four
/// bytes.
const LAYER_SIZE: usize = 4;
/// `BASE_GLYPH_PAINT_RECORD_SIZE`: one v1 `BaseGlyphV1Record` (gid,
/// paint offset) is six bytes.
const BASE_GLYPH_PAINT_RECORD_SIZE: usize = 6;
/// `LAYER_V1_LIST_PAINT_OFFSET_SIZE`: one v1 `LayerList` paint offset
/// is four bytes.
const LAYER_V1_LIST_PAINT_OFFSET_SIZE: usize = 4;
/// `COLRV0_HEADER_SIZE`: version, numBaseGlyphRecords,
/// baseGlyphRecordsOffset, layerRecordsOffset, numLayerRecords.
const COLRV0_HEADER_SIZE: usize = 14;
/// `COLRV1_HEADER_SIZE`: the v0 header plus five v1 `Offset32`
/// fields (baseGlyphs, layerList, clipList, varIdxMap, varStore).
const COLRV1_HEADER_SIZE: usize = 34;

/// The `0xFFFF` palette index meaning "the text foreground color"
/// (`FT_COLR_PAINT_FOREGROUND`).
pub const COLOR_FOREGROUND: u16 = 0xFFFF;

/// A parsed `COLR` table, borrowing the font bytes.
#[derive(Clone, Debug)]
pub struct ColrTable<'a> {
    /// The `COLR` version (0 or 1).
    version: u16,
    /// The number of v0 base glyph records.
    num_base_glyphs: u16,
    /// The number of v0 layer records.
    num_layers: u16,
    /// The v0 `BaseGlyphRecords` array (`6 * num_base_glyphs` bytes).
    base_glyphs: &'a [u8],
    /// The v0 `LayerRecords` array (`4 * num_layers` bytes).
    layers: &'a [u8],
    /// `true` when the (validated) v1 header advertises a non-empty
    /// `BaseGlyphV1List`.
    has_base_glyphs_v1: bool,
}

impl<'a> ColrTable<'a> {
    /// Parses a `COLR` table, mirroring `tt_face_load_colr`
    /// validation for validation (without the GX variation store,
    /// which this port does not model).
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the table is shorter than its
    ///   header, the version is above 1, an array offset is out of
    ///   range, or an array count overruns the table.
    pub fn parse(data: &'a [u8]) -> TtResult<Self> {
        if data.len() < COLRV0_HEADER_SIZE {
            return Err(TtError::INVALID_TABLE);
        }
        let mut reader = Reader::new(data);
        let version = reader.u16()?;
        if version != 0 && version != 1 {
            return Err(TtError::INVALID_TABLE);
        }
        let num_base_glyphs = reader.u16()?;
        let base_glyph_offset = reader.u32()? as usize;
        let layer_offset = reader.u32()? as usize;
        let num_layers = reader.u16()?;

        let base_glyphs = Self::array(data, base_glyph_offset, num_base_glyphs, BASE_GLYPH_SIZE)?;
        let layers = Self::array(data, layer_offset, num_layers, LAYER_SIZE)?;

        let mut has_base_glyphs_v1 = false;
        if version == 1 {
            if data.len() < COLRV1_HEADER_SIZE {
                return Err(TtError::INVALID_TABLE);
            }
            // BaseGlyphV1List: an Offset32 whose target must hold a
            // u32 count inside the table (`table_size - 4 <= offset`
            // is FreeType's rejection test).
            let base_glyphs_v1 = read_u32_at(data, 14)? as usize;
            if data.len() - 4 <= base_glyphs_v1 {
                return Err(TtError::INVALID_TABLE);
            }
            let count = read_u32_at(data, base_glyphs_v1)? as usize;
            if (data.len() - base_glyphs_v1) / BASE_GLYPH_PAINT_RECORD_SIZE < count {
                return Err(TtError::INVALID_TABLE);
            }
            has_base_glyphs_v1 = count != 0;

            // LayerList (optional): u32 count followed by Offset32s.
            let layer_list = read_u32_at(data, 18)? as usize;
            if data.len() <= layer_list {
                return Err(TtError::INVALID_TABLE);
            }
            if layer_list != 0 {
                if data.len() - 4 <= layer_list {
                    return Err(TtError::INVALID_TABLE);
                }
                let count = read_u32_at(data, layer_list)? as usize;
                if (data.len() - layer_list) / LAYER_V1_LIST_PAINT_OFFSET_SIZE < count {
                    return Err(TtError::INVALID_TABLE);
                }
            }
            // ClipList (optional).
            let clip_list = read_u32_at(data, 22)? as usize;
            if data.len() <= clip_list {
                return Err(TtError::INVALID_TABLE);
            }
        }

        Ok(ColrTable {
            version,
            num_base_glyphs,
            num_layers,
            base_glyphs,
            layers,
            has_base_glyphs_v1,
        })
    }

    /// Validates one record array: `offset + count * size` must stay
    /// inside `data`, mirroring FreeType's
    /// `table_size <= offset || (table_size - offset) / size < count`
    /// checks.
    fn array(data: &'a [u8], offset: usize, count: u16, size: usize) -> TtResult<&'a [u8]> {
        if offset > data.len() {
            return Err(TtError::INVALID_TABLE);
        }
        let bytes = count as usize;
        if data.len() - offset < bytes.saturating_mul(size) {
            return Err(TtError::INVALID_TABLE);
        }
        let length = bytes
            .checked_mul(size)
            .ok_or(TtError::INVALID_TABLE)?;
        data.get(offset..offset + length)
            .ok_or(TtError::INVALID_TABLE)
    }

    /// The `COLR` version (0 or 1).
    #[inline]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// The number of v0 base glyph records.
    #[inline]
    pub const fn num_base_glyphs(&self) -> u16 {
        self.num_base_glyphs
    }

    /// The number of v0 layer records.
    #[inline]
    pub const fn num_layers(&self) -> u16 {
        self.num_layers
    }

    /// `true` when the table also carries a validated (non-empty)
    /// v1 `BaseGlyphV1List`.  Paint rendering is not implemented;
    /// see the module documentation.
    #[inline]
    pub const fn has_v1_paints(&self) -> bool {
        self.has_base_glyphs_v1
    }

    /// Binary-searches the v0 base glyph record for `gindex`
    /// (`find_base_glyph_record`), returning
    /// `(first_layer_index, num_layers)`.
    ///
    /// Returns `None` when the glyph has no v0 color layers.
    pub fn base_glyph(&self, gindex: u32) -> Option<(u16, u16)> {
        if gindex > u16::MAX as u32 {
            return None;
        }
        let gid = gindex as u16;
        let mut min = 0usize;
        let mut max = self.num_base_glyphs as usize;
        while min < max {
            let mid = min + (max - min) / 2;
            let record = self.base_glyphs.get(mid * BASE_GLYPH_SIZE..)?;
            let record_gid = u16::from_be_bytes(record[0..2].try_into().unwrap_or([0, 0]));
            if record_gid < gid {
                min = mid + 1;
            } else if record_gid > gid {
                max = mid;
            } else {
                let first = u16::from_be_bytes(record[2..4].try_into().unwrap_or([0, 0]));
                let num = u16::from_be_bytes(record[4..6].try_into().unwrap_or([0, 0]));
                return Some((first, num));
            }
        }
        None
    }

    /// Layer record `index` of the v0 layer array, as
    /// `(glyph_index, palette_index)`.
    ///
    /// Returns `None` when the record is outside the validated array
    /// (FreeType stops the layer iteration in that case).
    pub fn layer(&self, index: u16) -> Option<(u16, u16)> {
        let record = self.layers.get((index as usize) * LAYER_SIZE..)?;
        let glyph = u16::from_be_bytes(record[0..2].try_into().unwrap_or([0, 0]));
        let color = u16::from_be_bytes(record[2..4].try_into().unwrap_or([0, 0]));
        Some((glyph, color))
    }

    /// Starts a v0 layer iteration over `gindex`, or returns `None`
    /// when the glyph has no layers (`FT_Get_Color_Glyph_Layer`'s
    /// first call).
    ///
    /// The iterator stops early when a record points at a glyph index
    /// `>= num_glyphs` or at a palette index `>= num_palette_entries`
    /// (other than [`COLOR_FOREGROUND`]), reproducing
    /// `tt_face_get_colr_layer`'s per-record validation.
    pub fn layers(&self, gindex: u32, num_glyphs: u16, num_palette_entries: u16) -> Option<LayerIter<'a>> {
        let (first, num) = self.base_glyph(gindex)?;
        if num == 0 {
            return None;
        }
        // FreeType rejects layer ranges that run past the table; this
        // port checks them against the validated layer array instead
        // (stricter, same effect for well-formed tables).
        let end = (first as u32).checked_add(num as u32)?;
        if end > self.num_layers as u32 {
            return None;
        }
        Some(LayerIter {
            layers: self.layers,
            next: first,
            remaining: num,
            num_glyphs,
            num_palette_entries,
        })
    }
}

/// The v0 layer iterator returned by [`ColrTable::layers`]
/// (`FT_LayerIterator` + `tt_face_get_colr_layer`).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerIter<'a> {
    /// The validated `LayerRecords` array.
    layers: &'a [u8],
    /// The next layer record index.
    next: u16,
    /// How many records remain.
    remaining: u16,
    /// The face's glyph count (record validation).
    num_glyphs: u16,
    /// The active palette's entry count (record validation).
    num_palette_entries: u16,
}

impl Iterator for LayerIter<'_> {
    /// `(glyph_index, palette_index)`; `palette_index` may be
    /// [`COLOR_FOREGROUND`].
    type Item = (u16, u16);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let index = self.next;
        self.next = self.next.checked_add(1)?;
        let (glyph, color) = layer_record(self.layers, index)?;
        if glyph >= self.num_glyphs || (color != COLOR_FOREGROUND && color >= self.num_palette_entries) {
            // `tt_face_get_colr_layer` aborts the iteration on an
            // out-of-range record.
            return None;
        }
        Some((glyph, color))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for LayerIter<'_> {}

/// Decodes layer record `index` of a raw `LayerRecords` slice.
fn layer_record(layers: &[u8], index: u16) -> Option<(u16, u16)> {
    let record = layers.get((index as usize) * LAYER_SIZE..)?;
    let glyph = read_u16_at(record, 0).ok()?;
    let color = read_u16_at(record, 2).ok()?;
    Some((glyph, color))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Builds a `COLR` v0 table from base glyph and layer records.
    fn colr_v0(bases: &[(u16, u16, u16)], layers: &[(u16, u16)]) -> Vec<u8> {
        let base_offset = COLRV0_HEADER_SIZE;
        let layer_offset = base_offset + bases.len() * BASE_GLYPH_SIZE;
        let mut out = Vec::new();
        out.extend_from_slice(&0u16.to_be_bytes()); // version
        out.extend_from_slice(&(bases.len() as u16).to_be_bytes());
        out.extend_from_slice(&(base_offset as u32).to_be_bytes());
        out.extend_from_slice(&(layer_offset as u32).to_be_bytes());
        out.extend_from_slice(&(layers.len() as u16).to_be_bytes());
        for &(gid, first, num) in bases {
            out.extend_from_slice(&gid.to_be_bytes());
            out.extend_from_slice(&first.to_be_bytes());
            out.extend_from_slice(&num.to_be_bytes());
        }
        for &(gid, color) in layers {
            out.extend_from_slice(&gid.to_be_bytes());
            out.extend_from_slice(&color.to_be_bytes());
        }
        out
    }

    #[test]
    fn binary_search_finds_base_glyphs_in_glyph_order() {
        let bytes = colr_v0(&[(10, 0, 2), (20, 2, 1)], &[(1, 0), (2, 1), (3, 2)]);
        let colr = ColrTable::parse(&bytes).unwrap();

        assert_eq!(colr.version(), 0);
        assert_eq!(colr.num_base_glyphs(), 2);
        assert_eq!(colr.num_layers(), 3);
        assert_eq!(colr.base_glyph(10), Some((0, 2)));
        assert_eq!(colr.base_glyph(20), Some((2, 1)));
        assert_eq!(colr.base_glyph(11), None);
        assert_eq!(colr.base_glyph(5), None);
        assert_eq!(colr.base_glyph(0xFFFF_FFFF), None);
    }

    #[test]
    fn layer_iteration_walks_records_and_stops_on_bad_records() {
        let bytes = colr_v0(&[(10, 0, 3)], &[(1, 0), (2, 7), (3, 2)]);
        let colr = ColrTable::parse(&bytes).unwrap();

        let layers: Vec<(u16, u16)> = colr.layers(10, 5, 8).unwrap().collect();
        assert_eq!(layers, alloc::vec![(1, 0), (2, 7), (3, 2)]);

        // Palette index past the palette: iteration stops right there.
        let layers: Vec<(u16, u16)> = colr.layers(10, 5, 7).unwrap().collect();
        assert_eq!(layers, alloc::vec![(1, 0)]);

        // Glyph index past the face: same.
        let layers: Vec<(u16, u16)> = colr.layers(10, 3, 8).unwrap().collect();
        assert_eq!(layers, alloc::vec![(1, 0), (2, 7)]);

        // The foreground sentinel is always accepted.
        let bytes = colr_v0(&[(10, 0, 1)], &[(1, COLOR_FOREGROUND)]);
        let colr = ColrTable::parse(&bytes).unwrap();
        assert_eq!(
            colr.layers(10, 5, 1).unwrap().collect::<Vec<_>>(),
            alloc::vec![(1, 0xFFFF)]
        );
    }

    #[test]
    fn layer_ranges_past_the_array_end_are_rejected() {
        let bytes = colr_v0(&[(10, 2, 5)], &[(1, 0), (2, 1)]);
        let colr = ColrTable::parse(&bytes).unwrap();
        assert_eq!(colr.layers(10, 5, 8), None);
    }

    #[test]
    fn zero_layer_records_yield_no_iteration() {
        let bytes = colr_v0(&[(10, 0, 0)], &[]);
        let colr = ColrTable::parse(&bytes).unwrap();
        assert!(colr.base_glyph(10).is_some());
        assert!(colr.layers(10, 5, 8).is_none());
    }

    #[test]
    fn malformed_tables_return_errors_and_never_panic() {
        // Shorter than the v0 header.
        assert!(matches!(ColrTable::parse(&[0; 13]), Err(TtError::INVALID_TABLE)));

        // Unknown version.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[0..2].copy_from_slice(&2u16.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // Base glyph offset past the end.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // Layer count overruns the table.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[12..14].copy_from_slice(&0xFFFFu16.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // v1 header cut short.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[0..2].copy_from_slice(&1u16.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));
    }

    #[test]
    fn v1_header_is_validated_but_paints_stay_unimplemented() {
        // v1 with a v0 header padded to 34 bytes: every v1 offset is
        // zero, so BaseGlyphV1List points at the table start whose
        // u32 "count" is garbage and overruns the table.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[0..2].copy_from_slice(&1u16.to_be_bytes());
        bytes.resize(COLRV1_HEADER_SIZE, 0);
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // A well-formed v1 header with an empty BaseGlyphV1List keeps
        // the v0 layers usable.
        //
        // Layout: v0 header (14) + v1 offsets (10) + base record (6)
        // + layer record (4) + BaseGlyphV1List count (4) + 2 pad bytes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes()); // version
        bytes.extend_from_slice(&1u16.to_be_bytes()); // numBaseGlyphRecords
        bytes.extend_from_slice(&34u32.to_be_bytes()); // base records at 34
        bytes.extend_from_slice(&40u32.to_be_bytes()); // layer records at 40
        bytes.extend_from_slice(&1u16.to_be_bytes()); // numLayerRecords
        bytes.extend_from_slice(&44u32.to_be_bytes()); // BaseGlyphV1List
        bytes.extend_from_slice(&0u32.to_be_bytes()); // layerList absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // clipList absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // varIdxMap absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // varStore absent
        bytes.extend_from_slice(&10u16.to_be_bytes()); // base glyph 10
        bytes.extend_from_slice(&0u16.to_be_bytes()); // first layer
        bytes.extend_from_slice(&1u16.to_be_bytes()); // layer count
        bytes.extend_from_slice(&1u16.to_be_bytes()); // layer glyph 1
        bytes.extend_from_slice(&0u16.to_be_bytes()); // palette entry 0
        bytes.extend_from_slice(&0u32.to_be_bytes()); // v1 list count = 0
        bytes.extend_from_slice(&[0, 0]); // padding

        let colr = ColrTable::parse(&bytes).unwrap();
        assert_eq!(colr.version(), 1);
        assert!(!colr.has_v1_paints());
        assert_eq!(colr.base_glyph(10), Some((0, 1)));
        assert_eq!(
            colr.layers(10, 5, 8).unwrap().collect::<Vec<_>>(),
            alloc::vec![(1, 0)]
        );
    }
}
