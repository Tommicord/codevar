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

//! `CPAL`: the color palette table.
//!
//! A faithful port of FreeType 2.13's `src/sfnt/ttcpal.c`
//! (`tt_face_load_cpal`, `tt_face_palette_set`): the table is parsed
//! directly out of the bytes handed to [`CpalTable::parse`], every
//! offset and length is validated up front, and palette colors are
//! served as decoded [`Color`] values on demand without copying the
//! whole table.
//!
//! ## Typical use
//!
//! ```ignore
//! let cpal = CpalTable::parse(font.table(tags::TAG_CPAL)?)?;
//! for color in cpal.palette(0)? {
//!     let [r, g, b, a] = color.rgba();
//! }
//! ```
//!
//! ## Deviations from FreeType 2.13
//!
//! * The palette lives in the caller's bytes instead of
//!   `face->palette`; [`CpalTable::palette`] replaces
//!   `tt_face_palette_set` and never mutates shared state.
//! * `face->palette_data.name_id` (the v0 `paletteNames` absence) and
//!   the `FT_Palette_*` public helpers are not modelled; the v1
//!   `paletteTypes`/`paletteLabels`/`paletteEntryLabels` arrays are
//!   exposed as slices instead.

use crate::cursor::Reader;
use codevar_truetype_core::{TtError, TtResult};

/// `CPAL_V0_HEADER_BASE_SIZE`: version, entries, palettes, colors and
/// the color-records array offset.
const CPAL_V0_HEADER_BASE_SIZE: usize = 12;
/// `COLOR_SIZE`: one `ColorRecord` is four bytes (BGRA on disk).
const COLOR_SIZE: usize = 4;

/// `FT_PALETTE_FOR_DARK_BACKGROUND`: the palette is meant to be shown
/// on a dark background (v1 `paletteTypes` bit 0).
pub const PALETTE_FOR_DARK_BACKGROUND: u16 = 0x0001;
/// `FT_PALETTE_USABLE_WITH_LIGHT_BACKGROUND`: the palette may also be
/// used on a light background (v1 `paletteTypes` bit 1).
pub const PALETTE_USABLE_WITH_LIGHT_BACKGROUND: u16 = 0x0002;

/// One `ColorRecord`: a color stored as blue/green/red/alpha bytes in
/// file order, exactly as `FT_Color` keeps them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Color {
    /// The blue component.
    pub blue: u8,
    /// The green component.
    pub green: u8,
    /// The red component.
    pub red: u8,
    /// The alpha component (255 = opaque).
    pub alpha: u8,
}

impl Color {
    /// A fully opaque color.
    #[inline]
    pub const fn opaque(red: u8, green: u8, blue: u8) -> Self {
        Color {
            blue,
            green,
            red,
            alpha: 0xFF,
        }
    }

    /// The color as an `[red, green, blue, alpha]` tuple.
    #[inline]
    pub const fn rgba(self) -> [u8; 4] {
        [self.red, self.green, self.blue, self.alpha]
    }
}

/// A parsed `CPAL` table, borrowing the font bytes.
///
/// The view is validated at [`CpalTable::parse`] time; every accessor
/// afterwards is infallible for in-range indices and bounds-checked
/// for out-of-range ones.
#[derive(Clone, Debug)]
pub struct CpalTable<'a> {
    /// The `CPAL` version (0 or 1).
    version: u16,
    /// The number of colors in every palette (`numPaletteEntries`).
    num_palette_entries: u16,
    /// The number of palettes (`numPalettes`).
    num_palettes: u16,
    /// The total number of color records (`numColorRecords`).
    num_color_records: u16,
    /// The `colorRecordIndices` array (`2 * num_palettes` bytes).
    color_indices: &'a [u8],
    /// The combined color-record array (`4 * num_color_records` bytes).
    colors: &'a [u8],
    /// v1 `paletteTypes`, one `u16` per palette; `None` without it.
    palette_types: Option<&'a [u8]>,
    /// v1 `paletteLabels`, one `u16` per palette; `None` without it.
    palette_labels: Option<&'a [u8]>,
    /// v1 `paletteEntryLabels`, one `u16` per entry; `None` without it.
    entry_labels: Option<&'a [u8]>,
}

impl<'a> CpalTable<'a> {
    /// Parses a `CPAL` table, mirroring `tt_face_load_cpal`'s
    /// validation step for step.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the table is shorter than the
    ///   v0 header, the version is above 1, an array offset is out of
    ///   range, a count overruns the table, or a v1 array is missing
    ///   or truncated.
    pub fn parse(data: &'a [u8]) -> TtResult<Self> {
        let mut reader = Reader::new(data);
        let version = reader.u16()?;
        if version > 1 {
            return Err(TtError::INVALID_TABLE);
        }
        let num_palette_entries = reader.u16()?;
        let num_palettes = reader.u16()?;
        let num_color_records = reader.u16()?;
        let colors_offset = reader.u32()? as usize;

        let indices_len = (num_palettes as usize)
            .checked_mul(2)
            .ok_or(TtError::INVALID_TABLE)?;
        if CPAL_V0_HEADER_BASE_SIZE
            .checked_add(indices_len)
            .ok_or(TtError::INVALID_TABLE)?
            > data.len()
        {
            return Err(TtError::INVALID_TABLE);
        }
        if colors_offset >= data.len() {
            return Err(TtError::INVALID_TABLE);
        }
        let color_bytes = (num_color_records as usize)
            .checked_mul(COLOR_SIZE)
            .ok_or(TtError::INVALID_TABLE)?;
        if color_bytes > data.len() - colors_offset {
            return Err(TtError::INVALID_TABLE);
        }
        if num_palette_entries > num_color_records {
            return Err(TtError::INVALID_TABLE);
        }

        let color_indices = data
            .get(CPAL_V0_HEADER_BASE_SIZE..CPAL_V0_HEADER_BASE_SIZE + indices_len)
            .ok_or(TtError::INVALID_TABLE)?;
        let colors = data
            .get(colors_offset..colors_offset + color_bytes)
            .ok_or(TtError::INVALID_TABLE)?;

        let mut palette_types = None;
        let mut palette_labels = None;
        let mut entry_labels = None;
        if version == 1 {
            let v1_header = CPAL_V0_HEADER_BASE_SIZE
                .checked_add(indices_len)
                .and_then(|offset| offset.checked_add(12))
                .ok_or(TtError::INVALID_TABLE)?;
            if v1_header > data.len() {
                return Err(TtError::INVALID_TABLE);
            }
            let mut reader =
                Reader::at(data, CPAL_V0_HEADER_BASE_SIZE + indices_len).ok_or(TtError::INVALID_TABLE)?;
            let type_offset = reader.u32()? as usize;
            let label_offset = reader.u32()? as usize;
            let entry_label_offset = reader.u32()? as usize;

            palette_types = Some(Self::v1_array(data, type_offset, indices_len)?);
            palette_labels = Some(Self::v1_array(data, label_offset, indices_len)?);
            let entry_len = (num_palette_entries as usize)
                .checked_mul(2)
                .ok_or(TtError::INVALID_TABLE)?;
            entry_labels = Some(Self::v1_array(data, entry_label_offset, entry_len)?);
        }

        Ok(CpalTable {
            version,
            num_palette_entries,
            num_palettes,
            num_color_records,
            color_indices,
            colors,
            palette_types,
            palette_labels,
            entry_labels,
        })
    }

    /// Validates one optional v1 array: a zero offset means "absent",
    /// anything else must describe `length` bytes inside the table.
    fn v1_array(data: &'a [u8], offset: usize, length: usize) -> TtResult<&'a [u8]> {
        if offset == 0 {
            return Ok(&[]);
        }
        if offset >= data.len() || length > data.len() - offset {
            return Err(TtError::INVALID_TABLE);
        }
        data.get(offset..offset + length)
            .ok_or(TtError::INVALID_TABLE)
    }

    /// The `CPAL` version (0 or 1).
    #[inline]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// The number of colors in every palette (`numPaletteEntries`).
    #[inline]
    pub const fn num_palette_entries(&self) -> u16 {
        self.num_palette_entries
    }

    /// The number of palettes (`numPalettes`).
    #[inline]
    pub const fn num_palettes(&self) -> u16 {
        self.num_palettes
    }

    /// The total number of color records (`numColorRecords`).
    #[inline]
    pub const fn num_color_records(&self) -> u16 {
        self.num_color_records
    }

    /// The combined index of palette `index`'s first color record.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_ARGUMENT`] — `index` names no palette.
    /// * [`TtError::INVALID_TABLE`] — the palette runs past the end
    ///   of the color-record array.
    pub fn palette_start(&self, index: u16) -> TtResult<usize> {
        if index >= self.num_palettes {
            return Err(TtError::INVALID_ARGUMENT);
        }
        let base = (index as usize) * 2;
        let bytes = self
            .color_indices
            .get(base..base + 2)
            .ok_or(TtError::INVALID_TABLE)?;
        let start = usize::from(u16::from_be_bytes(bytes.try_into().unwrap_or([0, 0])));
        let end = start
            .checked_add(self.num_palette_entries as usize)
            .ok_or(TtError::INVALID_TABLE)?;
        if end > self.num_color_records as usize {
            return Err(TtError::INVALID_TABLE);
        }
        Ok(start)
    }

    /// An iterator over the colors of palette `index`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_ARGUMENT`] — `index` names no palette.
    /// * [`TtError::INVALID_TABLE`] — the palette runs past the end
    ///   of the color-record array.
    pub fn palette(&self, index: u16) -> TtResult<PaletteColors<'a>> {
        let start = self.palette_start(index)?;
        Ok(PaletteColors {
            colors: self.colors,
            next: start,
            remaining: self.num_palette_entries,
        })
    }

    /// The single color at `entry` of palette `index`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_ARGUMENT`] — the palette or entry index
    ///   is out of range.
    /// * [`TtError::INVALID_TABLE`] — the color record is out of
    ///   range.
    pub fn color(&self, index: u16, entry: u16) -> TtResult<Color> {
        if entry >= self.num_palette_entries {
            return Err(TtError::INVALID_ARGUMENT);
        }
        let record = self
            .palette_start(index)?
            .checked_add(entry as usize)
            .ok_or(TtError::INVALID_TABLE)?;
        self.color_record(record)
    }

    /// The raw color record `record` of the combined array.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the record index is out of
    ///   range.
    fn color_record(&self, record: usize) -> TtResult<Color> {
        let offset = record
            .checked_mul(COLOR_SIZE)
            .ok_or(TtError::INVALID_TABLE)?;
        let bytes = self
            .colors
            .get(offset..offset + COLOR_SIZE)
            .ok_or(TtError::INVALID_TABLE)?;
        Ok(Color {
            blue: bytes[0],
            green: bytes[1],
            red: bytes[2],
            alpha: bytes[3],
        })
    }

    /// The v1 `paletteTypes` flags of palette `index`, or `None` when
    /// the array is absent (v0 tables) or `index` is out of range.
    pub fn palette_type(&self, index: u16) -> Option<u16> {
        let bytes = self.palette_types?;
        let base = (index as usize).checked_mul(2)?;
        let slice = bytes.get(base..base + 2)?;
        Some(u16::from_be_bytes(slice.try_into().unwrap_or([0, 0])))
    }

    /// The v1 `paletteLabels` name ID of palette `index`, or `None`
    /// when the array is absent or `index` is out of range.
    pub fn palette_label(&self, index: u16) -> Option<u16> {
        let bytes = self.palette_labels?;
        let base = (index as usize).checked_mul(2)?;
        let slice = bytes.get(base..base + 2)?;
        Some(u16::from_be_bytes(slice.try_into().unwrap_or([0, 0])))
    }

    /// The v1 `paletteEntryLabels` name ID of entry `entry`, or
    /// `None` when the array is absent or `entry` is out of range.
    pub fn palette_entry_label(&self, entry: u16) -> Option<u16> {
        let bytes = self.entry_labels?;
        let base = (entry as usize).checked_mul(2)?;
        let slice = bytes.get(base..base + 2)?;
        Some(u16::from_be_bytes(slice.try_into().unwrap_or([0, 0])))
    }

    /// `true` when palette `index` is flagged for dark backgrounds
    /// (`FT_PALETTE_FOR_DARK_BACKGROUND`).
    ///
    /// Tables without v1 palette types report `false`.
    pub fn is_dark_background(&self, index: u16) -> bool {
        self.palette_type(index)
            .is_some_and(|flags| flags & PALETTE_FOR_DARK_BACKGROUND != 0)
    }
}

/// An iterator over the [`Color`] values of one palette.
#[derive(Clone, Debug)]
pub struct PaletteColors<'a> {
    /// The combined color-record array.
    colors: &'a [u8],
    /// The next color-record index to decode.
    next: usize,
    /// How many colors remain.
    remaining: u16,
}

impl Iterator for PaletteColors<'_> {
    type Item = Color;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let offset = self.next.checked_mul(COLOR_SIZE)?;
        self.next += 1;
        let bytes = self.colors.get(offset..offset + COLOR_SIZE)?;
        Some(Color {
            blue: bytes[0],
            green: bytes[1],
            red: bytes[2],
            alpha: bytes[3],
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for PaletteColors<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Builds a v0 `CPAL` with `entries` colors in one palette.
    fn cpal_v0(entries: &[(u8, u8, u8, u8)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0u16.to_be_bytes()); // version
        out.extend_from_slice(&(entries.len() as u16).to_be_bytes());
        out.extend_from_slice(&1u16.to_be_bytes()); // numPalettes
        out.extend_from_slice(&(entries.len() as u16).to_be_bytes());
        out.extend_from_slice(&14u32.to_be_bytes()); // colorRecordsArrayOffset
        out.extend_from_slice(&0u16.to_be_bytes()); // colorRecordIndices[0]
        for &(blue, green, red, alpha) in entries {
            out.extend_from_slice(&[blue, green, red, alpha]);
        }
        out
    }

    #[test]
    fn parses_a_v0_palette_and_serves_colors_in_file_order() {
        let bytes = cpal_v0(&[(10, 20, 30, 255), (1, 2, 3, 128)]);
        let cpal = CpalTable::parse(&bytes).unwrap();

        assert_eq!(cpal.version(), 0);
        assert_eq!(cpal.num_palettes(), 1);
        assert_eq!(cpal.num_palette_entries(), 2);
        assert_eq!(cpal.num_color_records(), 2);

        let colors: Vec<Color> = cpal.palette(0).unwrap().collect();
        assert_eq!(
            colors,
            vec![
                Color {
                    blue: 10,
                    green: 20,
                    red: 30,
                    alpha: 255
                },
                Color {
                    blue: 1,
                    green: 2,
                    red: 3,
                    alpha: 128
                },
            ]
        );
        assert_eq!(cpal.color(0, 1).unwrap().rgba(), [3, 2, 1, 128]);
        assert_eq!(cpal.color(0, 0).unwrap().red, 30);
    }

    #[test]
    fn unknown_palettes_and_entries_are_rejected() {
        let bytes = cpal_v0(&[(0, 0, 0, 255)]);
        let cpal = CpalTable::parse(&bytes).unwrap();

        assert!(matches!(cpal.palette(1), Err(TtError::INVALID_ARGUMENT)));
        assert!(matches!(cpal.color(0, 7), Err(TtError::INVALID_ARGUMENT)));
        assert!(matches!(
            cpal.palette_start(0xFFFF),
            Err(TtError::INVALID_ARGUMENT)
        ));
    }

    #[test]
    fn malformed_tables_return_errors_and_never_panic() {
        // Shorter than the v0 header.
        assert!(matches!(
            CpalTable::parse(&[0, 0, 0, 1]),
            Err(TtError::INVALID_TABLE)
        ));

        // Unsupported version.
        let mut bytes = cpal_v0(&[(0, 0, 0, 255)]);
        bytes[0..2].copy_from_slice(&2u16.to_be_bytes());
        assert!(matches!(CpalTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // Color-record array offset past the end.
        let mut bytes = cpal_v0(&[(0, 0, 0, 255)]);
        bytes[8..12].copy_from_slice(&0xFFFFu32.to_be_bytes());
        assert!(matches!(CpalTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // numPaletteEntries above numColorRecords.
        let mut bytes = cpal_v0(&[(0, 0, 0, 255)]);
        bytes[2..4].copy_from_slice(&9u16.to_be_bytes());
        assert!(matches!(CpalTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // Truncated color records.
        let mut bytes = cpal_v0(&[(0, 0, 0, 255), (0, 0, 0, 255)]);
        bytes.truncate(bytes.len() - 3);
        assert!(matches!(CpalTable::parse(&bytes), Err(TtError::INVALID_TABLE)));
    }

    #[test]
    fn v1_palette_types_and_labels_are_exposed() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes()); // version
        bytes.extend_from_slice(&2u16.to_be_bytes()); // entries
        bytes.extend_from_slice(&2u16.to_be_bytes()); // palettes
        bytes.extend_from_slice(&4u16.to_be_bytes()); // colors
        bytes.extend_from_slice(&40u32.to_be_bytes()); // color records at 40
        bytes.extend_from_slice(&0u16.to_be_bytes()); // palette 0 start
        bytes.extend_from_slice(&2u16.to_be_bytes()); // palette 1 start
        // v1 offsets, immediately after the colorRecordIndices array:
        // types at 28, labels at 32, entry labels at 36
        bytes.extend_from_slice(&28u32.to_be_bytes());
        bytes.extend_from_slice(&32u32.to_be_bytes());
        bytes.extend_from_slice(&36u32.to_be_bytes());
        // paletteTypes: dark background, plain
        bytes.extend_from_slice(&PALETTE_FOR_DARK_BACKGROUND.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        // paletteLabels
        bytes.extend_from_slice(&10u16.to_be_bytes());
        bytes.extend_from_slice(&11u16.to_be_bytes());
        // paletteEntryLabels
        bytes.extend_from_slice(&20u16.to_be_bytes());
        bytes.extend_from_slice(&21u16.to_be_bytes());
        // color records: black/white and red/blue
        bytes.extend_from_slice(&[0, 0, 0, 255, 255, 255, 255, 255]);
        bytes.extend_from_slice(&[0, 0, 255, 255, 255, 0, 0, 255]);

        let cpal = CpalTable::parse(&bytes).unwrap();
        assert_eq!(cpal.version(), 1);
        assert_eq!(cpal.palette_type(0), Some(PALETTE_FOR_DARK_BACKGROUND));
        assert!(cpal.is_dark_background(0));
        assert!(!cpal.is_dark_background(1));
        assert_eq!(cpal.palette_label(1), Some(11));
        assert_eq!(cpal.palette_entry_label(1), Some(21));
        assert_eq!(cpal.color(1, 1).unwrap().blue, 0xFF);
    }

    #[test]
    fn v1_with_absent_arrays_parses_as_empty() {
        // The v1 header follows the color record indices (offset 14);
        // the color records live at their own offset.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&26u32.to_be_bytes()); // colors @26
        bytes.extend_from_slice(&0u16.to_be_bytes());
        // v1 header: three absent arrays.
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(&[9, 9, 9, 255]);

        let cpal = CpalTable::parse(&bytes).unwrap();
        assert_eq!(cpal.palette_type(0), None);
        assert_eq!(cpal.palette_label(0), None);
        assert_eq!(cpal.palette_entry_label(0), None);
        assert!(!cpal.is_dark_background(0));
    }
}
