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

//! The SFNT container: TTC header, table directory and checksums.
//!
//! Port of `ttload.c` (`check_table_dir`, `tt_face_load_font_dir`,
//! `tt_face_lookup_table`, `tt_face_goto_table`) and of `sfobjs.c`
//! (`sfnt_open_font`) from FreeType 2.6.  The container is parsed
//! straight out of the input slice: [`SfntDirectory::parse`] validates
//! every record against the slice bounds before it is stored, so later
//! [`SfntDirectory::table`] lookups are infallible slice selections.

use crate::tags::{TAG_HEAD, TAG_HMTX, TAG_TTCF, TAG_VMTX, is_sfnt_version};
use alloc::vec::Vec;
use bytes::Buf;
use codevar_truetype_core::{Tag, TtError, TtResult};

/// One 16-byte record of the table directory (`TT_TableRec`, `ttload.c`).
///
/// `offset` and `length` are byte offsets into the *whole* input slice
/// (collection members are addressed absolutely, exactly like FreeType's
/// `TT_TableRec.Offset`).  Both fields have been validated during
/// [`SfntDirectory::parse`]: `offset <= data.len()` and
/// `length <= data.len() - offset`, except that an overflowing `hmtx`/
/// `vmtx` record is sanitized to the bytes that remain (see
/// `tt_face_load_font_dir`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableRecord {
    /// The four-character table tag (`Tag`).
    pub tag: Tag,
    /// The table checksum as stored in the directory.
    pub checksum: u32,
    /// Absolute offset of the table inside the file.
    pub offset: u32,
    /// Length of the table in bytes (sanitized, see type docs).
    pub length: u32,
}

/// The header of a TrueType collection (`TTC_HeaderRec`, `sfobjs.c`).
///
/// Plain (non-collection) files are synthesized into a one-member TTC
/// exactly like `sfnt_open_font` does: [`Self::offsets`] holds the single
/// offset `0` and [`Self::is_collection`] is `false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtcHeader {
    /// The TTC version (`version` field; synthesized to `1 << 16` for
    /// plain files).
    pub version: u32,
    /// The offset of every sub-font's offset table.
    pub offsets: Vec<u32>,
    /// `true` when the file starts with the `'ttcf'` tag.
    pub collection: bool,
}

impl TtcHeader {
    /// The number of fonts in the collection (`ttc_header.count`).
    #[inline]
    pub fn num_faces(&self) -> usize {
        self.offsets.len()
    }

    /// The offset of sub-font `index`, or `None` when out of range.
    #[inline]
    pub fn offset(&self, index: usize) -> Option<u32> {
        self.offsets.get(index).copied()
    }

    /// `true` for a real `'ttcf'` collection, `false` for a synthesized
    /// single-font header.
    #[inline]
    pub fn is_collection(&self) -> bool {
        self.collection
    }

    /// `true` when the header names no fonts (only possible for a
    /// malformed collection, which [`parse_container`] rejects).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// The parsed offset table (directory) of a single SFNT font.
///
/// Port of `tt_face_load_font_dir` + `check_table_dir` (`ttload.c`):
/// the header is read at `base`, then every 16-byte record is validated
/// and the invalid ones are dropped the way FreeType drops them.
#[derive(Debug, Clone)]
pub struct SfntDirectory<'a> {
    data: &'a [u8],
    sfnt_version: u32,
    search_range: u16,
    entry_selector: u16,
    range_shift: u16,
    records: Vec<TableRecord>,
}

impl<'a> SfntDirectory<'a> {
    /// Parses the offset table at absolute offset `base` inside `data`.
    ///
    /// Mirrors `sfnt_open_font`/`tt_face_load_font_dir`:
    ///
    /// * fewer than 12 bytes from `base` → [`TtError::INVALID_TABLE`]
    ///   (truncated offset table; FreeType fails on the stream read);
    /// * an unrecognized version tag (including nested `'ttcf'` and
    ///   `'wOFF'`) → [`TtError::UNKNOWN_FILE_FORMAT`];
    /// * `num_tables == 0` → [`TtError::UNKNOWN_FILE_FORMAT`]
    ///   (`check_table_dir`: "no tables found");
    /// * a directory that does not fit into `data` →
    ///   [`TtError::INVALID_TABLE`] (deviation: FreeType clamps
    ///   `num_tables` to the readable entries);
    /// * records whose `offset`/`length` leave the file are dropped;
    ///   an overflowing `hmtx`/`vmtx` record is kept with a sanitized
    ///   (4-byte aligned) length, exactly like `tt_face_load_font_dir`;
    /// * when every record is dropped → [`TtError::UNKNOWN_FILE_FORMAT`].
    pub fn parse(data: &'a [u8], base: usize) -> TtResult<Self> {
        let mut header: &[u8] = data
            .get(base..)
            .and_then(|rest| rest.get(..12))
            .ok_or(TtError::INVALID_TABLE)?;
        let invalid = |_| TtError::INVALID_TABLE;
        let sfnt_version = header.try_get_u32().map_err(invalid)?;
        if !is_sfnt_version(sfnt_version) {
            return Err(TtError::UNKNOWN_FILE_FORMAT);
        }
        let num_tables = header.try_get_u16().map_err(invalid)?;
        let search_range = header.try_get_u16().map_err(invalid)?;
        let entry_selector = header.try_get_u16().map_err(invalid)?;
        let range_shift = header.try_get_u16().map_err(invalid)?;
        if num_tables == 0 {
            return Err(TtError::UNKNOWN_FILE_FORMAT);
        }
        let entries_start = base
            .checked_add(12)
            .ok_or(TtError::INVALID_TABLE)?;
        let entries_len = (num_tables as usize)
            .checked_mul(16)
            .ok_or(TtError::INVALID_TABLE)?;
        let entries: &[u8] = data
            .get(entries_start..)
            .and_then(|rest| rest.get(..entries_len))
            .ok_or(TtError::INVALID_TABLE)?;

        let size = data.len();
        let mut records = Vec::new();
        for index in 0..num_tables as usize {
            let pos = index * 16;
            let mut record_buf = entries
                .get(pos..pos + 16)
                .ok_or(TtError::INVALID_TABLE)?;
            let mut record = TableRecord {
                tag: record_buf.try_get_u32().map_err(invalid)?,
                checksum: record_buf.try_get_u32().map_err(invalid)?,
                offset: record_buf.try_get_u32().map_err(invalid)?,
                length: record_buf.try_get_u32().map_err(invalid)?,
            };
            let offset = record.offset as usize;
            if offset > size {
                continue;
            }
            let remaining = size - offset;
            if record.length as usize > remaining {
                if record.tag == TAG_HMTX || record.tag == TAG_VMTX {
                    record.length = (remaining & !3) as u32;
                    records.push(record);
                }
                continue;
            }
            records.push(record);
        }
        if records.is_empty() {
            return Err(TtError::UNKNOWN_FILE_FORMAT);
        }
        Ok(SfntDirectory {
            data,
            sfnt_version,
            search_range,
            entry_selector,
            range_shift,
            records,
        })
    }

    /// The raw input slice the directory was parsed from.
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The `sfntVersion` of the offset table (`format_tag`).
    #[inline]
    pub fn sfnt_version(&self) -> u32 {
        self.sfnt_version
    }

    /// The number of *valid* records kept after sanitization.
    #[inline]
    pub fn num_tables(&self) -> usize {
        self.records.len()
    }

    /// The declared `numTables` field is not retained separately; every
    /// record of the sanitized directory, in file order.
    #[inline]
    pub fn records(&self) -> &[TableRecord] {
        &self.records
    }

    /// The `searchRange` field (many fonts get it wrong; FreeType keeps
    /// it without validating — see the disabled check in
    /// `tt_face_load_font_dir`).
    #[inline]
    pub fn search_range(&self) -> u16 {
        self.search_range
    }

    /// The `entrySelector` field (see [`Self::search_range`]).
    #[inline]
    pub fn entry_selector(&self) -> u16 {
        self.entry_selector
    }

    /// The `rangeShift` field (see [`Self::search_range`]).
    #[inline]
    pub fn range_shift(&self) -> u16 {
        self.range_shift
    }

    /// `tt_face_lookup_table` (`ttload.c`): the first record with `tag`,
    /// treating zero-length tables as missing (Windows compatibility).
    ///
    /// Returns `None` when the table is absent or has zero length.
    #[inline]
    pub fn find(&self, tag: Tag) -> Option<&TableRecord> {
        self.records
            .iter()
            .find(|record| record.tag == tag && record.length != 0)
    }

    /// The bytes of the table named `tag`, or `None` when the table is
    /// missing.
    ///
    /// The slice is always inside `data` because records are validated
    /// during [`Self::parse`]; the extra bounds check is a defence in
    /// depth that also covers the sanitized `hmtx`/`vmtx` records.
    #[inline]
    pub fn table(&self, tag: Tag) -> Option<&'a [u8]> {
        let record = self.find(tag)?;
        let offset = record.offset as usize;
        let length = record.length as usize;
        self.data.get(offset..)?.get(..length)
    }

    /// The offset and length of the table named `tag`, without slicing.
    #[inline]
    pub fn table_range(&self, tag: Tag) -> Option<(usize, usize)> {
        let record = self.find(tag)?;
        Some((record.offset as usize, record.length as usize))
    }

    /// The OpenType checksum of `bytes`: the wrapping sum of the
    /// big-endian `u32` words, padding the tail with zero bytes to a
    /// 4-byte boundary.
    pub fn checksum(bytes: &[u8]) -> u32 {
        let mut buf = bytes;
        let mut sum = 0u32;
        while buf.remaining() >= 4 {
            sum = sum.wrapping_add(buf.try_get_u32().unwrap_or(0));
        }
        if buf.has_remaining() {
            let mut tail = [0u8; 4];
            let count = buf.remaining().min(4);
            tail[..count].copy_from_slice(&buf[..count]);
            sum = sum.wrapping_add(u32::from_be_bytes(tail));
        }
        sum
    }

    /// Verifies the stored checksum of the table named `tag`.
    ///
    /// Returns `None` when the table is missing and `Some(true)` when it
    /// matches.  For `'head'` the `checkSumAdjustment` field (bytes
    /// 8..12) is treated as zero, as required by the OpenType
    /// specification: the directory checksum is written before the
    /// adjustment is known.
    pub fn verify_checksum(&self, tag: Tag) -> Option<bool> {
        let record = self.find(tag)?;
        let bytes = self.table(tag)?;
        Some(Self::stored_checksum(tag, bytes) == record.checksum)
    }

    /// [`Self::checksum`] with the `'head'` `checkSumAdjustment`
    /// zeroed (see [`Self::verify_checksum`]).
    fn stored_checksum(tag: Tag, bytes: &[u8]) -> u32 {
        if tag == TAG_HEAD && bytes.len() >= 12 {
            let mut head = bytes.to_vec();
            head[8..12].copy_from_slice(&0u32.to_be_bytes());
            Self::checksum(&head)
        } else {
            Self::checksum(bytes)
        }
    }

    /// Verifies every present table checksum and returns the tags whose
    /// stored checksum does not match, in directory order.
    ///
    /// Optional integrity check: a font with a non-empty list is still
    /// usable, but the bytes were modified after the directory was
    /// written (or the producer wrote wrong checksums).
    pub fn verify_checksums(&self) -> Vec<Tag> {
        let mut bad = Vec::new();
        for record in &self.records {
            if record.length == 0 {
                continue;
            }
            let Some(bytes) = self.table(record.tag) else {
                continue;
            };
            let actual = Self::stored_checksum(record.tag, bytes);
            if actual != record.checksum {
                bad.push(record.tag);
            }
        }
        bad
    }
}

/// The container-level view of an SFNT file: TTC header plus the table
/// directory of one selected sub-font.
///
/// Port of `sfnt_open_font` + `sfnt_init_face` (`sfobjs.c`).  Use this
/// type for pure container questions (how many fonts, which tables);
/// [`crate::SfntFont`] layers the table parsers on top.
#[derive(Debug, Clone)]
pub struct SfntContainer<'a> {
    data: &'a [u8],
    ttc: TtcHeader,
    directory: SfntDirectory<'a>,
    face_index: i64,
}

impl<'a> SfntContainer<'a> {
    /// Opens sub-font `face_index` of the SFNT file in `data`.
    ///
    /// `face_index` is clamped to `0` when negative (as `sfnt_init_face`
    /// does).  Errors:
    ///
    /// * [`TtError::INVALID_FILE_FORMAT`] — fewer than 4 bytes (the
    ///   version tag does not fit);
    /// * [`TtError::UNKNOWN_FILE_FORMAT`] — the version tag is not an
    ///   SFNT container tag (see [`crate::tags::is_sfnt_version`]);
    /// * [`TtError::INVALID_TABLE`] — truncated `'ttcf'` header, or a
    ///   truncated offset table of the selected font;
    /// * [`TtError::ARRAY_TOO_LARGE`] — a `'ttcf'` font count that
    ///   cannot fit into the file (`sfnt_open_font`'s
    ///   `count > size / (28 + 4)` guard);
    /// * [`TtError::INVALID_OFFSET`] — the selected sub-font offset
    ///   leaves the file;
    /// * [`TtError::INVALID_ARGUMENT`] — `face_index` beyond the number
    ///   of fonts (`sfnt_init_face`);
    /// * whatever [`SfntDirectory::parse`] reports for the directory.
    pub fn open(data: &'a [u8], face_index: i32) -> TtResult<Self> {
        let mut version_buf: &[u8] = data;
        let version = version_buf
            .try_get_u32()
            .map_err(|_| TtError::INVALID_FILE_FORMAT)?;
        let ttc = if version == TAG_TTCF {
            parse_ttc_header(data)?
        } else {
            if !is_sfnt_version(version) {
                return Err(TtError::UNKNOWN_FILE_FORMAT);
            }
            TtcHeader {
                version: crate::tags::SFNT_TTC_VERSION_SYNTHESIZED,
                offsets: alloc::vec![0],
                collection: false,
            }
        };
        let count = ttc.num_faces();
        let index = if face_index < 0 { 0 } else { face_index as usize };
        if index >= count {
            return Err(TtError::INVALID_ARGUMENT);
        }
        let base = ttc
            .offset(index)
            .ok_or(TtError::INVALID_ARGUMENT)? as usize;
        if base >= data.len() {
            return Err(TtError::INVALID_OFFSET);
        }
        let directory = SfntDirectory::parse(data, base)?;
        Ok(SfntContainer {
            data,
            ttc,
            directory,
            face_index: index as i64,
        })
    }

    /// The raw file bytes.
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The collection header (synthesized for plain fonts).
    #[inline]
    pub fn ttc_header(&self) -> &TtcHeader {
        &self.ttc
    }

    /// The number of fonts in the file (`FT_FaceRec.num_faces`).
    #[inline]
    pub fn num_faces(&self) -> usize {
        self.ttc.num_faces()
    }

    /// The index of the open sub-font (`FT_FaceRec.face_index`).
    #[inline]
    pub fn face_index(&self) -> i64 {
        self.face_index
    }

    /// The directory of the selected sub-font.
    #[inline]
    pub fn directory(&self) -> &SfntDirectory<'a> {
        &self.directory
    }

    /// [`SfntDirectory::sfnt_version`] of the selected sub-font.
    #[inline]
    pub fn sfnt_version(&self) -> u32 {
        self.directory.sfnt_version()
    }

    /// [`SfntDirectory::num_tables`] of the selected sub-font.
    #[inline]
    pub fn num_tables(&self) -> usize {
        self.directory.num_tables()
    }

    /// [`SfntDirectory::records`] of the selected sub-font.
    #[inline]
    pub fn records(&self) -> &[TableRecord] {
        self.directory.records()
    }

    /// [`SfntDirectory::table`] for the selected sub-font.
    #[inline]
    pub fn table(&self, tag: Tag) -> Option<&'a [u8]> {
        self.directory.table(tag)
    }

    /// [`SfntDirectory::verify_checksum`] for the selected sub-font.
    #[inline]
    pub fn verify_checksum(&self, tag: Tag) -> Option<bool> {
        self.directory.verify_checksum(tag)
    }

    /// [`SfntDirectory::verify_checksums`] for the selected sub-font.
    #[inline]
    pub fn verify_checksums(&self) -> Vec<Tag> {
        self.directory.verify_checksums()
    }
}

/// `sfnt_open_font` (`sfobjs.c`): parses the `'ttcf'` collection header
/// at the start of `data`.
///
/// The `version` field is read as unsigned; FreeType stores it as a
/// signed `FT_Long`, which only affects trace output.
fn parse_ttc_header(data: &[u8]) -> TtResult<TtcHeader> {
    let mut header: &[u8] = data.get(..12).ok_or(TtError::INVALID_TABLE)?;
    let invalid = |_| TtError::INVALID_TABLE;
    let _tag = header.try_get_u32().map_err(invalid)?;
    let version = header.try_get_u32().map_err(invalid)?;
    let count = header.try_get_u32().map_err(invalid)?;
    if count == 0 {
        return Err(TtError::INVALID_TABLE);
    }
    if count as usize > data.len() / (28 + 4) {
        return Err(TtError::ARRAY_TOO_LARGE);
    }
    let offsets_bytes = data
        .get(12..)
        .and_then(|rest| rest.get(..count as usize * 4))
        .ok_or(TtError::INVALID_TABLE)?;
    let mut offsets_buf: &[u8] = offsets_bytes;
    let mut offsets = Vec::with_capacity(count as usize);
    for _ in 0..count {
        offsets.push(offsets_buf.try_get_u32().map_err(invalid)?);
    }
    Ok(TtcHeader {
        version,
        offsets,
        collection: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::{TAG_KERN, TAG_MAXP, TAG_WOFF};
    use crate::test_util::{build_font, build_ttc, default_font, default_tables, push_u16, push_u32};
    use codevar_truetype_core::make_tag;

    fn corrupt_all_offsets(font: &mut [u8], base: usize) {
        let num_tables = u16::from_be_bytes([font[base + 4], font[base + 5]]);
        for index in 0..num_tables as usize {
            let record = base + 12 + index * 16 + 8;
            font[record..record + 4].copy_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
        }
    }

    #[test]
    fn open_plain_font_synthesizes_one_member_ttc() {
        let font = default_font();
        let container = SfntContainer::open(&font, 0).unwrap();
        assert_eq!(container.num_faces(), 1);
        assert_eq!(container.face_index(), 0);
        assert_eq!(container.data().len(), font.len());
        let ttc = container.ttc_header();
        assert!(!ttc.is_collection());
        assert!(!ttc.is_empty());
        assert_eq!(ttc.version, crate::tags::SFNT_TTC_VERSION_SYNTHESIZED);
        assert_eq!(ttc.offset(0), Some(0));
        assert_eq!(ttc.offset(1), None);
        assert_eq!(container.sfnt_version(), crate::tags::SFNT_VERSION_1_0);
        assert_eq!(container.num_tables(), 8);
        assert_eq!(container.table(TAG_HEAD).map(|head| head.len()), Some(54));
        assert!(container.verify_checksums().is_empty());
    }

    #[test]
    fn open_clamps_negative_face_index() {
        let font = default_font();
        let container = SfntContainer::open(&font, -7).unwrap();
        assert_eq!(container.face_index(), 0);
    }

    #[test]
    fn open_rejects_truncated_and_unknown_versions() {
        assert!(matches!(
            SfntContainer::open(&[], 0),
            Err(TtError::INVALID_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntContainer::open(&[0x00, 0x01], 0),
            Err(TtError::INVALID_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntContainer::open(b"XXXX", 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntContainer::open(b"wOFF", 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntContainer::open(&[0x00, 0x03, 0x00, 0x00], 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
    }

    #[test]
    fn open_rejects_truncated_directory() {
        let font = default_font();
        assert!(matches!(
            SfntContainer::open(&font[..11], 0),
            Err(TtError::INVALID_TABLE)
        ));
        assert!(matches!(
            SfntContainer::open(&font[..12], 0),
            Err(TtError::INVALID_TABLE)
        ));
        let records_end = 12 + 8 * 16;
        assert!(matches!(
            SfntContainer::open(&font[..records_end - 1], 0),
            Err(TtError::INVALID_TABLE)
        ));
        let with_first_table = records_end + 80;
        assert!(SfntContainer::open(&font[..with_first_table], 0).is_ok());
        assert!(matches!(
            SfntContainer::open(&font[..records_end], 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntContainer::open(&font[..4], 0),
            Err(TtError::INVALID_TABLE)
        ));
    }

    #[test]
    fn open_rejects_zero_table_fonts() {
        let mut header = Vec::new();
        push_u32(&mut header, 0x0001_0000);
        push_u16(&mut header, 0);
        push_u16(&mut header, 0);
        push_u16(&mut header, 0);
        push_u16(&mut header, 0);
        assert!(matches!(
            SfntContainer::open(&header, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntDirectory::parse(&header, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
    }

    #[test]
    fn open_drops_records_outside_the_file() {
        let mut font = default_font();
        let num_tables = u16::from_be_bytes([font[4], font[5]]) as usize;
        let last = 12 + (num_tables - 1) * 16;
        let record_offset =
            u32::from_be_bytes([font[last + 8], font[last + 9], font[last + 10], font[last + 11]]) as usize;
        let record_length =
            u32::from_be_bytes([font[last + 12], font[last + 13], font[last + 14], font[last + 15]]);
        let file_len = font.len() as u32;
        font[last + 8..last + 12].copy_from_slice(&file_len.to_be_bytes());
        font[last + 12..last + 16].copy_from_slice(&record_length.to_be_bytes());
        let container = SfntContainer::open(&font, 0).unwrap();
        assert_eq!(container.num_tables(), num_tables - 1);
        let tag = u32::from_be_bytes([font[last], font[last + 1], font[last + 2], font[last + 3]]);
        assert!(container.directory().find(tag).is_none());
        assert!(record_offset < font.len());
    }

    #[test]
    fn open_rejects_when_every_record_is_outside_the_file() {
        let mut font = default_font();
        corrupt_all_offsets(&mut font, 0);
        assert!(matches!(
            SfntContainer::open(&font, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
    }

    #[test]
    fn overflowing_hmtx_record_is_sanitized_not_dropped() {
        let mut font = default_font();
        let num_tables = u16::from_be_bytes([font[4], font[5]]) as usize;
        let mut hmtx_record = 0usize;
        for index in 0..num_tables {
            let record = 12 + index * 16;
            if &font[record..record + 4] == b"hmtx" {
                hmtx_record = record;
            }
        }
        assert!(hmtx_record != 0);
        let length = u32::from_be_bytes([
            font[hmtx_record + 12],
            font[hmtx_record + 13],
            font[hmtx_record + 14],
            font[hmtx_record + 15],
        ]);
        font[hmtx_record + 12..hmtx_record + 16].copy_from_slice(&0xFFFFu32.to_be_bytes());
        let container = SfntContainer::open(&font, 0).unwrap();
        assert_eq!(container.num_tables(), num_tables);
        let (_, sanitized) = container
            .directory()
            .table_range(TAG_HMTX)
            .unwrap();
        let offset = container
            .directory()
            .find(TAG_HMTX)
            .unwrap()
            .offset as usize;
        assert_eq!(sanitized, (font.len() - offset) & !3);
        assert!(sanitized < length as usize + usize::MAX / 2);
        assert!(container.directory().table(TAG_HMTX).is_some());
    }

    #[test]
    fn overflowing_non_hmtx_record_is_dropped() {
        let mut font = default_font();
        let num_tables = u16::from_be_bytes([font[4], font[5]]) as usize;
        let mut head_record = 0usize;
        for index in 0..num_tables {
            let record = 12 + index * 16;
            if &font[record..record + 4] == b"head" {
                head_record = record;
            }
        }
        assert!(head_record != 0);
        font[head_record + 12..head_record + 16].copy_from_slice(&0xFFFFu32.to_be_bytes());
        let container = SfntContainer::open(&font, 0).unwrap();
        assert_eq!(container.num_tables(), num_tables - 1);
        assert!(container.table(TAG_HEAD).is_none());
        assert!(container.table(TAG_MAXP).is_some());
    }

    #[test]
    fn directory_find_treats_zero_length_as_missing() {
        let font = default_font();
        let container = SfntContainer::open(&font, 0).unwrap();
        let directory = container.directory();
        assert!(directory.find(TAG_KERN).is_some());
        assert!(
            directory
                .find(make_tag(b'n', b'o', b'p', b'e'))
                .is_none()
        );
        let zero = TableRecord {
            tag: TAG_HEAD,
            checksum: 0,
            offset: 0,
            length: 0,
        };
        assert!(
            directory
                .records()
                .iter()
                .any(|record| record.tag == zero.tag)
        );
        assert!(directory.find(TAG_HEAD).unwrap().length != 0);
        assert_eq!(
            directory
                .table_range(TAG_HEAD)
                .map(|(_, len)| len),
            Some(54)
        );
    }

    #[test]
    fn open_ttc_reads_every_offset_table() {
        let first = default_font();
        let tables: Vec<(Tag, Vec<u8>)> = default_tables()
            .into_iter()
            .map(|(tag, body)| {
                let mut body = body;
                if tag == crate::tags::TAG_MAXP {
                    body[4..6].copy_from_slice(&9u16.to_be_bytes());
                }
                (tag, body)
            })
            .collect();
        let second = build_font(&tables);
        let ttc_data = build_ttc(&[first.clone(), second.clone()]);
        let container = SfntContainer::open(&ttc_data, 1).unwrap();
        assert_eq!(container.num_faces(), 2);
        assert_eq!(container.face_index(), 1);
        assert!(container.ttc_header().is_collection());
        assert_eq!(container.ttc_header().version, 0x0001_0000);
        assert_eq!(
            container
                .table(crate::tags::TAG_MAXP)
                .map(|maxp| u16::from_be_bytes([maxp[4], maxp[5]])),
            Some(9)
        );
        assert!(matches!(
            SfntContainer::open(&ttc_data, 2),
            Err(TtError::INVALID_ARGUMENT)
        ));
        let clamped = SfntContainer::open(&ttc_data, -3).unwrap();
        assert_eq!(clamped.face_index(), 0);
        let first_container = SfntContainer::open(&ttc_data, 0).unwrap();
        assert_eq!(
            first_container
                .table(crate::tags::TAG_MAXP)
                .map(|maxp| u16::from_be_bytes([maxp[4], maxp[5]])),
            Some(5)
        );
    }

    #[test]
    fn open_ttc_rejects_broken_headers() {
        assert!(matches!(
            SfntContainer::open(b"ttcf\x00\x01\x00", 0),
            Err(TtError::INVALID_TABLE)
        ));
        let mut zero_count = Vec::new();
        zero_count.extend_from_slice(b"ttcf");
        push_u32(&mut zero_count, 0x0001_0000);
        push_u32(&mut zero_count, 0);
        assert!(matches!(
            SfntContainer::open(&zero_count, 0),
            Err(TtError::INVALID_TABLE)
        ));

        let mut huge_count = Vec::new();
        huge_count.extend_from_slice(b"ttcf");
        push_u32(&mut huge_count, 0x0001_0000);
        push_u32(&mut huge_count, 10_000);
        push_u32(&mut huge_count, 0);
        assert!(matches!(
            SfntContainer::open(&huge_count, 0),
            Err(TtError::ARRAY_TOO_LARGE)
        ));
    }

    #[test]
    fn open_ttc_rejects_bad_sub_font_offsets() {
        let font = default_font();
        let mut ttc_data = build_ttc(&[font.clone(), font.clone()]);
        let bad = (ttc_data.len() as u32 + 1).to_be_bytes();
        ttc_data[12..16].copy_from_slice(&bad);
        assert!(matches!(
            SfntContainer::open(&ttc_data, 0),
            Err(TtError::INVALID_OFFSET)
        ));
        assert!(SfntContainer::open(&ttc_data, 1).is_ok());
    }

    #[test]
    fn open_ttc_rejects_truncated_headers() {
        let font = default_font();
        let mut ttc_data = build_ttc(&[font.clone(), font.clone()]);
        ttc_data.truncate(16);
        assert!(matches!(
            SfntContainer::open(&ttc_data, 1),
            Err(TtError::ARRAY_TOO_LARGE)
        ));
        ttc_data.truncate(12);
        assert!(matches!(
            SfntContainer::open(&ttc_data, 0),
            Err(TtError::ARRAY_TOO_LARGE)
        ));
        ttc_data.truncate(8);
        assert!(matches!(
            SfntContainer::open(&ttc_data, 0),
            Err(TtError::INVALID_TABLE)
        ));
    }

    #[test]
    fn checksum_pads_the_tail_with_zero_bytes() {
        assert_eq!(SfntDirectory::checksum(&[]), 0);
        assert_eq!(SfntDirectory::checksum(&[0x01, 0x02, 0x03, 0x04]), 0x0102_0304);
        assert_eq!(SfntDirectory::checksum(&[0x01, 0x02, 0x03]), 0x0102_0300);
        assert_eq!(
            SfntDirectory::checksum(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
            0xFEFF_FFFF
        );
        let mut wrapped = [0u8; 8];
        wrapped.fill(0xFF);
        assert_eq!(SfntDirectory::checksum(&wrapped), 0xFFFF_FFFE);
        assert_eq!(SfntDirectory::checksum(&[0x80, 0x00, 0x00, 0x00]), 0x8000_0000);
        let mut long = [0u8; 12];
        long[0] = 0xFF;
        long[4] = 0xFF;
        long[8] = 0xFF;
        assert_eq!(SfntDirectory::checksum(&long), 0xFD00_0000);
    }

    #[test]
    fn verify_checksums_reports_tampered_tables() {
        let font = default_font();
        let container = SfntContainer::open(&font, 0).unwrap();
        assert!(container.verify_checksums().is_empty());
        assert_eq!(container.directory().verify_checksum(TAG_HEAD), Some(true));
        assert_eq!(
            container
                .directory()
                .verify_checksum(crate::tags::TAG_MAXP),
            Some(true)
        );
        assert_eq!(
            container
                .directory()
                .verify_checksum(make_tag(b'n', b'o', b'p', b'e')),
            None
        );

        let mut tampered = font.clone();
        let offset = container
            .directory()
            .find(TAG_HEAD)
            .unwrap()
            .offset as usize;
        tampered[offset + 20] ^= 0xFF;
        let tampered_container = SfntContainer::open(&tampered, 0).unwrap();
        assert_eq!(tampered_container.verify_checksums(), vec![TAG_HEAD]);
        assert_eq!(
            tampered_container
                .directory()
                .verify_checksum(TAG_HEAD),
            Some(false)
        );
    }

    #[test]
    fn verify_checksum_zeroes_head_adjustment_bytes() {
        let font = default_font();
        let container = SfntContainer::open(&font, 0).unwrap();
        let head = container.directory().table(TAG_HEAD).unwrap();
        assert_ne!(u32::from_be_bytes([head[8], head[9], head[10], head[11]]), 0);
        assert_ne!(
            SfntDirectory::checksum(head),
            SfntDirectory::stored_checksum(TAG_HEAD, head)
        );
        assert_eq!(container.directory().verify_checksum(TAG_HEAD), Some(true));
    }

    #[test]
    fn directory_exposes_header_fields_and_ranges() {
        let font = default_font();
        let container = SfntContainer::open(&font, 0).unwrap();
        let directory = container.directory();
        assert_eq!(directory.data().len(), font.len());
        assert_eq!(directory.num_tables(), directory.records().len());
        assert_eq!(directory.search_range(), 128);
        assert_eq!(directory.entry_selector(), 3);
        assert_eq!(directory.range_shift(), 0);
        let (offset, length) = directory
            .table_range(crate::tags::TAG_MAXP)
            .unwrap();
        assert!(offset + length <= font.len());
        assert_eq!(
            directory
                .table(crate::tags::TAG_MAXP)
                .unwrap()
                .len(),
            length
        );
        assert_eq!(container.records(), directory.records());
        assert_eq!(container.sfnt_version(), directory.sfnt_version());
        assert_eq!(
            container.verify_checksums().len(),
            directory.verify_checksums().len()
        );
    }

    #[test]
    fn nested_ttcf_directory_is_rejected() {
        let header = b"ttcf\x00\x01\x00\x00\x00\x00\x00\x01";
        assert!(matches!(
            SfntDirectory::parse(header, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
    }

    #[test]
    fn woff_version_is_never_accepted_as_a_directory() {
        let font = default_font();
        let mut woff = font.clone();
        woff[0..4].copy_from_slice(b"wOFF");
        assert!(matches!(
            SfntDirectory::parse(&woff, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert!(matches!(
            SfntContainer::open(&woff, 0),
            Err(TtError::UNKNOWN_FILE_FORMAT)
        ));
        assert_eq!(TAG_WOFF, make_tag(b'w', b'O', b'F', b'F'));
    }
}
