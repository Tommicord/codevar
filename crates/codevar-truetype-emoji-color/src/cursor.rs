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

//! Bounds-checked big-endian reader shared by the color table parsers.

use codevar_truetype_core::{TtError, TtResult};

/// A cursor over a borrowed byte slice.
///
/// Every read is bounds-checked against the underlying slice; a read
/// past the end reports [`TtError::INVALID_TABLE`] instead of
/// panicking, so malformed font tables can never crash the caller.
pub(crate) struct Reader<'a> {
    /// The bytes being read.
    data: &'a [u8],
    /// Read offset into `data`.
    pos: usize,
}

/// The parser crates use a subset of this reader; the unit tests
/// exercise the whole API.
#[allow(dead_code)]
impl<'a> Reader<'a> {
    /// Creates a reader positioned at the start of `data`.
    #[inline]
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    /// Creates a reader positioned at `pos`.
    ///
    /// Returns `None` when `pos` is past the end of `data`.
    #[inline]
    pub(crate) fn at(data: &'a [u8], pos: usize) -> Option<Self> {
        if pos > data.len() {
            return None;
        }
        Some(Reader { data, pos })
    }

    /// The total number of readable bytes.
    #[inline]
    pub(crate) const fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` when nothing can be read anymore.
    #[inline]
    pub(crate) const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The current read offset.
    #[inline]
    pub(crate) const fn pos(&self) -> usize {
        self.pos
    }

    /// Moves the cursor to `pos`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — `pos` is past the end.
    pub(crate) fn seek(&mut self, pos: usize) -> TtResult<()> {
        if pos > self.data.len() {
            return Err(TtError::INVALID_TABLE);
        }
        self.pos = pos;
        Ok(())
    }

    /// Reads `count` bytes starting at the cursor, advancing it.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than `count` bytes remain.
    pub(crate) fn take(&mut self, count: usize) -> TtResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(count)
            .ok_or(TtError::INVALID_TABLE)?;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or(TtError::INVALID_TABLE)?;
        self.pos = end;
        Ok(bytes)
    }

    /// Reads one unsigned byte.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the cursor is at the end.
    pub(crate) fn u8(&mut self) -> TtResult<u8> {
        let bytes = self.take(1)?;
        Ok(bytes[0])
    }

    /// Reads one signed byte.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the cursor is at the end.
    pub(crate) fn i8(&mut self) -> TtResult<i8> {
        Ok(i8::from_be_bytes(self.take(1)?.try_into().unwrap_or([0])))
    }

    /// Reads a big-endian `u16`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than two bytes remain.
    pub(crate) fn u16(&mut self) -> TtResult<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes(bytes.try_into().unwrap_or([0, 0])))
    }

    /// Reads a big-endian `i16`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than two bytes remain.
    pub(crate) fn i16(&mut self) -> TtResult<i16> {
        let bytes = self.take(2)?;
        Ok(i16::from_be_bytes(bytes.try_into().unwrap_or([0, 0])))
    }

    /// Reads a big-endian `u32`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than four bytes remain.
    pub(crate) fn u32(&mut self) -> TtResult<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes(bytes.try_into().unwrap_or([0, 0, 0, 0])))
    }

    /// Reads a big-endian `i32`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — fewer than four bytes remain.
    pub(crate) fn i32(&mut self) -> TtResult<i32> {
        let bytes = self.take(4)?;
        Ok(i32::from_be_bytes(bytes.try_into().unwrap_or([0, 0, 0, 0])))
    }
}

/// Reads a big-endian `u16` at `offset` without a cursor.
///
/// # Errors
///
/// * [`TtError::INVALID_TABLE`] — the two bytes are out of range.
pub(crate) fn read_u16_at(data: &[u8], offset: usize) -> TtResult<u16> {
    let bytes = data
        .get(
            offset
                ..offset
                    .checked_add(2)
                    .ok_or(TtError::INVALID_TABLE)?,
        )
        .ok_or(TtError::INVALID_TABLE)?;
    Ok(u16::from_be_bytes(bytes.try_into().unwrap_or([0, 0])))
}

/// Reads a big-endian `u32` at `offset` without a cursor.
///
/// # Errors
///
/// * [`TtError::INVALID_TABLE`] — the four bytes are out of range.
pub(crate) fn read_u32_at(data: &[u8], offset: usize) -> TtResult<u32> {
    let bytes = data
        .get(
            offset
                ..offset
                    .checked_add(4)
                    .ok_or(TtError::INVALID_TABLE)?,
        )
        .ok_or(TtError::INVALID_TABLE)?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap_or([0, 0, 0, 0])))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twenty bytes laid out as one read of every width:
    /// `u16`, `u32`, `u8`, `i8`, `i16`, `i32`, `u32`, `u16`.
    const BYTES: [u8; 20] = [
        0x12, 0x34, // [0..2]   u16
        0x56, 0x78, 0x9A, 0xBC, // [2..6]   u32
        0x01, // [6]      u8
        0xFE, // [7]      i8
        0x7F, 0x80, // [8..10]   i16
        0xFF, 0x80, 0x00, 0x00, // [10..14]  i32
        0x00, 0x01, 0xFF, 0xFF, // [14..18]  u32
        0xFF, 0xFF, // [18..20]  u16
    ];

    #[test]
    fn decodes_every_width_and_tracks_the_cursor() {
        let mut reader = Reader::new(&BYTES);
        assert_eq!(reader.len(), 20);
        assert!(!reader.is_empty());
        assert_eq!(reader.pos(), 0);

        assert_eq!(reader.u16(), Ok(0x1234));
        assert_eq!(reader.pos(), 2);
        assert_eq!(reader.u32(), Ok(0x5678_9ABC));
        assert_eq!(reader.pos(), 6);
        assert_eq!(reader.u8(), Ok(1));
        assert_eq!(reader.i8(), Ok(-2));
        assert_eq!(reader.i16(), Ok(0x7F80));
        assert_eq!(reader.i32(), Ok(-0x0080_0000));
        assert_eq!(reader.pos(), 14);
        assert_eq!(reader.u32(), Ok(0x0001_FFFF));
        assert_eq!(reader.u16(), Ok(u16::MAX));
        assert_eq!(reader.pos(), 20);

        // At the end every read fails instead of panicking, and the
        // cursor stays put.
        assert_eq!(reader.u8(), Err(TtError::INVALID_TABLE));
        assert_eq!(reader.pos(), 20);
    }

    #[test]
    fn at_accepts_the_end_of_the_slice_and_rejects_past_it() {
        assert!(Reader::at(&BYTES, 0).is_some());
        assert!(Reader::at(&BYTES, BYTES.len()).is_some());
        assert!(Reader::at(&BYTES, BYTES.len() + 1).is_none());
        assert!(Reader::at(&[], 0).is_some());
        assert!(Reader::at(&[], 1).is_none());
    }

    #[test]
    fn seek_moves_the_cursor_and_rejects_positions_past_the_end() {
        let mut reader = Reader::new(&BYTES);
        reader.seek(18).unwrap();
        assert_eq!(reader.pos(), 18);
        assert_eq!(reader.u16(), Ok(u16::MAX));
        assert_eq!(reader.pos(), 20);

        // A failed seek leaves the cursor where it was.
        assert!(matches!(reader.seek(21), Err(TtError::INVALID_TABLE)));
        assert_eq!(reader.pos(), 20);
        reader.seek(20).unwrap();
        assert_eq!(reader.pos(), 20);
    }

    #[test]
    fn take_bounds_reads_without_advancing_on_failure() {
        let mut reader = Reader::new(&BYTES);
        assert_eq!(reader.take(4).unwrap(), &BYTES[0..4]);
        assert_eq!(reader.pos(), 4);

        // A failed take leaves the cursor where it was.
        assert!(reader.take(17).is_err());
        assert_eq!(reader.pos(), 4);

        assert_eq!(reader.take(16).unwrap(), &BYTES[4..20]);
        assert_eq!(reader.pos(), 20);
        reader.take(0).unwrap();
        assert_eq!(reader.take(1), Err(TtError::INVALID_TABLE));

        // `pos + count` is checked, so `usize::MAX` cannot wrap the
        // end of the slice back into range.
        let mut at_start = Reader::at(&BYTES, 1).unwrap();
        assert_eq!(at_start.take(usize::MAX), Err(TtError::INVALID_TABLE));
        assert_eq!(at_start.pos(), 1);
    }

    #[test]
    fn an_empty_slice_yields_no_data() {
        let mut reader = Reader::new(&[]);
        assert_eq!(reader.len(), 0);
        assert!(reader.is_empty());
        assert_eq!(reader.pos(), 0);
        assert!(reader.take(0).unwrap().is_empty());
        assert_eq!(reader.u8(), Err(TtError::INVALID_TABLE));
        assert_eq!(reader.u16(), Err(TtError::INVALID_TABLE));
        assert_eq!(reader.u32(), Err(TtError::INVALID_TABLE));
        reader.seek(0).unwrap();
    }

    #[test]
    fn read_helpers_check_their_offset() {
        assert_eq!(read_u16_at(&BYTES, 0), Ok(0x1234));
        assert_eq!(read_u32_at(&BYTES, 2), Ok(0x5678_9ABC));
        assert_eq!(read_u16_at(&BYTES, 19), Err(TtError::INVALID_TABLE));
        assert_eq!(read_u32_at(&BYTES, 17), Err(TtError::INVALID_TABLE));
        assert_eq!(read_u16_at(&BYTES, usize::MAX), Err(TtError::INVALID_TABLE));
        assert_eq!(read_u32_at(&[], 0), Err(TtError::INVALID_TABLE));
    }
}
