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

//! Cursor based marshalling of D-Bus values.
//!
//! The reader and the writer track absolute positions inside a whole
//! message buffer, which is what the D-Bus alignment rules are defined
//! against. Values are decoded into borrowed slices and strings so that
//! reading a message does not allocate beyond what the caller asks for.

use alloc::vec::Vec;

use crate::dbus_error::{DbusError, DbusResult};
use crate::dbus_names::{is_valid_object_path, validate_object_path};
use crate::dbus_signature::{MAX_SIGNATURE_LEN, type_alignment, validate_signature, validate_single_type};

/// Maximum length of an array in bytes (2^26 as mandated by the spec).
pub const MAX_ARRAY_LEN: usize = 1 << 26;

/// Byte order of a marshalled message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little-endian encoding, the marker byte `l`.
    Little,
    /// Big-endian encoding, the marker byte `B`.
    Big,
}

impl ByteOrder {
    /// The endianness marker byte of this order.
    #[inline]
    #[must_use]
    pub const fn marker(self) -> u8 {
        match self {
            Self::Little => b'l',
            Self::Big => b'B',
        }
    }

    /// Returns the order for an endianness marker byte.
    #[inline]
    #[must_use]
    pub const fn from_marker(marker: u8) -> Option<Self> {
        match marker {
            b'l' => Some(Self::Little),
            b'B' => Some(Self::Big),
            _ => None,
        }
    }

    #[inline]
    const fn read_u16(self, bytes: [u8; 2]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(bytes),
            Self::Big => u16::from_be_bytes(bytes),
        }
    }

    #[inline]
    const fn read_u32(self, bytes: [u8; 4]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }

    #[inline]
    const fn read_u64(self, bytes: [u8; 8]) -> u64 {
        match self {
            Self::Little => u64::from_le_bytes(bytes),
            Self::Big => u64::from_be_bytes(bytes),
        }
    }
}

fn truncated() -> DbusError {
    DbusError::invalid_message("message truncated while decoding")
}

/// Cursor over marshalled bytes.
///
/// The reader never reads past its `end`, so nested containers can be
/// decoded through sub-readers without escaping their bounds.
#[derive(Debug, Clone)]
pub struct DbusReader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    order: ByteOrder,
}

impl<'a> DbusReader<'a> {
    /// Creates a reader over all of `data`.
    #[inline]
    #[must_use]
    pub fn new(data: &'a [u8], order: ByteOrder) -> Self {
        Self {
            data,
            pos: 0,
            end: data.len(),
            order,
        }
    }

    /// Creates a reader that starts at `pos` inside `data`.
    #[inline]
    #[must_use]
    pub fn with_position(data: &'a [u8], order: ByteOrder, pos: usize) -> Self {
        let mut reader = Self::new(data, order);
        reader.pos = pos.min(data.len());
        reader
    }

    /// Returns the byte order of the data.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> ByteOrder {
        self.order
    }

    /// Returns the absolute position of the cursor.
    #[inline]
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Returns the number of bytes left before the end of the range.
    #[inline]
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.end
            .saturating_sub(self.pos)
    }

    /// Returns `true` when no bytes remain in the range.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Advances the cursor to the next multiple of `alignment`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the padding or the
    /// value that follows does not fit in the range.
    pub fn align(&mut self, alignment: usize) -> DbusResult<()> {
        if alignment == 0 {
            return Ok(());
        }
        let padding = (alignment - (self.pos % alignment)) % alignment;
        let target = self
            .pos
            .saturating_add(padding);
        if target > self.end {
            return Err(truncated());
        }
        self.pos = target;
        Ok(())
    }

    /// Reads `len` raw bytes.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the bytes do not fit
    /// in the range.
    pub fn read_bytes(&mut self, len: usize) -> DbusResult<&'a [u8]> {
        let target = self
            .pos
            .saturating_add(len);
        if target > self.end {
            return Err(truncated());
        }
        let bytes = self
            .data
            .get(self.pos..target)
            .ok_or_else(truncated)?;
        self.pos = target;
        Ok(bytes)
    }

    /// Reads a `BYTE` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_u8(&mut self) -> DbusResult<u8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    /// Reads an `INT16` or `UINT16` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_u16(&mut self) -> DbusResult<u16> {
        self.align(2)?;
        let bytes = self.read_bytes(2)?;
        let mut array = [0u8; 2];
        array.copy_from_slice(bytes);
        Ok(self
            .order
            .read_u16(array))
    }

    /// Reads a `BOOLEAN`, `INT32` or `UINT32` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_u32(&mut self) -> DbusResult<u32> {
        self.align(4)?;
        let bytes = self.read_bytes(4)?;
        let mut array = [0u8; 4];
        array.copy_from_slice(bytes);
        Ok(self
            .order
            .read_u32(array))
    }

    /// Reads an `INT64` or `UINT64` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_u64(&mut self) -> DbusResult<u64> {
        self.align(8)?;
        let bytes = self.read_bytes(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(self
            .order
            .read_u64(array))
    }

    /// Reads an `INT16` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_i16(&mut self) -> DbusResult<i16> {
        Ok(self.read_u16()? as i16)
    }

    /// Reads an `INT32` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_i32(&mut self) -> DbusResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    /// Reads an `INT64` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_i64(&mut self) -> DbusResult<i64> {
        Ok(self.read_u64()? as i64)
    }

    /// Reads an `UNIX_FD` (`h`) value.
    ///
    /// The value is the index of the file descriptor inside the list
    /// announced by the `UNIX_FDS` header field; resolving the index
    /// to a real descriptor is the caller's job.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_fd(&mut self) -> DbusResult<u32> {
        self.read_u32()
    }

    /// Reads a `BOOLEAN` value, rejecting encodings other than 0 and 1.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range or is neither 0 nor 1.
    pub fn read_bool(&mut self) -> DbusResult<bool> {
        match self.read_u32()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DbusError::invalid_message("boolean is neither 0 nor 1")),
        }
    }

    /// Reads a `DOUBLE` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the value does not
    /// fit in the range.
    pub fn read_f64(&mut self) -> DbusResult<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Reads a `STRING` or `OBJECT_PATH` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the length is
    /// invalid, the trailing nul is missing, an embedded nul is
    /// present or the text is not valid UTF-8.
    pub fn read_str(&mut self) -> DbusResult<&'a str> {
        self.align(4)?;
        let length = self.read_u32()? as usize;
        let text = self.read_bytes(length)?;
        let nul = self.read_bytes(1)?;
        if nul[0] != 0 {
            return Err(DbusError::invalid_message("string is not nul terminated"));
        }
        if text.contains(&0) {
            return Err(DbusError::invalid_message("string contains an embedded nul"));
        }
        core::str::from_utf8(text).map_err(|_| DbusError::invalid_message("string is not valid utf-8"))
    }

    /// Reads an `OBJECT_PATH` value and validates it.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the encoding is
    /// broken and [`DbusError::InvalidName`] when the path is invalid.
    pub fn read_object_path(&mut self) -> DbusResult<&'a str> {
        let path = self.read_str()?;
        validate_object_path(path)?;
        Ok(path)
    }

    /// Reads a `SIGNATURE` value and validates it.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the encoding is
    /// broken and [`DbusError::InvalidSignature`] when the signature is
    /// invalid.
    pub fn read_signature(&mut self) -> DbusResult<&'a str> {
        let length = self.read_u8()? as usize;
        if length > MAX_SIGNATURE_LEN {
            return Err(DbusError::invalid_signature(alloc::format!(
                "signature of {length} bytes exceeds the limit of {MAX_SIGNATURE_LEN}"
            )));
        }
        let text = self.read_bytes(length)?;
        let nul = self.read_bytes(1)?;
        if nul[0] != 0 {
            return Err(DbusError::invalid_message("signature is not nul terminated"));
        }
        let sig = core::str::from_utf8(text)
            .map_err(|_| DbusError::invalid_message("signature is not valid utf-8"))?;
        validate_signature(sig)?;
        Ok(sig)
    }

    /// Reads an array header and returns a reader over its elements.
    ///
    /// `element_alignment` is the alignment of the array element type.
    /// The returned reader is limited to the element data and the
    /// parent cursor is moved past the array.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the length exceeds
    /// [`MAX_ARRAY_LEN`], the padding does not fit or the element data
    /// runs out of the parent range.
    pub fn read_array(&mut self, element_alignment: usize) -> DbusResult<Self> {
        self.align(8)?;
        let length = self.read_u32()? as usize;
        if length > MAX_ARRAY_LEN {
            return Err(DbusError::invalid_message(alloc::format!(
                "array of {length} bytes exceeds the limit of {MAX_ARRAY_LEN}"
            )));
        }
        self.align(element_alignment)?;
        let start = self.pos;
        let target = start.saturating_add(length);
        if target > self.end {
            return Err(truncated());
        }
        self.pos = target;
        Ok(Self {
            data: self.data,
            pos: start,
            end: target,
            order: self.order,
        })
    }

    /// Aligns the cursor to a struct or dict entry boundary.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the padding does not
    /// fit in the range.
    pub fn read_struct(&mut self) -> DbusResult<()> {
        self.align(8)
    }

    /// Reads a variant header and returns its signature.
    ///
    /// The caller must align and read the value itself, since only the
    /// signature describes its type.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the encoding is
    /// broken and [`DbusError::InvalidSignature`] when the signature is
    /// not a single valid type.
    pub fn read_variant_signature(&mut self) -> DbusResult<&'a str> {
        let sig = self.read_signature()?;
        validate_single_type(sig)?;
        Ok(sig)
    }
}

/// Builder that appends marshalled bytes to a buffer.
#[derive(Debug, Clone)]
pub struct DbusWriter {
    data: Vec<u8>,
    order: ByteOrder,
}

impl DbusWriter {
    /// Creates an empty writer using `order`.
    #[must_use]
    pub fn new(order: ByteOrder) -> Self {
        Self {
            data: Vec::new(),
            order,
        }
    }

    /// Creates a writer with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(order: ByteOrder, capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            order,
        }
    }

    /// Returns the byte order used by the writer.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> ByteOrder {
        self.order
    }

    /// Returns the current absolute position.
    #[inline]
    #[must_use]
    pub fn position(&self) -> usize {
        self.data
            .len()
    }

    /// Returns the bytes written so far.
    #[inline]
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consumes the writer, returning its buffer.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Appends padding up to the next multiple of `alignment`.
    pub fn align(&mut self, alignment: usize) {
        if alignment == 0 {
            return;
        }
        while !self
            .data
            .len()
            .is_multiple_of(alignment)
        {
            self.data
                .push(0);
        }
    }

    /// Appends raw bytes.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.data
            .extend_from_slice(bytes);
    }

    /// Appends a `BYTE` value.
    pub fn write_u8(&mut self, value: u8) {
        self.data
            .push(value);
    }

    /// Appends an `INT16` or `UINT16` value.
    pub fn write_u16(&mut self, value: u16) {
        self.align(2);
        match self.order {
            ByteOrder::Little => self
                .data
                .extend_from_slice(&value.to_le_bytes()),
            ByteOrder::Big => self
                .data
                .extend_from_slice(&value.to_be_bytes()),
        }
    }

    /// Appends a `BOOLEAN`, `INT32` or `UINT32` value.
    pub fn write_u32(&mut self, value: u32) {
        self.align(4);
        match self.order {
            ByteOrder::Little => self
                .data
                .extend_from_slice(&value.to_le_bytes()),
            ByteOrder::Big => self
                .data
                .extend_from_slice(&value.to_be_bytes()),
        }
    }

    /// Appends an `INT64` or `UINT64` value.
    pub fn write_u64(&mut self, value: u64) {
        self.align(8);
        match self.order {
            ByteOrder::Little => self
                .data
                .extend_from_slice(&value.to_le_bytes()),
            ByteOrder::Big => self
                .data
                .extend_from_slice(&value.to_be_bytes()),
        }
    }

    /// Appends an `INT16` value.
    pub fn write_i16(&mut self, value: i16) {
        self.write_u16(value as u16);
    }

    /// Appends an `INT32` value.
    pub fn write_i32(&mut self, value: i32) {
        self.write_u32(value as u32);
    }

    /// Appends an `INT64` value.
    pub fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    /// Appends a `BOOLEAN` value as 0 or 1.
    pub fn write_bool(&mut self, value: bool) {
        self.write_u32(u32::from(value));
    }

    /// Appends a `DOUBLE` value.
    pub fn write_f64(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    /// Appends an `UNIX_FD` (`h`) value as its 4-byte index.
    ///
    /// `index` selects the descriptor inside the list announced by
    /// the `UNIX_FDS` header field of the message being encoded.
    pub fn write_fd(&mut self, index: u32) {
        self.write_u32(index);
    }

    /// Appends a `STRING` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when `value` contains an
    /// embedded nul byte.
    pub fn write_str(&mut self, value: &str) -> DbusResult<()> {
        if value
            .as_bytes()
            .contains(&0)
        {
            return Err(DbusError::invalid_message("string contains an embedded nul"));
        }
        self.align(4);
        self.write_u32(value.len() as u32);
        self.write_bytes(value.as_bytes());
        self.write_u8(0);
        Ok(())
    }

    /// Appends an `OBJECT_PATH` value after validating it.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when `path` is not a valid
    /// object path.
    pub fn write_object_path(&mut self, path: &str) -> DbusResult<()> {
        if !is_valid_object_path(path) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid object path: {path}"
            )));
        }
        self.write_str(path)
    }

    /// Appends a `SIGNATURE` value after validating it.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `sig` is not a
    /// valid signature or exceeds [`MAX_SIGNATURE_LEN`].
    pub fn write_signature(&mut self, sig: &str) -> DbusResult<()> {
        validate_signature(sig)?;
        if sig.len() > MAX_SIGNATURE_LEN {
            return Err(DbusError::invalid_signature(alloc::format!(
                "signature of {} bytes exceeds the limit of {MAX_SIGNATURE_LEN}",
                sig.len()
            )));
        }
        self.write_u8(sig.len() as u8);
        self.write_bytes(sig.as_bytes());
        self.write_u8(0);
        Ok(())
    }

    /// Runs `body` while writing an array around it.
    ///
    /// The array length is patched in after `body` finishes, so padding
    /// after the last element is never counted, as required by the
    /// specification.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when the encoded length
    /// exceeds [`MAX_ARRAY_LEN`], plus whatever `body` returns.
    pub fn write_array<F>(&mut self, element_alignment: usize, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        self.align(8);
        let length_pos = self
            .data
            .len();
        self.write_u32(0);
        self.align(element_alignment);
        let start = self
            .data
            .len();
        body(self)?;
        let length = self
            .data
            .len()
            .saturating_sub(start);
        if length > MAX_ARRAY_LEN {
            return Err(DbusError::invalid_message(alloc::format!(
                "array of {length} bytes exceeds the limit of {MAX_ARRAY_LEN}"
            )));
        }
        self.patch_u32(length_pos, length as u32)
    }

    /// Runs `body` while writing a struct or dict entry around it.
    pub fn write_struct<F>(&mut self, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        self.align(8);
        body(self)
    }

    /// Writes a variant holding the value produced by `body`.
    ///
    /// `sig` must describe exactly what `body` writes.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `sig` is not a
    /// single valid type, plus whatever `body` returns.
    pub fn write_variant<F>(&mut self, sig: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        validate_single_type(sig)?;
        self.write_signature(sig)?;
        body(self)
    }

    /// Overwrites the `UINT32` stored at `position`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when `position` does not
    /// hold four bytes of the buffer.
    pub fn patch_u32(&mut self, position: usize, value: u32) -> DbusResult<()> {
        let end = position.saturating_add(4);
        let target = self
            .data
            .get_mut(position..end)
            .ok_or_else(|| DbusError::invalid_message("patch position out of bounds"))?;
        let bytes = match self.order {
            ByteOrder::Little => value.to_le_bytes(),
            ByteOrder::Big => value.to_be_bytes(),
        };
        target.copy_from_slice(&bytes);
        Ok(())
    }

    /// Returns the alignment of the first type of `sig`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `sig` is empty or
    /// does not start with a valid type.
    pub fn first_type_alignment(sig: &str) -> DbusResult<usize> {
        let code = sig
            .as_bytes()
            .first()
            .copied()
            .ok_or_else(|| DbusError::invalid_signature("signature does not contain a type"))?;
        type_alignment(code)
            .ok_or_else(|| DbusError::invalid_signature(alloc::format!("invalid type code: {code}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn byte_order_markers_round_trip() {
        assert_eq!(ByteOrder::Little.marker(), b'l');
        assert_eq!(ByteOrder::Big.marker(), b'B');
        assert_eq!(ByteOrder::from_marker(b'l'), Some(ByteOrder::Little));
        assert_eq!(ByteOrder::from_marker(b'B'), Some(ByteOrder::Big));
        assert_eq!(ByteOrder::from_marker(b'x'), None);
    }

    #[test]
    fn writes_and_reads_fixed_values() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer.write_u8(0x7f);
        writer.write_u16(0x1234);
        writer.write_i32(-42);
        writer.write_u64(0x0102_0304_0506_0708);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_f64(0.5);
        writer.write_i16(-2);

        let bytes = writer.into_bytes();
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(
            reader
                .read_u8()
                .unwrap(),
            0x7f
        );
        assert_eq!(
            reader
                .read_u16()
                .unwrap(),
            0x1234
        );
        assert_eq!(
            reader
                .read_i32()
                .unwrap(),
            -42
        );
        assert_eq!(
            reader
                .read_u64()
                .unwrap(),
            0x0102_0304_0506_0708
        );
        assert!(
            reader
                .read_bool()
                .unwrap()
        );
        assert!(
            !reader
                .read_bool()
                .unwrap()
        );
        assert_eq!(
            reader
                .read_f64()
                .unwrap(),
            0.5
        );
        assert_eq!(
            reader
                .read_i16()
                .unwrap(),
            -2
        );
        assert!(reader.is_empty());
    }

    #[test]
    fn big_endian_round_trip() {
        let mut writer = DbusWriter::new(ByteOrder::Big);
        writer.write_u32(0xdead_beef);
        writer.write_u16(0x1234);
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xde, 0xad, 0xbe, 0xef, 0x12, 0x34]);

        let mut reader = DbusReader::new(&bytes, ByteOrder::Big);
        assert_eq!(
            reader
                .read_u32()
                .unwrap(),
            0xdead_beef
        );
        assert_eq!(
            reader
                .read_u16()
                .unwrap(),
            0x1234
        );
    }

    #[test]
    fn rejects_invalid_boolean_encoding() {
        let bytes = vec![2u8, 0, 0, 0];
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert!(
            reader
                .read_bool()
                .is_err()
        );
    }

    #[test]
    fn writes_and_reads_strings() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer
            .write_str("foo")
            .unwrap();
        writer
            .write_str("+")
            .unwrap();
        writer
            .write_str("")
            .unwrap();
        writer
            .write_object_path("/a/b")
            .unwrap();
        writer
            .write_signature("a{sv}")
            .unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(
            bytes,
            vec![
                // "foo": length, text, trailing nul.
                3, 0, 0, 0, b'f', b'o', b'o', 0, // "+": length, text, trailing nul.
                1, 0, 0, 0, b'+', 0, // "": two pad bytes, length 0, trailing nul.
                0, 0, 0, 0, 0, 0, 0, // "/a/b": three pad bytes, length 4, text, trailing nul.
                0, 0, 0, 4, 0, 0, 0, b'/', b'a', b'/', b'b', 0,
                // Signature "a{sv}": one byte length, text, trailing nul.
                5, b'a', b'{', b's', b'v', b'}', 0,
            ]
        );

        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            "foo"
        );
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            "+"
        );
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            ""
        );
        assert_eq!(
            reader
                .read_object_path()
                .unwrap(),
            "/a/b"
        );
        assert_eq!(
            reader
                .read_signature()
                .unwrap(),
            "a{sv}"
        );
        assert!(reader.is_empty());
    }

    #[test]
    fn rejects_broken_strings() {
        // Length 3 without a trailing nul.
        let bytes = vec![3, 0, 0, 0, b'a', b'b', b'c'];
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert!(
            reader
                .read_str()
                .is_err()
        );

        // Embedded nul inside the text.
        let bytes = vec![3, 0, 0, 0, b'a', 0, b'c', 0];
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert!(
            reader
                .read_str()
                .is_err()
        );

        // Invalid UTF-8.
        let bytes = vec![1, 0, 0, 0, 0xff, 0];
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert!(
            reader
                .read_str()
                .is_err()
        );

        // Invalid object path.
        let mut writer = DbusWriter::new(ByteOrder::Little);
        assert!(
            writer
                .write_object_path("nope")
                .is_err()
        );
        assert!(
            writer
                .write_str("bad\0string")
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_signatures_on_the_wire() {
        let bytes = vec![1, b'z', 0];
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert!(
            reader
                .read_signature()
                .is_err()
        );
    }

    #[test]
    fn array_round_trip_patches_length() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer
            .write_array(4, |writer| {
                writer.write_str("one")?;
                writer.write_str("two")?;
                Ok(())
            })
            .unwrap();
        let bytes = writer.into_bytes();
        // 4 byte length + "one" (8 bytes) + "two" (8 bytes).
        assert_eq!(&bytes[0..4], &[16, 0, 0, 0]);

        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        let mut elements = reader
            .read_array(4)
            .unwrap();
        assert_eq!(
            elements
                .read_str()
                .unwrap(),
            "one"
        );
        assert_eq!(
            elements
                .read_str()
                .unwrap(),
            "two"
        );
        assert!(elements.is_empty());
        assert!(reader.is_empty());
    }

    #[test]
    fn empty_array_keeps_element_alignment_padding() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer
            .write_array(8, |_writer| Ok(()))
            .unwrap();
        let bytes = writer.into_bytes();
        // 4 bytes length then 4 bytes of padding to the 8 byte boundary.
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], &[0, 0, 0, 0]);

        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        let elements = reader
            .read_array(8)
            .unwrap();
        assert!(elements.is_empty());
        assert_eq!(reader.position(), 8);
    }

    #[test]
    fn struct_and_variant_round_trip() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer
            .write_struct(|writer| {
                writer.write_u8(1);
                writer.write_variant("s", |writer| writer.write_str("hello"))
            })
            .unwrap();
        let bytes = writer.into_bytes();

        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        reader
            .read_struct()
            .unwrap();
        assert_eq!(
            reader
                .read_u8()
                .unwrap(),
            1
        );
        assert_eq!(
            reader
                .read_variant_signature()
                .unwrap(),
            "s"
        );
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            "hello"
        );
    }

    #[test]
    fn reader_never_escapes_its_range() {
        let bytes = vec![4u8, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        let mut nested = reader
            .read_array(1)
            .unwrap();
        assert_eq!(
            nested
                .read_u8()
                .unwrap(),
            1
        );
        assert_eq!(
            nested
                .read_u8()
                .unwrap(),
            2
        );
        assert_eq!(nested.position(), 6);
        assert_eq!(reader.position(), 8);

        // The announced length runs past the end of the buffer.
        let mut short = DbusReader::new(&bytes[..6], ByteOrder::Little);
        assert!(
            short
                .read_array(1)
                .is_err()
        );
        assert!(
            short
                .align(8)
                .is_err()
        );
        assert!(
            short
                .read_bytes(4)
                .is_err()
        );
    }

    #[test]
    fn patch_rejects_out_of_bounds_positions() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer.write_u32(1);
        assert!(
            writer
                .patch_u32(0, 2)
                .is_ok()
        );
        assert!(
            writer
                .patch_u32(1, 2)
                .is_err()
        );
        assert_eq!(writer.bytes(), &[2, 0, 0, 0]);
    }

    #[test]
    fn fd_values_round_trip_with_alignment() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer.write_u8(1);
        // The `h` value must be padded to the 4 byte boundary.
        writer.write_fd(3);
        writer
            .write_array(8, |writer| {
                writer.write_struct(|writer| {
                    writer.write_fd(0);
                    writer.write_u32(42);
                    Ok(())
                })
            })
            .unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(bytes.len() % 4, 0);

        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(
            reader
                .read_u8()
                .unwrap(),
            1
        );
        assert_eq!(
            reader
                .read_fd()
                .unwrap(),
            3
        );
        let mut array = reader
            .read_array(8)
            .unwrap();
        array
            .read_struct()
            .unwrap();
        assert_eq!(
            array
                .read_fd()
                .unwrap(),
            0
        );
        assert_eq!(
            array
                .read_u32()
                .unwrap(),
            42
        );
        assert!(array.is_empty());
        assert!(reader.is_empty());
    }

    #[test]
    fn first_type_alignment_reports_element_alignment() {
        assert_eq!(DbusWriter::first_type_alignment("i"), Ok(4));
        assert_eq!(DbusWriter::first_type_alignment("h"), Ok(4));
        assert_eq!(DbusWriter::first_type_alignment("x"), Ok(8));
        assert_eq!(DbusWriter::first_type_alignment("(ii)"), Ok(8));
        assert!(DbusWriter::first_type_alignment("").is_err());
        assert!(DbusWriter::first_type_alignment("z").is_err());
    }
}
