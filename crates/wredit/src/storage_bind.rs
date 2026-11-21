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
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Bindings between the Wredit block-storage layer and the edit engine.
//!
//! A [`BoundDocument`] pairs an on-disk regular file (accessed through
//! [`WritableFile`]) with an in-memory edit buffer
//! ([`StreamWritable`]). Loading pulls the file's bytes into packed raw
//! elements; editing happens in the gap buffer; saving flushes staged
//! edits back through the writable file and persists them to the block
//! device.
//!
//! The edit buffer stores one Unicode scalar value per packed element
//! ([`make_packed`]`(scalar, [`COMMON_MASK`])`). Encoding is therefore a
//! property of the byte stream at the boundary: loading decodes the
//! file's bytes with the document's [`DocumentEncoding`], and saving
//! encodes the scalar stream back out. A leading byte-order mark is
//! stripped on load and re-emitted on save when the document was loaded
//! with one (or created as UTF-16/UTF-32).
//!
//! # Example
//!
//! ```rust
//! use codevar_wredit::error::Result;
//! use codevar_wredit::{DocumentEncoding, TextEditableEncoding};
//! use codevar_wredit::fs_block::MemoryDisk;
//! use codevar_wredit::fs::DiskFs;
//! use codevar_wredit::storage_bind::BoundDocument;
//!
//! fn edit_stored_file() -> Result<()> {
//!     let disk = MemoryDisk::new(256)?;
//!     let mut fs = DiskFs::format(disk, 256, 128)?;
//!     let inode = fs.create(2, "notes.txt", 0x8000)?;
//!     let mut bound: BoundDocument<'_, MemoryDisk, 256, 1024> =
//!         BoundDocument::open(&mut fs, inode)?;
//!     let cursor = bound.cursor_id();
//!     bound.stream_mut().insert_str(cursor, "X")?;
//!     bound.set_encoding(DocumentEncoding::Utf8)?;
//!     bound.save()?;
//!     Ok(())
//! }
//! ```

use alloc::vec::Vec;

use codevar_logger::{log_error, log_info};

use crate::encoding::{DocumentEncoding, is_valid_scalar};
use crate::encoding_dispatch::{DecodePolicy, EncodePolicy, decode, encode, encode_with_bom};
use crate::error::{Error, Result};
use crate::fs::DiskFs;
use crate::fs_block::BlockDevice;
use crate::fs_writable::WritableFile;
use crate::{
    Cursor, StreamWritable, TextEditableBufferQuery, TextEditableCursor, TextEditableDelete,
    TextEditableEncoding, TextEditableOverwrite, TextEditablePut, TextEditableReplace, TextEditableSelection,
    Writable, packed_char,
};

/// Whether a save emits the selected encoding's byte-order mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BomPolicy {
    /// Save without a byte-order mark.
    Omit,
    /// Prefix the encoded bytes with the encoding's mark.
    Emit,
}

/// A document that lives both on disk and in the edit engine.
///
/// Loading reads the file's bytes into packed raw elements; editing
/// happens in the in-memory gap buffer; saving flushes staged edits and
/// writes the result back through [`WritableFile`].
///
/// The writable file borrows the filesystem mutably for the lifetime
/// of the bound document, so the filesystem cannot be used directly
/// while a document is bound.
///
/// # Type parameters
///
/// * `D`       - the underlying block device
/// * `STAGING` - write-cache size of the backing [`WritableFile`], in bytes
/// * `GAP_SIZE` - private gap window size of the edit buffer, in elements
pub struct BoundDocument<'fs, D: BlockDevice, const STAGING: usize, const GAP_SIZE: usize> {
    /// The on-disk file being edited.
    file: WritableFile<'fs, D, STAGING>,
    /// Inode of the open file.
    inode: u32,
    /// In-memory edit buffer.
    stream: StreamWritable<u32, u32, GAP_SIZE>,
    /// Encoding used to decode the file and to encode it on save.
    encoding: DocumentEncoding,
    /// Whether a save re-emits the byte-order mark.
    bom: BomPolicy,
    /// File size at load time, used to detect growth on save.
    loaded_size: u64,
    /// `true` when the edit buffer has unsaved changes.
    dirty: bool,
    /// Id of the active cursor.
    cursor_id: usize,
}

impl<'fs, D: BlockDevice, const STAGING: usize, const GAP_SIZE: usize>
    BoundDocument<'fs, D, STAGING, GAP_SIZE>
{
    /// Opens an existing regular file for editing.
    ///
    /// The file's entire contents are read into the edit buffer. The
    /// returned document owns the mutable borrow of `fs` through the
    /// [`WritableFile`], so `fs` cannot be used directly until the
    /// document is dropped.
    ///
    /// The encoding is sniffed from a leading byte-order mark and falls
    /// back to [`DocumentEncoding::Utf8`] when there is none; use
    /// [`Self::open_with_encoding`] for a legacy file without a mark.
    /// Malformed bytes are replaced with U+FFFD so a damaged file still
    /// opens instead of failing the load.
    ///
    /// Staged gaps auto-commit as they fill, so files larger than one
    /// gap window load correctly.
    ///
    /// # Errors
    ///
    /// * [`Error::IsADirectory`] when `inode` is not a regular file
    /// * [`Error::NotFound`] when `inode` does not exist
    /// * Propagates device I/O errors from reading the file
    /// * [`Error::CapacityExhausted`] when the edit buffer cannot
    ///   grow to hold the file contents
    pub fn open(fs: &'fs mut DiskFs<D>, inode: u32) -> Result<Self> {
        Self::open_with_optional_encoding(fs, inode, None)
    }

    /// Opens an existing regular file with an explicit encoding.
    ///
    /// Unlike [`Self::open`], no byte-order-mark sniffing happens: the
    /// file is decoded with `encoding` and that encoding is retained for
    /// the next save. A matching leading mark is still detected and
    /// preserved. Malformed byte sequences are replaced with U+FFFD,
    /// matching the default load policy of [`Self::open`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::open`], plus [`Error::Encoding`] when
    /// `encoding` carries an out-of-range single-byte index.
    pub fn open_with_encoding(
        fs: &'fs mut DiskFs<D>,
        inode: u32,
        encoding: DocumentEncoding,
    ) -> Result<Self> {
        Self::open_with_optional_encoding(fs, inode, Some(encoding))
    }

    fn open_with_optional_encoding(
        fs: &'fs mut DiskFs<D>,
        inode: u32,
        requested: Option<DocumentEncoding>,
    ) -> Result<Self> {
        let ino = fs.read_inode(inode)?;
        if !ino.is_regular() {
            return Err(Error::IsADirectory {
                what: "bound document target",
            });
        }
        let loaded_size = u64::from(ino.size);
        let mut file = WritableFile::new(fs, inode)?;

        // Read the whole file in bounded chunks.
        let mut content = Vec::with_capacity(loaded_size.min(4096) as usize);
        let mut buf = [0u8; 512];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            content.extend_from_slice(&buf[..n]);
        }

        let encoding = requested
            .or_else(|| DocumentEncoding::sniff_bom(&content))
            .unwrap_or_default();
        Self::bind(file, inode, &content, encoding, loaded_size)
    }

    /// Builds a document from already-read bytes.
    fn bind(
        file: WritableFile<'fs, D, STAGING>,
        inode: u32,
        bytes: &[u8],
        encoding: DocumentEncoding,
        loaded_size: u64,
    ) -> Result<Self> {
        if !encoding.is_valid() {
            log_error!("storage_bind: rejecting an invalid document encoding");
            return Err(Error::Encoding);
        }
        let bom = if encoding.bom().is_empty() || !bytes.starts_with(encoding.bom()) {
            BomPolicy::Omit
        } else {
            BomPolicy::Emit
        };
        let scalars = decode(bytes, encoding, DecodePolicy::Replace)?;

        let mut stream = StreamWritable::with_capacity(scalars.len());
        let cursor_id = stream
            .append_cursor(Cursor::new())
            .ok_or(Error::LimitExceeded { what: "cursor pool" })?;
        stream.set_active_cursor(cursor_id);
        stream.set_document_encoding(encoding)?;
        stream.insert_scalars(cursor_id, &scalars)?;
        stream.flush_all();

        log_info!(
            "storage_bind: loaded inode {inode} ({loaded_size} bytes) as {}",
            encoding.name()
        );

        Ok(Self {
            file,
            inode,
            stream,
            encoding,
            bom,
            loaded_size,
            dirty: false,
            cursor_id,
        })
    }

    /// Creates a new empty regular file and binds it for editing.
    ///
    /// The document starts in [`DocumentEncoding::Utf8`]. Use
    /// [`Self::create_with_encoding`] to select another encoding.
    ///
    /// # Errors
    ///
    /// Propagates filesystem errors from creating the file and the
    /// writable file, and [`Error::LimitExceeded`] when the
    /// cursor pool is full.
    pub fn create(fs: &'fs mut DiskFs<D>, parent: u32, name: &str, mode: u16) -> Result<Self> {
        Self::create_with_encoding(fs, parent, name, mode, DocumentEncoding::Utf8)
    }

    /// Creates a new empty regular file with an explicit encoding.
    ///
    /// A UTF-16 or UTF-32 document is created with a byte-order mark so
    /// that a later [`Self::open`] can sniff the encoding; UTF-8 and
    /// single-byte documents are created without one.
    ///
    /// # Errors
    ///
    /// Same as [`Self::create`], plus [`Error::Encoding`] when
    /// `encoding` carries an out-of-range single-byte index.
    pub fn create_with_encoding(
        fs: &'fs mut DiskFs<D>,
        parent: u32,
        name: &str,
        mode: u16,
        encoding: DocumentEncoding,
    ) -> Result<Self> {
        if !encoding.is_valid() {
            log_error!("storage_bind: rejecting an invalid document encoding");
            return Err(Error::Encoding);
        }
        let inode = fs.create(parent, name, mode)?;
        let file = WritableFile::new(fs, inode)?;
        let mut stream = StreamWritable::new();
        let cursor_id = stream
            .append_cursor(Cursor::new())
            .ok_or(Error::LimitExceeded { what: "cursor pool" })?;
        stream.set_active_cursor(cursor_id);
        stream.set_document_encoding(encoding)?;

        // Without a mark a UTF-16/UTF-32 file cannot be told apart from
        // its alternatives (or from UTF-8) when it is reopened.
        let bom = match encoding {
            DocumentEncoding::Utf16Le
            | DocumentEncoding::Utf16Be
            | DocumentEncoding::Utf32Le
            | DocumentEncoding::Utf32Be => BomPolicy::Emit,
            DocumentEncoding::Utf8 | DocumentEncoding::SingleByte(_) => BomPolicy::Omit,
        };
        log_info!(
            "storage_bind: created inode {inode} named {name:?} as {}",
            encoding.name()
        );
        Ok(Self {
            file,
            inode,
            stream,
            encoding,
            bom,
            loaded_size: 0,
            dirty: false,
            cursor_id,
        })
    }

    /// Inode of the bound file.
    #[must_use]
    #[inline]
    pub const fn inode(&self) -> u32 {
        self.inode
    }

    /// Id of the active cursor.
    #[must_use]
    #[inline]
    pub const fn cursor_id(&self) -> usize {
        self.cursor_id
    }

    /// The encoding used to decode the file and to encode it on save.
    #[must_use]
    #[inline]
    pub const fn encoding(&self) -> DocumentEncoding {
        self.encoding
    }

    /// `true` when [`Self::save`] prefixes the file with the encoding's
    /// byte-order mark.
    #[must_use]
    #[inline]
    pub const fn emits_bom(&self) -> bool {
        matches!(self.bom, BomPolicy::Emit)
    }

    /// Selects the encoding used on the next save.
    ///
    /// The in-memory scalar stream is not transcoded; the document is
    /// simply written back through the new codec. The document is
    /// marked dirty because the on-disk bytes will change.
    ///
    /// # Errors
    ///
    /// [`Error::Encoding`] when `encoding` carries an
    /// out-of-range single-byte index; the current encoding is kept.
    pub fn set_encoding(&mut self, encoding: DocumentEncoding) -> Result<()> {
        if !encoding.is_valid() {
            log_error!("storage_bind: rejecting an invalid document encoding");
            return Err(Error::Encoding);
        }
        self.encoding = encoding;
        self.stream.set_document_encoding(encoding)?;
        self.dirty = true;
        Ok(())
    }

    /// Controls whether the next save emits the byte-order mark.
    ///
    /// Marks the document dirty when the policy changes.
    #[inline]
    pub fn set_bom_policy(&mut self, policy: BomPolicy) {
        if self.bom != policy {
            self.bom = policy;
            self.dirty = true;
        }
    }

    /// Borrows the edit buffer immutably.
    #[must_use]
    #[inline]
    pub const fn stream(&self) -> &StreamWritable<u32, u32, GAP_SIZE> {
        &self.stream
    }

    /// Borrows the edit buffer mutably.
    ///
    /// Marks the document dirty, since mutable access may change it.
    #[inline]
    pub fn stream_mut(&mut self) -> &mut StreamWritable<u32, u32, GAP_SIZE> {
        self.dirty = true;
        &mut self.stream
    }

    /// `true` when the edit buffer has unsaved changes.
    #[must_use]
    #[inline]
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Byte length of the file at the last load or save.
    #[must_use]
    #[inline]
    pub const fn loaded_size(&self) -> u64 {
        self.loaded_size
    }

    /// Number of committed elements in the edit buffer.
    #[must_use]
    #[inline]
    pub fn raw_length(&self) -> usize {
        self.stream.raw_length()
    }

    /// Commits every staged gap in the edit buffer.
    ///
    /// Returns the number of committed elements. Does not persist to
    /// disk; use [`Self::save`] for that.
    #[inline]
    pub fn flush_all(&mut self) -> usize {
        self.stream.flush_all()
    }

    /// Saves the edit buffer back to the on-disk file.
    ///
    /// Staged gaps are committed first, then the scalar stream is
    /// encoded with the document's [`DocumentEncoding`] and written
    /// from position 0 through [`WritableFile`]. The file is truncated
    /// to the encoded length, so a shorter save (a different encoding,
    /// or deleted text) never leaves stale trailing bytes.
    ///
    /// Encoding is strict: a scalar the target encoding cannot
    /// represent returns [`Error::UnmappableCharacter`] before
    /// anything is written, so the stored file is never partially
    /// rewritten.
    ///
    /// # Errors
    ///
    /// * [`Error::Encoding`] when the buffer holds a non-scalar
    ///   payload
    /// * [`Error::UnmappableCharacter`] when the target encoding
    ///   cannot represent a scalar
    /// * Propagates device I/O errors from writing, truncating and
    ///   syncing
    pub fn save(&mut self) -> Result<()> {
        self.flush_all();

        let length = self.stream.raw_length();
        let elements = self.stream.raw_slice(0, length).unwrap_or(&[]);
        let mut scalars = Vec::with_capacity(elements.len());
        for &element in elements {
            let value = packed_char(element);
            if !is_valid_scalar(value) {
                log_error!("storage_bind: buffer holds non-scalar value U+{value:04X}");
                return Err(Error::Encoding);
            }
            scalars.push(value);
        }

        let bytes = match self.bom {
            BomPolicy::Emit => encode_with_bom(&scalars, self.encoding, EncodePolicy::Strict)?,
            BomPolicy::Omit => encode(&scalars, self.encoding, EncodePolicy::Strict)?,
        };
        self.file.set_position(0)?;
        self.file.write(&bytes)?;
        self.file.truncate(bytes.len() as u64)?;
        self.file.sync_all()?;

        self.loaded_size = bytes.len() as u64;
        self.dirty = false;

        log_info!(
            "storage_bind: saved inode {} ({} bytes, {})",
            self.inode,
            bytes.len(),
            self.encoding.name()
        );

        Ok(())
    }

    /// Commits staged edits and saves to disk in one step.
    ///
    /// Equivalent to [`Self::flush_all`] followed by [`Self::save`].
    ///
    /// # Errors
    ///
    /// Propagates device I/O errors from saving.
    pub fn commit_and_save(&mut self) -> Result<()> {
        self.save()
    }

    // --- Delegated editing operations ---

    /// Stages `value` in the active cursor's private gap and advances
    /// the caret. Returns the caret's new logical position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn put(&mut self, value: u32) -> Result<usize> {
        self.dirty = true;
        self.stream.put(self.cursor_id, value)
    }

    /// Stages `value` in the cursor with `cursor_id`'s private gap and
    /// advances the caret. Returns the caret's new logical position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn put_at(&mut self, cursor_id: usize, value: u32) -> Result<usize> {
        self.dirty = true;
        self.stream.put(cursor_id, value)
    }

    /// Stages `value` in a shared gap the active cursor belongs to and
    /// advances the caret. Returns the new logical position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn put_shared(&mut self, value: u32) -> Result<usize> {
        self.dirty = true;
        self.stream.put_shared(self.cursor_id, value)
    }

    /// Deletes the element at the active caret (forward delete).
    /// Returns the number of removed elements.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn delete(&mut self) -> Result<usize> {
        self.dirty = true;
        self.stream.delete(self.cursor_id)
    }

    /// Deletes the element at `cursor_id`'s caret (forward delete).
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn delete_at(&mut self, cursor_id: usize) -> Result<usize> {
        self.dirty = true;
        self.stream.delete(cursor_id)
    }

    /// Deletes the element before the active caret (backspace).
    /// Returns the number of removed elements.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn backspace(&mut self) -> Result<usize> {
        self.dirty = true;
        self.stream.backspace(self.cursor_id)
    }

    /// Deletes the committed half-open range `[start, end)`.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn delete_range(&mut self, start: usize, end: usize) -> Result<usize> {
        self.dirty = true;
        self.stream.delete_range(start, end)
    }

    /// Deletes the element at the active caret and stages `value` in
    /// its place. Returns the new logical caret position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn replace(&mut self, value: u32) -> Result<usize> {
        self.dirty = true;
        self.stream.replace(self.cursor_id, value)
    }

    /// Replaces the committed range `[start, end)` with `values`,
    /// committed immediately.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn replace_range(&mut self, start: usize, end: usize, values: &[u32]) -> Result<usize> {
        self.dirty = true;
        self.stream.replace_range(start, end, values)
    }

    /// Stages a merge element that overwrites the element at the active
    /// caret on flush (or appends when the caret is at the end).
    /// Returns the new logical caret position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn overwrite(&mut self, value: u32) -> Result<usize> {
        self.dirty = true;
        self.stream.overwrite(self.cursor_id, value)
    }

    /// Overwrites committed elements in place starting at `start`;
    /// `values` extending past the document end are appended.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn overwrite_slice(&mut self, start: usize, values: &[u32]) -> Result<usize> {
        self.dirty = true;
        self.stream.overwrite_slice(start, values)
    }

    /// Moves the active caret forward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_forward(&mut self, amount: usize) -> Result<usize> {
        self.stream
            .move_cursor_forward(self.cursor_id, amount)
    }

    /// Moves `cursor_id`'s caret forward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_forward_at(&mut self, cursor_id: usize, amount: usize) -> Result<usize> {
        self.stream.move_cursor_forward(cursor_id, amount)
    }

    /// Moves the active caret backward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_backward(&mut self, amount: usize) -> Result<usize> {
        self.stream
            .move_cursor_backward(self.cursor_id, amount)
    }

    /// Moves `cursor_id`'s caret backward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_backward_at(&mut self, cursor_id: usize, amount: usize) -> Result<usize> {
        self.stream
            .move_cursor_backward(cursor_id, amount)
    }

    /// Moves the active caret to the next line; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_next_line(&mut self) -> Result<usize> {
        self.stream.move_cursor_next_line(self.cursor_id)
    }

    /// Moves the active caret to the previous line; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_prev_line(&mut self) -> Result<usize> {
        self.stream.move_cursor_prev_line(self.cursor_id)
    }

    /// Moves the active caret to `(row, col)`; returns the new position.
    #[inline]
    pub fn move_cursor_to(&mut self, row: usize, col: usize) -> Result<usize> {
        self.stream
            .move_cursor_to(self.cursor_id, row, col)
    }

    /// Preloads the active cursor's stack line ring around `row`.
    #[inline]
    pub fn request_region_load(&mut self, row: usize) -> Result<bool> {
        self.stream
            .request_region_load(self.cursor_id, row)
    }

    /// The active caret's row.
    #[inline]
    pub fn cursor_row(&self) -> Result<usize> {
        self.stream.cursor_row(self.cursor_id)
    }

    /// The active caret's column.
    #[inline]
    pub fn cursor_col(&self) -> Result<usize> {
        self.stream.cursor_col(self.cursor_id)
    }

    /// The width of the active caret's line.
    #[inline]
    pub fn cursor_width(&self) -> Result<usize> {
        self.stream.cursor_width(self.cursor_id)
    }

    /// Starts a selection at `position` on the active cursor.
    #[inline]
    pub fn set_selection_anchor(&mut self, position: usize) -> bool {
        self.stream
            .set_selection_anchor(self.cursor_id, position)
    }

    /// Extends the active cursor's selection to `position`.
    #[inline]
    pub fn set_selection_end(&mut self, position: usize) -> bool {
        self.stream
            .set_selection_end(self.cursor_id, position)
    }

    /// Deactivates the active cursor's selection.
    #[inline]
    pub fn clear_selection(&mut self) -> bool {
        self.stream.clear_selection(self.cursor_id)
    }

    /// The active cursor's non-empty selection range.
    #[inline]
    pub fn selection_range(&self) -> Option<(usize, usize)> {
        self.stream.selection_range(self.cursor_id)
    }

    /// The packed element at `position` in the committed document.
    #[inline]
    pub fn packed_at(&self, position: usize) -> Option<u32> {
        self.stream.packed_at(position)
    }

    /// The element type at `position`.
    #[inline]
    pub fn type_at(&self, position: usize) -> Option<u32> {
        self.stream.type_at(position)
    }

    /// The character at `position`.
    #[inline]
    pub fn char_at(&self, position: usize) -> Option<u32> {
        self.stream.char_at(position)
    }

    /// Width of the committed line containing `position`.
    #[inline]
    pub fn line_width_at(&self, position: usize) -> Option<usize> {
        self.stream.line_width_at(position)
    }

    /// The committed elements `[start, end)`.
    #[inline]
    pub fn raw_slice(&self, start: usize, end: usize) -> Option<&[u32]> {
        self.stream.raw_slice(start, end)
    }

    /// Makes `cursor_id` the active cursor. Returns `false` when the
    /// id is not live.
    #[inline]
    pub fn set_active_cursor(&mut self, cursor_id: usize) -> bool {
        let result = self.stream.set_active_cursor(cursor_id);
        if result {
            self.cursor_id = cursor_id;
        }
        result
    }

    /// Adds `cursor` to the pool; returns its id, or `None` when the
    /// fixed pool is full.
    #[inline]
    pub fn append_cursor(&mut self, cursor: Cursor<u32, u32, GAP_SIZE>) -> Option<usize> {
        self.stream.append_cursor(cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::COMMON_MASK;
    use crate::TextEditableEncoding;
    use crate::encoding::DocumentEncoding;
    use crate::encoding_dispatch::{EncodePolicy, encode, encode_from_str, encode_with_bom};
    use crate::fs::DiskFs;
    use crate::fs_block::MemoryDisk;
    use crate::make_packed;

    /// Concrete instantiation used by every test: 256-byte staging cache,
    /// 1024-element gap window.
    type TestBound<'a> = BoundDocument<'a, MemoryDisk, 256, 1024>;

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn create_and_save_round_trips() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let mut bound: TestBound<'_> = BoundDocument::create(&mut fs, 2, "hello.txt", 0x8000).unwrap();
        assert_eq!(bound.raw_length(), 0);
        assert!(!bound.is_dirty());

        for &byte in b"hello" {
            bound
                .put(make_packed(byte as u32, COMMON_MASK))
                .unwrap();
        }
        assert!(bound.is_dirty());
        bound.flush_all();
        assert!(bound.is_dirty());

        bound.save().unwrap();
        assert!(!bound.is_dirty());

        // Read back through the storage layer: `bound` must be dropped
        // before `fs` can be borrowed again because it holds a mutable
        // borrow of the filesystem.
        let inode = {
            let saved = bound.inode();
            drop(bound);
            saved
        };
        let mut buf = [0u8; 5];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), 5);
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn open_loads_existing_content() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "existing.txt", 0x8000).unwrap();
        fs.write_at(inode, 0, b"stored data").unwrap();

        let bound: TestBound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
        assert_eq!(bound.raw_length(), 11);
        assert!(!bound.is_dirty());

        // Verify the content is readable through the edit buffer.
        let chars: Vec<u32> = bound
            .raw_slice(0, 11)
            .unwrap()
            .iter()
            .map(|&e| packed_char(e))
            .collect();
        let text: String = chars.iter().map(|&c| c as u8 as char).collect();
        assert_eq!(text, "stored data");
    }

    #[test]
    fn save_overwrites_previous_content() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "overwrite.txt", 0x8000).unwrap();
        fs.write_at(inode, 0, b"old content here")
            .unwrap();

        let mut bound: TestBound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
        bound.delete_range(0, 11).unwrap(); // delete "old content"
        let replacement: Vec<u32> = b"new"
            .iter()
            .map(|&b| make_packed(b as u32, COMMON_MASK))
            .collect();
        bound.replace_range(0, 0, &replacement).unwrap();
        bound.flush_all();
        let inode = {
            let saved = bound.inode();
            bound.save().unwrap();
            drop(bound);
            saved
        };
        let mut buf = [0u8; 3];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), 3);
        assert_eq!(&buf, b"new");
    }

    #[test]
    fn directory_inode_is_rejected() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let dir = fs.mkdir(2, "subdir").unwrap();
        assert!(matches!(
            BoundDocument::<MemoryDisk, 256, 1024>::open(&mut fs, dir),
            Err(crate::error::Error::IsADirectory { .. })
        ));
    }

    #[test]
    fn dirty_flag_tracks_edits_and_save() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "dirty.txt", 0x8000).unwrap();
        let mut bound: TestBound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
        assert!(!bound.is_dirty());

        bound
            .put(make_packed(b'x' as u32, COMMON_MASK))
            .unwrap();
        assert!(bound.is_dirty());

        bound.save().unwrap();
        assert!(!bound.is_dirty());
    }

    #[test]
    fn load_larger_than_gap_window() {
        let disk = MemoryDisk::new(512).unwrap();
        let mut fs = DiskFs::format(disk, 512, 128).unwrap();
        let inode = fs.create(2, "big.txt", 0x8000).unwrap();

        // Write more content than one gap window (GAP_SIZE elements).
        let content = [0x65u8; 3000]; // 'e' repeated
        fs.write_at(inode, 0, &content).unwrap();

        let mut bound: TestBound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
        assert_eq!(bound.raw_length(), 3000);
        assert!(!bound.is_dirty());

        let first = bound.char_at(0).unwrap();
        let last = bound.char_at(2999).unwrap();
        assert_eq!(first, packed_char(make_packed(b'e' as u32, COMMON_MASK)));
        assert_eq!(last, packed_char(make_packed(b'e' as u32, COMMON_MASK)));

        bound.save().unwrap();
        assert!(!bound.is_dirty());
    }

    #[test]
    fn utf8_bom_is_stripped_on_load_and_reemitted_on_save() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "bom.txt", 0x8000).unwrap();
        fs.write_at(inode, 0, &[0xEF, 0xBB, 0xBF, b'h', b'i'])
            .unwrap();

        let mut bound: TestBound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
        assert_eq!(bound.encoding(), DocumentEncoding::Utf8);
        assert!(bound.emits_bom());
        assert_eq!(bound.raw_length(), 2);
        assert_eq!(bound.stream().read_string(0, 2).unwrap(), "hi");

        bound.save().unwrap();
        drop(bound);

        let mut buf = [0u8; 5];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), 5);
        assert_eq!(&buf, &[0xEF, 0xBB, 0xBF, b'h', b'i']);
    }

    #[test]
    fn utf16_le_round_trips_through_a_reopen() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let mut bound: TestBound<'_> =
            BoundDocument::create_with_encoding(&mut fs, 2, "utf16.txt", 0x8000, DocumentEncoding::Utf16Le)
                .unwrap();
        assert_eq!(bound.encoding(), DocumentEncoding::Utf16Le);
        assert!(bound.emits_bom());

        let cursor = bound.cursor_id();
        bound
            .stream_mut()
            .insert_str(cursor, "hé😀")
            .unwrap();
        bound.save().unwrap();
        let inode = bound.inode();
        drop(bound);

        let scalars = [u32::from('h'), u32::from('é'), u32::from('😀')];
        let expected = encode_with_bom(&scalars, DocumentEncoding::Utf16Le, EncodePolicy::Strict).unwrap();
        let mut buf = vec![0u8; expected.len()];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), expected.len());
        assert_eq!(buf, expected);

        let bound: TestBound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
        assert_eq!(bound.encoding(), DocumentEncoding::Utf16Le);
        assert!(bound.emits_bom());
        assert_eq!(bound.raw_length(), 3);
        assert_eq!(bound.stream().read_string(0, 3).unwrap(), "hé😀");
    }

    #[test]
    fn saving_a_shorter_document_truncates_the_file() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let mut bound: TestBound<'_> =
            BoundDocument::create_with_encoding(&mut fs, 2, "shrink.txt", 0x8000, DocumentEncoding::Utf16Le)
                .unwrap();
        let cursor = bound.cursor_id();
        bound
            .stream_mut()
            .insert_str(cursor, "abcdef")
            .unwrap();
        bound.save().unwrap();

        let inode = bound.inode();
        let long_size = bound.loaded_size();
        bound.delete_range(0, 5).unwrap();
        bound.save().unwrap();
        assert_eq!(bound.loaded_size(), 4); // BOM + one UTF-16 unit
        assert!(long_size > bound.loaded_size());
        drop(bound);

        // The stale tail is gone: reading past the new size returns 0.
        let mut tail = [0u8; 4];
        assert_eq!(fs.read_at(inode, 4, &mut tail).unwrap(), 0);
    }

    #[test]
    fn bomless_utf16_needs_an_explicit_encoding() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "bomless.txt", 0x8000).unwrap();
        let bytes = encode(
            &[u32::from('o'), u32::from('k')],
            DocumentEncoding::Utf16Le,
            EncodePolicy::Strict,
        )
        .unwrap();
        fs.write_at(inode, 0, &bytes).unwrap();

        let bound: TestBound<'_> =
            BoundDocument::open_with_encoding(&mut fs, inode, DocumentEncoding::Utf16Le).unwrap();
        assert_eq!(bound.encoding(), DocumentEncoding::Utf16Le);
        assert!(!bound.emits_bom());
        assert_eq!(bound.stream().read_string(0, 2).unwrap(), "ok");
    }

    #[test]
    fn single_byte_encoding_round_trips() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let encoding = DocumentEncoding::for_label("windows-1252").unwrap();
        let mut bound: TestBound<'_> =
            BoundDocument::create_with_encoding(&mut fs, 2, "latin.txt", 0x8000, encoding).unwrap();
        assert!(!bound.emits_bom());

        let text = "café €";
        let cursor = bound.cursor_id();
        bound
            .stream_mut()
            .insert_str(cursor, text)
            .unwrap();
        bound.save().unwrap();
        let inode = bound.inode();
        let size = bound.loaded_size() as usize;
        drop(bound);

        let expected = encode_from_str(text, encoding, EncodePolicy::Strict).unwrap();
        let mut buf = vec![0u8; size];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), size);
        assert_eq!(buf, expected);

        let bound: TestBound<'_> = BoundDocument::open_with_encoding(&mut fs, inode, encoding).unwrap();
        let length = bound.raw_length();
        assert_eq!(bound.stream().read_string(0, length).unwrap(), text);
    }

    #[test]
    fn save_rejects_unmappable_scalars_without_writing() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let encoding = DocumentEncoding::for_label("windows-1252").unwrap();
        let mut bound: TestBound<'_> =
            BoundDocument::create_with_encoding(&mut fs, 2, "strict.txt", 0x8000, encoding).unwrap();
        let cursor = bound.cursor_id();
        bound
            .stream_mut()
            .insert_str(cursor, "☃")
            .unwrap();

        assert!(matches!(
            bound.save(),
            Err(crate::error::Error::UnmappableCharacter { character: '☃' })
        ));
        assert_eq!(bound.loaded_size(), 0);
        assert!(bound.is_dirty());

        let inode = bound.inode();
        drop(bound);
        let mut buf = [0u8; 1];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), 0);
    }
}
