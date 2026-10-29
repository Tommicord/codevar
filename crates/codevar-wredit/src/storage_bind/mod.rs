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
//! The byte ↔ packed-element conversion is one byte per element: each
//! byte is stored as [`make_packed`]`(byte, [`COMMON_MASK`])`, so the
//! bound document is a byte-oriented view over the packed-element
//! engine.
//!
//! # Example
//!
//! ```rust
//! use codevar_wredit::error::WreditResult;
//! use codevar_wredit::edit::{COMMON_MASK, make_packed};
//! use codevar_wredit::storage::block::MemoryDisk;
//! use codevar_wredit::storage::fs::DiskFs;
//! use codevar_wredit::storage_bind::BoundDocument;
//!
//! fn edit_stored_file() -> WreditResult<()> {
//!     let disk = MemoryDisk::new(256)?;
//!     let mut fs = DiskFs::format(disk, 256, 128)?;
//!     let inode = fs.create(2, "notes.txt", 0x8000)?;
//!     let mut bound: BoundDocument<'_, MemoryDisk, 256, 1024> =
//!         BoundDocument::open(&mut fs, inode)?;
//!     bound.put(make_packed(b'X' as u32, COMMON_MASK))?;
//!     bound.save()?;
//!     Ok(())
//! }
//! ```

use alloc::vec::Vec;

use codevar_logger::log_info;

use crate::edit::{
    COMMON_MASK, Cursor, StreamWritable, TextEditableBufferQuery, TextEditableCursor, TextEditableDelete,
    TextEditableOverwrite, TextEditablePut, TextEditableReplace, TextEditableSelection, Writable,
    make_packed, packed_char,
};
use crate::error::{WreditError, WreditResult};
use crate::storage::block::BlockDevice;
use crate::storage::fs::DiskFs;
use crate::storage::writable::WritableFile;

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
    /// Staged gaps auto-commit as they fill, so files larger than one
    /// gap window load correctly.
    ///
    /// # Errors
    ///
    /// * [`WreditError::IsADirectory`] when `inode` is not a regular file
    /// * [`WreditError::NotFound`] when `inode` does not exist
    /// * Propagates device I/O errors from reading the file
    /// * [`WreditError::CapacityExhausted`] when the edit buffer cannot
    ///   grow to hold the file contents
    pub fn open(fs: &'fs mut DiskFs<D>, inode: u32) -> WreditResult<Self> {
        let ino = fs.read_inode(inode)?;
        if !ino.is_regular() {
            return Err(WreditError::IsADirectory {
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

        let mut stream = StreamWritable::with_capacity(content.len());
        let cursor_id = stream
            .append_cursor(Cursor::new())
            .ok_or(WreditError::LimitExceeded { what: "cursor pool" })?;
        stream.set_active_cursor(cursor_id);

        for &byte in &content {
            stream.put(cursor_id, make_packed(byte as u32, COMMON_MASK))?;
        }
        stream.flush_all();

        log_info!("storage_bind: loaded inode {inode} ({loaded_size} bytes) into edit buffer");

        Ok(Self {
            file,
            inode,
            stream,
            loaded_size,
            dirty: false,
            cursor_id,
        })
    }

    /// Creates a new empty regular file and binds it for editing.
    ///
    /// # Errors
    ///
    /// Propagates filesystem errors from creating the file and the
    /// writable file, and [`WreditError::LimitExceeded`] when the
    /// cursor pool is full.
    pub fn create(fs: &'fs mut DiskFs<D>, parent: u32, name: &str, mode: u16) -> WreditResult<Self> {
        let inode = fs.create(parent, name, mode)?;
        let file = WritableFile::new(fs, inode)?;
        let mut stream = StreamWritable::new();
        let cursor_id = stream
            .append_cursor(Cursor::new())
            .ok_or(WreditError::LimitExceeded { what: "cursor pool" })?;
        stream.set_active_cursor(cursor_id);

        log_info!("storage_bind: created inode {inode} named {name:?} for editing");

        Ok(Self {
            file,
            inode,
            stream,
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
    /// Staged gaps are committed first, then the document is written
    /// from position 0 through [`WritableFile`] and synced to the
    /// block device. The file size only grows: if the edited document
    /// is shorter than the loaded file, the old tail remains readable
    /// until the storage layer supports truncation.
    ///
    /// # Errors
    ///
    /// Propagates device I/O errors from writing and syncing.
    pub fn save(&mut self) -> WreditResult<()> {
        self.flush_all();

        let length = self.stream.raw_length();
        let elements = self.stream.raw_slice(0, length).unwrap_or(&[]);
        let mut bytes = Vec::with_capacity(elements.len());
        for &element in elements {
            bytes.push(packed_char(element) as u8);
        }

        self.file.set_position(0)?;
        self.file.write(&bytes)?;
        self.file.sync_all()?;

        self.loaded_size = bytes.len() as u64;
        self.dirty = false;

        log_info!("storage_bind: saved inode {} ({} bytes)", self.inode, bytes.len());

        Ok(())
    }

    /// Commits staged edits and saves to disk in one step.
    ///
    /// Equivalent to [`Self::flush_all`] followed by [`Self::save`].
    ///
    /// # Errors
    ///
    /// Propagates device I/O errors from saving.
    pub fn commit_and_save(&mut self) -> WreditResult<()> {
        self.save()
    }

    // --- Delegated editing operations ---

    /// Stages `value` in the active cursor's private gap and advances
    /// the caret. Returns the caret's new logical position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn put(&mut self, value: u32) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.put(self.cursor_id, value)
    }

    /// Stages `value` in the cursor with `cursor_id`'s private gap and
    /// advances the caret. Returns the caret's new logical position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn put_at(&mut self, cursor_id: usize, value: u32) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.put(cursor_id, value)
    }

    /// Stages `value` in a shared gap the active cursor belongs to and
    /// advances the caret. Returns the new logical position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn put_shared(&mut self, value: u32) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.put_shared(self.cursor_id, value)
    }

    /// Deletes the element at the active caret (forward delete).
    /// Returns the number of removed elements.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn delete(&mut self) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.delete(self.cursor_id)
    }

    /// Deletes the element at `cursor_id`'s caret (forward delete).
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn delete_at(&mut self, cursor_id: usize) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.delete(cursor_id)
    }

    /// Deletes the element before the active caret (backspace).
    /// Returns the number of removed elements.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn backspace(&mut self) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.backspace(self.cursor_id)
    }

    /// Deletes the committed half-open range `[start, end)`.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn delete_range(&mut self, start: usize, end: usize) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.delete_range(start, end)
    }

    /// Deletes the element at the active caret and stages `value` in
    /// its place. Returns the new logical caret position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn replace(&mut self, value: u32) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.replace(self.cursor_id, value)
    }

    /// Replaces the committed range `[start, end)` with `values`,
    /// committed immediately.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn replace_range(&mut self, start: usize, end: usize, values: &[u32]) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.replace_range(start, end, values)
    }

    /// Stages a merge element that overwrites the element at the active
    /// caret on flush (or appends when the caret is at the end).
    /// Returns the new logical caret position.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn overwrite(&mut self, value: u32) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.overwrite(self.cursor_id, value)
    }

    /// Overwrites committed elements in place starting at `start`;
    /// `values` extending past the document end are appended.
    ///
    /// Marks the document dirty.
    #[inline]
    pub fn overwrite_slice(&mut self, start: usize, values: &[u32]) -> WreditResult<usize> {
        self.dirty = true;
        self.stream.overwrite_slice(start, values)
    }

    /// Moves the active caret forward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_forward(&mut self, amount: usize) -> WreditResult<usize> {
        self.stream
            .move_cursor_forward(self.cursor_id, amount)
    }

    /// Moves `cursor_id`'s caret forward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_forward_at(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize> {
        self.stream.move_cursor_forward(cursor_id, amount)
    }

    /// Moves the active caret backward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_backward(&mut self, amount: usize) -> WreditResult<usize> {
        self.stream
            .move_cursor_backward(self.cursor_id, amount)
    }

    /// Moves `cursor_id`'s caret backward by `amount`; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_backward_at(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize> {
        self.stream
            .move_cursor_backward(cursor_id, amount)
    }

    /// Moves the active caret to the next line; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_next_line(&mut self) -> WreditResult<usize> {
        self.stream.move_cursor_next_line(self.cursor_id)
    }

    /// Moves the active caret to the previous line; returns the new
    /// position.
    #[inline]
    pub fn move_cursor_prev_line(&mut self) -> WreditResult<usize> {
        self.stream.move_cursor_prev_line(self.cursor_id)
    }

    /// Moves the active caret to `(row, col)`; returns the new position.
    #[inline]
    pub fn move_cursor_to(&mut self, row: usize, col: usize) -> WreditResult<usize> {
        self.stream
            .move_cursor_to(self.cursor_id, row, col)
    }

    /// Preloads the active cursor's stack line ring around `row`.
    #[inline]
    pub fn request_region_load(&mut self, row: usize) -> WreditResult<bool> {
        self.stream
            .request_region_load(self.cursor_id, row)
    }

    /// The active caret's row.
    #[inline]
    pub fn cursor_row(&self) -> WreditResult<usize> {
        self.stream.cursor_row(self.cursor_id)
    }

    /// The active caret's column.
    #[inline]
    pub fn cursor_col(&self) -> WreditResult<usize> {
        self.stream.cursor_col(self.cursor_id)
    }

    /// The width of the active caret's line.
    #[inline]
    pub fn cursor_width(&self) -> WreditResult<usize> {
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

    use crate::edit::COMMON_MASK;
    use crate::storage::block::MemoryDisk;
    use crate::storage::fs::DiskFs;

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
            Err(crate::error::WreditError::IsADirectory { .. })
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
}
