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

//! Operation traits of the edit engine.
//!
//! [`Writable`] is the structural contract every writable fulfils:
//! cursors, raw views, metadata, events and the shared gap table.
//! The `TextEditable*` traits then split the editing surface into
//! small, composable groups — put, delete, replace, overwrite, gap
//! management, cursor movement, selection and buffer queries — so a
//! frontend can depend on the smallest surface it needs.
//!
//! Fallible operations report [`WreditResult`]; invalid cursor ids
//! and out-of-range positions are rejected with an error (and a log
//! line), never a panic.

use alloc::sync::Arc;

use super::base::NAME_LEN;
use super::cursor::Cursor;
use super::shared_gap::SharedGapTable;
use super::writable::{EventBridge, SHARED_GAP_SLOTS};
use crate::error::WreditResult;

/// The structural contract of a multi-cursor text writable.
pub trait Writable<Raw, Buf, const GAP_SIZE: usize>: Sized {
    /// Creates an empty writable (no allocation).
    fn new() -> Self;

    /// Creates a writable with pre-grown raw capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Adds `cursor` to the pool; returns its id, or `None` when the
    /// fixed pool ([`super::writable::MAX_CURSORS`]) is full.
    fn append_cursor(&mut self, cursor: Cursor<Raw, Buf, GAP_SIZE>) -> Option<usize>;

    /// Removes the cursor with `cursor_id`, releasing any gap it
    /// owned. Returns `false` when the id was not live.
    fn delete_cursor(&mut self, cursor_id: usize) -> bool;

    /// The live cursor with `cursor_id`.
    fn cursor_from_id(&self, cursor_id: usize) -> Option<&Cursor<Raw, Buf, GAP_SIZE>>;

    /// A mutable pointer to the live cursor with `cursor_id`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the engine's single-thread contract:
    /// no other borrow of the writable (shared or mutable) may be
    /// live while the returned pointer is used, and the cursor must
    /// not outlive the writable.
    unsafe fn cursor_from_id_mut(&self, cursor_id: usize) -> Option<*mut Cursor<Raw, Buf, GAP_SIZE>>;

    /// Committed capacity head-room above the raw document length.
    fn v_space(&self) -> usize;

    /// The fixed shared gap descriptor table.
    fn shared_gaps(&self) -> &SharedGapTable<SHARED_GAP_SLOTS>;

    /// Id of the cursor driving single-cursor operations.
    fn active_cursor_id(&self) -> usize;

    /// Makes `cursor_id` the active cursor. Returns `false` when the
    /// id is not live.
    fn set_active_cursor(&mut self, cursor_id: usize) -> bool;

    /// The active cursor.
    fn active_cursor(&self) -> Option<&Cursor<Raw, Buf, GAP_SIZE>>;

    /// Read-only pointer to the committed raw document.
    fn raw_ptr(&self) -> *const Raw;

    /// Mutable pointer to the committed raw document.
    ///
    /// # Safety
    ///
    /// The caller must uphold the engine's single-thread contract
    /// and must only write elements that keep the packed document
    /// well-formed; the writable must not be mutably borrowed while
    /// the pointer is used.
    unsafe fn raw_mut_ptr(&self) -> *mut Raw;

    /// Number of committed elements in the raw document.
    fn raw_length(&self) -> usize;

    /// Installs (or removes, with `None`) the event bridge.
    fn set_event_bridge(&mut self, bridge: Option<Arc<dyn EventBridge>>);

    /// The installed event bridge, cloned out.
    fn event_bridge(&self) -> Option<Arc<dyn EventBridge>>;

    /// The document name as UTF-8 (lossy fallback on invalid bytes).
    fn name(&self) -> &str;

    /// The raw name bytes, NUL padded, for in-place editing.
    fn name_mut(&mut self) -> &mut [u8; NAME_LEN];

    /// User-defined flag bits.
    fn flags(&self) -> u32;

    /// Mutable user-defined flag bits.
    fn flags_mut(&mut self) -> &mut u32;

    /// Index of the gap (private index or [`MAX_GAPS`] + shared slot)
    /// whose commit base is closest to `pos`.
    ///
    /// # Safety
    ///
    /// Requires the caller's single-thread contract: the shared gap
    /// table is read through the interior-mutable cell.
    unsafe fn proximity(&self, pos: usize) -> Option<usize>;

    /// Allocates a shared gap slot for `cursor_mask`.
    ///
    /// # Safety
    ///
    /// Requires the caller's single-thread contract; the mask must
    /// only name live cursor ids (ids `>= 16` never fit a mask).
    unsafe fn new_shared_gap(&mut self, cursor_mask: u16) -> Option<usize>;

    /// The shared gap slot currently including `cursor_id`, if any.
    ///
    /// # Safety
    ///
    /// Requires the caller's single-thread contract.
    unsafe fn shared_gap_index(&self, cursor_id: usize) -> Option<usize>;

    /// `true` when the shared gap at `gap_index` includes `cursor_id`.
    ///
    /// # Safety
    ///
    /// Requires the caller's single-thread contract.
    unsafe fn is_cursor_in_shared_gap(&self, gap_index: usize, cursor_id: usize) -> bool;

    /// Selects the whole logical document with `cursor_id`.
    ///
    /// # Safety
    ///
    /// Requires the caller's single-thread contract; returns `false`
    /// when the cursor is not live.
    unsafe fn select_all(&self, cursor_id: usize) -> bool;
}

/// Inserting staged text at a cursor.
pub trait TextEditablePut<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Stages `value` in the cursor's private gap and advances the
    /// caret. Returns the caret's new logical position.
    fn put(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize>;

    /// Stages `value` in a *shared* gap the cursor belongs to and
    /// advances the caret. Returns the new logical position.
    fn put_shared(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize>;
}

/// Deleting committed text.
pub trait TextEditableDelete<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Deletes the element at the caret (forward delete). Returns
    /// the number of removed elements (0 at the document end).
    fn delete(&mut self, cursor_id: usize) -> WreditResult<usize>;

    /// Deletes the element before the caret (backspace). Returns the
    /// number of removed elements (0 at the document start).
    fn backspace(&mut self, cursor_id: usize) -> WreditResult<usize>;

    /// Deletes the committed half-open range `[start, end)`.
    fn delete_range(&mut self, start: usize, end: usize) -> WreditResult<usize>;
}

/// Replacing committed text.
pub trait TextEditableReplace<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Deletes the element at the caret and stages `value` in its
    /// place. Returns the new logical caret position.
    fn replace(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize>;

    /// Replaces the committed range `[start, end)` with `values`,
    /// committed immediately.
    fn replace_range(&mut self, start: usize, end: usize, values: &[Raw]) -> WreditResult<usize>;
}

/// Overwriting committed text in place.
pub trait TextEditableOverwrite<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Stages a merge element that overwrites the element at the
    /// caret on flush (or appends when the caret is at the end).
    fn overwrite(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize>;

    /// Overwrites committed elements in place starting at `start`;
    /// `values` extending past the document end are appended.
    fn overwrite_slice(&mut self, start: usize, values: &[Raw]) -> WreditResult<usize>;
}

/// Gap (flush and block shift) management.
pub trait TextEditableGap<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Commits the gap at `gap_index` (`0..MAX_GAPS` private,
    /// `MAX_GAPS + slot` shared). Returns the committed element count.
    fn flush_gap(&mut self, gap_index: usize) -> usize;

    /// Commits every gap. Returns the committed element count.
    fn flush_all(&mut self) -> usize;

    /// Shifts the gap's commit base by a signed `distance`.
    ///
    /// # Safety
    ///
    /// The caller must only shift gaps it knows to be allocated and
    /// must uphold the engine's single-thread contract; bases are
    /// clamped into the logical document so the shift cannot escape
    /// it.
    unsafe fn shift_gap_block(&mut self, gap_index: usize, distance: isize) -> usize;

    /// Shifts the gap's commit base right by `distance`.
    ///
    /// # Safety
    ///
    /// Same contract as [`TextEditableGap::shift_gap_block`].
    unsafe fn shift_right_gap_block(&mut self, gap_index: usize, distance: usize) -> usize;

    /// Shifts the gap's commit base left by `distance`.
    ///
    /// # Safety
    ///
    /// Same contract as [`TextEditableGap::shift_gap_block`].
    unsafe fn shift_left_gap_block(&mut self, gap_index: usize, distance: usize) -> usize;
}

/// Caret movement.
pub trait TextEditableCursor<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Moves the caret forward by `amount`; returns the new position.
    fn move_cursor_forward(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize>;

    /// Moves the caret backward by `amount`; returns the new position.
    fn move_cursor_backward(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize>;

    /// Moves the caret to the next line; returns the new position.
    fn move_cursor_next_line(&mut self, cursor_id: usize) -> WreditResult<usize>;

    /// Moves the caret to the previous line; returns the new position.
    fn move_cursor_prev_line(&mut self, cursor_id: usize) -> WreditResult<usize>;

    /// Moves the caret to `(row, col)`; returns the new position.
    fn move_cursor_to(&mut self, cursor_id: usize, row: usize, col: usize) -> WreditResult<usize>;

    /// Preloads the cursor's stack line ring around `row`.
    ///
    /// A cold random row jump has to scan the committed document to
    /// find the row; doing that scan once with this call overwrites the
    /// current ring and makes the following row/column work in the
    /// region constant time. Returns `false` for a document below the
    /// cache threshold or a row past the last committed line.
    fn request_region_load(&mut self, cursor_id: usize, row: usize) -> WreditResult<bool>;

    /// The caret's row.
    fn cursor_row(&self, cursor_id: usize) -> WreditResult<usize>;

    /// The caret's column.
    fn cursor_col(&self, cursor_id: usize) -> WreditResult<usize>;

    /// The width of the caret's line.
    fn cursor_width(&self, cursor_id: usize) -> WreditResult<usize>;
}

/// Selection management.
pub trait TextEditableSelection<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// Starts a selection at `position`; returns `false` for a dead
    /// cursor id.
    fn set_selection_anchor(&mut self, cursor_id: usize, position: usize) -> bool;

    /// Extends the selection to `position`.
    fn set_selection_end(&mut self, cursor_id: usize, position: usize) -> bool;

    /// Deactivates the cursor's selection.
    fn clear_selection(&mut self, cursor_id: usize) -> bool;

    /// The active, non-empty selection range.
    fn selection_range(&self, cursor_id: usize) -> Option<(usize, usize)>;
}

/// Read-only views over the committed raw document.
pub trait TextEditableBufferQuery<Raw, Buf, const GAP_SIZE: usize>: Writable<Raw, Buf, GAP_SIZE> {
    /// The committed elements `[start, end)`.
    fn raw_slice(&self, start: usize, end: usize) -> Option<&[Raw]>;

    /// The packed element at `position`.
    fn packed_at(&self, position: usize) -> Option<Raw>;

    /// The element type at `position`.
    fn type_at(&self, position: usize) -> Option<u32>;

    /// The character at `position`.
    fn char_at(&self, position: usize) -> Option<u32>;

    /// Width of the committed line containing `position`.
    fn line_width_at(&self, position: usize) -> Option<usize>;
}
