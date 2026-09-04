//! Copyright 2026 Codevar
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

/// Trait for inserting characters into a writable buffer.
///
/// This trait provides the fundamental operation of inserting characters at cursor
/// positions within a gap-buffer based text editing system. It handles both regular
/// gap buffers and shared gap suballocations for multi-cursor scenarios.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage (typically u32 for packed character/type data)
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer, must be a power of two
pub trait TextEditablePut<Raw, Buf, const GAP_SIZE: usize> {
    /// Inserts a character at the specified cursor position.
    ///
    /// This method writes a character to the buffer at the current position of the
    /// specified cursor. It handles both regular gap buffers and shared gap
    /// suballocations automatically. For newline characters (0x0A), it updates
    /// the vertical space counter and moves the cursor to the next line.
    ///
    /// # Arguments
    ///
    /// * `ch` - The character code to insert (typically a Unicode code point or packed value)
    /// * `points_new` - If true, advances the cursor forward after insertion; if false,
    ///   keeps the cursor at the insertion point (useful for overwriting)
    /// * `cursor_id` - The ID of the cursor to use for insertion. Must be a valid cursor ID
    ///
    /// # Safety
    ///
    /// This method is unsafe because:
    /// - The cursor_id must correspond to a valid cursor
    /// - The gap buffer must be properly allocated
    /// - When using shared gaps, the cursor must be in a valid region
    /// - The method performs raw pointer operations
    ///
    /// # Behavior
    ///
    /// - For newline characters (0x0A): increments vertical space counter and moves cursor to next line
    /// - For regular characters: writes to gap buffer, handles gap overflow by shifting blocks
    /// - When cursor is in shared gap: uses suballocated region, handles region movement if full
    /// - Automatically flushes gap buffer when it becomes full
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut writable = StreamWritable::new();
    /// let cursor_id = writable.base_mut().append_cursor();
    ///
    /// unsafe {
    ///     // Insert 'A' and advance cursor
    ///     writable.put(0x41, true, cursor_id);
    ///
    ///     // Insert newline
    ///     writable.put(0x0A, true, cursor_id);
    /// }
    /// ```
    unsafe fn put(&mut self, ch: u32, points_new: bool, cursor_id: usize);

    /// Puts a character using shared gap suballocation.
    ///
    /// This method handles character insertion when the cursor is in a shared gap.
    /// It uses the cursor's suballocated region within the shared gap buffer,
    /// checking remaining space and moving the region if needed.
    ///
    /// # Arguments
    ///
    /// * `ch` - The character to write
    /// * `points_new` - Whether this is a new point
    /// * `cursor_id` - The cursor ID to use
    ///
    /// # Safety
    ///
    /// The cursor must be valid and must be in a shared gap

    unsafe fn put_shared(&mut self, ch: u32, points_new: bool, cursor_id: usize);
}

/// Trait for deleting characters from a writable buffer.
///
/// This trait provides operations for removing characters from the buffer at cursor
/// positions. Deletion operations can work in both forward and backward directions,
/// and can delete single characters or ranges.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableDelete<Raw, Buf, const GAP_SIZE: usize> {
    /// Deletes the character at the current cursor position (forward deletion).
    ///
    /// Removes the character immediately following the cursor position, effectively
    /// deleting the character "in front of" the cursor. This is the standard delete
    /// operation in most text editors (Delete key).
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to use for deletion
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and the cursor must not be at the end of the buffer.
    unsafe fn delete(&mut self, cursor_id: usize);

    /// Deletes the character before the current cursor position (backward deletion).
    ///
    /// Removes the character immediately preceding the cursor position and moves
    /// the cursor backward. This is the standard backspace operation in most
    /// text editors (Backspace key).
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to use for deletion
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and the cursor must not be at the start of the buffer.
    unsafe fn backspace(&mut self, cursor_id: usize);

    /// Deletes a range of characters starting from the current cursor position.
    ///
    /// Removes `count` characters starting from the character immediately following
    /// the cursor position. This is useful for batch deletion operations.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to use for deletion
    /// * `count` - The number of characters to delete
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and there must be at least `count` characters
    /// available for deletion.
    unsafe fn delete_range(&mut self, cursor_id: usize, count: usize);
}

/// Trait for replacing characters in a writable buffer.
///
/// This trait provides operations for replacing existing characters with new ones
/// without changing the overall buffer structure. Replace operations are useful for
/// text transformation and correction operations.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableReplace<Raw, Buf, const GAP_SIZE: usize> {
    /// Replaces the character at the current cursor position.
    ///
    /// Overwrites the character at the cursor position with a new character without
    /// changing the buffer size or cursor position. This is useful for character-by-character
    /// corrections.
    ///
    /// # Arguments
    ///
    /// * `ch` - The new character to write
    /// * `cursor_id` - The ID of the cursor to use for replacement
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and the cursor must not be at the end of the buffer.
    unsafe fn replace(&mut self, ch: u32, cursor_id: usize);

    /// Replaces a range of characters starting from the current cursor position.
    ///
    /// Replaces `count` characters starting from the cursor position with characters
    /// from the provided slice. If the slice length differs from count, the buffer
    /// size will be adjusted accordingly.
    ///
    /// # Arguments
    ///
    /// * `chars` - Slice of new characters to write
    /// * `cursor_id` - The ID of the cursor to use for replacement
    /// * `count` - The number of characters to replace
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and there must be at least `count` characters
    /// available for replacement.
    unsafe fn replace_range(&mut self, chars: &[u32], cursor_id: usize, count: usize);
}

/// Trait for overwriting characters in a writable buffer.
///
/// This trait provides operations for overwriting characters without advancing the
/// cursor, which is useful for in-place editing operations where the cursor position
/// should remain fixed.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableOverwrite<Raw, Buf, const GAP_SIZE: usize> {
    /// Overwrites the character at the current cursor position without advancing.
    ///
    /// Writes a character at the cursor position but does not advance the cursor,
    /// allowing subsequent writes to continue at the same position. This is useful
    /// for fixed-position editing scenarios.
    ///
    /// # Arguments
    ///
    /// * `ch` - The character to write
    /// * `cursor_id` - The ID of the cursor to use for overwriting
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and the cursor must not be at the end of the buffer.
    unsafe fn overwrite(&mut self, ch: u32, cursor_id: usize);

    /// Overwrites multiple characters at the current cursor position without advancing.
    ///
    /// Writes characters from the provided slice starting at the cursor position
    /// without advancing the cursor. The cursor remains at its original position.
    ///
    /// # Arguments
    ///
    /// * `chars` - Slice of characters to write
    /// * `cursor_id` - The ID of the cursor to use for overwriting
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and there must be sufficient space in the buffer.
    unsafe fn overwrite_slice(&mut self, chars: &[u32], cursor_id: usize);
}

/// Trait for gap buffer management operations.
///
/// This trait provides low-level operations for managing gap buffers, including
/// flushing, shifting, and moving gap blocks. These operations are essential for
/// maintaining the gap buffer structure during editing operations.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableGap<Raw, Buf, const GAP_SIZE: usize> {
    /// Flushes the gap buffer for the specified cursor.
    ///
    /// Writes the contents of the gap buffer associated with the cursor to the
    /// main data buffer and resets the gap. This is typically called when a gap
    /// becomes full or when explicit synchronization is needed.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor whose gap should be flushed
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and the gap buffer must be properly allocated.
    unsafe fn flush_gap(&mut self, cursor_id: usize);

    /// Flushes all gap buffers for all cursors.
    ///
    /// Writes the contents of all gap buffers to the main data buffer and resets
    /// all gaps. This is useful for global synchronization operations.
    ///
    /// # Returns
    ///
    /// The total number of characters flushed across all gaps.
    ///
    /// # Safety
    ///
    /// All cursors must remain valid for the duration of the flush operation.
    unsafe fn flush_all(&mut self) -> usize;

    /// Shifts a gap block by the specified offset and size.
    ///
    /// Moves a contiguous block of data within the gap buffer by copying it to
    /// a new location. This is used internally to manage gap buffer space during
    /// insertions and deletions.
    ///
    /// # Arguments
    ///
    /// * `gap_id` - The ID of the gap containing the block
    /// * `offset` - The offset within the gap where the block starts
    /// * `size` - The size of the block to shift
    ///
    /// # Safety
    ///
    /// The gap_id must be valid and the offset/size must be within gap bounds.
    unsafe fn shift_gap_block(&mut self, gap_id: usize, offset: usize, size: usize);

    /// Shifts a gap block to the right.
    ///
    /// Convenience method that shifts a gap block starting from offset 0.
    ///
    /// # Arguments
    ///
    /// * `gap_id` - The ID of the gap containing the block
    /// * `count` - The number of positions to shift right
    ///
    /// # Safety
    ///
    /// The gap_id must be valid.
    unsafe fn shift_right_gap_block(&mut self, gap_id: usize, count: usize);

    /// Shifts a gap block to the left.
    ///
    /// Convenience method that shifts a gap block to offset 0.
    ///
    /// # Arguments
    ///
    /// * `gap_id` - The ID of the gap containing the block
    /// * `count` - The number of positions to shift left
    ///
    /// # Safety
    ///
    /// The gap_id must be valid.
    unsafe fn shift_left_gap_block(&mut self, gap_id: usize, count: usize);

    /// Moves a gap block to a new position.
    ///
    /// Similar to shift_gap_block but provides a more intuitive interface for
    /// moving blocks to specific positions.
    ///
    /// # Arguments
    ///
    /// * `gap_id` - The ID of the gap containing the block
    /// * `offset` - The new offset for the block
    /// * `size` - The size of the block to move
    ///
    /// # Safety
    ///
    /// The gap_id must be valid and the target position must be within bounds.
    unsafe fn move_gap_block(&mut self, gap_id: usize, offset: usize, size: usize);

    /// Sets the bounds of a gap buffer.
    ///
    /// Updates the start and end pointers of a gap buffer to define its active region.
    /// This is used to resize or reposition gaps during editing operations.
    ///
    /// # Arguments
    ///
    /// * `gap_id` - The ID of the gap to modify
    /// * `offset` - The offset from the start of the buffer for the gap start
    /// * `len` - The length of the gap
    ///
    /// # Safety
    ///
    /// The gap_id must be valid and the raw pointer must be valid for the specified range.
    unsafe fn set_gap_bounds(&mut self, gap_id: usize, offset: usize, len: usize);
}

/// Trait for cursor navigation operations.
///
/// This trait provides operations for moving cursors within the buffer, including
/// forward/backward movement, line navigation, and position queries.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableCursor<Raw, Buf, const GAP_SIZE: usize> {
    /// Moves the cursor forward by the specified number of characters.
    ///
    /// Advances the cursor position in the forward direction (toward the end of the buffer).
    /// The movement respects line boundaries and can cross multiple lines.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    /// * `count` - The number of characters to move forward
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and there must be sufficient characters to move.
    unsafe fn move_cursor_forward(&mut self, cursor_id: usize, count: usize);

    /// Moves the cursor backward by the specified number of characters.
    ///
    /// Moves the cursor position in the backward direction (toward the start of the buffer).
    /// The movement respects line boundaries and can cross multiple lines.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    /// * `count` - The number of characters to move backward
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and there must be sufficient characters to move backward.
    unsafe fn move_cursor_backward(&mut self, cursor_id: usize, count: usize);

    /// Moves the cursor to the next line.
    ///
    /// Moves the cursor to the start of the next line, preserving the column position
    /// if possible. This is typically used after newline insertion.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn move_cursor_next_line(&mut self, cursor_id: usize);

    /// Moves the cursor to the previous line.
    ///
    /// Moves the cursor to the start of the previous line, preserving the column
    /// position if possible.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn move_cursor_prev_line(&mut self, cursor_id: usize);

    /// Moves the cursor to the start of the current line.
    ///
    /// Positions the cursor at the first character of the current line.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn move_cursor_line_start(&mut self, cursor_id: usize);

    /// Moves the cursor to the end of the current line.
    ///
    /// Positions the cursor after the last character of the current line.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn move_cursor_line_end(&mut self, cursor_id: usize);

    /// Moves the cursor forward by one word.
    ///
    /// Advances the cursor to the start of the next word, where words are
    /// delimited by whitespace or punctuation.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn move_cursor_word_forward(&mut self, cursor_id: usize);

    /// Moves the cursor backward by one word.
    ///
    /// Moves the cursor to the start of the previous word, where words are
    /// delimited by whitespace or punctuation.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to move
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn move_cursor_word_backward(&mut self, cursor_id: usize);

    /// Gets the current cursor position as a (row, column) tuple.
    ///
    /// Returns the bidirectional index of the cursor, representing its position
    /// in terms of row (line number) and column (character position within line).
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to query
    ///
    /// # Returns
    ///
    /// A tuple of (row, column) representing the cursor position, or None if the cursor is invalid.
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn cursor_position(&self, cursor_id: usize) -> Option<(usize, usize)>;
}

/// Trait for buffer query operations.
///
/// This trait provides read-only operations for querying buffer state, including
/// character retrieval, line width calculations, and buffer statistics.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableQuery<Raw, Buf, const GAP_SIZE: usize> {
    /// Gets the character at the current cursor position.
    ///
    /// Returns the character immediately following the cursor position without
    /// modifying the buffer or cursor position.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to query
    ///
    /// # Returns
    ///
    /// The character code at the cursor position, or None if the cursor is at the end.
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn char_at_cursor(&mut self, cursor_id: usize) -> Option<u32>;

    /// Gets the width of the current line.
    ///
    /// Returns the number of characters in the line containing the cursor,
    /// excluding the newline character.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to query
    ///
    /// # Returns
    ///
    /// The width of the current line in characters, or 0 if the cursor is invalid.
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn line_width(&mut self, cursor_id: usize) -> usize;

    /// Gets the total number of lines in the buffer.
    ///
    /// Returns the vertical space count, which represents the number of newline
    /// characters encountered (plus 1 for the first line).
    ///
    /// # Returns
    ///
    /// The total number of lines in the buffer.
    fn line_count(&mut self) -> usize;

    /// Gets the total number of characters in the buffer.
    ///
    /// Returns the length of valid data in the buffer, excluding gap space.
    ///
    /// # Returns
    ///
    /// The total number of characters in the buffer.
    fn char_count(&mut self) -> usize;

    /// Checks if the buffer is empty.
    ///
    /// Returns true if the buffer contains no valid data.
    ///
    /// # Returns
    ///
    /// True if the buffer is empty, false otherwise.
    fn is_empty(&self) -> bool;
}

/// Trait for selection operations.
///
/// This trait provides operations for managing text selections, including creating,
/// extending, and manipulating selected ranges of text.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableSelection<Raw, Buf, const GAP_SIZE: usize> {
    /// Creates a selection from the current cursor position.
    ///
    /// Starts a selection at the current cursor position. Subsequent cursor movements
    /// will extend the selection.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to use for selection
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn start_selection(&mut self, cursor_id: usize);

    /// Extends the selection to the current cursor position.
    ///
    /// Updates the selection end point to the current cursor position.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to use for selection
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and a selection must have been started.
    unsafe fn extend_selection(&mut self, cursor_id: usize);

    /// Clears the current selection.
    ///
    /// Removes the selection without modifying the buffer contents.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor whose selection should be cleared
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn clear_selection(&mut self, cursor_id: usize);

    /// Deletes the currently selected text.
    ///
    /// Removes the text within the selection range and clears the selection.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor whose selection should be deleted
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid and a selection must exist.
    unsafe fn delete_selection(&mut self, cursor_id: usize);

    /// Gets the currently selected text range.
    ///
    /// Returns the start and end positions of the selection as (start, end).
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to query
    ///
    /// # Returns
    ///
    /// A tuple of (start, end) positions, or None if no selection exists.
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    unsafe fn selection_range(&self, cursor_id: usize) -> Option<(usize, usize)>;
}

/// Trait for undo/redo operations.
///
/// This trait provides operations for managing edit history, allowing undo and
/// redo of text editing operations for a production-ready editing experience.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type used for storage
/// * `Buf` - The buffer type for decoded character data
/// * `GAP_SIZE` - The size of the gap buffer
pub trait TextEditableHistory<Raw, Buf, const GAP_SIZE: usize> {
    /// Begins a new edit group for undo/redo.
    ///
    /// Starts a new group of related edits that should be undone/redone together.
    /// This is useful for compound operations like multi-character insertions.
    ///
    /// # Safety
    ///
    /// The history system must be properly initialized.
    unsafe fn begin_edit_group(&mut self);

    /// Ends the current edit group.
    ///
    /// Completes the current edit group and adds it to the undo history.
    ///
    /// # Safety
    ///
    /// An edit group must have been started.
    unsafe fn end_edit_group(&mut self);

    /// Undoes the last edit operation.
    ///
    /// Reverts the most recent edit operation, restoring the buffer to its
    /// previous state.
    ///
    /// # Returns
    ///
    /// True if an operation was undone, false if there is nothing to undo.
    ///
    /// # Safety
    ///
    /// The history system must be properly initialized.
    unsafe fn undo(&mut self) -> bool;

    /// Redoes the last undone operation.
    ///
    /// Reapplies the most recently undone operation.
    ///
    /// # Returns
    ///
    /// True if an operation was redone, false if there is nothing to redo.
    ///
    /// # Safety
    ///
    /// The history system must be properly initialized.
    unsafe fn redo(&mut self) -> bool;

    /// Checks if undo is available.
    ///
    /// Returns true if there are operations in the undo history.
    ///
    /// # Returns
    ///
    /// True if undo is available, false otherwise.
    fn can_undo(&self) -> bool;

    /// Checks if redo is available.
    ///
    /// Returns true if there are operations in the redo history.
    ///
    /// # Returns
    ///
    /// True if redo is available, false otherwise.
    fn can_redo(&self) -> bool;

    /// Clears the entire undo/redo history.
    ///
    /// Removes all history entries, typically called when opening a new file
    /// or performing a major operation that should not be undoable.
    ///
    /// # Safety
    ///
    /// The history system must be properly initialized.
    unsafe fn clear_history(&mut self);
}
