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

//! The Wredit cursor.
//!
//! A [`Cursor`] is a compact, interior-mutable view into a document:
//! it records a position, an optional selection anchor and a bidi
//! index, and it borrows the raw buffer through a raw pointer plus a
//! length (the pointer is only ever dereferenced for indices below
//! the recorded length). Interior mutability via [`Cell`] means a
//! cursor can be moved, re-anchored or re-selected through a shared
//! reference — exactly what the multi-cursor engine needs — without
//! `RefCell` bookkeeping or panic paths.

use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr;

/// A pair of positions marking the two ends of a bidi run.
///
/// The pair is not ordered: `p0` is the anchor side and `p1` the
/// moving side; ordering is derived on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BidiIndex {
    /// Anchor-side position.
    pub p0: usize,
    /// Moving-side position.
    pub p1: usize,
}

impl BidiIndex {
    /// Creates a bidi index over the two positions.
    #[must_use]
    pub const fn new(p0: usize, p1: usize) -> Self {
        Self { p0, p1 }
    }

    /// The ordered `(low, high)` pair.
    #[must_use]
    pub const fn ordered(self) -> (usize, usize) {
        if self.p0 <= self.p1 {
            (self.p0, self.p1)
        } else {
            (self.p1, self.p0)
        }
    }

    /// Number of positions between the two ends (zero when equal).
    #[must_use]
    pub const fn len(self) -> usize {
        self.p1.abs_diff(self.p0)
    }

    /// `true` when both ends coincide.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.p0 == self.p1
    }
}

/// A movable, selectable cursor over a packed raw document.
///
/// # Type parameters
///
/// * `Raw`      - packed raw element type of the document
/// * `Buf`      - element type of the shared staging arena
/// * `GAP_SIZE` - staging gap size in elements
///
/// The raw buffer is borrowed through a raw pointer; the cursor never
/// owns or frees it. [`Cursor::set_raw`] must be called (again) after
/// the document moves its buffer, and the buffer must stay alive and
/// at least `length` elements long while the cursor points into it.
pub struct Cursor<Raw, Buf, const GAP_SIZE: usize> {
    /// Selected range, kept normalized through [`Self::set_selection_end`].
    selection: Cell<BidiIndex>,
    /// Bidirectional-indexing scratch pair.
    bidi: Cell<BidiIndex>,
    /// Caret position in array (element) coordinates.
    pos: Cell<usize>,
    /// Selection anchor; also used by the engine as the logical base
    /// of the cursor's staged edits.
    anchor: Cell<usize>,
    /// Whether a selection is currently active.
    selection_active: Cell<bool>,
    /// Borrowed raw document; only indices `< raw_length` are valid.
    raw: Cell<*const Raw>,
    /// Logical document length: the clamping bound for positions,
    /// including elements still staged in a gap.
    length: Cell<usize>,
    /// Committed elements behind `raw`: the bound for dereferencing
    /// the raw view. Always `<= length`.
    raw_length: Cell<usize>,
    /// `Buf` participates only through the engine's type signature.
    _buf: PhantomData<*const Buf>,
}

impl<Raw, Buf, const GAP_SIZE: usize> Cursor<Raw, Buf, GAP_SIZE> {
    /// Creates a cursor at position 0 of an empty view.
    ///
    /// Call [`Cursor::set_raw`] before relying on row/column or
    /// movement queries.
    #[must_use]
    pub fn new() -> Self {
        Self {
            selection: Cell::new(BidiIndex::new(0, 0)),
            bidi: Cell::new(BidiIndex::new(0, 0)),
            pos: Cell::new(0),
            anchor: Cell::new(0),
            selection_active: Cell::new(false),
            raw: Cell::new(ptr::null()),
            length: Cell::new(0),
            raw_length: Cell::new(0),
            _buf: PhantomData,
        }
    }

    /// Rebinds the cursor to a raw document view.
    ///
    /// Both the raw dereference bound and the logical clamping bound
    /// are reset to `length`, and the caret, anchor and selection are
    /// clamped into `0..length` so the cursor can never address past
    /// the end of the new view.
    ///
    /// # Safety
    ///
    /// When `length > 0`, `ptr` must point to at least `length`
    /// initialized `Raw` elements that stay valid and out of aliasing
    /// writes for as long as the cursor uses this view.
    pub unsafe fn set_raw(&self, ptr: *const Raw, length: usize) {
        // Safety: the caller guarantees the pointee is valid for
        // `length` elements; this fn only stores the pointer.
        self.raw.set(ptr);
        self.raw_length.set(length);
        self.length.set(length);
        let clamped = self.pos.get().min(length);
        self.pos.set(clamped);
        self.anchor.set(self.anchor.get().min(length));
        if self.selection_active.get() {
            let sel = self.selection.get();
            self.selection
                .set(BidiIndex::new(sel.p0.min(length), sel.p1.min(length)));
        }
    }

    /// Rebinds the cursor to a raw document view *without* touching
    /// the logical position, anchor or selection.
    ///
    /// This is the engine's refresh path: after the raw buffer was
    /// reallocated or resized, the cursor keeps its logical state but
    /// dereferences the fresh allocation. The logical bound is only
    /// raised, never lowered, so positions that are still backed by
    /// staged edits survive.
    ///
    /// # Safety
    ///
    /// When `raw_length > 0`, `ptr` must point to at least
    /// `raw_length` initialized `Raw` elements that stay valid and
    /// out of aliasing writes for as long as the cursor uses this
    /// view.
    pub unsafe fn rebind_raw(&self, ptr: *const Raw, raw_length: usize) {
        // Safety: the caller guarantees the pointee is valid for
        // `raw_length` elements; this fn only stores the pointer.
        self.raw.set(ptr);
        self.raw_length.set(raw_length);
        if self.length.get() < raw_length {
            self.length.set(raw_length);
        }
    }

    /// Updates the logical clamping bound after staged edits changed
    /// the logical document length.
    ///
    /// The bound never drops below the committed raw view length, so
    /// dereferences against the raw view stay safe.
    pub fn set_logical_length(&self, logical_length: usize) {
        let bound = logical_length.max(self.raw_length.get());
        self.length.set(bound);
        let clamped = self.pos.get().min(bound);
        self.pos.set(clamped);
        self.anchor.set(self.anchor.get().min(bound));
        if self.selection_active.get() {
            let sel = self.selection.get();
            self.selection
                .set(BidiIndex::new(sel.p0.min(bound), sel.p1.min(bound)));
        }
    }

    /// The raw view this cursor was bound to.
    #[must_use]
    #[inline]
    pub fn raw_ptr(&self) -> *const Raw {
        self.raw.get()
    }

    /// Logical document length: the bound for caret positions.
    #[must_use]
    #[inline]
    pub fn length(&self) -> usize {
        self.length.get()
    }

    /// Committed raw view length: the bound for dereferences.
    #[must_use]
    #[inline]
    pub fn raw_view_length(&self) -> usize {
        self.raw_length.get()
    }

    /// Caret position in element coordinates.
    #[must_use]
    #[inline]
    pub fn array_pos(&self) -> usize {
        self.pos.get()
    }

    /// Moves the caret to `position`, clamped to the view length.
    ///
    /// Returns the resulting position.
    pub fn set_array_pos(&self, position: usize) -> usize {
        let clamped = position.min(self.length.get());
        self.pos.set(clamped);
        clamped
    }

    /// Selection anchor position.
    #[must_use]
    #[inline]
    pub fn anchor(&self) -> usize {
        self.anchor.get()
    }

    /// Overwrites the selection anchor without moving the caret.
    pub fn set_anchor(&self, anchor: usize) {
        self.anchor.set(anchor.min(self.length.get()));
    }

    /// `true` when a selection is active (even an empty one).
    #[must_use]
    #[inline]
    pub fn is_selection_active(&self) -> bool {
        self.selection_active.get()
    }

    /// Starts (or restarts) a selection with the anchor at `position`
    /// (clamped) and the moving end at the caret.
    ///
    /// Returns the new anchor.
    pub fn set_selection_anchor(&self, position: usize) -> usize {
        let anchor = position.min(self.length.get());
        self.anchor.set(anchor);
        self.selection
            .set(BidiIndex::new(anchor, self.pos.get()));
        self.selection_active.set(true);
        anchor
    }

    /// Extends the selection to `position` (clamped).
    ///
    /// The selection stays normalized (`low, high`); returning the
    /// new moving end.
    pub fn set_selection_end(&self, position: usize) -> usize {
        let end = position.min(self.length.get());
        let anchor = self.anchor.get();
        self.selection.set(BidiIndex::new(anchor, end));
        self.selection_active.set(true);
        end
    }

    /// Deactivates the selection and forgets its ends.
    pub fn clear_selection(&self) {
        self.selection_active.set(false);
        self.selection.set(BidiIndex::new(0, 0));
    }

    /// The selected half-open range, or `None` when the selection is
    /// inactive or empty.
    #[must_use]
    pub fn selection_range(&self) -> Option<(usize, usize)> {
        if !self.selection_active.get() {
            return None;
        }
        let (low, high) = self.selection.get().ordered();
        if low < high { Some((low, high)) } else { None }
    }
}

impl<Raw: Copy + Into<u32> + From<u32>, Buf, const GAP_SIZE: usize> Cursor<Raw, Buf, GAP_SIZE> {
    /// Element at `index` of the committed raw view, if in range.
    fn element_at(&self, index: usize) -> Option<u32> {
        let raw = self.raw.get();
        if index >= self.raw_length.get() || raw.is_null() {
            return None;
        }
        // Safety: `index < raw_length` and the caller of `set_raw`
        // guaranteed `raw` is valid for the committed view.
        let value: u32 = unsafe { *raw.add(index) }.into();
        Some(value)
    }

    /// `true` when `index` holds a newline element.
    fn is_newline(&self, index: usize) -> bool {
        self.element_at(index)
            .is_some_and(|element| super::base::packed_char(element) == u32::from(b'\n'))
    }

    /// First index of the line containing `position`.
    fn line_start(&self, position: usize) -> usize {
        let mut index = position.min(self.length.get());
        while index > 0 && !self.is_newline(index - 1) {
            index -= 1;
        }
        index
    }

    /// First index after the line containing `position`.
    fn line_end(&self, position: usize) -> usize {
        let mut index = position.min(self.length.get());
        while index < self.length.get() && !self.is_newline(index) {
            index += 1;
        }
        index
    }

    /// Places the caret at element `position`, clamped to the view.
    ///
    /// Returns the resulting position.
    fn place(&self, position: usize) -> usize {
        self.set_array_pos(position)
    }

    /// Creates a cursor bound to `ptr` with the caret at `(row, col)`.
    ///
    /// Row and column are clamped: a row past the last line parks the
    /// caret at the end of the document, a column past the line end
    /// parks it at that line end.
    ///
    /// # Safety
    ///
    /// Same contract as [`Cursor::set_raw`].
    pub unsafe fn at(&self, ptr: *const Raw, length: usize, row: usize, col: usize) -> Self {
        let cursor = Self::new();
        // Safety: forwarded to the caller's contract for `set_raw`.
        unsafe { cursor.set_raw(ptr, length) };
        cursor.move_to(row, col);
        cursor
    }

    /// Caret row: number of newlines before the caret.
    #[must_use]
    pub fn row(&self) -> usize {
        let position = self.pos.get().min(self.length.get());
        let mut row = 0;
        let mut index = 0;
        while index < position {
            if self.is_newline(index) {
                row += 1;
            }
            index += 1;
        }
        row
    }

    /// Caret column: elements since the start of the current line.
    #[must_use]
    pub fn col(&self) -> usize {
        let position = self.pos.get().min(self.length.get());
        position - self.line_start(position)
    }

    /// Width of the caret's line in elements (newlines excluded).
    #[must_use]
    pub fn width(&self) -> usize {
        let position = self.pos.get().min(self.length.get());
        let start = self.line_start(position);
        let end = self.line_end(position);
        end - start
    }

    /// Moves the caret forward by `amount`, stopping at the end of
    /// the view. Returns the new position.
    pub fn move_forward(&self, amount: usize) -> usize {
        let target = self.pos.get().saturating_add(amount);
        self.place(target)
    }

    /// Moves the caret backward by `amount`, stopping at 0. Returns
    /// the new position.
    pub fn move_backward(&self, amount: usize) -> usize {
        let target = self.pos.get().saturating_sub(amount);
        self.place(target)
    }

    /// Moves the caret to the same column of the next line.
    ///
    /// The column is clamped to the next line's width; at the last
    /// line the caret moves to the end of the document.
    /// Returns the new position.
    pub fn move_next_line(&self) -> usize {
        let position = self.pos.get().min(self.length.get());
        let column = position - self.line_start(position);
        let length = self.length.get();
        let target = match self.element_at(position) {
            Some(_) if self.is_newline(position) => {
                // Caret sits on the newline: step over it, the next
                // line starts right after.
                let next_start = position + 1;
                let next_end = self.line_end(next_start);
                next_start + column.min(next_end - next_start)
            }
            Some(_) => {
                // Mid-line: the next newline ends the current line.
                let end = self.line_end(position);
                if end < length {
                    let next_start = end + 1;
                    let next_end = self.line_end(next_start);
                    next_start + column.min(next_end - next_start)
                } else {
                    length
                }
            }
            None => length,
        };
        self.place(target)
    }

    /// Moves the caret to the same column of the previous line.
    ///
    /// On the first line the caret stops at the line start.
    /// Returns the new position.
    pub fn move_prev_line(&self) -> usize {
        let position = self.pos.get().min(self.length.get());
        let column = position - self.line_start(position);
        let start = self.line_start(position);
        if start == 0 {
            return self.place(0);
        }
        // The previous line ends at `start - 1` (the newline that
        // separates it from the current line).
        let prev_end = start - 1;
        let prev_start = self.line_start(prev_end);
        self.place(prev_start + column.min(prev_end - prev_start))
    }

    /// Places the caret at `(row, col)`, clamped. Returns the
    /// resulting position.
    pub fn move_to(&self, row: usize, col: usize) -> usize {
        let length = self.length.get();
        let mut index = 0;
        let mut current_row = 0;
        while current_row < row && index < length {
            if self.is_newline(index) {
                current_row += 1;
            }
            index += 1;
        }
        let line_end = self.line_end(index);
        self.place(index + col.min(line_end - index))
    }

    /// The current bidi index.
    #[must_use]
    pub fn bidi_index(&self) -> BidiIndex {
        self.bidi.get()
    }

    /// Overwrites the bidi index.
    pub fn set_bidi_index(&self, index: BidiIndex) {
        self.bidi.set(index);
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Clone for Cursor<Raw, Buf, GAP_SIZE> {
    fn clone(&self) -> Self {
        Self {
            selection: Cell::new(self.selection.get()),
            bidi: Cell::new(self.bidi.get()),
            pos: Cell::new(self.pos.get()),
            anchor: Cell::new(self.anchor.get()),
            selection_active: Cell::new(self.selection_active.get()),
            raw: Cell::new(self.raw.get()),
            length: Cell::new(self.length.get()),
            raw_length: Cell::new(self.raw_length.get()),
            _buf: PhantomData,
        }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Default for Cursor<Raw, Buf, GAP_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{BidiIndex, Cursor};

    // Justification: `unwrap` in tests is permitted by the project
    // lint configuration (`allow-unwrap-in-tests`).

    /// Packs a plain character the way the engine does.
    fn ch(c: impl Into<u32>) -> u32 {
        crate::edit::base::make_packed(c.into(), crate::edit::base::T_COMMON)
    }

    #[test]
    fn bidi_index_orders_and_measures() {
        let index = BidiIndex::new(7, 3);
        assert_eq!(index.ordered(), (3, 7));
        assert_eq!(index.len(), 4);
        assert!(!index.is_empty());
        assert!(BidiIndex::new(2, 2).is_empty());
    }

    #[test]
    fn cursor_starts_at_origin_and_clamps_positions() {
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        assert_eq!(cursor.array_pos(), 0);
        assert_eq!(cursor.set_array_pos(99), 0);

        let doc = [ch('a'), ch('b'), ch('c')];
        unsafe { cursor.set_raw(doc.as_ptr(), doc.len()) };
        assert_eq!(cursor.length(), 3);
        assert_eq!(cursor.set_array_pos(99), 3);
        assert_eq!(cursor.row(), 0);
        assert_eq!(cursor.col(), 3);
        assert_eq!(cursor.width(), 3);
    }

    #[test]
    fn forward_and_backward_saturate_at_the_bounds() {
        let doc = [ch('a'), ch('b'), ch('c')];
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        unsafe { cursor.set_raw(doc.as_ptr(), doc.len()) };
        assert_eq!(cursor.move_forward(2), 2);
        assert_eq!(cursor.move_forward(100), 3);
        assert_eq!(cursor.move_backward(1), 2);
        assert_eq!(cursor.move_backward(100), 0);
    }

    #[test]
    fn row_col_and_line_navigation() {
        // "abc\nde\nf"
        let doc = [
            ch('a'),
            ch('b'),
            ch('c'),
            ch('\n'),
            ch('d'),
            ch('e'),
            ch('\n'),
            ch('f'),
        ];
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        unsafe { cursor.set_raw(doc.as_ptr(), doc.len()) };

        cursor.set_array_pos(1);
        assert_eq!((cursor.row(), cursor.col(), cursor.width()), (0, 1, 3));

        // Down from row 0 into the 2-wide row 1; column 2 clamps to 1.
        assert_eq!(cursor.move_next_line(), 5);
        assert_eq!((cursor.row(), cursor.col()), (1, 1));

        // Down again into the 1-wide row 2; column 1 is that line's
        // end, so the caret parks at the end of the document.
        assert_eq!(cursor.move_next_line(), 8);
        assert_eq!((cursor.row(), cursor.col()), (2, 1));
        assert_eq!(cursor.move_next_line(), 8); // at the end already

        // Back up through row 1, then row 0, at the same column.
        assert_eq!(cursor.move_prev_line(), 5);
        assert_eq!(cursor.move_prev_line(), 1);
        assert_eq!(cursor.move_prev_line(), 0);
        assert_eq!(cursor.move_prev_line(), 0);

        cursor.set_array_pos(2);
        assert_eq!(cursor.move_to(1, 0), 4);
        assert_eq!(cursor.move_to(9, 9), 8);
    }

    #[test]
    fn selection_lifecycle() {
        let doc = [ch('a'), ch('b'), ch('c'), ch('d')];
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        unsafe { cursor.set_raw(doc.as_ptr(), doc.len()) };

        assert_eq!(cursor.selection_range(), None);
        cursor.set_array_pos(1);
        cursor.set_selection_anchor(1);
        assert!(cursor.is_selection_active());
        assert_eq!(cursor.selection_range(), None); // empty so far
        cursor.set_selection_end(3);
        assert_eq!(cursor.selection_range(), Some((1, 3)));
        // Selecting backwards normalizes the range.
        cursor.set_selection_end(0);
        assert_eq!(cursor.selection_range(), Some((0, 1)));
        cursor.clear_selection();
        assert!(!cursor.is_selection_active());
        assert_eq!(cursor.selection_range(), None);
    }

    #[test]
    fn set_raw_reclamps_everything() {
        let long = [ch('a'), ch('b'), ch('c'), ch('d')];
        let short = [ch('x')];
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        unsafe { cursor.set_raw(long.as_ptr(), long.len()) };
        cursor.set_array_pos(4);
        cursor.set_selection_anchor(3);
        cursor.set_selection_end(4);
        unsafe { cursor.set_raw(short.as_ptr(), short.len()) };
        assert_eq!(cursor.array_pos(), 1);
        assert_eq!(cursor.anchor(), 1);
        assert_eq!(cursor.selection_range(), None); // clamped to empty
    }

    #[test]
    fn empty_view_never_dereferences() {
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        // No raw view bound at all: every query must stay inert.
        assert_eq!(cursor.row(), 0);
        assert_eq!(cursor.col(), 0);
        assert_eq!(cursor.width(), 0);
        assert_eq!(cursor.move_next_line(), 0);
        assert_eq!(cursor.move_prev_line(), 0);
        assert_eq!(cursor.move_to(5, 5), 0);
        unsafe { cursor.set_raw(core::ptr::null(), 0) };
        assert_eq!(cursor.move_forward(3), 0);
    }
}
