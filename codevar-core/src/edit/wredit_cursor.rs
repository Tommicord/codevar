//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to
//! in writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use crate::edit::Writable;
use std::cell::RefCell;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BidiIndex {
    /// Primary index (typically row)
    pub p0: usize,
    /// Secondary index (typically column)
    pub p1: usize,
}

impl BidiIndex {
    /// Creates a new bidirectional index.

    pub fn new(p0: usize, p1: usize) -> Self {
        Self { p0, p1 }
    }

    /// Returns the primary index.

    pub fn primary(&self) -> usize {
        self.p0
    }

    /// Returns the secondary index.

    pub fn secondary(&self) -> usize {
        self.p1
    }
}

#[repr(C)]
pub struct Cursor<Raw, Buf, const GAP_SIZE: usize> {
    /// The cursor selection
    selection: RefCell<BidiIndex>,
    /// The cursor dimension tracking
    bidi: RefCell<BidiIndex>,
    /// The array position
    pos: RefCell<usize>,
    /// Anchor position for selection ranges.
    selection_anchor: RefCell<usize>,
    /// Whether a selection is currently active.
    selection_active: RefCell<bool>,
    /// Raw pointer to the writable data for read-only access
    ptr: *const Raw,
    /// Length of the raw data
    length: usize,
    _phantom: std::marker::PhantomData<(Raw, Buf)>,
}

impl<Raw, Buf, const GAP_SIZE: usize> Cursor<Raw, Buf, GAP_SIZE> {
    /// Creates a new cursor at the origin.

    pub fn new() -> Self {
        Self {
            selection: RefCell::new(BidiIndex::new(0, 0)),
            bidi: RefCell::new(BidiIndex::new(0, 0)),
            pos: RefCell::new(0),
            selection_anchor: RefCell::new(0),
            selection_active: RefCell::new(false),
            ptr: std::ptr::null(),
            length: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new cursor at the specified position.

    pub fn at(row: usize, col: usize) -> Self {
        Self {
            selection: RefCell::new(BidiIndex::new(row, col)),
            bidi: RefCell::new(BidiIndex::new(row, col)),
            pos: RefCell::new(0),
            selection_anchor: RefCell::new(0),
            selection_active: RefCell::new(false),
            ptr: std::ptr::null(),
            length: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sets the raw pointer and length for data access

    pub fn set_raw(&mut self, raw_ptr: *const Raw, raw_length: usize) {
        if raw_ptr.is_null() {
            return;
        }
        self.ptr = raw_ptr;
        self.length = raw_length;
    }

    /// Returns the line width (in elements) for the given row.

    fn line_width_at(&self, row: usize) -> usize {
        if self.ptr.is_null() || self.length == 0 {
            return 0;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.ptr as *const u8,
                std::mem::size_of::<Raw>() * self.length,
            )
        };
        let elem_size = std::mem::size_of::<Raw>();
        let mut current_row = 0;
        let mut line_start = 0;
        for (i, &byte) in bytes.iter().enumerate() {
            if current_row == row {
                if byte == 0x0A || byte == 0x00 {
                    return i.saturating_sub(line_start) / elem_size;
                }
            } else if byte == 0x0A {
                current_row += 1;
                line_start = i + 1;
            }
        }
        0
    }

    /// Maps `(row, col)` to a buffer array index without borrowing cursor state.
    fn array_map_pos(&self, row: usize, col: usize) -> Option<usize> {
        if self.ptr.is_null() || self.length == 0 {
            return None;
        }
        if col > self.line_width_at(row) {
            return None;
        }
        unsafe {
            let bytes = std::slice::from_raw_parts(
                self.ptr as *const u8,
                std::mem::size_of::<Raw>() * self.length,
            );
            let elem_size = std::mem::size_of::<Raw>();
            let mut current_row = 0;
            let mut current_col = 0;
            for (i, &byte) in bytes.iter().enumerate() {
                if byte == 0x0A {
                    current_row += 1;
                    current_col = 0;
                } else {
                    current_col += 1;
                }
                if current_row == row && current_col == col {
                    return Some(i / elem_size);
                }
            }
            None
        }
    }

    /// Returns the bidirectional index.

    pub fn bidi_index(&self) -> BidiIndex {
        self.bidi.borrow().clone()
    }

    /// Returns the mutable bidirectional index.

    pub fn bidi_index_mut(&mut self) -> BidiIndex {
        self.bidi.borrow().clone()
    }

    /// Returns the dimension.

    pub fn selection(&self) -> BidiIndex {
        self.selection.borrow().clone()
    }

    /// Returns the array position.

    pub fn array_pos(&self) -> usize {
        *self.pos.borrow()
    }

    /// Sets the array position.

    pub fn set_array_pos(&mut self, pos: usize) {
        *self.pos.borrow_mut() = pos.clamp(0, self.length);
    }

    /// Moves the cursor forward by the specified amount.

    pub fn move_forward(&mut self, amount: usize) {
        let (row, col) = {
            let mut bidi = self.bidi.borrow_mut();
            let mut selection = self.selection.borrow_mut();
            bidi.p1 = bidi.p1.saturating_add(amount);
            selection.p1 = bidi.p1;
            (bidi.p0, bidi.p1)
        };
        if let Some(array_pos) = self.array_map_pos(row, col) {
            *self.pos.borrow_mut() = array_pos.saturating_add(amount);
        } else {
            *self.pos.borrow_mut() = col;
        }
    }

    /// Moves the cursor backward by the specified amount.

    pub fn move_backward(&mut self, amount: usize) {
        let (row, col) = {
            let mut bidi = self.bidi.borrow_mut();
            let mut selection = self.selection.borrow_mut();
            if bidi.p1 >= amount {
                bidi.p1 -= amount;
                selection.p1 = bidi.p1;
            }
            (bidi.p0, bidi.p1)
        };
        if let Some(array_pos) = self.array_map_pos(row, col) {
            *self.pos.borrow_mut() = array_pos.saturating_sub(amount);
        }
    }

    /// Moves the cursor to the next line.

    pub fn move_next_line(&mut self) {
        let (row, col) = {
            let mut bidi = self.bidi.borrow_mut();
            let mut selection = self.selection.borrow_mut();
            bidi.p0 = bidi.p0.saturating_add(1);
            bidi.p1 = 0;
            selection.p0 = bidi.p0;
            selection.p1 = bidi.p1;
            (bidi.p0, bidi.p1)
        };
        if let Some(array_pos) = self.array_map_pos(row, col) {
            *self.pos.borrow_mut() = array_pos;
        }
    }

    /// Moves the cursor to the previous line.

    pub fn move_prev_line(&mut self) {
        let (row, col) = {
            let mut bidi = self.bidi.borrow_mut();
            let mut selection = self.selection.borrow_mut();
            if bidi.p0 > 0 {
                bidi.p0 -= 1;
            }
            bidi.p1 = 0;
            selection.p0 = bidi.p0;
            selection.p1 = bidi.p1;
            (bidi.p0, bidi.p1)
        };
        if let Some(array_pos) = self.array_map_pos(row, col) {
            *self.pos.borrow_mut() = array_pos;
        }
    }

    /// Moves the cursor to the specified position.

    pub fn move_to(&mut self, row: usize, col: usize) {
        {
            let mut bidi = self.bidi.borrow_mut();
            let mut selection = self.selection.borrow_mut();
            bidi.p0 = row;
            bidi.p1 = col;
            selection.p0 = bidi.p0;
            selection.p1 = bidi.p1;
        }
        if let Some(array_pos) = self.array_map_pos(row, col) {
            *self.pos.borrow_mut() = array_pos;
        }
    }

    /// Returns the row position.

    pub fn row(&self) -> usize {
        self.selection.borrow().p0
    }

    /// Returns the column position.

    pub fn col(&self) -> usize {
        self.selection.borrow().p1
    }

    /// Returns the line width of the current row.

    pub fn width(&self) -> usize {
        self.line_width_at(self.row())
    }

    /// Sets the anchor position for a new selection.

    pub fn set_selection_anchor(&self, pos: usize) {
        *self.selection_anchor.borrow_mut() = pos;
        *self.selection_active.borrow_mut() = true;
    }

    /// Sets the current selection end position.

    pub fn set_selection_end(&self, pos: usize) {
        let _ = pos;
        *self.selection_active.borrow_mut() = true;
    }

    /// Clears the current selection.

    pub fn clear_selection(&self) {
        *self.selection_active.borrow_mut() = false;
    }

    /// Returns the current selection range as buffer offsets.

    pub fn selection_range(&self) -> Option<(usize, usize)> {
        if *self.selection_active.borrow() {
            Some((*self.selection_anchor.borrow(), *self.pos.borrow()))
        } else {
            None
        }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Clone for Cursor<Raw, Buf, GAP_SIZE> {
    fn clone(&self) -> Self {
        Self {
            selection: RefCell::new(self.selection.borrow().clone()),
            bidi: RefCell::new(self.bidi.borrow().clone()),
            pos: RefCell::new(*self.pos.borrow()),
            selection_anchor: RefCell::new(*self.selection_anchor.borrow()),
            selection_active: RefCell::new(*self.selection_active.borrow()),
            ptr: self.ptr,
            length: self.length,
            _phantom: std::marker::PhantomData,
        }
    }
}
