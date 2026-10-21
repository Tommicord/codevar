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

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr;

use super::scan;

/// Documents at least this long get a lazily built line-start cache.
///
/// The threshold is deliberately small: the ring is stack-resident and
/// cheap to fill, and interactive row/column queries repeat far more
/// often than a full vectorised rescan. Above it, repeated work near
/// the caret becomes constant time.
const LINE_CACHE_MIN_ELEMENTS: usize = 1 << 6;

/// Number of line starts kept in the cursor's stack-resident ring.
///
/// Twenty-four starts cover several screenfuls around the caret while
/// keeping the whole multi-cursor pool inside the engine's inline size
/// budget; the wrap is a single compare because the sum of the head and
/// an offset is at most twice the capacity.
const LINE_RING_CAPACITY: usize = 0x20;

/// A fixed-capacity ring of consecutive line starts around the caret.
///
/// The entries are held on the stack (no allocation) and stay sorted by
/// row: `starts[head]` is the first cached line, and `first_row` names
/// its row. Pushing past either end evicts the opposite line, so the
/// window follows local caret motion without ever growing.
#[derive(Clone, Copy)]
pub(crate) struct LineRing {
    /// Element index where each cached line starts.
    starts: [u32; LINE_RING_CAPACITY],
    /// Ring slot of the first (lowest-row) cached start.
    head: u16,
    /// Number of valid starts; zero means "cache cold".
    len: u16,
    /// Row number of the first cached start.
    first_row: u32,
}

impl LineRing {
    /// An empty ring.
    #[must_use]
    pub(crate) const fn empty() -> Self {
        Self {
            starts: [0; LINE_RING_CAPACITY],
            head: 0,
            len: 0,
            first_row: 0,
        }
    }

    /// `true` when the ring holds no lines.
    #[must_use]
    #[inline]
    pub(crate) const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Number of cached lines.
    #[must_use]
    #[inline]
    const fn len(self) -> usize {
        self.len as usize
    }

    /// Drops the cached window without touching the 128-byte start
    /// array. The next lookup rebuilds it with `reload`.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.first_row = 0;
    }

    /// Row of the first cached line.
    #[must_use]
    #[inline]
    const fn first_row(self) -> usize {
        self.first_row as usize
    }

    /// Row of the last cached line.
    #[must_use]
    #[inline]
    pub(crate) fn last_row(self) -> usize {
        self.first_row() + self.len() - 1
    }

    /// Element index where the last cached line starts.
    #[must_use]
    #[inline]
    pub(crate) fn last_start(self) -> usize {
        self.at(self.len() - 1)
    }

    /// Element index where the first cached line starts.
    #[must_use]
    #[inline]
    pub(crate) fn first_start(self) -> usize {
        self.at(0)
    }

    /// Wraps a slot index that is below twice the capacity.
    #[must_use]
    #[inline]
    const fn wrap(index: usize) -> usize {
        if index >= LINE_RING_CAPACITY {
            index - LINE_RING_CAPACITY
        } else {
            index
        }
    }

    /// Element index where cached line `offset` starts.
    #[must_use]
    #[inline]
    fn at(self, offset: usize) -> usize {
        let slot = Self::wrap(usize::from(self.head) + offset);
        self.starts[slot] as usize
    }

    /// Offset of `row` inside the ring, if it is cached.
    #[must_use]
    #[inline]
    fn offset_of(self, row: usize) -> Option<usize> {
        let offset = row.checked_sub(self.first_row())?;
        (offset < self.len()).then_some(offset)
    }

    /// Replaces the ring with a sorted window of line starts.
    #[inline]
    fn reload(&mut self, first_row: usize, entries: &[u32]) {
        let count = entries.len().min(LINE_RING_CAPACITY);
        self.starts[..count].copy_from_slice(&entries[..count]);
        self.head = 0;
        self.len = count as u16;
        self.first_row = first_row as u32;
    }

    /// Shifts every cached start after `position` right by `delta`.
    ///
    /// Only valid for a newline-free insertion: the cached rows keep
    /// their numbers, so the window stays usable instead of being
    /// dropped and refilled from the document.
    #[inline]
    pub(crate) fn shift_after_insert(&mut self, position: usize, delta: usize) {
        if self.is_empty() || delta == 0 {
            return;
        }
        let position = position.min(u32::MAX as usize) as u32;
        let delta = delta.min(u32::MAX as usize) as u32;
        let head = usize::from(self.head);
        let len = self.len();
        // Cached starts are sorted, so only the suffix after the edit
        // moves. Typing at the end then costs one binary search instead
        // of walking (and dirtying) the whole window.
        let mut low = 0;
        let mut high = len;
        while low < high {
            let middle = (low + high) / 2;
            if self.at(middle) <= position as usize {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        for index in low..len {
            let slot = Self::wrap(head + index);
            self.starts[slot] = self.starts[slot].saturating_add(delta);
        }
    }

    /// Shifts every cached start after `position` left by `count`.
    ///
    /// Only valid for a newline-free removal, which preserves the row
    /// numbers of the cached window.
    #[inline]
    pub(crate) fn shift_after_remove(&mut self, position: usize, count: usize) {
        if self.is_empty() || count == 0 {
            return;
        }
        let position = position.min(u32::MAX as usize) as u32;
        let count = count.min(u32::MAX as usize) as u32;
        let head = usize::from(self.head);
        let len = self.len();
        let mut low = 0;
        let mut high = len;
        while low < high {
            let middle = (low + high) / 2;
            if self.at(middle) <= position as usize {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        for index in low..len {
            let slot = Self::wrap(head + index);
            self.starts[slot] = self.starts[slot].saturating_sub(count);
        }
    }

    /// Applies a removal of `start..end` that contains newlines.
    ///
    /// Every cached start strictly inside the removed range follows a
    /// newline that no longer exists, so it is dropped; starts past the
    /// range move left by `count`. The first cached row is untouched,
    /// which is what lets the caller keep `first_row` as-is.
    #[inline]
    pub(crate) fn trim_removed_range(&mut self, start: usize, end: usize, count: usize) {
        if self.is_empty() {
            return;
        }
        let head = usize::from(self.head);
        let len = self.len();
        let count = count.min(u32::MAX as usize) as u32;
        let mut write = 0;
        for read in 0..len {
            let slot = Self::wrap(head + read);
            let value = self.starts[slot];
            let position = value as usize;
            if position > start && position <= end {
                continue;
            }
            let value = if position > end {
                value.saturating_sub(count)
            } else {
                value
            };
            let target = Self::wrap(head + write);
            self.starts[target] = value;
            write += 1;
        }
        self.len = write as u16;
    }

    /// Appends the line starting at `start`, evicting the oldest line
    /// when the ring is full.
    #[inline]
    pub(crate) fn push_back(&mut self, start: u32) {
        let slot = Self::wrap(usize::from(self.head) + self.len());
        self.starts[slot] = start;
        if self.len() == LINE_RING_CAPACITY {
            self.head = Self::wrap(usize::from(self.head) + 1) as u16;
            self.first_row = self.first_row.saturating_add(1);
        } else {
            self.len += 1;
        }
    }

    /// Prepends the line starting at `start`, evicting the newest line
    /// when the ring is full.
    ///
    /// The caller only prepends ahead of an existing window, so there
    /// is always a row to decrement.
    #[inline]
    fn push_front(&mut self, start: u32) {
        let head = Self::wrap(usize::from(self.head) + LINE_RING_CAPACITY - 1);
        self.head = head as u16;
        self.starts[head] = start;
        self.first_row = self.first_row.saturating_sub(1);
        if self.len() < LINE_RING_CAPACITY {
            self.len += 1;
        }
    }
}

/// Number of sparse line starts retained for random row access.
///
/// The anchor array is a second-level index: it stores the start of
/// every `stride`-th row, so resolving an arbitrary row scans at most
/// one block of `stride` lines instead of the whole document. The
/// index lives on the heap and is only built after a distant jump, so
/// its footprint stays out of the engine's inline working set.
///
/// The capacity is deliberately generous: with one anchor per row a
/// document of up to [`LINE_ANCHOR_CAPACITY`] rows resolves any row
/// with a single lookup, and a random jump never has to touch the
/// document text at all. Cachegrind showed that scanning an anchor
/// block was the single largest source of L1 misses in the navigation
/// workload, and the scan is exactly what a denser index removes. The
/// backing store is allocated to the number of rows the document
/// actually has, so a short file never pays for the full capacity.
const LINE_ANCHOR_CAPACITY: usize = 8192;

/// Smallest distance between two sparse line anchors.
const LINE_ANCHOR_STRIDE: usize = 1;

/// Newline positions collected per bulk scan while building the index.
const LINE_ANCHOR_SCAN_CHUNK: usize = 256;

/// Sparse, document-level starts for random row/position lookup.
///
/// A [`LineRing`] is the L1-resident window around the caret; the
/// anchors are the fallback used when a jump leaves that window. They
/// are built lazily on the first distant lookup and invalidated by any
/// commit that changes the line layout. `len == 0` means "no valid
/// index", which doubles as the invalidation flag without a second
/// boolean.
#[derive(Clone)]
pub(crate) struct LineAnchors {
    /// Start element of rows `0, stride, 2 * stride, ...`.
    starts: Box<[u32]>,
    /// Number of valid entries in `starts`.
    len: u16,
    /// Rows between adjacent anchors.
    stride: u32,
    /// Number of addressable rows (newlines + one).
    rows: u32,
}

impl LineAnchors {
    /// An invalid/empty index.
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            starts: Vec::new().into_boxed_slice(),
            len: 0,
            stride: 0,
            rows: 0,
        }
    }

    /// `true` when a built index is available.
    #[must_use]
    #[inline]
    pub(crate) const fn is_valid(&self) -> bool {
        self.len != 0
    }

    /// Drops the index without clearing the backing array.
    #[inline]
    pub(crate) fn invalidate(&mut self) {
        self.len = 0;
    }

    /// Number of addressable rows in the indexed document.
    #[must_use]
    #[inline]
    pub(crate) const fn rows(&self) -> usize {
        self.rows as usize
    }

    /// Grows the indexed row count after newlines were appended after
    /// the last anchor.
    ///
    /// The stored starts stay valid when every anchor lies at or before
    /// the insertion point: the new rows only extend the final block.
    /// The index is rejected once the last row drifts more than one
    /// stride past the final anchor, which keeps fallback scans bounded.
    #[inline]
    pub(crate) fn extend_rows(&mut self, at: usize, newlines: usize) -> bool {
        if !self.is_valid() {
            return false;
        }
        if newlines == 0 {
            return true;
        }
        let last = self.len() - 1;
        if self.starts[last] as usize > at {
            return false;
        }
        let Some(rows) = self.rows().checked_add(newlines) else {
            return false;
        };
        if rows > u32::MAX as usize {
            return false;
        }
        let last_anchor_row = last * self.stride as usize;
        if (rows - 1).saturating_sub(last_anchor_row) > self.stride as usize {
            return false;
        }
        self.rows = rows as u32;
        true
    }

    /// Shrinks the index after a removal whose line starts only affect
    /// trailing anchors.
    ///
    /// `position..end` is the removed range. Anchors strictly after
    /// `position` cannot survive a newline removal inside that range,
    /// so they are dropped; the remaining starts and their row numbers
    /// stay correct because nothing before `position` moved.
    #[inline]
    pub(crate) fn trim_after(&mut self, position: usize, end: usize, removed_newlines: usize) -> bool {
        if !self.is_valid() {
            return false;
        }
        let len = self.len();
        // Anchors past the removed range would need shifting left,
        // which also renumbers their rows: fall back to a rebuild.
        if self.starts[len - 1] as usize > end {
            return false;
        }
        let position = position.min(u32::MAX as usize) as u32;
        let keep = self.starts[..len].partition_point(|start| *start <= position);
        if keep == 0 {
            return false;
        }
        let Some(rows) = self.rows().checked_sub(removed_newlines) else {
            return false;
        };
        if rows == 0 {
            return false;
        }
        let last_anchor_row = (keep - 1) * self.stride as usize;
        if (rows - 1).saturating_sub(last_anchor_row) > self.stride as usize {
            return false;
        }
        self.len = keep as u16;
        self.rows = rows as u32;
        true
    }

    /// Shifts every anchor after `position` right by `delta` elements.
    ///
    /// Only valid for an edit that inserts no newline: the row layout
    /// is unchanged, so the sparse starts merely move with the text.
    /// A start exactly at `position` is left in place because inserting
    /// at the beginning of a line does not move that line's start.
    #[inline]
    pub(crate) fn shift_after_insert(&mut self, position: usize, delta: usize) {
        if !self.is_valid() || delta == 0 {
            return;
        }
        let position = position.min(u32::MAX as usize) as u32;
        let delta = delta.min(u32::MAX as usize) as u32;
        let len = self.len();
        let low = self.starts[..len].partition_point(|start| *start <= position);
        let tail = &mut self.starts[low..len];
        // The common case cannot overflow, and a plain add lets the
        // compiler vectorise the tail: a keystroke in the middle of a
        // large document shifts thousands of anchors.
        if tail
            .last()
            .is_none_or(|last| *last <= u32::MAX - delta)
        {
            for start in tail.iter_mut() {
                *start += delta;
            }
            return;
        }
        for start in tail.iter_mut() {
            *start = start.saturating_add(delta);
        }
    }

    /// Shifts every anchor after `position` left by `count` elements.
    ///
    /// Only valid for a removal that contains no newline: no line start
    /// can fall strictly inside the removed range, so every later start
    /// simply moves left and `rows` stays the same.
    #[inline]
    pub(crate) fn shift_after_remove(&mut self, position: usize, count: usize) {
        if !self.is_valid() || count == 0 {
            return;
        }
        let position = position.min(u32::MAX as usize) as u32;
        let count = count.min(u32::MAX as usize) as u32;
        let len = self.len();
        let low = self.starts[..len].partition_point(|start| *start <= position);
        let tail = &mut self.starts[low..len];
        // Same reasoning as the insert path: the plain subtract
        // vectorises, the saturating one only runs in the overflow case.
        if tail.first().is_none_or(|first| *first >= count) {
            for start in tail.iter_mut() {
                *start -= count;
            }
            return;
        }
        for start in tail.iter_mut() {
            *start = start.saturating_sub(count);
        }
    }

    /// Number of stored anchors.
    #[must_use]
    #[inline]
    const fn len(&self) -> usize {
        self.len as usize
    }

    /// Builds the sparse index over `raw[..len]`.
    ///
    /// The first pass counts the newlines (vectorised); the second
    /// collects the newline positions in bulk and records every
    /// `stride`-th one. The stride grows with the document so the index
    /// never exceeds [`LINE_ANCHOR_CAPACITY`] entries.
    ///
    /// # Safety
    ///
    /// `raw` must be valid for reads of `len` initialized elements (or
    /// be null when `len == 0`).
    pub(crate) unsafe fn build<Raw: Copy + Into<u32>>(raw: *const Raw, len: usize) -> Self {
        if len == 0 || raw.is_null() {
            return Self::empty();
        }
        // First pass: collect the row starts in blocks. One vectorised
        // pass replaces one search call per line, and a document that
        // fits the anchor capacity keeps every row start, so the index
        // answers a jump with a lookup and never touches the text.
        let mut buffer = [0u32; LINE_ANCHOR_SCAN_CHUNK];
        let mut starts: Vec<u32> = Vec::new();
        starts.push(0);
        let mut newlines = 0usize;
        let mut from = 0;
        loop {
            // Safety: `from <= len` and `raw[..len]` is the committed
            // view supplied by the caller.
            let found = unsafe { scan::collect_newlines_forward(raw, len, from, &mut buffer) };
            if found == 0 {
                break;
            }
            for newline in &buffer[..found] {
                newlines += 1;
                from = *newline as usize + 1;
                if starts.len() <= LINE_ANCHOR_CAPACITY {
                    starts.push(*newline + 1);
                }
            }
        }
        let rows = newlines.saturating_add(1);
        if newlines <= LINE_ANCHOR_CAPACITY {
            // Dense index: every row start is known, so a row lookup
            // and a line-end lookup are plain array reads.
            return Self {
                starts: starts.into_boxed_slice(),
                len: rows as u16,
                stride: LINE_ANCHOR_STRIDE as u32,
                rows: rows as u32,
            };
        }
        // Very large document: keep one start every `stride` rows. The
        // first pass only counted the newlines, so the strided starts
        // need a second pass over the text.
        let stride = rows
            .div_ceil(LINE_ANCHOR_CAPACITY)
            .max(LINE_ANCHOR_STRIDE);
        let capacity = rows.div_ceil(stride);
        let mut sparse = vec![0u32; capacity].into_boxed_slice();
        let mut count = 1usize;
        let mut row = 0usize;
        let mut from = 0;
        while count < capacity {
            // Safety: same committed-view contract as above.
            let found = unsafe { scan::collect_newlines_forward(raw, len, from, &mut buffer) };
            if found == 0 {
                break;
            }
            for newline in &buffer[..found] {
                row += 1;
                from = *newline as usize + 1;
                if row.is_multiple_of(stride) {
                    sparse[count] = *newline + 1;
                    count += 1;
                    if count == capacity {
                        break;
                    }
                }
            }
        }
        Self {
            starts: sparse,
            len: count as u16,
            stride: stride as u32,
            rows: rows as u32,
        }
    }

    /// Anchor block containing `row`: `(first_row, first_start)`.
    ///
    /// Returns `None` for a row past the last line.
    #[must_use]
    #[inline]
    fn block_for_row(&self, row: usize) -> Option<(usize, usize)> {
        if !self.is_valid() || row >= self.rows as usize {
            return None;
        }
        let stride = self.stride as usize;
        let block = (row / stride).min(self.len() - 1);
        Some((block * stride, self.starts[block] as usize))
    }

    /// Exclusive end of `row`, when the index stores every row start.
    ///
    /// The next row starts right after the newline that ends this row,
    /// so the end is one element before that start. A dense index turns
    /// the column clamp of a jump into a lookup instead of a forward
    /// scan over the line.
    #[must_use]
    #[inline]
    pub(crate) fn row_end(&self, row: usize) -> Option<usize> {
        if !self.is_valid() || self.stride != 1 {
            return None;
        }
        let next = row + 1;
        if next >= self.len() {
            return None;
        }
        Some(self.starts[next] as usize - 1)
    }

    /// Anchor block at or before element `end`: `(first_row, first_start)`.
    #[must_use]
    #[inline]
    fn block_for_position(&self, end: usize) -> (usize, usize) {
        if !self.is_valid() {
            return (0, 0);
        }
        let stride = self.stride as usize;
        let starts = &self.starts[..self.len()];
        // First anchor strictly after `end`; the previous one is the
        // block that can reach `end` with a bounded scan.
        let mut low = 0;
        let mut high = starts.len();
        while low < high {
            let middle = (low + high) / 2;
            if starts[middle] as usize <= end {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        if low == 0 {
            return (0, 0);
        }
        let block = low - 1;
        (block * stride, starts[block] as usize)
    }

    /// Element index where `row` starts, using the nearest anchor.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::build`]; the row must be inside the
    /// indexed document.
    #[inline]
    unsafe fn row_start<Raw: Copy + Into<u32>>(
        &self,
        raw: *const Raw,
        len: usize,
        row: usize,
    ) -> Option<usize> {
        let (base_row, base_start) = self.block_for_row(row)?;
        if row == base_row {
            return Some(base_start);
        }
        let remaining = row - base_row;
        // Safety: `base_start <= len`, so the sub-slice is in bounds.
        let newline = unsafe { scan::find_nth_newline(raw.add(base_start), len - base_start, remaining) }?;
        Some(base_start + newline + 1)
    }
}

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
/// the document moves it's buffer, and the buffer must stay alive and
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

    /// `true` when the committed view is large enough to amortise the
    /// line ring and every line start fits its `u32` entry.
    #[inline]
    fn indexable(len: usize) -> bool {
        len >= LINE_CACHE_MIN_ELEMENTS && len <= u32::MAX as usize
    }

    /// Rebuilds the stack ring around line `row`, which starts at
    /// element `start`.
    ///
    /// The window holds up to [`LINE_RING_CAPACITY`] consecutive starts
    /// centred on `row`; local navigation then pushes either edge
    /// without scanning the document.
    fn rebuild_line_ring(cache: &mut LineRing, raw: *const Raw, len: usize, row: usize, start: usize) {
        if !Self::indexable(len) || start > len {
            *cache = LineRing::empty();
            return;
        }
        // The target sits in the middle when there is enough history;
        // near the document start it moves down so the window still
        // covers the following lines.
        let want_back = row.min(LINE_RING_CAPACITY / 2);
        let mut entries = [0u32; LINE_RING_CAPACITY];
        entries[want_back] = start as u32;
        // The vectorised collector fills the whole window with two
        // scans (one per direction) instead of one search kernel call
        // per line. It falls back to the per-line loop when the target
        // is close enough to the document start that fewer than
        // `want_back` newlines exist below it.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use codevar_base::cpuid::{Feature, has};

            if core::mem::size_of::<Raw>() == core::mem::size_of::<u32>() && has(Feature::Avx2) {
                let mut back = [0u32; LINE_RING_CAPACITY];
                let before = start.saturating_sub(1);
                let backs =
                    unsafe { scan::collect_newlines_backward(raw, len, before, &mut back[..want_back]) };
                if backs == want_back {
                    for (offset, &newline) in back[..backs].iter().enumerate() {
                        entries[want_back - 1 - offset] = newline.saturating_add(1);
                    }
                    let room = LINE_RING_CAPACITY - (want_back + 1);
                    let mut forward = [0u32; LINE_RING_CAPACITY];
                    let forwards =
                        unsafe { scan::collect_newlines_forward(raw, len, start, &mut forward[..room]) };
                    for (offset, &newline) in forward[..forwards].iter().enumerate() {
                        entries[want_back + 1 + offset] = newline.saturating_add(1);
                    }
                    let count = want_back + 1 + forwards;
                    let mut ring = LineRing::empty();
                    ring.reload(row - want_back, &entries[..count]);
                    *cache = ring;
                    return;
                }
            }
        }
        let mut back = 0;
        let mut cursor = start;
        while back < want_back {
            // The newline ending the line *before* the current one is
            // the last newline before `cursor - 1`.
            let before = cursor.saturating_sub(1);
            // Safety: `raw[..len]` is the committed view bound by
            // `set_raw`/`rebind_raw`, and `cursor <= len`.
            let previous = match unsafe { scan::find_newline_backward(raw, len, before) } {
                Some(newline) => newline + 1,
                None => 0,
            };
            cursor = previous;
            back += 1;
            entries[want_back - back] = cursor as u32;
        }
        let mut count = want_back + 1;
        let mut forward = start;
        while count < LINE_RING_CAPACITY {
            // Safety: same committed-view contract as above.
            let Some(newline) = (unsafe { scan::find_newline_forward(raw, len, forward) }) else {
                break;
            };
            forward = newline + 1;
            entries[count] = forward as u32;
            count += 1;
        }
        let mut ring = LineRing::empty();
        ring.reload(row - back, &entries[..count]);
        *cache = ring;
    }

    /// Row of `end` when the cached window covers it.
    ///
    /// Returns [`None`] on a miss; the caller then falls back to the
    /// vectorised full count and refreshes the ring.
    fn cached_row(&self, cache: &LineRing, rows: Option<usize>, end: usize) -> Option<usize> {
        if cache.is_empty() {
            return None;
        }
        let first = cache.at(0);
        if end < first {
            return None;
        }
        let last_start = cache.at(cache.len() - 1);
        if end < last_start {
            // First cached start strictly after `end`; the row is the
            // one just before it.
            let mut low = 0;
            let mut high = cache.len();
            while low < high {
                let middle = (low + high) / 2;
                if cache.at(middle) <= end {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            return Some(cache.first_row() + low - 1);
        }
        // When the window already reaches the document's last row,
        // nothing after `last_start` can be a newline: the caret is on
        // that row without scanning the (potentially long) final line.
        let last_row = cache.last_row();
        if let Some(rows) = rows
            && last_row.saturating_add(1) >= rows
        {
            return Some(last_row);
        }
        // `end` may still be on the last cached line; only one line
        // scan is needed to check, rather than counting from 0.
        if end <= self.line_end(last_start) {
            Some(last_row)
        } else {
            None
        }
    }

    /// Start and exclusive end of the logical line containing
    /// `position`.
    ///
    /// A single [`scan::line_bounds`] call performs the CPUID dispatch
    /// once and runs both SIMD directions, so callers that need both
    /// boundaries avoid two independent scanner calls.
    fn line_bounds(&self, position: usize) -> (usize, usize) {
        let length = self.length.get();
        let index = position.min(length);
        let raw = self.raw.get();
        let len = if raw.is_null() { 0 } else { self.raw_length.get() };
        // The committed view is the only part that can hold a newline;
        // staged positions past it share the last committed line.
        let before = index.min(len);
        // Safety: `raw[..len]` is the committed view bound by
        // `set_raw`/`rebind_raw`, and `before <= len`.
        let (start, end) = unsafe { scan::line_bounds(raw, len, before) };
        // `line_bounds` reports the committed length when no newline
        // follows; the logical line also covers the staged tail.
        let end = if end >= len { length } else { end };
        (start, end)
    }

    /// First index of the line containing `position`.
    fn line_start(&self, position: usize) -> usize {
        let index = position.min(self.length.get());
        let raw = self.raw.get();
        let len = if raw.is_null() { 0 } else { self.raw_length.get() };
        // The committed view is the only part that can hold a newline;
        // staged positions past it share the last committed line.
        let before = index.min(len);
        // Safety: `raw[..len]` is the committed view bound by
        // `set_raw`/`rebind_raw`.
        match unsafe { scan::find_newline_backward(raw, len, before) } {
            Some(newline) => newline + 1,
            None => 0,
        }
    }

    /// First index after the line containing `position`.
    fn line_end(&self, position: usize) -> usize {
        let length = self.length.get();
        let index = position.min(length);
        let raw = self.raw.get();
        let len = if raw.is_null() { 0 } else { self.raw_length.get() };
        if index >= len {
            // Past the committed view: no newline can follow.
            return length;
        }
        // Safety: `raw[..len]` is the committed view bound by
        // `set_raw`/`rebind_raw`, and `index < len`.
        match unsafe { scan::find_newline_forward(raw, len, index) } {
            Some(newline) => newline,
            None => length,
        }
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
        self.row_impl(None, None, 0)
    }

    /// Caret row using `cache` to avoid rescanning the document.
    pub(crate) fn row_cached(
        &self,
        cache: &mut LineRing,
        anchors: Option<&LineAnchors>,
        rows: usize,
    ) -> usize {
        self.row_impl(Some(cache), anchors, rows)
    }

    /// Shared row query. `cache`/`anchors` are absent for direct cursor
    /// callers, which have no engine-level line index.
    fn row_impl(
        &self,
        cache: Option<&mut LineRing>,
        anchors: Option<&LineAnchors>,
        rows_hint: usize,
    ) -> usize {
        let position = self.pos.get().min(self.length.get());
        let raw = self.raw.get();
        let len = if raw.is_null() { 0 } else { self.raw_length.get() };
        let end = position.min(len);
        if Self::indexable(len)
            && let Some(cache) = cache
        {
            let rows = if rows_hint != 0 {
                Some(rows_hint)
            } else {
                anchors
                    .filter(|anchors| anchors.is_valid())
                    .map(|anchors| anchors.rows())
            };
            if let Some(row) = self.cached_row(cache, rows, end) {
                return row;
            }
            // Cold or distant cache: count from the nearest sparse
            // anchor, then recentre the ring on the caret's line so
            // following queries stay local.
            let (base_row, base_start) = anchors.map_or((0, 0), |anchors| anchors.block_for_position(end));
            // Safety: `raw[..len]` is the committed view bound by
            // `set_raw`/`rebind_raw`, `base_start <= end <= len`,
            // so the sub-slice stays in bounds.
            let row = base_row + unsafe { scan::count_newlines(raw.add(base_start), end - base_start) };
            Self::rebuild_line_ring(cache, raw, len, row, self.line_start(end));
            return row;
        }
        // Safety: `raw[..len]` is the committed view bound by
        // `set_raw`/`rebind_raw`, and `end <= len`.
        unsafe { scan::count_newlines(raw, end) }
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
        let (start, end) = self.line_bounds(position);
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
        let length = self.length.get();
        let (start, current_end) = self.line_bounds(position);
        let column = position - start;
        if self.is_newline(position) {
            // Caret sits on the newline: step over it, the next line
            // starts right after.
            let next_start = position + 1;
            let next_end = self.line_bounds(next_start).1;
            return self.place(next_start + column.min(next_end - next_start));
        }
        if current_end < length {
            // Mid-line: the next newline ends the current line.
            let next_start = current_end + 1;
            let next_end = self.line_bounds(next_start).1;
            self.place(next_start + column.min(next_end - next_start))
        } else {
            self.place(length)
        }
    }

    /// Moves the caret to the same column of the previous line.
    ///
    /// On the first line the caret stops at the line start.
    /// Returns the new position.
    pub fn move_prev_line(&self) -> usize {
        let position = self.pos.get().min(self.length.get());
        let (start, _) = self.line_bounds(position);
        let column = position - start;
        if start == 0 {
            return self.place(0);
        }
        // The previous line ends at `start - 1` (the newline that
        // separates it from the current line).
        let prev_end = start - 1;
        let (prev_start, _) = self.line_bounds(prev_end);
        self.place(prev_start + column.min(prev_end - prev_start))
    }

    /// Places the caret at `(row, col)`, clamped. Returns the
    /// resulting position.
    pub fn move_to(&self, row: usize, col: usize) -> usize {
        self.move_to_impl(row, col, None, None)
    }

    /// Places the caret at `(row, col)` using `cache`/`anchors` for row
    /// lookup.
    pub(crate) fn move_to_cached(
        &self,
        cache: &mut LineRing,
        anchors: &LineAnchors,
        row: usize,
        col: usize,
    ) -> usize {
        self.move_to_impl(row, col, Some(cache), Some(anchors))
    }

    fn move_to_impl(
        &self,
        row: usize,
        col: usize,
        cache: Option<&mut LineRing>,
        anchors: Option<&LineAnchors>,
    ) -> usize {
        let length = self.length.get();
        let raw = self.raw.get();
        let len = if raw.is_null() { 0 } else { self.raw_length.get() };
        // Rows are counted over the committed view; the staged tail
        // holds no newline, so a row past the last committed line parks
        // the caret at the document end.
        let mut index = None;
        if let Some(cache) = cache {
            index = cache
                .offset_of(row)
                .map(|offset| cache.at(offset));
            if index.is_none() && !cache.is_empty() {
                if row.checked_add(1) == Some(cache.first_row()) {
                    // One row before the window: the previous line start
                    // is the newline before the line preceding
                    // `first_row`.
                    let before = cache.at(0).saturating_sub(1);
                    // Safety: `raw[..len]` is the committed view bound
                    // by `set_raw`/`rebind_raw`.
                    let start =
                        unsafe { scan::find_newline_backward(raw, len, before) }.map_or(0, |nl| nl + 1);
                    cache.push_front(start as u32);
                    index = Some(start);
                } else if cache.last_row().checked_add(1) == Some(row) {
                    // One row after the window: the newline that ends
                    // the last cached line begins the next one.
                    let last = cache.at(cache.len() - 1);
                    // Safety: same committed-view contract as above.
                    if let Some(newline) = unsafe { scan::find_newline_forward(raw, len, last) } {
                        let start = newline + 1;
                        cache.push_back(start as u32);
                        index = Some(start);
                    }
                }
            }
        }
        let index = match index {
            Some(index) => index,
            None if row == 0 => 0,
            None => {
                // A valid sparse index bounds the scan to one anchor
                // block; otherwise fall back to the full scan.
                let start = match anchors.filter(|anchors| anchors.is_valid()) {
                    // Safety: same committed-view contract as above.
                    Some(anchors) => unsafe { anchors.row_start(raw, len, row) },
                    // Safety: same committed-view contract as above.
                    None => unsafe { scan::find_nth_newline(raw, len, row) }.map(|newline| newline + 1),
                };
                match start {
                    // Deliberately leave the ring untouched on a distant
                    // jump: random access would pay to build a window it
                    // never reuses. `row_cached`/`request_region_load`
                    // rebuild it when local queries need it.
                    Some(start) => start,
                    None => length,
                }
            }
        };
        if col == 0 {
            // Column zero is the line start whatever the line width is:
            // avoid scanning to the newline just to clamp against it.
            return self.place(index);
        }
        // The last (or a past-the-end) row has no newline after it, so
        // its end is the logical end. The sparse index already knows
        // the row count, which avoids a full forward scan on tail
        // navigation.
        let last_row =
            anchors.is_some_and(|anchors| anchors.is_valid() && row.saturating_add(1) >= anchors.rows());
        // A dense index knows every row start, so the next row's start
        // gives this row's end without touching the document text.
        let line_end = if last_row {
            length
        } else {
            match anchors.and_then(|anchors| anchors.row_end(row)) {
                Some(end) => end,
                None => self.line_end(index),
            }
        };
        self.place(index + col.min(line_end.saturating_sub(index)))
    }

    /// Preloads the engine's line ring around `row`.
    ///
    /// Random row/column work uses the sparse anchors to bound the
    /// scan to one block; the rebuilt ring then makes the following
    /// [`Self::move_to_cached`] and [`Self::row_cached`] queries in
    /// that region constant time. Returns `true` when the cache was
    /// primed, or `false` for a short document or a row past the last
    /// committed line.
    pub(crate) fn request_region_load_cached(
        &self,
        cache: &mut LineRing,
        anchors: &LineAnchors,
        row: usize,
    ) -> bool {
        let raw = self.raw.get();
        let len = if raw.is_null() { 0 } else { self.raw_length.get() };
        if !Self::indexable(len) {
            *cache = LineRing::empty();
            return false;
        }
        let start = if row == 0 {
            0
        } else {
            let start = if anchors.is_valid() {
                // Safety: `raw[..len]` is the committed view bound by
                // `set_raw`/`rebind_raw`.
                unsafe { anchors.row_start(raw, len, row) }
            } else {
                // Safety: same committed-view contract.
                unsafe { scan::find_nth_newline(raw, len, row) }.map(|newline| newline + 1)
            };
            let Some(start) = start else {
                return false;
            };
            start
        };
        Self::rebuild_line_ring(cache, raw, len, row, start);
        true
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
    use super::{BidiIndex, Cursor, LineAnchors, LineRing};

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

    #[test]
    fn line_ring_primes_extends_and_overwrites() {
        // 200 lines of 30 chars plus '\n': 6200 elements, well above
        // the cache threshold, with an empty row 200 after the trailing
        // newline.
        const LINES: usize = 200;
        const WIDTH: usize = 30;
        const STRIDE: usize = WIDTH + 1;
        let mut doc = alloc::vec::Vec::with_capacity(LINES * STRIDE);
        for line in 0..LINES {
            for column in 0..WIDTH {
                doc.push(ch(u32::from(b'a' + ((line + column) % 26) as u8)));
            }
            doc.push(ch('\n'));
        }
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        // SAFETY: `doc` stays alive for the whole cursor lifetime.
        unsafe { cursor.set_raw(doc.as_ptr(), doc.len()) };
        // SAFETY: same live `doc` view as the cursor.
        let anchors = unsafe { LineAnchors::build(doc.as_ptr(), doc.len()) };
        assert!(anchors.is_valid());

        // A cold prime finds a distant row and makes its region local.
        let mut cache = LineRing::empty();
        assert!(cursor.request_region_load_cached(&mut cache, &anchors, 100));
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 100, 5),
            100 * STRIDE + 5
        );
        assert_eq!(cursor.row_cached(&mut cache, Some(&anchors), anchors.rows()), 100);
        assert_eq!(cursor.col(), 5);

        // Overwriting the ring with another region discards the first.
        assert!(cursor.request_region_load_cached(&mut cache, &anchors, 10));
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 10, 2),
            10 * STRIDE + 2
        );
        assert_eq!(cursor.row_cached(&mut cache, Some(&anchors), anchors.rows()), 10);

        // Adjacent moves extend the window without a full rescan.
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 11, 1),
            11 * STRIDE + 1
        );
        assert_eq!(cursor.row_cached(&mut cache, Some(&anchors), anchors.rows()), 11);
        assert_eq!(cursor.move_to_cached(&mut cache, &anchors, 9, 4), 9 * STRIDE + 4);
        assert_eq!(cursor.row_cached(&mut cache, Some(&anchors), anchors.rows()), 9);

        // A jump outside the window falls back, rebuilds, and remains
        // correct at the document tail.
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 150, 3),
            150 * STRIDE + 3
        );
        assert_eq!(cursor.row_cached(&mut cache, Some(&anchors), anchors.rows()), 150);
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 200, 0),
            LINES * STRIDE
        );
        assert_eq!(cursor.row_cached(&mut cache, Some(&anchors), anchors.rows()), 200);
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 201, 9),
            LINES * STRIDE
        );

        // Rows past the committed layout are not cacheable.
        assert!(!cursor.request_region_load_cached(&mut cache, &anchors, 10_000));
    }

    #[test]
    fn dense_anchors_resolve_rows_and_line_ends() {
        const LINES: usize = 64;
        const WIDTH: usize = 12;
        const STRIDE: usize = WIDTH + 1;
        let mut doc = alloc::vec::Vec::with_capacity(LINES * STRIDE);
        for line in 0..LINES {
            for column in 0..WIDTH {
                doc.push(ch(u32::from(b'a' + ((line + column) % 26) as u8)));
            }
            doc.push(ch('\n'));
        }
        let cursor: Cursor<u32, u32, 16> = Cursor::new();
        // SAFETY: `doc` stays alive for the whole cursor lifetime.
        unsafe { cursor.set_raw(doc.as_ptr(), doc.len()) };
        // SAFETY: same live `doc` view as the cursor.
        let anchors = unsafe { LineAnchors::build(doc.as_ptr(), doc.len()) };
        assert!(anchors.is_valid());
        // The index is dense: one anchor per row, so a jump resolves
        // without reading the document text.
        assert_eq!(anchors.stride, 1);
        assert_eq!(anchors.len, (LINES + 1) as u16);
        assert_eq!(anchors.rows(), LINES + 1);
        for row in 0..LINES {
            assert_eq!(anchors.starts[row] as usize, row * STRIDE);
        }
        // The line end is the newline before the next row start.
        for row in 0..LINES {
            assert_eq!(anchors.row_end(row), Some(row * STRIDE + WIDTH));
        }
        // A column past the line end is clamped to it.
        let mut cache = LineRing::empty();
        assert_eq!(
            cursor.move_to_cached(&mut cache, &anchors, 5, 100),
            5 * STRIDE + WIDTH
        );
    }

    #[test]
    fn shifted_anchor_starts_match_a_fresh_build() {
        const ROWS: usize = 300;
        const WIDTH: usize = 20;
        const STRIDE: usize = WIDTH + 1;
        let mut doc = alloc::vec::Vec::with_capacity(ROWS * STRIDE);
        for line in 0..ROWS {
            for column in 0..WIDTH {
                doc.push(ch(u32::from(b'a' + ((line + column) % 26) as u8)));
            }
            doc.push(ch('\n'));
        }
        // SAFETY: `doc` outlives the index and the edited copies below.
        let mut anchors = unsafe { LineAnchors::build(doc.as_ptr(), doc.len()) };
        assert!(anchors.is_valid());

        // A newline-free insertion in the middle shifts every start
        // after it; resolving any row must agree with a fresh build.
        let mut inserted = doc.clone();
        inserted.splice(50..50, [ch('x'), ch('y')]);
        anchors.shift_after_insert(50, 2);
        assert_eq!(anchors.rows(), ROWS + 1);
        let rebuilt = unsafe { LineAnchors::build(inserted.as_ptr(), inserted.len()) };
        for row in 0..=ROWS {
            assert_eq!(
                unsafe { anchors.row_start(inserted.as_ptr(), inserted.len(), row) },
                unsafe { rebuilt.row_start(inserted.as_ptr(), inserted.len(), row) },
                "row {row} after insertion"
            );
        }

        // A newline-free removal shifts later starts left.
        let position = 100 * STRIDE + 3;
        let mut removed = inserted.clone();
        removed.drain(position..position + 5);
        anchors.shift_after_remove(position, 5);
        let rebuilt = unsafe { LineAnchors::build(removed.as_ptr(), removed.len()) };
        for row in 0..=ROWS {
            assert_eq!(
                unsafe { anchors.row_start(removed.as_ptr(), removed.len(), row) },
                unsafe { rebuilt.row_start(removed.as_ptr(), removed.len(), row) },
                "row {row} after removal"
            );
        }
    }
}
