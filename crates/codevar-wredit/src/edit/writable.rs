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

//! Gap buffers, events and the writable core structures.
//!
//! A [`WritableGap`] is a classic gap buffer slot: a lazily
//! allocated window of `GAP_SIZE` elements into which a cursor
//! stages edits with O(1) writes. [`WritableGaps`] keeps a *fixed*
//! inline array of at most [`MAX_GAPS`] such slots — no arena, no
//! eager preallocation. [`WritableGapTable`](SharedGapTable)-based
//! shared gaps live in [`BaseWritable::shared_gaps`] and stage into
//! the lazily grown shared arena [`BaseWritable::buffer`].
//!
//! [`BaseWritable`] is the whole inline working state of the engine:
//! raw document, staging arena, gaps, shared gap table, cursors and
//! the event bridge. [`StreamWritable`] is the facade the operation
//! layer implements the `TextEditable*` traits on.

use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::ptr::NonNull;

use codevar_logger::log_error;

use super::base::{ENC_UNKNOWN, NAME_LEN, WritableAlignedPtr, ensure_merge_buffer};
use super::cursor::Cursor;
use super::shared_gap::SharedGapTable;
use crate::error::WreditError;

/// Maximum number of private (per-cursor) gaps held inline.
pub const MAX_GAPS: usize = 0x4;

/// Maximum number of cursors a writable can hold.
pub const MAX_CURSORS: usize = 0x1000;

/// Number of shared gap slot descriptors in the fixed table.
pub const SHARED_GAP_SLOTS: usize = 0x10;

/// `cursor_to_gap` sentinel: the cursor owns no gap.
pub const GAP_NONE: u16 = u16::MAX;

/// Alignment of the raw document and shared arena allocations.
pub(crate) const BUFFER_ALIGN: usize = 0x40;

/// A lazily allocated heap window of `GAP_SIZE` elements.
///
/// `gap_ptr` counts the staged elements (the offset of the write
/// cursor from the window base); the window itself is allocated on
/// the first [`WritableGap::push`] and released on drop.
pub struct WritableGapPtr<T: Copy, const GAP_SIZE: usize> {
    /// Offset of the next write position from the window base.
    gap_ptr: usize,
    /// The heap window; `None` until first use.
    buff: Option<NonNull<T>>,
}

impl<T: Copy, const GAP_SIZE: usize> WritableGapPtr<T, GAP_SIZE> {
    /// Creates an unallocated window.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gap_ptr: 0,
            buff: None,
        }
    }

    /// `true` once the window has been allocated.
    #[must_use]
    pub const fn is_allocated(&self) -> bool {
        self.buff.is_some()
    }

    /// Current staged-element count (write offset).
    #[must_use]
    pub const fn gap_ptr(&self) -> usize {
        self.gap_ptr
    }

    /// Sets the write offset, clamped to the window size.
    ///
    /// Returns the clamped offset.
    pub fn set_gap_ptr(&mut self, offset: usize) -> usize {
        self.gap_ptr = offset.min(GAP_SIZE);
        self.gap_ptr
    }

    /// Base pointer of the window, or a dangling pointer while
    /// unallocated.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        match self.buff {
            Some(ptr) => ptr.as_ptr(),
            None => NonNull::dangling().as_ptr(),
        }
    }

    /// Mutable base pointer of the window, or a dangling pointer
    /// while unallocated.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self.buff {
            Some(ptr) => ptr.as_ptr(),
            None => NonNull::dangling().as_ptr(),
        }
    }

    /// Allocates the `GAP_SIZE`-element window on the heap.
    ///
    /// Returns `true` on success (or when already allocated). A
    /// failed allocation is logged and reported as `false`; the gap
    /// simply refuses writes until memory becomes available.
    pub fn allocate(&mut self) -> bool {
        if self.buff.is_some() {
            return true;
        }
        match ensure_merge_buffer::<T>(GAP_SIZE, BUFFER_ALIGN) {
            Ok(ptr) => {
                self.buff = Some(ptr);
                true
            }
            Err(error) => {
                log_error!("wredit: gap window allocation failed: {error}");
                false
            }
        }
    }

    /// Releases the window (if any) and resets the write offset.
    pub fn free(&mut self) {
        self.gap_ptr = 0;
        if let Some(ptr) = self.buff.take() {
            // SAFETY: `buff` was allocated by `ensure_merge_buffer`
            // with `Layout::array::<T>(GAP_SIZE)` and the buffer
            // alignment; it is owned exclusively by this window.
            if let Ok(layout) = alloc::alloc::Layout::array::<T>(GAP_SIZE) {
                unsafe {
                    alloc::alloc::dealloc(ptr.as_ptr().cast::<u8>(), layout);
                }
            }
        }
    }
}

impl<T: Copy, const GAP_SIZE: usize> Default for WritableGapPtr<T, GAP_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const GAP_SIZE: usize> Drop for WritableGapPtr<T, GAP_SIZE> {
    fn drop(&mut self) {
        self.free();
    }
}

/// One private staging gap: a gap buffer plus the logical position
/// its staged content will be committed at.
///
/// `gap_start` points at the next write slot, `gap_end` one past the
/// window end; `base` is the raw position the staged run belongs to
/// when it is flushed.
pub struct WritableGap<T: Copy, const GAP_SIZE: usize> {
    /// Next write position inside the window.
    gap_start: *mut T,
    /// One past the window end.
    gap_end: *mut T,
    /// The lazily allocated window.
    gap_ptr: WritableGapPtr<T, GAP_SIZE>,
    /// Raw position this gap's staged run commits at.
    base: usize,
}

impl<T: Copy, const GAP_SIZE: usize> WritableGap<T, GAP_SIZE> {
    /// Creates an empty, unallocated gap committing at `base`.
    #[must_use]
    pub fn at(base: usize) -> Self {
        Self {
            gap_start: core::ptr::null_mut(),
            gap_end: core::ptr::null_mut(),
            gap_ptr: WritableGapPtr::new(),
            base,
        }
    }

    /// Raw position this gap commits at.
    #[must_use]
    pub const fn base(&self) -> usize {
        self.base
    }

    /// Updates the commit position.
    pub fn set_base(&mut self, base: usize) {
        self.base = base;
    }

    /// Number of staged elements.
    #[must_use]
    pub const fn staged(&self) -> usize {
        self.gap_ptr.gap_ptr()
    }

    /// `true` when nothing is staged.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.gap_ptr.gap_ptr() == 0
    }

    /// `true` when no more elements fit.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.gap_ptr.gap_ptr() >= GAP_SIZE
    }

    /// Pointer to the staged elements (only valid while
    /// `staged() > 0`).
    #[must_use]
    pub fn content(&self) -> *const T {
        self.gap_ptr.as_ptr()
    }

    /// Stages one element. Returns `false` when the gap is full or
    /// its window could not be allocated.
    pub fn push(&mut self, value: T) -> bool {
        let staged = self.gap_ptr.gap_ptr();
        if staged >= GAP_SIZE {
            return false;
        }
        if !self.gap_ptr.is_allocated() && !self.gap_ptr.allocate() {
            return false;
        }
        let base = self.gap_ptr.as_mut_ptr();
        // SAFETY: the window holds `GAP_SIZE` elements and
        // `staged < GAP_SIZE`, so `base.add(staged)` is in bounds and
        // aligned for `T`; `&mut self` guarantees exclusivity.
        unsafe {
            base.add(staged).write(value);
        }
        self.gap_start = unsafe { base.add(staged + 1) };
        self.gap_end = unsafe { base.add(GAP_SIZE) };
        self.gap_ptr.set_gap_ptr(staged + 1);
        true
    }

    /// Drops the staged elements, keeping the window allocated.
    pub fn clear(&mut self) {
        self.gap_ptr.set_gap_ptr(0);
        if self.gap_ptr.is_allocated() {
            let base = self.gap_ptr.as_mut_ptr();
            self.gap_start = base;
            // SAFETY: `base` points at the allocated window start.
            self.gap_end = unsafe { base.add(GAP_SIZE) };
        } else {
            self.gap_start = core::ptr::null_mut();
            self.gap_end = core::ptr::null_mut();
        }
    }
}

/// The fixed inline set of private gaps.
///
/// Slots are `Option`s: an absent slot is a free gap. Capacity is
/// [`MAX_GAPS`]; buffers are allocated per slot on first push, never
/// eagerly.
pub struct WritableGaps<Raw: Copy, const GAP_SIZE: usize> {
    /// The fixed slot array.
    gaps: [Option<WritableGap<Raw, GAP_SIZE>>; MAX_GAPS],
}

impl<Raw: Copy, const GAP_SIZE: usize> WritableGaps<Raw, GAP_SIZE> {
    /// Creates an empty set: every slot free.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gaps: core::array::from_fn(|_| None),
        }
    }

    /// Number of occupied slots.
    #[must_use]
    pub fn count(&self) -> usize {
        self.gaps
            .iter()
            .filter(|gap| gap.is_some())
            .count()
    }

    /// `true` when every slot is occupied.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.count() >= MAX_GAPS
    }

    /// Occupies the first free slot as a gap committing at `base`.
    ///
    /// Returns the slot index, or `None` when every slot is taken.
    pub fn alloc(&mut self, base: usize) -> Option<usize> {
        for (index, slot) in self.gaps.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(WritableGap::at(base));
                return Some(index);
            }
        }
        None
    }

    /// Releases the slot at `index`, dropping its staged elements
    /// and freeing its window.
    ///
    /// Returns `true` when a slot was freed.
    pub fn free(&mut self, index: usize) -> bool {
        match self.gaps.get_mut(index) {
            Some(slot) => slot.take().is_some(),
            None => false,
        }
    }

    /// The gap occupying `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&WritableGap<Raw, GAP_SIZE>> {
        self.gaps.get(index).and_then(Option::as_ref)
    }

    /// The gap occupying `index`, mutably.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut WritableGap<Raw, GAP_SIZE>> {
        self.gaps.get_mut(index).and_then(Option::as_mut)
    }

    /// Indices of every occupied slot, ascending.
    pub fn occupied(&self) -> impl Iterator<Item = usize> {
        self.gaps
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.as_ref().map(|_| index))
    }
}

impl<Raw: Copy, const GAP_SIZE: usize> Default for WritableGaps<Raw, GAP_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// The kind of a [`WritableEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritableEventType {
    /// The raw buffer changed at `start` for `end` elements.
    BufferChanged,
    /// A cursor moved from `start` to `end`.
    CursorMoved,
    /// A cursor's selection changed.
    CursorSelectionChanged,
    /// A cursor was added (id in `cursor`).
    CursorAdded,
    /// A cursor was removed (id in `cursor`).
    CursorRemoved,
    /// An operation failed; details in `error`.
    Error,
}

/// An engine event delivered through an [`EventBridge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritableEvent {
    /// What happened.
    pub kind: WritableEventType,
    /// The cursor the event refers to ([`MAX_CURSORS`] for
    /// document-level events).
    pub cursor: usize,
    /// Primary position payload (start of the change, old caret, ..).
    pub start: usize,
    /// Secondary position payload (element count, new caret, ..).
    pub end: usize,
    /// The error, for [`WritableEventType::Error`] events.
    pub error: Option<WreditError>,
}

impl WritableEvent {
    /// A buffer change of `len` elements at `start`.
    #[must_use]
    pub const fn buffer_changed(start: usize, len: usize) -> Self {
        Self {
            kind: WritableEventType::BufferChanged,
            cursor: MAX_CURSORS,
            start,
            end: len,
            error: None,
        }
    }

    /// A cursor move from `from` to `to`.
    #[must_use]
    pub const fn cursor_moved(cursor: usize, from: usize, to: usize) -> Self {
        Self {
            kind: WritableEventType::CursorMoved,
            cursor,
            start: from,
            end: to,
            error: None,
        }
    }

    /// A selection change on `cursor`.
    #[must_use]
    pub const fn selection_changed(cursor: usize) -> Self {
        Self {
            kind: WritableEventType::CursorSelectionChanged,
            cursor,
            start: 0,
            end: 0,
            error: None,
        }
    }

    /// A cursor was added or removed.
    #[must_use]
    pub const fn cursor_lifecycle(kind: WritableEventType, cursor: usize) -> Self {
        Self {
            kind,
            cursor,
            start: 0,
            end: 0,
            error: None,
        }
    }

    /// An error event carrying `error` for `cursor`.
    #[must_use]
    pub const fn error(cursor: usize, error: WreditError) -> Self {
        Self {
            kind: WritableEventType::Error,
            cursor,
            start: 0,
            end: 0,
            error: Some(error),
        }
    }
}

/// Receiver of [`WritableEvent`]s emitted by the engine.
///
/// Bridges must be thread-safe because a writable may be driven from
/// several threads in future embeddings; the engine itself stays
/// single-threaded per instance.
pub trait EventBridge: Send + Sync {
    /// Delivers one engine event.
    fn emit(&self, event: WritableEvent);
}

/// The inline working state of the Wredit engine.
///
/// Everything here is bounded: the raw document and the shared
/// staging arena are sparse heap buffers (empty until first use),
/// the private gaps are a fixed array of [`MAX_GAPS`] lazily
/// allocated windows, the shared gap descriptors are a fixed table of
/// [`SHARED_GAP_SLOTS`] 8-byte slots, and the cursors are a fixed
/// array of [`MAX_CURSORS`]. Together the structure stays well below
/// the engine's 4 KiB inline budget.
///
/// # Type parameters
///
/// * `Raw`      - packed raw element type (`u32` in practice)
/// * `Buf`      - element type of the shared staging arena
/// * `GAP_SIZE` - private/shared gap window size in elements
pub struct BaseWritable<Raw: Copy, Buf: Copy, const GAP_SIZE: usize> {
    /// The packed raw document; grows geometrically.
    pub(crate) raw: UnsafeCell<WritableAlignedPtr<Raw, 64>>,
    /// Shared staging arena for shared gaps; grows geometrically.
    pub(crate) buffer: UnsafeCell<WritableAlignedPtr<Buf, 64>>,
    /// Fixed inline array of private gaps.
    pub(crate) gaps: UnsafeCell<WritableGaps<Raw, GAP_SIZE>>,
    /// Fixed table of shared gap descriptors.
    pub(crate) shared_gaps: UnsafeCell<SharedGapTable<SHARED_GAP_SLOTS>>,
    /// Cursor -> gap mapping; [`GAP_NONE`] when the cursor owns no
    /// gap, `0..MAX_GAPS` for private gaps, `MAX_GAPS + slot` for
    /// shared gaps.
    pub(crate) cursor_to_gap: [u16; MAX_CURSORS],
    /// The cursor pool.
    pub(crate) cursors: UnsafeCell<[Option<Cursor<Raw, Buf, GAP_SIZE>>; MAX_CURSORS]>,
    /// Id of the cursor driving single-cursor operations.
    pub(crate) active_cursor_id: usize,
    /// Optional event receiver.
    pub(crate) event_bridge: UnsafeCell<Option<Arc<dyn EventBridge>>>,
    /// Document name (raw bytes, NUL padded).
    pub(crate) name: [u8; NAME_LEN],
    /// User-defined flag bits.
    pub(crate) flags: u32,
    /// Number of committed elements in `raw`.
    pub(crate) raw_length: usize,
    /// Committed capacity head-room (mirrors `raw.allocated()`).
    pub(crate) vspace: usize,
    /// Document encoding, one of the `ENC_*` constants.
    pub(crate) encoding: u16,
}

impl<Raw: Copy, Buf: Copy, const GAP_SIZE: usize> BaseWritable<Raw, Buf, GAP_SIZE> {
    /// Creates an empty writable: no allocation happens here.
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw: UnsafeCell::new(WritableAlignedPtr::new()),
            buffer: UnsafeCell::new(WritableAlignedPtr::new()),
            gaps: UnsafeCell::new(WritableGaps::new()),
            shared_gaps: UnsafeCell::new(SharedGapTable::new()),
            cursor_to_gap: [GAP_NONE; MAX_CURSORS],
            cursors: UnsafeCell::new(core::array::from_fn(|_| None)),
            active_cursor_id: 0,
            event_bridge: UnsafeCell::new(None),
            name: [0; NAME_LEN],
            flags: 0,
            raw_length: 0,
            vspace: 0,
            encoding: ENC_UNKNOWN,
        }
    }

    /// Creates a writable whose raw document pre-grows to hold at
    /// least `capacity` committed elements.
    ///
    /// The allocation is still geometric; a failed growth is logged
    /// and yields an empty (but usable) writable rather than an
    /// abort.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut writable = Self::new();
        if capacity > 0 {
            let raw = unsafe { &mut *writable.raw.get() };
            if raw.ensure_capacity(capacity).is_err() {
                log_error!("wredit: initial raw capacity allocation failed");
            }
            writable.vspace = raw.allocated();
        }
        writable
    }

    /// The raw view (shared reference, single-thread contract).
    #[must_use]
    #[inline]
    pub(crate) fn raw_ref(&self) -> &WritableAlignedPtr<Raw, 64> {
        // SAFETY: `UnsafeCell` interior access; the writable is
        // `!Sync`, so no concurrent access can race this borrow.
        unsafe { &*self.raw.get() }
    }

    /// The raw view (mutable reference).
    #[must_use]
    #[inline]
    pub(crate) fn raw_mut(&mut self) -> &mut WritableAlignedPtr<Raw, 64> {
        self.raw.get_mut()
    }

    /// The shared staging arena (mutable reference).
    #[must_use]
    #[inline]
    pub(crate) fn buffer_mut(&mut self) -> &mut WritableAlignedPtr<Buf, 64> {
        self.buffer.get_mut()
    }

    /// The shared staging arena (shared reference, single-thread contract).
    #[must_use]
    #[inline]
    pub(crate) fn buffer_ref(&self) -> &WritableAlignedPtr<Buf, 64> {
        // SAFETY: `UnsafeCell` interior access; `!Sync` writable.
        unsafe { &*self.buffer.get() }
    }

    /// The private gaps (mutable reference).
    #[must_use]
    #[inline]
    pub(crate) fn gaps_mut(&mut self) -> &mut WritableGaps<Raw, GAP_SIZE> {
        self.gaps.get_mut()
    }

    /// The private gaps (shared reference, single-thread contract).
    #[must_use]
    #[inline]
    pub(crate) fn gaps_ref(&self) -> &WritableGaps<Raw, GAP_SIZE> {
        // SAFETY: `UnsafeCell` interior access; `!Sync` writable.
        unsafe { &*self.gaps.get() }
    }

    /// The shared gap table (mutable reference).
    #[must_use]
    #[inline]
    pub(crate) fn shared_mut(&mut self) -> &mut SharedGapTable<SHARED_GAP_SLOTS> {
        self.shared_gaps.get_mut()
    }

    /// The shared gap table (shared reference, single-thread contract).
    #[must_use]
    #[inline]
    pub(crate) fn shared_ref(&self) -> &SharedGapTable<SHARED_GAP_SLOTS> {
        // SAFETY: `UnsafeCell` interior access; `!Sync` writable.
        unsafe { &*self.shared_gaps.get() }
    }

    /// The cursor pool (shared reference, single-thread contract).
    #[must_use]
    #[inline]
    pub(crate) fn cursors_ref(&self) -> &[Option<Cursor<Raw, Buf, GAP_SIZE>>; MAX_CURSORS] {
        // SAFETY: `UnsafeCell` interior access; `!Sync` writable.
        unsafe { &*self.cursors.get() }
    }

    /// The cursor pool (mutable reference).
    #[must_use]
    #[inline]
    pub(crate) fn cursors_mut(&mut self) -> &mut [Option<Cursor<Raw, Buf, GAP_SIZE>>; MAX_CURSORS] {
        self.cursors.get_mut()
    }

    /// `true` when `cursor_id` addresses a live cursor.
    #[must_use]
    pub(crate) fn valid_cursor_id(&self, cursor_id: usize) -> bool {
        cursor_id < MAX_CURSORS && self.cursors_ref()[cursor_id].is_some()
    }

    /// The cursor with `cursor_id`.
    #[must_use]
    pub(crate) fn cursor(&self, cursor_id: usize) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        self.cursors_ref()
            .get(cursor_id)
            .and_then(Option::as_ref)
    }

    /// Rebinds every cursor's raw view to the current document.
    ///
    /// Must be called after any operation that may reallocate or
    /// resize `raw`, otherwise cursors keep dangling pointers into
    /// the old allocation. Logical positions survive the rebind:
    /// only the dereference bound and the raw pointer change.
    pub(crate) fn refresh_cursor_views(&mut self) {
        let ptr = self.raw_ref().as_ptr();
        let raw_length = self.raw_length;
        let logical = self.logical_length();
        for cursor in self.cursors_mut().iter().flatten() {
            // SAFETY: `ptr`/`raw_length` describe the current raw
            // buffer, which outlives the cursors (both live in this
            // struct), and `logical >= raw_length` keeps the
            // dereference bound valid.
            unsafe { cursor.rebind_raw(ptr, raw_length) };
            cursor.set_logical_length(logical);
        }
    }

    /// The document encoding, one of the `ENC_*` constants.
    #[must_use]
    #[inline]
    pub fn encoding(&self) -> u16 {
        self.encoding
    }

    /// Logical document length: committed elements plus every staged
    /// element still sitting in a gap.
    #[must_use]
    pub(crate) fn logical_length(&self) -> usize {
        let mut total = self.raw_length;
        for index in self.gaps_ref().occupied() {
            if let Some(gap) = self.gaps_ref().get(index) {
                total = total.saturating_add(gap.staged());
            }
        }
        let shared = self.shared_ref();
        let mut slot = 0;
        while slot < SHARED_GAP_SLOTS {
            if let Some(descriptor) = shared.slot(slot)
                && descriptor.is_allocated()
            {
                total = total.saturating_add(descriptor.size());
            }
            slot += 1;
        }
        total
    }

    /// Shifts every cursor position at or after `position` by
    /// `delta` elements (saturating at zero and at the logical end).
    ///
    /// When `removing`, positions inside `(position, position + delta]`
    /// collapse onto `position` and everything further right shifts
    /// left; otherwise positions `>= position` shift right.
    pub(crate) fn shift_cursors(&mut self, position: usize, delta: usize, removing: bool) {
        let logical_end = self.logical_length();
        let end = position.saturating_add(delta);
        for cursor in self.cursors_mut().iter().flatten() {
            let pos = cursor.array_pos();
            let new_pos = if removing {
                if pos <= position {
                    pos
                } else if pos <= end {
                    position
                } else {
                    pos - delta
                }
            } else if pos > position {
                pos.saturating_add(delta).min(logical_end)
            } else {
                pos
            };
            cursor.set_array_pos(new_pos);
            let anchor = cursor.anchor();
            let new_anchor = if removing {
                if anchor <= position {
                    anchor
                } else if anchor <= end {
                    position
                } else {
                    anchor - delta
                }
            } else if anchor > position {
                anchor.saturating_add(delta).min(logical_end)
            } else {
                anchor
            };
            cursor.set_anchor(new_anchor);
        }
    }

    /// Shifts the commit base of every private gap past `position`.
    pub(crate) fn shift_gap_bases(&mut self, position: usize, delta: usize, removing: bool) {
        let gaps = self.gaps_mut();
        let indices: [usize; MAX_GAPS] = core::array::from_fn(|i| i);
        for index in indices {
            let Some(gap) = gaps.get_mut(index) else {
                continue;
            };
            let base = gap.base();
            if removing {
                if base > position {
                    if base <= position.saturating_add(delta) {
                        gap.set_base(position);
                    } else {
                        gap.set_base(base - delta);
                    }
                }
            } else if base >= position {
                gap.set_base(base.saturating_add(delta));
            }
        }
    }

    /// Shifts the region base of every shared gap past `position`.
    pub(crate) fn shift_shared_bases(&mut self, position: usize, delta: usize, removing: bool) {
        let shared = self.shared_mut();
        let mut slot = 0;
        while slot < SHARED_GAP_SLOTS {
            let Some(descriptor) = shared.slot(slot) else {
                slot += 1;
                continue;
            };
            if descriptor.is_allocated() {
                let base = descriptor.base();
                let new_base = if removing {
                    if base <= position {
                        base
                    } else if base <= position.saturating_add(delta) {
                        position
                    } else {
                        base - delta
                    }
                } else if base >= position {
                    base.saturating_add(delta)
                } else {
                    base
                };
                shared.set_region(slot, new_base, descriptor.size());
            }
            slot += 1;
        }
    }

    /// Emits `event` through the bridge, if one is installed.
    pub(crate) fn emit(&self, event: WritableEvent) {
        let bridge = {
            // SAFETY: single-thread contract; the borrow ends before
            // the (potentially re-entrant) delivery below.
            let slot = unsafe { &*self.event_bridge.get() };
            slot.clone()
        };
        if let Some(bridge) = bridge {
            bridge.emit(event);
        }
    }
}

impl<Raw: Copy, Buf: Copy, const GAP_SIZE: usize> Default for BaseWritable<Raw, Buf, GAP_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// The stream facade over a [`BaseWritable`].
///
/// Operation implementations live in [`super::ops`]; packed-element
/// helpers and flush wrappers live next to the base structure.
pub struct StreamWritable<Raw: Copy, Buf: Copy, const GAP_SIZE: usize> {
    /// The inline engine state.
    pub(crate) base: BaseWritable<Raw, Buf, GAP_SIZE>,
}

impl<Raw: Copy, Buf: Copy, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE> {
    /// Creates an empty stream writable.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseWritable::new(),
        }
    }

    /// Creates a stream writable with pre-grown raw capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            base: BaseWritable::with_capacity(capacity),
        }
    }

    /// The wrapped base structure.
    #[must_use]
    pub const fn base(&self) -> &BaseWritable<Raw, Buf, GAP_SIZE> {
        &self.base
    }

    /// The wrapped base structure, mutably.
    #[must_use]
    pub fn base_mut(&mut self) -> &mut BaseWritable<Raw, Buf, GAP_SIZE> {
        &mut self.base
    }
}

impl<Raw: Copy, Buf: Copy, const GAP_SIZE: usize> Default for StreamWritable<Raw, Buf, GAP_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{BaseWritable, MAX_CURSORS, MAX_GAPS, WritableEvent, WritableGapPtr, WritableGaps};
    use crate::edit::base::{T_COMMON, make_packed};
    use crate::edit::writable::WritableGap;

    // Justification: `unwrap` in tests is permitted by the project
    // lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn gap_ptr_allocates_lazily_and_frees() {
        let mut window: WritableGapPtr<u32, 32> = WritableGapPtr::new();
        assert!(!window.is_allocated());
        assert_eq!(window.gap_ptr(), 0);
        assert!(window.allocate());
        assert!(window.is_allocated());
        assert_eq!(window.set_gap_ptr(99), 32); // clamped
        window.free();
        assert!(!window.is_allocated());
    }

    #[test]
    fn gap_pushes_and_clears() {
        let mut gap: WritableGap<u32, 4> = WritableGap::at(10);
        assert_eq!(gap.base(), 10);
        assert!(gap.is_empty());
        assert!(gap.push(make_packed(1, T_COMMON)));
        assert!(gap.push(make_packed(2, T_COMMON)));
        assert_eq!(gap.staged(), 2);
        // The staged content is readable through `content`.
        let staged = unsafe { core::slice::from_raw_parts(gap.content(), gap.staged()) };
        assert_eq!(staged[1], make_packed(2, T_COMMON));
        gap.clear();
        assert!(gap.is_empty());
    }

    #[test]
    fn gap_rejects_pushes_past_its_window() {
        let mut gap: WritableGap<u32, 2> = WritableGap::at(0);
        assert!(gap.push(1));
        assert!(gap.push(2));
        assert!(gap.is_full());
        assert!(!gap.push(3));
        assert_eq!(gap.staged(), 2);
    }

    #[test]
    fn gaps_allocate_free_and_reuse_slots() {
        let mut gaps: WritableGaps<u32, 8> = WritableGaps::new();
        assert_eq!(gaps.count(), 0);
        assert_eq!(gaps.alloc(5), Some(0));
        assert_eq!(gaps.alloc(9), Some(1));
        assert_eq!(gaps.count(), 2);
        assert!(gaps.get(0).is_some_and(|gap| gap.base() == 5));
        assert!(gaps.free(0));
        assert!(!gaps.free(0));
        assert!(!gaps.free(MAX_GAPS + 3));
        assert_eq!(gaps.alloc(7), Some(0)); // lowest free slot reused
        assert!(gaps.get_mut(1).is_some_and(|gap| {
            gap.set_base(11);
            gap.base() == 11
        }));
        let occupied: alloc::vec::Vec<usize> = gaps.occupied().collect();
        assert_eq!(occupied, alloc::vec![0, 1]);
    }

    #[test]
    fn base_writable_starts_empty_without_allocations() {
        let base: BaseWritable<u32, u32, 32> = BaseWritable::new();
        assert_eq!(base.raw_ref().allocated(), 0);
        assert_eq!(base.raw_ref().count(), 0);
        assert_eq!(base.raw_length, 0);
        assert_eq!(base.logical_length(), 0);
        assert_eq!(base.vspace, 0);
        assert_eq!(base.active_cursor_id, 0);
        assert_eq!(base.encoding, crate::edit::base::ENC_UNKNOWN);
    }

    #[test]
    fn with_capacity_grows_the_raw_buffer_only() {
        let base: BaseWritable<u32, u32, 32> = BaseWritable::with_capacity(100);
        assert!(base.raw_ref().allocated() >= 100);
        assert_eq!(base.vspace, base.raw_ref().allocated());
        // The staging arena stays unallocated.
        assert_eq!(unsafe { &*base.buffer.get() }.allocated(), 0);
    }

    #[test]
    fn events_carry_their_payloads() {
        let changed = WritableEvent::buffer_changed(3, 7);
        assert_eq!(changed.start, 3);
        assert_eq!(changed.end, 7);
        let error = WritableEvent::error(2, crate::error::WreditError::InvalidCursor { cursor: 2 });
        assert!(error.error.is_some());
    }

    #[test]
    fn the_working_state_fits_the_inline_budget() {
        // The whole engine core (raw + arena + gaps + table + cursors
        // + metadata) must stay at or below 4096 bytes inline.
        assert!(
            core::mem::size_of::<BaseWritable<u32, u32, 128>>() <= 4096,
            "BaseWritable exceeds the 4096 byte inline budget: {} bytes",
            core::mem::size_of::<BaseWritable<u32, u32, 128>>()
        );
        // Cursor ids are bounded by the fixed pool.
        assert!(MAX_CURSORS <= u16::MAX as usize);
    }
}
