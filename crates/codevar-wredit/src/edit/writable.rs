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

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::cell::{Cell, UnsafeCell};
use core::ptr::NonNull;

use codevar_logger::log_error;

use super::base::{ENC_UNKNOWN, NAME_LEN, WritableAlignedPtr, ensure_merge_buffer};
use super::cursor::{Cursor, LineAnchors, LineRing};
use super::shared_gap::SharedGapTable;
use crate::error::WreditError;

/// Maximum number of private (per-cursor) gaps held inline.
pub const MAX_GAPS: usize = 0x4;

/// Maximum number of cursors a writable can hold.
///
/// The pool is a *fixed inline array*, so this bound is what keeps
/// [`BaseWritable`] inside the engine's 4 KiB inline budget: each
/// cursor plus its `cursor_to_gap` slot costs about 90 bytes, so the
/// budget only funds a few dozen entries (the test
/// `the_working_state_fits_the_inline_budget` pins the arithmetic).
/// The value also matches
/// [`SLOT_CURSOR_BITS`](super::shared_gap::SLOT_CURSOR_BITS), the width
/// of a shared gap's membership mask, so every cursor id this pool can
/// hand out is also addressable by a shared gap.
pub const MAX_CURSORS: usize = 0x10;

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
/// The slot array is allocated once, inline; occupancy is tracked by
/// a small bitmask and the windows themselves are *retained* across
/// `free`, so an interactive typing cycle (stage, flush, stage again)
/// does not allocate or free on every keystroke. Capacity is
/// [`MAX_GAPS`]; each window is still allocated lazily on first push.
pub struct WritableGaps<Raw: Copy, const GAP_SIZE: usize> {
    /// The fixed slot array; a slot keeps its window after release.
    gaps: [WritableGap<Raw, GAP_SIZE>; MAX_GAPS],
    /// Bit `i` is set while slot `i` is occupied.
    occupied: u8,
}

impl<Raw: Copy, const GAP_SIZE: usize> WritableGaps<Raw, GAP_SIZE> {
    /// Creates an empty set: every slot free and no window allocated.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gaps: core::array::from_fn(|_| WritableGap::at(0)),
            occupied: 0,
        }
    }

    /// Number of occupied slots.
    #[must_use]
    #[inline]
    pub fn count(&self) -> usize {
        self.occupied.count_ones() as usize
    }

    /// `true` when no slot is occupied.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.occupied == 0
    }

    /// Bit `i` is set when slot `i` is occupied.
    ///
    /// [`MAX_GAPS`] is four, so the mask fits a `u8`; flush and length
    /// bookkeeping walk the set bits instead of probing every slot.
    #[must_use]
    #[inline]
    pub fn occupied_mask(&self) -> u8 {
        self.occupied
    }

    /// `true` when every slot is occupied.
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count() >= MAX_GAPS
    }

    /// Occupies the first free slot as a gap committing at `base`.
    ///
    /// The slot keeps whatever window it allocated previously; only
    /// its staged count and base are reset. Returns the slot index, or
    /// `None` when every slot is taken.
    pub fn alloc(&mut self, base: usize) -> Option<usize> {
        let limit = if MAX_GAPS >= u8::BITS as usize {
            u8::MAX
        } else {
            (1u8 << MAX_GAPS) - 1
        };
        let free = !self.occupied & limit;
        if free == 0 {
            return None;
        }
        let index = free.trailing_zeros() as usize;
        let gap = &mut self.gaps[index];
        gap.clear();
        gap.set_base(base);
        self.occupied |= 1u8 << index;
        Some(index)
    }

    /// Releases the slot at `index`, dropping its staged elements but
    /// retaining its window for the next allocation.
    ///
    /// Returns `true` when a slot was freed.
    pub fn free(&mut self, index: usize) -> bool {
        if index >= MAX_GAPS || self.occupied & (1u8 << index) == 0 {
            return false;
        }
        self.occupied &= !(1u8 << index);
        self.gaps[index].clear();
        true
    }

    /// The gap occupying `index`.
    #[must_use]
    #[inline]
    pub fn get(&self, index: usize) -> Option<&WritableGap<Raw, GAP_SIZE>> {
        if index < MAX_GAPS && self.occupied & (1u8 << index) != 0 {
            self.gaps.get(index)
        } else {
            None
        }
    }

    /// The gap occupying `index`, mutably.
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut WritableGap<Raw, GAP_SIZE>> {
        if index < MAX_GAPS && self.occupied & (1u8 << index) != 0 {
            self.gaps.get_mut(index)
        } else {
            None
        }
    }

    /// Indices of every occupied slot, ascending.
    pub fn occupied(&self) -> impl Iterator<Item = usize> {
        let mut mask = self.occupied;
        core::iter::from_fn(move || {
            if mask == 0 {
                return None;
            }
            let index = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            Some(index)
        })
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
#[repr(C)]
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
    /// Document-level line-start ring shared by all cursors.
    ///
    /// Line starts describe the committed document, not a caret, so one
    /// ring serves every cursor. Keeping it out of the cursor pool is
    /// what keeps the hot working set small.
    pub(crate) line_cache: UnsafeCell<LineRing>,
    /// Id of the cursor driving single-cursor operations.
    pub(crate) active_cursor_id: usize,
    /// Bit `i` is set while cursor slot `i` is live.
    ///
    /// Kept alongside the pool so document-wide passes (rebinding,
    /// shifting) walk live cursors only instead of probing all
    /// [`MAX_CURSORS`] `Option`s on every commit.
    pub(crate) cursor_mask: u16,
    /// Optional event receiver.
    pub(crate) event_bridge: UnsafeCell<Option<Arc<dyn EventBridge>>>,
    /// Document name (raw bytes, NUL padded).
    pub(crate) name: [u8; NAME_LEN],
    /// User-defined flag bits.
    pub(crate) flags: u32,
    /// Number of committed elements in `raw`.
    pub(crate) raw_length: usize,
    /// Number of committed rows (newlines + 1), or `0` when unknown.
    ///
    /// This is a hint for the line ring's last-row fast path. It lets a
    /// caret row query avoid scanning the final line even when the
    /// sparse anchor index is invalid. Newline-free edits keep it
    /// exact; a merge marks it unknown until the next anchor build.
    pub(crate) rows: Cell<usize>,
    /// Committed capacity head-room (mirrors `raw.allocated()`).
    pub(crate) vspace: usize,
    /// Document encoding, one of the `ENC_*` constants.
    pub(crate) encoding: u16,
    /// Lazily allocated sparse row anchors for jumps outside the L1
    /// line ring.
    ///
    /// The sparse index lives on the heap so it does not enlarge the
    /// hot inline working state; only the pointer is part of the
    /// engine struct. `None` means it has never been built.
    pub(crate) line_anchors: UnsafeCell<Option<Box<LineAnchors>>>,
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
            line_cache: UnsafeCell::new(LineRing::empty()),
            active_cursor_id: 0,
            cursor_mask: 0,
            event_bridge: UnsafeCell::new(None),
            name: [0; NAME_LEN],
            flags: 0,
            raw_length: 0,
            rows: Cell::new(1),
            vspace: 0,
            encoding: ENC_UNKNOWN,
            line_anchors: UnsafeCell::new(None),
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
    /// Runs `f` with exclusive access to the document-level line ring.
    ///
    /// This avoids the 136-byte copy that `Cell::get`/`set` performed
    /// on every cached row query. The writable is `!Sync`, so the
    /// single-thread contract guarantees the mutable borrow does not
    /// overlap any other access to the ring.
    #[inline]
    pub(crate) fn with_line_cache<R>(&self, f: impl FnOnce(&mut LineRing) -> R) -> R {
        // SAFETY: see the method documentation.
        unsafe { f(&mut *self.line_cache.get()) }
    }

    /// Returns valid sparse row anchors, building them on first use.
    ///
    /// Building scans the committed document once. Any commit that
    /// changes the layout invalidates the index, so the next distant
    /// lookup rebuilds it. The backing allocation is reused across
    /// rebuilds; only the first lookup pays for it.
    #[must_use]
    pub(crate) fn ensure_line_anchors(&self) -> &LineAnchors
    where
        Raw: Into<u32>,
    {
        // SAFETY: the writable is `!Sync`; the single-thread contract
        // means no other reference can mutate the index concurrently.
        let slot = unsafe { &mut *self.line_anchors.get() };
        if let Some(anchors) = slot {
            if anchors.is_valid() {
                return anchors;
            }
            let raw = self.raw_ref().as_ptr();
            let len = self.raw_length;
            // SAFETY: `raw`/`len` describe the live committed view.
            **anchors = unsafe { LineAnchors::build(raw, len) };
            self.rows.set(anchors.rows().max(1));
            return anchors;
        }
        let raw = self.raw_ref().as_ptr();
        let len = self.raw_length;
        // SAFETY: `raw`/`len` describe the live committed view.
        let built = unsafe { LineAnchors::build(raw, len) };
        self.rows.set(built.rows().max(1));
        slot.insert(Box::new(built))
    }

    /// Returns the stored sparse line anchors when they are valid.
    ///
    /// Unlike [`Self::ensure_line_anchors`], this never builds the
    /// index. A caret-local row query can answer from the line ring, so
    /// forcing a full document build after a newline edit would make a
    /// single keystroke pay for an index it may not need yet.
    #[must_use]
    #[inline]
    pub(crate) fn valid_line_anchors(&self) -> Option<&LineAnchors> {
        // SAFETY: the writable is `!Sync`; the single-thread contract
        // means no other reference can mutate the index concurrently.
        let slot = unsafe { &*self.line_anchors.get() };
        slot.as_deref()
            .filter(|anchors| anchors.is_valid())
    }

    /// Rebinds every cursor to the current raw view and raises its
    /// logical clamping bound to the current document length.
    ///
    /// This is the cheap half of a refresh: it performs no cache
    /// invalidation, so callers that only moved a caret (or whose edit
    /// preserved the line layout) keep the line ring and sparse
    /// anchors warm.
    #[inline]
    fn rebind_cursor_views(&mut self) {
        let ptr = self.raw_ref().as_ptr();
        let raw_length = self.raw_length;
        let logical = self.logical_length();
        let mut mask = self.cursor_mask;
        let pool = self.cursors_mut();
        while mask != 0 {
            let id = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let Some(cursor) = pool[id].as_mut() else {
                continue;
            };
            // SAFETY: `ptr`/`raw_length` describe the current raw
            // buffer, which outlives the cursors (both live in this
            // struct), and `logical >= raw_length` keeps the
            // dereference bound valid.
            unsafe { cursor.rebind_raw(ptr, raw_length) };
            cursor.set_logical_length(logical);
        }
    }

    /// Rebinds the cursors and drops the document-level line indexes.
    ///
    /// Use after an edit that changed the line layout (inserted or
    /// removed a newline, or merged text with unknown layout effects).
    pub(crate) fn refresh_cursor_views(&mut self) {
        self.rebind_cursor_views();
        // The committed view may have changed: the document-level line
        // ring and sparse anchors are stale. Cursors themselves carry
        // no cache, so this is a single reset regardless of cursor
        // count.
        // SAFETY: exclusive access through `&mut self`; `clear` only
        // resets the three bookkeeping fields, leaving the start array
        // for the next `reload`.
        unsafe { (*self.line_cache.get()).clear() };
        // SAFETY: exclusive access through `&mut self`.
        if let Some(anchors) = unsafe { &mut *self.line_anchors.get() } {
            anchors.invalidate();
        }
    }

    /// Rebinds the cursors after newlines were inserted at `at`.
    ///
    /// The line ring survives when the run is purely newlines appended
    /// after the ring's last line: each new start is exactly one
    /// element apart. A sparse index whose anchors all sit at or before
    /// the insertion point also stays valid, so only its row count
    /// grows. Appending newlines then keeps both row indexes local
    /// instead of rebuilding them.
    pub(crate) fn refresh_cursor_views_newline_insert(
        &mut self,
        at: usize,
        newlines: usize,
        inserted: usize,
    ) {
        self.rebind_cursor_views();
        let rows = self.rows.get();
        // SAFETY: exclusive access through `&mut self`.
        let cache = unsafe { &mut *self.line_cache.get() };
        let at_tail = inserted == newlines
            && rows != 0
            && !cache.is_empty()
            && rows == cache.last_row() + 1 + newlines
            && at >= cache.last_start();
        if at_tail {
            for offset in 1..=newlines {
                let start = at.saturating_add(offset).min(u32::MAX as usize) as u32;
                cache.push_back(start);
            }
        } else {
            cache.clear();
        }
        // SAFETY: exclusive access through `&mut self`.
        if let Some(anchors) = unsafe { &mut *self.line_anchors.get() }
            && !anchors.extend_rows(at, newlines)
        {
            anchors.invalidate();
        }
    }

    /// Rebinds the cursors after `removed_newlines` were removed from
    /// `start..start + count`.
    ///
    /// A ring whose first line predates the removal keeps its row
    /// numbers; starts that followed a removed newline are dropped and
    /// later starts shift left. A sparse index that only loses trailing
    /// anchors keeps the remaining starts, so the next fallback scan
    /// stays local.
    pub(crate) fn refresh_cursor_views_newline_remove(
        &mut self,
        start: usize,
        count: usize,
        removed_newlines: usize,
    ) {
        self.rebind_cursor_views();
        let end = start.saturating_add(count);
        let rows = self.rows.get();
        // SAFETY: exclusive access through `&mut self`.
        let cache = unsafe { &mut *self.line_cache.get() };
        let inside_window = rows != 0
            && !cache.is_empty()
            && rows.saturating_add(removed_newlines) == cache.last_row() + 1
            && start >= cache.first_start();
        if inside_window {
            cache.trim_removed_range(start, end, count);
        } else {
            cache.clear();
        }
        // SAFETY: exclusive access through `&mut self`.
        if let Some(anchors) = unsafe { &mut *self.line_anchors.get() }
            && !anchors.trim_after(start, end, removed_newlines)
        {
            anchors.invalidate();
        }
    }

    /// Rebinds the cursors and shifts the line indexes after an edit
    /// that did not change the number of lines.
    ///
    /// `position` is the element index where `delta` elements were
    /// inserted (`removing == false`) or removed (`removing == true`).
    /// Both the L1 line ring and the sparse anchors are shifted in
    /// place, which keeps repeated row lookups O(1) while typing
    /// instead of refilling the ring and rebuilding the whole document
    /// index.
    pub(crate) fn refresh_cursor_views_shifted(&mut self, position: usize, delta: usize, removing: bool) {
        self.rebind_cursor_views();
        // SAFETY: exclusive access through `&mut self`.
        let cache = unsafe { &mut *self.line_cache.get() };
        if removing {
            cache.shift_after_remove(position, delta);
        } else {
            cache.shift_after_insert(position, delta);
        }
        // SAFETY: exclusive access through `&mut self`.
        if let Some(anchors) = unsafe { &mut *self.line_anchors.get() } {
            if removing {
                anchors.shift_after_remove(position, delta);
            } else {
                anchors.shift_after_insert(position, delta);
            }
        }
    }

    /// Drops the document-level line indexes after an in-place edit
    /// whose line layout may have changed.
    #[inline]
    pub(crate) fn invalidate_line_layout(&mut self) {
        // SAFETY: exclusive access through `&mut self`; `clear` only
        // resets the three bookkeeping fields.
        unsafe { (*self.line_cache.get()).clear() };
        // SAFETY: exclusive access through `&mut self`.
        if let Some(anchors) = unsafe { &mut *self.line_anchors.get() } {
            anchors.invalidate();
        }
    }

    /// Raises one cursor's logical clamping bound to the current
    /// document length.
    ///
    /// Moving a caret cannot change the committed layout, so this is
    /// all a pure movement needs: no line index is touched and no
    /// other cursor is disturbed.
    #[inline]
    pub(crate) fn sync_cursor_length(&self, cursor_id: usize) {
        let logical = self.logical_length();
        if let Some(cursor) = self.cursor(cursor_id) {
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
    #[inline]
    pub(crate) fn logical_length(&self) -> usize {
        // The common committed-only case avoids walking the gap and
        // shared-slot tables entirely.
        if self.gaps_ref().is_empty() && self.shared_ref().is_empty() {
            return self.raw_length;
        }
        let mut total = self.raw_length;
        let gaps = self.gaps_ref();
        if !gaps.is_empty() {
            for index in gaps.occupied() {
                if let Some(gap) = gaps.get(index) {
                    total = total.saturating_add(gap.staged());
                }
            }
        }
        // The common staged-typing case has one private gap and no
        // shared region: skip the 16-slot descriptor walk entirely.
        let shared = self.shared_ref();
        if !shared.is_empty() {
            let mut slot = 0;
            while slot < SHARED_GAP_SLOTS {
                if let Some(descriptor) = shared.slot(slot)
                    && descriptor.is_allocated()
                {
                    total = total.saturating_add(descriptor.size());
                }
                slot += 1;
            }
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
        let mut mask = self.cursor_mask;
        let pool = self.cursors_mut();
        while mask != 0 {
            let id = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let Some(cursor) = pool[id].as_mut() else {
                continue;
            };
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
        // + metad) must stay at or below 4096 bytes inline.
        assert!(
            core::mem::size_of::<BaseWritable<u32, u32, 128>>() <= 4096,
            "BaseWritable exceeds the 4096 byte inline budget: {} bytes",
            core::mem::size_of::<BaseWritable<u32, u32, 128>>()
        );
        // Cursor ids are bounded by the fixed pool.
        assert!(MAX_CURSORS <= u16::MAX as usize);
    }
}
