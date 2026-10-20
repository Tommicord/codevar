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

//! Operation implementations of the edit engine.
//!
//! The `TextEditable*` traits are implemented for [`StreamWritable`]
//! here, alongside the helpers shared by the concrete operations:
//! cursor validation, raw insert/remove bookkeeping and private-gap
//! staging.

use codevar_logger::log_error;

use super::base::{T_MERGE, make_packed, packed_char, packed_type};
use super::scan;
use super::shared_gap::{SLOT_CURSOR_BITS, SLOT_MAX_BASE, SLOT_MAX_SIZE};
use super::traits::{
    TextEditableBufferQuery, TextEditableCursor, TextEditableDelete, TextEditableGap, TextEditableOverwrite,
    TextEditablePut, TextEditableReplace, TextEditableSelection, Writable,
};
use super::writable::{GAP_NONE, MAX_CURSORS, MAX_GAPS, SHARED_GAP_SLOTS, StreamWritable, WritableEvent};
use crate::arch::memmove;
use crate::error::{WreditError, WreditResult};

impl<Raw, Buf, const GAP_SIZE: usize> Writable<Raw, Buf, GAP_SIZE> for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy,
    Buf: Copy,
{
    fn new() -> Self {
        Self {
            base: super::writable::BaseWritable::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            base: super::writable::BaseWritable::with_capacity(capacity),
        }
    }

    fn append_cursor(&mut self, cursor: super::cursor::Cursor<Raw, Buf, GAP_SIZE>) -> Option<usize> {
        // Resolve the view before parking the cursor in the pool.
        let ptr = self.base.raw_ref().as_ptr();
        let raw_length = self.base.raw_length;
        let logical = self.base.logical_length();
        let pool = self.base.cursors_mut();
        let id = pool.iter().position(Option::is_none)?;
        // SAFETY: `ptr`/`raw_length` describe the current raw buffer,
        // which lives inside `self.base` and outlives the pool entry.
        unsafe { cursor.rebind_raw(ptr, raw_length) };
        cursor.set_logical_length(logical);
        pool[id] = Some(cursor);
        self.base.cursor_mask |= 1u16 << id;
        self.base.active_cursor_id = id;
        self.base.emit(WritableEvent::cursor_lifecycle(
            super::writable::WritableEventType::CursorAdded,
            id,
        ));
        Some(id)
    }

    fn delete_cursor(&mut self, cursor_id: usize) -> bool {
        if cursor_id >= MAX_CURSORS {
            return false;
        }
        let pool = self.base.cursors_mut();
        if pool[cursor_id].take().is_none() {
            return false;
        }
        self.base.cursor_mask &= !(1u16 << cursor_id);
        // Release whatever the cursor owned.
        let mapping = self.base.cursor_to_gap[cursor_id];
        self.base.cursor_to_gap[cursor_id] = GAP_NONE;
        if mapping < MAX_GAPS as u16 {
            self.base.gaps_mut().free(mapping as usize);
        } else if mapping != GAP_NONE {
            let slot = usize::from(mapping - MAX_GAPS as u16);
            let shared = self.base.shared_mut();
            shared.leave_cursor(slot, cursor_id);
            if shared
                .slot(slot)
                .is_some_and(|descriptor| descriptor.cursor_mask() == 0)
            {
                shared.free_slot(slot);
            }
        }
        if self.base.active_cursor_id == cursor_id {
            let mask = self.base.cursor_mask;
            self.base.active_cursor_id = if mask == 0 {
                0
            } else {
                mask.trailing_zeros() as usize
            };
        }
        self.base.emit(WritableEvent::cursor_lifecycle(
            super::writable::WritableEventType::CursorRemoved,
            cursor_id,
        ));
        true
    }

    fn cursor_from_id(&self, cursor_id: usize) -> Option<&super::cursor::Cursor<Raw, Buf, GAP_SIZE>> {
        self.base.cursor(cursor_id)
    }

    unsafe fn cursor_from_id_mut(
        &self,
        cursor_id: usize,
    ) -> Option<*mut super::cursor::Cursor<Raw, Buf, GAP_SIZE>> {
        if cursor_id >= MAX_CURSORS {
            return None;
        }
        // SAFETY: the caller upholds the single-thread contract, so
        // this exclusive view of the cell cannot alias another live
        // borrow; the slot is only handed out when it is occupied.
        let pool = unsafe { &mut *self.base.cursors.get() };
        pool.get_mut(cursor_id)?
            .as_mut()
            .map(|cursor| cursor as *mut _)
    }

    fn v_space(&self) -> usize {
        self.base.vspace
    }

    fn shared_gaps(&self) -> &super::shared_gap::SharedGapTable<SHARED_GAP_SLOTS> {
        self.base.shared_ref()
    }

    fn active_cursor_id(&self) -> usize {
        self.base.active_cursor_id
    }

    fn set_active_cursor(&mut self, cursor_id: usize) -> bool {
        if !self.base.valid_cursor_id(cursor_id) {
            return false;
        }
        self.base.active_cursor_id = cursor_id;
        true
    }

    fn active_cursor(&self) -> Option<&super::cursor::Cursor<Raw, Buf, GAP_SIZE>> {
        self.base.cursor(self.base.active_cursor_id)
    }

    fn raw_ptr(&self) -> *const Raw {
        self.base.raw_ref().as_ptr()
    }

    unsafe fn raw_mut_ptr(&self) -> *mut Raw {
        // SAFETY: the caller upholds the single-thread contract; the
        // pointer aliases the raw cell only while they use it.
        unsafe { (&mut *self.base.raw.get()).as_mut_ptr() }
    }

    fn raw_length(&self) -> usize {
        self.base.raw_length
    }

    fn set_event_bridge(&mut self, bridge: Option<alloc::sync::Arc<dyn super::writable::EventBridge>>) {
        *self.base.event_bridge.get_mut() = bridge;
    }

    fn event_bridge(&self) -> Option<alloc::sync::Arc<dyn super::writable::EventBridge>> {
        // SAFETY: single-thread contract; the clone ends the borrow.
        let slot = unsafe { &*self.base.event_bridge.get() };
        slot.clone()
    }

    fn name(&self) -> &str {
        let end = self
            .base
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(self.base.name.len());
        core::str::from_utf8(&self.base.name[..end]).unwrap_or("")
    }

    fn name_mut(&mut self) -> &mut [u8; super::base::NAME_LEN] {
        &mut self.base.name
    }

    fn flags(&self) -> u32 {
        self.base.flags
    }

    fn flags_mut(&mut self) -> &mut u32 {
        &mut self.base.flags
    }

    unsafe fn proximity(&self, pos: usize) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for index in self.base.gaps_ref().occupied() {
            let Some(gap) = self.base.gaps_ref().get(index) else {
                continue;
            };
            let distance = gap.base().abs_diff(pos);
            if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                best = Some((index, distance));
            }
        }
        let shared = self.base.shared_ref();
        let mut slot = 0;
        while slot < SHARED_GAP_SLOTS {
            if let Some(descriptor) = shared.slot(slot)
                && descriptor.is_allocated()
            {
                let distance = descriptor.base().abs_diff(pos);
                let index = MAX_GAPS + slot;
                if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                    best = Some((index, distance));
                }
            }
            slot += 1;
        }
        best.map(|(index, _)| index)
    }

    unsafe fn new_shared_gap(&mut self, cursor_mask: u16) -> Option<usize> {
        self.base.shared_mut().alloc_slot(cursor_mask)
    }

    unsafe fn shared_gap_index(&self, cursor_id: usize) -> Option<usize> {
        self.base.shared_ref().slot_for_cursor(cursor_id)
    }

    unsafe fn is_cursor_in_shared_gap(&self, gap_index: usize, cursor_id: usize) -> bool {
        self.base
            .shared_ref()
            .is_cursor_in(gap_index, cursor_id)
    }

    unsafe fn select_all(&self, cursor_id: usize) -> bool {
        let Some(cursor) = self.base.cursor(cursor_id) else {
            return false;
        };
        let logical = self.base.logical_length();
        cursor.set_selection_anchor(0);
        cursor.set_selection_end(logical);
        self.base
            .emit(WritableEvent::selection_changed(cursor_id));
        true
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    /// Validates a cursor id for a fallible operation.
    ///
    /// An unknown id is logged and reported as
    /// [`WreditError::InvalidCursor`].
    pub(in crate::edit) fn validate_cursor(&self, cursor_id: usize) -> WreditResult<usize> {
        if !self.base.valid_cursor_id(cursor_id) {
            log_error!("wredit: invalid cursor id {cursor_id}");
            return Err(WreditError::InvalidCursor { cursor: cursor_id });
        }
        Ok(cursor_id)
    }

    /// The cursor's current gap mapping, or [`GAP_NONE`].
    pub(in crate::edit) fn cursor_mapping(&self, cursor_id: usize) -> u16 {
        self.base
            .cursor_to_gap
            .get(cursor_id)
            .copied()
            .unwrap_or(GAP_NONE)
    }

    /// Ensures the cursor owns a private gap staged at `base`.
    ///
    /// Returns the gap index, or an error when the fixed gap set is
    /// exhausted or the cursor is staged in a shared gap.
    pub(in crate::edit) fn ensure_private_gap(
        &mut self,
        cursor_id: usize,
        base: usize,
    ) -> WreditResult<usize> {
        let mapping = self.cursor_mapping(cursor_id);
        if mapping < MAX_GAPS as u16 {
            return Ok(usize::from(mapping));
        }
        if mapping != GAP_NONE {
            log_error!("wredit: cursor {cursor_id} is staged in a shared gap; flush before private staging");
            return Err(WreditError::InvalidOperation {
                what: "cursor staged in a shared gap",
            });
        }
        match self.base.gaps_mut().alloc(base) {
            Some(index) => {
                self.base.cursor_to_gap[cursor_id] = index as u16;
                Ok(index)
            }
            None => {
                log_error!("wredit: every private gap slot is in use");
                Err(WreditError::CapacityExhausted {
                    requested: 1,
                    available: 0,
                })
            }
        }
    }

    /// Resolves (allocating on first use) the shared gap slot the
    /// cursor belongs to.
    ///
    /// Returns [`WreditError::InvalidOperation`] when the cursor is
    /// staged in a private gap, and [`WreditError::CapacityExhausted`]
    /// when no slot or mask bit is left.
    pub(in crate::edit) fn ensure_shared_slot(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let mapping = self.cursor_mapping(cursor_id);
        if mapping == GAP_NONE {
            if cursor_id >= super::shared_gap::SLOT_CURSOR_BITS {
                log_error!("wredit: cursor {cursor_id} exceeds the shared gap mask");
                return Err(WreditError::InvalidCursor { cursor: cursor_id });
            }
            let mask = 1u16 << cursor_id;
            // SAFETY: the mask names only the validated `cursor_id`
            // and the single-thread contract holds (exclusive `self`).
            let slot = match unsafe { self.new_shared_gap(mask) } {
                Some(slot) => slot,
                None => {
                    log_error!("wredit: every shared gap slot is in use");
                    return Err(WreditError::CapacityExhausted {
                        requested: 1,
                        available: 0,
                    });
                }
            };
            self.base.cursor_to_gap[cursor_id] = (MAX_GAPS + slot) as u16;
            Ok(slot)
        } else if mapping >= MAX_GAPS as u16 {
            Ok(usize::from(mapping - MAX_GAPS as u16))
        } else {
            log_error!("wredit: cursor {cursor_id} is staged in a private gap");
            Err(WreditError::InvalidOperation {
                what: "cursor staged in a private gap",
            })
        }
    }

    /// Commits whatever the cursor has staged (private or shared) so
    /// the operation below sees a committed layout. Failures are
    /// logged inside the flush path and simply leave the staged data.
    pub(in crate::edit) fn flush_cursor_gap(&mut self, cursor_id: usize) {
        let mapping = self.cursor_mapping(cursor_id);
        if mapping == GAP_NONE {
            return;
        }
        if mapping < MAX_GAPS as u16 {
            let _ = self.flush_gap(usize::from(mapping));
        } else {
            let slot = usize::from(mapping - MAX_GAPS as u16);
            let _ = self.flush_gap(MAX_GAPS + slot);
        }
    }

    /// Stages `value` in the cursor's private gap at `base`,
    /// flushing the gap first when it is full.
    ///
    /// Returns the gap index that received (or would have received)
    /// the value.
    pub(in crate::edit) fn stage_private(
        &mut self,
        cursor_id: usize,
        value: Raw,
        base: usize,
    ) -> WreditResult<usize> {
        let mut index = self.ensure_private_gap(cursor_id, base)?;
        if self
            .base
            .gaps_ref()
            .get(index)
            .is_some_and(super::writable::WritableGap::is_full)
        {
            // Commit the pending run, then restage at the (updated)
            // cursor position so nothing is dropped.
            let _ = self.flush_gap(index);
            let position = self
                .base
                .cursor(cursor_id)
                .map(super::cursor::Cursor::array_pos)
                .unwrap_or(base);
            index = self.ensure_private_gap(cursor_id, position)?;
        }
        let pushed = match self.base.gaps_mut().get_mut(index) {
            Some(gap) => gap.push(value),
            None => false,
        };
        if !pushed {
            log_error!("wredit: gap window unavailable for staging");
            return Err(WreditError::CapacityExhausted {
                requested: GAP_SIZE,
                available: 0,
            });
        }
        Ok(index)
    }

    /// Advances `cursor_id`'s caret to `position` and announces the
    /// move.
    pub(in crate::edit) fn move_caret_to(&mut self, cursor_id: usize, position: usize) -> usize {
        let Some(from) = self
            .base
            .cursor(cursor_id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return position;
        };
        // Raise this caret's logical bound (staged edits grow it)
        // before it clamps against the new position. Only the moved
        // cursor needs the update: editing ops stage into a gap and
        // never reallocate the raw buffer, so the shared pointer view
        // stays valid.
        let logical = self.base.logical_length();
        let Some(cursor) = self.base.cursor(cursor_id) else {
            return position;
        };
        cursor.set_logical_length(logical);
        let to = cursor.set_array_pos(position);
        self.base
            .emit(WritableEvent::cursor_moved(cursor_id, from, to));
        to
    }

    /// Commits a raw insertion at `at` with full cursor/gap
    /// bookkeeping.
    ///
    /// Returns the number of inserted elements.
    pub(in crate::edit) fn commit_insertion(&mut self, at: usize, values: &[Raw]) -> usize {
        if values.is_empty() || at > self.base.raw_length {
            return 0;
        }
        let old_length = self.base.raw_length;
        let new_length = super::flush::write_bytes_to_raw(self.base.raw_mut(), old_length, at, values);
        if new_length == old_length {
            return 0;
        }
        let inserted = new_length - old_length;
        self.base.raw_length = new_length;
        self.base.raw_mut().set_count(new_length);
        self.base.vspace = self.base.raw_ref().allocated();
        // Inserting text without a newline keeps every row start, so
        // the sparse index can shift with the text instead of being
        // invalidated and rebuilt on the next row query.
        // SAFETY: `values` is a valid read-only range for its length.
        let newlines = unsafe { scan::count_newlines(values.as_ptr(), values.len()) };
        let layout_stable = newlines == 0;
        // Keep the committed row count exact for the line ring. A merge
        // can consume an existing element (possibly a newline), so its
        // effect is not known here; mark the hint unknown instead.
        let has_merge = values
            .iter()
            .any(|value| packed_type(Into::<u32>::into(*value)) == T_MERGE);
        let rows = self.base.rows.get();
        if has_merge {
            self.base.rows.set(0);
        } else if rows != 0 {
            self.base.rows.set(rows.saturating_add(newlines));
        }
        if layout_stable {
            self.base
                .refresh_cursor_views_shifted(at, inserted, false);
        } else if has_merge {
            self.base.refresh_cursor_views();
        } else {
            self.base
                .refresh_cursor_views_newline_insert(at, newlines, inserted);
        }
        self.base.shift_cursors(at, inserted, false);
        self.base.shift_gap_bases(at, inserted, false);
        self.base.shift_shared_bases(at, inserted, false);
        self.base
            .emit(WritableEvent::buffer_changed(at, inserted));
        inserted
    }

    /// Removes `count` committed elements at `start` with full
    /// cursor/gap bookkeeping.
    ///
    /// Returns the number of removed elements.
    pub(in crate::edit) fn commit_removal(&mut self, start: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let old_length = self.base.raw_length;
        let end = start.saturating_add(count);
        if end > old_length {
            log_error!("wredit: raw removal {start}..{end} exceeds length {old_length}");
            return 0;
        }
        // Scanning the doomed range before the move tells us whether a
        // line disappears. Without a newline the row count is intact,
        // so the sparse index only needs shifting.
        let removed_newlines = {
            let ptr = self.base.raw_ref().as_ptr();
            // SAFETY: `end <= raw_length`, so `start..end` is a valid
            // initialized range of the committed view.
            unsafe { scan::count_newlines(ptr.add(start), count) }
        };
        let layout_stable = removed_newlines == 0;
        {
            let raw = self.base.raw_mut();
            let ptr = raw.as_mut_ptr();
            let byte_len = (old_length - end).saturating_mul(core::mem::size_of::<Raw>());
            // Safety: `end <= raw_length <= allocated`, so both ranges
            // are in bounds; the arch `memmove` handles the tail
            // overlap with its destination.
            unsafe {
                memmove(ptr.add(start).cast::<u8>(), ptr.add(end).cast::<u8>(), byte_len);
            }
            let new_length = old_length - count;
            raw.set_count(new_length);
            self.base.raw_length = new_length;
        }
        self.base.vspace = self.base.raw_ref().allocated();
        self.base.shift_cursors(start, count, true);
        self.base.shift_gap_bases(start, count, true);
        self.base.shift_shared_bases(start, count, true);
        let rows = self.base.rows.get();
        if removed_newlines > 0 && rows != 0 {
            self.base
                .rows
                .set(rows.saturating_sub(removed_newlines).max(1));
        }
        if layout_stable {
            self.base
                .refresh_cursor_views_shifted(start, count, true);
        } else {
            self.base
                .refresh_cursor_views_newline_remove(start, count, removed_newlines);
        }
        self.base
            .emit(WritableEvent::buffer_changed(start, count));
        count
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableGap<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn flush_gap(&mut self, gap_index: usize) -> usize {
        // The inherent flusher implements the commit path; the trait
        // method is the contract-facing alias of it.
        StreamWritable::flush_gap(self, gap_index)
    }

    fn flush_all(&mut self) -> usize {
        StreamWritable::flush_all(self)
    }

    unsafe fn shift_gap_block(&mut self, gap_index: usize, distance: isize) -> usize {
        let logical_length = self.base.logical_length();
        let current = if gap_index < MAX_GAPS {
            self.base
                .gaps_ref()
                .get(gap_index)
                .map(|gap| gap.base())
        } else if gap_index - MAX_GAPS < SHARED_GAP_SLOTS {
            let slot = gap_index - MAX_GAPS;
            self.base
                .shared_ref()
                .slot(slot)
                .filter(|descriptor| descriptor.is_allocated())
                .map(|descriptor| descriptor.base())
        } else {
            None
        };
        let Some(current) = current else {
            log_error!("wredit: shift of the unallocated gap {gap_index}");
            return 0;
        };
        let moved = if distance >= 0 {
            current
                .saturating_add(usize::try_from(distance).unwrap_or(usize::MAX))
                .min(logical_length)
        } else {
            current.saturating_sub(distance.unsigned_abs())
        };
        if gap_index < MAX_GAPS {
            if let Some(gap) = self.base.gaps_mut().get_mut(gap_index) {
                gap.set_base(moved);
            }
        } else {
            let slot = gap_index - MAX_GAPS;
            let size = self
                .base
                .shared_ref()
                .slot(slot)
                .map(|descriptor| descriptor.size())
                .unwrap_or(0);
            self.base
                .shared_mut()
                .set_region(slot, moved, size);
        }
        moved
    }

    unsafe fn shift_right_gap_block(&mut self, gap_index: usize, distance: usize) -> usize {
        // SAFETY: forwarded to `shift_gap_block`, which clamps the
        // base into the logical document.
        unsafe { self.shift_gap_block(gap_index, isize::try_from(distance).unwrap_or(isize::MAX)) }
    }

    unsafe fn shift_left_gap_block(&mut self, gap_index: usize, distance: usize) -> usize {
        let magnitude = isize::try_from(distance).unwrap_or(isize::MAX);
        // SAFETY: forwarded to `shift_gap_block`, which clamps the
        // base into the logical document.
        unsafe { self.shift_gap_block(gap_index, -magnitude) }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableDelete<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn delete(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // Commit the cursor's own staged run first so the deletion
        // addresses a committed layout.
        self.flush_cursor_gap(id);
        let position = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position >= self.base.raw_length {
            return Ok(0);
        }
        Ok(self.commit_removal(position, 1))
    }

    fn backspace(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.flush_cursor_gap(id);
        let position = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position == 0 {
            return Ok(0);
        }
        let at = position - 1;
        Ok(self.commit_removal(at, 1))
    }

    fn delete_range(&mut self, start: usize, end: usize) -> WreditResult<usize> {
        let raw_length = self.base.raw_length;
        if start > end || end > raw_length {
            log_error!("wredit: delete_range {start}..{end} outside the document (length {raw_length})");
            return Err(WreditError::OutOfRange {
                start: start as u64,
                end: end as u64,
            });
        }
        let count = end - start;
        if count == 0 {
            return Ok(0);
        }
        Ok(self.commit_removal(start, count))
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableReplace<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn replace(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // A staged run at the caret commits first, then the element
        // it landed on is removed and the replacement stages.
        self.flush_cursor_gap(id);
        let position = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position < self.base.raw_length {
            let removed = self.commit_removal(position, 1);
            if removed == 0 {
                return Err(WreditError::InvalidOperation {
                    what: "replace could not remove the covered element",
                });
            }
        }
        self.put(id, value)
    }

    fn replace_range(&mut self, start: usize, end: usize, values: &[Raw]) -> WreditResult<usize> {
        let raw_length = self.base.raw_length;
        if start > end || end > raw_length {
            log_error!("wredit: replace_range {start}..{end} outside the document (length {raw_length})");
            return Err(WreditError::OutOfRange {
                start: start as u64,
                end: end as u64,
            });
        }
        let removed = end - start;
        if removed > 0 {
            let done = self.commit_removal(start, removed);
            if done == 0 {
                return Err(WreditError::InvalidOperation {
                    what: "replace_range could not remove the old range",
                });
            }
        }
        if values.is_empty() {
            return Ok(0);
        }
        let inserted = self.commit_insertion(start, values);
        if inserted == 0 {
            // The removal already applied; the insertion was refused
            // (allocation) — report it without unwinding anything.
            return Err(WreditError::CapacityExhausted {
                requested: values.len(),
                available: 0,
            });
        }
        Ok(inserted)
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableSelection<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn set_selection_anchor(&mut self, cursor_id: usize, position: usize) -> bool {
        let Ok(id) = self.validate_cursor(cursor_id) else {
            return false;
        };
        // The clamping bound must cover staged positions first.
        // Selecting cannot change the document layout, so only this
        // cursor's bound is raised and the line caches stay warm.
        self.base.sync_cursor_length(id);
        let Some(cursor) = self.base.cursor(id) else {
            return false;
        };
        cursor.set_selection_anchor(position);
        self.base
            .emit(WritableEvent::selection_changed(id));
        true
    }

    fn set_selection_end(&mut self, cursor_id: usize, position: usize) -> bool {
        let Ok(id) = self.validate_cursor(cursor_id) else {
            return false;
        };
        self.base.sync_cursor_length(id);
        let Some(cursor) = self.base.cursor(id) else {
            return false;
        };
        cursor.set_selection_end(position);
        self.base
            .emit(WritableEvent::selection_changed(id));
        true
    }

    fn clear_selection(&mut self, cursor_id: usize) -> bool {
        let Ok(id) = self.validate_cursor(cursor_id) else {
            return false;
        };
        let Some(cursor) = self.base.cursor(id) else {
            return false;
        };
        cursor.clear_selection();
        self.base
            .emit(WritableEvent::selection_changed(id));
        true
    }

    fn selection_range(&self, cursor_id: usize) -> Option<(usize, usize)> {
        match self.base.cursor(cursor_id) {
            Some(cursor) => cursor.selection_range(),
            None => {
                log_error!("wredit: selection query for dead cursor {cursor_id}");
                None
            }
        }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditablePut<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn put(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        let position = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        self.stage_private(id, value, position)?;
        Ok(self.move_caret_to(id, position.saturating_add(1)))
    }

    fn put_shared(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // The mask only addresses live cursors; ids at or above the
        // mask width can never join a shared gap.
        if id >= MAX_CURSORS || id >= SLOT_CURSOR_BITS {
            log_error!("wredit: cursor {id} cannot join a shared gap mask");
            return Err(WreditError::InvalidCursor { cursor: id });
        }
        let mut recovered = false;
        loop {
            let position = self
                .base
                .cursor(id)
                .map(super::cursor::Cursor::array_pos)
                .unwrap_or(0);
            let slot = self.ensure_shared_slot(id)?;
            let Some(descriptor) = self.base.shared_ref().slot(slot) else {
                log_error!("wredit: shared gap slot {slot} vanished");
                return Err(WreditError::InvalidOperation {
                    what: "shared gap slot missing",
                });
            };
            let size = descriptor.size();
            if size >= SLOT_MAX_SIZE {
                if recovered {
                    log_error!("wredit: shared gap region exhausted");
                    return Err(WreditError::CapacityExhausted {
                        requested: size + 1,
                        available: 0,
                    });
                }
                // Commit the full region and restage into a fresh one.
                let _ = self.flush_gap(MAX_GAPS + slot);
                recovered = true;
                continue;
            }
            // A fresh region commits where the caret stands; an
            // existing one appends after its members' text.
            let base = if size == 0 {
                position.min(SLOT_MAX_BASE)
            } else {
                descriptor.base()
            };
            let Some(offset) = base
                .checked_add(size)
                .and_then(|value| value.checked_add(1))
            else {
                log_error!("wredit: shared gap region offset overflow");
                return Err(WreditError::LimitExceeded {
                    what: "shared gap region offset",
                });
            };
            if self
                .base
                .buffer_mut()
                .ensure_capacity(offset)
                .is_err()
            {
                return Err(WreditError::CapacityExhausted {
                    requested: offset,
                    available: 0,
                });
            }
            let arena = self.base.buffer_mut();
            arena.set_count(offset);
            if let Some(slot_ref) = arena.slice_mut().get_mut(base + size) {
                *slot_ref = Buf::from(value);
            } else {
                log_error!("wredit: shared gap write escaped the staging arena");
                return Err(WreditError::InvalidOperation {
                    what: "shared gap write out of bounds",
                });
            }
            self.base
                .shared_mut()
                .set_region(slot, base, size + 1);
            return Ok(self.move_caret_to(id, position.saturating_add(1)));
        }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableOverwrite<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn overwrite(&mut self, cursor_id: usize, value: Raw) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        let position = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
            .unwrap_or(0);
        if position >= self.base.raw_length {
            // Nothing to overwrite at (or past) the end: append.
            return self.put(id, value);
        }
        // The merge element consumes the committed element that
        // follows the insertion point when the run is flushed.
        let merge = Raw::from(make_packed(packed_char(Into::<u32>::into(value)), T_MERGE));
        self.stage_private(id, merge, position)?;
        Ok(self.move_caret_to(id, position.saturating_add(1)))
    }

    fn overwrite_slice(&mut self, start: usize, values: &[Raw]) -> WreditResult<usize> {
        if values.is_empty() {
            return Ok(0);
        }
        let raw_length = self.base.raw_length;
        if start > raw_length {
            log_error!("wredit: overwrite_slice at {start} exceeds length {raw_length}");
            return Err(WreditError::InvalidPosition {
                position: start as u64,
                len: raw_length as u64,
            });
        }
        // The overlapping prefix is overwritten in place (no cursor
        // moves, no tail shifts); anything past the end appends.
        let overwritten = values.len().min(raw_length - start);
        let mut layout_changed = false;
        let mut removed_newlines = 0usize;
        let mut added_newlines = 0usize;
        if overwritten > 0 {
            let raw = self.base.raw_mut();
            if let Some(target) = raw
                .slice_mut()
                .get_mut(start..start + overwritten)
            {
                // An in-place overwrite keeps every position but can
                // move line boundaries. Neither side containing a
                // newline means the row layout is untouched and the
                // line caches stay valid.
                // Safety: both ranges are initialized and within
                // their respective slices.
                removed_newlines = unsafe { scan::count_newlines(target.as_ptr(), overwritten) };
                added_newlines = unsafe { scan::count_newlines(values.as_ptr(), overwritten) };
                layout_changed = removed_newlines > 0 || added_newlines > 0;
                target.copy_from_slice(&values[..overwritten]);
            } else {
                log_error!("wredit: overwrite_slice target escaped the raw buffer");
                return Err(WreditError::InvalidOperation {
                    what: "overwrite_slice target out of bounds",
                });
            }
        }
        let rows = self.base.rows.get();
        if rows != 0 {
            self.base.rows.set(
                rows.saturating_sub(removed_newlines)
                    .saturating_add(added_newlines)
                    .max(1),
            );
        }
        let mut changed = overwritten;
        let appended = values.len() - overwritten;
        if appended > 0 {
            let inserted = self.commit_insertion(raw_length, &values[overwritten..]);
            if inserted == 0 {
                return Err(WreditError::CapacityExhausted {
                    requested: appended,
                    available: 0,
                });
            }
            changed += inserted;
        }
        if layout_changed {
            // The in-place part changed a line boundary; the append
            // path (if any) has already refreshed the cursors.
            self.base.invalidate_line_layout();
        }
        if changed > 0 {
            self.base
                .emit(WritableEvent::buffer_changed(start, changed));
        }
        Ok(changed)
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableBufferQuery<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn raw_slice(&self, start: usize, end: usize) -> Option<&[Raw]> {
        if start > end {
            return None;
        }
        // The raw buffer's initialized count mirrors `raw_length`.
        let slice = self.base.raw_ref().slice();
        if end > slice.len() {
            return None;
        }
        slice.get(start..end)
    }

    fn packed_at(&self, position: usize) -> Option<Raw> {
        self.raw_slice(position, position.checked_add(1)?)
            .and_then(|elements| elements.first().copied())
    }

    fn type_at(&self, position: usize) -> Option<u32> {
        self.packed_at(position)
            .map(|element| packed_type(Into::<u32>::into(element)))
    }

    fn char_at(&self, position: usize) -> Option<u32> {
        self.packed_at(position)
            .map(|element| packed_char(Into::<u32>::into(element)))
    }

    fn line_width_at(&self, position: usize) -> Option<usize> {
        if position > self.base.raw_length {
            return None;
        }
        let elements = self.base.raw_ref().slice();
        let position = position.min(elements.len());
        // Safety: the slice is valid for its whole length; one CPUID
        // dispatch resolves both boundaries.
        let (start, end) = unsafe { scan::line_bounds(elements.as_ptr(), elements.len(), position) };
        Some(end - start)
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableCursor<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    fn move_cursor_forward(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // Element-wise movement works against the logical document,
        // so staged runs need no flush: only this cursor's clamping
        // bound has to be current (and moving cannot change the line
        // layout, so the document caches stay warm).
        self.base.sync_cursor_length(id);
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(|cursor| cursor.move_forward(amount))
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_backward(&mut self, cursor_id: usize, amount: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.base.sync_cursor_length(id);
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(|cursor| cursor.move_backward(amount))
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_next_line(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        // Line navigation reads the committed layout: commit whatever
        // this cursor has staged first. A successful flush already
        // rebinds every cursor, so only the moved cursor's clamping
        // bound needs settling (and the line indexes must not be
        // dropped a second time).
        self.flush_cursor_gap(id);
        self.base.sync_cursor_length(id);
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::move_next_line)
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_prev_line(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.flush_cursor_gap(id);
        self.base.sync_cursor_length(id);
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let to = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::move_prev_line)
            .unwrap_or(from);
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn move_cursor_to(&mut self, cursor_id: usize, row: usize, col: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        self.flush_cursor_gap(id);
        // Refresh only the clamping bound: the committed layout is
        // already current, so the document caches stay warm.
        self.base.sync_cursor_length(id);
        let Some(from) = self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return Ok(0);
        };
        let anchors = self.base.ensure_line_anchors();
        let to = self.base.with_line_cache(|cache| {
            self.base
                .cursor(id)
                .map(|cursor| cursor.move_to_cached(cache, anchors, row, col))
                .unwrap_or(from)
        });
        self.base
            .emit(WritableEvent::cursor_moved(id, from, to));
        Ok(to)
    }

    fn request_region_load(&mut self, cursor_id: usize, row: usize) -> WreditResult<bool> {
        let id = self.validate_cursor(cursor_id)?;
        let logical = self.base.logical_length();
        let Some(cursor) = self.base.cursor(id) else {
            return Ok(false);
        };
        cursor.set_logical_length(logical);
        let anchors = self.base.ensure_line_anchors();
        let loaded = self
            .base
            .with_line_cache(|cache| cursor.request_region_load_cached(cache, anchors, row));
        Ok(loaded)
    }

    fn cursor_row(&self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        let anchors = self.base.valid_line_anchors();
        let rows = self.base.rows.get();
        let row = self.base.with_line_cache(|cache| {
            self.base
                .cursor(id)
                .map(|cursor| cursor.row_cached(cache, anchors, rows))
                .unwrap_or(0)
        });
        Ok(row)
    }

    fn cursor_col(&self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        Ok(self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::col)
            .unwrap_or(0))
    }

    fn cursor_width(&self, cursor_id: usize) -> WreditResult<usize> {
        let id = self.validate_cursor(cursor_id)?;
        Ok(self
            .base
            .cursor(id)
            .map(super::cursor::Cursor::width)
            .unwrap_or(0))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::edit::base::{T_COMMON, make_packed};
    use crate::edit::traits::{
        TextEditableBufferQuery, TextEditableCursor, TextEditableDelete, TextEditableOverwrite,
        TextEditablePut, TextEditableReplace, TextEditableSelection,
    };

    // Justification: `unwrap` in tests is permitted by the project
    // lint configuration (`allow-unwrap-in-tests`).

    /// The instantiation the operation tests run against.
    pub(crate) type TestStream = StreamWritable<u32, u32, 16>;

    /// Packs a plain committed character.
    pub(crate) fn plain(character: impl Into<u32>) -> u32 {
        make_packed(character.into(), T_COMMON)
    }

    /// A stream with one active cursor at the document start.
    pub(crate) fn new_stream() -> TestStream {
        let mut stream = TestStream::new();
        let id = stream
            .append_cursor(crate::edit::cursor::Cursor::new())
            .unwrap();
        stream.set_active_cursor(id);
        stream
    }

    #[test]
    fn append_and_delete_cursors() {
        let mut stream = TestStream::new();
        assert!(stream.active_cursor().is_none());
        let first = stream.append_cursor(crate::edit::cursor::Cursor::new());
        assert_eq!(first, Some(0));
        let second = stream.append_cursor(crate::edit::cursor::Cursor::new());
        assert_eq!(second, Some(1));
        assert_eq!(stream.active_cursor_id(), 1);
        assert!(stream.delete_cursor(1));
        assert!(!stream.delete_cursor(1));
        assert!(!stream.delete_cursor(99));
        assert_eq!(stream.active_cursor_id(), 0);
        // The slot is reusable.
        assert_eq!(stream.append_cursor(crate::edit::cursor::Cursor::new()), Some(1));
    }

    #[test]
    fn cursor_pool_is_bounded() {
        let mut stream = TestStream::new();
        let mut appended = 0;
        for _ in 0..MAX_CURSORS {
            if stream
                .append_cursor(crate::edit::cursor::Cursor::new())
                .is_some()
            {
                appended += 1;
            }
        }
        assert_eq!(appended, MAX_CURSORS);
        assert!(
            stream
                .append_cursor(crate::edit::cursor::Cursor::new())
                .is_none()
        );
    }

    #[test]
    fn put_stages_and_flush_commits() {
        let mut stream = new_stream();
        for character in [b'h', b'i'] {
            stream.put(0, plain(character)).unwrap();
        }
        // Staged: the committed document is still empty.
        assert_eq!(stream.raw_length(), 0);
        assert_eq!(stream.flush_all(), 2);
        assert_eq!(stream.raw_length(), 2);
        assert_eq!(stream.char_at(0), Some(u32::from(b'h')));
        assert_eq!(stream.char_at(1), Some(u32::from(b'i')));
        assert_eq!(stream.packed_at(2), None);
        // The caret sits after the committed text.
        assert_eq!(stream.cursor_from_id(0).unwrap().array_pos(), 2);
    }

    #[test]
    fn multiple_put_flush_cycles_keep_text_in_order() {
        let mut stream = new_stream();
        for character in [b'a', b'b', b'c', b'd'] {
            stream.put(0, plain(character)).unwrap();
            stream.flush_all();
        }
        assert_eq!(stream.raw_length(), 4);
        assert_eq!(stream.char_at(3), Some(u32::from(b'd')));
    }

    #[test]
    fn deleting_at_the_end_is_a_noop() {
        let mut stream = new_stream();
        assert_eq!(stream.delete(0).unwrap(), 0);
        assert_eq!(stream.backspace(0).unwrap(), 0);
    }

    #[test]
    fn delete_and_backspace_remove_and_shift_the_caret() {
        let mut stream = new_stream();
        for character in [b'a', b'b', b'c'] {
            stream.put(0, plain(character)).unwrap();
        }
        stream.flush_all();
        // Caret at the end (3): backspace removes 'c'.
        assert_eq!(stream.backspace(0).unwrap(), 1);
        assert_eq!(stream.raw_length(), 2);
        assert_eq!(stream.cursor_from_id(0).unwrap().array_pos(), 2);
        // Move to 1 and forward-delete 'b' (index 1).
        stream.move_cursor_to(0, 0, 1).unwrap();
        assert_eq!(stream.delete(0).unwrap(), 1);
        assert_eq!(stream.raw_length(), 1);
        assert_eq!(stream.char_at(0), Some(u32::from(b'a')));
    }

    #[test]
    fn delete_range_validates_bounds() {
        let mut stream = new_stream();
        for character in [b'a', b'b', b'c'] {
            stream.put(0, plain(character)).unwrap();
        }
        stream.flush_all();
        assert!(stream.delete_range(2, 1).is_err());
        assert!(stream.delete_range(0, 9).is_err());
        assert_eq!(stream.delete_range(1, 3).unwrap(), 2);
        assert_eq!(stream.raw_length(), 1);
    }

    #[test]
    fn invalid_cursor_ids_are_rejected_not_panicking() {
        let mut stream = new_stream();
        assert!(stream.put(7, plain('x')).is_err());
        assert!(stream.move_cursor_forward(9, 1).is_err());
        assert!(stream.cursor_row(42).is_err());
        assert_eq!(stream.selection_range(42), None);
        assert_eq!(stream.flush_gap(99), 0);
    }

    #[test]
    fn two_cursors_edit_independently() {
        let mut stream = new_stream();
        let second = stream
            .append_cursor(crate::edit::cursor::Cursor::new())
            .unwrap();
        stream.put(0, plain('a')).unwrap();
        stream.put(second, plain('z')).unwrap();
        assert_eq!(stream.flush_all(), 2);
        // Cursor 0 staged first at base 0; cursor 1 staged at its own
        // base (also 0), committed after in slot order.
        assert_eq!(stream.raw_length(), 2);
        assert_eq!(stream.cursor_from_id(0).unwrap().array_pos(), 1);
        assert_eq!(stream.cursor_from_id(second).unwrap().array_pos(), 2);
    }

    #[test]
    fn shared_gap_stages_and_flushes() {
        let mut stream = new_stream();
        for character in [b'x', b'y'] {
            stream.put_shared(0, plain(character)).unwrap();
        }
        assert_eq!(stream.raw_length(), 0);
        assert_eq!(stream.flush_all(), 2);
        assert_eq!(stream.raw_length(), 2);
        assert_eq!(stream.char_at(0), Some(u32::from(b'x')));
        assert_eq!(stream.char_at(1), Some(u32::from(b'y')));
    }

    #[test]
    fn selection_is_tracked_and_queryable() {
        let mut stream = new_stream();
        for character in [b'a', b'b', b'c'] {
            stream.put(0, plain(character)).unwrap();
        }
        stream.flush_all();
        assert!(stream.set_selection_anchor(0, 0));
        assert!(stream.set_selection_end(0, 2));
        assert_eq!(stream.selection_range(0), Some((0, 2)));
        assert!(stream.clear_selection(0));
        assert_eq!(stream.selection_range(0), None);
        assert!(!stream.set_selection_anchor(9, 0));
    }

    #[test]
    fn line_navigation_works_after_flush() {
        let mut stream = new_stream();
        for character in [b'a', b'b', b'\n', b'c'] {
            stream.put(0, plain(character)).unwrap();
        }
        stream.flush_all();
        assert_eq!(stream.move_cursor_next_line(0).unwrap(), 4);
        assert_eq!(stream.cursor_row(0).unwrap(), 1);
        // Back to row 0, same column (1) -> position 1.
        assert_eq!(stream.move_cursor_prev_line(0).unwrap(), 1);
        assert_eq!(stream.cursor_row(0).unwrap(), 0);
    }

    #[test]
    fn buffer_queries_bound_check() {
        let mut stream = new_stream();
        stream.put(0, plain('q')).unwrap();
        stream.flush_all();
        assert_eq!(stream.raw_slice(0, 1), Some(&[plain('q')][..]));
        assert_eq!(stream.raw_slice(1, 0), None);
        assert_eq!(stream.raw_slice(0, 9), None);
        assert_eq!(stream.type_at(0), Some(T_COMMON));
        assert_eq!(stream.char_at(5), None);
        assert_eq!(stream.line_width_at(0), Some(1));
    }

    #[test]
    fn overwrite_replaces_in_place_on_flush() {
        let mut stream = new_stream();
        for character in [b'o', b'l', b'd'] {
            stream.put(0, plain(character)).unwrap();
        }
        stream.flush_all();
        stream.move_cursor_to(0, 0, 1).unwrap();
        stream.overwrite(0, plain('X')).unwrap();
        assert_eq!(stream.flush_all(), 0); // merge consumed, no growth
        assert_eq!(stream.raw_length(), 3);
        assert_eq!(stream.char_at(1), Some(u32::from(b'X')));
    }

    #[test]
    fn replace_swaps_the_element() {
        let mut stream = new_stream();
        for character in [b'a', b'b'] {
            stream.put(0, plain(character)).unwrap();
        }
        stream.flush_all();
        stream.move_cursor_to(0, 0, 1).unwrap();
        stream.replace(0, plain('Z')).unwrap();
        stream.flush_all();
        assert_eq!(stream.raw_length(), 2);
        assert_eq!(stream.char_at(1), Some(u32::from(b'Z')));
    }

    #[test]
    fn long_shared_runs_flush_across_chunk_boundaries() {
        let mut stream = new_stream();
        let count = 200usize;
        for index in 0..count {
            stream
                .put_shared(0, plain(b'a' + (index % 26) as u8))
                .unwrap();
        }
        assert_eq!(stream.flush_all(), count);
        assert_eq!(stream.raw_length(), count);
        for index in 0..count {
            assert_eq!(stream.char_at(index), Some(u32::from(b'a' + (index % 26) as u8)));
        }
    }

    #[test]
    fn plain_edits_preserve_the_sparse_line_index() {
        fn anchors_valid(stream: &TestStream) -> bool {
            // SAFETY: the test drives the writable single-threaded.
            unsafe { &*stream.base.line_anchors.get() }
                .as_ref()
                .is_some_and(|anchors| anchors.is_valid())
        }

        let mut stream = new_stream();
        for row in 0..200 {
            for column in 0..40 {
                stream
                    .put(0, plain(b'a' + ((row + column) % 26) as u8))
                    .unwrap();
            }
            stream.put(0, plain('\n')).unwrap();
        }
        stream.flush_all();
        // Prime the sparse index with a distant jump.
        stream.move_cursor_to(0, 200, 0).unwrap();
        assert_eq!(stream.cursor_row(0).unwrap(), 200);
        assert!(anchors_valid(&stream));

        // Typing without a newline, deleting it, and plain caret
        // motion all preserve the row layout.
        stream.move_cursor_to(0, 200, 0).unwrap();
        stream.put(0, plain('x')).unwrap();
        stream.flush_all();
        assert!(anchors_valid(&stream));
        assert_eq!(stream.cursor_row(0).unwrap(), 200);
        stream.backspace(0).unwrap();
        assert!(anchors_valid(&stream));
        stream.move_cursor_forward(0, 1).unwrap();
        assert!(anchors_valid(&stream));

        // Appending a newline extends the row count but leaves every
        // existing anchor start in place, so the index survives.
        stream.put(0, plain('\n')).unwrap();
        stream.flush_all();
        assert!(anchors_valid(&stream));
        assert_eq!(stream.cursor_row(0).unwrap(), 201);

        // A newline in the middle moves later starts and row numbers,
        // so the index must be invalidated.
        stream.move_cursor_to(0, 50, 0).unwrap();
        stream.put(0, plain('\n')).unwrap();
        stream.flush_all();
        assert!(!anchors_valid(&stream));
    }

    #[test]
    fn metadata_accessors_work() {
        let mut stream = new_stream();
        assert_eq!(stream.v_space(), 0);
        *stream.flags_mut() = 0b101;
        assert_eq!(stream.flags(), 0b101);
        stream.name_mut()[..4].copy_from_slice(b"doc\x00");
        assert_eq!(stream.name(), "doc");
        assert!(stream.event_bridge().is_none());
        // No gaps exist yet, so nothing is close to position 0.
        assert_eq!(unsafe { stream.proximity(0) }, None);
    }
}
