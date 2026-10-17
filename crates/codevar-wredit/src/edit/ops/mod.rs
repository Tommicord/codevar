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
//! Each `TextEditable*` trait is implemented by [`StreamWritable`]
//! in its own submodule; the helpers below (cursor validation, raw
//! insert/remove bookkeeping, private-gap staging) are shared by the
//! concrete operations.

pub mod op_buffer_query;
pub mod op_cursor;
pub mod op_delete;
pub mod op_gap;
pub mod op_overwrite;
pub mod op_put;
pub mod op_replace;
pub mod op_selection;

use codevar_logger::log_error;

use core::ptr;

use super::traits::Writable;
use super::writable::{GAP_NONE, MAX_CURSORS, MAX_GAPS, SHARED_GAP_SLOTS, StreamWritable, WritableEvent};
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
            let reactivated = self
                .base
                .cursors_ref()
                .iter()
                .position(Option::is_some)
                .unwrap_or(0);
            self.base.active_cursor_id = reactivated;
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
    pub(super) fn validate_cursor(&self, cursor_id: usize) -> WreditResult<usize> {
        if !self.base.valid_cursor_id(cursor_id) {
            log_error!("wredit: invalid cursor id {cursor_id}");
            return Err(WreditError::InvalidCursor { cursor: cursor_id });
        }
        Ok(cursor_id)
    }

    /// The cursor's current gap mapping, or [`GAP_NONE`].
    pub(super) fn cursor_mapping(&self, cursor_id: usize) -> u16 {
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
    pub(super) fn ensure_private_gap(&mut self, cursor_id: usize, base: usize) -> WreditResult<usize> {
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
    pub(super) fn ensure_shared_slot(&mut self, cursor_id: usize) -> WreditResult<usize> {
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
    pub(super) fn flush_cursor_gap(&mut self, cursor_id: usize) {
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
    pub(super) fn stage_private(&mut self, cursor_id: usize, value: Raw, base: usize) -> WreditResult<usize> {
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
    pub(super) fn move_caret_to(&mut self, cursor_id: usize, position: usize) -> usize {
        let Some(from) = self
            .base
            .cursor(cursor_id)
            .map(super::cursor::Cursor::array_pos)
        else {
            return position;
        };
        // Raise the logical bound (staged edits grow it) before the
        // caret clamps against it.
        self.base.refresh_cursor_views();
        let Some(cursor) = self.base.cursor(cursor_id) else {
            return position;
        };
        let to = cursor.set_array_pos(position);
        self.base
            .emit(WritableEvent::cursor_moved(cursor_id, from, to));
        to
    }

    /// Commits a raw insertion at `at` with full cursor/gap
    /// bookkeeping.
    ///
    /// Returns the number of inserted elements.
    pub(super) fn commit_insertion(&mut self, at: usize, values: &[Raw]) -> usize {
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
        self.base.refresh_cursor_views();
        self.base.shift_cursors(at, inserted, false);
        self.base.shift_gap_bases(at, inserted, false);
        self.base.shift_shared_bases(at, inserted, false);
        self.base.refresh_cursor_views();
        self.base
            .emit(WritableEvent::buffer_changed(at, inserted));
        inserted
    }

    /// Removes `count` committed elements at `start` with full
    /// cursor/gap bookkeeping.
    ///
    /// Returns the number of removed elements.
    pub(super) fn commit_removal(&mut self, start: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let old_length = self.base.raw_length;
        let end = start.saturating_add(count);
        if end > old_length {
            log_error!("wredit: raw removal {start}..{end} exceeds length {old_length}");
            return 0;
        }
        {
            let raw = self.base.raw_mut();
            let ptr = raw.as_mut_ptr();
            // Safety: `end <= raw_length <= allocated`, so both ranges
            // are in bounds; `ptr::copy` (memmove) tolerates the tail
            // overlap with its destination.
            unsafe {
                ptr::copy(ptr.add(end), ptr.add(start), old_length - end);
            }
            let new_length = old_length - count;
            raw.set_count(new_length);
            self.base.raw_length = new_length;
        }
        self.base.vspace = self.base.raw_ref().allocated();
        self.base.shift_cursors(start, count, true);
        self.base.shift_gap_bases(start, count, true);
        self.base.shift_shared_bases(start, count, true);
        self.base.refresh_cursor_views();
        self.base
            .emit(WritableEvent::buffer_changed(start, count));
        count
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
