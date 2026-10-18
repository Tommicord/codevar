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

//! Commit path: staged gap contents into the raw document.
//!
//! Edits accumulate in private gaps and shared gap regions; the
//! functions here commit them into the raw buffer in a single block
//! move per flush. [`flush_stream_write_scalar`] walks the staged
//! stream and resolves [`T_MERGE`] elements: a merge element
//! overwrites the committed element that follows the insertion point
//! (an overwrite edit) instead of growing the document. Plain
//! elements are batched into stack-resident chunks so a typical
//! flush performs O(chunks) block moves rather than one per element.
//!
//! Every entry point validates its inputs; an invalid position or a
//! failed allocation is logged and reported as a no-op (the staged
//! data stays intact and can be retried) — the commit path never
//! aborts and never panics.

use alloc::vec::Vec;
use core::mem::MaybeUninit;

use codevar_logger::log_error;

use super::base::{T_COMMON, T_MERGE, WritableAlignedPtr, make_packed, packed_char, packed_type};
use super::writable::{MAX_CURSORS, MAX_GAPS, SHARED_GAP_SLOTS, StreamWritable, WritableEvent};
use crate::arch::{memcpy, memmove};

/// Number of plain elements resolved on the stack per block move.
const FLUSH_CHUNK: usize = 0x40;

/// Inserts `src` into the raw document at `at`.
///
/// The raw buffer is grown first (geometric, checked arithmetic), so
/// the insertion cannot fail halfway. Returns the new document
/// length; an invalid position or a refused allocation leaves the
/// document untouched and returns `raw_length`.
pub(crate) fn write_bytes_to_raw<Raw: Copy>(
    raw: &mut WritableAlignedPtr<Raw, 64>,
    raw_length: usize,
    at: usize,
    src: &[Raw],
) -> usize {
    if src.is_empty() {
        return raw_length;
    }
    if at > raw_length {
        log_error!("wredit: raw insert position {at} exceeds length {raw_length}");
        return raw_length;
    }
    let Some(new_length) = raw_length.checked_add(src.len()) else {
        log_error!("wredit: raw insert length overflow");
        return raw_length;
    };
    if raw.ensure_capacity(new_length).is_err() {
        return raw_length;
    }
    let ptr = raw.as_mut_ptr();
    let tail_len = raw_length - at;
    let elem = core::mem::size_of::<Raw>();
    let byte_len = tail_len * elem;
    let base = ptr.cast::<u8>();
    // Prefetch the tail range ahead of the shift so the cache lines are
    // hot when the memmove runs. Prefetch both source (for the load
    // side) and destination (for the store side).
    if byte_len >= crate::arch::CACHE_LINE_SIZE {
        unsafe {
            crate::arch::prefetch(base.add(at * elem), crate::arch::PREFETCH_T0);
            crate::arch::prefetch(base.add((at + src.len()) * elem), crate::arch::PREFETCH_T0);
        }
    }
    // Safety: `ensure_capacity(new_length)` guarantees room for
    // `new_length` elements; `at <= raw_length` and `src.len() >= 1`
    // make both ranges in bounds. `memmove` tolerates the overlap
    // between the shifted tail and its destination.
    unsafe {
        memmove(base.add((at + src.len()) * elem), base.add(at * elem), byte_len);
        memcpy(
            base.add(at * elem),
            src.as_ptr().cast::<u8>(),
            core::mem::size_of_val(src),
        );
    }
    raw.set_count(new_length);
    new_length
}

/// Commits the staged stream `values` into the raw document at `at`,
/// resolving [`T_MERGE`] elements on the way.
///
/// Resolution rules:
///
/// * a [`T_COMMON`] (or unknown-type) element is inserted as-is;
/// * a [`T_MERGE`] element overwrites the committed element at the
///   current stream position when one exists (the overwrite semantics
///   behind [`super::traits::TextEditableOverwrite`]), otherwise it
///   is inserted as a plain element.
///
/// Returns `Some(new_document_length)` — which may equal
/// `raw_length` when every staged element was a merge that consumed
/// existing text. Returns `None` on an invalid `at`, a length
/// overflow or a refused allocation; the document is then left
/// untouched and the caller keeps the staged data to retry.
pub(crate) fn flush_stream_write_scalar<Raw: Copy + Into<u32> + From<u32>>(
    raw: &mut WritableAlignedPtr<Raw, 64>,
    raw_length: usize,
    at: usize,
    values: &[Raw],
) -> Option<usize> {
    if values.is_empty() {
        return Some(raw_length);
    }
    if at > raw_length {
        log_error!("wredit: flush position {at} exceeds length {raw_length}");
        return None;
    }
    // Worst case: every staged element inserts one new element.
    let Some(worst_case) = raw_length.checked_add(values.len()) else {
        log_error!("wredit: flush length overflow");
        return None;
    };
    if raw.ensure_capacity(worst_case).is_err() {
        return None;
    }
    let ptr = raw.as_mut_ptr();
    let mut length = raw_length;
    let mut stream = at;
    // Stack scratch for runs of plain elements; committing them as
    // one block per chunk keeps flushes O(chunks * tail) instead of
    // O(elements * tail).
    let mut pending: [MaybeUninit<Raw>; FLUSH_CHUNK] = [MaybeUninit::uninit(); FLUSH_CHUNK];
    let mut pending_len = 0;

    for &value in values {
        let element_type = packed_type(Into::<u32>::into(value));
        if element_type == T_MERGE {
            // Settle the pending plain run first, then overwrite or
            // insert the merge element at the stream position.
            // Safety: `ptr` is the live buffer, `stream <= length`
            // and the allocation covers the worst-case growth.
            length = unsafe { commit_pending(ptr, length, stream, &pending, pending_len) };
            stream += pending_len;
            pending_len = 0;
            // Safety: `stream <= length <= capacity`, so the write is
            // in bounds whether it replaces an element or appends.
            unsafe {
                ptr.add(stream).write(Raw::from(make_packed(
                    packed_char(Into::<u32>::into(value)),
                    T_COMMON,
                )));
            }
            if stream < length {
                stream += 1;
            } else {
                length += 1;
                stream += 1;
            }
        } else {
            pending[pending_len].write(value);
            pending_len += 1;
            if pending_len == FLUSH_CHUNK {
                // Safety: same invariants as above; the allocation
                // covers the worst-case growth.
                length = unsafe { commit_pending(ptr, length, stream, &pending, pending_len) };
                stream += pending_len;
                pending_len = 0;
            }
        }
    }
    // Safety: final settle; same invariants as above.
    length = unsafe { commit_pending(ptr, length, stream, &pending, pending_len) };
    raw.set_count(length);
    Some(length)
}

/// Inserts the first `count` entries of `pending` at `at`, shifting
/// the tail with one block move. Returns the new length.
///
/// # Safety
///
/// `ptr` must reference a buffer with room for `at + count` elements
/// (the caller pre-grew it), and `count <= FLUSH_CHUNK` entries of
/// `pending` must be initialized.
unsafe fn commit_pending<Raw: Copy>(
    ptr: *mut Raw,
    length: usize,
    at: usize,
    pending: &[MaybeUninit<Raw>; FLUSH_CHUNK],
    count: usize,
) -> usize {
    if count == 0 {
        return length;
    }
    let tail_len = length - at;
    let elem = core::mem::size_of::<Raw>();
    let byte_len = tail_len * elem;
    let base = ptr.cast::<u8>();
    // Prefetch the tail range ahead of the block move so the cache
    // lines are hot when the memmove runs.
    if byte_len >= crate::arch::CACHE_LINE_SIZE {
        unsafe {
            crate::arch::prefetch(base.add(at * elem), crate::arch::PREFETCH_T0);
            crate::arch::prefetch(base.add((at + count) * elem), crate::arch::PREFETCH_T0);
        }
    }
    // Safety: bounds follow from the caller's pre-growth contract;
    // `memmove` tolerates the tail overlap.
    unsafe {
        memmove(base.add((at + count) * elem), base.add(at * elem), byte_len);
        let dst = ptr.add(at);
        for (index, element) in pending.iter().enumerate().take(count) {
            dst.add(index).write(element.assume_init_read());
        }
    }
    length + count
}

/// Commits the `staged` elements behind `buffer` at `at` through the
/// scalar stream flusher.
///
/// Returns `None` when the commit was refused; see
/// [`flush_stream_write_scalar`].
///
/// # Safety
///
/// `buffer` must point to at least `staged` initialized `Raw`
/// elements that stay valid for the whole call and do not alias the
/// raw document.
pub(crate) fn flush_gap_buffer_to_raw<Raw: Copy + Into<u32> + From<u32>>(
    raw: &mut WritableAlignedPtr<Raw, 64>,
    raw_length: usize,
    buffer: *const Raw,
    staged: usize,
    at: usize,
) -> Option<usize> {
    if staged == 0 {
        return Some(raw_length);
    }
    if buffer.is_null() {
        log_error!("wredit: flush of a null gap buffer");
        return None;
    }
    // Safety: the caller guarantees `staged` initialized elements at
    // `buffer`; the slice shares that guarantee and is read-only.
    let values = unsafe { core::slice::from_raw_parts(buffer, staged) };
    flush_stream_write_scalar(raw, raw_length, at, values)
}

/// Allocates a fresh uninitialized merge buffer of `capacity`
/// elements.
///
/// The engine's single allocation funnel — re-exported from
/// [`super::base`] next to the commit path that uses it. A refusal
/// is logged and reported as an error, never aborted.
pub use super::base::ensure_merge_buffer;

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE>
where
    Raw: Copy + Into<u32> + From<u32> + From<Buf>,
    Buf: Copy + From<Raw>,
{
    /// Commits the staged content of the private gap `gap_index`.
    ///
    /// The staged run is inserted at the gap's base position, the
    /// owning cursor lands after its own text, other cursors and gap
    /// bases past the insertion point shift right, and the gap is
    /// released. Returns the number of committed elements (`0` for a
    /// bad index, an empty gap or a refused allocation).
    pub fn flush_gap(&mut self, gap_index: usize) -> usize {
        if gap_index < MAX_GAPS {
            self.flush_private_gap(gap_index)
        } else {
            self.flush_shared_slot(gap_index - MAX_GAPS)
        }
    }

    /// Commits every private gap and shared gap region, in index
    /// order. Returns the total number of committed elements.
    pub fn flush_all(&mut self) -> usize {
        let mut total: usize = 0;
        for index in 0..MAX_GAPS {
            total = total.saturating_add(self.flush_private_gap(index));
        }
        for slot in 0..SHARED_GAP_SLOTS {
            total = total.saturating_add(self.flush_shared_slot(slot));
        }
        total
    }

    /// Flushes one private gap; see [`Self::flush_gap`].
    fn flush_private_gap(&mut self, gap_index: usize) -> usize {
        let Some(gap) = self.base.gaps_ref().get(gap_index) else {
            return 0;
        };
        let staged = gap.staged();
        if staged == 0 {
            return 0;
        }
        let content = gap.content();
        let base = gap.base();
        let old_length = self.base.raw_length;
        let at = base.min(old_length);
        // Safety: `content` points at `staged` initialized elements
        // of the gap window (never aliased by the raw document).
        let Some(new_length) = flush_gap_buffer_to_raw(self.base.raw_mut(), old_length, content, staged, at)
        else {
            // Allocation refused (or position invalid): the staged
            // run stays put for a later retry.
            return 0;
        };
        let committed = new_length - old_length;
        self.base.raw_length = new_length;
        self.base.raw_mut().set_count(new_length);
        self.base.vspace = self.base.raw_ref().allocated();
        self.commit_bookkeeping(gap_index, at, committed, staged);
        self.base
            .emit(WritableEvent::buffer_changed(at, staged));
        committed
    }

    /// Flushes one shared gap slot; see [`Self::flush_gap`].
    fn flush_shared_slot(&mut self, slot: usize) -> usize {
        let Some(descriptor) = self.base.shared_ref().slot(slot) else {
            return 0;
        };
        if !descriptor.is_allocated() {
            return 0;
        }
        let size = descriptor.size();
        if size == 0 {
            return 0;
        }
        let base = descriptor.base();
        let region = base.saturating_add(size);
        // The arena is a `Buf` array; convert the region to `Raw`
        // (identity for the standard `u32` instantiation).
        let converted: Vec<Raw> = {
            let arena = self.base.buffer_ref();
            let Some(slice) = arena.slice().get(base..region) else {
                log_error!("wredit: shared gap region escapes the staging arena");
                return 0;
            };
            slice
                .iter()
                .map(|element| Raw::from(*element))
                .collect()
        };
        let old_length = self.base.raw_length;
        let at = base.min(old_length);
        let Some(new_length) = flush_stream_write_scalar(self.base.raw_mut(), old_length, at, &converted)
        else {
            return 0;
        };
        let committed = new_length - old_length;
        self.base.raw_length = new_length;
        self.base.raw_mut().set_count(new_length);
        self.base.vspace = self.base.raw_ref().allocated();
        // Release the slot before the cursor/membership bookkeeping.
        self.base.shared_mut().free_slot(slot);
        self.base.refresh_cursor_views();
        for id in 0..MAX_CURSORS {
            let Some(cursor) = self
                .base
                .cursors_ref()
                .get(id)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            if descriptor.contains_cursor(id) {
                // Members land after the whole shared run, whatever
                // each of them contributed.
                cursor.set_array_pos(at.saturating_add(size));
                cursor.set_anchor(at.saturating_add(size));
                if self.base.cursor_to_gap[id] == (MAX_GAPS + slot) as u16 {
                    self.base.cursor_to_gap[id] = super::writable::GAP_NONE;
                }
            } else {
                let pos = cursor.array_pos();
                if pos > at {
                    cursor.set_array_pos(pos.saturating_add(committed));
                }
                let anchor = cursor.anchor();
                if anchor > at {
                    cursor.set_anchor(anchor.saturating_add(committed));
                }
            }
        }
        self.base.shift_gap_bases(at, committed, false);
        self.base.shift_shared_bases(at, committed, false);
        self.base.refresh_cursor_views();
        self.base
            .emit(WritableEvent::buffer_changed(at, size));
        committed
    }

    /// Post-flush position bookkeeping for a private gap commit.
    ///
    /// The owning cursor lands right after the staged run it typed
    /// (`at + staged`, which covers merge elements that did not grow
    /// the document); every other cursor at or past the insertion
    /// point shifts by the growth `committed`.
    fn commit_bookkeeping(&mut self, gap_index: usize, at: usize, committed: usize, staged: usize) {
        // Raise the logical bounds first: the shifts below may push
        // positions past the *old* logical end (a caret parked at the
        // end while another cursor's staged run commits before it).
        self.base.refresh_cursor_views();
        for id in 0..MAX_CURSORS {
            let Some(cursor) = self
                .base
                .cursors_ref()
                .get(id)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            if self.base.cursor_to_gap[id] == gap_index as u16 {
                cursor.set_array_pos(at.saturating_add(staged));
                cursor.set_anchor(at.saturating_add(staged));
                self.base.cursor_to_gap[id] = super::writable::GAP_NONE;
            } else {
                let pos = cursor.array_pos();
                if pos > at {
                    cursor.set_array_pos(pos.saturating_add(committed));
                }
                let anchor = cursor.anchor();
                if anchor > at {
                    cursor.set_anchor(anchor.saturating_add(committed));
                }
            }
        }
        self.base.gaps_mut().free(gap_index);
        self.base.shift_gap_bases(at, committed, false);
        self.base.shift_shared_bases(at, committed, false);
        self.base.refresh_cursor_views();
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{flush_stream_write_scalar, write_bytes_to_raw};
    use crate::edit::base::{T_COMMON, T_MERGE, WritableAlignedPtr, make_packed};

    // Justification: `unwrap` in tests is permitted by the project
    // lint configuration (`allow-unwrap-in-tests`).

    fn plain(character: impl Into<u32>) -> u32 {
        make_packed(character.into(), T_COMMON)
    }

    fn merge(character: impl Into<u32>) -> u32 {
        make_packed(character.into(), T_MERGE)
    }

    #[test]
    fn write_bytes_inserts_and_shifts_the_tail() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        let length = write_bytes_to_raw(&mut raw, 0, 0, &[plain('a'), plain('b')]);
        assert_eq!(length, 2);
        let length = write_bytes_to_raw(&mut raw, length, 1, &[plain('X')]);
        assert_eq!(length, 3);
        assert_eq!(raw.slice(), &[plain('a'), plain('X'), plain('b')]);
    }

    #[test]
    fn write_bytes_rejects_out_of_range_positions() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        let length = write_bytes_to_raw(&mut raw, 0, 0, &[plain('a')]);
        assert_eq!(write_bytes_to_raw(&mut raw, length, 5, &[plain('b')]), length);
        assert_eq!(raw.slice(), &[plain('a')]);
    }

    #[test]
    fn flush_resolves_merge_as_overwrite() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        let mut length = write_bytes_to_raw(&mut raw, 0, 0, &[plain('h'), plain('i')]);
        // Overwrite the 'i' with a '!' via a merge element.
        length = flush_stream_write_scalar(&mut raw, length, 1, &[merge('!')]).unwrap();
        assert_eq!(length, 2);
        assert_eq!(raw.slice(), &[plain('h'), plain('!')]);
    }

    #[test]
    fn merge_past_the_end_inserts_instead() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        let mut length = write_bytes_to_raw(&mut raw, 0, 0, &[plain('a')]);
        length = flush_stream_write_scalar(&mut raw, length, 1, &[merge('z')]).unwrap();
        assert_eq!(length, 2);
        assert_eq!(raw.slice(), &[plain('a'), plain('z')]);
    }

    #[test]
    fn flush_batches_plain_runs_and_interleaves_merges() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        let mut length = write_bytes_to_raw(&mut raw, 0, 0, &[plain('x'), plain('y')]);
        // Plain run inserts, a merge overwrites the next committed
        // element, another plain run inserts after it.
        let values = [plain('1'), plain('2'), merge('M'), plain('3')];
        length = flush_stream_write_scalar(&mut raw, length, 1, &values).unwrap();
        assert_eq!(length, 5);
        assert_eq!(
            raw.slice(),
            &[plain('x'), plain('1'), plain('2'), plain('M'), plain('3')]
        );
    }

    #[test]
    fn flush_of_long_streams_spans_many_chunks() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        let values: Vec<u32> = (0..300)
            .map(|i| plain(b'a' + (i % 26) as u8))
            .collect();
        let length = flush_stream_write_scalar(&mut raw, 0, 0, &values).unwrap();
        assert_eq!(length, 300);
        assert_eq!(raw.slice().len(), 300);
        assert_eq!(raw.slice()[299], values[299]);
    }

    #[test]
    fn flush_of_empty_values_is_a_noop() {
        let mut raw: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        assert_eq!(flush_stream_write_scalar(&mut raw, 0, 0, &[]), Some(0));
    }
}
