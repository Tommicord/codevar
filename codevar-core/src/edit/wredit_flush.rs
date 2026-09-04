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

use super::{BaseWritable, SharedGap, WritableEvent, WritableEventType, WritableGap};
use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// Bit shift for the type field in a packed writable value.
const SHIFT_TYPE: u32 = 28;
/// Bit shift for the character field in a packed writable value.
const SHIFT_CHAR: u32 = 0;
/// Mask for the 4-bit type field.
const TYPE_MASK: u32 = 0xF;
/// Mask for the 24-bit character field.
const CHAR_MASK: u32 = 0xFFFFFF;
/// Packed type ID for mergeable gap data.
const T_MERGE: u32 = 0xD;
/// Packed type ID for common (direct) data.
const T_COMMON: u32 = 0xE;

/// Initial capacity for the temporary merge output buffer.
const INIT_MERGE_BUFFER_COUNT: usize = 4096;

/// Resolved gap state needed to perform a flush for one cursor.
struct FlushGapContext<Raw, const GAP_SIZE: usize> {
    gap: *mut WritableGap<Raw, GAP_SIZE>,
    gap_buf: *mut Raw,
    gap_fill: usize,
    raw_offset: usize,
    shared_gap_idx: Option<usize>,
}

/// Extracts the type field from a packed 32-bit value.

fn packed_type(packed: u32) -> u32 {
    (packed >> SHIFT_TYPE) & TYPE_MASK
}

/// Extracts the character field from a packed 32-bit value.
fn packed_char(packed: u32) -> u32 {
    (packed >> SHIFT_CHAR) & CHAR_MASK
}

/// Creates a packed 32-bit value from type and character fields.
fn make_packed(type_: u32, ch: u32) -> u32 {
    ((type_ & TYPE_MASK) << SHIFT_TYPE) | ((ch & CHAR_MASK) << SHIFT_CHAR)
}

/// Returns `true` when the raw slice contains at least one merge marker.
unsafe fn region_has_merge_markers(
    source: *const u32,
    offset: usize,
    count: usize,
) -> bool {
    if source.is_null() || count == 0 {
        return false;
    }
    for i in 0..count {
        if packed_type(source.add(offset + i).read()) == T_MERGE {
            return true;
        }
    }
    false
}

/// Scalar flush for gap buffer operations.
///
/// Reads packed values from `source`, resolves `T_MERGE` entries from the
/// circular `gap` buffer, and writes decoded bytes to `out`.
///
/// # Safety
///
/// All pointers must be valid for the given `offset`, `count`, and `gap_size`.
///
/// Returns the number of bytes written to `out`.
pub unsafe fn flush_stream_write_scalar(
    source: *const u32,
    out: *mut u8,
    gap: *mut u32,
    offset: usize,
    count: usize,
    gap_size: usize,
) -> usize {
    if source.is_null() || out.is_null() || gap.is_null() || count == 0 || gap_size == 0 {
        return 0;
    }
    let end = offset.saturating_add(count);
    let gap_mask = gap_size - 1;
    let mut index = offset;
    let mut written = 0usize;

    while index < end {
        let packed = source.add(index).read();
        let type_field = packed_type(packed);

        if type_field == T_MERGE {
            let mut merge_idx = 0usize;
            while merge_idx < gap_size {
                let gap_index = index.wrapping_add(merge_idx) & gap_mask;
                let gap_packed = gap.add(gap_index).read();
                if packed_type(gap_packed) != T_MERGE {
                    break;
                }
                let ch = packed_char(gap_packed);
                out.add(written).write(ch as u8);
                gap.add(gap_index).write(make_packed(T_COMMON, ch));
                merge_idx += 1;
                written += 1;
            }
            index = index.saturating_add(merge_idx);
        } else if type_field == T_COMMON {
            let ch = packed_char(packed);
            out.add(written).write(ch as u8);
            index += 1;
            written += 1;
        } else {
            index += 1;
        }
    }

    written
}

unsafe fn flush_stream_write(
    source: *const u32,
    out: *mut u8,
    gap: *mut u32,
    offset: usize,
    count: usize,
    gap_size: usize,
) -> usize {
    flush_stream_write_scalar(source, out, gap, offset, count, gap_size)
}

/// Copies packed values from a circular gap buffer into the raw buffer.
unsafe fn flush_gap_buffer_to_raw<Raw>(
    raw: *mut Raw,
    gap: *mut Raw,
    raw_offset: usize,
    gap_fill: usize,
    gap_size: usize,
) -> usize {
    if raw.is_null() || gap.is_null() || gap_fill == 0 || gap_size == 0 {
        return 0;
    }

    let gap_mask = gap_size - 1;
    let count = gap_fill.min(gap_size);

    for i in 0..count {
        let gap_index = i & gap_mask;
        let packed = gap.add(gap_index).read();
        raw.add(raw_offset + i).write(packed);
    }

    count
}

/// Writes decoded merge-buffer bytes back into the raw buffer as packed `u32` values.
unsafe fn write_bytes_to_raw(
    raw: *mut u32,
    raw_offset: usize,
    out: *const u8,
    count: usize,
) {
    for i in 0..count {
        let ch = out.add(i).read() as u32;
        raw.add(raw_offset + i).write(make_packed(T_COMMON, ch));
    }
}

/// Ensures the merge output buffer is allocated and returns its pointer.
unsafe fn ensure_merge_buffer<Buf>(
    buffer: &mut Option<NonNull<Buf>>,
    capacity: usize,
) -> *mut Buf {
    if buffer.is_none() {
        let layout = Layout::array::<Buf>(capacity).unwrap_or_else(|_| {
            // Fallback to a safe default layout if the requested one is invalid
            Layout::new::<Buf>()
        });
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        *buffer = NonNull::new(ptr as *mut Buf);
    }
    buffer.map_or(std::ptr::null_mut(), |b| b.as_ptr())
}

/// Computes the element offset of `gap_start` within the raw allocation.
unsafe fn raw_offset_from_gap<Raw>(raw: *const Raw, gap_start: *const Raw) -> usize {
    if raw.is_null() || gap_start.is_null() {
        return 0;
    }
    gap_start.offset_from(raw).max(0) as usize
}

/// Resolves the gap buffer and raw offset for the given cursor.
unsafe fn resolve_flush_context<Raw, Buf, const GAP_SIZE: usize>(
    base: &BaseWritable<Raw, Buf, GAP_SIZE>,
    cursor_id: usize,
) -> Option<FlushGapContext<Raw, GAP_SIZE>> {
    let raw = base.raw_mut_ptr();
    if raw.is_null() {
        return None;
    }

    if base.is_cursor_in_shared_gap(cursor_id) {
        let gap_idx = base
            .shared_gap_index(cursor_id)
            .unwrap_or_else(|| usize::MAX);
        let shared_gaps = base.shared_gaps_mut();
        let shared_gap = shared_gaps.get_mut(gap_idx)?;
        let gap = shared_gap.gap_mut() as *mut WritableGap<Raw, GAP_SIZE>;
        let gap_ref = unsafe { &mut *gap };
        let gap_buf = gap_ref.gap_ptr_mut().buff_mut();
        if gap_buf.is_null() {
            return None;
        }
        let gap_fill = gap_ref.gap_ptr().gap_ptr().min(GAP_SIZE);
        let raw_offset = raw_offset_from_gap(raw, gap_ref.gap_start());
        return Some(FlushGapContext {
            gap,
            gap_buf,
            gap_fill,
            raw_offset,
            shared_gap_idx: Some(gap_idx),
        });
    }

    let gaps = base.gaps_mut();
    let gap_ref = gaps.get_mut(0)?;
    let gap = gap_ref as *mut WritableGap<Raw, GAP_SIZE>;
    let gap_buf = gap_ref.gap_ptr_mut().buff_mut();
    if gap_buf.is_null() {
        return None;
    }
    let gap_fill = gap_ref.gap_ptr().gap_ptr().min(GAP_SIZE);
    let raw_offset = raw_offset_from_gap(raw, gap_ref.gap_start());
    Some(FlushGapContext {
        gap,
        gap_buf,
        gap_fill,
        raw_offset,
        shared_gap_idx: None,
    })
}

/// Resets remaining space and cursor positions for cursors in a shared gap.
unsafe fn reset_shared_gap_regions<Raw, Buf, const GAP_SIZE: usize>(
    base: &BaseWritable<Raw, Buf, GAP_SIZE>,
    shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
    shared_gap_idx: usize,
) {
    let cursor_to_gap = base.cursor_to_gap();
    for (cursor_id, &gap_idx) in cursor_to_gap.iter().enumerate() {
        if gap_idx != shared_gap_idx {
            continue;
        }
        if let Some(region) = shared_gap.cursor_region_mut(cursor_id) {
            let size = region.region_size();
            let start = region.region_start();
            region.set_region_size(size);
            region.set_cursor_position(start);
        }
    }
}

/// Clears the gap circular buffer and resets shared cursor regions.
unsafe fn reset_gap_after_flush<Raw, Buf, const GAP_SIZE: usize>(
    base: &BaseWritable<Raw, Buf, GAP_SIZE>,
    ctx: &FlushGapContext<Raw, GAP_SIZE>,
) {
    // Check if the gap pointer is valid before dereferencing
    if ctx.gap.is_null() {
        return;
    }
    let gap_ref = &mut *ctx.gap;
    let gap_ptr = gap_ref.gap_ptr_mut();
    let fill_bytes = GAP_SIZE * std::mem::size_of::<Raw>();
    gap_ptr.set_gap_ptr(0);
    let buff_ptr = gap_ptr.buff_mut();
    if !buff_ptr.is_null() && fill_bytes > 0 {
        let raw_ptr = base.raw_mut_ptr();
        let gap_start = gap_ref.gap_start();
        let gap_end = gap_ref.gap_end();
        // If gap_start equals raw_ptr, the gap buffer was never allocated separately
        // and buff_ptr is dangling/uninitialized
        if gap_start != raw_ptr && gap_end != raw_ptr {
            // Ensure the gap bounds are reasonable (end should be >= start)
            let gap_start_usize = gap_start as usize;
            let gap_end_usize = gap_end as usize;
            if gap_end_usize >= gap_start_usize {
                let gap_size = gap_end_usize - gap_start_usize;
                if gap_size >= fill_bytes {
                    std::ptr::write_bytes(buff_ptr, 0, fill_bytes);
                }
            }
        }
    }

    if let Some(gap_idx) = ctx.shared_gap_idx {
        let shared_gaps = base.shared_gaps_mut();
        if let Some(shared_gap) = shared_gaps.get_mut(gap_idx) {
            reset_shared_gap_regions(base, shared_gap, gap_idx);
        }
    }
}

/// Emits a `BufferChanged` event when an event bridge is configured.
fn emit_buffer_changed<Raw, Buf, const GAP_SIZE: usize>(
    base: &BaseWritable<Raw, Buf, GAP_SIZE>,
    cursor_id: usize,
    written: usize,
) {
    if let Some(bridge) = base.event_bridge() {
        let mut event = WritableEvent::new(WritableEventType::BufferChanged);
        event.cursor_id = cursor_id;
        event.size = written;
        event.message = format!("flush: merged {written} element(s) into raw buffer");
        let _ = bridge.emit_event(event);
    }
}

/// Performs the codevar-core merge: gap buffer and/or merge markers into raw storage.
unsafe fn perform_flush<Raw, Buf, const GAP_SIZE: usize>(
    base: &mut BaseWritable<Raw, Buf, GAP_SIZE>,
    ctx: &FlushGapContext<Raw, GAP_SIZE>,
) -> usize {
    let raw = base.raw_mut_ptr();
    let raw_u32 = raw as *mut u32;
    let gap_u32 = ctx.gap_buf as *mut u32;
    let count = ctx.gap_fill.min(GAP_SIZE);
    let merge_out = ensure_merge_buffer(base.buffer_mut(), INIT_MERGE_BUFFER_COUNT);

    let written = if region_has_merge_markers(raw_u32, ctx.raw_offset, count) {
        let bytes = flush_stream_write(
            raw_u32,
            merge_out as *mut u8,
            gap_u32,
            ctx.raw_offset,
            count,
            GAP_SIZE,
        );
        write_bytes_to_raw(raw_u32, ctx.raw_offset, merge_out as *const u8, bytes);
        bytes
    } else {
        flush_gap_buffer_to_raw(raw, ctx.gap_buf, ctx.raw_offset, ctx.gap_fill, GAP_SIZE)
    };

    written
}

/// Flushes the gap buffer for the cursor identified by `cursor_id`.
///
/// Merges pending gap data (including shared-gap cursor regions) into the raw
/// buffer at the gap bounds, resets the circular gap buffer, and rebalances
/// shared cursor regions when applicable.
///
/// # Safety
///
/// `cursor_id` must refer to a valid cursor with an allocated gap buffer.
///
/// Returns the number of elements written into the raw buffer.
pub unsafe fn flush_gap<Raw, Buf, const GAP_SIZE: usize>(
    base: &mut BaseWritable<Raw, Buf, GAP_SIZE>,
    cursor_id: usize,
) -> usize {
    if base.cursor_from_id(cursor_id).is_none() {
        return 0;
    }
    let Some(ctx) = resolve_flush_context(base, cursor_id) else {
        return 0;
    };
    if ctx.gap_fill == 0 {
        return 0;
    }
    let written = perform_flush(base, &ctx);
    if written > 0 {
        let new_len = base.raw_length().saturating_add(written);
        if new_len > *base.raw_length_mut() {
            *base.raw_length_mut() = new_len;
        }
    }
    reset_gap_after_flush(base, &ctx);
    emit_buffer_changed(base, cursor_id, written);
    written
}

/// Flushes every gap that contains pending data (regular and shared).
///
/// # Safety
///
/// All cursors must remain valid for the duration of the flush.
///
/// Returns the total number of elements merged into the raw buffer.
pub unsafe fn flush_all<Raw, Buf, const GAP_SIZE: usize>(
    base: &mut BaseWritable<Raw, Buf, GAP_SIZE>,
) -> usize {
    let cursor_count = base.cursor_count();
    let mut total = 0usize;

    for cursor_id in 0..cursor_count {
        total = total.saturating_add(flush_gap(base, cursor_id));
    }

    total
}
