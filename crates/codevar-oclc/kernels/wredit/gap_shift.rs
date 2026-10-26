// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law and agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// Gap buffer shift kernel: parallel insertion/deletion in a gap buffer.
//
// When the Wredit engine inserts or deletes a run of elements at
// position `at`, every element at or past `at` must shift by
// `delta` slots (positive for insert, negative for delete). This
// kernel performs that shift in parallel: each work item owns one
// output element and reads from `src[i - delta]` (insert) or
// `src[i - delta]` (delete), with bounds checks so no work item
// reads or writes out of range.
//
// The host allocates `dst` with room for `len + delta` elements and
// launches with a global size of at least `len + delta`. Work items
// past the valid range write nothing.

// Shifts `src[0..len]` into `dst`, opening a gap of `delta` slots at
// position `at` (insert direction). Elements before `at` stay put;
// elements at or past `at` move right by `delta`.
//
// `dst` must have room for `len + delta` elements.
#[kernel]
fn gap_shift_insert(src: *const int, len: int, at: int, delta: int, dst: *mut int) -> void {
    let i = get_global_id(0) as int;
    let out_len = len + delta;
    if i >= out_len {
        return;
    }
    if i < at {
        // Before the insertion point: copy as-is.
        if i < len {
            dst[i] = src[i];
        }
    } else {
        // At or past the insertion point: shift right by delta.
        let src_index = i - delta;
        if src_index < len {
            dst[i] = src[src_index];
        }
    }
}

// Shifts `src[0..len]` into `dst`, closing a gap of `delta` slots at
// position `at` (delete direction). Elements before `at` stay put;
// elements at or past `at + delta` move left by `delta`.
//
// `dst` must have room for `len - delta` elements.
#[kernel]
fn gap_shift_delete(src: *const int, len: int, at: int, delta: int, dst: *mut int) -> void {
    let i = get_global_id(0) as int;
    let out_len = len - delta;
    if i >= out_len {
        return;
    }
    if i < at {
        // Before the deletion point: copy as-is.
        dst[i] = src[i];
    } else {
        // At or past the deletion point: shift left by delta.
        dst[i] = src[i + delta];
    }
}

// In-place gap shift: moves the tail of `buf[0..len]` by `delta`
// slots within the same buffer. When `delta > 0` the tail moves
// right (opening a gap); when `delta < 0` it moves left (closing a
// gap). The direction is chosen to avoid clobbering not-yet-moved
// elements: positive delta copies backward, negative delta copies
// forward.
//
// Each work item handles one element; the host launches enough
// items to cover the tail range `[at, len)`.
#[kernel]
fn gap_shift_inplace(buf: *mut int, len: int, at: int, delta: int) -> void {
    let i = get_global_id(0) as int;
    if delta > 0 {
        // Moving right: copy from high to low to avoid clobbering.
        // Index `i` in [0, tail_len) maps to source `len - 1 - i`.
        let tail_len = len - at;
        if i >= tail_len {
            return;
        }
        let src_index = len - 1 - i;
        let dst_index = src_index + delta;
        if dst_index < len {
            buf[dst_index] = buf[src_index];
        }
    } else if delta < 0 {
        // Moving left: copy from low to high.
        let d = -delta;
        let tail_len = len - at;
        if i >= tail_len {
            return;
        }
        let src_index = at + d + i;
        let dst_index = at + i;
        if src_index < len {
            buf[dst_index] = buf[src_index];
        }
    }
}
