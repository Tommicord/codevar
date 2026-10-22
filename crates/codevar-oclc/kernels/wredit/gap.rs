// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// Multi-gap buffer kernel: compaction of the Bandler gap representation.
//
// The Bandler buffer stores `len` live code points in a `u32` array of
// `len + gap_len` slots, with the free run `[gap_start, gap_start +
// gap_len)` removed. Compaction closes that gap: every live element
// before it stays put, every element after it slides left by `gap_len`.
//
// The kernel is a pure element-wise copy with no cross-item
// dependencies, so a plain one-dimensional dispatch over the `len`
// output elements is enough. Work items with `i >= len` write nothing.

// Compacts `src` into `dst`, dropping the gap `[gap_start, gap_start +
// gap_len)` and writing exactly `len` elements.
//
// The host allocates `dst` with room for `len` elements and keeps
// `src` alive until the kernel event has been waited on.
#[kernel]
fn gap_compact(src: *const int, len: int, gap_start: int, gap_len: int, dst: *mut int) -> void {
    let i = get_global_id(0) as int;
    if i < len {
        if i < gap_start {
            dst[i] = src[i];
        } else {
            dst[i] = src[i + gap_len];
        }
    }
}
