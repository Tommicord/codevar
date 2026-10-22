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

// Parallel line-wrapping kernel for the Wedit engine.
//
// Given a code-point buffer and a maximum line width, this kernel
// identifies wrap opportunities (spaces after which a line may break)
// and computes per-chunk wrap counts. The host uses the partial counts
// to build a prefix-sum and then assigns each wrapped line a target
// column. Two kernels:
//
//   * count_wrap_points - counts spaces/break points per 64-element chunk
//   * assign_wrap_columns - computes the target wrap column for each line

// Counts wrap opportunities (spaces and explicit newlines) per
// 64-element chunk. Work item `gid` owns
// `data[gid*64 .. gid*64+64)` and writes the count to `counts[gid]`.
// Elements past `len` contribute nothing.
#[kernel]
fn count_wrap_points(data: *const int, len: int, counts: *mut int) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 64;
    let mut count = 0;
    let mut j = 0;
    while j < 64 {
        let i = base + j;
        if i < len {
            let cp = data[i];
            // A space (U+0020) or newline (U+000A) is a break point.
            if cp == 32 || cp == 10 {
                count = count + 1;
            }
        }
        j = j + 1;
    }
    counts[gid] = count;
}

// Given the prefix-summed wrap counts, computes the target wrap column
// for each line. Work item `i` owns line `i`: it reads the cumulative
// wrap count up to the line start and assigns a column based on
// `max_width` and the line's content length. This is a simplification
// — a full implementation would also scan the line for the best break
// point — but it gives the host a parallel starting point.
//
// `line_starts[i]` is the index in `data` where line `i` begins;
// `line_ends[i]` is the index one past the last element of line `i`.
#[kernel]
fn assign_wrap_columns(
    data: *const int,
    line_starts: *const int,
    line_ends: *const int,
    num_lines: int,
    max_width: int,
    wrap_counts: *const int,
    out_columns: *mut int,
) -> void {
    let i = get_global_id(0) as int;
    if i >= num_lines {
        return;
    }
    let start = line_starts[i];
    let end = line_ends[i];
    let line_len = end - start;
    if line_len <= max_width {
        // Line fits: no wrap needed, column is the line end.
        out_columns[i] = end;
        return;
    }
    // Line too long: find the last break point within max_width.
    let mut best = start + max_width;
    let mut j = start;
    while j < best && j < end {
        let cp = data[j];
        if cp == 32 || cp == 10 {
            best = j + 1; // break after this character
        }
        j = j + 1;
    }
    out_columns[i] = best;
}
