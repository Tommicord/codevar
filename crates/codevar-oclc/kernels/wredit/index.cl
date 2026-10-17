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

// Buffer indexing hot paths: locate-first and count queries over u32
// code-point buffers.

// Finds the index of the first `needle` code point in `buf[..n]`.
//
// Work item `gid` owns the contiguous chunk
// `[gid * chunk, (gid + 1) * chunk)`, `chunk = ceil(n / 256)`, scans
// it from its lowest index, and writes the first in-chunk match to
// `out[gid]` or `0xFFFFFFFFu` when the chunk holds no match. Chunks
// are disjoint and ordered by `gid`, so the minimum of the partials
// below `0xFFFFFFFFu` is exactly the first match of the whole range,
// which is what the host returns.
#[kernel]
fn find_first_u32(buf: *const uint, n: uint, needle: uint, out: *mut uint) -> void {
    let gid = get_global_id(0) as uint;
    let total = get_global_size(0) as uint;
    let chunk = (n + total - 1u) / total;
    let base = gid * chunk;
    let mut result = 0xFFFFFFFFu;
    let mut j = 0u;
    while j < chunk {
        let i = base + j;
        if i < n {
            if buf[i] == needle {
                result = i;
                break;
            }
        }
        j = j + 1u;
    }
    out[gid] = result;
}

// Counts the occurrences of `needle` in `buf[..n]`.
//
// Work item `gid` owns the contiguous chunk
// `[gid * chunk, (gid + 1) * chunk)`, `chunk = ceil(n / 256)`, counts
// the matches in it, and writes the partial to `out[gid]`. The host
// sums the partials.
#[kernel]
fn count_u32(buf: *const uint, n: uint, needle: uint, out: *mut uint) -> void {
    let gid = get_global_id(0) as uint;
    let total = get_global_size(0) as uint;
    let chunk = (n + total - 1u) / total;
    let base = gid * chunk;
    let mut count = 0u;
    let mut j = 0u;
    while j < chunk {
        let i = base + j;
        if i < n {
            if buf[i] == needle {
                count = count + 1u;
            }
        }
        j = j + 1u;
    }
    out[gid] = count;
}
