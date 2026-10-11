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

// Text statistics hot paths: line-oriented queries over code points.
//
// Input is the engine's internal `u32` code-point array, so one element
// equals one code point and no encoding rules apply — the same array
// the Bandler buffer keeps. Work item `gid` owns the 64-element chunk
// `[gid * 64, gid * 64 + 64)` and writes one partial count, which the
// host sums. Elements past `len` contribute nothing.

// Counts line feeds (U+000A) in a code-point range.
//
// Line counting feeds `Query`/`Selection` line-column resolution; a
// `counts[gid]` partial is the number of newlines inside the chunk and
// the host performs the prefix-sum reduction.
#[kernel]
fn count_newlines(data: *const int, len: int, counts: *mut int) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 64;
    let mut count = 0;
    let mut j = 0;
    while j < 64 {
        let i = base + j;
        if i < len {
            if data[i] == 10 {
                count = count + 1;
            }
        }
        j = j + 1;
    }
    counts[gid] = count;
}
