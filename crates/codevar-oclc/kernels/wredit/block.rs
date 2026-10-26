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

// Block storage hot paths: per-block statistics for the virtual
// address table of the Wredit engine.
//
// Both kernels split their input into fixed 64-byte chunks: work item
// `gid` owns `data[gid * 64 .. gid * 64 + 64)`, writes one partial
// result to `out[gid]`, and never touches any other slot, so no
// synchronization is needed and the host performs the final reduction.
// Extra work items past the end of the input write `0`, which the host
// ignores by only summing the first `ceil(len / 64)` partials.
//
// Launch with a global size of at least `ceil(len / 64)` (one
// dimension). The host guarantees `len <= 4096` for one block, so every
// accumulator stays far below the 32-bit limits.

// Counts the code points stored in a UTF-8 byte range.
//
// In UTF-8 a code point starts at every byte that is *not* a
// continuation byte (`10xxxxxx`, i.e. `128 <= b < 192`), which makes
// the count boundary-safe: a chunk may start in the middle of a
// multi-byte sequence and the arithmetic still totals exactly one per
// code point for the whole range. `counts[gid]` receives the number of
// non-continuation bytes in the chunk; the host sums the partials.
//
// For UTF-16 the host skips this kernel entirely (code units are two
// bytes, so the count is `len / 2`).
#[kernel]
fn char_count(data: *const uchar, len: int, counts: *mut int) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 64;
    let mut count = 0;
    let mut j = 0;
    while j < 64 {
        let i = base + j;
        if i < len {
            let byte = data[i];
            if byte < 128 {
                count = count + 1;
            } else {
                if byte >= 192 {
                    count = count + 1;
                }
            }
        }
        j = j + 1;
    }
    counts[gid] = count;
}

// Position-weighted checksum of a byte range.
//
// Computes `sum(data[i] * ((i % 13) + 1))` over the chunk, where the
// weight cycles with period 13 so that a one-byte edit changes both a
// value and the weight of every later byte in its period. The host
// sums the partials and reduces modulo 2^30, matching
// `codevar_wredit::simd::linear_checksum` exactly — the CPU and GPU
// paths must agree on every input or block-integrity checks would
// report false corruption.
//
// `i % 13` is spelled as `i - (i / 13) * 13` because integer division
// is the only primitive the pipeline needs to lower; the result is
// identical for the non-negative indices used here.
#[kernel]
fn block_checksum(data: *const uchar, len: int, partials: *mut int) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 64;
    let mut sum = 0;
    let mut j = 0;
    while j < 64 {
        let i = base + j;
        if i < len {
            let weight = i - (i / 13) * 13 + 1;
            sum = sum + (data[i] as int) * weight;
        }
        j = j + 1;
    }
    partials[gid] = sum;
}
