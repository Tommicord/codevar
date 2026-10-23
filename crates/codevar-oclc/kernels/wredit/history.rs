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

// Undo/redo history hot paths: integrity checksums over encoded
// transactions.
//
// The Wredit history serializes edit transactions to byte buffers when
// a cold transaction is compressed; each buffer carries a checksum so a
// truncated or corrupted undo stack is detected before it is replayed.
// This kernel is byte-identical in behaviour to
// `block_checksum` (same chunk size, same position weights) but
// lives in its own module so the history layer can pull it in without
// dragging the storage kernels along.

// Position-weighted checksum of a serialized transaction.
//
// Computes `sum(data[i] * ((i % 13) + 1))` over the chunk; the host
// sums the partials and reduces modulo 2^30, matching
// `codevar_wredit::simd::linear_checksum`. The weight cycle makes the
// result sensitive to insertions as well as value changes.
//
// `i % 13` is spelled as `i - (i / 13) * 13` so only integer division
// has to lower; for the non-negative indices used here the result is
// identical to a remainder.
#[kernel]
fn history_checksum(data: *const uchar, len: int, partials: *mut int) -> void {
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
