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

// Element-wise vector addition: `out[i] = a[i] + b[i]` for `i in 0..n`.
// Work item `gid` owns element `gid`: it reads `a[gid]` and `b[gid]`,
// writes `out[gid]`, and never touches another slot, so no
// synchronization is needed and the host performs no reduction. Launch
// with a global size of `n` (one dimension); there are no work items
// past the end because the host launches exactly `n` of them.
//
// This is the memory-bound reference workload: every work-item performs
// one load, one add and one store, so the native, interpreted-SPIR-V and
// GPU paths differ only in dispatch and execution overhead, not in work.
//
// Arguments:
//   `a`   — input vector of `n` integers
//   `b`   — input vector of `n` integers
//   `out` — output vector of `n` integers
//   `n`   — element count
#[kernel]
fn vector_add(a: *const int, b: *const int, out: *mut int, n: int) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        out[i] = a[i] + b[i];
    }
}
