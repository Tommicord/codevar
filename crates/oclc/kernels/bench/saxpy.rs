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

// Single-precision axpy: `out[i] = scale * x[i] + y[i]` for `i in 0..n`.
// Work item `gid` owns element `gid`: it reads `x[gid]` and `y[gid]`,
// writes `out[gid]`, and never touches another slot, so no
// synchronization is needed and the host performs no reduction. Launch
// with a global size of `n` (one dimension).
//
// Compared with `vector_add` this kernel multiplies as well as adds, so
// its arithmetic intensity is slightly higher while its memory footprint
// is the same: three vectors and two scalars (`scale` and `n`).
//
// Arguments:
//   `scale` — multiplier applied to every element of `x`
//   `x`     — input vector of `n` integers
//   `y`     — input vector of `n` integers
//   `out`   — output vector of `n` integers
//   `n`     — element count
#[kernel]
fn saxpy(scale: int, x: *const int, y: *const int, out: *mut int, n: int) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        out[i] = scale * x[i] + y[i];
    }
}
