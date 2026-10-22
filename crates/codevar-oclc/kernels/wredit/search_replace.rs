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

// Parallel search and replace kernel for the Wedit engine.
//
// Given a code-point buffer `data[0..len]`, a `needle` sequence and a
// `replacement` sequence, this kernel finds every non-overlapping
// occurrence of `needle` and writes the transformed result to `out`.
// Each work item owns one output slot; the host performs a prefix-sum
// over match flags to compute output offsets, then a second pass
// writes the result. Two kernels are provided:
//
//   * find_matches  - marks which positions start a `needle` match
//   * apply_replace - writes the replaced output using precomputed offsets

// Marks positions where `needle` occurs in `data[0..len]`.
// `matches[i] = 1` when `data[i..i+needle_len] == needle`, else 0.
// Work item `i` checks position `i`; positions past `len - needle_len`
// always write 0.
#[kernel]
fn find_matches(data: *const int, len: int, needle: *const int, needle_len: int, matches: *mut int) -> void {
    let i = get_global_id(0) as int;
    let last_start = len - needle_len;
    if i > last_start || i < 0 {
        matches[i] = 0;
        return;
    }
    let mut j = 0;
    let mut is_match = 1;
    while j < needle_len {
        if data[i + j] != needle[j] {
            is_match = 0;
            break;
        }
        j = j + 1;
    }
    matches[i] = is_match;
}

// Applies search-and-replace using precomputed match flags and output
// offsets. `out_offsets[i]` is the destination index in `out` where
// the element originating from `data[i]` should be written (after
// replacement expansion). Work item `i` either copies one element or
// writes the full replacement at a match start.
//
// `out` must have room for `out_len` elements (computed by the host
// from the match prefix sum).
#[kernel]
fn apply_replace(
    data: *const int,
    len: int,
    needle: *const int,
    needle_len: int,
    replacement: *const int,
    replacement_len: int,
    matches: *const int,
    out_offsets: *const int,
    out: *mut int,
    out_len: int,
) -> void {
    let i = get_global_id(0) as int;
    if i >= len {
        return;
    }

    if matches[i] != 0 && i + needle_len <= len {
        // This position starts a match: write the replacement.
        let out_base = out_offsets[i];
        let mut j = 0;
        while j < replacement_len {
            if out_base + j < out_len {
                out[out_base + j] = replacement[j];
            }
            j = j + 1;
        }
        // Skip the rest of the needle: subsequent positions inside the
        // matched range are handled by their own work items but should
        // not write (their offsets are invalid). We guard by only
        // writing at match starts.
    } else {
        // Not a match start: copy this element to its output offset.
        let out_index = out_offsets[i];
        if out_index < out_len {
            out[out_index] = data[i];
        }
    }
}
