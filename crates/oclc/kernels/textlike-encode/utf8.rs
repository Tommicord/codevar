//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! UTF-8 GPU kernels.
//!
//! UTF-8 is the canonical internal encoding of `codevar-textlike-encode`, so
//! the UTF-8 kernels are validation and identity pipelines: they verify that
//! the input is well-formed UTF-8 and either pass it through or replace
//! maximal invalid subparts with U+FFFD. The CPU performs a preprocessing
//! pass that validates sequence boundaries and computes per-scalar output
//! offsets; each GPU work item then processes one scalar and writes its
//! (possibly replaced) UTF-8 encoding at the pre-computed offset.
//!
//! Work items are independent and write disjoint output slots, so no
//! synchronization is needed and a plain 1-D dispatch suffices.
//!
//! Launch contract:
//!   Global size: `n` work items (one dimension), one per source scalar.
//!   Bounds guard: `i < n`; the host must prove that `src[0..src_len]`,
//!   `out[0..out_capacity]`, and `out_offsets[0..n]` are valid.
//!   Synchronization: none.

/// Validate and (re)emit one UTF-8 scalar at its pre-computed output offset.
///
/// The input is a sequence of already-segmented UTF-8 scalars: the CPU
/// pre-decodes the input into scalar boundaries and passes `src` as the raw
/// bytes together with `out_offsets[i]`, the output start for scalar `i`.
/// Each work item re-encodes scalar `i` from its source bytes; invalid
/// scalars are replaced with U+FFFD (EF BF BD).
///
/// Arguments:
///   `src`         — raw UTF-8 input bytes
///   `src_offsets` — per-scalar input start offset (`n` entries)
///   `src_lengths` — per-scalar input byte length (`n` entries, 1..=4)
///   `out`         — output UTF-8 bytes
///   `out_offsets` — per-scalar output start offset (`n` entries)
///   `n`           — number of scalars
#[kernel]
fn utf8_validate_and_emit(
    src: *const u8,
    src_offsets: *const int,
    src_lengths: *const int,
    out: *mut u8,
    out_offsets: *mut int,
    n: int,
) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        let s = src_offsets[i];
        let len = src_lengths[i];
        let out_base = out_offsets[i];
        let b0 = src[s];
        // Fast path: ASCII.
        if b0 < 0x80 && len == 1 {
            out[out_base] = b0;
            return;
        }
        // Validate the leading byte and continuation bytes. Any error
        // substitutes U+FFFD.
        let mut valid = 1;
        if len == 2 {
            if b0 < 0xC2 || b0 > 0xDF {
                valid = 0;
            } else if src[s + 1] < 0x80 || src[s + 1] > 0xBF {
                valid = 0;
            }
        } else if len == 3 {
            if b0 < 0xE0 || b0 > 0xEF {
                valid = 0;
            } else {
                let b1 = src[s + 1];
                let b2 = src[s + 2];
                if b1 < 0x80 || b1 > 0xBF || b2 < 0x80 || b2 > 0xBF {
                    valid = 0;
                }
                // Overlong and surrogate checks.
                if valid != 0 {
                    if b0 == 0xE0 && b1 < 0xA0 {
                        valid = 0;
                    }
                    if b0 == 0xED && b1 >= 0xA0 && b1 <= 0xBF {
                        valid = 0;
                    }
                }
            }
        } else if len == 4 {
            if b0 < 0xF0 || b0 > 0xF4 {
                valid = 0;
            } else {
                let b1 = src[s + 1];
                let b2 = src[s + 2];
                let b3 = src[s + 3];
                if b1 < 0x80 || b1 > 0xBF || b2 < 0x80 || b2 > 0xBF || b3 < 0x80 || b3 > 0xBF {
                    valid = 0;
                }
                if valid != 0 {
                    if b0 == 0xF0 && b1 < 0x90 {
                        valid = 0;
                    }
                    if b0 == 0xF4 && b1 > 0x8F {
                        valid = 0;
                    }
                }
            }
        } else {
            valid = 0;
        }
        if valid != 0 {
            let mut k = 0;
            while k < len {
                out[out_base + k] = src[s + k];
                k += 1;
            }
        } else {
            out[out_base] = 0xEFu8;
            out[out_base + 1] = 0xBFu8;
            out[out_base + 2] = 0xBDu8;
        }
    }
}

/// Encode one Unicode scalar value (UTF-32) to UTF-8 at its pre-computed offset.
///
/// Arguments:
///   `src`         — input UTF-32 code points, length `n`
///   `out`         — output UTF-8 bytes
///   `out_offsets` — per-code-point output start offset (`n` entries)
///   `n`           — number of code points
#[kernel]
fn utf8_encode_from_utf32(src: *const u32, out: *mut u8, out_offsets: *mut int, n: int) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        let cp = src[i];
        let out_base = out_offsets[i];
        if cp < 0x80 {
            out[out_base] = cp as u8;
        } else if cp < 0x800 {
            out[out_base] = ((cp >> 6) | 0xC0) as u8;
            out[out_base + 1] = ((cp & 0x3F) | 0x80) as u8;
        } else if cp < 0x10000 {
            out[out_base] = ((cp >> 12) | 0xE0) as u8;
            out[out_base + 1] = (((cp & 0xFC0) >> 6) | 0x80) as u8;
            out[out_base + 2] = ((cp & 0x3F) | 0x80) as u8;
        } else {
            out[out_base] = ((cp >> 18) | 0xF0) as u8;
            out[out_base + 1] = (((cp & 0x3F000) >> 12) | 0x80) as u8;
            out[out_base + 2] = (((cp & 0xFC0) >> 6) | 0x80) as u8;
            out[out_base + 3] = ((cp & 0x3F) | 0x80) as u8;
        }
    }
}
