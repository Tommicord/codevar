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

//! UTF-16BE GPU kernels.
//!
//! Mirror of `utf16le.rs` with big-endian byte order. Each work item owns one
//! UTF-16 code unit pair (two input bytes) and produces its UTF-8 encoding at
//! a pre-computed output offset. Surrogate pairs are pre-segmented by the CPU.
//! Work items write disjoint output slots, so no synchronization is needed and
//! a plain 1-D dispatch suffices.
//!
//! Launch contract:
//!   Global size: `n` work items (one dimension), one per UTF-16 scalar.
//!   Bounds guard: `i < n`; the host must prove that `src[0..src_len]`,
//!   `src_offsets[0..n]`, `src_lengths[0..n]`, `out[0..out_capacity]`, and
//!   `out_offsets[0..n]` are valid.
//!   Synchronization: none.

/// Decode one UTF-16BE scalar (one or two code units) to UTF-8 at its pre-computed offset.
///
/// Arguments:
///   `src`         — raw UTF-16BE input bytes
///   `src_offsets` — per-scalar input start offset in bytes (`n` entries)
///   `src_lengths` — per-scalar input length in bytes (`n` entries, 2 or 4)
///   `out`         — output UTF-8 bytes
///   `out_offsets` — per-scalar output start offset (`n` entries)
///   `n`           — number of scalars
#[kernel]
fn utf16be_decode_to_utf8(
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
        // Read the first code unit (big-endian).
        let u0 = ((src[s] as u32) << 8) | (src[s + 1] as u32);
        if len == 2 {
            let cp = if u0 < 0xD800 || u0 > 0xDFFF {
                u0
            } else {
                // Lone surrogate: replacement.
                0xFFFD
            };
            if cp < 0x80 {
                out[out_base] = cp as u8;
            } else if cp < 0x800 {
                out[out_base] = ((cp >> 6) | 0xC0) as u8;
                out[out_base + 1] = ((cp & 0x3F) | 0x80) as u8;
            } else {
                out[out_base] = ((cp >> 12) | 0xE0) as u8;
                out[out_base + 1] = (((cp & 0xFC0) >> 6) | 0x80) as u8;
                out[out_base + 2] = ((cp & 0x3F) | 0x80) as u8;
            }
        } else {
            // Surrogate pair: 4 bytes, two code units.
            let u1 = ((src[s + 2] as u32) << 8) | (src[s + 3] as u32);
            let cp = if u0 >= 0xD800 && u0 <= 0xDBFF && u1 >= 0xDC00 && u1 <= 0xDFFF {
                (((u0 - 0xD800) << 10) + (u1 - 0xDC00) + 0x10000)
            } else {
                0xFFFD
            };
            out[out_base] = ((cp >> 18) | 0xF0) as u8;
            out[out_base + 1] = (((cp & 0x3F000) >> 12) | 0x80) as u8;
            out[out_base + 2] = (((cp & 0xFC0) >> 6) | 0x80) as u8;
            out[out_base + 3] = ((cp & 0x3F) | 0x80) as u8;
        }
    }
}

/// Encode one Unicode scalar value (UTF-32) to UTF-16BE at its pre-computed offset.
///
/// Arguments:
///   `src`         — input UTF-32 code points, length `n`
///   `out`         — output UTF-16BE bytes (2 bytes per BMP unit, 4 per astral)
///   `out_offsets` — per-code-point output start offset in bytes (`n` entries)
///   `n`           — number of code points
#[kernel]
fn utf16be_encode_from_utf32(src: *const u32, out: *mut u8, out_offsets: *mut int, n: int) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        let cp = src[i];
        let out_base = out_offsets[i];
        if cp < 0x10000 {
            let unit = cp as u16;
            out[out_base] = ((unit >> 8) & 0xFF) as u8;
            out[out_base + 1] = (unit & 0xFF) as u8;
        } else {
            let v = cp - 0x10000;
            let high = (0xD800 + ((v >> 10) & 0x3FF)) as u16;
            let low = (0xDC00 + (v & 0x3FF)) as u16;
            out[out_base] = ((high >> 8) & 0xFF) as u8;
            out[out_base + 1] = (high & 0xFF) as u8;
            out[out_base + 2] = ((low >> 8) & 0xFF) as u8;
            out[out_base + 3] = (low & 0xFF) as u8;
        }
    }
}
