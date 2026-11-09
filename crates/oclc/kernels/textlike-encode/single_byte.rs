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

//! # Pipeline status
//!
//! **UNWIRED**: These kernels are compiled and embedded by `codevar-oclc`'s
//! build script, but they are not yet connected to the
//! `codevar-textlike-encode` `Encoding` pipeline. The host-side codec
//! selection (`crate::legacy::ENCODINGS`) still routes through the CPU
//! implementations in `crate::single_byte`. Wiring these kernels in requires
//! a host dispatcher that (1) uploads the normative index table for each
//! encoding, (2) runs the CPU prefix-sum pass to compute `out_offsets`, and
//! (3) launches the appropriate kernel with the encoded arguments.

//! Single-byte encoding (legacy) GPU kernels.
//!
//! Every legacy single-byte encoding in `codevar-textlike-encode` shares one
//! structure: bytes `0x80..=0xFF` are mapped through a 128-entry `u16`
//! pointer table indexed as `table[byte - 0x80]`, and the encoder also uses a
//! run of consecutive pointers for the contiguous part of the table. These
//! kernels are table-driven: the host passes the normative index table as a
//! `*const u16` argument, so one pair of kernels serves every single-byte
//! encoding (IBM866, ISO-8859-*, KOI8-*, macintosh, windows-*, x-mac-cyrillic,
//! x-user-defined).
//!
//! The CPU performs a preprocessing pass: it scans the input, computes the
//! per-element output length, runs an exclusive prefix sum to produce
//! `out_offsets[i]` (the output slot where element `i`'s result begins), and
//! launches one work item per input element. Work items are independent, so no
//! synchronization is needed and a plain 1-D dispatch suffices.
//!
//! Decode (`single_byte_decode_to_utf8`):
//!   Each work item reads one input byte `src[i]`. Bytes below 0x80 are ASCII
//!   and copy straight through. Bytes at or above 0x80 are looked up in
//!   `table[byte - 0x80]`: a zero entry is malformed and expands to U+FFFD
//!   (EF BF BD), otherwise the code point is encoded as UTF-8 (one, two, or
//!   three bytes) starting at `out[out_offsets[i]]`.
//!
//! Encode (`single_byte_encode_from_utf32`):
//!   Each work item reads one UTF-32 code point `src[i]`. Code points below
//!   0x80 are ASCII and copy straight through. Otherwise the code point is
//!   searched for in `table`: a hit writes the corresponding byte
//!   (`128 + pointer`) at `out[out_offsets[i]]`; a miss writes the U+FFFD
//!   replacement as three UTF-8 bytes (lossy path) — the caller's CPU side is
//!   responsible for emitting numeric character references instead when the
//!   strict encoder is used.
//!
//! Launch contract:
//!   Global size: `n` work items (one dimension). Extra work items are
//!   discarded by the bounds guard `i < n`.
//!   Bounds guard: `i < n`; the host must prove that `src[0..n]`,
//!   `table[0..128]`, and `out[0..out_offsets[n-1] + max_element_output]`
//!   are valid for the full launch domain.
//!   Synchronization: none; work items write disjoint output slots.

/// Decode one single-byte input element to UTF-8 at its pre-computed offset.
///
/// Arguments:
///   `src`        — input bytes, length `n`
///   `table`      — 128-entry `u16` pointer table for bytes `0x80..=0xFF`
///   `out`        — output UTF-8 bytes
///   `out_offsets` — per-input-byte output start offset (prefix sum, `n` entries)
///   `n`          — number of input bytes
#[kernel]
fn single_byte_decode_to_utf8(
    src: *const u8,
    table: *const u16,
    out: *mut u8,
    out_offsets: *mut int,
    n: int,
) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        let b = src[i];
        let out_base = out_offsets[i];
        if b < 0x80 {
            out[out_base] = b;
        } else {
            let cp = table[(b - 0x80) as int];
            if cp == 0 {
                // U+FFFD replacement: EF BF BD
                out[out_base] = 0xEFu8;
                out[out_base + 1] = 0xBFu8;
                out[out_base + 2] = 0xBDu8;
            } else if cp < 0x800 {
                out[out_base] = ((cp >> 6) | 0xC0) as u8;
                out[out_base + 1] = ((cp & 0x3F) | 0x80) as u8;
            } else {
                out[out_base] = ((cp >> 12) | 0xE0) as u8;
                out[out_base + 1] = (((cp & 0xFC0) >> 6) | 0x80) as u8;
                out[out_base + 2] = ((cp & 0x3F) | 0x80) as u8;
            }
        }
    }
}

/// Encode one UTF-32 code point to a single-byte output element at its pre-computed offset.
///
/// Arguments:
///   `src`        — input UTF-32 code points, length `n`
///   `table`      — 128-entry `u16` pointer table for bytes `0x80..=0xFF`
///   `out`        — output single-byte bytes
///   `out_offsets` — per-input-code-point output start offset (prefix sum, `n` entries)
///   `n`          — number of input code points
#[kernel]
fn single_byte_encode_from_utf32(
    src: *const u32,
    table: *const u16,
    out: *mut u8,
    out_offsets: *mut int,
    n: int,
) -> void {
    let i = get_global_id(0) as int;
    if i < n {
        let cp = src[i];
        let out_base = out_offsets[i];
        if cp < 0x80 {
            out[out_base] = cp as u8;
        } else if cp > 0xFFFF {
            // Astral: cannot map to a single byte. Lossy replacement with '?'.
            out[out_base] = 0x3Fu8;
        } else {
            let unit = cp as u16;
            let mut found = 0;
            let mut byte = 0u8;
            // The table is small (128 entries); a linear scan finds the
            // pointer for this code point. The host CPU pre-computes a
            // reverse map for the hot path; this kernel handles the general
            // case.
            let mut p = 0;
            while p < 128 {
                if table[p] == unit {
                    found = 1;
                    byte = (128 + p) as u8;
                    break;
                }
                p += 1;
            }
            if found != 0 {
                out[out_base] = byte;
            } else {
                // Unmappable: lossy replacement with '?'.
                out[out_base] = 0x3Fu8;
            }
        }
    }
}
