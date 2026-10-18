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

// Parallel UTF-8 validation kernel for the Wedit engine.
//
// The engine stores code points as u32 elements, but when importing
// text from UTF-8 sources the raw bytes must be validated. This kernel
// checks a byte buffer for well-formed UTF-8: each work item owns a
// 64-byte chunk and writes 1 to `valid[gid]` if the chunk is valid,
// 0 otherwise. The host ANDs all partials to get the final answer.
//
// Note: this validates bytes, not the engine's internal u32 code point
// array. It is used on import paths before transcoding into packed
// elements.

// Validates a 64-byte chunk for well-formed UTF-8.
// Work item `gid` owns `data[gid*64 .. gid*64+64)` and writes
// `valid[gid] = 1` if all complete sequences in the chunk are
// well-formed, else 0. A trailing incomplete sequence at the end of
// the buffer is treated as valid (the caller handles continuation).
#[kernel]
fn utf8_validate_chunk(data: *const uchar, len: int, valid: *mut int) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 64;
    let mut result = 1; // assume valid until proven otherwise
    let mut j = 0;
    while j < 64 {
        let i = base + j;
        if i >= len {
            break;
        }
        let b = data[i] as int;
        if b < 0x80 {
            // ASCII: always valid, advance 1.
            j = j + 1;
        } else if b >= 0xC2 && b <= 0xDF {
            // 2-byte sequence: 110xxxxx 10xxxxxx
            if i + 1 >= len {
                result = 1; // incomplete at end, acceptable
                break;
            }
            let b2 = data[i + 1] as int;
            if b2 < 0x80 || b2 > 0xBF {
                result = 0;
                break;
            }
            j = j + 2;
        } else if b >= 0xE0 && b <= 0xEF {
            // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
            if i + 2 >= len {
                result = 1;
                break;
            }
            let b2 = data[i + 1] as int;
            let b3 = data[i + 2] as int;
            if (b2 < 0x80 || b2 > 0xBF) || (b3 < 0x80 || b3 > 0xBF) {
                result = 0;
                break;
            }
            // Overlong check for E0: second byte must be >= 0xA0
            if b == 0xE0 && b2 < 0xA0 {
                result = 0;
                break;
            }
            // Overlong check for ED: second byte must be <= 0x9F
            if b == 0xED && b2 > 0x9F {
                result = 0;
                break;
            }
            j = j + 3;
        } else if b >= 0xF0 && b <= 0xF4 {
            // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            if i + 3 >= len {
                result = 1;
                break;
            }
            let b2 = data[i + 1] as int;
            let b3 = data[i + 2] as int;
            let b4 = data[i + 3] as int;
            if (b2 < 0x80 || b2 > 0xBF) || (b3 < 0x80 || b3 > 0xBF) || (b4 < 0x80 || b4 > 0xBF) {
                result = 0;
                break;
            }
            // Overlong check for F0: second byte must be >= 0x90
            if b == 0xF0 && b2 < 0x90 {
                result = 0;
                break;
            }
            // Range check for F4: second byte must be <= 0x8F
            if b == 0xF4 && b2 > 0x8F {
                result = 0;
                break;
            }
            j = j + 4;
        } else {
            // Invalid lead byte (0x80-0xBF stray continuation, 0xF5-0xFF)
            result = 0;
            break;
        }
    }
    valid[gid] = result;
}

// Counts the number of valid UTF-8 code points in a byte buffer.
// Work item `gid` owns a 64-byte chunk and writes the count of
// non-continuation bytes (code point starts) to `counts[gid]`.
// This is the byte-level equivalent of `count_utf8_units`.
#[kernel]
fn utf8_count_code_points(data: *const uchar, len: int, counts: *mut int) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 64;
    let mut count = 0;
    let mut j = 0;
    while j < 64 {
        let i = base + j;
        if i >= len {
            break;
        }
        let b = data[i] as int;
        // A code point starts at every byte that is not a continuation
        // byte (0x80-0xBF).
        if b < 0x80 || b >= 0xC0 {
            count = count + 1;
        }
        j = j + 1;
    }
    counts[gid] = count;
}
