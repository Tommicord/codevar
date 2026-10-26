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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! # Codevar gzip
//!
//! A faithful Rust port of the FreeType 2.6 gzip subsystem
//! (`src/gzip/`): the public `FT_Stream_OpenGzip` / `FT_Gzip_Uncompress`
//! API plus the embedded zlib 1.1.4 inflate pipeline that FreeType
//! bundles when it is built without a system zlib.
//!
//! | FreeType / zlib file | Section below |
//! |----------------------|---------------|
//! | `adler32.c`          | [`adler32`] |
//! | `inftrees.c/h`, `inffixed.h` | [`huft_build`], [`FIXED_TL`], [`FIXED_TD`] |
//! | `infcodes.c/h`       | [`BlocksState::inflate_codes`] |
//! | `infblock.c/h`       | [`BlocksState::inflate_blocks`] |
//! | `infutil.c/h`        | [`inflate_flush`], [`INFLATE_MASK`] |
//! | `inflate.c`          | [`Inflater::inflate_run`] |
//! | `ftgzip.c`           | [`GzipSource`], [`uncompress`], [`is_gzip`] |
//!
//! Memory is bounded: every allocation (32 KiB inflate window, 1440-entry
//! Huffman arena, ≤316 code-length slots, 4 KiB I/O buffers, 32 KiB
//! history ring, optional <40 KiB small-file image) is sized from
//! protocol constants, never from attacker-controlled length fields.
//!
//! ## Public API
//!
//! * [`GzipSource`] - the `FT_Stream_OpenGzip` layer over a
//!   [`StreamSource`](codevar_truetype_core::StreamSource).
//! * [`uncompress`] - `FT_Gzip_Uncompress`: one-shot raw DEFLATE.
//! * [`is_gzip`] - gzip header sniffer.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use codevar_truetype_core::StreamSource;
use spin::Mutex;

/// Re-export of the FreeType-style error code type used by this crate
/// (`TtError(i32)` with the `TT_*` constants), so callers can name the
/// error type without a separate dependency.
pub use codevar_truetype_core::TtError;

/// Re-export of `Result<T, TtError>`, the fallible alias used by every
/// public entry point of this crate.
pub use codevar_truetype_core::TtResult;

/// `Z_NO_FLUSH`: inflate as much as possible (ftgzip's `Z_NO_FLUSH`).
const Z_NO_FLUSH: i32 = 0;
/// `Z_FINISH`: no more input is coming (FT_Gzip_Uncompress).
const Z_FINISH: i32 = 4;
/// `Z_OK`: progress was made.
const Z_OK: i32 = 0;
/// `Z_STREAM_END`: the deflate stream finished.
const Z_STREAM_END: i32 = 1;
/// `Z_NEED_DICT`: a preset dictionary would be required.
const Z_NEED_DICT: i32 = 2;
/// `Z_STREAM_ERROR`: inconsistent internal state.
const Z_STREAM_ERROR: i32 = -2;
/// `Z_DATA_ERROR`: the input is corrupt.
const Z_DATA_ERROR: i32 = -3;
/// `Z_MEM_ERROR`: an allocation failed.
const Z_MEM_ERROR: i32 = -4;
/// `Z_BUF_ERROR`: no progress was possible.
const Z_BUF_ERROR: i32 = -5;

/// `MAX_WBITS`: log2 of the 32 KiB LZ77 window.
const MAX_WBITS: u32 = 15;
/// `Z_DEFLATED`: the only compression method gzip permits.
const Z_DEFLATED: u8 = 8;
/// `PRESET_DICT`: preset-dictionary flag of the zlib header.
const PRESET_DICT: u8 = 0x20;

/// `FT_GZIP_BUFFER_SIZE`: FreeType's input/output buffer size.
const FT_GZIP_BUFFER_SIZE: usize = 4096;

/// `FT_GZIP_HEAD_CRC` (bit 1): header CRC16 present.
const FT_GZIP_HEAD_CRC: u8 = 0x02;
/// `FT_GZIP_EXTRA_FIELD` (bit 2): FEXTRA field present.
const FT_GZIP_EXTRA_FIELD: u8 = 0x04;
/// `FT_GZIP_ORIG_NAME` (bit 3): original file name present.
const FT_GZIP_ORIG_NAME: u8 = 0x08;
/// `FT_GZIP_COMMENT` (bit 4): file comment present.
const FT_GZIP_COMMENT: u8 = 0x10;
/// `FT_GZIP_RESERVED` (bits 5..7): must be zero.
const FT_GZIP_RESERVED: u8 = 0xE0;

/// Maximum number of leading bytes scanned for the gzip magic, per the
/// specification of this port (see [`check_header`]).
const GZIP_MAGIC_SCAN: usize = 1024;

/// `SMALL_FILE_LIMIT`: files below this decompressed size are kept fully
/// in memory, mirroring FreeType's "load small fonts whole" trick
/// (`zip_size < 40 * 1024` in `FT_Stream_OpenGzip`).
const SMALL_FILE_LIMIT: u64 = 40 * 1024;

/// Size of the retained decompressed history ring used to serve backward
/// random access without resetting the inflater (at least the 32 KiB
/// LZ77 window, as required by this port's design notes).
const HISTORY_SIZE: u64 = 32 * 1024;

/// `BASE` from adler32.c: largest prime smaller than 65536.
const ADLER_BASE: u32 = 65521;
/// `NMAX` from adler32.c: largest n such that
/// `255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1`.
const ADLER_NMAX: usize = 5552;

/// Initial Adler-32 check value.
///
/// This is the value C's `adler32(0L, Z_NULL, 0)` returns; zlib uses it to
/// (re)start the running check of a freshly reset inflate stream.
#[inline]
const fn adler32_init() -> u32 {
    1
}

/// Port of zlib's `adler32`: computes the Adler-32 checksum of `buf`
/// continuing from `adler`.
///
/// # Performance
///
/// O(`buf.len()`); processes `ADLER_NMAX` bytes per modulo round exactly
/// like the C reference, so the reduction cost is amortized. Scalar-only;
/// the inner loop is a simple dual-sum that auto-vectorizes poorly by
/// nature (sequential dependency between `s2` and `s1`).
#[inline]
pub fn adler32(adler: u32, buf: &[u8]) -> u32 {
    let mut s1 = adler & 0xffff;
    let mut s2 = (adler >> 16) & 0xffff;
    let mut idx = 0usize;
    let mut len = buf.len();

    while len > 0 {
        let mut k = if len < ADLER_NMAX { len } else { ADLER_NMAX };
        len -= k;

        while k >= 16 {
            let end = idx + 16;
            while idx < end {
                s1 += u32::from(buf[idx]);
                s2 += s1;
                idx += 1;
            }
            k -= 16;
        }
        while k > 0 {
            s1 += u32::from(buf[idx]);
            s2 += s1;
            idx += 1;
            k -= 1;
        }
        s1 %= ADLER_BASE;
        s2 %= ADLER_BASE;
    }
    (s2 << 16) | s1
}

/// One entry of a zlib inflate Huffman table (`struct inflate_huft_s`).
///
/// `exop`/`bits` occupy what C packs into the `word.what` union, `base` is
/// the literal/length/distance value, extra-bit count, or (when `exop`
/// marks a subtable) the relative offset to the next table level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Huft {
    /// Number of extra bits, or the operation marker (see `inftrees.c`).
    exop: u8,
    /// Number of bits in this code or subcode.
    bits: u8,
    /// Literal, length base, distance base, or subtable offset.
    base: u32,
}

/// `MANY` from inftrees.h: capacity of the single Huffman arena allocated
/// per inflate stream (1440 entries, ~11 KiB).
const MANY: usize = 1440;

/// `BMAX` from inftrees.c: maximum bit length of any deflate code.
const BMAX: usize = 15;

/// `fixed_bl` from inffixed.h: first-level lookup bits of the fixed
/// literal/length tree.
const FIXED_BL: u32 = 9;
/// `fixed_bd` from inffixed.h: first-level lookup bits of the fixed
/// distance tree.
const FIXED_BD: u32 = 5;

/// `fixed literal/length tree, `fixed_tl` from inffixed.h`
const FIXED_TL: [Huft; 512] = [
    Huft {
        exop: 96,
        bits: 7,
        base: 256,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 80,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 16,
    },
    Huft {
        exop: 84,
        bits: 8,
        base: 115,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 31,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 112,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 48,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 192,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 10,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 96,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 32,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 160,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 0,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 128,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 64,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 224,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 6,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 88,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 24,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 144,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 59,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 120,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 56,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 208,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 17,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 104,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 40,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 176,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 8,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 136,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 72,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 240,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 4,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 84,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 20,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 227,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 43,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 116,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 52,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 200,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 13,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 100,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 36,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 168,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 4,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 132,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 68,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 232,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 8,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 92,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 28,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 152,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 83,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 124,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 60,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 216,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 23,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 108,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 44,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 184,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 12,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 140,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 76,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 248,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 3,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 82,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 18,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 163,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 35,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 114,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 50,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 196,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 11,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 98,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 34,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 164,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 2,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 130,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 66,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 228,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 7,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 90,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 26,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 148,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 67,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 122,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 58,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 212,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 19,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 106,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 42,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 180,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 10,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 138,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 74,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 244,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 5,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 86,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 22,
    },
    Huft {
        exop: 192,
        bits: 8,
        base: 0,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 51,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 118,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 54,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 204,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 15,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 102,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 38,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 172,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 6,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 134,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 70,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 236,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 9,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 94,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 30,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 156,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 99,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 126,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 62,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 220,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 27,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 110,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 46,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 188,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 14,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 142,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 78,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 252,
    },
    Huft {
        exop: 96,
        bits: 7,
        base: 256,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 81,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 17,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 131,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 31,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 113,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 49,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 194,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 10,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 97,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 33,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 162,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 1,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 129,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 65,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 226,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 6,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 89,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 25,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 146,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 59,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 121,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 57,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 210,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 17,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 105,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 41,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 178,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 9,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 137,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 73,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 242,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 4,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 85,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 21,
    },
    Huft {
        exop: 80,
        bits: 8,
        base: 258,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 43,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 117,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 53,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 202,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 13,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 101,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 37,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 170,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 5,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 133,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 69,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 234,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 8,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 93,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 29,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 154,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 83,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 125,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 61,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 218,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 23,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 109,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 45,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 186,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 13,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 141,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 77,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 250,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 3,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 83,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 19,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 195,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 35,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 115,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 51,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 198,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 11,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 99,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 35,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 166,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 3,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 131,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 67,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 230,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 7,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 91,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 27,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 150,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 67,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 123,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 59,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 214,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 19,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 107,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 43,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 182,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 11,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 139,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 75,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 246,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 5,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 87,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 23,
    },
    Huft {
        exop: 192,
        bits: 8,
        base: 0,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 51,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 119,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 55,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 206,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 15,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 103,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 39,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 174,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 7,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 135,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 71,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 238,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 9,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 95,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 31,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 158,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 99,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 127,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 63,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 222,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 27,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 111,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 47,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 190,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 15,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 143,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 79,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 254,
    },
    Huft {
        exop: 96,
        bits: 7,
        base: 256,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 80,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 16,
    },
    Huft {
        exop: 84,
        bits: 8,
        base: 115,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 31,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 112,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 48,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 193,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 10,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 96,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 32,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 161,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 0,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 128,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 64,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 225,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 6,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 88,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 24,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 145,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 59,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 120,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 56,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 209,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 17,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 104,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 40,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 177,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 8,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 136,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 72,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 241,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 4,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 84,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 20,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 227,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 43,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 116,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 52,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 201,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 13,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 100,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 36,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 169,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 4,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 132,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 68,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 233,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 8,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 92,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 28,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 153,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 83,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 124,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 60,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 217,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 23,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 108,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 44,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 185,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 12,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 140,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 76,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 249,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 3,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 82,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 18,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 163,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 35,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 114,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 50,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 197,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 11,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 98,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 34,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 165,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 2,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 130,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 66,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 229,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 7,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 90,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 26,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 149,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 67,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 122,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 58,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 213,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 19,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 106,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 42,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 181,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 10,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 138,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 74,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 245,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 5,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 86,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 22,
    },
    Huft {
        exop: 192,
        bits: 8,
        base: 0,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 51,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 118,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 54,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 205,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 15,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 102,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 38,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 173,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 6,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 134,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 70,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 237,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 9,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 94,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 30,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 157,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 99,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 126,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 62,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 221,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 27,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 110,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 46,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 189,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 14,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 142,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 78,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 253,
    },
    Huft {
        exop: 96,
        bits: 7,
        base: 256,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 81,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 17,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 131,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 31,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 113,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 49,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 195,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 10,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 97,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 33,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 163,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 1,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 129,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 65,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 227,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 6,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 89,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 25,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 147,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 59,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 121,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 57,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 211,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 17,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 105,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 41,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 179,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 9,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 137,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 73,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 243,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 4,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 85,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 21,
    },
    Huft {
        exop: 80,
        bits: 8,
        base: 258,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 43,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 117,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 53,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 203,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 13,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 101,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 37,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 171,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 5,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 133,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 69,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 235,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 8,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 93,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 29,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 155,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 83,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 125,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 61,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 219,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 23,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 109,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 45,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 187,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 13,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 141,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 77,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 251,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 3,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 83,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 19,
    },
    Huft {
        exop: 85,
        bits: 8,
        base: 195,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 35,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 115,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 51,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 199,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 11,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 99,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 35,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 167,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 3,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 131,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 67,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 231,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 7,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 91,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 27,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 151,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 67,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 123,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 59,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 215,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 19,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 107,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 43,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 183,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 11,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 139,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 75,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 247,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 5,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 87,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 23,
    },
    Huft {
        exop: 192,
        bits: 8,
        base: 0,
    },
    Huft {
        exop: 83,
        bits: 7,
        base: 51,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 119,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 55,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 207,
    },
    Huft {
        exop: 81,
        bits: 7,
        base: 15,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 103,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 39,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 175,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 7,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 135,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 71,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 239,
    },
    Huft {
        exop: 80,
        bits: 7,
        base: 9,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 95,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 31,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 159,
    },
    Huft {
        exop: 84,
        bits: 7,
        base: 99,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 127,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 63,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 223,
    },
    Huft {
        exop: 82,
        bits: 7,
        base: 27,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 111,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 47,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 191,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 15,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 143,
    },
    Huft {
        exop: 0,
        bits: 8,
        base: 79,
    },
    Huft {
        exop: 0,
        bits: 9,
        base: 255,
    },
];

/// `fixed distance tree, `fixed_td` from inffixed.h`
const FIXED_TD: [Huft; 32] = [
    Huft {
        exop: 80,
        bits: 5,
        base: 1,
    },
    Huft {
        exop: 87,
        bits: 5,
        base: 257,
    },
    Huft {
        exop: 83,
        bits: 5,
        base: 17,
    },
    Huft {
        exop: 91,
        bits: 5,
        base: 4097,
    },
    Huft {
        exop: 81,
        bits: 5,
        base: 5,
    },
    Huft {
        exop: 89,
        bits: 5,
        base: 1025,
    },
    Huft {
        exop: 85,
        bits: 5,
        base: 65,
    },
    Huft {
        exop: 93,
        bits: 5,
        base: 16385,
    },
    Huft {
        exop: 80,
        bits: 5,
        base: 3,
    },
    Huft {
        exop: 88,
        bits: 5,
        base: 513,
    },
    Huft {
        exop: 84,
        bits: 5,
        base: 33,
    },
    Huft {
        exop: 92,
        bits: 5,
        base: 8193,
    },
    Huft {
        exop: 82,
        bits: 5,
        base: 9,
    },
    Huft {
        exop: 90,
        bits: 5,
        base: 2049,
    },
    Huft {
        exop: 86,
        bits: 5,
        base: 129,
    },
    Huft {
        exop: 192,
        bits: 5,
        base: 24577,
    },
    Huft {
        exop: 80,
        bits: 5,
        base: 2,
    },
    Huft {
        exop: 87,
        bits: 5,
        base: 385,
    },
    Huft {
        exop: 83,
        bits: 5,
        base: 25,
    },
    Huft {
        exop: 91,
        bits: 5,
        base: 6145,
    },
    Huft {
        exop: 81,
        bits: 5,
        base: 7,
    },
    Huft {
        exop: 89,
        bits: 5,
        base: 1537,
    },
    Huft {
        exop: 85,
        bits: 5,
        base: 97,
    },
    Huft {
        exop: 93,
        bits: 5,
        base: 24577,
    },
    Huft {
        exop: 80,
        bits: 5,
        base: 4,
    },
    Huft {
        exop: 88,
        bits: 5,
        base: 769,
    },
    Huft {
        exop: 84,
        bits: 5,
        base: 49,
    },
    Huft {
        exop: 92,
        bits: 5,
        base: 12289,
    },
    Huft {
        exop: 82,
        bits: 5,
        base: 13,
    },
    Huft {
        exop: 90,
        bits: 5,
        base: 3073,
    },
    Huft {
        exop: 86,
        bits: 5,
        base: 193,
    },
    Huft {
        exop: 192,
        bits: 5,
        base: 24577,
    },
];

/// `inflate_mask` from infutil.c: AND-ing with `INFLATE_MASK[n]` keeps the
/// low `n` bits of the bit buffer.
const INFLATE_MASK: [u32; 17] = [
    0x0000, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff, 0x01ff, 0x03ff, 0x07ff, 0x0fff,
    0x1fff, 0x3fff, 0x7fff, 0xffff,
];

/// `cplens` from inftrees.c: copy lengths for literal/length codes 257..285.
const CPLENS: [u32; 31] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195,
    227, 258, 0, 0,
];

/// `cplext` from inftrees.c: extra bits for literal/length codes 257..285
/// (the two trailing `112`s mark codes 286/287 as invalid).
const CPLEXT: [u32; 31] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 112, 112,
];

/// `cpdist` from inftrees.c: copy offsets for distance codes 0..29.
const CPDIST: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577,
];

/// `cpdext` from inftrees.c: extra bits for distance codes.
const CPDEXT: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];

/// `border` from infblock.c: order of the bit-length code lengths.
const BORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// A handle to a Huffman lookup tree inside the shared `MANY`-entry arena,
/// or to one of the two fixed trees of `inffixed.h`.
///
/// C keeps raw `inflate_huft *` pointers; here a tree is identified by
/// which array it lives in, so the arena can be borrowed immutably while
/// the decode state is mutated (`None` is not representable: a null tree
/// is handled by the caller, see `CodesState::dtree`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TreeRef {
    /// The fixed literal/length tree (`fixed_tl`, 512 entries).
    FixedLit,
    /// The fixed distance tree (`fixed_td`, 32 entries).
    FixedDist,
    /// A dynamic tree starting at the given offset of the arena.
    Arena(u32),
}

/// Resolves entry `idx` of `tree`, mirroring C's `tree[idx]`.
///
/// Returns `None` when the index falls outside the tree (only reachable
/// with corrupt input); callers map that to an invalid-code error instead
/// of indexing out of bounds.
#[inline]
fn huft_at(tree: TreeRef, hufts: &[Huft], idx: u32) -> Option<Huft> {
    let idx = idx as usize;
    match tree {
        TreeRef::FixedLit => FIXED_TL.get(idx).copied(),
        TreeRef::FixedDist => FIXED_TD.get(idx).copied(),
        TreeRef::Arena(base) => hufts
            .get((base as usize).checked_add(idx)?)
            .copied(),
    }
}

/// Follows a sub-table link, mirroring C's `tree = t + t->base`.
///
/// `idx` is the index of the entry that carried the link. Fixed trees are
/// flat (they never contain sub-table links), so they yield `None`, which
/// the decoder treats as an invalid code.
#[inline]
fn huft_sub(tree: TreeRef, idx: u32, base: u32) -> Option<TreeRef> {
    match tree {
        TreeRef::Arena(t) => Some(TreeRef::Arena(t.checked_add(idx)?.checked_add(base)?)),
        TreeRef::FixedLit | TreeRef::FixedDist => None,
    }
}

/// Result of `inflate_trees_bits` (`tb`, `bb` and the zlib status code).
struct BitTree {
    /// zlib status: `Z_OK` or `Z_DATA_ERROR`.
    code: i32,
    /// The 19-symbol code-length decoding tree, `None` when empty.
    tb: Option<u32>,
    /// Actual lookup bits of the code-length tree (`bb`).
    bb: u32,
}

/// Result of `inflate_trees_dynamic` (`tl`, `td`, `bl`, `bd` and status).
struct DynTrees {
    /// zlib status: `Z_OK`, `Z_BUF_ERROR`, `Z_DATA_ERROR` or `Z_MEM_ERROR`.
    code: i32,
    /// Literal/length tree, `None` when the input had no codes.
    tl: Option<u32>,
    /// Distance tree, `None` when the block carries no distance codes.
    td: Option<u32>,
    /// Actual lookup bits of the literal/length tree (`bl`).
    bl: u32,
    /// Actual lookup bits of the distance tree (`bd`).
    bd: u32,
}

/// Working set of `inftrees.c`: the shared table arena (`hp`), the number
/// of entries used (`hn`) and the value work area (`v`).
///
/// The arena and work area are exactly the allocations C performs with
/// `ZALLOC`; sizes are protocol constants (`MANY` entries, 288 slots), so
/// no attacker-controlled length ever reaches an allocator.
struct HuffCtx<'a> {
    /// `hp`: space for the tables (up to [`MANY`] entries).
    hp: &'a mut [Huft],
    /// `hn`: entries of `hp` already in use; reset per tree set.
    hn: usize,
    /// `v`: work area, values in order of bit length (288 slots is the
    /// maximum needed by any caller).
    v: Vec<u32>,
}

impl<'a> HuffCtx<'a> {
    /// Allocates the work area (`inflate_trees_bits` /
    /// `inflate_trees_dynamic` `ZALLOC`); `None` reports `Z_MEM_ERROR`.
    fn new(hp: &'a mut [Huft]) -> Option<HuffCtx<'a>> {
        let mut v = Vec::new();
        v.try_reserve_exact(288).ok()?;
        v.resize(288, 0);
        Some(HuffCtx { hp, hn: 0, v })
    }

    /// Port of `huft_build` (inftrees.c): builds decoding tables for the
    /// code lengths in `b` into the shared arena.
    ///
    /// `s` is the number of simple-valued codes (`0..s-1` decode to
    /// themselves); `d`/`e` give base value and extra-bit count for the
    /// remaining codes; `m` holds the requested first-level lookup bits on
    /// entry and the achieved bits on return.
    ///
    /// Returns `Z_OK` on success, `Z_BUF_ERROR` for an incomplete code set
    /// (the tables are still built) or `Z_DATA_ERROR` for invalid input,
    /// together with the arena offset of the first table (`None` when the
    /// code set is empty, i.e. C's `*t = Z_NULL`).
    fn build(&mut self, b: &[u32], s: usize, d: &[u32], e: &[u32], m: &mut u32) -> (i32, Option<u32>) {
        let n = b.len();
        let hp = &mut *self.hp;
        let hn = &mut self.hn;
        let v = &mut self.v;
        // Generate counts for each bit length (`c[]`, assume <= BMAX).
        let mut c = [0u32; BMAX + 1];
        for &len in b {
            let Some(slot) = c.get_mut(len as usize) else {
                return (Z_DATA_ERROR, None); // longer than BMAX: invalid
            };
            *slot += 1;
        }
        if c[0] as usize == n {
            // null input, all zero length codes
            *m = 0;
            return (Z_OK, None);
        }
        // Find minimum and maximum length, bound *m by those.
        let mut l = *m as i32;
        let mut j = 1usize;
        while j <= BMAX && c[j] == 0 {
            j += 1;
        }
        if j > BMAX {
            return (Z_DATA_ERROR, None);
        }
        let k = j; // minimum code length
        if l < j as i32 {
            l = j as i32;
        }
        let mut i = BMAX;
        while i > 0 && c[i] == 0 {
            i -= 1;
        }
        let g = i; // maximum code length (>= 1 here)
        if l > g as i32 {
            l = g as i32;
        }
        *m = l as u32;
        let mut y: i32 = 1i32 << k;
        let mut jj = k;
        while jj < g {
            y -= c[jj] as i32;
            if y < 0 {
                return (Z_DATA_ERROR, None);
            }
            jj += 1;
            y <<= 1;
        }
        y -= c[g] as i32;
        if y < 0 {
            return (Z_DATA_ERROR, None);
        }
        c[g] += y as u32;
        let mut x = [0u32; BMAX + 1];
        let mut acc = 0u32;
        let mut pc = 1usize;
        let mut xc = 2usize;
        let mut ic = g;
        while ic > 1 {
            acc += c[pc];
            pc += 1;
            x[xc] = acc;
            xc += 1;
            ic -= 1;
        }
        for (sym, &len) in b.iter().enumerate() {
            if len != 0 {
                let slot = x[len as usize];
                x[len as usize] += 1;
                let Some(cell) = v.get_mut(slot as usize) else {
                    return (Z_DATA_ERROR, None);
                };
                *cell = sym as u32;
            }
        }
        let nval = x[g] as usize; // length of v
        x[0] = 0;
        let mut code: u32 = 0; // first Huffman code is zero
        let mut pv = 0usize; // cursor into v
        let mut h: i32 = -1;
        let mut w: i32 = -l; // bits decoded == (l * h)
        let mut u = [0u32; BMAX]; // table stack (arena offsets)
        let mut q_off: Option<u32> = None; // current table (`q`)
        let mut root: Option<u32> = None; // first table (`*t = q` at h == 0)
        let mut z: u32 = 0; // entries in current table

        let mut kk = k as i32; // k already is bits in shortest code
        while kk <= g as i32 {
            let mut a = c[kk as usize];
            while a > 0 {
                a -= 1;
                // Here `code` is the Huffman code of length kk for value *p
                // make tables up to required level.
                while kk > w + l {
                    h += 1;
                    w += l; // previous table always l bits.
                    let hidx = h as usize;
                    if hidx >= BMAX {
                        return (Z_DATA_ERROR, None);
                    }
                    let mut zmax = (g as i32 - w) as u32;
                    if zmax > l as u32 {
                        zmax = l as u32;
                    }
                    let mut jbits = (kk - w) as u32; // j
                    let mut f: u32 = 1u32 << jbits;
                    if f > a + 1 {
                        // too few codes for jbits-bit table.
                        f -= a + 1;
                        let mut xp = kk as usize; // xp = c + k
                        if jbits < zmax {
                            // try smaller tables up to zmax bits.
                            while {
                                jbits += 1;
                                jbits < zmax
                            } {
                                xp += 1;
                                let Some(&cx) = c.get(xp) else {
                                    return (Z_DATA_ERROR, None);
                                };
                                f <<= 1;
                                if f <= cx {
                                    break; // enough codes to use up jbits bits
                                }
                                f -= cx; // else deduct codes from patterns
                            }
                        }
                    }
                    z = 1u32 << jbits; // table entries for jbits-bit table
                    if *hn + z as usize > MANY {
                        return (Z_DATA_ERROR, None); // overflow of MANY
                    }
                    let q = *hn as u32;
                    if hp.len() < MANY {
                        return (Z_DATA_ERROR, None);
                    }
                    *hn += z as usize;
                    q_off = Some(q);
                    u[hidx] = q; // u[h] = q = hp + *hn
                    if h > 0 {
                        x[hidx] = code; // save pattern for backing up
                        let r = Huft {
                            exop: jbits as u8, // bits in this sub-table
                            bits: l as u8,     // bits to dump before this link
                            base: 0,
                        };
                        let sub = code >> (w - l);
                        let Some(base) = q
                            .checked_sub(u[hidx - 1])
                            .and_then(|b| b.checked_sub(sub))
                        else {
                            return (Z_DATA_ERROR, None);
                        };
                        let entry = Huft { base, ..r };
                        let Some(slot) = hp.get_mut((u[hidx - 1] + sub) as usize) else {
                            return (Z_DATA_ERROR, None);
                        };
                        *slot = entry;
                    } else {
                        root = Some(q); // first table is the returned result
                    }
                }
                let qo = match q_off {
                    Some(o) => o as usize,
                    None => return (Z_DATA_ERROR, None),
                };
                if qo + z as usize > hp.len() {
                    return (Z_DATA_ERROR, None);
                }
                let mut r = Huft {
                    exop: 0,
                    bits: (kk - w) as u8,
                    base: 0,
                };
                if pv >= nval {
                    r.exop = 128 + 64;
                } else {
                    let val = v[pv] as usize;
                    if val < s {
                        r.exop = if val < 256 { 0 } else { 32 + 64 };
                        r.base = v[pv];
                        pv += 1;
                    } else {
                        let idx = val - s;
                        let (Some(&ext), Some(&base)) = (e.get(idx), d.get(idx)) else {
                            return (Z_DATA_ERROR, None);
                        };
                        r.exop = (ext + 16 + 64) as u8;
                        r.base = base;
                        pv += 1;
                    }
                }
                let f = 1u32 << (kk - w);
                let mut jdx = code >> w;
                while jdx < z {
                    hp[qo + jdx as usize] = r;
                    jdx += f;
                }
                let mut bit = 1u32 << (kk - 1);
                while code & bit != 0 {
                    code ^= bit;
                    bit >>= 1;
                }
                code ^= bit;
                let mut mask = if (0..32).contains(&w) { (1u32 << w) - 1 } else { 0 };
                while h >= 0 && (code & mask) != x[h as usize] {
                    h -= 1;
                    w -= l;
                    mask = if (0..32).contains(&w) { (1u32 << w) - 1 } else { 0 };
                }
            }
            kk += 1;
        }
        (if y != 0 && g != 1 { Z_BUF_ERROR } else { Z_OK }, root)
    }

    /// Port of `inflate_trees_bits`: builds the 19-symbol code-length tree.
    fn trees_bits(&mut self, c: &[u32], bb: u32) -> BitTree {
        self.hn = 0; // hn = 0 (local in C)
        let mut m = bb;
        let (code, tb) = self.build(c, 19, &[], &[], &mut m);
        let mut code = code;
        if code == Z_DATA_ERROR {
            // "oversubscribed dynamic bit lengths tree"
        } else if code == Z_BUF_ERROR || m == 0 {
            // "incomplete dynamic bit lengths tree"
            code = Z_DATA_ERROR;
        }
        BitTree { code, tb, bb: m }
    }

    /// Port of `inflate_trees_dynamic`: builds the literal/length and the
    /// distance trees of a dynamic block from `nl + nd` code lengths.
    fn trees_dynamic(&mut self, nl: usize, nd: usize, c: &[u32], bl: u32, bd: u32) -> DynTrees {
        self.hn = 0; // hn = 0 (local in C)
        let Some(head) = c.get(..nl) else {
            return DynTrees {
                code: Z_DATA_ERROR,
                tl: None,
                td: None,
                bl,
                bd,
            };
        };
        let Some(end) = nl.checked_add(nd) else {
            return DynTrees {
                code: Z_DATA_ERROR,
                tl: None,
                td: None,
                bl,
                bd,
            };
        };
        let Some(tail) = c.get(nl..end) else {
            return DynTrees {
                code: Z_DATA_ERROR,
                tl: None,
                td: None,
                bl,
                bd,
            };
        };
        let mut m1 = bl;
        let (mut code, tl) = self.build(head, 257, &CPLENS, &CPLEXT, &mut m1);
        if code != Z_OK || m1 == 0 {
            if code == Z_DATA_ERROR {
                // "oversubscribed literal/length tree"
            } else if code != Z_MEM_ERROR {
                // "incomplete literal/length tree"
                code = Z_DATA_ERROR;
            }
            return DynTrees {
                code,
                tl,
                td: None,
                bl: m1,
                bd,
            };
        }

        let mut m2 = bd;
        let (code2, td) = self.build(tail, 0, &CPDIST, &CPDEXT, &mut m2);
        let mut code = code2;
        if code != Z_OK || (m2 == 0 && nl > 257) {
            if code == Z_DATA_ERROR {
                // "oversubscribed distance tree"
            } else if code == Z_BUF_ERROR {
                // "incomplete distance tree"
                code = Z_DATA_ERROR;
            } else if code != Z_MEM_ERROR {
                // "empty distance tree with lengths"
                code = Z_DATA_ERROR;
            }
            return DynTrees {
                code,
                tl,
                td,
                bl: m1,
                bd: m2,
            };
        }

        DynTrees {
            code,
            tl,
            td,
            bl: m1,
            bd: m2,
        }
    }
}

/// `struct internal_state` mode of inflate.c (gzip/zlib wrapper framing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InflMode {
    /// waiting for method byte
    Method,
    /// waiting for flag byte
    Flag,
    /// four dictionary check bytes to go
    Dict4,
    /// three dictionary check bytes to go
    Dict3,
    /// two dictionary check bytes to go
    Dict2,
    /// one dictionary check byte to go
    Dict1,
    /// waiting for `inflateSetDictionary` (never satisfiable here)
    Dict0,
    /// decompressing blocks
    Blocks,
    /// four check bytes to go
    Check4,
    /// three check bytes to go
    Check3,
    /// two check bytes to go
    Check2,
    /// one check byte to go
    Check1,
    /// finished check, done
    Done,
    /// got an error, stay here
    Bad,
}

/// `inflate_block_mode` from infutil.h: how a block is being decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockMode {
    /// get type bits (3, including end bit)
    Type,
    /// get lengths for stored
    Lens,
    /// processing stored block
    Stored,
    /// get table lengths
    Table,
    /// get bit lengths tree for a dynamic block
    Btree,
    /// get length, distance trees for a dynamic block
    Dtree,
    /// processing fixed or dynamic block
    Codes,
    /// output remaining window bytes
    Dry,
    /// finished last block, done
    Done,
    /// got a data error, stuck here
    Bad,
}

/// `inflate_codes_mode` from infcodes.c; the comment marks what the mode
/// waits for: `i:` input, `o:` output, `x:` nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodesMode {
    /// x: set up for LEN
    Start,
    /// i: get length/literal/eob next
    Len,
    /// i: getting length extra (have base)
    LenExt,
    /// i: get distance next
    Dist,
    /// i: getting distance extra
    DistExt,
    /// o: copying bytes in window, waiting for space
    Copy,
    /// o: got literal, waiting for output space
    Lit,
    /// o: got eob, possibly still output waiting
    Wash,
    /// x: got eob and all data flushed
    End,
    /// x: got error
    BadCode,
}

/// The `sub` union of `struct inflate_codes_state`.
#[derive(Clone, Copy, Debug)]
enum CodesSub {
    /// `code`: where in the tree we are (LEN / DIST).
    Code { tree: TreeRef, need: u32 },
    /// `lit`: the literal to emit (LIT).
    Lit(u32),
    /// `copy`: extra-bit count and distance (LENEXT / DISTEXT / COPY).
    Copy { get: u32, dist: u32 },
    /// Unused (START / WASH / END / BADCODE).
    None,
}

/// Port of `struct inflate_codes_state` (infcodes.c).
#[derive(Clone, Copy, Debug)]
struct CodesState {
    /// current mode
    mode: CodesMode,
    /// current match length (LENEXT / COPY)
    len: u32,
    /// mode dependent information
    sub: CodesSub,
    /// ltree bits decoded per branch
    lbits: u8,
    /// dtree bits decoded per branch
    dbits: u8,
    /// literal/length/eob tree
    ltree: TreeRef,
    /// distance tree; `None` when the block carries no distance codes
    /// (C keeps a null pointer that is never dereferenced in that case)
    dtree: Option<TreeRef>,
}

impl CodesState {
    /// Port of `inflate_codes_new`: builds the per-block decode state.
    fn new(lbits: u32, dbits: u32, ltree: TreeRef, dtree: Option<TreeRef>) -> Self {
        CodesState {
            mode: CodesMode::Start,
            len: 0,
            sub: CodesSub::None,
            lbits: lbits as u8,
            dbits: dbits as u8,
            ltree,
            dtree,
        }
    }
}

/// The `sub` union of `struct inflate_blocks_state`.
enum BlockSub {
    /// Unused (TYPE / LENS / DRY / DONE / BAD).
    None,
    /// `left`: bytes left to copy (STORED).
    Left(u32),
    /// `trees`: decoding info for dynamic block headers (TABLE / BTREE / DTREE).
    Trees(TreesState),
    /// `decode`: current block decoder (CODES).
    Decode(CodesState),
}

/// `sub.trees` of `struct inflate_blocks_state`.
struct TreesState {
    /// table lengths (14 bits)
    table: u32,
    /// index into blens
    index: u32,
    /// bit lengths of codes (allocated per block, `ZFREE`d after use)
    blens: Option<Vec<u32>>,
    /// bit length tree depth
    bb: u32,
    /// bit length decoding tree
    tb: Option<u32>,
}

/// zlib stream counters: `total_in`, `total_out` and `adler` of
/// `z_stream`. The rest of `z_stream` (buffers, state pointer) is
/// represented by the Rust types around it.
#[derive(Clone, Copy, Debug, Default)]
struct ZMeta {
    /// bytes consumed from the input so far
    total_in: u64,
    /// bytes produced into the caller's output so far
    total_out: u64,
    /// running check value (`z->adler`)
    adler: u32,
}

/// The sliding window plus read/write pointers and the check state of
/// `struct inflate_blocks_state` (infutil.h).
struct Win {
    /// `window`: `1 << wbits` bytes (32 KiB for wbits = 15)
    buf: Vec<u8>,
    /// `read`: window read pointer (index into `buf`)
    read: usize,
    /// `write`: window write pointer (index into `buf`)
    write: usize,
    /// `checkfn`: `adler32` for zlib streams, `None` for raw deflate
    checkfn: Option<fn(u32, &[u8]) -> u32>,
    /// `check`: check value of the output produced so far
    check: u32,
}

impl Win {
    /// Allocates the window (`inflate_blocks_new`'s `ZALLOC(z, 1, w)`);
    /// `None` reports `Z_MEM_ERROR`.
    fn new(size: usize, checkfn: Option<fn(u32, &[u8]) -> u32>) -> Option<Win> {
        let mut buf = Vec::new();
        buf.try_reserve_exact(size).ok()?;
        buf.resize(size, 0);
        Some(Win {
            buf,
            read: 0,
            write: 0,
            checkfn,
            check: if checkfn.is_some() { adler32_init() } else { 0 },
        })
    }

    /// Port of `inflate_blocks_reset`'s window/check half; returns the
    /// previous check value (`*c = s->check` in C).
    fn reset(&mut self, z: &mut ZMeta) -> u32 {
        let old = self.check;
        self.read = 0;
        self.write = 0;
        if self.checkfn.is_some() {
            self.check = adler32_init();
            z.adler = self.check;
        }
        old
    }

    /// `WAVAIL`: bytes that may still be written to the window.
    #[inline]
    fn wavail(&self) -> usize {
        if self.write < self.read {
            self.read - self.write - 1
        } else {
            self.buf.len() - self.write
        }
    }

    /// `WRAP`: wrap the write pointer to the window start when it reached
    /// the end and the window is not being read from the front.
    #[inline]
    fn wrap(&mut self) {
        if self.write == self.buf.len() && self.read != 0 {
            self.write = 0;
        }
    }

    /// Port of `inflate_flush` (infutil.c): copies pending window bytes to
    /// the caller's output buffer, updates counters and the check value.
    fn flush(&mut self, z: &mut ZMeta, out: &mut OutBuf<'_>, mut r: i32) -> i32 {
        let len = self.buf.len();
        let mut q = self.read;

        // bytes from the read pointer up to the write pointer (or the end)
        let mut n = (if q <= self.write { self.write } else { len }) - q;
        if n > out.avail() {
            n = out.avail();
        }
        if n != 0 && r == Z_BUF_ERROR {
            r = Z_OK;
        }
        z.total_out += n as u64;
        if let Some(f) = self.checkfn {
            self.check = f(self.check, &self.buf[q..q + n]);
            z.adler = self.check;
        }
        out.copy_from(&self.buf[q..q + n]);
        q += n;
        if q == len {
            q = 0;
            if self.write == len {
                self.write = 0;
            }
            let mut n2 = self.write - q;
            if n2 > out.avail() {
                n2 = out.avail();
            }
            if n2 != 0 && r == Z_BUF_ERROR {
                r = Z_OK;
            }
            z.total_out += n2 as u64;
            if let Some(f) = self.checkfn {
                self.check = f(self.check, &self.buf[q..q + n2]);
                z.adler = self.check;
            }
            out.copy_from(&self.buf[q..q + n2]);
            q += n2;
        }

        self.read = q;
        r
    }
}

/// Port of `struct inflate_blocks_state` (infutil.h).
struct BlocksState {
    /// current block mode
    mode: BlockMode,
    /// mode dependent information
    sub: BlockSub,
    /// true if this block is the last block
    last: bool,
    /// bits in bit buffer
    bitk: u32,
    /// bit buffer
    bitb: u64,
    /// single allocation for tree space (`MANY` entries)
    hufts: Vec<Huft>,
    /// sliding window, pointers and check state
    win: Win,
}

impl BlocksState {
    /// Port of `inflate_blocks_new` + `inflate_blocks_reset`; `None`
    /// reports `Z_MEM_ERROR`.
    fn new(wbits: u32, checkfn: Option<fn(u32, &[u8]) -> u32>) -> Option<BlocksState> {
        let mut hufts = Vec::new();
        hufts.try_reserve_exact(MANY).ok()?;
        hufts.resize(MANY, Huft::default());
        let win = Win::new(1usize << wbits, checkfn)?;
        Some(BlocksState {
            mode: BlockMode::Type,
            sub: BlockSub::None,
            last: false,
            bitk: 0,
            bitb: 0,
            hufts,
            win,
        })
    }

    /// Port of `inflate_blocks_reset`: drops the per-block allocations,
    /// clears the bit buffer and window pointers, and restarts the check.
    /// Returns the previous check value (`*c = s->check`).
    fn reset(&mut self, z: &mut ZMeta) -> u32 {
        // read the check value first, as `*c = s->check` does in C
        let old = self.win.check;
        // BTREE/DTREE free blens; CODES frees the decoder state
        self.sub = BlockSub::None;
        self.mode = BlockMode::Type;
        self.bitk = 0;
        self.bitb = 0;
        self.win.reset(z);
        old
    }
}

/// Port of `struct internal_state` (inflate.c): the wrapper-level state
/// plus the embedded block decoder and the stream counters.
struct Inflater {
    /// current inflate mode
    mode: InflMode,
    /// `sub.method`: method byte of the zlib header
    method: u8,
    /// `sub.check.was`: computed check value of the finished stream
    check_was: u64,
    /// `sub.check.need`: stream check value being assembled
    check_need: u64,
    /// no zlib header and no check (raw deflate, as gzip uses)
    nowrap: bool,
    /// log2(window size) (8..15)
    wbits: u32,
    /// current inflate_blocks state
    blocks: BlocksState,
    /// stream counters
    z: ZMeta,
}

impl Inflater {
    /// Port of `inflateInit2_` + `inflateReset`; `None` reports a
    /// `Z_STREAM_ERROR` (bad `wbits`) or `Z_MEM_ERROR`.
    fn new(nowrap: bool, wbits: u32) -> Option<Inflater> {
        if !(8..=15).contains(&wbits) {
            return None;
        }
        let checkfn = if nowrap {
            None
        } else {
            Some(adler32 as fn(u32, &[u8]) -> u32)
        };
        let blocks = BlocksState::new(wbits, checkfn)?;
        let mut inf = Inflater {
            mode: InflMode::Bad,
            method: 0,
            check_was: 0,
            check_need: 0,
            nowrap,
            wbits,
            blocks,
            z: ZMeta::default(),
        };
        inf.reset();
        Some(inf)
    }

    /// Port of `inflateReset`.
    fn reset(&mut self) {
        self.z.total_in = 0;
        self.z.total_out = 0;
        self.mode = if self.nowrap {
            InflMode::Blocks
        } else {
            InflMode::Method
        };
        self.blocks.reset(&mut self.z);
    }

    /// True once the deflate stream finished (`mode == DONE`).
    #[inline]
    fn at_stream_end(&self) -> bool {
        self.mode == InflMode::Done
    }
}

/// zlib's input cursor: `z->next_in` / `avail_in` plus, for the gzip
/// wrapper, FreeType's `FT_GZIP_BUFFER_SIZE` refill buffer and the
/// `stream->pos` it is read from (`ft_gzip_file_fill_input`).
struct Inp<'a> {
    /// where bytes come from
    kind: InpKind<'a>,
    /// bytes handed out minus bytes handed back since construction
    consumed: u64,
    /// next byte index relative to the buffer start; `-1` means `pre`
    idx: i64,
    /// the byte preceding the buffer: WASH can return one byte (see
    /// `unget`)
    pre: Option<u8>,
    /// number of valid bytes in the buffer
    len: usize,
}

/// Backing store of an [`Inp`].
enum InpKind<'a> {
    /// One-shot input (`FT_Gzip_Uncompress`): never refills.
    Slice(&'a [u8]),
    /// Gzip payload: refills from `src` into `buf` starting at `pos`.
    Source {
        /// underlying (compressed) stream
        src: &'a dyn StreamSource,
        /// read position inside `src`
        pos: &'a mut u64,
        /// the caller's 4 KiB input buffer
        buf: &'a mut Vec<u8>,
    },
}

impl<'a> Inp<'a> {
    /// A cursor over a fixed slice (`uncompress`).
    fn one_shot(src: &'a [u8]) -> Inp<'a> {
        Inp {
            kind: InpKind::Slice(src),
            consumed: 0,
            idx: 0,
            pre: None,
            len: src.len(),
        }
    }

    /// Bytes still available to `pull` (`z->avail_in`).
    #[inline]
    fn avail(&self) -> usize {
        if self.idx < 0 {
            self.len + 1
        } else {
            self.len.saturating_sub(self.idx as usize)
        }
    }

    /// True when no byte is available (`avail_in == 0`).
    #[inline]
    fn exhausted(&self) -> bool {
        self.avail() == 0
    }

    /// Port of `NEXTBYTE`: the next input byte, or `None` when the input
    /// is exhausted.
    #[inline]
    fn pull(&mut self) -> Option<u8> {
        if self.idx < 0 {
            let b = self.pre?;
            self.idx = 0;
            self.consumed += 1;
            return Some(b);
        }
        let i = self.idx as usize;
        if i >= self.len {
            return None;
        }
        let b = match &self.kind {
            InpKind::Slice(s) => *s.get(i)?,
            InpKind::Source { buf, .. } => *buf.get(i)?,
        };
        self.idx += 1;
        self.consumed += 1;
        Some(b)
    }

    /// Port of WASH's `p--; n++`: returns the last consumed byte to the
    /// stream so it is read again after the next `pull`.
    ///
    /// Only ever runs once per WASH visit and only after at least one byte
    /// was consumed, so `idx` never drops below `-1`.
    fn unget(&mut self) {
        if self.idx > 0 {
            self.idx -= 1;
            self.consumed -= 1;
        } else if self.idx == 0 && self.pre.is_some() {
            self.idx = -1;
            self.consumed -= 1;
        }
    }

    /// Moves the bytes counted in `consumed` into `z.total_in`
    /// (`UPDIN` at each `UPDATE`/`LEAVE`, batched per `inflate` call).
    fn take_consumed(&mut self, z: &mut ZMeta) {
        z.total_in += self.consumed;
        self.consumed = 0;
    }

    /// Port of `ft_gzip_file_fill_input`: reloads the buffer from the
    /// source. Reports [`TtError::INVALID_STREAM_OPERATION`] when no data
    /// is left (the C code returns the same error, which its caller turns
    /// into a short read).
    fn fill(&mut self) -> TtResult<()> {
        debug_assert!(self.exhausted(), "fill only when avail_in == 0");
        let prev_len = self.len;
        match &mut self.kind {
            InpKind::Slice(_) => Err(TtError::INVALID_STREAM_OPERATION),
            InpKind::Source { src, pos, buf } => {
                // the byte that precedes the new content in the stream
                if prev_len > 0 {
                    self.pre = buf.get(prev_len - 1).copied();
                }
                let at = **pos;
                let n = src.read_at(at, buf.as_mut_slice())?;
                if n == 0 {
                    return Err(TtError::INVALID_STREAM_OPERATION);
                }
                **pos = at.saturating_add(n as u64);
                self.len = n;
                self.idx = 0;
                Ok(())
            }
        }
    }
}

/// zlib's output cursor: `z->next_out` / `avail_out` for one
/// `FT_GZIP_BUFFER_SIZE` (or caller-sized) batch.
struct OutBuf<'a> {
    /// destination
    buf: &'a mut [u8],
    /// bytes written so far (`z->avail_out = buf.len() - pos`)
    pos: usize,
}

impl<'a> OutBuf<'a> {
    /// Wraps `buf` with nothing written yet.
    #[inline]
    fn new(buf: &'a mut [u8]) -> OutBuf<'a> {
        OutBuf { buf, pos: 0 }
    }

    /// `z->avail_out`.
    #[inline]
    fn avail(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// The bytes written so far (`z->next_out - original next_out`).
    #[inline]
    fn written(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    /// `zmemcpy` into the buffer, capped by the space left.
    #[inline]
    fn copy_from(&mut self, src: &[u8]) {
        let n = src.len().min(self.avail());
        self.buf[self.pos..self.pos + n].copy_from_slice(&src[..n]);
        self.pos += n;
    }
}

/// zlib's `NEEDBITS(j)` for the block layer (infutil.h): pulls input bytes
/// until `j` bits sit in the bit buffer. When the input runs out the macro
/// performs C's `LEAVE` (flush the window and return `r`), which is why it
/// may expand to a `return`.
macro_rules! needbits {
    ($self:expr, $z:expr, $inp:expr, $out:expr, $r:expr, $j:expr) => {
        while $self.bitk < $j {
            match $inp.pull() {
                Some(b) => {
                    $r = Z_OK;
                    $self.bitb |= (b as u64) << $self.bitk;
                    $self.bitk += 8;
                }
                None => return $self.win.flush($z, $out, $r),
            }
        }
    };
}

/// zlib's `DUMPBITS(j)`: removes `j` bits from the bit buffer (the caller
/// has ensured they are there; the clamp only guards the shift).
macro_rules! dumpbits {
    ($self:expr, $j:expr) => {{
        let n = $j;
        debug_assert!($self.bitk >= n, "DUMPBITS without enough bits");
        $self.bitb >>= n.min(63);
        $self.bitk = $self.bitk.saturating_sub(n);
    }};
}

/// zlib's `NEEDOUT` (infutil.h): guarantees one writable window byte by
/// wrapping the write pointer and/or flushing pending output to the
/// caller's buffer; performs C's `LEAVE` when no space can be had.
macro_rules! needout {
    ($win:expr, $z:expr, $out:expr, $r:expr) => {
        if $win.wavail() == 0 {
            $win.wrap();
            if $win.wavail() == 0 {
                $r = $win.flush($z, $out, $r);
                $win.wrap();
                if $win.wavail() == 0 {
                    return $win.flush($z, $out, $r);
                }
            }
        }
        $r = Z_OK;
    };
}

impl BlocksState {
    /// Port of `inflate_codes` (infcodes.c): decodes literals, matches and
    /// the end-of-block marker of the current block.
    ///
    /// All of C's `LOAD`/`UPDATE` bookkeeping is implicit: the bit buffer
    /// lives in `self.bitb`/`self.bitk`, the input cursor in `inp`, the
    /// window pointers in `self.win`.
    fn inflate_codes(&mut self, z: &mut ZMeta, mut r: i32, inp: &mut Inp<'_>, out: &mut OutBuf<'_>) -> i32 {
        // `inflate_codes` is only called with mode == CODES (see the
        // CODES arm of `inflate_blocks`); anything else is a logic error.
        let BlockSub::Decode(codes) = &mut self.sub else {
            return Z_STREAM_ERROR;
        };
        loop {
            match codes.mode {
                CodesMode::Start => {
                    // x: set up for LEN (SLOW build: no inflate_fast path)
                    codes.sub = CodesSub::Code {
                        tree: codes.ltree,
                        need: u32::from(codes.lbits),
                    };
                    codes.mode = CodesMode::Len;
                }
                CodesMode::Len => {
                    // i: get length/literal/eob next
                    let (tree, need) = match codes.sub {
                        CodesSub::Code { tree, need } => (tree, need),
                        _ => return Z_STREAM_ERROR,
                    };
                    needbits!(self, z, inp, out, r, need);
                    let Some(&mask) = INFLATE_MASK.get(need as usize) else {
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    };
                    let idx = (self.bitb as u32) & mask;
                    let Some(entry) = huft_at(tree, &self.hufts, idx) else {
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    };
                    dumpbits!(self, u32::from(entry.bits));
                    let e = u32::from(entry.exop);
                    if e == 0 {
                        // literal
                        codes.sub = CodesSub::Lit(entry.base);
                        codes.mode = CodesMode::Lit;
                    } else if e & 16 != 0 {
                        // length
                        codes.len = entry.base;
                        codes.sub = CodesSub::Copy { get: e & 15, dist: 0 };
                        codes.mode = CodesMode::LenExt;
                    } else if e & 64 == 0 {
                        // next table
                        let Some(next) = huft_sub(tree, idx, entry.base) else {
                            codes.mode = CodesMode::BadCode;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        codes.sub = CodesSub::Code { tree: next, need: e };
                    } else if e & 32 != 0 {
                        // end of block
                        codes.mode = CodesMode::Wash;
                    } else {
                        // invalid code
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    }
                }
                CodesMode::LenExt => {
                    // i: getting length extra (have base)
                    let get = match codes.sub {
                        CodesSub::Copy { get, .. } => get,
                        _ => return Z_STREAM_ERROR,
                    };
                    needbits!(self, z, inp, out, r, get);
                    let extra = (self.bitb as u32) & INFLATE_MASK[get as usize];
                    dumpbits!(self, get);
                    codes.len = codes.len.wrapping_add(extra);
                    let Some(dtree) = codes.dtree else {
                        // unreachable: a block without distance codes has
                        // no length codes either (see `trees_dynamic`)
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    };
                    codes.sub = CodesSub::Code {
                        tree: dtree,
                        need: u32::from(codes.dbits),
                    };
                    codes.mode = CodesMode::Dist;
                }
                CodesMode::Dist => {
                    // i: get distance next
                    let (tree, need) = match codes.sub {
                        CodesSub::Code { tree, need } => (tree, need),
                        _ => return Z_STREAM_ERROR,
                    };
                    needbits!(self, z, inp, out, r, need);
                    let Some(&mask) = INFLATE_MASK.get(need as usize) else {
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    };
                    let idx = (self.bitb as u32) & mask;
                    let Some(entry) = huft_at(tree, &self.hufts, idx) else {
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    };
                    dumpbits!(self, u32::from(entry.bits));
                    let e = u32::from(entry.exop);
                    if e & 16 != 0 {
                        // distance
                        codes.sub = CodesSub::Copy {
                            get: e & 15,
                            dist: entry.base,
                        };
                        codes.mode = CodesMode::DistExt;
                    } else if e & 64 == 0 {
                        // next table
                        let Some(next) = huft_sub(tree, idx, entry.base) else {
                            codes.mode = CodesMode::BadCode;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        codes.sub = CodesSub::Code { tree: next, need: e };
                    } else {
                        // invalid distance code
                        codes.mode = CodesMode::BadCode;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    }
                }
                CodesMode::DistExt => {
                    // i: getting distance extra
                    let (get, base) = match codes.sub {
                        CodesSub::Copy { get, dist } => (get, dist),
                        _ => return Z_STREAM_ERROR,
                    };
                    needbits!(self, z, inp, out, r, get);
                    let extra = (self.bitb as u32) & INFLATE_MASK[get as usize];
                    dumpbits!(self, get);
                    codes.sub = CodesSub::Copy {
                        get,
                        dist: base.wrapping_add(extra),
                    };
                    codes.mode = CodesMode::Copy;
                }
                CodesMode::Copy => {
                    // o: copying bytes in window, waiting for space
                    let dist = match codes.sub {
                        CodesSub::Copy { dist, .. } => dist,
                        _ => return Z_STREAM_ERROR,
                    };
                    let wlen = self.win.buf.len() as u64;
                    if wlen == 0 {
                        return Z_STREAM_ERROR;
                    }
                    // f = q - dist; while (f < window) f += end - window;
                    let mut f = self.win.write as i64 - dist as i64;
                    let mut guard = 0;
                    while f < 0 {
                        f += wlen as i64;
                        guard += 1;
                        if guard > 3 {
                            codes.mode = CodesMode::BadCode;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        }
                    }
                    while codes.len > 0 {
                        needout!(self.win, z, out, r);
                        let Some(&byte) = self.win.buf.get(f as usize) else {
                            codes.mode = CodesMode::BadCode;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        // OUTBYTE(*f++)
                        self.win.put_byte(byte);
                        let fnext = f + 1;
                        f = if fnext >= wlen as i64 { 0 } else { fnext };
                        codes.len -= 1;
                    }
                    codes.mode = CodesMode::Start;
                }
                CodesMode::Lit => {
                    // o: got literal, waiting for output space
                    let CodesSub::Lit(lit) = codes.sub else {
                        return Z_STREAM_ERROR;
                    };
                    needout!(self.win, z, out, r);
                    self.win.put_byte(lit as u8);
                    codes.mode = CodesMode::Start;
                }
                CodesMode::Wash => {
                    // o: got eob, possibly still output waiting
                    if self.bitk > 7 {
                        // return the unused byte, if any
                        self.bitk -= 8;
                        inp.unget();
                    }
                    // FLUSH
                    r = self.win.flush(z, out, r);
                    if self.win.read != self.win.write {
                        // LEAVE
                        return self.win.flush(z, out, r);
                    }
                    codes.mode = CodesMode::End;
                }
                CodesMode::End => {
                    // x: got eob and all data flushed
                    r = Z_STREAM_END;
                    return self.win.flush(z, out, r);
                }
                CodesMode::BadCode => {
                    // x: got error
                    r = Z_DATA_ERROR;
                    return self.win.flush(z, out, r);
                }
            }
        }
    }
}

impl Win {
    /// `OUTBYTE(a)` from infutil.h: appends one byte to the window.
    ///
    /// Callers must have obtained space with `NEEDOUT`; the guard exists
    /// only to keep the no-panic contract if that invariant were ever
    /// broken (it would drop the byte instead of indexing out of bounds).
    #[inline]
    fn put_byte(&mut self, b: u8) {
        debug_assert!(self.write < self.buf.len(), "NEEDOUT must guarantee space");
        if self.write < self.buf.len() {
            self.buf[self.write] = b;
            self.write += 1;
        }
    }
}

impl BlocksState {
    /// Mutable access to the dynamic-header state, when `mode` owns it.
    fn trees(&mut self) -> Option<&mut TreesState> {
        match &mut self.sub {
            BlockSub::Trees(t) => Some(t),
            _ => None,
        }
    }

    /// Shared access to the dynamic-header state, when `mode` owns it.
    fn trees_ref(&self) -> Option<&TreesState> {
        match &self.sub {
            BlockSub::Trees(t) => Some(t),
            _ => None,
        }
    }

    /// Port of `inflate_blocks` (infblock.c): interprets block types until
    /// the stream ends or an error occurs.
    ///
    /// C's switch fall-throughs (TABLE -> BTREE -> DTREE -> CODES -> DRY
    /// -> DONE) become mode re-dispatches; all state lives in the fields,
    /// so an early `LEAVE` resumes exactly where it stopped.
    fn inflate_blocks(&mut self, z: &mut ZMeta, mut r: i32, inp: &mut Inp<'_>, out: &mut OutBuf<'_>) -> i32 {
        loop {
            match self.mode {
                BlockMode::Type => {
                    // get type bits (3, including end bit)
                    needbits!(self, z, inp, out, r, 3u32);
                    let t = (self.bitb as u32) & 7;
                    self.last = t & 1 != 0;
                    match t >> 1 {
                        0 => {
                            // stored
                            dumpbits!(self, 3u32);
                            let pad = self.bitk & 7; // go to byte boundary
                            dumpbits!(self, pad);
                            self.mode = BlockMode::Lens;
                        }
                        1 => {
                            // fixed
                            dumpbits!(self, 3u32);
                            self.sub = BlockSub::Decode(CodesState::new(
                                FIXED_BL,
                                FIXED_BD,
                                TreeRef::FixedLit,
                                Some(TreeRef::FixedDist),
                            ));
                            self.mode = BlockMode::Codes;
                        }
                        2 => {
                            // dynamic
                            dumpbits!(self, 3u32);
                            self.mode = BlockMode::Table;
                        }
                        _ => {
                            // invalid block type (3)
                            dumpbits!(self, 3u32);
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        }
                    }
                }
                BlockMode::Lens => {
                    // get lengths of stored block
                    needbits!(self, z, inp, out, r, 32u32);
                    let b = self.bitb;
                    if ((!b >> 16) & 0xffff) != (b & 0xffff) {
                        // invalid stored block lengths
                        self.mode = BlockMode::Bad;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    }
                    let left = (b as u32) & 0xffff;
                    self.sub = BlockSub::Left(left);
                    // dump bits: b = k = 0
                    self.bitb = 0;
                    self.bitk = 0;
                    self.mode = if left != 0 {
                        BlockMode::Stored
                    } else if self.last {
                        BlockMode::Dry
                    } else {
                        BlockMode::Type
                    };
                }
                BlockMode::Stored => {
                    // copying bytes from input to window
                    if inp.exhausted() {
                        return self.win.flush(z, out, r); // LEAVE
                    }
                    needout!(self.win, z, out, r);
                    let left = match self.sub {
                        BlockSub::Left(l) => l,
                        _ => return Z_STREAM_ERROR,
                    };
                    let mut t = left as usize;
                    t = t.min(inp.avail());
                    t = t.min(self.win.wavail());
                    let mut copied = 0usize;
                    while copied < t {
                        match inp.pull() {
                            Some(b) => {
                                self.win.put_byte(b);
                                copied += 1;
                            }
                            None => break,
                        }
                    }
                    let left = left - copied as u32;
                    if left != 0 {
                        self.sub = BlockSub::Left(left);
                        continue;
                    }
                    self.sub = BlockSub::Left(0);
                    self.mode = if self.last {
                        BlockMode::Dry
                    } else {
                        BlockMode::Type
                    };
                }
                BlockMode::Table => {
                    // get number of entries in dynamic table
                    needbits!(self, z, inp, out, r, 14u32);
                    let table = (self.bitb as u32) & 0x3fff;
                    if (table & 0x1f) > 29 || ((table >> 5) & 0x1f) > 29 {
                        // too many length or distance symbols
                        self.mode = BlockMode::Bad;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    }
                    let ntotal = 258 + (table & 0x1f) + ((table >> 5) & 0x1f);
                    // ZALLOC(z, t, sizeof(uInt)) for the code lengths
                    let mut blens = Vec::new();
                    if blens.try_reserve_exact(ntotal as usize).is_err() {
                        r = Z_MEM_ERROR;
                        return self.win.flush(z, out, r);
                    }
                    blens.resize(ntotal as usize, 0);
                    dumpbits!(self, 14u32);
                    self.sub = BlockSub::Trees(TreesState {
                        table,
                        index: 0,
                        blens: Some(blens),
                        bb: 0,
                        tb: None,
                    });
                    self.mode = BlockMode::Btree;
                }
                BlockMode::Btree => {
                    // get bit lengths of code-length code
                    loop {
                        let Some((table, index)) = self.trees_ref().map(|t| (t.table, t.index)) else {
                            return Z_STREAM_ERROR;
                        };
                        if index >= 4 + (table >> 10) {
                            break;
                        }
                        needbits!(self, z, inp, out, r, 3u32);
                        let val = (self.bitb as u32) & 7;
                        dumpbits!(self, 3u32);
                        let Some(border) = BORDER.get(index as usize).copied() else {
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        let Some(trees) = self.trees() else {
                            return Z_STREAM_ERROR;
                        };
                        let Some(slot) = trees
                            .blens
                            .as_mut()
                            .and_then(|b| b.get_mut(border))
                        else {
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        *slot = val;
                        trees.index = index + 1;
                    }
                    // fill the rest of the code-length table with zeroes
                    while let Some(index) = self.trees_ref().map(|t| t.index) {
                        if index >= 19 {
                            break;
                        }
                        let Some(border) = BORDER.get(index as usize).copied() else {
                            break;
                        };
                        let Some(trees) = self.trees() else {
                            return Z_STREAM_ERROR;
                        };
                        let Some(slot) = trees
                            .blens
                            .as_mut()
                            .and_then(|b| b.get_mut(border))
                        else {
                            break;
                        };
                        *slot = 0;
                        trees.index = index + 1;
                    }
                    // build the code-length decoding tree
                    // borrow only `self.sub` so `self.hufts` stays free
                    let Some(blens) = (match &self.sub {
                        BlockSub::Trees(t) => t.blens.as_deref(),
                        _ => None,
                    }) else {
                        return Z_STREAM_ERROR;
                    };
                    let mut ctx = match HuffCtx::new(&mut self.hufts) {
                        Some(c) => c,
                        None => {
                            r = Z_MEM_ERROR;
                            return self.win.flush(z, out, r);
                        }
                    };
                    let res = ctx.trees_bits(blens, 7);
                    if res.code != Z_OK {
                        r = res.code;
                        if r == Z_DATA_ERROR {
                            // ZFREE(blens) and stay in BAD
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                        }
                        return self.win.flush(z, out, r);
                    }
                    let Some(trees) = self.trees() else {
                        return Z_STREAM_ERROR;
                    };
                    trees.index = 0;
                    trees.bb = res.bb;
                    trees.tb = res.tb;
                    self.mode = BlockMode::Dtree;
                }
                BlockMode::Dtree => {
                    // get length and distance trees for dynamic block
                    loop {
                        let Some((table, index, bb, tb)) = self
                            .trees_ref()
                            .map(|t| (t.table, t.index, t.bb, t.tb))
                        else {
                            return Z_STREAM_ERROR;
                        };
                        let ntotal = 258 + (table & 0x1f) + ((table >> 5) & 0x1f);
                        if index >= ntotal {
                            break;
                        }
                        let Some(tb) = tb else {
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        needbits!(self, z, inp, out, r, bb);
                        let Some(&bmask) = INFLATE_MASK.get(bb as usize) else {
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        let idx = (self.bitb as u32) & bmask;
                        let Some(entry) = huft_at(TreeRef::Arena(tb), &self.hufts, idx) else {
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        };
                        let hb = u32::from(entry.bits);
                        let c = entry.base;
                        if c < 16 {
                            dumpbits!(self, hb);
                            let Some(trees) = self.trees() else {
                                return Z_STREAM_ERROR;
                            };
                            let Some(slot) = trees
                                .blens
                                .as_mut()
                                .and_then(|b| b.get_mut(index as usize))
                            else {
                                self.sub = BlockSub::None;
                                self.mode = BlockMode::Bad;
                                r = Z_DATA_ERROR;
                                return self.win.flush(z, out, r);
                            };
                            *slot = c;
                            trees.index = index + 1;
                        } else if (16..=18).contains(&c) {
                            // repeat code (16, 17 or 18)
                            let i = if c == 18 { 7 } else { c - 14 };
                            let mut j: u32 = if c == 18 { 11 } else { 3 };
                            needbits!(self, z, inp, out, r, hb + i);
                            dumpbits!(self, hb);
                            let Some(&imask) = INFLATE_MASK.get(i as usize) else {
                                self.sub = BlockSub::None;
                                self.mode = BlockMode::Bad;
                                r = Z_DATA_ERROR;
                                return self.win.flush(z, out, r);
                            };
                            let extra = (self.bitb as u32) & imask;
                            dumpbits!(self, i);
                            j = j.wrapping_add(extra);
                            if index as u64 + j as u64 > u64::from(ntotal) || (c == 16 && index < 1) {
                                // invalid bit length repeat
                                self.sub = BlockSub::None;
                                self.mode = BlockMode::Bad;
                                r = Z_DATA_ERROR;
                                return self.win.flush(z, out, r);
                            }
                            let start = if c == 16 { index - 1 } else { index };
                            let fill = if c == 16 {
                                let Some(trees) = self.trees_ref() else {
                                    return Z_STREAM_ERROR;
                                };
                                match trees
                                    .blens
                                    .as_ref()
                                    .and_then(|b| b.get(start as usize))
                                {
                                    Some(&v) => v,
                                    None => {
                                        self.sub = BlockSub::None;
                                        self.mode = BlockMode::Bad;
                                        r = Z_DATA_ERROR;
                                        return self.win.flush(z, out, r);
                                    }
                                }
                            } else {
                                0
                            };
                            let Some(trees) = self.trees() else {
                                return Z_STREAM_ERROR;
                            };
                            let Some(blens) = trees.blens.as_mut() else {
                                return Z_STREAM_ERROR;
                            };
                            let start = index as usize;
                            let end = start + j as usize;
                            if end > blens.len() {
                                self.sub = BlockSub::None;
                                self.mode = BlockMode::Bad;
                                r = Z_DATA_ERROR;
                                return self.win.flush(z, out, r);
                            }
                            for cell in &mut blens[start..end] {
                                *cell = fill;
                            }
                            trees.index = end as u32;
                        } else {
                            // impossible for a table built by huft_build
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                            r = Z_DATA_ERROR;
                            return self.win.flush(z, out, r);
                        }
                    }
                    // `s->sub.trees.tb = Z_NULL` before building the
                    // literal/length and distance trees
                    if let Some(trees) = self.trees() {
                        trees.tb = None;
                    }
                    // borrow only `self.sub` so `self.hufts` stays free
                    let Some((table, blens)) = (match &self.sub {
                        BlockSub::Trees(t) => t.blens.as_deref().map(|b| (t.table, b)),
                        _ => None,
                    }) else {
                        return Z_STREAM_ERROR;
                    };
                    let nl = 257 + (table & 0x1f);
                    let nd = 1 + ((table >> 5) & 0x1f);
                    let mut ctx = match HuffCtx::new(&mut self.hufts) {
                        Some(c) => c,
                        None => {
                            r = Z_MEM_ERROR;
                            return self.win.flush(z, out, r);
                        }
                    };
                    let res = ctx.trees_dynamic(nl as usize, nd as usize, blens, 9, 6);
                    if res.code != Z_OK {
                        r = res.code;
                        if r == Z_DATA_ERROR {
                            // ZFREE(blens) and stay in BAD
                            self.sub = BlockSub::None;
                            self.mode = BlockMode::Bad;
                        }
                        return self.win.flush(z, out, r);
                    }
                    let Some(tl) = res.tl else {
                        self.sub = BlockSub::None;
                        self.mode = BlockMode::Bad;
                        r = Z_DATA_ERROR;
                        return self.win.flush(z, out, r);
                    };
                    // ZFREE(blens): replaced by the decoder state
                    self.sub = BlockSub::Decode(CodesState::new(
                        res.bl,
                        res.bd,
                        TreeRef::Arena(tl),
                        res.td.map(TreeRef::Arena),
                    ));
                    self.mode = BlockMode::Codes;
                }
                BlockMode::Codes => {
                    // UPDATE is implicit, then run the block decoder
                    let rr = self.inflate_codes(z, r, inp, out);
                    if rr != Z_STREAM_END {
                        return self.win.flush(z, out, rr);
                    }
                    r = Z_OK;
                    // inflate_codes_free: the decoder state is dropped
                    self.sub = BlockSub::None;
                    if !self.last {
                        self.mode = BlockMode::Type;
                        continue;
                    }
                    self.mode = BlockMode::Dry;
                }
                BlockMode::Dry => {
                    // output remaining window bytes
                    r = self.win.flush(z, out, r); // FLUSH
                    if self.win.read != self.win.write {
                        return self.win.flush(z, out, r); // LEAVE
                    }
                    self.mode = BlockMode::Done;
                }
                BlockMode::Done => {
                    // finished last block, done
                    r = Z_STREAM_END;
                    return self.win.flush(z, out, r);
                }
                BlockMode::Bad => {
                    // got a data error -- stay here
                    r = Z_DATA_ERROR;
                    return self.win.flush(z, out, r);
                }
            }
        }
    }
}

/// zlib's `NEEDBYTE` at the wrapper layer (inflate.c): returns `r` when the
/// input is exhausted, otherwise continues with `r = f`. Like [`needbits!`]
/// it may expand to a `return`.
macro_rules! needbyte {
    ($inp:expr, $r:expr, $f:expr) => {
        if $inp.exhausted() {
            return $r;
        }
        $r = $f;
    };
}

impl Inflater {
    /// Port of `inflate` (inflate.c). `f` is the flush flag: `Z_NO_FLUSH`
    /// while streaming (gzip refill) or `Z_FINISH` for one-shot use; C maps
    /// it to `Z_BUF_ERROR`/`Z_OK` internally, as done here.
    ///
    /// Input consumed during the call is folded into `self.z.total_in`
    /// before returning (C's `UPDIN`, batched once per call).
    fn inflate(&mut self, f: i32, inp: &mut Inp<'_>, out: &mut OutBuf<'_>) -> i32 {
        let r = self.inflate_run(f, inp, out);
        inp.take_consumed(&mut self.z);
        r
    }

    /// The body of `inflate` (inflate.c's `while (1) switch (mode)`).
    /// Every arm either makes progress or returns; an early return keeps
    /// all state in the fields, so the next call resumes exactly there.
    ///
    /// The `r = f` assignments of `NEEDBYTE` are sometimes dead (C keeps
    /// them for uniformity), hence the allow.
    #[allow(unused_assignments)]
    fn inflate_run(&mut self, f: i32, inp: &mut Inp<'_>, out: &mut OutBuf<'_>) -> i32 {
        // f = f == Z_FINISH ? Z_BUF_ERROR : Z_OK;
        let f = if f == Z_FINISH { Z_BUF_ERROR } else { Z_OK };
        // r = Z_BUF_ERROR; (initial return status)
        let mut r = Z_BUF_ERROR;
        loop {
            match self.mode {
                InflMode::Method => {
                    // waiting for method byte
                    needbyte!(inp, r, f);
                    let Some(method) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    if method & 0x0f != Z_DEFLATED {
                        self.mode = InflMode::Bad;
                        continue;
                    }
                    if u32::from(method >> 4) + 8 > self.wbits {
                        self.mode = InflMode::Bad;
                        continue;
                    }
                    self.method = method;
                    self.mode = InflMode::Flag;
                }
                InflMode::Flag => {
                    // waiting for flag byte
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    let b = u32::from(b);
                    if (u32::from(self.method) * 256 + b) % 31 != 0 {
                        self.mode = InflMode::Bad;
                        continue;
                    }
                    if b & u32::from(PRESET_DICT) == 0 {
                        self.mode = InflMode::Blocks;
                        continue;
                    }
                    self.mode = InflMode::Dict4;
                }
                InflMode::Dict4 => {
                    // four dictionary check bytes to go
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need = u64::from(b) << 24;
                    self.mode = InflMode::Dict3;
                }
                InflMode::Dict3 => {
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need += u64::from(b) << 16;
                    self.mode = InflMode::Dict2;
                }
                InflMode::Dict2 => {
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need += u64::from(b) << 8;
                    self.mode = InflMode::Dict1;
                }
                InflMode::Dict1 => {
                    // one dictionary check byte to go, then hand the
                    // dictionary id to the caller (never settable here)
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need += u64::from(b);
                    self.z.adler = self.check_need as u32;
                    self.mode = InflMode::Dict0;
                    return Z_NEED_DICT;
                }
                InflMode::Dict0 => {
                    // `inflateSetDictionary` was never called
                    self.mode = InflMode::Bad;
                    return Z_STREAM_ERROR;
                }
                InflMode::Blocks => {
                    r = self
                        .blocks
                        .inflate_blocks(&mut self.z, r, inp, out);
                    if r == Z_DATA_ERROR {
                        self.mode = InflMode::Bad;
                        continue;
                    }
                    if r == Z_OK {
                        r = f;
                    }
                    if r != Z_STREAM_END {
                        return r;
                    }
                    r = f;
                    self.check_was = u64::from(self.blocks.reset(&mut self.z));
                    if self.nowrap {
                        self.mode = InflMode::Done;
                        continue;
                    }
                    self.mode = InflMode::Check4;
                }
                InflMode::Check4 => {
                    // four check bytes to go (adler32 of the zlib stream)
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need = u64::from(b) << 24;
                    self.mode = InflMode::Check3;
                }
                InflMode::Check3 => {
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need += u64::from(b) << 16;
                    self.mode = InflMode::Check2;
                }
                InflMode::Check2 => {
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need += u64::from(b) << 8;
                    self.mode = InflMode::Check1;
                }
                InflMode::Check1 => {
                    needbyte!(inp, r, f);
                    let Some(b) = inp.pull() else {
                        return Z_STREAM_ERROR;
                    };
                    self.check_need += u64::from(b);
                    if self.check_was != self.check_need {
                        self.mode = InflMode::Bad;
                        continue;
                    }
                    self.mode = InflMode::Done;
                }
                InflMode::Done => {
                    // finished check, done
                    return Z_STREAM_END;
                }
                InflMode::Bad => {
                    // got a data error -- stay here
                    return Z_DATA_ERROR;
                }
            }
        }
    }
}

impl<'a> Inp<'a> {
    /// Resumes a persisted cursor: C keeps `avail_in` / `next_in` across
    /// `ft_gzip_file_fill_output` calls, so the input cursor of an
    /// interrupted batch must survive into the next one.
    fn resume(kind: InpKind<'a>, idx: i64, pre: Option<u8>, len: usize) -> Inp<'a> {
        Inp {
            kind,
            consumed: 0,
            idx,
            pre,
            len,
        }
    }
}

/// Reads exactly `buf.len()` bytes at `*pos` and advances `*pos`.
/// A short read (end of input) maps to [`TtError::INVALID_STREAM_READ`].
fn read_exact_at(src: &dyn StreamSource, pos: &mut u64, buf: &mut [u8]) -> TtResult<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let k = src.read_at(*pos + done as u64, &mut buf[done..])?;
        if k == 0 || k > buf.len() - done {
            return Err(TtError::INVALID_STREAM_READ);
        }
        done += k;
    }
    *pos += buf.len() as u64;
    Ok(())
}

/// Skips `n` bytes of header payload (FEXTRA) by reading and discarding.
fn skip_bytes(src: &dyn StreamSource, pos: &mut u64, mut n: u64) -> TtResult<()> {
    let mut chunk = [0u8; GZIP_MAGIC_SCAN];
    while n > 0 {
        let k = n.min(GZIP_MAGIC_SCAN as u64) as usize;
        read_exact_at(src, pos, &mut chunk[..k])?;
        n -= k as u64;
    }
    Ok(())
}

/// Skips a NUL-terminated string (FNAME / FCOMMENT).
fn skip_cstr(src: &dyn StreamSource, pos: &mut u64) -> TtResult<()> {
    let mut b = [0u8; 1];
    loop {
        read_exact_at(src, pos, &mut b)?;
        if b[0] == 0 {
            return Ok(());
        }
    }
}

/// Port of `ft_gzip_check_header`: validates and skips the gzip member
/// header, returning the offset of the first deflate byte (C's
/// `zip->start`).
///
/// Deviation: C reads the header at offset 0 only; here the first
/// [`GZIP_MAGIC_SCAN`] bytes are scanned for the `1F 8B` magic so members
/// with leading garbage (RFC 1952 concatenation) still open. Invalid
/// method / reserved flag bits report [`TtError::INVALID_FILE_FORMAT`],
/// a truncated header reports [`TtError::INVALID_STREAM_READ`].
fn check_header(src: &dyn StreamSource) -> TtResult<u64> {
    let mut head = [0u8; GZIP_MAGIC_SCAN];
    let n = src.read_at(0, &mut head)?.min(GZIP_MAGIC_SCAN);

    // find the two-byte magic inside the scanned prefix
    let mut i = 0usize;
    let mut found = false;
    while i + 1 < n {
        if head[i] == 0x1F && head[i + 1] == 0x8B {
            found = true;
            break;
        }
        i += 1;
    }
    if !found {
        return Err(TtError::INVALID_FILE_FORMAT);
    }

    let mut pos = i as u64;
    let mut hdr = [0u8; 4];
    read_exact_at(src, &mut pos, &mut hdr)?;
    if hdr[0] != 0x1F || hdr[1] != 0x8B || hdr[2] != Z_DEFLATED || (hdr[3] & FT_GZIP_RESERVED) != 0 {
        return Err(TtError::INVALID_FILE_FORMAT);
    }
    let flags = hdr[3];

    // skip mtime, xfl and os
    let mut rest = [0u8; 6];
    read_exact_at(src, &mut pos, &mut rest)?;

    if flags & FT_GZIP_EXTRA_FIELD != 0 {
        let mut le = [0u8; 2];
        read_exact_at(src, &mut pos, &mut le)?;
        let xlen = u64::from(u16::from_le_bytes([le[0], le[1]]));
        skip_bytes(src, &mut pos, xlen)?;
    }
    if flags & FT_GZIP_ORIG_NAME != 0 {
        skip_cstr(src, &mut pos)?;
    }
    if flags & FT_GZIP_COMMENT != 0 {
        skip_cstr(src, &mut pos)?;
    }
    if flags & FT_GZIP_HEAD_CRC != 0 {
        let mut hcrc = [0u8; 2];
        read_exact_at(src, &mut pos, &mut hcrc)?;
    }
    Ok(pos)
}

/// Appends the freshly decompressed `bytes` (stream offset `base`) to the
/// 32 KiB history ring used for backward seeks.
fn append_hist(hist: &mut [u8], base: u64, bytes: &[u8]) {
    if hist.is_empty() || bytes.is_empty() {
        return;
    }
    let hlen = hist.len();
    let start = (base % hlen as u64) as usize;
    let first = bytes.len().min(hlen - start);
    hist[start..start + first].copy_from_slice(&bytes[..first]);
    let rest = &bytes[first..];
    if !rest.is_empty() {
        let n2 = rest.len().min(hlen);
        hist[..n2].copy_from_slice(&rest[..n2]);
    }
}

/// Serves decompressed bytes at stream offset `pos` from the history ring.
/// Returns the number of bytes copied (0 when `pos` is outside
/// `[hist_start, produced)`).
fn hist_read(hist: &[u8], hist_start: u64, produced: u64, pos: u64, buf: &mut [u8]) -> usize {
    if hist.is_empty() || buf.is_empty() || pos < hist_start || pos >= produced {
        return 0;
    }
    let hlen = hist.len() as u64;
    // `pos >= hist_start` keeps `produced - pos` within one ring length
    let avail = (produced - pos).min(hlen);
    let n = (avail as usize).min(buf.len());
    let start = (pos % hlen) as usize;
    let first = n.min(hist.len() - start);
    buf[..first].copy_from_slice(&hist[start..start + first]);
    if first < n {
        buf[first..n].copy_from_slice(&hist[..n - first]);
    }
    n
}

/// The decompression side of FreeType's `FT_GZipFileRec`: two 4 KiB
/// buffers, the source read position, the history ring and the raw
/// inflate state. All mutation happens under `GzipSource`'s lock.
struct GzipFileState {
    /// output batch (C's `buffer` / `cursor` / `limit`)
    buffer: Vec<u8>,
    /// input refill buffer (C's `input`)
    input: Vec<u8>,
    /// read position inside the compressed source (C's `stream->pos`)
    src_pos: u64,
    /// decompressed bytes produced so far (stream offset)
    produced: u64,
    /// last 32 KiB of decompressed data for backward seeks
    hist: Vec<u8>,
    /// persisted cursor into `input` (see [`Inp::resume`])
    idx: i64,
    /// persisted `WASH` byte of the input cursor
    pre: Option<u8>,
    /// persisted valid length of `input`
    len: usize,
    /// raw inflate state (`inflateInit2(-MAX_WBITS)`)
    inf: Inflater,
}

impl GzipFileState {
    /// Creates the state positioned at `start` (after the gzip header).
    /// Fails with [`TtError::INVALID_FILE_FORMAT`] when zlib refuses the
    /// window size (C's `ft_gzip_file_init`) or [`TtError::OUT_OF_MEMORY`].
    fn new(start: u64) -> TtResult<GzipFileState> {
        let inf = Inflater::new(true, MAX_WBITS).ok_or(TtError::INVALID_FILE_FORMAT)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(FT_GZIP_BUFFER_SIZE)
            .map_err(|_| TtError::OUT_OF_MEMORY)?;
        buffer.resize(FT_GZIP_BUFFER_SIZE, 0);
        let mut input = Vec::new();
        input
            .try_reserve_exact(FT_GZIP_BUFFER_SIZE)
            .map_err(|_| TtError::OUT_OF_MEMORY)?;
        input.resize(FT_GZIP_BUFFER_SIZE, 0);
        let mut hist = Vec::new();
        hist.try_reserve_exact(HISTORY_SIZE as usize)
            .map_err(|_| TtError::OUT_OF_MEMORY)?;
        hist.resize(HISTORY_SIZE as usize, 0);
        Ok(GzipFileState {
            buffer,
            input,
            src_pos: start,
            produced: 0,
            hist,
            idx: 0,
            pre: None,
            len: 0,
            inf,
        })
    }

    /// Port of `ft_gzip_file_reset`: rewinds to `start` and restarts the
    /// inflater (the history ring becomes unreachable once `produced` is 0).
    fn reset(&mut self, start: u64) {
        self.src_pos = start;
        self.produced = 0;
        self.idx = 0;
        self.pre = None;
        self.len = 0;
        self.inf.reset();
    }

    /// Port of `ft_gzip_file_fill_output`: inflates one batch of up to
    /// [`FT_GZIP_BUFFER_SIZE`] bytes into `buffer` and appends it to the
    /// history ring. Returns the number of bytes produced (0 at end of
    /// stream with no data left).
    ///
    /// Like C, an error discards the whole batch; `Z_STREAM_END` with an
    /// empty batch yields `Ok(0)` (C reports an internal error there,
    /// which `ft_gzip_file_io` swallows into a short read).
    fn fill_output(&mut self, src: &dyn StreamSource) -> TtResult<usize> {
        let mut error: Option<TtError> = None;
        let n;
        {
            let GzipFileState {
                buffer,
                input,
                src_pos,
                inf,
                idx,
                pre,
                len,
                ..
            } = self;
            let mut inp = Inp::resume(
                InpKind::Source {
                    src,
                    pos: src_pos,
                    buf: input,
                },
                *idx,
                *pre,
                *len,
            );
            let mut out = OutBuf::new(buffer);
            loop {
                if out.avail() == 0 {
                    break;
                }
                if inp.exhausted()
                    && let Err(e) = inp.fill()
                {
                    error = Some(e);
                    break;
                }
                let err = inf.inflate(Z_NO_FLUSH, &mut inp, &mut out);
                if err == Z_STREAM_END {
                    break;
                }
                if err != Z_OK {
                    error = Some(TtError::INVALID_STREAM_OPERATION);
                    break;
                }
                // inflate only returns Z_OK for "out full" or "input
                // exhausted"; anything else would spin forever
                if out.avail() > 0 && !inp.exhausted() {
                    error = Some(TtError::INVALID_STREAM_OPERATION);
                    break;
                }
            }
            // persist the input cursor for the next batch
            *idx = inp.idx;
            *pre = inp.pre;
            *len = inp.len;
            n = if error.is_some() { 0 } else { out.written().len() };
        }
        if let Some(e) = error {
            return Err(e);
        }
        if n > 0 {
            append_hist(&mut self.hist, self.produced, &self.buffer[..n]);
            self.produced += n as u64;
        }
        Ok(n)
    }
}

/// Port of `FT_Stream_OpenGzip` + `FT_Gzip_File` over a shared compressed
/// [`StreamSource`].
///
/// Differences from C: the decompressed size is measured eagerly in
/// [`GzipSource::new`] (FT trusts the ISIZE trailer and falls back to
/// `0x7FFFFFFF`), a corrupt stream therefore fails at construction time,
/// and `size()` always reports the exact length. The small-file
/// optimization of FT (pre-load below 40 KiB) is kept: such files are
/// served lock-free from memory.
pub struct GzipSource {
    /// the compressed source
    inner: Arc<dyn StreamSource>,
    /// exact decompressed size, measured eagerly
    total_size: u64,
    /// first deflate byte (after the gzip header)
    data_start: u64,
    /// whole decompressed image of a small file
    small: Option<Vec<u8>>,
    /// streaming state; `None` once a small file was pre-loaded
    state: Mutex<Option<GzipFileState>>,
}

impl GzipSource {
    /// Wraps `source`, which must begin (after at most
    /// [`GZIP_MAGIC_SCAN`] leading bytes) with a valid gzip member.
    ///
    /// Errors: [`TtError::INVALID_FILE_FORMAT`] for a bad header,
    /// [`TtError::INVALID_STREAM_READ`] for a truncated header,
    /// [`TtError::INVALID_STREAM_OPERATION`] for a corrupt or truncated
    /// deflate stream, [`TtError::OUT_OF_MEMORY`] for allocation failure.
    pub fn new(source: Arc<dyn StreamSource>) -> TtResult<GzipSource> {
        let data_start = check_header(source.as_ref())?;
        let mut st = GzipFileState::new(data_start)?;

        // eager measurement (C has no equivalent pass)
        while !st.inf.at_stream_end() {
            let before = st.produced;
            st.fill_output(source.as_ref())?;
            if st.produced == before && !st.inf.at_stream_end() {
                return Err(TtError::INVALID_STREAM_OPERATION);
            }
        }
        let total_size = st.produced;
        st.reset(data_start);

        // FT's small-file trick: keep the whole image in memory
        let small = if total_size > 0 && total_size < SMALL_FILE_LIMIT {
            let mut image = Vec::new();
            image
                .try_reserve_exact(total_size as usize)
                .map_err(|_| TtError::OUT_OF_MEMORY)?;
            while st.produced < total_size {
                let before = st.produced;
                let n = st.fill_output(source.as_ref())?;
                let part = st
                    .buffer
                    .get(..n)
                    .ok_or(TtError::INVALID_STREAM_OPERATION)?;
                image.extend_from_slice(part);
                if st.produced == before {
                    break;
                }
            }
            if image.len() as u64 != total_size {
                return Err(TtError::INVALID_STREAM_OPERATION);
            }
            Some(image)
        } else {
            None
        };

        let state = if small.is_some() { None } else { Some(st) };
        Ok(GzipSource {
            inner: source,
            total_size,
            data_start,
            small,
            state: Mutex::new(state),
        })
    }
}

impl StreamSource for GzipSource {
    fn size(&self) -> Option<u64> {
        Some(self.total_size)
    }

    fn read_at(&self, pos: u64, buf: &mut [u8]) -> TtResult<usize> {
        if buf.is_empty() || pos >= self.total_size {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(self.total_size - pos) as usize;

        if let Some(small) = &self.small {
            let start = pos as usize;
            if start >= small.len() {
                return Ok(0);
            }
            let n = want.min(small.len() - start);
            if let Some(dst) = buf.get_mut(..n)
                && let Some(src) = small.get(start..start + n)
            {
                dst.copy_from_slice(src);
                return Ok(n);
            }
            return Ok(0);
        }

        let mut guard = self.state.lock();
        let Some(st) = guard.as_mut() else {
            // unreachable: the state exists exactly when `small` is None
            return Err(TtError::INVALID_STREAM_OPERATION);
        };
        let src: &dyn StreamSource = self.inner.as_ref();

        let mut filled = 0usize;
        let mut cur = pos;
        while filled < want {
            let hist_start = st.produced.saturating_sub(HISTORY_SIZE);
            if cur < hist_start {
                // seek is behind the ring: restart and skip forward again
                st.reset(self.data_start);
                continue;
            }
            if cur > st.produced {
                // forward seek: inflate and discard whole batches
                let before = st.produced;
                let _ = st.fill_output(src);
                if st.produced == before {
                    return Ok(filled); // end of stream while skipping
                }
                continue;
            }
            // `cur` is inside the ring window
            let n = hist_read(&st.hist, hist_start, st.produced, cur, &mut buf[filled..want]);
            if n == 0 {
                // `cur == produced`: need more data (or reached the end)
                if st.inf.at_stream_end() {
                    return Ok(filled);
                }
                let before = st.produced;
                let _ = st.fill_output(src);
                if st.produced == before {
                    return Ok(filled); // truncated tail: C returns short
                }
                continue;
            }
            filled += n;
            cur += n as u64;
        }
        Ok(filled)
    }
}

/// Port of `FT_Gzip_Uncompress` (zlib-wrapped raw DEFLATE in one shot):
/// inflates `src` into `dst` and returns the number of bytes written.
///
/// Error mapping follows FT: init failure [`TtError::INVALID_ARGUMENT`],
/// `Z_MEM_ERROR` [`TtError::OUT_OF_MEMORY`], `Z_BUF_ERROR` (output too
/// small or input exhausted under `Z_FINISH`) [`TtError::ARRAY_TOO_LARGE`],
/// `Z_DATA_ERROR` [`TtError::INVALID_TABLE`]. Deviation: the remaining
/// codes (`Z_STREAM_ERROR`, `Z_NEED_DICT`) report
/// [`TtError::INVALID_STREAM_OPERATION`] instead of FT's (buggy) `Ok`.
pub fn uncompress(dst: &mut [u8], src: &[u8]) -> TtResult<usize> {
    let mut inf = Inflater::new(false, MAX_WBITS).ok_or(TtError::INVALID_ARGUMENT)?;
    let mut inp = Inp::one_shot(src);
    let mut out = OutBuf::new(dst);
    let r = inf.inflate(Z_FINISH, &mut inp, &mut out);
    let n = out.written().len();
    match r {
        Z_STREAM_END => Ok(n),
        Z_OK | Z_BUF_ERROR => Err(TtError::ARRAY_TOO_LARGE),
        Z_MEM_ERROR => Err(TtError::OUT_OF_MEMORY),
        Z_DATA_ERROR => Err(TtError::INVALID_TABLE),
        _ => Err(TtError::INVALID_STREAM_OPERATION),
    }
}

/// Sniffs a gzip member: magic `1F 8B`, deflate method and no reserved
/// flag bits (the first test of `ft_gzip_check_header`).
pub fn is_gzip(buf: &[u8]) -> bool {
    let Some(&[m0, m1, m2, m3]) = buf.get(..4) else {
        return false;
    };
    m0 == 0x1F && m1 == 0x8B && m2 == Z_DEFLATED && (m3 & FT_GZIP_RESERVED) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known plaintext of `SMALL_GZ` / `SMALL_ZLIB`.
    const SMALL_PLAIN: [u8; 191] = [
        67, 111, 100, 101, 118, 97, 114, 32, 103, 122, 105, 112, 32, 102, 105, 120, 116, 117, 114, 101, 10,
        84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 32, 106,
        117, 109, 112, 115, 32, 111, 118, 101, 114, 32, 116, 104, 101, 32, 108, 97, 122, 121, 32, 100, 111,
        103, 46, 10, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 32, 97, 98, 99, 100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 32, 65, 66, 67,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 10, 115,
        101, 99, 111, 110, 100, 32, 108, 105, 110, 101, 32, 119, 105, 116, 104, 32, 116, 114, 97, 105, 108,
        105, 110, 103, 32, 115, 112, 97, 99, 101, 115, 32, 32, 10, 108, 97, 115, 116, 32, 108, 105, 110, 101,
        32, 119, 105, 116, 104, 111, 117, 116, 32, 110, 101, 119, 108, 105, 110, 101,
    ];

    /// gzip member of `SMALL_PLAIN`.
    const SMALL_GZ: [u8; 179] = [
        31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 69, 202, 71, 22, 130, 48, 24, 69, 225, 57, 171, 120, 43, 240, 216,
        203, 80, 177, 247, 222, 102, 17, 126, 32, 138, 9, 166, 0, 178, 122, 117, 228, 240, 59, 247, 186, 210,
        167, 148, 41, 132, 5, 79, 16, 240, 220, 88, 69, 206, 62, 34, 188, 44, 247, 30, 184, 41, 153, 9, 4,
        50, 199, 221, 62, 19, 13, 153, 146, 130, 249, 230, 152, 21, 111, 248, 50, 44, 57, 229, 74, 181, 86,
        111, 52, 91, 237, 14, 216, 205, 243, 41, 8, 35, 126, 127, 196, 79, 33, 147, 151, 210, 198, 166, 89,
        254, 46, 208, 237, 185, 253, 193, 112, 52, 158, 76, 103, 243, 197, 114, 181, 222, 108, 119, 251, 195,
        241, 116, 190, 92, 29, 77, 158, 20, 62, 98, 46, 8, 25, 55, 17, 140, 98, 252, 171, 16, 58, 97, 30,
        105, 192, 137, 153, 54, 255, 65, 90, 3, 65, 217, 207, 31, 229, 199, 100, 124, 191, 0, 0, 0,
    ];

    /// zlib-wrapped DEFLATE of `SMALL_PLAIN` for `uncompress`.
    const SMALL_ZLIB: [u8; 167] = [
        120, 218, 69, 202, 71, 22, 130, 48, 24, 69, 225, 57, 171, 120, 43, 240, 216, 203, 80, 177, 247, 222,
        102, 17, 126, 32, 138, 9, 166, 0, 178, 122, 117, 228, 240, 59, 247, 186, 210, 167, 148, 41, 132, 5,
        79, 16, 240, 220, 88, 69, 206, 62, 34, 188, 44, 247, 30, 184, 41, 153, 9, 4, 50, 199, 221, 62, 19,
        13, 153, 146, 130, 249, 230, 152, 21, 111, 248, 50, 44, 57, 229, 74, 181, 86, 111, 52, 91, 237, 14,
        216, 205, 243, 41, 8, 35, 126, 127, 196, 79, 33, 147, 151, 210, 198, 166, 89, 254, 46, 208, 237, 185,
        253, 193, 112, 52, 158, 76, 103, 243, 197, 114, 181, 222, 108, 119, 251, 195, 241, 116, 190, 92, 29,
        77, 158, 20, 62, 98, 46, 8, 25, 55, 17, 140, 98, 252, 171, 16, 58, 97, 30, 105, 192, 137, 153, 54,
        255, 65, 90, 3, 65, 217, 207, 31, 58, 250, 67, 123,
    ];

    /// gzip member of the 70000-byte `i % 251` pattern (`BIG_LEN`).
    const BIG_GZ: [u8; 609] = [
        31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 237, 207, 67, 130, 16, 0, 0, 0, 192, 205, 182, 185, 217, 182, 109,
        219, 182, 109, 215, 102, 219, 182, 109, 219, 182, 109, 91, 167, 94, 209, 109, 230, 7, 19, 16, 44,
        120, 136, 144, 161, 66, 135, 9, 27, 46, 124, 132, 136, 145, 34, 71, 137, 26, 45, 122, 140, 152, 177,
        98, 199, 137, 27, 47, 126, 130, 132, 137, 18, 7, 38, 73, 154, 44, 121, 138, 148, 169, 82, 167, 73,
        155, 46, 125, 134, 140, 153, 50, 103, 201, 154, 45, 123, 142, 156, 185, 114, 231, 201, 155, 47, 127,
        129, 130, 133, 10, 23, 41, 90, 172, 120, 137, 146, 165, 74, 151, 41, 91, 174, 124, 133, 138, 149, 42,
        87, 169, 90, 173, 122, 141, 154, 181, 106, 215, 169, 91, 175, 126, 131, 134, 141, 26, 55, 105, 218,
        172, 121, 139, 150, 173, 90, 183, 105, 219, 174, 125, 135, 142, 157, 58, 119, 233, 218, 173, 123,
        143, 158, 189, 122, 247, 233, 219, 175, 255, 128, 129, 131, 6, 15, 25, 58, 44, 104, 248, 136, 145,
        163, 70, 143, 25, 59, 110, 252, 132, 137, 147, 38, 79, 153, 58, 109, 250, 140, 153, 179, 102, 207,
        153, 59, 111, 254, 130, 133, 139, 22, 47, 89, 186, 108, 249, 138, 149, 171, 86, 175, 89, 187, 110,
        253, 134, 141, 155, 54, 111, 217, 186, 109, 251, 142, 157, 187, 118, 239, 217, 187, 111, 255, 129,
        131, 135, 14, 31, 57, 122, 236, 248, 137, 147, 167, 78, 159, 57, 123, 238, 252, 133, 139, 151, 46,
        95, 185, 122, 237, 250, 141, 155, 183, 110, 223, 185, 123, 239, 254, 131, 135, 143, 30, 63, 121, 250,
        236, 249, 139, 151, 175, 94, 191, 121, 251, 238, 253, 135, 143, 159, 62, 127, 249, 250, 237, 251,
        143, 159, 191, 126, 255, 249, 27, 160, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
        174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 254, 191, 234,
        255, 0, 193, 199, 225, 159, 112, 17, 1, 0,
    ];

    /// gzip member of an empty payload.
    const EMPTY_GZ: [u8; 20] = [31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    /// gzip member with FTEXT|FHCRC|FEXTRA|FNAME|FCOMMENT all set.
    const FLAGS_GZ: [u8; 133] = [
        31, 139, 8, 31, 135, 214, 18, 0, 3, 0, 8, 0, 69, 88, 4, 0, 65, 66, 67, 68, 102, 105, 120, 116, 117,
        114, 101, 45, 110, 97, 109, 101, 46, 103, 122, 0, 97, 32, 99, 111, 109, 109, 101, 110, 116, 32, 97,
        98, 111, 117, 116, 32, 116, 104, 105, 115, 32, 102, 105, 120, 116, 117, 114, 101, 0, 77, 225, 75,
        175, 202, 44, 80, 200, 72, 77, 76, 73, 45, 82, 40, 201, 47, 42, 41, 45, 74, 85, 72, 203, 172, 0, 209,
        86, 10, 169, 21, 37, 69, 137, 58, 10, 121, 137, 185, 169, 58, 10, 201, 249, 185, 185, 169, 121, 37,
        10, 137, 121, 41, 10, 25, 201, 69, 201, 92, 0, 20, 135, 87, 233, 59, 0, 0, 0,
    ];

    /// Known plaintext of `FLAGS_GZ`.
    const FLAGS_PLAIN: [u8; 59] = [
        103, 122, 105, 112, 32, 104, 101, 97, 100, 101, 114, 32, 116, 111, 114, 116, 117, 114, 101, 32, 102,
        105, 120, 116, 117, 114, 101, 58, 32, 101, 120, 116, 114, 97, 44, 32, 110, 97, 109, 101, 44, 32, 99,
        111, 109, 109, 101, 110, 116, 32, 97, 110, 100, 32, 104, 99, 114, 99, 10,
    ];

    /// zlib header plus a stored block with an invalid NLEN (bad data).
    const CORRUPT_ZLIB: [u8; 7] = [0x78, 0x9C, 0x01, 0x01, 0x00, 0x00, 0x00];

    /// Decompressed length of [`BIG_GZ`].
    const BIG_LEN: usize = 70_000;

    /// Expected `adler32(1, PATTERN_20K)` (Python's `zlib.adler32`).
    const ADLER_PATTERN_20K: u32 = 0x4414_0D23;

    /// A simple in-memory [`StreamSource`] for the tests.
    struct MemSource(Vec<u8>);

    impl StreamSource for MemSource {
        fn size(&self) -> Option<u64> {
            Some(self.0.len() as u64)
        }

        fn read_at(&self, pos: u64, buf: &mut [u8]) -> TtResult<usize> {
            let len = self.0.len() as u64;
            if pos >= len || buf.is_empty() {
                return Ok(0);
            }
            let n = ((len - pos) as usize).min(buf.len());
            let start = pos as usize;
            buf[..n].copy_from_slice(&self.0[start..start + n]);
            Ok(n)
        }
    }

    /// Boxes `data` as a shared source.
    fn mem(data: &[u8]) -> Arc<dyn StreamSource> {
        Arc::new(MemSource(data.to_vec()))
    }

    /// The decompressed pattern of [`BIG_GZ`] (period 251 is coprime to
    /// the 32 KiB history size, so ring-position bugs stay visible).
    fn pattern_byte(i: usize) -> u8 {
        (i % 251) as u8
    }

    #[test]
    fn adler32_matches_reference() {
        // known vector: adler32("abc") == 0x024D0127
        assert_eq!(adler32(adler32_init(), b"abc"), 0x024D_0127);
        assert_eq!(adler32(1, b""), 1);
        // reference value of a buffer larger than ADLER_NMAX (5552)
        let data: Vec<u8> = (0..20_000).map(pattern_byte).collect();
        assert_eq!(adler32(1, &data), ADLER_PATTERN_20K);
        // chunked accumulation must agree with the single pass
        let mut acc = 1u32;
        for chunk in data.chunks(1000) {
            acc = adler32(acc, chunk);
        }
        assert_eq!(acc, ADLER_PATTERN_20K);
    }

    #[test]
    fn is_gzip_sniffs_header() {
        assert!(is_gzip(&SMALL_GZ));
        assert!(is_gzip(&EMPTY_GZ));
        assert!(is_gzip(&[0x1F, 0x8B, 0x08, 0x00]));
        // too short
        assert!(!is_gzip(&[0x1F, 0x8B]));
        assert!(!is_gzip(b""));
        // reserved flag bits
        assert!(!is_gzip(&[0x1F, 0x8B, 0x08, 0xE0]));
        // wrong method
        assert!(!is_gzip(&[0x1F, 0x8B, 0x07, 0x00]));
        // no magic at all
        assert!(!is_gzip(b"no gzip magic here"));
    }

    #[test]
    fn uncompress_roundtrips_small_text() {
        let mut dst = vec![0u8; SMALL_PLAIN.len()];
        let n = uncompress(&mut dst, &SMALL_ZLIB).expect("valid zlib stream");
        assert_eq!(n, SMALL_PLAIN.len());
        assert_eq!(&dst[..n], &SMALL_PLAIN[..]);
        // a larger destination stays partly unwritten but succeeds
        let mut big = vec![0u8; SMALL_PLAIN.len() + 64];
        let n = uncompress(&mut big, &SMALL_ZLIB).expect("valid zlib stream");
        assert_eq!(n, SMALL_PLAIN.len());
        assert_eq!(&big[..n], &SMALL_PLAIN[..]);
    }

    #[test]
    fn uncompress_reports_destination_too_small() {
        let mut dst = vec![0u8; SMALL_PLAIN.len() - 1];
        let r = uncompress(&mut dst, &SMALL_ZLIB);
        assert_eq!(r.err(), Some(TtError::ARRAY_TOO_LARGE));
        // empty destination can never hold the output
        let mut none = [];
        let r = uncompress(&mut none, &SMALL_ZLIB);
        assert_eq!(r.err(), Some(TtError::ARRAY_TOO_LARGE));
    }

    #[test]
    fn uncompress_rejects_invalid_streams() {
        // stored block with a broken NLEN field
        let mut dst = vec![0u8; 64];
        let r = uncompress(&mut dst, &CORRUPT_ZLIB);
        assert_eq!(r.err(), Some(TtError::INVALID_TABLE));
        // no input at all: Z_FINISH reports Z_BUF_ERROR
        let r = uncompress(&mut dst, &[]);
        assert_eq!(r.err(), Some(TtError::ARRAY_TOO_LARGE));
        // gzip members are not zlib streams
        let r = uncompress(&mut dst, &SMALL_GZ);
        assert_eq!(r.err(), Some(TtError::INVALID_TABLE));
    }

    #[test]
    fn gzip_source_roundtrips_small_file() {
        let gz = GzipSource::new(mem(&SMALL_GZ)).expect("valid gzip member");
        assert_eq!(gz.size(), Some(SMALL_PLAIN.len() as u64));

        // whole file (buffer larger than the payload: partial read)
        let mut buf = vec![0u8; SMALL_PLAIN.len() + 10];
        let n = gz.read_at(0, &mut buf).expect("read");
        assert_eq!(n, SMALL_PLAIN.len());
        assert_eq!(&buf[..n], &SMALL_PLAIN[..]);

        // interior read
        let mut part = [0u8; 10];
        let n = gz.read_at(7, &mut part).expect("read");
        assert_eq!(n, 10);
        assert_eq!(&part[..], &SMALL_PLAIN[7..17]);

        // past the end
        assert_eq!(
            gz.read_at(SMALL_PLAIN.len() as u64, &mut part)
                .expect("eof"),
            0
        );
        assert_eq!(gz.read_at(u64::MAX, &mut part).expect("eof"), 0);
    }

    #[test]
    fn gzip_source_random_access_across_window_boundaries() {
        let expected: Vec<u8> = (0..BIG_LEN).map(pattern_byte).collect();
        let gz = GzipSource::new(mem(&BIG_GZ)).expect("valid gzip member");
        assert_eq!(gz.size(), Some(BIG_LEN as u64));

        // sequential scan in odd-sized chunks (crosses 32 KiB repeatedly)
        let mut out = Vec::new();
        let mut pos = 0u64;
        while pos < BIG_LEN as u64 {
            let mut buf = [0u8; 1000];
            let n = gz
                .read_at(pos, &mut buf)
                .expect("sequential read");
            assert!(n > 0, "stalled at {pos}");
            out.extend_from_slice(&buf[..n]);
            pos += n as u64;
        }
        assert_eq!(out, expected);

        // single reads spanning the 32 KiB window boundary
        for base in [32_766usize, 65_534] {
            let mut buf = [0u8; 8];
            let n = gz
                .read_at(base as u64, &mut buf)
                .expect("boundary read");
            assert_eq!(&buf[..n], &expected[base..base + n]);
        }

        // backward seek inside the history ring (no reset needed)
        let mut buf = [0u8; 64];
        let n = gz.read_at(40_000, &mut buf).expect("ring read");
        assert_eq!(&buf[..n], &expected[40_000..40_000 + n]);

        // backward seek behind the ring: full reset + skip forward
        let n = gz.read_at(1_000, &mut buf).expect("reset read");
        assert_eq!(&buf[..n], &expected[1_000..1_000 + n]);

        // forward seek from the reset position
        let n = gz
            .read_at(60_000, &mut buf)
            .expect("forward seek");
        assert_eq!(&buf[..n], &expected[60_000..60_000 + n]);

        // clamping at the end of the stream
        assert_eq!(gz.read_at(BIG_LEN as u64, &mut buf).expect("eof"), 0);
        let mut tail = [0u8; 100];
        let n = gz
            .read_at((BIG_LEN - 50) as u64, &mut tail)
            .expect("tail");
        assert_eq!(n, 50);
        assert_eq!(&tail[..50], &expected[BIG_LEN - 50..]);
    }

    #[test]
    fn gzip_source_tolerates_leading_garbage() {
        // magic inside the first KiB is found
        let mut data = vec![b'Z'; 100];
        data.extend_from_slice(&SMALL_GZ);
        let gz = GzipSource::new(mem(&data)).expect("garbage prefix tolerated");
        let mut buf = vec![0u8; SMALL_PLAIN.len()];
        let n = gz.read_at(0, &mut buf).expect("read");
        assert_eq!(&buf[..n], &SMALL_PLAIN[..]);

        // magic beyond the 1024-byte scan window is not found
        let mut data = vec![b'Z'; GZIP_MAGIC_SCAN + 10];
        data.extend_from_slice(&SMALL_GZ);
        let r = GzipSource::new(mem(&data));
        assert_eq!(r.err(), Some(TtError::INVALID_FILE_FORMAT));
    }

    #[test]
    fn gzip_source_rejects_truncated_stream() {
        // cut inside the deflate payload
        let r = GzipSource::new(mem(&SMALL_GZ[..15]));
        assert_eq!(r.err(), Some(TtError::INVALID_STREAM_OPERATION));
        // header only: nothing to inflate
        let r = GzipSource::new(mem(&SMALL_GZ[..10]));
        assert_eq!(r.err(), Some(TtError::INVALID_STREAM_OPERATION));
        // empty source: header cannot even be read
        let r = GzipSource::new(mem(&[]));
        assert_eq!(r.err(), Some(TtError::INVALID_FILE_FORMAT));
    }

    #[test]
    fn gzip_source_rejects_bad_magic() {
        let r = GzipSource::new(mem(b"this is not a gzip file at all"));
        assert_eq!(r.err(), Some(TtError::INVALID_FILE_FORMAT));
        // right magic, wrong method
        let r = GzipSource::new(mem(&[0x1F, 0x8B, 0x07, 0x00]));
        assert_eq!(r.err(), Some(TtError::INVALID_FILE_FORMAT));
    }

    #[test]
    fn gzip_source_parses_all_header_flags() {
        let gz = GzipSource::new(mem(&FLAGS_GZ)).expect("FEXTRA|FNAME|FCOMMENT|FHCRC");
        assert_eq!(gz.size(), Some(FLAGS_PLAIN.len() as u64));
        let mut buf = vec![0u8; FLAGS_PLAIN.len()];
        let n = gz.read_at(0, &mut buf).expect("read");
        assert_eq!(&buf[..n], FLAGS_PLAIN);
    }

    #[test]
    fn gzip_source_rejects_reserved_flags() {
        // FLG bits 5..7 must be zero
        let r = GzipSource::new(mem(&[0x1F, 0x8B, 0x08, 0xE0]));
        assert_eq!(r.err(), Some(TtError::INVALID_FILE_FORMAT));
        // truncated header after a valid magic
        let r = GzipSource::new(mem(&[0x1F, 0x8B, 0x08, 0x00]));
        assert_eq!(r.err(), Some(TtError::INVALID_STREAM_READ));
    }

    #[test]
    fn gzip_source_handles_empty_stream() {
        let gz = GzipSource::new(mem(&EMPTY_GZ)).expect("empty gzip member");
        assert_eq!(gz.size(), Some(0));
        let mut buf = [0u8; 4];
        assert_eq!(gz.read_at(0, &mut buf).expect("eof"), 0);
        assert_eq!(gz.read_at(100, &mut buf).expect("eof"), 0);
    }
}
