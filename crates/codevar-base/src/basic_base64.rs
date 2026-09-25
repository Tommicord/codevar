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
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Base64 (RFC 4648) codec with SIMD-accelerated encoding.
//!
//! [`encode`] produces the standard alphabet (`A-Z a-z 0-9 + /`) with `=`
//! padding and runs a vector kernel whenever the target provides one:
//!
//! | Target | Kernel | Selection |
//! |--------|--------|-----------|
//! | `x86` / `x86_64` | SSSE3 (`pshufb`) | run time, via [`basic_cpuid`] |
//! | `aarch64` | NEON (`tbl`/`bsl`) | run time, via [`basic_cpuid`] |
//! | `wasm32` + `simd128` | Wasm SIMD128 | compile time |
//! | anything else | scalar | — |
//!
//! Every kernel consumes whole 12-byte input blocks and emits 16 encoded
//! bytes per block. The trailing `0..11` bytes — including all `=`
//! padding — are always produced by the shared scalar tail, so every
//! path yields byte-identical output.
//!
//! [`decode`] is scalar: it accepts ASCII whitespace anywhere, requires
//! the whitespace-stripped length to be a multiple of four, and reports
//! malformed input through [`Base64Error`].
//!
//! [`basic_cpuid`]: crate::basic_cpuid

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use crate::basic_cpuid::{self, Feature};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// The RFC 4648 standard base64 alphabet.
const ENCODE_TABLE: [u8; 64] =
    *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Returns the exact length in bytes of the base64 encoding of an
/// `input_len`-byte input: `ceil(input_len / 3) * 4`.
///
/// # Performance
///
/// O(1); no allocation. [`encode`] pre-allocates exactly this many bytes.
#[must_use]
pub const fn encoded_len(input_len: usize) -> usize {
    input_len.div_ceil(3) * 4
}

/// Encodes `input` as standard base64 with `=` padding.
///
/// Selects the best SIMD kernel for this target and CPU (see the
/// [module documentation](self)); when none applies, a scalar block loop
/// runs instead. Both paths produce identical bytes, and the result
/// length is always [`encoded_len`]`(input.len())`.
///
/// # Performance
///
/// O(n) with a fixed per-block cost; 12 input bytes are processed per
/// vector iteration. Inputs shorter than 12 bytes take the scalar path
/// directly.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_base64::encode;
///
/// assert_eq!(encode(b"foobar"), "Zm9vYmFy");
/// assert_eq!(encode(b"f"), "Zg==");
/// assert_eq!(encode(b""), "");
/// ```
#[must_use]
pub fn encode(input: &[u8]) -> String {
    let mut out = alloc::vec![0u8; encoded_len(input.len())];

    let consumed = simd_encode(input, &mut out);
    let prefix = encoded_len(consumed);
    let written = prefix + encode_scalar(&input[consumed..], &mut out[prefix..]);
    debug_assert_eq!(written, out.len());

    // SAFETY: every byte written to `out` comes from `ENCODE_TABLE` (7-bit
    // ASCII) or the padding byte `=`, so `out` is always valid UTF-8.
    unsafe { String::from_utf8_unchecked(out) }
}

/// Decodes a standard base64 string into bytes.
///
/// ASCII whitespace anywhere in `input` is ignored. The remaining length
/// must be a multiple of four; `=` padding is only accepted in the final
/// quartet, and non-canonical trailing bits in the final sextet are
/// tolerated (they do not change the decoded bytes).
///
/// # Errors
///
/// - [`Base64Error::InvalidLength`] — the stripped input is not a
///   multiple of four bytes.
/// - [`Base64Error::InvalidCharacter`] — a byte outside the standard
///   alphabet (including URL-safe `-`/`_`) was found.
/// - [`Base64Error::InvalidPadding`] — `=` appears before the last
///   quartet or without a partner in the final quartet.
///
/// # Performance
///
/// Two passes over `input` (one to validate the length, one to decode),
/// a single output allocation, no intermediate buffers.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_base64::decode;
///
/// assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar");
/// assert_eq!(decode(" Zg==\n").unwrap(), b"f");
/// assert!(decode("-w==").is_err());
/// ```
pub fn decode(input: &str) -> Result<Vec<u8>, Base64Error> {
    let mut bytes = input.bytes().filter(|b| !b.is_ascii_whitespace());
    if !bytes.clone().count().is_multiple_of(4) {
        return Err(Base64Error::InvalidLength);
    }

    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    while let Some(a) = bytes.next() {
        // The length was validated above, so a full quartet exists here;
        // the arm below only keeps the compiler from assuming a panic.
        let (Some(b), Some(c_raw), Some(d_raw)) =
            (bytes.next(), bytes.next(), bytes.next())
        else {
            return Err(Base64Error::InvalidLength);
        };

        let a = decode_char(a)?;
        let b = decode_char(b)?;
        let (c, pad_c) = decode_char_or_pad(c_raw)?;
        let (d, pad_d) = decode_char_or_pad(d_raw)?;
        if pad_c && !pad_d {
            return Err(Base64Error::InvalidPadding);
        }

        let n = (u32::from(a) << 18)
            | (u32::from(b) << 12)
            | (u32::from(c) << 6)
            | u32::from(d);
        out.push(((n >> 16) & 0xff) as u8);
        if !pad_c {
            out.push(((n >> 8) & 0xff) as u8);
        }
        if !pad_d {
            out.push((n & 0xff) as u8);
        }
    }
    Ok(out)
}

/// Errors produced by [`decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Base64Error {
    /// The whitespace-stripped input length is not a multiple of four.
    InvalidLength,
    /// A byte outside the standard base64 alphabet; holds the offending
    /// byte value.
    InvalidCharacter(u8),
    /// `=` padding appeared in an invalid position.
    InvalidPadding,
}

impl fmt::Display for Base64Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength => f.write_str("base64 length must be a multiple of 4"),
            Self::InvalidCharacter(byte) => {
                write!(f, "invalid base64 character 0x{byte:02x}")
            }
            Self::InvalidPadding => f.write_str("invalid base64 padding"),
        }
    }
}

impl core::error::Error for Base64Error {}

/// Encodes `input` into `out` using the portable scalar path and returns
/// the number of bytes written.
///
/// # Panics
///
/// Debug builds assert that `out` holds at least
/// `encoded_len(input.len())` bytes; callers must provide that space.
fn encode_scalar(input: &[u8], out: &mut [u8]) -> usize {
    debug_assert!(out.len() >= encoded_len(input.len()));
    let mut i = 0;
    let mut o = 0;
    while i + 3 <= input.len() {
        let n = (u32::from(input[i]) << 16)
            | (u32::from(input[i + 1]) << 8)
            | u32::from(input[i + 2]);
        out[o] = ENCODE_TABLE[((n >> 18) & 0x3f) as usize];
        out[o + 1] = ENCODE_TABLE[((n >> 12) & 0x3f) as usize];
        out[o + 2] = ENCODE_TABLE[((n >> 6) & 0x3f) as usize];
        out[o + 3] = ENCODE_TABLE[(n & 0x3f) as usize];
        i += 3;
        o += 4;
    }
    match input.len() - i {
        1 => {
            let n = u32::from(input[i]) << 16;
            out[o] = ENCODE_TABLE[((n >> 18) & 0x3f) as usize];
            out[o + 1] = ENCODE_TABLE[((n >> 12) & 0x3f) as usize];
            out[o + 2] = b'=';
            out[o + 3] = b'=';
            o += 4;
        }
        2 => {
            let n = (u32::from(input[i]) << 16) | (u32::from(input[i + 1]) << 8);
            out[o] = ENCODE_TABLE[((n >> 18) & 0x3f) as usize];
            out[o + 1] = ENCODE_TABLE[((n >> 12) & 0x3f) as usize];
            out[o + 2] = ENCODE_TABLE[((n >> 6) & 0x3f) as usize];
            out[o + 3] = b'=';
            o += 4;
        }
        _ => {}
    }
    o
}

/// Decodes one base64 alphabet byte into its six-bit value.
fn decode_char(c: u8) -> Result<u8, Base64Error> {
    match c {
        b'A'..=b'Z' => Ok(c - b'A'),
        b'a'..=b'z' => Ok(c - b'a' + 26),
        b'0'..=b'9' => Ok(c - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(Base64Error::InvalidCharacter(c)),
    }
}

/// Decodes one base64 byte that may be `=` padding instead of alphabet.
fn decode_char_or_pad(c: u8) -> Result<(u8, bool), Base64Error> {
    if c == b'=' {
        Ok((0, true))
    } else {
        Ok((decode_char(c)?, false))
    }
}

/// Runs the best available SIMD encoder and returns how many input bytes
/// it consumed (always a multiple of 12), having written `4/3` of that
/// many bytes to `out`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn simd_encode(input: &[u8], out: &mut [u8]) -> usize {
    if basic_cpuid::has(Feature::Ssse3) {
        // SAFETY: `has` confirmed SSSE3 for this CPU (or the binary was
        // compiled with it); the kernel bounds-checks every access
        // against `input` and `out` before reading or writing.
        unsafe { ssse3::encode_blocks(input, out) }
    } else {
        0
    }
}

/// Runs the NEON encoder when the CPU reports Advanced SIMD support.
#[cfg(target_arch = "aarch64")]
fn simd_encode(input: &[u8], out: &mut [u8]) -> usize {
    if basic_cpuid::has(Feature::Neon) {
        // SAFETY: `has` confirmed NEON for this CPU; the kernel
        // bounds-checks every access against `input` and `out`.
        unsafe { neon::encode_blocks(input, out) }
    } else {
        0
    }
}

/// Runs the Wasm SIMD128 encoder, which the compiler gate above already
/// proved is enabled for this binary.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn simd_encode(input: &[u8], out: &mut [u8]) -> usize {
    // SAFETY: `simd128` is a compile-time target feature here, and the
    // kernel bounds-checks every access against `input` and `out`.
    unsafe { wasm_simd::encode_blocks(input, out) }
}

/// Targets without a vector kernel consume nothing and fall back to the
/// scalar tail for the whole input.
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
fn simd_encode(_input: &[u8], _out: &mut [u8]) -> usize {
    0
}

/// Tables shared by the NEON and Wasm SIMD128 kernels.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
mod simd_tables {
    /// Byte permutation applied to one loaded input block so that each
    /// 32-bit lane holds the bytes needed to extract four sextets:
    /// `[in1, in2, in0, in1]` per lane, little-endian byte order.
    pub(super) const RESHUFFLE_MASK: [u8; 16] =
        [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

    /// Per-lane AND masks for the multiply-based sextet extraction
    /// (equivalent to the `0x0FC0FC00` / `0x003F03F0` / `0x01000010`
    /// constants of the scalar formulation, broadcast per 32-bit lane).
    pub(super) const MUL_MASK_A: [u8; 16] = [
        0x00, 0xFC, 0xC0, 0x0F, 0x00, 0xFC, 0xC0, 0x0F, 0x00, 0xFC, 0xC0, 0x0F, 0x00,
        0xFC, 0xC0, 0x0F,
    ];

    /// See [`MUL_MASK_A`](MUL_MASK_A).
    pub(super) const MUL_MASK_B: [u8; 16] = [
        0xF0, 0x03, 0x3F, 0x00, 0xF0, 0x03, 0x3F, 0x00, 0xF0, 0x03, 0x3F, 0x00, 0xF0,
        0x03, 0x3F, 0x00,
    ];

    /// See [`MUL_MASK_A`](MUL_MASK_A).
    pub(super) const MUL_FACTOR: [u8; 16] = [
        0x10, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0x01, 0x10,
        0x00, 0x00, 0x01,
    ];

    /// Selects the shifted copy for each 16-bit lane: low lanes take the
    /// `>> 10` result, high lanes the `>> 6` result, reproducing a
    /// widening multiply high against `0x04000040`.
    pub(super) const MULHI_BLEND: [u8; 16] = [
        0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00,
        0x00, 0xFF, 0xFF,
    ];

    /// Offsets added to each sextet value `v` (0..64) to reach its
    /// alphabet byte, indexed by a derived position:
    ///
    /// | Position | `v` range | Offset | Result |
    /// |----------|-----------|--------|--------|
    /// | 0 | 0..=25 | `+65` | `A`–`Z` |
    /// | 1 | 26..=51 | `+71` | `a`–`z` |
    /// | 2..=11 | 52..=61 | `-4` | `0`–`9` |
    /// | 12 | 62 | `-19` | `+` |
    /// | 13 | 63 | `-16` | `/` |
    ///
    /// Stored as wrapping bytes because the vectors are added unsigned.
    pub(super) const TRANSLATE_OFFSETS: [u8; 16] = [
        65, 71, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 237, 240, 0, 0,
    ];
}

/// SSSE3 encoder: reshuffles 12 input bytes into 16 sextet lanes, then
/// translates them with one table lookup per lane.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod ssse3 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm_add_epi8, _mm_and_si128, _mm_cmpgt_epi8, _mm_mulhi_epu16,
        _mm_mullo_epi16, _mm_or_si128, _mm_set_epi8, _mm_set1_epi8, _mm_set1_epi32,
        _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128, _mm_sub_epi8, _mm_subs_epu8,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm_add_epi8, _mm_and_si128, _mm_cmpgt_epi8, _mm_mulhi_epu16,
        _mm_mullo_epi16, _mm_or_si128, _mm_set_epi8, _mm_set1_epi8, _mm_set1_epi32,
        _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128, _mm_sub_epi8, _mm_subs_epu8,
    };

    /// Input bytes consumed per iteration.
    const BLOCK: usize = 12;
    /// Encoded bytes produced per iteration.
    const OUT_BLOCK: usize = 16;

    /// Encodes complete 12-byte blocks of `input` into `out` and returns
    /// the number of input bytes consumed (`0..=input.len()`, a multiple
    /// of 12).
    ///
    /// # Safety
    ///
    /// The CPU must support SSSE3.
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn encode_blocks(input: &[u8], out: &mut [u8]) -> usize {
        let mut i = 0;
        let mut o = 0;
        while i + BLOCK <= input.len() && o + OUT_BLOCK <= out.len() {
            let mut staging = [0u8; 16];
            staging[..BLOCK].copy_from_slice(&input[i..i + BLOCK]);
            // SAFETY: `staging` is a readable 16-byte local (its unused
            // tail is zero), `translate`/`reshuffle` only manipulate
            // registers under the SSSE3 `target_feature` of this
            // function, and the loop condition guarantees 16 writable
            // bytes at `out[o..]`.
            unsafe {
                let bytes = core::ptr::read_unaligned(staging.as_ptr().cast::<__m128i>());
                let chars = translate(reshuffle(bytes));
                let dst = out[o..o + OUT_BLOCK].as_mut_ptr().cast::<__m128i>();
                _mm_storeu_si128(dst, chars);
            }
            i += BLOCK;
            o += OUT_BLOCK;
        }
        i
    }

    /// Rearranges one 16-byte block so every 32-bit lane carries the
    /// bytes of one input triplet, then extracts the four sextets per
    /// lane with AND/multiply/OR bit gymnastics.
    ///
    /// # Safety
    ///
    /// Requires SSSE3 (enabled by `target_feature` on this function).
    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn reshuffle(bytes: __m128i) -> __m128i {
        let shuffled = _mm_shuffle_epi8(
            bytes,
            _mm_set_epi8(10, 11, 9, 10, 7, 8, 6, 7, 4, 5, 3, 4, 1, 2, 0, 1),
        );
        let t0 = _mm_and_si128(shuffled, _mm_set1_epi32(0x0FC0FC00_u32 as i32));
        let t1 = _mm_mulhi_epu16(t0, _mm_set1_epi32(0x04000040_u32 as i32));
        let t2 = _mm_and_si128(shuffled, _mm_set1_epi32(0x003F03F0_u32 as i32));
        let t3 = _mm_mullo_epi16(t2, _mm_set1_epi32(0x01000010_u32 as i32));
        _mm_or_si128(t1, t3)
    }

    /// Maps sextet values `0..64` to their alphabet bytes with one
    /// saturating subtract, one signed compare, and one `pshufb` lookup
    /// into a 16-entry offset table. The lookup index derived for
    /// `v <= 63` never exceeds 13, so `pshufb` never falls outside the
    /// offset table.
    ///
    /// # Safety
    ///
    /// Requires SSSE3 (enabled by `target_feature` on this function).
    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn translate(sextets: __m128i) -> __m128i {
        let offsets = _mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,
        );
        let indices = _mm_subs_epu8(sextets, _mm_set1_epi8(51));
        let above = _mm_cmpgt_epi8(sextets, _mm_set1_epi8(25));
        let indices = _mm_sub_epi8(indices, above);
        _mm_add_epi8(sextets, _mm_shuffle_epi8(offsets, indices))
    }
}

/// NEON encoder: the same reshuffle/translate strategy as the SSSE3
/// kernel, using `tbl` lookups and byte-select for the translation.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::simd_tables::{
        MUL_FACTOR, MUL_MASK_A, MUL_MASK_B, MULHI_BLEND, RESHUFFLE_MASK,
        TRANSLATE_OFFSETS,
    };
    use core::arch::aarch64::{
        uint8x16_t, uint16x8_t, vaddq_u8, vandq_u8, vbslq_u8, vcgtq_s8, vdupq_n_s8,
        vdupq_n_u8, vmulq_u16, vorrq_u8, vqsubq_u8, vqtbl1q_u8, vreinterpretq_s8_u8,
        vreinterpretq_u8_u16, vreinterpretq_u16_u8, vshrq_n_u16, vsubq_u8,
    };

    /// Input bytes consumed per iteration.
    const BLOCK: usize = 12;
    /// Encoded bytes produced per iteration.
    const OUT_BLOCK: usize = 16;

    /// Encodes complete 12-byte blocks of `input` into `out` and returns
    /// the number of input bytes consumed.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON (Advanced SIMD).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn encode_blocks(input: &[u8], out: &mut [u8]) -> usize {
        let mut i = 0;
        let mut o = 0;
        while i + BLOCK <= input.len() && o + OUT_BLOCK <= out.len() {
            let mut staging = [0u8; 16];
            staging[..BLOCK].copy_from_slice(&input[i..i + BLOCK]);
            // SAFETY: `staging` is a readable 16-byte local; the loop
            // condition guarantees 16 writable bytes at `out[o..]`.
            unsafe {
                let bytes =
                    core::ptr::read_unaligned(staging.as_ptr().cast::<uint8x16_t>());
                let chars = translate(reshuffle(bytes));
                let dst = out[o..o + OUT_BLOCK].as_mut_ptr().cast::<uint8x16_t>();
                core::ptr::write_unaligned(dst, chars);
            }
            i += BLOCK;
            o += OUT_BLOCK;
        }
        i
    }

    /// See the SSSE3 kernel of the same name for the bit-level layout.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn reshuffle(bytes: uint8x16_t) -> uint8x16_t {
        // SAFETY: register-only NEON operations; table indices stay in
        // `0..12`, inside the 16-byte `tbl` operand.
        unsafe {
            let mask = load16::<uint8x16_t>(&RESHUFFLE_MASK);
            let shuffled = vqtbl1q_u8(bytes, mask);

            let t0 = vandq_u8(shuffled, load16::<uint8x16_t>(&MUL_MASK_A));
            let t1 = mulhi(t0);
            let t2 = vandq_u8(shuffled, load16::<uint8x16_t>(&MUL_MASK_B));
            let t3 = vmulq_u16(
                vreinterpretq_u16_u8(t2),
                vreinterpretq_u16_u8(load16::<uint8x16_t>(&MUL_FACTOR)),
            );
            vorrq_u8(vreinterpretq_u8_u16(t1), vreinterpretq_u8_u16(t3))
        }
    }

    /// Reproduces `_mm_mulhi_epu16(t, 0x04000040)`: the low 16-bit lane
    /// of every 32-bit group shifts right by 10, the high lane by 6.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mulhi(t: uint8x16_t) -> uint16x8_t {
        // SAFETY: register-only NEON operations.
        unsafe {
            let lanes = vreinterpretq_u16_u8(t);
            let low = vshrq_n_u16::<10>(lanes);
            let high = vshrq_n_u16::<6>(lanes);
            let blend = load16::<uint8x16_t>(&MULHI_BLEND);
            vreinterpretq_u16_u8(vbslq_u8(
                blend,
                vreinterpretq_u8_u16(high),
                vreinterpretq_u8_u16(low),
            ))
        }
    }

    /// See the SSSE3 kernel of the same name.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn translate(sextets: uint8x16_t) -> uint8x16_t {
        // SAFETY: register-only NEON operations; the lookup index never
        // exceeds 13, so `tbl` yields a real offset (never zero-fill).
        unsafe {
            let indices = vqsubq_u8(sextets, vdupq_n_u8(51));
            let above: uint8x16_t =
                vcgtq_s8(vreinterpretq_s8_u8(sextets), vdupq_n_s8(25));
            let indices = vsubq_u8(indices, above);
            let offsets = vqtbl1q_u8(load16::<uint8x16_t>(&TRANSLATE_OFFSETS), indices);
            vaddq_u8(sextets, offsets)
        }
    }

    /// Loads a 16-byte table without imposing alignment requirements.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load16<T>(bytes: &[u8; 16]) -> T {
        debug_assert_eq!(core::mem::size_of::<T>(), 16);
        // SAFETY: `T` is a 16-byte NEON vector (checked above) and
        // `bytes` is a readable 16-byte table.
        unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) }
    }
}

/// WebAssembly SIMD128 encoder: the same reshuffle/translate strategy
/// as the SSSE3 kernel, using `swizzle` lookups.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use super::simd_tables::{
        MUL_FACTOR, MUL_MASK_A, MUL_MASK_B, MULHI_BLEND, RESHUFFLE_MASK,
        TRANSLATE_OFFSETS,
    };
    use core::arch::wasm32::{
        u8x16_add, u8x16_gt, u8x16_splat, u8x16_sub, u8x16_sub_sat, u8x16_swizzle,
        u16x8_mul, u16x8_shr, v128, v128_and, v128_bitselect, v128_or,
    };

    /// Input bytes consumed per iteration.
    const BLOCK: usize = 12;
    /// Encoded bytes produced per iteration.
    const OUT_BLOCK: usize = 16;

    /// Encodes complete 12-byte blocks of `input` into `out` and returns
    /// the number of input bytes consumed.
    ///
    /// # Safety
    ///
    /// The `simd128` target feature must be enabled.
    #[target_feature(enable = "simd128")]
    pub(super) unsafe fn encode_blocks(input: &[u8], out: &mut [u8]) -> usize {
        let mut i = 0;
        let mut o = 0;
        while i + BLOCK <= input.len() && o + OUT_BLOCK <= out.len() {
            let mut staging = [0u8; 16];
            staging[..BLOCK].copy_from_slice(&input[i..i + BLOCK]);
            // SAFETY: `staging` is a readable 16-byte local; the loop
            // condition guarantees 16 writable bytes at `out[o..]`.
            unsafe {
                let bytes = core::ptr::read_unaligned(staging.as_ptr().cast::<v128>());
                let chars = translate(reshuffle(bytes));
                let dst = out[o..o + OUT_BLOCK].as_mut_ptr().cast::<v128>();
                core::ptr::write_unaligned(dst, chars);
            }
            i += BLOCK;
            o += OUT_BLOCK;
        }
        i
    }

    /// See the SSSE3 kernel of the same name for the bit-level layout.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn reshuffle(bytes: v128) -> v128 {
        // SAFETY: register-only SIMD128 operations; table indices stay
        // inside the 16-byte `swizzle` operand (out-of-range would
        // zero-fill, which cannot happen here).
        unsafe {
            let shuffled = u8x16_swizzle(bytes, load16(&RESHUFFLE_MASK));
            let t0 = v128_and(shuffled, load16(&MUL_MASK_A));
            let t1 = mulhi(t0);
            let t2 = v128_and(shuffled, load16(&MUL_MASK_B));
            let t3 = u16x8_mul(t2, load16(&MUL_FACTOR));
            v128_or(t1, t3)
        }
    }

    /// Reproduces `_mm_mulhi_epu16(t, 0x04000040)`: the low 16-bit lane
    /// of every 32-bit group shifts right by 10, the high lane by 6.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn mulhi(t: v128) -> v128 {
        // SAFETY: register-only SIMD128 operations.
        unsafe {
            let low = u16x8_shr(t, 10);
            let high = u16x8_shr(t, 6);
            v128_bitselect(high, low, load16(&MULHI_BLEND))
        }
    }

    /// See the SSSE3 kernel of the same name.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn translate(sextets: v128) -> v128 {
        // SAFETY: register-only SIMD128 operations; the lookup index
        // never exceeds 13, so `swizzle` never zero-fills.
        unsafe {
            let indices = u8x16_sub_sat(sextets, u8x16_splat(51));
            let above = u8x16_gt(sextets, u8x16_splat(25));
            let indices = u8x16_sub(indices, above);
            let offsets = u8x16_swizzle(load16(&TRANSLATE_OFFSETS), indices);
            u8x16_add(sextets, offsets)
        }
    }

    /// Loads a 16-byte table without imposing alignment requirements.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn load16(bytes: &[u8; 16]) -> v128 {
        // SAFETY: `v128` is 16 bytes wide and `bytes` is a readable
        // 16-byte table.
        unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<v128>()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// Independent, deliberately naive encoder used as the oracle for
    /// the production implementation (including its SIMD kernels).
    fn reference_encode(input: &[u8]) -> String {
        let mut out = String::new();
        for chunk in input.chunks(3) {
            let n = (u32::from(chunk[0]) << 16)
                | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
                | u32::from(*chunk.get(2).unwrap_or(&0));
            out.push(char::from(ENCODE_TABLE[((n >> 18) & 0x3f) as usize]));
            out.push(char::from(ENCODE_TABLE[((n >> 12) & 0x3f) as usize]));
            if chunk.len() > 1 {
                out.push(char::from(ENCODE_TABLE[((n >> 6) & 0x3f) as usize]));
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(char::from(ENCODE_TABLE[(n & 0x3f) as usize]));
            } else {
                out.push('=');
            }
        }
        out
    }

    /// Deterministic pseudo-random bytes (xorshift64*), so the sweep
    /// below needs no dependency on a RNG crate.
    fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u8
            })
            .collect()
    }

    #[test]
    fn encodes_rfc4648_test_vectors() {
        let vectors: [(&[u8], &str); 8] = [
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
            (b"Man", "TWFu"),
        ];
        for (input, expected) in vectors {
            assert_eq!(encode(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn decodes_rfc4648_test_vectors() {
        let vectors: [(&str, &[u8]); 8] = [
            ("", b""),
            ("Zg==", b"f"),
            ("Zm8=", b"fo"),
            ("Zm9v", b"foo"),
            ("Zm9vYg==", b"foob"),
            ("Zm9vYmE=", b"fooba"),
            ("Zm9vYmFy", b"foobar"),
            ("TWFu", b"Man"),
        ];
        for (input, expected) in vectors {
            assert_eq!(decode(input).unwrap(), expected, "input {input:?}");
        }
    }

    #[test]
    fn encoded_len_matches_output_and_formula() {
        for len in 0..=64usize {
            let input = pseudo_random_bytes(len, 0xA5A5_5A5A_0000_0000 ^ len as u64);
            assert_eq!(encoded_len(len), len.div_ceil(3) * 4, "len {len}");
            assert_eq!(encode(&input).len(), encoded_len(len), "len {len}");
        }
    }

    /// Sweeps every length around the 12-byte SIMD block boundary and
    /// compares the production encoder (SIMD on capable CPUs, scalar
    /// otherwise) against the independent reference.
    #[test]
    fn encode_matches_reference_for_all_block_boundaries() {
        for len in 0..=240usize {
            let input = pseudo_random_bytes(len, 0x1234_5678_9ABC_DEF0 ^ len as u64);
            assert_eq!(
                encode(&input),
                reference_encode(&input),
                "pseudo-random input of length {len}"
            );
        }
        // All-ones and all-zeros stress every alphabet boundary.
        for len in [0usize, 1, 2, 3, 11, 12, 13, 24, 25, 47, 48, 49, 96] {
            assert_eq!(
                encode(&vec![0xFF; len]),
                reference_encode(&vec![0xFF; len]),
                "0xFF input of length {len}"
            );
            assert_eq!(
                encode(&vec![0x00; len]),
                reference_encode(&vec![0x00; len]),
                "0x00 input of length {len}"
            );
        }
    }

    #[test]
    fn round_trips_every_byte_value() {
        let data: Vec<u8> = (0u16..=0xFF).map(|i| u8::try_from(i).unwrap()).collect();
        let decoded = decode(&encode(&data)).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trips_pseudo_random_lengths() {
        for len in 0..=128usize {
            let input = pseudo_random_bytes(len, 0xDEAD_BEEF_CAFE_0000 ^ len as u64);
            let encoded = encode(&input);
            assert_eq!(encoded.len(), encoded_len(len), "len {len}");
            assert_eq!(decode(&encoded).unwrap(), input, "len {len}");
        }
    }

    #[test]
    fn encode_uses_standard_alphabet_with_padding() {
        assert_eq!(encode(&[0xFF; 3]), "////");
        let plus_slash = encode(&[0xFB, 0xF0, 0x00]);
        assert!(
            plus_slash.contains('+') || plus_slash.contains('/'),
            "{plus_slash}"
        );
        assert_eq!(encode(&[0xFB]), "+w==");
        assert_eq!(encode(&[0x00]), "AA==");
        assert_eq!(encode(&[0x00, 0x00]), "AAA=");
        assert_eq!(encode(&[0x00, 0x00, 0x00]), "AAAA");
    }

    #[test]
    fn decode_accepts_ascii_whitespace_anywhere() {
        assert_eq!(decode("Zg==\n").unwrap(), vec![b'f']);
        assert_eq!(decode(" Zg== ").unwrap(), vec![b'f']);
        assert_eq!(decode("Zg\t==\r\n").unwrap(), vec![b'f']);
        assert_eq!(decode(" Zm 8 = ").unwrap(), b"fo".to_vec());
        assert_eq!(decode("   ").unwrap(), Vec::<u8>::new());
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_rejects_lengths_not_multiple_of_four() {
        for n in 0..8usize {
            if n % 4 == 0 {
                continue;
            }
            let input = "A".repeat(n);
            assert_eq!(
                decode(&input).unwrap_err(),
                Base64Error::InvalidLength,
                "n {n}"
            );
        }
        assert_eq!(decode("Zg").unwrap_err(), Base64Error::InvalidLength);
        assert_eq!(decode("Zg=").unwrap_err(), Base64Error::InvalidLength);
        assert_eq!(decode("Zg===").unwrap_err(), Base64Error::InvalidLength);
    }

    #[test]
    fn decode_rejects_invalid_alphabet_characters() {
        for input in ["****", "-_-_", "!!!!", "Zg=!", "Zgé"] {
            let error = decode(input).unwrap_err();
            match error {
                Base64Error::InvalidCharacter(byte) => {
                    assert!(
                        error.to_string().contains("invalid base64"),
                        "input {input:?}: {error}"
                    );
                    assert_eq!(
                        error.to_string(),
                        format!("invalid base64 character 0x{byte:02x}")
                    );
                }
                other => panic!("expected InvalidCharacter for {input:?}, got {other:?}"),
            }
        }
        // URL-safe alphabet is intentionally unsupported.
        assert!(decode("-w==").is_err());
        assert_eq!(encode(&[0xFB]), "+w==");
    }

    #[test]
    fn decode_rejects_malformed_padding_placement() {
        assert_eq!(decode("AA=A").unwrap_err(), Base64Error::InvalidPadding);
        assert_eq!(decode("Zg=A").unwrap_err(), Base64Error::InvalidPadding);
        assert!(matches!(
            decode("A==="),
            Err(Base64Error::InvalidCharacter(_))
        ));
        assert!(matches!(
            decode("=AAA"),
            Err(Base64Error::InvalidCharacter(_))
        ));
        assert!(matches!(
            decode("===="),
            Err(Base64Error::InvalidCharacter(_))
        ));
    }

    #[test]
    fn decode_handles_canonical_zero_padding_vectors() {
        assert_eq!(decode("AA==").unwrap(), vec![0u8]);
        assert_eq!(decode("AAA=").unwrap(), vec![0u8, 0u8]);
        assert_eq!(decode("AAAA").unwrap(), vec![0u8, 0u8, 0u8]);
        // Trailing bits outside the final sextet are ignored, not
        // rejected: "AZ==" keeps byte 1, "AB==" keeps byte 0.
        assert_eq!(decode("AZ==").unwrap(), vec![1u8]);
        assert_eq!(decode("AB==").unwrap(), vec![0u8]);
    }

    /// The exact messages matter: `codevar-wsocket` forwards
    /// `Base64Error`'s display text as its own decode error payload.
    #[test]
    fn error_messages_are_stable() {
        assert_eq!(
            Base64Error::InvalidLength.to_string(),
            "base64 length must be a multiple of 4"
        );
        assert_eq!(
            Base64Error::InvalidCharacter(b'!').to_string(),
            "invalid base64 character 0x21"
        );
        assert_eq!(
            Base64Error::InvalidPadding.to_string(),
            "invalid base64 padding"
        );
    }

    #[test]
    fn one_mib_round_trip() {
        let data: Vec<u8> = (0..1024 * 1024)
            .map(|i| u8::try_from(i % 256).unwrap())
            .collect();
        let encoded = encode(&data);
        assert_eq!(encoded.len(), data.len().div_ceil(3) * 4);
        assert_eq!(decode(&encoded).unwrap(), data);
    }

    #[test]
    fn many_small_round_trips_reuse_buffers() {
        for i in 0..1000usize {
            let input: Vec<u8> = (0..(i % 17))
                .map(|j| u8::try_from((j * 7 + i) % 256).unwrap())
                .collect();
            let encoded = encode(&input);
            assert_eq!(decode(&encoded).unwrap(), input, "iteration {i}");
        }
    }

    #[test]
    fn encode_output_never_contains_whitespace_or_newlines() {
        let data: Vec<u8> = (0u16..=255).map(|i| u8::try_from(i).unwrap()).collect();
        let encoded = encode(&data);
        assert!(!encoded.chars().any(char::is_whitespace));
        assert!(
            encoded.chars().all(|c| {
                c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
            })
        );
    }

    /// Verifies the dispatch wiring: `simd_encode` must hand whole
    /// blocks to the vector kernel when the CPU reports the feature,
    /// and consume nothing when it does not.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn simd_dispatch_consumes_blocks_when_feature_present() {
        let input = vec![0x42u8; 48];
        let mut out = vec![0u8; encoded_len(input.len())];
        let consumed = simd_encode(&input, &mut out);
        if basic_cpuid::has(Feature::Ssse3) {
            assert_eq!(consumed, input.len());
        } else {
            assert_eq!(consumed, 0);
        }
    }

    /// Pins the SIMD-vs-scalar contract directly: the vector kernel's
    /// block output must equal what the scalar encoder produces for the
    /// same prefix, for every block count.
    #[test]
    fn simd_blocks_match_scalar_prefix() {
        for len in 0..=96usize {
            let input = pseudo_random_bytes(len, 0x0BAD_F00D_DEAD_0000 ^ len as u64);
            let consumed = {
                let mut out = vec![0u8; encoded_len(len)];
                let consumed = simd_encode(&input, &mut out);
                let prefix = encoded_len(consumed);
                let written =
                    prefix + encode_scalar(&input[consumed..], &mut out[prefix..]);
                assert_eq!(written, out.len(), "len {len}");
                assert_eq!(
                    String::from_utf8(out).unwrap(),
                    reference_encode(&input),
                    "len {len}"
                );
                consumed
            };
            assert_eq!(consumed % 12, 0, "len {len}");
            assert!(consumed <= len, "len {len}");
            assert!(
                consumed + 12 > len || consumed == len,
                "SIMD stopped early at len {len}: consumed {consumed}"
            );
        }
    }
}
