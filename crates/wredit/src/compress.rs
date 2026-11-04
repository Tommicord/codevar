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

//! Small-string compression tuned for directory and file names.
//!
//! Directory names are short but repeat a lot of structure: `src`,
//! `lib`, `.rs`, `.toml`, `_test`, machine-file separators (`/`, `_`,
//! `-`, `.`) and the ASCII alphabet. General-purpose compressors are
//! useless at this size — they need hundreds of bytes of state before
//! they save anything — so this module uses a **static codebook**
//! instead, in the spirit of antirez's *smaz*: the strings that occur
//! most often are pre-listed and each compresses to a single byte.
//!
//! The table itself lives in a generated `codebook.rs`: a single flat
//! `CODEBOOK` byte array of length-prefixed entries produced by
//! `tools/gen_codebook.py`. The format here is a byte stream of three
//! kinds of token:
//!
//! | Token                    | Meaning                                        |
//! |--------------------------|------------------------------------------------|
//! | `0x00 ..= 0xFD`          | codebook entry `CODEBOOK[token]`               |
//! | `0xFE`                   | one literal byte (the next byte)               |
//! | `0xFF`                   | a literal run: `len - 1` byte, then `len` bytes |
//!
//! Because *every* byte can always be emitted as a literal, the
//! encoding is **lossless for arbitrary input**, never panics, and
//! [`decompress`] returns [`None`] (rather than a wrong string) on a
//! truncated or malformed block. Compression is greedy longest-match,
//! which keeps the encoder a few dozen lines and free of the hash
//! tables a full smaz implementation carries.
//!
//! On directories of code the codebook reaches the 4–8 byte range for
//! typical names (for example `"src/main.rs"`), which is why the
//! storage layer can keep a directory entry small while preserving the
//! name exactly: [`decompress`] is the inverse of [`compress`].
//!
//! # Examples
//!
//! ```
//! use codevar_wredit::compress::{compress, decompress};
//!
//! let name = b"src/main.rs";
//! let packed = compress(name);
//! assert!(packed.len() < name.len());
//! assert_eq!(decompress(&packed).as_deref(), Some(&name[..]));
//! ```

use crate::codebook::{CODEBOOK, CODEBOOK_ENTRIES};
use alloc::vec::Vec;

/// Codebook escape that introduces a single literal byte.
const LITERAL_BYTE: u8 = 0xFE;

/// Codebook escape that introduces a run of literal bytes.
const LITERAL_RUN: u8 = 0xFF;

/// Longest literal run a single [`LITERAL_RUN`] token can carry.
const MAX_RUN: usize = 0x100;

/// Offset of each codebook entry inside [`CODEBOOK`].
///
/// Derived at compile time so [`codebook_entry`] is an O(1) index
/// instead of a front-to-back walk. `codebook.rs` is generated, so the
/// table is computed here to stay in sync automatically.
const CODEBOOK_START: [u16; CODEBOOK_ENTRIES] = {
    let mut starts = [0u16; CODEBOOK_ENTRIES];
    let mut offset = 0usize;
    let mut index = 0usize;
    while index < CODEBOOK_ENTRIES {
        starts[index] = offset as u16;
        offset += CODEBOOK[offset] as usize + 1;
        index += 1;
    }
    starts
};

/// `BUCKET_BOUNDS[b]..BUCKET_BOUNDS[b + 1]` is the slice of
/// [`CODEBOOK_BUCKET`] that lists the entries whose first byte is `b`.
const BUCKET_BOUNDS: [u16; 257] = {
    let mut bounds = [0u16; 257];
    let mut index = 0usize;
    while index < CODEBOOK_ENTRIES {
        let first = CODEBOOK[(CODEBOOK_START[index] as usize) + 1] as usize;
        bounds[first + 1] += 1;
        index += 1;
    }
    let mut byte = 1usize;
    while byte <= 256 {
        bounds[byte] += bounds[byte - 1];
        byte += 1;
    }
    bounds
};

/// Codebook entry indices grouped by their first byte.
///
/// The encoder only has to test entries that can match the byte under
/// the cursor, which turns the per-token scan from `CODEBOOK_ENTRIES`
/// comparisons into a handful. Within a bucket the indices stay in table
/// order, so the longest-match tie-break is unchanged.
const CODEBOOK_BUCKET: [u8; CODEBOOK_ENTRIES] = {
    let mut bucket = [0u8; CODEBOOK_ENTRIES];
    let mut cursor = BUCKET_BOUNDS;
    let mut index = 0usize;
    while index < CODEBOOK_ENTRIES {
        let first = CODEBOOK[(CODEBOOK_START[index] as usize) + 1] as usize;
        bucket[cursor[first] as usize] = index as u8;
        cursor[first] += 1;
        index += 1;
    }
    bucket
};

/// Returns codebook entry `code`, or [`None`] when the token names no
/// entry.
///
/// O(1) thanks to the compile-time [`CODEBOOK_START`] index.
fn codebook_entry(code: usize) -> Option<&'static [u8]> {
    let start = usize::from(*CODEBOOK_START.get(code)?);
    let len = usize::from(*CODEBOOK.get(start)?);
    let body = start.checked_add(1)?;
    let end = body.checked_add(len)?;
    CODEBOOK.get(body..end)
}

/// Compresses `name` with the filename codebook.
///
/// The result is a self-describing byte stream; feed it to
/// [`decompress`] to recover `name` exactly.
///
/// # Examples
///
/// ```
/// use codevar_wredit::compress::compress;
///
/// assert_eq!(compress(b""), Vec::<u8>::new());
/// ```
#[must_use]
pub fn compress(name: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(name.len());
    compress_into(name, &mut out);
    out
}

/// Compresses `name`, appending the tokens to `out`.
///
/// Same output as [`compress`]; provided so a caller that already owns
/// a reusable buffer can avoid an allocation in the hot path.
pub fn compress_into(name: &[u8], out: &mut Vec<u8>) {
    // Literals are always a contiguous slice of `name`, so the encoder
    // tracks their start instead of copying them into a scratch buffer.
    let mut literal_start = 0;
    let mut index = 0;
    while index < name.len() {
        // Only entries that start with this byte can match, so scan that
        // bucket instead of the whole table.
        let first = usize::from(name[index]);
        let bucket = usize::from(BUCKET_BOUNDS[first])..usize::from(BUCKET_BOUNDS[first + 1]);
        let tail = &name[index..];
        let mut best_len = 0usize;
        let mut best_token = 0u8;
        for &code in &CODEBOOK_BUCKET[bucket] {
            let Some(entry) = codebook_entry(usize::from(code)) else {
                continue;
            };
            if entry.len() > best_len && tail.starts_with(entry) {
                best_len = entry.len();
                best_token = code;
            }
        }
        if best_len > 0 {
            flush_literals(&name[literal_start..index], out);
            out.push(best_token);
            index += best_len;
            literal_start = index;
        } else {
            index += 1;
        }
    }
    flush_literals(&name[literal_start..], out);
}

/// Emits literal bytes as [`LITERAL_BYTE`]/[`LITERAL_RUN`] tokens.
fn flush_literals(literals: &[u8], out: &mut Vec<u8>) {
    let mut start = 0;
    while start < literals.len() {
        let remaining = literals.len() - start;
        if remaining == 1 {
            out.push(LITERAL_BYTE);
            out.push(literals[start]);
            start += 1;
        } else {
            let take = remaining.min(MAX_RUN);
            out.push(LITERAL_RUN);
            // `take` is in 2..=256, so `take - 1` always fits a byte.
            out.push((take - 1) as u8);
            out.extend_from_slice(&literals[start..start + take]);
            start += take;
        }
    }
}

/// Reverses [`compress`].
///
/// Returns [`None`] when `block` is truncated or holds a token that is
/// not a codebook index, so a corrupt entry can never decode to a
/// plausible-looking wrong name.
///
/// # Examples
///
/// ```
/// use codevar_wredit::compress::decompress;
///
/// assert_eq!(decompress(&[0xFE, b'?']).as_deref(), Some(&b"?"[..]));
/// assert_eq!(decompress(&[0xFF, 0x00, b'x']).as_deref(), Some(&b"x"[..]));
/// assert_eq!(decompress(&[0xFE]), None);
/// ```
#[must_use]
pub fn decompress(block: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(block.len());
    let mut index = 0;
    while index < block.len() {
        let token = block[index];
        index += 1;
        match token {
            LITERAL_BYTE => {
                let byte = *block.get(index)?;
                index += 1;
                out.push(byte);
            }
            LITERAL_RUN => {
                let len = usize::from(*block.get(index)?) + 1;
                index += 1;
                let end = index.checked_add(len)?;
                out.extend_from_slice(block.get(index..end)?);
                index = end;
            }
            code => {
                let entry = codebook_entry(usize::from(code))?;
                out.extend_from_slice(entry);
            }
        }
    }
    Some(out)
}

/// Compresses `name` into `out`, returning the packed length.
///
/// Returns [`None`] when the packed form does not fit `out`, leaving
/// the caller free to fall back to storing the name verbatim. The name
/// itself is never truncated, so a successful return always carries
/// enough information to recover it.
///
/// # Examples
///
/// ```
/// use codevar_wredit::compress::{compress_bounded, decompress};
///
/// let mut block = [0u8; 8];
/// let len = compress_bounded(b"src/main.rs", &mut block);
/// assert!(len.is_some_and(|len| len <= block.len()));
/// assert_eq!(
///     decompress(&block[..len.unwrap_or(0)]).as_deref(),
///     Some(&b"src/main.rs"[..])
/// );
/// ```
#[must_use]
pub fn compress_bounded(name: &[u8], out: &mut [u8]) -> Option<usize> {
    let packed = compress(name);
    if packed.len() > out.len() {
        return None;
    }
    out[..packed.len()].copy_from_slice(&packed);
    Some(packed.len())
}

/// The 32-bit name fingerprint ([`crate::simd::hash_bytes`]).
///
/// This is the four-byte "compressed block" a directory entry keeps for
/// the fast rejection pass; it is a *digest*, not a reversible encoding
/// — pair it with [`compress`]/[`decompress`] or the storage layer's
/// name pool when the original bytes must come back.
#[must_use]
pub fn fingerprint32(name: &[u8]) -> u32 {
    crate::simd::hash_bytes(name)
}

/// The 64-bit name fingerprint ([`crate::simd::hash_bytes_wide`]).
///
/// The eight-byte sibling of [`fingerprint32`]; a 64-bit match implies
/// the 32-bit match because both fold the same recurrence.
#[must_use]
pub fn fingerprint64(name: &[u8]) -> u64 {
    crate::simd::hash_bytes_wide(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes, so the codec meets inputs no
    /// hand-written case would cover (including bytes no codebook entry
    /// mentions).
    fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            out.push(state as u8);
        }
        out
    }

    #[test]
    fn codebook_stays_below_the_escape_tokens() {
        assert!(CODEBOOK_ENTRIES < usize::from(LITERAL_BYTE));
        // The flat table must be exactly the sum of its records, with
        // no gaps and no empty entries.
        let mut offset = 0;
        let mut count = 0;
        while offset < CODEBOOK.len() {
            let len = usize::from(CODEBOOK[offset]);
            assert!(len > 0, "empty entry at {offset}");
            offset += len + 1;
            count += 1;
        }
        assert_eq!(offset, CODEBOOK.len(), "trailing bytes after the last entry");
        assert_eq!(count, CODEBOOK_ENTRIES);
    }

    #[test]
    fn round_trips_directory_names_exactly() {
        let names: &[&[u8]] = &[
            b"",
            b"a",
            b"src",
            b"src/main.rs",
            b"crates/codevar-wredit/src/storage/fs.rs",
            b"a_very_long_directory_name_that_should_compress_well",
            b"UPPER_CASE_NAME.TXT",
            b"weird\x00\xff\xfe\x01bytes",
        ];
        for name in names {
            let packed = compress(name);
            assert_eq!(decompress(&packed).as_deref(), Some(*name), "{name:?}");
        }
    }

    #[test]
    fn round_trips_random_bytes() {
        for seed in 1..60u32 {
            let name = pseudo_random(1 + (seed as usize * 37) % 300, seed);
            let packed = compress(&name);
            assert_eq!(decompress(&packed).as_deref(), Some(&name[..]), "seed {seed}");
        }
    }

    #[test]
    fn common_names_actually_shrink() {
        // These subtrees are exactly what the codebook targets.
        for name in [
            &b"src/main.rs"[..],
            b"crates/codevar-wredit/src/lib.rs",
            b"tests/test_storage.rs",
            b"docs/readme.md",
            b"node_modules",
        ] {
            let packed = compress(name);
            assert!(packed.len() < name.len(), "{name:?} packed to {packed:?}");
        }
    }

    #[test]
    fn decompress_rejects_malformed_blocks() {
        // A dangling literal byte / run header.
        assert_eq!(decompress(&[LITERAL_BYTE]), None);
        assert_eq!(decompress(&[LITERAL_RUN]), None);
        // A literal run that promises more bytes than the block holds.
        assert_eq!(decompress(&[LITERAL_RUN, 0x0A, b'x']), None);
        // A token in the codebook gap that names no entry.
        assert_eq!(decompress(&[LITERAL_BYTE - 1]), None);
    }

    #[test]
    fn bounded_compression_reports_fit() {
        let mut small = [0u8; 2];
        assert_eq!(compress_bounded(b"an-unexpectedly-long-name", &mut small), None);
        let mut roomy = [0u8; 32];
        let len = compress_bounded(b"src/main.rs", &mut roomy);
        assert_eq!(
            decompress(&roomy[..len.unwrap_or(0)]).as_deref(),
            Some(&b"src/main.rs"[..])
        );
    }

    #[test]
    fn fingerprints_share_the_simd_hash() {
        for name in [&b"src"[..], b"main.rs", b"a/b/c"] {
            assert_eq!(fingerprint32(name), crate::simd::hash_bytes(name));
            assert_eq!(fingerprint64(name), crate::simd::hash_bytes_wide(name));
        }
    }
}
