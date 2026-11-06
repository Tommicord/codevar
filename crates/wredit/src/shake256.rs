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

//! SHAKE256, the Keccak-based extendable-output function of FIPS 202.
//!
//! The history journal (`crate::history_ring_journal`) fingerprints the
//! raw element streams it retains, so a demoted, compacted or reloaded
//! window can be checked before it is spliced back into a live document.
//! That check sits on the *cold* path — a window closes every few
//! kilobytes of edits, not on every keystroke — which is exactly where a
//! real hash earns its cost: a weak 64-bit checksum over retained history
//! collides by accident at edit-corpus sizes, and a journal collision
//! silently restores a fabricated document.
//!
//! # Layout
//!
//! SHAKE256 is `sponge[Keccak-f[1600], pad10*1 || 1111, r = 1088]`: a
//! 1600-bit state of 25 little-endian 64-bit lanes, a 136-byte rate, a
//! 24-round permutation, and the `0x1F`/`0x80` delimited suffix.
//! [`keccak_f1600`] is the scalar reference the whole module is verified
//! against; the vector paths only ever reorder the same arithmetic.
//!
//! # Examples
//!
//! ```
//! use codevar_wredit::shake256::{shake256, shake256_digest, DIGEST_LEN};
//!
//! // FIPS 202 example: SHAKE256("abc", 512).
//! let mut out = [0u8; 64];
//! shake256(b"abc", &mut out);
//! assert_eq!(&out[..DIGEST_LEN], &shake256_digest(b"abc")[..]);
//! ```

/// Rate of SHAKE256 in bytes: message bytes absorbed per permutation.
pub const RATE_BYTES: usize = 136;

/// Default digest length, the 512 bits most callers fingerprint with.
pub const DIGEST_LEN: usize = 32;

/// Lanes in one Keccak-f[1600] state.
pub const LANE_COUNT: usize = 25;

/// Streams one interleaved permutation carries.
pub const BATCH_STREAMS: usize = 4;

/// Lanes in one interleaved batch state: [`LANE_COUNT`] per stream.
pub const BATCH_LANES: usize = LANE_COUNT * BATCH_STREAMS;

/// Lanes the rate fills (136 / 8).
const RATE_LANES: usize = RATE_BYTES / 8;

/// Packed elements the rate holds when absorbing a `u32` stream.
const RATE_WORDS: usize = RATE_BYTES / core::mem::size_of::<u32>();

/// Rounds of Keccak-f[1600].
const ROUNDS: usize = 24;

/// Domain separation byte of SHAKE: the `1111` suffix, then the leading
/// `1` of `pad10*1`, read as a byte.
const DOMAIN_BYTE: u8 = 0x1F;

/// Last byte of the final padded block (the trailing `1` of `pad10*1`).
const FINAL_BYTE: u8 = 0x80;

const _: () =
    assert!(RATE_BYTES.is_multiple_of(8) && RATE_WORDS.is_multiple_of(2) && RATE_LANES < LANE_COUNT);

/// Iota round constants of Keccak-f[1600] (FIPS 202, Table 4).
const RC: [u64; ROUNDS] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rho rotation offsets as `RHO[x][y]` (FIPS 202, Table 2).
const RHO: [[u32; 5]; 5] = [
    // x = 0
    [0, 36, 3, 41, 18],
    // x = 1
    [1, 44, 10, 45, 2],
    // x = 2
    [62, 6, 43, 15, 61],
    // x = 3
    [28, 55, 25, 21, 56],
    // x = 4
    [27, 20, 39, 8, 14],
];

/// Rho offset of every lane, indexed by the flat lane index `x + 5 * y`.
///
/// Derived from [`RHO`] at compile time, so a transcription slip in the
/// source table cannot silently change the permutation.
const RHO_ROT: [u32; LANE_COUNT] = {
    let mut out = [0u32; LANE_COUNT];
    let mut x = 0;
    while x < 5 {
        let mut y = 0;
        while y < 5 {
            out[x + 5 * y] = RHO[x][y];
            y += 1;
        }
        x += 1;
    }
    out
};

/// Pi destination of every lane: `B[PI_DST[i]] = rot(A[i], r_i)`, i.e.
/// lane `(x, y)` moves to `(y, 2x + 3y)`.
const PI_DST: [usize; LANE_COUNT] = {
    let mut out = [0usize; LANE_COUNT];
    let mut x = 0;
    while x < 5 {
        let mut y = 0;
        while y < 5 {
            out[x + 5 * y] = y + 5 * ((2 * x + 3 * y) % 5);
            y += 1;
        }
        x += 1;
    }
    out
};

/// Column parities of the theta step: `COL[x][y]` is lane `x + 5 * y`.
const COL: [[usize; 5]; 5] = {
    let mut out = [[0usize; 5]; 5];
    let mut x = 0;
    while x < 5 {
        let mut y = 0;
        while y < 5 {
            out[x][y] = x + 5 * y;
            y += 1;
        }
        x += 1;
    }
    out
};

/// Reads up to eight bytes as a little-endian lane, zero-filling missing
/// high bytes.
///
/// Short reads only happen on the padded tail, where zero-fill is exactly
/// what `pad10*1` demands.
#[inline]
fn lane_from(bytes: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    let len = bytes.len().min(8);
    word[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(word)
}

/// Little-endian lane made of two packed elements.
///
/// This is [`lane_from`] over the eight bytes of `low` followed by `high`,
/// which is what makes the word path bit-identical to the byte path
/// whatever the target endianness.
#[inline]
const fn lane_from_words(low: u32, high: u32) -> u64 {
    (low as u64) | ((high as u64) << 32)
}

/// Writes a lane little-endian into `out`, truncating a short tail.
#[inline]
fn lane_into(lane: u64, out: &mut [u8]) {
    let bytes = lane.to_le_bytes();
    let len = out.len().min(8);
    out[..len].copy_from_slice(&bytes[..len]);
}

/// The Keccak-f[1600] permutation, scalar reference.
///
/// Each round applies theta, then rho+pi as one re-indexing pass, then
/// chi, then iota — 24 rounds, no message schedule, no allocation. Every
/// test in this module and every vector path is checked against it, so it
/// stays as close to the specification as the compiler allows.
#[inline]
fn keccak_f1600(state: &mut [u64; LANE_COUNT]) {
    let mut b = [0u64; LANE_COUNT];
    let mut c = [0u64; 5];
    for &iota in RC.iter() {
        for (x, parity) in c.iter_mut().enumerate() {
            let column = &COL[x];
            *parity =
                state[column[0]] ^ state[column[1]] ^ state[column[2]] ^ state[column[3]] ^ state[column[4]];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= d;
            }
        }
        for (i, lane) in state.iter().enumerate() {
            b[PI_DST[i]] = lane.rotate_left(RHO_ROT[i]);
        }
        for y in 0..5 {
            for x in 0..5 {
                let row = |column: usize| b[(column % 5) + 5 * y];
                state[x + 5 * y] = row(x) ^ ((!row(x + 1)) & row(x + 2));
            }
        }
        state[0] ^= iota;
    }
}

/// An incremental SHAKE256 sponge.
///
/// The state is 25 lanes plus one partially filled rate block — 344 bytes
/// total, stack-resident and `Copy`, so an in-progress fingerprint can be
/// snapshotted without an allocation.
///
/// # Examples
///
/// ```
/// use codevar_wredit::shake256::{shake256, Shake256};
///
/// let mut sponge = Shake256::new();
/// sponge.update(b"ab");
/// sponge.update(b"c");
/// let mut split = [0u8; 32];
/// let mut one_shot = [0u8; 32];
/// sponge.squeeze(&mut split);
/// shake256(b"abc", &mut one_shot);
/// assert_eq!(one_shot, split);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Shake256 {
    /// The 1600-bit state; lanes 0..17 hold the rate.
    state: [u64; LANE_COUNT],
    /// The rate block currently being staged.
    block: [u8; RATE_BYTES],
    /// How many leading bytes of `block` hold message data.
    staged: usize,
}

impl Default for Shake256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Shake256 {
    /// A fresh sponge over the empty prefix.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: [0u64; LANE_COUNT],
            block: [0u8; RATE_BYTES],
            staged: 0,
        }
    }

    /// Restores the sponge to [`Self::new`] without reallocating.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Absorbs `bytes`.
    pub fn update(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        if self.staged != 0 {
            let take = (RATE_BYTES - self.staged).min(rest.len());
            let end = self.staged + take;
            self.block[self.staged..end].copy_from_slice(&rest[..take]);
            self.staged = end;
            rest = &rest[take..];
            if self.staged == RATE_BYTES {
                self.absorb_staged();
            }
        }
        let mut blocks = rest.chunks_exact(RATE_BYTES);
        for block in &mut blocks {
            self.absorb(block);
        }
        let tail = blocks.remainder();
        if !tail.is_empty() {
            self.block[..tail.len()].copy_from_slice(tail);
            self.staged = tail.len();
        }
    }

    /// Absorbs a stream of packed elements.
    ///
    /// Equivalent to absorbing the little-endian byte encoding of the same
    /// elements: the rate is a whole number of elements (34) and lanes are
    /// two elements wide, so the journal can fingerprint its `u32` pools
    /// without staging a byte copy of the window.
    pub fn update_words(&mut self, words: &[u32]) {
        if self.staged == 0 {
            let mut blocks = words.chunks_exact(RATE_WORDS);
            for block in &mut blocks {
                self.absorb_words(block);
            }
            self.stage_words(blocks.remainder());
            return;
        }
        self.stage_words(words);
    }

    /// Squeezes `output.len()` bytes of extendable output from the padded
    /// state, leaving the sponge usable for a later call.
    pub fn squeeze(&self, output: &mut [u8]) {
        let mut sponge = *self;
        sponge.finish();
        let mut written = 0;
        while written < output.len() {
            let take = (output.len() - written).min(RATE_BYTES);
            {
                let out = &mut output[written..written + take];
                for (chunk, lane) in out.chunks_mut(8).zip(sponge.state.iter()) {
                    lane_into(*lane, chunk);
                }
            }
            written += take;
            if written < output.len() {
                keccak_f1600(&mut sponge.state);
            }
        }
    }

    /// Absorbs the staged rate block.
    fn absorb_staged(&mut self) {
        let Shake256 { state, block, staged } = self;
        xor_lanes(state, block);
        keccak_f1600(state);
        // The stale tail of `block` is never read: only `staged` bytes are
        // message data, and `finish` zero-builds the block it pads.
        *staged = 0;
    }

    /// Absorbs one full rate block from the caller's buffer.
    #[inline]
    fn absorb(&mut self, block: &[u8]) {
        let Shake256 { state, .. } = self;
        xor_lanes(state, block);
        keccak_f1600(state);
    }

    /// Absorbs one full rate block given as packed elements.
    #[inline]
    fn absorb_words(&mut self, words: &[u32]) {
        let Shake256 { state, .. } = self;
        for (lane, pair) in state.iter_mut().zip(words.chunks_exact(2)) {
            *lane ^= lane_from_words(pair[0], pair[1]);
        }
        keccak_f1600(state);
    }

    /// Stages a sub-block run of elements into `block`.
    fn stage_words(&mut self, words: &[u32]) {
        let staged = self.staged;
        let need = core::mem::size_of_val(words);
        if staged + need <= RATE_BYTES {
            let region = &mut self.block[staged..staged + need];
            for (slot, word) in region.chunks_exact_mut(4).zip(words) {
                slot.copy_from_slice(&word.to_le_bytes());
            }
            self.staged = staged + need;
            return;
        }
        for &word in words {
            self.update(&word.to_le_bytes());
        }
    }

    /// Applies the delimited padding and absorbs the last block.
    fn finish(&mut self) {
        let mut padded = [0u8; RATE_BYTES];
        let staged = self.staged;
        padded[..staged].copy_from_slice(&self.block[..staged]);
        // `staged` is below the rate by construction: `update` absorbs and
        // resets as soon as a block fills, so the two suffix bytes never
        // land on message data.
        pad(&mut padded, staged);
        self.absorb(&padded);
    }
}

/// XORs up to a rate block of bytes into the lanes of `state`.
///
/// A short trailing chunk zero-fills, so a partial block lands exactly
/// as the spec's `pad10*1` requires; `block` is never empty (callers
/// pass whole rate blocks), so `chunks` cannot divide by zero.
#[inline]
fn xor_lanes(state: &mut [u64; LANE_COUNT], block: &[u8]) {
    for (lane, chunk) in state.iter_mut().zip(block.chunks(8)) {
        *lane ^= lane_from(chunk);
    }
}

/// Applies `1111 || pad10*1` to a block whose first `len` bytes are data.
#[inline]
fn pad(block: &mut [u8; RATE_BYTES], len: usize) {
    block[len] ^= DOMAIN_BYTE;
    block[RATE_BYTES - 1] ^= FINAL_BYTE;
}

/// One-shot SHAKE256: fills `output` with the extendable output of `bytes`.
///
/// # Examples
///
/// ```
/// let mut out = [0u8; 32];
/// codevar_wredit::shake256::shake256(b"", &mut out);
/// assert_eq!(&out[..8], &[0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13]);
/// ```
pub fn shake256(bytes: &[u8], output: &mut [u8]) {
    let mut sponge = Shake256::new();
    sponge.update(bytes);
    sponge.squeeze(output);
}

/// SHAKE256 of `bytes`, truncated to [`DIGEST_LEN`].
#[must_use]
pub fn shake256_digest(bytes: &[u8]) -> [u8; DIGEST_LEN] {
    let mut out = [0u8; DIGEST_LEN];
    shake256(bytes, &mut out);
    out
}

/// The 64-bit fingerprint of `bytes`: the first eight output bytes read
/// little-endian.
///
/// Used where the journal only needs a cheap "is this the window I
/// recorded?" test rather than a full digest.
#[must_use]
pub fn shake256_fingerprint(bytes: &[u8]) -> u64 {
    fingerprint_of(&shake256_digest(bytes))
}

/// The 64-bit fingerprint of a packed element stream.
///
/// Equivalent to [`shake256_fingerprint`] over the little-endian bytes of
/// the same elements, without making that copy.
#[must_use]
pub fn shake256_fingerprint_words(words: &[u32]) -> u64 {
    let mut sponge = Shake256::new();
    sponge.update_words(words);
    let mut digest = [0u8; DIGEST_LEN];
    sponge.squeeze(&mut digest);
    fingerprint_of(&digest)
}

/// Reads the leading eight bytes of a digest as a little-endian lane.
#[must_use]
fn fingerprint_of(digest: &[u8; DIGEST_LEN]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Decodes a hex string for the FIPS 202 vectors below.
    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(text.get(i..i + 2).unwrap_or("0"), 16).unwrap_or(0))
            .collect()
    }

    fn repeated(byte: u8, len: usize) -> Vec<u8> {
        alloc::vec![byte; len]
    }

    /// Decodes a hex string into exactly one [`DIGEST_LEN`] digest.
    fn digest_hex(text: &str) -> [u8; DIGEST_LEN] {
        let bytes = unhex(text);
        let mut out = [0u8; DIGEST_LEN];
        let len = bytes.len().min(DIGEST_LEN);
        out[..len].copy_from_slice(&bytes[..len]);
        out
    }

    #[test]
    fn pi_is_a_bijection_of_the_lanes() {
        let mut seen = [false; LANE_COUNT];
        for lane in PI_DST {
            if lane >= LANE_COUNT {
                panic!("pi destination {lane} escapes the state");
            }
            seen[lane] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "pi is not a bijection");
    }

    #[test]
    fn derived_tables_match_fips_202() {
        // Spot-check the flat tables against the specification.
        assert_eq!(RHO_ROT[0], 0); // (0, 0)
        assert_eq!(RHO_ROT[1], 1); // (1, 0)
        assert_eq!(RHO_ROT[2], 62); // (2, 0)
        assert_eq!(RHO_ROT[5], 36); // (0, 1)
        assert_eq!(RHO_ROT[20], 18); // (0, 4)
        assert_eq!(RHO_ROT[24], 14); // (4, 4)
        assert_eq!(PI_DST[0], 0); // (0, 0) -> (0, 0)
        assert_eq!(PI_DST[1], 10); // (1, 0) -> (0, 2)
        assert_eq!(COL[4][2], 14); // lane (4, 2)
    }

    #[test]
    fn shake256_matches_fips_202_vectors() {
        let cases: [(&[u8], &str); 3] = [
            (
                b"",
                "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f",
            ),
            (
                b"a",
                "867e2cb04f5a04dcbd592501a5e8fe9ceaafca50255626ca736c138042530ba4",
            ),
            (
                b"abc",
                "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739",
            ),
        ];
        for (message, expected) in cases {
            assert_eq!(shake256_digest(message), digest_hex(expected));
        }

        let rate = repeated(b'c', RATE_BYTES);
        assert_eq!(
            shake256_digest(&rate),
            digest_hex("98c385d334704a2b6af81f5f70ca16469e6cb17135f3abef05f4dbdc6dd959ca"),
            "a message of exactly one rate block still needs its padded block"
        );

        let rate_minus_one = repeated(b'a', RATE_BYTES - 2);
        assert_eq!(
            shake256_digest(&rate_minus_one),
            digest_hex("f1e03f378eb77904ba15bb642a84b90de52e293eafc27cef05883b1656aec341")
        );

        let rate_plus_one = repeated(b'd', RATE_BYTES + 1);
        assert_eq!(
            shake256_digest(&rate_plus_one),
            digest_hex("c3a1baa9bab9dfa4368e23aecb16ba7f8a71e76dc6f5f841c472975dfafe363f")
        );

        let two_blocks = repeated(b'e', 300);
        assert_eq!(
            shake256_digest(&two_blocks),
            digest_hex("eed5ff4da3a15c5543b0d5764bc9105f1de0b3bfc33ed087e7db9075b81ca6f1")
        );

        let long: Vec<u8> = (0u16..251)
            .map(|v| v as u8)
            .cycle()
            .take(1255)
            .collect();
        assert_eq!(
            shake256_digest(&long),
            digest_hex("d5df60d44342d786f2da67c23b842d8f4fedf1625a772b32535e187a49ddc5ab")
        );
    }

    #[test]
    fn squeeze_crosses_rate_blocks() {
        let expected = unhex(concat!(
            "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f",
            "d75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be",
            "141e96616fb13957692cc7edd0b45ae3dc07223c8e92937bef84bc0eab862853",
            "349ec75546f58fb7c2775c38462c5010d846c185c15111e595522a6bcd16cf86",
            "f3d122109e3b1fdd943b6aec468a2d621a7c06c6a957c62b54dafc3be87567d6",
            "77231395f6147293b68ceab7a9e0c58d864e8efde4e1b9a46cbe854713672f5c",
            "aaae314ed9083dab",
        ));
        let mut out = vec![0u8; 200];
        shake256(b"", &mut out);
        assert_eq!(out, expected);
        assert_eq!(&out[..DIGEST_LEN], &shake256_digest(b"")[..]);
    }

    #[test]
    fn split_updates_match_one_shot() {
        let message = repeated(b'x', 3 * RATE_BYTES + 7);
        for split in [0usize, 1, 8, RATE_BYTES - 1, RATE_BYTES, RATE_BYTES + 1, 300] {
            let mut sponge = Shake256::new();
            sponge.update(&message[..split]);
            sponge.update(&message[split..]);
            let mut out = [0u8; DIGEST_LEN];
            sponge.squeeze(&mut out);
            assert_eq!(out, shake256_digest(&message), "split at {split}");
            let mut again = [0u8; DIGEST_LEN];
            sponge.squeeze(&mut again);
            assert_eq!(out, again, "squeeze must not consume the sponge");
        }
    }

    #[test]
    fn reset_reuses_the_sponge() {
        let mut sponge = Shake256::new();
        sponge.update(b"discarded");
        sponge.reset();
        sponge.update(b"abc");
        let mut out = [0u8; DIGEST_LEN];
        sponge.squeeze(&mut out);
        assert_eq!(out, shake256_digest(b"abc"));
    }

    #[test]
    fn word_stream_equals_its_byte_encoding() {
        let words: Vec<u32> = (0..137u32).map(|i| 0xDE00_0000u32 ^ i).collect();
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(shake256_fingerprint_words(&words), shake256_fingerprint(&bytes));

        // Elements split across calls, including a mid-word boundary, must
        // still agree with the one-shot byte digest.
        let mut sponge = Shake256::new();
        sponge.update_words(&words[..1]);
        sponge.update(&bytes[4..12]);
        sponge.update_words(&words[3..]);
        let mut out = [0u8; DIGEST_LEN];
        sponge.squeeze(&mut out);
        assert_eq!(out, shake256_digest(&bytes));
    }

    #[test]
    fn padding_helper_matches_the_sponge() {
        let mut block = [0u8; RATE_BYTES];
        block[0] = b'a';
        pad(&mut block, 1);
        assert_eq!(block[0], b'a');
        assert_eq!(block[1], DOMAIN_BYTE);
        assert_eq!(block[RATE_BYTES - 1], FINAL_BYTE);
    }
}
