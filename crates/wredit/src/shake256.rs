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
//! One Keccak-f[1600] is a sequential network over 25 lanes, so widening
//! it buys nothing; the SIMD win comes from running four states in
//! lockstep. [`Batch`] holds them interleaved, with lane `i` of stream `k`
//! at flat index `i * 4 + k`, so one 256-bit register covers the same lane
//! of all four streams: theta and chi become single vector ops, iota is a
//! broadcast XOR, and rho+pi collapses to a register re-indexing.
//! [`shake256_digests_x4`] and [`shake256_fingerprints_x4_words`] are the
//! batched entry points a journal scan uses.
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

/// Lanes in one [`Batch`]: [`LANE_COUNT`] per stream.
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

/// Four interleaved Keccak-f[1600] states, one per stream.
///
/// Lane `i` of stream `k` lives at `lanes[i * BATCH_STREAMS + k]`. This is
/// the scalar fallback and the oracle for the vector kernel: it runs the
/// reference permutation once per stream after de-interleaving.
#[inline]
fn keccak_f1600_x4(lanes: &mut [u64; BATCH_LANES]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx2) && has(Feature::Sse41) {
            // SAFETY: the probes above proved AVX2 and SSE4.1, and the
            // same XGETBV check that backs `Feature::Avx2` proved the
            // OS-enabled AVX state. The kernel only touches `lanes`,
            // whose length is the fixed [`BATCH_LANES`].
            return unsafe { x86::keccak_f1600_x4_avx2(lanes) };
        }
    }
    keccak_f1600_x4_scalar(lanes);
}

/// The scalar four-way interleaved permutation.
#[inline]
fn keccak_f1600_x4_scalar(lanes: &mut [u64; BATCH_LANES]) {
    for stream in 0..BATCH_STREAMS {
        let mut state = [0u64; LANE_COUNT];
        for (i, lane) in state.iter_mut().enumerate() {
            *lane = lanes[i * BATCH_STREAMS + stream];
        }
        keccak_f1600(&mut state);
        for (i, lane) in state.iter().enumerate() {
            lanes[i * BATCH_STREAMS + stream] = *lane;
        }
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

/// Four streams parked side by side for one interleaved permutation.
#[derive(Clone, Copy)]
struct Batch {
    lanes: [u64; BATCH_LANES],
}

impl Batch {
    #[must_use]
    const fn new() -> Self {
        Self {
            lanes: [0u64; BATCH_LANES],
        }
    }

    #[inline]
    fn lane(&self, stream: usize, lane: usize) -> u64 {
        self.lanes[lane * BATCH_STREAMS + stream]
    }

    /// XORs up to a rate block of bytes into one stream's lanes.
    ///
    /// The stride of [`BATCH_STREAMS`] walks that stream's lanes only;
    /// `bytes.chunks` bounds the walk to the rate, and a short trailing
    /// chunk zero-fills.
    #[inline]
    fn xor_bytes(&mut self, stream: usize, bytes: &[u8]) {
        for (lane, chunk) in self
            .lanes
            .iter_mut()
            .skip(stream)
            .step_by(BATCH_STREAMS)
            .zip(bytes.chunks(8))
        {
            *lane ^= lane_from(chunk);
        }
    }

    /// XORs one rate block of packed elements into one stream's lanes.
    #[inline]
    fn xor_words(&mut self, stream: usize, words: &[u32]) {
        for (lane, pair) in self
            .lanes
            .iter_mut()
            .skip(stream)
            .step_by(BATCH_STREAMS)
            .zip(words.chunks_exact(2))
        {
            *lane ^= lane_from_words(pair[0], pair[1]);
        }
    }

    /// Advances all four states by one interleaved permutation.
    #[inline]
    fn permute(&mut self) {
        keccak_f1600_x4(&mut self.lanes);
    }

    /// Squeezes a [`DIGEST_LEN`] digest out of one stream.
    ///
    /// The output never crosses a rate block, so no further permutation is
    /// needed and the batch can stay untouched.
    fn digest(&self, stream: usize) -> [u8; DIGEST_LEN] {
        let mut out = [0u8; DIGEST_LEN];
        for (lane, chunk) in out.chunks_mut(8).enumerate() {
            lane_into(self.lane(stream, lane), chunk);
        }
        out
    }
}

/// What a stream can contribute to a [`Batch`].
///
/// Implemented for `&[u8]` and `&[u32]`; both write the same lane
/// positions, so a batched result is bit-identical to hashing each stream
/// alone.
trait Feed {
    /// Whole rate blocks the stream holds.
    fn blocks(&self) -> usize;

    /// XORs rate block `index` into `stream`'s lanes.
    fn absorb_block(&self, batch: &mut Batch, stream: usize, index: usize);

    /// XORs the delimited-padded tail into `stream`'s lanes.
    fn absorb_tail(&self, batch: &mut Batch, stream: usize);

    /// Hashes this stream alone into `out`.
    fn single(&self, out: &mut [u8; DIGEST_LEN]);
}

impl Feed for &[u8] {
    fn blocks(&self) -> usize {
        self.len() / RATE_BYTES
    }

    fn absorb_block(&self, batch: &mut Batch, stream: usize, index: usize) {
        let start = index * RATE_BYTES;
        batch.xor_bytes(stream, &self[start..start + RATE_BYTES]);
    }

    fn absorb_tail(&self, batch: &mut Batch, stream: usize) {
        let tail = self.len() % RATE_BYTES;
        let head = self.len() - tail;
        let mut block = [0u8; RATE_BYTES];
        block[..tail].copy_from_slice(&self[head..]);
        pad(&mut block, tail);
        batch.xor_bytes(stream, &block);
    }

    fn single(&self, out: &mut [u8; DIGEST_LEN]) {
        shake256(self, out);
    }
}

impl Feed for &[u32] {
    fn blocks(&self) -> usize {
        self.len() / RATE_WORDS
    }

    fn absorb_block(&self, batch: &mut Batch, stream: usize, index: usize) {
        let start = index * RATE_WORDS;
        batch.xor_words(stream, &self[start..start + RATE_WORDS]);
    }

    fn absorb_tail(&self, batch: &mut Batch, stream: usize) {
        let words = self.len() % RATE_WORDS;
        let head = self.len() - words;
        let mut block = [0u8; RATE_BYTES];
        for (slot, word) in block[..words * 4]
            .chunks_exact_mut(4)
            .zip(self[head..].iter())
        {
            slot.copy_from_slice(&word.to_le_bytes());
        }
        pad(&mut block, words * 4);
        batch.xor_bytes(stream, &block);
    }

    fn single(&self, out: &mut [u8; DIGEST_LEN]) {
        let mut sponge = Shake256::new();
        sponge.update_words(self);
        sponge.squeeze(out);
    }
}

/// Digests four streams through the interleaved permutation.
///
/// The caller must have checked that all four streams have the same number
/// of rate blocks, otherwise the tails would be padded out of lockstep.
fn batch_digests<T: Feed>(streams: [T; BATCH_STREAMS]) -> [[u8; DIGEST_LEN]; BATCH_STREAMS] {
    let mut batch = Batch::new();
    for index in 0..streams[0].blocks() {
        for (stream, feed) in streams.iter().enumerate() {
            feed.absorb_block(&mut batch, stream, index);
        }
        batch.permute();
    }
    for (stream, feed) in streams.iter().enumerate() {
        feed.absorb_tail(&mut batch, stream);
    }
    batch.permute();
    let mut out = [[0u8; DIGEST_LEN]; BATCH_STREAMS];
    for (stream, digest) in out.iter_mut().enumerate() {
        *digest = batch.digest(stream);
    }
    out
}

/// Four independent scalar sponges — the oracle for [`batch_digests`].
fn scalar_digests<T: Feed>(streams: [T; BATCH_STREAMS]) -> [[u8; DIGEST_LEN]; BATCH_STREAMS] {
    let mut out = [[0u8; DIGEST_LEN]; BATCH_STREAMS];
    for (digest, stream) in out.iter_mut().zip(streams) {
        stream.single(digest);
    }
    out
}

/// Whether every stream is the same length, the batched precondition.
#[must_use]
fn same_length(lengths: [usize; BATCH_STREAMS]) -> bool {
    lengths[1..].iter().all(|len| *len == lengths[0])
}

/// Digests of four byte streams, computed by interleaved permutations.
///
/// Streams of equal length absorb and permute in lockstep, so one vector
/// permutation advances all four. Unequal lengths cannot share a
/// permutation schedule and fall back to four independent scalar sponges
/// — the result is identical either way.
///
/// # Examples
///
/// ```
/// use codevar_wredit::shake256::{shake256_digest, shake256_digests_x4};
///
/// let alpha: &[u8] = b"alpha";
/// let beta: &[u8] = b"alpha";
/// let batch = shake256_digests_x4([alpha, beta, alpha, beta]);
/// assert_eq!(batch[0], shake256_digest(alpha));
/// ```
#[must_use]
pub fn shake256_digests_x4(streams: [&[u8]; BATCH_STREAMS]) -> [[u8; DIGEST_LEN]; BATCH_STREAMS] {
    let mut lengths = [0usize; BATCH_STREAMS];
    for (slot, stream) in lengths.iter_mut().zip(streams) {
        *slot = stream.len();
    }
    if same_length(lengths) {
        return batch_digests(streams);
    }
    scalar_digests(streams)
}

/// Fingerprints of four element streams, one interleaved permutation per
/// 136-byte block.
///
/// This is the shape of a journal scan: closed windows all hold the same
/// number of elements, so a group of four is checked at a fraction of the
/// sequential cost. Unequal lengths take the scalar path.
#[must_use]
pub fn shake256_fingerprints_x4_words(streams: [&[u32]; BATCH_STREAMS]) -> [u64; BATCH_STREAMS] {
    let mut lengths = [0usize; BATCH_STREAMS];
    for (slot, stream) in lengths.iter_mut().zip(streams) {
        *slot = stream.len();
    }
    let digests = if same_length(lengths) {
        batch_digests(streams)
    } else {
        scalar_digests(streams)
    };
    let mut out = [0u64; BATCH_STREAMS];
    for (slot, digest) in out.iter_mut().zip(digests) {
        *slot = fingerprint_of(&digest);
    }
    out
}

/// AVX2 four-way interleaved Keccak-f[1600].
///
/// Each of the 25 state lanes is one `__m256i` holding that lane of the
/// four streams, so the round is 25 vector registers moved through theta,
/// a rotated re-indexing (rho+pi), chi's `andnot`/`xor` and a broadcast
/// iota. Rotations of 64-bit lanes have no variable-form opcode in AVX2,
/// so each is a shift pair; lane 0 (offset 0) is passed through because a
/// shift by the width would zero it.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::{BATCH_LANES, COL, LANE_COUNT, PI_DST, RC, RHO_ROT};

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, _mm256_andnot_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256,
    };

    /// Rotates every 64-bit lane of `value` left by `LEFT` bits.
    ///
    /// AVX2 has no variable-width rotate for 64-bit lanes and the shift
    /// immediates are compile-time operands, so the two halves are passed
    /// as a pair of constants (`LEFT + RIGHT == 64`). Lane `(0, 0)`, whose
    /// rho offset is 0, never reaches this helper: a shift by the width
    /// would produce zero instead of the identity.
    #[inline(always)]
    unsafe fn rot<const LEFT: i32, const RIGHT: i32>(value: __m256i) -> __m256i {
        // SAFETY: pure register arithmetic; the caller proved AVX2.
        unsafe {
            _mm256_or_si256(
                _mm256_slli_epi64::<LEFT>(value),
                _mm256_srli_epi64::<RIGHT>(value),
            )
        }
    }

    /// rho+pi for one lane: rotate by the table offset, then scatter to the
    /// destination lane.
    ///
    /// The offset stays a table lookup (so the kernel cannot drift from
    /// [`RHO_ROT`]) while each arm instantiates a compile-time shift pair;
    /// the loop indices are constants after unrolling.
    #[inline(always)]
    unsafe fn rotate_scatter<const I: usize>(value: __m256i) -> __m256i {
        match RHO_ROT[I] {
            1 => unsafe { rot::<1, 63>(value) },
            2 => unsafe { rot::<2, 62>(value) },
            3 => unsafe { rot::<3, 61>(value) },
            6 => unsafe { rot::<6, 58>(value) },
            8 => unsafe { rot::<8, 56>(value) },
            10 => unsafe { rot::<10, 54>(value) },
            14 => unsafe { rot::<14, 50>(value) },
            15 => unsafe { rot::<15, 49>(value) },
            18 => unsafe { rot::<18, 46>(value) },
            20 => unsafe { rot::<20, 44>(value) },
            21 => unsafe { rot::<21, 43>(value) },
            25 => unsafe { rot::<25, 39>(value) },
            27 => unsafe { rot::<27, 37>(value) },
            28 => unsafe { rot::<28, 36>(value) },
            36 => unsafe { rot::<36, 28>(value) },
            39 => unsafe { rot::<39, 25>(value) },
            41 => unsafe { rot::<41, 23>(value) },
            43 => unsafe { rot::<43, 21>(value) },
            44 => unsafe { rot::<44, 20>(value) },
            45 => unsafe { rot::<45, 19>(value) },
            55 => unsafe { rot::<55, 9>(value) },
            56 => unsafe { rot::<56, 8>(value) },
            61 => unsafe { rot::<61, 3>(value) },
            62 => unsafe { rot::<62, 2>(value) },
            // rho offset 0: the identity, and the only offset the shift
            // pair above cannot express.
            _ => value,
        }
    }

    /// Runs 24 interleaved rounds over a `[u64; 100]` buffer.
    ///
    /// # Safety
    ///
    /// Requires AVX2 (and the OS-enabled AVX state behind it) on the
    /// executing CPU, as proven by the `CPUID` probe in
    /// [`super::keccak_f1600_x4`]. `lanes` is read and written as 25
    /// unaligned 256-bit vectors, which its fixed 800-byte length covers.
    #[target_feature(enable = "avx2")]
    pub unsafe fn keccak_f1600_x4_avx2(lanes: &mut [u64; BATCH_LANES]) {
        // SAFETY: the CPUID probe at the call site proved AVX2; every
        // access below stays inside the 100-element buffer.
        unsafe {
            let base = lanes.as_ptr().cast::<__m256i>();
            let out = lanes.as_mut_ptr().cast::<__m256i>();
            let zero = _mm256_setzero_si256();
            let mut a = [zero; LANE_COUNT];
            let mut b = [zero; LANE_COUNT];
            let mut c = [zero; 5];
            for (i, slot) in a.iter_mut().enumerate() {
                *slot = _mm256_loadu_si256(base.add(i));
            }
            for &iota in RC.iter() {
                for (x, parity) in c.iter_mut().enumerate() {
                    let column = &COL[x];
                    *parity = _mm256_xor_si256(
                        _mm256_xor_si256(a[column[0]], a[column[1]]),
                        _mm256_xor_si256(_mm256_xor_si256(a[column[2]], a[column[3]]), a[column[4]]),
                    );
                }
                for x in 0..5 {
                    let d = _mm256_xor_si256(c[(x + 4) % 5], rot::<1, 63>(c[(x + 1) % 5]));
                    for y in 0..5 {
                        a[x + 5 * y] = _mm256_xor_si256(a[x + 5 * y], d);
                    }
                }
                // rho + pi: rotate every lane by its table offset, then
                // scatter it. Spelling the lane indices as literals lets
                // each rotation compile to a single shift pair.
                macro_rules! rho_pi {
                    ($($i:literal),* $(,)?) => {
                        $(b[PI_DST[$i]] = rotate_scatter::<$i>(a[$i]);)*
                    };
                }
                rho_pi!(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                );
                for y in 0..5 {
                    for x in 0..5 {
                        let row = |column: usize| b[(column % 5) + 5 * y];
                        a[x + 5 * y] = _mm256_xor_si256(row(x), _mm256_andnot_si256(row(x + 1), row(x + 2)));
                    }
                }
                a[0] = _mm256_xor_si256(a[0], _mm256_set1_epi64x(iota as i64));
            }
            for (i, value) in a.iter().enumerate() {
                _mm256_storeu_si256(out.add(i), *value);
            }
        }
    }
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
    fn batch_matches_per_stream_hashes_for_every_length() {
        for len in [0usize, 1, 34, RATE_BYTES - 1, RATE_BYTES, RATE_BYTES + 1, 500] {
            let pools = [
                repeated(b'1', len),
                repeated(b'2', len),
                repeated(b'3', len),
                repeated(b'4', len),
            ];
            let streams: [&[u8]; BATCH_STREAMS] =
                [&pools[0][..], &pools[1][..], &pools[2][..], &pools[3][..]];
            let batch = shake256_digests_x4(streams);
            for (digest, stream) in batch.iter().zip(streams) {
                assert_eq!(*digest, shake256_digest(stream), "equal len {len}");
            }
        }
    }

    #[test]
    fn batch_falls_back_when_lengths_differ() {
        let pools: [Vec<u32>; BATCH_STREAMS] = [
            (0..1u32).collect(),
            (10..12u32).collect(),
            Vec::new(),
            (20..60u32).collect(),
        ];
        let streams: [&[u32]; BATCH_STREAMS] = [&pools[0][..], &pools[1][..], &pools[2][..], &pools[3][..]];
        let batch = shake256_fingerprints_x4_words(streams);
        for (slot, stream) in batch.iter().zip(streams) {
            assert_eq!(*slot, shake256_fingerprint_words(stream));
        }
    }

    #[test]
    fn batch_words_match_scalar_across_block_boundaries() {
        for blocks in [0usize, 1, 2, 5] {
            let len = blocks * RATE_WORDS;
            let pools: [Vec<u32>; BATCH_STREAMS] = [
                (0..len as u32).collect(),
                (0..len as u32).map(|i| i ^ 0x7F00).collect(),
                (0..len as u32)
                    .map(|i| i.wrapping_mul(7))
                    .collect(),
                (0..len as u32).map(|i| !i).collect(),
            ];
            let streams: [&[u32]; BATCH_STREAMS] =
                [&pools[0][..], &pools[1][..], &pools[2][..], &pools[3][..]];
            let batch = shake256_fingerprints_x4_words(streams);
            for (slot, stream) in batch.iter().zip(streams) {
                assert_eq!(*slot, shake256_fingerprint_words(stream), "{blocks} blocks");
            }
        }
    }

    #[test]
    fn interleaved_permutation_equals_four_separate_states() {
        // The oracle every vector kernel is compared against: interleaving
        // must not change any lane.
        let mut batch = Batch::new();
        let mut states = [[0u64; LANE_COUNT]; BATCH_STREAMS];
        for (stream, state) in states.iter_mut().enumerate() {
            for (i, lane) in state.iter_mut().enumerate() {
                *lane = 0x0123_4567_89AB_CDEFu64.rotate_left((i * 7 + stream) as u32);
                batch.lanes[i * BATCH_STREAMS + stream] = *lane;
            }
        }
        keccak_f1600_x4(&mut batch.lanes);
        for (stream, state) in states.iter_mut().enumerate() {
            keccak_f1600(state);
            for (i, lane) in state.iter().enumerate() {
                assert_eq!(batch.lane(stream, i), *lane);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_kernel_matches_the_scalar_interleaved_permutation() {
        use codevar_base::cpuid::{Feature, has};
        if !(has(Feature::Avx2) && has(Feature::Sse41)) {
            return; // the machine cannot run the kernel safely
        }
        // Three round-boundary patterns: zero state, all-ones state and a
        // spread that touches every lane of every stream.
        for seed in 0..3u64 {
            let mut fast = [0u64; BATCH_LANES];
            let mut slow = [0u64; BATCH_LANES];
            for (i, lane) in fast.iter_mut().enumerate() {
                *lane = if seed == 0 {
                    0
                } else if seed == 1 {
                    u64::MAX
                } else {
                    (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                };
            }
            slow.copy_from_slice(&fast);
            // SAFETY: the CPUID probe above proved AVX2 and the AVX
            // enabled state; `fast` is the fixed 100-element buffer the
            // kernel expects.
            unsafe { x86::keccak_f1600_x4_avx2(&mut fast) };
            keccak_f1600_x4_scalar(&mut slow);
            assert_eq!(fast, slow, "seed {seed}");
        }
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
