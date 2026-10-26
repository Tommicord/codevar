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

//! Keccak-f[1600] permutation and the SHAKE256 sponge core.
//!
//! The permutation here is the plain scalar reference: it is a
//! [`const fn`], has no data-dependent branches, memory indexing, or
//! rotation amounts, and therefore runs in constant time with respect
//! to the state contents. The batch helpers ([`keccakf_batch`]) apply
//! the same permutation to several independent states in lockstep over
//! a position-major layout — lane `b` of permutation position `p`
//! lives at `state[p * lanes + b]` — so the SIMD backends and the
//! scalar loop share one memory layout and produce bit-identical
//! results.

/// Number of 64-bit lanes in the Keccak-f[1600] state.
pub const STATE_LANES: usize = 25;

/// Largest batch the lockstep helpers accept (matches the widest SIMD
/// backend, AVX-512, which holds eight 64-bit lanes per vector).
pub const MAX_BATCH: usize = 8;

/// Rate of SHAKE256 in bytes (1088 bits).
pub const RATE: usize = 136;

/// Padding byte appended to the SHAKE256 input (`0b1111` domain
/// separator followed by the first bit of the pad).
pub const PAD_BYTE: u8 = 0x1F;

/// Round constants of Keccak-f[1600] (FIPS 202, table 1).
pub const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rho rotation offsets indexed by lane `i = x + 5 * y` (FIPS 202,
/// table 2).
pub const RHO_OFFSETS: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Applies one round of Keccak-f[1600] to `state` for round `round`.
///
/// Lanes are ordered `A[x, y] = state[x + 5 * y]` exactly as FIPS 202
/// defines them. This is a `const fn` so the sponge core can also run
/// at compile time.
#[inline]
pub const fn keccakf_round(state: &mut [u64; STATE_LANES], round: usize) {
    // θ: column parities and column diffs.
    let mut c = [0u64; 5];
    let mut x = 0;
    while x < 5 {
        c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        x += 1;
    }
    let mut d = [0u64; 5];
    x = 0;
    while x < 5 {
        d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        x += 1;
    }
    x = 0;
    while x < 5 {
        let mut y = 0;
        while y < 5 {
            state[x + 5 * y] ^= d[x];
            y += 1;
        }
        x += 1;
    }

    // ρ and π: B[y, 2x + 3y] = rotl(A[x, y], r[x, y]).
    let mut b = [0u64; STATE_LANES];
    let mut y = 0;
    while y < 5 {
        x = 0;
        while x < 5 {
            let offset = RHO_OFFSETS[x + 5 * y];
            let rotated = state[x + 5 * y].rotate_left(offset);
            b[y + 5 * ((2 * x + 3 * y) % 5)] = rotated;
            x += 1;
        }
        y += 1;
    }

    // χ: A[x, y] = B[x, y] ^ ((~B[x + 1, y]) & B[x + 2, y]).
    y = 0;
    while y < 5 {
        x = 0;
        while x < 5 {
            state[x + 5 * y] = b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
            x += 1;
        }
        y += 1;
    }

    // ι: A[0, 0] ^= RC[round].
    state[0] ^= ROUND_CONSTANTS[round];
}

/// Applies all 24 rounds of Keccak-f[1600] to `state`.
///
/// Constant time with respect to the state contents: the round count,
/// lane indices, and rotation amounts are all fixed at compile time.
/// This is a `const fn`.
#[inline]
pub const fn keccakf1600(state: &mut [u64; STATE_LANES]) {
    let mut round = 0;
    while round < ROUND_CONSTANTS.len() {
        keccakf_round(state, round);
        round += 1;
    }
}

/// XORs `bytes` into `state` starting at byte offset `offset`
/// (little-endian lane order, the FIPS 202 convention).
///
/// Reads past the end of `state` are ignored, so callers may pass a
/// full rate block without pre-checking the state length.
pub fn absorb_bytes(state: &mut [u64], offset: usize, bytes: &[u8]) {
    for (index, &byte) in bytes.iter().enumerate() {
        let absolute = offset + index;
        let lane = absolute / 8;
        let shift = (absolute % 8) * 8;
        if let Some(slot) = state.get_mut(lane) {
            *slot ^= (byte as u64) << shift;
        }
    }
}

/// Writes `out.len()` bytes of the sponge state starting at byte
/// offset `offset` (little-endian lane order). Bytes past the end of
/// `state` are written as zero.
pub fn squeeze_bytes(state: &[u64], offset: usize, out: &mut [u8]) {
    for (index, byte) in out.iter_mut().enumerate() {
        let absolute = offset + index;
        let lane = absolute / 8;
        let shift = (absolute % 8) * 8;
        *byte = state
            .get(lane)
            .map_or(0, |lane_value| ((lane_value >> shift) & 0xFF) as u8);
    }
}

/// Applies the SHAKE256 padding to the in-progress block at byte
/// offset `offset`: XOR the domain byte there, XOR the 10*1 marker
/// into the last byte of the rate block, then permute.
pub fn pad_and_permute(state: &mut [u64; STATE_LANES], offset: usize) {
    let lane = offset / 8;
    let shift = (offset % 8) * 8;
    state[lane] ^= (PAD_BYTE as u64) << shift;
    let last_lane = (RATE - 1) / 8;
    let last_shift = ((RATE - 1) % 8) * 8;
    state[last_lane] ^= 0x80u64 << last_shift;
    keccakf1600(state);
}

/// Applies Keccak-f[1600] to `lanes` independent states stored in the
/// position-major batch layout (`state[p * lanes + b]`).
///
/// `state.len()` must be exactly `STATE_LANES * lanes` and `lanes`
/// must be in `1..=MAX_BATCH`; anything else is a no-op. Each of the
/// 24 rounds walks the permutation positions in a fixed order with no
/// data-dependent control flow, so the whole call is constant time
/// with respect to the state contents.
pub fn keccakf_batch(state: &mut [u64], lanes: usize) {
    if lanes == 0 || lanes > MAX_BATCH || state.len() != STATE_LANES * lanes {
        return;
    }
    let mut b_plane = [0u64; STATE_LANES * MAX_BATCH];
    let mut round = 0;
    while round < ROUND_CONSTANTS.len() {
        keccakf_batch_round(state, &mut b_plane, lanes, round);
        round += 1;
    }
}

/// One batched Keccak-f round over the position-major layout.
fn keccakf_batch_round(state: &mut [u64], b_plane: &mut [u64], lanes: usize, round: usize) {
    // θ: per-lane column parities and diffs.
    for lane in 0..lanes {
        let mut parities = [0u64; 5];
        for x in 0..5 {
            parities[x] = state[x * lanes + lane]
                ^ state[(x + 5) * lanes + lane]
                ^ state[(x + 10) * lanes + lane]
                ^ state[(x + 15) * lanes + lane]
                ^ state[(x + 20) * lanes + lane];
        }
        for x in 0..5 {
            let diff = parities[(x + 4) % 5] ^ parities[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[(x + 5 * y) * lanes + lane] ^= diff;
            }
        }
    }

    // ρ and π.
    for y in 0..5 {
        for x in 0..5 {
            let offset = RHO_OFFSETS[x + 5 * y];
            let dest = y + 5 * ((2 * x + 3 * y) % 5);
            for lane in 0..lanes {
                b_plane[dest * lanes + lane] = state[(x + 5 * y) * lanes + lane].rotate_left(offset);
            }
        }
    }

    // χ.
    for y in 0..5 {
        for x in 0..5 {
            let cur = x + 5 * y;
            let nxt = (x + 1) % 5 + 5 * y;
            let nxt2 = (x + 2) % 5 + 5 * y;
            for lane in 0..lanes {
                let i = cur * lanes + lane;
                let j = nxt * lanes + lane;
                let k = nxt2 * lanes + lane;
                state[i] = b_plane[i] ^ ((!b_plane[j]) & b_plane[k]);
            }
        }
    }

    // ι.
    let rc = ROUND_CONSTANTS[round];
    for slot in &mut state[..lanes] {
        *slot ^= rc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_deterministic_and_diffuses() {
        let mut state = [0u64; STATE_LANES];
        keccakf1600(&mut state);
        assert_ne!(state[0], 0);
        assert_ne!(state[0], 1, "the iota step must have mixed the state");
        let mut again = [0u64; STATE_LANES];
        keccakf1600(&mut again);
        assert_eq!(state, again);
    }

    #[test]
    fn batch_layout_matches_scalar_permutation() {
        let mut single_a = [0u64; STATE_LANES];
        let mut single_b = [0x0123_4567_89AB_CDEFu64; STATE_LANES];
        keccakf1600(&mut single_a);
        keccakf1600(&mut single_b);

        let mut batch = [0u64; STATE_LANES * 2];
        for p in 0..STATE_LANES {
            batch[p * 2] = 0;
            batch[p * 2 + 1] = 0x0123_4567_89AB_CDEF;
        }
        keccakf_batch(&mut batch, 2);
        for p in 0..STATE_LANES {
            assert_eq!(batch[p * 2], single_a[p], "lane a at position {p}");
            assert_eq!(batch[p * 2 + 1], single_b[p], "lane b at position {p}");
        }
    }

    #[test]
    fn absorb_and_squeeze_roundtrip_bytes() {
        let mut state = [0u64; STATE_LANES];
        absorb_bytes(&mut state, 0, b"abc");
        let mut out = [0u8; 3];
        squeeze_bytes(&state, 0, &mut out);
        assert_eq!(&out, b"abc");
    }
}
