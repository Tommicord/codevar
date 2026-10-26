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

//! # SHAKE256
//!
//! FIPS 202 SHAKE256 extendable-output function, used across Codevar —
//! for example by the Mergen merge algorithm — to derive deterministic
//! content digests.
//!
//! ## Features
//!
//! - One-shot ([`shake256`]) and streaming ([`Shake256`]) APIs
//! - Compile-time evaluation via [`shake256_const`] (`const fn`)
//! - Multi-buffer batch hashing ([`shake256_x2`], [`shake256_x4`],
//!   [`shake256_x8`]): independent digests whose Keccak-f[1600]
//!   permutations run in lockstep on one SIMD backend
//! - Backend dispatch: scalar, SSE2, SSSE3, SSE4.1, AVX2 (4- and
//!   8-lane batches), NEON, WASM SIMD128
//! - Convenience digests: [`content_digest`] (32 bytes) and
//!   [`content_fingerprint`] (8 bytes)
//! - Constant-time helpers in [`ct`]: branchless mask predicates
//!   (`eq`/`lt`/`gt`/…), selection, and byte comparison for callers
//!   such as the Mergen merge algorithm
//!
//! ## Determinism
//!
//! Every backend computes bit-identical output for the same input;
//! backend selection only affects performance.
//!
//! ## Constant-time behavior
//!
//! The Keccak-f[1600] permutation has no data-dependent branches,
//! memory indexing, or rotation amounts, so the hashing functions are
//! constant time with respect to message *contents* (run time still
//! scales with the message *length*, which is public).
//!
//! ## Examples
//!
//! ```
//! let mut out = [0u8; 32];
//! codevar_shake::shake256(b"", &mut out);
//! assert_eq!(
//!     out,
//!     [
//!         0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13, 0x23, 0x3b, 0x3f, 0xeb, 0x74, 0x3e,
//!         0xeb, 0x24, 0x3f, 0xcd, 0x52, 0xea, 0x62, 0xb8, 0x1b, 0x82, 0xb5, 0x0c, 0x27, 0x64,
//!         0x6e, 0xd5, 0x76, 0x2f,
//!     ]
//! );
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

mod keccak;
mod simd;

pub mod ct;

use keccak::{MAX_BATCH, PAD_BYTE, RATE, STATE_LANES};

/// Computes SHAKE256 over `input`, writing the first `output.len()`
/// bytes of the XOF stream into `output`.
///
/// Any output length is accepted; SHAKE256 is an extendable-output
/// function, so longer outputs simply squeeze more blocks.
///
/// # Examples
///
/// ```
/// let mut out = [0u8; 32];
/// codevar_shake::shake256(b"abc", &mut out);
/// let mut reference = [0u8; 32];
/// codevar_shake::shake256_const(b"abc", &mut reference);
/// assert_eq!(out, reference);
/// ```
#[inline]
pub fn shake256(input: &[u8], output: &mut [u8]) {
    let mut hasher = Shake256::new();
    hasher.update(input);
    hasher.finalize();
    hasher.squeeze(output);
}

/// Computes SHAKE256 at compile time.
///
/// A `const fn` over the scalar permutation, so it can initialize
/// `const` items. The result is bit-identical to [`shake256`].
///
/// # Examples
///
/// ```
/// const KEY: [u8; 32] = {
///     let mut out = [0u8; 32];
///     codevar_shake::shake256_const(b"codevar", &mut out);
///     out
/// };
/// let mut runtime = [0u8; 32];
/// codevar_shake::shake256(b"codevar", &mut runtime);
/// assert_eq!(KEY, runtime);
/// ```
#[inline]
pub const fn shake256_const(input: &[u8], output: &mut [u8]) {
    let mut state = [0u64; STATE_LANES];
    let mut offset = 0usize;
    let mut index = 0usize;
    while index < input.len() {
        let lane = offset / 8;
        let shift = (offset % 8) * 8;
        state[lane] ^= (input[index] as u64) << shift;
        offset += 1;
        if offset == RATE {
            keccak::keccakf1600(&mut state);
            offset = 0;
        }
        index += 1;
    }
    // SHAKE256 padding: 0b1111 domain separator at `offset`, 10*1
    // marker in the last byte of the rate block.
    let lane = offset / 8;
    let shift = (offset % 8) * 8;
    state[lane] ^= (PAD_BYTE as u64) << shift;
    state[(RATE - 1) / 8] ^= 0x80u64 << (((RATE - 1) % 8) * 8);
    keccak::keccakf1600(&mut state);

    let mut produced = 0usize;
    let mut block = 0usize;
    while produced < output.len() {
        let take = if RATE - block < output.len() - produced {
            RATE - block
        } else {
            output.len() - produced
        };
        let mut k = 0usize;
        while k < take {
            let absolute = block + k;
            output[produced + k] = ((state[absolute / 8] >> ((absolute % 8) * 8)) & 0xFF) as u8;
            k += 1;
        }
        produced += take;
        block += take;
        if block == RATE {
            keccak::keccakf1600(&mut state);
            block = 0;
        }
    }
}

/// Streaming SHAKE256 state.
///
/// Absorb with [`update`](Self::update), then [`finalize`](Self::finalize)
/// and [`squeeze`](Self::squeeze) any number of output bytes. Calls to
/// `squeeze` auto-finalize when `finalize` has not run yet, so the
/// common one-shot usage never needs an explicit `finalize`.
#[derive(Clone)]
pub struct Shake256 {
    /// Keccak state in FIPS lane order.
    state: [u64; STATE_LANES],
    /// Bytes absorbed into (or squeezed from) the current rate block.
    offset: usize,
    /// Whether padding has been applied and squeezing has begun.
    squeezing: bool,
}

impl Shake256 {
    /// Creates a fresh sponge in the absorbing state.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: [0; STATE_LANES],
            offset: 0,
            squeezing: false,
        }
    }

    /// Absorbs `input` into the sponge.
    ///
    /// Permutations are dispatched through the SIMD batch backend
    /// even for this single state, so the hot path shares code with
    /// the multi-buffer APIs. Calling `update` after
    /// [`finalize`](Self::finalize) is a logic error and the bytes are
    /// silently ignored (this function never panics).
    #[inline]
    pub fn update(&mut self, input: &[u8]) {
        if self.squeezing {
            return;
        }
        let mut index = 0usize;
        while index < input.len() {
            let take = RATE - self.offset;
            let end = core::cmp::min(index + take, input.len());
            keccak::absorb_bytes(&mut self.state, self.offset, &input[index..end]);
            self.offset += end - index;
            index = end;
            if self.offset == RATE {
                self.permute_one();
                self.offset = 0;
            }
        }
    }

    /// Applies the SHAKE256 padding and switches to the squeezing
    /// state. Idempotent.
    #[inline]
    pub fn finalize(&mut self) {
        if self.squeezing {
            return;
        }
        keccak::pad_and_permute(&mut self.state, self.offset);
        self.offset = 0;
        self.squeezing = true;
    }

    /// Writes the next `output.len()` bytes of the XOF stream into
    /// `output`. Auto-finalizes when the sponge is still absorbing.
    #[inline]
    pub fn squeeze(&mut self, output: &mut [u8]) {
        if !self.squeezing {
            self.finalize();
        }
        let mut produced = 0usize;
        while produced < output.len() {
            let take = if RATE - self.offset < output.len() - produced {
                RATE - self.offset
            } else {
                output.len() - produced
            };
            keccak::squeeze_bytes(&self.state, self.offset, &mut output[produced..produced + take]);
            produced += take;
            self.offset += take;
            if self.offset == RATE {
                self.permute_one();
                self.offset = 0;
            }
        }
    }

    /// Resets the sponge to the state [`new`](Self::new) produces.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Permutates the single state through the batched backend.
    #[inline]
    fn permute_one(&mut self) {
        let mut flat = [0u64; STATE_LANES * MAX_BATCH];
        flat[..STATE_LANES].copy_from_slice(&self.state);
        simd::keccakf_batch(&mut flat[..STATE_LANES], 1);
        self.state.copy_from_slice(&flat[..STATE_LANES]);
    }
}

impl Default for Shake256 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Computes two independent SHAKE256 digests whose Keccak-f[1600]
/// permutations run in lockstep on the SIMD backend whenever the two
/// inputs reach a rate boundary together (in particular, for
/// equal-length inputs, on every block).
#[inline]
pub fn shake256_x2(a_input: &[u8], b_input: &[u8], a_output: &mut [u8], b_output: &mut [u8]) {
    shake256_batch([a_input, b_input], [a_output, b_output]);
}

/// Computes four independent SHAKE256 digests in SIMD lockstep; see
/// [`shake256_x2`] for the lockstep conditions.
#[inline]
pub fn shake256_x4(inputs: [&[u8]; 4], outputs: [&mut [u8]; 4]) {
    shake256_batch(inputs, outputs);
}

/// Computes eight independent SHAKE256 digests in SIMD lockstep; see
/// [`shake256_x2`] for the lockstep conditions.
#[inline]
pub fn shake256_x8(inputs: [&[u8]; 8], outputs: [&mut [u8]; 8]) {
    shake256_batch(inputs, outputs);
}

/// 32-byte content digest of `input`, suitable as an opaque change
/// identifier.
#[inline]
#[must_use]
pub fn content_digest(input: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    shake256(input, &mut out);
    out
}

/// 8-byte little-endian fingerprint of [`content_digest`], a compact
/// change identifier for cache keys and block ordering.
#[inline]
#[must_use]
pub fn content_fingerprint(input: &[u8]) -> u64 {
    let digest = content_digest(input);
    u64::from_le_bytes(digest[..8].try_into().unwrap_or([0; 8]))
}

/// Shared implementation of the `LANES`-buffer batch APIs.
///
/// Every lane is an independent sponge; permutations of lanes that
/// hit a rate boundary in the same absorb/squeeze step are gathered
/// into one position-major batch and executed by the SIMD backend.
/// Results are bit-identical to running [`shake256`] per lane.
fn shake256_batch<const LANES: usize>(inputs: [&[u8]; LANES], outputs: [&mut [u8]; LANES]) {
    let mut states = [[0u64; STATE_LANES]; LANES];
    let mut offsets = [0usize; LANES];

    // Absorb: walk every input in lockstep so equal-length buffers
    // permute together on each rate boundary.
    let mut max_len = 0usize;
    for input in &inputs {
        max_len = max_len.max(input.len());
    }
    for index in 0..max_len {
        let mut pending = [false; LANES];
        for lane in 0..LANES {
            if let Some(&byte) = inputs[lane].get(index) {
                let offset = offsets[lane];
                let word = offset / 8;
                let shift = (offset % 8) * 8;
                states[lane][word] ^= (byte as u64) << shift;
                offsets[lane] = offset + 1;
                if offsets[lane] == RATE {
                    pending[lane] = true;
                }
            }
        }
        flush_permutes(&mut states, &mut offsets, &pending);
    }

    // Pad every lane and run the final absorbing permutation.
    let mut pending = [false; LANES];
    for lane in 0..LANES {
        let offset = offsets[lane];
        let word = offset / 8;
        let shift = (offset % 8) * 8;
        states[lane][word] ^= (PAD_BYTE as u64) << shift;
        states[lane][(RATE - 1) / 8] ^= 0x80u64 << (((RATE - 1) % 8) * 8);
        pending[lane] = true;
        offsets[lane] = 0;
    }
    flush_permutes(&mut states, &mut offsets, &pending);

    // Squeeze until every output is full.
    let mut produced = [0usize; LANES];
    loop {
        let mut pending = [false; LANES];
        let mut active = false;
        for lane in 0..LANES {
            let needed = outputs[lane].len();
            if produced[lane] >= needed {
                continue;
            }
            active = true;
            let take = if RATE - offsets[lane] < needed - produced[lane] {
                RATE - offsets[lane]
            } else {
                needed - produced[lane]
            };
            keccak::squeeze_bytes(
                &states[lane],
                offsets[lane],
                &mut outputs[lane][produced[lane]..produced[lane] + take],
            );
            produced[lane] += take;
            offsets[lane] += take;
            if offsets[lane] == RATE {
                pending[lane] = true;
            }
        }
        if !active {
            break;
        }
        flush_permutes(&mut states, &mut offsets, &pending);
    }
}

/// Gathers every lane marked in `pending` into position-major batches
/// of at most [`MAX_BATCH`] lanes, permutes each batch through the
/// SIMD backend, and writes the results back.
fn flush_permutes<const LANES: usize>(
    states: &mut [[u64; STATE_LANES]; LANES],
    offsets: &mut [usize; LANES],
    pending: &[bool; LANES],
) {
    let mut chunk = [0usize; MAX_BATCH];
    let mut filled = 0usize;
    for (lane, &is_pending) in pending.iter().enumerate() {
        if !is_pending {
            continue;
        }
        chunk[filled] = lane;
        filled += 1;
        if filled == MAX_BATCH {
            flush_chunk(states, offsets, &chunk, filled);
            filled = 0;
        }
    }
    if filled > 0 {
        flush_chunk(states, offsets, &chunk, filled);
    }
}

/// Permutes one gathered chunk of `count` lanes and writes it back.
fn flush_chunk<const LANES: usize>(
    states: &mut [[u64; STATE_LANES]; LANES],
    offsets: &mut [usize; LANES],
    chunk: &[usize; MAX_BATCH],
    count: usize,
) {
    if count == 0 || count > MAX_BATCH {
        return;
    }
    let mut flat = [0u64; STATE_LANES * MAX_BATCH];
    for (index, &lane) in chunk.iter().take(count).enumerate() {
        for position in 0..STATE_LANES {
            flat[position * count + index] = states[lane][position];
        }
    }
    simd::keccakf_batch(&mut flat[..STATE_LANES * count], count);
    for (index, &lane) in chunk.iter().take(count).enumerate() {
        for position in 0..STATE_LANES {
            states[lane][position] = flat[position * count + index];
        }
        offsets[lane] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected outputs generated with Python's `hashlib.shake_256`.
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn known_answer_vectors() {
        let mut out = [0u8; 32];
        shake256(b"", &mut out);
        assert_eq!(
            hex(&out),
            "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
        );

        shake256(b"The quick brown fox jumps over the lazy dog", &mut out);
        assert_eq!(
            hex(&out),
            "2f671343d9b2e1604dc9dcf0753e5fe15c7c64a0d283cbbf722d411a0e36f6ca"
        );

        shake256(b"abc", &mut out);
        assert_eq!(
            hex(&out),
            "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739"
        );

        let mut long = [0u8; 64];
        shake256(b"a", &mut long);
        assert_eq!(
            hex(&long),
            "867e2cb04f5a04dcbd592501a5e8fe9ceaafca50255626ca736c138042530ba4\
             36b7b1ec0e06a279bc790733bb0aee6fa802683c7b355063c434e91189b0c651"
        );
    }

    #[test]
    fn const_matches_runtime() {
        let mut runtime = [0u8; 64];
        shake256(b"compile-time vs run-time", &mut runtime);
        let mut constant = [0u8; 64];
        shake256_const(b"compile-time vs run-time", &mut constant);
        assert_eq!(runtime, constant);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let message = b"streaming state must match the one-shot path";
        let mut one_shot = [0u8; 48];
        shake256(message, &mut one_shot);

        let mut hasher = Shake256::new();
        hasher.update(&message[..10]);
        hasher.update(&message[10..25]);
        hasher.update(&message[25..]);
        hasher.finalize();
        let mut streamed = [0u8; 48];
        hasher.squeeze(&mut streamed);
        assert_eq!(one_shot, streamed);

        // Squeeze in odd-sized pieces.
        let mut piecewise = [0u8; 48];
        let mut hasher = Shake256::new();
        hasher.update(message);
        hasher.squeeze(&mut piecewise[..1]);
        hasher.squeeze(&mut piecewise[1..30]);
        hasher.squeeze(&mut piecewise[30..]);
        assert_eq!(one_shot, piecewise);
    }

    #[test]
    fn multi_rate_messages_match_one_shot() {
        // Longer than one rate block (136 bytes) to exercise the
        // absorb-permute-absorb path in every API.
        let long_a = [0xA5u8; 400];
        let long_b = [0x3Cu8; 137];
        let mut reference_a = [0u8; 32];
        let mut reference_b = [0u8; 32];
        shake256(&long_a, &mut reference_a);
        shake256(&long_b, &mut reference_b);

        let mut batch_a = [0u8; 32];
        let mut batch_b = [0u8; 32];
        shake256_x2(&long_a, &long_b, &mut batch_a, &mut batch_b);
        assert_eq!(batch_a, reference_a);
        assert_eq!(batch_b, reference_b);
    }

    #[test]
    fn four_buffer_batch_matches_one_shot() {
        let inputs: [&[u8]; 4] = [b"", b"a", &[7u8; 136], &[9u8; 300]];
        let mut references = [[0u8; 32]; 4];
        for (index, input) in inputs.iter().enumerate() {
            shake256(input, &mut references[index]);
        }

        let mut outs = [[0u8; 32]; 4];
        {
            let mut views: [&mut [u8]; 4] = Default::default();
            for (view, out) in views.iter_mut().zip(outs.iter_mut()) {
                *view = out;
            }
            shake256_x4(inputs, views);
        }
        assert_eq!(outs, references);
    }

    #[test]
    fn batch_apis_match_one_shot() {
        let inputs: [&[u8]; 8] = [
            b"",
            b"a",
            b"ab",
            b"abc",
            &[0u8; 135],
            &[0u8; 136],
            &[0u8; 137],
            b"The quick brown fox jumps over the lazy dog",
        ];
        let mut references = [[0u8; 32]; 8];
        for (index, input) in inputs.iter().enumerate() {
            shake256(input, &mut references[index]);
        }

        let mut outs = [[0u8; 32]; 8];
        {
            let mut views: [&mut [u8]; 8] = Default::default();
            for (view, out) in views.iter_mut().zip(outs.iter_mut()) {
                *view = out;
            }
            shake256_x8(inputs, views);
        }
        assert_eq!(outs, references);
    }

    #[test]
    fn empty_output_is_allowed() {
        let mut out: [u8; 0] = [];
        shake256(b"anything", &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn fingerprint_is_first_eight_digest_bytes() {
        assert_eq!(
            content_fingerprint(b"block"),
            u64::from_le_bytes(
                content_digest(b"block")[..8]
                    .try_into()
                    .unwrap_or([0; 8])
            )
        );
    }

    #[test]
    fn update_after_finalize_is_ignored() {
        let mut hasher = Shake256::new();
        hasher.update(b"first");
        hasher.finalize();
        hasher.update(b"late bytes are dropped");
        let mut out = [0u8; 16];
        hasher.squeeze(&mut out);
        let mut reference = [0u8; 16];
        shake256(b"first", &mut reference);
        assert_eq!(out, reference);
    }
}
