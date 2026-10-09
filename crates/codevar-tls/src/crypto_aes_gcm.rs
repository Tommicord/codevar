//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! AES-128/256-GCM (NIST SP 800-38D) with constant-time software paths and
//! AES-NI / PCLMULQDQ hardware acceleration.
//!
//! # Side-channel hardening
//!
//! The software path is branch-free with respect to secret data:
//!
//! * [`gf_mul`] (GHASH multiplication in GF(2^128)) selects the running value
//!   with a masking XOR instead of a secret-dependent branch.
//! * S-box substitution uses [`lookup256`], a 256-entry scan whose memory
//!   access pattern does not depend on the (secret) state byte.
//! * `xtime` and the GCM counter increment use masking / wrapping arithmetic.
//! * Tag comparison uses branch-free equality ([`ct_eq`]).
//!
//! The hardware paths use AES-NI (`AESENC`) and PCLMULQDQ (`CLMUL`) which are
//! inherently data-independent; key expansion via `AESKEYGENASSIST` matches the
//! software FIPS-197 schedule exactly (verified by tests).
//!
//! # Performance
//!
//! * AES-NI: one block encrypt in ~10 `AESENC` rounds; key expansion 10/14
//!   `AESKEYGENASSIST` steps.
//! * PCLMUL GHASH: 3 `CLMUL` (Karatsuba) + 3 `CLMUL` (reduction) per block,
//!   with a bit-reversal within each byte to switch between the GCM bit order
//!   and the CLMUL polynomial order.
//! * Software fallback: table-driven AES with constant-time table scans;
//!   intended for targets without AES-NI where ChaCha20-Poly1305 is preferred
//!   for traffic anyway (see [`crate::crypto_device`]).

use crate::crypto_ct::{ct_eq, lookup256, mask_u8, mask_u128};
use crate::crypto_device::capabilities;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const BLOCK: usize = 16;
const TAG_LEN: usize = 16;

/// Seals with AES-GCM. `key` must be 16 (AES-128) or 32 (AES-256) bytes.
///
/// Returns `ciphertext || tag` where `tag` is 16 bytes.
/// The 96-bit `nonce` is the standard TLS/GCM invocation nonce.
///
/// # Errors
///
/// Returns `Err(())` when `key` is not 16 or 32 bytes long.
#[allow(clippy::result_unit_err)]
pub fn seal(key: &[u8], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, ()> {
    let aes = AesKey::new(key)?;
    let mut out = vec![0u8; plaintext.len() + TAG_LEN];
    let (ct, tag_slot) = out.split_at_mut(plaintext.len());
    let tag = gcm_seal(&aes, nonce, aad, plaintext, ct)?;
    tag_slot.copy_from_slice(&tag);
    Ok(out)
}

/// Opens ciphertext||tag produced by [`seal`].
///
/// The authenticator is verified with a branch-free comparison before any
/// plaintext is released (encrypt-then-MAC order is enforced by returning
/// `Err` on any mismatch).
///
/// # Errors
///
/// Returns `Err(())` when the key length is invalid, the input is shorter
/// than the 16-byte tag, or authentication fails.
#[allow(clippy::result_unit_err)]
pub fn open(key: &[u8], nonce: &[u8; 12], aad: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, ()> {
    if ciphertext.len() < TAG_LEN {
        return Err(());
    }
    let aes = AesKey::new(key)?;
    let (ct, tag) = ciphertext.split_at(ciphertext.len() - TAG_LEN);
    let mut plain = vec![0u8; ct.len()];
    gcm_open(&aes, nonce, aad, ct, tag, &mut plain)?;
    Ok(plain)
}

#[derive(Zeroize, ZeroizeOnDrop)]
struct AesKey {
    /// Round keys as 16-byte blocks (Nb*(Nr+1)).
    round_keys: Vec<[u8; 16]>,
    nr: usize,
}

impl AesKey {
    fn new(key: &[u8]) -> Result<Self, ()> {
        let nr = match key.len() {
            16 => 10,
            32 => 14,
            _ => return Err(()),
        };
        Ok(Self {
            round_keys: expand_key_dispatch(key, nr),
            nr,
        })
    }

    fn encrypt_block(&self, input: &[u8; 16], output: &mut [u8; 16]) {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if capabilities().aes_hw {
                // SAFETY: aes_hw was probed via is_x86_feature_detected!("aes").
                unsafe {
                    aes_encrypt_block_ni(self, input, output);
                }
                return;
            }
        }
        aes_encrypt_block_soft(self, input, output);
    }
}

/// Expands the key with AES-NI when the AES feature is present, otherwise with
/// the constant-time software schedule.
fn expand_key_dispatch(key: &[u8], nr: usize) -> Vec<[u8; 16]> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if capabilities().aes_hw {
            // SAFETY: `aes_hw` is set from is_x86_feature_detected!("aes"),
            // which is exactly the feature `expand_key_ni` is gated on.
            // `AesKey::new` only reaches here with 16- or 32-byte keys.
            return unsafe { expand_key_ni(key) };
        }
    }
    expand_key(key, nr)
}

fn gcm_seal(
    aes: &AesKey,
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
    ciphertext: &mut [u8],
) -> Result<[u8; 16], ()> {
    if ciphertext.len() != plaintext.len() {
        return Err(());
    }
    let mut h = [0u8; 16];
    aes.encrypt_block(&[0u8; 16], &mut h);
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let mut counter = j0;
    ctr32_inc(&mut counter);
    gctr(aes, &counter, plaintext, ciphertext);

    let s = ghash(&h, aad, ciphertext);
    let mut tag = [0u8; 16];
    let mut j0_enc = [0u8; 16];
    aes.encrypt_block(&j0, &mut j0_enc);
    for i in 0..16 {
        tag[i] = s[i] ^ j0_enc[i];
    }
    Ok(tag)
}

fn gcm_open(
    aes: &AesKey,
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8],
    plaintext: &mut [u8],
) -> Result<(), ()> {
    if plaintext.len() != ciphertext.len() || tag.len() != TAG_LEN {
        return Err(());
    }
    let mut h = [0u8; 16];
    aes.encrypt_block(&[0u8; 16], &mut h);
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let s = ghash(&h, aad, ciphertext);
    let mut expected = [0u8; 16];
    let mut j0_enc = [0u8; 16];
    aes.encrypt_block(&j0, &mut j0_enc);
    for i in 0..16 {
        expected[i] = s[i] ^ j0_enc[i];
    }
    if !ct_eq(&expected, tag) {
        return Err(());
    }
    let mut counter = j0;
    ctr32_inc(&mut counter);
    gctr(aes, &counter, ciphertext, plaintext);
    Ok(())
}

fn gctr(aes: &AesKey, icb: &[u8; 16], input: &[u8], output: &mut [u8]) {
    let mut cb = *icb;
    let mut offset = 0;
    while offset < input.len() {
        let mut keystream = [0u8; 16];
        aes.encrypt_block(&cb, &mut keystream);
        let n = (input.len() - offset).min(BLOCK);
        for i in 0..n {
            output[offset + i] = input[offset + i] ^ keystream[i];
        }
        ctr32_inc(&mut cb);
        offset += n;
    }
}

/// Increments the low 32 bits of the counter block (GCM mod 2^32) with
/// wrapping arithmetic only — no secret-dependent carry branch.
#[inline(always)]
fn ctr32_inc(block: &mut [u8; 16]) {
    let ctr = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);
    block[12..16].copy_from_slice(&ctr.wrapping_add(1).to_be_bytes());
}

/// GHASH over `aad || ciphertext || len(aad)||len(ciphertext)`.
///
/// Dispatches to PCLMULQDQ when available, otherwise to the branch-free
/// software implementation.
fn ghash(h: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if capabilities().clmul_hw {
            // SAFETY: clmul_hw was probed via is_x86_feature_detected!("pclmulqdq").
            return unsafe { ghash_clmul(h, aad, ciphertext) };
        }
    }
    ghash_soft(h, aad, ciphertext)
}

fn ghash_soft(h: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut y = [0u8; 16];
    ghash_update(&mut y, h, aad);
    ghash_update(&mut y, h, ciphertext);

    let mut len_block = [0u8; 16];
    let aad_bits = (aad.len() as u64).wrapping_mul(8);
    let ct_bits = (ciphertext.len() as u64).wrapping_mul(8);
    len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
    len_block[8..].copy_from_slice(&ct_bits.to_be_bytes());
    xor_block(&mut y, &len_block);
    gf_mul(&mut y, h);
    y
}

fn ghash_update(y: &mut [u8; 16], h: &[u8; 16], data: &[u8]) {
    let mut offset = 0;
    while offset < data.len() {
        let mut block = [0u8; 16];
        let n = (data.len() - offset).min(16);
        block[..n].copy_from_slice(&data[offset..offset + n]);
        xor_block(y, &block);
        gf_mul(y, h);
        offset += n;
    }
}

#[inline(always)]
fn xor_block(a: &mut [u8; 16], b: &[u8; 16]) {
    for i in 0..16 {
        a[i] ^= b[i];
    }
}

/// Multiplication in GF(2^128) with the GCM reduction polynomial.
///
/// Implements NIST SP 800-38D §6.3 as straight-line code: the conditional XOR
/// and the conditional reduction are selected with masks instead of branches,
/// so the control flow and memory access pattern are independent of `x` and
/// `y`. O(length) in the bit index with a fixed 128-iteration loop.
fn gf_mul(x: &mut [u8; 16], y: &[u8; 16]) {
    let mut z = 0u128;
    let mut v = u128::from_be_bytes(*y);
    for i in 0..128usize {
        let bit = u128::from((x[i / 8] >> (7 - (i % 8))) & 1);
        z ^= v & mask_u128(bit);
        // v = v * x  (right shift in the GCM bit order) with conditional
        // reduction by 0xE1 when the shifted-out bit was set.
        let reduce = mask_u128(v & 1);
        v >>= 1;
        v ^= (u128::from(0xE1u8) << 120) & reduce;
    }
    *x = z.to_be_bytes();
}

static SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca,
    0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd,
    0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23,
    0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a,
    0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20,
    0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d,
    0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38,
    0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46,
    0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3,
    0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4,
    0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
    0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86,
    0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55,
    0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
    0x16,
];

static RCON: [u8; 11] = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// FIPS-197 key expansion in software.
///
/// Every S-box access goes through [`lookup256`] and every branch depends only
/// on the key *length* (public), never on key material.
fn expand_key(key: &[u8], nr: usize) -> Vec<[u8; 16]> {
    let nk = key.len() / 4;
    let total_words = 4 * (nr + 1);
    let mut w = vec![0u32; total_words];
    for i in 0..nk {
        w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }
    for i in nk..total_words {
        let mut temp = w[i - 1];
        if i % nk == 0 {
            temp = sub_word(rot_word(temp)) ^ (u32::from(RCON[i / nk]) << 24);
        } else if nk > 6 && i % nk == 4 {
            temp = sub_word(temp);
        }
        w[i] = w[i - nk] ^ temp;
    }
    let mut round_keys = Vec::with_capacity(nr + 1);
    for r in 0..=nr {
        let mut block = [0u8; 16];
        for c in 0..4 {
            block[c * 4..c * 4 + 4].copy_from_slice(&w[r * 4 + c].to_be_bytes());
        }
        round_keys.push(block);
    }
    w.zeroize();
    round_keys
}

#[inline(always)]
fn rot_word(x: u32) -> u32 {
    x.rotate_left(8)
}

#[inline(always)]
fn sub_word(x: u32) -> u32 {
    let b = x.to_be_bytes();
    u32::from_be_bytes([
        lookup256(&SBOX, b[0]),
        lookup256(&SBOX, b[1]),
        lookup256(&SBOX, b[2]),
        lookup256(&SBOX, b[3]),
    ])
}

fn aes_encrypt_block_soft(key: &AesKey, input: &[u8; 16], output: &mut [u8; 16]) {
    let mut state = *input;
    add_round_key(&mut state, &key.round_keys[0]);
    for round in 1..key.nr {
        sub_bytes(&mut state);
        shift_rows(&mut state);
        mix_columns(&mut state);
        add_round_key(&mut state, &key.round_keys[round]);
    }
    sub_bytes(&mut state);
    shift_rows(&mut state);
    add_round_key(&mut state, &key.round_keys[key.nr]);
    *output = state;
}

#[inline(always)]
fn add_round_key(state: &mut [u8; 16], rk: &[u8; 16]) {
    for i in 0..16 {
        state[i] ^= rk[i];
    }
}

#[inline(always)]
fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = lookup256(&SBOX, *b);
    }
}

#[inline(always)]
fn shift_rows(state: &mut [u8; 16]) {
    // row 1
    let t = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = t;
    // row 2
    let t0 = state[2];
    let t1 = state[6];
    state[2] = state[10];
    state[6] = state[14];
    state[10] = t0;
    state[14] = t1;
    // row 3
    let t = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = t;
}

/// Multiply by x in GF(2^8) — branch-free via a mask instead of `if high`.
#[inline(always)]
fn xtime(a: u8) -> u8 {
    (a << 1) ^ (0x1b & mask_u8(u8::from(a & 0x80 != 0)))
}

fn mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let i = c * 4;
        let a0 = state[i];
        let a1 = state[i + 1];
        let a2 = state[i + 2];
        let a3 = state[i + 3];
        let t = a0 ^ a1 ^ a2 ^ a3;
        state[i] ^= t ^ xtime(a0 ^ a1);
        state[i + 1] ^= t ^ xtime(a1 ^ a2);
        state[i + 2] ^= t ^ xtime(a2 ^ a3);
        state[i + 3] ^= t ^ xtime(a3 ^ a0);
    }
}

/// Broadcasts word 3 of an `AESKEYGENASSIST` result and XORs it onto the
/// prefix-XOR ("chain") of `prev` — the FIPS-197 schedule step for AES-128
/// and for the even-numbered AES-256 round keys.
///
/// The round constant is a const generic because `AESKEYGENASSIST` only
/// accepts compile-time immediates.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn chain_ff<const IMM: i32>(prev: __m128i) -> __m128i {
    // SAFETY: caller guarantees the AES feature (see `expand_key_ni`).
    let t = _mm_aeskeygenassist_si128(prev, IMM);
    let mut k = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    _mm_xor_si128(k, _mm_shuffle_epi32(t, 0xFF))
}

/// Same as [`chain_ff`] but broadcasts word 2 (plain `SubWord`, no rotation
/// or round constant) of `next` — the AES-256 odd-numbered round-key step.
/// The prefix-XOR chain starts from `prev` (= `rk[n-2]`).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn chain_aa<const IMM: i32>(prev: __m128i, next: __m128i) -> __m128i {
    // SAFETY: caller guarantees the AES feature (see `expand_key_ni`).
    let t = _mm_aeskeygenassist_si128(next, IMM);
    let mut k = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    _mm_xor_si128(k, _mm_shuffle_epi32(t, 0xAA))
}

/// Like [`chain_ff`] but the assist input is `next` (= `rk[n-1]`) while the
/// prefix-XOR chain starts from `prev` (= `rk[n-2]`) — AES-256 even-numbered
/// round keys, where `RotWord(SubWord(rk[n-1].w3)) ^ Rcon` spans two keys.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn chain_ff_from<const IMM: i32>(prev: __m128i, next: __m128i) -> __m128i {
    // SAFETY: caller guarantees the AES feature (see `expand_key_ni`).
    let t = _mm_aeskeygenassist_si128(next, IMM);
    let mut k = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    _mm_xor_si128(k, _mm_shuffle_epi32(t, 0xFF))
}

/// Expands a 16- or 32-byte AES key with `AESKEYGENASSIST`.
///
/// `AESKEYGENASSIST` has a const immediate operand, so each round is written
/// out with a literal round constant. The schedule is algebraically identical
/// to [`expand_key`] (verified by test on both key sizes).
///
/// # Safety
///
/// The caller must ensure the AES-NI feature is present
/// (`is_x86_feature_detected!("aes")`) and that `key` is 16 or 32 bytes.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "aes")]
unsafe fn expand_key_ni(key: &[u8]) -> Vec<[u8; 16]> {
    debug_assert!(matches!(key.len(), 16 | 32));
    let mut round_keys: Vec<[u8; 16]> = Vec::with_capacity(if key.len() == 16 { 11 } else { 15 });
    // SAFETY: feature + key length are the documented preconditions.
    unsafe {
        let mut push = |k: __m128i| {
            let mut b = [0u8; 16];
            _mm_storeu_si128(b.as_mut_ptr().cast::<__m128i>(), k);
            round_keys.push(b);
        };
        if key.len() == 16 {
            let k0 = _mm_loadu_si128(key.as_ptr().cast::<__m128i>());
            push(k0);
            let k1 = chain_ff::<0x01>(k0);
            push(k1);
            let k2 = chain_ff::<0x02>(k1);
            push(k2);
            let k3 = chain_ff::<0x04>(k2);
            push(k3);
            let k4 = chain_ff::<0x08>(k3);
            push(k4);
            let k5 = chain_ff::<0x10>(k4);
            push(k5);
            let k6 = chain_ff::<0x20>(k5);
            push(k6);
            let k7 = chain_ff::<0x40>(k6);
            push(k7);
            let k8 = chain_ff::<0x80>(k7);
            push(k8);
            let k9 = chain_ff::<0x1b>(k8);
            push(k9);
            let k10 = chain_ff::<0x36>(k9);
            push(k10);
        } else {
            let k0 = _mm_loadu_si128(key.as_ptr().cast::<__m128i>());
            push(k0);
            let k1 = _mm_loadu_si128(key.as_ptr().add(16).cast::<__m128i>());
            push(k1);
            // rk[n] = chain4(rk[n-2]) ^ assist(rk[n-1])
            let k2 = chain_ff_from::<0x01>(k0, k1);
            push(k2);
            let k3 = chain_aa::<0x00>(k1, k2);
            push(k3);
            let k4 = chain_ff_from::<0x02>(k2, k3);
            push(k4);
            let k5 = chain_aa::<0x00>(k3, k4);
            push(k5);
            let k6 = chain_ff_from::<0x04>(k4, k5);
            push(k6);
            let k7 = chain_aa::<0x00>(k5, k6);
            push(k7);
            let k8 = chain_ff_from::<0x08>(k6, k7);
            push(k8);
            let k9 = chain_aa::<0x00>(k7, k8);
            push(k9);
            let k10 = chain_ff_from::<0x10>(k8, k9);
            push(k10);
            let k11 = chain_aa::<0x00>(k9, k10);
            push(k11);
            let k12 = chain_ff_from::<0x20>(k10, k11);
            push(k12);
            let k13 = chain_aa::<0x00>(k11, k12);
            push(k13);
            let k14 = chain_ff_from::<0x40>(k12, k13);
            push(k14);
        }
    }
    round_keys
}

/// Encrypts one block with `AESENC`.
///
/// # Safety
///
/// The caller must ensure the AES-NI feature is present; `round_keys` must
/// contain `nr + 1` blocks.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "aes")]
unsafe fn aes_encrypt_block_ni(key: &AesKey, input: &[u8; 16], output: &mut [u8; 16]) {
    // SAFETY: callers ensure AES-NI is present; unaligned load/store are valid for any
    // 16-byte buffer; `round_keys` has length `nr + 1`.
    unsafe {
        let mut state = _mm_loadu_si128(input.as_ptr().cast::<__m128i>());
        state = _mm_xor_si128(state, _mm_loadu_si128(key.round_keys[0].as_ptr().cast()));
        for round in 1..key.nr {
            state = _mm_aesenc_si128(state, _mm_loadu_si128(key.round_keys[round].as_ptr().cast()));
        }
        state = _mm_aesenclast_si128(state, _mm_loadu_si128(key.round_keys[key.nr].as_ptr().cast()));
        _mm_storeu_si128(output.as_mut_ptr().cast::<__m128i>(), state);
    }
}

/// Reverses the bit order inside every byte (an involution).
///
/// GCM stores the coefficient of x^0 as the first (most significant) bit of
/// byte 0, while `CLMUL` treats bit 0 of the least-significant byte as x^0 —
/// exactly a bit reversal within each byte, with byte order preserved.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn bitrev_bytes(v: __m128i) -> __m128i {
    // SAFETY: SSE2 is available (pclmulqdq implies sse2; x86_64 has it at baseline).
    // The three swap steps XOR the bit index with 1, 2 and 4, i.e. bit-reverse
    // the low three bits of every index — a reversal inside each byte that
    // never moves bits across a byte boundary. Shift counts are compile-time
    // immediates, so they are written out per step.
    let v = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi16(0x5555)),
        _mm_and_si128(_mm_slli_epi16(v, 1), _mm_set1_epi16(0xAAAAu16 as i16)),
    );
    let v = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi16(v, 2), _mm_set1_epi16(0x3333)),
        _mm_and_si128(_mm_slli_epi16(v, 2), _mm_set1_epi16(0xCCCCu16 as i16)),
    );
    _mm_or_si128(
        _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi16(0x0F0F)),
        _mm_and_si128(_mm_slli_epi16(v, 4), _mm_set1_epi16(0xF0F0u16 as i16)),
    )
}

/// Full 256-bit carry-less product via Karatsuba (3 `CLMUL`) -> `(hi, lo)`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul256(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    // SAFETY: caller guarantees PCLMULQDQ.
    let v0 = _mm_clmulepi64_si128(a, b, 0x00);
    let v1 = _mm_clmulepi64_si128(a, b, 0x11);
    let v2 = _mm_clmulepi64_si128(
        _mm_xor_si128(a, _mm_srli_si128(a, 8)),
        _mm_xor_si128(b, _mm_srli_si128(b, 8)),
        0x00,
    );
    let mid = _mm_xor_si128(_mm_xor_si128(v0, v1), v2);
    let lo = _mm_xor_si128(v0, _mm_slli_si128(mid, 8));
    let hi = _mm_xor_si128(v1, _mm_srli_si128(mid, 8));
    (hi, lo)
}

/// Reduces a 256-bit product modulo x^128 + x^7 + x^2 + x + 1.
///
/// Splits the high half into 64-bit limbs, multiplies each by 0x87 (the
/// reduced polynomial), folds, then folds the ≤7-bit remainder once more
/// with a final `CLMUL` by 0x87.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn reduce256(lo: __m128i, hi: __m128i) -> __m128i {
    // SAFETY: caller guarantees PCLMULQDQ.
    let r = _mm_set1_epi64x(0x87);
    let t0 = _mm_clmulepi64_si128(hi, r, 0x00);
    let t1 = _mm_clmulepi64_si128(hi, r, 0x11);
    let p_lo = _mm_xor_si128(t0, _mm_slli_si128(t1, 8));
    let p_hi = _mm_srli_si128(t1, 8);
    let p2 = _mm_clmulepi64_si128(p_hi, r, 0x00);
    _mm_xor_si128(lo, _mm_xor_si128(p_lo, p2))
}

/// One GHASH multiplication in the CLMUL domain.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn gfmul_clmul(x: __m128i, y: __m128i) -> __m128i {
    // SAFETY: caller guarantees PCLMULQDQ.
    unsafe {
        let (hi, lo) = clmul256(x, y);
        reduce256(lo, hi)
    }
}

/// GHASH using PCLMULQDQ, accumulating in the natural polynomial domain.
///
/// # Safety
///
/// The caller must ensure the PCLMULQDQ feature is present.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn ghash_clmul(h: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    // SAFETY: feature preconditions documented on this function; unaligned
    // loads/stores of 16-byte buffers are valid for any alignment.
    unsafe {
        let h_nat = bitrev_bytes(_mm_loadu_si128(h.as_ptr().cast::<__m128i>()));
        let mut y = _mm_setzero_si128();
        y = ghash_blocks_clmul(y, h_nat, aad);
        y = ghash_blocks_clmul(y, h_nat, ciphertext);

        let mut len_block = [0u8; 16];
        let aad_bits = (aad.len() as u64).wrapping_mul(8);
        let ct_bits = (ciphertext.len() as u64).wrapping_mul(8);
        len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
        len_block[8..].copy_from_slice(&ct_bits.to_be_bytes());
        let len_nat = bitrev_bytes(_mm_loadu_si128(len_block.as_ptr().cast::<__m128i>()));
        y = gfmul_clmul(_mm_xor_si128(y, len_nat), h_nat);

        let mut out = [0u8; 16];
        _mm_storeu_si128(out.as_mut_ptr().cast::<__m128i>(), bitrev_bytes(y));
        out
    }
}

/// Folds `data` (padded with zeros) into GHASH state `y` in the CLMUL domain.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn ghash_blocks_clmul(mut y: __m128i, h_nat: __m128i, data: &[u8]) -> __m128i {
    // SAFETY: feature preconditions; loads are bounded by `data.len()`.
    unsafe {
        let mut offset = 0;
        while offset < data.len() {
            let mut block = [0u8; 16];
            let n = (data.len() - offset).min(16);
            block[..n].copy_from_slice(&data[offset..offset + n]);
            let b = bitrev_bytes(_mm_loadu_si128(block.as_ptr().cast::<__m128i>()));
            y = gfmul_clmul(_mm_xor_si128(y, b), h_nat);
            offset += n;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG so hardware/software comparisons are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn fill(&mut self, out: &mut [u8]) {
            for chunk in out.chunks_mut(8) {
                let bytes = self.next().to_le_bytes();
                let n = chunk.len();
                chunk.copy_from_slice(&bytes[..n]);
            }
        }
    }

    /// One NIST SP 800-38D known-answer vector:
    /// (nonce, key, aad, plaintext, ciphertext, tag).
    type GcmVector = ([u8; 12], Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    /// NIST SP 800-38D / GCM known-answer vectors.
    fn nist_vectors() -> Vec<GcmVector> {
        let pt_record: Vec<u8> = b"TLS 1.3 record payload - constant time test!!".to_vec();
        vec![
            (
                [0u8; 12],
                vec![0u8; 16],
                vec![],
                vec![],
                vec![],
                vec![
                    0x58, 0xe2, 0xfc, 0xce, 0xfa, 0x7e, 0x30, 0x61, 0x36, 0x7f, 0x1d, 0x57, 0xa4, 0xe7, 0x45,
                    0x5a,
                ],
            ),
            (
                [0u8; 12],
                vec![0u8; 16],
                vec![],
                vec![0u8; 16],
                vec![
                    0x03, 0x88, 0xda, 0xce, 0x60, 0xb6, 0xa3, 0x92, 0xf3, 0x28, 0xc2, 0xb9, 0x71, 0xb2, 0xfe,
                    0x78,
                ],
                vec![
                    0xab, 0x6e, 0x47, 0xd4, 0x2c, 0xec, 0x13, 0xbd, 0xf5, 0x3a, 0x67, 0xb2, 0x12, 0x57, 0xbd,
                    0xdf,
                ],
            ),
            (
                [0u8; 12],
                vec![0u8; 32],
                vec![],
                vec![],
                vec![],
                vec![
                    0x53, 0x0f, 0x8a, 0xfb, 0xc7, 0x45, 0x36, 0xb9, 0xa9, 0x63, 0xb4, 0xf1, 0xc4, 0xcb, 0x73,
                    0x8b,
                ],
            ),
            (
                [0u8; 12],
                vec![0u8; 32],
                vec![],
                vec![0u8; 16],
                vec![
                    0xce, 0xa7, 0x40, 0x3d, 0x4d, 0x60, 0x6b, 0x6e, 0x07, 0x4e, 0xc5, 0xd3, 0xba, 0xf3, 0x9d,
                    0x18,
                ],
                vec![
                    0xd0, 0xd1, 0xc8, 0xa7, 0x99, 0x99, 0x6b, 0xf0, 0x26, 0x5b, 0x98, 0xb5, 0xd4, 0x8a, 0xb9,
                    0x19,
                ],
            ),
            (
                [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                (0u8..32).collect::<Vec<u8>>(),
                vec![0x17, 0x03, 0x03, 0x00, 0x30],
                pt_record.clone(),
                vec![
                    0x13, 0x4e, 0x85, 0x3b, 0xf4, 0xcb, 0xf1, 0x3b, 0xff, 0x24, 0xf4, 0xe4, 0xc3, 0x8d, 0x58,
                    0x1d, 0xe2, 0xaf, 0xeb, 0x5b, 0x91, 0x1f, 0x7f, 0x51, 0x18, 0x04, 0x8a, 0xeb, 0x6e, 0x1d,
                    0x61, 0xdc, 0x75, 0x30, 0xda, 0x95, 0xc2, 0xa4, 0x32, 0xec, 0x11, 0xd7, 0x0b, 0xcc, 0xa9,
                ],
                vec![
                    0x1a, 0x37, 0x21, 0x32, 0x12, 0x61, 0x6e, 0x39, 0xa2, 0x25, 0xb0, 0x75, 0x6c, 0x18, 0x86,
                    0x78,
                ],
            ),
            (
                [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                (0u8..16).collect::<Vec<u8>>(),
                vec![0x17, 0x03, 0x03, 0x00, 0x30],
                pt_record,
                vec![
                    0xc7, 0x20, 0xf4, 0xee, 0x57, 0x35, 0xc4, 0x74, 0x39, 0xb7, 0x02, 0xe5, 0x44, 0xc7, 0x50,
                    0x78, 0xd2, 0x5f, 0x76, 0x89, 0x32, 0x89, 0xdd, 0xdb, 0xc6, 0x42, 0x9e, 0x43, 0x67, 0x30,
                    0xc3, 0x02, 0xf1, 0x84, 0xaf, 0x9e, 0x99, 0xe5, 0x97, 0x97, 0x25, 0x87, 0x7d, 0xc6, 0xe1,
                ],
                vec![
                    0x69, 0xfa, 0xd6, 0x1c, 0xcb, 0x8b, 0x68, 0x8b, 0x5e, 0xe0, 0xad, 0x10, 0xc0, 0x98, 0x02,
                    0xd5,
                ],
            ),
        ]
    }

    #[test]
    fn nist_gcm_known_answer_vectors() {
        for (nonce, key, aad, pt, ct, tag) in nist_vectors() {
            let sealed = seal(&key, &nonce, &aad, &pt).expect("valid key length");
            assert_eq!(&sealed[..pt.len()], ct.as_slice(), "ciphertext mismatch");
            assert_eq!(&sealed[pt.len()..], tag.as_slice(), "tag mismatch");

            let opened = open(&key, &nonce, &aad, &sealed).expect("valid ciphertext authenticates");
            assert_eq!(opened, pt, "round-trip plaintext mismatch");
        }
    }

    #[test]
    fn seal_rejects_invalid_key_length() {
        let nonce = [0u8; 12];
        for len in [0usize, 1, 15, 17, 24, 31, 33] {
            let key = vec![0u8; len];
            assert!(seal(&key, &nonce, b"aad", b"pt").is_err(), "key len {len}");
            assert!(open(&key, &nonce, b"aad", &[0u8; 32]).is_err(), "key len {len}");
        }
    }

    #[test]
    fn open_rejects_tampering() {
        let nonce = [7u8; 12];
        let key = [9u8; 32];
        let aad = b"header".as_slice();
        let pt = b"authenticated payload".as_slice();
        let sealed = seal(&key, &nonce, aad, pt).expect("valid key length");
        assert_eq!(sealed.len(), pt.len() + TAG_LEN);

        // Correct input still opens.
        assert_eq!(open(&key, &nonce, aad, &sealed).expect("valid tag"), pt);

        // Flip one bit in the tag / ciphertext / aad / nonce / key.
        for pos in [pt.len(), pt.len() + 7, 0, pt.len().saturating_sub(1)] {
            let mut bad = sealed.clone();
            bad[pos] ^= 0x01;
            assert!(open(&key, &nonce, aad, &bad).is_err(), "flip at {pos}");
        }
        let mut bad_aad = aad.to_vec();
        bad_aad[0] ^= 0x01;
        assert!(open(&key, &nonce, &bad_aad, &sealed).is_err());
        let mut bad_nonce = nonce;
        bad_nonce[0] ^= 0x01;
        assert!(open(&key, &bad_nonce, aad, &sealed).is_err());
        let mut bad_key = key;
        bad_key[31] ^= 0x01;
        assert!(open(&bad_key, &nonce, aad, &sealed).is_err());

        // Input shorter than the tag.
        assert!(open(&key, &nonce, aad, &[0u8; TAG_LEN - 1]).is_err());
    }

    #[test]
    fn seal_open_roundtrip_across_sizes() {
        let key = [0x2au8; 16];
        let nonce = [0x55u8; 12];
        let mut rng = Lcg::new(0xA5A5_1234);
        for len in [0usize, 1, 15, 16, 17, 31, 64, 100, 257] {
            let mut pt = vec![0u8; len];
            rng.fill(&mut pt);
            let mut aad = vec![0u8; len % 40];
            rng.fill(&mut aad);
            let sealed = seal(&key, &nonce, &aad, &pt).expect("valid key length");
            let opened = open(&key, &nonce, &aad, &sealed).expect("valid ciphertext authenticates");
            assert_eq!(opened, pt, "len {len}");
        }
    }

    /// Independent GF(2^128) reference vectors (bit-serial NIST algorithm
    /// computed externally with Python).
    #[test]
    fn gf_mul_matches_reference_vectors() {
        let vectors: [([u8; 16], [u8; 16], [u8; 16]); 4] = [
            (
                [
                    13, 48, 43, 210, 220, 231, 251, 158, 158, 242, 184, 33, 81, 218, 130, 238,
                ],
                [58, 74, 1, 67, 195, 6, 40, 188, 152, 245, 217, 86, 66, 24, 8, 175],
                [
                    95, 128, 39, 197, 192, 3, 48, 160, 249, 228, 160, 94, 113, 158, 42, 169,
                ],
            ),
            (
                [
                    201, 145, 70, 154, 8, 172, 164, 223, 8, 14, 173, 33, 78, 23, 79, 92,
                ],
                [
                    35, 181, 25, 196, 88, 2, 86, 244, 97, 199, 96, 232, 137, 42, 198, 239,
                ],
                [64, 71, 172, 2, 133, 25, 193, 46, 21, 106, 0, 86, 161, 0, 174, 145],
            ),
            (
                [
                    247, 247, 212, 155, 100, 165, 175, 227, 34, 36, 17, 49, 140, 12, 160, 227,
                ],
                [
                    210, 127, 56, 69, 14, 12, 170, 39, 230, 86, 197, 247, 24, 181, 250, 83,
                ],
                [
                    169, 53, 172, 102, 9, 202, 159, 76, 145, 179, 105, 219, 109, 160, 126, 76,
                ],
            ),
            (
                [
                    108, 211, 107, 84, 244, 67, 51, 61, 250, 49, 124, 176, 168, 182, 178, 164,
                ],
                [
                    230, 224, 246, 237, 118, 21, 198, 84, 188, 158, 235, 62, 21, 0, 103, 215,
                ],
                [
                    253, 178, 112, 127, 119, 168, 112, 205, 164, 21, 57, 26, 185, 231, 28, 227,
                ],
            ),
        ];
        for (x, y, want) in vectors {
            let mut got = x;
            gf_mul(&mut got, &y);
            assert_eq!(got, want, "x={x:02x?} y={y:02x?}");
        }

        // Structural properties: multiply by the field one (0x80 in byte 0 is
        // the coefficient of x^0 in GCM bit order) and by zero.
        let one = [0x80u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let zero = [0u8; 16];
        for (x, _, _) in vectors {
            let mut got = x;
            gf_mul(&mut got, &one);
            assert_eq!(got, x, "x * 1 must equal x");
            let mut got = x;
            gf_mul(&mut got, &zero);
            assert_eq!(got, zero, "x * 0 must equal 0");
        }
    }

    /// FIPS-197 appendix C known-answer test for the pure software AES path
    /// (the NIST GCM vectors above exercise the AES-NI path on this host).
    #[test]
    fn soft_aes_encrypt_matches_fips197() {
        let plaintext = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];

        // C.1: AES-128, key 000102...0f.
        let key128: Vec<u8> = (0u8..16).collect();
        let aes128 = AesKey {
            round_keys: expand_key(&key128, 10),
            nr: 10,
        };
        let mut out = [0u8; 16];
        aes_encrypt_block_soft(&aes128, &plaintext, &mut out);
        assert_eq!(
            out,
            [
                0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
                0x5a,
            ]
        );

        // C.2: AES-256, key 000102...1f.
        let key256: Vec<u8> = (0u8..32).collect();
        let aes256 = AesKey {
            round_keys: expand_key(&key256, 14),
            nr: 14,
        };
        let mut out = [0u8; 16];
        aes_encrypt_block_soft(&aes256, &plaintext, &mut out);
        assert_eq!(
            out,
            [
                0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60,
                0x89,
            ]
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn aes_ni_key_expansion_matches_software() {
        if !std::is_x86_feature_detected!("aes") {
            return;
        }
        let mut rng = Lcg::new(0xDEAD_BEEF);
        for len in [16usize, 32] {
            for _ in 0..50 {
                let mut key = vec![0u8; len];
                rng.fill(&mut key);
                let nr = if len == 16 { 10 } else { 14 };
                let sw = expand_key(&key, nr);
                // SAFETY: AES-NI was detected above and the key length is valid.
                let ni = unsafe { expand_key_ni(&key) };
                assert_eq!(ni, sw, "len {len} key {key:02x?}");
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn aes_ni_block_encrypt_matches_software() {
        if !std::is_x86_feature_detected!("aes") {
            return;
        }
        let mut rng = Lcg::new(0x0BAD_F00D);
        for len in [16usize, 32] {
            let mut key = vec![0u8; len];
            rng.fill(&mut key);
            let nr = if len == 16 { 10 } else { 14 };
            let aes = AesKey {
                round_keys: expand_key(&key, nr),
                nr,
            };
            for _ in 0..32 {
                let mut input = [0u8; 16];
                rng.fill(&mut input);
                let mut soft = [0u8; 16];
                aes_encrypt_block_soft(&aes, &input, &mut soft);
                let mut ni = [0u8; 16];
                // SAFETY: AES-NI was detected above; key/input buffers are 16 bytes.
                unsafe { aes_encrypt_block_ni(&aes, &input, &mut ni) };
                assert_eq!(ni, soft, "input {input:02x?}");
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn clmul_ghash_matches_software_ghash() {
        if !std::is_x86_feature_detected!("pclmulqdq") {
            return;
        }
        let mut rng = Lcg::new(0xC1C1_C1C1);
        for len in [0usize, 1, 15, 16, 17, 45, 64, 100, 257] {
            for _ in 0..8 {
                let mut h = [0u8; 16];
                rng.fill(&mut h);
                let mut aad = vec![0u8; len % 50];
                rng.fill(&mut aad);
                let mut ct = vec![0u8; len];
                rng.fill(&mut ct);
                // SAFETY: PCLMULQDQ was detected above.
                let hw = unsafe { ghash_clmul(&h, &aad, &ct) };
                let sw = ghash_soft(&h, &aad, &ct);
                assert_eq!(hw, sw, "len {len} h={h:02x?}");
            }
        }
    }
}
