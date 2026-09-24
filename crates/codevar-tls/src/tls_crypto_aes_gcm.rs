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

//! AES-128/256-GCM (NIST SP 800-38D) with optional AES-NI acceleration.

use subtle::ConstantTimeEq;
use zeroize::{Zeroize, ZeroizeOnDrop};

const BLOCK: usize = 16;
const TAG_LEN: usize = 16;

/// Seals with AES-GCM. `key` must be 16 (AES-128) or 32 (AES-256) bytes.
#[allow(clippy::result_unit_err)]
pub fn seal(
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, ()> {
    let aes = AesKey::new(key)?;
    let mut out = vec![0u8; plaintext.len() + TAG_LEN];
    let (ct, tag_slot) = out.split_at_mut(plaintext.len());
    let tag = gcm_seal(&aes, nonce, aad, plaintext, ct)?;
    tag_slot.copy_from_slice(&tag);
    Ok(out)
}

/// Opens ciphertext||tag produced by [`seal`].
#[allow(clippy::result_unit_err)]
pub fn open(
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
) -> Result<Vec<u8>, ()> {
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
        match key.len() {
            16 => Ok(Self {
                round_keys: expand_key(key, 10),
                nr: 10,
            }),
            32 => Ok(Self {
                round_keys: expand_key(key, 14),
                nr: 14,
            }),
            _ => Err(()),
        }
    }

    fn encrypt_block(&self, input: &[u8; 16], output: &mut [u8; 16]) {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if crate::tls_crypto_device::capabilities().aes_hw {
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
    if !bool::from(expected.ct_eq(tag)) {
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

fn ctr32_inc(block: &mut [u8; 16]) {
    for i in (12..16).rev() {
        block[i] = block[i].wrapping_add(1);
        if block[i] != 0 {
            break;
        }
    }
}

fn ghash(h: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
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
fn gf_mul(x: &mut [u8; 16], y: &[u8; 16]) {
    let mut z = [0u8; 16];
    let mut v = *y;
    for i in 0..128 {
        if (x[i / 8] >> (7 - (i % 8))) & 1 == 1 {
            xor_block(&mut z, &v);
        }
        let lsb = v[15] & 1;
        // shift v right by 1 bit
        for j in (1..16).rev() {
            v[j] = (v[j] >> 1) | (v[j - 1] << 7);
        }
        v[0] >>= 1;
        if lsb != 0 {
            v[0] ^= 0xe1;
        }
    }
    *x = z;
}

static SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7,
    0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf,
    0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5,
    0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e,
    0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
    0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef,
    0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff,
    0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d,
    0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee,
    0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5,
    0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25, 0x2e,
    0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
    0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55,
    0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16,
];

static RCON: [u8; 11] = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
];

fn expand_key(key: &[u8], nr: usize) -> Vec<[u8; 16]> {
    let nk = key.len() / 4;
    let total_words = 4 * (nr + 1);
    let mut w = vec![0u32; total_words];
    for i in 0..nk {
        w[i] = u32::from_be_bytes([
            key[4 * i],
            key[4 * i + 1],
            key[4 * i + 2],
            key[4 * i + 3],
        ]);
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

fn rot_word(x: u32) -> u32 {
    x.rotate_left(8)
}

fn sub_word(x: u32) -> u32 {
    let b = x.to_be_bytes();
    u32::from_be_bytes([
        SBOX[b[0] as usize],
        SBOX[b[1] as usize],
        SBOX[b[2] as usize],
        SBOX[b[3] as usize],
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

fn add_round_key(state: &mut [u8; 16], rk: &[u8; 16]) {
    for i in 0..16 {
        state[i] ^= rk[i];
    }
}

fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

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

fn xtime(a: u8) -> u8 {
    (a << 1) ^ (if a & 0x80 != 0 { 0x1b } else { 0 })
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "aes")]
unsafe fn aes_encrypt_block_ni(key: &AesKey, input: &[u8; 16], output: &mut [u8; 16]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_loadu_si128,
        _mm_storeu_si128, _mm_xor_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_loadu_si128,
        _mm_storeu_si128, _mm_xor_si128,
    };

    // SAFETY: callers ensure AES-NI is present; unaligned load/store are valid for any
    // 16-byte buffer; `round_keys` has length `nr + 1`.
    unsafe {
        let mut state = _mm_loadu_si128(input.as_ptr().cast::<__m128i>());
        state = _mm_xor_si128(state, _mm_loadu_si128(key.round_keys[0].as_ptr().cast()));
        for round in 1..key.nr {
            state = _mm_aesenc_si128(
                state,
                _mm_loadu_si128(key.round_keys[round].as_ptr().cast()),
            );
        }
        state = _mm_aesenclast_si128(
            state,
            _mm_loadu_si128(key.round_keys[key.nr].as_ptr().cast()),
        );
        _mm_storeu_si128(output.as_mut_ptr().cast::<__m128i>(), state);
    }
}
