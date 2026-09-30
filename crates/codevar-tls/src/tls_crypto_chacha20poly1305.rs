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

use zeroize::Zeroize;

const KEY_LEN: usize = 32;
const TAG_LEN: usize = 16;

/// Seals `plaintext` under ChaCha20-Poly1305.
///
/// Output layout: ciphertext or 16-byte tag.
#[allow(clippy::result_unit_err)]
pub fn seal(key: &[u8], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, ()> {
    if key.len() != KEY_LEN {
        return Err(());
    }
    let mut out = vec![0u8; plaintext.len() + TAG_LEN];
    let (ct, tag_out) = out.split_at_mut(plaintext.len());
    chacha20_xor(key, nonce, 1, plaintext, ct);
    let tag = poly1305_aead_tag(key, nonce, aad, ct);
    tag_out.copy_from_slice(&tag);
    Ok(out)
}

/// Opens ciphertext or tag produced by [`seal`].
#[allow(clippy::result_unit_err)]
pub fn open(key: &[u8], nonce: &[u8; 12], aad: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, ()> {
    if key.len() != KEY_LEN || ciphertext.len() < TAG_LEN {
        return Err(());
    }
    let (ct, tag) = ciphertext.split_at(ciphertext.len() - TAG_LEN);
    let expected = poly1305_aead_tag(key, nonce, aad, ct);
    if !crate::tls_crypto_ct::ct_eq(&expected, tag) {
        return Err(());
    }
    let mut plain = vec![0u8; ct.len()];
    chacha20_xor(key, nonce, 1, ct, &mut plain);
    Ok(plain)
}

#[inline(always)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

fn chacha20_block(key: &[u8], nonce: &[u8; 12], counter: u32, out: &mut [u8; 64]) {
    // SAFETY: callers pass a 32-byte key; fixed offsets are in-bounds.
    let kw = unsafe { read_key_words(key) };
    let mut state = [
        0x6170_7865,
        0x3320_646e,
        0x7962_2d32,
        0x6b20_6574,
        kw[0],
        kw[1],
        kw[2],
        kw[3],
        kw[4],
        kw[5],
        kw[6],
        kw[7],
        counter,
        u32::from_le_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]),
        u32::from_le_bytes([nonce[4], nonce[5], nonce[6], nonce[7]]),
        u32::from_le_bytes([nonce[8], nonce[9], nonce[10], nonce[11]]),
    ];
    let mut w = state;
    for _ in 0..10 {
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }
    for i in 0..16 {
        state[i] = state[i].wrapping_add(w[i]);
        out[i * 4..i * 4 + 4].copy_from_slice(&state[i].to_le_bytes());
    }
    w.zeroize();
    state.zeroize();
}

/// # Safety
/// `key.len()` must be >= 32.
#[inline(always)]
unsafe fn read_key_words(key: &[u8]) -> [u32; 8] {
    // SAFETY: caller guarantees key is at least 32 bytes.
    unsafe {
        let p = key.as_ptr();
        [
            u32::from_le(core::ptr::read_unaligned(p.cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(4).cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(8).cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(12).cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(16).cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(20).cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(24).cast())),
            u32::from_le(core::ptr::read_unaligned(p.add(28).cast())),
        ]
    }
}

fn chacha20_xor(key: &[u8], nonce: &[u8; 12], mut counter: u32, input: &[u8], output: &mut [u8]) {
    let mut offset = 0;
    while offset < input.len() {
        let mut block = [0u8; 64];
        chacha20_block(key, nonce, counter, &mut block);
        let n = (input.len() - offset).min(64);
        for i in 0..n {
            output[offset + i] = input[offset + i] ^ block[i];
        }
        block.zeroize();
        counter = counter.wrapping_add(1);
        offset += n;
    }
}

fn poly1305_aead_tag(key: &[u8], nonce: &[u8; 12], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut block = [0u8; 64];
    chacha20_block(key, nonce, 0, &mut block);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&block[..32]);
    block.zeroize();

    let mut mac_data = Vec::with_capacity(aad.len() + ciphertext.len() + 48);
    mac_data.extend_from_slice(aad);
    pad16(&mut mac_data);
    mac_data.extend_from_slice(ciphertext);
    pad16(&mut mac_data);
    mac_data.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_data.extend_from_slice(&(ciphertext.len() as u64).to_le_bytes());

    let tag = poly1305(&otk, &mac_data);
    otk.zeroize();
    tag
}

fn pad16(buf: &mut Vec<u8>) {
    let rem = buf.len() % 16;
    if rem != 0 {
        buf.extend(core::iter::repeat_n(0u8, 16 - rem));
    }
}

/// Poly1305 MAC (RFC 8439 §2.5) over an already-formatted message.
fn poly1305(key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    // Clamp r.
    let mut r_bytes = [0u8; 16];
    r_bytes.copy_from_slice(&key[..16]);
    r_bytes[3] &= 15;
    r_bytes[7] &= 15;
    r_bytes[11] &= 15;
    r_bytes[15] &= 15;
    r_bytes[4] &= 252;
    r_bytes[8] &= 252;
    r_bytes[12] &= 252;

    // 26-bit limbs for r.
    let t0 = u32::from_le_bytes([r_bytes[0], r_bytes[1], r_bytes[2], r_bytes[3]]);
    let t1 = u32::from_le_bytes([r_bytes[4], r_bytes[5], r_bytes[6], r_bytes[7]]);
    let t2 = u32::from_le_bytes([r_bytes[8], r_bytes[9], r_bytes[10], r_bytes[11]]);
    let t3 = u32::from_le_bytes([r_bytes[12], r_bytes[13], r_bytes[14], r_bytes[15]]);
    let r0 = t0 & 0x3ff_ffff;
    let r1 = ((t0 >> 26) | (t1 << 6)) & 0x3ff_ffff;
    let r2 = ((t1 >> 20) | (t2 << 12)) & 0x3ff_ffff;
    let r3 = ((t2 >> 14) | (t3 << 18)) & 0x3ff_ffff;
    let r4 = (t3 >> 8) & 0x00ff_ffff;

    let s1 = r1.wrapping_mul(5);
    let s2 = r2.wrapping_mul(5);
    let s3 = r3.wrapping_mul(5);
    let s4 = r4.wrapping_mul(5);

    let mut h = [0u32; 5];
    let mut offset = 0;
    while offset < msg.len() {
        let take = (msg.len() - offset).min(16);
        let mut blk = [0u8; 17];
        blk[..take].copy_from_slice(&msg[offset..offset + take]);
        blk[take] = 1;

        let t0 = u32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        let t1 = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let t2 = u32::from_le_bytes([blk[8], blk[9], blk[10], blk[11]]);
        let t3 = u32::from_le_bytes([blk[12], blk[13], blk[14], blk[15]]);
        let t4 = u32::from(blk[16]);

        h[0] = h[0].wrapping_add(t0 & 0x3ff_ffff);
        h[1] = h[1].wrapping_add(((t0 >> 26) | (t1 << 6)) & 0x3ff_ffff);
        h[2] = h[2].wrapping_add(((t1 >> 20) | (t2 << 12)) & 0x3ff_ffff);
        h[3] = h[3].wrapping_add(((t2 >> 14) | (t3 << 18)) & 0x3ff_ffff);
        h[4] = h[4].wrapping_add(((t3 >> 8) | (t4 << 24)) & 0x3ff_ffff);

        let m0 = u64::from(h[0]) * u64::from(r0)
            + u64::from(h[1]) * u64::from(s4)
            + u64::from(h[2]) * u64::from(s3)
            + u64::from(h[3]) * u64::from(s2)
            + u64::from(h[4]) * u64::from(s1);
        let mut m1 = u64::from(h[0]) * u64::from(r1)
            + u64::from(h[1]) * u64::from(r0)
            + u64::from(h[2]) * u64::from(s4)
            + u64::from(h[3]) * u64::from(s3)
            + u64::from(h[4]) * u64::from(s2);
        let mut m2 = u64::from(h[0]) * u64::from(r2)
            + u64::from(h[1]) * u64::from(r1)
            + u64::from(h[2]) * u64::from(r0)
            + u64::from(h[3]) * u64::from(s4)
            + u64::from(h[4]) * u64::from(s3);
        let mut m3 = u64::from(h[0]) * u64::from(r3)
            + u64::from(h[1]) * u64::from(r2)
            + u64::from(h[2]) * u64::from(r1)
            + u64::from(h[3]) * u64::from(r0)
            + u64::from(h[4]) * u64::from(s4);
        let mut m4 = u64::from(h[0]) * u64::from(r4)
            + u64::from(h[1]) * u64::from(r3)
            + u64::from(h[2]) * u64::from(r2)
            + u64::from(h[3]) * u64::from(r1)
            + u64::from(h[4]) * u64::from(r0);
        let mut c = m0 >> 26;
        h[0] = m0 as u32 & 0x3ff_ffff;
        m1 += c;
        c = m1 >> 26;
        h[1] = m1 as u32 & 0x3ff_ffff;
        m2 += c;
        c = m2 >> 26;
        h[2] = m2 as u32 & 0x3ff_ffff;
        m3 += c;
        c = m3 >> 26;
        h[3] = m3 as u32 & 0x3ff_ffff;
        m4 += c;
        c = m4 >> 26;
        h[4] = m4 as u32 & 0x3ff_ffff;
        h[0] = h[0].wrapping_add((c as u32).wrapping_mul(5));
        c = u64::from(h[0] >> 26);
        h[0] &= 0x3ff_ffff;
        h[1] = h[1].wrapping_add(c as u32);

        offset += take;
    }
    let mut c = h[1] >> 26;
    h[1] &= 0x3ff_ffff;
    h[2] = h[2].wrapping_add(c);
    c = h[2] >> 26;
    h[2] &= 0x3ff_ffff;
    h[3] = h[3].wrapping_add(c);
    c = h[3] >> 26;
    h[3] &= 0x3ff_ffff;
    h[4] = h[4].wrapping_add(c);
    c = h[4] >> 26;
    h[4] &= 0x3ff_ffff;
    h[0] = h[0].wrapping_add(c.wrapping_mul(5));
    c = h[0] >> 26;
    h[0] &= 0x3ff_ffff;
    h[1] = h[1].wrapping_add(c);

    // compute h + -p
    let mut g = [0u32; 5];
    let mut c = 5u32;
    for i in 0..4 {
        g[i] = h[i].wrapping_add(c);
        c = g[i] >> 26;
        g[i] &= 0x3ff_ffff;
    }
    g[4] = h[4].wrapping_add(c).wrapping_sub(1 << 26);

    let mask = (g[4] >> 31).wrapping_sub(1);
    let not_mask = !mask;
    for i in 0..5 {
        h[i] = (h[i] & not_mask) | (g[i] & mask);
    }

    // h = h % 2^128  (pack limbs) then h += s
    let mut f0 = (u64::from(h[0]) | (u64::from(h[1]) << 26)) & 0xffffffff;
    let mut f1 = ((u64::from(h[1]) >> 6) | (u64::from(h[2]) << 20)) & 0xffffffff;
    let mut f2 = ((u64::from(h[2]) >> 12) | (u64::from(h[3]) << 14)) & 0xffffffff;
    let mut f3 = ((u64::from(h[3]) >> 18) | (u64::from(h[4]) << 8)) & 0xffffffff;

    let s0 = u32::from_le_bytes([key[16], key[17], key[18], key[19]]);
    let s1 = u32::from_le_bytes([key[20], key[21], key[22], key[23]]);
    let s2 = u32::from_le_bytes([key[24], key[25], key[26], key[27]]);
    let s3 = u32::from_le_bytes([key[28], key[29], key[30], key[31]]);

    f0 = f0.wrapping_add(u64::from(s0));
    f1 = f1
        .wrapping_add(u64::from(s1))
        .wrapping_add(f0 >> 32);
    f2 = f2
        .wrapping_add(u64::from(s2))
        .wrapping_add(f1 >> 32);
    f3 = f3
        .wrapping_add(u64::from(s3))
        .wrapping_add(f2 >> 32);

    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&(f0 as u32).to_le_bytes());
    tag[4..8].copy_from_slice(&(f1 as u32).to_le_bytes());
    tag[8..12].copy_from_slice(&(f2 as u32).to_le_bytes());
    tag[12..16].copy_from_slice(&(f3 as u32).to_le_bytes());
    tag
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 8439 §2.8.2 AEAD test vector key.
    const RFC_KEY: [u8; 32] = [
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90,
        0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    ];
    /// RFC 8439 §2.8.2 nonce.
    const RFC_NONCE: [u8; 12] = [
        0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    ];
    /// RFC 8439 §2.8.2 additional data.
    const RFC_AAD: [u8; 12] = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];
    const RFC_PLAINTEXT: &[u8] = b"Ladies and Gentlemen of the class of '99: If I could offer you \
only one tip for the future, sunscreen would be it.";
    /// RFC 8439 §2.8.2 expected ciphertext || tag.
    const RFC_CT_TAG: &str = "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d\
63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36\
92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3f\
f4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd0600691";

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn rfc8439_aead_vector_seals_and_opens() {
        let expected = unhex(RFC_CT_TAG);
        let out = seal(&RFC_KEY, &RFC_NONCE, &RFC_AAD, RFC_PLAINTEXT).unwrap();
        assert_eq!(out, expected);
        assert_eq!(out.len(), RFC_PLAINTEXT.len() + TAG_LEN);
        // Tag is the final 16 bytes of the published vector.
        assert_eq!(
            &out[out.len() - TAG_LEN..],
            &unhex("1ae10b594f09e26a7e902ecbd0600691")[..]
        );
        let back = open(&RFC_KEY, &RFC_NONCE, &RFC_AAD, &out).unwrap();
        assert_eq!(back, RFC_PLAINTEXT);
    }

    #[test]
    fn rfc8439_empty_plaintext_produces_tag_only() {
        // Independently computed: ChaCha20-Poly1305 over empty PT/AAD.
        let expected = unhex("a0784d7a4716f3feb4f64e7f4b39bf04");
        let out = seal(&RFC_KEY, &RFC_NONCE, &[], &[]).unwrap();
        assert_eq!(out, expected);
        assert_eq!(out.len(), TAG_LEN);
        let back = open(&RFC_KEY, &RFC_NONCE, &[], &out).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn round_trip_across_block_boundaries() {
        // 64-byte ChaCha20 block edges plus multi-block payloads.
        for len in [0usize, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000] {
            let plain: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            for aad in [b"".as_slice(), b"hdr", &[0u8; 16][..], &[0u8; 31][..]] {
                let sealed = seal(&RFC_KEY, &RFC_NONCE, aad, &plain).unwrap();
                assert_eq!(sealed.len(), len + TAG_LEN);
                let opened = open(&RFC_KEY, &RFC_NONCE, aad, &sealed).unwrap();
                assert_eq!(opened, plain, "round trip failed for len={len}");
            }
        }
    }

    #[test]
    fn tampering_ciphertext_tag_or_aad_is_rejected() {
        let sealed = seal(&RFC_KEY, &RFC_NONCE, &RFC_AAD, RFC_PLAINTEXT).unwrap();

        // Flip a ciphertext bit.
        let mut bad_ct = sealed.clone();
        bad_ct[0] ^= 0x01;
        assert!(open(&RFC_KEY, &RFC_NONCE, &RFC_AAD, &bad_ct).is_err());

        // Flip a tag bit.
        let mut bad_tag = sealed.clone();
        let last = bad_tag.len() - 1;
        bad_tag[last] ^= 0x80;
        assert!(open(&RFC_KEY, &RFC_NONCE, &RFC_AAD, &bad_tag).is_err());

        // Different AAD.
        let mut wrong_aad = RFC_AAD;
        wrong_aad[0] ^= 0x01;
        assert!(open(&RFC_KEY, &RFC_NONCE, &wrong_aad, &sealed).is_err());

        // Different nonce.
        let mut wrong_nonce = RFC_NONCE;
        wrong_nonce[11] ^= 0x01;
        assert!(open(&RFC_KEY, &wrong_nonce, &RFC_AAD, &sealed).is_err());

        // Different key.
        let mut wrong_key = RFC_KEY;
        wrong_key[0] ^= 0x01;
        assert!(open(&wrong_key, &RFC_NONCE, &RFC_AAD, &sealed).is_err());

        // Truncated input below the tag length.
        assert!(open(&RFC_KEY, &RFC_NONCE, &RFC_AAD, &sealed[..TAG_LEN - 1]).is_err());
    }

    #[test]
    fn invalid_key_lengths_are_rejected() {
        assert!(seal(&[], &RFC_NONCE, &[], b"x").is_err());
        assert!(seal(&[0u8; 16], &RFC_NONCE, &[], b"x").is_err());
        assert!(seal(&[0u8; 31], &RFC_NONCE, &[], b"x").is_err());
        assert!(seal(&[0u8; 33], &RFC_NONCE, &[], b"x").is_err());
        assert!(open(&[0u8; 16], &RFC_NONCE, &[], &[0u8; 16]).is_err());
    }

    #[test]
    fn aad_does_not_change_ciphertext_but_changes_tag() {
        let a = seal(&RFC_KEY, &RFC_NONCE, b"aad-one", RFC_PLAINTEXT).unwrap();
        let b = seal(&RFC_KEY, &RFC_NONCE, b"aad-two", RFC_PLAINTEXT).unwrap();
        assert_eq!(a[..RFC_PLAINTEXT.len()], b[..RFC_PLAINTEXT.len()]);
        assert_ne!(a[RFC_PLAINTEXT.len()..], b[RFC_PLAINTEXT.len()..]);
    }

    #[test]
    fn distinct_nonces_produce_distinct_keystreams() {
        let mut nonce2 = RFC_NONCE;
        nonce2[11] = 1;
        let a = seal(&RFC_KEY, &RFC_NONCE, &RFC_AAD, RFC_PLAINTEXT).unwrap();
        let b = seal(&RFC_KEY, &nonce2, &RFC_AAD, RFC_PLAINTEXT).unwrap();
        assert_ne!(a, b);
    }
}
