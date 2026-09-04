//! Copyright 2026 Codevar
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

//! TLS cryptographic primitive unit tests (hash, PRF, HKDF, AEAD, ct_eq).

use codevar_colab::network::tls_aead::{AeadKey, TlsAead, nonce_xor_seq};
use codevar_colab::network::tls_crypto_aes_gcm;
use codevar_colab::network::tls_crypto_chacha20poly1305;
use codevar_colab::network::tls_crypto_hash::{hash_empty, hash_message};
use codevar_colab::network::tls_hkdf::{derive_secret, hkdf_expand_label, hkdf_extract};
use codevar_colab::network::tls_ids::{AeadAlgorithm, HashAlgorithm};
use codevar_colab::network::tls_prf::{ct_eq, hmac_hash, tls12_prf};

#[test]
fn ct_eq_constant_time_compare() {
    assert!(ct_eq(b"abc", b"abc"));
    assert!(!ct_eq(b"abc", b"abd"));
    assert!(!ct_eq(b"abc", b"ab"));
    assert!(ct_eq(b"", b""));
}

#[test]
fn hash_empty_and_message() {
    let e256 = hash_empty(HashAlgorithm::Sha256);
    assert_eq!(e256.len(), 32);
    // SHA-256 of empty string
    assert_eq!(
        e256,
        hex_to_bytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    );

    let msg = hash_message(HashAlgorithm::Sha256, b"abc");
    assert_eq!(
        msg,
        hex_to_bytes("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    );

    let e384 = hash_empty(HashAlgorithm::Sha384);
    assert_eq!(e384.len(), 48);
}

#[test]
fn tls12_prf_deterministic_and_length() {
    let out = tls12_prf(HashAlgorithm::Sha256, b"secret", b"label", b"seed", 48).unwrap();
    assert_eq!(out.len(), 48);
    let out2 =
        tls12_prf(HashAlgorithm::Sha256, b"secret", b"label", b"seed", 48).unwrap();
    assert_eq!(out, out2);

    let out384 = tls12_prf(HashAlgorithm::Sha384, b"s", b"l", b"d", 16).unwrap();
    assert_eq!(out384.len(), 16);
}

#[test]
fn hmac_hash_lengths() {
    let mac = hmac_hash(HashAlgorithm::Sha256, b"key", b"data").unwrap();
    assert_eq!(mac.len(), 32);
    let mac384 = hmac_hash(HashAlgorithm::Sha384, b"key", b"data").unwrap();
    assert_eq!(mac384.len(), 48);
}

#[test]
fn hkdf_extract_expand_label_and_derive_secret() {
    let prk = hkdf_extract(HashAlgorithm::Sha256, b"salt", b"ikm");
    assert_eq!(prk.len(), 32);

    let okm =
        hkdf_expand_label(HashAlgorithm::Sha256, &prk, b"key", b"context", 16).unwrap();
    assert_eq!(okm.len(), 16);

    let secret = derive_secret(
        HashAlgorithm::Sha256,
        &prk,
        b"c hs traffic",
        Some(&hash_empty(HashAlgorithm::Sha256)),
    )
    .unwrap();
    assert_eq!(secret.len(), 32);

    let secret2 = derive_secret(HashAlgorithm::Sha256, &prk, b"derived", None).unwrap();
    assert_eq!(secret2.len(), 32);
}

#[test]
fn nonce_xor_seq_layout() {
    let iv = [0u8; 12];
    let n0 = nonce_xor_seq(&iv, 0);
    assert_eq!(n0, [0u8; 12]);
    let n1 = nonce_xor_seq(&iv, 1);
    assert_eq!(n1[11], 1);
    assert_eq!(&n1[..11], &[0u8; 11]);
}

#[test]
fn aead_tls13_aes128_roundtrip() {
    let key =
        AeadKey::new(AeadAlgorithm::Aes128Gcm, vec![0x42; 16], vec![0x11; 12]).unwrap();
    let aad = b"aad-header";
    let pt = b"application data";
    let ct = TlsAead::encrypt_tls13(&key, 7, aad, pt).unwrap();
    assert!(ct.len() > pt.len()); // tag overhead
    let out = TlsAead::decrypt_tls13(&key, 7, aad, &ct).unwrap();
    assert_eq!(out, pt);

    // Wrong sequence fails
    assert!(TlsAead::decrypt_tls13(&key, 8, aad, &ct).is_err());
    // Wrong AAD fails
    assert!(TlsAead::decrypt_tls13(&key, 7, b"bad", &ct).is_err());
}

#[test]
fn aead_tls13_chacha_roundtrip() {
    let key = AeadKey::new(
        AeadAlgorithm::ChaCha20Poly1305,
        vec![0x7a; 32],
        vec![0x22; 12],
    )
    .unwrap();
    let pt = b"chacha payload";
    let ct = TlsAead::encrypt_tls13(&key, 1, b"", pt).unwrap();
    let out = TlsAead::decrypt_tls13(&key, 1, b"", &ct).unwrap();
    assert_eq!(out, pt);
}

#[test]
fn aead_key_rejects_wrong_lengths() {
    assert!(AeadKey::new(AeadAlgorithm::Aes128Gcm, vec![0; 15], vec![0; 12]).is_err());
    assert!(AeadKey::new(AeadAlgorithm::Aes256Gcm, vec![0; 32], vec![0; 8]).is_err());
}

#[test]
fn raw_aes_gcm_and_chacha_seal_open() {
    let nonce = [9u8; 12];
    let aad = b"aad";
    let pt = b"plaintext-bytes";

    let aes_key = [3u8; 16];
    let ct = tls_crypto_aes_gcm::seal(&aes_key, &nonce, aad, pt).unwrap();
    let out = tls_crypto_aes_gcm::open(&aes_key, &nonce, aad, &ct).unwrap();
    assert_eq!(out, pt);
    assert!(tls_crypto_aes_gcm::open(&aes_key, &nonce, b"x", &ct).is_err());

    let cha_key = [5u8; 32];
    let ct = tls_crypto_chacha20poly1305::seal(&cha_key, &nonce, aad, pt).unwrap();
    let out = tls_crypto_chacha20poly1305::open(&cha_key, &nonce, aad, &ct).unwrap();
    assert_eq!(out, pt);
    assert!(
        tls_crypto_chacha20poly1305::open(&cha_key, &nonce, aad, &ct[..ct.len() - 1])
            .is_err()
    );
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}
