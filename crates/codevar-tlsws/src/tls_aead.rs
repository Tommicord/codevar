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

//! AEAD record protection (RFC 8446 §5.2, RFC 5288, RFC 7905).

use crate::tls_alert::AlertDescription;
use crate::tls_crypto_aes_gcm;
use crate::tls_crypto_chacha20poly1305;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::AeadAlgorithm;
use zeroize::Zeroize;

/// Directional AEAD key + IV.
#[derive(Clone)]
pub struct AeadKey {
    alg: AeadAlgorithm,
    key: Vec<u8>,
    iv: Vec<u8>,
}

impl Drop for AeadKey {
    fn drop(&mut self) {
        self.key.zeroize();
        self.iv.zeroize();
    }
}

impl AeadKey {
    /// Creates a key from raw bytes.
    pub fn new(alg: AeadAlgorithm, key: Vec<u8>, iv: Vec<u8>) -> TlsResult<Self> {
        if key.len() != alg.key_len() {
            return Err(TlsError::Internal(format!(
                "AEAD key length {} != {}",
                key.len(),
                alg.key_len()
            )));
        }
        if iv.len() != alg.iv_len() && iv.len() != alg.tls12_fixed_iv_len() {
            return Err(TlsError::Internal(format!(
                "AEAD IV length {} unexpected",
                iv.len()
            )));
        }
        Ok(Self { alg, key, iv })
    }

    /// Algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> AeadAlgorithm {
        self.alg
    }

    /// TLS 1.2 GCM salt (first 4 bytes of the implicit IV).
    #[must_use]
    pub fn tls12_salt(&self) -> &[u8] {
        &self.iv
    }
}

/// Constructs the TLS 1.3 / RFC 7905 nonce: `IV XOR padded sequence number`.
#[must_use]
pub fn nonce_xor_seq(iv: &[u8], seq: u64) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    let copy_len = iv.len().min(12);
    nonce[12 - copy_len..].copy_from_slice(&iv[iv.len() - copy_len..]);
    let seq_bytes = seq.to_be_bytes();
    for i in 0..8 {
        nonce[4 + i] ^= seq_bytes[i];
    }
    nonce
}

/// Encrypts / decrypts TLS records.
pub struct TlsAead;

impl TlsAead {
    /// TLS 1.3 record encryption (RFC 8446 §5.2).
    pub fn encrypt_tls13(
        keys: &AeadKey,
        seq: u64,
        aad: &[u8],
        plaintext: &[u8],
    ) -> TlsResult<Vec<u8>> {
        let nonce = nonce_xor_seq(&keys.iv, seq);
        seal(keys.alg, &keys.key, &nonce, aad, plaintext)
    }

    /// TLS 1.3 record decryption.
    pub fn decrypt_tls13(
        keys: &AeadKey,
        seq: u64,
        aad: &[u8],
        ciphertext: &[u8],
    ) -> TlsResult<Vec<u8>> {
        let nonce = nonce_xor_seq(&keys.iv, seq);
        open(keys.alg, &keys.key, &nonce, aad, ciphertext)
    }

    /// TLS 1.2 AES-GCM (RFC 5288): nonce = 4-byte salt || 8-byte explicit nonce.
    pub fn encrypt_tls12_gcm(
        keys: &AeadKey,
        explicit_nonce: [u8; 8],
        aad: &[u8],
        plaintext: &[u8],
    ) -> TlsResult<Vec<u8>> {
        let mut nonce = [0u8; 12];
        if keys.iv.len() < 4 {
            return Err(TlsError::Internal("GCM salt too short".into()));
        }
        nonce[..4].copy_from_slice(&keys.iv[..4]);
        nonce[4..].copy_from_slice(&explicit_nonce);
        let mut out = Vec::with_capacity(8 + plaintext.len() + 16);
        out.extend_from_slice(&explicit_nonce);
        out.extend(seal(keys.alg, &keys.key, &nonce, aad, plaintext)?);
        Ok(out)
    }

    /// TLS 1.2 AES-GCM decryption.
    pub fn decrypt_tls12_gcm(
        keys: &AeadKey,
        ciphertext: &[u8],
        aad: &[u8],
    ) -> TlsResult<Vec<u8>> {
        if ciphertext.len() < 8 + 16 {
            return Err(TlsError::Alert(AlertDescription::BadRecordMac));
        }
        let explicit = &ciphertext[..8];
        let mut nonce = [0u8; 12];
        nonce[..4].copy_from_slice(&keys.iv[..4]);
        nonce[4..].copy_from_slice(explicit);
        open(keys.alg, &keys.key, &nonce, aad, &ciphertext[8..])
    }

    /// TLS 1.2 ChaCha20-Poly1305 (RFC 7905): nonce construction matches TLS 1.3.
    pub fn encrypt_tls12_chacha(
        keys: &AeadKey,
        seq: u64,
        aad: &[u8],
        plaintext: &[u8],
    ) -> TlsResult<Vec<u8>> {
        let nonce = nonce_xor_seq(&keys.iv, seq);
        seal(keys.alg, &keys.key, &nonce, aad, plaintext)
    }

    /// TLS 1.2 ChaCha20-Poly1305 decryption.
    pub fn decrypt_tls12_chacha(
        keys: &AeadKey,
        seq: u64,
        aad: &[u8],
        ciphertext: &[u8],
    ) -> TlsResult<Vec<u8>> {
        let nonce = nonce_xor_seq(&keys.iv, seq);
        open(keys.alg, &keys.key, &nonce, aad, ciphertext)
    }
}

fn seal(
    alg: AeadAlgorithm,
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
) -> TlsResult<Vec<u8>> {
    match alg {
        AeadAlgorithm::Aes128Gcm | AeadAlgorithm::Aes256Gcm => {
            tls_crypto_aes_gcm::seal(key, nonce, aad, plaintext)
                .map_err(|()| TlsError::crypto("AES-GCM encrypt"))
        }
        AeadAlgorithm::ChaCha20Poly1305 => {
            tls_crypto_chacha20poly1305::seal(key, nonce, aad, plaintext)
                .map_err(|()| TlsError::crypto("ChaCha20-Poly1305 encrypt"))
        }
    }
}

fn open(
    alg: AeadAlgorithm,
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
) -> TlsResult<Vec<u8>> {
    let result = match alg {
        AeadAlgorithm::Aes128Gcm | AeadAlgorithm::Aes256Gcm => {
            tls_crypto_aes_gcm::open(key, nonce, aad, ciphertext)
        }
        AeadAlgorithm::ChaCha20Poly1305 => {
            tls_crypto_chacha20poly1305::open(key, nonce, aad, ciphertext)
        }
    };
    result.map_err(|()| TlsError::Alert(AlertDescription::BadRecordMac))
}
