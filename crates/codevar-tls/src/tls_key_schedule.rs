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

use crate::tls_aead::AeadKey;
use crate::tls_error::TlsResult;
use crate::tls_hkdf::{derive_secret, hkdf_expand_label, hkdf_extract};
use crate::tls_ids::{AeadAlgorithm, CipherSuite, HashAlgorithm};
use crate::tls_prf::{hmac_hash, tls12_prf};
use crate::tls_record::TrafficKeys;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Directional traffic secrets derived from a TLS 1.3 secret.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct DirectionalSecrets {
    /// Traffic secret (for KeyUpdate / further derivation).
    pub secret: Vec<u8>,
    /// AEAD key.
    #[zeroize(skip)]
    pub key: AeadKey,
}

impl DirectionalSecrets {
    /// Builds traffic keys for the record layer (sequence starts at 0).
    #[must_use]
    pub fn traffic_keys(&self) -> TrafficKeys {
        TrafficKeys::new(self.key.clone())
    }
}

/// Full TLS 1.3 key schedule state after a successful (EC)DHE handshake.
#[derive(Clone, ZeroizeOnDrop)]
pub struct Tls13KeySchedule {
    #[zeroize(skip)]
    hash: HashAlgorithm,
    #[zeroize(skip)]
    aead: AeadAlgorithm,
    /// Handshake traffic secrets (client / server).
    #[zeroize(skip)]
    pub client_handshake: DirectionalSecrets,
    #[zeroize(skip)]
    pub server_handshake: DirectionalSecrets,
    /// Application traffic secrets after Finished.
    #[zeroize(skip)]
    pub client_application: Option<DirectionalSecrets>,
    #[zeroize(skip)]
    pub server_application: Option<DirectionalSecrets>,
    /// Master secret (kept for resumption exporter / NewSessionTicket).
    master_secret: Vec<u8>,
    /// Resumption master secret (derived after client Finished is in the transcript).
    resumption_master_secret: Option<Vec<u8>>,
    /// Client Finished key (`finished` binder from client handshake traffic secret).
    client_finished_key: Vec<u8>,
    /// Server Finished key.
    server_finished_key: Vec<u8>,
}

impl Tls13KeySchedule {
    /// Derives handshake traffic secrets from the ECDHE shared secret.
    ///
    /// `hello_hash` is `Transcript-Hash(ClientHello...ServerHello)`.
    pub fn from_handshake(suite: CipherSuite, ecdhe_secret: &[u8], hello_hash: &[u8]) -> TlsResult<Self> {
        let hash = suite.hash_algorithm();
        let aead = suite.aead_algorithm();
        let hash_len = hash.output_len();

        // Early Secret = HKDF-Extract(salt=0, key=0) for non-PSK handshakes.
        let zeros = vec![0u8; hash_len];
        let early_secret = hkdf_extract(hash, &zeros, &zeros);
        let derived = derive_secret(hash, &early_secret, b"derived", None)?;
        let handshake_secret = hkdf_extract(hash, &derived, ecdhe_secret);

        let client_hs_secret = derive_secret(hash, &handshake_secret, b"c hs traffic", Some(hello_hash))?;
        let server_hs_secret = derive_secret(hash, &handshake_secret, b"s hs traffic", Some(hello_hash))?;

        let client_handshake = traffic_from_secret(hash, aead, &client_hs_secret)?;
        let server_handshake = traffic_from_secret(hash, aead, &server_hs_secret)?;

        let client_finished_key = hkdf_expand_label(hash, &client_hs_secret, b"finished", &[], hash_len)?;
        let server_finished_key = hkdf_expand_label(hash, &server_hs_secret, b"finished", &[], hash_len)?;

        // Prepare master secret extract input: Derive-Secret(handshake, "derived", "")
        let hs_derived = derive_secret(hash, &handshake_secret, b"derived", None)?;
        let master_secret = hkdf_extract(hash, &hs_derived, &zeros);

        Ok(Self {
            hash,
            aead,
            client_handshake,
            server_handshake,
            client_application: None,
            server_application: None,
            master_secret,
            resumption_master_secret: None,
            client_finished_key,
            server_finished_key,
        })
    }

    /// Derives application traffic secrets using the transcript hash up to and
    /// including the server Finished (RFC 8446 §7.1: `c/s ap traffic`).
    pub fn derive_application_secrets(&mut self, server_finished_hash: &[u8]) -> TlsResult<()> {
        let client_ap = derive_secret(
            self.hash,
            &self.master_secret,
            b"c ap traffic",
            Some(server_finished_hash),
        )?;
        let server_ap = derive_secret(
            self.hash,
            &self.master_secret,
            b"s ap traffic",
            Some(server_finished_hash),
        )?;
        self.client_application = Some(traffic_from_secret(self.hash, self.aead, &client_ap)?);
        self.server_application = Some(traffic_from_secret(self.hash, self.aead, &server_ap)?);
        Ok(())
    }

    /// Derives the resumption master secret after the client Finished is added
    /// to the transcript.
    pub fn derive_resumption_master(&mut self, client_finished_hash: &[u8]) -> TlsResult<()> {
        let rms = derive_secret(
            self.hash,
            &self.master_secret,
            b"res master",
            Some(client_finished_hash),
        )?;
        self.resumption_master_secret = Some(rms);
        Ok(())
    }

    /// Computes Finished `verify_data` for the client.
    pub fn client_finished_verify(&self, transcript_hash: &[u8]) -> TlsResult<Vec<u8>> {
        hmac_hash(self.hash, &self.client_finished_key, transcript_hash)
    }

    /// Computes Finished `verify_data` for the server.
    pub fn server_finished_verify(&self, transcript_hash: &[u8]) -> TlsResult<Vec<u8>> {
        hmac_hash(self.hash, &self.server_finished_key, transcript_hash)
    }

    /// Hash algorithm.
    #[must_use]
    pub const fn hash_algorithm(&self) -> HashAlgorithm {
        self.hash
    }

    /// Performs a TLS 1.3 KeyUpdate on an application traffic secret.
    pub fn update_traffic_secret(hash: HashAlgorithm, secret: &[u8]) -> TlsResult<Vec<u8>> {
        hkdf_expand_label(hash, secret, b"traffic upd", &[], hash.output_len())
    }

    /// Re-derives AEAD keys from an updated application traffic secret.
    pub fn traffic_from_app_secret(&self, secret: &[u8]) -> TlsResult<DirectionalSecrets> {
        traffic_from_secret(self.hash, self.aead, secret)
    }
}

fn traffic_from_secret(
    hash: HashAlgorithm,
    aead: AeadAlgorithm,
    secret: &[u8],
) -> TlsResult<DirectionalSecrets> {
    let key = hkdf_expand_label(hash, secret, b"key", &[], aead.key_len())?;
    let iv = hkdf_expand_label(hash, secret, b"iv", &[], aead.iv_len())?;
    Ok(DirectionalSecrets {
        secret: secret.to_vec(),
        key: AeadKey::new(aead, key, iv)?,
    })
}

/// TLS 1.2 master secret and key block material.
#[derive(Clone, ZeroizeOnDrop)]
pub struct Tls12Keys {
    #[zeroize(skip)]
    hash: HashAlgorithm,
    #[zeroize(skip)]
    aead: AeadAlgorithm,
    /// 48-byte master secret.
    pub master_secret: Vec<u8>,
    /// Client write keys.
    #[zeroize(skip)]
    pub client_write: AeadKey,
    /// Server write keys.
    #[zeroize(skip)]
    pub server_write: AeadKey,
}

impl Tls12Keys {
    /// Computes the master secret and key block for an ECDHE handshake.
    ///
    /// When `extended_master_secret` is true, uses RFC 7627:
    /// `master_secret = PRF(pre_master, "extended master secret", session_hash)`.
    /// Otherwise uses the classic:
    /// `master_secret = PRF(pre_master, "master secret", client_random || server_random)`.
    pub fn derive(
        suite: CipherSuite,
        pre_master_secret: &[u8],
        client_random: &[u8; 32],
        server_random: &[u8; 32],
        session_hash: Option<&[u8]>,
    ) -> TlsResult<Self> {
        let hash = suite.hash_algorithm();
        let aead = suite.aead_algorithm();

        let master_secret = if let Some(sh) = session_hash {
            tls12_prf(hash, pre_master_secret, b"extended master secret", sh, 48)?
        } else {
            let mut seed = [0u8; 64];
            seed[..32].copy_from_slice(client_random);
            seed[32..].copy_from_slice(server_random);
            tls12_prf(hash, pre_master_secret, b"master secret", &seed, 48)?
        };

        // key_block = PRF(master, "key expansion", server_random || client_random)
        let key_len = aead.key_len();
        let iv_len = aead.tls12_fixed_iv_len();
        let need = 2 * key_len + 2 * iv_len;
        let mut seed = [0u8; 64];
        seed[..32].copy_from_slice(server_random);
        seed[32..].copy_from_slice(client_random);
        let key_block = tls12_prf(hash, &master_secret, b"key expansion", &seed, need)?;

        let (client_key, rest) = key_block.split_at(key_len);
        let (server_key, rest): (&[u8], &[u8]) = rest.split_at(key_len);
        let (client_iv, rest): (&[u8], &[u8]) = rest.split_at(iv_len);
        let (server_iv, _): (&[u8], &[u8]) = rest.split_at(iv_len);

        Ok(Self {
            hash,
            aead,
            master_secret,
            client_write: AeadKey::new(aead, client_key.to_vec(), client_iv.to_vec())?,
            server_write: AeadKey::new(aead, server_key.to_vec(), server_iv.to_vec())?,
        })
    }

    /// Computes TLS 1.2 Finished `verify_data` (12 bytes for AEAD suites).
    pub fn finished_verify(&self, label: &[u8], handshake_hash: &[u8]) -> TlsResult<Vec<u8>> {
        tls12_prf(self.hash, &self.master_secret, label, handshake_hash, 12)
    }

    /// Client write traffic keys.
    #[must_use]
    pub fn client_traffic(&self) -> TrafficKeys {
        TrafficKeys::new(self.client_write.clone())
    }

    /// Server write traffic keys.
    #[must_use]
    pub fn server_traffic(&self) -> TrafficKeys {
        TrafficKeys::new(self.server_write.clone())
    }

    /// Hash algorithm.
    #[must_use]
    pub const fn hash_algorithm(&self) -> HashAlgorithm {
        self.hash
    }

    /// AEAD algorithm.
    #[must_use]
    pub const fn aead_algorithm(&self) -> AeadAlgorithm {
        self.aead
    }
}

/// Labels for TLS 1.2 Finished verify_data.
pub mod finished_label {
    /// Client Finished label.
    pub const CLIENT: &[u8] = b"client finished";
    /// Server Finished label.
    pub const SERVER: &[u8] = b"server finished";
}

/// TLS 1.3 / TLS 1.2 key-schedule known-answer tests.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_ids::CipherSuite;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn tls13_handshake_secrets_match_known_answers() {
        // Inputs: ecdhe = 00..1f, hello_hash = 0xAA * 32.
        // Values computed independently with Python hmac/hashlib following RFC 8446 §7.1;
        // the early/derived secrets also match RFC 8448 §3.1/§3.2 known values.
        let ecdhe: Vec<u8> = (0u8..32).collect();
        let hello_hash = [0xAAu8; 32];
        let ks = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe,
            &hello_hash,
        )
        .unwrap();

        assert_eq!(ks.hash_algorithm(), HashAlgorithm::Sha256);
        assert_eq!(
            ks.client_handshake.secret,
            unhex("b4a822c84130bb5df25cbab7c34c5bf9629e298e0b76ac42f0c6f3e2668c1372")
        );
        assert_eq!(
            ks.server_handshake.secret,
            unhex("c774008ddd08971a1bb155d8eec043278fdc425171884c6a14b4983f06423cb5")
        );
        // Application secrets start out un-derived.
        assert!(ks.client_application.is_none());
        assert!(ks.server_application.is_none());

        // Handshake traffic key/iv for the client direction.
        assert_eq!(
            ks.client_handshake.key.algorithm(),
            AeadAlgorithm::Aes128Gcm
        );
        assert_eq!(
            ks.client_handshake.key.tls12_salt(),
            unhex("ab9ac6645231028fe8482919")
        );

        // Traffic keys start at sequence number 0.
        let tk = ks.client_handshake.traffic_keys();
        assert_eq!(tk.seq, 0);
        assert_eq!(tk.aead.algorithm(), AeadAlgorithm::Aes128Gcm);
    }

    #[test]
    fn tls13_application_and_resumption_secrets_match_known_answers() {
        let ecdhe: Vec<u8> = (0u8..32).collect();
        let mut ks = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe,
            &[0xAAu8; 32],
        )
        .unwrap();

        let server_finished_hash = [0xBBu8; 32];
        ks.derive_application_secrets(&server_finished_hash).unwrap();
        let (cap_secret, sap_secret) = {
            let cap = ks.client_application.as_ref().unwrap();
            let sap = ks.server_application.as_ref().unwrap();
            (cap.secret.clone(), sap.secret.clone())
        };
        assert_eq!(
            cap_secret,
            unhex("3c80122a7ecbc5d634e8a0cc7b2c43bfceb3143c561e7cb33aaff8c25d7b22e2")
        );
        assert_eq!(
            sap_secret,
            unhex("9fd216e69458a8e41e6173c5bd8cfe04c00b7487a223a7f402a70b349844ef9f")
        );
        // Handshake secrets are untouched by application derivation.
        assert_eq!(
            ks.client_handshake.secret,
            unhex("b4a822c84130bb5df25cbab7c34c5bf9629e298e0b76ac42f0c6f3e2668c1372")
        );

        // KeyUpdate on an application traffic secret.
        let updated = Tls13KeySchedule::update_traffic_secret(HashAlgorithm::Sha256, &cap_secret)
        .unwrap();
        assert_eq!(
            updated,
            unhex("db40e79dd0f41c2493040485f48f7736fa463a88b524f58f911640f606da12aa")
        );
        // Updated secret feeds new traffic keys of the same suite.
        let next = ks.traffic_from_app_secret(&updated).unwrap();
        assert_eq!(next.secret, updated);
        assert_eq!(next.key.algorithm(), AeadAlgorithm::Aes128Gcm);
        assert_eq!(next.traffic_keys().seq, 0);
        // One update round differs from the next (no fixed point in one step).
        let updated2 =
            Tls13KeySchedule::update_traffic_secret(HashAlgorithm::Sha256, &updated).unwrap();
        assert_ne!(updated, updated2);

        // Resumption master secret derives from the client Finished hash.
        let client_finished_hash = [0xCCu8; 32];
        ks.derive_resumption_master(&client_finished_hash).unwrap();
        // Deriving twice yields the same result (stateless re-computation).
        ks.derive_resumption_master(&client_finished_hash).unwrap();
    }

    #[test]
    fn tls13_finished_verify_matches_known_answers() {
        let ecdhe: Vec<u8> = (0u8..32).collect();
        let ks = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe,
            &[0xAAu8; 32],
        )
        .unwrap();
        let transcript = [0xDDu8; 32];
        let client = ks.client_finished_verify(&transcript).unwrap();
        let server = ks.server_finished_verify(&transcript).unwrap();
        assert_eq!(
            client,
            unhex("04a3e650913e837206d65d4e32ecbe9aac8a785b659349cc7a0ced9341b02d78")
        );
        assert_eq!(
            server,
            unhex("bbed8eec41dde617e4d2035e9ef35c9c603113ba72c187a26f93d294d4f9c23d")
        );
        assert_ne!(client, server);
        // Different transcripts give different verify_data.
        let other = ks.client_finished_verify(&[0xEEu8; 32]).unwrap();
        assert_ne!(client, other);
        assert_eq!(other.len(), 32);
    }

    #[test]
    fn tls13_sha384_suite_uses_48_byte_secrets() {
        let ecdhe: Vec<u8> = (0u8..48).collect();
        let mut ks = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes256GcmSha384,
            &ecdhe,
            &[0x55u8; 48],
        )
        .unwrap();
        assert_eq!(ks.hash_algorithm(), HashAlgorithm::Sha384);
        assert_eq!(ks.client_handshake.secret.len(), 48);
        assert_eq!(ks.server_handshake.secret.len(), 48);
        assert_eq!(ks.client_handshake.key.algorithm(), AeadAlgorithm::Aes256Gcm);
        assert_eq!(
            ks.client_handshake.key.tls12_salt().len(),
            AeadAlgorithm::Aes256Gcm.iv_len()
        );
        assert_eq!(ks.client_finished_verify(&[0x11u8; 48]).unwrap().len(), 48);

        ks.derive_application_secrets(&[0x22u8; 48]).unwrap();
        let cap = ks.client_application.as_ref().unwrap();
        assert_eq!(cap.secret.len(), 48);
        assert_eq!(cap.key.algorithm(), AeadAlgorithm::Aes256Gcm);
    }

    #[test]
    fn tls13_chacha_suite_uses_chacha_aead() {
        let ecdhe: Vec<u8> = (0u8..32).collect();
        let ks = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsChacha20Poly1305Sha256,
            &ecdhe,
            &[0xAAu8; 32],
        )
        .unwrap();
        assert_eq!(
            ks.client_handshake.key.algorithm(),
            AeadAlgorithm::ChaCha20Poly1305
        );
        assert_eq!(
            ks.server_handshake.key.algorithm(),
            AeadAlgorithm::ChaCha20Poly1305
        );
    }

    #[test]
    fn tls13_different_inputs_yield_different_secrets() {
        let ecdhe_a: Vec<u8> = (0u8..32).collect();
        let ecdhe_b: Vec<u8> = (1u8..33).collect();
        let a = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe_a,
            &[0xAAu8; 32],
        )
        .unwrap();
        let b = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe_b,
            &[0xAAu8; 32],
        )
        .unwrap();
        assert_ne!(a.client_handshake.secret, b.client_handshake.secret);

        let c = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe_a,
            &[0xBBu8; 32],
        )
        .unwrap();
        assert_ne!(a.client_handshake.secret, c.client_handshake.secret);
        // Deterministic for identical inputs.
        let d = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe_a,
            &[0xAAu8; 32],
        )
        .unwrap();
        assert_eq!(a.client_handshake.secret, d.client_handshake.secret);
    }

    #[test]
    fn tls12_master_secret_and_key_block_known_answers() {
        let pms = [0x03u8; 48];
        let client_random = [0xC1u8; 32];
        let server_random = [0x5Au8; 32];
        let keys = Tls12Keys::derive(
            CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
            &pms,
            &client_random,
            &server_random,
            None,
        )
        .unwrap();

        assert_eq!(keys.hash_algorithm(), HashAlgorithm::Sha256);
        assert_eq!(keys.aead_algorithm(), AeadAlgorithm::Aes128Gcm);
        assert_eq!(
            keys.master_secret,
            unhex(
                "73d201864b0b8ff114471583b7c7b04f22841a4713bd5b69a81dc97a4f07b1a8\
                 f9eb7cc115475731ab0dc83f99cb2bdc"
            )
        );
        // key_block = PRF(master, "key expansion", server_random || client_random):
        // 16-byte client key || 16-byte server key || 4-byte client iv || 4-byte server iv.
        assert_eq!(keys.client_write.tls12_salt(), unhex("7eab71c2"));
        assert_eq!(keys.server_write.tls12_salt(), unhex("218f490f"));
        assert_eq!(keys.client_write.algorithm(), AeadAlgorithm::Aes128Gcm);

        // Finished verify_data is 12 bytes and differs per label/direction.
        let transcript = [0xEEu8; 32];
        let client_fin = keys
            .finished_verify(finished_label::CLIENT, &transcript)
            .unwrap();
        let server_fin = keys
            .finished_verify(finished_label::SERVER, &transcript)
            .unwrap();
        assert_eq!(client_fin, unhex("2c5952b9dfb3fe31b9972394"));
        assert_eq!(server_fin, unhex("ae9c11a4b52ef6fb703edec7"));
        assert_eq!(client_fin.len(), 12);
        assert_ne!(client_fin, server_fin);

        // Traffic key wrappers start at sequence 0.
        assert_eq!(keys.client_traffic().seq, 0);
        assert_eq!(keys.server_traffic().seq, 0);
    }

    #[test]
    fn tls12_extended_master_secret_known_answer() {
        let pms = [0x03u8; 48];
        let session_hash = [0x99u8; 32];
        let ems_keys = Tls12Keys::derive(
            CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
            &pms,
            &[0xC1u8; 32],
            &[0x5Au8; 32],
            Some(&session_hash),
        )
        .unwrap();
        assert_eq!(
            ems_keys.master_secret,
            unhex(
                "5834c02ac9974ef56fe7fd14dd7991082ea5c79872e8836ac1a761211a6be38d\
                 749d1a128033adb5d5eb74eca8c5c74e"
            )
        );

        // EMS and classic derivation must differ for the same inputs.
        let classic = Tls12Keys::derive(
            CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
            &pms,
            &[0xC1u8; 32],
            &[0x5Au8; 32],
            None,
        )
        .unwrap();
        assert_ne!(ems_keys.master_secret, classic.master_secret);
        // Classic derivation is deterministic and independent of session hash.
        let classic2 = Tls12Keys::derive(
            CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
            &pms,
            &[0xC1u8; 32],
            &[0x5Au8; 32],
            None,
        )
        .unwrap();
        assert_eq!(classic.master_secret, classic2.master_secret);
        assert_ne!(
            classic.master_secret,
            Tls12Keys::derive(
                CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
                &pms,
                &[0xC2u8; 32],
                &[0x5Au8; 32],
                None,
            )
            .unwrap()
            .master_secret
        );
    }

    #[test]
    fn tls12_chacha_suite_key_block_layout() {
        // ChaCha20-Poly1305 uses 32-byte keys and 12-byte fixed IVs.
        let keys = Tls12Keys::derive(
            CipherSuite::TlsEcdheRsaWithChacha20Poly1305Sha256,
            &[0x03u8; 48],
            &[0xC1u8; 32],
            &[0x5Au8; 32],
            None,
        )
        .unwrap();
        assert_eq!(keys.aead_algorithm(), AeadAlgorithm::ChaCha20Poly1305);
        // From key_block: client iv at offset 64, server iv at offset 76.
        assert_eq!(
            keys.client_write.tls12_salt(),
            unhex("4ba2875c3528896ae6f666f0")
        );
        assert_eq!(keys.client_write.tls12_salt().len(), 12);
        assert_eq!(keys.server_write.tls12_salt().len(), 12);
        // Finished verify still yields 12 bytes.
        assert_eq!(
            keys.finished_verify(finished_label::CLIENT, &[0xEEu8; 32])
                .unwrap()
                .len(),
            12
        );
    }

    #[test]
    fn tls12_finished_verify_is_direction_and_transcript_sensitive() {
        let keys = Tls12Keys::derive(
            CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
            &[0x07u8; 48],
            &[0x11u8; 32],
            &[0x22u8; 32],
            None,
        )
        .unwrap();
        let t1 = keys
            .finished_verify(finished_label::CLIENT, &[0x01u8; 32])
            .unwrap();
        let t2 = keys
            .finished_verify(finished_label::CLIENT, &[0x02u8; 32])
            .unwrap();
        let t3 = keys
            .finished_verify(finished_label::SERVER, &[0x01u8; 32])
            .unwrap();
        assert_ne!(t1, t2);
        assert_ne!(t1, t3);
        // Custom labels (e.g. TLS 1.3 exporters) still produce 12 bytes.
        assert_eq!(
            keys.finished_verify(b"exporter master secret", &[0x01u8; 32])
                .unwrap()
                .len(),
            12
        );
    }

    #[test]
    fn finished_label_constants_are_correct() {
        assert_eq!(finished_label::CLIENT, b"client finished");
        assert_eq!(finished_label::SERVER, b"server finished");
    }

    #[test]
    fn directional_secrets_expose_traffic_keys() {
        let ecdhe: Vec<u8> = (0u8..32).collect();
        let ks = Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe,
            &[0xAAu8; 32],
        )
        .unwrap();
        let a = ks.client_handshake.traffic_keys();
        let b = ks.client_handshake.traffic_keys();
        assert_eq!(a.seq, 0);
        assert_eq!(b.seq, 0);
        assert_eq!(a.aead.algorithm(), b.aead.algorithm());
    }
}
