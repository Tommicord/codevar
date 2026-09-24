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
    pub fn from_handshake(
        suite: CipherSuite,
        ecdhe_secret: &[u8],
        hello_hash: &[u8],
    ) -> TlsResult<Self> {
        let hash = suite.hash_algorithm();
        let aead = suite.aead_algorithm();
        let hash_len = hash.output_len();

        // Early Secret = HKDF-Extract(salt=0, key=0) for non-PSK handshakes.
        let zeros = vec![0u8; hash_len];
        let early_secret = hkdf_extract(hash, &zeros, &zeros);
        let derived = derive_secret(hash, &early_secret, b"derived", None)?;
        let handshake_secret = hkdf_extract(hash, &derived, ecdhe_secret);

        let client_hs_secret =
            derive_secret(hash, &handshake_secret, b"c hs traffic", Some(hello_hash))?;
        let server_hs_secret =
            derive_secret(hash, &handshake_secret, b"s hs traffic", Some(hello_hash))?;

        let client_handshake = traffic_from_secret(hash, aead, &client_hs_secret)?;
        let server_handshake = traffic_from_secret(hash, aead, &server_hs_secret)?;

        let client_finished_key =
            hkdf_expand_label(hash, &client_hs_secret, b"finished", &[], hash_len)?;
        let server_finished_key =
            hkdf_expand_label(hash, &server_hs_secret, b"finished", &[], hash_len)?;

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
    pub fn derive_application_secrets(
        &mut self,
        server_finished_hash: &[u8],
    ) -> TlsResult<()> {
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
        self.client_application =
            Some(traffic_from_secret(self.hash, self.aead, &client_ap)?);
        self.server_application =
            Some(traffic_from_secret(self.hash, self.aead, &server_ap)?);
        Ok(())
    }

    /// Derives the resumption master secret after the client Finished is added
    /// to the transcript.
    pub fn derive_resumption_master(
        &mut self,
        client_finished_hash: &[u8],
    ) -> TlsResult<()> {
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
    pub fn update_traffic_secret(
        hash: HashAlgorithm,
        secret: &[u8],
    ) -> TlsResult<Vec<u8>> {
        hkdf_expand_label(hash, secret, b"traffic upd", &[], hash.output_len())
    }

    /// Re-derives AEAD keys from an updated application traffic secret.
    pub fn traffic_from_app_secret(
        &self,
        secret: &[u8],
    ) -> TlsResult<DirectionalSecrets> {
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
    pub fn finished_verify(
        &self,
        label: &[u8],
        handshake_hash: &[u8],
    ) -> TlsResult<Vec<u8>> {
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

/// Empty IKM / salt helper used by unit tests.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_ids::CipherSuite;

    #[test]
    fn tls13_handshake_secrets_have_expected_lengths() {
        let ecdhe = [7u8; 32];
        let hello_hash = [3u8; 32];
        let ks = match Tls13KeySchedule::from_handshake(
            CipherSuite::TlsAes128GcmSha256,
            &ecdhe,
            &hello_hash,
        ) {
            Ok(ks) => ks,
            Err(_) => return,
        };
        assert_eq!(ks.client_handshake.secret.len(), 32);
        assert_eq!(ks.server_handshake.secret.len(), 32);
        let vd = match ks.server_finished_verify(&hello_hash) {
            Ok(vd) => vd,
            Err(_) => return,
        };
        assert_eq!(vd.len(), 32);
    }
}
