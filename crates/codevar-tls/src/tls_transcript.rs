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

//! Handshake transcript hash (RFC 8446 §4.4.1).

use crate::tls_crypto_hash::{HashCtx, hash_message};
use crate::tls_ids::{HandshakeType, HashAlgorithm};

/// Running transcript of handshake messages.
#[derive(Clone)]
pub struct Transcript {
    alg: HashAlgorithm,
    hasher: HashCtx,
    /// Full transcript bytes (needed for TLS 1.2 PRF / EMS and binder construction).
    bytes: Vec<u8>,
}

impl Transcript {
    /// Creates an empty transcript for `alg`.
    #[must_use]
    pub fn new(alg: HashAlgorithm) -> Self {
        Self {
            alg,
            hasher: HashCtx::new(alg),
            bytes: Vec::new(),
        }
    }

    /// Hash algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> HashAlgorithm {
        self.alg
    }

    /// Rebinds the transcript to a new hash algorithm (after cipher suite selection).
    pub fn rebind_hash(&mut self, alg: HashAlgorithm) {
        if self.alg == alg {
            return;
        }
        self.alg = alg;
        self.hasher = HashCtx::new(alg);
        self.hasher.update(&self.bytes);
    }

    /// Appends a complete handshake message (`type || uint24 length || body`).
    pub fn add_message(&mut self, message: &[u8]) {
        self.hasher.update(message);
        self.bytes.extend_from_slice(message);
    }

    /// Replaces the first ClientHello with a TLS 1.3 `message_hash` after HRR
    /// (RFC 8446 §4.4.1).
    pub fn replace_client_hello_with_hash(&mut self, client_hello: &[u8]) {
        let ch_hash = hash_message(self.alg, client_hello);
        let mut synthetic = Vec::with_capacity(4 + ch_hash.len());
        synthetic.push(HandshakeType::MessageHash as u8);
        synthetic.push(0);
        synthetic.push(0);
        synthetic.push(ch_hash.len() as u8);
        synthetic.extend_from_slice(&ch_hash);

        // Rebuild: drop original ClientHello bytes, keep anything after it is not expected yet.
        self.bytes.clear();
        self.hasher = HashCtx::new(self.alg);
        self.add_message(&synthetic);
    }

    /// Current transcript hash.
    #[must_use]
    pub fn hash(&self) -> Vec<u8> {
        self.hasher.current()
    }

    /// Raw concatenated handshake bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_crypto_hash::hash_message;

    fn hs_msg(ty: HandshakeType, body: &[u8]) -> Vec<u8> {
        let mut out = vec![ty as u8];
        let len = body.len();
        out.push((len >> 16) as u8);
        out.push((len >> 8) as u8);
        out.push(len as u8);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn empty_transcript_hashes_to_digest_of_empty_input() {
        for alg in [HashAlgorithm::Sha256, HashAlgorithm::Sha384] {
            let t = Transcript::new(alg);
            assert_eq!(t.algorithm(), alg);
            assert!(t.bytes().is_empty());
            assert_eq!(t.hash(), hash_message(alg, b""));
        }
    }

    #[test]
    fn add_message_matches_one_shot_hash_of_concatenation() {
        let mut t = Transcript::new(HashAlgorithm::Sha256);
        let m1 = hs_msg(HandshakeType::ClientHello, b"client-hello-body");
        let m2 = hs_msg(HandshakeType::ServerHello, b"server-hello");
        let m3 = hs_msg(HandshakeType::Finished, &[0u8; 12]);

        t.add_message(&m1);
        assert_eq!(t.bytes(), &m1[..]);
        assert_eq!(t.hash(), hash_message(HashAlgorithm::Sha256, &m1));

        t.add_message(&m2);
        t.add_message(&m3);
        let mut all = m1.clone();
        all.extend_from_slice(&m2);
        all.extend_from_slice(&m3);
        assert_eq!(t.bytes(), &all[..]);
        assert_eq!(t.hash(), hash_message(HashAlgorithm::Sha256, &all));
    }

    #[test]
    fn rebind_hash_recomputes_digest_over_same_bytes() {
        let mut t = Transcript::new(HashAlgorithm::Sha256);
        let m1 = hs_msg(HandshakeType::ClientHello, b"abc");
        t.add_message(&m1);
        let sha256_digest = t.hash();

        t.rebind_hash(HashAlgorithm::Sha384);
        assert_eq!(t.algorithm(), HashAlgorithm::Sha384);
        assert_eq!(t.bytes(), &m1[..], "raw bytes survive the rebind");
        assert_eq!(t.hash(), hash_message(HashAlgorithm::Sha384, &m1));
        assert_ne!(t.hash(), sha256_digest);

        // Later messages still chain onto the rebound hasher.
        let m2 = hs_msg(HandshakeType::EncryptedExtensions, b"ee");
        t.add_message(&m2);
        let mut all = m1.clone();
        all.extend_from_slice(&m2);
        assert_eq!(t.hash(), hash_message(HashAlgorithm::Sha384, &all));
    }

    #[test]
    fn rebind_to_same_algorithm_is_a_no_op() {
        let mut t = Transcript::new(HashAlgorithm::Sha256);
        t.add_message(&hs_msg(HandshakeType::ClientHello, b"same"));
        let before = t.hash();
        t.rebind_hash(HashAlgorithm::Sha256);
        assert_eq!(t.hash(), before);
        assert_eq!(t.algorithm(), HashAlgorithm::Sha256);
    }

    #[test]
    fn replace_client_hello_with_hash_builds_synthetic_message_hash() {
        let mut t = Transcript::new(HashAlgorithm::Sha256);
        let client_hello = hs_msg(HandshakeType::ClientHello, b"first ch");
        t.add_message(&client_hello);

        t.replace_client_hello_with_hash(&client_hello);
        let ch_hash = hash_message(HashAlgorithm::Sha256, &client_hello);

        let mut expected = vec![HandshakeType::MessageHash as u8, 0, 0, ch_hash.len() as u8];
        expected.extend_from_slice(&ch_hash);
        assert_eq!(t.bytes(), &expected[..]);
        assert_eq!(t.hash(), hash_message(HashAlgorithm::Sha256, &expected));
        assert_eq!(t.bytes()[0], HandshakeType::MessageHash as u8);

        // Hashing the message_hash synthetic message equals hashing it directly.
        let direct = hash_message(HashAlgorithm::Sha256, &expected);
        assert_eq!(t.hash(), direct);
    }

    #[test]
    fn replace_client_hello_discards_previous_messages() {
        let mut t = Transcript::new(HashAlgorithm::Sha256);
        t.add_message(&hs_msg(HandshakeType::ClientHello, b"garbage before"));
        let hello = hs_msg(HandshakeType::ClientHello, b"real hello");
        t.replace_client_hello_with_hash(&hello);
        let expected_len = 4 + 32;
        assert_eq!(t.bytes().len(), expected_len);
        // Bytes recorded before the replacement are gone.
        assert!(!t.bytes().windows(7).any(|w| w == &b"garbage"[..]));
    }

    #[test]
    fn clone_diverges_independently() {
        let mut t = Transcript::new(HashAlgorithm::Sha256);
        t.add_message(&hs_msg(HandshakeType::ClientHello, b"base"));
        let mut c = t.clone();
        t.add_message(&hs_msg(HandshakeType::ServerHello, b"s"));
        c.add_message(&hs_msg(HandshakeType::Certificate, b"c"));
        assert_ne!(t.hash(), c.hash());
        assert!(t.bytes().len() > 4 && c.bytes().len() > 4);
        assert_ne!(t.bytes(), c.bytes());
    }
}
