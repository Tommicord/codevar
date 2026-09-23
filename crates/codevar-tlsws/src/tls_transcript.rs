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
