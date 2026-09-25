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

//! Transcript hash helpers.

use crate::tls_ids::HashAlgorithm;
use sha2::{Digest, Sha256, Sha384};

/// Computes the hash of an empty input (used by TLS 1.3 `Derive-Secret(..., "")`).
#[must_use]
pub fn hash_empty(alg: HashAlgorithm) -> Vec<u8> {
    hash_message(alg, &[])
}

/// Hashes `data` with the selected algorithm.
#[must_use]
pub fn hash_message(alg: HashAlgorithm, data: &[u8]) -> Vec<u8> {
    match alg {
        HashAlgorithm::Sha256 => Sha256::digest(data).to_vec(),
        HashAlgorithm::Sha384 => Sha384::digest(data).to_vec(),
    }
}

/// Incremental transcript hasher.
#[derive(Clone)]
pub struct HashCtx {
    alg: HashAlgorithm,
    sha256: Sha256,
    sha384: Sha384,
}

impl HashCtx {
    /// Creates a new hasher.
    #[must_use]
    pub fn new(alg: HashAlgorithm) -> Self {
        Self {
            alg,
            sha256: Sha256::new(),
            sha384: Sha384::new(),
        }
    }

    /// Hash algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> HashAlgorithm {
        self.alg
    }

    /// Feeds more handshake bytes.
    pub fn update(&mut self, data: &[u8]) {
        match self.alg {
            HashAlgorithm::Sha256 => self
                .sha256
                .update(data),
            HashAlgorithm::Sha384 => self
                .sha384
                .update(data),
        }
    }

    /// Current digest without consuming the hasher.
    #[must_use]
    pub fn current(&self) -> Vec<u8> {
        match self.alg {
            HashAlgorithm::Sha256 => self
                .sha256
                .clone()
                .finalize()
                .to_vec(),
            HashAlgorithm::Sha384 => self
                .sha384
                .clone()
                .finalize()
                .to_vec(),
        }
    }
}
