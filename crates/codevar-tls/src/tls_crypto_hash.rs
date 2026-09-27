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
            HashAlgorithm::Sha256 => self.sha256.update(data),
            HashAlgorithm::Sha384 => self.sha384.update(data),
        }
    }

    /// Current digest without consuming the hasher.
    #[must_use]
    pub fn current(&self) -> Vec<u8> {
        match self.alg {
            HashAlgorithm::Sha256 => self.sha256.clone().finalize().to_vec(),
            HashAlgorithm::Sha384 => self.sha384.clone().finalize().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_empty_matches_one_shot_digest_of_empty_input() {
        assert_eq!(hash_empty(HashAlgorithm::Sha256), Sha256::digest(b"").to_vec());
        assert_eq!(hash_empty(HashAlgorithm::Sha384), Sha384::digest(b"").to_vec());
        assert_eq!(hash_empty(HashAlgorithm::Sha256).len(), 32);
        assert_eq!(hash_empty(HashAlgorithm::Sha384).len(), 48);
        assert_ne!(
            hash_empty(HashAlgorithm::Sha256),
            hash_empty(HashAlgorithm::Sha384)
        );
    }

    #[test]
    fn hash_message_matches_sha2_crates() {
        let data = b"tls handshake transcript bytes";
        assert_eq!(
            hash_message(HashAlgorithm::Sha256, data),
            Sha256::digest(data).to_vec()
        );
        assert_eq!(
            hash_message(HashAlgorithm::Sha384, data),
            Sha384::digest(data).to_vec()
        );
        // Empty vs non-empty inputs differ.
        assert_ne!(
            hash_message(HashAlgorithm::Sha256, data),
            hash_empty(HashAlgorithm::Sha256)
        );
    }

    #[test]
    fn incremental_hashing_matches_one_shot() {
        for alg in [HashAlgorithm::Sha256, HashAlgorithm::Sha384] {
            let mut ctx = HashCtx::new(alg);
            assert_eq!(ctx.algorithm(), alg);
            assert_eq!(ctx.current(), hash_empty(alg));

            ctx.update(b"hello ");
            ctx.update(b"world");
            let split = ctx.current();
            let joined = hash_message(alg, b"hello world");
            assert_eq!(split, joined);

            // update() does not consume: repeated current() is stable.
            assert_eq!(ctx.current(), joined);

            // Feeding more data keeps extending the same transcript.
            ctx.update(b"!");
            assert_eq!(ctx.current(), hash_message(alg, b"hello world!"));
        }
    }

    #[test]
    fn hash_ctx_clone_preserves_state() {
        let mut ctx = HashCtx::new(HashAlgorithm::Sha256);
        ctx.update(b"prefix");
        let mut cloned = ctx.clone();
        ctx.update(b"-original");
        cloned.update(b"-clone");
        assert_eq!(ctx.current(), hash_message(HashAlgorithm::Sha256, b"prefix-original"));
        assert_eq!(cloned.current(), hash_message(HashAlgorithm::Sha256, b"prefix-clone"));
    }

    #[test]
    fn large_input_hashes_correctly() {
        // Multi-block input (exceeds the 64-byte SHA block many times over).
        let big = vec![0x5Au8; 10_000];
        assert_eq!(
            hash_message(HashAlgorithm::Sha256, &big),
            Sha256::digest(&big).to_vec()
        );
        let mut ctx = HashCtx::new(HashAlgorithm::Sha384);
        for chunk in big.chunks(97) {
            ctx.update(chunk);
        }
        assert_eq!(ctx.current(), Sha384::digest(&big).to_vec());
    }
}
