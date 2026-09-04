//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Client-side Mergen merge orchestration.

use crate::userclient::mergen::mergen_blockchain::{BlockChain, BlockChainUnit};
use crate::userclient::mergen::mergen_hash::generate_mergen_hash;

/// High-level client for deterministic merge ordering.
#[derive(Debug, Clone, Default)]
pub struct MergenClient {
    next_user_id: u64,
}

impl MergenClient {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(&self, left: impl AsRef<str>, right: impl AsRef<str>) -> String {
        let left = left.as_ref();
        let right = right.as_ref();

        let mut chain = BlockChain::new();
        for (index, chunk) in left.as_bytes().chunks(16).enumerate() {
            let content = String::from_utf8_lossy(chunk).into_owned();
            chain.push_block(BlockChainUnit::new(
                index << 4,
                content.chars().count() as u32,
                1,
                content,
                (index + 1) as u64,
                1,
            ));
        }
        for (index, chunk) in right.as_bytes().chunks(16).enumerate() {
            let content = String::from_utf8_lossy(chunk).into_owned();
            chain.push_block(BlockChainUnit::new(
                index << 4,
                content.chars().count() as u32,
                1,
                content,
                (index + 1) as u64 + 1000,
                2,
            ));
        }

        chain.resolve()
    }

    /// Merges multiple text snapshots in deterministic order.
    pub fn merge_many<T, I>(&self, inputs: I) -> String
    where
        T: AsRef<str>,
        I: IntoIterator<Item = T>,
    {
        let mut chain = BlockChain::new();
        for (index, value) in inputs.into_iter().enumerate() {
            let text = value.as_ref();
            for (chunk_index, chunk) in text.as_bytes().chunks(16).enumerate() {
                let content = String::from_utf8_lossy(chunk).into_owned();
                chain.push_block(BlockChainUnit::new(
                    chunk_index << 4,
                    content.chars().count() as u32,
                    1,
                    content,
                    ((index + 1) * 1000 + chunk_index + 1) as u64,
                    (index + 1) as u64,
                ));
            }
        }
        chain.resolve()
    }

    #[inline]
    pub fn ordering_hash(&self, timestamp: u64, user_id: u64) -> u64 {
        generate_mergen_hash(timestamp, user_id)
    }
}
