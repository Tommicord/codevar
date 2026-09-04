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

//! Mergen hash system for conflict-free text merging.
//!
//! This module provides the 8-byte hash system that combines timestamp
//! and user ID information for deterministic conflict resolution in the
//! Mergen algorithm. The hash format is optimized for both CPU and GPU
//! processing with simple integer comparisons.

use std::cmp::Ordering;

/// Generates an 8-byte hash combining timestamp and user information.
///
/// # Hash Format
/// - Bits 63-32: Timestamp (4 bytes, reduced precision to seconds)
/// - Bits 31-0: User ID/hash (4 bytes)
///
/// # Arguments
///
/// * `timestamp` - Unix timestamp in seconds
/// * `user_id` - User identifier or hash
///
/// # Returns
///
/// Combined 8-byte hash for ordering and conflict resolution
#[inline]
pub fn generate_mergen_hash(timestamp: u64, user_id: u64) -> u64 {
    let timestamp_part = (timestamp & 0xFFFFFFFF) as u32;
    let user_part = (user_id & 0xFFFFFFFF) as u32;
    // Combine into 8-byte hash: [timestamp (high 32 bits)] [user_id (low 32 bits)]
    ((timestamp_part as u64) << 32) | (user_part as u64)
}

/// Extracts timestamp from mergen hash.
///
/// # Arguments
///
/// * `hash` - The mergen hash to extract from
///
/// # Returns
///
/// Timestamp component (4 bytes)
#[inline]
pub fn extract_timestamp(hash: u64) -> u32 {
    (hash >> 32) as u32
}

/// Extracts user ID from mergen hash.
///
/// # Arguments
///
/// * `hash` - The mergen hash to extract from
///
/// # Returns
///
/// User ID component (4 bytes)
#[inline]
pub fn extract_user_id(hash: u64) -> u32 {
    (hash & 0xFFFFFFFF) as u32
}

/// Compares two hashes for ordering.
///
/// Earlier timestamps win, with user ID as tiebreaker for deterministic ordering.
/// This provides a total ordering for conflict-free merge resolution.
///
/// # Arguments
///
/// * `hash1` - First hash to compare
/// * `hash2` - Second hash to compare
///
/// # Returns
///
/// Ordering relationship between the hashes
#[inline]
pub fn compare_hashes(hash1: u64, hash2: u64) -> Ordering {
    let ts1 = extract_timestamp(hash1);
    let ts2 = extract_timestamp(hash2);

    match ts1.cmp(&ts2) {
        Ordering::Equal => extract_user_id(hash1).cmp(&extract_user_id(hash2)),
        other => other,
    }
}

/// Computes content hash for a block of text using xxHash64.
///
/// This is separate from the mergen hash (timestamp + user ID) and is used
/// for detecting actual content changes and cache invalidation.
///
/// xxHash64 is a fast non-cryptographic hash algorithm with excellent
/// distribution and collision resistance properties.
///
/// # Arguments
///
/// * `content` - Text content to hash
///
/// # Returns
///
/// 64-bit content hash
pub fn compute_content_hash(content: &str) -> u64 {
    hash(content.as_bytes(), 0)
}

const PRIME1: u64 = 0x9e3779b185ebca87;
const PRIME2: u64 = 0xc2b2ae3d27d4eb4f;
const PRIME3: u64 = 0x165667b19e3779f9;
const PRIME4: u64 = 0x85ebca77c2b2ae63;
const PRIME5: u64 = 0x27d4eb2f165667c5;

/// xxHash64 implementation for 64-bit hash
///
/// This is a pure Rust implementation of the xxHash64 algorithm (XXH64),
/// which provides excellent speed and hash quality for non-cryptographic purposes.
///
/// # Arguments
///
/// * `data` - Input data to hash
/// * `seed` - Seed value for hash initialization (can be 0 for most uses)
///
/// # Returns
///
/// 64-bit hash value
fn hash(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut hash: u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(PRIME1).wrapping_add(PRIME2);
        let mut v2 = seed.wrapping_add(PRIME2);
        let mut v3 = seed.wrapping_add(0);
        let mut v4 = seed.wrapping_sub(PRIME1);

        let mut ptr = 0;
        while len - ptr >= 32 {
            let k1 = read_u64_le(&data[ptr..]);
            let k2 = read_u64_le(&data[ptr + 8..]);
            let k3 = read_u64_le(&data[ptr + 16..]);
            let k4 = read_u64_le(&data[ptr + 24..]);

            v1 = round(v1, k1);
            v2 = round(v2, k2);
            v3 = round(v3, k3);
            v4 = round(v4, k4);

            ptr += 32;
        }

        hash = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12).wrapping_add(v4.rotate_left(18)));

        hash = merge_round(hash, v1);
        hash = merge_round(hash, v2);
        hash = merge_round(hash, v3);
        hash = merge_round(hash, v4);
    } else {
        hash = seed.wrapping_add(PRIME5);
    }
    hash = hash.wrapping_add(len as u64);
    let remaining = len & 31;
    let ptr = len - remaining;
    if remaining >= 8 {
        let k1 = read_u64_le(&data[ptr..]);
        hash ^= round(0, k1);
        hash = hash.wrapping_mul(PRIME1);
        hash = hash.wrapping_add(PRIME4);

        if remaining >= 16 {
            let k2 = read_u64_le(&data[ptr + 8..]);
            hash ^= round(0, k2);
            hash = hash.wrapping_mul(PRIME1);
            hash = hash.wrapping_add(PRIME4);

            if remaining >= 24 {
                let k3 = read_u64_le(&data[ptr + 16..]);
                hash ^= round(0, k3);
                hash = hash.wrapping_mul(PRIME1);
                hash = hash.wrapping_add(PRIME4);
            }
        }
    }
    let ptr = len - (remaining & 7);
    for &byte in &data[ptr..] {
        hash ^= (byte as u64).wrapping_mul(PRIME5);
        hash = hash.wrapping_mul(PRIME3);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(PRIME2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(PRIME3);
    hash ^= hash >> 32;
    hash
}

/// Reads a little-endian u64 from a byte slice.
#[inline]
fn read_u64_le(data: &[u8]) -> u64 {
    u64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]))
}

/// Round function for xxHash64.
#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME4))
        .wrapping_mul(PRIME1)
}

/// Merge round function for xxHash64.
#[inline]
fn merge_round(acc: u64, val: u64) -> u64 {
    acc.wrapping_add(val.rotate_left(23).wrapping_mul(PRIME2))
        .wrapping_add(PRIME3)
}
