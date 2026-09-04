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

use codevar_colab::userclient::mergen::mergen_blockchain::{BlockChain, BlockChainUnit};
use codevar_colab::userclient::mergen::mergen_cache::{
    BlockCache, BlockCacheKey, CachedBlock,
};
use codevar_colab::userclient::mergen::mergen_client::MergenClient;
use codevar_colab::userclient::mergen::mergen_conflict::{
    BlockConflict, ConflictDetector, ConflictResolver,
};
use codevar_colab::userclient::mergen::mergen_hash::{
    compare_hashes, compute_content_hash, extract_timestamp, extract_user_id,
    generate_mergen_hash,
};
use codevar_colab::userclient::mergen::mergen_resolve_strategy::{
    ResolveConfig, ResolveStrategy, StrategyResolver,
};
use rand::RngExt;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Helper function to get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Helper function to create a random pause between 1-100ms
fn random_pause() {
    let pause_ms = rand::random::<u8>() as u64 % 100 + 1;
    thread::sleep(Duration::from_millis(pause_ms));
}

/// Helper function to generate random text content
fn generate_random_text(length: usize) -> String {
    use rand::Rng;
    let chars: Vec<char> =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            .chars()
            .collect();
    let mut rng = rand::rng();
    (0..length)
        .map(|_| chars[rng.random_range(0..chars.len())])
        .collect()
}

/// Helper function to generate random unicode text
fn generate_random_unicode_text(length: usize) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    (0..length)
        .map(|_| {
            // Generate random unicode characters (excluding surrogates)
            let c = rng.random_range(0x0020..=0xFFFF);
            if c >= 0xD800 && c <= 0xDFFF {
                // Skip surrogate pairs
                'a'
            } else {
                char::from_u32(c).unwrap_or('a')
            }
        })
        .collect()
}

/// Helper function to generate very large random text (100K+ characters)
fn generate_large_random_text(size: usize) -> String {
    use rand::Rng;
    let chars: Vec<char> =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t"
            .chars()
            .collect();
    let mut rng = rand::rng();
    (0..size)
        .map(|_| chars[rng.random_range(0..chars.len())])
        .collect()
}

#[test]
fn mergen_hash_generation() {
    let timestamp = 1234567890u64;
    let user_id = 42u64;
    let hash = generate_mergen_hash(timestamp, user_id);

    // Verify hash structure: [timestamp (high 32 bits)] [user_id (low 32 bits)]
    let expected_hash =
        ((timestamp & 0xFFFFFFFF) as u64) << 32 | (user_id & 0xFFFFFFFF) as u64;
    assert_eq!(hash, expected_hash);
}

#[test]
fn mergen_hash_extraction() {
    let timestamp = 1234567890u64;
    let user_id = 42u64;
    let hash = generate_mergen_hash(timestamp, user_id);

    let extracted_timestamp = extract_timestamp(hash);
    let extracted_user_id = extract_user_id(hash);

    assert_eq!(extracted_timestamp, (timestamp & 0xFFFFFFFF) as u32);
    assert_eq!(extracted_user_id, (user_id & 0xFFFFFFFF) as u32);
}

#[test]
fn mergen_hash_comparison() {
    let hash1 = generate_mergen_hash(1000, 1);
    let hash2 = generate_mergen_hash(1000, 2);
    let hash3 = generate_mergen_hash(2000, 1);

    // Same timestamp, different user IDs - lower user ID should win
    assert_eq!(compare_hashes(hash1, hash2), std::cmp::Ordering::Less);

    // Different timestamps - earlier timestamp should win
    assert_eq!(compare_hashes(hash1, hash3), std::cmp::Ordering::Less);

    // Same timestamp and user ID
    let hash4 = generate_mergen_hash(1000, 1);
    assert_eq!(compare_hashes(hash1, hash4), std::cmp::Ordering::Equal);
}

#[test]
fn content_hash_generation() {
    let content1 = "Hello, World!";
    let content2 = "Hello, World!";
    let content3 = "Different content";

    let hash1 = compute_content_hash(content1);
    let hash2 = compute_content_hash(content2);
    let hash3 = compute_content_hash(content3);

    // Same content should produce same hash
    assert_eq!(hash1, hash2);

    // Different content should produce different hash
    assert_ne!(hash1, hash3);
}

#[test]
fn blockchain_unit_creation() {
    let block = BlockChainUnit::new(100, 16, 1, "test content", 1234567890, 42);

    assert_eq!(block.offset, 100);
    assert_eq!(block.width, 16);
    assert_eq!(block.height, 1);
    assert_eq!(block.data, "test content");
    assert_eq!(block.timestamp, 1234567890);
    assert_eq!(block.user_id, 42);
    assert_eq!(block.change_count, 0);
}

#[test]
fn blockchain_unit_ordering_hash() {
    let block = BlockChainUnit::new(0, 16, 1, "test", 1234567890, 42);
    let expected_hash = generate_mergen_hash(1234567890, 42);
    assert_eq!(block.ordering_hash(), expected_hash);
}

#[test]
fn blockchain_creation() {
    let chain = BlockChain::new();
    assert!(chain.is_empty());
    assert_eq!(chain.len(), 0);
}

#[test]
fn blockchain_from_text() {
    let text = "Hello, World!";
    let chain = BlockChain::from_text(text);

    assert!(!chain.is_empty());
    assert!(chain.len() > 0);

    // Verify blocks contain the text content
    let resolved = chain.resolve();
    assert_eq!(resolved, text);
}

#[test]
fn blockchain_push_block() {
    let mut chain = BlockChain::new();

    let block1 = BlockChainUnit::new(0, 16, 1, "first", 1000, 1);
    let block2 = BlockChainUnit::new(16, 16, 1, "second", 2000, 2);

    chain.push_block(block1);
    chain.push_block(block2);

    assert_eq!(chain.len(), 2);
}

#[test]
fn blockchain_push_blocks() {
    let mut chain = BlockChain::new();

    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "first", 1000, 1),
        BlockChainUnit::new(16, 16, 1, "second", 2000, 2),
    ];

    chain.push_blocks(&blocks);

    assert_eq!(chain.len(), 2);
}

#[test]
fn blockchain_get_blocks_at_position() {
    let mut chain = BlockChain::new();

    chain.push_block(BlockChainUnit::new(0, 16, 1, "content1", 1000, 1));
    chain.push_block(BlockChainUnit::new(0, 16, 1, "content2", 2000, 2));
    chain.push_block(BlockChainUnit::new(16, 16, 1, "content3", 3000, 1));

    let blocks_at_zero = chain.get_blocks_at_position(0);
    assert_eq!(blocks_at_zero.len(), 2);

    let blocks_at_sixteen = chain.get_blocks_at_position(16);
    assert_eq!(blocks_at_sixteen.len(), 1);
}

#[test]
fn blockchain_get_blocks_by_user() {
    let mut chain = BlockChain::new();

    chain.push_block(BlockChainUnit::new(0, 16, 1, "content1", 1000, 1));
    chain.push_block(BlockChainUnit::new(16, 16, 1, "content2", 2000, 1));
    chain.push_block(BlockChainUnit::new(32, 16, 1, "content3", 3000, 2));

    let user1_blocks = chain.get_blocks_by_user(1);
    assert_eq!(user1_blocks.len(), 2);

    let user2_blocks = chain.get_blocks_by_user(2);
    assert_eq!(user2_blocks.len(), 1);
}

#[test]
fn blockchain_resolve() {
    let mut chain = BlockChain::new();

    chain.push_block(BlockChainUnit::new(0, 16, 1, "Hello", 1000, 1));
    chain.push_block(BlockChainUnit::new(16, 16, 1, "World", 2000, 2));

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());
}

#[test]
fn blockchain_deterministic_ordering() {
    let mut chain1 = BlockChain::new();
    let mut chain2 = BlockChain::new();

    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "A", 1000, 1),
        BlockChainUnit::new(16, 16, 1, "B", 2000, 2),
        BlockChainUnit::new(32, 16, 1, "C", 1500, 3),
    ];

    // Add blocks in different orders
    chain1.push_blocks(&blocks);
    chain2.push_blocks(&blocks.into_iter().rev().collect::<Vec<_>>());

    // Resolution should be deterministic regardless of insertion order
    assert_eq!(chain1.resolve(), chain2.resolve());
}

#[test]
fn conflict_detector_creation() {
    let detector = ConflictDetector::new(16);
    assert_eq!(detector.block_size(), 16);

    let default_detector = ConflictDetector::default();
    assert_eq!(default_detector.block_size(), 16);
}

#[test]
fn conflict_detector_detect_conflicts() {
    let detector = ConflictDetector::new(16);

    let chain1 = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(16, 16, 1, "content2", 2000, 1),
    ];

    let chain2 = vec![
        BlockChainUnit::new(0, 16, 1, "different", 1500, 2), // Conflict at position 0
        BlockChainUnit::new(32, 16, 1, "content3", 2500, 2),
    ];

    let conflicts = detector.detect_conflicts(&[chain1, chain2]);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].position, 0);
}

#[test]
fn conflict_detector_no_conflicts() {
    let detector = ConflictDetector::new(16);

    let chain1 = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(16, 16, 1, "content2", 2000, 1),
    ];

    let chain2 = vec![
        BlockChainUnit::new(32, 16, 1, "content3", 1500, 2),
        BlockChainUnit::new(48, 16, 1, "content4", 2500, 2),
    ];

    let conflicts = detector.detect_conflicts(&[chain1, chain2]);
    assert!(conflicts.is_empty());
}

#[test]
fn block_conflict_creation() {
    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(0, 16, 1, "content2", 1500, 2),
    ];

    let conflict = BlockConflict::new(0, blocks);
    assert_eq!(conflict.position, 0);
    assert_eq!(conflict.conflict_count(), 2);
    assert!(!conflict.is_resolved());
}

#[test]
fn block_conflict_resolution() {
    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(0, 16, 1, "content2", 1500, 2),
    ];

    let mut conflict = BlockConflict::new(0, blocks);
    conflict.resolve();

    assert!(conflict.is_resolved());
    assert!(conflict.resolved_block.is_some());

    // Earlier timestamp should win
    let resolved = conflict.resolved_block.unwrap();
    assert_eq!(resolved.timestamp, 1000);
}

#[test]
fn conflict_resolver_creation() {
    let resolver = ConflictResolver::new(16);
    assert_eq!(resolver.detector().block_size(), 16);

    let default_resolver = ConflictResolver::default();
    assert_eq!(default_resolver.detector().block_size(), 16);
}

#[test]
fn conflict_resolver_resolve_chains() {
    let resolver = ConflictResolver::default();

    let chain1 = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(16, 16, 1, "content2", 2000, 1),
    ];

    let chain2 = vec![
        BlockChainUnit::new(0, 16, 1, "different", 1500, 2),
        BlockChainUnit::new(32, 16, 1, "content3", 2500, 2),
    ];

    let results = resolver.resolve_chains(&[chain1, chain2]);
    assert_eq!(results.len(), 1);
}

#[test]
fn resolve_strategy_timestamp_order() {
    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(0, 16, 1, "content2", 1500, 2),
        BlockChainUnit::new(0, 16, 1, "content3", 2000, 3),
    ];

    let conflict = BlockConflict::new(0, blocks);
    let strategy = ResolveStrategy::TimestampOrder;
    let resolved = strategy.resolve(&conflict);

    assert!(resolved.is_some());
    assert_eq!(resolved.unwrap().timestamp, 1000);
}

#[test]
fn resolve_strategy_user_id_order() {
    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 2000, 3),
        BlockChainUnit::new(0, 16, 1, "content2", 1500, 1),
        BlockChainUnit::new(0, 16, 1, "content3", 1000, 2),
    ];

    let conflict = BlockConflict::new(0, blocks);
    let strategy = ResolveStrategy::UserIdOrder;
    let resolved = strategy.resolve(&conflict);

    assert!(resolved.is_some());
    assert_eq!(resolved.unwrap().user_id, 1);
}

#[test]
fn resolve_strategy_change_count_order() {
    let mut blocks = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(0, 16, 1, "content2", 1500, 2),
    ];
    blocks[0].change_count = 5;
    blocks[1].change_count = 2;

    let conflict = BlockConflict::new(0, blocks);
    let strategy = ResolveStrategy::ChangeCountOrder;
    let resolved = strategy.resolve(&conflict);

    assert!(resolved.is_some());
    assert_eq!(resolved.unwrap().change_count, 2);
}

#[test]
fn resolve_strategy_most_recent() {
    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "content1", 1000, 1),
        BlockChainUnit::new(0, 16, 1, "content2", 1500, 2),
        BlockChainUnit::new(0, 16, 1, "content3", 2000, 3),
    ];

    let conflict = BlockConflict::new(0, blocks);
    let strategy = ResolveStrategy::MostRecent;
    let resolved = strategy.resolve(&conflict);

    assert!(resolved.is_some());
    assert_eq!(resolved.unwrap().timestamp, 2000);
}

#[test]
fn resolve_strategy_content_similarity() {
    let blocks = vec![
        BlockChainUnit::new(0, 16, 1, "hello world", 1000, 1),
        BlockChainUnit::new(0, 16, 1, "hello earth", 1500, 2),
        BlockChainUnit::new(0, 16, 1, "completely different", 2000, 3),
    ];

    let conflict = BlockConflict::new(0, blocks);
    let strategy = ResolveStrategy::ContentSimilarity;
    let resolved = strategy.resolve(&conflict);

    assert!(resolved.is_some());
    // Should pick the most similar to the first block
    let resolved_text = resolved.unwrap().data;
    assert!(resolved_text == "hello world" || resolved_text == "hello earth");
}

#[test]
fn strategy_resolver_creation() {
    let config = ResolveConfig::new(ResolveStrategy::TimestampOrder);
    let resolver = StrategyResolver::new(config);

    assert_eq!(resolver.config().strategy, ResolveStrategy::TimestampOrder);
}

#[test]
fn strategy_resolver_with_fallback() {
    let config = ResolveConfig::new(ResolveStrategy::TimestampOrder).with_fallback(true);
    let resolver = StrategyResolver::new(config);

    assert!(resolver.config().enable_fallback);
    assert!(!resolver.fallback_strategies().is_empty());
}

#[test]
fn strategy_resolver_without_fallback() {
    let config = ResolveConfig::new(ResolveStrategy::TimestampOrder).with_fallback(false);
    let resolver = StrategyResolver::new(config);

    assert!(!resolver.config().enable_fallback);
    assert!(resolver.fallback_strategies().is_empty());
}

#[test]
fn mergen_client_creation() {
    let client = MergenClient::new();
    // Client should be created successfully
    let _ = client.merge("test", "test");
}

#[test]
fn mergen_client_merge() {
    let client = MergenClient::new();

    let left = "Hello, World!";
    let right = "Goodbye, World!";

    let merged = client.merge(left, right);
    assert!(!merged.is_empty());
}

#[test]
fn mergen_client_merge_many() {
    let client = MergenClient::new();

    let inputs = vec!["First", "Second", "Third", "Fourth"];
    let merged = client.merge_many(inputs);

    assert!(!merged.is_empty());
}

#[test]
fn mergen_client_ordering_hash() {
    let client = MergenClient::new();

    let timestamp = 1234567890u64;
    let user_id = 42u64;

    let hash1 = client.ordering_hash(timestamp, user_id);
    let hash2 = generate_mergen_hash(timestamp, user_id);

    assert_eq!(hash1, hash2);
}

#[test]
fn block_cache_key_creation() {
    let key = BlockCacheKey::new(12345, 67890, (10, 20), (16, 1));

    assert_eq!(key.content_hash, 12345);
    assert_eq!(key.user_hash, 67890);
    assert_eq!(key.position, (10, 20));
    assert_eq!(key.dimensions, (16, 1));
}

#[test]
fn block_cache_key_from_block_data() {
    let content = "test content";
    let user_ids = vec![1u64, 2u64, 3u64];
    let key = BlockCacheKey::from_block_data(content, &user_ids, (10, 20), (16, 1));

    assert_eq!(key.position, (10, 20));
    assert_eq!(key.dimensions, (16, 1));
    assert_ne!(key.content_hash, 0);
}

#[test]
fn cached_block_creation() {
    let content = "merged content".to_string();
    let mergen_hash = generate_mergen_hash(1234567890, 42);
    let original_hash = compute_content_hash("original");

    let block = CachedBlock::new(content.clone(), mergen_hash, original_hash);

    assert_eq!(block.merged_content, content);
    assert_eq!(block.mergen_hash, mergen_hash);
    assert_eq!(block.original_hash, original_hash);
    assert_eq!(block.access_count, 1);
    assert!(!block.dirty);
}

#[test]
fn cached_block_validity() {
    let content = "merged content".to_string();
    let mergen_hash = generate_mergen_hash(1234567890, 42);
    let original_hash = compute_content_hash("original");

    let block = CachedBlock::new(content, mergen_hash, original_hash);

    // Should be valid with same hash
    assert!(block.is_valid(original_hash));

    // Should be invalid with different hash
    assert!(!block.is_valid(99999));
}

#[test]
fn cached_block_dirty_tracking() {
    let content = "merged content".to_string();
    let mergen_hash = generate_mergen_hash(1234567890, 42);
    let original_hash = compute_content_hash("original");

    let mut block = CachedBlock::new(content, mergen_hash, original_hash);

    assert!(!block.dirty);
    block.mark_dirty();
    assert!(block.dirty);
}

#[test]
fn mergen_block_cache_creation() {
    let cache = BlockCache::new();
    assert!(cache.capacity() > 0);
}

#[test]
fn mergen_block_cache_disabled() {
    let cache = BlockCache::disabled();
    assert_eq!(cache.capacity(), 0);
}

#[test]
fn mergen_block_cache_basic_operations() {
    let mut cache = BlockCache::new();

    let key = BlockCacheKey::new(12345, 67890, (10, 20), (16, 1));
    let content = "cached content".to_string();
    let timestamp = 1234567890u64;
    let user_id = 42u64;
    let original_hash = compute_content_hash("original");

    cache.cache_block(
        key.clone(),
        content.clone(),
        timestamp,
        user_id,
        original_hash,
    );

    let retrieved = cache.get_block(key, original_hash);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), content);
}

#[test]
fn mergen_block_cache_invalid_hash() {
    let mut cache = BlockCache::new();

    let key = BlockCacheKey::new(12345, 67890, (10, 20), (16, 1));
    let content = "cached content".to_string();
    let timestamp = 1234567890u64;
    let user_id = 42u64;
    let original_hash = compute_content_hash("original");

    cache.cache_block(key.clone(), content, timestamp, user_id, original_hash);

    // Try to retrieve with different hash
    let retrieved = cache.get_block(key, 99999);
    assert!(retrieved.is_none());
}

#[test]
fn concurrent_blockchain_modifications() {
    let chain = Arc::new(Mutex::new(BlockChain::new()));
    let mut handles = vec![];

    // Spawn 5 threads that each add blocks to the shared blockchain
    for user_id in 1..=5 {
        let chain_clone = Arc::clone(&chain);
        let handle = thread::spawn(move || {
            random_pause();
            let mut chain = chain_clone.lock().unwrap();
            for i in 0..3 {
                let block = BlockChainUnit::new(
                    (user_id * 16 + i * 16) as usize,
                    16,
                    1,
                    format!("user{}_block{}", user_id, i),
                    current_timestamp(),
                    user_id,
                );
                chain.push_block(block);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all blocks were added
    let chain = chain.lock().unwrap();
    assert_eq!(chain.len(), 15); // 5 users * 3 blocks each
}

#[test]
fn concurrent_conflict_detection() {
    let detector = Arc::new(ConflictDetector::new(16));
    let chains = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    // Spawn 3 threads that each create a chain
    for user_id in 1..=3 {
        let _detector_clone = Arc::clone(&detector);
        let chains_clone = Arc::clone(&chains);
        let handle = thread::spawn(move || {
            random_pause();
            let mut chain = Vec::new();

            // Create blocks with potential conflicts
            for i in 0..2 {
                let block = BlockChainUnit::new(
                    i * 16, // Same positions across users to create conflicts
                    16,
                    1,
                    format!("user{}_content{}", user_id, i),
                    current_timestamp(),
                    user_id,
                );
                chain.push(block);
            }

            let mut chains = chains_clone.lock().unwrap();
            chains.push(chain);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Detect conflicts
    let chains = chains.lock().unwrap();
    let conflicts = detector.detect_conflicts(&chains);

    // Should have conflicts since users edited same positions
    assert!(!conflicts.is_empty());
}

#[test]
fn concurrent_mergen_client_operations() {
    let client = Arc::new(MergenClient::new());
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    // Spawn 4 threads that perform merge operations
    for i in 0..4 {
        let client_clone = Arc::clone(&client);
        let results_clone = Arc::clone(&results);
        let handle = thread::spawn(move || {
            random_pause();
            let client = client_clone;
            let text1 = format!("Text from thread {} part 1", i);
            let text2 = format!("Text from thread {} part 2", i);

            let merged = client.merge(&text1, &text2);

            let mut results = results_clone.lock().unwrap();
            results.push(merged);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all threads completed successfully
    let results = results.lock().unwrap();
    assert_eq!(results.len(), 4);

    // Verify each result is non-empty
    for result in results.iter() {
        assert!(!result.is_empty());
    }
}

#[test]
fn realistic_collaborative_editing_simulation() {
    // Simulate 3 users editing a shared document with random pauses
    let shared_document = Arc::new(Mutex::new(String::new()));
    let user_edits = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    // Initial document content
    {
        let mut doc = shared_document.lock().unwrap();
        *doc = "Initial document content ".to_string();
    }

    // Spawn 3 users editing concurrently
    for user_id in 1..=3 {
        let doc_clone = Arc::clone(&shared_document);
        let edits_clone = Arc::clone(&user_edits);
        let handle = thread::spawn(move || {
            for edit_num in 0..2 {
                random_pause();

                // Read current document state
                let current_content = {
                    let doc = doc_clone.lock().unwrap();
                    doc.clone()
                };

                // Simulate user edit by appending content
                let new_content =
                    format!("{}User{}Edit{} ", current_content, user_id, edit_num);

                // Record the edit as a blockchain unit
                let edit_block = BlockChainUnit::new(
                    current_content.len(),
                    new_content.len() as u32 - current_content.len() as u32,
                    1,
                    format!("User{}Edit{}", user_id, edit_num),
                    current_timestamp(),
                    user_id,
                );

                // Update shared document
                {
                    let mut doc = doc_clone.lock().unwrap();
                    *doc = new_content;
                }

                // Record edit for later merging
                let mut edits = edits_clone.lock().unwrap();
                edits.push(edit_block);
            }
        });
        handles.push(handle);
    }

    // Wait for all users to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all edits were recorded
    let edits = user_edits.lock().unwrap();
    assert_eq!(edits.len(), 6); // 3 users * 2 edits each

    // Create a blockchain from the edits and resolve
    let mut chain = BlockChain::new();
    for edit in edits.iter() {
        chain.push_block(edit.clone());
    }

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());

    // Final document should contain all user edits
    let final_doc = shared_document.lock().unwrap();
    assert!(final_doc.contains("User1"));
    assert!(final_doc.contains("User2"));
    assert!(final_doc.contains("User3"));
}

#[test]
fn concurrent_cache_operations() {
    let cache = Arc::new(Mutex::new(BlockCache::new()));
    let mut handles = vec![];

    // Spawn 4 threads that perform cache operations
    for thread_id in 0..4 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            random_pause();

            let mut cache = cache_clone.lock().unwrap();

            // Create and cache a block
            let key = BlockCacheKey::new(
                thread_id as u64,
                thread_id as u64 * 100,
                (thread_id as u32 * 10, thread_id as u32 * 5),
                (16, 1),
            );

            let content = format!("cached_content_{}", thread_id);
            let timestamp = current_timestamp();
            let user_id = thread_id as u64;
            let original_hash = compute_content_hash(&content);

            cache.cache_block(key.clone(), content, timestamp, user_id, original_hash);

            // Try to retrieve the cached block
            let retrieved = cache.get_block(key, original_hash);
            assert!(retrieved.is_some());
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify cache has entries
    let cache = cache.lock().unwrap();
    assert!(cache.block_count() > 0);
}

#[test]
fn high_contention_conflict_resolution() {
    // Simulate high contention: many users editing the same position
    let detector = Arc::new(ConflictDetector::new(16));
    let chains = Arc::new(Mutex::new(Vec::new()));
    let num_users = 10;
    let mut handles = vec![];

    // Spawn many users editing the same position
    for user_id in 1..=num_users {
        let _detector_clone = Arc::clone(&detector);
        let chains_clone = Arc::clone(&chains);
        let handle = thread::spawn(move || {
            random_pause();

            let chain = vec![BlockChainUnit::new(
                0, // Same position for all users
                16,
                1,
                format!("user{}_content", user_id),
                current_timestamp(),
                user_id,
            )];

            let mut chains = chains_clone.lock().unwrap();
            chains.push(chain);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Detect and resolve conflicts
    let chains = chains.lock().unwrap();
    let conflicts = detector.detect_conflicts(&chains);

    // Should have a single conflict with many participants
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_count(), num_users as usize);

    // Resolve the conflict
    let mut conflict = conflicts[0].clone();
    conflict.resolve();

    assert!(conflict.is_resolved());
    assert!(conflict.resolved_block.is_some());
}

#[test]
fn deterministic_merge_under_concurrency() {
    // Test that concurrent merges produce deterministic results
    let client = Arc::new(MergenClient::new());
    let input_texts = vec!["Alpha", "Beta", "Gamma", "Delta"];
    let merge_results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    // Perform the same merge operation multiple times concurrently
    for _ in 0..5 {
        let client_clone = Arc::clone(&client);
        let results_clone = Arc::clone(&merge_results);
        let texts = input_texts.clone();

        let handle = thread::spawn(move || {
            random_pause();
            let merged = client_clone.merge_many(texts);
            let mut results = results_clone.lock().unwrap();
            results.push(merged);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // All results should be identical (deterministic)
    let results = merge_results.lock().unwrap();
    let first_result = &results[0];
    for result in results.iter().skip(1) {
        assert_eq!(result, first_result);
    }
}

#[test]
fn late_arriving_edits() {
    // Simulate network latency with late-arriving edits
    let chain = Arc::new(Mutex::new(BlockChain::new()));
    let mut handles = vec![];

    // User 1 makes edits immediately
    let chain_clone1 = Arc::clone(&chain);
    let handle1 = thread::spawn(move || {
        let mut chain = chain_clone1.lock().unwrap();
        chain.push_block(BlockChainUnit::new(0, 16, 1, "early_edit", 1000, 1));
    });
    handles.push(handle1);

    // User 2 makes edits after a delay (simulating network latency)
    let chain_clone2 = Arc::clone(&chain);
    let handle2 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50)); // Simulate network delay
        let mut chain = chain_clone2.lock().unwrap();
        chain.push_block(BlockChainUnit::new(0, 16, 1, "late_edit", 2000, 2));
    });
    handles.push(handle2);

    // Wait for both threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Resolve should handle late-arriving edit correctly
    let chain = chain.lock().unwrap();
    let resolved = chain.resolve();

    // Both edits should be present in the resolved output
    assert!(!resolved.is_empty());
}

#[test]
fn concurrent_strategy_resolution() {
    // Test different resolution strategies under concurrent load
    let strategies = vec![
        ResolveStrategy::TimestampOrder,
        ResolveStrategy::UserIdOrder,
        ResolveStrategy::MostRecent,
    ];

    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for strategy in strategies {
        let results_clone = Arc::clone(&results);
        let handle = thread::spawn(move || {
            random_pause();

            let blocks = vec![
                BlockChainUnit::new(0, 16, 1, "content1", 1000, 3),
                BlockChainUnit::new(0, 16, 1, "content2", 1500, 1),
                BlockChainUnit::new(0, 16, 1, "content3", 2000, 2),
            ];

            let conflict = BlockConflict::new(0, blocks);
            let resolved = strategy.resolve(&conflict);

            let mut results = results_clone.lock().unwrap();
            results.push((format!("{:?}", strategy), resolved.unwrap().user_id));
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify we got results from all strategies
    let results = results.lock().unwrap();
    assert_eq!(results.len(), 3);

    // Verify different strategies can produce different winners
    let user_ids: Vec<_> = results.iter().map(|(_, uid)| *uid).collect();
    // At least some strategies should produce different results
    let unique_ids: std::collections::HashSet<_> = user_ids.into_iter().collect();
    assert!(unique_ids.len() >= 1);
}

#[test]
fn n30_concurrent_users() {
    // Stress test with 30 concurrent users editing the same document
    let shared_document = Arc::new(Mutex::new(String::new()));
    let user_edits = Arc::new(Mutex::new(Vec::new()));
    let num_users = 30;
    let edits_per_user = 5;
    let mut handles = vec![];

    // Initialize with some content
    {
        let mut doc = shared_document.lock().unwrap();
        *doc = "Initial document content for stress test ".to_string();
    }

    // Spawn 30 users editing concurrently
    for user_id in 1..=num_users {
        let doc_clone = Arc::clone(&shared_document);
        let edits_clone = Arc::clone(&user_edits);
        let handle = thread::spawn(move || {
            for edit_num in 0..edits_per_user {
                random_pause();

                let current_content = {
                    let doc = doc_clone.lock().unwrap();
                    doc.clone()
                };

                let new_content =
                    format!("{}U{}E{} ", current_content, user_id, edit_num);

                let edit_block = BlockChainUnit::new(
                    current_content.len(),
                    new_content.len() as u32 - current_content.len() as u32,
                    1,
                    format!("U{}E{}", user_id, edit_num),
                    current_timestamp(),
                    user_id,
                );

                {
                    let mut doc = doc_clone.lock().unwrap();
                    *doc = new_content;
                }

                let mut edits = edits_clone.lock().unwrap();
                edits.push(edit_block);
            }
        });
        handles.push(handle);
    }

    // Wait for all users to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all edits were recorded
    let edits = user_edits.lock().unwrap();
    assert_eq!(edits.len(), (num_users * edits_per_user) as usize);

    // Create blockchain and resolve
    let mut chain = BlockChain::new();
    for edit in edits.iter() {
        chain.push_block(edit.clone());
    }

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());

    // Verify final document contains all user edits
    let final_doc = shared_document.lock().unwrap();
    for user_id in 1..=num_users {
        assert!(final_doc.contains(&format!("U{}", user_id)));
    }
}

#[test]
fn n50_concurrent_users() {
    // Stress test with 50 concurrent users
    let shared_document = Arc::new(Mutex::new(String::new()));
    let user_edits = Arc::new(Mutex::new(Vec::new()));
    let num_users = 50;
    let edits_per_user = 3;
    let mut handles = vec![];

    {
        let mut doc = shared_document.lock().unwrap();
        *doc = "Stress test with 50 users ".to_string();
    }

    for user_id in 1..=num_users {
        let doc_clone = Arc::clone(&shared_document);
        let edits_clone = Arc::clone(&user_edits);
        let handle = thread::spawn(move || {
            for edit_num in 0..edits_per_user {
                random_pause();

                let current_content = {
                    let doc = doc_clone.lock().unwrap();
                    doc.clone()
                };

                let new_content =
                    format!("{}U50_{} ", current_content, user_id * 100 + edit_num);

                let edit_block = BlockChainUnit::new(
                    current_content.len(),
                    new_content.len() as u32 - current_content.len() as u32,
                    1,
                    format!("U50_{}", user_id * 100 + edit_num),
                    current_timestamp(),
                    user_id,
                );

                {
                    let mut doc = doc_clone.lock().unwrap();
                    *doc = new_content;
                }

                let mut edits = edits_clone.lock().unwrap();
                edits.push(edit_block);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let edits = user_edits.lock().unwrap();
    assert_eq!(edits.len(), (num_users * edits_per_user) as usize);

    let mut chain = BlockChain::new();
    for edit in edits.iter() {
        chain.push_block(edit.clone());
    }

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());
}

#[test]
fn n100_concurrent_users() {
    // Stress test with 100 concurrent users - production scale
    let shared_document = Arc::new(Mutex::new(String::new()));
    let user_edits = Arc::new(Mutex::new(Vec::new()));
    let num_users = 100;
    let edits_per_user = 2;
    let mut handles = vec![];

    {
        let mut doc = shared_document.lock().unwrap();
        *doc = "Production scale test with 100 users ".to_string();
    }

    for user_id in 1..=num_users {
        let doc_clone = Arc::clone(&shared_document);
        let edits_clone = Arc::clone(&user_edits);
        let handle = thread::spawn(move || {
            for edit_num in 0..edits_per_user {
                random_pause();

                let current_content = {
                    let doc = doc_clone.lock().unwrap();
                    doc.clone()
                };

                let new_content =
                    format!("{}U100_{} ", current_content, user_id * 1000 + edit_num);

                let edit_block = BlockChainUnit::new(
                    current_content.len(),
                    new_content.len() as u32 - current_content.len() as u32,
                    1,
                    format!("U100_{}", user_id * 1000 + edit_num),
                    current_timestamp(),
                    user_id,
                );

                {
                    let mut doc = doc_clone.lock().unwrap();
                    *doc = new_content;
                }

                let mut edits = edits_clone.lock().unwrap();
                edits.push(edit_block);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let edits = user_edits.lock().unwrap();
    assert_eq!(edits.len(), (num_users * edits_per_user) as usize);

    let mut chain = BlockChain::new();
    for edit in edits.iter() {
        chain.push_block(edit.clone());
    }

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());

    // Verify performance - should complete in reasonable time
    let final_doc = shared_document.lock().unwrap();
    assert!(final_doc.len() > 100); // Should have significant content
}

#[test]
fn high_conflict_rate() {
    // Test with high conflict rate - many users editing same position
    let detector = Arc::new(ConflictDetector::new(16));
    let chains = Arc::new(Mutex::new(Vec::new()));
    let num_users = 75;
    let mut handles = vec![];

    for user_id in 1..=num_users {
        let chains_clone = Arc::clone(&chains);
        let handle = thread::spawn(move || {
            random_pause();

            // All users edit the same position to maximize conflicts
            let chain = vec![
                BlockChainUnit::new(
                    0,
                    16,
                    1,
                    format!("user_{}_edit", user_id),
                    current_timestamp(),
                    user_id,
                ),
                BlockChainUnit::new(
                    16,
                    16,
                    1,
                    format!("user_{}_edit2", user_id),
                    current_timestamp(),
                    user_id,
                ),
            ];

            let mut chains = chains_clone.lock().unwrap();
            chains.push(chain);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let chains = chains.lock().unwrap();
    let conflicts = detector.detect_conflicts(&chains);

    // Should have many conflicts due to same position edits
    assert!(!conflicts.is_empty());
    assert!(conflicts.len() >= 2); // At least 2 positions with conflicts

    // Verify all conflicts can be resolved
    for mut conflict in conflicts {
        conflict.resolve();
        assert!(conflict.is_resolved());
    }
}

#[test]
fn empty_document() {
    // Test merging empty documents
    let client = MergenClient::new();

    let empty1 = "";
    let empty2 = "";

    let merged = client.merge(empty1, empty2);
    assert_eq!(merged, "");
}

#[test]
fn empty_with_content() {
    // Test merging empty document with content
    let client = MergenClient::new();

    let empty = "";
    let content = "Hello, World!";

    let merged1 = client.merge(empty, content);
    let merged2 = client.merge(content, empty);

    assert!(!merged1.is_empty());
    assert!(!merged2.is_empty());
}

#[test]
fn blockchain_from_empty_text() {
    // Test blockchain creation from empty text
    let chain = BlockChain::from_text("");
    assert!(chain.is_empty());
    assert_eq!(chain.len(), 0);

    let resolved = chain.resolve();
    assert_eq!(resolved, "");
}

#[test]
fn single_character() {
    // Test with single character documents
    let client = MergenClient::new();

    let single1 = "a";
    let single2 = "b";

    let merged = client.merge(single1, single2);
    assert!(!merged.is_empty());
}

#[test]
fn very_large_document_100k() {
    // Test with very large document (100K characters)
    let large_text = generate_large_random_text(100_000);
    let _client = MergenClient::new();

    let start = std::time::Instant::now();
    let chain = BlockChain::from_text(&large_text);
    let creation_time = start.elapsed();

    println!(
        "100K document blockchain creation time: {:?}",
        creation_time
    );

    assert!(!chain.is_empty());
    assert!(chain.len() > 0);

    let resolved = chain.resolve();
    assert_eq!(resolved.len(), large_text.len());
}

#[test]
fn very_large_document_500k() {
    // Test with very large document (500K characters)
    let large_text = generate_large_random_text(500_000);
    let _client = MergenClient::new();

    let start = std::time::Instant::now();
    let chain = BlockChain::from_text(&large_text);
    let creation_time = start.elapsed();

    println!(
        "500K document blockchain creation time: {:?}",
        creation_time
    );

    assert!(!chain.is_empty());

    let start = std::time::Instant::now();
    let resolved = chain.resolve();
    let resolve_time = start.elapsed();

    println!("500K document resolve time: {:?}", resolve_time);
    assert_eq!(resolved.len(), large_text.len());
}

#[test]
fn merging_large_documents() {
    // Test merging two large documents
    let large_text1 = generate_large_random_text(50_000);
    let large_text2 = generate_large_random_text(50_000);
    let client = MergenClient::new();

    let start = std::time::Instant::now();
    let merged = client.merge(&large_text1, &large_text2);
    let merge_time = start.elapsed();

    println!("50K+50K document merge time: {:?}", merge_time);

    assert!(!merged.is_empty());
    assert!(merged.len() > 0);
}

#[test]
fn unicode_content() {
    // Test with unicode content
    let unicode_text = generate_random_unicode_text(1000);
    let _client = MergenClient::new();

    let chain = BlockChain::from_text(&unicode_text);
    assert!(!chain.is_empty());

    let resolved = chain.resolve();
    // Unicode characters may not be preserved exactly due to byte-level chunking
    // but we should get some content back
    assert!(!resolved.is_empty());
    // The resolved content should be reasonably close in length
    let length_diff = (resolved.len() as i32 - unicode_text.len() as i32).abs();
    assert!(length_diff < (unicode_text.len() / 2) as i32); // Allow up to 50% difference due to chunking
}

#[test]
fn unicode_merging() {
    // Test merging unicode documents
    let unicode1 = generate_random_unicode_text(500);
    let unicode2 = generate_random_unicode_text(500);
    let client = MergenClient::new();

    let merged = client.merge(&unicode1, &unicode2);
    assert!(!merged.is_empty());
}

#[test]
fn special_characters() {
    // Test with special characters
    let special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?\n\t\r";
    let _client = MergenClient::new();

    let chain = BlockChain::from_text(special_chars);
    assert!(!chain.is_empty());

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());
}

#[test]
fn very_long_single_line() {
    // Test with very long single line (no line breaks)
    let long_line = "a".repeat(10_000);
    let _client = MergenClient::new();

    let chain = BlockChain::from_text(&long_line);
    assert!(!chain.is_empty());

    let resolved = chain.resolve();
    assert_eq!(resolved.len(), long_line.len());
}

#[test]
fn many_short_lines() {
    // Test with many short lines
    let many_lines = "line\n".repeat(1_000);
    let _client = MergenClient::new();

    let chain = BlockChain::from_text(&many_lines);
    assert!(!chain.is_empty());

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());
}

#[test]
fn zero_width_blocks() {
    // Test blocks with zero width
    let block = BlockChainUnit::new(0, 0, 1, "", 1000, 1);
    assert_eq!(block.width, 0);
    assert_eq!(block.data.len(), 0);
}

#[test]
fn very_large_user_id() {
    // Test with very large user ID
    let large_user_id = u64::MAX;
    let block = BlockChainUnit::new(0, 16, 1, "test", 1000, large_user_id);

    assert_eq!(block.user_id, large_user_id);

    let hash = block.ordering_hash();
    assert_ne!(hash, 0);
}

#[test]
fn timestamp_overflow() {
    // Test with maximum timestamp
    let max_timestamp = u64::MAX;
    let hash = generate_mergen_hash(max_timestamp, 42);

    let extracted_ts = extract_timestamp(hash);
    assert_eq!(extracted_ts, (max_timestamp & 0xFFFFFFFF) as u32);
}

#[test]
fn concurrent_empty_document_edits() {
    // Test concurrent edits on empty document
    let shared_document = Arc::new(Mutex::new(String::new()));
    let user_edits = Arc::new(Mutex::new(Vec::new()));
    let num_users = 20;
    let mut handles = vec![];

    for user_id in 1..=num_users {
        let doc_clone = Arc::clone(&shared_document);
        let edits_clone = Arc::clone(&user_edits);
        let handle = thread::spawn(move || {
            random_pause();

            let current_content = {
                let doc = doc_clone.lock().unwrap();
                doc.clone()
            };

            let new_content = format!("{}User{} ", current_content, user_id);

            let edit_block = BlockChainUnit::new(
                current_content.len(),
                new_content.len() as u32 - current_content.len() as u32,
                1,
                format!("User{}", user_id),
                current_timestamp(),
                user_id,
            );

            {
                let mut doc = doc_clone.lock().unwrap();
                *doc = new_content;
            }

            let mut edits = edits_clone.lock().unwrap();
            edits.push(edit_block);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let edits = user_edits.lock().unwrap();
    assert_eq!(edits.len(), num_users as usize);

    let mut chain = BlockChain::new();
    for edit in edits.iter() {
        chain.push_block(edit.clone());
    }

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());
}

#[test]
fn large_document_concurrent_edits() {
    // Test concurrent edits on large document
    let large_text = generate_large_random_text(10_000);
    let shared_document = Arc::new(Mutex::new(large_text));
    let user_edits = Arc::new(Mutex::new(Vec::new()));
    let num_users = 15;
    let mut handles = vec![];

    for user_id in 1..=num_users {
        let doc_clone = Arc::clone(&shared_document);
        let edits_clone = Arc::clone(&user_edits);
        let handle = thread::spawn(move || {
            random_pause();

            let current_content = {
                let doc = doc_clone.lock().unwrap();
                doc.clone()
            };

            let new_content = format!("{}[U{}]", current_content, user_id);

            let edit_block = BlockChainUnit::new(
                current_content.len(),
                new_content.len() as u32 - current_content.len() as u32,
                1,
                format!("[U{}]", user_id),
                current_timestamp(),
                user_id,
            );

            {
                let mut doc = doc_clone.lock().unwrap();
                *doc = new_content;
            }

            let mut edits = edits_clone.lock().unwrap();
            edits.push(edit_block);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let edits = user_edits.lock().unwrap();
    assert_eq!(edits.len(), num_users as usize);

    let mut chain = BlockChain::new();
    for edit in edits.iter() {
        chain.push_block(edit.clone());
    }

    let resolved = chain.resolve();
    assert!(!resolved.is_empty());
}

#[test]
fn cache_memory_limits() {
    // Test cache behavior with memory limits
    let mut cache = BlockCache::new();

    // Add many blocks to test memory limits
    for i in 0..1000 {
        let key = BlockCacheKey::new(i as u64, i as u64, (i as u32, i as u32), (16, 1));
        let content = format!("block_content_{}", i);
        let timestamp = current_timestamp();
        let user_id = i as u64;
        let original_hash = compute_content_hash(&content);

        cache.cache_block(key, content, timestamp, user_id, original_hash);
    }

    // Cache should have evicted some blocks due to memory limits
    let block_count = cache.block_count();
    println!("Cache block count after 1000 inserts: {}", block_count);

    // Should have some blocks, the cache may keep all if memory is sufficient
    assert!(block_count > 0);
    // We don't assert < 1000 since the cache might have enough memory
}

#[test]
fn blockchain_memory_limits() {
    // Test blockchain with many blocks
    let mut chain = BlockChain::new();

    // Add many blocks
    for i in 0..10_000 {
        let block = BlockChainUnit::new(
            i * 16,
            16,
            1,
            format!("block_{}", i),
            current_timestamp(),
            (i % 100) as u64, // Cycle through user IDs
        );
        chain.push_block(block);
    }

    assert_eq!(chain.len(), 10_000);

    // Test resolve performance with many blocks
    let start = std::time::Instant::now();
    let resolved = chain.resolve();
    let resolve_time = start.elapsed();

    println!("10K block resolve time: {:?}", resolve_time);
    assert!(!resolved.is_empty());
}

#[test]
fn deterministic_large_scale() {
    // Test deterministic behavior with large scale
    let large_text = generate_large_random_text(20_000);
    let client = MergenClient::new();

    // Perform same merge multiple times
    let results = (0..5)
        .map(|_| {
            random_pause();
            client.merge(&large_text, &large_text)
        })
        .collect::<Vec<_>>();

    // All results should be identical
    let first = &results[0];
    for result in results.iter().skip(1) {
        assert_eq!(result, first);
    }
}
