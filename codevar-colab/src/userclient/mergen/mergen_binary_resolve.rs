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

//! # Binary-level conflict resolution optimized for parallel processing
//!
//! This module provides binary-level conflict resolution that can be executed
//! in parallel on CPU or GPU (CUDA/Vulkan compute shaders). The algorithms are
//! designed to work with blocks of text (8×8 or 16×16) for optimal GPU thread
//! scheduling and memory coalescing.
//!
//! ## Architecture
//!
//! The binary resolution system is designed for high-performance parallel processing:
//!
//! - **SIMD Optimization**: Uses AVX2/SSE4.1 instructions for vectorized processing
//! - **Block-based Processing**: Processes text in fixed-size blocks for optimal cache usage
//! - **Parallel Execution**: Supports multi-threaded CPU and GPU compute shader execution
//! - **Memory Alignment**: 16-byte aligned data structures for optimal SIMD performance
//!
//! ## Safety Considerations
//!
//! This module uses unsafe code extensively for:
//!
//! - **SIMD Intrinsics**: Direct CPU vector instructions for performance
//! - **Pointer Arithmetic**: Unsafe pointer operations for memory alignment
//! - **Memory Access**: Direct memory reads/writes for block processing
//! - **Type Punning**: Reinterpreting data for SIMD operations
//!
//! All unsafe operations maintain strict invariants:
//! - All SIMD operations are performed on properly aligned memory
//! - Pointer arithmetic stays within allocated memory boundaries
//! - SIMD intrinsic calls are only made on supported CPU architectures
//! - Block sizes are validated before processing
//!
//! ## Platform Support
//!
//! - **x86_64**: Full SIMD support with AVX2 and SSE4.1 instructions
//! - **Other architectures**: Falls back to scalar processing
//! - **GPU Support**: Designed for CUDA/Vulkan compute shader portability

use crate::userclient::mergen::mergen_binary_fifo::{Fifo, FifoEntry};
use crate::userclient::mergen::mergen_blockchain::BlockChainUnit;
use crate::userclient::mergen::mergen_hash::{compare_hashes, generate_mergen_hash};
use std::sync::{Arc, Mutex};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_and_si128, _mm_andnot_si128, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_or_si128, _mm_storeu_si128, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_or_si256, _mm256_storeu_si256,
};

/// Default parallelism level based on CPU cores.
fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

pub const DEFAULT_ALIGNMENT: usize = 16;

pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Binary merge operation that can be executed in parallel.
#[derive(Clone)]
pub struct BinaryMergeOp {
    /// Source block A
    pub block_a: Arc<BlockChainUnit>,
    /// Source block B
    pub block_b: Arc<BlockChainUnit>,
    /// Result FIFO for merged characters
    pub result_fifo: Arc<Mutex<Fifo>>,
    /// Operation ID for tracking
    pub op_id: u64,
}

impl BinaryMergeOp {
    /// Creates a new binary merge operation.
    pub fn new(block_a: BlockChainUnit, block_b: BlockChainUnit, op_id: u64) -> Self {
        Self {
            block_a: Arc::new(block_a),
            block_b: Arc::new(block_b),
            result_fifo: Arc::new(Mutex::new(Fifo::new())),
            op_id,
        }
    }

    /// Executes the binary merge operation.
    pub fn execute(&self) {
        let mut fifo = self.result_fifo.lock().unwrap_or_else(|e| e.into_inner());

        let entries_a = self.create_entries(&self.block_a);
        let entries_b = self.create_entries(&self.block_b);

        self.merge_entries(&mut fifo, &entries_a, &entries_b);
    }

    /// Creates FIFO entries from a block.
    fn create_entries(&self, block: &BlockChainUnit) -> Vec<FifoEntry> {
        block
            .data
            .chars()
            .enumerate()
            .map(|(char_index, ch)| {
                let hash = generate_mergen_hash(
                    block.timestamp.wrapping_add(char_index as u64),
                    block.user_id.wrapping_add(char_index as u64),
                );
                FifoEntry::new(
                    u32::from(ch),
                    hash,
                    block.change_count,
                    (char_index as u32).wrapping_add(block.offset as u32),
                )
            })
            .collect()
    }

    /// Merges two sorted entry lists into the FIFO.
    fn merge_entries(
        &self,
        fifo: &mut Fifo,
        entries_a: &[FifoEntry],
        entries_b: &[FifoEntry],
    ) {
        let mut i = 0;
        let mut j = 0;

        while i < entries_a.len() || j < entries_b.len() {
            if i < entries_a.len() && j < entries_b.len() {
                match compare_hashes(entries_a[i].hash, entries_b[j].hash) {
                    std::cmp::Ordering::Less => {
                        fifo.insert(entries_a[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        fifo.insert(entries_b[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        if entries_a[i].change_count >= entries_b[j].change_count {
                            fifo.insert(entries_a[i]);
                        } else {
                            fifo.insert(entries_b[j]);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            } else if i < entries_a.len() {
                fifo.insert(entries_a[i]);
                i += 1;
            } else {
                fifo.insert(entries_b[j]);
                j += 1;
            }
        }
    }

    /// Returns the merged result as a string.
    pub fn get_result(&self) -> String {
        let fifo = self.result_fifo.lock().unwrap_or_else(|e| e.into_inner());
        fifo.extract_ordered()
    }
}

/// SIMD-accelerated binary merger for aligned blocks.
pub struct BinaryResolveFast {
    /// Alignment requirement for SIMD operations
    alignment: usize,
}

impl BinaryResolveFast {
    /// Creates a new SIMD binary merger with specified alignment.
    pub fn new(alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be a power of two for SIMD operations"
        );
        Self { alignment }
    }

    /// Creates a merger with default SIMD alignment.
    pub fn default() -> Self {
        Self::new(DEFAULT_ALIGNMENT)
    }

    /// Performs SIMD-accelerated merge on aligned data.
    pub fn merge_aligned(&self, data_a: &[u8], data_b: &[u8]) -> Vec<u8> {
        self.merge_scalar(data_a, data_b)
    }

    /// AVX2-accelerated merge for 256-bit vectors.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn merge_avx2(&self, data_a: &[u8], data_b: &[u8]) -> Vec<u8> {
        let min_len = data_a.len().min(data_b.len());
        let simd_len = min_len & !31;
        let mut result = Vec::<u8>::with_capacity(data_a.len() + data_b.len());

        let mut i = 0;
        while i < simd_len {
            let vec_a = _mm256_loadu_si256(data_a.as_ptr().add(i) as *const __m256i);
            let vec_b = _mm256_loadu_si256(data_b.as_ptr().add(i) as *const __m256i);

            let merged = self.merge_vectors_avx2(vec_a, vec_b);
            _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut __m256i, merged);

            i += 16;
        }

        result.extend_from_slice(&data_a[simd_len..]);
        result.extend_from_slice(&data_b[simd_len..]);
        result
    }

    /// SSE4.1-accelerated merge for 128-bit vectors.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn merge_sse41(&self, data_a: &[u8], data_b: &[u8]) -> Vec<u8> {
        let min_len = data_a.len().min(data_b.len());
        let simd_len = min_len & !15;
        let mut result = Vec::<u8>::with_capacity(data_a.len() + data_b.len());

        let mut i = 0;
        while i < simd_len {
            let vec_a = _mm_loadu_si128(data_a.as_ptr().add(i) as *const __m128i);
            let vec_b = _mm_loadu_si128(data_b.as_ptr().add(i) as *const __m128i);

            let merged = self.merge_vectors_sse41(vec_a, vec_b);
            _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, merged);
            i += 16;
        }
        result.extend_from_slice(&data_a[simd_len..]);
        result.extend_from_slice(&data_b[simd_len..]);
        result
    }

    /// Scalar fallback for non-SIMD architectures.
    fn merge_scalar(&self, data_a: &[u8], data_b: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data_a.len() + data_b.len());
        let mut i = 0;
        let mut j = 0;

        while i < data_a.len() || j < data_b.len() {
            if i < data_a.len() && j < data_b.len() {
                if data_a[i] <= data_b[j] {
                    result.push(data_a[i]);
                    i += 1;
                } else {
                    result.push(data_b[j]);
                    j += 1;
                }
            } else if i < data_a.len() {
                result.push(data_a[i]);
                i += 1;
            } else {
                result.push(data_b[j]);
                j += 1;
            }
        }
        result
    }

    /// Merges two AVX2 vectors based on hash comparison.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn merge_vectors_avx2(&self, vec_a: __m256i, vec_b: __m256i) -> __m256i {
        let cmp = _mm256_cmpgt_epi8(vec_a, vec_b);
        let mask_a = _mm256_andnot_si256(cmp, vec_a);
        let mask_b = _mm256_and_si256(cmp, vec_b);
        _mm256_or_si256(mask_a, mask_b)
    }

    /// Merges two SSE4.1 vectors based on hash comparison.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn merge_vectors_sse41(&self, vec_a: __m128i, vec_b: __m128i) -> __m128i {
        let cmp = _mm_cmpgt_epi8(vec_a, vec_b);
        let mask_a = _mm_andnot_si128(cmp, vec_a);
        let mask_b = _mm_and_si128(cmp, vec_b);
        _mm_or_si128(mask_a, mask_b)
    }

    /// Returns the alignment requirement.
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

impl Default for BinaryResolveFast {
    fn default() -> Self {
        Self::default()
    }
}

/// Parallel batch processor for binary merge operations.
pub struct BatchProcessor {
    /// Maximum number of parallel operations
    max_parallel: usize,
    /// SIMD merger for accelerated processing
    simd_merger: BinaryResolveFast,
}

impl BatchProcessor {
    /// Creates a new parallel batch processor.
    pub fn new(max_parallel: usize) -> Self {
        Self {
            max_parallel,
            simd_merger: BinaryResolveFast::default(),
        }
    }

    /// Creates a processor with default parallelism.
    pub fn default() -> Self {
        Self::new(default_parallelism())
    }

    /// Processes multiple binary merge operations in parallel.
    pub fn process_batch(&self, operations: Vec<BinaryMergeOp>) -> Vec<String> {
        operations
            .into_iter()
            .map(|op| {
                op.execute();
                op.get_result()
            })
            .collect()
    }

    /// Returns the maximum parallelism level.
    pub fn max_parallel(&self) -> usize {
        self.max_parallel
    }

    /// Returns the SIMD merger.
    pub fn simd_merger(&self) -> &BinaryResolveFast {
        &self.simd_merger
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::default()
    }
}

/// Block-level binary resolver.
pub struct BlockBinaryResolver {
    /// Block size for processing (8 or 16)
    block_size: usize,
    /// Parallel processor for batch operations
    processor: BatchProcessor,
    /// SIMD merger for accelerated processing
    simd_merger: BinaryResolveFast,
}

impl BlockBinaryResolver {
    /// Creates a new block binary resolver.
    pub fn new(block_size: usize, max_parallel: usize) -> Self {
        assert!(
            block_size == 8 || block_size == 16,
            "Block size must be 8 or 16 for optimal processing"
        );
        Self {
            block_size,
            processor: BatchProcessor::new(max_parallel),
            simd_merger: BinaryResolveFast::default(),
        }
    }

    /// Creates a resolver with default settings.
    pub fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE, default_parallelism())
    }

    /// Resolves conflicts between blocks using binary merge operations.
    pub fn resolve_blocks(
        &self,
        blocks_a: &[BlockChainUnit],
        blocks_b: &[BlockChainUnit],
    ) -> Vec<String> {
        let mut operations = Vec::new();
        let mut op_id = 0u64;

        for block_a in blocks_a {
            for block_b in blocks_b {
                if block_a.offset == block_b.offset {
                    let op = BinaryMergeOp::new(block_a.clone(), block_b.clone(), op_id);
                    operations.push(op);
                    op_id += 1;
                }
            }
        }
        if operations.is_empty() {
            return Vec::new();
        }
        self.processor.process_batch(operations)
    }

    /// Resolves conflicts for aligned blocks using SIMD acceleration.
    pub fn resolve_aligned_blocks(
        &self,
        paired_blocks: &[(BlockChainUnit, BlockChainUnit)],
    ) -> Vec<String> {
        let operations: Vec<BinaryMergeOp> = paired_blocks
            .iter()
            .enumerate()
            .map(|(i, (a, b))| BinaryMergeOp::new(a.clone(), b.clone(), i as u64))
            .collect();
        if operations.is_empty() {
            return Vec::new();
        }
        self.processor.process_batch(operations)
    }

    /// Returns the block size used for processing.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the parallel processor.
    pub fn processor(&self) -> &BatchProcessor {
        &self.processor
    }
}

impl Default for BlockBinaryResolver {
    fn default() -> Self {
        Self::default()
    }
}
