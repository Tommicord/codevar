//! Copyright 2026 Codevar Project
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

//! Benchmarks for the Packlow codec.
//!
//! Three groups:
//!
//! - `pack/cpu` - the whole codec on a large corpus, packing and
//!   unpacking, measured in bytes per second.
//! - `pack/simd` - the vector kernels against the scalar reference.
//! - `pack/device` - the device kernels against the host ones, on the
//!   same work, with a check that both produce the same answer.

use std::time::Duration;

use codevar_packlow::codec::{Packlow, PacklowConfig};
use codevar_packlow::gpu::{self, BlockSet};
use codevar_packlow::simd;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Short per sample budget: the device dispatches cost milliseconds, so
/// the default five second measurement would make a run needlessly long.
const MEASUREMENT_TIME: Duration = Duration::from_secs(5);

/// One second of warm up: enough for the caches to settle while keeping
/// the device groups affordable.
const WARM_UP_TIME: Duration = Duration::from_secs(1);

/// The corpus sizes the whole codec is measured on.
const CORPUS_SIZES: [usize; 3] = [1 << 20, 8 << 20, 32 << 20];

/// How many blocks the device groups work on.
const DEVICE_BLOCKS: usize = 4096;

/// The size of one block in the device groups.
const DEVICE_BLOCK: usize = 4096;

/// A deterministic pseudo random byte generator.
fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect()
}

/// A workspace of near identical blocks, the shape the codec is built for.
fn corpus(size: usize) -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut out = Vec::with_capacity(size + DEVICE_BLOCK);
    let mut template = Vec::with_capacity(DEVICE_BLOCK);
    while out.len() < size {
        template.clear();
        while template.len() < DEVICE_BLOCK {
            let word = next().to_le_bytes();
            template.extend_from_slice(&word);
        }
        // a few blocks differ a little from the template
        if !out.is_empty() && next() % 4 == 0 {
            let at = (next() as usize) % template.len();
            template[at] = template[at].wrapping_add(1);
        }
        out.extend_from_slice(&template);
    }
    out.truncate(size);
    out
}

/// The whole codec on a large corpus.
fn bench_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack/cpu");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let codec = Packlow::new(PacklowConfig::default());
    for &size in &CORPUS_SIZES {
        let data = corpus(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(BenchmarkId::new("pack", size), |bencher| {
            bencher.iter(|| black_box(codec.pack(black_box(&data))));
        });
        let packed = codec.pack(&data);
        group.bench_function(BenchmarkId::new("unpack", size), |bencher| {
            bencher.iter(|| black_box(codec.unpack(black_box(&packed))));
        });
    }
    group.finish();
}

/// The vector kernels against the scalar reference.
fn bench_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack/simd");
    group.sample_size(50);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let blocks: Vec<Vec<u8>> = (0..64)
        .map(|index| pseudo_random(4096, index as u32))
        .collect();
    let references: Vec<Vec<u8>> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let mut copy = block.clone();
            let at = index % copy.len().max(1);
            if let Some(byte) = copy.get_mut(at) {
                *byte = byte.wrapping_add(1);
            }
            copy
        })
        .collect();
    group.throughput(Throughput::Bytes((blocks.len() * 4096) as u64));

    group.bench_function("plan_delta/scalar", |bencher| {
        bencher.iter(|| {
            for (block, reference) in blocks.iter().zip(references.iter()) {
                black_box(simd::scalar::plan_delta(reference, block));
            }
        });
    });
    group.bench_function("plan_delta/vector", |bencher| {
        bencher.iter(|| {
            for (block, reference) in blocks.iter().zip(references.iter()) {
                black_box(simd::plan_delta(reference, block));
            }
        });
    });
    group.bench_function("common_prefix/scalar", |bencher| {
        bencher.iter(|| {
            for (block, reference) in blocks.iter().zip(references.iter()) {
                black_box(simd::scalar::common_prefix(reference, block));
            }
        });
    });
    group.bench_function("common_prefix/vector", |bencher| {
        bencher.iter(|| {
            for (block, reference) in blocks.iter().zip(references.iter()) {
                black_box(simd::common_prefix(reference, block));
            }
        });
    });
    group.finish();
}

/// The device kernels against the host ones.
fn bench_device(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack/device");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    // a set of blocks and the near duplicates the device classifies
    let blocks: Vec<Vec<u8>> = (0..DEVICE_BLOCKS)
        .map(|index| pseudo_random(DEVICE_BLOCK, index as u32))
        .collect();
    let mut near = blocks.clone();
    for (index, block) in near.iter_mut().enumerate() {
        let at = index % block.len().max(1);
        if let Some(byte) = block.get_mut(at) {
            *byte = byte.wrapping_add(1);
        }
    }
    let block_refs: Vec<&[u8]> = blocks
        .iter()
        .map(|block| block.as_slice())
        .collect();
    let near_refs: Vec<&[u8]> = near
        .iter()
        .map(|block| block.as_slice())
        .collect();
    let target_set = BlockSet::new(&near_refs);
    let source_set = BlockSet::new(&block_refs);
    let pairs: Vec<(i32, i32)> = (0..DEVICE_BLOCKS)
        .map(|index| (index as i32, index as i32))
        .collect();

    group.throughput(Throughput::Bytes((DEVICE_BLOCKS * DEVICE_BLOCK) as u64));

    let cpu_signatures: Vec<u64> = blocks
        .iter()
        .map(|block| simd::block_signature(block))
        .collect();
    if let Some(gpu_signatures) = gpu::try_block_signatures(&target_set) {
        assert_eq!(
            cpu_signatures, gpu_signatures,
            "the device and the host must agree on the signatures"
        );
        group.bench_function("signatures/cpu", |bencher| {
            bencher.iter(|| {
                for block in &blocks {
                    black_box(simd::block_signature(black_box(block)));
                }
            });
        });
        group.bench_function("signatures/gpu", |bencher| {
            bencher.iter(|| black_box(gpu::try_block_signatures(black_box(&target_set))));
        });
    } else {
        group.bench_function("signatures/cpu", |bencher| {
            bencher.iter(|| {
                for block in &blocks {
                    black_box(simd::block_signature(black_box(block)));
                }
            });
        });
    }

    let cpu_prefixes: Vec<i32> = pairs
        .iter()
        .map(|(target, source)| {
            i32::try_from(simd::common_prefix(
                near_refs[*target as usize],
                block_refs[*source as usize],
            ))
            .unwrap_or(i32::MAX)
        })
        .collect();
    if let Some(gpu_prefixes) = gpu::try_common_prefix(&target_set, &source_set, &pairs) {
        assert_eq!(
            cpu_prefixes, gpu_prefixes,
            "the device and the host must agree on the prefix lengths"
        );
        group.bench_function("prefix/cpu", |bencher| {
            bencher.iter(|| {
                for (target, source) in &pairs {
                    black_box(simd::common_prefix(
                        near_refs[*target as usize],
                        block_refs[*source as usize],
                    ));
                }
            });
        });
        group.bench_function("prefix/gpu", |bencher| {
            bencher.iter(|| {
                black_box(gpu::try_common_prefix(
                    black_box(&target_set),
                    black_box(&source_set),
                    black_box(&pairs),
                ));
            });
        });
    } else {
        group.bench_function("prefix/cpu", |bencher| {
            bencher.iter(|| {
                for (target, source) in &pairs {
                    black_box(simd::common_prefix(
                        near_refs[*target as usize],
                        block_refs[*source as usize],
                    ));
                }
            });
        });
    }

    let cpu_plans: Vec<simd::DeltaPlan> = pairs
        .iter()
        .map(|(target, source)| simd::plan_delta(block_refs[*source as usize], near_refs[*target as usize]))
        .collect();
    if let Some(classification) = gpu::try_classify_deltas(&target_set, &source_set, &pairs, DEVICE_BLOCK) {
        for (index, plan) in cpu_plans.iter().enumerate() {
            let good = plan.representable;
            assert_eq!(
                classification.representable[index],
                i32::try_from(good).unwrap_or(i32::MAX),
                "the device and the host must agree on the reachable bytes"
            );
            for byte in 0..good {
                assert_eq!(
                    classification.exponent[index * DEVICE_BLOCK + byte],
                    plan.exponent[byte]
                );
                assert_eq!(
                    classification.number[index * DEVICE_BLOCK + byte],
                    plan.number[byte]
                );
            }
        }
        group.bench_function("classify/cpu", |bencher| {
            bencher.iter(|| {
                for (target, source) in &pairs {
                    black_box(simd::plan_delta(
                        block_refs[*source as usize],
                        near_refs[*target as usize],
                    ));
                }
            });
        });
        group.bench_function("classify/gpu", |bencher| {
            bencher.iter(|| {
                black_box(gpu::try_classify_deltas(
                    black_box(&target_set),
                    black_box(&source_set),
                    black_box(&pairs),
                    DEVICE_BLOCK,
                ));
            });
        });
    } else {
        group.bench_function("classify/cpu", |bencher| {
            bencher.iter(|| {
                for (target, source) in &pairs {
                    black_box(simd::plan_delta(
                        block_refs[*source as usize],
                        near_refs[*target as usize],
                    ));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_codec, bench_simd, bench_device,);
criterion_main!(benches);
