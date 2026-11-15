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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Criterion benchmarks for the Wredit element codec.
//!
//! The three costs the codec has are measured separately, because they are
//! optimized separately:
//!
//! * `werc/plane` - stage 1, the only part that vectorizes: the plane split
//!   and merge, where the SIMD tier is selected at run time and the scalar
//!   path is the reference.
//! * `werc/model` - stage 2 on its own, one pass of predict and update over a
//!   corpus, which is the serial bit loop no amount of SIMD can shorten.
//! * `werc/codec` - the framed stream, encode and decode, so the per-byte
//!   overhead of the container is visible next to the coding cost.
//!
//! `mem_bits` is deliberately small: the model's tables dominate its own
//! memory, and a bench that cache-thrashed on a 48 MiB model would measure
//! the memory system rather than the codec. Benchmarks are test-like code,
//! not production code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_wredit_werc::codec::{decode, encode};
use codevar_wredit_werc::element::{COMMON_MASK, pack};
use codevar_wredit_werc::frame::Params;
use codevar_wredit_werc::model::Model;
use codevar_wredit_werc::plane::{body_len, join, presence, split};

/// Element counts the streaming stages are measured at.
const SIZES: [usize; 2] = [4 * 1024, 64 * 1024];

/// The document the coding stages are measured with: real Rust source,
/// because the model's ratio and its speed both depend on what it sees, and
/// a pattern would make the match model look cheaper than it is.
const SOURCE: &str = r#"
fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("usage: werc_bench <elements.w32> [mem_bits]");
        return;
    }
    let raw = std::fs::read(&argv[0]).expect("read the dump");
    let elements: Vec<u32> = raw.chunks_exact(4).map(u32::from_le_bytes).collect();
    for (index, element) in elements.iter().enumerate() {
        println!("{index}: {:08X} type {} char {}", element, element >> 28, element & 0xFF);
    }
}
"#;

/// `count` elements of the source, repeated and truncated to length.
fn elements_of(count: usize) -> Vec<u32> {
    let bytes: Vec<u8> = SOURCE
        .bytes()
        .cycle()
        .take(count)
        .map(|b| if b == 0 { b' ' } else { b })
        .collect();
    bytes
        .iter()
        .map(|&b| pack(u32::from(b), COMMON_MASK))
        .collect()
}

/// Bytes the planes of `count` elements occupy, or the bench's own bug.
fn plane_body(count: usize, flags: u8) -> usize {
    body_len(count, flags).expect("the bench flags are the ones the split writes")
}

/// The parameters the coding loop is measured at: 1.2 MiB of tables.
const PARAMS: Params = Params {
    mem_bits: 16,
    sse_bits: 12,
    lr_shift: 11,
};

/// Stage 1 throughput, in and out.
fn bench_plane(c: &mut Criterion) {
    let mut group = c.benchmark_group("werc/plane");
    for &size in &SIZES {
        let elements = elements_of(size);
        let flags = presence(&elements);
        let body = plane_body(size, flags);
        let mut planes = vec![0u8; body];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("split", size), &elements, |b, elements| {
            b.iter(|| {
                let written = split(black_box(elements), flags, &mut planes).unwrap_or_default();
                black_box(written);
            });
        });
        let written = split(&elements, flags, &mut planes).unwrap_or_default();
        let source = planes[..written].to_vec();
        let mut back = vec![0u32; size];
        group.bench_with_input(BenchmarkId::new("merge", size), &source, |b, source| {
            b.iter(|| {
                let count = join(flags, black_box(source), &mut back).unwrap_or_default();
                black_box(count);
            });
        });
    }
    group.finish();
}

/// Stage 2 alone: the bit loop over a corpus, with the coder out of the way.
fn bench_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("werc/model");
    let elements = elements_of(8 * 1024);
    let flags = presence(&elements);
    let mut planes = vec![0u8; plane_body(elements.len(), flags)];
    let written = split(&elements, flags, &mut planes).unwrap_or_default();
    let planes = planes[..written].to_vec();
    group.throughput(Throughput::Bytes(written as u64));
    group.bench_function("predict_update", |b| {
        b.iter_batched(
            || {
                Model::new(
                    PARAMS.mem_bits.into(),
                    PARAMS.sse_bits.into(),
                    PARAMS.lr_shift.into(),
                )
                .expect("the bench parameters are in range")
            },
            |mut model| {
                for &byte in &planes {
                    let mut c0 = 1u32;
                    for nbits in 0..8u32 {
                        black_box(model.predict(c0, nbits));
                        let bit = (u32::from(byte) >> (7 - nbits)) & 1;
                        model.update(bit);
                        c0 = (c0 << 1) | bit;
                    }
                    model.byte_update(byte);
                }
                black_box(model);
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// The framed stream, both directions.
fn bench_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("werc/codec");
    let elements = elements_of(8 * 1024);
    let dictionary = elements_of(4 * 1024);
    group.throughput(Throughput::Bytes((elements.len() * 4) as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            black_box(encode(&PARAMS, &[], black_box(&elements)))
                .unwrap_or_default()
                .len()
        });
    });
    group.bench_function("encode_primed", |b| {
        b.iter(|| {
            black_box(encode(&PARAMS, black_box(&dictionary), black_box(&elements)))
                .unwrap_or_default()
                .len()
        });
    });
    let blob = encode(&PARAMS, &[], &elements).expect("the bench document codes");
    let count = elements.len();
    group.bench_function("decode", |b| {
        b.iter_batched(
            || vec![0u32; count],
            |mut back| decode(black_box(&blob), &[], &mut back).unwrap_or_default(),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_plane, bench_model, bench_codec);
criterion_main!(benches);
