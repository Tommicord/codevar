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

//! Criterion benchmarks for the storage ↔ edit bindings.
//!
//! Measures the latency of loading a stored file into the edit buffer,
//! saving the edit buffer back to storage, and full round-trip
//! (load → edit → save) cycles, across a range of document sizes.
//!
//! The storage backend is [`MemoryDisk`](codevar_wredit::storage::block::MemoryDisk),
//! so the numbers reflect the binding overhead (byte ↔ packed-element
//! conversion, gap staging, block writes) without real-device latency.

//! Benchmarks are test-like code, not production code; setup failures are
//! fatal configuration errors, not runtime conditions.
#![allow(clippy::unwrap_used)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_wredit::edit::{COMMON_MASK, make_packed};
use codevar_wredit::storage::block::MemoryDisk;
use codevar_wredit::storage::fs::DiskFs;
use codevar_wredit::storage_bind::BoundDocument;

/// Private gap window used by every benchmark document, in elements.
const GAP_SIZE: usize = 1024;

/// The concrete bound-document instantiation under test.
type Bound<'a> = BoundDocument<'a, MemoryDisk, 256, GAP_SIZE>;

/// Fills `dst` with `len` bytes drawn from a fixed seed.
fn payload(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push(b'a' + ((state as usize) % 26) as u8);
    }
    out
}

/// Packs `bytes` into `make_packed(byte, COMMON_MASK)` elements.
fn pack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .iter()
        .map(|&b| make_packed(b as u32, COMMON_MASK))
        .collect()
}

/// `bind/load` — reading a stored file into the edit buffer.
fn bench_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/load");
    for &size in &[4 * 1024usize, 64 * 1024, 1 << 20] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let disk = MemoryDisk::new(256).unwrap();
                let mut fs = DiskFs::format(disk, 256, 128).unwrap();
                let data = payload(size, 0x9E37_79B9);
                let inode = fs.create(2, "load.bin", 0x8000).unwrap();
                fs.write_at(inode, 0, &data).unwrap();
                let bound: Bound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
                black_box(bound.raw_length())
            });
        });
    }
    group.finish();
}

/// `bind/save` — writing the edit buffer back to storage.
fn bench_save(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/save");
    for &size in &[4 * 1024usize, 64 * 1024, 1 << 20] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let disk = MemoryDisk::new(256).unwrap();
                let mut fs = DiskFs::format(disk, 256, 128).unwrap();
                let data = payload(size, 0x00C0_FFEE);
                let inode = fs.create(2, "save.bin", 0x8000).unwrap();
                fs.write_at(inode, 0, &data).unwrap();
                let mut bound: Bound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
                let packed = pack(&data);
                bound.replace_range(0, 0, &packed).unwrap();
                bound.flush_all();
                black_box(bound.save())
            });
        });
    }
    group.finish();
}

/// `bind/roundtrip` — load, edit a few bytes, save.
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/roundtrip");
    for &size in &[4 * 1024usize, 64 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let disk = MemoryDisk::new(256).unwrap();
                let mut fs = DiskFs::format(disk, 256, 128).unwrap();
                let data = payload(size, 0xFEED_FACE);
                let inode = fs.create(2, "roundtrip.bin", 0x8000).unwrap();
                fs.write_at(inode, 0, &data).unwrap();
                let mut bound: Bound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
                // Edit: replace the middle byte.
                let mid = size / 2;
                bound
                    .replace_range(mid, mid + 1, &[make_packed(b'X' as u32, COMMON_MASK)])
                    .unwrap();
                bound.save().unwrap();
                black_box(bound.raw_length())
            });
        });
    }
    group.finish();
}

/// `bind/edit` — pure edit operations through the binding, no disk I/O
/// beyond the initial load.
fn bench_edit(c: &mut Criterion) {
    const OPS: usize = 64;
    let mut group = c.benchmark_group("bind/edit");
    for &size in &[16 * 1024usize, 64 * 1024] {
        group.throughput(Throughput::Elements(OPS as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let disk = MemoryDisk::new(256).unwrap();
                let mut fs = DiskFs::format(disk, 256, 128).unwrap();
                let data = payload(size, 0xABCD_EF01);
                let inode = fs.create(2, "edit.bin", 0x8000).unwrap();
                fs.write_at(inode, 0, &data).unwrap();
                let mut bound: Bound<'_> = BoundDocument::open(&mut fs, inode).unwrap();
                bound.flush_all();
                for k in 0..OPS {
                    let pos = (k * 7) % size;
                    bound
                        .replace_range(
                            pos,
                            pos + 1,
                            &[make_packed(b'Z' as u32 + (k as u8) as u32, COMMON_MASK)],
                        )
                        .unwrap();
                }
                black_box(bound.raw_length())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_load, bench_save, bench_roundtrip, bench_edit);
criterion_main!(benches);
