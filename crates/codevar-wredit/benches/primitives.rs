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

//! Criterion benchmarks for the Wredit primitives.
//!
//! Each vectorized kernel is measured against its own scalar reference
//! so the speedup from the SIMD/assembly path is visible, and the
//! hand-written `arch::mem*` routines are measured against the
//! compiler's `core::ptr`/libc lowering:
//!
//! * `simd/*`  - UTF-8 scanning, byte search/count/filter, checksum, hash
//! * `arch/*`  - assembly `memcpy`/`memmove`/`memchr` vs the libc path
//! * `codec/*` - path-name compression / decompression
//! * `change/*`- capped Levenshtein distance
//! * `storage/*` - in-memory block device read/write

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_wredit::arch;
use codevar_wredit::change::{ChangeFlags, change_distance};
use codevar_wredit::compress::{compress, decompress};
use codevar_wredit::simd;
use codevar_wredit::storage::{DiskFs, MemoryDisk};

/// The benchmark sizes for streaming kernels, in bytes.
const SIZES: [usize; 3] = [4 * 1024, 64 * 1024, 1 << 20];

/// Bytes with a repeating pattern; `2` is deliberately absent from it.
fn pattern(len: usize, modulus: usize) -> Vec<u8> {
    (0..len).map(|i| (i % modulus) as u8).collect()
}

/// Scalar reference for [`simd::count_utf8_units`]: a code unit starts at
/// every byte that is not a continuation byte.
fn count_utf8_units_scalar(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .filter(|&&byte| byte & 0xC0 != 0x80)
        .count() as u64
}

/// Counts one byte, vectorised vs scalar.
fn bench_count_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/count_byte");
    for &size in &SIZES {
        let data = pattern(size, 251);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("auto", size), &data, |b, data| {
            b.iter(|| black_box(simd::count_byte(black_box(data), 200)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| black_box(simd::count_byte_scalar(black_box(data), 200)));
        });
    }
    group.finish();
}

/// Finds a byte that only appears in the final position, so every block
/// of the haystack is scanned.
fn bench_find_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/find_byte");
    for &size in &SIZES {
        let mut data = vec![0x01u8; size];
        if let Some(last) = data.last_mut() {
            *last = 0xFF;
        }
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("auto", size), &data, |b, data| {
            b.iter(|| black_box(simd::find_byte(black_box(data), 0xFF)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| black_box(simd::find_byte_scalar(black_box(data), 0xFF)));
        });
    }
    group.finish();
}

/// Counts UTF-8 code units over ASCII and mixed-width text.
fn bench_utf8_units(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/count_utf8_units");
    for &size in &SIZES {
        let ascii = pattern(size, 128);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("ascii_auto", size), &ascii, |b, data| {
            b.iter(|| black_box(simd::count_utf8_units(black_box(data))));
        });
        group.bench_with_input(BenchmarkId::new("ascii_scalar", size), &ascii, |b, data| {
            b.iter(|| black_box(count_utf8_units_scalar(black_box(data))));
        });

        // A stream of two-byte `é` sequences: half of every block is a
        // continuation byte.
        let mut wide = Vec::with_capacity(size);
        while wide.len() < size {
            wide.extend_from_slice("é".as_bytes());
        }
        wide.truncate(size);
        group.bench_with_input(BenchmarkId::new("wide_auto", size), &wide, |b, data| {
            b.iter(|| black_box(simd::count_utf8_units(black_box(data))));
        });
    }
    group.finish();
}

/// Maps unit indices to byte offsets and back.
fn bench_utf8_map(c: &mut Criterion) {
    const SIZE: usize = 1 << 20;
    let data = pattern(SIZE, 128);
    let total = simd::count_utf8_units(&data);
    let mut group = c.benchmark_group("simd/utf8_map");
    group.throughput(Throughput::Bytes(SIZE as u64));

    group.bench_function("unit_to_byte_random", |b| {
        let mut state = 0x2468_ACE0u64;
        b.iter(|| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = state % total;
            black_box(simd::utf8_unit_index_to_byte(black_box(&data), unit))
        });
    });
    group.bench_function("byte_to_unit_random", |b| {
        let mut state = 0x1357_9BDFu64;
        b.iter(|| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let offset = (state as usize) % SIZE;
            black_box(simd::utf8_byte_index_to_unit(black_box(&data), offset))
        });
    });
    group.finish();
}

/// Removes a byte that is either absent or present in one of eight
/// positions.
fn bench_filter_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/filter_byte");
    for &size in &SIZES {
        let dense = pattern(size, 8);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("auto", size), &dense, |b, data| {
            b.iter(|| black_box(simd::filter_byte(black_box(data), 3)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &dense, |b, data| {
            b.iter(|| black_box(simd::filter_byte_scalar(black_box(data), 3)));
        });
    }
    group.finish();
}

/// The position-weighted integrity checksum.
fn bench_checksum(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/linear_checksum");
    for &size in &SIZES {
        let data = pattern(size, 251);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("auto", size), &data, |b, data| {
            b.iter(|| black_box(simd::linear_checksum(black_box(data))));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| black_box(simd::checksum_scalar(black_box(data))));
        });
    }
    group.finish();
}

/// The polynomial name hash.
fn bench_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/hash_bytes");
    for &size in &SIZES {
        let data = pattern(size, 251);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("auto", size), &data, |b, data| {
            b.iter(|| black_box(simd::hash_bytes(black_box(data))));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| black_box(simd::hash_bytes_scalar(black_box(data))));
        });
    }
    group.finish();
}

/// Hand-written assembly `memcpy`/`memmove` against the compiler's
/// intrinsic lowering (glibc `memcpy`/`memmove` on ELF).
fn bench_arch_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("arch/copy");
    for &size in &[64usize, 4096, 64 * 1024, 1 << 20] {
        group.throughput(Throughput::Bytes(size as u64));
        let src = vec![0xA5u8; size];

        group.bench_with_input(BenchmarkId::new("memcpy/asm", size), &size, |b, &size| {
            let mut dst = vec![0u8; size];
            let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
            b.iter(|| unsafe {
                arch::memcpy(black_box(dst), black_box(src), black_box(size));
            });
        });
        group.bench_with_input(BenchmarkId::new("memcpy/libc", size), &size, |b, &size| {
            let mut dst = vec![0u8; size];
            let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
            b.iter(|| unsafe {
                std::ptr::copy_nonoverlapping(black_box(src), black_box(dst), black_box(size));
            });
        });
        group.bench_with_input(BenchmarkId::new("memmove/asm", size), &size, |b, &size| {
            let mut dst = vec![0u8; size];
            let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
            b.iter(|| unsafe {
                arch::memmove(black_box(dst), black_box(src), black_box(size));
            });
        });
        group.bench_with_input(BenchmarkId::new("memmove/libc", size), &size, |b, &size| {
            let mut dst = vec![0u8; size];
            let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
            b.iter(|| unsafe {
                std::ptr::copy(black_box(src), black_box(dst), black_box(size));
            });
        });
        // Backward overlap (dst > src): the direction the edit engine
        // uses for every insert/flush tail shift.
        group.bench_with_input(BenchmarkId::new("memmove_back/asm", size), &size, |b, &size| {
            let mut buf = vec![0u8; size + 64];
            let (src, dst) = (buf.as_ptr(), unsafe { buf.as_mut_ptr().add(64) });
            b.iter(|| unsafe {
                arch::memmove(black_box(dst), black_box(src), black_box(size));
            });
        });
        group.bench_with_input(BenchmarkId::new("memmove_back/libc", size), &size, |b, &size| {
            let mut buf = vec![0u8; size + 64];
            let (src, dst) = (buf.as_ptr(), unsafe { buf.as_mut_ptr().add(64) });
            b.iter(|| unsafe {
                std::ptr::copy(black_box(src), black_box(dst), black_box(size));
            });
        });
    }
    group.finish();
}

/// First-occurrence search: assembly `memchr` vs the SIMD kernel vs a
/// plain Rust scan.
fn bench_arch_memchr(c: &mut Criterion) {
    let mut group = c.benchmark_group("arch/memchr");
    for &size in &SIZES {
        let mut data = vec![0x01u8; size];
        if let Some(last) = data.last_mut() {
            *last = 0xFF;
        }
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("asm", size), &data, |b, data| {
            b.iter(|| unsafe { black_box(arch::memchr(black_box(data.as_ptr()), 0xFF, data.len())) });
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, data| {
            b.iter(|| black_box(simd::find_byte(black_box(data), 0xFF)));
        });
        group.bench_with_input(BenchmarkId::new("rust_scan", size), &data, |b, data| {
            b.iter(|| black_box(data.iter().position(|&byte| byte == 0xFF)));
        });
    }
    group.finish();
}

/// Path-name compression and decompression.
fn bench_codec(c: &mut Criterion) {
    const NAMES: &[&[u8]] = &[
        b"src/main.rs",
        b"crates/codevar-wredit/src/storage/fs.rs",
        b"node_modules",
        b"tests/test_storage.rs",
        b"docs/readme.md",
        b"a_very_long_directory_name_that_should_compress_well",
    ];
    let packed: Vec<Vec<u8>> = NAMES.iter().map(|name| compress(name)).collect();
    let mut group = c.benchmark_group("codec");

    group.bench_function("compress_path_names", |b| {
        let mut index = 0usize;
        b.iter(|| {
            let name = NAMES[index % NAMES.len()];
            index += 1;
            black_box(compress(name))
        });
    });
    group.bench_function("decompress_path_names", |b| {
        let mut index = 0usize;
        b.iter(|| {
            let block = &packed[index % packed.len()];
            index += 1;
            black_box(decompress(block))
        });
    });
    group.finish();
}

/// Capped Levenshtein distance over text-sized inputs.
fn bench_change(c: &mut Criterion) {
    let mut group = c.benchmark_group("change/levenshtein");
    for &len in &[64usize, 256, 1024] {
        let a: Vec<u32> = (0..len as u32).map(|value| value % 13).collect();
        let b: Vec<u32> = (0..len as u32)
            .map(|value| (value * 5) % 13)
            .collect();
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &(a, b), |bencher, (a, b)| {
            bencher.iter(|| {
                black_box(change_distance(
                    black_box(a),
                    black_box(b),
                    8,
                    ChangeFlags::DEFAULT,
                ))
            });
        });
    }
    group.finish();
}

/// In-memory block-device reads and writes through `DiskFs`.
fn bench_storage(c: &mut Criterion) {
    const BLOCKS: u64 = 16_384;
    const WRITES: u64 = 64;
    let mut group = c.benchmark_group("storage");
    group.throughput(Throughput::Bytes(WRITES * 4096));

    group.bench_function("write_at_4k", |b| {
        b.iter_batched(
            || {
                let disk = MemoryDisk::new(BLOCKS).ok()?;
                let mut fs = DiskFs::format(disk, BLOCKS, 4096).ok()?;
                let ino = fs.create(2, "bench.bin", 0x8000).ok()?;
                Some((fs, ino))
            },
            |state| {
                if let Some((mut fs, ino)) = state {
                    let chunk = [0xA5u8; 4096];
                    for block in 0..WRITES {
                        let _ = fs.write_at(ino, block * 4096, &chunk);
                    }
                    let _ = fs.flush();
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("read_at_4k", |b| {
        b.iter_batched(
            || {
                let disk = MemoryDisk::new(BLOCKS).ok()?;
                let mut fs = DiskFs::format(disk, BLOCKS, 4096).ok()?;
                let ino = fs.create(2, "bench.bin", 0x8000).ok()?;
                let chunk = [0xA5u8; 4096];
                for block in 0..WRITES {
                    let _ = fs.write_at(ino, block * 4096, &chunk);
                }
                let _ = fs.flush();
                Some((fs, ino))
            },
            |state| {
                if let Some((mut fs, ino)) = state {
                    let mut buf = [0u8; 4096];
                    for block in 0..WRITES {
                        let _ = fs.read_at(ino, block * 4096, &mut buf);
                    }
                    black_box(buf);
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_count_byte,
    bench_find_byte,
    bench_utf8_units,
    bench_utf8_map,
    bench_filter_byte,
    bench_checksum,
    bench_hash,
    bench_arch_copy,
    bench_arch_memchr,
    bench_codec,
    bench_change,
    bench_storage,
);
criterion_main!(benches);
