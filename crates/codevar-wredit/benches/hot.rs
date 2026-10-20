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

//! Criterion benchmarks for the Wredit hardware-accelerated (HOT)
//! operations.
//!
//! Every benchmark runs the portable CPU reference next to the OpenCL
//! kernel behind [`codevar_wredit::hot`], so the cost of the GPU
//! round-trip (upload, dispatch, wait, readback) is directly visible
//! against the scalar work it replaces:
//!
//! * `hot/levenshtein`      - banded DP: `levenshtein_capped` vs `try_levenshtein`
//! * `hot/change_distance`  - the engine-level `DEFAULT` vs `HOT` flag
//! * `hot/gap_shift_insert` - two `ptr::copy` calls vs `gap_shift_insert`
//! * `hot/gap_shift_delete` - two `ptr::copy` calls vs `gap_shift_delete`
//! * `hot/find_matches`     - scalar scan vs `find_matches`
//! * `hot/utf8_validate`    - `str::from_utf8` vs `utf8_validate`
//!
//! The OpenCL pipeline is built on first use and cached for the
//! process lifetime, and only one kernel is cached at a time. Each
//! group therefore probes (and warms) its own kernel once before
//! timing, so the one-time SPIR-V build is never part of a sample.
//! When no usable OpenCL GPU is present, the GPU half of every group
//! is skipped and the CPU reference numbers still run.
//!
//! The CPU references are the operations the engine falls back to:
//! [`levenshtein_capped`] for the change distance, `ptr::copy` for the
//! gap shifts, a scalar needle scan for pattern matching, and the
//! standard library's UTF-8 validator.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_wredit::change::{ChangeFlags, change_distance, levenshtein_capped};
use codevar_wredit::hot;

/// Input lengths exercised by the change-distance groups.
const DISTANCE_LENGTHS: [usize; 3] = [64, 512, 4096];

/// Bands exercised for every change-distance input length.
const DISTANCE_CAPS: [u32; 2] = [8, 64];

/// Gap width opened/closed by the gap-buffer shift benchmarks.
const SHIFT_DELTA: usize = 64;

/// Streaming sizes exercised by the byte/pattern groups.
const STREAM_SIZES: [usize; 3] = [4 * 1024, 64 * 1024, 1 << 20];

/// Needle length used by the pattern-matching benchmarks.
const NEEDLE_LEN: usize = 16;

/// Short per-sample budget: several GPU kernels cost milliseconds per
/// dispatch, so the default five-second measurement time would make a
/// full run needlessly long.
const MEASUREMENT_TIME: Duration = Duration::from_secs(3);

/// One-second warm-up: enough for Criterion to settle the caches while
/// keeping the GPU groups affordable.
const WARM_UP_TIME: Duration = Duration::from_secs(1);

/// xorshift64 generator; deterministic inputs without a dependency.
struct Rng(u64);

impl Rng {
    #[inline]
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    #[inline]
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 { 0 } else { self.next_u64() % bound }
    }
}

/// `len` deterministic code points in `0..64`.
fn sequence(len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..len).map(|_| rng.below(64) as u32).collect()
}

/// A copy of `source` with periodic substitutions and a short deletion,
/// so the edit distance stays small and the banded DP has to visit its
/// whole band instead of exiting early.
fn edited(source: &[u32]) -> Vec<u32> {
    let mut edited = source.to_vec();
    let stride = (source.len() / 16).max(1);
    for index in (1..source.len()).step_by(stride) {
        if let Some(slot) = edited.get_mut(index) {
            *slot = (*slot + 1) % 64;
        }
    }
    if edited.len() > 64 {
        edited.drain(32..35).for_each(drop);
    }
    edited
}

/// CPU reference for the `find_matches` kernel: one pass over every
/// possible start, comparing the needle element by element.
fn find_matches_cpu(data: &[u32], needle: &[u32]) -> Vec<u32> {
    let mut matches = vec![0u32; data.len()];
    if needle.is_empty() || needle.len() > data.len() {
        return matches;
    }
    let last_start = data.len() - needle.len();
    for (start, slot) in matches
        .iter_mut()
        .enumerate()
        .take(last_start + 1)
    {
        let mut hit = 1u32;
        for offset in 0..needle.len() {
            if data[start + offset] != needle[offset] {
                hit = 0;
                break;
            }
        }
        *slot = hit;
    }
    matches
}

/// Builds `(data, needle)` for the pattern-matching benchmarks.
///
/// `dense` makes every position a full-length match (worst case for the
/// scalar comparison loop), while `sparse` fails on the first element
/// (the common case for real searches).
fn match_pattern(len: usize, needle_len: usize, dense: bool) -> (Vec<u32>, Vec<u32>) {
    if dense {
        (vec![7u32; len], vec![7u32; needle_len])
    } else {
        let data = (0..len)
            .map(|index| (index % 251) as u32)
            .collect();
        (data, vec![300u32; needle_len])
    }
}

/// ASCII bytes, all well-formed one-byte sequences.
fn ascii_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| (index % 128) as u8)
        .collect()
}

/// Repeating two-byte `é`, all well-formed.
fn wide_bytes(len: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(len);
    while data.len() < len {
        data.extend_from_slice("é".as_bytes());
    }
    data.truncate(len);
    data
}

/// ASCII with a stray `0xFF` at the very end: the validator has to scan
/// the whole buffer before rejecting it.
fn invalid_bytes(len: usize) -> Vec<u8> {
    let mut data = ascii_bytes(len);
    if let Some(last) = data.last_mut() {
        *last = 0xFF;
    }
    data
}

/// `true` when the Levenshtein kernel can be dispatched.
///
/// The probe builds and caches the pipeline, and the predicate also
/// guards against a driver that builds the program but fails the
/// dispatch: a `None` here means the group must stay CPU-only.
fn gpu_levenshtein_ready() -> bool {
    hot::try_levenshtein(&[1, 2, 3, 4], &[1, 2, 4], 4).is_some()
}

/// `true` when the insert-shift kernel can be dispatched.
fn gpu_gap_insert_ready() -> bool {
    let src = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let mut dst = [0u32; 10];
    // Safety: `dst` has room for `src.len() + 2` elements.
    unsafe { hot::gap_shift_insert(src.as_ptr(), src.len(), 4, 2, dst.as_mut_ptr()).is_some() }
}

/// `true` when the delete-shift kernel can be dispatched.
fn gpu_gap_delete_ready() -> bool {
    let src = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let mut dst = [0u32; 6];
    // Safety: `dst` has room for `src.len() - 2` elements.
    unsafe { hot::gap_shift_delete(src.as_ptr(), src.len(), 4, 2, dst.as_mut_ptr()).is_some() }
}

/// `true` when the pattern-matching kernel can be dispatched.
fn gpu_find_matches_ready() -> bool {
    let data = [1u32, 2, 3, 4];
    let needle = [2u32, 3];
    // Safety: both slices are valid for their stated lengths.
    unsafe { hot::find_matches(data.as_ptr(), data.len(), needle.as_ptr(), needle.len()).is_some() }
}

/// `true` when the UTF-8 validation kernel can be dispatched.
fn gpu_utf8_ready() -> bool {
    let data = b"benchmark";
    // Safety: `data` is valid for its length.
    unsafe { hot::utf8_validate(data.as_ptr(), data.len()).is_some() }
}

/// Capped Levenshtein DP, CPU reference against the GPU kernel.
fn bench_levenshtein(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot/levenshtein");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let gpu = gpu_levenshtein_ready();

    for &len in &DISTANCE_LENGTHS {
        let a = sequence(len, 0x9E37_79B9);
        let b = edited(&a);
        group.throughput(Throughput::Elements(len as u64));

        for &cap in &DISTANCE_CAPS {
            group.bench_function(BenchmarkId::new(format!("cpu/cap{cap}"), len), |bencher| {
                bencher.iter(|| black_box(levenshtein_capped(black_box(&a), black_box(&b), black_box(cap))));
            });

            // Probe the concrete input so a failed dispatch (rather
            // than a missing device) also keeps the group CPU-only.
            if gpu && hot::try_levenshtein(&a, &b, cap).is_some() {
                group.bench_function(BenchmarkId::new(format!("gpu/cap{cap}"), len), |bencher| {
                    bencher.iter(|| {
                        black_box(hot::try_levenshtein(black_box(&a), black_box(&b), black_box(cap)))
                    });
                });
            }
        }
    }
    group.finish();
}

/// The engine-level entry point: `DEFAULT` always runs the CPU
/// reference; `HOT` takes the GPU path and falls back when unavailable.
fn bench_change_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot/change_distance");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let gpu = gpu_levenshtein_ready();
    let cap = 64u32;

    for &len in &DISTANCE_LENGTHS {
        let a = sequence(len, 0x1234_5678);
        let b = edited(&a);
        group.throughput(Throughput::Elements(len as u64));

        group.bench_function(BenchmarkId::new("cpu", len), |bencher| {
            bencher.iter(|| {
                black_box(change_distance(
                    black_box(&a),
                    black_box(&b),
                    black_box(cap),
                    ChangeFlags::DEFAULT,
                ))
            });
        });

        if gpu && hot::try_levenshtein(&a, &b, cap).is_some() {
            group.bench_function(BenchmarkId::new("gpu", len), |bencher| {
                bencher.iter(|| {
                    black_box(change_distance(
                        black_box(&a),
                        black_box(&b),
                        black_box(cap),
                        ChangeFlags::HOT,
                    ))
                });
            });
        }
    }
    group.finish();
}

/// Opening a gap: the CPU path shifts the head and tail with two
/// `ptr::copy_nonoverlapping` calls, the GPU path uploads, dispatches,
/// waits, and reads the shifted buffer back.
fn bench_gap_shift_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot/gap_shift_insert");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let gpu = gpu_gap_insert_ready();

    for &len in &STREAM_SIZES {
        let src = sequence(len, 0x00C0_FFEE);
        let at = len / 2;
        let mut dst = vec![0u32; len + SHIFT_DELTA];
        group.throughput(Throughput::Elements((len + SHIFT_DELTA) as u64));

        group.bench_function(BenchmarkId::new("cpu", len), |bencher| {
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            bencher.iter(|| {
                // Safety: `dst` is `len + SHIFT_DELTA` elements long;
                // `src` is `len` elements long and does not alias it.
                unsafe {
                    core::ptr::copy_nonoverlapping(black_box(src_ptr), black_box(dst_ptr), at);
                    core::ptr::copy_nonoverlapping(
                        black_box(src_ptr.add(at)),
                        black_box(dst_ptr.add(at + SHIFT_DELTA)),
                        len - at,
                    );
                    // The kernel leaves the newly opened gap zeroed.
                    core::ptr::write_bytes(black_box(dst_ptr.add(at)), 0, SHIFT_DELTA);
                }
            });
        });

        if gpu {
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            // Probe this exact size before timing it.
            // Safety: the destination has room for `len + SHIFT_DELTA`.
            if unsafe { hot::gap_shift_insert(src_ptr, len, at, SHIFT_DELTA, dst_ptr).is_some() } {
                group.bench_function(BenchmarkId::new("gpu", len), |bencher| {
                    bencher.iter(|| {
                        // Safety: same buffer contract as the CPU case.
                        black_box(unsafe {
                            hot::gap_shift_insert(
                                black_box(src_ptr),
                                black_box(len),
                                black_box(at),
                                black_box(SHIFT_DELTA),
                                black_box(dst_ptr),
                            )
                        })
                    });
                });
            }
        }
    }
    group.finish();
}

/// Closing a gap: same CPU/GPU split as the insert direction.
fn bench_gap_shift_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot/gap_shift_delete");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let gpu = gpu_gap_delete_ready();

    for &len in &STREAM_SIZES {
        let src = sequence(len, 0x0BAD_F00D);
        let at = len / 2;
        let mut dst = vec![0u32; len - SHIFT_DELTA];
        group.throughput(Throughput::Elements((len - SHIFT_DELTA) as u64));

        group.bench_function(BenchmarkId::new("cpu", len), |bencher| {
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            bencher.iter(|| {
                // Safety: `dst` is `len - SHIFT_DELTA` elements long;
                // `src` is `len` elements long and does not alias it.
                unsafe {
                    core::ptr::copy_nonoverlapping(black_box(src_ptr), black_box(dst_ptr), at);
                    core::ptr::copy_nonoverlapping(
                        black_box(src_ptr.add(at + SHIFT_DELTA)),
                        black_box(dst_ptr.add(at)),
                        len - at - SHIFT_DELTA,
                    );
                }
            });
        });

        if gpu {
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            // Safety: the destination has room for `len - SHIFT_DELTA`.
            if unsafe { hot::gap_shift_delete(src_ptr, len, at, SHIFT_DELTA, dst_ptr).is_some() } {
                group.bench_function(BenchmarkId::new("gpu", len), |bencher| {
                    bencher.iter(|| {
                        // Safety: same buffer contract as the CPU case.
                        black_box(unsafe {
                            hot::gap_shift_delete(
                                black_box(src_ptr),
                                black_box(len),
                                black_box(at),
                                black_box(SHIFT_DELTA),
                                black_box(dst_ptr),
                            )
                        })
                    });
                });
            }
        }
    }
    group.finish();
}

/// Pattern matching: scalar start-by-start scan against the parallel
/// kernel, for both a worst-case (every position matches) and a
/// common-case (first element rejects) haystack.
fn bench_find_matches(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot/find_matches");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let gpu = gpu_find_matches_ready();

    for &len in &STREAM_SIZES {
        group.throughput(Throughput::Elements(len as u64));
        for &(label, dense) in &[("dense", true), ("sparse", false)] {
            let (data, needle) = match_pattern(len, NEEDLE_LEN, dense);
            let data_slice = data.as_slice();
            let needle_slice = needle.as_slice();

            group.bench_function(BenchmarkId::new(format!("cpu/{label}"), len), |bencher| {
                bencher.iter(|| black_box(find_matches_cpu(black_box(data_slice), black_box(needle_slice))));
            });

            if gpu {
                let data_ptr = data.as_ptr();
                let needle_ptr = needle.as_ptr();
                // Safety: both buffers are valid for their lengths.
                if unsafe { hot::find_matches(data_ptr, len, needle_ptr, needle.len()).is_some() } {
                    group.bench_function(BenchmarkId::new(format!("gpu/{label}"), len), |bencher| {
                        bencher.iter(|| {
                            // Safety: same buffer contract as the CPU case.
                            black_box(unsafe {
                                hot::find_matches(
                                    black_box(data_ptr),
                                    black_box(len),
                                    black_box(needle_ptr),
                                    black_box(NEEDLE_LEN),
                                )
                            })
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

/// UTF-8 validation on three shapes: all-ASCII, all two-byte
/// sequences, and a buffer whose final byte is invalid.
fn bench_utf8_validate(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot/utf8_validate");
    group.sample_size(20);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    let gpu = gpu_utf8_ready();

    for &len in &STREAM_SIZES {
        group.throughput(Throughput::Bytes(len as u64));
        let inputs = [
            ("ascii", ascii_bytes(len)),
            ("wide", wide_bytes(len)),
            ("invalid", invalid_bytes(len)),
        ];

        for (label, data) in inputs {
            group.bench_function(BenchmarkId::new(format!("cpu/{label}"), len), |bencher| {
                bencher.iter(|| black_box(core::str::from_utf8(black_box(data.as_slice())).is_ok()));
            });

            if gpu {
                let data_ptr = data.as_ptr();
                // Safety: `data` is valid for `len` bytes.
                if unsafe { hot::utf8_validate(data_ptr, len).is_some() } {
                    group.bench_function(BenchmarkId::new(format!("gpu/{label}"), len), |bencher| {
                        bencher.iter(|| {
                            // Safety: same buffer contract as the CPU case.
                            black_box(unsafe { hot::utf8_validate(black_box(data_ptr), black_box(len)) })
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_levenshtein,
    bench_change_distance,
    bench_gap_shift_insert,
    bench_gap_shift_delete,
    bench_find_matches,
    bench_utf8_validate,
);
criterion_main!(benches);
