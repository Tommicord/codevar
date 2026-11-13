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

//! Criterion benchmarks for the hot undo/redo path in
//! `crates/wredit/src/history_ring_journal.rs`.
//!
//! Target: lightweight undo/redo operations in the 5–15 ns range, and
//! most operations (record/restore/delta derivation) under 1 µs.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_wredit::history_ring_journal::{DeltaRing, Edit, HistoryStore, delta_common};

/// Builds a small edit (insertion of a single element) at `position`.
fn small_edit(position: usize) -> Edit {
    Edit::new(vec![42], vec![99], position, 1)
}

/// Records a sequence of `count` single-element edits.
fn bench_record(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring/record");
    group.measurement_time(core::time::Duration::from_secs(5));
    for &count in &[1usize, 16, 64, 256, 1024] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("single_element", count), &count, |b, &count| {
            b.iter_batched(
                || DeltaRing::new(64 * 1024).expect("ring"),
                |mut ring| {
                    for i in 0..count {
                        let _ = ring.record(small_edit(i % 16));
                    }
                    black_box(&ring);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Undo/redo a single lightweight step at the tip of a populated ring.
///
/// Uses `BatchSize::LargeInput` so the ring is built once and only the
/// undo+redo walk is timed — this measures the steady-state per-operation
/// latency (target: 5–15 ns for lightweight edits).
fn bench_undo_redo_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring/undo_redo_step");
    group.measurement_time(core::time::Duration::from_secs(5));
    for &count in &[16usize, 64, 256, 1024] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new(format!("ring_{}", count), count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut ring = DeltaRing::new(64 * 1024).expect("ring");
                        for i in 0..count {
                            let _ = ring.record(small_edit(i % 16));
                        }
                        ring
                    },
                    |mut ring| {
                        let ok = ring.undo(|_, _| Ok(()));
                        black_box(ok);
                        let ok = ring.redo(|_, _| Ok(()));
                        black_box(ok);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Undo/redo a long walk (many steps) to stress prefetch and slot math.
fn bench_undo_redo_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring/undo_redo_walk");
    group.measurement_time(core::time::Duration::from_secs(5));
    for &steps in &[1usize, 16, 64, 256] {
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("steps_{}", steps), steps),
            &steps,
            |b, &steps| {
                b.iter_batched(
                    || {
                        let mut ring = DeltaRing::new(64 * 1024).expect("ring");
                        for i in 0..1024 {
                            let _ = ring.record(small_edit(i % 16));
                        }
                        ring
                    },
                    |mut ring| {
                        for _ in 0..steps {
                            let _ = ring.undo(|_, _| Ok(()));
                        }
                        for _ in 0..steps {
                            let _ = ring.redo(|_, _| Ok(()));
                        }
                        black_box(&ring);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Restore to various depths (bounded-span replay).
fn bench_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring/restore");
    group.measurement_time(core::time::Duration::from_secs(5));
    for &back in &[1usize, 16, 64, 256, 1024] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new(format!("back_{}", back), back),
            &back,
            |b, &back| {
                b.iter_batched(
                    || {
                        let mut ring = DeltaRing::new(64 * 1024).expect("ring");
                        for i in 0..2048 {
                            let _ = ring.record(small_edit(i % 16));
                        }
                        ring
                    },
                    |mut ring| {
                        let ok = ring.restore(back, |_, _| Ok(()));
                        black_box(ok);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// delta_common prefix/suffix derivation over document states of various
/// sizes (the per-keystroke delta window computation).
fn bench_delta_common(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta/common");
    group.measurement_time(core::time::Duration::from_secs(5));
    for &len in &[16usize, 64, 256, 1024, 4096] {
        group.throughput(Throughput::Bytes((len * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("len_{}", len), len),
            &len,
            |bench, &len| {
                let a: Vec<u32> = (0..len as u32).collect();
                let mut b = a.clone();
                if len > 2 {
                    b[len / 2] ^= 1; // difference in the middle
                }
                bench.iter(|| {
                    let (p, s) = delta_common(black_box(&a), black_box(&b));
                    black_box((p, s))
                });
            },
        );
    }
    group.finish();
}

/// Full HistoryStore: record + undo + redo roundtrip.
fn bench_store_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("store/roundtrip");
    group.measurement_time(core::time::Duration::from_secs(5));
    for &count in &[16usize, 256, 1024] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("edits_{}", count), count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut store = HistoryStore::new(64 * 1024, 64, 256).expect("store");
                        for i in 0..count {
                            let _ = store.record(i % 16, &[i as u32], &[i as u32 + 1]);
                        }
                        store
                    },
                    |mut store| {
                        for _ in 0..count / 2 {
                            let _ = store.undo();
                        }
                        for _ in 0..count / 2 {
                            let _ = store.redo();
                        }
                        black_box(&store);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_record,
    bench_undo_redo_step,
    bench_undo_redo_walk,
    bench_restore,
    bench_delta_common,
    bench_store_roundtrip,
);
criterion_main!(benches);
