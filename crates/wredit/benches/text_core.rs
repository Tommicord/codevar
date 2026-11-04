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

//! Criterion benchmarks for the Wredit text-editing core.
//!
//! The engine is driven through its public `TextEditable*` traits on a
//! `StreamWritable<u32, u32, 1024>` document — the same instantiation
//! the unit tests exercise — so the numbers cover the full path a
//! frontend takes: caret bookkeeping, staging into the private gap,
//! and the flush layer's block moves.
//!
//! Grouped by concern:
//!
//! * `engine/append`   - stage one character and commit (end of file)
//! * `engine/insert`   - stage one character and commit at front/middle
//! * `engine/stage`    - stage without committing
//! * `engine/bulk`     - stage a whole buffer, then a single flush
//! * `engine/delete`   - remove a character (front/end) and a range
//! * `engine/replace`  - replace a range / overwrite in place
//! * `engine/flush`    - the raw gap-commit block move in isolation
//! * `engine/multicursor` - several private gaps committed at once
//! * `query/*`         - random `char_at` / `raw_slice` reads
//! * `nav/*`           - row/column resolution over a 5000-line file

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_wredit::edit::{
    COMMON_MASK, Cursor, StreamWritable, TextEditableBufferQuery, TextEditableCursor, TextEditableDelete,
    TextEditableOverwrite, TextEditablePut, TextEditableReplace, Writable, make_packed,
};

/// Private gap window used by every benchmark document, in elements.
///
/// 1024 matches the batch/paste oriented layout the engine is tuned
/// for, so a staged run commits every 1024 characters.
const GAP_SIZE: usize = 1024;

/// The concrete engine instantiation under test.
type Stream = StreamWritable<u32, u32, GAP_SIZE>;

/// Packs an ASCII character as a committed document element.
#[inline]
fn plain(character: u32) -> u32 {
    make_packed(character, COMMON_MASK)
}

/// Deterministic xorshift generator, so every run sees identical input
/// without pulling a random-number dependency into the build.
struct Rng(u64);

impl Rng {
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
    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }
}

/// Builds an empty stream with one active cursor and pre-grown capacity.
fn new_stream(capacity: usize) -> Stream {
    let mut stream = Stream::with_capacity(capacity);
    if let Some(id) = stream.append_cursor(Cursor::new()) {
        stream.set_active_cursor(id);
    }
    stream
}

/// Appends `text` to a fresh single-line document and commits it.
fn document(text: &[u32]) -> Stream {
    let mut stream = new_stream(text.len() + GAP_SIZE);
    for &element in text {
        let _ = stream.put(0, element);
    }
    let _ = stream.flush_all();
    stream
}

/// `len` lowercase ASCII letters drawn from a fixed seed.
fn ascii(len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..len)
        .map(|_| plain(b'a' as u32 + rng.below(26) as u32))
        .collect()
}

/// Stage one character and commit it, at the end of the document.
fn bench_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/append");
    for &size in &[4 * 1024usize, 64 * 1024] {
        let text = ascii(size, 0x9E37_79B9);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter_batched_ref(
                || new_stream(text.len() + GAP_SIZE),
                |stream| {
                    for &element in text {
                        let _ = stream.put(0, element);
                        let _ = stream.flush_all();
                    }
                    black_box(stream.raw_length())
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Stage one character and commit it at the front / middle, forcing the
/// flush layer to shift the whole tail every time.
fn bench_insert(c: &mut Criterion) {
    const OPS: usize = 32;
    let mut group = c.benchmark_group("engine/insert");
    for &size in &[16 * 1024usize, 64 * 1024] {
        let text = ascii(size, 0x00C0_FFEE);
        group.throughput(Throughput::Elements(OPS as u64));
        for &(label, at) in &[("front", 0usize), ("middle", size / 2)] {
            group.bench_with_input(BenchmarkId::new(label, size), &text, |b, text| {
                b.iter_batched_ref(
                    || document(text),
                    |stream| {
                        for k in 0..OPS {
                            let _ = stream.move_cursor_to(0, 0, at);
                            let _ = stream.put(0, plain(b'x' as u32 + (k % 26) as u32));
                            let _ = stream.flush_all();
                        }
                        black_box(stream.raw_length())
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

/// Stage characters into the private gap without committing them: the
/// cost is the geometric gap growth, not a block move.
fn bench_stage(c: &mut Criterion) {
    let text = ascii(8 * GAP_SIZE, 0x1234_5678);
    let mut group = c.benchmark_group("engine/stage");
    group.throughput(Throughput::Elements(text.len() as u64));
    group.bench_function("put_no_flush", |b| {
        b.iter_batched_ref(
            || new_stream(text.len() + GAP_SIZE),
            |stream| {
                for &element in &text {
                    let _ = stream.put(0, element);
                }
                black_box(stream.raw_length())
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Stage a whole buffer and commit it with a single flush.
fn bench_bulk(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/bulk");
    for &size in &[4 * 1024usize, 64 * 1024, 1 << 20] {
        let text = ascii(size, 0xFEED_FACE);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter_batched_ref(
                || new_stream(text.len() + GAP_SIZE),
                |stream| {
                    for &element in text {
                        let _ = stream.put(0, element);
                    }
                    black_box(stream.flush_all())
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Remove characters from the front (full tail shift) and the end
/// (append/truncate case).
fn bench_delete(c: &mut Criterion) {
    const OPS: usize = 32;
    let mut group = c.benchmark_group("engine/delete");
    for &size in &[16 * 1024usize, 64 * 1024] {
        let text = ascii(size, 0x0BAD_F00D);
        group.throughput(Throughput::Elements(OPS as u64));
        for &(label, at) in &[("front", 0usize), ("end", size)] {
            group.bench_with_input(BenchmarkId::new(label, size), &text, |b, text| {
                b.iter_batched_ref(
                    || document(text),
                    |stream| {
                        for _ in 0..OPS {
                            let _ = stream.move_cursor_to(0, 0, at);
                            let _ = stream.delete(0);
                        }
                        black_box(stream.raw_length())
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

/// Delete a 256-element range in the middle, shifting the tail once per
/// call.
fn bench_delete_range(c: &mut Criterion) {
    const CHUNK: usize = 256;
    let mut group = c.benchmark_group("engine/delete_range");
    for &size in &[16 * 1024usize, 64 * 1024] {
        let text = ascii(size, 0x1122_3344);
        group.throughput(Throughput::Elements(CHUNK as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter_batched_ref(
                || document(text),
                |stream| {
                    let mut at = size / 2;
                    for _ in 0..16 {
                        let _ = stream.delete_range(at, at + CHUNK);
                        at = at.saturating_sub(CHUNK);
                    }
                    black_box(stream.raw_length())
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Replace a middle range, and overwrite committed elements in place.
fn bench_replace(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/replace");
    for &size in &[16 * 1024usize, 64 * 1024] {
        let text = ascii(size, 0x5566_7788);
        let replacement = ascii(256, 0x99AA_BBCC);
        group.throughput(Throughput::Elements(replacement.len() as u64));
        group.bench_with_input(BenchmarkId::new("range", size), &text, |b, text| {
            b.iter_batched_ref(
                || document(text),
                |stream| {
                    let mid = size / 2;
                    let _ = stream.replace_range(mid, mid + 256, &replacement);
                    black_box(stream.raw_length())
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("overwrite", size), &text, |b, text| {
            b.iter_batched_ref(
                || document(text),
                |stream| {
                    let mid = size / 2;
                    let _ = stream.overwrite_slice(mid, &replacement);
                    black_box(stream.raw_length())
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// The isolated gap commit: stage a full window, then one `flush_gap`
/// performs the single tail `memmove` plus the staged `memcpy`.
fn bench_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/flush");
    for &size in &[16 * 1024usize, 64 * 1024] {
        let text = ascii(size, 0xABCD_EF01);
        group.throughput(Throughput::Elements(GAP_SIZE as u64));
        group.bench_with_input(BenchmarkId::new("gap_middle", size), &text, |b, text| {
            b.iter_batched_ref(
                || {
                    let mut stream = document(text);
                    let _ = stream.move_cursor_to(0, 0, size / 2);
                    for k in 0..GAP_SIZE {
                        let _ = stream.put(0, plain(b'a' as u32 + (k % 26) as u32));
                    }
                    stream
                },
                |stream| black_box(stream.flush_gap(0)),
                BatchSize::SmallInput,
            );
        });
        // Nothing staged: measures the descriptor scan overhead alone.
        group.bench_with_input(BenchmarkId::new("empty_all", size), &text, |b, text| {
            b.iter_batched_ref(
                || document(text),
                |stream| black_box(stream.flush_all()),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Several private gaps staged and committed in one `flush_all`.
///
/// The engine holds at most `MAX_GAPS` (4) private gaps, so the cursor
/// count stays within that bound.
fn bench_multicursor(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/multicursor_private");
    for &cursors in &[1usize, 2, 4] {
        let chars = cursors * 16;
        group.throughput(Throughput::Elements(chars as u64));
        group.bench_with_input(BenchmarkId::from_parameter(cursors), &cursors, |b, &cursors| {
            b.iter_batched_ref(
                || {
                    let mut stream = new_stream(4096);
                    for _ in 1..cursors {
                        let _ = stream.append_cursor(Cursor::new());
                    }
                    stream
                },
                |stream| {
                    for id in 0..cursors {
                        for k in 0..16u32 {
                            let _ = stream.put(id, plain(b'a' as u32 + k));
                        }
                    }
                    black_box(stream.flush_all())
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Random-access reads: `char_at` and `raw_slice`.
fn bench_queries(c: &mut Criterion) {
    const SIZE: usize = 1 << 20;
    let text = ascii(SIZE, 0xC0DE_CAFE);
    let stream = document(&text);
    let length = stream.raw_length();
    let mut group = c.benchmark_group("query");
    group.throughput(Throughput::Elements(SIZE as u64));

    group.bench_function("char_at_random", |b| {
        let mut rng = Rng::new(0x0F0F_0F0F);
        b.iter(|| black_box(stream.char_at(rng.below(length))));
    });
    group.bench_function("raw_slice_64", |b| {
        let mut rng = Rng::new(0x1E1E_1E1E);
        b.iter(|| {
            let start = rng.below(length - 64);
            black_box(stream.raw_slice(start, start + 64))
        });
    });
    group.finish();
}

/// Row/column navigation over a 5000-line, 40-column buffer: each jump
/// rescans the committed layout from the current caret.
fn bench_navigation(c: &mut Criterion) {
    const ROWS: usize = 5_000;
    const COLS: usize = 40;
    let mut text = Vec::with_capacity(ROWS * (COLS + 1));
    for row in 0..ROWS {
        for col in 0..COLS {
            text.push(plain(b'a' as u32 + ((row + col) % 26) as u32));
        }
        text.push(plain(b'\n' as u32));
    }
    let mut stream = document(&text);
    let mut group = c.benchmark_group("nav");
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_function("move_to_random_row", |b| {
        let mut rng = Rng::new(0x7E57_0000);
        b.iter(|| {
            let row = rng.below(ROWS);
            let col = rng.below(COLS);
            black_box(stream.move_cursor_to(0, row, col))
        });
    });
    group.bench_function("request_region_load_random_row", |b| {
        let mut rng = Rng::new(0x7E57_0C0D);
        b.iter(|| {
            let row = rng.below(ROWS);
            let loaded = stream.request_region_load(0, row);
            black_box(loaded)
        });
    });
    group.bench_function("preloaded_move_to_random_row", |b| {
        let mut rng = Rng::new(0x7E57_0A0B);
        b.iter(|| {
            let row = rng.below(ROWS);
            let col = rng.below(COLS);
            // Prime the stack ring, then resolve the row from it.
            let loaded = stream.request_region_load(0, row);
            let _ = black_box(loaded);
            black_box(stream.move_cursor_to(0, row, col))
        });
    });
    group.bench_function("line_width_at", |b| {
        let mut rng = Rng::new(0x7E57_1111);
        b.iter(|| black_box(stream.line_width_at(rng.below(stream.raw_length()))));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_append,
    bench_insert,
    bench_stage,
    bench_bulk,
    bench_delete,
    bench_delete_range,
    bench_replace,
    bench_flush,
    bench_multicursor,
    bench_queries,
    bench_navigation,
);
criterion_main!(benches);
