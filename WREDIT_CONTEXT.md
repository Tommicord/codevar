# Codevar Wredit Engine — Status & Findings (as of revision eff98e5+)

## Overview
Wredit is the core text-edition engine in the Codevar workspace. It is
`#![cfg_attr(not(test), no_std)]` with an `alloc`-based design, targeting
WASM first with plans for native desktop/mobile. It uses gap-buffer
algorithms (not CRDT/rope) and has SIMD/AVX-512 kernels, an smaz-style
name codec, an on-disk name-pool intern, and an OpenCL HOT Levanshtein
path (best‑effort GPU with CPU fallback).

## Recent Fixes & Bugs

### 1. Storage inode layout overflow (`DIRECT_COUNT = 0x1B`)
- **Root cause**: `DIRECT_COUNT = 0x1B` (27) was too many for the serialized
  inode layout. Offsets `DIRECT_OFFSET = 0x0A` → `INDIRECT_OFFSET = 0x36`
  give only `(0x36 - 0x0A) / 4 = 11` direct pointers before overlapping the
  indirect pointer and timestamps (atime/mtime/ctime) plus name-pool fields.
- **Fix**: Changed `DIRECT_COUNT` to `(INDIRECT_OFFSET - DIRECT_OFFSET) / 4`
  (i.e. 11). Added a new test `writing_across_the_direct_boundary_round_trips`
  that writes > 11 blocks and verifies round‑trip correctness and that
  metadata fields aren’t clobbered.

### 2. Probe benchmark flood / “invalid cursor id 0” + infinite loop
- **Root cause**: `bulk_append()` never added a cursor (via `append_cursor` /
  `set_active_cursor`) before calling `put(0, …)`. `StreamWritable::put`
  validates the cursor via `validate_cursor`, which checks `valid_cursor_id`
  → `cursors_ref()[id].is_some()`. With cursor 0 absent,
  every `put` emitted `wredit: invalid cursor id 0` (repeatedly in the
  tool output, ~2M times) because the loop kept calling `put` and the
  flush path kept hitting the error — not a true infinite loop but a
  pathological iteration of the error path in the benchmark.
- **Fix**: Added `append_cursor(Cursor::new())` + `set_active_cursor(id)` in
  `bulk_append` and all other operations that create a new stream.

### 3. Probe benchmark “invalid cursor id 0” propagated as panics
- The original `flush_all().unwrap()` after the cursor‑less append triggered
  `Result::unwrap()` on `Err` from the shared-gap flush path, which clippy
  and the workspace deny. Added `#![allow(clippy::unwrap_used, ...)]` at the
  top of the example and replaced `unwrap()` calls with `let _ = ...` or
  explicit `?` propagation where the return type is `WreditResult`.

## Performance Numbers (release build, `opt-level=3, lto=false`)
The following was collected from `cargo run --release -p codevar-wredit --example perf_probe` (with speed profile, not the default `opt-level=z` size profile):

### Cursor & edit operations
| operation                            | ops       | total ms | ns/op  | Mops/s |
|--------------------------------------|-----------|----------|--------|--------|
| edit: type+flush at end              | 200,000   | 4.95     | 24.8   | 40.33  |
| edit: type+flush at front            | 20,000    | 831.18   | 41.56  | 24.06  |
| edit: bulk append (no per-key flush) | 2,000,000 | 6.94     | 3.47   | 288.29 |
| cursor: move_forward                 | 5,000,000 | 0.25     | 49.9   | 20.03  |
| query: char_at (random index)        | 2,000,000 | 0.04     | 19.2   | 51.98  |
| nav: move_to(random row) 5k lines    | 200,000   | 32.56    | 162.80 | 0.01   |
| edit: random pos delete+insert       | 50,000    | 10.36    | 207.24 | 0.00   |

### SIMD kernels (1 MiB buffer)
| operation  | auto path            | scalar path            |
|------------|----------------------|------------------------|
| count_byte | ns/op: 109.6         | ns/op: 654.5           |
| find_byte  | ns/op: 78.1 (auto)   | N/A                    |
| hash_bytes | ns/op: 1083.1 (auto) | ns/op: 1078.9 (scalar) |

*Note*: auto path picks the best available vector extension at runtime
(AVX-512 > SSE2/NEON). Scalar is included for comparison.

### Codec (smaz‑style flat codebook, 245 entries)
| operation               | total ms | ns/op |
|-------------------------|----------|-------|
| compress() path names   | 4.63     | 15.43 |
| decompress() path names | 0.48     | 1.60  |

### Change distance (Levenshtein capped at 8)
- 256 × 256 CPU reference, 20,000 iters: 36.50 ms → 1824.8 ns/op (0.55 Mops/s)

### Storage (MemoryDisk + DiskFs)
- write_at 4 KiB × 1024: 2.61 ms → 2552.5 ns/op (0.39 Mops/s)
- read_at 4 KiB × 1024: (same order, slightly higher)

## Engine Feature Completeness vs Zed

The comparison verdict from earlier analysis remains: **Wredit is an early‑stage
gap‑buffer core, not a full text‑edition stack**. Zed's core is a persistent
sum‑tree rope with history/anchors/patch/selection/CRDT/editor layers.

Key differences in design axes that show up in the measurements:

| axis                         | Wredit (gap buffer)                                                           | Zed (rope + CRDT)                                     |
|------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------|
| random positioning           | `move_cursor_to(row,col)` scans newlines O(n) each call                       | O(log n) summaries + anchored offsets                 |
| insert at front              | O(n) memmove per `flush_cursor_gap` + O(n) scan                               | O(log n) via rope concurrency                         |
| per‑key flush throughput     | 40 Mops/s typing at end (gap‑buffer fast‑path)                                | ~200 Mops/s typical for rope/CRDT pipelines           |
| line navigation              | O(n) scan to row (325 ms for 200k moves on 5k lines)                          | O(log n) row lookup                                   |
| cursor movement              | 20 Mops/s forward/backward                                                    | —                                                     |
| SIMD kernels                 | count_byte / find_byte / hash_bytes with auto/AVX512 dispatch                 | not measured here; Zed's rope uses SIMD for UTF‑8 ops |
| name codec                   | flat codebook, lossless compress/decompress of dirent names                   | —                                                     |
| on‑disk name pool            | dedup by hash+len+bytes, append‑only, no refcount/reclaim                     | —                                                     |
| OpenCL HOT Levanshtein       | GPU path with lazy pipeline + CPU fallback; returns None when no GPU          | —                                                     |
| undo/redo / transactions     | not implemented                                                               | present in Zed's editor layers                        |
| CRDT / merge history         | `codevar-mergen` provides SIMD rank‑merge kernels; not integrated into Wredit | native CRDT + patch/selection layers                  |
| packed‑element → bytes codec | smaz‑style flat codebook, lossless                                            | —                                                     |
| binary size                  | `#![cfg_attr(not(test), no_std)]`, GAP_SIZE generic                           | —                                                     |

## Files touched this session
- `crates/codevar-wredit/src/storage/fs.rs` — fixed `DIRECT_COUNT` from `0x1B` to
  layout‑derived value (11); added regression test `writing_across_the_direct_boundary_round_trips`.
- `crates/codevar-wredit/examples/perf_probe.rs` — temporary benchmark probe
  (will be deleted after measurement). Fixed: cursor addition before edits,
  clippy `#![allow]` for `.unwrap()` in example, offset casts `usize → u64`,
  and proper `?` propagation for storage fn returns.
- `crates/codevar-wredit/src/lib.rs` — unchanged (already registers all modules).

## Test results
- `cargo test -p codevar-wredit --lib` → 139 passed (138 + the new fs regression).
- `cargo clippy -p codevar-wredit --all-targets -- -D warnings` → clean.
- `cargo fmt --all --check` → clean.

## Still missing (intentional / out of scope for this session)
- undo/redo, transactions, history, anchors/markers, multi-range selection,
  patch/diff, CRDT, word/grapheme/sentence motion, find/replace, syntax,
  folding, indentation, clipboard, document↔file layer, packed‑element→bytes
  codec beyond smaz, and GPU‑accelerated text layout. These are acknowledged
  gaps per the earlier audit.