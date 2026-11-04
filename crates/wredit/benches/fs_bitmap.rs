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

//! Probe: is the word-wise [`Bitmap`] scan faster than the byte-wise
//! one it replaced?
//!
//! [`Bitmap::find_free`] runs on every block and inode allocation, and
//! [`Bitmap::count_used`] on every `DiskFull` report, so both walk the
//! allocated prefix of a 4 KiB bitmap — 32 768 addresses. The previous
//! implementation advanced one byte per iteration; the current one
//! advances eight. This probe measures both against the same bitmaps in
//! one process, because end-to-end `storage/*` timings on a loaded
//! machine are dominated by whatever else is compiling.
//!
//! Benchmarks are test-like code, not production code; a failed disk
//! operation here is a bug in this file, not a runtime condition.
#![allow(clippy::unwrap_used)]

use std::hint::black_box;

use codevar_wredit::fs_bitmap::Bitmap;

/// Addresses the bitmaps cover, matching the storage default
/// (`BLOCK_SIZE * 8`).
const BITMAP_LIMIT: u32 = 4096 * 8;

/// Number of timed repetitions per pass.
const REPS: usize = 20_000;

/// How many timed passes to take the minimum over.
const PASSES: usize = 7;

/// A bitmap whose lowest `used` addresses are marked used.
fn prefix(used: u32) -> Bitmap {
    let mut bitmap = Bitmap::new();
    for index in 0..used {
        bitmap.set_used(index);
    }
    bitmap
}

/// The byte-wise `find_free` this probe compares against.
///
/// Verbatim copy of the implementation that the word-wise scan
/// replaced, kept here so the comparison is against real code.
fn find_free_bytewise(bitmap: &Bitmap, limit: u32) -> Option<u32> {
    let bytes = bitmap.as_bytes();
    let whole = (limit / 8) as usize;
    for (position, &byte) in bytes.iter().enumerate().take(whole) {
        if byte != 0xFF {
            let bit = (!byte).trailing_zeros();
            let index = (position as u32) * 8 + bit;
            if index < limit {
                return Some(index);
            }
            return None;
        }
    }
    let mut index = whole as u32 * 8;
    while index < limit && bitmap.is_used(index) {
        index += 1;
    }
    (index < limit).then_some(index)
}

/// The byte-wise `count_used` this probe compares against.
fn count_used_bytewise(bitmap: &Bitmap, limit: u32) -> u32 {
    let bytes = bitmap.as_bytes();
    let whole = (limit / 8) as usize;
    let mut count: u32 = 0;
    for &byte in &bytes[..whole] {
        count = count.saturating_add(byte.count_ones());
    }
    let mut index = whole as u32 * 8;
    while index < limit {
        if bitmap.is_used(index) {
            count += 1;
        }
        index += 1;
    }
    count
}

/// Best-of-`PASSES` nanoseconds per call, the least noise-contaminated
/// estimate available on a shared machine.
fn time(name: &str, mut call: impl FnMut() -> u64) -> f64 {
    for _ in 0..200 {
        black_box(call());
    }
    let mut best = f64::MAX;
    for _ in 0..PASSES {
        let start = std::time::Instant::now();
        for _ in 0..REPS {
            black_box(call());
        }
        let elapsed = start.elapsed().as_nanos() as f64 / REPS as f64;
        if elapsed < best {
            best = elapsed;
        }
    }
    println!("  {name:<24} {best:>9.1} ns/call");
    best
}

/// Checks that both scans agree, so a timing gain cannot be bought with
/// a different answer.
fn equivalence() {
    let mut state = 0x243F_6A88_85A3_08D3u64;
    for case in 0..64 {
        let mut bitmap = Bitmap::new();
        for index in 0..BITMAP_LIMIT {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 0xFF >= case.min(200) {
                bitmap.set_used(index);
            }
        }
        let limit = BITMAP_LIMIT - (case % 7) as u32 * 13;
        assert_eq!(
            bitmap.find_free(limit),
            find_free_bytewise(&bitmap, limit),
            "find_free disagrees at case {case}"
        );
        assert_eq!(
            bitmap.count_used(limit),
            count_used_bytewise(&bitmap, limit),
            "count_used disagrees at case {case}"
        );
    }
}

fn main() {
    equivalence();
    println!(
        "bitmap scans over a {}-address limit, best of {PASSES} x {REPS} calls:\n",
        BITMAP_LIMIT
    );
    for &used in &[8u32, 1_024, 8_192, 32_768] {
        let bitmap = black_box(prefix(used));
        println!("allocated prefix: {used} addresses");
        let word = time("find_free/word", || {
            bitmap.find_free(BITMAP_LIMIT).unwrap_or(u32::MAX) as u64
        });
        let byte = time("find_free/byte", || {
            find_free_bytewise(&bitmap, BITMAP_LIMIT).unwrap_or(u32::MAX) as u64
        });
        println!("  -> find_free gain: {:>6.1}%", (byte - word) * 100.0 / byte);
        let word = time("count_used/word", || bitmap.count_used(BITMAP_LIMIT) as u64);
        let byte = time("count_used/byte", || {
            count_used_bytewise(&bitmap, BITMAP_LIMIT) as u64
        });
        println!("  -> count_used gain: {:>6.1}%", (byte - word) * 100.0 / byte);
        println!();
    }
}
