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

//! Probe: where does the time in `DiskFs::read_at` and `write_at` go?
//!
//! `storage/read_at_4k` and `storage/write_at_4k` each move 256 KiB in
//! 64 whole-block transfers. This probe times the steps such a transfer
//! is made of — the raw device copy, the inode read, the block-table
//! walk, and the whole `read_at`/`write_at` call — from the same image,
//! so a cost is attributed to a layer instead of guessed at.
//!
//! Benchmarks are test-like code, not production code; a failed disk
//! operation here is a bug in this file, not a runtime condition.
#![allow(clippy::unwrap_used)]

use std::hint::black_box;
use std::time::Instant;

use codevar_wredit::{BLOCK_SIZE, BlockDevice, DiskFs, MemoryDisk};

/// Blocks in the image: enough for the file plus its metadata.
const IMAGE_BLOCKS: u64 = 4096;

/// Whole-block transfers per timed pass.
const TRANSFERS: u64 = 64;

/// Timed passes; the minimum is the least noise-contaminated estimate.
const PASSES: usize = 7;

/// An empty filesystem with a file inode already created.
fn empty_fs() -> (DiskFs<MemoryDisk>, u32) {
    let disk = MemoryDisk::new(IMAGE_BLOCKS).unwrap();
    let mut fs = DiskFs::format(disk, IMAGE_BLOCKS, 2048).unwrap();
    let inode = fs.create(2, "probe.bin", 0x8000).unwrap();
    (fs, inode)
}

/// A filesystem whose file already holds `TRANSFERS` blocks.
fn filled_fs() -> (DiskFs<MemoryDisk>, u32) {
    let (mut fs, inode) = empty_fs();
    let chunk = [0xA5u8; BLOCK_SIZE];
    for block in 0..TRANSFERS {
        fs.write_at(inode, block * BLOCK_SIZE as u64, &chunk)
            .unwrap();
    }
    fs.flush().unwrap();
    (fs, inode)
}

/// A device pre-filled with `IMAGE_BLOCKS` of payload.
fn filled_disk() -> MemoryDisk {
    let mut disk = MemoryDisk::new(IMAGE_BLOCKS).unwrap();
    let chunk = [0xA5u8; BLOCK_SIZE];
    for block in 0..IMAGE_BLOCKS {
        disk.write_block(block, &chunk).unwrap();
    }
    disk
}

/// Best-of-`PASSES` nanoseconds per pass, with the setup excluded from
/// the timer the same way criterion excludes `iter_batched` input.
fn time<T, S, R>(name: &str, mut setup: S, mut run: R)
where
    S: FnMut() -> T,
    R: FnMut(&mut T),
{
    for _ in 0..2 {
        let mut state = setup();
        run(&mut state);
    }
    let mut best = f64::MAX;
    for _ in 0..PASSES {
        let mut state = setup();
        let start = Instant::now();
        run(&mut state);
        let elapsed = start.elapsed().as_nanos() as f64;
        black_box(state);
        if elapsed < best {
            best = elapsed;
        }
    }
    println!(
        "  {name:<22} {:>11.1} ns/pass {:>11.1} ns/transfer",
        best,
        best / TRANSFERS as f64
    );
}

fn main() {
    println!(
        "{TRANSFERS} whole-block transfers of {} bytes, best of {PASSES} passes:\n",
        BLOCK_SIZE
    );

    println!("raw device, no filesystem:");
    time("read_block", filled_disk, |disk| {
        let mut buf = [0u8; BLOCK_SIZE];
        for block in 0..TRANSFERS {
            disk.read_block(block, &mut buf).unwrap();
        }
        black_box(buf);
    });
    time("write_block", filled_disk, |disk| {
        let chunk = [0x5Au8; BLOCK_SIZE];
        for block in 0..TRANSFERS {
            disk.write_block(block, &chunk).unwrap();
        }
    });

    println!("\nmetadata, no payload transfer:");
    time("read_inode", filled_fs, |(fs, inode)| {
        for _ in 0..TRANSFERS {
            black_box(fs.read_inode(*inode).unwrap());
        }
    });
    time("lookup path", filled_fs, |(fs, _)| {
        for _ in 0..TRANSFERS {
            black_box(fs.lookup("probe.bin", 2).unwrap());
        }
    });
    time("alloc_block", empty_fs, |(fs, _)| {
        for _ in 0..TRANSFERS {
            black_box(fs.alloc_block().unwrap());
        }
    });

    println!("\nwhole transfers through the filesystem:");
    time("read_at", filled_fs, |(fs, inode)| {
        let mut buf = [0u8; BLOCK_SIZE];
        for block in 0..TRANSFERS {
            black_box(
                fs.read_at(*inode, block * BLOCK_SIZE as u64, &mut buf)
                    .unwrap(),
            );
        }
        black_box(buf);
    });
    time("write_at", empty_fs, |(fs, inode)| {
        let chunk = [0xA5u8; BLOCK_SIZE];
        for block in 0..TRANSFERS {
            black_box(
                fs.write_at(*inode, block * BLOCK_SIZE as u64, &chunk)
                    .unwrap(),
            );
        }
    });
    time("write_at + flush", empty_fs, |(fs, inode)| {
        let chunk = [0xA5u8; BLOCK_SIZE];
        for block in 0..TRANSFERS {
            fs.write_at(*inode, block * BLOCK_SIZE as u64, &chunk)
                .unwrap();
        }
        fs.flush().unwrap();
    });
}
