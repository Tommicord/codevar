// Temporary performance probe for the Wredit engine. Not part of the
// crate's public surface; delete after collecting numbers.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use codevar_wredit::change::{ChangeFlags, change_distance};
use codevar_wredit::compress::{compress, decompress};
use codevar_wredit::edit::{
    Cursor, StreamWritable, T_COMMON, TextEditableBufferQuery, TextEditableCursor, TextEditableDelete,
    TextEditablePut, Writable, make_packed,
};
use codevar_wredit::simd;
use codevar_wredit::storage::{DiskFs, MemoryDisk};

type Stream = StreamWritable<u32, u32, 1024>;

fn plain(c: u32) -> u32 {
    make_packed(c, T_COMMON)
}

fn new_stream() -> Stream {
    let mut s = Stream::new();
    let id = s.append_cursor(Cursor::new()).unwrap();
    s.set_active_cursor(id);
    s
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() as usize) % n }
    }
}

fn report(name: &str, ops: u64, d: Duration) {
    let ns = d.as_secs_f64() * 1e9 / ops as f64;
    let mops = ops as f64 / d.as_secs_f64() / 1e6;
    println!(
        "{:<36} {:>10} ops {:>9.2} ms {:>8.1} ns/op {:>8.2} M/s",
        name,
        ops,
        d.as_secs_f64() * 1e3,
        ns,
        mops
    );
}

fn build(stream: &mut Stream, len: usize) {
    for i in 0..len {
        let _ = stream.put(0, plain(b'a' as u32 + (i as u32 % 26)));
    }
    let _ = stream.flush_all();
}

fn typing_append() {
    const N: usize = 200_000;
    let mut s = new_stream();
    let t = Instant::now();
    for i in 0..N {
        let _ = s.put(0, plain(b'a' as u32 + (i as u32 % 26)));
        let _ = s.flush_all();
    }
    report("edit: type+flush at end", N as u64, t.elapsed());
}

fn typing_front() {
    const N: usize = 20_000;
    let mut s = new_stream();
    let t = Instant::now();
    for i in 0..N {
        let _ = s.put(0, plain(b'a' as u32 + (i as u32 % 26)));
        let _ = s.flush_all();
        let _ = s.move_cursor_backward(0, 1);
    }
    report("edit: type+flush at front", N as u64, t.elapsed());
}

fn bulk_append() {
    const N: usize = 2_000_000;
    let mut s = Stream::with_capacity(N + 4096);
    let id = s.append_cursor(Cursor::new()).unwrap();
    s.set_active_cursor(id);
    let t = Instant::now();
    for i in 0..N {
        let _ = s.put(0, plain(b'a' as u32 + (i as u32 % 26)));
    }
    let _ = s.flush_all();
    report("edit: bulk append (no per-key flush)", N as u64, t.elapsed());
}

fn cursor_forward() {
    const N: usize = 5_000_000;
    let mut s = new_stream();
    build(&mut s, 1_000_000);
    let _ = s.move_cursor_to(0, 0, 0);
    let t = Instant::now();
    for _ in 0..N {
        let _ = s.move_cursor_forward(0, 1);
    }
    report("cursor: move_forward", N as u64, t.elapsed());
}

fn char_random_access() {
    const N: usize = 2_000_000;
    let mut s = new_stream();
    build(&mut s, 1_000_000);
    let mut rng = Rng(0x1234_5678);
    let mut sum = 0u64;
    let t = Instant::now();
    for _ in 0..N {
        sum += u64::from(black_box(s.char_at(rng.below(1_000_000)).unwrap_or(0)));
    }
    report("query: char_at (random index)", N as u64, t.elapsed());
    black_box(sum);
}

fn line_navigation() {
    const ROWS: usize = 5_000;
    const COLS: usize = 40;
    const N: usize = 200_000;
    let mut s = new_stream();
    for _ in 0..ROWS {
        for c in 0..COLS {
            let _ = s.put(0, plain(b'a' as u32 + (c as u32 % 26)));
        }
        let _ = s.put(0, plain(b'\n' as u32));
    }
    let _ = s.flush_all();
    let mut rng = Rng(0x9e37_79b9);
    let t = Instant::now();
    for _ in 0..N {
        let row = rng.below(ROWS);
        let _ = s.move_cursor_to(0, row, 20);
    }
    report("nav: move_to(random row) 5k lines", N as u64, t.elapsed());
}

fn random_edits() {
    const N: usize = 50_000;
    let mut s = new_stream();
    build(&mut s, 200_000);
    let mut rng = Rng(0xdead_beef);
    let length = s.raw_length();
    let t = Instant::now();
    for _ in 0..N {
        let pos = rng.below(length);
        // Position by row/col (single line: row 0), then replace a char.
        let _ = s.move_cursor_to(0, 0, pos);
        let _ = s.delete(0);
        let _ = s.put(0, plain(b'x' as u32));
        let _ = s.flush_all();
    }
    report("edit: random pos delete+insert", N as u64, t.elapsed());
}

fn simd_benches() {
    const LEN: usize = 1 << 20;
    let data: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
    let needle = 200u8;

    let iters = 200u64;
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..iters {
        acc += simd::count_byte(black_box(&data), needle);
    }
    report("simd: count_byte 1 MiB (auto/AVX512)", iters, t.elapsed());
    black_box(acc);
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..iters {
        acc += simd::count_byte_scalar(black_box(&data), needle);
    }
    report("simd: count_byte 1 MiB (scalar)", iters, t.elapsed());
    black_box(acc);

    let t = Instant::now();
    for _ in 0..iters {
        black_box(simd::find_byte(black_box(&data), needle));
    }
    report("simd: find_byte 1 MiB (auto)", iters, t.elapsed());

    let t = Instant::now();
    let mut h = 0u64;
    for _ in 0..iters {
        h ^= u64::from(simd::hash_bytes(black_box(&data)));
    }
    report("simd: hash_bytes 1 MiB (auto)", iters, t.elapsed());
    black_box(h);
    let t = Instant::now();
    let mut h = 0u64;
    for _ in 0..iters {
        h ^= u64::from(simd::hash_bytes_scalar(black_box(&data)));
    }
    report("simd: hash_bytes 1 MiB (scalar)", iters, t.elapsed());
    black_box(h);
}

fn codec_benches() {
    const N: usize = 300_000;
    let names: &[&[u8]] = &[
        b"src/main.rs",
        b"crates/codevar-wredit/src/storage/fs.rs",
        b"node_modules",
        b"tests/test_storage.rs",
        b"docs/readme.md",
        b"a_very_long_directory_name_that_should_compress_well",
    ];
    let t = Instant::now();
    let mut bytes = 0usize;
    for i in 0..N {
        let packed = compress(names[i % names.len()]);
        bytes += packed.len();
        black_box(packed);
    }
    report("codec: compress() path names", N as u64, t.elapsed());
    black_box(bytes);

    let packed: Vec<Vec<u8>> = names.iter().map(|n| compress(n)).collect();
    let t = Instant::now();
    let mut bytes = 0usize;
    for i in 0..N {
        let out = decompress(&packed[i % packed.len()]).unwrap();
        bytes += out.len();
        black_box(out);
    }
    report("codec: decompress() path names", N as u64, t.elapsed());
    black_box(bytes);
}

fn change_distance_benches() {
    const N: usize = 20_000;
    let a: Vec<u32> = (0..256u32).map(|i| i % 13).collect();
    let b: Vec<u32> = (0..256u32).map(|i| (i * 5) % 13).collect();
    let t = Instant::now();
    let mut acc = 0u32;
    for _ in 0..N {
        acc += change_distance(black_box(&a), black_box(&b), 8, ChangeFlags::DEFAULT);
    }
    report("change: levenshtein 256x256 cap8 (CPU)", N as u64, t.elapsed());
    black_box(acc);
}

fn storage_benches() {
    const N: usize = 1_024;
    let disk = MemoryDisk::new(8192).unwrap();
    let mut fs = DiskFs::format(disk, 8192, 4096).unwrap();
    let ino = fs.create(2, "bench.bin", 0x8000).unwrap();
    let chunk = [0xA5u8; 4096];
    let t = Instant::now();
    for i in 0..N {
        let _ = fs
            .write_at(ino, (i as u64) * 4096, &chunk)
            .unwrap();
    }
    fs.flush().unwrap();
    report("storage: write_at 4 KiB x1024", N as u64, t.elapsed());

    let mut buf = [0u8; 4096];
    let t = Instant::now();
    for i in 0..N {
        let _ = fs
            .read_at(ino, (i as u64) * 4096, &mut buf)
            .unwrap();
    }
    report("storage: read_at 4 KiB x1024", N as u64, t.elapsed());
    black_box(&buf);
}

fn main() {
    println!("Wredit engine performance probe (release build)\n");
    println!("CPU target features: {}", std::env::consts::ARCH);
    typing_append();
    typing_front();
    bulk_append();
    cursor_forward();
    char_random_access();
    line_navigation();
    random_edits();
    println!();
    simd_benches();
    println!();
    codec_benches();
    change_distance_benches();
    println!();
    storage_benches();
}
