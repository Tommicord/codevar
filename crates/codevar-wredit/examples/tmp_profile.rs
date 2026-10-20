// Temporary profiling harness. Deleted after collecting profiles.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use std::hint::black_box;

use codevar_wredit::compress::{compress, decompress};
use codevar_wredit::edit::{
    Cursor, StreamWritable, T_COMMON, TextEditableBufferQuery, TextEditableCursor, TextEditableDelete,
    TextEditablePut, Writable, make_packed,
};
use codevar_wredit::storage::{DiskFs, MemoryDisk};

type Stream = StreamWritable<u32, u32, 1024>;

fn plain(c: u8) -> u32 {
    make_packed(u32::from(c), T_COMMON)
}

fn new_stream() -> Stream {
    let mut stream = Stream::new();
    let id = stream.append_cursor(Cursor::new()).unwrap();
    stream.set_active_cursor(id);
    stream
}

fn build_text(stream: &mut Stream, len: usize) {
    for index in 0..len {
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
    }
    stream.flush_all();
}

fn build_lines(stream: &mut Stream, rows: usize, cols: usize) {
    for row in 0..rows {
        for column in 0..cols {
            stream
                .put(0, plain(b'a' + ((row + column) % 26) as u8))
                .unwrap();
        }
        stream.put(0, plain(b'\n')).unwrap();
    }
    stream.flush_all();
    let _ = stream.cursor_row(0).unwrap();
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

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn bench_type_end(n: usize) {
    let mut stream = new_stream();
    for index in 0..n {
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
        stream.flush_all();
        black_box(stream.cursor_row(0).unwrap());
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn bench_type_front(n: usize) {
    let mut stream = new_stream();
    for index in 0..n {
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
        stream.flush_all();
        let _ = stream.move_cursor_backward(0, 1);
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn bench_forward(n: usize) {
    let mut stream = new_stream();
    build_text(&mut stream, 1_000_000);
    let _ = stream.move_cursor_to(0, 0, 0);
    measure_forward(&mut stream, n);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn measure_forward(stream: &mut Stream, n: usize) {
    for _ in 0..n {
        black_box(stream.move_cursor_forward(0, 1).unwrap());
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn bench_char_at(n: usize) {
    let mut stream = new_stream();
    build_text(&mut stream, 1_000_000);
    measure_char_at(&mut stream, n);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn measure_char_at(stream: &mut Stream, n: usize) {
    let mut rng = Rng(0x1234_5678);
    for _ in 0..n {
        black_box(stream.char_at(rng.below(1_000_000)).unwrap_or(0));
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn bench_nav(n: usize) {
    let mut stream = Stream::with_capacity(5_000 * 41 + 4096);
    let id = stream.append_cursor(Cursor::new()).unwrap();
    stream.set_active_cursor(id);
    build_lines(&mut stream, 5_000, 40);
    measure_nav(&mut stream, n);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn measure_nav(stream: &mut Stream, n: usize) {
    let mut rng = Rng(0x9e37_79b9);
    for _ in 0..n {
        let row = rng.below(5_000);
        black_box(stream.move_cursor_to(0, row, 20).unwrap());
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn bench_nav_big(n: usize) {
    let mut stream = Stream::with_capacity(500_000 * 41 + 4096);
    let id = stream.append_cursor(Cursor::new()).unwrap();
    stream.set_active_cursor(id);
    for row in 0..500_000usize {
        for column in 0..40 {
            stream
                .put(0, plain(b'a' + ((row + column) % 26) as u8))
                .unwrap();
        }
        stream.put(0, plain(b'\n')).unwrap();
    }
    stream.flush_all();
    let _ = stream.cursor_row(0).unwrap();
    let mut rng = Rng(0x9e37_79b9);
    for _ in 0..n {
        let row = rng.below(500_000);
        black_box(stream.move_cursor_to(0, row, 20).unwrap());
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn bench_type_middle(n: usize) {
    let mut stream = Stream::with_capacity(16_000 * 41 + 4096);
    let id = stream.append_cursor(Cursor::new()).unwrap();
    stream.set_active_cursor(id);
    for row in 0..16_000usize {
        for column in 0..40 {
            stream
                .put(0, plain(b'a' + ((row + column) % 26) as u8))
                .unwrap();
        }
        stream.put(0, plain(b'\n')).unwrap();
    }
    stream.flush_all();
    let _ = stream.cursor_row(0).unwrap();
    // Park the caret in the middle of the document: every keystroke
    // then shifts the tail of the sparse index.
    let _ = stream.move_cursor_to(0, 8_000, 0).unwrap();
    for index in 0..n {
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
        stream.flush_all();
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn bench_random_edit(n: usize) {
    let mut stream = new_stream();
    build_text(&mut stream, 200_000);
    measure_random_edit(&mut stream, n);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn measure_random_edit(stream: &mut Stream, n: usize) {
    let mut rng = Rng(0xdead_beef);
    let length = stream.raw_length();
    for _ in 0..n {
        let pos = rng.below(length);
        let _ = stream.move_cursor_to(0, 0, pos);
        let _ = stream.delete(0);
        let _ = stream.put(0, plain(b'x'));
        let _ = stream.flush_all();
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn bench_compress(n: usize) {
    let names: &[&[u8]] = &[
        b"src/main.rs",
        b"crates/codevar-wredit/src/storage/fs.rs",
        b"node_modules",
        b"tests/test_storage.rs",
        b"docs/readme.md",
    ];
    for index in 0..n {
        black_box(compress(names[index % names.len()]));
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn bench_decompress(n: usize) {
    let names: &[&[u8]] = &[
        b"src/main.rs",
        b"crates/codevar-wredit/src/storage/fs.rs",
        b"node_modules",
        b"tests/test_storage.rs",
        b"docs/readme.md",
    ];
    let packed: Vec<Vec<u8>> = names.iter().map(|name| compress(name)).collect();
    for index in 0..n {
        black_box(decompress(&packed[index % packed.len()]).unwrap());
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn bench_storage_write(n: usize) {
    let disk = MemoryDisk::new(8192).unwrap();
    let mut fs = DiskFs::format(disk, 8192, 4096).unwrap();
    let ino = fs.create(2, "bench.bin", 0x8000).unwrap();
    let chunk = [0xA5u8; 4096];
    for index in 0..n {
        let _ = fs
            .write_at(ino, (index as u64) * 4096, &chunk)
            .unwrap();
    }
    fs.flush().unwrap();
}

fn main() {
    let workload = std::env::args().nth(1).unwrap_or_default();
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(20_000);
    match workload.as_str() {
        "type_end" => bench_type_end(n),
        "type_middle" => bench_type_middle(n),
        "type_front" => bench_type_front(n),
        "forward" => bench_forward(n),
        "char_at" => bench_char_at(n),
        "nav" => bench_nav(n),
        "nav_big" => bench_nav_big(n),
        "random_edit" => bench_random_edit(n),
        "compress" => bench_compress(n),
        "decompress" => bench_decompress(n),
        "storage_write" => bench_storage_write(n),
        other => println!("unknown workload {other}"),
    }
}
