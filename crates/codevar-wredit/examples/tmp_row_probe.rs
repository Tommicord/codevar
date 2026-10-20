// Temporary probe: row queries after edits. Deleted after collection.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use std::hint::black_box;
use std::time::Instant;

use codevar_wredit::edit::{
    Cursor, StreamWritable, T_COMMON, TextEditableCursor, TextEditableDelete, TextEditablePut, Writable,
    make_packed,
};

type Stream = StreamWritable<u32, u32, 1024>;

fn plain(c: u8) -> u32 {
    make_packed(u32::from(c), T_COMMON)
}

fn build() -> Stream {
    const ROWS: usize = 5_000;
    const COLS: usize = 40;
    let mut stream = Stream::with_capacity(ROWS * (COLS + 1) + 4096);
    let id = stream.append_cursor(Cursor::new()).unwrap();
    stream.set_active_cursor(id);
    for row in 0..ROWS {
        for column in 0..COLS {
            stream
                .put(0, plain(b'a' + ((row + column) % 26) as u8))
                .unwrap();
        }
        stream.put(0, plain(b'\n')).unwrap();
    }
    stream.flush_all();
    let _ = stream.cursor_row(0).unwrap();
    stream
}

fn main() {
    const N: usize = 200_000;

    // Typing at the end, no row query: the flush-only baseline.
    let mut stream = build();
    let start = Instant::now();
    for index in 0..N {
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
        stream.flush_all();
    }
    println!("put+flush at end:        {:?}", start.elapsed() / N as u32);

    // Typing at the end followed by a row query every keystroke.
    let mut stream = build();
    let start = Instant::now();
    for index in 0..N {
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
        stream.flush_all();
        black_box(stream.cursor_row(0).unwrap());
    }
    println!("put+flush+row at end:    {:?}", start.elapsed() / N as u32);

    // Same, but a newline is inserted and removed each cycle so the
    // sparse index is invalidated: this is the pre-shift behaviour.
    let mut stream = build();
    let start = Instant::now();
    let mut t_nl = core::time::Duration::ZERO;
    let mut t_bs = core::time::Duration::ZERO;
    let mut t_ch = core::time::Duration::ZERO;
    let mut t_row = core::time::Duration::ZERO;
    for index in 0..N {
        let phase = Instant::now();
        stream.put(0, plain(b'\n')).unwrap();
        stream.flush_all();
        t_nl += phase.elapsed();
        let phase = Instant::now();
        stream.backspace(0).unwrap();
        t_bs += phase.elapsed();
        let phase = Instant::now();
        stream
            .put(0, plain(b'a' + (index % 26) as u8))
            .unwrap();
        stream.flush_all();
        t_ch += phase.elapsed();
        let phase = Instant::now();
        black_box(stream.cursor_row(0).unwrap());
        t_row += phase.elapsed();
    }
    println!(
        "newline churn breakdown: nl={:?} bs={:?} ch={:?} row={:?}",
        t_nl / N as u32,
        t_bs / N as u32,
        t_ch / N as u32,
        t_row / N as u32
    );
    println!("newline churn + row:     {:?}", start.elapsed() / N as u32);

    // Warm row query only (caret stays put, ring is local).
    let stream = build();
    let start = Instant::now();
    for _ in 0..N {
        black_box(stream.cursor_row(0).unwrap());
    }
    println!("warm row query:          {:?}", start.elapsed() / N as u32);
}
