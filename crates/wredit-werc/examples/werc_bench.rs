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

//! Measure the codec on a dumped element array.
//!
//! Reads a file of little-endian `u32` elements - what the Wredit engine
//! holds in RAM for a document - codes it, decodes it back and reports the
//! ratio, the model's own budget and the throughput of both directions.
//!
//! ```text
//! cargo run --release -p codevar-wredit-werc --example werc_bench -- \
//!     dump.w32 22 16 11 [blob.bin]
//! ```
//!
//! The positional arguments are `mem_bits`, `sse_bits` and `lr_shift`, so a
//! run is reproducible from the command line alone. With a fourth argument the
//! encoded frame is written there, which is what lets `tools/wezc.py` be
//! diffed against this build byte for byte.

use std::env::args;
use std::fs;
use std::process::exit;
use std::time::Instant;

use codevar_wredit_werc::codec::{decode, encode};
use codevar_wredit_werc::frame::{Header, Params};

fn usage() -> ! {
    eprintln!("usage: werc_bench <elements.w32> [mem_bits] [sse_bits] [lr_shift] [blob out]");
    exit(2);
}

fn main() {
    let argv: Vec<String> = args().skip(1).collect();
    if argv.is_empty() {
        usage();
    }
    let path = &argv[0];
    let mem_bits = argv
        .get(1)
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(22);
    let sse_bits = argv
        .get(2)
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(16);
    let lr_shift = argv
        .get(3)
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(11);
    let params = Params {
        mem_bits,
        sse_bits,
        lr_shift,
    };

    let raw = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("cannot read {path}: {err}");
            exit(1);
        }
    };
    if raw.len() % 4 != 0 {
        eprintln!("{}: {} bytes is not a whole element array", path, raw.len());
        exit(1);
    }
    let elements: Vec<u32> = raw
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();

    let started = Instant::now();
    let blob = match encode(&params, &[], &elements) {
        Ok(blob) => blob,
        Err(err) => {
            eprintln!("encode failed: {}", err.label());
            exit(1);
        }
    };
    let encoded_in = started.elapsed();

    let mut back = vec![0u32; elements.len()];
    let started = Instant::now();
    let recovered = match decode(&blob, &[], &mut back) {
        Ok(count) => count,
        Err(err) => {
            eprintln!("decode failed: {}", err.label());
            exit(1);
        }
    };
    let decoded_in = started.elapsed();
    if recovered != elements.len() || back != elements {
        eprintln!("round trip mismatch: {recovered} of {} elements", elements.len());
        exit(1);
    }

    let header = Header::parse(&blob).unwrap_or(Header::new(params, 0, 0, 0));
    let plane_bytes = header.body_len().unwrap_or(0);
    let model = params
        .model()
        .map(|model| model.memory_bytes())
        .unwrap_or(0);
    let coded = blob.len() - codevar_wredit_werc::frame::HEADER_SIZE;
    let per_element = 8.0 * coded as f64 / elements.len().max(1) as f64;
    println!(
        "{path}: {} elements, planes {plane_bytes} B -> {coded} B coded, {per_element:.4} bpc",
        elements.len()
    );
    println!(
        "model {model} B, encode {:.1} MB/s, decode {:.1} MB/s",
        plane_bytes as f64 / 1e6 / encoded_in.as_secs_f64().max(1e-9),
        plane_bytes as f64 / 1e6 / decoded_in.as_secs_f64().max(1e-9)
    );
    if let Some(out) = argv.get(4)
        && let Err(err) = fs::write(out, &blob)
    {
        eprintln!("cannot write {out}: {err}");
        exit(1);
    }
}
