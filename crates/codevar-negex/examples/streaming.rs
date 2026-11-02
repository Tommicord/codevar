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

//! Streaming / chunked input processing (embedded-friendly).
//!
//! Compile the pattern once, then feed the input in fixed-size chunks.
//! Each chunk is matched independently and reported with its absolute
//! offset. This pattern needs no per-chunk allocation and a small fixed
//! buffer, so it fits embedded use cases.
//!
//! Run with: cargo run --example streaming -p codevar-negex

use codevar_negex::{ExecError, Negex};

const CHUNK: usize = 4;

fn main() {
    let engine = Negex::new();
    let pattern = "a+";
    let input = b"aaabaaacaaa";

    println!("pattern: {pattern:?}");
    println!("input:   {}", String::from_utf8_lossy(input));
    println!("chunk size: {CHUNK} bytes\n");

    let compiled = match engine.compile(pattern) {
        Ok(compiled) => compiled,
        Err(error) => {
            println!("compile failed: {error}");
            return;
        }
    };

    let mut offset = 0;
    let mut chunks = 0;
    let mut matches = 0;

    for chunk in input.chunks(CHUNK) {
        chunks += 1;
        match compiled.exec(chunk) {
            Ok(result) => {
                if result.is_match {
                    matches += 1;
                    let span = &chunk[result.start..result.end];
                    println!(
                        "chunk {chunks} (offset {offset}): match [{}, {}) {}",
                        result.start,
                        result.end,
                        String::from_utf8_lossy(span)
                    );
                } else {
                    println!("chunk {chunks} (offset {offset}): no match");
                }
            }
            Err(ExecError::NoTrampoline) => {
                println!("chunk {chunks} (offset {offset}): exec skipped - no native trampoline back end");
            }
            Err(ExecError::NoInterpreter) => {
                println!("chunk {chunks} (offset {offset}): exec skipped - no interpreter available");
            }
        }
        offset += chunk.len();
    }

    println!();
    println!("chunks processed: {chunks}");
    println!("chunks with a match: {matches}");
    println!(
        "\nThis is the embedded-friendly structure: one compiled regex, a fixed\n\
              chunk buffer, and match results reported per chunk with absolute offsets."
    );
}
