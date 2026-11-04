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

//! Resistance to catastrophic backtracking (ReDoS).
//!
//! The pattern `(a+)+b` against a long run of `a`'s with no trailing `b`
//! makes a naive backtracking engine explore exponentially many ways to
//! partition the `a`'s before failing. Negex compiles to a DFA, so the
//! match runs in time linear in the input length.
//!
//! Run with: cargo run --example re_dos_resistance -p codevar-negex

use codevar_negex::{ExecError, Negex};
use std::time::Instant;

fn main() {
    let engine = Negex::new();
    let pattern = "(a+)+b";

    let lengths = [10, 100, 1_000, 10_000];

    println!("pattern: {pattern:?}\n");
    println!(
        "{:<10} {:<12} {:<12} {:<10}",
        "input_len", "elapsed_ms", "match", "note"
    );
    println!("{}", "-".repeat(50));

    let compiled = match engine.compile(pattern) {
        Ok(compiled) => compiled,
        Err(error) => {
            println!("compile failed: {error}");
            return;
        }
    };

    for &len in &lengths {
        // A long run of 'a' with no trailing 'b': guaranteed non-match,
        // but a backtracking engine would enumerate 2^len partitions.
        let input = vec![b'a'; len];

        let start = Instant::now();
        let result = compiled.exec(&input);
        let elapsed = start.elapsed();

        match result {
            Ok(m) => {
                println!(
                    "{len:<10} {:<12.3} {:<12} {:<10}",
                    elapsed.as_secs_f64() * 1_000.0,
                    m.is_match,
                    "linear (DFA)"
                );
            }
            Err(ExecError::NoTrampoline) => {
                println!(
                    "{len:<10} {:<12.3} {:<12} {:<10}",
                    elapsed.as_secs_f64() * 1_000.0,
                    "skipped",
                    "no native trampoline back end"
                );
            }
            Err(ExecError::NoInterpreter) => {
                println!(
                    "{len:<10} {:<12.3} {:<12} {:<10}",
                    elapsed.as_secs_f64() * 1_000.0,
                    "skipped",
                    "no interpreter available"
                );
            }
        }
    }

    println!(
        "\nNote: elapsed stays roughly proportional to input_len. A naive\n\
              backtracking engine would take time exponential in `len` on this\n\
              input and effectively hang long before len = 10_000."
    );
}
