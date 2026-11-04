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

//! Alternation: `cat|dog` matches either alternative.
//!
//! Run with: cargo run --example alternation -p codevar-negex

use codevar_negex::{ExecError, Negex};

fn main() {
    let engine = Negex::new();
    let pattern = "cat|dog";

    let inputs = [
        "the cat sat",
        "a dog ran",
        "cats and dogs",
        "neither here nor there",
    ];

    println!("pattern: {pattern:?}\n");
    println!("{:<10} {:<28} {:<10}", "input", "match", "span");
    println!("{}", "-".repeat(50));

    let compiled = match engine.compile(pattern) {
        Ok(compiled) => compiled,
        Err(error) => {
            println!("compile failed: {error}");
            return;
        }
    };

    for input in inputs {
        match compiled.exec(input.as_bytes()) {
            Ok(result) => {
                let span = if result.is_match {
                    format!(
                        "[{}, {}) {}",
                        result.start,
                        result.end,
                        String::from_utf8_lossy(&input.as_bytes()[result.start..result.end])
                    )
                } else {
                    "-".to_string()
                };
                println!("{:<10} {:<28} {}", input, result.is_match, span);
            }
            Err(ExecError::NoTrampoline) => {
                println!("{input:<10} exec skipped: no native trampoline back end");
            }
            Err(ExecError::NoInterpreter) => {
                println!("{input:<10} exec skipped: no interpreter available");
            }
        }
    }
}
