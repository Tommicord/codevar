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

//! Groups: `(ab)+` matches one or more repetitions of the group.
//!
//! Run with: cargo run --example groups -p codevar-negex

use codevar_negex::{ExecError, Negex};

fn main() {
    let engine = Negex::new();
    let pattern = "(ab)+";

    let inputs = ["ab", "abab", "ababab", "abc", "aba", "ababxab", ""];

    println!("pattern: {pattern:?}\n");
    println!("{:<14} {:<8} {:<6} {:<6}", "input", "match", "start", "end");
    println!("{}", "-".repeat(40));

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
                println!(
                    "{input:<14} {:<8} {:<6} {:<6} {}",
                    result.is_match, result.start, result.end, span
                );
            }
            Err(ExecError::NoTrampoline) => {
                println!("{input:<14} exec skipped: no native trampoline back end");
            }
            Err(ExecError::NoInterpreter) => {
                println!("{input:<14} exec skipped: no interpreter available");
            }
        }
    }
}
