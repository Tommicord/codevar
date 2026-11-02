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

//! Quantifiers: `a*`, `a+`, `a?`, `a{2,4}` against several inputs.
//!
//! Run with: cargo run --example quantifiers -p codevar-negex

use codevar_negex::{ExecError, Negex};

fn main() {
    let engine = Negex::new();

    let cases: &[(&str, &str)] = &[
        ("a*", ""),
        ("a*", "aaa"),
        ("a*", "b"),
        ("a*", "baaab"),
        ("a+", ""),
        ("a+", "aaa"),
        ("a+", "baaab"),
        ("a?", ""),
        ("a?", "a"),
        ("a?", "aa"),
        ("a{2,4}", "a"),
        ("a{2,4}", "aa"),
        ("a{2,4}", "aaaa"),
        ("a{2,4}", "aaaaa"),
        ("a{2,4}", "baaab"),
    ];

    println!(
        "{:<10} {:<10} {:<10} {:<10} {:<10}",
        "pattern", "input", "match", "start", "end"
    );
    println!("{}", "-".repeat(50));

    for (pattern, input) in cases {
        let compiled = match engine.compile(pattern) {
            Ok(compiled) => compiled,
            Err(error) => {
                println!("{pattern:<10} {input:<10} compile error: {error}");
                continue;
            }
        };

        match compiled.exec(input.as_bytes()) {
            Ok(result) => {
                let span = if result.is_match {
                    String::from_utf8_lossy(&input.as_bytes()[result.start..result.end]).to_string()
                } else {
                    "-".to_string()
                };
                println!(
                    "{pattern:<10} {input:<10} {:<10} {:<10} {:<10} {}",
                    result.is_match, result.start, result.end, span
                );
            }
            Err(ExecError::NoTrampoline) => {
                println!("{pattern:<10} {input:<10} exec skipped: no native trampoline back end");
            }
            Err(ExecError::NoInterpreter) => {
                println!("{pattern:<10} {input:<10} exec skipped: no interpreter available");
            }
        }
    }
}
