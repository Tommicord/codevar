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

//! Character classes: `[0-9]`, `[^0-9]`, `\d`, `\w`, `\s`.
//!
//! Run with: cargo run --example character_classes -p codevar-negex

use codevar_negex::{ExecError, Negex};

fn main() {
    let engine = Negex::new();

    let cases: &[(&str, &str)] = &[
        (r"[0-9]", "abc123def"),
        (r"[0-9]", "abcdef"),
        (r"[^0-9]", "12345"),
        (r"[^0-9]", "12a45"),
        (r"\d", "price: 9 dollars"),
        (r"\d", "no digits here"),
        (r"\w", "hello_world"),
        (r"\w", "-!@-"),
        (r"\s", "a b\tc"),
        (r"\s", "nospace"),
    ];

    println!(
        "{:<12} {:<18} {:<8} {:<6} {:<6}",
        "class", "input", "match", "start", "end"
    );
    println!("{}", "-".repeat(52));

    for (pattern, input) in cases {
        let compiled = match engine.compile(pattern) {
            Ok(compiled) => compiled,
            Err(error) => {
                println!("{pattern:<12} {input:<18} compile error: {error}");
                continue;
            }
        };

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
                    "{pattern:<12} {input:<18} {:<8} {:<6} {:<6} {}",
                    result.is_match, result.start, result.end, span
                );
            }
            Err(ExecError::NoTrampoline) => {
                println!("{pattern:<12} {input:<18} exec skipped: no native trampoline back end");
            }
            Err(ExecError::NoInterpreter) => {
                println!("{pattern:<12} {input:<18} exec skipped: no interpreter available");
            }
        }
    }
}
