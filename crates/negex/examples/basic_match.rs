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

//! Compile a literal pattern and run it against an input buffer.
//!
//! Run with: cargo run --example basic_match -p codevar-negex

use codevar_negex::{ExecError, MatchResult, Negex};

fn main() {
    let engine = Negex::new();
    let pattern = "hello";
    let input = b"hello world";

    println!("pattern: {pattern:?}");
    println!("input:   {}", String::from_utf8_lossy(input));

    let compiled = match engine.compile(pattern) {
        Ok(compiled) => compiled,
        Err(error) => {
            println!("compile failed: {error}");
            return;
        }
    };

    match compiled.exec(input) {
        Ok(result) => print_match(result, input),
        Err(ExecError::NoTrampoline) => {
            println!("exec skipped: no native trampoline back end available on this target");
        }
        Err(ExecError::NoInterpreter) => {
            println!("exec skipped: no interpreter available");
        }
    }
}

fn print_match(result: MatchResult, input: &[u8]) {
    println!("is_match: {}", result.is_match);
    if result.is_match {
        let span = &input[result.start..result.end];
        println!(
            "span: {}..{} -> {}",
            result.start,
            result.end,
            String::from_utf8_lossy(span)
        );
    } else {
        println!("no match span");
    }
}
