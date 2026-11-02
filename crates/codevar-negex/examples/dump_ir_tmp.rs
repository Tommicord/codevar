// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing
// permissions and limitations under the License.

//! Scratch IR dumper. Delete after use.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use codevar_logger::{LogLevel, set_min_log_level};
use codevar_negex_codegen::{Determinized, build, lower_dfa};
use codevar_negex_parse::parse;
use codevar_ocl_native_codegen::opt::{OptConfig, Optimizations, optimize};

fn main() {
    set_min_log_level(LogLevel::Error);
    let pattern = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a+".to_string());
    let level: u32 = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "1".to_string())
        .parse()
        .unwrap();
    let output = parse(&pattern);
    let (nfa, start) = build(&output.ast).unwrap();
    let dfa = Determinized::from_nfa(&nfa, start).unwrap();
    let mut module = lower_dfa(&dfa).unwrap();
    let optimizations = Optimizations::from_bits(level);
    if !optimizations.is_empty() {
        let config = OptConfig {
            optimizations,
            ..OptConfig::default()
        };
        optimize(&mut module, &config).unwrap();
    }
    println!("{}", module);
}
