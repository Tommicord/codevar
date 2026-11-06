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

//! Scratch spill/perf probe. Delete after use.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use codevar_negex::Trampoline;
use codevar_negex_abi::MatchItem;
use codevar_negex_codegen::{Determinized, build, lower_dfa};
use codevar_negex_native_codegen::compile_negex_match;
use codevar_ocl_native_codegen::opt::{OptConfig, Optimization, Optimizations, optimize};
use codevar_ocl_native_codegen::pipeline::CodegenOptions;
use codevar_ocl_tc::host_target;
use std::time::Instant;

const LEN: usize = 262_144;
const REPS: usize = 40;

/// Compiles `pattern` and returns (machine code, entry offset).
fn compile_code(pattern: &str, optimizations: Optimizations, skip_cse: bool) -> Option<(Vec<u8>, u32)> {
    let output = codevar_negex_parse::parse(pattern);
    let (nfa, start) = build(&output.ast).ok()?;
    let dfa = Determinized::from_nfa(&nfa, start).ok()?;
    let module = lower_dfa(&dfa).ok()?;
    let mut bits = optimizations.bits();
    if skip_cse {
        bits &= !Optimization::CommonSubexpression.bit();
    }
    let options = CodegenOptions {
        opt: OptConfig {
            optimizations: Optimizations::from_bits(bits),
            inline_threshold: 48,
            unroll_limit: 8,
        },
        vector: Default::default(),
    };
    let target = host_target().ok()?;
    let native = compile_negex_match(&module, target.as_target(), &options).ok()?;
    Some((native.bytes, native.entry_offset))
}

/// Writes the compiled code of `pattern` to `path` for objdump.
fn write_disasm(pattern: &str, path: &str, opts: Optimizations, skip_cse: bool) {
    match compile_code(pattern, opts, skip_cse) {
        Some((bytes, entry)) => {
            if std::fs::write(path, &bytes).is_err() {
                println!("WRITE-FAIL {path}");
                return;
            }
            println!("{pattern}\t{path}\tlen={}\tentry={entry}", bytes.len());
        }
        None => println!("{pattern}\tNO-CODE"),
    }
}

fn run_native(pattern: &str, input_byte: u8, optimizations: Optimizations, skip_cse: bool) -> Option<f64> {
    let output = codevar_negex_parse::parse(pattern);
    let (nfa, start) = build(&output.ast).ok()?;
    let dfa = Determinized::from_nfa(&nfa, start).ok()?;
    let module = lower_dfa(&dfa).ok()?;
    let mut bits = optimizations.bits();
    if skip_cse {
        bits &= !Optimization::CommonSubexpression.bit();
    }
    let options = CodegenOptions {
        opt: OptConfig {
            optimizations: Optimizations::from_bits(bits),
            inline_threshold: 48,
            unroll_limit: 8,
        },
        vector: Default::default(),
    };
    let target = host_target().ok()?;
    let native = compile_negex_match(&module, target.as_target(), &options).ok()?;
    let trampoline = Trampoline::new(&native.bytes, native.entry_offset).ok()?;
    let table = codevar_negex_codegen::transition_table(&dfa);
    let input = vec![input_byte; LEN];
    let mut item = MatchItem::with_table(&input, &table);
    // warm-up
    trampoline.call(&mut item as *mut MatchItem);
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let mut item = MatchItem::with_table(&input, &table);
        let t = Instant::now();
        trampoline.call(&mut item as *mut MatchItem);
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / LEN as f64);
    }
    Some(best)
}

fn run_interp(pattern: &str, input_byte: u8) -> f64 {
    let output = codevar_negex_parse::parse(pattern);
    let (nfa, start) = build(&output.ast).unwrap();
    let dfa = Determinized::from_nfa(&nfa, start).unwrap();
    let interp = codevar_negex::Interpreter::new(dfa);
    let input = vec![input_byte; LEN];
    let mut item = MatchItem::new(&input);
    interp.exec(&input, &mut item);
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let mut item = MatchItem::new(&input);
        let t = Instant::now();
        interp.exec(&input, &mut item);
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / LEN as f64);
    }
    best
}

fn dump_mir(pattern: &str, bits: u32) {
    let output = codevar_negex_parse::parse(pattern);
    let (nfa, start) = build(&output.ast).unwrap();
    let dfa = Determinized::from_nfa(&nfa, start).unwrap();
    let mut module = lower_dfa(&dfa).unwrap();
    let optimizations = Optimizations::from_bits(bits);
    if !optimizations.is_empty() {
        let config = OptConfig {
            optimizations,
            ..OptConfig::default()
        };
        optimize(&mut module, &config).unwrap();
    }
    let target = host_target().unwrap();
    let entry = module.find_function("negex_match").unwrap();
    let selection = codevar_ocl_native_codegen::isel::select_module(&module, entry).unwrap();
    let mut machine = selection.module;
    if std::env::var("PREALLOC").is_ok() {
        for f in &machine.functions {
            println!(
                "v22={:?} v23={:?} v24={:?}",
                f.vreg_types.get(22),
                f.vreg_types.get(23),
                f.vreg_types.get(24)
            );
            break;
        }
        for f in &machine.functions {
            for (bi, block) in f.blocks.iter().enumerate() {
                println!("== block {bi}");
                for (ii, inst) in block.insts.iter().enumerate() {
                    println!(
                        "  {ii}: {:?} dest={:?} args={:?}",
                        inst.op,
                        inst.dest,
                        inst.args.as_slice()
                    );
                }
            }
        }
        return;
    }
    let stats =
        codevar_ocl_native_codegen::regalloc::allocate(&mut machine, target.as_target().registers()).unwrap();
    println!("stats: {stats:?}");
    for f in &machine.functions {
        for (bi, block) in f.blocks.iter().enumerate() {
            println!("== block {bi}");
            for (ii, inst) in block.insts.iter().enumerate() {
                println!(
                    "  {ii}: {:?} dest={:?} args={:?}",
                    inst.op,
                    inst.dest,
                    inst.args.as_slice()
                );
            }
        }
    }
}

/// Reports whether `pattern` compiles at `bits`, with noise-free counters.
fn report_compile(pattern: &str, bits: u32) {
    let mut why = String::new();
    let counted: Option<(u32, usize, usize, u32)> = (|| -> Option<(u32, usize, usize, u32)> {
        let output = codevar_negex_parse::parse(pattern);
        let (nfa, start) = build(&output.ast).ok()?;
        let dfa = Determinized::from_nfa(&nfa, start).ok()?;
        let mut module = lower_dfa(&dfa).ok()?;
        let optimizations = Optimizations::from_bits(bits);
        let config = OptConfig {
            optimizations,
            ..OptConfig::default()
        };
        optimize(&mut module, &config).ok()?;
        let target = host_target().ok()?;
        let entry = module.find_function("negex_match")?;
        let selection = codevar_ocl_native_codegen::isel::select_module(&module, entry).ok()?;
        let mut machine = selection.module;
        let stats = match codevar_ocl_native_codegen::regalloc::allocate(
            &mut machine,
            target.as_target().registers(),
        ) {
            Ok(stats) => stats,
            Err(err) => {
                why = format!("{err}");
                return None;
            }
        };
        let mut stack_ops = 0usize;
        let mut insts = 0usize;
        let mut reg_reg_copies = 0usize;
        for f in &machine.functions {
            for block in &f.blocks {
                for inst in &block.insts {
                    insts += 1;
                    if matches!(
                        inst.op,
                        codevar_ocl_native_codegen::mir::MOp::SpillStore { .. }
                            | codevar_ocl_native_codegen::mir::MOp::SpillLoad { .. }
                    ) {
                        stack_ops += 1;
                    }
                    if let codevar_ocl_native_codegen::mir::MOp::Copy = inst.op {
                        if let (
                            Some(codevar_ocl_native_codegen::mir::Destination::Physical(d)),
                            Some(codevar_ocl_native_codegen::mir::Operand::Physical(s)),
                        ) = (inst.dest, inst.args.as_slice().first())
                        {
                            if d != *s {
                                reg_reg_copies += 1;
                            }
                        }
                    }
                }
            }
        }
        let image = target
            .as_target()
            .lower_module(&machine, selection.entry)
            .ok()?
            .instruction_count;
        Some((
            stats.spills,
            stack_ops,
            reg_reg_copies,
            if image == 0 { insts as u32 } else { image },
        ))
    })();
    match counted {
        Some((spills, stack_ops, copies, insts)) => {
            println!("{pattern}\tcompiled\t{spills}\t{stack_ops}\t{insts}\tmoves={copies}")
        }
        None if !why.is_empty() => println!("{pattern}\tFAILED\t{why}"),
        None => println!("{pattern}\tNO-CODE"),
    }
}

fn main() {
    codevar_logger::set_min_log_level(codevar_logger::LogLevel::Error);
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("check") {
        let pattern = args.get(2).map(String::as_str).unwrap_or("a");
        let len: usize = args
            .get(3)
            .map(|s| s.parse().unwrap())
            .unwrap_or(1);
        let byte = *args
            .get(4)
            .map(String::as_bytes)
            .unwrap_or(b"a")
            .first()
            .unwrap_or(&b'a');
        let output = codevar_negex_parse::parse(pattern);
        let (nfa, start) = build(&output.ast).unwrap();
        let dfa = Determinized::from_nfa(&nfa, start).unwrap();
        let table = codevar_negex_codegen::transition_table(&dfa);
        println!("dfa: start={} states={}", dfa.start, dfa.states.len());
        for (i, st) in dfa.states.iter().enumerate() {
            let trans: Vec<String> = st
                .transitions
                .iter()
                .enumerate()
                .filter_map(|(b, t)| t.map(|t| format!("{:02x}->{t}", b as u8)))
                .collect();
            println!("  state {i} accept={} {}", st.accept, trans.join(" "));
        }
        println!(
            "entry(0,'a')={:08x} entry(1,0)={:08x} accepts={} row={}",
            table[(0 + 1) * 256 + 0x61],
            table[(1 + 1) * 256],
            codevar_negex_codegen::entry_accepts(table[(0 + 1) * 256 + 0x61]),
            codevar_negex_codegen::entry_row(table[(0 + 1) * 256 + 0x61]),
        );
        for (name, opts) in [("O0", Optimizations::O0), ("O2", Optimizations::O2)] {
            let module = lower_dfa(&dfa).unwrap();
            let options = CodegenOptions {
                opt: OptConfig {
                    optimizations: opts,
                    ..OptConfig::default()
                },
                vector: Default::default(),
            };
            let target = host_target().unwrap();
            let Some(native) = compile_negex_match(&module, target.as_target(), &options).ok() else {
                println!("{pattern}\t{len}\t{name}\tNO-CODE");
                continue;
            };
            let trampoline = Trampoline::new(&native.bytes, native.entry_offset).unwrap();
            let input = vec![byte; len];
            let mut item = MatchItem::with_table(&input, &table);
            trampoline.call(&mut item as *mut MatchItem);
            let mut interp_item = MatchItem::new(&input);
            let interp = codevar_negex::Interpreter::new(dfa.clone());
            interp.exec(&input, &mut interp_item);
            println!(
                "{pattern}\t{len}\t{name}\tnative=({}, {}, {})  interp=({}, {}, {})",
                item.result, item.start, item.end, interp_item.result, interp_item.start, interp_item.end
            );
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("dump") {
        let pattern = args.get(2).map(String::as_str).unwrap_or("a+");
        let level: u32 = args
            .get(3)
            .map(|s| s.parse().unwrap())
            .unwrap_or(Optimizations::O2.bits());
        dump_mir(pattern, level);
        return;
    }
    if args.get(1).map(String::as_str) == Some("disasm") {
        let pattern = args.get(2).map(String::as_str).unwrap_or("a+");
        write_disasm(pattern, "/tmp/negex_o2.bin", Optimizations::O2, false);
        write_disasm(pattern, "/tmp/negex_o0.bin", Optimizations::O0, false);
        write_disasm(pattern, "/tmp/negex_nocse.bin", Optimizations::O2, true);
        return;
    }
    if args.get(1).map(String::as_str) == Some("compile") {
        // Deterministic check: does the allocator produce code at all,
        // and how many values does it spill?  Immune to machine noise.
        let cases = [
            r"\d+",
            "a+",
            "a*",
            r"\w+",
            "cat|dog|bird|fish",
            "[a-zA-Z0-9._%+-]+@localhost",
            "(a+)+b",
        ];
        println!("pattern\tlevel\tstatus\tspills\tstack-ops\tinsts\tmoves");
        for (level, bits) in [("O2", Optimizations::O2.bits()), ("O0", Optimizations::O0.bits())] {
            for pattern in cases {
                print!("{level}\t");
                report_compile(pattern, bits);
            }
        }
        return;
    }
    let cases = [(r"\d+", b'7'), ("a+", b'a'), ("a*", b'a'), (r"\w+", b'a')];
    let levels: [(&str, Optimizations, bool); 6] = [
        ("O0", Optimizations::O0, false),
        ("O1", Optimizations::O1, false),
        ("O2", Optimizations::O2, false),
        ("O3", Optimizations::O3, false),
        ("O1-cse", Optimizations::O1, true),
        ("O2-cse", Optimizations::O2, true),
    ];
    println!("pattern\tlevel\tns/B");
    for (pattern, byte) in cases {
        for (name, opts, skip) in levels {
            match run_native(pattern, byte, opts, skip) {
                Some(ns) => println!("{pattern}\t{name}\t{ns:.2}"),
                None => println!("{pattern}\t{name}\tNO-CODE"),
            }
        }
        println!("{pattern}\tinterp\t{:.2}", run_interp(pattern, byte));
    }
}

// scratch: dump before allocation
