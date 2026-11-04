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

//! Criterion benchmarks comparing the native compiled matcher with the
//! DFA interpreter.
//!
//! Negex runs the same DFA through two executors: machine code produced
//! by `codevar-negex-native-codegen` and called through a
//! [`codevar_negex::Trampoline`], and the Rust
//! [`codevar_negex::Interpreter`] that walks the transition tables one
//! byte at a time.  Every benchmark here hands *both* executors the same
//! job, so the numbers compare the executors rather than the work they
//! were given:
//!
//! * `search` — the end-to-end engine.  Each side runs the leftmost-start
//!   scan and then the anchored match over the whole input.  This is what
//!   a caller of `CompiledRegex::exec` pays.
//! * `anchored` — the matcher core alone.  Both sides are given an input
//!   made of one byte the pattern keeps consuming, anchored at offset zero,
//!   so the timed work grows with the input instead of collapsing to a
//!   one-byte tail or a first-byte rejection.  The gap between
//!   `search/native` and `anchored/native` is the reverse-automaton search
//!   plus the per-call trampoline the engine maps.
//! * `compilation` — the one-time cost, split into the automaton build
//!   and the full native codegen pipeline.
//!
//! Two guards keep the comparison honest:
//!
//! * A case is skipped when its pattern does not reach native code.
//!   `CompiledRegex::exec` silently falls back to the interpreter when
//!   codegen fails, which would otherwise put interpreter timings in the
//!   `native` rows.  Today `(a+)+b`, `cat|dog|bird|fish` and the email
//!   pattern are skipped: keeping a value live across the back edge of a
//!   wide-alternation DFA needs more than the eight allocatable integer
//!   registers, and the allocator refuses to spill a loop-carried value
//!   rather than emit code that reads a register another value reused.
//! * Every measured case asserts that both executors report the same
//!   span, so a miscompile fails the benchmark instead of measuring it.

// A benchmark is not production code: a setup step that cannot produce the
// thing being measured has nowhere to go, so it stops the run here rather
// than propagating an error criterion would only print.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_logger::{LogLevel, log_warn, set_min_log_level};
use codevar_negex::{CompiledRegex, Interpreter, MatchItem, Negex, Trampoline};
use codevar_negex_codegen::{Determinized, build, lower_dfa, transition_table};
use codevar_negex_native_codegen::{compile_negex_match, default_options};
use codevar_negex_parse::parse;
use codevar_ocl_tc::host_target;

/// Two-second measurement window: the largest inputs take a millisecond
/// per iteration, so a longer window buys precision nobody reads at the
/// cost of a run that never finishes.
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

/// One-second warm-up: enough to settle the caches and the branch
/// predictor before measurement starts.
const WARM_UP_TIME: Duration = Duration::from_secs(1);

/// Input sizes, spanning the per-call overhead and the asymptotic scan.
const SIZES: [usize; 3] = [1_024, 16_384, 262_144];

/// One benchmark case.
///
/// `filler` is the byte the input is built from and `sample` is placed at
/// the end of it, so a match always costs a full scan of the input.  The
/// filler is chosen per pattern so that the scan does real work: bytes the
/// automaton keeps consuming for the quantifier cases, bytes it dies on
/// for the literal cases.
///
/// `anchor` is the byte the `anchored` tier repeats across the whole
/// input.  It has to be a byte that keeps the automaton alive to the last
/// position, otherwise the anchored match dies on the first byte and the
/// tier times a constant instead of a scan; `None` means the pattern has
/// no such byte (a fixed-length match cannot scale with the input) and the
/// case is left out of that tier.
struct Case {
    /// The benchmark id.
    name: &'static str,
    /// The pattern to compile.
    pattern: &'static str,
    /// The byte the scanned input is filled with.
    filler: u8,
    /// A sample of the pattern's own alphabet, placed at the end.
    sample: &'static [u8],
    /// The byte the anchored tier fills the whole input with.
    anchor: Option<u8>,
}

static CASES: [Case; 8] = [
    Case {
        name: "literal",
        pattern: "helloworld",
        filler: b'x',
        sample: b"helloworld",
        anchor: None,
    },
    Case {
        name: "digit",
        pattern: r"\d+",
        filler: b'a',
        sample: b"7",
        anchor: Some(b'7'),
    },
    Case {
        name: "word_run",
        pattern: r"\w+",
        filler: b' ',
        sample: b"variableName123",
        anchor: Some(b'a'),
    },
    Case {
        name: "star",
        pattern: "a*",
        filler: b'a',
        sample: b"ab",
        anchor: Some(b'a'),
    },
    Case {
        name: "plus",
        pattern: "a+",
        filler: b'a',
        sample: b"ab",
        anchor: Some(b'a'),
    },
    Case {
        name: "alternation",
        pattern: "cat|dog|bird|fish",
        filler: b'x',
        sample: b"bird",
        anchor: None,
    },
    Case {
        name: "email",
        pattern: r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        filler: b' ',
        sample: b"user@example.com",
        anchor: None,
    },
    Case {
        name: "nested_star",
        pattern: "(a+)+b",
        filler: b'a',
        sample: b"b",
        anchor: Some(b'a'),
    },
];

/// A case that reached native code, with both executors ready.
struct Prepared<'a> {
    /// The pattern description.
    case: &'a Case,
    /// The engine, used for the end-to-end tier.
    compiled: CompiledRegex,
    /// The interpreter executor.
    interp: Interpreter,
    /// The compiled matcher, mapped once.
    trampoline: Trampoline,
    /// The transition table the compiled matcher steps through.
    trans_table: Vec<u32>,
}

/// An input of `size` bytes: the filler, then the sample at the very end.
fn input_for(case: &Case, size: usize) -> Vec<u8> {
    let mut input = vec![case.filler; size.saturating_sub(case.sample.len())];
    input.extend_from_slice(case.sample);
    input
}

/// Compiles every case that reaches native code, once.
///
/// The trampoline is built here rather than in the loop so that the timed
/// native side does not pay an `mmap` per call.  A case without native
/// code is skipped: `CompiledRegex::exec` would fall back to the
/// interpreter and the `native` label would not mean what it says.
fn prepare(engine: &Negex, pool: &'static [Case]) -> Vec<Prepared<'static>> {
    let mut ready = Vec::new();
    for case in pool {
        let compiled = engine.compile(case.pattern).unwrap();
        let Some(interp) = compiled.interpreter() else {
            log_warn!("negex-bench: skipping '{}': no interpreter", case.name);
            continue;
        };
        let Some((trampoline, trans_table)) = prepare_trampoline(case.pattern) else {
            log_warn!(
                "negex-bench: skipping '{}': no native code for '{}'",
                case.name,
                case.pattern
            );
            continue;
        };
        ready.push(Prepared {
            case,
            compiled,
            interp,
            trampoline,
            trans_table,
        });
    }
    ready
}

/// The native code of `pattern`, mapped into executable memory.
///
/// This is the same pipeline `Negex::compile` runs.
fn prepare_trampoline(pattern: &str) -> Option<(Trampoline, Vec<u32>)> {
    let output = parse(pattern);
    if !output.errors.is_empty() {
        return None;
    }
    let (nfa, start) = build(&output.ast).ok()?;
    let dfa = Determinized::from_nfa(&nfa, start).ok()?;
    let module = lower_dfa(&dfa).ok()?;
    let target = host_target().ok()?;
    let native = compile_negex_match(&module, target.as_target(), &default_options()).ok()?;
    let trampoline = Trampoline::new(native.bytes(), native.entry_offset()).ok()?;
    Some((trampoline, transition_table(&dfa)))
}

/// The end-to-end engine: leftmost search plus anchored match.
fn bench_search(c: &mut Criterion) {
    set_min_log_level(LogLevel::Error);
    let engine = Negex::new();
    let prepared = prepare(&engine, &CASES);

    let mut group = c.benchmark_group("search");
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);

    for entry in &prepared {
        for &size in &SIZES {
            let input = input_for(entry.case, size);
            let native = entry.compiled.exec(&input).unwrap();
            let interpreted = entry
                .compiled
                .exec_interpreted_with(&entry.interp, &input)
                .unwrap();
            assert_eq!(
                native, interpreted,
                "the executors disagree on '{}'",
                entry.case.name
            );

            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("native/{}", entry.case.name), size),
                &input,
                |b, input| {
                    b.iter(|| {
                        black_box(
                            entry
                                .compiled
                                .exec(black_box(input))
                                .unwrap()
                                .is_match(),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("interpreted/{}", entry.case.name), size),
                &input,
                |b, input| {
                    b.iter(|| {
                        black_box(
                            entry
                                .compiled
                                .exec_interpreted_with(&entry.interp, black_box(input))
                                .unwrap()
                                .is_match(),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

/// The matcher core alone, on an input anchored at zero.
fn bench_anchored(c: &mut Criterion) {
    set_min_log_level(LogLevel::Error);
    let engine = Negex::new();
    let prepared = prepare(&engine, &CASES);

    let mut group = c.benchmark_group("anchored");
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);

    for entry in &prepared {
        let Some(anchor) = entry.case.anchor else {
            continue;
        };
        for &size in &SIZES {
            let input = vec![anchor; size];
            let native = run_native(&entry.trampoline, &entry.trans_table, &input);
            let interpreted = run_interp(&entry.interp, &input);
            assert_eq!(
                native, interpreted,
                "the anchored executors disagree on '{}'",
                entry.case.name
            );
            assert!(
                native.0 == 0 || native.1 as usize == size,
                "'{}' does not scan the whole anchored input",
                entry.case.name
            );

            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("native/{}", entry.case.name), size),
                &input,
                |b, input| {
                    b.iter(|| {
                        black_box(run_native(
                            &entry.trampoline,
                            &entry.trans_table,
                            black_box(input),
                        ))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("interpreted/{}", entry.case.name), size),
                &input,
                |b, input| {
                    b.iter(|| black_box(run_interp(&entry.interp, black_box(input))));
                },
            );
        }
    }
    group.finish();
}

/// Runs the compiled matcher over `input`, anchored at zero.
fn run_native(trampoline: &Trampoline, table: &[u32], input: &[u8]) -> (u32, u32) {
    let mut item = MatchItem::with_table(input, table);
    trampoline.call(&raw mut item);
    (item.result, item.end)
}

/// Runs the interpreter over `input`, anchored at zero.
fn run_interp(interp: &Interpreter, input: &[u8]) -> (u32, u32) {
    let mut item = MatchItem::new(input);
    interp.exec(input, &mut item);
    (item.result, item.end)
}

/// The one-time cost, split into the automaton and the native codegen.
fn bench_compilation(c: &mut Criterion) {
    set_min_log_level(LogLevel::Error);
    let engine = Negex::new();

    let mut group = c.benchmark_group("compilation");
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);

    for case in &CASES {
        group.bench_function(BenchmarkId::new("automaton", case.name), |b| {
            b.iter(|| {
                let output = parse(black_box(case.pattern));
                let (nfa, start) = build(&output.ast).unwrap();
                let dfa = Determinized::from_nfa(&nfa, start).unwrap();
                let reverse = dfa.reverse().unwrap();
                black_box((dfa.states.len(), reverse.states.len()));
            });
        });
        group.bench_function(BenchmarkId::new("native", case.name), |b| {
            b.iter(|| {
                let compiled = engine.compile(black_box(case.pattern)).unwrap();
                black_box(&compiled);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search, bench_anchored, bench_compilation);
criterion_main!(benches);
