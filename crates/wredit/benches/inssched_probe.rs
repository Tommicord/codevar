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
//! CONDITIONS OF ANY KIND, either express or implied. See the
//! License for the specific language governing permissions and
//! limitations under the License.

//! Probe: does `codevar-inssched` instruction scheduling make a Wredit
//! kernel actually run faster?
//!
//! [`schedule_block`] is a per-cycle simulation: it returns a permutation
//! plus a *predicted* makespan, and emits nothing. So "scheduled code" has
//! no runtime meaning until a caller applies a permutation to real
//! instructions. This probe closes that gap in two halves.
//!
//! 1. Model: express Wredit's hash Horner chain (`simd::hash_step`) as a
//!    straight-line block and ask three strategies for a makespan on an
//!    in-order machine (`bare::cortex_m`) and an out-of-order one
//!    (`desktop::x86_64`).
//! 2. Reality: hand-write the same recurrence in the orders the model can
//!    choose between — single dependency chain, per-lane interleave, and
//!    grouped-by-stage (all multiplies, then all adds, then all masks) —
//!    and measure them.
//!
//! The verdict is the pair of rankings: if the model's predicted ordering
//! gain does not show up in measured time on this out-of-order host, then
//! reordering a straight-line block is worth nothing here and the crate
//! is an estimator for in-order targets, not an optimizer for x86-64.
//!
//! Benchmarks are test-like code, not production code; a failed
//! instruction construction is a bug in this file, not a runtime condition.
#![allow(clippy::unwrap_used)]

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;

use codevar_inssched::Reg;
use codevar_inssched::Unit;
use codevar_inssched::asm::Instruction;
use codevar_inssched::bare::cortex_m;
use codevar_inssched::desktop::x86_64;
use codevar_inssched::sched::schedule_block;
use codevar_inssched::{InstFlags, Latency, RegisterPressure, SourceOrder};

use codevar_wredit::simd;

/// Words folded per block in the multi-lane variants.
const LANES: usize = 8;

/// The hash recurrence's constants, mirrored from `simd.rs`.
const HASH_PRIME: u64 = 0x1_0000_01B3;
const HASH_MASK: u64 = (1u64 << 52) - 1;

/// Nominal core clock, only used to turn measured nanoseconds into an
/// approximate cycle count comparable with the model's prediction.
const NOMINAL_GHZ: f64 = 3.2;

/// One step of the recurrence: `h = (h * PRIME + word) & MASK`.
#[inline(always)]
fn hash_step(hash: u64, word: u64) -> u64 {
    ((hash.wrapping_mul(HASH_PRIME) & HASH_MASK) + word) & HASH_MASK
}

/// Builds the block the model schedules: `lanes` independent copies of
/// `rounds` recurrence steps, each copy being a load, a multiply, an add
/// and a mask over one word.
///
/// Register `0` holds the shared accumulator of the single-chain form;
/// lane `l` owns registers `1 + 3*l .. 1 + 3*l + 2`.
fn build_block(lanes: usize, rounds: usize) -> Vec<Instruction> {
    let mut block = Vec::with_capacity(lanes * rounds * 4);
    let prime = Reg::new(200);
    let mask = Reg::new(201);
    for lane in 0..lanes {
        let acc = Reg::new(1 + 3 * lane as u16);
        let tmp = Reg::new(2 + 3 * lane as u16);
        let word = Reg::new(3 + 3 * lane as u16);
        for round in 0..rounds {
            // The word for this step comes from memory.
            block.push(
                Instruction::new(
                    Unit::Load,
                    InstFlags::READS_MEMORY,
                    0,
                    &[Reg::new(100 + (lane * rounds + round) as u16)],
                    &[word],
                )
                .unwrap(),
            );
            block.push(Instruction::new(Unit::Mul, InstFlags::NONE, 0, &[acc, prime], &[tmp]).unwrap());
            block.push(Instruction::new(Unit::Alu, InstFlags::NONE, 0, &[tmp, word], &[tmp]).unwrap());
            block.push(Instruction::new(Unit::Alu, InstFlags::NONE, 0, &[tmp, mask], &[acc]).unwrap());
        }
    }
    block
}

/// Predicted cycles per hash step for one strategy on one machine.
fn predicted(block: &[Instruction], machine: &codevar_inssched::machine::Machine, strategy: &str) -> u64 {
    let steps = (block.len() / 4).max(1) as u64;
    let makespan = match strategy {
        "SourceOrder" => schedule_block(block, machine, &SourceOrder).map(|s| u64::from(s.makespan)),
        "Latency" => schedule_block(block, machine, &Latency).map(|s| u64::from(s.makespan)),
        _ => schedule_block(block, machine, &RegisterPressure).map(|s| u64::from(s.makespan)),
    }
    .unwrap_or(u64::MAX);
    // Four instructions per step, of which one is the load: report the
    // makespan per recurrence step so variants with different lane counts
    // stay comparable.
    makespan / steps
}

/// Prints the model's side of the probe.
fn report_model() {
    let in_order = cortex_m();
    let out_of_order = x86_64();
    println!("inssched probe — predicted cycles per hash step (8 steps/block)\n");
    println!("{:<12} {:>14} {:>14} {:>14}", "machine", "SourceOrder", "Latency", "RegPressure");
    for lanes in [1usize, LANES] {
        let block = build_block(lanes, 8);
        println!(
            "{:<12} {:>14} {:>14} {:>14}   [{lanes} lane(s)]",
            "cortex_m",
            predicted(&block, &in_order, "SourceOrder"),
            predicted(&block, &in_order, "Latency"),
            predicted(&block, &in_order, "RegisterPressure"),
        );
        println!(
            "{:<12} {:>14} {:>14} {:>14}   [{lanes} lane(s)]",
            "x86_64",
            predicted(&block, &out_of_order, "SourceOrder"),
            predicted(&block, &out_of_order, "Latency"),
            predicted(&block, &out_of_order, "RegisterPressure"),
        );
    }
    println!();
}

/// Control: the engine's own scalar hash over the whole buffer, which is
/// one serial dependency chain.
#[inline]
fn chain_from_words(words: &[u64]) -> u64 {
    let mut hash = 0u64;
    for &word in words {
        hash = hash_step(hash, word);
    }
    hash
}

/// One accumulator per lane, advanced lane-by-lane: the order the source
/// is written in, and the order `SourceOrder` keeps.
#[inline]
fn lanes_interleaved(words: &[u64]) -> u64 {
    let mut acc = [0u64; LANES];
    for chunk in words.chunks_exact(LANES) {
        for (lane, &word) in chunk.iter().enumerate() {
            acc[lane] = hash_step(acc[lane], word);
        }
    }
    acc.iter().fold(0u64, |sum, &v| sum ^ v)
}

/// Grouped by stage — all multiplies, then all adds, then all masks —
/// which is the shape a latency-driven scheduler produces to keep the
/// independent multiplies off each other's critical path.
#[inline]
fn lanes_grouped(words: &[u64]) -> u64 {
    let mut acc = [0u64; LANES];
    let mut products = [0u64; LANES];
    for chunk in words.chunks_exact(LANES) {
        for lane in 0..LANES {
            products[lane] = acc[lane].wrapping_mul(HASH_PRIME) & HASH_MASK;
        }
        for lane in 0..LANES {
            products[lane] = (products[lane] + chunk[lane]) & HASH_MASK;
        }
        for lane in 0..LANES {
            acc[lane] = products[lane];
        }
    }
    acc.iter().fold(0u64, |sum, &v| sum ^ v)
}

/// Reads `bytes` as little-endian words, the way `simd::hash_state` does.
fn words_of(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut buffer = [0u8; 8];
            buffer.copy_from_slice(chunk);
            u64::from_le_bytes(buffer) & HASH_MASK
        })
        .collect()
}

fn bench_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("inssched/hash_order");
    for &size in &[4 * 1024usize, 64 * 1024] {
        let bytes: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let words = words_of(&bytes);
        let words = black_box(words);
        group.throughput(criterion::Throughput::Bytes(size as u64));

        // The engine's own kernel, as written today.
        group.bench_function(format!("wredit_scalar_{size}"), |b| {
            b.iter(|| black_box(simd::hash_bytes_scalar(black_box(&bytes))));
        });
        // Same recurrence, one chain, measured through the probe's helper.
        group.bench_function(format!("chain_1lane_{size}"), |b| {
            b.iter(|| black_box(chain_from_words(black_box(&words))));
        });
        // Eight lanes, source order.
        group.bench_function(format!("lanes_source_{size}"), |b| {
            b.iter(|| black_box(lanes_interleaved(black_box(&words))));
        });
        // Eight lanes, grouped by stage.
        group.bench_function(format!("lanes_grouped_{size}"), |b| {
            b.iter(|| black_box(lanes_grouped(black_box(&words))));
        });
    }
    group.finish();
}

/// Times each order directly and prints cycles per hash step, so the
/// measured ranking sits next to the model's predicted one.
fn report_measured() {
    let bytes: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
    let words = black_box(words_of(&bytes));
    let steps = words.len();
    let reps = 2_000;

    let mut best = f64::MAX;
    let mut rows = Vec::new();
    for (name, run) in [
        ("chain_1lane", &chain_from_words as &dyn Fn(&[u64]) -> u64),
        ("lanes_source", &lanes_interleaved),
        ("lanes_grouped", &lanes_grouped),
    ] {
        // Warm the caches, then take the best of five timed passes: the
        // minimum is the least noise-contaminated estimate of cost.
        for _ in 0..50 {
            black_box(run(black_box(&words)));
        }
        let mut ns_per_pass = f64::MAX;
        for _ in 0..5 {
            let start = std::time::Instant::now();
            for _ in 0..reps {
                black_box(run(black_box(&words)));
            }
            let elapsed = start.elapsed().as_nanos() as f64 / reps as f64;
            if elapsed < ns_per_pass {
                ns_per_pass = elapsed;
            }
        }
        let cycles = ns_per_pass / steps as f64 * NOMINAL_GHZ;
        if cycles < best {
            best = cycles;
        }
        rows.push((name, ns_per_pass, cycles));
    }
    println!("measured cost of the same recurrence (64 KiB, {} steps/pass):\n", steps);
    println!("{:<16} {:>12} {:>14} {:>10}", "order", "ns/pass", "cycles/step", "vs best");
    for (name, ns, cycles) in rows {
        println!("{name:<16} {ns:>12.1} {cycles:>14.3} {:>9.2}x", cycles / best);
    }
    println!();
}

fn main() {
    report_model();
    report_measured();
    let mut criterion = Criterion::default()
        .sample_size(30)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1))
        .configure_from_args();
    bench_orders(&mut criterion);
    criterion.final_summary();
}
