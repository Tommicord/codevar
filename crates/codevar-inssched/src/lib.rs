//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! A custom, single-threaded instruction scheduler for raw machine assembly.
//!
//! Codevar's editor targets run on everything from Cortex-M parts to desktop
//! cores, and the same straight-line assembly benefits from being ordered
//! differently on each. This crate is a self-contained machine-level
//! scheduler: give it a block of decoded instructions and a [`Machine`] model,
//! and it returns a legal issue order that hides latency on a stalled
//! in-order pipeline or relieves register pressure on a wide out-of-order one.
//!
//! The design follows the shape real compilers converged on — LLVM's
//! `ScheduleDAG` plus its pluggable `MachineSchedStrategy` — and is described
//! below.
//!
//! # The three layers
//!
//! 1. [`asm`] — the [`Instruction`] model. Instructions are *opcode-agnostic*:
//!    the target lowers its real assembly to the registers an instruction
//!    reads and writes, the [`Unit`](asm::Unit) it executes on, and its
//!    ordering [`InstFlags`](asm::InstFlags). There is no mnemonic table to
//!    maintain and no hard-coded instruction set; a new architecture is a new
//!    [`Machine`] and nothing else.
//! 2. [`dag`] — the [`ScheduleDAG`](dag::ScheduleDAG): [`SUnit`](dag::SUnit)
//!    nodes and [`SDep`](dag::SDep) edges tagged data / anti / output /
//!    order. Renaming cores drop the anti- and output-edges, which is the
//!    single most important difference between the two families.
//! 3. [`sched`] — a top-down list scheduler plus the three
//!    [`SchedStrategy`](sched::SchedStrategy) heuristics: source order,
//!    critical-path [`Latency`](sched::Latency), and
//!    [`RegisterPressure`](sched::RegisterPressure).
//!
//! Two ready-made model families sit on top:
//!
//! * [`bare`] — stalled, in-order, non-renaming cores (Cortex-M, ARM7,
//!   RV32 embedded, MIPS32 with its delay slot, an in-order dual-issue part).
//! * [`desktop`] — wide, renaming, out-of-order cores (x86-64, aarch64, a
//!   wider six-wide part).
//!
//! # Example
//!
//! ```rust
//! use codevar_inssched::asm::{InstFlags, Instruction, Reg, Unit};
//! use codevar_inssched::bare;
//! use codevar_inssched::sched::{Latency, schedule_block};
//!
//! // Load into r1, then use it: the load's latency should push the use out.
//! let load = Instruction::new(Unit::Load, InstFlags::READS_MEMORY, 0, &[Reg::new(0)], &[Reg::new(1)])?;
//! let use_it = Instruction::new(Unit::Alu, InstFlags::NONE, 0, &[Reg::new(1)], &[Reg::new(2)])?;
//!
//! let schedule = schedule_block(&[load, use_it], &bare::cortex_m(), &Latency)?;
//! assert_eq!(schedule.cycles, vec![0, 2]); // the use waits for the load
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Performance
//!
//! The scheduler is allocation-light: the graph, the ready list and the
//! result are three flat `Vec`s sized from the block length, the strategy is a
//! zero-sized generic so it inlines, and the hot path is O(n²) in the block
//! length — the right shape for the small straight-line blocks the model
//! schedules. No unbounded recursion, no panics, and `no_std` with `alloc`.
//!
//! # What it deliberately does not do
//!
//! It schedules *within a basic block* and assumes the caller has already
//! split control flow: a terminator is always placed last. It does not
//! allocate registers, so on a non-renaming core the targets must supply
//! physical register numbers directly. It never duplicates, deletes or
//! speculates an instruction.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod asm;
pub mod bare;
pub mod dag;
pub mod desktop;
pub mod error;
pub mod machine;
pub mod sched;

pub use asm::{InstFlags, Instruction, Reg, Unit};
pub use dag::{SDep, SDepKind, SUnit, ScheduleDAG};
pub use error::{BuildError, ScheduleError};
pub use machine::{Issue, Machine, UnitSpec};
pub use sched::{BlockSchedule, Latency, RegisterPressure, SchedStrategy, Slot, SourceOrder, schedule_block};
