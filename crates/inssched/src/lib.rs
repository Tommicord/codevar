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

//! A custom, single-threaded instruction scheduler and asynchronous executor.
//!
//! Codevar's editor targets run on everything from Cortex-M parts to desktop
//! cores. This crate covers both kinds of scheduling that cost on a single
//! core: the order of instructions *inside* a straight-line block, and the
//! order in which many asynchronous tasks share the core.
//!
//! # The instruction level
//!
//! The design follows the shape real compilers converged on — LLVM's
//! `ScheduleDAG` plus its pluggable `MachineSchedStrategy`:
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
//! 4. [`decode`] — turns raw machine code into a [`Function`]
//!    of [`BasicBlock`]s, so the scheduler sees the real
//!    I/O loops (a poll, a branch back, a fence, a syscall) instead of asking
//!    the caller to pre-split control flow. [`Rv32i`] is the
//!    built-in decoder; implement [`Decoder`] for other ISAs.
//!
//! Two ready-made model families sit on top:
//!
//! * [`bare`] — stalled, in-order, non-renaming cores (Cortex-M, ARM7,
//!   RV32 embedded, MIPS32 with its delay slot, an in-order dual-issue part).
//! * [`desktop`] — wide, renaming, out-of-order cores (x86-64, aarch64, a
//!   wider six-wide part).
//!
//! # The program level: a real executor
//!
//! [`executor`] is not a simulator: it owns real [`Future`]s and polls them.
//! Each task gets its own [`Waker`](core::task::Waker); a wired wake puts the
//! task back on the ready queue, and the [`SchedulePolicy`] decides which ready
//! task is polled next. A future that returns `Pending` without waking is
//! parked until something wakes it, which is the standard Rust async contract.
//!
//! Because the executor is cooperative, it cannot interrupt a future in the
//! middle of a poll. A task yields exactly where it awaits. [`yield_now`] is
//! the explicit "give someone else a turn" point, and [`Timer`] provides a
//! deterministic virtual clock plus a deadline queue so [`Sleep`] works
//! without a hardware timer.
//!
//! The clock is virtual: a poll costs [`Config::poll_cost`] cycles, a dispatch
//! costs [`Config::switch_cost`], and an idle executor jumps to the next
//! deadline. [`Stats`] turns that into CPU utilization, idle cycles, waiting
//! time and turnaround.
//!
//! [`Task::cost_from_block`] and [`Task::cost_from_function`] are the seam
//! between the two levels: they schedule a task's straight-line program (or its
//! decoded blocks) with the instruction layer and use the resulting makespan as
//! the cost estimate the program level accounts against.
//!
//! # Example
//!
//! Ordering instructions within one program:
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
//! Multiplexing futures on one core with the executor:
//!
//! ```rust
//! use codevar_inssched::executor::{Config, Executor};
//! use codevar_inssched::policy::RoundRobin;
//! use codevar_inssched::task::Priority;
//!
//! let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
//! let timer = executor.timer().clone();
//! executor.spawn("render", Priority::NORMAL, async move {
//!     timer.sleep(2).await;
//! });
//! executor.spawn("compile", Priority::NORMAL, async {});
//!
//! let stats = executor.run()?;
//! assert_eq!(stats.completions, 2);
//! assert!(stats.polls >= 3);
//! # Ok::<(), codevar_inssched::ExecutorError>(())
//! ```
//!
//! # Performance
//!
//! The instruction scheduler is allocation-light: the graph, the ready list
//! and the result are three flat `Vec`s sized from the block length, the
//! strategy is a zero-sized generic so it inlines, and the hot path is O(n²) in
//! the block length — the right shape for the small straight-line blocks the
//! model schedules.
//!
//! The executor allocates one boxed future per task, one small `Arc` wake flag
//! per task, and keeps the task table, ready queue and pending queue as flat
//! `Vec`s. A wake is a single atomic store; a tick scans the task table for
//! fired wakes and asks the policy to choose from the ready queue. Sleepers
//! live in a binary heap ordered by deadline. No unbounded recursion and no
//! panics; `no_std` with `alloc`.
//!
//! # What it deliberately does not do
//!
//! * The instruction layer schedules *within a basic block*: a terminator is
//!   always placed last, and [`decode`] supplies the block boundaries from real
//!   machine code. Scheduling never crosses a block edge, a barrier or a call.
//!   The layer does not allocate registers, so on a non-renaming core the
//!   targets must supply physical register numbers. It never duplicates,
//!   deletes or speculates an instruction.
//! * [`decode`] recovers blocks, not a full call graph: an indirect target
//!   stays indirect, a target outside the decoded range stays external, and
//!   code reached only through a jump table is not discovered.
//! * The executor is single-core and cooperative: it never preempts a future
//!   mid-poll, and the scheduling policy only chooses among tasks that are
//!   already ready.
//! * There is no I/O reactor. Integrate one through
//!   [`Executor::waker`](executor::Executor::waker): clone the waker into the
//!   event source and wake the task when the event arrives.
//! * Spawned futures are `'static` and produce `()`. Spawning from inside a
//!   task needs access to the executor, which the current API does not provide.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod asm;
pub mod bare;
pub mod dag;
pub mod decode;
pub mod desktop;
pub mod error;
pub mod executor;
pub mod machine;
pub mod policy;
pub mod sched;
pub mod task;

pub use asm::{InstFlags, Instruction, Reg, Unit};
pub use dag::{SDep, SDepKind, SUnit, ScheduleDAG};
pub use decode::{
    BasicBlock, ControlFlow, Decoded, Decoder, Function, FunctionSchedule, Rv32i, decode_function,
    schedule_function,
};
pub use error::{BuildError, DecodeError, ExecutorError, ScheduleError};
pub use executor::{
    Config, Executor, Phase, Sleep, Stats, Tick, Timer, YieldNow, block_on, schedule_phases, yield_now,
};
pub use machine::{Issue, Machine, UnitSpec};
pub use policy::{RoundRobin, SchedulePolicy, ShortestJobFirst};
pub use sched::{BlockSchedule, Latency, RegisterPressure, SchedStrategy, Slot, SourceOrder, schedule_block};
pub use task::{Priority, Task, TaskId, TaskState};
