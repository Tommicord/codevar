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

//! Error types.
//!
//! Every fallible operation in this crate returns one of the two enums below.
//! Neither panics: a malformed instruction or a machine model that cannot make
//! progress both surface as values.

use core::fmt;

use crate::asm::Unit;

/// Why a basic block could not be scheduled.
///
/// The list scheduler is total on acyclic dependence graphs, so a returned
/// error always describes an invalid *machine model* — never a valid program
/// that is merely awkward.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScheduleError {
    /// The ready list drained without every instruction issuing.
    ///
    /// Reported when the scheduler hits its internal progress guard, which
    /// indicates a machine model that cannot place a particular instruction.
    Stalled,
    /// A functional unit was configured with zero issue ports.
    ///
    /// Every unit an instruction could use must accept at least one
    /// instruction per cycle, or those instructions are unschedulable.
    NoPorts {
        /// The unit with no ports.
        unit: Unit,
    },
    /// The block has more instructions than the `u16` index space can address.
    BlockTooLarge,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stalled => f.write_str("scheduler made no progress on a non-empty ready list"),
            Self::NoPorts { unit } => {
                write!(f, "functional unit {} has no issue ports", unit.name())
            }
            Self::BlockTooLarge => f.write_str("basic block exceeds the 16-bit index space"),
        }
    }
}

impl core::error::Error for ScheduleError {}

/// Why an [`Instruction`](crate::asm::Instruction) could not be constructed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BuildError {
    /// More source registers than [`MAX_READS`](crate::asm::MAX_READS).
    TooManyReads,
    /// More destination registers than [`MAX_WRITES`](crate::asm::MAX_WRITES).
    TooManyWrites,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyReads => f.write_str("too many source registers for one instruction"),
            Self::TooManyWrites => f.write_str("too many destination registers for one instruction"),
        }
    }
}

impl core::error::Error for BuildError {}

/// Why a program-level schedule could not complete.
///
/// The [`Kernel`](crate::kernel::Kernel) explores only legal schedules, so a
/// returned error always describes an invalid *configuration* or a task that
/// can never finish — never a valid schedule that merely took long.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelError {
    /// The time slice was configured as zero cycles.
    ///
    /// A zero quantum would let the timer fire before any work runs, so the
    /// kernel rejects it up front rather than spinning.
    ZeroQuantum,
    /// The kernel hit its progress guard with a task still unfinished.
    ///
    /// Reported when the simulated clock exceeds every legal bound, which
    /// indicates a task whose work can never be discharged.
    Stalled,
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroQuantum => f.write_str("the time slice (quantum) must be at least one cycle"),
            Self::Stalled => f.write_str("the kernel made no progress with a task still runnable"),
        }
    }
}

impl core::error::Error for KernelError {}
