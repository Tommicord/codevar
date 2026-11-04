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

/// Why a byte range could not be decoded into basic blocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// The range ended in the middle of an instruction.
    Truncated {
        /// The address of the incomplete instruction.
        address: u64,
        /// How many bytes the range still had.
        available: usize,
    },
    /// An instruction starts at an address the target cannot encode.
    ///
    /// RV32I, for example, requires four-byte alignment.
    Misaligned {
        /// The offending address.
        address: u64,
    },
    /// The bytes do not encode an instruction this decoder supports.
    InvalidEncoding {
        /// The address of the instruction.
        address: u64,
        /// The raw encoding word, for diagnostics.
        word: u32,
    },
    /// A [`Decoder`](crate::decode::Decoder) reported a zero-length or
    /// oversized instruction, so the range cannot be walked safely.
    InvalidLength {
        /// The address the decoder claimed to decode.
        address: u64,
    },
    /// A direct control transfer lands in the middle of an instruction.
    MisalignedTarget {
        /// The target that is not an instruction boundary.
        target: u64,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { address, available } => write!(
                f,
                "instruction at {address:#x} is truncated ({available} byte(s) left)"
            ),
            Self::Misaligned { address } => {
                write!(f, "instruction address {address:#x} is not aligned")
            }
            Self::InvalidEncoding { address, word } => {
                write!(f, "invalid encoding {word:#010x} at {address:#x}")
            }
            Self::InvalidLength { address } => {
                write!(f, "decoder reported an invalid length at {address:#x}")
            }
            Self::MisalignedTarget { target } => {
                write!(f, "branch target {target:#x} is not an instruction boundary")
            }
        }
    }
}

impl core::error::Error for DecodeError {}

/// Why an executor run could not complete.
///
/// The [`Executor`](crate::executor::Executor) only stops early when a task is
/// still unfinished and nothing can make progress: no timer deadline, no
/// future arrival, and no waker. An externally driven event loop uses
/// [`Executor::run_until_idle`](crate::executor::Executor::run_until_idle)
/// instead, because a parked task waiting for I/O is normal there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExecutorError {
    /// A task is unfinished and no timer, arrival or waker can resume it.
    ///
    /// This is the deadlock case: every unfinished future returned `Pending`
    /// without arming a wake, and the virtual clock has nowhere to advance.
    Stalled,
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stalled => f.write_str("the executor made no progress with a task still pending"),
        }
    }
}

impl core::error::Error for ExecutorError {}
