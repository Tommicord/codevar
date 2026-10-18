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

//! Machine-instruction model.
//!
//! The scheduler is *opcode-agnostic*: it never asks what an instruction
//! means, only which registers it reads and writes, which functional unit it
//! executes on, and whether it has ordering side effects. This is the seam
//! LLVM draws between `MachineInstr` and its `TargetInstrInfo` — the target
//! lowers real assembly to a [`Instruction`], and the scheduler consumes the
//! decoded facts. Keeping opcodes out of the crate is deliberate: there is no
//! instruction table to keep in sync, no hard-coded mnemonic list, and a new
//! architecture needs no change here.
//!
//! [`Instruction`] is small and `Copy`, so a block is a plain `&[Instruction]`
//! slice the scheduler can scan in the hot path without indirection.

use core::fmt;
use core::ops::{BitOr, BitOrAssign};

use crate::error::BuildError;
use crate::machine::Machine;

/// Maximum number of source registers modeled per instruction.
pub const MAX_READS: usize = 4;
/// Maximum number of destination registers modeled per instruction.
pub const MAX_WRITES: usize = 2;

/// A physical register, interned by the caller.
///
/// Identifiers are small dense indices. [`Reg::NONE`] is the sentinel used
/// internally where no register is present; real registers use ids below
/// [`Reg::NONE`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Reg(u16);

impl Reg {
    /// Sentinel meaning "no register".
    pub const NONE: Self = Self(u16::MAX);

    /// Creates a register from its interned id.
    ///
    /// An id of `u16::MAX` yields [`Reg::NONE`].
    #[inline]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    /// The interned id.
    #[inline]
    pub const fn id(self) -> u16 {
        self.0
    }

    /// Whether this is the [`Reg::NONE`] sentinel.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u16::MAX
    }
}

impl fmt::Debug for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            f.write_str("Reg(NONE)")
        } else {
            write!(f, "Reg({})", self.0)
        }
    }
}

/// The functional unit an instruction executes on.
///
/// The variants name the *kind* of execution resource, so one machine model
/// describes a Cortex-M and a desktop core with the same table while still
/// distinguishing a 1-cycle ALU op from a 20-cycle divide.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Unit {
    /// Integer arithmetic and logic.
    Alu = 0,
    /// Integer multiply.
    Mul = 1,
    /// Integer divide and remainder.
    Div = 2,
    /// Memory read.
    Load = 3,
    /// Memory write.
    Store = 4,
    /// Control transfer (branch, call, return).
    Branch = 5,
    /// Floating-point arithmetic.
    Float = 6,
    /// Floating-point divide and square root.
    FloatDiv = 7,
    /// Serializing/system instructions (fences, inline assembly, syscalls).
    System = 8,
}

impl Unit {
    /// The number of units, and the length of a per-unit table.
    pub const COUNT: usize = 9;

    /// Every unit in table order, matching [`Unit::index`].
    pub const ALL: [Unit; Self::COUNT] = [
        Unit::Alu,
        Unit::Mul,
        Unit::Div,
        Unit::Load,
        Unit::Store,
        Unit::Branch,
        Unit::Float,
        Unit::FloatDiv,
        Unit::System,
    ];

    /// The dense index of this unit, in `0..`[`Unit::COUNT`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// A short stable name for diagnostics.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Unit::Alu => "alu",
            Unit::Mul => "mul",
            Unit::Div => "div",
            Unit::Load => "load",
            Unit::Store => "store",
            Unit::Branch => "branch",
            Unit::Float => "float",
            Unit::FloatDiv => "float-div",
            Unit::System => "system",
        }
    }

    /// Whether the unit issues memory traffic.
    #[inline]
    pub const fn is_memory(self) -> bool {
        matches!(self, Unit::Load | Unit::Store)
    }
}

impl fmt::Debug for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Per-instruction properties the scheduler and dependence analysis need.
///
/// A bit mask rather than a group of `bool` fields: one `u16` instead of many
/// padding bytes, matching the crate's storage convention.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct InstFlags(u16);

impl InstFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// The instruction changes control flow.
    pub const BRANCH: Self = Self(1 << 0);
    /// The branch is conditional (its delay slot still executes).
    pub const CONDITIONAL: Self = Self(1 << 1);
    /// The instruction ends its basic block.
    pub const TERMINATOR: Self = Self(1 << 2);
    /// The instruction is a subroutine call.
    pub const CALL: Self = Self(1 << 3);
    /// The instruction serializes memory and instruction ordering.
    pub const BARRIER: Self = Self(1 << 4);
    /// The instruction has an effect that must not be duplicated or dropped.
    pub const SIDE_EFFECT: Self = Self(1 << 5);
    /// The instruction reads memory.
    pub const READS_MEMORY: Self = Self(1 << 6);
    /// The instruction writes memory.
    pub const WRITES_MEMORY: Self = Self(1 << 7);
    /// The instruction may legally be moved into a branch delay slot.
    pub const DELAY_SAFE: Self = Self(1 << 8);
    /// Flags that stop reordering from crossing an instruction.
    pub const ORDERING: Self = Self(Self::BARRIER.0 | Self::CALL.0);

    /// The raw bits.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// `true` when every bit of `other` is set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// `true` when no bit is set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The union of two flag sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether any bit of `other` is set in `self`.
    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl BitOr for InstFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for InstFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for InstFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstFlags({:#06x})", self.0)
    }
}

/// One machine instruction, decoded for the scheduler.
///
/// Construct with [`Instruction::new`], which copies the register sets into
/// fixed-size inline storage and rejects anything larger than the model
/// supports.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Instruction {
    unit: Unit,
    flags: InstFlags,
    latency: u16,
    nreads: u8,
    nwrites: u8,
    reads: [Reg; MAX_READS],
    writes: [Reg; MAX_WRITES],
}

impl Instruction {
    /// Builds an instruction from its source and destination registers.
    ///
    /// `latency` is an explicit override, or `0` to use the machine model's
    /// latency for `unit`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::TooManyReads`] or [`BuildError::TooManyWrites`]
    /// when the register sets exceed [`MAX_READS`]/[`MAX_WRITES`].
    pub fn new(
        unit: Unit,
        flags: InstFlags,
        latency: u16,
        reads: &[Reg],
        writes: &[Reg],
    ) -> Result<Self, BuildError> {
        if reads.len() > MAX_READS {
            return Err(BuildError::TooManyReads);
        }
        if writes.len() > MAX_WRITES {
            return Err(BuildError::TooManyWrites);
        }
        let mut instruction = Self {
            unit,
            flags,
            latency,
            nreads: reads.len() as u8,
            nwrites: writes.len() as u8,
            reads: [Reg::NONE; MAX_READS],
            writes: [Reg::NONE; MAX_WRITES],
        };
        instruction.reads[..reads.len()].copy_from_slice(reads);
        instruction.writes[..writes.len()].copy_from_slice(writes);
        Ok(instruction)
    }

    /// The functional unit.
    #[inline]
    pub const fn unit(&self) -> Unit {
        self.unit
    }

    /// The instruction's flags.
    #[inline]
    pub const fn flags(&self) -> InstFlags {
        self.flags
    }

    /// The explicit latency override, or `0` to use the machine default.
    #[inline]
    pub const fn latency_override(&self) -> u16 {
        self.latency
    }

    /// The source registers.
    #[inline]
    pub fn reads(&self) -> &[Reg] {
        &self.reads[..self.nreads as usize]
    }

    /// The destination registers.
    #[inline]
    pub fn writes(&self) -> &[Reg] {
        &self.writes[..self.nwrites as usize]
    }

    /// `true` when the instruction transfers control.
    #[inline]
    pub const fn is_terminator(&self) -> bool {
        self.flags.contains(InstFlags::TERMINATOR)
    }

    /// The latency this instruction resolves with under `machine`, honoring
    /// any [`latency_override`](Self::latency_override).
    #[inline]
    pub fn latency(&self, machine: &Machine) -> u16 {
        if self.latency == 0 {
            u16::from(machine.unit(self.unit).latency)
        } else {
            self.latency
        }
    }
}
