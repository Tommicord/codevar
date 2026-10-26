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

//! Machine descriptions.
//!
//! A [`Machine`] is the micro-architectural contract the scheduler reasons
//! against. It is data, not code, so the two families the crate ships —
//! [`bare`](crate::bare) and [`desktop`](crate::desktop) — are built from the
//! same table and the same scheduler; only the numbers and the [`Issue`] mode
//! differ.
//!
//! Three properties drive every scheduling decision:
//!
//! * **Issue mode** ([`Issue`]) — single- or multi-issue, in-order or
//!   out-of-order with renaming. This decides whether anti- and
//!   output-dependence edges survive in the graph.
//! * **Functional-unit table** ([`UnitSpec`]) — per-unit [`latency`] (when the
//!   result is readable) and [`ports`] (how many can start per cycle).
//! * **Control pipeline** (`delay_slots`) — whether a taken transfer exposes
//!   architectural delay slots that must be filled.
//!
//! [`latency`]: UnitSpec::latency
//! [`ports`]: UnitSpec::ports

use crate::asm::{Instruction, Unit};

/// The issue width and ordering discipline of a single hardware thread.
///
/// "Single-threaded" is assumed throughout: there is exactly one instruction
/// stream, and the scheduler only ever reorders instructions that a single
/// core executes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Issue {
    /// One instruction per cycle, issued strictly in program order, with no
    /// register renaming. Anti- and output-dependences are real hazards.
    SingleInOrder,
    /// `width` instructions per cycle, issued in program order, with no
    /// renaming.
    MultiInOrder {
        /// Instructions issued per cycle.
        width: u8,
    },
    /// `width` instructions per cycle with out-of-order issue and register
    /// renaming, so only true dependences constrain reordering.
    OutOfOrder {
        /// Instructions issued per cycle.
        width: u8,
    },
}

impl Issue {
    /// The maximum instructions issued per cycle.
    #[inline]
    pub const fn width(self) -> u8 {
        match self {
            Issue::SingleInOrder => 1,
            Issue::MultiInOrder { width } | Issue::OutOfOrder { width } => width,
        }
    }

    /// Whether the core renames registers, dropping WAR/WAW hazards.
    #[inline]
    pub const fn renames(self) -> bool {
        matches!(self, Issue::OutOfOrder { .. })
    }

    /// Whether the core may issue out of order.
    #[inline]
    pub const fn is_out_of_order(self) -> bool {
        matches!(self, Issue::OutOfOrder { .. })
    }
}

/// The cost of one functional unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UnitSpec {
    /// Cycles until the result is readable by a dependent.
    pub latency: u8,
    /// Instructions of this unit that may start per cycle. Must be non-zero
    /// for every unit an instruction uses.
    pub ports: u8,
}

impl UnitSpec {
    /// Creates a unit cost.
    #[inline]
    pub const fn new(latency: u8, ports: u8) -> Self {
        Self { latency, ports }
    }
}

impl Default for UnitSpec {
    #[inline]
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// A complete micro-architecture model for one hardware thread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Machine {
    /// A human-readable name, used in diagnostics and tests.
    pub name: &'static str,
    /// The issue discipline.
    pub issue: Issue,
    /// Per-unit cost, indexed by [`Unit::index`].
    pub units: [UnitSpec; Unit::COUNT],
    /// Number of architectural branch delay slots (zero on interlocked cores).
    pub delay_slots: u8,
}

impl Machine {
    /// The maximum instructions issued per cycle.
    #[inline]
    pub const fn issue_width(&self) -> u8 {
        self.issue.width()
    }

    /// Whether the core renames registers, and so hides WAR/WAW hazards.
    #[inline]
    pub const fn renames(&self) -> bool {
        self.issue.renames()
    }

    /// Whether the core issues out of order.
    #[inline]
    pub const fn is_out_of_order(&self) -> bool {
        self.issue.is_out_of_order()
    }

    /// The latency a dependent waits for an instruction resolved by `machine`.
    #[inline]
    pub fn latency(&self, instruction: &Instruction) -> u16 {
        instruction.latency(self)
    }

    /// The unit cost for `unit`.
    #[inline]
    pub fn unit(&self, unit: Unit) -> UnitSpec {
        self.units[unit.index()]
    }

    /// Validates the model before it is used.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::NoPorts`] when a unit an instruction could use
    /// has no issue ports. Every unit is checked, because the scheduler
    /// discovers a zero-port unit only by failing to place an instruction on
    /// it.
    pub fn validate(&self) -> Result<(), crate::error::ScheduleError> {
        if self.issue_width() == 0 {
            return Err(crate::error::ScheduleError::NoPorts { unit: Unit::Alu });
        }
        for unit in Unit::ALL {
            if self.unit(unit).ports == 0 {
                return Err(crate::error::ScheduleError::NoPorts { unit });
            }
        }
        Ok(())
    }
}
