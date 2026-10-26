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

//! Bare-metal machine models.
//!
//! Embedded cores are the forcing case for instruction scheduling: they are
//! single-issue and in order, so a dependent instruction placed one cycle
//! after a long-latency producer *stalls the pipeline in place*. The models
//! here therefore carry real latencies for loads, multiplies and divides, and
//! [`Machine::renames`] is `false`, so the dependence graph keeps the
//! anti- and output-dependences that a renamer would have removed.
//!
//! | Preset | Core family | Issue | Delay slots |
//! |--------|-------------|-------|-------------|
//! | [`cortex_m`] | ARMv7-M (Cortex-M3/M4) | single, in-order | 0 |
//! | [`riscv_embedded`] | RV32I/M embedded | single, in-order | 0 |
//! | [`arm7`] | ARM7TDMI | single, in-order | 0 |
//! | [`mips32`] | MIPS32 | single, in-order | 1 |
//! | [`dual_issue`] | in-order dual-issue embedded | 2-wide, in-order | 0 |
//!
//! The numbers are *representative* — chosen from published pipeline
//! documentation to drive good schedules — not cycle-exact. Replace a value
//! with the measured latency for a specific part when it matters.

use crate::asm::Unit;
use crate::machine::{Issue, Machine, UnitSpec};

/// Assembles a machine from its three variable properties.
///
/// The unit table is given in [`Unit`] order: ALU, multiply, divide, load,
/// store, branch, float, float-divide, system.
const fn machine(
    name: &'static str,
    issue: Issue,
    delay_slots: u8,
    units: [(u8, u8); Unit::COUNT],
) -> Machine {
    let mut table = [UnitSpec::new(1, 1); Unit::COUNT];
    let mut index = 0;
    while index < Unit::COUNT {
        table[index] = UnitSpec::new(units[index].0, units[index].1);
        index += 1;
    }
    Machine {
        name,
        issue,
        units: table,
        delay_slots,
    }
}

/// ARMv7-M (Cortex-M3/M4): three-stage in-order pipeline, interlocked loads.
///
/// The load latency of 2 and the multiply latency of 2 are what make
/// scheduling worthwhile: without it, every load is immediately followed by a
/// use that stalls.
pub const fn cortex_m() -> Machine {
    machine(
        "cortex-m3/m4",
        Issue::SingleInOrder,
        0,
        [
            (1, 1),
            (2, 1),
            (12, 1),
            (2, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (14, 1),
            (4, 1),
        ],
    )
}

/// An RV32I/M embedded core: in-order, one-cycle ALU, multi-cycle multiply and
/// divide, no delay slots.
pub const fn riscv_embedded() -> Machine {
    machine(
        "riscv32 embedded",
        Issue::SingleInOrder,
        0,
        [
            (1, 1),
            (3, 1),
            (10, 1),
            (2, 1),
            (1, 1),
            (1, 1),
            (4, 1),
            (20, 1),
            (2, 1),
        ],
    )
}

/// ARM7TDMI: in-order, no architectural delay slots; branches flush a short
/// pipeline and loads take an extra cycle.
pub const fn arm7() -> Machine {
    machine(
        "arm7tdmi",
        Issue::SingleInOrder,
        0,
        [
            (1, 1),
            (2, 1),
            (1, 1),
            (2, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
        ],
    )
}

/// MIPS32: the canonical one-delay-slot machine. Every branch and jump exposes
/// one slot, so the scheduler fills it or leaves a nop.
pub const fn mips32() -> Machine {
    machine(
        "mips32",
        Issue::SingleInOrder,
        1,
        [
            (1, 1),
            (2, 1),
            (32, 1),
            (2, 1),
            (1, 1),
            (1, 1),
            (4, 1),
            (30, 1),
            (1, 1),
        ],
    )
}

/// An in-order dual-issue embedded core: two instructions per cycle with no
/// renaming, so port pressure and anti-dependences both bind.
pub const fn dual_issue() -> Machine {
    machine(
        "embedded dual-issue",
        Issue::MultiInOrder { width: 2 },
        0,
        [
            (1, 2),
            (2, 1),
            (10, 1),
            (2, 1),
            (1, 1),
            (1, 1),
            (4, 1),
            (18, 1),
            (2, 1),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::Issue;

    #[test]
    fn all_bare_presets_validate() {
        for preset in [cortex_m(), riscv_embedded(), arm7(), mips32(), dual_issue()] {
            preset
                .validate()
                .expect("bare presets must be schedulable");
        }
    }

    #[test]
    fn bare_presets_do_not_rename() {
        for preset in [cortex_m(), riscv_embedded(), arm7(), mips32(), dual_issue()] {
            assert!(!preset.renames(), "{} must not rename", preset.name);
        }
    }

    #[test]
    fn mips_exposes_a_delay_slot() {
        assert_eq!(mips32().delay_slots, 1);
        assert_eq!(cortex_m().delay_slots, 0);
    }

    #[test]
    fn dual_issue_is_two_wide() {
        assert!(matches!(dual_issue().issue, Issue::MultiInOrder { width: 2 }));
    }
}
