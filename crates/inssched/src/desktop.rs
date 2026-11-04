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

//! Desktop machine models.
//!
//! Desktop cores are wide and out of order, so the scheduler's job changes
//! character. Register renaming removes anti- and output-dependences, leaving
//! only true dependences; the binding constraint becomes *issue bandwidth* —
//! the finite number of ports per functional unit — and *register pressure*.
//! The models therefore advertise [`Issue::OutOfOrder`], which tells the
//! dependence analysis to drop WAR/WAW edges, and a port count per unit that
//! caps same-cycle issue.
//!
//! Pair these models with the [`RegisterPressure`](crate::sched::RegisterPressure)
//! strategy: on a core that hides latency in hardware, reducing live ranges
//! buys more than reordering for latency.
//!
//! | Preset | Core family | Issue | Renaming |
//! |--------|-------------|-------|----------|
//! | [`x86_64`] | representative Skylake-class | 4-wide, OoO | yes |
//! | [`aarch64`] | representative OoO ARMv8-A | 4-wide, OoO | yes |
//! | [`wide_desktop`] | wide OoO (Firestorm-class) | 6-wide, OoO | yes |

use crate::asm::Unit;
use crate::machine::{Issue, Machine, UnitSpec};

/// Assembles a machine from its variable properties.
///
/// The unit table is given in [`Unit`] order: ALU, multiply, divide, load,
/// store, branch, float, float-divide, system.
const fn machine(name: &'static str, issue: Issue, units: [(u8, u8); Unit::COUNT]) -> Machine {
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
        delay_slots: 0,
    }
}

/// A representative four-wide x86-64 core: four integer ports, two load ports,
/// deep divide latency.
pub const fn x86_64() -> Machine {
    machine(
        "x86-64 (4-wide OoO)",
        Issue::OutOfOrder { width: 4 },
        [
            (1, 4),
            (3, 1),
            (20, 1),
            (4, 2),
            (1, 1),
            (1, 1),
            (4, 2),
            (14, 1),
            (1, 1),
        ],
    )
}

/// A representative four-wide out-of-order ARMv8-A core.
pub const fn aarch64() -> Machine {
    machine(
        "aarch64 (4-wide OoO)",
        Issue::OutOfOrder { width: 4 },
        [
            (1, 4),
            (3, 2),
            (10, 1),
            (4, 2),
            (1, 2),
            (1, 1),
            (3, 3),
            (10, 1),
            (4, 1),
        ],
    )
}

/// A wide six-wide out-of-order core, the desktop counterpart to
/// [`crate::bare::dual_issue`].
pub const fn wide_desktop() -> Machine {
    machine(
        "wide desktop (6-wide OoO)",
        Issue::OutOfOrder { width: 6 },
        [
            (1, 6),
            (3, 2),
            (12, 1),
            (4, 3),
            (1, 3),
            (1, 1),
            (3, 4),
            (12, 1),
            (5, 1),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_desktop_presets_validate() {
        for preset in [x86_64(), aarch64(), wide_desktop()] {
            preset
                .validate()
                .expect("desktop presets must be schedulable");
        }
    }

    #[test]
    fn desktop_presets_rename() {
        for preset in [x86_64(), aarch64(), wide_desktop()] {
            assert!(preset.renames(), "{} must rename", preset.name);
            assert!(preset.is_out_of_order());
        }
    }
}
