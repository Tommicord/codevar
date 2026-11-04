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

//! List scheduling and the pluggable priority strategies.
//!
//! The core is a top-down, per-cycle list scheduler over the
//! [`ScheduleDAG`]:
//!
//! 1. A node is **ready** once every predecessor has issued.
//! 2. It is **data-ready** at a cycle once every producer's result is
//!    readable, i.e. `producer_cycle + edge_latency <= cycle`.
//! 3. It is **issuable** when its functional unit still has a free port this
//!    cycle and the issue width is not exceeded.
//!
//! Among the issuable nodes, a [`SchedStrategy`] picks one. The crate ships
//! the three strategies a production scheduler relies on — mirroring LLVM's
//! `SourceList`, `LatencyPriorityQueue` and `RegReductionPriorityQueue`:
//!
//! * [`SourceOrder`] keeps the original order and only repairs stalls. It is
//!   the safe default when scheduling is not known to help.
//! * [`Latency`] is the classic critical-path (highest-level-first) heuristic.
//!   It hides latency, which is what a stalled in-order pipeline needs most.
//! * [`RegisterPressure`] prefers instructions that *end* live ranges, keeping
//!   the number of simultaneously live registers down. It is the right choice
//!   when the machine is out of order and pressure, not latency, is the limit.
//!
//! The strategy is a generic type parameter, so selecting one costs nothing at
//! run time and a target can supply its own.

use alloc::vec;
use alloc::vec::Vec;

use crate::asm::{InstFlags, Instruction, Unit};
use crate::dag::ScheduleDAG;
use crate::error::ScheduleError;
use crate::machine::Machine;

/// One position in a scheduled block.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Slot {
    /// The instruction originally at this index in the block.
    Instruction(u16),
    /// A no-op inserted to satisfy a branch delay slot.
    Nop,
}

impl Slot {
    /// The instruction index, or `None` for a nop.
    #[inline]
    pub const fn instruction(self) -> Option<u16> {
        match self {
            Slot::Instruction(index) => Some(index),
            Slot::Nop => None,
        }
    }
}

/// The schedule of one basic block.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BlockSchedule {
    /// The final order, including any inserted delay-slot nops.
    pub slots: Vec<Slot>,
    /// The issue cycle of every slot, parallel to [`slots`](Self::slots).
    ///
    /// Exact for a block with no delay slots. A delay-slot nop and any
    /// instruction moved into a slot carry a best-effort cycle.
    pub cycles: Vec<u16>,
    /// Cycles from the first issue to the last instruction's issue, inclusive.
    pub makespan: u16,
    /// Number of delay-slot nops inserted.
    pub nops: u16,
}

impl BlockSchedule {
    /// The slot at `index`, or [`Slot::Nop`] when out of range.
    #[inline]
    pub fn slot(&self, index: usize) -> Slot {
        self.slots
            .get(index)
            .copied()
            .unwrap_or(Slot::Nop)
    }
}

/// A priority heuristic that picks which ready instruction issues next.
///
/// Implementations are expected to be cheap, pure functions of the graph: the
/// scheduler may call [`pick`](SchedStrategy::pick) several times per cycle.
/// `candidates` is never empty.
pub trait SchedStrategy {
    /// A short name for diagnostics.
    fn name(&self) -> &'static str;

    /// Chooses one of `candidates` (indices into [`ScheduleDAG::units`]).
    fn pick(&self, dag: &ScheduleDAG, candidates: &[u16]) -> u16;
}

/// Keeps the original program order, repairing only latency and resource
/// stalls. The conservative default.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceOrder;

impl SchedStrategy for SourceOrder {
    #[inline]
    fn name(&self) -> &'static str {
        "source-order"
    }

    #[inline]
    fn pick(&self, _dag: &ScheduleDAG, candidates: &[u16]) -> u16 {
        *candidates.iter().min().unwrap_or(&0)
    }
}

/// Highest-level-first: always issue the instruction on the longest remaining
/// path. The classic critical-path heuristic for latency-bound cores.
#[derive(Clone, Copy, Debug, Default)]
pub struct Latency;

impl SchedStrategy for Latency {
    #[inline]
    fn name(&self) -> &'static str {
        "latency"
    }

    #[inline]
    fn pick(&self, dag: &ScheduleDAG, candidates: &[u16]) -> u16 {
        let mut best = candidates[0];
        let mut best_height = dag.units[best as usize].height;
        for &candidate in &candidates[1..] {
            let height = dag.units[candidate as usize].height;
            if height > best_height || (height == best_height && candidate < best) {
                best = candidate;
                best_height = height;
            }
        }
        best
    }
}

/// Register-pressure reduction: prefers instructions that end live ranges over
/// those that start them, breaking ties toward the critical path. The right
/// heuristic for an out-of-order core.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegisterPressure;

impl RegisterPressure {
    /// Estimated change in live registers from issuing `node`.
    ///
    /// A node that has consumers left *creates* a live value; a node that
    /// consumes the last use of a producer *kills* one.
    #[inline]
    fn pressure_delta(dag: &ScheduleDAG, node: u16) -> i32 {
        let index = node as usize;
        let creates = i32::from(dag.consumer_left(index) > 0);
        let mut kills = 0i32;
        for &producer in dag.data_predecessors(index) {
            if dag.consumer_left(producer as usize) == 1 {
                kills += 1;
            }
        }
        creates - kills
    }
}

impl SchedStrategy for RegisterPressure {
    #[inline]
    fn name(&self) -> &'static str {
        "register-pressure"
    }

    #[inline]
    fn pick(&self, dag: &ScheduleDAG, candidates: &[u16]) -> u16 {
        let mut best = candidates[0];
        let mut best_delta = Self::pressure_delta(dag, best);
        let mut best_height = dag.units[best as usize].height;
        for &candidate in &candidates[1..] {
            let delta = Self::pressure_delta(dag, candidate);
            let height = dag.units[candidate as usize].height;
            let better = delta < best_delta
                || (delta == best_delta && height > best_height)
                || (delta == best_delta && height == best_height && candidate < best);
            if better {
                best = candidate;
                best_delta = delta;
                best_height = height;
            }
        }
        best
    }
}

/// Schedules one basic block with `strategy`.
///
/// # Errors
///
/// Returns [`ScheduleError::BlockTooLarge`] for a block past the 16-bit index
/// space, [`ScheduleError::NoPorts`] for a machine model that cannot issue a
/// unit an instruction uses, and [`ScheduleError::Stalled`] if the internal
/// progress guard trips (which indicates a malformed machine model).
pub fn schedule_block<S: SchedStrategy>(
    block: &[Instruction],
    machine: &Machine,
    strategy: &S,
) -> Result<BlockSchedule, ScheduleError> {
    machine.validate()?;
    if block.len() > u16::MAX as usize {
        return Err(ScheduleError::BlockTooLarge);
    }
    if block.is_empty() {
        return Ok(BlockSchedule {
            slots: Vec::new(),
            cycles: Vec::new(),
            makespan: 0,
            nops: 0,
        });
    }

    let mut dag = ScheduleDAG::build(block, machine);
    let schedule = list_schedule(block, machine, strategy, &mut dag)?;
    Ok(fill_delay_slots(block, machine, &dag, schedule))
}

/// The top-down per-cycle list scheduler.
fn list_schedule<S: SchedStrategy>(
    block: &[Instruction],
    machine: &Machine,
    strategy: &S,
    dag: &mut ScheduleDAG,
) -> Result<BlockSchedule, ScheduleError> {
    let count = block.len();
    let width = machine.issue_width();
    let mut indegree: Vec<u16> = (0..count)
        .map(|index| dag.indegree(index))
        .collect();
    let mut ready: Vec<u16> = (0..count as u16)
        .filter(|&index| indegree[index as usize] == 0)
        .collect();
    let mut available = vec![0u16; count];
    let mut ports = [0u8; Unit::COUNT];
    let mut candidates: Vec<u16> = Vec::with_capacity(width as usize);
    let mut slots: Vec<Slot> = Vec::with_capacity(count + machine.delay_slots as usize);
    let mut cycles: Vec<u16> = Vec::with_capacity(count + machine.delay_slots as usize);

    let bound = progress_bound(block, machine);
    let mut cycle = 0u16;
    let mut remaining = count;

    while remaining > 0 {
        let mut issued = 0u8;
        while issued < width {
            candidates.clear();
            for &node in &ready {
                let index = node as usize;
                if available[index] > cycle {
                    continue;
                }
                let unit = dag.units[index].unit.index();
                if ports[unit] >= machine.units[unit].ports {
                    continue;
                }
                if block[index].is_terminator() && remaining > 1 {
                    continue;
                }
                candidates.push(node);
            }
            if candidates.is_empty() {
                break;
            }

            let node = strategy.pick(dag, &candidates);
            let index = node as usize;
            dag.units[index].cycle = cycle;
            ports[dag.units[index].unit.index()] += 1;
            issued += 1;
            remaining -= 1;
            slots.push(Slot::Instruction(node));
            cycles.push(cycle);

            if let Some(position) = ready.iter().position(|&x| x == node) {
                ready.swap_remove(position);
            }

            for dep in dag.succ(index) {
                let target = dep.target as usize;
                let ready_at = cycle.saturating_add(dep.latency);
                if ready_at > available[target] {
                    available[target] = ready_at;
                }
            }
            dag.release(index);
            for dep in dag.succ(index) {
                let target = dep.target as usize;
                indegree[target] -= 1;
                if indegree[target] == 0 {
                    ready.push(dep.target);
                }
            }
        }

        if issued == 0 && cycle >= bound {
            return Err(ScheduleError::Stalled);
        }
        cycle = cycle.saturating_add(1);
        ports = [0u8; Unit::COUNT];
    }

    let makespan = cycles
        .last()
        .map_or(0, |&last| last.saturating_add(1));
    Ok(BlockSchedule {
        slots,
        cycles,
        makespan,
        nops: 0,
    })
}

/// An upper bound on the cycles a legal schedule can need, used to detect a
/// machine model that cannot make progress.
fn progress_bound(block: &[Instruction], machine: &Machine) -> u16 {
    let mut total = 0u32;
    for instruction in block {
        total += u32::from(instruction.latency(machine)) + 1;
    }
    total += 16;
    if total > u32::from(u16::MAX) {
        u16::MAX
    } else {
        total as u16
    }
}

/// Fills architectural branch delay slots by swapping the instruction before
/// the branch into the slot, or inserting a nop when that is not legal.
fn fill_delay_slots(
    block: &[Instruction],
    machine: &Machine,
    dag: &ScheduleDAG,
    mut schedule: BlockSchedule,
) -> BlockSchedule {
    if machine.delay_slots == 0 {
        return schedule;
    }
    let Some(branch_position) = schedule
        .slots
        .iter()
        .rposition(|slot| matches!(slot, Slot::Instruction(index) if block[*index as usize].is_terminator()))
    else {
        return schedule;
    };
    let branch = match schedule.slots[branch_position] {
        Slot::Instruction(index) => index,
        Slot::Nop => return schedule,
    };
    let branch_cycle = schedule.cycles[branch_position];

    for slot_index in 0..machine.delay_slots {
        let nop_cycle = branch_cycle.saturating_add(u16::from(slot_index) + 1);
        let movable = if branch_position == 0 {
            None
        } else {
            match schedule.slots[branch_position - 1] {
                Slot::Instruction(candidate) if can_fill_delay_slot(block, dag, candidate, branch) => {
                    Some(candidate)
                }
                _ => None,
            }
        };

        match movable {
            Some(candidate) => {
                let cycle = schedule.cycles.remove(branch_position - 1);
                schedule.slots.remove(branch_position - 1);
                schedule
                    .slots
                    .insert(branch_position, Slot::Instruction(candidate));
                schedule.cycles.insert(branch_position, cycle);
            }
            None => {
                schedule
                    .slots
                    .insert(branch_position + 1, Slot::Nop);
                schedule
                    .cycles
                    .insert(branch_position + 1, nop_cycle);
                schedule.nops += 1;
            }
        }
    }
    schedule
}

/// Whether `candidate` may be moved into the delay slot of `branch`.
fn can_fill_delay_slot(block: &[Instruction], dag: &ScheduleDAG, candidate: u16, branch: u16) -> bool {
    if !block[candidate as usize]
        .flags()
        .contains(InstFlags::DELAY_SAFE)
    {
        return false;
    }
    // The instruction scheduled immediately before the branch can only feed
    // the branch; if it does, moving it after the branch would read a value
    // the branch has not produced yet.
    !dag.has_edge(candidate as usize, branch as usize) && !dag.has_edge(branch as usize, candidate as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{InstFlags, Reg, Unit};
    use crate::bare;

    fn alu(reads: &[Reg], writes: &[Reg]) -> Instruction {
        Instruction::new(Unit::Alu, InstFlags::DELAY_SAFE, 0, reads, writes).expect("fits")
    }

    #[test]
    fn independent_instructions_all_issue_in_order() {
        let block = [alu(&[], &[Reg::new(0)]), alu(&[Reg::new(0)], &[Reg::new(1)])];
        let schedule = schedule_block(&block, &bare::cortex_m(), &SourceOrder).expect("schedules");
        assert_eq!(schedule.slots.len(), 2);
        assert_eq!(schedule.cycles, vec![0, 1]);
    }

    #[test]
    fn data_latency_delays_a_dependent() {
        // Cortex-M load latency is 2, so the dependent cannot issue before 2.
        let load = Instruction::new(
            Unit::Load,
            InstFlags::READS_MEMORY | InstFlags::DELAY_SAFE,
            0,
            &[Reg::new(0)],
            &[Reg::new(1)],
        )
        .expect("fits");
        let use_it = alu(&[Reg::new(1)], &[Reg::new(2)]);
        let schedule = schedule_block(&[load, use_it], &bare::cortex_m(), &Latency).expect("schedules");
        assert_eq!(schedule.cycles[1], 2);
        assert_eq!(schedule.makespan, 3);
    }

    #[test]
    fn renaming_removes_anti_dependences() {
        // write r0; read r0; write r0 again. Without renaming the second write
        // must wait for the read (WAR); with renaming it may issue at once.
        let first = alu(&[], &[Reg::new(0)]);
        let read = alu(&[Reg::new(0)], &[Reg::new(1)]);
        let third = alu(&[], &[Reg::new(0)]);
        let block = [first, read, third];

        let in_order = schedule_block(&block, &bare::cortex_m(), &Latency).expect("bare schedule");
        // WAR: `third` cannot issue in cycle 0 alongside `read` on a single-issue
        // in-order core, and it depends on `read`.
        assert!(in_order.cycles[2] > in_order.cycles[1]);

        let ooo = schedule_block(&block, &crate::desktop::x86_64(), &Latency).expect("ooo schedule");
        // With renaming `third` has no dependence on `read`, so it issues in
        // cycle 1 at the latest, in parallel with the read.
        assert!(ooo.cycles[2] <= 1);
    }

    #[test]
    fn delay_slot_is_filled_or_nop() {
        let before = alu(&[], &[Reg::new(0)]);
        let branch = Instruction::new(
            Unit::Branch,
            InstFlags::BRANCH | InstFlags::TERMINATOR,
            0,
            &[],
            &[],
        )
        .expect("fits");
        let schedule = schedule_block(&[before, branch], &bare::mips32(), &SourceOrder).expect("schedules");
        // Branch last, and the safe ALU op is moved into the one delay slot.
        assert_eq!(schedule.slots, vec![Slot::Instruction(1), Slot::Instruction(0)]);
        assert_eq!(schedule.nops, 0);
    }

    #[test]
    fn unsafe_delay_slot_becomes_a_nop() {
        // A store carries SIDE_EFFECT and is not DELAY_SAFE.
        let store = Instruction::new(
            Unit::Store,
            InstFlags::WRITES_MEMORY | InstFlags::SIDE_EFFECT,
            0,
            &[Reg::new(0)],
            &[],
        )
        .expect("fits");
        let branch = Instruction::new(
            Unit::Branch,
            InstFlags::BRANCH | InstFlags::TERMINATOR,
            0,
            &[],
            &[],
        )
        .expect("fits");
        let schedule = schedule_block(&[store, branch], &bare::mips32(), &SourceOrder).expect("schedules");
        // The store must not move past the branch, so the slot becomes a nop:
        // [store, branch, nop].
        assert_eq!(
            schedule.slots,
            vec![Slot::Instruction(0), Slot::Instruction(1), Slot::Nop]
        );
        assert_eq!(schedule.nops, 1);
    }
}
