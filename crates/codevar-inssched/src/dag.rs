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

//! The scheduling dependence graph.
//!
//! This is the crate's [`ScheduleDAG`](crate::dag::ScheduleDAG): a directed
//! acyclic graph whose *nodes* are [`SUnit`]s (one per instruction, carrying
//! latency and the register-pressure bookkeeping the strategies read) and
//! whose *edges* are [`SDep`]s tagged with a [`SDepKind`] — the same shape
//! LLVM's `ScheduleDAG` uses, and the reason a single list-scheduling core can
//! drive both an in-order core and an out-of-order one.
//!
//! Edge construction is where the two machine families diverge:
//!
//! * A core **without renaming** keeps *anti*- (read-after-write) and
//!   *output*-dependences, because reusing an architectural register is a real
//!   hazard.
//! * A core **with renaming** drops them, leaving only true data dependences —
//!   which is exactly why an out-of-order scheduler has more freedom.
//!
//! Memory is modelled conservatively: without alias information, every store
//! is ordered after every earlier load and store, and every load after every
//! earlier store. Two loads never conflict. An instruction carrying
//! [`InstFlags::BARRIER`] or [`InstFlags::CALL`] is ordered against every
//! instruction before and after it: a call can read and write memory the
//! scheduler cannot see, so moving traffic across one is unsound, and that is
//! exactly the shape an I/O path has.
//!
//! Every edge points from a lower to a higher original index, so the priority
//! passes need no separate topological sort.

use alloc::vec;
use alloc::vec::Vec;

use crate::asm::{InstFlags, Instruction, Reg, Unit};
use crate::machine::Machine;

/// Why two instructions are ordered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SDepKind {
    /// True (read-after-write) dependence: the source produces a value the
    /// sink consumes. Carries the producer's latency.
    Data,
    /// Anti-dependence (write-after-read): the sink overwrites a register the
    /// source still reads. Absent on renaming cores.
    Anti,
    /// Output dependence (write-after-write): two definitions of one register.
    /// Absent on renaming cores.
    Output,
    /// Imposed ordering: memory traffic without alias information, or a
    /// barrier. Carries no data, only an order.
    Order,
}

impl SDepKind {
    /// Whether the edge represents a true value flow.
    #[inline]
    pub const fn is_data(self) -> bool {
        matches!(self, SDepKind::Data)
    }

    /// A short name for diagnostics.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            SDepKind::Data => "data",
            SDepKind::Anti => "anti",
            SDepKind::Output => "output",
            SDepKind::Order => "order",
        }
    }
}

/// One dependence edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SDep {
    /// The index of the dependent [`SUnit`].
    pub target: u16,
    /// Why the two nodes are ordered.
    pub kind: SDepKind,
    /// Cycles the sink must wait after the source issues.
    pub latency: u16,
}

/// One node of the graph: an instruction plus its scheduling state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SUnit {
    /// The block index of the instruction this unit represents.
    pub index: u16,
    /// The functional unit it executes on.
    pub unit: Unit,
    /// The latency of its result.
    pub latency: u16,
    /// Longest path from any root to this node, in cycles (bottom-up priority).
    pub depth: u16,
    /// Longest path from this node to any leaf, in cycles (top-down priority).
    pub height: u16,
    /// The issue cycle chosen by the scheduler, or `u16::MAX` while pending.
    pub cycle: u16,
}

impl SUnit {
    /// Whether the unit has been placed.
    #[inline]
    pub const fn is_scheduled(&self) -> bool {
        self.cycle != u16::MAX
    }
}

/// The dependence graph of one basic block.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScheduleDAG {
    /// The nodes, indexed by block position.
    pub units: Vec<SUnit>,
    succ: Vec<Vec<SDep>>,
    pred: Vec<Vec<SDep>>,
    /// Data predecessors of each node (producers feeding its reads).
    pub(crate) data_preds: Vec<Vec<u16>>,
    /// Number of unscheduled data successors of each node.
    pub(crate) consumer_left: Vec<u16>,
    /// Unscheduled predecessor count, for the ready list.
    pub(crate) indegree: Vec<u16>,
}

impl ScheduleDAG {
    /// Builds the graph for `block` under `machine`.
    pub fn build(block: &[Instruction], machine: &Machine) -> Self {
        let count = block.len();
        let mut dag = Self {
            units: block
                .iter()
                .enumerate()
                .map(|(index, instruction)| SUnit {
                    index: index as u16,
                    unit: instruction.unit(),
                    latency: instruction.latency(machine),
                    depth: 0,
                    height: 0,
                    cycle: u16::MAX,
                })
                .collect(),
            succ: vec![Vec::new(); count],
            pred: vec![Vec::new(); count],
            data_preds: vec![Vec::new(); count],
            consumer_left: vec![0; count],
            indegree: vec![0; count],
        };

        let registers = max_register(block) as usize + 1;
        let mut last_writer = vec![Reg::NONE; registers];
        let mut readers: Vec<Vec<u16>> = vec![Vec::new(); registers];
        let mut memory_reads: Vec<u16> = Vec::new();
        let mut last_memory_write: Option<u16> = None;
        let mut last_ordering: Option<u16> = None;
        let track_anti = !machine.renames();

        for (index, instruction) in block.iter().enumerate() {
            let current = index as u16;
            let flags = instruction.flags();

            // A barrier or a call orders everything before it and everything
            // after it. The edge to every later instruction is added when
            // that instruction is visited, by remembering the last one here.
            let orders = flags.intersects(InstFlags::ORDERING);
            if let Some(ordering) = last_ordering {
                dag.add_edge(ordering, current, SDepKind::Order, 1);
            }
            if orders {
                for earlier in 0..current {
                    dag.add_edge(earlier, current, SDepKind::Order, 1);
                }
                last_ordering = Some(current);
            }

            for &reg in instruction.reads() {
                let writer = last_writer[reg.id() as usize];
                if !writer.is_none() {
                    dag.add_data_edge(writer.id(), current, dag.units[writer.id() as usize].latency);
                }
            }

            for &reg in instruction.writes() {
                let slot = reg.id() as usize;
                let writer = last_writer[slot];
                if !writer.is_none() {
                    dag.add_edge(writer.id(), current, SDepKind::Output, 1);
                }
                if track_anti {
                    for &reader in &readers[slot] {
                        dag.add_edge(reader, current, SDepKind::Anti, 1);
                    }
                }
            }

            if flags.intersects(InstFlags::READS_MEMORY | InstFlags::WRITES_MEMORY)
                && let Some(writer) = last_memory_write
            {
                dag.add_edge(writer, current, SDepKind::Order, 1);
            }
            if flags.contains(InstFlags::READS_MEMORY) {
                memory_reads.push(current);
            }

            if flags.contains(InstFlags::WRITES_MEMORY) {
                for &reader in &memory_reads {
                    dag.add_edge(reader, current, SDepKind::Order, 1);
                }
                last_memory_write = Some(current);
                memory_reads.clear();
            }

            for &reg in instruction.reads() {
                readers[reg.id() as usize].push(current);
            }
            for &reg in instruction.writes() {
                let slot = reg.id() as usize;
                last_writer[slot] = Reg::new(current);
                readers[slot].clear();
            }
        }

        dag.finish();
        dag
    }

    /// Adds an edge, ignoring a duplicate and never lowering an existing
    /// latency.
    ///
    /// An ordering edge may be inserted before the data edge between the same
    /// pair, so a later data edge upgrades the kind: the pair is a true
    /// dependence even when a fence or a call also ordered it.
    fn add_edge(&mut self, from: u16, to: u16, kind: SDepKind, latency: u16) {
        if from == to {
            return;
        }
        if let Some(existing) = self.succ[from as usize]
            .iter_mut()
            .find(|dep| dep.target == to)
        {
            if kind.is_data() {
                existing.kind = SDepKind::Data;
            }
            if latency > existing.latency {
                existing.latency = latency;
            }
            if let Some(reverse) = self.pred[to as usize]
                .iter_mut()
                .find(|dep| dep.target == from)
            {
                if kind.is_data() {
                    reverse.kind = SDepKind::Data;
                }
                if latency > reverse.latency {
                    reverse.latency = latency;
                }
            }
            return;
        }
        self.succ[from as usize].push(SDep {
            target: to,
            kind,
            latency,
        });
        self.pred[to as usize].push(SDep {
            target: from,
            kind,
            latency,
        });
        self.indegree[to as usize] += 1;
    }

    /// Adds a data edge and records the producer/consumer relationship the
    /// register-pressure strategy reads.
    fn add_data_edge(&mut self, from: u16, to: u16, latency: u16) {
        let recorded = self.data_preds[to as usize].contains(&from);
        self.add_edge(from, to, SDepKind::Data, latency);
        if !recorded {
            self.data_preds[to as usize].push(from);
            self.consumer_left[from as usize] += 1;
        }
    }

    /// Computes the depth and height priorities once the graph is complete.
    fn finish(&mut self) {
        let count = self.units.len();
        for index in 0..count {
            let mut best = 0u16;
            for dep in &self.pred[index] {
                let candidate = self.units[dep.target as usize]
                    .depth
                    .saturating_add(dep.latency);
                if candidate > best {
                    best = candidate;
                }
            }
            self.units[index].depth = best;
        }
        for index in (0..count).rev() {
            let mut best = 0u16;
            for dep in &self.succ[index] {
                let candidate = dep
                    .latency
                    .saturating_add(self.units[dep.target as usize].height);
                if candidate > best {
                    best = candidate;
                }
            }
            self.units[index].height = best;
        }
    }

    /// The number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Whether the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// The successors of `index`.
    #[inline]
    pub fn succ(&self, index: usize) -> &[SDep] {
        &self.succ[index]
    }

    /// The predecessors of `index`.
    #[inline]
    pub fn pred(&self, index: usize) -> &[SDep] {
        &self.pred[index]
    }

    /// The number of data predecessors of `index`.
    #[inline]
    pub fn data_predecessors(&self, index: usize) -> &[u16] {
        &self.data_preds[index]
    }

    /// The number of consumers still waiting on `index`'s result.
    #[inline]
    pub fn consumer_left(&self, index: usize) -> u16 {
        self.consumer_left[index]
    }

    /// The unscheduled predecessor count of `index`.
    #[inline]
    pub fn indegree(&self, index: usize) -> u16 {
        self.indegree[index]
    }

    /// Whether a direct edge exists from `from` to `to`.
    #[inline]
    pub(crate) fn has_edge(&self, from: usize, to: usize) -> bool {
        self.succ[from]
            .iter()
            .any(|dep| dep.target as usize == to)
    }

    /// Records that `index` issued: releases one dependency of each successor
    /// and drops one pending consumer from each of its data producers.
    #[inline]
    pub(crate) fn release(&mut self, index: usize) {
        for dep in &self.succ[index] {
            let target = dep.target as usize;
            self.indegree[target] -= 1;
        }
        for &producer in &self.data_preds[index] {
            let producer = producer as usize;
            if self.consumer_left[producer] > 0 {
                self.consumer_left[producer] -= 1;
            }
        }
    }
}

/// The highest register id appearing in `block`.
fn max_register(block: &[Instruction]) -> u16 {
    let mut max = 0u16;
    for instruction in block {
        for &reg in instruction.reads() {
            max = max.max(reg.id());
        }
        for &reg in instruction.writes() {
            max = max.max(reg.id());
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::SDepKind;
    use crate::asm::{InstFlags, Instruction, Reg, Unit};
    use crate::desktop;
    use crate::sched::{Latency, schedule_block};

    fn instruction(unit: Unit, flags: InstFlags, reads: &[Reg], writes: &[Reg]) -> Instruction {
        Instruction::new(unit, flags, 0, reads, writes).expect("instruction fits")
    }

    #[test]
    fn a_fence_orders_memory_around_it() {
        // Two loads never conflict with each other, so only the fence can
        // keep them apart.
        let before = instruction(
            Unit::Load,
            InstFlags::READS_MEMORY,
            &[Reg::new(0)],
            &[Reg::new(1)],
        );
        let fence = instruction(
            Unit::System,
            InstFlags::BARRIER | InstFlags::SIDE_EFFECT,
            &[],
            &[],
        );
        let after = instruction(
            Unit::Load,
            InstFlags::READS_MEMORY,
            &[Reg::new(2)],
            &[Reg::new(3)],
        );

        let dag = super::ScheduleDAG::build(&[before, fence, after], &desktop::x86_64());
        assert!(dag.has_edge(0, 1));
        assert!(dag.has_edge(1, 2));

        let schedule =
            schedule_block(&[before, fence, after], &desktop::x86_64(), &Latency).expect("schedules");
        assert_eq!(schedule.cycles, vec![0, 1, 2]);
    }

    #[test]
    fn a_call_orders_work_around_it() {
        // The two ALU ops touch different registers; without the call edge a
        // four-wide core could start both in cycle zero.
        let before = instruction(Unit::Alu, InstFlags::NONE, &[Reg::new(3)], &[Reg::new(1)]);
        let call = instruction(Unit::Branch, InstFlags::CALL | InstFlags::SIDE_EFFECT, &[], &[]);
        let after = instruction(Unit::Alu, InstFlags::NONE, &[Reg::new(4)], &[Reg::new(2)]);

        let dag = super::ScheduleDAG::build(&[before, call, after], &desktop::x86_64());
        assert!(dag.has_edge(0, 1));
        assert!(dag.has_edge(1, 2));

        let schedule =
            schedule_block(&[before, call, after], &desktop::x86_64(), &Latency).expect("schedules");
        assert_eq!(schedule.cycles, vec![0, 1, 2]);
    }

    #[test]
    fn an_ordering_edge_never_lowers_an_existing_latency() {
        // A load feeding the call already has a data edge with the load's
        // latency; the call's order edge must not drop it to one cycle.
        let load = instruction(
            Unit::Load,
            InstFlags::READS_MEMORY,
            &[Reg::new(0)],
            &[Reg::new(1)],
        );
        let call = instruction(
            Unit::Branch,
            InstFlags::CALL | InstFlags::SIDE_EFFECT,
            &[Reg::new(1)],
            &[],
        );

        let dag = super::ScheduleDAG::build(&[load, call], &desktop::x86_64());
        let edge = dag
            .succ(0)
            .iter()
            .find(|dep| dep.target == 1)
            .expect("edge");
        assert_eq!(edge.kind, SDepKind::Data);
        assert_eq!(edge.latency, 4);
    }
}
