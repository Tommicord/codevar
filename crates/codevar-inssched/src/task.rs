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

//! The program-level unit of work: an asynchronous task and its control block.
//!
//! Where [`asm::Instruction`](crate::asm::Instruction) describes one machine
//! instruction, a [`Task`] describes one *future* the single core must
//! multiplex: its scheduling metadata (name, [`priority`](Task::priority),
//! arrival, cost estimate) and the accounting the executor keeps across polls.
//!
//! The future itself lives in the executor, not in the [`Task`]. Keeping the
//! metadata separate is what lets the scheduling policies stay independent of
//! the future's concrete type: a policy reads numbers, never the future.

use alloc::string::String;

use crate::asm::Instruction;
use crate::error::ScheduleError;
use crate::machine::Machine;
use crate::sched::SchedStrategy;

/// Identifies a task within one [`Executor`](crate::executor::Executor).
///
/// Ids are dense indices assigned by
/// [`Executor::spawn`](crate::executor::Executor::spawn), so a task can be
/// looked up in O(1) and a control block carries no pointer back to its owner.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct TaskId(u32);

impl TaskId {
    /// Creates an id from its dense index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The dense index this id wraps.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A task's scheduling priority: a larger level is more urgent.
///
/// Priority is an *ordering*, not a cost, so it is a newtype over `u8` rather
/// than a bare integer: the three named levels cover the usual system /
/// interactive / background split, and any other value is still expressible.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct Priority(u8);

impl Priority {
    /// The least urgent level, for background work.
    pub const LOWEST: Self = Self(0);
    /// The default level for ordinary work.
    pub const NORMAL: Self = Self(128);
    /// The most urgent level, for critical system work.
    pub const HIGHEST: Self = Self(u8::MAX);

    /// Creates a priority from its raw level.
    #[inline]
    pub const fn new(level: u8) -> Self {
        Self(level)
    }

    /// The raw level.
    #[inline]
    pub const fn level(self) -> u8 {
        self.0
    }
}

impl Default for Priority {
    #[inline]
    fn default() -> Self {
        Self::NORMAL
    }
}

/// The lifecycle state of a task.
///
/// These are the states a real executor's task table records: spawned but not
/// yet released ([`New`](Self::New)), runnable ([`Ready`](Self::Ready)),
/// being polled ([`Running`](Self::Running)), parked because its future
/// returned `Pending` without a wake ([`Blocked`](Self::Blocked)), and
/// finished ([`Terminated`](Self::Terminated)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TaskState {
    /// Spawned, but its arrival cycle has not been reached.
    New,
    /// Runnable: its future may be polled now.
    Ready,
    /// Currently being polled on the core.
    Running,
    /// Parked: the future returned `Pending` and no wake is pending.
    Blocked,
    /// The future returned `Ready` and the task finished.
    Terminated,
}

impl TaskState {
    /// A short stable name for diagnostics.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Terminated => "terminated",
        }
    }
}

/// The schedulable metadata and accounting of one asynchronous task.
///
/// Built by [`Task::new`], or with a cost derived from a straight-line block
/// via [`Task::cost_from_block`]. The future is supplied separately when the
/// task is spawned on an [`Executor`](crate::executor::Executor).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Task {
    id: TaskId,
    name: String,
    priority: Priority,
    state: TaskState,
    cost: u32,
    remaining: u32,
    polls: u32,
    cpu: u64,
    arrival: u64,
    completion: Option<u64>,
}

impl Task {
    /// Creates a task with an estimated cost of `cost` cycles, released at
    /// cycle `arrival`.
    ///
    /// `cost` is a scheduling *hint*, used by heuristics such as
    /// [`ShortestJobFirst`](crate::policy::ShortestJobFirst) and by the
    /// executor's accounting; the future itself decides when it is done. Use
    /// [`cost_from_block`](Self::cost_from_block) when the work is a known
    /// straight-line block, or `0` when no estimate exists.
    ///
    /// The id is a placeholder — [`Executor::spawn_with`] assigns the real one
    /// from the executor's task table. A standalone task (in a test, or a
    /// caller's own queue) takes its id from [`with_id`](Self::with_id).
    ///
    /// [`Executor::spawn_with`]: crate::executor::Executor::spawn_with
    #[must_use]
    pub fn new(name: impl Into<String>, priority: Priority, cost: u32, arrival: u64) -> Self {
        Self {
            id: TaskId::new(0),
            name: name.into(),
            priority,
            state: TaskState::New,
            cost,
            remaining: cost,
            polls: 0,
            cpu: 0,
            arrival,
            completion: None,
        }
    }

    /// Returns the task with `id` assigned, for use outside an
    /// [`Executor`](crate::executor::Executor).
    #[must_use]
    pub fn with_id(mut self, id: TaskId) -> Self {
        self.id = id;
        self
    }

    /// Estimates the cost of a straight-line program by scheduling it.
    ///
    /// The instruction scheduler's makespan is the number of cycles the block
    /// needs on `machine`, which is the natural cost estimate a program-level
    /// scheduler wants. This is the seam between the two layers: an
    /// instruction order supplies the number the executor accounts against.
    ///
    /// # Errors
    ///
    /// Propagates the instruction scheduler's errors (an empty block yields a
    /// cost of zero).
    pub fn cost_from_block<S: SchedStrategy>(
        block: &[Instruction],
        machine: &Machine,
        strategy: &S,
    ) -> Result<u32, ScheduleError> {
        let schedule = crate::sched::schedule_block(block, machine, strategy)?;
        Ok(u32::from(schedule.makespan))
    }

    /// Estimates the cost of a multi-block function by scheduling every block.
    ///
    /// Control flow splits the instructions into basic blocks; each block is
    /// scheduled independently and the resulting makespans are summed. The sum
    /// is one pass over the static code, which is the right estimate for a
    /// task whose cost is dominated by a loop body or a long straight run.
    ///
    /// # Errors
    ///
    /// Propagates the instruction scheduler's errors (an empty function yields
    /// a cost of zero).
    pub fn cost_from_function<S: SchedStrategy>(
        function: &crate::decode::Function,
        machine: &Machine,
        strategy: &S,
    ) -> Result<u32, ScheduleError> {
        let schedule = crate::decode::schedule_function(function, machine, strategy)?;
        Ok(schedule.makespan())
    }

    /// The id assigned by the executor.
    #[inline]
    pub const fn id(&self) -> TaskId {
        self.id
    }

    /// The task's name, for diagnostics.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The task's scheduling priority.
    #[inline]
    pub const fn priority(&self) -> Priority {
        self.priority
    }

    /// The current lifecycle state.
    #[inline]
    pub const fn state(&self) -> TaskState {
        self.state
    }

    /// The estimated CPU cost the task declared when it was spawned.
    #[inline]
    pub const fn cost(&self) -> u32 {
        self.cost
    }

    /// The estimated CPU cost not yet charged to the task's polls.
    #[inline]
    pub const fn remaining(&self) -> u32 {
        self.remaining
    }

    /// How many times the task's future has been polled.
    #[inline]
    pub const fn polls(&self) -> u32 {
        self.polls
    }

    /// Virtual CPU cycles charged to the task's polls.
    #[inline]
    pub const fn cpu(&self) -> u64 {
        self.cpu
    }

    /// The cycle at which the task becomes runnable.
    #[inline]
    pub const fn arrival(&self) -> u64 {
        self.arrival
    }

    /// The cycle at which the task finished, once it has.
    #[inline]
    pub const fn completion(&self) -> Option<u64> {
        self.completion
    }

    /// Whether the task's future has completed.
    #[inline]
    pub const fn is_terminated(&self) -> bool {
        matches!(self.state, TaskState::Terminated)
    }

    /// The turnaround time from arrival to completion, or `None` while the
    /// task is unfinished.
    #[inline]
    pub fn turnaround(&self) -> Option<u64> {
        self.completion
            .map(|completion| completion.saturating_sub(self.arrival))
    }

    /// The virtual cycles the task spent waiting, or `None` while it is
    /// unfinished.
    ///
    /// The core is serialized, so a task's waiting is its turnaround minus the
    /// CPU cycles charged to its polls.
    #[inline]
    pub fn waiting(&self) -> Option<u64> {
        self.turnaround()
            .map(|turnaround| turnaround.saturating_sub(self.cpu))
    }

    /// Overrides the task's priority.
    #[inline]
    pub fn set_priority(&mut self, priority: Priority) {
        self.priority = priority;
    }

    /// Records the task's lifecycle state.
    #[inline]
    pub(crate) fn set_state(&mut self, state: TaskState) {
        self.state = state;
    }

    /// Charges one poll of `cost` cycles to the task.
    #[inline]
    pub(crate) fn advance(&mut self, cost: u32) {
        self.polls = self.polls.saturating_add(1);
        self.cpu = self.cpu.saturating_add(u64::from(cost));
        self.remaining = self.remaining.saturating_sub(cost);
    }

    /// Records that the task finished at cycle `at`.
    #[inline]
    pub(crate) fn set_completion(&mut self, at: u64) {
        self.completion = Some(at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{InstFlags, Reg, Unit};
    use crate::bare;
    use crate::sched::Latency;

    #[test]
    fn waiting_is_turnaround_minus_cpu() {
        let mut task = Task::new("t", Priority::NORMAL, 5, 10).with_id(TaskId::new(0));
        task.advance(2);
        task.set_completion(20);
        assert_eq!(task.turnaround(), Some(10));
        assert_eq!(task.waiting(), Some(8));
    }

    #[test]
    fn cost_from_block_uses_the_schedule_makespan() {
        let load = Instruction::new(
            Unit::Load,
            InstFlags::READS_MEMORY,
            0,
            &[Reg::new(0)],
            &[Reg::new(1)],
        )
        .expect("fits");
        let use_it =
            Instruction::new(Unit::Alu, InstFlags::NONE, 0, &[Reg::new(1)], &[Reg::new(2)]).expect("fits");
        // The dependent ALU op waits for the load: makespan 3 cycles.
        let cost = Task::cost_from_block(&[load, use_it], &bare::cortex_m(), &Latency).expect("schedules");
        assert_eq!(cost, 3);
    }

    #[test]
    fn advance_never_goes_below_zero_remaining() {
        let mut task = Task::new("t", Priority::NORMAL, 2, 0).with_id(TaskId::new(0));
        task.advance(5);
        assert_eq!(task.remaining(), 0);
        assert_eq!(task.cpu(), 5);
        assert_eq!(task.polls(), 1);
    }
}
