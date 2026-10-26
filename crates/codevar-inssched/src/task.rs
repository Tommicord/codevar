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

//! The program-level unit of work: a task and its control block.
//!
//! Where [`asm::Instruction`](crate::asm::Instruction) describes one machine
//! instruction, a [`Task`] describes one *program* the single core must
//! multiplex: how much computation it needs ([`burst`](Task::burst)), how
//! urgent it is ([`priority`](Task::priority)), and the CPU state the kernel
//! saves and restores across a switch ([`Context`]).
//!
//! This is the process-control-block (PCB) layer: the kernel never inspects a
//! task's code, it only reads the numbers the scheduler needs and swaps the
//! saved context in and out of the core. That keeps the two layers of this
//! crate independent — the instruction scheduler orders *within* a program,
//! the [`kernel`](crate::kernel) orders *between* programs.

use alloc::string::String;

use crate::asm::Instruction;
use crate::error::ScheduleError;
use crate::machine::Machine;
use crate::sched::SchedStrategy;

/// Number of general-purpose registers saved in a [`Context`].
pub const REGISTER_COUNT: usize = 8;

/// Identifies a task within one [`Kernel`](crate::kernel::Kernel).
///
/// Ids are dense indices assigned by
/// [`Kernel::spawn`](crate::kernel::Kernel::spawn), so a task can be looked up
/// in O(1) and a PCB carries no pointer back to its owner.
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
/// The variants are the classic process states a PCB records: admitted but
/// not yet released ([`New`](Self::New)), on the ready queue
/// ([`Ready`](Self::Ready)), executing ([`Running`](Self::Running)), waiting
/// for an event such as I/O ([`Blocked`](Self::Blocked)), and finished
/// ([`Terminated`](Self::Terminated)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TaskState {
    /// Spawned but not yet released into the ready queue.
    New,
    /// Runnable, waiting for the core.
    Ready,
    /// Currently executing on the core.
    Running,
    /// Waiting for an external event (I/O, a timer) before it can run.
    Blocked,
    /// Finished all of its work.
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

/// The CPU state a context switch saves and restores.
///
/// A switch has to preserve the *program counter* (how far the task had run)
/// and the register file (its live values). The scheduler never interprets
/// the register contents; it only copies the block to and from the core.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Context {
    /// The logical program counter: cycles of the task already executed.
    pub program_counter: u32,
    /// The architectural register file at the moment of the switch.
    pub registers: [u64; REGISTER_COUNT],
}

impl Context {
    /// An all-zero context, as a freshly spawned task has.
    pub const EMPTY: Self = Self {
        program_counter: 0,
        registers: [0; REGISTER_COUNT],
    };
}

impl Default for Context {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

/// One schedulable program and its PCB.
///
/// Built by [`Task::new`], or from a straight-line block of instructions with
/// [`Task::burst_from_block`] so the instruction scheduler supplies the burst.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Task {
    id: TaskId,
    name: String,
    priority: Priority,
    state: TaskState,
    burst: u32,
    remaining: u32,
    slice_left: u32,
    arrival: u64,
    completion: Option<u64>,
    wake_at: Option<u64>,
    context: Context,
}

impl Task {
    /// Creates a task that needs `burst` cycles of CPU time, released at
    /// cycle `arrival`.
    ///
    /// The id is a placeholder — [`Kernel::spawn`](crate::kernel::Kernel::spawn)
    /// assigns the real one from the kernel's task table. A standalone task
    /// (in a test, or a caller's own ready queue) takes its id from
    /// [`with_id`](Self::with_id).
    #[must_use]
    pub fn new(name: impl Into<String>, priority: Priority, burst: u32, arrival: u64) -> Self {
        Self {
            id: TaskId::new(0),
            name: name.into(),
            priority,
            state: TaskState::New,
            burst,
            remaining: burst,
            slice_left: 0,
            arrival,
            completion: None,
            wake_at: None,
            context: Context::EMPTY,
        }
    }

    /// Returns the task with `id` assigned, for use outside a
    /// [`Kernel`](crate::kernel::Kernel).
    #[must_use]
    pub fn with_id(mut self, id: TaskId) -> Self {
        self.id = id;
        self
    }

    /// Estimates the burst of a straight-line program by scheduling it.
    ///
    /// The instruction scheduler's makespan is the number of cycles the block
    /// needs on `machine`, which is exactly the CPU time a program-level
    /// scheduler wants for [`burst`](Self::burst). This is the seam between
    /// the two layers: an instruction order determines the cost the kernel
    /// schedules against.
    ///
    /// # Errors
    ///
    /// Propagates the instruction scheduler's errors (an empty block yields a
    /// burst of zero).
    pub fn burst_from_block<S: SchedStrategy>(
        block: &[Instruction],
        machine: &Machine,
        strategy: &S,
    ) -> Result<u32, ScheduleError> {
        let schedule = crate::sched::schedule_block(block, machine, strategy)?;
        Ok(u32::from(schedule.makespan))
    }

    /// The id assigned by the kernel.
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

    /// Total CPU cycles the task needs to finish.
    #[inline]
    pub const fn burst(&self) -> u32 {
        self.burst
    }

    /// CPU cycles the task still has to run.
    #[inline]
    pub const fn remaining(&self) -> u32 {
        self.remaining
    }

    /// Cycles left in the task's current time slice.
    #[inline]
    pub const fn slice_left(&self) -> u32 {
        self.slice_left
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

    /// The saved CPU context.
    #[inline]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Whether the task has run to completion.
    #[inline]
    pub const fn is_terminated(&self) -> bool {
        matches!(self.state, TaskState::Terminated)
    }

    /// The cycle by which a blocked task may run again, or `None`.
    #[inline]
    pub const fn wake_at(&self) -> Option<u64> {
        self.wake_at
    }

    /// The turnaround time from arrival to completion, or `None` while the
    /// task is unfinished.
    #[inline]
    pub fn turnaround(&self) -> Option<u64> {
        self.completion
            .map(|completion| completion.saturating_sub(self.arrival))
    }

    /// The cycles the task spent waiting in the ready queue, or `None` while
    /// it is unfinished.
    ///
    /// The core is serialized, so a task's waiting is its turnaround minus the
    /// CPU time it actually consumed.
    #[inline]
    pub fn waiting(&self) -> Option<u64> {
        self.turnaround()
            .map(|turnaround| turnaround.saturating_sub(u64::from(self.burst)))
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

    /// Charges one cycle of execution to the task.
    #[inline]
    pub(crate) fn advance(&mut self, cycles: u32) {
        let executed = cycles.min(self.remaining);
        self.remaining -= executed;
        self.context.program_counter = self
            .context
            .program_counter
            .saturating_add(executed);
    }

    /// Arms the task's time slice.
    #[inline]
    pub(crate) fn set_slice_left(&mut self, slice: u32) {
        self.slice_left = slice;
    }

    /// Consumes one cycle of the time slice.
    #[inline]
    pub(crate) fn consume_slice(&mut self) {
        self.slice_left = self.slice_left.saturating_sub(1);
    }

    /// Saves a CPU context into the task.
    #[inline]
    pub(crate) fn set_context(&mut self, context: Context) {
        self.context = context;
    }

    /// Records that the task finished at cycle `at`.
    #[inline]
    pub(crate) fn set_completion(&mut self, at: u64) {
        self.completion = Some(at);
    }

    /// Marks the task as waiting for an event until cycle `at`.
    #[inline]
    pub(crate) fn set_wake_at(&mut self, at: u64) {
        self.wake_at = Some(at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{InstFlags, Reg, Unit};
    use crate::bare;
    use crate::sched::Latency;

    #[test]
    fn waiting_is_turnaround_minus_burst() {
        let mut task = Task::new("t", Priority::NORMAL, 5, 10).with_id(TaskId::new(0));
        task.set_completion(20);
        assert_eq!(task.turnaround(), Some(10));
        assert_eq!(task.waiting(), Some(5));
    }

    #[test]
    fn burst_from_block_uses_the_schedule_makespan() {
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
        let burst = Task::burst_from_block(&[load, use_it], &bare::cortex_m(), &Latency).expect("schedules");
        assert_eq!(burst, 3);
    }

    #[test]
    fn advance_never_runs_past_the_burst() {
        let mut task = Task::new("t", Priority::NORMAL, 2, 0).with_id(TaskId::new(0));
        task.advance(5);
        assert_eq!(task.remaining(), 0);
        assert_eq!(task.context().program_counter, 2);
    }
}
