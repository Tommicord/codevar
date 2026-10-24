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

//! The single-core program-level kernel.
//!
//! This is the top layer of the crate. It models one CPU core multiplexing
//! many programs, exactly the way an operating system does on a uniprocessor:
//!
//! 1. **Time allocation.** The core runs a task for at most one *quantum*
//!    (`Config::quantum` cycles) before it must give the core up.
//! 2. **Timer interrupt.** An interval [`Timer`] counts down the quantum and
//!    *fires* when it expires, which is the interrupt that ends the slice.
//! 3. **Context switch.** On a fire — or when a task blocks on I/O — the kernel
//!    saves the [`Cpu`]'s registers and program counter into the task's
//!    [`Context`] and loads the next ready task's state back into the core.
//!    A switch costs `Config::switch_cost` cycles of overhead.
//! 4. **Scheduling policy.** Which task gets the core is decided by a
//!    [`SchedulePolicy`](crate::policy::SchedulePolicy) — round-robin,
//!    priority, or shortest job first.
//!
//! The kernel is a *model*: it does not execute machine instructions, it
//! accounts for their cost. Pair it with [`Task::burst_from_block`] to derive
//! those costs from the instruction scheduler, and the two layers compose into
//! a complete picture of one core running many programs.
//!
//! # Example
//!
//! ```rust
//! use codevar_inssched::kernel::{Config, Kernel};
//! use codevar_inssched::policy::RoundRobin;
//! use codevar_inssched::task::{Priority, Task};
//!
//! let mut kernel = Kernel::new(RoundRobin, Config::default());
//! kernel.spawn(Task::new("render", Priority::NORMAL, 6, 0));
//! kernel.spawn(Task::new("compile", Priority::NORMAL, 6, 0));
//! let stats = kernel.run()?;
//! // Both tasks ran, and the timer forced at least one preemption.
//! assert_eq!(stats.completions, 2);
//! assert!(stats.context_switches >= 2);
//! # Ok::<(), codevar_inssched::error::KernelError>(())
//! ```

use alloc::vec::Vec;

use crate::error::KernelError;
use crate::policy::SchedulePolicy;
use crate::task::{Context, Priority, REGISTER_COUNT, Task, TaskId, TaskState};

/// Time-slice and switch-overhead configuration of one core.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    /// Cycles a task may run before the timer fires: its time slice, or
    /// quantum.
    pub quantum: u32,
    /// Cycles lost to one context switch.
    pub switch_cost: u32,
}

impl Config {
    /// Creates a configuration from its quantum and switch cost.
    #[inline]
    pub const fn new(quantum: u32, switch_cost: u32) -> Self {
        Self { quantum, switch_cost }
    }
}

impl Default for Config {
    /// A four-cycle slice with a one-cycle switch: the smallest numbers that
    /// still show both effects.
    #[inline]
    fn default() -> Self {
        Self::new(4, 1)
    }
}

/// The interval timer that raises the preemption interrupt.
///
/// The kernel arms it when it dispatches a task and advances it once per
/// executed cycle; [`advance`](Self::advance) returns `true` on the cycle the
/// slice expires, which is the "timer interrupt" the kernel acts on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Timer {
    quantum: u32,
    remaining: u32,
    fires: u64,
}

impl Timer {
    /// Creates a timer for `quantum`-cycle slices. A zero quantum is clamped
    /// to one so the timer always makes progress.
    #[must_use]
    pub const fn new(quantum: u32) -> Self {
        let quantum = if quantum == 0 { 1 } else { quantum };
        Self {
            quantum,
            remaining: quantum,
            fires: 0,
        }
    }

    /// The configured quantum.
    #[inline]
    pub const fn quantum(&self) -> u32 {
        self.quantum
    }

    /// Cycles left before the current slice expires.
    #[inline]
    pub const fn remaining(&self) -> u32 {
        self.remaining
    }

    /// How many times the timer has fired.
    #[inline]
    pub const fn fires(&self) -> u64 {
        self.fires
    }

    /// Updates the quantum (and re-arms the slice).
    #[inline]
    pub fn set_quantum(&mut self, quantum: u32) {
        self.quantum = if quantum == 0 { 1 } else { quantum };
        self.arm();
    }

    /// Starts a fresh slice at the configured quantum.
    #[inline]
    pub fn arm(&mut self) {
        self.remaining = self.quantum;
    }

    /// Advances one cycle; `true` when the slice expires on this cycle.
    #[inline]
    pub fn advance(&mut self) -> bool {
        self.remaining = self.remaining.saturating_sub(1);
        if self.remaining == 0 {
            self.fires += 1;
            true
        } else {
            false
        }
    }
}

impl Default for Timer {
    #[inline]
    fn default() -> Self {
        Self::new(Config::default().quantum)
    }
}

/// The single modeled core: its program counter and register file.
///
/// A context switch copies this state to and from a task's
/// [`Context`](crate::task::Context). The scheduler never interprets the
/// registers, but an embedder running real code behind a task can read and
/// write them between [`tick`](Kernel::tick) calls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cpu {
    /// The running task's logical program counter.
    pub program_counter: u32,
    /// The architecturally visible register file.
    pub registers: [u64; REGISTER_COUNT],
}

impl Cpu {
    /// A zeroed core, as it is at boot.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            program_counter: 0,
            registers: [0; REGISTER_COUNT],
        }
    }

    /// Snapshots the core state for a context switch.
    #[inline]
    pub const fn context(&self) -> Context {
        Context {
            program_counter: self.program_counter,
            registers: self.registers,
        }
    }

    /// Restores a saved context onto the core.
    #[inline]
    pub fn load(&mut self, context: &Context) {
        self.program_counter = context.program_counter;
        self.registers = context.registers;
    }

    /// Reads a register, or `0` when `index` is out of range.
    #[inline]
    pub fn register(&self, index: usize) -> u64 {
        self.registers.get(index).copied().unwrap_or(0)
    }

    /// Writes a register; out-of-range writes are ignored.
    #[inline]
    pub fn set_register(&mut self, index: usize, value: u64) {
        if let Some(slot) = self.registers.get_mut(index) {
            *slot = value;
        }
    }
}

impl Default for Cpu {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// What the core did during one [`tick`](Kernel::tick).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tick {
    /// Nothing was runnable: the core idled for one cycle.
    Idle,
    /// The core spent the cycle switching context.
    Switching,
    /// The named task executed one cycle.
    Ran(TaskId),
    /// The named task's slice expired; the timer preempted it.
    Preempted(TaskId),
    /// The named task finished all of its work.
    Finished(TaskId),
}

/// Aggregate counters collected over a [`Kernel::run`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    /// Total cycles elapsed on the modeled clock.
    pub clock: u64,
    /// Tasks that reached [`TaskState::Terminated`].
    pub completions: u32,
    /// Context switches performed.
    pub context_switches: u64,
    /// Cycles spent switching context.
    pub overhead_cycles: u64,
    /// Cycles the core had nothing to run.
    pub idle_cycles: u64,
    /// Cycles a task executed.
    pub busy_cycles: u64,
    /// Sum of every finished task's waiting time.
    pub waiting_total: u64,
    /// Sum of every finished task's turnaround time.
    pub turnaround_total: u64,
}

impl Stats {
    /// Mean cycles a finished task spent waiting, or `0` with no completions.
    #[must_use]
    pub fn average_waiting(&self) -> f64 {
        if self.completions == 0 {
            0.0
        } else {
            self.waiting_total as f64 / f64::from(self.completions)
        }
    }

    /// Mean arrival-to-completion time of a finished task, or `0` with none.
    #[must_use]
    pub fn average_turnaround(&self) -> f64 {
        if self.completions == 0 {
            0.0
        } else {
            self.turnaround_total as f64 / f64::from(self.completions)
        }
    }

    /// Fraction of elapsed cycles the core was executing a task, in `0.0..=1.0`.
    #[must_use]
    pub fn cpu_utilization(&self) -> f64 {
        if self.clock == 0 {
            0.0
        } else {
            self.busy_cycles as f64 / self.clock as f64
        }
    }
}

/// A single-core program-level kernel: one core, many tasks, one policy.
#[derive(Debug)]
pub struct Kernel<P: SchedulePolicy> {
    policy: P,
    config: Config,
    tasks: Vec<Task>,
    ready: Vec<TaskId>,
    pending: Vec<TaskId>,
    blocked: Vec<TaskId>,
    running: Option<TaskId>,
    switching: u32,
    timer: Timer,
    cpu: Cpu,
    clock: u64,
    switches: u64,
    overhead: u64,
    idle: u64,
    busy: u64,
}

impl<P: SchedulePolicy> Kernel<P> {
    /// Creates an empty kernel for one core.
    #[must_use]
    pub fn new(policy: P, config: Config) -> Self {
        Self {
            policy,
            config,
            tasks: Vec::new(),
            ready: Vec::new(),
            pending: Vec::new(),
            blocked: Vec::new(),
            running: None,
            switching: 0,
            timer: Timer::new(config.quantum),
            cpu: Cpu::new(),
            clock: 0,
            switches: 0,
            overhead: 0,
            idle: 0,
            busy: 0,
        }
    }

    /// Adds a task to the kernel and returns the id the kernel assigned.
    ///
    /// The task is admitted at its [`arrival`](Task::arrival) cycle: before
    /// then it sits in the New state, and a burst of zero finishes at once
    /// without ever taking the core.
    pub fn spawn(&mut self, task: Task) -> TaskId {
        let id = TaskId::new(self.tasks.len() as u32);
        self.tasks.push(task.with_id(id));
        self.pending.push(id);
        id
    }

    /// The scheduling policy.
    #[inline]
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// The configuration.
    #[inline]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The modeled clock, in cycles since boot.
    #[inline]
    pub const fn clock(&self) -> u64 {
        self.clock
    }

    /// The running task, if the core is busy.
    #[inline]
    pub const fn running(&self) -> Option<TaskId> {
        self.running
    }

    /// The ready queue, in policy order.
    #[inline]
    pub fn ready_queue(&self) -> &[TaskId] {
        &self.ready
    }

    /// The tasks still waiting to be admitted.
    #[inline]
    pub fn pending(&self) -> &[TaskId] {
        &self.pending
    }

    /// The tasks blocked on an event.
    #[inline]
    pub fn blocked(&self) -> &[TaskId] {
        &self.blocked
    }

    /// The modeled core.
    #[inline]
    pub const fn cpu(&self) -> &Cpu {
        &self.cpu
    }

    /// The interval timer.
    #[inline]
    pub const fn timer(&self) -> &Timer {
        &self.timer
    }

    /// The task table, indexed by [`TaskId::index`].
    #[inline]
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// The task with `id`, if it exists.
    #[inline]
    pub fn task(&self, id: TaskId) -> Option<&Task> {
        self.tasks.get(id.index() as usize)
    }

    /// Whether every task has terminated.
    #[inline]
    pub fn all_terminated(&self) -> bool {
        self.tasks.iter().all(Task::is_terminated)
    }

    /// Blocks `id` for `cycles`, modeling a wait for I/O.
    ///
    /// A blocked task leaves the ready queue and returns to it when `cycles`
    /// have elapsed (or when [`wake`](Self::wake) is called). Blocking the
    /// running task immediately releases the core and saves its context, which
    /// is the "yield while waiting" half of a context switch.
    pub fn block(&mut self, id: TaskId, cycles: u64) {
        let Some(task) = self.tasks.get(id.index() as usize) else {
            return;
        };
        if task.is_terminated() {
            return;
        }
        self.ready.retain(|&queued| queued != id);
        if self.running == Some(id) {
            let context = self.cpu.context();
            if let Some(task) = self.tasks.get_mut(id.index() as usize) {
                task.set_context(context);
            }
            self.running = None;
        }
        let wake = self.clock.saturating_add(cycles);
        if let Some(task) = self.tasks.get_mut(id.index() as usize) {
            task.set_state(TaskState::Blocked);
            task.set_wake_at(wake);
        }
        self.blocked.push(id);
    }

    /// Wakes a blocked task immediately.
    pub fn wake(&mut self, id: TaskId) {
        if self.blocked.contains(&id) {
            self.blocked.retain(|&blocked| blocked != id);
            if let Some(task) = self.tasks.get_mut(id.index() as usize) {
                task.set_state(TaskState::Ready);
                if task.burst() == 0 {
                    task.set_state(TaskState::Terminated);
                    task.set_completion(self.clock);
                } else {
                    self.ready.push(id);
                }
            }
        }
    }

    /// Advances the modeled core by one cycle.
    ///
    /// One call does exactly one of: admit a newly arrived task, burn a
    /// switching cycle, let a task execute one cycle, or idle. It is the
    /// primitive [`run`](Self::run) loops over, and the hook an embedder uses
    /// to drive the core one cycle at a time.
    pub fn tick(&mut self) -> Tick {
        self.admit();

        if self.switching > 0 {
            self.switching -= 1;
            self.clock += 1;
            self.overhead += 1;
            return Tick::Switching;
        }

        if self.running.is_none() {
            if self.ready.is_empty() {
                self.clock += 1;
                self.idle += 1;
                return Tick::Idle;
            }
            let index = self
                .policy
                .pick(&self.tasks, &self.ready, self.clock);
            let id = self.ready.remove(index);
            self.dispatch(id);
            if self.switching > 0 {
                self.switching -= 1;
                self.clock += 1;
                self.overhead += 1;
                return Tick::Switching;
            }
        }

        let Some(id) = self.running else {
            // `dispatch` left nothing running (a zero-cost context switch to a
            // task that then terminated); the cycle is idle.
            self.clock += 1;
            self.idle += 1;
            return Tick::Idle;
        };
        let index = id.index() as usize;

        if let Some(task) = self.tasks.get_mut(index) {
            task.advance(1);
            task.consume_slice();
        }
        self.cpu.program_counter = self.cpu.program_counter.saturating_add(1);
        self.clock += 1;
        self.busy += 1;
        let fired = self.timer.advance();

        let finished = self
            .tasks
            .get(index)
            .is_none_or(|task| task.remaining() == 0);
        if finished {
            let context = self.cpu.context();
            if let Some(task) = self.tasks.get_mut(index) {
                task.set_context(context);
                task.set_state(TaskState::Terminated);
                task.set_completion(self.clock);
            }
            self.running = None;
            return Tick::Finished(id);
        }

        if fired {
            let context = self.cpu.context();
            if let Some(task) = self.tasks.get_mut(index) {
                task.set_context(context);
                task.set_state(TaskState::Ready);
            }
            self.ready.push(id);
            self.running = None;
            return Tick::Preempted(id);
        }

        Tick::Ran(id)
    }

    /// Runs every task to completion and returns the aggregate statistics.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::ZeroQuantum`] when the configuration asks for a
    /// zero-cycle slice, and [`KernelError::Stalled`] if the progress guard
    /// trips (which would mean a task that can never finish).
    pub fn run(&mut self) -> Result<Stats, KernelError> {
        if self.config.quantum == 0 {
            return Err(KernelError::ZeroQuantum);
        }
        let bound = self.progress_bound();
        while !self.all_terminated() {
            if self.clock >= bound {
                return Err(KernelError::Stalled);
            }
            self.tick();
        }
        Ok(self.stats())
    }

    /// Admits newly arrived tasks and wakes blocked ones whose wait elapsed.
    fn admit(&mut self) {
        let clock = self.clock;
        let mut index = 0;
        while index < self.pending.len() {
            let id = self.pending[index];
            let arrived = self
                .tasks
                .get(id.index() as usize)
                .is_none_or(|task| task.arrival() <= clock);
            if arrived {
                self.pending.swap_remove(index);
                self.admit_one(id);
            } else {
                index += 1;
            }
        }
        let mut index = 0;
        while index < self.blocked.len() {
            let id = self.blocked[index];
            let awake = self
                .tasks
                .get(id.index() as usize)
                .is_none_or(|task| task.wake_at().is_none_or(|wake| wake <= clock));
            if awake {
                self.blocked.swap_remove(index);
                self.wake_one(id);
            } else {
                index += 1;
            }
        }
    }

    /// Moves one admitted task from New to Ready (or straight to Terminated).
    fn admit_one(&mut self, id: TaskId) {
        let Some(task) = self.tasks.get_mut(id.index() as usize) else {
            return;
        };
        if task.burst() == 0 {
            task.set_state(TaskState::Terminated);
            task.set_completion(self.clock);
        } else {
            task.set_state(TaskState::Ready);
            self.ready.push(id);
        }
    }

    /// Moves one woken task from Blocked to Ready (or to Terminated).
    fn wake_one(&mut self, id: TaskId) {
        self.admit_one(id);
    }

    /// Performs the switch-in half of a context switch.
    fn dispatch(&mut self, id: TaskId) {
        let quantum = self.config.quantum.max(1);
        let preemptive = self.policy.preemptive();
        let (slice, context) = match self.tasks.get_mut(id.index() as usize) {
            Some(task) => {
                task.set_state(TaskState::Running);
                let slice = if preemptive {
                    quantum
                } else {
                    task.remaining().max(1)
                };
                task.set_slice_left(slice);
                (slice, *task.context())
            }
            None => return,
        };
        self.cpu.load(&context);
        self.timer.set_quantum(slice);
        self.timer.arm();
        self.running = Some(id);
        self.switches += 1;
        if self.config.switch_cost > 0 {
            self.switching = self.config.switch_cost;
        }
    }

    /// Builds the aggregate statistics from the current task table.
    fn stats(&self) -> Stats {
        let mut completions = 0u32;
        let mut waiting_total = 0u64;
        let mut turnaround_total = 0u64;
        for task in &self.tasks {
            if let (Some(waiting), Some(turnaround)) = (task.waiting(), task.turnaround()) {
                completions += 1;
                waiting_total = waiting_total.saturating_add(waiting);
                turnaround_total = turnaround_total.saturating_add(turnaround);
            }
        }
        Stats {
            clock: self.clock,
            completions,
            context_switches: self.switches,
            overhead_cycles: self.overhead,
            idle_cycles: self.idle,
            busy_cycles: self.busy,
            waiting_total,
            turnaround_total,
        }
    }

    /// An upper bound on the cycles a legal schedule can need, used to detect
    /// a task that can never make progress.
    fn progress_bound(&self) -> u64 {
        let mut work = 0u64;
        let mut latest = 0u64;
        for task in &self.tasks {
            work = work.saturating_add(u64::from(task.burst()));
            latest = latest.max(task.arrival());
        }
        let switches = work
            .saturating_add(self.tasks.len() as u64)
            .saturating_mul(u64::from(self.config.switch_cost));
        latest
            .saturating_add(work)
            .saturating_add(switches)
            .saturating_add(64)
    }
}

/// A convenience kernel over the standard round-robin policy.
pub type RoundRobinKernel = Kernel<crate::policy::RoundRobin>;

impl RoundRobinKernel {
    /// Creates a round-robin kernel with `config`.
    #[must_use]
    pub fn round_robin(config: Config) -> Self {
        Kernel::new(crate::policy::RoundRobin, config)
    }
}

/// One named unit of work to schedule at boot.
///
/// A convenience for the common case where an entry point wants to account for
/// its startup phases on the modeled core without building a [`Kernel`] by
/// hand. See [`schedule_phases`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Phase<'a> {
    /// A short name, for diagnostics.
    pub name: &'a str,
    /// The phase's scheduling priority.
    pub priority: Priority,
    /// The phase's CPU cost, in cycles.
    pub burst: u32,
}

/// Schedules a set of boot [`Phase`]s on one core and returns the statistics.
///
/// Every phase arrives at cycle zero and is submitted to a fresh [`Kernel`],
/// which is then run to completion. This is the entry point's shorthand for
/// "show me how the single core shared these phases"; it never fails for a
/// legal configuration.
///
/// # Errors
///
/// Propagates the kernel's errors (a zero quantum, or a phase that cannot
/// finish).
pub fn schedule_phases<P: SchedulePolicy>(
    phases: &[Phase<'_>],
    policy: P,
    config: Config,
) -> Result<Stats, KernelError> {
    let mut kernel = Kernel::new(policy, config);
    for phase in phases {
        kernel.spawn(Task::new(phase.name, phase.priority, phase.burst, 0));
    }
    kernel.run()
}

