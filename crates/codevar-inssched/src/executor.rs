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

//! The single-core asynchronous executor.
//!
//! This is the program-level layer of the crate, and it executes real futures.
//! It owns one [`Future`] per task, builds a [`Waker`] for each of them, and
//! polls only the tasks whose waker has fired:
//!
//! 1. **Spawn.** [`Executor::spawn`] boxes the future, allocates a per-task
//!    wake signal, and puts the task on the ready queue.
//! 2. **Poll.** [`Executor::tick`] lets the [`SchedulePolicy`] choose a ready
//!    task and polls it with that task's waker.
//! 3. **Park.** A future that returns `Pending` without waking is parked; it
//!    stays out of the ready queue until a `Waker` clone is used, at which
//!    point it becomes ready again.
//! 4. **Timer.** [`Timer`] provides a virtual clock and a deadline queue, so
//!    [`Sleep`] completes deterministically without a hardware timer.
//!
//! Unlike an operating-system kernel, the executor is *cooperative*: it cannot
//! interrupt a future in the middle of a poll. A task yields exactly where its
//! code awaits. The scheduling policy therefore decides which ready task is
//! polled next, not which running task is preempted.
//!
//! The clock is virtual, not wall-clock. Each poll costs [`Config::poll_cost`]
//! virtual cycles, each dispatch costs [`Config::switch_cost`], and an idle
//! executor jumps to the next deadline. That keeps schedules reproducible and
//! lets the [`Stats`] accounting answer the same questions the old model did:
//! CPU utilization, idle cycles, waiting time, and turnaround.
//!
//! # Example
//!
//! ```rust
//! use codevar_inssched::executor::{Config, Executor};
//! use codevar_inssched::policy::RoundRobin;
//! use codevar_inssched::task::Priority;
//!
//! let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
//! let timer = executor.timer().clone();
//! executor.spawn("sleeper", Priority::NORMAL, async move {
//!     timer.sleep(2).await;
//! });
//! executor.spawn("ready", Priority::NORMAL, async {});
//!
//! let stats = executor.run()?;
//! assert_eq!(stats.completions, 2);
//! assert!(stats.polls >= 3);
//! # Ok::<(), codevar_inssched::ExecutorError>(())
//! ```

use alloc::boxed::Box;
use alloc::collections::BinaryHeap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::cmp::Ordering as CmpOrdering;
use core::fmt;
use core::future::Future;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context as TaskContext, Poll, RawWaker, RawWakerVTable, Waker};

use crate::error::ExecutorError;
use crate::policy::SchedulePolicy;
use crate::task::{Priority, Task, TaskId, TaskState};

/// Virtual-time and switch-overhead configuration of one core.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    /// Virtual CPU cycles charged for each poll.
    ///
    /// A poll is one turn of the executor: it runs the future until it awaits
    /// or completes. Charging more than one cycle models a heavier turn.
    pub poll_cost: u32,
    /// Virtual cycles lost when the executor dispatches a different task.
    pub switch_cost: u32,
}

impl Config {
    /// Creates a configuration from its poll cost and switch cost.
    #[inline]
    pub const fn new(poll_cost: u32, switch_cost: u32) -> Self {
        Self {
            poll_cost,
            switch_cost,
        }
    }
}

impl Default for Config {
    /// A one-cycle poll with a one-cycle switch: the smallest numbers that
    /// still show both effects.
    #[inline]
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// What the core did during one [`Executor::tick`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tick {
    /// Nothing was ready: the executor did not poll a task.
    Idle,
    /// The named task was polled and its future is still pending.
    Polled(TaskId),
    /// The named task's future completed.
    Finished(TaskId),
}

/// Aggregate counters collected over an executor run.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    /// Total virtual cycles elapsed.
    pub clock: u64,
    /// Tasks whose futures completed.
    pub completions: u32,
    /// Futures polled.
    pub polls: u64,
    /// Times the executor dispatched a task.
    pub context_switches: u64,
    /// Virtual cycles spent switching context.
    pub overhead_cycles: u64,
    /// Virtual cycles the core had nothing ready to poll.
    pub idle_cycles: u64,
    /// Virtual cycles spent polling futures.
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

    /// Fraction of elapsed cycles the core was polling a future, in
    /// `0.0..=1.0`.
    #[must_use]
    pub fn cpu_utilization(&self) -> f64 {
        if self.clock == 0 {
            0.0
        } else {
            self.busy_cycles as f64 / self.clock as f64
        }
    }
}

/// A single-core executor: one core, many futures, one scheduling policy.
pub struct Executor<P: SchedulePolicy> {
    policy: P,
    config: Config,
    tasks: Vec<Task>,
    jobs: Vec<Job>,
    ready: Vec<TaskId>,
    pending: Vec<TaskId>,
    last_ran: Option<TaskId>,
    timer: Timer,
    switches: u64,
    overhead: u64,
    busy: u64,
    polls: u64,
}

impl<P: SchedulePolicy> fmt::Debug for Executor<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executor")
            .field("policy", &self.policy.name())
            .field("tasks", &self.tasks.len())
            .field("ready", &self.ready.len())
            .field("pending", &self.pending.len())
            .field("clock", &self.timer.now())
            .finish()
    }
}

impl<P: SchedulePolicy> Executor<P> {
    /// Creates an empty executor for one core.
    #[must_use]
    pub fn new(policy: P, config: Config) -> Self {
        Self {
            policy,
            config,
            tasks: Vec::new(),
            jobs: Vec::new(),
            ready: Vec::new(),
            pending: Vec::new(),
            last_ran: None,
            timer: Timer::new(),
            switches: 0,
            overhead: 0,
            busy: 0,
            polls: 0,
        }
    }

    /// Spawns `future` as a task with the given name and priority.
    ///
    /// The task arrives at the current virtual cycle. Use
    /// [`spawn_with`](Self::spawn_with) when it needs an arrival cycle or a
    /// cost estimate for the scheduling policies.
    pub fn spawn<F>(&mut self, name: impl Into<String>, priority: Priority, future: F) -> TaskId
    where
        F: Future<Output = ()> + 'static,
    {
        let now = self.timer.now();
        self.spawn_with(Task::new(name, priority, 0, now), future)
    }

    /// Spawns `future` with an existing [`Task`] control block.
    ///
    /// The executor overrides the task's id; its name, priority, arrival cycle
    /// and cost estimate are kept.
    pub fn spawn_with<F>(&mut self, task: Task, future: F) -> TaskId
    where
        F: Future<Output = ()> + 'static,
    {
        let id = TaskId::new(self.tasks.len() as u32);
        let task = task.with_id(id);
        let arrival = task.arrival();
        let signal = Arc::new(TaskSignal::new());
        let waker = waker_for(&signal);
        self.tasks.push(task);
        self.jobs.push(Job {
            future: Some(Box::pin(future)),
            signal,
            waker,
        });
        if arrival <= self.timer.now() {
            if let Some(task) = self.tasks.last_mut() {
                task.set_state(TaskState::Ready);
            }
            self.ready.push(id);
        } else {
            self.pending.push(id);
        }
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

    /// The current virtual cycle.
    #[inline]
    pub fn clock(&self) -> u64 {
        self.timer.now()
    }

    /// The virtual timer and clock.
    #[inline]
    pub const fn timer(&self) -> &Timer {
        &self.timer
    }

    /// The ready queue, in its current order.
    #[inline]
    pub fn ready_queue(&self) -> &[TaskId] {
        &self.ready
    }

    /// The tasks still waiting for their arrival cycle.
    #[inline]
    pub fn pending(&self) -> &[TaskId] {
        &self.pending
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

    /// A waker that marks the task with `id` ready when used.
    ///
    /// This is the hook for an I/O reactor or a foreign event source: clone
    /// the waker into the producer, and call [`Waker::wake`] when the event
    /// arrives. The task is polled on the next [`tick`](Self::tick).
    #[inline]
    pub fn waker(&self, id: TaskId) -> Option<Waker> {
        self.jobs
            .get(id.index() as usize)
            .map(|job| job.waker.clone())
    }

    /// Marks the task with `id` ready.
    ///
    /// Returns `false` when the id is not in the task table.
    pub fn wake(&self, id: TaskId) -> bool {
        match self.jobs.get(id.index() as usize) {
            Some(job) => {
                job.signal.wake();
                true
            }
            None => false,
        }
    }

    /// Whether every task's future has completed.
    #[inline]
    pub fn all_terminated(&self) -> bool {
        self.tasks.iter().all(Task::is_terminated)
    }

    /// Advances the executor by one turn.
    ///
    /// The turn admits arrived tasks, harvests fired wakers, lets the policy
    /// choose a ready task, and polls it. A task that returns `Pending`
    /// without waking is parked; one that wakes itself stays ready.
    pub fn tick(&mut self) -> Tick {
        self.harvest();
        if self.ready.is_empty() {
            return Tick::Idle;
        }
        let now = self.timer.now();
        let picked = self.policy.pick(&self.tasks, &self.ready, now);
        let index = picked.min(self.ready.len().saturating_sub(1));
        let id = self.ready.remove(index);
        self.run_task(id)
    }

    /// Runs every task to completion and returns the aggregate statistics.
    ///
    /// When no task is ready, the executor advances the virtual clock to the
    /// next timer deadline or task arrival. If neither exists, the remaining
    /// futures can only be woken by an external event, so the run stops with
    /// [`ExecutorError::Stalled`]; use [`run_until_idle`](Self::run_until_idle)
    /// and [`waker`](Self::waker) to drive such tasks from an event loop.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::Stalled`] when a task is still unfinished and
    /// no timer, arrival or waker can make progress.
    pub fn run(&mut self) -> Result<Stats, ExecutorError> {
        loop {
            if self.all_terminated() {
                return Ok(self.stats());
            }
            match self.tick() {
                Tick::Polled(_) | Tick::Finished(_) => {}
                Tick::Idle => {
                    let now = self.timer.now();
                    let next = match (self.timer.next_deadline(), self.next_arrival()) {
                        (Some(deadline), Some(arrival)) => Some(deadline.min(arrival)),
                        (deadline, arrival) => deadline.or(arrival),
                    };
                    let Some(cycle) = next else {
                        return Err(ExecutorError::Stalled);
                    };
                    self.timer.advance_to(cycle.max(now));
                }
            }
        }
    }

    /// Polls ready tasks until none is ready, then returns the statistics.
    ///
    /// This does **not** advance the virtual clock, so parked sleepers stay
    /// parked. It is the primitive an event loop uses: call it, wait for an
    /// external event or the next [`Timer`] deadline, wake the relevant task,
    /// and call it again.
    pub fn run_until_idle(&mut self) -> Stats {
        loop {
            if matches!(self.tick(), Tick::Idle) {
                return self.stats();
            }
        }
    }

    /// The earliest arrival cycle of a task that has not been admitted.
    fn next_arrival(&self) -> Option<u64> {
        self.pending
            .iter()
            .filter_map(|id| {
                self.tasks
                    .get(id.index() as usize)
                    .map(Task::arrival)
            })
            .min()
    }

    /// Admits arrived tasks and moves woken tasks back to the ready queue.
    fn harvest(&mut self) {
        let now = self.timer.now();
        let mut index = 0;
        while index < self.pending.len() {
            let id = self.pending[index];
            let arrival = self
                .tasks
                .get(id.index() as usize)
                .map_or(u64::MAX, Task::arrival);
            if arrival > now {
                index += 1;
                continue;
            }
            self.pending.swap_remove(index);
            if let Some(task) = self.tasks.get_mut(id.index() as usize)
                && !task.is_terminated()
            {
                task.set_state(TaskState::Ready);
                self.ready.push(id);
            }
        }

        for (index, job) in self.jobs.iter().enumerate() {
            if !job.signal.take() {
                continue;
            }
            if let Some(task) = self.tasks.get_mut(index)
                && !task.is_terminated()
                && !matches!(task.state(), TaskState::New | TaskState::Ready)
            {
                task.set_state(TaskState::Ready);
                self.ready.push(TaskId::new(index as u32));
            }
        }
    }

    /// Polls one selected task and updates the accounting.
    fn run_task(&mut self, id: TaskId) -> Tick {
        let index = id.index() as usize;
        let Some(signal) = self
            .jobs
            .get(index)
            .map(|job| Arc::clone(&job.signal))
        else {
            return Tick::Idle;
        };
        let Some(waker) = self.jobs.get(index).map(|job| job.waker.clone()) else {
            return Tick::Idle;
        };
        let Some(task) = self.tasks.get_mut(index) else {
            return Tick::Idle;
        };
        task.set_state(TaskState::Running);

        if self.last_ran != Some(id) {
            self.switches = self.switches.saturating_add(1);
            self.overhead = self
                .overhead
                .saturating_add(u64::from(self.config.switch_cost));
            self.timer
                .advance_by(u64::from(self.config.switch_cost));
        }
        self.last_ran = Some(id);

        // The wake that made this task ready is consumed by this poll: the
        // future observes the event itself when it is polled.
        let _ = signal.take();

        let mut context = TaskContext::from_waker(&waker);
        let polled = match self
            .jobs
            .get_mut(index)
            .and_then(|job| job.future.as_mut())
        {
            Some(future) => future.as_mut().poll(&mut context),
            None => Poll::Ready(()),
        };

        let cost = self.config.poll_cost;
        self.polls = self.polls.saturating_add(1);
        self.busy = self.busy.saturating_add(u64::from(cost));
        self.timer.advance_by(u64::from(cost));
        if let Some(task) = self.tasks.get_mut(index) {
            task.advance(cost);
        }

        match polled {
            Poll::Ready(()) => {
                if let Some(job) = self.jobs.get_mut(index) {
                    job.future = None;
                }
                let completion = self.timer.now();
                if let Some(task) = self.tasks.get_mut(index) {
                    task.set_state(TaskState::Terminated);
                    task.set_completion(completion);
                }
                Tick::Finished(id)
            }
            Poll::Pending => {
                let woken = signal.take();
                if let Some(task) = self.tasks.get_mut(index) {
                    if woken {
                        task.set_state(TaskState::Ready);
                        self.ready.push(id);
                    } else {
                        task.set_state(TaskState::Blocked);
                    }
                }
                Tick::Polled(id)
            }
        }
    }

    /// Builds the aggregate statistics from the current task table.
    fn stats(&self) -> Stats {
        let mut completions = 0u32;
        let mut waiting_total = 0u64;
        let mut turnaround_total = 0u64;
        for task in &self.tasks {
            if let (Some(waiting), Some(turnaround)) = (task.waiting(), task.turnaround()) {
                completions = completions.saturating_add(1);
                waiting_total = waiting_total.saturating_add(waiting);
                turnaround_total = turnaround_total.saturating_add(turnaround);
            }
        }
        let clock = self.timer.now();
        let idle_cycles = clock.saturating_sub(self.busy.saturating_add(self.overhead));
        Stats {
            clock,
            completions,
            polls: self.polls,
            context_switches: self.switches,
            overhead_cycles: self.overhead,
            idle_cycles,
            busy_cycles: self.busy,
            waiting_total,
            turnaround_total,
        }
    }
}

/// One named unit of work to schedule at boot.
///
/// A convenience for the common case where an entry point wants to account for
/// its startup phases on the executor without writing a future per phase. See
/// [`schedule_phases`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Phase<'a> {
    /// A short name, for diagnostics.
    pub name: &'a str,
    /// The phase's scheduling priority.
    pub priority: Priority,
    /// The phase's estimated CPU cost, in cycles.
    pub burst: u32,
}

/// Runs a set of boot [`Phase`]s on the executor and returns the statistics.
///
/// Every phase arrives at cycle zero and is spawned as a CPU-burn future that
/// yields until its cost is charged; the executor then runs them using the
/// supplied policy. This is the entry point's shorthand for "show me how the
/// core shared these phases"; it never fails for a legal configuration.
///
/// # Errors
///
/// Propagates the executor's errors (a phase future that cannot finish).
pub fn schedule_phases<P: SchedulePolicy>(
    phases: &[Phase<'_>],
    policy: P,
    config: Config,
) -> Result<Stats, ExecutorError> {
    let mut executor = Executor::new(policy, config);
    for phase in phases {
        let polls = burn_polls(phase.burst, config.poll_cost);
        executor.spawn_with(
            Task::new(phase.name, phase.priority, phase.burst, 0),
            Burn::new(polls),
        );
    }
    executor.run()
}

/// Polls needed to charge `cycles` at `poll_cost` cycles per poll.
fn burn_polls(cycles: u32, poll_cost: u32) -> u32 {
    if poll_cost == 0 {
        u32::from(cycles > 0)
    } else {
        cycles.div_ceil(poll_cost)
    }
}

/// A future that keeps one task runnable while its cost is charged.
struct Burn {
    remaining: u32,
}

impl Burn {
    fn new(polls: u32) -> Self {
        Self {
            remaining: polls.saturating_sub(1),
        }
    }
}

impl Future for Burn {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.remaining == 0 {
            return Poll::Ready(());
        }
        this.remaining -= 1;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

/// The per-task wake signal behind a [`Waker`].
struct TaskSignal {
    ready: AtomicBool,
}

impl TaskSignal {
    fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
        }
    }

    fn wake(&self) {
        self.ready.store(true, Ordering::Release);
    }

    fn take(&self) -> bool {
        self.ready.swap(false, Ordering::AcqRel)
    }
}

static WAKER_VTABLE: RawWakerVTable =
    RawWakerVTable::new(clone_signal, wake_signal, wake_signal_by_ref, drop_signal);

/// Builds a [`Waker`] that sets `signal` when awoken.
fn waker_for(signal: &Arc<TaskSignal>) -> Waker {
    let data = Arc::into_raw(Arc::clone(signal)).cast::<()>();
    // SAFETY: `data` is an `Arc::into_raw` pointer to a live `TaskSignal`, and
    // `WAKER_VTABLE` only ever reconstructs and consumes that same type.
    unsafe { Waker::from_raw(RawWaker::new(data, &WAKER_VTABLE)) }
}

/// Clones the waker's `Arc<TaskSignal>` reference.
///
/// # Safety
///
/// `data` must be a pointer produced by `Arc::into_raw` for a `TaskSignal`.
unsafe fn clone_signal(data: *const ()) -> RawWaker {
    let signal = unsafe { Arc::from_raw(data.cast::<TaskSignal>()) };
    let cloned = Arc::clone(&signal);
    let _ = Arc::into_raw(signal);
    RawWaker::new(Arc::into_raw(cloned).cast::<()>(), &WAKER_VTABLE)
}

/// Consumes the waker's `Arc<TaskSignal>` reference and sets the signal.
///
/// # Safety
///
/// `data` must be a pointer produced by `Arc::into_raw` for a `TaskSignal`
/// whose reference is owned by this call.
unsafe fn wake_signal(data: *const ()) {
    let signal = unsafe { Arc::from_raw(data.cast::<TaskSignal>()) };
    signal.wake();
}

/// Borrows the waker's `Arc<TaskSignal>` reference and sets the signal.
///
/// # Safety
///
/// `data` must be a pointer produced by `Arc::into_raw` for a `TaskSignal`
/// that stays alive for the duration of the call.
unsafe fn wake_signal_by_ref(data: *const ()) {
    let signal = unsafe { Arc::from_raw(data.cast::<TaskSignal>()) };
    signal.wake();
    let _ = Arc::into_raw(signal);
}

/// Releases the waker's `Arc<TaskSignal>` reference.
///
/// # Safety
///
/// `data` must be a pointer produced by `Arc::into_raw` for a `TaskSignal`
/// whose reference is owned by this call.
unsafe fn drop_signal(data: *const ()) {
    drop(unsafe { Arc::from_raw(data.cast::<TaskSignal>()) });
}

/// One spawned future and its wake signal.
struct Job {
    future: Option<Pin<Box<dyn Future<Output = ()>>>>,
    signal: Arc<TaskSignal>,
    waker: Waker,
}

/// A virtual clock with a deadline-ordered queue of sleepers.
///
/// The executor owns one, and a clone can be moved into a future so the future
/// can `await` a [`Sleep`]. Deadlines are measured on the virtual clock, which
/// the executor advances as it charges polls and jumps over idle time.
#[derive(Clone)]
pub struct Timer {
    shared: Arc<SpinLock<TimerShared>>,
}

impl Timer {
    /// Creates a stopped timer at cycle zero.
    pub(crate) fn new() -> Self {
        Self {
            shared: Arc::new(SpinLock::new(TimerShared {
                clock: 0,
                sequence: 0,
                wheel: BinaryHeap::new(),
            })),
        }
    }

    /// The current virtual cycle.
    #[inline]
    pub fn now(&self) -> u64 {
        self.shared.lock().clock
    }

    /// A future that completes once `cycles` virtual cycles have elapsed.
    ///
    /// A zero-cycle sleep is ready on its first poll.
    pub fn sleep(&self, cycles: u64) -> Sleep {
        self.sleep_until(self.now().saturating_add(cycles))
    }

    /// A future that completes at the absolute virtual `deadline`.
    pub fn sleep_until(&self, deadline: u64) -> Sleep {
        Sleep {
            timer: self.clone(),
            deadline,
            registered: false,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The earliest live deadline, or `None` when nothing is sleeping.
    pub fn next_deadline(&self) -> Option<u64> {
        let mut shared = self.shared.lock();
        loop {
            match shared.wheel.peek() {
                None => return None,
                Some(timeout) if timeout.cancelled.load(Ordering::Acquire) => {}
                Some(timeout) => return Some(timeout.deadline),
            }
            let _ = shared.wheel.pop();
        }
    }

    /// Advances the clock by `cycles` and wakes every deadline that came due.
    ///
    /// Returns the number of live sleepers woken.
    pub fn advance_by(&self, cycles: u64) -> usize {
        self.advance_to(self.now().saturating_add(cycles))
    }

    /// Advances the clock to `cycle` and wakes every deadline that came due.
    ///
    /// The clock never moves backwards. Wakers run after the queue lock is
    /// released, so waking cannot deadlock against a sleeper registration.
    pub fn advance_to(&self, cycle: u64) -> usize {
        let mut woken = Vec::new();
        {
            let mut shared = self.shared.lock();
            if cycle > shared.clock {
                shared.clock = cycle;
            }
            let now = shared.clock;
            loop {
                let due = match shared.wheel.peek() {
                    Some(timeout) => timeout.deadline <= now,
                    None => false,
                };
                if !due {
                    break;
                }
                if let Some(timeout) = shared.wheel.pop()
                    && !timeout.cancelled.load(Ordering::Acquire)
                {
                    woken.push(timeout.waker);
                }
            }
        }
        let count = woken.len();
        for waker in woken {
            waker.wake();
        }
        count
    }

    /// Registers a sleeper's waker at `deadline`.
    fn register(&self, deadline: u64, waker: Waker, cancelled: Arc<AtomicBool>) {
        let mut shared = self.shared.lock();
        let sequence = shared.sequence;
        shared.sequence = shared.sequence.saturating_add(1);
        shared.wheel.push(Timeout {
            deadline,
            sequence,
            waker,
            cancelled,
        });
    }
}

impl fmt::Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Timer")
            .field("now", &self.now())
            .finish()
    }
}

/// A future that completes at a virtual deadline.
///
/// Created by [`Timer::sleep`] or [`Timer::sleep_until`]. Dropping it before
/// the deadline cancels the timer entry.
#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct Sleep {
    timer: Timer,
    deadline: u64,
    registered: bool,
    cancelled: Arc<AtomicBool>,
}

impl Future for Sleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.timer.now() >= this.deadline {
            return Poll::Ready(());
        }
        if !this.registered {
            this.timer
                .register(this.deadline, cx.waker().clone(), Arc::clone(&this.cancelled));
            this.registered = true;
        }
        Poll::Pending
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// A future that yields the core once and then completes.
///
/// Created by [`yield_now`]. It is the cooperative equivalent of a timer
/// preemption: a long-running task can call it between units of work so the
/// scheduler can poll another ready task first.
#[derive(Clone, Copy, Debug, Default)]
#[must_use = "futures do nothing unless polled"]
pub struct YieldNow {
    yielded: bool,
}

/// Yields the core once and then completes.
#[must_use = "futures do nothing unless polled"]
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

impl Future for YieldNow {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.yielded {
            return Poll::Ready(());
        }
        this.yielded = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

/// Drives a borrowed future to completion on the current thread.
///
/// Unlike [`Executor::spawn`], this accepts a future that borrows from the
/// caller. The future is polled with a real [`Waker`], so an event source can
/// resume it from another thread; `idle` runs only while the future is pending
/// *and* no wake is outstanding. A future that wakes itself is therefore
/// re-polled immediately, never after an unnecessary wait.
///
/// `idle` is the platform's blocking wait: park the thread on the event source
/// or sleep for a bounded interval and return. It must not block forever,
/// because a wake that arrives while it runs is only observed once it returns.
///
/// # Example
///
/// ```rust
/// use core::future::poll_fn;
/// use core::task::Poll;
///
/// let mut polls = 0;
/// let answer = codevar_inssched::executor::block_on(
///     poll_fn(|cx| {
///         polls += 1;
///         if polls < 2 {
///             cx.waker().wake_by_ref();
///             Poll::Pending
///         } else {
///             Poll::Ready(42)
///         }
///     }),
///     || {},
/// );
/// assert_eq!(answer, 42);
/// ```
pub fn block_on<F, I>(future: F, mut idle: I) -> F::Output
where
    F: Future,
    I: FnMut(),
{
    let mut future = core::pin::pin!(future);
    let signal = Arc::new(TaskSignal::new());
    let waker = waker_for(&signal);
    let mut context = TaskContext::from_waker(&waker);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        while !signal.take() {
            idle();
        }
    }
}

/// One registered deadline.
struct Timeout {
    deadline: u64,
    sequence: u64,
    waker: Waker,
    cancelled: Arc<AtomicBool>,
}

impl PartialEq for Timeout {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.sequence == other.sequence
    }
}

impl Eq for Timeout {}

impl PartialOrd for Timeout {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timeout {
    /// Reversed, so the earliest deadline is the heap's maximum.
    fn cmp(&self, other: &Self) -> CmpOrdering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// The virtual clock and its deadline queue.
struct TimerShared {
    clock: u64,
    sequence: u64,
    wheel: BinaryHeap<Timeout>,
}

/// A minimal spin lock for the timer queue.
///
/// The executor is single-threaded, so the lock is uncontended in normal use;
/// it exists because a cloned [`Waker`] may be used from another thread, and
/// the timer state is shared through an [`Arc`].
struct SpinLock<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: sending the lock transfers sole ownership of `T`, which is `Send`.
unsafe impl<T: Send> Send for SpinLock<T> {}

// SAFETY: every access to `T` goes through `lock`, which serializes readers
// and writers; therefore a shared reference to the lock is safe to send.
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    fn lock(&self) -> SpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        SpinGuard { lock: self }
    }
}

/// An exclusive borrow of a [`SpinLock`].
struct SpinGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> Deref for SpinGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the guard proves this thread holds the lock, so no other
        // access to `value` can overlap this shared borrow.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the guard proves this thread holds the lock exclusively, so
        // no other access to `value` can overlap this unique borrow.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Priority as PolicyPriority, RoundRobin};
    use core::future::poll_fn;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU64;

    fn log_lock(log: &Arc<Mutex<Vec<&'static str>>>, what: &'static str) {
        if let Ok(mut entries) = log.lock() {
            entries.push(what);
        }
    }

    #[test]
    fn polls_a_future_to_completion() {
        let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let id = executor.spawn("job", Priority::NORMAL, async move {
            flag.store(true, Ordering::Release);
        });
        let stats = executor.run().expect("runs");
        assert!(done.load(Ordering::Acquire));
        assert_eq!(stats.completions, 1);
        assert_eq!(stats.polls, 1);
        assert_eq!(executor.task(id).map(Task::state), Some(TaskState::Terminated));
    }

    #[test]
    fn yielding_tasks_share_the_core() {
        let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
        let log = Arc::new(Mutex::new(Vec::new()));
        for (name, entry) in [("a", "a"), ("b", "b")] {
            let log = Arc::clone(&log);
            executor.spawn(name, Priority::NORMAL, async move {
                for _ in 0..2 {
                    log_lock(&log, entry);
                    yield_now().await;
                }
            });
        }
        let stats = executor.run().expect("runs");
        assert_eq!(stats.completions, 2);
        let entries = log.lock().expect("not poisoned");
        assert_eq!(&*entries, &["a", "b", "a", "b"]);
    }

    #[test]
    fn sleep_uses_the_virtual_clock() {
        let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
        let timer = executor.timer().clone();
        let woke_at = Arc::new(AtomicU64::new(0));
        let at = Arc::clone(&woke_at);
        executor.spawn("sleeper", Priority::NORMAL, async move {
            timer.sleep(5).await;
            at.store(timer.now(), Ordering::Release);
        });
        let stats = executor.run().expect("runs");
        assert_eq!(woke_at.load(Ordering::Acquire), 5);
        assert_eq!(stats.clock, 6);
        assert_eq!(stats.idle_cycles, 4);
        assert_eq!(stats.completions, 1);
    }

    #[test]
    fn external_wake_resumes_a_parked_task() {
        let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
        let ready = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ready);
        let id = executor.spawn("io", Priority::NORMAL, async move {
            poll_fn(|_cx| {
                if flag.load(Ordering::Acquire) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
        });
        let stats = executor.run_until_idle();
        assert_eq!(stats.completions, 0);
        assert_eq!(executor.task(id).map(Task::state), Some(TaskState::Blocked));

        ready.store(true, Ordering::Release);
        assert!(executor.wake(id));
        let stats = executor.run_until_idle();
        assert_eq!(stats.completions, 1);
    }

    #[test]
    fn priority_runs_the_urgent_task_first() {
        let mut executor = Executor::new(PolicyPriority, Config::new(1, 0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let low = Arc::clone(&log);
        let high = Arc::clone(&log);
        executor.spawn("low", Priority::LOWEST, async move {
            log_lock(&low, "low");
        });
        executor.spawn("high", Priority::HIGHEST, async move {
            log_lock(&high, "high");
        });
        executor.run().expect("runs");
        let entries = log.lock().expect("not poisoned");
        assert_eq!(&*entries, &["high", "low"]);
    }

    #[test]
    fn a_parked_task_with_no_waker_stalls_the_run() {
        let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
        executor.spawn("parked", Priority::NORMAL, core::future::pending::<()>());
        assert_eq!(executor.run(), Err(ExecutorError::Stalled));
    }

    #[test]
    fn a_late_task_waits_for_its_arrival() {
        let mut executor = Executor::new(RoundRobin, Config::new(1, 0));
        executor.spawn_with(Task::new("late", Priority::NORMAL, 0, 3), async {});
        let stats = executor.run().expect("runs");
        assert_eq!(stats.completions, 1);
        assert_eq!(stats.idle_cycles, 3);
    }

    #[test]
    fn dropping_a_sleep_cancels_its_deadline() {
        let timer = Timer::new();
        drop(timer.sleep(10));
        assert_eq!(timer.next_deadline(), None);
        assert_eq!(timer.advance_to(10), 0);
    }

    #[test]
    fn block_on_does_not_idle_while_a_wake_is_pending() {
        let mut polls = 0;
        let mut idles = 0;
        let output = block_on(
            poll_fn(|cx| {
                polls += 1;
                if polls < 3 {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(polls)
                }
            }),
            || idles += 1,
        );
        assert_eq!(output, 3);
        assert_eq!(idles, 0);
    }

    #[test]
    fn block_on_resumes_on_an_external_wake() {
        let mut spawned = false;
        let output = block_on(
            poll_fn(|cx| {
                if !spawned {
                    spawned = true;
                    let waker = cx.waker().clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        waker.wake();
                    });
                    return Poll::Pending;
                }
                Poll::Ready(7)
            }),
            || std::thread::sleep(std::time::Duration::from_millis(1)),
        );
        assert_eq!(output, 7);
    }

    #[test]
    fn schedule_phases_drives_every_phase() {
        let phases = [
            Phase {
                name: "heap",
                priority: Priority::HIGHEST,
                burst: 2,
            },
            Phase {
                name: "window",
                priority: Priority::NORMAL,
                burst: 6,
            },
        ];
        let stats = schedule_phases(&phases, RoundRobin, Config::new(1, 0)).expect("runs");
        assert_eq!(stats.completions, 2);
        assert_eq!(stats.busy_cycles, 8);
    }
}
