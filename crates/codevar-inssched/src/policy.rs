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

//! The program-level scheduling algorithms.
//!
//! A [`SchedulePolicy`] answers one question — *which ready task should run
//! next?* — and the [`Kernel`](crate::kernel::Kernel) asks it every time the
//! core frees up. This is the program-level counterpart of
//! [`SchedStrategy`](crate::sched::SchedStrategy): the instruction strategy
//! orders instructions inside one program, the policy orders programs on the
//! core.
//!
//! Three algorithms cover the classic ground:
//!
//! * [`RoundRobin`] — every task gets the same slice, in circular order. The
//!   fair default, and the only one the hardware timer alone can drive.
//! * [`Priority`] — the most urgent ready task runs first. Critical system
//!   work preempts background work.
//! * [`ShortestJobFirst`] — the shortest remaining task runs first. It is
//!   *non-preemptive*, so a running task keeps the core until it finishes,
//!   which is what makes it optimal for average waiting time.
//!
//! Every policy is a zero-sized type: selecting one costs nothing at run time
//! and a caller can supply its own by implementing the trait.

use crate::task::{Task, TaskId};

/// A priority heuristic that picks which ready task runs next.
///
/// Implementations are expected to be cheap and pure: the kernel calls
/// [`pick`](SchedulePolicy::pick) once per dispatch. `ready` is never empty,
/// and the returned value is an index into it.
pub trait SchedulePolicy {
    /// A short name for diagnostics.
    fn name(&self) -> &'static str;

    /// Chooses one of `ready` (indices into `tasks`) to run next.
    fn pick(&self, tasks: &[Task], ready: &[TaskId], now: u64) -> usize;

    /// Whether the running task may be preempted when its time slice expires.
    ///
    /// A preemptive policy is driven by the timer; a non-preemptive one runs
    /// its choice to completion, so the kernel arms the whole remaining burst
    /// as its slice.
    #[inline]
    fn preemptive(&self) -> bool {
        true
    }
}

/// The task a ready entry refers to.
#[inline]
fn entry<'a>(tasks: &'a [Task], id: &TaskId) -> &'a Task {
    &tasks[id.index() as usize]
}

/// Round Robin: the head of the ready queue, so a task that was preempted goes
/// to the back and every task gets an equal share of the core.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundRobin;

impl SchedulePolicy for RoundRobin {
    #[inline]
    fn name(&self) -> &'static str {
        "round-robin"
    }

    #[inline]
    fn pick(&self, _tasks: &[Task], _ready: &[TaskId], _now: u64) -> usize {
        0
    }
}

/// Priority scheduling: the most urgent ready task runs, breaking ties by
/// earlier arrival and then by lower id so the choice is deterministic.
#[derive(Clone, Copy, Debug, Default)]
pub struct Priority;

impl SchedulePolicy for Priority {
    #[inline]
    fn name(&self) -> &'static str {
        "priority"
    }

    #[inline]
    fn pick(&self, tasks: &[Task], ready: &[TaskId], _now: u64) -> usize {
        let mut best = 0usize;
        let mut best_task = entry(tasks, &ready[0]);
        for (index, id) in ready.iter().enumerate().skip(1) {
            let candidate = entry(tasks, id);
            if better_priority(candidate, best_task) {
                best = index;
                best_task = candidate;
            }
        }
        best
    }
}

/// Compares two tasks under the priority policy.
#[inline]
fn better_priority(candidate: &Task, incumbent: &Task) -> bool {
    candidate.priority() > incumbent.priority()
        || (candidate.priority() == incumbent.priority()
            && (candidate.arrival() < incumbent.arrival()
                || (candidate.arrival() == incumbent.arrival() && candidate.id() < incumbent.id())))
}

/// Shortest Job First: the ready task with the least remaining work runs
/// first. Non-preemptive, so a dispatched task keeps the core to completion.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShortestJobFirst;

impl SchedulePolicy for ShortestJobFirst {
    #[inline]
    fn name(&self) -> &'static str {
        "shortest-job-first"
    }

    #[inline]
    fn pick(&self, tasks: &[Task], ready: &[TaskId], _now: u64) -> usize {
        let mut best = 0usize;
        let mut best_task = entry(tasks, &ready[0]);
        for (index, id) in ready.iter().enumerate().skip(1) {
            let candidate = entry(tasks, id);
            if better_shortest(candidate, best_task) {
                best = index;
                best_task = candidate;
            }
        }
        best
    }

    #[inline]
    fn preemptive(&self) -> bool {
        false
    }
}

/// Compares two tasks under the shortest-job policy.
#[inline]
fn better_shortest(candidate: &Task, incumbent: &Task) -> bool {
    candidate.remaining() < incumbent.remaining()
        || (candidate.remaining() == incumbent.remaining()
            && (candidate.arrival() < incumbent.arrival()
                || (candidate.arrival() == incumbent.arrival() && candidate.id() < incumbent.id())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Priority as Prio;
    use alloc::vec;
    use alloc::vec::Vec;

    fn tasks() -> Vec<Task> {
        vec![
            Task::new("a", Prio::new(10), 5, 0).with_id(TaskId::new(0)),
            Task::new("b", Prio::new(200), 5, 0).with_id(TaskId::new(1)),
            Task::new("c", Prio::new(10), 2, 0).with_id(TaskId::new(2)),
        ]
    }

    #[test]
    fn round_robin_takes_the_head() {
        let tasks = tasks();
        let ready = [TaskId::new(2), TaskId::new(0)];
        assert_eq!(RoundRobin.pick(&tasks, &ready, 0), 0);
    }

    #[test]
    fn priority_takes_the_most_urgent() {
        let tasks = tasks();
        let ready = [TaskId::new(0), TaskId::new(1), TaskId::new(2)];
        // Task 1 has the highest priority.
        assert_eq!(Priority.pick(&tasks, &ready, 0), 1);
    }

    #[test]
    fn shortest_job_takes_the_least_remaining() {
        let tasks = tasks();
        let ready = [TaskId::new(0), TaskId::new(1), TaskId::new(2)];
        // Task 2 has the shortest burst and is non-preemptive.
        assert_eq!(ShortestJobFirst.pick(&tasks, &ready, 0), 2);
        assert!(!ShortestJobFirst.preemptive());
        assert!(RoundRobin.preemptive());
    }
}
