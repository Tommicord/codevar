//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Robust async task system inspired by async-task library.
//!
//! This module provides a task abstraction for building executors,
//! It provides a simple, cross-platform compatible task system for executing
//! async operations.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Task execution state using atomic bit flags for efficient state transitions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    /// Task is scheduled and ready to run
    Scheduled = 0x01,
    /// Task is currently running
    Running = 0x02,
    /// Task is completed with a result
    Completed = 0x04,
    /// Task was cancelled
    Cancelled = 0x08,
    /// Task handle was detached
    Detached = 0x10,
    /// Task has an active awaiter
    Awaiter = 0x20,
    /// Task is closed (both runnable and task handles dropped)
    Closed = 0x40,
}

/// Error type for task operations.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Task execution failed
    ExecutionFailed,
    /// Task was cancelled
    Cancelled,
    /// Task is still running
    StillRunning,
    /// Task was detached
    Detached,
    /// Task was closed
    Closed,
}

/// Schedule information passed to the schedule function.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScheduleInfo {
    /// Whether this is a wake-up notification
    pub wake_up: bool,
}

impl ScheduleInfo {
    /// Creates new schedule info.
    pub fn new(wake_up: bool) -> Self {
        Self { wake_up }
    }
}

/// Schedule function type for task scheduling.
pub type ScheduleFn = fn(*const (), ScheduleInfo);

/// Task state container using atomic operations for thread-safe state transitions.
#[repr(C)]
struct DaemonStateInner<T> {
    /// Current task state as atomic bit flags
    state: AtomicU8,
    /// Reference count for task lifecycle management
    ref_count: AtomicUsize,
    /// Mutex-protected result value
    result: Mutex<Option<T>>,
    /// Mutex-protected waker for async operations
    waker: Mutex<Option<Waker>>,
    /// Flag indicating if the task is detached
    detached: AtomicBool,
    /// Schedule function for the task
    schedule: Mutex<Option<ScheduleFn>>,
}

impl<T> DaemonStateInner<T> {
    /// Creates a new task state in scheduled state.
    fn new() -> Self {
        Self {
            state: AtomicU8::new(DaemonState::Scheduled as u8),
            ref_count: AtomicUsize::new(2), // Runnable + Task
            result: Mutex::new(None),
            waker: Mutex::new(None),
            detached: AtomicBool::new(false),
            schedule: Mutex::new(None),
        }
    }

    /// Increments the reference count.
    fn increment_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrements the reference count and returns whether this was the last reference.
    fn decrement_ref(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1
    }

    /// Checks if the task is scheduled.
    fn is_scheduled(&self) -> bool {
        (self.state.load(Ordering::Acquire) & DaemonState::Scheduled as u8) != 0
    }

    /// Checks if the task is running.
    fn is_running(&self) -> bool {
        (self.state.load(Ordering::Acquire) & DaemonState::Running as u8) != 0
    }

    /// Checks if the task is completed.
    fn is_completed(&self) -> bool {
        (self.state.load(Ordering::Acquire) & DaemonState::Completed as u8) != 0
    }

    /// Checks if the task was cancelled.
    fn is_cancelled(&self) -> bool {
        (self.state.load(Ordering::Acquire) & DaemonState::Cancelled as u8) != 0
    }

    /// Checks if the task is detached.
    fn is_detached(&self) -> bool {
        self.detached.load(Ordering::Acquire)
    }

    /// Checks if the task is closed.
    fn is_closed(&self) -> bool {
        (self.state.load(Ordering::Acquire) & DaemonState::Closed as u8) != 0
    }

    /// Transitions to running state.
    fn set_running(&self) -> bool {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if state & DaemonState::Closed as u8 != 0 {
                return false;
            }
            let new_state =
                (state & !(DaemonState::Scheduled as u8)) | DaemonState::Running as u8;
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(s) => state = s,
            }
        }
    }

    /// Marks the task as completed with a result.
    fn complete(&self, value: T) {
        if let Ok(mut result) = self.result.lock() {
            *result = Some(value);
        }
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            let new_state = if state & DaemonState::Closed as u8 != 0 {
                (state & !(DaemonState::Running as u8 | DaemonState::Scheduled as u8))
                    | DaemonState::Completed as u8
                    | DaemonState::Closed as u8
            } else {
                (state & !(DaemonState::Running as u8 | DaemonState::Scheduled as u8))
                    | DaemonState::Completed as u8
            };
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(s) => state = s,
            }
        }

        // Wake any waiting task
        if let Ok(mut waker) = self.waker.lock() {
            if let Some(waker) = waker.take() {
                waker.wake();
            }
        }
    }

    /// Marks the task as cancelled.
    fn cancel(&self) {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            let new_state = (state | DaemonState::Cancelled as u8)
                & !(DaemonState::Running as u8 | DaemonState::Scheduled as u8);
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(s) => state = s,
            }
        }

        // Wake any waiting task
        if let Ok(mut waker) = self.waker.lock() {
            if let Some(waker) = waker.take() {
                waker.wake();
            }
        }
    }

    /// Marks the task as detached.
    fn detach(&self) {
        self.detached.store(true, Ordering::Release);
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            let new_state = state | DaemonState::Detached as u8;
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(s) => state = s,
            }
        }
    }

    /// Marks the task as closed.
    fn close(&self) {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            let new_state = (state | DaemonState::Closed as u8)
                & !(DaemonState::Running as u8 | DaemonState::Scheduled as u8);
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(s) => state = s,
            }
        }

        // Wake any waiting task
        if let Ok(mut waker) = self.waker.lock() {
            if let Some(waker) = waker.take() {
                waker.wake();
            }
        }
    }

    /// Takes the result value if available.
    fn take_result(&self) -> Option<T> {
        if let Ok(mut result) = self.result.lock() {
            result.take()
        } else {
            None
        }
    }

    /// Sets the waker for async operations.
    fn set_waker(&self, waker: Waker) {
        if let Ok(mut waker_guard) = self.waker.lock() {
            *waker_guard = Some(waker);
        }
    }

    /// Sets the schedule function.
    fn set_schedule(&self, schedule: ScheduleFn) {
        if let Ok(mut schedule_guard) = self.schedule.lock() {
            *schedule_guard = Some(schedule);
        }
    }

    /// Calls the schedule function.
    fn call_schedule(&self, ptr: *const (), info: ScheduleInfo) {
        if let Ok(schedule_guard) = self.schedule.lock() {
            if let Some(schedule) = *schedule_guard {
                schedule(ptr, info);
            }
        }
    }
}

/// Runnable task handle for scheduling and polling.
#[repr(C)]
pub struct Runnable<T> {
    inner: Arc<DaemonStateInner<T>>,
    future: Option<Pin<Box<dyn Future<Output = T> + Send>>>,
}

impl<T> Runnable<T> {
    /// Creates a new runnable task from a future with a schedule function.
    ///
    /// # Arguments
    ///
    /// * `future` - The future to execute
    /// * `schedule` - Function to call when the task needs to be rescheduled
    ///
    /// # Returns
    ///
    /// A new runnable task and its corresponding task handle.
    pub fn new<F>(future: F, schedule: ScheduleFn) -> (Self, Daemon<T>)
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let inner = Arc::new(DaemonStateInner::new());
        inner.set_schedule(schedule);
        let task = Daemon {
            inner: inner.clone(),
        };
        let runnable = Self {
            inner,
            future: Some(Box::pin(future)),
        };
        (runnable, task)
    }

    /// Runs the task once, polling its future.
    ///
    /// # Returns
    ///
    /// `true` if the task is still running, `false` if completed or cancelled.
    pub fn run(&mut self) -> bool {
        if self.inner.is_closed()
            || self.inner.is_cancelled()
            || self.inner.is_completed()
        {
            self.inner.decrement_ref();
            return false;
        }

        if !self.inner.set_running() {
            self.inner.decrement_ref();
            return false;
        }

        if let Some(mut future) = self.future.take() {
            unsafe {
                let ptr = Arc::as_ptr(&self.inner) as *const ();
                let waker = Waker::from_raw(std::task::RawWaker::new(
                    ptr,
                    &std::task::RawWakerVTable::new(
                        Self::clone_waker,
                        Self::wake,
                        Self::wake_by_ref,
                        Self::drop_waker,
                    ),
                ));
                let mut context = Context::from_waker(&waker);

                match future.as_mut().poll(&mut context) {
                    Poll::Ready(result) => {
                        self.inner.complete(result);
                        self.future = None;
                        self.inner.decrement_ref();
                        false
                    }
                    Poll::Pending => {
                        self.future = Some(future);
                        // Reset to scheduled state for next run
                        let mut state = self.inner.state.load(Ordering::Acquire);
                        loop {
                            let new_state = if state & DaemonState::Closed as u8 != 0 {
                                state & !(DaemonState::Running as u8)
                            } else {
                                (state & !(DaemonState::Running as u8))
                                    | DaemonState::Scheduled as u8
                            };
                            match self.inner.state.compare_exchange_weak(
                                state,
                                new_state,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => break,
                                Err(s) => state = s,
                            }
                        }
                        if self.inner.is_scheduled() {
                            self.inner.call_schedule(ptr, ScheduleInfo::new(true));
                        }
                        self.inner.decrement_ref();
                        true
                    }
                }
            }
        } else {
            self.inner.decrement_ref();
            false
        }
    }

    /// Schedules the task for running.
    pub fn schedule(&self) {
        let ptr = Arc::as_ptr(&self.inner) as *const ();
        self.inner.call_schedule(ptr, ScheduleInfo::new(false));
    }

    /// Checks if the runnable is still valid.
    pub fn is_valid(&self) -> bool {
        self.future.is_some() && !self.inner.is_cancelled() && !self.inner.is_closed()
    }

    // Raw waker vtable functions
    unsafe fn clone_waker(ptr: *const ()) -> std::task::RawWaker {
        let inner = &*(ptr as *const DaemonStateInner<T>);
        inner.increment_ref();
        std::task::RawWaker::new(
            ptr,
            &std::task::RawWakerVTable::new(
                Self::clone_waker,
                Self::wake,
                Self::wake_by_ref,
                Self::drop_waker,
            ),
        )
    }

    unsafe fn wake(ptr: *const ()) {
        Self::wake_by_ref(ptr);
        Self::drop_waker(ptr);
    }

    unsafe fn wake_by_ref(ptr: *const ()) {
        let inner = &*(ptr as *const DaemonStateInner<T>);
        let mut state = inner.state.load(Ordering::Acquire);
        loop {
            let new_state = state | DaemonState::Scheduled as u8;
            match inner.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if state & DaemonState::Running as u8 == 0 {
                        inner.call_schedule(ptr, ScheduleInfo::new(true));
                    }
                    break;
                }
                Err(s) => state = s,
            }
        }
    }

    unsafe fn drop_waker(ptr: *const ()) {
        let inner = &*(ptr as *const DaemonStateInner<T>);
        if inner.decrement_ref() {
            // Last reference dropped, clean up
            if inner.is_closed() {
                // Already cleaned up
                return;
            }
        }
    }
}

impl<T> Drop for Runnable<T> {
    fn drop(&mut self) {
        self.inner.close();
        self.future = None;
    }
}

/// Task handle for awaiting results and controlling task lifecycle.
#[repr(C)]
pub struct Daemon<T> {
    inner: Arc<DaemonStateInner<T>>,
}

impl<T> Daemon<T> {
    /// Creates a new task from a future for simple use cases.
    ///
    /// # Arguments
    ///
    /// * `future` - The future to execute
    ///
    /// # Returns
    ///
    /// A new task that will execute the future synchronously on first poll.
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let inner = Arc::new(DaemonStateInner::new());

        let inner_clone = inner.clone();
        std::thread::spawn(move || {
            let waker = Waker::noop();
            let mut context = Context::from_waker(&waker);
            let mut future =
                Some(Box::pin(future) as Pin<Box<dyn Future<Output = T> + Send>>);

            loop {
                if inner_clone.is_cancelled() {
                    inner_clone.cancel();
                    inner_clone.decrement_ref();
                    return;
                }

                if let Some(ref mut fut) = future {
                    match fut.as_mut().poll(&mut context) {
                        Poll::Ready(result) => {
                            inner_clone.complete(result);
                            inner_clone.decrement_ref();
                            return;
                        }
                        Poll::Pending => {
                            std::thread::yield_now();
                        }
                    }
                } else {
                    inner_clone.decrement_ref();
                    return;
                }
            }
        });
        Self { inner }
    }

    /// Checks if the task is completed.
    ///
    /// # Returns
    ///
    /// `true` if the task has completed successfully.
    pub fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }

    /// Checks if the task is still running.
    ///
    /// # Returns
    ///
    /// `true` if the task is still executing.
    pub fn is_running(&self) -> bool {
        self.inner.is_running() || self.inner.is_scheduled()
    }

    /// Checks if the task was cancelled.
    ///
    /// # Returns
    ///
    /// `true` if the task was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// Attempts to get the result without blocking.
    ///
    /// # Returns
    ///
    /// `Ok(result)` if the task completed, `Err(DaemonError::StillRunning)` if still running,
    /// or `Err(DaemonError::Cancelled)` if the task was cancelled.
    pub fn try_get(&self) -> Result<T, Error> {
        if self.inner.is_completed() {
            if let Some(result) = self.inner.take_result() {
                Ok(result)
            } else {
                Err(Error::ExecutionFailed)
            }
        } else if self.inner.is_cancelled() {
            Err(Error::Cancelled)
        } else if self.inner.is_closed() {
            Err(Error::Closed)
        } else {
            Err(Error::StillRunning)
        }
    }

    /// Blocks until the task completes and returns the result.
    ///
    /// # Returns
    ///
    /// `Ok(result)` if the task completed successfully, or an error if the task failed.
    pub fn get(self) -> Result<T, Error> {
        loop {
            if self.inner.is_completed() {
                if let Some(result) = self.inner.take_result() {
                    return Ok(result);
                } else {
                    return Err(Error::ExecutionFailed);
                }
            } else if self.inner.is_cancelled() {
                return Err(Error::Cancelled);
            } else if self.inner.is_closed() {
                return Err(Error::Closed);
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Cancels the task if it's still running.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Detaches the task to let it keep running in the background.
    ///
    /// The task will continue running even when the Task handle is dropped.
    pub fn detach(self) {
        self.inner.detach();
        // Prevent Drop from cancelling the task
        std::mem::forget(self);
    }
}

impl<T> Drop for Daemon<T> {
    fn drop(&mut self) {
        if !self.inner.is_detached() {
            self.inner.close();
        }
        self.inner.decrement_ref();
    }
}

impl<T> Future for Daemon<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.inner.is_completed() {
            if let Some(result) = self.inner.take_result() {
                Poll::Ready(result)
            } else {
                Poll::Pending
            }
        } else if self.inner.is_cancelled() || self.inner.is_closed() {
            Poll::Pending
        } else {
            self.inner.set_waker(cx.waker().clone());
            Poll::Pending
        }
    }
}

unsafe impl<T: Send> Send for Daemon<T> {}
unsafe impl<T: Sync> Sync for Daemon<T> {}
unsafe impl<T: Send> Send for Runnable<T> {}

#[cfg(test)]
mod tests {
    use super::{Daemon, Error, Runnable, ScheduleInfo};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};
    use std::time::{Duration, Instant};

    fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        predicate()
    }

    #[test]
    fn daemon_completes_with_value() {
        let daemon = Daemon::new(async { 42u32 });
        let value = daemon.get().expect("daemon result");
        assert_eq!(value, 42);
    }

    #[test]
    fn daemon_try_get_while_running() {
        let gate = Arc::new(AtomicBool::new(false));
        let gate_clone = Arc::clone(&gate);
        let daemon = Daemon::new(async move {
            while !gate_clone.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(2));
            }
            7u8
        });

        assert!(matches!(daemon.try_get(), Err(Error::StillRunning)));
        gate.store(true, Ordering::Release);
        assert!(wait_until(Duration::from_millis(500), || {
            daemon.is_completed()
        }));
        assert_eq!(daemon.try_get().expect("ready"), 7);
    }

    #[test]
    fn daemon_cancel_before_completion() {
        let release = Arc::new(AtomicBool::new(false));
        let polled = Arc::new(AtomicUsize::new(0));
        let daemon = Daemon::new(GateFuture {
            release: Arc::clone(&release),
            value: 1,
            polled: Arc::clone(&polled),
        });
        assert!(wait_until(Duration::from_millis(200), || {
            polled.load(Ordering::Acquire) >= 1
        }));
        daemon.cancel();
        assert!(daemon.is_cancelled());
        assert!(matches!(
            daemon.get(),
            Err(Error::Cancelled) | Err(Error::Closed)
        ));
    }

    /// Future that stays pending until `release` is set, then yields `value`.
    struct GateFuture {
        release: Arc<AtomicBool>,
        value: u32,
        polled: Arc<AtomicUsize>,
    }

    impl Future for GateFuture {
        type Output = u32;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.polled.fetch_add(1, Ordering::AcqRel);
            if self.release.load(Ordering::Acquire) {
                Poll::Ready(self.value)
            } else {
                // Give the Daemon::new loop time to observe cancel between polls.
                std::thread::sleep(Duration::from_millis(1));
                Poll::Pending
            }
        }
    }

    #[test]
    fn daemon_cancel_stops_pending_future() {
        let release = Arc::new(AtomicBool::new(false));
        let polled = Arc::new(AtomicUsize::new(0));
        let daemon = Daemon::new(GateFuture {
            release: Arc::clone(&release),
            value: 99,
            polled: Arc::clone(&polled),
        });

        assert!(wait_until(Duration::from_millis(200), || {
            polled.load(Ordering::Acquire) >= 1
        }));
        daemon.cancel();
        assert!(wait_until(Duration::from_millis(200), || {
            daemon.is_cancelled()
        }));
        assert!(!daemon.is_completed());
        assert!(matches!(
            daemon.try_get(),
            Err(Error::Cancelled) | Err(Error::StillRunning)
        ));
    }

    #[test]
    fn daemon_detach_keeps_running() {
        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        let daemon = Daemon::new(async move {
            std::thread::sleep(Duration::from_millis(30));
            done_flag.store(true, Ordering::Release);
            1u32
        });
        daemon.detach();
        assert!(wait_until(Duration::from_millis(500), || {
            done.load(Ordering::Acquire)
        }));
    }

    #[test]
    fn daemon_drop_closes_without_detach() {
        let polled = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        {
            let _daemon = Daemon::new(GateFuture {
                release: Arc::clone(&release),
                value: 1,
                polled: Arc::clone(&polled),
            });
            assert!(wait_until(Duration::from_millis(200), || {
                polled.load(Ordering::Acquire) >= 1
            }));
        }
        // Dropped handle should not panic; background thread eventually exits.
        std::thread::sleep(Duration::from_millis(20));
    }

    #[test]
    fn runnable_runs_ready_future_to_completion() {
        fn noop_schedule(_ptr: *const (), _info: ScheduleInfo) {}

        let (mut runnable, daemon) = Runnable::new(async { 123u16 }, noop_schedule);
        assert!(runnable.is_valid());
        let still_running = runnable.run();
        assert!(!still_running);
        assert!(daemon.is_completed());
        assert_eq!(daemon.try_get().expect("result"), 123);
    }

    #[test]
    fn runnable_pending_then_ready() {
        fn noop_schedule(_ptr: *const (), _info: ScheduleInfo) {}

        let release = Arc::new(AtomicBool::new(false));
        let polled = Arc::new(AtomicUsize::new(0));
        let (mut runnable, daemon) = Runnable::new(
            GateFuture {
                release: Arc::clone(&release),
                value: 5,
                polled: Arc::clone(&polled),
            },
            noop_schedule,
        );

        assert!(runnable.run());
        assert!(matches!(daemon.try_get(), Err(Error::StillRunning)));
        release.store(true, Ordering::Release);
        assert!(!runnable.run());
        assert_eq!(daemon.get().expect("done"), 5);
    }

    #[test]
    fn daemon_is_running_while_pending() {
        let release = Arc::new(AtomicBool::new(false));
        let polled = Arc::new(AtomicUsize::new(0));
        let daemon = Daemon::new(GateFuture {
            release: Arc::clone(&release),
            value: 3,
            polled: Arc::clone(&polled),
        });
        assert!(wait_until(Duration::from_millis(200), || {
            daemon.is_running() || daemon.is_completed()
        }));
        release.store(true, Ordering::Release);
        assert_eq!(daemon.get().expect("value"), 3);
    }

    #[test]
    fn daemon_get_returns_execution_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let order_clone = Arc::clone(&order);
        let daemon = Daemon::new(async move {
            if let Ok(mut guard) = order_clone.lock() {
                guard.push("run");
            }
            "ok"
        });
        let result = daemon.get().expect("result");
        assert_eq!(result, "ok");
        let recorded = order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(recorded.as_slice(), ["run"]);
    }
}
