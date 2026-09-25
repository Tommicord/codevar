//! Copyright 2026 Codevar Project
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

//! Event loop for Wayland client
//!
//! The loop multiplexes file descriptor sources, timers, idle tasks and
//! re-checkable sources on top of two pluggable clocks: a [`WlPoller`]
//! waits for readiness and a [`WlClock`] supplies monotonic time. Because
//! the crate runs without an operating system socket layer neither an
//! embedded loop descriptor nor signal sources are provided; both are
//! documented deviations from libwayland.
//!
//! # Dispatch order
//!
//! [`WlEventLoop::dispatch`] mirrors `wl_event_loop_dispatch`: idle tasks
//! run first, then the poller waits, expired timers fire before file
//! descriptor sources, idle tasks run again and finally every source
//! marked with [`WlEventLoop::check`] is dispatched until all of its
//! callbacks return zero.

use core::time::Duration;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use crate::wl_conn::WlHandle;
use crate::wl_error::{WlError, WlResult};

/// Set of event bits as returned by pollers and transports.
pub use crate::wl_handle::WlPollEvents;

type TimerCallback<P, C> = Box<dyn FnMut(&mut WlEventLoop<P, C>, WlEventSourceId) -> i32>;
type IdleCallback<P, C> = Box<dyn FnMut(&mut WlEventLoop<P, C>, WlEventSourceId)>;

/// Single descriptor handed to [`WlPoller::poll`].
///
/// The event loop fills in `handle` and `interest` and expects the poller
/// to report readiness through `revents`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WlPollEntry {
    /// Opaque handle of the polled transport.
    pub handle: WlHandle,
    /// Events the source is interested in.
    pub interest: WlPollEvents,
    /// Events reported by the poller; empty when nothing is ready.
    pub revents: WlPollEvents,
}

/// Waits for readiness of the sources of an event loop.
///
/// Implementations wrap `epoll`, a condition variable over in-memory
/// transports or any other readiness mechanism. The call receives every
/// file descriptor source of the loop and must fill in `revents` for the
/// entries that are ready. `timeout` bounds the wait: `None` blocks until
/// an entry is ready, `Some(Duration::ZERO)` polls without blocking.
pub trait WlPoller {
    /// Waits until one of `entries` is ready or `timeout` elapses.
    ///
    /// Returns the number of entries with a non-empty `revents` mask.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Io`] when the underlying wait fails and
    /// [`WlError::Disconnected`] when the poller is gone.
    fn poll(&mut self, entries: &mut [WlPollEntry], timeout: Option<Duration>) -> WlResult<usize>;
}

/// Monotonic time source for timers.
///
/// The value only has to increase and is expressed in milliseconds.
pub trait WlClock {
    /// Returns the current monotonic time in milliseconds.
    fn now_ms(&self) -> u64;
}

/// Stable identifier of an event source.
///
/// Identifiers are reused after a source is removed, so an id is only
/// valid while the generation it was created with is current. Every
/// operation on the loop validates both halves and rejects stale ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WlEventSourceId {
    index: usize,
    generation: u32,
}

impl WlEventSourceId {
    /// Returns the slot index of the source.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// Returns the generation of the source slot.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Callback of a file descriptor source.
///
/// `events` carries the readiness reported by the poller, or
/// [`WlPollEvents::EMPTY`] when the source was dispatched through
/// [`WlEventLoop::check`]. Returning non-zero keeps the source on the
/// check list.
type FdFunc<P, C> = dyn FnMut(&mut WlEventLoop<P, C>, WlEventSourceId, WlPollEvents) -> i32;

enum WlSourceKind<P, C> {
    Fd {
        handle: WlHandle,
        interest: WlPollEvents,
        callback: Option<Box<FdFunc<P, C>>>,
    },
    Timer {
        deadline: Option<u64>,
        callback: Option<TimerCallback<P, C>>,
    },
    Idle {
        callback: Option<IdleCallback<P, C>>,
    },
}

enum WlSourceSlot<P, C> {
    Free {
        generation: u32,
    },
    Live {
        generation: u32,
        kind: WlSourceKind<P, C>,
    },
}

/// Event loop with file descriptor, timer, idle and check sources.
///
/// The loop owns a poller `P` and a clock `C`; both are reachable through
/// accessors so callers can share state with the callbacks.
pub struct WlEventLoop<P, C> {
    poller: P,
    clock: C,
    slots: Vec<WlSourceSlot<P, C>>,
    free: Vec<usize>,
    idle: VecDeque<WlEventSourceId>,
    check: Vec<WlEventSourceId>,
}

impl<P: WlPoller, C: WlClock> WlEventLoop<P, C> {
    /// Creates an empty event loop over `poller` and `clock`.
    #[must_use]
    pub fn new(poller: P, clock: C) -> Self {
        Self {
            poller,
            clock,
            slots: Vec::new(),
            free: Vec::new(),
            idle: VecDeque::new(),
            check: Vec::new(),
        }
    }

    /// Returns the poller.
    #[inline]
    #[must_use]
    pub const fn poller(&self) -> &P {
        &self.poller
    }

    /// Mutably returns the poller.
    #[inline]
    #[must_use]
    pub fn poller_mut(&mut self) -> &mut P {
        &mut self.poller
    }

    /// Returns the clock.
    #[inline]
    #[must_use]
    pub const fn clock(&self) -> &C {
        &self.clock
    }

    /// Mutably returns the clock.
    #[inline]
    #[must_use]
    pub fn clock_mut(&mut self) -> &mut C {
        &mut self.clock
    }

    /// Watches `handle` for `interest` events.
    ///
    /// Unlike libwayland the loop does not duplicate the handle: the
    /// transport that owns it stays responsible for closing it.
    pub fn add_fd<F>(&mut self, handle: WlHandle, interest: WlPollEvents, callback: F) -> WlEventSourceId
    where
        F: FnMut(&mut Self, WlEventSourceId, WlPollEvents) -> i32 + 'static,
    {
        self.alloc_slot(WlSourceKind::Fd {
            handle,
            interest,
            callback: Some(Box::new(callback)),
        })
    }

    /// Adds a timer source.
    ///
    /// Timers start disarmed and fire once after [`Self::timer_update`]
    /// armed them.
    pub fn add_timer<F>(&mut self, callback: F) -> WlEventSourceId
    where
        F: FnMut(&mut Self, WlEventSourceId) -> i32 + 'static,
    {
        self.alloc_slot(WlSourceKind::Timer {
            deadline: None,
            callback: Some(Box::new(callback)),
        })
    }

    /// Adds an idle task.
    ///
    /// Idle tasks run before the loop waits, again after sources were
    /// dispatched, and are removed as soon as they ran.
    pub fn add_idle<F>(&mut self, callback: F) -> WlEventSourceId
    where
        F: FnMut(&mut Self, WlEventSourceId) + 'static,
    {
        let id = self.alloc_slot(WlSourceKind::Idle {
            callback: Some(Box::new(callback)),
        });
        self.idle.push_back(id);
        id
    }

    /// Marks `id` to be re-checked after every dispatch.
    ///
    /// Checked sources are dispatched with an empty event mask until all
    /// of their callbacks return zero, which drains events that arrived
    /// as a side effect of other dispatches.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] when `id` is stale or refers to
    /// an idle source.
    pub fn check(&mut self, id: WlEventSourceId) -> WlResult<()> {
        match self.slot(id) {
            Some(WlSourceKind::Fd { .. }) | Some(WlSourceKind::Timer { .. }) => {}
            Some(WlSourceKind::Idle { .. }) => {
                return Err(WlError::InvalidState(String::from(
                    "idle sources cannot be checked",
                )));
            }
            None => return Err(stale_source()),
        }
        if !self.check.contains(&id) {
            self.check.push(id);
        }
        Ok(())
    }

    /// Changes the events watched by a file descriptor source.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] when `id` is stale or does not
    /// refer to a file descriptor source.
    pub fn fd_update(&mut self, id: WlEventSourceId, interest: WlPollEvents) -> WlResult<()> {
        match self.slot_mut(id) {
            Some(WlSourceKind::Fd {
                interest: current, ..
            }) => {
                *current = interest;
                Ok(())
            }
            Some(_) => Err(WlError::InvalidState(String::from(
                "source is not a file descriptor source",
            ))),
            None => Err(stale_source()),
        }
    }

    /// Arms or disarms a timer.
    ///
    /// `milliseconds` of zero disarms the timer, any other value lets it
    /// fire once after the given number of milliseconds from now.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] when `id` is stale or does not
    /// refer to a timer.
    pub fn timer_update(&mut self, id: WlEventSourceId, milliseconds: u32) -> WlResult<()> {
        let now = self.clock.now_ms();
        match self.slot_mut(id) {
            Some(WlSourceKind::Timer { deadline, .. }) => {
                *deadline = if milliseconds == 0 {
                    None
                } else {
                    Some(now + u64::from(milliseconds))
                };
                Ok(())
            }
            Some(_) => Err(WlError::InvalidState(String::from("source is not a timer"))),
            None => Err(stale_source()),
        }
    }

    /// Removes `id` from the loop.
    ///
    /// The callback of the source is dropped and the identifier becomes
    /// stale.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] when `id` is stale.
    pub fn remove_source(&mut self, id: WlEventSourceId) -> WlResult<()> {
        if self.free_source(id) {
            Ok(())
        } else {
            Err(stale_source())
        }
    }

    /// Returns `true` while `id` refers to a live source.
    #[must_use]
    pub fn is_active(&self, id: WlEventSourceId) -> bool {
        self.slot(id).is_some()
    }

    /// Runs all pending idle tasks.
    ///
    /// Every idle source is dispatched once and removed afterwards; tasks
    /// queued while draining run in the same pass.
    pub fn dispatch_idle(&mut self) {
        while let Some(id) = self.idle.pop_front() {
            let Some(mut callback) = self.take_idle(id) else {
                continue;
            };
            callback(&mut *self, id);
            self.free_source(id);
        }
    }

    /// Waits for events and dispatches the ready sources.
    ///
    /// `timeout` bounds the wait; `None` blocks until a source or an
    /// armed timer is ready. Idle tasks run before the wait, timers fire
    /// before file descriptor sources, idle tasks run again and checked
    /// sources are re-dispatched until every callback returns zero.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Io`] when the poller fails.
    pub fn dispatch(&mut self, timeout: Option<Duration>) -> WlResult<()> {
        self.dispatch_idle();

        let now = self.clock.now_ms();
        let wait = match (timeout, self.next_deadline()) {
            (Some(limit), Some(deadline)) => {
                Some(limit.min(Duration::from_millis(deadline.saturating_sub(now))))
            }
            (Some(limit), None) => Some(limit),
            (None, Some(deadline)) => Some(Duration::from_millis(deadline.saturating_sub(now))),
            (None, None) => None,
        };

        let mut entries = Vec::new();
        let mut pending = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            if let WlSourceSlot::Live {
                generation,
                kind: WlSourceKind::Fd { handle, interest, .. },
            } = slot
            {
                entries.push(WlPollEntry {
                    handle: *handle,
                    interest: *interest,
                    revents: WlPollEvents::EMPTY,
                });
                pending.push(WlEventSourceId {
                    index,
                    generation: *generation,
                });
            }
        }

        self.poller.poll(&mut entries, wait)?;

        let now = self.clock.now_ms();
        for id in self.take_expired_timers(now) {
            self.invoke_timer(id);
        }

        for (entry, id) in entries.iter().zip(&pending) {
            if !entry.revents.is_empty() {
                self.invoke_fd(*id, entry.revents);
            }
        }

        self.dispatch_idle();
        while self.post_dispatch_check() {}
        Ok(())
    }

    /// Returns the earliest armed timer deadline.
    fn next_deadline(&self) -> Option<u64> {
        self.slots
            .iter()
            .filter_map(|slot| match slot {
                WlSourceSlot::Live {
                    kind:
                        WlSourceKind::Timer {
                            deadline: Some(deadline),
                            ..
                        },
                    ..
                } => Some(*deadline),
                _ => None,
            })
            .min()
    }

    /// Collects every timer due at `now` and disarms it first, matching
    /// the ordering guarantees of `wl_timer_heap_dispatch`.
    fn take_expired_timers(&mut self, now: u64) -> Vec<WlEventSourceId> {
        let mut expired = Vec::new();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let WlSourceSlot::Live {
                generation,
                kind:
                    WlSourceKind::Timer {
                        deadline: slot_deadline,
                        ..
                    },
            } = slot
                && let Some(deadline) = slot_deadline
                && *deadline <= now
            {
                expired.push(WlEventSourceId {
                    index,
                    generation: *generation,
                });
                *slot_deadline = None;
            }
        }
        expired
    }

    /// Dispatches the sources on the check list once.
    ///
    /// Returns `true` while any callback returned non-zero or new sources
    /// were checked during the pass.
    fn post_dispatch_check(&mut self) -> bool {
        let snapshot = self.check.clone();
        let mut recheck = false;
        for id in snapshot.clone() {
            let result = match self.slot(id) {
                Some(WlSourceKind::Fd { .. }) => self.invoke_fd(id, WlPollEvents::EMPTY),
                Some(WlSourceKind::Timer { .. }) => self.invoke_timer(id),
                _ => None,
            };
            recheck |= result.is_some_and(|code| code != 0);
        }
        recheck || self.check != snapshot
    }

    fn alloc_slot(&mut self, kind: WlSourceKind<P, C>) -> WlEventSourceId {
        if let Some(index) = self.free.pop() {
            let generation = match self.slots.get(index) {
                Some(WlSourceSlot::Free { generation }) | Some(WlSourceSlot::Live { generation, .. }) => {
                    generation.wrapping_add(1)
                }
                None => 1,
            };
            self.slots[index] = WlSourceSlot::Live { generation, kind };
            WlEventSourceId { index, generation }
        } else {
            let generation = 1u32;
            self.slots
                .push(WlSourceSlot::Live { generation, kind });
            WlEventSourceId {
                index: self.slots.len() - 1,
                generation,
            }
        }
    }

    fn slot(&self, id: WlEventSourceId) -> Option<&WlSourceKind<P, C>> {
        match self.slots.get(id.index)? {
            WlSourceSlot::Live { generation, kind } if *generation == id.generation => Some(kind),
            _ => None,
        }
    }

    fn slot_mut(&mut self, id: WlEventSourceId) -> Option<&mut WlSourceKind<P, C>> {
        match self.slots.get_mut(id.index)? {
            WlSourceSlot::Live { generation, kind } if *generation == id.generation => Some(kind),
            _ => None,
        }
    }

    fn free_source(&mut self, id: WlEventSourceId) -> bool {
        let generation = match self.slots.get(id.index) {
            Some(WlSourceSlot::Live { generation, .. }) if *generation == id.generation => *generation,
            _ => return false,
        };
        self.slots[id.index] = WlSourceSlot::Free { generation };
        self.free.push(id.index);
        self.idle.retain(|other| *other != id);
        self.check.retain(|other| *other != id);
        true
    }

    fn take_fd(&mut self, id: WlEventSourceId) -> Option<Box<FdFunc<P, C>>> {
        match self.slot_mut(id)? {
            WlSourceKind::Fd { callback, .. } => callback.take(),
            _ => None,
        }
    }

    fn take_timer(&mut self, id: WlEventSourceId) -> Option<TimerCallback<P, C>> {
        match self.slot_mut(id)? {
            WlSourceKind::Timer { callback, .. } => callback.take(),
            _ => None,
        }
    }

    fn take_idle(&mut self, id: WlEventSourceId) -> Option<IdleCallback<P, C>> {
        match self.slot_mut(id)? {
            WlSourceKind::Idle { callback } => callback.take(),
            _ => None,
        }
    }

    /// Restores a callback that was taken out for invocation.
    ///
    /// Nothing is restored when the source was removed or replaced while
    /// its callback ran.
    fn restore_fd(&mut self, id: WlEventSourceId, callback: Box<FdFunc<P, C>>) {
        if let Some(WlSourceKind::Fd {
            callback: slot_callback,
            ..
        }) = self.slot_mut(id)
            && slot_callback.is_none()
        {
            *slot_callback = Some(callback);
        }
    }

    fn restore_timer(&mut self, id: WlEventSourceId, callback: TimerCallback<P, C>) {
        if let Some(WlSourceKind::Timer {
            callback: slot_callback,
            ..
        }) = self.slot_mut(id)
            && slot_callback.is_none()
        {
            *slot_callback = Some(callback);
        }
    }

    fn invoke_fd(&mut self, id: WlEventSourceId, events: WlPollEvents) -> Option<i32> {
        let mut callback = self.take_fd(id)?;
        let result = callback(&mut *self, id, events);
        self.restore_fd(id, callback);
        Some(result)
    }

    fn invoke_timer(&mut self, id: WlEventSourceId) -> Option<i32> {
        let mut callback = self.take_timer(id)?;
        let result = callback(&mut *self, id);
        self.restore_timer(id, callback);
        Some(result)
    }
}

fn stale_source() -> WlError {
    WlError::InvalidState(String::from("event source does not exist"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use core::cell::{Cell, RefCell};

    struct FakeClock {
        now: u64,
    }

    impl WlClock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.now
        }
    }

    #[derive(Default)]
    struct FakePoller {
        ready: Vec<(WlHandle, WlPollEvents)>,
        polls: usize,
        last_timeout: Option<Duration>,
        last_interests: Vec<(WlHandle, WlPollEvents)>,
    }

    impl WlPoller for FakePoller {
        fn poll(&mut self, entries: &mut [WlPollEntry], timeout: Option<Duration>) -> WlResult<usize> {
            self.polls += 1;
            self.last_timeout = timeout;
            self.last_interests = entries
                .iter()
                .map(|entry| (entry.handle, entry.interest))
                .collect();
            let mut ready = 0;
            for entry in entries.iter_mut() {
                let events = self
                    .ready
                    .iter()
                    .find(|(handle, _)| *handle == entry.handle)
                    .map_or(WlPollEvents::EMPTY, |(_, events)| *events);
                entry.revents = events.intersection(entry.interest);
                if !entry.revents.is_empty() {
                    ready += 1;
                }
            }
            Ok(ready)
        }
    }

    #[test]
    fn dispatches_ready_file_descriptors() {
        let poller = FakePoller {
            ready: alloc::vec![(7, WlPollEvents::READABLE)],
            ..FakePoller::default()
        };
        let mut event_loop = WlEventLoop::new(poller, FakeClock { now: 0 });
        let seen = Rc::new(RefCell::new(Vec::new()));
        let log = Rc::clone(&seen);
        let source = event_loop.add_fd(7, WlPollEvents::READABLE, move |_, _, events| {
            log.borrow_mut().push(events);
            0
        });

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(*seen.borrow(), [WlPollEvents::READABLE]);
        assert_eq!(event_loop.poller().polls, 1);

        event_loop
            .fd_update(source, WlPollEvents::WRITABLE)
            .unwrap();
        seen.borrow_mut().clear();
        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert!(seen.borrow().is_empty());
        assert_eq!(event_loop.poller().last_interests, [(7, WlPollEvents::WRITABLE)]);
    }

    #[test]
    fn timers_fire_once_after_their_deadline() {
        let mut event_loop = WlEventLoop::new(FakePoller::default(), FakeClock { now: 0 });
        let fired = Rc::new(Cell::new(0u32));
        let counter = Rc::clone(&fired);
        let timer = event_loop.add_timer(move |_, _| {
            counter.set(counter.get() + 1);
            0
        });
        event_loop.timer_update(timer, 100).unwrap();

        event_loop
            .dispatch(Some(Duration::from_millis(50)))
            .unwrap();
        assert_eq!(fired.get(), 0);
        assert_eq!(event_loop.poller().last_timeout, Some(Duration::from_millis(50)));

        event_loop
            .dispatch(Some(Duration::from_millis(1000)))
            .unwrap();
        assert_eq!(event_loop.poller().last_timeout, Some(Duration::from_millis(100)));

        event_loop.clock_mut().now = 100;
        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(fired.get(), 1);

        event_loop.clock_mut().now = 10_000;
        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(fired.get(), 1);
    }

    #[test]
    fn idle_tasks_run_before_the_poll_and_are_removed() {
        let mut event_loop = WlEventLoop::new(FakePoller::default(), FakeClock { now: 0 });
        let fired = Rc::new(Cell::new(0u32));
        let polls_when_run = Rc::new(Cell::new(usize::MAX));
        let counter = Rc::clone(&fired);
        let polls = Rc::clone(&polls_when_run);
        event_loop.add_idle(move |loop_, _| {
            polls.set(loop_.poller().polls);
            counter.set(counter.get() + 1);
        });

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(fired.get(), 1);
        assert_eq!(polls_when_run.get(), 0);

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(fired.get(), 1);
        assert_eq!(event_loop.poller().polls, 2);
    }

    #[test]
    fn checked_sources_recheck_until_they_return_zero() {
        let mut event_loop = WlEventLoop::new(FakePoller::default(), FakeClock { now: 0 });
        let calls = Rc::new(Cell::new(0u32));
        let events_seen = Rc::new(RefCell::new(Vec::new()));
        let counter = Rc::clone(&calls);
        let log = Rc::clone(&events_seen);
        let source = event_loop.add_fd(3, WlPollEvents::READABLE, move |_, _, events| {
            counter.set(counter.get() + 1);
            log.borrow_mut().push(events);
            if counter.get() == 1 { 1 } else { 0 }
        });
        event_loop.check(source).unwrap();

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(*events_seen.borrow(), [WlPollEvents::EMPTY, WlPollEvents::EMPTY]);

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn removing_a_source_invalidates_its_identifier() {
        let poller = FakePoller {
            ready: alloc::vec![(9, WlPollEvents::READABLE)],
            ..FakePoller::default()
        };
        let mut event_loop = WlEventLoop::new(poller, FakeClock { now: 0 });
        let fired = Rc::new(Cell::new(0u32));
        let counter = Rc::clone(&fired);
        let source = event_loop.add_fd(9, WlPollEvents::READABLE, move |loop_, own, _| {
            counter.set(counter.get() + 1);
            loop_.remove_source(own).unwrap();
            0
        });

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(fired.get(), 1);
        assert!(!event_loop.is_active(source));
        assert!(event_loop.remove_source(source).is_err());

        event_loop.dispatch(Some(Duration::ZERO)).unwrap();
        assert_eq!(fired.get(), 1);
    }
}
