//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The fan-out hub behind every subscription.
//!
//! The hub is the replacement for an observer list: a subscriber owns a
//! bounded queue and a demand counter, and the hub never calls user code.
//! Waking a parked task is the only side effect, and it happens after the
//! state lock is released.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll, Waker};

use spin::Mutex;

use crate::class::ClassSet;
use crate::error::ControlError;
use crate::event::Event;
use crate::stream::Subscription;

/// Maximum number of subscribers a single hub can hold.
///
/// The cap keeps a misbehaving consumer from growing the wake buffer without
/// bound; a UI has one or two input streams, never thousands.
const MAX_SUBSCRIBERS: usize = 64;

/// One subscription's state.
struct Subscriber {
    /// Identifier returned to the caller's [`Subscription`].
    id: u64,
    /// Classes whose events this subscriber wants.
    classes: ClassSet,
    /// Maximum number of events the queue may hold.
    capacity: usize,
    /// Granted but not yet consumed events.
    queue: VecDeque<Arc<Event>>,
    /// How many further events the subscriber is willing to receive.
    demand: u64,
    /// Events dropped because the queue was full or demand was zero.
    dropped: u64,
    /// Terminal error to deliver after the queue drains.
    terminal: Option<ControlError>,
    /// Set once the terminal error has been returned.
    finished: bool,
    /// Task to wake when an event or a lag notice arrives.
    waker: Option<Waker>,
}

impl Subscriber {
    /// Whether an event of `class` belongs to this subscriber.
    fn accepts(&self, class: Option<crate::class::DeviceClass>) -> bool {
        match class {
            Some(class) => self.classes.contains(class),
            // A device removal cannot be attributed to a class, so every
            // subscriber hears it and can match the identifier itself.
            None => true,
        }
    }
}

/// Shared state of a publisher.
pub(crate) struct Hub {
    /// Subscribers, guarded because backends emit from worker threads.
    state: Mutex<HubState>,
    /// Source of subscription identifiers.
    next_id: AtomicU64,
}

#[derive(Default)]
struct HubState {
    subscribers: Vec<Subscriber>,
}

impl Hub {
    /// Creates a hub with no subscribers.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(HubState::default()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Registers a subscriber and grants it an initial prefetch of
    /// `capacity` events.
    ///
    /// The prefetch is bounded by the queue length, so a backend that emits
    /// device-added events while it starts cannot lose them before the caller
    /// polls for the first time.
    pub(crate) fn subscribe(
        self: &Arc<Self>,
        classes: ClassSet,
        capacity: usize,
    ) -> Result<Subscription, ControlError> {
        if capacity == 0 {
            return Err(ControlError::ZeroCapacity);
        }
        let mut state = self.state.lock();
        if state.subscribers.len() >= MAX_SUBSCRIBERS {
            return Err(ControlError::TooManySubscribers);
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let subscriber = Subscriber {
            id,
            classes,
            capacity,
            queue: VecDeque::with_capacity(capacity.min(64)),
            demand: capacity as u64,
            dropped: 0,
            terminal: None,
            finished: false,
            waker: None,
        };
        state.subscribers.push(subscriber);
        Ok(Subscription::new(Arc::clone(self), id))
    }

    /// Delivers one event to every subscriber whose classes accept it.
    ///
    /// A subscriber with no demand, or with a full queue, counts a drop
    /// instead of growing memory. The count is reported as
    /// [`ControlError::Lagged`] on the next poll, which makes loss visible
    /// without turning it into a fatal error.
    pub(crate) fn emit(&self, event: Event) {
        let event = Arc::new(event);
        let class = event.class();
        let mut wakers = Vec::new();
        {
            let mut state = self.state.lock();
            for subscriber in &mut state.subscribers {
                if subscriber.terminal.is_some() || !subscriber.accepts(class) {
                    continue;
                }
                if subscriber.demand > 0 && subscriber.queue.len() < subscriber.capacity {
                    subscriber.queue.push_back(Arc::clone(&event));
                    subscriber.demand -= 1;
                    if let Some(waker) = subscriber.waker.take() {
                        wakers.push(waker);
                    }
                } else {
                    subscriber.dropped = subscriber.dropped.saturating_add(1);
                }
            }
        }
        for waker in wakers {
            waker.wake();
        }
    }

    /// Adds demand to one subscription.
    pub(crate) fn request(&self, id: u64, amount: u64) -> Result<(), ControlError> {
        let waker = {
            let mut state = self.state.lock();
            let Some(subscriber) = state
                .subscribers
                .iter_mut()
                .find(|subscriber| subscriber.id == id)
            else {
                return Err(ControlError::Closed);
            };
            if amount == 0 {
                return Ok(());
            }
            subscriber.demand = subscriber.demand.saturating_add(amount);
            subscriber.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    /// Polls one subscription for the next event.
    pub(crate) fn poll(
        &self,
        id: u64,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Arc<Event>, ControlError>>> {
        let mut state = self.state.lock();
        let Some(subscriber) = state
            .subscribers
            .iter_mut()
            .find(|subscriber| subscriber.id == id)
        else {
            return Poll::Ready(None);
        };

        if let Some(event) = subscriber.queue.pop_front() {
            // The task is being polled now; a stale waker would only cause a
            // spurious wake-up later.
            subscriber.waker = None;
            return Poll::Ready(Some(Ok(event)));
        }

        // A lag report is a diagnostic, not data: it is delivered even when
        // the consumer has spent its demand, otherwise a consumer that stops
        // requesting would never learn that it missed events.
        if subscriber.dropped > 0 {
            let dropped = subscriber.dropped;
            subscriber.dropped = 0;
            subscriber.demand = subscriber.demand.saturating_sub(1);
            return Poll::Ready(Some(Err(ControlError::Lagged { dropped })));
        }

        if subscriber.finished {
            subscriber.waker = None;
            return Poll::Ready(None);
        }

        if let Some(error) = subscriber.terminal.take() {
            subscriber.finished = true;
            subscriber.waker = None;
            return Poll::Ready(Some(Err(error)));
        }

        subscriber.waker = Some(cx.waker().clone());
        Poll::Pending
    }

    /// Ends every subscription that intersects `classes`.
    ///
    /// Events already queued are delivered first; the terminal error follows
    /// them, so a consumer never loses an event it was already granted.
    pub(crate) fn close(&self, classes: ClassSet, error: ControlError) {
        let mut wakers = Vec::new();
        {
            let mut state = self.state.lock();
            for subscriber in &mut state.subscribers {
                if subscriber.terminal.is_some() || !subscriber.classes.intersects(classes) {
                    continue;
                }
                subscriber.terminal = Some(error.clone());
                if let Some(waker) = subscriber.waker.take() {
                    wakers.push(waker);
                }
            }
        }
        for waker in wakers {
            waker.wake();
        }
    }

    /// Ends every subscription, usually because the backend failed fatally.
    pub(crate) fn fail(&self, error: ControlError) {
        let mut wakers = Vec::new();
        {
            let mut state = self.state.lock();
            for subscriber in &mut state.subscribers {
                if subscriber.terminal.is_some() {
                    continue;
                }
                subscriber.terminal = Some(error.clone());
                if let Some(waker) = subscriber.waker.take() {
                    wakers.push(waker);
                }
            }
        }
        for waker in wakers {
            waker.wake();
        }
    }

    /// Removes a subscription, releasing its queue.
    pub(crate) fn cancel(&self, id: u64) {
        let mut state = self.state.lock();
        if let Some(index) = state
            .subscribers
            .iter()
            .position(|subscriber| subscriber.id == id)
        {
            state.subscribers.swap_remove(index);
        }
    }

    /// Number of live subscriptions, for tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn subscriber_count(&self) -> usize {
        self.state.lock().subscribers.len()
    }
}
