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

//! Reactive input streams.
//!
//! An input publisher is not an observer list. A caller subscribes, receives
//! a [`Subscription`], and then *requests* a number of events. The publisher
//! grants at most that many; a consumer that stops requesting is never
//! called, and a consumer that falls behind loses events in a bounded way
//! instead of growing memory. The model is the Reactive Streams
//! `Publisher`/`Subscription` contract expressed with Rust's `Poll`, so it
//! composes with any executor and needs no runtime.

use alloc::sync::Arc;
use core::future::poll_fn;
use core::task::{Context, Poll};

use crate::class::ClassSet;
use crate::error::ControlError;
use crate::event::Event;
use crate::hub::Hub;

/// The number of events a subscription queues by default.
///
/// The value is large enough to absorb a burst of pointer motion between two
/// frames and small enough that a stalled consumer is noticed in one frame.
pub const DEFAULT_CAPACITY: usize = 256;

/// A source of input events that supports demand-based subscription.
///
/// The associated error is the publisher's construction error (for example a
/// backend that fails to start); errors observed while streaming are
/// delivered as [`ControlError`] items by each [`Subscription`].
pub trait Publisher {
    /// The error returned when a subscription cannot be created.
    type Error;

    /// Creates a subscription for `classes`.
    ///
    /// `capacity` bounds the number of events the subscription may queue.
    /// The subscription starts with a prefetch of that many events, so a
    /// backend that reports device additions during registration does not
    /// lose them before the first poll.
    fn subscribe(&mut self, classes: ClassSet, capacity: usize) -> Result<Subscription, Self::Error>;
}

/// One consumer's view of an input publisher.
///
/// A subscription is a stream of [`Event`] values. It is also a
/// Reactive Streams `Subscription`: [`request`](Self::request) grants demand,
/// [`cancel`](Self::cancel) releases the consumer, and dropping it cancels it
/// automatically.
///
/// Events are delivered as `Arc<Event>` so fan-out to several consumers
/// clones a pointer rather than the event payload.
pub struct Subscription {
    /// Hub that owns the queue.
    hub: Arc<Hub>,
    /// Identifier of this subscriber in the hub.
    id: u64,
    /// Set once the consumer released the subscription.
    cancelled: bool,
}

impl Subscription {
    /// Creates a subscription handle for a subscriber the hub already added.
    pub(crate) fn new(hub: Arc<Hub>, id: u64) -> Self {
        Self {
            hub,
            id,
            cancelled: false,
        }
    }

    /// Grants permission for `amount` more events.
    ///
    /// The demand is additive and saturates instead of wrapping. A request
    /// for zero events is a no-op, as the Reactive Streams specification
    /// requires.
    pub fn request(&mut self, amount: u64) -> Result<(), ControlError> {
        if self.cancelled {
            return Err(ControlError::Closed);
        }
        self.hub.request(self.id, amount)
    }

    /// Polls for the next event.
    ///
    /// The returned error is either [`ControlError::Lagged`], which is
    /// non-terminal, or a terminal error after which the stream yields
    /// `None`. Callers driving the stream manually must call
    /// [`request`](Self::request) before polling when the prefetch is
    /// exhausted; [`next`](Self::next) does so automatically.
    pub fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<Arc<Event>, ControlError>>> {
        if self.cancelled {
            return Poll::Ready(None);
        }
        self.hub.poll(self.id, cx)
    }

    /// Waits for the next event, requesting one event of demand first.
    ///
    /// This is the ergonomic async entry point. It never busy-waits: it parks
    /// the task through the subscription's waker.
    pub async fn next(&mut self) -> Option<Result<Arc<Event>, ControlError>> {
        // A closed subscription has no demand to grant; polling it once more
        // yields the terminal item or `None`.
        let _ = self.request(1);
        poll_fn(|cx| self.poll_next(cx)).await
    }

    /// Releases the subscription immediately.
    ///
    /// Cancellation is idempotent. It does not unregister the device class:
    /// registration is owned by [`Control`](crate::Control), which releases it
    /// only when its owner asks.
    pub fn cancel(&mut self) {
        if !self.cancelled {
            self.cancelled = true;
            self.hub.cancel(self.id);
        }
    }

    /// Whether the consumer released this subscription.
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl core::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Subscription")
            .field("id", &self.id)
            .field("cancelled", &self.cancelled)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use core::task::{Context, Waker};

    use super::Publisher;
    use crate::class::{ClassSet, DeviceClass};
    use crate::error::ControlError;
    use crate::event::{Event, InputEvent, KeyCode, KeyEvent, KeyState, Modifiers};
    use crate::hub::Hub;

    struct TestPublisher {
        hub: Arc<Hub>,
    }

    impl Publisher for TestPublisher {
        type Error = ControlError;

        fn subscribe(
            &mut self,
            classes: ClassSet,
            capacity: usize,
        ) -> Result<super::Subscription, Self::Error> {
            self.hub.subscribe(classes, capacity)
        }
    }

    fn publisher() -> (TestPublisher, Arc<Hub>) {
        let hub = Arc::new(Hub::new());
        let publisher = TestPublisher {
            hub: Arc::clone(&hub),
        };
        (publisher, hub)
    }

    fn key_event(value: u32) -> Event {
        Event::untimed(InputEvent::Key(KeyEvent {
            device: crate::device::DeviceId::new(1),
            code: KeyCode::Native(value),
            state: KeyState::Pressed,
            modifiers: Modifiers::NONE,
        }))
    }

    fn poll(subscription: &mut super::Subscription) -> Option<Result<Arc<Event>, ControlError>> {
        let mut context = Context::from_waker(Waker::noop());
        match subscription.poll_next(&mut context) {
            core::task::Poll::Ready(value) => value,
            core::task::Poll::Pending => None,
        }
    }

    #[test]
    fn subscription_starts_with_bounded_prefetch() {
        let (mut publisher, hub) = publisher();
        let subscription = publisher
            .subscribe(ClassSet::KEYBOARD, 4)
            .unwrap();
        hub.emit(key_event(1));
        hub.emit(key_event(2));
        hub.emit(key_event(3));
        hub.emit(key_event(4));
        hub.emit(key_event(5));

        let mut subscription = subscription;
        assert_eq!(poll(&mut subscription).map(|item| item.is_ok()), Some(true));
        assert_eq!(poll(&mut subscription).map(|item| item.is_ok()), Some(true));
        assert_eq!(poll(&mut subscription).map(|item| item.is_ok()), Some(true));
        assert_eq!(poll(&mut subscription).map(|item| item.is_ok()), Some(true));
        assert!(matches!(
            poll(&mut subscription),
            Some(Err(ControlError::Lagged { dropped: 1 }))
        ));
        assert!(poll(&mut subscription).is_none());
    }

    #[test]
    fn request_grants_more_events_after_prefetch_is_spent() {
        let (mut publisher, hub) = publisher();
        let mut subscription = publisher
            .subscribe(ClassSet::KEYBOARD, 1)
            .unwrap();
        hub.emit(key_event(1));
        assert!(poll(&mut subscription).is_some());

        subscription.request(2).unwrap();
        hub.emit(key_event(2));
        hub.emit(key_event(3));
        assert!(poll(&mut subscription).is_some());
        assert!(poll(&mut subscription).is_some());
    }

    #[test]
    fn zero_demand_drops_events_without_blocking_the_backend() {
        let (mut publisher, hub) = publisher();
        let mut subscription = publisher
            .subscribe(ClassSet::KEYBOARD, 1)
            .unwrap();
        subscription.request(1).unwrap();
        hub.emit(key_event(1));
        assert!(poll(&mut subscription).is_some());

        // The prefetch was one event; further emission must not block or
        // grow the queue.
        for value in 2..=5 {
            hub.emit(key_event(value));
        }
        subscription.request(1).unwrap();
        // Event 2 was granted before the queue filled; events 3, 4 and 5 were
        // dropped. The queued event arrives first, then the lag report.
        assert!(matches!(poll(&mut subscription), Some(Ok(_))));
        assert!(matches!(
            poll(&mut subscription),
            Some(Err(ControlError::Lagged { dropped: 3 }))
        ));
    }

    #[test]
    fn classes_filter_events_before_they_reach_the_queue() {
        let (mut publisher, hub) = publisher();
        let mut subscription = publisher.subscribe(ClassSet::POINTER, 4).unwrap();
        hub.emit(key_event(1));
        assert!(poll(&mut subscription).is_none());

        hub.emit(Event::untimed(InputEvent::Pointer(
            crate::event::PointerEvent::Motion {
                position: None,
                delta: Some(crate::event::Point::new(1.0, 2.0)),
            },
        )));
        assert!(poll(&mut subscription).is_some());
    }

    #[test]
    fn dropping_a_subscription_removes_it_from_the_hub() {
        let (mut publisher, hub) = publisher();
        let subscription = publisher
            .subscribe(ClassSet::KEYBOARD, 4)
            .unwrap();
        assert_eq!(hub.subscriber_count(), 1);
        drop(subscription);
        assert_eq!(hub.subscriber_count(), 0);
    }

    #[test]
    fn cancel_is_idempotent_and_ends_the_stream() {
        let (mut publisher, hub) = publisher();
        let mut subscription = publisher
            .subscribe(ClassSet::KEYBOARD, 4)
            .unwrap();
        subscription.cancel();
        subscription.cancel();
        assert!(subscription.is_cancelled());
        assert!(poll(&mut subscription).is_none());
        assert!(matches!(subscription.request(1), Err(ControlError::Closed)));
        assert_eq!(hub.subscriber_count(), 0);
    }

    #[test]
    fn capacity_zero_is_rejected() {
        let (mut publisher, _hub) = publisher();
        assert!(matches!(
            publisher.subscribe(ClassSet::KEYBOARD, 0),
            Err(ControlError::ZeroCapacity)
        ));
    }

    #[test]
    fn device_class_has_one_distinct_bit() {
        let mut seen = 0;
        for class in DeviceClass::ALL {
            assert_eq!(seen & class.bit(), 0);
            seen |= class.bit();
        }
        assert_eq!(seen, ClassSet::ALL.bits());
    }
}
