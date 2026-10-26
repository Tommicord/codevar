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

//! The seam between the portable control layer and an operating system.
//!
//! A backend owns the native devices. The control layer tells it when a
//! device class becomes used for the first time and when its owner releases
//! it; the backend turns native records into [`InputEvent`]s and hands them
//! to an [`EventSink`]. The sink is thread-safe, so a backend may run its
//! blocking message loop on a worker thread without making the control layer
//! depend on that thread.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::class::DeviceClass;
use crate::device::DeviceInfo;
use crate::error::ControlError;
use crate::event::{Event, InputEvent, Timestamp};
use crate::hub::Hub;

/// A backend's only channel back into the control layer.
///
/// The sink is cheap to clone and safe to keep on a worker thread. Emitting
/// never blocks: when a consumer is not keeping up, the event is dropped and
/// the loss is reported to that consumer as
/// [`ControlError::Lagged`](crate::ControlError::Lagged).
#[derive(Clone)]
pub struct EventSink {
    /// Hub that fans the event out to the subscribers.
    hub: Arc<Hub>,
}

impl EventSink {
    /// Wraps a hub. Only the control layer creates sinks.
    pub(crate) fn new(hub: Arc<Hub>) -> Self {
        Self { hub }
    }

    /// Emits an event whose backend has no timestamp.
    pub fn emit(&self, payload: InputEvent) {
        self.hub.emit(Event::untimed(payload));
    }

    /// Emits an event at a backend-provided time.
    pub fn emit_at(&self, timestamp: Timestamp, payload: InputEvent) {
        self.hub.emit(Event::new(timestamp, payload));
    }

    /// Emits an already timestamped event.
    pub fn emit_event(&self, event: Event) {
        self.hub.emit(event);
    }

    /// Ends every subscription with a terminal error.
    ///
    /// A backend calls this when its device loop cannot continue, for example
    /// after the device node disappears or the platform revokes permission.
    /// Registration state stays with the control layer; the caller decides
    /// whether to unregister and open again.
    pub fn fail(&self, error: ControlError) {
        self.hub.fail(error);
    }
}

impl core::fmt::Debug for EventSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventSink")
            .finish_non_exhaustive()
    }
}

/// An operating-system input implementation.
///
/// # Contract
///
/// - [`start`](Self::start) is called at most once per class until the class
///   is stopped. The first call for a class is exact: it happens only when a
///   caller opens a stream for that class, and any other class may be started
///   later. A backend that shares one native session between classes keeps
///   its own count and starts the session on the first class.
/// - [`stop`](Self::stop) is called exactly once for every successful
///   `start`, in no particular order. The backend must release the native
///   resource and join any worker it owns before returning, so a caller that
///   unregisters a class can be sure the device is closed.
/// - Events may be emitted at any time between `start` and `stop`. After
///   `stop` returns, the backend must not use the sink again.
/// - Neither method may panic.
pub trait Backend: 'static {
    /// Starts the native resources for `class`.
    fn start(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError>;

    /// Releases the native resources for `class`.
    fn stop(&mut self, class: DeviceClass);

    /// Returns the devices the backend knows about.
    ///
    /// A backend that can enumerate only while a class is registered returns
    /// the devices of the registered classes, plus static entries it can
    /// describe without opening a device.
    fn devices(&self) -> Vec<DeviceInfo>;
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::EventSink;
    use crate::class::ClassSet;
    use crate::error::ControlError;
    use crate::hub::Hub;
    use crate::stream::Subscription;

    #[test]
    fn fail_ends_a_live_subscription_with_the_backend_error() {
        let hub = Arc::new(Hub::new());
        let sink = EventSink::new(Arc::clone(&hub));
        let mut subscription: Subscription = hub
            .subscribe(ClassSet::KEYBOARD, 4)
            .expect("a fresh hub accepts a subscription");
        sink.fail(ControlError::backend("worker stopped"));

        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        assert!(matches!(
            subscription.poll_next(&mut context),
            core::task::Poll::Ready(Some(Err(ControlError::Backend { .. })))
        ));
        assert!(matches!(
            subscription.poll_next(&mut context),
            core::task::Poll::Ready(None)
        ));
    }
}
