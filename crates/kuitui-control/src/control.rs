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

//! Lazy registration and the convenient front end to a backend.
//!
//! A [`Control`] owns exactly one backend and the registration state of its
//! device classes. Nothing is opened at construction time: the first
//! [`open`](Control::open) for a class calls [`Backend::start`], and later
//! opens of the same class share that registration. [`unregister`] is the
//! explicit release; the class is registered again the next time it is
//! opened.

use alloc::sync::Arc;
use alloc::vec::Vec;

use codevar_logger::{log_debug, log_warn};

use crate::backend::{Backend, EventSink};
use crate::class::{ClassSet, DeviceClass};
use crate::device::DeviceInfo;
use crate::error::ControlError;
use crate::hub::Hub;
use crate::stream::{DEFAULT_CAPACITY, Publisher, Subscription};

/// Owns a backend, its lazy registration state and the shared event hub.
///
/// The control is generic over the backend so it has no virtual dispatch and
/// so a single-threaded backend (the browser) does not force the whole control
/// to be `Send`.
pub struct Control<B: Backend> {
    /// The operating-system implementation.
    backend: B,
    /// Fan-out point shared with the backend workers.
    hub: Arc<Hub>,
    /// Classes whose native resources are currently open.
    registered: ClassSet,
}

impl<B: Backend> Control<B> {
    /// Creates a control that has opened no native resources yet.
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            hub: Arc::new(Hub::new()),
            registered: ClassSet::NONE,
        }
    }

    /// The backend, for platform-specific queries that are not part of the
    /// portable contract.
    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// The backend mutably, for platform-specific configuration performed
    /// before any class is opened.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Opens a stream for one class with the default queue length.
    ///
    /// This is the lazy registration point: the class's native resources are
    /// started on the first call and reused by later calls.
    pub fn open(&mut self, class: DeviceClass) -> Result<Subscription, ControlError> {
        self.open_with_capacity(class, DEFAULT_CAPACITY)
    }

    /// Opens a stream for one class with an explicit queue length.
    pub fn open_with_capacity(
        &mut self,
        class: DeviceClass,
        capacity: usize,
    ) -> Result<Subscription, ControlError> {
        self.open_many_with_capacity(ClassSet::from_class(class), capacity)
    }

    /// Opens a stream that receives every event of `classes` with the
    /// default queue length.
    pub fn open_many(&mut self, classes: ClassSet) -> Result<Subscription, ControlError> {
        self.open_many_with_capacity(classes, DEFAULT_CAPACITY)
    }

    /// Opens a stream that receives every event of `classes`.
    pub fn open_many_with_capacity(
        &mut self,
        classes: ClassSet,
        capacity: usize,
    ) -> Result<Subscription, ControlError> {
        if classes.is_empty() {
            return Err(ControlError::NoClasses);
        }
        if capacity == 0 {
            return Err(ControlError::ZeroCapacity);
        }

        // Subscribe before starting so events a backend emits while it opens
        // its devices (device-added records, initial state) are delivered.
        let mut subscription = self.hub.subscribe(classes, capacity)?;
        let mut started = ClassSet::NONE;
        for class in classes.iter() {
            if self.registered.contains(class) {
                continue;
            }
            let result = self
                .backend
                .start(class, EventSink::new(Arc::clone(&self.hub)));
            if let Err(error) = result {
                // Roll back only the classes this call started. Classes that
                // were already registered keep their native resources.
                for started_class in started.iter() {
                    self.backend.stop(started_class);
                    self.registered.remove(started_class);
                }
                subscription.cancel();
                log_warn!("input: could not start {class}: {error}");
                return Err(error);
            }
            started.insert(class);
            self.registered.insert(class);
            log_debug!("input: started {class} devices");
        }
        Ok(subscription)
    }

    /// Releases the native resources of `class`.
    ///
    /// Live subscriptions that include the class receive
    /// [`ControlError::Unregistered`] after their queued events, then end.
    /// The class is registered again by the next open.
    pub fn unregister(&mut self, class: DeviceClass) -> Result<(), ControlError> {
        if !self.registered.contains(class) {
            return Err(ControlError::NotRegistered(class));
        }
        self.backend.stop(class);
        self.registered.remove(class);
        self.hub
            .close(ClassSet::from_class(class), ControlError::Unregistered(class));
        log_debug!("input: stopped {class} devices");
        Ok(())
    }

    /// Releases every registered class.
    pub fn unregister_all(&mut self) {
        for class in self.registered.iter() {
            self.backend.stop(class);
            self.hub
                .close(ClassSet::from_class(class), ControlError::Unregistered(class));
        }
        self.registered = ClassSet::NONE;
    }

    /// Whether `class` currently has open native resources.
    #[must_use]
    pub fn is_registered(&self, class: DeviceClass) -> bool {
        self.registered.contains(class)
    }

    /// The classes with open native resources.
    #[must_use]
    pub const fn registered_classes(&self) -> ClassSet {
        self.registered
    }

    /// The devices the backend currently knows about.
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceInfo> {
        self.backend.devices()
    }
}

impl<B: Backend> Publisher for Control<B> {
    type Error = ControlError;

    fn subscribe(&mut self, classes: ClassSet, capacity: usize) -> Result<Subscription, Self::Error> {
        self.open_many_with_capacity(classes, capacity)
    }
}

impl<B: Backend> Drop for Control<B> {
    fn drop(&mut self) {
        // A subscription outliving its control must not park forever: the hub
        // ends every stream, then the native resources are released.
        self.hub.fail(ControlError::Closed);
        for class in self.registered.iter() {
            self.backend.stop(class);
        }
        self.registered = ClassSet::NONE;
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::task::{Context, Waker};

    use spin::Mutex;

    use super::Control;
    use crate::backend::{Backend, EventSink};
    use crate::class::{ClassSet, DeviceClass};
    use crate::device::{DeviceId, DeviceInfo};
    use crate::error::ControlError;
    use crate::event::{Event, InputEvent, KeyCode, KeyEvent, KeyState, Modifiers};
    use crate::stream::Subscription;

    #[derive(Default)]
    struct MockState {
        starts: Vec<DeviceClass>,
        stops: Vec<DeviceClass>,
        sinks: Vec<EventSink>,
        fail_on: Option<DeviceClass>,
    }

    #[derive(Clone, Default)]
    struct MockBackend {
        state: Arc<Mutex<MockState>>,
    }

    impl MockBackend {
        fn failing_on(class: DeviceClass) -> Self {
            let backend = Self::default();
            backend.state.lock().fail_on = Some(class);
            backend
        }

        fn starts(&self) -> Vec<DeviceClass> {
            self.state.lock().starts.clone()
        }

        fn stops(&self) -> Vec<DeviceClass> {
            self.state.lock().stops.clone()
        }

        fn emit(&self, event: InputEvent) {
            let sinks = self.state.lock().sinks.clone();
            for sink in sinks {
                sink.emit(event.clone());
            }
        }
    }

    impl Backend for MockBackend {
        fn start(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
            let mut state = self.state.lock();
            if state.fail_on == Some(class) {
                return Err(ControlError::Unsupported(class));
            }
            state.starts.push(class);
            state.sinks.push(sink);
            Ok(())
        }

        fn stop(&mut self, class: DeviceClass) {
            self.state.lock().stops.push(class);
        }

        fn devices(&self) -> Vec<DeviceInfo> {
            let state = self.state.lock();
            state
                .starts
                .iter()
                .map(|class| DeviceInfo::new(DeviceId::new(class.bit() as u64), *class, String::from("mock")))
                .collect()
        }
    }

    fn poll(subscription: &mut Subscription) -> Option<Result<Arc<Event>, ControlError>> {
        let mut context = Context::from_waker(Waker::noop());
        match subscription.poll_next(&mut context) {
            core::task::Poll::Ready(value) => value,
            core::task::Poll::Pending => None,
        }
    }

    fn key_event(value: u32) -> InputEvent {
        InputEvent::Key(KeyEvent {
            device: DeviceId::new(7),
            code: KeyCode::Native(value),
            state: KeyState::Pressed,
            modifiers: Modifiers::NONE,
        })
    }

    #[test]
    fn registration_is_lazy_and_shared_between_opens() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        assert!(backend.starts().is_empty());

        let first = control.open(DeviceClass::Keyboard).unwrap();
        assert_eq!(backend.starts(), alloc::vec![DeviceClass::Keyboard]);
        assert!(control.is_registered(DeviceClass::Keyboard));

        let second = control.open(DeviceClass::Keyboard).unwrap();
        assert_eq!(backend.starts(), alloc::vec![DeviceClass::Keyboard]);

        drop(first);
        drop(second);
        assert!(backend.stops().is_empty());

        control.unregister(DeviceClass::Keyboard).unwrap();
        assert_eq!(backend.stops(), alloc::vec![DeviceClass::Keyboard]);
        assert!(!control.is_registered(DeviceClass::Keyboard));
    }

    #[test]
    fn unregister_then_open_registers_again() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        control.open(DeviceClass::Pointer).unwrap();
        control.unregister(DeviceClass::Pointer).unwrap();
        control.open(DeviceClass::Pointer).unwrap();
        assert_eq!(
            backend.starts(),
            alloc::vec![DeviceClass::Pointer, DeviceClass::Pointer]
        );
    }

    #[test]
    fn events_are_fanned_out_to_matching_sessions() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        let mut keyboard = control.open(DeviceClass::Keyboard).unwrap();
        let mut pointer = control.open(DeviceClass::Pointer).unwrap();

        backend.emit(key_event(1));
        assert!(poll(&mut keyboard).is_some());
        assert!(poll(&mut pointer).is_none());
    }

    #[test]
    fn unregister_ends_live_subscriptions() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend);
        let mut subscription = control.open(DeviceClass::Keyboard).unwrap();
        control.unregister(DeviceClass::Keyboard).unwrap();

        assert!(matches!(
            poll(&mut subscription),
            Some(Err(ControlError::Unregistered(DeviceClass::Keyboard)))
        ));
        assert!(poll(&mut subscription).is_none());
    }

    #[test]
    fn failed_start_rolls_back_the_classes_started_by_the_call() {
        let backend = MockBackend::failing_on(DeviceClass::Pointer);
        let mut control = Control::new(backend.clone());

        let result = control.open_many(ClassSet::KEYBOARD | ClassSet::POINTER);
        assert!(matches!(
            result,
            Err(ControlError::Unsupported(DeviceClass::Pointer))
        ));
        assert_eq!(backend.starts(), alloc::vec![DeviceClass::Keyboard]);
        assert_eq!(backend.stops(), alloc::vec![DeviceClass::Keyboard]);
        assert!(!control.is_registered(DeviceClass::Keyboard));
    }

    #[test]
    fn empty_class_set_and_zero_capacity_are_rejected_before_start() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        assert!(matches!(
            control.open_many(ClassSet::NONE),
            Err(ControlError::NoClasses)
        ));
        assert!(matches!(
            control.open_with_capacity(DeviceClass::Keyboard, 0),
            Err(ControlError::ZeroCapacity)
        ));
        assert!(backend.starts().is_empty());
    }

    #[test]
    fn unregistering_an_unused_class_reports_it() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend);
        assert!(matches!(
            control.unregister(DeviceClass::Microphone),
            Err(ControlError::NotRegistered(DeviceClass::Microphone))
        ));
    }

    #[test]
    fn device_removal_reaches_every_class() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        let mut keyboard = control.open(DeviceClass::Keyboard).unwrap();
        let mut pointer = control.open(DeviceClass::Pointer).unwrap();

        backend.emit(InputEvent::DeviceRemoved(DeviceId::new(3)));
        assert!(matches!(poll(&mut keyboard), Some(Ok(_))));
        assert!(matches!(poll(&mut pointer), Some(Ok(_))));
    }

    #[test]
    fn dropping_the_control_stops_devices_and_ends_streams() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        let mut subscription = control.open(DeviceClass::Keyboard).unwrap();
        drop(control);

        assert_eq!(backend.stops(), alloc::vec![DeviceClass::Keyboard]);
        assert!(matches!(poll(&mut subscription), Some(Err(ControlError::Closed))));
        assert!(poll(&mut subscription).is_none());
    }

    #[test]
    fn unregister_all_releases_every_registered_class() {
        let backend = MockBackend::default();
        let mut control = Control::new(backend.clone());
        control.open(DeviceClass::Keyboard).unwrap();
        control.open(DeviceClass::Pointer).unwrap();
        control.unregister_all();
        assert!(!control.is_registered(DeviceClass::Keyboard));
        assert!(!control.is_registered(DeviceClass::Pointer));
        assert_eq!(backend.stops().len(), 2);
    }
}
