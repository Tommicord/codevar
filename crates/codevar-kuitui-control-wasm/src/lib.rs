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

//! Browser input backend for `codevar-kuitui-control`.
//!
//! The backend attaches DOM listeners to the window when a class is first
//! opened and removes them when the class is unregistered, so the page sees
//! no listeners for a class nobody uses. It owns no thread and no timer: the
//! browser invokes the closures on the main thread, and the caller must
//! register from that thread.
//!
//! The browser does not expose hardware identity, so the keyboard and the
//! pointer are reported as one fixed device each. Gamepads are enumerated
//! from the `gamepadconnected` and `gamepaddisconnected` events, because that
//! is the only moment the browser reveals them.
//!
//! On a host target the backend reports every class as unsupported; the DOM
//! is what provides the events, and there is none outside the browser.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod keymap;
#[cfg(target_arch = "wasm32")]
mod listeners;

use alloc::rc::Rc;
#[cfg(target_arch = "wasm32")]
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

#[cfg(target_arch = "wasm32")]
use codevar_kuitui_control::DeviceId;
use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};

/// DOM event backend.
#[derive(Default)]
pub struct WasmBackend {
    /// One entry per class that is registered.
    sessions: Vec<Session>,
}

/// One registered class.
struct Session {
    /// Class the session serves.
    class: DeviceClass,
    /// Devices reported for this class.
    devices: Rc<RefCell<Vec<DeviceInfo>>>,
    /// Live listeners, removed when the session is dropped.
    ///
    /// Nothing reads this field: it exists so that dropping the session runs
    /// the listener cleanup instead of leaking the closures.
    #[allow(dead_code)]
    #[cfg(target_arch = "wasm32")]
    listeners: listeners::Listeners,
}

impl WasmBackend {
    /// Creates a backend that has attached no listeners.
    #[must_use]
    pub const fn new() -> Self {
        Self { sessions: Vec::new() }
    }

    /// The device the browser exposes for a class before a lossless one
    /// appears.
    ///
    /// A gamepad has no such entry because the browser reveals it only when
    /// it connects.
    #[cfg(target_arch = "wasm32")]
    fn builtin(class: DeviceClass) -> Option<DeviceInfo> {
        let (id, name) = match class {
            DeviceClass::Keyboard => (DeviceId::new(1), "Keyboard"),
            DeviceClass::Pointer => (DeviceId::new(2), "Pointer"),
            _ => return None,
        };
        Some(DeviceInfo::new(id, class, String::from(name)))
    }

    /// Attaches the listeners of `class` and records the session.
    #[cfg(target_arch = "wasm32")]
    fn attach(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
        let devices = Rc::new(RefCell::new(Self::builtin(class).into_iter().collect()));
        let listeners = listeners::Listeners::attach(class, Rc::clone(&devices), sink)?;
        self.sessions.push(Session {
            class,
            devices,
            listeners,
        });
        Ok(())
    }
}

impl Backend for WasmBackend {
    fn start(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
        if self
            .sessions
            .iter()
            .any(|session| session.class == class)
        {
            return Ok(());
        }
        #[cfg(target_arch = "wasm32")]
        return self.attach(class, sink);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = sink;
            Err(ControlError::Unsupported(class))
        }
    }

    fn stop(&mut self, class: DeviceClass) {
        if let Some(index) = self
            .sessions
            .iter()
            .position(|session| session.class == class)
        {
            self.sessions.swap_remove(index);
        }
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        self.sessions
            .iter()
            .flat_map(|session| session.devices.borrow().clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::Backend;
    #[cfg(not(target_arch = "wasm32"))]
    use codevar_kuitui_control::{Control, ControlError, DeviceClass};

    use super::WasmBackend;

    #[test]
    fn nothing_is_reported_before_a_class_is_opened() {
        assert!(WasmBackend::new().devices().is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_host_build_reports_every_class_as_unsupported() {
        let mut control = Control::new(WasmBackend::new());
        assert!(matches!(
            control.open(DeviceClass::Keyboard),
            Err(ControlError::Unsupported(DeviceClass::Keyboard))
        ));
    }
}
