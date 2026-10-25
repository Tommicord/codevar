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

//! Linux input backend for `codevar-kuitui-control`.
//!
//! The backend reads `/dev/input/event*` directly through evdev. A node is
//! opened read-only and non-blocking, then classified from its capability
//! bitmaps, so a caller learns which classes exist without opening a node of
//! another class.
//!
//! This backend does not go through the XDG desktop portal. The portal's
//! RemoteDesktop interface is the *implementation* side of a remote-desktop
//! session: it injects events into a compositor instead of delivering the
//! events of local hardware. Reading evdev is the facility a client uses for
//! local input, and it is also what the browser and compositor paths reduce
//! to.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod device;
mod fd;

use alloc::vec::Vec;

use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};
use codevar_logger::log_warn;

/// Linux evdev input backend.
#[derive(Default)]
pub struct LinuxBackend;

impl LinuxBackend {
    /// Creates a backend that has opened no devices.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Backend for LinuxBackend {
    fn start(&mut self, class: DeviceClass, _sink: EventSink) -> Result<(), ControlError> {
        Err(ControlError::Unsupported(class))
    }

    fn stop(&mut self, _class: DeviceClass) {}

    fn devices(&self) -> Vec<DeviceInfo> {
        match device::scan(None) {
            Ok(devices) => devices
                .into_iter()
                .map(|device| device.info)
                .collect(),
            Err(error) => {
                log_warn!("input: could not enumerate evdev devices: {error}");
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::Backend;

    use super::LinuxBackend;

    #[test]
    fn enumeration_never_panics_in_a_headless_environment() {
        let backend = LinuxBackend::new();
        let _ = backend.devices();
    }
}
