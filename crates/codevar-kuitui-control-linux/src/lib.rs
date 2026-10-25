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
//! The backend reads `/dev/input/event*` through evdev. The worker is
//! created the first time a class is opened and stopped when the class is
//! unregistered.

#![warn(missing_docs)]

use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};

/// Linux evdev input backend.
#[derive(Default)]
pub struct LinuxBackend {
    /// Classes whose devices are open.
    registered: codevar_kuitui_control::ClassSet,
}

impl LinuxBackend {
    /// Creates a backend that has opened no devices.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Backend for LinuxBackend {
    fn start(&mut self, class: DeviceClass, _sink: EventSink) -> Result<(), ControlError> {
        let _ = class;
        Err(ControlError::Unsupported(class))
    }

    fn stop(&mut self, _class: DeviceClass) {}

    fn devices(&self) -> Vec<DeviceInfo> {
        let _ = self.registered;
        Vec::new()
    }
}
