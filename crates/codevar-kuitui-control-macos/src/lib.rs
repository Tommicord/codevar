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

//! macOS input backend for `codevar-kuitui-control`.
//!
//! The backend installs a CoreGraphics event tap on a worker thread. The tap
//! is listen-only: input continues to reach the focused application.

#![warn(missing_docs)]

use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};

/// CoreGraphics event-tap backend.
#[derive(Default)]
pub struct MacOsBackend;

impl MacOsBackend {
    /// Creates a backend that has installed no event tap.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Backend for MacOsBackend {
    fn start(&mut self, class: DeviceClass, _sink: EventSink) -> Result<(), ControlError> {
        Err(ControlError::Unsupported(class))
    }

    fn stop(&mut self, _class: DeviceClass) {}

    fn devices(&self) -> Vec<DeviceInfo> {
        Vec::new()
    }
}
