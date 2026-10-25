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

//! Windows input backend for `codevar-kuitui-control`.
//!
//! The backend listens through Win32 Raw Input on a message-only window. It
//! owns no thread: the caller pumps its own message loop and asks the backend
//! to decode the records that arrived since the previous call.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg_attr(not(windows), allow(dead_code))]
mod decode;
#[cfg_attr(not(windows), allow(dead_code))]
mod keymap;

use alloc::vec::Vec;

use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};

/// Win32 Raw Input backend.
#[derive(Default)]
pub struct WindowsBackend;

impl WindowsBackend {
    /// Creates a backend that has opened no devices.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Backend for WindowsBackend {
    fn start(&mut self, class: DeviceClass, _sink: EventSink) -> Result<(), ControlError> {
        let _ = class;
        Err(ControlError::Unsupported(class))
    }

    fn stop(&mut self, _class: DeviceClass) {}

    fn devices(&self) -> Vec<DeviceInfo> {
        Vec::new()
    }
}
