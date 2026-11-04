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
//! The backend installs a listen-only CoreGraphics event tap, so input keeps
//! reaching the focused application. It owns no thread: the tap lives on the
//! run loop of the thread that first opens a stream, that thread continues to
//! pump its own run loop, and `poll` services the events that are ready.
//!
//! A listen-only tap sees every keystroke, so macOS requires the Input
//! Monitoring permission. The first open asks for it and fails with
//! [`ControlError::PermissionDenied`] when the user declines.
//!
//! A class is registered when its first stream opens and removed when the
//! last one closes, which tears the tap down. A tap cannot attribute an event
//! to one physical device, so the whole keyboard stream shares one identifier
//! and so does the pointer stream. Gamepads need a HID report parser that the
//! backend does not have yet, and microphones are not event-tap devices, so
//! both are reported as unsupported.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod decode;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod keymap;
#[cfg(target_os = "macos")]
mod tap;

#[cfg(target_os = "macos")]
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(target_os = "macos")]
use codevar_kuitui_control::InputEvent;
use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};

/// CoreGraphics event-tap backend.
#[derive(Default)]
pub struct MacOsBackend {
    /// The tap, installed when the first stream opens.
    #[cfg(target_os = "macos")]
    tap: Option<tap::Tap>,
}

impl MacOsBackend {
    /// Creates a backend that has installed no event tap.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(target_os = "macos")]
            tap: None,
        }
    }

    /// Decodes the events the run loop has ready.
    ///
    /// Returns the number of events decoded. Call it from the thread that
    /// opened the streams: the tap is attached to that thread's run loop, and
    /// CoreFoundation run loops are thread-local.
    ///
    /// Without a tap there is nothing to service, so the call reports zero and
    /// does not run the loop.
    #[cfg(target_os = "macos")]
    pub fn poll(&mut self) -> usize {
        match &mut self.tap {
            Some(tap) => tap.pump(),
            None => 0,
        }
    }
}

impl Backend for MacOsBackend {
    fn start(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
        #[cfg(target_os = "macos")]
        {
            if self
                .tap
                .as_ref()
                .is_some_and(|tap| tap.has(class))
            {
                return Ok(());
            }
            let (device, name) = match class {
                DeviceClass::Keyboard => (tap::KEYBOARD_DEVICE, "Keyboard"),
                DeviceClass::Pointer => (tap::POINTER_DEVICE, "Pointer"),
                _ => return Err(ControlError::Unsupported(class)),
            };
            if self.tap.is_none() {
                self.tap = Some(tap::Tap::create()?);
            }
            let info = DeviceInfo::new(device, class, String::from(name));
            sink.emit(InputEvent::DeviceAdded(info.clone()));
            let Some(tap) = &mut self.tap else {
                return Ok(());
            };
            tap.add(class, sink, info);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sink;
            Err(ControlError::Unsupported(class))
        }
    }

    fn stop(&mut self, class: DeviceClass) {
        #[cfg(target_os = "macos")]
        {
            let Some(tap) = &mut self.tap else {
                return;
            };
            tap.remove(class);
            if tap.devices().is_empty() {
                self.tap = None;
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = class;
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        #[cfg(target_os = "macos")]
        {
            self.tap
                .as_ref()
                .map(tap::Tap::devices)
                .unwrap_or_default()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::Backend;
    #[cfg(not(target_os = "macos"))]
    use codevar_kuitui_control::{Control, ControlError, DeviceClass};

    use super::MacOsBackend;

    #[test]
    fn nothing_is_reported_before_a_class_is_opened() {
        assert!(MacOsBackend::new().devices().is_empty());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_host_build_reports_every_class_as_unsupported() {
        let mut control = Control::new(MacOsBackend::new());
        assert!(matches!(
            control.open(DeviceClass::Keyboard),
            Err(ControlError::Unsupported(DeviceClass::Keyboard))
        ));
    }
}
