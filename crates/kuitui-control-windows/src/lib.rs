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
//! The backend listens through Win32 Raw Input on a message-only window that
//! it creates on the calling thread. It owns no thread: the caller pumps its
//! own message loop and calls `poll` to decode the records that arrived since
//! the previous call, all on the thread that opened the backend.
//!
//! Raw input is registered with `RIDEV_INPUTSINK`, so a consumer sees the
//! device stream even while another window has focus. That matches the Linux
//! backend, which reads the device nodes directly. Use one backend per thread:
//! the registration is process-wide, so a second window would take the stream
//! over from the first.
//!
//! A class is registered when its first stream opens and removed when the
//! last one closes. The keyboard and pointer are served; gamepads need a HID
//! report parser that the backend does not have yet, and microphones are not
//! raw input devices, so both are reported as unsupported.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg_attr(not(windows), allow(dead_code))]
mod decode;
#[cfg(windows)]
mod device;
#[cfg_attr(not(windows), allow(dead_code))]
mod keymap;
#[cfg(windows)]
mod raw;

#[cfg(windows)]
use alloc::string::String;
use alloc::vec::Vec;

use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceInfo, EventSink};
#[cfg(windows)]
use codevar_kuitui_control::{DeviceId, InputEvent};

/// Win32 Raw Input backend.
#[derive(Default)]
pub struct WindowsBackend {
    /// One entry per class that is registered.
    #[cfg(windows)]
    sessions: Vec<Session>,
    /// Message-only window the input is registered against.
    #[cfg(windows)]
    window: Option<raw::Window>,
    /// Queue buffer kept between reads.
    #[cfg(windows)]
    buffer: Vec<u64>,
    /// Device count observed at the previous poll.
    #[cfg(windows)]
    device_count: u32,
}

impl WindowsBackend {
    /// Creates a backend that has opened no devices.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(windows)]
            sessions: Vec::new(),
            #[cfg(windows)]
            window: None,
            #[cfg(windows)]
            buffer: Vec::new(),
            #[cfg(windows)]
            device_count: 0,
        }
    }

    /// Decodes every record that arrived since the previous call.
    ///
    /// Returns the number of raw records decoded. Call it from the thread
    /// that created the backend, after pumping that thread's message queue.
    #[cfg(windows)]
    pub fn poll(&mut self) -> Result<usize, ControlError> {
        let mut emitted = 0usize;
        let sessions = &mut self.sessions;
        raw::drain(&mut self.buffer, |record| {
            if dispatch(sessions, record) {
                emitted += 1;
            }
        })?;
        self.refresh()?;
        Ok(emitted)
    }

    /// Registers `class` against the message-only window.
    #[cfg(windows)]
    fn open(&mut self, class: DeviceClass, usage: u16, sink: EventSink) -> Result<(), ControlError> {
        let window = match &self.window {
            Some(window) => window.handle(),
            None => {
                let window = raw::Window::create()?;
                let handle = window.handle();
                self.window = Some(window);
                handle
            }
        };
        raw::register(usage, window)?;
        let devices = match device::enumerate(class) {
            Ok(devices) => devices,
            Err(error) => {
                raw::unregister(usage);
                return Err(error);
            }
        };
        let mut states = Vec::with_capacity(devices.len());
        for info in &devices {
            sink.emit(InputEvent::DeviceAdded(info.clone()));
            states.push(decode::DeviceState::new(info.id));
        }
        self.sessions.push(Session {
            class,
            sink,
            devices,
            states,
        });
        self.device_count = device::count().unwrap_or_default();
        Ok(())
    }

    /// Re-enumerates the devices after the system reports a different count.
    #[cfg(windows)]
    fn refresh(&mut self) -> Result<(), ControlError> {
        if self.sessions.is_empty() {
            return Ok(());
        }
        let count = device::count()?;
        if count == self.device_count {
            return Ok(());
        }
        self.device_count = count;
        for session in &mut self.sessions {
            session.refresh()?;
        }
        Ok(())
    }
}

impl Backend for WindowsBackend {
    fn start(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
        #[cfg(windows)]
        {
            if self
                .sessions
                .iter()
                .any(|session| session.class == class)
            {
                return Ok(());
            }
            let usage = match class {
                DeviceClass::Keyboard => raw::USAGE_KEYBOARD,
                DeviceClass::Pointer => raw::USAGE_MOUSE,
                _ => return Err(ControlError::Unsupported(class)),
            };
            self.open(class, usage, sink)
        }
        #[cfg(not(windows))]
        {
            let _ = sink;
            Err(ControlError::Unsupported(class))
        }
    }

    fn stop(&mut self, class: DeviceClass) {
        #[cfg(windows)]
        {
            let Some(index) = self
                .sessions
                .iter()
                .position(|session| session.class == class)
            else {
                return;
            };
            self.sessions.swap_remove(index);
            match class {
                DeviceClass::Keyboard => raw::unregister(raw::USAGE_KEYBOARD),
                DeviceClass::Pointer => raw::unregister(raw::USAGE_MOUSE),
                _ => {}
            }
            if self.sessions.is_empty() {
                self.window = None;
                self.buffer = Vec::new();
                self.device_count = 0;
            }
        }
        #[cfg(not(windows))]
        let _ = class;
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        #[cfg(windows)]
        {
            self.sessions
                .iter()
                .flat_map(|session| session.devices.iter().cloned())
                .collect()
        }
        #[cfg(not(windows))]
        {
            Vec::new()
        }
    }
}

/// One registered class.
#[cfg(windows)]
struct Session {
    /// Class the session serves.
    class: DeviceClass,
    /// Where its events go.
    sink: EventSink,
    /// Devices reported for the class.
    devices: Vec<DeviceInfo>,
    /// Decoding state of each device.
    states: Vec<decode::DeviceState>,
}

#[cfg(windows)]
impl Session {
    /// Applies one keyboard record.
    fn key(&mut self, device: DeviceId, record: decode::KeyRecord) {
        let sink = self.sink.clone();
        self.state(device)
            .key(record, &mut |event| sink.emit(event));
    }

    /// Applies one pointer record.
    fn mouse(&mut self, device: DeviceId, record: decode::MouseRecord) {
        let sink = self.sink.clone();
        self.state(device)
            .mouse(record, &mut |event| sink.emit(event));
    }

    /// The decoding state of `device`, adopted when it was not enumerated.
    fn state(&mut self, device: DeviceId) -> &mut decode::DeviceState {
        if let Some(index) = self
            .states
            .iter()
            .position(|state| state.id() == device)
        {
            return &mut self.states[index];
        }
        let info = DeviceInfo::new(device, self.class, String::new());
        self.sink
            .emit(InputEvent::DeviceAdded(info.clone()));
        self.devices.push(info);
        self.states.push(decode::DeviceState::new(device));
        let index = self.states.len() - 1;
        &mut self.states[index]
    }

    /// Replaces the device list with the one the system reports now.
    fn refresh(&mut self) -> Result<(), ControlError> {
        let devices = device::enumerate(self.class)?;
        for info in &devices {
            if !self
                .devices
                .iter()
                .any(|device| device.id == info.id)
            {
                self.sink
                    .emit(InputEvent::DeviceAdded(info.clone()));
                self.devices.push(info.clone());
                self.states
                    .push(decode::DeviceState::new(info.id));
            }
        }
        let sink = self.sink.clone();
        self.devices.retain(|device| {
            let present = devices.iter().any(|info| info.id == device.id);
            if !present {
                sink.emit(InputEvent::DeviceRemoved(device.id));
            }
            present
        });
        self.states
            .retain(|state| devices.iter().any(|info| info.id == state.id()));
        Ok(())
    }
}

/// Routes one record to the session that serves its class.
#[cfg(windows)]
fn dispatch(sessions: &mut [Session], record: raw::Record) -> bool {
    match record {
        raw::Record::Key { device, key } => {
            let Some(session) = sessions
                .iter_mut()
                .find(|session| session.class == DeviceClass::Keyboard)
            else {
                return false;
            };
            session.key(device, key);
            true
        }
        raw::Record::Mouse { device, mouse } => {
            let Some(session) = sessions
                .iter_mut()
                .find(|session| session.class == DeviceClass::Pointer)
            else {
                return false;
            };
            session.mouse(device, mouse);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::Backend;
    #[cfg(not(windows))]
    use codevar_kuitui_control::{Control, ControlError, DeviceClass};

    use super::WindowsBackend;

    #[test]
    fn nothing_is_reported_before_a_class_is_opened() {
        assert!(WindowsBackend::new().devices().is_empty());
    }

    #[cfg(not(windows))]
    #[test]
    fn a_host_build_reports_every_class_as_unsupported() {
        let mut control = Control::new(WindowsBackend::new());
        assert!(matches!(
            control.open(DeviceClass::Keyboard),
            Err(ControlError::Unsupported(DeviceClass::Keyboard))
        ));
    }
}
