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
//! The backend reads `/dev/input/event*` directly through evdev. It owns no
//! thread: the registered classes are cooperative tasks on a
//! [`codevar_inssched`] executor, and the caller drives them with
//! [`LinuxBackend::poll`] from its own event loop. A caller that already polls
//! descriptors can add [`LinuxBackend::fd`] to its poll set and then call
//! `poll(0)` without waiting again.
//!
//! Hotplug is discovered by rescanning the input directory whenever a `poll`
//! call times out, so a bounded timeout (a few hundred milliseconds) is what
//! turns a plugged device into [`DeviceAdded`](codevar_kuitui_control::InputEvent)
//! and [`DeviceRemoved`](codevar_kuitui_control::InputEvent) events.
//!
//! This backend does not go through the XDG desktop portal. The portal's
//! RemoteDesktop interface is the *implementation* side of a remote-desktop
//! session: it injects events into a compositor instead of delivering the
//! events of local hardware. Reading evdev is the facility a client uses for
//! local input, and it is also what the browser and compositor paths reduce
//! to.
//!
//! Audio capture needs a separate audio stack (PipeWire, PulseAudio or
//! ALSA), so [`DeviceClass::Microphone`] is reported as unsupported rather
//! than silently returning no events.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod device;
mod fd;
mod keymap;
mod task;

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::ffi::c_int;

use codevar_inssched::executor::{Config, Executor};
use codevar_inssched::policy::RoundRobin;
use codevar_inssched::task::{Priority, TaskId};
use codevar_kuitui_control::{Backend, ControlError, DeviceClass, DeviceId, DeviceInfo, EventSink};
use codevar_logger::log_warn;

use crate::device::OpenDevice;
use crate::fd::{Fd, SysError, check_int};
use crate::task::ClassTask;

/// Number of `epoll_event` entries read in one wait.
const EVENT_BATCH: usize = 32;

/// Linux evdev input backend.
pub struct LinuxBackend {
    /// One cooperative task per class that has been opened.
    sessions: Vec<Session>,
    /// Executor that polls the class tasks.
    executor: Executor<RoundRobin>,
    /// Readiness set shared by every registered device.
    epoll: Option<Fd>,
}

/// The state owned by one registered class.
struct Session {
    /// Class the task serves.
    class: DeviceClass,
    /// Set to end the task.
    stop: Rc<Cell<bool>>,
    /// The executor task that reads the devices.
    task: TaskId,
    /// Devices discovered after the task started.
    inbox: Rc<RefCell<Vec<OpenDevice>>>,
    /// Devices the task currently owns.
    devices: Rc<RefCell<Vec<DeviceInfo>>>,
}

impl LinuxBackend {
    /// Creates a backend that has opened no devices.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            executor: Executor::new(RoundRobin, Config::new(1, 1)),
            epoll: None,
        }
    }

    /// Waits for input and delivers every event that is ready.
    ///
    /// The caller runs this from its event loop. `timeout_ms` is passed to
    /// `epoll_wait`: `0` returns immediately, a negative value waits until
    /// input arrives, and a bounded value also allows the timeout path to
    /// rescan for hotplug. When the call is used as the only wait in the
    /// program, a bounded timeout such as 250 ms keeps device changes timely.
    pub fn poll(&mut self, timeout_ms: c_int) {
        let Some(epoll) = &self.epoll else {
            return;
        };
        let timed_out = match epoll_wait(epoll, timeout_ms) {
            Ok(0) => true,
            Ok(_) => false,
            Err(error) if error.is(libc::EINTR) => false,
            Err(error) => {
                log_warn!("input: epoll wait failed: {error}");
                return;
            }
        };
        if timed_out && timeout_ms != 0 {
            self.rescan();
        }
        for session in &self.sessions {
            let _ = self.executor.wake(session.task);
        }
        self.executor.run_until_idle();
    }

    /// The descriptor to add to the caller's own poll set, if a class is
    /// registered.
    ///
    /// After the descriptor reports readiness, `poll(0)` delivers the events
    /// without waiting again.
    #[must_use]
    pub fn fd(&self) -> Option<c_int> {
        self.epoll.as_ref().map(Fd::raw)
    }

    /// Publishes devices that appeared since the last scan.
    fn rescan(&mut self) {
        for index in 0..self.sessions.len() {
            let class = self.sessions[index].class;
            let scanned = match device::scan(Some(class)) {
                Ok(devices) => devices,
                Err(error) => {
                    log_warn!("input: rescan failed: {error}");
                    continue;
                }
            };
            let known: Vec<DeviceId> = {
                let devices = self.sessions[index].devices.borrow();
                let inbox = self.sessions[index].inbox.borrow();
                devices
                    .iter()
                    .map(|info| info.id)
                    .chain(inbox.iter().map(|device| device.info.id))
                    .collect()
            };
            let mut fresh = Vec::new();
            for device in scanned {
                if known.contains(&device.info.id) {
                    continue;
                }
                if let Some(epoll) = &self.epoll
                    && let Err(error) = epoll_add(epoll, &device.fd, index as u64)
                {
                    log_warn!("input: could not watch device {}: {error}", device.info.id.get());
                    continue;
                }
                fresh.push(device);
            }
            self.sessions[index]
                .inbox
                .borrow_mut()
                .extend(fresh);
        }
    }

    /// Ends one task and waits for it to release its devices.
    fn release(&mut self, index: usize) {
        let session = self.sessions.swap_remove(index);
        session.stop.set(true);
        let _ = self.executor.wake(session.task);
        self.executor.run_until_idle();
    }
}

impl Default for LinuxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for LinuxBackend {
    fn start(&mut self, class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
        if self
            .sessions
            .iter()
            .any(|session| session.class == class)
        {
            return Ok(());
        }
        if class == DeviceClass::Microphone {
            return Err(ControlError::Unsupported(class));
        }

        if self.epoll.is_none() {
            self.epoll = Some(epoll_create().map_err(ControlError::backend)?);
        }
        let Some(epoll) = &self.epoll else {
            return Err(ControlError::backend("epoll was not created"));
        };

        let devices = device::scan(Some(class))?;
        let mut opened = Vec::with_capacity(devices.len());
        for device in devices {
            match epoll_add(epoll, &device.fd, self.sessions.len() as u64) {
                Ok(()) => opened.push(device),
                Err(error) => log_warn!("input: could not watch device {}: {error}", device.info.id.get()),
            }
        }

        let published = Rc::new(RefCell::new(
            opened
                .iter()
                .map(|device| device.info.clone())
                .collect(),
        ));
        let inbox = Rc::new(RefCell::new(Vec::new()));
        let stop = Rc::new(Cell::new(false));
        let task = ClassTask::new(
            class,
            sink,
            opened,
            Rc::clone(&stop),
            Rc::clone(&inbox),
            Rc::clone(&published),
        );
        let task = self
            .executor
            .spawn("kuitui-input", Priority::NORMAL, task);
        self.sessions.push(Session {
            class,
            stop,
            task,
            inbox,
            devices: published,
        });
        Ok(())
    }

    fn stop(&mut self, class: DeviceClass) {
        if let Some(index) = self
            .sessions
            .iter()
            .position(|session| session.class == class)
        {
            self.release(index);
        }
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        if self.sessions.is_empty() {
            return match device::scan(None) {
                Ok(devices) => devices
                    .into_iter()
                    .map(|device| device.info)
                    .collect(),
                Err(error) => {
                    log_warn!("input: could not enumerate evdev devices: {error}");
                    Vec::new()
                }
            };
        }

        let mut devices = Vec::new();
        for session in &self.sessions {
            devices.extend(session.devices.borrow().iter().cloned());
        }
        devices
    }
}

/// Creates an epoll descriptor.
fn epoll_create() -> Result<Fd, SysError> {
    // SAFETY: the call has no pointer arguments.
    let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    // SAFETY: `fd` is a new descriptor that no other value owns on success.
    check_int(fd, "epoll_create1").map(Fd::from_raw)
}

/// Watches `device` for readable data under `token`.
fn epoll_add(epoll: &Fd, device: &Fd, token: u64) -> Result<(), SysError> {
    let mut event = libc::epoll_event {
        events: libc::EPOLLIN as u32,
        u64: token,
    };
    // SAFETY: both descriptors are open and `event` is a valid, aligned
    // `epoll_event` for the duration of the call.
    let result = unsafe { libc::epoll_ctl(epoll.raw(), libc::EPOLL_CTL_ADD, device.raw(), &mut event) };
    check_int(result, "epoll_ctl").map(|_| ())
}

/// Waits for readable descriptors.
fn epoll_wait(epoll: &Fd, timeout_ms: c_int) -> Result<usize, SysError> {
    let mut events: [libc::epoll_event; EVENT_BATCH] =
        core::array::from_fn(|_| libc::epoll_event { events: 0, u64: 0 });
    // SAFETY: `events` is valid for writes of `EVENT_BATCH` elements and that
    // length is passed in the same call.
    let result =
        unsafe { libc::epoll_wait(epoll.raw(), events.as_mut_ptr(), EVENT_BATCH as c_int, timeout_ms) };
    check_int(result, "epoll_wait").map(|ready| ready as usize)
}

#[cfg(test)]
mod tests {
    use codevar_kuitui_control::{Backend, Control, ControlError, DeviceClass};

    use super::LinuxBackend;

    #[test]
    fn enumeration_never_panics_in_a_headless_environment() {
        let backend = LinuxBackend::new();
        let _ = backend.devices();
    }

    #[test]
    fn microphone_is_reported_as_unsupported() {
        let mut control = Control::new(LinuxBackend::new());
        assert!(matches!(
            control.open(DeviceClass::Microphone),
            Err(ControlError::Unsupported(DeviceClass::Microphone))
        ));
    }

    #[test]
    fn an_empty_class_registers_and_releases_cleanly() {
        let mut control = Control::new(LinuxBackend::new());
        match control.open(DeviceClass::Gamepad) {
            Ok(_stream) => {
                assert!(control.is_registered(DeviceClass::Gamepad));
                control.unregister(DeviceClass::Gamepad).unwrap();
                assert!(!control.is_registered(DeviceClass::Gamepad));
            }
            Err(ControlError::PermissionDenied { .. }) => {}
            Err(error) => panic!("unexpected start failure: {error}"),
        }
    }
}
