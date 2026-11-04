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

//! One cooperative executor task per registered class.
//!
//! The task owns the device descriptors and translates their records into
//! [`InputEvent`]s. It never blocks: each poll drains every ready device until
//! the kernel reports `EAGAIN`, then returns [`Poll::Pending`] and waits for
//! the backend's `poll` method to wake it after `epoll` reports readiness.
//!
//! Devices discovered while the task runs arrive through the `inbox`; the
//! task adopts them on its next poll, which keeps hotplug inside the task
//! rather than in a second owner of the descriptors.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::Pin;
use core::task::{Context as TaskContext, Poll};

use codevar_kuitui_control::{
    DeviceClass, DeviceId, DeviceInfo, EventSink, InputEvent, KeyEvent, KeyState, Modifiers, Point,
    PointerButton, PointerEvent, Timestamp,
};

use crate::device::{BTN_LEFT, EV_ABS, EV_KEY, EV_REL, OpenDevice, REL_WHEEL};
use crate::fd::Fd;
use crate::keymap;

/// Event type that ends a batch of records.
const EV_SYN: u16 = 0x00;
/// `SYN_REPORT` ends a batch of records and is the point at which motion is
/// published.
const SYN_REPORT: u16 = 0;
/// Relative horizontal motion.
const REL_X: u16 = 0x00;
/// Relative vertical motion.
const REL_Y: u16 = 0x01;
/// Horizontal wheel motion.
const REL_HWHEEL: u16 = 0x06;
/// Absolute horizontal position.
const ABS_X: u16 = 0x00;
/// Absolute vertical position.
const ABS_Y: u16 = 0x01;
/// Horizontal position of a multitouch contact.
const ABS_MT_POSITION_X: u16 = 0x35;
/// Vertical position of a multitouch contact.
const ABS_MT_POSITION_Y: u16 = 0x36;
/// Secondary mouse button.
const BTN_RIGHT: u16 = 0x111;
/// Middle mouse button.
const BTN_MIDDLE: u16 = 0x112;
/// Back mouse button.
const BTN_SIDE: u16 = 0x113;
/// Forward mouse button.
const BTN_EXTRA: u16 = 0x114;
/// Number of records read in one syscall.
const READ_BATCH: usize = 64;
/// Upper bound on a record, used to size the read buffer.
const RECORD_MAX: usize = 32;
/// Caps lock key code.
const KEY_CAPSLOCK: u16 = 58;
/// Num lock key code.
const KEY_NUMLOCK: u16 = 69;

/// One `struct input_event`, decoded without depending on the libc layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Record {
    /// Kernel timestamp.
    timestamp: Timestamp,
    /// Event type.
    kind: u16,
    /// Event code.
    code: u16,
    /// Event value.
    value: i32,
}

/// Whether a read left the device open or discovered that it disappeared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadOutcome {
    /// The descriptor is still usable.
    Open,
    /// The kernel reported that the node is gone.
    Removed,
}

/// Which kind of motion the current batch has accumulated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Motion {
    /// No motion since the last report.
    #[default]
    Idle,
    /// Relative movement from a mouse.
    Relative,
    /// Absolute position from a tablet or touch surface.
    Absolute,
}

/// Per-device decoding state.
struct DeviceState {
    /// The non-blocking descriptor.
    fd: Fd,
    /// Stable identifier carried by every event.
    id: DeviceId,
    /// Modifiers and lock state held by this device.
    modifiers: Modifiers,
    /// Relative movement accumulated since the last report.
    delta: Point,
    /// Wheel movement accumulated since the last report.
    wheel: Point,
    /// Last absolute position reported by the device.
    position: Option<Point>,
    /// Motion accumulated since the last report.
    motion: Motion,
}

impl DeviceState {
    /// Wraps an opened device.
    fn new(device: OpenDevice) -> Self {
        Self {
            fd: device.fd,
            id: device.info.id,
            modifiers: Modifiers::NONE,
            delta: Point::ORIGIN,
            wheel: Point::ORIGIN,
            position: None,
            motion: Motion::Idle,
        }
    }

    /// Applies one kernel record.
    fn apply(&mut self, class: DeviceClass, sink: &EventSink, record: Record) {
        match record.kind {
            EV_SYN if record.code == SYN_REPORT => self.flush(class, sink, record.timestamp),
            EV_KEY if class == DeviceClass::Keyboard => self.key(sink, record),
            EV_KEY if class == DeviceClass::Pointer => self.button(sink, record),
            EV_REL if class == DeviceClass::Pointer => self.relative(record.code, record.value),
            EV_ABS if class == DeviceClass::Pointer => self.absolute(record.code, record.value),
            _ => {}
        }
    }

    /// Applies a key transition and publishes it with the resulting state.
    fn key(&mut self, sink: &EventSink, record: Record) {
        let Some(state) = key_state(record.value) else {
            return;
        };
        if state != KeyState::Repeat
            && let Some(modifier) = keymap::modifier(record.code)
        {
            if state == KeyState::Pressed {
                self.modifiers.insert(modifier);
            } else {
                self.modifiers.remove(modifier);
            }
        }
        if state == KeyState::Pressed {
            match record.code {
                KEY_CAPSLOCK => toggle(&mut self.modifiers, Modifiers::CAPS_LOCK),
                KEY_NUMLOCK => toggle(&mut self.modifiers, Modifiers::NUM_LOCK),
                _ => {}
            }
        }
        sink.emit_at(
            record.timestamp,
            InputEvent::Key(KeyEvent {
                device: self.id,
                code: keymap::key_code(record.code),
                state,
                modifiers: self.modifiers,
            }),
        );
    }

    /// Applies a pointer button transition.
    fn button(&mut self, sink: &EventSink, record: Record) {
        let Some(state) = key_state(record.value) else {
            return;
        };
        if state == KeyState::Repeat {
            return;
        }
        sink.emit_at(
            record.timestamp,
            InputEvent::Pointer(PointerEvent::Button {
                button: pointer_button(record.code),
                state,
                position: self.position,
                modifiers: self.modifiers,
            }),
        );
    }

    /// Accumulates one relative axis record.
    fn relative(&mut self, code: u16, value: i32) {
        match code {
            REL_X => {
                self.delta.x += value as f32;
                self.motion = Motion::Relative;
            }
            REL_Y => {
                self.delta.y += value as f32;
                self.motion = Motion::Relative;
            }
            REL_HWHEEL => self.wheel.x += value as f32,
            REL_WHEEL => self.wheel.y += value as f32,
            _ => {}
        }
    }

    /// Records one absolute position component.
    fn absolute(&mut self, code: u16, value: i32) {
        let position = self.position.get_or_insert(Point::ORIGIN);
        match code {
            ABS_X | ABS_MT_POSITION_X => position.x = value as f32,
            ABS_Y | ABS_MT_POSITION_Y => position.y = value as f32,
            _ => return,
        }
        self.motion = Motion::Absolute;
    }

    /// Publishes the motion and wheel state accumulated by one batch.
    fn flush(&mut self, class: DeviceClass, sink: &EventSink, timestamp: Timestamp) {
        if class == DeviceClass::Pointer {
            match self.motion {
                Motion::Relative => {
                    sink.emit_at(
                        timestamp,
                        InputEvent::Pointer(PointerEvent::Motion {
                            position: None,
                            delta: Some(self.delta),
                        }),
                    );
                    self.delta = Point::ORIGIN;
                }
                Motion::Absolute => {
                    if let Some(position) = self.position {
                        sink.emit_at(
                            timestamp,
                            InputEvent::Pointer(PointerEvent::Motion {
                                position: Some(position),
                                delta: None,
                            }),
                        );
                    }
                }
                Motion::Idle => {}
            }
            if self.wheel != Point::ORIGIN {
                sink.emit_at(
                    timestamp,
                    InputEvent::Pointer(PointerEvent::Scroll {
                        delta: self.wheel,
                        position: self.position,
                        modifiers: self.modifiers,
                    }),
                );
                self.wheel = Point::ORIGIN;
            }
        }
        self.motion = Motion::Idle;
    }
}

/// The future that serves one device class.
pub(crate) struct ClassTask {
    /// Class whose devices this task owns.
    class: DeviceClass,
    /// Fan-out point back into the control layer.
    sink: EventSink,
    /// Set when the class is no longer registered.
    stop: Rc<Cell<bool>>,
    /// Devices this task reads.
    devices: Vec<DeviceState>,
    /// Devices discovered after the task started.
    inbox: Rc<RefCell<Vec<OpenDevice>>>,
    /// Snapshot that [`crate::LinuxBackend::devices`] reads.
    published: Rc<RefCell<Vec<DeviceInfo>>>,
}

impl ClassTask {
    /// Creates a task for `class` from the devices found at registration.
    pub(crate) fn new(
        class: DeviceClass,
        sink: EventSink,
        devices: Vec<OpenDevice>,
        stop: Rc<Cell<bool>>,
        inbox: Rc<RefCell<Vec<OpenDevice>>>,
        published: Rc<RefCell<Vec<DeviceInfo>>>,
    ) -> Self {
        Self {
            class,
            sink,
            stop,
            devices: devices
                .into_iter()
                .map(DeviceState::new)
                .collect(),
            inbox,
            published,
        }
    }

    /// Publishes devices the driver discovered since the last poll.
    fn adopt(&mut self) {
        let pending = core::mem::take(&mut *self.inbox.borrow_mut());
        for device in pending {
            let info = device.info.clone();
            self.sink
                .emit(InputEvent::DeviceAdded(info.clone()));
            publish(&self.published, info);
            self.devices.push(DeviceState::new(device));
        }
    }

    /// Reads every queued record of every device and retires the vanished
    /// ones.
    fn drain(&mut self) {
        let mut index = 0;
        while index < self.devices.len() {
            if read_available(&mut self.devices[index], self.class, &self.sink) == ReadOutcome::Removed {
                let device = self.devices.swap_remove(index);
                self.sink
                    .emit(InputEvent::DeviceRemoved(device.id));
                unpublish(&self.published, device.id);
                codevar_logger::log_debug!("input: device {} disappeared", device.id.get());
            } else {
                index += 1;
            }
        }
    }
}

impl Future for ClassTask {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _context: &mut TaskContext<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.stop.get() {
            return Poll::Ready(());
        }
        this.adopt();
        this.drain();
        Poll::Pending
    }
}

/// Adds or replaces one entry in the snapshot.
fn publish(published: &RefCell<Vec<DeviceInfo>>, info: DeviceInfo) {
    let mut devices = published.borrow_mut();
    devices.retain(|known| known.id != info.id);
    devices.push(info);
}

/// Removes one entry from the snapshot.
fn unpublish(published: &RefCell<Vec<DeviceInfo>>, id: DeviceId) {
    published
        .borrow_mut()
        .retain(|known| known.id != id);
}

/// Reads every record currently queued on `state`.
fn read_available(state: &mut DeviceState, class: DeviceClass, sink: &EventSink) -> ReadOutcome {
    let mut buffer = [0u8; RECORD_MAX * READ_BATCH];
    loop {
        let read = match state.fd.read(&mut buffer) {
            Ok(0) => return ReadOutcome::Removed,
            Ok(read) => read,
            Err(error) if error.is(libc::EAGAIN) => return ReadOutcome::Open,
            Err(error) if error.is(libc::EINTR) => continue,
            Err(error) if error.is(libc::ENODEV) || error.is(libc::EIO) => {
                return ReadOutcome::Removed;
            }
            Err(error) => {
                codevar_logger::log_warn!("input: could not read device {}: {error}", state.id.get());
                return ReadOutcome::Open;
            }
        };
        let record_size = record_size();
        let complete = read - read % record_size;
        for chunk in buffer[..complete].chunks_exact(record_size) {
            if let Some(record) = read_record(chunk) {
                state.apply(class, sink, record);
            }
        }
        if complete != read {
            codevar_logger::log_debug!("input: ignored a short record on device {}", state.id.get());
        }
    }
}

/// The size of one `struct input_event` on this target.
const fn record_size() -> usize {
    size_of::<libc::timeval>() + 8
}

/// Decodes one record from its native byte representation.
fn read_record(bytes: &[u8]) -> Option<Record> {
    let (sec, usec) = if size_of::<libc::time_t>() == 8 {
        (
            i64::from_ne_bytes(bytes.get(0..8)?.try_into().ok()?),
            i64::from_ne_bytes(bytes.get(8..16)?.try_into().ok()?),
        )
    } else {
        (
            i64::from(i32::from_ne_bytes(bytes.get(0..4)?.try_into().ok()?)),
            i64::from(i32::from_ne_bytes(bytes.get(4..8)?.try_into().ok()?)),
        )
    };
    let offset = size_of::<libc::timeval>();
    let kind = u16::from_ne_bytes(bytes.get(offset..offset + 2)?.try_into().ok()?);
    let code = u16::from_ne_bytes(
        bytes
            .get(offset + 2..offset + 4)?
            .try_into()
            .ok()?,
    );
    let value = i32::from_ne_bytes(
        bytes
            .get(offset + 4..offset + 8)?
            .try_into()
            .ok()?,
    );
    let secs = u64::try_from(sec).unwrap_or(0);
    let micros = u64::try_from(usec).unwrap_or(0);
    Some(Record {
        timestamp: Timestamp::from_micros(
            secs.saturating_mul(1_000_000)
                .saturating_add(micros),
        ),
        kind,
        code,
        value,
    })
}

/// Maps a kernel key value to a transition.
fn key_state(value: i32) -> Option<KeyState> {
    match value {
        0 => Some(KeyState::Released),
        1 => Some(KeyState::Pressed),
        2 => Some(KeyState::Repeat),
        _ => None,
    }
}

/// Maps a mouse button code to the portable button name.
fn pointer_button(code: u16) -> PointerButton {
    match code {
        BTN_LEFT => PointerButton::Primary,
        BTN_RIGHT => PointerButton::Secondary,
        BTN_MIDDLE => PointerButton::Middle,
        BTN_SIDE => PointerButton::Back,
        BTN_EXTRA => PointerButton::Forward,
        other => PointerButton::Other(other),
    }
}

/// Flips one lock flag.
fn toggle(modifiers: &mut Modifiers, flag: Modifiers) {
    if modifiers.contains(flag) {
        modifiers.remove(flag);
    } else {
        modifiers.insert(flag);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use codevar_kuitui_control::{KeyState, PointerButton, Timestamp};

    use super::{
        BTN_EXTRA, BTN_LEFT, BTN_MIDDLE, BTN_RIGHT, BTN_SIDE, EV_KEY, key_state, pointer_button, read_record,
    };

    fn record_bytes(kind: u16, code: u16, value: i32) -> Vec<u8> {
        let time_size = size_of::<libc::timeval>();
        let mut bytes = alloc::vec![0u8; time_size + 8];
        if size_of::<libc::time_t>() == 8 {
            bytes[0..8].copy_from_slice(&7i64.to_ne_bytes());
            bytes[8..16].copy_from_slice(&500_000i64.to_ne_bytes());
        } else {
            bytes[0..4].copy_from_slice(&7i32.to_ne_bytes());
            bytes[4..8].copy_from_slice(&500_000i32.to_ne_bytes());
        }
        bytes[time_size..time_size + 2].copy_from_slice(&kind.to_ne_bytes());
        bytes[time_size + 2..time_size + 4].copy_from_slice(&code.to_ne_bytes());
        bytes[time_size + 4..time_size + 8].copy_from_slice(&value.to_ne_bytes());
        bytes
    }

    #[test]
    fn a_record_carries_the_kernel_timestamp_and_fields() {
        let bytes = record_bytes(EV_KEY, 30, 1);
        let record = read_record(&bytes).unwrap();
        assert_eq!(record.timestamp, Timestamp::from_micros(7_500_000));
        assert_eq!(record.kind, EV_KEY);
        assert_eq!(record.code, 30);
        assert_eq!(record.value, 1);
    }

    #[test]
    fn truncated_records_are_rejected_instead_of_panicking() {
        assert!(read_record(&[]).is_none());
        assert!(read_record(&[0u8; 4]).is_none());
    }

    #[test]
    fn key_values_map_to_transitions() {
        assert_eq!(key_state(0), Some(KeyState::Released));
        assert_eq!(key_state(1), Some(KeyState::Pressed));
        assert_eq!(key_state(2), Some(KeyState::Repeat));
        assert_eq!(key_state(3), None);
    }

    #[test]
    fn mouse_buttons_keep_their_identity() {
        assert_eq!(pointer_button(BTN_LEFT), PointerButton::Primary);
        assert_eq!(pointer_button(BTN_RIGHT), PointerButton::Secondary);
        assert_eq!(pointer_button(BTN_MIDDLE), PointerButton::Middle);
        assert_eq!(pointer_button(BTN_SIDE), PointerButton::Back);
        assert_eq!(pointer_button(BTN_EXTRA), PointerButton::Forward);
        assert_eq!(pointer_button(0x120), PointerButton::Other(0x120));
    }
}
