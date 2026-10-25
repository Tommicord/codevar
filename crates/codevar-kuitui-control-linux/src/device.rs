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

//! Discovery and classification of evdev device nodes.
//!
//! A node is opened read-only and non-blocking, then asked for its event
//! type and key bitmaps. Those bitmaps decide which [`DeviceClass`] the node
//! belongs to, so a caller that opens the pointer class never opens a
//! keyboard node.

use std::ffi::CString;
use std::fs;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use codevar_kuitui_control::{ControlError, DeviceCapability, DeviceClass, DeviceId, DeviceInfo};

/// Directory that holds the evdev character devices.
pub(crate) const INPUT_DIR: &str = "/dev/input";

/// Event type reported by a key or button record.
pub(crate) const EV_KEY: u16 = 0x01;
/// Event type reported by a relative axis record.
pub(crate) const EV_REL: u16 = 0x02;
/// Event type reported by an absolute axis record.
pub(crate) const EV_ABS: u16 = 0x03;

/// Highest event type; the bitmaps are sized from it.
const EV_MAX: u16 = 0x1f;
/// Highest key code.
const KEY_MAX: u16 = 0x2ff;
/// Highest relative axis code.
const REL_MAX: u16 = 0x0f;

/// Vertical wheel motion.
pub(crate) const REL_WHEEL: u16 = 0x08;

/// Primary mouse button.
pub(crate) const BTN_LEFT: u16 = 0x110;

/// First joystick button.
const BTN_JOYSTICK: u16 = 0x120;
/// Last directional-pad button.
const BTN_DPAD_RIGHT: u16 = 0x13f;

/// First letter key.
const KEY_A: u16 = 30;
/// Last letter key.
const KEY_Z: u16 = 50;
/// The enter key.
const KEY_ENTER: u16 = 28;

const IOC_READ: u32 = 2;

/// Encodes a Linux `_IOC` request.
const fn ioc(dir: u32, kind: u8, number: u8, size: usize) -> libc::c_ulong {
    ((dir << 30) | ((kind as u32) << 8) | (number as u32) | ((size as u32) << 16)) as libc::c_ulong
}

/// `EVIOCGBIT(ev, len)`: the bitmap of codes the device reports for `ev`.
const fn eviocgbit(ev: u16, len: usize) -> libc::c_ulong {
    ioc(IOC_READ, b'E', 0x20 + ev as u8, len)
}

/// `EVIOCGNAME(len)`: the device name as a NUL-terminated byte string.
const fn eviocgname(len: usize) -> libc::c_ulong {
    ioc(IOC_READ, b'E', 0x06, len)
}

/// `EVIOCGID`: the device's bus, vendor and product identifiers.
const fn eviocgid() -> libc::c_ulong {
    ioc(IOC_READ, b'E', 0x02, size_of::<libc::input_id>())
}

/// A bitmap returned by `EVIOCGBIT`.
struct Bits(Vec<u8>);

impl Bits {
    /// Reads one bitmap from `fd`, or `None` when the kernel rejects it.
    fn read(fd: &OwnedFd, event_type: u16, len: usize) -> Option<Self> {
        let mut bytes = vec![0u8; len];
        // SAFETY: `bytes` is valid for writes of `len` bytes and the request
        // encodes exactly that length, so the kernel cannot write past it.
        let result = unsafe { libc::ioctl(fd.as_raw_fd(), eviocgbit(event_type, len), bytes.as_mut_ptr()) };
        (result >= 0).then_some(Self(bytes))
    }

    /// Whether the bitmap contains `code`.
    fn get(&self, code: u16) -> bool {
        let index = code as usize;
        self.0
            .get(index / 8)
            .is_some_and(|byte| byte & (1 << (index % 8)) != 0)
    }

    /// Whether the bitmap contains any code in `range`.
    fn any_in(&self, range: core::ops::RangeInclusive<u16>) -> bool {
        range.into_iter().any(|code| self.get(code))
    }
}

/// Opens a node with the flags the worker needs.
fn open_node(path: &Path) -> io::Result<OwnedFd> {
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: `path` is a valid NUL-terminated string and the return value is
    // checked before it is turned into an `OwnedFd`.
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` is a freshly opened descriptor that no other value owns.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Reads the NUL-terminated device name.
fn read_name(fd: &OwnedFd) -> String {
    const NAME_LEN: usize = 256;
    let mut bytes = [0u8; NAME_LEN];
    // SAFETY: the buffer is `NAME_LEN` bytes and the request encodes that
    // length, so the kernel copies at most the buffer size.
    let result = unsafe { libc::ioctl(fd.as_raw_fd(), eviocgname(NAME_LEN), bytes.as_mut_ptr()) };
    if result < 0 {
        return String::new();
    }
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(NAME_LEN);
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Reads the bus, vendor and product identifiers.
fn read_ids(fd: &OwnedFd) -> libc::input_id {
    let mut ids = libc::input_id {
        bustype: 0,
        vendor: 0,
        product: 0,
        version: 0,
    };
    // SAFETY: `ids` is a live, properly aligned `input_id` and the request
    // size matches `size_of::<input_id>()`.
    let _ = unsafe { libc::ioctl(fd.as_raw_fd(), eviocgid(), &mut ids) };
    ids
}

/// Returns the device number, which stays stable while the node is connected.
fn read_device_number(fd: &OwnedFd) -> u64 {
    let mut stat = core::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `stat` points to writable memory of the right type and `fd` is
    // an open descriptor.
    let result = unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    // SAFETY: `fstat` succeeded, which initializes the whole structure.
    let stat = unsafe { stat.assume_init() };
    stat.st_rdev
}

/// Decides the class and the capability set of a node.
fn classify(fd: &OwnedFd) -> Option<(DeviceClass, DeviceCapability)> {
    let events = Bits::read(fd, 0, EV_MAX as usize / 8 + 1)?;
    let keys = Bits::read(fd, EV_KEY, KEY_MAX as usize / 8 + 1)?;

    if events.get(EV_ABS) && keys.any_in(BTN_JOYSTICK..=BTN_DPAD_RIGHT) {
        return Some((DeviceClass::Gamepad, DeviceCapability::GAMEPAD));
    }
    if keys.get(BTN_LEFT) && (events.get(EV_REL) || events.get(EV_ABS)) {
        let mut capabilities = DeviceCapability::POINTER_MOTION | DeviceCapability::POINTER_BUTTONS;
        if Bits::read(fd, EV_REL, REL_MAX as usize / 8 + 1).is_some_and(|relative| relative.get(REL_WHEEL)) {
            capabilities |= DeviceCapability::SCROLL;
        }
        return Some((DeviceClass::Pointer, capabilities));
    }
    if events.get(EV_KEY) && (keys.any_in(KEY_A..=KEY_Z) || keys.get(KEY_ENTER)) {
        return Some((DeviceClass::Keyboard, DeviceCapability::KEY));
    }
    None
}

/// Describes one node, or `None` when it is not of `wanted`.
fn describe(fd: &OwnedFd, wanted: Option<DeviceClass>) -> Option<DeviceInfo> {
    let (class, capabilities) = classify(fd)?;
    if wanted.is_some_and(|wanted| wanted != class) {
        return None;
    }
    let name = read_name(fd);
    let ids = read_ids(fd);
    let id = DeviceId::new(read_device_number(fd));
    Some(
        DeviceInfo::new(id, class, name)
            .with_ids(ids.vendor, ids.product)
            .with_capabilities(capabilities),
    )
}

/// Opens every node of `wanted`, or every node when it is `None`.
pub(crate) fn scan(wanted: Option<DeviceClass>) -> Result<Vec<DeviceInfo>, ControlError> {
    let entries = match fs::read_dir(INPUT_DIR) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            return Err(ControlError::permission_denied(INPUT_DIR));
        }
        Err(error) => return Err(ControlError::backend(error)),
    };

    let mut devices = Vec::new();
    let mut denied = 0usize;
    for entry in entries.flatten() {
        if !entry.file_name().as_bytes().starts_with(b"event") {
            continue;
        }
        match open_node(&entry.path()) {
            Ok(fd) => devices.extend(describe(&fd, wanted)),
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => denied += 1,
            Err(error) => {
                codevar_logger::log_debug!("input: skipping {}: {error}", entry.path().display());
            }
        }
    }

    if devices.is_empty() && denied > 0 {
        return Err(ControlError::permission_denied(INPUT_DIR));
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::{Bits, EV_ABS, EV_KEY, REL_WHEEL, ioc};

    #[test]
    fn ioctl_requests_match_the_kernel_encoding() {
        assert_eq!(super::eviocgname(256), 0x8100_4506);
        assert_eq!(super::eviocgid(), 0x8008_4502);
        assert_eq!(super::eviocgbit(EV_KEY, 96), 0x8060_4521);
        assert_eq!(ioc(2, b'E', 0x06, 256), 0x8100_4506);
    }

    #[test]
    fn bitmap_bits_are_little_endian_within_each_byte() {
        let bits = Bits(vec![0b0000_0010, 0b0000_0001]);
        assert!(bits.get(1));
        assert!(bits.get(8));
        assert!(!bits.get(0));
        assert!(!bits.get(9));
        assert!(!bits.get(400));
        assert!(bits.any_in(0..=1));
        assert!(!bits.any_in(2..=7));
    }

    #[test]
    fn constants_match_linux_input_h() {
        assert_eq!(EV_ABS, 0x03);
        assert_eq!(REL_WHEEL, 0x08);
    }
}
