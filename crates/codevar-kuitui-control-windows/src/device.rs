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

//! Raw Input device enumeration.
//!
//! Win32 exposes a device's type and, for HID collections, its usage page,
//! usage, vendor, and product identifiers. It does not expose a friendly
//! name, so every device is reported with an empty name.

use alloc::string::String;
use alloc::vec::Vec;
use core::mem::size_of;

use codevar_kuitui_control::{ControlError, DeviceClass, DeviceId, DeviceInfo};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::UI::Input::{
    GetRawInputDeviceInfoW, GetRawInputDeviceList, RAWINPUTDEVICELIST, RID_DEVICE_INFO, RIDI_DEVICEINFO,
    RIM_TYPEHID, RIM_TYPEKEYBOARD, RIM_TYPEMOUSE,
};

/// The Generic Desktop usage page.
const GENERIC_DESKTOP: u16 = 0x01;

/// The number of devices the system currently reports.
///
/// The count changes when a device is plugged or unplugged, which makes it a
/// cheap reason to look at the full list again.
pub(crate) fn count() -> Result<u32, ControlError> {
    let mut count = 0u32;
    // SAFETY: the null list requests the count, which the system writes into
    // `count`; `size_of` describes the list element type the function expects.
    let result = unsafe { GetRawInputDeviceList(None, &mut count, size_of::<RAWINPUTDEVICELIST>() as u32) };
    if result == u32::MAX {
        return Err(ControlError::backend("raw input device count failed"));
    }
    Ok(count)
}

/// The devices that belong to `class`.
pub(crate) fn enumerate(class: DeviceClass) -> Result<Vec<DeviceInfo>, ControlError> {
    let mut devices = Vec::new();
    for raw in list()? {
        if let Some(info) = describe(&raw)
            && info.class == class
        {
            devices.push(info);
        }
    }
    Ok(devices)
}

/// Every device the system reports.
fn list() -> Result<Vec<RAWINPUTDEVICELIST>, ControlError> {
    let mut count = count()?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut devices = alloc::vec![RAWINPUTDEVICELIST::default(); count as usize];
    // SAFETY: `devices` has room for `count` entries and `count` describes
    // that capacity, as the input meaning of the parameter requires.
    let written = unsafe {
        GetRawInputDeviceList(
            Some(devices.as_mut_ptr()),
            &mut count,
            size_of::<RAWINPUTDEVICELIST>() as u32,
        )
    };
    if written == u32::MAX {
        return Err(ControlError::backend("raw input device list failed"));
    }
    devices.truncate(count as usize);
    Ok(devices)
}

/// Describes one listed device, or ignores it when it is not an input class
/// the backend serves.
fn describe(device: &RAWINPUTDEVICELIST) -> Option<DeviceInfo> {
    if device.dwType == RIM_TYPEKEYBOARD {
        Some(DeviceInfo::new(
            identifier(device.hDevice),
            DeviceClass::Keyboard,
            String::new(),
        ))
    } else if device.dwType == RIM_TYPEMOUSE {
        Some(DeviceInfo::new(
            identifier(device.hDevice),
            DeviceClass::Pointer,
            String::new(),
        ))
    } else if device.dwType == RIM_TYPEHID {
        gamepad(device.hDevice)
    } else {
        None
    }
}

/// Describes a HID collection that is a game controller.
fn gamepad(handle: HANDLE) -> Option<DeviceInfo> {
    let mut info = RID_DEVICE_INFO {
        cbSize: size_of::<RID_DEVICE_INFO>() as u32,
        ..Default::default()
    };
    let mut size = size_of::<RID_DEVICE_INFO>() as u32;
    // SAFETY: `info` is a writable structure of the declared size and `handle`
    // came from the same enumeration that produced `device`.
    let result = unsafe {
        GetRawInputDeviceInfoW(
            Some(handle),
            RIDI_DEVICEINFO,
            Some((&mut info as *mut RID_DEVICE_INFO).cast()),
            &mut size,
        )
    };
    if result == u32::MAX || info.dwType != RIM_TYPEHID {
        return None;
    }
    // SAFETY: `dwType` reported the HID variant, so the union holds it.
    let hid = unsafe { info.Anonymous.hid };
    if hid.usUsagePage != GENERIC_DESKTOP || !is_game_controller(hid.usUsage) {
        return None;
    }
    let id = identifier(handle);
    Some(DeviceInfo::new(id, DeviceClass::Gamepad, String::new()).with_ids(
        u16::try_from(hid.dwVendorId).unwrap_or_default(),
        u16::try_from(hid.dwProductId).unwrap_or_default(),
    ))
}

/// Whether a Generic Desktop usage describes a game controller.
fn is_game_controller(usage: u16) -> bool {
    (4..=8).contains(&usage)
}

/// The portable identifier of a device handle.
fn identifier(handle: HANDLE) -> DeviceId {
    DeviceId::new(handle.0 as usize as u64)
}
