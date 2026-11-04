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

//! The message-only window and the raw input queue behind it.
//!
//! Windows delivers raw records to the thread that owns the target window,
//! so the backend creates the window on the caller's thread and reads the
//! queue from [`drain`]. No message is dispatched for input: the buffered read
//! removes the corresponding `WM_INPUT` messages itself, which keeps a
//! high-frequency mouse from forcing one round trip per sample.

use alloc::vec::Vec;
use core::mem::{size_of, size_of_val};

use codevar_kuitui_control::{ControlError, DeviceId};
use windows::Win32::Foundation::{
    ERROR_CLASS_ALREADY_EXISTS, GetLastError, HANDLE, HINSTANCE, HWND, LPARAM, LRESULT, WPARAM,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::{
    GetRawInputBuffer, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RIDEV_INPUTSINK, RIDEV_REMOVE,
    RIM_TYPEKEYBOARD, RIM_TYPEMOUSE, RegisterRawInputDevices,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, HWND_MESSAGE, RegisterClassW, WINDOW_EX_STYLE,
    WINDOW_STYLE, WNDCLASSW,
};
use windows::core::PCWSTR;

use crate::decode::{KeyRecord, MouseRecord};

/// The Generic Desktop usage page.
const USAGE_PAGE_GENERIC: u16 = 0x01;
/// The Generic Desktop usage of the mouse collection.
pub(crate) const USAGE_MOUSE: u16 = 0x02;
/// The Generic Desktop usage of the keyboard collection.
pub(crate) const USAGE_KEYBOARD: u16 = 0x06;

/// Initial queue buffer, matching the size of the documented example.
const INITIAL_WORDS: usize = 64 * size_of::<RAWINPUT>() / size_of::<u64>();
/// Upper bound for the queue buffer, so a corrupt count cannot grow forever.
const MAX_WORDS: usize = 64 * 1024;

/// One record read from the queue.
pub(crate) enum Record {
    /// A keyboard transition.
    Key {
        /// Device that produced it.
        device: DeviceId,
        /// The transition.
        key: KeyRecord,
    },
    /// A pointer transition.
    Mouse {
        /// Device that produced it.
        device: DeviceId,
        /// The transition.
        mouse: MouseRecord,
    },
}

/// The hidden window that raw input is registered against.
pub(crate) struct Window(HWND);

impl Window {
    /// Creates the message-only window on the calling thread.
    pub(crate) fn create() -> Result<Self, ControlError> {
        let module = unsafe { GetModuleHandleW(PCWSTR::null()) }
            .map_err(|_| ControlError::backend("module handle unavailable"))?;
        let instance = HINSTANCE(module.0);
        let name: Vec<u16> = "CodevarRawInput\0".encode_utf16().collect();
        let class = WNDCLASSW {
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            lpszClassName: PCWSTR::from_raw(name.as_ptr()),
            ..Default::default()
        };
        // SAFETY: `class` points at the class name for the whole call, and the
        // class is registered on the thread that will own the window.
        let atom = unsafe { RegisterClassW(&class) };
        if atom == 0 {
            // SAFETY: no arguments and no caller state involved.
            let error = unsafe { GetLastError() };
            if error != ERROR_CLASS_ALREADY_EXISTS {
                return Err(ControlError::backend("raw input window class failed"));
            }
        }
        // SAFETY: the class name outlives the call, `HWND_MESSAGE` asks for a
        // message-only window, and the instance owns the window procedure.
        let window = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                PCWSTR::from_raw(name.as_ptr()),
                PCWSTR::null(),
                WINDOW_STYLE(0),
                0,
                0,
                0,
                0,
                Some(HWND_MESSAGE),
                None,
                Some(instance),
                None,
            )
        }
        .map_err(|_| ControlError::backend("raw input window creation failed"))?;
        Ok(Self(window))
    }

    /// The handle raw input is registered against.
    pub(crate) fn handle(&self) -> HWND {
        self.0
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // SAFETY: the window belongs to this thread and is destroyed once.
        let _ = unsafe { DestroyWindow(self.0) };
    }
}

/// Receives one window message and leaves it to the default handling.
///
/// Raw input is never read here: the buffered read in [`drain`] removes the
/// input messages before they reach the procedure.
///
/// # Safety
///
/// The signature is the one Win32 requires of a registered window procedure.
unsafe extern "system" fn window_proc(window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: forwarding the arguments unchanged is the documented default.
    unsafe { DefWindowProcW(window, message, wparam, lparam) }
}

/// Subscribes the window to one Generic Desktop collection.
pub(crate) fn register(usage: u16, window: HWND) -> Result<(), ControlError> {
    let device = RAWINPUTDEVICE {
        usUsagePage: USAGE_PAGE_GENERIC,
        usUsage: usage,
        dwFlags: RIDEV_INPUTSINK,
        hwndTarget: window,
    };
    // SAFETY: the slice has one well-formed entry and its size is declared.
    unsafe { RegisterRawInputDevices(&[device], size_of::<RAWINPUTDEVICE>() as u32) }
        .map_err(|_| ControlError::backend("raw input registration failed"))
}

/// Removes the subscription of one collection.
///
/// The result is ignored because the only remaining action would be to report
/// an error from a method that cannot return one; a failed removal only leaves
/// records arriving for a class that no longer has a session.
pub(crate) fn unregister(usage: u16) {
    let device = RAWINPUTDEVICE {
        usUsagePage: USAGE_PAGE_GENERIC,
        usUsage: usage,
        dwFlags: RIDEV_REMOVE,
        hwndTarget: HWND::default(),
    };
    // SAFETY: `RIDEV_REMOVE` requires a null target, which is what the
    // default handle is, and the slice declares its own size.
    let _ = unsafe { RegisterRawInputDevices(&[device], size_of::<RAWINPUTDEVICE>() as u32) };
}

/// Reads every queued record and hands it to `emit`.
///
/// The buffer is kept between calls. It grows when Windows reports more data
/// than it holds and is never shrunk.
pub(crate) fn drain(buffer: &mut Vec<u64>, mut emit: impl FnMut(Record)) -> Result<(), ControlError> {
    if buffer.is_empty() {
        buffer.resize(INITIAL_WORDS, 0);
    }
    loop {
        let mut bytes = u32::try_from(buffer.len() * size_of::<u64>()).unwrap_or(u32::MAX);
        // SAFETY: the buffer is aligned for `RAWINPUT` because it is a slice
        // of `u64`, and `bytes` describes exactly its length.
        let count = unsafe {
            GetRawInputBuffer(
                Some(buffer.as_mut_ptr().cast::<RAWINPUT>()),
                &mut bytes,
                size_of::<RAWINPUTHEADER>() as u32,
            )
        };
        if count == u32::MAX {
            if buffer.len() >= MAX_WORDS {
                return Err(ControlError::backend("raw input queue exceeds its limit"));
            }
            buffer.resize((buffer.len() * 2).min(MAX_WORDS), 0);
            continue;
        }
        if count == 0 {
            return Ok(());
        }
        read_blocks(buffer, count, &mut emit);
    }
}

/// Walks the records the system wrote into `buffer`.
fn read_blocks(buffer: &[u64], count: u32, emit: &mut impl FnMut(Record)) {
    let capacity = size_of_val(buffer);
    let base = buffer.as_ptr().cast::<u8>();
    let header = size_of::<RAWINPUTHEADER>();
    let mut offset = 0usize;
    for _ in 0..count {
        if offset.saturating_add(header) > capacity {
            return;
        }
        // SAFETY: `offset` stays within `capacity`. The buffer is a slice of
        // `u64`, so the cast starts eight-byte aligned; reading by copy keeps
        // the access correct for a record whose alignment the system defines
        // instead of Rust's type.
        let raw = unsafe {
            base.add(offset)
                .cast::<RAWINPUT>()
                .read_unaligned()
        };
        emit_block(raw, emit);
        let size = raw.header.dwSize as usize;
        if size < header {
            return;
        }
        let Some(next) = offset.checked_add(align(size, size_of::<usize>())) else {
            return;
        };
        offset = next;
    }
}

/// Turns one raw block into a record when its type is one the backend serves.
fn emit_block(raw: RAWINPUT, emit: &mut impl FnMut(Record)) {
    let device = identifier(raw.header.hDevice);
    if raw.header.dwType == RIM_TYPEKEYBOARD.0 {
        // SAFETY: `dwType` reported the keyboard variant, so the union holds
        // a `RAWKEYBOARD` that was initialized by the system.
        let keyboard = unsafe { raw.data.keyboard };
        emit(Record::Key {
            device,
            key: KeyRecord {
                vk: keyboard.VKey,
                make_code: keyboard.MakeCode,
                flags: keyboard.Flags,
            },
        });
    } else if raw.header.dwType == RIM_TYPEMOUSE.0 {
        // SAFETY: `dwType` reported the mouse variant, so the union holds a
        // `RAWMOUSE` that was initialized by the system.
        let mouse = unsafe { raw.data.mouse };
        // SAFETY: the button structure is the first member of the mouse union
        // and is the form the system fills in.
        let buttons = unsafe { mouse.Anonymous.Anonymous };
        emit(Record::Mouse {
            device,
            mouse: MouseRecord {
                flags: mouse.usFlags.0,
                button_flags: buttons.usButtonFlags,
                button_data: buttons.usButtonData,
                x: mouse.lLastX,
                y: mouse.lLastY,
            },
        });
    }
}

/// The portable identifier of a device handle.
fn identifier(handle: HANDLE) -> DeviceId {
    DeviceId::new(handle.0 as usize as u64)
}

/// Rounds `size` up to a multiple of `alignment`.
fn align(size: usize, alignment: usize) -> usize {
    size.div_ceil(alignment) * alignment
}
