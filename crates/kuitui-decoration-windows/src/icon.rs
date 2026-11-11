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

//! Window icons.
//!
//! Decodes a [`DecorationIcon`](codevar_kuitui_decoration::DecorationIcon)
//! (premultiplied RGBA8) into an `HICON` with `CreateIconIndirect`, and can
//! load an icon from a module resource.

use codevar_kuitui_decoration::DecorationIcon;
use codevar_logger::log_warn;

#[cfg(windows)]
use windows::Win32::Foundation::HINSTANCE;

/// An icon handle owned by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Icon(pub isize);

impl Icon {
    /// Creates an icon from a Win32 `HICON`.
    #[must_use]
    pub const fn from_handle(handle: isize) -> Self {
        Self(handle)
    }

    /// The raw `HICON` handle.
    #[must_use]
    pub const fn handle(self) -> isize {
        self.0
    }

    /// Whether the icon is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Creates an `HICON` from premultiplied RGBA8 pixels.
///
/// Returns `None` when the icon is empty or when the native call fails.
#[cfg(windows)]
pub fn create_icon(icon: &DecorationIcon) -> Option<Icon> {
    use windows::Win32::Graphics::Gdi::{
        CreateBitmap, CreateIconIndirect, DeleteObject, GetDC, HBITMAP, HDC, ICONINFO, ReleaseDC,
    };

    if icon.is_empty() {
        return None;
    }

    let width = icon.width as i32;
    let height = icon.height as i32;
    if width <= 0 || height <= 0 {
        return None;
    }

    // Build an ICONINFO with a 32bpp premultiplied RGBA bitmap and a mask.
    unsafe {
        let hdc = GetDC(None);
        let hbm_color = CreateBitmap(
            width,
            height,
            1,
            32,
            icon.pixels.as_ptr() as *const core::ffi::c_void,
        );
        let hbm_mask = CreateBitmap(width, height, 1, 1, core::ptr::null());
        if hbm_color.is_invalid() || hbm_mask.is_invalid() {
            if !hbm_color.is_invalid() {
                let _ = DeleteObject(hbm_color);
            }
            if !hbm_mask.is_invalid() {
                let _ = DeleteObject(hbm_mask);
            }
            log_warn!("failed to create icon bitmaps");
            return None;
        }

        let info = ICONINFO {
            fIcon: windows::Win32::Foundation::BOOL::from(true),
            xHotspot: 0,
            yHotspot: 0,
            hbmMask: hbm_mask,
            hbmColor: hbm_color,
        };

        let icon_handle = CreateIconIndirect(&info);
        let _ = DeleteObject(hbm_color);
        let _ = DeleteObject(hbm_mask);
        let _ = ReleaseDC(None, hdc);

        if icon_handle.is_invalid() {
            log_warn!("CreateIconIndirect failed");
            None
        } else {
            Some(Icon(icon_handle.0 as isize))
        }
    }
}

#[cfg(not(windows))]
pub fn create_icon(_icon: &DecorationIcon) -> Option<Icon> {
    None
}

/// Loads an icon from a module resource by name.
#[cfg(windows)]
pub fn load_icon_from_resource(instance: isize, name: &str) -> Option<Icon> {
    use windows::Win32::UI::WindowsAndMessaging::{IMAGE_ICON, LR_DEFAULTSIZE, LoadImageW};
    use windows::core::PCWSTR;

    let wide: Vec<u16> = name
        .encode_utf16()
        .collect::<Vec<_>>()
        .into_iter()
        .chain(Some(0))
        .collect();
    let pcwstr = unsafe { PCWSTR::from_raw(wide.as_ptr()) };
    unsafe {
        let handle = LoadImageW(
            HINSTANCE(instance as *mut _),
            pcwstr,
            IMAGE_ICON,
            0,
            0,
            LR_DEFAULTSIZE,
        );
        if handle.is_invalid() {
            log_warn!("LoadImageW failed for icon resource {name}");
            None
        } else {
            Some(Icon(handle.0 as isize))
        }
    }
}

#[cfg(not(windows))]
pub fn load_icon_from_resource(_instance: isize, _name: &str) -> Option<Icon> {
    None
}

/// Frees an icon handle created by this module.
#[cfg(windows)]
pub fn destroy_icon(icon: Icon) {
    if icon.is_empty() {
        return;
    }
    use windows::Win32::Graphics::Gdi::HICON;
    use windows::Win32::UI::WindowsAndMessaging::DestroyIcon;
    unsafe {
        let _ = DestroyIcon(HICON(icon.handle() as *mut _));
    }
}

#[cfg(not(windows))]
pub fn destroy_icon(_icon: Icon) {}

/// Sets the big and small icons of a window with `WM_SETICON`.
#[cfg(windows)]
pub fn set_window_icon(window: isize, icon: Icon) -> bool {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{SendMessageW, WM_SETICON};

    if icon.is_empty() {
        return false;
    }

    let handle = windows::Win32::Foundation::HICON(icon.handle() as *mut _);
    unsafe {
        let _ = SendMessageW(
            HWND(window as *mut _),
            WM_SETICON,
            0, // ICON_SMALL
            handle as *mut _ as isize,
        );
        let _ = SendMessageW(
            HWND(window as *mut _),
            WM_SETICON,
            1, // ICON_BIG
            handle as *mut _ as isize,
        );
    }
    true
}

#[cfg(not(windows))]
pub fn set_window_icon(_window: isize, _icon: Icon) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_icon_is_none() {
        let icon = DecorationIcon::new();
        assert!(create_icon(&icon).is_none());
    }

    #[test]
    fn icon_handle_wraps_pointer() {
        let icon = Icon::from_handle(0x1234);
        assert_eq!(icon.handle(), 0x1234);
        assert!(!icon.is_empty());
    }
}
