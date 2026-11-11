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

//! Win32 window management.
//!
//! Registers a window class, creates surfaces with `CreateWindowExW`,
//! destroys them with `DestroyWindow`, and applies titles, bounds and
//! styles to a live window.

use codevar_kuitui_decoration::{DecorationBounds, DecorationError, DecorationModifiers, DecorationResult};

use crate::styles::style_from_modifiers;

/// A Win32 window handle owned by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Window(pub isize);

impl Window {
    /// Creates a window handle from a raw HWND.
    #[must_use]
    pub const fn from_handle(handle: isize) -> Self {
        Self(handle)
    }

    /// The raw HWND handle.
    #[must_use]
    pub const fn handle(self) -> isize {
        self.0
    }

    /// Whether the handle is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The class name of the decoration window class, null-terminated.
#[cfg(windows)]
const CLASS_NAME: &[u16] = &[
    'C' as u16, 'o' as u16, 'd' as u16, 'e' as u16, 'v' as u16, 'a' as u16, 'r' as u16, 'D' as u16,
    'e' as u16, 'c' as u16, 'o' as u16, 'r' as u16, 'a' as u16, 't' as u16, 'i' as u16, 'o' as u16,
    'n' as u16, 0,
];

/// Registers the decoration window class and returns its atom.
#[cfg(windows)]
pub fn register_class() -> Result<u16, DecorationError> {
    use windows::Win32::Foundation::HINSTANCE;
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CS_HREDRAW, CS_VREDRAW, IDC_ARROW, IDI_APPLICATION, LoadCursorW, LoadIconW, RegisterClassExW,
        WNDCLASSEXW,
    };

    let instance = unsafe { GetModuleHandleW(None) }
        .map_err(|e| DecorationError::Os(format!("GetModuleHandleW: {e}")))?;

    let wnd_class = WNDCLASSEXW {
        cbSize: core::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(window_procedure),
        hInstance: instance,
        hIcon: unsafe { LoadIconW(None, IDI_APPLICATION) },
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW) },
        hbrBackground: None,
        lpszMenuName: None,
        lpszClassName: windows::Win32::Foundation::PCWSTR::from_raw(CLASS_NAME.as_ptr()),
        ..Default::default()
    };

    let atom = unsafe { RegisterClassExW(&wnd_class) };
    if atom == 0 {
        return Err(DecorationError::Os("RegisterClassExW failed".into()));
    }
    Ok(atom)
}

#[cfg(not(windows))]
pub fn register_class() -> Result<u16, DecorationError> {
    Err(DecorationError::Unavailable)
}

/// The window procedure for decoration windows.
#[cfg(windows)]
extern "system" fn window_procedure(
    hwnd: windows::Win32::Foundation::HWND,
    message: u32,
    wparam: windows::Win32::Foundation::WPARAM,
    lparam: windows::Win32::Foundation::LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    use windows::Win32::UI::WindowsAndMessaging::{
        DefWindowProcW, PostQuitMessage, WM_ACTIVATE, WM_CLOSE, WM_DESTROY, WM_NCCALCSIZE, WM_SIZE,
    };

    match message {
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            windows::Win32::Foundation::LRESULT::default()
        }
        WM_CLOSE => DefWindowProcW(hwnd, message, wparam, lparam),
        _ => DefWindowProcW(hwnd, message, wparam, lparam),
    }
}

#[cfg(not(windows))]
extern "system" fn window_procedure(_hwnd: isize, _message: u32, _wparam: isize, _lparam: isize) -> isize {
    0
}

/// Creates a top-level window for `title`, `bounds` and `modifiers`.
#[cfg(windows)]
pub fn create_window(
    title: &str,
    bounds: DecorationBounds,
    modifiers: DecorationModifiers,
) -> Result<Window, DecorationError> {
    use windows::Win32::Foundation::HINSTANCE;
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CW_USEDEFAULT, CreateWindowExW, WINDOW_EX_STYLE, WINDOW_STYLE,
    };

    let instance = unsafe { GetModuleHandleW(None) }
        .map_err(|e| DecorationError::Os(format!("GetModuleHandleW: {e}")))?;

    let style: WINDOW_STYLE = style_from_modifiers(modifiers);
    let ex_style: WINDOW_EX_STYLE = Default::default();

    let wide: Vec<u16> = title.encode_utf16().chain(Some(0)).collect();

    let hwnd = unsafe {
        CreateWindowExW(
            ex_style,
            windows::Win32::Foundation::PCWSTR::from_raw(CLASS_NAME.as_ptr()),
            windows::Win32::Foundation::PCWSTR::from_raw(wide.as_ptr()),
            style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            bounds.width as i32,
            bounds.height as i32,
            None,
            None,
            instance,
            None,
        )
    };

    if hwnd.is_invalid() {
        return Err(DecorationError::Os("CreateWindowExW failed".into()));
    }
    Ok(Window(hwnd.0 as isize))
}

#[cfg(not(windows))]
pub fn create_window(
    _title: &str,
    _bounds: DecorationBounds,
    _modifiers: DecorationModifiers,
) -> Result<Window, DecorationError> {
    Err(DecorationError::Unavailable)
}

/// Sets the title of `window`.
#[cfg(windows)]
pub fn set_title(window: Window, title: &str) -> DecorationResult<()> {
    use windows::Win32::UI::WindowsAndMessaging::SetWindowTextW;

    let wide: Vec<u16> = title.encode_utf16().chain(Some(0)).collect();
    unsafe {
        let ok = SetWindowTextW(
            windows::Win32::Foundation::HWND(window.handle() as *mut _),
            windows::Win32::Foundation::PCWSTR::from_raw(wide.as_ptr()),
        );
        if ok.as_bool() {
            Ok(())
        } else {
            Err(DecorationError::Os("SetWindowTextW failed".into()))
        }
    }
}

#[cfg(not(windows))]
pub fn set_title(_window: Window, _title: &str) -> DecorationResult<()> {
    Err(DecorationError::Unavailable)
}

/// Sets the bounds of `window` with `MoveWindow`.
#[cfg(windows)]
pub fn set_bounds(window: Window, bounds: DecorationBounds) -> DecorationResult<()> {
    use windows::Win32::UI::WindowsAndMessaging::MoveWindow;

    if bounds.is_empty() {
        return Err(DecorationError::InvalidBounds(bounds));
    }

    unsafe {
        let ok = MoveWindow(
            windows::Win32::Foundation::HWND(window.handle() as *mut _),
            bounds.x,
            bounds.y,
            bounds.width as i32,
            bounds.height as i32,
            true,
        );
        if ok.as_bool() {
            Ok(())
        } else {
            Err(DecorationError::Os("MoveWindow failed".into()))
        }
    }
}

#[cfg(not(windows))]
pub fn set_bounds(_window: Window, _bounds: DecorationBounds) -> DecorationResult<()> {
    Err(DecorationError::Unavailable)
}

/// Applies `modifiers` to `window` through `SetWindowLongPtrW`/`ShowWindow`.
#[cfg(windows)]
pub fn set_modifiers(window: Window, modifiers: DecorationModifiers) -> DecorationResult<()> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        GWL_STYLE, SW_MAXIMIZE, SW_MINIMIZE, SW_RESTORE, SetWindowLongPtrW, ShowWindow,
    };

    unsafe {
        let hwnd = HWND(window.handle() as *mut _);
        let new_style = style_from_modifiers(modifiers);
        let _ = SetWindowLongPtrW(hwnd, GWL_STYLE, new_style as isize);

        if modifiers.has(DecorationModifier::Minimized) {
            ShowWindow(hwnd, SW_MINIMIZE);
        } else if modifiers.has(DecorationModifier::Maximized) {
            ShowWindow(hwnd, SW_MAXIMIZE);
        } else {
            ShowWindow(hwnd, SW_RESTORE);
        }
    }

    Ok(())
}

#[cfg(not(windows))]
pub fn set_modifiers(_window: Window, _modifiers: DecorationModifiers) -> DecorationResult<()> {
    Err(DecorationError::Unavailable)
}

/// Destroys `window`.
#[cfg(windows)]
pub fn destroy_window(window: Window) -> DecorationResult<()> {
    use windows::Win32::UI::WindowsAndMessaging::DestroyWindow;

    if window.is_empty() {
        return Ok(());
    }

    unsafe {
        let ok = DestroyWindow(windows::Win32::Foundation::HWND(window.handle() as *mut _));
        if ok.as_bool() {
            Ok(())
        } else {
            Err(DecorationError::Os("DestroyWindow failed".into()))
        }
    }
}

#[cfg(not(windows))]
pub fn destroy_window(_window: Window) -> DecorationResult<()> {
    Err(DecorationError::Unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_handle_wraps_pointer() {
        let window = Window::from_handle(0x1000);
        assert_eq!(window.handle(), 0x1000);
        assert!(!window.is_empty());
        assert!(Window::from_handle(0).is_empty());
    }
}
