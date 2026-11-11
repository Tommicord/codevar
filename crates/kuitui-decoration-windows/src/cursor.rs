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

//! Mouse cursors.
//!
//! Maps a decoration's requested cursor to a Win32 cursor and installs it
//! with `SetCursor`, loading the standard cursor with `LoadCursorW` or a
//! system cursor with `SetSystemCursor`.

use codevar_kuitui_decoration::DecorationIcon;
use codevar_logger::log_warn;

/// A cursor handle owned by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cursor(pub isize);

impl Cursor {
    /// Creates a cursor from a Win32 `HCURSOR`.
    #[must_use]
    pub const fn from_handle(handle: isize) -> Self {
        Self(handle)
    }

    /// The raw `HCURSOR` handle.
    #[must_use]
    pub const fn handle(self) -> isize {
        self.0
    }

    /// Whether the cursor is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The standard Win32 cursors the backend can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StockCursor {
    /// `IDC_ARROW`: the default arrow cursor.
    Arrow,
    /// `IDC_IBEAM`: the text insertion cursor.
    IBeam,
    /// `IDC_WAIT`: the hourglass / wait cursor.
    Wait,
    /// `IDC_CROSS`: the crosshair cursor.
    Cross,
    /// `IDC_HAND`: the hand cursor.
    Hand,
    /// `IDC_NO`: the "not allowed" cursor.
    No,
    /// `IDC_SIZEALL`: the four-way resize cursor.
    SizeAll,
    /// `IDC_SIZENS`: the vertical resize cursor.
    SizeNS,
    /// `IDC_SIZEWE`: the horizontal resize cursor.
    SizeWE,
    /// `IDC_SIZENWSE`: the top-left/bottom-right resize cursor.
    SizeNWSE,
    /// `IDC_SIZENESW`: the top-right/bottom-left resize cursor.
    SizeNESW,
}

impl StockCursor {
    /// The `IDC_*` resource name of the cursor.
    #[cfg(windows)]
    #[must_use]
    pub const fn name(self) -> windows::Win32::Foundation::PCWSTR {
        use windows::Win32::UI::WindowsAndMessaging::*;
        match self {
            StockCursor::Arrow => IDC_ARROW,
            StockCursor::IBeam => IDC_IBEAM,
            StockCursor::Wait => IDC_WAIT,
            StockCursor::Cross => IDC_CROSS,
            StockCursor::Hand => IDC_HAND,
            StockCursor::No => IDC_NO,
            StockCursor::SizeAll => IDC_SIZEALL,
            StockCursor::SizeNS => IDC_SIZENS,
            StockCursor::SizeWE => IDC_SIZEWE,
            StockCursor::SizeNWSE => IDC_SIZENWSE,
            StockCursor::SizeNESW => IDC_SIZENESW,
        }
    }
}

/// Loads a stock cursor by its `IDC_*` name.
#[cfg(windows)]
pub fn load_stock_cursor(cursor: StockCursor) -> Option<Cursor> {
    use windows::Win32::Foundation::HINSTANCE;
    use windows::Win32::UI::WindowsAndMessaging::{HCURSOR, LoadCursorW};

    unsafe {
        let handle: HCURSOR = LoadCursorW(HINSTANCE::default(), cursor.name());
        if handle.is_invalid() {
            log_warn!("LoadCursorW failed for {:?}", cursor);
            None
        } else {
            Some(Cursor(handle.0 as isize))
        }
    }
}

#[cfg(not(windows))]
pub fn load_stock_cursor(_cursor: StockCursor) -> Option<Cursor> {
    None
}

/// Installs `cursor` as the cursor of the calling thread.
#[cfg(windows)]
pub fn set_cursor(cursor: Cursor) -> Option<Cursor> {
    use windows::Win32::UI::WindowsAndMessaging::SetCursor;

    if cursor.is_empty() {
        return None;
    }
    unsafe {
        let previous = SetCursor(windows::Win32::UI::WindowsAndMessaging::HCURSOR(
            cursor.handle() as *mut _
        ));
        if previous.is_invalid() {
            None
        } else {
            Some(Cursor(previous.0 as isize))
        }
    }
}

#[cfg(not(windows))]
pub fn set_cursor(_cursor: Cursor) -> Option<Cursor> {
    None
}

/// Creates a cursor from premultiplied RGBA8 pixels.
#[cfg(windows)]
pub fn create_cursor_from_icon(icon: &DecorationIcon) -> Option<Cursor> {
    super::icon::create_icon(icon).map(|icon| Cursor(icon.handle()))
}

#[cfg(not(windows))]
pub fn create_cursor_from_icon(_icon: &DecorationIcon) -> Option<Cursor> {
    None
}

/// Frees a cursor handle.
#[cfg(windows)]
pub fn destroy_cursor(cursor: Cursor) {
    if cursor.is_empty() {
        return;
    }
    use windows::Win32::UI::WindowsAndMessaging::{DestroyCursor, HCURSOR};
    unsafe {
        let _ = DestroyCursor(HCURSOR(cursor.handle() as *mut _));
    }
}

#[cfg(not(windows))]
pub fn destroy_cursor(_cursor: Cursor) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cursor_is_none() {
        let cursor = Cursor::from_handle(0);
        assert!(cursor.is_empty());
    }

    #[test]
    fn cursor_handle_wraps_pointer() {
        let cursor = Cursor::from_handle(0x4321);
        assert_eq!(cursor.handle(), 0x4321);
        assert!(!cursor.is_empty());
    }
}
