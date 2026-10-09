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

//! Console utilities and ANSI escape code handling for Codevar
//!
//! This module provides a comprehensive set of utilities for console manipulation,
//! It includes:
//!
//! - ANSI escape sequence constants and builders
//! - Color and style management (foreground, background, bold, italic, underline, etc.)
//! - Cursor control (position, movement, visibility)
//! - Screen/line clearing operations
//! - Terminal capability detection
//! - Cross-platform console handling (Windows, Unix, WASM)
//!
//! # Example
//!
//! ```rust
//! use codevar_consoleutil::{AnsiColor, AnsiStyle, Cursor, Clear, Terminal};
//!
//! // Print colored text
//! println!("{}Red text{}", AnsiColor::Red.fg(), AnsiColor::Default.fg());
//!
//! // Move cursor and clear line
//! print!("{}{}", Cursor::up(1), Clear::current_line());
//!
//! // Check terminal capabilities
//! if Terminal::supports_ansi() {
//!     println!("ANSI supported!");
//! }
//! ```

#![cfg_attr(not(test), no_std)]
extern crate alloc;

use core::fmt;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
use windows::Win32::System::Console::{
    ENABLE_VIRTUAL_TERMINAL_PROCESSING, GetConsoleMode, GetStdHandle, STD_OUTPUT_HANDLE, SetConsoleMode,
};
pub mod ansi;
pub mod clear;
pub mod cursor;
pub mod style;
pub mod terminal;

pub use ansi::{AnsiBuilder, AnsiCode, AnsiSequence};
pub use clear::Clear;
pub use cursor::Cursor;
pub use style::{AnsiColor, AnsiStyle, Style, StyleAttr, StyledText};
pub use terminal::{Terminal, TerminalCaps, TerminalInfo};

/// Initialize ANSI support on Windows (enables virtual terminal processing)
/// This is a no-op on non-Windows platforms.
///
/// # Example
/// ```rust
/// use codevar_consoleutil::init_ansi_support;
/// init_ansi_support();
/// ```
pub fn init_ansi_support() {
    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        if handle.0 != 0 {
            let mut mode = 0u32;
            if GetConsoleMode(handle, &mut mode).is_ok() {
                let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING.0);
            }
        }
    }
}

/// Check if the current terminal supports ANSI escape sequences
///
/// # Example
/// ```rust
/// use codevar_consoleutil::supports_ansi;
/// if supports_ansi() {
///     println!("Terminal supports ANSI colors");
/// }
/// ```
pub fn supports_ansi() -> bool {
    Terminal::supports_ansi()
}

/// Strip ANSI escape sequences from a string
///
/// # Example
/// ```rust
/// use codevar_consoleutil::strip_ansi;
/// let clean = strip_ansi("\x1b[31mRed\x1b[0m text");
/// assert_eq!(clean, "Red text");
/// ```
pub fn strip_ansi(input: &str) -> alloc::string::String {
    let mut result = alloc::string::String::with_capacity(input.len());
    let mut in_escape = false;
    let mut in_csi = false;

    for ch in input.chars() {
        if !in_escape {
            if ch == '\x1b' {
                in_escape = true;
            } else {
                result.push(ch);
            }
        } else if !in_csi {
            if ch == '[' {
                in_csi = true;
            } else {
                in_escape = false;
                in_csi = false;
            }
        } else if ch.is_ascii_alphabetic() || ch == '~' {
            in_escape = false;
            in_csi = false;
        }
    }
    result
}

/// Convert UTF-8 string to UTF-16 for Windows console APIs
///
/// Uses codevar-textlike-encode for proper encoding handling.
#[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
pub fn utf8_to_utf16(input: &str) -> alloc::vec::Vec<u16> {
    let mut encoder = Encoder::new(&UTF_8, VariantEncoder::Utf8(Utf8Encoder));
    let mut dst = alloc::vec::Vec::with_capacity(input.len() * 2);
    dst.resize(input.len() * 2, 0);

    let (result, read, written) = encoder.encode_from_utf8_raw(input, &mut dst, true);
    dst.truncate(written);
    debug_assert_eq!(result, codevar_textlike_encode::encoding::CoderResult::InputEmpty);
    debug_assert_eq!(read, input.len());
    dst
}

/// Write bytes to stdout with automatic encoding handling
///
/// On Windows, converts to UTF-16 and uses WriteConsoleW.
/// On Unix/WASM, writes directly to stdout.
pub fn write_stdout(bytes: &[u8]) -> Result<(), ConsoleError> {
    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    {
        use core::ffi::c_void;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Console::{GetStdHandle, STD_OUTPUT_HANDLE, WriteConsoleW};
        use windows::core::PCWSTR;

        let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        if handle.0 == 0 || handle.0 == -1isize as _ {
            return Err(ConsoleError::HandleUnavailable);
        }
        let utf16 = {
            let mut encoder = Encoder::new(&UTF_8, VariantEncoder::Utf16(Utf16Encoder));
            let mut dst = alloc::vec::Vec::with_capacity(input.len() * 2);
            dst.resize(input.len() * 2, 0);

            let (result, read, written) = encoder.encode_from_utf8_raw(input, &mut dst, true);
            dst.truncate(written);
            debug_assert_eq!(result, codevar_textlike_encode::encoding::CoderResult::InputEmpty);
            debug_assert_eq!(read, input.len());
            dst
        };
        let mut written: u32 = 0;
        let result = unsafe {
            WriteConsoleW(
                handle,
                utf16.as_ptr() as *const c_void,
                utf16.len() as u32,
                &mut written,
                core::ptr::null_mut(),
            )
        };
        if !result.as_bool() {
            return Err(ConsoleError::SyscallError(
                unsafe { windows::Win32::Foundation::GetLastError().0 } as i32,
            ));
        }
        Ok(())
    }
    #[cfg(not(all(target_os = "windows", not(target_arch = "wasm32"))))]
    {
        use codevar_io::{Stdout, Write};
        let mut stdout = Stdout::new();
        stdout
            .write_all(bytes)
            .map_err(|e| ConsoleError::IoError(e as i32))?;
        stdout
            .flush()
            .map_err(|e| ConsoleError::IoError(e as i32))?;
        Ok(())
    }
}

/// Write bytes to stderr with automatic encoding handling
pub fn write_stderr(bytes: &[u8]) -> Result<(), ConsoleError> {
    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    {
        use core::ffi::c_void;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, WriteConsoleW};
        use windows::core::PCWSTR;

        let handle = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
        if handle.0 == 0 || handle.0 == -1isize as _ {
            return Err(ConsoleError::HandleUnavailable);
        }
        let utf16 = utf8_to_utf16(unsafe { core::str::from_utf8_unchecked(bytes) });
        let mut written: u32 = 0;
        let result = unsafe {
            WriteConsoleW(
                handle,
                utf16.as_ptr() as *const c_void,
                utf16.len() as u32,
                &mut written,
                core::ptr::null_mut(),
            )
        };
        if !result.as_bool() {
            return Err(ConsoleError::SyscallError(
                unsafe { windows::Win32::Foundation::GetLastError().0 } as i32,
            ));
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "windows", not(target_arch = "wasm32"))))]
    {
        use codevar_io::{Stderr, Write};
        let mut stderr = Stderr::new();
        stderr
            .write_all(bytes)
            .map_err(|e| ConsoleError::IoError(e as i32))?;
        stderr
            .flush()
            .map_err(|e| ConsoleError::IoError(e as i32))?;
        Ok(())
    }
}

/// Error type for console operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleError {
    /// System call failed
    SyscallError(i32),
    /// Io error when writing to the console
    IoError(i32),
    /// Encoding error
    EncodingError,
    /// Console handle not available
    HandleUnavailable,
    /// Buffer too small
    BufferTooSmall,
}

impl fmt::Display for ConsoleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConsoleError::SyscallError(code) => write!(f, "syscall error: {}", code),
            ConsoleError::EncodingError => write!(f, "encoding error"),
            ConsoleError::HandleUnavailable => write!(f, "console handle unavailable"),
            ConsoleError::BufferTooSmall => write!(f, "buffer too small"),
            &ConsoleError::IoError(_) => write!(f, "I/O error"),
        }
    }
}
impl core::error::Error for ConsoleError {}

/// Global flag to force ANSI output even when not detected
static FORCE_ANSI: AtomicBool = AtomicBool::new(false);

/// Global flag to disable ANSI output
static DISABLE_ANSI: AtomicBool = AtomicBool::new(false);

/// Force enable ANSI output (useful for piping to less -R, etc.)
pub fn force_ansi(enable: bool) {
    FORCE_ANSI.store(enable, Ordering::Relaxed);
}

/// Disable ANSI output globally
pub fn disable_ansi(enable: bool) {
    DISABLE_ANSI.store(enable, Ordering::Relaxed);
}

/// Check if ANSI is forced enabled
pub fn is_ansi_forced() -> bool {
    FORCE_ANSI.load(Ordering::Relaxed)
}

/// Check if ANSI is disabled
pub fn is_ansi_disabled() -> bool {
    DISABLE_ANSI.load(Ordering::Relaxed)
}

/// Reset ANSI state to auto-detection
pub fn reset_ansi_state() {
    FORCE_ANSI.store(false, Ordering::Relaxed);
    DISABLE_ANSI.store(false, Ordering::Relaxed);
}

/// Terminal width detection
static TERMINAL_WIDTH: AtomicU8 = AtomicU8::new(80);

/// Get terminal width (cached, updated on first call or via update_terminal_size)
pub fn terminal_width() -> u16 {
    TERMINAL_WIDTH.load(Ordering::Relaxed) as u16
}

/// Update terminal width (call on SIGWINCH or resize events)
pub fn update_terminal_width(width: u16) {
    TERMINAL_WIDTH.store(width.min(255) as u8, Ordering::Relaxed);
}

/// Detect terminal width from environment or syscalls
pub fn detect_terminal_width() -> u16 {
    if let Some(columns) = codevar_env::env_var("COLUMNS")
        && let Ok(w) = columns.parse::<u16>()
    {
        update_terminal_width(w);
        return w;
    }
    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    {
        use windows::Win32::Foundation::COORD;
        use windows::Win32::System::Console::{GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE};

        let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        if handle.0 != 0 {
            let mut info = windows::Win32::System::Console::CONSOLE_SCREEN_BUFFER_INFO::default();
            if unsafe { GetConsoleScreenBufferInfo(handle, &mut info) }.as_bool() {
                let width = (info.srWindow.Right - info.srWindow.Left + 1) as u16;
                update_terminal_width(width);
                return width;
            }
        }
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        not(target_arch = "wasm32")
    ))]
    {
        use libc::{STDOUT_FILENO, TIOCGWINSZ, ioctl, winsize};
        let mut win_size: winsize = unsafe { core::mem::zeroed() };
        if unsafe { ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut win_size) } >= 0 && win_size.ws_col > 0 {
            update_terminal_width(win_size.ws_col);
            return win_size.ws_col;
        }
    }
    80
}

/// Terminal height detection
static TERMINAL_HEIGHT: AtomicU8 = AtomicU8::new(24);

/// Get terminal height (cached)
pub fn terminal_height() -> u16 {
    TERMINAL_HEIGHT.load(Ordering::Relaxed) as u16
}

/// Update terminal height
pub fn update_terminal_height(height: u16) {
    TERMINAL_HEIGHT.store(height.min(255) as u8, Ordering::Relaxed);
}

/// Detect terminal height
pub fn detect_terminal_height() -> u16 {
    if let Some(lines) = codevar_env::env_var("LINES")
        && let Ok(h) = lines.parse::<u16>()
    {
        update_terminal_height(h);
        return h;
    }
    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    {
        use windows::Win32::System::Console::{GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE};

        let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        if handle.0 != 0 {
            let mut info = windows::Win32::System::Console::CONSOLE_SCREEN_BUFFER_INFO::default();
            if unsafe { GetConsoleScreenBufferInfo(handle, &mut info) }.as_bool() {
                let height = (info.srWindow.Bottom - info.srWindow.Top + 1) as u16;
                update_terminal_height(height);
                return height;
            }
        }
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        not(target_arch = "wasm32")
    ))]
    {
        use libc::{STDOUT_FILENO, TIOCGWINSZ, ioctl, winsize};
        let mut win_size: winsize = unsafe { core::mem::zeroed() };
        if unsafe { ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut win_size) } >= 0 && win_size.ws_row > 0 {
            update_terminal_height(win_size.ws_row);
            return win_size.ws_row;
        }
    }
    24 // Default fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("\x1b[31mRed\x1b[0m"), "Red");
        assert_eq!(strip_ansi("Normal text"), "Normal text");
        assert_eq!(strip_ansi("\x1b[1;32mBold Green\x1b[0m"), "Bold Green");
        assert_eq!(strip_ansi("\x1b[2J\x1b[H"), "");
    }

    #[test]
    fn test_ansi_state() {
        reset_ansi_state();
        assert!(!is_ansi_forced());
        assert!(!is_ansi_disabled());

        force_ansi(true);
        assert!(is_ansi_forced());

        disable_ansi(true);
        assert!(is_ansi_disabled());

        reset_ansi_state();
    }
}
