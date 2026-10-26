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

//! Terminal capability detection and information

use alloc::string::String;
use alloc::string::ToString;
use core::sync::atomic::{AtomicU16, Ordering};

/// Terminal capability flags using bitwise operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct TerminalCaps(u16);

impl TerminalCaps {
    pub const NONE: Self = Self(0);
    pub const ANSI: Self = Self(1 << 0);
    pub const TRUECOLOR: Self = Self(1 << 1);
    pub const COLORS_256: Self = Self(1 << 2);
    pub const MOUSE: Self = Self(1 << 3);
    pub const BRACKETED_PASTE: Self = Self(1 << 4);
    pub const UNICODE: Self = Self(1 << 5);
    pub const ALT_SCREEN: Self = Self(1 << 6);
    pub const FOCUS_REPORTING: Self = Self(1 << 7);

    #[inline]
    pub const fn empty() -> Self {
        Self::NONE
    }

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    #[inline]
    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[inline]
    pub const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    #[inline]
    pub const fn toggle(&mut self, other: Self) {
        self.0 ^= other.0;
    }
}

impl core::ops::BitOr for TerminalCaps {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl core::ops::BitOrAssign for TerminalCaps {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        self.insert(other);
    }
}

impl core::ops::BitAnd for TerminalCaps {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.intersection(other)
    }
}

impl core::ops::BitAndAssign for TerminalCaps {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

impl core::ops::BitXor for TerminalCaps {
    type Output = Self;
    #[inline]
    fn bitxor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}

impl core::ops::BitXorAssign for TerminalCaps {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0;
    }
}

impl core::ops::Not for TerminalCaps {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

/// Terminal information and capabilities
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalInfo {
    pub width: u16,
    pub height: u16,
    pub caps: TerminalCaps,
    pub term_program: Option<String>,
    pub term: Option<String>,
    pub colorterm: Option<String>,
}

impl Default for TerminalInfo {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
            caps: TerminalCaps::NONE,
            term_program: None,
            term: None,
            colorterm: None,
        }
    }
}

impl TerminalInfo {
    #[inline]
    pub fn supports_ansi(self) -> bool {
        self.caps.contains(TerminalCaps::ANSI)
    }

    #[inline]
    pub fn supports_truecolor(self) -> bool {
        self.caps.contains(TerminalCaps::TRUECOLOR)
    }

    #[inline]
    pub fn supports_256_colors(self) -> bool {
        self.caps.contains(TerminalCaps::COLORS_256)
    }

    #[inline]
    pub fn supports_mouse(self) -> bool {
        self.caps.contains(TerminalCaps::MOUSE)
    }

    #[inline]
    pub fn supports_bracketed_paste(self) -> bool {
        self.caps.contains(TerminalCaps::BRACKETED_PASTE)
    }

    #[inline]
    pub fn supports_unicode(self) -> bool {
        self.caps.contains(TerminalCaps::UNICODE)
    }

    #[inline]
    pub fn supports_alt_screen(self) -> bool {
        self.caps.contains(TerminalCaps::ALT_SCREEN)
    }

    #[inline]
    pub fn supports_focus_reporting(self) -> bool {
        self.caps.contains(TerminalCaps::FOCUS_REPORTING)
    }
}

/// Terminal capabilities and utilities
pub struct Terminal;

impl Terminal {
    /// Check if the terminal supports ANSI escape sequences
    pub fn supports_ansi() -> bool {
        // Check if forced
        if super::is_ansi_forced() {
            return true;
        }
        if super::is_ansi_disabled() {
            return false;
        }
        if let Some(colorterm) = codevar_env::env_var("COLORTERM") {
            let ct_lower = colorterm.to_lowercase();
            if ct_lower == "truecolor" || ct_lower == "24bit" {
                return true;
            }
        }
        #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
        {
            if Self::windows_vt_supported() {
                return true;
            }
        }
        false
    }

    /// Check if terminal supports truecolor (24-bit)
    pub fn supports_truecolor() -> bool {
        if !Self::supports_ansi() {
            return false;
        }
        if let Some(colorterm) = codevar_env::env_var("COLORTERM") {
            let ct_lower = colorterm.to_lowercase();
            if ct_lower == "truecolor" || ct_lower == "24bit" {
                return true;
            }
        }
        false
    }

    /// Check if terminal supports mouse reporting
    pub fn supports_mouse() -> bool {
        Self::supports_ansi()
    }

    /// Check if terminal supports bracketed paste
    pub fn supports_bracketed_paste() -> bool {
        Self::supports_ansi()
    }

    /// Check if terminal supports Unicode
    pub fn supports_unicode() -> bool {
        if codevar_env::env_var("LANG").is_some_and(|lang| lang.to_lowercase().contains("utf")) {
            return true;
        }
        if codevar_env::env_var("LC_ALL").is_some_and(|lc_all| lc_all.to_lowercase().contains("utf")) {
            return true;
        }
        if codevar_env::env_var("LC_CTYPE").is_some_and(|lc_ctype| lc_ctype.to_lowercase().contains("utf")) {
            return true;
        }

        // Most modern terminals support Unicode
        Self::supports_ansi()
    }

    /// Get terminal width
    pub fn width() -> u16 {
        TERMINAL_WIDTH.load(Ordering::Relaxed)
    }

    /// Get terminal height
    pub fn height() -> u16 {
        TERMINAL_HEIGHT.load(Ordering::Relaxed)
    }

    /// Update terminal size
    pub fn update_size(width: u16, height: u16) {
        TERMINAL_WIDTH.store(width, Ordering::Relaxed);
        TERMINAL_HEIGHT.store(height, Ordering::Relaxed);
    }

    pub fn detect_size() -> (u16, u16) {
        if let (Some(w), Some(h)) = (codevar_env::env_var("COLUMNS"), codevar_env::env_var("LINES"))
            && let (Ok(w), Ok(h)) = (w.parse::<u16>(), h.parse::<u16>())
        {
            Self::update_size(w, h);
            return (w, h);
        }
        #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
        {
            use windows::Win32::System::Console::{
                GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE,
            };

            let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
            if handle.0 != 0 {
                let mut info = windows::Win32::System::Console::CONSOLE_SCREEN_BUFFER_INFO::default();
                if unsafe { GetConsoleScreenBufferInfo(handle, &mut info) }.as_bool() {
                    let width = (info.srWindow.Right - info.srWindow.Left + 1) as u16;
                    let height = (info.srWindow.Bottom - info.srWindow.Top + 1) as u16;
                    Self::update_size(width, height);
                    return (width, height);
                }
            }
        }
        #[cfg(all(
            any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
            not(target_arch = "wasm32")
        ))]
        {
            use libc::{STDOUT_FILENO, TIOCGWINSZ, ioctl, winsize};
            let mut ws: winsize = unsafe { core::mem::zeroed() };
            if unsafe { ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut ws) } >= 0 && ws.ws_col > 0 && ws.ws_row > 0 {
                let width = ws.ws_col;
                let height = ws.ws_row;
                Self::update_size(width, height);
                return (width, height);
            }
        }
        (80, 24)
    }

    /// Get comprehensive terminal info
    pub fn info() -> TerminalInfo {
        let _ = Self::detect_size();
        let mut caps = TerminalCaps::NONE;
        if Self::supports_ansi() {
            caps.insert(TerminalCaps::ANSI);
        }
        if Self::supports_truecolor() {
            caps.insert(TerminalCaps::TRUECOLOR);
            caps.insert(TerminalCaps::COLORS_256);
        }
        if Self::supports_mouse() {
            caps.insert(TerminalCaps::MOUSE);
        }
        if Self::supports_bracketed_paste() {
            caps.insert(TerminalCaps::BRACKETED_PASTE);
        }
        if Self::supports_unicode() {
            caps.insert(TerminalCaps::UNICODE);
        }
        caps.insert(TerminalCaps::ALT_SCREEN);
        caps.insert(TerminalCaps::FOCUS_REPORTING);

        TerminalInfo {
            width: Self::width(),
            height: Self::height(),
            caps,
            term_program: codevar_env::env_var("TERM_PROGRAM"),
            term: codevar_env::env_var("TERM"),
            colorterm: codevar_env::env_var("COLORTERM"),
        }
    }

    /// Initialize terminal (enable ANSI on Windows, detect size)
    pub fn init() {
        super::init_ansi_support();
        {
            let _ = Self::detect_size();
        }
    }

    /// Reset terminal to default state
    pub fn reset() -> String {
        alloc::format!(
            "{}{}{}{}",
            super::ansi::sequences::RESET,
            super::cursor::Cursor::show(),
            super::clear::Clear::screen_and_home(),
            super::cursor::Cursor::home()
        )
    }

    /// Enable alternate screen buffer
    pub fn alternate_screen_on() -> String {
        "\x1b[?1049h".to_string()
    }

    /// Disable alternate screen buffer
    pub fn alternate_screen_off() -> String {
        "\x1b[?1049l".to_string()
    }

    /// Enable bracketed paste mode
    pub fn bracketed_paste_on() -> String {
        "\x1b[?2004h".to_string()
    }

    /// Disable bracketed paste mode
    pub fn bracketed_paste_off() -> String {
        "\x1b[?2004l".to_string()
    }

    /// Enable mouse reporting (SGR mode)
    pub fn mouse_on() -> String {
        "\x1b[?1000h\x1b[?1006h".to_string()
    }

    /// Disable mouse reporting
    pub fn mouse_off() -> String {
        "\x1b[?1000l\x1b[?1006l".to_string()
    }

    /// Enable focus reporting
    pub fn focus_reporting_on() -> String {
        "\x1b[?1004h".to_string()
    }

    /// Disable focus reporting
    pub fn focus_reporting_off() -> String {
        "\x1b[?1004l".to_string()
    }

    /// Check Windows VT support
    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    fn windows_vt_supported() -> bool {
        use windows::Win32::System::Console::{GetConsoleMode, GetStdHandle, STD_OUTPUT_HANDLE};
        let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        if handle.0 == 0 || handle.0 == -1isize as _ {
            return false;
        }
        let mut mode = 0u32;
        unsafe { GetConsoleMode(handle, &mut mode) }.is_ok()
    }
}

/// Global terminal width
static TERMINAL_WIDTH: AtomicU16 = AtomicU16::new(80);

/// Global terminal height
static TERMINAL_HEIGHT: AtomicU16 = AtomicU16::new(24);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_info_default() {
        let info = TerminalInfo::default();
        assert_eq!(info.width, 80);
        assert_eq!(info.height, 24);
        assert!(!info.supports_ansi());
    }

    #[test]
    fn test_terminal_caps_bitflags() {
        let caps = TerminalCaps::ANSI | TerminalCaps::TRUECOLOR | TerminalCaps::MOUSE;
        assert!(caps.contains(TerminalCaps::ANSI));
        assert!(caps.contains(TerminalCaps::TRUECOLOR));
        assert!(caps.contains(TerminalCaps::MOUSE));
        assert!(!caps.contains(TerminalCaps::BRACKETED_PASTE));

        let mut caps2 = TerminalCaps::ANSI;
        caps2 |= TerminalCaps::TRUECOLOR;
        assert!(caps2.contains(TerminalCaps::ANSI));
        assert!(caps2.contains(TerminalCaps::TRUECOLOR));

        caps2 &= TerminalCaps::ANSI;
        assert!(caps2.contains(TerminalCaps::ANSI));
        assert!(!caps2.contains(TerminalCaps::TRUECOLOR));
    }

    #[test]
    fn test_terminal_reset() {
        let reset = Terminal::reset();
        assert!(reset.contains("\x1b[0m"));
        assert!(reset.contains("\x1b[?25h"));
        assert!(reset.contains("\x1b[2J"));
        assert!(reset.contains("\x1b[H"));
    }
}
