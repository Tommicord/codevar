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

//! Cursor control utilities

use super::ansi::{csi0, csi1, csi2};
use alloc::string::String;
use alloc::string::ToString;
use core::fmt;

/// Cursor position (1-indexed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub row: u16,
    pub col: u16,
}

impl Position {
    /// Create a new position
    pub const fn new(row: u16, col: u16) -> Self {
        Self { row, col }
    }

    /// Origin position (1, 1)
    pub const fn origin() -> Self {
        Self { row: 1, col: 1 }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1b[{};{}H", self.row, self.col)
    }
}

/// Cursor movement and control
pub struct Cursor;

impl Cursor {
    /// Move cursor up N lines
    pub fn up(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[A".to_string();
        }
        csi1(n, 'A')
    }

    /// Move cursor down N lines
    pub fn down(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[B".to_string();
        }
        csi1(n, 'B')
    }

    /// Move cursor forward (right) N columns
    pub fn forward(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[C".to_string();
        }
        csi1(n, 'C')
    }

    /// Move cursor backward (left) N columns
    pub fn backward(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[D".to_string();
        }
        csi1(n, 'D')
    }

    /// Move cursor to beginning of next line N lines down
    pub fn next_line(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[E".to_string();
        }
        csi1(n, 'E')
    }

    /// Move cursor to beginning of previous line N lines up
    pub fn prev_line(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[F".to_string();
        }
        csi1(n, 'F')
    }

    /// Move cursor to column N (1-indexed)
    pub fn column(n: u16) -> String {
        if n <= 1 {
            return "\x1b[G".to_string();
        }
        csi1(n, 'G')
    }

    /// Move cursor to row, column (1-indexed)
    pub fn position(row: u16, col: u16) -> String {
        if row <= 1 && col <= 1 {
            return "\x1b[H".to_string();
        }
        csi2(row, col, 'H')
    }

    /// Move cursor to row, column (alternative form)
    pub fn cup(row: u16, col: u16) -> String {
        if row <= 1 && col <= 1 {
            return "\x1b[f".to_string();
        }
        csi2(row, col, 'f')
    }

    /// Move cursor to Position
    pub fn goto(pos: Position) -> String {
        Self::position(pos.row, pos.col)
    }

    /// Move cursor home (1, 1)
    pub fn home() -> String {
        "\x1b[H".to_string()
    }

    /// Save cursor position (DECSC)
    pub fn save() -> String {
        csi0('s')
    }

    /// Restore cursor position (DECRC)
    pub fn restore() -> String {
        csi0('u')
    }

    /// Hide cursor (DECTCEM)
    pub fn hide() -> String {
        "\x1b[?25l".to_string()
    }

    /// Show cursor (DECTCEM)
    pub fn show() -> String {
        "\x1b[?25h".to_string()
    }

    /// Get cursor position (DSR - Device Status Report)
    /// Returns the sequence to request cursor position report
    pub fn get_position() -> String {
        csi1(6, 'n')
    }

    /// Enable application cursor keys mode (DECCKM)
    pub fn enable_keypad() -> String {
        "\x1b[?1h".to_string()
    }

    /// Disable application cursor keys mode
    pub fn disable_keypad() -> String {
        "\x1b[?1l".to_string()
    }

    /// Move cursor to start of line (column 1)
    pub fn start_of_line() -> String {
        "\r".to_string()
    }

    /// Move cursor to end of line (requires knowing terminal width)
    pub fn end_of_line(width: u16) -> String {
        Self::column(width)
    }

    /// Move cursor to line N (relative to current position)
    pub fn line(line: u16) -> String {
        if line == 0 {
            return String::new();
        }
        csi1(line, 'd')
    }

    /// Move cursor up N lines and to column 1
    pub fn up_and_home(n: u16) -> String {
        if n == 0 {
            return "\r".to_string();
        }
        alloc::format!("\x1b[{}A\r", n)
    }
}

/// Cursor visibility state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Visible,
    Hidden,
    Default,
}

impl Visibility {
    /// Get the ANSI sequence for this visibility
    pub fn sequence(self) -> String {
        match self {
            Visibility::Visible => Cursor::show(),
            Visibility::Hidden => Cursor::hide(),
            Visibility::Default => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_movement() {
        assert_eq!(Cursor::up(1), "\x1b[A");
        assert_eq!(Cursor::up(5), "\x1b[5A");
        assert_eq!(Cursor::down(1), "\x1b[B");
        assert_eq!(Cursor::forward(1), "\x1b[C");
        assert_eq!(Cursor::backward(1), "\x1b[D");
    }

    #[test]
    fn test_cursor_position() {
        assert_eq!(Cursor::position(1, 1), "\x1b[H");
        assert_eq!(Cursor::position(10, 20), "\x1b[10;20H");
        assert_eq!(Cursor::column(5), "\x1b[5G");
    }

    #[test]
    fn test_cursor_save_restore() {
        assert_eq!(Cursor::save(), "\x1b[s");
        assert_eq!(Cursor::restore(), "\x1b[u");
    }

    #[test]
    fn test_cursor_visibility() {
        assert_eq!(Cursor::hide(), "\x1b[?25l");
        assert_eq!(Cursor::show(), "\x1b[?25h");
    }

    #[test]
    fn test_position_display() {
        let pos = Position::new(10, 20);
        assert_eq!(pos.to_string(), "\x1b[10;20H");
    }
}
