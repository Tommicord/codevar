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

//! Screen and line clearing utilities

use super::ansi::csi1;
use alloc::string::String;
use alloc::string::ToString;

/// Clear operations
pub struct Clear;

impl Clear {
    /// Clear entire screen (ED 2)
    pub fn screen() -> String {
        csi1(2, 'J')
    }

    /// Clear from cursor to end of screen (ED 0)
    pub fn screen_down() -> String {
        csi1(0, 'J')
    }

    /// Clear from cursor to beginning of screen (ED 1)
    pub fn screen_up() -> String {
        csi1(1, 'J')
    }

    /// Clear entire screen and scrollback (ED 3)
    pub fn screen_saved() -> String {
        csi1(3, 'J')
    }

    /// Clear entire line (EL 2)
    pub fn line() -> String {
        csi1(2, 'K')
    }

    /// Clear from cursor to end of line (EL 0)
    pub fn line_end() -> String {
        csi1(0, 'K')
    }

    /// Clear from cursor to beginning of line (EL 1)
    pub fn line_start() -> String {
        csi1(1, 'K')
    }

    /// Clear current line and move to beginning (equivalent to EL 2 + CR)
    pub fn current_line() -> String {
        alloc::format!("{}\r", csi1(2, 'K'))
    }

    /// Clear N lines from current position downwards
    pub fn lines_down(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        let mut result = String::new();
        for _ in 0..n {
            result.push_str(&Self::line());
            result.push_str("\x1b[B"); // Move down
        }
        // Move back up to original line
        if n > 1 {
            result.push_str(&super::cursor::Cursor::up(n - 1));
        }
        result
    }

    /// Clear N lines from current position upwards
    pub fn lines_up(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        let mut result = String::new();
        for _ in 0..n {
            result.push_str(&super::cursor::Cursor::up(1));
            result.push_str(&Self::line());
        }
        result
    }

    /// Clear screen and move cursor to home
    pub fn screen_and_home() -> String {
        alloc::format!("{}\x1b[H", csi1(2, 'J'))
    }

    /// Clear screen, scrollback, and move cursor to home
    pub fn all() -> String {
        alloc::format!("{}\x1b[H", csi1(3, 'J'))
    }

    /// Clear from cursor to end of line and move to next line
    pub fn line_and_next() -> String {
        alloc::format!("{}\x1b[E", csi1(0, 'K'))
    }

    /// Clear tab stops
    pub fn tab_stop() -> String {
        csi1(0, 'g')
    }

    /// Clear all tab stops
    pub fn all_tab_stops() -> String {
        csi1(3, 'g')
    }
}

/// Scroll operations
pub struct Scroll;

impl Scroll {
    /// Scroll up N lines (SU)
    pub fn up(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[S".to_string();
        }
        csi1(n, 'S')
    }

    /// Scroll down N lines (SD)
    pub fn down(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[T".to_string();
        }
        csi1(n, 'T')
    }

    /// Scroll up one line and clear the new bottom line
    pub fn up_and_clear() -> String {
        alloc::format!("{}{}", "\x1b[S", csi1(2, 'K'))
    }

    /// Scroll down one line and clear the new top line
    pub fn down_and_clear() -> String {
        alloc::format!("{}{}", "\x1b[T", csi1(2, 'K'))
    }
}

/// Insert/Delete operations
pub struct Edit;

impl Edit {
    /// Insert N blank lines (IL)
    pub fn insert_lines(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[L".to_string();
        }
        csi1(n, 'L')
    }

    /// Delete N lines (DL)
    pub fn delete_lines(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[M".to_string();
        }
        csi1(n, 'M')
    }

    /// Insert N blank characters (ICH)
    pub fn insert_chars(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[@".to_string();
        }
        csi1(n, '@')
    }

    /// Delete N characters (DCH)
    pub fn delete_chars(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[P".to_string();
        }
        csi1(n, 'P')
    }

    /// Erase N characters (ECH)
    pub fn erase_chars(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[X".to_string();
        }
        csi1(n, 'X')
    }

    /// Repeat preceding character N times (REP)
    pub fn repeat_char(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        csi1(n, 'b')
    }
}

/// Tab operations
pub struct Tab;

impl Tab {
    /// Set tab stop at current column (HTS)
    pub fn set() -> String {
        "\x1bH".to_string()
    }

    /// Move forward N tabs (CHT)
    pub fn forward(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[I".to_string();
        }
        csi1(n, 'I')
    }

    /// Move backward N tabs (CBT)
    pub fn backward(n: u16) -> String {
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return "\x1b[Z".to_string();
        }
        csi1(n, 'Z')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_screen() {
        assert_eq!(Clear::screen(), "\x1b[2J");
        assert_eq!(Clear::screen_down(), "\x1b[0J");
        assert_eq!(Clear::screen_up(), "\x1b[1J");
        assert_eq!(Clear::screen_saved(), "\x1b[3J");
    }

    #[test]
    fn test_clear_line() {
        assert_eq!(Clear::line(), "\x1b[2K");
        assert_eq!(Clear::line_end(), "\x1b[0K");
        assert_eq!(Clear::line_start(), "\x1b[1K");
        assert_eq!(Clear::current_line(), "\x1b[2K\r");
    }

    #[test]
    fn test_scroll() {
        assert_eq!(Scroll::up(1), "\x1b[S");
        assert_eq!(Scroll::up(5), "\x1b[5S");
        assert_eq!(Scroll::down(1), "\x1b[T");
        assert_eq!(Scroll::down(5), "\x1b[5T");
    }

    #[test]
    fn test_edit() {
        assert_eq!(Edit::insert_lines(1), "\x1b[L");
        assert_eq!(Edit::delete_lines(1), "\x1b[M");
        assert_eq!(Edit::insert_chars(1), "\x1b[@");
        assert_eq!(Edit::delete_chars(1), "\x1b[P");
    }

    #[test]
    fn test_tab() {
        assert_eq!(Tab::set(), "\x1bH");
        assert_eq!(Tab::forward(1), "\x1b[I");
        assert_eq!(Tab::backward(1), "\x1b[Z");
    }
}
