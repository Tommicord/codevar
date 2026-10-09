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

//! ANSI escape code definitions and builder

use alloc::string::String;
use alloc::string::ToString;
use core::fmt;

/// ANSI escape code prefix
pub const CSI: &str = "\x1b[";
/// ANSI escape code prefix (alternative)
pub const ESC: &str = "\x1b";
/// OSC (Operating System Command) prefix
pub const OSC: &str = "\x1b]";
/// BEL (Bell) terminator for OSC
pub const BEL: &str = "\x07";
/// ST (String Terminator) for OSC
pub const ST: &str = "\x1b\\";

/// ANSI code trait for building escape sequences
pub trait AnsiCode {
    /// Returns the ANSI code as a string
    fn code(&self) -> &'static str;

    /// Returns the full escape sequence
    fn sequence(&self) -> String {
        alloc::format!("{}{}", CSI, self.code())
    }
}

/// ANSI escape sequence builder
#[derive(Debug, Clone, Default)]
pub struct AnsiBuilder<const N: usize = 16> {
    params: heapless::Vec<u16, N>,
    command: Option<char>,
    private: bool,
}

impl<const N: usize> AnsiBuilder<N> {
    /// Create a new ANSI builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter
    pub fn param(mut self, param: u16) -> Self {
        let _ = self.params.push(param);
        self
    }

    /// Add multiple parameters
    pub fn params(mut self, params: &[u16]) -> Self {
        let _ = self.params.extend_from_slice(params);
        self
    }

    /// Set the command character
    pub fn command(mut self, cmd: char) -> Self {
        self.command = Some(cmd);
        self
    }

    /// Mark as private sequence (prefixed with ? or >)
    pub fn private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Build the ANSI sequence
    pub fn build(self) -> String {
        let mut result = String::with_capacity(8 + self.params.len() * 3);
        result.push_str(CSI);
        if self.private {
            result.push('?');
        }
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                result.push(';');
            }
            result.push_str(&param.to_string());
        }
        if let Some(cmd) = self.command {
            result.push(cmd);
        }
        result
    }

    /// Build as a raw escape sequence (without CSI prefix)
    pub fn build_raw(self) -> String {
        let mut result = String::with_capacity(8 + self.params.len() * 3);
        result.push_str(ESC);
        if self.private {
            result.push('?');
        }
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                result.push(';');
            }
            result.push_str(&param.to_string());
        }
        if let Some(cmd) = self.command {
            result.push(cmd);
        }
        result
    }
}

/// Build an OSC (Operating System Command) sequence
pub fn osc(command: u16, payload: &str) -> String {
    alloc::format!("{}{};{}{}", OSC, command, payload, ST)
}

/// Build a simple CSI sequence with parameters
pub fn csi(params: &[u16], command: char) -> String {
    let mut result = String::with_capacity(8 + params.len() * 3);
    result.push_str(CSI);
    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            result.push(';');
        }
        result.push_str(&param.to_string());
    }
    result.push(command);
    result
}

/// Build a CSI sequence with a single parameter
pub fn csi1(param: u16, command: char) -> String {
    alloc::format!("{}{}{}", CSI, param, command)
}

/// Build a CSI sequence with two parameters
pub fn csi2(param1: u16, param2: u16, command: char) -> String {
    alloc::format!("{}{};{}{}", CSI, param1, param2, command)
}

/// Build a CSI sequence with no parameters
pub fn csi0(command: char) -> String {
    alloc::format!("{}{}", CSI, command)
}

/// SGR (Select Graphic Rendition) parameter codes
pub mod sgr {
    /// Reset all attributes
    pub const RESET: u16 = 0;
    /// Bold/bright
    pub const BOLD: u16 = 1;
    /// Dim/faint
    pub const DIM: u16 = 2;
    /// Italic
    pub const ITALIC: u16 = 3;
    /// Underline
    pub const UNDERLINE: u16 = 4;
    /// Slow blink
    pub const SLOW_BLINK: u16 = 5;
    /// Rapid blink
    pub const RAPID_BLINK: u16 = 6;
    /// Reverse video
    pub const REVERSE: u16 = 7;
    /// Conceal/hide
    pub const CONCEAL: u16 = 8;
    /// Crossed out
    pub const CROSSED_OUT: u16 = 9;
    /// Primary font
    pub const PRIMARY_FONT: u16 = 10;
    /// Alternative fonts 1-9
    pub const ALT_FONT_1: u16 = 11;
    pub const ALT_FONT_2: u16 = 12;
    pub const ALT_FONT_3: u16 = 13;
    pub const ALT_FONT_4: u16 = 14;
    pub const ALT_FONT_5: u16 = 15;
    pub const ALT_FONT_6: u16 = 16;
    pub const ALT_FONT_7: u16 = 17;
    pub const ALT_FONT_8: u16 = 18;
    pub const ALT_FONT_9: u16 = 19;
    /// Fraktur (Gothic)
    pub const FRAKTUR: u16 = 20;
    /// Double underline
    pub const DOUBLE_UNDERLINE: u16 = 21;
    /// Normal intensity (reset bold/dim)
    pub const NORMAL_INTENSITY: u16 = 22;
    /// Not italic, not fraktur
    pub const NOT_ITALIC_FRAKTUR: u16 = 23;
    /// Not underline
    pub const NOT_UNDERLINE: u16 = 24;
    /// Not blink
    pub const NOT_BLINK: u16 = 25;
    /// Proportional spacing
    pub const PROPORTIONAL: u16 = 26;
    /// Not reverse
    pub const NOT_REVERSE: u16 = 27;
    /// Not concealed
    pub const NOT_CONCEAL: u16 = 28;
    /// Not crossed out
    pub const NOT_CROSSED_OUT: u16 = 29;
    /// Foreground colors (30-37)
    pub const FG_BLACK: u16 = 30;
    pub const FG_RED: u16 = 31;
    pub const FG_GREEN: u16 = 32;
    pub const FG_YELLOW: u16 = 33;
    pub const FG_BLUE: u16 = 34;
    pub const FG_MAGENTA: u16 = 35;
    pub const FG_CYAN: u16 = 36;
    pub const FG_WHITE: u16 = 37;
    /// Default foreground
    pub const FG_DEFAULT: u16 = 39;
    /// Background colors (40-47)
    pub const BG_BLACK: u16 = 40;
    pub const BG_RED: u16 = 41;
    pub const BG_GREEN: u16 = 42;
    pub const BG_YELLOW: u16 = 43;
    pub const BG_BLUE: u16 = 44;
    pub const BG_MAGENTA: u16 = 45;
    pub const BG_CYAN: u16 = 46;
    pub const BG_WHITE: u16 = 47;
    /// Default background
    pub const BG_DEFAULT: u16 = 49;
    /// Framed
    pub const FRAMED: u16 = 51;
    /// Encircled
    pub const ENCIRCLED: u16 = 52;
    /// Overlined
    pub const OVERLINED: u16 = 53;
    /// Not framed/encircled
    pub const NOT_FRAMED_ENCIRCLED: u16 = 54;
    /// Not overlined
    pub const NOT_OVERLINED: u16 = 55;
    /// Bright foreground colors (90-97)
    pub const FG_BRIGHT_BLACK: u16 = 90;
    pub const FG_BRIGHT_RED: u16 = 91;
    pub const FG_BRIGHT_GREEN: u16 = 92;
    pub const FG_BRIGHT_YELLOW: u16 = 93;
    pub const FG_BRIGHT_BLUE: u16 = 94;
    pub const FG_BRIGHT_MAGENTA: u16 = 95;
    pub const FG_BRIGHT_CYAN: u16 = 96;
    pub const FG_BRIGHT_WHITE: u16 = 97;
    /// Bright background colors (100-107)
    pub const BG_BRIGHT_BLACK: u16 = 100;
    pub const BG_BRIGHT_RED: u16 = 101;
    pub const BG_BRIGHT_GREEN: u16 = 102;
    pub const BG_BRIGHT_YELLOW: u16 = 103;
    pub const BG_BRIGHT_BLUE: u16 = 104;
    pub const BG_BRIGHT_MAGENTA: u16 = 105;
    pub const BG_BRIGHT_CYAN: u16 = 106;
    pub const BG_BRIGHT_WHITE: u16 = 107;
}

/// Cursor control codes
pub mod cursor {
    use super::*;

    /// Move cursor up N lines
    pub fn up(n: u16) -> String {
        csi1(n, 'A')
    }
    /// Move cursor down N lines
    pub fn down(n: u16) -> String {
        csi1(n, 'B')
    }
    /// Move cursor forward N columns
    pub fn forward(n: u16) -> String {
        csi1(n, 'C')
    }
    /// Move cursor backward N columns
    pub fn backward(n: u16) -> String {
        csi1(n, 'D')
    }
    /// Move cursor to beginning of next line N lines down
    pub fn next_line(n: u16) -> String {
        csi1(n, 'E')
    }
    /// Move cursor to beginning of previous line N lines up
    pub fn prev_line(n: u16) -> String {
        csi1(n, 'F')
    }
    /// Move cursor to column N (1-indexed)
    pub fn column(n: u16) -> String {
        csi1(n, 'G')
    }
    /// Move cursor to row N, column M (1-indexed)
    pub fn position(row: u16, col: u16) -> String {
        csi2(row, col, 'H')
    }
    /// Same as position
    pub fn cup(row: u16, col: u16) -> String {
        csi2(row, col, 'f')
    }
    /// Save cursor position (DECSC)
    pub fn save() -> String {
        csi0('s')
    }
    /// Restore cursor position (DECRC)
    pub fn restore() -> String {
        csi0('u')
    }
    /// Hide cursor
    pub fn hide() -> String {
        csi0('?') + "25l"
    }
    /// Show cursor
    pub fn show() -> String {
        csi0('?') + "25h"
    }
    /// Enable DECCKM (cursor keys mode)
    pub fn enable_keypad() -> String {
        csi0('?') + "1h"
    }
    /// Disable DECCKM
    pub fn disable_keypad() -> String {
        csi0('?') + "1l"
    }
}

/// Erase/clear codes
pub mod erase {
    use super::*;

    /// Clear from cursor to end of screen
    pub fn screen_down() -> String {
        csi1(0, 'J')
    }
    /// Clear from cursor to beginning of screen
    pub fn screen_up() -> String {
        csi1(1, 'J')
    }
    /// Clear entire screen
    pub fn screen() -> String {
        csi1(2, 'J')
    }
    /// Clear entire screen and scrollback
    pub fn screen_saved() -> String {
        csi1(3, 'J')
    }
    /// Clear from cursor to end of line
    pub fn line_right() -> String {
        csi1(0, 'K')
    }
    /// Clear from cursor to beginning of line
    pub fn line_left() -> String {
        csi1(1, 'K')
    }
    /// Clear entire line
    pub fn line() -> String {
        csi1(2, 'K')
    }
}

/// Scroll codes
pub mod scroll {
    use super::*;

    /// Scroll up N lines
    pub fn up(n: u16) -> String {
        csi1(n, 'S')
    }
    /// Scroll down N lines
    pub fn down(n: u16) -> String {
        csi1(n, 'T')
    }
}

/// Mode setting codes (DEC private modes)
pub mod mode {
    use super::*;

    /// Set mode (DECSET)
    pub fn set(mode: u16) -> String {
        alloc::format!("{}?{}h", CSI, mode)
    }

    /// Reset mode (DECRST)
    pub fn reset(mode: u16) -> String {
        alloc::format!("{}?{}l", CSI, mode)
    }

    /// Application cursor keys (DECCKM)
    pub const CURSOR_KEYS: u16 = 1;
    /// ANSI/VT52 mode (DECANM)
    pub const ANSI_MODE: u16 = 2;
    /// 132 column mode (DECCOLM)
    pub const COLUMNS_132: u16 = 3;
    /// Smooth scroll (DECSCLM)
    pub const SMOOTH_SCROLL: u16 = 4;
    /// Reverse video (DECSCNM)
    pub const REVERSE_VIDEO: u16 = 5;
    /// Origin mode (DECOM)
    pub const ORIGIN: u16 = 6;
    /// Auto-wrap (DECAWM)
    pub const AUTO_WRAP: u16 = 7;
    /// Auto-repeat (DECARM)
    pub const AUTO_REPEAT: u16 = 8;
    /// Print form feed (DECPFF)
    pub const PRINT_FF: u16 = 18;
    /// Print extent (DECPEX)
    pub const PRINT_EXTENT: u16 = 19;
    /// Show cursor (DECTCEM)
    pub const SHOW_CURSOR: u16 = 25;
    /// Enable bracketed paste
    pub const BRACKETED_PASTE: u16 = 2004;
    /// Enable focus reporting
    pub const FOCUS_REPORTING: u16 = 1004;
    /// Enable mouse tracking (X10)
    pub const MOUSE_X10: u16 = 9;
    /// Enable mouse tracking (VT200)
    pub const MOUSE_VT200: u16 = 1000;
    /// Enable mouse tracking (button event)
    pub const MOUSE_BUTTON: u16 = 1002;
    /// Enable mouse tracking (any event)
    pub const MOUSE_ANY: u16 = 1003;
    /// Enable extended mouse mode
    pub const MOUSE_EXTENDED: u16 = 1005;
    /// Enable SGR mouse mode
    pub const MOUSE_SGR: u16 = 1006;
    /// Enable urxvt mouse mode
    pub const MOUSE_URXVT: u16 = 1015;
}

/// Device status report codes
pub mod dsr {
    use super::*;

    /// Request cursor position report (CPR)
    pub fn cursor_position() -> String {
        csi1(6, 'n')
    }
    /// Request terminal status
    pub fn status() -> String {
        csi1(5, 'n')
    }
    /// Request device attributes (primary)
    pub fn primary_da() -> String {
        csi0('c')
    }
    /// Request device attributes (secondary)
    pub fn secondary_da() -> String {
        csi1(0, 'c')
    }
}

/// Title and window manipulation (OSC)
pub mod title {
    use super::*;

    /// Set window title and icon name
    pub fn set_title(title: &str) -> String {
        osc(0, title)
    }
    /// Set window title
    pub fn set_window_title(title: &str) -> String {
        osc(2, title)
    }
    /// Set icon name
    pub fn set_icon_name(name: &str) -> String {
        osc(1, name)
    }
}

/// Hyperlink support (OSC 8)
pub mod hyperlink {
    use super::*;

    /// Create hyperlink
    pub fn link(uri: &str, text: &str) -> String {
        alloc::format!("{}8;;{}{}{}{}8;;{}", OSC, uri, ST, text, OSC, ST)
    }

    /// Create hyperlink with explicit ID
    pub fn link_with_id(id: &str, uri: &str, text: &str) -> String {
        alloc::format!("{}8;{};{}{}{}{}8;;{}", OSC, id, uri, ST, text, OSC, ST)
    }
}

/// ANSI sequence representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnsiSequence {
    sequence: String,
}

impl AnsiSequence {
    /// Create a new ANSI sequence from a string
    pub fn new(sequence: String) -> Self {
        Self { sequence }
    }

    /// Get the raw sequence
    pub fn as_str(&self) -> &str {
        &self.sequence
    }

    /// Get the sequence as bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.sequence.as_bytes()
    }
}

impl fmt::Display for AnsiSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.sequence)
    }
}

impl From<&str> for AnsiSequence {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl From<String> for AnsiSequence {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Common ANSI sequences as constants
pub mod sequences {
    /// Reset all attributes
    pub const RESET: &str = "\x1b[0m";
    /// Bold
    pub const BOLD: &str = "\x1b[1m";
    /// Dim
    pub const DIM: &str = "\x1b[2m";
    /// Italic
    pub const ITALIC: &str = "\x1b[3m";
    /// Underline
    pub const UNDERLINE: &str = "\x1b[4m";
    /// Blink
    pub const BLINK: &str = "\x1b[5m";
    /// Reverse
    pub const REVERSE: &str = "\x1b[7m";
    /// Hidden
    pub const HIDDEN: &str = "\x1b[8m";
    /// Strikethrough
    pub const STRIKETHROUGH: &str = "\x1b[9m";

    /// Clear screen
    pub const CLEAR_SCREEN: &str = "\x1b[2J";
    /// Clear screen and move cursor to home
    pub const CLEAR_SCREEN_HOME: &str = "\x1b[2J\x1b[H";
    /// Clear line
    pub const CLEAR_LINE: &str = "\x1b[2K";
    /// Clear to end of line
    pub const CLEAR_LINE_END: &str = "\x1b[0K";
    /// Clear to beginning of line
    pub const CLEAR_LINE_START: &str = "\x1b[1K";

    /// Cursor home
    pub const CURSOR_HOME: &str = "\x1b[H";
    /// Cursor up 1
    pub const CURSOR_UP: &str = "\x1b[A";
    /// Cursor down 1
    pub const CURSOR_DOWN: &str = "\x1b[B";
    /// Cursor forward 1
    pub const CURSOR_FORWARD: &str = "\x1b[C";
    /// Cursor back 1
    pub const CURSOR_BACK: &str = "\x1b[D";
    /// Save cursor position
    pub const CURSOR_SAVE: &str = "\x1b[s";
    /// Restore cursor position
    pub const CURSOR_RESTORE: &str = "\x1b[u";
    /// Hide cursor
    pub const CURSOR_HIDE: &str = "\x1b[?25l";
    /// Show cursor
    pub const CURSOR_SHOW: &str = "\x1b[?25h";

    /// Enable bracketed paste
    pub const BRACKETED_PASTE_ON: &str = "\x1b[?2004h";
    /// Disable bracketed paste
    pub const BRACKETED_PASTE_OFF: &str = "\x1b[?2004l";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ansi_builder() {
        let seq = AnsiBuilder::<16>::new()
            .param(1)
            .param(31)
            .command('m')
            .build();
        assert_eq!(seq, "\x1b[1;31m");
    }

    #[test]
    fn test_csi() {
        assert_eq!(csi(&[1, 31], 'm'), "\x1b[1;31m");
        assert_eq!(csi1(5, 'm'), "\x1b[5m");
        assert_eq!(csi2(10, 20, 'H'), "\x1b[10;20H");
        assert_eq!(csi0('m'), "\x1b[m");
    }

    #[test]
    fn test_osc() {
        assert_eq!(osc(2, "Title"), "\x1b]2;Title\x1b\\");
    }

    #[test]
    fn test_cursor() {
        assert_eq!(cursor::up(5), "\x1b[5A");
        assert_eq!(cursor::position(10, 20), "\x1b[10;20H");
    }

    #[test]
    fn test_erase() {
        assert_eq!(erase::screen(), "\x1b[2J");
        assert_eq!(erase::line(), "\x1b[2K");
    }

    #[test]
    fn test_mode() {
        assert_eq!(mode::set(25), "\x1b[?25h");
        assert_eq!(mode::reset(25), "\x1b[?25l");
    }

    #[test]
    fn test_hyperlink() {
        let link = hyperlink::link("https://example.com", "Click here");
        assert!(link.contains("https://example.com"));
        assert!(link.contains("Click here"));
    }
}
