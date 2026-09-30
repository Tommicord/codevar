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

//! Style and color definitions for ANSI output

use super::console_ansi::sgr;
use alloc::string::String;
use alloc::string::ToString;
use core::fmt;

/// Style attribute flags using bitwise operations
/// This replaces individual boolean fields for better memory efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct StyleAttr(u32);

impl StyleAttr {
    pub const NONE: Self = Self(0);
    pub const BOLD: Self = Self(1 << 0);
    pub const DIM: Self = Self(1 << 1);
    pub const ITALIC: Self = Self(1 << 2);
    pub const UNDERLINE: Self = Self(1 << 3);
    pub const DOUBLE_UNDERLINE: Self = Self(1 << 4);
    pub const BLINK: Self = Self(1 << 5);
    pub const RAPID_BLINK: Self = Self(1 << 6);
    pub const REVERSE: Self = Self(1 << 7);
    pub const CONCEAL: Self = Self(1 << 8);
    pub const CROSSED_OUT: Self = Self(1 << 9);
    pub const FRAMED: Self = Self(1 << 10);
    pub const ENCIRCLED: Self = Self(1 << 11);
    pub const OVERLINED: Self = Self(1 << 12);

    /// All text decoration attributes (underline, double_underline, overlined)
    pub const TEXT_DECORATION: Self = Self(Self::UNDERLINE.0 | Self::DOUBLE_UNDERLINE.0 | Self::OVERLINED.0);

    /// All blink attributes
    pub const BLINK_ATTRS: Self = Self(Self::BLINK.0 | Self::RAPID_BLINK.0);

    /// All intensity attributes (bold, dim)
    pub const INTENSITY: Self = Self(Self::BOLD.0 | Self::DIM.0);

    #[inline]
    pub const fn empty() -> Self {
        Self::NONE
    }

    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
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

impl core::ops::BitOr for StyleAttr {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl core::ops::BitOrAssign for StyleAttr {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        self.insert(other);
    }
}

impl core::ops::BitAnd for StyleAttr {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.intersection(other)
    }
}

impl core::ops::BitAndAssign for StyleAttr {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

impl core::ops::BitXor for StyleAttr {
    type Output = Self;
    #[inline]
    fn bitxor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}

impl core::ops::BitXorAssign for StyleAttr {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0;
    }
}

impl core::ops::Not for StyleAttr {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

/// ANSI foreground colors (standard 8 colors)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AnsiColor {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7,
    BrightBlack = 8,
    BrightRed = 9,
    BrightGreen = 10,
    BrightYellow = 11,
    BrightBlue = 12,
    BrightMagenta = 13,
    BrightCyan = 14,
    BrightWhite = 15,
    Default = 255,
}

impl AnsiColor {
    /// Get the SGR code for foreground
    pub const fn fg_code(self) -> u16 {
        match self {
            AnsiColor::Black => sgr::FG_BLACK,
            AnsiColor::Red => sgr::FG_RED,
            AnsiColor::Green => sgr::FG_GREEN,
            AnsiColor::Yellow => sgr::FG_YELLOW,
            AnsiColor::Blue => sgr::FG_BLUE,
            AnsiColor::Magenta => sgr::FG_MAGENTA,
            AnsiColor::Cyan => sgr::FG_CYAN,
            AnsiColor::White => sgr::FG_WHITE,
            AnsiColor::BrightBlack => sgr::FG_BRIGHT_BLACK,
            AnsiColor::BrightRed => sgr::FG_BRIGHT_RED,
            AnsiColor::BrightGreen => sgr::FG_BRIGHT_GREEN,
            AnsiColor::BrightYellow => sgr::FG_BRIGHT_YELLOW,
            AnsiColor::BrightBlue => sgr::FG_BRIGHT_BLUE,
            AnsiColor::BrightMagenta => sgr::FG_BRIGHT_MAGENTA,
            AnsiColor::BrightCyan => sgr::FG_BRIGHT_CYAN,
            AnsiColor::BrightWhite => sgr::FG_BRIGHT_WHITE,
            AnsiColor::Default => sgr::FG_DEFAULT,
        }
    }

    /// Get the SGR code for background
    pub const fn bg_code(self) -> u16 {
        match self {
            AnsiColor::Black => sgr::BG_BLACK,
            AnsiColor::Red => sgr::BG_RED,
            AnsiColor::Green => sgr::BG_GREEN,
            AnsiColor::Yellow => sgr::BG_YELLOW,
            AnsiColor::Blue => sgr::BG_BLUE,
            AnsiColor::Magenta => sgr::BG_MAGENTA,
            AnsiColor::Cyan => sgr::BG_CYAN,
            AnsiColor::White => sgr::BG_WHITE,
            AnsiColor::BrightBlack => sgr::BG_BRIGHT_BLACK,
            AnsiColor::BrightRed => sgr::BG_BRIGHT_RED,
            AnsiColor::BrightGreen => sgr::BG_BRIGHT_GREEN,
            AnsiColor::BrightYellow => sgr::BG_BRIGHT_YELLOW,
            AnsiColor::BrightBlue => sgr::BG_BRIGHT_BLUE,
            AnsiColor::BrightMagenta => sgr::BG_BRIGHT_MAGENTA,
            AnsiColor::BrightCyan => sgr::BG_BRIGHT_CYAN,
            AnsiColor::BrightWhite => sgr::BG_BRIGHT_WHITE,
            AnsiColor::Default => sgr::BG_DEFAULT,
        }
    }

    /// Get foreground ANSI sequence
    pub fn fg(self) -> String {
        if self == AnsiColor::Default {
            return "\x1b[39m".to_string();
        }
        alloc::format!("\x1b[{}m", self.fg_code())
    }

    /// Get background ANSI sequence
    pub fn bg(self) -> String {
        if self == AnsiColor::Default {
            return "\x1b[49m".to_string();
        }
        alloc::format!("\x1b[{}m", self.bg_code())
    }

    /// Get bright foreground variant
    pub const fn bright(self) -> AnsiColor {
        match self {
            AnsiColor::Black => AnsiColor::BrightBlack,
            AnsiColor::Red => AnsiColor::BrightRed,
            AnsiColor::Green => AnsiColor::BrightGreen,
            AnsiColor::Yellow => AnsiColor::BrightYellow,
            AnsiColor::Blue => AnsiColor::BrightBlue,
            AnsiColor::Magenta => AnsiColor::BrightMagenta,
            AnsiColor::Cyan => AnsiColor::BrightCyan,
            AnsiColor::White => AnsiColor::BrightWhite,
            c @ AnsiColor::BrightBlack
            | c @ AnsiColor::BrightRed
            | c @ AnsiColor::BrightGreen
            | c @ AnsiColor::BrightYellow
            | c @ AnsiColor::BrightBlue
            | c @ AnsiColor::BrightMagenta
            | c @ AnsiColor::BrightCyan
            | c @ AnsiColor::BrightWhite
            | c @ AnsiColor::Default => c,
        }
    }

    /// Get dim/normal variant
    pub const fn dim(self) -> AnsiColor {
        match self {
            AnsiColor::BrightBlack => AnsiColor::Black,
            AnsiColor::BrightRed => AnsiColor::Red,
            AnsiColor::BrightGreen => AnsiColor::Green,
            AnsiColor::BrightYellow => AnsiColor::Yellow,
            AnsiColor::BrightBlue => AnsiColor::Blue,
            AnsiColor::BrightMagenta => AnsiColor::Magenta,
            AnsiColor::BrightCyan => AnsiColor::Cyan,
            AnsiColor::BrightWhite => AnsiColor::White,
            c => c,
        }
    }

    /// Create from RGB values (approximates to nearest ANSI color)
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        // Simple approximation to 16-color palette
        let brightness = (r as u16 + g as u16 + b as u16) / 3;
        let is_bright = brightness > 128;

        // Find dominant color
        let max = r.max(g).max(b);

        if max == r && max == g && max == b {
            // Grayscale
            if brightness < 64 {
                AnsiColor::Black
            } else if brightness < 192 {
                AnsiColor::BrightBlack // Gray
            } else {
                AnsiColor::White
            }
        } else if max == r && max == g {
            // Yellow-ish
            if is_bright {
                AnsiColor::BrightYellow
            } else {
                AnsiColor::Yellow
            }
        } else if max == r && max == b {
            // Magenta-ish
            if is_bright {
                AnsiColor::BrightMagenta
            } else {
                AnsiColor::Magenta
            }
        } else if max == g && max == b {
            // Cyan-ish
            if is_bright {
                AnsiColor::BrightCyan
            } else {
                AnsiColor::Cyan
            }
        } else if max == r {
            // Red-ish
            if is_bright {
                AnsiColor::BrightRed
            } else {
                AnsiColor::Red
            }
        } else if max == g {
            // Green-ish
            if is_bright {
                AnsiColor::BrightGreen
            } else {
                AnsiColor::Green
            }
        } else {
            // Blue-ish
            if is_bright {
                AnsiColor::BrightBlue
            } else {
                AnsiColor::Blue
            }
        }
    }

    /// Create from 256-color index
    pub fn from_256(index: u8) -> Self {
        match index {
            // SAFETY: `AnsiColor` is `#[repr(u8)]` with explicit discriminants
            // covering `0..=15`, so every value in these ranges is a valid variant.
            0..=7 => unsafe { core::mem::transmute::<u8, AnsiColor>(index) },
            // SAFETY: same invariant as the `0..=7` arm above.
            8..=15 => unsafe { core::mem::transmute::<u8, AnsiColor>(index) },
            16..=231 => {
                // 6x6x6 color cube, map to nearest 16-color
                let idx = index - 16;
                let r = (idx / 36) * 51;
                let g = ((idx % 36) / 6) * 51;
                let b = (idx % 6) * 51;
                Self::from_rgb(r, g, b)
            }
            232..=255 => {
                // Grayscale ramp
                let gray = (index - 232) * 10 + 8;
                if gray < 64 {
                    AnsiColor::Black
                } else if gray < 192 {
                    AnsiColor::BrightBlack
                } else {
                    AnsiColor::White
                }
            }
        }
    }

    /// Get color name
    pub const fn name(self) -> &'static str {
        match self {
            AnsiColor::Black => "black",
            AnsiColor::Red => "red",
            AnsiColor::Green => "green",
            AnsiColor::Yellow => "yellow",
            AnsiColor::Blue => "blue",
            AnsiColor::Magenta => "magenta",
            AnsiColor::Cyan => "cyan",
            AnsiColor::White => "white",
            AnsiColor::BrightBlack => "bright_black",
            AnsiColor::BrightRed => "bright_red",
            AnsiColor::BrightGreen => "bright_green",
            AnsiColor::BrightYellow => "bright_yellow",
            AnsiColor::BrightBlue => "bright_blue",
            AnsiColor::BrightMagenta => "bright_magenta",
            AnsiColor::BrightCyan => "bright_cyan",
            AnsiColor::BrightWhite => "bright_white",
            AnsiColor::Default => "default",
        }
    }
}

/// ANSI text styles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum AnsiStyle {
    Reset = sgr::RESET,
    Bold = sgr::BOLD,
    Dim = sgr::DIM,
    Italic = sgr::ITALIC,
    Underline = sgr::UNDERLINE,
    SlowBlink = sgr::SLOW_BLINK,
    RapidBlink = sgr::RAPID_BLINK,
    Reverse = sgr::REVERSE,
    Conceal = sgr::CONCEAL,
    CrossedOut = sgr::CROSSED_OUT,
    DoubleUnderline = sgr::DOUBLE_UNDERLINE,
    Framed = sgr::FRAMED,
    Encircled = sgr::ENCIRCLED,
    Overlined = sgr::OVERLINED,
    NotBold = sgr::NORMAL_INTENSITY,
    NotItalic = sgr::NOT_ITALIC_FRAKTUR,
    NotUnderline = sgr::NOT_UNDERLINE,
    NotBlink = sgr::NOT_BLINK,
    NotReverse = sgr::NOT_REVERSE,
    NotConceal = sgr::NOT_CONCEAL,
    NotCrossedOut = sgr::NOT_CROSSED_OUT,
    NotFramed = sgr::NOT_FRAMED_ENCIRCLED,
    NotOverlined = sgr::NOT_OVERLINED,
}

impl AnsiStyle {
    /// Get the ANSI sequence for this style
    pub fn sequence(self) -> String {
        if self == AnsiStyle::Reset {
            return "\x1b[0m".to_string();
        }
        alloc::format!("\x1b[{}m", self as u16)
    }

    /// Get the reset sequence for this style
    pub fn reset_sequence(self) -> String {
        match self {
            AnsiStyle::Bold | AnsiStyle::Dim => AnsiStyle::NotBold.sequence(),
            AnsiStyle::Italic => AnsiStyle::NotItalic.sequence(),
            AnsiStyle::Underline | AnsiStyle::DoubleUnderline => AnsiStyle::NotUnderline.sequence(),
            AnsiStyle::SlowBlink | AnsiStyle::RapidBlink => AnsiStyle::NotBlink.sequence(),
            AnsiStyle::Reverse => AnsiStyle::NotReverse.sequence(),
            AnsiStyle::Conceal => AnsiStyle::NotConceal.sequence(),
            AnsiStyle::CrossedOut => AnsiStyle::NotCrossedOut.sequence(),
            AnsiStyle::Framed | AnsiStyle::Encircled => AnsiStyle::NotFramed.sequence(),
            AnsiStyle::Overlined => AnsiStyle::NotOverlined.sequence(),
            _ => AnsiStyle::Reset.sequence(),
        }
    }
}

/// Combined style with foreground, background, and attributes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Style {
    pub fg: Option<AnsiColor>,
    pub bg: Option<AnsiColor>,
    pub attrs: StyleAttr,
}

impl Style {
    /// Create a new empty style
    pub const fn new() -> Self {
        Self {
            fg: None,
            bg: None,
            attrs: StyleAttr::NONE,
        }
    }

    /// Set foreground color
    pub const fn fg(mut self, color: AnsiColor) -> Self {
        self.fg = Some(color);
        self
    }

    /// Set background color
    pub const fn bg(mut self, color: AnsiColor) -> Self {
        self.bg = Some(color);
        self
    }

    /// Enable bold
    pub const fn bold(mut self) -> Self {
        self.attrs.insert(StyleAttr::BOLD);
        self
    }

    /// Enable dim
    pub const fn dim(mut self) -> Self {
        self.attrs.insert(StyleAttr::DIM);
        self
    }

    /// Enable italic
    pub const fn italic(mut self) -> Self {
        self.attrs.insert(StyleAttr::ITALIC);
        self
    }

    /// Enable underline
    pub const fn underline(mut self) -> Self {
        self.attrs.insert(StyleAttr::UNDERLINE);
        self
    }

    /// Enable double underline
    pub const fn double_underline(mut self) -> Self {
        self.attrs.insert(StyleAttr::DOUBLE_UNDERLINE);
        self
    }

    /// Enable blink
    pub const fn blink(mut self) -> Self {
        self.attrs.insert(StyleAttr::BLINK);
        self
    }

    /// Enable rapid blink
    pub const fn rapid_blink(mut self) -> Self {
        self.attrs.insert(StyleAttr::RAPID_BLINK);
        self
    }

    /// Enable reverse
    pub const fn reverse(mut self) -> Self {
        self.attrs.insert(StyleAttr::REVERSE);
        self
    }

    /// Enable conceal
    pub const fn conceal(mut self) -> Self {
        self.attrs.insert(StyleAttr::CONCEAL);
        self
    }

    /// Enable crossed out
    pub const fn crossed_out(mut self) -> Self {
        self.attrs.insert(StyleAttr::CROSSED_OUT);
        self
    }

    /// Enable framed
    pub const fn framed(mut self) -> Self {
        self.attrs.insert(StyleAttr::FRAMED);
        self
    }

    /// Enable encircled
    pub const fn encircled(mut self) -> Self {
        self.attrs.insert(StyleAttr::ENCIRCLED);
        self
    }

    /// Enable overlined
    pub const fn overlined(mut self) -> Self {
        self.attrs.insert(StyleAttr::OVERLINED);
        self
    }

    /// Check if an attribute is set
    #[inline]
    pub const fn has_attr(self, attr: StyleAttr) -> bool {
        self.attrs.contains(attr)
    }

    /// Build the ANSI sequence for this style
    pub fn build(self) -> String {
        let mut params = heapless::Vec::<u16, 16>::new();

        if self.attrs.contains(StyleAttr::BOLD) {
            let _ = params.push(sgr::BOLD);
        }
        if self.attrs.contains(StyleAttr::DIM) {
            let _ = params.push(sgr::DIM);
        }
        if self.attrs.contains(StyleAttr::ITALIC) {
            let _ = params.push(sgr::ITALIC);
        }
        if self.attrs.contains(StyleAttr::UNDERLINE) {
            let _ = params.push(sgr::UNDERLINE);
        }
        if self.attrs.contains(StyleAttr::DOUBLE_UNDERLINE) {
            let _ = params.push(sgr::DOUBLE_UNDERLINE);
        }
        if self.attrs.contains(StyleAttr::BLINK) {
            let _ = params.push(sgr::SLOW_BLINK);
        }
        if self.attrs.contains(StyleAttr::RAPID_BLINK) {
            let _ = params.push(sgr::RAPID_BLINK);
        }
        if self.attrs.contains(StyleAttr::REVERSE) {
            let _ = params.push(sgr::REVERSE);
        }
        if self.attrs.contains(StyleAttr::CONCEAL) {
            let _ = params.push(sgr::CONCEAL);
        }
        if self.attrs.contains(StyleAttr::CROSSED_OUT) {
            let _ = params.push(sgr::CROSSED_OUT);
        }
        if self.attrs.contains(StyleAttr::FRAMED) {
            let _ = params.push(sgr::FRAMED);
        }
        if self.attrs.contains(StyleAttr::ENCIRCLED) {
            let _ = params.push(sgr::ENCIRCLED);
        }
        if self.attrs.contains(StyleAttr::OVERLINED) {
            let _ = params.push(sgr::OVERLINED);
        }

        if let Some(fg) = self.fg {
            let _ = params.push(fg.fg_code());
        }

        if let Some(bg) = self.bg {
            let _ = params.push(bg.bg_code());
        }

        if params.is_empty() {
            String::new()
        } else {
            super::console_ansi::csi(&params, 'm')
        }
    }

    /// Build the reset sequence for this style
    pub fn reset(&self) -> String {
        let mut params = heapless::Vec::<u16, 16>::new();

        if self.attrs.intersects(StyleAttr::INTENSITY) {
            let _ = params.push(sgr::NORMAL_INTENSITY);
        }
        if self.attrs.contains(StyleAttr::ITALIC) {
            let _ = params.push(sgr::NOT_ITALIC_FRAKTUR);
        }
        if self.attrs.intersects(StyleAttr::TEXT_DECORATION) {
            let _ = params.push(sgr::NOT_UNDERLINE);
        }
        if self.attrs.intersects(StyleAttr::BLINK_ATTRS) {
            let _ = params.push(sgr::NOT_BLINK);
        }
        if self.attrs.contains(StyleAttr::REVERSE) {
            let _ = params.push(sgr::NOT_REVERSE);
        }
        if self.attrs.contains(StyleAttr::CONCEAL) {
            let _ = params.push(sgr::NOT_CONCEAL);
        }
        if self.attrs.contains(StyleAttr::CROSSED_OUT) {
            let _ = params.push(sgr::NOT_CROSSED_OUT);
        }
        if self
            .attrs
            .intersects(StyleAttr::FRAMED | StyleAttr::ENCIRCLED)
        {
            let _ = params.push(sgr::NOT_FRAMED_ENCIRCLED);
        }
        if self.attrs.contains(StyleAttr::OVERLINED) {
            let _ = params.push(sgr::NOT_OVERLINED);
        }

        if self.fg.is_some() {
            let _ = params.push(sgr::FG_DEFAULT);
        }
        if self.bg.is_some() {
            let _ = params.push(sgr::BG_DEFAULT);
        }

        if params.is_empty() {
            String::new()
        } else {
            super::console_ansi::csi(&params, 'm')
        }
    }

    /// Apply style to text
    pub fn apply(self, text: &str) -> StyledText {
        let mut s = heapless::String::new();
        let _ = s.push_str(text);
        StyledText { style: self, text: s }
    }
}

/// Text with an applied style
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyledText {
    style: Style,
    text: heapless::String<256>,
}

impl StyledText {
    /// Create new styled text
    pub fn new(style: Style, text: &str) -> Self {
        let mut s = heapless::String::new();
        let _ = s.push_str(text);
        Self { style, text: s }
    }

    /// Get the raw text without styling
    pub fn plain(&self) -> &str {
        self.text.as_str()
    }

    /// Get the style
    pub fn style(&self) -> Style {
        self.style
    }
}

impl fmt::Display for StyledText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.style == Style::new() {
            f.write_str(self.text.as_str())
        } else {
            write!(
                f,
                "{}{}{}",
                self.style.build(),
                self.text.as_str(),
                self.style.reset()
            )
        }
    }
}

impl From<StyledText> for String {
    fn from(st: StyledText) -> String {
        st.to_string()
    }
}

/// Predefined styles for common use cases
pub mod presets {
    use super::*;

    /// Error style (red, bold)
    pub fn error() -> Style {
        Style::new().fg(AnsiColor::Red).bold()
    }

    /// Warning style (yellow, bold)
    pub fn warning() -> Style {
        Style::new().fg(AnsiColor::Yellow).bold()
    }

    /// Success style (green, bold)
    pub fn success() -> Style {
        Style::new().fg(AnsiColor::Green).bold()
    }

    /// Info style (blue, bold)
    pub fn info() -> Style {
        Style::new().fg(AnsiColor::Blue).bold()
    }

    /// Debug style (cyan)
    pub fn debug() -> Style {
        Style::new().fg(AnsiColor::Cyan)
    }

    /// Header style (bold, underlined)
    pub fn header() -> Style {
        Style::new().bold().underline()
    }

    /// Highlight style (reverse)
    pub fn highlight() -> Style {
        Style::new().reverse()
    }

    /// Muted style (dim)
    pub fn muted() -> Style {
        Style::new().dim()
    }

    /// Link style (blue, underline)
    pub fn link() -> Style {
        Style::new().fg(AnsiColor::Blue).underline()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ansi_color_fg() {
        assert_eq!(AnsiColor::Red.fg(), "\x1b[31m");
        assert_eq!(AnsiColor::Green.fg(), "\x1b[32m");
        assert_eq!(AnsiColor::Default.fg(), "\x1b[39m");
    }

    #[test]
    fn test_ansi_color_bg() {
        assert_eq!(AnsiColor::Red.bg(), "\x1b[41m");
        assert_eq!(AnsiColor::Default.bg(), "\x1b[49m");
    }

    #[test]
    fn test_ansi_style() {
        assert_eq!(AnsiStyle::Bold.sequence(), "\x1b[1m");
        assert_eq!(AnsiStyle::Reset.sequence(), "\x1b[0m");
    }

    #[test]
    fn test_style_builder() {
        let style = Style::new()
            .fg(AnsiColor::Red)
            .bg(AnsiColor::White)
            .bold()
            .underline();

        let seq = style.build();
        assert!(seq.contains("31")); // Red fg
        assert!(seq.contains("47")); // White bg
        assert!(seq.contains("1")); // Bold
        assert!(seq.contains("4")); // Underline
    }

    #[test]
    fn test_text() {
        let style = Style::new().fg(AnsiColor::Red).bold();
        let text = style.apply("Hello");
        let result = text.to_string();
        assert!(result.contains("Hello"));
        assert!(result.contains("\x1b[1;31m") || result.contains("\x1b[31;1m"));
        assert!(result.contains("31") || result.contains("1"));
        // reset() generates specific reset codes (e.g., \x1b[22;39m), not necessarily \x1b[0m
        assert!(result.contains("\x1b["));
    }

    #[test]
    fn test_presets() {
        let error = presets::error();
        assert!(error.has_attr(StyleAttr::BOLD));
        assert_eq!(error.fg, Some(AnsiColor::Red));

        let success = presets::success();
        assert!(success.has_attr(StyleAttr::BOLD));
        assert_eq!(success.fg, Some(AnsiColor::Green));
    }

    #[test]
    fn test_style_attr_bitflags() {
        let attrs = StyleAttr::BOLD | StyleAttr::ITALIC | StyleAttr::UNDERLINE;
        assert!(attrs.contains(StyleAttr::BOLD));
        assert!(attrs.contains(StyleAttr::ITALIC));
        assert!(attrs.contains(StyleAttr::UNDERLINE));
        assert!(!attrs.contains(StyleAttr::DIM));

        let mut attrs2 = StyleAttr::BOLD;
        attrs2 |= StyleAttr::ITALIC;
        assert!(attrs2.contains(StyleAttr::BOLD));
        assert!(attrs2.contains(StyleAttr::ITALIC));

        attrs2 &= StyleAttr::BOLD;
        assert!(attrs2.contains(StyleAttr::BOLD));
        assert!(!attrs2.contains(StyleAttr::ITALIC));
    }
}
