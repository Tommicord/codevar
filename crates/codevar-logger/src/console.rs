//! Console utilities re-exported from codevar-consoleutil
//!
//! This module provides comprehensive console manipulation utilities including:
//! - ANSI escape sequences and builders
//! - Color and style management (foreground, background, bold, italic, etc.)
//! - Cursor control (position, movement, visibility)
//! - Screen/line clearing operations
//! - Terminal capability detection
//! - Cross-platform console handling (Windows, Unix, WASM)

#![allow(dead_code)]

// Re-export all of consoleutil's public API
pub use codevar_consoleutil::{
    AnsiBuilder,
    // ANSI
    AnsiCode,
    // Style
    AnsiColor,
    AnsiSequence,
    AnsiStyle,
    // Clear
    Clear,
    // Errors
    ConsoleError,
    // Cursor
    Cursor,
    Style,
    StyleAttr,
    StyledText,
    // Terminal
    Terminal,
    TerminalCaps,
    TerminalInfo,
    // Modules for sub-exports
    ansi,
    clear,
    cursor,
    detect_terminal_height,
    detect_terminal_width,
    disable_ansi,
    // ANSI state
    force_ansi,
    // ANSI support
    init_ansi_support,
    is_ansi_disabled,
    is_ansi_forced,
    reset_ansi_state,
    strip_ansi,
    style,
    supports_ansi,
    terminal,
    terminal_height,
    // Terminal size
    terminal_width,
    update_terminal_height,
    update_terminal_width,
    write_stderr,
    // Console I/O
    write_stdout,
};

// Re-export types from submodules
pub use codevar_consoleutil::ansi::sequences;
pub use codevar_consoleutil::clear::{Edit, Scroll, Tab};
pub use codevar_consoleutil::cursor::{Position, Visibility};
pub use codevar_consoleutil::style::presets;
