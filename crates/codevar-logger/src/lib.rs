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

//! Logging to the console utility
//!
//! Provides a zero-allocation, platform-aware logging system with support for:
//! - Windows (Win32 API: WriteConsoleW, WriteFile) via consoleutil
//! - Linux/Android (write syscall to stdout/stderr) via consoleutil
//! - macOS/iOS/FreeBSD (write syscall to stdout/stderr) via consoleutil
//! - WASM (console.log via web-sys) via consoleutil
//!
//! Features:
//! - Log levels: debug, info, error, irr (irrecoverable)
//! - Timestamped logging (UTC via codevar-time-util) in format "[<timestamp>:<level> message]"
//! - Raw logging (no timestamp, no level)
//! - No heap allocation (streams bytes directly)
//! - UTF-8/Unicode support via codevar-textlike-encode (via consoleutil)
//! - ANSI color/style support via consoleutil
//! - Terminal detection for ANSI color support via codevar-io
//! - `no_std` compatible

#![cfg_attr(not(test), no_std)]
extern crate alloc;

use core::fmt;
use core::sync::atomic::{AtomicU8, Ordering};

use codevar_consoleutil::{AnsiColor, AnsiStyle, ConsoleError, write_stderr, write_stdout};
use codevar_io::{IsTerminal, Stderr, Stdout};
use codevar_timeutil::{format_utc_iso8601, utc_now};

/// Global flag to enable/disable ANSI colors (defaults to terminal detection)
static USE_ANSI_COLORS: AtomicU8 = AtomicU8::new(2); // 0=disabled, 1=enabled, 2=auto

/// Sets whether to use ANSI colors in log output
///
/// - `true`: Always use colors
/// - `false`: Never use colors
/// - Default (auto): Detect based on terminal
pub fn set_ansi_colors(enabled: Option<bool>) {
    USE_ANSI_COLORS.store(
        match enabled {
            Some(true) => 1,
            Some(false) => 0,
            None => 2,
        },
        Ordering::Relaxed,
    );
}

/// Checks if ANSI colors should be used for the given stream
fn should_use_ansi(stream_is_terminal: bool) -> bool {
    match USE_ANSI_COLORS.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => stream_is_terminal,
    }
}

/// Log level enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    /// Debug-level messages for detailed diagnostics
    Debug = 0,
    /// Informational messages about normal operation
    Info = 1,
    /// Error conditions that don't halt execution
    Error = 2,
    /// Irrecoverable failures requiring immediate attention
    Irr = 3,
}

impl LogLevel {
    /// Returns the level as a static string slice
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Error => "ERROR",
            LogLevel::Irr => "IRR",
        }
    }
}

/// Global minimum log level filter (atomic for thread-safe runtime changes)
static MIN_LOG_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Debug as u8);

/// Sets the global minimum log level
#[inline]
pub fn set_min_log_level(level: LogLevel) {
    MIN_LOG_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Gets the current global minimum log level
#[inline]
pub fn min_log_level() -> LogLevel {
    match MIN_LOG_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Debug,
        1 => LogLevel::Info,
        2 => LogLevel::Error,
        _ => LogLevel::Irr,
    }
}

/// Checks if a log level is enabled
#[inline]
pub fn is_enabled(level: LogLevel) -> bool {
    level as u8 >= MIN_LOG_LEVEL.load(Ordering::Relaxed)
}

/// Trait for platform-specific console output
pub trait LogWriter: Sync {
    /// Writes bytes to the console/stdout
    fn write_stdout(&self, bytes: &[u8]) -> Result<(), LogError>;
    /// Writes bytes to stderr
    fn write_stderr(&self, bytes: &[u8]) -> Result<(), LogError>;
    /// Flushes any buffered output
    fn flush(&self) -> Result<(), LogError>;
}

/// Error type for logging operations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogError {
    /// The underlying system call failed
    SyscallError(i32),
    /// The buffer was too small for the formatted output
    BufferTooSmall,
    /// UTF-8 encoding error
    EncodingError,
    /// Console handle not available (e.g., detached from terminal)
    HandleUnavailable,
    /// I/O error
    IoError(i32),
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogError::SyscallError(code) => write!(f, "syscall error: {}", code),
            LogError::BufferTooSmall => write!(f, "buffer too small"),
            LogError::EncodingError => write!(f, "encoding error"),
            LogError::HandleUnavailable => write!(f, "console handle unavailable"),
            LogError::IoError(code) => write!(f, "I/O error: {}", code),
        }
    }
}

impl core::error::Error for LogError {}

impl From<ConsoleError> for LogError {
    fn from(e: ConsoleError) -> Self {
        match e {
            ConsoleError::SyscallError(code) => LogError::SyscallError(code),
            ConsoleError::IoError(code) => LogError::IoError(code),
            ConsoleError::EncodingError => LogError::EncodingError,
            ConsoleError::HandleUnavailable => LogError::HandleUnavailable,
            ConsoleError::BufferTooSmall => LogError::BufferTooSmall,
        }
    }
}

/// Default console writer implementation using consoleutil
pub struct DefaultLogWriter;

impl DefaultLogWriter {
    /// Creates a new default console writer
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for DefaultLogWriter {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl LogWriter for DefaultLogWriter {
    #[inline]
    fn write_stdout(&self, bytes: &[u8]) -> Result<(), LogError> {
        write_stdout(bytes).map_err(Into::into)
    }

    #[inline]
    fn write_stderr(&self, bytes: &[u8]) -> Result<(), LogError> {
        write_stderr(bytes).map_err(Into::into)
    }

    #[inline]
    fn flush(&self) -> Result<(), LogError> {
        Ok(())
    }
}

/// Global console writer instance (can be swapped for testing or custom outputs)
static LOG_WRITER: spin::Mutex<Option<&'static dyn LogWriter>> = spin::Mutex::new(None);

/// Sets a custom global console writer
pub fn set_log_writer(writer: &'static dyn LogWriter) {
    *LOG_WRITER.lock() = Some(writer);
}

/// Gets the global console writer (falls back to default)
fn get_log_writer() -> &'static dyn LogWriter {
    LOG_WRITER
        .lock()
        .as_ref()
        .copied()
        .unwrap_or(&DEFAULT_WRITER)
}

static DEFAULT_WRITER: DefaultLogWriter = DefaultLogWriter::new();

/// Maximum size of a single log line buffer
const LOG_LINE_BUFFER_SIZE: usize = 512;

/// Writes the timestamp portion of a log line: "[<timestamp>:"
///
/// Appends the UTC timestamp in ISO8601 format followed by a colon.
/// Returns the number of bytes written.
#[inline]
fn format_timestamp(buffer: &mut [u8], offset: usize) -> Result<usize, LogError> {
    let ts = utc_now();
    let ts_len = format_utc_iso8601(ts, &mut buffer[offset..]).map_err(|_| LogError::BufferTooSmall)?;
    if offset + ts_len >= buffer.len() {
        return Err(LogError::BufferTooSmall);
    }
    Ok(ts_len)
}

/// Writes the log level portion: "<LEVEL> "
///
/// Appends the level followed by a space. Color escape codes for the
/// timestamp are handled by the caller.
/// Returns the number of bytes written.
#[inline]
fn format_level(buffer: &mut [u8], offset: usize, level: LogLevel) -> Result<usize, LogError> {
    let level_str = level.as_str();
    let level_len = level_str.len();
    let required = level_len + 1; // +1 for space
    if offset + required >= buffer.len() {
        return Err(LogError::BufferTooSmall);
    }
    let mut idx = offset;
    buffer[idx..idx + level_len].copy_from_slice(level_str.as_bytes());
    idx += level_len;
    buffer[idx] = b' ';
    idx += 1;
    Ok(idx - offset)
}

/// Writes the message portion: "message\n"
///
/// Appends the message followed by a newline.
/// Returns the number of bytes written.
#[inline]
fn format_message(buffer: &mut [u8], offset: usize, message: &str) -> Result<usize, LogError> {
    let msg_bytes = message.as_bytes();
    let remaining = buffer.len() - offset;
    let msg_len = msg_bytes.len().min(remaining.saturating_sub(1)); // -1 for newline
    if msg_len == 0 && !msg_bytes.is_empty() {
        return Err(LogError::BufferTooSmall);
    }
    buffer[offset..offset + msg_len].copy_from_slice(&msg_bytes[..msg_len]);
    let idx = offset + msg_len;
    buffer[idx] = b'\n';
    Ok(msg_len + 1)
}

/// Writes a log message with timestamp and level in format "[<timestamp>]:<level> message"
///
/// Messages at ERROR and IRR levels are written to stderr; others to stdout.
/// ANSI colors are automatically enabled only when the output stream is a terminal.
/// Use `set_ansi_colors(Some(true))` to force colors or `set_ansi_colors(Some(false))` to disable.
pub fn log_with_timestamp(level: LogLevel, message: &str) -> Result<(), LogError> {
    if !is_enabled(level) {
        return Ok(());
    }
    let writer = get_log_writer();
    let mut buffer = [0u8; LOG_LINE_BUFFER_SIZE];
    let mut idx = 0;
    let use_color = if level >= LogLevel::Error {
        should_use_ansi(Stderr::new().is_terminal())
    } else {
        should_use_ansi(Stdout::new().is_terminal())
    };
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    if use_color {
        let color_seq = AnsiColor::Green.fg();
        let color_bytes = color_seq.as_bytes();
        let reset_seq = AnsiStyle::Reset.sequence();
        let reset_bytes = reset_seq.as_bytes();
        let required = color_bytes.len() + reset_bytes.len() + 1;
        if idx + required >= buffer.len() {
            return Err(LogError::BufferTooSmall);
        }
        buffer[idx..idx + color_bytes.len()].copy_from_slice(color_bytes);
        idx += color_bytes.len();
        buffer[idx] = b'[';
        idx += 1;
        let ts_written = format_timestamp(&mut buffer, idx)?;
        idx += ts_written;
        buffer[idx] = b']';
        idx += 1;
        buffer[idx] = b':';
        idx += 1;
        let level_written = format_level(&mut buffer, idx, level)?;
        idx += level_written;
        let msg_written = format_message(&mut buffer, idx, message)?;
        idx += msg_written;
        buffer[idx..idx + reset_bytes.len()].copy_from_slice(reset_bytes);
        idx += reset_bytes.len();
    } else {
        buffer[idx] = b'[';
        idx += 1;
        let ts_written = format_timestamp(&mut buffer, idx)?;
        idx += ts_written;
        buffer[idx] = b']';
        idx += 1;
        buffer[idx] = b':';
        idx += 1;
        let level_written = format_level(&mut buffer, idx, level)?;
        idx += level_written;
        let msg_written = format_message(&mut buffer, idx, message)?;
        idx += msg_written;
    }
    let output_bytes = &buffer[..idx];
    if level >= LogLevel::Error {
        writer.write_stderr(output_bytes)
    } else {
        writer.write_stdout(output_bytes)
    }
}

/// Writes raw bytes to the console without timestamp or level
pub fn log_raw(bytes: &[u8]) -> Result<(), LogError> {
    let writer = get_log_writer();
    writer.write_stdout(bytes)
}

#[doc(hidden)]
pub mod logger {
    pub extern crate alloc;
    use crate::{LogError, LogLevel, log_with_timestamp};
    use alloc::string::ToString;
    use codevar_consoleutil::console_style::presets;

    /// Log with error style
    pub fn log_error(message: &str) -> Result<(), LogError> {
        let style = presets::error();
        let styled = style.apply(message);
        log_with_timestamp(LogLevel::Error, &styled.to_string())
    }

    /// Log with warning style
    pub fn log_warn(message: &str) -> Result<(), LogError> {
        let style = presets::warning();
        let styled = style.apply(message);
        log_with_timestamp(LogLevel::Error, &styled.to_string())
    }

    /// Log with irr style
    pub fn log_irr(message: &str) -> Result<(), LogError> {
        let style = presets::highlight();
        let styled = style.apply(message);
        log_with_timestamp(LogLevel::Error, &styled.to_string())
    }

    /// Log with success style
    pub fn log_success(message: &str) -> Result<(), LogError> {
        let style = presets::success();
        let styled = style.apply(message);
        log_with_timestamp(LogLevel::Info, &styled.to_string())
    }

    /// Log with info style
    pub fn log_info(message: &str) -> Result<(), LogError> {
        let style = presets::info();
        let styled = style.apply(message);
        log_with_timestamp(LogLevel::Info, &styled.to_string())
    }

    /// Log with debug style (cyan)
    pub fn log_debug(message: &str) -> Result<(), LogError> {
        let style = presets::debug();
        let styled = style.apply(message);
        log_with_timestamp(LogLevel::Debug, &styled.to_string())
    }
}

/// Macro for styled debug logging
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Debug) {
            let _ = $crate::logger::log_debug(&$crate::logger::alloc::format!($($arg)*));
        }
    }};
}

/// Macro for styled info logging
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Info) {
            let _ = $crate::logger::log_info(&$crate::logger::alloc::format!($($arg)*));
        }
    }};
}

/// Macro for styled info logging
#[macro_export]
macro_rules! log_success {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Info) {
            let _ = $crate::logger::log_success(&$crate::logger::alloc::format!($($arg)*));
        }
    }};
}

/// Macro for styled error logging
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Error) {
            let _ = $crate::logger::log_error(&$crate::logger::alloc::format!($($arg)*));
        }
    }};
}

/// Macro for styled warning logging
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Error) {
            let _ = $crate::logger::log_warn(&$crate::logger::alloc::format!($($arg)*));
        }
    }};
}

/// Macro for irrecoverable-level logging with timestamp
#[macro_export]
macro_rules! log_irr {
    ($($arg:tt)*) => {{
        let _ = $crate::logger::log_irr(&$crate::logger::alloc::format!($($arg)*));
    }};
}

#[cfg(test)]
mod tests {
    use crate::{
        DEFAULT_WRITER, LogError, LogLevel, LogWriter, is_enabled, log_raw, log_with_timestamp,
        set_log_writer, set_min_log_level,
    };

    #[test]
    fn test_log_levels() {
        assert_eq!(LogLevel::Debug as u8, 0);
        assert_eq!(LogLevel::Info as u8, 1);
        assert_eq!(LogLevel::Error as u8, 2);
        assert_eq!(LogLevel::Irr as u8, 3);
    }

    #[test]
    fn test_log_level_strings() {
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Irr.as_str(), "IRR");
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Irr);
    }

    #[test]
    fn test_min_log_level() {
        set_min_log_level(LogLevel::Info);
        assert!(!is_enabled(LogLevel::Debug));
        assert!(is_enabled(LogLevel::Info));
        assert!(is_enabled(LogLevel::Error));
        assert!(is_enabled(LogLevel::Irr));

        set_min_log_level(LogLevel::Error);
        assert!(!is_enabled(LogLevel::Debug));
        assert!(!is_enabled(LogLevel::Info));
        assert!(is_enabled(LogLevel::Error));
        assert!(is_enabled(LogLevel::Irr));

        // Reset for other tests
        set_min_log_level(LogLevel::Debug);
    }

    #[test]
    fn test_log_raw() {
        let result = log_raw(b"test raw message\n");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));
    }

    #[test]
    fn test_log_with_timestamp() {
        let result = log_with_timestamp(LogLevel::Debug, "debug message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_with_timestamp(LogLevel::Info, "info message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_with_timestamp(LogLevel::Error, "error message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_with_timestamp(LogLevel::Irr, "irrecoverable message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));
    }

    #[test]
    fn test_logs() {
        let result = crate::logger::log_error("error message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = crate::logger::log_warn("warning message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = crate::logger::log_success("success message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = crate::logger::log_info("info message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = crate::logger::log_debug("debug message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));
    }

    #[test]
    fn test_custom_log_writer() {
        struct TestWriter {
            stdout: spin::Mutex<alloc::vec::Vec<u8>>,
            stderr: spin::Mutex<alloc::vec::Vec<u8>>,
        }

        impl LogWriter for TestWriter {
            fn write_stdout(&self, bytes: &[u8]) -> Result<(), LogError> {
                self.stdout.lock().extend_from_slice(bytes);
                Ok(())
            }
            fn write_stderr(&self, bytes: &[u8]) -> Result<(), LogError> {
                self.stderr.lock().extend_from_slice(bytes);
                Ok(())
            }
            fn flush(&self) -> Result<(), LogError> {
                Ok(())
            }
        }

        static TEST_WRITER: TestWriter = TestWriter {
            stdout: spin::Mutex::new(alloc::vec::Vec::new()),
            stderr: spin::Mutex::new(alloc::vec::Vec::new()),
        };

        set_log_writer(&TEST_WRITER);

        log_with_timestamp(LogLevel::Info, "test message").unwrap();
        log_with_timestamp(LogLevel::Error, "error message").unwrap();

        let stdout = TEST_WRITER.stdout.lock();
        let stderr = TEST_WRITER.stderr.lock();

        assert!(!stdout.is_empty());
        assert!(!stderr.is_empty());
        assert!(
            core::str::from_utf8(&stdout)
                .unwrap()
                .contains("test message")
        );
        assert!(
            core::str::from_utf8(&stderr)
                .unwrap()
                .contains("error message")
        );
        set_log_writer(&DEFAULT_WRITER);
    }
}
