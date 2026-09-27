//! Async Logging Framework for Codevar
//!
//! Provides a zero-allocation, platform-aware logging system with support for:
//! - Windows (Win32 API: WriteConsoleW, WriteFile) via consoleutil
//! - Linux/Android (write syscall to stdout/stderr) via consoleutil
//! - macOS/iOS/FreeBSD (write syscall to stdout/stderr) via consoleutil
//! - WASM (console.log via web-sys) via consoleutil
//!
//! Features:
//! - Log levels: debug, info, error, irr (irrecoverable)
//! - Timestamped logging (UTC via codevar-timeutil)
//! - Raw logging (no timestamp, no level)
//! - No heap allocation (streams bytes directly)
//! - UTF-8/Unicode support via codevar-textlike-encode (via consoleutil)
//! - ANSI color/style support via consoleutil
//! - `no_std` compatible

#![allow(dead_code)]

use core::fmt;
use core::sync::atomic::{AtomicU8, Ordering};

use crate::console::{AnsiColor, presets, write_stderr, write_stdout};
use codevar_timeutil::{format_utc_iso8601, utc_now};

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

impl From<crate::console::ConsoleError> for LogError {
    fn from(e: crate::console::ConsoleError) -> Self {
        match e {
            crate::console::ConsoleError::SyscallError(code) => LogError::SyscallError(code),
            crate::console::ConsoleError::IoError(code) => LogError::IoError(code),
            crate::console::ConsoleError::EncodingError => LogError::EncodingError,
            crate::console::ConsoleError::HandleUnavailable => LogError::HandleUnavailable,
            crate::console::ConsoleError::BufferTooSmall => LogError::BufferTooSmall,
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

/// Maximum size of the timestamp buffer (ISO 8601 UTC format: YYYY-MM-DDTHH:MM:SS.sssssssssZ)
const TIMESTAMP_BUFFER_SIZE: usize = 32;

/// Maximum size of a single log line buffer (timestamp + level + message + newline)
const LOG_LINE_BUFFER_SIZE: usize = 512;

/// Writes a log message with timestamp and level
///
/// This function:
/// 1. Gets the current UTC timestamp
/// 2. Formats it as ISO 8601
/// 3. Prepends the log level
/// 4. Appends the message
/// 5. Writes everything to the appropriate console stream (stdout for debug/info, stderr for error/irr)
/// 6. Does not allocate on the heap
pub fn log_with_timestamp(level: LogLevel, message: &str) -> Result<(), LogError> {
    if !is_enabled(level) {
        return Ok(());
    }

    let writer = get_log_writer();
    let mut buffer = [0u8; LOG_LINE_BUFFER_SIZE];
    let mut idx = 0;

    // Write timestamp
    let ts = utc_now();
    let ts_len = format_utc_iso8601(ts, &mut buffer[idx..]).map_err(|_| LogError::BufferTooSmall)?;
    idx += ts_len;

    // Write space separator
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b' ';
    idx += 1;

    // Write level
    let level_str = level.as_str();
    let level_len = level_str.len();
    if idx + level_len >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx..idx + level_len].copy_from_slice(level_str.as_bytes());
    idx += level_len;

    // Write space separator
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b' ';
    idx += 1;

    // Write message
    let msg_bytes = message.as_bytes();
    let remaining = LOG_LINE_BUFFER_SIZE - idx;
    let msg_len = msg_bytes.len().min(remaining.saturating_sub(1)); // Reserve 1 byte for newline
    if msg_len == 0 && !msg_bytes.is_empty() {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx..idx + msg_len].copy_from_slice(&msg_bytes[..msg_len]);
    idx += msg_len;

    // Write newline
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b'\n';
    idx += 1;

    // Write to appropriate stream
    let output_bytes = &buffer[..idx];
    if level >= LogLevel::Error {
        writer.write_stderr(output_bytes)
    } else {
        writer.write_stdout(output_bytes)
    }
}

/// Writes a log message with timestamp, level, and ANSI color styling
///
/// This function adds color to the log level prefix based on the log level:
/// - Debug: Cyan
/// - Info: Green
/// - Error: Red
/// - Irr: Magenta
pub fn log_with_timestamp_styled(level: LogLevel, message: &str) -> Result<(), LogError> {
    if !is_enabled(level) {
        return Ok(());
    }

    let writer = get_log_writer();
    let mut buffer = [0u8; LOG_LINE_BUFFER_SIZE];
    let mut idx = 0;

    // Write timestamp
    let ts = utc_now();
    let ts_len = format_utc_iso8601(ts, &mut buffer[idx..]).map_err(|_| LogError::BufferTooSmall)?;
    idx += ts_len;

    // Write space separator
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b' ';
    idx += 1;

    // Write ANSI color for level
    let level_color = match level {
        LogLevel::Debug => AnsiColor::Cyan,
        LogLevel::Info => AnsiColor::Green,
        LogLevel::Error => AnsiColor::Red,
        LogLevel::Irr => AnsiColor::Magenta,
    };
    let color_seq = level_color.fg();
    let color_bytes = color_seq.as_bytes();
    if idx + color_bytes.len() >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx..idx + color_bytes.len()].copy_from_slice(color_bytes);
    idx += color_bytes.len();

    // Write level
    let level_str = level.as_str();
    let level_len = level_str.len();
    if idx + level_len >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx..idx + level_len].copy_from_slice(level_str.as_bytes());
    idx += level_len;

    // Write ANSI reset
    let reset_seq = "\x1b[0m";
    let reset_bytes = reset_seq.as_bytes();
    if idx + reset_bytes.len() >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx..idx + reset_bytes.len()].copy_from_slice(reset_bytes);
    idx += reset_bytes.len();

    // Write space separator
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b' ';
    idx += 1;

    // Write message
    let msg_bytes = message.as_bytes();
    let remaining = LOG_LINE_BUFFER_SIZE - idx;
    let msg_len = msg_bytes.len().min(remaining.saturating_sub(1));
    if msg_len == 0 && !msg_bytes.is_empty() {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx..idx + msg_len].copy_from_slice(&msg_bytes[..msg_len]);
    idx += msg_len;

    // Write newline
    if idx >= LOG_LINE_BUFFER_SIZE {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b'\n';
    idx += 1;

    // Write to appropriate stream
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

/// Writes a raw string to the console without timestamp or level
#[inline]
pub fn log_raw_str(message: &str) -> Result<(), LogError> {
    log_raw(message.as_bytes())
}

/// Pre-styled logging functions using preset styles

/// Log with error style (red, bold)
pub fn log_error_styled(message: &str) -> Result<(), LogError> {
    let style = presets::error();
    let styled = style.apply(message);
    log_raw_str(&styled.to_string())
}

/// Log with warning style (yellow, bold)
pub fn log_warn_styled(message: &str) -> Result<(), LogError> {
    let style = presets::warning();
    let styled = style.apply(message);
    log_raw_str(&styled.to_string())
}

/// Log with success style (green, bold)
pub fn log_success_styled(message: &str) -> Result<(), LogError> {
    let style = presets::success();
    let styled = style.apply(message);
    log_raw_str(&styled.to_string())
}

/// Log with info style (blue, bold)
pub fn log_info_styled(message: &str) -> Result<(), LogError> {
    let style = presets::info();
    let styled = style.apply(message);
    log_raw_str(&styled.to_string())
}

/// Log with debug style (cyan)
pub fn log_debug_styled(message: &str) -> Result<(), LogError> {
    let style = presets::debug();
    let styled = style.apply(message);
    log_raw_str(&styled.to_string())
}

/// Log with timestamp style (dim, cyan)
pub fn log_timestamp_styled(message: &str) -> Result<(), LogError> {
    let style = presets::timestamp();
    let styled = style.apply(message);
    log_raw_str(&styled.to_string())
}

/// Macro for debug-level logging with timestamp
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Debug) {
            let _ = $crate::log::log_with_timestamp($crate::log::LogLevel::Debug, &format!($($arg)*));
        }
    }};
}

/// Macro for info-level logging with timestamp
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Info) {
            let _ = $crate::log::log_with_timestamp($crate::log::LogLevel::Info, &format!($($arg)*));
        }
    }};
}

/// Macro for error-level logging with timestamp
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Error) {
            let _ = $crate::log::log_with_timestamp($crate::log::LogLevel::Error, &format!($($arg)*));
        }
    }};
}

/// Macro for irrecoverable-level logging with timestamp
#[macro_export]
macro_rules! log_irr {
    ($($arg:tt)*) => {{
        // IRR level is always logged regardless of filter
        let _ = $crate::log::log_with_timestamp($crate::log::LogLevel::Irr, &format!($($arg)*));
    }};
}

/// Macro for raw logging (no timestamp, no level)
#[macro_export]
macro_rules! log_raw {
    ($($arg:tt)*) => {{
        let _ = $crate::log::log_raw_str(&format!($($arg)*));
    }};
}

/// Macro for raw logging with pre-formatted string (avoids format! overhead)
#[macro_export]
macro_rules! log_raw_str {
    ($msg:expr) => {{
        let _ = $crate::log::log_raw_str($msg);
    }};
}

/// Macro for styled debug logging
#[macro_export]
macro_rules! log_debug_styled {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Debug) {
            let _ = $crate::log::log_debug_styled(&format!($($arg)*));
        }
    }};
}

/// Macro for styled info logging
#[macro_export]
macro_rules! log_info_styled {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Info) {
            let _ = $crate::log::log_info_styled(&format!($($arg)*));
        }
    }};
}

/// Macro for styled error logging
#[macro_export]
macro_rules! log_error_styled {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Error) {
            let _ = $crate::log::log_error_styled(&format!($($arg)*));
        }
    }};
}

/// Macro for styled warning logging
#[macro_export]
macro_rules! log_warn_styled {
    ($($arg:tt)*) => {{
        if $crate::log::is_enabled($crate::log::LogLevel::Error) {
            let _ = $crate::log::log_warn_styled(&format!($($arg)*));
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_log_with_timestamp_styled() {
        let result = log_with_timestamp_styled(LogLevel::Debug, "debug message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_with_timestamp_styled(LogLevel::Info, "info message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_with_timestamp_styled(LogLevel::Error, "error message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_with_timestamp_styled(LogLevel::Irr, "irrecoverable message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));
    }

    #[test]
    fn test_styled_logs() {
        let result = log_error_styled("error message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_warn_styled("warning message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_success_styled("success message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_info_styled("info message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));

        let result = log_debug_styled("debug message");
        assert!(result.is_ok() || matches!(result, Err(LogError::HandleUnavailable)));
    }

    #[test]
    fn test_utc_timestamp_format() {
        let ts = utc_now();
        let mut buf = [0u8; TIMESTAMP_BUFFER_SIZE];
        let len = format_utc_iso8601(ts, &mut buf).expect("buffer large enough");
        let s = core::str::from_utf8(&buf[..len]).expect("valid utf-8");
        assert!(len >= 20);
        assert_eq!(s.chars().nth(4), Some('-'));
        assert_eq!(s.chars().nth(7), Some('-'));
        assert_eq!(s.chars().nth(10), Some('T'));
        assert_eq!(s.chars().nth(13), Some(':'));
        assert_eq!(s.chars().nth(16), Some(':'));
        assert_eq!(s.chars().last(), Some('Z'));
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

        assert!(stdout.len() > 0);
        assert!(stderr.len() > 0);
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

        // Restore default
        set_log_writer(&DEFAULT_WRITER);
    }
}
