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
//! - No heap allocation: each line is formatted in place into a fixed circular
//!   staging buffer of 4 entries, then streamed entry-by-entry to the console
//! - Deferred flushing via [`drain`]; logging otherwise stages a line and then
//!   immediately streams every staged entry out (oldest first)
//! - Overflow safety: when the staging buffer is full the oldest entry is
//!   dropped, so memory stays bounded even if the console is unavailable
//! - UTF-8/Unicode support via codevar-textlike-encode (via consoleutil)
//! - ANSI color/style support via consoleutil
//! - Terminal detection for ANSI color support via codevar-io
//! - `no_std` compatible

#![cfg_attr(not(test), no_std)]
#[cfg(test)]
extern crate alloc;

use core::fmt;
use core::fmt::Write as _;
use core::sync::atomic::{AtomicU8, Ordering};

use codevar_console_util::{ConsoleError, write_stderr, write_stdout};
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

/// Writes the timestamp portion of a log line
///
/// Appends only the UTC timestamp in ISO8601 format; the caller surrounds it
/// with `[` and `]:`.
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

/// Number of formatted log lines the circular staging buffer can hold.
///
/// Four entries keep enough recent context to diagnose a failure after the
/// console stopped accepting writes while costing only a few KiB of `.bss`.
const RING_CAPACITY: usize = 4;

/// ANSI sequence opening the green color used for a whole log line.
///
/// Matches [`codevar_console_util::AnsiColor::Green`]`::fg()`; a unit test
/// asserts the two stay identical.
const LINE_COLOR_OPEN: &str = "\x1b[32m";

/// ANSI sequence closing every style on a log line.
///
/// Matches [`codevar_console_util::AnsiStyle::Reset`]`::sequence()`; a unit test
/// asserts the two stay identical.
const LINE_RESET: &str = "\x1b[0m";

/// SGR open/close pair wrapping the body of a styled log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StyleCodes {
    /// Sequence written before the message body.
    open: &'static str,
    /// Sequence written after the message body.
    close: &'static str,
}

/// Preset style applied around the body of a [`log_styled`] message.
///
/// The sequences are byte-for-byte the ones produced by the
/// [`codevar_console_util::style::presets`] styles; unit tests keep them in
/// sync so the logger never needs a heap allocation to render them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Red bold, matching `presets::error()`.
    Error,
    /// Yellow bold, matching `presets::warning()`.
    Warning,
    /// Green bold, matching `presets::success()`.
    Success,
    /// Blue bold, matching `presets::info()`.
    Info,
    /// Cyan, matching `presets::debug()`.
    Debug,
    /// Reverse video, matching `presets::highlight()`.
    Highlight,
}

/// Returns the SGR pair for `preset`.
const fn style_codes(preset: Preset) -> StyleCodes {
    match preset {
        Preset::Error => StyleCodes {
            open: "\x1b[1;91m",
            close: "\x1b[22;39m",
        },
        Preset::Warning => StyleCodes {
            open: "\x1b[1;93m",
            close: "\x1b[22;39m",
        },
        Preset::Success => StyleCodes {
            open: "\x1b[1;92m",
            close: "\x1b[22;39m",
        },
        Preset::Info => StyleCodes {
            open: "\x1b[1;94m",
            close: "\x1b[22;39m",
        },
        Preset::Debug => StyleCodes {
            open: "\x1b[96m",
            close: "\x1b[39m",
        },
        Preset::Highlight => StyleCodes {
            open: "\x1b[7m",
            close: "\x1b[27m",
        },
    }
}

/// One staged log line: fully formatted bytes plus the level that decides
/// which console stream receives it when the entry is drained.
struct RingEntry {
    /// Level the line was logged at, used for stdout/stderr routing.
    level: LogLevel,
    /// Number of valid bytes at the start of `bytes`.
    len: usize,
    /// Formatted line bytes, `[<timestamp>]:<LEVEL> <body>\n`.
    bytes: [u8; LOG_LINE_BUFFER_SIZE],
}

impl RingEntry {
    /// An empty slot for array initialization of the ring.
    const EMPTY: Self = Self {
        level: LogLevel::Debug,
        len: 0,
        bytes: [0; LOG_LINE_BUFFER_SIZE],
    };

    /// Returns the formatted bytes of this entry.
    #[inline]
    fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// Fixed-size circular staging buffer for formatted log lines.
///
/// Entries are appended at `(head + len) % RING_CAPACITY` and streamed out of
/// `head` first. When all four slots are occupied a new line overwrites the
/// oldest one (`head` advances), so the newest diagnostics always survive a
/// period during which the console rejected writes.
struct LogRing {
    /// Slot storage; only `entries[(head + i) % RING_CAPACITY]` for
    /// `i < len` hold valid entries.
    entries: [RingEntry; RING_CAPACITY],
    /// Index of the oldest staged entry.
    head: usize,
    /// Number of staged entries, `0..=RING_CAPACITY`.
    len: usize,
}

impl LogRing {
    /// Creates an empty ring.
    const fn new() -> Self {
        Self {
            entries: [const { RingEntry::EMPTY }; RING_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    /// Formats `args` into the next slot and links the slot into the ring.
    ///
    /// The slot is only committed after formatting succeeded, so an error
    /// leaves the ring exactly as it was.
    fn stage(
        &mut self,
        level: LogLevel,
        use_color: bool,
        style: Option<StyleCodes>,
        args: fmt::Arguments<'_>,
    ) -> Result<(), LogError> {
        let idx = if self.len == RING_CAPACITY {
            self.head
        } else {
            (self.head + self.len) % RING_CAPACITY
        };
        let staged_len = write_line(&mut self.entries[idx].bytes, level, use_color, style, args)?;
        self.entries[idx].level = level;
        self.entries[idx].len = staged_len;
        if self.len == RING_CAPACITY {
            self.head = (self.head + 1) % RING_CAPACITY;
        } else {
            self.len += 1;
        }
        Ok(())
    }
}

/// The global circular staging buffer.
static RING: spin::Mutex<LogRing> = spin::Mutex::new(LogRing::new());

/// Streams every staged entry to the console, oldest first.
///
/// Each entry is handed to the writer as a single zero-copy slice read
/// straight out of its ring slot: entries are never concatenated into a
/// larger output buffer, so memory use stays flat regardless of how many
/// lines are pending. Entries are popped only after their write succeeded;
/// the first failure stops the drain and leaves that entry (and every entry
/// after it) staged for a later retry.
///
/// The ring lock is held for the whole drain, so `LogWriter` implementations
/// must not log from inside their write methods.
fn stream_to_console(ring: &mut LogRing, writer: &dyn LogWriter) -> Result<(), LogError> {
    let mut streamed = 0usize;
    let mut result = Ok(());
    while streamed < ring.len {
        let idx = (ring.head + streamed) % RING_CAPACITY;
        let entry = &ring.entries[idx];
        let write_result = if entry.level >= LogLevel::Error {
            writer.write_stderr(entry.bytes())
        } else {
            writer.write_stdout(entry.bytes())
        };
        match write_result {
            Ok(()) => streamed += 1,
            Err(error) => {
                result = Err(error);
                break;
            }
        }
    }
    if streamed > 0 {
        ring.head = (ring.head + streamed) % RING_CAPACITY;
        ring.len -= streamed;
    }
    result
}

/// Formats one complete log line into `buffer` and returns its length.
///
/// Layout: `[<timestamp>]:<LEVEL> <body>\n`, optionally wrapped in the green
/// line color (`\x1b[32m…\x1b[0m`) and with `style`'s SGR pair around the
/// body. The body is streamed straight from `args` into `buffer` through a
/// truncating sink: content that does not fit is dropped, and only an
/// impossibly small buffer yields [`LogError::BufferTooSmall`].
fn write_line(
    buffer: &mut [u8],
    level: LogLevel,
    use_color: bool,
    style: Option<StyleCodes>,
    args: fmt::Arguments<'_>,
) -> Result<usize, LogError> {
    let mut idx = 0;
    if use_color {
        if idx + LINE_COLOR_OPEN.len() > buffer.len() {
            return Err(LogError::BufferTooSmall);
        }
        buffer[idx..idx + LINE_COLOR_OPEN.len()].copy_from_slice(LINE_COLOR_OPEN.as_bytes());
        idx += LINE_COLOR_OPEN.len();
    }
    if idx >= buffer.len() {
        return Err(LogError::BufferTooSmall);
    }
    buffer[idx] = b'[';
    idx += 1;
    idx += format_timestamp(buffer, idx)?;
    for byte in [b']', b':'] {
        if idx >= buffer.len() {
            return Err(LogError::BufferTooSmall);
        }
        buffer[idx] = byte;
        idx += 1;
    }
    idx += format_level(buffer, idx, level)?;
    let tail = 1 + if use_color { LINE_RESET.len() } else { 0 };
    if idx + tail > buffer.len() {
        return Err(LogError::BufferTooSmall);
    }
    let body_end = buffer.len() - tail;
    idx = {
        let mut sink = TruncSink::new(&mut buffer[idx..body_end]);
        if let Some(codes) = style {
            let _ = sink.write_str(codes.open);
        }
        let _ = fmt::write(&mut sink, args);
        if let Some(codes) = style {
            let _ = sink.write_str(codes.close);
        }
        idx + sink.len
    };
    buffer[idx] = b'\n';
    idx += 1;
    if use_color {
        buffer[idx..idx + LINE_RESET.len()].copy_from_slice(LINE_RESET.as_bytes());
        idx += LINE_RESET.len();
    }
    Ok(idx)
}

/// Stages one log line into the circular buffer, then streams the ring out.
///
/// Shared implementation behind [`log_with_timestamp`] and [`log_styled`]:
/// the line is formatted in place into the next ring slot (never through a
/// heap `String`), after which every staged entry is streamed to the console.
fn log_line(level: LogLevel, style: Option<StyleCodes>, args: fmt::Arguments<'_>) -> Result<(), LogError> {
    if !is_enabled(level) {
        return Ok(());
    }
    let use_color = if level >= LogLevel::Error {
        should_use_ansi(Stderr::new().is_terminal())
    } else {
        should_use_ansi(Stdout::new().is_terminal())
    };
    let writer = get_log_writer();
    let mut ring = RING.lock();
    ring.stage(level, use_color, style, args)?;
    stream_to_console(&mut ring, writer)
}

/// Writes a log message with timestamp and level in format "[<timestamp>]:<level> message"
///
/// The line is staged in the circular buffer and then streamed to the
/// console; messages at ERROR and IRR levels are written to stderr, others
/// to stdout. ANSI colors are automatically enabled only when the output
/// stream is a terminal.
/// Use `set_ansi_colors(Some(true))` to force colors or `set_ansi_colors(Some(false))` to disable.
pub fn log_with_timestamp(level: LogLevel, message: &str) -> Result<(), LogError> {
    log_line(level, None, format_args!("{}", message))
}

/// Formats `args` as a log line whose body is wrapped in `preset`'s style.
///
/// This is the allocation-free entry point behind the `log_error!` and
/// friends macros: formatting goes directly into the circular staging buffer
/// and is then streamed to the console.
pub fn log_styled(level: LogLevel, preset: Preset, args: fmt::Arguments<'_>) -> Result<(), LogError> {
    log_line(level, Some(style_codes(preset)), args)
}

/// Streams any staged log lines to the console, oldest first.
///
/// Logging already streams immediately after staging each line, so this only
/// matters when earlier console writes failed (their entries stayed staged
/// for retry) or when a caller wants an explicit flush point.
pub fn drain() -> Result<(), LogError> {
    let writer = get_log_writer();
    stream_to_console(&mut RING.lock(), writer)
}

/// Writes raw bytes to the console without timestamp or level
pub fn log_raw(bytes: &[u8]) -> Result<(), LogError> {
    let writer = get_log_writer();
    writer.write_stdout(bytes)
}

/// `core::fmt::Write` adapter that collects formatted text into a
/// caller-provided buffer.
///
/// Bytes that would overflow the buffer are dropped, so the sink never fails
/// and never allocates. Used both to stage log-line bodies and to collect a
/// panic report; a truncation can split a multi-byte UTF-8 scalar, which
/// [`report_panic`] handles by falling back to a static message.
struct TruncSink<'a> {
    /// Destination buffer, borrowed for the duration of the format call.
    buf: &'a mut [u8],
    /// Number of bytes written so far, always `<= buf.len()`.
    len: usize,
}

impl<'a> TruncSink<'a> {
    /// Creates a sink writing into `buf`.
    #[inline]
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, len: 0 }
    }
}

impl core::fmt::Write for TruncSink<'_> {
    #[inline]
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let bytes = text.as_bytes();
        let available = self.buf.len().saturating_sub(self.len);
        let written = bytes.len().min(available);
        self.buf[self.len..self.len + written].copy_from_slice(&bytes[..written]);
        self.len += written;
        Ok(())
    }
}

/// Reports a panic to stderr without allocating.
///
/// The report is emitted at the [`LogLevel::Irr`] level, which always routes
/// to stderr, and is formatted into a fixed stack buffer so the caller (a
/// `#[panic_handler]`) stays safe even when the heap is unavailable or
/// corrupted. Truncation is preferred over a nested failure while panicking.
pub fn report_panic(info: &core::panic::PanicInfo<'_>) {
    let mut buffer = [0u8; LOG_LINE_BUFFER_SIZE];
    let len = {
        let mut sink = TruncSink::new(&mut buffer);
        if fmt::write(&mut sink, format_args!("{info}")).is_err() {
            sink.len = 0;
        }
        sink.len
    };
    let message = core::str::from_utf8(&buffer[..len]).unwrap_or("unformattable panic report");
    let _ = log_with_timestamp(LogLevel::Irr, message);
}

#[doc(hidden)]
pub mod logger {
    use crate::{LogError, LogLevel, Preset, log_styled};

    /// Log with error style
    pub fn log_error(message: &str) -> Result<(), LogError> {
        log_styled(LogLevel::Error, Preset::Error, format_args!("{message}"))
    }

    /// Log with warning style
    pub fn log_warn(message: &str) -> Result<(), LogError> {
        log_styled(LogLevel::Error, Preset::Warning, format_args!("{message}"))
    }

    /// Log with irr style
    pub fn log_irr(message: &str) -> Result<(), LogError> {
        log_styled(LogLevel::Error, Preset::Highlight, format_args!("{message}"))
    }

    /// Log with success style
    pub fn log_success(message: &str) -> Result<(), LogError> {
        log_styled(LogLevel::Info, Preset::Success, format_args!("{message}"))
    }

    /// Log with info style
    pub fn log_info(message: &str) -> Result<(), LogError> {
        log_styled(LogLevel::Info, Preset::Info, format_args!("{message}"))
    }

    /// Log with debug style (cyan)
    pub fn log_debug(message: &str) -> Result<(), LogError> {
        log_styled(LogLevel::Debug, Preset::Debug, format_args!("{message}"))
    }
}

/// Macro for styled debug logging
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Debug) {
            let _ = $crate::log_styled(
                $crate::LogLevel::Debug,
                $crate::Preset::Debug,
                format_args!($($arg)*),
            );
        }
    }};
}

/// Macro for styled info logging
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Info) {
            let _ = $crate::log_styled(
                $crate::LogLevel::Info,
                $crate::Preset::Info,
                format_args!($($arg)*),
            );
        }
    }};
}

/// Macro for styled info logging
#[macro_export]
macro_rules! log_success {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Info) {
            let _ = $crate::log_styled(
                $crate::LogLevel::Info,
                $crate::Preset::Success,
                format_args!($($arg)*),
            );
        }
    }};
}

/// Macro for styled error logging
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Error) {
            let _ = $crate::log_styled(
                $crate::LogLevel::Error,
                $crate::Preset::Error,
                format_args!($($arg)*),
            );
        }
    }};
}

/// Macro for styled warning logging
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        if $crate::is_enabled($crate::LogLevel::Error) {
            let _ = $crate::log_styled(
                $crate::LogLevel::Error,
                $crate::Preset::Warning,
                format_args!($($arg)*),
            );
        }
    }};
}

/// Macro for irrecoverable-level logging with timestamp
#[macro_export]
macro_rules! log_irr {
    ($($arg:tt)*) => {{
        let _ = $crate::log_styled(
            $crate::LogLevel::Error,
            $crate::Preset::Highlight,
            format_args!($($arg)*),
        );
    }};
}

#[cfg(test)]
mod tests {
    use codevar_console_util::{AnsiColor, AnsiStyle, style::presets};

    use crate::{
        DEFAULT_WRITER, LINE_COLOR_OPEN, LINE_RESET, LogError, LogLevel, LogRing, LogWriter, Preset,
        RING_CAPACITY, is_enabled, log_raw, log_styled, log_with_timestamp, set_log_writer,
        set_min_log_level, stream_to_console, style_codes,
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
        log_styled(LogLevel::Error, Preset::Error, format_args!("styled-{}", 7)).unwrap();
        log_error!("macro-{}", 9);

        let stdout = TEST_WRITER.stdout.lock();
        let stderr = TEST_WRITER.stderr.lock();

        assert!(!stdout.is_empty());
        assert!(!stderr.is_empty());
        assert!(
            core::str::from_utf8(&stdout)
                .unwrap()
                .contains("test message")
        );
        let stderr_text = core::str::from_utf8(&stderr).unwrap();
        assert!(stderr_text.contains("error message"));
        assert!(stderr_text.contains("\x1b[1;91mstyled-7\x1b[22;39m"));
        assert!(stderr_text.contains("\x1b[1;91mmacro-9\x1b[22;39m"));
        set_log_writer(&DEFAULT_WRITER);
    }

    /// Console writer capturing both streams for ring-level tests.
    struct RingWriter {
        stdout: spin::Mutex<Vec<u8>>,
        stderr: spin::Mutex<Vec<u8>>,
    }

    impl RingWriter {
        const fn new() -> Self {
            Self {
                stdout: spin::Mutex::new(Vec::new()),
                stderr: spin::Mutex::new(Vec::new()),
            }
        }
    }

    impl LogWriter for RingWriter {
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

    /// Writer whose console is unavailable.
    struct OfflineWriter;

    impl LogWriter for OfflineWriter {
        fn write_stdout(&self, _: &[u8]) -> Result<(), LogError> {
            Err(LogError::HandleUnavailable)
        }
        fn write_stderr(&self, _: &[u8]) -> Result<(), LogError> {
            Err(LogError::HandleUnavailable)
        }
        fn flush(&self) -> Result<(), LogError> {
            Ok(())
        }
    }

    #[test]
    fn preset_styles_match_console_util() {
        assert_eq!(style_codes(Preset::Error).open, presets::error().build());
        assert_eq!(style_codes(Preset::Error).close, presets::error().reset());
        assert_eq!(style_codes(Preset::Warning).open, presets::warning().build());
        assert_eq!(style_codes(Preset::Warning).close, presets::warning().reset());
        assert_eq!(style_codes(Preset::Success).open, presets::success().build());
        assert_eq!(style_codes(Preset::Success).close, presets::success().reset());
        assert_eq!(style_codes(Preset::Info).open, presets::info().build());
        assert_eq!(style_codes(Preset::Info).close, presets::info().reset());
        assert_eq!(style_codes(Preset::Debug).open, presets::debug().build());
        assert_eq!(style_codes(Preset::Debug).close, presets::debug().reset());
        assert_eq!(style_codes(Preset::Highlight).open, presets::highlight().build());
        assert_eq!(style_codes(Preset::Highlight).close, presets::highlight().reset());
        assert_eq!(LINE_COLOR_OPEN, AnsiColor::Green.fg());
        assert_eq!(LINE_RESET, AnsiStyle::Reset.sequence());
    }

    #[test]
    fn ring_overwrites_oldest_and_streams_fifo() {
        let mut ring = LogRing::new();
        for marker in 0..(RING_CAPACITY + 2) {
            ring.stage(LogLevel::Info, false, None, format_args!("marker-{marker}"))
                .expect("512-byte slot always fits a short marker");
        }
        assert_eq!(ring.len, RING_CAPACITY);

        let writer = RingWriter::new();
        stream_to_console(&mut ring, &writer).expect("recording writer never fails");
        assert_eq!(ring.len, 0);

        let stdout = writer.stdout.lock();
        let output = core::str::from_utf8(&stdout).expect("lines are ASCII");
        assert!(!output.contains("marker-0"), "oldest entry must be evicted");
        assert!(!output.contains("marker-1"), "second-oldest must be evicted");
        let positions: Vec<usize> = (2..(RING_CAPACITY + 2))
            .map(|marker| {
                output
                    .find(&format!("marker-{marker}"))
                    .expect("newest entries must survive")
            })
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(positions, sorted, "entries must stream oldest-first");
    }

    #[test]
    fn failed_console_write_keeps_entries_staged_for_retry() {
        let mut ring = LogRing::new();
        ring.stage(LogLevel::Error, false, None, format_args!("retry-me"))
            .expect("512-byte slot always fits a short message");
        let error = stream_to_console(&mut ring, &OfflineWriter)
            .expect_err("offline console must surface the write error");
        assert_eq!(error, LogError::HandleUnavailable);
        assert_eq!(ring.len, 1, "failed entry stays staged for a later drain");

        let writer = RingWriter::new();
        stream_to_console(&mut ring, &writer).expect("recording writer never fails");
        assert_eq!(ring.len, 0);
        let stderr = writer.stderr.lock();
        assert!(
            core::str::from_utf8(&stderr)
                .expect("lines are ASCII")
                .contains("retry-me")
        );
    }

    #[test]
    fn staged_entries_route_by_level() {
        let mut ring = LogRing::new();
        ring.stage(LogLevel::Info, false, None, format_args!("info-side"))
            .expect("512-byte slot always fits a short message");
        ring.stage(LogLevel::Error, false, None, format_args!("error-side"))
            .expect("512-byte slot always fits a short message");

        let writer = RingWriter::new();
        stream_to_console(&mut ring, &writer).expect("recording writer never fails");

        let stdout = writer.stdout.lock();
        let stdout_text = core::str::from_utf8(&stdout).expect("lines are ASCII");
        assert!(stdout_text.contains("info-side"));
        assert!(!stdout_text.contains("error-side"));
        let stderr = writer.stderr.lock();
        let stderr_text = core::str::from_utf8(&stderr).expect("lines are ASCII");
        assert!(stderr_text.contains("error-side"));
        assert!(!stderr_text.contains("info-side"));
    }

    #[test]
    fn oversized_bodies_truncate_instead_of_failing() {
        use crate::LOG_LINE_BUFFER_SIZE;

        let mut ring = LogRing::new();
        let long = "x".repeat(LOG_LINE_BUFFER_SIZE * 2);
        ring.stage(LogLevel::Info, false, None, format_args!("{long}"))
            .expect("oversized bodies truncate instead of erroring");
        assert_eq!(ring.len, 1);
        let entry = &ring.entries[0];
        assert_eq!(entry.len, LOG_LINE_BUFFER_SIZE);
        assert_eq!(entry.bytes[entry.len - 1], b'\n');
    }
}
