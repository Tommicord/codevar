//! Copyright 2026 Codevar
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

//! High-performance multiplatform async logger with worker pool processing.
//!
//! This module provides a comprehensive logging system inspired by ezlog,
//! featuring async processing, bit-packed configuration, and cross-platform support.

use crate::base::base_comm::{bounded, unbounded};
use crate::fcware::compression::{Codec, compress, decompress};
use crate::logtrace::log_error::{Error, Result};
use crate::logtrace::log_fmt::Formatter;
use crate::logtrace::log_level::Level;
use crate::logtrace::log_queue::Queue;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::TrySendError;
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::base::{Daemon, Receiver, Sender};

/// Default number of worker threads for async log processing.
const DEFAULT_WORKER_COUNT: usize = 4;

/// Default channel capacity for log messages.
const DEFAULT_CHANNEL_CAPACITY: usize = 8192;

/// Maximum log entry size in bytes.
const MAX_LOG_ENTRY_SIZE: usize = 8192;

/// Maximum log file size in bytes
const MAX_FILE_SIZE: u64 = 131072;

const LOG_NAME_MAX: usize = 255;
const LOG_NAME_BYTES: usize = 32;

/// Platform-specific console output implementation.
mod platform_console {
    use super::Level;
    use std::io::{self, Write};

    /// Writes a formatted log line to the platform console.
    ///
    /// This function uses platform-specific APIs:
    /// - Android: __android_log_write
    /// - iOS/macOS: os_log (unified logging)
    /// - Linux: systemd journal if available, otherwise stdout
    /// - Windows: OutputDebugString
    /// - Other: stdout/stderr
    pub fn write_console_line(level: Level, tag: &str, message: &str) {
        if message.is_empty() {
            return;
        }

        #[cfg(target_os = "android")]
        {
            write_android_line(level, tag, message);
        }

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            write_apple_line(level, tag, message);
        }

        #[cfg(target_os = "linux")]
        {
            write_linux_line(level, tag, message);
        }

        #[cfg(target_os = "windows")]
        {
            write_windows_line(level, tag, message);
        }

        #[cfg(target_arch = "wasm32")]
        {
            write_wasm_line(level, tag, message);
        }

        #[cfg(not(any(
            target_os = "android",
            target_os = "ios",
            target_os = "macos",
            target_os = "linux",
            target_os = "windows",
            target_arch = "wasm32"
        )))]
        {
            write_fallback_line(level, tag, message);
        }
    }

    #[cfg(target_os = "android")]
    fn write_android_line(level: Level, tag: &str, message: &str) {
        use std::os::raw::c_char;

        let android_priority = android_priority(level);
        if let (Ok(c_tag), Ok(c_msg)) = (CString::new(tag), CString::new(message)) {
            unsafe {
                extern "C" {
                    fn __android_log_write(
                        prio: std::os::raw::c_int,
                        tag: *const c_char,
                        text: *const c_char,
                    ) -> std::os::raw::c_int;
                }
                __android_log_write(android_priority, c_tag.as_ptr(), c_msg.as_ptr());
            }
        }
    }

    #[cfg(target_os = "android")]
    fn android_priority(level: Level) -> std::os::raw::c_int {
        match level {
            Level::Trace => android_log_sys::LogPriority::Verbose as i32,
            Level::Debug => android_log_sys::LogPriority::Debug as i32,
            Level::Info => android_log_sys::LogPriority::Info as i32,
            Level::Warn => android_log_sys::LogPriority::Warn as i32,
            Level::Error => android_log_sys::LogPriority::Error as i32,
            Level::Fatal => android_log_sys::LogPriority::Fatal as i32,
        }
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn write_apple_line(level: Level, tag: &str, message: &str) {
        #[cfg(feature = "apple-os-log")]
        {
            use std::ffi::CString;
            if let Ok(c_formatted) = CString::new(message) {
                unsafe {
                    extern "C" {
                        fn os_log_with_type(format: *const std::os::raw::c_char);
                    }
                    os_log_with_type(c_formatted.as_ptr());
                }
            }
        }
        #[cfg(not(feature = "apple-os-log"))]
        {
            let _ = io::stdout().write_all(message.as_bytes());
        }
    }

    #[cfg(target_os = "linux")]
    fn write_linux_line(level: Level, tag: &str, message: &str) {
        // Try systemd journal first, fallback to stdout
        #[cfg(feature = "systemd-journal")]
        {
            use std::ffi::CString;
            if let Ok(c_formatted) = CString::new(message) {
                match std::fs::OpenOptions::new().write(true).open("/dev/log") {
                    Ok(mut file) => {
                        let _ = file.write_all(c_formatted.as_bytes());
                    }
                    Err(_) => {
                        write_fallback_line(level, tag, message);
                    }
                }
            }
        }
        #[cfg(not(feature = "systemd-journal"))]
        {
            write_fallback_line(level, tag, message);
        }
    }

    #[cfg(target_os = "windows")]
    fn write_windows_line(level: Level, tag: &str, message: &str) {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        let wide: Vec<u16> = OsStr::new(message)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        unsafe {
            extern "C" {
                fn OutputDebugStringW(lp_output_string: *const u16);
            }
            OutputDebugStringW(wide.as_ptr());
        }
        // Also write to console for visibility
        write_fallback_line(level, tag, message);
    }

    #[cfg(target_arch = "wasm32")]
    fn write_wasm_line(level: Level, tag: &str, message: &str) {
        web_sys::console::log_1(&message.into());
    }

    pub(crate) fn write_fallback_line(level: Level, tag: &str, message: &str) {
        let mut stdout = io::stdout().lock();
        let mut stderr = io::stderr().lock();
        let output = if level >= Level::Error {
            &mut stderr as &mut dyn Write
        } else {
            &mut stdout as &mut dyn Write
        };
        let _ = output.write_all(message.as_bytes());
    }
}

fn write_fallback_line(level: Level, tag: &str, message: &str) {
    platform_console::write_fallback_line(level, tag, message);
}

/// Fluent builder for logger-specific bit-packed flags.
/// This keeps the underlying flag layout stable while providing a builder-style API
/// instead of mutating shared flags via ad-hoc setters.
#[derive(Debug)]
pub struct LogConfigBuilder {
    /// Bit-packed flags using a single byte
    flags: AtomicU8,
}

impl LogConfigBuilder {
    const ASYNC: u8 = 0x01;
    const COMPRESSION: u8 = 0x02;
    const ENCRYPTION: u8 = 0x04;
    const FILE_OUTPUT: u8 = 0x08;
    const CONSOLE: u8 = 0x10;
    const TIMESTAMP: u8 = 0x20;
    const THREAD_ID: u8 = 0x40;
    const SOURCE_LOCATION: u8 = 0x80;
    const DEFAULT: u8 = Self::ASYNC
        | Self::ENCRYPTION
        | Self::CONSOLE
        | Self::TIMESTAMP
        | Self::THREAD_ID;

    #[inline]
    pub fn new() -> Self {
        Self {
            flags: AtomicU8::new(Self::DEFAULT),
        }
    }

    #[inline]
    pub fn build(self) -> LogConfigFlags {
        LogConfigFlags {
            flags: Arc::new(self.flags),
        }
    }

    #[inline]
    pub fn with_async(self, enabled: bool) -> Self {
        self.set_async_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_compression(self, enabled: bool) -> Self {
        self.set_compression_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_encryption(self, enabled: bool) -> Self {
        self.set_encryption_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_file_output(self, enabled: bool) -> Self {
        self.set_file_output_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_console_output(self, enabled: bool) -> Self {
        self.set_console_output_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_timestamp(self, enabled: bool) -> Self {
        self.set_timestamp_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_thread_id(self, enabled: bool) -> Self {
        self.set_thread_id_enabled(enabled);
        self
    }

    #[inline]
    pub fn with_source_location(self, enabled: bool) -> Self {
        self.set_source_location_enabled(enabled);
        self
    }

    #[inline]
    fn set_flag(&self, mask: u8, enabled: bool) {
        let mut flags = self.flags.load(Ordering::Relaxed);
        if enabled {
            flags |= mask;
        } else {
            flags &= !mask;
        }
        self.flags.store(flags, Ordering::Relaxed);
    }

    #[inline]
    pub fn async_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::ASYNC) != 0
    }

    #[inline]
    pub fn set_async_enabled(&self, enabled: bool) {
        self.set_flag(Self::ASYNC, enabled);
    }

    #[inline]
    pub fn compression_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::COMPRESSION) != 0
    }

    #[inline]
    pub fn set_compression_enabled(&self, enabled: bool) {
        self.set_flag(Self::COMPRESSION, enabled);
    }

    #[inline]
    pub fn encryption_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::ENCRYPTION) != 0
    }

    #[inline]
    pub fn set_encryption_enabled(&self, enabled: bool) {
        self.set_flag(Self::ENCRYPTION, enabled);
    }

    #[inline]
    pub fn file_output_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::FILE_OUTPUT) != 0
    }

    #[inline]
    pub fn set_file_output_enabled(&self, enabled: bool) {
        self.set_flag(Self::FILE_OUTPUT, enabled);
    }

    #[inline]
    pub fn console_output_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::CONSOLE) != 0
    }

    #[inline]
    pub fn set_console_output_enabled(&self, enabled: bool) {
        self.set_flag(Self::CONSOLE, enabled);
    }

    #[inline]
    pub fn timestamp_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::TIMESTAMP) != 0
    }

    #[inline]
    pub fn set_timestamp_enabled(&self, enabled: bool) {
        self.set_flag(Self::TIMESTAMP, enabled);
    }

    #[inline]
    pub fn thread_id_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::THREAD_ID) != 0
    }

    #[inline]
    pub fn set_thread_id_enabled(&self, enabled: bool) {
        self.set_flag(Self::THREAD_ID, enabled);
    }

    #[inline]
    pub fn source_location_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::SOURCE_LOCATION) != 0
    }

    #[inline]
    pub fn set_source_location_enabled(&self, enabled: bool) {
        self.set_flag(Self::SOURCE_LOCATION, enabled);
    }
}

/// Bit-packed configuration flags for logger settings.
/// This replaces multiple boolean fields with a single byte for better cache efficiency.
#[derive(Debug)]
#[repr(C)]
pub struct LogConfigFlags {
    /// Bit-packed flags using a single byte
    flags: Arc<AtomicU8>,
}

impl LogConfigFlags {
    const ASYNC: u8 = 0x01;
    const COMPRESSION: u8 = 0x02;
    const ENCRYPTION: u8 = 0x04;
    const CONSOLE: u8 = 0x8;
    const TIMESTAMP: u8 = 0x10;
    const THREAD_ID: u8 = 0x20;
    const SOURCE_LOCATION: u8 = 0x40;
    const DEFAULT: u8 = LogConfigBuilder::DEFAULT;

    #[inline]
    pub fn builder() -> LogConfigBuilder {
        LogConfigBuilder::new()
    }

    #[inline]
    pub fn new() -> Self {
        Self::builder().build()
    }

    #[inline]
    pub fn async_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::ASYNC) != 0
    }

    #[inline]
    pub fn compression_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::COMPRESSION) != 0
    }

    #[inline]
    pub fn encryption_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::ENCRYPTION) != 0
    }

    #[inline]
    pub fn console_output_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::CONSOLE) != 0
    }

    #[inline]
    pub fn timestamp_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::TIMESTAMP) != 0
    }

    #[inline]
    pub fn thread_id_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::THREAD_ID) != 0
    }

    #[inline]
    pub fn source_location_enabled(&self) -> bool {
        (self.flags.load(Ordering::Relaxed) & Self::SOURCE_LOCATION) != 0
    }
}

impl Default for LogConfigFlags {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LogConfigFlags {
    fn clone(&self) -> Self {
        Self {
            flags: Arc::new(AtomicU8::new(self.flags.load(Ordering::Relaxed))),
        }
    }
}

impl From<LogConfigBuilder> for LogConfigFlags {
    fn from(builder: LogConfigBuilder) -> Self {
        builder.build()
    }
}

/// Log entry with bit-packed metadata for efficient storage and processing.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Bit-packed metadata flags
    metadata: Arc<AtomicU64>,

    /// Timestamp in milliseconds since Unix epoch
    timestamp: Arc<AtomicU64>,

    /// Log level
    level: Level,

    /// Thread ID
    thread_id: Arc<AtomicU64>,

    /// Source file name (optional)
    file: Option<String>,

    /// Source line number (optional)
    line: Option<u32>,

    /// Module path (optional)
    module: Option<String>,

    /// Formatted message content
    message: String,

    /// Padding to prevent false sharing in multi-threaded scenarios
    _pad: [u8; 64],
}

impl LogEntry {
    /// Creates a new log entry.
    #[inline]
    pub fn new(level: Level, message: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let thread_id = thread_id_to_u64();
        Self {
            metadata: Arc::new(AtomicU64::new(0)),
            timestamp: Arc::new(AtomicU64::new(timestamp)),
            level,
            thread_id: Arc::new(AtomicU64::new(thread_id)),
            file: None,
            line: None,
            module: None,
            message,
            _pad: [0u8; 64],
        }
    }

    /// Creates a new log entry with source location.
    #[inline]
    pub fn with_location(
        level: Level,
        message: String,
        file: String,
        line: u32,
        module: String,
    ) -> Self {
        let mut entry = Self::new(level, message);
        entry.file = Some(file);
        entry.line = Some(line);
        entry.module = Some(module);
        entry
    }

    /// Returns the log level.
    #[inline]
    pub fn level(&self) -> Level {
        self.level
    }

    /// Returns the timestamp.
    #[inline]
    pub fn timestamp(&self) -> u64 {
        self.timestamp.load(Ordering::Relaxed)
    }

    /// Returns the thread ID.
    #[inline]
    pub fn thread_id(&self) -> u64 {
        self.thread_id.load(Ordering::Relaxed)
    }

    /// Returns the message content.
    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the source file if available.
    #[inline]
    pub fn file(&self) -> Option<&str> {
        self.file.as_deref()
    }

    /// Returns the source line if available.
    #[inline]
    pub fn line(&self) -> Option<u32> {
        self.line
    }

    /// Returns the module path if available.
    #[inline]
    pub fn module(&self) -> Option<&str> {
        self.module.as_deref()
    }

    /// Formats the log entry for output.
    pub fn format(
        &self,
        formatter: &mut Formatter,
        flags: &LogConfigFlags,
    ) -> Result<()> {
        if flags.timestamp_enabled() {
            formatter.append_str("[")?;
            formatter.format_u64(self.timestamp())?;
            formatter.append_str("] ")?;
        }
        formatter.append_str(self.level().as_str())?;
        formatter.append_str(": ")?;
        if flags.thread_id_enabled() {
            formatter.append_str("[TID:")?;
            formatter.format_u64(self.thread_id())?;
            formatter.append_str("] ")?;
        }
        if let Some(module) = self.module() {
            formatter.append_str("[")?;
            formatter.append_str(module)?;
            formatter.append_str("] ")?;
        }
        formatter.append_str(self.message())?;
        if flags.source_location_enabled() {
            if let Some(file) = self.file() {
                formatter.append_str(" (")?;
                formatter.append_str(file)?;
                if let Some(line) = self.line() {
                    formatter.append_char(':')?;
                    formatter.format_u32(line)?;
                }
                formatter.append_char(')')?;
            }
        }
        formatter.append_char('\n')
    }
}

/// Converts current thread ID to u64.
#[inline]
fn thread_id_to_u64() -> u64 {
    // Use a thread-local counter for cross-platform compatibility
    thread_local! {
        static THREAD_COUNTER: AtomicUsize = AtomicUsize::new(0);
    }
    THREAD_COUNTER.with(|counter| counter.fetch_add(1, Ordering::Relaxed) as u64)
}

/// Message types for async log processing.
#[derive(Debug, Clone)]
pub enum LogMessage {
    /// Log entry to be processed
    Entry(LogEntry),
    /// Flush request
    Flush,
    /// Shutdown request
    Shutdown,
}

/// Logger configuration with bit-packed flags.
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Logger tag for identification
    tag: String,

    /// Minimum log level to process
    min_level: Level,

    /// Bit-packed configuration flags
    flags: LogConfigFlags,

    /// Number of worker threads
    worker_count: usize,

    /// Channel capacity for log messages
    channel_capacity: usize,

    /// Maximum log file size in bytes
    max_file_size: u64,

    /// Log file name prefix
    log_name: String,
}

impl LoggerConfig {
    /// Creates a new logger configuration with the given tag.
    #[inline]
    pub fn new(tag: String) -> Self {
        Self {
            tag,
            min_level: Level::Debug,
            flags: LogConfigFlags::new(),
            worker_count: DEFAULT_WORKER_COUNT,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
            max_file_size: MAX_FILE_SIZE,
            log_name: String::new(),
        }
    }

    /// Sets the minimum log level.
    #[inline]
    pub fn with_min_level(mut self, level: Level) -> Self {
        self.min_level = level;
        self
    }

    /// Sets the configuration flags.
    #[inline]
    pub fn with_flags(mut self, flags: LogConfigFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Sets the number of worker threads.
    #[inline]
    pub fn with_worker_count(mut self, count: usize) -> Self {
        self.worker_count = count.max(1);
        self
    }

    /// Sets the channel capacity.
    #[inline]
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity.max(100);
        self
    }

    /// Sets the maximum file size.
    #[inline]
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size.max(1024);
        self
    }

    /// Sets the log name (defaults to tag if not set).
    #[inline]
    pub fn with_log_name(mut self, name: String) -> Self {
        self.log_name = name;
        self
    }

    /// Returns the logger tag.
    #[inline]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Returns the minimum log level.
    #[inline]
    pub fn min_level(&self) -> Level {
        self.min_level
    }

    /// Returns the configuration flags.
    #[inline]
    pub fn flags(&self) -> &LogConfigFlags {
        &self.flags
    }

    /// Returns the worker count.
    #[inline]
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Returns the channel capacity.
    #[inline]
    pub fn channel_capacity(&self) -> usize {
        self.channel_capacity
    }

    /// Returns the maximum file size.
    #[inline]
    pub fn max_file_size(&self) -> u64 {
        self.max_file_size
    }

    /// Returns the log name (or tag if not set).
    #[inline]
    pub fn log_name(&self) -> &str {
        if self.log_name.is_empty() {
            &self.tag
        } else {
            &self.log_name
        }
    }
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

/// High-performance async logger with worker pool.
pub struct Logger {
    /// Logger configuration
    config: LoggerConfig,

    /// Channel sender for log messages
    sender: Sender<LogMessage>,

    /// Channel receiver for log messages (held by workers)
    _receiver: Receiver<LogMessage>,

    /// Worker handles
    workers: Vec<Daemon<()>>,

    /// Flag indicating if logger is running
    running: Arc<AtomicBool>,

    /// Internal queue for synchronous fallback
    sync_queue: Arc<Mutex<Queue<LogEntry>>>,

    /// Statistics counters
    stats: Arc<LoggerStats>,

    /// Serializes file updates, including compressed read-modify-write cycles.
    file_lock: Arc<Mutex<()>>,
}

/// Logger statistics for monitoring.
#[derive(Debug, Default)]
pub struct LoggerStats {
    /// Total number of log entries processed
    total_entries: AtomicUsize,

    /// Number of dropped entries due to channel overflow
    dropped_entries: AtomicUsize,

    /// Number of flush operations
    flush_count: AtomicUsize,

    /// Number of errors encountered
    error_count: AtomicUsize,
}

impl LoggerStats {
    #[inline]
    pub fn total_entries(&self) -> usize {
        self.total_entries.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn dropped_entries(&self) -> usize {
        self.dropped_entries.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn flush_count(&self) -> usize {
        self.flush_count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn error_count(&self) -> usize {
        self.error_count.load(Ordering::Relaxed)
    }
}

impl Logger {
    /// Creates a new logger with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_core::logtrace::{LogConfigFlags, Logger, LoggerConfig};
    ///
    /// let flags = LogConfigFlags::builder()
    ///     .with_console_output(true)
    ///     .with_async(false)
    ///     .with_timestamp(true)
    ///     .build();
    /// let config = LoggerConfig::new("editor".to_string()).with_flags(flags);
    /// let logger = Logger::new(config)?;
    /// logger.log(crate::logtrace::Level::Info, "ready".to_string())?;
    /// # Ok::<(), codevar_core::logtrace::LogError>(())
    /// ```
    pub fn new(config: LoggerConfig) -> Result<Self> {
        let (sender, receiver) = if config.flags().async_enabled() {
            bounded(config.channel_capacity())
        } else {
            unbounded()
        };
        let running = Arc::new(AtomicBool::new(true));
        let sync_queue = Arc::new(Mutex::new(Queue::new()?));
        let stats = Arc::new(LoggerStats::default());
        let file_lock = Arc::new(Mutex::new(()));
        let mut workers = Vec::with_capacity(config.worker_count());

        for worker_id in 0..config.worker_count() {
            let worker_receiver = receiver.clone();
            let worker_running = running.clone();
            let worker_stats = stats.clone();
            let worker_config = config.clone();
            let worker_sync_queue = sync_queue.clone();
            let worker_file_lock = file_lock.clone();
            workers.push(Daemon::new(async move {
                Self::worker_loop(
                    worker_receiver,
                    worker_running,
                    worker_stats,
                    worker_config,
                    worker_sync_queue,
                    worker_file_lock,
                );
            }));
        }
        Ok(Self {
            config,
            sender,
            _receiver: receiver,
            workers,
            running,
            sync_queue,
            stats,
            file_lock,
        })
    }

    /// Worker thread main loop for processing log messages.
    fn worker_loop(
        mut receiver: Receiver<LogMessage>,
        running: Arc<AtomicBool>,
        stats: Arc<LoggerStats>,
        config: LoggerConfig,
        sync_queue: Arc<Mutex<Queue<LogEntry>>>,
        file_lock: Arc<Mutex<()>>,
    ) {
        let mut formatter = Formatter::new();

        while running.load(Ordering::Relaxed) {
            match receiver.recv() {
                Ok(LogMessage::Entry(entry)) => {
                    Self::process_entry(
                        &entry,
                        &mut formatter,
                        &config,
                        &stats,
                        &file_lock,
                    );
                    stats.total_entries.fetch_add(1, Ordering::Relaxed);
                }
                Ok(LogMessage::Flush) => {
                    // Self::flush(&config, &stats, &file_lock);
                    stats.flush_count.fetch_add(1, Ordering::Relaxed);
                }
                Ok(LogMessage::Shutdown) => {
                    break;
                }
                Err(_) => {
                    // Channel disconnected, exit loop
                    break;
                }
            }
        }
        if let Ok(mut queue) = sync_queue.lock() {
            while let Some(entry) = queue.pop() {
                Self::process_entry(&entry, &mut formatter, &config, &stats, &file_lock);
            }
        }
    }

    /// Processes a single log entry.
    fn process_entry(
        entry: &LogEntry,
        formatter: &mut Formatter,
        config: &LoggerConfig,
        stats: &LoggerStats,
        file_lock: &Mutex<()>,
    ) {
        formatter.reset();
        if let Err(_) = entry.format(formatter, config.flags()) {
            stats.error_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let formatted = formatter.as_str();
        if config.flags().console_output_enabled() {
            platform_console::write_console_line(entry.level(), config.tag(), formatted);
        }
    }

    /// Logs a message at the specified level.
    ///
    /// For formatted messages, prefer the exported logging macros. They pass
    /// `format_args!` directly to [`Formatter::format_args`] and accept any
    /// number of formatting arguments without a separate `format!` call.
    ///
    /// ```
    /// codevar_core::info!("opened {} files", file_count);
    /// ```
    pub fn log(&self, level: Level, message: String) -> Result<()> {
        if level < self.config.min_level() {
            return Ok(());
        }
        let entry = LogEntry::new(level, message);
        if self.config.flags().async_enabled() {
            match self.sender.try_send(LogMessage::Entry(entry.clone())) {
                Ok(_) => Ok(()),
                Err(TrySendError::Full(_)) => {
                    // Channel full, try synchronous fallback
                    self.stats.dropped_entries.fetch_add(1, Ordering::Relaxed);
                    self.sync_log(entry)
                }
                Err(TrySendError::Disconnected(_)) => {
                    Err(Error::LockError("Logger channel disconnected".to_string()))
                }
            }
        } else {
            self.sync_log(entry)
        }
    }

    /// Logs a message with source location.
    pub fn log_with_location(
        &self,
        level: Level,
        message: String,
        file: String,
        line: u32,
        module: String,
    ) -> Result<()> {
        if level < self.config.min_level() {
            return Ok(());
        }
        let entry = LogEntry::with_location(level, message, file, line, module);
        if self.config.flags().async_enabled() {
            match self.sender.try_send(LogMessage::Entry(entry.clone())) {
                Ok(_) => Ok(()),
                Err(TrySendError::Full(_)) => {
                    self.stats.dropped_entries.fetch_add(1, Ordering::Relaxed);
                    self.sync_log(entry)
                }
                Err(TrySendError::Disconnected(_)) => {
                    Err(Error::LockError("Logger channel disconnected".to_string()))
                }
            }
        } else {
            self.sync_log(entry)
        }
    }

    /// Synchronous fallback for when channel is full or async is disabled.
    fn sync_log(&self, entry: LogEntry) -> Result<()> {
        let mut formatter = Formatter::new();
        Self::process_entry(
            &entry,
            &mut formatter,
            &self.config,
            &self.stats,
            &self.file_lock,
        );
        self.stats.total_entries.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Shuts down the logger and waits for all workers to finish.
    pub fn shutdown(&mut self) -> Result<()> {
        self.running.store(false, Ordering::Relaxed);
        for _ in 0..self.config.worker_count() {
            let _ = self.sender.send(LogMessage::Shutdown);
        }
        let workers = std::mem::take(&mut self.workers);
        workers.into_iter().for_each(move |worker| worker.cancel());
        Ok(())
    }

    /// Returns the active logger configuration.
    #[inline]
    pub fn config(&self) -> &LoggerConfig {
        &self.config
    }

    /// Returns logger statistics.
    #[inline]
    pub fn stats(&self) -> &LoggerStats {
        &self.stats
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Logger registry for managing multiple loggers with tags.
struct LoggerRegistry {
    loggers: HashMap<String, Arc<Logger>>,
    default_tag: Option<String>,
}

impl LoggerRegistry {
    fn new() -> Self {
        Self {
            loggers: HashMap::new(),
            default_tag: None,
        }
    }
    fn register(&mut self, tag: String, logger: Logger) -> Result<()> {
        if self.loggers.contains_key(&tag) {
            return Err(Error::AlreadyInitialized);
        }
        self.loggers.insert(tag.clone(), Arc::new(logger));
        if self.default_tag.is_none() {
            self.default_tag = Some(tag);
        }

        Ok(())
    }

    fn get(&self, tag: &str) -> Result<Arc<Logger>> {
        self.loggers
            .get(tag)
            .cloned()
            .ok_or_else(|| Error::InvalidParameter(format!("Logger '{}' not found", tag)))
    }

    fn default(&self) -> Result<Arc<Logger>> {
        if let Some(ref default_tag) = self.default_tag {
            self.get(default_tag)
        } else {
            Err(Error::NotInitialized)
        }
    }

    fn set_default(&mut self, tag: String) -> Result<()> {
        if !self.loggers.contains_key(&tag) {
            return Err(Error::InvalidParameter(format!(
                "Logger '{}' not found",
                tag
            )));
        }
        self.default_tag = Some(tag);
        Ok(())
    }

    fn remove(&mut self, tag: &str) -> Result<()> {
        if let Some(logger) = self.loggers.remove(tag) {
            if let Some(mut logger_arc) = Arc::into_inner(logger) {
                let _ = logger_arc.shutdown();
            }
            if self.default_tag.as_ref() == Some(&tag.to_string()) {
                self.default_tag = self.loggers.keys().next().cloned();
            }
            Ok(())
        } else {
            Err(Error::InvalidParameter(format!(
                "Logger '{}' not found",
                tag
            )))
        }
    }

    fn shutdown_all(&mut self) -> Result<()> {
        let loggers = std::mem::take(&mut self.loggers);
        for (_, logger) in loggers {
            if let Some(mut logger_arc) = Arc::into_inner(logger) {
                let _ = logger_arc.shutdown();
            }
        }
        self.default_tag = None;
        Ok(())
    }
}

/// Global logger registry (using RwLock for concurrent access).
static LOGGER_REGISTRY: LazyLock<RwLock<LoggerRegistry>> =
    LazyLock::new(|| RwLock::new(LoggerRegistry::new()));

/// Initializes a logger with the given configuration and tag.
pub fn init_logger(config: LoggerConfig) -> Result<()> {
    let tag = config.tag().to_string();
    let logger = Logger::new(config)?;
    let mut registry = LOGGER_REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    registry.register(tag, logger)
}

/// Sets the default logger by tag.
pub fn set_default_logger(tag: String) -> Result<()> {
    let mut registry = LOGGER_REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    registry.set_default(tag)
}

/// Returns a reference to the default logger.
pub fn default_logger() -> Result<Arc<Logger>> {
    let registry = LOGGER_REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    registry.default()
}

/// Returns a reference to a specific logger by tag.
pub fn logger(tag: &str) -> Result<Arc<Logger>> {
    let registry = LOGGER_REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    registry.get(tag)
}

/// Removes a logger by tag.
pub fn remove_logger(tag: &str) -> Result<()> {
    let mut registry = LOGGER_REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    registry.remove(tag)
}

/// Shuts down all loggers.
pub fn shutdown_all() -> Result<()> {
    let mut registry = LOGGER_REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    registry.shutdown_all()
}

/// Convenience function to log a message at the trace level using the default logger.
pub fn trace(message: String) -> Result<()> {
    default_logger()?.log(Level::Trace, message)
}

fn format_message(args: std::fmt::Arguments<'_>) -> Result<String> {
    let mut formatter = Formatter::new();
    formatter.format_args(args)?;
    Ok(formatter.as_str().to_owned())
}

#[doc(hidden)]
pub fn log_formatted(level: Level, args: std::fmt::Arguments<'_>) -> Result<()> {
    default_logger()?.log(level, format_message(args)?)
}

/// Convenience function to log a message at the debug level using the default logger.
pub fn debug(message: String) -> Result<()> {
    default_logger()?.log(Level::Debug, message)
}

/// Convenience function to log a message at the info level using the default logger.
pub fn info(message: String) -> Result<()> {
    default_logger()?.log(Level::Info, message)
}

/// Convenience function to log a message at the warn level using the default logger.
pub fn warn(message: String) -> Result<()> {
    default_logger()?.log(Level::Warn, message)
}

/// Convenience function to log a message at the error level using the default logger.
pub fn error(message: String) -> Result<()> {
    default_logger()?.log(Level::Error, message)
}

/// Convenience function to log a message at the fatal level using the default logger.
pub fn fatal(message: String) -> Result<()> {
    default_logger()?.log(Level::Fatal, message)
}

/// Convenience function to log a message with source location at the trace level using the default logger.
pub fn trace_with_location(
    message: String,
    file: String,
    line: u32,
    module: String,
) -> Result<()> {
    default_logger()?.log_with_location(Level::Trace, message, file, line, module)
}

/// Convenience function to log a message with source location at the debug level using the default logger.
pub fn debug_with_location(
    message: String,
    file: String,
    line: u32,
    module: String,
) -> Result<()> {
    default_logger()?.log_with_location(Level::Debug, message, file, line, module)
}

/// Convenience function to log a message with source location at the info level using the default logger.
pub fn info_with_location(
    message: String,
    file: String,
    line: u32,
    module: String,
) -> Result<()> {
    default_logger()?.log_with_location(Level::Info, message, file, line, module)
}

/// Convenience function to log a message with source location at the warn level using the default logger.
pub fn warn_with_location(
    message: String,
    file: String,
    line: u32,
    module: String,
) -> Result<()> {
    default_logger()?.log_with_location(Level::Warn, message, file, line, module)
}

/// Convenience function to log a message with source location at the error level using the default logger.
pub fn error_with_location(
    message: String,
    file: String,
    line: u32,
    module: String,
) -> Result<()> {
    default_logger()?.log_with_location(Level::Error, message, file, line, module)
}

/// Convenience function to log a message with source location at the fatal level using the default logger.
pub fn fatal_with_location(
    message: String,
    file: String,
    line: u32,
    module: String,
) -> Result<()> {
    default_logger()?.log_with_location(Level::Fatal, message, file, line, module)
}

/// Logs a trace message with formatting arguments.
///
/// # Examples
///
/// ```ignore
/// codevar_core::trace!("received {} bytes", byte_count);
/// ```
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        $crate::logtrace::log_formatted(
            $crate::logtrace::Level::Trace,
            format_args!($($arg)*),
        )
    };
}

/// Logs a debug message with formatting arguments.
///
/// # Examples
///
/// ```ignore
/// codevar_core::debug!("processing block {}", block_id);
/// ```
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        $crate::logtrace::log_formatted(
            $crate::logtrace::Level::Debug,
            format_args!($($arg)*),
        )
    };
}

/// Logs an info message with formatting arguments.
///
/// # Examples
///
/// ```ignore
/// codevar_core::info!("opened {} files", file_count);
/// ```
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        $crate::logtrace::log_formatted(
            $crate::logtrace::Level::Info,
            format_args!($($arg)*),
        )
    };
}

/// Logs a warning message with formatting arguments.
///
/// # Examples
///
/// ```ignore
/// codevar_core::warn!("retrying request {}, attempt {}", request_id, attempt);
/// ```
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        $crate::logtrace::log_formatted(
            $crate::logtrace::Level::Warn,
            format_args!($($arg)*),
        )
    };
}

/// Logs an error message with formatting arguments.
///
/// # Examples
///
/// ```ignore
/// codevar_core::error!("failed to open {}: {}", path, reason);
/// ```
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::logtrace::log_formatted(
            $crate::logtrace::Level::Error,
            format_args!($($arg)*),
        )
    };
}

#[cfg(test)]
mod tests {
    use crate::logtrace::{Level, LogConfigFlags, Logger, LoggerConfig};
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn logger_config_default_has_debug_min_level() {
        let config = LoggerConfig::new("demo".to_string());
        assert_eq!(config.min_level(), Level::Debug);
    }

    fn logger_config_tag_is_preserved() {
        let config = LoggerConfig::new("demo".to_string());
        assert_eq!(config.tag(), "demo");
    }

    fn logger_config_with_min_level_updates_level() {
        let config = LoggerConfig::new("demo".to_string()).with_min_level(Level::Warn);
        assert_eq!(config.min_level(), Level::Warn);
    }

    fn logger_config_with_worker_count_keeps_positive() {
        let config = LoggerConfig::new("demo".to_string()).with_worker_count(0);
        assert_eq!(config.worker_count(), 1);
    }

    fn logger_config_with_channel_capacity_keeps_positive() {
        let config = LoggerConfig::new("demo".to_string()).with_channel_capacity(0);
        assert!(config.channel_capacity() >= 100);
    }

    fn logger_config_with_max_file_size_keeps_positive() {
        let config = LoggerConfig::new("demo".to_string()).with_max_file_size(0);
        assert!(config.max_file_size() >= 1024);
    }

    fn logger_config_log_name_defaults_to_tag() {
        let config = LoggerConfig::new("demo".to_string());
        assert_eq!(config.log_name(), "demo");
    }

    fn logger_config_custom_name_is_used() {
        let config =
            LoggerConfig::new("demo".to_string()).with_log_name("custom".to_string());
        assert_eq!(config.log_name(), "custom");
    }

    fn logger_config_flags_default_async_enabled() {
        let config = LoggerConfig::new("demo".to_string());
        assert!(config.flags().async_enabled());
    }

    fn logger_config_flags_can_toggle_async() {
        let flags = LogConfigFlags::builder()
            .with_async(false);
        assert!(!flags.async_enabled());
    }

    fn logger_config_flags_can_toggle_console_output() {
        let flags = LogConfigFlags::builder()
            .with_console_output(false);
        assert!(!flags.console_output_enabled());
    }

    fn logger_config_flags_can_toggle_file_output() {
        let flags = LogConfigFlags::builder()
            .with_file_output(true);
        assert!(flags.file_output_enabled());
    }

    fn logger_config_flags_can_toggle_timestamp() {
        let flags = LogConfigFlags::builder()
            .with_timestamp(false)
            .build();
        assert!(!flags.timestamp_enabled());
    }

    fn logger_config_flags_can_toggle_thread_id() {
        let flags = LogConfigFlags::builder()
            .with_thread_id(false);
        assert!(!flags.thread_id_enabled());
    }

    fn logger_config_flags_can_toggle_source_location() {
        let flags = LogConfigFlags::builder()
            .with_source_location(true);
        assert!(flags.source_location_enabled());
    }

    fn logger_config_flags_clone_matches_value() {
        let flags = LogConfigFlags::new();
        let clone = flags.clone();
        assert_eq!(clone.async_enabled(), flags.async_enabled());
    }

    fn logger_new_constructs_logger() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128);
        let logger = Logger::new(config).unwrap();
        assert_eq!(logger.stats().total_entries(), 0);
    }

    fn logger_log_syncs_when_async_disabled() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128)
            .with_flags(LogConfigFlags::new());
        let logger = Logger::new(config).unwrap();
        logger.log(Level::Info, "hello".to_string()).unwrap();
    }

    fn logger_log_with_location_is_supported() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128);
        let logger = Logger::new(config).unwrap();
        logger
            .log_with_location(
                Level::Info,
                "msg".to_string(),
                "test.rs".to_string(),
                42,
                "module".to_string(),
            )
            .unwrap();
    }

    fn logger_min_level_rejects_messages_below_threshold() {
        let config = LoggerConfig::new("demo".to_string())
            .with_min_level(Level::Error)
            .with_worker_count(1)
            .with_channel_capacity(128);
        let logger = Logger::new(config).unwrap();
        assert!(logger.log(Level::Info, "ignored".to_string()).is_ok());
    }

    fn logger_shutdown_completes_cleanly() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128);
        let mut logger = Logger::new(config).unwrap();
        logger.shutdown().unwrap();
    }

    fn logger_drop_works_without_leaking() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128);
        let logger = Logger::new(config).unwrap();
        drop(logger);
    }

    fn logger_stats_are_default_zero() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128);
        let logger = Logger::new(config).unwrap();
        let stats = logger.stats();
        assert_eq!(stats.total_entries(), 0);
        assert_eq!(stats.dropped_entries(), 0);
        assert_eq!(stats.flush_count(), 0);
        assert_eq!(stats.error_count(), 0);
    }

    fn logger_is_cloneable_via_arc() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(1)
            .with_channel_capacity(128);
        let logger = Arc::new(Logger::new(config).unwrap());
        let clone = logger.clone();
        assert_eq!(Arc::strong_count(&logger), 2);
        drop(clone);
    }

    fn logger_handles_concurrent_senders() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(2)
            .with_channel_capacity(256);
        let logger = Arc::new(Logger::new(config).unwrap());
        let barrier = Arc::new(Barrier::new(4));
        let mut threads = Vec::new();

        for _ in 0..3 {
            let logger = logger.clone();
            let barrier = barrier.clone();
            threads.push(thread::spawn(move || {
                barrier.wait();
                for i in 0..20 {
                    let _ = logger.log(Level::Info, format!("msg{i}"));
                }
            }));
        }

        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    fn logger_can_process_many_messages() {
        let config = LoggerConfig::new("demo".to_string())
            .with_worker_count(2)
            .with_channel_capacity(1024);
        let logger = Arc::new(Logger::new(config).unwrap());
        for i in 0..200 {
            logger.log(Level::Debug, format!("msg{i}")).unwrap();
        }
    }

    fn logger_flags_can_be_reused_after_reset() {
        let flags = LogConfigFlags::builder()
            .with_async(true)
            .with_console_output(false)
            .with_timestamp(true)
            .with_thread_id(true);
        assert!(flags.async_enabled());
        assert!(!flags.console_output_enabled());
        assert!(flags.timestamp_enabled());
        assert!(flags.thread_id_enabled());
    }

    fn logger_config_default_flags_match_expected() {
        let flags = LogConfigFlags::new();
        assert!(flags.async_enabled());
        assert!(!flags.compression_enabled());
        assert!(flags.encryption_enabled());
        assert!(flags.console_output_enabled());
        assert!(flags.timestamp_enabled());
        assert!(flags.thread_id_enabled());
    }

    fn logger_config_can_disable_encryption() {
        let flags = LogConfigFlags::builder()
            .with_encryption(false);
        assert!(!flags.encryption_enabled());
    }

    fn logger_config_can_disable_compression() {
        let flags = LogConfigFlags::builder()
            .with_compression(false);
        assert!(!flags.compression_enabled());
    }

    fn logger_config_can_disable_source_location() {
        let flags = LogConfigFlags::builder()
            .with_source_location(false);
        assert!(!flags.source_location_enabled());
    }

    fn logger_config_can_toggle_encryption_and_compression() {
        let flags = LogConfigFlags::builder()
            .with_encryption(false)
            .with_compression(false);
        assert!(!flags.encryption_enabled());
        assert!(!flags.compression_enabled());
    }

    fn logger_config_multiple_instances_are_independent() {
        let first = LoggerConfig::new("a".to_string());
        let second = LoggerConfig::new("b".to_string());
        assert_ne!(first.tag(), second.tag());
    }

    fn logger_config_max_file_size_is_clamped() {
        let config = LoggerConfig::new("demo".to_string()).with_max_file_size(42);
        assert!(config.max_file_size() >= 1024);
    }

    fn logger_config_default_is_stable() {
        let config = LoggerConfig::default();
        assert_eq!(config.tag(), "default");
    }

    #[test]
    fn generated_log_names_are_hex_and_seeded() {
        let first = super::generate_log_name(b"first-seed");
        let second = super::generate_log_name(b"second-seed");
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    #[test]
    fn compressed_file_output_roundtrips_through_fcware() {
        let log_name = format!("codevar-log-test-{}", std::process::id());
        let path = std::env::temp_dir().join(format!("{log_name}.log"));
        let _ = std::fs::remove_file(&path);
        let flags = LogConfigFlags::builder()
            .with_async(false)
            .with_console_output(false)
            .with_file_output(true)
            .with_compression(true)
            .build();
        let config = LoggerConfig::new("compressed_test".to_string())
            .with_log_name(log_name)
            .with_flags(flags);
        let logger = Logger::new(config).unwrap();

        logger.log(Level::Info, "first".to_string()).unwrap();
        logger.log(Level::Info, "second".to_string()).unwrap();

        let encoded = std::fs::read(&path).unwrap();
        let decoded = crate::fcware::compression::decompress(&encoded).unwrap();
        let output = String::from_utf8(decoded).unwrap();
        assert!(output.contains("first"));
        assert!(output.contains("second"));
        let _ = std::fs::remove_file(path);
    }
}
