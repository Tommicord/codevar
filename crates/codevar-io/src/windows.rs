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

//! Windows-specific standard I/O implementation.
//!
//! This module provides implementations of stdin, stdout, and stderr for Windows
//! using Win32 console APIs with proper locking support.

use crate::{ErrorKind, ErrorType, Read, ReadExactError, Write};
use core::ffi::c_void;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};

/// Windows API constants
const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6; // -10
const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5; // -11
const STD_ERROR_HANDLE: u32 = 0xFFFFFFF4; // -12

const INVALID_HANDLE_VALUE: isize = -1;
const ERROR_INVALID_HANDLE: u32 = 6;

const CP_UTF8: u32 = 65001;
const MB_ERR_INVALID_CHARS: u32 = 0x0008;
const WC_ERR_INVALID_CHARS: u32 = 0x0008;

unsafe extern "system" {
    fn GetStdHandle(n_std_handle: u32) -> *mut c_void;
    fn GetConsoleMode(h_console_handle: *mut c_void, lp_mode: *mut u32) -> i32;
    fn SetLastError(dw_err_code: u32);
    fn ReadConsoleW(
        h_console_input: *mut c_void,
        lp_buffer: *mut c_void,
        n_number_of_chars_to_read: u32,
        lp_number_of_chars_read: *mut u32,
        p_input_control: *mut c_void,
    ) -> i32;
    fn WriteConsoleW(
        h_console_output: *mut c_void,
        lp_buffer: *const c_void,
        n_number_of_chars_to_write: u32,
        lp_number_of_chars_written: *mut u32,
        lp_reserved: *mut c_void,
    ) -> i32;
    fn WriteFile(
        h_file: *mut c_void,
        lp_buffer: *const c_void,
        n_number_of_bytes_to_write: u32,
        lp_number_of_bytes_written: *mut u32,
        lp_overlapped: *mut c_void,
    ) -> i32;
    fn MultiByteToWideChar(
        code_page: u32,
        dw_flags: u32,
        lp_multi_byte_str: *const i8,
        cb_multi_byte: i32,
        lp_wide_char_str: *mut u16,
        cch_wide_char: i32,
    ) -> i32;
    fn WideCharToMultiByte(
        code_page: u32,
        dw_flags: u32,
        lp_wide_char_str: *const u16,
        cch_wide_char: i32,
        lp_multi_byte_str: *mut i8,
        cb_multi_byte: i32,
        lp_default_char: *mut i8,
        lp_used_default_char: *mut i32,
    ) -> i32;
    fn GetConsoleOutputCP() -> u32;
    fn GetLastError() -> u32;
}

/// Incomplete UTF-8 state for handling partial sequences
#[derive(Debug)]
struct IncompleteUtf8 {
    bytes: [u8; 4],
    len: u8,
}

impl IncompleteUtf8 {
    const fn new() -> Self {
        Self {
            bytes: [0; 4],
            len: 0,
        }
    }

    fn read(&mut self, buf: &mut [u8]) -> usize {
        let to_write = core::cmp::min(buf.len(), self.len as usize);
        if to_write > 0 {
            buf[..to_write].copy_from_slice(&self.bytes[..to_write]);
            if to_write < self.len as usize {
                self.bytes.copy_within(to_write.., 0);
                self.len -= to_write as u8;
            } else {
                self.len = 0;
            }
        }
        to_write
    }
}

/// Write mode for stdout/stderr
#[derive(Debug, Clone, Copy)]
enum WriteMode {
    /// Direct write (pipe, file, or UTF-8 console)
    Passthrough(*mut c_void),
    /// UTF-16 console write
    Utf16Console(*mut c_void),
}

unsafe impl Send for WriteMode {}
unsafe impl Sync for WriteMode {}

/// Simple spinlock for synchronization
struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    fn lock(&self) {
        while self.locked.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }
    }

    fn try_lock(&self) -> bool {
        !self.locked.swap(true, Ordering::Acquire)
    }

    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

/// Lock guard for SpinLock
struct SpinLockGuard<'a> {
    lock: &'a SpinLock,
}

impl<'a> SpinLockGuard<'a> {
    fn new(lock: &'a SpinLock) -> Self {
        lock.lock();
        Self { lock }
    }
}

impl Drop for SpinLockGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock();
    }
}

/// Global locks for standard streams
static STDIN_LOCK: SpinLock = SpinLock::new();
static STDOUT_LOCK: SpinLock = SpinLock::new();
static STDERR_LOCK: SpinLock = SpinLock::new();

/// A handle to the standard input stream of a process.
#[derive(Debug)]
pub struct Stdin {
    surrogate: u16,
    incomplete_utf8: IncompleteUtf8,
}

/// A locked reference to the [`Stdin`] handle.
#[derive(Debug)]
pub struct StdinLock<'a> {
    _guard: SpinLockGuard<'a>,
}

impl StdinLock<'static> {
    /// Creates a new lock on the global stdin
    fn new() -> Self {
        Self {
            _guard: SpinLockGuard::new(&STDIN_LOCK),
        }
    }
}

/// A handle to the global standard output stream.
#[derive(Debug)]
pub struct Stdout {
    incomplete_utf8: IncompleteUtf8,
    write_mode: Option<WriteMode>,
}

/// A locked reference to the [`Stdout`] handle.
#[derive(Debug)]
pub struct StdoutLock<'a> {
    inner: &'a mut Stdout,
    _guard: SpinLockGuard<'a>,
}

impl StdoutLock<'static> {
    fn new(stdout: &'static mut Stdout) -> Self {
        Self {
            inner: stdout,
            _guard: SpinLockGuard::new(&STDOUT_LOCK),
        }
    }
}

/// A handle to the standard error stream.
#[derive(Debug)]
pub struct Stderr {
    incomplete_utf8: IncompleteUtf8,
    write_mode: Option<WriteMode>,
}

/// A locked reference to the [`Stderr`] handle.
#[derive(Debug)]
pub struct StderrLock<'a> {
    inner: &'a mut Stderr,
    _guard: SpinLockGuard<'a>,
}

impl StderrLock<'static> {
    fn new(stderr: &'static mut Stderr) -> Self {
        Self {
            inner: stderr,
            _guard: SpinLockGuard::new(&STDERR_LOCK),
        }
    }
}
static mut STDOUT_INSTANCE: Option<Stdout> = None;
static mut STDERR_INSTANCE: Option<Stderr> = None;
static mut STDIN_INSTANCE: Option<Stdin> = None;

fn get_stdout_instance() -> &'static mut Stdout {
    unsafe {
        if STDOUT_INSTANCE.is_none() {
            STDOUT_INSTANCE = Some(Stdout::new());
        }
        STDOUT_INSTANCE
            .as_mut()
            .expect("STDOUT_INSTANCE should be initialized")
    }
}

fn get_stderr_instance() -> &'static mut Stderr {
    unsafe {
        if STDERR_INSTANCE.is_none() {
            STDERR_INSTANCE = Some(Stderr::new());
        }
        STDERR_INSTANCE
            .as_mut()
            .expect("STDERR_INSTANCE should be initialized")
    }
}
fn get_stdin_instance() -> &'static mut Stderr {
    unsafe {
        if STDIN_INSTANCE.is_none() {
            STDIN_INSTANCE = Some(Stdin::new());
        }
        STDIN_INSTANCE
            .as_mut()
            .expect("STDERR_INSTANCE should be initialized")
    }
}

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin {
            surrogate: 0,
            incomplete_utf8: IncompleteUtf8::new(),
        }
    }

    pub fn lock(&self) -> StdinLock<'static> {
        StdinLock::new()
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout {
            incomplete_utf8: IncompleteUtf8::new(),
            write_mode: None,
        }
    }

    pub fn lock(&self) -> StdoutLock<'static> {
        StdoutLock::new(get_stdout_instance())
    }

    /// Forgets the cached stream state, so the next write re-queries the OS.
    pub fn refresh(&mut self) {
        self.write_mode = None;
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr {
            incomplete_utf8: IncompleteUtf8::new(),
            write_mode: None,
        }
    }

    pub fn lock(&self) -> StderrLock<'static> {
        StderrLock::new(get_stderr_instance())
    }

    pub fn refresh(&mut self) {
        self.write_mode = None;
    }
}

/// Get a standard handle
fn get_handle(handle_id: u32) -> Result<*mut c_void, ErrorKind> {
    let handle = unsafe { GetStdHandle(handle_id) };
    if handle.is_null() || handle as isize == INVALID_HANDLE_VALUE {
        Err(ErrorKind::NotFound)
    } else {
        Ok(handle)
    }
}

/// Check if a handle is a console
fn is_console(handle: *mut c_void) -> bool {
    let mut mode = 0;
    unsafe { GetConsoleMode(handle, &mut mode) != 0 }
}

/// Check if the console is using UTF-8 code page
fn is_utf8_console() -> bool {
    unsafe { GetConsoleOutputCP() == CP_UTF8 }
}

/// Write data to a standard handle
fn write(
    handle_id: u32,
    data: &[u8],
    incomplete_utf8: &mut IncompleteUtf8,
    write_mode: &mut Option<WriteMode>,
) -> Result<usize, ErrorKind> {
    if data.is_empty() {
        return Ok(0);
    }
    let mode = match *write_mode {
        Some(mode) => mode,
        None => {
            let handle = get_handle(handle_id)?;
            let mode = if !is_console(handle) || is_utf8_console() {
                WriteMode::Passthrough(handle)
            } else {
                WriteMode::Utf16Console(handle)
            };
            *write_mode = Some(mode);
            mode
        }
    };

    match mode {
        WriteMode::Passthrough(handle) => write_file(handle, data),
        WriteMode::Utf16Console(handle) => write_console_utf16(data, incomplete_utf8, handle),
    }
}

/// Write to a file/pipe handle
fn write_file(handle: *mut c_void, data: &[u8]) -> Result<usize, ErrorKind> {
    let mut written = 0;
    let ret = unsafe {
        WriteFile(
            handle,
            data.as_ptr() as *const c_void,
            data.len() as u32,
            &mut written,
            core::ptr::null_mut(),
        )
    };
    if ret == 0 {
        Err(last_error_to_errorkind())
    } else {
        Ok(written as usize)
    }
}

/// Write UTF-8 data to console as UTF-16
fn write_console_utf16(
    data: &[u8],
    incomplete_utf8: &mut IncompleteUtf8,
    handle: *mut c_void,
) -> Result<usize, ErrorKind> {
    if incomplete_utf8.len > 0 {
        if data[0] >> 6 != 0b10 {
            incomplete_utf8.len = 0;
            return Err(ErrorKind::InvalidData);
        }
        incomplete_utf8.bytes[incomplete_utf8.len as usize] = data[0];
        incomplete_utf8.len += 1;
        let char_width = utf8_char_width(incomplete_utf8.bytes[0]);
        if (incomplete_utf8.len as usize) < char_width {
            return Ok(1);
        }
        let s = core::str::from_utf8(&incomplete_utf8.bytes[0..incomplete_utf8.len as usize]);
        incomplete_utf8.len = 0;
        match s {
            Ok(s) => {
                let written = write_valid_utf8_to_console(handle, s)?;
                return Ok(1);
            }
            Err(_) => return Err(ErrorKind::InvalidData),
        }
    }
    let len = core::cmp::min(data.len(), MAX_BUFFER_SIZE / 2);
    let utf8 = match core::str::from_utf8(&data[..len]) {
        Ok(s) => s,
        Err(e) if e.valid_up_to() == 0 => {
            let first_byte_char_width = utf8_char_width(data[0]);
            if first_byte_char_width > 1 && data.len() < first_byte_char_width {
                incomplete_utf8.bytes[0] = data[0];
                incomplete_utf8.len = 1;
                return Ok(1);
            } else {
                return Err(ErrorKind::InvalidData);
            }
        }
        Err(e) => core::str::from_utf8(&data[..e.valid_up_to()]).unwrap(),
    };
    write_valid_utf8_to_console(handle, utf8)
}

/// Write valid UTF-8 to console
fn write_valid_utf8_to_console(handle: *mut c_void, utf8: &str) -> Result<usize, ErrorKind> {
    let mut utf16 = [0u16; MAX_BUFFER_SIZE / 2];
    let utf8 = &utf8[..utf8.floor_char_boundary(utf16.len())];

    let result = unsafe {
        MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            utf8.as_ptr() as *const i8,
            utf8.len() as i32,
            utf16.as_mut_ptr(),
            utf16.len() as i32,
        )
    };
    if result == 0 {
        return Err(ErrorKind::InvalidData);
    }

    let utf16 = &utf16[..result as usize];
    let mut written = 0;
    let ret = unsafe {
        WriteConsoleW(
            handle,
            utf16.as_ptr() as *const c_void,
            utf16.len() as u32,
            &mut written,
            core::ptr::null_mut(),
        )
    };
    if ret == 0 {
        Err(last_error_to_errorkind())
    } else {
        // Calculate how many UTF-8 bytes were written
        let mut count = 0;
        for &ch in utf16[..written as usize].iter() {
            count += match ch {
                0x0000..=0x007F => 1,
                0x0080..=0x07FF => 2,
                0xDC00..=0xDFFF => 1, // Low surrogate
                _ => 3,
            };
        }
        Ok(count)
    }
}

/// Read UTF-16 from console and convert to UTF-8
fn read_u16s(
    handle: *mut c_void,
    buf: &mut [MaybeUninit<u16>],
    surrogate: &mut u16,
) -> Result<usize, ErrorKind> {
    const CTRL_Z: u16 = 0x1A;
    const CTRL_Z_MASK: u32 = 1 << CTRL_Z;
    #[repr(C)]
    struct ConsoleReadConsoleControl {
        n_length: u32,
        n_initial_chars: u32,
        dw_ctrl_wakeup_mask: u32,
        dw_control_key_state: u32,
    }
    let input_control = ConsoleReadConsoleControl {
        n_length: core::mem::size_of::<ConsoleReadConsoleControl>() as u32,
        n_initial_chars: 0,
        dw_ctrl_wakeup_mask: CTRL_Z_MASK,
        dw_control_key_state: 0,
    };
    let mut amount = 0;
    loop {
        unsafe { SetLastError(0) };
        let ret = unsafe {
            ReadConsoleW(
                handle,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as u32,
                &mut amount,
                &input_control as *const _ as *mut c_void,
            )
        };

        if ret == 0 {
            let err = unsafe { GetLastError() };
            if err == 0x3E3 {
                // ERROR_OPERATION_ABORTED
                continue;
            }
            return Err(last_error_to_errorkind());
        }
        break;
    }

    if amount > 0 && unsafe { buf[amount as usize - 1].assume_init() } == CTRL_Z {
        amount -= 1;
    }
    Ok(amount as usize)
}

/// Convert UTF-16 to UTF-8
fn utf16_to_utf8(utf16: &[u16], utf8: &mut [u8]) -> Result<usize, ErrorKind> {
    if utf16.is_empty() {
        return Ok(0);
    }
    let result = unsafe {
        WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            utf16.as_ptr(),
            utf16.len() as i32,
            utf8.as_mut_ptr() as *mut i8,
            utf8.len() as i32,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    if result == 0 {
        Err(ErrorKind::InvalidData)
    } else {
        Ok(result as usize)
    }
}

/// Get the width of a UTF-8 character from its first byte
fn utf8_char_width(first_byte: u8) -> usize {
    if first_byte <= 0x7F {
        1
    } else if first_byte <= 0xDF {
        2
    } else if first_byte <= 0xEF {
        3
    } else {
        4
    }
}

/// Convert last error to ErrorKind
fn last_error_to_errorkind() -> ErrorKind {
    let err = unsafe { GetLastError() };
    match err {
        ERROR_INVALID_HANDLE => ErrorKind::NotFound,
        0x57 => ErrorKind::InvalidInput, // ERROR_INVALID_PARAMETER
        0x6D => ErrorKind::BrokenPipe,   // ERROR_BROKEN_PIPE
        _ => ErrorKind::Other,
    }
}

impl ErrorType for Stdin {
    type Error = ErrorKind;
}

impl ErrorType for StdinLock<'_> {
    type Error = ErrorKind;
}

impl ErrorType for Stdout {
    type Error = ErrorKind;
}

impl ErrorType for StdoutLock<'_> {
    type Error = ErrorKind;
}

impl ErrorType for Stderr {
    type Error = ErrorKind;
}

impl ErrorType for StderrLock<'_> {
    type Error = ErrorKind;
}

fn handle_read() {}

impl Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let handle = get_handle(STD_INPUT_HANDLE)?;
        if !is_console(handle) {
            return Err(ErrorKind::Unsupported);
        }
        let mut bytes_copied = self.incomplete_utf8.read(buf);
        if bytes_copied == buf.len() {
            Ok(bytes_copied)
        } else if buf.len() - bytes_copied < 4 {
            // Not enough space for a full UTF-8 character
            let mut utf16_buf = [MaybeUninit::new(0); 2];
            let read = read_u16s(handle, &mut utf16_buf, &mut self.surrogate)?;
            let read_bytes = utf16_to_utf8(
                unsafe { core::slice::from_raw_parts(utf16_buf.as_ptr() as *const u16, read) },
                &mut self.incomplete_utf8.bytes,
            )?;
            self.incomplete_utf8.len = read_bytes as u8;
            bytes_copied += self
                .incomplete_utf8
                .read(&mut buf[bytes_copied..]);
            Ok(bytes_copied)
        } else {
            let mut utf16_buf = [MaybeUninit::new(0); MAX_BUFFER_SIZE / 2];
            let amount = core::cmp::min(buf.len() / 3, utf16_buf.len());
            let read = read_u16s(handle, &mut utf16_buf, &mut self.surrogate)?;
            match utf16_to_utf8(
                unsafe { core::slice::from_raw_parts(utf16_buf.as_ptr() as *const u16, read) },
                buf,
            ) {
                Ok(value) => Ok(bytes_copied + value),
                Err(e) => Err(e),
            }
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        let mut total_read = 0;
        while total_read < buf.len() {
            match self.read(&mut buf[total_read..]) {
                Ok(0) => break,
                Ok(n) => total_read += n,
                Err(e) => return Err(ReadExactError::Other(e)),
            }
        }
        if total_read == buf.len() {
            Ok(())
        } else {
            Err(ReadExactError::UnexpectedEof)
        }
    }
}

impl Read for StdinLock<'_> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let stdin = STDIN_INSTANCE.lock();
        (*stdin).read(buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        let stdin = STDIN_INSTANCE.lock();
        (*stdin).read_exact(buf)
    }
}

impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        write(
            STD_OUTPUT_HANDLE,
            buf,
            &mut self.incomplete_utf8,
            &mut self.write_mode,
        )
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        let mut written = 0;
        while written < buf.len() {
            match self.write(&buf[written..]) {
                Ok(0) => return Err(ErrorKind::WriteZero),
                Ok(n) => written += n,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl Write for StdoutLock<'_> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.inner.write_all(buf)
    }
}

impl Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        write(
            STD_ERROR_HANDLE,
            buf,
            &mut self.incomplete_utf8,
            &mut self.write_mode,
        )
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        let mut written = 0;
        while written < buf.len() {
            match self.write(&buf[written..]) {
                Ok(0) => return Err(ErrorKind::WriteZero),
                Ok(n) => written += n,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl Write for StderrLock<'_> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.inner.write_all(buf)
    }
}

/// Check if an error is a "bad file descriptor" error.
pub fn is_ebadf(err: &ErrorKind) -> bool {
    *err == ErrorKind::NotFound
}

/// Returns a writer suitable for panic output.
pub fn panic_output() -> impl Write<Error = ErrorKind> {
    Stderr::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdin_creation() {
        let stdin = Stdin::new();
        let _lock = stdin.lock();
    }

    #[test]
    fn test_stdout_creation() {
        let stdout = Stdout::new();
        let _lock = stdout.lock();
    }

    #[test]
    fn test_stderr_creation() {
        let stderr = Stderr::new();
        let _lock = stderr.lock();
    }

    #[test]
    fn test_spinlock() {
        let lock = SpinLock::new();
        let _guard = SpinLockGuard::new(&lock);
        // Lock is automatically released when guard is dropped
    }

    #[test]
    fn test_spinlock_try_lock() {
        let lock = SpinLock::new();
        assert!(lock.try_lock());
        // Second try_lock should fail
        assert!(!lock.try_lock());
        lock.unlock();
        assert!(lock.try_lock());
    }

    #[test]
    fn test_incomplete_utf8() {
        let mut incomplete = IncompleteUtf8::new();
        assert_eq!(incomplete.len, 0);

        // Test reading from empty incomplete buffer
        let mut buf = [0u8; 4];
        assert_eq!(incomplete.read(&mut buf), 0);

        // Add some bytes
        incomplete.bytes = [0xE2, 0x82, 0xAC, 0]; // UTF-8 for €
        incomplete.len = 3;

        let mut buf = [0u8; 2];
        assert_eq!(incomplete.read(&mut buf), 2);
        assert_eq!(buf, [0xE2, 0x82]);
        assert_eq!(incomplete.len, 1);
        assert_eq!(incomplete.bytes[0], 0xAC);
    }

    #[test]
    fn test_utf8_char_width() {
        assert_eq!(utf8_char_width(0x41), 1); // 'A'
        assert_eq!(utf8_char_width(0xC2), 2); // 2-byte sequence start
        assert_eq!(utf8_char_width(0xE2), 3); // 3-byte sequence start
        assert_eq!(utf8_char_width(0xF0), 4); // 4-byte sequence start
    }

    #[test]
    fn test_write_mode() {
        let handle = 0x1 as *mut c_void;
        let mode = WriteMode::Passthrough(handle);
        assert!(matches!(mode, WriteMode::Passthrough(h) if h == handle));

        let mode = WriteMode::Utf16Console(handle);
        assert!(matches!(mode, WriteMode::Utf16Console(h) if h == handle));
    }

    #[test]
    fn test_is_ebadf() {
        assert!(is_ebadf(&ErrorKind::NotFound));
        assert!(!is_ebadf(&ErrorKind::Other));
    }
}
