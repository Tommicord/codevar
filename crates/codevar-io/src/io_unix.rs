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

//! Unix-specific standard I/O implementation.
//!
//! This module provides implementations of stdin, stdout, and stderr for Unix-like
//! systems (Linux, macOS, Android, etc.) using file descriptors.

use crate::{ErrorKind, ErrorType, Read, ReadExactError, Write};
use core::mem::ManuallyDrop;

#[cfg(target_os = "hermit")]
use hermit_abi::{EBADF, STDERR_FILENO, STDIN_FILENO, STDOUT_FILENO};
#[cfg(any(target_family = "unix", target_os = "wasi"))]
use libc::{STDERR_FILENO, STDIN_FILENO, STDOUT_FILENO, c_int};

/// A handle to the standard input stream of a process.
///
/// Each handle is a shared reference to the standard input file descriptor.
/// Reads to this handle are not synchronized.
#[derive(Debug)]
pub struct Stdin;

/// A locked reference to the [`Stdin`] handle.
///
/// This handle implements both the [`Read`] and [`BufRead`] traits.
#[derive(Debug)]
pub struct StdinLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdin>,
}

/// A handle to the global standard output stream of the current process.
///
/// Each handle shares the standard output file descriptor.
/// Access is not synchronized by default.
#[derive(Debug)]
pub struct Stdout;

/// A locked reference to the [`Stdout`] handle.
///
/// This handle implements the [`Write`] trait.
#[derive(Debug)]
pub struct StdoutLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdout>,
}

/// A handle to the standard error stream of a process.
///
/// This handle is not buffered.
#[derive(Debug)]
pub struct Stderr;

/// A locked reference to the [`Stderr`] handle.
///
/// This handle implements the [`Write`] trait.
#[derive(Debug)]
pub struct StderrLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stderr>,
}

impl Stdin {
    /// Creates a new handle to the standard input of the current process.
    pub const fn new() -> Stdin {
        Stdin
    }

    /// Locks this handle to the standard input stream, returning a readable guard.
    ///
    /// The lock is released when the returned lock goes out of scope.
    pub fn lock(&self) -> StdinLock<'static> {
        StdinLock {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl Default for Stdin {
    fn default() -> Self {
        Self::new()
    }
}

impl Stdout {
    /// Creates a new handle to the standard output of the current process.
    pub const fn new() -> Stdout {
        Stdout
    }

    /// Locks this handle to the standard output stream, returning a writable guard.
    ///
    /// The lock is released when the returned lock goes out of scope.
    pub fn lock(&self) -> StdoutLock<'static> {
        StdoutLock {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl Default for Stdout {
    fn default() -> Self {
        Self::new()
    }
}

impl Stderr {
    /// Creates a new handle to the standard error of the current process.
    pub const fn new() -> Stderr {
        Stderr
    }

    /// Locks this handle to the standard error stream, returning a writable guard.
    ///
    /// The lock is released when the returned lock goes out of scope.
    pub fn lock(&self) -> StderrLock<'static> {
        StderrLock {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl Default for Stderr {
    fn default() -> Self {
        Self::new()
    }
}

/// File descriptor wrapper for safe reading/writing.
struct FileDesc {
    fd: i32,
}

impl FileDesc {
    /// Creates a new FileDesc from a raw file descriptor.
    ///
    /// # Safety
    /// The caller must ensure that `fd` is a valid open file descriptor.
    unsafe fn from_raw_fd(fd: i32) -> Self {
        Self { fd }
    }

    /// Reads data from the file descriptor.
    fn read(&self, buf: &mut [u8]) -> Result<usize, crate::ErrorKind> {
        let ret = unsafe { libc::read(self.fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if ret < 0 {
            let errno = unsafe { *libc::__errno_location() };
            Err(Self::errno_to_errorkind(errno))
        } else {
            Ok(ret as usize)
        }
    }

    /// Writes data to the file descriptor.
    fn write(&self, buf: &[u8]) -> Result<usize, crate::ErrorKind> {
        let ret = unsafe { libc::write(self.fd, buf.as_ptr() as *const libc::c_void, buf.len()) };
        if ret < 0 {
            let errno = unsafe { *libc::__errno_location() };
            Err(Self::errno_to_errorkind(errno))
        } else {
            Ok(ret as usize)
        }
    }

    /// Converts errno to ErrorKind.
    fn errno_to_errorkind(errno: c_int) -> crate::ErrorKind {
        match errno {
            libc::EBADF => ErrorKind::NotFound,
            libc::EPIPE => ErrorKind::BrokenPipe,
            e if e == libc::EAGAIN || e == libc::EWOULDBLOCK => ErrorKind::TimedOut,
            libc::EINTR => ErrorKind::Interrupted,
            libc::ENOMEM => ErrorKind::OutOfMemory,
            _ => ErrorKind::Other,
        }
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

impl Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        // SAFETY: STDIN_FILENO is always valid for standard input
        let fd = unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(STDIN_FILENO)) };
        fd.read(buf)
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
        let fd = unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(STDIN_FILENO)) };
        fd.read(buf)
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

impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let fd = unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(STDOUT_FILENO)) };
        fd.write(buf)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // On Unix, stdout is typically line-buffered when connected to a terminal,
        // but we don't have a way to force flush.
        // The underlying file descriptor write is synchronous.
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
        let fd = unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(STDOUT_FILENO)) };
        fd.write(buf)
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

impl Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let fd = unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(STDERR_FILENO)) };
        fd.write(buf)
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
        let fd = unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(STDERR_FILENO)) };
        fd.write(buf)
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
    fn test_errno_to_errorkind() {
        assert_eq!(FileDesc::errno_to_errorkind(libc::EBADF), ErrorKind::NotFound);
        assert_eq!(FileDesc::errno_to_errorkind(libc::EPIPE), ErrorKind::BrokenPipe);
        assert_eq!(FileDesc::errno_to_errorkind(libc::EAGAIN), ErrorKind::TimedOut);
        assert_eq!(
            FileDesc::errno_to_errorkind(libc::EWOULDBLOCK),
            ErrorKind::TimedOut
        );
        assert_eq!(FileDesc::errno_to_errorkind(libc::EINTR), ErrorKind::Interrupted);
        assert_eq!(FileDesc::errno_to_errorkind(libc::ENOMEM), ErrorKind::OutOfMemory);
        assert_eq!(FileDesc::errno_to_errorkind(9999), ErrorKind::Other);
    }
}
