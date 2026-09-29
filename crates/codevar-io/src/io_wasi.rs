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

//! WASI-specific standard I/O implementation.
//!
//! This module provides implementations of stdin, stdout, and stderr for WASI
//! using standard file descriptors.

use crate::{ErrorKind, ErrorType, Read, ReadExactError, Write};

/// WASI file descriptor constants
const STDIN_FILENO: i32 = 0;
const STDOUT_FILENO: i32 = 1;
const STDERR_FILENO: i32 = 2;

/// A handle to the standard input stream of a process.
#[derive(Debug)]
pub struct Stdin;

/// A locked reference to the [`Stdin`] handle.
#[derive(Debug)]
pub struct StdinLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdin>,
}

/// A handle to the global standard output stream of the current process.
#[derive(Debug)]
pub struct Stdout;

/// A locked reference to the [`Stdout`] handle.
#[derive(Debug)]
pub struct StdoutLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdout>,
}

/// A handle to the standard error stream of a process.
#[derive(Debug)]
pub struct Stderr;

/// A locked reference to the [`Stderr`] handle.
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
    pub fn lock(&self) -> StdinLock<'static> {
        StdinLock {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl Stdout {
    /// Creates a new handle to the standard output of the current process.
    pub const fn new() -> Stdout {
        Stdout
    }

    /// Locks this handle to the standard output stream, returning a writable guard.
    pub fn lock(&self) -> StdoutLock<'static> {
        StdoutLock {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl Stderr {
    /// Creates a new handle to the standard error of the current process.
    pub const fn new() -> Stderr {
        Stderr
    }

    /// Locks this handle to the standard error stream, returning a writable guard.
    pub fn lock(&self) -> StderrLock<'static> {
        StderrLock {
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Read from a WASI file descriptor using the `fd_read` syscall
fn fd_read(fd: i32, buf: &mut [u8]) -> Result<usize, ErrorKind> {
    #[cfg(target_os = "wasi")]
    {
        use core::mem::MaybeUninit;
        // WASI fd_read syscall
        // For WASI, we need to use the wasi libc or direct syscalls
        // This is a simplified implementation
        let iov = wasi::Iovec {
            buf: buf.as_mut_ptr(),
            buf_len: buf.len(),
        };
        let mut nread: usize = 0;
        let ret = unsafe { wasi::fd_read(fd, &iov, 1, &mut nread) };
        if ret == 0 {
            Ok(nread)
        } else {
            Err(errno_to_errorkind(ret))
        }
    }
    #[cfg(not(target_os = "wasi"))]
    {
        // Fallback for non-WASI targets
        Err(ErrorKind::Unsupported)
    }
}

/// Write to a WASI file descriptor using the `fd_write` syscall
fn fd_write(fd: i32, buf: &[u8]) -> Result<usize, ErrorKind> {
    #[cfg(target_os = "wasi")]
    {
        let ciov = wasi::Ciovec {
            buf: buf.as_ptr(),
            buf_len: buf.len(),
        };
        let mut nwritten: usize = 0;
        let ret = unsafe { wasi::fd_write(fd, &ciov, 1, &mut nwritten) };
        if ret == 0 {
            Ok(nwritten)
        } else {
            Err(errno_to_errorkind(ret))
        }
    }
    #[cfg(not(target_os = "wasi"))]
    {
        // Fallback for non-WASI targets
        Err(ErrorKind::Unsupported)
    }
}

/// Convert WASI errno to ErrorKind
fn errno_to_errorkind(errno: wasi::Errno) -> ErrorKind {
    match errno {
        wasi::ERRNO_BADF => ErrorKind::NotFound,
        wasi::ERRNO_PIPE => ErrorKind::BrokenPipe,
        wasi::ERRNO_AGAIN => ErrorKind::TimedOut,
        wasi::ERRNO_INTR => ErrorKind::Interrupted,
        wasi::ERRNO_NOMEM => ErrorKind::OutOfMemory,
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

impl Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        fd_read(STDIN_FILENO, buf)
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
        fd_read(STDIN_FILENO, buf)
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
        fd_write(STDOUT_FILENO, buf)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // WASI stdout is unbuffered, no explicit flush needed
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
        fd_write(STDOUT_FILENO, buf)
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
        fd_write(STDERR_FILENO, buf)
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
        fd_write(STDERR_FILENO, buf)
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

/// Check if an error is a "bad file descriptor" error.
pub fn is_ebadf(err: &ErrorKind) -> bool {
    *err == ErrorKind::NotFound
}

/// Standard input buffer size for WASI.
pub const STDIN_BUF_SIZE: usize = 8192;

/// Returns a writer suitable for panic output.
pub fn panic_output() -> impl Write<Error =ErrorKind> {
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
    fn test_is_ebadf() {
        assert!(is_ebadf(&ErrorKind::NotFound));
        assert!(!is_ebadf(&ErrorKind::Other));
    }
}
