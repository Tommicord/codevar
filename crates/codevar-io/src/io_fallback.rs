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

//! Fallback standard I/O implementation for bare-metal/no_std environments.
//!
//! This module provides minimal implementations of stdin, stdout, and stderr
//! for platforms that don't have standard OS support (bare-metal, embedded, etc.).
//! These implementations return appropriate errors indicating the operations
//! are not supported.

use crate::{ErrorKind, ErrorType, Read, ReadExactError, Write};

/// A handle to the standard input stream of a process.
///
/// On bare-metal platforms, standard input is typically not available.
/// Read operations will return `ErrorKind::Unsupported`.
#[derive(Debug)]
pub struct Stdin;

/// A locked reference to the [`Stdin`] handle.
#[derive(Debug)]
pub struct StdinLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdin>,
}

/// A handle to the global standard output stream of the current process.
///
/// On bare-metal platforms, standard output may be connected to a serial port,
/// display, or other output device. Write operations will return
/// `ErrorKind::Unsupported` unless a custom implementation is provided.
#[derive(Debug)]
pub struct Stdout;

/// A locked reference to the [`Stdout`] handle.
#[derive(Debug)]
pub struct StdoutLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdout>,
}

/// A handle to the standard error stream of a process.
///
/// On bare-metal platforms, standard error is typically the same as stdout.
/// Write operations will return `ErrorKind::Unsupported` unless a custom
/// implementation is provided.
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
    fn read(&mut self, _buf: &mut [u8]) -> Result<usize, Self::Error> {
        // Standard input is not available on bare-metal platforms
        Err(ErrorKind::Unsupported)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        // Can't read exact on unsupported input
        let _ = self.read(buf);
        Err(ReadExactError::UnexpectedEof)
    }
}

impl Read for StdinLock<'_> {
    fn read(&mut self, _buf: &mut [u8]) -> Result<usize, Self::Error> {
        Err(ErrorKind::Unsupported)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        let _ = self.read(buf);
        Err(ReadExactError::UnexpectedEof)
    }
}

impl Write for Stdout {
    fn write(&mut self, _buf: &[u8]) -> Result<usize, Self::Error> {
        // Standard output is not available on bare-metal platforms
        // Users should provide their own implementation via the StdIo trait
        Err(ErrorKind::Unsupported)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.write(buf)?;
        Ok(())
    }
}

impl Write for StdoutLock<'_> {
    fn write(&mut self, _buf: &[u8]) -> Result<usize, Self::Error> {
        Err(ErrorKind::Unsupported)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.write(buf)?;
        Ok(())
    }
}

impl Write for Stderr {
    fn write(&mut self, _buf: &[u8]) -> Result<usize, Self::Error> {
        // Standard error is not available on bare-metal platforms
        Err(ErrorKind::Unsupported)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.write(buf)?;
        Ok(())
    }
}

impl Write for StderrLock<'_> {
    fn write(&mut self, _buf: &[u8]) -> Result<usize, Self::Error> {
        Err(ErrorKind::Unsupported)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.write(buf)?;
        Ok(())
    }
}

/// Check if an error is a "bad file descriptor" error.
pub fn is_ebadf(err: &ErrorKind) -> bool {
    *err == ErrorKind::NotFound
}

/// Standard input buffer size for fallback.
pub const STDIN_BUF_SIZE: usize = 0;

/// Returns a writer suitable for panic output.
/// On bare-metal, this will return an unsupported writer.
pub fn panic_output() -> impl Write<Error =ErrorKind> {
    Stderr::new()
}

/// Trait for providing custom standard I/O implementations on bare-metal platforms.
///
/// Implement this trait to provide platform-specific standard I/O.
/// The implementations can then be used through the standard I/O functions.
pub trait StdIoCustom {
    /// Read from standard input.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ErrorKind>;

    /// Write to standard output.
    fn write(&mut self, buf: &[u8]) -> Result<usize, ErrorKind>;

    /// Write to standard error.
    fn write_err(&mut self, buf: &[u8]) -> Result<usize, ErrorKind>;

    /// Flush standard output.
    fn flush_out(&mut self) -> Result<(), ErrorKind>;

    /// Flush standard error.
    fn flush_err(&mut self) -> Result<(), ErrorKind>;
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
    fn test_stdin_read_unsupported() {
        let mut stdin = Stdin::new();
        let mut buf = [0u8; 10];
        let result = stdin.read(&mut buf);
        assert_eq!(result, Err(ErrorKind::Unsupported));
    }

    #[test]
    fn test_stdout_write_unsupported() {
        let mut stdout = Stdout::new();
        let result = stdout.write(b"test");
        assert_eq!(result, Err(ErrorKind::Unsupported));
    }

    #[test]
    fn test_stderr_write_unsupported() {
        let mut stderr = Stderr::new();
        let result = stderr.write(b"test");
        assert_eq!(result, Err(ErrorKind::Unsupported));
    }
}
