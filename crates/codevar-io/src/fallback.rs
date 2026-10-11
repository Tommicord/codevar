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
//! This module provides the three standard streams for platforms that have
//! no operating system behind them. On bare metal (`target_os = "none"`)
//! stdout and stderr are the platform console from `codevar-target` — the
//! debug agent's semihosting channel on ARM — while stdin, which has no
//! hardware equivalent, always reports [`ErrorKind::Unsupported`]. On other
//! exotic targets (WebAssembly and friends) every stream reports
//! [`ErrorKind::Unsupported`], because there is nothing to write to.
//!
//! A console that is missing on bare metal stops the application rather
//! than returning an error: a logger whose output is silently dropped is
//! indistinguishable from a hung program, and the project's policy for a
//! missing required component is a loud, diagnosable stop.

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
/// On bare metal this is the platform console (see the module docs); on
/// other exotic targets writes return `ErrorKind::Unsupported`.
#[derive(Debug)]
pub struct Stdout;

/// A locked reference to the [`Stdout`] handle.
#[derive(Debug)]
pub struct StdoutLock<'a> {
    _phantom: core::marker::PhantomData<&'a mut Stdout>,
}

/// A handle to the standard error stream of a process.
///
/// On bare metal this is the platform console on the error stream; on other
/// exotic targets writes return `ErrorKind::Unsupported`.
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

/// Writes `buf` to the standard output stream of this target.
///
/// Returns the number of bytes accepted, which is all of them or none: the
/// bare-metal console writes the whole buffer or stops the application, and
/// the exotic targets without a console accept nothing.
#[inline]
fn write_stdout(buf: &[u8]) -> Result<usize, ErrorKind> {
    #[cfg(target_os = "none")]
    {
        codevar_target::console_write(codevar_target::Stream::Stdout, buf);
        Ok(buf.len())
    }
    #[cfg(not(target_os = "none"))]
    {
        let _ = buf;
        Err(ErrorKind::Unsupported)
    }
}

/// Writes `buf` to the standard error stream of this target.
///
/// See [`write_stdout`] for the acceptance rules.
#[inline]
fn write_stderr(buf: &[u8]) -> Result<usize, ErrorKind> {
    #[cfg(target_os = "none")]
    {
        codevar_target::console_write(codevar_target::Stream::Stderr, buf);
        Ok(buf.len())
    }
    #[cfg(not(target_os = "none"))]
    {
        let _ = buf;
        Err(ErrorKind::Unsupported)
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
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        write_stdout(buf)
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
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        write_stdout(buf)
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
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        write_stderr(buf)
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
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        write_stderr(buf)
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
