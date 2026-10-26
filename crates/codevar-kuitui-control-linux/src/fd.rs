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

//! Owned descriptors and errno-based errors.
//!
//! The backend issues raw syscalls through `libc`, so it needs the two things
//! `std` normally provides: a descriptor that closes itself and an error that
//! carries the kernel's errno.

use core::ffi::c_int;
use core::fmt;

/// An owned file descriptor that closes itself on drop.
#[derive(Debug)]
pub(crate) struct Fd(c_int);

impl Fd {
    /// Takes ownership of `fd`.
    ///
    /// The caller must not close `fd` again.
    #[must_use]
    pub(crate) const fn from_raw(fd: c_int) -> Self {
        Self(fd)
    }

    /// The raw descriptor.
    #[must_use]
    pub(crate) const fn raw(&self) -> c_int {
        self.0
    }

    /// Reads into `buffer`.
    pub(crate) fn read(&self, buffer: &mut [u8]) -> Result<usize, SysError> {
        // SAFETY: `buffer` is valid for writes of its full length and the
        // descriptor is open for as long as `self` lives.
        let read = unsafe { libc::read(self.0, buffer.as_mut_ptr().cast(), buffer.len()) };
        ssize(read, "read")
    }
}

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: `self.0` is an open descriptor that no other value owns.
        let _ = unsafe { libc::close(self.0) };
    }
}

/// A system call that returned a negative result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SysError {
    /// The call that failed.
    operation: &'static str,
    /// The kernel's `errno` value.
    code: c_int,
}

impl SysError {
    /// Captures the current `errno` after `operation` failed.
    #[must_use]
    pub(crate) fn last(operation: &'static str) -> Self {
        Self {
            operation,
            code: errno(),
        }
    }

    /// Whether the failure is `code`.
    #[must_use]
    pub(crate) const fn is(self, code: c_int) -> bool {
        self.code == code
    }

    /// Whether the kernel denied access to the resource.
    #[must_use]
    pub(crate) const fn permission_denied(self) -> bool {
        self.code == libc::EACCES || self.code == libc::EPERM
    }
}

impl fmt::Display for SysError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed (errno {})", self.operation, self.code)
    }
}

/// Checks a call that returns `-1` on failure.
pub(crate) fn check_int(value: c_int, operation: &'static str) -> Result<c_int, SysError> {
    if value < 0 {
        Err(SysError::last(operation))
    } else {
        Ok(value)
    }
}

/// Checks a call that returns a byte count or `-1`.
pub(crate) fn ssize(value: libc::ssize_t, operation: &'static str) -> Result<usize, SysError> {
    if value < 0 {
        Err(SysError::last(operation))
    } else {
        Ok(value as usize)
    }
}

/// The calling thread's `errno` value.
fn errno() -> c_int {
    // SAFETY: on Linux `__errno_location` returns a valid pointer to the
    // calling thread's errno.
    unsafe { *libc::__errno_location() }
}

#[cfg(test)]
mod tests {
    use super::{Fd, SysError, check_int};

    #[test]
    fn a_negative_result_carries_the_operation() {
        let error = check_int(-1, "open").unwrap_err();
        assert_eq!(error.to_string(), "open failed (errno 0)");
    }

    #[test]
    fn only_access_errors_count_as_permission_denied() {
        let denied = SysError {
            operation: "open",
            code: libc::EACCES,
        };
        let missing = SysError {
            operation: "open",
            code: libc::ENOENT,
        };
        assert!(denied.permission_denied());
        assert!(!missing.permission_denied());
        assert!(missing.is(libc::ENOENT));
    }

    #[test]
    fn a_descriptor_reads_what_was_written() {
        let mut descriptors = [-1; 2];
        // SAFETY: `descriptors` has room for the two descriptors that `pipe2`
        // writes on success.
        let created = unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) };
        assert_eq!(created, 0);

        // SAFETY: `pipe2` succeeded, so both entries are new descriptors.
        let read = Fd::from_raw(descriptors[0]);
        let write = Fd::from_raw(descriptors[1]);
        let byte = 7u8;
        // SAFETY: `byte` is a valid source for one byte.
        let written = unsafe { libc::write(write.raw(), (&byte as *const u8).cast(), 1) };
        assert_eq!(written, 1);

        let mut buffer = [0u8; 1];
        assert_eq!(read.read(&mut buffer), Ok(1));
        assert_eq!(buffer[0], 7);
    }
}
