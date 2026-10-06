//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Filesystem and stdio access for the driver.
//!
//! The driver reads source files, reads stdin (the `-` input convention), and
//! writes `-o` outputs. On Unix the implementation is a thin, `no_std`-safe
//! layer over `libc` (`open`/`read`/`write`/`close` with `EINTR` retry) so the
//! compiler runs without `std`; other platforms fall back to [`std::fs`] when
//! the `std` feature is enabled and report [`FsError::Unsupported`]
//! otherwise. All reads are size-capped to bound allocations.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Largest source file (or stdin stream) the driver will read: 64 MiB.
#[cfg(any(unix, feature = "std"))]
const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

/// A filesystem or stdio failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FsError {
    /// The current platform/build has no I/O backend (non-Unix without the
    /// `std` feature). Only reachable on such targets, hence the lint
    /// allowance on Unix builds.
    #[allow(dead_code)]
    Unsupported,
    /// An OS call failed; `op` names the operation and `detail` describes it.
    Io {
        /// Name of the failed operation (`open`, `read`, `write`).
        op: &'static str,
        /// Human-readable error description.
        detail: String,
    },
    /// The stream contents are not valid UTF-8.
    InvalidUtf8,
    /// The stream exceeds the caller's byte limit.
    TooLarge {
        /// The limit in bytes that was exceeded.
        max: usize,
    },
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => {
                write!(
                    f,
                    "no filesystem backend is available on this platform \
                     (enable the `std` feature)"
                )
            }
            Self::Io { op, detail } => write!(f, "{op} failed: {detail}"),
            Self::InvalidUtf8 => write!(f, "stream does not contain valid UTF-8"),
            Self::TooLarge { max } => write!(f, "stream exceeds the {max}-byte size limit"),
        }
    }
}

impl core::error::Error for FsError {}

/// Reads the compile input: a file path, or stdin when `path` is `"-"`.
///
/// # Errors
///
/// Propagates [`FsError`] from the underlying read; non-UTF-8 input is
/// reported as [`FsError::InvalidUtf8`] because the Codevar dialect source
/// must be Unicode.
pub(crate) fn read_source(path: &str) -> Result<String, FsError> {
    if path == "-" {
        read_stdin()
    } else {
        read_file(path)
    }
}

#[cfg(unix)]
fn to_cstring(text: &str) -> Result<alloc::ffi::CString, FsError> {
    alloc::ffi::CString::new(text).map_err(|_| FsError::Io {
        op: "open",
        detail: String::from("path contains a NUL byte"),
    })
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn last_errno() -> i32 {
    0
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn last_errno() -> i32 {
    unsafe { *libc::__errno_location() }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn last_errno() -> i32 {
    unsafe { *libc::__error() }
}

#[cfg(unix)]
fn io_error(op: &'static str, errno: i32) -> FsError {
    FsError::Io {
        op,
        detail: alloc::format!("os error {errno}"),
    }
}

/// Reads up to `max` bytes from `fd`, retrying `EINTR` and stopping at EOF.
#[cfg(unix)]
fn read_fd_bytes(fd: libc::c_int, op: &'static str, max: usize) -> Result<Vec<u8>, FsError> {
    let mut out = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if read < 0 {
            let errno = last_errno();
            if errno == libc::EINTR {
                continue;
            }
            return Err(io_error(op, errno));
        }
        if read == 0 {
            break;
        }
        let read = read as usize;
        if out.len() + read > max {
            return Err(FsError::TooLarge { max });
        }
        out.extend_from_slice(&buffer[..read]);
    }
    Ok(out)
}

/// Reads `path` as raw bytes with a hard size cap.
#[cfg(unix)]
fn read_path_bytes(path: &str, max: usize) -> Result<Vec<u8>, FsError> {
    let c_path = to_cstring(path)?;
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY) };
    if fd < 0 {
        return Err(io_error("open", last_errno()));
    }
    let result = read_fd_bytes(fd, "read", max);
    unsafe {
        libc::close(fd);
    }
    result
}

#[cfg(unix)]
fn decode_utf8(bytes: Vec<u8>) -> Result<String, FsError> {
    String::from_utf8(bytes).map_err(|_| FsError::InvalidUtf8)
}

#[cfg(unix)]
pub(crate) fn read_file(path: &str) -> Result<String, FsError> {
    decode_utf8(read_path_bytes(path, MAX_SOURCE_BYTES)?)
}

#[cfg(unix)]
pub(crate) fn read_file_bytes(path: &str, max: usize) -> Result<Vec<u8>, FsError> {
    read_path_bytes(path, max)
}

#[cfg(unix)]
pub(crate) fn read_stdin() -> Result<String, FsError> {
    decode_utf8(read_fd_bytes(0, "read stdin", MAX_SOURCE_BYTES)?)
}

#[cfg(unix)]
pub(crate) fn write_file(path: &str, contents: &str) -> Result<(), FsError> {
    let c_path = to_cstring(path)?;
    let flags = libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC;
    let fd = unsafe { libc::open(c_path.as_ptr(), flags, 0o666 as libc::mode_t) };
    if fd < 0 {
        return Err(io_error("open", last_errno()));
    }
    let bytes = contents.as_bytes();
    let mut written = 0usize;
    while written < bytes.len() {
        let count = unsafe { libc::write(fd, bytes[written..].as_ptr().cast(), bytes.len() - written) };
        if count < 0 {
            let errno = last_errno();
            if errno == libc::EINTR {
                continue;
            }
            unsafe {
                libc::close(fd);
            }
            return Err(io_error("write", errno));
        }
        if count == 0 {
            unsafe {
                libc::close(fd);
            }
            return Err(FsError::Io {
                op: "write",
                detail: String::from("made no progress"),
            });
        }
        written += count as usize;
    }
    unsafe {
        libc::close(fd);
    }
    Ok(())
}

#[cfg(all(not(unix), feature = "std"))]
pub(crate) fn read_file(path: &str) -> Result<String, FsError> {
    let contents = std::fs::read_to_string(path).map_err(|error| std_error("read", error))?;
    enforce_limit(contents.as_bytes().len(), MAX_SOURCE_BYTES)?;
    Ok(contents)
}

#[cfg(all(not(unix), feature = "std"))]
pub(crate) fn read_file_bytes(path: &str, max: usize) -> Result<Vec<u8>, FsError> {
    let bytes = std::fs::read(path).map_err(|error| std_error("read", error))?;
    enforce_limit(bytes.len(), max)?;
    Ok(bytes)
}

#[cfg(all(not(unix), feature = "std"))]
pub(crate) fn read_stdin() -> Result<String, FsError> {
    let contents =
        std::io::read_to_string(std::io::stdin()).map_err(|error| std_error("read stdin", error))?;
    enforce_limit(contents.as_bytes().len(), MAX_SOURCE_BYTES)?;
    Ok(contents)
}

#[cfg(all(not(unix), feature = "std"))]
pub(crate) fn write_file(path: &str, contents: &str) -> Result<(), FsError> {
    std::fs::write(path, contents).map_err(|error| std_error("write", error))
}

#[cfg(all(not(unix), feature = "std"))]
fn std_error(op: &'static str, error: std::io::Error) -> FsError {
    if error.kind() == std::io::ErrorKind::InvalidData {
        return FsError::InvalidUtf8;
    }
    FsError::Io {
        op,
        detail: error.to_string(),
    }
}

#[cfg(all(not(unix), feature = "std"))]
fn enforce_limit(size: usize, max: usize) -> Result<(), FsError> {
    if size > max {
        return Err(FsError::TooLarge { max });
    }
    Ok(())
}

#[cfg(all(not(unix), not(feature = "std")))]
pub(crate) fn read_file(_path: &str) -> Result<String, FsError> {
    Err(FsError::Unsupported)
}

#[cfg(all(not(unix), not(feature = "std")))]
pub(crate) fn read_file_bytes(_path: &str, _max: usize) -> Result<Vec<u8>, FsError> {
    Err(FsError::Unsupported)
}

#[cfg(all(not(unix), not(feature = "std")))]
pub(crate) fn read_stdin() -> Result<String, FsError> {
    Err(FsError::Unsupported)
}

#[cfg(all(not(unix), not(feature = "std")))]
pub(crate) fn write_file(_path: &str, _contents: &str) -> Result<(), FsError> {
    Err(FsError::Unsupported)
}
