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

//! Command-line argument acquisition for the driver.
//!
//! `codevar-oclc` never calls [`std::env::args`] directly; instead arguments
//! are sourced through a layered strategy so the compiler can run in `no_std`
//! environments (modeled on how `rustc`'s `raw_args` isolates argument
//! collection from the compiler proper, and on `purestd`/`procfs`-style
//! `no_std` argv implementations):
//!
//! 1. [`from_c_args`] — copy `argc`/`argv` at the C entry point. This is what
//!    the freestanding (`no_std`) binary uses; it works on any platform whose
//!    C runtime calls `main(argc, argv)`.
//! 3. [`from_cmdline`] *(Linux)* — read `/proc/self/cmdline`, the canonical
//!    `no_std` fallback: NUL-separated bytes written by the kernel at `execve`
//!    time.
//!
//! [`raw_args`] picks the first layer available for the current build.
//!
//! # Examples
//!
//! ```
//! use codevar_oclc::argv::parse_cmdline_bytes;
//!
//! let fields = parse_cmdline_bytes(b"cc\0--emit\0tokens\0")?;
//! assert_eq!(fields, vec![b"cc".to_vec(), b"--emit".to_vec(), b"tokens".to_vec()]);
//! # Ok::<(), codevar_oclc::argv::ArgvError>(())
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use bsize::BSize;
use codevar_pathbuf::{PathBuf, PathError};
use core::ffi::{c_char, c_int};

/// Maximum accepted size of the OS command line (2 MiB).
///
/// The Linux kernel caps a command line at `MAX_ARG_STRLEN` (typically
/// 128 KiB) per argument; 2 MiB leaves generous headroom while bounding the
/// allocation a hostile or corrupted `/proc/self/cmdline` can trigger.
#[cfg(target_os = "linux")]
const CMDLINE_MAX_BYTES: usize = BSize::mb(2).bytes();

/// A failure to obtain the process command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgvError {
    /// The current platform/build provides no argument source; enable the
    /// `std` feature or pass arguments explicitly.
    Unsupported,
    /// The argument at `index` (0-based, counting `argv[0]`) is not valid
    /// Unicode. The driver requires UTF-8 arguments so diagnostics and paths
    /// stay deterministic across platforms.
    InvalidUtf8 {
        /// 0-based index of the offending argument.
        index: usize,
    },
    /// Reading `/proc/self/cmdline` failed; `detail` describes the OS error.
    Io(String),
    /// The OS command line exceeds [`CMDLINE_MAX_BYTES`].
    TooLarge,
}

impl core::fmt::Display for ArgvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported => {
                write!(
                    f,
                    "no command-line argument source is available on this platform \
                     (enable the `std` feature)"
                )
            }
            Self::InvalidUtf8 { index } => write!(f, "argument {index} is not valid Unicode"),
            Self::Io(str) => write!(f, "failed to read the OS command line: {str}"),
            Self::TooLarge => write!(f, "the OS command line exceeds the size limit"),
        }
    }
}

impl core::error::Error for ArgvError {}

/// Copies process arguments out of a C `argc`/`argv` pair.
///
/// The returned vector always includes `argv[0]` (the program name) as its
/// first element when `argc > 0`, matching [`std::env::args`]. A `null`
/// `argv` or non-positive `argc` yields an empty vector instead of trapping,
/// so callers can pass raw entry-point values unconditionally.
///
/// # Errors
///
/// Returns [`ArgvError::InvalidUtf8`] with the 0-based argument index when an
/// argument is not valid UTF-8.
///
/// # Safety
///
/// - `argv` must be null, or must point to at least `argc` readable pointers.
/// - Every non-null entry below `argc` must point to a NUL-terminated byte
///   string that outlives this call (the C runtime contract; the strings are
///   copied before returning, so the caller may free them afterward).
/// - The `argc`/`argv` pair must come from the process entry point or an
///   equivalent, still-live source; reading a freed `argv` is undefined.
pub unsafe fn from_c_args(argc: c_int, argv: *const *const c_char) -> Result<Vec<String>, ArgvError> {
    if argc <= 0 || argv.is_null() {
        return Ok(Vec::new());
    }
    let count = argc as usize;
    let mut args = Vec::with_capacity(count);
    for index in 0..count {
        let pointer = unsafe { *argv.add(index) };
        if pointer.is_null() {
            break;
        }
        let bytes = unsafe { core::ffi::CStr::from_ptr(pointer) }.to_bytes();
        let text = core::str::from_utf8(bytes).map_err(|_| ArgvError::InvalidUtf8 { index })?;
        args.push(String::from(text));
    }
    Ok(args)
}

/// Reads the command line from `/proc/self/cmdline` (Linux, `no_std`-safe).
///
/// The kernel writes each argument's bytes followed by a NUL separator;
/// spaces are ordinary bytes *inside* an argument because the shell already
/// performed word splitting before `execve`. Exactly one trailing NUL is
/// stripped before splitting so that empty arguments are preserved, and an
/// empty file (a reaped/zombie process) yields an empty vector.
///
/// # Errors
///
/// - [`ArgvError::Io`] when the file cannot be read.
/// - [`ArgvError::TooLarge`] when the file exceeds 2 MiB.
/// - [`ArgvError::InvalidUtf8`] when an argument is not valid UTF-8.
#[cfg(target_os = "linux")]
pub fn from_cmdline() -> Result<Vec<String>, ArgvError> {
    let bytes = PathBuf::from_str("/proc/self/cmdline")
        .map_err(|_| ArgvError::Io("cannot read /proc/self/cmdline".to_string()))?
        .read()
        .map_err(|e| match e {
            PathError::Unsupported => ArgvError::Unsupported,
            PathError::Empty => ArgvError::Io("empty path".to_string()),
            PathError::InvalidCharacter(_) => ArgvError::Io("invalid utf8 char in path".to_string()),
            _ => ArgvError::Io(e.to_string()),
        })?;
    if bytes.len() > CMDLINE_MAX_BYTES {
        return Err(ArgvError::TooLarge);
    }
    decode_fields(parse_cmdline_bytes(&bytes)?)
}

/// Splits raw `/proc/self/cmdline` bytes into one byte-string per argument.
///
/// The split rules are exact: a single trailing NUL (the kernel's terminator
/// for the final argument) is removed, then the remainder is split on NUL
/// **keeping empty fields**, so `b"a\0\0b\0"` yields `["a", "", "b"]` and
/// `b""` yields `[]`.
///
/// # Errors
///
/// Currently infallible; the `Result` is kept so size or encoding policies
/// added later do not break the API.
///
/// # Examples
///
/// ```
/// use codevar_oclc::argv::parse_cmdline_bytes;
///
/// let fields = parse_cmdline_bytes(b"prog\0\0-a\0b c\0")?;
/// assert_eq!(fields.len(), 4);
/// assert_eq!(fields[1], b"");
/// assert_eq!(fields[3], b"b c");
/// # Ok::<(), codevar_oclc::argv::ArgvError>(())
/// ```
pub fn parse_cmdline_bytes(bytes: &[u8]) -> Result<Vec<Vec<u8>>, ArgvError> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let body = match bytes.last() {
        Some(0) => &bytes[..bytes.len() - 1],
        _ => bytes,
    };
    Ok(body
        .split(|byte| *byte == 0)
        .map(<[u8]>::to_vec)
        .collect())
}

/// Decodes byte fields produced by [`parse_cmdline_bytes`] into strings.
///
/// # Errors
///
/// Returns [`ArgvError::InvalidUtf8`] naming the first field that is not
/// valid UTF-8.
pub(crate) fn decode_fields(fields: Vec<Vec<u8>>) -> Result<Vec<String>, ArgvError> {
    let mut args = Vec::with_capacity(fields.len());
    for (index, field) in fields.into_iter().enumerate() {
        match String::from_utf8(field) {
            Ok(text) => args.push(text),
            Err(_) => return Err(ArgvError::InvalidUtf8 { index }),
        }
    }
    Ok(args)
}

/// Returns the full command line (`argv[0]` first) for a `no_std` Linux build.
///
/// See [`raw_args`] for the resolution order.
///
/// # Errors
///
/// Propagates [`from_cmdline`]'s errors.
pub fn raw_args() -> Result<Vec<String>, ArgvError> {
    from_cmdline()
}
