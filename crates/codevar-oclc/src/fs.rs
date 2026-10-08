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
//! Every read and write the driver performs goes through this module so the
//! path validation, the UTF-8 policy, and the size limit live in exactly one
//! place. Backed by [`codevar_pathbuf`] for files and by [`codevar_io`] for
//! standard input, both of which work in `no_std` builds.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use bsize::BSize;
use codevar_io::{ErrorKind, Read, Stdin};
use codevar_pathbuf::{PathBuf, PathError};

/// Largest source file (or stdin stream) the driver will read: 64 MiB.
const MAX_SOURCE_BYTES: usize = BSize::mb(64).bytes();

/// A filesystem or stdio failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FsError {
    /// The path was rejected or the OS call failed.
    Path(PathError),
    /// The stream does not contain valid UTF-8.
    InvalidUtf8,
    /// The stream exceeds [`MAX_SOURCE_BYTES`].
    TooLarge {
        /// The limit in bytes that was exceeded.
        max: usize,
    },
    /// A standard-input read failed; `op` names the operation and `detail`
    /// describes the underlying error.
    Io {
        /// Name of the failed operation (`read stdin`).
        op: &'static str,
        /// Human-readable error description.
        detail: String,
    },
}

impl From<PathError> for FsError {
    #[inline]
    fn from(error: PathError) -> Self {
        Self::Path(error)
    }
}

impl core::fmt::Display for FsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Path(error) => write!(f, "{error}"),
            Self::InvalidUtf8 => f.write_str("stream does not contain valid UTF-8"),
            Self::TooLarge { max } => write!(f, "stream exceeds the {max}-byte size limit"),
            Self::Io { op, detail } => write!(f, "{op} failed: {detail}"),
        }
    }
}

impl core::error::Error for FsError {}

/// Reads `path` as UTF-8 text.
///
/// # Errors
///
/// Returns [`FsError::Path`] when the path is invalid or the file cannot be
/// read, [`FsError::TooLarge`] when it exceeds [`MAX_SOURCE_BYTES`], and
/// [`FsError::InvalidUtf8`] when its contents are not valid UTF-8 (Codevar
/// source must be Unicode).
pub(crate) fn read_file(path: &str) -> Result<String, FsError> {
    let bytes = PathBuf::from_str(path)?.read()?;
    decode(bytes)
}

/// Reads the compile input: the file at `path`, or standard input when
/// `path` is the `-` convention.
///
/// # Errors
///
/// Same as [`read_file`]; stdin failures surface as [`FsError::Io`].
pub(crate) fn read_source(path: &str) -> Result<String, FsError> {
    if path == "-" {
        read_stdin()
    } else {
        read_file(path)
    }
}

/// Writes `contents` to `path`, creating it or truncating it.
///
/// # Errors
///
/// Returns [`FsError::Path`] when the path is invalid or the file cannot be
/// written.
pub(crate) fn write_file(path: &str, contents: &str) -> Result<(), FsError> {
    PathBuf::from_str(path)?.write(contents.as_bytes())?;
    Ok(())
}

/// Drains standard input and decodes it as UTF-8.
///
/// Interrupted reads are retried; a short read is normal and simply continues
/// the loop until the stream reports EOF.
fn read_stdin() -> Result<String, FsError> {
    let mut stdin = Stdin::new();
    let mut bytes: Vec<u8> = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let read = match stdin.read(&mut buffer) {
            Ok(read) => read,
            Err(ErrorKind::Interrupted) => continue,
            Err(error) => {
                return Err(FsError::Io {
                    op: "read stdin",
                    detail: error.to_string(),
                });
            }
        };
        if read == 0 {
            break;
        }
        if bytes.len() + read > MAX_SOURCE_BYTES {
            return Err(FsError::TooLarge {
                max: MAX_SOURCE_BYTES,
            });
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    decode(bytes)
}

/// Enforces the size limit and decodes `bytes` as UTF-8.
fn decode(bytes: Vec<u8>) -> Result<String, FsError> {
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(FsError::TooLarge {
            max: MAX_SOURCE_BYTES,
        });
    }
    String::from_utf8(bytes).map_err(|_| FsError::InvalidUtf8)
}
