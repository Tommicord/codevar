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

#![cfg_attr(not(test), no_std)]
extern crate alloc;

mod io_cursor;
mod io_error;
mod io_impls;
mod io_std;
mod io_traits;

#[cfg(feature = "async")]
mod io_async_traits;

#[cfg(all(target_family = "unix", not(target_os = "wasi")))]
mod io_unix;

#[cfg(target_os = "windows")]
mod io_windows;

#[cfg(target_os = "wasi")]
mod io_wasi;

#[cfg(not(any(target_family = "unix", target_os = "windows", target_os = "wasi")))]
mod io_fallback;

pub use io_cursor::Cursor;
pub use io_error::{Error, ErrorKind, ErrorType, ReadExactError, SeekFrom, SliceWriteError, WriteFmtError};
pub use io_traits::{BufRead, Read, ReadReady, Seek, Write, WriteReady};

#[cfg(feature = "async")]
pub use io_async_traits::{AsyncBufRead, AsyncRead, AsyncReadReady, AsyncSeek, AsyncWrite, AsyncWriteReady};

// Re-export standard I/O types
#[cfg(all(target_family = "unix", not(target_os = "wasi")))]
pub use io_unix::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};

#[cfg(target_os = "windows")]
pub use io_windows::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};

#[cfg(target_os = "wasi")]
pub use io_wasi::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};

#[cfg(not(any(target_family = "unix", target_os = "windows", target_os = "wasi")))]
pub use io_fallback::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};
