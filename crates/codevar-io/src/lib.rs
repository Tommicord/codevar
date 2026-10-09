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

//! Cross-platform I/O utilities for Codevar
//!
//! This crate provides:
//! - Standard I/O handles (stdin, stdout, stderr) with locking support
//! - Terminal detection via the [`IsTerminal`] trait
//! - Cross-platform implementations for Unix, Windows, WASI, and bare-metal
//! - Buffered and async I/O traits
//! - `no_std` compatible

#![cfg_attr(not(test), no_std)]
extern crate alloc;

mod cursor;
mod error;
mod impls;
mod terminal;
mod traits;

#[cfg(feature = "async")]
mod async_traits;

#[cfg(all(target_family = "unix", not(target_os = "wasi")))]
mod unix;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "wasi")]
mod wasi;

#[cfg(not(any(target_family = "unix", target_os = "windows", target_os = "wasi")))]
mod fallback;

pub use cursor::Cursor;
pub use error::{
    ErrorKind, ErrorType, IoError, IoResult, ReadExactError, SeekFrom, SliceWriteError, WriteFmtError,
};
pub use terminal::{IsTerminal, is_terminal};
pub use traits::{BufRead, Read, ReadReady, Seek, Write, WriteReady};

#[cfg(feature = "async")]
pub use async_traits::{AsyncBufRead, AsyncRead, AsyncReadReady, AsyncSeek, AsyncWrite, AsyncWriteReady};

// Re-export standard I/O types
#[cfg(all(target_family = "unix", not(target_os = "wasi")))]
pub use unix::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};

#[cfg(target_os = "windows")]
pub use windows::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};

#[cfg(target_os = "wasi")]
pub use wasi::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};

#[cfg(not(any(target_family = "unix", target_os = "windows", target_os = "wasi")))]
pub use fallback::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock};
