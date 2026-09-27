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

//! Standard I/O handles for stdin, stdout, and stderr.
//!
//! This module provides cross-platform implementations of standard I/O streams
//! that work in `no_std` environments. It supports:
//! - Unix-like systems (Linux, macOS, Android, FreeBSD, etc.) via file descriptors
//! - Windows via Win32 console APIs
//! - WASI via standard file descriptors
//! - Bare-metal/no_std environments with optional custom implementations

use crate::{ErrorType, Read, Write};

/// A trait for types that can be used as standard I/O streams.
/// This allows custom implementations for bare-metal environments.
#[allow(dead_code)]
pub trait StdIo: ErrorType + Read + Write {
    /// Try to lock the stream for exclusive access.
    /// Returns None if locking is not supported.
    fn try_lock(&mut self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }
}
