//! Copyright 2026 Codevar
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

//! Error types for the logging system.

use std::fmt;

/// Comprehensive error types for logging operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Logger is not initialized
    NotInitialized,
    /// Logger is already initialized
    AlreadyInitialized,
    /// Queue capacity exceeded
    QueueCapacityExceeded {
        /// Current capacity
        capacity: usize,
        /// Requested capacity
        requested: usize,
    },
    /// Invalid log level
    InvalidLogLevel(String),
    /// Platform-specific error
    PlatformError(String),
    /// Format error
    FormatError(String),
    /// Lock synchronization error
    LockError(String),
    /// Memory allocation error
    AllocationError(String),
    /// I/O error
    IoError(String),
    /// Invalid parameter
    InvalidParameter(String),
    /// Buffer overflow
    BufferOverflow {
        /// Buffer size
        buffer_size: usize,
        /// Required size
        required_size: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotInitialized => write!(f, "Logger is not initialized"),
            Error::AlreadyInitialized => write!(f, "Logger is already initialized"),
            Error::QueueCapacityExceeded {
                capacity,
                requested,
            } => {
                write!(
                    f,
                    "Queue capacity exceeded: capacity={}, requested={}",
                    capacity, requested
                )
            }
            Error::InvalidLogLevel(level) => {
                write!(f, "Invalid log level: {}", level)
            }
            Error::PlatformError(msg) => {
                write!(f, "Platform error: {}", msg)
            }
            Error::FormatError(msg) => {
                write!(f, "Format error: {}", msg)
            }
            Error::LockError(msg) => {
                write!(f, "Thread synchronization error: {}", msg)
            }
            Error::AllocationError(msg) => {
                write!(f, "Memory allocation error: {}", msg)
            }
            Error::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            Error::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            Error::BufferOverflow {
                buffer_size,
                required_size,
            } => {
                write!(
                    f,
                    "Buffer overflow: buffer_size={}, required_size={}",
                    buffer_size, required_size
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// Result type alias for logging operations.
pub type Result<T> = std::result::Result<T, Error>;
