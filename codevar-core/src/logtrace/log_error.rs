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
pub enum LogError {
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

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogError::NotInitialized => write!(f, "Logger is not initialized"),
            LogError::AlreadyInitialized => write!(f, "Logger is already initialized"),
            LogError::QueueCapacityExceeded {
                capacity,
                requested,
            } => {
                write!(
                    f,
                    "Queue capacity exceeded: capacity={}, requested={}",
                    capacity, requested
                )
            }
            LogError::InvalidLogLevel(level) => {
                write!(f, "Invalid log level: {}", level)
            }
            LogError::PlatformError(msg) => {
                write!(f, "Platform error: {}", msg)
            }
            LogError::FormatError(msg) => {
                write!(f, "Format error: {}", msg)
            }
            LogError::LockError(msg) => {
                write!(f, "Thread synchronization error: {}", msg)
            }
            LogError::AllocationError(msg) => {
                write!(f, "Memory allocation error: {}", msg)
            }
            LogError::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            LogError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            LogError::BufferOverflow {
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

impl std::error::Error for LogError {}

/// Result type alias for logging operations.
pub type LogResult<T> = Result<T, LogError>;
