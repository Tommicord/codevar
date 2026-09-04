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

//! Action cache error types.

use std::fmt;

/// Errors that can occur in action cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionCacheError {
    /// Ring buffer is full and cannot accept more entries.
    BufferFull,
    /// Invalid capacity specified.
    InvalidCapacity,
    /// File ID not found.
    FileNotFound(u64),
    /// Configuration validation failed.
    ConfigValidationFailed,
    /// Not available victims to evict
    NotAvailableVictims,
}

impl fmt::Display for ActionCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferFull => write!(f, "Action cache buffer is full"),
            Self::InvalidCapacity => write!(f, "Invalid capacity"),
            Self::FileNotFound(id) => write!(f, "File not found: {id}"),
            Self::ConfigValidationFailed => write!(f, "Config validation failed"),
            Self::NotAvailableVictims => write!(f, "Not available victims"),
        }
    }
}

impl std::error::Error for ActionCacheError {}

/// Result type for action cache operations.
pub type ActionCacheResult<T> = Result<T, ActionCacheError>;
