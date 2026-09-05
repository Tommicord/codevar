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

//! Eviction Policy Error Types
//!
//! Comprehensive error types for cache eviction operations with
//! detailed context for debugging and monitoring.

use std::fmt;

/// Errors that can occur during eviction policy operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Buffer is empty, no victims available
    EmptyBuffer,
    /// Requested victim count exceeds available entries
    InsufficientEntries { requested: usize, available: usize },
    /// Index out of bounds for buffer access
    IndexOutOfBounds { index: usize, buffer_len: usize },
    /// Invalid threshold value (must be 0-10000 basis points)
    InvalidThreshold { value: u64, min: u64, max: u64 },
    /// Telemetry data invalid or inconsistent
    InvalidTelemetry { reason: &'static str },
    /// Score calculation overflow
    ScoreOverflow { frequency: u64, age_seconds: u64 },
    /// Strategy switch cooldown active
    StrategySwitchCooldown { remaining_ms: u64 },
    /// Recompute interval not elapsed
    RecomputeCooldown { remaining_ms: u64 },
    /// Configuration validation failed
    ConfigValidationFailed {
        field: &'static str,
        reason: &'static str,
    },
    /// Internal state inconsistency
    InternalInconsistency { details: String },
    /// Concurrency conflict during mutation
    ConcurrencyConflict,
    /// Telemetry-related error
    TelemetryError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyBuffer => write!(f, "Eviction buffer is empty"),
            Error::InsufficientEntries {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Insufficient entries: requested {}, available {}",
                    requested, available
                )
            }
            Error::IndexOutOfBounds { index, buffer_len } => {
                write!(
                    f,
                    "Index {} out of bounds for buffer length {}",
                    index, buffer_len
                )
            }
            Error::InvalidThreshold { value, min, max } => {
                write!(
                    f,
                    "Invalid threshold {} (must be {}-{} basis points)",
                    value, min, max
                )
            }
            Error::InvalidTelemetry { reason } => {
                write!(f, "Invalid telemetry: {}", reason)
            }
            Error::ScoreOverflow {
                frequency,
                age_seconds,
            } => {
                write!(
                    f,
                    "Score overflow: frequency={}, age_seconds={}",
                    frequency, age_seconds
                )
            }
            Error::StrategySwitchCooldown { remaining_ms } => {
                write!(
                    f,
                    "Strategy switch cooldown active: {}ms remaining",
                    remaining_ms
                )
            }
            Error::RecomputeCooldown { remaining_ms } => {
                write!(f, "Recompute cooldown active: {}ms remaining", remaining_ms)
            }
            Error::ConfigValidationFailed { field, reason } => {
                write!(f, "Config validation failed for {}: {}", field, reason)
            }
            Error::InternalInconsistency { details } => {
                write!(f, "Internal inconsistency: {}", details)
            }
            Error::ConcurrencyConflict => {
                write!(f, "Concurrency conflict during eviction")
            }
            Error::TelemetryError(msg) => write!(f, "Telemetry error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

/// Result type for eviction operations
pub type Result<T> = std::result::Result<T, Error>;