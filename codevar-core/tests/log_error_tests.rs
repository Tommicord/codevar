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

use codevar_core::logtrace::Error;

#[test]
fn log_error_display_not_initialized() {
    assert_eq!(
        Error::NotInitialized.to_string(),
        "Logger is not initialized"
    );
}

#[test]
fn log_error_display_already_initialized() {
    assert_eq!(
        Error::AlreadyInitialized.to_string(),
        "Logger is already initialized"
    );
}

#[test]
fn log_error_display_invalid_log_level() {
    assert_eq!(
        Error::InvalidLogLevel("bad".to_string()).to_string(),
        "Invalid log level: bad"
    );
}

#[test]
fn log_error_display_platform_error() {
    assert_eq!(
        Error::PlatformError("x".to_string()).to_string(),
        "Platform error: x"
    );
}

#[test]
fn log_error_display_format_error() {
    assert_eq!(
        Error::FormatError("fmt".to_string()).to_string(),
        "Format error: fmt"
    );
}

#[test]
fn log_error_display_lock_error() {
    assert_eq!(
        Error::LockError("lock".to_string()).to_string(),
        "Thread synchronization error: lock"
    );
}

#[test]
fn log_error_display_allocation_error() {
    assert_eq!(
        Error::AllocationError("alloc".to_string()).to_string(),
        "Memory allocation error: alloc"
    );
}

#[test]
fn log_error_display_io_error() {
    assert_eq!(
        Error::IoError("io".to_string()).to_string(),
        "I/O error: io"
    );
}

#[test]
fn log_error_display_invalid_parameter() {
    assert_eq!(
        Error::InvalidParameter("bad".to_string()).to_string(),
        "Invalid parameter: bad"
    );
}

#[test]
fn log_error_display_buffer_overflow() {
    assert_eq!(
        Error::BufferOverflow {
            buffer_size: 4,
            required_size: 8
        }
        .to_string(),
        "Buffer overflow: buffer_size=4, required_size=8"
    );
}

#[test]
fn log_error_display_queue_capacity_exceeded() {
    assert_eq!(
        Error::QueueCapacityExceeded {
            capacity: 16,
            requested: 33
        }
        .to_string(),
        "Queue capacity exceeded: capacity=16, requested=33"
    );
}

#[test]
fn log_error_clone_preserves_value() {
    let e = Error::InvalidParameter("value".to_string());
    let clone = e.clone();
    assert_eq!(clone, e);
}

#[test]
fn log_error_partial_eq_for_same_variant() {
    assert_eq!(Error::NotInitialized, Error::NotInitialized);
    assert_ne!(Error::NotInitialized, Error::AlreadyInitialized);
}

#[test]
fn log_error_debug_string_is_present() {
    let text = format!("{:?}", Error::FormatError("fmt".to_string()));
    assert!(text.contains("FormatError"));
}

#[test]
fn log_error_is_error_trait_compatible() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<Error>();
}

#[test]
fn queue_capacity_error_has_expected_state() {
    let e = Error::QueueCapacityExceeded {
        capacity: 2,
        requested: 5,
    };
    match e {
        Error::QueueCapacityExceeded {
            capacity,
            requested,
        } => {
            assert_eq!(capacity, 2);
            assert_eq!(requested, 5);
        }
        _ => panic!("unexpected variant"),
    }
}

#[test]
fn buffer_overflow_error_has_expected_state() {
    let e = Error::BufferOverflow {
        buffer_size: 32,
        required_size: 64,
    };
    match e {
        Error::BufferOverflow {
            buffer_size,
            required_size,
        } => {
            assert_eq!(buffer_size, 32);
            assert_eq!(required_size, 64);
        }
        _ => panic!("unexpected variant"),
    }
}

#[test]
fn invalid_level_error_carries_message() {
    let e = Error::InvalidLogLevel("invalid".to_string());
    assert!(matches!(e, Error::InvalidLogLevel(_)));
}

#[test]
fn io_error_carries_message() {
    let e = Error::IoError("network".to_string());
    assert!(matches!(e, Error::IoError(_)));
}
