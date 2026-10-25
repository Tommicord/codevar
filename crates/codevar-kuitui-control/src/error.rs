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

//! The error shared by the control layer and every backend.

use alloc::string::{String, ToString};
use core::fmt;

use crate::class::DeviceClass;

/// Why an input operation could not complete.
///
/// The variants divide into two groups: *terminal* errors end a subscription,
/// and [`ControlError::Lagged`] does not. A lagged stream is still live; the
/// consumer missed events because it did not keep up with the bounded queue,
/// and it can keep reading.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ControlError {
    /// The requested class set was empty.
    NoClasses,
    /// A stream was requested with a zero-length queue.
    ZeroCapacity,
    /// The backend has no implementation for this class.
    Unsupported(DeviceClass),
    /// The operating system denied access to the device or to the
    /// permission it requires.
    PermissionDenied {
        /// What was denied, for the log and the permission prompt.
        resource: String,
    },
    /// The backend failed for a platform-specific reason.
    Backend {
        /// A human-readable explanation from the backend.
        message: String,
    },
    /// No stream uses this class, so it has no registration to release.
    NotRegistered(DeviceClass),
    /// A subscription fell behind and this many events were dropped.
    Lagged {
        /// Number of events dropped before the next delivered event.
        dropped: u64,
    },
    /// The class was unregistered by its owner; the stream has ended.
    Unregistered(DeviceClass),
    /// The subscription was cancelled or its hub was closed.
    Closed,
    /// The publisher already has the maximum number of live subscriptions.
    TooManySubscribers,
}

impl ControlError {
    /// Creates an [`ControlError::Backend`] from any displayable message.
    #[must_use]
    pub fn backend(message: impl fmt::Display) -> Self {
        Self::Backend {
            message: message.to_string(),
        }
    }

    /// Creates a [`ControlError::PermissionDenied`] for a named resource.
    #[must_use]
    pub fn permission_denied(resource: impl Into<String>) -> Self {
        Self::PermissionDenied {
            resource: resource.into(),
        }
    }
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlError::NoClasses => f.write_str("no device class was requested"),
            ControlError::ZeroCapacity => f.write_str("a subscription queue must hold at least one event"),
            ControlError::Unsupported(class) => write!(f, "the backend does not support {class} devices"),
            ControlError::PermissionDenied { resource } => {
                write!(
                    f,
                    "access to {resource} was denied; grant the application input permission"
                )
            }
            ControlError::Backend { message } => write!(f, "input backend failed: {message}"),
            ControlError::NotRegistered(class) => write!(f, "{class} devices are not registered"),
            ControlError::Lagged { dropped } => write!(f, "{dropped} input events were dropped"),
            ControlError::Unregistered(class) => write!(f, "{class} devices were unregistered"),
            ControlError::Closed => f.write_str("the input subscription is closed"),
            ControlError::TooManySubscribers => {
                f.write_str("the input publisher has no subscriber slots left")
            }
        }
    }
}

impl core::error::Error for ControlError {}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::ControlError;
    use crate::class::DeviceClass;

    #[test]
    fn backend_constructor_keeps_the_message() {
        let error = ControlError::backend("device node vanished");
        assert_eq!(
            error,
            ControlError::Backend {
                message: String::from("device node vanished")
            }
        );
    }

    #[test]
    fn permission_errors_name_the_resource() {
        let error = ControlError::permission_denied("/dev/input/event0");
        assert_eq!(
            error,
            ControlError::PermissionDenied {
                resource: String::from("/dev/input/event0")
            }
        );
    }

    #[test]
    fn display_explains_every_recoverable_case() {
        assert!(alloc::format!("{}", ControlError::NoClasses).contains("no device class"));
        assert!(alloc::format!("{}", ControlError::ZeroCapacity).contains("at least one"));
        assert!(
            alloc::format!("{}", ControlError::Unsupported(DeviceClass::Microphone)).contains("microphone")
        );
        assert!(alloc::format!("{}", ControlError::Lagged { dropped: 7 }).contains('7'));
        assert!(alloc::format!("{}", ControlError::Unregistered(DeviceClass::Keyboard)).contains("keyboard"));
        assert!(alloc::format!("{}", ControlError::TooManySubscribers).contains("subscriber"));
    }
}
