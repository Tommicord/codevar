//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Portal error domain shared by every Codevar portal.
//!
//! Mirrors `XDG_DESKTOP_PORTAL_ERROR` from the C reference
//! (`shared/xdp-utils.c`): the seven `org.freedesktop.portal.Error.*`
//! D-Bus error names that portal methods return on failure. The type
//! carries a human readable message so it can be converted to and from
//! [`DbusError::Remote`] without losing information.

use alloc::string::{String, ToString};
use core::fmt;

use codevar_dbus::DbusError;

/// Result type for portal operations.
pub type XdpResult<T> = Result<T, PortalError>;

/// A portal-level failure with an `org.freedesktop.portal.Error.*`
/// error code and a human readable message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortalError {
    /// `org.freedesktop.portal.Error.Failed` — an unspecified failure.
    Failed(String),
    /// `org.freedesktop.portal.Error.InvalidArgument` — an argument
    /// failed validation.
    InvalidArgument(String),
    /// `org.freedesktop.portal.Error.NotFound` — a referenced object
    /// does not exist.
    NotFound(String),
    /// `org.freedesktop.portal.Error.Exists` — the object already
    /// exists.
    Exists(String),
    /// `org.freedesktop.portal.Error.NotAllowed` — the operation is
    /// not permitted.
    NotAllowed(String),
    /// `org.freedesktop.portal.Error.Cancelled` — the operation was
    /// cancelled.
    Cancelled(String),
    /// `org.freedesktop.portal.Error.WindowDestroyed` — the parent
    /// window went away.
    WindowDestroyed(String),
}

impl PortalError {
    /// Returns the D-Bus error name registered for this code, e.g.
    /// `org.freedesktop.portal.Error.Failed`.
    #[inline]
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Failed(_) => "org.freedesktop.portal.Error.Failed",
            Self::InvalidArgument(_) => "org.freedesktop.portal.Error.InvalidArgument",
            Self::NotFound(_) => "org.freedesktop.portal.Error.NotFound",
            Self::Exists(_) => "org.freedesktop.portal.Error.Exists",
            Self::NotAllowed(_) => "org.freedesktop.portal.Error.NotAllowed",
            Self::Cancelled(_) => "org.freedesktop.portal.Error.Cancelled",
            Self::WindowDestroyed(_) => "org.freedesktop.portal.Error.WindowDestroyed",
        }
    }

    /// Returns the human readable message attached to the error.
    #[inline]
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Failed(message)
            | Self::InvalidArgument(message)
            | Self::NotFound(message)
            | Self::Exists(message)
            | Self::NotAllowed(message)
            | Self::Cancelled(message)
            | Self::WindowDestroyed(message) => message,
        }
    }

    /// Builds an error from a D-Bus error `name` and `message`.
    ///
    /// Returns `None` when `name` is not part of the portal error
    /// domain.
    #[inline]
    #[must_use]
    pub fn from_name(name: &str, message: String) -> Option<Self> {
        let error = match name {
            "org.freedesktop.portal.Error.Failed" => Self::Failed(message),
            "org.freedesktop.portal.Error.InvalidArgument" => Self::InvalidArgument(message),
            "org.freedesktop.portal.Error.NotFound" => Self::NotFound(message),
            "org.freedesktop.portal.Error.Exists" => Self::Exists(message),
            "org.freedesktop.portal.Error.NotAllowed" => Self::NotAllowed(message),
            "org.freedesktop.portal.Error.Cancelled" => Self::Cancelled(message),
            "org.freedesktop.portal.Error.WindowDestroyed" => Self::WindowDestroyed(message),
            _ => return None,
        };
        Some(error)
    }

    /// Converts the error into a D-Bus remote error that can be sent
    /// back to a caller with `Connection::send`.
    #[must_use]
    pub fn into_dbus_error(self) -> DbusError {
        let name = self
            .name()
            .to_string();
        DbusError::remote(name, self.message())
    }
}

impl fmt::Display for PortalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name(), self.message())
    }
}

impl core::error::Error for PortalError {}

impl From<DbusError> for PortalError {
    fn from(error: DbusError) -> Self {
        if let DbusError::Remote { name, message } = &error
            && let Some(portal) = Self::from_name(name, message.clone())
        {
            return portal;
        }
        Self::Failed(error.to_string())
    }
}

impl From<PortalError> for DbusError {
    fn from(error: PortalError) -> Self {
        error.into_dbus_error()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // unwrap() is safe here: every assertion below pins a concrete
    // error built by the code under test.
    #[test]
    fn names_match_the_c_error_domain() {
        let pairs = [
            (
                PortalError::Failed(String::new()),
                "org.freedesktop.portal.Error.Failed",
            ),
            (
                PortalError::InvalidArgument(String::new()),
                "org.freedesktop.portal.Error.InvalidArgument",
            ),
            (
                PortalError::NotFound(String::new()),
                "org.freedesktop.portal.Error.NotFound",
            ),
            (
                PortalError::Exists(String::new()),
                "org.freedesktop.portal.Error.Exists",
            ),
            (
                PortalError::NotAllowed(String::new()),
                "org.freedesktop.portal.Error.NotAllowed",
            ),
            (
                PortalError::Cancelled(String::new()),
                "org.freedesktop.portal.Error.Cancelled",
            ),
            (
                PortalError::WindowDestroyed(String::new()),
                "org.freedesktop.portal.Error.WindowDestroyed",
            ),
        ];
        for (error, name) in pairs {
            assert_eq!(error.name(), name);
            let roundtrip = PortalError::from_name(name, String::from("m")).unwrap();
            assert_eq!(
                roundtrip,
                PortalError::from_name(error.name(), String::from("m")).unwrap()
            );
        }
    }

    #[test]
    fn from_name_rejects_foreign_errors() {
        assert!(PortalError::from_name("org.freedesktop.DBus.Error.Failed", String::new()).is_none());
    }

    #[test]
    fn converts_to_and_from_dbus_errors() {
        let error = PortalError::InvalidArgument(String::from("bad option"));
        let dbus = error
            .clone()
            .into_dbus_error();
        assert!(matches!(dbus, DbusError::Remote { .. }));
        if let DbusError::Remote { name, message } = &dbus {
            assert_eq!(name, "org.freedesktop.portal.Error.InvalidArgument");
            assert_eq!(message, "bad option");
        }
        let back = PortalError::from(dbus);
        assert_eq!(back, error);

        let other = PortalError::from(DbusError::Timeout);
        assert!(matches!(other, PortalError::Failed(_)));
    }
}
