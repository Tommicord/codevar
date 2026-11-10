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

//! Errors of the window decoration layer.

use alloc::fmt;
use alloc::string::String;
use alloc::vec::Vec;

/// The error type of every decoration operation.
///
/// A decoration fails when the native windowing system rejects a
/// request: the surface already has a role, a window style is invalid,
/// or the compositor or the window manager refused the request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecorationError {
    /// The native windowing system is not available on this target.
    Unavailable,
    /// The requested decoration kind is not supported by the backend.
    Unsupported(crate::types::DecorationKind),
    /// The compositor or the window manager refused the request.
    Refused(String),
    /// The requested size or bounds are invalid.
    InvalidBounds(crate::types::DecorationBounds),
    /// The native call failed with the given operating-system error.
    Os(String),
    /// A decoration was requested on a surface that already has a role.
    AlreadyDecorated,
    /// The backend is not initialized.
    NotInitialized,
}

impl fmt::Display for DecorationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => f.write_str("windowing system is not available"),
            Self::Unsupported(kind) => {
                write!(f, "decoration kind {kind:?} is not supported by this backend")
            }
            Self::Refused(message) => write!(f, "request was refused: {message}"),
            Self::InvalidBounds(bounds) => write!(f, "invalid decoration bounds: {bounds:?}"),
            Self::Os(message) => write!(f, "operating-system error: {message}"),
            Self::AlreadyDecorated => f.write_str("surface already has a decoration role"),
            Self::NotInitialized => f.write_str("decoration backend is not initialized"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecorationError {}

/// The result type of every decoration operation.
pub type DecorationResult<T> = Result<T, DecorationError>;

/// Converts a wayland error into a decoration error.
#[cfg(feature = "wayland")]
impl From<codevar_wayland::WlError> for DecorationError {
    fn from(error: codevar_wayland::WlError) -> Self {
        Self::Os(format!("{error}"))
    }
}

/// Converts a wayland protocol error into a decoration error.
#[cfg(feature = "wayland")]
impl From<codevar_wayland::WlProtocolError> for DecorationError {
    fn from(error: codevar_wayland::WlProtocolError) -> Self {
        Self::Refused(format!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_has_a_display_form() {
        let errors = [
            DecorationError::Unavailable,
    DecorationError::Unsupported(crate::types::DecorationKind::TitleBar),
    DecorationError::Refused("no seat".into()),
            DecorationError::InvalidBounds(crate::types::DecorationBounds::new(0, 0, 0, 0)),
            DecorationError::Os("boom".into()),
            DecorationError::AlreadyDecorated,
            DecorationError::NotInitialized,
        ];
        for error in errors {
            let _ = format!("{error}");
        }
    }
}
