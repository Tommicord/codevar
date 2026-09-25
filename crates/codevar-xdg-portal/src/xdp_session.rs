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

//! Session handle objects, ported from `desktop-portal/xdp-session.c` and
//! `desktop-portal/xdp-session-persistence.c`.
//!
//! A `SessionHandle` represents a persistent portal session (e.g. for
//! ScreenCast, RemoteDesktop, GlobalShortcuts). The path format is:
//! `/org/freedesktop/portal/desktop/session/<sender_sanitized>/<token>`.
//!
//! Sessions are created by a portal's `CreateSession` method (which uses
//! a request handle). The session itself is exported on the bus and has
//! a `Close` method and a `Closed` signal. Unlike requests, sessions
//! persist across multiple method calls and are closed explicitly by the
//! caller or when the caller disconnects.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};

use crate::xdp_app_info::AppInfo;
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{is_valid_token, generate_token, OptionMap};

const SESSION_BASE_PATH: &str = "/org/freedesktop/portal/desktop/session";

/// A handle for a persistent portal session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionHandle {
    /// Full D-Bus object path of the session handle.
    pub path: String,
    /// The token component of the path.
    pub token: String,
    /// The caller's unique bus name.
    pub sender: String,
    /// Resolved application info for the caller.
    pub app_info: AppInfo,
    /// Creation time in milliseconds since epoch (from transport).
    pub created_ms: u64,
    /// Whether the session has been closed.
    pub is_closed: bool,
    /// Close reason (from Session.Close), if closed.
    pub reason: Option<u32>,
}

impl SessionHandle {
    /// Creates a new session handle.
    pub fn new(
        sender: &str,
        token: String,
        app_info: AppInfo,
        created_ms: u64,
    ) -> Self {
        let sanitized = sanitize_sender(sender);
        let path = format!("{}/{}/{}", SESSION_BASE_PATH, sanitized, token);
        Self {
            path,
            token,
            sender: sender.to_string(),
            app_info,
            created_ms,
            is_closed: false,
            reason: None,
        }
    }

    /// Returns the sanitized sender component used in the path.
    #[must_use]
    pub fn sanitized_sender(&self) -> String {
        sanitize_sender(&self.sender)
    }
}

/// Sanitizes a sender name for use in an object path: replaces `:` and `.` with `_`.
fn sanitize_sender(sender: &str) -> String {
    sender.replace(':', "_").replace('.', "_")
}

/// Extracts the `session_handle_token` from an option map, or generates one.
///
/// Mirrors `lookup_session_token` in `xdp-session.c`.
pub fn extract_session_token(options: &OptionMap) -> Result<String, PortalError> {
    if let Some(value) = options.get("session_handle_token") {
        match value {
            crate::xdp_utils::PortalValue::Str(token) => {
                if is_valid_token(token) {
                    Ok(token.clone())
                } else {
                    Err(PortalError::InvalidArgument(format!(
                        "Invalid session_handle_token: {}",
                        token
                    )))
                }
            }
            _ => Err(PortalError::InvalidArgument(String::from(
                "session_handle_token must be a string",
            ))),
        }
    } else {
        generate_token()
    }
}

/// Validates a token and builds a session path, checking for collisions.
pub fn build_session_path(
    sender: &str,
    token: &str,
    claimed_paths: &BTreeMap<String, SessionHandle>,
) -> XdpResult<String> {
    if !is_valid_token(token) {
        return Err(PortalError::InvalidArgument(format!(
            "Invalid token: {}",
            token
        )));
    }
    let sanitized = sanitize_sender(sender);
    let path = format!("{}/{}/{}", SESSION_BASE_PATH, sanitized, token);
    if claimed_paths.contains_key(&path) {
        return Err(PortalError::Exists(format!(
            "Session path already claimed: {}",
            path
        )));
    }
    Ok(path)
}

/// Session close reasons, matching `org.freedesktop.portal.Session.Close` reasons.
pub mod close_reason {
    /// Normal closure by the application.
    pub const NORMAL: u32 = 0;
    /// The application was disconnected.
    pub const DISCONNECTED: u32 = 1;
    /// The session was cancelled.
    pub const CANCELLED: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_app_info::AppInfo;
    use crate::xdp_utils::{is_valid_token, generate_token, OptionMap, PortalValue};

    #[test]
    fn builds_session_path_with_token() {
        let mut claimed = BTreeMap::new();
        let path = build_session_path(":1.42", "abc123", &claimed).unwrap();
        assert_eq!(path, "/org/freedesktop/portal/desktop/session/_1_42/abc123");
    }

    #[test]
    fn rejects_invalid_token() {
        let mut claimed = BTreeMap::new();
        let err = build_session_path(":1.42", "has-dash", &claimed).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_duplicate_path() {
        let mut claimed = BTreeMap::new();
        let path = build_session_path(":1.42", "token1", &claimed).unwrap();
        let handle = SessionHandle::new(":1.42", "token1".to_string(), AppInfo::host(":1.42"), 0);
        claimed.insert(path.clone(), handle);
        let err = build_session_path(":1.42", "token1", &claimed).unwrap_err();
        assert!(matches!(err, PortalError::Exists(_)));
    }

    #[test]
    fn extracts_session_token_from_options() {
        let mut options = OptionMap::new();
        options.insert(
            "session_handle_token".to_string(),
            PortalValue::Str("my_session".to_string()),
        );
        let token = extract_session_token(&options).unwrap();
        assert_eq!(token, "my_session");
    }

    #[test]
    fn generates_token_when_missing() {
        let options = OptionMap::new();
        let token = extract_session_token(&options).unwrap();
        assert!(is_valid_token(&token));
    }

    #[test]
    fn rejects_non_string_session_token() {
        let mut options = OptionMap::new();
        options.insert(
            "session_handle_token".to_string(),
            PortalValue::U32(42),
        );
        let err = extract_session_token(&options).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn session_handle_creation() {
        let app_info = AppInfo::host(":1.42");
        let handle = SessionHandle::new(":1.42", "test_token".to_string(), app_info, 12345);
        assert_eq!(handle.path, "/org/freedesktop/portal/desktop/session/_1_42/test_token");
        assert_eq!(handle.token, "test_token");
        assert_eq!(handle.sender, ":1.42");
        assert_eq!(handle.created_ms, 12345);
        assert!(!handle.is_closed);
        assert!(handle.reason.is_none());
    }
}