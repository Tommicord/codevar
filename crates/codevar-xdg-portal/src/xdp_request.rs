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

//! Request handle objects
//!
//! A `RequestHandle` represents a pending portal operation that returns a
//! handle object path. The path format (since portal protocol 0.9) is:
//! `/org/freedesktop/portal/desktop/request/<sender_sanitized>/<token>` where
//! `sender_sanitized` replaces `:` and `.` with `_`.
//!
//! The handle is created at the start of a portal method that declares
//! `uses_request: true` in its `MethodInfo`. The token comes from the
//! `handle_token` option (or is generated randomly). The handle path is
//! claimed so no other request can use it.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};

use crate::xdp_app_info::AppInfo;
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{is_valid_token, generate_token, OptionMap};

pub const REQUEST_BASE_PATH: &str = "/org/freedesktop/portal/desktop/request";

/// A handle for a portal request that returns a request object path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestHandle {
    /// Full D-Bus object path of the request handle.
    pub path: String,
    /// The token component of the path.
    pub token: String,
    /// The caller's unique bus name.
    pub sender: String,
    /// Resolved application info for the caller.
    pub app_info: AppInfo,
    /// Creation time in milliseconds since epoch (from transport).
    pub created_ms: u64,
    /// Whether the request has been closed/completed.
    pub is_closed: bool,
    /// Optional implementation request path for forwarding Close.
    pub impl_request_path: Option<String>,
}

impl RequestHandle {
    /// Creates a new request handle.
    pub fn new(
        sender: &str,
        token: String,
        app_info: AppInfo,
        created_ms: u64,
    ) -> Self {
        let sanitized = sanitize_sender(sender);
        let path = format!("{}/{}/{}", REQUEST_BASE_PATH, sanitized, token);
        Self {
            path,
            token,
            sender: sender.to_string(),
            app_info,
            created_ms,
            is_closed: false,
            impl_request_path: None,
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

/// Extracts the `handle_token` from an option map, or generates one.
///
/// Mirrors `get_token` in `xdp-request.c`.
pub fn extract_handle_token(options: &OptionMap) -> Result<String, PortalError> {
    if let Some(value) = options.get("handle_token") {
        match value {
            crate::xdp_utils::PortalValue::Str(token) => {
                if is_valid_token(token) {
                    Ok(token.clone())
                } else {
                    Err(PortalError::InvalidArgument(format!(
                        "Invalid handle_token: {}",
                        token
                    )))
                }
            }
            _ => Err(PortalError::InvalidArgument(String::from(
                "handle_token must be a string",
            ))),
        }
    } else {
        generate_token()
    }
}

/// Validates a token and builds a request path, checking for collisions.
///
/// Mirrors the path construction in `xdp_request_init_invocation`.
pub fn build_request_path(
    sender: &str,
    token: &str,
    claimed_paths: &BTreeMap<String, RequestHandle>,
) -> XdpResult<String> {
    if !is_valid_token(token) {
        return Err(PortalError::InvalidArgument(format!(
            "Invalid token: {}",
            token
        )));
    }
    let sanitized = sanitize_sender(sender);
    let path = format!("{}/{}/{}", REQUEST_BASE_PATH, sanitized, token);
    if claimed_paths.contains_key(&path) {
        return Err(PortalError::Exists(format!(
            "Request path already claimed: {}",
            path
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_app_info::AppInfo;
    use crate::xdp_utils::{is_valid_token, generate_token, OptionMap, PortalValue};

    #[test]
    fn sanitizes_sender_for_path() {
        assert_eq!(sanitize_sender(":1.42"), "_1_42");
        assert_eq!(sanitize_sender("org.example.App"), "org_example_App");
        assert_eq!(sanitize_sender("a:b.c:d"), "a_b_c_d");
    }

    #[test]
    fn builds_request_path_with_token() {
        let mut claimed = BTreeMap::new();
        let path = build_request_path(":1.42", "abc123", &claimed).unwrap();
        assert_eq!(path, "/org/freedesktop/portal/desktop/request/_1_42/abc123");
    }

    #[test]
    fn rejects_invalid_token() {
        let mut claimed = BTreeMap::new();
        let err = build_request_path(":1.42", "has-dash", &claimed).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_duplicate_path() {
        let mut claimed = BTreeMap::new();
        let path = build_request_path(":1.42", "token1", &claimed).unwrap();
        let handle = RequestHandle::new(":1.42", "token1".to_string(), AppInfo::host(":1.42"), 0);
        claimed.insert(path.clone(), handle);
        let err = build_request_path(":1.42", "token1", &claimed).unwrap_err();
        assert!(matches!(err, PortalError::Exists(_)));
    }

    #[test]
    fn extracts_handle_token_from_options() {
        let mut options = OptionMap::new();
        options.insert(
            "handle_token".to_string(),
            PortalValue::Str("my_token".to_string()),
        );
        let token = extract_handle_token(&options).unwrap();
        assert_eq!(token, "my_token");
    }

    #[test]
    fn generates_token_when_missing() {
        let options = OptionMap::new();
        let token = extract_handle_token(&options).unwrap();
        assert!(is_valid_token(&token));
    }

    #[test]
    fn rejects_non_string_handle_token() {
        let mut options = OptionMap::new();
        options.insert(
            "handle_token".to_string(),
            PortalValue::U32(42),
        );
        let err = extract_handle_token(&options).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn request_handle_creation() {
        let app_info = AppInfo::host(":1.42");
        let handle = RequestHandle::new(":1.42", "test_token".to_string(), app_info, 12345);
        assert_eq!(handle.path, "/org/freedesktop/portal/desktop/request/_1_42/test_token");
        assert_eq!(handle.token, "test_token");
        assert_eq!(handle.sender, ":1.42");
        assert_eq!(handle.created_ms, 12345);
        assert!(!handle.is_closed);
        assert!(handle.impl_request_path.is_none());
    }
}