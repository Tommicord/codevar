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

//! Backend-agnostic window abstraction for the launcher.
//!
//! - [`Window`]: an object-safe trait for one open window, so
//!   backends can be handed out as `Box<dyn Window>`, the same
//!   opaque-handle shape GLFW and SDL use;
//! - [`WndBackend`]: a factory trait per windowing system, listed in
//!   [`backends`] and probed by [`open`], mirroring SDL's video
//!   driver enumeration;
//! - plain data types ([`WndConfig`], [`WndSize`], [`WndEvent`],
//!   [`WndError`]) that are backend independent, with
//!   `#[non_exhaustive]` enums so future backends and events do not
//!   break callers, following `raw-window-handle`'s
//!   `RawWindowHandle` policy.
//!
//! The compiled-in backends live in the platform modules
//! (`wnd_linux` on Unix); this file only owns the abstraction.

use crate::wnd_linux;
use alloc::boxed::Box;
use alloc::string::String;
use codevar_env::{ENV_APP_ID, ENV_NAME};
use codevar_time_core::TimeDuration;

/// Identifies a windowing backend, analogous to SDL's video driver
/// names and the platform variants of `raw-window-handle`'s
/// `RawWindowHandle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WndBackendId {
    /// Native Wayland window driven through `xdg-shell`.
    Wayland,
}

impl WndBackendId {
    /// Returns the low-ASCII driver name of the backend
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wayland => "wayland",
        }
    }
}

impl core::fmt::Display for WndBackendId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// A window size in surface-local pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WndSize {
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
}

impl WndSize {
    /// Creates a size from `width` by `height` pixels.
    #[must_use]
    pub const fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }
}

impl core::fmt::Display for WndSize {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Description of the window the launcher wants to open.
///
/// The compositor may override [`Self::size`]; the opened window
/// reports the size it actually got through [`Window::size`] and
/// [`WndEvent::Resized`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WndConfig {
    /// Title shown in the window decorations.
    pub title: String,
    /// Application id announced to the compositor, typically the
    /// desktop file identifier (`net.codevar`).
    pub app_id: String,
    /// Requested window size in surface-local pixels.
    pub geom: WndSize,
}

impl WndConfig {
    /// Creates a configuration for a `size` window titled `title`.
    #[must_use]
    pub fn new(title: &str, app_id: &str, geom: WndSize) -> Self {
        Self {
            title: String::from(title),
            app_id: String::from(app_id),
            geom,
        }
    }
}

impl Default for WndConfig {
    fn default() -> Self {
        Self::new(ENV_NAME, ENV_APP_ID, WndSize::new(1024, 768))
    }
}

/// Window-level events reported by [`Window::poll`].
///
/// The enum is `#[non_exhaustive]`: future backends may report more
/// events (input, focus, frame callbacks) without breaking callers,
/// which must therefore keep a wildcard match arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WndEvent {
    /// The user or the compositor asked the window to close; the
    /// window reports [`Window::is_open`] `false` from then on.
    CloseRequested,
    /// The compositor resized the window to the carried size.
    Resized(WndSize),
}

/// Errors reported by window operations.
///
/// Failures are classified so callers can decide whether to retry:
/// [`Self::is_transient`] tells the two apart, and every variant keeps
/// the backend that reported it in `id` (except [`Self::Unsupported`],
/// which by definition names no single backend).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WndError {
    /// No compiled-in backend can serve the current session.
    Unsupported(String),
    /// The compositor socket of the session could not be reached; the
    /// session may still be starting up, so the attempt is retryable.
    NoCompositor {
        /// Backend that failed to reach its compositor.
        id: WndBackendId,
        /// Why the connection attempt failed.
        message: String,
    },
    /// The compositor accepted the connection but did not answer the
    /// handshake in time.
    Timeout {
        /// Backend whose handshake stalled.
        id: WndBackendId,
        /// Which protocol answer was missing.
        message: String,
    },
    /// The connection to the compositor was lost mid-flight.
    Disconnected {
        /// Backend that lost its connection.
        id: WndBackendId,
        /// How the loss was observed.
        message: String,
    },
    /// The compositor reported a protocol-level rejection
    /// (`wl_display.error` or a malformed request).
    Protocol {
        /// Backend the compositor complained about.
        id: WndBackendId,
        /// The compositor's complaint.
        message: String,
    },
    /// A backend is present but the operation failed; `id` records
    /// which one so logs keep the provenance of the failure.
    Backend {
        /// Backend that reported the failure.
        id: WndBackendId,
        /// Backend-specific description of the failure.
        message: String,
    },
}

impl WndError {
    /// Creates a [`Self::Backend`] error attributed to `id`.
    #[must_use]
    pub fn backend(id: WndBackendId, message: impl Into<String>) -> Self {
        Self::Backend {
            id,
            message: message.into(),
        }
    }

    /// Returns `true` when the failure may resolve on its own and is
    /// therefore worth retrying.
    ///
    /// The session may still be coming up ([`Self::NoCompositor`]),
    /// the compositor may answer slowly ([`Self::Timeout`]), or a
    /// local resource hiccup may clear ([`Self::Backend`]). By
    /// contrast [`Self::Unsupported`], [`Self::Protocol`] and
    /// [`Self::Disconnected`] are the compositor's permanent verdicts
    /// and are never retried.
    #[must_use]
    pub const fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::NoCompositor { .. } | Self::Timeout { .. } | Self::Backend { .. }
        )
    }
}

impl core::fmt::Display for WndError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "no usable windowing backend: {reason}"),
            Self::NoCompositor { id, message } => write!(f, "{id}: no compositor connection: {message}"),
            Self::Timeout { id, message } => write!(f, "{id} timed out: {message}"),
            Self::Disconnected { id, message } => write!(f, "{id} disconnected: {message}"),
            Self::Protocol { id, message } => write!(f, "{id} protocol error: {message}"),
            Self::Backend { id, message } => write!(f, "{id} backend: {message}"),
        }
    }
}

impl core::error::Error for WndError {}

/// An open window on the user's display.
///
/// The trait is object-safe so backends can be handed out as
/// `Box<dyn Window>`, the opaque-handle shape GLFW (`GLFWwindow*`)
/// and SDL (`SDL_Window*`) expose. Events are delivered through a
/// sink on [`Self::poll`], mirroring winit's `WindowEvent` delivery
/// and the SDL main-loop convention of draining the queue once per
/// frame.
pub trait Window {
    /// Returns the backend that created the window.
    #[must_use]
    fn backend(&self) -> WndBackendId;

    /// Returns `true` while the window is mapped and neither this
    /// process nor the compositor has closed it.
    #[must_use]
    fn is_open(&self) -> bool;

    /// Destroys the window; later [`Self::poll`] calls do nothing.
    fn close(&mut self);

    /// Returns the window size in surface-local pixels.
    #[must_use]
    fn size(&self) -> WndSize;

    /// Sets the title shown in the window decorations.
    ///
    /// Reports success without sending anything once the window has
    /// been closed.
    ///
    /// # Errors
    ///
    /// Returns a classified [`WndError`] when the request could not be
    /// queued on the connection; [`WndError::is_transient`] reports
    /// whether retrying may help.
    fn set_title(&mut self, title: &str) -> Result<(), WndError>;

    /// Dispatches protocol events for up to `timeout` and reports the
    /// window-level changes they caused to `sink`.
    ///
    /// Returns the number of protocol events dispatched. A `timeout`
    /// of `None` waits until at least one event arrived.
    ///
    /// # Errors
    ///
    /// Returns a classified [`WndError`] when the connection failed;
    /// [`WndError::is_transient`] reports whether the poll is worth
    /// retrying, and the window must be discarded once a permanent
    /// error ([`WndError::Disconnected`], [`WndError::Protocol`])
    /// surfaced.
    fn poll(
        &mut self,
        timeout: Option<TimeDuration>,
        sink: &mut dyn FnMut(WndEvent),
    ) -> Result<usize, WndError>;
}

/// A windowing backend compiled into the launcher.
pub trait WndBackend {
    /// Returns the identifier of this backend.
    #[must_use]
    fn id(&self) -> WndBackendId;

    /// Returns `true` when the current session can host windows of
    /// this backend (for Wayland: a compositor socket is reachable).
    #[must_use]
    fn is_available(&self) -> bool;

    /// Opens a window matching `config`.
    ///
    /// # Errors
    ///
    /// Returns a classified [`WndError`]: [`WndError::NoCompositor`]
    /// when the session's socket is unreachable, [`WndError::Timeout`]
    /// when the handshake stalled, [`WndError::Unsupported`] when the
    /// compositor lacks a required global, and [`WndError::Backend`]
    /// for local resource failures. [`WndError::is_transient`] says
    /// whether the attempt may succeed later.
    fn open(&self, config: &WndConfig) -> Result<Box<dyn Window>, WndError>;
}

/// Returns the windowing backends compiled into the launcher, in
/// probe order.
#[must_use]
pub fn backends() -> &'static [&'static dyn WndBackend] {
    wnd_linux::BACKENDS
}

/// Opens a window on the first available backend of [`backends`].
///
/// Backends that report [`WndBackend::is_available`] `false` are
/// skipped; if every candidate fails, the last error is returned so
/// logs keep the most relevant failure.
///
/// # Errors
///
/// Returns [`WndError::Unsupported`] when no backend is available in
/// this session, or the chosen backend's error otherwise.
pub fn open(config: &WndConfig) -> Result<Box<dyn Window>, WndError> {
    let mut failure: Option<WndError> = None;
    for backend in backends() {
        if !backend.is_available() {
            continue;
        }
        match backend.open(config) {
            Ok(window) => return Ok(window),
            Err(error) => failure = Some(error),
        }
    }
    Err(failure.unwrap_or_else(|| {
        WndError::Unsupported(String::from("no backend reported a usable display session"))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_the_codevar_toplevel() {
        let config = WndConfig::default();
        assert_eq!(config.title, ENV_NAME);
        assert_eq!(config.app_id, ENV_APP_ID);
        assert_eq!(config.geom, WndSize::new(1024, 768));
    }

    #[test]
    fn wayland_backend_is_registered_and_named_like_sdl_drivers() {
        let backends = backends();
        assert!(!backends.is_empty(), "the launcher ships a backend");
        let wayland = backends
            .iter()
            .find(|backend| backend.id() == WndBackendId::Wayland)
            .expect("the Wayland backend must be registered");
        assert_eq!(wayland.id().name(), "wayland");
        assert_eq!(wayland.id().to_string(), "wayland");
    }

    #[test]
    fn backend_errors_keep_their_provenance() {
        let error = WndError::backend(WndBackendId::Wayland, "handshake failed");
        assert_eq!(
            error,
            WndError::Backend {
                id: WndBackendId::Wayland,
                message: String::from("handshake failed"),
            }
        );
        assert_eq!(error.to_string(), "wayland backend: handshake failed");
    }

    #[test]
    fn unsupported_errors_read_as_backend_problems() {
        let error = WndError::Unsupported(String::from("headless session"));
        assert_eq!(error.to_string(), "no usable windowing backend: headless session");
    }

    #[test]
    fn transient_errors_are_only_the_ones_that_may_clear() {
        let id = WndBackendId::Wayland;
        for transient in [
            WndError::NoCompositor {
                id,
                message: String::from("socket refused"),
            },
            WndError::Timeout {
                id,
                message: String::from("no configure"),
            },
            WndError::Backend {
                id,
                message: String::from("mmap failed"),
            },
        ] {
            assert!(transient.is_transient(), "{transient} may clear");
        }
        for permanent in [
            WndError::Unsupported(String::from("no wl_compositor global")),
            WndError::Disconnected {
                id,
                message: String::from("peer hung up"),
            },
            WndError::Protocol {
                id,
                message: String::from("invalid surface"),
            },
        ] {
            assert!(!permanent.is_transient(), "{permanent} is final");
        }
    }

    #[test]
    fn classified_errors_name_their_backend() {
        let error = WndError::Timeout {
            id: WndBackendId::Wayland,
            message: String::from("initial configure never arrived"),
        };
        assert_eq!(
            error.to_string(),
            "wayland timed out: initial configure never arrived"
        );
    }
}
