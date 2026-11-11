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

//! Win32 backend for `codevar-kuitui-decoration`.
//!
//! This backend implements the [`codevar_kuitui_decoration::Decoration`] and
//! [`codevar_kuitui_decoration::DecorationBackend`] traits on top of the
//! Win32 windowing API: window styles, extended styles, DWM window
//! attributes, non-client metrics, window icons and mouse cursors.
//!
//! Every [`DecorationModifier`](codevar_kuitui_decoration::DecorationModifier)
//! that Win32 can express — activated, maximized, fullscreen, minimized,
//! resizable, suspended, closable and menu — is mapped to and from window
//! styles (`WS_CAPTION`, `WS_THICKFRAME`, `WS_MINIMIZEBOX`,
//! `WS_MAXIMIZEBOX`, `WS_SYSMENU`, `WS_POPUP`, `WS_DISABLED`), show states
//! (`SW_MINIMIZE`, `WS_MAXIMIZE`, `SW_SHOW`) and DWM attributes.
//!
//! # Platform gating
//!
//! All Win32-specific code is gated behind `#[cfg(windows)]`. On other
//! targets the crate still compiles and every backend method returns
//! [`codevar_kuitui_decoration::DecorationError::Unavailable`], exactly as
//! the `codevar-kuitui-control-windows` backend does.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(windows)]
mod cursor;
#[cfg(windows)]
mod dwm;
#[cfg(windows)]
mod icon;
#[cfg(windows)]
mod metrics;
#[cfg(windows)]
mod styles;
#[cfg(windows)]
mod window;

use codevar_kuitui::event::{KeyEvent, PointerEvent};
use codevar_kuitui_decoration::{
    Decoration, DecorationBackend, DecorationBounds, DecorationError, DecorationIcon, DecorationId,
    DecorationMetrics, DecorationModifiers, DecorationResult,
};

/// The Win32 decoration backend.
#[derive(Default)]
pub struct WindowsBackend {
    /// The window the backend owns, if any.
    #[cfg(windows)]
    window: Option<crate::window::Window>,
    /// The windows the backend owns, keyed by their decoration id.
    /// Stored as raw handles so the field exists on every target.
    windows: alloc::vec::Vec<(DecorationId, isize)>,
}

impl WindowsBackend {
    /// Creates a backend that owns no window yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(windows)]
            window: None,
            windows: alloc::vec::Vec::new(),
        }
    }

    /// Finds the window bound to `id`, if any.
    #[cfg(windows)]
    fn find_window(&self, id: DecorationId) -> Option<crate::window::Window> {
        self.windows
            .iter()
            .find(|(decoration_id, _)| *decoration_id == id)
            .map(|(_, handle)| crate::window::Window::from_handle(*handle))
    }

    /// Binds `id` to `window`, replacing any previous binding.
    #[cfg(windows)]
    fn bind_window(&mut self, id: DecorationId, window: crate::window::Window) {
        let handle = window.handle();
        if let Some(entry) = self
            .windows
            .iter_mut()
            .find(|(decoration_id, _)| *decoration_id == id)
        {
            entry.1 = handle;
        } else {
            self.windows.push((id, handle));
        }
    }

    /// Unbinds the window bound to `id` and returns it, if any.
    #[cfg(windows)]
    fn unbind_window(&mut self, id: DecorationId) -> Option<crate::window::Window> {
        if let Some(index) = self
            .windows
            .iter()
            .position(|(decoration_id, _)| *decoration_id == id)
        {
            let handle = self.windows.swap_remove(index).1;
            Some(crate::window::Window::from_handle(handle))
        } else {
            None
        }
    }
}

impl DecorationBackend for WindowsBackend {
    fn create_surface(&mut self, decoration: &dyn Decoration) -> DecorationResult<DecorationId> {
        let id = DecorationId::new(self.windows.len() as u64 + 1);

        #[cfg(windows)]
        {
            // Register the window class (idempotent: ignore if already present).
            let _ = crate::window::register_class();

            let bounds = decoration.bounds();
            let modifiers = decoration.active_modifiers();
            let title = match decoration.kind() {
                codevar_kuitui_decoration::DecorationKind::Window => "Codevar",
                _ => "Codevar Window",
            };

            let window = crate::window::create_window(title, bounds, modifiers)?;
            self.bind_window(id, window);

            if let Some(icon) = decoration.icon() {
                let _ = self.set_icon(id, icon);
            }

            crate::window::set_title(window, title)?;
        }

        #[cfg(not(windows))]
        {
            let _ = decoration;
            let _ = id;
        }

        Ok(id)
    }

    fn destroy_surface(&mut self, id: DecorationId) -> DecorationResult<()> {
        let _ = id;
        #[cfg(windows)]
        {
            if let Some(window) = self.unbind_window(id) {
                crate::window::destroy_window(window)?;
            }
        }
        Ok(())
    }

    fn set_title(&mut self, id: DecorationId, title: &str) -> DecorationResult<()> {
        #[cfg(windows)]
        {
            let window = self
                .find_window(id)
                .ok_or(DecorationError::NotInitialized)?;
            crate::window::set_title(window, title)
        }
        #[cfg(not(windows))]
        {
            let _ = id;
            let _ = title;
            Err(DecorationError::Unavailable)
        }
    }

    fn set_icon(&mut self, id: DecorationId, icon: &DecorationIcon) -> DecorationResult<()> {
        #[cfg(windows)]
        {
            let window = self
                .find_window(id)
                .ok_or(DecorationError::NotInitialized)?;
            let hicon = crate::icon::create_icon(icon)
                .ok_or_else(|| DecorationError::Os("failed to create icon".into()))?;
            if crate::icon::set_window_icon(window.handle(), hicon) {
                Ok(())
            } else {
                Err(DecorationError::Os("failed to set window icon".into()))
            }
        }
        #[cfg(not(windows))]
        {
            let _ = id;
            let _ = icon;
            Err(DecorationError::Unavailable)
        }
    }

    fn set_bounds(&mut self, id: DecorationId, bounds: DecorationBounds) -> DecorationResult<()> {
        #[cfg(windows)]
        {
            let window = self
                .find_window(id)
                .ok_or(DecorationError::NotInitialized)?;
            crate::window::set_bounds(window, bounds)
        }
        #[cfg(not(windows))]
        {
            let _ = id;
            let _ = bounds;
            Err(DecorationError::Unavailable)
        }
    }

    fn set_modifiers(&mut self, id: DecorationId, modifiers: DecorationModifiers) -> DecorationResult<()> {
        #[cfg(windows)]
        {
            let window = self
                .find_window(id)
                .ok_or(DecorationError::NotInitialized)?;
            crate::window::set_modifiers(window, modifiers)?;
            crate::dwm::apply_modifiers(window.handle(), modifiers);
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let _ = id;
            let _ = modifiers;
            Err(DecorationError::Unavailable)
        }
    }

    fn metrics(&self, _id: DecorationId) -> DecorationMetrics {
        #[cfg(windows)]
        {
            return crate::metrics::decoration_metrics();
        }
        #[cfg(not(windows))]
        {
            DecorationMetrics::default()
        }
    }

    fn pointer_event(&mut self, id: DecorationId, event: &PointerEvent) {
        #[cfg(windows)]
        {
            let _ = id;
            let _ = event;
        }
        #[cfg(not(windows))]
        {
            let _ = id;
            let _ = event;
        }
    }

    fn key_event(&mut self, _id: DecorationId, _event: &KeyEvent) {}
}

impl Drop for WindowsBackend {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            for (_, window) in self.windows.drain(..) {
                let _ = crate::window::destroy_window(window);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_creates_with_no_window() {
        let backend = WindowsBackend::new();
        assert!(backend.windows.is_empty());
    }

    #[test]
    fn metrics_have_sensible_defaults() {
        let backend = WindowsBackend::new();
        let metrics = backend.metrics(DecorationId::new(1));
        assert!(metrics.title_bar_height > 0);
        assert!(metrics.border_thickness > 0);
    }
}
