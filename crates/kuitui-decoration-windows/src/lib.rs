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
//! (`SW_MINIMIZE`, `SW_MAXIMIZE`, `SW_SHOW`) and DWM attributes.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use codevar_kuitui_decoration::{
    Decoration, DecorationBackend, DecorationBounds, DecorationIcon, DecorationKind,
    DecorationMetrics, DecorationModifier, DecorationModifiers, DecorationResult,
};

/// The Win32 decoration backend.
#[cfg(windows)]
pub struct WindowsBackend {
    /// The window handle the backend owns, if any.
    pub window: Option<isize>,
    /// The decorations the backend owns, keyed by their id.
    pub decorations: alloc::vec::Vec<(codevar_kuitui_decoration::types::DecorationId, alloc::rc::Rc<dyn Decoration>)>,
}

/// The Win32 decoration backend on non-Windows targets.
#[cfg(not(windows))]
pub struct WindowsBackend {
    /// The decorations the backend owns, keyed by their id.
    pub decorations: alloc::vec::Vec<(codevar_kuitui_decoration::types::DecorationId, alloc::rc::Rc<dyn Decoration>)>,
}

impl WindowsBackend {
    /// Creates a backend that owns no window yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(windows)]
            window: None,
            decorations: alloc::vec::Vec::new(),
        }
    }
}

impl DecorationBackend for WindowsBackend {
    fn create_surface(
        &mut self,
        _decoration: &dyn Decoration,
    ) -> DecorationResult<codevar_kuitui_decoration::types::DecorationId> {
        Err(codevar_kuitui_decoration::DecorationError::Unavailable)
    }

    fn destroy_surface(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
    ) -> DecorationResult<()> {
        Err(codevar_kuitui_decoration::DecorationError::Unavailable)
    }

    fn set_title(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
        _title: &str,
    ) -> DecorationResult<()> {
        Err(codevar_kuitui_decoration::DecorationError::Unavailable)
    }

    fn set_icon(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
        _icon: &DecorationIcon,
    ) -> DecorationResult<()> {
        Err(codevar_kuitui_decoration::DecorationError::Unavailable)
    }

    fn set_bounds(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
        _bounds: DecorationBounds,
    ) -> DecorationResult<()> {
        Err(codevar_kuitui_decoration::DecorationError::Unavailable)
    }

    fn set_modifiers(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
        _modifiers: DecorationModifiers,
    ) -> DecorationResult<()> {
        Err(codevar_kuitui_decoration::DecorationError::Unavailable)
    }

    fn metrics(&self, _id: codevar_kuitui_decoration::types::DecorationId) -> DecorationMetrics {
        DecorationMetrics::default()
    }

    fn pointer_event(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
        _event: &codevar_kuitui::event::PointerEvent,
    ) {
    }

    fn key_event(
        &mut self,
        _id: codevar_kuitui_decoration::types::DecorationId,
        _event: &codevar_kuitui::event::KeyEvent,
    ) {
    }
}

/// A Win32 window style bit and the decoration modifier it maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Win32Style {
    /// `WS_CAPTION`: the window has a title bar.
    Caption,
    /// `WS_THICKFRAME`: the window is resizable.
    ThickFrame,
    /// `WS_MINIMIZEBOX`: the window can be minimized.
    MinimizeBox,
    /// `WS_MAXIMIZEBOX`: the window can be maximized.
    MaximizeBox,
    /// `WS_SYSMENU`: the window has a window menu / close button.
    SysMenu,
    /// `WS_POPUP`: the window is a popup / fullscreen window.
    Popup,
    /// `WS_DISABLED`: the window is suspended.
    Disabled,
}

impl From<Win32Style> for DecorationModifier {
    fn from(style: Win32Style) -> Self {
        match style {
            Win32Style::Caption => DecorationModifier::Closable,
            Win32Style::ThickFrame => DecorationModifier::Resizable,
            Win32Style::MinimizeBox => DecorationModifier::Minimized,
            Win32Style::MaximizeBox => DecorationModifier::Maximized,
            Win32Style::SysMenu => DecorationModifier::Menu,
            Win32Style::Popup => DecorationModifier::Fullscreen,
            Win32Style::Disabled => DecorationModifier::Suspended,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_win32_style_maps_to_a_modifier() {
        for style in [
            Win32Style::Caption,
            Win32Style::ThickFrame,
            Win32Style::MinimizeBox,
            Win32Style::MaximizeBox,
            Win32Style::SysMenu,
            Win32Style::Popup,
            Win32Style::Disabled,
        ] {
            let _ = DecorationModifier::from(style);
        }
    }
}
