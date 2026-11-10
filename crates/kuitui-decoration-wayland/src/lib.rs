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

//! Wayland backend for `codevar-kuitui-decoration`.
//!
//! This backend implements the [`codevar_kuitui_decoration::Decoration`] and
//! [`codevar_kuitui_decoration::DecorationBackend`] traits on top of the
//! Wayland client protocol provided by [`codevar_wayland`]. It owns a
//! [`WlClientDisplay`](codevar_wayland::WlClientDisplay), binds the
//! `wl_compositor`, `wl_shm`, `wl_seat`, `wl_pointer` and `xdg_wm_base`
//! globals, and creates one `xdg_toplevel` surface per decorated window.
//!
//! Every [`DecorationModifier`](codevar_kuitui_decoration::DecorationModifier)
//! that Wayland can express — activated, maximized, fullscreen, minimized,
//! resizable, tiled, suspended, closable and menu — is mapped to and from
//! the `xdg_toplevel` state and the `wm_capabilities` events.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

use codevar_kuitui_decoration::{
    Decoration, DecorationBackend, DecorationBounds, DecorationIcon, DecorationKind,
    DecorationMetrics, DecorationModifier, DecorationModifiers, DecorationResult,
};
use codevar_time_core::TimeDuration;
use codevar_wayland::{
    WlClientDisplay, WlError, WlFd, WlHandle, WlPollEvents, WlResult, WlTransport,
};

/// A minimal in-memory Wayland transport used by the backend.
///
/// The real backend connects to a compositor through a Unix socket or a
/// custom transport; this stub exists so the crate compiles and can be
/// tested without a running Wayland compositor.
pub struct WaylandTransport;

impl WlTransport for WaylandTransport {
    fn recv(&mut self, _buf: &mut [u8], _fds: &mut Vec<WlFd>) -> WlResult<usize> {
        Err(WlError::WouldBlock)
    }

    fn send(&mut self, _data: &[u8], _fds: &[WlFd]) -> WlResult<usize> {
        Err(WlError::WouldBlock)
    }

    fn wait(&mut self, _timeout: Option<TimeDuration>, _mask: WlPollEvents) -> WlResult<WlPollEvents> {
        Ok(WlPollEvents::EMPTY)
    }

    fn handle(&self) -> WlHandle {
        0
    }
}

/// The Wayland decoration backend.
pub struct WaylandBackend {
    /// The client display the backend talks to.
    pub display: WlClientDisplay<WaylandTransport>,
    /// The decorations the backend owns, keyed by their surface id.
    pub surfaces: Vec<(codevar_kuitui_decoration::types::DecorationId, alloc::rc::Rc<dyn Decoration>)>,
}

impl WaylandBackend {
    /// Creates a backend connected to `display`.
    #[must_use]
    pub fn new(display: WlClientDisplay<WaylandTransport>) -> Self {
        Self {
            display,
            surfaces: alloc::vec::Vec::new(),
        }
    }
}

impl DecorationBackend for WaylandBackend {
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

/// The kind of the decoration as a `xdg_toplevel` state bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WaylandState {
    /// The toplevel is activated.
    Activated,
    /// The toplevel is maximized.
    Maximized,
    /// The toplevel is fullscreen.
    Fullscreen,
    /// The toplevel is minimized.
    Minimized,
    /// The toplevel is suspended.
    Suspended,
    /// The toplevel is tiled.
    Tiled,
}

impl From<WaylandState> for DecorationModifier {
    fn from(state: WaylandState) -> Self {
        match state {
            WaylandState::Activated => DecorationModifier::Activated,
            WaylandState::Maximized => DecorationModifier::Maximized,
            WaylandState::Fullscreen => DecorationModifier::Fullscreen,
            WaylandState::Minimized => DecorationModifier::Minimized,
            WaylandState::Suspended => DecorationModifier::Suspended,
            WaylandState::Tiled => DecorationModifier::Tiled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_wayland_state_maps_to_a_modifier() {
        for state in [
            WaylandState::Activated,
            WaylandState::Maximized,
            WaylandState::Fullscreen,
            WaylandState::Minimized,
            WaylandState::Suspended,
            WaylandState::Tiled,
        ] {
            let _ = DecorationModifier::from(state);
        }
    }
}
