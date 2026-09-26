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

//! `xdg-shell` window roles for toolbars and desktop shells.
//!
//! The tables mirror `stable/xdg-shell/xdg-shell.xml` of
//! `wayland-protocols` message by message: signatures, `since`
//! versions, argument interfaces and the opcode order a real
//! compositor expects on the wire. The reference repository
//! (<https://github.com/XQuartz/wayland>) only vendors the core
//! protocol, so the stable xdg-shell protocol is taken from its
//! upstream source.
//!
//! The handshake a window performs is: `xdg_wm_base.get_xdg_surface`
//! on a committed [`wl_surface`](crate::SURFACE_INTERFACE),
//! `xdg_surface.get_toplevel`, an empty commit, then
//! `xdg_surface.ack_configure` with the serial of the first
//! `xdg_surface.configure` event before the first buffer is attached.

use core::fmt;

use crate::wl_core::{OUTPUT_INTERFACE, SEAT_INTERFACE, SURFACE_INTERFACE};
use crate::wl_handle::{WlInterface, WlMessage};

/// `xdg_wm_base.destroy` request opcode.
pub const XDG_WM_BASE_DESTROY: u32 = 0;
/// `xdg_wm_base.create_positioner` request opcode.
pub const XDG_WM_BASE_CREATE_POSITIONER: u32 = 1;
/// `xdg_wm_base.get_xdg_surface` request opcode.
pub const XDG_WM_BASE_GET_XDG_SURFACE: u32 = 2;
/// `xdg_wm_base.pong` request opcode.
pub const XDG_WM_BASE_PONG: u32 = 3;
/// `xdg_wm_base.ping` event opcode.
pub const XDG_WM_BASE_PING: u32 = 0;

/// `xdg_positioner.destroy` request opcode.
pub const XDG_POSITIONER_DESTROY: u32 = 0;
/// `xdg_positioner.set_size` request opcode.
pub const XDG_POSITIONER_SET_SIZE: u32 = 1;
/// `xdg_positioner.set_anchor_rect` request opcode.
pub const XDG_POSITIONER_SET_ANCHOR_RECT: u32 = 2;
/// `xdg_positioner.set_anchor` request opcode.
pub const XDG_POSITIONER_SET_ANCHOR: u32 = 3;
/// `xdg_positioner.set_gravity` request opcode.
pub const XDG_POSITIONER_SET_GRAVITY: u32 = 4;
/// `xdg_positioner.set_constraint_adjustment` request opcode.
pub const XDG_POSITIONER_SET_CONSTRAINT_ADJUSTMENT: u32 = 5;
/// `xdg_positioner.set_offset` request opcode.
pub const XDG_POSITIONER_SET_OFFSET: u32 = 6;
/// `xdg_positioner.set_reactive` request opcode.
pub const XDG_POSITIONER_SET_REACTIVE: u32 = 7;
/// `xdg_positioner.set_parent_size` request opcode.
pub const XDG_POSITIONER_SET_PARENT_SIZE: u32 = 8;
/// `xdg_positioner.set_parent_configure` request opcode.
pub const XDG_POSITIONER_SET_PARENT_CONFIGURE: u32 = 9;

/// `xdg_surface.destroy` request opcode.
pub const XDG_SURFACE_DESTROY: u32 = 0;
/// `xdg_surface.get_toplevel` request opcode.
pub const XDG_SURFACE_GET_TOPLEVEL: u32 = 1;
/// `xdg_surface.get_popup` request opcode.
pub const XDG_SURFACE_GET_POPUP: u32 = 2;
/// `xdg_surface.set_window_geometry` request opcode.
pub const XDG_SURFACE_SET_WINDOW_GEOMETRY: u32 = 3;
/// `xdg_surface.ack_configure` request opcode.
pub const XDG_SURFACE_ACK_CONFIGURE: u32 = 4;
/// `xdg_surface.configure` event opcode.
pub const XDG_SURFACE_CONFIGURE: u32 = 0;

/// `xdg_toplevel.destroy` request opcode.
pub const XDG_TOPLEVEL_DESTROY: u32 = 0;
/// `xdg_toplevel.set_parent` request opcode.
pub const XDG_TOPLEVEL_SET_PARENT: u32 = 1;
/// `xdg_toplevel.set_title` request opcode.
pub const XDG_TOPLEVEL_SET_TITLE: u32 = 2;
/// `xdg_toplevel.set_app_id` request opcode.
pub const XDG_TOPLEVEL_SET_APP_ID: u32 = 3;
/// `xdg_toplevel.show_window_menu` request opcode.
pub const XDG_TOPLEVEL_SHOW_WINDOW_MENU: u32 = 4;
/// `xdg_toplevel.move` request opcode.
pub const XDG_TOPLEVEL_MOVE: u32 = 5;
/// `xdg_toplevel.resize` request opcode.
pub const XDG_TOPLEVEL_RESIZE: u32 = 6;
/// `xdg_toplevel.set_max_size` request opcode.
pub const XDG_TOPLEVEL_SET_MAX_SIZE: u32 = 7;
/// `xdg_toplevel.set_min_size` request opcode.
pub const XDG_TOPLEVEL_SET_MIN_SIZE: u32 = 8;
/// `xdg_toplevel.set_maximized` request opcode.
pub const XDG_TOPLEVEL_SET_MAXIMIZED: u32 = 9;
/// `xdg_toplevel.unset_maximized` request opcode.
pub const XDG_TOPLEVEL_UNSET_MAXIMIZED: u32 = 10;
/// `xdg_toplevel.set_fullscreen` request opcode.
pub const XDG_TOPLEVEL_SET_FULLSCREEN: u32 = 11;
/// `xdg_toplevel.unset_fullscreen` request opcode.
pub const XDG_TOPLEVEL_UNSET_FULLSCREEN: u32 = 12;
/// `xdg_toplevel.set_minimized` request opcode.
pub const XDG_TOPLEVEL_SET_MINIMIZED: u32 = 13;
/// `xdg_toplevel.configure` event opcode.
pub const XDG_TOPLEVEL_CONFIGURE: u32 = 0;
/// `xdg_toplevel.close` event opcode.
pub const XDG_TOPLEVEL_CLOSE: u32 = 1;
/// `xdg_toplevel.configure_bounds` event opcode.
pub const XDG_TOPLEVEL_CONFIGURE_BOUNDS: u32 = 2;
/// `xdg_toplevel.wm_capabilities` event opcode.
pub const XDG_TOPLEVEL_WM_CAPABILITIES: u32 = 3;

/// `xdg_popup.destroy` request opcode.
pub const XDG_POPUP_DESTROY: u32 = 0;
/// `xdg_popup.grab` request opcode.
pub const XDG_POPUP_GRAB: u32 = 1;
/// `xdg_popup.reposition` request opcode.
pub const XDG_POPUP_REPOSITION: u32 = 2;
/// `xdg_popup.configure` event opcode.
pub const XDG_POPUP_CONFIGURE: u32 = 0;
/// `xdg_popup.popup_done` event opcode.
pub const XDG_POPUP_POPUP_DONE: u32 = 1;
/// `xdg_popup.repositioned` event opcode.
pub const XDG_POPUP_REPOSITIONED: u32 = 2;

/// `xdg_wm_base` window management interface.
pub static XDG_WM_BASE_INTERFACE: WlInterface = WlInterface {
    name: "xdg_wm_base",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("create_positioner", "n", &[Some(&XDG_POSITIONER_INTERFACE)]),
        WlMessage::new(
            "get_xdg_surface",
            "no",
            &[Some(&XDG_SURFACE_INTERFACE), Some(&SURFACE_INTERFACE)],
        ),
        WlMessage::new("pong", "u", &[None]),
    ],
    events: &[WlMessage::new("ping", "u", &[None])],
};

/// `xdg_positioner` placement description for popups.
pub static XDG_POSITIONER_INTERFACE: WlInterface = WlInterface {
    name: "xdg_positioner",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("set_size", "ii", &[None, None]),
        WlMessage::new("set_anchor_rect", "iiii", &[None, None, None, None]),
        WlMessage::new("set_anchor", "u", &[None]),
        WlMessage::new("set_gravity", "u", &[None]),
        WlMessage::new("set_constraint_adjustment", "u", &[None]),
        WlMessage::new("set_offset", "ii", &[None, None]),
        WlMessage::new("set_reactive", "3", &[]),
        WlMessage::new("set_parent_size", "3ii", &[None, None]),
        WlMessage::new("set_parent_configure", "3u", &[None]),
    ],
    events: &[],
};

/// `xdg_surface` role object of a mapped window.
pub static XDG_SURFACE_INTERFACE: WlInterface = WlInterface {
    name: "xdg_surface",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("get_toplevel", "n", &[Some(&XDG_TOPLEVEL_INTERFACE)]),
        WlMessage::new(
            "get_popup",
            "n?oo",
            &[
                Some(&XDG_POPUP_INTERFACE),
                Some(&XDG_SURFACE_INTERFACE),
                Some(&XDG_POSITIONER_INTERFACE),
            ],
        ),
        WlMessage::new("set_window_geometry", "iiii", &[None, None, None, None]),
        WlMessage::new("ack_configure", "u", &[None]),
    ],
    events: &[WlMessage::new("configure", "u", &[None])],
};

/// `xdg_toplevel` role object of a regular window.
pub static XDG_TOPLEVEL_INTERFACE: WlInterface = WlInterface {
    name: "xdg_toplevel",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("set_parent", "?o", &[Some(&XDG_TOPLEVEL_INTERFACE)]),
        WlMessage::new("set_title", "s", &[None]),
        WlMessage::new("set_app_id", "s", &[None]),
        WlMessage::new(
            "show_window_menu",
            "ouii",
            &[Some(&SEAT_INTERFACE), None, None, None],
        ),
        WlMessage::new("move", "ou", &[Some(&SEAT_INTERFACE), None]),
        WlMessage::new("resize", "ouu", &[Some(&SEAT_INTERFACE), None, None]),
        WlMessage::new("set_max_size", "ii", &[None, None]),
        WlMessage::new("set_min_size", "ii", &[None, None]),
        WlMessage::new("set_maximized", "", &[]),
        WlMessage::new("unset_maximized", "", &[]),
        WlMessage::new("set_fullscreen", "?o", &[Some(&OUTPUT_INTERFACE)]),
        WlMessage::new("unset_fullscreen", "", &[]),
        WlMessage::new("set_minimized", "", &[]),
    ],
    events: &[
        WlMessage::new("configure", "iia", &[None, None, None]),
        WlMessage::new("close", "", &[]),
        WlMessage::new("configure_bounds", "4ii", &[None, None]),
        WlMessage::new("wm_capabilities", "5a", &[None]),
    ],
};

/// `xdg_popup` role object of a transient popup.
pub static XDG_POPUP_INTERFACE: WlInterface = WlInterface {
    name: "xdg_popup",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("grab", "ou", &[Some(&SEAT_INTERFACE), None]),
        WlMessage::new("reposition", "3ou", &[Some(&XDG_POSITIONER_INTERFACE), None]),
    ],
    events: &[
        WlMessage::new("configure", "iiii", &[None, None, None, None]),
        WlMessage::new("popup_done", "", &[]),
        WlMessage::new("repositioned", "3u", &[None]),
    ],
};

/// Protocol error codes of `enum xdg_wm_base.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum XdgWmBaseError {
    /// The client tried to give a surface the same role twice.
    Role = 0,
    /// The client destroyed a surface that still had a role object.
    DefunctSurfaces = 1,
    /// The client did not grab the topmost popup.
    NotTheTopmostPopup = 2,
    /// The client specified an invalid popup parent.
    InvalidPopupParent = 3,
    /// The client committed an invalid surface state.
    InvalidSurfaceState = 4,
    /// The client used an invalid positioner.
    InvalidPositioner = 5,
    /// The compositor gave up waiting for the client to respond.
    Unresponsive = 6,
}

impl XdgWmBaseError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Role),
            1 => Some(Self::DefunctSurfaces),
            2 => Some(Self::NotTheTopmostPopup),
            3 => Some(Self::InvalidPopupParent),
            4 => Some(Self::InvalidSurfaceState),
            5 => Some(Self::InvalidPositioner),
            6 => Some(Self::Unresponsive),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Role => "role",
            Self::DefunctSurfaces => "defunct_surfaces",
            Self::NotTheTopmostPopup => "not_the_topmost_popup",
            Self::InvalidPopupParent => "invalid_popup_parent",
            Self::InvalidSurfaceState => "invalid_surface_state",
            Self::InvalidPositioner => "invalid_positioner",
            Self::Unresponsive => "unresponsive",
        }
    }
}

impl fmt::Display for XdgWmBaseError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Protocol error codes of `enum xdg_positioner.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum XdgPositionerError {
    /// Positioner constraints were incomplete or contradictory.
    InvalidInput = 0,
}

impl XdgPositionerError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::InvalidInput),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
        }
    }
}

impl fmt::Display for XdgPositionerError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Protocol error codes of `enum xdg_surface.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum XdgSurfaceError {
    /// The role object was used before `get_toplevel` or `get_popup`.
    NotConstructed = 1,
    /// `get_toplevel` or `get_popup` was called twice.
    AlreadyConstructed = 2,
    /// A buffer was committed before the first configure was acked.
    UnconfiguredBuffer = 3,
    /// The acked configure serial is unknown.
    InvalidSerial = 4,
    /// The committed window geometry is invalid.
    InvalidSize = 5,
    /// The role object was destroyed before its surface.
    DefunctRoleObject = 6,
}

impl XdgSurfaceError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::NotConstructed),
            2 => Some(Self::AlreadyConstructed),
            3 => Some(Self::UnconfiguredBuffer),
            4 => Some(Self::InvalidSerial),
            5 => Some(Self::InvalidSize),
            6 => Some(Self::DefunctRoleObject),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotConstructed => "not_constructed",
            Self::AlreadyConstructed => "already_constructed",
            Self::UnconfiguredBuffer => "unconfigured_buffer",
            Self::InvalidSerial => "invalid_serial",
            Self::InvalidSize => "invalid_size",
            Self::DefunctRoleObject => "defunct_role_object",
        }
    }
}

impl fmt::Display for XdgSurfaceError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Protocol error codes of `enum xdg_toplevel.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum XdgToplevelError {
    /// The resize edge mask is invalid.
    InvalidResizeEdge = 0,
    /// The parent toplevel is invalid.
    InvalidParent = 1,
    /// The size bounds are invalid.
    InvalidSize = 2,
}

impl XdgToplevelError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::InvalidResizeEdge),
            1 => Some(Self::InvalidParent),
            2 => Some(Self::InvalidSize),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidResizeEdge => "invalid_resize_edge",
            Self::InvalidParent => "invalid_parent",
            Self::InvalidSize => "invalid_size",
        }
    }
}

impl fmt::Display for XdgToplevelError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Protocol error codes of `enum xdg_popup.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum XdgPopupError {
    /// The grab was not on the topmost popup.
    InvalidGrab = 0,
}

impl XdgPopupError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::InvalidGrab),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidGrab => "invalid_grab",
        }
    }
}

impl fmt::Display for XdgPopupError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[&WlInterface] = &[
        &XDG_WM_BASE_INTERFACE,
        &XDG_POSITIONER_INTERFACE,
        &XDG_SURFACE_INTERFACE,
        &XDG_TOPLEVEL_INTERFACE,
        &XDG_POPUP_INTERFACE,
    ];

    const REQUEST_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (&XDG_WM_BASE_INTERFACE, XDG_WM_BASE_DESTROY, "destroy"),
        (
            &XDG_WM_BASE_INTERFACE,
            XDG_WM_BASE_CREATE_POSITIONER,
            "create_positioner",
        ),
        (
            &XDG_WM_BASE_INTERFACE,
            XDG_WM_BASE_GET_XDG_SURFACE,
            "get_xdg_surface",
        ),
        (&XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PONG, "pong"),
        (&XDG_POSITIONER_INTERFACE, XDG_POSITIONER_DESTROY, "destroy"),
        (&XDG_POSITIONER_INTERFACE, XDG_POSITIONER_SET_SIZE, "set_size"),
        (
            &XDG_POSITIONER_INTERFACE,
            XDG_POSITIONER_SET_ANCHOR_RECT,
            "set_anchor_rect",
        ),
        (&XDG_POSITIONER_INTERFACE, XDG_POSITIONER_SET_ANCHOR, "set_anchor"),
        (
            &XDG_POSITIONER_INTERFACE,
            XDG_POSITIONER_SET_GRAVITY,
            "set_gravity",
        ),
        (
            &XDG_POSITIONER_INTERFACE,
            XDG_POSITIONER_SET_CONSTRAINT_ADJUSTMENT,
            "set_constraint_adjustment",
        ),
        (&XDG_POSITIONER_INTERFACE, XDG_POSITIONER_SET_OFFSET, "set_offset"),
        (
            &XDG_POSITIONER_INTERFACE,
            XDG_POSITIONER_SET_REACTIVE,
            "set_reactive",
        ),
        (
            &XDG_POSITIONER_INTERFACE,
            XDG_POSITIONER_SET_PARENT_SIZE,
            "set_parent_size",
        ),
        (
            &XDG_POSITIONER_INTERFACE,
            XDG_POSITIONER_SET_PARENT_CONFIGURE,
            "set_parent_configure",
        ),
        (&XDG_SURFACE_INTERFACE, XDG_SURFACE_DESTROY, "destroy"),
        (&XDG_SURFACE_INTERFACE, XDG_SURFACE_GET_TOPLEVEL, "get_toplevel"),
        (&XDG_SURFACE_INTERFACE, XDG_SURFACE_GET_POPUP, "get_popup"),
        (
            &XDG_SURFACE_INTERFACE,
            XDG_SURFACE_SET_WINDOW_GEOMETRY,
            "set_window_geometry",
        ),
        (&XDG_SURFACE_INTERFACE, XDG_SURFACE_ACK_CONFIGURE, "ack_configure"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_DESTROY, "destroy"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_SET_PARENT, "set_parent"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_SET_TITLE, "set_title"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_SET_APP_ID, "set_app_id"),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_SHOW_WINDOW_MENU,
            "show_window_menu",
        ),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_MOVE, "move"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_RESIZE, "resize"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_SET_MAX_SIZE, "set_max_size"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_SET_MIN_SIZE, "set_min_size"),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_SET_MAXIMIZED,
            "set_maximized",
        ),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_UNSET_MAXIMIZED,
            "unset_maximized",
        ),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_SET_FULLSCREEN,
            "set_fullscreen",
        ),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_UNSET_FULLSCREEN,
            "unset_fullscreen",
        ),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_SET_MINIMIZED,
            "set_minimized",
        ),
        (&XDG_POPUP_INTERFACE, XDG_POPUP_DESTROY, "destroy"),
        (&XDG_POPUP_INTERFACE, XDG_POPUP_GRAB, "grab"),
        (&XDG_POPUP_INTERFACE, XDG_POPUP_REPOSITION, "reposition"),
    ];

    const EVENT_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (&XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PING, "ping"),
        (&XDG_SURFACE_INTERFACE, XDG_SURFACE_CONFIGURE, "configure"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_CONFIGURE, "configure"),
        (&XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_CLOSE, "close"),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_CONFIGURE_BOUNDS,
            "configure_bounds",
        ),
        (
            &XDG_TOPLEVEL_INTERFACE,
            XDG_TOPLEVEL_WM_CAPABILITIES,
            "wm_capabilities",
        ),
        (&XDG_POPUP_INTERFACE, XDG_POPUP_CONFIGURE, "configure"),
        (&XDG_POPUP_INTERFACE, XDG_POPUP_POPUP_DONE, "popup_done"),
        (&XDG_POPUP_INTERFACE, XDG_POPUP_REPOSITIONED, "repositioned"),
    ];

    #[test]
    fn every_message_slot_matches_its_signature() {
        for interface in ALL {
            for message in interface
                .requests
                .iter()
                .chain(interface.events.iter())
            {
                assert!(!message.name.is_empty(), "unnamed message");
                assert_eq!(
                    message.arg_count(),
                    message.types.len(),
                    "{}.{} signature and types table disagree",
                    interface.name,
                    message.name
                );
            }
        }
    }

    #[test]
    fn no_message_needs_a_newer_version_than_its_interface() {
        for interface in ALL {
            for message in interface
                .requests
                .iter()
                .chain(interface.events.iter())
            {
                assert!(
                    message.since() <= interface.version,
                    "{}.{} is since {} but the interface is version {}",
                    interface.name,
                    message.name,
                    message.since(),
                    interface.version
                );
            }
        }
    }

    #[test]
    fn opcodes_index_the_reference_tables() {
        for (interface, opcode, expected) in REQUEST_OPCODES {
            let message = interface
                .request_at(*opcode)
                .unwrap_or_else(|| panic!("{} has no request {opcode}", interface.name));
            assert_eq!(message.name, *expected, "{}.{}", interface.name, expected);
        }
        for (interface, opcode, expected) in EVENT_OPCODES {
            let message = interface
                .event_at(*opcode)
                .unwrap_or_else(|| panic!("{} has no event {opcode}", interface.name));
            assert_eq!(message.name, *expected, "{}.{}", interface.name, expected);
        }
    }

    #[test]
    fn signatures_and_versions_match_the_reference_protocol() {
        assert_eq!(XDG_WM_BASE_INTERFACE.version, 7);
        assert_eq!(XDG_TOPLEVEL_INTERFACE.version, 7);

        let get_xdg_surface = XDG_WM_BASE_INTERFACE
            .request_at(XDG_WM_BASE_GET_XDG_SURFACE)
            .unwrap_or_else(|| panic!("xdg_wm_base has no get_xdg_surface"));
        assert_eq!(get_xdg_surface.signature, "no");
        assert_eq!(
            get_xdg_surface
                .type_at(1)
                .map(|interface| interface.name),
            Some("wl_surface")
        );

        let get_popup = XDG_SURFACE_INTERFACE
            .request_at(XDG_SURFACE_GET_POPUP)
            .unwrap_or_else(|| panic!("xdg_surface has no get_popup"));
        assert_eq!(get_popup.signature, "n?oo");

        let configure = *XDG_TOPLEVEL_INTERFACE
            .event_at(XDG_TOPLEVEL_CONFIGURE)
            .unwrap_or_else(|| panic!("xdg_toplevel has no configure"));
        assert_eq!(configure.signature, "iia");
        assert_eq!(configure.array_count(), 1);

        let move_request = *XDG_TOPLEVEL_INTERFACE
            .request_at(XDG_TOPLEVEL_MOVE)
            .unwrap_or_else(|| panic!("xdg_toplevel has no move"));
        assert_eq!(move_request.signature, "ou");
        assert_eq!(
            move_request
                .type_at(0)
                .map(|interface| interface.name),
            Some("wl_seat")
        );

        let set_reactive = XDG_POSITIONER_INTERFACE
            .request_at(XDG_POSITIONER_SET_REACTIVE)
            .unwrap_or_else(|| panic!("xdg_positioner has no set_reactive"));
        assert_eq!(set_reactive.signature, "3");
        assert_eq!(set_reactive.since(), 3);

        let wm_capabilities = *XDG_TOPLEVEL_INTERFACE
            .event_at(XDG_TOPLEVEL_WM_CAPABILITIES)
            .unwrap_or_else(|| panic!("xdg_toplevel has no wm_capabilities"));
        assert_eq!(wm_capabilities.signature, "5a");
        assert_eq!(wm_capabilities.since(), 5);
    }

    #[test]
    fn error_codes_round_trip() {
        assert_eq!(XdgWmBaseError::InvalidPositioner.code(), 5);
        assert_eq!(
            XdgWmBaseError::from_code(5),
            Some(XdgWmBaseError::InvalidPositioner)
        );
        assert_eq!(XdgWmBaseError::from_code(7), None);
        assert_eq!(XdgWmBaseError::Role.name(), "role");
        assert_eq!(XdgPositionerError::InvalidInput.code(), 0);
        assert_eq!(
            XdgPositionerError::from_code(0),
            Some(XdgPositionerError::InvalidInput)
        );
        // xdg_surface.error has no entry at code 0.
        assert_eq!(XdgSurfaceError::from_code(0), None);
        assert_eq!(
            XdgSurfaceError::from_code(4),
            Some(XdgSurfaceError::InvalidSerial)
        );
        assert_eq!(
            XdgSurfaceError::DefunctRoleObject.to_string(),
            "defunct_role_object"
        );
        assert_eq!(
            XdgToplevelError::from_code(1),
            Some(XdgToplevelError::InvalidParent)
        );
        assert_eq!(XdgPopupError::from_code(0), Some(XdgPopupError::InvalidGrab));
        assert_eq!(XdgPopupError::from_code(1), None);
    }
}
