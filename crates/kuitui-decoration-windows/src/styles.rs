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

//! Win32 window styles and extended styles.
//!
//! Maps every [`DecorationModifier`](codevar_kuitui_decoration::DecorationModifier)
//! Win32 can express to and from window styles (`WS_*`) and extended styles
//! (`WS_EX_*`).

use codevar_kuitui_decoration::{DecorationModifier, DecorationModifiers};

/// A Win32 window style and the decoration modifier it maps to.
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

impl Win32Style {
    /// The `WS_*` value of the style.
    #[cfg(windows)]
    #[must_use]
    pub const fn ws(self) -> windows::Win32::UI::WindowsAndMessaging::WINDOW_STYLE {
        use windows::Win32::UI::WindowsAndMessaging::*;
        match self {
            Win32Style::Caption => WS_CAPTION,
            Win32Style::ThickFrame => WS_THICKFRAME,
            Win32Style::MinimizeBox => WS_MINIMIZEBOX,
            Win32Style::MaximizeBox => WS_MAXIMIZEBOX,
            Win32Style::SysMenu => WS_SYSMENU,
            Win32Style::Popup => WS_POPUP,
            Win32Style::Disabled => WS_DISABLED,
        }
    }

    /// The decoration modifier this style maps to.
    #[must_use]
    pub const fn modifier(self) -> DecorationModifier {
        match self {
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

/// A Win32 extended style and the decoration modifier it maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Win32ExStyle {
    /// `WS_EX_TOOLWINDOW`: the window is a tool window (no taskbar button).
    ToolWindow,
    /// `WS_EX_TOPMOST`: the window stays above all others.
    Topmost,
    /// `WS_EX_LAYERED`: the window supports per-pixel alpha.
    Layered,
    /// `WS_EX_ACCEPTFILES`: the window accepts drag-and-drop files.
    AcceptFiles,
}

impl Win32ExStyle {
    /// The `WS_EX_*` value of the style.
    #[cfg(windows)]
    #[must_use]
    pub const fn ex(self) -> windows::Win32::UI::WindowsAndMessaging::WINDOW_EX_STYLE {
        use windows::Win32::UI::WindowsAndMessaging::*;
        match self {
            Win32ExStyle::ToolWindow => WS_EX_TOOLWINDOW,
            Win32ExStyle::Topmost => WS_EX_TOPMOST,
            Win32ExStyle::Layered => WS_EX_LAYERED,
            Win32ExStyle::AcceptFiles => WS_EX_ACCEPTFILES,
        }
    }
}

#[cfg(windows)]
mod win32_styles {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::*;

    /// Builds the `WS_*` style flags from a set of decoration modifiers.
    #[must_use]
    pub fn style_from_modifiers(modifiers: DecorationModifiers) -> WINDOW_STYLE {
        let mut style = WS_OVERLAPPED | WS_CAPTION;
        if modifiers.has(DecorationModifier::Resizable) {
            style |= WS_THICKFRAME;
        }
        if modifiers.has(DecorationModifier::Minimized) || modifiers.has(DecorationModifier::Maximized) {
            style |= WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
        }
        if modifiers.has(DecorationModifier::Closable) || modifiers.has(DecorationModifier::Menu) {
            style |= WS_SYSMENU;
        }
        if modifiers.has(DecorationModifier::Fullscreen) {
            style = WS_POPUP;
        }
        if modifiers.has(DecorationModifier::Suspended) {
            style |= WS_DISABLED;
        }
        style
    }

    /// Reads the decoration modifiers implied by a window's `WS_*` style.
    #[must_use]
    pub fn modifiers_from_style(style: WINDOW_STYLE, ex_style: WINDOW_EX_STYLE) -> DecorationModifiers {
        let mut modifiers = DecorationModifiers::NONE;
        if style & WS_THICKFRAME != WINDOW_STYLE(0) {
            modifiers = modifiers.with(DecorationModifier::Resizable);
        }
        if style & WS_MINIMIZEBOX != WINDOW_STYLE(0) {
            modifiers = modifiers.with(DecorationModifier::Minimized);
        }
        if style & WS_MAXIMIZEBOX != WINDOW_STYLE(0) {
            modifiers = modifiers.with(DecorationModifier::Maximized);
        }
        if style & WS_SYSMENU != WINDOW_STYLE(0) {
            modifiers = modifiers.with(DecorationModifier::Closable);
            modifiers = modifiers.with(DecorationModifier::Menu);
        }
        if style & WS_POPUP != WINDOW_STYLE(0) {
            modifiers = modifiers.with(DecorationModifier::Fullscreen);
        }
        if style & WS_DISABLED != WINDOW_STYLE(0) {
            modifiers = modifiers.with(DecorationModifier::Suspended);
        }
        let _ = ex_style;
        modifiers
    }
}

#[cfg(not(windows))]
mod win32_styles {
    use super::*;
    /// Builds the `WS_*` style flags from a set of decoration modifiers.
    #[must_use]
    pub fn style_from_modifiers(_modifiers: DecorationModifiers) -> u32 {
        0
    }
    /// Reads the decoration modifiers implied by a window's `WS_*` style.
    #[must_use]
    pub fn modifiers_from_style(_style: u32, _ex_style: u32) -> DecorationModifiers {
        DecorationModifiers::NONE
    }
}

pub use win32_styles::{modifiers_from_style, style_from_modifiers};
