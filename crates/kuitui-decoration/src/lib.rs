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

//! Window decoration widgets for the Kuitui UI framework.
//!
//! A window of the Codevar editor is drawn by the application, not by
//! the windowing system: the title bar, the window icon, the mouse
//! cursor, the resize edges and the theme are widgets of the tree.
//! The widgets in this module draw them and report what the user did
//! with them.
//!
//! # The two layers
//!
//! 1. **The core crate** ([`codevar_kuitui_decoration`]) defines the
//!    platform-agnostic abstraction: the [`Decoration`] trait, the
//!    [`DecorationKind`], [`DecorationState`] and [`DecorationModifier`]
//!    types, the error model and the theme hooks. Nothing here depends on
//!    Wayland or Win32.
//! 2. **The backend crates** (`codevar-kuitui-decoration-wayland` and
//!    `codevar-kuitui-decoration-windows`) implement the same trait on top
//!    of the native windowing system. A target compiles only the backend
//!    it can run, exactly as `codevar-kuitui-control` splits evdev from
//!    Raw Input.
//!
//! # Supported decoration modifiers
//!
//! Every modifier that Wayland (`xdg_toplevel` state) and Win32 (window
//! styles, extended styles and DWM attributes) expose is represented:
//!
//! | Modifier | Wayland | Win32 |
//! | --- | --- | --- |
//! | Activated | `xdg_toplevel.configure` / `wm_capabilities` | `WS_CAPTION` + activation |
//! | Maximized | `xdg_toplevel.set_maximized` | `WS_MAXIMIZEBOX` / `SW_MAXIMIZE` |
//! | Fullscreen | `xdg_toplevel.set_fullscreen` | `WS_POPUP` / `SW_SHOWMAXIMIZED` |
//! | Minimized | `xdg_toplevel.set_minimized` | `WS_MINIMIZEBOX` / `SW_MINIMIZE` |
//! | Resizable | `xdg_toplevel.resize` edges | `WS_THICKFRAME` |
//! | Tiled | `xdg_toplevel.configure_bounds` / `wm_capabilities` | `SWP_*` tiling |
//! | Suspended | `xdg_toplevel.configure` serial `0` | `WS_DISABLED` |
//! | Close | `xdg_toplevel.close` | `WS_SYSMENU` / `WM_CLOSE` |
//! | Menu | `xdg_toplevel.show_window_menu` | `WS_SYSMENU` |
//!
//! # Writing a decoration
//!
//! A decoration implements the [`Decoration`] trait. Every hook has a
//! default implementation, so a decoration only overrides what it needs:
//!
//! ```
//! use codevar_kuitui_decoration::prelude::*;
//!
//! /// A decoration that paints nothing and accepts every modifier.
//! struct NullDecoration;
//!
//! impl codevar_kuitui::widget::Widget for NullDecoration {
//!     fn measure(
//!         &mut self,
//!         _scope: &mut codevar_kuitui::widget::MeasureScope<'_>,
//!         _c: codevar_kuitui::geometry::Constraints,
//!     ) -> codevar_kuitui::geometry::Size {
//!         _c.enforce(codevar_kuitui::geometry::Size::ZERO)
//!     }
//! }
//!
//! impl Decoration for NullDecoration {
//!     fn kind(&self) -> DecorationKind {
//!         DecorationKind::TitleBar
//!     }
//!
//!     fn id(&self) -> Option<DecorationId> {
//!         None
//!     }
//!
//!     fn bind(&mut self, _id: DecorationId) {}
//!
//!     fn unbind(&mut self) {}
//!
//!     fn supported_modifiers(&self) -> DecorationModifiers {
//!         DecorationModifiers::ALL
//!     }
//! }
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod error;
pub mod modifiers;
pub mod state;
pub mod theme;
pub mod traits;
pub mod types;
pub mod widgets;

pub use error::{DecorationError, DecorationResult};
pub use modifiers::{DecorationModifier, DecorationModifiers};
pub use state::{DecorationState, DecorationStates};
pub use theme::DecorationTheme;
pub use traits::{Decoration, DecorationBackend, DecorationHost};
pub use types::{
    DecorationBounds, DecorationIcon, DecorationId, DecorationKind, DecorationMetrics, DecorationResizeEdge,
};
pub use widgets::{
    DecorationButton, DecorationCursor, DecorationIconWidget, DecorationResizeEdgeWidget,
    DecorationResizeHandle, DecorationTitleBar, DecorationWindowFrame,
};

pub mod prelude {
    //! The traits and types a decoration implementation usually needs.
    pub use crate::{
        Decoration, DecorationBackend, DecorationBounds, DecorationButton, DecorationCursor, DecorationError,
        DecorationHost, DecorationIcon, DecorationIconWidget, DecorationId, DecorationKind,
        DecorationMetrics, DecorationModifier, DecorationModifiers, DecorationResizeEdgeWidget,
        DecorationResizeHandle, DecorationResult, DecorationState, DecorationStates, DecorationTheme,
        DecorationTitleBar, DecorationWindowFrame,
    };
}
