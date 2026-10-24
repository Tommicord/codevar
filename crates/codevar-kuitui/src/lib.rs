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

//! Kuitui — the user-interface framework of Codevar.
//!
//! Kuitui is a retained-mode, Compose-inspired UI framework. It is
//! deliberately **backend agnostic**: it computes a widget tree, a
//! layout and a flat *display list* of drawing primitives, and leaves
//! the actual rasterization to a backend. The desktop backend lives in
//! `codevar-gpu-core` (Vulkan) and the web backend in
//! `codevar-gpu-wg` (WebGPU); both consume the very same display list.
//!
//! # The three layers
//!
//! 1. **Composition** — a tree of [`Node`]s is built by a *content*
//!    closure. Re-running the closure *recomposes* the tree. A value
//!    that must outlive a recomposition is created with
//!    [`Composer::remember`], exactly as Compose keeps a value in a
//!    `remember` block; a value that is *read* by a widget and
//!    *written* by the application is a [`MutableState`], which marks
//!    the composition dirty when it is written.
//! 2. **Layout** — the tree is measured and placed in two passes, with
//!    the same *measure / place* split Jetpack Compose uses
//!    ([`Widget::measure`], [`Widget::place`]).
//! 3. **Painting** — the placed tree is recorded into a [`DrawList`],
//!    a flat sequence of [`Draw`]s that a backend replays.
//!
//! # Where Kuitui stands
//!
//! Kuitui is deliberately *not* an immediate-mode library like `egui`:
//! the tree is retained and only the parts that changed are rebuilt,
//! which is what a code editor needs, where the tree is large and
//! most of it does not change from one frame to the next. It is not a
//! retained-mode library with widgets that own their children either:
//! the containers lay out the children the tree gave them, which is
//! the split Compose makes between a composable that emits children
//! and the layout that arranges them.
//!
//! # Writing a widget
//!
//! A widget implements the [`Widget`] trait. Every hook has a default
//! implementation, so a widget only overrides what it needs:
//!
//! ```
//! use codevar_kuitui::prelude::*;
//!
//! /// A widget that paints a filled square.
//! struct Square {
//!     side: f32,
//! }
//!
//! impl Square {
//!     fn new(side: f32) -> Self {
//!         Self { side }
//!     }
//! }
//!
//! impl Widget for Square {
//!     fn measure(&mut self, _scope: &mut MeasureScope<'_>, _c: Constraints) -> Size {
//!         Size::new(self.side, self.side)
//!     }
//!
//!     fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
//!         scope.fill_rect(bounds, Color::RED);
//!     }
//! }
//! ```
//!
//! # Composing
//!
//! [`kuitui!`] builds a tree with a Compose-like syntax:
//!
//! ```
//! use codevar_kuitui::prelude::*;
//!
//! let tree = kuitui! {
//!     Column(spacing: 8.0) {
//!         Text("Codevar")
//!         Spacer(height: 12.0)
//!         Text("Kuitui")
//!     }
//! };
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod color;
pub mod compose;
pub mod event;
pub mod geometry;
mod macros;
pub mod node;
pub mod paint;
/// Compiled UI shaders.
///
/// `build.rs` compiles every GLSL source under the crate's `shaders/`
/// directory to SPIR-V (`SHADERS_COMPILED`) and embeds the images here,
/// so a backend builds its UI pipelines from the same programs instead
/// of shipping private copies. When `glslc` was unavailable at build
/// time the modules are empty and [`shaders::SHADERS_COMPILED`] is
/// `false`.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}
pub mod state;
pub mod text;
pub mod theme;
pub mod widget;
pub mod widgets;
pub mod window;

pub use color::{Color, ColorScheme};
pub use compose::{Composer, Composition, Recomposition};
pub use event::{Event, EventResult, KeyEvent, PointerButton, PointerEvent, PointerEventKind};
pub use geometry::{Alignment, Constraints, EdgeInsets, Offset, Rect, Size};
#[doc(hidden)]
pub use macros::__private;
pub use node::{Modifier, Node};
pub use paint::{
    ClippedOps, ColorTextOp, CornerRadii, Draw, DrawCommand, DrawList, DrawVisitor, LineOp, Paint,
    PaintStyle, RectOp, TextOp,
};
pub use state::{DirtyState, MutableState, State};
pub use text::{FontMetrics, FontWeight, TextSpan, TextStyle};
pub use theme::{Theme, ThemeMode, Typography};
pub use widget::{DrawScope, EventScope, Key, MeasureScope, PlaceScope, Widget};
pub use widgets::{ColorableText, Column, Frame, Row, Spacer, Stack, Text};
pub use window::{ResizeEdge, ResizeHandle, TitleBar, WindowButton, WindowFrame};

/// The traits and types a widget implementation usually needs.
///
/// Importing the prelude brings in the geometry, the theme, the
/// scopes and the base components, so that a widget file does not have
/// to spell out where each of them comes from.
pub mod prelude {
    pub use crate::color::{Color, ColorScheme};
    pub use crate::compose::{Composer, Composition, Recomposition};
    pub use crate::event::{
        Event, EventResult, KeyEvent, Modifiers, PointerButton, PointerButtons, PointerEvent,
        PointerEventKind,
    };
    pub use crate::geometry::{Alignment, Constraints, EdgeInsets, Horizontal, Offset, Rect, Size, Vertical};
    pub use crate::kuitui;
    pub use crate::node::{Modifier, Node};
    pub use crate::paint::{
        ClippedOps, ColorTextOp, CornerRadii, Draw, DrawCommand, DrawList, DrawVisitor, LineOp, Paint,
        PaintStyle, RectOp, TextOp,
    };
    pub use crate::state::{DirtyState, MutableState, State};
    pub use crate::text::{
        ApproximateFontMetrics, FontMetrics, FontWeight, TextAlign, TextContent, TextSpan, TextStyle,
    };
    pub use crate::theme::{Theme, ThemeMode, Typography};
    pub use crate::widget::{Children, DrawScope, EventScope, Key, MeasureScope, PlaceScope, Widget};
    pub use crate::widgets::{ColorableText, Column, Frame, Row, Spacer, Stack, Text};
    pub use crate::window::{ResizeEdge, ResizeHandle, TitleBar, WindowButton, WindowFrame};
}
