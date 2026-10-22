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
//!    closure. Re-running the closure *recomposes* the tree: nodes are
//!    matched against their previous counterparts by [`Key`], so widget
//!    state survives a recomposition. Values that must outlive a
//!    recomposition are created with [`Composer::remember`].
//! 2. **Layout** — the tree is measured and placed in two passes, with
//!    the same *measure / place* split Jetpack Compose uses
//!    ([`Widget::measure`], [`Widget::place`]).
//! 3. **Painting** — the placed tree is recorded into a [`DrawList`],
//!    a flat sequence of [`DrawOp`]s that a backend replays.
//!
//! # Writing a widget
//!
//! A widget implements the [`Widget`] trait. Every hook has a default
//! implementation, so a widget only overrides what it needs:
//!
//! ```ignore
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
//! ```ignore
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
#![forbid(unsafe_code)]

extern crate alloc;

pub mod color;
pub mod event;
pub mod geometry;
pub mod paint;
pub mod state;
pub mod text;
pub mod theme;
pub mod widget;

pub use color::{Color, ColorScheme};
pub use event::{Event, EventResult, KeyEvent, PointerButton, PointerEvent, PointerEventKind};
pub use geometry::{Alignment, Constraints, EdgeInsets, Offset, Rect, Size};
pub use paint::{CornerRadii, DrawList, DrawOp, Paint};
pub use state::{MutableState, State};
pub use text::{FontMetrics, FontWeight, TextSpan, TextStyle};
pub use theme::{Theme, ThemeMode, Typography};
pub use widget::{DrawScope, EventScope, Key, MeasureScope, PlaceScope, Widget};
