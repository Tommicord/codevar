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

//! Declarative widget system, built on the compositor pipes
//! of [`codevar_gpu_core`].
//!
//! # Design
//!
//! The crate separates *what the UI looks like* from *how it is built*:
//!
//! * **Description layer** — a [`widget::Widget`] only describes its
//!   appearance: an intrinsic size ([`widget::Widget::measure`]), an
//!   arranged rectangle ([`widget::Widget::arrange`]) and a stream of
//!   [`scene::DrawOp`]s ([`widget::Widget::describe`]). Widgets never
//!   touch Vulkan, buffers or kernel.
//! * **Tree driver** — [`tree::WidgetTree`] owns the widget tree and
//!   drives every lifecycle state ([`widget::WidgetPhase`]): `build`,
//!   `startLifecycle`, `beforePass`, `afterPass`, `resume`, `pause`,
//!   `endLifecycle` and `dispose`. It performs layout, hit testing and
//!   collects one [`scene::Scene`] per frame.
//! * **Backend** — [`render::WidgetPipe`] is a
//!   `codevar_gpu_core::gpu_base::PipeSource` that records the frame's
//!   [`scene::Scene`] into its own offscreen target with a push-constant
//!   quad pipeline and hands the target to the compositor's mix pass.
//!
//! Host code keeps a [`control::WidgetHandle`] after registering the
//! pipe, so input and lifecycle commands still reach the tree while the
//! compositor owns the pipe.
//!
//! # Modifier
//!
//! Styling and layout tweaks travel through the [`modifier::Modifier`]
//! of every widget (padding, margin, background, color, rasterizer,
//! interaction and visibility). The modifier deliberately stops at
//! basic layout/style facts: no gradients, no animations, no shadows.
//!
//! # References
//!
//! The API follows established widget frameworks:
//!
//! * **Jetpack Compose** ([androidx/androidx](https://github.com/androidx/androidx))
//!   — composable descriptions, chained `Modifier` builders, lifecycle
//!   hooks around composition and drawing.
//! * **Flutter** ([flutter/flutter](https://github.com/flutter/flutter))
//!   — `Widget`/`Element` split with `initState`/`dispose` style
//!   lifecycle and immutable widget descriptions.
//! * **Iced** ([iced-rs/iced](https://github.com/iced-rs/iced)) — an
//!   owned widget tree driven by a retained renderer.
//! * **GPUI** ([zed-industries/gpui](https://github.com/zed-industries/gpui))
//!   — element trees with explicit layout and paint phases.
//! * **Slint** ([slint-ui/slint](https://github.com/slint-ui/slint)) —
//!   declarative component descriptions kept separate from rendering.
//! * **zgui** ([zortax/zgui](https://github.com/zortax/zgui)) — a
//!   retained widget tree feeding staged, renderer-agnostic passes.
//!
//! # Example
//!
//! ```
//! use codevar_gpu_widget::geometry::{Color, Size};
//! use codevar_gpu_widget::modifier::Modifier;
//! use codevar_gpu_widget::scene::PassContext;
//! use codevar_gpu_widget::tree::WidgetTree;
//! use codevar_gpu_widget::widget::{BuildContext, WidgetPhase};
//! use codevar_gpu_widget::widgets::{Container, ContainerParams};
//!
//! let root = Container::new(ContainerParams::default())
//!     .with_modifier(Modifier::new().background(Color::rgb(0.1, 0.1, 0.1)));
//! let mut tree = WidgetTree::new(Box::new(root));
//! let mut registry = TextureRegistry::new();
//! tree.mount(&mut registry, Size::new(640.0, 480.0))
//!     .unwrap_or_else(|error| panic!("mount failed: {error}"));
//! assert_eq!(tree.phase(), WidgetPhase::Started);
//! ```
#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod control;
pub mod error;
pub mod geometry;
pub mod layout;
pub mod modifier;
pub mod prelude;
pub mod render;
pub mod scene;
pub mod texture;
pub mod traits;
pub mod tree;
pub mod widget;
pub mod widgets;
