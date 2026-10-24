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

//! Basic rendering widgets for the Kuitui framework.
//!
//! `codevar-kuitui-widget` sits on top of [`codevar_kuitui`] and adds
//! the widgets that paint rather than arrange: a canvas that draws
//! lines, rectangles, polygons and linear gradients, and a text widget
//! that lays out and rasterizes text with the repository's
//! `codevar-truetype-*` font stack.
//!
//! Kuitui is backend agnostic: a widget never talks to a graphics API,
//! it records [`codevar_kuitui::paint::Draw`]s into the display list
//! and a backend replays them. The widgets here follow that contract,
//! and go one step further for the effects the current op set cannot
//! express, by producing the geometry and atlases a richer backend
//! consumes directly.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod canvas;
pub mod font;
pub mod shaders;
pub mod shape;
pub mod text;

pub use canvas::{Canvas, LinearGradient, ScanlineSpan, Shape};
pub use font::{
    Atlas, AtlasRegion, ColorableGlyphAtlas, Font, FontError, GlyphAtlas, GlyphQuad, GlyphRun, RasterGlyph,
    ShapeError,
};
pub use shaders::{PushConstants, TextVertex, UiVertex};
pub use shape::{TextShaper, shape_list, shape_list_with_color};
pub use text::Text;
