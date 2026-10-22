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

//! The user interface of the compositor: the vertices of a frame.
//!
//! The compositor renders the user interface of the application. The
//! application lays a tree of widgets out and records it into a
//! [`codevar_kuitui::DrawList`]; this module turns that display list
//! into the vertices of the frame, and the Vulkan layer of the
//! compositor draws them.
//!
//! # The vertex format
//!
//! Every primitive of the display list becomes a quad of four
//! vertices and six indices. The vertices carry everything the
//! fragment shader needs:
//!
//! * the rectangle of the quad and the radii of its four corners, so
//!   that a rounded rectangle is one primitive;
//! * the color of the primitive;
//! * the kind of the primitive — a fill, a stroke or a glyph — and
//!   the parameters that go with it;
//! * the clipping rectangle of the primitive.
//!
//! The clipping rectangle travels with the vertices instead of with
//! the draw call, because a single frame may hold several different
//! clips and a scissor rectangle is a state of the pipeline, not of
//! the geometry.

/// The kind of a primitive, as the fragment shader sees it.
pub mod kind {
    /// The primitive is filled with its color.
    pub const FILL: f32 = 0.0;
    /// The primitive is outlined with its color.
    pub const STROKE: f32 = 1.0;
    /// The primitive is a glyph of the atlas.
    pub const GLYPH: f32 = 2.0;
}

/// A vertex of a quad of the user interface.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UiVertex {
    /// The position of the corner, in logical pixels.
    pub position: [f32; 2],
    /// The size of the quad, in logical pixels.
    pub size: [f32; 2],
    /// The radii of the four corners of the quad.
    pub radii: [f32; 4],
    /// The color of the primitive, in the `0.0 ..= 1.0` range.
    pub color: [f32; 4],
    /// The kind of the primitive and its parameters.
    ///
    /// For a fill the parameters are unused, for a stroke the first
    /// one is the width of the outline in pixels and for a glyph the
    /// four of them are the coordinates of the glyph in the atlas.
    pub params: [f32; 4],
    /// The clipping rectangle of the primitive, as `x`, `y`, `width`
    /// and `height`.
    pub clip: [f32; 4],
}

/// The size of a vertex, in bytes.
pub const UI_VERTEX_SIZE: u32 = 80;

/// A glyph of the atlas the compositor draws the text with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Glyph {
    /// The rectangle of the glyph in the atlas, as `u0`, `v0`, `u1`
    /// and `v1`.
    pub uv: [f32; 4],
    /// The width of the glyph, in logical pixels.
    pub width: f32,
    /// The height of the glyph, in logical pixels.
    pub height: f32,
    /// The horizontal distance from the origin of the text to the left
    /// edge of the glyph.
    pub bearing_x: f32,
    /// The vertical distance from the baseline to the top edge of the
    /// glyph.
    pub bearing_y: f32,
    /// The horizontal advance of the glyph.
    pub advance: f32,
}

/// The source of the glyphs of the text.
///
/// The compositor does not rasterize glyphs; the application owns the
/// font and hands the compositor the glyphs it needs through this
/// trait.
pub trait GlyphSource {
    /// The glyph of `character` at `size`, or `None` when the atlas
    /// does not hold it.
    fn glyph(&self, character: char, size: u16) -> Option<Glyph>;
}

/// An error reported while a display list is turned into vertices.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UiError {
    /// The display list holds an operation the compositor cannot
    /// draw.
    UnsupportedOperation,
    /// A line has no direction, so it has no width.
    DegenerateLine,
}

impl core::fmt::Display for UiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedOperation => f.write_str("the display list holds an unsupported operation"),
            Self::DegenerateLine => f.write_str("a line of the display list has no direction"),
        }
    }
}

impl core::error::Error for UiError {}
