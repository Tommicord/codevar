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

use alloc::vec::Vec;
use codevar_kuitui::geometry::{Offset, Rect};
use codevar_kuitui::paint::{DrawList, DrawOp, PaintStyle};

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

/// The vertices of a frame of the user interface.
///
/// The builder owns the vertex and the index buffers of the frame and
/// fills them from a display list. It is reused across frames: the
/// buffers keep their memory, so a steady-state frame does not
/// allocate.
#[derive(Debug, Default)]
pub struct UiQuadBuilder {
    /// The vertices of the frame.
    vertices: Vec<UiVertex>,
    /// The indices of the frame.
    indices: Vec<u32>,
    /// The number of glyphs the display list asked for and the atlas
    /// did not hold.
    missing_glyphs: u32,
}

impl UiQuadBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            missing_glyphs: 0,
        }
    }

    /// Empties the builder, keeping its memory.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.missing_glyphs = 0;
    }

    /// The vertices of the frame.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        &self.vertices
    }

    /// The indices of the frame.
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The number of quads of the frame.
    #[inline]
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    /// The number of glyphs the display list asked for and the atlas
    /// did not hold.
    #[inline]
    #[must_use]
    pub const fn missing_glyphs(&self) -> u32 {
        self.missing_glyphs
    }

    /// The size of the vertex buffer of the frame, in bytes.
    #[inline]
    #[must_use]
    pub fn vertex_bytes(&self) -> u64 {
        self.vertices.len() as u64 * u64::from(UI_VERTEX_SIZE)
    }

    /// Turns a display list into the vertices of a frame.
    ///
    /// # Errors
    ///
    /// Returns [`UiError::UnsupportedOperation`] when the display list
    /// holds an operation the compositor cannot draw.
    pub fn build(&mut self, list: &DrawList, glyphs: &dyn GlyphSource) -> Result<(), UiError> {
        self.clear();
        for (op, clip) in list.clipped_ops() {
            match op {
                DrawOp::Rect { rect, radii, paint } => {
                    self.push_rect(*rect, *radii, paint.color, paint.style, clip)
                }
                DrawOp::Line { from, to, paint } => {
                    self.push_line(*from, *to, paint.color, paint.style, clip)?;
                }
                DrawOp::Text { origin, text, style } => {
                    self.push_text(*origin, text, style.color, style.font_size, clip, glyphs);
                }
            }
        }
        Ok(())
    }

    /// Adds a rectangle to the frame.
    fn push_rect(
        &mut self,
        rect: Rect,
        radii: codevar_kuitui::paint::CornerRadii,
        color: codevar_kuitui::color::Color,
        style: PaintStyle,
        clip: Option<Rect>,
    ) {
        let (kind, width) = match style {
            PaintStyle::Fill => (kind::FILL, 0.0),
            PaintStyle::Stroke(width) => (kind::STROKE, width),
        };
        let params = if kind == kind::GLYPH {
            [0.0; 4]
        } else {
            [width, 0.0, 0.0, 0.0]
        };
        let corner = [
            radii.top_left,
            radii.top_right,
            radii.bottom_right,
            radii.bottom_left,
        ];
        let clip = clip_rect(clip);
        self.push_quad(
            rect.origin,
            rect.size,
            corner,
            color,
            [kind, params[0], params[1], params[2]],
            clip,
        );
    }

    /// Adds a line to the frame.
    ///
    /// The line becomes a quad of the width of the stroke, centered on
    /// the segment.
    ///
    /// # Errors
    ///
    /// Returns [`UiError::DegenerateLine`] when the two ends of the
    /// line are the same point.
    fn push_line(
        &mut self,
        from: Offset,
        to: Offset,
        color: codevar_kuitui::color::Color,
        style: PaintStyle,
        clip: Option<Rect>,
    ) -> Result<(), UiError> {
        let width = match style {
            PaintStyle::Fill => 1.0,
            PaintStyle::Stroke(width) => width.max(0.0),
        };
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        let length = libm::sqrtf(dx * dx + dy * dy);
        if length <= f32::EPSILON {
            return Err(UiError::DegenerateLine);
        }
        // The normal of the segment, scaled by half the width of the
        // stroke.
        let half = width * 0.5;
        let nx = -dy / length * half;
        let ny = dx / length * half;
        let corners = [
            Offset::new(from.x + nx, from.y + ny),
            Offset::new(to.x + nx, to.y + ny),
            Offset::new(to.x - nx, to.y - ny),
            Offset::new(from.x - nx, from.y - ny),
        ];
        let min_x = corners
            .iter()
            .map(|corner| corner.x)
            .fold(f32::INFINITY, f32::min);
        let min_y = corners
            .iter()
            .map(|corner| corner.y)
            .fold(f32::INFINITY, f32::min);
        let max_x = corners
            .iter()
            .map(|corner| corner.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let max_y = corners
            .iter()
            .map(|corner| corner.y)
            .fold(f32::NEG_INFINITY, f32::max);
        let rect = Rect::from_min_size(Offset::new(min_x, min_y), max_x - min_x, max_y - min_y);
        self.push_quad(
            rect.origin,
            rect.size,
            [0.0; 4],
            color,
            [kind::FILL, 0.0, 0.0, 0.0],
            clip_rect(clip),
        );
        Ok(())
    }

    /// Adds a run of text to the frame.
    ///
    /// Every character of the run becomes a quad of the atlas; a
    /// character the atlas does not hold is skipped and counted.
    fn push_text(
        &mut self,
        origin: Offset,
        text: &str,
        color: codevar_kuitui::color::Color,
        font_size: f32,
        clip: Option<Rect>,
        glyphs: &dyn GlyphSource,
    ) {
        let size = libm::roundf(font_size.max(0.0)) as u16;
        let mut pen = origin.x;
        for character in text.chars() {
            let Some(glyph) = glyphs.glyph(character, size) else {
                self.missing_glyphs += 1;
                continue;
            };
            let rect = Rect::from_min_size(
                Offset::new(pen + glyph.bearing_x, origin.y - glyph.bearing_y),
                glyph.width,
                glyph.height,
            );
            self.push_quad(
                rect.origin,
                rect.size,
                [0.0; 4],
                color,
                [kind::GLYPH, glyph.uv[0], glyph.uv[1], glyph.uv[2]],
                clip_rect(clip),
            );
            // The atlas rectangle of a glyph is four numbers, but the
            // vertex format carries three of them; the fourth one is
            // recovered by the shader from the size of the quad.
            pen += glyph.advance;
        }
    }

    /// Adds a quad of four vertices and six indices to the frame.
    fn push_quad(
        &mut self,
        origin: Offset,
        size: codevar_kuitui::geometry::Size,
        radii: [f32; 4],
        color: codevar_kuitui::color::Color,
        params: [f32; 4],
        clip: [f32; 4],
    ) {
        let base = self.vertices.len() as u32;
        let corners = [
            origin,
            Offset::new(origin.x + size.width, origin.y),
            Offset::new(origin.x + size.width, origin.y + size.height),
            Offset::new(origin.x, origin.y + size.height),
        ];
        for corner in corners {
            self.vertices.push(UiVertex {
                position: [corner.x, corner.y],
                size: [size.width, size.height],
                radii,
                color: [color.red, color.green, color.blue, color.alpha],
                params,
                clip,
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// The clipping rectangle of a primitive, as the shader sees it.
fn clip_rect(clip: Option<Rect>) -> [f32; 4] {
    match clip {
        Some(rect) => [rect.origin.x, rect.origin.y, rect.size.width, rect.size.height],
        None => [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
    }
}

#[cfg(test)]
mod tests {
    //! The conversion of a display list into the vertices of a frame.

    use super::*;
    use alloc::vec;
    use codevar_kuitui::color::Color;
    use codevar_kuitui::geometry::Size;
    use codevar_kuitui::paint::{CornerRadii, Paint};

    /// A source of glyphs that holds one square glyph per character.
    struct Squares;

    impl GlyphSource for Squares {
        fn glyph(&self, character: char, _size: u16) -> Option<Glyph> {
            if character == ' ' {
                return None;
            }
            Some(Glyph {
                uv: [0.0, 0.0, 1.0, 1.0],
                width: 8.0,
                height: 12.0,
                bearing_x: 0.0,
                bearing_y: 10.0,
                advance: 8.0,
            })
        }
    }

    fn rect_list() -> DrawList {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::new(1.0, 2.0), 10.0, 20.0),
            CornerRadii::uniform(4.0),
            Paint::fill(Color::RED),
        );
        list
    }

    #[test]
    fn a_rectangle_becomes_one_quad() {
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&rect_list(), &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.quad_count(), 1);
        assert_eq!(builder.vertices().len(), 4);
        assert_eq!(builder.indices().len(), 6);
    }

    #[test]
    fn a_rectangle_carries_its_geometry_and_its_color() {
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&rect_list(), &Squares)
            .expect("the list is drawable");
        let vertex = builder.vertices()[0];
        assert_eq!(vertex.position, [1.0, 2.0]);
        assert_eq!(vertex.size, [10.0, 20.0]);
        assert_eq!(vertex.radii, [4.0, 4.0, 4.0, 4.0]);
        assert_eq!(vertex.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(vertex.params[0], kind::FILL);
    }

    #[test]
    fn a_stroke_carries_its_width() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::stroke(Color::BLACK, 2.0),
        );
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&list, &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.vertices()[0].params[0], kind::STROKE);
        assert_eq!(builder.vertices()[0].params[1], 2.0);
    }

    #[test]
    fn a_line_becomes_a_quad_of_the_width_of_the_stroke() {
        let mut list = DrawList::new();
        list.line(
            Offset::new(0.0, 0.0),
            Offset::new(10.0, 0.0),
            Paint::stroke(Color::BLACK, 2.0),
        );
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&list, &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.quad_count(), 1);
        let vertex = builder.vertices()[0];
        assert_eq!(vertex.size, [10.0, 2.0]);
        assert_eq!(vertex.position, [0.0, -1.0]);
    }

    #[test]
    fn a_line_without_a_direction_is_an_error() {
        let mut list = DrawList::new();
        list.line(
            Offset::new(3.0, 3.0),
            Offset::new(3.0, 3.0),
            Paint::stroke(Color::BLACK, 1.0),
        );
        let mut builder = UiQuadBuilder::new();
        assert_eq!(builder.build(&list, &Squares), Err(UiError::DegenerateLine));
    }

    #[test]
    fn every_character_of_a_text_becomes_a_quad() {
        let mut list = DrawList::new();
        list.text(
            Offset::new(0.0, 20.0),
            codevar_kuitui::text::TextSpan::new("abc").text,
            codevar_kuitui::text::TextStyle::new(16.0),
        );
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&list, &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.quad_count(), 3);
        assert_eq!(builder.vertices()[0].params[0], kind::GLYPH);
    }

    #[test]
    fn a_character_the_atlas_does_not_hold_is_skipped_and_counted() {
        let mut list = DrawList::new();
        list.text(
            Offset::new(0.0, 20.0),
            codevar_kuitui::text::TextSpan::new("a b").text,
            codevar_kuitui::text::TextStyle::new(16.0),
        );
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&list, &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.quad_count(), 2);
        assert_eq!(builder.missing_glyphs(), 1);
    }

    #[test]
    fn the_clip_of_an_operation_travels_with_its_vertices() {
        let mut list = DrawList::new();
        list.push_clip(Rect::from_min_size(Offset::new(5.0, 5.0), 10.0, 10.0));
        list.rect(
            Rect::from_min_size(Offset::ZERO, 100.0, 100.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&list, &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.vertices()[0].clip, [5.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn an_operation_without_a_clip_is_not_clipped() {
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&rect_list(), &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.vertices()[0].clip[2], f32::INFINITY);
    }

    #[test]
    fn clearing_the_builder_empties_it() {
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&rect_list(), &Squares)
            .expect("the list is drawable");
        builder.clear();
        assert_eq!(builder.quad_count(), 0);
        assert!(builder.vertices().is_empty());
    }

    #[test]
    fn the_size_of_a_vertex_matches_the_format() {
        assert_eq!(core::mem::size_of::<UiVertex>(), UI_VERTEX_SIZE as usize);
    }

    #[test]
    fn the_indices_of_two_quads_do_not_overlap() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );
        list.rect(
            Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );
        let mut builder = UiQuadBuilder::new();
        builder
            .build(&list, &Squares)
            .expect("the list is drawable");
        assert_eq!(builder.indices()[0], 0);
        assert_eq!(builder.indices()[6], 4);
        let _ = vec![0_u8; 0];
        let _ = Size::ZERO;
    }
}
