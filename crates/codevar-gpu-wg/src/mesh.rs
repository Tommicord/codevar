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

//! The mesh of a frame: the pure logic that turns a display list into
//! quads.
//!
//! The display list of Kuitui is flat, so the whole frame is one draw
//! call: every rectangle, every line and every glyph becomes a quad,
//! and the shader decides what the quad looks like. This module holds
//! that translation and nothing else, so it can be tested without a
//! device.

use alloc::vec::Vec;

use codevar_kuitui::paint::PaintStyle;
use codevar_kuitui::text::ApproximateFontMetrics;
use codevar_kuitui::{
    Color, CornerRadii, DrawList, DrawOp, FontMetrics, Offset, Paint, Rect, Size, TextStyle,
};

use crate::text::GlyphAtlas;

/// The kind of a primitive, as the shader sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    /// A rectangle, filled or outlined.
    Shape,
    /// A glyph of the atlas.
    Glyph,
    /// A straight line.
    Line,
}

impl PrimitiveKind {
    /// The value the shader expects.
    const fn value(self) -> f32 {
        match self {
            Self::Shape => 0.0,
            Self::Glyph => 1.0,
            Self::Line => 2.0,
        }
    }
}

/// A vertex of the mesh of a frame.
///
/// The layout of the vertex is the layout the shader expects, so the
/// vertex is written to the vertex buffer as it is.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Vertex {
    /// The corner of the quad, in logical pixels.
    pub position: [f32; 2],
    /// The shape the corner belongs to: the origin and the size of a
    /// rectangle, or the two ends of a line.
    pub shape: [f32; 4],
    /// The radii of the four corners of a rectangle.
    pub radii: [f32; 4],
    /// The color of the primitive, not premultiplied.
    pub color: [f32; 4],
    /// The texture coordinate of the corner.
    pub uv: [f32; 2],
    /// The stroke width of the primitive and the kind of the primitive.
    pub params: [f32; 2],
    /// The clipping rectangle of the primitive, in logical pixels.
    pub clip: [f32; 4],
}

/// The mesh of a frame: the vertices and the indices of every quad.
///
/// A mesh keeps its memory between the frames, so a steady-state frame
/// performs no allocation at all.
#[derive(Debug, Default)]
pub struct Mesh {
    /// The vertices of the quads.
    vertices: Vec<Vertex>,
    /// The indices of the quads.
    indices: Vec<u32>,
}

impl Mesh {
    /// Creates an empty mesh.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Empties the mesh, keeping its memory.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// The vertices of the mesh.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// The indices of the mesh.
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The number of vertices of the mesh.
    #[inline]
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// The number of indices of the mesh.
    #[inline]
    #[must_use]
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// The number of quads of the mesh.
    #[inline]
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    /// Whether the mesh holds no quad.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Adds a quad to the mesh.
    fn push(&mut self, quad: Quad) {
        let base = u32::try_from(self.vertices.len()).unwrap_or(u32::MAX);
        for (corner, uv) in quad.corners.iter().zip(quad.uvs.iter()) {
            self.vertices.push(Vertex {
                position: *corner,
                shape: quad.shape,
                radii: quad.radii,
                color: quad.color,
                uv: *uv,
                params: quad.params,
                clip: quad.clip,
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// The description of one quad of a frame.
#[derive(Debug, Clone, Copy)]
struct Quad {
    /// The corners of the quad, in logical pixels.
    corners: [[f32; 2]; 4],
    /// The shape the quad draws.
    shape: [f32; 4],
    /// The radii of the corners of the shape.
    radii: [f32; 4],
    /// The color of the quad, not premultiplied.
    color: [f32; 4],
    /// The texture coordinate of every corner.
    uvs: [[f32; 2]; 4],
    /// The stroke width of the quad and the kind of the primitive.
    params: [f32; 2],
    /// The clipping rectangle of the quad, in logical pixels.
    clip: [f32; 4],
}

/// Fills `mesh` with the quads of `list`.
///
/// The `viewport` is the size of the target, in logical pixels: it is
/// the clipping rectangle of the operations that are not clipped by
/// the display list itself.
pub fn build_mesh(list: &DrawList, atlas: &GlyphAtlas, viewport: Size, mesh: &mut Mesh) {
    mesh.clear();
    let full = [0.0, 0.0, viewport.width, viewport.height];
    for (op, clip) in list.clipped_ops() {
        let clip = match clip {
            Some(rect) => [rect.left(), rect.top(), rect.size.width, rect.size.height],
            None => full,
        };
        match op {
            DrawOp::Rect { rect, radii, paint } => {
                if let Some(quad) = shape_quad(*rect, *radii, *paint, clip) {
                    mesh.push(quad);
                }
            }
            DrawOp::Line { from, to, paint } => {
                if let Some(quad) = line_quad(*from, *to, *paint, clip) {
                    mesh.push(quad);
                }
            }
            DrawOp::Text { origin, text, style } => {
                text_quads(atlas, *origin, text, style, clip, mesh);
            }
        }
    }
}

/// The quad of a rectangle.
fn shape_quad(rect: Rect, radii: CornerRadii, paint: Paint, clip: [f32; 4]) -> Option<Quad> {
    if rect.size.width <= 0.0 || rect.size.height <= 0.0 {
        return None;
    }
    let stroke = paint.stroke_width();
    // A stroke is centered on the outline of the shape, so the quad has
    // to reach half a stroke width beyond it.
    let grow = if stroke > 0.0 { stroke * 0.5 } else { 0.0 };
    let left = rect.left() - grow;
    let top = rect.top() - grow;
    let right = rect.right() + grow;
    let bottom = rect.bottom() + grow;
    Some(Quad {
        corners: [[left, top], [right, top], [right, bottom], [left, bottom]],
        shape: [rect.left(), rect.top(), rect.size.width, rect.size.height],
        radii: [
            radii.top_left,
            radii.top_right,
            radii.bottom_right,
            radii.bottom_left,
        ],
        color: color_of(paint.color),
        uvs: [[0.0; 2]; 4],
        params: [stroke, PrimitiveKind::Shape.value()],
        clip,
    })
}

/// The quad of a line.
fn line_quad(from: Offset, to: Offset, paint: Paint, clip: [f32; 4]) -> Option<Quad> {
    let thickness = match paint.style {
        PaintStyle::Fill => 1.0,
        PaintStyle::Stroke(width) => width,
    };
    if thickness <= 0.0 {
        return None;
    }
    let half = thickness * 0.5;
    let delta = to.sub(from);
    let length = (delta.x * delta.x + delta.y * delta.y).sqrt();
    // The normal points up, so that the corners of the quad run from
    // the top left to the bottom right, the way the other quads do.
    let normal = if length > 0.0 {
        Offset::new(delta.y / length, -delta.x / length)
    } else {
        Offset::ZERO
    };
    let above = normal.scale(half);
    let below = Offset::new(-above.x, -above.y);
    Some(Quad {
        corners: [
            [from.x + above.x, from.y + above.y],
            [to.x + above.x, to.y + above.y],
            [to.x + below.x, to.y + below.y],
            [from.x + below.x, from.y + below.y],
        ],
        shape: [from.x, from.y, to.x, to.y],
        radii: [0.0; 4],
        color: color_of(paint.color),
        uvs: [[0.0; 2]; 4],
        params: [thickness, PrimitiveKind::Line.value()],
        clip,
    })
}

/// The quads of a run of text.
///
/// A run is laid out with the advance of every glyph of the atlas. A
/// character the atlas does not hold is skipped, and the run keeps its
/// shape with the approximate metrics of Kuitui.
fn text_quads(
    atlas: &GlyphAtlas,
    origin: Offset,
    text: &str,
    style: &TextStyle,
    clip: [f32; 4],
    mesh: &mut Mesh,
) {
    let size = font_size_key(style.font_size);
    let mut pen = 0.0_f32;
    for character in text.chars() {
        let Some(glyph) = atlas.lookup(character, size) else {
            pen += approximate_advance(character, style);
            continue;
        };
        let width = glyph.width as f32;
        let height = glyph.height as f32;
        if width > 0.0 && height > 0.0 {
            let x = origin.x + pen + glyph.bearing_x as f32;
            let y = origin.y - glyph.bearing_y as f32;
            let uv = atlas.uv_rect(glyph);
            mesh.push(Quad {
                corners: [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                shape: [x, y, width, height],
                radii: [0.0; 4],
                color: color_of(style.color),
                uvs: [[uv[0], uv[1]], [uv[2], uv[1]], [uv[2], uv[3]], [uv[0], uv[3]]],
                params: [0.0, PrimitiveKind::Glyph.value()],
                clip,
            });
        }
        pen += glyph.advance + style.letter_spacing;
    }
}

/// The color of a primitive, as the shader expects it.
fn color_of(color: Color) -> [f32; 4] {
    [color.red, color.green, color.blue, color.alpha]
}

/// The key of a glyph in the atlas, for a font size.
fn font_size_key(font_size: f32) -> u16 {
    if !font_size.is_finite() || font_size <= 0.0 {
        return 0;
    }
    font_size.round().min(f32::from(u16::MAX)) as u16
}

/// The advance of a character the atlas does not hold.
fn approximate_advance(character: char, style: &TextStyle) -> f32 {
    let mut buffer = [0_u8; 4];
    let text = character.encode_utf8(&mut buffer);
    ApproximateFontMetrics.advance(text, style)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec;
    use codevar_kuitui::TextStyle;

    use crate::text::GlyphDraft;

    /// The viewport the tests draw in.
    const VIEWPORT: Size = Size::new(200.0, 100.0);

    /// An atlas that holds the glyphs `a` and `b`.
    fn atlas() -> GlyphAtlas {
        let mut atlas = GlyphAtlas::new(64, 64);
        let _ = atlas.insert(
            GlyphDraft::new('a', 16, 4, 12)
                .with_bearings(0, 12)
                .with_advance(8.0)
                .with_coverage(vec![255; 4 * 12]),
        );
        let _ = atlas.insert(
            GlyphDraft::new('b', 16, 4, 12)
                .with_bearings(0, 12)
                .with_advance(8.0)
                .with_coverage(vec![255; 4 * 12]),
        );
        atlas
    }

    #[test]
    fn every_rectangle_of_a_list_becomes_a_quad() {
        let mut list = DrawList::new();
        for index in 0..5 {
            list.rect(
                Rect::from_min_size(Offset::new(index as f32, 0.0), 10.0, 10.0),
                CornerRadii::ZERO,
                Paint::fill(Color::RED),
            );
        }
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.quad_count(), 5);
        assert_eq!(mesh.vertex_count(), 20);
        assert_eq!(mesh.index_count(), 30);
    }

    #[test]
    fn an_empty_list_produces_no_quad() {
        let mut mesh = Mesh::new();
        build_mesh(&DrawList::new(), &atlas(), VIEWPORT, &mut mesh);
        assert!(mesh.is_empty());
        assert_eq!(mesh.quad_count(), 0);
    }

    #[test]
    fn the_clipping_rectangle_of_an_operation_is_kept() {
        let mut list = DrawList::new();
        list.push_clip(Rect::from_min_size(Offset::new(10.0, 20.0), 30.0, 40.0));
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.quad_count(), 1);
        for vertex in mesh.vertices() {
            assert_eq!(vertex.clip, [10.0, 20.0, 30.0, 40.0]);
        }
    }

    #[test]
    fn an_operation_without_a_clip_is_bounded_by_the_viewport() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        for vertex in mesh.vertices() {
            assert_eq!(vertex.clip, [0.0, 0.0, 200.0, 100.0]);
        }
    }

    #[test]
    fn a_stroked_rectangle_reaches_half_a_stroke_width_beyond_its_outline() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::new(10.0, 10.0), 20.0, 20.0),
            CornerRadii::ZERO,
            Paint::stroke(Color::RED, 4.0),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        let vertices = mesh.vertices();
        assert_eq!(vertices[0].position, [8.0, 8.0]);
        assert_eq!(vertices[2].position, [32.0, 32.0]);
        // The shape the shader measures is the outline itself.
        assert_eq!(vertices[0].shape, [10.0, 10.0, 20.0, 20.0]);
        assert_eq!(vertices[0].params, [4.0, 0.0]);
    }

    #[test]
    fn the_corners_of_a_rounded_rectangle_carry_their_radii() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::new(1.0, 2.0, 3.0, 4.0),
            Paint::fill(Color::RED),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.vertices()[0].radii, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn a_line_becomes_a_quad_around_the_segment() {
        let mut list = DrawList::new();
        list.line(
            Offset::new(0.0, 0.0),
            Offset::new(10.0, 0.0),
            Paint::stroke(Color::RED, 2.0),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.quad_count(), 1);
        let vertices = mesh.vertices();
        assert_eq!(vertices[0].position, [0.0, -1.0]);
        assert_eq!(vertices[1].position, [10.0, -1.0]);
        assert_eq!(vertices[2].position, [10.0, 1.0]);
        assert_eq!(vertices[3].position, [0.0, 1.0]);
        assert_eq!(vertices[0].shape, [0.0, 0.0, 10.0, 0.0]);
        assert_eq!(vertices[0].params, [2.0, 2.0]);
    }

    #[test]
    fn a_run_of_text_becomes_one_quad_per_glyph_of_the_atlas() {
        let mut list = DrawList::new();
        list.text(Offset::new(0.0, 20.0), Rc::from("ab"), TextStyle::new(16.0));
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.quad_count(), 2);
        let vertices = mesh.vertices();
        // The first glyph sits at the origin of the run, the second one
        // after the advance of the first one.
        assert_eq!(vertices[0].position, [0.0, 8.0]);
        assert_eq!(vertices[4].position, [8.0, 8.0]);
        assert_eq!(vertices[0].params, [0.0, 1.0]);
    }

    #[test]
    fn a_character_the_atlas_does_not_hold_produces_no_quad() {
        let mut list = DrawList::new();
        list.text(Offset::new(0.0, 20.0), Rc::from("zz"), TextStyle::new(16.0));
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.quad_count(), 0);
    }

    #[test]
    fn a_rectangle_without_an_area_produces_no_quad() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 0.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.quad_count(), 0);
    }

    #[test]
    fn the_indices_of_the_mesh_address_every_quad() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        list.line(
            Offset::ZERO,
            Offset::new(10.0, 10.0),
            Paint::stroke(Color::RED, 1.0),
        );
        let mut mesh = Mesh::new();
        build_mesh(&list, &atlas(), VIEWPORT, &mut mesh);
        assert_eq!(mesh.indices(), &[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
    }
}
