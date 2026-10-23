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

//! Replay of a kuitui [`DrawList`] into Vulkan-ready vertices.
//!
//! A frame is recorded by kuitui as a FIFO of [`DrawCommand`]s (see
//! [`codevar_kuitui::paint`]); the GPU backend replays that queue in
//! order. This module owns the CPU half of the replay: it drains the
//! queue and expands each operation into a triangle list ([`UiVertex`]),
//! recording one [`DrawSpan`] per operation so the renderer can bind the
//! operation's resolved clip as a scissor rectangle without re-deriving
//! the clip stack.
//!
//! The vertex layout matches the `ui` shader pair compiled by
//! `codevar-kuitui` (`ui.vert`/`ui.frag`): location 0 is the position in
//! logical pixels and location 1 is the colour. The viewport mapping and
//! the clip scissor are applied by the renderer, so the expansion stays a
//! pure function of the display list and is unit-testable without a
//! device.
//!
//! # Scope
//!
//! Geometry operations ([`DrawOp::Rect`] and [`DrawOp::Line`]) are
//! expanded here. A [`DrawOp::Text`] run carries no glyph positions — it
//! is shaped by the font backend, which owns the glyph atlas and the
//! `text` shader — so it is counted but not expanded ([`UiGeometry::text_runs`]).
//! Rounded corners are likewise not representable by the colour-only `ui`
//! shader and are drawn square; both limits are recorded here rather than
//! silently ignored.

use alloc::vec::Vec;

use codevar_kuitui::geometry::{Offset, Rect};
use codevar_kuitui::paint::{DrawCommand, DrawList, DrawOp, Paint, PaintStyle};

/// A position (in logical pixels) and colour of one UI vertex.
///
/// The layout is the input of `ui.vert`: a `vec2` at location 0 and a
/// `vec4` at location 1, packed without padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiVertex {
    /// Position in logical pixels, top-left origin.
    pub position: [f32; 2],
    /// Straight (non-premultiplied) RGBA colour.
    pub color: [f32; 4],
}

/// One operation's slice of a [`UiGeometry`] vertex buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawSpan {
    /// Index of the first vertex of the operation.
    pub first_vertex: u32,
    /// Number of vertices (a multiple of three) to draw.
    pub vertex_count: u32,
    /// Clip resolved by kuitui when the operation was recorded; `None`
    /// means the whole surface.
    pub clip: Option<Rect>,
}

/// The triangle list and draw spans of one replayed display list.
///
/// Reuse a single value across frames: [`UiGeometry::expand`] clears and
/// refills it, so a steady-state frame keeps the allocations.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct UiGeometry {
    /// Triangle-list vertices of every geometry operation.
    vertices: Vec<UiVertex>,
    /// One span per geometry operation, in paint order.
    spans: Vec<DrawSpan>,
    /// Number of text runs skipped because a font backend replays them.
    text_runs: u32,
}

impl UiGeometry {
    /// Creates an empty geometry buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            spans: Vec::new(),
            text_runs: 0,
        }
    }

    /// Empties the geometry, keeping its allocations.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.spans.clear();
        self.text_runs = 0;
    }

    /// The triangle-list vertices, in paint order.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        &self.vertices
    }

    /// The draw spans, in paint order.
    #[inline]
    #[must_use]
    pub fn spans(&self) -> &[DrawSpan] {
        &self.spans
    }

    /// Number of text runs skipped during the last expansion.
    #[inline]
    #[must_use]
    pub const fn text_runs(&self) -> u32 {
        self.text_runs
    }

    /// Whether the geometry holds no draw span.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Drains `list` in FIFO order and expands every command into it.
    ///
    /// The list is left empty: the queue is a one-shot hand-off from the
    /// UI thread to the renderer, which clears and re-records it next
    /// frame.
    pub fn expand(&mut self, list: &mut DrawList) {
        self.clear();
        while let Some(command) = list.pop_front() {
            self.push_command(command);
        }
    }

    /// Expands one command, appending its vertices and its span.
    pub fn push_command(&mut self, command: DrawCommand) {
        let first_vertex = self.vertices.len();
        match &command.op {
            DrawOp::Rect { rect, paint, .. } => self.push_rect(*rect, *paint),
            DrawOp::Line { from, to, paint } => self.push_line(*from, *to, *paint),
            DrawOp::Text { .. } => self.text_runs += 1,
        }
        let vertex_count = self.vertices.len() - first_vertex;
        if vertex_count == 0 {
            return;
        }
        self.spans.push(DrawSpan {
            first_vertex: first_vertex as u32,
            vertex_count: vertex_count as u32,
            clip: command.clip,
        });
    }

    /// Expands a rectangle: a filled quad, or four quads for an outline.
    fn push_rect(&mut self, rect: Rect, paint: Paint) {
        let color = vertex_color(paint);
        let width = paint.stroke_width();
        if width <= 0.0 || matches!(paint.style, PaintStyle::Fill) {
            self.push_quad(
                Offset::new(rect.left(), rect.top()),
                Offset::new(rect.right(), rect.top()),
                Offset::new(rect.right(), rect.bottom()),
                Offset::new(rect.left(), rect.bottom()),
                color,
            );
            return;
        }
        // An outline wider than half the rectangle would invert itself;
        // clamp it so the four edges keep their orientation.
        let width = width
            .min(rect.size.width * 0.5)
            .min(rect.size.height * 0.5);
        if width <= 0.0 {
            return;
        }
        let (left, top, right, bottom) = (rect.left(), rect.top(), rect.right(), rect.bottom());
        // Top, bottom, left and right edges, each an axis-aligned quad.
        self.push_quad(
            Offset::new(left, top),
            Offset::new(right, top),
            Offset::new(right, top + width),
            Offset::new(left, top + width),
            color,
        );
        self.push_quad(
            Offset::new(left, bottom - width),
            Offset::new(right, bottom - width),
            Offset::new(right, bottom),
            Offset::new(left, bottom),
            color,
        );
        self.push_quad(
            Offset::new(left, top + width),
            Offset::new(left + width, top + width),
            Offset::new(left + width, bottom - width),
            Offset::new(left, bottom - width),
            color,
        );
        self.push_quad(
            Offset::new(right - width, top + width),
            Offset::new(right, top + width),
            Offset::new(right, bottom - width),
            Offset::new(right - width, bottom - width),
            color,
        );
    }

    /// Expands a line into the quad `from`–`to` spans at the paint width.
    fn push_line(&mut self, from: Offset, to: Offset, paint: Paint) {
        let color = vertex_color(paint);
        let width = if paint.stroke_width() > 0.0 {
            paint.stroke_width()
        } else {
            1.0
        };
        let delta = to.sub(from);
        let length = from.distance_to(to);
        if length <= f32::EPSILON {
            return;
        }
        // Unit normal of the line, scaled to half the stroke width.
        let half = width * 0.5;
        let normal = Offset::new(-delta.y / length * half, delta.x / length * half);
        let start = from.sub(normal);
        let end = to.sub(normal);
        self.push_quad(
            start,
            end,
            end.add(normal.scale(2.0)),
            start.add(normal.scale(2.0)),
            color,
        );
    }

    /// Appends the two triangles `a, b, c` and `a, c, d`.
    fn push_quad(&mut self, a: Offset, b: Offset, c: Offset, d: Offset, color: [f32; 4]) {
        self.vertices.push(UiVertex {
            position: [a.x, a.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [b.x, b.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [c.x, c.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [a.x, a.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [c.x, c.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [d.x, d.y],
            color,
        });
    }
}

/// The straight RGBA vertex colour of a paint.
#[inline]
fn vertex_color(paint: Paint) -> [f32; 4] {
    [
        paint.color.red,
        paint.color.green,
        paint.color.blue,
        paint.color.alpha,
    ]
}

#[cfg(test)]
mod tests {
    //! Display-list expansion: geometry, span order and clip carry-over.

    use super::*;
    use alloc::rc::Rc;
    use codevar_kuitui::color::Color;
    use codevar_kuitui::paint::CornerRadii;
    use codevar_kuitui::text::TextStyle;

    /// A black fill, the colour most tests do not care about.
    fn fill() -> Paint {
        Paint::fill(Color::BLACK)
    }

    #[test]
    fn a_filled_rectangle_expands_to_two_triangles() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::new(4.0, 8.0), 10.0, 20.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.vertices().len(), 6);
        assert_eq!(geometry.spans().len(), 1);
        let span = geometry.spans()[0];
        assert_eq!(span.first_vertex, 0);
        assert_eq!(span.vertex_count, 6);
        assert_eq!(span.clip, None);

        let first = geometry.vertices()[0];
        assert_eq!(first.position, [4.0, 8.0]);
        assert_eq!(
            first.color,
            [
                Color::RED.red,
                Color::RED.green,
                Color::RED.blue,
                Color::RED.alpha
            ]
        );
        assert!(list.is_empty(), "expand drains the FIFO queue");
    }

    #[test]
    fn spans_follow_the_paint_order_and_tile_the_buffer() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
            CornerRadii::ZERO,
            fill(),
        );
        list.line(
            Offset::ZERO,
            Offset::new(4.0, 0.0),
            Paint::stroke(Color::WHITE, 2.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        let spans = geometry.spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].first_vertex, 0);
        assert_eq!(spans[0].vertex_count, 6);
        assert_eq!(spans[1].first_vertex, 6);
        assert_eq!(spans[1].vertex_count, 6);
        assert_eq!(geometry.vertices().len(), 12);
    }

    #[test]
    fn a_span_carries_the_command_clip() {
        let mut list = DrawList::new();
        let clip = Rect::from_min_size(Offset::new(5.0, 5.0), 20.0, 20.0);
        list.push_clip(clip);
        list.rect(
            Rect::from_min_size(Offset::ZERO, 100.0, 100.0),
            CornerRadii::ZERO,
            fill(),
        );
        list.pop_clip();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
            CornerRadii::ZERO,
            fill(),
        );

        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        assert_eq!(geometry.spans()[0].clip, Some(clip));
        assert_eq!(geometry.spans()[1].clip, None);
    }

    #[test]
    fn an_outlined_rectangle_expands_to_four_edges() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::stroke(Color::WHITE, 2.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.spans().len(), 1);
        assert_eq!(geometry.spans()[0].vertex_count, 24, "four quads");
    }

    #[test]
    fn an_outline_wider_than_the_rectangle_is_clamped() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
            CornerRadii::ZERO,
            Paint::stroke(Color::WHITE, 100.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        // The four edges still get emitted, none of them inverted.
        assert_eq!(geometry.spans()[0].vertex_count, 24);
        assert!(
            geometry
                .vertices()
                .iter()
                .all(|vertex| vertex.position[0] >= 0.0 && vertex.position[0] <= 4.0)
        );
    }

    #[test]
    fn a_line_expands_to_a_quad_around_its_axis() {
        let mut list = DrawList::new();
        list.line(
            Offset::new(0.0, 0.0),
            Offset::new(10.0, 0.0),
            Paint::stroke(Color::WHITE, 2.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.spans()[0].vertex_count, 6);
        // A horizontal line of width 2 is offset by ±1 on the y axis.
        let ys: Vec<f32> = geometry
            .vertices()
            .iter()
            .map(|vertex| vertex.position[1])
            .collect();
        assert!(
            ys.iter()
                .all(|y| (*y - -1.0).abs() < 1e-6 || (*y - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn a_degenerate_line_draws_nothing() {
        let mut list = DrawList::new();
        list.line(Offset::ZERO, Offset::ZERO, Paint::stroke(Color::WHITE, 2.0));
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        assert!(geometry.is_empty());
        assert_eq!(geometry.vertices().len(), 0);
    }

    #[test]
    fn a_text_run_is_counted_but_not_expanded() {
        let mut list = DrawList::new();
        list.text(Offset::ZERO, Rc::from("hi"), TextStyle::default());
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        assert_eq!(geometry.text_runs(), 1);
        assert!(geometry.is_empty());
    }

    #[test]
    fn expanding_clears_the_previous_frame() {
        let mut geometry = UiGeometry::new();
        let mut first = DrawList::new();
        first.rect(
            Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
            CornerRadii::ZERO,
            fill(),
        );
        geometry.expand(&mut first);
        assert_eq!(geometry.vertices().len(), 6);

        let mut second = DrawList::new();
        second.line(
            Offset::ZERO,
            Offset::new(1.0, 0.0),
            Paint::stroke(Color::WHITE, 1.0),
        );
        geometry.expand(&mut second);
        assert_eq!(geometry.vertices().len(), 6);
        assert_eq!(geometry.spans().len(), 1);
    }
}
