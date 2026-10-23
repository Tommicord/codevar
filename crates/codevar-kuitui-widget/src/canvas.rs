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

//! A canvas that paints lines, rectangles, polygons and gradients.
//!
//! Kuitui's display list carries three primitives — [`DrawOp::Rect`],
//! [`DrawOp::Line`] and [`DrawOp::Text`] — and a backend replays exactly
//! those. This widget keeps a richer vocabulary ([`Shape`]) and
//! *expands* every shape into the op set the list understands:
//!
//! * a [`Shape::Rect`] maps to one `Rect` op;
//! * a [`Shape::Line`] maps to one `Line` op;
//! * a filled [`Shape::Polygon`] maps to one 1-pixel-high `Rect` op per
//!   scanline ([`polygon_spans`]), a stroked one to one `Line` op per
//!   edge ([`polygon_edges`]);
//! * a [`Shape::Gradient`] maps to a run of constant-color `Rect` bands
//!   ([`gradient_bands`]).
//!
//! # Extension points
//!
//! The scanline and band expansion is a deliberate fallback: it is exact
//! at the pixel grid the display list already quantizes to, but a
//! backend that has a real triangle pipeline or a gradient shader should
//! not pay for it. Such a backend reads [`Canvas::shapes`] directly and
//! consumes the original [`Shape`] and [`LinearGradient`] values, which
//! carry the resolution-independent geometry the expanded ops throw
//! away.
//!
//! [`DrawOp::Rect`]: codevar_kuitui::paint::DrawOp::Rect
//! [`DrawOp::Line`]: codevar_kuitui::paint::DrawOp::Line
//! [`DrawOp::Text`]: codevar_kuitui::paint::DrawOp::Text

use alloc::vec::Vec;

use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::{Constraints, Offset, Rect, Size};
use codevar_kuitui::paint::{CornerRadii, Paint};
use codevar_kuitui::widget::{DrawScope, MeasureScope, Widget};

/// A linear gradient between two colors along a straight axis.
///
/// The gradient is stored in its original, resolution-independent form
/// so a backend with a gradient shader can consume it directly; the
/// [`gradient_bands`] fallback only needs [`Self::color_at`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradient {
    /// The start of the gradient axis.
    pub start: Offset,
    /// The end of the gradient axis.
    pub end: Offset,
    /// The color at [`Self::start`].
    pub from: Color,
    /// The color at [`Self::end`].
    pub to: Color,
}

impl LinearGradient {
    /// Creates a gradient between two colors along an axis.
    #[inline]
    #[must_use]
    pub const fn new(start: Offset, end: Offset, from: Color, to: Color) -> Self {
        Self { start, end, from, to }
    }

    /// The color at `factor` along the axis.
    ///
    /// `factor` is clamped to `0.0 ..= 1.0`, so `0.0` yields
    /// [`Self::from`] and `1.0` yields [`Self::to`].
    #[inline]
    #[must_use]
    pub fn color_at(&self, factor: f32) -> Color {
        self.from.lerp(self.to, factor)
    }

    /// The normalized position of `point` projected onto the axis.
    ///
    /// A degenerate axis (start equal to end) reports `0.0`.
    #[must_use]
    pub fn factor_at(&self, point: Offset) -> f32 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let length_squared = dx * dx + dy * dy;
        if length_squared <= 0.0 {
            return 0.0;
        }
        let factor = ((point.x - self.start.x) * dx + (point.y - self.start.y) * dy) / length_squared;
        factor.clamp(0.0, 1.0)
    }

    /// The color at the point on the axis nearest to `point`.
    #[inline]
    #[must_use]
    pub fn color_at_point(&self, point: Offset) -> Color {
        self.color_at(self.factor_at(point))
    }
}

/// One primitive recorded in a [`Canvas`].
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// A rectangle, sharp or rounded, filled or stroked.
    Rect {
        /// The rectangle in canvas coordinates.
        rect: Rect,
        /// The radii of its corners.
        radii: CornerRadii,
        /// The paint it is drawn with.
        paint: Paint,
    },
    /// A straight line of a given width.
    Line {
        /// The start of the line.
        from: Offset,
        /// The end of the line.
        to: Offset,
        /// The color of the line.
        color: Color,
        /// The width of the line in pixels.
        width: f32,
    },
    /// A closed polygon, filled or stroked.
    ///
    /// The points are given in canvas coordinates and implicitly close
    /// from the last point back to the first.
    Polygon {
        /// The corners of the polygon, in order.
        points: Vec<Offset>,
        /// The paint it is drawn with.
        paint: Paint,
    },
    /// A rectangle filled with a linear gradient.
    ///
    /// `steps` is the number of constant-color bands the fallback emits
    /// along the dominant axis of the `gradient`.
    Gradient {
        /// The rectangle to fill.
        rect: Rect,
        /// The gradient painted across it.
        gradient: LinearGradient,
        /// The number of bands the display-list fallback emits.
        steps: u32,
    },
}

/// A horizontal run of a scanline-filled polygon.
///
/// `y` is the top edge of the one-pixel band, `left` and `right` are the
/// band's horizontal extent, all in the coordinate space of the polygon.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScanlineSpan {
    /// The top edge of the one-pixel band.
    pub y: f32,
    /// The left edge of the band.
    pub left: f32,
    /// The right edge of the band.
    pub right: f32,
}

/// A widget that paints a fixed size canvas from a list of [`Shape`]s.
///
/// The canvas measures to the size it was created with, clamped by the
/// constraints it is given, and paints every shape in insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct Canvas {
    /// The size the canvas asks for.
    size: Size,
    /// The shapes to paint, in paint order.
    shapes: Vec<Shape>,
}

impl Canvas {
    /// Creates an empty canvas of the given size.
    #[inline]
    #[must_use]
    pub const fn new(size: Size) -> Self {
        Self {
            size,
            shapes: Vec::new(),
        }
    }

    /// The size the canvas asks for.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Resizes the canvas.
    #[inline]
    pub const fn set_size(&mut self, size: Size) {
        self.size = size;
    }

    /// The shapes to paint, in paint order.
    #[inline]
    #[must_use]
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }

    /// Removes every shape, keeping the allocated memory.
    pub fn clear(&mut self) {
        self.shapes.clear();
    }

    /// Records an already-built shape.
    pub fn push(&mut self, shape: Shape) -> &mut Self {
        self.shapes.push(shape);
        self
    }

    /// Records a rectangle, sharp or rounded, filled or stroked.
    pub fn rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) -> &mut Self {
        self.push(Shape::Rect { rect, radii, paint })
    }

    /// Records a filled rectangle with sharp corners.
    pub fn fill_rect(&mut self, rect: Rect, color: Color) -> &mut Self {
        self.rect(rect, CornerRadii::ZERO, Paint::fill(color))
    }

    /// Records a filled rectangle with rounded corners.
    pub fn fill_rounded_rect(&mut self, rect: Rect, radii: CornerRadii, color: Color) -> &mut Self {
        self.rect(rect, radii, Paint::fill(color))
    }

    /// Records a stroked rectangle.
    pub fn stroke_rect(&mut self, rect: Rect, radii: CornerRadii, color: Color, width: f32) -> &mut Self {
        self.rect(rect, radii, Paint::stroke(color, width))
    }

    /// Records a straight line.
    pub fn line(&mut self, from: Offset, to: Offset, color: Color, width: f32) -> &mut Self {
        self.push(Shape::Line {
            from,
            to,
            color,
            width,
        })
    }

    /// Records a closed polygon, filled or stroked.
    pub fn polygon(&mut self, points: Vec<Offset>, paint: Paint) -> &mut Self {
        self.push(Shape::Polygon { points, paint })
    }

    /// Records a rectangle filled with a linear gradient.
    pub fn gradient(&mut self, rect: Rect, gradient: LinearGradient, steps: u32) -> &mut Self {
        self.push(Shape::Gradient {
            rect,
            gradient,
            steps,
        })
    }

    /// Expands one shape into the operations of the display list.
    fn draw_shape(scope: &mut DrawScope<'_>, shape: &Shape) {
        match shape {
            Shape::Rect { rect, radii, paint } => match paint.style {
                codevar_kuitui::paint::PaintStyle::Fill => {
                    scope.fill_rounded_rect(*rect, *radii, paint.color);
                }
                codevar_kuitui::paint::PaintStyle::Stroke(width) => {
                    scope.stroke_rect(*rect, *radii, paint.color, width);
                }
            },
            Shape::Line {
                from,
                to,
                color,
                width,
            } => scope.draw_line(*from, *to, *color, *width),
            Shape::Polygon { points, paint } => match paint.style {
                codevar_kuitui::paint::PaintStyle::Fill => {
                    for span in polygon_spans(points) {
                        let rect =
                            Rect::from_min_size(Offset::new(span.left, span.y), span.right - span.left, 1.0);
                        scope.fill_rect(rect, paint.color);
                    }
                }
                codevar_kuitui::paint::PaintStyle::Stroke(width) => {
                    for (from, to) in polygon_edges(points) {
                        scope.draw_line(from, to, paint.color, width);
                    }
                }
            },
            Shape::Gradient {
                rect,
                gradient,
                steps,
            } => {
                for (band, color) in gradient_bands(*rect, gradient, *steps) {
                    scope.fill_rect(band, color);
                }
            }
        }
    }
}

impl Widget for Canvas {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(self.size)
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, _bounds: Rect) {
        for shape in &self.shapes {
            Self::draw_shape(scope, shape);
        }
    }
}

/// Expands a closed polygon into horizontal one-pixel spans.
///
/// The fill uses the even-odd rule sampled at the center of each pixel
/// row, which reproduces the polygon at the resolution of the display
/// list without a triangulator. The returned spans are in the same
/// coordinate space as `points`.
#[must_use]
pub fn polygon_spans(points: &[Offset]) -> Vec<ScanlineSpan> {
    let mut spans = Vec::new();
    if points.len() < 3 {
        return spans;
    }
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;
    for point in points {
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }
    let first_row = floor_to_i32(min_y);
    let last_row = ceil_to_i32(max_y);
    let mut crossings: Vec<f32> = Vec::new();
    for row in first_row..last_row {
        let sample = row as f32 + 0.5;
        crossings.clear();
        for index in 0..points.len() {
            let a = points[index];
            let b = points[(index + 1) % points.len()];
            let a_above = a.y <= sample;
            let b_above = b.y <= sample;
            if a_above == b_above {
                continue;
            }
            let delta = b.y - a.y;
            if delta == 0.0 {
                continue;
            }
            crossings.push(a.x + (sample - a.y) * (b.x - a.x) / delta);
        }
        if crossings.len() < 2 {
            continue;
        }
        crossings.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for pair in crossings.chunks_exact(2) {
            let left = floor_to_i32(pair[0]) as f32;
            let right = ceil_to_i32(pair[1]) as f32;
            if right > left {
                spans.push(ScanlineSpan {
                    y: row as f32,
                    left,
                    right,
                });
            }
        }
    }
    spans
}

/// The edges of a closed polygon, including the closing edge.
#[must_use]
pub fn polygon_edges(points: &[Offset]) -> Vec<(Offset, Offset)> {
    let mut edges = Vec::new();
    if points.len() < 2 {
        return edges;
    }
    for index in 0..points.len() {
        edges.push((points[index], points[(index + 1) % points.len()]));
    }
    edges
}

/// Expands a gradient inside `rect` into constant-color bands.
///
/// The bands run perpendicular to the dominant axis of `gradient`:
/// vertical stripes for a mostly horizontal axis, horizontal stripes
/// otherwise. Each band is painted with the gradient color at its own
/// center, so the ramp reads correctly; a backend with a gradient shader
/// should consume [`LinearGradient`] directly instead.
#[must_use]
pub fn gradient_bands(rect: Rect, gradient: &LinearGradient, steps: u32) -> Vec<(Rect, Color)> {
    let mut bands = Vec::new();
    if steps == 0 || rect.size.width <= 0.0 || rect.size.height <= 0.0 {
        return bands;
    }
    let dx = gradient.end.x - gradient.start.x;
    let dy = gradient.end.y - gradient.start.y;
    let horizontal = dx * dx >= dy * dy;
    let step = steps as f32;
    for index in 0..steps {
        let factor = (index as f32 + 0.5) / step;
        let color = gradient.color_at(factor);
        let band = if horizontal {
            let width = rect.size.width / step;
            Rect::from_min_size(
                Offset::new(rect.left() + index as f32 * width, rect.top()),
                width,
                rect.size.height,
            )
        } else {
            let height = rect.size.height / step;
            Rect::from_min_size(
                Offset::new(rect.left(), rect.top() + index as f32 * height),
                rect.size.width,
                height,
            )
        };
        bands.push((band, color));
    }
    bands
}

/// Truncates an `f32` towards negative infinity, without `std`.
#[inline]
fn floor_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if truncated as f32 > value {
        truncated.saturating_sub(1)
    } else {
        truncated
    }
}

/// Truncates an `f32` towards positive infinity, without `std`.
#[inline]
fn ceil_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if (truncated as f32) < value {
        truncated.saturating_add(1)
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_kuitui::paint::DrawList;
    use codevar_kuitui::paint::DrawOp;
    use codevar_kuitui::theme::Theme;

    fn square() -> Vec<Offset> {
        alloc::vec![
            Offset::new(0.0, 0.0),
            Offset::new(2.0, 0.0),
            Offset::new(2.0, 2.0),
            Offset::new(0.0, 2.0),
        ]
    }

    #[test]
    fn a_square_fills_two_scanlines() {
        let spans = polygon_spans(&square());
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].y, 0.0);
        assert_eq!(spans[0].left, 0.0);
        assert_eq!(spans[0].right, 2.0);
        assert_eq!(spans[1].y, 1.0);
    }

    #[test]
    fn a_triangle_has_a_closing_edge() {
        let triangle = alloc::vec![
            Offset::new(0.0, 0.0),
            Offset::new(4.0, 0.0),
            Offset::new(0.0, 4.0),
        ];
        let edges = polygon_edges(&triangle);
        assert_eq!(edges.len(), 3);
        assert_eq!(edges[2], (Offset::new(0.0, 4.0), Offset::new(0.0, 0.0)));
    }

    #[test]
    fn a_degenerate_polygon_fills_nothing() {
        assert!(polygon_spans(&[]).is_empty());
        assert!(polygon_spans(&[Offset::ZERO, Offset::new(1.0, 1.0)]).is_empty());
    }

    #[test]
    fn the_gradient_endpoints_are_the_axis_colors() {
        let gradient = LinearGradient::new(Offset::ZERO, Offset::new(10.0, 0.0), Color::BLACK, Color::WHITE);
        assert_eq!(gradient.color_at(0.0), Color::BLACK);
        assert_eq!(gradient.color_at(1.0), Color::WHITE);
        assert_eq!(
            gradient
                .color_at_point(Offset::new(5.0, 100.0))
                .red,
            0.5
        );
    }

    #[test]
    fn a_degenerate_gradient_axis_reports_the_start() {
        let gradient = LinearGradient::new(Offset::ZERO, Offset::ZERO, Color::BLACK, Color::WHITE);
        assert_eq!(gradient.factor_at(Offset::new(3.0, 3.0)), 0.0);
    }

    #[test]
    fn gradient_bands_cover_the_rectangle() {
        let rect = Rect::from_min_size(Offset::ZERO, 8.0, 4.0);
        let gradient = LinearGradient::new(Offset::ZERO, Offset::new(8.0, 0.0), Color::BLACK, Color::WHITE);
        let bands = gradient_bands(rect, &gradient, 4);
        assert_eq!(bands.len(), 4);
        assert_eq!(bands[0].0.left(), 0.0);
        assert_eq!(bands[0].0.size.width, 2.0);
        assert_eq!(bands[3].0.right(), 8.0);
        assert!(bands[3].1.red > bands[0].1.red);
    }

    #[test]
    fn gradient_bands_are_empty_for_zero_steps() {
        let rect = Rect::from_min_size(Offset::ZERO, 8.0, 4.0);
        let gradient = LinearGradient::new(Offset::ZERO, Offset::new(8.0, 0.0), Color::BLACK, Color::WHITE);
        assert!(gradient_bands(rect, &gradient, 0).is_empty());
    }

    /// A widget container without children, used to build a scope.
    struct NoChildren;

    impl codevar_kuitui::widget::Children for NoChildren {
        fn child_count(&self) -> usize {
            0
        }

        fn child_size(&self, _index: usize) -> Size {
            Size::ZERO
        }

        fn child_flex(&self, _index: usize) -> f32 {
            0.0
        }

        fn measure_child(&mut self, _index: usize, _theme: &Theme, constraints: Constraints) -> Size {
            constraints.enforce(Size::ZERO)
        }

        fn place_child(&mut self, _index: usize, _position: Offset) {}

        fn dispatch_child(
            &mut self,
            _index: usize,
            _theme: &Theme,
            _path: &mut Vec<usize>,
            _focus: &mut Option<Vec<usize>>,
            _event: &codevar_kuitui::event::Event,
        ) -> codevar_kuitui::event::EventResult {
            codevar_kuitui::event::EventResult::Ignored
        }
    }

    #[test]
    fn the_canvas_measures_to_its_own_size() {
        let theme = Theme::light();
        let mut children = NoChildren;
        let mut canvas = Canvas::new(Size::new(30.0, 20.0));
        let size = canvas.measure(
            &mut MeasureScope::new(&theme, &mut children),
            Constraints::unbounded(),
        );
        assert_eq!(size, Size::new(30.0, 20.0));
    }

    #[test]
    fn the_canvas_expands_a_filled_polygon_into_one_op_per_scanline() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut canvas = Canvas::new(Size::new(4.0, 4.0));
        canvas.polygon(square(), Paint::fill(Color::RED));
        let mut scope = DrawScope::new(&mut list, &theme, Offset::ZERO);
        canvas.draw(&mut scope, Rect::from_min_size(Offset::ZERO, 4.0, 4.0));
        assert_eq!(list.len(), 2);
        for op in list.ops() {
            assert!(matches!(op, DrawOp::Rect { .. }));
        }
    }

    #[test]
    fn the_canvas_draws_a_line_and_a_gradient() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut canvas = Canvas::new(Size::new(8.0, 4.0));
        canvas.line(Offset::ZERO, Offset::new(8.0, 0.0), Color::WHITE, 1.0);
        canvas.gradient(
            Rect::from_min_size(Offset::ZERO, 8.0, 4.0),
            LinearGradient::new(Offset::ZERO, Offset::new(8.0, 0.0), Color::BLACK, Color::WHITE),
            3,
        );
        let mut scope = DrawScope::new(&mut list, &theme, Offset::ZERO);
        canvas.draw(&mut scope, Rect::from_min_size(Offset::ZERO, 8.0, 4.0));
        assert!(matches!(list.ops().next(), Some(DrawOp::Line { .. })));
        assert_eq!(list.len(), 4);
    }
}
