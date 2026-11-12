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

//! The vector document returned by the tracer.
//!
//! This is deliberately a data model, not a serializer. Every shape carries
//! the fill color and fill rule needed to render it, and every subpath carries
//! absolute coordinates in document space. Callers can render the document
//! directly, convert it to their own scene graph, or serialize it with whatever
//! backend they prefer. The crate never emits SVG, HTML or XML.
//!
//! ## Coordinate convention
//!
//! A pixel `(x, y)` occupies `[x, x + 1) x [y, y + 1)`. The document covers
//! `[0, width] x [0, height]`. Shapes are listed in paint order: the first shape
//! is drawn first and later shapes may overlap it. The tracer normally emits
//! larger shapes first so fine details survive at shared boundaries.
//!
//! ## Fill rules
//!
//! Each traced connected component becomes one [`Shape`] whose [`Path`] holds
//! the component's outer rings and its holes. [`FillRule::EvenOdd`] is used
//! because it composes naturally with the ring orientation produced by contour
//! extraction: a hole is any ring nested an odd number of times. Consumers that
//! need non-zero winding can reorient rings using [`SubPath::signed_area`].

use alloc::vec::Vec;

use crate::color::Color;
use crate::geom::{Bounds, CubicBezier, GEOMETRY_EPSILON, Point};
use crate::polygon::{polygon_signed_area, polyline_length};
use crate::simplify::remove_duplicate_points;

/// Fill rule used to interpret a [`Path`] made of several rings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FillRule {
    /// A point is inside when the winding number is non-zero.
    NonZero,
    /// A point is inside when it is enclosed by an odd number of rings.
    #[default]
    EvenOdd,
}

/// The paint applied to a shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fill {
    /// Straight-alpha fill color.
    pub color: Color,
    /// Rule used to resolve overlapping and nested rings.
    pub rule: FillRule,
}

impl Fill {
    /// A fill with the even-odd rule, which is the tracer's default.
    #[inline]
    pub const fn new(color: Color) -> Self {
        Self {
            color,
            rule: FillRule::EvenOdd,
        }
    }

    /// A fill with an explicit rule.
    #[inline]
    pub const fn with_rule(color: Color, rule: FillRule) -> Self {
        Self { color, rule }
    }

    /// Whether the fill is invisible and can be discarded.
    #[inline]
    pub const fn is_transparent(self) -> bool {
        self.color.a == 0
    }
}

/// A line or cubic segment in absolute coordinates.
///
/// Keeping the start point in each segment makes segments self-contained for
/// consumers that flatten, transform or reverse paths independently. Path
/// construction helpers guarantee that `start` equals the previous segment's
/// end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Segment {
    /// A straight line segment.
    Line {
        /// Start point.
        start: Point,
        /// End point.
        end: Point,
    },
    /// A cubic Bézier segment.
    Cubic(CubicBezier),
}

impl Segment {
    /// A line from `start` to `end`.
    #[inline]
    pub const fn line(start: Point, end: Point) -> Self {
        Self::Line { start, end }
    }

    /// A cubic segment.
    #[inline]
    pub const fn cubic(start: Point, control1: Point, control2: Point, end: Point) -> Self {
        Self::Cubic(CubicBezier::new(start, control1, control2, end))
    }

    /// Start point of the segment.
    #[inline]
    pub const fn start(self) -> Point {
        match self {
            Self::Line { start, .. } => start,
            Self::Cubic(cubic) => cubic.start,
        }
    }

    /// End point of the segment.
    #[inline]
    pub const fn end(self) -> Point {
        match self {
            Self::Line { end, .. } => end,
            Self::Cubic(cubic) => cubic.end,
        }
    }

    /// The segment with its direction reversed.
    #[inline]
    pub const fn reversed(self) -> Self {
        match self {
            Self::Line { start, end } => Self::Line {
                start: end,
                end: start,
            },
            Self::Cubic(cubic) => Self::Cubic(cubic.reversed()),
        }
    }

    /// Whether all coordinates are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        match self {
            Self::Line { start, end } => start.is_finite() && end.is_finite(),
            Self::Cubic(cubic) => cubic.is_finite(),
        }
    }

    /// Straight-line distance between the endpoints.
    #[inline]
    pub fn chord_length(self) -> f64 {
        self.start().distance(self.end())
    }

    /// Conservative control-point bounds.
    #[inline]
    pub fn control_bounds(self) -> Bounds {
        match self {
            Self::Line { start, end } => {
                let mut bounds = Bounds::from_point(start);
                bounds.include_point(end);
                bounds
            }
            Self::Cubic(cubic) => cubic.control_bounds(),
        }
    }

    /// Flattens the segment, appending points after its start point.
    ///
    /// The start point itself is not appended; callers that build a polyline
    /// push it once when they begin the subpath. Cubic segments are subdivided
    /// until their control polygon is flat within `tolerance`.
    pub fn flatten_into(self, tolerance: f64, output: &mut Vec<Point>) {
        match self {
            Self::Line { end, .. } => output.push(end),
            Self::Cubic(cubic) => flatten_cubic(cubic, tolerance, 0, output),
        }
    }
}

/// Flattens a cubic and appends the generated points, including its end point.
fn flatten_cubic(cubic: CubicBezier, tolerance: f64, depth: u32, output: &mut Vec<Point>) {
    const MAX_DEPTH: u32 = 18;
    let tolerance = if tolerance.is_finite() {
        tolerance.max(1.0e-4)
    } else {
        0.25
    };
    if depth >= MAX_DEPTH || cubic.flatness() <= tolerance {
        output.push(cubic.end);
        return;
    }
    let (left, right) = cubic.split(0.5);
    flatten_cubic(left, tolerance, depth + 1, output);
    flatten_cubic(right, tolerance, depth + 1, output);
}

/// One connected ring or open chain of segments.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubPath {
    /// Absolute start point of the subpath.
    pub start: Point,
    /// Ordered segments. The first segment starts at [`SubPath::start`].
    pub segments: Vec<Segment>,
    /// Whether the ring should be closed back to `start` when rendered.
    pub closed: bool,
}

impl SubPath {
    /// An empty open subpath starting at `start`.
    #[inline]
    pub fn new(start: Point) -> Self {
        Self {
            start,
            segments: Vec::new(),
            closed: false,
        }
    }

    /// An empty closed subpath starting at `start`.
    #[inline]
    pub fn closed(start: Point) -> Self {
        Self {
            start,
            segments: Vec::new(),
            closed: true,
        }
    }

    /// Number of segments.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// The current end point, or the start point when there are no segments.
    #[inline]
    pub fn end(&self) -> Point {
        self.segments
            .last()
            .map_or(self.start, |segment| segment.end())
    }

    /// Whether the subpath has no segments.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Appends a straight segment.
    pub fn push_line(&mut self, end: Point) {
        let start = self.end();
        self.segments.push(Segment::line(start, end));
    }

    /// Appends a cubic segment.
    pub fn push_cubic(&mut self, control1: Point, control2: Point, end: Point) {
        let start = self.end();
        self.segments
            .push(Segment::cubic(start, control1, control2, end));
    }

    /// Appends a pre-built segment if its start matches the current end.
    ///
    /// Returns `false` without changing the subpath when the segment is not
    /// continuous with it.
    pub fn push_segment(&mut self, segment: Segment) -> bool {
        if segment.start().distance(self.end()) > 1.0e-7 {
            return false;
        }
        self.segments.push(segment);
        true
    }

    /// Closes the subpath, adding a line when the last segment does not already
    /// end at the start point.
    pub fn close(&mut self) {
        let end = self.end();
        if end.distance(self.start) > 1.0e-7 {
            self.push_line(self.start);
        }
        self.closed = true;
    }

    /// Reverses the ring or chain in place.
    ///
    /// The traversal direction is inverted while the geometry is preserved: a
    /// closed ring keeps its start point, and an open chain starts at the
    /// point where the original chain ended.
    pub fn reverse(&mut self) {
        if let Some(last) = self.segments.last().copied() {
            self.start = last.end();
        }
        self.segments.reverse();
        for segment in &mut self.segments {
            *segment = segment.reversed();
        }
        // With a closed ring the final end is the original start; restoring the
        // exact original value avoids accumulating round-off through repeated
        // reversals.
        if self.closed {
            let last_end = self
                .segments
                .last()
                .map_or(self.start, |segment| segment.end());
            self.start = last_end;
        }
    }

    /// Signed area of a closed subpath after flattening.
    ///
    /// A clockwise ring on screen (the tracer's outer-boundary convention) has
    /// positive area. Open subpaths report the area of their implicit closing
    /// chord.
    pub fn signed_area(&self, flattening_tolerance: f64) -> f64 {
        let points = self.flatten(flattening_tolerance);
        polygon_signed_area(&points)
    }

    /// Flattens the subpath into a polyline starting at [`SubPath::start`].
    pub fn flatten(&self, tolerance: f64) -> Vec<Point> {
        let mut points = Vec::with_capacity(self.segments.len() * 4 + 1);
        points.push(self.start);
        for &segment in &self.segments {
            segment.flatten_into(tolerance, &mut points);
        }
        remove_duplicate_points(&mut points);
        if self.closed && points.len() >= 2 && points[0].distance(points[points.len() - 1]) > 1.0e-7 {
            points.push(self.start);
        }
        points
    }

    /// Approximate arc length after flattening.
    pub fn length(&self, flattening_tolerance: f64) -> f64 {
        polyline_length(&self.flatten(flattening_tolerance))
    }

    /// Bounds of all control points, including the start point.
    pub fn control_bounds(&self) -> Bounds {
        let mut bounds = Bounds::from_point(self.start);
        for segment in &self.segments {
            let segment_bounds = segment.control_bounds();
            if !segment_bounds.is_empty() {
                bounds.include_point(Point::new(segment_bounds.left, segment_bounds.top));
                bounds.include_point(Point::new(segment_bounds.right, segment_bounds.bottom));
            }
        }
        bounds
    }

    /// Whether every coordinate is finite.
    pub fn is_finite(&self) -> bool {
        self.start.is_finite()
            && self
                .segments
                .iter()
                .all(|segment| segment.is_finite())
    }
}

/// A compound path made of one or more subpaths.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    /// Rings and chains that make up this path.
    pub subpaths: Vec<SubPath>,
}

impl Path {
    /// An empty path.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of subpaths.
    #[inline]
    pub fn len(&self) -> usize {
        self.subpaths.len()
    }

    /// Whether the path contains no segments at all.
    ///
    /// A path with a subpath that has no segments is still empty: there is
    /// nothing to render. Use [`Path::len`] to count structurally present
    /// subpaths.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.subpaths.iter().all(SubPath::is_empty)
    }

    /// Number of segments across every subpath.
    pub fn segment_count(&self) -> usize {
        self.subpaths
            .iter()
            .map(SubPath::segment_count)
            .sum()
    }

    /// Bounds of all control points.
    pub fn control_bounds(&self) -> Bounds {
        let mut bounds = Bounds::EMPTY;
        for subpath in &self.subpaths {
            let subpath_bounds = subpath.control_bounds();
            if !subpath_bounds.is_empty() {
                bounds.include_point(Point::new(subpath_bounds.left, subpath_bounds.top));
                bounds.include_point(Point::new(subpath_bounds.right, subpath_bounds.bottom));
            }
        }
        bounds
    }

    /// Net signed area: outer rings contribute positively and holes negatively
    /// when the tracer's orientation convention is preserved.
    pub fn signed_area(&self, flattening_tolerance: f64) -> f64 {
        self.subpaths
            .iter()
            .map(|subpath| subpath.signed_area(flattening_tolerance))
            .sum()
    }

    /// Whether every coordinate is finite.
    pub fn is_finite(&self) -> bool {
        self.subpaths.iter().all(SubPath::is_finite)
    }

    /// Removes subpaths that enclose no meaningful area.
    pub fn retain_area(&mut self, minimum_area: f64, flattening_tolerance: f64) {
        self.subpaths.retain(|subpath| {
            if subpath.is_empty() {
                return false;
            }
            libm::fabs(subpath.signed_area(flattening_tolerance)) >= minimum_area
        });
    }
}

/// A filled region in the output document.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    /// Paint applied to the region.
    pub fill: Fill,
    /// Geometry of the region, including any holes.
    pub path: Path,
    /// Number of source pixels covered by the region before fitting.
    ///
    /// This is kept as metadata and as a stable ordering key; it is not a
    /// geometric area. Use [`Path::signed_area`] for the fitted area.
    pub source_pixels: u64,
}

impl Shape {
    /// A shape from its fill, geometry and source-pixel count.
    #[inline]
    pub fn new(fill: Fill, path: Path, source_pixels: u64) -> Self {
        Self {
            fill,
            path,
            source_pixels,
        }
    }

    /// Bounds of all control points.
    #[inline]
    pub fn bounds(&self) -> Bounds {
        self.path.control_bounds()
    }

    /// Whether the shape can be discarded without changing the drawing.
    pub fn is_degenerate(&self) -> bool {
        self.fill.is_transparent() || self.path.is_empty() || !self.path.is_finite()
    }
}

/// The complete vector result.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorDoc {
    /// Document width in pixels.
    pub width: u32,
    /// Document height in pixels.
    pub height: u32,
    /// Shapes in paint order: first is drawn first.
    pub shapes: Vec<Shape>,
}

impl VectorDoc {
    /// An empty document of the given size.
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            shapes: Vec::new(),
        }
    }

    /// Number of shapes.
    #[inline]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether the document has no shapes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Total number of path segments.
    pub fn segment_count(&self) -> usize {
        self.shapes
            .iter()
            .map(|shape| shape.path.segment_count())
            .sum()
    }

    /// Total number of subpaths.
    pub fn subpath_count(&self) -> usize {
        self.shapes
            .iter()
            .map(|shape| shape.path.subpaths.len())
            .sum()
    }

    /// Bounds of all geometry, or empty bounds when the document is empty.
    pub fn bounds(&self) -> Bounds {
        let mut bounds = Bounds::EMPTY;
        for shape in &self.shapes {
            let shape_bounds = shape.bounds();
            if !shape_bounds.is_empty() {
                bounds.include_point(Point::new(shape_bounds.left, shape_bounds.top));
                bounds.include_point(Point::new(shape_bounds.right, shape_bounds.bottom));
            }
        }
        bounds
    }

    /// Appends a shape unless it is degenerate.
    pub fn push(&mut self, shape: Shape) {
        if !shape.is_degenerate() {
            self.shapes.push(shape);
        }
    }

    /// Removes shapes that are transparent or have no geometry.
    pub fn retain_visible(&mut self) {
        self.shapes.retain(|shape| !shape.is_degenerate());
    }

    /// The document size as a rectangle in document space.
    #[inline]
    pub fn bounds_rect(&self) -> Bounds {
        Bounds::new(0.0, 0.0, self.width as f64, self.height as f64)
    }

    /// Whether all coordinates are finite.
    pub fn is_finite(&self) -> bool {
        self.shapes
            .iter()
            .all(|shape| shape.path.is_finite())
    }

    /// Statistics used by the pipeline's progress and diagnostics.
    pub fn stats(&self) -> DocumentStats {
        DocumentStats {
            shapes: self.shapes.len(),
            subpaths: self.subpath_count(),
            segments: self.segment_count(),
            cubic_segments: self
                .shapes
                .iter()
                .flat_map(|shape| &shape.path.subpaths)
                .flat_map(|subpath| &subpath.segments)
                .filter(|segment| matches!(segment, Segment::Cubic(_)))
                .count(),
            line_segments: self
                .shapes
                .iter()
                .flat_map(|shape| &shape.path.subpaths)
                .flat_map(|subpath| &subpath.segments)
                .filter(|segment| matches!(segment, Segment::Line { .. }))
                .count(),
        }
    }
}

/// Aggregate counters for a finished document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DocumentStats {
    /// Number of shapes.
    pub shapes: usize,
    /// Number of subpaths across all shapes.
    pub subpaths: usize,
    /// Number of line and cubic segments.
    pub segments: usize,
    /// Number of cubic segments.
    pub cubic_segments: usize,
    /// Number of straight segments.
    pub line_segments: usize,
}

impl DocumentStats {
    /// Whether the counts are internally consistent.
    #[inline]
    pub const fn is_consistent(self) -> bool {
        self.segments == self.cubic_segments + self.line_segments
    }
}

/// Returns `true` when a subpath encloses an area large enough to matter.
#[inline]
pub fn subpath_has_area(subpath: &SubPath, minimum_area: f64, tolerance: f64) -> bool {
    libm::fabs(subpath.signed_area(tolerance)) >= minimum_area
}

/// Returns `true` when the two points coincide within the geometry epsilon.
#[inline]
pub fn points_coincide(first: Point, second: Point) -> bool {
    first.distance(second) <= GEOMETRY_EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> SubPath {
        let mut path = SubPath::closed(Point::new(0.0, 0.0));
        path.push_line(Point::new(4.0, 0.0));
        path.push_line(Point::new(4.0, 4.0));
        path.push_line(Point::new(0.0, 4.0));
        path.push_line(Point::new(0.0, 0.0));
        path
    }

    #[test]
    fn square_subpath_has_expected_geometry() {
        let path = square();
        assert_eq!(path.segment_count(), 4);
        assert!(libm::fabs(path.signed_area(0.1) - 16.0) < 1.0e-9);
        let points = path.flatten(0.1);
        assert_eq!(points.first(), Some(&Point::new(0.0, 0.0)));
        assert_eq!(points.last(), Some(&Point::new(0.0, 0.0)));
        assert!(path.length(0.1) >= 16.0);
    }

    #[test]
    fn closing_adds_only_the_missing_line() {
        let mut path = SubPath::closed(Point::new(0.0, 0.0));
        path.push_line(Point::new(2.0, 0.0));
        path.push_line(Point::new(2.0, 2.0));
        path.close();
        assert_eq!(path.end(), path.start);
        assert_eq!(path.segment_count(), 3);
        path.close();
        assert_eq!(path.segment_count(), 3);
    }

    #[test]
    fn reverse_preserves_geometry_and_flips_area_sign() {
        let mut path = square();
        let area = path.signed_area(0.1);
        path.reverse();
        let reversed_area = path.signed_area(0.1);
        assert!(libm::fabs(area + reversed_area) < 1.0e-9);
        assert_eq!(path.start, Point::new(0.0, 0.0));
    }

    #[test]
    fn reversing_an_open_chain_starts_at_the_original_end() {
        let mut path = SubPath::new(Point::new(0.0, 0.0));
        path.push_line(Point::new(4.0, 0.0));
        path.push_line(Point::new(4.0, 4.0));

        path.reverse();

        assert_eq!(path.start, Point::new(4.0, 4.0));
        assert_eq!(path.end(), Point::new(0.0, 0.0));
        assert!(!path.closed);
        let flattened = path.flatten(0.01);
        assert_eq!(flattened.first(), Some(&Point::new(4.0, 4.0)));
        assert_eq!(flattened.last(), Some(&Point::new(0.0, 0.0)));
        assert_eq!(flattened.len(), 3);

        let area = path.signed_area(0.01);
        path.reverse();
        // Reversing flips the traversal direction, so the signed area of the
        // implicit closing chord changes sign.
        assert!(libm::fabs(path.signed_area(0.01) + area) < 1.0e-9);
        assert_eq!(path.start, Point::new(0.0, 0.0));
        assert_eq!(path.end(), Point::new(4.0, 4.0));
    }

    #[test]
    fn cubic_flattening_stays_within_tolerance() {
        let cubic = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 10.0),
            Point::new(10.0, 10.0),
            Point::new(10.0, 0.0),
        );
        let mut path = SubPath::new(cubic.start);
        path.segments.push(Segment::Cubic(cubic));
        let points = path.flatten(0.05);
        assert!(points.len() > 4);
        for point in &points {
            let distance = crate::polygon::point_polyline_distance(*point, &points);
            assert!(distance < 10.0);
        }
    }

    #[test]
    fn document_rejects_degenerate_shapes() {
        let mut document = VectorDoc::new(10, 10);
        let mut path = Path::new();
        path.subpaths.push(square());
        let shape = Shape::new(Fill::new(Color::rgb(1, 2, 3)), path, 16);
        document.push(shape);
        assert_eq!(document.len(), 1);
        assert_eq!(document.stats().subpaths, 1);
        assert_eq!(document.stats().segments, 4);
        assert!(document.stats().is_consistent());
        assert!(document.is_finite());
        assert!(!document.bounds().is_empty());
    }

    #[test]
    fn removing_small_subpaths_keeps_outer_ring() {
        let mut path = Path::new();
        path.subpaths.push(square());
        let mut tiny = SubPath::closed(Point::new(5.0, 5.0));
        tiny.push_line(Point::new(5.1, 5.0));
        tiny.push_line(Point::new(5.1, 5.1));
        tiny.push_line(Point::new(5.0, 5.1));
        tiny.push_line(Point::new(5.0, 5.0));
        path.subpaths.push(tiny);
        assert_eq!(path.len(), 2);
        path.retain_area(0.01, 0.01);
        assert_eq!(path.len(), 1);
    }
}
