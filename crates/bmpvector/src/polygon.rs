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

//! Polygon and polyline helpers shared by the tracing and fitting stages.
//!
//! The routines here work on closed rings and open chains in *document
//! space*, the same space the [`crate::geom`] primitives live in. A clockwise
//! ring on screen has a positive signed area, which keeps the tracer's
//! orientation convention consistent from the contour walk through the
//! optimizer.
//!
//! All floating point helpers go through [`libm`] so the crate builds on
//! `no_std` targets. The small wrapper functions are intentionally `#[inline]`
//! because they sit inside the fitting inner loops.

use libm::{fabs, sqrt};

use crate::geom::{GEOMETRY_EPSILON, Point};

/// Distance from `point` to the infinite line through `a` and `b`.
#[inline]
pub fn point_line_distance(point: Point, a: Point, b: Point) -> f64 {
    let direction = b - a;
    let length = direction.length();
    if length <= GEOMETRY_EPSILON {
        point.distance(a)
    } else {
        fabs((point - a).cross(direction)) / length
    }
}

/// Projects `point` onto the infinite line through `a` and `b`.
#[inline]
pub fn project_on_line(point: Point, a: Point, b: Point) -> Point {
    let direction = b - a;
    let denominator = direction.length_sq();
    if denominator <= GEOMETRY_EPSILON {
        return a;
    }
    let t = (point - a).dot(direction) / denominator;
    a + direction * t
}

/// Distance from `point` to the closed segment `[a, b]`.
#[inline]
pub fn point_segment_distance(point: Point, a: Point, b: Point) -> f64 {
    let direction = b - a;
    let denominator = direction.length_sq();
    if denominator <= GEOMETRY_EPSILON {
        return point.distance(a);
    }
    let t = ((point - a).dot(direction) / denominator).clamp(0.0, 1.0);
    point.distance(a + direction * t)
}

/// Distance from `point` to the nearest segment of a polyline.
pub fn point_polyline_distance(point: Point, points: &[Point]) -> f64 {
    if points.is_empty() {
        return f64::INFINITY;
    }
    if points.len() == 1 {
        return point.distance(points[0]);
    }
    let mut best = f64::INFINITY;
    for pair in points.windows(2) {
        best = best.min(point_segment_distance(point, pair[0], pair[1]));
    }
    best
}

/// Twice the signed area of a polygon ring using the shoelace formula.
///
/// The result is positive for a clockwise ring on screen (the outer boundary
/// convention used by [`crate::contour`]).
pub fn polygon_double_area(points: &[Point]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        sum += current.cross(next);
    }
    sum
}

/// Signed area of a polygon ring.
#[inline]
pub fn polygon_signed_area(points: &[Point]) -> f64 {
    polygon_double_area(points) * 0.5
}

/// Unsigned area of a polygon ring.
#[inline]
pub fn polygon_area(points: &[Point]) -> f64 {
    fabs(polygon_signed_area(points))
}

/// Area-weighted centroid of a polygon ring.
///
/// Returns `None` for a degenerate ring whose area is zero.
pub fn polygon_centroid(points: &[Point]) -> Option<Point> {
    if points.len() < 3 {
        return None;
    }
    let mut double_area = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        let cross = current.cross(next);
        double_area += cross;
        cx += (current.x + next.x) * cross;
        cy += (current.y + next.y) * cross;
    }
    if fabs(double_area) <= GEOMETRY_EPSILON {
        return None;
    }
    let scale = 1.0 / (3.0 * double_area);
    Some(Point::new(cx * scale, cy * scale))
}

/// Total length of an open polyline.
pub fn polyline_length(points: &[Point]) -> f64 {
    points
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum()
}

/// Total length of a closed ring.
pub fn ring_length(points: &[Point]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut length = polyline_length(points);
    length += points[points.len() - 1].distance(points[0]);
    length
}

/// Whether a closed ring is wound clockwise on screen.
///
/// Screen coordinates put `y` downwards, so a positive shoelace sum is a
/// clockwise visual traversal. Outer contours produced by the tracer use this
/// orientation and inner contours use the opposite one.
#[inline]
pub fn is_clockwise(points: &[Point]) -> bool {
    polygon_signed_area(points) > 0.0
}

/// Signed turn angle at `vertex` for the path `previous -> vertex -> next`.
///
/// Positive angles turn clockwise on screen, negative angles turn
/// counter-clockwise, and a straight continuation is zero.
pub fn turn_angle(previous: Point, vertex: Point, next: Point) -> f64 {
    let incoming = vertex - previous;
    let outgoing = next - vertex;
    incoming.signed_angle_to(outgoing)
}

/// The unsigned deflection from a straight line at `vertex`, in `[0, pi]`.
#[inline]
pub fn deflection(previous: Point, vertex: Point, next: Point) -> f64 {
    let incoming = vertex - previous;
    let outgoing = next - vertex;
    incoming.angle_between(outgoing)
}

/// Winding number of `point` with respect to a polygon ring.
///
/// The value is `+1` for a clockwise ring on screen, `-1` for a
/// counter-clockwise ring, and `0` for a point outside. Points exactly on an
/// edge return zero; callers that need boundary detection should test the
/// distance to the edges separately.
pub fn winding_number(point: Point, ring: &[Point]) -> i32 {
    if ring.len() < 3 {
        return 0;
    }
    let mut winding = 0;
    for index in 0..ring.len() {
        let a = ring[index];
        let b = ring[(index + 1) % ring.len()];
        let side = (b - a).cross(point - a);
        if a.y <= point.y {
            if b.y > point.y && side > 0.0 {
                winding += 1;
            }
        } else if b.y <= point.y && side < 0.0 {
            winding -= 1;
        }
    }
    winding
}

/// Whether `point` lies inside a ring using the non-zero rule.
#[inline]
pub fn point_in_polygon_nonzero(point: Point, ring: &[Point]) -> bool {
    winding_number(point, ring) != 0
}

/// Whether `point` lies inside a ring using the even-odd rule.
pub fn point_in_polygon_even_odd(point: Point, ring: &[Point]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    for index in 0..ring.len() {
        let a = ring[index];
        let b = ring[(index + 1) % ring.len()];
        let crosses = (a.y > point.y) != (b.y > point.y);
        if crosses {
            let x = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if x > point.x {
                inside = !inside;
            }
        }
    }
    inside
}

/// Intersection of two infinite lines, or `None` when they are parallel.
pub fn line_intersection(a0: Point, a1: Point, b0: Point, b1: Point) -> Option<Point> {
    let da = a1 - a0;
    let db = b1 - b0;
    let denominator = da.cross(db);
    if fabs(denominator) <= GEOMETRY_EPSILON {
        return None;
    }
    let offset = b0 - a0;
    let t = offset.cross(db) / denominator;
    Some(a0 + da * t)
}

/// Safe square root returning zero for non-positive input.
#[inline]
pub fn non_negative_sqrt(value: f64) -> f64 {
    if value <= 0.0 { 0.0 } else { sqrt(value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CubicBezier, PixelRect};

    fn close(a: f64, b: f64) -> bool {
        fabs(a - b) <= 1.0e-9
    }

    #[test]
    fn point_arithmetic_and_operators_agree() {
        let a = Point::new(3.0, 4.0);
        let b = Point::new(-1.0, 2.0);
        assert_eq!(a + b, Point::new(2.0, 6.0));
        assert_eq!(a - b, Point::new(4.0, 2.0));
        assert_eq!(a * 2.0, a.scale(2.0));
        assert_eq!(2.0 * a, a.scale(2.0));
        assert_eq!(a / 2.0, a.scale(0.5));
        assert_eq!(-a, Point::new(-3.0, -4.0));
        assert!(close(a.length(), 5.0));
        assert!(close(a.dot(b), 5.0));
        assert!(close(a.cross(b), 10.0));
    }

    #[test]
    fn normalization_rejects_degenerate_vectors() {
        assert!(Point::ZERO.normalized().is_none());
        assert!(Point::new(f64::NAN, 0.0).normalized().is_none());
        let unit = Point::new(0.0, 3.0)
            .normalized()
            .expect("non-zero vector");
        assert!(close(unit.x, 0.0));
        assert!(close(unit.y, 1.0));
    }

    #[test]
    fn polygon_area_sign_matches_screen_orientation() {
        let clockwise = [
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ];
        assert!(close(polygon_signed_area(&clockwise), 4.0));
        assert!(is_clockwise(&clockwise));

        let mut reversed = clockwise;
        reversed.reverse();
        assert!(close(polygon_signed_area(&reversed), -4.0));
        assert!(!is_clockwise(&reversed));
    }

    #[test]
    fn centroid_of_square_is_its_center() {
        let square = [
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ];
        let centroid = polygon_centroid(&square).expect("non-degenerate ring");
        assert!(close(centroid.x, 2.0));
        assert!(close(centroid.y, 2.0));
    }

    #[test]
    fn winding_number_handles_orientation() {
        let clockwise = [
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ];
        assert_eq!(winding_number(Point::new(2.0, 2.0), &clockwise), 1);
        assert_eq!(winding_number(Point::new(8.0, 2.0), &clockwise), 0);
        let mut reversed = clockwise;
        reversed.reverse();
        assert_eq!(winding_number(Point::new(2.0, 2.0), &reversed), -1);
        assert!(point_in_polygon_nonzero(Point::new(2.0, 2.0), &clockwise));
        assert!(point_in_polygon_even_odd(Point::new(2.0, 2.0), &clockwise));
    }

    #[test]
    fn segment_distance_clamps_to_endpoints() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(10.0, 0.0);
        assert!(close(point_segment_distance(Point::new(5.0, 3.0), a, b), 3.0));
        assert!(close(point_segment_distance(Point::new(-4.0, 0.0), a, b), 4.0));
        assert!(close(point_segment_distance(Point::new(14.0, 0.0), a, b), 4.0));
    }

    #[test]
    fn cubic_endpoints_and_split_match() {
        let cubic = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(3.0, 2.0),
            Point::new(4.0, 0.0),
        );
        assert!(close(cubic.eval(0.0).x, 0.0));
        assert!(close(cubic.eval(1.0).x, 4.0));
        let (left, right) = cubic.split(0.35);
        let sample = cubic.eval(0.35);
        assert!(close(left.end.x, sample.x));
        assert!(close(right.start.y, sample.y));
        assert!(cubic.flatness() >= 0.0);
    }

    #[test]
    fn pixel_rect_union_and_translation_work() {
        let a = PixelRect::new(2, 3, 4, 5);
        let b = PixelRect::new(0, 4, 3, 9);
        let union = a.union(b);
        assert_eq!(union, PixelRect::new(0, 3, 4, 9));
        assert_eq!(union.area(), 24);
        assert!(union.contains(3, 8));
        assert!(!union.contains(4, 8));
        let translated = a.translated(0.5, -0.5);
        assert!(close(translated.left, 2.5));
        assert!(close(translated.bottom, 4.5));
    }
}
