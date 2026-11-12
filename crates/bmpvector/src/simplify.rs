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

//! Polygon simplification and corner classification.
//!
//! Contour extraction produces a vertex for every pixel edge: a 100-pixel
//! square arrives as hundreds of collinear points, and a shallow diagonal
//! arrives as a staircase. Feeding that directly to a least-squares curve fit
//! would waste work and produce noisy tangents. This module reduces the ring in
//! two steps:
//!
//! 1. duplicate and collinear vertices are removed;
//! 2. a cyclic Ramer-Douglas-Peucker pass keeps only vertices that are more
//!    than `tolerance` away from the chord they would otherwise lie on.
//!
//! The resulting polygon is still a faithful raster outline, but it has a
//! manageable vertex count and its edges point in meaningful directions. A
//! second pass classifies vertices whose deflection is sharper than the corner
//! threshold; the curve fitter splits at those vertices so rounded arcs stay
//! smooth while real corners remain sharp.

use alloc::vec::Vec;

use libm::fabs;

use crate::geom::{GEOMETRY_EPSILON, Point};
use crate::polygon::{deflection, point_line_distance, polygon_signed_area};

/// A vertex range on a closed ring.
///
/// `end` may be greater than the number of source vertices; callers index the
/// ring modulo its length. A run with `start == end` represents the complete
/// ring as one chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingRun {
    /// Index of the first vertex.
    pub start: usize,
    /// Index of the last vertex, possibly one full turn past `start`.
    pub end: usize,
}

impl RingRun {
    /// Number of edges covered by the run.
    #[inline]
    pub const fn edge_count(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the run covers the complete ring.
    #[inline]
    pub fn is_full_ring(self, vertex_count: usize) -> bool {
        vertex_count > 0 && self.start % vertex_count == self.end % vertex_count
    }
}

/// Removes consecutive duplicate points in place.
pub fn remove_duplicate_points(points: &mut Vec<Point>) {
    points.dedup_by(|current, previous| current.distance(*previous) <= GEOMETRY_EPSILON);
    while points.len() >= 2 && points[0].distance(points[points.len() - 1]) <= GEOMETRY_EPSILON {
        points.pop();
    }
}

/// Removes vertices that lie within `tolerance` of the line through their
/// neighbours.
///
/// The operation is repeated until it reaches a fixed point because removing a
/// vertex can make the next one redundant. A ring with fewer than three
/// vertices is left untouched.
pub fn remove_collinear(points: &mut Vec<Point>, tolerance: f64) {
    if points.len() < 3 {
        return;
    }
    let tolerance = tolerance.max(0.0);
    let mut removed = true;
    while removed && points.len() >= 3 {
        removed = false;
        let mut index = 0;
        while index < points.len() && points.len() >= 3 {
            let previous = points[(index + points.len() - 1) % points.len()];
            let current = points[index];
            let next = points[(index + 1) % points.len()];
            let incoming = current - previous;
            let outgoing = next - current;
            // A reversal is a spike, not a collinear continuation: removing it
            // would erase thin features whose opposite edges lie on one line.
            let continues_forward = incoming.dot(outgoing) > 0.0;
            if continues_forward && point_line_distance(current, previous, next) <= tolerance {
                points.remove(index);
                removed = true;
            } else {
                index += 1;
            }
        }
    }
}

/// An open Ramer-Douglas-Peucker simplification.
///
/// The first and last points are always retained. The implementation uses an
/// explicit stack instead of recursion so a pathological contour cannot exhaust
/// the call stack.
pub fn simplify_open(points: &[Point], tolerance: f64) -> Vec<Point> {
    if points.len() <= 2 || tolerance <= 0.0 || !tolerance.is_finite() {
        return points.to_vec();
    }

    let mut keep = alloc::vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(32);
    stack.push((0, points.len() - 1));

    while let Some((first, last)) = stack.pop() {
        if last <= first + 1 {
            continue;
        }
        let mut farthest_index = first;
        let mut farthest_distance = tolerance;
        for index in first + 1..last {
            let distance = point_line_distance(points[index], points[first], points[last]);
            if distance > farthest_distance {
                farthest_distance = distance;
                farthest_index = index;
            }
        }
        if farthest_index != first {
            keep[farthest_index] = true;
            stack.push((first, farthest_index));
            stack.push((farthest_index, last));
        }
    }

    let mut output = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if keep[index] {
            output.push(*point);
        }
    }
    output
}

/// A cyclic Ramer-Douglas-Peucker simplification.
///
/// The ring is cut at the vertex farthest from the first vertex, each half is
/// simplified independently, and the halves are joined again. The closing
/// vertex is not duplicated in the result.
pub fn simplify_ring(points: &[Point], tolerance: f64) -> Vec<Point> {
    let mut ring = points.to_vec();
    remove_duplicate_points(&mut ring);
    if ring.len() <= 3 || tolerance <= 0.0 || !tolerance.is_finite() {
        remove_collinear(&mut ring, 0.0);
        return ring;
    }

    let mut farthest_index = 0;
    let mut farthest_distance = 0.0;
    for (index, point) in ring.iter().enumerate().skip(1) {
        let distance = point.distance_sq(ring[0]);
        if distance > farthest_distance {
            farthest_distance = distance;
            farthest_index = index;
        }
    }
    if farthest_index == 0 {
        return ring;
    }

    let first_half = simplify_open(&ring[..=farthest_index], tolerance);
    let mut second_half = Vec::with_capacity(ring.len() - farthest_index + 1);
    second_half.extend_from_slice(&ring[farthest_index..]);
    second_half.push(ring[0]);
    let second_simplified = simplify_open(&second_half, tolerance);

    let mut output = Vec::with_capacity(first_half.len() + second_simplified.len());
    output.extend(first_half);
    if second_simplified.len() > 1 {
        output.extend_from_slice(&second_simplified[1..]);
    }
    remove_duplicate_points(&mut output);
    remove_collinear(&mut output, 0.0);

    // A simplification can collapse a very thin ring to a line. Keeping the
    // original outline is safer than handing degenerate geometry to the fitter.
    if output.len() < 3 {
        return ring;
    }
    output
}

/// Classifies which vertices of a closed ring are corners.
///
/// A vertex is a corner when the direction change between its incoming and
/// outgoing edges exceeds `threshold_degrees` and both adjacent edges are at
/// least `minimum_arm` long. The arm-length guard keeps a single rasterized
/// staircase notch from being promoted to a corner after RDP.
pub fn classify_corners(points: &[Point], threshold_degrees: f64, minimum_arm: f64) -> Vec<bool> {
    let count = points.len();
    let mut corners = alloc::vec![false; count];
    if count < 3 {
        return corners;
    }
    let threshold = threshold_degrees.clamp(0.0, 180.0) * core::f64::consts::PI / 180.0;
    let minimum_arm = minimum_arm.max(0.0);

    for index in 0..count {
        let previous = points[(index + count - 1) % count];
        let current = points[index];
        let next = points[(index + 1) % count];
        let incoming_length = current.distance(previous);
        let outgoing_length = next.distance(current);
        if incoming_length < minimum_arm || outgoing_length < minimum_arm {
            continue;
        }
        let angle = deflection(previous, current, next);
        if angle >= threshold {
            corners[index] = true;
        }
    }

    // A ring where every vertex was classified as a corner is a tiny polygon,
    // not a smooth curve. Keeping all corners would emit a polygon of lines
    // anyway, which is the right result, so no extra filtering is needed.
    corners
}

/// Splits a ring into runs that begin and end at corner vertices.
///
/// When the ring has no corners, a single run covers the complete ring. The
/// returned ranges are ordered along the ring and collectively cover every
/// edge exactly once.
pub fn split_runs(corners: &[bool]) -> Vec<RingRun> {
    let count = corners.len();
    if count == 0 {
        return Vec::new();
    }
    let corner_indices: Vec<usize> = corners
        .iter()
        .enumerate()
        .filter_map(|(index, &is_corner)| if is_corner { Some(index) } else { None })
        .collect();

    if corner_indices.is_empty() {
        return alloc::vec![RingRun { start: 0, end: count }];
    }
    if corner_indices.len() == 1 {
        let index = corner_indices[0];
        return alloc::vec![RingRun {
            start: index,
            end: index + count,
        }];
    }

    let mut runs = Vec::with_capacity(corner_indices.len());
    for window in 0..corner_indices.len() {
        let start = corner_indices[window];
        let next = corner_indices[(window + 1) % corner_indices.len()];
        let end = if next > start { next } else { next + count };
        runs.push(RingRun { start, end });
    }
    runs
}

/// Collects the points belonging to a run, following the ring cyclically.
pub fn collect_run(points: &[Point], run: RingRun) -> Vec<Point> {
    if points.is_empty() || run.end < run.start {
        return Vec::new();
    }
    let count = points.len();
    let mut output = Vec::with_capacity(run.edge_count() + 1);
    for index in run.start..=run.end {
        output.push(points[index % count]);
    }
    remove_duplicate_points(&mut output);
    output
}

/// Returns the runs of a ring after simplification and corner classification.
///
/// This is the configuration stage used by the curve fitter: it returns the
/// simplified polygon together with the run boundaries, so callers do not have
/// to repeat corner detection.
pub fn prepare_ring(
    points: &[Point],
    line_tolerance: f64,
    corner_threshold_degrees: f64,
    minimum_arm: f64,
) -> (Vec<Point>, Vec<RingRun>) {
    let polygon = simplify_ring(points, line_tolerance);
    let corners = classify_corners(&polygon, corner_threshold_degrees, minimum_arm);
    let runs = split_runs(&corners);
    (polygon, runs)
}

/// Measures how far a simplified polygon departs from the source ring.
///
/// Used by diagnostics and tests: a large deviation means the line tolerance
/// was set too aggressively for the artwork.
pub fn maximum_polygon_deviation(source: &[Point], polygon: &[Point]) -> f64 {
    if source.is_empty() || polygon.is_empty() {
        return f64::INFINITY;
    }
    let mut maximum: f64 = 0.0;
    for point in source {
        let distance = crate::polygon::point_polyline_distance(*point, polygon);
        maximum = maximum.max(distance);
    }
    maximum
}

/// Signed area of a simplified ring, for diagnostics.
#[inline]
pub fn ring_area(points: &[Point]) -> f64 {
    polygon_signed_area(points)
}

/// Returns `true` when the ring has no meaningful extent.
#[inline]
pub fn is_degenerate_ring(points: &[Point], minimum_area: f64) -> bool {
    points.len() < 3 || fabs(polygon_signed_area(points)) < minimum_area
}

/// Utility used by tests and by callers that want to report the angular size of
/// a particular vertex.
#[inline]
pub fn vertex_deflection_degrees(points: &[Point], index: usize) -> f64 {
    let count = points.len();
    if count < 3 {
        return 0.0;
    }
    let previous = points[(index + count - 1) % count];
    let current = points[index % count];
    let next = points[(index + 1) % count];
    deflection(previous, current, next) * 180.0 / core::f64::consts::PI
}

/// Returns the smaller absolute angle between two directions in radians.
#[inline]
pub fn folded_angle(first: Point, second: Point) -> f64 {
    fabs(first.signed_angle_to(second))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lattice_square(size: usize) -> Vec<Point> {
        let mut points = Vec::new();
        for x in 0..=size {
            points.push(Point::new(x as f64, 0.0));
        }
        for y in 1..=size {
            points.push(Point::new(size as f64, y as f64));
        }
        for x in (0..size).rev() {
            points.push(Point::new(x as f64, size as f64));
        }
        for y in (1..size).rev() {
            points.push(Point::new(0.0, y as f64));
        }
        points
    }

    #[test]
    fn duplicate_removal_handles_closing_vertex() {
        let mut points = alloc::vec![
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 0.0),
        ];
        remove_duplicate_points(&mut points);
        assert_eq!(points, alloc::vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)]);
    }

    #[test]
    fn collinear_points_collapse_to_square_corners() {
        let mut ring = lattice_square(20);
        remove_collinear(&mut ring, 0.0);
        assert_eq!(ring.len(), 4);
    }

    #[test]
    fn ring_simplification_keeps_corners_and_area() {
        let ring = lattice_square(40);
        let simplified = simplify_ring(&ring, 0.5);
        assert_eq!(simplified.len(), 4);
        assert!(libm::fabs(polygon_signed_area(&simplified) - 1600.0) < 1.0e-9);
    }

    #[test]
    fn staircase_diagonal_becomes_a_line() {
        let mut ring = Vec::new();
        for step in 0..20 {
            ring.push(Point::new(step as f64, step as f64));
            ring.push(Point::new(step as f64 + 1.0, step as f64));
        }
        ring.push(Point::new(20.0, 20.0));
        ring.push(Point::new(0.0, 20.0));
        let simplified = simplify_ring(&ring, 1.0);
        // The staircase collapses to its diagonal; the other two sides are
        // still represented by their endpoints.
        assert!(simplified.len() <= 4);
    }

    #[test]
    fn corner_classifier_separates_sharp_and_smooth_turns() {
        let mut ring = Vec::new();
        for index in 0..64 {
            let angle = index as f64 * core::f64::consts::TAU / 64.0;
            ring.push(Point::new(libm::cos(angle) * 50.0, libm::sin(angle) * 50.0));
        }
        let corners = classify_corners(&ring, 70.0, 1.0);
        assert!(corners.iter().all(|flag| !flag));

        let square = [
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let corners = classify_corners(&square, 70.0, 1.0);
        assert_eq!(corners, alloc::vec![true, true, true, true]);
    }

    #[test]
    fn run_splitting_covers_every_edge() {
        let corners = [true, false, false, true, false];
        let runs = split_runs(&corners);
        assert_eq!(runs.len(), 2);
        let covered: usize = runs.iter().map(|run| run.edge_count()).sum();
        assert_eq!(covered, corners.len());
        assert_eq!(runs[0], RingRun { start: 0, end: 3 });
        assert_eq!(runs[1], RingRun { start: 3, end: 5 });
    }

    #[test]
    fn run_collection_wraps_around() {
        let points = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        ];
        let run = RingRun { start: 3, end: 5 };
        let collected = collect_run(&points, run);
        assert_eq!(
            collected,
            alloc::vec![Point::new(3.0, 0.0), Point::new(0.0, 0.0), Point::new(1.0, 0.0)]
        );
    }

    #[test]
    fn full_ring_run_is_recognized() {
        let run = RingRun { start: 2, end: 2 };
        assert!(run.is_full_ring(5));
        let run = RingRun { start: 2, end: 7 };
        assert!(run.is_full_ring(5));
        let run = RingRun { start: 2, end: 5 };
        assert!(!run.is_full_ring(5));
    }
}
