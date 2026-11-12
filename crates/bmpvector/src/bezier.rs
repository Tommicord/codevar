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

//! Least-squares cubic Bézier fitting.
//!
//! The fitter follows the classic two-stage shape-preserving approach:
//!
//! 1. parameterize the polyline by chord length;
//! 2. solve a two-unknown least-squares system for the lengths of the endpoint
//!    tangents, with the tangent *directions* supplied by the caller (a corner,
//!    a smoothed endpoint, or an interior split);
//! 3. optionally refine the parameterization with Newton steps and solve again;
//! 4. measure both point-to-curve error and curve-to-polyline error, and split
//!    at the worst point until the error is within tolerance.
//!
//! Solving for tangent lengths instead of all four control points keeps the
//! curve attached to the polyline at both ends and avoids the overshoot that a
//! naive global fit produces at corners. Measuring curve-to-polyline error as
//! well as point-to-curve error catches a cubic that passes through every sample
//! while ballooning between them.
//!
//! All arithmetic is done in `f64` and routed through [`libm`] so the module is
//! usable on `no_std` targets and produces identical results across platforms.

use alloc::vec::Vec;

use libm::fabs;

use crate::geom::{CubicBezier, Point, clamp};
use crate::polygon::{point_polyline_distance, polyline_length};
use crate::vector::Segment;

/// Parameters for a single cubic fitting run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierFitOptions {
    /// Maximum permitted deviation in pixels.
    pub tolerance: f64,
    /// Flatness below which a cubic is emitted as a straight line.
    pub line_tolerance: f64,
    /// Maximum recursive subdivision depth.
    pub max_depth: u32,
    /// Whether to refine the chord-length parameterization with Newton steps.
    pub reparameterize: bool,
}

impl Default for BezierFitOptions {
    fn default() -> Self {
        Self {
            tolerance: 0.45,
            line_tolerance: 0.25,
            max_depth: 12,
            reparameterize: true,
        }
    }
}

impl BezierFitOptions {
    /// Returns options clamped to a usable range.
    pub fn sanitized(self) -> Self {
        Self {
            tolerance: if self.tolerance.is_finite() {
                self.tolerance.max(1.0e-4)
            } else {
                0.45
            },
            line_tolerance: if self.line_tolerance.is_finite() {
                self.line_tolerance.max(0.0)
            } else {
                0.25
            },
            max_depth: self.max_depth.min(64),
            reparameterize: self.reparameterize,
        }
    }
}

/// Fits a polyline with optional explicit endpoint tangents.
///
/// `points` must contain at least two distinct points. For a closed chain the
/// caller should pass the first point again at the end; the shared seam is
/// fitted like any other smooth continuation.
///
/// A chain that contains a non-finite coordinate cannot be fitted, and it
/// yields no segments. Rejecting it here keeps the fitter from degrading such
/// a chain into a single straight chord, which would silently discard the
/// intermediate geometry.
pub fn fit_chain(
    points: &[Point],
    tangent_start: Option<Point>,
    tangent_end: Option<Point>,
    options: BezierFitOptions,
) -> Vec<Segment> {
    let options = options.sanitized();
    if !points.iter().all(|point| point.is_finite()) {
        return Vec::new();
    }
    if tangent_start.is_some_and(|tangent| !tangent.is_finite())
        || tangent_end.is_some_and(|tangent| !tangent.is_finite())
    {
        return Vec::new();
    }
    let mut chain = points.to_vec();
    chain.dedup_by(|a, b| a.distance(*b) <= 1.0e-9);
    if chain.len() < 2 {
        return Vec::new();
    }
    if chain.len() == 2 {
        return alloc::vec![Segment::line(chain[0], chain[1])];
    }

    let start = tangent_start
        .and_then(Point::normalized)
        .or_else(|| estimate_tangent(&chain, true))
        .unwrap_or_else(|| {
            (chain[1] - chain[0])
                .normalized()
                .unwrap_or(Point::new(1.0, 0.0))
        });
    let end = tangent_end
        .and_then(Point::normalized)
        .or_else(|| estimate_tangent(&chain, false))
        .unwrap_or_else(|| {
            (chain[chain.len() - 1] - chain[chain.len() - 2])
                .normalized()
                .unwrap_or(Point::new(-1.0, 0.0))
        });

    let mut output = Vec::new();
    fit_recursive(&chain, start, end, options, 0, &mut output);
    output
}

/// Fits a closed ring as one or more cubic segments.
///
/// The ring is treated as a chain that starts and ends at the same point. The
/// seam tangent is estimated from the vertices on both sides of the start so a
/// smooth circle does not acquire a visible join.
pub fn fit_ring(points: &[Point], options: BezierFitOptions) -> Vec<Segment> {
    if points.len() < 3 {
        return Vec::new();
    }
    let tension = ring_seam_tangent(points);
    let mut chain = Vec::with_capacity(points.len() + 1);
    chain.extend_from_slice(points);
    chain.push(points[0]);
    fit_chain(&chain, Some(tension), Some(tension), options)
}

/// Estimates the tangent at the seam of a closed ring.
pub fn ring_seam_tangent(points: &[Point]) -> Point {
    let count = points.len();
    if count < 2 {
        return Point::new(1.0, 0.0);
    }
    let next = points[1 % count];
    let previous = points[(count - 1) % count];
    (next - previous)
        .normalized()
        .unwrap_or(Point::new(1.0, 0.0))
}

/// Estimates an endpoint tangent from a polyline.
///
/// The immediate segment is preferred; if it is degenerate the next non-zero
/// segment is used instead. `at_start` selects the beginning rather than the
/// end of the chain.
pub fn estimate_tangent(points: &[Point], at_start: bool) -> Option<Point> {
    if points.len() < 2 {
        return None;
    }
    if at_start {
        for index in 1..points.len() {
            if let Some(direction) = (points[index] - points[0]).normalized() {
                return Some(direction);
            }
        }
    } else {
        let last = points.len() - 1;
        for index in (0..last).rev() {
            if let Some(direction) = (points[last] - points[index]).normalized() {
                return Some(direction);
            }
        }
    }
    None
}

/// Estimates an interior tangent around `index`.
///
/// The chord across the neighbours is less sensitive to a single rasterized
/// vertex than the incoming or outgoing segment alone.
pub fn estimate_interior_tangent(points: &[Point], index: usize) -> Option<Point> {
    if points.len() < 2 {
        return None;
    }
    let index = index.min(points.len() - 1);
    if index == 0 {
        return estimate_tangent(points, true);
    }
    if index == points.len() - 1 {
        return estimate_tangent(points, false);
    }
    let before = points[index - 1];
    let after = points[index + 1];
    (after - before)
        .normalized()
        .or_else(|| (points[index] - before).normalized())
        .or_else(|| (after - points[index]).normalized())
}

#[derive(Clone, Copy, Debug)]
struct FitError {
    maximum: f64,
    split_index: usize,
}

fn fit_recursive(
    points: &[Point],
    tangent_start: Point,
    tangent_end: Point,
    options: BezierFitOptions,
    depth: u32,
    output: &mut Vec<Segment>,
) {
    if points.len() < 2 {
        return;
    }
    if points.len() == 2 {
        output.push(Segment::line(points[0], points[1]));
        return;
    }

    let mut parameters = chord_parameters(points);
    let mut cubic = generate_bezier(points, &parameters, tangent_start, tangent_end);
    if options.reparameterize {
        // Chord-length parameters are only a starting point. Re-solve for the
        // control points, then use the new curve to refine the parameters
        // again; each round lowers the fixed-point error left by the previous
        // approximation. Keeping the best round avoids returning a worse curve
        // if a Newton step temporarily overshoots.
        const ROUNDS: u32 = 3;
        const NEWTON_STEPS: u32 = 4;
        let mut best_parameters = parameters.clone();
        let mut best_cubic = cubic;
        let mut best_error = measure_error(points, &best_parameters, best_cubic);
        let mut refined = parameters.clone();
        let mut current = cubic;
        for _ in 0..ROUNDS {
            reparameterize(points, &mut refined, current, NEWTON_STEPS);
            let candidate = generate_bezier(points, &refined, tangent_start, tangent_end);
            if !candidate.is_finite() {
                break;
            }
            let error = measure_error(points, &refined, candidate);
            if error.maximum < best_error.maximum {
                best_error = error;
                best_parameters.clone_from(&refined);
                best_cubic = candidate;
            }
            current = candidate;
        }
        parameters = best_parameters;
        cubic = best_cubic;
    }

    if !cubic.is_finite() {
        // The inputs are finite, so this only guards against a numerical
        // overflow inside the solver. Emitting the polyline preserves the
        // intermediate geometry instead of collapsing the chain into a single
        // chord.
        for window in points.windows(2) {
            output.push(Segment::line(window[0], window[1]));
        }
        return;
    }

    let error = measure_error(points, &parameters, cubic);
    let within_tolerance = error.maximum <= options.tolerance;
    let at_depth_limit = depth >= options.max_depth;

    if within_tolerance || at_depth_limit {
        if cubic.flatness() <= options.line_tolerance
            && cubic.control1.distance(cubic.start) + cubic.control2.distance(cubic.end)
                <= cubic.start.distance(cubic.end) * 2.0 + options.line_tolerance
        {
            output.push(Segment::line(cubic.start, cubic.end));
        } else {
            output.push(Segment::Cubic(cubic));
        }
        return;
    }

    let split = error
        .split_index
        .clamp(1, points.len().saturating_sub(2).max(1));
    let interior = estimate_interior_tangent(points, split).unwrap_or_else(|| {
        (points[split] - points[split - 1])
            .normalized()
            .unwrap_or(tangent_end)
    });

    fit_recursive(
        &points[..=split],
        tangent_start,
        interior,
        options,
        depth + 1,
        output,
    );
    fit_recursive(
        &points[split..],
        interior,
        tangent_end,
        options,
        depth + 1,
        output,
    );
}

/// Chord-length parameterization normalized to `[0, 1]`.
pub fn chord_parameters(points: &[Point]) -> Vec<f64> {
    let count = points.len();
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return alloc::vec![0.0];
    }
    let mut parameters = Vec::with_capacity(count);
    parameters.push(0.0);
    let mut cumulative = 0.0;
    let mut previous = points[0];
    for &point in &points[1..] {
        cumulative += previous.distance(point);
        parameters.push(cumulative);
        previous = point;
    }
    if cumulative <= 1.0e-12 {
        for (index, parameter) in parameters.iter_mut().enumerate() {
            *parameter = index as f64 / (count - 1) as f64;
        }
    } else {
        for parameter in &mut parameters {
            *parameter /= cumulative;
        }
    }
    parameters[count - 1] = 1.0;
    parameters
}

/// Solves for a cubic with fixed endpoint tangents and free tangent lengths.
///
/// This is the normal-equations form of the least-squares problem. When the
/// tangent directions are nearly parallel the system is ill-conditioned, so the
/// geometric fallback `chord / 3` is used instead.
pub fn generate_bezier(
    points: &[Point],
    parameters: &[f64],
    tangent_start: Point,
    tangent_end: Point,
) -> CubicBezier {
    let count = points.len();
    if count < 2 || parameters.len() != count {
        return CubicBezier::new(
            points.first().copied().unwrap_or_default(),
            Point::ZERO,
            Point::ZERO,
            points.last().copied().unwrap_or_default(),
        );
    }
    let start = points[0];
    let end = points[count - 1];
    let chord = start.distance(end);
    let fallback = (chord / 3.0).max(1.0e-6);

    let mut c00 = 0.0;
    let mut c01 = 0.0;
    let mut c11 = 0.0;
    let mut x0 = 0.0;
    let mut x1 = 0.0;

    for index in 1..count - 1 {
        let u = clamp(parameters[index], 0.0, 1.0);
        let one_minus = 1.0 - u;
        let b0 = one_minus * one_minus * one_minus;
        let b1 = 3.0 * u * one_minus * one_minus;
        let b2 = 3.0 * u * u * one_minus;
        let b3 = u * u * u;

        let a0 = tangent_start * b1;
        let a1 = tangent_end * -b2;
        let target = points[index] - start * (b0 + b1) - end * (b2 + b3);

        c00 += a0.dot(a0);
        c01 += a0.dot(a1);
        c11 += a1.dot(a1);
        x0 += a0.dot(target);
        x1 += a1.dot(target);
    }

    let determinant = c00 * c11 - c01 * c01;
    let mut alpha0 = 0.0;
    let mut alpha1 = 0.0;
    if fabs(determinant) > 1.0e-12 {
        alpha0 = (x0 * c11 - x1 * c01) / determinant;
        alpha1 = (c00 * x1 - c01 * x0) / determinant;
    }
    if !alpha0.is_finite() {
        alpha0 = fallback;
    }
    if !alpha1.is_finite() {
        alpha1 = fallback;
    }
    // The two lengths are independent unknowns of the normal equations, so a
    // non-positive solution only invalidates the one it belongs to. Replacing
    // both would throw away a perfectly good length and cost an extra
    // subdivision. The geometric estimate is a stable stand-in for the
    // offending value; the recursive error check decides whether the result
    // is acceptable.
    if alpha0 <= 0.0 {
        alpha0 = fallback;
    }
    if alpha1 <= 0.0 {
        alpha1 = fallback;
    }
    alpha0 = clamp(alpha0, 0.0, chord * 4.0 + 1.0);
    alpha1 = clamp(alpha1, 0.0, chord * 4.0 + 1.0);

    CubicBezier::new(
        start,
        start + tangent_start * alpha0,
        end - tangent_end * alpha1,
        end,
    )
}

fn measure_error(points: &[Point], parameters: &[f64], cubic: CubicBezier) -> FitError {
    let mut maximum: f64 = 0.0;
    let mut split_index = points.len() / 2;

    for (index, (&point, &parameter)) in points.iter().zip(parameters).enumerate() {
        let distance = point.distance(cubic.eval(clamp(parameter, 0.0, 1.0)));
        if distance > maximum {
            maximum = distance;
            split_index = index;
        }
    }

    // Sample the curve and compare it with the polyline. This catches a cubic
    // that interpolates the samples but bows away between them.
    const CURVE_SAMPLES: usize = 24;
    for sample in 1..CURVE_SAMPLES {
        let t = sample as f64 / CURVE_SAMPLES as f64;
        let point = cubic.eval(t);
        let distance = point_polyline_distance(point, points);
        if distance > maximum {
            maximum = distance;
            let index = libm::round(t * (points.len() - 1) as f64) as usize;
            split_index = index.clamp(1, points.len().saturating_sub(2).max(1));
        }
    }

    if !maximum.is_finite() {
        maximum = f64::INFINITY;
    }
    FitError { maximum, split_index }
}

fn reparameterize(points: &[Point], parameters: &mut [f64], cubic: CubicBezier, iterations: u32) {
    let count = points.len();
    if count < 3 || parameters.len() != count {
        return;
    }
    for _ in 0..iterations {
        for index in 1..count - 1 {
            let point = points[index];
            let u = clamp(parameters[index], 0.0, 1.0);
            let curve_point = cubic.eval(u);
            let first = cubic.derivative(u);
            let second = cubic.second_derivative(u);
            let difference = curve_point - point;
            let numerator = difference.dot(first);
            let denominator = first.dot(first) + difference.dot(second);
            if fabs(denominator) > 1.0e-12 {
                parameters[index] = clamp(u - numerator / denominator, 0.0, 1.0);
            }
        }
        parameters[0] = 0.0;
        parameters[count - 1] = 1.0;
    }
}

/// Maximum distance between a fitted cubic and the polyline that produced it.
///
/// Exposed for tests and for callers that want to audit a fitted path.
pub fn chain_error(points: &[Point], segments: &[Segment]) -> f64 {
    if segments.is_empty() {
        return f64::INFINITY;
    }
    // Flatten each cubic once. Re-flattening inside the point loop would turn a
    // linear scan into a much more expensive operation for curves.
    let polylines: Vec<Vec<Point>> = segments
        .iter()
        .map(|&segment| flatten_segment(segment, 0.01))
        .collect();
    let mut maximum: f64 = 0.0;
    for &point in points {
        let mut best = f64::INFINITY;
        for polyline in &polylines {
            best = best.min(point_polyline_distance(point, polyline));
        }
        maximum = maximum.max(best);
    }
    maximum
}

fn flatten_segment(segment: Segment, tolerance: f64) -> Vec<Point> {
    let mut points = Vec::with_capacity(16);
    points.push(segment.start());
    segment.flatten_into(tolerance, &mut points);
    points
}

/// Approximate length of a segment by flattening it at half-pixel tolerance.
pub fn segment_length(segment: Segment) -> f64 {
    match segment {
        Segment::Line { start, end } => start.distance(end),
        Segment::Cubic(cubic) => {
            let mut points = Vec::with_capacity(17);
            points.push(cubic.start);
            cubic.sample_into(16, &mut points);
            polyline_length(&points)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        fabs(a - b) <= tolerance
    }

    #[test]
    fn straight_chain_becomes_one_line() {
        let points = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        ];
        let segments = fit_chain(&points, None, None, BezierFitOptions::default());
        assert_eq!(segments.len(), 1);
        assert!(matches!(segments[0], Segment::Line { .. }));
    }

    #[test]
    fn quarter_circle_fits_with_small_error() {
        let mut points = Vec::new();
        for index in 0..=32 {
            let angle = index as f64 * core::f64::consts::FRAC_PI_2 / 32.0;
            points.push(Point::new(libm::cos(angle) * 50.0, libm::sin(angle) * 50.0));
        }
        let segments = fit_chain(
            &points,
            Some(Point::new(1.0, 0.0)),
            Some(Point::new(0.0, 1.0)),
            BezierFitOptions::default(),
        );
        assert!(!segments.is_empty());
        let error = chain_error(&points, &segments);
        assert!(error < 0.6, "error was {error}");
        assert!(
            segments
                .iter()
                .any(|segment| matches!(segment, Segment::Cubic(_)))
        );
    }

    #[test]
    fn cubic_sample_reproduces_a_cubic() {
        let source = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(20.0, 40.0),
            Point::new(80.0, -20.0),
            Point::new(100.0, 10.0),
        );
        let mut points = Vec::new();
        for index in 0..=32 {
            let t = index as f64 / 32.0;
            points.push(source.eval(t));
        }
        let segments = fit_chain(
            &points,
            Some(source.tangent(0.0).expect("tangent")),
            Some(source.tangent(1.0).expect("tangent")),
            BezierFitOptions::default(),
        );
        assert_eq!(segments.len(), 1);
        assert!(chain_error(&points, &segments) < 0.5);
    }

    #[test]
    fn s_curve_splits_until_tolerance_is_met() {
        let mut points = Vec::new();
        for index in 0..=48 {
            let t = index as f64 / 48.0;
            points.push(Point::new(
                t * 100.0,
                libm::sin(t * core::f64::consts::TAU) * 30.0,
            ));
        }
        let segments = fit_chain(&points, None, None, BezierFitOptions::default());
        let error = chain_error(&points, &segments);
        assert!(segments.len() >= 2);
        assert!(error < 0.7, "segments {} error {error}", segments.len());
    }

    #[test]
    fn chord_parameters_are_monotonic() {
        let points = [Point::new(0.0, 0.0), Point::new(3.0, 0.0), Point::new(3.0, 4.0)];
        let parameters = chord_parameters(&points);
        assert_eq!(parameters.first(), Some(&0.0));
        assert_eq!(parameters.last(), Some(&1.0));
        assert!(parameters[0] < parameters[1]);
        assert!(parameters[1] < parameters[2]);
        assert!(close(parameters[1], 3.0 / 7.0, 1.0e-12));
    }

    #[test]
    fn interior_tangent_uses_both_neighbours() {
        let points = [Point::new(0.0, 0.0), Point::new(1.0, 1.0), Point::new(2.0, 0.0)];
        let tangent = estimate_interior_tangent(&points, 1).expect("tangent");
        assert!(close(tangent.x, 1.0, 1.0e-12));
        assert!(close(tangent.y, 0.0, 1.0e-12));
    }

    #[test]
    fn ring_seam_tangent_points_along_the_ring() {
        let points = [Point::new(1.0, 0.0), Point::new(0.0, 1.0), Point::new(-1.0, 0.0)];
        let tangent = ring_seam_tangent(&points);
        assert!(tangent.is_finite());
        assert!(close(tangent.length(), 1.0, 1.0e-12));
    }

    #[test]
    fn fit_ring_produces_closed_continuity() {
        let mut points = Vec::new();
        for index in 0..64 {
            let angle = index as f64 * core::f64::consts::TAU / 64.0;
            points.push(Point::new(libm::cos(angle) * 20.0, libm::sin(angle) * 20.0));
        }
        let segments = fit_ring(&points, BezierFitOptions::default());
        assert!(segments.len() >= 2);
        assert!(
            segments[0]
                .start()
                .distance(segments[segments.len() - 1].end())
                < 1.0e-7
        );
        assert!(chain_error(&points, &segments) < 1.0);
    }

    #[test]
    fn segment_length_is_finite() {
        let cubic = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(20.0, -10.0),
            Point::new(30.0, 0.0),
        );
        assert!(segment_length(Segment::Cubic(cubic)) > 30.0);
        assert!(close(
            segment_length(Segment::line(Point::new(0.0, 0.0), Point::new(3.0, 4.0))),
            5.0,
            1.0e-12
        ));
    }
}
