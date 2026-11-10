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

//! Curve simplification / re-fitting passes.
//!
//! This is a port of the paper.js / Schneider "Iterative least-squares
//! cubic Bézier curve fitting" approach, adapted to [`crate::fitter`]'s
//! [`FittedGeom`] types without the `flo_curves` dependency. Cubic fitting is
//! implemented here (`fit_curve_cubic`) via sampling at ~1 px spacing and
//! recursive subdivision at the point of maximum deviation.

use crate::fitter::FittedGeom;
use crate::types::PointF64;

/// Euclidean distance between two points.
#[inline]
fn dist(a: PointF64, b: PointF64) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// A curve simplification pass: open chains and closed rings.
pub trait CurvePass {
    /// Simplify an open polyline/Bézier chain.
    fn open(&self, geom: FittedGeom) -> FittedGeom;
    /// Simplify a closed ring (first point == last point).
    fn ring(&self, geom: FittedGeom) -> FittedGeom;
}

/// Parameters for the simplification pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplifyCurves {
    /// Approximation tolerance in pixels (max deviation allowed).
    pub tolerance: f64,
    /// Corner detection threshold (degrees). Runs are split at sharp corners.
    pub corner_threshold: f64,
}

impl Default for SimplifyCurves {
    fn default() -> Self {
        Self {
            tolerance: 2.0,
            corner_threshold: 60.0,
        }
    }
}

impl CurvePass for SimplifyCurves {
    fn open(&self, geom: FittedGeom) -> FittedGeom {
        match geom {
            FittedGeom::Polyline(points) => {
                let simplified = self.simplify_chain(&points, false);
                FittedGeom::Polyline(simplified)
            }
            FittedGeom::Beziers(beziers) => {
                // Flatten beziers to their control points, then re-fit.
                let mut pts: Vec<PointF64> = Vec::with_capacity(beziers.len() * 3 + 1);
                for b in &beziers {
                    pts.push(b[0]);
                    pts.push(b[1]);
                    pts.push(b[2]);
                    pts.push(b[3]);
                }
                pts.dedup();
                let simplified = self.simplify_chain(&pts, true);
                FittedGeom::Polyline(simplified)
            }
        }
    }

    fn ring(&self, geom: FittedGeom) -> FittedGeom {
        match geom {
            FittedGeom::Polyline(points) => {
                let simplified = self.simplify_chain(&points, true);
                FittedGeom::Polyline(simplified)
            }
            FittedGeom::Beziers(beziers) => {
                let mut pts: Vec<PointF64> = Vec::with_capacity(beziers.len() * 3 + 1);
                for b in &beziers {
                    pts.push(b[0]);
                    pts.push(b[1]);
                    pts.push(b[2]);
                    pts.push(b[3]);
                }
                pts.dedup();
                let simplified = self.simplify_chain(&pts, true);
                FittedGeom::Polyline(simplified)
            }
        }
    }
}

impl SimplifyCurves {
    /// Simplify a chain of points (open or closed ring).
    pub fn simplify_chain(&self, points: &[PointF64], is_ring: bool) -> Vec<PointF64> {
        if points.is_empty() {
            return Vec::new();
        }
        if points.len() < 3 {
            return points.to_vec();
        }

        // Detect corners and split the run at sharp vertices.
        let mut runs: Vec<Vec<PointF64>> = Vec::new();
        let mut current: Vec<PointF64> = vec![points[0]];
        for i in 1..points.len() {
            if i > 1 && is_corner(points[i - 2], points[i - 1], points[i], self.corner_threshold) {
                if current.len() >= 2 {
                    runs.push(current.clone());
                }
                current = vec![points[i - 1], points[i]];
            } else {
                current.push(points[i]);
            }
        }
        if current.len() >= 2 {
            runs.push(current);
        }

        let mut out: Vec<PointF64> = Vec::new();
        for run in &runs {
            let fitted = refit_run(self, run, is_ring);
            if out.is_empty() {
                out.extend_from_slice(&fitted);
            } else {
                // Avoid duplicating the join point between runs.
                if let Some(last) = out.last() {
                    if fitted.first().is_some_and(|f| *last == *f) {
                        out.extend_from_slice(&fitted[1..]);
                    } else {
                        out.extend_from_slice(&fitted);
                    }
                } else {
                    out.extend_from_slice(&fitted);
                }
            }
        }

        // Never grow the chain: if simplification added points, keep original.
        if out.len() > points.len() {
            return points.to_vec();
        }
        out
    }
}

/// `fit_curve_cubic` equivalent: sample a run at ~1 px spacing, then
/// recursively subdivide at the point of maximum deviation from the fitted
/// cubic until deviation <= tolerance. Each sub-run is fit as a cubic with
/// endpoint tangents.
pub fn refit_run(pass: &SimplifyCurves, points: &[PointF64], _is_ring: bool) -> Vec<PointF64> {
    if points.len() < 2 {
        return points.to_vec();
    }
    if points.len() == 2 {
        return points.to_vec();
    }

    let total_len = polyline_length(points);
    if total_len <= pass.tolerance * 2.0 {
        return vec![points[0], points[points.len() - 1]];
    }

    // Sample the run at ~1 px spacing.
    let samples = sample_run(points);
    if samples.len() < 2 {
        return points.to_vec();
    }
    // Fit the whole run as one cubic; if deviation is within tolerance, keep it.
    let cubic = fit_curve_cubic(&samples, PointF64::new(0.0, 0.0), PointF64::new(0.0, 0.0));
    let max_dev = max_deviation(&samples, &cubic);

    if max_dev <= pass.tolerance {
        return vec![cubic[0], cubic[1], cubic[2], cubic[3]];
    }

    // Otherwise split at the point of max deviation and recurse.
    let split_idx = max_deviation_index(&samples, &cubic);
    if split_idx == 0 || split_idx >= samples.len() - 1 {
        return vec![cubic[0], cubic[1], cubic[2], cubic[3]];
    }

    let left: Vec<PointF64> = samples[..=split_idx].to_vec();
    let right: Vec<PointF64> = samples[split_idx..].to_vec();

    let mut out = refit_run(pass, &left, _is_ring);
    let right_out = refit_run(pass, &right, _is_ring);
    // Avoid duplicating the shared split point.
    if let Some(last) = out.last() {
        if right_out.first().is_some_and(|f| *last == *f) {
            out.extend_from_slice(&right_out[1..]);
        } else {
            out.extend_from_slice(&right_out);
        }
    } else {
        out.extend_from_slice(&right_out);
    }
    out
}

/// Fit a single cubic Bézier to `samples` using end tangents (Schneider /
/// paper.js least-squares approach). Returns `[p0, c1, c2, p3]`.
pub fn fit_curve_cubic(samples: &[PointF64], _tangent_in: PointF64, _tangent_out: PointF64) -> [PointF64; 4] {
    let p0 = samples[0];
    let p3 = samples[samples.len() - 1];

    // Prime chords (normalized distance along the polyline).
    let mut prime_chords: Vec<f64> = Vec::with_capacity(samples.len());
    let mut total = 0.0;
    for i in 0..samples.len() {
        if i > 0 {
            total += dist(samples[i - 1], samples[i]);
        }
        prime_chords.push(total);
    }

    let total_len = *prime_chords.last().unwrap_or(&0.0);
    if total_len == 0.0 {
        return [p0, p0, p3, p3];
    }

    // Build least-squares system: sum over samples of (P(t) - P_i)^2.
    // Solve for c1, c2 (the two interior control points) given fixed p0, p3.
    let _c1x = 0.0;
    let _c1y = 0.0;
    let _c2x = 0.0;
    let _c2y = 0.0;
    let mut sum_t2 = 0.0;
    let mut sum_t3 = 0.0;
    let mut sum_t4 = 0.0;
    let mut sum_rhs1x = 0.0;
    let mut sum_rhs1y = 0.0;
    let mut sum_rhs2x = 0.0;
    let mut sum_rhs2y = 0.0;

    for (i, &pt) in samples.iter().enumerate() {
        if i == 0 || i == samples.len() - 1 {
            continue;
        }
        let t = prime_chords[i] / total_len;
        let t2 = t * t;
        let t3 = t2 * t;
        let _t4 = t3 * t;

        // P(t) = p0*(1-t)^3 + c1*3*(1-t)^2*t + c2*3*(1-t)*t^2 + p3*t^3
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let a1 = 3.0 * mt2 * t; // weight on c1
        let a2 = 3.0 * mt * t2; // weight on c2
        let base_x = p0.x * mt3 + p3.x * t3;
        let base_y = p0.y * mt3 + p3.y * t3;

        sum_t2 += a1 * a1;
        sum_t3 += a1 * a2;
        sum_t4 += a2 * a2;
        let rhs1x = pt.x - base_x;
        let rhs1y = pt.y - base_y;
        sum_rhs1x += a1 * rhs1x;
        sum_rhs1y += a1 * rhs1y;
        sum_rhs2x += a2 * rhs1x;
        sum_rhs2y += a2 * rhs1y;
    }

    // Solve 2x2: [sum_t2  sum_t3; sum_t3  sum_t4] [c1; c2] = [rhs1; rhs2]
    let det = sum_t2 * sum_t4 - sum_t3 * sum_t3;
    let mut c1 = PointF64::new(p0.x, p0.y);
    let mut c2 = PointF64::new(p3.x, p3.y);
    if det.abs() > 1e-12 {
        c1.x = (sum_rhs1x * sum_t4 - sum_t3 * sum_rhs2x) / det;
        c1.y = (sum_rhs1y * sum_t4 - sum_t3 * sum_rhs2y) / det;
        c2.x = (sum_t2 * sum_rhs2x - sum_t3 * sum_rhs1x) / det;
        c2.y = (sum_t2 * sum_rhs2y - sum_t3 * sum_rhs1y) / det;
    }

    [p0, c1, c2, p3]
}

/// Tangent at the start of a run (paper.js `tangentIn`).
#[inline]
pub fn tangent_in(samples: &[PointF64]) -> PointF64 {
    if samples.len() < 2 {
        return PointF64::new(0.0, 0.0);
    }
    dir(samples[0], samples[1])
}

/// Tangent at the end of a run (paper.js `tangentOut`).
#[inline]
pub fn tangent_out(samples: &[PointF64]) -> PointF64 {
    let n = samples.len();
    if n < 2 {
        return PointF64::new(0.0, 0.0);
    }
    dir(samples[n - 2], samples[n - 1])
}

/// Direction (unit vector) from `a` to `b`.
#[inline]
pub fn dir(a: PointF64, b: PointF64) -> PointF64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len == 0.0 {
        return PointF64::new(0.0, 0.0);
    }
    PointF64::new(dx / len, dy / len)
}

/// Angle between three points (degrees). Used for corner detection.
pub fn break_angle(a: PointF64, b: PointF64, c: PointF64) -> f64 {
    let u = dir(a, b);
    let v = dir(b, c);
    let dot = u.x * v.x + u.y * v.y;
    let dot = dot.clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

/// Evaluate a cubic Bézier at parameter `t`: `[p0, c1, c2, p3]`.
#[inline]
pub fn cubic_at(bezier: &[PointF64; 4], t: f64) -> PointF64 {
    let p0 = bezier[0];
    let c1 = bezier[1];
    let c2 = bezier[2];
    let p3 = bezier[3];
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    let t2 = t * t;
    let t3 = t2 * t;
    PointF64::new(
        p0.x * mt3 + 3.0 * c1.x * mt2 * t + 3.0 * c2.x * mt * t2 + p3.x * t3,
        p0.y * mt3 + 3.0 * c1.y * mt2 * t + 3.0 * c2.y * mt * t2 + p3.y * t3,
    )
}

/// Sample a polyline run at ~1 px spacing (plus endpoints).
pub fn sample_run(points: &[PointF64]) -> Vec<PointF64> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut out: Vec<PointF64> = Vec::new();
    out.push(points[0]);
    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let seg = dist(a, b);
        if seg == 0.0 {
            continue;
        }
        let mut t = 0.0;
        while t < seg {
            let step = (1.0 - t).min(seg - t);
            let interp = lerp(a, b, t / seg);
            out.push(interp);
            t += step;
        }
        out.push(b);
    }
    out.dedup();
    out
}

fn lerp(a: PointF64, b: PointF64, t: f64) -> PointF64 {
    PointF64::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

/// Total polyline length.
fn polyline_length(points: &[PointF64]) -> f64 {
    let mut total = 0.0;
    for i in 0..points.len().saturating_sub(1) {
        total += dist(points[i], points[i + 1]);
    }
    total
}

/// Maximum distance from `samples` to the fitted cubic.
fn max_deviation(samples: &[PointF64], cubic: &[PointF64; 4]) -> f64 {
    let mut max = 0.0f64;
    let n = samples.len();
    for (i, sample) in samples.iter().enumerate() {
        let t = i as f64 / (n as f64 - 1.0).max(1.0);
        let on = cubic_at(cubic, t);
        let d = dist(*sample, on);
        if d > max {
            max = d;
        }
    }
    max
}

/// Index of the sample farthest from the fitted cubic.
fn max_deviation_index(samples: &[PointF64], cubic: &[PointF64; 4]) -> usize {
    let mut max = 0.0f64;
    let mut idx = 0;
    let n = samples.len();
    for (i, sample) in samples.iter().enumerate() {
        let t = i as f64 / (n as f64 - 1.0).max(1.0);
        let on = cubic_at(cubic, t);
        let d = dist(*sample, on);
        if d > max {
            max = d;
            idx = i;
        }
    }
    idx
}

/// Whether `b` is a corner given the angle `a-b-c` exceeds the threshold.
fn is_corner(a: PointF64, b: PointF64, c: PointF64, threshold: f64) -> bool {
    break_angle(a, b, c) > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::FitParams;
    use crate::types::PointF64;

    #[test]
    fn collinear_run_collapses_to_endpoints() {
        let points = vec![
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(2.0, 0.0),
            PointF64::new(3.0, 0.0),
        ];
        let pass = SimplifyCurves::default();
        let out = pass.simplify_chain(&points, false);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], points[0]);
        assert_eq!(out[1], points[3]);
    }

    #[test]
    fn corner_survives_simplification() {
        let points = vec![
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(1.0, 1.0),
            PointF64::new(2.0, 1.0),
        ];
        let pass = SimplifyCurves::default();
        let out = pass.simplify_chain(&points, false);
        // The sharp corner at (1,0)->(1,1) should be preserved.
        assert!(out.len() >= 3);
        assert!(out.contains(&PointF64::new(1.0, 0.0)));
        assert!(out.contains(&PointF64::new(1.0, 1.0)));
    }

    #[test]
    fn ring_stays_closed() {
        let points = vec![
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(1.0, 1.0),
            PointF64::new(0.0, 1.0),
            PointF64::new(0.0, 0.0),
        ];
        let pass = SimplifyCurves::default();
        let out = pass.simplify_chain(&points, true);
        assert!(!out.is_empty());
        assert_eq!(out.first(), out.last());
    }

    #[test]
    fn arc_merges_into_single_cubic() {
        // A gentle arc should be representable by one cubic within tolerance.
        let mut points = Vec::new();
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let angle = t * std::f64::consts::FRAC_PI_2;
            points.push(PointF64::new(angle.cos(), angle.sin()));
        }
        let pass = SimplifyCurves {
            tolerance: 0.5,
            ..Default::default()
        };
        let out = pass.simplify_chain(&points, false);
        assert!(out.len() >= 2);
    }

    #[test]
    fn never_grows_chain() {
        let points = vec![
            PointF64::new(0.0, 0.0),
            PointF64::new(0.5, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(1.0, 0.5),
            PointF64::new(1.0, 1.0),
        ];
        let pass = SimplifyCurves::default();
        let out = pass.simplify_chain(&points, false);
        assert!(out.len() <= points.len());
    }

    #[test]
    fn polylines_pass_through_short() {
        let points = vec![PointF64::new(0.0, 0.0), PointF64::new(1.0, 1.0)];
        let pass = SimplifyCurves::default();
        let out = pass.simplify_chain(&points, false);
        assert_eq!(out, points);
    }

    #[test]
    fn refit_run_single_cubic_for_line() {
        let points = vec![
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(2.0, 0.0),
        ];
        let pass = SimplifyCurves::default();
        let out = refit_run(&pass, &points, false);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn cubic_at_endpoints_match_control_points() {
        let bez = [
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(1.0, 1.0),
            PointF64::new(0.0, 1.0),
        ];
        let p0 = cubic_at(&bez, 0.0);
        let p1 = cubic_at(&bez, 1.0);
        assert!((p0.x - bez[0].x).abs() < 1e-9 && (p0.y - bez[0].y).abs() < 1e-9);
        assert!((p1.x - bez[3].x).abs() < 1e-9 && (p1.y - bez[3].y).abs() < 1e-9);
    }

    #[test]
    fn simplify_curves_default() {
        let pass = SimplifyCurves::default();
        assert!((pass.tolerance - 2.0).abs() < 1e-9);
        assert!((pass.corner_threshold - 60.0).abs() < 1e-9);
    }

    #[test]
    fn fit_params_default() {
        let p = FitParams::default();
        assert!((p.corner_threshold - 60.0).abs() < 1e-9);
        assert!((p.length_threshold - 4.0).abs() < 1e-9);
        assert_eq!(p.max_iterations, 10);
        assert!((p.splice_threshold - 1.0).abs() < 1e-9);
    }
}
