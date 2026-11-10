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

//! Optimizer passes that post-process a [`VectorDoc`] after fitting.
//!
//! Faithful port of visioncortex/vtracer's `crates/vtracer/src/optimize.rs`:
//! a `QuantizePass` that snaps path points to a fixed precision grid, and a
//! `CleanupPass` that drops zero-length and collinear-redundant `LineTo`
//! segments.

use crate::ir::{PathCmd, VectorDoc};
use crate::types::PointF64;

/// A single optimization pass over a [`VectorDoc`].
///
/// Passes run in order after fitting; each one mutates the document in place.
pub trait OptimizerPass {
    /// Run this pass on `doc`, mutating it in place.
    fn run(&self, doc: &mut VectorDoc);
}

/// Round a floating point coordinate to the given precision (number of
/// fractional digits).
#[inline]
fn round(value: f64, precision: u32) -> f64 {
    if precision == 0 {
        return value.round();
    }
    let scale = 10_f64.powi(precision as i32);
    (value * scale).round() / scale
}

/// Round a point's coordinates to the given precision.
#[inline]
fn round_pt(pt: PointF64, precision: u32) -> PointF64 {
    PointF64::new(round(pt.x, precision), round(pt.y, precision))
}

/// Quantization pass: snap every path point to a fixed-precision grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizePass {
    precision: u32,
}

impl QuantizePass {
    /// Create a quantization pass rounding to `precision` fractional digits.
    #[inline]
    pub fn new(precision: u32) -> Self {
        Self { precision }
    }

    /// The number of fractional digits points are rounded to.
    #[inline]
    pub fn precision(self) -> u32 {
        self.precision
    }
}

impl OptimizerPass for QuantizePass {
    fn run(&self, doc: &mut VectorDoc) {
        for shape in &mut doc.shapes {
            for subpath in &mut shape.path.subpaths {
                for cmd in &mut subpath.commands {
                    match cmd {
                        PathCmd::MoveTo(p) => *p = round_pt(*p, self.precision),
                        PathCmd::LineTo(p) => *p = round_pt(*p, self.precision),
                        PathCmd::CubicTo(c1, c2, c3) => {
                            *c1 = round_pt(*c1, self.precision);
                            *c2 = round_pt(*c2, self.precision);
                            *c3 = round_pt(*c3, self.precision);
                        }
                        PathCmd::Close => {}
                    }
                }
            }
        }
    }
}

/// Maximum distance for two points to be considered coincident.
const COINCIDENT_EPS: f64 = 1e-6;
/// Maximum cross-product magnitude for three points to be considered collinear.
const COLLINEAR_EPS: f64 = 1e-4;

/// Whether two points are coincident within epsilon.
#[inline]
fn approx_eq(a: PointF64, b: PointF64) -> bool {
    (a.x - b.x).abs() < COINCIDENT_EPS && (a.y - b.y).abs() < COINCIDENT_EPS
}

/// Whether three points are collinear within epsilon.
#[inline]
fn collinear(a: PointF64, b: PointF64, c: PointF64) -> bool {
    let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    cross.abs() < COLLINEAR_EPS
}

/// Clean up a single subpath: drop coincident points and collinear middle
/// points on `LineTo` runs.
fn cleanup_subpath(commands: &mut Vec<PathCmd>) {
    if commands.is_empty() {
        return;
    }

    let mut out: Vec<PathCmd> = Vec::with_capacity(commands.len());
    let mut prev: Option<PointF64> = None;

    for cmd in commands.drain(..) {
        match cmd {
            PathCmd::MoveTo(p) => {
                out.push(PathCmd::MoveTo(p));
                prev = Some(p);
            }
            PathCmd::LineTo(p) => {
                if let Some(last) = prev {
                    if approx_eq(last, p) {
                        // zero-length segment; drop it
                        continue;
                    }
                    // if we already have a line-to run, check collinearity of
                    // the last three points
                    let needs_push = match out.last() {
                        Some(PathCmd::LineTo(_)) => {
                            // need the point before last line-to
                            let before = out
                                .iter()
                                .rev()
                                .skip_while(|c| matches!(c, PathCmd::LineTo(_)))
                                .find_map(|c| match c {
                                    PathCmd::MoveTo(pt) | PathCmd::LineTo(pt) => Some(*pt),
                                    _ => None,
                                });
                            match before {
                                Some(a) => !collinear(a, last, p),
                                None => true,
                            }
                        }
                        _ => true,
                    };
                    if needs_push {
                        out.push(PathCmd::LineTo(p));
                        prev = Some(p);
                    } else {
                        // drop the redundant collinear line-to; replace it with
                        // this endpoint so the run continues
                        if let Some(PathCmd::LineTo(last_pt)) = out.last_mut() {
                            *last_pt = p;
                        }
                        prev = Some(p);
                    }
                } else {
                    out.push(PathCmd::LineTo(p));
                    prev = Some(p);
                }
            }
            PathCmd::CubicTo(..) | PathCmd::Close => {
                out.push(cmd);
                prev = match cmd {
                    PathCmd::CubicTo(_, _, end) => Some(end),
                    PathCmd::Close => None,
                    _ => prev,
                };
            }
        }
    }

    *commands = out;
}

/// Cleanup pass: drop zero-length and collinear-redundant `LineTo` segments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CleanupPass;

impl CleanupPass {
    /// Create a new cleanup pass with default thresholds.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl OptimizerPass for CleanupPass {
    fn run(&self, doc: &mut VectorDoc) {
        for shape in &mut doc.shapes {
            for subpath in &mut shape.path.subpaths {
                cleanup_subpath(&mut subpath.commands);
            }
            // drop subpaths that became empty
            shape
                .path
                .subpaths
                .retain(|sp| !sp.commands.is_empty());
        }
        doc.shapes.retain(|s| !s.path.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{MultiPath, Paint, Shape, SubPath};
    use crate::types::Color;

    fn doc_with_line(a: PointF64, b: PointF64, c: PointF64) -> VectorDoc {
        let mut doc = VectorDoc::new(100, 100);
        let mut sp = SubPath::new();
        sp.commands.push(PathCmd::MoveTo(a));
        sp.commands.push(PathCmd::LineTo(b));
        sp.commands.push(PathCmd::LineTo(c));
        let mut mp = MultiPath::new();
        mp.push(sp);
        doc.shapes.push(Shape {
            paint: Paint::Solid(Color::new(0, 0, 0)),
            path: mp,
        });
        doc
    }

    #[test]
    fn quantize_rounds_points() {
        let mut doc = doc_with_line(
            PointF64::new(1.234, 2.567),
            PointF64::new(3.891, 4.123),
            PointF64::new(5.0, 6.0),
        );
        QuantizePass::new(1).run(&mut doc);
        let cmd = &doc.shapes[0].path.subpaths[0].commands;
        match &cmd[1] {
            PathCmd::LineTo(p) => {
                assert!((p.x - 3.9).abs() < 1e-9);
                assert!((p.y - 4.1).abs() < 1e-9);
            }
            _ => panic!("expected LineTo"),
        }
    }

    #[test]
    fn cleanup_drops_zero_length() {
        let mut doc = VectorDoc::new(100, 100);
        let mut sp = SubPath::new();
        sp.commands
            .push(PathCmd::MoveTo(PointF64::new(0.0, 0.0)));
        sp.commands
            .push(PathCmd::LineTo(PointF64::new(1.0, 1.0)));
        sp.commands
            .push(PathCmd::LineTo(PointF64::new(1.0, 1.0)));
        sp.commands
            .push(PathCmd::LineTo(PointF64::new(2.0, 2.0)));
        let mut mp = MultiPath::new();
        mp.push(sp);
        doc.shapes.push(Shape {
            paint: Paint::Solid(Color::new(0, 0, 0)),
            path: mp,
        });
        CleanupPass.run(&mut doc);
        let cmd = &doc.shapes[0].path.subpaths[0].commands;
        assert_eq!(cmd.len(), 2);
    }

    #[test]
    fn cleanup_drops_collinear() {
        let mut doc = doc_with_line(
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 1.0),
            PointF64::new(2.0, 2.0),
        );
        CleanupPass.run(&mut doc);
        let cmd = &doc.shapes[0].path.subpaths[0].commands;
        // b is collinear between a and c, so it should be dropped.
        assert_eq!(cmd.len(), 2);
        match &cmd[1] {
            PathCmd::LineTo(p) => {
                assert!((p.x - 2.0).abs() < 1e-9);
                assert!((p.y - 2.0).abs() < 1e-9);
            }
            _ => panic!("expected LineTo"),
        }
    }
}
