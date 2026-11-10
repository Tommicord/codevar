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

//! Compositing: turn a [`Segmentation`] into a [`VectorDoc`] by fitting each
//! region's boundary to vector geometry.
//!
//! Two strategies:
//! * [`Compositing::Stacked`] — painter's-algorithm compositing: each layer is
//!   traced independently and stacked bottom-to-top.
//! * [`Compositing::Mosaic`] — seam-free mosaic: shared boundary segments are
//!   fit once and referenced by both adjacent faces (see [`crate::mosaic`]).

use crate::error::Error;
use crate::fitter::CurveFitter;
use crate::ir::{MultiPath, RegionMask, Segmentation, Shape, VectorDoc};
use crate::mosaic::{SegmentFitter, compose_mosaic};
use crate::progress::{CancelToken, Ctx, Phase};
use crate::simplify::CurvePass;

/// How regions are composited into vector shapes.
pub enum Compositing {
    /// Painter's-algorithm stacking: trace each layer independently.
    Stacked(Box<dyn CurveFitter>),
    /// Seam-free mosaic: fit each shared boundary once, reference from both faces.
    Mosaic {
        fitter: Box<dyn SegmentFitter>,
        /// Maximum color difference (in channel units) for merging adjacent
        /// same-paint faces before fitting.
        merge_diff: i32,
    },
}

impl Compositing {
    /// Composite a segmentation into a vector document (no cancellation).
    #[inline]
    pub fn compose(&self, seg: &Segmentation, passes: &[Box<dyn CurvePass>]) -> VectorDoc {
        let token = CancelToken::new();
        let ctx = Ctx::new(&token);
        match self.compose_with(seg, passes, ctx) {
            Ok(doc) => doc,
            // A fresh token is never cancelled, so this branch is unreachable
            // at runtime; it is handled to avoid `.expect()` in production.
            Err(_) => VectorDoc::new(seg.width, seg.height),
        }
    }

    /// Composite a segmentation into a vector document with progress + cancel.
    pub fn compose_with(
        &self,
        seg: &Segmentation,
        passes: &[Box<dyn CurvePass>],
        ctx: Ctx<'_>,
    ) -> Result<VectorDoc, Error> {
        match self {
            Compositing::Stacked(fitter) => compose_stacked_with(fitter.as_ref(), seg, passes, ctx),
            Compositing::Mosaic { fitter, merge_diff } => {
                Ok(compose_mosaic(seg, fitter.as_ref(), *merge_diff, passes))
            }
        }
    }
}

/// Fit one region's mask into a [`MultiPath`]: run the fitter, apply each
/// simplification pass's `.ring()`, then push each result as a closed subpath.
#[inline]
pub fn fit_region(fitter: &dyn CurveFitter, mask: &RegionMask, passes: &[Box<dyn CurvePass>]) -> MultiPath {
    let geoms = fitter.fit_region(mask);
    let mut multipath = MultiPath::new();
    for geom in geoms {
        let mut geom = geom;
        for pass in passes {
            geom = pass.ring(geom);
        }
        multipath.push(geom.into_closed_subpath());
    }
    multipath
}

/// Compose stacked with per-layer progress and cancellation.
fn compose_stacked_with(
    fitter: &dyn CurveFitter,
    seg: &Segmentation,
    passes: &[Box<dyn CurvePass>],
    ctx: Ctx<'_>,
) -> Result<VectorDoc, Error> {
    let mut doc = VectorDoc::new(seg.width, seg.height);
    let total = seg.layers.len().max(1);
    for (i, layer) in seg.layers.iter().enumerate() {
        ctx.check()?;
        let path = fit_region(fitter, &layer.mask, passes);
        doc.shapes.push(Shape {
            paint: layer.paint,
            path,
        });
        let percent = ((i + 1) * 100 / total) as u8;
        ctx.report(Phase::Compose, percent as f32);
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::PolygonFitter;
    use crate::ir::{Layer, Paint};
    use crate::types::{BinaryImage, Color, PointI32};

    fn mask(w: usize, h: usize, on: &[(usize, usize)]) -> RegionMask {
        let mut image = BinaryImage::new_w_h(w, h);
        for &(x, y) in on {
            image.set_pixel(x, y, true);
        }
        RegionMask::new(image, PointI32::new(0, 0))
    }

    #[test]
    fn fit_region_produces_closed_subpaths() {
        let fitter = PolygonFitter;
        let mask = mask(4, 4, &[(1, 1), (2, 1), (1, 2), (2, 2)]);
        let passes: &[Box<dyn CurvePass>] = &[];
        let mp = fit_region(&fitter, &mask, passes);
        assert!(!mp.is_empty());
        for sp in &mp.subpaths {
            assert!(sp.start().is_some());
        }
    }

    #[test]
    fn compose_stacked_preserves_paint_order() {
        let fitter = PolygonFitter;
        let seg = Segmentation {
            width: 8,
            height: 8,
            layers: vec![
                Layer {
                    paint: Paint::Solid(Color::new(10, 0, 0)),
                    mask: mask(4, 4, &[(0, 0), (1, 0), (0, 1), (1, 1)]),
                },
                Layer {
                    paint: Paint::Solid(Color::new(0, 10, 0)),
                    mask: mask(4, 4, &[(2, 2), (3, 2), (2, 3), (3, 3)]),
                },
            ],
        };
        let doc = Compositing::Stacked(Box::new(fitter)).compose(&seg, &[]);
        assert_eq!(doc.shapes.len(), 2);
        assert_eq!(doc.shapes[0].paint.color(), Color::new(10, 0, 0));
        assert_eq!(doc.shapes[1].paint.color(), Color::new(0, 10, 0));
    }

    #[test]
    fn compose_with_honors_cancellation() {
        let fitter = PolygonFitter;
        let seg = Segmentation {
            width: 4,
            height: 4,
            layers: vec![Layer {
                paint: Paint::Solid(Color::new(5, 5, 5)),
                mask: mask(4, 4, &[(1, 1)]),
            }],
        };
        let token = CancelToken::new();
        let ctx = Ctx::new(&token);
        let _ = Compositing::Stacked(Box::new(fitter)).compose_with(&seg, &[], ctx);
        token.cancel();
        let ctx = Ctx::new(&token);
        let res = Compositing::Stacked(Box::new(fitter)).compose_with(&seg, &[], ctx);
        assert!(res.is_err());
    }
}
