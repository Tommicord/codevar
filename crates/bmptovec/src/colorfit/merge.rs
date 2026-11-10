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

//! Merge adjacent same-paint layers into single layers.
//!
//! Port of visioncortex/vtracer's `crates/vtracer/src/colorfit/merge.rs`.
//! Single-pass: flush the current run whenever the paint changes, unioning
//! all masks in the run via `RegionMask::union_all` (NOT pairwise folding).

use crate::colorfit::ColorFitter;
use crate::ir::{Layer, Segmentation};

/// Merge consecutive layers sharing the same paint into a single layer whose
/// mask is the union of the run's masks. Runs are maximal same-paint
/// sequences; layers of a different paint break a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeAdjacent;

impl MergeAdjacent {
    /// Create a new `MergeAdjacent` fitter.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl ColorFitter for MergeAdjacent {
    fn fit(&self, seg: &mut Segmentation) {
        if seg.layers.is_empty() {
            return;
        }

        let mut merged: Vec<Layer> = Vec::with_capacity(seg.layers.len());
        let mut run_start = 0usize;

        while run_start < seg.layers.len() {
            let run_paint = seg.layers[run_start].paint;
            let mut run_end = run_start + 1;
            while run_end < seg.layers.len() && seg.layers[run_end].paint == run_paint {
                run_end += 1;
            }

            let run_masks: Vec<&RegionMask> = seg.layers[run_start..run_end]
                .iter()
                .map(|layer| &layer.mask)
                .collect();
            let merged_mask = RegionMask::union_all(&run_masks);

            merged.push(Layer {
                paint: run_paint,
                mask: merged_mask,
            });

            run_start = run_end;
        }

        seg.layers = merged;
    }
}

use crate::ir::RegionMask;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Layer, Paint, RegionMask};
    use crate::types::{BinaryImage, Color, PointI32};

    fn mask_at(x: i32, y: i32, w: usize, h: usize, foreground: &[(usize, usize)]) -> RegionMask {
        let mut image = BinaryImage::new_w_h(w, h);
        for &(px, py) in foreground {
            image.set_pixel(px, py, true);
        }
        RegionMask::new(image, PointI32::new(x, y))
    }

    #[test]
    fn merge_union_same_paint_run() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(5, 5, 5)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0), (1, 0)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(5, 5, 5)),
            mask: mask_at(2, 0, 2, 2, &[(0, 0), (0, 1)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(9, 9, 9)),
            mask: mask_at(0, 2, 2, 2, &[(0, 0)]),
        });

        MergeAdjacent.fit(&mut seg);

        assert_eq!(seg.layers.len(), 2);
        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(5, 5, 5)));
        assert_eq!(seg.layers[0].mask.area(), 4);
        assert_eq!(seg.layers[1].paint, Paint::Solid(Color::new(9, 9, 9)));
        assert_eq!(seg.layers[1].mask.area(), 1);
    }

    #[test]
    fn alternating_paints_are_not_merged() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(1, 1, 1)),
            mask: mask_at(0, 0, 1, 1, &[(0, 0)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(2, 2, 2)),
            mask: mask_at(0, 0, 1, 1, &[(0, 0)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(1, 1, 1)),
            mask: mask_at(0, 0, 1, 1, &[(0, 0)]),
        });

        MergeAdjacent.fit(&mut seg);

        assert_eq!(seg.layers.len(), 3);
        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(1, 1, 1)));
        assert_eq!(seg.layers[1].paint, Paint::Solid(Color::new(2, 2, 2)));
        assert_eq!(seg.layers[2].paint, Paint::Solid(Color::new(1, 1, 1)));
    }

    #[test]
    fn empty_segmentation_is_noop() {
        let mut seg = Segmentation::new(4, 4);
        MergeAdjacent.fit(&mut seg);
        assert!(seg.layers.is_empty());
    }
}
