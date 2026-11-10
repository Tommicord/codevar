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

//! Color fitting: snap segment paints to a palette (fixed, auto-quantized,
//! or identity), then merge adjacent same-paint runs.
//!
//! Faithful port of visioncortex/vtracer's `crates/vtracer/src/colorfit/`.
//!
//! `Identity` leaves paints untouched. `FixedPalette` snaps every layer to
//! the nearest palette color in OKLab. `AutoQuantize` builds a palette by
//! area-weighted median cut over layer colors and snaps to it.
//! `MergeAdjacent` unions consecutive same-paint layers into single layers.

mod merge;
mod oklab;
mod palette;
mod quantize;

pub use merge::MergeAdjacent;
pub use oklab::Oklab;
pub use palette::FixedPalette;
pub use quantize::AutoQuantize;

/// A strategy for assigning a single paint color to each segmentation layer.
pub trait ColorFitter {
    /// Rewrite each layer's paint in place according to this fitter.
    fn fit(&self, seg: &mut Segmentation);
}

/// No-op fitter: leaves every layer's paint untouched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

impl ColorFitter for Identity {
    fn fit(&self, _seg: &mut Segmentation) {
        // nothing to do
    }
}

use crate::ir::Segmentation;

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
    fn identity_leaves_paints_untouched() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(10, 20, 30)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0), (1, 0), (0, 1), (1, 1)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(200, 210, 220)),
            mask: mask_at(2, 0, 2, 2, &[(0, 0), (1, 0), (0, 1), (1, 1)]),
        });

        Identity.fit(&mut seg);

        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(10, 20, 30)));
        assert_eq!(seg.layers[1].paint, Paint::Solid(Color::new(200, 210, 220)));
    }

    #[test]
    fn fixed_palette_snaps_nearest() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(10, 10, 10)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });

        let palette = FixedPalette::new(vec![Color::new(0, 0, 0), Color::new(255, 255, 255)]);
        palette.fit(&mut seg);

        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(0, 0, 0)));
    }

    #[test]
    fn empty_palette_is_noop() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(42, 43, 44)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });

        FixedPalette::new(vec![]).fit(&mut seg);
        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(42, 43, 44)));
    }

    #[test]
    fn merge_adjacent_union_same_paint() {
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

        // Same-paint first two layers merged into one; third untouched.
        assert_eq!(seg.layers.len(), 2);
        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(5, 5, 5)));
        assert_eq!(seg.layers[1].paint, Paint::Solid(Color::new(9, 9, 9)));
    }
}
