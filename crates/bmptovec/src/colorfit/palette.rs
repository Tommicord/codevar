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

//! Fixed palette color fitting: snap each layer's paint to the nearest
//! palette color in OKLab space.
//!
//! Port of visioncortex/vtracer's `crates/vtracer/src/colorfit/palette.rs`.

use crate::colorfit::ColorFitter;
use crate::colorfit::oklab::Oklab;
use crate::ir::{Paint, Segmentation};
use crate::types::Color;

/// A fixed set of palette colors. Every layer is snapped to the nearest
/// color in OKLab space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedPalette {
    pub colors: Vec<Color>,
}

impl FixedPalette {
    /// Create a new fixed palette from the given colors.
    #[inline]
    pub fn new(colors: Vec<Color>) -> Self {
        Self { colors }
    }

    /// Snap a single color to the nearest palette color in OKLab space.
    /// Returns the original color when the palette is empty.
    #[inline]
    pub fn snap(&self, color: &Color) -> Color {
        if self.colors.is_empty() {
            return *color;
        }
        let target = Oklab::from_color(color);
        let mut best = self.colors[0];
        let mut best_dist = target.distance_squared(&Oklab::from_color(&best));
        for &c in &self.colors[1..] {
            let dist = target.distance_squared(&Oklab::from_color(&c));
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        best
    }
}

impl ColorFitter for FixedPalette {
    fn fit(&self, seg: &mut Segmentation) {
        for layer in seg.layers.iter_mut() {
            match layer.paint {
                Paint::Solid(c) => {
                    layer.paint = Paint::Solid(self.snap(&c));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Layer, RegionMask};
    use crate::types::{BinaryImage, Color, PointI32};

    fn mask_at(x: i32, y: i32, w: usize, h: usize, foreground: &[(usize, usize)]) -> RegionMask {
        let mut image = BinaryImage::new_w_h(w, h);
        for &(px, py) in foreground {
            image.set_pixel(px, py, true);
        }
        RegionMask::new(image, PointI32::new(x, y))
    }

    #[test]
    fn snap_nearest_palette_color() {
        let palette = FixedPalette::new(vec![Color::new(0, 0, 0), Color::new(255, 255, 255)]);
        // dark red is far nearer black than white in OKLab
        assert_eq!(palette.snap(&Color::new(10, 0, 0)), Color::new(0, 0, 0));
        assert_eq!(
            palette.snap(&Color::new(245, 245, 245)),
            Color::new(255, 255, 255)
        );
    }

    #[test]
    fn empty_palette_returns_original() {
        let palette = FixedPalette::new(vec![]);
        let c = Color::new(42, 43, 44);
        assert_eq!(palette.snap(&c), c);
    }

    #[test]
    fn fit_rewrites_all_layers() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(10, 10, 10)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(240, 240, 240)),
            mask: mask_at(2, 0, 2, 2, &[(0, 0)]),
        });

        let palette = FixedPalette::new(vec![Color::new(0, 0, 0), Color::new(255, 255, 255)]);
        palette.fit(&mut seg);

        assert_eq!(seg.layers[0].paint, Paint::Solid(Color::new(0, 0, 0)));
        assert_eq!(seg.layers[1].paint, Paint::Solid(Color::new(255, 255, 255)));
    }
}
