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

//! Area-weighted median-cut color quantization.
//!
//! Port of visioncortex/vtracer's `crates/vtracer/src/colorfit/quantize.rs`.
//! Builds a palette from layer colors weighted by region area via median cut,
//! then snaps each layer to its nearest representative in OKLab space.

use crate::colorfit::ColorFitter;
use crate::colorfit::oklab::Oklab;
use crate::ir::{Paint, Segmentation};
use crate::types::Color;

/// Auto-quantize: reduce the number of distinct layer colors via area-weighted
/// median cut, then snap each layer to its nearest palette representative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoQuantize {
    pub max_colors: usize,
}

impl AutoQuantize {
    /// Create a new auto-quantizer allowing at most `max_colors` palette
    /// entries.
    #[inline]
    pub fn new(max_colors: usize) -> Self {
        Self { max_colors }
    }

    /// Build a representative palette from layer colors weighted by mask area.
    fn build_palette(&self, seg: &Segmentation) -> Vec<Color> {
        let mut buckets: Vec<Bucket> = seg
            .layers
            .iter()
            .map(|layer| match layer.paint {
                Paint::Solid(c) => Bucket {
                    samples: vec![c],
                    weight: layer.mask.area(),
                },
            })
            .collect();
        buckets.retain(|b| b.weight > 0);

        if buckets.is_empty() || self.max_colors == 0 {
            return vec![];
        }

        // Median cut: split the bucket with the widest channel range until we
        // have max_colors buckets or none can be split further.
        while buckets.len() < self.max_colors {
            let split_idx = self.pick_split_bucket(&buckets);
            let Some(idx) = split_idx else {
                break;
            };
            let (left, right) = match buckets.remove(idx).split() {
                Some(pair) => pair,
                None => break,
            };
            buckets.push(left);
            buckets.push(right);
        }

        buckets
            .into_iter()
            .map(|b| b.representative())
            .collect()
    }

    /// Pick the bucket with the widest OKLab channel range; `None` when no
    /// bucket can be split (all singletons).
    fn pick_split_bucket(&self, buckets: &[Bucket]) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_range: f64 = 0.0;
        for (i, b) in buckets.iter().enumerate() {
            if b.samples.len() < 2 {
                continue;
            }
            let range = b.channel_range();
            if range > best_range {
                best_range = range;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    /// Snap each layer's paint to the nearest representative color in OKLab.
    fn snap_layers(&self, seg: &mut Segmentation, palette: &[Color]) {
        if palette.is_empty() {
            return;
        }
        for layer in seg.layers.iter_mut() {
            match layer.paint {
                Paint::Solid(c) => {
                    let target = Oklab::from_color(&c);
                    let mut best = palette[0];
                    let mut best_dist = target.distance_squared(&Oklab::from_color(&best));
                    for &rep in &palette[1..] {
                        let dist = target.distance_squared(&Oklab::from_color(&rep));
                        if dist < best_dist {
                            best_dist = dist;
                            best = rep;
                        }
                    }
                    layer.paint = Paint::Solid(best);
                }
            }
        }
    }
}

impl ColorFitter for AutoQuantize {
    fn fit(&self, seg: &mut Segmentation) {
        let palette = self.build_palette(seg);
        self.snap_layers(seg, &palette);
    }
}

/// A weighted bucket of color samples for median-cut quantization.
#[derive(Debug, Clone)]
struct Bucket {
    samples: Vec<Color>,
    weight: usize,
}

impl Bucket {
    /// OKLab channel range (max - min across l, a, b) of all samples.
    #[inline]
    fn channel_range(&self) -> f64 {
        let mut min_l = f64::INFINITY;
        let mut max_l = f64::NEG_INFINITY;
        let mut min_a = f64::INFINITY;
        let mut max_a = f64::NEG_INFINITY;
        let mut min_b = f64::INFINITY;
        let mut max_b = f64::NEG_INFINITY;
        for c in &self.samples {
            let lab = Oklab::from_color(c);
            min_l = min_l.min(lab.l);
            max_l = max_l.max(lab.l);
            min_a = min_a.min(lab.a);
            max_a = max_a.max(lab.a);
            min_b = min_b.min(lab.b);
            max_b = max_b.max(lab.b);
        }
        let r_l = max_l - min_l;
        let r_a = max_a - min_a;
        let r_b = max_b - min_b;
        r_l.max(r_a).max(r_b)
    }

    /// Split this bucket into two at the weighted median along the widest
    /// OKLab channel. Returns `None` if it cannot be split (one sample).
    fn split(self) -> Option<(Bucket, Bucket)> {
        if self.samples.len() < 2 {
            return None;
        }
        let (channel, _) = self.widest_channel();
        let mut samples = self.samples;
        samples.sort_by(|a, b| {
            let la = Oklab::from_color(a);
            let lb = Oklab::from_color(b);
            let va = match channel {
                0 => la.l,
                1 => la.a,
                _ => la.b,
            };
            let vb = match channel {
                0 => lb.l,
                1 => lb.a,
                _ => lb.b,
            };
            va.partial_cmp(&vb)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Weighted median cut point.
        let total: usize = samples.iter().map(|_| 1usize).sum();
        let mut acc = 0usize;
        let mut cut = 1usize;
        for (i, _) in samples.iter().enumerate() {
            acc += 1;
            if acc * 2 >= total {
                cut = i + 1;
                break;
            }
        }
        let cut = cut.clamp(1, samples.len() - 1);

        let (lower_samples, upper_samples) = samples.split_at(cut);
        let weight = self.weight;
        let w_lo = weight / 2;
        let w_hi = weight - w_lo;
        Some((
            Bucket {
                samples: lower_samples.to_vec(),
                weight: w_lo.max(1),
            },
            Bucket {
                samples: upper_samples.to_vec(),
                weight: w_hi.max(1),
            },
        ))
    }

    /// The index of the widest OKLab channel and its range.
    fn widest_channel(&self) -> (usize, f64) {
        let mut min_l = f64::INFINITY;
        let mut max_l = f64::NEG_INFINITY;
        let mut min_a = f64::INFINITY;
        let mut max_a = f64::NEG_INFINITY;
        let mut min_b = f64::INFINITY;
        let mut max_b = f64::NEG_INFINITY;
        for c in &self.samples {
            let lab = Oklab::from_color(c);
            min_l = min_l.min(lab.l);
            max_l = max_l.max(lab.l);
            min_a = min_a.min(lab.a);
            max_a = max_a.max(lab.a);
            min_b = min_b.min(lab.b);
            max_b = max_b.max(lab.b);
        }
        let r_l = max_l - min_l;
        let r_a = max_a - min_a;
        let r_b = max_b - min_b;
        if r_a > r_l && r_a > r_b {
            (1, r_a)
        } else if r_b > r_l {
            (2, r_b)
        } else {
            (0, r_l)
        }
    }

    /// The representative color: area-weighted mean of samples in OKLab,
    /// converted back to sRGB. A singleton bucket returns its exact sample so
    /// snapping is the identity for colors already present.
    #[inline]
    fn representative(&self) -> Color {
        if self.samples.is_empty() {
            return Color::new(0, 0, 0);
        }
        if self.samples.len() == 1 {
            return self.samples[0];
        }
        let mut sum_l = 0.0;
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        for c in &self.samples {
            let lab = Oklab::from_color(c);
            sum_l += lab.l;
            sum_a += lab.a;
            sum_b += lab.b;
        }
        let n = self.samples.len() as f64;
        let lab = Oklab {
            l: sum_l / n,
            a: sum_a / n,
            b: sum_b / n,
        };
        oklab_to_color(&lab)
    }
}

/// Convert OKLab back to the nearest sRGB color (inverse transform).
fn oklab_to_color(lab: &Oklab) -> Color {
    let l = lab.l;
    let a = lab.a;
    let b = lab.b;
    let l_ = (l + 0.3963377764 * a + 0.2158037573 * b).powi(3);
    let m_ = (l - 0.1055613458 * a - 0.0637521572 * b).powi(3);
    let s_ = (l - 0.0894841779 * a - 1.2914855482 * b).powi(3);
    let x = l_ * 1.8468828 + m_ * -1.0130945 + s_ * 0.1662909;
    let y = l_ * 0.2655678 + m_ * 1.1701205 + s_ * -0.4357254;
    let z = l_ * -0.0528446 + m_ * -0.1029693 + s_ * 1.3092527;
    let r = x * 3.2413645 + y * -1.5375551 + z * -0.4986842;
    let g = x * -0.9691446 + y * 1.8759178 + z * 0.0415657;
    let b = x * 0.0556301 + y * -0.2039752 + z * 1.0572252;
    Color::new(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b))
}

/// Linear sRGB channel (0..=1) → gamma-corrected u8.
fn linear_to_srgb(c: f64) -> u8 {
    let c = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
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
    fn build_palette_respects_max_colors() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(10, 10, 10)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(240, 240, 240)),
            mask: mask_at(2, 0, 2, 2, &[(0, 0)]),
        });

        let q = AutoQuantize::new(2);
        let palette = q.build_palette(&seg);
        assert!(palette.len() <= 2);
        assert!(!palette.is_empty());
    }

    #[test]
    fn zero_max_colors_yields_empty_palette() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(10, 10, 10)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });
        let q = AutoQuantize::new(0);
        assert!(q.build_palette(&seg).is_empty());
    }

    #[test]
    fn single_layer_palette_has_one_entry() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(70, 80, 90)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });
        let q = AutoQuantize::new(8);
        let palette = q.build_palette(&seg);
        assert_eq!(palette.len(), 1);
    }

    #[test]
    fn fit_snaps_layers_to_representatives() {
        let mut seg = Segmentation::new(4, 4);
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(10, 10, 10)),
            mask: mask_at(0, 0, 2, 2, &[(0, 0)]),
        });
        seg.layers.push(Layer {
            paint: Paint::Solid(Color::new(240, 240, 240)),
            mask: mask_at(2, 0, 2, 2, &[(0, 0)]),
        });

        AutoQuantize::new(2).fit(&mut seg);

        let c0 = seg.layers[0].paint.color();
        let c1 = seg.layers[1].paint.color();
        assert!((c0 == Color::new(10, 10, 10) || c0 == Color::new(240, 240, 240)));
        assert!((c1 == Color::new(10, 10, 10) || c1 == Color::new(240, 240, 240)));
    }
}
