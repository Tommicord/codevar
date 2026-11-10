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

//! Watershed frontend: hierarchical region-growing over the pixel graph,
//! cut at a `detail` level. Port of vtracer's `frontend/watershed.rs`.

use crate::error::Error;
use crate::ir::{Layer, Paint, RegionMask, Segmentation};
use crate::types::{BinaryImage, Color, ColorImage, PointI32};

/// Result of the hierarchical watershed: one region per level.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WatershedHierarchy {
    pub levels: Vec<RegionMask>,
}

impl WatershedHierarchy {
    #[inline]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

/// Squared color distance between two colors.
#[inline]
fn color_dist2(a: Color, b: Color) -> u32 {
    let dr = a.r as i32 - b.r as i32;
    let dg = a.g as i32 - b.g as i32;
    let db = a.b as i32 - b.b as i32;
    (dr * dr + dg * dg + db * db) as u32
}

/// Region-grow a single watershed basin from `seed` using a flood fill that
/// accepts pixels within `threshold` of the basin mean, 4-connected.
fn grow_basin(
    img: &ColorImage,
    seed: (usize, usize),
    threshold: u32,
    taken: &mut BinaryImage,
) -> BinaryImage {
    let w = img.width;
    let h = img.height;
    let mut mask = BinaryImage::new_w_h(w, h);

    let mut stack: Vec<(usize, usize)> = vec![seed];
    let mut sum_r: u64 = 0;
    let mut sum_g: u64 = 0;
    let mut sum_b: u64 = 0;
    let mut count: u64 = 0;

    while let Some((cx, cy)) = stack.pop() {
        if taken.get_pixel(cx, cy) || mask.get_pixel(cx, cy) {
            continue;
        }
        let c = img.get_pixel(cx, cy);
        // Accept if within threshold of the current running mean (or seed for
        // the first pixel).
        let accept = if count == 0 {
            true
        } else {
            let mean = Color::new(
                (sum_r / count) as u8,
                (sum_g / count) as u8,
                (sum_b / count) as u8,
            );
            color_dist2(c, mean) <= threshold
        };
        if !accept {
            continue;
        }

        taken.set_pixel(cx, cy, true);
        mask.set_pixel(cx, cy, true);
        sum_r += c.r as u64;
        sum_g += c.g as u64;
        sum_b += c.b as u64;
        count += 1;

        if cx > 0 {
            stack.push((cx - 1, cy));
        }
        if cx + 1 < w {
            stack.push((cx + 1, cy));
        }
        if cy > 0 {
            stack.push((cx, cy - 1));
        }
        if cy + 1 < h {
            stack.push((cx, cy + 1));
        }
    }

    mask
}

/// Compute a bounding box cropped mask for `mask` restricted to non-taken
/// pixels, offset to the box origin.
fn crop_mask(mask: &BinaryImage) -> (BinaryImage, PointI32) {
    let w = mask.width;
    let h = mask.height;
    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y) {
                any = true;
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if !any {
        return (BinaryImage::new_w_h(0, 0), PointI32::new(0, 0));
    }
    let bw = max_x - min_x + 1;
    let bh = max_y - min_y + 1;
    let mut out = BinaryImage::new_w_h(bw, bh);
    for yy in 0..bh {
        for xx in 0..bw {
            out.set_pixel(xx, yy, mask.get_pixel(min_x + xx, min_y + yy));
        }
    }
    (out, PointI32::new(min_x as i32, min_y as i32))
}

/// Hierarchical region-growing watershed.
///
/// Seeds are spread across the image at a spacing determined by `detail`;
/// each basin grows until it hits a taken pixel or exceeds the color
/// threshold. Higher `detail` means finer (more, smaller) regions.
pub struct WatershedFrontend {
    pub detail: u32,
    pub min_area: usize,
}

impl Default for WatershedFrontend {
    fn default() -> Self {
        Self {
            detail: 8,
            min_area: 20,
        }
    }
}

impl WatershedFrontend {
    /// Build a hierarchy of region masks, one per watershed level.
    pub fn hierarchy(&self, img: &ColorImage) -> Result<WatershedHierarchy, Error> {
        if img.width == 0 || img.height == 0 {
            return Err(Error::EmptyImage);
        }

        let mut taken = BinaryImage::new_w_h(img.width, img.height);
        let mut levels: Vec<RegionMask> = Vec::new();

        let spacing = self.detail.max(1);
        let threshold = 600u32; // squared color distance threshold

        // Multiple passes with decreasing seed spacing to form a hierarchy.
        for pass in 0..3usize {
            let step = spacing.saturating_sub(pass as u32).max(1);
            for y in (0..img.height).step_by(step as usize) {
                for x in (0..img.width).step_by(step as usize) {
                    if taken.get_pixel(x, y) {
                        continue;
                    }
                    let mask = grow_basin(img, (x, y), threshold, &mut taken);
                    let (crop, offset) = crop_mask(&mask);
                    if crop.width == 0 || crop.height == 0 {
                        continue;
                    }
                    let area: usize = (0..crop.height)
                        .map(|yy| {
                            (0..crop.width)
                                .filter(|&xx| crop.get_pixel(xx, yy))
                                .count()
                        })
                        .sum();
                    if area < self.min_area {
                        continue;
                    }
                    levels.push(RegionMask::new(crop, offset));
                }
            }
        }

        Ok(WatershedHierarchy { levels })
    }

    /// Segment into flat paint layers, one per hierarchy level, painted with
    /// the region's mean color, bottom-to-top.
    pub fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        if img.width == 0 || img.height == 0 {
            return Err(Error::EmptyImage);
        }

        let hierarchy = self.hierarchy(img)?;
        let mut seg = Segmentation::new(img.width as u32, img.height as u32);

        // Bottom-to-top: iterate levels in order; earlier levels are lower.
        for region in hierarchy.levels {
            let mean = region_mean_color(img, &region);
            let layer = Layer {
                paint: Paint::Solid(mean),
                mask: region,
            };
            seg.layers.push(layer);
        }

        Ok(seg)
    }
}

/// Mean color of the pixels covered by `region` (placed at its offset).
fn region_mean_color(img: &ColorImage, region: &RegionMask) -> Color {
    let mut r: u64 = 0;
    let mut g: u64 = 0;
    let mut b: u64 = 0;
    let mut count: u64 = 0;
    let ox = region.offset.x.max(0) as usize;
    let oy = region.offset.y.max(0) as usize;
    for yy in 0..region.image.height {
        for xx in 0..region.image.width {
            if region.image.get_pixel(xx, yy) {
                let x = ox + xx;
                let y = oy + yy;
                if x < img.width && y < img.height {
                    let c = img.get_pixel(x, y);
                    r += c.r as u64;
                    g += c.g as u64;
                    b += c.b as u64;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return Color::new(0, 0, 0);
    }
    Color::new((r / count) as u8, (g / count) as u8, (b / count) as u8)
}

impl crate::frontend::Frontend for WatershedFrontend {
    fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        WatershedFrontend::segment(self, img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_image_errors() {
        let img = ColorImage::new(0, 0, vec![]);
        let f = WatershedFrontend::default();
        assert!(matches!(f.segment(&img), Err(Error::EmptyImage)));
    }

    #[test]
    fn hierarchy_produces_regions() {
        let img = ColorImage::new(
            4,
            4,
            vec![
                Color::new(0, 0, 0),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
            ],
        );
        let f = WatershedFrontend {
            detail: 2,
            min_area: 1,
        };
        let hier = f.hierarchy(&img).unwrap();
        assert!(!hier.is_empty());
    }

    #[test]
    fn segment_layers_bottom_to_top() {
        let img = ColorImage::new(
            2,
            2,
            vec![
                Color::new(10, 10, 10),
                Color::new(12, 12, 12),
                Color::new(200, 200, 200),
                Color::new(202, 202, 202),
            ],
        );
        let f = WatershedFrontend {
            detail: 1,
            min_area: 1,
        };
        let seg = f.segment(&img).unwrap();
        assert!(!seg.layers.is_empty());
        assert_eq!(seg.width, 2);
        assert_eq!(seg.height, 2);
    }

    #[test]
    fn color_dist2_symmetric() {
        let a = Color::new(10, 20, 30);
        let b = Color::new(40, 50, 60);
        assert_eq!(color_dist2(a, b), color_dist2(b, a));
    }
}
