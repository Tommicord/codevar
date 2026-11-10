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

//! Binary frontend: threshold an image, label connected components, and emit
//! one black paint layer per cluster. Port of vtracer's `binary` stage.

use crate::error::Error;
use crate::ir::{Layer, RegionMask, Segmentation};
use crate::types::{BinaryImage, Color, ColorImage, SummedAreaTable};

/// Thresholding strategy for the binary frontend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// A fixed intensity cutoff: pixels darker than this are foreground.
    Fixed(u8),
    /// Bradley–Roth adaptive thresholding over a sliding window.
    Adaptive { window: u32, t: f64 },
}

impl Threshold {
    /// Adaptive thresholding with a default window and factor.
    #[inline]
    pub fn adaptive() -> Self {
        Threshold::Adaptive { window: 15, t: 70.0 }
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Threshold::Fixed(128)
    }
}

/// Intensity of a color, matching [`Color::intensity`] but operating on the
/// raw channels so callers can avoid constructing a `Color`.
#[inline]
pub fn intensity(c: Color) -> u32 {
    (c.r as u32 + c.g as u32 + c.b as u32) / 3
}

/// Binarize `img` using `threshold`, returning a foreground mask.
pub fn binarize(img: &ColorImage, threshold: Threshold) -> BinaryImage {
    match threshold {
        Threshold::Fixed(cutoff) => {
            let mut out = BinaryImage::new_w_h(img.width, img.height);
            for y in 0..img.height {
                for x in 0..img.width {
                    let dark = intensity(img.get_pixel(x, y)) <= cutoff as u32;
                    out.set_pixel(x, y, dark);
                }
            }
            out
        }
        Threshold::Adaptive { window, t } => adaptive_bradley_roth(img, window, t),
    }
}

/// Bradley–Roth adaptive thresholding.
///
/// A pixel is foreground when `value * count <= sum * factor`, where
/// `sum`/`count` are the summed-area-table sum and pixel count of the window
/// around the pixel, and `factor = 1 - t / 100`.
pub fn adaptive_bradley_roth(img: &ColorImage, window: u32, t: f64) -> BinaryImage {
    let w = img.width;
    let h = img.height;
    let side = window as usize;
    let sat = SummedAreaTable::from_color_image(img);
    let factor = 1.0 - t / 100.0;
    let mut out = BinaryImage::new_w_h(w, h);

    let half = side / 2;
    for y in 0..h {
        let y0 = y.saturating_sub(half);
        let y1 = (y + half).min(h.saturating_sub(1));
        for x in 0..w {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half).min(w.saturating_sub(1));
            let rw = x1.saturating_sub(x0) + 1;
            let rh = y1.saturating_sub(y0) + 1;
            let sum = sat.get_region_sum_x_y_w_h(x0, y0, rw, rh);
            let count = (rw * rh) as u32;
            let value = intensity(img.get_pixel(x, y));
            let dark = (value as u64) * (count as u64) <= (sum as u64) * (factor as u64);
            out.set_pixel(x, y, dark);
        }
    }
    out
}

/// A connected component found by the binary labeling pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    /// Bounding box of the component, in full-canvas coordinates.
    pub rect: (usize, usize, usize, usize),
    /// Number of foreground pixels in the component.
    pub size: usize,
}

impl Cluster {
    /// Width of the bounding box.
    #[inline]
    pub fn width(&self) -> usize {
        self.rect.2
    }

    /// Height of the bounding box.
    #[inline]
    pub fn height(&self) -> usize {
        self.rect.3
    }

    /// Crop a foreground mask to the component's bounding box.
    pub fn to_binary_image(&self, src: &BinaryImage) -> BinaryImage {
        let (x0, y0, w, h) = self.rect;
        let mut out = BinaryImage::new_w_h(w, h);
        for dy in 0..h {
            for dx in 0..w {
                let v = src.get_pixel(x0 + dx, y0 + dy);
                out.set_pixel(dx, dy, v);
            }
        }
        out
    }
}

/// Connected components labeled from a binary image.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Clusters {
    pub components: Vec<Cluster>,
}

impl Clusters {
    /// Number of components.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

/// Connected-component labeling over a binary image.
///
/// Uses a two-pass union-find labeling with 4- or 8-connectivity, then
/// collapses each label into a [`Cluster`] with a bounding box and size.
pub fn to_clusters(bin: &BinaryImage, diagonal: bool) -> Clusters {
    let w = bin.width;
    let h = bin.height;
    let n = w * h;
    if n == 0 {
        return Clusters::default();
    }

    let mut labels: Vec<Option<usize>> = vec![None; n];
    let mut parent: Vec<usize> = Vec::new();

    let mut next_label: usize = 0;

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    // First pass: assign provisional labels and union equivalences.
    for y in 0..h {
        for x in 0..w {
            if !bin.get_pixel(x, y) {
                continue;
            }
            let idx = y * w + x;
            let mut neighbors: Vec<usize> = Vec::new();
            // 4-connectivity neighbors already scanned: top and left.
            if y > 0 && bin.get_pixel(x, y - 1) {
                neighbors.push((y - 1) * w + x);
            }
            if x > 0 && bin.get_pixel(x - 1, y) {
                neighbors.push(y * w + (x - 1));
            }
            if diagonal {
                // 8-connectivity: top-left and top-right.
                if y > 0 && x > 0 && bin.get_pixel(x - 1, y - 1) {
                    neighbors.push((y - 1) * w + (x - 1));
                }
                if y > 0 && x + 1 < w && bin.get_pixel(x + 1, y - 1) {
                    neighbors.push((y - 1) * w + (x + 1));
                }
            }

            if neighbors.is_empty() {
                labels[idx] = Some(next_label);
                parent.push(next_label);
                next_label += 1;
            } else {
                let mut min_label: Option<usize> = None;
                for &nb in &neighbors {
                    if let Some(l) = labels[nb] {
                        min_label = Some(min_label.map_or(l, |m| m.min(l)));
                    }
                }
                let min_label = min_label.unwrap_or(0);
                labels[idx] = Some(min_label);
                // Union all neighboring labels with the chosen minimum label.
                for &nb in &neighbors {
                    if let Some(l) = labels[nb] {
                        let a = find(&mut parent, min_label);
                        let b = find(&mut parent, l);
                        if a != b {
                            parent[a] = b;
                        }
                    }
                }
            }
        }
    }

    // Compact labels to root IDs.
    let mut root_of: Vec<Option<usize>> = vec![None; next_label];
    let mut clusters: Vec<Cluster> = Vec::new();

    // Initialize cluster bounds for each root.
    for y in 0..h {
        for x in 0..w {
            if !bin.get_pixel(x, y) {
                continue;
            }
            let idx = y * w + x;
            let lbl = labels[idx].unwrap_or(0);
            let root = find(&mut parent, lbl);
            let slot = match root_of[root] {
                Some(s) => s,
                None => {
                    let s = clusters.len();
                    root_of[root] = Some(s);
                    clusters.push(Cluster {
                        rect: (x, y, 1, 1),
                        size: 0,
                    });
                    s
                }
            };
            let c = &mut clusters[slot];
            c.size += 1;
            if x < c.rect.0 {
                c.rect.0 = x;
            }
            if y < c.rect.1 {
                c.rect.1 = y;
            }
            let x2 = x + 1;
            let y2 = y + 1;
            if x2 > c.rect.0 + c.rect.2 {
                c.rect.2 = x2 - c.rect.0;
            }
            if y2 > c.rect.1 + c.rect.3 {
                c.rect.3 = y2 - c.rect.1;
            }
        }
    }

    Clusters { components: clusters }
}

/// Filter clusters by minimum area, keeping those whose size >= min_area.
pub fn filter_by_area(clusters: Clusters, min_area: usize) -> Clusters {
    Clusters {
        components: clusters
            .components
            .into_iter()
            .filter(|c| c.size >= min_area)
            .collect(),
    }
}

/// Binary frontend: threshold and emit one black layer per connected component.
pub struct BinaryFrontend {
    pub threshold: Threshold,
    pub diagonal: bool,
    pub min_area: usize,
}

impl Default for BinaryFrontend {
    fn default() -> Self {
        Self {
            threshold: Threshold::default(),
            diagonal: false,
            min_area: 10,
        }
    }
}

impl BinaryFrontend {
    /// Build a segmentation from a binarized image.
    pub fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        if img.width == 0 || img.height == 0 {
            return Err(Error::EmptyImage);
        }

        let bin = binarize(img, self.threshold);
        let clusters = to_clusters(&bin, self.diagonal);
        let clusters = filter_by_area(clusters, self.min_area);

        let mut segmentation = Segmentation::new(img.width as u32, img.height as u32);
        for cluster in clusters.components {
            let mask = cluster.to_binary_image(&bin);
            let offset = crate::types::PointI32::new(cluster.rect.0 as i32, cluster.rect.1 as i32);
            let region_mask = RegionMask::new(mask, offset);
            let layer = Layer {
                paint: crate::ir::Paint::Solid(Color::new(0, 0, 0)),
                mask: region_mask,
            };
            segmentation.layers.push(layer);
        }

        Ok(segmentation)
    }
}

impl crate::frontend::Frontend for BinaryFrontend {
    fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        BinaryFrontend::segment(self, img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_black(w: usize, h: usize) -> ColorImage {
        let pixels = vec![Color::new(0, 0, 0); w * h];
        ColorImage::new(w, h, pixels)
    }

    #[test]
    fn empty_image_errors() {
        let img = ColorImage::new(0, 0, vec![]);
        let f = BinaryFrontend::default();
        assert!(matches!(f.segment(&img), Err(Error::EmptyImage)));
    }

    #[test]
    fn fixed_threshold_marks_dark_pixels() {
        let img = ColorImage::new(
            2,
            2,
            vec![
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(60, 60, 60),
            ],
        );
        let f = BinaryFrontend {
            threshold: Threshold::Fixed(128),
            diagonal: false,
            min_area: 1,
        };
        let seg = f.segment(&img).unwrap();
        assert_eq!(seg.layers.len(), 2);
    }

    #[test]
    fn adaptive_bradley_roth_runs() {
        let img = ColorImage::new(
            4,
            4,
            vec![
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
            ],
        );
        let bin = adaptive_bradley_roth(&img, 3, 70.0);
        assert_eq!(bin.width, 4);
        assert_eq!(bin.height, 4);
    }

    #[test]
    fn min_area_filters_small_clusters() {
        let img = solid_black(3, 3);
        let f = BinaryFrontend {
            threshold: Threshold::Fixed(128),
            diagonal: false,
            min_area: 100,
        };
        let seg = f.segment(&img).unwrap();
        assert!(seg.layers.is_empty());
    }
}
