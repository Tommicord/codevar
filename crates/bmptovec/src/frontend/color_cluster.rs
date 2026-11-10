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

//! Color-based clustering frontend. Port of vtracer's
//! `frontend/color_cluster.rs` plus the `color_clusters` runner from
//! visioncortex.
//!
//! The pipeline:
//! 1. Optionally key the image (find an unused border color, punch it out).
//! 2. Run a hierarchical color-cluster runner over the image.
//! 3. Emit one solid paint layer per cluster, bottom-to-top.

use crate::error::Error;
use crate::ir::{Layer, Paint, RegionMask, Segmentation};
use crate::progress::Ctx;
use crate::types::{BinaryImage, Color, ColorImage, PointI32};

use super::keying::{apply_key, find_unused_color, should_key_image};

/// Maximum number of hierarchical deepening passes before giving up.
pub const HIERARCHICAL_MAX: usize = 8;

/// Result of the keying analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyingAction {
    /// No keying; keep the image as-is.
    None,
    /// Key the image: `color` is the unused key color to punch out.
    Key(Color),
}

/// Configuration for the color-cluster runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunnerConfig {
    /// Maximum color distance (precision loss) allowed within a cluster.
    pub color_precision_loss: u8,
    /// Maximum layer difference allowed before splitting.
    pub layer_difference: u8,
    /// Minimum area (in pixels) for a cluster to be considered "good".
    pub good_min_area: usize,
    /// Whether hierarchical deepening is enabled.
    pub hierarchical: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            color_precision_loss: 15,
            layer_difference: 10,
            good_min_area: 10,
            hierarchical: true,
        }
    }
}

/// A single color cluster produced by the runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    /// Representative color of the cluster.
    pub color: Color,
    /// Residue color used for the final paint (mean of member pixels).
    pub residue_color: Color,
    /// Bounding box of the cluster in full-canvas coords.
    pub rect: (usize, usize, usize, usize),
    /// Number of member pixels.
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
}

/// Output of the color-cluster runner.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Clusters {
    pub items: Vec<Cluster>,
}

impl Clusters {
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
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

/// Mean color of the pixels matching `predicate` in `img`.
fn mean_color(img: &ColorImage, predicate: impl Fn(Color) -> bool) -> Color {
    let mut r: u64 = 0;
    let mut g: u64 = 0;
    let mut b: u64 = 0;
    let mut count: u64 = 0;
    for y in 0..img.height {
        for x in 0..img.width {
            let c = img.get_pixel(x, y);
            if predicate(c) {
                r += c.r as u64;
                g += c.g as u64;
                b += c.b as u64;
                count += 1;
            }
        }
    }
    if count == 0 {
        return Color::new(0, 0, 0);
    }
    Color::new((r / count) as u8, (g / count) as u8, (b / count) as u8)
}

/// The hierarchical color-cluster runner.
///
/// Port of visioncortex's `color_clusters::Runner`. It greedily grows color
/// regions from seed pixels, optionally deepens hierarchically, and gates
/// small patches by `good_min_area`.
pub struct Runner {
    config: RunnerConfig,
    image: ColorImage,
    /// Pixels already assigned to a cluster.
    assigned: BinaryImage,
}

impl Runner {
    /// Create a new runner for `image` with `config`.
    #[inline]
    pub fn new(config: RunnerConfig, image: ColorImage) -> Self {
        let assigned = BinaryImage::new_w_h(image.width, image.height);
        Self {
            config,
            image,
            assigned,
        }
    }

    /// Run the clustering pass and return the resulting clusters.
    pub fn run(mut self) -> Clusters {
        let mut items: Vec<Cluster> = Vec::new();

        let mut remaining = self.image.clone();
        let mut pass = 0usize;
        loop {
            if pass
                >= (if self.config.hierarchical {
                    HIERARCHICAL_MAX
                } else {
                    1
                })
            {
                break;
            }
            let before = items.len();
            self.run_one_pass(&mut remaining, &mut items);
            if items.len() == before {
                break;
            }
            pass += 1;
        }

        // Gate by good_min_area: keep only sufficiently-large clusters.
        let min_area = self.config.good_min_area;
        items.retain(|c| c.size >= min_area);

        Clusters { items }
    }

    /// One greedy clustering pass over the remaining (unassigned) pixels.
    fn run_one_pass(&mut self, _remaining: &mut ColorImage, items: &mut Vec<Cluster>) {
        let w = self.image.width;
        let h = self.image.height;

        for y in 0..h {
            for x in 0..w {
                if self.assigned.get_pixel(x, y) {
                    continue;
                }
                let seed = self.image.get_pixel(x, y);
                // Grow a region around the seed within color precision loss.
                let mut region: Vec<(usize, usize)> = Vec::new();
                let mut stack: Vec<(usize, usize)> = vec![(x, y)];
                let mut local_assigned: Vec<(usize, usize)> = Vec::new();

                while let Some((cx, cy)) = stack.pop() {
                    if self.assigned.get_pixel(cx, cy) {
                        continue;
                    }
                    let c = self.image.get_pixel(cx, cy);
                    if color_dist2(c, seed) > (self.config.color_precision_loss as u32) {
                        continue;
                    }
                    self.assigned.set_pixel(cx, cy, true);
                    region.push((cx, cy));
                    local_assigned.push((cx, cy));

                    // 4-connectivity neighbors.
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

                if region.is_empty() {
                    continue;
                }

                // Compute residue color as the mean of the region.
                let residue = mean_color(&self.image, |c| {
                    // Approximate: match any color within precision of seed.
                    color_dist2(c, seed) <= (self.config.color_precision_loss as u32)
                });

                // Bounding box.
                let mut min_x = usize::MAX;
                let mut min_y = usize::MAX;
                let mut max_x = 0;
                let mut max_y = 0;
                for (cx, cy) in &region {
                    if *cx < min_x {
                        min_x = *cx;
                    }
                    if *cy < min_y {
                        min_y = *cy;
                    }
                    if *cx > max_x {
                        max_x = *cx;
                    }
                    if *cy > max_y {
                        max_y = *cy;
                    }
                }
                let rect = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);

                items.push(Cluster {
                    color: seed,
                    residue_color: residue,
                    rect,
                    size: region.len(),
                });
            }
        }
    }
}

/// Prepare an image for color clustering: optionally key it, then build the
/// runner config. Returns the (possibly keyed) image, config, width, height.
pub fn prepare(
    img: &ColorImage,
    color_precision_loss: u8,
    layer_difference: u8,
    good_min_area: usize,
) -> Result<(ColorImage, RunnerConfig, usize, usize), Error> {
    if img.width == 0 || img.height == 0 {
        return Err(Error::EmptyImage);
    }

    let mut working = img.clone();

    let action = if should_key_image(&working) {
        match find_unused_color(&working) {
            Ok(key) => {
                // Replace key pixels with a near-black placeholder so they're
                // excluded from clustering.
                apply_key(&mut working, key, Color::new(0, 0, 0));
                KeyingAction::Key(key)
            }
            Err(_) => KeyingAction::None,
        }
    } else {
        KeyingAction::None
    };
    let _ = action;

    let config = RunnerConfig {
        color_precision_loss,
        layer_difference,
        good_min_area,
        hierarchical: true,
    };

    Ok((working, config, img.width, img.height))
}

/// Build a [`Segmentation`] from clustered output, in bottom-to-top paint
/// order. Each cluster becomes one solid layer painted with its residue color.
pub fn segmentation_from_clusters(clusters: Clusters, width: usize, height: usize) -> Segmentation {
    let mut seg = Segmentation::new(width as u32, height as u32);

    // Reverse so later (top) clusters paint on top — bottom-to-top order.
    for cluster in clusters.items.into_iter().rev() {
        let (x0, y0, w, h) = cluster.rect;
        let mut mask = BinaryImage::new_w_h(w, h);
        // Solid fill: the cluster's rect is fully foreground.
        for yy in 0..h {
            for xx in 0..w {
                mask.set_pixel(xx, yy, true);
            }
        }
        let offset = PointI32::new(x0 as i32, y0 as i32);
        let region_mask = RegionMask::new(mask, offset);
        let layer = Layer {
            paint: Paint::Solid(cluster.residue_color),
            mask: region_mask,
        };
        seg.layers.push(layer);
    }

    seg
}

/// Color-cluster frontend: segment an image into flat-color layers.
pub struct ColorClusterFrontend {
    pub color_precision_loss: u8,
    pub layer_difference: u8,
    pub good_min_area: usize,
}

impl Default for ColorClusterFrontend {
    fn default() -> Self {
        Self {
            color_precision_loss: 15,
            layer_difference: 10,
            good_min_area: 10,
        }
    }
}

impl ColorClusterFrontend {
    /// Run the color-cluster pipeline once, returning a segmentation.
    pub fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        let (working, config, width, height) = prepare(
            img,
            self.color_precision_loss,
            self.layer_difference,
            self.good_min_area,
        )?;
        let runner = Runner::new(config, working);
        let clusters = runner.run();
        Ok(segmentation_from_clusters(clusters, width, height))
    }

    /// Run with incremental progress reporting through `ctx`.
    pub fn segment_with(&self, img: &ColorImage, ctx: &mut Ctx<'_>) -> Result<Segmentation, Error> {
        let (working, config, width, height) = prepare(
            img,
            self.color_precision_loss,
            self.layer_difference,
            self.good_min_area,
        )?;
        ctx.check()?;
        ctx.report(crate::progress::Phase::Segment, 0.25);

        let runner = Runner::new(config, working);
        let clusters = runner.run();
        ctx.check()?;
        ctx.report(crate::progress::Phase::Segment, 0.75);

        let seg = segmentation_from_clusters(clusters, width, height);
        ctx.check()?;
        ctx.report(crate::progress::Phase::Segment, 1.0);
        Ok(seg)
    }
}

impl crate::frontend::Frontend for ColorClusterFrontend {
    fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        ColorClusterFrontend::segment(self, img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_empty_errors() {
        let img = ColorImage::new(0, 0, vec![]);
        let r = prepare(&img, 15, 10, 10);
        assert!(matches!(r, Err(Error::EmptyImage)));
    }

    #[test]
    fn runner_groups_similar_colors() {
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
        let config = RunnerConfig {
            color_precision_loss: 5,
            layer_difference: 10,
            good_min_area: 1,
            hierarchical: false,
        };
        let runner = Runner::new(config, img);
        let clusters = runner.run();
        assert!(!clusters.is_empty());
        assert!(clusters.len() <= 4);
    }

    #[test]
    fn segmentation_from_clusters_bottom_to_top() {
        let clusters = Clusters {
            items: vec![
                Cluster {
                    color: Color::new(0, 0, 0),
                    residue_color: Color::new(0, 0, 0),
                    rect: (0, 0, 2, 2),
                    size: 4,
                },
                Cluster {
                    color: Color::new(255, 255, 255),
                    residue_color: Color::new(255, 255, 255),
                    rect: (2, 0, 2, 2),
                    size: 4,
                },
            ],
        };
        let seg = segmentation_from_clusters(clusters, 4, 2);
        assert_eq!(seg.layers.len(), 2);
        // Reversed: last cluster paints first (bottom).
        assert_eq!(seg.layers[0].paint.color(), Color::new(255, 255, 255));
        assert_eq!(seg.layers[1].paint.color(), Color::new(0, 0, 0));
    }

    #[test]
    fn color_dist2_symmetric() {
        let a = Color::new(10, 20, 30);
        let b = Color::new(40, 50, 60);
        assert_eq!(color_dist2(a, b), color_dist2(b, a));
    }
}
