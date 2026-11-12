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

//! Palette selection and per-pixel color assignment.
//!
//! The quantizer turns a raster into an [`IndexedImage`]: one palette index
//! per pixel, plus the palette those indices refer to.
//!
//! All histogram work is deterministic: colors are sorted by packed RGBA value
//! before clustering, so equal inputs always produce equal palettes.

use alloc::vec;
use alloc::vec::Vec;

use libm::fabs;

use crate::color::{Color, Oklab};
use crate::error::{Error, Result};
use crate::geom::clamp;
use crate::image::ColorImage;
use crate::palette::{
    BinaryOptions, BinaryThreshold, ColorMode, IndexedImage, MAX_PALETTE_ENTRIES, NO_INDEX, Palette,
};

/// Builds the indexed image used by segmentation.
///
/// The image is first composited over `background` when one is supplied. When
/// it is not, pixels whose alpha is below `alpha_cutoff` become [`NO_INDEX`] and
/// the remaining pixels retain their alpha in the palette. `alpha_cutoff` of
/// zero means only fully transparent pixels are dropped.
pub fn quantize(
    image: &ColorImage,
    mode: &ColorMode,
    background: Option<Color>,
    alpha_cutoff: u8,
) -> Result<IndexedImage> {
    ColorMode::try_from(mode)?;
    let flattened;
    let source = if let Some(background) = background {
        flattened = image.composite_over(background);
        &flattened
    } else {
        image
    };

    if let ColorMode::Binary(options) = mode {
        return quantize_binary(source, options, alpha_cutoff);
    }

    let histogram = build_histogram(source, alpha_cutoff);
    if histogram.is_empty() {
        return Ok(IndexedImage::empty(source.width(), source.height()));
    }

    let palette = match mode {
        ColorMode::Exact => build_exact_palette(&histogram)?,
        ColorMode::Fixed(palette) => palette.clone(),
        ColorMode::Auto {
            max_colors,
            refine_iterations,
        } => build_auto_palette(&histogram, *max_colors, *refine_iterations),
        ColorMode::Binary(_) => {
            // Handled above. Keeping the arm explicit avoids silently changing
            // behavior if a new mode is added.
            return Err(Error::InvalidConfig { field: "color_mode" });
        }
    };

    assign_histogram_pixels(source, &histogram, &palette, alpha_cutoff)
}

/// One distinct color and how many pixels use it.
#[derive(Clone, Copy, Debug)]
struct HistogramEntry {
    packed: u32,
    color: Color,
    count: u64,
    lab: Oklab,
}

fn build_histogram(image: &ColorImage, alpha_cutoff: u8) -> Vec<HistogramEntry> {
    let mut packed: Vec<u32> = Vec::with_capacity(image.pixel_count());
    for pixel in image.as_pixels() {
        if pixel.a < alpha_cutoff || pixel.a == 0 {
            continue;
        }
        packed.push(pixel.to_rgba_u32());
    }
    packed.sort_unstable();

    let mut entries = Vec::new();
    let mut cursor = 0;
    while cursor < packed.len() {
        let value = packed[cursor];
        let mut end = cursor + 1;
        while end < packed.len() && packed[end] == value {
            end += 1;
        }
        let count = (end - cursor) as u64;
        let color = Color::from_rgba_u32(value);
        entries.push(HistogramEntry {
            packed: value,
            color,
            count,
            lab: color.to_oklab(),
        });
        cursor = end;
    }
    entries
}

fn build_exact_palette(histogram: &[HistogramEntry]) -> Result<Palette> {
    if histogram.len() > MAX_PALETTE_ENTRIES {
        return Err(Error::PaletteTooLarge {
            entries: histogram.len(),
        });
    }
    Ok(Palette::from_quantizer(
        histogram
            .iter()
            .map(|entry| entry.color)
            .collect(),
    ))
}

fn build_auto_palette(histogram: &[HistogramEntry], max_colors: usize, refine_iterations: u8) -> Palette {
    let max_colors = max_colors.min(histogram.len()).max(1);
    if histogram.len() <= max_colors {
        return Palette::from_quantizer(
            histogram
                .iter()
                .map(|entry| entry.color)
                .collect(),
        );
    }

    let mut buckets: Vec<Vec<usize>> = vec![(0..histogram.len()).collect()];
    while buckets.len() < max_colors {
        let mut best: Option<(usize, usize, f64)> = None;
        for (bucket_index, bucket) in buckets.iter().enumerate() {
            if bucket.len() < 2 {
                continue;
            }
            let (axis, score) = split_score(bucket, histogram);
            if score <= 0.0 || !score.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, _, best_score)| score > best_score) {
                best = Some((bucket_index, axis, score));
            }
        }
        let Some((bucket_index, axis, _)) = best else {
            break;
        };
        let bucket = buckets.swap_remove(bucket_index);
        let (left, right) = split_bucket(bucket, histogram, axis);
        if left.is_empty() || right.is_empty() {
            break;
        }
        buckets.push(left);
        buckets.push(right);
    }

    let mut centroids: Vec<Oklab> = buckets
        .iter()
        .map(|bucket| bucket_mean(bucket, histogram))
        .collect();

    for _ in 0..refine_iterations {
        let mut sums = vec![Oklab::default(); centroids.len()];
        let mut weights = vec![0.0_f64; centroids.len()];
        for entry in histogram {
            let nearest = nearest_centroid(entry.lab, &centroids);
            let weight = entry.count as f64;
            sums[nearest].add_scaled(entry.lab, weight);
            weights[nearest] += weight;
        }
        for (centroid, (sum, weight)) in centroids
            .iter_mut()
            .zip(sums.iter().zip(&weights))
        {
            if *weight > 0.0 {
                let mut updated = *sum;
                updated.divide(*weight);
                *centroid = updated;
            }
        }
    }

    let mut colors: Vec<Color> = Vec::with_capacity(centroids.len());
    let mut seen: Vec<u32> = Vec::with_capacity(centroids.len());
    for centroid in centroids {
        let color = centroid.to_color();
        let packed = color.to_rgba_u32();
        if seen.binary_search(&packed).is_err() {
            let position = seen
                .binary_search(&packed)
                .unwrap_or_else(|position| position);
            seen.insert(position, packed);
            colors.push(color);
        }
    }

    if colors.is_empty() {
        Palette::from_quantizer(
            histogram
                .iter()
                .map(|entry| entry.color)
                .collect(),
        )
    } else {
        Palette::from_quantizer(colors)
    }
}

fn split_score(bucket: &[usize], histogram: &[HistogramEntry]) -> (usize, f64) {
    let mut weight = 0.0_f64;
    let mut mean = [0.0_f64; 4];
    let mut squares = [0.0_f64; 4];
    for &index in bucket {
        let entry = &histogram[index];
        let w = entry.count as f64;
        weight += w;
        let values = [entry.lab.l, entry.lab.a, entry.lab.b, entry.lab.alpha];
        for axis in 0..4 {
            mean[axis] += values[axis] * w;
            squares[axis] += values[axis] * values[axis] * w;
        }
    }
    if weight <= 0.0 {
        return (0, 0.0);
    }
    let mut best_axis = 0;
    let mut best_variance = 0.0;
    for axis in 0..4 {
        let average = mean[axis] / weight;
        let variance = (squares[axis] / weight - average * average).max(0.0);
        if variance > best_variance {
            best_variance = variance;
            best_axis = axis;
        }
    }
    (best_axis, weight * best_variance)
}

fn lab_component(lab: Oklab, axis: usize) -> f64 {
    match axis {
        0 => lab.l,
        1 => lab.a,
        2 => lab.b,
        _ => lab.alpha,
    }
}

fn split_bucket(
    mut bucket: Vec<usize>,
    histogram: &[HistogramEntry],
    axis: usize,
) -> (Vec<usize>, Vec<usize>) {
    bucket.sort_unstable_by(|&a, &b| {
        lab_component(histogram[a].lab, axis)
            .total_cmp(&lab_component(histogram[b].lab, axis))
            .then_with(|| histogram[a].packed.cmp(&histogram[b].packed))
    });
    let total: u64 = bucket
        .iter()
        .map(|&index| histogram[index].count)
        .sum();
    let half = total / 2;
    let mut cumulative = 0_u64;
    let mut split = bucket.len() / 2;
    for (position, &index) in bucket.iter().enumerate().take(bucket.len() - 1) {
        cumulative += histogram[index].count;
        if cumulative >= half {
            split = position + 1;
            break;
        }
    }
    let split = split.clamp(1, bucket.len() - 1);
    let right = bucket.split_off(split);
    (bucket, right)
}

fn bucket_mean(bucket: &[usize], histogram: &[HistogramEntry]) -> Oklab {
    let mut sum = Oklab::default();
    let mut weight = 0.0_f64;
    for &index in bucket {
        let entry = &histogram[index];
        let w = entry.count as f64;
        sum.add_scaled(entry.lab, w);
        weight += w;
    }
    sum.divide(weight);
    sum
}

fn nearest_centroid(target: Oklab, centroids: &[Oklab]) -> usize {
    let mut best_index = 0;
    let mut best_distance = f64::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = target.distance_sq(*centroid);
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    best_index
}

fn assign_histogram_pixels(
    image: &ColorImage,
    histogram: &[HistogramEntry],
    palette: &Palette,
    alpha_cutoff: u8,
) -> Result<IndexedImage> {
    let mut palette_for_color = Vec::with_capacity(histogram.len());
    for entry in histogram {
        let index = palette
            .nearest(entry.color)
            .map(|(index, _)| index as u32)
            .ok_or(Error::EmptyPalette)?;
        palette_for_color.push(index);
    }

    let mut indices = Vec::with_capacity(image.pixel_count());
    for pixel in image.as_pixels() {
        if pixel.a < alpha_cutoff || pixel.a == 0 {
            indices.push(NO_INDEX);
            continue;
        }
        let packed = pixel.to_rgba_u32();
        let position = histogram
            .binary_search_by_key(&packed, |entry| entry.packed)
            .map_err(|_| Error::NumericalFailure {
                operation: "histogram lookup",
            })?;
        indices.push(palette_for_color[position]);
    }

    Ok(IndexedImage::from_quantizer(
        image.width(),
        image.height(),
        indices,
        palette.clone(),
    ))
}

fn quantize_binary(image: &ColorImage, options: &BinaryOptions, alpha_cutoff: u8) -> Result<IndexedImage> {
    let options = BinaryOptions::try_from(options)?;
    let palette = Palette::from_quantizer(vec![options.background, options.foreground]);
    let threshold = match options.threshold {
        BinaryThreshold::Global(value) => ThresholdMap::Global(value),
        BinaryThreshold::Otsu => ThresholdMap::Global(otsu_threshold(image, alpha_cutoff)),
        BinaryThreshold::Adaptive { radius, offset } => ThresholdMap::Adaptive {
            integral: image.integral_luma(),
            radius,
            offset,
        },
    };

    let mut indices = Vec::with_capacity(image.pixel_count());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let pixel = image.pixel_unchecked(x, y);
            if pixel.a < alpha_cutoff || pixel.a == 0 {
                indices.push(NO_INDEX);
                continue;
            }
            let foreground = threshold.is_foreground(x, y, pixel.intensity());
            indices.push(if foreground { 1 } else { 0 });
        }
    }

    Ok(IndexedImage::from_quantizer(
        image.width(),
        image.height(),
        indices,
        palette,
    ))
}

enum ThresholdMap {
    Global(u8),
    Adaptive {
        integral: crate::image::IntegralImage,
        radius: u32,
        offset: i16,
    },
}

impl ThresholdMap {
    fn is_foreground(&self, x: u32, y: u32, luma: u8) -> bool {
        match self {
            Self::Global(cutoff) => luma < *cutoff,
            Self::Adaptive {
                integral,
                radius,
                offset,
            } => {
                let mean = integral
                    .local_mean(x, y, *radius)
                    .unwrap_or(luma as f64);
                let cutoff = clamp(mean + *offset as f64, 0.0, 255.0);
                (luma as f64) < cutoff
            }
        }
    }
}

fn otsu_threshold(image: &ColorImage, alpha_cutoff: u8) -> u8 {
    let mut histogram = [0_u64; 256];
    let mut total = 0_u64;
    for pixel in image.as_pixels() {
        if pixel.a < alpha_cutoff || pixel.a == 0 {
            continue;
        }
        histogram[pixel.intensity() as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return 0;
    }

    let total_sum: f64 = histogram
        .iter()
        .enumerate()
        .map(|(value, &count)| value as f64 * count as f64)
        .sum();
    let mut background_weight = 0_u64;
    let mut background_sum = 0.0_f64;
    let mut best_variance = -1.0_f64;
    let mut best_threshold = 0_u8;

    for (value, &count) in histogram.iter().enumerate() {
        background_weight += count;
        if background_weight == 0 {
            continue;
        }
        let foreground_weight = total - background_weight;
        if foreground_weight == 0 {
            break;
        }
        background_sum += value as f64 * count as f64;
        let background_mean = background_sum / background_weight as f64;
        let foreground_mean = (total_sum - background_sum) / foreground_weight as f64;
        let difference = background_mean - foreground_mean;
        let variance = background_weight as f64 * foreground_weight as f64 * difference * difference;
        if variance > best_variance {
            best_variance = variance;
            best_threshold = value as u8;
        }
    }

    // Otsu returns the last value of the background class. Adding one makes the
    // comparison `luma < cutoff` put that value on the foreground side, which
    // matches the intuitive meaning of "threshold at N".
    best_threshold.saturating_add(1)
}

/// Returns the absolute difference between two channel values, for tests and
/// diagnostics.
#[inline]
pub fn channel_distance(first: u8, second: u8) -> u8 {
    first.abs_diff(second)
}

/// Checks whether two colors are perceptually indistinguishable at the given
/// squared Oklab distance.
#[inline]
pub fn colors_match(first: Color, second: Color, maximum_distance_sq: f64) -> bool {
    first.to_oklab().distance_sq(second.to_oklab()) <= maximum_distance_sq
}

/// A helper that merges colors closer than `distance` in Oklab.
///
/// Not used by the automatic pipeline, which relies on k-means, but useful for
/// callers that post-process a fixed palette.
pub fn deduplicate_palette(colors: &[Color], maximum_distance_sq: f64) -> Palette {
    let mut retained: Vec<Color> = Vec::with_capacity(colors.len());
    for &color in colors {
        let duplicate = retained
            .iter()
            .any(|existing| colors_match(*existing, color, maximum_distance_sq));
        if !duplicate {
            retained.push(color);
        }
    }
    Palette::from_quantizer(retained)
}

/// The fraction of pixels that differ between two indexed images.
///
/// The images must share dimensions; otherwise the function returns `None`.
pub fn assignment_difference(first: &IndexedImage, second: &IndexedImage) -> Option<f64> {
    if first.width() != second.width()
        || first.height() != second.height()
        || first.pixel_count() != second.pixel_count()
    {
        return None;
    }
    if first.pixel_count() == 0 {
        return Some(0.0);
    }
    let differing = first
        .indices()
        .iter()
        .zip(second.indices())
        .filter(|(a, b)| a != b)
        .count();
    Some(differing as f64 / first.pixel_count() as f64)
}

/// Whether a palette color is a neutral gray within `tolerance` in Oklab.
pub fn is_neutral(color: Color, tolerance: f64) -> bool {
    color.to_oklab().chroma() <= fabs(tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ColorImage;

    fn colored_image() -> ColorImage {
        ColorImage::from_rgb8(
            4,
            2,
            &[
                255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255,
            ],
        )
        .expect("image")
    }

    #[test]
    fn exact_palette_contains_every_color_and_assigns_them() {
        let image = colored_image();
        let indexed = quantize(&image, &ColorMode::Exact, None, 1).expect("quantize");
        assert_eq!(indexed.palette().len(), 4);
        assert_eq!(indexed.color_at(0, 0), Some(Color::rgb(255, 0, 0)));
        assert_eq!(indexed.color_at(1, 0), Some(Color::rgb(255, 0, 0)));
        assert_eq!(indexed.color_at(2, 0), Some(Color::rgb(0, 0, 255)));
        assert_eq!(indexed.color_at(1, 1), Some(Color::BLACK));
        assert_eq!(indexed.color_at(3, 1), Some(Color::WHITE));
    }

    #[test]
    fn fixed_palette_uses_nearest_entry() {
        let image = ColorImage::filled(2, 2, Color::rgb(200, 10, 10)).expect("image");
        let palette = Palette::from_slice(&[Color::rgb(255, 0, 0), Color::rgb(0, 0, 255)]).expect("palette");
        let indexed = quantize(&image, &ColorMode::Fixed(palette), None, 1).expect("quantize");
        assert_eq!(indexed.index(0, 0), 0);
        assert_eq!(indexed.palette().len(), 2);
    }

    #[test]
    fn auto_palette_respects_the_color_budget() {
        let mut pixels = Vec::new();
        for value in 0..64_u8 {
            pixels.push(Color::rgb(value * 4, 255 - value * 4, value * 2));
        }
        let image = ColorImage::new(8, 8, pixels).expect("image");
        let indexed = quantize(
            &image,
            &ColorMode::Auto {
                max_colors: 4,
                refine_iterations: 3,
            },
            None,
            1,
        )
        .expect("quantize");
        assert!(indexed.palette().len() <= 4);
        assert!(!indexed.palette().is_empty());
        assert_eq!(indexed.pixel_count(), 64);
    }

    #[test]
    fn fully_transparent_image_yields_empty_index_map() {
        let image = ColorImage::filled(3, 3, Color::TRANSPARENT).expect("image");
        let indexed = quantize(&image, &ColorMode::Exact, None, 1).expect("quantize");
        assert!(indexed.is_empty());
        assert_eq!(indexed.transparent_pixels(), 9);
        assert_eq!(indexed.palette().len(), 0);
    }

    #[test]
    fn background_compositing_removes_transparency() {
        let image = ColorImage::from_rgba8(1, 1, &[255, 0, 0, 0]).expect("image");
        let indexed = quantize(&image, &ColorMode::Exact, Some(Color::WHITE), 1).expect("quantize");
        assert_eq!(indexed.transparent_pixels(), 0);
        assert_eq!(indexed.color_at(0, 0), Some(Color::WHITE));
    }

    #[test]
    fn binary_global_threshold_splits_luma() {
        let image = ColorImage::from_luma8(4, 1, &[0, 50, 200, 255]).expect("image");
        let options = BinaryOptions::new(BinaryThreshold::Global(128));
        let indexed = quantize(&image, &ColorMode::Binary(options), None, 1).expect("quantize");
        assert_eq!(indexed.index(0, 0), 1);
        assert_eq!(indexed.index(1, 0), 1);
        assert_eq!(indexed.index(2, 0), 0);
        assert_eq!(indexed.index(3, 0), 0);
        assert_eq!(indexed.color_at(0, 0), Some(Color::BLACK));
        assert_eq!(indexed.color_at(2, 0), Some(Color::WHITE));
    }

    #[test]
    fn otsu_threshold_separates_a_bimodal_image() {
        let mut pixels = Vec::new();
        for _ in 0..16 {
            pixels.push(Color::rgb(10, 10, 10));
        }
        for _ in 0..16 {
            pixels.push(Color::rgb(240, 240, 240));
        }
        let image = ColorImage::new(32, 1, pixels).expect("image");
        let threshold = otsu_threshold(&image, 1);
        assert!(threshold > 10 && threshold <= 240);
    }

    #[test]
    fn adaptive_threshold_tracks_local_contrast() {
        let mut luma = vec![240_u8; 100];
        for y in 4..6 {
            for x in 4..6 {
                luma[y * 10 + x] = 20;
            }
        }
        let image = ColorImage::from_luma8(10, 10, &luma).expect("image");
        let options = BinaryOptions::new(BinaryThreshold::Adaptive { radius: 3, offset: 0 });
        let indexed = quantize(&image, &ColorMode::Binary(options), None, 1).expect("quantize");
        assert_eq!(indexed.index(5, 5), 1);
    }

    #[test]
    fn invalid_modes_are_rejected() {
        let image = ColorImage::filled(1, 1, Color::BLACK).expect("image");
        assert!(matches!(
            quantize(
                &image,
                &ColorMode::Auto {
                    max_colors: 0,
                    refine_iterations: 0
                },
                None,
                1
            ),
            Err(Error::InvalidConfig { .. })
        ));
        assert!(matches!(
            quantize(
                &image,
                &ColorMode::Binary(BinaryOptions::new(BinaryThreshold::Adaptive {
                    radius: 0,
                    offset: 0
                })),
                None,
                1
            ),
            Err(Error::InvalidConfig { .. })
        ));
    }

    #[test]
    fn deduplication_and_nearest_are_stable() {
        let palette = Palette::from_slice(&[
            Color::rgb(255, 0, 0),
            Color::rgb(255, 1, 0),
            Color::rgb(0, 0, 255),
        ])
        .expect("palette");
        let (index, _) = palette
            .nearest(Color::rgb(250, 0, 0))
            .expect("nearest");
        assert_eq!(index, 0);

        let deduplicated = deduplicate_palette(palette.colors(), 0.0001);
        assert_eq!(deduplicated.len(), 2);
    }
}
