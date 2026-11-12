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

//! End-to-end tracing pipeline.
//!
//! [`trace`] performs the complete sequence described in the crate docs and
//! returns a [`VectorDoc`]. [`trace_with_stats`] additionally returns counters
//! for every stage, which is useful for profiles, regression tests and UI
//! progress reporting.
//!
//! The pipeline is single-pass over the components but performs several
//! linear-time passes over the pixels (histogram, assignment, despeckling,
//! labeling). No stage mutates the caller's image.

use crate::color::Color;
use crate::component::{Component, DespeckleOptions, DespeckleReport, Segmentation};
use crate::config::{Config, PaintOrder};
use crate::contour::trace_component;
use crate::error::{Error, Result};
use crate::fit::fit_contours_checked;
use crate::image::ColorImage;
use crate::optimize::{OptimizeReport, optimize_document};
use crate::quantize::quantize;
use crate::segment::{despeckle, label_components};
use crate::vector::{DocumentStats, Fill, Shape, VectorDoc};

/// Counters collected while tracing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceStats {
    /// Number of palette entries used by the segmentation.
    pub palette_entries: usize,
    /// Number of connected components found after despeckling.
    pub components: usize,
    /// Number of connected components skipped because they were below the
    /// configured minimum source area or fully transparent.
    pub skipped_components: usize,
    /// Number of contour rings traced across all components.
    pub contours: u64,
    /// Number of lattice vertices before polygon simplification.
    pub contour_vertices: u64,
    /// Number of subpaths that survived fitting.
    pub fitted_subpaths: usize,
    /// Number of components whose geometry could not be fitted.
    pub fit_failures: usize,
    /// Despeckle summary.
    pub despeckle: DespeckleReport,
    /// Optimizer summary.
    pub optimize: OptimizeReport,
    /// Final document counters.
    pub document: DocumentStats,
}

impl TraceStats {
    /// Whether the reported segment counts agree with each other.
    #[inline]
    pub const fn segments_are_consistent(&self) -> bool {
        self.document.is_consistent()
    }
}

/// Traces an image with the given configuration.
pub fn trace(image: &ColorImage, config: &Config) -> Result<VectorDoc> {
    trace_with_stats(image, config).map(|(document, _)| document)
}

/// Traces an image and returns stage statistics with the document.
pub fn trace_with_stats(image: &ColorImage, config: &Config) -> Result<(VectorDoc, TraceStats)> {
    let config = Config::try_from(config)?;
    let mut stats = TraceStats::default();
    let mut indexed = quantize(image, &config.color_mode, config.background, config.alpha_cutoff)?;
    stats.palette_entries = indexed.palette().len();

    if indexed.is_empty() {
        return Ok((VectorDoc::new(image.width(), image.height()), stats));
    }

    stats.despeckle = despeckle(&mut indexed, config.connectivity, config.despeckle)?;
    let mut segmentation = label_components(&indexed, config.connectivity)?;
    stats.components = segmentation.components.len();

    if config.paint_order == PaintOrder::AreaDescending {
        segmentation.sort_by_area_descending();
    }

    let mut document = VectorDoc::new(image.width(), image.height());
    build_shapes(&segmentation, image.width(), &config, &mut document, &mut stats)?;

    if config.run_optimizer {
        stats.optimize = optimize_document(&mut document, config.optimize);
    }
    stats.document = document.stats();
    Ok((document, stats))
}

fn build_shapes(
    segmentation: &Segmentation,
    image_width: u32,
    config: &Config,
    document: &mut VectorDoc,
    stats: &mut TraceStats,
) -> Result<()> {
    for (component_id, component) in segmentation.components.iter().enumerate() {
        let Some(color) = segmentation.color_of(component) else {
            return Err(Error::InvalidPaletteEntry {
                index: component.palette_index as usize,
            });
        };
        if color.a == 0 || component.area < config.minimum_shape_area {
            stats.skipped_components += 1;
            continue;
        }

        let contours = trace_component(component, image_width, component_id)?;
        stats.contours += contours.len() as u64;
        stats.contour_vertices += contours
            .iter()
            .map(|contour| contour.vertex_count() as u64)
            .sum::<u64>();

        let Some(path) = fit_contours_checked(&contours, config.fit) else {
            stats.fit_failures += 1;
            continue;
        };
        stats.fitted_subpaths += path.len();
        let shape = Shape::new(Fill::new(color), path, component.area);
        document.push(shape);
    }
    Ok(())
}

/// Traces a single component in isolation.
///
/// Useful for callers that already performed segmentation, for example because
/// they want to apply their own color assignment. The component's pixels are
/// unpacked using `image_width`, and the returned shape uses `color` as its
/// fill.
pub fn trace_component_to_shape(
    component: &Component,
    image_width: u32,
    color: Color,
    config: &Config,
) -> Result<Option<Shape>> {
    let config = Config::try_from(config)?;
    if color.a == 0 || component.area < config.minimum_shape_area {
        return Ok(None);
    }
    let contours = trace_component(component, image_width, component.palette_index as usize)?;
    let Some(path) = fit_contours_checked(&contours, config.fit) else {
        return Ok(None);
    };
    Ok(Some(Shape::new(Fill::new(color), path, component.area)))
}

/// Traces a segmentation that was produced elsewhere.
///
/// The segmentation's palette must contain every referenced index. The result
/// is assembled with the same paint-order rule as [`trace`].
pub fn trace_segmentation(segmentation: &Segmentation, config: &Config) -> Result<VectorDoc> {
    let config = Config::try_from(config)?;
    let mut ordered = segmentation.clone();
    if config.paint_order == PaintOrder::AreaDescending {
        ordered.sort_by_area_descending();
    }
    let mut document = VectorDoc::new(segmentation.width, segmentation.height);
    let mut stats = TraceStats::default();
    build_shapes(&ordered, segmentation.width, &config, &mut document, &mut stats)?;
    if config.run_optimizer {
        optimize_document(&mut document, config.optimize);
    }
    Ok(document)
}

/// A configuration that keeps the segmentation unchanged and only traces
/// visible components; handy when benchmarking individual stages.
#[inline]
pub fn stage_config() -> Config {
    Config {
        despeckle: DespeckleOptions::disabled(),
        run_optimizer: false,
        ..Config::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::config::Config;
    use crate::palette::BinaryThreshold;

    fn black_square_on_white() -> ColorImage {
        ColorImage::from_rgb8(
            4,
            4,
            &[
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, //
                255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, //
                255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, //
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
        )
        .expect("image")
    }

    #[test]
    fn single_color_image_becomes_one_rectangle() {
        let image = ColorImage::filled(8, 5, Color::rgb(12, 34, 56)).expect("image");
        let config = Config::exact();
        let document = trace(&image, &config).expect("trace");
        assert_eq!(document.width, 8);
        assert_eq!(document.height, 5);
        assert_eq!(document.shapes.len(), 1);
        let shape = &document.shapes[0];
        assert_eq!(shape.fill.color, Color::rgb(12, 34, 56));
        assert!(libm::fabs(shape.path.signed_area(0.05) - 40.0) < 0.5);
    }

    #[test]
    fn black_square_produces_background_and_foreground() {
        let image = black_square_on_white();
        let config = Config::exact();
        let (document, stats) = trace_with_stats(&image, &config).expect("trace");
        assert_eq!(document.shapes.len(), 2);
        assert_eq!(stats.palette_entries, 2);
        assert_eq!(stats.components, 2);
        assert!(stats.segments_are_consistent());

        let black = document
            .shapes
            .iter()
            .find(|shape| shape.fill.color == Color::BLACK)
            .expect("black shape");
        assert!(libm::fabs(black.path.signed_area(0.05) - 4.0) < 0.5);
        assert_eq!(black.source_pixels, 4);

        let white = document
            .shapes
            .iter()
            .find(|shape| shape.fill.color == Color::WHITE)
            .expect("white shape");
        assert_eq!(white.path.len(), 2);
    }

    #[test]
    fn paint_order_starts_with_larger_regions() {
        let image = black_square_on_white();
        let config = Config {
            paint_order: PaintOrder::AreaDescending,
            ..Config::exact()
        };
        let document = trace(&image, &config).expect("trace");
        assert!(document.shapes[0].source_pixels >= document.shapes[1].source_pixels);
    }

    #[test]
    fn transparent_image_produces_empty_document() {
        let image = ColorImage::filled(4, 4, Color::TRANSPARENT).expect("image");
        let document = trace(&image, &Config::exact()).expect("trace");
        assert!(document.is_empty());
        assert_eq!(document.width, 4);
        assert_eq!(document.height, 4);
    }

    #[test]
    fn background_compositing_fills_transparent_pixels() {
        let image = ColorImage::from_rgba8(2, 1, &[255, 0, 0, 0, 0, 0, 255, 0]).expect("image");
        let config = Config::exact().with_background(Color::WHITE);
        let document = trace(&image, &config).expect("trace");
        assert_eq!(document.shapes.len(), 1);
        assert_eq!(document.shapes[0].fill.color, Color::WHITE);
    }

    #[test]
    fn minimum_shape_area_can_drop_everything() {
        let image = ColorImage::filled(1, 1, Color::BLACK).expect("image");
        let config = Config {
            minimum_shape_area: 2,
            ..Config::exact()
        };
        let (document, stats) = trace_with_stats(&image, &config).expect("trace");
        assert!(document.is_empty());
        assert_eq!(stats.skipped_components, 1);
    }

    #[test]
    fn binary_line_art_traces_both_colors() {
        let image = ColorImage::from_luma8(
            4,
            4,
            &[
                255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255,
            ],
        )
        .expect("image");
        let config = Config::binary(BinaryThreshold::Global(128));
        let document = trace(&image, &config).expect("trace");
        assert!(
            document
                .shapes
                .iter()
                .any(|shape| shape.fill.color == Color::BLACK)
        );
        assert!(
            document
                .shapes
                .iter()
                .any(|shape| shape.fill.color == Color::WHITE)
        );
    }

    #[test]
    fn exact_color_blocks_keep_their_fill() {
        let image = ColorImage::from_rgb8(2, 1, &[255, 0, 0, 0, 255, 0]).expect("image");
        let document = trace(&image, &Config::exact()).expect("trace");
        assert_eq!(document.shapes.len(), 2);
        let colors: Vec<Color> = document
            .shapes
            .iter()
            .map(|shape| shape.fill.color)
            .collect();
        assert!(colors.contains(&Color::rgb(255, 0, 0)));
        assert!(colors.contains(&Color::rgb(0, 255, 0)));
    }

    #[test]
    fn invalid_config_is_reported_before_tracing() {
        let image = ColorImage::filled(1, 1, Color::BLACK).expect("image");
        let config = Config {
            fit: crate::fit::FitSettings {
                curve_tolerance: 0.0,
                ..crate::fit::FitSettings::default()
            },
            ..Config::default()
        };
        assert!(matches!(trace(&image, &config), Err(Error::InvalidConfig { .. })));
    }

    #[test]
    fn stats_report_contours_and_segments() {
        let image = black_square_on_white();
        let (_, stats) = trace_with_stats(&image, &Config::exact()).expect("trace");
        assert_eq!(stats.contours, 3);
        assert!(stats.contour_vertices >= 8);
        assert!(stats.document.segments >= 8);
        assert!(stats.segments_are_consistent());
    }

    #[test]
    fn tracing_a_segmentation_matches_the_pipeline_shape_count() {
        let image = black_square_on_white();
        let config = Config::exact();
        let indexed =
            quantize(&image, &config.color_mode, config.background, config.alpha_cutoff).expect("quantize");
        let segmentation = label_components(&indexed, config.connectivity).expect("label");
        let document = trace_segmentation(&segmentation, &config).expect("trace");
        assert_eq!(document.shapes.len(), 2);
    }

    #[test]
    fn individual_component_helper_matches_shape_geometry() {
        let image = black_square_on_white();
        let config = Config::exact();
        let indexed =
            quantize(&image, &config.color_mode, config.background, config.alpha_cutoff).expect("quantize");
        let segmentation = label_components(&indexed, config.connectivity).expect("label");
        let black_component = segmentation
            .components
            .iter()
            .find(|component| component.color(&segmentation.palette) == Some(Color::BLACK))
            .expect("black component");
        let shape = trace_component_to_shape(black_component, image.width(), Color::BLACK, &config)
            .expect("trace")
            .expect("shape");
        assert!(libm::fabs(shape.path.signed_area(0.05) - 4.0) < 0.5);
    }
}
