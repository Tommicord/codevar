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

//! End-to-end tests that exercise the public API only.

use codevar_bmpvector::{
    BinaryOptions, BinaryThreshold, Color, ColorImage, ColorMode, Config, Connectivity, CornerPolicy, Error,
    FillRule, Palette, Preset, Segment, trace, trace_with_stats,
};

const EPSILON: f64 = 0.5;

fn area_of(document: &codevar_bmpvector::VectorDoc, color: Color) -> Option<f64> {
    document
        .shapes
        .iter()
        .find(|shape| shape.fill.color == color)
        .map(|shape| shape.path.signed_area(0.05))
}

fn shape_count_of(document: &codevar_bmpvector::VectorDoc, color: Color) -> usize {
    document
        .shapes
        .iter()
        .filter(|shape| shape.fill.color == color)
        .count()
}

#[test]
fn solid_region_traces_to_one_shape() {
    let image = ColorImage::filled(8, 6, Color::rgb(12, 34, 56)).expect("image");
    let document = Config::exact().trace(&image).expect("trace");

    assert_eq!(document.width, 8);
    assert_eq!(document.height, 6);
    assert_eq!(document.shapes.len(), 1);

    let shape = &document.shapes[0];
    assert_eq!(shape.fill.color, Color::rgb(12, 34, 56));
    assert_eq!(shape.fill.rule, FillRule::EvenOdd);
    assert_eq!(shape.source_pixels, 48);
    assert_eq!(shape.path.subpaths.len(), 1);
    assert!(libm::fabs(shape.path.signed_area(0.05) - 48.0) < EPSILON);

    let bounds = shape.bounds();
    assert!(libm::fabs(bounds.left) < 1.0e-6);
    assert!(libm::fabs(bounds.top) < 1.0e-6);
    assert!(libm::fabs(bounds.right - 8.0) < 1.0e-6);
    assert!(libm::fabs(bounds.bottom - 6.0) < 1.0e-6);
}

#[test]
fn ring_with_transparent_interior_keeps_its_hole() {
    // 6x6 black border with a fully transparent 4x4 interior. Exact mode must
    // keep the transparency instead of inventing a background color.
    let mut bytes = Vec::new();
    for y in 0..6 {
        for x in 0..6 {
            let border = x == 0 || y == 0 || x == 5 || y == 5;
            if border {
                bytes.extend_from_slice(&[0, 0, 0, 255]);
            } else {
                bytes.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    let image = ColorImage::from_rgba8(6, 6, &bytes).expect("image");
    let document = Config::exact().trace(&image).expect("trace");

    assert_eq!(document.shapes.len(), 1);
    let shape = &document.shapes[0];
    assert_eq!(shape.fill.color, Color::BLACK);
    assert_eq!(shape.path.subpaths.len(), 2);
    assert_eq!(shape.source_pixels, 20);
    assert!(
        libm::fabs(shape.path.signed_area(0.05) - 20.0) < EPSILON,
        "net area {}",
        shape.path.signed_area(0.05)
    );
}

#[test]
fn opaque_regions_keep_their_own_colors() {
    let red = Color::rgb(220, 20, 20);
    let blue = Color::rgb(20, 20, 220);
    let image = ColorImage::from_rgb8(
        6,
        2,
        &[
            220, 20, 20, 220, 20, 20, 220, 20, 20, 20, 20, 220, 20, 20, 220, 20, 20, 220, //
            220, 20, 20, 220, 20, 20, 220, 20, 20, 20, 20, 220, 20, 20, 220, 20, 20, 220,
        ],
    )
    .expect("image");
    let document = Config::exact().trace(&image).expect("trace");

    assert_eq!(document.shapes.len(), 2);
    assert_eq!(shape_count_of(&document, red), 1);
    assert_eq!(shape_count_of(&document, blue), 1);
    assert!(libm::fabs(area_of(&document, red).expect("red") - 6.0) < EPSILON);
    assert!(libm::fabs(area_of(&document, blue).expect("blue") - 6.0) < EPSILON);
}

#[test]
fn transparent_input_produces_no_shapes_without_background() {
    let image = ColorImage::from_rgba8(4, 4, &[0, 0, 0, 0].repeat(16)).expect("image");
    let document = Config::exact().trace(&image).expect("trace");
    assert!(document.shapes.is_empty());
    assert!(document.is_empty());
}

#[test]
fn background_compositing_covers_transparent_pixels() {
    let mut bytes = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            if x < 2 && y < 2 {
                bytes.extend_from_slice(&[255, 0, 0, 255]);
            } else {
                bytes.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    let image = ColorImage::from_rgba8(4, 4, &bytes).expect("image");

    let without = Config::exact().trace(&image).expect("trace");
    assert_eq!(without.shapes.len(), 1);
    assert!(libm::fabs(area_of(&without, Color::rgb(255, 0, 0)).expect("red") - 4.0) < EPSILON);

    let with = Config::exact()
        .with_background(Color::WHITE)
        .trace(&image)
        .expect("trace");
    assert_eq!(with.shapes.len(), 2);
    assert!(libm::fabs(area_of(&with, Color::WHITE).expect("white") - 12.0) < EPSILON);
    assert!(libm::fabs(area_of(&with, Color::rgb(255, 0, 0)).expect("red") - 4.0) < EPSILON);
}

#[test]
fn binary_global_threshold_splits_luma() {
    // The default despeckle pass removes components below four pixels, so the
    // fixture is large enough for both regions to survive it.
    let image = ColorImage::from_luma8(8, 2, &[0, 0, 0, 0, 192, 192, 192, 192].repeat(2)).expect("image");
    let document = Config::binary(BinaryThreshold::Global(128))
        .trace(&image)
        .expect("trace");

    assert_eq!(document.shapes.len(), 2);
    assert!(libm::fabs(area_of(&document, Color::BLACK).expect("black") - 8.0) < EPSILON);
    assert!(libm::fabs(area_of(&document, Color::WHITE).expect("white") - 8.0) < EPSILON);
}

#[test]
fn binary_otsu_and_adaptive_modes_trace_the_whole_image() {
    let image = ColorImage::from_luma8(
        8,
        8,
        &[
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240, //
            0, 0, 0, 0, 240, 240, 240, 240,
        ],
    )
    .expect("image");

    for threshold in [
        BinaryThreshold::Otsu,
        BinaryThreshold::Adaptive { radius: 2, offset: 0 },
    ] {
        let document = Config::binary(threshold)
            .trace(&image)
            .expect("trace");
        let total: f64 = document
            .shapes
            .iter()
            .map(|shape| shape.path.signed_area(0.05))
            .sum();
        assert!(
            libm::fabs(total - 64.0) < EPSILON,
            "threshold {threshold:?} total {total}"
        );
        assert!(shape_count_of(&document, Color::BLACK) >= 1);
        assert!(shape_count_of(&document, Color::WHITE) >= 1);
    }
}

#[test]
fn fixed_palette_snaps_regions_to_the_nearest_entry() {
    let palette = Palette::from_slice(&[Color::rgb(255, 0, 0), Color::rgb(0, 0, 255)]).expect("palette");
    let image =
        ColorImage::from_rgb8(4, 1, &[250, 10, 10, 240, 20, 20, 10, 10, 250, 20, 20, 240]).expect("image");
    let document = Config::exact()
        .with_palette(palette)
        .trace(&image)
        .expect("trace");

    assert_eq!(document.shapes.len(), 2);
    assert!(libm::fabs(area_of(&document, Color::rgb(255, 0, 0)).expect("red") - 2.0) < EPSILON);
    assert!(libm::fabs(area_of(&document, Color::rgb(0, 0, 255)).expect("blue") - 2.0) < EPSILON);
}

#[test]
fn presets_trace_without_errors() {
    let image = ColorImage::from_luma8(6, 6, &[0, 60, 120, 180, 240, 255].repeat(6)).expect("image");
    for preset in [
        Preset::Logo,
        Preset::Poster,
        Preset::Photo,
        Preset::LineArt,
        Preset::Exact,
        Preset::Binary(BinaryThreshold::Otsu),
    ] {
        let document = Config::from(preset)
            .trace(&image)
            .expect("preset trace");
        assert_eq!(document.width, 6);
        assert_eq!(document.height, 6);
    }
}

#[test]
fn statistics_agree_with_the_document() {
    let image = ColorImage::from_rgb8(4, 4, &[10, 20, 30].repeat(16)).expect("image");
    let (document, stats) = trace_with_stats(&image, &Config::exact()).expect("trace");

    assert_eq!(stats.palette_entries, 1);
    assert_eq!(stats.components, 1);
    assert_eq!(stats.skipped_components, 0);
    assert_eq!(stats.document.shapes, document.shapes.len());
    assert_eq!(stats.document.segments, document.segment_count());
    assert_eq!(stats.document.subpaths, document.subpath_count());
    assert!(stats.contours >= 1);
    assert!(stats.fitted_subpaths >= 1);
    assert!(stats.segments_are_consistent());
}

#[test]
fn tracing_is_deterministic() {
    let image = ColorImage::from_luma8(5, 5, &[0, 80, 160, 240, 255].repeat(5)).expect("image");
    let config = Config::logo();
    let first = trace(&image, &config).expect("first");
    let second = trace(&image, &config).expect("second");
    assert_eq!(first, second);
}

#[test]
fn segmentation_can_be_traced_directly() {
    let image = ColorImage::from_rgb8(4, 2, &[200, 10, 10].repeat(8)).expect("image");
    let indexed =
        codevar_bmpvector::quantize::quantize(&image, &ColorMode::Exact, None, 1).expect("quantize");
    let segmentation =
        codevar_bmpvector::segment::label_components(&indexed, Connectivity::Four).expect("label");
    let document =
        codevar_bmpvector::pipeline::trace_segmentation(&segmentation, &Config::exact()).expect("trace");
    assert_eq!(document.shapes.len(), 1);
    assert!(libm::fabs(document.shapes[0].path.signed_area(0.05) - 8.0) < EPSILON);
}

#[test]
fn paths_are_lattice_aligned_and_closed() {
    let image = ColorImage::from_rgb8(3, 3, &[0, 0, 0].repeat(9)).expect("image");
    let document = Config::exact().trace(&image).expect("trace");
    let shape = &document.shapes[0];
    assert!(shape.path.is_finite());

    for subpath in &shape.path.subpaths {
        assert!(subpath.closed);
        for segment in &subpath.segments {
            match *segment {
                Segment::Line { start, end } => {
                    assert!(start.x >= 0.0 && start.x <= 3.0);
                    assert!(start.y >= 0.0 && start.y <= 3.0);
                    assert!(end.x >= 0.0 && end.x <= 3.0);
                    assert!(end.y >= 0.0 && end.y <= 3.0);
                }
                Segment::Cubic(cubic) => {
                    assert!(cubic.start.x >= 0.0 && cubic.start.x <= 3.0);
                    assert!(cubic.end.y >= 0.0 && cubic.end.y <= 3.0);
                }
            }
        }
    }
}

#[test]
fn invalid_configurations_are_rejected_before_tracing() {
    let image = ColorImage::filled(2, 2, Color::BLACK).expect("image");

    let zero_colors = Config {
        color_mode: ColorMode::Auto {
            max_colors: 0,
            refine_iterations: 0,
        },
        ..Config::default()
    };
    assert!(matches!(
        trace(&image, &zero_colors),
        Err(Error::InvalidConfig { field: "max_colors" })
    ));

    let translucent_background = Config {
        background: Some(Color::rgba(1, 2, 3, 4)),
        ..Config::exact()
    };
    assert!(matches!(
        trace(&image, &translucent_background),
        Err(Error::InvalidConfig { field: "background" })
    ));

    let bad_corners = Config::exact().with_corners(CornerPolicy {
        threshold_degrees: 200.0,
        minimum_arm: 0.0,
    });
    assert!(matches!(
        trace(&image, &bad_corners),
        Err(Error::InvalidConfig {
            field: "corner_threshold_degrees"
        })
    ));
}

#[test]
fn builders_reject_invalid_parameters() {
    assert!(matches!(
        BinaryOptions::try_new(BinaryThreshold::Adaptive { radius: 0, offset: 0 }),
        Err(Error::InvalidConfig {
            field: "binary_radius"
        })
    ));
    assert!(CornerPolicy::try_new(-1.0, 0.0).is_err());
    assert!(Palette::try_from([].as_slice()).is_err());
    assert!(Palette::new(Vec::new()).is_err());
    assert!(Palette::from_slice(&[Color::BLACK]).is_ok());
}
