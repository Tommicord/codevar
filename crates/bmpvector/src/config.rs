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

//! Trace configuration and style presets.
//!
//! The configuration is intentionally explicit: every stage that can trade
//! fidelity for compactness has its own field. Presets are starting points, not
//! opaque bundles, so a caller can take a preset and then adjust one tolerance
//! without discovering hidden settings.

use crate::color::Color;
use crate::component::{Connectivity, DespeckleOptions};
use crate::error::{Error, Result};
use crate::fit::{CornerPolicy, FitSettings};
use crate::image::ColorImage;
use crate::optimize::OptimizeSettings;
use crate::palette::{BinaryOptions, BinaryThreshold, ColorMode, Palette};
use crate::pipeline;
use crate::vector::VectorDoc;

/// Order in which shapes are written to the output document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PaintOrder {
    /// Keep the first-pixel raster order produced by component labeling.
    Source,
    /// Draw larger source regions first, so fine details remain on top.
    #[default]
    AreaDescending,
}

/// A named starting configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Flat logos, icons and pictograms: few colors and crisp corners.
    Logo,
    /// Posters and illustrations: moderate color count and smooth curves.
    Poster,
    /// Photographs: more colors, stronger despeckling and relaxed curvature.
    Photo,
    /// Black-and-white line art traced through Otsu thresholding.
    LineArt,
    /// Every distinct input color is preserved.
    Exact,
    /// A caller-selected binary threshold.
    Binary(BinaryThreshold),
}

/// Complete tracing configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Palette strategy.
    pub color_mode: ColorMode,
    /// Optional opaque background used to flatten transparency before
    /// quantization. `None` preserves alpha and drops fully transparent pixels.
    pub background: Option<Color>,
    /// Pixels with alpha below this value are treated as fully transparent.
    pub alpha_cutoff: u8,
    /// Pixel adjacency used for connected components.
    pub connectivity: Connectivity,
    /// Small-component cleanup.
    pub despeckle: DespeckleOptions,
    /// Polygon simplification and curve fitting.
    pub fit: FitSettings,
    /// Post-fitting cleanup.
    pub optimize: OptimizeSettings,
    /// Shape paint order.
    pub paint_order: PaintOrder,
    /// Source components smaller than this number of pixels are not emitted.
    ///
    /// This filter is applied after fitting and is independent of
    /// [`Config::despeckle`]; it is useful when a caller wants the segmentation
    /// unchanged but does not want microscopic output shapes.
    pub minimum_shape_area: u64,
    /// Whether to run the post-fitting optimizer.
    pub run_optimizer: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            color_mode: ColorMode::Auto {
                max_colors: 16,
                refine_iterations: 4,
            },
            background: None,
            alpha_cutoff: 1,
            connectivity: Connectivity::Eight,
            despeckle: DespeckleOptions::default(),
            fit: FitSettings::default(),
            optimize: OptimizeSettings::default(),
            paint_order: PaintOrder::AreaDescending,
            minimum_shape_area: 0,
            run_optimizer: true,
        }
    }
}

impl Config {
    /// The configuration for a preset.
    pub fn preset(preset: Preset) -> Self {
        match preset {
            Preset::Logo => Self::logo(),
            Preset::Poster => Self::poster(),
            Preset::Photo => Self::photo(),
            Preset::LineArt => Self::line_art(),
            Preset::Exact => Self::exact(),
            Preset::Binary(threshold) => Self::binary(threshold),
        }
    }

    /// Flat logo/icon preset.
    pub fn logo() -> Self {
        Self {
            color_mode: ColorMode::Auto {
                max_colors: 8,
                refine_iterations: 4,
            },
            despeckle: DespeckleOptions {
                minimum_area: 3,
                passes: 2,
            },
            fit: FitSettings::line_art(),
            optimize: OptimizeSettings::default(),
            ..Self::default()
        }
    }

    /// Illustration/poster preset.
    pub fn poster() -> Self {
        Self {
            color_mode: ColorMode::Auto {
                max_colors: 12,
                refine_iterations: 5,
            },
            despeckle: DespeckleOptions {
                minimum_area: 5,
                passes: 2,
            },
            fit: FitSettings::default(),
            optimize: OptimizeSettings::default(),
            ..Self::default()
        }
    }

    /// Photograph preset.
    pub fn photo() -> Self {
        Self {
            color_mode: ColorMode::Auto {
                max_colors: 32,
                refine_iterations: 6,
            },
            despeckle: DespeckleOptions {
                minimum_area: 8,
                passes: 2,
            },
            fit: FitSettings::painterly(),
            optimize: OptimizeSettings {
                cubic_merge_tolerance: 0.5,
                ..OptimizeSettings::default()
            },
            ..Self::default()
        }
    }

    /// Black-and-white line-art preset using Otsu thresholding.
    pub fn line_art() -> Self {
        Self {
            color_mode: ColorMode::Binary(BinaryOptions::new(BinaryThreshold::Otsu)),
            background: Some(Color::WHITE),
            despeckle: DespeckleOptions {
                minimum_area: 2,
                passes: 2,
            },
            fit: FitSettings::line_art(),
            ..Self::default()
        }
    }

    /// Exact-color preset.
    pub fn exact() -> Self {
        Self {
            color_mode: ColorMode::Exact,
            despeckle: DespeckleOptions::disabled(),
            ..Self::default()
        }
    }

    /// Binary preset with an explicit threshold.
    pub fn binary(threshold: BinaryThreshold) -> Self {
        Self {
            color_mode: ColorMode::Binary(BinaryOptions::new(threshold)),
            background: Some(Color::WHITE),
            ..Self::default()
        }
    }

    /// Uses a fixed palette instead of selecting one.
    pub fn with_palette(mut self, palette: Palette) -> Self {
        self.color_mode = ColorMode::Fixed(palette);
        self
    }

    /// Sets the maximum number of automatically selected colors.
    pub fn with_max_colors(mut self, max_colors: usize) -> Self {
        self.color_mode = ColorMode::Auto {
            max_colors,
            refine_iterations: 4,
        };
        self
    }

    /// Sets an opaque compositing background.
    pub fn with_background(mut self, background: Color) -> Self {
        self.background = Some(background.with_alpha(255));
        self
    }

    /// Removes the compositing background, preserving alpha.
    pub fn with_transparency(mut self) -> Self {
        self.background = None;
        self
    }

    /// Replaces the curve-fitting settings.
    pub fn with_fit_settings(mut self, fit: FitSettings) -> Self {
        self.fit = fit;
        self
    }

    /// Replaces the corner policy while keeping the other fit settings.
    pub fn with_corners(mut self, corners: CornerPolicy) -> Self {
        self.fit.corners = corners;
        self
    }

    /// Replaces the post-fitting optimizer settings.
    pub fn with_optimizer(mut self, optimize: OptimizeSettings) -> Self {
        self.optimize = optimize;
        self
    }

    /// Traces an image with this configuration.
    pub fn trace(&self, image: &ColorImage) -> Result<VectorDoc> {
        pipeline::trace(image, self)
    }

    /// Traces an image and returns pipeline statistics.
    pub fn trace_with_stats(&self, image: &ColorImage) -> Result<(VectorDoc, pipeline::TraceStats)> {
        pipeline::trace_with_stats(image, self)
    }
}

/// A convenience constructor for the default configuration.
impl From<Preset> for Config {
    fn from(preset: Preset) -> Self {
        Self::preset(preset)
    }
}

/// Validates a configuration and returns an owned copy.
///
/// This is the canonical way to turn a possibly hand-built [`Config`] into a
/// value the pipeline accepts. Presets and [`Config::default`] always
/// round-trip successfully; a field that is out of range is reported as
/// [`Error::InvalidConfig`] before any pixel is read.
impl TryFrom<&Config> for Config {
    type Error = Error;

    fn try_from(config: &Config) -> Result<Self> {
        ColorMode::try_from(&config.color_mode)?;
        FitSettings::try_from(&config.fit)?;
        OptimizeSettings::try_from(&config.optimize)?;
        if config
            .background
            .is_some_and(|background| background.a != 255)
        {
            return Err(Error::InvalidConfig { field: "background" });
        }
        Ok(config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips_through_try_from() {
        Config::try_from(&Config::default()).expect("default config");
    }

    #[test]
    fn every_preset_round_trips_through_try_from() {
        for preset in [
            Preset::Logo,
            Preset::Poster,
            Preset::Photo,
            Preset::LineArt,
            Preset::Exact,
            Preset::Binary(BinaryThreshold::Otsu),
            Preset::Binary(BinaryThreshold::Global(128)),
            Preset::Binary(BinaryThreshold::Adaptive { radius: 4, offset: 2 }),
        ] {
            Config::try_from(&Config::preset(preset))
                .unwrap_or_else(|error| panic!("preset {preset:?} rejected: {error}"));
        }
    }

    #[test]
    fn invalid_background_is_rejected() {
        let config = Config {
            background: Some(Color::rgba(1, 2, 3, 4)),
            ..Config::default()
        };
        assert!(matches!(
            Config::try_from(&config),
            Err(Error::InvalidConfig { field: "background" })
        ));
    }

    #[test]
    fn zero_max_colors_is_rejected() {
        let config = Config {
            color_mode: ColorMode::Auto {
                max_colors: 0,
                refine_iterations: 0,
            },
            ..Config::default()
        };
        assert!(matches!(
            Config::try_from(&config),
            Err(Error::InvalidConfig { field: "max_colors" })
        ));
    }

    #[test]
    fn binary_options_validate_through_try_from() {
        let options = BinaryOptions::new(BinaryThreshold::Adaptive { radius: 0, offset: 0 });
        assert!(BinaryOptions::try_from(&options).is_err());
        assert!(BinaryOptions::try_new(BinaryThreshold::Global(90)).is_ok());
    }

    #[test]
    fn fit_settings_constructors_validate() {
        assert!(FitSettings::try_new(0.5, CornerPolicy::default(), 0.45, 0.25, 12, true, 0.0).is_ok());
        assert!(FitSettings::try_new(0.5, CornerPolicy::default(), 0.0, 0.25, 12, true, 0.0).is_err());
        assert!(CornerPolicy::try_new(200.0, 0.0).is_err());
        assert_eq!(
            CornerPolicy::try_from((70.0, 0.0)).expect("corner policy"),
            CornerPolicy::default()
        );
    }

    #[test]
    fn builders_replace_only_the_selected_field() {
        let palette = Palette::from_slice(&[Color::BLACK, Color::WHITE]).expect("palette");
        let config = Config::default()
            .with_palette(palette.clone())
            .with_background(Color::rgb(1, 2, 3))
            .with_max_colors(4)
            .with_corners(CornerPolicy::sharp());
        assert_eq!(
            config.color_mode,
            ColorMode::Auto {
                max_colors: 4,
                refine_iterations: 4
            }
        );
        assert_eq!(config.background, Some(Color::rgb(1, 2, 3)));
        assert_eq!(config.fit.corners, CornerPolicy::sharp());
    }

    #[test]
    fn preset_conversion_matches_constructor() {
        assert_eq!(Config::from(Preset::Logo), Config::logo());
        assert_eq!(Config::from(Preset::Photo), Config::photo());
        assert_eq!(
            Config::from(Preset::Binary(BinaryThreshold::Global(64))),
            Config::binary(BinaryThreshold::Global(64))
        );
    }

    #[test]
    fn transparency_builder_clears_background() {
        let config = Config::default()
            .with_background(Color::WHITE)
            .with_transparency();
        assert_eq!(config.background, None);
    }
}
