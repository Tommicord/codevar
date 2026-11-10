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
//! permissions and limitations under the language.

//! Pipeline configuration: frontends, color fitters, fit mode, optimizers,
//! and SVG writer settings, assembled into a [`Pipeline`] by [`Config::build`].
//!
//! Faithful port of visioncortex/vtracer's `crates/vtracer/src/config.rs`
//! (with `visioncortex::Color` replaced by `crate::types::Color`).

use crate::colorfit::{AutoQuantize, ColorFitter, Identity, MergeAdjacent};
use crate::compose::Compositing;
use crate::error::Error;
use crate::fitter::{CurveFitter, FitParams, PixelFitter, PolygonFitter, SplineFitter};
use crate::frontend::{BinaryFrontend, Frontend, Threshold, WatershedFrontend};
use crate::mosaic::{PixelSegmentFitter, PolygonSegmentFitter, SegmentFitter, SplineSegmentFitter};
use crate::optimize::{CleanupPass, OptimizerPass, QuantizePass};
use crate::pipeline::Pipeline;
use crate::simplify::{CurvePass, SimplifyCurves};
use crate::svg::SvgWriter;
use std::str::FromStr;

/// Clustering strategy used by the frontend to form regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clustering {
    /// Color-cluster the image into flat regions.
    ColorCluster,
    /// Binary threshold into foreground/background.
    Binary,
    /// Watershed region growing.
    Watershed,
}

/// Hierarchical layer composition strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hierarchical {
    /// Stacked layers (painter's algorithm).
    Stacked,
    /// Cutout layers (holes punched out of lower layers).
    Cutout,
}

/// Curve fitting mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    /// Fit each pixel as a tiny polygon (pixel mode).
    Pixel,
    /// Fit each region as a polygon.
    Polygon,
    /// Fit each region as cubic splines.
    Spline,
}

/// Quality/size preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// Black-and-white line art.
    Bw,
    /// Posterized flat colors.
    Poster,
    /// Photographic / continuous tone.
    Photo,
}

/// Key describing a segmentation request, used by [`Session`] to detect when
/// re-segmentation is required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentKey {
    pub width: u32,
    pub height: u32,
    pub clustering: Clustering,
    pub hierarchical: Hierarchical,
    pub threshold: u8,
}

impl SegmentKey {
    pub fn new(
        width: u32,
        height: u32,
        clustering: Clustering,
        hierarchical: Hierarchical,
        threshold: u8,
    ) -> Self {
        Self {
            width,
            height,
            clustering,
            hierarchical,
            threshold,
        }
    }
}

/// Full pipeline configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub clustering: Clustering,
    pub hierarchical: Hierarchical,
    pub fit_mode: FitMode,
    pub preset: Preset,
    pub threshold: u8,
    pub block_size: u16,
    pub filter_speckle: bool,
    pub speckle_size: u32,
    pub color_quantization: u8,
    pub corner_threshold: f64,
    pub line_threshold: f64,
    pub curve_threshold: f64,
    pub splice_threshold: f64,
    pub tangent_threshold: f64,
    pub iterative_approx: bool,
    pub max_iterations: u32,
    pub precision: u32,
    pub svg_relative: bool,
    pub svg_shorthands: bool,
    pub svg_precision: Option<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self::from_preset(Preset::Bw)
    }
}

impl Config {
    /// Build a config from a preset with sensible defaults.
    pub fn from_preset(preset: Preset) -> Self {
        match preset {
            Preset::Bw => Self {
                clustering: Clustering::Binary,
                hierarchical: Hierarchical::Stacked,
                fit_mode: FitMode::Pixel,
                preset: Preset::Bw,
                threshold: 127,
                block_size: 5,
                filter_speckle: true,
                speckle_size: 10,
                color_quantization: 0,
                corner_threshold: 60.0,
                line_threshold: 8.0,
                curve_threshold: 3.0,
                splice_threshold: 1.0,
                tangent_threshold: 3.0,
                iterative_approx: true,
                max_iterations: 10,
                precision: 3,
                svg_relative: true,
                svg_shorthands: true,
                svg_precision: Some(3),
            },
            Preset::Poster => Self {
                clustering: Clustering::ColorCluster,
                hierarchical: Hierarchical::Stacked,
                fit_mode: FitMode::Polygon,
                preset: Preset::Poster,
                threshold: 127,
                block_size: 5,
                filter_speckle: true,
                speckle_size: 10,
                color_quantization: 8,
                corner_threshold: 60.0,
                line_threshold: 8.0,
                curve_threshold: 3.0,
                splice_threshold: 1.0,
                tangent_threshold: 3.0,
                iterative_approx: true,
                max_iterations: 10,
                precision: 3,
                svg_relative: true,
                svg_shorthands: true,
                svg_precision: Some(3),
            },
            Preset::Photo => Self {
                clustering: Clustering::Watershed,
                hierarchical: Hierarchical::Stacked,
                fit_mode: FitMode::Spline,
                preset: Preset::Photo,
                threshold: 127,
                block_size: 5,
                filter_speckle: true,
                speckle_size: 10,
                color_quantization: 16,
                corner_threshold: 60.0,
                line_threshold: 8.0,
                curve_threshold: 3.0,
                splice_threshold: 1.0,
                tangent_threshold: 3.0,
                iterative_approx: true,
                max_iterations: 10,
                precision: 3,
                svg_relative: true,
                svg_shorthands: true,
                svg_precision: Some(3),
            },
        }
    }

    /// Convert degrees to radians.
    #[inline]
    pub fn deg2rad(deg: f64) -> f64 {
        deg * std::f64::consts::PI / 180.0
    }

    /// Build the fit parameters for this config's fit mode.
    pub fn fit_params(&self) -> FitParams {
        FitParams {
            corner_threshold: self.corner_threshold,
            length_threshold: self.line_threshold,
            max_iterations: self.max_iterations as usize,
            splice_threshold: self.splice_threshold,
        }
    }

    /// Build the frontend for this config.
    pub fn frontend(&self) -> Box<dyn Frontend> {
        let min_area = self.speckle_area() as usize;
        match self.clustering {
            Clustering::Binary => Box::new(BinaryFrontend {
                threshold: Threshold::Fixed(self.threshold),
                diagonal: false,
                min_area,
            }),
            Clustering::ColorCluster => Box::new(BinaryFrontend {
                threshold: Threshold::Fixed(self.threshold),
                diagonal: false,
                min_area,
            }),
            Clustering::Watershed => Box::new(WatershedFrontend { detail: 8, min_area }),
        }
    }

    /// Speckle filtering area threshold.
    #[inline]
    pub fn speckle_area(&self) -> u32 {
        self.speckle_size
    }

    /// Build the color fitters for this config.
    pub fn color_fitters(&self) -> Vec<Box<dyn ColorFitter>> {
        match self.preset {
            Preset::Bw => vec![Box::new(Identity)],
            Preset::Poster => {
                let n = self.color_quantization.max(1) as usize;
                vec![Box::new(AutoQuantize::new(n)), Box::new(MergeAdjacent)]
            }
            Preset::Photo => {
                let n = self.color_quantization.max(2) as usize;
                vec![Box::new(AutoQuantize::new(n)), Box::new(MergeAdjacent)]
            }
        }
    }

    /// Build the segment fitter for this config's fit mode.
    pub fn fitter(&self) -> Box<dyn SegmentFitter> {
        match self.fit_mode {
            FitMode::Pixel => Box::new(PixelSegmentFitter),
            FitMode::Polygon => Box::new(PolygonSegmentFitter),
            FitMode::Spline => Box::new(SplineSegmentFitter),
        }
    }

    /// Build the segment fitter (alias for compositing).
    pub fn segment_fitter(&self) -> Box<dyn SegmentFitter> {
        self.fitter()
    }

    /// Build the region (curve) fitter for stacked compositing mode.
    pub fn curve_fitter(&self) -> Box<dyn CurveFitter> {
        match self.fit_mode {
            FitMode::Pixel => Box::new(PixelFitter),
            FitMode::Polygon => Box::new(PolygonFitter),
            FitMode::Spline => Box::new(SplineFitter::new(self.fit_params())),
        }
    }

    /// Build the curve simplification passes.
    pub fn curve_passes(&self) -> Vec<Box<dyn CurvePass>> {
        vec![Box::new(SimplifyCurves {
            tolerance: 2.0,
            corner_threshold: self.corner_threshold,
        })]
    }

    /// Build the optimizer passes.
    pub fn optimizers(&self) -> Vec<Box<dyn OptimizerPass>> {
        vec![
            Box::new(QuantizePass::new(self.precision)),
            Box::new(CleanupPass::new()),
        ]
    }

    /// Build the SVG writer.
    pub fn writer(&self) -> SvgWriter {
        SvgWriter {
            relative: self.svg_relative,
            shorthands: self.svg_shorthands,
            precision: self.svg_precision,
        }
    }

    /// Build the segment key for this config and canvas size.
    pub fn segment_key(&self, width: u32, height: u32) -> SegmentKey {
        SegmentKey::new(width, height, self.clustering, self.hierarchical, self.threshold)
    }

    /// Assemble a full [`Pipeline`] from this config.
    pub fn build(&self) -> Result<Pipeline, Error> {
        let compositing = match self.hierarchical {
            Hierarchical::Stacked => Compositing::Stacked(self.curve_fitter()),
            Hierarchical::Cutout => Compositing::Mosaic {
                fitter: self.segment_fitter(),
                merge_diff: 0,
            },
        };
        Ok(Pipeline::new(
            self.frontend(),
            self.color_fitters(),
            compositing,
            self.curve_passes(),
            self.optimizers(),
            self.writer(),
        ))
    }
}

impl FromStr for Clustering {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s.trim().to_lowercase().as_str() {
            "color-cluster" | "colorcluster" | "cc" => Ok(Clustering::ColorCluster),
            "binary" | "bin" => Ok(Clustering::Binary),
            "watershed" | "ws" => Ok(Clustering::Watershed),
            _ => Err(Error::Unsupported(format!("unknown clustering: {s}"))),
        }
    }
}

impl FromStr for Hierarchical {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s.trim().to_lowercase().as_str() {
            "stacked" => Ok(Hierarchical::Stacked),
            "cutout" => Ok(Hierarchical::Cutout),
            _ => Err(Error::Unsupported(format!("unknown hierarchical: {s}"))),
        }
    }
}

impl FromStr for FitMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s.trim().to_lowercase().as_str() {
            "pixel" => Ok(FitMode::Pixel),
            "polygon" => Ok(FitMode::Polygon),
            "spline" => Ok(FitMode::Spline),
            _ => Err(Error::Unsupported(format!("unknown fit mode: {s}"))),
        }
    }
}

impl FromStr for Preset {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s.trim().to_lowercase().as_str() {
            "bw" => Ok(Preset::Bw),
            "poster" => Ok(Preset::Poster),
            "photo" => Ok(Preset::Photo),
            _ => Err(Error::Unsupported(format!("unknown preset: {s}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_bw_preset() {
        assert_eq!(Config::default().preset, Preset::Bw);
    }

    #[test]
    fn from_preset_fields() {
        let cfg = Config::from_preset(Preset::Photo);
        assert_eq!(cfg.clustering, Clustering::Watershed);
        assert_eq!(cfg.fit_mode, FitMode::Spline);
        assert_eq!(cfg.color_quantization, 16);
    }

    #[test]
    fn from_str_roundtrip() {
        assert_eq!("binary".parse::<Clustering>().unwrap(), Clustering::Binary);
        assert_eq!("spline".parse::<FitMode>().unwrap(), FitMode::Spline);
        assert_eq!("photo".parse::<Preset>().unwrap(), Preset::Photo);
        assert!("nonsense".parse::<Preset>().is_err());
    }

    #[test]
    fn deg2rad_converts() {
        assert!((Config::deg2rad(180.0) - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn build_produces_pipeline() {
        let cfg = Config::default();
        let _pipe = cfg.build().expect("config builds");
    }
}
