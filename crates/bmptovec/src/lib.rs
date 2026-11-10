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

//! # codevar-bmptovec
//!
//! Convert a raw bitmap of pixels into an SVG vector document.
//!
//! This is a faithful port of the [visioncortex/vtracer](https://github.com/visioncortex/vtracer)
//! raster-to-vector pipeline, adapted to the Codevar workspace. It performs no
//! image decoding and emits no HTML — it takes a decoded [`ColorImage`] of raw
//! pixels and produces a [`VectorDoc`] (a struct containing all the
//! mathematics of the vector map) plus an [`SvgWriter`] that serializes it to
//! an SVG string.
//!
//! ## The pipeline
//!
//! ```text
//! image ──▶ Frontend ──▶ ColorFitter ──▶ Compositing ──▶ Optimizer ──▶ SvgWriter ──▶ SVG
//! ```
//!
//! Two intermediate representations carry the work between stages:
//!
//! * a [`Segmentation`] — the frontend's output: flat-color regions (paint
//!   layers) over the canvas;
//! * a [`VectorDoc`] — the output document: resolved shapes with fitted vector
//!   paths, which the optimizer passes shrink and the writer serializes.
//!
//! ## Quick start
//!
//! ```rust
//! use codevar_bmptovec::{ColorImage, Config};
//! # fn load() -> ColorImage { todo!() }
//! let img: ColorImage = load();
//!
//! // one-shot: raw bitmap → SVG string, all defaults
//! let svg = Config::default().build()?.to_svg(&img)?;
//! # Ok::<(), codevar_bmptovec::Error>(())
//! ```

pub mod colorfit;
pub mod compose;
pub mod config;
pub mod error;
pub mod fitter;
pub mod frontend;
pub mod ir;
pub mod mosaic;
pub mod optimize;
pub mod pipeline;
pub mod progress;
pub mod session;
pub mod simplify;
pub mod svg;
pub mod types;

pub use colorfit::{AutoQuantize, ColorFitter, FixedPalette, Identity, MergeAdjacent};
pub use compose::Compositing;
pub use config::{Clustering, Config, FitMode, Hierarchical, Preset, SegmentKey};
pub use error::Error;
pub use fitter::{CurveFitter, FitParams, PixelFitter, PolygonFitter, SplineFitter};
pub use frontend::{BinaryFrontend, Frontend, Threshold, WatershedFrontend};
pub use ir::{Paint, RegionMask, Segmentation, VectorDoc};
pub use mosaic::{PixelSegmentFitter, PolygonSegmentFitter, SegmentFitter, SplineSegmentFitter};
pub use optimize::{CleanupPass, OptimizerPass, QuantizePass};
pub use pipeline::Pipeline;
pub use progress::{CancelToken, Phase, Progress};
pub use session::Session;
pub use simplify::{CurvePass, SimplifyCurves};
pub use svg::SvgWriter;

// Re-export the value types callers need at the boundary.
pub use types::{BinaryImage, Color, ColorImage, PointF64, PointI32, SummedAreaTable};
