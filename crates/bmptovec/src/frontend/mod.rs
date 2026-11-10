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

//! Frontend stages: turn a decoded [`ColorImage`] into a [`Segmentation`].
//!
//! Two frontends are provided:
//! * [`BinaryFrontend`] — threshold the image, label connected components,
//!   and emit one black paint layer per cluster.
//! * [`WatershedFrontend`] — hierarchical region-growing watershed.
//!
//! The shared color-cluster runner and keying helpers live in
//! [`color_cluster`] and [`keying`] and are used by the color-based pipeline.

pub mod binary;
pub mod color_cluster;
mod keying;
pub mod watershed;

pub use binary::{BinaryFrontend, Threshold};
pub use watershed::{WatershedFrontend, WatershedHierarchy};

use crate::error::Error;
use crate::ir::Segmentation;
use crate::progress::Ctx;
use crate::types::ColorImage;

/// A stage that converts a raster image into flat-color region layers.
pub trait Frontend {
    /// Segment `img` into paint layers with no progress reporting.
    fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error>;

    /// Segment `img` with cancellation and progress reporting.
    ///
    /// The default implementation calls [`Frontend::segment`], then checks
    /// cancellation and reports 100% for the [`Phase::Segment`] phase.
    fn segment_with(&self, img: &ColorImage, ctx: &mut Ctx<'_>) -> Result<Segmentation, Error> {
        let seg = self.segment(img)?;
        ctx.check()?;
        ctx.report(crate::progress::Phase::Segment, 1.0);
        Ok(seg)
    }
}
