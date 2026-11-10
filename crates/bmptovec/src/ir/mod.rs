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

//! Core intermediate representation shared by the pipeline stages.
//!
//! Two IRs flow through the pipeline:
//!
//! * [`Segmentation`] — the frontend output: ordered paint layers over a
//!   raster canvas (painter's algorithm, bottom to top). This is what the
//!   [`crate::colorfit`] stages rewrite.
//! * [`VectorDoc`] — the output document: resolved shapes with fitted paths.
//!   This is what the [`crate::optimize`] passes and the [`crate::svg`] writer
//!   operate on.

mod region;
mod vector;

pub use region::{Layer, RegionMask, Segmentation};
pub use vector::{MultiPath, PathCmd, Shape, SubPath, VectorDoc};

use crate::types::Color;

/// The final appearance of a region. Only solid colors are supported today;
/// the enum leaves room for gradients and patterns later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paint {
    Solid(Color),
}

impl Paint {
    /// The representative solid color of this paint.
    #[inline]
    pub fn color(&self) -> Color {
        match self {
            Paint::Solid(c) => *c,
        }
    }
}
