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

//! Bitmap to color vector graphics.
//!
//! `codevar-bmpvector` turns a decoded raster image into a structured vector
//! document. It performs no image decoding and emits no SVG, HTML or XML: the
//! output is a document of filled shapes whose paths contain absolute line and
//! cubic segments. Every shape carries its fill color, so color artwork is
//! preserved rather than reduced to a monochrome outline.
//!
//! # Coordinate convention
//!
//! `x` grows right, `y` grows down, and a pixel `(x, y)` covers
//! `[x, x + 1) x [y, y + 1)`. All paths therefore live in
//! `[0, width] x [0, height]`.
//!
//! # Determinism
//!
//! Every stage iterates in a deterministic order and uses no hash-based
//! iteration or floating-point randomness. Tracing the same image with the same
//! configuration produces the same document on every run.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod bezier;
pub mod color;
pub mod component;
pub mod contour;
pub mod error;
pub mod fit;
pub mod geom;
pub mod image;
pub mod optimize;
pub mod palette;
pub mod polygon;
pub mod quantize;
pub mod segment;
pub mod simplify;
pub mod vector;

pub use color::{Color, Oklab};
pub use component::{Component, Connectivity, DespeckleOptions, Segmentation};
pub use error::{Error, Result};
pub use fit::{CornerPolicy, FitSettings};
pub use geom::{Bounds, CubicBezier, PixelRect, Point};
pub use image::{ColorImage, IntegralImage};
pub use optimize::OptimizeSettings;
pub use palette::{BinaryOptions, BinaryThreshold, ColorMode, IndexedImage, Palette};
pub use vector::{DocumentStats, Fill, FillRule, Path, Segment, Shape, SubPath, VectorDoc};
