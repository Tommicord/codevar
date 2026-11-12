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

//! Palette types and the indexed raster they produce.
//!
//! The palette describes how many colors the tracer may use and which
//! colors are allowed. Four strategies are available:
//!
//! * [`ColorMode::Exact`] keeps every distinct color in the image;
//! * [`ColorMode::Fixed`] uses a caller-supplied palette;
//! * [`ColorMode::Auto`] performs weighted median-cut followed by optional
//!   k-means refinement in Oklab;
//! * [`ColorMode::Binary`] thresholds luma globally, with Otsu's method or with
//!   an adaptive local-mean threshold.
//!
//! The output is an [`IndexedImage`]: one palette index per pixel. A dedicated
//! sentinel marks pixels that are transparent and must not produce geometry.
//! Every constructor validates its input, so an invalid palette or threshold
//! is rejected before any pixel is read.

use alloc::vec;
use alloc::vec::Vec;

use crate::color::Color;
use crate::error::{Error, Result};

/// Sentinel index used for pixels that are not part of any shape.
pub const NO_INDEX: u32 = u32::MAX;

/// Largest palette this crate will materialize.
///
/// The bound keeps the per-pixel index map and the nearest-color scans within a
/// predictable memory budget. Exact mode returns [`Error::PaletteTooLarge`] for
/// images that exceed it; callers can switch to [`ColorMode::Auto`] instead.
pub const MAX_PALETTE_ENTRIES: usize = 65_536;

/// How the tracer chooses colors.
#[derive(Clone, Debug, PartialEq)]
pub enum ColorMode {
    /// Select at most `max_colors` colors with weighted median cut and refine
    /// them with `refine_iterations` k-means updates.
    Auto {
        /// Maximum number of palette entries. Must be in
        /// `1..=MAX_PALETTE_ENTRIES`.
        max_colors: usize,
        /// Number of Lloyd iterations after the median-cut initialization.
        /// Zero keeps the median-cut representatives.
        refine_iterations: u8,
    },
    /// Use a caller-supplied palette exactly as given.
    Fixed(Palette),
    /// Keep every distinct color in the image.
    Exact,
    /// Reduce to a foreground and a background color using a luma threshold.
    Binary(BinaryOptions),
}

impl Default for ColorMode {
    fn default() -> Self {
        Self::Auto {
            max_colors: 16,
            refine_iterations: 4,
        }
    }
}

impl TryFrom<&ColorMode> for ColorMode {
    type Error = Error;

    fn try_from(mode: &ColorMode) -> Result<Self> {
        match mode {
            ColorMode::Auto { max_colors, .. } => {
                if *max_colors == 0 || *max_colors > MAX_PALETTE_ENTRIES {
                    return Err(Error::InvalidConfig { field: "max_colors" });
                }
                Ok(mode.clone())
            }
            ColorMode::Fixed(palette) => {
                if palette.is_empty() {
                    return Err(Error::EmptyPalette);
                }
                Ok(mode.clone())
            }
            ColorMode::Exact => Ok(ColorMode::Exact),
            ColorMode::Binary(options) => {
                BinaryOptions::try_from(options)?;
                Ok(mode.clone())
            }
        }
    }
}
/// Threshold selection for [`ColorMode::Binary`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BinaryThreshold {
    /// A fixed cutoff in `0..=255`. A pixel is foreground when its luma is
    /// strictly less than the cutoff.
    Global(u8),
    /// Otsu's between-class variance maximization over the image histogram.
    #[default]
    Otsu,
    /// Compare each pixel with the mean luma of a square window.
    ///
    /// A pixel is foreground when `luma < local_mean + offset`. A positive
    /// offset therefore selects more foreground, and a negative offset selects
    /// less. `radius` is in pixels and must be at least one.
    Adaptive {
        /// Half-size of the square mean window.
        radius: u32,
        /// Bias added to the local mean before comparison.
        offset: i16,
    },
}

/// Colors and threshold used by binary tracing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryOptions {
    /// Threshold strategy.
    pub threshold: BinaryThreshold,
    /// Paint assigned to foreground pixels.
    pub foreground: Color,
    /// Paint assigned to background pixels.
    pub background: Color,
}

impl Default for BinaryOptions {
    fn default() -> Self {
        Self {
            threshold: BinaryThreshold::Otsu,
            foreground: Color::BLACK,
            background: Color::WHITE,
        }
    }
}

impl BinaryOptions {
    /// Binary tracing with the default black foreground and white background.
    #[inline]
    pub const fn new(threshold: BinaryThreshold) -> Self {
        Self {
            threshold,
            foreground: Color::BLACK,
            background: Color::WHITE,
        }
    }

    /// Builds binary options after checking the threshold parameters.
    pub fn try_new(threshold: BinaryThreshold) -> Result<Self> {
        let options = Self::new(threshold);
        Self::try_from(&options)
    }
}

impl TryFrom<&BinaryOptions> for BinaryOptions {
    type Error = Error;

    fn try_from(options: &BinaryOptions) -> Result<Self> {
        if let BinaryThreshold::Adaptive { radius, .. } = options.threshold
            && radius == 0
        {
            return Err(Error::InvalidConfig {
                field: "binary_radius",
            });
        }
        Ok(*options)
    }
}

/// A validated list of paint colors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Palette {
    colors: Vec<Color>,
}

impl Palette {
    /// Builds a palette from a non-empty color list.
    pub fn new(colors: Vec<Color>) -> Result<Self> {
        if colors.is_empty() {
            return Err(Error::EmptyPalette);
        }
        if colors.len() > MAX_PALETTE_ENTRIES {
            return Err(Error::PaletteTooLarge {
                entries: colors.len(),
            });
        }
        Ok(Self { colors })
    }

    /// Builds a palette from a slice.
    pub fn from_slice(colors: &[Color]) -> Result<Self> {
        Self::new(colors.to_vec())
    }

    /// An empty palette, used only for fully transparent images.
    pub(crate) fn empty() -> Self {
        Self { colors: Vec::new() }
    }

    /// Builds a palette from colors already produced by the quantizer.
    ///
    /// The length is validated by the quantizer before this constructor is
    /// called; the debug assertion documents the internal invariant.
    pub(crate) fn from_quantizer(colors: Vec<Color>) -> Self {
        debug_assert!(colors.len() <= MAX_PALETTE_ENTRIES);
        Self { colors }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Whether the palette contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Read-only view of the entries.
    #[inline]
    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    /// The entry at `index`, or `None` when it is out of range.
    #[inline]
    pub fn get(&self, index: usize) -> Option<Color> {
        self.colors.get(index).copied()
    }

    /// Iterates over the entries in palette order.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, Color> {
        self.colors.iter()
    }

    /// Finds the perceptually nearest entry.
    ///
    /// Ties are broken by the lower palette index, which keeps assignment
    /// deterministic. Returns `None` only for an empty palette.
    pub fn nearest(&self, color: Color) -> Option<(usize, f64)> {
        let target = color.to_oklab();
        let mut best_index = 0;
        let mut best_distance = f64::INFINITY;
        for (index, entry) in self.colors.iter().enumerate() {
            let distance = target.distance_sq(entry.to_oklab());
            if distance < best_distance {
                best_distance = distance;
                best_index = index;
            }
        }
        if best_distance.is_finite() {
            Some((best_index, best_distance))
        } else {
            None
        }
    }
}

impl TryFrom<&[Color]> for Palette {
    type Error = Error;

    fn try_from(colors: &[Color]) -> Result<Self> {
        Self::new(colors.to_vec())
    }
}

impl TryFrom<Vec<Color>> for Palette {
    type Error = Error;

    fn try_from(colors: Vec<Color>) -> Result<Self> {
        Self::new(colors)
    }
}

/// A raster where every pixel stores a palette index.
///
/// [`NO_INDEX`] marks fully transparent pixels, which never produce shapes.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexedImage {
    width: u32,
    height: u32,
    indices: Vec<u32>,
    palette: Palette,
}

impl IndexedImage {
    /// Image width in pixels.
    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Number of pixels.
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.indices.len()
    }

    /// The palette used by the indices.
    #[inline]
    pub fn palette(&self) -> &Palette {
        &self.palette
    }

    /// Read-only view of the index map.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Mutable view of the index map, used by the despeckle pass.
    #[inline]
    pub fn indices_mut(&mut self) -> &mut [u32] {
        &mut self.indices
    }

    /// The index at `(x, y)`, or [`NO_INDEX`] when out of range.
    #[inline]
    pub fn index(&self, x: u32, y: u32) -> u32 {
        if x >= self.width || y >= self.height {
            return NO_INDEX;
        }
        self.indices[y as usize * self.width as usize + x as usize]
    }

    /// Replaces the index at `(x, y)`.
    #[inline]
    pub fn set_index(&mut self, x: u32, y: u32, index: u32) {
        if x < self.width && y < self.height {
            self.indices[y as usize * self.width as usize + x as usize] = index;
        }
    }

    /// The paint at `(x, y)`, or `None` for transparent and out-of-range
    /// pixels.
    pub fn color_at(&self, x: u32, y: u32) -> Option<Color> {
        let index = self.index(x, y);
        if index == NO_INDEX {
            None
        } else {
            self.palette.get(index as usize)
        }
    }

    /// Number of pixels marked [`NO_INDEX`].
    pub fn transparent_pixels(&self) -> usize {
        self.indices
            .iter()
            .filter(|&&index| index == NO_INDEX)
            .count()
    }

    /// Whether the index map contains no visible pixel.
    pub fn is_empty(&self) -> bool {
        self.palette.is_empty()
            || self
                .indices
                .iter()
                .all(|&index| index == NO_INDEX)
    }

    /// Builds the empty result for a fully transparent image.
    pub(crate) fn empty(width: u32, height: u32) -> Self {
        Self::from_parts(
            width,
            height,
            vec![NO_INDEX; width as usize * height as usize],
            Palette::empty(),
        )
    }

    /// Builds an indexed image from validated parts.
    ///
    /// Used by the despeckle tests and by internal callers that already hold a
    /// palette. The pixel count and palette indices are checked by the caller.
    pub(crate) fn from_parts(width: u32, height: u32, indices: Vec<u32>, palette: Palette) -> Self {
        debug_assert_eq!(indices.len(), width as usize * height as usize);
        Self {
            width,
            height,
            indices,
            palette,
        }
    }

    /// Builds an indexed image from a palette the quantizer produced.
    ///
    /// The palette indices are already validated by the caller, so this
    /// constructor only checks the pixel count.
    pub(crate) fn from_quantizer(width: u32, height: u32, indices: Vec<u32>, palette: Palette) -> Self {
        Self::from_parts(width, height, indices, palette)
    }
}
