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

//! The types that describe a connected region of an indexed raster.
//!
//! Once every pixel has a palette index, the next step is to group adjacent
//! pixels of the same index into components. A [`Component`] records the
//! palette index, the pixel bounds, the area and the packed pixel offsets, and
//! a [`Segmentation`] holds the components of a whole image.
//!
//! The labels are deterministic and follow raster order, so the same image
//! always produces the same component identifiers.

use alloc::vec;
use alloc::vec::Vec;

use crate::color::Color;
use crate::geom::PixelRect;
use crate::palette::Palette;

/// Pixel adjacency used by component labeling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Connectivity {
    /// Only horizontal and vertical neighbours are connected.
    Four,
    /// Diagonal neighbours also connect.
    #[default]
    Eight,
}

impl Connectivity {
    /// Number of neighbours considered.
    #[inline]
    pub const fn count(self) -> usize {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Offsets of the neighbours that precede a pixel in raster order.
    ///
    /// The union-find pass only needs to inspect already-visited neighbours;
    /// the remaining directions are covered when they are reached later.
    #[inline]
    pub const fn preceding_offsets(self) -> &'static [(i32, i32)] {
        match self {
            Self::Four => &[(-1, 0), (0, -1)],
            Self::Eight => &[(-1, 0), (0, -1), (-1, -1), (1, -1)],
        }
    }

    /// All four-connected neighbour offsets.
    #[inline]
    pub const fn four_offsets() -> &'static [(i32, i32)] {
        &[(-1, 0), (1, 0), (0, -1), (0, 1)]
    }
}

/// One connected region of equal palette index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// Palette index shared by every pixel in the component.
    pub palette_index: u32,
    /// Axis-aligned pixel bounds, with exclusive right and bottom edges.
    pub bounds: PixelRect,
    /// Number of pixels in the component.
    pub area: u64,
    /// Packed row-major pixel offsets (`y * width + x`).
    ///
    /// Keeping the pixel list lets contour extraction build a compact local
    /// mask without rescanning the whole image for each component.
    pub pixels: Vec<u32>,
}

impl Component {
    /// A component with a single pixel.
    pub(crate) fn single(palette_index: u32, x: u32, y: u32, packed: u32) -> Self {
        Self {
            palette_index,
            bounds: PixelRect::single(x, y),
            area: 1,
            pixels: vec![packed],
        }
    }

    /// Appends a pixel that belongs to this component.
    pub(crate) fn push(&mut self, x: u32, y: u32, packed: u32) {
        self.bounds.include(x, y);
        self.area += 1;
        self.pixels.push(packed);
    }

    /// Whether `(x, y)` belongs to the component's bounding rectangle.
    #[inline]
    pub fn bounds_contain(&self, x: u32, y: u32) -> bool {
        self.bounds.contains(x, y)
    }

    /// The paint color of the component.
    #[inline]
    pub fn color(&self, palette: &Palette) -> Option<Color> {
        palette.get(self.palette_index as usize)
    }

    /// Width of the component's bounding box.
    #[inline]
    pub fn width(&self) -> u32 {
        self.bounds.width()
    }

    /// Height of the component's bounding box.
    #[inline]
    pub fn height(&self) -> u32 {
        self.bounds.height()
    }

    /// Approximate compactness, useful for ordering or diagnostics.
    ///
    /// The value is `area / bounding_box_area` in `(0, 1]`.
    pub fn fill_ratio(&self) -> f64 {
        let box_area = self.bounds.area();
        if box_area == 0 {
            0.0
        } else {
            self.area as f64 / box_area as f64
        }
    }
}

/// All components of an indexed image plus the palette used to paint them.
#[derive(Clone, Debug, PartialEq)]
pub struct Segmentation {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Components in first-pixel raster order.
    pub components: Vec<Component>,
    /// Palette referenced by the components.
    pub palette: Palette,
}

impl Segmentation {
    /// Number of components.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether there are no components.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Total number of pixels covered by all components.
    pub fn total_area(&self) -> u64 {
        self.components
            .iter()
            .map(|component| component.area)
            .sum()
    }

    /// The paint color of a component, or `None` for an invalid index.
    #[inline]
    pub fn color_of(&self, component: &Component) -> Option<Color> {
        component.color(&self.palette)
    }

    /// Sorts components from largest to smallest source area.
    ///
    /// The sort is stable, so equal-area components keep their raster order.
    /// Painting large areas first lets smaller details remain visible where
    /// independently fitted curves meet.
    pub fn sort_by_area_descending(&mut self) {
        self.components.sort_by(|first, second| {
            second
                .area
                .cmp(&first.area)
                .then_with(|| first.bounds.top.cmp(&second.bounds.top))
                .then_with(|| first.bounds.left.cmp(&second.bounds.left))
        });
    }

    /// Iterates over components with their resolved colors.
    pub fn colored_components(&self) -> impl Iterator<Item = (&Component, Color)> {
        self.components.iter().filter_map(|component| {
            component
                .color(&self.palette)
                .map(|color| (component, color))
        })
    }
}

/// Options for the small-component cleanup pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DespeckleOptions {
    /// Components with fewer pixels than this are repainted with a neighbour.
    ///
    /// Zero disables the pass.
    pub minimum_area: u32,
    /// Maximum number of cleanup passes. Each pass re-labels the image, so a
    /// region that shrinks below the threshold after a neighbour was removed
    /// is handled on the next pass.
    pub passes: u8,
}

impl Default for DespeckleOptions {
    fn default() -> Self {
        Self {
            minimum_area: 4,
            passes: 2,
        }
    }
}

impl DespeckleOptions {
    /// A configuration that leaves the index map untouched.
    #[inline]
    pub const fn disabled() -> Self {
        Self {
            minimum_area: 0,
            passes: 0,
        }
    }

    /// Whether the pass would do any work.
    #[inline]
    pub const fn is_enabled(self) -> bool {
        self.minimum_area > 0 && self.passes > 0
    }
}

/// Result of a despeckle run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DespeckleReport {
    /// Number of passes that actually changed the index map.
    pub passes_run: u8,
    /// Number of components removed or merged into a neighbour.
    pub removed_components: u64,
    /// Number of pixels repainted.
    pub reassigned_pixels: u64,
}
