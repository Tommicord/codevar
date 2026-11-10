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

//! Value types used at the crate boundary.
//!
//! These are self-contained ports of the visioncortex value types so the
//! crate does not depend on an external image-decoding crate. No image
//! decoding happens here — callers hand over a decoded [`ColorImage`] of raw
//! pixels.

use core::fmt;

/// An opaque RGB color with full 8-bit channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    /// A new opaque RGB color.
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Grayscale intensity (0..=255), the metric every thresholding method
    /// uses so fixed and adaptive thresholds agree on what "dark" means.
    #[inline]
    pub fn intensity(self) -> u32 {
        (self.r as u32 + self.g as u32 + self.b as u32) / 3
    }

    /// Hex string `#rrggbb` (lowercase), used by the SVG writer for fills.
    #[inline]
    pub fn to_hex_string(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

/// A 2D point in floating-point (document) space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PointF64 {
    pub x: f64,
    pub y: f64,
}

impl PointF64 {
    /// A new point.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A 2D point in integer (pixel) space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct PointI32 {
    pub x: i32,
    pub y: i32,
}

impl PointI32 {
    /// A new point.
    #[inline]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// A decoded raster image: rows of `Color`, top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorImage {
    pub width: usize,
    pub height: usize,
    /// Row-major pixels, `pixels[y * width + x]`.
    pub pixels: Vec<Color>,
}

impl ColorImage {
    /// A new image from width, height, and row-major pixels.
    #[inline]
    pub fn new(width: usize, height: usize, pixels: Vec<Color>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The pixel at `(x, y)`. Panics only on out-of-bounds access (a caller
    /// bug); production callers should guarantee bounds.
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> Color {
        self.pixels[y * self.width + x]
    }

    /// Binarize into a foreground mask using `predicate`.
    pub fn to_binary_image<F>(&self, predicate: F) -> BinaryImage
    where
        F: Fn(Color) -> bool,
    {
        let mut out = BinaryImage::new_w_h(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                out.set_pixel(x, y, predicate(self.get_pixel(x, y)));
            }
        }
        out
    }
}

/// A binary (foreground/background) image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryImage {
    pub width: usize,
    pub height: usize,
    /// Row-major bits, one byte per pixel for simplicity and speed.
    pub data: Vec<u8>,
}

impl BinaryImage {
    /// A new all-false image.
    #[inline]
    pub fn new_w_h(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    /// Whether `(x, y)` is foreground.
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> bool {
        self.data[y * self.width + x] != 0
    }

    /// Set `(x, y)` to foreground (`true`) or background (`false`).
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, value: bool) {
        self.data[y * self.width + x] = if value { 1 } else { 0 };
    }
}

/// A summed-area table (integral image) over a color image's intensities,
/// enabling O(1) rectangular sum queries for adaptive thresholding.
#[derive(Clone, Debug)]
pub struct SummedAreaTable {
    pub width: usize,
    pub height: usize,
    /// Row-major `u32` sums, `(width + 1) * (height + 1)` entries with a
    /// zero border row/column so region queries never branch.
    data: Vec<u32>,
}

impl SummedAreaTable {
    /// Build the SAT from a color image, summing each pixel's intensity.
    pub fn from_color_image(img: &ColorImage) -> Self {
        let (w, h) = (img.width, img.height);
        let mut data = vec![0u32; (w + 1) * (h + 1)];
        // prefix sums: data[(y+1)*(w+1) + (x+1)] = intensity + left + top - diag
        for y in 0..h {
            let row_offset = (y + 1) * (w + 1);
            let prev_row_offset = y * (w + 1);
            let mut row_sum = 0u32;
            for x in 0..w {
                row_sum += img.get_pixel(x, y).intensity();
                data[row_offset + (x + 1)] = row_sum + data[prev_row_offset + (x + 1)];
            }
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }

    /// Sum of intensities in the rectangle `[x0, x0+w) × [y0, y0+h)`.
    #[inline]
    pub fn get_region_sum_x_y_w_h(&self, x0: usize, y0: usize, w: usize, h: usize) -> u32 {
        let stride = self.width + 1;
        let x1 = x0 + w;
        let y1 = y0 + h;
        let a = self.data[y0 * stride + x0];
        let b = self.data[y0 * stride + x1];
        let c = self.data[y1 * stride + x0];
        let d = self.data[y1 * stride + x1];
        d - b - c + a
    }
}
