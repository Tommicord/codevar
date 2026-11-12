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

//! Decoded bitmap input and raster helpers.
//!
//! The crate does not decode PNG, JPEG, BMP or any other container format. It
//! starts from an already decoded [`ColorImage`], which keeps image-format
//! dependencies out of the workspace and lets callers hand over pixels from a
//! decoder they already trust. Several byte-slice constructors are provided for
//! the common interleaved layouts.
//!
//! Pixel coordinates are unsigned and every accessor validates bounds, so a
//! malformed image cannot cause an out-of-bounds read. The integral image is
//! used by adaptive thresholding and by the despeckle stage's area checks.

use alloc::vec;
use alloc::vec::Vec;

use crate::color::Color;
use crate::error::{Error, Result};
use crate::geom::PixelRect;

/// A decoded, row-major image with straight-alpha colors.
///
/// `pixels[y * width + x]` is the pixel whose top-left corner is `(x, y)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorImage {
    width: u32,
    height: u32,
    pixels: Vec<Color>,
}

impl ColorImage {
    /// Builds an image from its dimensions and row-major pixel buffer.
    ///
    /// Returns [`Error::EmptyImage`] for a zero extent,
    /// [`Error::ImageTooLarge`] when the dimensions cannot be represented on
    /// this target, and [`Error::PixelCountMismatch`] when the buffer length is
    /// wrong.
    pub fn new(width: u32, height: u32, pixels: Vec<Color>) -> Result<Self> {
        let expected = checked_pixel_count(width, height)?;
        if pixels.len() != expected {
            return Err(Error::PixelCountMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Builds a uniform image.
    pub fn filled(width: u32, height: u32, color: Color) -> Result<Self> {
        let count = checked_pixel_count(width, height)?;
        Ok(Self {
            width,
            height,
            pixels: vec![color; count],
        })
    }

    /// Builds an image by evaluating `generator` for every pixel.
    ///
    /// The generator receives `(x, y)` in row-major order.
    pub fn from_fn<F>(width: u32, height: u32, mut generator: F) -> Result<Self>
    where
        F: FnMut(u32, u32) -> Color,
    {
        let count = checked_pixel_count(width, height)?;
        let mut pixels = Vec::with_capacity(count);
        for y in 0..height {
            for x in 0..width {
                pixels.push(generator(x, y));
            }
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Builds an image from interleaved RGBA bytes.
    pub fn from_rgba8(width: u32, height: u32, bytes: &[u8]) -> Result<Self> {
        let count = checked_pixel_count(width, height)?;
        let expected = count
            .checked_mul(4)
            .ok_or(Error::ImageTooLarge { width, height })?;
        if bytes.len() != expected {
            return Err(Error::BufferLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        let pixels = bytes
            .chunks_exact(4)
            .map(|chunk| Color::rgba(chunk[0], chunk[1], chunk[2], chunk[3]))
            .collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Builds an image from interleaved RGB bytes, adding full alpha.
    pub fn from_rgb8(width: u32, height: u32, bytes: &[u8]) -> Result<Self> {
        let count = checked_pixel_count(width, height)?;
        let expected = count
            .checked_mul(3)
            .ok_or(Error::ImageTooLarge { width, height })?;
        if bytes.len() != expected {
            return Err(Error::BufferLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        let pixels = bytes
            .chunks_exact(3)
            .map(|chunk| Color::rgb(chunk[0], chunk[1], chunk[2]))
            .collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Builds an opaque grayscale image from one byte per pixel.
    pub fn from_luma8(width: u32, height: u32, bytes: &[u8]) -> Result<Self> {
        let count = checked_pixel_count(width, height)?;
        if bytes.len() != count {
            return Err(Error::BufferLengthMismatch {
                expected: count,
                actual: bytes.len(),
            });
        }
        let pixels = bytes.iter().copied().map(Color::gray).collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Builds an opaque grayscale image from 16-bit luma samples.
    pub fn from_luma16(width: u32, height: u32, samples: &[u16]) -> Result<Self> {
        let count = checked_pixel_count(width, height)?;
        if samples.len() != count {
            return Err(Error::BufferLengthMismatch {
                expected: count,
                actual: samples.len(),
            });
        }
        let pixels = samples
            .iter()
            .map(|&sample| {
                // Scale 16-bit luma to 8-bit with rounding rather than a plain
                // truncating shift, which would darken every sample.
                let value = ((sample as u32 * 255 + 32_767) / 65_535) as u8;
                Color::gray(value)
            })
            .collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

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
        self.pixels.len()
    }

    /// Read-only view of the row-major pixel buffer.
    #[inline]
    pub fn as_pixels(&self) -> &[Color] {
        &self.pixels
    }

    /// Mutable view of the row-major pixel buffer.
    #[inline]
    pub fn as_pixels_mut(&mut self) -> &mut [Color] {
        &mut self.pixels
    }

    /// Returns the pixel at `(x, y)`, or `None` when it is outside the image.
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> Option<Color> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = y as usize * self.width as usize + x as usize;
        self.pixels.get(index).copied()
    }

    /// Returns the pixel at `(x, y)` without a bounds check.
    ///
    /// The caller must prove `x < width` and `y < height`; all internal call
    /// sites derive the coordinates from loops over the image dimensions.
    #[inline]
    pub(crate) fn pixel_unchecked(&self, x: u32, y: u32) -> Color {
        debug_assert!(x < self.width && y < self.height);
        let index = y as usize * self.width as usize + x as usize;
        self.pixels[index]
    }

    /// The rectangle covering the whole image.
    #[inline]
    pub const fn bounds(&self) -> PixelRect {
        PixelRect::new(0, 0, self.width, self.height)
    }

    /// Whether any pixel has a non-opaque alpha channel.
    pub fn has_transparency(&self) -> bool {
        self.pixels.iter().any(|pixel| !pixel.is_opaque())
    }

    /// Whether every pixel is fully transparent.
    pub fn is_fully_transparent(&self) -> bool {
        self.pixels
            .iter()
            .all(|pixel| pixel.is_transparent())
    }

    /// Composites the image over an opaque background and returns an opaque
    /// image.
    ///
    /// The background alpha is forced to 255; a transparent background would
    /// not remove any source color and would only make the result harder to
    /// reason about.
    pub fn composite_over(&self, background: Color) -> Self {
        let background = background.with_alpha(255);
        let pixels = self
            .pixels
            .iter()
            .map(|pixel| pixel.composite_over(background))
            .collect();
        Self {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    /// Returns a new image with every pixel transformed by `transform`.
    pub fn map<F>(&self, mut transform: F) -> Self
    where
        F: FnMut(Color) -> Color,
    {
        let pixels = self
            .pixels
            .iter()
            .copied()
            .map(&mut transform)
            .collect();
        Self {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    /// Produces an interleaved RGBA byte buffer.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pixels.len() * 4);
        for pixel in &self.pixels {
            bytes.extend_from_slice(&pixel.to_rgba8());
        }
        bytes
    }

    /// Produces an interleaved RGB byte buffer, dropping alpha.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pixels.len() * 3);
        for pixel in &self.pixels {
            bytes.extend_from_slice(&pixel.to_rgb8());
        }
        bytes
    }

    /// Rec. 601 luma for every pixel in row-major order.
    pub fn luma8(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .map(|pixel| pixel.intensity())
            .collect()
    }

    /// A summed-area table over pixel luma.
    pub fn integral_luma(&self) -> IntegralImage {
        let width = self.width as usize;
        let height = self.height as usize;
        let stride = width + 1;
        let mut data = vec![0_u64; (width + 1) * (height + 1)];
        for y in 0..height {
            let row = self.pixels[y * width..(y + 1) * width]
                .iter()
                .map(|pixel| pixel.intensity() as u64);
            let mut row_sum = 0_u64;
            for (x, luma) in row.enumerate() {
                row_sum += luma;
                data[(y + 1) * stride + x + 1] = row_sum + data[y * stride + x + 1];
            }
        }
        IntegralImage { width, height, data }
    }

    /// Returns `(x, y)` of the first pixel whose alpha is below `threshold`.
    pub fn first_transparent(&self, threshold: u8) -> Option<(u32, u32)> {
        for (index, pixel) in self.pixels.iter().enumerate() {
            if pixel.a < threshold {
                let x = (index % self.width as usize) as u32;
                let y = (index / self.width as usize) as u32;
                return Some((x, y));
            }
        }
        None
    }
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }
    width
        .checked_mul(height)
        .map(|count| count as usize)
        .ok_or(Error::ImageTooLarge { width, height })
}

/// A summed-area table over luma values.
///
/// The table stores `(width + 1) * (height + 1)` entries with a zero row and
/// column so a rectangle query is four lookups with no boundary branches.
#[derive(Clone, Debug)]
pub struct IntegralImage {
    width: usize,
    height: usize,
    data: Vec<u64>,
}

impl IntegralImage {
    /// Table width in pixels.
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Table height in pixels.
    #[inline]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Sum of luma in the half-open rectangle `[x0, x1) x [y0, y1)`.
    ///
    /// Coordinates are clamped to the image, so callers that use a window
    /// around a border pixel do not need special cases.
    pub fn rectangle_sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
        if self.width == 0 || self.height == 0 {
            return 0;
        }
        let x0 = x0.min(self.width);
        let x1 = x1.min(self.width).max(x0);
        let y0 = y0.min(self.height);
        let y1 = y1.min(self.height).max(y0);
        let stride = self.width + 1;
        let a = self.data[y0 * stride + x0];
        let b = self.data[y0 * stride + x1];
        let c = self.data[y1 * stride + x0];
        let d = self.data[y1 * stride + x1];
        d + a - b - c
    }

    /// Mean luma in the half-open rectangle, or `None` when it is empty.
    pub fn rectangle_mean(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Option<f64> {
        let x0 = x0.min(self.width);
        let x1 = x1.min(self.width).max(x0);
        let y0 = y0.min(self.height);
        let y1 = y1.min(self.height).max(y0);
        let area = (x1 - x0) * (y1 - y0);
        if area == 0 {
            return None;
        }
        Some(self.rectangle_sum(x0, y0, x1, y1) as f64 / area as f64)
    }

    /// Mean luma of a square window centred on `(x, y)`.
    pub fn local_mean(&self, x: u32, y: u32, radius: u32) -> Option<f64> {
        let radius = radius as usize;
        let cx = x as usize;
        let cy = y as usize;
        let x0 = cx.saturating_sub(radius);
        let y0 = cy.saturating_sub(radius);
        let x1 = (cx + radius + 1).min(self.width);
        let y1 = (cy + radius + 1).min(self.height);
        self.rectangle_mean(x0, y0, x1, y1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_validates_dimensions_and_length() {
        assert!(matches!(
            ColorImage::new(0, 5, Vec::new()),
            Err(Error::EmptyImage)
        ));
        assert!(matches!(
            ColorImage::new(2, 2, vec![Color::BLACK; 3]),
            Err(Error::PixelCountMismatch { .. })
        ));
        let image = ColorImage::new(2, 2, vec![Color::BLACK; 4]).expect("valid image");
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.pixel_count(), 4);
    }

    #[test]
    fn byte_constructors_round_trip() {
        let image = ColorImage::from_rgba8(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("valid rgba data");
        assert_eq!(image.pixel(0, 0), Some(Color::rgba(1, 2, 3, 4)));
        assert_eq!(image.to_rgba8(), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rgb = ColorImage::from_rgb8(1, 2, &[9, 8, 7, 6, 5, 4]).expect("valid rgb data");
        assert_eq!(rgb.pixel(0, 0), Some(Color::rgb(9, 8, 7)));
        assert_eq!(rgb.to_rgb8(), vec![9, 8, 7, 6, 5, 4]);

        let luma = ColorImage::from_luma8(2, 1, &[0, 255]).expect("valid luma data");
        assert_eq!(luma.pixel(1, 0), Some(Color::WHITE));
    }

    #[test]
    fn invalid_buffers_are_rejected() {
        assert!(matches!(
            ColorImage::from_rgba8(2, 2, &[0; 15]),
            Err(Error::BufferLengthMismatch { .. })
        ));
        assert!(matches!(
            ColorImage::from_rgb8(1, 1, &[0; 2]),
            Err(Error::BufferLengthMismatch { .. })
        ));
        assert!(matches!(
            ColorImage::from_luma8(1, 1, &[0; 2]),
            Err(Error::BufferLengthMismatch { .. })
        ));
    }

    #[test]
    fn bounds_and_pixel_access_are_safe() {
        let image = ColorImage::filled(3, 2, Color::rgb(10, 20, 30)).expect("image");
        assert_eq!(image.bounds(), PixelRect::new(0, 0, 3, 2));
        assert_eq!(image.pixel(2, 1), Some(Color::rgb(10, 20, 30)));
        assert_eq!(image.pixel(3, 0), None);
        assert_eq!(image.pixel(0, 2), None);
    }

    #[test]
    fn compositing_produces_opaque_pixels() {
        let image = ColorImage::from_rgba8(2, 1, &[255, 0, 0, 255, 0, 0, 255, 0]).expect("image");
        let flattened = image.composite_over(Color::WHITE);
        assert_eq!(flattened.pixel(0, 0), Some(Color::rgb(255, 0, 0)));
        assert_eq!(flattened.pixel(1, 0), Some(Color::WHITE));
        assert!(!flattened.has_transparency());
        assert!(image.has_transparency());
    }

    #[test]
    fn integral_image_queries_rectangles() {
        let image = ColorImage::from_luma8(3, 2, &[10, 20, 30, 40, 50, 60]).expect("image");
        let integral = image.integral_luma();
        assert_eq!(integral.rectangle_sum(0, 0, 3, 2), 210);
        assert_eq!(integral.rectangle_sum(1, 0, 2, 1), 20);
        assert_eq!(integral.rectangle_sum(0, 0, 0, 0), 0);
        assert_eq!(integral.rectangle_mean(0, 0, 3, 2), Some(35.0));
        assert_eq!(integral.rectangle_mean(2, 2, 2, 2), None);
        assert_eq!(integral.local_mean(0, 0, 1), Some(30.0));
    }

    #[test]
    fn map_and_luma_helpers_preserve_dimensions() {
        let image = ColorImage::from_luma8(2, 2, &[0, 10, 20, 30]).expect("image");
        let doubled = image.map(|pixel| pixel.with_alpha(128));
        assert_eq!(doubled.width(), 2);
        assert!(doubled.has_transparency());
        assert_eq!(image.luma8(), vec![0, 10, 20, 30]);
        assert_eq!(image.first_transparent(255), None);
    }
}
