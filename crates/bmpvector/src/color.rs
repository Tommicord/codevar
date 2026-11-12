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

//! Color representation, compositing and perceptual color space conversion.
//!
//! Palette selection and pixel assignment operate in the Oklab color space.
//! Oklab is perceptually uniform enough that a simple squared Euclidean
//! distance produces balanced clusters, while its inverse transform is cheap
//! enough for the per-color work this crate performs. The alpha channel is
//! carried as a fourth coordinate so a palette can represent translucent
//! artwork without a separate code path.
//!
//! sRGB transfer functions are implemented here directly instead of relying on
//! `std` or a color crate. That keeps the dependency footprint to [`libm`] and
//! makes the exact transfer curve visible to the quantizer.

use crate::geom::clamp;
use core::fmt;
use libm::{cbrt, pow};

/// Weight applied to the alpha difference when comparing Oklab colors.
///
/// The value is one because alpha is already normalized to `[0, 1]` and a fully
/// transparent and a fully opaque copy of the same RGB color are genuinely
/// different paints. Callers that want to ignore alpha can compare the RGB
/// channels directly.
pub const ALPHA_DISTANCE_WEIGHT: f64 = 1.0;

/// An 8-bit straight-alpha color.
///
/// Channels are stored as straight (non-premultiplied) alpha, matching the
/// convention callers use for image buffers. [`Color::composite_over`] performs
/// the premultiplication internally when a background is supplied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct Color {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel, where zero is fully transparent and 255 fully opaque.
    pub a: u8,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(255, 255, 255);

    /// An opaque color from red, green and blue channels.
    #[inline]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// A color with an explicit alpha channel.
    #[inline]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// An opaque gray color.
    #[inline]
    pub const fn gray(value: u8) -> Self {
        Self::rgb(value, value, value)
    }

    /// A color from a packed `0xRRGGBBAA` integer.
    #[inline]
    pub const fn from_rgba_u32(packed: u32) -> Self {
        Self {
            r: ((packed >> 24) & 0xff) as u8,
            g: ((packed >> 16) & 0xff) as u8,
            b: ((packed >> 8) & 0xff) as u8,
            a: (packed & 0xff) as u8,
        }
    }

    /// The color packed into a `0xRRGGBBAA` integer.
    #[inline]
    pub const fn to_rgba_u32(self) -> u32 {
        ((self.r as u32) << 24) | ((self.g as u32) << 16) | ((self.b as u32) << 8) | self.a as u32
    }

    /// A color from a four-byte slice, or `None` when it is too short.
    #[inline]
    pub fn from_rgba8(bytes: &[u8]) -> Option<Self> {
        match bytes {
            [r, g, b, a, ..] => Some(Self::rgba(*r, *g, *b, *a)),
            _ => None,
        }
    }

    /// The four channels as an array.
    #[inline]
    pub const fn to_rgba8(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// The three RGB channels as an array.
    #[inline]
    pub const fn to_rgb8(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// Whether the color is completely transparent.
    #[inline]
    pub const fn is_transparent(self) -> bool {
        self.a == 0
    }

    /// Whether the color is fully opaque.
    #[inline]
    pub const fn is_opaque(self) -> bool {
        self.a == 255
    }

    /// The same color with a different alpha channel.
    #[inline]
    pub const fn with_alpha(self, alpha: u8) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a: alpha,
        }
    }

    /// Alpha as a fraction in `[0, 1]`.
    #[inline]
    pub fn alpha_f64(self) -> f64 {
        self.a as f64 / 255.0
    }

    /// Rec. 601 luma rounded to the nearest integer.
    ///
    /// Integer weights keep the result deterministic across targets and avoid
    /// a floating-point load in the binary threshold hot path.
    #[inline]
    pub const fn intensity(self) -> u8 {
        let weighted = self.r as u32 * 299 + self.g as u32 * 587 + self.b as u32 * 114;
        ((weighted + 500) / 1000) as u8
    }

    /// Linear-light relative luminance in `[0, 1]`.
    pub fn relative_luminance(self) -> f64 {
        0.212_672_9 * srgb_to_linear(self.r)
            + 0.715_152_2 * srgb_to_linear(self.g)
            + 0.072_175_0 * srgb_to_linear(self.b)
    }

    /// Composites `self` over an opaque `background` using source-over alpha.
    ///
    /// The output is always opaque because the destination alpha is 255. This
    /// is the operation used when the caller asks the tracer to flatten an
    /// image onto a solid background before palette selection.
    pub fn composite_over(self, background: Self) -> Self {
        let source_alpha = self.alpha_f64();
        if source_alpha >= 1.0 {
            return self.with_alpha(255);
        }
        if source_alpha <= 0.0 {
            return background.with_alpha(255);
        }
        let inverse = 1.0 - source_alpha;
        let red = self.r as f64 * source_alpha + background.r as f64 * inverse;
        let green = self.g as f64 * source_alpha + background.g as f64 * inverse;
        let blue = self.b as f64 * source_alpha + background.b as f64 * inverse;
        Self::rgba(round_to_u8(red), round_to_u8(green), round_to_u8(blue), 255)
    }

    /// Composites `self` over another translucent color.
    ///
    /// Implements the full source-over operator, including the output alpha
    /// term. A fully transparent source returns the destination untouched.
    pub fn blend_over(self, destination: Self) -> Self {
        let source_alpha = self.alpha_f64();
        let destination_alpha = destination.alpha_f64();
        if source_alpha <= 0.0 {
            return destination;
        }
        let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
        if output_alpha <= 0.0 {
            return Self::TRANSPARENT;
        }
        let inverse = destination_alpha * (1.0 - source_alpha);
        let red = (self.r as f64 * source_alpha + destination.r as f64 * inverse) / output_alpha;
        let green = (self.g as f64 * source_alpha + destination.g as f64 * inverse) / output_alpha;
        let blue = (self.b as f64 * source_alpha + destination.b as f64 * inverse) / output_alpha;
        Self::rgba(
            round_to_u8(red),
            round_to_u8(green),
            round_to_u8(blue),
            round_to_u8(output_alpha * 255.0),
        )
    }

    /// Converts the color to the Oklab color space with alpha.
    #[inline]
    pub fn to_oklab(self) -> Oklab {
        Oklab::from_color(self)
    }

    /// Builder-style alpha replacement.
    #[inline]
    pub const fn alpha(self, alpha: u8) -> Self {
        self.with_alpha(alpha)
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a == 255 {
            write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            write!(f, "#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }
}

/// Oklab color with a normalized alpha coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Oklab {
    /// Perceptual lightness in `[0, 1]`.
    pub l: f64,
    /// Green-red opponent axis.
    pub a: f64,
    /// Blue-yellow opponent axis.
    pub b: f64,
    /// Alpha in `[0, 1]`.
    pub alpha: f64,
}

impl Oklab {
    /// A color-space coordinate from its four components.
    #[inline]
    pub const fn new(l: f64, a: f64, b: f64, alpha: f64) -> Self {
        Self { l, a, b, alpha }
    }

    /// Converts an sRGB color to Oklab.
    pub fn from_color(color: Color) -> Self {
        let r = srgb_to_linear(color.r);
        let g = srgb_to_linear(color.g);
        let b = srgb_to_linear(color.b);

        let l = 0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b;
        let m = 0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b;
        let s = 0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b;

        let l_ = cbrt(l);
        let m_ = cbrt(m);
        let s_ = cbrt(s);

        Self {
            l: 0.210_454_255_3 * l_ + 0.793_617_785_0 * m_ - 0.004_072_046_8 * s_,
            a: 1.977_998_495_1 * l_ - 2.428_592_205_0 * m_ + 0.450_593_709_9 * s_,
            b: 0.025_904_037_1 * l_ + 0.782_771_766_2 * m_ - 0.808_675_766_0 * s_,
            alpha: color.alpha_f64(),
        }
    }

    /// Converts the coordinate back to a clamped sRGB color.
    ///
    /// Out-of-gamut coordinates are clipped per linear channel. Clipping is
    /// deterministic and preserves lightness better than a chroma-only gamut
    /// mapping for the small palettes this crate creates.
    pub fn to_color(self) -> Color {
        let l_ = self.l + 0.396_337_777_4 * self.a + 0.215_803_757_3 * self.b;
        let m_ = self.l - 0.105_561_345_8 * self.a - 0.063_854_172_8 * self.b;
        let s_ = self.l - 0.089_484_177_5 * self.a - 1.291_485_548_0 * self.b;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        let red = 4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s;
        let green = -1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s;
        let blue = -0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s;

        Color::rgba(
            linear_to_srgb_u8(red),
            linear_to_srgb_u8(green),
            linear_to_srgb_u8(blue),
            round_to_u8(clamp(self.alpha, 0.0, 1.0) * 255.0),
        )
    }

    /// Squared distance in four dimensions with the configured alpha weight.
    #[inline]
    pub fn distance_sq(self, other: Self) -> f64 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        let dalpha = (self.alpha - other.alpha) * ALPHA_DISTANCE_WEIGHT;
        dl * dl + da * da + db * db + dalpha * dalpha
    }

    /// Linear interpolation of the color coordinates.
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            l: self.l + (other.l - self.l) * t,
            a: self.a + (other.a - self.a) * t,
            b: self.b + (other.b - self.b) * t,
            alpha: self.alpha + (other.alpha - self.alpha) * t,
        }
    }

    /// Weighted addition used by the k-means accumulator.
    #[inline]
    pub fn add_scaled(&mut self, other: Self, weight: f64) {
        self.l += other.l * weight;
        self.a += other.a * weight;
        self.b += other.b * weight;
        self.alpha += other.alpha * weight;
    }

    /// Divides every coordinate by `weight`.
    #[inline]
    pub fn divide(&mut self, weight: f64) {
        if weight > 0.0 {
            let inverse = 1.0 / weight;
            self.l *= inverse;
            self.a *= inverse;
            self.b *= inverse;
            self.alpha *= inverse;
        }
    }

    /// Chroma (distance from the neutral axis).
    #[inline]
    pub fn chroma(self) -> f64 {
        libm::hypot(self.a, self.b)
    }

    /// Hue angle in radians.
    #[inline]
    pub fn hue(self) -> f64 {
        libm::atan2(self.b, self.a)
    }
}

/// Squared perceptual distance between two colors.
#[inline]
pub fn color_distance_sq(first: Color, second: Color) -> f64 {
    Oklab::from_color(first).distance_sq(Oklab::from_color(second))
}

/// Converts an 8-bit sRGB channel to linear light.
#[inline]
pub fn srgb_to_linear(channel: u8) -> f64 {
    let value = channel as f64 / 255.0;
    if value <= 0.040_45 {
        value / 12.92
    } else {
        pow((value + 0.055) / 1.055, 2.4)
    }
}

/// Converts a linear-light channel in `[0, 1]` to an 8-bit sRGB channel.
#[inline]
pub fn linear_to_srgb(channel: f64) -> f64 {
    let value = clamp(channel, 0.0, 1.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    }
}

/// Converts a linear-light channel to a rounded 8-bit value.
#[inline]
pub fn linear_to_srgb_u8(channel: f64) -> u8 {
    round_to_u8(linear_to_srgb(channel) * 255.0)
}

/// Rounds a finite value to the nearest `u8`, clamping out-of-range input.
#[inline]
pub fn round_to_u8(value: f64) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    let rounded = libm::round(value);
    if rounded <= 0.0 {
        0
    } else if rounded >= 255.0 {
        255
    } else {
        rounded as u8
    }
}

/// Blends two colors in Oklab and returns the nearest sRGB color.
pub fn mix_oklab(first: Color, second: Color, t: f64) -> Color {
    first
        .to_oklab()
        .lerp(second.to_oklab(), clamp(t, 0.0, 1.0))
        .to_color()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        libm::fabs(a - b) <= tolerance
    }

    #[test]
    fn color_packing_round_trips() {
        let color = Color::rgba(12, 34, 56, 78);
        assert_eq!(Color::from_rgba_u32(color.to_rgba_u32()), color);
        assert_eq!(Color::from_rgba8(&[12, 34, 56, 78]), Some(color));
        assert_eq!(Color::from_rgba8(&[12, 34]), None);
        assert_eq!(color.to_rgba8(), [12, 34, 56, 78]);
    }

    #[test]
    fn oklab_round_trip_stays_within_one_channel_step() {
        let samples = [
            Color::BLACK,
            Color::WHITE,
            Color::rgb(255, 0, 0),
            Color::rgb(0, 255, 0),
            Color::rgb(0, 0, 255),
            Color::rgb(120, 70, 200),
            Color::rgba(30, 200, 90, 128),
        ];
        for color in samples {
            let back = color.to_oklab().to_color();
            assert!(back.r.abs_diff(color.r) <= 1);
            assert!(back.g.abs_diff(color.g) <= 1);
            assert!(back.b.abs_diff(color.b) <= 1);
            assert!(back.a.abs_diff(color.a) <= 1);
        }
    }

    #[test]
    fn perceptual_distance_matches_visual_order() {
        let red = Color::rgb(220, 20, 20);
        let orange = Color::rgb(240, 120, 20);
        let blue = Color::rgb(20, 20, 220);
        let red_orange = color_distance_sq(red, orange);
        let red_blue = color_distance_sq(red, blue);
        assert!(red_orange < red_blue);
        assert_eq!(color_distance_sq(red, red), 0.0);
    }

    #[test]
    fn compositing_over_opaque_background_is_exact() {
        assert_eq!(
            Color::rgba(255, 0, 0, 255).composite_over(Color::WHITE),
            Color::rgb(255, 0, 0)
        );
        assert_eq!(
            Color::rgba(255, 0, 0, 0).composite_over(Color::WHITE),
            Color::WHITE
        );
        let half = Color::rgba(0, 0, 0, 128).composite_over(Color::WHITE);
        assert!(half.r.abs_diff(127) <= 1);
        assert_eq!(half.a, 255);
    }

    #[test]
    fn source_over_blending_computes_output_alpha() {
        let source = Color::rgba(255, 0, 0, 128);
        let destination = Color::rgba(0, 0, 255, 255);
        let blended = source.blend_over(destination);
        assert_eq!(blended.a, 255);
        assert!(blended.r > 127);
        assert!(blended.b.abs_diff(127) <= 1);

        let transparent = Color::TRANSPARENT.blend_over(destination);
        assert_eq!(transparent, destination);
    }

    #[test]
    fn intensity_uses_weighted_luma() {
        assert_eq!(Color::WHITE.intensity(), 255);
        assert_eq!(Color::BLACK.intensity(), 0);
        let green = Color::rgb(0, 255, 0);
        let red = Color::rgb(255, 0, 0);
        assert!(green.intensity() > red.intensity());
    }

    #[test]
    fn srgb_transfer_round_trips() {
        for value in [0_u8, 1, 12, 64, 128, 200, 254, 255] {
            let linear = srgb_to_linear(value);
            let back = linear_to_srgb(linear);
            assert!(close(back * 255.0, value as f64, 0.6));
        }
    }

    #[test]
    fn display_switches_to_eight_digit_hex_for_alpha() {
        assert_eq!(Color::rgb(1, 2, 3).to_string(), "#010203");
        assert_eq!(Color::rgba(1, 2, 3, 4).to_string(), "#01020304");
    }
}
