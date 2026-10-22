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

//! Colors and the color scheme of the theme.
//!
//! Colors are stored as non-premultiplied sRGB components in the
//! `0.0 ..= 1.0` range. The backend decides how they reach the screen:
//! a Vulkan backend writes them into an `R8G8B8A8_UNORM` attachment, a
//! WebGPU backend into an `rgba8unorm` texture, and a browser canvas
//! through `CanvasRenderingContext2D::fill_rect`.

use core::fmt;

/// A non-premultiplied sRGB color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// The red component.
    pub red: f32,
    /// The green component.
    pub green: f32,
    /// The blue component.
    pub blue: f32,
    /// The alpha component; `0.0` is fully transparent.
    pub alpha: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Color = Color {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        alpha: 0.0,
    };
    /// Opaque black.
    pub const BLACK: Color = Color {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        alpha: 1.0,
    };
    /// Opaque white.
    pub const WHITE: Color = Color {
        red: 1.0,
        green: 1.0,
        blue: 1.0,
        alpha: 1.0,
    };
    /// Opaque red.
    pub const RED: Color = Color {
        red: 1.0,
        green: 0.0,
        blue: 0.0,
        alpha: 1.0,
    };
    /// Opaque green.
    pub const GREEN: Color = Color {
        red: 0.0,
        green: 1.0,
        blue: 0.0,
        alpha: 1.0,
    };
    /// Opaque blue.
    pub const BLUE: Color = Color {
        red: 0.0,
        green: 0.0,
        blue: 1.0,
        alpha: 1.0,
    };

    /// Creates a color from four sRGB components.
    #[inline]
    #[must_use]
    pub const fn new(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// Creates an opaque color from three sRGB components.
    #[inline]
    #[must_use]
    pub const fn rgb(red: f32, green: f32, blue: f32) -> Self {
        Self::new(red, green, blue, 1.0)
    }

    /// Creates a color from 8-bit sRGB components.
    #[inline]
    #[must_use]
    pub fn from_rgba8(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self::new(
            f32::from(red) / 255.0,
            f32::from(green) / 255.0,
            f32::from(blue) / 255.0,
            f32::from(alpha) / 255.0,
        )
    }

    /// Creates an opaque color from 8-bit sRGB components.
    #[inline]
    #[must_use]
    pub fn from_rgb8(red: u8, green: u8, blue: u8) -> Self {
        Self::from_rgba8(red, green, blue, 255)
    }

    /// The color with its alpha replaced by `alpha`.
    #[inline]
    #[must_use]
    pub const fn with_alpha(self, alpha: f32) -> Self {
        Self {
            red: self.red,
            green: self.green,
            blue: self.blue,
            alpha,
        }
    }

    /// The color with its alpha multiplied by `factor`.
    #[inline]
    #[must_use]
    pub const fn multiply_alpha(self, factor: f32) -> Self {
        Self {
            red: self.red,
            green: self.green,
            blue: self.blue,
            alpha: self.alpha * factor,
        }
    }

    /// The color as four 8-bit sRGB components, rounded and clamped.
    #[must_use]
    pub fn to_rgba8(self) -> (u8, u8, u8, u8) {
        (
            quantize(self.red),
            quantize(self.green),
            quantize(self.blue),
            quantize(self.alpha),
        )
    }

    /// The linear interpolation between two colors.
    ///
    /// `factor` is clamped to `0.0 ..= 1.0`; `0.0` yields `self` and
    /// `1.0` yields `other`.
    #[must_use]
    pub fn lerp(self, other: Color, factor: f32) -> Color {
        let factor = factor.clamp(0.0, 1.0);
        let inverse = 1.0 - factor;
        Color::new(
            self.red * inverse + other.red * factor,
            self.green * inverse + other.green * factor,
            self.blue * inverse + other.blue * factor,
            self.alpha * inverse + other.alpha * factor,
        )
    }

    /// Whether every component is finite.
    #[inline]
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.red.is_finite() && self.green.is_finite() && self.blue.is_finite() && self.alpha.is_finite()
    }
}

/// Converts a normalized component into an 8-bit value.
fn quantize(component: f32) -> u8 {
    (libm::roundf(component.clamp(0.0, 1.0) * 255.0)) as u8
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (red, green, blue, alpha) = self.to_rgba8();
        write!(f, "#{red:02X}{green:02X}{blue:02X}{alpha:02X}")
    }
}

/// The set of colors a widget can draw with.
///
/// The scheme mirrors the Material 3 role names so a port of an
/// existing design is a matter of filling in the table, and every
/// widget asks the theme for a role instead of hard-coding a color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorScheme {
    /// The main accent color.
    pub primary: Color,
    /// The color drawn on top of [`Self::primary`].
    pub on_primary: Color,
    /// A lower-emphasis variant of [`Self::primary`].
    pub primary_container: Color,
    /// The color drawn on top of [`Self::primary_container`].
    pub on_primary_container: Color,
    /// The secondary accent color.
    pub secondary: Color,
    /// The color drawn on top of [`Self::secondary`].
    pub on_secondary: Color,
    /// The background of the window.
    pub background: Color,
    /// The color drawn on top of [`Self::background`].
    pub on_background: Color,
    /// The background of raised surfaces.
    pub surface: Color,
    /// The color drawn on top of [`Self::surface`].
    pub on_surface: Color,
    /// A lower-emphasis variant of [`Self::surface`].
    pub surface_variant: Color,
    /// The color drawn on top of [`Self::surface_variant`].
    pub on_surface_variant: Color,
    /// The color of borders and separators.
    pub outline: Color,
    /// The color used to report an error.
    pub error: Color,
    /// The color drawn on top of [`Self::error`].
    pub on_error: Color,
}

impl ColorScheme {
    /// The light color scheme.
    #[must_use]
    pub const fn light() -> Self {
        Self {
            primary: Color::rgb(0.00, 0.40, 0.85),
            on_primary: Color::WHITE,
            primary_container: Color::rgb(0.82, 0.90, 1.00),
            on_primary_container: Color::rgb(0.02, 0.16, 0.35),
            secondary: Color::rgb(0.30, 0.32, 0.36),
            on_secondary: Color::WHITE,
            background: Color::rgb(0.98, 0.98, 0.99),
            on_background: Color::rgb(0.08, 0.09, 0.11),
            surface: Color::WHITE,
            on_surface: Color::rgb(0.08, 0.09, 0.11),
            surface_variant: Color::rgb(0.91, 0.92, 0.94),
            on_surface_variant: Color::rgb(0.31, 0.32, 0.35),
            outline: Color::rgb(0.55, 0.57, 0.60),
            error: Color::rgb(0.75, 0.11, 0.11),
            on_error: Color::WHITE,
        }
    }

    /// The dark color scheme.
    #[must_use]
    pub const fn dark() -> Self {
        Self {
            primary: Color::rgb(0.55, 0.75, 1.00),
            on_primary: Color::rgb(0.03, 0.12, 0.28),
            primary_container: Color::rgb(0.14, 0.30, 0.52),
            on_primary_container: Color::rgb(0.87, 0.94, 1.00),
            secondary: Color::rgb(0.76, 0.78, 0.82),
            on_secondary: Color::rgb(0.20, 0.21, 0.24),
            background: Color::rgb(0.07, 0.08, 0.09),
            on_background: Color::rgb(0.92, 0.93, 0.95),
            surface: Color::rgb(0.10, 0.11, 0.13),
            on_surface: Color::rgb(0.92, 0.93, 0.95),
            surface_variant: Color::rgb(0.20, 0.21, 0.24),
            on_surface_variant: Color::rgb(0.75, 0.76, 0.80),
            outline: Color::rgb(0.42, 0.44, 0.47),
            error: Color::rgb(1.00, 0.54, 0.54),
            on_error: Color::rgb(0.25, 0.03, 0.03),
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::light()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_bit_components_round_trip() {
        let color = Color::from_rgb8(18, 52, 86);
        assert_eq!(color.to_rgba8(), (18, 52, 86, 255));
    }

    #[test]
    fn alpha_is_carried_separately_from_the_color() {
        let color = Color::from_rgba8(255, 0, 0, 128);
        let (red, green, blue, alpha) = color.to_rgba8();
        assert_eq!((red, green, blue), (255, 0, 0));
        assert_eq!(alpha, 128);
    }

    #[test]
    fn lerp_endpoints_return_the_two_colors() {
        let black = Color::BLACK;
        let white = Color::WHITE;
        assert_eq!(black.lerp(white, 0.0), black);
        assert_eq!(black.lerp(white, 1.0), white);
        let middle = black.lerp(white, 0.5);
        assert!((middle.red - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_clamps_the_factor() {
        let black = Color::BLACK;
        let white = Color::WHITE;
        assert_eq!(black.lerp(white, -1.0), black);
        assert_eq!(black.lerp(white, 2.0), white);
    }

    #[test]
    fn colors_render_as_rgba_hex() {
        assert_eq!(Color::WHITE.to_string(), "#FFFFFFFF");
        assert_eq!(Color::TRANSPARENT.to_string(), "#00000000");
    }

    #[test]
    fn the_dark_scheme_is_darker_than_the_light_one() {
        let light = ColorScheme::light();
        let dark = ColorScheme::dark();
        assert!(dark.background.red < light.background.red);
        assert!(dark.on_background.red > light.on_background.red);
    }
}
