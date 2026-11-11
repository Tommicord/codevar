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

//! The theme hooks of the decoration layer.
//!
//! A decoration is drawn with the theme of the window it decorates. The
//! theme supplies the colors, the type scale and the font metrics, and a
//! backend can override any of them with the values the native windowing
//! system reports.

use alloc::rc::Rc;

use codevar_kuitui::color::{Color, ColorScheme};
use codevar_kuitui::theme::{Theme, ThemeMode, Typography};

/// The theme a decoration is drawn with.
///
/// The theme wraps a [`codevar_kuitui::theme::Theme`] and adds the
/// decoration-specific roles: the color of the title bar, the color of
/// the resize border and the color of the active and inactive borders.
#[derive(Clone)]
pub struct DecorationTheme {
    /// The underlying kuitui theme.
    pub theme: Theme,
    /// The color of the title bar of an active window.
    pub active_title_bar: Color,
    /// The color of the title bar of an inactive window.
    pub inactive_title_bar: Color,
    /// The color of the resize border of an active window.
    pub active_border: Color,
    /// The color of the resize border of an inactive window.
    pub inactive_border: Color,
}

impl DecorationTheme {
    /// Creates a decoration theme from a kuitui theme.
    #[must_use]
    pub fn new(theme: Theme) -> Self {
        let on_surface = theme.colors.on_surface;
        let surface = theme.colors.surface;
        Self {
            theme,
            active_title_bar: on_surface,
            inactive_title_bar: Color {
                red: on_surface.red * 0.6,
                green: on_surface.green * 0.6,
                blue: on_surface.blue * 0.6,
                alpha: on_surface.alpha,
            },
            active_border: on_surface,
            inactive_border: Color {
                red: surface.red,
                green: surface.green,
                blue: surface.blue,
                alpha: surface.alpha * 0.5,
            },
        }
    }

    /// The theme of the requested appearance.
    #[must_use]
    pub fn for_mode(mode: ThemeMode) -> Self {
        Self::new(Theme::for_mode(mode))
    }

    /// The light decoration theme.
    #[must_use]
    pub fn light() -> Self {
        Self::for_mode(ThemeMode::Light)
    }

    /// The dark decoration theme.
    #[must_use]
    pub fn dark() -> Self {
        Self::for_mode(ThemeMode::Dark)
    }

    /// The title bar color for a window whose active state is `active`.
    #[must_use]
    pub const fn title_bar_color(&self, active: bool) -> Color {
        if active {
            self.active_title_bar
        } else {
            self.inactive_title_bar
        }
    }

    /// The border color for a window whose active state is `active`.
    #[must_use]
    pub const fn border_color(&self, active: bool) -> Color {
        if active {
            self.active_border
        } else {
            self.inactive_border
        }
    }

    /// The type scale of the theme.
    #[inline]
    #[must_use]
    pub const fn typography(&self) -> &Typography {
        &self.theme.typography
    }

    /// The color scheme of the theme.
    #[inline]
    #[must_use]
    pub const fn colors(&self) -> &ColorScheme {
        &self.theme.colors
    }

    /// The kuitui theme this decoration theme wraps.
    #[inline]
    #[must_use]
    pub const fn kuitui_theme(&self) -> &Theme {
        &self.theme
    }

    /// The decoration theme with a different set of font metrics.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Rc<dyn codevar_kuitui::text::FontMetrics>) -> Self {
        self.theme = self.theme.with_metrics(metrics);
        self
    }
}

impl Default for DecorationTheme {
    fn default() -> Self {
        Self::light()
    }
}

impl core::fmt::Debug for DecorationTheme {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecorationTheme")
            .field("theme", &self.theme)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_and_dark_themes_have_different_colors() {
        assert_ne!(
            DecorationTheme::light().title_bar_color(true),
            DecorationTheme::dark().title_bar_color(true)
        );
    }

    #[test]
    fn inactive_title_bar_is_dimmer_than_active() {
        let theme = DecorationTheme::light();
        let active = theme.title_bar_color(true);
        let inactive = theme.title_bar_color(false);
        assert!(inactive.red <= active.red);
        assert!(inactive.green <= active.green);
        assert!(inactive.blue <= active.blue);
    }
}
