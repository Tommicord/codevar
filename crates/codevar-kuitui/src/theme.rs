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

//! The theme: the colors, the type scale and the font metrics a widget
//! tree is drawn with.
//!
//! A theme is passed to every widget through the measure and draw
//! scopes, so a widget never has to reach for a global. Changing the
//! theme of a running application is therefore a matter of swapping
//! the value the composition hands out.

use alloc::rc::Rc;

use crate::color::ColorScheme;
use crate::text::{ApproximateFontMetrics, FontMetrics, TextStyle};

/// The type scale of a theme.
///
/// The five roles are enough for a user interface: a large headline, a
/// section title, a regular body and a small label.
#[derive(Debug, Clone, PartialEq)]
pub struct Typography {
    /// The largest style, used by the title bar of a window.
    pub display: TextStyle,
    /// The style of a section heading.
    pub headline: TextStyle,
    /// The style of a subsection heading.
    pub title: TextStyle,
    /// The style of ordinary text.
    pub body: TextStyle,
    /// The style of small, secondary text.
    pub label: TextStyle,
}

impl Typography {
    /// Creates a type scale from five styles.
    #[must_use]
    pub const fn new(
        display: TextStyle,
        headline: TextStyle,
        title: TextStyle,
        body: TextStyle,
        label: TextStyle,
    ) -> Self {
        Self {
            display,
            headline,
            title,
            body,
            label,
        }
    }

    /// The default type scale, built from `body` and derived sizes.
    ///
    /// The sizes follow the 1.2 ratio of the Material type scale,
    /// rounded to whole pixels.
    #[must_use]
    pub fn scaled(base_size: f32) -> Self {
        let body = TextStyle::new(base_size);
        Self {
            display: TextStyle::new(libm::roundf(base_size * 3.0)),
            headline: TextStyle::new(libm::roundf(base_size * 2.0)),
            title: TextStyle::new(libm::roundf(base_size * 1.5)),
            body,
            label: TextStyle::new(libm::roundf(base_size * 0.75)),
        }
    }
}

impl Default for Typography {
    fn default() -> Self {
        Self::scaled(16.0)
    }
}

/// Whether the application asks for a light or a dark appearance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThemeMode {
    /// The light appearance.
    #[default]
    Light,
    /// The dark appearance.
    Dark,
}

/// Everything a widget needs to know about the appearance of the
/// application.
#[derive(Clone)]
pub struct Theme {
    /// The role table of the theme.
    pub colors: ColorScheme,
    /// The type scale of the theme.
    pub typography: Typography,
    /// The metrics of the font the theme is drawn with.
    pub metrics: Rc<dyn FontMetrics>,
}

impl Theme {
    /// Creates a theme from its three parts.
    #[must_use]
    pub fn new(colors: ColorScheme, typography: Typography, metrics: Rc<dyn FontMetrics>) -> Self {
        Self {
            colors,
            typography,
            metrics,
        }
    }

    /// The light theme, measured with the approximate font metrics.
    #[must_use]
    pub fn light() -> Self {
        Self::for_mode(ThemeMode::Light)
    }

    /// The dark theme, measured with the approximate font metrics.
    #[must_use]
    pub fn dark() -> Self {
        Self::for_mode(ThemeMode::Dark)
    }

    /// The theme of the requested appearance, measured with the
    /// approximate font metrics.
    #[must_use]
    pub fn for_mode(mode: ThemeMode) -> Self {
        let colors = match mode {
            ThemeMode::Light => ColorScheme::light(),
            ThemeMode::Dark => ColorScheme::dark(),
        };
        Self::new(colors, Typography::default(), Rc::new(ApproximateFontMetrics))
    }

    /// The theme with a different set of font metrics.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Rc<dyn FontMetrics>) -> Self {
        self.metrics = metrics;
        self
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::light()
    }
}

impl core::fmt::Debug for Theme {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Theme")
            .field("colors", &self.colors)
            .field("typography", &self.typography)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_light_and_dark_themes_differ_in_their_colors() {
        assert_ne!(Theme::light().colors, Theme::dark().colors);
    }

    #[test]
    fn the_type_scale_grows_from_the_body_to_the_display() {
        let typography = Typography::scaled(16.0);
        assert!(typography.display.font_size > typography.headline.font_size);
        assert!(typography.headline.font_size > typography.title.font_size);
        assert!(typography.title.font_size > typography.body.font_size);
        assert!(typography.body.font_size > typography.label.font_size);
    }

    #[test]
    fn a_theme_can_be_rebuilt_with_real_font_metrics() {
        let theme = Theme::light().with_metrics(Rc::new(ApproximateFontMetrics));
        assert_eq!(theme.metrics.line_height(&TextStyle::new(10.0)), 14.0);
    }
}
