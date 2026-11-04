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

//! Text content, styling and font metrics.
//!
//! Kuitui does not rasterize glyphs. It measures text through the
//! [`FontMetrics`] trait and hands the measured runs to the backend,
//! which owns the font atlas. An application that only needs an
//! approximate layout can use [`ApproximateFontMetrics`]; one that has
//! a real font can implement the trait over `codevar-truetype-core`
//! and get exact measurements.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::color::Color;

/// The weight of a font.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontWeight {
    /// The lightest weight.
    Thin,
    /// The regular weight.
    Normal,
    /// A weight between [`Self::Normal`] and [`Self::Bold`].
    Medium,
    /// A weight between [`Self::Medium`] and [`Self::Bold`].
    SemiBold,
    /// The heaviest weight.
    Bold,
}

impl FontWeight {
    /// The numeric weight, as the CSS `font-weight` property defines it.
    #[must_use]
    pub const fn value(self) -> u16 {
        match self {
            Self::Thin => 100,
            Self::Normal => 400,
            Self::Medium => 500,
            Self::SemiBold => 600,
            Self::Bold => 700,
        }
    }
}

/// The visual attributes of a run of text.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    /// The size of the em square, in logical pixels.
    pub font_size: f32,
    /// The distance between two baselines, in logical pixels.
    pub line_height: f32,
    /// The weight of the glyphs.
    pub weight: FontWeight,
    /// The color of the glyphs.
    pub color: Color,
    /// The extra space inserted between two glyphs, in logical pixels.
    pub letter_spacing: f32,
    /// Whether the glyphs are slanted.
    pub italic: bool,
}

impl TextStyle {
    /// Creates a style with the given size and the default attributes.
    #[must_use]
    pub const fn new(font_size: f32) -> Self {
        Self {
            font_size,
            line_height: font_size * 1.4,
            weight: FontWeight::Normal,
            color: Color::BLACK,
            letter_spacing: 0.0,
            italic: false,
        }
    }

    /// The style with a different color.
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// The style with a different weight.
    #[must_use]
    pub const fn with_weight(mut self, weight: FontWeight) -> Self {
        self.weight = weight;
        self
    }

    /// The style with a different line height.
    #[must_use]
    pub const fn with_line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }

    /// The style with a different letter spacing.
    #[must_use]
    pub const fn with_letter_spacing(mut self, letter_spacing: f32) -> Self {
        self.letter_spacing = letter_spacing;
        self
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self::new(16.0)
    }
}

/// The horizontal alignment of the lines of a text block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextAlign {
    /// Aligned with the left edge.
    Left,
    /// Aligned with the horizontal center.
    Center,
    /// Aligned with the right edge.
    Right,
}

/// A run of text sharing one [`TextStyle`].
#[derive(Debug, Clone, PartialEq)]
pub struct TextSpan {
    /// The characters of the run.
    pub text: Rc<str>,
    /// The style of the run.
    pub style: TextStyle,
}

impl TextSpan {
    /// Creates a span with the default style.
    #[must_use]
    pub fn new(text: impl AsRef<str>) -> Self {
        Self {
            text: Rc::from(text.as_ref()),
            style: TextStyle::default(),
        }
    }

    /// Creates a span with an explicit style.
    #[must_use]
    pub fn with_style(text: impl AsRef<str>, style: TextStyle) -> Self {
        Self {
            text: Rc::from(text.as_ref()),
            style,
        }
    }
}

/// A block of text made of one or more [`TextSpan`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct TextContent {
    /// The spans of the block, in reading order.
    pub spans: Vec<TextSpan>,
    /// The horizontal alignment of the lines.
    pub align: TextAlign,
    /// The maximum number of lines to keep, or `None` for no limit.
    pub max_lines: Option<u32>,
}

impl TextContent {
    /// Creates a single-span block with the default style.
    #[must_use]
    pub fn new(text: impl AsRef<str>) -> Self {
        Self {
            spans: alloc::vec![TextSpan::new(text)],
            align: TextAlign::Left,
            max_lines: None,
        }
    }

    /// Creates a block from a list of spans.
    #[must_use]
    pub fn from_spans(spans: Vec<TextSpan>) -> Self {
        Self {
            spans,
            align: TextAlign::Left,
            max_lines: None,
        }
    }

    /// The concatenated characters of every span.
    #[must_use]
    pub fn plain_text(&self) -> String {
        let mut text = String::new();
        for span in &self.spans {
            text.push_str(&span.text);
        }
        text
    }

    /// The lines of the content.
    ///
    /// The lines are split once, when the content is built, so that
    /// drawing the text does not allocate.
    #[must_use]
    pub fn lines(&self) -> Vec<Rc<str>> {
        self.plain_text()
            .split('\n')
            .map(Rc::from)
            .collect()
    }

    /// The number of lines the block occupies when it is not truncated.
    #[must_use]
    pub fn line_count(&self) -> u32 {
        let mut lines = 1_u32;
        for span in &self.spans {
            for byte in span.text.as_bytes() {
                if *byte == b'\n' {
                    lines += 1;
                }
            }
        }
        lines
    }
}

/// The measured extent of a piece of text.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TextSize {
    /// The width of the widest line.
    pub width: f32,
    /// The total height of every line.
    pub height: f32,
}

/// The measurements a text widget needs from the font it is drawn with.
///
/// Implementing this trait over a real font is what turns the
/// approximate layout of [`ApproximateFontMetrics`] into an exact
/// one. The trait is object-safe so a theme can carry the metrics of
/// the application.
pub trait FontMetrics {
    /// The distance between two baselines for `style`.
    fn line_height(&self, style: &TextStyle) -> f32;

    /// The distance from the top of a line box to its baseline.
    ///
    /// A text run is recorded at its baseline, not at the top of its
    /// line box, so the widget needs this to place the run. The default
    /// puts the baseline at 80% of the em square, which is what the
    /// [`ApproximateFontMetrics`] layout assumes; a metric backed by a
    /// real font overrides it with the font's own ascent.
    fn ascent(&self, style: &TextStyle) -> f32 {
        style.font_size * 0.8
    }

    /// The horizontal advance of `text` when drawn with `style`.
    fn advance(&self, text: &str, style: &TextStyle) -> f32;

    /// The extent of `text` when drawn with `style`.
    ///
    /// The default implementation measures the widest line and counts
    /// the lines, which is correct for every font that does not
    /// kern across line breaks.
    fn measure(&self, text: &str, style: &TextStyle) -> TextSize {
        let mut width = 0.0_f32;
        let mut lines = 0_u32;
        for line in text.split('\n') {
            width = width.max(self.advance(line, style));
            lines += 1;
        }
        TextSize {
            width,
            height: self.line_height(style) * lines as f32,
        }
    }
}

/// A font metric that assumes every glyph is a fixed-width box.
///
/// The advance of a glyph is `0.6 * font_size`, the average width of
/// the Latin alphabet in a proportional font. It is enough to lay a
/// user interface out before the real font is loaded, and it keeps the
/// framework free of a font dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ApproximateFontMetrics;

impl FontMetrics for ApproximateFontMetrics {
    fn line_height(&self, style: &TextStyle) -> f32 {
        if style.line_height > 0.0 {
            style.line_height
        } else {
            style.font_size * 1.4
        }
    }

    fn advance(&self, text: &str, style: &TextStyle) -> f32 {
        let count = text.chars().count() as f32;
        let weight_factor = match style.weight {
            FontWeight::Thin => 0.92,
            FontWeight::Normal => 1.0,
            FontWeight::Medium => 1.04,
            FontWeight::SemiBold => 1.08,
            FontWeight::Bold => 1.12,
        };
        count * style.font_size * 0.6 * weight_factor + count * style.letter_spacing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_approximate_metrics_grow_with_the_font_size() {
        let metrics = ApproximateFontMetrics;
        let small = metrics.measure("codevar", &TextStyle::new(10.0));
        let large = metrics.measure("codevar", &TextStyle::new(20.0));
        assert!(large.width > small.width);
        assert!(large.height > small.height);
    }

    #[test]
    fn the_approximate_metrics_count_the_lines() {
        let metrics = ApproximateFontMetrics;
        let style = TextStyle::new(10.0);
        let single = metrics.measure("one", &style);
        let double = metrics.measure("one\ntwo", &style);
        assert!((double.height - single.height * 2.0).abs() < 1e-4);
    }

    #[test]
    fn the_approximate_metrics_scale_with_the_weight() {
        let metrics = ApproximateFontMetrics;
        let normal = metrics.advance("codevar", &TextStyle::new(10.0));
        let bold = metrics.advance("codevar", &TextStyle::new(10.0).with_weight(FontWeight::Bold));
        assert!(bold > normal);
    }

    #[test]
    fn the_default_ascent_leaves_room_for_the_descent() {
        let metrics = ApproximateFontMetrics;
        let style = TextStyle::new(10.0);
        let ascent = metrics.ascent(&style);
        assert!(ascent > 0.0);
        assert!(ascent < metrics.line_height(&style));
    }

    #[test]
    fn a_text_content_counts_its_lines() {
        let content = TextContent::new("one\ntwo\nthree");
        assert_eq!(content.line_count(), 3);
        assert_eq!(content.plain_text(), "one\ntwo\nthree");
    }

    #[test]
    fn a_text_content_concatenates_its_spans() {
        let content = TextContent::from_spans(alloc::vec![TextSpan::new("Code"), TextSpan::new("var"),]);
        assert_eq!(content.plain_text(), "Codevar");
    }

    #[test]
    fn the_default_style_uses_a_readable_line_height() {
        let style = TextStyle::new(20.0);
        assert!((style.line_height - 28.0).abs() < 1e-4);
        assert_eq!(style.weight, FontWeight::Normal);
    }
}
