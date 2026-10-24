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

//! The text widget.

use alloc::rc::Rc;
use alloc::vec::Vec;

use crate::color::Color;
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::text::{TextAlign, TextContent, TextStyle};
use crate::theme::Theme;
use crate::widget::{DrawScope, MeasureScope, Widget};

/// Draws a run of text.
///
/// The text is measured with the metrics of the theme, so the widget
/// knows how big it is before it is drawn. The style is the body style
/// of the theme unless the application asks for another one.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    /// The content of the text.
    content: TextContent,
    /// The lines of the content, split once when the content changed.
    lines: Vec<Rc<str>>,
    /// The style that replaces the body style of the theme.
    style: Option<TextStyle>,
    /// The color that replaces the color of the style.
    color: Option<Color>,
    /// The maximum number of lines to draw.
    max_lines: Option<u32>,
}

impl Text {
    /// Creates a text widget that draws `text`.
    #[must_use]
    pub fn new(text: impl AsRef<str>) -> Self {
        let content = TextContent::new(text);
        let lines = content.lines();
        Self {
            content,
            lines,
            style: None,
            color: None,
            max_lines: None,
        }
    }

    /// Creates a text widget that draws `content`.
    #[must_use]
    pub fn with_content(content: TextContent) -> Self {
        let lines = content.lines();
        Self {
            content,
            lines,
            style: None,
            color: None,
            max_lines: None,
        }
    }

    /// The text with the given style.
    #[must_use]
    pub fn style(mut self, style: TextStyle) -> Self {
        self.style = Some(style);
        self
    }

    /// The text with the given color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// The text with the given horizontal alignment.
    #[must_use]
    pub fn align(mut self, align: TextAlign) -> Self {
        self.content.align = align;
        self
    }

    /// The text with at most `max_lines` lines.
    ///
    /// The lines that do not fit are not drawn and do not count
    /// towards the height of the text.
    #[must_use]
    pub fn max_lines(mut self, max_lines: u32) -> Self {
        self.max_lines = Some(max_lines);
        self
    }

    /// The content of the text.
    #[inline]
    #[must_use]
    pub const fn content(&self) -> &TextContent {
        &self.content
    }

    /// The style the text is drawn with.
    fn resolved_style(&self, theme: &Theme) -> TextStyle {
        let style = match &self.style {
            Some(style) => style.clone(),
            None => theme.typography.body.clone(),
        };
        match self.color {
            Some(color) => style.with_color(color),
            None => style.with_color(theme.colors.on_surface),
        }
    }

    /// The lines the text draws.
    fn drawn_lines(&self) -> &[Rc<str>] {
        match self.max_lines {
            Some(max) if (max as usize) < self.lines.len() => &self.lines[..max as usize],
            _ => &self.lines,
        }
    }
}

impl Widget for Text {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let theme = scope.theme();
        let style = self.resolved_style(theme);
        let metrics = theme.metrics.as_ref();
        let lines = self.drawn_lines();
        let mut width = 0.0_f32;
        for line in lines {
            width = width.max(metrics.advance(line, &style));
        }
        let height = metrics.line_height(&style) * lines.len() as f32;
        constraints.enforce(Size::new(width, height))
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let theme = scope.theme();
        let style = self.resolved_style(theme);
        let metrics = theme.metrics.as_ref();
        let line_height = metrics.line_height(&style);
        let ascent = metrics.ascent(&style);
        let mut y = bounds.top();
        for line in self.drawn_lines() {
            let width = metrics.advance(line, &style);
            let x = match self.content.align {
                TextAlign::Left => bounds.left(),
                TextAlign::Center => bounds.left() + (bounds.size.width - width) * 0.5,
                TextAlign::Right => bounds.left() + bounds.size.width - width,
            };
            scope.draw_text(
                Offset::new(x - bounds.left(), y - bounds.top() + ascent),
                Rc::clone(line),
                style.clone(),
            );
            y += line_height;
        }
    }
}

/// Draws a single line of text whose glyphs carry their own color.
///
/// This is the color counterpart of [`Text`]: it is measured with the
/// metrics of the theme like any other text, but records a
/// [`ColorTextOp`](crate::paint::ColorTextOp) instead of a
/// [`TextOp`](crate::paint::TextOp). A backend that owns a color font
/// rasterizes the run through that font's COLR/CPAL layers or bitmap
/// strikes, so the glyphs keep the colors the font gives them.
///
/// The font itself is supplied to the shaping pass, not to the widget,
/// which is what keeps the core crate free of a font dependency.
///
/// The widget is single-line: a `'\n'` is not a line break, it is a
/// character with no glyph. Use one `ColorableText` per line.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorableText {
    /// The characters of the line.
    text: Rc<str>,
    /// The style that replaces the body style of the theme.
    style: Option<TextStyle>,
    /// The color that replaces the color of the style.
    color: Option<Color>,
}

impl ColorableText {
    /// Creates a color text widget that draws `text`.
    #[must_use]
    pub fn new(text: impl AsRef<str>) -> Self {
        Self {
            text: Rc::from(text.as_ref()),
            style: None,
            color: None,
        }
    }

    /// The line with the given style.
    #[must_use]
    pub fn style(mut self, style: TextStyle) -> Self {
        self.style = Some(style);
        self
    }

    /// The line with the given color.
    ///
    /// A color glyph paints the colors of the font. This color is the
    /// fallback a backend uses when it draws the run through the
    /// outline pipeline instead.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// The characters of the line.
    #[inline]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The style the line is drawn with.
    fn resolved_style(&self, theme: &Theme) -> TextStyle {
        let style = match &self.style {
            Some(style) => style.clone(),
            None => theme.typography.body.clone(),
        };
        match self.color {
            Some(color) => style.with_color(color),
            None => style.with_color(theme.colors.on_surface),
        }
    }
}

impl Widget for ColorableText {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let theme = scope.theme();
        let style = self.resolved_style(theme);
        let metrics = theme.metrics.as_ref();
        constraints.enforce(Size::new(
            metrics.advance(&self.text, &style),
            metrics.line_height(&style),
        ))
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, _bounds: Rect) {
        let theme = scope.theme();
        let style = self.resolved_style(theme);
        let ascent = theme.metrics.as_ref().ascent(&style);
        scope.draw_color_text(Offset::new(0.0, ascent), Rc::clone(&self.text), style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::paint::{ColorTextOp, TextOp};

    #[test]
    fn a_text_measures_the_width_of_its_longest_line() {
        let theme = Theme::light();
        let mut node = Node::new(Text::new("hello"));
        let size = node.measure(&theme, Constraints::unbounded());
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn a_text_with_several_lines_is_taller_than_a_text_with_one() {
        let theme = Theme::light();
        let mut single = Node::new(Text::new("one"));
        let mut double = Node::new(Text::new("one\ntwo"));
        let one = single.measure(&theme, Constraints::unbounded());
        let two = double.measure(&theme, Constraints::unbounded());
        assert!(two.height > one.height);
        assert_eq!(two.width, one.width);
    }

    #[test]
    fn a_text_draws_one_operation_per_line() {
        let theme = Theme::light();
        let mut list = crate::paint::DrawList::new();
        let mut node = Node::new(Text::new("one\ntwo\nthree"));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        node.draw(&theme, &mut list);
        assert_eq!(list.len(), 3);
        let op = list
            .ops()
            .next()
            .expect("a text run was recorded");
        let any: &dyn core::any::Any = op;
        assert!(any.downcast_ref::<TextOp>().is_some());
    }

    #[test]
    fn a_text_run_is_recorded_at_its_baseline() {
        let theme = Theme::light();
        let mut list = crate::paint::DrawList::new();
        let mut node = Node::new(Text::new("hello"));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        node.draw(&theme, &mut list);
        let op = list
            .ops()
            .next()
            .expect("a text run was recorded");
        let any: &dyn core::any::Any = op;
        let text = any
            .downcast_ref::<TextOp>()
            .expect("expected a text run");
        assert!(
            text.origin.y > 0.0,
            "the baseline sits below the top of the line box"
        );
        assert!(
            text.origin.y < theme.metrics.line_height(&text.style),
            "the baseline stays inside the line box"
        );
    }

    #[test]
    fn a_text_with_a_limit_draws_at_most_that_many_lines() {
        let theme = Theme::light();
        let mut list = crate::paint::DrawList::new();
        let mut node = Node::new(Text::new("one\ntwo\nthree").max_lines(2));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        node.draw(&theme, &mut list);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn the_color_of_the_text_replaces_the_color_of_the_theme() {
        let theme = Theme::light();
        let mut node = Node::new(Text::new("hello").color(Color::RED));
        let mut list = crate::paint::DrawList::new();
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        node.draw(&theme, &mut list);
        let op = list
            .ops()
            .next()
            .expect("a text run was recorded");
        let any: &dyn core::any::Any = op;
        let text = any
            .downcast_ref::<TextOp>()
            .expect("expected a text run");
        assert_eq!(text.style.color, Color::RED);
    }

    #[test]
    fn a_colorable_text_measures_with_the_theme() {
        let theme = Theme::light();
        let mut node = Node::new(ColorableText::new("😀"));
        let size = node.measure(&theme, Constraints::unbounded());
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn a_colorable_text_draws_one_color_text_operation() {
        let theme = Theme::light();
        let mut list = crate::paint::DrawList::new();
        let mut node = Node::new(ColorableText::new("😀").color(Color::RED));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        node.draw(&theme, &mut list);
        assert_eq!(list.len(), 1);
        let op = list
            .ops()
            .next()
            .expect("a color text run was recorded");
        let any: &dyn core::any::Any = op;
        let text = any
            .downcast_ref::<ColorTextOp>()
            .expect("expected a color text run");
        assert_eq!(&*text.text, "😀");
        assert!(text.origin.y > 0.0);
        assert_eq!(text.style.color, Color::RED);
    }

    #[test]
    fn the_macro_builds_a_colorable_text() {
        let theme = Theme::light();
        let mut tree = crate::kuitui! {
            Column {
                ColorableText("😀")
            }
        };
        tree.measure(&theme, Constraints::unbounded());
        tree.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        let mut list = crate::paint::DrawList::new();
        tree.draw(&theme, &mut list);
        let op = list
            .ops()
            .next()
            .expect("a color text run was recorded");
        let any: &dyn core::any::Any = op;
        assert!(any.downcast_ref::<ColorTextOp>().is_some());
    }
}
