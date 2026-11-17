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

//! A text widget backed by a real font.
//!
//! Kuitui's own [`codevar_kuitui::widgets::Text`] measures with the
//! metrics of the theme, which only has to be approximately right. This
//! widget measures with the [`Font`] it was built with, so its layout
//! matches the glyphs that will actually be drawn.
//!
//! It draws in two ways, and a backend chooses one:
//!
//! * [`Widget::draw`] records a [`TextOp`] op, which is all a
//!   backend that owns its own glyph atlas needs;
//! * [`Text::shape_into`] rasterizes the line into the shared
//!   [`TextFrame`], which is what a backend that draws the `text`
//!   shader consumes. The frame owns the atlas, so reshaping a line
//!   every frame only rasterizes glyphs that are not resident yet.
//!
//! The widget is single-line: a `'\n'` in the text is not a line break,
//! it is a character with no glyph. Use one `Text` per line.
//!
//! [`TextOp`]: codevar_kuitui::paint::TextOp

use alloc::rc::Rc;

use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::{Constraints, Offset, Rect, Size};
use codevar_kuitui::text::{FontMetrics, TextStyle};
use codevar_kuitui::widget::{DrawScope, MeasureScope, Widget};
use codevar_kuitui_text::TextFrame;

use crate::font::{Font, ShapeError};

/// Draws a single line of text with a real font.
#[derive(Debug, Clone)]
pub struct Text {
    /// The characters of the line.
    text: Rc<str>,
    /// The font the line is measured and drawn with.
    font: Rc<Font>,
    /// The style of the line.
    style: TextStyle,
}

impl Text {
    /// Creates a line of text drawn with `font`.
    ///
    /// The style defaults to the font's size with the default line
    /// height.
    #[must_use]
    pub fn new(text: impl AsRef<str>, font: Rc<Font>) -> Self {
        let style = TextStyle::new(f32::from(font.ppem()));
        Self {
            text: Rc::from(text.as_ref()),
            font,
            style,
        }
    }

    /// The line with an explicit style.
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// The line with a different color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.style = self.style.with_color(color);
        self
    }

    /// The characters of the line.
    #[inline]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The font the line is drawn with.
    #[inline]
    #[must_use]
    pub fn font(&self) -> &Font {
        &self.font
    }

    /// The style of the line.
    #[inline]
    #[must_use]
    pub const fn style(&self) -> &TextStyle {
        &self.style
    }

    /// Shapes the line into a shared text frame.
    ///
    /// This is the renderer path: [`Font::shape_into`] rasterizes the
    /// glyphs that are not resident yet and pushes their quads into
    /// `frame`, whose atlas and mesh the backend draws. A backend that
    /// replays the whole display list can call
    /// [`crate::shape::shape_list`] instead and shape every run at once.
    ///
    /// `origin` is the baseline origin of the line and `clip` is the clip
    /// the run was recorded with.
    ///
    /// # Errors
    ///
    /// The [`ShapeError`] of the glyph pipeline or of the shared atlas.
    pub fn shape_into(
        &self,
        frame: &TextFrame,
        origin: Offset,
        clip: Option<Rect>,
    ) -> Result<(), ShapeError> {
        self.font
            .shape_into(frame, origin, &self.text, &self.style, clip)
    }
}

impl Widget for Text {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let measured = self.font.measure(&self.text, &self.style);
        constraints.enforce(Size::new(measured.width, measured.height))
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, _bounds: Rect) {
        scope.draw_text(
            Offset::new(0.0, self.font.ascent()),
            Rc::clone(&self.text),
            self.style.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_kuitui::paint::{DrawList, TextOp};
    use codevar_kuitui::theme::Theme;

    const FONT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/ttf/space-mono.regular.ttf"
    );

    fn font() -> Rc<Font> {
        let bytes = std::fs::read(FONT).unwrap();
        Rc::new(Font::from_bytes(bytes).unwrap().with_ppem(24))
    }

    /// A widget container without children, used to build a scope.
    struct NoChildren;

    impl codevar_kuitui::widget::Children for NoChildren {
        fn child_count(&self) -> usize {
            0
        }

        fn child_size(&self, _index: usize) -> Size {
            Size::ZERO
        }

        fn child_flex(&self, _index: usize) -> f32 {
            0.0
        }

        fn measure_child(&mut self, _index: usize, _theme: &Theme, constraints: Constraints) -> Size {
            constraints.enforce(Size::ZERO)
        }

        fn place_child(&mut self, _index: usize, _position: Offset) {}

        fn dispatch_child(
            &mut self,
            _index: usize,
            _theme: &Theme,
            _path: &mut alloc::vec::Vec<usize>,
            _focus: &mut Option<alloc::vec::Vec<usize>>,
            _event: &codevar_kuitui::event::Event,
        ) -> codevar_kuitui::event::EventResult {
            codevar_kuitui::event::EventResult::Ignored
        }
    }

    #[test]
    fn a_text_measures_with_the_font() {
        let theme = Theme::light();
        let mut children = NoChildren;
        let mut text = Text::new("Codevar", font());
        let size = text.measure(
            &mut MeasureScope::new(&theme, &mut children),
            Constraints::unbounded(),
        );
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn a_text_draws_one_text_operation() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut text = Text::new("Codevar", font()).color(Color::RED);
        let mut scope = DrawScope::new(&mut list, &theme, Offset::ZERO);
        text.draw(&mut scope, Rect::from_min_size(Offset::ZERO, 100.0, 40.0));
        assert_eq!(list.len(), 1);
        let op = list
            .ops()
            .next()
            .expect("a text run was recorded");
        let any: &dyn core::any::Any = op;
        let text = any
            .downcast_ref::<TextOp>()
            .expect("expected text");
        assert!(text.origin.y > 0.0);
        assert_eq!(text.style.color, Color::RED);
    }

    #[test]
    fn a_text_shapes_every_glyph_into_the_shared_frame() {
        let text = Text::new("Codevar", font());
        let atlas = codevar_kuitui_text::TextAtlas::new(codevar_kuitui_text::AtlasConfig::DEFAULT)
            .expect("the default atlas config is valid");
        let frame = TextFrame::new(atlas).expect("valid atlas config creates frame");
        frame.begin_frame();

        text.shape_into(&frame, Offset::ZERO, None)
            .expect("the line shapes");

        let runs = frame.runs();
        assert_eq!(runs.len(), 1, "one page and no clip is one run");
        assert_eq!(runs[0].quad_count(), 7, "one quad per glyph of \"Codevar\"");
    }
}
