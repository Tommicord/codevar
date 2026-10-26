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

//! Shaping the text runs of a display list into a shared text frame.
//!
//! Kuitui records a laid-out tree as a flat
//! [`DrawList`](codevar_kuitui::paint::DrawList). The geometry operations
//! are replayed by a backend that visits them; the text operations are
//! different, because drawing them needs a font and the renderer's shared
//! atlas. [`shape_list`] walks the list and, for every text run, rasterizes
//! it with a [`Font`] into a [`TextFrame`], tagging each quad with the clip
//! the command was recorded with.
//!
//! Geometry is left untouched: a backend such as `UiPipe` drains it from the
//! same list. One frame therefore yields two products — the geometry the `ui`
//! shader draws and the glyph coverage the `text` shader samples — which is
//! exactly the split the compositor mixes.

use alloc::rc::Rc;

use codevar_kuitui::geometry::{Offset, Rect};
use codevar_kuitui::paint::{DrawCommand, DrawList, DrawVisitor};
use codevar_kuitui::text::TextStyle;
use codevar_kuitui_text::TextFrame;

use crate::font::{Font, ShapeError};

/// Shapes every text run of `list` into `frame`.
///
/// The list is not drained, so a geometry backend can still replay it. `font`
/// supplies the raster and the advances; the frame's atlas receives the
/// coverage and its mesh builder the quads.
///
/// A color run has no color font here, so it is shaped through [`font`]'s
/// outline pipeline like any other run. Use [`shape_list_with_color`] when
/// the list contains runs that need the color pipeline.
///
/// # Errors
///
/// The first [`ShapeError`] a run reports.
pub fn shape_list(frame: &TextFrame, font: &Font, list: &DrawList) -> Result<(), ShapeError> {
    let mut shaper = TextShaper::new(frame, font);
    shape_with(&mut shaper, list)
}

/// Shapes every text run of `list`, including color runs, into `frame`.
///
/// Coverage runs use `font`; color runs use `color_font`, whose COLR/CPAL
/// layers or bitmap strikes produce the premultiplied BGRA glyphs of the
/// frame's color atlas. A character `color_font` has no glyph for is
/// shaped through `font` into the coverage atlas, so a color run that
/// mixes text and emoji keeps both. The list is not drained.
///
/// # Errors
///
/// The first [`ShapeError`] a run reports.
pub fn shape_list_with_color(
    frame: &TextFrame,
    font: &Font,
    color_font: &Font,
    list: &DrawList,
) -> Result<(), ShapeError> {
    let mut shaper = TextShaper::with_color_font(frame, font, color_font);
    shape_with(&mut shaper, list)
}

/// Shapes `list` with an already configured shaper.
fn shape_with(shaper: &mut TextShaper<'_>, list: &DrawList) -> Result<(), ShapeError> {
    for command in list.commands() {
        shaper.push_command(command);
        if let Some(error) = shaper.error() {
            return Err(error);
        }
    }
    Ok(())
}

/// A [`DrawVisitor`] that keeps only the text runs of a display list.
///
/// The visitor remembers the clip of the command it is visiting, so every
/// quad it pushes joins the run of the command that recorded it.
pub struct TextShaper<'a> {
    /// The frame the coverage and the quads go to.
    frame: &'a TextFrame,
    /// The font the coverage runs are rasterized with.
    font: &'a Font,
    /// The font the color runs are rasterized with, if one was supplied.
    color_font: Option<&'a Font>,
    /// The clip of the command being visited.
    clip: Option<Rect>,
    /// The first failure, kept because `DrawVisitor` cannot return one.
    error: Option<ShapeError>,
}

impl<'a> TextShaper<'a> {
    /// Creates a shaper that fills `frame` with the glyphs of `font`.
    #[inline]
    #[must_use]
    pub const fn new(frame: &'a TextFrame, font: &'a Font) -> Self {
        Self {
            frame,
            font,
            color_font: None,
            clip: None,
            error: None,
        }
    }

    /// Creates a shaper that draws color runs with `color_font`.
    #[inline]
    #[must_use]
    pub const fn with_color_font(frame: &'a TextFrame, font: &'a Font, color_font: &'a Font) -> Self {
        Self {
            frame,
            font,
            color_font: Some(color_font),
            clip: None,
            error: None,
        }
    }

    /// Shapes the text run of `command`, if it has one.
    pub fn push_command(&mut self, command: &DrawCommand) {
        self.clip = command.clip;
        command.op.visit(self);
    }

    /// The first error a run reported, if any.
    #[inline]
    #[must_use]
    pub const fn error(&self) -> Option<ShapeError> {
        self.error
    }
}

impl DrawVisitor for TextShaper<'_> {
    fn text(&mut self, origin: Offset, text: &Rc<str>, style: &TextStyle) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self
            .font
            .shape_into(self.frame, origin, text, style, self.clip)
        {
            self.error = Some(error);
        }
    }

    fn color_text(&mut self, origin: Offset, text: &Rc<str>, style: &TextStyle) {
        if self.error.is_some() {
            return;
        }
        let result = match self.color_font {
            Some(font) => font.shape_into_color(self.frame, self.font, origin, text, style, self.clip),
            None => self
                .font
                .shape_into(self.frame, origin, text, style, self.clip),
        };
        if let Err(error) = result {
            self.error = Some(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use codevar_kuitui::color::Color;
    use codevar_kuitui::paint::{CornerRadii, Paint};
    use codevar_kuitui_text::{AtlasConfig, TextAtlas};

    const FONT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/tf/codevar.tf.space-mono.regular.ttf"
    );

    const EMOJI_FONT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../codevar-truetype-emoji-color/tests/resources/NotoColorEmoji-colr-v1-subset.ttf"
    );

    fn font(ppem: u16) -> Font {
        let bytes = std::fs::read(FONT).unwrap();
        Font::from_bytes(bytes).unwrap().with_ppem(ppem)
    }

    fn emoji_font(ppem: u16) -> Font {
        let bytes = std::fs::read(EMOJI_FONT).unwrap();
        Font::from_bytes(bytes).unwrap().with_ppem(ppem)
    }

    fn frame() -> TextFrame {
        let atlas = TextAtlas::new(AtlasConfig::DEFAULT).unwrap();
        let frame = TextFrame::new(atlas).unwrap();
        frame.begin_frame();
        frame
    }

    #[test]
    fn only_the_text_runs_reach_the_frame() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 8.0, 8.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        list.text(Offset::new(2.0, 20.0), Rc::from("Hi"), TextStyle::new(24.0));

        let frame = frame();
        shape_list(&frame, &font(24), &list).unwrap();

        let runs = frame.runs();
        assert_eq!(runs.len(), 1, "the rectangle contributes no text");
        assert_eq!(runs[0].quad_count(), 2);
        assert_eq!(frame.atlas().glyph_count(), 2, "H and i are distinct glyphs");
    }

    #[test]
    fn a_text_run_carries_the_command_clip() {
        let clip = Rect::from_min_size(Offset::new(1.0, 1.0), 40.0, 40.0);
        let mut list = DrawList::new();
        list.push_clip(clip);
        list.text(Offset::ZERO, Rc::from("Hi"), TextStyle::new(24.0));
        list.pop_clip();

        let frame = frame();
        shape_list(&frame, &font(24), &list).unwrap();

        assert_eq!(frame.runs()[0].clip, Some(clip));
    }

    #[test]
    fn shaping_leaves_the_list_drainable_for_geometry() {
        let mut list = DrawList::new();
        list.text(Offset::ZERO, Rc::from("Hi"), TextStyle::new(24.0));
        list.rect(
            Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );

        let frame = frame();
        shape_list(&frame, &font(24), &list).unwrap();

        assert_eq!(list.len(), 2, "the list is not drained");
    }

    #[test]
    fn a_color_text_run_reaches_the_color_atlas() {
        let mut list = DrawList::new();
        list.color_text(Offset::new(0.0, 24.0), Rc::from("😀"), TextStyle::new(24.0));

        let frame = frame();
        shape_list_with_color(&frame, &font(24), &emoji_font(24), &list).unwrap();

        assert!(frame.runs().is_empty(), "a color run is not coverage");
        let runs = frame.color_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].quad_count(), 1);
        assert_eq!(frame.color_atlas().glyph_count(), 1);
    }

    #[test]
    fn a_mixed_color_run_shapes_text_and_emoji_into_both_atlases() {
        let mut list = DrawList::new();
        list.color_text(Offset::new(0.0, 24.0), Rc::from("Hi 😀"), TextStyle::new(24.0));

        let frame = frame();
        shape_list_with_color(&frame, &font(24), &emoji_font(24), &list).unwrap();

        let runs = frame.runs();
        assert_eq!(runs.len(), 1, "the text glyphs form one coverage run");
        assert_eq!(runs[0].quad_count(), 2, "H and i come from the text font");
        drop(runs);
        assert_eq!(frame.atlas().glyph_count(), 2);
        let color_runs = frame.color_runs();
        assert_eq!(color_runs.len(), 1);
        assert_eq!(color_runs[0].quad_count(), 1, "the emoji stays in color");
        drop(color_runs);
        assert_eq!(frame.color_atlas().glyph_count(), 1);
    }

    #[test]
    fn a_color_run_of_plain_text_falls_back_to_coverage() {
        let mut list = DrawList::new();
        list.color_text(Offset::new(0.0, 24.0), Rc::from("Hi"), TextStyle::new(24.0));

        let frame = frame();
        shape_list_with_color(&frame, &font(24), &emoji_font(24), &list).unwrap();

        assert!(
            frame.color_runs().is_empty(),
            "the color face has no glyph for either character"
        );
        assert_eq!(frame.runs()[0].quad_count(), 2);
    }

    #[test]
    fn a_color_text_run_carries_the_command_clip() {
        let clip = Rect::from_min_size(Offset::new(3.0, 4.0), 20.0, 20.0);
        let mut list = DrawList::new();
        list.push_clip(clip);
        list.color_text(Offset::ZERO, Rc::from("😀"), TextStyle::new(24.0));
        list.pop_clip();

        let frame = frame();
        shape_list_with_color(&frame, &font(24), &emoji_font(24), &list).unwrap();

        assert_eq!(frame.color_runs()[0].clip, Some(clip));
    }

    #[test]
    fn a_color_text_run_without_a_color_font_falls_back_to_coverage() {
        let mut list = DrawList::new();
        list.color_text(Offset::ZERO, Rc::from("Hi"), TextStyle::new(24.0));

        let frame = frame();
        shape_list(&frame, &font(24), &list).unwrap();

        assert!(frame.color_runs().is_empty());
        assert_eq!(frame.runs()[0].quad_count(), 2);
    }
}
