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

//! Behavior of the canvas-backed `concrete` decoration.

use codevar_kuitui::{
    Color, ColorScheme, Constraints, CornerRadii, DrawList, DrawVisitor, Modifier, Node, Offset, Paint,
    PaintStyle, Rect, Size, Text, Theme,
};
use codevar_kuitui_button::{Button, ButtonColors, ButtonPalette, Concrete, TextButton};
use codevar_kuitui_widget::canvas::LinearGradient;

const RED: Color = Color::new(0.86, 0.15, 0.15, 1.0);
const GREEN: Color = Color::new(0.09, 0.64, 0.29, 1.0);
const BLUE: Color = Color::new(0.14, 0.39, 0.92, 1.0);

/// The rectangles and the lines a node records, in paint order.
#[derive(Default)]
struct Painted {
    fills: Vec<(Rect, Color)>,
    outlines: Vec<(Rect, f32)>,
    lines: Vec<(Color, f32)>,
}

impl DrawVisitor for Painted {
    fn rect(&mut self, rect: Rect, _radii: CornerRadii, paint: Paint) {
        match paint.style {
            PaintStyle::Fill => self.fills.push((rect, paint.color)),
            PaintStyle::Stroke(width) => self.outlines.push((rect, width)),
        }
    }

    fn line(&mut self, _from: Offset, _to: Offset, paint: Paint) {
        if let PaintStyle::Stroke(width) = paint.style {
            self.lines.push((paint.color, width));
        }
    }
}

/// Measures, places and draws a node of `size` decorated by `modifier`.
fn painted(modifier: Modifier, size: Size) -> Painted {
    let theme = Theme::light();
    let mut node = Node::new(Text::new("")).modifier(modifier);
    node.measure(&theme, Constraints::tight(size));
    node.place(Rect::new(Offset::ZERO, size));
    let mut list = DrawList::new();
    node.draw(&theme, &mut list);
    let mut recorder = Painted::default();
    for command in list.commands() {
        command.op.visit(&mut recorder);
    }
    recorder
}

#[test]
fn the_decoration_paints_the_canvas_of_the_node() {
    let size = Size::new(64.0, 32.0);
    let modifier = Concrete::with_size(size)
        .drawn(|canvas| {
            canvas.fill_rect(Rect::from_min_size(Offset::ZERO, 64.0, 32.0), RED);
        })
        .modifier();
    let recorder = painted(modifier, size);
    assert_eq!(recorder.fills.len(), 1);
    assert_eq!(recorder.fills[0].0.size, size);
    assert_eq!(recorder.fills[0].1, RED);
}

#[test]
fn a_stroke_and_a_line_of_the_canvas_reach_the_display_list() {
    let size = Size::new(64.0, 32.0);
    let modifier = Concrete::with_size(size)
        .drawn(|canvas| {
            canvas.stroke_rect(
                Rect::from_min_size(Offset::new(2.0, 2.0), 60.0, 28.0),
                CornerRadii::ZERO,
                BLUE,
                3.0,
            );
            canvas.line(Offset::ZERO, Offset::new(64.0, 32.0), RED, 2.0);
        })
        .modifier();
    let recorder = painted(modifier, size);
    assert_eq!(recorder.outlines.len(), 1, "the rectangle is outlined");
    assert_eq!(recorder.outlines[0].1, 3.0);
    assert_eq!(recorder.lines.len(), 1, "the line is drawn");
    assert_eq!(recorder.lines[0].1, 2.0);
}

#[test]
fn an_edited_canvas_paints_on_the_next_frame() {
    let size = Size::new(64.0, 32.0);
    let decoration = Concrete::with_size(size);
    let canvas = decoration.canvas();
    assert!(
        painted(decoration.modifier(), size)
            .fills
            .is_empty(),
        "an empty canvas paints nothing"
    );
    canvas
        .borrow_mut()
        .fill_rect(Rect::from_min_size(Offset::ZERO, 16.0, 16.0), BLUE);
    let recorder = painted(decoration.modifier(), size);
    assert_eq!(recorder.fills.len(), 1, "the decoration reads the handle");
    assert_eq!(recorder.fills[0].1, BLUE);
}

#[test]
fn a_clear_of_the_canvas_paints_nothing_again() {
    let size = Size::new(64.0, 32.0);
    let decoration = Concrete::with_size(size).drawn(|canvas| {
        canvas.fill_rect(Rect::from_min_size(Offset::ZERO, 8.0, 8.0), RED);
    });
    let canvas = decoration.canvas();
    assert_eq!(painted(decoration.modifier(), size).fills.len(), 1);
    canvas.borrow_mut().clear();
    assert!(
        painted(decoration.modifier(), size)
            .fills
            .is_empty()
    );
}

#[test]
fn a_polygon_of_the_canvas_fills_by_scanline() {
    let size = Size::new(64.0, 32.0);
    let modifier = Concrete::with_size(size)
        .drawn(|canvas| {
            canvas.polygon(
                vec![
                    Offset::new(0.0, 0.0),
                    Offset::new(64.0, 0.0),
                    Offset::new(32.0, 32.0),
                ],
                Paint::fill(GREEN),
            );
        })
        .modifier();
    let recorder = painted(modifier, size);
    assert!(
        recorder.fills.len() > 8,
        "a triangle is one band per row it covers"
    );
    assert!(
        recorder
            .fills
            .iter()
            .all(|(_, color)| *color == GREEN),
        "every band carries the paint of the polygon"
    );
}

#[test]
fn a_gradient_of_the_canvas_paints_its_bands() {
    let size = Size::new(64.0, 32.0);
    let modifier = Concrete::with_size(size)
        .drawn(|canvas| {
            canvas.gradient(
                Rect::from_min_size(Offset::ZERO, 64.0, 32.0),
                LinearGradient::new(Offset::ZERO, Offset::new(64.0, 0.0), RED, BLUE),
                4,
            );
        })
        .modifier();
    let recorder = painted(modifier, size);
    assert_eq!(recorder.fills.len(), 4, "the gradient emits its steps");
    let first = recorder.fills.first().map(|(_, color)| color.red);
    let last = recorder.fills.last().map(|(_, color)| color.red);
    assert!(
        first.is_some_and(|red| last.is_some_and(|last| red > last)),
        "the band at the start of the axis is nearer the from color"
    );
}

#[test]
fn the_decoration_draws_over_the_container_of_a_button() {
    let size = Size::new(120.0, 40.0);
    let scheme = ColorScheme::light();
    let modifier = Concrete::with_size(size)
        .drawn(|canvas| {
            canvas.fill_rect(Rect::from_min_size(Offset::ZERO, 120.0, 40.0), GREEN);
        })
        .modifier();
    let mut node = TextButton::new(|| {}, "Save")
        .colors(ButtonPalette::fixed(ButtonColors::filled(&scheme)))
        .modifier(modifier)
        .node();
    let theme = Theme::light();
    node.measure(&theme, Constraints::tight(size));
    node.place(Rect::new(Offset::ZERO, size));
    let mut list = DrawList::new();
    node.draw(&theme, &mut list);
    let mut recorder = Painted::default();
    for command in list.commands() {
        command.op.visit(&mut recorder);
    }
    // The chrome paints in the background pass of the node and the canvas
    // in the decoration pass that follows it, so the decoration is the one
    // a reader sees: a container painted over it hid every canvas.
    assert_eq!(recorder.fills.len(), 2, "the container and its canvas");
    assert_eq!(recorder.fills[0].1, scheme.primary);
    assert_eq!(
        recorder.fills[1].1, GREEN,
        "the concrete operation paints over the chrome of the button"
    );
}
