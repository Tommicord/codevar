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

//! Behavior of the label and of the two text buttons.

use codevar_kuitui::prelude::*;
use codevar_kuitui::{DrawList, DrawVisitor, Offset, TextStyle};
use codevar_kuitui_button::*;
use std::rc::Rc;

/// The runs a display list draws, in the order it recorded them.
#[derive(Default)]
struct Runs {
    plain: Vec<(Rc<str>, TextStyle)>,
    colored: Vec<(Rc<str>, TextStyle)>,
}

impl Runs {
    /// The characters of every plain run, concatenated.
    fn drawn(&self) -> String {
        self.plain
            .iter()
            .map(|(text, _)| text.as_ref())
            .collect::<String>()
    }
}

impl DrawVisitor for Runs {
    fn text(&mut self, _origin: Offset, text: &Rc<str>, style: &TextStyle) {
        self.plain.push((Rc::clone(text), style.clone()));
    }

    fn color_text(&mut self, _origin: Offset, text: &Rc<str>, style: &TextStyle) {
        self.colored
            .push((Rc::clone(text), style.clone()));
    }
}

/// Measures, places and draws `node` in `theme`, and collects the runs.
fn runs_in(node: &mut Node, theme: &Theme) -> Runs {
    let size = node.measure(theme, Constraints::unbounded());
    node.place(Rect::new(Offset::ZERO, size));
    let mut list = DrawList::new();
    node.draw(theme, &mut list);
    let mut recorder = Runs::default();
    for command in list.commands() {
        command.op.visit(&mut recorder);
    }
    recorder
}

fn label(content: TextContent, flags: ButtonFlags) -> Node {
    Node::new(Label::<Text>::new(
        content,
        ButtonPalette::role_of(ButtonColors::text),
        flags,
    ))
}

#[test]
fn the_label_paints_the_content_color_of_its_palette() {
    let theme = Theme::light();
    let mut node = label(TextContent::new("Save"), ButtonFlags::STANDARD);
    let drawn = runs_in(&mut node, &theme);
    assert_eq!(drawn.drawn(), "Save");
    assert_eq!(drawn.plain.len(), 1);
    assert_eq!(
        drawn.plain[0].1.color,
        ButtonColors::text(&theme.colors).content_for(ButtonFlags::STANDARD),
        "the label takes the content color the button resolved"
    );
}

#[test]
fn a_disabled_button_dims_its_label() {
    let theme = Theme::light();
    let flags = ButtonFlags::HOVERABLE;
    let mut node = label(TextContent::new("Save"), flags);
    let drawn = runs_in(&mut node, &theme);
    assert_eq!(
        drawn.plain[0].1.color,
        ButtonColors::text(&theme.colors).content_for(flags),
        "a button that answers nothing draws the disabled content"
    );
    assert_ne!(
        drawn.plain[0].1.color,
        ButtonColors::text(&theme.colors).content_for(ButtonFlags::STANDARD)
    );
}

#[test]
fn the_label_rebuilds_when_the_theme_changes() {
    let mut node = label(TextContent::new("Save"), ButtonFlags::STANDARD);
    let light = runs_in(&mut node, &Theme::light());
    let dark = runs_in(&mut node, &Theme::dark());
    assert_ne!(
        light.plain[0].1.color, dark.plain[0].1.color,
        "the color of the label belongs to the theme of the frame"
    );
    assert_eq!(
        dark.plain[0].1.color,
        ButtonColors::text(&Theme::dark().colors).content_for(ButtonFlags::STANDARD)
    );
}

#[test]
fn every_span_of_the_content_reaches_the_label() {
    let theme = Theme::light();
    let content = TextContent::from_spans(vec![TextSpan::new("Save"), TextSpan::new(" as draft")]);
    let mut node = label(content, ButtonFlags::STANDARD);
    assert_eq!(runs_in(&mut node, &theme).drawn(), "Save as draft");
}

#[test]
fn the_label_draws_any_script_the_font_covers() {
    let theme = Theme::light();
    let mut node = label(TextContent::new("保存 🔥 σִבָּה"), ButtonFlags::STANDARD);
    assert_eq!(runs_in(&mut node, &theme).drawn(), "保存 🔥 σִבָּה");
}

#[test]
fn a_text_button_draws_its_label_inside_the_container() {
    let theme = Theme::light();
    let mut node = TextButton::new(|| {}, "Save").node();
    let drawn = runs_in(&mut node, &theme);
    assert_eq!(drawn.drawn(), "Save");
    assert!(
        drawn.colored.is_empty(),
        "an ordinary button records no color run"
    );
}

#[test]
fn a_colorable_button_records_a_color_run() {
    let theme = Theme::light();
    let mut node = ColorableTextButton::new(|| {}, "🔥 Save").node();
    let drawn = runs_in(&mut node, &theme);
    assert_eq!(drawn.colored.len(), 1);
    assert_eq!(drawn.colored[0].0.as_ref(), "🔥 Save");
    assert!(
        drawn.plain.is_empty(),
        "the colorable button records the run as color text"
    );
}
