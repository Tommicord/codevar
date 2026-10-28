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

//! Behavior of the [`Button`] trait and of the chrome it builds.

use codevar_kuitui::{
    ColorScheme, Constraints, CornerRadii, DrawList, DrawVisitor, Effect, EffectKinds, Event,
    InteractionSource, InteractionState, Node, Offset, Paint, PointerEvent, PointerEventKind,
    ReactiveComposition, Rect, Size, Text, TextStyle, Theme, element,
};
use codevar_kuitui_button::*;
use core::cell::Cell;
use std::rc::Rc;

/// The button of the tests: it counts the clicks it answers.
struct TestButton {
    handler: Click,
    counter: Rc<Cell<u32>>,
    source: InteractionSource,
    flags: ButtonFlags,
    palette: ButtonPalette,
    shape: CornerRadii,
    elevation: Option<ButtonElevation>,
    border: Option<BorderStroke>,
    labels: Vec<&'static str>,
}

impl TestButton {
    fn new() -> Self {
        let counter = Rc::new(Cell::new(0));
        let counted = Rc::clone(&counter);
        let handler: Click = Rc::new(move || counted.set(counted.get() + 1));
        Self {
            handler,
            counter,
            source: InteractionSource::new(),
            flags: ButtonFlags::STANDARD,
            palette: ButtonPalette::fixed(ButtonColors::filled(&ColorScheme::light())),
            shape: defaults::SHAPE,
            elevation: None,
            border: None,
            labels: Vec::new(),
        }
    }

    fn with_flags(mut self, flags: ButtonFlags) -> Self {
        self.flags = flags;
        self
    }

    fn with_palette(mut self, palette: ButtonPalette) -> Self {
        self.palette = palette;
        self
    }

    fn with_shape(mut self, shape: CornerRadii) -> Self {
        self.shape = shape;
        self
    }

    fn with_elevation(mut self, elevation: ButtonElevation) -> Self {
        self.elevation = Some(elevation);
        self
    }

    fn with_border(mut self, border: BorderStroke) -> Self {
        self.border = Some(border);
        self
    }

    fn with_labels(mut self, labels: &[&'static str]) -> Self {
        self.labels = labels.to_vec();
        self
    }
}

impl Button for TestButton {
    fn on_click(&self) -> &Click {
        &self.handler
    }

    fn flags(&self) -> ButtonFlags {
        self.flags
    }

    fn shape(&self) -> CornerRadii {
        self.shape
    }

    fn colors(&self) -> ButtonPalette {
        self.palette
    }

    fn elevation(&self) -> Option<ButtonElevation> {
        self.elevation
    }

    fn border(&self) -> Option<BorderStroke> {
        self.border
    }

    fn interaction_source(&self) -> InteractionSource {
        self.source.clone()
    }

    fn content(&self) -> Vec<Node> {
        self.labels
            .iter()
            .map(|label| Node::new(Text::new(label)))
            .collect()
    }
}

/// Collects the rectangles and the text runs a display list records.
#[derive(Default)]
struct Recorder {
    rects: Vec<(Rect, CornerRadii, Paint)>,
    runs: Vec<(Offset, TextStyle)>,
}

impl DrawVisitor for Recorder {
    fn rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) {
        self.rects.push((rect, radii, paint));
    }

    fn text(&mut self, origin: Offset, _text: &Rc<str>, style: &TextStyle) {
        self.runs.push((origin, style.clone()));
    }
}

/// Replays `node` in a box of `size` and visits every command it
/// records.
fn replay<T>(node: &mut Node, theme: &Theme, size: Size, read: impl Fn(Recorder) -> T) -> T {
    node.measure(theme, Constraints::tight(size));
    node.place(Rect::new(Offset::ZERO, size));
    let mut list = DrawList::new();
    node.draw(theme, &mut list);
    let mut recorder = Recorder::default();
    for command in list.commands() {
        command.op.visit(&mut recorder);
    }
    read(recorder)
}

/// The rectangles of the frame `node` draws in a box of `size`, with
/// the layer each was recorded in.
fn recorded(node: &mut Node, theme: &Theme, size: Size) -> Vec<(Option<Effect>, Paint)> {
    let mut effects = Vec::new();
    node.measure(theme, Constraints::tight(size));
    node.place(Rect::new(Offset::ZERO, size));
    let mut list = DrawList::new();
    node.draw(theme, &mut list);
    for command in list.commands() {
        let mut visitor = Recorder::default();
        command.op.visit(&mut visitor);
        for (_, _, paint) in visitor.rects {
            effects.push((command.effect, paint));
        }
    }
    effects
}

/// A composition of `size` whose tree is a row holding `button`.
///
/// The button is a child and not the root on purpose: only a child
/// is hit-tested against its own bounds, which is what lets a
/// release outside the button be told apart from one inside it.
fn composed(button: TestButton, size: Size) -> ReactiveComposition {
    let button = Rc::new(button);
    let mut composition = ReactiveComposition::new(Theme::light());
    composition.set_content(move |_| {
        let button = Rc::clone(&button);
        element! {
            Row {
                Node(button.node())
            }
        }
    });
    composition.recompose(true);
    composition.layout(size, true);
    composition
}

fn pointer(kind: PointerEventKind, position: Offset) -> Event {
    Event::Pointer(PointerEvent::new(kind, position))
}

#[test]
fn the_padding_is_measured_around_the_content() {
    let theme = Theme::light();
    let unbounded = Constraints::unbounded();
    let mut first = Node::new(Text::new("one"));
    let mut second = Node::new(Text::new("two words"));
    let a = first.measure(&theme, unbounded);
    let b = second.measure(&theme, unbounded);
    let mut node = TestButton::new()
        .with_labels(&["one", "two words"])
        .node();
    let size = node.measure(&theme, unbounded);
    assert_eq!(
        size,
        Size::new(
            a.width + b.width + defaults::CONTENT_SPACING + defaults::CONTENT_PADDING.horizontal(),
            a.height.max(b.height) + defaults::CONTENT_PADDING.vertical(),
        )
    );
}

#[test]
fn the_content_row_is_centered_in_the_container() {
    let theme = Theme::light();
    let size = Size::new(200.0, 60.0);
    let mut label = Node::new(Text::new("Save"));
    let label_size = label.measure(&theme, Constraints::unbounded());
    let mut node = TestButton::new().with_labels(&["Save"]).node();
    let runs = replay(&mut node, &theme, size, |recorder| recorder.runs);
    assert_eq!(runs.len(), 1);
    let expected_x = (size.width - label_size.width) * 0.5;
    assert!(
        (runs[0].0.x - expected_x).abs() < 0.01,
        "the label is centered in the container: x is {}, expected {}",
        runs[0].0.x,
        expected_x
    );
    assert!(
        runs[0].0.y > defaults::CONTENT_PADDING.top,
        "the baseline of the label is inside the vertical padding"
    );
}

#[test]
fn a_rounded_container_is_recorded_inside_a_layer() {
    let theme = Theme::light();
    let size = Size::new(120.0, 40.0);
    let mut node = TestButton::new().with_labels(&["Save"]).node();
    let shapes = replay(&mut node, &theme, size, |recorder| recorder.rects);
    assert_eq!(shapes.len(), 1);
    assert_eq!(shapes[0].0.size, size);
    assert_eq!(shapes[0].1, defaults::SHAPE);
    let rects = recorded(&mut node, &theme, size);
    assert!(
        rects[0]
            .0
            .is_some_and(|effect| effect.kinds.is_empty() && effect.radii == defaults::SHAPE),
        "the corners alone are a layer the backend has to replay"
    );
}

#[test]
fn a_square_container_needs_no_layer() {
    let theme = Theme::light();
    let mut node = TestButton::new()
        .with_shape(CornerRadii::ZERO)
        .with_labels(&["Save"])
        .node();
    let rects = recorded(&mut node, &theme, Size::new(120.0, 40.0));
    assert_eq!(rects.len(), 1);
    assert_eq!(rects[0].0, None);
}

#[test]
fn a_text_role_button_records_no_container() {
    let theme = Theme::light();
    let mut node = TestButton::new()
        .with_palette(ButtonPalette::role_of(ButtonColors::text))
        .with_labels(&["Save"])
        .node();
    let rects = recorded(&mut node, &theme, Size::new(120.0, 40.0));
    assert!(rects.is_empty(), "an empty container is not worth a rectangle");
}

#[test]
fn the_hover_draws_a_state_layer_over_the_container() {
    let theme = Theme::light();
    let button = TestButton::new().with_labels(&["Save"]);
    let source = button.interaction_source();
    let mut node = button.node();
    source.set(InteractionState::HOVERED);
    let rects = recorded(&mut node, &theme, Size::new(120.0, 40.0));
    assert_eq!(rects.len(), 2, "the container and its state layer");
    assert_eq!(rects[0].1.color, ColorScheme::light().primary);
    assert_eq!(
        rects[1].1.color.alpha,
        defaults::HOVER_OVERLAY_ALPHA,
        "the layer is the hover tint of the state layer color"
    );
}

#[test]
fn an_elevated_button_blurs_its_own_silhouette() {
    let theme = Theme::light();
    let mut node = TestButton::new()
        .with_elevation(ButtonElevation::elevated())
        .with_labels(&["Save"])
        .node();
    let rects = recorded(&mut node, &theme, Size::new(120.0, 40.0));
    assert_eq!(rects.len(), 2, "the shadow and the container");
    assert!(
        rects[0]
            .0
            .is_some_and(|effect| { effect.kinds.contains(EffectKinds::BLUR) && effect.blur > 0.0 }),
        "the shadow is a blurred layer"
    );
    assert!(
        rects[0].1.color.alpha < 1.0,
        "the silhouette is translucent, or it would hide the container"
    );
}

#[test]
fn a_release_inside_fires_the_click_once() {
    let size = Size::new(240.0, 80.0);
    let button = TestButton::new().with_labels(&["Save"]);
    let counter = Rc::clone(&button.counter);
    let source = button.interaction_source();
    let mut composition = composed(button, size);
    let inside = Offset::new(10.0, 10.0);
    composition.handle_event(&pointer(PointerEventKind::Down, inside));
    assert_eq!(counter.get(), 0, "a press does not click");
    assert!(
        source.state().contains(InteractionState::PRESSED),
        "the press is reported to the source"
    );
    composition.handle_event(&pointer(PointerEventKind::Up, inside));
    assert_eq!(counter.get(), 1);
    assert!(
        !source.state().contains(InteractionState::PRESSED),
        "the release clears the press"
    );
}

#[test]
fn a_release_outside_the_button_does_not_click() {
    let size = Size::new(240.0, 80.0);
    let button = TestButton::new().with_labels(&["Save"]);
    let counter = Rc::clone(&button.counter);
    let mut composition = composed(button, size);
    composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(10.0, 10.0)));
    composition.handle_event(&pointer(PointerEventKind::Up, Offset::new(230.0, 70.0)));
    assert_eq!(counter.get(), 0, "a drag that leaves the button is not a click");
}

#[test]
fn a_disabled_button_neither_clicks_nor_paints_its_container() {
    let theme = Theme::light();
    let button = TestButton::new()
        .with_flags(ButtonFlags::HOVERABLE | ButtonFlags::FOCUSABLE)
        .with_border(BorderStroke::outline(&ColorScheme::light()))
        .with_labels(&["Save"]);
    let counter = Rc::clone(&button.counter);
    let colors = button.palette.resolve(&ColorScheme::light());
    let mut node = button.node();
    let rects = recorded(&mut node, &theme, Size::new(120.0, 40.0));
    assert!(
        rects[0].1.color == colors.disabled_container,
        "a disabled button draws the disabled container"
    );
    let stroke = rects[1].1;
    assert_eq!(stroke.color.alpha, defaults::DISABLED_OUTLINE_ALPHA);
    let mut composition = composed(
        TestButton::new()
            .with_flags(ButtonFlags::HOVERABLE)
            .with_labels(&["Save"]),
        Size::new(240.0, 80.0),
    );
    composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(10.0, 10.0)));
    composition.handle_event(&pointer(PointerEventKind::Up, Offset::new(10.0, 10.0)));
    assert_eq!(counter.get(), 0, "a disabled button answers nothing");
}

#[test]
fn a_toggle_flips_its_selection_and_clicks() {
    let size = Size::new(240.0, 80.0);
    let button = TestButton::new()
        .with_flags(ButtonFlags::STANDARD | ButtonFlags::SELECTABLE)
        .with_labels(&["Bold"]);
    let counter = Rc::clone(&button.counter);
    let source = button.interaction_source();
    let mut composition = composed(button, size);
    let inside = Offset::new(10.0, 10.0);
    composition.handle_event(&pointer(PointerEventKind::Down, inside));
    composition.handle_event(&pointer(PointerEventKind::Up, inside));
    assert_eq!(counter.get(), 1);
    assert!(
        source
            .state()
            .contains(InteractionState::SELECTED),
        "the release selected the toggle"
    );
    composition.handle_event(&pointer(PointerEventKind::Down, inside));
    composition.handle_event(&pointer(PointerEventKind::Up, inside));
    assert!(
        !source
            .state()
            .contains(InteractionState::SELECTED),
        "the second release cleared it again"
    );
    assert_eq!(counter.get(), 2);
}

#[test]
fn a_selected_toggle_paints_the_container_and_the_label_of_the_same_state() {
    let theme = Theme::light();
    let colors = ButtonColors::tonal(&theme.colors);
    let source = InteractionSource::new();
    let mut node = TextButton::new(|| {}, "Bold")
        .flags(ButtonFlags::STANDARD | ButtonFlags::SELECTABLE)
        .colors(ButtonPalette::fixed(colors))
        .interaction_source(source.clone())
        .node();
    let size = Size::new(120.0, 40.0);
    let (resting, rest_label) = replay(&mut node, &theme, size, |r| {
        (r.rects[0].2.color, r.runs[0].1.color)
    });
    assert_eq!(resting, colors.container);
    assert_eq!(rest_label, colors.content);
    source.set(InteractionState::SELECTED);
    let (selected, label) = replay(&mut node, &theme, size, |r| {
        (r.rects[0].2.color, r.runs[0].1.color)
    });
    assert_eq!(selected, colors.selected_container);
    assert_eq!(
        label, colors.selected_content,
        "the label follows the container into the on-color of the role it took"
    );
}

#[test]
fn a_focusable_button_takes_the_focus_on_press() {
    let size = Size::new(240.0, 80.0);
    let button = TestButton::new()
        .with_flags(ButtonFlags::STANDARD | ButtonFlags::FOCUSABLE)
        .with_labels(&["Save"]);
    let source = button.interaction_source();
    let mut composition = composed(button, size);
    composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(10.0, 10.0)));
    let focused: Vec<usize> = Vec::from(composition.focus().unwrap_or_default());
    assert_eq!(
        focused,
        vec![0, 0],
        "the row is the first child of the container and the button the first child of the row"
    );
    assert!(
        source.state().contains(InteractionState::FOCUSED),
        "the focus is reported to the source"
    );
    assert_eq!(
        source.state().bits() & InteractionState::PRESSED.bits(),
        InteractionState::PRESSED.bits(),
        "taking the focus does not swallow the press"
    );
}
