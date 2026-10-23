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

//! The widgets that decorate a window.
//!
//! A window of the Codevar editor is drawn by the application, not by
//! the windowing system: the title bar, the buttons of the window and
//! the edges that resize it are widgets of the tree. The widgets in
//! this module draw them and report what the user did with them.
//!
//! * [`TitleBar`] draws the title of the window and its buttons.
//! * [`WindowFrame`] puts a title bar above the content of the
//!   window.
//! * [`ResizeHandle`] covers one edge of the window and reports the
//!   drag that resizes it.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::color::Color;
use crate::event::{Event, EventResult, PointerEventKind};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::text::TextStyle;
use crate::theme::Theme;
use crate::widget::{DrawScope, EventScope, MeasureScope, PlaceScope, Widget};

/// A button of the title bar of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowButton {
    /// The button that minimizes the window.
    Minimize,
    /// The button that maximizes or restores the window.
    Maximize,
    /// The button that closes the window.
    Close,
}

impl WindowButton {
    /// The width of the button, in pixels.
    pub const SIZE: f32 = 32.0;

    /// The symbol the button draws.
    #[must_use]
    pub const fn glyph(self) -> &'static str {
        match self {
            Self::Minimize => "\u{2500}",
            Self::Maximize => "\u{25a1}",
            Self::Close => "\u{2715}",
        }
    }

    /// The color the button is drawn with when it is hovered.
    #[must_use]
    pub fn hover_color(self, theme: &Theme) -> Color {
        match self {
            Self::Close => theme.colors.error,
            _ => theme.colors.surface_variant,
        }
    }
}

/// The title bar of a window.
///
/// The title bar spans the whole width of the window, draws the title
/// and the buttons of the window, and reports the button the user
/// pressed.
#[derive(Clone)]
pub struct TitleBar {
    /// The title of the window.
    title: String,
    /// The buttons of the window, from the left to the right.
    buttons: Vec<WindowButton>,
    /// The height of the title bar.
    height: f32,
    /// The handler called when a button is pressed.
    on_click: Option<Rc<dyn Fn(WindowButton)>>,
    /// The rectangle of every button, filled during the place pass.
    button_rects: Vec<(WindowButton, Rect)>,
    /// The button the pointer is over.
    hovered: Option<WindowButton>,
    /// The button the pointer went down on.
    pressed: Option<WindowButton>,
}

impl core::fmt::Debug for TitleBar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TitleBar")
            .field("title", &self.title)
            .field("buttons", &self.buttons)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl TitleBar {
    /// Creates a title bar with the given title and no button.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            buttons: Vec::new(),
            height: 32.0,
            on_click: None,
            button_rects: Vec::new(),
            hovered: None,
            pressed: None,
        }
    }

    /// The title bar with the given buttons.
    #[must_use]
    pub fn buttons(mut self, buttons: Vec<WindowButton>) -> Self {
        self.buttons = buttons;
        self
    }

    /// The title bar with the given height.
    #[must_use]
    pub const fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// The title bar that calls `on_click` when a button is pressed.
    #[must_use]
    pub fn on_click(mut self, on_click: impl Fn(WindowButton) + 'static) -> Self {
        self.on_click = Some(Rc::new(on_click));
        self
    }

    /// The title of the window.
    #[inline]
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The button the pointer is over.
    #[inline]
    #[must_use]
    pub const fn hovered(&self) -> Option<WindowButton> {
        self.hovered
    }

    /// The style the title is drawn with.
    fn title_style(&self, theme: &Theme) -> TextStyle {
        theme
            .typography
            .label
            .clone()
            .with_color(theme.colors.on_surface)
    }

    /// The button at `position`, if any.
    fn button_at(&self, position: Offset) -> Option<WindowButton> {
        self.button_rects
            .iter()
            .find(|(_, rect)| rect.contains(position))
            .map(|(button, _)| *button)
    }
}

impl Widget for TitleBar {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let theme = scope.theme();
        let style = self.title_style(theme);
        let metrics = theme.metrics.as_ref();
        let title_width = metrics.advance(&self.title, &style);
        let buttons_width = self.buttons.len() as f32 * WindowButton::SIZE;
        let width = (title_width + buttons_width).min(constraints.max_width);
        constraints.enforce(Size::new(width, self.height))
    }

    fn place(&mut self, _scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.button_rects.clear();
        let mut x = bounds.right();
        for button in self.buttons.iter().rev() {
            x -= WindowButton::SIZE;
            let rect = Rect::from_min_size(
                Offset::new(x, bounds.top()),
                WindowButton::SIZE,
                bounds.size.height,
            );
            self.button_rects.push((*button, rect));
        }
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let theme = scope.theme();
        let style = self.title_style(theme);
        let metrics = theme.metrics.as_ref();
        scope.fill_rect(bounds, theme.colors.surface_variant);

        let title_width = metrics.advance(&self.title, &style);
        let buttons_width = self.buttons.len() as f32 * WindowButton::SIZE;
        let available = bounds.size.width - buttons_width;
        let x = bounds.left() + (available - title_width) * 0.5;
        let baseline = bounds.top() + (bounds.size.height - style.font_size) * 0.5;
        scope.draw_text(
            Offset::new(x - bounds.left(), baseline - bounds.top()),
            Rc::from(self.title.as_str()),
            style,
        );

        for (button, rect) in &self.button_rects {
            let color = if self.hovered == Some(*button) {
                button.hover_color(theme)
            } else {
                theme.colors.surface_variant
            };
            scope.fill_rect(*rect, color);
            let glyph = scope
                .theme()
                .typography
                .label
                .clone()
                .with_color(theme.colors.on_surface);
            let glyph_width = metrics.advance(button.glyph(), &glyph);
            let glyph_x = rect.left() + (rect.size.width - glyph_width) * 0.5;
            let glyph_y = rect.top() + (rect.size.height - glyph.font_size) * 0.5;
            scope.draw_text(
                Offset::new(glyph_x - bounds.left(), glyph_y - bounds.top()),
                Rc::from(button.glyph()),
                glyph,
            );
        }
    }

    fn on_event(&mut self, _scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        let Event::Pointer(pointer) = event else {
            return EventResult::Ignored;
        };
        match pointer.kind {
            PointerEventKind::Move => {
                self.hovered = self.button_at(pointer.position);
                EventResult::Ignored
            }
            PointerEventKind::Leave => {
                self.hovered = None;
                self.pressed = None;
                EventResult::Ignored
            }
            PointerEventKind::Down => {
                self.pressed = self.button_at(pointer.position);
                if self.pressed.is_some() {
                    return EventResult::Consumed;
                }
                EventResult::Ignored
            }
            PointerEventKind::Up => {
                let pressed = self.pressed.take();
                let released = self.button_at(pointer.position);
                self.pressed = None;
                if pressed.is_some() && pressed == released {
                    if let Some(handler) = &self.on_click
                        && let Some(button) = pressed
                    {
                        handler(button);
                    }
                    return EventResult::Consumed;
                }
                EventResult::Ignored
            }
            _ => EventResult::Ignored,
        }
    }

    fn hit_testable(&self) -> bool {
        true
    }
}

/// A window with a title bar above its content.
///
/// The first child of the frame is the title bar and the second one is
/// the content of the window. The title bar keeps the height it asks
/// for and the content takes the rest of the window.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WindowFrame {
    /// The space between the title bar and the content.
    spacing: f32,
}

impl WindowFrame {
    /// Creates a frame without spacing.
    #[must_use]
    pub const fn new() -> Self {
        Self { spacing: 0.0 }
    }

    /// The frame with `spacing` pixels between the title bar and the
    /// content.
    #[must_use]
    pub const fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }
}

impl Widget for WindowFrame {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let title = scope.measure_child(
            0,
            Constraints::new(0.0, constraints.max_width, 0.0, constraints.max_height),
        );
        let remaining = (constraints.max_height - title.height - self.spacing).max(0.0);
        let content = scope.measure_child(1, Constraints::new(0.0, constraints.max_width, 0.0, remaining));
        constraints.enforce(Size::new(
            title.width.max(content.width),
            title.height + self.spacing + content.height,
        ))
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, _bounds: Rect) {
        let title = scope.child_size(0);
        scope.place_child(0, Offset::ZERO);
        scope.place_child(1, Offset::new(0.0, title.height + self.spacing));
    }
}

/// The edge of a window that a resize handle covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResizeEdge {
    /// The top edge.
    Top,
    /// The bottom edge.
    Bottom,
    /// The left edge.
    Left,
    /// The right edge.
    Right,
    /// The top-left corner.
    TopLeft,
    /// The top-right corner.
    TopRight,
    /// The bottom-left corner.
    BottomLeft,
    /// The bottom-right corner.
    BottomRight,
}

impl ResizeEdge {
    /// The thickness of the handle that covers the edge.
    pub const THICKNESS: f32 = 6.0;

    /// Whether the edge is on the top of the window.
    #[must_use]
    pub const fn is_top(self) -> bool {
        matches!(self, Self::Top | Self::TopLeft | Self::TopRight)
    }

    /// Whether the edge is on the bottom of the window.
    #[must_use]
    pub const fn is_bottom(self) -> bool {
        matches!(self, Self::Bottom | Self::BottomLeft | Self::BottomRight)
    }

    /// Whether the edge is on the left of the window.
    #[must_use]
    pub const fn is_left(self) -> bool {
        matches!(self, Self::Left | Self::TopLeft | Self::BottomLeft)
    }

    /// Whether the edge is on the right of the window.
    #[must_use]
    pub const fn is_right(self) -> bool {
        matches!(self, Self::Right | Self::TopRight | Self::BottomRight)
    }
}

/// A handle that covers one edge of a window and reports the drag
/// that resizes it.
///
/// The handle draws nothing; it only tells the application that the
/// user grabbed the edge, so the application can resize the window.
#[derive(Clone)]
pub struct ResizeHandle {
    /// The edge the handle covers.
    edge: ResizeEdge,
    /// The handler called when the handle is grabbed.
    on_grab: Option<Rc<dyn Fn(ResizeEdge)>>,
    /// Whether the pointer went down on the handle.
    pressed: bool,
}

impl core::fmt::Debug for ResizeHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResizeHandle")
            .field("edge", &self.edge)
            .field("pressed", &self.pressed)
            .finish_non_exhaustive()
    }
}

impl ResizeHandle {
    /// Creates a handle that covers `edge`.
    #[must_use]
    pub const fn new(edge: ResizeEdge) -> Self {
        Self {
            edge,
            on_grab: None,
            pressed: false,
        }
    }

    /// The handle that calls `on_grab` when it is grabbed.
    #[must_use]
    pub fn on_grab(mut self, on_grab: impl Fn(ResizeEdge) + 'static) -> Self {
        self.on_grab = Some(Rc::new(on_grab));
        self
    }

    /// The edge the handle covers.
    #[inline]
    #[must_use]
    pub const fn edge(&self) -> ResizeEdge {
        self.edge
    }

    /// Whether the pointer went down on the handle.
    #[inline]
    #[must_use]
    pub const fn is_pressed(&self) -> bool {
        self.pressed
    }
}

impl Widget for ResizeHandle {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::ZERO)
    }

    fn on_event(&mut self, _scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        let Event::Pointer(pointer) = event else {
            return EventResult::Ignored;
        };
        match pointer.kind {
            PointerEventKind::Down => {
                self.pressed = true;
                if let Some(handler) = &self.on_grab {
                    handler(self.edge);
                }
                EventResult::Consumed
            }
            PointerEventKind::Up | PointerEventKind::Cancel | PointerEventKind::Leave => {
                self.pressed = false;
                EventResult::Ignored
            }
            _ => EventResult::Ignored,
        }
    }

    fn hit_testable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::paint::{DrawList, DrawOp};

    #[test]
    fn a_title_bar_is_as_wide_as_the_window() {
        let theme = Theme::light();
        let mut node = Node::new(TitleBar::new("Codevar"));
        let size = node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, f32::INFINITY));
        assert_eq!(size.height, 32.0);
        assert!(size.width <= 400.0);
    }

    #[test]
    fn a_title_bar_draws_its_background_and_its_title() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut node = Node::new(TitleBar::new("Codevar"));
        node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, f32::INFINITY));
        node.place(Rect::new(Offset::ZERO, Size::new(400.0, 32.0)));
        node.draw(&theme, &mut list);
        assert!(list.len() >= 2);
        assert!(matches!(list.ops().next(), Some(DrawOp::Rect { .. })));
        assert!(matches!(list.ops().nth(1), Some(DrawOp::Text { .. })));
    }

    #[test]
    fn a_window_frame_puts_the_content_below_the_title_bar() {
        let theme = Theme::light();
        let mut node = Node::new(WindowFrame::new())
            .child(Node::new(TitleBar::new("Codevar")))
            .child(Node::new(crate::widgets::Column::new()));
        node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, 300.0));
        node.place(Rect::new(Offset::ZERO, Size::new(400.0, 300.0)));
        let title = node.children()[0].bounds();
        let content = node.children()[1].bounds();
        assert_eq!(title.top(), 0.0);
        assert_eq!(content.top(), 32.0);
    }

    #[test]
    fn a_pressed_button_calls_the_handler() {
        let theme = Theme::light();
        let pressed = Rc::new(core::cell::Cell::new(0_u32));
        let observed = Rc::clone(&pressed);
        let mut node = Node::new(
            TitleBar::new("Codevar")
                .buttons(vec![WindowButton::Close])
                .on_click(move |_button| observed.set(observed.get() + 1)),
        );
        node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, f32::INFINITY));
        node.place(Rect::new(Offset::ZERO, Size::new(400.0, 32.0)));

        let down = Event::Pointer(crate::event::PointerEvent::new(
            PointerEventKind::Down,
            Offset::new(390.0, 16.0),
        ));
        let up = Event::Pointer(crate::event::PointerEvent::new(
            PointerEventKind::Up,
            Offset::new(390.0, 16.0),
        ));
        node.widget_mut().on_event(
            &mut crate::widget::EventScope::new(&mut Vec::new(), &theme, Vec::new(), &mut None),
            &down,
        );
        node.widget_mut().on_event(
            &mut crate::widget::EventScope::new(&mut Vec::new(), &theme, Vec::new(), &mut None),
            &up,
        );
        assert_eq!(pressed.get(), 1);
    }

    #[test]
    fn a_resize_handle_reports_the_edge_it_covers() {
        let theme = Theme::light();
        let grabbed = Rc::new(core::cell::Cell::new(0_u32));
        let observed = Rc::clone(&grabbed);
        let mut node = Node::new(ResizeHandle::new(ResizeEdge::Left).on_grab(move |_edge| {
            observed.set(observed.get() + 1);
        }));
        node.measure(&theme, Constraints::unbounded());
        let down = Event::Pointer(crate::event::PointerEvent::new(
            PointerEventKind::Down,
            Offset::new(1.0, 1.0),
        ));
        node.widget_mut().on_event(
            &mut crate::widget::EventScope::new(&mut Vec::new(), &theme, Vec::new(), &mut None),
            &down,
        );
        assert_eq!(grabbed.get(), 1);
    }
}
