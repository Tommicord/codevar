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

//! The [`Widget`] trait and the scopes a widget is handed.
//!
//! # The contract
//!
//! A widget is a retained node of the tree. Its life inside the tree
//! is driven by four phases, and each phase has its own scope type so
//! that a widget can only reach what is valid in that phase:
//!
//! | Phase | Hook | Scope | The widget may |
//! | --- | --- | --- | --- |
//! | measure | [`Widget::measure`] | [`MeasureScope`] | ask for the size of its children |
//! | place | [`Widget::place`] | [`PlaceScope`] | give its children their position |
//! | draw | [`Widget::draw`] | [`DrawScope`] | paint into the display list |
//! | event | [`Widget::on_event`] | [`EventScope`] | forward events to its children |
//!
//! Every hook has a default implementation, so a widget only overrides
//! what it actually needs: a leaf that paints a rectangle implements
//! [`Widget::measure`] and [`Widget::draw`] and nothing else.
//!
//! # The measure / place split
//!
//! Measuring and placing are two separate passes, exactly as in
//! Jetpack Compose. Measuring asks a widget how big it wants to be
//! inside a set of [`Constraints`]; placing tells it where it ended
//! up. The split is what makes it possible to lay a widget out before
//! its siblings are known, which is how a scroll view measures its
//! content without knowing the size of the viewport.
//!
//! # Coordinates
//!
//! Every scope works in the coordinate space of the widget that owns
//! it: the origin is the top-left corner of the widget, not of the
//! window. The framework translates the coordinates when it records
//! them, so a widget never has to know where it sits.

use alloc::vec::Vec;

use crate::event::{Event, EventResult};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::paint::{CornerRadii, DrawList, Paint};
use crate::theme::Theme;

/// The identity of a widget among its siblings.
///
/// A widget that does not name itself is identified by its position,
/// which is enough for a static tree. A widget that names itself keeps
/// its state when its position changes, which is what a list needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    /// The widget is identified by its position among its siblings.
    Index(usize),
    /// The widget is identified by a name chosen by the application.
    Named(&'static str),
    /// The widget is identified by a number chosen by the application.
    Number(u64),
}

impl Key {
    /// The name of the key, for logs and debugging.
    #[must_use]
    pub const fn describe(&self) -> &'static str {
        match self {
            Self::Index(_) => "index",
            Self::Named(_) => "named",
            Self::Number(_) => "number",
        }
    }
}

/// A widget of the Kuitui tree.
///
/// The trait is object-safe: a tree stores its widgets as
/// `Box<dyn Widget>`, so an application can mix widgets of different
/// types in one tree without boxing every one of them by hand.
pub trait Widget {
    /// The identity of this widget among its siblings.
    ///
    /// Returning `None` — the default — identifies the widget by its
    /// position, which is enough for a static tree. A widget that
    /// names itself keeps its identity when its position changes,
    /// which is what a list of items needs.
    fn key(&self) -> Option<Key> {
        None
    }

    /// Reports the size this widget wants inside `constraints`.
    ///
    /// The returned size must fit inside the constraints: a widget
    /// that is given a maximum width may not report a wider one. Use
    /// [`Constraints::enforce`] to clamp the size of a child.
    ///
    /// The default implementation reports an empty size, which is
    /// right for a widget that draws nothing.
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let _ = scope;
        constraints.enforce(Size::ZERO)
    }

    /// Places the widget and its children inside `bounds`.
    ///
    /// `bounds` is the rectangle the widget was given by its parent,
    /// in the coordinate space of the parent. A widget that has
    /// children calls [`PlaceScope::place_child`] for each of them;
    /// the framework then places them in turn.
    ///
    /// The default implementation places nothing.
    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        let _ = (scope, bounds);
    }

    /// Paints the widget inside `bounds`.
    ///
    /// `bounds` is the rectangle the widget was given by its parent,
    /// in the coordinate space of the parent. The scope translates the
    /// coordinates the widget records into the coordinate space of the
    /// window.
    ///
    /// The children of the widget are painted *after* this hook
    /// returns, so a widget that paints a background here draws it
    /// behind its children, as Compose's `Modifier.background` does.
    ///
    /// The default implementation paints nothing.
    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let _ = (scope, bounds);
    }

    /// Handles an event that reached this widget.
    ///
    /// The framework offers an event to the deepest child that
    /// contains it first and only offers it to the widget when no
    /// child consumed it. Returning [`EventResult::Consumed`] stops
    /// the event from travelling further.
    ///
    /// The default implementation ignores the event.
    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        let _ = (scope, event);
        EventResult::Ignored
    }

    /// Whether the widget reacts to pointer events at all.
    ///
    /// A widget that returns `false` is skipped by the hit test, so
    /// the events that fall inside it reach the widgets below it. The
    /// default is `false`.
    fn hit_testable(&self) -> bool {
        false
    }

    /// How much of the free space of the parent this widget takes.
    ///
    /// A widget that returns `0.0` — the default — is laid out at the
    /// size it asked for. A widget that returns a positive number
    /// shares the free space of its parent with the other flexible
    /// widgets, in proportion to the numbers they return. This is the
    /// weight of Compose's `Modifier.weight`.
    fn flex(&self) -> f32 {
        0.0
    }
}

/// The children a widget lays out.
///
/// The framework stores the children of a node in a slice and hands
/// that slice to the scopes as a `Children`. A widget that keeps its
/// children somewhere else — in an arena, in a map keyed by name —
/// can implement this trait for its own storage and use the scopes
/// unchanged.
pub trait Children {
    /// The number of children.
    fn child_count(&self) -> usize;

    /// The size the child at `index` was measured at.
    fn child_size(&self, index: usize) -> Size;

    /// How much of the free space of the parent the child at `index`
    /// takes.
    fn child_flex(&self, index: usize) -> f32;

    /// Measures the child at `index` inside `constraints`.
    fn measure_child(&mut self, index: usize, theme: &Theme, constraints: Constraints) -> Size;

    /// Places the child at `index` at the given position.
    fn place_child(&mut self, index: usize, position: Offset);

    /// Offers an event to the child at `index`.
    fn dispatch_child(
        &mut self,
        index: usize,
        theme: &Theme,
        path: &mut Vec<usize>,
        focus: &mut Option<Vec<usize>>,
        event: &Event,
    ) -> EventResult;
}

/// The scope of a measure pass.
///
/// A widget uses it to ask for the size of its children and to reach
/// the theme it is measured with.
pub struct MeasureScope<'a> {
    /// The theme of the tree.
    theme: &'a Theme,
    /// The children of the widget being measured.
    children: &'a mut dyn Children,
}

impl<'a> MeasureScope<'a> {
    /// Creates a scope for a widget with the given children.
    #[must_use]
    pub fn new(theme: &'a Theme, children: &'a mut dyn Children) -> Self {
        Self { theme, children }
    }

    /// The theme the tree is measured with.
    #[inline]
    #[must_use]
    pub const fn theme(&self) -> &'a Theme {
        self.theme
    }

    /// The number of children of the widget being measured.
    #[inline]
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.child_count()
    }

    /// Measures the child at `index` inside `constraints`.
    ///
    /// Measuring a child twice is allowed but wasteful; the result is
    /// cached by the framework for the rest of the pass.
    ///
    /// A child that does not exist reports an empty size, so a widget
    /// does not have to guard against a child count that changed
    /// between two calls.
    #[must_use]
    pub fn measure_child(&mut self, index: usize, constraints: Constraints) -> Size {
        self.children
            .measure_child(index, self.theme, constraints)
    }

    /// The size the child at `index` was measured at.
    #[must_use]
    pub fn child_size(&self, index: usize) -> Size {
        self.children.child_size(index)
    }

    /// How much of the free space of the parent the child at `index`
    /// takes.
    #[must_use]
    pub fn child_flex(&self, index: usize) -> f32 {
        self.children.child_flex(index)
    }

    /// Measures every child with the same constraints.
    #[must_use]
    pub fn measure_children(&mut self, constraints: Constraints) -> Vec<Size> {
        let mut sizes = Vec::with_capacity(self.children.child_count());
        for index in 0..self.children.child_count() {
            sizes.push(self.measure_child(index, constraints));
        }
        sizes
    }
}

/// The scope of a place pass.
///
/// A widget uses it to give its children their position.
pub struct PlaceScope<'a> {
    /// The children of the widget being placed.
    children: &'a mut dyn Children,
    /// The origin of the widget being placed, in the coordinate space
    /// of the window.
    origin: Offset,
}

impl<'a> PlaceScope<'a> {
    /// Creates a scope for a widget with the given children.
    #[must_use]
    pub fn new(children: &'a mut dyn Children, origin: Offset) -> Self {
        Self { children, origin }
    }

    /// The number of children of the widget being placed.
    #[inline]
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.child_count()
    }

    /// The size the child at `index` was measured at.
    #[must_use]
    pub fn child_size(&self, index: usize) -> Size {
        self.children.child_size(index)
    }

    /// How much of the free space of the parent the child at `index`
    /// takes.
    #[must_use]
    pub fn child_flex(&self, index: usize) -> f32 {
        self.children.child_flex(index)
    }

    /// Places the child at `index` at `offset` from the origin of the
    /// widget.
    ///
    /// The offset is relative to the top-left corner of the widget
    /// being placed. A child that does not exist is ignored, so a
    /// widget does not have to guard against a child count that
    /// changed between two calls.
    pub fn place_child(&mut self, index: usize, offset: Offset) {
        let position = self.origin.add(offset);
        self.children.place_child(index, position);
    }
}

/// The scope of a draw pass.
///
/// A widget uses it to paint into the display list.
pub struct DrawScope<'a> {
    /// The display list of the frame.
    list: &'a mut DrawList,
    /// The theme of the tree.
    theme: &'a Theme,
    /// The origin of the widget being drawn, in the coordinate space
    /// of the window.
    origin: Offset,
}

impl<'a> DrawScope<'a> {
    /// Creates a scope for a widget at `origin`.
    #[must_use]
    pub fn new(list: &'a mut DrawList, theme: &'a Theme, origin: Offset) -> Self {
        Self { list, theme, origin }
    }

    /// The theme the tree is drawn with.
    #[inline]
    #[must_use]
    pub const fn theme(&self) -> &'a Theme {
        self.theme
    }

    /// The origin of the widget being drawn.
    #[inline]
    #[must_use]
    pub const fn origin(&self) -> Offset {
        self.origin
    }

    /// Fills a rectangle.
    pub fn fill_rect(&mut self, rect: Rect, color: crate::color::Color) {
        self.list.rect(
            Rect::new(self.origin.add(rect.origin), rect.size),
            CornerRadii::ZERO,
            Paint::fill(color),
        );
    }

    /// Fills a rectangle with rounded corners.
    pub fn fill_rounded_rect(&mut self, rect: Rect, radii: CornerRadii, color: crate::color::Color) {
        self.list.rect(
            Rect::new(self.origin.add(rect.origin), rect.size),
            radii,
            Paint::fill(color),
        );
    }

    /// Outlines a rectangle.
    pub fn stroke_rect(&mut self, rect: Rect, radii: CornerRadii, color: crate::color::Color, width: f32) {
        self.list.rect(
            Rect::new(self.origin.add(rect.origin), rect.size),
            radii,
            Paint::stroke(color, width),
        );
    }

    /// Draws a straight line.
    pub fn draw_line(&mut self, from: Offset, to: Offset, color: crate::color::Color, width: f32) {
        self.list.line(
            self.origin.add(from),
            self.origin.add(to),
            Paint::stroke(color, width),
        );
    }

    /// Draws a run of text with its baseline origin at `origin`.
    pub fn draw_text(&mut self, origin: Offset, text: alloc::rc::Rc<str>, style: crate::text::TextStyle) {
        self.list
            .text(self.origin.add(origin), text, style);
    }

    /// Restricts the following operations to `rect`.
    ///
    /// The rectangle is relative to the widget being drawn.
    pub fn push_clip(&mut self, rect: Rect) {
        self.list
            .push_clip(Rect::new(self.origin.add(rect.origin), rect.size));
    }

    /// Ends the innermost clip.
    pub fn pop_clip(&mut self) {
        self.list.pop_clip();
    }
}

/// The scope of an event dispatch.
///
/// A widget uses it to forward an event to its children and to manage
/// the focus.
pub struct EventScope<'a> {
    /// The children of the widget that receives the event.
    children: &'a mut dyn Children,
    /// The theme of the tree.
    theme: &'a Theme,
    /// The path of the widget that receives the event.
    path: Vec<usize>,
    /// The path of the focused widget, if any.
    focus: &'a mut Option<Vec<usize>>,
}

impl<'a> EventScope<'a> {
    /// Creates a scope for the widget at `path`.
    #[must_use]
    pub fn new(
        children: &'a mut dyn Children,
        theme: &'a Theme,
        path: Vec<usize>,
        focus: &'a mut Option<Vec<usize>>,
    ) -> Self {
        Self {
            children,
            theme,
            path,
            focus,
        }
    }

    /// The theme of the tree.
    #[inline]
    #[must_use]
    pub const fn theme(&self) -> &'a Theme {
        self.theme
    }

    /// The number of children of the widget that receives the event.
    #[inline]
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.child_count()
    }

    /// Offers an event to the child at `index`.
    ///
    /// The event is offered to the child and to its own children; the
    /// result says whether the event was consumed.
    pub fn dispatch_to_child(&mut self, index: usize, event: &Event) -> EventResult {
        self.children
            .dispatch_child(index, self.theme, &mut self.path, &mut *self.focus, event)
    }

    /// Gives the focus to the widget that receives the event.
    pub fn request_focus(&mut self) {
        *self.focus = Some(self.path.clone());
    }

    /// Takes the focus away from every widget.
    pub fn clear_focus(&mut self) {
        *self.focus = None;
    }

    /// Whether the widget that receives the event has the focus.
    #[must_use]
    pub fn is_focused(&self) -> bool {
        self.focus.as_ref() == Some(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;

    /// A container without children, used to build a scope.
    struct NoChildren;

    impl Children for NoChildren {
        fn child_count(&self) -> usize {
            0
        }

        fn measure_child(&mut self, _index: usize, _theme: &Theme, constraints: Constraints) -> Size {
            constraints.enforce(Size::ZERO)
        }

        fn child_size(&self, _index: usize) -> Size {
            Size::ZERO
        }

        fn child_flex(&self, _index: usize) -> f32 {
            0.0
        }

        fn place_child(&mut self, _index: usize, _position: Offset) {}

        fn dispatch_child(
            &mut self,
            _index: usize,
            _theme: &Theme,
            _path: &mut Vec<usize>,
            _focus: &mut Option<Vec<usize>>,
            _event: &Event,
        ) -> EventResult {
            EventResult::Ignored
        }
    }

    /// A widget that only paints a square of a fixed size.
    struct Square {
        side: f32,
    }

    impl Widget for Square {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(self.side, self.side))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, Color::RED);
        }
    }

    #[test]
    fn a_widget_reports_the_size_it_wants() {
        let theme = Theme::light();
        let mut children = NoChildren;
        let mut scope = MeasureScope::new(&theme, &mut children);
        let mut square = Square { side: 12.0 };
        let size = square.measure(&mut scope, Constraints::unbounded());
        assert_eq!(size, Size::new(12.0, 12.0));
    }

    #[test]
    fn a_widget_is_clamped_by_its_constraints() {
        let theme = Theme::light();
        let mut children = NoChildren;
        let mut scope = MeasureScope::new(&theme, &mut children);
        let mut square = Square { side: 12.0 };
        let size = square.measure(&mut scope, Constraints::fixed(5.0, 5.0));
        assert_eq!(size, Size::new(5.0, 5.0));
    }

    #[test]
    fn a_widget_draws_into_the_display_list() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut scope = DrawScope::new(&mut list, &theme, Offset::new(10.0, 20.0));
        let mut square = Square { side: 4.0 };
        square.draw(&mut scope, Rect::from_min_size(Offset::ZERO, 4.0, 4.0));
        assert_eq!(list.len(), 1);
        let op = list
            .ops()
            .next()
            .expect("a rectangle was recorded");
        let any: &dyn core::any::Any = op;
        let rect = any
            .downcast_ref::<crate::paint::RectOp>()
            .expect("expected a rectangle");
        assert_eq!(rect.rect.origin, Offset::new(10.0, 20.0));
    }
}
