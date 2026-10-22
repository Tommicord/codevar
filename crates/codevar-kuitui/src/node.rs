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

//! The node of the widget tree and the operations the framework runs
//! on it.
//!
//! A [`Node`] couples a [`Widget`] with the children it has and with
//! the modifier that decorates it. The framework drives the tree
//! through four operations — measure, place, draw and dispatch — and
//! this module implements them.
//!
//! # The layout
//!
//! Measuring happens from the root down: every node asks its children
//! for their size and reports its own. Placing happens from the root
//! down as well, but in two steps: the widget gives its children
//! their position, and the framework then places each child in turn.
//! Because the position of a child is known before the child is
//! placed, a child never has to ask where it is.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;

use crate::color::Color;
use crate::event::{Event, EventResult};
use crate::geometry::{Constraints, EdgeInsets, Offset, Rect, Size};
use crate::paint::{CornerRadii, DrawList, Paint};
use crate::theme::Theme;
use crate::widget::{Children, DrawScope, EventScope, Key, MeasureScope, PlaceScope, Widget};

/// A decoration applied to a widget.
///
/// A modifier wraps a widget the way Compose's `Modifier` does: it
/// changes the size the widget is measured with, the space it is
/// given and the way it is painted, without the widget knowing. The
/// operations are applied in the order they were added.
#[derive(Clone, Default)]
pub struct Modifier {
    /// The operations of the modifier, in order.
    ops: Vec<ModifierOp>,
}

/// The handler a [`Modifier::clickable`] modifier calls.
pub type ClickHandler = Rc<dyn Fn(&Event) -> EventResult>;

/// One operation of a [`Modifier`].
#[derive(Clone)]
enum ModifierOp {
    /// Reserves space around the widget.
    Padding(EdgeInsets),
    /// Forces the size of the widget.
    Size {
        /// The forced width, if any.
        width: Option<f32>,
        /// The forced height, if any.
        height: Option<f32>,
    },
    /// Paints a background behind the widget.
    Background(Color),
    /// Paints a border around the widget.
    Border {
        /// The color of the border.
        color: Color,
        /// The width of the border.
        width: f32,
        /// The radii of the corners of the border.
        radii: CornerRadii,
    },
    /// Clips the widget to its own rectangle.
    Clip,
    /// Handles the events that reach the widget.
    Clickable(ClickHandler),
}

impl core::fmt::Debug for Modifier {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Modifier")
            .field("operations", &self.ops.len())
            .finish()
    }
}

impl Modifier {
    /// A modifier that changes nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self { ops: Vec::new() }
    }

    /// Reserves `padding` around the widget.
    #[must_use]
    pub fn padding(mut self, padding: EdgeInsets) -> Self {
        self.ops.push(ModifierOp::Padding(padding));
        self
    }

    /// Forces the width and the height of the widget.
    ///
    /// A `None` leaves the axis alone.
    #[must_use]
    pub fn size(mut self, width: Option<f32>, height: Option<f32>) -> Self {
        self.ops.push(ModifierOp::Size { width, height });
        self
    }

    /// Paints `color` behind the widget.
    #[must_use]
    pub fn background(mut self, color: Color) -> Self {
        self.ops.push(ModifierOp::Background(color));
        self
    }

    /// Paints a border of `width` pixels around the widget.
    #[must_use]
    pub fn border(mut self, color: Color, width: f32) -> Self {
        self.ops.push(ModifierOp::Border {
            color,
            width,
            radii: CornerRadii::ZERO,
        });
        self
    }

    /// Paints a border with rounded corners around the widget.
    #[must_use]
    pub fn rounded_border(mut self, color: Color, width: f32, radii: CornerRadii) -> Self {
        self.ops
            .push(ModifierOp::Border { color, width, radii });
        self
    }

    /// Clips the widget to its own rectangle.
    #[must_use]
    pub fn clip(mut self) -> Self {
        self.ops.push(ModifierOp::Clip);
        self
    }

    /// Handles the events that reach the widget.
    ///
    /// The handler is called with the event after the children and the
    /// widget had their chance to consume it. Returning
    /// [`EventResult::Consumed`] stops the event from travelling
    /// further.
    #[must_use]
    pub fn clickable(mut self, handler: impl Fn(&Event) -> EventResult + 'static) -> Self {
        self.ops
            .push(ModifierOp::Clickable(Rc::new(handler)));
        self
    }

    /// Whether the modifier changes nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The space reserved around the widget.
    fn padding_insets(&self) -> EdgeInsets {
        let mut padding = EdgeInsets::ZERO;
        for op in &self.ops {
            if let ModifierOp::Padding(insets) = op {
                padding.left += insets.left;
                padding.top += insets.top;
                padding.right += insets.right;
                padding.bottom += insets.bottom;
            }
        }
        padding
    }

    /// The constraints the widget itself is measured with.
    fn inner_constraints(&self, constraints: Constraints) -> Constraints {
        let padding = self.padding_insets();
        Constraints::new(
            (constraints.min_width - padding.horizontal()).max(0.0),
            (constraints.max_width - padding.horizontal()).max(0.0),
            (constraints.min_height - padding.vertical()).max(0.0),
            (constraints.max_height - padding.vertical()).max(0.0),
        )
    }

    /// The size the widget takes once the padding is added.
    fn outer_size(&self, measured: Size) -> Size {
        let padding = self.padding_insets();
        let mut size = Size::new(
            measured.width + padding.horizontal(),
            measured.height + padding.vertical(),
        );
        for op in &self.ops {
            if let ModifierOp::Size { width, height } = op {
                if let Some(width) = width {
                    size.width = *width;
                }
                if let Some(height) = height {
                    size.height = *height;
                }
            }
        }
        size
    }

    /// The rectangle the widget itself is given inside `bounds`.
    fn inner_bounds(&self, bounds: Rect) -> Rect {
        let padding = self.padding_insets();
        Rect::new(
            Offset::new(bounds.left() + padding.left, bounds.top() + padding.top),
            Size::new(
                (bounds.size.width - padding.horizontal()).max(0.0),
                (bounds.size.height - padding.vertical()).max(0.0),
            ),
        )
    }

    /// Whether the modifier clips the widget.
    fn clips(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, ModifierOp::Clip))
    }

    /// Whether the modifier handles the events of the widget.
    fn is_clickable(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, ModifierOp::Clickable(_)))
    }

    /// The background of the widget, if any.
    fn background_color(&self) -> Option<Color> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Background(color) => Some(*color),
            _ => None,
        })
    }

    /// The border of the widget, if any.
    fn border_paint(&self) -> Option<(Color, f32, CornerRadii)> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Border { color, width, radii } => Some((*color, *width, *radii)),
            _ => None,
        })
    }

    /// The click handler of the widget, if any.
    fn click_handler(&self) -> Option<ClickHandler> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Clickable(handler) => Some(Rc::clone(handler)),
            _ => None,
        })
    }
}

/// A node of the widget tree.
///
/// A node owns a widget, the children the widget lays out, and the
/// modifier that decorates it. The framework drives the tree; an
/// application builds it.
pub struct Node {
    /// The widget of the node.
    widget: Box<dyn Widget>,
    /// The children of the node.
    children: Vec<Node>,
    /// The decoration of the node.
    modifier: Modifier,
    /// The size the node was measured at.
    pub(crate) size: Size,
    /// The rectangle the node was placed in.
    pub(crate) bounds: Rect,
}

impl Node {
    /// Creates a node holding `widget`.
    #[must_use]
    pub fn new(widget: impl Widget + 'static) -> Self {
        Self {
            widget: Box::new(widget),
            children: Vec::new(),
            modifier: Modifier::none(),
            size: Size::ZERO,
            bounds: Rect::ZERO,
        }
    }

    /// Creates a node holding `widget` and identifying it with `key`.
    #[must_use]
    pub fn with_key(key: Key, widget: impl Widget + 'static) -> Self {
        Self {
            widget: Box::new(Identified {
                key,
                inner: Box::new(widget),
            }),
            children: Vec::new(),
            modifier: Modifier::none(),
            size: Size::ZERO,
            bounds: Rect::ZERO,
        }
    }

    /// Adds a child to the node.
    #[must_use]
    pub fn child(mut self, child: Node) -> Self {
        self.children.push(child);
        self
    }

    /// Adds several children to the node.
    #[must_use]
    pub fn with_children(mut self, children: impl IntoIterator<Item = Node>) -> Self {
        self.children.extend(children);
        self
    }

    /// Decorates the node.
    #[must_use]
    pub fn modifier(mut self, modifier: Modifier) -> Self {
        self.modifier = modifier;
        self
    }

    /// The widget of the node.
    #[inline]
    #[must_use]
    pub fn widget(&self) -> &dyn Widget {
        self.widget.as_ref()
    }

    /// The widget of the node, for mutation.
    #[inline]
    pub fn widget_mut(&mut self) -> &mut (dyn Widget + 'static) {
        self.widget.as_mut()
    }

    /// The children of the node.
    #[inline]
    #[must_use]
    pub fn children(&self) -> &[Node] {
        &self.children
    }

    /// The children of the node, for mutation.
    #[inline]
    pub fn children_mut(&mut self) -> &mut [Node] {
        &mut self.children
    }

    /// The modifier of the node.
    #[inline]
    #[must_use]
    pub const fn modifier_ref(&self) -> &Modifier {
        &self.modifier
    }

    /// The size the node was measured at.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// The rectangle the node was placed in.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    /// The identity of the node among its siblings.
    ///
    /// A widget that does not name itself is identified by its
    /// position.
    #[must_use]
    pub fn effective_key(&self, index: usize) -> Key {
        self.widget.key().unwrap_or(Key::Index(index))
    }

    /// Whether the node reacts to pointer events.
    #[must_use]
    pub fn hit_testable(&self) -> bool {
        self.widget.hit_testable() || self.modifier.is_clickable()
    }

    /// Measures the node inside `constraints`.
    ///
    /// The framework calls this for every node of the tree; an
    /// application that drives the tree by hand may call it too.
    pub fn measure(&mut self, theme: &Theme, constraints: Constraints) -> Size {
        let inner = self.modifier.inner_constraints(constraints);
        let measured = {
            let Node { widget, children, .. } = self;
            let mut scope = MeasureScope::new(theme, children);
            widget.measure(&mut scope, inner)
        };
        let size = self.modifier.outer_size(measured);
        self.size = size;
        size
    }

    /// Places the node inside `bounds`.
    ///
    /// The framework calls this for every node of the tree; an
    /// application that drives the tree by hand may call it too.
    pub fn place(&mut self, bounds: Rect) {
        self.bounds = bounds;
        let inner = self.modifier.inner_bounds(bounds);
        {
            let Node { widget, children, .. } = self;
            let mut scope = PlaceScope::new(children, inner.origin);
            widget.place(&mut scope, inner);
        }
        for child in &mut self.children {
            let child_bounds = child.bounds;
            child.place(child_bounds);
        }
    }

    /// Draws the node into `list`.
    ///
    /// The framework calls this for every node of the tree; an
    /// application that drives the tree by hand may call it too.
    pub fn draw(&mut self, theme: &Theme, list: &mut DrawList) {
        let bounds = self.bounds;
        let inner = self.modifier.inner_bounds(bounds);
        if let Some(color) = self.modifier.background_color() {
            list.rect(bounds, CornerRadii::ZERO, Paint::fill(color));
        }
        let clipped = self.modifier.clips();
        if clipped {
            list.push_clip(inner);
        }
        {
            let Node {
                widget, children: _, ..
            } = self;
            let mut scope = DrawScope::new(&mut *list, theme, inner.origin);
            widget.draw(&mut scope, inner);
        }
        for child in &mut self.children {
            child.draw(theme, &mut *list);
        }
        if clipped {
            list.pop_clip();
        }
        if let Some((color, width, radii)) = self.modifier.border_paint() {
            list.rect(bounds, radii, Paint::stroke(color, width));
        }
    }
}

/// Offers an event to the tree rooted at `node`.
///
/// The deepest child that contains the pointer is offered the event
/// first; the widget of a node is offered it only when no child
/// consumed it. The click handler of the modifier is offered it last.
pub(crate) fn dispatch_event(
    node: &mut Node,
    theme: &Theme,
    path: &mut Vec<usize>,
    focus: &mut Option<Vec<usize>>,
    event: &Event,
) -> EventResult {
    if let Event::Pointer(pointer) = event {
        let mut index = node.children.len();
        while index > 0 {
            index -= 1;
            let inside = node.children[index]
                .bounds
                .contains(pointer.position);
            if inside && node.children[index].hit_testable() {
                path.push(index);
                let result = dispatch_event(&mut node.children[index], theme, path, focus, event);
                path.pop();
                if result.is_consumed() {
                    return result;
                }
                break;
            }
        }
    }
    let handler = node.modifier.click_handler();
    let Node { widget, children, .. } = node;
    let mut scope = EventScope::new(children, theme, path.clone(), focus);
    let result = widget.on_event(&mut scope, event);
    drop(scope);
    if result.is_consumed() {
        return result;
    }
    if let Some(handler) = handler {
        return handler(event);
    }
    EventResult::Ignored
}

impl Children for Vec<Node> {
    fn child_count(&self) -> usize {
        self.len()
    }

    fn measure_child(&mut self, index: usize, theme: &Theme, constraints: Constraints) -> Size {
        match self.get_mut(index) {
            Some(child) => child.measure(theme, constraints),
            None => constraints.enforce(Size::ZERO),
        }
    }

    fn place_child(&mut self, index: usize, position: Offset) {
        let Some(child) = self.get_mut(index) else {
            return;
        };
        child.bounds = Rect::new(position, child.size);
    }

    fn dispatch_child(
        &mut self,
        index: usize,
        theme: &Theme,
        path: &mut Vec<usize>,
        focus: &mut Option<Vec<usize>>,
        event: &Event,
    ) -> EventResult {
        let Some(child) = self.get_mut(index) else {
            return EventResult::Ignored;
        };
        path.push(index);
        let result = dispatch_event(child, theme, path, focus, event);
        path.pop();
        result
    }
}

/// A widget that gives another widget a name.
struct Identified {
    /// The name of the widget.
    key: Key,
    /// The widget itself.
    inner: Box<dyn Widget>,
}

impl Widget for Identified {
    fn key(&self) -> Option<Key> {
        Some(self.key)
    }

    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        self.inner.measure(scope, constraints)
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.inner.place(scope, bounds);
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        self.inner.draw(scope, bounds);
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        self.inner.on_event(scope, event)
    }

    fn hit_testable(&self) -> bool {
        self.inner.hit_testable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::DrawOp;

    /// A widget that fills the rectangle it is given.
    struct Filler;

    impl Widget for Filler {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, Color::BLUE);
        }
    }

    #[test]
    fn a_node_measures_to_the_size_of_its_widget() {
        let theme = Theme::light();
        let mut node = Node::new(Filler);
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(10.0, 10.0));
        assert_eq!(node.size(), size);
    }

    #[test]
    fn a_padding_modifier_grows_the_node() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().padding(EdgeInsets::all(5.0)));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(20.0, 20.0));
    }

    #[test]
    fn a_padding_modifier_shrinks_the_space_of_the_widget() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().padding(EdgeInsets::all(5.0)));
        let size = node.measure(&theme, Constraints::fixed(20.0, 20.0));
        assert_eq!(size, Size::new(20.0, 20.0));
    }

    #[test]
    fn a_size_modifier_forces_the_size() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().size(Some(30.0), Some(40.0)));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(30.0, 40.0));
    }

    #[test]
    fn a_background_modifier_paints_before_the_widget() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut node = Node::new(Filler).modifier(Modifier::none().background(Color::RED));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        node.draw(&theme, &mut list);
        assert_eq!(list.len(), 2);
        match &list.ops()[0] {
            DrawOp::Rect { paint, .. } => assert_eq!(paint.color, Color::RED),
            other => panic!("expected a rectangle, got {other:?}"),
        }
    }

    #[test]
    fn the_children_are_drawn_after_their_parent() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut parent = Node::new(Filler);
        parent.children.push(Node::new(Filler));
        parent.measure(&theme, Constraints::unbounded());
        parent.place(Rect::from_min_size(Offset::ZERO, 20.0, 20.0));
        parent.draw(&theme, &mut list);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn a_named_node_keeps_its_name() {
        let node = Node::with_key(Key::Named("root"), Filler);
        assert_eq!(node.effective_key(0), Key::Named("root"));
    }

    #[test]
    fn an_unnamed_node_is_identified_by_its_position() {
        let node = Node::new(Filler);
        assert_eq!(node.effective_key(3), Key::Index(3));
    }
}
