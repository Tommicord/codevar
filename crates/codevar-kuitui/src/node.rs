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
use crate::effect::{Effect, Gradient, InteractionSource, InteractionState};
use crate::event::{Event, EventResult, PointerEventKind};
use crate::geometry::{Alignment, Constraints, EdgeInsets, Offset, Rect, Size};
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

/// The handler a [`Modifier::selectable`] modifier calls when the
/// choice of the widget flips.
///
/// The `bool` the handler receives is the state *after* the flip, so a
/// handler that only reports what happened never has to read the
/// [`InteractionSource`] back.
pub type SelectHandler = Rc<dyn Fn(&Event, bool) -> EventResult>;

/// The closure a [`Modifier::concrete`] modifier draws with.
///
/// The closure gets the same scope a widget gets, in the coordinate
/// space of the node it decorates, which is what lets a canvas widget
/// paint a node from the outside.
pub type ConcreteDraw = Rc<dyn Fn(&mut DrawScope<'_>, Rect)>;

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
    Background {
        /// The color of the container.
        color: Color,
        /// The radii of the corners of the container.
        radii: CornerRadii,
    },
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
    /// Positions the widget inside the space its padding leaves.
    Align(Alignment),
    /// Paints the container of the node with a ramp between two colors.
    Gradient {
        /// The ramp the container is painted with.
        gradient: Gradient,
        /// The radii of the corners of the container.
        radii: CornerRadii,
    },
    /// Blurs the silhouette of the container.
    Blur {
        /// The standard deviation of the blur, in logical pixels.
        radius: f32,
        /// The radii of the corners of the container.
        radii: CornerRadii,
    },
    /// Evaluates the container fill on a coarse grid.
    Pixelate(f32),
    /// Rotates the container around its center.
    Rotate(f32),
    /// Paints the node with a closure instead of a fixed primitive.
    Concrete(ConcreteDraw),
    /// Handles the events that reach the widget.
    Clickable {
        /// The handler the events are offered to.
        handler: ClickHandler,
        /// The states the press is reported to, if the caller has one.
        source: Option<InteractionSource>,
    },
    /// Reports the pointer states of the widget to a source.
    Hoverable(InteractionSource),
    /// Flips the choice of the widget on a release inside it.
    Selectable {
        /// The states the choice is kept in.
        source: InteractionSource,
        /// The handler the flip is reported to.
        handler: SelectHandler,
    },
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
        self.ops.push(ModifierOp::Background {
            color,
            radii: CornerRadii::ZERO,
        });
        self
    }

    /// Paints `color` behind the widget, with a rounded container.
    ///
    /// The corners are asked of the backend that replays the layer: the
    /// container becomes an effect layer whose signed distance rounds it,
    /// and the recorded rectangle keeps the color for a backend that
    /// draws a plain fill.
    #[must_use]
    pub fn rounded_background(mut self, color: Color, radii: CornerRadii) -> Self {
        self.ops
            .push(ModifierOp::Background { color, radii });
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

    /// Positions the widget inside the space its padding leaves.
    ///
    /// A widget measured smaller than its parent is by default given
    /// the whole padded rectangle; `align` gives it the rectangle of
    /// its own size, placed where the [`Alignment`] asks for it, which
    /// is what Compose's `Modifier.align` does inside a `Box`.
    #[must_use]
    pub fn align(mut self, alignment: Alignment) -> Self {
        self.ops.push(ModifierOp::Align(alignment));
        self
    }

    /// Paints the container of the node with `gradient`.
    ///
    /// The ramp is rasterized by the backend from the geometry of the
    /// layer, the way every other effect is, and the rectangle is
    /// recorded with the leading color of the ramp so a backend that
    /// ignores effects still draws the container.
    #[must_use]
    pub fn gradient(mut self, gradient: Gradient, radii: CornerRadii) -> Self {
        self.ops
            .push(ModifierOp::Gradient { gradient, radii });
        self
    }

    /// Blurs the silhouette of the container by `sigma` logical pixels.
    ///
    /// The blur is the analytic falloff of the rounded box, so its cost
    /// does not grow with `sigma`: the backend evaluates one distance
    /// per fragment whatever the radius.
    #[must_use]
    pub fn blur(mut self, sigma: f32, radii: CornerRadii) -> Self {
        self.ops
            .push(ModifierOp::Blur { radius: sigma, radii });
        self
    }

    /// Evaluates the container fill on a grid of `pixel`-sized blocks.
    ///
    /// The grid quantizes where the fill is *sampled*, not what it is, so
    /// it shows on the ramp of a [`Modifier::gradient`] and not on a flat
    /// color: every block of a solid container reads the same color, and
    /// the mosaic is invisible.
    #[must_use]
    pub fn pixelate(mut self, pixel: f32) -> Self {
        self.ops.push(ModifierOp::Pixelate(pixel));
        self
    }

    /// Rotates the container around its center by `radians`.
    #[must_use]
    pub fn rotate(mut self, radians: f32) -> Self {
        self.ops.push(ModifierOp::Rotate(radians));
        self
    }

    /// Paints the node by running `draw` after its background.
    ///
    /// This is the escape hatch of the modifier chain: anything a
    /// [`DrawScope`] can record becomes the decoration of the node
    /// without a new operation being added to the framework, which is
    /// how a canvas widget paints a node it does not own.
    #[must_use]
    pub fn concrete(mut self, draw: impl Fn(&mut DrawScope<'_>, Rect) + 'static) -> Self {
        self.ops.push(ModifierOp::Concrete(Rc::new(draw)));
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
        self.ops.push(ModifierOp::Clickable {
            handler: Rc::new(handler),
            source: None,
        });
        self
    }

    /// Handles the events that reach the widget and reports the press to
    /// `source`.
    ///
    /// The source carries [`InteractionState::PRESSED`] while a button
    /// is held inside the node, which is what lets a widget pick the
    /// colors of its pressed state without owning the pointer logic.
    #[must_use]
    pub fn clickable_with(
        mut self,
        source: InteractionSource,
        handler: impl Fn(&Event) -> EventResult + 'static,
    ) -> Self {
        self.ops.push(ModifierOp::Clickable {
            handler: Rc::new(handler),
            source: Some(source),
        });
        self
    }

    /// Reports the pointer states of the widget to `source`.
    ///
    /// The source carries [`InteractionState::HOVERED`] while the
    /// pointer is inside the node and [`InteractionState::PRESSED`]
    /// while a button is held in it. Hovering never consumes an event:
    /// the pointer moves for the widgets below too.
    #[must_use]
    pub fn hoverable(mut self, source: InteractionSource) -> Self {
        self.ops.push(ModifierOp::Hoverable(source));
        self
    }

    /// Flips the choice of the widget when a button is released inside
    /// it, and reports the flip to `handler`.
    ///
    /// The choice is [`InteractionState::SELECTED`] in `source`, so the
    /// widget and the application that owns the source agree on what is
    /// selected without the widget owning a flag of its own.
    #[must_use]
    pub fn selectable(
        mut self,
        source: InteractionSource,
        handler: impl Fn(&Event, bool) -> EventResult + 'static,
    ) -> Self {
        self.ops.push(ModifierOp::Selectable {
            source,
            handler: Rc::new(handler),
        });
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

    /// Whether the modifier reacts to the events of the widget.
    fn interacts(&self) -> bool {
        self.ops.iter().any(|op| {
            matches!(
                op,
                ModifierOp::Clickable { .. } | ModifierOp::Hoverable(_) | ModifierOp::Selectable { .. }
            )
        })
    }

    /// The background of the widget, if any.
    fn background_color(&self) -> Option<Color> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Background { color, .. } => Some(*color),
            _ => None,
        })
    }

    /// The color the container of the node is recorded with.
    ///
    /// A gradient container is recorded with the leading color of its
    /// ramp: a backend that replays the layer evaluates the ramp itself,
    /// and a backend that does not still draws a filled container.
    fn container_color(&self) -> Option<Color> {
        let gradient = self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Gradient { gradient, .. } => Some(*gradient),
            _ => None,
        });
        match (gradient, self.background_color()) {
            (Some(gradient), _) => Some(gradient.from),
            (None, color) => color,
        }
    }

    /// The radii of the container of the node, from the op that asks
    /// for a rounded one.
    ///
    /// An op that asks for no rounding does not answer: a background
    /// added after a blur keeps the corners the blur asked for.
    fn container_radii(&self) -> CornerRadii {
        self.ops
            .iter()
            .rev()
            .find_map(|op| match op {
                ModifierOp::Blur { radii, .. }
                | ModifierOp::Gradient { radii, .. }
                | ModifierOp::Background { radii, .. } => (*radii != CornerRadii::ZERO).then_some(*radii),
                _ => None,
            })
            .unwrap_or(CornerRadii::ZERO)
    }

    /// The alignment the widget is placed with, if any.
    fn alignment(&self) -> Option<Alignment> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Align(alignment) => Some(*alignment),
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
            ModifierOp::Clickable { handler, .. } => Some(Rc::clone(handler)),
            _ => None,
        })
    }

    /// The layer effect the container of the node is rasterized with.
    ///
    /// The effect describes the *container*, not the content: it is
    /// evaluated from the geometry of the node rectangle, so a blurred
    /// node draws a blurred container under text that stays crisp.
    /// `None` means the container needs no shader and is drawn as a
    /// plain rectangle.
    fn effect(&self, bounds: Rect) -> Option<Effect> {
        let mut effect = Effect::new(bounds);
        for op in &self.ops {
            effect = match *op {
                ModifierOp::Blur { radius, radii } => effect.blur(radius, radii),
                ModifierOp::Pixelate(pixel) => effect.pixelate(pixel),
                ModifierOp::Rotate(radians) => effect.rotate(radians),
                ModifierOp::Gradient { gradient, radii } => effect.gradient(gradient, radii),
                ModifierOp::Background { radii, .. } if radii != CornerRadii::ZERO => effect.shape(radii),
                _ => effect,
            };
        }
        (!effect.is_identity()).then_some(effect)
    }

    /// Runs the custom-draw operations of the modifier, in the order
    /// they were added.
    fn draw_concrete(&self, scope: &mut DrawScope<'_>, bounds: Rect) {
        for op in &self.ops {
            if let ModifierOp::Concrete(draw) = op {
                draw(scope, bounds);
            }
        }
    }

    /// Reports the pointer to the interaction sources of the modifier.
    ///
    /// Reporting never consumes: the same move is a hover for this node
    /// and a hover for the node under it.
    fn track_interaction(&self, bounds: Rect, event: &Event) {
        for op in &self.ops {
            let source = match op {
                ModifierOp::Hoverable(source) => source,
                ModifierOp::Clickable {
                    source: Some(source), ..
                } => source,
                _ => continue,
            };
            report_pointer(source, bounds, event);
        }
    }

    /// Flips the choice of the widget when the event releases a button
    /// inside it.
    fn select(&self, bounds: Rect, event: &Event) -> Option<EventResult> {
        if !released_inside(bounds, event) {
            return None;
        }
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Selectable { source, handler } => {
                let selected = !source
                    .state()
                    .contains(InteractionState::SELECTED);
                if selected {
                    source.set(InteractionState::SELECTED);
                } else {
                    source.clear(InteractionState::SELECTED);
                }
                Some(handler(event, selected))
            }
            _ => None,
        })
    }
}

/// Applies one pointer event to the states of `source`.
///
/// The pointer is tested against `bounds` because the tree reports the
/// position in window coordinates, and a move that leaves a node is the
/// event that clears its hover — no enter or leave event is required,
/// which is what lets the same rule work on a backend that coalesces
/// motion and on one that does not.
fn report_pointer(source: &InteractionSource, bounds: Rect, event: &Event) {
    let Event::Pointer(pointer) = event else {
        return;
    };
    let inside = bounds.contains(pointer.position);
    match pointer.kind {
        PointerEventKind::Move | PointerEventKind::Enter | PointerEventKind::Leave => {
            if inside {
                source.set(InteractionState::HOVERED);
            } else {
                source.clear(InteractionState::HOVERED | InteractionState::PRESSED);
            }
        }
        PointerEventKind::Down => {
            if inside {
                source.set(InteractionState::HOVERED | InteractionState::PRESSED);
            }
        }
        PointerEventKind::Up => {
            source.clear(InteractionState::PRESSED);
            if !inside {
                source.clear(InteractionState::HOVERED);
            }
        }
        PointerEventKind::Cancel => {
            source.clear(InteractionState::HOVERED | InteractionState::PRESSED);
        }
    }
}

/// Whether `event` releases a button inside `bounds`.
fn released_inside(bounds: Rect, event: &Event) -> bool {
    matches!(event, Event::Pointer(pointer)
        if pointer.kind == PointerEventKind::Up && bounds.contains(pointer.position))
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
    /// The size the widget itself reported, before the modifier grew
    /// it with its padding.
    measured: Size,
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
            measured: Size::ZERO,
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
            measured: Size::ZERO,
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
        self.widget.hit_testable() || self.modifier.interacts()
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
        self.measured = measured;
        let size = self.modifier.outer_size(measured);
        self.size = size;
        size
    }

    /// The rectangle the widget is given inside `bounds`.
    ///
    /// The padding of the modifier takes the space it asks for; when the
    /// modifier also aligns the widget, the widget is given the rectangle
    /// it measured instead of the whole padded space, placed where the
    /// alignment asks for it.
    fn content_rect(&self, bounds: Rect) -> Rect {
        let inner = self.modifier.inner_bounds(bounds);
        let Some(alignment) = self.modifier.alignment() else {
            return inner;
        };
        let size = self.measured.min(inner.size);
        Rect::new(
            Offset::new(
                alignment.align_x(inner.origin, inner.size.width, size.width),
                alignment.align_y(inner.origin, inner.size.height, size.height),
            ),
            size,
        )
    }

    /// Places the node inside `bounds`.
    ///
    /// The framework calls this for every node of the tree; an
    /// application that drives the tree by hand may call it too.
    pub fn place(&mut self, bounds: Rect) {
        self.bounds = bounds;
        let inner = self.content_rect(bounds);
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
    ///
    /// The paint order is the container of the modifier chain, the
    /// background of the widget, the decorations of the chain, the
    /// content of the widget, the children and last the border of the
    /// chain, so a decoration sits above the container it decorates and
    /// below the content it shares the node with.
    pub fn draw(&mut self, theme: &Theme, list: &mut DrawList) {
        let bounds = self.bounds;
        let inner = self.content_rect(bounds);
        let container = self.modifier.container_color();
        let radii = self.modifier.container_radii();
        let effect = self.modifier.effect(bounds);
        if let (Some(effect), Some(color)) = (effect, container) {
            // The container is the layer: a backend with the effect
            // program rasterizes the rounded box from its geometry, and
            // the recorded color is what a backend without one draws.
            list.push_effect(effect);
            list.rect(bounds, radii, Paint::fill(color));
            list.pop_effect();
        } else if let Some(color) = container {
            list.rect(bounds, radii, Paint::fill(color));
        }
        {
            // The widget paints its own container before the decorations
            // of the chain run, so an opaque container no longer hides
            // the canvas a `concrete` modifier draws over it.
            let Node {
                widget, children: _, ..
            } = self;
            let mut scope = DrawScope::new(&mut *list, theme, inner.origin);
            widget.draw_background(&mut scope, inner);
        }
        {
            let mut scope = DrawScope::new(&mut *list, theme, inner.origin);
            self.modifier.draw_concrete(&mut scope, inner);
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
    node.modifier
        .track_interaction(node.bounds, event);
    let Node { widget, children, .. } = node;
    let mut scope = EventScope::new(children, theme, path.clone(), focus);
    let result = widget.on_event(&mut scope, event);
    drop(scope);
    if result.is_consumed() {
        return result;
    }
    if let Some(result) = node.modifier.select(node.bounds, event) {
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

    fn child_size(&self, index: usize) -> Size {
        self.get(index)
            .map_or(Size::ZERO, |child| child.size)
    }

    fn child_flex(&self, index: usize) -> f32 {
        self.get(index)
            .map_or(0.0, |child| child.widget.flex())
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
    use crate::effect::EffectKinds;
    use crate::event::PointerEvent;
    use crate::paint::RectOp;
    use core::cell::Cell;

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

    /// A widget that fills its box in the background pass, the way the
    /// chrome of a button does, and paints its content in the content
    /// pass.
    struct Layered;

    impl Widget for Layered {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn draw_background(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, Color::RED);
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(
                Rect::from_min_size(Offset::ZERO, 2.0, bounds.size.height),
                Color::BLUE,
            );
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
        let op = list
            .ops()
            .next()
            .expect("a rectangle was recorded");
        let any: &dyn core::any::Any = op;
        let rect = any
            .downcast_ref::<RectOp>()
            .expect("expected a rectangle");
        assert_eq!(rect.paint.color, Color::RED);
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

    #[test]
    fn an_aligned_widget_keeps_the_size_it_measured() {
        let theme = Theme::light();
        let mut node = Node::new(Local).modifier(Modifier::none().align(Alignment::CENTER));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 40.0, 40.0));
        let mut list = DrawList::new();
        node.draw(&theme, &mut list);
        let op = list
            .ops()
            .next()
            .expect("the widget filled its rectangle");
        let any: &dyn core::any::Any = op;
        let rect = any
            .downcast_ref::<RectOp>()
            .expect("expected a rectangle");
        assert_eq!(
            rect.rect,
            Rect::from_min_size(Offset::new(15.0, 15.0), 10.0, 10.0)
        );
    }

    /// A widget that fills the rectangle it is given, recording it in
    /// its own coordinate space the way [`DrawScope`] asks for.
    struct Local;

    impl Widget for Local {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(
                Rect::from_min_size(Offset::ZERO, bounds.size.width, bounds.size.height),
                Color::BLUE,
            );
        }
    }

    #[test]
    fn a_gradient_container_is_recorded_with_the_leading_color() {
        let theme = Theme::light();
        let gradient = Gradient::horizontal(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            Color::RED,
            Color::BLUE,
        );
        let mut node =
            Node::new(Filler).modifier(Modifier::none().gradient(gradient, CornerRadii::uniform(4.0)));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        let mut list = DrawList::new();
        node.draw(&theme, &mut list);
        let command = list
            .commands()
            .next()
            .expect("the container was recorded");
        let effect = command
            .effect
            .expect("the command carries the layer");
        assert!(effect.kinds.contains(EffectKinds::GRADIENT));
        assert_eq!(effect.radii, CornerRadii::uniform(4.0));
        let any: &dyn core::any::Any = &*command.op;
        let rect = any
            .downcast_ref::<RectOp>()
            .expect("expected a rectangle");
        assert_eq!(rect.paint.color, Color::RED);
    }

    #[test]
    fn a_rounded_background_opens_a_layer_for_its_corners() {
        let theme = Theme::light();
        let radii = CornerRadii::uniform(4.0);
        let bounds = Rect::from_min_size(Offset::ZERO, 10.0, 10.0);
        let mut node = Node::new(Filler).modifier(Modifier::none().rounded_background(Color::RED, radii));
        node.measure(&theme, Constraints::unbounded());
        node.place(bounds);
        let mut list = DrawList::new();
        node.draw(&theme, &mut list);
        let command = list
            .commands()
            .next()
            .expect("the container was recorded");
        let effect = command
            .effect
            .expect("rounded corners are a request to the backend");
        assert!(effect.kinds.is_empty(), "the layer only rounds");
        assert_eq!(effect.radii, radii);
        assert_eq!(effect.bounds, bounds);
        assert_eq!(effect.coverage(), bounds);
    }

    #[test]
    fn a_square_background_needs_no_layer() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().background(Color::RED));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        let mut list = DrawList::new();
        node.draw(&theme, &mut list);
        let command = list
            .commands()
            .next()
            .expect("the container was recorded");
        assert_eq!(command.effect, None, "a square fill is already exact");
    }

    #[test]
    fn a_blurred_container_carries_its_radius_and_grows_the_layer() {
        let bounds = Rect::from_min_size(Offset::ZERO, 30.0, 20.0);
        let effect = Effect::new(bounds).blur(6.0, CornerRadii::uniform(2.0));
        assert!(effect.kinds.contains(EffectKinds::BLUR));
        assert_eq!(effect.coverage().size, Size::new(66.0, 56.0));
    }

    #[test]
    fn a_concrete_modifier_draws_before_the_widget() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut node = Node::new(Filler).modifier(Modifier::none().concrete(|scope, bounds| {
            scope.fill_rect(
                Rect::from_min_size(Offset::ZERO, bounds.size.width, 2.0),
                Color::GREEN,
            );
        }));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        node.draw(&theme, &mut list);
        assert_eq!(list.len(), 2);
        let first = list
            .commands()
            .next()
            .expect("the modifier drew first");
        let any: &dyn core::any::Any = &*first.op;
        let rect = any
            .downcast_ref::<RectOp>()
            .expect("expected a rectangle");
        assert_eq!(rect.paint.color, Color::GREEN);
    }

    #[test]
    fn a_widget_background_paints_under_the_decorations_of_the_chain() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut node = Node::new(Layered).modifier(Modifier::none().concrete(|scope, bounds| {
            scope.fill_rect(
                Rect::from_min_size(Offset::ZERO, bounds.size.width, 2.0),
                Color::GREEN,
            );
        }));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        node.draw(&theme, &mut list);

        // The decoration has to sit between the two passes of the widget,
        // or an opaque container would record it under itself and a
        // backend would draw a canvas no one can see.
        let painted = list
            .commands()
            .map(|command| {
                let any: &dyn core::any::Any = &*command.op;
                any.downcast_ref::<RectOp>()
                    .expect("every pass records a rectangle")
                    .paint
                    .color
            })
            .collect::<Vec<_>>();
        assert_eq!(painted, [Color::RED, Color::GREEN, Color::BLUE]);
    }

    #[test]
    fn a_hoverable_modifier_reports_the_pointer_states() {
        let theme = Theme::light();
        let source = InteractionSource::new();
        let mut node = Node::new(Filler).modifier(Modifier::none().hoverable(source.clone()));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        let mut focus = None;
        let mut path = Vec::new();
        dispatch_event(
            &mut node,
            &theme,
            &mut path,
            &mut focus,
            &Event::Pointer(PointerEvent::new(PointerEventKind::Move, Offset::new(5.0, 5.0))),
        );
        assert!(source.state().contains(InteractionState::HOVERED));
        dispatch_event(
            &mut node,
            &theme,
            &mut path,
            &mut focus,
            &Event::Pointer(PointerEvent::new(PointerEventKind::Move, Offset::new(50.0, 50.0))),
        );
        assert!(!source.state().contains(InteractionState::HOVERED));
    }

    #[test]
    fn a_selectable_modifier_flips_the_choice_on_release() {
        let theme = Theme::light();
        let source = InteractionSource::new();
        let seen = Rc::new(Cell::new(None));
        let reported = Rc::clone(&seen);
        let mut node = Node::new(Filler).modifier(Modifier::none().selectable(
            source.clone(),
            move |_event, selected| {
                reported.replace(Some(selected));
                EventResult::Consumed
            },
        ));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        let mut focus = None;
        let mut path = Vec::new();
        let release = Event::Pointer(PointerEvent::new(PointerEventKind::Up, Offset::new(5.0, 5.0)));
        let result = dispatch_event(&mut node, &theme, &mut path, &mut focus, &release);
        assert!(result.is_consumed());
        assert_eq!(seen.get(), Some(true));
        assert!(
            source
                .state()
                .contains(InteractionState::SELECTED)
        );
        dispatch_event(&mut node, &theme, &mut path, &mut focus, &release);
        assert_eq!(seen.get(), Some(false));
        assert!(
            !source
                .state()
                .contains(InteractionState::SELECTED)
        );
    }

    #[test]
    fn a_press_is_reported_to_the_source_of_a_clickable_modifier() {
        let theme = Theme::light();
        let source = InteractionSource::new();
        let mut node = Node::new(Filler)
            .modifier(Modifier::none().clickable_with(source.clone(), |_| EventResult::Ignored));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        assert!(node.hit_testable(), "the node takes part in the hit test");
        let mut focus = None;
        let mut path = Vec::new();
        dispatch_event(
            &mut node,
            &theme,
            &mut path,
            &mut focus,
            &Event::Pointer(PointerEvent::new(PointerEventKind::Down, Offset::new(5.0, 5.0))),
        );
        assert!(source.state().contains(InteractionState::PRESSED));
    }
}
