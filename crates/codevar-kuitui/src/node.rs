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
///
/// The layout operations — [`Modifier::size_in`], [`Modifier::fill_max_size`],
/// [`Modifier::aspect_ratio`], [`Modifier::size`] — bound the box of the
/// node itself, padding included, and are folded in the order they were
/// added. [`Modifier::weight`] and [`Modifier::z_index`] do not change
/// the size at all: the first is read by the parent when it distributes
/// free space, the second orders the node among its siblings.
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
    /// Bounds the size of the widget between a minimum and a maximum.
    SizeIn {
        /// The smallest width.
        min_width: f32,
        /// The largest width.
        max_width: f32,
        /// The smallest height.
        min_height: f32,
        /// The largest height.
        max_height: f32,
    },
    /// Takes a share of the space the parent grants on a bounded axis.
    FillMax {
        /// The share of the width, if the node fills it.
        width: Option<f32>,
        /// The share of the height, if the node fills it.
        height: Option<f32>,
    },
    /// Forces the width and the height to keep a ratio.
    AspectRatio(f32),
    /// Moves the widget when it is placed.
    Offset(Offset),
    /// The share of the free space the node claims in a row or column.
    Weight(f32),
    /// The paint order of the node among its siblings.
    ZIndex(i32),
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
    Clip(CornerRadii),
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

    /// Bounds the size of the widget between `min` and `max` on both axes.
    ///
    /// A maximum of [`f32::INFINITY`] leaves the axis unbounded and a
    /// minimum of zero asks for nothing. The bounds are clamped into the
    /// constraints the parent grants, so a minimum the parent cannot
    /// satisfy does not make the node overflow it.
    #[must_use]
    pub fn size_in(mut self, min_width: f32, min_height: f32, max_width: f32, max_height: f32) -> Self {
        self.ops.push(ModifierOp::SizeIn {
            min_width: min_width.min(max_width),
            max_width,
            min_height: min_height.min(max_height),
            max_height,
        });
        self
    }

    /// Bounds the width of the widget between `min` and `max`.
    ///
    /// The height is left as the parent constrains it.
    #[must_use]
    pub fn width_in(mut self, min: f32, max: f32) -> Self {
        self.ops.push(ModifierOp::SizeIn {
            min_width: min.min(max),
            max_width: max,
            min_height: 0.0,
            max_height: f32::INFINITY,
        });
        self
    }

    /// Bounds the height of the widget between `min` and `max`.
    ///
    /// The width is left as the parent constrains it.
    #[must_use]
    pub fn height_in(mut self, min: f32, max: f32) -> Self {
        self.ops.push(ModifierOp::SizeIn {
            min_width: 0.0,
            max_width: f32::INFINITY,
            min_height: min.min(max),
            max_height: max,
        });
        self
    }

    /// Fills the bounded axes of the space the parent grants.
    ///
    /// On an axis the parent bounds, the node becomes exactly `fraction`
    /// of the space; on an unbounded axis the measured size stands, so a
    /// column item does not grow to an infinite height. A fraction of one
    /// fills the space, and a fraction that is zero or not finite asks
    /// for nothing.
    #[must_use]
    pub fn fill_max_size(mut self, fraction: f32) -> Self {
        self.ops.push(ModifierOp::FillMax {
            width: fill_fraction(fraction),
            height: fill_fraction(fraction),
        });
        self
    }

    /// Fills the bounded width with `fraction` of the space granted.
    #[must_use]
    pub fn fill_max_width(mut self, fraction: f32) -> Self {
        self.ops.push(ModifierOp::FillMax {
            width: fill_fraction(fraction),
            height: None,
        });
        self
    }

    /// Fills the bounded height with `fraction` of the space granted.
    #[must_use]
    pub fn fill_max_height(mut self, fraction: f32) -> Self {
        self.ops.push(ModifierOp::FillMax {
            width: None,
            height: fill_fraction(fraction),
        });
        self
    }

    /// Forces the widget to keep `ratio` between its width and its height.
    ///
    /// The height is derived from the width when the width is bounded,
    /// and the width from the height otherwise, so the ratio holds on
    /// the axis the parent constrains without overflowing the other. A
    /// ratio that is zero, negative or not finite changes nothing, and
    /// so does a ratio applied to a box with no bounded axis to derive
    /// from.
    #[must_use]
    pub fn aspect_ratio(mut self, ratio: f32) -> Self {
        self.ops.push(ModifierOp::AspectRatio(ratio));
        self
    }

    /// Moves the widget by `x` and `y` when it is placed.
    ///
    /// The offset is applied at place time, after the parent has laid the
    /// node out: the measured size and the position of the siblings do
    /// not change, so a node can be nudged — or moved out of a card it
    /// decorates — without a relayout. Several offsets add up.
    #[must_use]
    pub fn offset(mut self, x: f32, y: f32) -> Self {
        self.ops
            .push(ModifierOp::Offset(Offset::new(x, y)));
        self
    }

    /// Claims `weight` shares of the free space of a row or a column.
    ///
    /// This is the chain form of [`Widget::flex`]: a row or a column
    /// distributes its free space in proportion to the weights it reads
    /// from its children, and a weight on the chain takes precedence over
    /// the one the widget reports. A weight of zero claims nothing.
    #[must_use]
    pub fn weight(mut self, weight: f32) -> Self {
        self.ops.push(ModifierOp::Weight(weight));
        self
    }

    /// Paints the node above the siblings with a lower index.
    ///
    /// The index orders the children of the same parent both when the
    /// parent paints and when it hit tests: a child with a higher index
    /// paints last and is offered the pointer first. Children with the
    /// same index keep their position in the tree.
    #[must_use]
    pub fn z_index(mut self, index: i32) -> Self {
        self.ops.push(ModifierOp::ZIndex(index));
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
    ///
    /// The clip covers everything the node paints over its own bounds —
    /// the custom drawings of [`Modifier::concrete`], the widget and its
    /// children — and not the container or the border the chain itself
    /// paints, which already carry their own corners.
    #[must_use]
    pub fn clip(mut self) -> Self {
        self.ops.push(ModifierOp::Clip(CornerRadii::ZERO));
        self
    }

    /// Clips the widget to its own rectangle, rounded with `radii`.
    ///
    /// A canvas that paints the whole node box cannot otherwise respect
    /// the shape of the node it decorates: its square corners would cover
    /// the rounding the container was given. The radii travel with the
    /// clip, so a backend that rasterizes a signed distance cuts the
    /// shapes and one that only knows scissors still clips to the box.
    #[must_use]
    pub fn clip_rounded(mut self, radii: CornerRadii) -> Self {
        self.ops.push(ModifierOp::Clip(radii));
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

    /// The translation the node is placed with.
    fn placement_offset(&self) -> Offset {
        self.ops
            .iter()
            .fold(Offset::ZERO, |offset, op| match op {
                ModifierOp::Offset(delta) => offset.add(*delta),
                _ => offset,
            })
    }

    /// The constraints the box of the node is measured with.
    ///
    /// The constraint operations of the chain are folded in the order
    /// they were added, and they bound the box of the node itself —
    /// padding included. Folding in order is what makes
    /// `width_in(.., 400.0).fill_max_width(1.0)` fill up to 400 pixels
    /// instead of filling the parent and being cut after.
    fn outer_constraints(&self, constraints: Constraints) -> Constraints {
        let mut outer = constraints;
        for op in &self.ops {
            outer = match *op {
                ModifierOp::SizeIn {
                    min_width,
                    max_width,
                    min_height,
                    max_height,
                } => intersect(outer, min_width, max_width, min_height, max_height),
                ModifierOp::FillMax { width, height } => fill(outer, width, height),
                ModifierOp::Size { width, height } => force(outer, width, height),
                ModifierOp::AspectRatio(ratio) => aspect(outer, ratio),
                _ => outer,
            };
        }
        outer
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

    /// The size the widget reports inside `outer`.
    ///
    /// The box is clamped into the constraints the chain folded, which
    /// is what turns a minimum into growth and a maximum into a cut, and
    /// what makes a fill or a ratio exact even for a widget that reports
    /// another size.
    fn outer_size(&self, measured: Size, outer: Constraints) -> Size {
        let padding = self.padding_insets();
        outer.enforce(Size::new(
            measured.width + padding.horizontal(),
            measured.height + padding.vertical(),
        ))
    }

    /// The weight the chain claims in a row or a column, if any.
    fn claimed_weight(&self) -> Option<f32> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::Weight(weight) => Some(*weight),
            _ => None,
        })
    }

    /// The paint order of the node among its siblings, if any.
    fn paint_order(&self) -> Option<i32> {
        self.ops.iter().rev().find_map(|op| match op {
            ModifierOp::ZIndex(index) => Some(*index),
            _ => None,
        })
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

    /// The corners of the clip the modifier asks for, or `None` when it
    /// does not clip at all.
    ///
    /// A chain can name both a rounded and a plain clip; the rounding it
    /// asked for anywhere wins, because a plain clip is only narrowing a
    /// rectangle that is already rounded.
    fn clip_radii(&self) -> Option<CornerRadii> {
        let mut radii = CornerRadii::ZERO;
        let mut clipping = false;
        for op in &self.ops {
            if let ModifierOp::Clip(shape) = op {
                clipping = true;
                if !shape.is_zero() {
                    radii = *shape;
                }
            }
        }
        clipping.then_some(radii)
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

/// The share of a bounded axis a fill asks for, if the fraction is usable.
fn fill_fraction(fraction: f32) -> Option<f32> {
    (fraction.is_finite() && fraction > 0.0).then(|| fraction.min(1.0))
}

/// The constraints with `min..max` intersected into them.
///
/// The bounds of the request are clamped into the bounds of `outer`
/// rather than replacing them, so a minimum the parent cannot satisfy
/// keeps the node inside the parent instead of overflowing it.
fn intersect(
    outer: Constraints,
    min_width: f32,
    max_width: f32,
    min_height: f32,
    max_height: f32,
) -> Constraints {
    Constraints::new(
        min_width.clamp(outer.min_width, outer.max_width),
        max_width.clamp(outer.min_width, outer.max_width),
        min_height.clamp(outer.min_height, outer.max_height),
        max_height.clamp(outer.min_height, outer.max_height),
    )
}

/// The constraints with the given axes forced to a share of their bound.
///
/// An unbounded axis has no space to share, so it keeps the constraints
/// it had: a column measures its children with an unbounded height and a
/// child that fills the width must not also fill an infinite height.
fn fill(outer: Constraints, width: Option<f32>, height: Option<f32>) -> Constraints {
    let width = match width {
        Some(fraction) if outer.has_bounded_width() => {
            let target = (outer.max_width * fraction).clamp(outer.min_width, outer.max_width);
            (target, target)
        }
        _ => (outer.min_width, outer.max_width),
    };
    let height = match height {
        Some(fraction) if outer.has_bounded_height() => {
            let target = (outer.max_height * fraction).clamp(outer.min_height, outer.max_height);
            (target, target)
        }
        _ => (outer.min_height, outer.max_height),
    };
    Constraints::new(width.0, width.1, height.0, height.1)
}

/// The constraints with the axes `width` and `height` force, if given.
fn force(outer: Constraints, width: Option<f32>, height: Option<f32>) -> Constraints {
    Constraints::new(
        width.unwrap_or(outer.min_width),
        width.unwrap_or(outer.max_width),
        height.unwrap_or(outer.min_height),
        height.unwrap_or(outer.max_height),
    )
}

/// The constraints that force the size `ratio` asks for.
///
/// The ratio is width over height. A bounded width derives the height,
/// and when both axes are bounded the width is used as long as the
/// height it derives still fits; otherwise the height derives the width.
/// With no bounded axis there is nothing to derive from and the
/// constraints are left alone.
fn aspect(outer: Constraints, ratio: f32) -> Constraints {
    if !ratio.is_finite() || ratio <= 0.0 {
        return outer;
    }
    if outer.has_bounded_width() {
        let height = outer.max_width / ratio;
        if !outer.has_bounded_height() || height <= outer.max_height {
            return Constraints::fixed(outer.max_width, height);
        }
    }
    if outer.has_bounded_height() {
        return Constraints::fixed(outer.max_height * ratio, outer.max_height);
    }
    outer
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
    /// The rectangle the node was laid out in, before its own offset.
    layout: Rect,
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
            layout: Rect::ZERO,
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
            layout: Rect::ZERO,
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

    /// The share of the free space the node claims in a row or column.
    ///
    /// The weight of the modifier chain takes precedence over the one
    /// the widget reports, so a decoration can make a fixed-size widget
    /// flexible — or the other way around — without touching it.
    #[must_use]
    pub fn flex(&self) -> f32 {
        self.modifier
            .claimed_weight()
            .unwrap_or_else(|| self.widget.flex())
    }

    /// The paint order of the node among its siblings.
    ///
    /// A node that does not name an order paints where the tree puts it.
    #[inline]
    #[must_use]
    pub fn z_index(&self) -> i32 {
        self.modifier.paint_order().unwrap_or(0)
    }

    /// Measures the node inside `constraints`.
    ///
    /// The framework calls this for every node of the tree; an
    /// application that drives the tree by hand may call it too.
    pub fn measure(&mut self, theme: &Theme, constraints: Constraints) -> Size {
        let outer = self.modifier.outer_constraints(constraints);
        let inner = self.modifier.inner_constraints(outer);
        let measured = {
            let Node { widget, children, .. } = self;
            let mut scope = MeasureScope::new(theme, children);
            widget.measure(&mut scope, inner)
        };
        self.measured = measured;
        let size = self.modifier.outer_size(measured, outer);
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
    ///
    /// `bounds` is the rectangle the parent laid the node out in. The
    /// offset of the modifier chain moves the node and everything it
    /// contains from there; the rectangle it was laid out in is kept, so
    /// placing the tree again does not accumulate the offset.
    pub fn place(&mut self, bounds: Rect) {
        self.layout = bounds;
        self.bounds = bounds.translate(self.modifier.placement_offset());
        let inner = self.content_rect(self.bounds);
        {
            let Node { widget, children, .. } = self;
            let mut scope = PlaceScope::new(children, inner.origin);
            widget.place(&mut scope, inner);
        }
        for child in &mut self.children {
            let layout = child.layout;
            child.place(layout);
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
    /// below the content it shares the node with. A clip the chain asks
    /// for covers the decorations, the content and the children — the
    /// painting the node does *over* its own box — and never the container
    /// or the border, which are rounded by their own radii.
    ///
    /// The children paint in the order of the tree, except that a child
    /// with a [`Modifier::z_index`] paints above the siblings with a
    /// lower one.
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
        let clip = self.modifier.clip_radii();
        if let Some(radii) = clip {
            // The clip opens over the decorations rather than only over
            // the content: a canvas that paints the whole node box would
            // otherwise cover the rounding of the container it decorates.
            list.push_rounded_clip(inner, radii);
        }
        {
            let mut scope = DrawScope::new(&mut *list, theme, inner.origin);
            self.modifier.draw_concrete(&mut scope, inner);
        }
        {
            let Node {
                widget, children: _, ..
            } = self;
            let mut scope = DrawScope::new(&mut *list, theme, inner.origin);
            widget.draw(&mut scope, inner);
        }
        self.draw_children(theme, &mut *list);
        if clip.is_some() {
            list.pop_clip();
        }
        if let Some((color, width, radii)) = self.modifier.border_paint() {
            list.rect(bounds, radii, Paint::stroke(color, width));
        }
    }

    /// Draws the children in the order they asked to be painted in.
    ///
    /// Children with the same index paint in the order of the tree. A
    /// child that asks for a higher index paints above its siblings
    /// without moving in the layout, which is how an overlay sits over
    /// the content of its parent. The order is computed only when some
    /// child asks for one, so a plain tree pays nothing for the sort.
    fn draw_children(&mut self, theme: &Theme, list: &mut DrawList) {
        if self
            .children
            .iter()
            .all(|child| child.z_index() == 0)
        {
            for child in &mut self.children {
                child.draw(theme, &mut *list);
            }
            return;
        }
        let mut order: Vec<usize> = (0..self.children.len()).collect();
        order.sort_by_key(|index| self.children[*index].z_index());
        for index in order {
            self.children[index].draw(theme, &mut *list);
        }
    }
}

/// Offers an event to the tree rooted at `node`.
///
/// The deepest child that contains the pointer is offered the event
/// first; the widget of a node is offered it only when no child
/// consumed it. The click handler of the modifier is offered it last.
///
/// A child that asked for a higher `z_index` is offered the pointer
/// before its siblings, the way it paints above them: the topmost node
/// under the pointer is the one that reacts to it.
pub(crate) fn dispatch_event(
    node: &mut Node,
    theme: &Theme,
    path: &mut Vec<usize>,
    focus: &mut Option<Vec<usize>>,
    event: &Event,
) -> EventResult {
    if let Event::Pointer(pointer) = event {
        let mut target: Option<(i32, usize)> = None;
        for (index, child) in node.children.iter().enumerate() {
            if !child.bounds.contains(pointer.position) || !child.hit_testable() {
                continue;
            }
            let order = (child.z_index(), index);
            if target.is_none_or(|best| order > best) {
                target = Some(order);
            }
        }
        if let Some((_, index)) = target {
            path.push(index);
            let result = dispatch_event(&mut node.children[index], theme, path, focus, event);
            path.pop();
            if result.is_consumed() {
                return result;
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
        self.get(index).map_or(0.0, Node::flex)
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
        child.layout = Rect::new(position, child.size);
        child.bounds = child.layout;
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
    fn the_clip_of_the_chain_covers_its_own_decoration() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let radii = CornerRadii::uniform(4.0);
        let mut node = Node::new(Layered).modifier(Modifier::none().clip_rounded(radii).concrete(
            |scope, bounds| {
                scope.fill_rect(
                    Rect::from_min_size(Offset::ZERO, bounds.size.width, 2.0),
                    Color::GREEN,
                );
            },
        ));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        node.draw(&theme, &mut list);

        // The canvas is the pass that would otherwise paint the square
        // corners of the box over the rounding the node was given, so it
        // is recorded inside the clip and the container is not.
        let clipped = list
            .commands()
            .map(|command| (command.clip, command.clip_radii))
            .collect::<Vec<_>>();
        let inner = Rect::from_min_size(Offset::ZERO, 10.0, 10.0);
        assert_eq!(
            clipped,
            [
                (None, CornerRadii::ZERO),
                (Some(inner), radii),
                (Some(inner), radii),
            ],
            "the decoration and the content are cut to the clip, the background of the widget is not"
        );
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

    /// A widget that reports the same size whatever it is given.
    struct Fixed;

    impl Widget for Fixed {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, _constraints: Constraints) -> Size {
            Size::new(10.0, 10.0)
        }
    }

    /// A widget that paints itself in `color`.
    struct Painted(Color);

    impl Widget for Painted {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, self.0);
        }
    }

    /// A widget that claims a share of the free space of its parent.
    struct Flexy(f32);

    impl Widget for Flexy {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn flex(&self) -> f32 {
            self.0
        }
    }

    /// A widget that stacks its children in the same rectangle.
    struct Stack;

    impl Widget for Stack {
        fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            for index in 0..scope.child_count() {
                let _ = scope.measure_child(index, constraints);
            }
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn place(&mut self, scope: &mut PlaceScope<'_>, _bounds: Rect) {
            for index in 0..scope.child_count() {
                scope.place_child(index, Offset::ZERO);
            }
        }
    }

    #[test]
    fn an_offset_modifier_moves_the_node_without_changing_its_size() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().offset(5.0, -3.0));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(10.0, 10.0));
        let layout = Rect::from_min_size(Offset::new(2.0, 4.0), 10.0, 10.0);
        node.place(layout);
        assert_eq!(
            node.bounds(),
            Rect::from_min_size(Offset::new(7.0, 1.0), 10.0, 10.0)
        );
        node.place(layout);
        assert_eq!(
            node.bounds().origin,
            Offset::new(7.0, 1.0),
            "placing the same layout again does not accumulate the offset"
        );
    }

    #[test]
    fn an_offset_modifier_carries_the_children_along() {
        let theme = Theme::light();
        let mut node = Node::new(Stack).modifier(Modifier::none().offset(5.0, 5.0));
        node.children.push(Node::new(Filler));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        assert_eq!(
            node.children()[0].bounds(),
            Rect::from_min_size(Offset::new(5.0, 5.0), 10.0, 10.0)
        );
    }

    #[test]
    fn offsets_add_up() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(
            Modifier::none()
                .offset(3.0, 4.0)
                .offset(-1.0, 2.0),
        );
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        assert_eq!(node.bounds().origin, Offset::new(2.0, 6.0));
    }

    #[test]
    fn a_size_in_modifier_cuts_and_grows_the_widget() {
        let theme = Theme::light();
        let mut cut = Node::new(Fixed).modifier(Modifier::none().size_in(0.0, 0.0, 8.0, 6.0));
        assert_eq!(
            cut.measure(&theme, Constraints::unbounded()),
            Size::new(8.0, 6.0),
            "a widget larger than the maximum is cut to it"
        );
        let mut grown =
            Node::new(Fixed).modifier(Modifier::none().size_in(20.0, 15.0, f32::INFINITY, f32::INFINITY));
        assert_eq!(
            grown.measure(&theme, Constraints::unbounded()),
            Size::new(20.0, 15.0),
            "a widget smaller than the minimum grows to it"
        );
    }

    #[test]
    fn a_size_in_modifier_does_not_overflow_the_parent() {
        let theme = Theme::light();
        let mut node =
            Node::new(Fixed).modifier(Modifier::none().size_in(100.0, 100.0, f32::INFINITY, f32::INFINITY));
        assert_eq!(
            node.measure(&theme, Constraints::fixed(40.0, 40.0)),
            Size::new(40.0, 40.0),
            "a minimum the parent cannot grant keeps the node inside it"
        );
    }

    #[test]
    fn a_width_in_modifier_leaves_the_height_alone() {
        let theme = Theme::light();
        let mut node = Node::new(Fixed).modifier(Modifier::none().width_in(0.0, 5.0));
        assert_eq!(
            node.measure(&theme, Constraints::unbounded()),
            Size::new(5.0, 10.0)
        );
        let mut node = Node::new(Fixed).modifier(Modifier::none().height_in(0.0, 4.0));
        assert_eq!(
            node.measure(&theme, Constraints::unbounded()),
            Size::new(10.0, 4.0)
        );
    }

    #[test]
    fn a_fill_max_size_modifier_fills_only_the_bounded_axes() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().fill_max_size(1.0));
        assert_eq!(
            node.measure(&theme, Constraints::fixed(40.0, 30.0)),
            Size::new(40.0, 30.0)
        );
        assert_eq!(
            node.measure(&theme, Constraints::loose(40.0, f32::INFINITY)),
            Size::new(40.0, 10.0),
            "an unbounded axis keeps the size the widget measured"
        );
    }

    #[test]
    fn a_fill_max_size_modifier_honors_the_fraction() {
        let theme = Theme::light();
        let mut node = Node::new(Filler).modifier(Modifier::none().fill_max_size(0.5));
        assert_eq!(
            node.measure(&theme, Constraints::loose(40.0, 30.0)),
            Size::new(20.0, 15.0),
            "half of the space the parent grants"
        );
    }

    #[test]
    fn an_aspect_ratio_modifier_derives_the_missing_axis() {
        let theme = Theme::light();
        let mut node = Node::new(Fixed).modifier(Modifier::none().aspect_ratio(2.0));
        assert_eq!(
            node.measure(&theme, Constraints::loose(100.0, f32::INFINITY)),
            Size::new(100.0, 50.0),
            "a bounded width derives the height"
        );
        assert_eq!(
            node.measure(&theme, Constraints::fixed(80.0, 60.0)),
            Size::new(80.0, 40.0)
        );
        assert_eq!(
            node.measure(&theme, Constraints::loose(100.0, 40.0)),
            Size::new(80.0, 40.0),
            "the height derives the width when the ratio does not fit the other way"
        );
    }

    #[test]
    fn an_aspect_ratio_without_a_bounded_axis_changes_nothing() {
        let theme = Theme::light();
        let mut node = Node::new(Fixed).modifier(Modifier::none().aspect_ratio(2.0));
        assert_eq!(
            node.measure(&theme, Constraints::unbounded()),
            Size::new(10.0, 10.0)
        );
    }

    #[test]
    fn a_weight_modifier_takes_precedence_over_the_widget() {
        let children = vec![
            Node::new(Flexy(2.0)),
            Node::new(Filler).modifier(Modifier::none().weight(3.0)),
            Node::new(Filler),
        ];
        assert_eq!(children.child_flex(0), 2.0, "the widget still reports its own");
        assert_eq!(children.child_flex(1), 3.0, "the chain overrides the widget");
        assert_eq!(children.child_flex(2), 0.0, "a node that claims nothing is rigid");
    }

    #[test]
    fn a_z_index_modifier_paints_a_child_above_its_siblings() {
        let theme = Theme::light();
        let mut parent = Node::new(Filler);
        parent
            .children
            .push(Node::new(Painted(Color::RED)).modifier(Modifier::none().z_index(1)));
        parent
            .children
            .push(Node::new(Painted(Color::GREEN)));
        parent.measure(&theme, Constraints::unbounded());
        parent.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        let mut list = DrawList::new();
        parent.draw(&theme, &mut list);
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
        assert_eq!(painted, [Color::BLUE, Color::GREEN, Color::RED]);
    }

    #[test]
    fn a_z_index_modifier_offers_the_pointer_to_the_topmost_child() {
        let theme = Theme::light();
        let hits = Rc::new(Cell::new(0_u8));
        let top = Rc::clone(&hits);
        let bottom = Rc::clone(&hits);
        let mut parent = Node::new(Stack);
        parent.children.push(
            Node::new(Filler).modifier(Modifier::none().z_index(1).clickable(move |_| {
                top.set(top.get() + 1);
                EventResult::Consumed
            })),
        );
        parent
            .children
            .push(Node::new(Filler).modifier(Modifier::none().clickable(move |_| {
                bottom.set(bottom.get() + 10);
                EventResult::Consumed
            })));
        parent.measure(&theme, Constraints::unbounded());
        parent.place(Rect::from_min_size(Offset::ZERO, 10.0, 10.0));
        let mut focus = None;
        let mut path = Vec::new();
        dispatch_event(
            &mut parent,
            &theme,
            &mut path,
            &mut focus,
            &Event::Pointer(PointerEvent::new(PointerEventKind::Down, Offset::new(5.0, 5.0))),
        );
        assert_eq!(hits.get(), 1, "the child painted on top takes the press");
    }
}
