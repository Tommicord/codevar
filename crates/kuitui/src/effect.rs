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

//! The layer effects and the interaction states a modifier asks for.
//!
//! An [`Effect`] is a description of *how* a rectangle is rasterized,
//! not of what is drawn inside it: the geometry stays the same, and the
//! backend evaluates the shape in a shader. Every effect a modifier
//! builds — [`EffectKinds::BLUR`], [`EffectKinds::PIXELATE`],
//! [`EffectKinds::ROTATE`], [`EffectKinds::GRADIENT`],
//! [`EffectKinds::STROKE`] — is defined by
//! the closed form of a rounded box, so the fragment shader costs a
//! distance evaluation instead of a texture pass:
//!
//! * blur is the Gaussian-integral silhouette of the rounded box, the
//!   analytic form of blurring a filled rectangle;
//! * pixelate quantizes the evaluation grid of the fill, which turns a
//!   smooth gradient into a mosaic of its own colors;
//! * rotate maps the sample point back into the frame of the shape, so
//!   the rectangle itself never moves in memory;
//! * gradient evaluates the ramp between the two endpoints of an axis;
//! * stroke keeps the band of the rounded box that lies within one stroke
//!   width of its edge, which is how an outlined rectangle follows its own
//!   corners instead of being drawn as four straight edges.
//!
//! An effect travels with the operations recorded while it was open,
//! the same way a clip does, which is what keeps a backend from having
//! to model a stack of its own.

use core::ops::{BitOr, BitOrAssign};

use crate::color::Color;
use crate::geometry::{Offset, Rect};
use crate::paint::CornerRadii;

/// How many times a Gaussian extends into the tail of the curve.
///
/// The fragment shader evaluates the analytic falloff, which is smooth
/// well past two sigmas; three is where the remaining energy stops
/// being visible, and it is the extent the geometry has to cover.
const BLUR_SIGMA_SPAN: f32 = 3.0;

/// The effects a [`crate::node::Modifier`] asks a backend to apply.
///
/// The set is a bit flag rather than a struct of booleans so that a
/// layer carries one word and a backend tests it without a branch per
/// effect. The bits combine: a rounded box can be rotated *and*
/// blurred *and* painted with a ramp at the same time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EffectKinds(u8);

impl EffectKinds {
    /// No effect at all.
    pub const NONE: EffectKinds = EffectKinds(0);
    /// The silhouette of the rectangle is blurred.
    pub const BLUR: EffectKinds = EffectKinds(1 << 0);
    /// The fill of the rectangle is evaluated on a coarse grid.
    pub const PIXELATE: EffectKinds = EffectKinds(1 << 1);
    /// The rectangle is rotated around its center.
    pub const ROTATE: EffectKinds = EffectKinds(1 << 2);
    /// The rectangle is painted with a ramp between two colors.
    pub const GRADIENT: EffectKinds = EffectKinds(1 << 3);
    /// The rectangle is outlined: only the band of the rounded box within
    /// [`Effect::stroke`] of its edge is covered.
    pub const STROKE: EffectKinds = EffectKinds(1 << 4);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: EffectKinds) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no effect at all is selected.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set with the bits of `other` added.
    #[inline]
    #[must_use]
    pub const fn union(self, other: EffectKinds) -> Self {
        EffectKinds(self.0 | other.0)
    }
}

impl BitOr for EffectKinds {
    type Output = EffectKinds;

    #[inline]
    fn bitor(self, rhs: EffectKinds) -> EffectKinds {
        EffectKinds(self.0 | rhs.0)
    }
}

impl BitOrAssign for EffectKinds {
    #[inline]
    fn bitor_assign(&mut self, rhs: EffectKinds) {
        self.0 |= rhs.0;
    }
}

/// The ramp between two colors along a straight axis.
///
/// The axis is expressed in the coordinate space of the layer it
/// paints, so the same ramp keeps its direction when the layer moves.
/// This is the core counterpart of the canvas gradient of
/// `codevar_kuitui_widget`, which carries the same resolution-
/// independent geometry for a shader to consume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient {
    /// The point of the layer that carries [`Self::from`].
    pub start: Offset,
    /// The point of the layer that carries [`Self::to`].
    pub end: Offset,
    /// The color at [`Self::start`].
    pub from: Color,
    /// The color at [`Self::end`].
    pub to: Color,
}

impl Gradient {
    /// Creates a ramp between two colors along an axis.
    #[inline]
    #[must_use]
    pub const fn new(start: Offset, end: Offset, from: Color, to: Color) -> Self {
        Self { start, end, from, to }
    }

    /// The color at `factor` along the axis, clamped to `0.0 ..= 1.0`.
    #[inline]
    #[must_use]
    pub fn color_at(&self, factor: f32) -> Color {
        self.from.lerp(self.to, factor.clamp(0.0, 1.0))
    }

    /// The normalized position of `point` projected onto the axis.
    ///
    /// A degenerate axis — one whose endpoints coincide — reports `0.0`,
    /// which makes the whole layer [`Self::from`].
    #[must_use]
    pub fn factor_at(&self, point: Offset) -> f32 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let length_squared = dx * dx + dy * dy;
        if length_squared <= 0.0 {
            return 0.0;
        }
        let factor = ((point.x - self.start.x) * dx + (point.y - self.start.y) * dy) / length_squared;
        factor.clamp(0.0, 1.0)
    }

    /// The color of the ramp at the point of the axis nearest `point`.
    #[inline]
    #[must_use]
    pub fn color_at_point(&self, point: Offset) -> Color {
        self.color_at(self.factor_at(point))
    }

    /// The top-to-bottom ramp of a layer.
    #[must_use]
    pub fn vertical(rect: Rect, from: Color, to: Color) -> Self {
        Self::new(
            Offset::new(rect.left(), rect.top()),
            Offset::new(rect.left(), rect.bottom()),
            from,
            to,
        )
    }

    /// The leading-to-trailing ramp of a layer.
    #[must_use]
    pub fn horizontal(rect: Rect, from: Color, to: Color) -> Self {
        Self::new(
            Offset::new(rect.left(), rect.top()),
            Offset::new(rect.right(), rect.top()),
            from,
            to,
        )
    }
}

/// How a rectangle is rasterized.
///
/// An effect is `Copy` and small on purpose: it is resolved once per
/// node and then travels by value with every operation recorded inside
/// the layer, so a backend never dereferences a stack while replaying.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Effect {
    /// The rectangle the effect is evaluated over, in the coordinate
    /// space of the window.
    pub bounds: Rect,
    /// The radii of the corners of the rectangle.
    pub radii: CornerRadii,
    /// Which of the parameters below a backend has to honor.
    pub kinds: EffectKinds,
    /// The standard deviation of the blur, in logical pixels.
    pub blur: f32,
    /// The edge of a pixelate block, in logical pixels.
    pub pixel: f32,
    /// The width of the outline, in logical pixels.
    ///
    /// The band is measured inward from the edge of the rounded box, so a
    /// stroked layer covers the area whose signed distance is between
    /// `-stroke` and `0`. It is only read with
    /// [`EffectKinds::STROKE`] set.
    pub stroke: f32,
    /// The rotation of the rectangle around its center, in radians.
    pub rotation: f32,
    /// The ramp the rectangle is painted with, if any.
    pub gradient: Option<Gradient>,
}

impl Effect {
    /// An effect that rasterizes `bounds` exactly as a flat fill would
    /// be rasterized.
    #[inline]
    #[must_use]
    pub const fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            radii: CornerRadii::ZERO,
            kinds: EffectKinds::NONE,
            blur: 0.0,
            pixel: 0.0,
            stroke: 0.0,
            rotation: 0.0,
            gradient: None,
        }
    }

    /// Whether the effect asks the backend for anything at all.
    ///
    /// Rounded corners count: an identity effect is one a plain
    /// rectangle would be drawn exactly like.
    #[inline]
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        let radii = self.radii;
        self.kinds.is_empty()
            && radii.top_left == 0.0
            && radii.top_right == 0.0
            && radii.bottom_right == 0.0
            && radii.bottom_left == 0.0
    }

    /// The rectangle with rounded corners and no other effect.
    ///
    /// A backend that replays the layer rounds the corners from the
    /// signed distance; a backend that does not draws the recorded
    /// rectangle square, which is what it did before corners existed.
    #[inline]
    #[must_use]
    pub const fn shape(mut self, radii: CornerRadii) -> Self {
        self.radii = radii;
        self
    }

    /// The rectangle blurred by `sigma` logical pixels.
    #[inline]
    #[must_use]
    pub const fn blur(mut self, sigma: f32, radii: CornerRadii) -> Self {
        self.kinds = self.kinds.union(EffectKinds::BLUR);
        self.blur = sigma;
        self.radii = radii;
        self
    }

    /// The rectangle whose fill is evaluated on a grid of `pixel`
    /// logical pixels.
    #[inline]
    #[must_use]
    pub const fn pixelate(mut self, pixel: f32) -> Self {
        self.kinds = self.kinds.union(EffectKinds::PIXELATE);
        self.pixel = pixel;
        self
    }

    /// The rectangle outlined by the band of `width` logical pixels that
    /// lies inside its edge.
    ///
    /// The outline follows whatever corners the layer already carries, so a
    /// shape asks for its rounding once: a layer that was given radii and
    /// then outlined draws the rounded ring, and one that was only outlined
    /// draws a rectangular one.
    #[inline]
    #[must_use]
    pub const fn stroke(mut self, width: f32) -> Self {
        self.kinds = self.kinds.union(EffectKinds::STROKE);
        self.stroke = width;
        self
    }

    /// The rectangle rotated by `radians` around its center.
    #[inline]
    #[must_use]
    pub const fn rotate(mut self, radians: f32) -> Self {
        self.kinds = self.kinds.union(EffectKinds::ROTATE);
        self.rotation = radians;
        self
    }

    /// The rectangle painted with `gradient`.
    #[inline]
    #[must_use]
    pub const fn gradient(mut self, gradient: Gradient, radii: CornerRadii) -> Self {
        self.kinds = self.kinds.union(EffectKinds::GRADIENT);
        self.gradient = Some(gradient);
        self.radii = radii;
        self
    }

    /// The rectangle a backend has to cover to draw this effect.
    ///
    /// Two geometric facts grow the rectangle: a Gaussian bleeds
    /// [`BLUR_SIGMA_SPAN`] sigmas past its edge, and a rotated rectangle
    /// needs the axis-aligned box that contains it. The rounded corners
    /// never add to the extent, because they only ever remove area.
    #[must_use]
    pub fn coverage(&self) -> Rect {
        let blur = if self.kinds.contains(EffectKinds::BLUR) {
            self.blur * BLUR_SIGMA_SPAN
        } else {
            0.0
        };
        if !self.kinds.contains(EffectKinds::ROTATE) {
            return self.bounds.inflate(blur);
        }
        let half_width = self.bounds.size.width * 0.5;
        let half_height = self.bounds.size.height * 0.5;
        let cos = libm::fabsf(libm::cosf(self.rotation));
        let sin = libm::fabsf(libm::sinf(self.rotation));
        let width = (half_width * cos + half_height * sin) * 2.0;
        let height = (half_width * sin + half_height * cos) * 2.0;
        let center = self.bounds.center();
        Rect::from_min_size(
            Offset::new(center.x - width * 0.5, center.y - height * 0.5),
            width,
            height,
        )
        .inflate(blur)
    }
}

/// The interaction states a widget can be in at the same time.
///
/// A widget is hovered, pressed, selected and focused *concurrently*,
/// so the states are a bit set rather than four booleans; the
/// combination is what a [`crate::node::Modifier`] reports to the
/// widget it decorates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct InteractionState(u8);

impl InteractionState {
    /// No interaction applies.
    pub const NONE: InteractionState = InteractionState(0);
    /// The pointer is inside the widget.
    pub const HOVERED: InteractionState = InteractionState(1 << 0);
    /// A button is held down inside the widget.
    pub const PRESSED: InteractionState = InteractionState(1 << 1);
    /// The widget was chosen from a set.
    pub const SELECTED: InteractionState = InteractionState(1 << 2);
    /// The widget receives keyboard events.
    pub const FOCUSED: InteractionState = InteractionState(1 << 3);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: InteractionState) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no state at all applies.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The state with `bits` turned on.
    #[inline]
    #[must_use]
    pub const fn with(self, bits: InteractionState) -> Self {
        InteractionState(self.0 | bits.0)
    }

    /// The state with `bits` turned off.
    #[inline]
    #[must_use]
    pub const fn without(self, bits: InteractionState) -> Self {
        InteractionState(self.0 & !bits.0)
    }
}

impl BitOr for InteractionState {
    type Output = InteractionState;

    #[inline]
    fn bitor(self, rhs: InteractionState) -> InteractionState {
        InteractionState(self.0 | rhs.0)
    }
}

impl BitOrAssign for InteractionState {
    #[inline]
    fn bitor_assign(&mut self, rhs: InteractionState) {
        self.0 |= rhs.0;
    }
}

/// The shared interaction states of a widget.
///
/// This is the Kuitui counterpart of Compose's `InteractionSource`:
/// a handle a modifier writes when the pointer moves over the widget it
/// decorates and that the widget — and the application that owns the
/// handle — reads to pick its colors. Cloning the source yields another
/// handle to the same states, which is what lets the application keep
/// observing them after the modifier is dropped.
#[derive(Debug, Clone, Default)]
pub struct InteractionSource {
    /// The states shared by the handles.
    state: alloc::rc::Rc<core::cell::Cell<InteractionState>>,
}

impl InteractionSource {
    /// A source with no interaction in progress.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A source already in `state`.
    #[must_use]
    pub fn with_state(state: InteractionState) -> Self {
        Self {
            state: alloc::rc::Rc::new(core::cell::Cell::new(state)),
        }
    }

    /// The states the widget is currently in.
    #[inline]
    #[must_use]
    pub fn state(&self) -> InteractionState {
        self.state.get()
    }

    /// Turns `bits` on and returns whether they changed.
    #[inline]
    pub fn set(&self, bits: InteractionState) -> bool {
        self.replace(self.state.get().with(bits))
    }

    /// Turns `bits` off and returns whether they changed.
    #[inline]
    pub fn clear(&self, bits: InteractionState) -> bool {
        self.replace(self.state.get().without(bits))
    }

    /// Installs `state` and returns whether it changed.
    fn replace(&self, state: InteractionState) -> bool {
        let previous = self.state.replace(state);
        previous != state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Size;

    #[test]
    fn effect_kinds_combine_and_test() {
        let kinds = EffectKinds::BLUR | EffectKinds::ROTATE;
        assert!(kinds.contains(EffectKinds::BLUR));
        assert!(kinds.contains(EffectKinds::ROTATE));
        assert!(!kinds.contains(EffectKinds::PIXELATE));
        assert_eq!(kinds.bits(), 1 | 4);
    }

    #[test]
    fn interaction_states_stack_and_release() {
        let source = InteractionSource::new();
        assert!(source.state().is_empty());
        assert!(source.set(InteractionState::HOVERED | InteractionState::PRESSED));
        assert!(!source.set(InteractionState::HOVERED), "already hovered");
        source.clear(InteractionState::PRESSED);
        assert_eq!(source.state(), InteractionState::HOVERED);
    }

    #[test]
    fn a_cloned_source_shares_the_states() {
        let source = InteractionSource::new();
        let other = source.clone();
        source.set(InteractionState::SELECTED);
        assert!(other.state().contains(InteractionState::SELECTED));
    }

    #[test]
    fn the_ramp_is_clamped_at_both_ends() {
        let gradient = Gradient::horizontal(
            Rect::from_min_size(Offset::ZERO, 100.0, 20.0),
            Color::BLACK,
            Color::WHITE,
        );
        assert_eq!(gradient.color_at(-1.0), Color::BLACK);
        assert_eq!(gradient.color_at(2.0), Color::WHITE);
        assert_eq!(gradient.factor_at(Offset::new(50.0, 0.0)), 0.5);
    }

    #[test]
    fn a_degenerate_ramp_paints_the_start_color() {
        let gradient = Gradient::new(Offset::ZERO, Offset::ZERO, Color::RED, Color::BLUE);
        assert_eq!(gradient.factor_at(Offset::new(3.0, 4.0)), 0.0);
        assert_eq!(gradient.color_at_point(Offset::new(3.0, 4.0)), Color::RED);
    }

    #[test]
    fn the_blur_coverage_reaches_three_sigmas() {
        let bounds = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
        let effect = Effect::new(bounds).blur(4.0, CornerRadii::uniform(6.0));
        let coverage = effect.coverage();
        assert_eq!(coverage.size, Size::new(64.0, 44.0));
        assert!(effect.kinds.contains(EffectKinds::BLUR));
    }

    #[test]
    fn a_quarter_turn_exchanges_the_coverage_extents() {
        let bounds = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
        let effect = Effect::new(bounds).rotate(core::f32::consts::FRAC_PI_2);
        let coverage = effect.coverage();
        assert!((coverage.size.width - 20.0).abs() < 1e-3);
        assert!((coverage.size.height - 40.0).abs() < 1e-3);
        assert_eq!(coverage.center(), bounds.center());
    }

    #[test]
    fn an_identity_effect_covers_its_own_rectangle() {
        let bounds = Rect::from_min_size(Offset::new(5.0, 6.0), 12.0, 30.0);
        let effect = Effect::new(bounds);
        assert!(effect.is_identity());
        assert_eq!(effect.coverage(), bounds);
    }

    #[test]
    fn rounded_corners_are_an_effect_of_their_own() {
        let bounds = Rect::from_min_size(Offset::ZERO, 20.0, 10.0);
        let effect = Effect::new(bounds).shape(CornerRadii::uniform(3.0));
        assert!(!effect.is_identity(), "a plain fill has no corners");
        assert!(effect.kinds.is_empty(), "a corner is not a kind");
        assert_eq!(effect.coverage(), bounds, "rounding only removes area");
        assert!(
            Effect::new(bounds)
                .shape(CornerRadii::ZERO)
                .is_identity()
        );
    }

    #[test]
    fn an_outline_is_a_layer_effect_of_its_own() {
        let bounds = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
        let effect = Effect::new(bounds).stroke(3.0);
        assert!(effect.kinds.contains(EffectKinds::STROKE));
        assert_eq!(effect.stroke, 3.0);
        assert!(!effect.is_identity(), "a ring is not a plain fill");
        assert_eq!(effect.coverage(), bounds, "an outline never leaves the box");
    }

    #[test]
    fn an_outline_keeps_the_rounding_of_its_layer() {
        let bounds = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
        let radii = CornerRadii::uniform(6.0);
        let effect = Effect::new(bounds).shape(radii).stroke(2.0);
        assert_eq!(effect.radii, radii);
        assert!(effect.kinds.contains(EffectKinds::STROKE));
        assert!(
            !effect.kinds.contains(EffectKinds::BLUR),
            "an outline does not bleed past its edge"
        );
    }
}
