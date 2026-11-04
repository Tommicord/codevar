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
//! CONDITIONS OF ANY KIND, either express or implied. See the
//! License for the specific language governing permissions and
//! limitations under the License.

//! The check box: a [`CheckState`] a gesture, a key or the application
//! writes.
//!
//! The box is the smallest widget that answers the pointer through a
//! [`GestureRegion`], which is what makes the region a component rather
//! than a detail of scrolling: a box reads the same press, slop and
//! release rules a list does, so a finger that lands on a box and then
//! travels still scrolls the list that holds it. The box claims a press
//! only when it *completes* one — a release that stayed inside the target
//! and never left through the slop.
//!
//! # The value outlives the render
//!
//! The value is a [`MutableState`] handle, so cloning the widget yields
//! another view of the same value and a recomposition in the middle of a
//! press cannot lose what the press wrote. A content closure therefore
//! creates it once — with
//! [`use_state`](codevar_kuitui_rea::RenderCtx::use_state) — and every
//! later render hands the same handle to the box.
//!
//! The handle is also how the application drives the box: writing the
//! state changes what the box paints without a press. An application that
//! wants to be told when the *user* changed the value passes a handler
//! with [`Checkbox::on_change`], which a write the application made itself
//! never calls — the same split Compose draws between `checked` and
//! `onCheckedChange`.
//!
//! # What the states mean to the box
//!
//! [`CheckState::Indeterminate`] answers a click only while
//! [`CheckboxFlags::TRI_STATE`] is set: a tri-state box *cycles* —
//! unchecked, checked, indeterminate — where an ordinary box toggles,
//! which is the difference between the platform's `TriStateCheckbox` and
//! its `Checkbox`. A write from the application keeps whatever state it
//! names under either flag.
//!
//! The [`InteractionSource`](crate::effect::InteractionSource) of the box
//! reports the *interaction* states — hovered, pressed, focused — and
//! deliberately not the selection: the value has a handle of its own, and
//! a second copy of it in the source would be a second thing to keep in
//! sync.

use alloc::rc::Rc;
use core::ops::{BitOr, BitOrAssign};

use crate::color::{Color, ColorScheme};
use crate::effect::{InteractionSource, InteractionState};
use crate::event::{Event, EventResult, Key, PointerEventKind};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::paint::CornerRadii;
use crate::state::MutableState;
use crate::widget::{DrawScope, EventScope, MeasureScope, PlaceScope, Widget};
use crate::widgets::{Gesture, GestureRegion, ViewportDispatch};

/// The handler a check box calls with the value the user chose.
pub type CheckChange = Rc<dyn Fn(CheckState)>;

/// The geometry and the alphas of the Material 3 check box.
///
/// The numbers are the ones the platform component is specified with, so a
/// box drawn here occupies the same pixels as the one it stands for: the
/// frame is 20 logical pixels on a side, its unchecked outline is 2 pixels
/// wide, and a checked box fills an 18-pixel square — the frame contracts
/// by the stroke it no longer needs.
pub mod defaults {
    use crate::geometry::Offset;

    /// The side of the frame of an unchecked box.
    pub const BOX: f32 = 20.0;

    /// The side of the container of a checked box.
    pub const CHECKED_BOX: f32 = 18.0;

    /// The width of the outline of an unchecked box, and of the glyph.
    pub const STROKE: f32 = 2.0;

    /// The radius of the corners of the box.
    pub const CORNER: f32 = 2.0;

    /// The side of the square a press is answered in.
    ///
    /// The touch target is the box with the padding a finger needs around
    /// it, and the state layer is painted over that whole square — which
    /// is why the layer is drawn as a circle of this diameter.
    pub const TOUCH_TARGET: f32 = 40.0;

    /// The alpha of the state layer while the pointer is over the box.
    pub const HOVER_ALPHA: f32 = 0.08;

    /// The alpha of the state layer while a button is held in the box.
    pub const PRESSED_ALPHA: f32 = 0.10;

    /// The alpha of the state layer while the box has the keyboard focus.
    pub const FOCUSED_ALPHA: f32 = 0.12;

    /// The alpha of the content and the outline of a box that answers
    /// nothing.
    pub const DISABLED_CONTENT_ALPHA: f32 = 0.38;

    /// The alpha of the container of a box that answers nothing.
    pub const DISABLED_CONTAINER_ALPHA: f32 = 0.12;

    /// The vertices of the check mark, as fractions of the box drawn in.
    pub const CHECK_MARK: [Offset; 3] = [
        Offset::new(0.204, 0.527),
        Offset::new(0.387, 0.710),
        Offset::new(0.811, 0.286),
    ];

    /// The ends of the bar of an indeterminate box, as fractions of it.
    pub const INDETERMINATE_BAR: [Offset; 2] = [Offset::new(0.2375, 0.5), Offset::new(0.7625, 0.5)];
}

/// What the box of a check box holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CheckState {
    /// Nothing is chosen.
    #[default]
    Unchecked,
    /// The value is on.
    Checked,
    /// The value is neither on nor off — the state of a group whose
    /// members disagree.
    Indeterminate,
}

impl CheckState {
    /// Whether the box is on.
    ///
    /// An indeterminate box is not checked: it shows a bar rather than a
    /// mark, and a click on it leaves the indeterminate state behind
    /// rather than confirming it.
    #[inline]
    #[must_use]
    pub const fn is_checked(self) -> bool {
        matches!(self, Self::Checked)
    }

    /// The state a click of an ordinary box lands on.
    #[inline]
    #[must_use]
    pub const fn toggled(self) -> Self {
        match self {
            Self::Checked | Self::Indeterminate => Self::Unchecked,
            Self::Unchecked => Self::Checked,
        }
    }

    /// The state a click of a tri-state box lands on.
    #[inline]
    #[must_use]
    pub const fn cycled(self) -> Self {
        match self {
            Self::Unchecked => Self::Checked,
            Self::Checked => Self::Indeterminate,
            Self::Indeterminate => Self::Unchecked,
        }
    }
}

/// What a check box does, as a set of options instead of booleans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CheckboxFlags(u8);

impl CheckboxFlags {
    /// The box only paints the value it is given.
    pub const NONE: CheckboxFlags = CheckboxFlags(0);
    /// The box answers a click and a key. Clearing this bit is how a box
    /// is disabled: the press is never reported, the value is never
    /// written by the pointer, and the content is drawn with the disabled
    /// alphas.
    pub const ENABLED: CheckboxFlags = CheckboxFlags(1 << 0);
    /// The box takes the keyboard focus when a button is pressed inside
    /// it, and reports [`InteractionState::FOCUSED`] to its source.
    pub const FOCUSABLE: CheckboxFlags = CheckboxFlags(1 << 1);
    /// The box can hold [`CheckState::Indeterminate`], so a click cycles
    /// through three states instead of toggling two.
    pub const TRI_STATE: CheckboxFlags = CheckboxFlags(1 << 2);
    /// An enabled box that takes the focus: what the platform component
    /// is when the caller leaves every parameter defaulted.
    pub const STANDARD: CheckboxFlags = CheckboxFlags(Self::ENABLED.0 | Self::FOCUSABLE.0);
    /// A box that does everything the framework can ask of it.
    pub const ALL: CheckboxFlags = CheckboxFlags(Self::ENABLED.0 | Self::FOCUSABLE.0 | Self::TRI_STATE.0);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: CheckboxFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no option at all is selected.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set with the bits of `other` added.
    #[inline]
    #[must_use]
    pub const fn union(self, other: CheckboxFlags) -> Self {
        CheckboxFlags(self.0 | other.0)
    }

    /// The set with the bits of `other` removed.
    #[inline]
    #[must_use]
    pub const fn without(self, other: CheckboxFlags) -> Self {
        CheckboxFlags(self.0 & !other.0)
    }
}

impl BitOr for CheckboxFlags {
    type Output = CheckboxFlags;

    #[inline]
    fn bitor(self, rhs: CheckboxFlags) -> CheckboxFlags {
        CheckboxFlags(self.0 | rhs.0)
    }
}

impl BitOrAssign for CheckboxFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: CheckboxFlags) {
        self.0 |= rhs.0;
    }
}

/// The colors of the container, the outline, the glyph and the state
/// layer of a check box.
///
/// The four are decided together because they disagree on purpose: a
/// checked box has a container and no outline, an unchecked one the other
/// way around, and an indeterminate box keeps the outline of the unchecked
/// box while its glyph becomes a bar. Returning them as one value is what
/// stops the drawing code from re-deriving the rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckboxColors {
    /// The fill of the box.
    pub container: Color,
    /// The outline of the box.
    pub border: Color,
    /// The glyph — the mark or the bar — of the box.
    pub content: Color,
    /// The color the state layer is tinted with.
    pub state_layer: Color,
}

impl CheckboxColors {
    /// The colors of a box holding `state` in `scheme` under `flags`.
    ///
    /// An enabled unchecked box has no container at all: the surface it
    /// sits on shows through, which is why the state layer is drawn behind
    /// the box instead of being blended into it.
    #[must_use]
    pub const fn for_state(state: CheckState, scheme: &ColorScheme, flags: CheckboxFlags) -> Self {
        let state_layer = if flags.contains(CheckboxFlags::ENABLED) {
            scheme.primary
        } else {
            scheme.on_surface
        };
        match state {
            CheckState::Checked => Self {
                container: if flags.contains(CheckboxFlags::ENABLED) {
                    scheme.primary
                } else {
                    scheme
                        .on_surface
                        .with_alpha(defaults::DISABLED_CONTAINER_ALPHA)
                },
                border: Color::TRANSPARENT,
                content: if flags.contains(CheckboxFlags::ENABLED) {
                    scheme.on_primary
                } else {
                    scheme
                        .on_surface
                        .with_alpha(defaults::DISABLED_CONTENT_ALPHA)
                },
                state_layer,
            },
            CheckState::Indeterminate => {
                // The bar is drawn in the color of the frame it stands in,
                // so the two share one decision and the box never looks
                // like two components that disagree.
                let border = if flags.contains(CheckboxFlags::ENABLED) {
                    scheme.on_surface_variant
                } else {
                    scheme
                        .on_surface
                        .with_alpha(defaults::DISABLED_CONTENT_ALPHA)
                };
                Self {
                    container: Color::TRANSPARENT,
                    border,
                    content: border,
                    state_layer,
                }
            }
            CheckState::Unchecked => Self {
                container: Color::TRANSPARENT,
                border: if flags.contains(CheckboxFlags::ENABLED) {
                    scheme.on_surface_variant
                } else {
                    scheme
                        .on_surface
                        .with_alpha(defaults::DISABLED_CONTENT_ALPHA)
                },
                content: Color::TRANSPARENT,
                state_layer,
            },
        }
    }

    /// The state layer of a box in `interaction`.
    ///
    /// The layers of Material 3 do not add up: the strongest one that
    /// applies is what shows, so a box that is focused and hovered is no
    /// darker than a focused one. A box in no state at all returns a fully
    /// transparent color, which the drawing code skips.
    #[must_use]
    pub const fn overlay_for(&self, interaction: InteractionState) -> Color {
        let mut alpha = 0.0;
        if interaction.contains(InteractionState::HOVERED) && defaults::HOVER_ALPHA > alpha {
            alpha = defaults::HOVER_ALPHA;
        }
        if interaction.contains(InteractionState::PRESSED) && defaults::PRESSED_ALPHA > alpha {
            alpha = defaults::PRESSED_ALPHA;
        }
        if interaction.contains(InteractionState::FOCUSED) && defaults::FOCUSED_ALPHA > alpha {
            alpha = defaults::FOCUSED_ALPHA;
        }
        self.state_layer.with_alpha(alpha)
    }
}

/// A box that holds one of three values and answers the pointer for it.
///
/// The box measures itself to the [`TOUCH_TARGET`](defaults::TOUCH_TARGET)
/// square, so the target — not the frame — is what the layout reserves, and
/// paints the frame centered in it.
///
/// ```text
/// let agreed = ctx.use_state(|| MutableState::new(CheckState::Unchecked));
/// Checkbox::new().toggle(agreed.clone()).on_change(move |state| log(state))
/// ```
///
/// # The press it does not take
///
/// A press inside the box is reported to the interaction source but not
/// consumed, so an ancestor viewport sees the same down and can start the
/// drag of a finger that lands on the box and travels. Only the
/// *completion* of a press is consumed, and the value is written by that
/// and nothing else. A release the box takes stops there: the ancestor is
/// told the press it held is over, which is what keeps the hover after a
/// click from scrolling the list.
#[derive(Clone)]
pub struct Checkbox {
    /// The value the box shows and a gesture writes.
    state: MutableState<CheckState>,
    /// What the box answers to.
    flags: CheckboxFlags,
    /// The region that turns the events of the target into gestures.
    region: GestureRegion,
    /// The interaction states the application observes.
    interaction: InteractionSource,
    /// The handler called with the value the *user* chose.
    on_change: Option<CheckChange>,
    /// The square the box was last placed in, in window coordinates.
    ///
    /// A widget is handed no bounds when an event reaches it, so the box
    /// records the rectangle it was placed in: the region needs it to
    /// decide which events belong to it.
    bounds: Rect,
}

impl Checkbox {
    /// An enabled check box holding [`CheckState::Unchecked`].
    ///
    /// The value of a box built this way belongs to the box, so it is only
    /// worth keeping for a tree that is never recomposed; an application
    /// that reads or drives the value hands a handle with [`Self::toggle`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: MutableState::new(CheckState::Unchecked),
            flags: CheckboxFlags::STANDARD,
            region: GestureRegion::new(ViewportDispatch::default()),
            interaction: InteractionSource::new(),
            on_change: None,
            bounds: Rect::ZERO,
        }
    }

    /// The box that shows and writes `state`.
    #[must_use]
    pub fn toggle(mut self, state: MutableState<CheckState>) -> Self {
        self.state = state;
        self
    }

    /// The box that answers the gestures its table names.
    ///
    /// A slot of the table observes or overrides that one gesture and keeps
    /// the built-in answer of the others, so a box that has to report every
    /// click of a group writes [`on_click`](ViewportDispatch::on_click) and
    /// still toggles.
    #[must_use]
    pub fn dispatch(mut self, table: ViewportDispatch) -> Self {
        self.region = GestureRegion::new(table);
        self
    }

    /// The box that recognizes its gestures through `region`.
    ///
    /// The region is a handle like [`Self::toggle`] and
    /// [`Self::interaction_with`]: the recognition of the press in flight
    /// lives behind an [`Rc`](alloc::rc::Rc), so a content closure that
    /// builds one region and hands the same one every render keeps a press
    /// that spans a recomposition. A region built with each render forgets
    /// the press the moment the box writes the value the tree reads back, and
    /// the release that follows finds no press to answer for.
    #[must_use]
    pub fn region(mut self, region: GestureRegion) -> Self {
        self.region = region;
        self
    }

    /// The box that reports the value the user chose to `handler`.
    #[must_use]
    pub fn on_change(mut self, handler: impl Fn(CheckState) + 'static) -> Self {
        self.on_change = Some(Rc::new(handler));
        self
    }

    /// The box that reports its interaction states to `source`.
    ///
    /// The source is a handle, so the application keeps observing the same
    /// states across the recompositions that rebuilt the widget — which is
    /// how a row that tints itself on the hover of its box reads it.
    #[must_use]
    pub fn interaction_with(mut self, source: InteractionSource) -> Self {
        self.interaction = source;
        self
    }

    /// The box that answers — or, with `on` cleared, ignores — the pointer
    /// and the keyboard.
    #[must_use]
    pub fn enabled(mut self, on: bool) -> Self {
        self.set(CheckboxFlags::ENABLED, on);
        self
    }

    /// The box that takes — or, with `on` cleared, gives up — the keyboard
    /// focus.
    #[must_use]
    pub fn focusable(mut self, on: bool) -> Self {
        self.set(CheckboxFlags::FOCUSABLE, on);
        self
    }

    /// The box that cycles through the indeterminate state as well.
    #[must_use]
    pub fn tri_state(mut self, on: bool) -> Self {
        self.set(CheckboxFlags::TRI_STATE, on);
        self
    }

    /// Turns `bits` on when `on` names them, and off when it does not.
    fn set(&mut self, bits: CheckboxFlags, on: bool) {
        self.flags = if on {
            self.flags.union(bits)
        } else {
            self.flags.without(bits)
        };
    }

    /// The value the box shows.
    #[inline]
    #[must_use]
    pub const fn value(&self) -> &MutableState<CheckState> {
        &self.state
    }

    /// The options the box was built with.
    #[inline]
    #[must_use]
    pub const fn flags(&self) -> CheckboxFlags {
        self.flags
    }

    /// The interaction states of the box.
    #[inline]
    #[must_use]
    pub const fn interaction(&self) -> &InteractionSource {
        &self.interaction
    }

    /// Whether a press is held inside the box.
    #[inline]
    #[must_use]
    pub fn is_pressed(&self) -> bool {
        self.region.is_pressed()
    }

    /// Writes the value a click of these flags produces and reports it.
    fn flip(&self) {
        flip(&self.state, &self.on_change, self.flags);
    }
}

impl Default for Checkbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Checkbox {
    /// Reports the touch target, not the frame: the square a press is
    /// answered in is the space the layout has to reserve.
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::new(defaults::TOUCH_TARGET, defaults::TOUCH_TARGET))
    }

    /// Records the square the gestures are recognized inside.
    fn place(&mut self, _scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.bounds = bounds;
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let state = self.state.get();
        let colors = CheckboxColors::for_state(state, &scope.theme().colors, self.flags);
        let overlay = colors.overlay_for(self.interaction.state());
        let radii = CornerRadii::uniform(defaults::CORNER);
        // The layer first: it tints the surface the box sits on, and a
        // checked container has to cover it rather than mix with it.
        if overlay.alpha > 0.0 {
            let layer = square_in(bounds, defaults::TOUCH_TARGET);
            scope.fill_rounded_rect(layer, CornerRadii::uniform(defaults::TOUCH_TARGET / 2.0), overlay);
        }
        let container = square_in(bounds, defaults::CHECKED_BOX);
        if colors.container.alpha > 0.0 {
            scope.fill_rounded_rect(container, radii, colors.container);
        }
        // The outline is stroked on the 20-pixel path, so it straddles the
        // edge the way the platform draws it and an unchecked box reads
        // larger than the filled one it toggles to.
        let frame = square_in(bounds, defaults::BOX);
        if colors.border.alpha > 0.0 {
            scope.stroke_rect(frame, radii, colors.border, defaults::STROKE);
        }
        draw_glyph(scope, frame, state, colors.content);
    }

    /// The box reads the pointer only while it is enabled: a disabled box
    /// lets the press fall through to whatever can still answer it, which
    /// is what lets a list scroll over its disabled rows.
    fn hit_testable(&self) -> bool {
        self.flags.contains(CheckboxFlags::ENABLED)
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        if !self.flags.contains(CheckboxFlags::ENABLED) {
            return EventResult::Ignored;
        }
        if matches!(event, Event::Pointer(pointer) if pointer.kind == PointerEventKind::Down)
            && self.flags.contains(CheckboxFlags::FOCUSABLE)
        {
            scope.request_focus();
            self.interaction.set(InteractionState::FOCUSED);
        }
        if let Event::Key(key) = event {
            // A key belongs to the box only while it has the focus, and it
            // writes the value on the *release* the way a button does.
            let chosen = matches!(key.key, Key::Space | Key::Enter);
            if chosen && !key.pressed && scope.is_focused() {
                self.flip();
                return EventResult::Consumed;
            }
            return EventResult::Ignored;
        }
        let interaction = self.interaction.clone();
        let state = self.state.clone();
        let on_change = self.on_change.clone();
        let flags = self.flags;
        self.region
            .on_event(event, self.bounds, move |gesture| {
                match gesture {
                    Gesture::Press { .. } => {
                        interaction.set(InteractionState::PRESSED);
                        // Reported, not taken: an ancestor has to see the same
                        // down to drag with it.
                        EventResult::Ignored
                    }
                    Gesture::Hover { inside: true } => {
                        interaction.set(InteractionState::HOVERED);
                        EventResult::Ignored
                    }
                    Gesture::Hover { .. } => {
                        interaction.clear(InteractionState::HOVERED | InteractionState::PRESSED);
                        EventResult::Ignored
                    }
                    Gesture::Drag { .. } => {
                        // The finger travelled, so this press will never become
                        // a click; the travel itself is the ancestor's business.
                        interaction.clear(InteractionState::PRESSED);
                        EventResult::Ignored
                    }
                    Gesture::Cancel => {
                        interaction.clear(InteractionState::HOVERED | InteractionState::PRESSED);
                        EventResult::Ignored
                    }
                    Gesture::Click { .. } => {
                        interaction.clear(InteractionState::PRESSED);
                        flip(&state, &on_change, flags);
                        EventResult::Consumed
                    }
                    _ => EventResult::Ignored,
                }
            })
    }
}

/// Writes the value a click of `flags` produces and reports it.
///
/// A tri-state box cycles and an ordinary one toggles, and either way the
/// write is what marks the state dirty — the recomposition that follows is
/// the reason the box is drawn with its new glyph.
fn flip(state: &MutableState<CheckState>, on_change: &Option<CheckChange>, flags: CheckboxFlags) {
    let current = state.get();
    let next = if flags.contains(CheckboxFlags::TRI_STATE) {
        current.cycled()
    } else {
        current.toggled()
    };
    state.set(next);
    if let Some(handler) = on_change {
        handler(next);
    }
}

/// The square of `side` centered in `bounds`, in the space of the widget.
///
/// The box a widget is placed in is in the coordinates of the window while
/// the drawing scope starts at the origin of that box, so every piece of
/// the geometry is measured from the corner the widget was given.
#[inline]
fn square_in(bounds: Rect, side: f32) -> Rect {
    let center = bounds.center();
    let half = side / 2.0;
    Rect::from_min_size(
        Offset::new(center.x - half - bounds.left(), center.y - half - bounds.top()),
        side,
        side,
    )
}

/// The point of `rect` that `fraction` of it names.
#[inline]
fn point_in(rect: Rect, fraction: Offset) -> Offset {
    Offset::new(
        rect.left() + rect.size.width * fraction.x,
        rect.top() + rect.size.height * fraction.y,
    )
}

/// Draws the glyph of `state` inside `rect`.
///
/// The mark is the two strokes that meet at its corner rather than one
/// path, because the framework has no polyline primitive and the corner
/// join is the one thing a check mark must not round off.
fn draw_glyph(scope: &mut DrawScope<'_>, rect: Rect, state: CheckState, color: Color) {
    if color.alpha <= 0.0 {
        return;
    }
    match state {
        CheckState::Checked => {
            let [first, middle, last] = defaults::CHECK_MARK.map(|point| point_in(rect, point));
            scope.draw_line(first, middle, color, defaults::STROKE);
            scope.draw_line(middle, last, color, defaults::STROKE);
        }
        CheckState::Indeterminate => {
            let [first, last] = defaults::INDETERMINATE_BAR.map(|point| point_in(rect, point));
            scope.draw_line(first, last, color, defaults::STROKE);
        }
        CheckState::Unchecked => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::Composition;
    use crate::event::{KeyEvent, PointerEvent};
    use crate::node::{self, Node};
    use crate::paint::{DrawList, LineOp, PaintStyle, RectOp};
    use crate::theme::Theme;
    use crate::widgets::{ScrollAxes, ScrollState, Viewport};
    use core::cell::Cell;

    /// The square a test box is placed in, in window coordinates.
    const TARGET: Rect = Rect::from_min_size(Offset::ZERO, 40.0, 40.0);

    /// The center of [`TARGET`], where a press belongs to the box.
    const CENTER: Offset = Offset::new(20.0, 20.0);

    /// A point no box of a [`TARGET`] contains.
    const OUTSIDE: Offset = Offset::new(70.0, 70.0);

    /// A tree of one check box, with the state dispatching an event needs.
    struct Harness {
        theme: Theme,
        node: Node,
        focus: Option<Vec<usize>>,
    }

    impl Harness {
        /// A tree of `checkbox`, measured and placed in [`TARGET`].
        fn new(checkbox: Checkbox) -> Self {
            let theme = Theme::light();
            let mut node = Node::new(checkbox);
            node.measure(&theme, Constraints::unbounded());
            node.place(TARGET);
            Self {
                theme,
                node,
                focus: None,
            }
        }

        /// Offers `event` to the tree and reports what it answered.
        fn offer(&mut self, event: &Event) -> EventResult {
            let mut path = Vec::new();
            node::dispatch_event(&mut self.node, &self.theme, &mut path, &mut self.focus, event)
        }

        /// Offers a pointer event of `kind` at `position` to the tree.
        fn pointer(&mut self, kind: PointerEventKind, position: Offset) -> EventResult {
            self.offer(&Event::Pointer(PointerEvent::new(kind, position)))
        }

        /// Offers the release of `key` to the tree.
        fn release(&mut self, key: Key) -> EventResult {
            self.offer(&Event::Key(KeyEvent::new(key, false)))
        }

        /// The box as it paints itself right now.
        fn canvas(&mut self) -> Canvas {
            let mut list = DrawList::new();
            self.node.draw(&self.theme, &mut list);
            Canvas::of(&list)
        }
    }

    /// The shapes a draw list recorded, split by how they were painted.
    #[derive(Default)]
    struct Canvas {
        /// The filled rectangles.
        fills: Vec<(Rect, Color)>,
        /// The outlined rectangles.
        outlines: Vec<(Rect, Color)>,
        /// The strokes of the glyph: their ends and their color.
        lines: Vec<(Offset, Offset, Color)>,
    }

    impl Canvas {
        /// The recorded shapes of `list`, in the order it recorded them.
        fn of(list: &DrawList) -> Self {
            let mut canvas = Self::default();
            for op in list.ops() {
                let any: &dyn core::any::Any = op;
                if let Some(rect) = any.downcast_ref::<RectOp>() {
                    match rect.paint.style {
                        PaintStyle::Fill => canvas.fills.push((rect.rect, rect.paint.color)),
                        PaintStyle::Stroke(_) => canvas
                            .outlines
                            .push((rect.rect, rect.paint.color)),
                    }
                } else if let Some(line) = any.downcast_ref::<LineOp>() {
                    canvas
                        .lines
                        .push((line.from, line.to, line.paint.color));
                }
            }
            canvas
        }
    }

    /// Whether `actual` is `expected` to within a hundredth of a pixel.
    fn close(actual: Offset, expected: Offset) -> bool {
        (actual.x - expected.x).abs() < 0.01 && (actual.y - expected.y).abs() < 0.01
    }

    #[test]
    fn a_check_box_measures_to_its_touch_target() {
        let harness = Harness::new(Checkbox::new());
        assert_eq!(harness.node.size(), Size::new(40.0, 40.0));
    }

    #[test]
    fn a_click_toggles_the_value_and_reports_it() {
        let value = MutableState::new(CheckState::Unchecked);
        let seen = Rc::new(Cell::new(None));
        let reported = Rc::clone(&seen);
        let mut harness = Harness::new(
            Checkbox::new()
                .toggle(value.clone())
                .on_change(move |state| reported.set(Some(state))),
        );
        harness.pointer(PointerEventKind::Down, CENTER);
        assert!(
            harness
                .pointer(PointerEventKind::Up, CENTER)
                .is_consumed(),
            "a press completed inside the box is the box's click"
        );
        assert_eq!(value.get(), CheckState::Checked);
        assert_eq!(seen.get(), Some(CheckState::Checked));
        harness.pointer(PointerEventKind::Down, CENTER);
        harness.pointer(PointerEventKind::Up, CENTER);
        assert_eq!(value.get(), CheckState::Unchecked);
        assert_eq!(seen.get(), Some(CheckState::Unchecked));
    }

    #[test]
    fn the_press_of_a_box_is_reported_but_not_taken() {
        let source = InteractionSource::new();
        let mut harness = Harness::new(Checkbox::new().interaction_with(source.clone()));
        assert!(
            !harness
                .pointer(PointerEventKind::Down, CENTER)
                .is_consumed()
        );
        assert!(source.state().contains(InteractionState::PRESSED));
        // The leave of the pointer ends the press the way it ends the
        // hover: a button held in a box the pointer left is not a click.
        harness.pointer(PointerEventKind::Move, OUTSIDE);
        assert!(!source.state().contains(InteractionState::PRESSED));
        assert!(!source.state().contains(InteractionState::HOVERED));
    }

    #[test]
    fn a_release_outside_the_target_is_not_a_click() {
        let value = MutableState::new(CheckState::Unchecked);
        let mut harness = Harness::new(Checkbox::new().toggle(value.clone()));
        harness.pointer(PointerEventKind::Down, CENTER);
        // The press still belongs to the box, so the release is its to
        // answer — and answering it is not the same as completing a click.
        assert!(
            harness
                .pointer(PointerEventKind::Up, OUTSIDE)
                .is_consumed()
        );
        assert_eq!(value.get(), CheckState::Unchecked);
    }

    #[test]
    fn a_press_that_travelled_never_becomes_a_click() {
        let value = MutableState::new(CheckState::Unchecked);
        let source = InteractionSource::new();
        let mut harness = Harness::new(
            Checkbox::new()
                .toggle(value.clone())
                .interaction_with(source.clone()),
        );
        harness.pointer(PointerEventKind::Down, CENTER);
        let travelled = Offset::new(20.0, 30.0);
        assert!(
            !harness
                .pointer(PointerEventKind::Move, travelled)
                .is_consumed(),
            "a drag that started on a box is left to the list that holds it"
        );
        assert!(!source.state().contains(InteractionState::PRESSED));
        harness.pointer(PointerEventKind::Up, travelled);
        assert_eq!(value.get(), CheckState::Unchecked);
    }

    #[test]
    fn a_tri_state_box_cycles_through_the_bar() {
        let value = MutableState::new(CheckState::Unchecked);
        let mut harness = Harness::new(
            Checkbox::new()
                .toggle(value.clone())
                .tri_state(true),
        );
        for expected in [
            CheckState::Checked,
            CheckState::Indeterminate,
            CheckState::Unchecked,
        ] {
            harness.pointer(PointerEventKind::Down, CENTER);
            harness.pointer(PointerEventKind::Up, CENTER);
            assert_eq!(value.get(), expected);
        }
    }

    #[test]
    fn a_disabled_box_answers_nothing() {
        let value = MutableState::new(CheckState::Unchecked);
        let mut harness = Harness::new(
            Checkbox::new()
                .toggle(value.clone())
                .enabled(false),
        );
        assert!(
            !harness.node.hit_testable(),
            "the press falls through to whatever can still answer it"
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Down, CENTER)
                .is_consumed()
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Up, CENTER)
                .is_consumed()
        );
        assert!(!harness.release(Key::Space).is_consumed());
        assert_eq!(value.get(), CheckState::Unchecked);
    }

    #[test]
    fn a_key_release_toggles_the_box_that_has_the_focus() {
        let value = MutableState::new(CheckState::Unchecked);
        let mut harness = Harness::new(Checkbox::new().toggle(value.clone()));
        // The press is what takes the focus, so the keyboard answers the
        // box the pointer last touched rather than the root of the tree.
        harness.pointer(PointerEventKind::Down, CENTER);
        harness.pointer(PointerEventKind::Up, CENTER);
        assert_eq!(value.get(), CheckState::Checked);
        assert!(harness.release(Key::Space).is_consumed());
        assert_eq!(value.get(), CheckState::Unchecked);
        assert!(harness.release(Key::Enter).is_consumed());
        assert_eq!(value.get(), CheckState::Checked);
        // The press of the same key is not the release of it: a key held
        // down repeats, and a box that flipped on every repeat would
        // flicker under a finger that rests on the bar.
        assert!(
            !harness
                .offer(&Event::Key(KeyEvent::new(Key::Space, true)))
                .is_consumed()
        );
        assert_eq!(value.get(), CheckState::Checked);
    }

    #[test]
    fn an_unchecked_box_paints_its_outline_only() {
        let mut harness = Harness::new(Checkbox::new().focusable(false));
        let canvas = harness.canvas();
        assert!(canvas.fills.is_empty(), "no container and no layer at rest");
        assert!(canvas.lines.is_empty());
        assert_eq!(
            canvas.outlines,
            [(
                Rect::from_min_size(Offset::new(10.0, 10.0), 20.0, 20.0),
                Theme::light().colors.on_surface_variant
            )]
        );
    }

    #[test]
    fn a_checked_box_paints_its_container_and_its_mark() {
        let value = MutableState::new(CheckState::Checked);
        let mut harness = Harness::new(Checkbox::new().toggle(value));
        let canvas = harness.canvas();
        assert_eq!(
            canvas.fills,
            [(
                Rect::from_min_size(Offset::new(11.0, 11.0), 18.0, 18.0),
                Theme::light().colors.primary
            )]
        );
        assert!(
            canvas.outlines.is_empty(),
            "the frame of a checked box is its container"
        );
        assert_eq!(canvas.lines.len(), 2, "the mark is two strokes");
        // The vertices of the mark are the Material ones, measured from
        // the 20-pixel frame the box paints its glyph in.
        assert!(close(canvas.lines[0].0, Offset::new(14.08, 20.54)));
        assert!(close(canvas.lines[0].1, Offset::new(17.74, 24.20)));
        assert!(close(canvas.lines[1].1, Offset::new(26.22, 15.72)));
    }

    #[test]
    fn an_indeterminate_box_keeps_its_frame_and_paints_a_bar() {
        let value = MutableState::new(CheckState::Indeterminate);
        let mut harness = Harness::new(Checkbox::new().toggle(value));
        let canvas = harness.canvas();
        assert_eq!(canvas.outlines.len(), 1, "the frame of the group stays");
        assert!(canvas.fills.is_empty());
        assert_eq!(canvas.lines.len(), 1);
        assert_eq!(canvas.lines[0].0.y, canvas.lines[0].1.y, "the bar is level");
        assert!((canvas.lines[0].1.x - canvas.lines[0].0.x - 10.5).abs() < 0.01);
    }

    #[test]
    fn a_disabled_box_paints_the_disabled_roles() {
        let value = MutableState::new(CheckState::Checked);
        let mut harness = Harness::new(Checkbox::new().toggle(value).enabled(false));
        let canvas = harness.canvas();
        let scheme = Theme::light().colors;
        assert_eq!(
            canvas.fills[0].1,
            scheme
                .on_surface
                .with_alpha(defaults::DISABLED_CONTAINER_ALPHA)
        );
        assert_eq!(
            canvas.lines[0].2,
            scheme
                .on_surface
                .with_alpha(defaults::DISABLED_CONTENT_ALPHA)
        );
    }

    #[test]
    fn the_state_layer_shades_the_target_of_a_held_box() {
        let mut harness = Harness::new(Checkbox::new().focusable(false));
        harness.pointer(PointerEventKind::Down, CENTER);
        let canvas = harness.canvas();
        assert_eq!(canvas.fills[0].0, TARGET, "the layer is the target");
        assert!(
            (canvas.fills[0].1.alpha - defaults::PRESSED_ALPHA).abs() < 0.001,
            "{:?}",
            canvas.fills[0].1
        );
    }

    #[test]
    fn a_drag_from_a_box_still_scrolls_its_list() {
        let offset = ScrollState::new(ScrollAxes::VERTICAL);
        let value = MutableState::new(CheckState::Unchecked);
        let theme = Theme::light();
        let mut node = Node::new(Viewport::new().scroll(offset.clone()))
            .child(Node::new(Checkbox::new().toggle(value.clone())));
        node.measure(&theme, Constraints::tight(Size::new(40.0, 20.0)));
        node.place(Rect::from_min_size(Offset::ZERO, 40.0, 20.0));
        let mut focus = None;
        let mut offer = |event: &Event| {
            let mut path = Vec::new();
            node::dispatch_event(&mut node, &theme, &mut path, &mut focus, event)
        };
        let start = Offset::new(20.0, 10.0);
        offer(&Event::Pointer(PointerEvent::new(PointerEventKind::Down, start)));
        assert!(
            offer(&Event::Pointer(PointerEvent::new(
                PointerEventKind::Move,
                Offset::new(20.0, 5.0)
            )))
            .is_consumed(),
            "the list answers the drag the box left to it"
        );
        assert_eq!(offset.offset().y, 5.0);
        offer(&Event::Pointer(PointerEvent::new(
            PointerEventKind::Up,
            Offset::new(20.0, 5.0),
        )));
        assert_eq!(value.get(), CheckState::Unchecked, "a drag is not a click");
    }

    #[test]
    fn the_hover_after_a_click_does_not_scroll_the_list() {
        let offset = ScrollState::new(ScrollAxes::VERTICAL);
        let value = MutableState::new(CheckState::Unchecked);
        let theme = Theme::light();
        let mut node = Node::new(Viewport::new().scroll(offset.clone()))
            .child(Node::new(Checkbox::new().toggle(value.clone())));
        node.measure(&theme, Constraints::tight(Size::new(40.0, 20.0)));
        node.place(Rect::from_min_size(Offset::ZERO, 40.0, 20.0));
        let mut focus = None;
        let mut offer = |event: &Event| {
            let mut path = Vec::new();
            node::dispatch_event(&mut node, &theme, &mut path, &mut focus, event)
        };
        let start = Offset::new(20.0, 10.0);
        offer(&Event::Pointer(PointerEvent::new(PointerEventKind::Down, start)));
        offer(&Event::Pointer(PointerEvent::new(PointerEventKind::Up, start)));
        assert_eq!(value.get(), CheckState::Checked);
        // The box took the release, so the list was told its press ended
        // rather than left holding it: this move is a hover, not a drag.
        offer(&Event::Pointer(PointerEvent::new(
            PointerEventKind::Move,
            Offset::new(20.0, 0.0),
        )));
        assert_eq!(offset.offset(), Offset::ZERO);
    }

    #[test]
    fn a_click_survives_the_recomposition_that_rebuilt_the_box() {
        let value = MutableState::new(CheckState::Unchecked);
        let region = GestureRegion::default();
        let mut composition = Composition::new(Theme::light());
        let toggled = value.clone();
        let held = region.clone();
        composition.set_content(move |_| {
            Node::new(
                Checkbox::new()
                    .toggle(toggled.clone())
                    .region(held.clone()),
            )
        });
        composition.recompose(true);
        composition.layout(TARGET.size, true);
        let start = Offset::new(20.0, 20.0);
        composition.handle_event(&Event::Pointer(PointerEvent::new(PointerEventKind::Down, start)));
        // A value the tree reads back rebuilds the widget while the finger is
        // still down; only the region the host owns carries the press over.
        composition.recompose(true);
        composition.layout(TARGET.size, true);
        composition.handle_event(&Event::Pointer(PointerEvent::new(PointerEventKind::Up, start)));
        assert_eq!(
            value.get(),
            CheckState::Checked,
            "the release found the press the region the host owns still held"
        );
    }

    #[test]
    fn a_region_built_with_the_box_forgets_a_press_a_recompose_ends() {
        let value = MutableState::new(CheckState::Unchecked);
        let mut composition = Composition::new(Theme::light());
        let toggled = value.clone();
        composition.set_content(move |_| Node::new(Checkbox::new().toggle(toggled.clone())));
        composition.recompose(true);
        composition.layout(TARGET.size, true);
        let start = Offset::new(20.0, 20.0);
        composition.handle_event(&Event::Pointer(PointerEvent::new(PointerEventKind::Down, start)));
        composition.recompose(true);
        composition.layout(TARGET.size, true);
        composition.handle_event(&Event::Pointer(PointerEvent::new(PointerEventKind::Up, start)));
        // The same sequence as the test above, with the region named by the
        // rebuild rather than the host: the new region never saw the press, so
        // the release answers for nothing and the box stays as it was.
        assert_eq!(value.get(), CheckState::Unchecked);
    }
}
