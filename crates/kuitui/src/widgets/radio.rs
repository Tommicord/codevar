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

//! The radio button: one [`RadioGroup`] holding one of many values, each
//! of them named by a button.
//!
//! A radio is the check box's sibling: it reads the pointer through the
//! same [`GestureRegion`], reports the same interaction states, and claims
//! a press only when it completes one. What makes it a different component
//! is not the shape but where its value lives. A check box holds its own
//! state and a click flips it; a radio holds the *option it stands for* and
//! the group holds the *option that is chosen*, so exclusivity — the one
//! rule that makes a radio a radio — is a property of the state and can
//! never be broken by the tree the buttons are arranged in.
//!
//! # The group outlives the render
//!
//! [`RadioGroup`] is a handle, like [`InteractionSource`](crate::effect::InteractionSource):
//! cloning it yields another view of the same selection, so a content
//! closure creates it once — with
//! [`use_state`](codevar_kuitui_rea::RenderCtx::use_state) — and hands the
//! same handle to every button of the group. Each button *reads* the group
//! while it draws, which is what makes the group reactive: a click on one
//! button writes the state, the recomposition that follows draws every
//! sibling against the new value, and the button that lost its dot never
//! has to be told about it.
//!
//! # What a click cannot do
//!
//! A click on the chosen button changes nothing: a radio group is emptied
//! by [`RadioGroup::clear`], which only the application calls, because the
//! platform rule a radio answers to is that one of the options always
//! stands. A click on an unchosen button moves the selection, and the
//! handler of [`Radio::on_change`] runs for that move alone — an
//! application that selects an option itself with
//! [`RadioGroup::select`] changes what the buttons paint without being
//! told that it did, the same split Compose draws between `selected` and
//! `onClick`.
//!
//! # The value of a button
//!
//! `T` is the tag of the options — the enum or the id the application
//! compares — and not the state of the group: the group stores
//! `Option<T>` so that "nothing chosen yet" is a state the widget can hold
//! without a sentinel value of its own.

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

/// The handler a radio calls with the option the user chose.
pub type RadioChange<T> = Rc<dyn Fn(&T)>;

/// The geometry and the alphas of the Material 3 radio button.
///
/// The numbers are the ones the platform component is specified with: the
/// ring is 20 logical pixels across and 2 wide whether the button is chosen
/// or not, and a chosen button fills a dot of 10 pixels — half the ring —
/// in the same color as the ring it stands in.
pub mod defaults {
    /// The diameter of the ring of a radio.
    pub const OUTER: f32 = 20.0;

    /// The width of the ring.
    ///
    /// Unlike the frame of a check box, the ring does not contract when the
    /// button is chosen: a radio keeps its outline in both states, which is
    /// why the dot is a separate diameter rather than a filled frame.
    pub const STROKE: f32 = 2.0;

    /// The diameter of the dot of a chosen radio.
    pub const DOT: f32 = 10.0;

    /// The side of the square a press is answered in.
    ///
    /// The touch target is the ring with the padding a finger needs around
    /// it, and the state layer is painted over that whole square — which is
    /// why the layer is drawn as a circle of this diameter.
    pub const TOUCH_TARGET: f32 = 40.0;

    /// The alpha of the state layer while the pointer is over the button.
    pub const HOVER_ALPHA: f32 = 0.08;

    /// The alpha of the state layer while a button is held in the target.
    pub const PRESSED_ALPHA: f32 = 0.10;

    /// The alpha of the state layer while the button has the keyboard
    /// focus.
    pub const FOCUSED_ALPHA: f32 = 0.12;

    /// The alpha of the ring and the dot of a button that answers nothing.
    pub const DISABLED_CONTENT_ALPHA: f32 = 0.38;
}

/// What a radio button does, as a set of options instead of booleans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RadioFlags(u8);

impl RadioFlags {
    /// The button only paints the selection it is given.
    pub const NONE: RadioFlags = RadioFlags(0);
    /// The button answers a click and a key. Clearing this bit is how a
    /// button is disabled: the press is never reported, the group is never
    /// written by the pointer, and the content is drawn with the disabled
    /// alpha.
    pub const ENABLED: RadioFlags = RadioFlags(1 << 0);
    /// The button takes the keyboard focus when a button is pressed inside
    /// it, and reports [`InteractionState::FOCUSED`] to its source.
    pub const FOCUSABLE: RadioFlags = RadioFlags(1 << 1);
    /// An enabled button that takes the focus: what the platform component
    /// is when the caller leaves every parameter defaulted. A radio has no
    /// third option to turn on, so this is the whole set.
    pub const STANDARD: RadioFlags = RadioFlags(Self::ENABLED.0 | Self::FOCUSABLE.0);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: RadioFlags) -> bool {
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
    pub const fn union(self, other: RadioFlags) -> Self {
        RadioFlags(self.0 | other.0)
    }

    /// The set with the bits of `other` removed.
    #[inline]
    #[must_use]
    pub const fn without(self, other: RadioFlags) -> Self {
        RadioFlags(self.0 & !other.0)
    }
}

impl BitOr for RadioFlags {
    type Output = RadioFlags;

    #[inline]
    fn bitor(self, rhs: RadioFlags) -> RadioFlags {
        RadioFlags(self.0 | rhs.0)
    }
}

impl BitOrAssign for RadioFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: RadioFlags) {
        self.0 |= rhs.0;
    }
}

/// The colors of the ring, the dot and the state layer of a radio.
///
/// A radio has no container: the surface it sits on shows through the ring
/// in both states, so the two colors that change are the outline and the
/// dot it holds. They are decided together because a chosen button draws
/// the dot in the color of its ring, and an application that tinted one of
/// them would otherwise have to keep the two in step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadioColors {
    /// The ring of the button.
    pub ring: Color,
    /// The dot of a chosen button.
    pub dot: Color,
    /// The color the state layer is tinted with.
    pub state_layer: Color,
}

impl RadioColors {
    /// The colors of a button standing for an option that is — or is not —
    /// `chosen` in `scheme` under `flags`.
    #[must_use]
    pub const fn for_chosen(chosen: bool, scheme: &ColorScheme, flags: RadioFlags) -> Self {
        let enabled = flags.contains(RadioFlags::ENABLED);
        let state_layer = if enabled {
            scheme.primary
        } else {
            scheme.on_surface
        };
        // A disabled button loses the color of its role rather than the
        // shape of it: the ring still shows what the outline of the option
        // was, and a chosen one still shows its dot, both at the disabled
        // alpha the platform specifies.
        let disabled = scheme
            .on_surface
            .with_alpha(defaults::DISABLED_CONTENT_ALPHA);
        Self {
            ring: if !enabled {
                disabled
            } else if chosen {
                scheme.primary
            } else {
                scheme.on_surface_variant
            },
            dot: if enabled { scheme.primary } else { disabled },
            state_layer,
        }
    }

    /// The state layer of a button in `interaction`.
    ///
    /// The layers of Material 3 do not add up: the strongest one that
    /// applies is what shows, so a button that is focused and hovered is no
    /// darker than a focused one. A button in no state at all returns a
    /// fully transparent color, which the drawing code skips.
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

/// The option a group of radio buttons holds as chosen.
///
/// The group is a handle over a `MutableState<Option<T>>`, so cloning it
/// yields another view of the same selection and a recomposition in the
/// middle of a press cannot lose what the press wrote. The `Option` is the
/// state the platform allows and a click does not: no option chosen at
/// all, which is how a group starts before the user has answered it.
///
/// Every write marks the state dirty, which is what makes the buttons that
/// read it recompose — and a write that would leave the same option chosen
/// is skipped, so re-selecting an option does not cost a recomposition of
/// the group.
pub struct RadioGroup<T> {
    /// The chosen option, shared with every button of the group.
    selection: MutableState<Option<T>>,
}

impl<T> Clone for RadioGroup<T> {
    fn clone(&self) -> Self {
        Self {
            selection: self.selection.clone(),
        }
    }
}

impl<T> RadioGroup<T> {
    /// The group with no option chosen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            selection: MutableState::new(None),
        }
    }

    /// The group with `value` chosen.
    #[must_use]
    pub fn with_value(value: T) -> Self {
        Self {
            selection: MutableState::new(Some(value)),
        }
    }

    /// The option the group holds, if any.
    ///
    /// A button reads the selection through [`Self::is_selected`] rather
    /// than through this, because comparing against a clone of the whole
    /// option is the work the group can do without copying the tag.
    #[must_use]
    pub fn current(&self) -> Option<T>
    where
        T: Clone,
    {
        self.selection.get()
    }

    /// The number of writes the selection received.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.selection.revision()
    }
}

impl<T: PartialEq> RadioGroup<T> {
    /// Whether `value` is the option the group holds.
    #[must_use]
    pub fn is_selected(&self, value: &T) -> bool {
        // The comparison runs inside the borrow of the state, so no clone
        // of the tag is made to answer a question a reference can settle.
        self.selection
            .with(|chosen| chosen.as_ref() == Some(value))
    }

    /// Chooses `value`, and reports whether the selection moved.
    ///
    /// The same option chosen twice is not a change: the state is left
    /// alone, so no dirty flag is set and the group is not recomposed.
    pub fn select(&self, value: &T) -> bool
    where
        T: Clone,
    {
        if self.is_selected(value) {
            return false;
        }
        self.selection.set(Some(value.clone()));
        true
    }

    /// Gives up the chosen option, and reports whether there was one.
    ///
    /// A click never calls this: an option of a radio group is given up by
    /// the application, not by the user choosing nothing.
    pub fn clear(&self) -> bool {
        if self.selection.with(|chosen| chosen.is_none()) {
            return false;
        }
        self.selection.set(None);
        true
    }
}

impl<T> Default for RadioGroup<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A button that stands for one option of a [`RadioGroup`].
///
/// The button measures itself to the [`TOUCH_TARGET`](defaults::TOUCH_TARGET)
/// square, so the target — not the ring — is what the space the group takes
/// in a row reserves, and paints the ring centered in it.
///
/// ```text
/// let chosen = ctx.use_state(|| RadioGroup::<Scheme>::new());
/// Row::new().child(Radio::new(Scheme::Dark, chosen.clone()))
///           .child(Radio::new(Scheme::Light, chosen.clone()))
/// ```
///
/// # The press it does not take
///
/// A press inside the target is reported to the interaction source but not
/// consumed, so an ancestor viewport sees the same down and can start the
/// drag of a finger that lands on the button and travels. Only the
/// *completion* of a press is consumed, and the group is written by that
/// and nothing else. A release the button takes stops there: the ancestor is
/// told the press it held is over, which is what keeps the hover after a
/// click from scrolling the list.
#[derive(Clone)]
pub struct Radio<T> {
    /// The option this button stands for.
    value: T,
    /// The selection this button reads and writes.
    group: RadioGroup<T>,
    /// What the button answers to.
    flags: RadioFlags,
    /// The region that turns the events of the target into gestures.
    region: GestureRegion,
    /// The interaction states the application observes.
    interaction: InteractionSource,
    /// The handler called with the option the *user* chose.
    on_change: Option<RadioChange<T>>,
    /// The square the button was last placed in, in window coordinates.
    ///
    /// A widget is handed no bounds when an event reaches it, so the button
    /// records the rectangle it was placed in: the region needs it to
    /// decide which events belong to it.
    bounds: Rect,
}

impl<T: Clone + PartialEq + 'static> Radio<T> {
    /// The button that stands for `value` in `group`.
    ///
    /// The button is enabled and takes the focus, which is what the
    /// platform component is when the caller leaves every parameter
    /// defaulted.
    #[must_use]
    pub fn new(value: T, group: RadioGroup<T>) -> Self {
        Self {
            value,
            group,
            flags: RadioFlags::STANDARD,
            region: GestureRegion::new(ViewportDispatch::default()),
            interaction: InteractionSource::new(),
            on_change: None,
            bounds: Rect::ZERO,
        }
    }

    /// The button that answers the gestures its table names.
    ///
    /// A slot of the table observes or overrides that one gesture and keeps
    /// the built-in answer of the others, so a group that has to watch
    /// every press writes [`on_press`](ViewportDispatch::on_press) and
    /// still chooses its option.
    #[must_use]
    pub fn dispatch(mut self, table: ViewportDispatch) -> Self {
        self.region = GestureRegion::new(table);
        self
    }

    /// The button that recognizes its gestures through `region`.
    ///
    /// The region is a handle like [`Self::interaction_with`]: the recognition
    /// of the press in flight lives behind an [`Rc`](alloc::rc::Rc), so a
    /// content closure that builds one region per option and hands the same
    /// one every render keeps a press that spans a recomposition. A region
    /// built with each render forgets the press the moment the button moves
    /// the selection the group reads back, and the release that follows finds
    /// no press to answer for.
    #[must_use]
    pub fn region(mut self, region: GestureRegion) -> Self {
        self.region = region;
        self
    }

    /// The button that reports the option the user chose to `handler`.
    ///
    /// The handler runs when the selection *moves*: a click on the chosen
    /// button and a write the application made itself both leave it alone,
    /// so what the handler sees is the answer the user gave rather than
    /// every change of the state.
    #[must_use]
    pub fn on_change(mut self, handler: impl Fn(&T) + 'static) -> Self {
        self.on_change = Some(Rc::new(handler));
        self
    }

    /// The button that reports its interaction states to `source`.
    ///
    /// The source is a handle, so the application keeps observing the same
    /// states across the recompositions that rebuilt the button — which is
    /// how a row that tints itself on the hover of its button reads it.
    #[must_use]
    pub fn interaction_with(mut self, source: InteractionSource) -> Self {
        self.interaction = source;
        self
    }

    /// The button that answers — or, with `on` cleared, ignores — the
    /// pointer and the keyboard.
    #[must_use]
    pub fn enabled(mut self, on: bool) -> Self {
        self.set(RadioFlags::ENABLED, on);
        self
    }

    /// The button that takes — or, with `on` cleared, gives up — the
    /// keyboard focus.
    #[must_use]
    pub fn focusable(mut self, on: bool) -> Self {
        self.set(RadioFlags::FOCUSABLE, on);
        self
    }

    /// Turns `bits` on when `on` names them, and off when it does not.
    fn set(&mut self, bits: RadioFlags, on: bool) {
        self.flags = if on {
            self.flags.union(bits)
        } else {
            self.flags.without(bits)
        };
    }

    /// The option this button stands for.
    #[inline]
    #[must_use]
    pub const fn option(&self) -> &T {
        &self.value
    }

    /// The selection this button reads and writes.
    #[inline]
    #[must_use]
    pub const fn group(&self) -> &RadioGroup<T> {
        &self.group
    }

    /// The options the button was built with.
    #[inline]
    #[must_use]
    pub const fn flags(&self) -> RadioFlags {
        self.flags
    }

    /// The interaction states of the button.
    #[inline]
    #[must_use]
    pub const fn interaction(&self) -> &InteractionSource {
        &self.interaction
    }

    /// Whether this button's option is the one the group holds.
    #[must_use]
    pub fn is_selected(&self) -> bool {
        self.group.is_selected(&self.value)
    }

    /// Whether a press is held inside the button.
    #[inline]
    #[must_use]
    pub fn is_pressed(&self) -> bool {
        self.region.is_pressed()
    }

    /// Chooses this button's option and reports it when the selection moved.
    fn choose(&self) {
        if self.group.select(&self.value)
            && let Some(handler) = &self.on_change
        {
            handler(&self.value);
        }
    }
}

impl<T: Clone + PartialEq + 'static> Widget for Radio<T> {
    /// Reports the touch target, not the ring: the square a press is
    /// answered in is the space the layout has to reserve.
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::new(defaults::TOUCH_TARGET, defaults::TOUCH_TARGET))
    }

    /// Records the square the gestures are recognized inside.
    fn place(&mut self, _scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.bounds = bounds;
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let chosen = self.is_selected();
        let colors = RadioColors::for_chosen(chosen, &scope.theme().colors, self.flags);
        let overlay = colors.overlay_for(self.interaction.state());
        let ring = square_in(bounds, defaults::OUTER);
        let radii = CornerRadii::uniform(defaults::OUTER / 2.0);
        // The layer first: it tints the surface the button sits on, and the
        // ring and the dot are drawn over it rather than mixed into it.
        if overlay.alpha > 0.0 {
            scope.fill_rounded_rect(
                square_in(bounds, defaults::TOUCH_TARGET),
                CornerRadii::uniform(defaults::TOUCH_TARGET / 2.0),
                overlay,
            );
        }
        if colors.ring.alpha > 0.0 {
            scope.stroke_rect(ring, radii, colors.ring, defaults::STROKE);
        }
        if chosen && colors.dot.alpha > 0.0 {
            let dot = square_in(bounds, defaults::DOT);
            scope.fill_rounded_rect(dot, CornerRadii::uniform(defaults::DOT / 2.0), colors.dot);
        }
    }

    /// The button reads the pointer only while it is enabled: a disabled
    /// button lets the press fall through to whatever can still answer it,
    /// which is what lets a list scroll over its disabled rows.
    fn hit_testable(&self) -> bool {
        self.flags.contains(RadioFlags::ENABLED)
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        if !self.flags.contains(RadioFlags::ENABLED) {
            return EventResult::Ignored;
        }
        if matches!(event, Event::Pointer(pointer) if pointer.kind == PointerEventKind::Down)
            && self.flags.contains(RadioFlags::FOCUSABLE)
        {
            scope.request_focus();
            self.interaction.set(InteractionState::FOCUSED);
        }
        if let Event::Key(key) = event {
            // A key belongs to the button only while it has the focus, and
            // it writes the group on the *release* the way a button does.
            let chosen = matches!(key.key, Key::Space | Key::Enter);
            if chosen && !key.pressed && scope.is_focused() {
                self.choose();
                return EventResult::Consumed;
            }
            return EventResult::Ignored;
        }
        let interaction = self.interaction.clone();
        let group = self.group.clone();
        let value = self.value.clone();
        let on_change = self.on_change.clone();
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
                        if group.select(&value)
                            && let Some(handler) = &on_change
                        {
                            handler(&value);
                        }
                        EventResult::Consumed
                    }
                    _ => EventResult::Ignored,
                }
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::Composition;
    use crate::event::{KeyEvent, PointerEvent};
    use crate::node::{self, Node};
    use crate::paint::{DrawList, PaintStyle, RectOp};
    use crate::theme::Theme;
    use crate::widgets::{Row, ScrollAxes, ScrollState, Viewport};
    use core::cell::Cell;

    /// The square a one-option harness places its button in.
    const TARGET: Rect = Rect::from_min_size(Offset::ZERO, 40.0, 40.0);

    /// The rectangle a two-option harness places its row in.
    const ROW: Rect = Rect::from_min_size(Offset::ZERO, 80.0, 40.0);

    /// The center of the first option of a [`ROW`] — and of a [`TARGET`].
    const FIRST: Offset = Offset::new(20.0, 20.0);

    /// The center of the second option of a [`ROW`].
    const SECOND: Offset = Offset::new(60.0, 20.0);

    /// A point no option of a [`ROW`] contains.
    const OUTSIDE: Offset = Offset::new(120.0, 20.0);

    /// A tree of radio buttons, with the state dispatching an event needs.
    struct Harness {
        theme: Theme,
        node: Node,
        focus: Option<Vec<usize>>,
    }

    impl Harness {
        /// A tree of `radio` alone, measured and placed in [`TARGET`].
        fn single(radio: Radio<u32>) -> Self {
            Self::of(Node::new(radio), TARGET)
        }

        /// A row of the options 0 and 1 of `group`, placed in [`ROW`].
        ///
        /// The buttons give up the keyboard focus: the tests of this row
        /// read the selection off the canvas, and a click would otherwise
        /// leave a focused state layer in front of the dot they look for.
        fn options(group: RadioGroup<u32>) -> Self {
            let row = Node::new(Row::new())
                .child(Node::new(
                    Radio::new(0, RadioGroup::clone(&group)).focusable(false),
                ))
                .child(Node::new(Radio::new(1, group).focusable(false)));
            Self::of(row, ROW)
        }

        /// The tree of `node`, measured and laid out in `bounds`.
        fn of(mut node: Node, bounds: Rect) -> Self {
            let theme = Theme::light();
            node.measure(&theme, Constraints::unbounded());
            node.place(bounds);
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

        /// Completes a click on the option whose center is `position`.
        fn click(&mut self, position: Offset) -> EventResult {
            self.pointer(PointerEventKind::Down, position);
            self.pointer(PointerEventKind::Up, position)
        }

        /// The tree as it paints itself right now.
        fn canvas(&mut self) -> Canvas {
            let mut list = DrawList::new();
            self.node.draw(&self.theme, &mut list);
            Canvas::of(&list)
        }
    }

    /// The shapes a draw list recorded, split by how they were painted.
    #[derive(Default)]
    struct Canvas {
        /// The filled circles: the dot and the state layer.
        fills: Vec<(Rect, Color)>,
        /// The outlined circles: the ring.
        outlines: Vec<(Rect, Color)>,
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
                }
            }
            canvas
        }
    }

    #[test]
    fn a_radio_measures_to_its_touch_target() {
        let harness = Harness::single(Radio::new(0, RadioGroup::new()));
        assert_eq!(harness.node.size(), Size::new(40.0, 40.0));
        let harness = Harness::options(RadioGroup::new());
        assert_eq!(harness.node.size(), Size::new(80.0, 40.0));
    }

    #[test]
    fn a_click_chooses_the_option_it_stands_for_and_reports_it() {
        let group = RadioGroup::new();
        let seen = Rc::new(Cell::new(None));
        let reported = Rc::clone(&seen);
        let mut harness =
            Harness::single(Radio::new(7, group.clone()).on_change(move |value| reported.set(Some(*value))));
        harness.pointer(PointerEventKind::Down, FIRST);
        assert!(
            harness
                .pointer(PointerEventKind::Up, FIRST)
                .is_consumed(),
            "a press completed inside the target is the button's click"
        );
        assert_eq!(group.current(), Some(7));
        assert_eq!(seen.get(), Some(7));
    }

    #[test]
    fn a_click_moves_the_selection_off_the_option_that_had_it() {
        let group = RadioGroup::new();
        let mut harness = Harness::options(group.clone());
        assert!(harness.click(FIRST).is_consumed());
        assert_eq!(group.current(), Some(0));
        assert!(harness.click(SECOND).is_consumed());
        assert_eq!(group.current(), Some(1));
        assert!(harness.click(FIRST).is_consumed());
        assert_eq!(group.current(), Some(0));
        // Only the chosen option paints a dot, so the tree that reads the
        // group is exclusive on the screen as well as in the state.
        let canvas = harness.canvas();
        assert_eq!(canvas.outlines.len(), 2, "both options keep their ring");
        assert_eq!(
            canvas.fills,
            [(
                Rect::from_min_size(Offset::new(15.0, 15.0), 10.0, 10.0),
                Theme::light().colors.primary
            )]
        );
    }

    #[test]
    fn a_click_on_the_chosen_option_changes_nothing_and_reports_nothing() {
        let group = RadioGroup::new();
        let seen = Rc::new(Cell::new(None));
        let reported = Rc::clone(&seen);
        let mut harness =
            Harness::single(Radio::new(2, group.clone()).on_change(move |value| reported.set(Some(*value))));
        harness.click(FIRST);
        assert_eq!(seen.get(), Some(2));
        let revision = group.revision();
        // The release is still the button's to answer — a click on a chosen
        // option is a click, not a miss — but the selection does not move
        // and the handler does not run for it.
        assert!(harness.click(FIRST).is_consumed());
        assert_eq!(group.revision(), revision);
        assert_eq!(seen.get(), Some(2));
    }

    #[test]
    fn the_press_of_a_radio_is_reported_but_not_taken() {
        let source = InteractionSource::new();
        let mut harness = Harness::single(Radio::new(0, RadioGroup::new()).interaction_with(source.clone()));
        assert!(
            !harness
                .pointer(PointerEventKind::Down, FIRST)
                .is_consumed()
        );
        assert!(source.state().contains(InteractionState::PRESSED));
        // The leave of the pointer ends the press the way it ends the
        // hover: a button held in a target the pointer left is not a click.
        harness.pointer(PointerEventKind::Move, OUTSIDE);
        assert!(!source.state().contains(InteractionState::PRESSED));
        assert!(!source.state().contains(InteractionState::HOVERED));
    }

    #[test]
    fn a_release_outside_the_target_is_not_a_choice() {
        let group = RadioGroup::new();
        let mut harness = Harness::single(Radio::new(0, group.clone()));
        harness.pointer(PointerEventKind::Down, FIRST);
        assert!(
            harness
                .pointer(PointerEventKind::Up, OUTSIDE)
                .is_consumed()
        );
        assert_eq!(group.current(), None);
    }

    #[test]
    fn a_press_that_travelled_never_becomes_a_choice() {
        let group = RadioGroup::new();
        let source = InteractionSource::new();
        let mut harness = Harness::single(Radio::new(0, group.clone()).interaction_with(source.clone()));
        harness.pointer(PointerEventKind::Down, FIRST);
        let travelled = Offset::new(20.0, 90.0);
        assert!(
            !harness
                .pointer(PointerEventKind::Move, travelled)
                .is_consumed(),
            "a drag that started on a button is left to the list that holds it"
        );
        assert!(!source.state().contains(InteractionState::PRESSED));
        harness.pointer(PointerEventKind::Up, travelled);
        assert_eq!(group.current(), None);
    }

    #[test]
    fn a_disabled_radio_answers_nothing() {
        let group = RadioGroup::new();
        let mut harness = Harness::single(Radio::new(4, group.clone()).enabled(false));
        assert!(
            !harness.node.hit_testable(),
            "the press falls through to whatever can still answer it"
        );
        assert!(!harness.click(FIRST).is_consumed());
        assert!(!harness.release(Key::Space).is_consumed());
        assert_eq!(group.current(), None, "a click is not a choice");
    }

    #[test]
    fn a_key_release_chooses_the_focused_radio() {
        let group = RadioGroup::new();
        let mut harness = Harness::single(Radio::new(3, group.clone()));
        // The press takes the focus and the release misses the target, so
        // the button is focused and nothing is chosen yet: the key is the
        // only thing left that can answer.
        harness.pointer(PointerEventKind::Down, FIRST);
        harness.pointer(PointerEventKind::Up, OUTSIDE);
        assert_eq!(group.current(), None);
        assert!(harness.release(Key::Space).is_consumed());
        assert_eq!(group.current(), Some(3));
        assert!(harness.release(Key::Enter).is_consumed());
        // The press of the same key is not the release of it: a key held
        // down repeats, and a button that answered every repeat would
        // flicker under a finger that rests on the bar.
        assert!(
            !harness
                .offer(&Event::Key(KeyEvent::new(Key::Space, true)))
                .is_consumed()
        );
        assert_eq!(group.current(), Some(3));
    }

    #[test]
    fn an_application_write_chooses_an_option_no_press_made() {
        let group = RadioGroup::new();
        let mut harness = Harness::options(group.clone());
        assert!(harness.canvas().fills.is_empty());
        assert!(group.select(&1));
        let canvas = harness.canvas();
        assert_eq!(
            canvas.fills,
            // The dot is 15 pixels from the corner of the target it sits
            // in, and the second option of the row starts 40 pixels along
            // it — the scope paints a widget in the box it was placed in.
            [(
                Rect::from_min_size(Offset::new(55.0, 15.0), 10.0, 10.0),
                Theme::light().colors.primary
            )],
            "the second option holds the dot the write gave it"
        );
    }

    #[test]
    fn an_unchosen_radio_paints_only_its_ring() {
        let mut harness = Harness::single(Radio::new(0, RadioGroup::new()).focusable(false));
        let canvas = harness.canvas();
        assert!(canvas.fills.is_empty(), "no dot and no layer at rest");
        assert_eq!(
            canvas.outlines,
            [(
                Rect::from_min_size(Offset::new(10.0, 10.0), 20.0, 20.0),
                Theme::light().colors.on_surface_variant
            )]
        );
    }

    #[test]
    fn a_chosen_radio_keeps_its_ring_and_fills_its_dot() {
        let group = RadioGroup::with_value(0);
        let mut harness = Harness::single(Radio::new(0, group));
        let canvas = harness.canvas();
        let scheme = Theme::light().colors;
        assert_eq!(
            canvas.outlines,
            [(
                Rect::from_min_size(Offset::new(10.0, 10.0), 20.0, 20.0),
                scheme.primary
            )],
            "the ring of a chosen radio is the color of its dot"
        );
        assert_eq!(
            canvas.fills,
            [(
                Rect::from_min_size(Offset::new(15.0, 15.0), 10.0, 10.0),
                scheme.primary
            )]
        );
    }

    #[test]
    fn a_disabled_radio_paints_the_disabled_roles() {
        let group = RadioGroup::with_value(0);
        let mut harness = Harness::single(Radio::new(0, group).enabled(false));
        let canvas = harness.canvas();
        let disabled = Theme::light()
            .colors
            .on_surface
            .with_alpha(defaults::DISABLED_CONTENT_ALPHA);
        assert_eq!(canvas.outlines[0].1, disabled);
        assert_eq!(canvas.fills[0].1, disabled);
    }

    #[test]
    fn the_state_layer_shades_the_target_of_a_held_radio() {
        let mut harness = Harness::single(Radio::new(0, RadioGroup::new()).focusable(false));
        harness.pointer(PointerEventKind::Down, FIRST);
        let canvas = harness.canvas();
        assert_eq!(canvas.fills[0].0, TARGET, "the layer is the target");
        assert!(
            (canvas.fills[0].1.alpha - defaults::PRESSED_ALPHA).abs() < 0.001,
            "{:?}",
            canvas.fills[0].1
        );
    }

    #[test]
    fn a_drag_from_a_radio_still_scrolls_its_list() {
        let offset = ScrollState::new(ScrollAxes::VERTICAL);
        let group = RadioGroup::new();
        let theme = Theme::light();
        let mut node =
            Node::new(Viewport::new().scroll(offset.clone())).child(Node::new(Radio::new(0, group.clone())));
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
            "the list answers the drag the button left to it"
        );
        assert_eq!(offset.offset().y, 5.0);
        offer(&Event::Pointer(PointerEvent::new(
            PointerEventKind::Up,
            Offset::new(20.0, 5.0),
        )));
        assert_eq!(group.current(), None, "a drag is not a choice");
    }

    #[test]
    fn the_hover_after_a_click_does_not_scroll_the_list() {
        let offset = ScrollState::new(ScrollAxes::VERTICAL);
        let group = RadioGroup::new();
        let theme = Theme::light();
        let mut node =
            Node::new(Viewport::new().scroll(offset.clone())).child(Node::new(Radio::new(0, group.clone())));
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
        assert_eq!(group.current(), Some(0));
        // The button took the release, so the list was told its press ended
        // rather than left holding it: this move is a hover, not a drag.
        offer(&Event::Pointer(PointerEvent::new(
            PointerEventKind::Move,
            Offset::new(20.0, 0.0),
        )));
        assert_eq!(offset.offset(), Offset::ZERO);
    }

    #[test]
    fn a_choice_survives_the_recomposition_that_rebuilt_the_button() {
        let group = RadioGroup::new();
        let kept = group.clone();
        let region = GestureRegion::default();
        let held = region.clone();
        let mut composition = Composition::new(Theme::light());
        composition.set_content(move |_| Node::new(Radio::new(7, kept.clone()).region(held.clone())));
        composition.recompose(true);
        composition.layout(TARGET.size, true);
        let start = Offset::new(20.0, 20.0);
        composition.handle_event(&Event::Pointer(PointerEvent::new(PointerEventKind::Down, start)));
        // Choosing writes the group the tree reads back, so the host
        // recomposes with the finger still down: the button is built again,
        // and only the region the host owns still knows about the press.
        composition.recompose(true);
        composition.layout(TARGET.size, true);
        composition.handle_event(&Event::Pointer(PointerEvent::new(PointerEventKind::Up, start)));
        assert_eq!(group.current(), Some(7));
    }

    #[test]
    fn a_group_holds_no_option_until_one_is_chosen() {
        let group = RadioGroup::<u32>::new();
        assert_eq!(group.current(), None);
        assert!(!group.is_selected(&0));
        assert!(group.select(&0));
        assert!(group.is_selected(&0));
        assert!(!group.is_selected(&1));
        // Repeating a choice is not a write, and so costs no recomposition;
        // giving it up is, and giving it up twice is not.
        let revision = group.revision();
        assert!(!group.select(&0));
        assert_eq!(group.revision(), revision);
        assert!(group.clear());
        assert_eq!(group.current(), None);
        assert!(!group.clear());
    }
}
