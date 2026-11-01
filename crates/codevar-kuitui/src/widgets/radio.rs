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

