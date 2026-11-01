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

//! The slider: a [`MutableState`] of `f32` written by a drag along a track.
//!
//! A slider is the third answer the [`GestureRegion`] gives — after the
//! viewport that scrolls its content and the button that flips a value, the
//! widget that *tracks the pointer*. That is what makes it the hardest of
//! the three: the value is not a state to toggle but a position to follow,
//! so every move of a drag it owns has to be turned into a number, and the
//! claim on those moves has to be settled without stealing the scroll of the
//! list that holds it.
//!
//! # The claim a drag makes
//!
//! A press inside the slider moves the thumb to where the finger landed —
//! the way a track bar answers a tap — and is reported to the interaction
//! source but *not* consumed, so an ancestor viewport takes the same down
//! and can scroll with it. The first drag past the slop is what the two
//! contend over, and the axis that travelled further wins: a slider is a
//! horizontal control, so a finger that came down on the track and went
//! sideways drags the thumb while one that went down the list scrolls it.
//! The decision is sticky for the rest of the press, which is the rule a
//! horizontal drag carries in the platform: once the ancestor has the
//! gesture, a move that happens to turn sideways does not take it back.
//!
//! The claim is state, not a field: writing the value marks the state dirty,
//! the recomposition that follows rebuilds the widget, and a claim kept in a
//! plain field would be lost in the middle of the drag that made it.
//! [`DragClaim`] is therefore a handle over a cell, shared by every copy of
//! the slider, the way the region keeps its recognition shared.
//!
//! # The value and its ticks
//!
//! The thumb holds a value in `from ..= to`, and a slider with `steps` snaps
//! to the ticks that divide that span into `steps + 1` equal parts — the
//! count is the number of marks *between* the ends, which is what the
//! platform's `steps` parameter means and why a snapped value can be one of
//! `steps + 2` of them. Snapping happens on the raw value a position
//! produced, so the thumb never lands between two marks and both ends of the
//! range stay reachable.
//!
//! A position, not a delta, decides the value: the platform computes the
//! thumb from where the finger is rather than from how far the last report
//! said it moved, so a drag that skips pixels still lands the thumb under the
//! finger and no rounding drifts along the way.
//!
//! The two handlers split the way a check box splits its value from its
//! report: [`Slider::on_change`] sees every value the *user* moved the thumb
//! to, one call per move that changed it, while [`Slider::on_finish`] sees
//! the end of the gesture. An application that writes the state itself
//! triggers neither — it changes what the slider paints, which is the same
//! rule Compose draws between `value` and `onValueChange`.
//!
//! # Why the thumb is a bar
//!
//! Material 3 gave the slider a thumb 4 pixels wide and 20 tall rather than
//! the disc of the previous version, and the track a height of 4, so the
//! drawing here reserves 10 pixels at each end — half the height of the thumb
//! — for its center to travel in. The state layer is a circle of 40 pixels
//! over that center, which is why the widget measures only 20 tall and lets
//! the layer overflow it: the platform reserves the height of the thumb, not
//! the height of the halo.

use alloc::rc::Rc;
use core::cell::Cell;
use core::ops::{BitOr, BitOrAssign};

use crate::color::{Color, ColorScheme};
use crate::effect::{InteractionSource, InteractionState};
use crate::event::{Event, EventResult, Key, PointerEventKind};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::paint::CornerRadii;
use crate::state::MutableState;
use crate::widget::{DrawScope, EventScope, MeasureScope, PlaceScope, Widget};
use crate::widgets::{Gesture, GestureRegion, ViewportDispatch};

/// The handler a slider calls with each value the user dragged to.
pub type SliderChange = Rc<dyn Fn(f32)>;

/// The handler a slider calls when the gesture that moved it ended.
pub type SliderFinished = Rc<dyn Fn()>;

/// The geometry and the alphas of the Material 3 slider.
pub mod defaults {
    /// The height of the track.
    pub const TRACK: f32 = 4.0;

    /// The height of the thumb — and the height the slider measures.
    pub const THUMB_HEIGHT: f32 = 20.0;

    /// The width of the thumb.
    ///
    /// Material 3 made the thumb a bar the width of the track rather than
    /// the disc of Material 2, which is why one constant serves both.
    pub const THUMB_WIDTH: f32 = 4.0;

    /// The distance each end of the track keeps for the center of the thumb.
    ///
    /// Half the height of the thumb, so the bar stands fully inside the box
    /// at both ends of the range.
    pub const TRACK_INSET: f32 = 10.0;

    /// The radius of a tick mark.
    pub const TICK_RADIUS: f32 = 2.0;

    /// The diameter of the state layer drawn over the thumb.
    pub const HALO: f32 = 40.0;

    /// The width a slider asks for when nothing bounds it.
    pub const PREFERRED_WIDTH: f32 = 200.0;

    /// The alpha of the state layer while the pointer is over the slider.
    pub const HOVER_ALPHA: f32 = 0.08;

    /// The alpha of the state layer while a drag holds the thumb.
    pub const PRESSED_ALPHA: f32 = 0.10;

    /// The alpha of the state layer while the slider has the keyboard focus.
    pub const FOCUSED_ALPHA: f32 = 0.12;

    /// The alpha of the active track of a slider that answers nothing.
    pub const DISABLED_ACTIVE_ALPHA: f32 = 0.38;

    /// The alpha of the inactive track of a slider that answers nothing.
    pub const DISABLED_INACTIVE_ALPHA: f32 = 0.12;

    /// The alpha of the thumb and the ticks of a disabled slider.
    pub const DISABLED_THUMB_ALPHA: f32 = 0.38;

    /// The share of the range a key moves a continuous slider by.
    pub const KEY_FRACTION: f32 = 0.1;
}

/// What a slider does, as a set of options instead of booleans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SliderFlags(u8);

impl SliderFlags {
    /// The slider only paints the value it is given.
    pub const NONE: SliderFlags = SliderFlags(0);
    /// The slider answers a drag, a tap and a key. Clearing this bit is how
    /// a slider is disabled: the pointer never moves the thumb, the press is
    /// never reported, and the track is drawn with the disabled alphas.
    pub const ENABLED: SliderFlags = SliderFlags(1 << 0);
    /// The slider takes the keyboard focus when a button is pressed inside
    /// it, and reports [`InteractionState::FOCUSED`] to its source.
    pub const FOCUSABLE: SliderFlags = SliderFlags(1 << 1);
    /// An enabled slider that takes the focus: what the platform component
    /// is when the caller leaves every parameter defaulted.
    pub const STANDARD: SliderFlags = SliderFlags(Self::ENABLED.0 | Self::FOCUSABLE.0);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: SliderFlags) -> bool {
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
    pub const fn union(self, other: SliderFlags) -> Self {
        SliderFlags(self.0 | other.0)
    }

    /// The set with the bits of `other` removed.
    #[inline]
    #[must_use]
    pub const fn without(self, other: SliderFlags) -> Self {
        SliderFlags(self.0 & !other.0)
    }
}

impl BitOr for SliderFlags {
    type Output = SliderFlags;

    #[inline]
    fn bitor(self, rhs: SliderFlags) -> SliderFlags {
        SliderFlags(self.0 | rhs.0)
    }
}

impl BitOrAssign for SliderFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: SliderFlags) {
        self.0 |= rhs.0;
    }
}

/// The colors of the track, the thumb, the ticks and the state layer.
///
/// Six colors decided together: the active part of the track and the thumb
/// carry the value, the inactive part and the ticks carry the range, and the
/// two disagree on purpose — a tick the thumb passed is drawn in the color of
/// the content that sits on the active bar, or the marks would vanish under
/// the bar they are supposed to divide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliderColors {
    /// The track from its start to the thumb.
    pub active: Color,
    /// The track from the thumb to its end.
    pub inactive: Color,
    /// The thumb.
    pub thumb: Color,
    /// The ticks the thumb has not reached.
    pub tick: Color,
    /// The ticks the thumb has passed.
    pub active_tick: Color,
    /// The color the state layer is tinted with.
    pub state_layer: Color,
}

impl SliderColors {
    /// The colors of a slider that is — or is not — enabled in `scheme`.
    ///
    /// The colors do not depend on the value: the thumb decides which of the
    /// two track colors each half of the track is drawn in, so writing the
    /// value recomposes the drawing without recomputing the palette.
    #[must_use]
    pub const fn for_enabled(scheme: &ColorScheme, flags: SliderFlags) -> Self {
        if flags.contains(SliderFlags::ENABLED) {
            Self {
                active: scheme.primary,
                inactive: scheme.surface_variant,
                thumb: scheme.primary,
                tick: scheme.on_surface_variant,
                active_tick: scheme.on_primary,
                state_layer: scheme.primary,
            }
        } else {
            let thumb = scheme
                .on_surface_variant
                .with_alpha(defaults::DISABLED_THUMB_ALPHA);
            Self {
                active: scheme
                    .on_surface
                    .with_alpha(defaults::DISABLED_ACTIVE_ALPHA),
                inactive: scheme
                    .surface_variant
                    .with_alpha(defaults::DISABLED_INACTIVE_ALPHA),
                thumb,
                tick: thumb,
                active_tick: thumb,
                state_layer: scheme.on_surface,
            }
        }
    }

    /// The state layer of a slider in `interaction`.
    ///
    /// The layers of Material 3 do not add up: the strongest one that
    /// applies is what shows, so a slider that is focused and hovered is no
    /// darker than a focused one. A slider in no state at all returns a fully
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

/// Who owns the moves of the press a slider is holding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Claim {
    /// No drag has passed the slop yet.
    #[default]
    Undecided,
    /// The slider drags its thumb with the pointer.
    Mine,
    /// The pointer went the ancestor's way, and the ancestor keeps the
    /// gesture for the rest of the press.
    Ancestors,
}

/// The claim of the drag a slider is holding, shared with every copy of it.
///
/// The slider is rebuilt by the recomposition its own drag causes, so the
/// claim cannot be a field of the widget: it is a handle over a cell, the way
/// [`GestureRegion`] keeps its recognition.
#[derive(Clone, Default)]
struct DragClaim(Rc<Cell<Claim>>);

impl DragClaim {
    /// The claim of the drag right now.
    #[inline]
    fn get(&self) -> Claim {
        self.0.get()
    }

    /// The claim the next moves are answered under.
    #[inline]
    fn set(&self, claim: Claim) {
        self.0.set(claim);
    }
}

/// The box a value is computed from, in the coordinates of the window.
///
/// The geometry of the track and the range of the value live together because
/// every conversion the slider makes needs both: a position becomes a
/// fraction of the usable track, and that fraction becomes a value in the
/// range. A copy of it is handed to the gesture closure, which is what keeps
/// the math off the widget the closure cannot borrow.
#[derive(Clone, Copy)]
struct Track {
    /// The window coordinate the track starts at.
    left: f32,
    /// The distance the center of the thumb travels along.
    width: f32,
    /// The value at the start of the range.
    from: f32,
    /// The value at the end of the range.
    to: f32,
    /// The number of ticks between the ends of the range.
    steps: usize,
}

impl Track {
    /// The fraction of the range `x` sits at, with the ends clamped in.
    ///
    /// A track with no room for the thumb to travel has no fraction to
    /// report, so it answers the start of the range.
    #[inline]
    fn fraction_at(&self, x: f32) -> f32 {
        if self.width <= 0.0 {
            return 0.0;
        }
        ((x - self.left - defaults::TRACK_INSET) / self.width).clamp(0.0, 1.0)
    }

    /// The value at `x`, snapped to a tick if the range has them.
    #[inline]
    fn value_at(&self, x: f32) -> f32 {
        self.snapped(self.from + self.fraction_at(x) * (self.to - self.from))
    }

    /// The window coordinate of the center of the thumb holding `value`.
    #[inline]
    fn position_of(&self, value: f32) -> f32 {
        self.left + defaults::TRACK_INSET + self.fraction_of(value) * self.width
    }

    /// The share of the track `value` sits at.
    #[inline]
    fn fraction_of(&self, value: f32) -> f32 {
        let span = self.to - self.from;
        if span == 0.0 {
            0.0
        } else {
            ((value - self.from) / span).clamp(0.0, 1.0)
        }
    }

    /// The distance between two ticks of this range.
    ///
    /// `steps` marks sit between the ends, so the range is cut into
    /// `steps + 1` equal parts.
    #[inline]
    fn tick(&self) -> f32 {
        (self.to - self.from) / (self.steps as f32 + 1.0)
    }

    /// The value `raw` lands on once the ticks have their say.
    ///
    /// The result is clamped into the range, so a value a key pushed past an
    /// end comes back to it rather than leaving the track.
    #[inline]
    fn snapped(&self, raw: f32) -> f32 {
        let raw = raw.clamp(self.from.min(self.to), self.from.max(self.to));
        if self.steps == 0 {
            return raw;
        }
        let step = self.tick();
        if step <= 0.0 {
            return raw;
        }
        (self.from + libm::roundf((raw - self.from) / step) * step)
            .clamp(self.from.min(self.to), self.from.max(self.to))
    }
}

/// A track that holds a value and drags it with the pointer.
///
/// The slider fills the width it is given and is as tall as its thumb, and
/// writes the [`MutableState`] it was handed: a volume bar, a zoom level or a
/// preference all read the same state and repaint from it.
///
/// ```text
/// let volume = ctx.use_state(|| MutableState::new(0.5_f32));
/// Slider::new(volume.clone()).range(0.0, 1.0).steps(4)
///     .on_change(move |value| mixer.set(value))
/// ```
#[derive(Clone)]
pub struct Slider {
    /// The value the thumb shows and a drag writes.
    value: MutableState<f32>,
    /// The value at the start of the track.
    from: f32,
    /// The value at the end of the track.
    to: f32,
    /// The number of ticks between the ends of the range.
    steps: usize,
    /// What the slider answers to.
    flags: SliderFlags,
    /// The region that turns the events of the track into gestures.
    region: GestureRegion,
    /// The interaction states the application observes.
    interaction: InteractionSource,
    /// The handler called with each value the *user* dragged to.
    on_change: Option<SliderChange>,
    /// The handler called when the gesture that dragged ended.
    on_finish: Option<SliderFinished>,
    /// Who owns the moves of the press the slider is holding.
    claim: DragClaim,
    /// The box the track was last placed in, in window coordinates.
    ///
    /// A widget is handed no bounds when an event reaches it, so the slider
    /// records the rectangle it was placed in: the position of the thumb is a
    /// fraction of that box.
    bounds: Rect,
}

impl Slider {
    /// The slider that shows and writes `value` over `0.0 ..= 1.0`.
    #[must_use]
    pub fn new(value: MutableState<f32>) -> Self {
        Self {
            value,
            from: 0.0,
            to: 1.0,
            steps: 0,
            flags: SliderFlags::STANDARD,
            region: GestureRegion::new(ViewportDispatch::default()),
            interaction: InteractionSource::new(),
            on_change: None,
            on_finish: None,
            claim: DragClaim::default(),
            bounds: Rect::ZERO,
        }
    }

    /// The slider over `from ..= to`.
    ///
    /// The range is not reordered: a slider whose end equals its start holds
    /// one value, and every position on it reads that value.
    #[must_use]
    pub const fn range(mut self, from: f32, to: f32) -> Self {
        self.from = from;
        self.to = to;
        self
    }

    /// The slider that snaps to `steps` marks evenly spaced between the ends.
    ///
    /// Zero — the default — makes it continuous: the thumb stops wherever the
    /// finger does.
    #[must_use]
    pub const fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// The slider that answers the gestures its table names.
    ///
    /// A slot of the table observes or overrides that one gesture and keeps
    /// the built-in answer of the others, so a slider that has to log where a
    /// drag ended writes [`on_drop`](ViewportDispatch::on_drop) and still
    /// drags.
    #[must_use]
    pub fn dispatch(mut self, table: ViewportDispatch) -> Self {
        self.region = GestureRegion::new(table);
        self
    }

    /// The slider that reports each value the user dragged to.
    #[must_use]
    pub fn on_change(mut self, handler: impl Fn(f32) + 'static) -> Self {
        self.on_change = Some(Rc::new(handler));
        self
    }

    /// The slider that reports the end of the gesture that dragged.
    ///
    /// A drag writes the value on every move and this once, which is how a
    /// caller commits an expensive change — a seek, a save — without paying
    /// for it on each of the moves.
    #[must_use]
    pub fn on_finish(mut self, handler: impl Fn() + 'static) -> Self {
        self.on_finish = Some(Rc::new(handler));
        self
    }

    /// The slider that reports its interaction states to `source`.
    ///
    /// The source is a handle, so the application keeps observing the same
    /// states across the recompositions that rebuilt the slider — which is how
    /// a row that dims its label while its slider is held reads it.
    #[must_use]
    pub fn interaction_with(mut self, source: InteractionSource) -> Self {
        self.interaction = source;
        self
    }

    /// The slider that answers — or, with `on` cleared, ignores — the pointer
    /// and the keyboard.
    #[must_use]
    pub fn enabled(mut self, on: bool) -> Self {
        self.set(SliderFlags::ENABLED, on);
        self
    }

    /// The slider that takes — or, with `on` cleared, gives up — the keyboard
    /// focus.
    #[must_use]
    pub fn focusable(mut self, on: bool) -> Self {
        self.set(SliderFlags::FOCUSABLE, on);
        self
    }

    /// Turns `bits` on when `on` names them, and off when it does not.
    fn set(&mut self, bits: SliderFlags, on: bool) {
        self.flags = if on {
            self.flags.union(bits)
        } else {
            self.flags.without(bits)
        };
    }

    /// The value the slider shows.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> &MutableState<f32> {
        &self.value
    }

    /// The options the slider was built with.
    #[inline]
    #[must_use]
    pub const fn flags(&self) -> SliderFlags {
        self.flags
    }

    /// The interaction states of the slider.
    #[inline]
    #[must_use]
    pub const fn interaction(&self) -> &InteractionSource {
        &self.interaction
    }

    /// Whether a press is held inside the slider.
    #[inline]
    #[must_use]
    pub fn is_pressed(&self) -> bool {
        self.region.is_pressed()
    }

    /// Whether a drag of this slider owns the moves of the pointer.
    #[inline]
    #[must_use]
    pub fn is_tracking(&self) -> bool {
        self.claim.get() == Claim::Mine
    }

    /// The geometry the value of the slider is read and written from.
    fn track(&self) -> Track {
        Track {
            left: self.bounds.left(),
            width: (self.bounds.size.width - 2.0 * defaults::TRACK_INSET).max(0.0),
            from: self.from,
            to: self.to,
            steps: self.steps,
        }
    }

    /// The share of the range a single key press moves the value by.
    ///
    /// A slider with ticks moves by one of them; a continuous one has no tick
    /// to step on, so it moves by a tenth of its range.
    fn key_step(&self) -> f32 {
        if self.steps == 0 {
            (self.to - self.from) * defaults::KEY_FRACTION
        } else {
            self.track().tick()
        }
    }

    /// Writes `value` and reports it, and says whether it moved.
    fn write(&self, value: f32) -> bool {
        write(&self.value, &self.on_change, value)
    }
}

/// Writes the value and reports it, and says whether it moved.
///
/// A value that did not change is not written and not reported: the
/// recomposition a write costs is only worth paying for a thumb that moved,
/// and a handler that heard a move that changed nothing would be a lie.
/// The gesture closure and the widget both answer through this, so the rule
/// is stated once.
fn write(value: &MutableState<f32>, on_change: &Option<SliderChange>, next: f32) -> bool {
    if value.get() == next {
        return false;
    }
    value.set(next);
    if let Some(handler) = on_change {
        handler(next);
    }
    true
}

impl Widget for Slider {
    /// Reports the height of the thumb and the whole width it is given.
    ///
    /// A slider is a bar: it never asks for less than the space it is in, and
    /// with nothing bounding it it takes the width the platform defaults to.
    /// The state layer is wider and taller than the box and is not clipped by
    /// it, which is why the box is the height of the thumb rather than of the
    /// halo over it.
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let width = if constraints.has_bounded_width() {
            constraints.max_width
        } else {
            defaults::PREFERRED_WIDTH
        };
        constraints.enforce(Size::new(width, defaults::THUMB_HEIGHT))
    }

    /// Records the box the value of the thumb is measured against.
    fn place(&mut self, _scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.bounds = bounds;
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let value = self.value.get();
        let colors = SliderColors::for_enabled(&scope.theme().colors, self.flags);
        let overlay = colors.overlay_for(self.interaction.state());
        let track = self.track();
        let center_y = bounds.size.height / 2.0;
        // Everything below is in the space of the widget: the scope starts at
        // the corner the slider was placed in, so the window coordinate of
        // the thumb is only ever a subtraction away.
        let thumb = track.position_of(value) - bounds.left();
        let radii = CornerRadii::uniform(defaults::TRACK / 2.0);
        // The layer first: it sits under the thumb it announces, and the thumb
        // is drawn over it rather than mixed into it.
        if overlay.alpha > 0.0 {
            let halo = defaults::HALO;
            scope.fill_rounded_rect(
                Rect::from_min_size(Offset::new(thumb - halo / 2.0, center_y - halo / 2.0), halo, halo),
                CornerRadii::uniform(halo / 2.0),
                overlay,
            );
        }
        let rail = Rect::from_min_size(
            Offset::new(defaults::TRACK_INSET, center_y - defaults::TRACK / 2.0),
            track.width,
            defaults::TRACK,
        );
        if colors.inactive.alpha > 0.0 && track.width > 0.0 {
            scope.fill_rounded_rect(rail, radii, colors.inactive);
        }
        let active = (thumb - defaults::TRACK_INSET).max(0.0);
        if active > 0.0 && colors.active.alpha > 0.0 {
            scope.fill_rounded_rect(
                Rect::from_min_size(
                    Offset::new(defaults::TRACK_INSET, rail.origin.y),
                    active,
                    defaults::TRACK,
                ),
                radii,
                colors.active,
            );
        }
        draw_ticks(scope, &track, center_y, value, &colors);
        scope.fill_rounded_rect(
            Rect::from_min_size(
                Offset::new(
                    thumb - defaults::THUMB_WIDTH / 2.0,
                    center_y - defaults::THUMB_HEIGHT / 2.0,
                ),
                defaults::THUMB_WIDTH,
                defaults::THUMB_HEIGHT,
            ),
            CornerRadii::uniform(defaults::THUMB_WIDTH / 2.0),
            colors.thumb,
        );
    }

    /// The slider reads the pointer only while it is enabled: a disabled
    /// slider lets the press fall through to whatever can still answer it,
    /// which is what lets a list scroll over its disabled rows.
    fn hit_testable(&self) -> bool {
        self.flags.contains(SliderFlags::ENABLED)
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        if !self.flags.contains(SliderFlags::ENABLED) {
            return EventResult::Ignored;
        }
        if matches!(event, Event::Pointer(pointer) if pointer.kind == PointerEventKind::Down)
            && self.flags.contains(SliderFlags::FOCUSABLE)
        {
            scope.request_focus();
            self.interaction.set(InteractionState::FOCUSED);
        }
        if let Event::Key(key) = event {
            // A slider answers a key *press*, not a release: an arrow held
            // down repeats, and the thumb of a slider is expected to run with
            // it. Only the horizontal keys are the slider's; the vertical ones
            // belong to whatever scrolls.
            if !key.pressed || !scope.is_focused() {
                return EventResult::Ignored;
            }
            let step = self.key_step();
            let value = self.value.get();
            let next = match key.key {
                Key::ArrowLeft => value - step,
                Key::ArrowRight => value + step,
                Key::Home => self.from,
                Key::End => self.to,
                _ => return EventResult::Ignored,
            };
            self.write(self.track().snapped(next));
            return EventResult::Consumed;
        }
        // The gesture of a drag is a delta, but the thumb of a slider is a
        // position, so the value is read off where the pointer is.
        let position = match event {
            Event::Pointer(pointer) => Some(pointer.position),
            Event::Scroll(wheel) => Some(wheel.position),
            _ => None,
        };
        let interaction = self.interaction.clone();
        let value = self.value.clone();
        let on_change = self.on_change.clone();
        let on_finish = self.on_finish.clone();
        let claim = self.claim.clone();
        let track = self.track();
        self.region
            .on_event(event, self.bounds, move |gesture| {
                match gesture {
                    Gesture::Press { position } => {
                        interaction.set(InteractionState::PRESSED);
                        // A tap on the track moves the thumb to the tap, and the
                        // press is reported rather than taken: an ancestor has to
                        // see the same down to scroll with it.
                        write(&value, &on_change, track.value_at(position.x));
                        EventResult::Ignored
                    }
                    Gesture::Drag { delta } => {
                        let Some(current) = position else {
                            return EventResult::Ignored;
                        };
                        if claim.get() == Claim::Undecided {
                            if delta.x.abs() > delta.y.abs() {
                                claim.set(Claim::Mine);
                            } else {
                                // The pointer went the ancestor's way, and it keeps
                                // the gesture for the rest of this press: the thumb
                                // gives up the press it was holding.
                                claim.set(Claim::Ancestors);
                                interaction.clear(InteractionState::PRESSED);
                                return EventResult::Ignored;
                            }
                        }
                        if claim.get() != Claim::Mine {
                            return EventResult::Ignored;
                        }
                        write(&value, &on_change, track.value_at(current.x));
                        EventResult::Consumed
                    }
                    Gesture::Hover { inside: true } => {
                        interaction.set(InteractionState::HOVERED);
                        EventResult::Ignored
                    }
                    Gesture::Hover { .. } => {
                        interaction.clear(InteractionState::HOVERED | InteractionState::PRESSED);
                        claim.set(Claim::Undecided);
                        EventResult::Ignored
                    }
                    Gesture::Cancel => {
                        // The press is over without a gesture that ended: the
                        // thumb keeps the value the press gave it and the
                        // caller is told the changing is done, the way the
                        // platform reports the end of a cancelled drag.
                        end(&interaction, &claim, &on_finish);
                        EventResult::Ignored
                    }
                    Gesture::Click { position } => {
                        // The release settles the thumb where the finger left it,
                        // then reports the end of the gesture.
                        write(&value, &on_change, track.value_at(position.x));
                        end(&interaction, &claim, &on_finish);
                        EventResult::Consumed
                    }
                    Gesture::Drop { .. } => {
                        if claim.get() == Claim::Ancestors {
                            // The drag belonged to an ancestor, so the release
                            // that ended it is its to answer; the slider only
                            // gives up the decision it made.
                            claim.set(Claim::Undecided);
                            return EventResult::Ignored;
                        }
                        end(&interaction, &claim, &on_finish);
                        EventResult::Consumed
                    }
                    _ => EventResult::Ignored,
                }
            })
    }
}

/// Lets go of the press and reports that the gesture that dragged has ended.
///
/// The claim is given up with the press, so the next drag of the next press
/// is free to be decided again.
fn end(interaction: &InteractionSource, claim: &DragClaim, on_finish: &Option<SliderFinished>) {
    interaction.clear(InteractionState::PRESSED);
    claim.set(Claim::Undecided);
    if let Some(handler) = on_finish {
        handler();
    }
}

/// Draws the `steps` marks between the ends of the range.
///
/// The marks are painted over both halves of the track, after the bar: the
/// ones the thumb passed take the color of the content that sits on that bar,
/// or they would vanish under it and the active half would read as one solid
/// run instead of the steps it is made of.
fn draw_ticks(scope: &mut DrawScope<'_>, track: &Track, center_y: f32, value: f32, colors: &SliderColors) {
    if track.steps == 0 || track.width <= 0.0 {
        return;
    }
    let thumb = track.position_of(value) - track.left;
    let side = defaults::TICK_RADIUS * 2.0;
    let radii = CornerRadii::uniform(defaults::TICK_RADIUS);
    for index in 1..=track.steps {
        let x = defaults::TRACK_INSET + track.width * index as f32 / (track.steps as f32 + 1.0);
        let mark = Rect::from_min_size(
            Offset::new(x - defaults::TICK_RADIUS, center_y - defaults::TICK_RADIUS),
            side,
            side,
        );
        let passed = thumb >= x;
        let color = if passed { colors.active_tick } else { colors.tick };
        if color.alpha > 0.0 {
            scope.fill_rounded_rect(mark, radii, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{KeyEvent, PointerEvent};
    use crate::node::{self, Node};
    use crate::paint::{DrawList, PaintStyle, RectOp};
    use crate::theme::Theme;
    use crate::widgets::{ScrollAxes, ScrollState, Viewport};
    use core::cell::Cell;

    /// The box a test slider is placed in, in window coordinates.
    const BOX: Rect = Rect::from_min_size(Offset::ZERO, 100.0, 20.0);

    /// The point a tap on the middle of the track answers.
    const MIDDLE: Offset = Offset::new(50.0, 10.0);

    /// The center of the thumb at the start of the range.
    const START: Offset = Offset::new(10.0, 10.0);

    /// The center of the thumb at the end of the range.
    const END: Offset = Offset::new(90.0, 10.0);

    /// A point no track of a [`BOX`] contains.
    const OUTSIDE: Offset = Offset::new(150.0, 10.0);

    /// A tree of one slider, with the state dispatching an event needs.
    struct Harness {
        theme: Theme,
        node: Node,
        focus: Option<Vec<usize>>,
    }

    impl Harness {
        /// A tree of `slider`, measured and placed in [`BOX`].
        fn new(slider: Slider) -> Self {
            let theme = Theme::light();
            let mut node = Node::new(slider);
            node.measure(&theme, Constraints::tight(Size::new(100.0, 20.0)));
            node.place(BOX);
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

        /// Offers the press of `key` to the tree.
        fn press(&mut self, key: Key) -> EventResult {
            self.offer(&Event::Key(KeyEvent::new(key, true)))
        }

        /// The slider as it paints itself right now.
        fn canvas(&mut self) -> Canvas {
            let mut list = DrawList::new();
            self.node.draw(&self.theme, &mut list);
            Canvas::of(&list)
        }
    }

    /// The filled shapes a draw list recorded, in the order it recorded them.
    #[derive(Default)]
    struct Canvas {
        /// The fills: the rail, the active bar, the ticks, the thumb, the layer.
        fills: Vec<(Rect, Color)>,
    }

    impl Canvas {
        /// The recorded fills of `list`.
        fn of(list: &DrawList) -> Self {
            let mut canvas = Self::default();
            for op in list.ops() {
                let any: &dyn core::any::Any = op;
                if let Some(rect) = any.downcast_ref::<RectOp>()
                    && matches!(rect.paint.style, PaintStyle::Fill)
                {
                    canvas.fills.push((rect.rect, rect.paint.color));
                }
            }
            canvas
        }
    }

    /// A slider, the value state it writes and the number of reports it made.
    struct Counted {
        value: MutableState<f32>,
        changes: Rc<Cell<usize>>,
        finishes: Rc<Cell<usize>>,
    }

    impl Counted {
        /// A continuous slider over `0.0 ..= 1.0` counting what it reports.
        fn slider(&self) -> Slider {
            let changes = Rc::clone(&self.changes);
            let finishes = Rc::clone(&self.finishes);
            Slider::new(self.value.clone())
                .on_change(move |_| changes.set(changes.get() + 1))
                .on_finish(move || finishes.set(finishes.get() + 1))
        }
    }

    fn counted(start: f32) -> Counted {
        Counted {
            value: MutableState::new(start),
            changes: Rc::new(Cell::new(0)),
            finishes: Rc::new(Cell::new(0)),
        }
    }

    #[test]
    fn a_slider_takes_the_width_it_is_given_and_the_height_of_its_thumb() {
        let theme = Theme::light();
        let mut node = Node::new(Slider::new(MutableState::new(0.0)));
        assert_eq!(
            node.measure(&theme, Constraints::new(0.0, 120.0, 0.0, f32::INFINITY)),
            Size::new(120.0, 20.0)
        );
        assert_eq!(
            node.measure(&theme, Constraints::unbounded()),
            Size::new(defaults::PREFERRED_WIDTH, defaults::THUMB_HEIGHT)
        );
    }

    #[test]
    fn a_tap_on_the_track_moves_the_thumb_to_the_tap() {
        let counted = counted(0.0);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider());
        harness.pointer(PointerEventKind::Down, MIDDLE);
        assert_eq!(value.get(), 0.5, "the thumb lands under the finger");
        assert!(
            harness
                .pointer(PointerEventKind::Up, MIDDLE)
                .is_consumed()
        );
        assert_eq!(counted.changes.get(), 1, "the release settled on the same value");
        assert_eq!(
            counted.finishes.get(),
            1,
            "the tap is a gesture that ends, and ends once"
        );
    }

    #[test]
    fn a_drag_moves_the_thumb_to_where_the_pointer_is() {
        let counted = counted(0.0);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider());
        harness.pointer(PointerEventKind::Down, START);
        assert_eq!(value.get(), 0.0);
        let dragged = Offset::new(50.0, 10.0);
        assert!(
            harness
                .pointer(PointerEventKind::Move, dragged)
                .is_consumed(),
            "a drag the slider owns stops there: the ancestor does not scroll with it"
        );
        assert_eq!(value.get(), 0.5);
        assert!(
            harness
                .pointer(PointerEventKind::Move, END)
                .is_consumed()
        );
        assert_eq!(value.get(), 1.0);
        assert_eq!(counted.changes.get(), 2);
        assert!(
            harness
                .pointer(PointerEventKind::Up, END)
                .is_consumed()
        );
        assert_eq!(counted.finishes.get(), 1);
    }

    #[test]
    fn a_pointer_past_the_end_of_the_track_lands_the_thumb_on_the_end() {
        let counted = counted(0.5);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider());
        harness.pointer(PointerEventKind::Down, MIDDLE);
        harness.pointer(PointerEventKind::Move, Offset::new(300.0, 10.0));
        assert_eq!(value.get(), 1.0);
        harness.pointer(PointerEventKind::Up, Offset::new(300.0, 10.0));
        harness.pointer(PointerEventKind::Down, MIDDLE);
        harness.pointer(PointerEventKind::Move, Offset::new(-80.0, 10.0));
        assert_eq!(value.get(), 0.0);
    }

    #[test]
    fn the_press_of_a_slider_is_reported_but_not_taken() {
        let source = InteractionSource::new();
        let value = MutableState::new(0.0);
        let mut harness = Harness::new(Slider::new(value.clone()).interaction_with(source.clone()));
        assert!(
            !harness
                .pointer(PointerEventKind::Down, MIDDLE)
                .is_consumed(),
            "an ancestor has to see the down to be able to scroll with it"
        );
        assert!(source.state().contains(InteractionState::PRESSED));
        // A move of the axis the track does not run along is left to the
        // ancestor, and the press the slider held goes with it.
        assert!(
            !harness
                .pointer(PointerEventKind::Move, Offset::new(50.0, 2.0))
                .is_consumed(),
            "the vertical move is the list's, not the track's"
        );
        assert!(!source.state().contains(InteractionState::PRESSED));
        assert!(!source.state().contains(InteractionState::HOVERED));
        assert_eq!(value.get(), 0.5, "the press wrote before the drag left");
    }

    #[test]
    fn a_release_outside_the_track_keeps_the_value_the_press_took() {
        let counted = counted(0.0);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider().focusable(false));
        harness.pointer(PointerEventKind::Down, MIDDLE);
        assert!(
            harness
                .pointer(PointerEventKind::Up, OUTSIDE)
                .is_consumed()
        );
        assert_eq!(value.get(), 0.5, "the thumb of a slider stays where it was left");
        assert_eq!(counted.finishes.get(), 1);
        // The press is gone with the gesture: nothing shades the thumb now.
        assert_eq!(
            harness.canvas().fills,
            [
                rail(Theme::light().colors.surface_variant),
                active(40.0, Theme::light().colors.primary),
                thumb(50.0, Theme::light().colors.primary),
            ]
        );
    }

    #[test]
    fn a_drag_that_went_down_the_list_is_left_to_the_list() {
        let offset = ScrollState::new(ScrollAxes::VERTICAL);
        let counted = counted(0.5);
        let value = counted.value.clone();
        let theme = Theme::light();
        // The box of the viewport is shorter than the slider it holds, so
        // there is a range to scroll and the list has somewhere to go.
        let mut node = Node::new(Viewport::new().scroll(offset.clone())).child(Node::new(counted.slider()));
        node.measure(&theme, Constraints::tight(Size::new(100.0, 10.0)));
        node.place(Rect::from_min_size(Offset::ZERO, 100.0, 10.0));
        let mut focus = None;
        let mut offer = |event: &Event| {
            let mut path = Vec::new();
            node::dispatch_event(&mut node, &theme, &mut path, &mut focus, event)
        };
        let at =
            |x: f32, y: f32| Event::Pointer(PointerEvent::new(PointerEventKind::Move, Offset::new(x, y)));
        offer(&Event::Pointer(PointerEvent::new(
            PointerEventKind::Down,
            Offset::new(50.0, 5.0),
        )));
        assert!(
            offer(&at(50.0, 0.0)).is_consumed(),
            "the list answers the drag the slider gave up"
        );
        assert_eq!(offset.offset().y, 5.0);
        assert_eq!(value.get(), 0.5);
        // The decision is sticky: a move that turns sideways afterwards does
        // not take the gesture back from the list that has it.
        assert!(!offer(&at(80.0, 0.0)).is_consumed());
        assert_eq!(value.get(), 0.5);
        offer(&Event::Pointer(PointerEvent::new(
            PointerEventKind::Up,
            Offset::new(80.0, 0.0),
        )));
        assert_eq!(counted.changes.get(), 0);
        assert_eq!(
            counted.finishes.get(),
            0,
            "the gesture that ended belonged to the list, not to the track"
        );
    }

    #[test]
    fn a_slider_with_ticks_snaps_the_thumb_to_a_mark() {
        let counted = counted(0.0);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider().steps(4));
        // A quarter of the way between the first and the second mark, and the
        // nearer mark takes the thumb.
        harness.pointer(PointerEventKind::Down, Offset::new(35.0, 10.0));
        assert_eq!(value.get(), 0.4);
        // The ends of the range stay reachable through the marks.
        harness.pointer(PointerEventKind::Up, Offset::new(35.0, 10.0));
        harness.pointer(PointerEventKind::Down, END);
        assert_eq!(value.get(), 1.0);
    }

    #[test]
    fn a_key_press_steps_the_focused_slider() {
        let counted = counted(0.0);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider().steps(4));
        // The press is what takes the focus, so the keyboard answers the
        // slider the pointer last touched rather than the root of the tree.
        harness.pointer(PointerEventKind::Down, START);
        harness.pointer(PointerEventKind::Up, START);
        assert_eq!(value.get(), 0.0);
        assert!(harness.press(Key::ArrowRight).is_consumed());
        assert_eq!(value.get(), 0.2);
        assert!(harness.press(Key::ArrowRight).is_consumed());
        assert_eq!(value.get(), 0.4);
        assert!(harness.press(Key::ArrowLeft).is_consumed());
        assert_eq!(value.get(), 0.2);
        // A key held down repeats, and the thumb runs with it: the release of
        // the same key is not a second step.
        assert!(
            !harness
                .offer(&Event::Key(KeyEvent::new(Key::ArrowRight, false)))
                .is_consumed()
        );
        assert_eq!(value.get(), 0.2);
        assert!(harness.press(Key::End).is_consumed());
        assert_eq!(value.get(), 1.0);
        assert!(harness.press(Key::Home).is_consumed());
        assert_eq!(value.get(), 0.0);
        assert_eq!(
            counted.changes.get(),
            5,
            "the press that changed nothing is not counted, the five keys that moved the thumb are"
        );
    }

    #[test]
    fn a_slider_answers_only_the_keys_of_its_own_axis_and_only_when_it_is_focused() {
        let counted = counted(0.5);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider().focusable(false));
        // Without the focus the slider is not the keyboard's to move.
        harness.pointer(PointerEventKind::Down, MIDDLE);
        assert!(!harness.press(Key::ArrowRight).is_consumed());
        assert_eq!(value.get(), 0.5);
        // The vertical keys belong to whatever scrolls.
        let mut harness = Harness::new(Slider::new(value));
        harness.pointer(PointerEventKind::Down, MIDDLE);
        assert!(!harness.press(Key::ArrowDown).is_consumed());
        assert!(!harness.press(Key::Tab).is_consumed());
    }

    #[test]
    fn a_disabled_slider_answers_nothing() {
        let counted = counted(0.5);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider().enabled(false));
        assert!(
            !harness.node.hit_testable(),
            "the press falls through to whatever can still answer it"
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Down, START)
                .is_consumed()
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Move, END)
                .is_consumed()
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Up, END)
                .is_consumed()
        );
        assert!(!harness.press(Key::End).is_consumed());
        assert_eq!(value.get(), 0.5);
        assert_eq!(counted.changes.get(), 0);
    }

    #[test]
    fn an_application_write_moves_the_thumb_without_a_report() {
        let counted = counted(0.0);
        let value = counted.value.clone();
        let mut harness = Harness::new(counted.slider());
        value.set(0.5);
        assert_eq!(
            harness.canvas().fills,
            [
                rail(Theme::light().colors.surface_variant),
                active(40.0, Theme::light().colors.primary),
                thumb(50.0, Theme::light().colors.primary),
            ],
            "the thumb of a written value is the thumb of a dragged one"
        );
        assert_eq!(counted.changes.get(), 0);
    }

    #[test]
    fn the_track_is_painted_in_two_colors_up_to_the_thumb() {
        let mut harness = Harness::new(Slider::new(MutableState::new(0.25)));
        let scheme = Theme::light().colors;
        assert_eq!(
            harness.canvas().fills,
            [
                rail(scheme.surface_variant),
                active(20.0, scheme.primary),
                thumb(30.0, scheme.primary),
            ]
        );
        // A thumb at the start of the range has no active track at all: the
        // bar is drawn only once it has something to cover.
        let mut harness = Harness::new(Slider::new(MutableState::new(0.0)));
        assert_eq!(
            harness.canvas().fills,
            [rail(scheme.surface_variant), thumb(10.0, scheme.primary)]
        );
    }

    #[test]
    fn the_ticks_are_painted_over_both_halves_of_the_track() {
        let mut harness = Harness::new(Slider::new(MutableState::new(0.5)).steps(3));
        let scheme = Theme::light().colors;
        assert_eq!(
            harness.canvas().fills,
            [
                rail(scheme.surface_variant),
                active(40.0, scheme.primary),
                // The marks sit at a quarter, a half and three quarters of the
                // distance the thumb travels; the two it passed are drawn in
                // the color of the content over the active bar.
                mark(30.0, scheme.on_primary),
                mark(50.0, scheme.on_primary),
                mark(70.0, scheme.on_surface_variant),
                thumb(50.0, scheme.primary),
            ]
        );
    }

    #[test]
    fn the_halo_shades_the_thumb_of_a_held_slider() {
        let mut harness = Harness::new(Slider::new(MutableState::new(0.5)).focusable(false));
        harness.pointer(PointerEventKind::Down, MIDDLE);
        let canvas = harness.canvas();
        let halo = defaults::HALO;
        assert_eq!(
            canvas.fills[0].0,
            Rect::from_min_size(Offset::new(30.0, 10.0 - halo / 2.0), halo, halo),
            "the layer is a circle over the center of the thumb"
        );
        assert!(
            (canvas.fills[0].1.alpha - defaults::PRESSED_ALPHA).abs() < 0.001,
            "{:?}",
            canvas.fills[0].1
        );
    }

    #[test]
    fn a_disabled_slider_paints_the_disabled_alphas() {
        let mut harness = Harness::new(
            Slider::new(MutableState::new(0.5))
                .steps(3)
                .enabled(false),
        );
        let scheme = Theme::light().colors;
        assert_eq!(
            harness.canvas().fills,
            [
                rail(
                    scheme
                        .surface_variant
                        .with_alpha(defaults::DISABLED_INACTIVE_ALPHA)
                ),
                active(
                    40.0,
                    scheme
                        .on_surface
                        .with_alpha(defaults::DISABLED_ACTIVE_ALPHA)
                ),
                mark(
                    30.0,
                    scheme
                        .on_surface_variant
                        .with_alpha(defaults::DISABLED_THUMB_ALPHA)
                ),
                mark(
                    50.0,
                    scheme
                        .on_surface_variant
                        .with_alpha(defaults::DISABLED_THUMB_ALPHA)
                ),
                mark(
                    70.0,
                    scheme
                        .on_surface_variant
                        .with_alpha(defaults::DISABLED_THUMB_ALPHA)
                ),
                thumb(
                    50.0,
                    scheme
                        .on_surface_variant
                        .with_alpha(defaults::DISABLED_THUMB_ALPHA)
                ),
            ]
        );
    }

    /// The inactive rail of a [`BOX`], in the space of the widget.
    fn rail(color: Color) -> (Rect, Color) {
        (
            Rect::from_min_size(
                Offset::new(defaults::TRACK_INSET, 10.0 - defaults::TRACK / 2.0),
                80.0,
                defaults::TRACK,
            ),
            color,
        )
    }

    /// The active bar of a [`BOX`] `width` pixels along the rail.
    fn active(width: f32, color: Color) -> (Rect, Color) {
        (
            Rect::from_min_size(
                Offset::new(defaults::TRACK_INSET, 10.0 - defaults::TRACK / 2.0),
                width,
                defaults::TRACK,
            ),
            color,
        )
    }

    /// The thumb of a [`BOX`] centered at local `x`.
    fn thumb(x: f32, color: Color) -> (Rect, Color) {
        (
            Rect::from_min_size(
                Offset::new(x - defaults::THUMB_WIDTH / 2.0, 0.0),
                defaults::THUMB_WIDTH,
                defaults::THUMB_HEIGHT,
            ),
            color,
        )
    }

    /// The tick mark of a [`BOX`] centered at local `x`.
    fn mark(x: f32, color: Color) -> (Rect, Color) {
        (
            Rect::from_min_size(
                Offset::new(x - defaults::TICK_RADIUS, 10.0 - defaults::TICK_RADIUS),
                defaults::TICK_RADIUS * 2.0,
                defaults::TICK_RADIUS * 2.0,
            ),
            color,
        )
    }
}
