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

//! The pieces a select is built from: its options, the mode of its list,
//! the flags of its behavior and the colors of both halves.
//!
//! The types here decide the numbers and the roles of the widget without
//! answering a pointer: the geometry is the Material 3 list item the
//! platform component is specified with, the state layer is the strongest
//! interaction alpha rather than the sum of them, and the disabled select
//! loses the color of its roles rather than the shape of them. Keeping
//! those decisions apart from the widget is what lets a select and the row
//! it paints agree on what a color means.
//!
//! The widget is the fourth built on the [`GestureRegion`], and the first
//! that needs two of them: the field answers a press the way a check box
//! does, and the list answers the pointer the way a [`Viewport`] does,
//! moving the same [`ScrollState`] by the same leftover delta — the slop,
//! the velocity window and the fling decay are the platform's, so a list
//! that re-derived them would scroll differently from its own page. The
//! framework has no popup layer, so the list is part of the layout: while it
//! is open the widget measures itself as the field *plus* the list, which
//! pushes its neighbours down rather than covering them.
//!
//! [`Viewport`]: crate::widgets::Viewport

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::ops::{BitOr, BitOrAssign, Range};

use codevar_kuitui_control::Timestamp;

use crate::color::{Color, ColorScheme};
use crate::effect::{InteractionSource, InteractionState};
use crate::event::{Event, EventResult, Key, PointerEventKind};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::paint::CornerRadii;
use crate::state::MutableState;
use crate::theme::Theme;
use crate::widget::{DrawScope, EventScope, MeasureScope, PlaceScope, Widget};
use crate::widgets::{Gesture, GestureRegion, ScrollAxes, ScrollState, ViewportDispatch};

use super::viewport::scrolled;

/// The handler a select calls with the position of the option the user
/// chose.
pub type SelectChange = Rc<dyn Fn(usize)>;

/// The geometry and the alphas of the Material 3 select.
///
/// The field and the list are both one-line list items — the platform
/// gives a single-line item a container of 56 logical pixels with 16 of
/// padding on each side — and the menu is cut with the extra-small corner
/// its container shape names. The dropdown icon is the 24-pixel box the
/// trailing icon slot reserves.
pub mod defaults {
    use crate::geometry::Offset;

    /// The height of the field, and of one row of the list.
    pub const FIELD_HEIGHT: f32 = 56.0;

    /// The height of one row of the list.
    pub const ITEM_HEIGHT: f32 = 56.0;

    /// The space between an edge of the widget and the text it draws.
    pub const PADDING: f32 = 16.0;

    /// The side of the box the dropdown icon is drawn in.
    pub const ICON_BOX: f32 = 24.0;

    /// The radius of the corners of the field and of the list.
    pub const CORNER: f32 = 4.0;

    /// The width of the outline of a field that is not focused.
    pub const OUTLINE: f32 = 1.0;

    /// The width of the outline of the field that has the focus.
    pub const FOCUSED_OUTLINE: f32 = 2.0;

    /// The number of rows measured to decide the width of the widget.
    pub const MAX_ITEMS_MEASURED: usize = 15;

    /// The alpha of the state layer while the pointer is over the widget.
    pub const HOVER_ALPHA: f32 = 0.08;

    /// The alpha of the state layer while a button is held in it.
    pub const PRESSED_ALPHA: f32 = 0.10;

    /// The alpha of the state layer while the field has the keyboard focus.
    pub const FOCUSED_ALPHA: f32 = 0.12;

    /// The alpha of the text and the icon of a select that answers nothing.
    pub const DISABLED_CONTENT_ALPHA: f32 = 0.38;

    /// The alpha of the outline and the container of a select that answers
    /// nothing.
    pub const DISABLED_OUTLINE_ALPHA: f32 = 0.12;

    /// The vertices of the dropdown icon, as fractions of its box.
    ///
    /// The three points are the outline of the platform's arrow: a stroke
    /// along the two segments that meet at the bottom is the icon, drawn
    /// the way the check mark of a box is drawn because the framework has
    /// no polyline primitive.
    pub const CARET: [Offset; 3] = [
        Offset::new(0.2917, 0.4167),
        Offset::new(0.5, 0.625),
        Offset::new(0.7083, 0.4167),
    ];

    /// The width of the strokes of the dropdown icon.
    pub const CARET_STROKE: f32 = 2.0;
}

/// Whether the list is open, and which option it points at while it is.
///
/// The pointed at option is part of the mode rather than a state of its
/// own because only the open list has one: a closed select has nowhere to
/// show it, and a write that left it behind would be a value no draw could
/// read. It is the option the keys last moved, or the row a press fell on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectMode {
    /// The list is closed and the field shows the chosen option.
    Closed,
    /// The list is open and showing its options.
    Open {
        /// The option the keys or the last press point at.
        highlight: usize,
    },
}

impl SelectMode {
    /// Whether the list is open.
    #[inline]
    #[must_use]
    pub const fn is_open(self) -> bool {
        matches!(self, Self::Open { .. })
    }

    /// The option the mode points at, if it points at one.
    #[inline]
    #[must_use]
    pub const fn highlight(self) -> Option<usize> {
        match self {
            Self::Closed => None,
            Self::Open { highlight } => Some(highlight),
        }
    }
}

/// What a select does, as a set of options instead of booleans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SelectFlags(u8);

impl SelectFlags {
    /// The select only paints the option it is given.
    pub const NONE: SelectFlags = SelectFlags(0);
    /// The select answers a click and a key. Clearing this bit is how a
    /// select is disabled: the press is never reported, the list never
    /// opens, and the content is drawn with the disabled alphas.
    pub const ENABLED: SelectFlags = SelectFlags(1 << 0);
    /// The field takes the keyboard focus when a button is pressed inside
    /// it, and reports [`InteractionState::FOCUSED`] to its source. The
    /// keys of an open list are answered only while the field has it.
    pub const FOCUSABLE: SelectFlags = SelectFlags(1 << 1);
    /// An enabled field that takes the focus: what the platform component
    /// is when the caller leaves every parameter defaulted.
    pub const STANDARD: SelectFlags = SelectFlags(Self::ENABLED.0 | Self::FOCUSABLE.0);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: SelectFlags) -> bool {
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
    pub const fn union(self, other: SelectFlags) -> Self {
        SelectFlags(self.0 | other.0)
    }

    /// The set with the bits of `other` removed.
    #[inline]
    #[must_use]
    pub const fn without(self, other: SelectFlags) -> Self {
        SelectFlags(self.0 & !other.0)
    }
}

impl BitOr for SelectFlags {
    type Output = SelectFlags;

    #[inline]
    fn bitor(self, rhs: SelectFlags) -> SelectFlags {
        SelectFlags(self.0 | rhs.0)
    }
}

impl BitOrAssign for SelectFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: SelectFlags) {
        self.0 |= rhs.0;
    }
}

/// The colors of the field, the list, their content and the state layer.
///
/// Eight colors decided together, because the roles the platform names are
/// not all in the scheme this framework carries: the menu container is the
/// tonal surface step the scheme has, and the chosen row takes the
/// container-and-content pair it does have rather than the secondary pair
/// it lacks. Returning the eight as one value is what stops the drawing
/// code from re-deriving the substitutions — and from disagreeing with
/// itself about which one a disabled select uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectColors {
    /// The text of the field and of a row that is not chosen.
    pub label: Color,
    /// The placeholder of the field and the dropdown icon.
    pub muted: Color,
    /// The outline of a field that is not focused.
    pub outline: Color,
    /// The outline of the field that has the focus.
    pub focus: Color,
    /// The container of the list.
    pub container: Color,
    /// The container of the chosen row.
    pub selected: Color,
    /// The text of the chosen row.
    pub on_selected: Color,
    /// The color the state layer is tinted with.
    pub state_layer: Color,
}

impl SelectColors {
    /// The colors of a select in `scheme` under `flags`.
    ///
    /// A disabled select loses the color of its roles rather than the shape
    /// of them: the outline still shows where the field ends, the chosen
    /// row still shows which option stands, and both are drawn at the alpha
    /// the platform gives the content of a component that answers nothing.
    #[must_use]
    pub const fn for_scheme(scheme: &ColorScheme, flags: SelectFlags) -> Self {
        if flags.contains(SelectFlags::ENABLED) {
            Self {
                label: scheme.on_surface,
                muted: scheme.on_surface_variant,
                outline: scheme.outline,
                focus: scheme.on_surface,
                container: scheme.surface_variant,
                selected: scheme.primary_container,
                on_selected: scheme.on_primary_container,
                state_layer: scheme.on_surface,
            }
        } else {
            let content = scheme
                .on_surface
                .with_alpha(defaults::DISABLED_CONTENT_ALPHA);
            let edge = scheme
                .on_surface
                .with_alpha(defaults::DISABLED_OUTLINE_ALPHA);
            Self {
                label: content,
                muted: content,
                outline: edge,
                focus: edge,
                container: edge,
                selected: edge,
                on_selected: content,
                state_layer: scheme.on_surface,
            }
        }
    }

    /// The state layer of the field in `interaction`.
    ///
    /// The layers of Material 3 do not add up: the strongest one that
    /// applies is what shows, so a field that is focused and hovered is no
    /// darker than a focused one. A field in no state at all returns a
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

    /// The layer of the row the list points at.
    ///
    /// The pointed at row is tinted with the pressed alpha while a button
    /// is held inside the list and with the hover alpha otherwise, which is
    /// the difference between the row a finger is on and the row the keys
    /// are on.
    #[must_use]
    pub const fn row_overlay_for(&self, pressed: bool) -> Color {
        let alpha = if pressed {
            defaults::PRESSED_ALPHA
        } else {
            defaults::HOVER_ALPHA
        };
        self.state_layer.with_alpha(alpha)
    }
}

/// The options of a select, shared by handle.
///
/// The labels are behind an [`Rc`] because a select is rebuilt by every
/// recomposition of the tree that holds it: they are the input of the
/// widget and not its state, so the handle is what keeps the copy of them
/// to one across the rebuilds rather than a fresh allocation per frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectOptions {
    /// The label of each option, in the order the list shows them.
    labels: Rc<Vec<Rc<str>>>,
}

impl SelectOptions {
    /// The options labelled by `labels`, in that order.
    #[must_use]
    pub fn new(labels: &[&str]) -> Self {
        Self {
            labels: Rc::new(
                labels
                    .iter()
                    .map(|label| Rc::from(*label))
                    .collect(),
            ),
        }
    }

    /// The number of options.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether there is no option to choose.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// The label of the option at `index`, if there is one.
    #[must_use]
    pub fn label(&self, index: usize) -> Option<Rc<str>> {
        self.labels.get(index).cloned()
    }
}
/// A field that names one option of a list it opens.
///
/// The field is measured to the width of its own content and to the height
/// of a one-line item; the list it opens takes the width of the box and as
/// much height as there is room for, and scrolls with the pointer the way a
/// viewport scrolls.
///
/// ```text
/// let chosen = ctx.use_state(|| MutableState::new(None::<usize>));
/// Select::new().options(SelectOptions::new(&["Red", "Green", "Blue"]))
///     .select(chosen.clone()).placeholder("Pick a color")
///     .on_change(move |index| log(index))
/// ```
///
/// A press inside the field is reported and not taken, so an ancestor
/// viewport sees the same down and can drag with it; only the click that
/// completes in the field is taken. A press inside the list scrolls it, and
/// a row is chosen by a click that completes inside it — a finger that
/// travels never answers for the option it passed over.
#[derive(Clone)]
pub struct Select {
    /// The options the list offers.
    options: SelectOptions,
    /// The option the field shows, shared with the application.
    selection: MutableState<Option<usize>>,
    /// Whether the list is open, and what it points at.
    mode: MutableState<SelectMode>,
    /// The text of the field while nothing is chosen.
    placeholder: Option<Rc<str>>,
    /// The offset the list scrolls with.
    scroll: ScrollState,
    /// The region that reads the field.
    field: GestureRegion,
    /// The region that reads the open list.
    list: GestureRegion,
    /// The interaction states the application observes.
    interaction: InteractionSource,
    /// The handler called with the option the *user* chose.
    on_change: Option<SelectChange>,
    /// What the select answers to.
    flags: SelectFlags,
    /// The tallest list the widget asks for, in logical pixels.
    max_height: f32,
    /// The box of the field, in window coordinates.
    field_bounds: Rect,
    /// The box of the open list, in window coordinates.
    list_bounds: Rect,
}

impl Select {
    /// A select with no options, nothing chosen and the list closed.
    ///
    /// The selection and the mode belong to the widget when built this way,
    /// so an application that reads or drives either hands a handle with
    /// [`Self::select`] and [`Self::mode_with`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: SelectOptions::default(),
            selection: MutableState::new(None),
            mode: MutableState::new(SelectMode::Closed),
            placeholder: None,
            scroll: ScrollState::new(ScrollAxes::VERTICAL),
            field: GestureRegion::new(ViewportDispatch::default()),
            list: GestureRegion::new(ViewportDispatch::default()),
            interaction: InteractionSource::new(),
            on_change: None,
            flags: SelectFlags::STANDARD,
            max_height: f32::INFINITY,
            field_bounds: Rect::ZERO,
            list_bounds: Rect::ZERO,
        }
    }

    /// The select that offers `options`.
    #[must_use]
    pub fn options(mut self, options: SelectOptions) -> Self {
        self.options = options;
        self
    }

    /// The select that shows and writes `selection`.
    #[must_use]
    pub fn select(mut self, selection: MutableState<Option<usize>>) -> Self {
        self.selection = selection;
        self
    }

    /// The select whose open state is `mode`.
    #[must_use]
    pub fn mode_with(mut self, mode: MutableState<SelectMode>) -> Self {
        self.mode = mode;
        self
    }

    /// The select that shows `text` in the field while nothing is chosen.
    #[must_use]
    pub fn placeholder(mut self, text: impl AsRef<str>) -> Self {
        self.placeholder = Some(Rc::from(text.as_ref()));
        self
    }

    /// The select whose list is at most `height` logical pixels tall — the
    /// stand-in for the room a platform popup has left in the window, which
    /// the constraints of the parent cap further.
    #[must_use]
    pub const fn max_height(mut self, height: f32) -> Self {
        self.max_height = height;
        self
    }

    /// The select that scrolls with `state`, so the application keeps the
    /// offset of the list across the recompositions that rebuilt it.
    #[must_use]
    pub fn scroll(mut self, state: ScrollState) -> Self {
        self.scroll = state;
        self
    }

    /// The select that answers the gestures its table names on the field.
    ///
    /// A slot observes or overrides that one gesture and keeps the built-in
    /// answer of the others, so an application that logs every press of the
    /// field writes [`on_press`](ViewportDispatch::on_press) and the field
    /// still opens and closes on a click.
    #[must_use]
    pub fn dispatch(mut self, table: ViewportDispatch) -> Self {
        self.field = GestureRegion::new(table);
        self
    }

    /// The select that answers the gestures its table names on the list.
    ///
    /// The table of the list is a second one because the same gesture means
    /// two things: a click on the field opens, a click on a row chooses, and
    /// a drag on the field belongs to the page while a drag on a row scrolls
    /// it.
    #[must_use]
    pub fn list_dispatch(mut self, table: ViewportDispatch) -> Self {
        self.list = GestureRegion::new(table);
        self
    }

    /// The select that reports the option the user chose to `handler`.
    #[must_use]
    pub fn on_change(mut self, handler: impl Fn(usize) + 'static) -> Self {
        self.on_change = Some(Rc::new(handler));
        self
    }

    /// The select that reports its interaction states to `source`.
    #[must_use]
    pub fn interaction_with(mut self, source: InteractionSource) -> Self {
        self.interaction = source;
        self
    }

    /// The select that answers — or, with `on` cleared, ignores — the
    /// pointer and the keyboard.
    #[must_use]
    pub fn enabled(mut self, on: bool) -> Self {
        self.set(SelectFlags::ENABLED, on);
        self
    }

    /// The select that takes — or, with `on` cleared, gives up — the
    /// keyboard focus.
    #[must_use]
    pub fn focusable(mut self, on: bool) -> Self {
        self.set(SelectFlags::FOCUSABLE, on);
        self
    }

    /// Turns `bits` on when `on` names them, and off when it does not.
    fn set(&mut self, bits: SelectFlags, on: bool) {
        self.flags = if on {
            self.flags.union(bits)
        } else {
            self.flags.without(bits)
        };
    }

    /// The options the list offers.
    #[inline]
    #[must_use]
    pub const fn options_ref(&self) -> &SelectOptions {
        &self.options
    }

    /// The option the field shows.
    #[inline]
    #[must_use]
    pub fn selected(&self) -> Option<usize> {
        self.selection.get()
    }

    /// The handle that holds the option the field shows.
    #[inline]
    #[must_use]
    pub const fn selection(&self) -> &MutableState<Option<usize>> {
        &self.selection
    }

    /// The handle that holds whether the list is open.
    #[inline]
    #[must_use]
    pub const fn mode(&self) -> &MutableState<SelectMode> {
        &self.mode
    }

    /// Whether the list is open.
    #[inline]
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.mode.get().is_open()
    }

    /// The state the list scrolls with.
    #[inline]
    #[must_use]
    pub const fn scroll_state(&self) -> &ScrollState {
        &self.scroll
    }

    /// The interaction states of the field.
    #[inline]
    #[must_use]
    pub const fn interaction(&self) -> &InteractionSource {
        &self.interaction
    }

    /// The options the select was built with.
    #[inline]
    #[must_use]
    pub const fn flags(&self) -> SelectFlags {
        self.flags
    }

    /// The text the field shows: the chosen option, or the placeholder.
    fn shown(&self) -> Option<Rc<str>> {
        self.selection
            .get()
            .and_then(|index| self.options.label(index))
            .or_else(|| self.placeholder.clone())
    }

    /// The height the list is allowed to take: the tightest of the rows it
    /// holds, the cap the application named, and the room the constraints
    /// leave under the field — the last of these is the platform's
    /// `maxHeight`, measured against the box of the widget rather than the
    /// screen.
    fn menu_height(&self, constraints: Constraints) -> f32 {
        let rows = self.options.len() as f32 * defaults::ITEM_HEIGHT;
        let mut height = rows.min(self.max_height);
        if constraints.has_bounded_height() {
            height = height.min((constraints.max_height - defaults::FIELD_HEIGHT).max(0.0));
        }
        height.max(0.0)
    }

    /// The window of rows whose labels decide the width of the widget: from
    /// the chosen option forwards for at most
    /// [`MAX_ITEMS_MEASURED`](defaults::MAX_ITEMS_MEASURED) of them, taken
    /// backwards by whatever the forward run could not reach. This is
    /// `Spinner.measureContentWidth`, row for row.
    fn measured_rows(&self) -> Range<usize> {
        let count = self.options.len();
        let start_of = self
            .selection
            .get()
            .unwrap_or(0)
            .min(count.saturating_sub(1));
        let end = count.min(start_of + defaults::MAX_ITEMS_MEASURED);
        let reached = end - start_of;
        let start = start_of.saturating_sub(defaults::MAX_ITEMS_MEASURED - reached);
        start..end
    }

    /// The width the field and its rows ask for. The field is wider than its
    /// text by the padding of both edges and the box of the trailing icon,
    /// the way `Spinner` adds its paddings to the width the adapter measured.
    fn intrinsic_width(&self, theme: &Theme) -> f32 {
        let metrics = theme.metrics.as_ref();
        let field_style = &theme.typography.body;
        let row_style = &theme.typography.label;
        let chrome = 2.0 * defaults::PADDING + defaults::ICON_BOX;
        let mut width = chrome;
        if let Some(text) = self.shown() {
            width = width.max(metrics.advance(&text, field_style) + chrome);
        }
        for index in self.measured_rows() {
            if let Some(label) = self.options.label(index) {
                width = width.max(metrics.advance(&label, row_style) + 2.0 * defaults::PADDING);
            }
        }
        width
    }

    /// The row of the list that `position` falls on, if it falls on one.
    ///
    /// The offset is what makes the answer right: the row under the pointer
    /// is the one whose place in the content, not whose place on screen, the
    /// position names.
    fn row_at(&self, position: Offset) -> Option<usize> {
        if !self.list_bounds.contains(position) {
            return None;
        }
        let y = position.y - self.list_bounds.top() + self.scroll.offset().y;
        if y < 0.0 {
            return None;
        }
        let index = (y / defaults::ITEM_HEIGHT) as usize;
        if index < self.options.len() {
            Some(index)
        } else {
            None
        }
    }

    /// Points the list at `index`, opening it when it is closed.
    ///
    /// A write that named the option the list already pointed at is skipped:
    /// the state marks every write dirty, and a pointer that only came to
    /// rest on the row it was already over has nothing to recompose.
    fn set_highlight(&self, index: usize) {
        let index = index.min(self.options.len().saturating_sub(1));
        let wanted = SelectMode::Open { highlight: index };
        if self.mode.get() != wanted {
            self.mode.set(wanted);
        }
    }

    /// Opens the list, pointing at the chosen option — which is the row on
    /// screen when the list appears, as it is for the platform before it
    /// shows its popup at all. An empty list does not open, because a popup
    /// with no row answers nothing.
    fn open(&self) {
        if self.options.is_empty() {
            return;
        }
        let highlight = self
            .selection
            .get()
            .unwrap_or(0)
            .min(self.options.len().saturating_sub(1));
        self.set_highlight(highlight);
        self.scroll_into_view(highlight);
    }

    /// Closes the list, skipping a write that left it closed: a press on an
    /// already closed field would else dirty the state and cost a
    /// recomposition for a list that was never there.
    fn close(&self) {
        if self.mode.get() != SelectMode::Closed {
            self.mode.set(SelectMode::Closed);
        }
    }

    /// Opens the list when it is closed, and puts it away when it is open.
    fn toggle(&self) {
        if self.mode.get().is_open() {
            self.close();
        } else {
            self.open();
        }
    }

    /// Scrolls the shortest distance that leaves `index` on screen: the row
    /// is pulled to whichever edge it left, never further. This is what a
    /// key repeat and the opening of the list have in common — the pointed
    /// at row is always the row the user is about to look at.
    fn scroll_into_view(&self, index: usize) {
        let window = self.scroll.viewport().height;
        if window <= 0.0 {
            return;
        }
        let top = index as f32 * defaults::ITEM_HEIGHT;
        let offset = self.scroll.offset();
        if top < offset.y {
            self.scroll.scroll_to(Offset::new(offset.x, top));
        } else if top + defaults::ITEM_HEIGHT > offset.y + window {
            self.scroll
                .scroll_to(Offset::new(offset.x, top + defaults::ITEM_HEIGHT - window));
        }
    }

    /// Chooses `index` and closes the list: the selection is written, the
    /// handler hears about it, and only then is the list dismissed, as the
    /// platform does. A choice of the option already chosen writes and
    /// reports nothing but still closes, because a tap on it was an answer.
    fn choose(&self, index: usize) {
        if index < self.options.len() && self.selection.get() != Some(index) {
            self.selection.set(Some(index));
            if let Some(handler) = &self.on_change {
                handler(index);
            }
        }
        self.close();
    }

    /// The answer of the field to `event`.
    fn answer_field(&self, event: &Event) -> EventResult {
        self.field
            .on_event(event, self.field_bounds, |gesture| match gesture {
                Gesture::Press { .. } => {
                    self.interaction.set(InteractionState::PRESSED);
                    // Reported, not taken: an ancestor has to see the same
                    // down to drag with it.
                    EventResult::Ignored
                }
                Gesture::Hover { inside: true } => {
                    self.interaction.set(InteractionState::HOVERED);
                    EventResult::Ignored
                }
                Gesture::Hover { .. } => {
                    self.interaction
                        .clear(InteractionState::HOVERED | InteractionState::PRESSED);
                    EventResult::Ignored
                }
                Gesture::Drag { .. } | Gesture::Cancel => {
                    // The finger travelled, or the press was cancelled, so
                    // it will never become the click that opens the list.
                    self.interaction.clear(InteractionState::PRESSED);
                    EventResult::Ignored
                }
                Gesture::Click { .. } => {
                    self.interaction.clear(InteractionState::PRESSED);
                    self.toggle();
                    EventResult::Consumed
                }
                _ => EventResult::Ignored,
            })
    }

    /// The answer of the open list to `event`.
    fn answer_list(&self, event: &Event) -> EventResult {
        self.list
            .on_event(event, self.list_bounds, |gesture| match gesture {
                Gesture::Press { position } => {
                    // A new press stops a fling the way a hand stops a
                    // spinning wheel: the pointer that came down owns the
                    // content from here.
                    self.scroll.stop();
                    if let Some(index) = self.row_at(*position) {
                        self.set_highlight(index);
                    }
                    EventResult::Ignored
                }
                Gesture::Drag { delta } => scrolled(&self.scroll, Offset::new(-delta.x, -delta.y)),
                Gesture::Drop { .. } => {
                    // The rows the drag moved are already where they belong,
                    // and the speed of the release is the fling's business.
                    EventResult::Ignored
                }
                Gesture::Fling { velocity } => {
                    if *velocity == Offset::ZERO {
                        return EventResult::Ignored;
                    }
                    self.scroll
                        .start_fling(Offset::new(-velocity.x, -velocity.y));
                    EventResult::Consumed
                }
                Gesture::ScrollY { delta } => scrolled(&self.scroll, Offset::new(0.0, *delta)),
                Gesture::ScrollX { delta } => scrolled(&self.scroll, Offset::new(*delta, 0.0)),
                Gesture::Click { position } => match self.row_at(*position) {
                    Some(index) => {
                        self.choose(index);
                        EventResult::Consumed
                    }
                    None => EventResult::Ignored,
                },
                Gesture::Cancel => {
                    // The row a press pointed at keeps the pointer of the
                    // keys, as a `ListView` does when its touch mode ends
                    // without a click; the tint that says a button is held is
                    // read from the region, so nothing is left to let go of.
                    EventResult::Ignored
                }
                _ => EventResult::Ignored,
            })
    }

    /// The answer of the select to a key. The confirm key is answered on the
    /// *release*, the way the buttons of this framework are, so a finger
    /// that rests on the bar does not flicker the list open and shut on
    /// every repeat; the browsing keys are answered on the press, because
    /// `AbsListView` moves its selection on the down event and a key held
    /// toward the end of a list has to reach it.
    fn answer_key(&self, key: Key, pressed: bool, focused: bool) -> EventResult {
        if !focused {
            // The framework routes a key to the widget that has the focus,
            // so an answer here is one it was not asked for.
            return EventResult::Ignored;
        }
        let open = self.mode.get();
        let highlight = open.highlight().unwrap_or(0);
        if pressed {
            if !open.is_open() {
                // Closed, the keys answer nothing: a spinner opens on a
                // confirm key and does not browse its adapter until the list
                // is on screen.
                return EventResult::Ignored;
            }
            let moved = if self.options.is_empty() {
                None
            } else {
                let last = self.options.len() - 1;
                match key {
                    Key::ArrowDown => Some((highlight + 1).min(last)),
                    Key::ArrowUp => Some(highlight.saturating_sub(1)),
                    Key::Home => Some(0),
                    Key::End => Some(last),
                    Key::PageDown => return self.page(1),
                    Key::PageUp => return self.page(-1),
                    _ => None,
                }
            };
            if let Some(index) = moved {
                self.set_highlight(index);
                self.scroll_into_view(index);
                return EventResult::Consumed;
            }
            return EventResult::Ignored;
        }
        if matches!(key, Key::Space | Key::Enter) {
            if open.is_open() {
                self.choose(highlight);
            } else {
                self.toggle();
            }
            return EventResult::Consumed;
        }
        if key == Key::Escape && open.is_open() {
            self.close();
            return EventResult::Consumed;
        }
        EventResult::Ignored
    }

    /// Scrolls a page of the list and points at the row the page moved in. A
    /// page that could not move answers nothing: there was no row to point
    /// at, and no reason to take the key from whatever else could use it.
    fn page(&self, direction: i32) -> EventResult {
        let window = self.scroll.viewport().height;
        if window <= 0.0 || self.options.is_empty() {
            return EventResult::Ignored;
        }
        let asked = if direction > 0 { window } else { -window };
        let left = self.scroll.scroll_by(Offset::new(0.0, asked));
        if left.y == asked {
            return EventResult::Ignored;
        }
        let offset = self.scroll.offset().y;
        let index = if direction > 0 {
            (offset + window - defaults::ITEM_HEIGHT).max(0.0) / defaults::ITEM_HEIGHT
        } else {
            offset / defaults::ITEM_HEIGHT
        };
        let index = (index as usize).min(self.options.len() - 1);
        self.set_highlight(index);
        EventResult::Consumed
    }

    /// The rows the open list paints. Only the rows the window shows are
    /// visited: a thousand options at 56 pixels a row draw the ten that fit
    /// rather than recording a thousand rectangles for the clip to discard.
    fn visible_rows(&self, menu: f32, offset: f32) -> Range<usize> {
        let first = (offset / defaults::ITEM_HEIGHT).max(0.0) as usize;
        // The row that only its first pixel has come into the window is
        // still painted, because the clip — not the range — is what cuts it.
        let end =
            libm::ceilf((offset + menu) / defaults::ITEM_HEIGHT).min(self.options.len() as f32) as usize;
        first.min(end)..end
    }
}

impl Default for Select {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Select {
    /// Reports the field, and the list as far as the box of the widget
    /// reaches when the list is open.
    ///
    /// The layout of the scroll state is recorded for the list *whether or
    /// not it is open*: the room the rows have is decided here, and a select
    /// that told the state its size only once it was open would scroll
    /// against a content of zero on the frame it opened, with the option it
    /// points at unscrolled.
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let width = self.intrinsic_width(scope.theme());
        let menu = self.menu_height(constraints);
        let rows = self.options.len() as f32 * defaults::ITEM_HEIGHT;
        self.scroll
            .set_layout(Size::new(width, menu), Size::new(width, rows));
        let height = defaults::FIELD_HEIGHT + if self.is_open() { menu } else { 0.0 };
        constraints.enforce(Size::new(width, height))
    }

    /// Records the two boxes the two regions recognize inside.
    fn place(&mut self, _scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.field_bounds = Rect::from_min_size(bounds.origin, bounds.size.width, defaults::FIELD_HEIGHT);
        let menu = (bounds.size.height - defaults::FIELD_HEIGHT).max(0.0);
        self.list_bounds = if self.is_open() && menu > 0.0 {
            Rect::from_min_size(
                Offset::new(bounds.left(), bounds.top() + defaults::FIELD_HEIGHT),
                bounds.size.width,
                menu,
            )
        } else {
            Rect::ZERO
        };
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let theme = scope.theme();
        let colors = SelectColors::for_scheme(&theme.colors, self.flags);
        let interaction = self.interaction.state();
        let focused =
            self.flags.contains(SelectFlags::FOCUSABLE) && interaction.contains(InteractionState::FOCUSED);
        let mode = self.mode.get();
        let open = mode.is_open();
        let selection = self.selection.get();
        let offset = self.scroll.offset().y;
        let width = bounds.size.width;
        let field = Rect::from_min_size(Offset::ZERO, width, defaults::FIELD_HEIGHT);
        let menu = (bounds.size.height - defaults::FIELD_HEIGHT).max(0.0);
        let list = Rect::from_min_size(Offset::new(0.0, defaults::FIELD_HEIGHT), width, menu);
        let metrics = theme.metrics.as_ref();
        let field_style = theme.typography.body.clone();
        let row_style = theme.typography.label.clone();
        let field_ascent = metrics.ascent(&field_style);
        let field_line = metrics.line_height(&field_style);
        let row_ascent = metrics.ascent(&row_style);
        let row_line = metrics.line_height(&row_style);
        let radii = CornerRadii::uniform(defaults::CORNER);

        // The layer first, so the outline and the text of the field sit on
        // top of the shade the interaction left.
        let overlay = colors.overlay_for(interaction);
        if overlay.alpha > 0.0 {
            scope.fill_rounded_rect(field, radii, overlay);
        }
        if colors.outline.alpha > 0.0 {
            scope.stroke_rect(
                field,
                radii,
                if focused { colors.focus } else { colors.outline },
                if focused {
                    defaults::FOCUSED_OUTLINE
                } else {
                    defaults::OUTLINE
                },
            );
        }
        if let Some(text) = self.shown() {
            let style = field_style.with_color(if selection.is_some() {
                colors.label
            } else {
                colors.muted
            });
            let y = (defaults::FIELD_HEIGHT - field_line) * 0.5;
            scope.draw_text(Offset::new(defaults::PADDING, y + field_ascent), text, style);
        }
        draw_caret(scope, field, open, colors.muted);

        if !open {
            return;
        }
        scope.fill_rounded_rect(list, radii, colors.container);
        // The rows are cut to the box they scroll inside, rounding and all:
        // the row that runs past the bottom edge would otherwise paint over
        // the corner the container was drawn with.
        scope.push_rounded_clip(list, radii);
        let pressed = self.list.is_pressed();
        for index in self.visible_rows(menu, offset) {
            let Some(label) = self.options.label(index) else {
                continue;
            };
            let row = Rect::from_min_size(
                Offset::new(
                    0.0,
                    defaults::FIELD_HEIGHT + index as f32 * defaults::ITEM_HEIGHT - offset,
                ),
                width,
                defaults::ITEM_HEIGHT,
            );
            let chosen = selection == Some(index);
            if chosen {
                scope.fill_rect(row, colors.selected);
            }
            if mode.highlight() == Some(index) {
                scope.fill_rect(row, colors.row_overlay_for(pressed));
            }
            let style = row_style
                .clone()
                .with_color(if chosen { colors.on_selected } else { colors.label });
            let y = row.top() + (defaults::ITEM_HEIGHT - row_line) * 0.5;
            scope.draw_text(Offset::new(defaults::PADDING, y + row_ascent), label, style);
        }
        scope.pop_clip();
    }

    /// The select reads the pointer only while it is enabled: a disabled
    /// select lets the press fall through to the page it sits in.
    fn hit_testable(&self) -> bool {
        self.flags.contains(SelectFlags::ENABLED)
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        if !self.flags.contains(SelectFlags::ENABLED) {
            return EventResult::Ignored;
        }
        let down = matches!(event, Event::Pointer(pointer)
            if pointer.kind == PointerEventKind::Down);
        if down && self.flags.contains(SelectFlags::FOCUSABLE) {
            scope.request_focus();
            self.interaction.set(InteractionState::FOCUSED);
        }
        if matches!(event, Event::Focus(gained) if !gained) {
            self.interaction.clear(InteractionState::FOCUSED);
            self.close();
            return EventResult::Ignored;
        }
        if !down
            && self
                .interaction
                .state()
                .contains(InteractionState::FOCUSED)
            && !scope.is_focused()
        {
            // Another widget took the focus while the field had it. The
            // framework tells no widget about the loss — only the one that
            // asks for the focus wins — so the field learns it the next time
            // an event reaches it, and puts its list away.
            self.interaction.clear(InteractionState::FOCUSED);
            self.close();
        }
        if let Event::Key(key) = event {
            return self.answer_key(key.key, key.pressed, scope.is_focused());
        }
        let field = self.answer_field(event);
        if field.is_consumed() || !self.is_open() {
            return field;
        }
        let list = self.answer_list(event);
        if list.is_consumed() { list } else { field }
    }

    /// Runs the fling the list started.
    fn advance(&mut self, elapsed: Timestamp) -> bool {
        self.scroll.advance(elapsed)
    }
}

/// Draws the dropdown icon of the field, turned the way the list is: down
/// while the list is closed and up while it is open, which is how a select
/// says that pressing it again puts the rows away.
fn draw_caret(scope: &mut DrawScope<'_>, field: Rect, open: bool, color: Color) {
    let left = field.right() - defaults::PADDING - defaults::ICON_BOX;
    let top = field.top() + (defaults::FIELD_HEIGHT - defaults::ICON_BOX) * 0.5;
    let [first, middle, last] = defaults::CARET.map(|corner| {
        let y = if open { 1.0 - corner.y } else { corner.y };
        Offset::new(left + corner.x * defaults::ICON_BOX, top + y * defaults::ICON_BOX)
    });
    scope.draw_line(first, middle, color, defaults::CARET_STROKE);
    scope.draw_line(middle, last, color, defaults::CARET_STROKE);
}
