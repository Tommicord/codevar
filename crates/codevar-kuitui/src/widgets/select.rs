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
use crate::input::mouse::{BoundId, VertexBounds};
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

    /// Names the two boxes the widget paints, each cut with the corner the
    /// drawing rounds it with.
    ///
    /// The node box of a select is the field and the list stacked, so it does
    /// cover the ink; what it gets wrong is the rounding of it. The field
    /// rounds its bottom corners and the list its top ones, so the pixels the
    /// two arcs leave out where the regions meet hold no row to choose, and a
    /// press there belongs to the page underneath rather than to a select the
    /// pointer was never over.
    fn hit_shapes(&self, _bounds: Rect, shapes: &mut Vec<VertexBounds>) {
        let radii = CornerRadii::uniform(defaults::CORNER);
        if !self.field_bounds.is_empty() {
            shapes.extend(VertexBounds::rounded_rect(
                BoundId::new(0),
                self.field_bounds,
                radii,
            ));
        }
        if !self.list_bounds.is_empty() {
            shapes.extend(VertexBounds::rounded_rect(
                BoundId::new(1),
                self.list_bounds,
                radii,
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{KeyEvent, PointerEvent, ScrollEvent};
    use crate::node::{self, Node};
    use crate::paint::{DrawList, LineOp, PaintStyle, RectOp, TextOp};
    use crate::theme::Theme;
    use alloc::string::String;
    use core::cell::RefCell;

    /// The chrome a field is wider than its own text by: the padding of both
    /// edges and the box of the trailing icon.
    const CHROME: f32 = 2.0 * defaults::PADDING + defaults::ICON_BOX;

    /// A point inside the field of every select a test lays out.
    const FIELD: Offset = Offset::new(28.0, 28.0);

    /// A point no field, row or list of any size contains.
    const OUTSIDE: Offset = Offset::new(10_000.0, 10_000.0);

    /// The label wide enough to outmeasure the numbered options a test builds.
    const WIDE: &str = "the widest option of them all";

    /// Four options whose widest label is the last of them.
    const COLORS: [&str; 4] = ["Red", "Green", "Blue", "Yellow"];

    /// Whether two lengths agree to within a hundredth of a pixel.
    fn near(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() < 0.01
    }

    /// The center of the row `index` of an open list, in window coordinates.
    fn row_center(index: usize) -> Offset {
        Offset::new(
            28.0,
            defaults::FIELD_HEIGHT + (index as f32 + 0.5) * defaults::ITEM_HEIGHT,
        )
    }

    /// `count` options labelled `00`, `01`, …, the one at `widest` carrying
    /// the label no other of them can outmeasure.
    fn options(count: usize, widest: usize) -> SelectOptions {
        let names: Vec<String> = (0..count)
            .map(|index| {
                if index == widest {
                    String::from(WIDE)
                } else {
                    alloc::format!("{index:02}")
                }
            })
            .collect();
        let labels: Vec<&str> = names.iter().map(String::as_str).collect();
        SelectOptions::new(&labels)
    }

    /// What one call of the change handler saw.
    struct Report {
        /// The position the handler was given.
        index: usize,
        /// The option the field showed when it ran.
        chosen: Option<usize>,
        /// Whether the list was still open when it ran.
        opening: bool,
    }

    /// A tree of one select: the state the widget writes and the test reads,
    /// and the constraints it is laid out under.
    struct Harness {
        theme: Theme,
        node: Node,
        focus: Option<Vec<usize>>,
        constraints: Constraints,
        selection: MutableState<Option<usize>>,
        mode: MutableState<SelectMode>,
        scroll: ScrollState,
        interaction: InteractionSource,
        reports: Rc<RefCell<Vec<Report>>>,
    }

    impl Harness {
        /// A tree of `select`, measured for `constraints` and placed at the
        /// origin, with a change handler that records every report it makes.
        fn new(constraints: Constraints, select: Select) -> Self {
            let theme = Theme::light();
            let selection = select.selection().clone();
            let mode = select.mode().clone();
            let scroll = select.scroll_state().clone();
            let interaction = select.interaction().clone();
            let reports = Rc::new(RefCell::new(Vec::new()));
            let seen = Rc::clone(&reports);
            let chosen = selection.clone();
            let opening = mode.clone();
            let select = select.on_change(move |index| {
                seen.borrow_mut().push(Report {
                    index,
                    chosen: chosen.get(),
                    opening: opening.get().is_open(),
                });
            });
            let mut node = Node::new(select);
            let size = node.measure(&theme, constraints);
            node.place(Rect::from_min_size(Offset::ZERO, size.width, size.height));
            Self {
                theme,
                node,
                focus: None,
                constraints,
                selection,
                mode,
                scroll,
                interaction,
                reports,
            }
        }

        /// A select over `options` whose list shows `rows` of them at a time,
        /// with the whole page unbounded.
        fn list(options: SelectOptions, rows: usize) -> Self {
            let select = Select::new()
                .options(options)
                .max_height(rows as f32 * defaults::ITEM_HEIGHT);
            Self::new(Constraints::unbounded(), select)
        }

        /// Lays the tree out again: what a recomposition does once the list
        /// has opened and the widget asks for the height of its rows.
        fn relayout(&mut self) {
            let size = self.node.measure(&self.theme, self.constraints);
            self.node
                .place(Rect::from_min_size(Offset::ZERO, size.width, size.height));
        }

        /// Opens the list the way a pointer does and lays out its rows.
        fn open(&mut self) {
            self.pointer(PointerEventKind::Down, FIELD);
            self.pointer(PointerEventKind::Up, FIELD);
            self.relayout();
        }

        /// Offers `event` to the tree and reports what it answered.
        fn offer(&mut self, event: &Event) -> EventResult {
            let mut path = Vec::new();
            node::dispatch_event(&mut self.node, &self.theme, &mut path, &mut self.focus, event)
        }

        /// Offers a pointer event of `kind` at `position`.
        fn pointer(&mut self, kind: PointerEventKind, position: Offset) -> EventResult {
            self.offer(&Event::Pointer(PointerEvent::new(kind, position)))
        }

        /// Offers a pointer event of `kind` at `position` that the backend
        /// observed at `micros`, which is what makes a speed measurable.
        fn pointer_at(&mut self, kind: PointerEventKind, position: Offset, micros: u64) -> EventResult {
            self.offer(&Event::Pointer(
                PointerEvent::new(kind, position).with_time(Timestamp::from_micros(micros)),
            ))
        }

        /// Offers the wheel `delta` under `position`.
        fn wheel(&mut self, delta: Offset, position: Offset) -> EventResult {
            self.offer(&Event::Scroll(ScrollEvent::new(delta, position)))
        }

        /// Offers the press of `key` to the tree.
        fn press(&mut self, key: Key) -> EventResult {
            self.offer(&Event::Key(KeyEvent::new(key, true)))
        }

        /// Offers the release of `key` to the tree.
        fn release(&mut self, key: Key) -> EventResult {
            self.offer(&Event::Key(KeyEvent::new(key, false)))
        }

        /// The select as it paints itself right now.
        fn canvas(&mut self) -> Canvas {
            let mut list = DrawList::new();
            self.node.draw(&self.theme, &mut list);
            Canvas::of(&list)
        }
    }

    /// A page with one select in it: the tree that decides whether a point
    /// reaches the widget at all, which a tree whose root *is* the select
    /// cannot show because the gate only guards a child.
    struct Page {
        theme: Theme,
        node: Node,
        interaction: InteractionSource,
        mode: MutableState<SelectMode>,
        selection: MutableState<Option<usize>>,
        select: Rect,
        focus: Option<Vec<usize>>,
    }

    impl Page {
        /// A page holding `select`, laid out at the origin at the size it
        /// asked for.
        fn of(select: Select) -> Self {
            let theme = Theme::light();
            let interaction = select.interaction().clone();
            let mode = select.mode().clone();
            let selection = select.selection().clone();
            let mut child = Node::new(select);
            let size = child.measure(&theme, Constraints::unbounded());
            let mut node = Node::new(crate::widgets::Row::new()).with_children(Vec::from([child]));
            node.measure(&theme, Constraints::unbounded());
            node.place(Rect::from_min_size(Offset::ZERO, size.width, size.height));
            let select = node.children()[0].bounds;
            Self {
                theme,
                node,
                interaction,
                mode,
                selection,
                select,
                focus: None,
            }
        }

        /// The select of `colors`, four options deep enough to show a list.
        fn colors() -> Self {
            Self::of(
                Select::new()
                    .options(SelectOptions::new(&COLORS))
                    .max_height(4.0 * defaults::ITEM_HEIGHT),
            )
        }

        /// Lays the page out again, which is what a recomposition does once
        /// the list has opened and the widget asked for the height of rows.
        fn relayout(&mut self) {
            let size = self
                .node
                .measure(&self.theme, Constraints::unbounded());
            self.node
                .place(Rect::from_min_size(Offset::ZERO, size.width, size.height));
            self.select = self.node.children()[0].bounds;
        }

        /// Offers a pointer event of `kind` at `position` to the page.
        fn pointer(&mut self, kind: PointerEventKind, position: Offset) -> EventResult {
            let mut path = Vec::new();
            node::dispatch_event(
                &mut self.node,
                &self.theme,
                &mut path,
                &mut self.focus,
                &Event::Pointer(PointerEvent::new(kind, position)),
            )
        }

        /// Presses and releases at `position`, the way a pointer that wants
        /// one thing from the widget does.
        fn click(&mut self, position: Offset) {
            self.pointer(PointerEventKind::Down, position);
            self.pointer(PointerEventKind::Up, position);
        }

        /// Whether the list of the select is on the screen.
        fn opened(&self) -> bool {
            self.mode.get().is_open()
        }

        /// Presses and releases the field the way a pointer that wants the
        /// list open does, then lays the page out for the rows it added.
        fn open(&mut self) {
            let field = self.select.center();
            self.click(field);
            self.relayout();
        }

        /// Whether the press the page answers for is the one the field took.
        fn pressed(&self) -> bool {
            self.interaction
                .state()
                .contains(InteractionState::PRESSED)
        }
    }

    /// The shapes a draw list recorded, split by how they were painted.
    #[derive(Default)]
    struct Canvas {
        /// The filled rectangles, in paint order.
        fills: Vec<(Rect, Color)>,
        /// The outlined rectangles with the width of their stroke.
        outlines: Vec<(Rect, Color, f32)>,
        /// The strokes of the icon: their ends and their color.
        lines: Vec<(Offset, Offset, Color)>,
        /// The runs of text: their baseline, their characters, their color.
        texts: Vec<(Offset, Rc<str>, Color)>,
        /// The box every command was cut to, and the corners of that box.
        clips: Vec<(Option<Rect>, CornerRadii)>,
    }

    impl Canvas {
        /// The recorded shapes of `list`, in the order it recorded them.
        fn of(list: &DrawList) -> Self {
            let mut canvas = Self::default();
            for command in list.commands() {
                canvas
                    .clips
                    .push((command.clip, command.clip_radii));
                let any: &dyn core::any::Any = &*command.op;
                if let Some(rect) = any.downcast_ref::<RectOp>() {
                    match rect.paint.style {
                        PaintStyle::Fill => canvas.fills.push((rect.rect, rect.paint.color)),
                        PaintStyle::Stroke(width) => {
                            canvas
                                .outlines
                                .push((rect.rect, rect.paint.color, width))
                        }
                    }
                } else if let Some(line) = any.downcast_ref::<LineOp>() {
                    canvas
                        .lines
                        .push((line.from, line.to, line.paint.color));
                } else if let Some(text) = any.downcast_ref::<TextOp>() {
                    canvas
                        .texts
                        .push((text.origin, text.text.clone(), text.style.color));
                }
            }
            canvas
        }
    }

    /// The baseline the label of row `index` sits on in a list scrolled by
    /// `offset`, as the theme's own metrics put it.
    fn row_baseline(theme: &Theme, index: usize, offset: f32) -> f32 {
        let metrics = theme.metrics.as_ref();
        let style = &theme.typography.label;
        defaults::FIELD_HEIGHT + index as f32 * defaults::ITEM_HEIGHT - offset
            + (defaults::ITEM_HEIGHT - metrics.line_height(style)) * 0.5
            + metrics.ascent(style)
    }

    #[test]
    fn a_select_measures_its_field_and_the_widest_row_it_shows() {
        let theme = Theme::light();
        let harness = Harness::new(
            Constraints::unbounded(),
            Select::new().options(SelectOptions::new(&COLORS)),
        );
        let widest = theme
            .metrics
            .as_ref()
            .advance("Yellow", &theme.typography.label)
            + 2.0 * defaults::PADDING;
        assert!(
            widest > CHROME,
            "the labels of the test ask for more room than the chrome of the field"
        );
        assert!(
            near(harness.node.size().width, widest),
            "the row that needs the most room sets the width of the field"
        );
        assert_eq!(harness.node.size().height, defaults::FIELD_HEIGHT);
    }

    #[test]
    fn the_width_is_measured_over_the_window_of_rows_around_the_choice() {
        let theme = Theme::light();
        let widest = theme
            .metrics
            .as_ref()
            .advance(WIDE, &theme.typography.label)
            + 2.0 * defaults::PADDING;
        let field_of = |index: usize| {
            theme
                .metrics
                .as_ref()
                .advance(&alloc::format!("{index:02}"), &theme.typography.body)
                + CHROME
        };
        let mut harness = Harness::new(Constraints::unbounded(), Select::new().options(options(40, 39)));
        assert!(
            near(harness.node.size().width, CHROME),
            "with nothing chosen the window of rows measured is the first fifteen"
        );
        harness.selection.set(Some(20));
        harness.relayout();
        assert!(
            near(harness.node.size().width, field_of(20)),
            "the window from the choice runs 20 .. 35, so the label at 39 cannot widen it"
        );
        harness.selection.set(Some(30));
        harness.relayout();
        assert!(
            near(harness.node.size().width, widest),
            "the rows the forward run could not reach are measured backwards"
        );
    }

    #[test]
    fn an_open_list_adds_its_rows_to_the_height_of_the_field() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        assert_eq!(harness.node.size().height, defaults::FIELD_HEIGHT);
        harness.open();
        assert_eq!(
            harness.node.size().height,
            defaults::FIELD_HEIGHT + 4.0 * defaults::ITEM_HEIGHT,
            "the list is part of the layout: it pushes the page rather than covering it"
        );
        harness.pointer(PointerEventKind::Down, FIELD);
        harness.pointer(PointerEventKind::Up, FIELD);
        harness.relayout();
        assert_eq!(harness.node.size().height, defaults::FIELD_HEIGHT);
    }

    #[test]
    fn the_menu_is_cut_to_the_room_the_page_leaves_before_the_list_opens() {
        let mut harness = Harness::new(
            Constraints::new(0.0, f32::INFINITY, 0.0, 100.0),
            Select::new().options(options(20, usize::MAX)),
        );
        assert_eq!(
            harness.node.size().height,
            defaults::FIELD_HEIGHT,
            "a closed select takes no room for the list it is not showing"
        );
        assert!(
            near(harness.scroll.viewport().height, 100.0 - defaults::FIELD_HEIGHT),
            "the list learns the window it will get while it is still closed, so the row it \
             points at is on screen the frame it appears"
        );
        assert!(near(
            harness.scroll.content().height,
            20.0 * defaults::ITEM_HEIGHT
        ));
        harness.open();
        assert_eq!(
            harness.node.size().height,
            100.0,
            "the field plus the room the page left under it"
        );
    }

    #[test]
    fn a_select_with_no_options_never_opens() {
        let mut harness = Harness::new(Constraints::unbounded(), Select::new());
        assert!(
            !harness
                .pointer(PointerEventKind::Down, FIELD)
                .is_consumed(),
            "the press of a field with no list is still reported and not taken"
        );
        assert!(
            harness
                .pointer(PointerEventKind::Up, FIELD)
                .is_consumed(),
            "the field still answers the click it holds"
        );
        assert!(
            !harness.mode.get().is_open(),
            "a list with no row has nothing to show"
        );
    }

    #[test]
    fn a_click_on_the_field_opens_the_list_and_the_next_puts_it_away() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.pointer(PointerEventKind::Down, FIELD);
        assert!(
            harness
                .interaction
                .state()
                .contains(InteractionState::PRESSED),
            "the field reports the button it is holding"
        );
        harness.open();
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 0 });
        assert!(
            !harness
                .pointer(PointerEventKind::Down, FIELD)
                .is_consumed(),
            "the down is left to the page so an ancestor can drag with it"
        );
        assert!(
            harness
                .pointer(PointerEventKind::Up, FIELD)
                .is_consumed()
        );
        assert_eq!(harness.mode.get(), SelectMode::Closed);
    }

    #[test]
    fn opening_points_at_the_chosen_row_and_pulls_it_onto_the_screen() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.selection.set(Some(17));
        harness.relayout();
        harness.open();
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 17 });
        assert!(
            near(
                harness.scroll.offset().y,
                17.0 * defaults::ITEM_HEIGHT + defaults::ITEM_HEIGHT - harness.scroll.viewport().height
            ),
            "the chosen row is scrolled the shortest distance that leaves it on screen"
        );
    }

    #[test]
    fn a_press_on_the_field_is_reported_and_a_drag_leaves_it() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.pointer(PointerEventKind::Down, FIELD);
        assert!(
            harness
                .interaction
                .state()
                .contains(InteractionState::PRESSED)
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Move, Offset::new(28.0, 40.0))
                .is_consumed(),
            "a pointer that travelled past the slop is the page's drag, not a click"
        );
        assert!(
            !harness
                .interaction
                .state()
                .contains(InteractionState::PRESSED),
            "the field lets go of a press that will never become a click"
        );
        harness.pointer(PointerEventKind::Up, Offset::new(28.0, 40.0));
        assert!(!harness.mode.get().is_open());
    }

    #[test]
    fn a_release_outside_the_field_ends_the_press_and_opens_nothing() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.pointer(PointerEventKind::Down, FIELD);
        assert!(
            harness
                .pointer(PointerEventKind::Up, OUTSIDE)
                .is_consumed(),
            "the region that held the press still ends it, even when nothing completed"
        );
        assert!(
            !harness.mode.get().is_open(),
            "a pointer that left the box before releasing pressed no field"
        );
        assert!(
            !harness
                .interaction
                .state()
                .contains(InteractionState::PRESSED)
        );
    }

    #[test]
    fn a_click_on_a_row_chooses_it_writes_and_then_dismisses() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.open();
        assert!(
            !harness
                .pointer(PointerEventKind::Down, row_center(2))
                .is_consumed()
        );
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 2 });
        assert!(
            harness
                .pointer(PointerEventKind::Up, row_center(2))
                .is_consumed()
        );
        assert_eq!(harness.selection.get(), Some(2));
        assert!(
            !harness.mode.get().is_open(),
            "the list is dismissed after the choice"
        );
        let reports = harness.reports.borrow();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].index, 2);
        assert_eq!(
            reports[0].chosen,
            Some(2),
            "the handler is told about the choice the field already shows"
        );
        assert!(
            reports[0].opening,
            "the handler runs before the list is put away, as the platform orders them"
        );
    }

    #[test]
    fn choosing_the_option_already_chosen_reports_nothing_but_still_closes() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.selection.set(Some(1));
        harness.open();
        harness.pointer(PointerEventKind::Down, row_center(1));
        assert!(
            harness
                .pointer(PointerEventKind::Up, row_center(1))
                .is_consumed()
        );
        assert_eq!(harness.selection.get(), Some(1));
        assert!(
            harness.reports.borrow().is_empty(),
            "a tap on the option already chosen is an answer, not a change"
        );
        assert!(!harness.mode.get().is_open());
    }

    #[test]
    fn a_drag_that_travelled_never_answers_for_the_option_it_passed_over() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        harness.pointer(PointerEventKind::Down, row_center(1));
        harness.pointer(PointerEventKind::Move, row_center(0));
        harness.pointer(PointerEventKind::Up, row_center(0));
        assert!(
            harness.selection.get().is_none(),
            "the finger travelled over the rows, so it chose none of them"
        );
        assert!(harness.reports.borrow().is_empty());
        assert!(harness.mode.get().is_open(), "the list stayed on screen");
    }

    #[test]
    fn a_drag_scrolls_the_rows_and_hands_the_page_what_they_cannot_use() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        harness.pointer(PointerEventKind::Down, Offset::new(28.0, 140.0));
        assert!(
            !harness
                .pointer(PointerEventKind::Move, Offset::new(28.0, 170.0))
                .is_consumed(),
            "a drag that would pull the list past its first row belongs to the page"
        );
        assert!(near(harness.scroll.offset().y, 0.0));
        assert!(
            harness
                .pointer(PointerEventKind::Move, Offset::new(28.0, 110.0))
                .is_consumed(),
            "the other way the rows have room, and they take the drag"
        );
        assert!(near(harness.scroll.offset().y, 60.0));
        harness.pointer(PointerEventKind::Up, Offset::new(28.0, 110.0));
    }

    #[test]
    fn the_wheel_moves_the_axis_the_list_can_scroll() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        assert!(
            harness
                .wheel(Offset::new(0.0, 40.0), row_center(0))
                .is_consumed()
        );
        assert!(near(harness.scroll.offset().y, 40.0));
        assert!(
            !harness
                .wheel(Offset::new(30.0, 0.0), row_center(0))
                .is_consumed(),
            "a list that only runs down leaves the sideways part of the wheel to the page"
        );
        assert!(
            !harness
                .wheel(Offset::new(0.0, 10.0), FIELD)
                .is_consumed()
        );
        assert!(
            near(harness.scroll.offset().y, 40.0),
            "the wheel under the field is not the list's"
        );
    }

    #[test]
    fn a_fast_release_throws_the_list_and_the_next_press_catches_it() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        harness.pointer_at(PointerEventKind::Down, Offset::new(28.0, 140.0), 0);
        harness.pointer_at(PointerEventKind::Move, Offset::new(28.0, 120.0), 8_000);
        harness.pointer_at(PointerEventKind::Move, Offset::new(28.0, 90.0), 16_000);
        assert!(
            harness
                .pointer_at(PointerEventKind::Up, Offset::new(28.0, 90.0), 16_000)
                .is_consumed()
        );
        assert!(
            harness.scroll.is_flinging(),
            "a release the backend timed is fast enough to carry the rows"
        );
        let thrown = harness.scroll.offset().y;
        harness
            .scroll
            .advance(Timestamp::from_micros(16_000));
        assert!(
            harness.scroll.offset().y > thrown,
            "the fling keeps moving the content after the pointer left it"
        );
        harness.pointer(PointerEventKind::Down, Offset::new(28.0, 100.0));
        assert!(
            !harness.scroll.is_flinging(),
            "the pointer that came down owns the content from here"
        );
    }

    #[test]
    fn a_select_without_the_focus_answers_no_key() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        assert!(
            !harness.press(Key::ArrowDown).is_consumed(),
            "the framework routes a key to the widget that has the focus, so an answer here \
             would be one nobody asked for"
        );
        assert!(
            !harness.release(Key::Space).is_consumed(),
            "the page keeps a key a field without the focus would rather have"
        );
        assert_eq!(harness.mode.get(), SelectMode::Closed);
    }

    #[test]
    fn the_confirm_key_released_opens_and_the_next_chooses_the_row_the_keys_reached() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        harness.release(Key::Escape);
        harness.relayout();
        assert_eq!(harness.mode.get(), SelectMode::Closed);
        assert!(
            harness.release(Key::Space).is_consumed(),
            "the confirm key is answered on its release, so a bar held down cannot flicker \
             the list open and shut"
        );
        harness.relayout();
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 0 });
        for _ in 0..3 {
            assert!(harness.press(Key::ArrowDown).is_consumed());
        }
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 3 });
        assert!(harness.release(Key::Enter).is_consumed());
        assert_eq!(harness.selection.get(), Some(3));
        assert_eq!(harness.reports.borrow().len(), 1);
        assert_eq!(harness.reports.borrow()[0].index, 3);
        assert!(!harness.mode.get().is_open());
    }

    #[test]
    fn the_browsing_keys_answer_nothing_until_the_list_is_on_screen() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.open();
        harness.release(Key::Escape);
        harness.relayout();
        for key in [Key::ArrowDown, Key::ArrowUp, Key::Home, Key::End, Key::PageDown] {
            assert!(
                !harness.press(key).is_consumed(),
                "a spinner opens on a confirm key and does not browse its adapter before the \
                 list is on screen"
            );
        }
        assert_eq!(harness.mode.get(), SelectMode::Closed);
    }

    #[test]
    fn the_arrows_move_the_pointer_of_the_list_and_keep_the_row_on_screen() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        assert!(harness.press(Key::ArrowDown).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 1 });
        assert!(
            near(harness.scroll.offset().y, 0.0),
            "the second row still fits the window, so nothing moved"
        );
        assert!(harness.press(Key::ArrowDown).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 2 });
        assert!(
            near(harness.scroll.offset().y, defaults::ITEM_HEIGHT),
            "the row that left the window is pulled back to its edge, and no further"
        );
        assert!(harness.press(Key::ArrowUp).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 1 });
        assert!(harness.press(Key::Home).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 0 });
        assert!(near(harness.scroll.offset().y, 0.0));
        assert!(harness.press(Key::End).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 19 });
        assert!(
            near(
                harness.scroll.offset().y,
                20.0 * defaults::ITEM_HEIGHT - harness.scroll.viewport().height
            ),
            "the end of the list is the last row against the bottom edge"
        );
        assert!(
            !harness.press(Key::ArrowLeft).is_consumed(),
            "a sideways key is not the list's to take"
        );
    }

    #[test]
    fn a_page_moves_a_window_of_rows_and_points_at_the_row_that_arrived() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        assert!(harness.press(Key::PageDown).is_consumed());
        assert!(
            near(harness.scroll.offset().y, harness.scroll.viewport().height),
            "a page is the height of the window, as it is for a list"
        );
        assert_eq!(
            harness.mode.get(),
            SelectMode::Open { highlight: 3 },
            "the keys point at the row the page moved in at the bottom edge"
        );
        assert!(harness.press(Key::PageUp).is_consumed());
        assert!(near(harness.scroll.offset().y, 0.0));
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 0 });
        assert!(
            !harness.press(Key::PageUp).is_consumed(),
            "a page with nothing before the first row has nothing to move"
        );
        harness.scroll.scroll_to(harness.scroll.max());
        assert!(
            !harness.press(Key::PageDown).is_consumed(),
            "and the same at the end of the content"
        );
        assert_eq!(harness.mode.get(), SelectMode::Open { highlight: 0 });
    }

    #[test]
    fn escape_puts_the_list_away_and_a_second_escape_is_no_one_s() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.open();
        assert!(
            !harness.press(Key::Escape).is_consumed(),
            "a press that names no move is left where it fell"
        );
        assert!(harness.mode.get().is_open());
        assert!(harness.release(Key::Escape).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Closed);
        assert!(
            !harness.release(Key::Escape).is_consumed(),
            "a list already put away has nothing to dismiss"
        );
    }

    #[test]
    fn a_field_that_lost_the_focus_puts_its_list_away() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.open();
        assert!(
            !harness.offer(&Event::Focus(false)).is_consumed(),
            "the loss of the focus is not the select's to take"
        );
        assert_eq!(harness.mode.get(), SelectMode::Closed);
        assert!(
            !harness
                .interaction
                .state()
                .contains(InteractionState::FOCUSED)
        );
    }

    #[test]
    fn the_field_lends_its_list_the_frame_it_learns_another_widget_took_the_focus() {
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        harness.open();
        // Only the widget that asks for the focus wins it, and no widget is
        // told it lost one, so the field finds out with the next event that
        // reaches it — and its list would else answer keys on screen.
        harness.focus = None;
        harness.pointer(PointerEventKind::Move, OUTSIDE);
        assert_eq!(harness.mode.get(), SelectMode::Closed);
        assert!(
            !harness
                .interaction
                .state()
                .contains(InteractionState::FOCUSED)
        );
    }

    #[test]
    fn the_field_shows_the_placeholder_and_then_the_option_chosen() {
        let theme = Theme::light();
        let metrics = theme.metrics.as_ref();
        let baseline = (defaults::FIELD_HEIGHT - metrics.line_height(&theme.typography.body)) * 0.5
            + metrics.ascent(&theme.typography.body);
        let colors = SelectColors::for_scheme(&theme.colors, SelectFlags::STANDARD);
        let mut harness = Harness::new(
            Constraints::unbounded(),
            Select::new()
                .options(SelectOptions::new(&COLORS))
                .placeholder("Pick a color"),
        );
        let canvas = harness.canvas();
        assert_eq!(&*canvas.texts[0].1, "Pick a color");
        assert_eq!(
            canvas.texts[0].2, colors.muted,
            "the question of a field with nothing chosen is the secondary content"
        );
        assert!(near(canvas.texts[0].0.y, baseline));
        assert!(near(canvas.texts[0].0.x, defaults::PADDING));
        harness.selection.set(Some(2));
        harness.relayout();
        let canvas = harness.canvas();
        assert_eq!(&*canvas.texts[0].1, "Blue");
        assert_eq!(canvas.texts[0].2, colors.label);
        assert_eq!(canvas.outlines[0].1, colors.outline);
        assert!(
            near(canvas.outlines[0].2, defaults::OUTLINE),
            "the field without the focus outlines itself at the lighter width"
        );
    }

    #[test]
    fn the_caret_turns_with_the_list() {
        let theme = Theme::light();
        let colors = SelectColors::for_scheme(&theme.colors, SelectFlags::STANDARD);
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        let closed = harness.canvas();
        assert_eq!(
            closed.lines.len(),
            2,
            "the icon is the two segments that meet at its point"
        );
        assert!(
            closed.lines[0].0.y < closed.lines[0].1.y,
            "a closed list is offered by an arrow pointing down"
        );
        assert!(closed.lines[1].0.y > closed.lines[1].1.y);
        assert_eq!(closed.lines[0].2, colors.muted);
        let left = harness.node.size().width - defaults::PADDING - defaults::ICON_BOX;
        assert!(near(
            closed.lines[0].0.x,
            left + defaults::CARET[0].x * defaults::ICON_BOX
        ));
        harness.open();
        let open = harness.canvas();
        assert!(
            open.lines[0].0.y > open.lines[0].1.y,
            "an open list turns the icon about: pressing again puts the rows away"
        );
        assert!(open.lines[1].0.y < open.lines[1].1.y);
    }

    #[test]
    fn the_list_paints_only_the_rows_the_window_shows() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        let canvas = harness.canvas();
        assert_eq!(
            canvas.texts.len(),
            2,
            "of twenty options the two that fit are what the list visits"
        );
        assert_eq!(&*canvas.texts[0].1, "00");
        assert_eq!(&*canvas.texts[1].1, "01");
        harness.scroll.scroll_to(Offset::new(0.0, 30.0));
        let canvas = harness.canvas();
        assert_eq!(
            canvas.texts.len(),
            3,
            "a row that only its first pixel has come into the window is painted and cut by \
             the clip, not dropped by the range"
        );
        assert_eq!(&*canvas.texts[2].1, "02");
        assert!(near(canvas.texts[2].0.y, row_baseline(&harness.theme, 2, 30.0)));
        harness.scroll.scroll_to(harness.scroll.max());
        let canvas = harness.canvas();
        assert_eq!(canvas.texts.len(), 2);
        assert_eq!(&*canvas.texts[0].1, "18");
        assert_eq!(&*canvas.texts[1].1, "19");
    }

    #[test]
    fn the_rows_are_cut_to_the_rounded_box_of_the_list() {
        let mut harness = Harness::list(options(20, usize::MAX), 2);
        harness.open();
        let width = harness.node.size().width;
        let box_of_list = Rect::from_min_size(
            Offset::new(0.0, defaults::FIELD_HEIGHT),
            width,
            2.0 * defaults::ITEM_HEIGHT,
        );
        let radii = CornerRadii::uniform(defaults::CORNER);
        let canvas = harness.canvas();
        let (whole, cut): (Vec<_>, Vec<_>) = canvas
            .clips
            .into_iter()
            .partition(|(clip, _)| clip.is_none());
        assert_eq!(
            whole.len(),
            5,
            "the layer of the field, its outline, the two strokes of the icon and the \
             container of the list are painted whole"
        );
        assert_eq!(
            cut.len(),
            3,
            "the two rows the window shows, and the tint the pointed at one of them carries"
        );
        assert!(
            cut.iter()
                .all(|(clip, shape)| *clip == Some(box_of_list) && *shape == radii),
            "the rows are cut to the rounding of the box they scroll inside, so a row running \
             past the bottom edge cannot paint over the corner the container was drawn with"
        );
    }

    #[test]
    fn the_chosen_row_takes_the_container_and_a_held_row_the_darker_layer() {
        let theme = Theme::light();
        let colors = SelectColors::for_scheme(&theme.colors, SelectFlags::STANDARD);
        let mut harness = Harness::new(
            Constraints::unbounded(),
            Select::new()
                .options(SelectOptions::new(&COLORS))
                .max_height(4.0 * defaults::ITEM_HEIGHT),
        );
        harness.selection.set(Some(1));
        harness.relayout();
        harness.open();
        let width = harness.node.size().width;
        let green = Rect::from_min_size(
            Offset::new(0.0, defaults::FIELD_HEIGHT + defaults::ITEM_HEIGHT),
            width,
            defaults::ITEM_HEIGHT,
        );
        let canvas = harness.canvas();
        assert!(
            canvas.fills.contains(&(green, colors.selected)),
            "the chosen row is the one carrying the container"
        );
        assert!(
            canvas
                .fills
                .contains(&(green, colors.row_overlay_for(false))),
            "and the row the keys point at is tinted with the lighter layer while no button is held"
        );
        assert_eq!(
            canvas.texts[2].2, colors.on_selected,
            "the first text is the field, the second row 0, the third the chosen row"
        );
        assert_eq!(canvas.texts[1].2, colors.label);
        harness.pointer(PointerEventKind::Down, row_center(1));
        let held = harness.canvas();
        assert!(
            held.fills
                .contains(&(green, colors.row_overlay_for(true))),
            "the row a button is held in takes the pressed layer"
        );
    }

    #[test]
    fn the_layers_take_the_strongest_alpha_and_the_heavier_outline() {
        let theme = Theme::light();
        let colors = SelectColors::for_scheme(&theme.colors, SelectFlags::STANDARD);
        assert_eq!(colors.overlay_for(InteractionState::NONE).alpha, 0.0);
        assert_eq!(
            colors
                .overlay_for(InteractionState::HOVERED)
                .alpha,
            defaults::HOVER_ALPHA
        );
        assert_eq!(
            colors
                .overlay_for(InteractionState::HOVERED | InteractionState::PRESSED)
                .alpha,
            defaults::PRESSED_ALPHA
        );
        assert_eq!(
            colors
                .overlay_for(
                    InteractionState::FOCUSED | InteractionState::PRESSED | InteractionState::HOVERED
                )
                .alpha,
            defaults::FOCUSED_ALPHA,
            "the layers of Material 3 do not add up: the strongest one that applies is what shows"
        );
        let mut harness = Harness::list(SelectOptions::new(&COLORS), 4);
        assert!(
            harness.canvas().fills.is_empty(),
            "a field at rest paints no layer at all"
        );
        harness.pointer(PointerEventKind::Down, FIELD);
        let canvas = harness.canvas();
        assert_eq!(
            canvas.fills[0].0,
            Rect::from_min_size(Offset::ZERO, harness.node.size().width, defaults::FIELD_HEIGHT),
            "the layer is the field, painted before the outline and the text sit on it"
        );
        assert_eq!(
            canvas.outlines[0].1, colors.focus,
            "the field with the focus outlines itself with the color of the focus"
        );
        assert!(near(canvas.outlines[0].2, defaults::FOCUSED_OUTLINE));
    }

    #[test]
    fn a_disabled_select_answers_nothing_and_paints_the_disabled_alphas() {
        let mut harness = Harness::new(
            Constraints::unbounded(),
            Select::new()
                .options(SelectOptions::new(&COLORS))
                .placeholder("Pick a color")
                .enabled(false),
        );
        assert!(
            !harness.node.hit_testable(),
            "a disabled select takes no part in the hit test, so the page under it is pressed"
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Down, FIELD)
                .is_consumed()
        );
        assert!(
            !harness
                .pointer(PointerEventKind::Up, FIELD)
                .is_consumed()
        );
        assert!(!harness.release(Key::Space).is_consumed());
        assert_eq!(harness.mode.get(), SelectMode::Closed);
        let canvas = harness.canvas();
        assert!(
            near(canvas.outlines[0].1.alpha, defaults::DISABLED_OUTLINE_ALPHA),
            "the outline still shows where the field ends"
        );
        assert!(
            near(canvas.texts[0].2.alpha, defaults::DISABLED_CONTENT_ALPHA),
            "and the content loses the color of its role rather than the shape of it"
        );
        assert!(near(canvas.lines[0].2.alpha, defaults::DISABLED_CONTENT_ALPHA));
        assert_eq!(
            canvas.fills,
            Vec::new(),
            "a field that answers nothing is never in an interaction state"
        );
    }

    #[test]
    fn a_press_in_the_corner_the_field_rounds_away_reaches_the_page() {
        let mut page = Page::colors();
        let corner = Offset::new(page.select.left() + 0.5, page.select.top() + 0.5);
        assert!(
            page.select.contains(corner),
            "the point is inside the box the node was laid out in"
        );

        page.click(corner);
        assert!(
            !page.opened(),
            "the arc cut that pixel off the field, so the select never learned of the press"
        );
        assert!(!page.pressed());

        page.click(page.select.center());
        assert!(page.opened(), "the same press inside the arc reaches the field");
    }

    #[test]
    fn a_press_in_the_notch_the_two_regions_round_between_them_reaches_the_page() {
        let mut page = Page::colors();
        page.open();
        assert!(
            page.select.size.height > defaults::FIELD_HEIGHT,
            "the list the page now holds is the half of the box below the field"
        );

        let notch = page.select.left() + 0.5;
        let in_list = Offset::new(notch, page.select.top() + defaults::FIELD_HEIGHT + 0.5);
        assert!(
            page.select.contains(in_list),
            "the box of the list still covers the pixel its own corner rounds away"
        );
        page.click(in_list);
        assert_eq!(
            page.selection.get(),
            None,
            "the corner the list rounds holds no row to choose"
        );
        assert!(
            page.opened(),
            "and the press did not dismiss the list the pointer was not over"
        );

        page.click(row_center(0));
        assert_eq!(
            page.selection.get(),
            Some(0),
            "a press on an inked row still reaches the list"
        );
    }
}
