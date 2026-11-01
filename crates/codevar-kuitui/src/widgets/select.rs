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

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::ops::{BitOr, BitOrAssign};

use crate::color::{Color, ColorScheme};
use crate::effect::InteractionState;

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
