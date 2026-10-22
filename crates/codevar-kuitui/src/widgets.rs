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

//! The base components of the Kuitui tree.
//!
//! These are the widgets a user interface is built from. They are
//! *containers*: they do not own their children, they lay out the
//! children the tree gave them. That is the same split Compose makes
//! between a composable that emits children and the layout that
//! arranges them.
//!
//! | Widget | Arranges its children |
//! | --- | --- |
//! | [`Column`] | one below the other |
//! | [`Row`] | one next to the other |
//! | [`Stack`] | all of them on top of each other |
//! | [`Frame`] | a single child, aligned in the available space |
//! | [`Spacer`] | nothing, it only takes space |
//! | [`Text`] | nothing, it draws a run of text |

pub mod text;

use alloc::vec::Vec;

use crate::geometry::{Alignment, Constraints, Offset, Rect, Size};
use crate::widget::{MeasureScope, PlaceScope, Widget};

pub use text::Text;

/// Lays its children out one below the other.
///
/// The children are measured with the width of the column and an
/// unbounded height. The column is as wide as its widest child and as
/// tall as the sum of its children and the spacing between them.
///
/// A child that claims a share of the free space — see
/// [`Widget::flex`] — takes it from the height of the column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Column {
    /// The space between two children.
    spacing: f32,
    /// The horizontal alignment of the children.
    alignment: Alignment,
}

impl Column {
    /// Creates a column without spacing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            spacing: 0.0,
            alignment: Alignment::TOP_START,
        }
    }

    /// The column with `spacing` pixels between two children.
    #[must_use]
    pub const fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// The column with the given horizontal alignment.
    ///
    /// Only the horizontal component of the alignment is used: the
    /// children of a column are always stacked vertically.
    #[must_use]
    pub const fn alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }
}

impl Default for Column {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Column {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let count = scope.child_count();
        if count == 0 {
            return constraints.enforce(Size::ZERO);
        }
        let gaps = self.spacing * (count as f32 - 1.0).max(0.0);
        let child_constraints = Constraints::new(0.0, constraints.max_width, 0.0, f32::INFINITY);
        let mut sizes = Vec::with_capacity(count);
        let mut flexes = Vec::with_capacity(count);
        for index in 0..count {
            sizes.push(scope.measure_child(index, child_constraints));
            flexes.push(scope.child_flex(index));
        }

        let mut total = gaps;
        let mut total_flex = 0.0;
        for index in 0..count {
            total += sizes[index].height;
            if flexes[index] > 0.0 {
                total_flex += flexes[index];
            }
        }

        // Hand the free space of the column to the flexible children,
        // in proportion to the share each of them claims.
        if constraints.has_bounded_height() && total_flex > 0.0 {
            let free = constraints.max_height - total;
            if free > 0.0 {
                for index in 0..count {
                    if flexes[index] <= 0.0 {
                        continue;
                    }
                    let share = free * flexes[index] / total_flex;
                    let allocated = Size::new(sizes[index].width, sizes[index].height + share);
                    let _ = scope.measure_child(index, Constraints::tight(allocated));
                    let measured = scope.child_size(index);
                    total += measured.height - sizes[index].height;
                    sizes[index] = measured;
                }
            }
        }

        let width = sizes
            .iter()
            .map(|size| size.width)
            .fold(0.0_f32, f32::max);
        constraints.enforce(Size::new(width, total))
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        let count = scope.child_count();
        let mut y = bounds.top();
        for index in 0..count {
            let size = scope.child_size(index);
            let x = self
                .alignment
                .align_x(bounds.origin, bounds.size.width, size.width);
            scope.place_child(index, Offset::new(x - bounds.left(), y - bounds.top()));
            y += size.height + self.spacing;
        }
    }
}

/// Lays its children out one next to the other.
///
/// The children are measured with an unbounded width and the height of
/// the row. The row is as wide as the sum of its children and the
/// spacing between them, and as tall as its tallest child.
///
/// A child that claims a share of the free space — see [`Widget::flex`]
/// — takes it from the width of the row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    /// The space between two children.
    spacing: f32,
    /// The vertical alignment of the children.
    alignment: Alignment,
}

impl Row {
    /// Creates a row without spacing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            spacing: 0.0,
            alignment: Alignment::TOP_START,
        }
    }

    /// The row with `spacing` pixels between two children.
    #[must_use]
    pub const fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// The row with the given vertical alignment.
    ///
    /// Only the vertical component of the alignment is used: the
    /// children of a row are always laid out one after the other.
    #[must_use]
    pub const fn alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Row {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let count = scope.child_count();
        if count == 0 {
            return constraints.enforce(Size::ZERO);
        }
        let gaps = self.spacing * (count as f32 - 1.0).max(0.0);
        let child_constraints = Constraints::new(0.0, f32::INFINITY, 0.0, constraints.max_height);
        let mut sizes = Vec::with_capacity(count);
        let mut flexes = Vec::with_capacity(count);
        for index in 0..count {
            sizes.push(scope.measure_child(index, child_constraints));
            flexes.push(scope.child_flex(index));
        }

        let mut total = gaps;
        let mut total_flex = 0.0;
        for index in 0..count {
            total += sizes[index].width;
            if flexes[index] > 0.0 {
                total_flex += flexes[index];
            }
        }

        if constraints.has_bounded_width() && total_flex > 0.0 {
            let free = constraints.max_width - total;
            if free > 0.0 {
                for index in 0..count {
                    if flexes[index] <= 0.0 {
                        continue;
                    }
                    let share = free * flexes[index] / total_flex;
                    let allocated = Size::new(sizes[index].width + share, sizes[index].height);
                    let _ = scope.measure_child(index, Constraints::tight(allocated));
                    let measured = scope.child_size(index);
                    total += measured.width - sizes[index].width;
                    sizes[index] = measured;
                }
            }
        }

        let height = sizes
            .iter()
            .map(|size| size.height)
            .fold(0.0_f32, f32::max);
        constraints.enforce(Size::new(total, height))
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        let count = scope.child_count();
        let mut x = bounds.left();
        for index in 0..count {
            let size = scope.child_size(index);
            let y = self
                .alignment
                .align_y(bounds.origin, bounds.size.height, size.height);
            scope.place_child(index, Offset::new(x - bounds.left(), y - bounds.top()));
            x += size.width + self.spacing;
        }
    }
}

/// Lays its children out on top of each other.
///
/// The stack is as big as its biggest child and every child is aligned
/// inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stack {
    /// The alignment of the children.
    alignment: Alignment,
}

impl Stack {
    /// Creates a stack whose children are aligned at the top left.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alignment: Alignment::TOP_START,
        }
    }

    /// The stack with the given alignment.
    #[must_use]
    pub const fn alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Stack {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let count = scope.child_count();
        let mut width = 0.0_f32;
        let mut height = 0.0_f32;
        for index in 0..count {
            let size = scope.measure_child(index, constraints);
            width = width.max(size.width);
            height = height.max(size.height);
        }
        constraints.enforce(Size::new(width, height))
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        let count = scope.child_count();
        for index in 0..count {
            let size = scope.child_size(index);
            let x = self
                .alignment
                .align_x(bounds.origin, bounds.size.width, size.width);
            let y = self
                .alignment
                .align_y(bounds.origin, bounds.size.height, size.height);
            scope.place_child(index, Offset::new(x - bounds.left(), y - bounds.top()));
        }
    }
}

/// Holds a single child and aligns it in the available space.
///
/// The frame is as big as the space it is given and the child is
/// aligned inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// The alignment of the child.
    alignment: Alignment,
}

impl Frame {
    /// Creates a frame whose child is aligned at the top left.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alignment: Alignment::TOP_START,
        }
    }

    /// The frame with the given alignment.
    #[must_use]
    pub const fn alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Frame {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let size = scope.measure_child(0, constraints);
        constraints.enforce(size)
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        let size = scope.child_size(0);
        let x = self
            .alignment
            .align_x(bounds.origin, bounds.size.width, size.width);
        let y = self
            .alignment
            .align_y(bounds.origin, bounds.size.height, size.height);
        scope.place_child(0, Offset::new(x - bounds.left(), y - bounds.top()));
    }
}

/// Takes space and draws nothing.
///
/// A spacer with a flex factor takes its share of the free space of
/// its parent, which is how the children of a row or a column are
/// pushed apart.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Spacer {
    /// The width of the spacer.
    width: f32,
    /// The height of the spacer.
    height: f32,
    /// The share of the free space the spacer takes.
    flex: f32,
}

impl Spacer {
    /// Creates a spacer without a size.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            width: 0.0,
            height: 0.0,
            flex: 0.0,
        }
    }

    /// The spacer with the given width.
    #[must_use]
    pub const fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// The spacer with the given height.
    #[must_use]
    pub const fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// The spacer with the given share of the free space of its
    /// parent.
    #[must_use]
    pub const fn flex(mut self, flex: f32) -> Self {
        self.flex = flex;
        self
    }
}

impl Widget for Spacer {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::new(self.width, self.height))
    }

    fn flex(&self) -> f32 {
        self.flex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::theme::Theme;

    /// A widget that asks for a fixed size.
    struct Fixed {
        width: f32,
        height: f32,
    }

    impl Widget for Fixed {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(self.width, self.height))
        }
    }

    fn fixed(width: f32, height: f32) -> Node {
        Node::new(Fixed { width, height })
    }

    #[test]
    fn a_column_is_as_wide_as_its_widest_child() {
        let theme = Theme::light();
        let mut node = Node::new(Column::new())
            .child(fixed(10.0, 5.0))
            .child(fixed(30.0, 5.0));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(30.0, 10.0));
    }

    #[test]
    fn a_column_adds_the_spacing_between_its_children() {
        let theme = Theme::light();
        let mut node = Node::new(Column::new().spacing(4.0))
            .child(fixed(10.0, 5.0))
            .child(fixed(10.0, 5.0));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(10.0, 14.0));
    }

    #[test]
    fn a_row_is_as_wide_as_the_sum_of_its_children() {
        let theme = Theme::light();
        let mut node = Node::new(Row::new())
            .child(fixed(10.0, 5.0))
            .child(fixed(20.0, 5.0));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(30.0, 5.0));
    }

    #[test]
    fn a_flexible_child_takes_the_free_space_of_its_parent() {
        let theme = Theme::light();
        let mut node = Node::new(Row::new())
            .child(fixed(10.0, 5.0))
            .child(Node::new(Spacer::new().flex(1.0)));
        let size = node.measure(&theme, Constraints::new(0.0, 50.0, 0.0, f32::INFINITY));
        assert_eq!(size.width, 50.0);
    }

    #[test]
    fn two_flexible_children_share_the_free_space() {
        let theme = Theme::light();
        let mut node = Node::new(Row::new())
            .child(Node::new(Spacer::new().flex(1.0)))
            .child(Node::new(Spacer::new().flex(3.0)));
        let size = node.measure(&theme, Constraints::new(0.0, 40.0, 0.0, f32::INFINITY));
        assert_eq!(size.width, 40.0);
        assert_eq!(node.children()[0].size().width, 10.0);
        assert_eq!(node.children()[1].size().width, 30.0);
    }

    #[test]
    fn a_stack_is_as_big_as_its_biggest_child() {
        let theme = Theme::light();
        let mut node = Node::new(Stack::new())
            .child(fixed(10.0, 5.0))
            .child(fixed(30.0, 20.0));
        let size = node.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(30.0, 20.0));
    }

    #[test]
    fn a_column_places_its_children_one_below_the_other() {
        let theme = Theme::light();
        let mut node = Node::new(Column::new().spacing(2.0))
            .child(fixed(10.0, 5.0))
            .child(fixed(10.0, 5.0));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(20.0, 20.0)));
        assert_eq!(node.children()[0].bounds().origin, Offset::new(0.0, 0.0));
        assert_eq!(node.children()[1].bounds().origin, Offset::new(0.0, 7.0));
    }

    #[test]
    fn a_row_aligns_its_children_vertically() {
        let theme = Theme::light();
        let mut node = Node::new(Row::new().alignment(Alignment::BOTTOM)).child(fixed(10.0, 5.0));
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(20.0, 20.0)));
        assert_eq!(node.children()[0].bounds().origin, Offset::new(0.0, 15.0));
    }
}
