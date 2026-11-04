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

//! Geometry primitives shared by layout, painting and hit testing.
//!
//! Every length in Kuitui is a `f32` count of *logical* pixels. The
//! framework never rounds during layout: rounding happens once, when
//! the display list is converted into device pixels by the backend,
//! so a layout pass is exact and reproducible.

use core::fmt;

/// A two-dimensional size in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Size {
    /// Horizontal extent in pixels.
    pub width: f32,
    /// Vertical extent in pixels.
    pub height: f32,
}

impl Size {
    /// The size of an empty box.
    pub const ZERO: Size = Size {
        width: 0.0,
        height: 0.0,
    };

    /// A size whose extents are unbounded.
    pub const INFINITY: Size = Size {
        width: f32::INFINITY,
        height: f32::INFINITY,
    };

    /// Creates a size of `width` by `height` pixels.
    #[inline]
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Horizontal extent in pixels.
    #[inline]
    #[must_use]
    pub const fn width(self) -> f32 {
        self.width
    }

    /// Vertical extent in pixels.
    #[inline]
    #[must_use]
    pub const fn height(self) -> f32 {
        self.height
    }

    /// Whether both extents are zero or negative.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Whether both extents are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.width.is_finite() && self.height.is_finite()
    }

    /// The component-wise minimum of two sizes.
    #[inline]
    #[must_use]
    pub fn min(self, other: Size) -> Size {
        Size::new(self.width.min(other.width), self.height.min(other.height))
    }

    /// The component-wise maximum of two sizes.
    #[inline]
    #[must_use]
    pub fn max(self, other: Size) -> Size {
        Size::new(self.width.max(other.width), self.height.max(other.height))
    }

    /// Multiplies both extents by `factor`.
    #[inline]
    #[must_use]
    pub fn scale(self, factor: f32) -> Size {
        Size::new(self.width * factor, self.height * factor)
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// A two-dimensional position in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Offset {
    /// Horizontal position in pixels.
    pub x: f32,
    /// Vertical position in pixels.
    pub y: f32,
}

impl Offset {
    /// The origin of the coordinate space.
    pub const ZERO: Offset = Offset { x: 0.0, y: 0.0 };

    /// Creates an offset at `x` by `y`.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Horizontal position in pixels.
    #[inline]
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Vertical position in pixels.
    #[inline]
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Adds two offsets component-wise.
    #[inline]
    #[must_use]
    pub const fn add(self, other: Offset) -> Offset {
        Offset::new(self.x + other.x, self.y + other.y)
    }

    /// Subtracts two offsets component-wise.
    #[inline]
    #[must_use]
    pub const fn sub(self, other: Offset) -> Offset {
        Offset::new(self.x - other.x, self.y - other.y)
    }

    /// Multiplies both components by `factor`.
    #[inline]
    #[must_use]
    pub const fn scale(self, factor: f32) -> Offset {
        Offset::new(self.x * factor, self.y * factor)
    }

    /// The Euclidean distance between two offsets.
    #[inline]
    #[must_use]
    pub fn distance_to(self, other: Offset) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        libm_sqrt(dx * dx + dy * dy)
    }
}

impl fmt::Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// Computes a square root without pulling in `std`.
///
/// `f32::sqrt` is only available with `std`; the `libm` crate is the
/// `no_std` replacement and is already a workspace dependency of the
/// repository.
#[inline]
fn libm_sqrt(value: f32) -> f32 {
    libm::sqrtf(value)
}

/// An axis-aligned rectangle in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    /// The top-left corner of the rectangle.
    pub origin: Offset,
    /// The extent of the rectangle.
    pub size: Size,
}

impl Rect {
    /// An empty rectangle at the origin.
    pub const ZERO: Rect = Rect {
        origin: Offset::ZERO,
        size: Size::ZERO,
    };

    /// Creates a rectangle at `origin` with the given `size`.
    #[inline]
    #[must_use]
    pub const fn new(origin: Offset, size: Size) -> Self {
        Self { origin, size }
    }

    /// Creates a rectangle from its top-left corner and its extents.
    #[inline]
    #[must_use]
    pub const fn from_min_size(origin: Offset, width: f32, height: f32) -> Self {
        Self::new(origin, Size::new(width, height))
    }

    /// Distance between the left edge and the origin.
    #[inline]
    #[must_use]
    pub const fn left(self) -> f32 {
        self.origin.x
    }

    /// Distance between the top edge and the origin.
    #[inline]
    #[must_use]
    pub const fn top(self) -> f32 {
        self.origin.y
    }

    /// Distance between the left edge and the right edge.
    #[inline]
    #[must_use]
    pub const fn right(self) -> f32 {
        self.origin.x + self.size.width
    }

    /// Distance between the top edge and the bottom edge.
    #[inline]
    #[must_use]
    pub const fn bottom(self) -> f32 {
        self.origin.y + self.size.height
    }

    /// The geometric center of the rectangle.
    #[inline]
    #[must_use]
    pub fn center(self) -> Offset {
        Offset::new(
            self.origin.x + self.size.width * 0.5,
            self.origin.y + self.size.height * 0.5,
        )
    }

    /// Whether the rectangle has no area.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.size.is_empty()
    }

    /// Whether the point lies inside the rectangle.
    ///
    /// The left and top edges are inclusive, the right and bottom edges
    /// are exclusive, so a rectangle of zero width contains nothing.
    #[inline]
    #[must_use]
    pub fn contains(self, point: Offset) -> bool {
        point.x >= self.left() && point.x < self.right() && point.y >= self.top() && point.y < self.bottom()
    }

    /// Whether the other rectangle lies entirely inside this one.
    #[inline]
    #[must_use]
    pub fn contains_rect(self, other: Rect) -> bool {
        other.left() >= self.left()
            && other.top() >= self.top()
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// Whether the two rectangles share at least one point.
    #[inline]
    #[must_use]
    pub fn intersects(self, other: Rect) -> bool {
        self.left() < other.right()
            && other.left() < self.right()
            && self.top() < other.bottom()
            && other.top() < self.bottom()
    }

    /// The overlapping region of two rectangles, or [`Rect::ZERO`] when
    /// they do not overlap.
    #[must_use]
    pub fn intersection(self, other: Rect) -> Rect {
        if !self.intersects(other) {
            return Rect::ZERO;
        }
        let left = self.left().max(other.left());
        let top = self.top().max(other.top());
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        Rect::from_min_size(Offset::new(left, top), right - left, bottom - top)
    }

    /// The smallest rectangle containing both rectangles.
    #[must_use]
    pub fn union(self, other: Rect) -> Rect {
        let left = self.left().min(other.left());
        let top = self.top().min(other.top());
        let right = self.right().max(other.right());
        let bottom = self.bottom().max(other.bottom());
        Rect::from_min_size(Offset::new(left, top), right - left, bottom - top)
    }

    /// Grows the rectangle by `amount` on every side.
    #[must_use]
    pub fn inflate(self, amount: f32) -> Rect {
        Rect::from_min_size(
            Offset::new(self.left() - amount, self.top() - amount),
            self.size.width + amount * 2.0,
            self.size.height + amount * 2.0,
        )
    }

    /// Shrinks the rectangle by `amount` on every side.
    #[must_use]
    pub fn deflate(self, amount: f32) -> Rect {
        self.inflate(-amount)
    }

    /// Moves the rectangle by `delta`.
    #[must_use]
    pub fn translate(self, delta: Offset) -> Rect {
        Rect::new(self.origin.add(delta), self.size)
    }

    /// The rectangle with its origin at the coordinate-space origin.
    #[must_use]
    pub fn at_origin(self) -> Rect {
        Rect::new(Offset::ZERO, self.size)
    }
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.origin, self.size)
    }
}

/// Padding or border widths applied to the four edges of a box.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EdgeInsets {
    /// Width of the left edge.
    pub left: f32,
    /// Width of the top edge.
    pub top: f32,
    /// Width of the right edge.
    pub right: f32,
    /// Width of the bottom edge.
    pub bottom: f32,
}

impl EdgeInsets {
    /// No padding at all.
    pub const ZERO: EdgeInsets = EdgeInsets {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    /// Creates insets from the four edge widths.
    #[inline]
    #[must_use]
    pub const fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// The same width on all four edges.
    #[inline]
    #[must_use]
    pub const fn all(amount: f32) -> Self {
        Self::new(amount, amount, amount, amount)
    }

    /// The same width on the two horizontal edges.
    #[inline]
    #[must_use]
    pub const fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self::new(horizontal, vertical, horizontal, vertical)
    }

    /// The same width on the two vertical edges.
    #[inline]
    #[must_use]
    pub const fn only(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self::new(left, top, right, bottom)
    }

    /// The width of the two vertical edges.
    #[inline]
    #[must_use]
    pub const fn horizontal(self) -> f32 {
        self.left + self.right
    }

    /// The width of the two horizontal edges.
    #[inline]
    #[must_use]
    pub const fn vertical(self) -> f32 {
        self.top + self.bottom
    }

    /// Whether every edge is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.left == 0.0 && self.top == 0.0 && self.right == 0.0 && self.bottom == 0.0
    }
}

/// The horizontal component of an [`Alignment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Horizontal {
    /// Aligned with the left edge.
    Left,
    /// Aligned with the horizontal center.
    Center,
    /// Aligned with the right edge.
    Right,
}

/// The vertical component of an [`Alignment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vertical {
    /// Aligned with the top edge.
    Top,
    /// Aligned with the vertical center.
    Center,
    /// Aligned with the bottom edge.
    Bottom,
}

/// The alignment of a child inside the space its parent gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Alignment {
    /// The horizontal component.
    pub horizontal: Horizontal,
    /// The vertical component.
    pub vertical: Vertical,
}

impl Alignment {
    /// Aligns to the top-left corner.
    pub const TOP_START: Alignment = Alignment {
        horizontal: Horizontal::Left,
        vertical: Vertical::Top,
    };
    /// Aligns to the top center.
    pub const TOP: Alignment = Alignment {
        horizontal: Horizontal::Center,
        vertical: Vertical::Top,
    };
    /// Aligns to the top-right corner.
    pub const TOP_END: Alignment = Alignment {
        horizontal: Horizontal::Right,
        vertical: Vertical::Top,
    };
    /// Aligns to the middle-left.
    pub const START: Alignment = Alignment {
        horizontal: Horizontal::Left,
        vertical: Vertical::Center,
    };
    /// Aligns to the center.
    pub const CENTER: Alignment = Alignment {
        horizontal: Horizontal::Center,
        vertical: Vertical::Center,
    };
    /// Aligns to the middle-right.
    pub const END: Alignment = Alignment {
        horizontal: Horizontal::Right,
        vertical: Vertical::Center,
    };
    /// Aligns to the bottom-left corner.
    pub const BOTTOM_START: Alignment = Alignment {
        horizontal: Horizontal::Left,
        vertical: Vertical::Bottom,
    };
    /// Aligns to the bottom center.
    pub const BOTTOM: Alignment = Alignment {
        horizontal: Horizontal::Center,
        vertical: Vertical::Bottom,
    };
    /// Aligns to the bottom-right corner.
    pub const BOTTOM_END: Alignment = Alignment {
        horizontal: Horizontal::Right,
        vertical: Vertical::Bottom,
    };

    /// Creates an alignment from its two components.
    #[inline]
    #[must_use]
    pub const fn new(horizontal: Horizontal, vertical: Vertical) -> Self {
        Self { horizontal, vertical }
    }

    /// The horizontal offset that aligns a child of `size` inside
    /// `available` starting at `origin`.
    #[must_use]
    pub fn align_x(self, origin: Offset, available: f32, size: f32) -> f32 {
        match self.horizontal {
            Horizontal::Left => origin.x,
            Horizontal::Center => origin.x + (available - size) * 0.5,
            Horizontal::Right => origin.x + available - size,
        }
    }

    /// The vertical offset that aligns a child of `size` inside
    /// `available` starting at `origin`.
    #[must_use]
    pub fn align_y(self, origin: Offset, available: f32, size: f32) -> f32 {
        match self.vertical {
            Vertical::Top => origin.y,
            Vertical::Center => origin.y + (available - size) * 0.5,
            Vertical::Bottom => origin.y + available - size,
        }
    }
}

/// The bounds a widget must fit into during a measure pass.
///
/// The constraints are always consistent: `min_width <= max_width` and
/// `min_height <= max_height`. A maximum of [`f32::INFINITY`] means the
/// axis is unbounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constraints {
    /// Smallest allowed width.
    pub min_width: f32,
    /// Largest allowed width.
    pub max_width: f32,
    /// Smallest allowed height.
    pub min_height: f32,
    /// Largest allowed height.
    pub max_height: f32,
}

impl Constraints {
    /// Creates constraints from the four bounds.
    #[inline]
    #[must_use]
    pub const fn new(min_width: f32, max_width: f32, min_height: f32, max_height: f32) -> Self {
        Self {
            min_width,
            max_width,
            min_height,
            max_height,
        }
    }

    /// Constraints that force an exact size.
    #[inline]
    #[must_use]
    pub const fn fixed(width: f32, height: f32) -> Self {
        Self::new(width, width, height, height)
    }

    /// Constraints that force an exact size.
    #[inline]
    #[must_use]
    pub const fn tight(size: Size) -> Self {
        Self::fixed(size.width, size.height)
    }

    /// Constraints with no minimum and the given maxima.
    #[inline]
    #[must_use]
    pub const fn loose(max_width: f32, max_height: f32) -> Self {
        Self::new(0.0, max_width, 0.0, max_height)
    }

    /// Constraints without any bound.
    #[inline]
    #[must_use]
    pub const fn unbounded() -> Self {
        Self::new(0.0, f32::INFINITY, 0.0, f32::INFINITY)
    }

    /// Whether the width axis has an upper bound.
    #[inline]
    #[must_use]
    pub const fn has_bounded_width(&self) -> bool {
        self.max_width.is_finite()
    }

    /// Whether the height axis has an upper bound.
    #[inline]
    #[must_use]
    pub const fn has_bounded_height(&self) -> bool {
        self.max_height.is_finite()
    }

    /// Whether the width is forced to a single value.
    #[inline]
    #[must_use]
    pub const fn has_fixed_width(&self) -> bool {
        self.min_width == self.max_width
    }

    /// Whether the height is forced to a single value.
    #[inline]
    #[must_use]
    pub const fn has_fixed_height(&self) -> bool {
        self.min_height == self.max_height
    }

    /// Clamps `size` into the bounds of the constraints.
    #[must_use]
    pub fn enforce(&self, size: Size) -> Size {
        Size::new(
            size.width.clamp(self.min_width, self.max_width),
            size.height
                .clamp(self.min_height, self.max_height),
        )
    }

    /// The constraints with the given minimum applied.
    #[must_use]
    pub fn with_min_width(mut self, min_width: f32) -> Self {
        self.min_width = min_width.min(self.max_width);
        self
    }

    /// The constraints with the given minimum applied.
    #[must_use]
    pub fn with_min_height(mut self, min_height: f32) -> Self {
        self.min_height = min_height.min(self.max_height);
        self
    }

    /// The constraints with the given maximum applied.
    #[must_use]
    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_width = max_width.max(self.min_width);
        self
    }

    /// The constraints with the given maximum applied.
    #[must_use]
    pub fn with_max_height(mut self, max_height: f32) -> Self {
        self.max_height = max_height.max(self.min_height);
        self
    }
}

impl Default for Constraints {
    fn default() -> Self {
        Self::unbounded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rect_contains_its_own_origin_but_not_its_excluded_edges() {
        let rect = Rect::from_min_size(Offset::new(10.0, 20.0), 30.0, 40.0);
        assert!(rect.contains(Offset::new(10.0, 20.0)));
        assert!(rect.contains(Offset::new(39.0, 59.0)));
        assert!(!rect.contains(Offset::new(40.0, 60.0)));
        assert!(!rect.contains(Offset::new(9.0, 20.0)));
    }

    #[test]
    fn intersection_of_disjoint_rectangles_is_empty() {
        let a = Rect::from_min_size(Offset::ZERO, 10.0, 10.0);
        let b = Rect::from_min_size(Offset::new(20.0, 20.0), 10.0, 10.0);
        assert!(!a.intersects(b));
        assert_eq!(a.intersection(b), Rect::ZERO);
    }

    #[test]
    fn intersection_keeps_the_overlapping_region() {
        let a = Rect::from_min_size(Offset::ZERO, 10.0, 10.0);
        let b = Rect::from_min_size(Offset::new(5.0, 5.0), 10.0, 10.0);
        assert_eq!(
            a.intersection(b),
            Rect::from_min_size(Offset::new(5.0, 5.0), 5.0, 5.0)
        );
    }

    #[test]
    fn union_covers_both_rectangles() {
        let a = Rect::from_min_size(Offset::ZERO, 10.0, 10.0);
        let b = Rect::from_min_size(Offset::new(20.0, 20.0), 10.0, 10.0);
        assert_eq!(a.union(b), Rect::from_min_size(Offset::ZERO, 30.0, 30.0));
    }

    #[test]
    fn inflate_grows_and_deflate_shrinks_by_the_same_amount() {
        let rect = Rect::from_min_size(Offset::new(10.0, 10.0), 10.0, 10.0);
        let grown = rect.inflate(5.0);
        assert_eq!(grown.left(), 5.0);
        assert_eq!(grown.size, Size::new(20.0, 20.0));
        assert_eq!(grown.deflate(5.0), rect);
    }

    #[test]
    fn constraints_enforce_their_bounds() {
        let constraints = Constraints::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(constraints.enforce(Size::new(0.0, 0.0)), Size::new(10.0, 30.0));
        assert_eq!(
            constraints.enforce(Size::new(100.0, 100.0)),
            Size::new(20.0, 40.0)
        );
        assert_eq!(constraints.enforce(Size::new(15.0, 35.0)), Size::new(15.0, 35.0));
    }

    #[test]
    fn tight_and_fixed_constraints_agree() {
        let size = Size::new(12.0, 34.0);
        assert_eq!(Constraints::tight(size), Constraints::fixed(12.0, 34.0));
        assert!(Constraints::tight(size).has_fixed_width());
        assert!(Constraints::tight(size).has_fixed_height());
    }

    #[test]
    fn unbounded_constraints_accept_any_finite_size() {
        let constraints = Constraints::unbounded();
        assert!(!constraints.has_bounded_width());
        assert!(!constraints.has_bounded_height());
        assert_eq!(constraints.enforce(Size::new(1e9, 1e9)), Size::new(1e9, 1e9));
    }

    #[test]
    fn alignment_places_a_child_in_the_expected_corner() {
        let origin = Offset::new(100.0, 200.0);
        assert_eq!(Alignment::TOP_START.align_x(origin, 50.0, 10.0), 100.0);
        assert_eq!(Alignment::TOP_START.align_y(origin, 50.0, 10.0), 200.0);
        assert_eq!(Alignment::CENTER.align_x(origin, 50.0, 10.0), 120.0);
        assert_eq!(Alignment::CENTER.align_y(origin, 50.0, 10.0), 220.0);
        assert_eq!(Alignment::BOTTOM_END.align_x(origin, 50.0, 10.0), 140.0);
        assert_eq!(Alignment::BOTTOM_END.align_y(origin, 50.0, 10.0), 240.0);
    }

    #[test]
    fn edge_insets_sum_the_opposite_edges() {
        let insets = EdgeInsets::symmetric(4.0, 8.0);
        assert_eq!(insets.horizontal(), 8.0);
        assert_eq!(insets.vertical(), 16.0);
        assert!(!insets.is_zero());
        assert!(EdgeInsets::ZERO.is_zero());
    }

    #[test]
    fn distance_between_offsets_is_euclidean() {
        let distance = Offset::new(0.0, 0.0).distance_to(Offset::new(3.0, 4.0));
        assert!((distance - 5.0).abs() < 1e-5);
    }
}
