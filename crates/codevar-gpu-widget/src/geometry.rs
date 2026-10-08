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

//! Geometry primitives shared by layout, scenes and hit testing.
//!
//! Coordinates are logical pixels with the origin at the top-left
//! corner and the y axis pointing down, matching the viewport
//! convention used by the compositor pipes.

/// A point in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Position {
    /// Distance from the left edge.
    pub x: f32,
    /// Distance from the top edge.
    pub y: f32,
}

impl Position {
    /// The origin `(0, 0)`.
    pub const ZERO: Self = Self::new(0.0, 0.0);

    /// Creates a position from `x` and `y`.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Whether both components are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// An extent in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Size {
    /// Horizontal extent.
    pub width: f32,
    /// Vertical extent.
    pub height: f32,
}

impl Size {
    /// A size with both extents zero.
    pub const ZERO: Self = Self::new(0.0, 0.0);

    /// Creates a size from `width` and `height`.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// A square size of `value` per axis.
    #[must_use]
    pub const fn square(value: f32) -> Self {
        Self::new(value, value)
    }

    /// Whether both extents are finite and non-negative.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.width.is_finite() && self.height.is_finite() && self.width >= 0.0 && self.height >= 0.0
    }

    /// The size clamped into `[min, max]` per axis.
    ///
    /// The comparison is written with `max`/`min` instead of
    /// `f32::clamp` so an inverted bound never panics: the result
    /// always stays inside `max`.
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            width: self.width.max(min.width).min(max.width),
            height: self.height.max(min.height).min(max.height),
        }
    }
}

/// Insets from the four edges of a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Insets {
    /// Inset from the top edge.
    pub top: f32,
    /// Inset from the right edge.
    pub right: f32,
    /// Inset from the bottom edge.
    pub bottom: f32,
    /// Inset from the left edge.
    pub left: f32,
}

impl Insets {
    /// No insets at all.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    /// Creates insets from the four edges.
    #[must_use]
    pub const fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    /// The same inset on all four edges.
    #[must_use]
    pub const fn all(value: f32) -> Self {
        Self::new(value, value, value, value)
    }

    /// Equal top/bottom and equal left/right insets.
    #[must_use]
    pub const fn symmetric(vertical: f32, horizontal: f32) -> Self {
        Self::new(vertical, horizontal, vertical, horizontal)
    }

    /// Sum of the left and right insets.
    #[must_use]
    pub const fn horizontal(self) -> f32 {
        self.left + self.right
    }

    /// Sum of the top and bottom insets.
    #[must_use]
    pub const fn vertical(self) -> f32 {
        self.top + self.bottom
    }

    /// Whether every inset is finite and non-negative.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.top.is_finite()
            && self.right.is_finite()
            && self.bottom.is_finite()
            && self.left.is_finite()
            && self.top >= 0.0
            && self.right >= 0.0
            && self.bottom >= 0.0
            && self.left >= 0.0
    }
}

/// An axis-aligned rectangle in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Rect {
    /// Position of the top-left corner.
    pub origin: Position,
    /// Extent of the rectangle.
    pub size: Size,
}

impl Rect {
    /// A rectangle at the origin with zero extent.
    pub const ZERO: Self = Self {
        origin: Position::ZERO,
        size: Size::ZERO,
    };

    /// Creates a rectangle from its top-left corner and extent.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            origin: Position::new(x, y),
            size: Size::new(width, height),
        }
    }

    /// A rectangle at `origin` with `size`.
    #[must_use]
    pub const fn from_origin(origin: Position, size: Size) -> Self {
        Self { origin, size }
    }

    /// A rectangle at the origin with `size`.
    #[must_use]
    pub const fn from_size(size: Size) -> Self {
        Self {
            origin: Position::ZERO,
            size,
        }
    }

    /// Distance from the left edge to the right edge.
    #[must_use]
    pub fn right(self) -> f32 {
        self.origin.x + self.size.width
    }

    /// Distance from the top edge to the bottom edge.
    #[must_use]
    pub fn bottom(self) -> f32 {
        self.origin.y + self.size.height
    }

    /// Center point of the rectangle.
    #[must_use]
    pub fn center(self) -> Position {
        Position::new(
            self.origin.x + self.size.width * 0.5,
            self.origin.y + self.size.height * 0.5,
        )
    }

    /// Whether `point` lies inside the rectangle (inclusive of the
    /// top-left edge, exclusive of the bottom-right edge).
    #[must_use]
    pub fn contains(self, point: Position) -> bool {
        point.x >= self.origin.x
            && point.y >= self.origin.y
            && point.x < self.right()
            && point.y < self.bottom()
    }

    /// Whether the rectangle has no area.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.size.width <= 0.0 || self.size.height <= 0.0
    }

    /// Whether both the origin and the extent are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.origin.is_finite() && self.size.is_valid()
    }

    /// Shrinks the rectangle by `insets` on each edge.
    ///
    /// The extent never becomes negative: insets larger than the
    /// rectangle collapse it to an empty rectangle at its center.
    #[must_use]
    pub fn deflate(self, insets: Insets) -> Self {
        let width = (self.size.width - insets.horizontal()).max(0.0);
        let height = (self.size.height - insets.vertical()).max(0.0);
        Self::new(
            self.origin.x + insets.left,
            self.origin.y + insets.top,
            width,
            height,
        )
    }

    /// Grows the rectangle by `insets` on each edge.
    #[must_use]
    pub fn inflate(self, insets: Insets) -> Self {
        Self::new(
            self.origin.x - insets.left,
            self.origin.y - insets.top,
            self.size.width + insets.horizontal(),
            self.size.height + insets.vertical(),
        )
    }

    /// Intersection of two rectangles, or an empty rectangle when
    /// they do not overlap.
    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        let x0 = self.origin.x.max(other.origin.x);
        let y0 = self.origin.y.max(other.origin.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        if x1 <= x0 || y1 <= y0 {
            return Self::ZERO;
        }
        Self::new(x0, y0, x1 - x0, y1 - y0)
    }

    /// Smallest rectangle containing both inputs.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let x0 = self.origin.x.min(other.origin.x);
        let y0 = self.origin.y.min(other.origin.y);
        let x1 = self.right().max(other.right());
        let y1 = self.bottom().max(other.bottom());
        Self::new(x0, y0, x1 - x0, y1 - y0)
    }
}

/// An RGBA color with components in `[0.0, 1.0]`.
///
/// Colors travel to the renderer straight (non premultiplied); the
/// blend state of the widget pipeline handles coverage.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Color {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
    /// Alpha component.
    pub a: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);

    /// An opaque color from `r`, `g` and `b`.
    #[must_use]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::rgba(r, g, b, 1.0)
    }

    /// A color from `r`, `g`, `b` and `a`.
    #[must_use]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// The color as a `[r, g, b, a]` array for push constants.
    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// The same color with a different alpha channel.
    #[must_use]
    pub const fn with_alpha(self, a: f32) -> Self {
        Self::rgba(self.r, self.g, self.b, a)
    }

    /// Whether every component is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.r.is_finite() && self.g.is_finite() && self.b.is_finite() && self.a.is_finite()
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::TRANSPARENT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deflating insets the rectangle on every edge and never yields a
    /// negative extent, even for insets larger than the rectangle.
    #[test]
    fn deflate_shrinks_and_clamps() {
        let rect = Rect::new(10.0, 20.0, 100.0, 40.0);
        let shrunk = rect.deflate(Insets::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(shrunk, Rect::new(14.0, 21.0, 94.0, 36.0));
        let collapsed = rect.deflate(Insets::all(1_000.0));
        assert!(collapsed.is_empty());
        assert_eq!(collapsed.origin.x, 1010.0);
    }

    /// Intersection of disjoint rectangles is empty, overlapping ones
    /// clip to the shared region, and union grows to cover both.
    #[test]
    fn intersect_and_union() {
        let left = Rect::new(0.0, 0.0, 10.0, 10.0);
        let right = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(left.intersect(right), Rect::new(5.0, 5.0, 5.0, 5.0));
        assert_eq!(left.union(right), Rect::new(0.0, 0.0, 15.0, 15.0));
        let far = Rect::new(100.0, 100.0, 1.0, 1.0);
        assert!(left.intersect(far).is_empty());
        assert_eq!(left.union(far), Rect::new(0.0, 0.0, 101.0, 101.0));
    }

    /// Hit testing treats the top-left edge as inside and the
    /// bottom-right edge as outside, so siblings never both claim a
    /// point on their shared boundary.
    #[test]
    fn contains_is_half_open() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!(rect.contains(Position::new(0.0, 0.0)));
        assert!(rect.contains(Position::new(9.9, 9.9)));
        assert!(!rect.contains(Position::new(10.0, 5.0)));
        assert!(!rect.contains(Position::new(5.0, 10.0)));
        assert!(!rect.contains(Position::new(-1.0, 5.0)));
    }

    /// Clamping keeps a size inside its bounds per axis.
    #[test]
    fn size_clamp_bounds_each_axis() {
        let min = Size::new(10.0, 20.0);
        let max = Size::new(100.0, 200.0);
        assert_eq!(Size::new(5.0, 250.0).clamp(min, max), Size::new(10.0, 200.0));
        assert_eq!(Size::new(50.0, 50.0).clamp(min, max), Size::new(50.0, 50.0));
    }
}
