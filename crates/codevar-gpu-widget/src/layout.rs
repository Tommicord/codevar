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

//! Box constraints and axis types of the widget layout pass.
//!
//! Layout runs in two phases driven by [`crate::tree::WidgetTree`]:
//! a bottom-up measure phase that answers *how much space does this
//! subtree want* under a [`Constraints`], and a top-down arrange
//! phase that hands every subtree its final rectangle.

use crate::geometry::{Insets, Size};

/// The two directions of a one-dimensional layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Left to right.
    Horizontal,
    /// Top to bottom.
    Vertical,
}

impl Axis {
    /// The extent of `size` along this axis.
    #[must_use]
    pub const fn of(self, size: Size) -> f32 {
        match self {
            Self::Horizontal => size.width,
            Self::Vertical => size.height,
        }
    }

    /// `size` with the extent along this axis replaced by `value`.
    #[must_use]
    pub const fn with(self, size: Size, value: f32) -> Size {
        match self {
            Self::Horizontal => Size::new(value, size.height),
            Self::Vertical => Size::new(size.width, value),
        }
    }

    /// The other axis.
    #[must_use]
    pub const fn cross(self) -> Self {
        match self {
            Self::Horizontal => Self::Vertical,
            Self::Vertical => Self::Horizontal,
        }
    }
}

/// Lower and upper bound of a size handed to [`crate::widget::Widget::measure`].
///
/// A bound of [`f32::INFINITY`] means *unbounded*: the widget may be
/// as large as it wants on that axis. The pair is only valid when the
/// minimum never exceeds the maximum (see [`Constraints::is_valid`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constraints {
    /// Smallest acceptable size.
    pub min: Size,
    /// Largest acceptable size.
    pub max: Size,
}

impl Constraints {
    /// Constraints that accept any non-negative size.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            min: Size::ZERO,
            max: Size {
                width: f32::INFINITY,
                height: f32::INFINITY,
            },
        }
    }

    /// Constraints that only accept exactly `size`.
    #[must_use]
    pub const fn tight(size: Size) -> Self {
        Self { min: size, max: size }
    }

    /// Constraints that accept anything up to `size`.
    #[must_use]
    pub const fn loose(size: Size) -> Self {
        Self {
            min: Size::ZERO,
            max: size,
        }
    }

    /// Constraints from explicit bounds.
    #[must_use]
    pub const fn new(min: Size, max: Size) -> Self {
        Self { min, max }
    }

    /// Whether the bounds are usable: non-negative, finite minima,
    /// no NaN maxima and `min <= max` on both axes (maxima may be
    /// infinite).
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.min.is_valid()
            && self.max.width >= 0.0
            && self.max.height >= 0.0
            && !self.max.width.is_nan()
            && !self.max.height.is_nan()
            && self.min.width <= self.max.width
            && self.min.height <= self.max.height
    }

    /// Whether both axes have a finite maximum.
    #[must_use]
    pub fn is_bounded(self) -> bool {
        self.max.width.is_finite() && self.max.height.is_finite()
    }

    /// `size` moved into `[min, max]` on both axes.
    #[must_use]
    pub fn constrain(self, size: Size) -> Size {
        size.clamp(self.min, self.max)
    }

    /// The constraints of a box whose edges were moved inwards by
    /// `insets` (padding or margin).
    ///
    /// Infinite maxima stay infinite; the minimum is lowered to the
    /// maximum so the result is always valid.
    #[must_use]
    pub fn deflate(self, insets: Insets) -> Self {
        let horizontal = insets.horizontal();
        let vertical = insets.vertical();
        let max = Size::new(
            (self.max.width - horizontal).max(0.0),
            (self.max.height - vertical).max(0.0),
        );
        let min = Size::new(
            (self.min.width - horizontal).max(0.0),
            (self.min.height - vertical).max(0.0),
        );
        Self {
            min: Size::new(min.width.min(max.width), min.height.min(max.height)),
            max,
        }
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

    /// A tight constraint forces the exact size; a loose one only
    /// caps it; unbounded constraints never cap anything.
    #[test]
    fn constrain_respects_bounds() {
        let tight = Constraints::tight(Size::new(100.0, 50.0));
        assert_eq!(tight.constrain(Size::new(10.0, 999.0)), Size::new(100.0, 50.0));

        let loose = Constraints::loose(Size::new(100.0, 50.0));
        assert_eq!(loose.constrain(Size::new(10.0, 999.0)), Size::new(10.0, 50.0));

        let open = Constraints::unbounded();
        assert_eq!(open.constrain(Size::new(10.0, 999.0)), Size::new(10.0, 999.0));
    }

    /// Deflating keeps infinite maxima infinite, never produces a
    /// negative bound and keeps `min <= max`.
    #[test]
    fn deflate_keeps_valid_bounds() {
        let constraints = Constraints::new(Size::new(80.0, 40.0), Size::new(100.0, 50.0));
        let inner = constraints.deflate(Insets::all(10.0));
        assert_eq!(inner, Constraints::new(Size::new(60.0, 20.0), Size::new(80.0, 30.0)));

        let oversized = Constraints::tight(Size::new(5.0, 5.0)).deflate(Insets::all(10.0));
        assert_eq!(oversized, Constraints::tight(Size::ZERO));

        let open = Constraints::unbounded().deflate(Insets::all(10.0));
        assert_eq!(open.max.width, f32::INFINITY);
        assert!(open.is_valid());
    }

    /// The axis helpers read and write the matching extent.
    #[test]
    fn axis_selects_extent() {
        let size = Size::new(3.0, 4.0);
        assert_eq!(Axis::Horizontal.of(size), 3.0);
        assert_eq!(Axis::Vertical.of(size), 4.0);
        assert_eq!(Axis::Vertical.with(size, 9.0), Size::new(3.0, 9.0));
        assert_eq!(Axis::Horizontal.cross(), Axis::Vertical);
    }
}
