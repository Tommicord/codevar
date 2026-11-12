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

//! Geometry primitives shared by the tracing and fitting stages.
//!
//! Coordinates are in *document space*: the top-left pixel corner is
//! `(0.0, 0.0)`, `x` grows to the right and `y` grows downwards. A pixel with
//! integer coordinates `(x, y)` therefore covers the square
//! `[x, x + 1) x [y, y + 1)`. This convention makes contour vertices land on
//! integer coordinates without a half-pixel correction and keeps the sign of
//! polygon areas intuitive: a clockwise screen-space ring has positive area.
//!
//! All floating point helpers go through [`libm`] so the crate builds on
//! `no_std` targets. The small wrapper functions are intentionally `#[inline]`
//! because they sit inside the fitting inner loops.

use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use libm::{atan2, cos, fabs, hypot, sin};

/// Tolerance used when comparing coordinates that are expected to coincide.
pub const GEOMETRY_EPSILON: f64 = 1.0e-9;

/// A point or a displacement in document space.
///
/// The same type is used for positions and vectors. Keeping one type avoids
/// conversions at call sites and makes the fitting code read like the
/// mathematics it implements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

impl Point {
    /// The origin.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// A point at the given coordinates.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// A point with both coordinates set to `value`.
    #[inline]
    pub const fn splat(value: f64) -> Self {
        Self { x: value, y: value }
    }

    /// Whether both coordinates are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Scalar multiplication.
    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Two-dimensional cross product (`self x other`).
    #[inline]
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Squared Euclidean length.
    #[inline]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        hypot(self.x, self.y)
    }

    /// Squared distance to another point.
    #[inline]
    pub fn distance_sq(self, other: Self) -> f64 {
        (self - other).length_sq()
    }

    /// Distance to another point.
    #[inline]
    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    /// The unit vector with the same direction, or `None` for the zero vector
    /// and for non-finite input.
    #[inline]
    pub fn normalized(self) -> Option<Self> {
        if !self.is_finite() {
            return None;
        }
        let length = self.length();
        if length <= GEOMETRY_EPSILON {
            None
        } else {
            Some(self.scale(1.0 / length))
        }
    }

    /// Left-hand normal in the mathematical sense (`(-y, x)`), normalized when
    /// possible.
    #[inline]
    pub fn normal(self) -> Option<Self> {
        Self::new(-self.y, self.x).normalized()
    }

    /// Linear interpolation with `t` not clamped.
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    }

    /// Rotates the point around the origin by `radians`.
    #[inline]
    pub fn rotated(self, radians: f64) -> Self {
        let (s, c) = (sin(radians), cos(radians));
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }

    /// The angle of the vector measured from the positive x axis.
    #[inline]
    pub fn angle(self) -> f64 {
        atan2(self.y, self.x)
    }

    /// Unsigned angle between two vectors in `[0, pi]`.
    ///
    /// Returns zero when either vector has no direction.
    #[inline]
    pub fn angle_between(self, other: Self) -> f64 {
        let denominator = self.length() * other.length();
        if denominator <= GEOMETRY_EPSILON {
            return 0.0;
        }
        let sine = self.cross(other) / denominator;
        let cosine = self.dot(other) / denominator;
        atan2(fabs(sine), cosine)
    }

    /// Signed angle from `self` to `other` in `(-pi, pi]`.
    #[inline]
    pub fn signed_angle_to(self, other: Self) -> f64 {
        atan2(self.cross(other), self.dot(other))
    }

    /// Converts the point to a tuple.
    #[inline]
    pub const fn to_tuple(self) -> (f64, f64) {
        (self.x, self.y)
    }
}

impl Add for Point {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Point {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Point {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Point::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Point {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f64> for Point {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<Point> for f64 {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: Point) -> Self::Output {
        rhs.scale(self)
    }
}

impl Div<f64> for Point {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self.scale(1.0 / rhs)
    }
}

impl Neg for Point {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Point::new(-self.x, -self.y)
    }
}

/// Axis-aligned integer rectangle measured in whole pixels.
///
/// `right` and `bottom` are exclusive. The rectangle is empty when either
/// extent is zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PixelRect {
    /// Left edge, inclusive.
    pub left: u32,
    /// Top edge, inclusive.
    pub top: u32,
    /// Right edge, exclusive.
    pub right: u32,
    /// Bottom edge, exclusive.
    pub bottom: u32,
}

impl PixelRect {
    /// A rectangle from inclusive-exclusive edges.
    #[inline]
    pub const fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// A rectangle covering a single point.
    #[inline]
    pub const fn single(x: u32, y: u32) -> Self {
        Self::new(x, y, x + 1, y + 1)
    }

    /// Width in pixels.
    #[inline]
    pub const fn width(self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    /// Height in pixels.
    #[inline]
    pub const fn height(self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }

    /// Number of pixels covered by the rectangle.
    #[inline]
    pub const fn area(self) -> u64 {
        self.width() as u64 * self.height() as u64
    }

    /// Whether the rectangle contains no pixels.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    /// Whether `(x, y)` lies inside the rectangle.
    #[inline]
    pub const fn contains(self, x: u32, y: u32) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    /// Extends the rectangle so it contains `(x, y)`.
    #[inline]
    pub fn include(&mut self, x: u32, y: u32) {
        self.left = self.left.min(x);
        self.top = self.top.min(y);
        self.right = self.right.max(x + 1);
        self.bottom = self.bottom.max(y + 1);
    }

    /// Union of two rectangles. An empty rectangle acts as the identity.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self::new(
            self.left.min(other.left),
            self.top.min(other.top),
            self.right.max(other.right),
            self.bottom.max(other.bottom),
        )
    }

    /// The rectangle translated by `(dx, dy)` in document space.
    #[inline]
    pub fn translated(self, dx: f64, dy: f64) -> Bounds {
        Bounds::new(
            self.left as f64 + dx,
            self.top as f64 + dy,
            self.right as f64 + dx,
            self.bottom as f64 + dy,
        )
    }
}

/// Floating-point axis-aligned bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Bounds {
    /// Left edge.
    pub left: f64,
    /// Top edge.
    pub top: f64,
    /// Right edge.
    pub right: f64,
    /// Bottom edge.
    pub bottom: f64,
}

impl Bounds {
    /// Empty bounds that absorb the first point added to them.
    pub const EMPTY: Self = Self {
        left: f64::INFINITY,
        top: f64::INFINITY,
        right: f64::NEG_INFINITY,
        bottom: f64::NEG_INFINITY,
    };

    /// Bounds from explicit edges.
    #[inline]
    pub const fn new(left: f64, top: f64, right: f64, bottom: f64) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Bounds around a single point.
    #[inline]
    pub const fn from_point(point: Point) -> Self {
        Self {
            left: point.x,
            top: point.y,
            right: point.x,
            bottom: point.y,
        }
    }

    /// Whether the bounds are empty or non-finite.
    #[inline]
    pub fn is_empty(self) -> bool {
        !(self.left.is_finite() && self.top.is_finite() && self.right.is_finite() && self.bottom.is_finite())
            || self.left > self.right
            || self.top > self.bottom
    }

    /// Width, or zero for empty bounds.
    #[inline]
    pub fn width(self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.right - self.left
        }
    }

    /// Height, or zero for empty bounds.
    #[inline]
    pub fn height(self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.bottom - self.top
        }
    }

    /// Grows the bounds to include `point`.
    #[inline]
    pub fn include_point(&mut self, point: Point) {
        if point.is_finite() {
            self.left = self.left.min(point.x);
            self.top = self.top.min(point.y);
            self.right = self.right.max(point.x);
            self.bottom = self.bottom.max(point.y);
        }
    }

    /// Computes bounds around a sequence of points.
    pub fn from_points(points: &[Point]) -> Self {
        let mut bounds = Self::EMPTY;
        for &point in points {
            bounds.include_point(point);
        }
        bounds
    }
}

/// A cubic Bézier segment in absolute document coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct CubicBezier {
    /// Start point.
    pub start: Point,
    /// First control point.
    pub control1: Point,
    /// Second control point.
    pub control2: Point,
    /// End point.
    pub end: Point,
}

impl CubicBezier {
    /// A cubic from four absolute points.
    #[inline]
    pub const fn new(start: Point, control1: Point, control2: Point, end: Point) -> Self {
        Self {
            start,
            control1,
            control2,
            end,
        }
    }

    /// Evaluates the curve at `t`.
    #[inline]
    pub fn eval(self, t: f64) -> Point {
        let one_minus = 1.0 - t;
        let b0 = one_minus * one_minus * one_minus;
        let b1 = 3.0 * t * one_minus * one_minus;
        let b2 = 3.0 * t * t * one_minus;
        let b3 = t * t * t;
        Point::new(
            self.start.x * b0 + self.control1.x * b1 + self.control2.x * b2 + self.end.x * b3,
            self.start.y * b0 + self.control1.y * b1 + self.control2.y * b2 + self.end.y * b3,
        )
    }

    /// First derivative at `t`.
    #[inline]
    pub fn derivative(self, t: f64) -> Point {
        let one_minus = 1.0 - t;
        let d0 = 3.0 * one_minus * one_minus;
        let d1 = 6.0 * one_minus * t;
        let d2 = 3.0 * t * t;
        Point::new(
            (self.control1.x - self.start.x) * d0
                + (self.control2.x - self.control1.x) * d1
                + (self.end.x - self.control2.x) * d2,
            (self.control1.y - self.start.y) * d0
                + (self.control2.y - self.control1.y) * d1
                + (self.end.y - self.control2.y) * d2,
        )
    }

    /// Second derivative at `t`.
    #[inline]
    pub fn second_derivative(self, t: f64) -> Point {
        let one_minus = 1.0 - t;
        Point::new(
            6.0 * one_minus * (self.control2.x - 2.0 * self.control1.x + self.start.x)
                + 6.0 * t * (self.end.x - 2.0 * self.control2.x + self.control1.x),
            6.0 * one_minus * (self.control2.y - 2.0 * self.control1.y + self.start.y)
                + 6.0 * t * (self.end.y - 2.0 * self.control2.y + self.control1.y),
        )
    }

    /// A tangent direction at `t`, or `None` when the derivative vanishes.
    #[inline]
    pub fn tangent(self, t: f64) -> Option<Point> {
        self.derivative(t).normalized()
    }

    /// Splits the curve at `t`, returning the two halves.
    pub fn split(self, t: f64) -> (Self, Self) {
        let p01 = self.start.lerp(self.control1, t);
        let p12 = self.control1.lerp(self.control2, t);
        let p23 = self.control2.lerp(self.end, t);
        let p012 = p01.lerp(p12, t);
        let p123 = p12.lerp(p23, t);
        let middle = p012.lerp(p123, t);
        (
            Self::new(self.start, p01, p012, middle),
            Self::new(middle, p123, p23, self.end),
        )
    }

    /// Reverses the direction of the curve.
    #[inline]
    pub const fn reversed(self) -> Self {
        Self::new(self.end, self.control2, self.control1, self.start)
    }

    /// Whether every coordinate is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.start.is_finite()
            && self.control1.is_finite()
            && self.control2.is_finite()
            && self.end.is_finite()
    }

    /// Conservative bounds from the four control points.
    #[inline]
    pub fn control_bounds(self) -> Bounds {
        let mut bounds = Bounds::from_point(self.start);
        bounds.include_point(self.control1);
        bounds.include_point(self.control2);
        bounds.include_point(self.end);
        bounds
    }

    /// Maximum distance of the two control points from the chord.
    ///
    /// This is a cheap proxy used to decide whether a fitted cubic is
    /// indistinguishable from a straight line.
    pub fn flatness(self) -> f64 {
        let chord = self.end - self.start;
        let length = chord.length();
        if length <= GEOMETRY_EPSILON {
            return self
                .control1
                .distance(self.start)
                .max(self.control2.distance(self.start));
        }
        let normal = Point::new(-chord.y / length, chord.x / length);
        let d1 = fabs((self.control1 - self.start).dot(normal));
        let d2 = fabs((self.control2 - self.start).dot(normal));
        d1.max(d2)
    }

    /// Appends `samples` evenly spaced points along the curve to `output`.
    pub fn sample_into(self, samples: u32, output: &mut Vec<Point>) {
        if samples == 0 {
            return;
        }
        output.reserve(samples as usize + 1);
        output.push(self.start);
        for index in 1..=samples {
            let t = index as f64 / samples as f64;
            output.push(self.eval(t));
        }
    }
}

/// Clamps `value` to `[low, high]` without relying on `std`.
#[inline]
pub fn clamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}
