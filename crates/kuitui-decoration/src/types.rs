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

//! The types shared by every decoration and every backend.
//!
//! These are the values a decoration carries: its identity, its kind, its
//! bounds, its metrics, its icon and its resize edges. They are
//! deliberately plain data so a backend can copy them into a native
//! structure without a conversion step.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_kuitui::geometry::{Constraints, Offset, Rect, Size};

/// The identity of a decoration among the decorations of one window.
///
/// Two decorations that represent the same native object — for example
/// the title bar of the same window on Wayland and on Win32 — carry the
/// same id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DecorationId(u64);

impl DecorationId {
    /// Creates a decoration id from a native handle.
    #[must_use]
    pub const fn new(handle: u64) -> Self {
        Self(handle)
    }

    /// The native handle the id wraps.
    #[inline]
    #[must_use]
    pub const fn handle(self) -> u64 {
        self.0
    }
}

impl Default for DecorationId {
    fn default() -> Self {
        Self(0)
    }
}

/// The kind of a decoration: which part of a window it decorates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecorationKind {
    /// The title bar of the window.
    TitleBar,
    /// The frame that wraps the content and holds the title bar.
    WindowFrame,
    /// The window icon, drawn in the title bar and in the task bar.
    WindowIcon,
    /// The mouse cursor that hovers over the window.
    MouseCursor,
    /// One edge or corner that resizes the window.
    ResizeEdge,
    /// The theme of the window: its colors, its type scale and its fonts.
    Theme,
    /// The whole decorated window: frame, title bar, icon and edges.
    Window,
}

/// The bounds of a decoration, in pixels of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecorationBounds {
    /// The x coordinate of the left edge.
    pub x: i32,
    /// The y coordinate of the top edge.
    pub y: i32,
    /// The width of the decoration.
    pub width: u32,
    /// The height of the decoration.
    pub height: u32,
}

impl DecorationBounds {
    /// Empty bounds at the origin.
    pub const EMPTY: Self = Self::new(0, 0, 0, 0);

    /// Creates bounds at the origin with the given size.
    #[must_use]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The rectangle the bounds describe.
    #[must_use]
    pub const fn rect(self) -> Rect {
        Rect::from_min_size(Offset::new(self.x as f32, self.y as f32), self.width as f32, self.height as f32)
    }

    /// Whether the bounds are empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl Default for DecorationBounds {
    fn default() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

/// The metrics a decoration asks the windowing system for.
///
/// These are the values the native API reports: the height of the title
/// bar, the thickness of the resize border, the size of the icons and the
/// double-click distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecorationMetrics {
    /// The height of the title bar, in pixels.
    pub title_bar_height: u32,
    /// The thickness of the resize border, in pixels.
    pub border_thickness: u32,
    /// The size of a small icon, in pixels.
    pub small_icon_size: u32,
    /// The size of a large icon, in pixels.
    pub large_icon_size: u32,
    /// The distance, in pixels, within which two clicks form a double-click.
    pub double_click_distance: u32,
    /// The time, in milliseconds, within which two clicks form a double-click.
    pub double_click_time: u32,
}

impl Default for DecorationMetrics {
    fn default() -> Self {
        Self {
            title_bar_height: 32,
            border_thickness: 6,
            small_icon_size: 16,
            large_icon_size: 32,
            double_click_distance: 4,
            double_click_time: 500,
        }
    }
}

/// The icon of a window, as a decoded image.
#[derive(Debug, Clone)]
pub struct DecorationIcon {
    /// The width of the icon, in pixels.
    pub width: u32,
    /// The height of the icon, in pixels.
    pub height: u32,
    /// The pixels of the icon, in premultiplied RGBA8 order.
    pub pixels: Vec<u8>,
}

impl DecorationIcon {
    /// Creates an empty icon.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        }
    }

    /// Creates an icon from raw premultiplied RGBA8 pixels.
    #[must_use]
    pub fn from_rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Whether the icon holds any pixels.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }
}

impl Default for DecorationIcon {
    fn default() -> Self {
        Self::new()
    }
}

/// One edge or corner of a window that a resize handle covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecorationResizeEdge {
    /// The top edge.
    Top,
    /// The bottom edge.
    Bottom,
    /// The left edge.
    Left,
    /// The right edge.
    Right,
    /// The top-left corner.
    TopLeft,
    /// The top-right corner.
    TopRight,
    /// The bottom-left corner.
    BottomLeft,
    /// The bottom-right corner.
    BottomRight,
}

impl DecorationResizeEdge {
    /// The thickness of the handle that covers the edge, in pixels.
    pub const THICKNESS: u32 = 6;

    /// Whether the edge is on the top of the window.
    #[must_use]
    pub const fn is_top(self) -> bool {
        matches!(self, Self::Top | Self::TopLeft | Self::TopRight)
    }

    /// Whether the edge is on the bottom of the window.
    #[must_use]
    pub const fn is_bottom(self) -> bool {
        matches!(self, Self::Bottom | Self::BottomLeft | Self::BottomRight)
    }

    /// Whether the edge is on the left of the window.
    #[must_use]
    pub const fn is_left(self) -> bool {
        matches!(self, Self::Left | Self::TopLeft | Self::BottomLeft)
    }

    /// Whether the edge is on the right of the window.
    #[must_use]
    pub const fn is_right(self) -> bool {
        matches!(self, Self::Right | Self::TopRight | Self::BottomRight)
    }
}

impl Default for DecorationResizeEdge {
    fn default() -> Self {
        Self::Top
    }
}

/// A measured decoration: its desired size and the constraints it was
/// measured under.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecorationMeasure {
    /// The size the decoration wants.
    pub size: Size,
    /// The constraints it was measured under.
    pub constraints: Constraints,
}

impl DecorationMeasure {
    /// Creates a measured decoration.
    #[must_use]
    pub const fn new(size: Size, constraints: Constraints) -> Self {
        Self { size, constraints }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_report_their_rect_and_emptiness() {
        let bounds = DecorationBounds::new(10, 20, 100, 50);
        assert_eq!(
            bounds.rect(),
            Rect::from_min_size(Offset::new(10.0, 20.0), 100.0, 50.0)
        );
        assert!(!bounds.is_empty());
        assert!(DecorationBounds::new(0, 0, 0, 50).is_empty());
        assert!(DecorationBounds::new(0, 0, 100, 0).is_empty());
    }

    #[test]
    fn edges_report_their_side() {
        assert!(DecorationResizeEdge::TopLeft.is_top());
        assert!(DecorationResizeEdge::TopLeft.is_left());
        assert!(!DecorationResizeEdge::TopLeft.is_bottom());
        assert!(!DecorationResizeEdge::TopLeft.is_right());
        assert!(DecorationResizeEdge::BottomRight.is_bottom());
        assert!(DecorationResizeEdge::BottomRight.is_right());
    }
}
