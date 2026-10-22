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

//! The display list: the flat sequence of drawing primitives a backend
//! replays.
//!
//! Kuitui never talks to a graphics API. A laid-out tree is recorded
//! into a [`DrawList`], and a backend turns that list into GPU work:
//! `codevar-gpu-core` replays it with Vulkan, `codevar-gpu-wg` replays
//! it with WebGPU. Because both consume the same list, a user
//! interface looks the same on the desktop and in a browser.
//!
//! The list is a *display list*, not a command buffer: it is a flat
//! sequence of primitives in paint order, with the clipping rectangle
//! that applies to each of them already resolved. A backend therefore
//! never has to model a clip stack.

use alloc::rc::Rc;
use alloc::vec::Vec;

use crate::color::Color;
use crate::geometry::{Offset, Rect};
use crate::text::TextStyle;

/// The radii of the four corners of a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CornerRadii {
    /// The radius of the top-left corner.
    pub top_left: f32,
    /// The radius of the top-right corner.
    pub top_right: f32,
    /// The radius of the bottom-right corner.
    pub bottom_right: f32,
    /// The radius of the bottom-left corner.
    pub bottom_left: f32,
}

impl CornerRadii {
    /// Sharp corners.
    pub const ZERO: CornerRadii = CornerRadii {
        top_left: 0.0,
        top_right: 0.0,
        bottom_right: 0.0,
        bottom_left: 0.0,
    };

    /// The same radius on all four corners.
    #[must_use]
    pub const fn uniform(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        }
    }

    /// Creates radii from the four corner values.
    #[must_use]
    pub const fn new(top_left: f32, top_right: f32, bottom_right: f32, bottom_left: f32) -> Self {
        Self {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        }
    }

    /// Whether every corner is sharp.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.top_left == 0.0 && self.top_right == 0.0 && self.bottom_right == 0.0 && self.bottom_left == 0.0
    }
}

/// How a shape is painted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaintStyle {
    /// The shape is filled with the color of the paint.
    Fill,
    /// The shape is outlined with the color of the paint, the given
    /// number of pixels wide.
    Stroke(f32),
}

/// The color and the style a shape is painted with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paint {
    /// The color of the shape.
    pub color: Color,
    /// Whether the shape is filled or outlined.
    pub style: PaintStyle,
}

impl Paint {
    /// A paint that fills a shape.
    #[must_use]
    pub const fn fill(color: Color) -> Self {
        Self {
            color,
            style: PaintStyle::Fill,
        }
    }

    /// A paint that outlines a shape.
    #[must_use]
    pub const fn stroke(color: Color, width: f32) -> Self {
        Self {
            color,
            style: PaintStyle::Stroke(width),
        }
    }

    /// The width of the outline, or `0.0` for a fill.
    #[must_use]
    pub const fn stroke_width(&self) -> f32 {
        match self.style {
            PaintStyle::Fill => 0.0,
            PaintStyle::Stroke(width) => width,
        }
    }
}

/// One drawing operation of a display list.
#[derive(Debug, Clone, PartialEq)]
pub enum DrawOp {
    /// A rectangle, filled or outlined.
    Rect {
        /// The rectangle.
        rect: Rect,
        /// The radii of its corners.
        radii: CornerRadii,
        /// The paint it is drawn with.
        paint: Paint,
    },
    /// A straight line.
    Line {
        /// The start of the line.
        from: Offset,
        /// The end of the line.
        to: Offset,
        /// The paint it is drawn with.
        paint: Paint,
    },
    /// A run of text.
    Text {
        /// The baseline origin of the run.
        origin: Offset,
        /// The characters of the run.
        text: Rc<str>,
        /// The style of the run.
        style: TextStyle,
    },
}

/// The list of drawing operations of one frame.
///
/// A list is reused across frames: [`DrawList::clear`] empties it
/// without giving the memory back, so a steady-state frame performs no
/// allocation at all.
#[derive(Debug, Default)]
pub struct DrawList {
    /// The operations, in paint order.
    ops: Vec<DrawOp>,
    /// The clipping rectangle of each operation, in the same order.
    op_clips: Vec<Option<Rect>>,
    /// The rectangles of the open clips.
    clips: Vec<Rect>,
}

impl DrawList {
    /// Creates an empty list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ops: Vec::new(),
            op_clips: Vec::new(),
            clips: Vec::new(),
        }
    }

    /// Empties the list, keeping its memory.
    pub fn clear(&mut self) {
        self.ops.clear();
        self.op_clips.clear();
        self.clips.clear();
    }

    /// The operations of the list, in paint order.
    #[must_use]
    pub fn ops(&self) -> &[DrawOp] {
        &self.ops
    }

    /// The number of operations of the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the list holds no operation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The number of clips that are currently open.
    #[must_use]
    pub fn clip_depth(&self) -> usize {
        self.clips.len()
    }

    /// The clipping rectangle that applies to the operations recorded
    /// from now on.
    ///
    /// Without an open clip the whole surface is visible.
    #[must_use]
    pub fn current_clip(&self) -> Option<Rect> {
        self.clips.last().copied()
    }

    /// Restricts the following operations to `rect`.
    ///
    /// The clips nest: an operation is only visible inside the
    /// intersection of every open clip.
    pub fn push_clip(&mut self, rect: Rect) {
        self.clips.push(match self.current_clip() {
            Some(current) => current.intersection(rect),
            None => rect,
        });
    }

    /// Ends the innermost clip.
    ///
    /// Popping more clips than were pushed is a no-op, so a backend
    /// never has to guard against an unbalanced list.
    pub fn pop_clip(&mut self) {
        self.clips.pop();
    }

    /// Records an operation.
    pub fn push(&mut self, op: DrawOp) {
        let clip = self.current_clip();
        self.ops.push(op);
        self.op_clips.push(clip);
    }

    /// Records a rectangle.
    pub fn rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) {
        self.push(DrawOp::Rect { rect, radii, paint });
    }

    /// Records a line.
    pub fn line(&mut self, from: Offset, to: Offset, paint: Paint) {
        self.push(DrawOp::Line { from, to, paint });
    }

    /// Records a run of text.
    pub fn text(&mut self, origin: Offset, text: Rc<str>, style: TextStyle) {
        self.push(DrawOp::Text { origin, text, style });
    }

    /// The operations of the list together with the clipping rectangle
    /// that applies to each of them.
    ///
    /// The clip of an operation is the intersection of every clip that
    /// was open when it was recorded.
    pub fn clipped_ops(&self) -> ClippedOps<'_> {
        ClippedOps {
            ops: self.ops.iter(),
            clips: self.op_clips.iter(),
        }
    }
}

/// The iterator returned by [`DrawList::clipped_ops`].
pub struct ClippedOps<'a> {
    /// The operations of the list.
    ops: core::slice::Iter<'a, DrawOp>,
    /// The clipping rectangle of each operation.
    clips: core::slice::Iter<'a, Option<Rect>>,
}

impl<'a> Iterator for ClippedOps<'a> {
    type Item = (&'a DrawOp, Option<Rect>);

    fn next(&mut self) -> Option<Self::Item> {
        let op = self.ops.next()?;
        let clip = self.clips.next().copied().flatten();
        Some((op, clip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_filled_paint_has_no_stroke_width() {
        let paint = Paint::fill(Color::RED);
        assert_eq!(paint.stroke_width(), 0.0);
        assert_eq!(paint.style, PaintStyle::Fill);
    }

    #[test]
    fn a_stroked_paint_carries_its_width() {
        let paint = Paint::stroke(Color::RED, 2.5);
        assert_eq!(paint.stroke_width(), 2.5);
    }

    #[test]
    fn a_list_records_its_operations_in_order() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );
        list.line(
            Offset::ZERO,
            Offset::new(1.0, 1.0),
            Paint::stroke(Color::WHITE, 1.0),
        );
        assert_eq!(list.len(), 2);
        assert!(matches!(list.ops()[0], DrawOp::Rect { .. }));
        assert!(matches!(list.ops()[1], DrawOp::Line { .. }));
    }

    #[test]
    fn clips_nest_by_intersection() {
        let mut list = DrawList::new();
        list.push_clip(Rect::from_min_size(Offset::ZERO, 100.0, 100.0));
        assert_eq!(
            list.current_clip(),
            Some(Rect::from_min_size(Offset::ZERO, 100.0, 100.0))
        );
        list.push_clip(Rect::from_min_size(Offset::new(50.0, 50.0), 100.0, 100.0));
        assert_eq!(
            list.current_clip(),
            Some(Rect::from_min_size(Offset::new(50.0, 50.0), 50.0, 50.0))
        );
        list.pop_clip();
        assert_eq!(
            list.current_clip(),
            Some(Rect::from_min_size(Offset::ZERO, 100.0, 100.0))
        );
    }

    #[test]
    fn popping_more_clips_than_pushed_is_harmless() {
        let mut list = DrawList::new();
        list.pop_clip();
        assert_eq!(list.clip_depth(), 0);
        assert_eq!(list.current_clip(), None);
    }

    #[test]
    fn clearing_a_list_keeps_it_usable() {
        let mut list = DrawList::new();
        list.line(Offset::ZERO, Offset::ZERO, Paint::fill(Color::BLACK));
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.clip_depth(), 0);
    }
}
