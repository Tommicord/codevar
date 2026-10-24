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

use alloc::boxed::Box;
use alloc::collections::{VecDeque, vec_deque};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;

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
///
/// A display list is *open*: kuitui stores boxed operations and never
/// inspects them, and a backend replays them by *visiting* each one with
/// a [`DrawVisitor`]. A new kind of primitive is therefore added by
/// implementing this trait and pushing the value — neither kuitui nor
/// the backends that do not care about it change.
///
/// The [`Any`] and [`fmt::Debug`] supertraits let a consumer identify
/// and format an operation without the core having to name every kind.
pub trait Draw: Any + fmt::Debug {
    /// Dispatches this operation to the matching method of `visitor`.
    fn visit(&self, visitor: &mut dyn DrawVisitor);
}

/// A consumer of the operations of a [`DrawList`].
///
/// A backend implements this trait to turn operations into its own
/// drawing model — Vulkan vertices, WebGPU quads, an SVG stream. Every
/// method has a default no-op implementation, so a backend that only
/// supports some primitives overrides only those.
pub trait DrawVisitor {
    /// Visits a rectangle, filled or outlined.
    fn rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) {
        let _ = (rect, radii, paint);
    }

    /// Visits a straight line.
    fn line(&mut self, from: Offset, to: Offset, paint: Paint) {
        let _ = (from, to, paint);
    }

    /// Visits a run of text.
    fn text(&mut self, origin: Offset, text: &Rc<str>, style: &TextStyle) {
        let _ = (origin, text, style);
    }
}

/// A filled or outlined, possibly rounded, rectangle.
#[derive(Debug)]
pub struct RectOp {
    /// The rectangle.
    pub rect: Rect,
    /// The radii of its corners.
    pub radii: CornerRadii,
    /// The paint it is drawn with.
    pub paint: Paint,
}

impl Draw for RectOp {
    #[inline]
    fn visit(&self, visitor: &mut dyn DrawVisitor) {
        visitor.rect(self.rect, self.radii, self.paint);
    }
}

/// A straight line between two points.
#[derive(Debug)]
pub struct LineOp {
    /// The start of the line.
    pub from: Offset,
    /// The end of the line.
    pub to: Offset,
    /// The paint it is drawn with.
    pub paint: Paint,
}

impl Draw for LineOp {
    #[inline]
    fn visit(&self, visitor: &mut dyn DrawVisitor) {
        visitor.line(self.from, self.to, self.paint);
    }
}

/// A run of text: its baseline origin, its characters and its style.
#[derive(Debug)]
pub struct TextOp {
    /// The baseline origin of the run.
    pub origin: Offset,
    /// The characters of the run.
    pub text: Rc<str>,
    /// The style of the run.
    pub style: TextStyle,
}

impl Draw for TextOp {
    #[inline]
    fn visit(&self, visitor: &mut dyn DrawVisitor) {
        visitor.text(self.origin, &self.text, &self.style);
    }
}

/// One queued drawing operation together with the clip that was open
/// when it was recorded.
///
/// A command is the unit a backend drains from the [`DrawList`]: the
/// operation and its resolved clip travel together, so a consumer can
/// never observe one without the other the way two parallel arrays
/// would allow.
#[derive(Debug)]
pub struct DrawCommand {
    /// The operation to replay.
    pub op: Box<dyn Draw>,
    /// The clipping rectangle that applied when the operation was
    /// recorded; `None` means the whole surface.
    pub clip: Option<Rect>,
}

/// The FIFO queue of drawing operations of one frame.
///
/// Kuitui records a laid-out tree into a queue and a backend drains it in
/// paint order: [`DrawList::push`] appends to the back and
/// [`DrawList::pop_front`] hands the oldest command to the renderer.
/// Recording and replaying therefore share one ordered structure, which
/// is what lets a compositor pipe consume a frame in order without
/// copying it.
///
/// A list is reused across frames: [`DrawList::clear`] empties it
/// without giving the memory back, so a steady-state frame performs no
/// allocation at all.
#[derive(Debug, Default)]
pub struct DrawList {
    /// The queued commands, oldest first (the queue).
    commands: VecDeque<DrawCommand>,
    /// The rectangles of the open clips.
    clips: Vec<Rect>,
}

impl DrawList {
    /// Creates an empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            commands: VecDeque::new(),
            clips: Vec::new(),
        }
    }

    /// Empties the queue, keeping its memory.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.clips.clear();
    }

    /// The operations of the queue, in paint order.
    pub fn ops(&self) -> impl Iterator<Item = &dyn Draw> + '_ {
        self.commands.iter().map(|command| &*command.op)
    }

    /// The commands of the queue, in paint order.
    pub fn commands(&self) -> impl Iterator<Item = &DrawCommand> + '_ {
        self.commands.iter()
    }

    /// The number of commands in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the queue holds no command.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Removes and returns the oldest command, or `None` when the queue
    /// is empty.
    ///
    /// This is the FIFO drain a backend replay uses: commands come out in
    /// the order they were recorded.
    pub fn pop_front(&mut self) -> Option<DrawCommand> {
        self.commands.pop_front()
    }

    /// The oldest command without removing it.
    #[must_use]
    pub fn peek_front(&self) -> Option<&DrawCommand> {
        self.commands.front()
    }

    /// Removes every command in paint order, leaving the clip stack
    /// untouched.
    pub fn drain(&mut self) -> impl Iterator<Item = DrawCommand> + '_ {
        self.commands.drain(..)
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

    /// Enqueues an operation with the clip that is open right now.
    pub fn push(&mut self, op: impl Draw + 'static) {
        self.push_boxed(Box::new(op));
    }

    /// Enqueues an already-boxed operation with the clip that is open
    /// right now.
    ///
    /// This is the storage-level entry point for a consumer that already
    /// holds a trait object; [`DrawList::push`] is the boxing shorthand.
    pub fn push_boxed(&mut self, op: Box<dyn Draw>) {
        let clip = self.current_clip();
        self.commands.push_back(DrawCommand { op, clip });
    }

    /// Records a rectangle.
    pub fn rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) {
        self.push(RectOp { rect, radii, paint });
    }

    /// Records a line.
    pub fn line(&mut self, from: Offset, to: Offset, paint: Paint) {
        self.push(LineOp { from, to, paint });
    }

    /// Records a run of text.
    pub fn text(&mut self, origin: Offset, text: Rc<str>, style: TextStyle) {
        self.push(TextOp { origin, text, style });
    }

    /// The operations of the list together with the clipping rectangle
    /// that applies to each of them.
    ///
    /// The clip of an operation is the intersection of every clip that
    /// was open when it was recorded.
    pub fn clipped_ops(&self) -> ClippedOps<'_> {
        ClippedOps {
            commands: self.commands.iter(),
        }
    }
}

/// The iterator returned by [`DrawList::clipped_ops`].
pub struct ClippedOps<'a> {
    /// The queued commands.
    commands: vec_deque::Iter<'a, DrawCommand>,
}

impl<'a> Iterator for ClippedOps<'a> {
    type Item = (&'a dyn Draw, Option<Rect>);

    fn next(&mut self) -> Option<Self::Item> {
        let command = self.commands.next()?;
        Some((&*command.op, command.clip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::any::Any;

    /// Downcasts an operation to its concrete type, the test-side
    /// counterpart of the `Draw` trait's openness.
    fn as_op<T: Any>(op: &dyn Draw) -> Option<&T> {
        let any: &dyn Any = op;
        any.downcast_ref::<T>()
    }

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
        let first = list
            .ops()
            .next()
            .expect("a rectangle was recorded");
        assert!(as_op::<RectOp>(first).is_some());
        let second = list.ops().nth(1).expect("a line was recorded");
        assert!(as_op::<LineOp>(second).is_some());
    }

    #[test]
    fn commands_drain_in_fifo_order() {
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
        let front = list
            .peek_front()
            .expect("a rectangle is at the front");
        assert!(as_op::<RectOp>(&*front.op).is_some());
        let first = list
            .pop_front()
            .expect("a rectangle was recorded");
        assert!(as_op::<RectOp>(&*first.op).is_some());
        let second = list.pop_front().expect("a line was recorded");
        assert!(as_op::<LineOp>(&*second.op).is_some());
        assert!(list.pop_front().is_none());
        assert!(list.is_empty());
    }

    #[test]
    fn a_command_carries_the_clip_that_was_open_when_it_was_recorded() {
        let mut list = DrawList::new();
        let clip = Rect::from_min_size(Offset::new(5.0, 5.0), 20.0, 20.0);
        list.push_clip(clip);
        list.rect(
            Rect::from_min_size(Offset::ZERO, 100.0, 100.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );
        list.pop_clip();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );

        let commands: Vec<DrawCommand> = list.drain().collect();
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].clip, Some(clip));
        assert_eq!(commands[1].clip, None);
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
