//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Replay of a kuitui [`DrawList`] into WebGPU-ready vertices.
//!
//! This is the `wasm32` counterpart of `codevar-gpu-vk`'s `ui` module: the
//! display list kuitui records is drained in FIFO order and expanded into a
//! triangle list ([`UiVertex`]) with one [`DrawSpan`] per operation, so the
//! pipe can bind the operation's resolved clip as a scissor without
//! re-deriving the clip stack.
//!
//! The vertex layout is the input of the `ui.vert`/`ui.frag` pair compiled by
//! `codevar-kuitui`: a `vec2` position at location 0 and a `vec4` color at
//! location 1, packed without padding. Because both backends read the same
//! embedded SPIR-V, both also push the same block, which is why
//! [`EffectParams`] here keeps the field order of `ui_effect.vert.glsl` and
//! why [`EFFECT_PUSH_WORDS`] counts its 32-bit words instead of its bytes:
//! `wgpu-hal` moves immediate data as words, `ash` as bytes.
//!
//! # Scope
//!
//! Geometry operations ([`RectOp`] and [`LineOp`]) are expanded into vertices;
//! a [`TextOp`] run carries no polygon geometry and is drawn by the text pass
//! of `codevar-kuitui-text`, so it is left to the visitor's default no-op.
//! Rounded corners are not representable by the color-only `ui` shader and
//! a command that carries an [`Effect`] is the exception — the span keeps
//! the effect, the pipe binds the `ui_effect` program for it, and the
//! rectangle is widened to the coverage of the layer so the blurred halo
//! has fragments to shade. A rounded rectangle joins it: it is expanded as
//! a layer of that program instead of being drawn square, so its corners
//! and its outline come from the same signed distance the Vulkan backend
//! cuts them from. A clip that names corners joins it the same way, as the
//! layer of the span when the operation carries none; where it does, one
//! span is cut by one box, so only a layer that paints the whole clip takes
//! its corners and the rest of the clip falls back to its scissor rectangle.
//!
//! # Target
//!
//! The expansion, the effect block and the scissor arithmetic are
//! target-independent, so their tests run on the host. The pipe that consumes
//! them is `UiPipe`, built on `wasm32` only like the rest of the backend.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use libm::{ceilf, floorf};

use codevar_kuitui::color::Color;
use codevar_kuitui::effect::Effect;
use codevar_kuitui::geometry::{Offset, Rect};
use codevar_kuitui::paint::{CornerRadii, DrawCommand, DrawList, DrawVisitor, Paint, PaintStyle};

#[cfg(target_arch = "wasm32")]
pub use crate::ui::pipe::UiPipe;

/// Default number of vertices a `UiPipe` vertex buffer can hold.
pub const DEFAULT_VERTEX_CAPACITY: usize = 64 * 1024;

// The immediate blocks below are pushed by `UiPipe`, which exists on
// `wasm32` only, so they are compiled there and under `test`, where the host
// checks their layout against the shader.

/// Byte size of the `ui.vert` immediate block (`vec2 viewport`).
#[cfg(target_arch = "wasm32")]
const VIEWPORT_PUSH_BYTES: u32 = 8;

/// Byte size of the `ui_effect` immediate block.
#[cfg(any(target_arch = "wasm32", test))]
const EFFECT_PUSH_BYTES: u32 = core::mem::size_of::<EffectParams>() as u32;

/// Number of 32-bit words in the `ui_effect` immediate block.
#[cfg(any(target_arch = "wasm32", test))]
const EFFECT_PUSH_WORDS: usize = (EFFECT_PUSH_BYTES / 4) as usize;

/// A position (in logical pixels) and color of one UI vertex.
///
/// The layout is the input of `ui.vert`: a `vec2` at location 0 and a
/// `vec4` at location 1, packed without padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiVertex {
    /// Position in logical pixels, top-left origin.
    pub position: [f32; 2],
    /// Straight (non-premultiplied) RGBA color.
    pub color: [f32; 4],
}

/// One operation's slice of a [`UiGeometry`] vertex buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawSpan {
    /// Index of the first vertex of the operation.
    pub first_vertex: u32,
    /// Number of vertices (a multiple of three) to draw.
    pub vertex_count: u32,
    /// Clip resolved by kuitui when the operation was recorded; `None`
    /// means the whole surface.
    pub clip: Option<Rect>,
    /// Layer rasterizing the span: the effect the command carried, or the
    /// rounded box its clip asked for when the operation had none.
    ///
    /// A scissor only knows rectangles, so the corners of a clip travel to
    /// the signed-distance program the way those of a rectangle do. One span
    /// is cut by one box, so a layer of its own keeps the rounding it asked
    /// for, unless it paints the whole clip. `None` means the plain `ui`
    /// program draws the span.
    pub effect: Option<Effect>,
}

/// The triangle list and draw spans of one replayed display list.
///
/// Reuse a single value across frames: [`UiGeometry::expand`] clears and
/// refills it, so a steady-state frame keeps the allocations.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct UiGeometry {
    /// Triangle-list vertices of every geometry operation.
    vertices: Vec<UiVertex>,
    /// One span per geometry operation, in paint order.
    spans: Vec<DrawSpan>,
    /// The effect of the command being expanded, which decides how wide a
    /// quad a rectangle needs.
    effect: Option<Effect>,
}

impl UiGeometry {
    /// Creates an empty geometry buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            spans: Vec::new(),
            effect: None,
        }
    }

    /// Empties the geometry, keeping its allocations.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.spans.clear();
        self.effect = None;
    }

    /// The triangle-list vertices, in paint order.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        &self.vertices
    }

    /// The draw spans, in paint order.
    #[inline]
    #[must_use]
    pub fn spans(&self) -> &[DrawSpan] {
        &self.spans
    }

    /// Whether the geometry holds no draw span.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Drains `list` in FIFO order and expands every command into it.
    ///
    /// The list is left empty: the queue is a one-shot hand-off from the UI
    /// side to the pipe, which clears and re-records it next frame.
    pub fn expand(&mut self, list: &mut DrawList) {
        self.clear();
        while let Some(command) = list.pop_front() {
            self.push_command(command);
        }
    }

    /// Expands one command, appending its vertices and its span.
    ///
    /// The operation is replayed through [`DrawVisitor`]: dispatching on its
    /// kind is the operation's job, not the backend's. The effect of the
    /// command is open while the visitor runs, so a rectangle knows how wide
    /// a quad the shader needs.
    pub fn push_command(&mut self, command: DrawCommand) {
        let first_vertex = self.vertices.len();
        let clip = command.clip;
        self.effect = command.effect;
        command.op.visit(self);
        // The visitor sees the open layer and may replace it: a rounded
        // rectangle asks the signed-distance program to cut its corners,
        // so the layer the span carries is the one the visitor left.
        let effect = self.effect.take();
        let vertex_count = self.vertices.len() - first_vertex;
        if vertex_count == 0 {
            return;
        }
        self.spans.push(DrawSpan {
            first_vertex: first_vertex as u32,
            vertex_count: vertex_count as u32,
            clip,
            effect: rounded_clip(effect, clip, command.clip_radii),
        });
    }

    /// Expands a rectangle with the flat program: a filled quad, or four
    /// quads for an outline.
    ///
    /// The corners of the operation are sharp; a rounded rectangle goes
    /// through [`UiGeometry::push_rounded_rect`], which asks the signed
    /// distance of the layer to cut them.
    fn push_rect(&mut self, rect: Rect, paint: Paint) {
        let color = vertex_color(paint);
        let width = paint.stroke_width();
        if width <= 0.0 || matches!(paint.style, PaintStyle::Fill) {
            // The effect program shades from the signed distance of the
            // layer, so a blurred or rotated layer needs the fragments of
            // its halo, not only its rectangle.
            let rect = match self.effect {
                Some(effect) => rect.union(effect.coverage()),
                None => rect,
            };
            self.push_quad(
                Offset::new(rect.left(), rect.top()),
                Offset::new(rect.right(), rect.top()),
                Offset::new(rect.right(), rect.bottom()),
                Offset::new(rect.left(), rect.bottom()),
                color,
            );
            return;
        }
        // An outline wider than half the rectangle would invert itself;
        // clamp it so the four edges keep their orientation.
        let width = width
            .min(rect.size.width * 0.5)
            .min(rect.size.height * 0.5);
        if width <= 0.0 {
            return;
        }
        let (left, top, right, bottom) = (rect.left(), rect.top(), rect.right(), rect.bottom());
        // Top, bottom, left and right edges, each an axis-aligned quad.
        self.push_quad(
            Offset::new(left, top),
            Offset::new(right, top),
            Offset::new(right, top + width),
            Offset::new(left, top + width),
            color,
        );
        self.push_quad(
            Offset::new(left, bottom - width),
            Offset::new(right, bottom - width),
            Offset::new(right, bottom),
            Offset::new(left, bottom),
            color,
        );
        self.push_quad(
            Offset::new(left, top + width),
            Offset::new(left + width, top + width),
            Offset::new(left + width, bottom - width),
            Offset::new(left, bottom - width),
            color,
        );
        self.push_quad(
            Offset::new(right - width, top + width),
            Offset::new(right, top + width),
            Offset::new(right, bottom - width),
            Offset::new(right - width, bottom - width),
            color,
        );
    }

    /// Expands a rounded rectangle through the signed-distance program.
    ///
    /// A curve is a property of the rasterization, not of the geometry, so
    /// the operation is one quad the layer cuts: the corners come from
    /// `radii` and an outline from the stroke width of the paint, which
    /// `ui_effect.frag.glsl` subtracts as the band between the box and the
    /// box inset by that width. A layer the command already carries keeps
    /// its own bounds and its blur, rotation or ramp; the radii of the
    /// operation are what the recorded rectangle asked for.
    fn push_rounded_rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) {
        let color = vertex_color(paint);
        let width = paint.stroke_width();
        let layer = self
            .effect
            .unwrap_or_else(|| Effect::new(rect))
            .shape(radii);
        let layer = if width > 0.0 { layer.stroke(width) } else { layer };
        self.effect = Some(layer);
        let rect = rect.union(layer.coverage());
        self.push_quad(
            Offset::new(rect.left(), rect.top()),
            Offset::new(rect.right(), rect.top()),
            Offset::new(rect.right(), rect.bottom()),
            Offset::new(rect.left(), rect.bottom()),
            color,
        );
    }

    /// Expands a line into the quad `from`–`to` spans at the paint width.
    fn push_line(&mut self, from: Offset, to: Offset, paint: Paint) {
        let color = vertex_color(paint);
        let width = if paint.stroke_width() > 0.0 {
            paint.stroke_width()
        } else {
            1.0
        };
        let delta = to.sub(from);
        let length = from.distance_to(to);
        if length <= f32::EPSILON {
            return;
        }
        // Unit normal of the line, scaled to half the stroke width.
        let half = width * 0.5;
        let normal = Offset::new(-delta.y / length * half, delta.x / length * half);
        let start = from.sub(normal);
        let end = to.sub(normal);
        self.push_quad(
            start,
            end,
            end.add(normal.scale(2.0)),
            start.add(normal.scale(2.0)),
            color,
        );
    }

    /// Appends the two triangles `a, b, c` and `a, c, d`.
    fn push_quad(&mut self, a: Offset, b: Offset, c: Offset, d: Offset, color: [f32; 4]) {
        self.vertices.push(UiVertex {
            position: [a.x, a.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [b.x, b.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [c.x, c.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [a.x, a.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [c.x, c.y],
            color,
        });
        self.vertices.push(UiVertex {
            position: [d.x, d.y],
            color,
        });
    }
}

impl DrawVisitor for UiGeometry {
    fn rect(&mut self, rect: Rect, radii: CornerRadii, paint: Paint) {
        if radii.is_zero() {
            self.push_rect(rect, paint);
        } else {
            self.push_rounded_rect(rect, radii, paint);
        }
    }

    fn line(&mut self, from: Offset, to: Offset, paint: Paint) {
        self.push_line(from, to, paint);
    }
}

/// The straight RGBA vertex color of a paint.
#[inline]
fn vertex_color(paint: Paint) -> [f32; 4] {
    channels(paint.color)
}

/// The RGBA components of a color, in the order the shaders read them.
#[inline]
fn channels(color: Color) -> [f32; 4] {
    [color.red, color.green, color.blue, color.alpha]
}

/// The layer that cuts a span to the rounded box its clip named.
///
/// A scissor only knows rectangles, so the corners of a clip travel to the
/// signed-distance program the way those of a rectangle do. One span is cut
/// by one box: the layer of an operation keeps its own rounding, and only a
/// layer that names none *and* covers the whole clip — the canvas that
/// paints the box it decorates — takes the corners of the clip. A sharp
/// clip, or one that never left the surface, needs no program at all.
#[inline]
fn rounded_clip(effect: Option<Effect>, clip: Option<Rect>, radii: CornerRadii) -> Option<Effect> {
    if radii.is_zero() {
        return effect;
    }
    match (effect, clip) {
        (None, Some(rect)) => Some(Effect::new(rect).shape(radii)),
        (Some(layer), Some(rect)) if layer.radii.is_zero() && layer.bounds == rect => {
            Some(layer.shape(radii))
        }
        (layer, _) => layer,
    }
}

/// The immediate data of the `ui_effect` shader pair.
///
/// The field order is the `PushConstants` block of
/// `ui_effect.vert.glsl` and `ui_effect.frag.glsl`, and it is the same block
/// the Vulkan backend pushes; a test pins every offset so reordering a field
/// cannot silently hand the shader a rotated pixel grid.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(any(target_arch = "wasm32", test))]
struct EffectParams {
    /// The color at the start of the ramp; unread when the ramp bit is
    /// clear, because then the fragment keeps the vertex color.
    color_from: [f32; 4],
    /// The color at the end of the ramp.
    color_to: [f32; 4],
    /// Size of the render target, the viewport mapping of the vertex stage.
    viewport: [f32; 2],
    /// Top-left of the layer rectangle.
    origin: [f32; 2],
    /// Size of the layer rectangle.
    size: [f32; 2],
    /// The point of the layer that carries `color_from`.
    axis_start: [f32; 2],
    /// The point of the layer that carries `color_to`.
    axis_end: [f32; 2],
    /// The blur sigma and the pixelate grid, in logical pixels.
    blur_pixel: [f32; 2],
    /// The corner radii: top-left, top-right, bottom-right, bottom-left.
    radii: [f32; 4],
    /// The rotation of the layer around its center, in radians.
    rotation: f32,
    /// The [`codevar_kuitui::effect::EffectKinds`] bits the fragment stage
    /// tests.
    kinds: u32,
    /// The width of the outline, in logical pixels; unread when the stroke
    /// bit is clear, because then the fragment keeps the whole rounded box.
    stroke: f32,
}

#[cfg(any(target_arch = "wasm32", test))]
impl EffectParams {
    /// The block as the 32-bit words `CommandEncoder::set_immediates` takes.
    ///
    /// Each member keeps the offset it has in the shader: the words are the
    /// bit patterns of the floats and of the kind bits, in field order.
    fn words(&self) -> [u32; EFFECT_PUSH_WORDS] {
        let groups: [&[f32]; 9] = [
            &self.color_from,
            &self.color_to,
            &self.viewport,
            &self.origin,
            &self.size,
            &self.axis_start,
            &self.axis_end,
            &self.blur_pixel,
            &self.radii,
        ];
        let mut words = [0u32; EFFECT_PUSH_WORDS];
        let mut at = 0usize;
        for group in groups {
            for value in group {
                words[at] = value.to_bits();
                at += 1;
            }
        }
        words[at] = self.rotation.to_bits();
        words[at + 1] = self.kinds;
        words[at + 2] = self.stroke.to_bits();
        words
    }
}

/// The parameters that rasterize `effect` over a `viewport` surface.
///
/// The two ramp colors and the axis are read only when the ramp bit is set,
/// which an [`Effect`] keeps consistent with its own gradient by construction;
/// without a ramp the colors are transparent and the axis collapses on the
/// center of the layer.
#[cfg(any(target_arch = "wasm32", test))]
fn effect_params(effect: &Effect, viewport: [f32; 2]) -> EffectParams {
    let center = effect.bounds.center();
    let center = [center.x, center.y];
    let (axis_start, axis_end, color_from, color_to) = match effect.gradient {
        Some(gradient) => (
            [gradient.start.x, gradient.start.y],
            [gradient.end.x, gradient.end.y],
            channels(gradient.from),
            channels(gradient.to),
        ),
        None => (center, center, [0.0; 4], [0.0; 4]),
    };
    let radii = effect.radii;
    EffectParams {
        color_from,
        color_to,
        viewport,
        origin: [effect.bounds.left(), effect.bounds.top()],
        size: [effect.bounds.size.width, effect.bounds.size.height],
        axis_start,
        axis_end,
        blur_pixel: [effect.blur, effect.pixel],
        radii: [
            radii.top_left,
            radii.top_right,
            radii.bottom_right,
            radii.bottom_left,
        ],
        rotation: effect.rotation,
        kinds: u32::from(effect.kinds.bits()),
        stroke: effect.stroke,
    }
}

/// A pixel rectangle inside a render target, as the UI pipe scissors to.
///
/// `wgpu-hal`'s own rectangle is only named on `wasm32`, so the scissor
/// arithmetic keeps its own copy and the pipe converts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scissor {
    /// Left pixel.
    pub x: u32,
    /// Top pixel.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The scissor rectangle of a command clip inside a `width`×`height` target.
///
/// A clip is snapped outward to whole pixels and clamped to the target, so
/// the scissor is always a valid, non-negative rectangle.
#[must_use]
pub fn scissor_for(clip: Option<Rect>, width: u32, height: u32) -> Scissor {
    let full = Scissor {
        x: 0,
        y: 0,
        width,
        height,
    };
    let Some(clip) = clip else {
        return full;
    };
    let clamp_x = width as f32;
    let clamp_y = height as f32;
    let left = floorf(clip.left()).max(0.0).min(clamp_x) as u32;
    let top = floorf(clip.top()).max(0.0).min(clamp_y) as u32;
    let right = ceilf(clip.right()).max(0.0).min(clamp_x) as u32;
    let bottom = ceilf(clip.bottom()).max(0.0).min(clamp_y) as u32;
    Scissor {
        x: left,
        y: top,
        width: right.saturating_sub(left),
        height: bottom.saturating_sub(top),
    }
}

/// A shared, single-threaded handle to the display list a `UiPipe` replays.
///
/// The interface records a frame into the handle and the pipe drains it, so
/// both sides see the same FIFO queue. The handle is `Rc`-based because a
/// [`DrawList`] owns `Rc<str>` text runs and is therefore confined to one
/// thread, exactly like the UI that produces it.
#[derive(Debug, Clone, Default)]
pub struct UiFrame {
    /// The shared queue.
    list: Rc<RefCell<DrawList>>,
}

impl UiFrame {
    /// Creates an empty frame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            list: Rc::new(RefCell::new(DrawList::new())),
        }
    }

    /// Runs `record` with exclusive access to the queue.
    ///
    /// The UI side records one frame per call; the pipe borrows the same
    /// queue, and the two borrows never overlap because the frame is fully
    /// recorded before it is composited.
    pub fn with<R>(&self, record: impl FnOnce(&mut DrawList) -> R) -> R {
        record(&mut self.list.borrow_mut())
    }

    /// Whether the queue holds no command.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.list.borrow().is_empty()
    }
}

#[cfg(target_arch = "wasm32")]
mod pipe;

#[cfg(test)]
mod tests;
