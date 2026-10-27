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

//! Replay of a kuitui [`DrawList`] into Vulkan-ready vertices.
//!
//! A frame is recorded by kuitui as a FIFO of [`DrawCommand`]s (see
//! [`codevar_kuitui::paint`]); the GPU backend replays that queue in
//! order. This module owns the CPU half of the replay: it drains the
//! queue and expands each operation into a triangle list ([`UiVertex`]),
//! recording one [`DrawSpan`] per operation so the renderer can bind the
//! operation's resolved clip as a scissor rectangle without re-deriving
//! the clip stack.
//!
//! The vertex layout matches the `ui` shader pair compiled by
//! `codevar-kuitui` (`ui.vert`/`ui.frag`): location 0 is the position in
//! logical pixels and location 1 is the color. The viewport mapping and
//! the clip scissor are applied by the renderer, so the expansion stays a
//! pure function of the display list and is unit-testable without a
//! device.
//!
//! # Scope
//!
//! Geometry operations ([`RectOp`] and [`LineOp`]) are expanded here
//! into vertices. A [`TextOp`] run carries no polygon geometry — it is
//! drawn by the text pass, which owns the glyph atlas and the `text`
//! shader — so this pass leaves it to the trait's default no-op.
//! Rounded corners are not representable by the color-only `ui` shader, so
//! a rectangle that asks for them is expanded as a layer of the `ui_effect`
//! program instead, exactly like a command that already carries an
//! [`Effect`]: the span keeps the layer, the renderer binds the effect
//! program for it, and the rectangle is expanded to the coverage of the
//! layer so a blurred halo has fragments to shade. The corner radii travel
//! inside the effect, so the same span draws them from the signed distance
//! instead of the vertex grid, and an outline of a rounded rectangle
//! follows those corners rather than being cut as four straight edges.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use ash::vk;
use libm::{ceilf, floorf};

use codevar_kuitui::color::Color;
use codevar_kuitui::effect::Effect;
use codevar_kuitui::geometry::{Offset, Rect};
use codevar_kuitui::paint::{CornerRadii, DrawCommand, DrawList, DrawVisitor, Paint, PaintStyle};

use crate::base::CompositorError;
use crate::mix::shader_module;
use crate::offscreen::{OffscreenFramebuffer, OffscreenTarget, PipeOffscreen};
use crate::pipe::PipeSource;
use crate::pipe_ctx::{PipeCtx, PipeFuture, PipeOutcome};
use crate::pipeline::{PipelineContext, find_memory_type};

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
    /// Layer effect resolved by kuitui when the operation was recorded;
    /// `None` means the plain `ui` program draws the span.
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
    /// The list is left empty: the queue is a one-shot hand-off from the
    /// UI thread to the renderer, which clears and re-records it next
    /// frame.
    pub fn expand(&mut self, list: &mut DrawList) {
        self.clear();
        while let Some(command) = list.pop_front() {
            self.push_command(command);
        }
    }

    /// Expands one command, appending its vertices and its span.
    ///
    /// The operation is replayed through [`DrawVisitor`]: dispatching on
    /// its kind is the operation's job, not the backend's. The effect of
    /// the command is open while the visitor runs, so a rectangle knows
    /// how wide a quad the shader needs.
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
            effect,
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

/// Default number of vertices a [`UiPipe`] vertex buffer can hold.
pub const DEFAULT_VERTEX_CAPACITY: usize = 64 * 1024;

/// Byte size of the `ui.vert` push-constant block (`vec2 viewport`).
const VIEWPORT_PUSH_BYTES: u32 = 8;

/// Byte size of the `ui_effect` push-constant block.
const EFFECT_PUSH_BYTES: u32 = core::mem::size_of::<EffectParams>() as u32;

/// The push constants of the `ui_effect` shader pair.
///
/// The field order is the `PushConstants` block of
/// `ui_effect.vert.glsl` and `ui_effect.frag.glsl`; `glslc` lays its
/// members out at offsets 0, 16, 32, 40, 48, 56, 64, 72, 80, 96, 100 and
/// 104, and a test pins every offset so reordering a field cannot silently
/// hand the shader a rotated pixel grid.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct EffectParams {
    /// The color at the start of the ramp; unread when the ramp bit is
    /// clear, because then the fragment keeps the vertex color.
    color_from: [f32; 4],
    /// The color at the end of the ramp.
    color_to: [f32; 4],
    /// Size of the render target, the viewport mapping of the vertex
    /// stage.
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
    /// The [`codevar_kuitui::effect::EffectKinds`] bits the fragment
    /// stage tests.
    kinds: u32,
    /// The width of the outline, in logical pixels; unread when the
    /// stroke bit is clear, because then the fragment keeps the whole
    /// rounded box.
    stroke: f32,
}

/// The parameters that rasterize `effect` over a `viewport` surface.
///
/// The two ramp colors and the axis are read only when the ramp bit is
/// set, which an [`Effect`] keeps consistent with its own gradient by
/// construction; without a ramp the colors are transparent and the axis
/// collapses on the center of the layer.
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

/// A shared, single-threaded handle to the display list a [`UiPipe`]
/// replays.
///
/// The interface records a frame into the handle and the renderer drains
/// it, so both sides see the same FIFO queue. The handle is `Rc`-based
/// because a [`DrawList`] owns `Rc<str>` text runs and is therefore
/// confined to one thread, exactly like the UI that produces it.
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
    /// The UI thread records one frame per call; the renderer borrows the
    /// same queue from its pipe, and the two borrows never overlap because
    /// the frame is fully recorded before it is composited.
    pub fn with<R>(&self, record: impl FnOnce(&mut DrawList) -> R) -> R {
        record(&mut self.list.borrow_mut())
    }

    /// Whether the queue holds no command.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.list.borrow().is_empty()
    }
}

/// The scissor rectangle of a command clip inside a `width`×`height`
/// target.
///
/// A clip is snapped outward to whole pixels and clamped to the target, so
/// the scissor is always a valid, non-negative rectangle.
pub fn scissor_for(clip: Option<Rect>, width: u32, height: u32) -> vk::Rect2D {
    let full = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    };
    let Some(clip) = clip else {
        return full;
    };
    let clamp_x = width as f32;
    let clamp_y = height as f32;
    let left = floorf(clip.left()).max(0.0).min(clamp_x) as i32;
    let top = floorf(clip.top()).max(0.0).min(clamp_y) as i32;
    let right = ceilf(clip.right()).max(0.0).min(clamp_x) as i32;
    let bottom = ceilf(clip.bottom()).max(0.0).min(clamp_y) as i32;
    vk::Rect2D {
        offset: vk::Offset2D { x: left, y: top },
        extent: vk::Extent2D {
            width: (right - left).max(0) as u32,
            height: (bottom - top).max(0) as u32,
        },
    }
}

/// A compositor pipe that replays a [`UiFrame`] into an offscreen target.
///
/// The pipe owns the target the mix pass folds into the presentation
/// image, the `ui.vert`/`ui.frag` pipeline, the `ui_effect` pipeline that
/// rasterizes a layer effect, and a host-visible vertex buffer that is
/// refilled every frame. [`PipeSource::pipe_entry`] drains the frame's FIFO
/// queue, expands it with [`UiGeometry`], uploads the vertices and records
/// one scissor-clipped draw per command, binding the effect program for
/// the spans that carry one.
///
/// Register it with [`Compositor::add_pipe`](crate::compositor::Compositor::add_pipe)
/// and keep the [`UiFrame`] to record subsequent frames.
pub struct UiPipe<'p> {
    /// The shared display list, drained once per frame.
    frame: UiFrame,
    /// CPU-side replay scratch, reused every frame.
    geometry: UiGeometry,
    /// The offscreen target the pipe renders into.
    target: OffscreenTarget<'p>,
    /// Device that owns every handle below.
    device: &'p ash::Device,
    /// Pipeline layout: the `ui.vert` push-constant range.
    pipeline_layout: vk::PipelineLayout,
    /// The `ui.vert`/`ui.frag` graphics pipeline.
    pipeline: vk::Pipeline,
    /// Pipeline layout: the `ui_effect` push-constant range.
    effect_pipeline_layout: vk::PipelineLayout,
    /// The `ui_effect.vert`/`ui_effect.frag` graphics pipeline.
    effect_pipeline: vk::Pipeline,
    /// Host-visible vertex buffer.
    buffer: vk::Buffer,
    /// Memory bound to `buffer`.
    memory: vk::DeviceMemory,
    /// Persistent host mapping of `memory`, valid for `capacity` vertices.
    mapped: *mut UiVertex,
    /// Number of vertices `buffer` holds.
    capacity: usize,
    /// Clear color of the target at the start of the frame.
    clear: [f32; 4],
    /// Mix weight of the target in the frame.
    weight: f32,
}

impl<'p> UiPipe<'p> {
    /// Creates a pipe with the [`DEFAULT_VERTEX_CAPACITY`] vertex buffer.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`UiPipe::with_capacity`].
    pub fn new(context: &'p PipelineContext, frame: UiFrame) -> Result<Self, CompositorError> {
        Self::with_capacity(context, frame, DEFAULT_VERTEX_CAPACITY)
    }

    /// Creates a pipe whose vertex buffer holds `capacity` vertices.
    ///
    /// A capacity of zero is raised to one so the buffer mapping stays
    /// valid. When the UI shaders were not compiled (no `glslc` at build
    /// time, `SHADERS_COMPILED == false`) pipeline creation fails with a
    /// [`CompositorError::ShaderLoad`].
    ///
    /// # Errors
    ///
    /// * the [`OffscreenTarget::new`] failures;
    /// * [`CompositorError::PipelineLayoutCreate`], [`CompositorError::ShaderLoad`],
    ///   [`CompositorError::ShaderModuleCreate`], [`CompositorError::PipelineCreate`];
    /// * [`CompositorError::BufferCreate`], [`CompositorError::NoSuitableMemoryType`],
    ///   [`CompositorError::MemoryAllocate`], [`CompositorError::MemoryBind`],
    ///   [`CompositorError::BufferMap`].
    pub fn with_capacity(
        context: &'p PipelineContext,
        frame: UiFrame,
        capacity: usize,
    ) -> Result<Self, CompositorError> {
        let capacity = capacity.max(1);
        let device = context.device();
        let target = OffscreenTarget::new(
            context,
            context.width(),
            context.height(),
            PipelineContext::color_format(),
        )?;
        let (buffer, memory, mapped) = create_vertex_buffer(context, capacity)?;
        let pipes = match create_ui_pipes(context) {
            Ok(pipes) => pipes,
            Err(err) => {
                // SAFETY: the buffer and its mapping were just created by
                // `create_vertex_buffer` and are not referenced by any
                // in-flight command buffer.
                unsafe {
                    device.unmap_memory(memory);
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(err);
            }
        };
        Ok(Self {
            frame,
            geometry: UiGeometry::new(),
            target,
            device,
            pipeline_layout: pipes.layout,
            pipeline: pipes.pipeline,
            effect_pipeline_layout: pipes.effect_layout,
            effect_pipeline: pipes.effect_pipeline,
            buffer,
            memory,
            mapped,
            capacity,
            clear: [0.0, 0.0, 0.0, 0.0],
            weight: 1.0,
        })
    }

    /// The frame this pipe replays.
    #[inline]
    #[must_use]
    pub fn frame(&self) -> &UiFrame {
        &self.frame
    }

    /// Sets the color the target is cleared to at the start of a frame.
    pub fn set_clear(&mut self, clear: [f32; 4]) {
        self.clear = clear;
    }

    /// Sets the mix weight of the target in `[0.0, 1.0]`.
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }

    /// Drains the shared queue into the CPU replay scratch.
    fn expand_frame(&mut self) {
        let mut list = self.frame.list.borrow_mut();
        self.geometry.expand(&mut list);
    }

    /// Copies the current geometry into the mapped vertex buffer.
    fn upload(&self) -> Result<(), CompositorError> {
        let vertices = self.geometry.vertices();
        if vertices.len() > self.capacity {
            return Err(CompositorError::Internal(
                "the UI frame needs more vertices than the buffer holds",
            ));
        }
        if vertices.is_empty() {
            return Ok(());
        }
        // SAFETY: `mapped` points at `capacity` writable `UiVertex` slots
        // of the host-visible, coherent allocation owned by this pipe, and
        // `vertices` holds at most `capacity` initialized elements of a
        // disjoint allocation.
        unsafe {
            core::ptr::copy_nonoverlapping(vertices.as_ptr(), self.mapped, vertices.len());
        }
        Ok(())
    }

    /// Records the draw calls of one frame inside the open rendering scope.
    ///
    /// The two programs share the vertex buffer, so a frame walks its spans
    /// in paint order and binds the effect program only for the spans that
    /// carry an effect.
    fn record(&self, ctx: &mut PipeCtx) {
        if self.geometry.is_empty() {
            return;
        }
        let device = ctx.device();
        let command_buffer = ctx.command_buffer();
        // SAFETY: `record` runs inside the dynamic-rendering scope opened
        // by `begin_render`, both pipelines were created for the target
        // format, and the vertex binding matches the input layout of both.
        unsafe {
            let buffers = [self.buffer];
            let offsets = [0_u64];
            device.cmd_bind_vertex_buffers(command_buffer, 0, &buffers, &offsets);

            let viewport = [ctx.width() as f32, ctx.height() as f32];
            let viewport_bytes =
                core::slice::from_raw_parts(viewport.as_ptr().cast::<u8>(), VIEWPORT_PUSH_BYTES as usize);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                viewport_bytes,
            );

            let mut current: Option<Rect> = None;
            let mut on_effect = false;
            for span in self.geometry.spans() {
                let wants_effect = span.effect.is_some();
                if wants_effect != on_effect {
                    on_effect = wants_effect;
                    // The blocks of the two layouts overlap, and a push
                    // only carries the bytes of the range it names, so a
                    // switch always re-pushes what the new pipeline reads.
                    current = None;
                    if wants_effect {
                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.effect_pipeline,
                        );
                    } else {
                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline,
                        );
                        device.cmd_push_constants(
                            command_buffer,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            viewport_bytes,
                        );
                    }
                }
                if let Some(effect) = span.effect {
                    let params = effect_params(&effect, viewport);
                    let bytes = core::slice::from_raw_parts(
                        core::ptr::from_ref(&params).cast::<u8>(),
                        EFFECT_PUSH_BYTES as usize,
                    );
                    device.cmd_push_constants(
                        command_buffer,
                        self.effect_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytes,
                    );
                }
                if span.clip != current {
                    current = span.clip;
                    let scissor = scissor_for(span.clip, ctx.width(), ctx.height());
                    device.cmd_set_scissor(command_buffer, 0, core::slice::from_ref(&scissor));
                }
                device.cmd_draw(command_buffer, span.vertex_count, 1, span.first_vertex, 0);
            }
        }
    }
}

impl PipeSource for UiPipe<'_> {
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
        Box::pin(async move {
            renderer.expand_frame();
            renderer.upload()?;
            ctx.begin_render(0, Some(renderer.clear))?;
            renderer.record(ctx);
            ctx.end_render();
            Ok(PipeOutcome::Keep)
        })
    }
}

impl PipeOffscreen for UiPipe<'_> {
    fn offscreen_framebuffer_count(&self) -> usize {
        1
    }

    fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer> {
        if index == 0 {
            Some(self.target.framebuffer())
        } else {
            None
        }
    }

    fn mix_weight(&self, _index: usize) -> f32 {
        self.weight
    }
}

impl Drop for UiPipe<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created by `with_capacity` on this
        // device, and the compositor waits for the device before the pipe
        // is released, so none is referenced by an in-flight command
        // buffer.
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_pipeline(self.effect_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.effect_pipeline_layout, None);
        }
    }
}

/// The pipeline handles a [`UiPipe`] owns, created together.
struct UiPipelines {
    /// Layout of the plain `ui` program.
    layout: vk::PipelineLayout,
    /// The plain `ui` program.
    pipeline: vk::Pipeline,
    /// Layout of the `ui_effect` program.
    effect_layout: vk::PipelineLayout,
    /// The `ui_effect` program.
    effect_pipeline: vk::Pipeline,
}

/// Creates both UI programs.
///
/// A frame needs both: the plain program draws every span without an
/// effect and the effect program draws the ones with it, so a pipe that
/// cannot make one cannot draw the frame.
fn create_ui_pipes(context: &PipelineContext) -> Result<UiPipelines, CompositorError> {
    let device = context.device();
    let (layout, pipeline) = create_ui_program(context)?;
    let (effect_layout, effect_pipeline) = match create_effect_program(context) {
        Ok(pair) => pair,
        Err(err) => {
            // SAFETY: neither plain handle is referenced by a command
            // buffer; the pipe that would own them was never built.
            unsafe {
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(layout, None);
            }
            return Err(err);
        }
    };
    Ok(UiPipelines {
        layout,
        pipeline,
        effect_layout,
        effect_pipeline,
    })
}

/// Creates the layout and pipeline of the `ui.vert`/`ui.frag` program.
fn create_ui_program(
    context: &PipelineContext,
) -> Result<(vk::PipelineLayout, vk::Pipeline), CompositorError> {
    create_program(
        context,
        vk::ShaderStageFlags::VERTEX,
        VIEWPORT_PUSH_BYTES,
        codevar_kuitui::shaders::UI_VERT_SPIRV,
        codevar_kuitui::shaders::UI_FRAG_SPIRV,
    )
}

/// Creates the layout and pipeline of the `ui_effect` program.
fn create_effect_program(
    context: &PipelineContext,
) -> Result<(vk::PipelineLayout, vk::Pipeline), CompositorError> {
    create_program(
        context,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        EFFECT_PUSH_BYTES,
        codevar_kuitui::shaders::UI_EFFECT_VERT_SPIRV,
        codevar_kuitui::shaders::UI_EFFECT_FRAG_SPIRV,
    )
}

/// Creates a pipeline layout holding one `size`-byte push-constant block
/// visible to `stages`, and the graphics pipeline of the shader pair that
/// reads it.
///
/// Both programs share the [`UiVertex`] input layout, the triangle-list
/// topology and the straight-alpha blend, so they differ only by their
/// shaders and by the stage flags of their block.
fn create_program(
    context: &PipelineContext,
    stages: vk::ShaderStageFlags,
    size: u32,
    vertex_spirv: &[u8],
    fragment_spirv: &[u8],
) -> Result<(vk::PipelineLayout, vk::Pipeline), CompositorError> {
    let device = context.device();
    let push_range = vk::PushConstantRange::default()
        .stage_flags(stages)
        .offset(0)
        .size(size);
    let push_ranges = [push_range];
    let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
    // SAFETY: the push range is well-formed and lives for the call.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(CompositorError::PipelineLayoutCreate)?;

    match create_graphics_pipeline(device, pipeline_layout, vertex_spirv, fragment_spirv) {
        Ok(pipeline) => Ok((pipeline_layout, pipeline)),
        Err(err) => {
            // SAFETY: the layout was just created and owns no pipeline yet.
            unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
            Err(err)
        }
    }
}

/// Builds a graphics pipeline from two embedded SPIR-V images.
fn create_graphics_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vertex_spirv: &[u8],
    fragment_spirv: &[u8],
) -> Result<vk::Pipeline, CompositorError> {
    let vertex_module = shader_module(device, vertex_spirv)?;
    let fragment_module = match shader_module(device, fragment_spirv) {
        Ok(module) => module,
        Err(err) => {
            // SAFETY: the vertex module was just created and is unused.
            unsafe { device.destroy_shader_module(vertex_module, None) };
            return Err(err);
        }
    };
    let pipeline = build_ui_pipeline(device, pipeline_layout, vertex_module, fragment_module);
    // SAFETY: the modules are not needed once the pipeline exists.
    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }
    pipeline
}

/// Creates the graphics pipeline described by the [`UiVertex`] layout.
fn build_ui_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
) -> Result<vk::Pipeline, CompositorError> {
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(c"main"),
    ];
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(core::mem::size_of::<UiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attributes = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: core::mem::offset_of!(UiVertex, position) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: core::mem::offset_of!(UiVertex, color) as u32,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(core::slice::from_ref(&binding))
        .vertex_attribute_descriptions(&attributes);
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    };
    let blend_attachments = [blend_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_format = PipelineContext::color_format();
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(core::slice::from_ref(&color_format));
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);
    // SAFETY: every create-info is well-formed, the rendering info declares
    // the single color attachment format of dynamic rendering, and both
    // modules are alive for the call.
    let pipelines = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            core::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, err)| CompositorError::PipelineCreate(err))?;
    pipelines
        .first()
        .copied()
        .ok_or(CompositorError::Internal("the driver returned no UI pipeline"))
}

/// Creates a host-visible vertex buffer and maps it for the pipe's life.
fn create_vertex_buffer(
    context: &PipelineContext,
    capacity: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut UiVertex), CompositorError> {
    let device = context.device();
    let size = (capacity * core::mem::size_of::<UiVertex>()) as u64;
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: a plain, well-formed buffer descriptor.
    let buffer = unsafe { device.create_buffer(&info, None) }.map_err(CompositorError::BufferCreate)?;

    // SAFETY: `buffer` was just created on this device.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    // SAFETY: the instance and physical device outlive `context`.
    let properties = unsafe {
        context
            .instance()
            .get_physical_device_memory_properties(context.physical_device())
    };
    let Some(memory_type) = find_memory_type(
        &properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ) else {
        // SAFETY: nothing references `buffer` yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(CompositorError::NoSuitableMemoryType);
    };
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    // SAFETY: the allocation matches an advertised memory type and size.
    let memory = match unsafe { device.allocate_memory(&allocate_info, None) } {
        Ok(memory) => memory,
        Err(err) => {
            // SAFETY: nothing references `buffer` yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(CompositorError::MemoryAllocate(err));
        }
    };
    // SAFETY: `memory` was just allocated for this device and `buffer` is
    // not bound to anything.
    if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: neither object is referenced elsewhere.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(CompositorError::MemoryBind(err));
    }
    // SAFETY: the memory is host-visible and not yet used by the GPU.
    let mapped = match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
        Ok(pointer) => pointer.cast::<UiVertex>(),
        Err(err) => {
            // SAFETY: the memory is bound to `buffer` but never submitted.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(CompositorError::BufferMap(err));
        }
    };
    Ok((buffer, memory, mapped))
}

#[cfg(test)]
mod tests {
    //! Display-list expansion: geometry, span order and clip carry-over.

    use super::*;
    use alloc::rc::Rc;
    use codevar_kuitui::color::Color;
    use codevar_kuitui::effect::{EffectKinds, Gradient};
    use codevar_kuitui::paint::CornerRadii;
    use codevar_kuitui::text::TextStyle;

    /// A black fill, the color most tests do not care about.
    fn fill() -> Paint {
        Paint::fill(Color::BLACK)
    }

    #[test]
    fn a_filled_rectangle_expands_to_two_triangles() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::new(4.0, 8.0), 10.0, 20.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.vertices().len(), 6);
        assert_eq!(geometry.spans().len(), 1);
        let span = geometry.spans()[0];
        assert_eq!(span.first_vertex, 0);
        assert_eq!(span.vertex_count, 6);
        assert_eq!(span.clip, None);
        assert_eq!(span.effect, None, "a plain record needs no effect program");

        let first = geometry.vertices()[0];
        assert_eq!(first.position, [4.0, 8.0]);
        assert_eq!(
            first.color,
            [
                Color::RED.red,
                Color::RED.green,
                Color::RED.blue,
                Color::RED.alpha
            ]
        );
        assert!(list.is_empty(), "expand drains the FIFO queue");
    }

    #[test]
    fn spans_follow_the_paint_order_and_tile_the_buffer() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
            CornerRadii::ZERO,
            fill(),
        );
        list.line(
            Offset::ZERO,
            Offset::new(4.0, 0.0),
            Paint::stroke(Color::WHITE, 2.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        let spans = geometry.spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].first_vertex, 0);
        assert_eq!(spans[0].vertex_count, 6);
        assert_eq!(spans[1].first_vertex, 6);
        assert_eq!(spans[1].vertex_count, 6);
        assert_eq!(geometry.vertices().len(), 12);
    }

    #[test]
    fn a_span_carries_the_command_clip() {
        let mut list = DrawList::new();
        let clip = Rect::from_min_size(Offset::new(5.0, 5.0), 20.0, 20.0);
        list.push_clip(clip);
        list.rect(
            Rect::from_min_size(Offset::ZERO, 100.0, 100.0),
            CornerRadii::ZERO,
            fill(),
        );
        list.pop_clip();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
            CornerRadii::ZERO,
            fill(),
        );

        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        assert_eq!(geometry.spans()[0].clip, Some(clip));
        assert_eq!(geometry.spans()[1].clip, None);
    }

    #[test]
    fn an_effect_span_keeps_the_layer_and_widens_its_quad() {
        let bounds = Rect::from_min_size(Offset::new(10.0, 10.0), 40.0, 20.0);
        let effect = Effect::new(bounds).blur(6.0, CornerRadii::uniform(3.0));
        let mut list = DrawList::new();
        list.push_effect(effect);
        list.rect(bounds, CornerRadii::uniform(3.0), fill());
        list.pop_effect();

        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        let span = geometry.spans()[0];
        assert_eq!(span.effect, Some(effect));
        // Three sigmas of a six-pixel blur reach 18 pixels past every
        // edge, and the shader needs those fragments to draw the halo.
        assert_eq!(geometry.vertices()[0].position, [-8.0, -8.0]);
        assert_eq!(geometry.vertices()[1].position, [68.0, -8.0]);
    }

    #[test]
    fn a_rounded_rectangle_is_cut_by_the_effect_program() {
        let bounds = Rect::from_min_size(Offset::new(10.0, 10.0), 40.0, 20.0);
        let radii = CornerRadii::uniform(8.0);
        let mut list = DrawList::new();
        list.rect(bounds, radii, fill());

        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        // The corners are a property of the rasterization, so the operation
        // is one quad and a layer that carries nothing but its rounding.
        let span = geometry.spans()[0];
        assert_eq!(span.vertex_count, 6);
        assert_eq!(span.effect, Some(Effect::new(bounds).shape(radii)));
        assert_eq!(geometry.vertices()[0].position, [10.0, 10.0]);
    }

    #[test]
    fn a_rounded_outline_is_a_ring_instead_of_four_edges() {
        let bounds = Rect::from_min_size(Offset::ZERO, 30.0, 12.0);
        let radii = CornerRadii::uniform(6.0);
        let mut list = DrawList::new();
        list.rect(bounds, radii, Paint::stroke(Color::WHITE, 2.0));

        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.spans().len(), 1);
        assert_eq!(geometry.spans()[0].vertex_count, 6, "one quad");
        assert_eq!(
            geometry.spans()[0].effect,
            Some(Effect::new(bounds).shape(radii).stroke(2.0))
        );
    }

    #[test]
    fn the_effect_block_keeps_the_shader_offsets() {
        assert_eq!(EFFECT_PUSH_BYTES, 108);
        assert_eq!(core::mem::offset_of!(EffectParams, color_from), 0);
        assert_eq!(core::mem::offset_of!(EffectParams, color_to), 16);
        assert_eq!(core::mem::offset_of!(EffectParams, viewport), 32);
        assert_eq!(core::mem::offset_of!(EffectParams, origin), 40);
        assert_eq!(core::mem::offset_of!(EffectParams, size), 48);
        assert_eq!(core::mem::offset_of!(EffectParams, axis_start), 56);
        assert_eq!(core::mem::offset_of!(EffectParams, axis_end), 64);
        assert_eq!(core::mem::offset_of!(EffectParams, blur_pixel), 72);
        assert_eq!(core::mem::offset_of!(EffectParams, radii), 80);
        assert_eq!(core::mem::offset_of!(EffectParams, rotation), 96);
        assert_eq!(core::mem::offset_of!(EffectParams, kinds), 100);
        assert_eq!(core::mem::offset_of!(EffectParams, stroke), 104);
    }

    #[test]
    fn the_kind_bits_are_the_ones_the_shader_tests() {
        // `ui_effect.frag.glsl` names these KIND_BLUR, KIND_PIXELATE,
        // KIND_ROTATE, KIND_GRADIENT and KIND_STROKE; the two sides meet
        // in the push constants and nothing else checks them.
        assert_eq!(EffectKinds::BLUR.bits(), 1);
        assert_eq!(EffectKinds::PIXELATE.bits(), 2);
        assert_eq!(EffectKinds::ROTATE.bits(), 4);
        assert_eq!(EffectKinds::GRADIENT.bits(), 8);
        assert_eq!(EffectKinds::STROKE.bits(), 16);
    }

    #[test]
    fn the_effect_parameters_carry_the_ramp_and_the_grid() {
        let bounds = Rect::from_min_size(Offset::new(4.0, 6.0), 40.0, 20.0);
        let gradient = Gradient::new(
            Offset::new(4.0, 16.0),
            Offset::new(44.0, 16.0),
            Color::RED,
            Color::BLUE,
        );
        let effect = Effect::new(bounds)
            .pixelate(5.0)
            .rotate(0.5)
            .gradient(gradient, CornerRadii::uniform(2.0));

        let params = effect_params(&effect, [800.0, 600.0]);
        assert_eq!(params.viewport, [800.0, 600.0]);
        assert_eq!(params.origin, [4.0, 6.0]);
        assert_eq!(params.size, [40.0, 20.0]);
        assert_eq!(params.axis_start, [4.0, 16.0]);
        assert_eq!(params.axis_end, [44.0, 16.0]);
        assert_eq!(params.color_from, channels(Color::RED));
        assert_eq!(params.color_to, channels(Color::BLUE));
        assert_eq!(params.blur_pixel, [0.0, 5.0]);
        assert_eq!(params.rotation, 0.5);
        assert_eq!(params.radii, [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(params.kinds, 2 | 4 | 8);
        assert_eq!(params.stroke, 0.0);
    }

    #[test]
    fn the_effect_parameters_carry_the_width_of_an_outline() {
        let bounds = Rect::from_min_size(Offset::new(0.0, 0.0), 30.0, 12.0);
        let effect = Effect::new(bounds).stroke(2.5).gradient(
            Gradient::new(Offset::ZERO, Offset::new(30.0, 0.0), Color::RED, Color::BLUE),
            CornerRadii::uniform(6.0),
        );

        let params = effect_params(&effect, [64.0, 64.0]);
        assert_eq!(params.stroke, 2.5);
        assert_eq!(params.kinds, 8 | 16);
        assert_eq!(params.radii, [6.0; 4]);
    }

    #[test]
    fn a_layer_without_a_ramp_collapses_its_axis() {
        let bounds = Rect::from_min_size(Offset::new(2.0, 2.0), 20.0, 10.0);
        let effect = Effect::new(bounds).blur(3.0, CornerRadii::ZERO);

        let params = effect_params(&effect, [100.0, 100.0]);
        let center = [12.0, 7.0];
        assert_eq!(params.axis_start, center);
        assert_eq!(params.axis_end, center);
        assert_eq!(params.blur_pixel, [3.0, 0.0]);
        assert_eq!(params.kinds, 1);
    }

    #[test]
    fn an_outlined_rectangle_expands_to_four_edges() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::stroke(Color::WHITE, 2.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.spans().len(), 1);
        assert_eq!(geometry.spans()[0].vertex_count, 24, "four quads");
    }

    #[test]
    fn an_outline_wider_than_the_rectangle_is_clamped() {
        let mut list = DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
            CornerRadii::ZERO,
            Paint::stroke(Color::WHITE, 100.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        // The four edges still get emitted, none of them inverted.
        assert_eq!(geometry.spans()[0].vertex_count, 24);
        assert!(
            geometry
                .vertices()
                .iter()
                .all(|vertex| vertex.position[0] >= 0.0 && vertex.position[0] <= 4.0)
        );
    }

    #[test]
    fn a_line_expands_to_a_quad_around_its_axis() {
        let mut list = DrawList::new();
        list.line(
            Offset::new(0.0, 0.0),
            Offset::new(10.0, 0.0),
            Paint::stroke(Color::WHITE, 2.0),
        );
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);

        assert_eq!(geometry.spans()[0].vertex_count, 6);
        // A horizontal line of width 2 is offset by ±1 on the y axis.
        let ys: Vec<f32> = geometry
            .vertices()
            .iter()
            .map(|vertex| vertex.position[1])
            .collect();
        assert!(
            ys.iter()
                .all(|y| (*y - -1.0).abs() < 1e-6 || (*y - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn a_degenerate_line_draws_nothing() {
        let mut list = DrawList::new();
        list.line(Offset::ZERO, Offset::ZERO, Paint::stroke(Color::WHITE, 2.0));
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        assert!(geometry.is_empty());
        assert_eq!(geometry.vertices().len(), 0);
    }

    #[test]
    fn a_text_run_is_left_to_the_text_pass() {
        let mut list = DrawList::new();
        list.text(Offset::ZERO, Rc::from("hi"), TextStyle::default());
        let mut geometry = UiGeometry::new();
        geometry.expand(&mut list);
        assert!(geometry.is_empty());
        assert_eq!(geometry.vertices().len(), 0);
    }

    #[test]
    fn expanding_clears_the_previous_frame() {
        let mut geometry = UiGeometry::new();
        let mut first = DrawList::new();
        first.rect(
            Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
            CornerRadii::ZERO,
            fill(),
        );
        geometry.expand(&mut first);
        assert_eq!(geometry.vertices().len(), 6);

        let mut second = DrawList::new();
        second.line(
            Offset::ZERO,
            Offset::new(1.0, 0.0),
            Paint::stroke(Color::WHITE, 1.0),
        );
        geometry.expand(&mut second);
        assert_eq!(geometry.vertices().len(), 6);
        assert_eq!(geometry.spans().len(), 1);
    }

    #[test]
    fn the_scissor_of_an_unclipped_command_covers_the_target() {
        let scissor = scissor_for(None, 800, 600);
        assert_eq!(scissor.offset.x, 0);
        assert_eq!(scissor.offset.y, 0);
        assert_eq!(scissor.extent.width, 800);
        assert_eq!(scissor.extent.height, 600);
    }

    #[test]
    fn the_scissor_of_a_clip_is_snapped_outward_and_clamped() {
        let clip = Rect::from_min_size(Offset::new(10.4, 20.6), 100.0, 50.0);
        let scissor = scissor_for(Some(clip), 800, 600);
        assert_eq!(scissor.offset.x, 10);
        assert_eq!(scissor.offset.y, 20);
        assert_eq!(scissor.extent.width, 101);
        assert_eq!(scissor.extent.height, 51);

        let outside = Rect::from_min_size(Offset::new(-50.0, -50.0), 20.0, 20.0);
        let clamped = scissor_for(Some(outside), 800, 600);
        assert_eq!(clamped.offset.x, 0);
        assert_eq!(clamped.offset.y, 0);
        assert_eq!(clamped.extent.width, 0);
        assert_eq!(clamped.extent.height, 0);
    }

    #[test]
    fn a_recorded_frame_expands_and_drains() {
        let frame = UiFrame::new();
        frame.with(|list| {
            list.rect(
                Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
                CornerRadii::ZERO,
                Paint::fill(Color::BLACK),
            );
        });
        assert!(!frame.is_empty());

        let mut geometry = UiGeometry::new();
        frame.with(|list| geometry.expand(list));
        assert_eq!(geometry.spans().len(), 1);
        assert!(frame.is_empty(), "the renderer consumed the queue");
    }
}
