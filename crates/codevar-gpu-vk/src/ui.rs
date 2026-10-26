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
//! Rounded corners are likewise not representable by the color-only `ui`
//! shader and are drawn square; both limits are recorded here rather than
//! silently ignored.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use ash::vk;
use libm::{ceilf, floorf};

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
}

impl UiGeometry {
    /// Creates an empty geometry buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            spans: Vec::new(),
        }
    }

    /// Empties the geometry, keeping its allocations.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.spans.clear();
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
    /// its kind is the operation's job, not the backend's.
    pub fn push_command(&mut self, command: DrawCommand) {
        let first_vertex = self.vertices.len();
        let clip = command.clip;
        command.op.visit(self);
        let vertex_count = self.vertices.len() - first_vertex;
        if vertex_count == 0 {
            return;
        }
        self.spans.push(DrawSpan {
            first_vertex: first_vertex as u32,
            vertex_count: vertex_count as u32,
            clip,
        });
    }

    /// Expands a rectangle: a filled quad, or four quads for an outline.
    fn push_rect(&mut self, rect: Rect, paint: Paint) {
        let color = vertex_color(paint);
        let width = paint.stroke_width();
        if width <= 0.0 || matches!(paint.style, PaintStyle::Fill) {
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
    fn rect(&mut self, rect: Rect, _radii: CornerRadii, paint: Paint) {
        self.push_rect(rect, paint);
    }

    fn line(&mut self, from: Offset, to: Offset, paint: Paint) {
        self.push_line(from, to, paint);
    }
}

/// The straight RGBA vertex color of a paint.
#[inline]
fn vertex_color(paint: Paint) -> [f32; 4] {
    [
        paint.color.red,
        paint.color.green,
        paint.color.blue,
        paint.color.alpha,
    ]
}

/// Default number of vertices a [`UiPipe`] vertex buffer can hold.
pub const DEFAULT_VERTEX_CAPACITY: usize = 64 * 1024;

/// Byte size of the `ui.vert` push-constant block (`vec2 viewport`).
const VIEWPORT_PUSH_BYTES: u32 = 8;

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
/// image, the `ui.vert`/`ui.frag` pipeline and a host-visible vertex
/// buffer that is refilled every frame. [`PipeSource::pipe_entry`] drains
/// the frame's FIFO queue, expands it with [`UiGeometry`], uploads the
/// vertices and records one scissor-clipped draw per command.
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
        let (pipeline_layout, pipeline) = match create_ui_pipeline(context) {
            Ok(pair) => pair,
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
            pipeline_layout,
            pipeline,
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
    fn record(&self, ctx: &mut PipeCtx) {
        if self.geometry.is_empty() {
            return;
        }
        let device = ctx.device();
        let command_buffer = ctx.command_buffer();
        // SAFETY: `record` runs inside the dynamic-rendering scope opened
        // by `begin_render`, the pipeline was created for the target
        // format, and the vertex binding matches the pipeline layout.
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            let buffers = [self.buffer];
            let offsets = [0_u64];
            device.cmd_bind_vertex_buffers(command_buffer, 0, &buffers, &offsets);

            let viewport = [ctx.width() as f32, ctx.height() as f32];
            let bytes =
                core::slice::from_raw_parts(viewport.as_ptr().cast::<u8>(), VIEWPORT_PUSH_BYTES as usize);
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes,
            );

            let mut current: Option<Rect> = None;
            for span in self.geometry.spans() {
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
        }
    }
}

/// Creates the `ui.vert`/`ui.frag` pipeline layout and pipeline.
fn create_ui_pipeline(
    context: &PipelineContext,
) -> Result<(vk::PipelineLayout, vk::Pipeline), CompositorError> {
    let device = context.device();
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(VIEWPORT_PUSH_BYTES);
    let push_ranges = [push_range];
    let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
    // SAFETY: the push range is well-formed and lives for the call.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(CompositorError::PipelineLayoutCreate)?;

    match create_ui_graphics_pipeline(device, pipeline_layout) {
        Ok(pipeline) => Ok((pipeline_layout, pipeline)),
        Err(err) => {
            // SAFETY: the layout was just created and owns no pipeline yet.
            unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
            Err(err)
        }
    }
}

/// Builds the graphics pipeline from the shaders `codevar-kuitui` compiled.
fn create_ui_graphics_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, CompositorError> {
    let vertex_module = shader_module(device, codevar_kuitui::shaders::UI_VERT_SPIRV)?;
    let fragment_module = match shader_module(device, codevar_kuitui::shaders::UI_FRAG_SPIRV) {
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
