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

//! The compositor pipe that replays a [`UiFrame`] on WebGPU.
//!
//! [`UiPipe`] is the browser counterpart of `codevar-gpu-vk`'s `UiPipe`: it
//! owns one offscreen target the mix pass folds into the canvas, the `ui` and
//! `ui_effect` graphics pipelines built from the shaders `codevar-kuitui`
//! compiles, and the vertex stream the expanded display list travels through.
//! The CPU half of the replay — the triangle list, the effect block and the
//! scissor arithmetic — lives in the parent module, where the host can test
//! it.
//!
//! # Why the vertices stage through a copy
//!
//! WebGL2 cannot bind a CPU-mapped buffer for drawing, so the pipe never draws
//! from the buffer it fills. It writes the expanded frame into a
//! `MAP_WRITE | COPY_SRC` staging buffer and records a `copy_buffer_to_buffer`
//! into a `VERTEX | COPY_DST` buffer it then binds. The whole frame goes over
//! in one copy, and the buffer stays alive across frames, so a steady-state
//! frame costs one mapping and one copy.

use alloc::boxed::Box;
use core::mem::{ManuallyDrop, size_of};

use wgpu_hal::BufferBinding;
use wgpu_hal::CommandEncoder as _;
use wgpu_types as wgt;

use codevar_kuitui::geometry::Rect;
use codevar_kuitui::shaders::{UI_EFFECT_FRAG_SPIRV, UI_EFFECT_VERT_SPIRV, UI_FRAG_SPIRV, UI_VERT_SPIRV};

use crate::graphics::{self, GpuBuffer, GraphicsPipeline, GraphicsPipelineDesc};
use crate::offscreen::{OffscreenFramebuffer, OffscreenTarget, PipeOffscreen};
use crate::pipe::PipeSource;
use crate::pipe_ctx::{PipeCtx, PipeFuture, PipeOutcome};
use crate::pipeline::WgContext;
use crate::types::CompositorError;

use super::{
    DEFAULT_VERTEX_CAPACITY, EFFECT_PUSH_BYTES, UiFrame, UiGeometry, UiVertex, VIEWPORT_PUSH_BYTES,
    effect_params, scissor_for,
};

/// Usage of the vertex buffer: drawable and writable by a copy.
const VERTEX_USAGE: wgt::BufferUses = wgt::BufferUses::VERTEX.union(wgt::BufferUses::COPY_DST);

/// Usage of the staging buffer: CPU-writable and copyable from.
const STAGING_USAGE: wgt::BufferUses = wgt::BufferUses::MAP_WRITE.union(wgt::BufferUses::COPY_SRC);

/// The `UiVertex` fields at their `ui.vert` locations.
///
/// The layout is the one both UI shader pairs declare: a `vec2` position at
/// location 0 and a `vec4` color at location 1.
const VERTEX_ATTRIBUTES: [wgt::VertexAttribute; 2] = [
    wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    },
    wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x4,
        offset: 8,
        shader_location: 1,
    },
];

/// A compositor pipe that replays a [`UiFrame`] into an offscreen target.
///
/// Register it with the compositor and keep the [`UiFrame`] to record the next
/// frames. Register the text pipe after this one so the mix pass folds the
/// glyph layer over the geometry.
pub struct UiPipe<'p> {
    /// The pipeline context that owns every GPU handle below.
    context: &'p WgContext,
    /// The shared display list, drained once per frame.
    frame: UiFrame,
    /// CPU-side replay scratch, reused every frame.
    geometry: UiGeometry,
    /// The offscreen target the mix pass folds into the canvas.
    target: OffscreenTarget<'p>,
    /// The `ui.vert`/`ui.frag` program for the spans without an effect.
    pipeline: ManuallyDrop<GraphicsPipeline>,
    /// The `ui_effect` program for the spans that carry one.
    effect_pipeline: ManuallyDrop<GraphicsPipeline>,
    /// Vertex stream the spans are drawn from.
    vertices: ManuallyDrop<GpuBuffer>,
    /// Staging buffer the vertices travel through.
    staging: ManuallyDrop<GpuBuffer>,
    /// Color the target is cleared to at the start of the frame.
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
    pub fn new(context: &'p WgContext, frame: UiFrame) -> Result<Self, CompositorError> {
        Self::with_capacity(context, frame, DEFAULT_VERTEX_CAPACITY)
    }

    /// Creates a pipe whose vertex buffer starts at `capacity` vertices.
    ///
    /// A capacity of zero is raised to one so a buffer is always allocated.
    /// When the UI shaders were not compiled (no `glslc` at build time,
    /// `SHADERS_COMPILED == false`) pipeline creation fails with a
    /// [`CompositorError::ShaderLoad`].
    ///
    /// # Errors
    ///
    /// * the [`OffscreenTarget::new`] failures;
    /// * [`CompositorError::Graphics`] — the driver refused a buffer or a
    ///   pipeline;
    /// * [`CompositorError::ShaderLoad`] — an embedded shader is malformed.
    pub fn with_capacity(
        context: &'p WgContext,
        frame: UiFrame,
        capacity: usize,
    ) -> Result<Self, CompositorError> {
        let device = context.device();
        let vertex_bytes = (capacity.max(1) * size_of::<UiVertex>()) as u64;
        let target = OffscreenTarget::new(context, context.width(), context.height(), context.format())?;
        let vertices = GpuBuffer::new(device, VERTEX_USAGE, vertex_bytes)?;
        let staging = match GpuBuffer::new(device, STAGING_USAGE, vertex_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                vertices.destroy(device);
                return Err(err);
            }
        };
        let pipeline = match create_program(
            device,
            context,
            "codevar-ui",
            UI_VERT_SPIRV,
            UI_FRAG_SPIRV,
            VIEWPORT_PUSH_BYTES,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                staging.destroy(device);
                vertices.destroy(device);
                return Err(err);
            }
        };
        let effect_pipeline = match create_program(
            device,
            context,
            "codevar-ui-effect",
            UI_EFFECT_VERT_SPIRV,
            UI_EFFECT_FRAG_SPIRV,
            EFFECT_PUSH_BYTES,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                pipeline.destroy(device);
                staging.destroy(device);
                vertices.destroy(device);
                return Err(err);
            }
        };
        Ok(Self {
            context,
            frame,
            geometry: UiGeometry::new(),
            target,
            pipeline: ManuallyDrop::new(pipeline),
            effect_pipeline: ManuallyDrop::new(effect_pipeline),
            vertices: ManuallyDrop::new(vertices),
            staging: ManuallyDrop::new(staging),
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
}

impl UiPipe<'_> {
    /// Drains the shared queue into the CPU replay scratch.
    fn expand_frame(&mut self) {
        let mut list = self.frame.list.borrow_mut();
        self.geometry.expand(&mut list);
    }

    /// Stages the current geometry and copies it into the vertex buffer.
    ///
    /// A frame with no vertices leaves the buffer untouched, so an empty UI
    /// still shows what the previous frame drew under the clear color.
    fn upload(&mut self, ctx: &mut PipeCtx) -> Result<(), CompositorError> {
        let device = self.context.device();
        let vertices = self.geometry.vertices();
        if vertices.is_empty() {
            return Ok(());
        }
        // SAFETY: `UiVertex` is `repr(C)` with no padding and `vertices` is a
        // contiguous slice of initialized vertices, so its bytes are the
        // vertex stream the shader reads.
        let bytes =
            unsafe { core::slice::from_raw_parts(vertices.as_ptr().cast::<u8>(), size_of_val(vertices)) };
        let needed = bytes.len() as u64;
        ensure_buffer(device, &mut self.vertices, VERTEX_USAGE, needed)?;
        ensure_buffer(device, &mut self.staging, STAGING_USAGE, needed)?;
        self.staging.write(device, 0, bytes)?;
        let encoder = ctx.command_encoder();
        graphics::copy_buffer(encoder, &self.staging, 0, &self.vertices, 0, needed)
    }

    /// Records the draw calls of one frame inside the open rendering scope.
    ///
    /// The two programs share the vertex buffer, so a frame walks its spans in
    /// paint order and binds the effect program only for the spans that carry
    /// an effect.
    fn record(&self, ctx: &mut PipeCtx) {
        if self.geometry.is_empty() {
            return;
        }
        let width = ctx.width();
        let height = ctx.height();
        let viewport = [(width as f32).to_bits(), (height as f32).to_bits()];
        let viewport_bytes = viewport.as_slice();
        let encoder = ctx.command_encoder();
        // SAFETY: `record` runs inside the rendering scope opened by
        // `begin_render`, both pipelines were built for the target format, and
        // the bound vertex buffer matches the input layout of both.
        unsafe {
            encoder.set_vertex_buffer(0, BufferBinding::new_unchecked(self.vertices.raw(), 0, None));
            encoder.set_render_pipeline(self.pipeline.raw());
            encoder.set_immediates(self.pipeline.layout(), 0, viewport_bytes);

            let mut on_effect = false;
            let mut bound_clip: Option<Option<Rect>> = None;
            for span in self.geometry.spans() {
                let wants_effect = span.effect.is_some();
                if wants_effect != on_effect {
                    on_effect = wants_effect;
                    bound_clip = None;
                    if wants_effect {
                        encoder.set_render_pipeline(self.effect_pipeline.raw());
                    } else {
                        encoder.set_render_pipeline(self.pipeline.raw());
                        encoder.set_immediates(self.pipeline.layout(), 0, viewport_bytes);
                    }
                }
                if let Some(effect) = span.effect {
                    let params = effect_params(&effect, [width as f32, height as f32]);
                    encoder.set_immediates(self.effect_pipeline.layout(), 0, &params.words());
                }
                if bound_clip != Some(span.clip) {
                    bound_clip = Some(span.clip);
                    let scissor = scissor_for(span.clip, width, height);
                    encoder.set_scissor_rect(&wgpu_hal::Rect {
                        x: scissor.x,
                        y: scissor.y,
                        w: scissor.width,
                        h: scissor.height,
                    });
                }
                encoder.draw(span.first_vertex, span.vertex_count, 0, 1);
            }
        }
    }

    /// Drains the frame, uploads its vertices and records it into the target.
    fn render(&mut self, ctx: &mut PipeCtx) -> Result<(), CompositorError> {
        self.expand_frame();
        self.upload(ctx)?;
        ctx.begin_render(self.target.framebuffer(), Some(self.clear))?;
        self.record(ctx);
        ctx.end_render();
        Ok(())
    }
}

impl PipeSource for UiPipe<'_> {
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
        Box::pin(async move {
            renderer.render(ctx)?;
            Ok(PipeOutcome::Keep)
        })
    }
}

impl PipeOffscreen for UiPipe<'_> {
    fn offscreen_framebuffer_count(&self) -> usize {
        1
    }

    fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer<'_>> {
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
        let device = self.context.device();
        // SAFETY: each field is taken exactly once, here, and the pipe is not
        // used afterwards; the compositor waited for the device before it
        // dropped the pipe, so none of the handles is referenced by a
        // submission in flight.
        unsafe {
            ManuallyDrop::take(&mut self.staging).destroy(device);
            ManuallyDrop::take(&mut self.vertices).destroy(device);
            ManuallyDrop::take(&mut self.effect_pipeline).destroy(device);
            ManuallyDrop::take(&mut self.pipeline).destroy(device);
        }
    }
}

/// Builds one of the two UI programs from an embedded shader pair.
///
/// Both programs share the [`UiVertex`] layout and the single color
/// attachment of the target, so they differ only by their shaders and by the
/// width of the immediate block the vertex stage reads.
///
/// `immediate_size` is the byte size of that block: the plain program maps the
/// viewport with 8 bytes, the effect program takes the whole
/// [`EFFECT_PUSH_BYTES`] block. A sampled-image layout is not passed — the UI
/// shaders read no texture.
fn create_program(
    device: &wgpu_hal::gles::Device,
    context: &WgContext,
    label: &str,
    vertex_spirv: &[u8],
    fragment_spirv: &[u8],
    immediate_size: u32,
) -> Result<GraphicsPipeline, CompositorError> {
    let descriptor = GraphicsPipelineDesc {
        label,
        vertex_spirv,
        fragment_spirv,
        vertex_stride: size_of::<UiVertex>() as u32,
        attributes: &VERTEX_ATTRIBUTES,
        color_format: context.format(),
        bind_group_layout: None,
        immediate_size,
    };
    GraphicsPipeline::new(device, &descriptor)
}

/// Grows `buffer` to hold at least `required` bytes.
///
/// A no-op when the current allocation already fits, so the steady state never
/// touches the allocator.
fn ensure_buffer(
    device: &wgpu_hal::gles::Device,
    buffer: &mut GpuBuffer,
    usage: wgt::BufferUses,
    required: u64,
) -> Result<(), CompositorError> {
    if required <= buffer.size() {
        return Ok(());
    }
    let grown = required.max(buffer.size().saturating_mul(2));
    let replacement = GpuBuffer::new(device, usage, grown)?;
    let previous = core::mem::replace(buffer, replacement);
    previous.destroy(device);
    Ok(())
}
