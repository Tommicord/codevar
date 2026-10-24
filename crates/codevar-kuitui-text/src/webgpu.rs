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

//! The WebGPU text pipe: upload the coverage the atlas changed and draw the
//! indexed runs.
//!
//! [`TextPipe`] is the browser counterpart of the `vulkan` adapter and a
//! compositor [`PipeSource`] like the polygon UI pipe. It owns one offscreen
//! target the mix pass folds into the canvas, a `text.vert`/`text.frag`
//! pipeline with one bind group per atlas page, and the buffers the vertex
//! stream, the index stream and the coverage travel through.
//!
//! # Why the uploads stage through a copy
//!
//! WebGL2 cannot bind a CPU-mapped buffer for drawing and cannot copy out of
//! one, so the pipe never draws from the buffer it fills. It writes coverage
//! and geometry into a `MAP_WRITE | COPY_SRC` staging buffer and records a
//! `copy_buffer_to_buffer` (geometry) or `copy_buffer_to_texture` (coverage)
//! into a buffer or texture the GPU owns. The backend reads the staged bytes
//! only when the queue runs, so every span stays at its own staging offset:
//! one frame never overwrites another span's data.
//!
//! # Why there is no readback
//!
//! The CPU atlas is the source of truth for coverage, so the GPU never has
//! to send a texel back. Each frame the pipe takes the atlas's dirty
//! rectangles, coalesces them with [`coalesce`], packs their bytes into the
//! staging buffer and records one copy per rectangle.
//!
//! # Why it is demand-driven
//!
//! [`TextFrame::take_dirty`] is the only thing that lets a frame rebuild the
//! mesh and re-upload coverage. A frame that was not marked skips both and
//! replays the persistent vertex and index buffers, so a still UI does no
//! host-to-device transfer at all.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::{ManuallyDrop, size_of};

use wgpu_hal::BufferBinding;
use wgpu_hal::CommandEncoder as _;
use wgpu_hal::Device as _;
use wgpu_hal::gles::{Device, Sampler};
use wgpu_types as wgt;

use codevar_gpu_wg::base::{
    CompositorError, OffscreenFramebuffer, OffscreenTarget, PipeCtx, PipeFuture, PipeOffscreen, PipeOutcome,
    PipeSource, WgContext,
};
use codevar_gpu_wg::graphics::{
    self, GpuBuffer, GraphicsPipeline, GraphicsPipelineDesc, SampledImage, SampledImageLayout,
    SampledTexture2D, TextureRegion,
};
use codevar_kuitui::geometry::Rect;
use codevar_kuitui::shaders::{TEXT_FRAG_SPIRV, TEXT_VERT_SPIRV};
use libm::{ceilf, floorf};

use crate::atlas::{DirtyRect, TextFrame, TextVertex};
use crate::dirty::coalesce;
use crate::mesh::Index;

/// Format of every atlas page texture.
///
/// The fragment shader reads the red channel as coverage, so one channel is
/// enough and the page costs a quarter of an RGBA one.
const PAGE_FORMAT: wgt::TextureFormat = wgt::TextureFormat::R8Unorm;

/// Byte size of the `text.vert` immediate block (`vec2 viewport`).
const VIEWPORT_PUSH_BYTES: u32 = 8;

/// Initial size of each staging buffer.
const DEFAULT_STAGING_BYTES: u64 = 256 * 1024;

/// Initial number of vertices the vertex buffer holds.
const DEFAULT_VERTEX_CAPACITY: usize = 16 * 1024;

/// Initial number of indices the index buffer holds.
const DEFAULT_INDEX_CAPACITY: usize = 32 * 1024;

/// Usage of the vertex buffer: drawable and writable by a copy.
const VERTEX_USAGE: wgt::BufferUses = wgt::BufferUses::VERTEX.union(wgt::BufferUses::COPY_DST);

/// Usage of the index buffer: drawable and writable by a copy.
const INDEX_USAGE: wgt::BufferUses = wgt::BufferUses::INDEX.union(wgt::BufferUses::COPY_DST);

/// Usage of a staging buffer: CPU-writable and copyable from.
const STAGING_USAGE: wgt::BufferUses = wgt::BufferUses::MAP_WRITE.union(wgt::BufferUses::COPY_SRC);

/// The `TextVertex` fields at their `text.vert` locations.
const VERTEX_ATTRIBUTES: [wgt::VertexAttribute; 3] = [
    wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    },
    wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x2,
        offset: 8,
        shader_location: 1,
    },
    wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 2,
    },
];

/// One run as the command encoder sees it: a slice of the shared vertex and
/// index buffers plus the page and clip to bind before drawing it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DrawRun {
    /// Atlas page, used to pick the bind group.
    page: u32,
    /// First index into the shared index buffer.
    first_index: u32,
    /// Number of indices to draw.
    index_count: u32,
    /// Base vertex added to every index by `draw_indexed`.
    vertex_offset: i32,
    /// Clip to set as the scissor before the draw.
    clip: Option<Rect>,
}

/// One atlas page on the GPU: an R8 texture and the bind group that pairs it
/// with the shared sampler.
struct WgPage {
    /// The page texture and its sampled view.
    texture: SampledTexture2D,
    /// The texture and sampler bound to the fragment shader.
    image: SampledImage,
}

/// A compositor pipe that renders a [`TextFrame`] into an offscreen target.
///
/// Create it with [`TextPipe::new`], register it with the compositor and keep
/// the [`TextFrame`] to record the UI's text. Register the polygon UI pipe
/// first so the mix pass folds the text over the geometry.
pub struct TextPipe<'p> {
    /// The pipeline context that owns every GPU handle below.
    context: &'p WgContext,
    /// The shared frame the pipe drains.
    frame: TextFrame,
    /// The offscreen target the mix pass folds into the canvas.
    target: OffscreenTarget<'p>,
    /// The `text.vert`/`text.frag` pipeline.
    pipeline: ManuallyDrop<GraphicsPipeline>,
    /// Set 0: one sampled page texture.
    image_layout: ManuallyDrop<SampledImageLayout>,
    /// The clamp-to-edge, linear-filtering sampler every page shares.
    sampler: ManuallyDrop<Sampler>,
    /// Persistent vertex stream, filled by a copy.
    vertices: ManuallyDrop<GpuBuffer>,
    /// Persistent index stream, filled by a copy.
    indices: ManuallyDrop<GpuBuffer>,
    /// Staging buffer the atlas coverage travels through.
    staging: ManuallyDrop<GpuBuffer>,
    /// Staging buffer the vertex and index data travel through.
    mesh_staging: ManuallyDrop<GpuBuffer>,
    /// R8 page textures, created lazily in atlas page order.
    pages: Vec<WgPage>,
    /// Scratch for the coalesced dirty rectangles, reused every frame.
    regions: Vec<DirtyRect>,
    /// Draw runs of the current mesh, reused every frame.
    draws: Vec<DrawRun>,
    /// Scratch for one rectangle's padded rows, reused every frame.
    packed: Vec<u8>,
    /// Color the text target is cleared to before a redraw.
    clear: [f32; 4],
    /// Mix weight of the text target.
    weight: f32,
}

impl<'p> TextPipe<'p> {
    /// Creates a pipe that renders `frame` into a new offscreen target.
    ///
    /// The target, the sampler, the buffers and the pipeline are all created
    /// here. Every partially created object is released if a later step
    /// fails.
    ///
    /// # Errors
    ///
    /// * the [`OffscreenTarget::new`] failures;
    /// * [`CompositorError::Graphics`] — the driver refused a sampler,
    ///   buffer, bind group or pipeline;
    /// * [`CompositorError::ShaderLoad`] — an embedded shader is malformed.
    pub fn new(context: &'p WgContext, frame: TextFrame) -> Result<Self, CompositorError> {
        let device = context.device();
        let target = OffscreenTarget::new(context, context.width(), context.height(), context.format())?;
        let image_layout = SampledImageLayout::new(device)?;
        let sampler = match graphics::create_sampler(device) {
            Ok(sampler) => sampler,
            Err(err) => {
                image_layout.destroy(device);
                return Err(err);
            }
        };
        let vertex_bytes = (DEFAULT_VERTEX_CAPACITY * size_of::<TextVertex>()) as u64;
        let index_bytes = (DEFAULT_INDEX_CAPACITY * size_of::<Index>()) as u64;
        let vertices = match GpuBuffer::new(device, VERTEX_USAGE, vertex_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                // SAFETY: the sampler is owned here and referenced by no bind
                // group that outlives it.
                unsafe { device.destroy_sampler(sampler) };
                image_layout.destroy(device);
                return Err(err);
            }
        };
        let indices = match GpuBuffer::new(device, INDEX_USAGE, index_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                vertices.destroy(device);
                // SAFETY: the sampler is owned here and referenced by no bind
                // group that outlives it.
                unsafe { device.destroy_sampler(sampler) };
                image_layout.destroy(device);
                return Err(err);
            }
        };
        let staging = match GpuBuffer::new(device, STAGING_USAGE, DEFAULT_STAGING_BYTES) {
            Ok(buffer) => buffer,
            Err(err) => {
                indices.destroy(device);
                vertices.destroy(device);
                // SAFETY: the sampler is owned here and referenced by no bind
                // group that outlives it.
                unsafe { device.destroy_sampler(sampler) };
                image_layout.destroy(device);
                return Err(err);
            }
        };
        let mesh_staging = match GpuBuffer::new(device, STAGING_USAGE, DEFAULT_STAGING_BYTES) {
            Ok(buffer) => buffer,
            Err(err) => {
                staging.destroy(device);
                indices.destroy(device);
                vertices.destroy(device);
                // SAFETY: the sampler is owned here and referenced by no bind
                // group that outlives it.
                unsafe { device.destroy_sampler(sampler) };
                image_layout.destroy(device);
                return Err(err);
            }
        };
        let descriptor = GraphicsPipelineDesc {
            label: "codevar-text",
            vertex_spirv: TEXT_VERT_SPIRV,
            fragment_spirv: TEXT_FRAG_SPIRV,
            vertex_stride: size_of::<TextVertex>() as u32,
            attributes: &VERTEX_ATTRIBUTES,
            color_format: context.format(),
            bind_group_layout: Some(image_layout.raw()),
            immediate_size: VIEWPORT_PUSH_BYTES,
        };
        let pipeline = match GraphicsPipeline::new(device, &descriptor) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                mesh_staging.destroy(device);
                staging.destroy(device);
                indices.destroy(device);
                vertices.destroy(device);
                // SAFETY: the sampler is owned here and referenced by no bind
                // group that outlives it.
                unsafe { device.destroy_sampler(sampler) };
                image_layout.destroy(device);
                return Err(err);
            }
        };
        Ok(Self {
            context,
            frame,
            target,
            pipeline: ManuallyDrop::new(pipeline),
            image_layout: ManuallyDrop::new(image_layout),
            sampler: ManuallyDrop::new(sampler),
            vertices: ManuallyDrop::new(vertices),
            indices: ManuallyDrop::new(indices),
            staging: ManuallyDrop::new(staging),
            mesh_staging: ManuallyDrop::new(mesh_staging),
            pages: Vec::new(),
            regions: Vec::new(),
            draws: Vec::new(),
            packed: Vec::new(),
            clear: [0.0; 4],
            weight: 1.0,
        })
    }

    /// The frame this pipe renders.
    #[inline]
    #[must_use]
    pub fn frame(&self) -> &TextFrame {
        &self.frame
    }

    /// Sets the color the target is cleared to before a redraw.
    pub fn set_clear(&mut self, clear: [f32; 4]) {
        self.clear = clear;
    }

    /// Sets the mix weight of the target in `[0.0, 1.0]`.
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }
}

impl TextPipe<'_> {
    /// Uploads the atlas rectangles written since the last frame.
    ///
    /// The rectangles are coalesced, packed into the staging buffer with
    /// 256-byte-aligned rows and copied into the page textures. The atlas's
    /// dirty list is cleared once every rectangle is recorded, so a failed
    /// upload is retried next frame.
    fn upload_atlas(&mut self, ctx: &mut PipeCtx) -> Result<(), CompositorError> {
        let device = self.context.device();
        {
            let atlas = self.frame.atlas();
            if atlas.dirty_rects().is_empty() {
                return Ok(());
            }
            let page_size = atlas.config().page_size;
            coalesce(atlas.dirty_rects(), &mut self.regions);

            let mut required = 0u64;
            for region in &self.regions {
                required = required.saturating_add(staged_bytes(*region));
            }
            ensure_buffer(device, &mut self.staging, STAGING_USAGE, required)?;

            let mut offset = 0u64;
            for region in &self.regions {
                let page = region.page;
                ensure_pages(
                    device,
                    &self.image_layout,
                    &self.sampler,
                    &mut self.pages,
                    page,
                    page_size,
                )?;
                let Some(pixels) = atlas.page_pixels(page) else {
                    continue;
                };
                let bytes_per_row = align256(region.width);
                let stride = bytes_per_row as usize;
                let needed = stride * region.height as usize;
                self.packed.clear();
                self.packed.resize(needed, 0);
                let source_stride = page_size as usize;
                for row in 0..region.height as usize {
                    let start = (region.y as usize + row) * source_stride + region.x as usize;
                    let Some(source) = pixels.get(start..start + region.width as usize) else {
                        continue;
                    };
                    let destination = row * stride;
                    self.packed[destination..destination + region.width as usize].copy_from_slice(source);
                }
                self.staging.write(device, offset, &self.packed)?;
                let encoder = ctx.command_encoder();
                graphics::upload_texture(
                    encoder,
                    &self.staging,
                    offset,
                    bytes_per_row,
                    &self.pages[page as usize].texture,
                    TextureRegion {
                        x: region.x,
                        y: region.y,
                        width: region.width,
                        height: region.height,
                    },
                )?;
                offset = offset.saturating_add(staged_bytes(*region));
            }
        }

        self.frame.with(|atlas, _| atlas.clear_dirty());
        Ok(())
    }

    /// Rebuilds the shared vertex and index buffers from the frame's runs.
    fn upload_mesh(&mut self, ctx: &mut PipeCtx) -> Result<(), CompositorError> {
        let device = self.context.device();
        self.draws.clear();
        let runs = self.frame.runs();
        if runs.is_empty() {
            return Ok(());
        }

        let mut vertex_total = 0usize;
        let mut index_total = 0usize;
        for run in runs.iter() {
            vertex_total = vertex_total.saturating_add(run.mesh.vertices().len());
            index_total = index_total.saturating_add(run.mesh.indices().len());
        }
        if vertex_total > i32::MAX as usize || index_total > u32::MAX as usize {
            return Err(CompositorError::Internal(
                "the text mesh has more vertices or indices than a draw can address",
            ));
        }
        let vertex_bytes = (vertex_total * size_of::<TextVertex>()) as u64;
        let index_bytes = (index_total * size_of::<Index>()) as u64;
        ensure_buffer(device, &mut self.vertices, VERTEX_USAGE, vertex_bytes)?;
        ensure_buffer(device, &mut self.indices, INDEX_USAGE, index_bytes)?;
        ensure_buffer(
            device,
            &mut self.mesh_staging,
            STAGING_USAGE,
            vertex_bytes.saturating_add(index_bytes).max(1),
        )?;

        let mut vertex_base = 0usize;
        let mut index_base = 0usize;
        for run in runs.iter() {
            let vertices = run.mesh.vertices();
            let indices = run.mesh.indices();
            if !vertices.is_empty() {
                // SAFETY: `TextVertex` is `repr(C)` and `vertices` is a
                // contiguous slice of initialized vertices.
                let bytes = unsafe {
                    core::slice::from_raw_parts(
                        vertices.as_ptr().cast::<u8>(),
                        core::mem::size_of_val(vertices),
                    )
                };
                self.mesh_staging
                    .write(device, (vertex_base * size_of::<TextVertex>()) as u64, bytes)?;
            }
            if !indices.is_empty() {
                // SAFETY: `Index` is a 32-bit integer and `indices` is a
                // contiguous slice of initialized indices.
                let bytes = unsafe {
                    core::slice::from_raw_parts(
                        indices.as_ptr().cast::<u8>(),
                        core::mem::size_of_val(indices),
                    )
                };
                self.mesh_staging.write(
                    device,
                    vertex_bytes + (index_base * size_of::<Index>()) as u64,
                    bytes,
                )?;
            }
            self.draws.push(DrawRun {
                page: run.page,
                first_index: index_base as u32,
                index_count: indices.len() as u32,
                vertex_offset: vertex_base as i32,
                clip: run.clip,
            });
            vertex_base += vertices.len();
            index_base += indices.len();
        }

        let encoder = ctx.command_encoder();
        graphics::copy_buffer(encoder, &self.mesh_staging, 0, &self.vertices, 0, vertex_bytes)?;
        graphics::copy_buffer(
            encoder,
            &self.mesh_staging,
            vertex_bytes,
            &self.indices,
            0,
            index_bytes,
        )?;
        Ok(())
    }

    /// Records one indexed draw per run inside the open rendering scope.
    fn record_draws(&self, ctx: &mut PipeCtx) {
        if self.draws.is_empty() {
            return;
        }
        let width = ctx.width();
        let height = ctx.height();
        let viewport = [(width as f32).to_bits(), (height as f32).to_bits()];
        let encoder = ctx.command_encoder();
        // SAFETY: `record_draws` runs inside the dynamic-rendering scope
        // opened by `begin_render`, the pipeline was built for the target
        // format and the bindings match its layout.
        unsafe {
            encoder.set_render_pipeline(self.pipeline.raw());
            encoder.set_vertex_buffer(0, BufferBinding::new_unchecked(self.vertices.raw(), 0, None));
            encoder.set_index_buffer(
                BufferBinding::new_unchecked(self.indices.raw(), 0, None),
                wgt::IndexFormat::Uint32,
            );
            encoder.set_immediates(self.pipeline.layout(), 0, &viewport);

            let mut bound_page = u32::MAX;
            let mut bound_clip: Option<Option<Rect>> = None;
            for draw in &self.draws {
                if draw.page != bound_page {
                    bound_page = draw.page;
                    if let Some(page) = self.pages.get(draw.page as usize) {
                        encoder.set_bind_group(self.pipeline.layout(), 0, page.image.raw(), &[]);
                    }
                }
                if bound_clip != Some(draw.clip) {
                    bound_clip = Some(draw.clip);
                    let scissor = scissor_for(draw.clip, width, height);
                    encoder.set_scissor_rect(&scissor);
                }
                encoder.draw_indexed(draw.first_index, draw.index_count, draw.vertex_offset, 0, 1);
            }
        }
    }

    /// Drains the frame's dirty signal, uploads what changed and records the
    /// text into this pipe's target.
    fn render(&mut self, ctx: &mut PipeCtx) -> Result<(), CompositorError> {
        let dirty = self.frame.is_dirty();
        let atlas_dirty = !self.frame.atlas().dirty_rects().is_empty();
        if dirty || atlas_dirty {
            self.upload_atlas(ctx)?;
            self.upload_mesh(ctx)?;
        }
        ctx.begin_render(self.target.framebuffer(), Some(self.clear))?;
        self.record_draws(ctx);
        ctx.end_render();
        if dirty {
            self.frame.take_dirty();
        }
        Ok(())
    }
}

impl PipeSource for TextPipe<'_> {
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
        Box::pin(async move {
            renderer.render(ctx)?;
            Ok(PipeOutcome::Keep)
        })
    }
}

impl PipeOffscreen for TextPipe<'_> {
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

impl Drop for TextPipe<'_> {
    fn drop(&mut self) {
        let device = self.context.device();
        for page in self.pages.drain(..) {
            page.image.destroy(device);
            page.texture.destroy(device);
        }
        // SAFETY: each field is taken exactly once, here, and the pipe is not
        // used afterwards.
        unsafe {
            ManuallyDrop::take(&mut self.mesh_staging).destroy(device);
            ManuallyDrop::take(&mut self.staging).destroy(device);
            ManuallyDrop::take(&mut self.indices).destroy(device);
            ManuallyDrop::take(&mut self.vertices).destroy(device);
            device.destroy_sampler(ManuallyDrop::take(&mut self.sampler));
            ManuallyDrop::take(&mut self.image_layout).destroy(device);
            ManuallyDrop::take(&mut self.pipeline).destroy(device);
        }
    }
}

/// Grows `buffer` to hold at least `required` bytes.
///
/// A no-op when the current allocation already fits, so the steady state
/// never touches the allocator.
fn ensure_buffer(
    device: &Device,
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

/// Creates the page textures up to and including `page`.
fn ensure_pages(
    device: &Device,
    layout: &SampledImageLayout,
    sampler: &Sampler,
    pages: &mut Vec<WgPage>,
    page: u32,
    page_size: u32,
) -> Result<(), CompositorError> {
    while pages.len() <= page as usize {
        let texture = SampledTexture2D::new(device, page_size, page_size, PAGE_FORMAT)?;
        let image = match layout.bind_group(device, sampler, texture.view()) {
            Ok(image) => image,
            Err(err) => {
                texture.destroy(device);
                return Err(err);
            }
        };
        pages.push(WgPage { texture, image });
    }
    Ok(())
}

/// Rounds `value` up to the 256-byte row alignment a texture copy requires.
#[inline]
fn align256(value: u32) -> u32 {
    value.saturating_add(255) & !255
}

/// Bytes a dirty rectangle occupies in the staging buffer.
#[inline]
fn staged_bytes(region: DirtyRect) -> u64 {
    u64::from(align256(region.width)) * u64::from(region.height)
}

/// The scissor rectangle of a command clip inside a `width`×`height` target.
///
/// A clip is snapped outward to whole pixels and clamped to the target, so the
/// scissor is always a valid, non-negative rectangle.
fn scissor_for(clip: Option<Rect>, width: u32, height: u32) -> wgpu_hal::Rect<u32> {
    let full = wgpu_hal::Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
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
    wgpu_hal::Rect {
        x: left,
        y: top,
        w: right.saturating_sub(left),
        h: bottom.saturating_sub(top),
    }
}

#[cfg(test)]
mod tests {
    //! Staging layout arithmetic.

    use super::*;
    use codevar_kuitui::geometry::Offset;

    #[test]
    fn rows_are_padded_to_a_256_byte_boundary() {
        assert_eq!(align256(0), 0);
        assert_eq!(align256(1), 256);
        assert_eq!(align256(255), 256);
        assert_eq!(align256(256), 256);
        assert_eq!(align256(257), 512);
    }

    #[test]
    fn a_rectangle_is_sized_by_its_padded_rows() {
        let region = DirtyRect {
            page: 0,
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        };
        assert_eq!(staged_bytes(region), 512, "two rows of 256 bytes");

        let wide = DirtyRect {
            page: 0,
            x: 0,
            y: 0,
            width: 512,
            height: 4,
        };
        assert_eq!(staged_bytes(wide), 2048);
    }

    #[test]
    fn an_absent_clip_covers_the_whole_target() {
        let scissor = scissor_for(None, 800, 600);
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.w, 800);
        assert_eq!(scissor.h, 600);
    }

    #[test]
    fn a_clip_is_snapped_outward_and_clamped() {
        let clip = Rect::from_min_size(Offset::new(-4.5, 10.25), 124.5, 19.25);
        let scissor = scissor_for(Some(clip), 50, 50);
        assert_eq!(scissor.x, 0, "negative edges clamp to zero");
        assert_eq!(scissor.y, 10, "the top snaps down");
        assert_eq!(scissor.w, 50, "the right edge clamps to the target");
        assert_eq!(scissor.h, 20, "the bottom snaps up");
    }
}
