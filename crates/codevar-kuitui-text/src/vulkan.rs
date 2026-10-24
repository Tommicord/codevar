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

//! The Vulkan text pipe: upload the coverage the atlas changed and draw
//! the indexed runs.
//!
//! A [`TextPipe`] is a compositor [`PipeSource`] like the polygon UI pipe.
//! It owns one offscreen target the mix pass folds into the presentation
//! image, a `text.vert`/`text.frag` pipeline with one descriptor set per
//! atlas page, and three persistent host-visible buffers: the vertex
//! stream, the index stream and the staging buffer the atlas coverage
//! travels through.
//!
//! # Why there is no readback
//!
//! The CPU atlas is the source of truth for coverage, so the GPU never has
//! to send a texel back. Each frame the pipe takes the atlas's dirty
//! rectangles, coalesces them with [`coalesce`], packs their bytes into the
//! staging buffer and records one `vkCmdCopyBufferToImage` per rectangle.
//! The page images stay resident between frames and are only transitioned
//! around those copies.
//!
//! # Why it is demand-driven
//!
//! [`TextFrame::take_dirty`] is the only thing that lets a frame rebuild the
//! mesh and re-upload coverage. A frame that was not marked skips both and
//! replays the persistent vertex and index buffers, so a still UI does no
//! host-to-device transfer at all.
//!
//! # One frame in flight
//!
//! The compositor waits on the previous frame's fence before a pipe future
//! runs, so growing a buffer here may destroy its predecessor: no command
//! buffer in flight references it.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::size_of;

use ash::vk;

use codevar_gpu_vk::base::CompositorError;
use codevar_gpu_vk::mix::load_spir_v;
use codevar_gpu_vk::offscreen::{OffscreenFramebuffer, OffscreenTarget, PipeOffscreen};
use codevar_gpu_vk::pipe::PipeSource;
use codevar_gpu_vk::pipe_ctx::{PipeCtx, PipeFuture, PipeOutcome};
use codevar_gpu_vk::pipeline::PipelineContext;
use codevar_gpu_vk::ui::scissor_for;
use codevar_kuitui::geometry::Rect;
use codevar_kuitui::shaders::{TEXT_FRAG_SPIRV, TEXT_VERT_SPIRV};

use crate::atlas::{DirtyRect, TextFrame, TextVertex};
use crate::dirty::coalesce;
use crate::mesh::Index;

/// Format of every atlas page image.
///
/// The fragment shader reads the red channel as coverage, so one channel is
/// enough and the page costs a quarter of an RGBA one.
const PAGE_FORMAT: vk::Format = vk::Format::R8_UNORM;

/// Byte size of the `text.vert` push-constant block (`vec2 viewport`).
const VIEWPORT_PUSH_BYTES: u32 = 8;

/// Initial size of the atlas staging buffer.
const DEFAULT_STAGING_BYTES: u64 = 256 * 1024;

/// Initial number of vertices the vertex buffer holds.
const DEFAULT_VERTEX_CAPACITY: usize = 16 * 1024;

/// Initial number of indices the index buffer holds.
const DEFAULT_INDEX_CAPACITY: usize = 32 * 1024;

/// One run as the command buffer sees it: a slice of the shared vertex and
/// index buffers plus the page and clip to bind before drawing it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DrawRun {
    /// Atlas page, used to pick the descriptor set.
    page: u32,
    /// First index into the shared index buffer.
    first_index: u32,
    /// Number of indices to draw.
    index_count: u32,
    /// Base vertex added to every index by `vkCmdDrawIndexed`.
    vertex_offset: i32,
    /// Clip to set as the scissor before the draw.
    clip: Option<Rect>,
}

/// A host-visible, coherent buffer that grows but never shrinks.
///
/// The mapping follows the allocation, so `mapped` is valid for `size` bytes
/// for the buffer's whole life. Growing destroys the old buffer and
/// allocation; that is safe because one frame is in flight and the
/// compositor already waited for it.
struct MappedBuffer {
    /// Device that owns the handles. Kept for `Drop`, which has no context.
    device: *const ash::Device,
    /// The device buffer.
    buffer: vk::Buffer,
    /// Memory bound to `buffer`.
    memory: vk::DeviceMemory,
    /// Persistent mapping of `memory`.
    mapped: *mut u8,
    /// Size of the allocation in bytes.
    size: u64,
    /// Usage flags the buffer was created with.
    usage: vk::BufferUsageFlags,
}

impl MappedBuffer {
    /// Allocates a mapped buffer of `size` bytes with `usage`.
    fn new(
        context: &PipelineContext,
        usage: vk::BufferUsageFlags,
        size: u64,
    ) -> Result<Self, CompositorError> {
        let size = size.max(1);
        let (buffer, memory, mapped) = create_mapped_buffer(context, usage, size)?;
        Ok(Self {
            device: context.device() as *const ash::Device,
            buffer,
            memory,
            mapped,
            size,
            usage,
        })
    }

    /// Grows the buffer to hold at least `required` bytes.
    ///
    /// A no-op when the current allocation already fits, so the steady state
    /// never touches the allocator.
    fn ensure(&mut self, context: &PipelineContext, required: u64) -> Result<(), CompositorError> {
        if required <= self.size {
            return Ok(());
        }
        let grown = required.max(self.size.saturating_mul(2));
        let (buffer, memory, mapped) = create_mapped_buffer(context, self.usage, grown)?;
        // SAFETY: the device outlives this buffer (it comes from the
        // `PipelineContext` the pipe borrows), the compositor waited for the
        // previous frame's fence before this future ran, so no command
        // buffer references the old buffer, and the mapping is owned here.
        unsafe { self.release() };
        self.buffer = buffer;
        self.memory = memory;
        self.mapped = mapped;
        self.size = grown;
        Ok(())
    }

    /// Unmaps and destroys the buffer and its memory.
    ///
    /// # Safety
    ///
    /// The device must be alive and no submitted command buffer may still
    /// reference `self.buffer`.
    unsafe fn release(&mut self) {
        // SAFETY: the caller guarantees the device outlives the buffer and no
        // submitted command buffer still references it.
        unsafe {
            let device = &*self.device;
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        // SAFETY: the device comes from the `PipelineContext` the pipe
        // borrows and therefore outlives it, and the compositor releases a
        // pipe only after waiting for the frame that could reference it.
        unsafe { self.release() };
    }
}

/// One atlas page on the GPU: an R8 image, its view, the descriptor that
/// binds it to the sampler and the layout it is currently in.
struct PageTexture {
    /// The page image.
    image: vk::Image,
    /// Device-local memory bound to `image`.
    memory: vk::DeviceMemory,
    /// View bound by the page's descriptor set.
    view: vk::ImageView,
    /// Descriptor set holding `view` for the fragment shader.
    descriptor: vk::DescriptorSet,
    /// Layout `image` is currently in.
    layout: vk::ImageLayout,
}

impl PageTexture {
    /// Destroys the image, memory and view.
    ///
    /// The descriptor set is freed when the pool is destroyed, so it is not
    /// released here.
    ///
    /// # Safety
    ///
    /// `device` must be the device that created the handles and no submitted
    /// command buffer may reference them.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller guarantees `device` created the handles and no
        // submitted command buffer references them.
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A compositor pipe that renders a [`TextFrame`] into an offscreen target.
///
/// Create it with [`TextPipe::new`], register it with
/// [`Compositor::add_pipe`](codevar_gpu_vk::compositor::Compositor::add_pipe)
/// and keep the [`TextFrame`] to record the UI's text. Register the polygon UI
/// pipe first so the mix pass folds the text over the geometry.
pub struct TextPipe<'p> {
    /// The pipeline context that owns every handle below.
    context: &'p PipelineContext,
    /// The shared frame the pipe drains.
    frame: TextFrame,
    /// The offscreen target the mix pass folds into the frame.
    target: OffscreenTarget<'p>,
    /// Pipeline layout: the `text.vert` push range and set 0.
    pipeline_layout: vk::PipelineLayout,
    /// Set 0 layout: one combined image sampler.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Pool the per-page descriptor sets come from.
    descriptor_pool: vk::DescriptorPool,
    /// The `text.vert`/`text.frag` graphics pipeline.
    pipeline: vk::Pipeline,
    /// Sampler shared by every page descriptor.
    sampler: vk::Sampler,
    /// R8 page images, created lazily in atlas page order.
    pages: Vec<PageTexture>,
    /// Persistent host-visible vertex stream.
    vertices: MappedBuffer,
    /// Persistent host-visible index stream.
    indices: MappedBuffer,
    /// Persistent host-visible staging buffer for coverage uploads.
    staging: MappedBuffer,
    /// Scratch for the coalesced dirty rectangles, reused every frame.
    regions: Vec<DirtyRect>,
    /// Draw runs of the current mesh, reused every frame.
    draws: Vec<DrawRun>,
    /// Color the text target is cleared to before a redraw.
    clear: [f32; 4],
    /// Mix weight of the text target.
    weight: f32,
}

impl<'p> TextPipe<'p> {
    /// Creates a pipe that renders `frame` into a new offscreen target.
    ///
    /// The target, the three buffers, the sampler, the descriptor pool and
    /// the pipeline are all created here. Every partially created object is
    /// released if a later step fails.
    ///
    /// # Errors
    ///
    /// * the [`OffscreenTarget::new`] failures;
    /// * [`CompositorError::BufferCreate`], [`CompositorError::NoSuitableMemoryType`],
    ///   [`CompositorError::MemoryAllocate`], [`CompositorError::MemoryBind`],
    ///   [`CompositorError::BufferMap`] — a buffer could not be created;
    /// * [`CompositorError::SamplerCreate`], [`CompositorError::DescriptorSetLayoutCreate`],
    ///   [`CompositorError::DescriptorPoolCreate`], [`CompositorError::PipelineLayoutCreate`],
    ///   [`CompositorError::ShaderLoad`], [`CompositorError::ShaderModuleCreate`],
    ///   [`CompositorError::PipelineCreate`] — the text pipeline could not be built.
    pub fn new(context: &'p PipelineContext, frame: TextFrame) -> Result<Self, CompositorError> {
        let max_pages = frame.atlas().config().max_pages.max(1);
        let target = OffscreenTarget::new(
            context,
            context.width(),
            context.height(),
            PipelineContext::color_format(),
        )?;
        let vertex_bytes = (DEFAULT_VERTEX_CAPACITY * size_of::<TextVertex>()) as u64;
        let index_bytes = (DEFAULT_INDEX_CAPACITY * size_of::<Index>()) as u64;
        let vertices = MappedBuffer::new(context, vk::BufferUsageFlags::VERTEX_BUFFER, vertex_bytes)?;
        let indices = MappedBuffer::new(context, vk::BufferUsageFlags::INDEX_BUFFER, index_bytes)?;
        let staging = MappedBuffer::new(context, vk::BufferUsageFlags::TRANSFER_SRC, DEFAULT_STAGING_BYTES)?;

        let mut pipe = Self {
            context,
            frame,
            target,
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            pipeline: vk::Pipeline::null(),
            sampler: vk::Sampler::null(),
            pages: Vec::new(),
            vertices,
            indices,
            staging,
            regions: Vec::new(),
            draws: Vec::new(),
            clear: [0.0; 4],
            weight: 1.0,
        };
        pipe.sampler = create_sampler(context.device())?;
        pipe.descriptor_set_layout = create_descriptor_set_layout(context.device())?;
        pipe.descriptor_pool = create_descriptor_pool(context.device(), max_pages)?;
        let (layout, pipeline) = create_text_pipeline(context, pipe.descriptor_set_layout)?;
        pipe.pipeline_layout = layout;
        pipe.pipeline = pipeline;
        Ok(pipe)
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
    /// four-byte-aligned rows and copied into the page images. Each page is
    /// transitioned out of and back into `SHADER_READ_ONLY_OPTIMAL` around
    /// its copies. The atlas's dirty list is cleared once every rectangle is
    /// recorded, so a failed upload is retried next frame.
    fn upload_atlas(&mut self, ctx: &mut PipeCtx) -> Result<(), CompositorError> {
        let context = self.context;
        let device = ctx.device();
        let command_buffer = ctx.command_buffer();

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
            self.staging.ensure(context, required)?;

            let mut offset = 0u64;
            for region in &self.regions {
                let page = region.page;
                ensure_pages(
                    context,
                    &mut self.pages,
                    self.descriptor_pool,
                    self.descriptor_set_layout,
                    self.sampler,
                    page,
                    page_size,
                )?;
                let Some(pixels) = atlas.page_pixels(page) else {
                    continue;
                };
                let image = self.pages[page as usize].image;
                let old_layout = self.pages[page as usize].layout;
                if old_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
                    // SAFETY: `image` belongs to this pipe, the command buffer
                    // is recording and the page is not an attachment.
                    unsafe {
                        record_image_barrier(
                            device,
                            command_buffer,
                            image,
                            old_layout,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        );
                    }
                }

                let stride = align4(region.width) as usize;
                let source_stride = page_size as usize;
                // SAFETY: `pixels` is the full `page_size * page_size` page
                // owned by the atlas, the staging buffer has room for this
                // rectangle's padded rows and the mapping is writable.
                unsafe {
                    for row in 0..region.height as usize {
                        let source = pixels
                            .as_ptr()
                            .add((region.y as usize + row) * source_stride + region.x as usize);
                        let destination = self
                            .staging
                            .mapped
                            .add(offset as usize + row * stride);
                        core::ptr::copy_nonoverlapping(source, destination, region.width as usize);
                    }
                }

                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .buffer_row_length(align4(region.width))
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: region.x as i32,
                        y: region.y as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: region.width,
                        height: region.height,
                        depth: 1,
                    });
                // SAFETY: the staging buffer holds this rectangle's rows at
                // `offset`, the image is in `TRANSFER_DST_OPTIMAL` and the
                // region lies inside the page.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        self.staging.buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        core::slice::from_ref(&copy),
                    );
                    record_image_barrier(
                        device,
                        command_buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                }
                self.pages[page as usize].layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                offset = offset.saturating_add(staged_bytes(*region));
            }
        }

        self.frame.with(|atlas, _| atlas.clear_dirty());
        Ok(())
    }

    /// Rebuilds the shared vertex and index buffers from the frame's runs.
    fn upload_mesh(&mut self) -> Result<(), CompositorError> {
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
        self.vertices
            .ensure(self.context, (vertex_total * size_of::<TextVertex>()) as u64)?;
        self.indices
            .ensure(self.context, (index_total * size_of::<Index>()) as u64)?;

        let mut vertex_base = 0usize;
        let mut index_base = 0usize;
        for run in runs.iter() {
            let vertices = run.mesh.vertices();
            let indices = run.mesh.indices();
            // SAFETY: the buffers were grown to hold every run's data, the
            // mappings cover their whole allocation, and the source slices
            // are distinct from the destinations.
            unsafe {
                if !vertices.is_empty() {
                    core::ptr::copy_nonoverlapping(
                        vertices.as_ptr().cast::<u8>(),
                        self.vertices
                            .mapped
                            .add(vertex_base * size_of::<TextVertex>()),
                        core::mem::size_of_val(vertices),
                    );
                }
                if !indices.is_empty() {
                    core::ptr::copy_nonoverlapping(
                        indices.as_ptr().cast::<u8>(),
                        self.indices
                            .mapped
                            .add(index_base * size_of::<Index>()),
                        core::mem::size_of_val(indices),
                    );
                }
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
        Ok(())
    }

    /// Records one indexed draw per run inside the open rendering scope.
    fn record_draws(&self, ctx: &mut PipeCtx) {
        if self.draws.is_empty() {
            return;
        }
        let device = ctx.device();
        let command_buffer = ctx.command_buffer();
        // SAFETY: `record_draws` runs inside the dynamic-rendering scope
        // opened by `begin_render`, the pipeline was built for the target
        // format and the bindings match its layout.
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            let buffers = [self.vertices.buffer];
            let offsets = [0_u64];
            device.cmd_bind_vertex_buffers(command_buffer, 0, &buffers, &offsets);
            device.cmd_bind_index_buffer(command_buffer, self.indices.buffer, 0, vk::IndexType::UINT32);

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

            let mut bound_page = u32::MAX;
            let mut bound_clip: Option<Option<Rect>> = None;
            for draw in &self.draws {
                if draw.page != bound_page {
                    bound_page = draw.page;
                    if let Some(page) = self.pages.get(draw.page as usize) {
                        let sets = [page.descriptor];
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &sets,
                            &[],
                        );
                    }
                }
                if bound_clip != Some(draw.clip) {
                    bound_clip = Some(draw.clip);
                    let scissor = scissor_for(draw.clip, ctx.width(), ctx.height());
                    device.cmd_set_scissor(command_buffer, 0, core::slice::from_ref(&scissor));
                }
                device.cmd_draw_indexed(
                    command_buffer,
                    draw.index_count,
                    1,
                    draw.first_index,
                    draw.vertex_offset,
                    0,
                );
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
            self.upload_mesh()?;
        }
        ctx.begin_render(0, Some(self.clear))?;
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

impl Drop for TextPipe<'_> {
    fn drop(&mut self) {
        let device = self.context.device();
        for page in self.pages.drain(..) {
            // SAFETY: the page was created by this pipe on this device and
            // the compositor waited for the frame that could reference it.
            unsafe { page.destroy(device) };
        }
        // SAFETY: every handle was created by `new` on this device and is not
        // referenced by a submitted command buffer.
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
            }
        }
    }
}

/// Finds the memory type index that supports `required` among `type_bits`.
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let count = properties.memory_type_count.min(32);
    (0..count).find(|index| {
        let supported = type_bits & (1 << index) != 0;
        let flags = properties.memory_types[*index as usize].property_flags;
        supported && flags.contains(required)
    })
}

/// Creates a host-visible, coherent buffer and maps it for its whole life.
fn create_mapped_buffer(
    context: &PipelineContext,
    usage: vk::BufferUsageFlags,
    size: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), CompositorError> {
    let device = context.device();
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
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
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let Some(memory_type) = find_memory_type(&properties, requirements.memory_type_bits, wanted) else {
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
    // SAFETY: `memory` was just allocated for this device and `buffer` is not
    // bound to anything.
    if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: neither object is referenced elsewhere.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(CompositorError::MemoryBind(err));
    }
    // SAFETY: the memory is host-visible and not yet used by the GPU.
    let mapped = match unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty()) }
    {
        Ok(pointer) => pointer.cast::<u8>(),
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

/// Creates the linear, clamped sampler every page descriptor shares.
fn create_sampler(device: &ash::Device) -> Result<vk::Sampler, CompositorError> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0);
    // SAFETY: a plain, well-formed sampler descriptor.
    unsafe { device.create_sampler(&info, None) }.map_err(CompositorError::SamplerCreate)
}

/// Creates set 0: one combined image sampler read by the fragment shader.
fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, CompositorError> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(core::slice::from_ref(&binding));
    // SAFETY: the binding is well-formed and lives for the call.
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(CompositorError::DescriptorSetLayoutCreate)
}

/// Creates a pool holding one combined image sampler per atlas page.
fn create_descriptor_pool(
    device: &ash::Device,
    max_pages: usize,
) -> Result<vk::DescriptorPool, CompositorError> {
    let max_sets = max_pages as u32;
    let size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(max_sets);
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(core::slice::from_ref(&size));
    // SAFETY: the pool sizes are well-formed and fit the set count.
    unsafe { device.create_descriptor_pool(&info, None) }.map_err(CompositorError::DescriptorPoolCreate)
}

/// Creates the `text.vert`/`text.frag` pipeline layout and pipeline.
fn create_text_pipeline(
    context: &PipelineContext,
    set_layout: vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::Pipeline), CompositorError> {
    let device = context.device();
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(VIEWPORT_PUSH_BYTES);
    let set_layouts = [set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(core::slice::from_ref(&push_range));
    // SAFETY: the set layout and push range are well-formed and live for the
    // call.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(CompositorError::PipelineLayoutCreate)?;

    match build_text_pipeline(device, pipeline_layout) {
        Ok(pipeline) => Ok((pipeline_layout, pipeline)),
        Err(err) => {
            // SAFETY: the layout was just created and owns no pipeline yet.
            unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
            Err(err)
        }
    }
}

/// Builds the graphics pipeline from the shaders `codevar-kuitui` compiled.
fn build_text_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, CompositorError> {
    let vertex_module = create_shader_module(device, TEXT_VERT_SPIRV)?;
    let fragment_module = match create_shader_module(device, TEXT_FRAG_SPIRV) {
        Ok(module) => module,
        Err(err) => {
            // SAFETY: the vertex module was just created and is unused.
            unsafe { device.destroy_shader_module(vertex_module, None) };
            return Err(err);
        }
    };
    let pipeline = build_text_graphics_pipeline(device, pipeline_layout, vertex_module, fragment_module);
    // SAFETY: the modules are not needed once the pipeline exists.
    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }
    pipeline
}

/// Creates a shader module from an embedded SPIR-V image.
fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, CompositorError> {
    let words = load_spir_v(bytes)?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is a validated, 4-byte-aligned SPIR-V module that
    // outlives the call.
    unsafe { device.create_shader_module(&info, None) }.map_err(CompositorError::ShaderModuleCreate)
}

/// Creates the graphics pipeline described by the [`TextVertex`] layout.
fn build_text_graphics_pipeline(
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
        .stride(size_of::<TextVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attributes = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: core::mem::offset_of!(TextVertex, position) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: core::mem::offset_of!(TextVertex, uv) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: core::mem::offset_of!(TextVertex, color) as u32,
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
        .ok_or(CompositorError::Internal("the driver returned no text pipeline"))
}

/// Creates the GPU textures for atlas pages `0..=page`.
fn ensure_pages(
    context: &PipelineContext,
    pages: &mut Vec<PageTexture>,
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    page: u32,
    page_size: u32,
) -> Result<(), CompositorError> {
    while pages.len() <= page as usize {
        let texture = create_page_texture(context, pool, set_layout, sampler, page_size)?;
        pages.push(texture);
    }
    Ok(())
}

/// Creates one R8 page image, its device memory, view and descriptor.
fn create_page_texture(
    context: &PipelineContext,
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    page_size: u32,
) -> Result<PageTexture, CompositorError> {
    let device = context.device();
    let image = create_page_image(context, page_size)?;
    let memory = match allocate_page_memory(context, image) {
        Ok(memory) => memory,
        Err(err) => {
            // SAFETY: `image` is bound to nothing yet.
            unsafe { device.destroy_image(image, None) };
            return Err(err);
        }
    };
    let view = match create_page_view(device, image) {
        Ok(view) => view,
        Err(err) => {
            // SAFETY: both objects were just created and nothing references
            // them.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return Err(err);
        }
    };
    let descriptor = match allocate_page_descriptor(device, pool, set_layout, sampler, view) {
        Ok(descriptor) => descriptor,
        Err(err) => {
            // SAFETY: the view, image and memory were just created and are not
            // referenced by a submitted command buffer.
            unsafe {
                device.destroy_image_view(view, None);
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return Err(err);
        }
    };
    Ok(PageTexture {
        image,
        memory,
        view,
        descriptor,
        layout: vk::ImageLayout::UNDEFINED,
    })
}

/// Creates a `page_size` × `page_size` R8 image.
fn create_page_image(context: &PipelineContext, page_size: u32) -> Result<vk::Image, CompositorError> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(PAGE_FORMAT)
        .extent(vk::Extent3D {
            width: page_size,
            height: page_size,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: a plain, well-formed image descriptor.
    unsafe { context.device().create_image(&info, None) }.map_err(CompositorError::ImageCreate)
}

/// Allocates device-local memory for a page image and binds it.
fn allocate_page_memory(
    context: &PipelineContext,
    image: vk::Image,
) -> Result<vk::DeviceMemory, CompositorError> {
    let device = context.device();
    // SAFETY: `image` was just created on this device.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    // SAFETY: the instance and physical device outlive `context`.
    let properties = unsafe {
        context
            .instance()
            .get_physical_device_memory_properties(context.physical_device())
    };
    // Prefer device-local memory; fall back to any type the image accepts.
    let memory_type = find_memory_type(
        &properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .or_else(|| {
        find_memory_type(
            &properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
        )
    })
    .ok_or(CompositorError::NoSuitableMemoryType)?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    // SAFETY: the allocation matches an advertised memory type and size.
    let memory =
        unsafe { device.allocate_memory(&allocate_info, None) }.map_err(CompositorError::MemoryAllocate)?;
    // SAFETY: `memory` was allocated for this image and the image is not bound.
    if let Err(err) = unsafe { device.bind_image_memory(image, memory, 0) } {
        // SAFETY: nothing references `memory` yet.
        unsafe { device.free_memory(memory, None) };
        return Err(CompositorError::MemoryBind(err));
    }
    Ok(memory)
}

/// Creates a `TYPE_2D` R8 view over a page image.
fn create_page_view(device: &ash::Device, image: vk::Image) -> Result<vk::ImageView, CompositorError> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(PAGE_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the view matches the image's format and extent.
    unsafe { device.create_image_view(&info, None) }.map_err(CompositorError::ImageViewCreate)
}

/// Allocates and writes the descriptor set that binds a page view.
fn allocate_page_descriptor(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    view: vk::ImageView,
) -> Result<vk::DescriptorSet, CompositorError> {
    let layouts = [set_layout];
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    // SAFETY: the pool was created against this set layout and holds enough
    // descriptors for one set per page.
    let sets =
        unsafe { device.allocate_descriptor_sets(&info) }.map_err(CompositorError::DescriptorSetAllocate)?;
    let set = *sets.first().ok_or(CompositorError::Internal(
        "the driver returned no text descriptor set",
    ))?;
    let image_info = vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(core::slice::from_ref(&image_info));
    // SAFETY: the set is not referenced by a pending command buffer and the
    // binding matches the set layout.
    unsafe { device.update_descriptor_sets(core::slice::from_ref(&write), &[]) };
    Ok(set)
}

/// Records a layout transition of a page image.
///
/// # Safety
///
/// The command buffer must be recording and `image` must not be the render
/// target of an open rendering scope.
unsafe fn record_image_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_stage, src_access) = transition_source(old_layout);
    let (dst_stage, dst_access) = transition_destination(new_layout);
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the caller guarantees the command buffer is recording and
    // `image` is not the target of an open rendering scope.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            core::slice::from_ref(&barrier),
        );
    }
}
/// The source stage and access a barrier waits on for `layout`.
fn transition_source(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
        }
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

/// The destination stage and access a barrier publishes for `layout`.
fn transition_destination(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
        }
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

/// Rounds `value` up to the next multiple of four.
///
/// `vkCmdCopyBufferToImage` requires a four-byte-aligned `bufferOffset`, and
/// an R8 row of an odd width is not a multiple of four by itself.
#[inline]
fn align4(value: u32) -> u32 {
    (value + 3) & !3
}

/// Bytes a dirty rectangle occupies in the staging buffer.
#[inline]
fn staged_bytes(region: DirtyRect) -> u64 {
    u64::from(align4(region.width)) * u64::from(region.height)
}

#[cfg(test)]
mod tests {
    //! Staging layout arithmetic.

    use super::*;

    #[test]
    fn rows_are_padded_to_four_bytes() {
        assert_eq!(align4(0), 0);
        assert_eq!(align4(1), 4);
        assert_eq!(align4(3), 4);
        assert_eq!(align4(4), 4);
        assert_eq!(align4(5), 8);
        assert_eq!(align4(1024), 1024);
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
        assert_eq!(staged_bytes(region), 8, "two rows of four bytes");

        let wide = DirtyRect {
            page: 0,
            x: 0,
            y: 0,
            width: 16,
            height: 4,
        };
        assert_eq!(staged_bytes(wide), 64);
        assert_eq!(
            staged_bytes(wide),
            staged_bytes(wide),
            "aligned rows stay aligned"
        );
    }
}
