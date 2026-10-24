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

//! Reusable graphics primitives for the WebGPU backend.
//!
//! `codevar-gpu-wg` drives `wgpu-hal` directly, so a pipe that wants to draw
//! must create its own shader modules, pipeline layout, bind groups,
//! textures and buffers. This module factors those calls out of the pipes: a
//! pipe describes what it wants — a 2D vertex pipeline that samples one
//! texture and may receive a block of immediate data — and the helpers here
//! perform the raw, error-prone `wgpu-hal` calls.
//!
//! The module plays the role `codevar-gpu-vk`'s `ui` and `mix` modules play
//! on the desktop, but is shaped around `wgpu-hal` instead of `ash`. It
//! compiles on `wasm32` only; the [SPIR-V loader](load_spir_v) is
//! target-independent, so its tests run on the host.

use alloc::vec::Vec;

use crate::types::CompositorError;

/// The first word of every valid SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Loads embedded SPIR-V bytes into the 32-bit words `wgpu-hal` expects.
///
/// Shader images are embedded as bytes (`include_bytes!`) so a backend can
/// name them without pulling in a build-time dependency; `wgpu-hal` consumes
/// them as words.
///
/// # Errors
///
/// * [`CompositorError::ShaderLoad`] — the byte length is not a multiple of
///   four, the module is empty, or it does not start with the SPIR-V magic
///   number.
pub fn load_spir_v(bytes: &[u8]) -> Result<Vec<u32>, CompositorError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(CompositorError::ShaderLoad(alloc::format!(
            "SPIR-V byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let words: Vec<u32> = chunks
        .map(|group| u32::from_le_bytes([group[0], group[1], group[2], group[3]]))
        .collect();
    match words.first() {
        Some(&magic) if magic == SPIRV_MAGIC => Ok(words),
        Some(&magic) => Err(CompositorError::ShaderLoad(alloc::format!(
            "bad SPIR-V magic {magic:#010x}"
        ))),
        None => Err(CompositorError::ShaderLoad(alloc::string::String::from(
            "the SPIR-V module is empty",
        ))),
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use alloc::vec::Vec;

    use wgpu_hal::gles::{
        BindGroup, BindGroupLayout, Buffer, CommandEncoder, Device, PipelineLayout, RenderPipeline, Sampler,
        ShaderModule, Texture, TextureView,
    };
    use wgpu_hal::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutFlags, BufferCopy,
        BufferDescriptor, BufferTextureCopy, CommandEncoder as _, CopyExtent, Device as _, FormatAspects,
        MemoryFlags, PipelineLayoutDescriptor, PipelineLayoutFlags, ProgrammableStage,
        RenderPipelineDescriptor, SamplerDescriptor, ShaderInput, ShaderModuleDescriptor, TextureBinding,
        TextureCopyBase, TextureDescriptor, TextureViewDescriptor, VertexBufferLayout, VertexProcessor,
    };
    use wgpu_types as wgt;

    use super::load_spir_v;
    use crate::types::CompositorError;

    /// Creates a shader module from embedded SPIR-V bytes.
    ///
    /// Both stages of a pipeline share this call; the module is released with
    /// [`destroy_shader_module`] once the pipeline that links it exists.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::ShaderLoad`] — the embedded module is malformed.
    /// * [`CompositorError::Graphics`] — the driver rejected the module.
    pub fn create_shader_module(
        device: &Device,
        label: &str,
        bytes: &[u8],
    ) -> Result<ShaderModule, CompositorError> {
        let words = load_spir_v(bytes)?;
        let descriptor = ShaderModuleDescriptor {
            label: Some(label),
            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
        };
        // SAFETY: `words` is a validated SPIR-V module that lives for the
        // call and the descriptor borrows nothing past it.
        unsafe { device.create_shader_module(&descriptor, ShaderInput::SpirV(&words)) }
            .map_err(|err| CompositorError::graphics("create_shader_module", err))
    }

    /// Releases a shader module once the pipeline that linked it exists.
    ///
    /// # Safety
    ///
    /// The caller must ensure no pipeline built from `module` is still in
    /// use.
    pub unsafe fn destroy_shader_module(device: &Device, module: ShaderModule) {
        // SAFETY: the caller upholds the contract above.
        unsafe { device.destroy_shader_module(module) };
    }

    /// The bind-group layout a 2D pipeline uses to sample one texture.
    ///
    /// The SPIR-V `sampler2D` the shared UI shaders declare is a *combined*
    /// image sampler, but `wgpu-hal`'s GLSL backend splits it into a texture
    /// and an implicit sampler. The layout therefore declares a filterable
    /// `Texture2D` at binding 0 and a filtering `Sampler` at binding 1; the
    /// driver binds both to the same texture unit.
    pub struct SampledImageLayout {
        layout: BindGroupLayout,
    }

    impl SampledImageLayout {
        /// Creates the layout described on the type.
        ///
        /// # Errors
        ///
        /// [`CompositorError::Graphics`] when the driver rejects the layout.
        pub fn new(device: &Device) -> Result<Self, CompositorError> {
            let entries = [
                wgt::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgt::ShaderStages::FRAGMENT,
                    ty: wgt::BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Float { filterable: true },
                        view_dimension: wgt::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgt::ShaderStages::FRAGMENT,
                    ty: wgt::BindingType::Sampler(wgt::SamplerBindingType::Filtering),
                    count: None,
                },
            ];
            let descriptor = BindGroupLayoutDescriptor {
                label: Some("codevar-sampled-image-layout"),
                flags: BindGroupLayoutFlags::empty(),
                entries: &entries,
            };
            // SAFETY: the entries are unique and sorted by binding, and the
            // descriptor borrows nothing past the call.
            let layout = unsafe { device.create_bind_group_layout(&descriptor) }
                .map_err(|err| CompositorError::graphics("create_bind_group_layout", err))?;
            Ok(Self { layout })
        }

        /// The raw layout, for a pipeline descriptor.
        #[inline]
        #[must_use]
        pub fn raw(&self) -> &BindGroupLayout {
            &self.layout
        }

        /// Pairs `view` with `sampler` into a bind group of this layout.
        ///
        /// # Errors
        ///
        /// [`CompositorError::Graphics`] when the driver rejects the group.
        pub fn bind_group(
            &self,
            device: &Device,
            sampler: &Sampler,
            view: &TextureView,
        ) -> Result<SampledImage, CompositorError> {
            let entries = [
                BindGroupEntry {
                    binding: 0,
                    resource_index: 0,
                    count: 1,
                },
                BindGroupEntry {
                    binding: 1,
                    resource_index: 0,
                    count: 1,
                },
            ];
            let textures = [TextureBinding {
                view,
                usage: wgt::TextureUses::RESOURCE,
            }];
            let samplers = [sampler];
            let descriptor = BindGroupDescriptor {
                label: Some("codevar-sampled-image"),
                layout: &self.layout,
                buffers: &[],
                samplers: &samplers,
                textures: &textures,
                entries: &entries,
                acceleration_structures: &[],
                external_textures: &[],
            };
            // SAFETY: the entries match the layout — a texture at binding 0
            // and a sampler at binding 1 — and both resources outlive the
            // group.
            let group = unsafe { device.create_bind_group(&descriptor) }
                .map_err(|err| CompositorError::graphics("create_bind_group", err))?;
            Ok(SampledImage { group })
        }

        /// Releases the layout.
        pub fn destroy(self, device: &Device) {
            // SAFETY: the layout is owned here and no bind group of it
            // outlives the call.
            unsafe { device.destroy_bind_group_layout(self.layout) };
        }
    }

    /// A sampled texture paired with a sampler, ready to bind.
    pub struct SampledImage {
        group: BindGroup,
    }

    impl SampledImage {
        /// The raw bind group, for `CommandEncoder::set_bind_group`.
        #[inline]
        #[must_use]
        pub fn raw(&self) -> &BindGroup {
            &self.group
        }

        /// Releases the bind group.
        pub fn destroy(self, device: &Device) {
            // SAFETY: the group is owned here and no pipeline that uses it
            // outlives the call.
            unsafe { device.destroy_bind_group(self.group) };
        }
    }

    /// Creates the clamp-to-edge, linear-filtering sampler the 2D pipelines
    /// share.
    ///
    /// # Errors
    ///
    /// [`CompositorError::Graphics`] when the driver rejects the sampler.
    pub fn create_sampler(device: &Device) -> Result<Sampler, CompositorError> {
        let descriptor = SamplerDescriptor {
            label: Some("codevar-linear-sampler"),
            address_modes: [wgt::AddressMode::ClampToEdge; 3],
            mag_filter: wgt::FilterMode::Linear,
            min_filter: wgt::FilterMode::Linear,
            mipmap_filter: wgt::MipmapFilterMode::Nearest,
            lod_clamp: 0.0..32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        };
        // SAFETY: the descriptor borrows nothing and the device outlives the
        // returned sampler.
        unsafe { device.create_sampler(&descriptor) }
            .map_err(|err| CompositorError::graphics("create_sampler", err))
    }

    /// A 2D texture with a sampled view, used as an atlas page.
    ///
    /// The texture carries `COPY_DST | RESOURCE`, so it can receive an
    /// upload and then be sampled.
    pub struct SampledTexture2D {
        texture: Texture,
        view: TextureView,
        width: u32,
        height: u32,
        format: wgt::TextureFormat,
    }

    impl SampledTexture2D {
        /// Allocates a `width` × `height` sampled texture in `format`.
        ///
        /// # Errors
        ///
        /// [`CompositorError::Graphics`] when the driver rejects the texture
        /// or its view; a partially created texture is released first.
        pub fn new(
            device: &Device,
            width: u32,
            height: u32,
            format: wgt::TextureFormat,
        ) -> Result<Self, CompositorError> {
            let descriptor = TextureDescriptor {
                label: Some("codevar-sampled-texture"),
                size: wgt::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgt::TextureDimension::D2,
                format,
                usage: wgt::TextureUses::COPY_DST | wgt::TextureUses::RESOURCE,
                memory_flags: MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            // SAFETY: the extent is non-zero and the device outlives the
            // returned handles.
            let texture = unsafe { device.create_texture(&descriptor) }
                .map_err(|err| CompositorError::graphics("create_texture", err))?;
            let view_descriptor = TextureViewDescriptor {
                label: Some("codevar-sampled-texture-view"),
                format,
                dimension: wgt::TextureViewDimension::D2,
                usage: wgt::TextureUses::RESOURCE,
                range: wgt::ImageSubresourceRange {
                    aspect: wgt::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                },
            };
            // SAFETY: the view covers the only mip level and layer of a
            // texture created on this device.
            let view = match unsafe { device.create_texture_view(&texture, &view_descriptor) } {
                Ok(view) => view,
                Err(err) => {
                    // SAFETY: the texture is referenced by nothing else yet.
                    unsafe { device.destroy_texture(texture) };
                    return Err(CompositorError::graphics("create_texture_view", err));
                }
            };
            Ok(Self {
                texture,
                view,
                width,
                height,
                format,
            })
        }

        /// The color texture.
        #[inline]
        #[must_use]
        pub fn texture(&self) -> &Texture {
            &self.texture
        }

        /// The sampled view of the texture.
        #[inline]
        #[must_use]
        pub fn view(&self) -> &TextureView {
            &self.view
        }

        /// Width in pixels.
        #[inline]
        #[must_use]
        pub const fn width(&self) -> u32 {
            self.width
        }

        /// Height in pixels.
        #[inline]
        #[must_use]
        pub const fn height(&self) -> u32 {
            self.height
        }

        /// Format of the texture.
        #[inline]
        #[must_use]
        pub const fn format(&self) -> wgt::TextureFormat {
            self.format
        }

        /// Releases the view and the texture.
        pub fn destroy(self, device: &Device) {
            // SAFETY: both handles are owned here and referenced by nothing
            // else.
            unsafe {
                device.destroy_texture_view(self.view);
                device.destroy_texture(self.texture);
            }
        }
    }

    /// Everything a 2D vertex pipeline needs to be built.
    pub struct GraphicsPipelineDesc<'a> {
        /// Debug label.
        pub label: &'a str,
        /// Vertex stage SPIR-V.
        pub vertex_spirv: &'a [u8],
        /// Fragment stage SPIR-V.
        pub fragment_spirv: &'a [u8],
        /// Bytes between consecutive vertices.
        pub vertex_stride: u32,
        /// The vertex attributes, at their shader locations.
        pub attributes: &'a [wgt::VertexAttribute],
        /// Format of the single color attachment.
        pub color_format: wgt::TextureFormat,
        /// The sampled-image layout the pipeline expects at group 0.
        pub bind_group_layout: Option<&'a BindGroupLayout>,
        /// Size, in bytes, of the immediate-data block (0 for none).
        pub immediate_size: u32,
    }

    /// A pipeline layout and the graphics pipeline built from it.
    pub struct GraphicsPipeline {
        layout: PipelineLayout,
        pipeline: RenderPipeline,
    }

    impl GraphicsPipeline {
        /// Builds a pipeline from `desc`.
        ///
        /// The blend state is straight alpha (`src_alpha`,
        /// `one_minus_src_alpha`), matching the desktop backend.
        ///
        /// # Errors
        ///
        /// * [`CompositorError::ShaderLoad`] — a stage is not valid SPIR-V.
        /// * [`CompositorError::Graphics`] — the driver rejected the layout
        ///   or the pipeline.
        pub fn new(device: &Device, desc: &GraphicsPipelineDesc<'_>) -> Result<Self, CompositorError> {
            let vertex_module = create_shader_module(device, desc.label, desc.vertex_spirv)?;
            let fragment_module = match create_shader_module(device, desc.label, desc.fragment_spirv) {
                Ok(module) => module,
                Err(err) => {
                    // SAFETY: the vertex module is referenced by nothing.
                    unsafe { device.destroy_shader_module(vertex_module) };
                    return Err(err);
                }
            };

            let group_layouts = [desc.bind_group_layout];
            let bind_group_layouts: &[Option<&BindGroupLayout>] = if desc.bind_group_layout.is_some() {
                &group_layouts
            } else {
                &[]
            };
            let layout_descriptor = PipelineLayoutDescriptor {
                label: Some(desc.label),
                flags: PipelineLayoutFlags::empty(),
                bind_group_layouts,
                immediate_size: desc.immediate_size,
            };
            // SAFETY: the descriptor borrows the bind-group layouts and the
            // device outlives the returned layout.
            let layout = match unsafe { device.create_pipeline_layout(&layout_descriptor) } {
                Ok(layout) => layout,
                Err(err) => {
                    // SAFETY: both modules are referenced by nothing.
                    unsafe {
                        device.destroy_shader_module(vertex_module);
                        device.destroy_shader_module(fragment_module);
                    }
                    return Err(CompositorError::graphics("create_pipeline_layout", err));
                }
            };

            // The inferred type is `naga::back::PipelineConstants`; the
            // shared shaders declare no overridable constants.
            let constants = Default::default();
            let vertex_stage = ProgrammableStage {
                module: &vertex_module,
                entry_point: "main",
                constants: &constants,
                zero_initialize_workgroup_memory: false,
            };
            let fragment_stage = ProgrammableStage {
                module: &fragment_module,
                entry_point: "main",
                constants: &constants,
                zero_initialize_workgroup_memory: false,
            };
            let vertex_buffer = Some(VertexBufferLayout {
                array_stride: u64::from(desc.vertex_stride),
                step_mode: wgt::VertexStepMode::Vertex,
                attributes: desc.attributes,
            });
            let vertex_buffers = core::slice::from_ref(&vertex_buffer);
            let color_target = Some(wgt::ColorTargetState {
                format: desc.color_format,
                blend: Some(wgt::BlendState::ALPHA_BLENDING),
                write_mask: wgt::ColorWrites::ALL,
            });
            let color_targets = core::slice::from_ref(&color_target);
            let pipeline_descriptor = RenderPipelineDescriptor {
                label: Some(desc.label),
                layout: &layout,
                vertex_processor: VertexProcessor::Standard {
                    vertex_buffers,
                    vertex_stage,
                },
                primitive: wgt::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgt::MultisampleState::default(),
                fragment_stage: Some(fragment_stage),
                color_targets,
                multiview_mask: None,
                cache: None,
            };
            // SAFETY: the modules, layout and vertex state live for the
            // call; the pipeline keeps its own references afterwards.
            let pipeline = match unsafe { device.create_render_pipeline(&pipeline_descriptor) } {
                Ok(pipeline) => pipeline,
                Err(err) => {
                    // SAFETY: the layout and both modules are referenced by
                    // nothing.
                    unsafe {
                        device.destroy_pipeline_layout(layout);
                        device.destroy_shader_module(vertex_module);
                        device.destroy_shader_module(fragment_module);
                    }
                    return Err(CompositorError::graphics("create_render_pipeline", err));
                }
            };
            // The program is linked; the modules have no further use. The
            // layout is kept: `set_bind_group` and `set_immediates` name it.
            // SAFETY: the pipeline no longer references either module.
            unsafe {
                device.destroy_shader_module(vertex_module);
                device.destroy_shader_module(fragment_module);
            }
            Ok(Self { layout, pipeline })
        }

        /// The pipeline layout, for `set_bind_group` and `set_immediates`.
        #[inline]
        #[must_use]
        pub fn layout(&self) -> &PipelineLayout {
            &self.layout
        }

        /// The graphics pipeline, for `set_render_pipeline`.
        #[inline]
        #[must_use]
        pub fn raw(&self) -> &RenderPipeline {
            &self.pipeline
        }

        /// Releases the pipeline and its layout.
        pub fn destroy(self, device: &Device) {
            // SAFETY: both handles are owned here and referenced by nothing
            // else.
            unsafe {
                device.destroy_render_pipeline(self.pipeline);
                device.destroy_pipeline_layout(self.layout);
            }
        }
    }

    /// A GPU buffer plus its byte size.
    pub struct GpuBuffer {
        buffer: Buffer,
        size: u64,
    }

    impl GpuBuffer {
        /// Allocates a buffer of `size` bytes with `usage`.
        ///
        /// A `MAP_WRITE` buffer is a CPU staging buffer: on WebGL2 the driver
        /// keeps it host-side, which is why a pipe uploads through it rather
        /// than drawing from it directly.
        ///
        /// # Errors
        ///
        /// * [`CompositorError::Internal`] — `size` is zero.
        /// * [`CompositorError::Graphics`] — the driver refused the buffer.
        pub fn new(device: &Device, usage: wgt::BufferUses, size: u64) -> Result<Self, CompositorError> {
            if size == 0 {
                return Err(CompositorError::Internal("GpuBuffer::new: zero size"));
            }
            let memory_flags = if usage.intersects(wgt::BufferUses::MAP_WRITE) {
                MemoryFlags::PREFER_COHERENT
            } else {
                MemoryFlags::empty()
            };
            let descriptor = BufferDescriptor {
                label: Some("codevar-buffer"),
                size,
                usage,
                memory_flags,
            };
            // SAFETY: the size is non-zero and the device outlives the
            // returned buffer.
            let buffer = unsafe { device.create_buffer(&descriptor) }
                .map_err(|err| CompositorError::graphics("create_buffer", err))?;
            Ok(Self { buffer, size })
        }

        /// The raw buffer.
        #[inline]
        #[must_use]
        pub fn raw(&self) -> &Buffer {
            &self.buffer
        }

        /// Size in bytes.
        #[inline]
        #[must_use]
        pub const fn size(&self) -> u64 {
            self.size
        }

        /// Writes `data` at `offset`.
        ///
        /// The buffer must carry `BufferUses::MAP_WRITE`; use this to fill a
        /// staging buffer, then move it to a drawable buffer with
        /// [`copy_buffer`] or to a texture with [`upload_texture`].
        ///
        /// # Errors
        ///
        /// * [`CompositorError::Internal`] — the write leaves the buffer.
        /// * [`CompositorError::Graphics`] — the driver refused the mapping.
        pub fn write(&self, device: &Device, offset: u64, data: &[u8]) -> Result<(), CompositorError> {
            if data.is_empty() {
                return Ok(());
            }
            let end = offset
                .checked_add(data.len() as u64)
                .ok_or(CompositorError::Internal("GpuBuffer::write: offset overflow"))?;
            if end > self.size {
                return Err(CompositorError::Internal(
                    "GpuBuffer::write: data leaves the buffer",
                ));
            }
            let range = offset..end;
            // SAFETY: `range` lies inside the buffer (checked above) and the
            // buffer is not mapped concurrently.
            let mapping = unsafe { device.map_buffer(&self.buffer, range.clone()) }
                .map_err(|err| CompositorError::graphics("map_buffer", err))?;
            // SAFETY: the driver mapped at least `data.len()` bytes starting
            // at `offset`; the source and destination do not overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
            }
            if !mapping.is_coherent {
                // SAFETY: the range was just mapped and is still mapped.
                unsafe { device.flush_mapped_ranges(&self.buffer, core::iter::once(range)) };
            }
            // SAFETY: the buffer was mapped above and is not mapped after.
            unsafe { device.unmap_buffer(&self.buffer) };
            Ok(())
        }

        /// Releases the buffer.
        pub fn destroy(self, device: &Device) {
            // SAFETY: the buffer is owned here and is not mapped.
            unsafe { device.destroy_buffer(self.buffer) };
        }
    }

    /// Records a copy of `len` bytes from `staging` into `dst`.
    ///
    /// WebGL2 cannot bind a CPU-mapped buffer as a vertex or index buffer, so
    /// a pipe fills a `MAP_WRITE | COPY_SRC` staging buffer and copies it into
    /// a `COPY_DST` buffer it then draws from.
    ///
    /// # Errors
    ///
    /// [`CompositorError::Internal`] — the copy is empty or leaves either
    /// buffer.
    pub fn copy_buffer(
        encoder: &mut CommandEncoder,
        staging: &GpuBuffer,
        dst: &GpuBuffer,
        len: u64,
    ) -> Result<(), CompositorError> {
        if len == 0 {
            return Ok(());
        }
        if len > staging.size() || len > dst.size() {
            return Err(CompositorError::Internal("copy_buffer: copy leaves a buffer"));
        }
        let size = wgt::BufferSize::new(len).ok_or(CompositorError::Internal("copy_buffer: empty copy"))?;
        let copies = core::iter::once(BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        });
        // SAFETY: both buffers belong to this encoder's device, the copy
        // stays inside both (checked above), and the encoder is recording.
        unsafe { encoder.copy_buffer_to_buffer(staging.raw(), dst.raw(), copies) };
        Ok(())
    }

    /// Records a copy of `texture`'s whole extent from `staging`.
    ///
    /// `bytes_per_row` is the staging row stride and must be a multiple of
    /// 256 (the WebGPU copy alignment); only the texture's width reaches it
    /// from each row, so the caller pads the rows. `staging` must be a
    /// `MAP_WRITE | COPY_SRC` buffer already filled with [`GpuBuffer::write`].
    ///
    /// # Errors
    ///
    /// [`CompositorError::Internal`] — the staged rows do not fit `staging`.
    pub fn upload_texture(
        encoder: &mut CommandEncoder,
        staging: &GpuBuffer,
        texture: &SampledTexture2D,
        bytes_per_row: u32,
    ) -> Result<(), CompositorError> {
        let height = texture.height();
        let needed = u64::from(bytes_per_row) * u64::from(height);
        if needed > staging.size() {
            return Err(CompositorError::Internal(
                "upload_texture: staging buffer too small",
            ));
        }
        let region = BufferTextureCopy {
            buffer_layout: wgt::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            texture_base: TextureCopyBase {
                mip_level: 0,
                array_layer: 0,
                origin: wgt::Origin3d::ZERO,
                aspect: FormatAspects::COLOR,
            },
            size: CopyExtent {
                width: texture.width(),
                height,
                depth: 1,
            },
        };
        // SAFETY: the region fits the staging buffer (checked above) and the
        // texture's only mip level and layer; the encoder is recording.
        unsafe {
            encoder.copy_buffer_to_texture(staging.raw(), texture.texture(), core::iter::once(region));
        }
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
pub use imp::{
    GpuBuffer, GraphicsPipeline, GraphicsPipelineDesc, SampledImage, SampledImageLayout, SampledTexture2D,
    copy_buffer, create_sampler, create_shader_module, destroy_shader_module, upload_texture,
};

#[cfg(test)]
mod tests {
    //! The target-independent SPIR-V loader.

    use super::*;

    #[test]
    fn rejects_a_length_that_is_not_a_multiple_of_four() {
        let error = load_spir_v(&[0x03, 0x02, 0x23]).expect_err("a partial word is rejected");
        assert!(error.to_string().contains("multiple of 4"), "{error}");
    }

    #[test]
    fn rejects_an_empty_module() {
        let error = load_spir_v(&[]).expect_err("an empty module is rejected");
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn rejects_a_wrong_magic_number() {
        let error = load_spir_v(&[1, 2, 3, 4]).expect_err("a bad magic is rejected");
        assert!(error.to_string().contains("magic"), "{error}");
    }

    #[test]
    fn loads_a_minimal_module() {
        let bytes = SPIRV_MAGIC.to_le_bytes();
        let words = load_spir_v(&bytes).expect("the magic alone is a valid header");
        assert_eq!(words, [SPIRV_MAGIC]);
    }
}
