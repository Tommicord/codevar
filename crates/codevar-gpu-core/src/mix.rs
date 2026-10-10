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

//! The fullscreen mix pass: embedded SPIR-V loading, the [`MixPush`]
//! push-constant payload and the descriptor set, sampler and graphics
//! pipeline that fold the offscreen framebuffers into the presentation
//! target.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use ash::vk;

use crate::base::{Compositor, CompositorError};
use crate::pipeline::PipelineContext;

/// Embedded vertex shader of the mix pass.
static MIX_VERT_SPIRV: &[u8] = include_bytes!("shaders/mix.vert.spv");

/// Embedded fragment shader of the mix pass.
static MIX_FRAG_SPIRV: &[u8] = include_bytes!("shaders/mix.frag.spv");

/// SPIR-V magic number as stored in a little-endian module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Decodes a little-endian SPIR-V blob into shader words.
///
/// This is the `no_std` replacement for `ash::util::read_spv` (which is
/// only available with ash's `std` feature): it validates the file length
/// and the module magic, then widens each 4-byte group into a `u32`.
///
/// # Errors
///
/// * [`CompositorError::ShaderLoad`] — the byte length is not a multiple
///   of 4, or the SPIR-V magic number does not match.
pub fn load_spir_v(bytes: &[u8]) -> Result<Vec<u32>, CompositorError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(CompositorError::ShaderLoad(format!(
            "SPIR-V byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let words: Vec<u32> = chunks
        .map(|group| u32::from_le_bytes([group[0], group[1], group[2], group[3]]))
        .collect();
    match words.first() {
        Some(&magic) if magic == SPIRV_MAGIC => Ok(words),
        Some(&magic) => Err(CompositorError::ShaderLoad(format!(
            "bad SPIR-V magic {magic:#010x}"
        ))),
        None => Err(CompositorError::ShaderLoad(String::from(
            "the SPIR-V module is empty",
        ))),
    }
}

/// Builds a shader module from an embedded SPIR-V blob.
///
/// # Errors
///
/// * [`CompositorError::ShaderLoad`] — the embedded module is malformed.
/// * [`CompositorError::ShaderModuleCreate`] — `vkCreateShaderModule`
///   failed.
fn shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, CompositorError> {
    let words = load_spir_v(bytes)?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is a validated, 4-byte-aligned SPIR-V module that
    // outlives the call.
    unsafe { device.create_shader_module(&info, None) }.map_err(CompositorError::ShaderModuleCreate)
}

/// Push constant payload of the mix pass.
///
/// The layout mirrors `MixPush` in `kernel/mix.frag`: `count` at offset
/// 0, then dynamic weight array starting at offset 16. The explicit
/// padding keeps the block identical under the std140, std430 and scalar
/// layout rules, so the byte range pushed by the compositor matches what
/// the shader reads.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MixPush {
    /// Number of framebuffers actually mixed (0 clears the target).
    pub count: u32,
    /// Explicit padding up to offset 16 (matches `layout(offset = 16)`).
    pub pad: [u32; 3],
    /// Mix weights of framebuffers (dynamic size).
    pub weights: Vec<f32>,
}

impl MixPush {
    /// Creates a new MixPush with the given weights.
    #[must_use]
    pub fn new(weights: Vec<f32>) -> Self {
        let count = weights.len() as u32;
        Self {
            count,
            pad: [0; 3],
            weights,
        }
    }

    /// Returns the total byte size of this push constant structure.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        16 + core::mem::size_of::<f32>() * self.weights.len()
    }
}

/// GPU resources of the mix pass, created by [`Compositor::start`](Compositor::start) and
/// released when dropped (which [`Compositor::stop`](Compositor::stop) triggers).
///
/// The descriptor set covers a dynamic number of slots based on the
/// actual number of framebuffers needed.
pub(crate) struct MixResources<'p> {
    /// Device that owns every handle below.
    pub(crate) device: &'p ash::Device,
    /// Set layout with dynamic array binding at binding 0.
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    /// Pool backing the single mix descriptor set.
    pub(crate) descriptor_pool: vk::DescriptorPool,
    /// The mix descriptor set, rewritten every frame.
    pub(crate) descriptor_set: vk::DescriptorSet,
    /// Linear sampler used for the framebuffer reads.
    pub(crate) sampler: vk::Sampler,
    /// Pipeline layout: the set layout plus the `MixPush` range.
    pub(crate) pipeline_layout: vk::PipelineLayout,
    /// Fullscreen-triangle graphics pipeline of the mix pass.
    pub(crate) pipeline: vk::Pipeline,
    /// Maximum number of framebuffers this descriptor set can hold.
    pub(crate) max_framebuffers: u32,
}

impl<'p> MixResources<'p> {
    /// All-null resources for `device`: [`Drop`] is a no-op on them.
    fn empty(device: &'p ash::Device, max_framebuffers: u32) -> Self {
        Self {
            device,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            sampler: vk::Sampler::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            max_framebuffers,
        }
    }

    /// Creates every resource of the mix pass.
    ///
    /// # Errors
    ///
    /// Propagates the [`CompositorError`] of the first step that fails;
    /// dropping the partially built resources releases everything that
    /// was already created.
    pub(crate) fn create(
        context: &'p PipelineContext,
        max_framebuffers: u32,
    ) -> Result<Self, CompositorError> {
        let mut resources = Self::empty(context.device(), max_framebuffers);
        resources.create_descriptors(context)?;
        resources.create_pipeline(context)?;
        Ok(resources)
    }

    /// Creates the descriptor set layout, pool, set and sampler.
    fn create_descriptors(&mut self, context: &PipelineContext) -> Result<(), CompositorError> {
        let device = context.device();
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(self.max_framebuffers)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: the binding array is well-formed and outlives the call.
        self.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
            .map_err(CompositorError::DescriptorSetLayoutCreate)?;

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(self.max_framebuffers);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: one set of samplers matches the pool.
        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(CompositorError::DescriptorPoolCreate)?;

        let layouts = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are valid and the pool has room.
        let sets = unsafe { device.allocate_descriptor_sets(&allocate_info) }
            .map_err(CompositorError::DescriptorSetAllocate)?;
        self.descriptor_set = sets
            .first()
            .copied()
            .ok_or(CompositorError::Internal(
                "the driver allocated no mix descriptor set",
            ))?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .min_lod(0.0)
            .max_lod(0.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false);
        // SAFETY: every field of `sampler_info` is within spec limits.
        self.sampler =
            unsafe { device.create_sampler(&sampler_info, None) }.map_err(CompositorError::SamplerCreate)?;
        Ok(())
    }

    /// Creates the pipeline layout and the mix graphics pipeline.
    fn create_pipeline(&mut self, context: &PipelineContext) -> Result<(), CompositorError> {
        let device = context.device();
        // Use a fixed maximum size for push constants (256 bytes should be enough
        // for 16-bit count + 12 padding + up to 60 weights)
        let max_push_size = 256u32;
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(max_push_size);
        let push_ranges = [push_range];
        let set_layouts = [self.descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: the layout references this pass's set layout and push
        // range, both well-formed and alive for the call.
        self.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(CompositorError::PipelineLayoutCreate)?;

        let vertex_module = shader_module(device, MIX_VERT_SPIRV)?;
        let fragment_module = match shader_module(device, MIX_FRAG_SPIRV) {
            Ok(module) => module,
            Err(err) => {
                // SAFETY: the vertex module was just created.
                unsafe { device.destroy_shader_module(vertex_module, None) };
                return Err(err);
            }
        };
        let pipeline = create_mix_graphics_pipeline(
            device,
            self.pipeline_layout,
            vertex_module,
            fragment_module,
            PipelineContext::color_format(),
        );
        // SAFETY: both modules are no longer needed once the pipeline
        // was created (or failed to be).
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }
        self.pipeline = pipeline?;
        Ok(())
    }
}

impl Drop for MixResources<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created by `create` on this device,
        // is not referenced by any in-flight command buffer (the
        // compositor waits for the device before releasing these
        // resources), and the descriptor set goes away with its pool.
        // Null handles are never passed to the driver.
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.sampler, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
        }
    }
}

/// Creates the fullscreen-triangle pipeline of the mix pass.
fn create_mix_graphics_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    color_format: vk::Format,
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
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
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
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
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
    // SAFETY: every create-info is well-formed, the rendering info
    // declares the single color attachment format of dynamic rendering,
    // and the modules are alive for the call.
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
        .ok_or(CompositorError::Internal("the driver returned no mix pipeline"))
}

/// Builds the push-constant payload of the mix pass.
pub(crate) fn mix_push(weights: &[f32]) -> MixPush {
    MixPush::new(weights.to_vec())
}

#[cfg(test)]
mod tests {
    //! Push-constant layout and embedded SPIR-V loading.

    use super::*;

    #[test]
    fn mix_push_matches_the_shader_layout() {
        let weights = vec![1.0, 1.0, 1.0, 0.5];
        let push = mix_push(&weights);
        assert_eq!(push.count, 4);
        assert_eq!(push.weights, weights);
        assert_eq!(push.pad, [0; 3]);
        assert_eq!(push.byte_size(), 16 + 4 * 4);
    }

    #[test]
    fn load_spir_v_accepts_embedded_modules_and_rejects_garbage() {
        let module = load_spir_v(include_bytes!("shaders/mix.vert.spv")).unwrap();
        assert_eq!(module[0], SPIRV_MAGIC);

        let fragment = load_spir_v(include_bytes!("shaders/mix.frag.spv")).unwrap();
        assert!(fragment.len() > module.len());

        assert!(load_spir_v(&[]).is_err(), "empty module");
        assert!(load_spir_v(&[0_u8; 8]).is_err(), "bad magic");
        let embedded = include_bytes!("shaders/mix.vert.spv");
        assert!(load_spir_v(&embedded[..6]).is_err(), "not a multiple of 4");
    }
}
