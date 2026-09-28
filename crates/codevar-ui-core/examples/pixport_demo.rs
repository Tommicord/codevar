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

//! Pixport terminal demo: renders Vulkan frames directly to the terminal
//! as ANSI true-color output without creating a Wayland window.
//!
//! This example demonstrates the [`Pixport`] and [`run_terminal_render_loop`]
//! functionality by:
//! 1. Creating a Vulkan pipeline with offscreen render target
//! 2. Rendering frames using the renderer subsystem
//! 3. Capturing each frame via Pixport and writing ANSI to stdout
//!
//! Run it:
//!
//! ```text
//! cargo run -p codevar-ui-core --example pixport_demo
//! ```
//!
//! Note: This requires a terminal that supports true color (24-bit RGB)
//! and UTF-8. Most modern terminals (kitty, alacritty, wezterm, gnome-terminal,
//! Windows Terminal, etc.) support this.

use ash::vk;
use codevar_consoleutil::console_ansi::{cursor, erase};
use codevar_ui_core::ui_pipeline::PipelineContext;
use codevar_ui_core::ui_renderer::{
    FrameContext, RenderLayer, RendererError, RendererSubsystem, load_spir_v,
};
use codevar_ui_core::ui_terminal_pixport::ansi::PixportFilterType;
use codevar_ui_core::ui_terminal_pixport::{CellDensity, ClearPolicy, PixportConfig, render_loop};

/// Render target width in pixels.
const WIDTH: u32 = 640;
/// Render target height in pixels.
const HEIGHT: u32 = 480;

/// Embedded vertex shader SPIR-V.
static VERT_SPIRV: &[u8] = include_bytes!("shaders/triangle.vert.spv");
/// Embedded fragment shader SPIR-V.
static FRAG_SPIRV: &[u8] = include_bytes!("shaders/triangle.frag.spv");

fn main() {
    run();
}

/// Drives the whole example: Vulkan pipeline, render loop, pixport capture, and teardown.
fn run() {
    codevar_consoleutil::init_ansi_support();
    let term_cols = codevar_consoleutil::detect_terminal_width();
    let term_rows = codevar_consoleutil::detect_terminal_height();
    let modifiers = [0u64]; // Linear modifier
    let pipeline = PipelineContext::new(WIDTH, HEIGHT, &modifiers).unwrap();
    let _target = pipeline.render_target();
    let mut renderer = RendererSubsystem::new(&pipeline);
    renderer.start().unwrap();
    let pixport_config = PixportConfig {
        max_cols: term_cols,
        max_rows: term_rows,
        cell_density: CellDensity::HalfBlocks,
        clear_policy: ClearPolicy::ClearBeforeFrame,
        filter: PixportFilterType::Lanczos3,
    };
    render_loop::run_terminal_render_loop(
        &pipeline,
        renderer,
        TriangleLayer::new(&pipeline).unwrap(),
        codevar_ui_core::ui_terminal_pixport::render_loop::RenderLoopConfig {
            frame_rate: codevar_time_core::TimeDuration::from_millis(16),
            pixport_config,
        },
    )
    .unwrap();
}

/// Creates a shader module from validated SPIR-V words.
fn create_shader_module(device: &ash::Device, words: &[u32]) -> Result<vk::ShaderModule, RendererError> {
    let module_info = vk::ShaderModuleCreateInfo::default().code(words);
    unsafe { device.create_shader_module(&module_info, None) }.map_err(RendererError::ShaderModuleCreate)
}

/// Creates the graphics pipeline for the placeholder triangle.
fn create_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
) -> Result<vk::Pipeline, RendererError> {
    let color_format = PipelineContext::color_format();
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
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
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
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&color_blend_attachment));
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
    let result = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            core::slice::from_ref(&pipeline_info),
            None,
        )
    };
    let pipelines = result.map_err(|(_, err)| RendererError::PipelineCreate(err))?;
    pipelines
        .first()
        .copied()
        .ok_or(RendererError::Internal("driver returned no graphics pipeline"))
}

struct TriangleLayer<'p> {
    context: &'p PipelineContext,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl<'p> TriangleLayer<'p> {
    fn new(context: &'p PipelineContext) -> Result<Self, RendererError> {
        let device = context.device();
        let vertex_words = load_spir_v(VERT_SPIRV)?;
        let fragment_words = load_spir_v(FRAG_SPIRV)?;
        let vertex_module = create_shader_module(device, &vertex_words)?;
        let fragment_module = match create_shader_module(device, &fragment_words) {
            Ok(module) => module,
            Err(err) => {
                unsafe { device.destroy_shader_module(vertex_module, None) };
                return Err(err);
            }
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(err) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(fragment_module, None);
                }
                return Err(RendererError::PipelineLayoutCreate(err));
            }
        };

        let pipeline = match create_pipeline(device, pipeline_layout, vertex_module, fragment_module) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(fragment_module, None);
                }
                return Err(err);
            }
        };

        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }

        Ok(Self {
            context,
            pipeline_layout,
            pipeline,
        })
    }
}

impl RenderLayer for TriangleLayer<'_> {
    fn name(&self) -> &str {
        "triangle"
    }

    fn priority(&self) -> i32 {
        100
    }

    fn render(&mut self, frame: &mut FrameContext<'_>) -> Result<(), RendererError> {
        let device = frame.device();
        let command_buffer = frame.command_buffer();
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
        Ok(())
    }
}

impl Drop for TriangleLayer<'_> {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_shaders_are_valid_spir_v() {
        assert!(load_spir_v(VERT_SPIRV).is_ok());
        assert!(load_spir_v(FRAG_SPIRV).is_ok());
    }
}
