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

//! Terminal pixel port: reads Vulkan render target pixels and writes
//! them to the terminal as ANSI true-color escape sequences.
//!
//! This module provides the [`Pixport`] struct which:
//! - Creates a Vulkan staging buffer for image readback
//! - Copies the current render target image to the staging buffer
//! - Maps the buffer and converts pixels to ANSI RGB sequences
//! - Writes the resulting ANSI string to stdout
//! - Handles terminal size detection and adjustment

use alloc::string::String;
use core::fmt;

use self::simd::{AnsiColorConverter, PixportFilterType, ansi_cells_to_string};
use crate::ui_pipeline::PipelineContext;
use ash::vk;

use codevar_consoleutil::{detect_terminal_height, detect_terminal_width, write_stdout};
use codevar_wl_protocol::DRM_FORMAT_XRGB8888;

/// Error returned by [`Pixport`] operations.
#[derive(Debug)]
pub enum PixportError {
    /// Vulkan command buffer allocation failed.
    CommandAllocation(vk::Result),
    /// Vulkan command buffer recording failed.
    CommandRecord(vk::Result),
    /// Vulkan command buffer submission failed.
    Submit(vk::Result),
    /// Buffer creation failed.
    BufferCreate(vk::Result),
    /// Memory allocation failed.
    MemoryAllocate(vk::Result),
    /// No suitable memory type found for the staging buffer.
    NoSuitableMemoryType,
    /// Memory mapping failed.
    MemoryMap(vk::Result),
    /// Fence wait failed.
    FenceWait(vk::Result),
    /// Image layout transition failed.
    ImageLayoutTransition(vk::Result),
    /// Terminal write failed.
    TerminalWrite(codevar_consoleutil::ConsoleError),
    /// Terminal size detection failed.
    TerminalSizeUnavailable,
    /// The render target format is not supported for readback.
    UnsupportedFormat,
    /// Internal invariant violated.
    Internal(&'static str),
}

impl fmt::Display for PixportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandAllocation(err) => write!(f, "command buffer allocation failed: {err:?}"),
            Self::CommandRecord(err) => write!(f, "command buffer recording failed: {err:?}"),
            Self::Submit(err) => write!(f, "command buffer submission failed: {err:?}"),
            Self::BufferCreate(err) => write!(f, "staging buffer creation failed: {err:?}"),
            Self::MemoryAllocate(err) => write!(f, "staging buffer memory allocation failed: {err:?}"),
            Self::NoSuitableMemoryType => write!(f, "no suitable memory type for staging buffer"),
            Self::MemoryMap(err) => write!(f, "staging buffer memory map failed: {err:?}"),
            Self::FenceWait(err) => write!(f, "fence wait failed: {err:?}"),
            Self::ImageLayoutTransition(err) => write!(f, "image layout transition failed: {err:?}"),
            Self::TerminalWrite(err) => write!(f, "terminal write failed: {err}"),
            Self::TerminalSizeUnavailable => write!(f, "could not detect terminal size"),
            Self::UnsupportedFormat => write!(f, "render target format not supported for readback"),
            Self::Internal(msg) => write!(f, "internal pixport error: {msg}"),
        }
    }
}

impl core::error::Error for PixportError {}

impl From<codevar_consoleutil::ConsoleError> for PixportError {
    fn from(err: codevar_consoleutil::ConsoleError) -> Self {
        Self::TerminalWrite(err)
    }
}

impl From<crate::ui_renderer::RendererError> for PixportError {
    fn from(_err: crate::ui_renderer::RendererError) -> Self {
        Self::Internal("renderer error")
    }
}

/// Result type for pixport operations.
pub type PixportResult<T> = Result<T, PixportError>;

/// Configuration for the pixport readback.
#[derive(Debug, Clone, Copy)]
pub struct PixportConfig {
    /// Maximum number of terminal columns to render (clamped to terminal width).
    pub max_cols: u16,
    /// Maximum number of terminal rows to render (clamped to terminal height).
    pub max_rows: u16,
    /// Whether to use half-block characters (▀) for 2x vertical density.
    /// When true, each terminal cell represents 2 image rows.
    pub use_half_blocks: bool,
    /// Whether to clear the terminal before each frame.
    pub clear_before_frame: bool,
    /// Resampling filter used when mapping framebuffer pixels to terminal cells.
    pub filter: PixportFilterType,
}

impl Default for PixportConfig {
    fn default() -> Self {
        Self {
            max_cols: 0, // use terminal width
            max_rows: 0, // use terminal height
            use_half_blocks: true,
            clear_before_frame: true,
            filter: PixportFilterType::default(),
        }
    }
}

/// Pixel format for the render target (must match the pipeline's color format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixportFormat {
    /// B8G8R8A8 (4 bytes per pixel, Vulkan's B8G8R8A8_UNORM)
    Bgra8888,
    /// R8G8B8A8 (4 bytes per pixel)
    Rgba8888,
}

impl PixportFormat {
    /// Bytes per pixel for this format.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::Bgra8888 | Self::Rgba8888 => 4,
        }
    }
}

/// Finds a memory type index matching the requirements.
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for (index, memory_type) in properties.memory_types[..properties.memory_type_count as usize]
        .iter()
        .enumerate()
    {
        if type_bits & (1 << index) == 0 {
            continue;
        }
        if memory_type.property_flags.contains(preferred) {
            return Some(index as u32);
        }
    }
    None
}

/// The pixport: reads Vulkan render target and writes ANSI to terminal.
pub struct Pixport<'p> {
    context: &'p PipelineContext,
    staging_buffer: Option<vk::Buffer>,
    staging_memory: Option<vk::DeviceMemory>,
    staging_mapped: *mut u8,
    staging_size: vk::DeviceSize,
    command_buffer: Option<vk::CommandBuffer>,
    fence: Option<vk::Fence>,
    config: PixportConfig,
    pixel_format: PixportFormat,
    last_terminal_cols: u16,
    last_terminal_rows: u16,
}

impl<'p> Pixport<'p> {
    /// Creates a new pixport for the given pipeline context.
    ///
    /// The pixport borrows the pipeline context and must not outlive it.
    pub fn new(context: &'p PipelineContext, config: PixportConfig) -> PixportResult<Self> {
        let render_target = context.render_target();
        if render_target.drm_format != DRM_FORMAT_XRGB8888 {
            return Err(PixportError::UnsupportedFormat);
        }
        let command_buffer = Self::allocate_command_buffer(context)?;
        let fence = Self::create_fence(context)?;
        let terminal_cols = detect_terminal_width();
        let terminal_rows = detect_terminal_height();

        Ok(Self {
            context,
            staging_buffer: None,
            staging_memory: None,
            staging_mapped: core::ptr::null_mut(),
            staging_size: 0,
            command_buffer: Some(command_buffer),
            fence: Some(fence),
            config,
            pixel_format: PixportFormat::Bgra8888,
            last_terminal_cols: terminal_cols,
            last_terminal_rows: terminal_rows,
        })
    }

    /// Allocates a command buffer from the pipeline's command pool.
    fn allocate_command_buffer(context: &PipelineContext) -> PixportResult<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let allocated = unsafe {
            context
                .device()
                .allocate_command_buffers(&allocate_info)
        }
        .map_err(PixportError::CommandAllocation)?;
        let command_buffer = allocated
            .first()
            .copied()
            .ok_or(PixportError::Internal("driver allocated no command buffer"))?;

        Ok(command_buffer)
    }

    /// Creates a fence for synchronization.
    fn create_fence(context: &PipelineContext) -> PixportResult<vk::Fence> {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::empty());
        unsafe { context.device().create_fence(&fence_info, None) }.map_err(PixportError::FenceWait)
    }

    /// Ensures the staging buffer exists and is large enough for the current render target.
    fn ensure_staging_buffer(&mut self) -> PixportResult<()> {
        let render_target = self.context.render_target();
        let width = render_target.width as u64;
        let height = render_target.height as u64;
        let bpp = self.pixel_format.bytes_per_pixel() as u64;
        let row_stride = width * bpp;
        let aligned_stride = (row_stride + 255) & !255;
        let required_size = aligned_stride * height;
        let needs_recreate = self.staging_size < required_size;
        if needs_recreate {
            self.destroy_staging();
            let buffer_info = vk::BufferCreateInfo::default()
                .size(required_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe {
                self.context
                    .device()
                    .create_buffer(&buffer_info, None)
            }
            .map_err(PixportError::BufferCreate)?;
            let requirements = unsafe {
                self.context
                    .device()
                    .get_buffer_memory_requirements(buffer)
            };
            let mem_properties = unsafe {
                self.context
                    .instance()
                    .get_physical_device_memory_properties(self.context.physical_device())
            };
            let memory_type = find_memory_type(
                &mem_properties,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or(PixportError::NoSuitableMemoryType)?;
            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory = unsafe {
                self.context
                    .device()
                    .allocate_memory(&allocate_info, None)
            }
            .map_err(PixportError::MemoryAllocate)?;
            unsafe {
                self.context
                    .device()
                    .bind_buffer_memory(buffer, memory, 0)
            }
            .map_err(|_| PixportError::Internal("failed to bind buffer memory"))?;
            let mapped_ptr = unsafe {
                self.context
                    .device()
                    .map_memory(memory, 0, required_size, vk::MemoryMapFlags::empty())
            }
            .map_err(PixportError::MemoryMap)? as *mut u8;
            self.staging_buffer = Some(buffer);
            self.staging_memory = Some(memory);
            self.staging_mapped = mapped_ptr;
            self.staging_size = required_size;
        }
        Ok(())
    }

    fn destroy_staging(&mut self) {
        if !self.staging_mapped.is_null() {
            unsafe {
                self.context
                    .device()
                    .unmap_memory(self.staging_memory.unwrap())
            };
            self.staging_mapped = core::ptr::null_mut();
        }
        if let Some(mem) = self.staging_memory.take() {
            unsafe { self.context.device().free_memory(mem, None) };
        }
        if let Some(buf) = self.staging_buffer.take() {
            unsafe { self.context.device().destroy_buffer(buf, None) };
        }
        self.staging_size = 0;
    }

    /// Transitions the render target image to TRANSFER_SRC_OPTIMAL layout.
    fn transition_image_to_transfer_src(&self, command_buffer: vk::CommandBuffer) -> PixportResult<()> {
        let image = self.context.image();
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
        unsafe {
            self.context
                .device()
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
        Ok(())
    }

    /// Transitions the render target image back to COLOR_ATTACHMENT_OPTIMAL.
    fn transition_image_to_color_attachment(&self, command_buffer: vk::CommandBuffer) -> PixportResult<()> {
        let image = self.context.image();
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
        unsafe {
            self.context
                .device()
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
        Ok(())
    }

    /// Records the copy from image to buffer.
    fn record_copy_image_to_buffer(&self, command_buffer: vk::CommandBuffer) -> PixportResult<()> {
        let render_target = self.context.render_target();
        let staging_buf = self
            .staging_buffer
            .as_ref()
            .ok_or(PixportError::Internal("staging buffer not initialized"))?;
        let width = render_target.width;
        let height = render_target.height;
        let bpp = self.pixel_format.bytes_per_pixel();
        let row_stride = width * bpp;
        let _aligned_stride = (row_stride + 255) & !255;

        let copy_region = vk::BufferImageCopy2::default()
            .buffer_offset(0)
            .buffer_row_length(width)
            .buffer_image_height(height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        let copy_info = vk::CopyImageToBufferInfo2::default()
            .src_image(self.context.image())
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_buffer(*staging_buf)
            .regions(core::slice::from_ref(&copy_region));

        unsafe {
            self.context
                .device()
                .cmd_copy_image_to_buffer2(command_buffer, &copy_info)
        };
        Ok(())
    }

    /// Reads the current frame from the GPU and converts to ANSI.
    pub fn capture_frame(&mut self) -> PixportResult<String> {
        self.ensure_staging_buffer()?;
        let command_buffer = self
            .command_buffer
            .ok_or(PixportError::Internal("command buffer not allocated"))?;
        let fence = self
            .fence
            .ok_or(PixportError::Internal("fence not created"))?;
        unsafe {
            self.context
                .device()
                .reset_fences(core::slice::from_ref(&fence))
        }
        .map_err(PixportError::FenceWait)?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .device()
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .map_err(PixportError::CommandRecord)?;
        self.transition_image_to_transfer_src(command_buffer)?;
        self.record_copy_image_to_buffer(command_buffer)?;
        self.transition_image_to_color_attachment(command_buffer)?;
        unsafe {
            self.context
                .device()
                .end_command_buffer(command_buffer)
        }
        .map_err(PixportError::CommandRecord)?;
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            self.context.device().queue_submit(
                self.context.queue(),
                core::slice::from_ref(&submit_info),
                fence,
            )
        }
        .map_err(PixportError::Submit)?;
        unsafe {
            self.context
                .device()
                .wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX)
        }
        .map_err(PixportError::FenceWait)?;
        let bytes = unsafe { core::slice::from_raw_parts(self.staging_mapped, self.staging_size as usize) };
        let ansi = self.pixels_to_ansi(bytes)?;
        Ok(ansi)
    }

    /// Converts raw pixel data to ANSI escape sequences.
    fn pixels_to_ansi(&self, pixels: &[u8]) -> PixportResult<String> {
        let render_target = self.context.render_target();
        let width = render_target.width as usize;
        let height = render_target.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel() as usize;
        let stride = width * bpp;

        let term_cols = if self.config.max_cols > 0 {
            self.config.max_cols.min(self.last_terminal_cols)
        } else {
            self.last_terminal_cols
        } as usize;
        let term_rows = if self.config.max_rows > 0 {
            self.config.max_rows.min(self.last_terminal_rows)
        } else {
            self.last_terminal_rows
        } as usize;

        if term_cols == 0 || term_rows == 0 {
            return Ok(if self.config.clear_before_frame {
                String::from("\x1b[2J\x1b[H")
            } else {
                String::new()
            });
        }
        let converter = AnsiColorConverter::with_filter(self.pixel_format, self.config.filter);
        let cells = converter.pixels_to_ansi_cells(
            pixels,
            width,
            height,
            stride,
            term_cols,
            term_rows,
            self.config.use_half_blocks,
        );
        Ok(ansi_cells_to_string(
            &cells,
            term_cols,
            term_rows,
            self.config.clear_before_frame,
        ))
    }

    /// Writes the captured frame to the terminal.
    pub fn present_frame(&mut self, ansi: &str) -> PixportResult<()> {
        write_stdout(ansi.as_bytes())?;
        Ok(())
    }

    /// Captures and presents a single frame.
    pub fn capture_and_present(&mut self) -> PixportResult<()> {
        let ansi = self.capture_frame()?;
        self.present_frame(&ansi)?;
        Ok(())
    }

    /// Updates the terminal size from the OS.
    pub fn update_terminal_size(&mut self) {
        self.last_terminal_cols = detect_terminal_width();
        self.last_terminal_rows = detect_terminal_height();
    }

    /// Returns the current terminal size.
    #[inline]
    #[must_use]
    pub fn terminal_size(&self) -> (u16, u16) {
        (self.last_terminal_cols, self.last_terminal_rows)
    }

    /// Returns the render target size.
    #[inline]
    #[must_use]
    pub fn render_target_size(&self) -> (u32, u32) {
        let rt = self.context.render_target();
        (rt.width, rt.height)
    }

    /// Sets the pixport configuration.
    pub fn set_config(&mut self, config: PixportConfig) {
        self.config = config;
    }

    /// Returns the current configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &PixportConfig {
        &self.config
    }
}

impl Drop for Pixport<'_> {
    fn drop(&mut self) {
        if !self.staging_mapped.is_null() {
            unsafe {
                self.context
                    .device()
                    .unmap_memory(self.staging_memory.unwrap())
            };
            self.staging_mapped = core::ptr::null_mut();
        }
        if let Some(mem) = self.staging_memory.take() {
            unsafe { self.context.device().free_memory(mem, None) };
        }
        if let Some(buf) = self.staging_buffer.take() {
            unsafe { self.context.device().destroy_buffer(buf, None) };
        }
        if let Some(fence) = self.fence.take() {
            unsafe { self.context.device().destroy_fence(fence, None) };
        }
        if let Some(cmd_buf) = self.command_buffer.take() {
            unsafe {
                self.context
                    .device()
                    .free_command_buffers(self.context.command_pool(), core::slice::from_ref(&cmd_buf))
            };
        }
    }
}

/// Pixel format conversion and resampling for terminal output.
///
/// This module converts raw framebuffer pixels into ANSI true-color cells.
/// It provides a single converter, [`AnsiColorConverter`], that supports
/// multiple resampling filters ([`PixportFilterType`]) using a robust scalar
/// implementation. The design is SIMD-ready: a hardware-accelerated path
/// can be selected via [`AnsiColorConverter::is_simd_available`] when
/// available.
///
/// # Resampling
///
/// Upscaling and downscaling use a separable 1-D convolution (a horizontal
/// pass followed by a vertical pass). Kernel widths adapt to the scale
/// factor (see [`PixportFilterType`]) so heavy downscaling stays anti-aliased.
pub mod simd {
    use alloc::string::String;
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use super::PixportFormat;

    /// RGB pixel with 8-bit channels.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(C)]
    pub struct RgbPixel {
        pub r: u8,
        pub g: u8,
        pub b: u8,
    }

    /// Pair of pixels used for half-block rendering (upper / lower).
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub struct RgbPixelPair {
        pub upper: RgbPixel,
        pub lower: RgbPixel,
    }

    /// ANSI cell: foreground color, optional background and a character.
    #[derive(Debug, Clone)]
    pub struct AnsiCell {
        pub fg: RgbPixel,
        pub bg: Option<RgbPixel>,
        pub char: char,
    }

    /// Resampling filter used when mapping framebuffer pixels to terminal cells.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[non_exhaustive]
    pub enum PixportFilterType {
        /// Point sampling. Exact, fastest, no antialiasing.
        Nearest,
        /// Linear interpolation over a 2x2 neighborhood.
        #[default]
        Bilinear,
        /// Cubic interpolation using the Catmull-Rom kernel.
        Bicubic,
        /// Cubic interpolation using the Mitchell-Netravali kernel.
        Mitchell,
        /// High-quality Lanczos-3 windowed sinc interpolation.
        Lanczos3,
    }

    /// Precomputed 1-D filter weights for a single output coordinate.
    struct WeightWindow {
        start: usize,
        len: usize,
        weights: Vec<f32>,
    }

    impl WeightWindow {
        fn tap(&self, i: usize) -> (usize, f32) {
            (self.start + i, self.weights[i])
        }
    }

    fn filter_weight(filter: PixportFilterType, x: f32) -> f32 {
        match filter {
            PixportFilterType::Nearest => 1.0,
            PixportFilterType::Bilinear => {
                let a = x.abs();
                if a < 1.0 { 1.0 - a } else { 0.0 }
            }
            PixportFilterType::Bicubic => {
                let a = x.abs();
                const A: f32 = -0.5;
                if a < 1.0 {
                    ((A + 2.0) * a - (A + 3.0)) * a * a + 1.0
                } else if a < 2.0 {
                    (((a - 5.0) * a + 8.0) * a - 4.0) * A
                } else {
                    0.0
                }
            }
            PixportFilterType::Mitchell => {
                let a = x.abs();
                if a < 1.0 {
                    (7.0 * a / 6.0 - 2.0) * a * a + 16.0 / 18.0
                } else if a < 2.0 {
                    ((2.0 - 7.0 * a / 18.0) * a - 10.0 / 3.0) * a + 16.0 / 9.0
                } else {
                    0.0
                }
            }
            PixportFilterType::Lanczos3 => {
                let ax = x.abs();
                if ax < 3.0 { sinc(ax) * sinc(ax / 3.0) } else { 0.0 }
            }
        }
    }

    fn sinc(x: f32) -> f32 {
        if x.abs() < 1e-6 {
            1.0
        } else {
            (x * core::f32::consts::PI).sin() / x
        }
    }

    fn filter_support(filter: PixportFilterType) -> f32 {
        match filter {
            PixportFilterType::Nearest => 0.0,
            PixportFilterType::Bilinear => 1.0,
            PixportFilterType::Bicubic => 2.0,
            PixportFilterType::Mitchell => 2.0,
            PixportFilterType::Lanczos3 => 3.0,
        }
    }

    fn build_window(filter: PixportFilterType, center: f32, size: usize, filter_scale: f32) -> WeightWindow {
        if size == 0 {
            return WeightWindow {
                start: 0,
                len: 0,
                weights: Vec::new(),
            };
        }
        if filter == PixportFilterType::Nearest {
            let idx = center.round() as usize;
            let idx = idx.clamp(0, size - 1);
            return WeightWindow {
                start: idx,
                len: 1,
                weights: vec![1.0],
            };
        }
        let radius = filter_support(filter) * filter_scale;
        let mut x_min = (center - radius).max(0.0) as usize;
        let mut x_max_exclusive = ((center + radius).ceil().min(size as f32)) as usize;
        // Ensure at least one tap (edge/clamp case).
        if x_max_exclusive <= x_min {
            x_max_exclusive = x_min + 1;
        }
        x_min = x_min.min(x_max_exclusive.saturating_sub(1));
        let mut weights = Vec::with_capacity(x_max_exclusive - x_min);
        let mut sum = 0.0f32;
        for x in x_min..x_max_exclusive {
            let w = filter_weight(filter, (x as f32 - center) / filter_scale);
            weights.push(w);
            sum += w;
        }
        if sum != 0.0 {
            let inv = 1.0 / sum;
            for w in &mut weights {
                *w *= inv;
            }
        }
        WeightWindow {
            start: x_min,
            len: weights.len(),
            weights,
        }
    }

    #[inline]
    fn pixel_rgb(row: &[u8], x: usize, format: PixportFormat) -> (f32, f32, f32) {
        let bpp = format.bytes_per_pixel() as usize;
        let o = x * bpp;
        match format {
            PixportFormat::Bgra8888 => (row[o + 2] as f32, row[o + 1] as f32, row[o] as f32),
            PixportFormat::Rgba8888 => (row[o] as f32, row[o + 1] as f32, row[o + 2] as f32),
        }
    }

    #[inline]
    fn clamp_u8(v: f32) -> u8 {
        v.clamp(0.0, 255.0).round() as u8
    }

    fn resample_rgb_f32(
        pixels: &[u8],
        width: usize,
        height: usize,
        stride: usize,
        format: PixportFormat,
        out_w: usize,
        out_h: usize,
        filter: PixportFilterType,
        dst: &mut [f32],
    ) {
        let need = out_w * out_h * 3;
        if need == 0 || dst.len() < need {
            let n = dst.len().min(need);
            dst[..n].fill(0.0);
            return;
        }
        if width == 0 || height == 0 || out_w == 0 || out_h == 0 {
            dst[..need].fill(0.0);
            return;
        }
        let scale_x = width as f32 / out_w as f32;
        let scale_y = height as f32 / out_h as f32;
        let fs_x = scale_x.max(1.0);
        let fs_y = scale_y.max(1.0);
        let bpp = format.bytes_per_pixel() as usize;

        let x_windows: Vec<WeightWindow> = (0..out_w)
            .map(|x| {
                let center = (x as f32 + 0.5) * scale_x - 0.5;
                build_window(filter, center, width, fs_x)
            })
            .collect();

        let mut temp = vec![0.0f32; out_w * height * 3];
        for y in 0..height {
            let row = pixels.get(y * stride..y * stride + width * bpp);
            for x in 0..out_w {
                let win = &x_windows[x];
                let base = (y * out_w + x) * 3;
                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;
                for i in 0..win.len {
                    let (px, w) = win.tap(i);
                    if let Some(pr) = row {
                        let (pr_, pg_, pb_) = pixel_rgb(pr, px, format);
                        r += pr_ * w;
                        g += pg_ * w;
                        b += pb_ * w;
                    }
                }
                temp[base] = r;
                temp[base + 1] = g;
                temp[base + 2] = b;
            }
        }

        let y_windows: Vec<WeightWindow> = (0..out_h)
            .map(|y| {
                let center = (y as f32 + 0.5) * scale_y - 0.5;
                build_window(filter, center, height, fs_y)
            })
            .collect();

        for oy in 0..out_h {
            let ywin = &y_windows[oy];
            for ox in 0..out_w {
                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;
                for j in 0..ywin.len {
                    let (py, wy) = ywin.tap(j);
                    let src = (py * out_w + ox) * 3;
                    r += temp[src] * wy;
                    g += temp[src + 1] * wy;
                    b += temp[src + 2] * wy;
                }
                let d = (oy * out_w + ox) * 3;
                dst[d] = r;
                dst[d + 1] = g;
                dst[d + 2] = b;
            }
        }
    }

    /// Converts framebuffer pixels into ANSI terminal cells using the configured filter.
    ///
    /// `stride` is the byte distance between consecutive rows and must be at
    /// least `width * pixel_format.bytes_per_pixel()`.
    pub struct AnsiColorConverter {
        pixel_format: PixportFormat,
        filter: PixportFilterType,
    }

    impl AnsiColorConverter {
        /// Creates a converter with the default filter ([`PixportFilterType::Bilinear`]).
        pub fn new(pixel_format: PixportFormat) -> Self {
            Self {
                pixel_format,
                filter: PixportFilterType::default(),
            }
        }

        /// Creates a converter with an explicit filter.
        pub fn with_filter(pixel_format: PixportFormat, filter: PixportFilterType) -> Self {
            Self { pixel_format, filter }
        }

        /// Returns the pixel format this converter expects.
        pub const fn pixel_format(&self) -> PixportFormat {
            self.pixel_format
        }

        /// Returns the currently configured filter.
        pub const fn filter(&self) -> PixportFilterType {
            self.filter
        }

        /// Changes the resampling filter.
        pub fn set_filter(&mut self, filter: PixportFilterType) {
            self.filter = filter;
        }

        /// Returns `false`: the current implementation runs the scalar fallback.
        /// A SIMD-accelerated path can be selected here when available.
        pub const fn is_simd_available(&self) -> bool {
            false
        }

        /// Samples the framebuffer at fractional coordinates `(x, y)`.
        ///
        /// Integer values correspond to pixel centers. Uses the configured
        /// filter with unit filter scale.
        pub fn sample_pixel(
            &self,
            pixels: &[u8],
            width: usize,
            height: usize,
            stride: usize,
            x: f32,
            y: f32,
        ) -> RgbPixel {
            if width == 0 || height == 0 {
                return RgbPixel { r: 0, g: 0, b: 0 };
            }
            let xwin = build_window(self.filter, x, width, 1.0);
            let ywin = build_window(self.filter, y, height, 1.0);
            let bpp = self.pixel_format.bytes_per_pixel() as usize;
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            for j in 0..ywin.len {
                let (py, wy) = ywin.tap(j);
                let row = match pixels.get(py * stride..py * stride + width * bpp) {
                    Some(r) => r,
                    None => continue,
                };
                for i in 0..xwin.len {
                    let (px, wx) = xwin.tap(i);
                    let (pr_, pg_, pb_) = pixel_rgb(row, px, self.pixel_format);
                    r += pr_ * wx * wy;
                    g += pg_ * wx * wy;
                    b += pb_ * wx * wy;
                }
            }
            RgbPixel {
                r: clamp_u8(r),
                g: clamp_u8(g),
                b: clamp_u8(b),
            }
        }

        /// Resamples the framebuffer into `out_w x out_h` RGB pixels.
        pub fn resample(
            &self,
            pixels: &[u8],
            width: usize,
            height: usize,
            stride: usize,
            out_w: usize,
            out_h: usize,
        ) -> Vec<RgbPixel> {
            let n = out_w * out_h;
            let mut temp = vec![0.0f32; n * 3];
            resample_rgb_f32(
                pixels,
                width,
                height,
                stride,
                self.pixel_format,
                out_w,
                out_h,
                self.filter,
                &mut temp,
            );
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(RgbPixel {
                    r: clamp_u8(temp[i * 3]),
                    g: clamp_u8(temp[i * 3 + 1]),
                    b: clamp_u8(temp[i * 3 + 2]),
                });
            }
            out
        }

        /// Converts a framebuffer slice into ANSI terminal cells.
        ///
        /// `term_cols` and `term_rows` give the terminal grid size. When
        /// `use_half_blocks` is true, each terminal row represents two
        /// framebuffer rows (upper / lower) for 2x vertical density.
        pub fn pixels_to_ansi_cells(
            &self,
            pixels: &[u8],
            width: usize,
            height: usize,
            stride: usize,
            term_cols: usize,
            term_rows: usize,
            use_half_blocks: bool,
        ) -> Vec<AnsiCell> {
            if width == 0 || height == 0 || term_cols == 0 || term_rows == 0 {
                return Vec::new();
            }
            let out_w = term_cols;
            let out_h = if use_half_blocks {
                term_rows.saturating_mul(2)
            } else {
                term_rows
            };
            let rgb = self.resample(pixels, width, height, stride, out_w, out_h);
            let mut cells = Vec::with_capacity(term_cols * term_rows);
            let ch = if use_half_blocks { '▀' } else { '█' };
            for ty in 0..term_rows {
                for tx in 0..term_cols {
                    let upper = rgb[(ty * 2) * out_w + tx];
                    if use_half_blocks {
                        let lower = rgb[(ty * 2 + 1) * out_w + tx];
                        cells.push(AnsiCell {
                            fg: upper,
                            bg: Some(lower),
                            char: ch,
                        });
                    } else {
                        cells.push(AnsiCell {
                            fg: upper,
                            bg: None,
                            char: ch,
                        });
                    }
                }
            }
            cells
        }
    }

    /// Joins ANSI cells into one escape-sequence string.
    ///
    /// `_term_rows` is retained for API compatibility.
    pub fn ansi_cells_to_string(
        cells: &[AnsiCell],
        term_cols: usize,
        _term_rows: usize,
        clear_before_frame: bool,
    ) -> String {
        if term_cols == 0 {
            return String::new();
        }
        let mut ansi = String::with_capacity(cells.len() * 20);
        let mut current_fg: Option<RgbPixel> = None;
        let mut current_bg: Option<RgbPixel> = None;

        if clear_before_frame {
            ansi.push_str("\x1b[2J\x1b[H");
        }

        for (idx, cell) in cells.iter().enumerate() {
            let term_x = idx % term_cols;
            if term_x == 0 {
                ansi.push_str(&format!("\x1b[{};1H", idx / term_cols + 1));
            }
            if current_fg != Some(cell.fg) {
                ansi.push_str(&format!("\x1b[38;2;{};{};{}m", cell.fg.r, cell.fg.g, cell.fg.b));
                current_fg = Some(cell.fg);
            }
            if let Some(bg) = cell.bg {
                if current_bg != Some(bg) {
                    ansi.push_str(&format!("\x1b[48;2;{};{};{}m", bg.r, bg.g, bg.b));
                    current_bg = Some(bg);
                }
            } else if current_bg.is_some() {
                ansi.push_str("\x1b[49m");
                current_bg = None;
            }
            ansi.push(cell.char);
        }
        ansi.push_str("\x1b[0m");
        ansi
    }
}

/// High-level render loop for terminal-based rendering.
pub mod render_loop {
    use super::{Pixport, PixportConfig, PixportResult};
    use crate::ui_pipeline::PipelineContext;
    use crate::ui_renderer::{RenderLayer, RendererSubsystem};
    use alloc::boxed::Box;
    use codevar_consoleutil::console_ansi::{cursor, erase};

    /// Configuration for the terminal render loop.
    #[derive(Debug, Clone)]
    pub struct RenderLoopConfig {
        /// Frame rate of Render loop
        pub frame_rate: codevar_time_core::TimeDuration,
        /// Pixport configuration.
        pub pixport_config: PixportConfig,
    }

    impl Default for RenderLoopConfig {
        fn default() -> Self {
            Self {
                frame_rate: codevar_time_core::TimeDuration::from_millis(16),
                pixport_config: PixportConfig::default(),
            }
        }
    }

    /// Runs a render loop that captures frames and outputs to terminal.
    pub fn run_terminal_render_loop<'p, L>(
        pipeline: &'p PipelineContext,
        mut renderer: RendererSubsystem<'p>,
        layer: L,
        config: RenderLoopConfig,
    ) -> PixportResult<()>
    where
        L: RenderLayer + 'p,
    {
        codevar_consoleutil::init_ansi_support();
        let term_cols = codevar_consoleutil::detect_terminal_width();
        let term_rows = codevar_consoleutil::detect_terminal_height();
        let mut pixport_config = config.pixport_config;
        if pixport_config.max_cols == 0 {
            pixport_config.max_cols = term_cols;
        }
        if pixport_config.max_rows == 0 {
            pixport_config.max_rows = term_rows;
        }
        renderer.layers_mut().add(Box::new(layer))?;
        let mut pixport = Pixport::new(pipeline, pixport_config)?;
        codevar_consoleutil::write_stdout(cursor::hide().as_bytes())?;
        codevar_consoleutil::write_stdout(erase::screen().as_bytes())?;

        let frame_duration = config.frame_rate;
        loop {
            let frame_start = codevar_time_core::SystemTime::monotonic_nanos();
            renderer.begin_frame(None)?;
            renderer.render_frame()?;
            let sync_file = renderer.end_frame()?;
            wait_for_gpu_sync(&sync_file)?;
            drop(sync_file);
            pixport.capture_and_present()?;
            let elapsed = codevar_time_core::SystemTime::monotonic_nanos() - frame_start;
            let elapsed_dur = codevar_time_core::TimeDuration::from_nanos(elapsed);
            let remaining = frame_duration.saturating_sub(elapsed_dur);
            sleep_for(remaining);
        }
    }

    /// Sleeps for `duration` using `nanosleep(2)`
    fn sleep_for(duration: codevar_time_core::TimeDuration) {
        let duration = if duration.as_nanos() > i64::MAX as u128 {
            codevar_time_core::TimeDuration::from_secs(i64::MAX as u64)
        } else {
            duration
        };
        let req = libc::timespec {
            tv_sec: duration.as_secs() as libc::time_t,
            tv_nsec: duration.subsec_nanos() as libc::c_long,
        };
        let _ = unsafe { libc::nanosleep(&req, core::ptr::null_mut()) };
    }

    /// Wait for GPU synchronization using a sync file.
    fn wait_for_gpu_sync(sync_file: &crate::ui_pipeline::OwnedFd) -> PixportResult<()> {
        let mut descriptor = libc::pollfd {
            fd: sync_file.as_raw(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 5000) };
        if ready <= 0 {
            return Err(super::PixportError::Internal("timed out waiting for GPU"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui_terminal_pixport::simd::{AnsiCell, RgbPixel};

    #[test]
    fn test_pixel_format_bpp() {
        assert_eq!(PixportFormat::Bgra8888.bytes_per_pixel(), 4);
        assert_eq!(PixportFormat::Rgba8888.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_pixport_config_default() {
        let config = PixportConfig::default();
        assert_eq!(config.max_cols, 0);
        assert_eq!(config.max_rows, 0);
        assert!(config.use_half_blocks);
        assert!(config.clear_before_frame);
        assert_eq!(config.filter, PixportFilterType::default());
    }

    #[test]
    fn test_nearest_color_converter() {
        let converter =
            simd::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Nearest);
        // 2x1 image: red, green
        let mut pixels = vec![0u8; 2 * 1 * 4];
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]); // red (BGRA)
        pixels[4..8].copy_from_slice(&[0, 255, 0, 255]); // green (BGRA)

        let cells = converter.pixels_to_ansi_cells(&pixels, 2, 1, 8, 2, 1, false);

        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].fg, simd::RgbPixel { r: 255, g: 0, b: 0 });
        assert_eq!(cells[1].fg, simd::RgbPixel { r: 0, g: 255, b: 0 });
        assert_eq!(cells[0].char, '█');
    }

    #[test]
    fn test_nearest_half_blocks() {
        let converter =
            simd::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Nearest);
        // 2x2 image
        let mut pixels = vec![0u8; 2 * 2 * 4];
        // Row 0: Red, Green
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]);
        pixels[4..8].copy_from_slice(&[0, 255, 0, 255]);
        // Row 1: Blue, Yellow
        pixels[8..12].copy_from_slice(&[255, 0, 0, 255]);
        pixels[12..16].copy_from_slice(&[0, 255, 255, 255]);

        let cells = converter.pixels_to_ansi_cells(&pixels, 2, 2, 8, 2, 1, true);

        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].fg, simd::RgbPixel { r: 255, g: 0, b: 0 }); // upper red
        assert_eq!(cells[0].bg, Some(simd::RgbPixel { r: 0, g: 0, b: 255 })); // lower blue
        assert_eq!(cells[1].fg, simd::RgbPixel { r: 0, g: 255, b: 0 }); // upper green
        assert_eq!(cells[1].bg, Some(simd::RgbPixel { r: 255, g: 255, b: 0 })); // lower yellow
        assert_eq!(cells[0].char, '▀');
    }

    #[test]
    fn test_converter_defaults() {
        let c = simd::AnsiColorConverter::new(PixportFormat::Bgra8888);
        assert!(!c.is_simd_available());
        assert_eq!(c.filter(), PixportFilterType::Bilinear);
        let c2 = simd::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Lanczos3);
        assert_eq!(c2.filter(), PixportFilterType::Lanczos3);
    }

    #[test]
    fn test_bilinear_upsample_midpoint() {
        let mut pixels = vec![0u8; 2 * 1 * 4];
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]); // red
        pixels[4..8].copy_from_slice(&[255, 0, 0, 255]); // blue (BGRA: B=255)
        let c = simd::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Bilinear);
        let mid = c.sample_pixel(&pixels, 2, 1, 8, 0.5, 0.0);
        assert_eq!(mid, simd::RgbPixel { r: 128, g: 0, b: 128 });
    }

    #[test]
    fn test_upsample_gradient() {
        let mut pixels = vec![0u8; 2 * 1 * 4];
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]); // red
        pixels[4..8].copy_from_slice(&[255, 0, 0, 255]); // blue
        let c = simd::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Bilinear);
        let rgb = c.resample(&pixels, 2, 1, 8, 3, 1);
        assert_eq!(rgb[0], simd::RgbPixel { r: 255, g: 0, b: 0 }); // left edge -> red
        assert!(rgb[2].b > rgb[2].r); // right side leans blue
        assert!(rgb[1].r > 0 && rgb[1].b > 0); // middle is a blend
    }

    #[test]
    fn test_resize_constant_image() {
        let v = 100u8;
        let pixels = vec![v; 2 * 2 * 4];
        for filter in [
            PixportFilterType::Nearest,
            PixportFilterType::Bilinear,
            PixportFilterType::Bicubic,
            PixportFilterType::Mitchell,
            PixportFilterType::Lanczos3,
        ] {
            let c = simd::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, filter);
            let rgb = c.resample(&pixels, 2, 2, 8, 8, 8);
            assert!(
                rgb.iter()
                    .all(|p| p.r == v && p.g == v && p.b == v),
                "filter {filter:?} failed"
            );
        }
    }

    #[test]
    fn test_ansi_cells_to_string() {
        let cells = vec![
            AnsiCell {
                fg: RgbPixel { r: 255, g: 0, b: 0 },
                bg: Some(RgbPixel { r: 0, g: 0, b: 255 }),
                char: '▀',
            },
            AnsiCell {
                fg: RgbPixel { r: 0, g: 255, b: 0 },
                bg: Some(RgbPixel { r: 255, g: 255, b: 0 }),
                char: '▀',
            },
        ];

        let ansi = ansi_cells_to_string(&cells, 2, 1, true);

        assert!(ansi.contains("\x1b[2J\x1b[H"));
        assert!(ansi.contains("\x1b[38;2;255;0;0m"));
        assert!(ansi.contains("\x1b[48;2;0;0;255m"));
        assert!(ansi.contains("\x1b[38;2;0;255;0m"));
        assert!(ansi.contains("\x1b[48;2;255;255;0m"));
        assert!(ansi.contains("▀"));
        assert!(ansi.ends_with("\x1b[0m"));
    }
}
