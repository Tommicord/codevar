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
//! The resampling filter may be set to [`PixportFilterType::Auto`], which
//! picks a concrete filter from a terminal-size ladder built with the
//! [`pixport_auto_filter_table!`] macro.

use alloc::string::String;
use core::fmt;
use core::task::{Context, Poll, Waker};

use self::ansi::{AnsiColorConverter, PixportFilterType, ansi_cells_to_string};
use self::frame_pipe::{FramePipe, QueuedFrame};
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
    /// Driver allocated no command buffer.
    NoCommandBuffer,
    /// Failed to bind buffer memory.
    BindBufferMemory(vk::Result),
    /// Staging buffer not initialized.
    StagingNotReady,
    /// Command buffer not allocated.
    CommandBufferNotAllocated,
    /// Fence not created.
    FenceNotCreated,
    /// Render target dimensions exceed maximum.
    RenderTargetTooLarge,
    /// Frame dimensions overflow during validation.
    FrameDimensionsOverflow,
    /// Frame buffer size mismatch.
    FrameSizeMismatch,
    /// GPU sync poll failed (libc::poll returned error).
    GpuSyncPollFailed,
    /// GPU sync timed out.
    GpuSyncTimeout,
    /// Renderer subsystem error.
    Renderer(crate::ui_renderer::RendererError),
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
            Self::NoCommandBuffer => write!(f, "driver allocated no command buffer"),
            Self::BindBufferMemory(err) => write!(f, "failed to bind buffer memory: {err:?}"),
            Self::StagingNotReady => write!(f, "staging buffer not initialized"),
            Self::CommandBufferNotAllocated => write!(f, "command buffer not allocated"),
            Self::FenceNotCreated => write!(f, "fence not created"),
            Self::RenderTargetTooLarge => write!(f, "render target too large"),
            Self::FrameDimensionsOverflow => write!(f, "frame dimensions overflow"),
            Self::FrameSizeMismatch => write!(f, "frame buffer size mismatch"),
            Self::GpuSyncPollFailed => write!(f, "GPU sync poll failed"),
            Self::GpuSyncTimeout => write!(f, "GPU sync timed out"),
            Self::Renderer(err) => write!(f, "renderer error: {err}"),
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
    fn from(err: crate::ui_renderer::RendererError) -> Self {
        Self::Renderer(err)
    }
}

/// Result type for pixport operations.
pub type PixportResult<T> = Result<T, PixportError>;

/// Vertical density of the terminal glyph grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellDensity {
    /// Each terminal cell draws one framebuffer row with '█'.
    FullBlocks,
    /// Each terminal cell draws two framebuffer rows with '▀'.
    #[default]
    HalfBlocks,
}

impl CellDensity {
    /// Returns `true` if half-block mode is active.
    #[inline]
    #[must_use]
    pub const fn is_half_blocks(self) -> bool {
        matches!(self, Self::HalfBlocks)
    }
}

/// Screen clear policy for frame presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClearPolicy {
    /// Clear the entire screen before each frame.
    #[default]
    ClearBeforeFrame,
    /// Preserve existing screen contents.
    Preserve,
}

impl ClearPolicy {
    /// Returns `true` if the screen should be cleared.
    #[inline]
    #[must_use]
    pub const fn should_clear(self) -> bool {
        matches!(self, Self::ClearBeforeFrame)
    }
}

/// Configuration for the pixport readback.
#[derive(Debug, Clone, Copy, Default)]
pub struct PixportConfig {
    /// Maximum number of terminal columns to render (clamped to terminal width).
    pub max_cols: u16,
    /// Maximum number of terminal rows to render (clamped to terminal height).
    pub max_rows: u16,
    /// Vertical density of the terminal glyph grid.
    pub cell_density: CellDensity,
    /// Screen clear policy for frame presentation.
    pub clear_policy: ClearPolicy,
    /// Resampling filter used when mapping framebuffer pixels to terminal cells.
    ///
    /// [`PixportFilterType::Auto`] picks a concrete filter from the terminal
    /// size ladder, re-evaluating it after every resize.
    pub filter: PixportFilterType,
}

/// Escape sequence that clears the whole screen and homes the cursor.
const CLEAR_SEQUENCE: &str = "\x1b[2J\x1b[H";

/// Describes a terminal grid size change reported by
/// [`Pixport::poll_terminal_resize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalResize {
    /// Columns before the resize.
    pub previous_cols: u16,
    /// Rows before the resize.
    pub previous_rows: u16,
    /// Columns after the resize.
    pub cols: u16,
    /// Rows after the resize.
    pub rows: u16,
}

/// Compares a cached terminal grid against a freshly detected one.
///
/// Returns `Some` when either dimension changed. Split out of
/// [`Pixport::poll_terminal_resize`] so the decision is unit-testable
/// without a Vulkan context or a real terminal.
fn detect_resize(previous_cols: u16, previous_rows: u16, cols: u16, rows: u16) -> Option<TerminalResize> {
    if previous_cols == cols && previous_rows == rows {
        None
    } else {
        Some(TerminalResize {
            previous_cols,
            previous_rows,
            cols,
            rows,
        })
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
    /// Set when the terminal was resized since the last presented frame:
    /// the next frame must clear the screen even if `clear_before_frame`
    /// is disabled, otherwise stale cells survive outside the new grid.
    force_clear: bool,
    /// Waker of a suspended [`Pixport::capture_frame_async`] future, woken
    /// by [`Pixport::notify_capture`].
    capture_waker: Option<Waker>,
}

/// How long a single [`Pixport::poll_capture`] call blocks on the readback
/// fence before returning [`Poll::Pending`]. The bounded slice guarantees
/// progress under re-polling executors (see `present_loop::block_on`) while
/// keeping idle wake-ups cheap.
const CAPTURE_POLL_SLICE_NS: codevar_time_core::TimeDuration =
    codevar_time_core::TimeDuration::from_millis(1);

impl<'p> Pixport<'p> {
    /// Creates a new pixport for the given pipeline context.
    ///
    /// The pixport borrows the pipeline context and must not outlive it.
    pub fn new(context: &'p PipelineContext, config: PixportConfig) -> PixportResult<Self> {
        let present_target = context.present_target();
        if present_target.drm_format != DRM_FORMAT_XRGB8888 {
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
            force_clear: false,
            capture_waker: None,
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
            .ok_or(PixportError::NoCommandBuffer)?;

        Ok(command_buffer)
    }

    /// Creates a fence for synchronization.
    fn create_fence(context: &PipelineContext) -> PixportResult<vk::Fence> {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::empty());
        unsafe { context.device().create_fence(&fence_info, None) }.map_err(PixportError::FenceWait)
    }

    /// Ensures the staging buffer exists and is large enough for the current render target.
    fn ensure_staging_buffer(&mut self) -> PixportResult<()> {
        let present_target = self.context.present_target();
        let width = present_target.width as u64;
        let height = present_target.height as u64;
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
            .map_err(PixportError::BindBufferMemory)?;
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
        let present_target = self.context.present_target();
        let staging_buf = self
            .staging_buffer
            .as_ref()
            .ok_or(PixportError::StagingNotReady)?;
        let width = present_target.width;
        let height = present_target.height;
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

    /// Records and submits the GPU→staging readback without waiting for it.
    ///
    /// The copy is queued behind the frame that produced the render target,
    /// so it can progress while the caller waits on other GPU work.
    fn submit_readback(&mut self) -> PixportResult<()> {
        let command_buffer = self
            .command_buffer
            .ok_or(PixportError::CommandBufferNotAllocated)?;
        let fence = self.fence.ok_or(PixportError::FenceNotCreated)?;
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
        Ok(())
    }

    /// Blocks until the readback submitted by [`Self::submit_readback`] has
    /// finished and the staging buffer is CPU-visible.
    fn wait_readback(&self) -> PixportResult<()> {
        let fence = self.fence.ok_or(PixportError::FenceNotCreated)?;
        unsafe {
            self.context
                .device()
                .wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX)
        }
        .map_err(PixportError::FenceWait)?;
        Ok(())
    }

    /// Byte length of a tightly packed `width x height` readback.
    fn pixel_len(&self, width: usize, height: usize) -> PixportResult<usize> {
        width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(self.pixel_format.bytes_per_pixel() as usize))
            .ok_or(PixportError::RenderTargetTooLarge)
    }

    /// Copies the completed readback out of the staging buffer into an owned
    /// [`QueuedFrame`] so the staging buffer can be reused immediately.
    fn finish_capture(&mut self) -> PixportResult<QueuedFrame> {
        let present_target = self.context.present_target();
        let width = present_target.width as usize;
        let height = present_target.height as usize;
        let len = self.pixel_len(width, height)?;
        if self.staging_mapped.is_null() || (self.staging_size as usize) < len {
            return Err(PixportError::StagingNotReady);
        }
        // SAFETY: `staging_mapped` maps `staging_size` bytes of
        // `staging_memory` (checked above; both are kept alive by
        // `ensure_staging_buffer` and `Drop`), and the fence has signalled
        // so the GPU no longer writes to the buffer.
        let pixels = unsafe { core::slice::from_raw_parts(self.staging_mapped, len) }.to_vec();
        QueuedFrame::new(
            present_target.width,
            present_target.height,
            self.pixel_format,
            pixels,
        )
    }

    /// Reads the current frame from the GPU and converts it to ANSI,
    /// blocking until the readback completes.
    pub fn capture_frame(&mut self) -> PixportResult<String> {
        self.ensure_staging_buffer()?;
        self.submit_readback()?;
        self.wait_readback()?;
        self.staging_to_ansi()
    }

    /// Copies the staging buffer into an ANSI frame for the current terminal
    /// grid, consuming any forced repaint from a resize.
    fn staging_to_ansi(&mut self) -> PixportResult<String> {
        let present_target = self.context.present_target();
        let width = present_target.width as usize;
        let height = present_target.height as usize;
        let len = self.pixel_len(width, height)?;
        if self.staging_mapped.is_null() || (self.staging_size as usize) < len {
            return Err(PixportError::StagingNotReady);
        }
        // SAFETY: see `finish_capture`; the fence has signaled, so the copy
        // is complete and the mapping outlives this borrow.
        let pixels = unsafe { core::slice::from_raw_parts(self.staging_mapped, len) };
        self.pixels_to_ansi(pixels)
    }

    /// Submits the readback for the current render target without waiting.
    ///
    /// Pair with [`Self::finish_capture_async`] (or [`Self::poll_capture`])
    /// to collect the pixels later; [`Self::capture_frame_async`] does both.
    pub fn begin_capture(&mut self) -> PixportResult<()> {
        self.ensure_staging_buffer()?;
        self.submit_readback()
    }

    /// Polls an in-flight readback started by [`Self::begin_capture`].
    ///
    /// Blocks for at most [`CAPTURE_POLL_SLICE_NS`] before returning
    /// [`Poll::Pending`], so progress is guaranteed on any executor that
    /// re-polls. The waker is stored and can be triggered out of band with
    /// [`Self::notify_capture`].
    pub fn poll_capture(&mut self, cx: &mut Context<'_>) -> Poll<PixportResult<QueuedFrame>> {
        let fence = match self.fence {
            Some(fence) => fence,
            None => return Poll::Ready(Err(PixportError::FenceNotCreated)),
        };
        // SAFETY: `fence` was created by `create_fence` for this device and
        // stays alive while `Pixport` exists.
        let waited = unsafe {
            self.context.device().wait_for_fences(
                core::slice::from_ref(&fence),
                true,
                CAPTURE_POLL_SLICE_NS.as_millis() as u64,
            )
        };
        match waited {
            Ok(()) => {
                self.capture_waker = None;
                Poll::Ready(self.finish_capture())
            }
            Err(vk::Result::TIMEOUT) => {
                self.capture_waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Err(err) => Poll::Ready(Err(PixportError::FenceWait(err))),
        }
    }

    /// Async capture: submits the GPU readback and resolves with the
    /// framebuffer queued on the CPU side.
    ///
    /// # Errors
    ///
    /// Propagates the same failures as [`Self::capture_frame`].
    pub async fn capture_frame_async(&mut self) -> PixportResult<QueuedFrame> {
        self.begin_capture()?;
        self.finish_capture_async().await
    }

    /// Async half of [`Self::capture_frame_async`]: waits for a readback
    /// previously submitted with [`Self::begin_capture`] and copies it out.
    ///
    /// # Errors
    ///
    /// Returns [`PixportError::FenceWait`] when the driver rejects the fence
    /// wait, or [`PixportError::StagingNotReady`] when staging is not set up.
    pub async fn finish_capture_async(&mut self) -> PixportResult<QueuedFrame> {
        core::future::poll_fn(|cx| self.poll_capture(cx)).await
    }

    /// Wakes a suspended [`Self::capture_frame_async`] /
    /// [`Self::finish_capture_async`] future.
    ///
    /// Intended for drivers that observe GPU progress (for example through a
    /// `sync_file`); the re-polling `present_loop::block_on` driver does not
    /// need it. No-op when no capture is pending.
    pub fn notify_capture(&mut self) {
        if let Some(waker) = self.capture_waker.take() {
            waker.wake();
        }
    }

    /// Converts raw pixel data to ANSI escape sequences.
    ///
    /// The frame is prefixed with a full clear when `clear_before_frame` is
    /// set **or** when a resize forced a repaint (see
    /// [`Self::poll_terminal_resize`]); the flag is consumed here.
    fn pixels_to_ansi(&mut self, pixels: &[u8]) -> PixportResult<String> {
        let clear = self.config.clear_policy.should_clear() || self.force_clear;
        self.force_clear = false;
        let present_target = self.context.present_target();
        let width = present_target.width as usize;
        let height = present_target.height as usize;
        let stride = width * self.pixel_format.bytes_per_pixel() as usize;
        self.convert_pixels(pixels, width, height, stride, clear)
    }

    /// Converts a framebuffer slice into an ANSI frame for the current
    /// terminal grid, clamped by `max_cols` / `max_rows`.
    ///
    /// [`PixportFilterType::Auto`] resolves against the effective terminal
    /// grid, so a resize immediately changes resampling quality.
    fn convert_pixels(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        stride: usize,
        clear: bool,
    ) -> PixportResult<String> {
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
            return Ok(if clear {
                String::from(CLEAR_SEQUENCE)
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
            self.config.cell_density.is_half_blocks(),
        );
        Ok(ansi_cells_to_string(&cells, term_cols, term_rows, clear))
    }

    /// Writes the captured frame to the terminal.
    pub fn present_frame(&mut self, ansi: &str) -> PixportResult<()> {
        write_stdout(ansi.as_bytes())?;
        Ok(())
    }

    /// Presents one queued framebuffer as a *clear → render* pair.
    ///
    /// Conversion uses the **current** terminal grid and filter, so frames
    /// queued before a resize are drawn correctly afterwards.
    pub fn present_queued(&mut self, frame: &QueuedFrame) -> PixportResult<()> {
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let stride = width * frame.format().bytes_per_pixel() as usize;
        let content = self.convert_pixels(frame.pixels(), width, height, stride, false)?;
        let mut out = String::with_capacity(CLEAR_SEQUENCE.len() + content.len());
        out.push_str(CLEAR_SEQUENCE);
        out.push_str(&content);
        write_stdout(out.as_bytes())?;
        Ok(())
    }

    /// Drains `pipe` and presents every queued framebuffer in order
    /// (*clear → render → clear → render → …*).
    ///
    /// Returns how many frames were written.
    pub fn present_pending(&mut self, pipe: &mut FramePipe) -> PixportResult<usize> {
        let mut presented = 0usize;
        while let Some(frame) = pipe.try_recv() {
            self.present_queued(&frame)?;
            presented += 1;
        }
        Ok(presented)
    }

    /// Captures and presents a single frame.
    pub fn capture_and_present(&mut self) -> PixportResult<()> {
        let ansi = self.capture_frame()?;
        self.present_frame(&ansi)?;
        Ok(())
    }

    /// Detects a terminal window resize and refreshes the cached grid.
    ///
    /// Returns the change when the size differs from the last observation.
    /// The next presented frame is forced to clear the screen so no stale
    /// cells survive, and [`PixportFilterType::Auto`] re-resolves its filter
    /// against the new grid.
    pub fn poll_terminal_resize(&mut self) -> Option<TerminalResize> {
        let cols = detect_terminal_width();
        let rows = detect_terminal_height();
        let resize = detect_resize(self.last_terminal_cols, self.last_terminal_rows, cols, rows)?;
        self.last_terminal_cols = cols;
        self.last_terminal_rows = rows;
        self.force_clear = true;
        Some(resize)
    }

    /// Updates the terminal size from the OS.
    ///
    /// Equivalent to [`Self::poll_terminal_resize`] with the change ignored;
    /// the forced repaint on the next frame still applies.
    pub fn update_terminal_size(&mut self) {
        let _ = self.poll_terminal_resize();
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
    pub fn present_target_size(&self) -> (u32, u32) {
        let rt = self.context.present_target();
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

pub mod ansi {
    use super::PixportFormat;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::fmt::Write;

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
        /// Picks a concrete filter from [`AUTO_FILTER_LADDER`] based on the
        /// terminal grid size: quality-oriented filters for small grids
        /// (cheap because few output pixels), fast filters for large grids.
        Auto,
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

    /// Fallback used by [`PixportFilterType::Auto`] when the ladder is empty
    /// or an unresolved `Auto` reaches the kernel functions: the base rung,
    /// i.e. the filter chosen for the smallest grids.
    const BASE_AUTO_FILTER: PixportFilterType = PixportFilterType::Lanczos3;

    /// One rung of the [`PixportFilterType::Auto`] ladder: from a
    /// `cols x rows` terminal grid upwards, `filter` is selected.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AutoFilterEntry {
        /// Minimum terminal columns for this rung.
        pub cols: u16,
        /// Minimum terminal rows for this rung.
        pub rows: u16,
        /// Filter selected once the terminal grid reaches this rung.
        pub filter: PixportFilterType,
    }

    /// Builds an `&'static [AutoFilterEntry]` ladder for
    /// [`PixportFilterType::Auto`].
    ///
    /// Grid combinations may be written either compactly (`10x20`) or
    /// spaced (`10 x 20`); each entry maps a grid to a concrete filter:
    ///
    /// ```ignore
    /// pub const LADDER: &[AutoFilterEntry] = pixport_auto_filter_table![
    ///     10x20 => PixportFilterType::Lanczos3,
    ///     240x480 => PixportFilterType::Nearest,
    /// ];
    /// ```
    ///
    /// One form must be used for every entry of a single invocation. Rungs
    /// should be listed in ascending grid size; each applies from its grid
    /// up to (but excluding) the next one.
    ///
    /// Compact entries are split at expansion time by [`parse_size_pair`];
    /// a malformed combination resolves to `(0, 0)` and therefore to the
    /// smallest rung of the ladder.
    #[macro_export]
    macro_rules! pixport_auto_filter_table {
        ($($cols:literal x $rows:literal => $filter:expr),+ $(,)?) => {
            &[$( $crate::ui_terminal_pixport::ansi::AutoFilterEntry {
                cols: $cols,
                rows: $rows,
                filter: $filter,
            }),+]
        };
        ($($grid:literal => $filter:expr),+ $(,)?) => {
            &[$( $crate::ui_terminal_pixport::ansi::AutoFilterEntry {
                cols: $crate::ui_terminal_pixport::ansi::parse_size_pair(stringify!($grid)).0,
                rows: $crate::ui_terminal_pixport::ansi::parse_size_pair(stringify!($grid)).1,
                filter: $filter,
            }),+]
        };
    }

    /// Splits a compact `"COLSxROWS"` combination such as `"10x20"` into
    /// `(cols, rows)`.
    ///
    /// Malformed input yields `(0, 0)`, which the `Auto` resolver treats as
    /// "below the first rung" and maps to [`BASE_AUTO_FILTER`].
    #[doc(hidden)]
    pub const fn parse_size_pair(combination: &str) -> (u16, u16) {
        let bytes = combination.as_bytes();
        let mut split = 0usize;
        while split < bytes.len() && bytes[split] != b'x' {
            split += 1;
        }
        if split == 0 || split + 1 >= bytes.len() {
            return (0, 0);
        }
        match (
            parse_u16_in(bytes, 0, split),
            parse_u16_in(bytes, split + 1, bytes.len()),
        ) {
            (Some(cols), Some(rows)) => (cols, rows),
            _ => (0, 0),
        }
    }

    /// Parses decimal `bytes[start..end]` as a `u16`, rejecting overflow
    /// and junk. Index-based (rather than slice ranges) so the whole chain
    /// stays `const`.
    const fn parse_u16_in(bytes: &[u8], start: usize, end: usize) -> Option<u16> {
        if start >= end {
            return None;
        }
        let mut value: u32 = 0;
        let mut i = start;
        while i < end {
            let digit = bytes[i];
            if !digit.is_ascii_digit() {
                return None;
            }
            value = value * 10 + (digit - b'0') as u32;
            if value > u16::MAX as u32 {
                return None;
            }
            i += 1;
        }
        Some(value as u16)
    }

    /// Terminal-size ladder consumed by [`PixportFilterType::Auto`].
    ///
    /// Small grids get the expensive, anti-aliased filters (their output is
    /// tiny, so the cost is negligible while heavy downscaling needs the
    /// quality); as the grid grows the ladder steps down to cheaper filters
    /// so large terminals stay within their frame budget.
    pub const AUTO_FILTER_LADDER: &[AutoFilterEntry] = crate::pixport_auto_filter_table![
        10x20 => PixportFilterType::Lanczos3,
        20x40 => PixportFilterType::Mitchell,
        40x80 => PixportFilterType::Bicubic,
        120x240 => PixportFilterType::Bilinear,
        240x480 => PixportFilterType::Nearest,
    ];

    /// Resolves [`PixportFilterType::Auto`] for a `cols x rows` grid by
    /// walking [`AUTO_FILTER_LADDER`].
    ///
    /// Rungs are compared by total cell count (`cols * rows`), so terminals
    /// whose aspect ratio differs from the ladder entries still advance
    /// through it. The last rung whose cell count is reached wins; grids
    /// smaller than the first rung get [`BASE_AUTO_FILTER`].
    pub const fn resolve_auto_filter(cols: usize, rows: usize) -> PixportFilterType {
        if AUTO_FILTER_LADDER.is_empty() {
            return BASE_AUTO_FILTER;
        }
        let cells = cols.saturating_mul(rows);
        let mut picked = AUTO_FILTER_LADDER[0].filter;
        let mut i = 0;
        while i < AUTO_FILTER_LADDER.len() {
            let rung = &AUTO_FILTER_LADDER[i];
            if cells >= (rung.cols as usize) * (rung.rows as usize) {
                picked = rung.filter;
            }
            i += 1;
        }
        picked
    }

    impl PixportFilterType {
        /// Returns `true` for [`Self::Auto`].
        #[inline]
        #[must_use]
        pub const fn is_auto(self) -> bool {
            matches!(self, Self::Auto)
        }

        /// Resolves [`Self::Auto`] for a `cols x rows` output grid.
        ///
        /// Concrete variants return themselves, so resolution is idempotent
        /// and safe to apply at every resampling layer.
        #[inline]
        #[must_use]
        pub const fn resolve(self, cols: usize, rows: usize) -> Self {
            match self {
                Self::Auto => resolve_auto_filter(cols, rows),
                other => other,
            }
        }
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

    /// Kernel weight for `filter`.
    ///
    /// [`PixportFilterType::Auto`] is resolved by the callers before the
    /// kernels are built; if one reaches here anyway it uses the base
    /// fallback, mirroring [`resolve_auto_filter`] on an empty ladder.
    fn filter_weight(filter: PixportFilterType, x: f32) -> f32 {
        match filter {
            PixportFilterType::Auto | PixportFilterType::Lanczos3 => {
                let ax = x.abs();
                if ax < 3.0 { sinc(ax) * sinc(ax / 3.0) } else { 0.0 }
            }
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
        }
    }

    fn sinc(x: f32) -> f32 {
        if x.abs() < 1e-6 {
            1.0
        } else {
            (x * core::f32::consts::PI).sin() / x
        }
    }

    /// Filter radius for `filter`; see [`filter_weight`] for how
    /// [`PixportFilterType::Auto`] is treated.
    fn filter_support(filter: PixportFilterType) -> f32 {
        match filter {
            PixportFilterType::Auto | PixportFilterType::Lanczos3 => 3.0,
            PixportFilterType::Nearest => 0.0,
            PixportFilterType::Bilinear => 1.0,
            PixportFilterType::Bicubic => 2.0,
            PixportFilterType::Mitchell => 2.0,
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

        /// Samples the framebuffer at fractional coordinates `(x, y)`.
        ///
        /// Integer values correspond to pixel centers. Uses the configured
        /// filter with unit filter scale, resolving [`PixportFilterType::Auto`]
        /// against the source dimensions.
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
            let filter = self.filter.resolve(width, height);
            let xwin = build_window(filter, x, width, 1.0);
            let ywin = build_window(filter, y, height, 1.0);
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
        ///
        /// [`PixportFilterType::Auto`] resolves against the output grid, so
        /// every layer of the pipeline picks the filter for its own size.
        pub fn resample(
            &self,
            pixels: &[u8],
            width: usize,
            height: usize,
            stride: usize,
            out_w: usize,
            out_h: usize,
        ) -> Vec<RgbPixel> {
            let filter = self.filter.resolve(out_w, out_h);
            self.resample_with_filter(pixels, width, height, stride, out_w, out_h, filter)
        }

        /// [`Self::resample`] with an already resolved filter.
        pub fn resample_with_filter(
            &self,
            pixels: &[u8],
            width: usize,
            height: usize,
            stride: usize,
            out_w: usize,
            out_h: usize,
            filter: PixportFilterType,
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
                filter,
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
        ///
        /// [`PixportFilterType::Auto`] resolves against the terminal grid
        /// (not the doubled half-block grid) so the selection is stable
        /// across `use_half_blocks` toggles.
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
            let filter = self.filter.resolve(term_cols, term_rows);
            let rgb = self.resample_with_filter(pixels, width, height, stride, out_w, out_h, filter);
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
        let mut ansi = String::with_capacity(cells.len() * 24);
        let mut current_fg: Option<RgbPixel> = None;
        let mut current_bg: Option<RgbPixel> = None;

        if clear_before_frame {
            ansi.push_str("\x1b[2J\x1b[H");
        }
        for (idx, cell) in cells.iter().enumerate() {
            let term_x = idx % term_cols;
            if term_x == 0 {
                let _ = write!(ansi, "\x1b[{};1H", idx / term_cols + 1);
            }
            if current_fg != Some(cell.fg) {
                let _ = write!(ansi, "\x1b[38;2;{};{};{}m", cell.fg.r, cell.fg.g, cell.fg.b);
                current_fg = Some(cell.fg);
            }
            if let Some(bg) = cell.bg {
                if current_bg != Some(bg) {
                    let _ = write!(ansi, "\x1b[48;2;{};{};{}m", bg.r, bg.g, bg.b);
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

/// Bounded CPU-side queue between asynchronous GPU readbacks and terminal
/// presentation.
///
/// The producer (`Pixport::finish_capture_async`) and the consumer
/// (`Pixport::present_pending`) are decoupled: several frames may be in
/// flight while the terminal still paints the previous one. When the pipe
/// overflows, the *oldest* frame is dropped so the screen always shows the
/// most recent content; [`FramePipe::dropped`] counts them for diagnostics.
pub mod frame_pipe {
    use super::{PixportError, PixportFormat, PixportResult};
    use alloc::collections::VecDeque;
    use alloc::vec::Vec;
    use core::task::{Context, Poll, Waker};

    /// A completed GPU→CPU readback awaiting terminal presentation.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct QueuedFrame {
        /// Monotonic production sequence number (0-based).
        seq: u64,
        /// Framebuffer width in pixels.
        width: u32,
        /// Framebuffer height in pixels.
        height: u32,
        /// Pixel layout of `pixels`.
        format: PixportFormat,
        /// Tightly packed pixels.
        pixels: Vec<u8>,
    }

    impl QueuedFrame {
        /// Wraps a readback buffer with its metadata.
        ///
        /// # Errors
        ///
        /// Returns [`PixportError::FrameDimensionsOverflow`] when `pixels.len()` differs
        /// from the tightly packed `width * height * bytes_per_pixel`
        /// size, catching corrupted or truncated readbacks.
        pub fn new(width: u32, height: u32, format: PixportFormat, pixels: Vec<u8>) -> PixportResult<Self> {
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(format.bytes_per_pixel() as usize))
                .ok_or(PixportError::FrameDimensionsOverflow)?;
            if pixels.len() != expected {
                return Err(PixportError::FrameSizeMismatch);
            }
            Ok(Self {
                seq: 0,
                width,
                height,
                format,
                pixels,
            })
        }

        /// Sequence number assigned by [`FramePipe::push`].
        #[must_use]
        pub const fn seq(&self) -> u64 {
            self.seq
        }

        /// Framebuffer width in pixels.
        #[must_use]
        pub const fn width(&self) -> u32 {
            self.width
        }

        /// Framebuffer height in pixels.
        #[must_use]
        pub const fn height(&self) -> u32 {
            self.height
        }

        /// Pixel layout of the stored buffer.
        #[must_use]
        pub const fn format(&self) -> PixportFormat {
            self.format
        }

        /// Tightly packed pixel buffer.
        #[must_use]
        pub fn pixels(&self) -> &[u8] {
            &self.pixels
        }
    }

    /// Bounded FIFO of [`QueuedFrame`]s with a single waker slot.
    #[derive(Debug)]
    pub struct FramePipe {
        queue: VecDeque<QueuedFrame>,
        capacity: usize,
        next_seq: u64,
        dropped: u64,
        closed: bool,
        waker: Option<Waker>,
    }

    impl FramePipe {
        /// Creates a pipe holding at most `capacity` frames.
        ///
        /// A capacity of 0 is raised to 1 so producers never spin on a pipe
        /// that can never hold a frame.
        #[must_use]
        pub fn new(capacity: usize) -> Self {
            Self {
                queue: VecDeque::with_capacity(capacity.min(16)),
                capacity: capacity.max(1),
                next_seq: 0,
                dropped: 0,
                closed: false,
                waker: None,
            }
        }

        /// Maximum number of queued frames.
        #[must_use]
        pub const fn capacity(&self) -> usize {
            self.capacity
        }

        /// Number of queued frames.
        #[must_use]
        pub fn len(&self) -> usize {
            self.queue.len()
        }

        /// Whether no frames are queued.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.queue.is_empty()
        }

        /// How many frames were dropped because the pipe was full.
        #[must_use]
        pub const fn dropped(&self) -> u64 {
            self.dropped
        }

        /// Whether the pipe was closed by [`Self::close`].
        #[must_use]
        pub const fn is_closed(&self) -> bool {
            self.closed
        }

        /// Enqueues a frame, returning the oldest frame evicted by overflow.
        ///
        /// Always wakes the consumer: even an eviction means new content is
        /// available for presentation.
        pub fn push(&mut self, frame: QueuedFrame) -> Option<QueuedFrame> {
            if self.closed {
                return None;
            }
            let mut evicted = None;
            if self.queue.len() >= self.capacity {
                evicted = self.queue.pop_front();
                if evicted.is_some() {
                    self.dropped += 1;
                }
            }
            let mut frame = frame;
            frame.seq = self.next_seq;
            self.next_seq = self.next_seq.wrapping_add(1);
            self.queue.push_back(frame);
            self.wake();
            evicted
        }

        /// Dequeues the oldest frame without blocking.
        #[must_use]
        pub fn try_recv(&mut self) -> Option<QueuedFrame> {
            let frame = self.queue.pop_front();
            if frame.is_some() {
                self.wake();
            }
            frame
        }

        /// Polls for the next frame.
        ///
        /// Returns [`Poll::Pending`] (and stores `cx`'s waker) while empty
        /// and open; [`Poll::Ready(None)`] once [`Self::close`]d and drained.
        pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<QueuedFrame>> {
            match self.queue.pop_front() {
                Some(frame) => {
                    self.wake();
                    Poll::Ready(Some(frame))
                }
                None if self.closed => Poll::Ready(None),
                None => {
                    let store = match &self.waker {
                        Some(existing) if existing.will_wake(cx.waker()) => false,
                        _ => true,
                    };
                    if store {
                        self.waker = Some(cx.waker().clone());
                    }
                    Poll::Pending
                }
            }
        }

        /// Async variant of [`Self::poll_recv`]: resolves with the next
        /// frame, or `None` once the pipe is closed and drained.
        pub async fn recv(&mut self) -> Option<QueuedFrame> {
            core::future::poll_fn(|cx| self.poll_recv(cx)).await
        }

        /// Closes the pipe: no further frames are accepted, pending waiters
        /// finish once the queue drains.
        pub fn close(&mut self) {
            self.closed = true;
            self.wake();
        }

        /// Drops every queued frame without counting them as dropped.
        pub fn clear(&mut self) {
            self.queue.clear();
        }

        /// Wakes a stored consumer waker, if any.
        fn wake(&mut self) {
            if let Some(waker) = self.waker.take() {
                waker.wake();
            }
        }
    }
}

/// High-level render loop for terminal-based rendering.
pub mod present_loop {
    use super::frame_pipe::FramePipe;
    use super::{Pixport, PixportConfig, PixportError, PixportResult};
    use crate::ui_pipeline::{OwnedFd, PipelineContext};
    use crate::ui_renderer::{RenderLayer, RendererSubsystem};
    use alloc::boxed::Box;
    use codevar_consoleutil::console_ansi::{cursor, erase};
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll, Waker};

    /// Configuration for the terminal render loop.
    #[derive(Debug, Clone)]
    pub struct RenderLoopConfig {
        /// Frame rate of Render loop
        pub frame_rate: codevar_time_core::TimeDuration,
        /// Pixport configuration.
        pub pixport_config: PixportConfig,
    }

    /// Runs a render loop that captures frames and outputs to terminal.
    ///
    /// `max_cols == 0` / `max_rows == 0` follow the live terminal size: the
    /// loop polls for resizes every frame, forces a repaint on change, and
    /// lets [`super::ansi::PixportFilterType::Auto`] re-pick its filter.
    pub fn run_terminal_present_loop<'p, L>(
        pipeline: &'p PipelineContext,
        mut renderer: RendererSubsystem<'p>,
        layer: L,
        config: RenderLoopConfig,
    ) -> PixportResult<()>
    where
        L: RenderLayer + 'p,
    {
        codevar_consoleutil::init_ansi_support();
        renderer.layers_mut().add(Box::new(layer))?;
        let mut pixport = Pixport::new(pipeline, config.pixport_config)?;
        codevar_consoleutil::write_stdout(cursor::hide().as_bytes())?;
        codevar_consoleutil::write_stdout(erase::screen().as_bytes())?;

        let frame_duration = config.frame_rate;
        loop {
            let frame_start = codevar_time_core::SystemTime::monotonic_nanos();
            let _ = pixport.poll_terminal_resize();
            renderer.begin_frame(None)?;
            renderer.present_frame()?;
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
    fn wait_for_gpu_sync(sync_file: &OwnedFd) -> PixportResult<()> {
        let mut descriptor = libc::pollfd {
            fd: sync_file.as_raw(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 5000) };
        if ready <= 0 {
            return Err(PixportError::GpuSyncTimeout);
        }
        Ok(())
    }

    /// Deadline for [`wait_gpu_sync_async`]: 5 seconds, matching the
    /// blocking [`wait_for_gpu_sync`] timeout.
    const GPU_SYNC_TIMEOUT_NS: u64 = 5_000_000_000;

    /// Async variant of [`wait_for_gpu_sync`].
    ///
    /// Checks the sync file in bounded non-blocking slices (1 ms sleep plus
    /// a cooperative yield between them) so an executor stays responsive,
    /// giving up after [`GPU_SYNC_TIMEOUT_NS`].
    ///
    /// # Errors
    ///
    /// Returns [`PixportError::GpuSyncPollFailed`] when `poll(2)` fails or
    /// [`PixportError::GpuSyncTimeout`] when the deadline passes before the
    /// GPU signals readiness.
    pub async fn wait_gpu_sync_async(sync_file: &OwnedFd) -> PixportResult<()> {
        let deadline = codevar_time_core::SystemTime::monotonic_nanos().saturating_add(GPU_SYNC_TIMEOUT_NS);
        loop {
            let mut descriptor = libc::pollfd {
                fd: sync_file.as_raw(),
                events: libc::POLLIN,
                revents: 0,
            };
            let ready = unsafe { libc::poll(&mut descriptor, 1, 0) };
            if ready > 0 {
                return Ok(());
            }
            if ready < 0 {
                return Err(PixportError::GpuSyncPollFailed);
            }
            if codevar_time_core::SystemTime::monotonic_nanos() >= deadline {
                return Err(PixportError::GpuSyncTimeout);
            }
            sleep_for(codevar_time_core::TimeDuration::from_millis(1));
            yield_now().await;
        }
    }

    /// Yields to the executor once: the first poll returns [`Poll::Pending`]
    /// after waking the current task, the second [`Poll::Ready`].
    async fn yield_now() {
        struct YieldNow(bool);

        impl Future for YieldNow {
            type Output = ();
            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                if self.0 {
                    Poll::Ready(())
                } else {
                    self.0 = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        YieldNow(false).await
    }

    /// Drives `future` to completion on the current thread.
    ///
    /// A minimal single-threaded executor for running
    /// [`run_terminal_present_loop_async`] without an async runtime. Every
    /// future in this module re-registers its waker or makes progress from
    /// re-polling alone, so a no-op waker plus a 1 ms sleep on
    /// [`Poll::Pending`] is sufficient (and never busy-spins).
    pub fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(output) => return output,
                Poll::Pending => sleep_for(codevar_time_core::TimeDuration::from_millis(1)),
            }
        }
    }

    /// Asynchronous counterpart of [`run_terminal_present_loop`].
    ///
    /// Renders a frame, then awaits the GPU sync file and the readback
    /// fence before queueing the framebuffer on `pipe`. Every frame that
    /// completes is presented in order as a *clear → render* pair
    /// (`super::Pixport::present_pending`), so overlapping GPU work never
    /// tears the terminal: `clear → render → clear → render → …`.
    ///
    /// When the pipe is full, its oldest frame is dropped so the terminal
    /// converges on the newest frame (see
    /// [`super::frame_pipe::FramePipe::dropped`]).
    ///
    /// Drive it with [`block_on`] or another executor that keeps `pipe`
    /// borrowed for the future's lifetime:
    ///
    /// ```ignore
    /// let mut pipe = FramePipe::default();
    /// present_loop::block_on(present_loop::run_terminal_present_loop_async(
    ///     pipeline, renderer, layer, config, &mut pipe,
    /// ))?;
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates the same rendering, Vulkan and I/O failures as
    /// [`run_terminal_present_loop`].
    pub async fn run_terminal_present_loop_async<'p, L>(
        pipeline: &'p PipelineContext,
        mut renderer: RendererSubsystem<'p>,
        layer: L,
        config: RenderLoopConfig,
        pipe: &mut FramePipe,
    ) -> PixportResult<()>
    where
        L: RenderLayer + 'p,
    {
        codevar_consoleutil::init_ansi_support();
        renderer.layers_mut().add(Box::new(layer))?;
        let mut pixport = Pixport::new(pipeline, config.pixport_config)?;
        codevar_consoleutil::write_stdout(cursor::hide().as_bytes())?;
        codevar_consoleutil::write_stdout(erase::screen().as_bytes())?;

        let frame_duration = config.frame_rate;
        loop {
            let frame_start = codevar_time_core::SystemTime::monotonic_nanos();
            let _ = pixport.poll_terminal_resize();
            renderer.begin_frame(None)?;
            renderer.present_frame()?;
            let sync_file = renderer.end_frame()?;
            pixport.begin_capture()?;
            wait_gpu_sync_async(&sync_file).await?;
            drop(sync_file);
            let frame = pixport.finish_capture_async().await?;
            pipe.push(frame);
            pixport.present_pending(pipe)?;
            let elapsed = codevar_time_core::SystemTime::monotonic_nanos() - frame_start;
            let elapsed_dur = codevar_time_core::TimeDuration::from_nanos(elapsed);
            let remaining = frame_duration.saturating_sub(elapsed_dur);
            sleep_for(remaining);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui_terminal_pixport::ansi::{AnsiCell, RgbPixel};

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
        assert!(config.clear_policy.should_clear());
        assert!(config.cell_density.is_half_blocks());
        assert_eq!(config.filter, PixportFilterType::default());
    }

    #[test]
    fn test_nearest_color_converter() {
        let converter =
            ansi::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Nearest);
        // 2x1 image: red, green
        let mut pixels = vec![0u8; 2 * 1 * 4];
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]); // red (BGRA)
        pixels[4..8].copy_from_slice(&[0, 255, 0, 255]); // green (BGRA)

        let cells = converter.pixels_to_ansi_cells(&pixels, 2, 1, 8, 2, 1, false);

        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].fg, ansi::RgbPixel { r: 255, g: 0, b: 0 });
        assert_eq!(cells[1].fg, ansi::RgbPixel { r: 0, g: 255, b: 0 });
        assert_eq!(cells[0].char, '█');
    }

    #[test]
    fn test_nearest_half_blocks() {
        let converter =
            ansi::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Nearest);
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
        assert_eq!(cells[0].fg, ansi::RgbPixel { r: 255, g: 0, b: 0 }); // upper red
        assert_eq!(cells[0].bg, Some(ansi::RgbPixel { r: 0, g: 0, b: 255 })); // lower blue
        assert_eq!(cells[1].fg, ansi::RgbPixel { r: 0, g: 255, b: 0 }); // upper green
        assert_eq!(cells[1].bg, Some(ansi::RgbPixel { r: 255, g: 255, b: 0 })); // lower yellow
        assert_eq!(cells[0].char, '▀');
    }

    #[test]
    fn test_converter_defaults() {
        let c = ansi::AnsiColorConverter::new(PixportFormat::Bgra8888);
        assert_eq!(c.filter(), PixportFilterType::Bilinear);
        let c2 = ansi::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Lanczos3);
        assert_eq!(c2.filter(), PixportFilterType::Lanczos3);
    }

    #[test]
    fn test_bilinear_upsample_midpoint() {
        let mut pixels = vec![0u8; 2 * 1 * 4];
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]); // red
        pixels[4..8].copy_from_slice(&[255, 0, 0, 255]); // blue (BGRA: B=255)
        let c = ansi::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Bilinear);
        let mid = c.sample_pixel(&pixels, 2, 1, 8, 0.5, 0.0);
        assert_eq!(mid, ansi::RgbPixel { r: 128, g: 0, b: 128 });
    }

    #[test]
    fn test_upsample_gradient() {
        let mut pixels = vec![0u8; 2 * 1 * 4];
        pixels[0..4].copy_from_slice(&[0, 0, 255, 255]); // red
        pixels[4..8].copy_from_slice(&[255, 0, 0, 255]); // blue
        let c = ansi::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, PixportFilterType::Bilinear);
        let rgb = c.resample(&pixels, 2, 1, 8, 3, 1);
        assert_eq!(rgb[0], ansi::RgbPixel { r: 255, g: 0, b: 0 });
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
            let c = ansi::AnsiColorConverter::with_filter(PixportFormat::Bgra8888, filter);
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
