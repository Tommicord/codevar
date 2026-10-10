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

//! Per-frame recording context handed to one pipe: the framebuffer
//! snapshot, the pipe's command buffer and the dynamic-rendering scope
//! opened with [`PipeCtx::begin_render`].

use alloc::boxed::Box;
use alloc::vec::Vec;
use ash::vk;
use core::future::Future;
use core::pin::Pin;

use crate::base::CompositorError;
use crate::offscreen::OffscreenFramebuffer;

/// What a pipe asks the compositor to do once its future resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipeOutcome {
    /// Stay registered; the compositor requests a new future for the
    /// next frame.
    Keep,
    /// Drop the pipe after the frame completes. Its renderer is released
    /// at the next [`Compositor::begin_frame`](Compositor::begin_frame).
    Remove,
}

/// Result of one poll of a pipe future.
pub type PipeResult = Result<PipeOutcome, CompositorError>;

/// Async body of one pipe for the lifetime of a single frame.
///
/// The lifetime is the one borrowed from the pipe's renderer and
/// [`PipeCtx`]; it is erased once in `trampoline` (see the safety
/// comment there), so the compositor can store the future next to — and
/// drop it before — the data it borrows.
pub type PipeFuture<'a> = Pin<Box<dyn Future<Output = PipeResult> + 'a>>;

/// Type-erased entry point a [`Pipe`](Pipe) dispatches to.
pub type PipeEntry = fn(PipeEnv) -> PipeFuture<'static>;

/// Raw arguments handed to a [`PipeEntry`].
///
/// The pointers are the pipe's renderer (of the concrete `PipeSource`
/// type) and the frame's [`PipeCtx`]; both stay valid for as long as the
/// returned future exists.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PipeEnv {
    /// `*mut <T as PipeSource>` for the renderer `T` of this pipe.
    pub renderer: *mut (),
    /// Frame context owned by the compositor for this pipe.
    pub ctx: *mut PipeCtx,
}

/// Per-frame recording state handed to one pipe.
///
/// A pipe draws into its own offscreen targets: [`PipeCtx::begin_render`]
/// opens a dynamic-rendering scope on one of them and
/// [`PipeCtx::end_render`] closes it. Layout transitions are recorded by
/// the compositor before any pipe runs, so a pipe must never transition
/// the image it renders into (see the module documentation).
///
/// The struct is lifetime-free: the compositor builds it from a
/// [`PipelineContext`](crate::pipeline::PipelineContext) that outlives every pipe and hands a pointer to
/// it through [`PipeEnv`], which keeps the self-referential pipe futures
/// borrow-checkable at the cost of one documented lifetime erasure.
pub struct PipeCtx {
    /// Logical device of the owning pipeline; valid for `'p`.
    device: *const ash::Device,
    /// Primary command buffer being recorded for this pipe this frame.
    command_buffer: vk::CommandBuffer,
    /// Presentation target width in pixels (pipe viewport hint).
    width: u32,
    /// Presentation target height in pixels (pipe viewport hint).
    height: u32,
    /// Format of the presentation target.
    format: vk::Format,
    /// Number of frames composited so far (0 for the first frame).
    frame_index: u64,
    /// Global index of this pipe's first framebuffer inside
    /// [`PipeCtx::framebuffers`].
    base: usize,
    /// Snapshot of the frame's framebuffers.
    framebuffers: Vec<OffscreenFramebuffer>,
}

impl PipeCtx {
    /// Builds the context for one pipe of one frame.
    #[allow(
        clippy::too_many_arguments,
        reason = "raw Vulkan handle plus the frame geometry this context snapshots"
    )]
    pub(crate) fn new(
        device: *const ash::Device,
        command_buffer: vk::CommandBuffer,
        width: u32,
        height: u32,
        format: vk::Format,
        frame_index: u64,
        base: usize,
        framebuffers: &[OffscreenFramebuffer],
    ) -> Self {
        Self {
            device,
            command_buffer,
            width,
            height,
            format,
            frame_index,
            base,
            framebuffers: framebuffers.to_vec(),
        }
    }

    /// The logical device owning this pipe's command buffer.
    ///
    /// # Panics
    ///
    /// Never panics: the compositor only builds contexts whose device
    /// pointer comes from a live [`PipelineContext`](crate::pipeline::PipelineContext).
    #[inline]
    #[must_use]
    pub fn device(&self) -> &ash::Device {
        // SAFETY: the device pointer is copied once from the
        // `PipelineContext` that owns this compositor and is never
        // mutated; that context outlives every pipe (enforced by `'p`),
        // so the pointee stays valid while the context exists.
        unsafe { &*self.device }
    }

    /// The primary command buffer being recorded for this pipe.
    ///
    /// It is in the recording state between [`PipeCtx::begin_render`]
    /// and [`PipeCtx::end_render`] while the pipe's future runs.
    #[inline]
    #[must_use]
    pub const fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Presentation target width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Presentation target height in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Vulkan format of the presentation target.
    #[inline]
    #[must_use]
    pub const fn color_format(&self) -> vk::Format {
        self.format
    }

    /// Number of frames composited before this one.
    #[inline]
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Global index of this pipe's first framebuffer.
    #[inline]
    #[must_use]
    pub const fn base(&self) -> usize {
        self.base
    }

    /// Number of framebuffers collected for this frame.
    #[inline]
    #[must_use]
    pub fn framebuffer_count(&self) -> usize {
        self.framebuffers.len()
    }

    /// The pipe-local framebuffer at `local_index` (its global slot is
    /// [`PipeCtx::base`] + `local_index`).
    #[must_use]
    pub fn framebuffer(&self, local_index: usize) -> Option<OffscreenFramebuffer> {
        self.framebuffers
            .get(self.base + local_index)
            .copied()
    }

    /// Opens a dynamic-rendering scope on the pipe-local framebuffer at
    /// `local_index`.
    ///
    /// `clear` selects the attachment load operation: `Some(color)`
    /// clears the target first (always pass this for a pipe's first
    /// frame, when the image still holds `UNDEFINED` contents) and
    /// `None` loads the previous contents. The viewport and scissor are
    /// set to the framebuffer's extent.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — no framebuffer was collected at
    ///   that index, so the pipe cannot render.
    ///
    /// # Safety of use
    ///
    /// Only call this while the pipe's command buffer is recording and
    /// after the compositor's layout pass ran for the frame (the
    /// framebuffer is then in `COLOR_ATTACHMENT_OPTIMAL`).
    pub fn begin_render(
        &mut self,
        local_index: usize,
        clear: Option<[f32; 4]>,
    ) -> Result<(), CompositorError> {
        let framebuffer = self
            .framebuffer(local_index)
            .filter(OffscreenFramebuffer::is_valid)
            .ok_or(CompositorError::Internal(
                "begin_render: no framebuffer collected at this index",
            ))?;
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(framebuffer.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(if clear.is_some() {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            })
            .store_op(vk::AttachmentStoreOp::STORE);
        let attachment = match clear {
            Some(color) => attachment.clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            }),
            None => attachment,
        };
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: framebuffer.width,
                height: framebuffer.height,
            },
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&attachment));
        // SAFETY: the command buffer is recording (checked by the
        // caller's contract), the image view is compatible with the
        // `COLOR_ATTACHMENT_OPTIMAL` layout recorded by the layout pass,
        // and the render area fits the image; dynamic rendering needs no
        // render pass or framebuffer object.
        unsafe {
            self.device()
                .cmd_begin_rendering(self.command_buffer, &rendering_info)
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: framebuffer.width as f32,
            height: framebuffer.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: the rendering scope opened above has not been ended and
        // the framebuffer extent is non-zero (validated by
        // `OffscreenFramebuffer::is_valid`).
        unsafe {
            self.device()
                .cmd_set_viewport(self.command_buffer, 0, core::slice::from_ref(&viewport));
            self.device()
                .cmd_set_scissor(self.command_buffer, 0, core::slice::from_ref(&area));
        }
        Ok(())
    }

    /// Closes the dynamic-rendering scope opened by
    /// [`PipeCtx::begin_render`].
    ///
    /// Recording a mismatched begin/end pair corrupts the command
    /// buffer, so pair every successful `begin_render` with exactly one
    /// call to this method — including on the pipe's error paths.
    pub fn end_render(&mut self) {
        // SAFETY: the command buffer is recording inside a rendering
        // scope opened by `begin_render` (the pipe's pairing contract).
        unsafe {
            self.device()
                .cmd_end_rendering(self.command_buffer)
        };
    }
}
