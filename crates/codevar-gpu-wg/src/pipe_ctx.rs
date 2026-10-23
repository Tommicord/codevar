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

//! Per-frame recording context handed to one pipe: the pipeline context,
//! the frame's command encoder and the dynamic-rendering scope opened with
//! [`PipeCtx::begin_render`].
//!
//! The shape mirrors `codevar-gpu-vk`, with one difference the API forces:
//! Vulkan records into a shared `VkCommandBuffer` handle, while `wgpu-hal`
//! exposes the encoder as a trait object that lives in the renderer. The
//! context therefore borrows the encoder through an erased pointer and
//! hands it back with [`PipeCtx::command_encoder`].
//!
//! The struct is lifetime-free: the compositor builds it from a
//! [`WgContext`] that outlives every pipe and hands a pointer to it through
//! [`PipeEnv`], which keeps the self-referential pipe futures
//! borrow-checkable at the cost of one documented lifetime erasure.

use alloc::boxed::Box;
use core::future::Future;
use core::pin::Pin;

use wgpu_types as wgt;

use crate::offscreen::OffscreenFramebuffer;
use crate::pipeline::WgContext;
use crate::types::CompositorError;

#[cfg(target_arch = "wasm32")]
use wgpu_hal::CommandEncoder as _;

/// What a pipe asks the compositor to do once its future resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipeOutcome {
    /// Stay registered; the compositor requests a new future for the
    /// next frame.
    Keep,
    /// Drop the pipe after the frame completes. Its renderer is released
    /// at the next `Compositor::begin_frame`.
    Remove,
}

/// Result of one poll of a pipe future.
pub type PipeResult = Result<PipeOutcome, CompositorError>;

/// Async body of one pipe for the lifetime of a single frame.
///
/// The lifetime is the one borrowed from the pipe's renderer and
/// [`PipeCtx`]; it is erased once in the pipe trampoline, so the compositor
/// can store the future next to — and drop it before — the data it borrows.
pub type PipeFuture<'a> = Pin<Box<dyn Future<Output = PipeResult> + 'a>>;

/// Type-erased entry point a [`Pipe`](crate::pipe::Pipe) dispatches to.
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
/// opens a dynamic-rendering scope on one of them, the pipe records its
/// commands through [`PipeCtx::command_encoder`] and
/// [`PipeCtx::end_render`] closes the scope. Layout transitions are
/// recorded by the compositor before any pipe runs, so a pipe must never
/// transition the image it renders into.
pub struct PipeCtx {
    /// Pipeline context that owns the device every command of the frame
    /// uses; valid for `'p`.
    context: *const WgContext,
    /// The frame's command encoder, erased to `*mut ()` so the concrete
    /// `wgpu-hal` type is not named off the web.
    #[cfg_attr(
        not(target_arch = "wasm32"),
        allow(dead_code, reason = "read by `command_encoder`, which only exists on wasm32")
    )]
    encoder: *mut (),
    /// Presentation target width in pixels (pipe viewport hint).
    width: u32,
    /// Presentation target height in pixels (pipe viewport hint).
    height: u32,
    /// Format of the presentation target.
    format: wgt::TextureFormat,
    /// Number of frames composited so far (0 for the first frame).
    frame_index: u64,
}

impl PipeCtx {
    /// Builds the context for one pipe of one frame.
    ///
    /// `encoder` is `*mut wgpu_hal::gles::CommandEncoder` on `wasm32` and
    /// is never dereferenced on any other target.
    #[allow(
        clippy::too_many_arguments,
        dead_code,
        reason = "the pipeline context plus the frame geometry this context snapshots"
    )]
    pub(crate) fn new(
        context: *const WgContext,
        encoder: *mut (),
        width: u32,
        height: u32,
        format: wgt::TextureFormat,
        frame_index: u64,
    ) -> Self {
        Self {
            context,
            encoder,
            width,
            height,
            format,
            frame_index,
        }
    }

    /// The pipeline context owning this pipe's device.
    ///
    /// # Panics
    ///
    /// Never panics: the compositor only builds contexts whose context
    /// pointer comes from a live [`WgContext`].
    #[inline]
    #[must_use]
    pub fn device(&self) -> &WgContext {
        // SAFETY: the context pointer is copied once from the `WgContext`
        // that owns this compositor and is never mutated; that context
        // outlives every pipe, so the pointee stays valid while the context
        // exists.
        unsafe { &*self.context }
    }

    /// The frame's command encoder, in the recording state between
    /// [`PipeCtx::begin_render`] and [`PipeCtx::end_render`].
    ///
    /// # Panics
    ///
    /// Never panics: the compositor only builds contexts whose encoder
    /// pointer comes from the renderer that owns the frame.
    #[cfg(target_arch = "wasm32")]
    #[inline]
    #[must_use]
    pub fn command_encoder(&mut self) -> &mut wgpu_hal::gles::CommandEncoder {
        // SAFETY: the encoder pointer is the frame's encoder, copied from
        // the renderer that exclusively owns it for the duration of this
        // context; no other reference to it exists while `self` is lent out.
        unsafe { &mut *self.encoder.cast() }
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

    /// `wgpu-types` format of the presentation target.
    #[inline]
    #[must_use]
    pub const fn color_format(&self) -> wgt::TextureFormat {
        self.format
    }

    /// Number of frames composited before this one.
    #[inline]
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Opens a dynamic-rendering scope on `framebuffer`.
    ///
    /// `clear` selects the attachment load operation: `Some(color)` clears
    /// the target first (always pass this for a pipe's first frame, when
    /// the image still holds undefined contents) and `None` loads the
    /// previous contents. The viewport and scissor are set to the
    /// framebuffer's extent.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the framebuffer refers to no real
    ///   texture, so the pipe cannot render.
    /// * [`CompositorError::UnsupportedTarget`] — the crate was compiled
    ///   off the web, where no encoder exists.
    ///
    /// # Safety of use
    ///
    /// Only call this while the pipe's encoder is recording and after the
    /// compositor's layout pass ran for the frame (the framebuffer is then
    /// a color target of the pass).
    pub fn begin_render(
        &mut self,
        framebuffer: OffscreenFramebuffer<'_>,
        clear: Option<[f32; 4]>,
    ) -> Result<(), CompositorError> {
        #[cfg(target_arch = "wasm32")]
        {
            use wgpu_hal::{Attachment, AttachmentOps, ColorAttachment, RenderPassDescriptor};

            if !framebuffer.is_valid() {
                return Err(CompositorError::Internal(
                    "begin_render: the framebuffer holds no texture",
                ));
            }
            let attachment = ColorAttachment {
                target: Attachment {
                    view: framebuffer.view,
                    usage: wgt::TextureUses::COLOR_TARGET,
                },
                depth_slice: None,
                resolve_target: None,
                ops: if clear.is_some() {
                    AttachmentOps::LOAD_CLEAR
                } else {
                    AttachmentOps::LOAD
                } | AttachmentOps::STORE,
                clear_value: match clear {
                    Some(color) => wgt::Color {
                        r: f64::from(color[0]),
                        g: f64::from(color[1]),
                        b: f64::from(color[2]),
                        a: f64::from(color[3]),
                    },
                    None => wgt::Color::TRANSPARENT,
                },
            };
            let attachments = [Some(attachment)];
            let extent = wgt::Extent3d {
                width: framebuffer.width,
                height: framebuffer.height,
                depth_or_array_layers: 1,
            };
            let pass = RenderPassDescriptor {
                label: Some("codevar-pipe"),
                extent,
                sample_count: 1,
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            let encoder = self.command_encoder();
            // SAFETY: the encoder is recording (checked by the caller's
            // contract), the view is compatible with the color-target usage
            // the layout pass recorded, and the render area fits the image.
            unsafe { encoder.begin_render_pass(&pass) }
                .map_err(|err| CompositorError::graphics("begin_render_pass", err))?;
            let viewport = wgpu_hal::Rect {
                x: 0.0,
                y: 0.0,
                w: framebuffer.width as f32,
                h: framebuffer.height as f32,
            };
            let scissor = wgpu_hal::Rect {
                x: 0,
                y: 0,
                w: framebuffer.width,
                h: framebuffer.height,
            };
            // SAFETY: the rendering scope opened above has not been ended
            // and the framebuffer extent is non-zero (checked by
            // `OffscreenFramebuffer::is_valid`).
            unsafe {
                encoder.set_viewport(&viewport, 0.0..1.0);
                encoder.set_scissor_rect(&scissor);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (framebuffer, clear);
            Err(CompositorError::UnsupportedTarget)
        }
    }

    /// Closes the dynamic-rendering scope opened by
    /// [`PipeCtx::begin_render`].
    ///
    /// Recording a mismatched begin/end pair corrupts the command buffer,
    /// so pair every successful `begin_render` with exactly one call to
    /// this method — including on the pipe's error paths.
    pub fn end_render(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            // SAFETY: the encoder is recording inside a rendering scope
            // opened by `begin_render` (the pipe's pairing contract).
            unsafe { self.command_encoder().end_render_pass() };
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Nothing to close off the web.
        }
    }
}
