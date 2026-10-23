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

//! The WebGPU renderer subsystem: the three-step frame lifecycle of
//! [`GpuRenderer`].
//!
//! [`WgRenderer`] owns one command encoder and the canvas texture of the
//! frame in flight. [`GpuRenderer::begin_frame`] acquires the canvas
//! texture, [`GpuRenderer::submit_frame`] records a clear pass over it and
//! submits it, and [`GpuRenderer::end_frame`] presents it. The Vulkan
//! subsystem records the same three steps and exports a `sync_file`
//! instead of presenting in the browser.
//!
//! The real subsystem is compiled on `wasm32` only; every other target
//! compiles the stub at the bottom, whose frame methods report
//! [`CompositorError::UnsupportedTarget`].

use codevar_gpu_core::{
    Backend, FrameCompletion, GpuCapabilities, GpuFeature, GpuFeatures, GpuLimits, GpuRenderer,
};

use crate::types::{CompositorError, CompositorPhase, CompositorState};

/// The features the WebGPU renderer reports.
#[inline]
#[must_use]
pub const fn renderer_features() -> GpuFeatures {
    GpuFeatures::NONE
        .with(GpuFeature::SrgbSurface)
        .with(GpuFeature::DirectDisplayList)
        .with(GpuFeature::Compute)
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    use core::iter;

    use wgpu_hal::gles::{CommandEncoder, Texture};
    use wgpu_hal::{
        Attachment, AttachmentOps, ColorAttachment, CommandEncoder as _, CommandEncoderDescriptor,
        Device as _, Queue as _, RenderPassDescriptor, StateTransition, Surface as _, TextureBarrier,
        TextureViewDescriptor,
    };
    use wgpu_types as wgt;

    use crate::pipeline::WgContext;

    /// The full subresource range of a single-level, single-layer texture.
    fn full_range() -> wgt::ImageSubresourceRange {
        wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        }
    }

    /// The WebGPU frame lifecycle.
    ///
    /// The `'p` lifetime keeps the renderer from outliving the
    /// [`WgContext`] that owns its device, queue and canvas surface.
    pub struct WgRenderer<'p> {
        /// Device, queue and canvas surface every frame uses.
        context: &'p WgContext,
        /// The single command encoder, reset and reused every frame.
        encoder: CommandEncoder,
        /// Whether the subsystem accepts frames.
        state: CompositorState,
        /// Progress of the frame in flight.
        phase: CompositorPhase,
        /// Canvas texture acquired for the frame in flight, presented (or
        /// discarded on `Drop`) when the frame ends.
        pending: Option<Texture>,
        /// Value the next submission stores in the context fence.
        fence_value: u64,
        /// Number of frames fully presented.
        frame_index: u64,
    }

    impl<'p> WgRenderer<'p> {
        /// Creates the subsystem in the [`CompositorState::Stopped`] state.
        ///
        /// Call [`WgRenderer::start`] before the first frame.
        ///
        /// # Errors
        ///
        /// [`CompositorError::Graphics`] when the driver rejects the
        /// command encoder.
        pub fn new(context: &'p WgContext) -> Result<Self, CompositorError> {
            let descriptor = CommandEncoderDescriptor {
                label: Some("codevar-renderer"),
                queue: context.queue(),
            };
            // SAFETY: the context owns the queue and outlives the encoder.
            let encoder = unsafe {
                context
                    .device()
                    .create_command_encoder(&descriptor)
            }
            .map_err(|err| CompositorError::graphics("create_command_encoder", err))?;
            Ok(Self {
                context,
                encoder,
                state: CompositorState::Stopped,
                phase: CompositorPhase::Idle,
                pending: None,
                fence_value: 0,
                frame_index: 0,
            })
        }

        /// The lifecycle state.
        #[inline]
        #[must_use]
        pub const fn state(&self) -> CompositorState {
            self.state
        }

        /// Progress of the frame in flight.
        #[inline]
        #[must_use]
        pub const fn phase(&self) -> CompositorPhase {
            self.phase
        }

        /// Moves from [`CompositorState::Stopped`] to
        /// [`CompositorState::Running`].
        ///
        /// # Errors
        ///
        /// [`CompositorError::InvalidState`] when the subsystem is not
        /// stopped.
        pub fn start(&mut self) -> Result<(), CompositorError> {
            match self.state {
                CompositorState::Stopped => {
                    self.state = CompositorState::Running;
                    Ok(())
                }
                state => Err(CompositorError::InvalidState {
                    operation: "start",
                    state,
                }),
            }
        }

        /// Moves from [`CompositorState::Running`] to
        /// [`CompositorState::Paused`].
        ///
        /// # Errors
        ///
        /// [`CompositorError::InvalidState`] when the subsystem is not
        /// running.
        pub fn pause(&mut self) -> Result<(), CompositorError> {
            match self.state {
                CompositorState::Running => {
                    self.state = CompositorState::Paused;
                    Ok(())
                }
                state => Err(CompositorError::InvalidState {
                    operation: "pause",
                    state,
                }),
            }
        }

        /// Moves from [`CompositorState::Paused`] back to
        /// [`CompositorState::Running`].
        ///
        /// # Errors
        ///
        /// [`CompositorError::InvalidState`] when the subsystem is not
        /// paused.
        pub fn resume(&mut self) -> Result<(), CompositorError> {
            match self.state {
                CompositorState::Paused => {
                    self.state = CompositorState::Running;
                    Ok(())
                }
                state => Err(CompositorError::InvalidState {
                    operation: "resume",
                    state,
                }),
            }
        }

        /// Moves back to [`CompositorState::Stopped`].
        ///
        /// # Errors
        ///
        /// [`CompositorError::FramePhase`] when a frame is in flight.
        pub fn stop(&mut self) -> Result<(), CompositorError> {
            if self.phase != CompositorPhase::Idle {
                return Err(CompositorError::FramePhase {
                    operation: "stop",
                    expected: CompositorPhase::Idle,
                    actual: self.phase,
                });
            }
            self.state = CompositorState::Stopped;
            Ok(())
        }

        /// Acquires the canvas texture of the next frame.
        ///
        /// `wait` must be `None` or [`FrameCompletion::Complete`]: the
        /// browser queue serialises its own submissions, so a foreign
        /// completion handle cannot be awaited and is rejected rather than
        /// ignored.
        ///
        /// # Errors
        ///
        /// * [`CompositorError::InvalidState`] when the subsystem is not
        ///   running.
        /// * [`CompositorError::FramePhase`] when a frame is already in
        ///   flight.
        /// * [`CompositorError::Graphics`] when the canvas texture cannot be
        ///   acquired.
        pub fn begin_frame(&mut self, wait: Option<FrameCompletion>) -> Result<(), CompositorError> {
            if self.state != CompositorState::Running {
                return Err(CompositorError::InvalidState {
                    operation: "begin_frame",
                    state: self.state,
                });
            }
            if self.phase != CompositorPhase::Idle {
                return Err(CompositorError::FramePhase {
                    operation: "begin_frame",
                    expected: CompositorPhase::Idle,
                    actual: self.phase,
                });
            }
            if let Some(completion) = wait
                && !matches!(completion, FrameCompletion::Complete)
            {
                return Err(CompositorError::Internal(
                    "the WebGPU renderer cannot wait on a foreign frame completion",
                ));
            }
            // SAFETY: no frame is in flight, so the surface has no acquired
            // texture, and the fence is the one every submission signals.
            let acquired = unsafe {
                self.context
                    .surface()
                    .acquire_texture(None, self.context.fence())
            }
            .map_err(|err| CompositorError::graphics("acquire_texture", err))?;
            self.pending = Some(acquired.texture);
            self.phase = CompositorPhase::Recording;
            Ok(())
        }

        /// Records a clear pass over the acquired canvas texture and submits
        /// it, signalling the context fence.
        ///
        /// # Errors
        ///
        /// * [`CompositorError::FramePhase`] when no frame is being
        ///   recorded.
        /// * [`CompositorError::Graphics`] when a graphics call fails.
        pub fn submit_frame(&mut self) -> Result<(), CompositorError> {
            if self.phase != CompositorPhase::Recording {
                return Err(CompositorError::FramePhase {
                    operation: "submit_frame",
                    expected: CompositorPhase::Recording,
                    actual: self.phase,
                });
            }
            let texture = match self.pending.as_ref() {
                Some(texture) => texture,
                None => {
                    return Err(CompositorError::Internal(
                        "submit_frame ran without an acquired canvas texture",
                    ));
                }
            };
            let device = self.context.device();

            let view_descriptor = TextureViewDescriptor {
                label: Some("codevar-surface-view"),
                format: self.context.format(),
                dimension: wgt::TextureViewDimension::D2,
                usage: wgt::TextureUses::COLOR_TARGET,
                range: full_range(),
            };
            // SAFETY: the view covers the whole surface texture, which was
            // created on this device.
            let view = unsafe { device.create_texture_view(texture, &view_descriptor) }
                .map_err(|err| CompositorError::graphics("create_texture_view", err))?;

            // SAFETY: the encoder is not recording; the previous frame was
            // reset after its submission.
            unsafe { self.encoder.begin_encoding(Some("codevar-frame")) }
                .map_err(|err| CompositorError::graphics("begin_encoding", err))?;

            let barrier = TextureBarrier {
                texture,
                range: full_range(),
                usage: StateTransition {
                    from: wgt::TextureUses::PRESENT,
                    to: wgt::TextureUses::COLOR_TARGET,
                },
            };
            // SAFETY: the barrier names the acquired surface texture and the
            // usage it is being moved to.
            unsafe {
                self.encoder
                    .transition_textures(iter::once(barrier))
            };

            let attachment = ColorAttachment {
                target: Attachment {
                    view: &view,
                    usage: wgt::TextureUses::COLOR_TARGET,
                },
                depth_slice: None,
                resolve_target: None,
                ops: AttachmentOps::LOAD_CLEAR | AttachmentOps::STORE,
                clear_value: wgt::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            };
            let attachments = [Some(attachment)];
            let pass = RenderPassDescriptor {
                label: Some("codevar-clear"),
                extent: self.context.config().extent,
                sample_count: 1,
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            // SAFETY: the attachment view is alive for the whole pass and is
            // a colour target of the pass's only attachment slot.
            unsafe { self.encoder.begin_render_pass(&pass) }
                .map_err(|err| CompositorError::graphics("begin_render_pass", err))?;
            // SAFETY: a render pass is recording.
            unsafe { self.encoder.end_render_pass() };

            // SAFETY: a render pass was begun and ended on this encoder.
            let command_buffer = unsafe { self.encoder.end_encoding() }
                .map_err(|err| CompositorError::graphics("end_encoding", err))?;

            self.fence_value += 1;
            // SAFETY: the command buffer writes only to the surface texture,
            // which is listed as a surface texture so `submit` transitions it
            // to the present state; the fence belongs to this context.
            unsafe {
                self.context.queue().submit(
                    &[&command_buffer],
                    &[texture],
                    (self.context.fence(), self.fence_value),
                )
            }
            .map_err(|err| CompositorError::graphics("submit", err))?;
            // SAFETY: the browser backend executes submissions synchronously,
            // so the command buffer is free to be recycled immediately.
            unsafe { self.encoder.reset_all(iter::once(command_buffer)) };
            // SAFETY: the render pass that referenced the view has ended and
            // the view is referenced by nothing else.
            unsafe { device.destroy_texture_view(view) };

            self.phase = CompositorPhase::Submitted;
            Ok(())
        }

        /// Presents the submitted canvas texture and retires the frame.
        ///
        /// # Errors
        ///
        /// * [`CompositorError::FramePhase`] when no frame was submitted.
        /// * [`CompositorError::Graphics`] when presentation fails.
        pub fn end_frame(&mut self) -> Result<FrameCompletion, CompositorError> {
            if self.phase != CompositorPhase::Submitted {
                return Err(CompositorError::FramePhase {
                    operation: "end_frame",
                    expected: CompositorPhase::Submitted,
                    actual: self.phase,
                });
            }
            let texture = match self.pending.take() {
                Some(texture) => texture,
                None => {
                    return Err(CompositorError::Internal(
                        "end_frame ran without a submitted canvas texture",
                    ));
                }
            };
            // SAFETY: the texture was acquired from this surface, submitted
            // (which moved it to the present state) and is not referenced
            // elsewhere.
            unsafe {
                self.context
                    .queue()
                    .present(self.context.surface(), texture)
            }
            .map_err(|err| CompositorError::graphics("present", err))?;
            self.frame_index += 1;
            self.phase = CompositorPhase::Idle;
            Ok(FrameCompletion::Complete)
        }

        /// The number of frames fully presented.
        #[inline]
        #[must_use]
        pub const fn frame_index(&self) -> u64 {
            self.frame_index
        }

        /// Blocks until every submitted frame finished.
        ///
        /// # Errors
        ///
        /// [`CompositorError::Graphics`] when the queue is lost.
        pub fn wait_idle(&self) -> Result<(), CompositorError> {
            // SAFETY: nothing is submitted concurrently; `self` has exclusive
            // access to the context.
            unsafe { self.context.queue().wait_for_idle() }
                .map_err(|err| CompositorError::graphics("wait_for_idle", err))
        }
    }

    impl Drop for WgRenderer<'_> {
        fn drop(&mut self) {
            // An acquired but never presented texture must be returned to the
            // swapchain or the surface wedges on the next acquire.
            if let Some(texture) = self.pending.take() {
                // SAFETY: the texture was acquired from this surface and was
                // neither presented nor discarded.
                unsafe { self.context.surface().discard_texture(texture) };
            }
        }
    }

    impl GpuRenderer for WgRenderer<'_> {
        type Error = CompositorError;

        fn backend(&self) -> Backend {
            Backend::WebGpu
        }

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: renderer_features(),
                limits: GpuLimits::MINIMAL,
            }
        }

        fn begin_frame(&mut self, wait: Option<FrameCompletion>) -> Result<(), Self::Error> {
            WgRenderer::begin_frame(self, wait)
        }

        fn submit_frame(&mut self) -> Result<(), Self::Error> {
            WgRenderer::submit_frame(self)
        }

        fn end_frame(&mut self) -> Result<FrameCompletion, Self::Error> {
            WgRenderer::end_frame(self)
        }

        fn frame_index(&self) -> u64 {
            WgRenderer::frame_index(self)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::*;

    use crate::pipeline::WgContext;

    /// The WebGPU frame lifecycle (unavailable off the web).
    #[derive(Debug)]
    pub struct WgRenderer<'p> {
        /// Kept for parity with the wasm renderer, which borrows its context.
        context: &'p WgContext,
        /// Whether the subsystem accepts frames.
        state: CompositorState,
        /// Progress of the frame in flight.
        phase: CompositorPhase,
        /// Number of frames fully presented.
        frame_index: u64,
    }

    impl<'p> WgRenderer<'p> {
        /// Creates the subsystem in the [`CompositorState::Stopped`] state.
        ///
        /// # Errors
        ///
        /// Never; the placeholder always succeeds.
        pub fn new(context: &'p WgContext) -> Result<Self, CompositorError> {
            Ok(Self {
                context,
                state: CompositorState::Stopped,
                phase: CompositorPhase::Idle,
                frame_index: 0,
            })
        }

        /// The lifecycle state.
        #[inline]
        #[must_use]
        pub const fn state(&self) -> CompositorState {
            self.state
        }

        /// Progress of the frame in flight.
        #[inline]
        #[must_use]
        pub const fn phase(&self) -> CompositorPhase {
            self.phase
        }

        /// The context the renderer was created against.
        #[inline]
        #[must_use]
        pub const fn context(&self) -> &'p WgContext {
            self.context
        }

        /// Moves from [`CompositorState::Stopped`] to
        /// [`CompositorState::Running`].
        ///
        /// # Errors
        ///
        /// [`CompositorError::InvalidState`] when the subsystem is not
        /// stopped.
        pub fn start(&mut self) -> Result<(), CompositorError> {
            match self.state {
                CompositorState::Stopped => {
                    self.state = CompositorState::Running;
                    Ok(())
                }
                state => Err(CompositorError::InvalidState {
                    operation: "start",
                    state,
                }),
            }
        }

        /// Moves back to [`CompositorState::Stopped`].
        ///
        /// # Errors
        ///
        /// Never; the placeholder is always idle.
        pub fn stop(&mut self) -> Result<(), CompositorError> {
            self.state = CompositorState::Stopped;
            Ok(())
        }

        /// Always [`CompositorError::UnsupportedTarget`]: a frame only
        /// exists on a live browser device.
        ///
        /// # Errors
        ///
        /// Always [`CompositorError::UnsupportedTarget`].
        pub fn begin_frame(&mut self, _wait: Option<FrameCompletion>) -> Result<(), CompositorError> {
            Err(CompositorError::UnsupportedTarget)
        }

        /// Always [`CompositorError::UnsupportedTarget`].
        ///
        /// # Errors
        ///
        /// Always [`CompositorError::UnsupportedTarget`].
        pub fn submit_frame(&mut self) -> Result<(), CompositorError> {
            Err(CompositorError::UnsupportedTarget)
        }

        /// Always [`CompositorError::UnsupportedTarget`].
        ///
        /// # Errors
        ///
        /// Always [`CompositorError::UnsupportedTarget`].
        pub fn end_frame(&mut self) -> Result<FrameCompletion, CompositorError> {
            Err(CompositorError::UnsupportedTarget)
        }

        /// The number of frames fully presented.
        #[inline]
        #[must_use]
        pub const fn frame_index(&self) -> u64 {
            self.frame_index
        }
    }

    impl GpuRenderer for WgRenderer<'_> {
        type Error = CompositorError;

        fn backend(&self) -> Backend {
            Backend::WebGpu
        }

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: renderer_features(),
                limits: GpuLimits::MINIMAL,
            }
        }

        fn begin_frame(&mut self, wait: Option<FrameCompletion>) -> Result<(), Self::Error> {
            WgRenderer::begin_frame(self, wait)
        }

        fn submit_frame(&mut self) -> Result<(), Self::Error> {
            WgRenderer::submit_frame(self)
        }

        fn end_frame(&mut self) -> Result<FrameCompletion, Self::Error> {
            WgRenderer::end_frame(self)
        }

        fn frame_index(&self) -> u64 {
            WgRenderer::frame_index(self)
        }
    }
}

pub use imp::WgRenderer;

#[cfg(test)]
mod tests {
    //! The off-web stub's lifecycle, which needs no browser.

    use super::*;

    #[test]
    fn the_renderer_reports_webgpu_capabilities() {
        let capabilities = GpuCapabilities {
            backend: Backend::WebGpu,
            features: renderer_features(),
            limits: GpuLimits::MINIMAL,
        };
        assert!(capabilities.supports(GpuFeature::SrgbSurface));
        assert!(!capabilities.supports(GpuFeature::DmaBufExport));
    }
}
