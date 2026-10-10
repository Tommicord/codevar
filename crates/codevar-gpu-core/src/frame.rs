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

//! Frame execution: the layout pass, the pipe-driving phase, the mix
//! pass, completion export and the [`block_on`] executor.

use alloc::boxed::Box;
use alloc::vec::Vec;
use ash::vk;
use core::future::Future;
use core::task::{Context, Poll, Waker};

use crate::base::CompositorComm;
use crate::base::OffscreenFramebuffer;
use crate::compositor::{Compositor, PipeState};
use crate::mix::mix_push;
use crate::pipe_ctx::{PipeEnv, PipeOutcome};
use crate::pipeline::{OwnedFd, PipelineContext};
use crate::resources::{
    DEFAULT_CLEAR_COLOR, ImageTransition, export_sync_fd, record_image_transition, transition_source,
};
use crate::types::{CompositorError, CompositorMessage, CompositorPhase};

/// Records a clear-only rendering scope that defines one framebuffer's
/// contents after its transition into `COLOR_ATTACHMENT_OPTIMAL`.
///
/// The compositor clears unknown contents itself so correctness never
/// depends on a pipe cooperating with [`PipeCtx::begin_render`].
fn record_framebuffer_clear(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    framebuffer: &OffscreenFramebuffer,
) {
    let attachment = vk::RenderingAttachmentInfo::default()
        .image_view(framebuffer.view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: DEFAULT_CLEAR_COLOR,
            },
        });
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
    // SAFETY: the command buffer is recording, the caller transitioned
    // the image to `COLOR_ATTACHMENT_OPTIMAL` first, the view is
    // compatible with that layout, and the render area fits the image.
    unsafe {
        device.cmd_begin_rendering(command_buffer, &rendering_info);
        device.cmd_end_rendering(command_buffer);
    }
}

/// Submits one recorded command buffer with optional wait and signal
/// semaphores (the frame's GPU-to-GPU chain).
///
/// # Errors
///
/// * [`CompositorError::Submit`] — the queue rejected the submission.
fn submit_commands(
    context: &PipelineContext,
    command_buffer: vk::CommandBuffer,
    wait: &[vk::Semaphore],
    wait_stages: &[vk::PipelineStageFlags],
    signal: &[vk::Semaphore],
    fence: vk::Fence,
) -> Result<(), CompositorError> {
    debug_assert_eq!(wait.len(), wait_stages.len());
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(wait)
        .wait_dst_stage_mask(wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(signal);
    // SAFETY: all handles belong to this frame and are alive; wait
    // semaphores are fresh binary semaphores (or the frame's temporary
    // SYNC_FD import) and signal semaphores and the fence are
    // unsignaled, as required by `vkQueueSubmit`.
    unsafe {
        context
            .device()
            .queue_submit(context.queue(), core::slice::from_ref(&submit_info), fence)
    }
    .map_err(CompositorError::Submit)
}

impl<'p> Compositor<'p> {
    /// Drives the frame one step closer to completion.
    ///
    /// Called repeatedly — usually from an async executor through
    /// [`Compositor::composite_frame`] — until it resolves:
    ///
    /// * `Recording` → records and submits the layout pass.
    /// * `Pipes` → polls every pipe future with `cx`; [`Poll::Pending`]
    ///   means at least one future yielded, and the compositor is woken
    ///   through the waker that future stored.
    /// * `Mixing` → records and submits the mix pass; the frame is then
    ///   on the queue and [`Compositor::end_frame`] may export it.
    ///
    /// A failure discards the half-recorded frame (see
    /// `Compositor::discard_frame`) and reports the error. Calling
    /// this outside the driving phases reports
    /// [`CompositorError::FramePhase`] without touching the frame.
    ///
    /// # Errors
    ///
    /// Propagates the recording, pipe and submission errors of the
    /// phase it drives; the frame is discarded first.
    pub fn poll_composite(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), CompositorError>> {
        loop {
            match self.phase {
                CompositorPhase::Recording => {
                    if let Err(err) = self.record_and_submit_layout() {
                        return self.fail_frame(err);
                    }
                    self.phase = CompositorPhase::Pipes;
                }
                CompositorPhase::Pipes => match self.drive_pipes(cx) {
                    Ok(true) => self.phase = CompositorPhase::Mixing,
                    Ok(false) => return Poll::Pending,
                    Err(err) => return self.fail_frame(err),
                },
                CompositorPhase::Mixing => {
                    if let Err(err) = self.record_and_submit_mix() {
                        return self.fail_frame(err);
                    }
                    self.phase = CompositorPhase::Submitted;
                    return Poll::Ready(Ok(()));
                }
                CompositorPhase::Idle | CompositorPhase::Submitted => {
                    return Poll::Ready(Err(CompositorError::FramePhase {
                        operation: "poll_composite",
                        expected: CompositorPhase::Recording,
                        actual: self.phase,
                    }));
                }
            }
        }
    }

    /// Discards the frame in progress and wraps `error` as its result.
    fn fail_frame(&mut self, error: CompositorError) -> Poll<Result<(), CompositorError>> {
        self.discard_frame();
        Poll::Ready(Err(error))
    }

    /// Records the layout pass: parks every framebuffer in the layout
    /// the pipes (or a frozen mix) read or write it in, defines the
    /// contents of unknown targets and submits the pass as the head of
    /// the frame's semaphore chain.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the frame resources or the
    ///   chain semaphores are missing.
    /// * [`CompositorError::CommandRecord`] — the layout command buffer
    ///   could not be opened or closed.
    /// * [`CompositorError::Submit`] — the submission was rejected.
    fn record_and_submit_layout(&mut self) -> Result<(), CompositorError> {
        let context = self.context;
        let device = context.device();
        let (layout_command_buffer, chain_first) = match self.start_resources.as_ref() {
            Some(resources) => (resources.layout_command_buffer, resources.chain_sems[0]),
            None => {
                return Err(CompositorError::Internal(
                    "a frame was driven without start resources",
                ));
            }
        };
        if chain_first == vk::Semaphore::null() {
            return Err(CompositorError::Internal(
                "the chain semaphore pool is incomplete",
            ));
        }
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool was reset when this frame began, the buffer
        // was allocated from it, and `&mut self` excludes any other
        // recording.
        unsafe { device.begin_command_buffer(layout_command_buffer, &begin_info) }
            .map_err(CompositorError::CommandRecord)?;

        let visible = self.visible;
        let framebuffer_count = self.framebuffers.len();
        // Park every framebuffer in the layout the rest of the frame
        // reads or writes it in.
        for index in 0..framebuffer_count {
            let framebuffer = self.framebuffers[index];
            let old_layout = self
                .layout_of(framebuffer.image)
                .unwrap_or(vk::ImageLayout::UNDEFINED);
            if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                continue;
            }
            let (src_stage, src_access) = transition_source(old_layout);
            record_image_transition(
                device,
                layout_command_buffer,
                &ImageTransition::new(
                    framebuffer.image,
                    old_layout,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_stage,
                    src_access,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
            );
        }
        // Define the contents of every framebuffer whose tracked layout
        // is unknown: a pipe loading them — and a frozen frame, which
        // runs no pipe at all — would otherwise read garbage.
        for (index, framebuffer) in self.framebuffers.iter().enumerate() {
            if index < self.undefined_mask.len() && !self.undefined_mask[index] {
                continue;
            }
            record_framebuffer_clear(device, layout_command_buffer, framebuffer);
        }
        if !visible {
            // Pipes are skipped while hidden: the mix pass samples the
            // framebuffers directly, so park them in the sampled layout.
            for framebuffer in self.framebuffers.iter() {
                record_image_transition(
                    device,
                    layout_command_buffer,
                    &ImageTransition::new(
                        framebuffer.image,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::AccessFlags::SHADER_READ,
                    ),
                );
            }
        }
        let end_layout = if visible {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        // Collect images first to avoid borrow checker issues
        let images: Vec<vk::Image> = self
            .framebuffers
            .iter()
            .map(|fb| fb.image)
            .collect();
        for image in images {
            self.set_layout(image, end_layout);
        }

        // SAFETY: the buffer is still recording; everything recorded
        // above targets images of this frame.
        unsafe { device.end_command_buffer(layout_command_buffer) }
            .map_err(CompositorError::CommandRecord)?;
        let import_wait_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let (wait, wait_stages): (&[vk::Semaphore], &[vk::PipelineStageFlags]) =
            match self.frame_sync.as_ref() {
                Some(frame_sync) if frame_sync.import_sem != vk::Semaphore::null() => (
                    core::slice::from_ref(&frame_sync.import_sem),
                    core::slice::from_ref(&import_wait_stage),
                ),
                _ => (&[], &[]),
            };
        let signal = [chain_first];
        submit_commands(
            context,
            layout_command_buffer,
            wait,
            wait_stages,
            &signal,
            vk::Fence::null(),
        )
    }

    /// Polls the recording pipe futures and submits every resolved
    /// prefix of the chain in queue order.
    ///
    /// Returns `true` once all pipes of the frame are on the queue.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Pipe`] — a future failed; the index is the
    ///   pipe's position in the queue.
    /// * [`CompositorError::CommandRecord`] — a pipe command buffer
    ///   could not be opened or closed.
    /// * [`CompositorError::Submit`] — the submission was rejected.
    /// * [`CompositorError::Internal`] — the queue diverged from the
    ///   frame slots or the frame resources are missing.
    fn drive_pipes(&mut self, cx: &mut Context<'_>) -> Result<bool, CompositorError> {
        let context = self.context;
        let device = context.device();
        for index in 0..self.active.len() {
            if self.active[index].state != PipeState::Recording {
                continue;
            }
            let (entry, renderer) = match self.queue.pipe(index) {
                Some(pipe) => (pipe.entry(), pipe.renderer()),
                None => {
                    return Err(CompositorError::Internal(
                        "the pipe queue diverged from the frame slots",
                    ));
                }
            };
            if self.active[index].future.is_none() {
                let begin_info =
                    vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                let command_buffer = self.active[index].command_buffer;
                // SAFETY: the pool was reset when this frame began and
                // no other code records this slot's buffer.
                unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
                    .map_err(CompositorError::CommandRecord)?;
                let env = PipeEnv {
                    renderer,
                    ctx: core::ptr::from_mut(&mut self.active[index].ctx),
                };
                self.active[index].future = Some(entry(env));
            }
            let poll_result = {
                let active = &mut self.active[index];
                let Some(future) = active.future.as_mut() else {
                    return Err(CompositorError::Internal("a recording pipe has no future"));
                };
                future.as_mut().poll(cx)
            };
            match poll_result {
                Poll::Pending => {}
                Poll::Ready(Ok(outcome)) => {
                    // Release the future before closing the buffer it
                    // recorded: it borrows the renderer and the context.
                    let active = &mut self.active[index];
                    active.future = None;
                    // SAFETY: the future recorded into this buffer and
                    // has just been dropped, so recording is finished.
                    unsafe { device.end_command_buffer(active.command_buffer) }
                        .map_err(CompositorError::CommandRecord)?;
                    active.state = PipeState::Ready;
                    if outcome == PipeOutcome::Remove {
                        self.queue.mark_removed(index);
                    }
                }
                Poll::Ready(Err(source)) => {
                    return Err(CompositorError::Pipe {
                        index,
                        source: Box::new(source),
                    });
                }
            }
        }
        // Submit the resolved prefix: pipe *k* waits on `chain[k]` (the
        // layout pass signals `chain[0]`) and signals `chain[k + 1]`.
        let resources = self
            .start_resources
            .as_ref()
            .ok_or(CompositorError::Internal(
                "a frame was driven without start resources",
            ))?;
        let mut index = 0;
        while index < self.active.len() && self.active[index].state == PipeState::Ready {
            let wait = resources.chain_sems[index];
            let signal = resources.chain_sems[index + 1];
            if wait == vk::Semaphore::null() || signal == vk::Semaphore::null() {
                return Err(CompositorError::Internal(
                    "the chain semaphore pool is incomplete",
                ));
            }
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            submit_commands(
                context,
                self.active[index].command_buffer,
                core::slice::from_ref(&wait),
                &wait_stages,
                core::slice::from_ref(&signal),
                vk::Fence::null(),
            )?;
            self.active[index].state = PipeState::Submitted;
            index += 1;
        }
        Ok(self
            .active
            .iter()
            .all(|slot| slot.state == PipeState::Submitted))
    }

    /// Rewrites the mix descriptor set for this frame.
    ///
    /// The write is safe because exactly one frame is in flight and
    /// [`Compositor::begin_frame`] waited for the previous submission's
    /// fence, so the set is never in use.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the frame resources are missing.
    fn update_mix_descriptors(&self) -> Result<(), CompositorError> {
        let resources = self
            .start_resources
            .as_ref()
            .ok_or(CompositorError::Internal(
                "a frame was driven without start resources",
            ))?;
        let count = self.framebuffers.len();
        if count == 0 {
            return Ok(());
        }
        let sampler = resources.mix.sampler;
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(count);
        for framebuffer in self.framebuffers.iter() {
            image_infos.push(
                vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(framebuffer.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            );
        }
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(resources.mix.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)];
        // SAFETY: the set was allocated from this pool against a layout
        // declaring combined image samplers at binding 0, every view is a
        // valid color target collected for this frame, and the set is not
        // referenced by pending commands.
        unsafe {
            self.context
                .device()
                .update_descriptor_sets(&writes, &[])
        };
        Ok(())
    }

    /// Records and submits the mix pass: samples every framebuffer into
    /// the presentation target and hands it off in `GENERAL` for the
    /// external compositor.
    ///
    /// With no framebuffers the pass only clears the target, which is
    /// what the zero-pipe path of the launcher relies on.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the frame resources, the chain
    ///   semaphore or the frame's synchronization are missing.
    /// * [`CompositorError::CommandRecord`] — the mix command buffer
    ///   could not be opened or closed.
    /// * [`CompositorError::Submit`] — the submission was rejected.
    fn record_and_submit_mix(&mut self) -> Result<(), CompositorError> {
        let context = self.context;
        let device = context.device();
        let (mix_command_buffer, frame_fence) = match self.start_resources.as_ref() {
            Some(resources) => (resources.mix_command_buffer, resources.frame_fence),
            None => {
                return Err(CompositorError::Internal(
                    "a frame was driven without start resources",
                ));
            }
        };
        let chain_wait = self
            .start_resources
            .as_ref()
            .map_or(vk::Semaphore::null(), |resources| {
                resources.chain_sems[self.active.len()]
            });
        if chain_wait == vk::Semaphore::null() {
            return Err(CompositorError::Internal(
                "the chain semaphore pool is incomplete",
            ));
        }
        let present_sem = match self.frame_sync.as_ref() {
            Some(frame_sync) => frame_sync.present_sem,
            None => {
                return Err(CompositorError::Internal(
                    "the frame was begun without synchronization",
                ));
            }
        };
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool was reset when this frame began and no other
        // code records this buffer.
        unsafe { device.begin_command_buffer(mix_command_buffer, &begin_info) }
            .map_err(CompositorError::CommandRecord)?;

        let count = self.framebuffers.len();
        if self.visible {
            // The pipes left their targets in the attachment layout; the
            // mix shader samples them. A hidden frame skips this (the
            // layout pass already parked them in the sampled layout).
            for framebuffer in self.framebuffers.iter() {
                record_image_transition(
                    device,
                    mix_command_buffer,
                    &ImageTransition::new(
                        framebuffer.image,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::AccessFlags::SHADER_READ,
                    ),
                );
            }
        }
        // The pass clears the presentation target below, so its previous
        // contents are discarded; the closing transition leaves it in
        // `GENERAL`, the conservative state for non-Vulkan consumers.
        record_image_transition(
            device,
            mix_command_buffer,
            &ImageTransition::new(
                context.image(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
        );
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: context.width(),
                height: context.height(),
            },
        };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(context.image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: DEFAULT_CLEAR_COLOR,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&attachment));
        // SAFETY: the command buffer is recording, the presentation view
        // matches the `COLOR_ATTACHMENT_OPTIMAL` layout recorded above,
        // and dynamic rendering needs no render pass object.
        unsafe { device.cmd_begin_rendering(mix_command_buffer, &rendering_info) };

        if count > 0 {
            self.update_mix_descriptors()?;
            let resources = self
                .start_resources
                .as_ref()
                .ok_or(CompositorError::Internal(
                    "a frame was driven without start resources",
                ))?;
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: context.width() as f32,
                height: context.height() as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            // SAFETY: the rendering scope is open, the pipeline declares
            // viewport and scissor dynamic, and every handle belongs to
            // this frame.
            unsafe {
                device.cmd_set_viewport(mix_command_buffer, 0, core::slice::from_ref(&viewport));
                device.cmd_set_scissor(mix_command_buffer, 0, core::slice::from_ref(&area));
                device.cmd_bind_pipeline(
                    mix_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    resources.mix.pipeline,
                );
                let sets = [resources.mix.descriptor_set];
                device.cmd_bind_descriptor_sets(
                    mix_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    resources.mix.pipeline_layout,
                    0,
                    &sets,
                    &[],
                );
            }
            let push = mix_push(&self.weights);
            let mut push_bytes: Vec<u8> = Vec::with_capacity(push.byte_size());
            push_bytes.extend_from_slice(&push.count.to_le_bytes());
            push_bytes.extend_from_slice(&[0u8; 12]); // padding
            for weight in &push.weights {
                push_bytes.extend_from_slice(&weight.to_le_bytes());
            }
            // SAFETY: the push range declared when the pipeline layout
            // was created is large enough for the actual data, and the
            // bytes outlive the call.
            unsafe {
                device.cmd_push_constants(
                    mix_command_buffer,
                    resources.mix.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &push_bytes,
                );
                // Fullscreen triangle generated by `mix.vert`.
                device.cmd_draw(mix_command_buffer, 3, 1, 0, 0);
            }
        }
        // SAFETY: the rendering scope opened above has not been ended.
        unsafe { device.cmd_end_rendering(mix_command_buffer) };
        record_image_transition(
            device,
            mix_command_buffer,
            &ImageTransition::new(
                context.image(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
        );
        // SAFETY: the buffer is still recording; everything recorded
        // above targets images of this frame.
        unsafe { device.end_command_buffer(mix_command_buffer) }.map_err(CompositorError::CommandRecord)?;

        let wait = [chain_wait];
        let wait_stages =
            [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::FRAGMENT_SHADER];
        let signal = [present_sem];
        submit_commands(
            context,
            mix_command_buffer,
            &wait,
            &wait_stages,
            &signal,
            frame_fence,
        )?;
        self.in_flight = true;
        Ok(())
    }
}

impl<'p> Compositor<'p> {
    /// Finishes the frame: exports GPU completion of the mix pass as a
    /// `sync_file` descriptor for the presentation layer.
    ///
    /// The frame's semaphores stay alive until the next
    /// [`Compositor::begin_frame`] waits for the frame fence; the
    /// exported descriptor remains valid after they are destroyed.
    ///
    /// Calling this while a frame is still being recorded discards that
    /// frame first (so the compositor never wedges); calling it when no
    /// frame was submitted only reports the phase error.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::FramePhase`] — no frame is on the queue.
    /// * [`CompositorError::SemaphoreExport`] — the driver refused to
    ///   export the signal semaphore; the frame is discarded first.
    pub fn end_frame(&mut self) -> Result<OwnedFd, CompositorError> {
        if self.phase != CompositorPhase::Submitted {
            if self.phase != CompositorPhase::Idle {
                self.discard_frame();
            }
            return Err(CompositorError::FramePhase {
                operation: "end_frame",
                expected: CompositorPhase::Submitted,
                actual: self.phase,
            });
        }
        let context = self.context;
        let present_sem = match self.frame_sync.as_ref() {
            Some(frame_sync) => frame_sync.present_sem,
            None => {
                self.discard_frame();
                return Err(CompositorError::Internal(
                    "the frame was submitted without synchronization",
                ));
            }
        };
        match export_sync_fd(context, present_sem) {
            Ok(sync_file) => {
                // The presentation side may not be listening (it may
                // have no endpoint yet); the export above succeeded.
                let _ = self
                    .comm
                    .send_message(CompositorMessage::FrameComposited {
                        frame_index: self.frame_index,
                    });
                self.frame_index += 1;
                self.phase = CompositorPhase::Idle;
                Ok(sync_file)
            }
            Err(err) => {
                self.discard_frame();
                Err(err)
            }
        }
    }

    /// Returns to the idle phase after a failed frame: waits for the
    /// device, resets the chain fence, command pool and semaphores and
    /// forgets the layouts of every image the abandoned frame touched.
    ///
    /// Cleanup is best-effort — the original error is reported by the
    /// caller, and a resource that cannot be reset leaves the compositor
    /// idle with the frame's submissions no longer in flight.
    fn discard_frame(&mut self) {
        let context = self.context;
        let device = context.device();
        // SAFETY: `&mut self` excludes concurrent use; a failed wait
        // (device lost) is tolerated because the state below must be
        // reset regardless.
        let _ = unsafe { device.device_wait_idle() };
        if let Some(resources) = self.start_resources.as_ref() {
            let fence = resources.frame_fence;
            if fence != vk::Fence::null() {
                // SAFETY: the device is idle, so the fence has no pending
                // signal operation; resetting a signaled fence returns
                // it to the unsignaled state the next submission needs.
                let _ = unsafe { device.reset_fences(core::slice::from_ref(&fence)) };
            }
            // SAFETY: the device is idle, so resetting the pool cannot
            // discard in-flight commands; it releases the buffers the
            // abandoned frame recorded.
            let _ = unsafe {
                device.reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty())
            };
        }
        // Futures first (they borrow the renderers and the contexts),
        // then the frame's synchronization and layout tracking.
        self.active.clear();
        self.frame_sync = None;
        self.in_flight = false;
        self.layouts.clear();
        self.undefined_mask.clear();
        self.phase = CompositorPhase::Idle;
        self.reset_chain_semaphores();
    }

    /// Destroys and recreates the chain semaphores after a discarded
    /// frame.
    ///
    /// A binary semaphore left signaled by the abandoned submission
    /// cannot be signaled again, so the whole chain is rebuilt; a
    /// failure leaves slots null and the frame loop rejects the
    /// incomplete pool with [`CompositorError::Internal`] instead of
    /// handing a null handle to the driver.
    fn reset_chain_semaphores(&mut self) {
        let Some(resources) = self.start_resources.as_mut() else {
            return;
        };
        let device = resources.context.device();
        for semaphore in resources.chain_sems.iter_mut() {
            if *semaphore != vk::Semaphore::null() {
                // SAFETY: the device is idle, so no submission still
                // references a chain semaphore, and each handle was
                // created on this device.
                unsafe { device.destroy_semaphore(*semaphore, None) };
                *semaphore = vk::Semaphore::null();
            }
        }
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        for semaphore in resources.chain_sems.iter_mut() {
            // SAFETY: a plain, well-formed create info; a failure leaves
            // the slot null, which the frame loop detects.
            *semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap_or(vk::Semaphore::null());
        }
    }

    /// Runs one whole frame asynchronously: begins it, drives every
    /// phase to submission and exports completion.
    ///
    /// `wait_sync_file` is the optional `sync_file` to wait on before
    /// rendering (ownership is consumed); the returned descriptor
    /// signals when the mix pass finished.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Compositor::begin_frame`],
    /// [`Compositor::poll_composite`] and [`Compositor::end_frame`].
    pub async fn composite_frame(
        &mut self,
        wait_sync_file: Option<OwnedFd>,
    ) -> Result<OwnedFd, CompositorError> {
        self.begin_frame(wait_sync_file)?;
        core::future::poll_fn(|cx| self.poll_composite(cx)).await?;
        self.end_frame()
    }

    /// Drives one frame to submission without exporting it, leaving the
    /// compositor in the `Submitted` phase for a custom
    /// [`Compositor::end_frame`].
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Compositor::begin_frame`] and
    /// [`Compositor::poll_composite`].
    pub async fn draw_frame(&mut self, wait_sync_file: Option<OwnedFd>) -> Result<(), CompositorError> {
        self.begin_frame(wait_sync_file)?;
        core::future::poll_fn(|cx| self.poll_composite(cx)).await
    }
}

/// Drives `future` to completion on the current thread.
///
/// A minimal single-threaded executor: the future is polled with a
/// no-op waker and every [`Poll::Pending`] is retried after a short
/// sleep, so pipe futures that wait on the communication channels make
/// progress through re-polling alone. Use it from synchronous code (the
/// display loop); executors should await
/// [`Compositor::composite_frame`] directly instead.
#[must_use]
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                let delay = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 1_000_000,
                };
                // SAFETY: `delay` is a valid timespec, the remaining
                // time output is null (unused), and an `EINTR` just
                // retries on the next loop iteration.
                unsafe { libc::nanosleep(&delay, core::ptr::null_mut()) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The synchronous single-threaded executor.

    use super::*;

    #[test]
    fn block_on_drives_pending_futures_to_completion() {
        let mut polls = 0_u32;
        let future = core::future::poll_fn(|_cx| {
            polls += 1;
            if polls < 3 {
                Poll::<u32>::Pending
            } else {
                Poll::Ready(polls)
            }
        });
        assert_eq!(block_on(future), 3);
    }
}
