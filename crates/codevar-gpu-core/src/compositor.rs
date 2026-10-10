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

//! The [`Compositor`]: lifecycle, pipe registration and the frame begin
//! that retires the previous submission and snapshots the frame.

use alloc::rc::Rc;
use alloc::vec::Vec;
use ash::vk;

use crate::comm::{CompositorComm, CompositorEndpoint, RendererEndpoint, comm_pair};
use crate::offscreen::{OffscreenFramebuffer, PipeOffscreen, sanitize_weight};
use crate::pipe::{PipeHandoff, PipeQueue, PipeSource};
use crate::pipe_ctx::{PipeCtx, PipeFuture};
use crate::pipeline::{OwnedFd, PipelineContext};
use crate::resources::{FrameSync, StartResources, create_binary_semaphore, import_sync_fd};
use crate::types::{CompositorError, CompositorMessage, CompositorPhase, CompositorState};

pub use crate::frame::block_on;

/// Per-frame lifecycle of one pipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipeState {
    /// Its future has not resolved yet.
    Recording,
    /// Resolved and its command buffer was closed; waiting for its turn
    /// in the submission chain.
    Ready,
    /// On the queue; nothing more is polled for it this frame.
    Submitted,
}

/// One pipe's state for the frame in progress.
///
/// Fields drop in declaration order, so the future — which borrows the
/// context and the renderer — is released before anything it references.
pub(crate) struct ActivePipe {
    /// Future built on first poll; `None` after it resolved.
    pub(crate) future: Option<PipeFuture<'static>>,
    /// Frame context handed to the future.
    pub(crate) ctx: PipeCtx,
    /// Command buffer recorded by the future (the queue slot's buffer).
    pub(crate) command_buffer: vk::CommandBuffer,
    /// Progress through the frame.
    pub(crate) state: PipeState,
}

/// Compositor-based renderer: drives pipes, then mixes their offscreen
/// framebuffers into the presentation target.
///
/// A compositor is created stopped; call [`Compositor::start`], register
/// pipes with [`Compositor::add_pipe`] and drive frames through
/// [`Compositor::begin_frame`] / [`Compositor::poll_composite`] /
/// [`Compositor::end_frame`] (or the [`Compositor::draw_frame`] helper).
/// Exactly one frame may be in flight at a time.
///
/// The compositor borrows its [`PipelineContext`] for `'p` and must be
/// dropped before it: dropping waits for the device and releases the
/// renderers of every registered pipe.
pub struct Compositor<'p> {
    /// Device-level state borrowed for the compositor's lifetime.
    pub(crate) context: &'p PipelineContext,
    /// Lifecycle state.
    pub(crate) state: CompositorState,
    /// Phase of the frame in progress.
    pub(crate) phase: CompositorPhase,
    /// Pipes of the frame in progress; dropped before `queue` so their
    /// futures release the renderers they borrow.
    pub(crate) active: Vec<ActivePipe>,
    /// Registered pipes in mix order.
    pub(crate) queue: PipeQueue,
    /// Channels to the presentation side.
    pub(crate) comm: CompositorEndpoint,
    /// Framebuffer snapshot of the current frame.
    pub(crate) framebuffers: Vec<OffscreenFramebuffer>,
    /// First mix slot of each queued pipe (see [`PipeCtx::base`]).
    pub(crate) bases: Vec<usize>,
    /// Per-slot mix weights, sanitized by the collector.
    pub(crate) weights: Vec<f32>,
    /// Whether the snapshot must be rebuilt at the next `begin_frame`.
    pub(crate) framebuffers_dirty: bool,
    /// Whether the pipes are driven (see [`CompositorMessage::VisibilityChanged`]).
    pub(crate) visible: bool,
    /// Visibility change queued for the next `begin_frame`.
    pub(crate) pending_visible: Option<bool>,
    /// Tracked layout of every composited image; empty means "contents
    /// unknown" (the next layout pass discards them).
    pub(crate) layouts: Vec<(vk::Image, vk::ImageLayout)>,
    /// Bitmask tracking which framebuffers still hold `UNDEFINED` content.
    pub(crate) undefined_mask: Vec<bool>,
    /// Frames whose `sync_file` was exported so far.
    pub(crate) frame_index: u64,
    /// Whether the frame fence still has to be waited on.
    pub(crate) in_flight: bool,
    /// Frame synchronization of the frame in progress.
    pub(crate) frame_sync: Option<FrameSync<'p>>,
    /// Resources allocated by [`Compositor::start`].
    pub(crate) start_resources: Option<StartResources<'p>>,
}

impl<'p> Compositor<'p> {
    /// Creates a stopped compositor for `context` with no pipes.
    ///
    /// Fetch the presentation-side endpoint with
    /// [`Compositor::renderer_endpoint`] before handing frames out.
    #[must_use]
    pub fn new(context: &'p PipelineContext) -> Self {
        let (endpoint, _) = comm_pair();
        Self {
            context,
            state: CompositorState::Stopped,
            phase: CompositorPhase::Idle,
            active: Vec::new(),
            queue: PipeQueue::new(),
            comm: endpoint,
            framebuffers: Vec::new(),
            bases: Vec::new(),
            weights: Vec::new(),
            framebuffers_dirty: true,
            visible: true,
            pending_visible: None,
            layouts: Vec::new(),
            undefined_mask: Vec::new(),
            frame_index: 0,
            in_flight: false,
            frame_sync: None,
            start_resources: None,
        }
    }

    /// Current lifecycle state.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> CompositorState {
        self.state
    }

    /// Current frame phase (see [`CompositorPhase`]).
    #[inline]
    #[must_use]
    pub const fn phase(&self) -> CompositorPhase {
        self.phase
    }

    /// Number of frames whose completion `sync_file` was exported.
    #[inline]
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Whether the pipes are currently driven each frame.
    #[inline]
    #[must_use]
    pub const fn visible(&self) -> bool {
        self.visible
    }

    /// Requests a visibility change for the next frame.
    ///
    /// The change is applied by [`Compositor::begin_frame`] so a frame
    /// already in progress never mixes a half-recorded state.
    pub fn set_visible(&mut self, visible: bool) {
        self.pending_visible = Some(visible);
    }

    /// The compositor's own endpoint (messages and retired pipes flow
    /// out through it, registration flows in).
    #[inline]
    #[must_use]
    pub fn comm(&self) -> &CompositorEndpoint {
        &self.comm
    }

    /// A fresh endpoint for the presentation side of the same channels.
    #[must_use]
    pub fn renderer_endpoint(&self) -> RendererEndpoint {
        RendererEndpoint {
            hub: Rc::clone(&self.comm.hub),
        }
    }

    /// Number of registered pipes.
    #[must_use]
    pub fn pipe_count(&self) -> usize {
        self.queue.len()
    }

    /// The pipe registry (order equals mix order).
    #[must_use]
    pub fn queue(&self) -> &PipeQueue {
        &self.queue
    }

    /// Framebuffer snapshot collected for the current frame.
    #[must_use]
    pub fn framebuffers(&self) -> &[OffscreenFramebuffer] {
        &self.framebuffers
    }

    /// Marks the framebuffer snapshot stale; it is rebuilt by the next
    /// [`Compositor::begin_frame`].
    pub fn invalidate_framebuffers(&mut self) {
        self.framebuffers_dirty = true;
    }

    /// Allocates the frame resources and enters the running state.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   stopped.
    /// * [`CompositorError::CommandAllocation`], [`CompositorError::SemaphoreCreate`],
    ///   [`CompositorError::FenceCreate`] — frame resources could not be
    ///   created; everything allocated first is released again.
    pub fn start(&mut self) -> Result<(), CompositorError> {
        if self.state != CompositorState::Stopped {
            return Err(CompositorError::InvalidState {
                operation: "start",
                state: self.state,
            });
        }
        // Start with minimal resources (0 pipes, 8 framebuffers as initial capacity)
        self.start_resources = Some(StartResources::create(self.context, 0, 8)?);
        self.state = CompositorState::Running;
        self.phase = CompositorPhase::Idle;
        self.framebuffers_dirty = true;
        Ok(())
    }

    /// Suspends frame operations, keeping pipes and resources.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   running.
    /// * [`CompositorError::FramePhase`] — a frame is in progress; finish
    ///   it first (or stop the compositor).
    pub fn pause(&mut self) -> Result<(), CompositorError> {
        if self.state != CompositorState::Running {
            return Err(CompositorError::InvalidState {
                operation: "pause",
                state: self.state,
            });
        }
        self.require_phase(CompositorPhase::Idle, "pause")?;
        self.state = CompositorState::Paused;
        Ok(())
    }

    /// Resumes a paused compositor.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   paused.
    pub fn resume(&mut self) -> Result<(), CompositorError> {
        if self.state != CompositorState::Paused {
            return Err(CompositorError::InvalidState {
                operation: "resume",
                state: self.state,
            });
        }
        self.state = CompositorState::Running;
        Ok(())
    }

    /// Waits for the device, releases the frame resources and returns to
    /// the stopped state.
    ///
    /// Registered pipes are kept (and dropped with the compositor). A
    /// frame that was begun but not finished is discarded.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is already
    ///   stopped.
    pub fn stop(&mut self) -> Result<(), CompositorError> {
        if self.state == CompositorState::Stopped {
            return Err(CompositorError::InvalidState {
                operation: "stop",
                state: self.state,
            });
        }
        self.quiesce();
        self.start_resources = None;
        self.state = CompositorState::Stopped;
        Ok(())
    }

    /// Checks that the frame is in `expected` phase.
    fn require_phase(
        &self,
        expected: CompositorPhase,
        operation: &'static str,
    ) -> Result<(), CompositorError> {
        if self.phase == expected {
            Ok(())
        } else {
            Err(CompositorError::FramePhase {
                operation,
                expected,
                actual: self.phase,
            })
        }
    }

    /// Waits for the device (ignoring device loss), retires the frame's
    /// synchronization, discards any recorded frame and resets the
    /// command pool.
    fn quiesce(&mut self) {
        let context = self.context;
        // SAFETY: the caller has exclusive access, so no frame can be
        // recording or submitting concurrently; a failed wait (device
        // lost) is tolerated because teardown must still run.
        unsafe {
            let _ = context.device().device_wait_idle();
        }
        self.frame_sync = None;
        self.in_flight = false;
        self.active.clear();
        if self.start_resources.is_some() {
            // The device is idle, so resetting the pool cannot discard
            // in-flight commands.
            // SAFETY: the pool belongs to `context`.
            unsafe {
                let _ = context
                    .device()
                    .reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty());
            }
        }
        self.phase = CompositorPhase::Idle;
    }

    /// Waits for the frame fence, retires the frame's semaphores and
    /// resets the command pool for a fresh frame.
    fn retire_previous_frame(&mut self) -> Result<(), CompositorError> {
        let context = self.context;
        let device = context.device();
        let fence = self
            .start_resources
            .as_ref()
            .map(|resources| resources.frame_fence)
            .ok_or(CompositorError::Internal(
                "the compositor is running without frame resources",
            ))?;
        if self.in_flight {
            // SAFETY: the fence was created by `StartResources::create`,
            // `&mut self` excludes concurrent use, and `u64::MAX` means
            // the only failure is device loss (which is reported).
            unsafe { device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX) }
                .map_err(CompositorError::FenceWait)?;
            // SAFETY: the fence just signaled, so it is in the
            // signaled state and may be reset.
            unsafe { device.reset_fences(core::slice::from_ref(&fence)) }
                .map_err(CompositorError::FenceWait)?;
            self.in_flight = false;
        }
        // The frame completed (or never reached the queue), so its
        // semaphores are no longer referenced: drop them.
        self.frame_sync = None;
        // SAFETY: no submission is pending — either the fence above was
        // waited on, or the frame was discarded after `device_wait_idle`.
        unsafe { device.reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty()) }
            .map_err(CompositorError::CommandPoolReset)
    }

    /// Creates this frame's semaphores: an optional wait semaphore
    /// importing `wait_sync_file`, plus the exportable signal semaphore.
    fn create_frame_sync(&self, wait_sync_file: Option<OwnedFd>) -> Result<FrameSync<'p>, CompositorError> {
        let device = self.context.device();
        let import_sem = if let Some(sync_file) = wait_sync_file {
            let semaphore = create_binary_semaphore(device, false)?;
            match import_sync_fd(self.context, semaphore, sync_file) {
                Ok(()) => semaphore,
                Err(err) => {
                    // SAFETY: the semaphore was just created and never
                    // submitted.
                    unsafe { device.destroy_semaphore(semaphore, None) };
                    return Err(err);
                }
            }
        } else {
            vk::Semaphore::null()
        };
        let present_sem = match create_binary_semaphore(device, true) {
            Ok(semaphore) => semaphore,
            Err(err) => {
                if import_sem != vk::Semaphore::null() {
                    // SAFETY: created above, never submitted.
                    unsafe { device.destroy_semaphore(import_sem, None) };
                }
                return Err(err);
            }
        };
        Ok(FrameSync {
            device,
            import_sem,
            present_sem,
        })
    }
}

impl<'p> Compositor<'p> {
    /// Registers `renderer` as a pipe in the next frame.
    ///
    /// The compositor takes ownership of `renderer`; it is released when
    /// the pipe is removed (see [`Compositor::remove_pipe`]) or when the
    /// compositor is dropped.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::FramePhase`] — a frame is in progress; the
    ///   queue may only grow between frames so slot indices stay stable.
    pub fn add_pipe<T: PipeSource + PipeOffscreen>(
        &mut self,
        renderer: T,
    ) -> Result<usize, CompositorError> {
        self.add_handoff(PipeHandoff::from_renderer(renderer))
    }

    /// Registers a pre-built [`PipeHandoff`]; see [`Compositor::add_pipe`].
    ///
    /// # Errors
    ///
    /// * [`CompositorError::FramePhase`] — a frame is in progress.
    pub fn add_handoff(&mut self, handoff: PipeHandoff) -> Result<usize, CompositorError> {
        if self.phase != CompositorPhase::Idle {
            return Err(CompositorError::FramePhase {
                operation: "add_pipe",
                expected: CompositorPhase::Idle,
                actual: self.phase,
            });
        }
        let index = self.queue.push(handoff);
        self.framebuffers_dirty = true;
        Ok(index)
    }

    /// Marks the pipe at `index` for removal; its renderer is released
    /// at the next [`Compositor::begin_frame`], once the frame that
    /// still referenced it has retired.
    ///
    /// Returns `false` when no pipe is registered at that index or it
    /// was already marked.
    pub fn remove_pipe(&mut self, index: usize) -> bool {
        self.queue.mark_removed(index)
    }

    /// The layout currently tracked for `image` (`None` = unknown, i.e.
    /// the next transition discards its contents).
    pub(crate) fn layout_of(&self, image: vk::Image) -> Option<vk::ImageLayout> {
        self.layouts
            .iter()
            .find(|(tracked, _)| *tracked == image)
            .map(|(_, layout)| *layout)
    }

    /// Records the layout an image ends this frame in.
    pub(crate) fn set_layout(&mut self, image: vk::Image, layout: vk::ImageLayout) {
        if let Some(entry) = self
            .layouts
            .iter_mut()
            .find(|(tracked, _)| *tracked == image)
        {
            entry.1 = layout;
        } else {
            self.layouts.push((image, layout));
        }
    }

    /// Applies the presentation side's messages, registers the pipes it
    /// queued, rebuilds the framebuffer snapshot when needed and applies
    /// pending weight and visibility changes.
    fn drain_comm(&mut self) {
        let mut weight_updates: Vec<(usize, f32)> = Vec::new();
        while let Some(message) = self.comm.try_recv_message() {
            match message {
                CompositorMessage::FramebuffersChanged => self.framebuffers_dirty = true,
                CompositorMessage::MixWeightChanged { slot, weight } => {
                    weight_updates.push((slot, weight));
                }
                CompositorMessage::VisibilityChanged { visible } => self.pending_visible = Some(visible),
                // Presentation-side notifications have no effect on the
                // frame loop; they are accepted so both endpoints share
                // one message type.
                CompositorMessage::FrameComposited { .. } | CompositorMessage::Custom(_) => {}
            }
        }
        while let Some(handoff) = self.comm.try_recv_pipe() {
            let _index = self.queue.push(handoff);
            self.framebuffers_dirty = true;
        }
        if self.framebuffers_dirty {
            self.recollect();
            self.framebuffers_dirty = false;
        }
        for (slot, weight) in weight_updates {
            if slot < self.weights.len() {
                self.weights[slot] = sanitize_weight(weight);
            }
        }
        if let Some(visible) = self.pending_visible.take() {
            self.visible = visible;
        }
    }

    /// Hands pipes marked for removal back to the presentation side.
    ///
    /// Only runs after [`Compositor::retire_previous_frame`] waited for
    /// the fence, so releasing the renderers no longer races the GPU.
    fn release_removed_pipes(&mut self) {
        if !self.queue.has_removed() {
            return;
        }
        for handoff in self.queue.drain_removed() {
            self.framebuffers_dirty = true;
            // On `Full`/`Closed` the handoff is released here instead:
            // it can no longer be rendered and nobody is listening.
            let _ = self.comm.send_pipe(handoff);
        }
    }

    /// Rebuilds the framebuffer snapshot and mix weights from the queue.
    fn recollect(&mut self) {
        self.framebuffers.clear();
        self.weights.clear();
        self.bases = self
            .queue
            .collect_framebuffers(&mut self.framebuffers, &mut self.weights);
    }

    /// Flags the framebuffers whose contents the layout pass has to
    /// discard (unknown tracked layout), so pipes clear them instead of
    /// loading garbage.
    fn update_undefined_mask(&mut self) {
        self.undefined_mask.clear();
        for framebuffer in self.framebuffers.iter() {
            self.undefined_mask
                .push(self.layout_of(framebuffer.image).is_none());
        }
    }

    /// Creates one active slot per queued pipe for this frame.
    fn build_active_pipes(&mut self) {
        let context = self.context;
        let Some(resources) = self.start_resources.as_ref() else {
            return;
        };
        let format = PipelineContext::color_format();
        self.active.clear();
        for index in 0..self.queue.len() {
            let command_buffer = resources
                .pipe_command_buffers
                .get(index)
                .copied()
                .unwrap_or(vk::CommandBuffer::null());
            let base = self.bases.get(index).copied().unwrap_or(0);
            let ctx = PipeCtx::new(
                context.device(),
                command_buffer,
                context.width(),
                context.height(),
                format,
                self.frame_index,
                base,
                &self.framebuffers,
            );
            self.active.push(ActivePipe {
                future: None,
                ctx,
                command_buffer,
                state: PipeState::Recording,
            });
        }
    }

    /// Reallocates resources if the pipe count or framebuffer count has changed.
    fn reallocate_if_needed(&mut self) -> Result<(), CompositorError> {
        let current_pipe_count = self.queue.len();
        let current_framebuffer_count = self.framebuffers.len();
        let max_framebuffers = current_framebuffer_count.max(8) as u32;
        let Some(resources) = self.start_resources.as_ref() else {
            return Ok(());
        };
        if resources.pipe_command_buffers.len() != current_pipe_count
            || resources.mix.max_framebuffers < max_framebuffers
        {
            self.quiesce();
            self.start_resources = Some(StartResources::create(
                self.context,
                current_pipe_count,
                max_framebuffers,
            )?);
        }
        Ok(())
    }

    /// Starts one frame: retires the previous submission, drains the
    /// communication channels, refreshes the framebuffer snapshot and
    /// creates the frame's synchronization objects.
    ///
    /// `wait_sync_file` is an optional `sync_file` fd (for example a DRM
    /// release point) to wait on before rendering; ownership is consumed
    /// — on success the driver takes the descriptor, on failure it is
    /// closed here. The first frame passes `None`.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   running.
    /// * [`CompositorError::FramePhase`] — another frame is still in
    ///   progress.
    /// * [`CompositorError::FenceWait`], [`CompositorError::CommandPoolReset`],
    ///   [`CompositorError::SemaphoreCreate`], [`CompositorError::SemaphoreImport`]
    ///   — the previous frame could not be retired or this frame's
    ///   synchronization could not be created.
    pub fn begin_frame(&mut self, wait_sync_file: Option<OwnedFd>) -> Result<(), CompositorError> {
        if self.state != CompositorState::Running {
            return Err(CompositorError::InvalidState {
                operation: "begin_frame",
                state: self.state,
            });
        }
        self.require_phase(CompositorPhase::Idle, "begin_frame")?;
        self.retire_previous_frame()?;
        // Release retired pipes before draining: the recollect below
        // must not snapshot framebuffers of a pipe that just left.
        self.release_removed_pipes();
        self.drain_comm();
        self.update_undefined_mask();
        self.reallocate_if_needed()?;
        let frame_sync = self.create_frame_sync(wait_sync_file)?;
        self.frame_sync = Some(frame_sync);
        self.build_active_pipes();
        self.phase = CompositorPhase::Recording;
        Ok(())
    }
}

impl Drop for Compositor<'_> {
    fn drop(&mut self) {
        self.quiesce();
        // Unpark the presentation side: its receive futures resolve to
        // `None` once the channels close.
        self.comm.close();
        // Field order keeps this implicit step documented: `active` (the
        // pipe futures) drops before `queue` (the renderers they borrow)
        // and `start_resources` (whose `Drop` assumes an idle device)
        // drops last.
    }
}
