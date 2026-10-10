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

//! Compositor-based rendering: pipes, offscreen framebuffers and the
//! fragment-shader mix pass.
//!
//! This module provides a *pipe* model: every
//! contributor to a frame is a [`PipeSource`] whose
//! [`PipeSource::pipe_entry`] returns an async [`PipeFuture`]. The
//! compositor drives those futures from [`Compositor::poll_composite`],
//! submits each pipe as its own queue submission chained with binary
//! semaphores, and finally folds every offscreen framebuffer registered
//! through [`PipeOffscreen`] into the presentation target with a
//! fullscreen `mix()` pass (`kernel/mix.frag`).
//!
//! # Frame lifecycle
//!
//! 1. [`Compositor::begin_frame`] retires the previous submission,
//!    drains the [`CompositorComm`] channels, refreshes the framebuffer
//!    snapshot and creates the frame's wait semaphore.
//! 2. [`Compositor::poll_composite`] walks three phases: the *layout*
//!    pass transitions every framebuffer to
//!    `COLOR_ATTACHMENT_OPTIMAL`, the *pipe* phase polls and submits the
//!    pipe futures in queue order, and the *mix* pass composites the
//!    framebuffers into the presentation target and hands it off to
//!    `GENERAL`.
//! 3. [`Compositor::end_frame`] exports GPU completion of the mix pass as
//!    a `sync_file` descriptor for the presentation layer.
//!
//! [`Compositor::composite_frame`] and [`Compositor::draw_frame`] chain
//! the steps for async callers; [`block_on`] drives them from
//! synchronous code.
//!
//! # Synchronization
//!
//! Submissions are chained GPU-to-GPU with binary semaphores: the layout
//! pass signals `chain[0]`, pipe *k* waits on the previous signal and
//! signals `chain[k + 1]`, and the mix pass waits on the last signal and
//! signals an exportable semaphore. A single fence covers the whole
//! chain; it is waited on at the start of the next frame (GPU-to-CPU),
//! which also returns every chain semaphore to the unsignaled state so
//! the pools can be reused.
//!
//! The compositor owns the image layouts: a pipe records draws but never
//! transitions the framebuffer it is attached to (see
//! [`PipeCtx::begin_render`]).

pub use crate::comm::{
    CompositorComm, CompositorEndpoint, RecvFuture, RendererEndpoint, SendFuture, comm_pair,
};
pub use crate::compositor::{Compositor, block_on};
pub use crate::mix::{MixPush, load_spir_v};
pub use crate::offscreen::{OffscreenFramebuffer, OffscreenTarget, PipeOffscreen};
pub use crate::pipe::{Pipe, PipeHandoff, PipeQueue, PipeSource};
pub use crate::pipe_ctx::{PipeCtx, PipeEntry, PipeEnv, PipeFuture, PipeOutcome, PipeResult};
pub use crate::types::{CommError, CompositorError, CompositorMessage, CompositorPhase, CompositorState};
