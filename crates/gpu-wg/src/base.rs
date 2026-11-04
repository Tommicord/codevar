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
//! This module re-exports the primitives a caller of the WebGPU backend
//! names. It mirrors `codevar-gpu-vk`'s `base` module so the two backends
//! are used the same way; the Vulkan-only pieces (dma-buf export,
//! `sync_file` completion) have no counterpart here.
//!
//! # The pipe model
//!
//! Every contributor to a frame is a pipe whose renderer implements
//! [`PipeOffscreen`]: it owns one or more offscreen color targets and
//! reports a mix weight for each. The compositor collects those targets
//! once per frame and folds them into the canvas with a fullscreen mix
//! pass.
//!
//! # Frame lifecycle
//!
//! [`WgRenderer`] implements [`GpuRenderer`]: [`begin_frame`] acquires the
//! canvas texture, [`submit_frame`] records and submits the frame, and
//! [`end_frame`] presents it. The browser composites the canvas once the
//! queue work completes, so the completion handle is
//! [`FrameCompletion::Complete`].
//!
//! [`begin_frame`]: codevar_gpu_core::GpuRenderer::begin_frame
//! [`submit_frame`]: codevar_gpu_core::GpuRenderer::submit_frame
//! [`end_frame`]: codevar_gpu_core::GpuRenderer::end_frame
//! [`FrameCompletion::Complete`]: codevar_gpu_core::FrameCompletion::Complete

pub use crate::offscreen::{OffscreenFramebuffer, OffscreenTarget, PipeOffscreen};
pub use crate::pipe::{Pipe, PipeHandoff, PipeQueue, PipeSource};
pub use crate::pipe_ctx::{PipeCtx, PipeEntry, PipeEnv, PipeFuture, PipeOutcome, PipeResult};
pub use crate::pipeline::{WgConfig, WgContext, WgContextError};
pub use crate::platform::{WgError, WgPlatform, WgSurface, webgpu_features};
pub use crate::renderer::{WgRenderer, renderer_features};
pub use crate::types::{CommError, CompositorError, CompositorMessage, CompositorPhase, CompositorState};
