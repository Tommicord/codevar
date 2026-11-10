//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Progress reporting and cancellation for long-running stages.

/// A phase of the pipeline, used for progress reporting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    /// Region forming (segmentation).
    #[default]
    Segment,
    /// Curve fitting / compositing.
    Compose,
    /// Optimization passes.
    Optimize,
    /// SVG serialization.
    Serialize,
}

/// A coarse progress value, 0..=100.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    pub phase: Phase,
    pub percent: u8,
}

impl Progress {
    pub const fn new(phase: Phase, percent: u8) -> Self {
        Self { phase, percent }
    }
}

/// A token that lets a caller request cancellation of a long-running stage.
#[derive(Debug, Default)]
pub struct CancelToken {
    cancelled: core::sync::atomic::AtomicBool,
}

impl CancelToken {
    /// A token that is not cancelled.
    pub const fn new() -> Self {
        Self {
            cancelled: core::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Request cancellation. After this, [`Ctx::check`] returns an error.
    pub fn cancel(&self) {
        self.cancelled
            .store(true, core::sync::atomic::Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
            .load(core::sync::atomic::Ordering::Relaxed)
    }
}

/// Context passed to incremental stages: progress reporting + cancellation.
pub struct Ctx<'a> {
    pub token: &'a CancelToken,
}

impl<'a> Ctx<'a> {
    pub const fn new(token: &'a CancelToken) -> Self {
        Self { token }
    }

    /// Report progress for a phase.
    pub fn report(&self, _phase: Phase, _percent: f32) {
        // No-op by default; callers may observe progress through their own
        // wrapper. Kept for API parity with the reference implementation.
        let _ = _phase;
        let _ = _percent;
    }

    /// Return [`Error::Cancelled`] if cancellation was requested.
    pub fn check(&self) -> Result<(), Error> {
        if self.token.is_cancelled() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

use crate::error::Error;
