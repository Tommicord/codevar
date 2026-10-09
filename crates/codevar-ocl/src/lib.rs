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

//! Dynamic OpenCL runtime for Codevar.
//!
//! This crate talks to vendor OpenCL implementations (Intel, AMD, NVIDIA,
//! pocl, …) by loading their libraries at *runtime* — there is no
//! compile-time linkage against `libOpenCL`. Call [`Runtime::load`] to
//! discover platforms through the system ICD loader, or
//! [`Runtime::load_direct`] to scan vendor libraries directly.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod api;
mod buffer;
mod context;
mod dispatch;
mod error;
mod event;
mod icd;
mod kernel;
mod loader;
mod platform;
mod program;
mod query;
mod queue;
mod runtime;
pub mod sys;

pub use buffer::Buffer;
pub use context::Context;
pub use error::{Error, Result};
pub use event::{CommandState, Event, ProfilingTimestamp};
pub use kernel::{Arg, Kernel};
pub use platform::{Device, DeviceKind, Platform};
pub use program::Program;
pub use queue::{CommandQueue, QueueProperties};
pub use runtime::Runtime;
