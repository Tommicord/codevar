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

//! SPIR-V program loading for the embedded driver.
//!
//! A program "binary" handed to `clCreateProgramWithBinary` is a SPIR-V
//! module in the OpenCL embedded subset. This module validates it once,
//! at load time, into flat interpreter tables:
//!
//! * [`decode`] — header and instruction framing.
//! * [`ops`] — the opcode and enumerant constants of that subset.
//! * [`types`] — type, layout and scalar-lane walkers.
//! * [`builtin`] — `OpenCL.std` arities and linkage-import dispatch.
//! * [`module`] — the two-pass loader that rejects anything outside the
//!   subset and materializes [`module::SpirvModule`].
//!
//! Execution itself lives with the work-group scheduler; nothing in the
//! loader runs twice per launch.

mod builtin;
mod decode;
mod module;
mod ops;
mod types;

pub(crate) use module::{load, SpirvEntry, SpirvModule};
