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

//! The `cl*` entry points — the C boundary of the driver.
//!
//! Every function here is an `#[unsafe(no_mangle)] extern "C"` symbol the
//! application links against directly (no ICD loader, no `dlopen`). The
//! layer does exactly three things:
//!
//! 1. validate handles and arguments per the OpenCL spec, mapping each
//!    failure to its precise `CL_*` code (the [`crate::Error`] mapping
//!    covers driver-internal failures only);
//! 2. marshal buffers, launches and events onto the object layer;
//! 3. execute buffer commands inline (a single address space makes them
//!    `memcpy`s) and hand kernel launches to the queue's device.
//!
//! Error-code policy: validation that the spec distinguishes (work
//! dimension vs. global size vs. work-group size, argument index vs.
//! size) is checked here so each case reports its own `CL_*` value;
//! everything else funnels through [`crate::Error::to_cl`].
