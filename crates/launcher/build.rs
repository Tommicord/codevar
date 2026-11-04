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

//! Link configuration for the `codevar` binary.
//!
//! [`codevar_rt::build::configure_link`] does the work: it is a no-op on
//! hosted targets, and on bare-metal `arm` it points the linker at
//! `codevar-rt`'s `link.x` (which lays the vector table at the base of flash
//! and `INCLUDE`s the matching `memory.x`). Nothing here has to know where
//! those scripts live.
//!
//! All `println!` use in this file is reserved for the `cargo:` build-script
//! protocol on stdout, which Cargo parses as metadata directives — it is not
//! application logging. Human-readable messages go through `cargo:warning`
//! directives.

use std::process::ExitCode;

fn main() -> ExitCode {
    codevar_rt::build::configure_link("codevar-launcher")
}
