//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The Codevar OpenCL compiler driver (`codevar-oclc`).
//!
//! This crate contains the entire compiler front end that does not depend on
//! `std`: argument acquisition ([`argv`]), the [`Driver`] pipeline, source
//! input/output, and process setup. The binary is a thin shell around
//! [`setup::init`] + [`run_from_env`], exactly as `rustc` splits
//! `rustc_driver::main` from `rustc_interface`, and `clang` splits its
//! generated `main` from `clang_main`.
//!
//! # `no_std` operation
//!
//! The library is `no_std` + [`alloc`] and compiles with the optional `std`
//! feature disabled. Arguments then come from the C entry point
//! ([`argv::from_c_args`]) or Linux `/proc/self/cmdline`
//! ([`argv::from_cmdline`]); files are read through [`fs`]; diagnostics are
//! written with `codevar-console-util`. Enabling the `std` feature adds the
//! `std::env::args_os` argument backend.
//!
//! # Exit codes
//!
//! | Code | Meaning                                                        |
//! |------|----------------------------------------------------------------|
//! | 0    | success, including `--help` and `--version`                     |
//! | 1    | compilation, lexical, or I/O failure; unimplemented stage      |
//! | 2    | invalid command line (bad flag, missing `INPUT`, bad UTF-8)    |
//! | 101  | internal compiler error (panic) reported by the ICE hook       |
//!
//! # Examples
//!
//! ```
//! let args = vec![String::from("codevar-oclc"), String::from("--version")];
//! assert_eq!(codevar_oclc::run(&args).code(), 0);
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
pub mod argv;
pub mod driver;
mod fs;

#[cfg(test)]
mod tests;

pub use driver::{Driver, Exit, run, run_from_env};

/// OpenCL kernels compiled and embedded in the executable at build time.
///
/// The build script compiles every `*.cl` file in the crate's `kernels/`
/// directory (overridable with the `CODEVAR_KERNEL_DIR` environment
/// variable) to SPIR-V, stages `<name>.spv` and `<name>.cl` in `OUT_DIR`,
/// and generates one [`kernels::KERNEL_MODULES`] registry entry plus three
/// statics per file — `<PREFIX>_SPIRV`, `<PREFIX>_SOURCE`, and
/// `<PREFIX>_ENTRY_POINTS` — exactly like shader embedding. A missing or
/// empty kernel directory yields an empty registry instead of a build
/// failure.
///
/// # Examples
///
/// Every embedded module carries a valid SPIR-V header:
///
/// ```
/// for module in codevar_oclc::kernels::KERNEL_MODULES {
///     assert!(module.spirv.len() >= 20, "truncated SPIR-V for {}", module.name);
///     assert_eq!(&module.spirv[0..4], &[0x03, 0x02, 0x23, 0x07]);
/// }
/// ```
pub mod kernels {
    /// One OpenCL kernel source file compiled and embedded at build time.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct KernelModule {
        /// File stem identifying the module (`vector_add.cl` → `vector_add`).
        pub name: &'static str,
        /// Original OpenCL dialect source text.
        pub source: &'static str,
        /// Compiled SPIR-V image of the source.
        pub spirv: &'static [u8],
        /// Names of the `#[kernel]` entry points the source declares.
        pub entry_points: &'static [&'static str],
    }

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
