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
//! The library is `no_std` + [`alloc`] and compiles with the `std` feature
//! disabled. Arguments then come from the C entry point ([`argv::from_c_args`])
//! or Linux `/proc/self/cmdline` ([`argv::from_cmdline`]); files are read
//! through a small `libc` layer ([`fs`]); diagnostics are written with
//! `codevar-console-util`. The `std` feature (default) adds
//! `std::env::args_os` and `std::fs` backends plus the ICE panic hook.
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

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod argv;
pub mod driver;
mod fs;
pub mod setup;

#[cfg(test)]
mod tests;

pub use driver::{Driver, Exit, run, run_from_env};
