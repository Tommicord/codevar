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

//! The `codevar-oclc` entry point.
//!
//! A `#[no_main]` Unix binary: the C runtime calls `main(argc, argv)`, which
//! copies the arguments with [`codevar_oclc::argv::from_c_args`], installs
//! the process hooks (logger level, signal handlers, and the ICE panic hook),
//! and hands the command line to [`codevar_oclc::run`].
//!
//! The binary provides its own `#[global_allocator]` — a growable
//! [`codevar_tlsf_alloc::LockedGrowableTlsf`] heap that maps virtual pages
//! on demand via `mmap` (no manual region setup, no `malloc`) — and links
//! `libc` explicitly (for `mmap` and friends, which `core` references) and
//! `libgcc_s` (for `_Unwind_Resume`, referenced by the precompiled `alloc`
//! EH landing pads).

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
extern crate alloc;

#[cfg(not(test))]
#[global_allocator]
static HEAP: LockedGrowableTlsf = LockedGrowableTlsf::new();

#[cfg(not(test))]
use codevar_logger::LogLevel;
use codevar_oclc::driver;
#[cfg(not(test))]
use codevar_tlsf_alloc::LockedGrowableTlsf;

/// Freestanding entry: copy `argc`/`argv`, initialize, run, return the exit
/// code to the C runtime.
///
/// # Safety
///
/// The `argc`/`argv` contract of [`codevar_oclc::argv::from_c_args`] is
/// satisfied by any C runtime that calls `main(argc, argv)`; this function
/// must not be called with fabricated values.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn main(
    argc: core::ffi::c_int,
    argv: *const *const core::ffi::c_char,
) -> core::ffi::c_int {
    let args = match unsafe { codevar_cli_arg_parse::from_c_args(argc, argv) } {
        Ok(args) => args,
        Err(error) => {
            let message = alloc::format!("error: invalid command line: {error}\n");
            let _ = codevar_consoleutil::write_stderr(message.as_bytes());
            return driver::Exit::Usage.code();
        }
    };
    codevar_logger::set_min_log_level(LogLevel::Error);
    codevar_sig_module_base::init();
    match codevar_sig_handler::install() {
        Ok(()) => {}
        Err(error) => codevar_logger::log_warn!("failed to install signal handler: {error}"),
    }
    driver::run(&args).code()
}

#[cfg(all(not(test), unix))]
#[link(name = "c")]
#[link(name = "gcc_s")]
unsafe extern "C" {}
