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

//! The codevar launcher entry point.
#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
extern crate alloc;

#[cfg(not(test))]
use codevar_launcher::driver;
#[cfg(not(test))]
use codevar_launcher::driver::Exit;
#[cfg(not(test))]
use codevar_logger::log_error;
#[cfg(not(test))]
use codevar_tlsf_alloc::LockedGrowableTlsf;

#[cfg(not(test))]
#[global_allocator]
static HEAP: LockedGrowableTlsf = LockedGrowableTlsf::new();

/// Freestanding entry: initialize the process hooks, run the window
/// event loop, then return the exit code to the C runtime.
///
/// # Safety
///
/// The C runtime must call this exactly once with the standard
/// `argc`/`argv` contract. Installing the signal handler here is
/// process-global and must not run concurrently with other initialization.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn main(
    argc: core::ffi::c_int,
    argv: *const *const core::ffi::c_char,
) -> core::ffi::c_int {
    let args = match unsafe { codevar_cli_arg_parse::from_c_args(argc, argv) } {
        Ok(args) => args,
        Err(error) => {
            log_error!("invalid command line: {}", error);
            return Exit::Usage.code();
        }
    };
    driver::run(&args).code()
}

#[cfg(all(not(test), unix))]
#[link(name = "c")]
#[link(name = "gcc_s")]
unsafe extern "C" {}
