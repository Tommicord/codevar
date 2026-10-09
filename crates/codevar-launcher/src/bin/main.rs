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
use codevar_logger::{log_info, log_irr};
#[cfg(not(test))]
use codevar_tlsf_alloc::LockedTlsf;

#[cfg(not(test))]
#[global_allocator]
static HEAP: LockedTlsf = LockedTlsf::new();
#[cfg(not(test))]
static HEAP_SMEMORY: spin::Mutex<[u8; 0x4000]> = spin::Mutex::new([0u8; 0x4000]);

/// Freestanding entry: initialize the heap and process hooks, then return
/// the exit code to the C runtime.
///
/// # Safety
///
/// The C runtime must call this exactly once with the standard
/// `argc`/`argv` contract. Installing the heap region and the signal
/// handler here is process-global and must not run concurrently with
/// other initialization.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn main(
    _argc: core::ffi::c_int,
    _argv: *const *const core::ffi::c_char,
) -> core::ffi::c_int {
    unsafe {
        let mut smemory = HEAP_SMEMORY.lock();
        if HEAP
            .lock()
            .add_region(smemory.as_mut_ptr(), smemory.len())
            .is_err()
        {
            log_irr!("failed to init heap");
            return libc::EXIT_FAILURE;
        }
    }
    codevar_sig_module_base::init();
    if let Err(e) = codevar_sig_handler::install() {
        log_info!("error installing signal handler: {}", e);
    } else {
        log_info!("installed signal handler");
    }
    libc::EXIT_SUCCESS
}

#[cfg(all(not(test), unix))]
#[link(name = "c")]
#[link(name = "gcc_s")]
unsafe extern "C" {}
