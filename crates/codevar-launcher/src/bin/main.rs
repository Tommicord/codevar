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

// The vector table and `Reset` stub of the bare-metal image.
//
// On any other target this expands to nothing, so the hosted build keeps its
// C runtime as the entry point.
#[cfg(all(not(test), target_arch = "arm", target_os = "none"))]
codevar_rt::vector_table!();

/// The process heap on a Unix host: it grows by mapping fresh pages, so it
/// needs no boot-time configuration.
#[cfg(all(not(test), unix))]
use codevar_tlsf_alloc::LockedGrowableTlsf;

#[cfg(all(not(test), unix))]
#[global_allocator]
static HEAP: LockedGrowableTlsf = LockedGrowableTlsf::new();

/// Size of the static heap handed to the allocator on targets without a page
/// mapper (bare metal, wasm): there is no `mmap` to grow with, so the region
/// is part of the image.
///
/// A Cortex-M device counts its RAM in tens of kilobytes, so the region is a
/// single 8 KiB block there rather than the 64 KiB used elsewhere: small
/// enough to leave room for the stack, the framebuffers and the image's own
/// `.data`/`.bss` in a 64 KiB device, large enough for the launcher's
/// transient allocations.
#[cfg(all(not(test), not(unix)))]
const HEAP_REGION_BYTES: usize = if cfg!(all(target_arch = "arm", target_os = "none")) {
    0x2000
} else {
    0x1_0000
};

/// Backing store of the heap on targets without a page mapper.
#[cfg(all(not(test), not(unix)))]
static mut HEAP_REGION: [u8; HEAP_REGION_BYTES] = [0; HEAP_REGION_BYTES];

/// The fixed-region allocator used where there is no page mapper.
///
/// [`codevar_tlsf_alloc::LockedGrowableTlsf`] maps fresh pages on demand and
/// therefore needs `mmap`; `LockedTlsf` serves a pre-registered region instead,
/// which is all bare metal and wasm have to offer.
#[cfg(all(not(test), not(unix)))]
#[global_allocator]
static HEAP: codevar_tlsf_alloc::LockedTlsf = codevar_tlsf_alloc::LockedTlsf::new();

/// Registers the static heap region before the first allocation runs.
///
/// Must be the first thing every entry point does: `LockedTlsf` serves no
/// request until a region is contributed, so allocating first would abort.
#[cfg(all(not(test), not(unix)))]
fn init_heap() {
    let start = core::ptr::addr_of_mut!(HEAP_REGION).cast::<u8>();
    // SAFETY: `HEAP_REGION` exists only to back `HEAP`; nothing else aliases
    // it, the region spans the whole array, and this runs before any
    // allocation is attempted.
    let registered = unsafe { HEAP.lock().add_region(start, HEAP_REGION_BYTES) };
    if registered.is_err() {
        log_error!("heap region rejected by the allocator, cannot continue");
        codevar_target::halt();
    }
}

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
    #[cfg(not(unix))]
    init_heap();
    let args = match unsafe { codevar_cli_arg_parse::from_c_args(argc, argv) } {
        Ok(args) => args,
        Err(error) => {
            log_error!("invalid command line: {}", error);
            return Exit::Usage.code();
        }
    };
    driver::run(&args).code()
}

/// Reset handler for a target with no C runtime.
///
/// Nothing calls [`main`] there (there is no CRT to supply `argc`/`argv`), so
/// the linker's entry point comes straight here: bring up the heap, run the
/// driver with an empty argument list, then halt — there is no process left
/// to return an exit status to.
#[cfg(all(not(test), target_os = "none"))]
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    init_heap();
    if let Exit::Failure = driver::run(&[]) {
        log_error!("launcher stopped on a failure status");
    }
    codevar_target::halt()
}

/// Terminates the process with `code` after a panic report has been emitted.
///
/// Never returns. On a C runtime this is `_exit` (async-signal-safe and free
/// of destructor/flush work that could re-enter the allocator); on a
/// freestanding target there is no exit status to return, so the core is
/// halted — the same stop every unsupported service takes, and without
/// burning power in a spin loop that no console will ever show.
#[cfg(not(test))]
fn exit_after_panic(code: core::ffi::c_int) -> ! {
    #[cfg(unix)]
    unsafe {
        libc::_exit(code)
    }
    #[cfg(target_os = "none")]
    {
        let _ = code;
        codevar_target::halt()
    }
    #[cfg(all(not(unix), not(target_os = "none")))]
    {
        let _ = code;
        loop {
            core::hint::spin_loop();
        }
    }
}

/// Reports the panic through the logger and exits with the failure status.
///
/// The report itself is allocation-free (see
/// [`codevar_logger::report_panic`]), so this stays safe even when the heap
/// is corrupted or the panic happened inside the allocator.
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    codevar_logger::report_panic(info);
    exit_after_panic(Exit::Failure.code())
}

#[cfg(all(not(test), unix))]
#[link(name = "c")]
#[link(name = "gcc_s")]
unsafe extern "C" {}

/// Link-time stand-in for the personality routine referenced by the
/// precompiled `alloc` landing pads.
///
/// With `-C panic=abort` no unwinding ever reaches it, so an empty body is
/// safe; it exists only to satisfy the linker (`DW.ref.rust_eh_personality`
/// is emitted by the code generation unit that owns the panic path).
#[cfg(not(test))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
