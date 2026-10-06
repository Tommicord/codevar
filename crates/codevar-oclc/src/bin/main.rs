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
//! Two build modes, one pipeline:
//!
//! - **`std` (default):** a regular `main` that installs the hooks and runs
//!   the driver over `std::env::args_os`.
//! - **freestanding (`--no-default-features`, Unix):** a `#[no_main]` binary
//!   that receives `argc`/`argv` through the C runtime, copies them with
//!   [`codevar_oclc::argv::from_c_args`], and provides its own
//!   `#[panic_handler]` (ICE report + exit 101) and `#[global_allocator]`
//!   (`malloc`/`free`), so the compiler runs without linking `std`.
//!
//! Because the linker runs with `-nodefaultlibs` when `std` is absent, the
//! freestanding binary links `libc` explicitly (for `memcpy` and friends,
//! which `core` references, plus `malloc`/`_exit`) and `libgcc_s` (for
//! `_Unwind_Resume`, referenced by the precompiled `alloc` EH landing pads).

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![cfg_attr(all(not(feature = "std"), not(test)), no_main)]

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Installs process hooks and runs the driver over the OS command line.
#[cfg(feature = "std")]
fn main() {
    codevar_oclc::setup::init();
    std::process::exit(codevar_oclc::run_from_env().code());
}

/// Freestanding entry: copy `argc`/`argv`, initialize, run, return the exit
/// code to the C runtime.
///
/// # Safety
///
/// The `argc`/`argv` contract of [`codevar_oclc::argv::from_c_args`] is
/// satisfied by any C runtime that calls `main(argc, argv)`; this function
/// must not be called with fabricated values.
#[cfg(all(not(feature = "std"), not(test), unix))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn main(
    argc: core::ffi::c_int,
    argv: *const *const core::ffi::c_char,
) -> core::ffi::c_int {
    let args = match unsafe { codevar_oclc::argv::from_c_args(argc, argv) } {
        Ok(args) => args,
        Err(error) => {
            let message = alloc::format!("codevar-oclc: error: invalid command line: {error}\n");
            let _ = codevar_consoleutil::write_stderr(message.as_bytes());
            return codevar_oclc::Exit::Usage.code();
        }
    };
    codevar_oclc::setup::init();
    codevar_oclc::run(&args).code()
}

/// The freestanding binary requires Unix (`libc` startup, `malloc`, and
/// `/proc` or the C `argv` contract); use the default `std` feature elsewhere.
#[cfg(all(not(feature = "std"), not(test), not(unix)))]
compile_error!("codevar-oclc without the `std` feature requires a Unix target");

#[cfg(all(not(feature = "std"), not(test), unix))]
#[link(name = "c")]
#[link(name = "gcc_s")]
unsafe extern "C" {}

/// Link-time stand-in for the personality routine referenced by the
/// precompiled `alloc` landing pads. With `-C panic=abort` no unwinding ever
/// reaches it (the [`handle_panic`] hook exits the process instead), so an
/// empty body is safe; it exists to satisfy the linker.
#[cfg(all(not(feature = "std"), not(test), unix))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}

/// Reports a panic as an internal compiler error and exits with code 101.
///
/// Formatting the banner itself may panic on allocation failure; the nested
/// panic aborts the process, which still surfaces as a crash rather than
/// undefined behavior.
#[cfg(all(not(feature = "std"), not(test), unix))]
#[panic_handler]
fn handle_panic(info: &core::panic::PanicInfo<'_>) -> ! {
    let message = alloc::format!(
        "internal compiler error: {info}\nthis is a bug in codevar-oclc: {}\n",
        codevar_oclc::driver::BUG_REPORT_URL
    );
    let _ = codevar_consoleutil::write_stderr(message.as_bytes());
    unsafe { libc::_exit(codevar_oclc::Exit::Ice.code()) }
}

/// `malloc`-backed global allocator for the freestanding binary.
#[cfg(all(not(feature = "std"), not(test), unix))]
#[global_allocator]
static GLOBAL: SystemAllocator = SystemAllocator;

/// A global allocator delegating to the C runtime's `malloc`/`free`/`realloc`.
#[cfg(all(not(feature = "std"), not(test), unix))]
struct SystemAllocator;

#[cfg(all(not(feature = "std"), not(test), unix))]
unsafe impl core::alloc::GlobalAlloc for SystemAllocator {
    /// # Safety
    ///
    /// `layout.size()` may be zero; `malloc(0)` satisfies the allocator
    /// contract by returning either null or a unique free-able pointer.
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        unsafe { libc::malloc(layout.size()).cast() }
    }

    /// # Safety
    ///
    /// `ptr` must originate from this allocator and must not be used after
    /// this call.
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: core::alloc::Layout) {
        unsafe { libc::free(ptr.cast()) };
    }

    /// # Safety
    ///
    /// `ptr` must originate from this allocator; on failure the original
    /// block remains valid (the `realloc` contract).
    unsafe fn realloc(&self, ptr: *mut u8, _layout: core::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe { libc::realloc(ptr.cast(), new_size).cast() }
    }
}
