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

//! Process-level setup, mirroring rustc's `main` hook order.
//!
//! [`init`] runs exactly once, before any compilation work:
//!
//! 1. Raise the logger's minimum level (with the `std` feature) so stdout
//!    stays reserved for compiler output (only warnings/errors reach stderr).
//! 2. Install the Codevar signal handlers (backtrace on crash) where the
//!    `std` feature provides the module-base registry.
//! 3. Install the ICE (internal compiler error) panic hook: a bug banner on
//!    stderr plus exit code 101, matching rustc. Without `std`, the
//!    freestanding binary's own `#[panic_handler]` performs the same report.

#[cfg(feature = "std")]
use alloc::format;
#[cfg(feature = "std")]
use codevar_logger::{LogLevel, set_min_log_level};

/// Initializes logging, signal handling, and the ICE hook.
///
/// Failure to install signal handling degrades gracefully (a warning is
/// logged where the `std` feature provides the logger, and the process
/// continues with default OS behavior); everything else is infallible.
pub fn init() {
    #[cfg(feature = "std")]
    set_min_log_level(LogLevel::Error);
    #[cfg(feature = "std")]
    codevar_sig_module_base::init();
    match codevar_sig_handler::install() {
        Ok(()) => {}
        #[cfg(feature = "std")]
        Err(error) => codevar_logger::log_warn!("failed to install signal handler: {error}"),
        #[cfg(not(feature = "std"))]
        Err(_) => {}
    }
    install_ice_hook();
}

/// Replaces the panic hook with one that reports an internal compiler error
/// to stderr and exits with [`Exit::Ice`](crate::Exit::Ice)'s code (101).
///
/// The hook never unwinds and never prints to stdout; it is only installed by
/// [`init`], which the binary entry point calls once.
#[cfg(feature = "std")]
fn install_ice_hook() {
    std::panic::set_hook(std::boxed::Box::new(|info| {
        let message = format!(
            "internal compiler error: {info}\nthis is a bug in codevar-oclc: {}\n",
            crate::driver::BUG_REPORT_URL
        );
        let _ = codevar_consoleutil::write_stderr(message.as_bytes());
        std::process::exit(crate::Exit::Ice.code());
    }));
}

/// No-op without `std`: the freestanding binary provides a
/// `#[panic_handler]` that emits the same ICE banner.
#[cfg(not(feature = "std"))]
fn install_ice_hook() {}
