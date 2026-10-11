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

//! Bare-metal runtime for Codevar's `*-none-eabi` targets.
//!
//! A hosted operating system loads an executable, arranges a stack and calls
//! `main`. A Cortex-M core does none of that: it reads the first two words of
//! the image — the initial stack pointer and the reset address — and jumps.
//! So a bare-metal image must carry a *vector table* at address `0`, and a
//! linker script must place it there while laying out `.data`, `.bss` and the
//! stack. This crate supplies both halves:
//!
//! - [`vector_table!`] emits the table and the `Reset` stub *into the binary
//!   that invokes it*, so the table always lands at the start of the flash
//!   image no matter how aggressively the linker garbage-collects.
//! - `link.x` (with `memory.x`) describes the memory map;
//!   `build::configure_link` hands them to the linker from a binary's
//!   `build.rs`, on bare-metal `arm` targets only.
//! - `codevar_rt_reset` runs after `Reset` installs the stack pointer: it
//!   copies `.data` from flash, zeroes `.bss`, and hands control to the
//!   binary's own `_start`.
//!
//! Everything else in the table points at `DefaultHandler`, which
//! [`codevar_target::halt`]s. Without a description of the machine's
//! peripherals an interrupt is not something this runtime can dispatch, so
//! trapping it and stopping is the diagnosable answer; jumping through an
//! unset vector would lock the core up with no way to tell why.
//!
//! Two exceptions are wired up because they carry services, not faults:
//! `SVCall` interprets the `SVC` instruction the target uses to reach a
//! debug agent (semihosting), and `SysTick` is the periodic tick
//! `codevar-target` builds its clock on. Both fail closed — an unrecognized
//! call or a tick nobody started ends in the same halt.
//!
//! # Portability
//!
//! The crate is empty on every other target: hosted binaries keep their C
//! runtime and never see the table, the script, or the reset path. Use the
//! same gating in a binary that wants to run on both:
//!
//! ```rust,ignore
//! #[cfg(all(target_arch = "arm", target_os = "none"))]
//! codevar_rt::vector_table!();
//! ```
//!
//! `build::configure_link` needs no gating at all — it decides from
//! `CARGO_CFG_TARGET_ARCH`/`CARGO_CFG_TARGET_OS`, so an ordinary
//! `cargo build` on a workstation is untouched.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

/// Linker-script configuration for a binary's `build.rs`.
///
/// A Codevar binary that wants to boot on bare metal has to tell the linker
/// about `link.x`/`memory.x`, and every one of them would otherwise have to
/// re-derive where those files live. Call [`build::configure_link`] from the
/// package's build script instead: it is a no-op on every hosted target, and
/// on bare-metal `arm` it points the linker at this package's scripts by
/// absolute path.
///
/// The module needs `std` (build scripts are `std` programs), so it is
/// behind the `std` feature. Cargo keeps build-dependency features out of
/// the target graph under resolver 2, which means a `thumbv7m-none-eabi`
/// build of a binary that depends on this crate never compiles it.
#[cfg(feature = "std")]
pub mod build {
    use std::path::PathBuf;
    use std::process::ExitCode;

    /// Triple components that need the bare-metal link script.
    const TARGET_ARCH: &str = "arm";
    const TARGET_OS: &str = "none";

    /// Emits the `cargo:` directives that link a bare-metal `arm` binary
    /// against this package's scripts, and reports the outcome as an
    /// [`ExitCode`] for a `build.rs` to return.
    ///
    /// A no-op (and a success) on every other target, so callers need no
    /// `cfg` of their own.
    ///
    /// # Diagnostics
    ///
    /// Failures are reported as `cargo:warning=<package>: <reason>` lines on
    /// stdout — the build-script protocol, not application logging — so a
    /// missing script stops the build with a message that names the package.
    ///
    /// # Panics
    ///
    /// Never; a missing `link.x`/`memory.x` is the `Err`/failure path above.
    pub fn configure_link(package: &str) -> ExitCode {
        match stage() {
            Ok(()) => ExitCode::SUCCESS,
            Err(reason) => {
                for line in format!("{package}: {reason}").lines() {
                    println!("cargo:warning={line}");
                }
                ExitCode::FAILURE
            }
        }
    }

    /// The directives themselves, split out so [`configure_link`] owns the
    /// reporting.
    ///
    /// Reads `CARGO_CFG_TARGET_ARCH`/`CARGO_CFG_TARGET_OS` from the
    /// environment Cargo hands every build script.
    fn stage() -> Result<(), String> {
        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if arch != TARGET_ARCH || os != TARGET_OS {
            return Ok(());
        }
        // `env!` is evaluated when *this crate* is compiled, so the path is
        // this package's source directory however it was laid out — a
        // packaged dependency included.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let scripts = [root.join("link.x"), root.join("memory.x")];
        for script in &scripts {
            if !script.is_file() {
                return Err(format!(
                    "link script '{}' does not exist; the `codevar-rt` package \
                     must ship it next to its sources",
                    script.display()
                ));
            }
        }
        // `-T` selects the script; `-L` is what makes its `INCLUDE memory.x`
        // resolve, and both are absolute so no search order can substitute a
        // different `memory.x`.
        println!("cargo:rustc-link-search=native={}", root.display());
        println!("cargo:rustc-link-arg=-T{}", scripts[0].display());
        for script in &scripts {
            println!("cargo:rerun-if-changed={}", script.display());
        }
        Ok(())
    }
}

/// Emits the vector table and `Reset` stub of a Cortex-M image.
///
/// Invoke it exactly once, at the root of a *binary* crate, and only where
/// `arm` + `target_os = "none"` holds:
///
/// ```rust,ignore
/// #[cfg(all(target_arch = "arm", target_os = "none"))]
/// codevar_rt::vector_table!();
/// ```
///
/// The expansion is a single [`core::arch::global_asm!`] block that defines
/// three things:
///
/// - `__vector_table`, a `.vector_table` section holding the initial stack
///   pointer (`_stack_top`, provided by `link.x`), the reset address, the
///   system exceptions and 48 external interrupt slots. Reset has its own
///   handler, `SVCall` dispatches semihosting and `SysTick` counts ticks
///   (both defined by this crate and `codevar-target`); everything else
///   points at `DefaultHandler`.
/// - `Reset`, the reset handler: it masks interrupts, installs `_stack_top`
///   as the main stack pointer — independent of what the board firmware left
///   there — and calls `codevar_rt_reset`.
/// - a literal pool for `Reset`'s `_stack_top` load, emitted by `.ltorg`.
///
/// # Panics
///
/// The macro itself cannot fail; the *link* fails if the binary's `build.rs`
/// did not install `link.x`, because `_stack_top` and `ENTRY(Reset)` would be
/// undefined.
#[cfg(all(target_arch = "arm", target_os = "none"))]
#[macro_export]
macro_rules! vector_table {
    () => {
        ::core::arch::global_asm!(
            r#"
                .section .vector_table, "a"
                .global __vector_table
            __vector_table:
                .word   _stack_top
                .word   Reset
                .word   DefaultHandler
                .word   DefaultHandler
                .word   DefaultHandler
                .word   DefaultHandler
                .word   DefaultHandler
                .word   0
                .word   0
                .word   0
                .word   0
                .word   SVCall
                .word   DefaultHandler
                .word   0
                .word   DefaultHandler
                .word   SysTick
                .rept   48
                .word   DefaultHandler
                .endr

                .section .text, "ax"
                .global Reset
                .type Reset, %function
                .thumb_func
            Reset:
                cpsid   i
                ldr     r0, =_stack_top
                msr     msp, r0
                bl      codevar_rt_reset
            1:
                b       1b
                .ltorg
                .size Reset, . - Reset
            "#
        );
    };
}

/// Reset-path helpers, `arm` bare metal only: they name linker symbols that
/// `link.x` defines, so they must not exist anywhere else.
#[cfg(all(target_arch = "arm", target_os = "none"))]
mod init {
    // Linker-defined: `.data`'s RAM bounds, `.bss`'s bounds, and `.data`'s
    // load address in flash. `link.x` defines all five.
    unsafe extern "C" {
        static mut _sdata: u8;
        static mut _edata: u8;
        static mut _sbss: u8;
        static mut _ebss: u8;
        /// Load address of `.data`, in flash.
        static _sidata: u8;
    }

    /// Copies `.data` into place and zeroes `.bss`.
    ///
    /// Both ranges are produced by `link.x`; an empty `.data` or `.bss`
    /// collapses to a zero-length copy, which is well defined for `u8`.
    ///
    /// Must run exactly once, before anything reads a `static`, and before
    /// the first allocation — `Reset` calls it before `_start` for that
    /// reason.
    pub(super) fn initialize_memory() {
        let data_len = (&raw const _edata as usize).wrapping_sub(&raw const _sdata as usize);
        let bss_len = (&raw const _ebss as usize).wrapping_sub(&raw const _sbss as usize);
        // SAFETY: `_sidata .. _sidata + data_len` is the flash image of `.data`
        // and `_sdata .. _edata` is the disjoint RAM range `link.x` placed it
        // in; both symbols come from the same script, and `u8` has no alignment
        // or validity constraint. `_sbss .. _ebss` is the `.bss` range the
        // script reserved for zero-initialized data, which is exactly what
        // `write_bytes` must produce.
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::addr_of!(_sidata),
                core::ptr::addr_of_mut!(_sdata),
                data_len,
            );
            core::ptr::write_bytes(core::ptr::addr_of_mut!(_sbss), 0, bss_len);
        }
    }

    // The binary's freestanding entry point, defined by its `main.rs`.
    unsafe extern "C" {
        fn _start() -> !;
    }

    /// Invoked by `Reset` once the stack pointer is live.
    ///
    /// Sets up `.data`/`.bss`, unmasks interrupts, and transfers control to
    /// the binary's `_start`, which owns heap registration, logging and the
    /// application itself. Interrupts stay masked through the memory
    /// initialization because a handler must never observe a half-copied
    /// `.data`; they are unmasked only once statics are trustworthy, and
    /// before `_start` so the runtime's own services — a SysTick-driven
    /// clock, a semihosting `SVC` — are live from the first line of the
    /// application. This function never returns; if the binary did not
    /// define `_start`, or the image cannot run at all,
    /// [`codevar_target::halt`] stops the core.
    #[unsafe(no_mangle)]
    extern "C" fn codevar_rt_reset() -> ! {
        initialize_memory();
        // SAFETY: `cpsie i` only unmasks interrupts. It is deliberately not
        // `nomem`: the compiler must not move any of the `.data`/`.bss`
        // writes past it, and must assume an interrupt handler can run
        // immediately after — which is exactly what the two `asm!` options
        // (no `nomem`, no `nostack`) guarantee.
        unsafe {
            core::arch::asm!("cpsie i", options(nostack));
        }
        // SAFETY: `_start` is the entry point the binary declares for
        // freestanding targets; it is `extern "C"`, takes no arguments, and
        // never returns, which is exactly the contract here.
        unsafe { _start() }
    }
}

/// Every unhandled exception lands here: mask interrupts and stop.
///
/// A fresh vector table has no interrupt routing, no peripheral drivers and
/// no scheduler, so "handle it" is not an option yet. Halting keeps the core
/// in a defined, observable state instead of re-entering the fault.
#[cfg(all(target_arch = "arm", target_os = "none"))]
#[unsafe(no_mangle)]
extern "C" fn DefaultHandler() -> ! {
    codevar_target::halt()
}

/// The `SVCall` handler: semihosting, or stop.
///
/// A `SVC` instruction is how code running on the target asks the *debug
/// agent* for an operating-system service — writing to the console,
/// reading the wall clock — and it is the only request channel a bare-metal
/// image has. The ARMv7-M architecture specifies that a `SVC` is taken with
/// the instruction's own bits in the exception frame, so this handler
///
/// - recognizes Codevar's call (`SVC #0xAB`), reports the call to
///   `codevar-target` (which records that no agent is listening, so later
///   probes fail fast instead of faulting again), and returns `-1` in the
///   stacked `r0` — "the agent refused", which the caller turns into an
///   ordinary unsupported-service stop, and
/// - treats *any other* `SVC` as the unknown-service case it is: a branch
///   to [`DefaultHandler`], which halts with the fault observable.
///
/// Written as one `global_asm!` block because the handler must address the
/// hardware exception frame through `sp` *at entry*: a prologue would move
/// `sp` and hide it. It never pushes anything it does not pop, keeps `sp`
/// 8-byte aligned across the call (AAPCS), and returns with `bx lr`, where
/// `lr` is the hardware-provided `EXC_RETURN` value.
#[cfg(all(target_arch = "arm", target_os = "none"))]
core::arch::global_asm!(
    r#"
        .section .text.SVCall, "ax"
        .global SVCall
        .type SVCall, %function
        .thumb_func
    SVCall:
        ldr     r0, [sp, #24]
        ldrh    r1, [r0, #-2]
        movw    r2, #0xDFAB
        cmp     r1, r2
        beq     3f
        ldrh    r1, [r0]
        cmp     r1, r2
        bne     2f
        adds    r0, #2
        str     r0, [sp, #24]
    3:
        push    {r4, lr}
        bl      codevar_semihosting_unavailable
        pop     {r4, lr}
        movs    r0, #0
        subs    r0, r0, #1
        str     r0, [sp]
        bx      lr
    2:
        b       DefaultHandler
        .size   SVCall, . - SVCall
    "#
);
