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

//! Target platform services: the seam between portable code and the host.
//!
//! Codevar's crates are `no_std` and portable, but several of them need an
//! operating-system service (a clock, a console, a file system, a signal
//! handler). This crate owns the two policies for the targets that do not
//! provide one:
//!
//! - [`halt`] stops the processor forever, which is the correct way to
//!   "stop the application" when there is no OS to return an error to.
//! - [`abort_unsupported`] reports the missing service on the best available
//!   console and then [`halt`]s, so a porting gap surfaces as a loud,
//!   diagnosable stop instead of a silent wrong answer.
//!
//! On hosted targets the same calls exist (so portable code can use them
//! unconditionally) but are unreachable in practice, because every service
//! has a real implementation there.
//!
//! # Bare-metal backends
//!
//! On `target_os = "none"` every service still exists — so portable code
//! needs no target-specific `cfg` — but each one is backed by hardware:
//!
//! | Service  | ARM backend                       | Everyone else        |
//! |----------|-----------------------------------|----------------------|
//! | clock    | [`systick`] (SysTick)             | stop the application |
//! | console  | [`semihosting`] (BKPT, debug agent) | stop the application |
//!
//! ARM is where the two mechanisms live that a Rust bare-metal image can
//! assume without a board support package: SysTick, which ARMv7-M makes
//! mandatory, and semihosting, which every debug agent implements (through
//! `bkpt #0xAB`, the M-profile encoding). Both are
//! *probed* at run time ([`systick`] by register read-back,
//! [`semihosting`] by one trial call), so a part or session without them is
//! a clean "unsupported" stop rather than a fault. A new architecture needs
//! one `cfg` branch per service here; the callers never change.

#![cfg_attr(not(test), no_std)]

/// Which of the two standard streams a console write goes to.
///
/// The values are the file handles POSIX fixes for them (1 and 2), which is
/// also what semihosting expects.
#[cfg(target_os = "none")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

/// Semihosting: the console and wall clock a debug agent provides to a
/// bare-metal image, with detection so an absent agent is a clean
/// "unsupported" rather than a fault.
#[cfg(all(target_arch = "arm", target_os = "none"))]
pub mod semihosting;

/// SysTick-based uptime: the timer every Cortex-M part provides.
#[cfg(all(target_arch = "arm", target_os = "none"))]
pub mod systick;

/// The clock arithmetic, compiled wherever it can be tested: on the host it
/// exists only under `test`, on bare-metal ARM it exists for real.
#[cfg(any(test, all(target_arch = "arm", target_os = "none")))]
mod tick;

use core::sync::atomic::Ordering;
#[cfg(all(target_arch = "arm", target_os = "none"))]
pub use systick::uptime_nanos;

/// Nanoseconds since reset on targets with no timer backend.
///
/// A clock is not something to approximate, so rather than returning zero —
/// a plausible-looking answer that would make every timestamp a lie — the
/// application stops here. Adding a backend for another architecture means
/// providing its `uptime_nanos` in a branch of this `cfg`.
///
/// # Never returns
///
/// Diverges through [`abort_unsupported`].
#[cfg(all(target_os = "none", not(target_arch = "arm")))]
pub fn uptime_nanos() -> u64 {
    abort_unsupported("clock (no timer backend for this architecture)")
}

/// Writes to the console on targets with no console backend.
///
/// As with [`uptime_nanos`], a silent success would be worse than a stop:
/// the diagnostic path is the one place where losing output must be
/// visible.
///
/// # Never returns
///
/// Diverges through [`abort_unsupported`].
#[cfg(all(target_os = "none", not(target_arch = "arm")))]
pub fn console_write(_stream: Stream, _bytes: &[u8]) {
    abort_unsupported("console write (no console backend for this architecture)");
}

/// Stops the processor forever.
///
/// On bare-metal ARM (`thumb*`/`arm*-none-eabi`) this masks interrupts and
/// executes `wfi`, so the core neither spins nor wakes; on every other
/// target it spins with [`core::hint::spin_loop`] until reset or power-off,
/// which is the only portable "do nothing, forever" a `no_std` binary can
/// perform.
#[inline]
pub fn halt() -> ! {
    // SAFETY: `cpsid i` only masks interrupts and `wfi` is a wake-up hint;
    // neither touches memory nor requires particular alignment, so the
    // `nomem`/`nostack` options hold. `wfi` may return immediately when an
    // interrupt is already pending, which the outer loop absorbs.
    #[cfg(all(target_arch = "arm", target_os = "none"))]
    unsafe {
        core::arch::asm!("cpsid i", "wfi", options(nomem, nostack));
    }
    loop {
        core::hint::spin_loop();
    }
}

/// Data memory barrier: every explicit memory access before this point is
/// globally visible before any access after it.
///
/// Used after writing a descriptor and before ringing a doorbell so the
/// device on the other side can never observe the bell first.
#[inline]
pub fn dmb() {
    #[cfg(all(target_arch = "arm", target_os = "none"))]
    // SAFETY: `dmb sy` takes no operands and cannot fault.
    unsafe {
        core::arch::asm!("dmb sy", options(nostack, preserves_flags));
    }
    #[cfg(not(all(target_arch = "arm", target_os = "none")))]
    core::sync::atomic::compiler_fence(Ordering::SeqCst);
}

/// Data synchronization barrier: like [`dmb`] but also waits for outstanding
/// transactions (buffers, bus writes) to drain — the stronger form needed
/// before a device doorbell write is issued.
#[inline]
pub fn dsb() {
    #[cfg(all(target_arch = "arm", target_os = "none"))]
    // SAFETY: `dsb sy` takes no operands and cannot fault.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    #[cfg(not(all(target_arch = "arm", target_os = "none")))]
    core::sync::atomic::compiler_fence(Ordering::SeqCst);
}

/// Send event: sets the event register of every core in the multiprocessor.
///
/// A core that reaches `wfe` after this call does not sleep — the event
/// register is already set — which is what makes the doorbell
/// "store descriptor, then `sev`" free of lost wakeups.
#[inline]
pub fn sev() {
    #[cfg(all(target_arch = "arm", target_os = "none"))]
    // SAFETY: `sev` takes no operands and cannot fault.
    unsafe {
        core::arch::asm!("sev", options(nostack, preserves_flags));
    }
    #[cfg(not(all(target_arch = "arm", target_os = "none")))]
    core::hint::spin_loop();
}

/// Wait for event: sleep until an `sev` (or an interrupt) arrives.
///
/// On targets without the event register this is a spin hint — hosted tests
/// poll the epoch counter instead of sleeping, which is correct because the
/// polling loop re-checks the condition on every iteration.
#[inline]
pub fn wfe() {
    #[cfg(all(target_arch = "arm", target_os = "none"))]
    // SAFETY: `wfe` takes no operands and cannot fault.
    unsafe {
        core::arch::asm!("wfe", options(nostack, preserves_flags));
    }
    #[cfg(not(all(target_arch = "arm", target_os = "none")))]
    core::hint::spin_loop();
}

/// Terminates the application because `operation` has no implementation on
/// this target.
///
/// A one-line diagnostic is written to the standard error stream where the
/// target provides one — a hosted stderr, or the semihosting agent on bare
/// metal — and then the machine is halted. Callers use this for
/// operating-system services — a clock, a console, a file system — whose
/// absence cannot be expressed as a returned error.
///
/// # Examples
///
/// ```rust
/// # use codevar_target::abort_unsupported;
/// # if false {
/// abort_unsupported("file system write");
/// # }
/// ```
#[cold]
#[inline(never)]
pub fn abort_unsupported(operation: &str) -> ! {
    report(operation);
    halt()
}

/// Writes `bytes` to the platform console, or stops the application when
/// there is no console to write to.
///
/// The platform's detection runs first: if the channel exists but is down
/// — no agent listening, for instance — the failure is loud instead of
/// silently dropped, because a program that cannot report its progress on a
/// target without a debugger is indistinguishable from a hung one.
///
/// # Never returns
///
/// Diverges via [`abort_unsupported`] when the console is unavailable.
#[cfg(all(target_arch = "arm", target_os = "none"))]
pub fn console_write(stream: Stream, bytes: &[u8]) {
    if !semihosting::write(stream, bytes) {
        abort_unsupported("console write");
    }
}

/// Writes a best-effort "unsupported" diagnostic for `operation`.
///
/// On Unix this is a pair of raw `write(2)` calls to the standard error
/// file descriptor: they are async-signal-safe and do not allocate, so the
/// report stays safe even from a panic handler or with a corrupted heap.
/// Targets without a standard error stream get no report; the halt that
/// follows is the actual signal to the user.
#[cfg(all(unix, not(target_arch = "wasm32")))]
#[inline]
fn report(operation: &str) {
    // SAFETY: `write(2)` reads `count` bytes from `buf` and does not retain
    // the pointer; both slices are live static/stack bytes for the duration
    // of the call, and ignoring a short write only loses diagnostics.
    unsafe {
        let _ = libc::write(
            STDERR_FILENO,
            UNSUPPORTED_PREFIX.as_ptr().cast(),
            UNSUPPORTED_PREFIX.len(),
        );
        let _ = libc::write(STDERR_FILENO, operation.as_ptr().cast(), operation.len());
        let _ = libc::write(STDERR_FILENO, SUFFIX.as_ptr().cast(), SUFFIX.len());
    }
}

/// Bare metal reports through the agent, when there is one.
///
/// Three independent writes rather than a joined buffer: no allocation, no
/// buffering, and each call degrades on its own — the prefix may reach the
/// agent even if the operation name does not. Crucially this never aborts,
/// so a report from inside [`abort_unsupported`] cannot recurse.
#[cfg(all(target_arch = "arm", target_os = "none"))]
#[inline]
fn report(operation: &str) {
    let _ = semihosting::write(Stream::Stderr, UNSUPPORTED_PREFIX);
    let _ = semihosting::write(Stream::Stderr, operation.as_bytes());
    let _ = semihosting::write(Stream::Stderr, SUFFIX);
}

/// Targets without a standard error stream emit no diagnostic.
#[cfg(not(any(
    all(unix, not(target_arch = "wasm32")),
    all(target_arch = "arm", target_os = "none")
)))]
#[inline]
const fn report(_operation: &str) {}

/// Spins until `ready` reports completion, sleeping in between.
///
/// The wait primitive of the driver: every protocol point that blocks on
/// another core (work-group claim counters, descriptor-ring status words,
/// worker registration) funnels through this one loop so the architecture
/// specific event instruction stays in a single place.
///
/// The predicate runs with acquire semantics on its loads; callers pair it
/// with a release store on the producing side.
#[inline]
pub fn wait_until(mut ready: impl FnMut() -> bool) {
    while !ready() {
        wfe();
    }
}

/// Standard error file descriptor, as fixed by POSIX.
#[cfg(all(unix, not(target_arch = "wasm32")))]
const STDERR_FILENO: i32 = 2;

/// Leading text of the unsupported-service diagnostic.
#[cfg(any(
    all(unix, not(target_arch = "wasm32")),
    all(target_arch = "arm", target_os = "none")
))]
const UNSUPPORTED_PREFIX: &[u8] = b"codevar: unsupported on this target: ";

/// Trailing text of the unsupported-service diagnostic.
#[cfg(any(
    all(unix, not(target_arch = "wasm32")),
    all(target_arch = "arm", target_os = "none")
))]
const SUFFIX: &[u8] = b"\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halt_is_diverging() {
        // `halt()` is `-> !`, so this only checks that the call type-checks
        // in a position that requires divergence.
        let _: fn() -> ! = halt;
    }

    #[test]
    fn abort_is_diverging() {
        let _: fn(&str) -> ! = abort_unsupported;
    }

    #[test]
    fn barrier_and_event_do_not_trap() {
        super::dmb();
        super::dsb();
        super::sev();
        super::wfe();
        super::wait_until(|| true);
    }
}
