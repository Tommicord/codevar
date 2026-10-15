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

//! ARM semihosting: the host console, wall clock and exit status of a
//! bare-metal image.
//!
//! Semihosting lets a Cortex-M core use facilities of the machine it is
//! *running on* — a console, a wall clock, a file system — through a debug
//! agent (QEMU, a JTAG probe, a debugger): the guest executes `bkpt #0xAB`
//! with an operation number in `r0` and an argument in `r1`, and the agent,
//! which intercepts the instruction before any exception is taken, performs
//! the call and returns in `r0`.
//!
//! `bkpt #0xAB` — not `svc` — is the M-profile encoding. ARM's semihosting
//! ABI (DUI0471) fixes one instruction number per profile: `SVC 0x123456` in
//! ARM state, `SVC 0xAB` in ARMv7-A/R Thumb, and `BKPT 0xAB` for ARMv6-M
//! and ARMv7-M, because a Cortex-M `SVC` is reserved for the vector table
//! rather than the debug channel. Every M-profile agent expects the
//! breakpoint form — `cortex-m-semihosting`, OpenOCD, QEMU and the LLVM
//! bare-metal toolchain all issue it.
//!
//! The channel is optional, so it has to be *detected* rather than assumed.
//! With no agent listening the same `bkpt` is taken as a debug event nobody
//! services: the core raises `HardFault` (`DebugMonitor`, where a part has
//! it enabled). `codevar-rt` installs one handler on both vectors that
//! recognizes the semihosting breakpoint inside the flash image, calls
//! [`codevar_semihosting_unavailable`] below, skips the instruction and
//! fails the call. A probe therefore degrades to a `false` answer instead of
//! a lockup, which is what makes [`available`] trustworthy.
//!
//! Availability is cached: at most one probe call is issued over the
//! program's lifetime.
//!
//! # Termination
//!
//! [`exit`] hands the image's verdict to the agent as a process status:
//! `SYS_EXIT` reporting `ADP_Stopped_ApplicationExit` becomes the runner's
//! status `0` and every other reason its `1`, so a test running on the
//! target can pass or fail the `cargo run` that launched QEMU.
//!
//! # Console handles
//!
//! `SYS_WRITE` needs a *handle*, not a file number: the semihosting fd
//! namespace starts empty and is populated only by `SYS_OPEN`. Opening the
//! pseudo-file `":tt"` yields the standard streams as the spec defines them
//! — mode `w` is the host's stdout, mode `a` its stderr — so each stream is
//! opened once, lazily, and the handle is cached beside the channel state.

use core::sync::atomic::{AtomicI32, AtomicU8, Ordering};

/// Semihosting operation numbers (ARM DUI0471, "Semihosting operations").
mod op {
    /// Open a file; `":tt"` names one of the standard streams.
    pub const SYS_OPEN: u32 = 0x01;
    /// Write `len` bytes from a buffer to a stream (stdout/stderr).
    pub const SYS_WRITE: u32 = 0x05;
    /// Seconds since 1970-01-01T00:00:00Z, as counted by the host.
    pub const SYS_TIME: u32 = 0x11;
    /// Report an exception, or that execution has finished
    /// (`angel_SWIreason_ReportException` in earlier revisions of the spec).
    pub const SYS_EXIT: u32 = 0x18;
}

/// `":tt"`, the pseudo-file the spec fixes for the standard streams.
const TTY_PATH: &[u8] = b":tt\0";

/// `SYS_OPEN` mode for write access — the spec maps it to stdout.
const MODE_WRITE: usize = 4;
/// `SYS_OPEN` mode for append access — the spec maps it to stderr.
const MODE_APPEND: usize = 8;
/// Length of the mode string handed to `SYS_OPEN`, in bytes.
const MODE_LEN: usize = 1;

/// The `bkpt` immediate that selects semihosting in Thumb state (`0xAB`).
const BKPT: u32 = 0xAB;

/// `ADP_Stopped_ApplicationExit` — "the application finished normally"
/// (ARM DUI0471, software reason codes of `SYS_EXIT`).
const REASON_APPLICATION_EXIT: usize = 0x0002_0026;
/// `ADP_Stopped_InternalError` — "the image stopped for an unexpected
/// reason". Every reason other than `ADP_Stopped_ApplicationExit` reaches
/// an agent as an abnormal stop, which is how a failing test reports the
/// failure on the 32-bit API.
const REASON_INTERNAL_ERROR: usize = 0x0002_0024;

/// The channel has not been probed yet.
const STATE_UNKNOWN: u8 = 0;
/// An agent answered a call.
const STATE_AVAILABLE: u8 = 1;
/// No agent: the `bkpt` reached the vector table instead of a debug agent.
const STATE_UNAVAILABLE: u8 = 2;

/// Cached channel state, updated by [`available`] and
/// [`codevar_semihosting_unavailable`].
static STATE: AtomicU8 = AtomicU8::new(STATE_UNKNOWN);

/// `SYS_OPEN` has not been issued for stdout yet.
const NOT_OPEN: i32 = i32::MIN;

/// Lazily opened `SYS_WRITE` handle for [`Stream::Stdout`].
///
/// [`Stream::Stdout`]: crate::Stream::Stdout
static STDOUT_HANDLE: AtomicI32 = AtomicI32::new(NOT_OPEN);
/// Lazily opened `SYS_WRITE` handle for [`Stream::Stderr`].
///
/// [`Stream::Stderr`]: crate::Stream::Stderr
static STDERR_HANDLE: AtomicI32 = AtomicI32::new(NOT_OPEN);

/// Flags the channel as dead because a semihosting `bkpt` became a fault
/// that reached the vector table instead of a debug agent.
///
/// Called from `codevar-rt`'s assembly `DebugEvent` handler, which is why
/// this function has a stable symbol: the handler branches to it by name. It
/// runs in exception context, so it only stores one byte.
#[unsafe(no_mangle)]
extern "C" fn codevar_semihosting_unavailable() {
    STATE.store(STATE_UNAVAILABLE, Ordering::Relaxed);
}

/// Returns whether a debug agent is servicing semihosting calls.
///
/// The first call issues one probe (`SYS_TIME`); later calls read the cached
/// answer. A `false` result means this target has no semihosting console or
/// wall clock — callers must treat those as missing components.
pub fn available() -> bool {
    match STATE.load(Ordering::Relaxed) {
        STATE_AVAILABLE => true,
        STATE_UNAVAILABLE => false,
        _ => probe(),
    }
}

/// Probes the channel with `SYS_TIME`, which needs no argument and is always
/// implemented where semihosting exists at all.
fn probe() -> bool {
    // SAFETY: `SYS_TIME` takes no argument (the value is ignored by every
    // agent) and returns a scalar in `r0`, so no memory is touched.
    let seconds = unsafe { call(op::SYS_TIME, 0) };
    // Only `codevar-rt`'s `DebugEvent` handler can mark the channel dead,
    // and only for a call the agent did not intercept; a negative return is
    // the handler's `-1` failure value.
    if STATE.load(Ordering::Relaxed) == STATE_UNAVAILABLE || seconds < 0 {
        STATE.store(STATE_UNAVAILABLE, Ordering::Relaxed);
        false
    } else {
        STATE.store(STATE_AVAILABLE, Ordering::Relaxed);
        true
    }
}

/// Returns the host's wall-clock seconds since the Unix epoch.
///
/// `None` when there is no agent: a bare-metal image with no debug connection
/// has no real-world clock at all, and must fall back to counting from
/// startup (or stop, if it needs real dates).
pub fn wall_seconds() -> Option<u64> {
    if !available() {
        return None;
    }
    // SAFETY: `SYS_TIME` takes no argument and returns a scalar in `r0`.
    let seconds = unsafe { call(op::SYS_TIME, 0) };
    if seconds < 0 {
        STATE.store(STATE_UNAVAILABLE, Ordering::Relaxed);
        None
    } else {
        Some(seconds as u64)
    }
}

/// Terminates the image, reporting `code` as the process status.
///
/// `SYS_EXIT` (`angel_SWIreason_ReportException` in earlier revisions of the
/// spec) is how a target tells the debug agent that execution has finished.
/// On the 32-bit M-profile API the *parameter register is the reason code
/// itself* — there is no argument block — and the spec gives a 32-bit caller
/// no room for an exit status: an agent maps `ADP_Stopped_ApplicationExit`
/// to status `0` and every other reason to status `1`, which is exactly what
/// QEMU does. So `code == 0` reports a normal application exit and any other
/// `code` an abnormal one; carrying the full status would need the v2.0
/// `SYS_EXIT_EXTENDED` operation, gated on its `SH_EXT_EXIT_EXTENDED`
/// feature bit.
///
/// The status a runner observes is therefore the pass/fail verdict of a test
/// built on this channel — `cargo run` under QEMU exits `0` for `exit(0)` and
/// non-zero for anything else.
///
/// # Never returns
///
/// The agent is expected to stop the emulation or the image. When no agent
/// is listening (`available()` is `false`), or the agent lets execution
/// continue instead of terminating, this falls through to [`crate::halt`],
/// so the core stops in every case; the return register of `SYS_EXIT` is
/// "corrupted" by the spec, so a returned value carries no information and
/// is ignored.
pub fn exit(code: i32) -> ! {
    if available() {
        let reason = if code == 0 {
            REASON_APPLICATION_EXIT
        } else {
            REASON_INTERNAL_ERROR
        };
        // SAFETY: on this (32-bit) API `SYS_EXIT` takes the reason code as
        // the whole parameter, not a pointer, so no memory is read or
        // written; the agent either terminates the image — in which case
        // nothing below runs — or returns with the registers as it found
        // them, which the caller treats as a refusal.
        let _ = unsafe { call(op::SYS_EXIT, reason) };
    }
    crate::halt()
}

/// Writes `bytes` to the agent's console for `stream`.
///
/// Returns `false` when the channel is down, the stream could not be opened,
/// or the agent rejected the write; it never aborts, so it is safe to call
/// from a diagnostic path that may itself be running *because* the console
/// is missing.
pub fn write(stream: crate::Stream, bytes: &[u8]) -> bool {
    if !available() {
        return false;
    }
    let Some(handle) = handle_for(stream) else {
        return false;
    };
    // The `SYS_WRITE` block: handle, buffer, length. One machine word each.
    let block: [usize; 3] = [handle as usize, bytes.as_ptr() as usize, bytes.len()];
    // SAFETY: `block` is a live `[usize; 3]` for the duration of the call and
    // the agent only reads it; `bytes` is a live slice the agent reads
    // without retaining.
    let result = unsafe { call(op::SYS_WRITE, block.as_ptr() as usize) };
    // The spec's return is the count of bytes *not* written: 0 on success.
    result == 0
}

/// The `SYS_WRITE` handle of `stream`, opening the standard stream on first
/// use.
///
/// The semihosting fd namespace starts empty — even stdout and stderr have
/// no handle until the guest opens the pseudo-file `":tt"` (ARM DUI0471,
/// `SYS_OPEN`). Mode `w` selects the host's stdout and mode `a` its stderr,
/// so each stream is opened once and its handle cached; a failed open is
/// cached too, so a dead channel costs one attempt rather than one per
/// write. Returns `None` when the agent refused the open.
fn handle_for(stream: crate::Stream) -> Option<i32> {
    let (slot, mode) = match stream {
        crate::Stream::Stdout => (&STDOUT_HANDLE, MODE_WRITE),
        crate::Stream::Stderr => (&STDERR_HANDLE, MODE_APPEND),
    };
    let handle = slot.load(Ordering::Relaxed);
    if handle != NOT_OPEN {
        return (handle >= 0).then_some(handle);
    }
    // The `SYS_OPEN` block: path pointer, mode, mode length.
    let block: [usize; 3] = [TTY_PATH.as_ptr() as usize, mode, MODE_LEN];
    // SAFETY: `block` is a live `[usize; 3]` and `TTY_PATH` a live
    // NUL-terminated string for the duration of the call; the agent only
    // reads them without retaining either.
    let opened = unsafe { call(op::SYS_OPEN, block.as_ptr() as usize) } as i32;
    slot.store(opened, Ordering::Relaxed);
    (opened >= 0).then_some(opened)
}

/// Issues one semihosting call and returns the raw `r0`.
///
/// # Safety
///
/// `argument` must be a valid argument for `operation`: for the write
/// operations a pointer to a live three-word block whose second word points
/// at readable memory for the duration of the call.
#[inline]
unsafe fn call(operation: u32, argument: usize) -> isize {
    let mut r0 = operation;
    let r1 = argument;
    // SAFETY: the operands name the only registers the calling convention
    // requires for a semihosting call, and the default `options` tell the
    // compiler that the call may read memory (the write buffer). `bkpt`
    // itself does not touch the stack beyond the exception frame the core
    // pushes when the event goes unserviced. An agent leaves the registers
    // alone except for `r0`; without one, `codevar-rt`'s `DebugEvent`
    // handler writes `-1` into the stacked `r0`, which is the value observed
    // here.
    unsafe {
        core::arch::asm!(
            "bkpt {code}",
            code = const BKPT,
            inlateout("r0") r0,
            inlateout("r1") r1 => _,
            options(nostack),
        );
    }
    r0 as isize
}
