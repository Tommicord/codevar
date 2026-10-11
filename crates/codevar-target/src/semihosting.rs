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

//! ARM semihosting: the host console and wall clock of a bare-metal image.
//!
//! Semihosting lets a Cortex-M core use facilities of the machine it is
//! *running on* — a console, a wall clock, a file system — through a debug
//! agent (QEMU, a JTAG probe, a debugger): the guest executes `svc 0xAB`
//! with an operation number in `r0` and an argument in `r1`, and the agent,
//! which intercepts the instruction before any exception is taken, performs
//! the call and returns in `r0`.
//!
//! The channel is optional, so it has to be *detected* rather than assumed.
//! With no agent listening the same `svc` is taken as an ordinary `SVCall`
//! exception; `codevar-rt` installs an `SVCall` vector that recognizes the
//! semihosting encoding, calls [`mark_unavailable`], skips the instruction
//! and fails the call. A probe therefore degrades to a `false` answer and a
//! diagnosable halt instead of a lockup, which is what makes
//! [`available`] trustworthy.
//!
//! Availability is cached: at most one probe call is issued over the
//! program's lifetime.

use core::sync::atomic::{AtomicU8, Ordering};

/// Semihosting operation numbers (ARM DUI0471, "Semihosting operations").
mod op {
    /// Write `len` bytes from a buffer to a stream (stdout/stderr).
    pub const SYS_WRITE: u32 = 0x05;
    /// Seconds since 1970-01-01T00:00:00Z, as counted by the host.
    pub const SYS_TIME: u32 = 0x11;
}

/// The `svc` immediate that selects semihosting in Thumb state.
const SVC: u32 = 0xAB;

/// The channel has not been probed yet.
const STATE_UNKNOWN: u8 = 0;
/// An agent answered a call.
const STATE_AVAILABLE: u8 = 1;
/// No agent: the `SVCall` vector handled the call itself.
const STATE_UNAVAILABLE: u8 = 2;

/// Cached channel state, updated by [`available`] and [`mark_unavailable`].
static STATE: AtomicU8 = AtomicU8::new(STATE_UNKNOWN);

/// Flags the channel as dead because a semihosting `svc` reached the vector
/// table instead of a debug agent.
///
/// Called from `codevar-rt`'s assembly `SVCall` handler, which is why this
/// function has a stable symbol: the handler branches to it by name. It runs
/// in exception context, so it only stores one byte.
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
    // Only the `SVCall` handler can mark the channel dead, and only for a
    // call the agent did not intercept; a negative return is the handler's
    // `-1` failure value.
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

/// Which of the two standard streams a console write goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    /// Standard output (semihosting file handle 1).
    Stdout,
    /// Standard error (semihosting file handle 2).
    Stderr,
}

/// Writes `bytes` to the agent's console for `stream`.
///
/// Returns `false` when the channel is down or the agent rejected the write;
/// it never aborts, so it is safe to call from a diagnostic path that may
/// itself be running *because* the console is missing.
pub fn write(stream: Stream, bytes: &[u8]) -> bool {
    if !available() {
        return false;
    }
    // The `SYS_WRITE` block: handle, buffer, length. One machine word each.
    let block: [usize; 3] = [handle(stream) as usize, bytes.as_ptr() as usize, bytes.len()];
    // SAFETY: `block` is a live `[usize; 3]` for the duration of the call and
    // the agent only reads it; `bytes` is a live slice the agent reads
    // without retaining.
    let result = unsafe { call(op::SYS_WRITE, block.as_ptr() as usize) };
    // A successful `SYS_WRITE` returns 0; a short or failed write returns -1.
    result == 0
}

/// The semihosting file handle of `stream`, as fixed by POSIX.
const fn handle(stream: Stream) -> i32 {
    match stream {
        Stream::Stdout => 1,
        Stream::Stderr => 2,
    }
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
    let mut r1 = argument;
    // SAFETY: the operands name the only registers the calling convention
    // requires for a semihosting call, and the default `options` tell the
    // compiler that the call may read memory (the write buffer). `svc` itself
    // does not touch the stack beyond the exception frame the hardware
    // pushes. A debug agent leaves the registers alone except for `r0`;
    // without one, `codevar-rt`'s `SVCall` handler writes `-1` into the
    // stacked `r0`, which is the value observed here.
    unsafe {
        core::arch::asm!(
            "svc {code}",
            code = const SVC,
            inlateout("r0") r0,
            inlateout("r1") r1 => _,
            options(nostack),
        );
    }
    r0 as isize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_follow_posix() {
        assert_eq!(handle(Stream::Stdout), 1);
        assert_eq!(handle(Stream::Stderr), 2);
    }
}
