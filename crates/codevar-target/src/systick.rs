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

//! SysTick: the timer every Cortex-M provides, and the uptime built on it.
//!
//! A bare-metal image has no operating-system clock to ask, so it measures
//! time itself: SysTick is a 24-bit down-counter that is *architecturally
//! mandatory* on ARMv7-M parts, which makes it the one timer a portable
//! image can assume. This module
//!
//! - detects the counter (a part that omits it implements the registers as
//!   read-as-zero/write-ignored, so the presence check is a read-back),
//! - calibrates the core clock against the semihosting wall clock when an
//!   agent is present, and otherwise uses `CODEVAR_CPU_HZ`,
//! - runs it as a periodic tick, counting whole ticks in [`SysTick`] (the
//!   vector entry `codevar-rt` installs) plus the fraction of the current
//!   tick from the counter itself, and
//! - answers [`uptime_nanos`], the "time since reset" every embedded clock
//!   is built from.
//!
//! Nothing here is a wall clock: with no agent and no battery-backed
//! register the concept does not exist, and a caller that needs real dates
//! must supply an epoch of its own (see `codevar-time-core`).

use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

use crate::semihosting;

/// SysTick control and status, `0xE000_E010`.
const SYST_CSR: *mut u32 = 0xE000_E010 as *mut u32;
/// Reload value, `0xE000_E014`.
const SYST_RVR: *mut u32 = 0xE000_E014 as *mut u32;
/// Current value, `0xE000_E018`.
const SYST_CVR: *mut u32 = 0xE000_E018 as *mut u32;

/// `CSR.ENABLE`: the counter is running.
const CSR_ENABLE: u32 = 1 << 0;
/// `CSR.TICKINT`: wrap raises `SysTick`.
const CSR_TICKINT: u32 = 1 << 1;
/// `CSR.CLKSOURCE`: count the core clock, not the (optional) reference clock.
const CSR_CLKSOURCE: u32 = 1 << 2;
/// `CSR.COUNTFLAG`: set when the counter reached zero since the last read.
const CSR_COUNTFLAG: u32 = 1 << 16;
/// The three writable control bits, for presence probing and read-backs.
const CSR_CONTROL: u32 = CSR_ENABLE | CSR_TICKINT | CSR_CLKSOURCE;

/// Largest 24-bit reload value (used while calibrating).
const RELOAD_MAX: u32 = 0x00FF_FFFF;
/// Nominal tick period: one millisecond.
const TARGET_TICK_HZ: u32 = 1_000;
/// A core clock below this cannot be divided into a millisecond, so the
/// maximum period is used instead of an interrupt storm.
const MIN_TICK_HZ: u32 = 10_000;
/// Core clock assumed when there is no agent to calibrate against: the
/// 12 MHz system clock of QEMU's `lm3s6965evb`, and a common board default.
const DEFAULT_HZ: u32 = 12_000_000;
/// Sanity bounds for a calibrated core clock; outside them the measurement
/// is discarded rather than believed.
const MIN_PLAUSIBLE_HZ: u32 = 1_000;
const MAX_PLAUSIBLE_HZ: u32 = 400_000_000;
/// Upper bound on wall-clock probe iterations during calibration, so a
/// broken agent cannot hang the first clock read.
const MAX_CALIBRATION_PROBES: u32 = 8_000_000;

/// The counter is not running yet.
const STATE_UNINITIALISED: u8 = 0;
/// The counter is running and [`HZ`] holds the core clock.
const STATE_RUNNING: u8 = 1;
/// This part has no SysTick; time comes from the agent or not at all.
const STATE_ABSENT: u8 = 2;

/// Lifecycle of the timer, written by [`start`].
static STATE: AtomicU8 = AtomicU8::new(STATE_UNINITIALISED);
/// Core clock in Hz once known; `0` until [`start`] finishes.
static HZ: AtomicU32 = AtomicU32::new(0);
/// Reload value in force, published before the counter is enabled.
static RELOAD: AtomicU32 = AtomicU32::new(RELOAD_MAX);
/// Nanoseconds one tick covers at the current reload, published before the
/// counter is enabled and read only after [`STATE`] reports `RUNNING`, so no
/// interrupt is involved.
static mut TICK_NANOS: u64 = 0;
/// Whole ticks elapsed since reset.
///
/// `u64` has no atomic form on Cortex-M (the target stops at 32-bit
/// atomics), so the handler writes it and readers mask interrupts — see
/// [`read_counts`].
static mut TICKS: u64 = 0;

/// Counts one SysTick tick; installed in the vector table by `codevar-rt`.
///
/// Runs in exception context with interrupts of its own priority blocked, so
/// it cannot nest, and no other code writes [`TICKS`].
#[unsafe(no_mangle)]
extern "C" fn SysTick() {
    // SAFETY: see the type-level docs — single writer, no nesting, and every
    // reader masks interrupts for the duration of the pair.
    unsafe {
        *core::ptr::addr_of_mut!(TICKS) += 1;
    }
}

/// Nanoseconds since reset, or stops the application when this target has no
/// clock at all.
///
/// The first call starts (and calibrates) the counter, which takes about one
/// second when a semihosting agent is present. Accuracy afterwards is one
/// tick — a millisecond, unless the core is slower than 10 kHz — plus the
/// core-clock error left by calibration. The value never goes backwards and
/// is unaffected by the wall clock.
///
/// With no SysTick but a semihosting agent, the agent's wall clock is used
/// instead (one-second resolution). With neither, the call never returns: a
/// device that cannot express elapsed time is missing a required component,
/// and the project's policy for that is [`crate::abort_unsupported`].
pub fn uptime_nanos() -> u64 {
    let hz = ensure_running();
    if hz == 0 {
        return match semihosting::wall_seconds() {
            Some(seconds) => seconds.saturating_mul(1_000_000_000),
            None => crate::abort_unsupported("clock (SysTick and semihosting both missing)"),
        };
    }
    let (ticks, counter) = read_counts();
    // SAFETY: `TICK_NANOS` is written once by `start` before the counter is
    // enabled, and `STATE` only reports `RUNNING` after that write, so this
    // read observes the initialized value with no concurrent writer.
    let tick_nanos = unsafe { *core::ptr::addr_of!(TICK_NANOS) };
    let reload = RELOAD.load(Ordering::Relaxed);
    // The counter counts down from `reload`, so cycles inside this tick are
    // `reload - counter`.
    let within_tick = reload.saturating_sub(counter) as u64;
    ticks
        .saturating_mul(tick_nanos)
        .saturating_add(within_tick.saturating_mul(1_000_000_000) / u64::from(hz))
}

/// Core clock in Hz, starting the counter on first use.
///
/// Returns `0` when this part has no SysTick, which callers must treat as
/// "no timer" rather than as a frequency.
fn ensure_running() -> u32 {
    match STATE.load(Ordering::Acquire) {
        STATE_RUNNING => HZ.load(Ordering::Relaxed),
        STATE_ABSENT => 0,
        _ => start(),
    }
}

/// Detects, calibrates and starts SysTick exactly once.
fn start() -> u32 {
    // Presence: a part without SysTick ignores writes and reads back zero.
    // `COUNTFLAG` is masked because it may be set by a wrap that happened
    // between the write and this read without meaning anything.
    unsafe {
        core::ptr::write_volatile(SYST_RVR, RELOAD_MAX);
        core::ptr::write_volatile(SYST_CVR, 0);
        core::ptr::write_volatile(SYST_CSR, CSR_CLKSOURCE | CSR_TICKINT);
        let readback = core::ptr::read_volatile(SYST_CSR) & (CSR_CONTROL & !CSR_ENABLE);
        if readback != (CSR_CLKSOURCE | CSR_TICKINT) {
            STATE.store(STATE_ABSENT, Ordering::Release);
            return 0;
        }
    }

    let hz = calibrate().unwrap_or(FALLBACK_HZ);
    let reload = tick_reload(hz);
    // Publish everything a reader needs before the counter can interrupt
    // anyone, then start it: from here the `SysTick` vector owns `TICKS`.
    let tick_nanos = (u64::from(reload) + 1).saturating_mul(1_000_000_000) / u64::from(hz);
    RELOAD.store(reload, Ordering::Relaxed);
    HZ.store(hz, Ordering::Relaxed);
    // SAFETY: single-threaded initialization — nothing reads `TICK_NANOS`
    // until `STATE` flips below, and `TICKS` is only written while the
    // counter is still disabled.
    unsafe {
        *core::ptr::addr_of_mut!(TICK_NANOS) = tick_nanos;
        core::ptr::write_volatile(SYST_RVR, reload);
        core::ptr::write_volatile(SYST_CVR, 0);
        *core::ptr::addr_of_mut!(TICKS) = 0;
        core::ptr::write_volatile(SYST_CSR, CSR_CLKSOURCE | CSR_TICKINT | CSR_ENABLE);
    }
    STATE.store(STATE_RUNNING, Ordering::Release);
    hz
}

/// Reload value for a `1 / TARGET_TICK_HZ` second tick at `hz`.
fn tick_reload(hz: u32) -> u32 {
    if hz < MIN_TICK_HZ {
        RELOAD_MAX
    } else {
        (hz / TARGET_TICK_HZ) - 1
    }
}

/// Measures the core clock against the agent's wall clock.
///
/// The window is aligned to a whole-second boundary first, so the measured
/// interval is a whole number of seconds to within one probe call: the
/// counter runs at its maximum period while two successive boundary
/// crossings are waited for, and `cycles / seconds` is the core clock.
///
/// Returns `None` when there is no agent, when the count looks impossible,
/// or when the agent never advances the clock.
fn calibrate() -> Option<u32> {
    if !semihosting::available() {
        return None;
    }
    // Align to a boundary: the loop exits just after a second ticks over.
    let boundary = semihosting::wall_seconds()?;
    let mut probes = 0;
    while semihosting::wall_seconds()? == boundary {
        probes += 1;
        if probes >= MAX_CALIBRATION_PROBES {
            return None;
        }
        core::hint::spin_loop();
    }

    // Measure exactly one further second.
    unsafe {
        core::ptr::write_volatile(SYST_RVR, RELOAD_MAX);
        core::ptr::write_volatile(SYST_CVR, 0);
        core::ptr::write_volatile(SYST_CSR, CSR_CLKSOURCE | CSR_TICKINT | CSR_ENABLE);
        *core::ptr::addr_of_mut!(TICKS) = 0;
    }
    let start = semihosting::wall_seconds()?;
    while semihosting::wall_seconds()? == start {
        probes += 1;
        if probes >= MAX_CALIBRATION_PROBES {
            stop_counter();
            return None;
        }
        core::hint::spin_loop();
    }
    let end = semihosting::wall_seconds()?;
    let (ticks, counter) = read_counts();
    stop_counter();

    let seconds = end.checked_sub(start)?;
    let cycles = (ticks as u128) * u128::from(RELOAD_MAX + 1)
        + u128::from(RELOAD_MAX.saturating_sub(counter));
    let hz = u32::try_from(cycles / u128::from(seconds)).ok()?;
    if (MIN_PLAUSIBLE_HZ..=MAX_PLAUSIBLE_HZ).contains(&hz) {
        Some(hz)
    } else {
        None
    }
}

/// Stops the counter between calibration and the real start.
fn stop_counter() {
    // SAFETY: the address is the architecturally fixed SysTick control
    // register; writing zero disables the counter and its interrupt.
    unsafe {
        core::ptr::write_volatile(SYST_CSR, 0);
    }
}

/// Reads whole ticks and the current counter with interrupts masked, so the
/// pair is a consistent snapshot.
#[inline]
fn read_counts() -> (u64, u32) {
    let primask: u32;
    let ticks: u64;
    let counter: u32;
    // SAFETY: `mrs`/`msr primask` only read and write the interrupt mask,
    // which is banked to the core; `TICKS`' only writer is the `SysTick`
    // handler, which cannot run while `PRIMASK` is set, so the two loads
    // observe a coherent pair. Omitting `nomem` makes each `asm!` a compiler
    // barrier, which is what keeps the loads inside the masked window; the
    // saved mask is restored unconditionally, so the caller's interrupt
    // state (including being inside an exception) survives.
    unsafe {
        core::arch::asm!(
            "mrs {primask}, primask",
            "cpsid i",
            primask = out(reg) primask,
            options(nostack),
        );
        ticks = *core::ptr::addr_of!(TICKS);
        counter = core::ptr::read_volatile(SYST_CVR);
        core::arch::asm!(
            "msr primask, {primask}",
            primask = in(reg) primask,
            options(nostack),
        );
    }
    (ticks, counter)
}

/// The core clock used when no agent is available to calibrate against.
///
/// Overridable per build with the `CODEVAR_CPU_HZ` environment variable
/// (read by this crate's `build.rs`), because a board without a debug probe
/// is exactly the case where nobody can measure it at run time.
const FALLBACK_HZ: u32 = match parse_hz(option_env!("CODEVAR_CPU_HZ")) {
    hz if hz >= MIN_PLAUSIBLE_HZ && hz <= MAX_PLAUSIBLE_HZ => hz,
    _ => DEFAULT_HZ,
};

/// Parses a decimal `u32`, yielding `0` for anything that is not a plain
/// positive number — a malformed `CODEVAR_CPU_HZ` silently falls back to the
/// default rather than failing the build.
const fn parse_hz(text: Option<&str>) -> u32 {
    let Some(text) = text else {
        return 0;
    };
    let bytes = text.as_bytes();
    let mut value: u32 = 0;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !byte.is_ascii_digit() || value > u32::MAX / 10 {
            return 0;
        }
        value = value * 10 + (byte - b'0') as u32;
        index += 1;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reload_divides_the_tick() {
        assert_eq!(tick_reload(12_000_000), 11_999);
        assert_eq!(tick_reload(1_000_000), 999);
    }

    #[test]
    fn very_slow_cores_keep_the_full_period() {
        assert_eq!(tick_reload(9_000), RELOAD_MAX);
    }

    #[test]
    fn hz_parsing_rejects_junk() {
        assert_eq!(parse_hz(None), 0);
        assert_eq!(parse_hz(Some("")), 0);
        assert_eq!(parse_hz(Some("48000000")), 48_000_000);
        assert_eq!(parse_hz(Some("12MHz")), 0);
        assert_eq!(parse_hz(Some("-1")), 0);
    }

    #[test]
    fn fallback_clock_is_always_a_clock() {
        // Whatever `CODEVAR_CPU_HZ` was at compile time, the fallback must
        // land inside the plausible band or the const `match` above would
        // have selected the documented default instead.
        assert!((MIN_PLAUSIBLE_HZ..=MAX_PLAUSIBLE_HZ).contains(&FALLBACK_HZ));
        assert_eq!(parse_hz(None), 0);
    }
}
