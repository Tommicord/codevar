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

//! The arithmetic behind the SysTick clock, kept apart from its registers.
//!
//! Everything here is pure: how a core clock becomes a reload value, what a
//! plausible core clock looks like, and how `CODEVAR_CPU_HZ` is read. The
//! register-level side lives in `systick`, which only exists on bare-metal
//! ARM — so putting the arithmetic in its own module is what makes it
//! testable by an ordinary `cargo test` on the development machine, where no
//! SysTick exists to test against.

/// Largest 24-bit reload value: the longest one SysTick can hold, used while
/// calibrating and on cores too slow for a millisecond tick.
pub(crate) const RELOAD_MAX: u32 = 0x00FF_FFFF;
/// Nominal tick period: one millisecond.
pub(crate) const TARGET_TICK_HZ: u32 = 1_000;
/// A core clock below this cannot be divided into a millisecond, so the
/// maximum period is used instead of an interrupt storm.
pub(crate) const MIN_TICK_HZ: u32 = 10_000;
/// Core clock assumed when there is no agent to calibrate against: the
/// 12 MHz system clock of QEMU's `lm3s6965evb`, and a common board default.
pub(crate) const DEFAULT_HZ: u32 = 12_000_000;
/// Lower bound of a believable core clock; below it the measurement is
/// discarded rather than believed.
pub(crate) const MIN_PLAUSIBLE_HZ: u32 = 1_000;
/// Upper bound of a believable core clock.
pub(crate) const MAX_PLAUSIBLE_HZ: u32 = 400_000_000;

/// Reload value for a `1 / TARGET_TICK_HZ` second tick at `hz`.
///
/// Cores too slow to divide into a millisecond get the full 24-bit period
/// instead, which trades tick granularity for a sane interrupt rate — the
/// fractional part of a tick is read from the counter anyway, so uptime
/// keeps its precision either way. For example, a 12 MHz core gets
/// `11_999`, so the counter wraps 1000 times per second.
pub(crate) const fn tick_reload(hz: u32) -> u32 {
    if hz < MIN_TICK_HZ {
        RELOAD_MAX
    } else {
        (hz / TARGET_TICK_HZ) - 1
    }
}

/// The core clock used when no agent is available to calibrate against.
///
/// Overridable per build with the `CODEVAR_CPU_HZ` environment variable
/// (published by this crate's `build.rs`), because a board without a debug
/// probe is exactly the case where nobody can measure the clock at run time.
/// A value outside the plausible band falls back to [`DEFAULT_HZ`] rather
/// than being believed.
pub(crate) const FALLBACK_HZ: u32 = match parse_hz(option_env!("CODEVAR_CPU_HZ")) {
    hz if hz >= MIN_PLAUSIBLE_HZ && hz <= MAX_PLAUSIBLE_HZ => hz,
    _ => DEFAULT_HZ,
};

/// Compile-time guarantee that the fallback always answers with a
/// believable clock: whatever `CODEVAR_CPU_HZ` was at build time, the
/// `match` above selects [`DEFAULT_HZ`] when the value would land
/// outside the plausible band.
const _: () = assert!(FALLBACK_HZ >= MIN_PLAUSIBLE_HZ && FALLBACK_HZ <= MAX_PLAUSIBLE_HZ);

/// Parses a decimal `u32`, yielding `0` for anything that is not a plain
/// positive number — a malformed `CODEVAR_CPU_HZ` falls back to the default
/// instead of failing the build or, worse, compiling as a wrong clock.
const fn parse_hz(text: Option<&str>) -> u32 {
    let Some(text) = text else {
        return 0;
    };
    let bytes = text.as_bytes();
    let mut value: u32 = 0;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !byte.is_ascii_digit() {
            return 0;
        }
        let digit = (byte - b'0') as u32;
        // Reject before multiplying: a 10-digit number must fail rather than
        // wrap, and `u32::from` is not const-stable on the MSRV yet.
        if value > (u32::MAX - digit) / 10 {
            return 0;
        }
        value = value * 10 + digit;
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
        // The slowest core that still gets a millisecond tick: 10 periods
        // per millisecond, so a reload of 9.
        assert_eq!(tick_reload(MIN_TICK_HZ), 9);
    }

    #[test]
    fn very_slow_cores_keep_the_full_period() {
        assert_eq!(tick_reload(MIN_TICK_HZ - 1), RELOAD_MAX);
        assert_eq!(tick_reload(0), RELOAD_MAX);
    }

    #[test]
    fn hz_parsing_rejects_junk() {
        assert_eq!(parse_hz(None), 0);
        assert_eq!(parse_hz(Some("")), 0);
        assert_eq!(parse_hz(Some("48000000")), 48_000_000);
        assert_eq!(parse_hz(Some("12MHz")), 0);
        assert_eq!(parse_hz(Some("-1")), 0);
        assert_eq!(parse_hz(Some("12_000_000")), 0);
    }

    #[test]
    fn overflow_is_rejected_not_wrapped() {
        // One digit too many must not silently become a smaller number.
        assert_eq!(parse_hz(Some("42949672960")), 0);
        assert_eq!(parse_hz(Some("4294967295")), u32::MAX);
    }
}
