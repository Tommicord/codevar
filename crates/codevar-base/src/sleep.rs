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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Portable thread sleep for operating systems, `no_std` targets, and bare metal.
//!
//! None of them fit here directly: the first three need a hardware timer or
//! an async runtime, and the wasm ones cannot block the caller. This module
//! therefore talks to the OS on OS targets and degrades to a documented
//! no-op everywhere else.

use codevar_time_core::TimeDuration;
#[cfg(unix)]
use crate::time::{SystemTime, TimeVal};

/// Reason a [`sleep`] call could not complete.
///
/// The fallible variants can only be produced on Unix; Windows, wasm, and
/// bare-metal targets always return `Ok(())` because their sleep paths are
/// either infallible (`Sleep`) or a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepError {
    /// The platform sleep primitive rejected the request.
    ///
    /// On Unix the value is the error number returned by `clock_nanosleep`
    /// (for example `EINVAL` for an unsupported clock).
    Syscall(i32),
    /// The sleep was interrupted (typically by a POSIX signal) before the
    /// requested duration elapsed, and the remaining time could neither be
    /// measured from the monotonic clock nor recovered from the kernel.
    Interrupted,
}

impl core::fmt::Display for SleepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Syscall(code) => {
                write!(f, "sleep system call failed with error code {code}")
            }
            Self::Interrupted => f.write_str("sleep interrupted before the requested duration elapsed"),
        }
    }
}

impl core::error::Error for SleepError {}

/// Blocks the current thread for at least `duration`.
///
/// On operating systems the wait is delegated to the kernel scheduler, so
/// the CPU is released while sleeping instead of being spun away. The
/// duration is never rounded *down*: sub-millisecond requests become one
/// millisecond on Windows (the platform's typical timer granularity), and
/// Unix sleeps with nanosecond resolution. A zero-length duration returns
/// immediately without entering the kernel.
///
/// # Errors
///
/// Returns [`SleepError::Syscall`] when the platform sleep primitive
/// rejects the request, or [`SleepError::Interrupted`] when a sleep keeps
/// being interrupted and no remaining time can be proven. The no-op targets
/// (bare metal, wasm, unknown) never return an error.
///
/// # Examples
///
/// ```rust
/// use codevar_time_core::TimeDuration;
///
/// let slept = codevar_base::sleep::sleep(TimeDuration::from_millis(1));
/// assert!(slept.is_ok());
/// ```
#[inline]
pub fn sleep(duration: TimeDuration) -> Result<(), SleepError> {
    if duration.is_zero() {
        return Ok(());
    }
    sleep_platform(duration)
}

/// Blocks the current thread for at least `nanos` nanoseconds.
///
/// See [`sleep`] for platform behavior and error semantics.
///
/// # Examples
///
/// ```rust
/// let slept = codevar_base::sleep::sleep_nanos(1_000);
/// assert!(slept.is_ok());
/// ```
#[inline]
pub fn sleep_nanos(nanos: u64) -> Result<(), SleepError> {
    sleep(TimeDuration::from_nanos(nanos))
}

/// Blocks the current thread for at least `micros` microseconds.
///
/// See [`sleep`] for platform behavior and error semantics.
///
/// # Examples
///
/// ```rust
/// let slept = codevar_base::sleep::sleep_micros(1);
/// assert!(slept.is_ok());
/// ```
#[inline]
pub fn sleep_micros(micros: u64) -> Result<(), SleepError> {
    sleep(TimeDuration::from_micros(micros))
}

/// Blocks the current thread for at least `millis` milliseconds.
///
/// See [`sleep`] for platform behavior and error semantics.
///
/// # Examples
///
/// ```rust
/// let slept = codevar_base::sleep::sleep_millis(1);
/// assert!(slept.is_ok());
/// ```
#[inline]
pub fn sleep_millis(millis: u64) -> Result<(), SleepError> {
    sleep(TimeDuration::from_millis(millis))
}

/// Blocks the current thread for at least `secs` seconds.
///
/// See [`sleep`] for platform behavior and error semantics.
///
/// # Examples
///
/// ```rust
/// let slept = codevar_base::sleep::sleep_secs(0);
/// assert!(slept.is_ok());
/// ```
#[inline]
pub fn sleep_secs(secs: u64) -> Result<(), SleepError> {
    sleep(TimeDuration::from_secs(secs))
}

cfg_if::cfg_if! {
    if #[cfg(unix)] {
        fn sleep_platform(duration: TimeDuration) -> Result<(), SleepError> {
            unix::sleep(duration)
        }
    } else if #[cfg(windows)] {
        fn sleep_platform(duration: TimeDuration) -> Result<(), SleepError> {
            windows::sleep(duration)
        }
    } else {
        /// Bare metal (`target_os = "none"`), wasm, and unknown targets:
        /// there is no scheduler to suspend the thread on, so the wait is a
        /// documented no-op.
        #[inline]
        fn sleep_platform(_duration: TimeDuration) -> Result<(), SleepError> {
            Ok(())
        }
    }
}

/// Unix sleep implementation.
#[cfg(unix)]
mod unix {
    use codevar_time_core::TimeDuration;
    use crate::sleep::{to_timespec, SleepError, remaining_after_interrupt};
    use crate::time::SystemTime;

    /// Sleeps via `clock_nanosleep(CLOCK_MONOTONIC, …)`, resuming after signals.
    ///
    /// # Errors
    ///
    /// Returns [`SleepError::Syscall`] with the error number reported by
    /// `clock_nanosleep`, or [`SleepError::Interrupted`] when an `EINTR` cannot
    /// be resolved against either the monotonic clock or the kernel-reported
    /// leftover time.
    pub fn sleep(duration: TimeDuration) -> Result<(), SleepError> {
        let requested_nanos = duration.as_nanos();
        let started = SystemTime::monotonic().ok();
        let mut remaining = duration;

        loop {
            let request = to_timespec(remaining);
            let mut leftover = request;

            // SAFETY: `request` and `leftover` are valid `timespec` values that
            // live on this stack frame for the whole call; `clock_nanosleep`
            // only reads `request` and writes at most `leftover`, so the two
            // never alias and no other code touches them concurrently.
            let rc = unsafe { libc::clock_nanosleep(libc::CLOCK_MONOTONIC, 0, &request, &mut leftover) };

            if rc == 0 {
                return Ok(());
            }
            if rc != libc::EINTR {
                return Err(SleepError::Syscall(rc));
            }

            // The wait was interrupted: recompute what is left and sleep again
            // so the caller still observes a full-duration sleep.
            match remaining_after_interrupt(started, requested_nanos, &request, &leftover)? {
                Some(left) => remaining = left,
                None => return Ok(()),
            }
        }
    }
}

/// Computes the pending part of a sleep after an `EINTR`.
///
/// The monotonic clock is authoritative: while it works, the wait is
/// shortened by the time already elapsed, which also guarantees progress
/// on every retry. Otherwise, the kernel-reported `leftover` is used, but
/// only when it is a valid `timespec` that actually shrank.
///
/// # Returns
///
/// * `Ok(Some(left))` — sleep for `left` more (always shorter than before).
/// * `Ok(None)` — the requested duration is already satisfied.
/// * `Err(`[`SleepError::Interrupted`]`)` — no progress can be proven and
///   the caller must not spin.
///
/// # Errors
///
/// Returns [`SleepError::Interrupted`]; see above.
#[cfg(unix)]
fn remaining_after_interrupt(
    started: Option<TimeVal>,
    requested_nanos: u128,
    request: &libc::timespec,
    leftover: &libc::timespec,
) -> Result<Option<TimeDuration>, SleepError> {
    if let Some(start) = started
        && let Ok(elapsed) = SystemTime::elapsed_since(start)
    {
        let elapsed = u128::from(elapsed);
        if elapsed >= requested_nanos {
            return Ok(None);
        }
        return Ok(Some(nanos_to_duration(requested_nanos - elapsed)));
    }

    match (timespec_nanos(request), timespec_nanos(leftover)) {
        (Some(requested), Some(left)) if left < requested => Ok(Some(nanos_to_duration(left))),
        _ => Err(SleepError::Interrupted),
    }
}

/// Windows sleep implementation.
#[cfg(windows)]
mod windows {
    use codevar_time_core::TimeDuration;
    use crate::sleep::SleepError;

    /// Sleeps via `kernel32!Sleep`, rounding up to whole milliseconds.
    ///
    /// # Errors
    ///
    /// Always `Ok(())`: `Sleep` has no failure mode.
    pub fn sleep(duration: TimeDuration) -> Result<(), SleepError> {
        windows_link::link!(
        "kernel32.dll" "system"
        fn Sleep(dw_milliseconds: u32)
    );

        let mut millis = crate::sleep::millis_ceil(duration);
        while millis > u128::from(u32::MAX) {
            // SAFETY: `Sleep` accepts any timeout value and cannot fail.
            unsafe { crate::sleep::Sleep(u32::MAX) };
            millis -= u128::from(u32::MAX);
        }
        let tail = millis as u32;
        if tail > 0 {
            // SAFETY: `Sleep` accepts any timeout value and cannot fail.
            unsafe { crate::sleep::Sleep(tail) };
        }
        Ok(())
    }
}

/// Rounds `duration` up to whole milliseconds so `Sleep` never wakes early.
///
/// A non-zero sub-millisecond request becomes one millisecond; the result
/// fits in `u128` without overflow for any [`TimeDuration`].
#[cfg(windows)]
fn millis_ceil(duration: TimeDuration) -> u128 {
    const NANOS_PER_MILLI: u128 = 1_000_000;
    let nanos = duration.as_nanos();
    let millis = nanos / NANOS_PER_MILLI;
    if nanos % NANOS_PER_MILLI == 0 {
        millis
    } else {
        millis + 1
    }
}

/// Splits `duration` into the `timespec` fields expected by libc.
///
/// Seconds are clamped to the range of `libc::time_t` so durations beyond
/// the clock's representable range become `EINVAL` from the kernel instead
/// of a wrapped, negative value.
#[cfg(unix)]
fn to_timespec(duration: TimeDuration) -> libc::timespec {
    let secs = libc::time_t::try_from(duration.as_secs()).unwrap_or(libc::time_t::MAX);
    libc::timespec {
        tv_sec: secs,
        tv_nsec: duration.subsec_nanos() as libc::c_long,
    }
}

/// Reads a `timespec` as nanoseconds, or `None` if it is out of range.
///
/// Negative seconds/nanoseconds and `tv_nsec >= 1_000_000_000` are treated
/// as "no usable value" rather than being reinterpreted.
#[cfg(unix)]
fn timespec_nanos(ts: &libc::timespec) -> Option<u128> {
    let secs = i128::from(ts.tv_sec);
    let nanos = i128::from(ts.tv_nsec);
    if secs < 0 || !(0..1_000_000_000).contains(&nanos) {
        return None;
    }
    Some(secs as u128 * 1_000_000_000 + nanos as u128)
}

/// Converts a nanosecond budget into a [`TimeDuration`], saturating on overflow.
#[cfg(unix)]
fn nanos_to_duration(nanos: u128) -> TimeDuration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;
    let secs = u64::try_from(nanos / NANOS_PER_SEC).unwrap_or(u64::MAX);
    let subsec = u32::try_from(nanos % NANOS_PER_SEC).unwrap_or(0);
    TimeDuration::new(secs, subsec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::SystemTime;

    #[test]
    fn zero_duration_returns_immediately() {
        assert_eq!(sleep(TimeDuration::ZERO), Ok(()));
        assert_eq!(sleep_nanos(0), Ok(()));
        assert_eq!(sleep_micros(0), Ok(()));
        assert_eq!(sleep_millis(0), Ok(()));
        assert_eq!(sleep_secs(0), Ok(()));
    }

    #[test]
    fn sleeps_elapse_at_least_the_requested_duration() {
        let start = SystemTime::monotonic().unwrap();
        sleep(TimeDuration::from_millis(2)).unwrap();
        sleep_millis(2).unwrap();
        sleep_micros(1_000).unwrap();
        sleep_nanos(1_000_000).unwrap();
        let elapsed = SystemTime::elapsed_since(start).unwrap();
        assert!(
            elapsed >= 6_000_000,
            "elapsed {elapsed} ns is below the 6 ms requested"
        );
    }

    #[test]
    fn error_display_is_descriptive() {
        assert!(alloc::format!("{}", SleepError::Syscall(22)).contains("22"));
        assert!(alloc::format!("{}", SleepError::Interrupted).contains("interrupted"));
    }

    #[cfg(unix)]
    #[test]
    fn timespec_splits_seconds_and_nanos() {
        let ts = to_timespec(TimeDuration::new(42, 123_456_789));
        assert_eq!(ts.tv_sec as u64, 42);
        assert_eq!(ts.tv_nsec as u64, 123_456_789);
    }

    #[cfg(unix)]
    #[test]
    fn timespec_clamps_durations_beyond_the_clock_range() {
        let ts = to_timespec(TimeDuration::from_secs(u64::MAX));
        assert!(ts.tv_sec > 0, "clamped seconds must not wrap negative");
        assert_eq!(ts.tv_nsec, 0);
    }

    #[cfg(unix)]
    #[test]
    fn timespec_rejects_invalid_values() {
        let mut ts = libc::timespec {
            tv_sec: 1,
            tv_nsec: 1_000_000_000,
        };
        assert_eq!(timespec_nanos(&ts), None);
        ts.tv_sec = -1;
        ts.tv_nsec = 0;
        assert_eq!(timespec_nanos(&ts), None);
        ts.tv_sec = 1;
        ts.tv_nsec = 5;
        assert_eq!(timespec_nanos(&ts), Some(1_000_000_005));
    }

    #[cfg(unix)]
    #[test]
    fn nanos_to_duration_round_trips() {
        assert_eq!(nanos_to_duration(0), TimeDuration::ZERO);
        assert_eq!(nanos_to_duration(1_500_000_123), TimeDuration::new(1, 500_000_123));
        let max = nanos_to_duration(u128::MAX);
        assert_eq!(max, TimeDuration::new(u64::MAX, 768_211_455));
    }

    #[cfg(windows)]
    #[test]
    fn windows_millis_round_up_sub_millisecond_requests() {
        assert_eq!(millis_ceil(TimeDuration::ZERO), 0);
        assert_eq!(millis_ceil(TimeDuration::from_nanos(1)), 1);
        assert_eq!(millis_ceil(TimeDuration::from_millis(7)), 7);
        assert_eq!(millis_ceil(TimeDuration::from_nanos(1_000_001)), 2);
    }
}
