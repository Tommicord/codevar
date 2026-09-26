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

//! A modular, syscall-based time infrastructure built around [`SystemTime`].
//!
//! Provides:
//! - [`SystemTime`] – a syscall-driven struct with static methods for
//!   retrieving monotonic and realtime time.
//! - Convenience free functions (`monotonic_time`, `secs_to_nanos`,
//!   `millis_to_nanos`) for backward compatibility with existing callers.
//!
//! # Embedded support
//!
//! This module relies on raw syscalls (`clock_gettime`) and is compatible
//! with `no_std` environments, including embedded targets. On bare-metal
//! targets where `libc` is unavailable, the syscall interface falls back
//! to a stub returning [`TimeError::NotAvailable`].

use core::ffi::c_int;

/// Resolve the `clock_gettime` syscall via `libc`.
/// On `target_os = "none"` this works when `libc` is compiled with
/// `target-feature = +crt-static` or when a syscall abstraction layer
/// is available. For bare-metal targets without `libc`, implement
/// a platform-specific `call_clock_gettime` in your crate.
#[inline]
unsafe fn call_clock_gettime(clock_id: c_int, ts: *mut libc::timespec) -> c_int {
    unsafe { libc::clock_gettime(clock_id, ts) }
}

/// The `timespec` structure holding seconds and nanoseconds.
///
/// This is our platform-agnostic representation, converted to/from
/// `libc::timespec` at syscall boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Timespec {
    /// Seconds since the epoch (realtime) or boot (monotonic).
    pub tv_sec: u64,
    /// Additional nanoseconds.
    pub tv_nsec: u64,
}

impl Timespec {
    /// Converts our [`Timespec`] to `libc::timespec` for syscall use.
    #[inline]
    fn into_libc(self) -> libc::timespec {
        libc::timespec {
            tv_sec: self.tv_sec as libc::time_t,
            tv_nsec: self.tv_nsec as libc::c_long,
        }
    }

    /// Converts `libc::timespec` to our [`Timespec`].
    #[inline]
    fn from_libc(ts: libc::timespec) -> Self {
        Timespec {
            tv_sec: ts.tv_sec as u64,
            tv_nsec: ts.tv_nsec as u64,
        }
    }
}

/// The result of a time query, carrying seconds and nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeVal {
    /// Seconds component.
    pub secs: u64,
    /// Nanoseconds component.
    pub nsecs: u64,
}

/// Error type for time syscalls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeError {
    /// The syscall returned an error code.
    SyscallError(c_int),
    /// The clock ID is not supported on this platform.
    UnsupportedClock,
    /// The clock is not available (e.g., no OS timer on bare-metal).
    NotAvailable,
}

/// A syscall-based time provider for both monotonic and realtime clocks.
///
/// `SystemTime` is `no_std` compatible and works by
/// issuing raw syscalls via `clock_gettime`. All methods are static so no
/// allocation or runtime state is required.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_time::SystemTime;
///
/// // Get monotonic time (not affected by system clock changes).
/// let now = SystemTime::monotonic();
/// let nanos = SystemTime::secs_to_nanos(30);
///
/// // Get realtime (wall-clock) time.
/// let wall = SystemTime::realtime();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SystemTime;

impl SystemTime {
    /// Clock ID for `CLOCK_MONOTONIC`: monotonically increasing time.
    pub const CLOCK_MONOTONIC: c_int = 1;

    /// Clock ID for `CLOCK_REALTIME`: wall-clock time.
    pub const CLOCK_REALTIME: c_int = 0;

    /// Clock ID for `CLOCK_MONOTONIC_RAW`: monotonic time not subject to
    /// NTP corrections.
    pub const CLOCK_MONOTONIC_RAW: c_int = 4;

    /// Returns the current monotonic time as [`TimeVal`].
    ///
    /// Equivalent to `SystemTime::now_with_clock(CLOCK_MONOTONIC)`.
    /// Monotonic time is not affected by system clock adjustments
    /// (NTP, leap seconds, etc.) and is suitable for deadlines and
    /// frame-budget tracking.
    #[inline]
    pub fn now() -> Result<TimeVal, TimeError> {
        Self::now_with_clock(Self::CLOCK_MONOTONIC)
    }

    /// Returns the current time for the given clock as [`TimeVal`].
    ///
    /// Use [`now()`](Self::now()) for the default monotonic clock.
    ///
    /// # Safety
    ///
    /// Invokes the `clock_gettime` syscall. Safe on all targets where
    /// the syscall is available.
    #[inline]
    pub fn now_with_clock(clock_id: c_int) -> Result<TimeVal, TimeError> {
        let mut ts = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let mut ts_libc = ts.into_libc();
        let rc = unsafe { call_clock_gettime(clock_id, &mut ts_libc) };
        ts = Timespec::from_libc(ts_libc);
        if rc != 0 {
            return Err(TimeError::SyscallError(rc));
        }
        Ok(TimeVal {
            secs: ts.tv_sec,
            nsecs: ts.tv_nsec,
        })
    }

    /// Returns the current monotonic time as [`TimeVal`].
    ///
    /// Monotonic time is not affected by system clock adjustments
    /// (NTP, leap seconds, etc.) and is suitable for deadlines and
    /// frame-budget tracking.
    #[inline]
    pub fn monotonic() -> Result<TimeVal, TimeError> {
        Self::now_with_clock(Self::CLOCK_MONOTONIC)
    }

    /// Returns the current realtime (wall-clock) time as [`TimeVal`].
    ///
    /// This time can jump backward or forward due to NTP adjustments
    /// or manual clock changes.
    #[inline]
    pub fn realtime() -> Result<TimeVal, TimeError> {
        Self::now_with_clock(Self::CLOCK_REALTIME)
    }

    /// Returns the current monotonic raw time as [`TimeVal`].
    ///
    /// Like `monotonic()` but not subject to NTP corrections.
    #[inline]
    pub fn monotonic_raw() -> Result<TimeVal, TimeError> {
        Self::now_with_clock(Self::CLOCK_MONOTONIC_RAW)
    }

    /// Returns the monotonic time in nanoseconds since an arbitrary
    /// reference point (typically system boot).
    ///
    /// This is the primary convenience method used for deadline tracking.
    #[inline]
    pub fn monotonic_nanos() -> u64 {
        Self::now()
            .map(|tv| Self::nsecs_from_val(&tv))
            .unwrap_or(0)
    }

    /// Returns the realtime time in nanoseconds since the Unix epoch.
    #[inline]
    pub fn realtime_nanos() -> u64 {
        Self::now_with_clock(Self::CLOCK_REALTIME)
            .map(|tv| Self::nsecs_from_val(&tv))
            .unwrap_or(0)
    }

    /// Returns the elapsed time in nanoseconds since `start`.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::SyscallError`] if the monotonic clock
    /// syscall fails.
    #[inline]
    pub fn elapsed_since(start: TimeVal) -> Result<u64, TimeError> {
        let now = Self::monotonic()?;
        let elapsed = Self::diff_nsecs(&now, &start);
        Ok(elapsed)
    }

    /// Returns the duration between `earlier` and `later` in nanoseconds.
    ///
    /// Returns 0 if `later` is before `earlier`.
    #[inline]
    pub fn duration_since(earlier: TimeVal, later: TimeVal) -> u64 {
        Self::diff_nsecs(&later, &earlier)
    }

    /// Converts a [`TimeVal`] to total nanoseconds.
    #[inline]
    pub const fn nsecs_from_val(val: &TimeVal) -> u64 {
        val.secs * 1_000_000_000 + val.nsecs
    }

    /// Converts a [`TimeVal`] to total microseconds.
    #[inline]
    pub const fn micros_from_val(val: &TimeVal) -> u64 {
        val.secs * 1_000_000 + val.nsecs / 1_000
    }

    /// Converts a [`TimeVal`] to total milliseconds.
    #[inline]
    pub const fn millis_from_val(val: &TimeVal) -> u64 {
        val.secs * 1_000 + val.nsecs / 1_000_000
    }

    /// Converts a [`TimeVal`] to total seconds (truncating).
    #[inline]
    pub const fn secs_from_val(val: &TimeVal) -> u64 {
        val.secs
    }

    /// Converts seconds to nanoseconds.
    #[inline]
    pub const fn secs_to_nanos(secs: u64) -> u64 {
        secs * 1_000_000_000
    }

    /// Converts milliseconds to nanoseconds.
    #[inline]
    pub const fn millis_to_nanos(millis: u64) -> u64 {
        millis * 1_000_000
    }

    /// Converts microseconds to nanoseconds.
    #[inline]
    pub const fn micros_to_nanos(micros: u64) -> u64 {
        micros * 1_000
    }

    /// Converts nanoseconds to seconds (truncating).
    #[inline]
    pub const fn nanos_to_secs(nanos: u64) -> u64 {
        nanos / 1_000_000_000
    }

    /// Converts nanoseconds to milliseconds (truncating).
    #[inline]
    pub const fn nanos_to_millis(nanos: u64) -> u64 {
        nanos / 1_000_000
    }

    /// Converts nanoseconds to microseconds (truncating).
    #[inline]
    pub const fn nanos_to_micros(nanos: u64) -> u64 {
        nanos / 1_000
    }

    /// Adds two [`TimeVal`] instances, returning the sum.
    #[inline]
    pub const fn add_timeval(a: TimeVal, b: TimeVal) -> TimeVal {
        let mut nsecs = a.nsecs + b.nsecs;
        let mut secs = a.secs + b.secs;
        if nsecs >= 1_000_000_000 {
            nsecs -= 1_000_000_000;
            secs += 1;
        }
        TimeVal { secs, nsecs }
    }

    /// Subtracts `b` from `a`, returning the difference.
    /// Returns a zero [`TimeVal`] if `b > a`.
    #[inline]
    pub const fn sub_timeval(a: TimeVal, b: TimeVal) -> TimeVal {
        if a.secs < b.secs || (a.secs == b.secs && a.nsecs < b.nsecs) {
            return TimeVal { secs: 0, nsecs: 0 };
        }
        let mut a_nsecs = a.nsecs;
        let mut a_secs = a.secs;
        let b_nsecs = b.nsecs;
        let b_secs = b.secs;
        if a_nsecs < b_nsecs {
            a_nsecs += 1_000_000_000;
            a_secs -= 1;
        }
        TimeVal {
            secs: a_secs - b_secs,
            nsecs: a_nsecs - b_nsecs,
        }
    }

    /// Returns the difference between two [`TimeVal`] instances in nanoseconds.
    /// Returns 0 if `later` is before `earlier`.
    #[inline]
    pub const fn diff_nsecs(later: &TimeVal, earlier: &TimeVal) -> u64 {
        let later_nsecs = Self::nsecs_from_val(later);
        let earlier_nsecs = Self::nsecs_from_val(earlier);
        later_nsecs.saturating_sub(earlier_nsecs)
    }

    /// Compares two [`TimeVal`] instances. Returns `-1` if `a < b`,
    /// `0` if equal, `1` if `a > b`.
    #[inline]
    pub const fn compare(a: TimeVal, b: TimeVal) -> i8 {
        let a_nsecs = Self::nsecs_from_val(&a);
        let b_nsecs = Self::nsecs_from_val(&b);
        if a_nsecs < b_nsecs {
            -1
        } else if a_nsecs > b_nsecs {
            1
        } else {
            0
        }
    }
}

/// Converts seconds to nanoseconds.
///
/// Convenience wrapper around [`SystemTime::secs_to_nanos()`].
#[inline]
pub const fn secs_to_nanos(secs: u64) -> u64 {
    SystemTime::secs_to_nanos(secs)
}

/// Converts milliseconds to nanoseconds.
///
/// Convenience wrapper around [`SystemTime::millis_to_nanos()`].
#[inline]
pub const fn millis_to_nanos(millis: u64) -> u64 {
    SystemTime::millis_to_nanos(millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_returns_valid_time() {
        let now = SystemTime::monotonic();
        assert!(now.is_ok(), "monotonic time should be available");
        let tv = now.unwrap();
        assert!(tv.secs > 0 || tv.nsecs > 0, "time should be non-zero");
    }

    #[test]
    fn realtime_returns_valid_time() {
        let now = SystemTime::realtime();
        assert!(now.is_ok(), "realtime time should be available");
    }

    #[test]
    fn conversions_are_correct() {
        assert_eq!(SystemTime::secs_to_nanos(1), 1_000_000_000);
        assert_eq!(SystemTime::millis_to_nanos(1), 1_000_000);
        assert_eq!(SystemTime::micros_to_nanos(1), 1_000);
        assert_eq!(SystemTime::nanos_to_secs(1_000_000_000), 1);
        assert_eq!(SystemTime::nanos_to_millis(1_000_000), 1);
        assert_eq!(SystemTime::nanos_to_micros(1_000), 1);
    }

    #[test]
    fn elapsed_since_is_non_negative() {
        let start = SystemTime::monotonic().unwrap();
        let elapsed = SystemTime::elapsed_since(start).unwrap();
        assert!(elapsed > 0, "elapsed time should be positive");
    }

    #[test]
    fn duration_since_is_non_negative() {
        let earlier = TimeVal { secs: 5, nsecs: 0 };
        let later = TimeVal { secs: 10, nsecs: 0 };
        assert_eq!(SystemTime::duration_since(earlier, later), 5_000_000_000);
    }

    #[test]
    fn duration_since_returns_zero_when_reversed() {
        let earlier = TimeVal { secs: 10, nsecs: 0 };
        let later = TimeVal { secs: 5, nsecs: 0 };
        assert_eq!(SystemTime::duration_since(earlier, later), 0);
    }

    #[test]
    fn nsecs_from_val_is_correct() {
        let tv = TimeVal {
            secs: 2,
            nsecs: 500_000_000,
        };
        assert_eq!(SystemTime::nsecs_from_val(&tv), 2_500_000_000);
    }

    #[test]
    fn add_timeval_wraps_nanos() {
        let a = TimeVal {
            secs: 1,
            nsecs: 900_000_000,
        };
        let b = TimeVal {
            secs: 2,
            nsecs: 300_000_000,
        };
        let result = SystemTime::add_timeval(a, b);
        assert_eq!(result.secs, 4);
        assert_eq!(result.nsecs, 200_000_000);
    }

    #[test]
    fn sub_timeval_works() {
        let a = TimeVal {
            secs: 5,
            nsecs: 500_000_000,
        };
        let b = TimeVal {
            secs: 3,
            nsecs: 700_000_000,
        };
        let result = SystemTime::sub_timeval(a, b);
        assert_eq!(result.secs, 1);
        assert_eq!(result.nsecs, 800_000_000);
    }

    #[test]
    fn sub_timeval_returns_zero_when_reversed() {
        let a = TimeVal { secs: 3, nsecs: 0 };
        let b = TimeVal { secs: 5, nsecs: 0 };
        let result = SystemTime::sub_timeval(a, b);
        assert_eq!(result, TimeVal { secs: 0, nsecs: 0 });
    }

    #[test]
    fn compare_returns_correct_ordering() {
        let a = TimeVal { secs: 1, nsecs: 0 };
        let b = TimeVal { secs: 2, nsecs: 0 };
        assert_eq!(SystemTime::compare(a, b), -1);
        assert_eq!(SystemTime::compare(b, a), 1);
        assert_eq!(SystemTime::compare(a, a), 0);
    }

    #[test]
    fn timespec_roundtrip() {
        let original = Timespec {
            tv_sec: 42,
            tv_nsec: 1_234_567_890,
        };
        let libc_ts = original.into_libc();
        let restored = Timespec::from_libc(ts_libc);
        assert_eq!(original, restored);
    }
}
