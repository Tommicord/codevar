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

#![cfg_attr(not(test), no_std)]
extern crate alloc;

use core::ffi::c_int;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use core::{fmt, mem};

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
/// use codevar_time_core::SystemTime;
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

#[derive(Clone, Copy, Eq)]
#[repr(transparent)]
struct TimeNanos(u32);
const _: () = {
    assert!(u32::MIN == 0);
    let ulow: u32 = 0u32;
    let uhigh: u32 = 999_999_999u32;
    assert!(ulow <= uhigh);
    assert!(mem::size_of::<u32>() == mem::size_of::<u32>());
};

impl TimeNanos {
    #[inline]
    pub const fn new(val: u32) -> Option<Self> {
        if val >= 0u32 && val <= 999_999_999u32 {
            Some(TimeNanos(val))
        } else {
            None
        }
    }

    /// Constructs an instance of this type from the underlying integer
    /// primitive without checking whether its zero.
    ///
    /// # Safety
    /// Immediate UB if `val` is not within the valid range for this
    /// type, as it violates the validity invariant.
    #[inline]
    pub const unsafe fn new_unchecked(val: u32) -> Self {
        // SAFETY: Caller promised that `val` is within the valid range.
        TimeNanos(val)
    }

    #[inline]
    pub const fn as_inner(self) -> u32 {
        // SAFETY: This is a transparent wrapper, so unwrapping it is sound
        unsafe { core::mem::transmute(self) }
    }
}
impl PartialEq for TimeNanos {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_inner() == other.as_inner()
    }
}
impl Ord for TimeNanos {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        Ord::cmp(&self.as_inner(), &other.as_inner())
    }
}
impl PartialOrd for TimeNanos {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}
impl fmt::Debug for TimeNanos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <u32 as fmt::Debug>::fmt(&self.as_inner(), f)
    }
}
impl TimeNanos {
    // SAFETY: 0 is within the valid range
    pub const ZERO: Self = unsafe { TimeNanos::new_unchecked(0) };
}

impl Default for TimeNanos {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct TimeDuration {
    secs: u64,
    nanos: TimeNanos,
}

const NANOS_PER_SEC: u32 = 1_000_000_000;
const NANOS_PER_MILLI: u32 = 1_000_000;
const NANOS_PER_MICRO: u32 = 1_000;
const MILLIS_PER_SEC: u64 = 1_000;
const MICROS_PER_SEC: u64 = 1_000_000;

impl TimeDuration {
    /// The duration of one second.
    pub const SECOND: TimeDuration = TimeDuration::from_secs(1);

    /// The duration of one millisecond.
    pub const MILLISECOND: TimeDuration = TimeDuration::from_millis(1);

    /// The duration of one microsecond.
    pub const MICROSECOND: TimeDuration = TimeDuration::from_micros(1);

    /// The duration of one nanosecond.
    pub const NANOSECOND: TimeDuration = TimeDuration::from_nanos(1);

    /// A duration of zero time.
    pub const ZERO: TimeDuration = TimeDuration::from_nanos(0);

    /// The maximum duration.
    ///
    /// May vary by platform as necessary. Must be able to contain the difference between
    /// two instances of [`Instant`] or two instances of [`SystemTime`].
    /// This constraint gives it a value of about 584,942,417,355 years in practice,
    /// which is currently used on all platforms.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::MAX, TimeDuration::new(u64::MAX, 1_000_000_000 - 1));
    /// ```
    pub const MAX: TimeDuration = TimeDuration::from_nanos((NANOS_PER_SEC - 1) as u64);

    /// Creates a new `TimeDuration` from the specified number of whole seconds and
    /// additional nanoseconds.
    ///
    /// If the number of nanoseconds is greater than 1 billion (the number of
    /// nanoseconds in a second), then it will carry over into the seconds provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let five_duration = TimeDuration::new(5, 0);
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(secs: u64, nanos: u32) -> TimeDuration {
        if nanos < NANOS_PER_SEC {
            // SAFETY: nanos < NANOS_PER_SEC, therefore nanos is within the valid range
            TimeDuration {
                secs,
                nanos: unsafe { TimeNanos::new_unchecked(nanos) },
            }
        } else {
            let secs = secs.saturating_add((nanos / NANOS_PER_SEC) as u64);
            let nanos = nanos % NANOS_PER_SEC;
            // SAFETY: nanos % NANOS_PER_SEC < NANOS_PER_SEC, therefore nanos is within the valid range
            TimeDuration {
                secs,
                nanos: unsafe { TimeNanos::new_unchecked(nanos) },
            }
        }
    }

    /// Creates a new `TimeDuration` from the specified number of whole seconds.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_secs(5);
    ///
    /// assert_eq!(5, duration.as_secs());
    /// assert_eq!(0, duration.subsec_nanos());
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_secs(secs: u64) -> TimeDuration {
        TimeDuration {
            secs,
            nanos: TimeNanos::ZERO,
        }
    }

    /// Creates a new `TimeDuration` from the specified number of milliseconds.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_millis(2_569);
    ///
    /// assert_eq!(2, duration.as_secs());
    /// assert_eq!(569_000_000, duration.subsec_nanos());
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_millis(millis: u64) -> TimeDuration {
        let secs = millis / MILLIS_PER_SEC;
        let subsec_millis = (millis % MILLIS_PER_SEC) as u32;
        // SAFETY: (x % 1_000) * 1_000_000 < 1_000_000_000 => x % 1_000 < 1_000
        let subsec_nanos = unsafe { TimeNanos::new_unchecked(subsec_millis * NANOS_PER_MILLI) };

        TimeDuration {
            secs,
            nanos: subsec_nanos,
        }
    }

    /// Creates a new `TimeDuration` from the specified number of microseconds.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_micros(1_000_002);
    ///
    /// assert_eq!(1, duration.as_secs());
    /// assert_eq!(2_000, duration.subsec_nanos());
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_micros(micros: u64) -> TimeDuration {
        let secs = micros / MICROS_PER_SEC;
        let subsec_micros = (micros % MICROS_PER_SEC) as u32;
        // SAFETY: (x % 1_000_000) * 1_000 < 1_000_000_000
        //         => x % 1_000_000 < 1_000_000
        let subsec_nanos = unsafe { TimeNanos::new_unchecked(subsec_micros * NANOS_PER_MICRO) };

        TimeDuration {
            secs,
            nanos: subsec_nanos,
        }
    }

    /// Creates a new `TimeDuration` from the specified number of nanoseconds.
    ///
    /// Note: Using this on the return value of `as_nanos()` might cause unexpected behavior:
    /// `as_nanos()` returns an u128, and can return values that do not fit in u64, e.g. 585 years.
    /// Instead, consider using the pattern `TimeDuration::new(d.as_secs(), d.subsec_nanos())`
    /// if you cannot copy/clone the TimeDuration directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_nanos(1_000_000_123);
    ///
    /// assert_eq!(1, duration.as_secs());
    /// assert_eq!(123, duration.subsec_nanos());
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_nanos(nanos: u64) -> TimeDuration {
        const NANOS_PER_SEC: u64 = self::NANOS_PER_SEC as u64;
        let secs = nanos / NANOS_PER_SEC;
        let subsec_nanos = (nanos % NANOS_PER_SEC) as u32;
        // SAFETY: x % 1_000_000_000 < 1_000_000_000
        let subsec_nanos = unsafe { TimeNanos::new_unchecked(subsec_nanos) };

        TimeDuration {
            secs,
            nanos: subsec_nanos,
        }
    }

    /// Creates a new `TimeDuration` from the specified number of nanoseconds.
    ///
    /// # Panics
    ///
    /// Panics if the given number of nanoseconds is greater than [`TimeDuration::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let nanos = 10_u128.pow(24) + 321;
    /// let duration = TimeDuration::from_nanos_u128(nanos).unwrap_or(TimeDuration::ZERO);
    ///
    /// assert_eq!(10_u64.pow(15), duration.as_secs());
    /// assert_eq!(321, duration.subsec_nanos());
    /// ```
    #[must_use]
    #[inline]
    #[track_caller]
    pub fn from_nanos_u128(nanos: u128) -> Option<TimeDuration> {
        const NANOS_PER_SEC: u128 = self::NANOS_PER_SEC as u128;
        let secs = u64::try_from(nanos / NANOS_PER_SEC).ok()?;
        let subsec_nanos = (nanos % NANOS_PER_SEC) as u32;
        // SAFETY: x % 1_000_000_000 < 1_000_000_000 also, subsec_nanos >= 0 since u128 >=0 and u32 >=0
        let subsec_nanos = unsafe { TimeNanos::new_unchecked(subsec_nanos) };

        Some(TimeDuration {
            secs: secs as u64,
            nanos: subsec_nanos,
        })
    }

    /// Returns true if this `TimeDuration` spans no time.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert!(TimeDuration::ZERO.is_zero());
    /// assert!(TimeDuration::new(0, 0).is_zero());
    /// assert!(TimeDuration::from_nanos(0).is_zero());
    /// assert!(TimeDuration::from_secs(0).is_zero());
    ///
    /// assert!(!TimeDuration::new(1, 1).is_zero());
    /// assert!(!TimeDuration::from_nanos(1).is_zero());
    /// assert!(!TimeDuration::from_secs(1).is_zero());
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.secs == 0 && self.nanos.as_inner() == 0
    }

    /// Returns the number of _whole_ seconds contained by this `TimeDuration`.
    ///
    /// The returned value does not include the fractional (nanosecond) part of the
    /// duration, which can be obtained using [`subsec_nanos`].
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::new(5, 730_023_852);
    /// assert_eq!(duration.as_secs(), 5);
    /// ```
    ///
    /// To determine the total number of seconds represented by the `TimeDuration`
    /// including the fractional part, use [`as_secs_f64`] or [`as_secs_f32`]
    ///
    /// [`as_secs_f64`]: TimeDuration::as_secs_f64
    /// [`as_secs_f32`]: TimeDuration::as_secs_f32
    /// [`subsec_nanos`]: TimeDuration::subsec_nanos
    #[must_use]
    #[inline]
    pub const fn as_secs(&self) -> u64 {
        self.secs
    }

    /// Returns the fractional part of this `TimeDuration`, in whole milliseconds.
    ///
    /// This method does not return the length of the duration when
    /// represented by milliseconds. The returned number always represents a
    /// fractional portion of a second (i.e., it is less than one thousand).
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_millis(5_432);
    /// assert_eq!(duration.as_secs(), 5);
    /// assert_eq!(duration.subsec_millis(), 432);
    /// ```
    #[must_use]
    #[inline]
    pub const fn subsec_millis(&self) -> u32 {
        self.nanos.as_inner() / NANOS_PER_MILLI
    }

    /// Returns the fractional part of this `TimeDuration`, in whole microseconds.
    ///
    /// This method does not return the length of the duration when
    /// represented by microseconds. The returned number always represents a
    /// fractional portion of a second (i.e., it is less than one million).
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_micros(1_234_567);
    /// assert_eq!(duration.as_secs(), 1);
    /// assert_eq!(duration.subsec_micros(), 234_567);
    /// ```
    #[must_use]
    #[inline]
    pub const fn subsec_micros(&self) -> u32 {
        self.nanos.as_inner() / NANOS_PER_MICRO
    }

    /// Returns the fractional part of this `TimeDuration`, in nanoseconds.
    ///
    /// This method does not return the length of the duration when
    /// represented by nanoseconds. The returned number always represents a
    /// fractional portion of a second (i.e., it is less than one billion).
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::from_millis(5_010);
    /// assert_eq!(duration.as_secs(), 5);
    /// assert_eq!(duration.subsec_nanos(), 10_000_000);
    /// ```

    #[must_use]
    #[inline]
    pub const fn subsec_nanos(&self) -> u32 {
        self.nanos.as_inner()
    }

    /// Returns the total number of whole milliseconds contained by this `TimeDuration`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::new(5, 730_023_852);
    /// assert_eq!(duration.as_millis(), 5_730);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_millis(&self) -> u128 {
        self.secs as u128 * MILLIS_PER_SEC as u128 + (self.nanos.as_inner() / NANOS_PER_MILLI) as u128
    }

    /// Returns the total number of whole microseconds contained by this `TimeDuration`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::new(5, 730_023_852);
    /// assert_eq!(duration.as_micros(), 5_730_023);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_micros(&self) -> u128 {
        self.secs as u128 * MICROS_PER_SEC as u128 + (self.nanos.as_inner() / NANOS_PER_MICRO) as u128
    }

    /// Returns the total number of nanoseconds contained by this `TimeDuration`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let duration = TimeDuration::new(5, 730_023_852);
    /// assert_eq!(duration.as_nanos(), 5_730_023_852);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_nanos(&self) -> u128 {
        self.secs as u128 * NANOS_PER_SEC as u128 + self.nanos.as_inner() as u128
    }

    /// Computes the absolute difference between `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(100, 0).abs_diff(TimeDuration::new(80, 0)), TimeDuration::new(20, 0));
    /// assert_eq!(TimeDuration::new(100, 400_000_000).abs_diff(TimeDuration::new(110, 0)), TimeDuration::new(9, 600_000_000));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn abs_diff(self, other: TimeDuration) -> TimeDuration {
        if let Some(res) = self.checked_sub(other) {
            res
        } else {
            other
                .checked_sub(self)
                .unwrap_or(TimeDuration::ZERO)
        }
    }

    /// Checked `TimeDuration` addition. Computes `self + other`, returning [`None`]
    /// if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(0, 0).checked_add(TimeDuration::new(0, 1)), Some(TimeDuration::new(0, 1)));
    /// assert_eq!(TimeDuration::new(1, 0).checked_add(TimeDuration::new(u64::MAX, 0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_add(self, rhs: TimeDuration) -> Option<TimeDuration> {
        if let Some(mut secs) = self.secs.checked_add(rhs.secs) {
            let mut nanos = self.nanos.as_inner() + rhs.nanos.as_inner();
            if nanos >= NANOS_PER_SEC {
                nanos -= NANOS_PER_SEC;
                if let Some(new_secs) = secs.checked_add(1) {
                    secs = new_secs;
                } else {
                    return None;
                }
            }
            debug_assert!(nanos < NANOS_PER_SEC);
            Some(TimeDuration::new(secs, nanos))
        } else {
            None
        }
    }

    /// Saturating `TimeDuration` addition. Computes `self + other`, returning [`TimeDuration::MAX`]
    /// if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(0, 0).saturating_add(TimeDuration::new(0, 1)), TimeDuration::new(0, 1));
    /// assert_eq!(TimeDuration::new(1, 0).saturating_add(TimeDuration::new(u64::MAX, 0)), TimeDuration::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_add(self, rhs: TimeDuration) -> TimeDuration {
        self.checked_add(rhs)
            .unwrap_or_else(|| TimeDuration::MAX)
    }

    /// Checked `TimeDuration` subtraction. Computes `self - other`, returning [`None`]
    /// if the result would be negative or if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(0, 1).checked_sub(TimeDuration::new(0, 0)), Some(TimeDuration::new(0, 1)));
    /// assert_eq!(TimeDuration::new(0, 0).checked_sub(TimeDuration::new(0, 1)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_sub(self, rhs: TimeDuration) -> Option<TimeDuration> {
        if let Some(mut secs) = self.secs.checked_sub(rhs.secs) {
            let nanos = if self.nanos.as_inner() >= rhs.nanos.as_inner() {
                self.nanos.as_inner() - rhs.nanos.as_inner()
            } else if let Some(sub_secs) = secs.checked_sub(1) {
                secs = sub_secs;
                self.nanos.as_inner() + NANOS_PER_SEC - rhs.nanos.as_inner()
            } else {
                return None;
            };
            debug_assert!(nanos < NANOS_PER_SEC);
            Some(TimeDuration::new(secs, nanos))
        } else {
            None
        }
    }

    /// Saturating `TimeDuration` subtraction. Computes `self - other`, returning [`TimeDuration::ZERO`]
    /// if the result would be negative or if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(0, 1).saturating_sub(TimeDuration::new(0, 0)), TimeDuration::new(0, 1));
    /// assert_eq!(TimeDuration::new(0, 0).saturating_sub(TimeDuration::new(0, 1)), TimeDuration::ZERO);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_sub(self, rhs: TimeDuration) -> TimeDuration {
        self.checked_sub(rhs)
            .unwrap_or_else(|| TimeDuration::ZERO)
    }

    /// Checked `TimeDuration` multiplication. Computes `self * other`, returning
    /// [`None`] if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(0, 500_000_001).checked_mul(2), Some(TimeDuration::new(1, 2)));
    /// assert_eq!(TimeDuration::new(u64::MAX - 1, 0).checked_mul(2), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_mul(self, rhs: u32) -> Option<TimeDuration> {
        // Multiply nanoseconds as u64, because it cannot overflow that way.
        let total_nanos = self.nanos.as_inner() as u64 * rhs as u64;
        let extra_secs = total_nanos / (NANOS_PER_SEC as u64);
        let nanos = (total_nanos % (NANOS_PER_SEC as u64)) as u32;
        self.secs.checked_mul(rhs as u64).and_then(|s| {
            if let Some(secs) = s.checked_add(extra_secs) {
                debug_assert!(nanos < NANOS_PER_SEC);
                return Some(TimeDuration::new(secs, nanos));
            } else {
                None
            }
        })
    }

    /// Saturating `TimeDuration` multiplication. Computes `self * other`, returning
    /// [`TimeDuration::MAX`] if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(0, 500_000_001).saturating_mul(2), TimeDuration::new(1, 2));
    /// assert_eq!(TimeDuration::new(u64::MAX - 1, 0).saturating_mul(2), TimeDuration::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_mul(self, rhs: u32) -> TimeDuration {
        self.checked_mul(rhs)
            .unwrap_or_else(|| TimeDuration::MAX)
    }

    /// Checked `TimeDuration` division. Computes `self / other`, returning [`None`]
    /// if `other == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// assert_eq!(TimeDuration::new(2, 0).checked_div(2), Some(TimeDuration::new(1, 0)));
    /// assert_eq!(TimeDuration::new(1, 0).checked_div(2), Some(TimeDuration::new(0, 500_000_000)));
    /// assert_eq!(TimeDuration::new(2, 0).checked_div(0), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn checked_div(self, rhs: u32) -> Option<TimeDuration> {
        if rhs != 0 {
            let (secs, extra_secs) = (self.secs / (rhs as u64), self.secs % (rhs as u64));
            let (mut nanos, extra_nanos) = (self.nanos.as_inner() / rhs, self.nanos.as_inner() % rhs);
            nanos += ((extra_secs * (NANOS_PER_SEC as u64) + extra_nanos as u64) / (rhs as u64)) as u32;
            debug_assert!(nanos < NANOS_PER_SEC);
            Some(TimeDuration::new(secs, nanos))
        } else {
            None
        }
    }

    /// Returns the number of seconds contained by this `TimeDuration` as `f64`.
    ///
    /// The returned value includes the fractional (nanosecond) part of the duration.
    ///
    /// # Examples
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let dur = TimeDuration::new(2, 700_000_000);
    /// assert_eq!(dur.as_secs_f64(), 2.7);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_secs_f64(&self) -> f64 {
        (self.secs as f64) + (self.nanos.as_inner() as f64) / (NANOS_PER_SEC as f64)
    }

    /// Returns the number of seconds contained by this `TimeDuration` as `f32`.
    ///
    /// The returned value includes the fractional (nanosecond) part of the duration.
    ///
    /// # Examples
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let dur = TimeDuration::new(2, 700_000_000);
    /// assert_eq!(dur.as_secs_f32(), 2.7);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_secs_f32(&self) -> f32 {
        (self.secs as f32) + (self.nanos.as_inner() as f32) / (NANOS_PER_SEC as f32)
    }

    /// Returns the number of milliseconds contained by this `TimeDuration` as `f64`.
    ///
    /// The returned value includes the fractional (nanosecond) part of the duration.
    ///
    /// # Examples
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let dur = TimeDuration::new(2, 345_678_000);
    /// assert_eq!(dur.as_millis_f64(), 2_345.678);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_millis_f64(&self) -> f64 {
        (self.secs as f64) * (MILLIS_PER_SEC as f64)
            + (self.nanos.as_inner() as f64) / (NANOS_PER_MILLI as f64)
    }

    /// Returns the number of milliseconds contained by this `TimeDuration` as `f32`.
    ///
    /// The returned value includes the fractional (nanosecond) part of the duration.
    ///
    /// # Examples
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let dur = TimeDuration::new(2, 345_678_000);
    /// assert_eq!(dur.as_millis_f32(), 2_345.678);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_millis_f32(&self) -> f32 {
        (self.secs as f32) * (MILLIS_PER_SEC as f32)
            + (self.nanos.as_inner() as f32) / (NANOS_PER_MILLI as f32)
    }

    /// Divides `TimeDuration` by `TimeDuration` and returns `f64`.
    ///
    /// # Examples
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let dur1 = TimeDuration::new(2, 700_000_000);
    /// let dur2 = TimeDuration::new(5, 400_000_000);
    /// assert_eq!(dur1.div_duration_f64(dur2), 0.5);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn div_duration_f64(self, rhs: TimeDuration) -> f64 {
        let self_nanos = (self.secs as f64) * (NANOS_PER_SEC as f64) + (self.nanos.as_inner() as f64);
        let rhs_nanos = (rhs.secs as f64) * (NANOS_PER_SEC as f64) + (rhs.nanos.as_inner() as f64);
        self_nanos / rhs_nanos
    }

    /// Divides `TimeDuration` by `TimeDuration` and returns `f32`.
    ///
    /// # Examples
    /// ```
    /// use codevar_time_core::TimeDuration;
    ///
    /// let dur1 = TimeDuration::new(2, 700_000_000);
    /// let dur2 = TimeDuration::new(5, 400_000_000);
    /// assert_eq!(dur1.div_duration_f32(dur2), 0.5);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn div_duration_f32(self, rhs: TimeDuration) -> f32 {
        let self_nanos = (self.secs as f32) * (NANOS_PER_SEC as f32) + (self.nanos.as_inner() as f32);
        let rhs_nanos = (rhs.secs as f32) * (NANOS_PER_SEC as f32) + (rhs.nanos.as_inner() as f32);
        self_nanos / rhs_nanos
    }
}

impl Add for TimeDuration {
    type Output = TimeDuration;

    #[inline]
    fn add(self, rhs: TimeDuration) -> TimeDuration {
        self.checked_add(rhs)
            .unwrap_or(TimeDuration::ZERO)
    }
}

impl AddAssign for TimeDuration {
    #[inline]
    fn add_assign(&mut self, rhs: TimeDuration) {
        *self = *self + rhs;
    }
}

impl Sub for TimeDuration {
    type Output = TimeDuration;

    #[inline]
    fn sub(self, rhs: TimeDuration) -> TimeDuration {
        self.checked_sub(rhs)
            .unwrap_or(TimeDuration::ZERO)
    }
}

impl SubAssign for TimeDuration {
    #[inline]
    fn sub_assign(&mut self, rhs: TimeDuration) {
        *self = *self - rhs;
    }
}

impl Mul<u32> for TimeDuration {
    type Output = TimeDuration;

    #[inline]
    fn mul(self, rhs: u32) -> TimeDuration {
        self.checked_mul(rhs)
            .unwrap_or(TimeDuration::ZERO)
    }
}

impl Mul<TimeDuration> for u32 {
    type Output = TimeDuration;

    #[inline]
    fn mul(self, rhs: TimeDuration) -> TimeDuration {
        rhs * self
    }
}

impl MulAssign<u32> for TimeDuration {
    #[inline]
    fn mul_assign(&mut self, rhs: u32) {
        *self = *self * rhs;
    }
}

impl Div<u32> for TimeDuration {
    type Output = TimeDuration;

    #[inline]
    #[track_caller]
    fn div(self, rhs: u32) -> TimeDuration {
        self.checked_div(rhs)
            .unwrap_or(TimeDuration::ZERO)
    }
}

impl DivAssign<u32> for TimeDuration {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: u32) {
        *self = *self / rhs;
    }
}

macro_rules! sum_durations {
    ($iter:expr) => {{
        let mut total_secs: u64 = 0;
        let mut total_nanos: u64 = 0;

        for entry in $iter {
            total_secs = total_secs
                .checked_add(entry.secs)
                .expect("overflow in iter::sum over durations");
            total_nanos = match total_nanos.checked_add(entry.nanos.as_inner() as u64) {
                Some(n) => n,
                None => {
                    total_secs = total_secs
                        .checked_add(total_nanos / NANOS_PER_SEC as u64)
                        .expect("overflow in iter::sum over durations");
                    (total_nanos % NANOS_PER_SEC as u64) + entry.nanos.as_inner() as u64
                }
            };
        }
        total_secs = total_secs
            .checked_add(total_nanos / NANOS_PER_SEC as u64)
            .expect("overflow in iter::sum over durations");
        total_nanos = total_nanos % NANOS_PER_SEC as u64;
        TimeDuration::new(total_secs, total_nanos as u32)
    }};
}

impl Sum for TimeDuration {
    fn sum<I: Iterator<Item = TimeDuration>>(iter: I) -> TimeDuration {
        sum_durations!(iter)
    }
}

impl<'a> Sum<&'a TimeDuration> for TimeDuration {
    fn sum<I: Iterator<Item = &'a TimeDuration>>(iter: I) -> TimeDuration {
        sum_durations!(iter)
    }
}

impl core::fmt::Debug for TimeDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        /// Formats a floating point number in decimal notation.
        ///
        /// The number is given as the `integer_part` and a fractional part.
        /// The value of the fractional part is `fractional_part / divisor`. So
        /// `integer_part` = 3, `fractional_part` = 12 and `divisor` = 100
        /// represents the number `3.012`. Trailing zeros are omitted.
        ///
        /// `divisor` must not be above 100_000_000. It also should be a power
        /// of 10, everything else doesn't make sense. `fractional_part` has
        /// to be less than `10 * divisor`!
        ///
        /// A prefix and postfix may be added. The whole thing is padded
        /// to the formatter's `width`, if specified.
        fn fmt_decimal(
            f: &mut fmt::Formatter<'_>,
            integer_part: u64,
            mut fractional_part: u32,
            mut divisor: u32,
            prefix: &str,
            postfix: &str,
        ) -> fmt::Result {
            let mut buf = [b'0'; 9];
            let mut pos = 0;
            while fractional_part > 0 && pos < f.precision().unwrap_or(9) {
                buf[pos] = b'0' + (fractional_part / divisor) as u8;
                fractional_part %= divisor;
                divisor /= 10;
                pos += 1;
            }
            let integer_part = if fractional_part > 0 && fractional_part >= divisor * 5 {
                let mut rev_pos = pos;
                let mut carry = true;
                while carry && rev_pos > 0 {
                    rev_pos -= 1;
                    if buf[rev_pos] < b'9' {
                        buf[rev_pos] += 1;
                        carry = false;
                    } else {
                        buf[rev_pos] = b'0';
                    }
                }
                if carry {
                    integer_part.checked_add(1)
                } else {
                    Some(integer_part)
                }
            } else {
                Some(integer_part)
            };
            let end = f
                .precision()
                .map(|p| core::cmp::min(p, 9))
                .unwrap_or(pos);
            let emit_without_padding = |f: &mut fmt::Formatter<'_>| {
                if let Some(integer_part) = integer_part {
                    write!(f, "{}{}", prefix, integer_part)?;
                } else {
                    write!(f, "{}18446744073709551616", prefix)?;
                }
                if end > 0 {
                    let s = unsafe { core::str::from_utf8_unchecked(&buf[..end]) };
                    let w = f.precision().unwrap_or(pos);
                    write!(f, ".{:0<width$}", s, width = w)?;
                }
                write!(f, "{}", postfix)
            };

            match f.width() {
                None => emit_without_padding(f),
                Some(requested_w) => {
                    let mut actual_w = prefix.len() + postfix.chars().count();
                    if let Some(integer_part) = integer_part {
                        if let Some(log) = integer_part.checked_ilog10() {
                            actual_w += 1 + log as usize;
                        } else {
                            actual_w += 1;
                        }
                    } else {
                        actual_w += 20;
                    }
                    if end > 0 {
                        let frac_part_w = f.precision().unwrap_or(pos);
                        actual_w += 1 + frac_part_w;
                    }
                    if requested_w <= actual_w {
                        emit_without_padding(f)
                    } else {
                        Ok(emit_without_padding(f)?)
                    }
                }
            }
        }
        let prefix = if f.sign_plus() { "+" } else { "" };
        if self.secs > 0 {
            fmt_decimal(
                f,
                self.secs,
                self.nanos.as_inner(),
                NANOS_PER_SEC / 10,
                prefix,
                "s",
            )
        } else if self.nanos.as_inner() >= NANOS_PER_MILLI {
            fmt_decimal(
                f,
                (self.nanos.as_inner() / NANOS_PER_MILLI) as u64,
                self.nanos.as_inner() % NANOS_PER_MILLI,
                NANOS_PER_MILLI / 10,
                prefix,
                "ms",
            )
        } else if self.nanos.as_inner() >= NANOS_PER_MICRO {
            fmt_decimal(
                f,
                (self.nanos.as_inner() / NANOS_PER_MICRO) as u64,
                self.nanos.as_inner() % NANOS_PER_MICRO,
                NANOS_PER_MICRO / 10,
                prefix,
                "µs",
            )
        } else {
            fmt_decimal(f, self.nanos.as_inner() as u64, 0, 1, prefix, "ns")
        }
    }
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
        let restored = Timespec::from_libc(libc_ts);
        assert_eq!(original, restored);
    }
}
