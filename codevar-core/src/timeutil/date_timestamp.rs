//! Copyright 2026 Codevar
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

//! The [`Timestamp`] struct and associated `impl`s.

use alloc::string::String;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use deranged::{ri64, ri128, ru8, ru32};
use std::time::SystemTime;

use crate::timeutil::date::Date;
use crate::timeutil::date_error::ComponentRange;
use crate::timeutil::date_internal_macro::{const_try, div_floor, ensure_ranged};
use crate::timeutil::date_month::Month;
use crate::timeutil::date_num_fmt::{
    str_from_raw_parts, truncated_subsecond_from_nanos, u64_pad_none,
};
use crate::timeutil::date_offset_time::OffsetDateTime;
use crate::timeutil::date_signed_duration::SignedDuration;
use crate::timeutil::date_time::Time;
use crate::timeutil::date_unit::{
    Day, Hour, Microsecond, Millisecond, Minute, Nanosecond, Second,
};
use crate::timeutil::date_utc_offset::UtcOffset;
use crate::timeutil::date_utc_time::UtcDateTime;
use crate::timeutil::date_util::{Overflow, leap_ordinal_to_month_day};
use crate::timeutil::date_weekday::Weekday;

/// The range of valid seconds for a [`Timestamp`].
pub(crate) type Seconds =
    ri64<{ UtcDateTime::MIN.unix_timestamp() }, { UtcDateTime::MAX.unix_timestamp() }>;
type Nanoseconds = ru32<0, 999_999_999>;

// Validate that the minimum time is midnight and the maximum is one nanosecond before midnight.
// This is necessary because the soundness of some functions relies on this fact.
const _: () = {
    assert!(Timestamp::MIN.time().as_u64() == Time::MIDNIGHT.as_u64());
    assert!(Timestamp::MAX.time().as_u64() == Time::MAX.as_u64());
};

/// By explicitly inserting this enum where padding is expected, the compiler is able to better
/// perform niche value optimization.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Padding {
    #[allow(clippy::missing_docs_in_private_items)]
    Optimize,
}

/// A Unix timestamp with nanosecond precision.
///
/// This type represents a point in time as a number of seconds and nanoseconds elapsed since the
/// Unix epoch (1970-01-01 00:00:00 UTC). Negative values represent times before the Unix epoch.
#[derive(Clone, Copy, Eq)]
#[cfg_attr(not(docsrs), repr(C))]
pub struct Timestamp {
    #[cfg(target_endian = "big")]
    seconds: Seconds,
    #[cfg(target_endian = "big")]
    nanoseconds: Nanoseconds,
    #[cfg(target_endian = "big")]
    padding: Padding,

    #[cfg(target_endian = "little")]
    padding: Padding,
    #[cfg(target_endian = "little")]
    nanoseconds: Nanoseconds,
    #[cfg(target_endian = "little")]
    seconds: Seconds,
}

impl Hash for Timestamp {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_i128(self.as_i128());
    }
}

impl PartialEq for Timestamp {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_i128() == other.as_i128()
    }
}

impl PartialOrd for Timestamp {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_i128().cmp(&other.as_i128())
    }
}

impl From<SystemTime> for Timestamp {
    #[inline]
    fn from(value: SystemTime) -> Self {
        match value.duration_since(SystemTime::UNIX_EPOCH) {
            Ok(duration) => {
                let nanos = i128::from(duration.as_secs()) * 1_000_000_000_i128
                    + i128::from(duration.subsec_nanos());
                Self::from_nanoseconds(nanos).unwrap_or_else(|_| Self::MAX)
            }
            Err(_) => {
                let Ok(duration) = SystemTime::UNIX_EPOCH.duration_since(value) else {
                    return Self::MIN;
                };
                let nanos = -((i128::from(duration.as_secs()) * 1_000_000_000_i128)
                    + i128::from(duration.subsec_nanos()));
                Self::from_nanoseconds(nanos).unwrap_or_else(|_| Self::MIN)
            }
        }
    }
}

impl Timestamp {
    #[inline]
    const fn as_i128(self) -> i128 {
        unsafe { core::mem::transmute(self) }
    }

    /// A `Timestamp` representing the Unix epoch (1970-01-01 00:00:00 UTC).
    pub const UNIX_EPOCH: Self =
        Self::new_ranged(Seconds::new_static::<0>(), Nanoseconds::new_static::<0>());

    /// The minimum valid `Timestamp`.
    ///
    /// The moment in time represented by this value may vary depending on the feature flags
    /// enabled.
    pub const MIN: Self = Self::new_ranged(Seconds::MIN, Nanoseconds::MIN);

    /// The maximum valid `Timestamp`.
    ///
    /// The moment in time represented by this value may vary depending on the feature flags
    /// enabled.
    pub const MAX: Self = Self::new_ranged(Seconds::MAX, Nanoseconds::MAX);

    /// Create a new `Timestamp` representing the current moment in time.
    #[inline]
    pub fn now() -> Self {
        SystemTime::now().into()
    }

    /// Create a `Timestamp` from the provided seconds and nanoseconds values without checking if
    /// they are valid.
    ///
    /// # Safety
    ///
    /// Both `seconds` and `nanoseconds` must be in range.
    #[doc(hidden)]
    #[inline]
    pub const unsafe fn new_unchecked(seconds: i64, nanoseconds: u32) -> Self {
        // Safety: The caller must ensure both values are valid.
        unsafe {
            Self::new_ranged(
                Seconds::new_unchecked(seconds),
                Nanoseconds::new_unchecked(nanoseconds),
            )
        }
    }

    /// Create a `Timestamp` from the provided seconds and nanoseconds values that are known to be
    /// in range.
    #[inline]
    pub(crate) const fn new_ranged(seconds: Seconds, nanoseconds: Nanoseconds) -> Self {
        Self {
            seconds,
            nanoseconds,
            padding: Padding::Optimize,
        }
    }

    /// Create a `Timestamp` from the provided Unix timestamp in seconds and nanoseconds, returning
    /// an error if the resulting value is out of range.
    #[inline]
    pub const fn new(seconds: i64, nanoseconds: u32) -> Result<Self, ComponentRange> {
        Ok(Self::new_ranged(
            ensure_ranged!(Seconds: seconds),
            ensure_ranged!(Nanoseconds: nanoseconds),
        ))
    }

    /// Create a `Timestamp` from the provided Unix timestamp in seconds, returning an error if the
    /// resulting value is out of range.
    #[inline]
    pub const fn from_seconds(seconds: i64) -> Result<Self, ComponentRange> {
        Ok(Self::new_ranged(
            ensure_ranged!(Seconds: seconds),
            Nanoseconds::new_static::<0>(),
        ))
    }

    /// Create a `Timestamp` from the provided Unix timestamp in milliseconds, returning an error if
    /// the resulting value is out of range.
    #[inline]
    pub const fn from_milliseconds(milliseconds: i64) -> Result<Self, ComponentRange> {
        const MAX: i64 = Seconds::MAX.get() * Millisecond::per_t::<i64>(Second)
            + (Nanoseconds::MAX.get() as i64) / Nanosecond::per_t::<i64>(Millisecond);
        const MIN: i64 = Seconds::MIN.get() * Millisecond::per_t::<i64>(Second)
            + (Nanoseconds::MIN.get() as i64) / Nanosecond::per_t::<i64>(Millisecond);
        ensure_ranged!(ri64<MIN, MAX>: milliseconds);
        let mut seconds = milliseconds / Millisecond::per_t::<i64>(Second);
        let nanoseconds = (milliseconds.rem_euclid(Millisecond::per_t(Second))
            * Nanosecond::per_t::<i64>(Millisecond)) as u32;

        if milliseconds < 0 && nanoseconds != 0 {
            seconds -= 1;
        }
        // Safety: The value provided was checked to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds) })
    }

    /// Create a `Timestamp` from the provided Unix timestamp in microseconds, returning an error if
    /// the resulting value is out of range.
    #[inline]
    pub const fn from_microseconds(microseconds: i128) -> Result<Self, ComponentRange> {
        const MAX: i128 = Seconds::MAX.get() as i128 * Microsecond::per_t::<i128>(Second)
            + (Nanoseconds::MAX.get() as i128) / Nanosecond::per_t::<i128>(Microsecond);
        const MIN: i128 = Seconds::MIN.get() as i128 * Microsecond::per_t::<i128>(Second)
            + (Nanoseconds::MIN.get() as i128) / Nanosecond::per_t::<i128>(Microsecond);

        ensure_ranged!(ri128<MIN, MAX>: microseconds);

        let mut seconds = (microseconds / Microsecond::per_t::<i128>(Second)) as i64;
        let nanoseconds = (microseconds.rem_euclid(Microsecond::per_t(Second))
            * Nanosecond::per_t::<i128>(Microsecond)) as u32;

        if microseconds < 0 && nanoseconds != 0 {
            seconds -= 1;
        }

        // Safety: The value provided was checked to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds) })
    }

    /// Create a `Timestamp` from the provided Unix timestamp in nanoseconds, returning an error if
    /// the resulting value is out of range.
    #[inline]
    pub const fn from_nanoseconds(nanoseconds: i128) -> Result<Self, ComponentRange> {
        const MAX: i128 = Seconds::MAX.get() as i128 * Nanosecond::per_t::<i128>(Second)
            + Nanoseconds::MAX.get() as i128;
        const MIN: i128 = Seconds::MIN.get() as i128 * Nanosecond::per_t::<i128>(Second)
            + Nanoseconds::MIN.get() as i128;

        ensure_ranged!(ri128<MIN, MAX>: nanoseconds);

        let input_is_negative = nanoseconds < 0;
        let mut seconds = (nanoseconds / Nanosecond::per_t::<i128>(Second)) as i64;
        let nanoseconds = nanoseconds.rem_euclid(Nanosecond::per_t(Second)) as u32;

        if input_is_negative && nanoseconds != 0 {
            seconds -= 1;
        }

        // Safety: The value provided was checked to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds) })
    }

    /// Convert the `Timestamp` to an [`OffsetDateTime`] at the provided offset.
    ///
    /// # Panics
    ///
    /// This panics if the resulting date-time with the provided offset is outside the supported
    /// range. Consider using [`checked_to_offset`](Self::checked_to_offset) for a non-panicking
    /// alternative.
    #[inline]
    pub const fn to_offset(self, offset: UtcOffset) -> OffsetDateTime {
        match self.to_utc() {
            Some(utc) => utc.to_offset(offset),
            None => OffsetDateTime::new_in_offset(Date::MIN, Time::MIDNIGHT, offset),
        }
    }

    /// Convert the `Timestamp` to an [`OffsetDateTime`] with the provided offset, returning `None`
    /// if the resulting value is out of range.
    #[inline]
    pub const fn checked_to_offset(self, offset: UtcOffset) -> Option<OffsetDateTime> {
        match self.to_utc() {
            Some(utc) => utc.checked_to_offset(offset),
            None => None,
        }
    }

    /// Convert the `Timestamp` to a [`UtcDateTime`].
    #[inline]
    pub const fn to_utc(self) -> Option<UtcDateTime> {
        let Ok(utc_dt) = UtcDateTime::from_unix_timestamp(self.seconds.get()) else {
            return None;
        };
        let Ok(utc_dt) = utc_dt.replace_nanosecond(self.nanoseconds.get()) else {
            return None;
        };
        Some(utc_dt)
    }

    /// Get the seconds and nanoseconds of the timestamp as ranged values.
    #[inline]
    pub(crate) const fn as_parts_ranged(self) -> (Seconds, Nanoseconds) {
        (self.seconds, self.nanoseconds)
    }

    /// Get the number of seconds since the Unix epoch.
    ///
    /// Negative values represent moments before the Unix epoch.
    #[inline]
    pub const fn as_seconds(self) -> i64 {
        self.seconds.get()
    }

    /// Get the number of milliseconds since the Unix epoch.
    ///
    /// Negative values represent moments before the Unix epoch.
    /// ```
    #[inline]
    pub const fn as_milliseconds(self) -> i64 {
        self.seconds.get() * Millisecond::per_t::<i64>(Second)
            + (self.nanoseconds.get() / Nanosecond::per_t::<u32>(Millisecond)) as i64
    }

    /// Get the number of microseconds since the Unix epoch.
    #[inline]
    pub const fn as_microseconds(self) -> i128 {
        self.seconds.get() as i128 * Microsecond::per_t::<i128>(Second)
            + (self.nanoseconds.get() / Nanosecond::per_t::<u32>(Microsecond)) as i128
    }

    /// Get the number of nanoseconds since the Unix epoch.
    ///
    /// Negative values represent moments before the Unix epoch.
    #[inline]
    pub const fn as_nanoseconds(self) -> i128 {
        self.seconds.get() as i128 * Nanosecond::per_t::<i128>(Second)
            + self.nanoseconds.get() as i128
    }

    /// Get the [`Date`] of the timestamp in UTC.
    #[inline]
    pub const fn date(self) -> Date {
        match self.to_utc() {
            Some(utc) => utc.date(),
            None => Date::MIN,
        }
    }

    /// Get the [`Time`] of the timestamp in UTC.
    #[inline]
    pub const fn time(self) -> Time {
        let within_day = self.as_seconds().rem_euclid(Second::per_t::<i64>(Day)) as u32;

        let hour = within_day / Second::per_t::<u32>(Hour);
        let minute = (within_day - hour * Second::per_t::<u32>(Hour))
            / Second::per_t::<u32>(Minute);
        let second = within_day
            - hour * Second::per_t::<u32>(Hour)
            - minute * Second::per_t::<u32>(Minute);

        // Safety: All values are guaranteed to be in range.
        unsafe {
            Time::from_hms_nanos_unchecked(
                hour as u8,
                minute as u8,
                second as u8,
                self.nanosecond(),
            )
        }
    }

    /// Compute the year, leap year status, and ordinal day of the timestamp in UTC.
    ///
    /// This algorithm is essentially identical to `Date::from_julian_day_unchecked`. Instead of
    /// returning `Date`, it returns the components as a tuple. By not bitpacking the values, it
    /// allows the compiler to see through the function boundary and better optimize methods.
    #[inline]
    const fn year_leap_ordinal(self) -> (i32, bool, u16) {
        const ERAS: u32 = 5_949;
        const D_SHIFT: u32 = 146097 * ERAS + 719_528;
        const Y_SHIFT: u32 = 400 * ERAS;

        const CEN_MUL: u32 = ((4u64 << 47) / 146_097) as u32;
        const JUL_MUL: u32 = ((4u64 << 40) / 1_461 + 1) as u32;
        const CEN_CUT: u32 = ((365u64 << 32) / 36_525) as u32;

        let raw_day = div_floor!(self.as_seconds(), Second::per_t::<i64>(Day)) as i32;

        let day = raw_day.cast_unsigned().wrapping_add(D_SHIFT);
        let c_n = (day as u64 * CEN_MUL as u64) >> 15;
        let cen = (c_n >> 32) as u32;
        let cpt = c_n as u32;
        let ijy = cpt > CEN_CUT || cen.is_multiple_of(4);
        let jul = day - cen / 4 + cen;
        let y_n = (jul as u64 * JUL_MUL as u64) >> 8;
        let yrs = (y_n >> 32) as u32;
        let ypt = y_n as u32;

        let year = yrs.wrapping_sub(Y_SHIFT).cast_signed();
        let ordinal = ((ypt as u64 * 1_461) >> 34) as u32 + ijy as u32;
        let leap = yrs.is_multiple_of(4) & ijy;

        (year, leap, ordinal as u16)
    }

    /// Get the year of the timestamp in UTC.
    #[inline]
    pub const fn year(self) -> i32 {
        self.year_leap_ordinal().0
    }

    /// Get the month of the timestamp in UTC.
    #[inline]
    pub const fn month(self) -> Month {
        let (_, leap, ordinal) = self.year_leap_ordinal();
        leap_ordinal_to_month_day(leap, ordinal).0
    }

    /// Get the day of the month of the timestamp in UTC.
    ///
    /// The returned value will always be in the range `1..=31`.
    #[inline]
    pub const fn day(self) -> u8 {
        let (_, leap, ordinal) = self.year_leap_ordinal();
        leap_ordinal_to_month_day(leap, ordinal).1
    }

    /// Get the day of the year of the timestamp in UTC.
    ///
    /// The returned value will always be in the range `1..=366`.
    #[inline]
    pub const fn ordinal(self) -> u16 {
        self.year_leap_ordinal().2
    }

    /// Get the ISO week number of the timestamp in UTC.
    ///
    /// The returned value will always be in the range `1..=53`.
    #[inline]
    pub const fn iso_week(self) -> u8 {
        self.date().iso_week()
    }

    /// Get the Sunday-based week number of the timestamp in UTC.
    ///
    /// The returned value will always be in the range `0..=53`.
    #[inline]
    pub const fn sunday_based_week(self) -> u8 {
        self.date().sunday_based_week()
    }

    /// Get the Monday-based week number of the timestamp in UTC.
    ///
    /// The returned value will always be in the range `0..=53`.
    #[inline]
    pub const fn monday_based_week(self) -> u8 {
        self.date().monday_based_week()
    }

    /// Get the calendar date (year, month, day) of the timestamp in UTC.
    #[inline]
    pub const fn to_calendar_date(self) -> (i32, Month, u8) {
        let (year, leap, ordinal) = self.year_leap_ordinal();
        let (month, day) = leap_ordinal_to_month_day(leap, ordinal);
        (year, month, day)
    }

    /// Get the ordinal date (year, ordinal day) of the timestamp in UTC.
    #[inline]
    pub const fn to_ordinal_date(self) -> (i32, u16) {
        let (year, _, ordinal) = self.year_leap_ordinal();
        (year, ordinal)
    }

    /// Get the ISO week date (year, week number, weekday) of the timestamp in UTC.
    #[inline]
    pub const fn to_iso_week_date(self) -> (i32, u8, Weekday) {
        self.date().to_iso_week_date()
    }

    /// Get the weekday of the timestamp in UTC.
    #[inline]
    pub const fn weekday(self) -> Weekday {
        // 365,961,669 is obtained by starting with the smallest timestamp (with large-dates
        // enabled), dividing by 86,400 to get the number of days, then rounding down to get a
        // multiple of 7. This value is negated as we want to end with a positive number. Finally, 3
        // is added to shift the zero value to Monday, matching the internal representation of
        // `Weekday`.
        match (div_floor!(self.seconds.get(), 86_400) + 365_961_669) % 7 {
            0 => Weekday::Monday,
            1 => Weekday::Tuesday,
            2 => Weekday::Wednesday,
            3 => Weekday::Thursday,
            4 => Weekday::Friday,
            5 => Weekday::Saturday,
            6 => Weekday::Sunday,
            _ => unreachable!(),
        }
    }

    /// Get the Julian day of the timestamp.
    #[inline]
    pub const fn to_julian_day(self) -> i32 {
        const UNIX_EPOCH_JULIAN_DAY: i32 = Date::UNIX_EPOCH.to_julian_day();
        div_floor!(self.seconds.get(), 86_400) as i32 + UNIX_EPOCH_JULIAN_DAY
    }

    /// Get the hours, minutes, and seconds of the timestamp in UTC.
    #[inline]
    pub const fn as_hms(self) -> (u8, u8, u8) {
        self.time().as_hms()
    }

    /// Get the hours, minutes, seconds, and milliseconds of the timestamp in UTC.
    #[inline]
    pub const fn as_hms_milli(self) -> (u8, u8, u8, u16) {
        self.time().as_hms_milli()
    }

    /// Get the hours, minutes, seconds, and microseconds of the timestamp in UTC.
    #[inline]
    pub const fn as_hms_micro(self) -> (u8, u8, u8, u32) {
        self.time().as_hms_micro()
    }

    /// Get the hours, minutes, seconds, and nanoseconds of the timestamp in UTC.
    #[inline]
    pub const fn as_hms_nano(self) -> (u8, u8, u8, u32) {
        self.time().as_hms_nano()
    }

    /// Get the hour of the timestamp in UTC.
    #[inline]
    pub const fn hour(self) -> u8 {
        self.time().hour()
    }

    /// Get the minute of the timestamp in UTC.
    #[inline]
    pub const fn minute(self) -> u8 {
        (div_floor!(self.seconds.get(), Second::per_t::<i64>(Minute)))
            .rem_euclid(Minute::per_t(Hour)) as u8
    }

    /// Get the second of the timestamp in UTC.
    #[inline]
    pub const fn second(self) -> u8 {
        self.seconds.get().rem_euclid(Second::per_t(Minute)) as u8
    }

    /// Get the millisecond of the timestamp in UTC.
    #[inline]
    pub const fn millisecond(self) -> u16 {
        (self.nanoseconds.get() / Nanosecond::per_t::<u32>(Millisecond)) as u16
    }

    /// Get the microsecond of the timestamp in UTC.
    #[inline]
    pub const fn microsecond(self) -> u32 {
        self.nanoseconds.get() / Nanosecond::per_t::<u32>(Microsecond)
    }

    /// Get the nanosecond of the timestamp in UTC.

    #[inline]
    pub const fn nanosecond(self) -> u32 {
        self.nanoseconds.get()
    }

    /// Add a [`SignedDuration`] to the timestamp. Returns `Overflow::Positive` or
    /// `Overflow::Negative` if the result is out of range.
    #[inline]
    const fn add(self, duration: SignedDuration) -> Result<Self, Overflow> {
        let (second_adj, nanoseconds) = if duration.is_negative() {
            let nanos = self.nanoseconds.get() as i32 + duration.subsec_nanoseconds();
            if nanos < 0 {
                (-1, (nanos + Nanosecond::per_t::<i32>(Second)) as u32)
            } else {
                (0, nanos as u32)
            }
        } else {
            let nanos = self.nanoseconds.get() + duration.subsec_nanoseconds() as u32;
            if nanos >= Nanosecond::per_t(Second) {
                (1, nanos - Nanosecond::per_t::<u32>(Second))
            } else {
                (0, nanos)
            }
        };

        let seconds = match self.seconds.get().checked_add(duration.whole_seconds()) {
            Some(seconds) => seconds,
            None if duration.is_negative() => return Err(Overflow::Negative),
            None => return Err(Overflow::Positive),
        };
        let seconds = match seconds.checked_add(second_adj) {
            Some(seconds) => seconds,
            None if second_adj < 0 => return Err(Overflow::Negative),
            None => return Err(Overflow::Positive),
        };

        // Check if the resulting seconds are within the valid range
        if seconds < Seconds::MIN.get() {
            return Err(Overflow::Negative);
        } else if seconds > Seconds::MAX.get() {
            return Err(Overflow::Positive);
        }

        // Safety: Both values are guaranteed to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds) })
    }

    /// Subtract a [`SignedDuration`] from the timestamp. Returns `Overflow::Positive` or
    /// `Overflow::Negative` if the result is out of range.
    #[inline]
    const fn sub(self, duration: SignedDuration) -> Result<Self, Overflow> {
        let nanos = self.nanoseconds.get() as i32 - duration.subsec_nanoseconds();
        let (second_adj, nanoseconds) = if duration.is_negative() {
            if nanos >= Nanosecond::per_t::<i32>(Second) {
                (1, (nanos - Nanosecond::per_t::<i32>(Second)) as u32)
            } else if nanos < 0 {
                (-1, (nanos + Nanosecond::per_t::<i32>(Second)) as u32)
            } else {
                (0, nanos as u32)
            }
        } else {
            if nanos < 0 {
                (-1, (nanos + Nanosecond::per_t::<i32>(Second)) as u32)
            } else {
                (0, nanos as u32)
            }
        };

        let seconds = match self.seconds.get().checked_sub(duration.whole_seconds()) {
            Some(seconds) => seconds,
            None if duration.is_negative() => return Err(Overflow::Positive),
            None => return Err(Overflow::Negative),
        };
        let seconds = match seconds.checked_add(second_adj) {
            Some(seconds) => seconds,
            None if second_adj < 0 => return Err(Overflow::Negative),
            None => return Err(Overflow::Positive),
        };

        // Check if the resulting seconds are within the valid range
        if seconds < Seconds::MIN.get() {
            return Err(Overflow::Negative);
        } else if seconds > Seconds::MAX.get() {
            return Err(Overflow::Positive);
        }

        // Safety: Both values are guaranteed to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds) })
    }

    /// Add a [`std::time::Duration`] to the timestamp. Returns `Overflow::Positive` or
    /// `Overflow::Negative` if the result is out of range.
    #[inline]
    const fn add_std(self, duration: StdDuration) -> Result<Self, Overflow> {
        let Some(mut seconds) =
            self.seconds.get().checked_add_unsigned(duration.as_secs())
        else {
            return Err(Overflow::Positive);
        };
        let mut nanoseconds = self.nanoseconds.get() + duration.subsec_nanos();

        if nanoseconds >= Nanosecond::per_t(Second) {
            nanoseconds -= Nanosecond::per_t::<u32>(Second);
            let Some(new_seconds) = seconds.checked_add(1) else {
                return Err(Overflow::Positive);
            };
            seconds = new_seconds;
        }

        // Check if the resulting seconds are within the valid range
        if seconds < Seconds::MIN.get() {
            return Err(Overflow::Negative);
        } else if seconds > Seconds::MAX.get() {
            return Err(Overflow::Positive);
        }

        // Safety: Both values are guaranteed to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds) })
    }

    /// Subtract a [`std::time::Duration`] from the timestamp. Returns `Overflow::Positive` or
    /// `Overflow::Negative` if the result is out of range.
    #[inline]
    const fn sub_std(self, duration: StdDuration) -> Result<Self, Overflow> {
        let Some(mut seconds) =
            self.seconds.get().checked_sub_unsigned(duration.as_secs())
        else {
            return Err(Overflow::Negative);
        };
        let mut nanoseconds =
            self.nanoseconds.get() as i32 - duration.subsec_nanos() as i32;

        if nanoseconds < 0 {
            nanoseconds += Nanosecond::per_t::<i32>(Second);
            let Some(new_seconds) = seconds.checked_sub(1) else {
                return Err(Overflow::Negative);
            };
            seconds = new_seconds;
        }

        // Check if the resulting seconds are within the valid range
        if seconds < Seconds::MIN.get() {
            return Err(Overflow::Negative);
        } else if seconds > Seconds::MAX.get() {
            return Err(Overflow::Positive);
        }

        // Safety: Both values are guaranteed to be in range.
        Ok(unsafe { Self::new_unchecked(seconds, nanoseconds as u32) })
    }

    /// Checked addition of a [`SignedDuration`], returning `None` if the result is out of range.
    #[inline]
    pub const fn checked_add(self, duration: SignedDuration) -> Option<Self> {
        match self.add(duration) {
            Ok(timestamp) => Some(timestamp),
            Err(Overflow::Positive | Overflow::Negative) => None,
        }
    }

    /// Checked subtraction of a [`SignedDuration`], returning `None` if the result is out of range.
    #[inline]
    pub const fn checked_sub(self, duration: SignedDuration) -> Option<Self> {
        match self.sub(duration) {
            Ok(timestamp) => Some(timestamp),
            Err(Overflow::Positive | Overflow::Negative) => None,
        }
    }

    /// Saturating addition of a [`SignedDuration`].
    ///
    /// Returns [`Timestamp::MAX`] or [`Timestamp::MIN`] if the result is out of range.
    #[inline]
    pub const fn saturating_add(self, duration: SignedDuration) -> Self {
        match self.add(duration) {
            Ok(timestamp) => timestamp,
            Err(Overflow::Positive) => Self::MAX,
            Err(Overflow::Negative) => Self::MIN,
        }
    }

    /// Saturating addition of a [`std::time::Duration`].
    #[inline]
    pub const fn saturating_add_std(self, duration: StdDuration) -> Self {
        match self.add_std(duration) {
            Ok(timestamp) => timestamp,
            Err(Overflow::Positive) => Self::MAX,
            Err(Overflow::Negative) => Self::MIN,
        }
    }

    /// Saturating subtraction of a [`SignedDuration`].
    ///
    /// Returns [`Timestamp::MAX`] or [`Timestamp::MIN`] if the result is out of range.
    #[inline]
    pub const fn saturating_sub(self, duration: SignedDuration) -> Self {
        match self.sub(duration) {
            Ok(timestamp) => timestamp,
            Err(Overflow::Positive) => Self::MAX,
            Err(Overflow::Negative) => Self::MIN,
        }
    }

    /// Saturating subtraction of a [`std::time::Duration`].
    #[inline]
    pub const fn saturating_sub_std(self, duration: StdDuration) -> Self {
        match self.sub_std(duration) {
            Ok(timestamp) => timestamp,
            Err(Overflow::Positive) => Self::MAX,
            Err(Overflow::Negative) => Self::MIN,
        }
    }
}

/// Methods that replace part of the `Timestamp`.
impl Timestamp {
    /// Replace the time, preserving the date.
    #[inline]
    #[must_use = "This method does not mutate the original `Timestamp`."]
    pub const fn replace_time(self, time: Time) -> Self {
        let seconds_since_midnight = time.hour() as i64 * Second::per_t::<i64>(Hour)
            + time.minute() as i64 * Second::per_t::<i64>(Minute)
            + time.second() as i64;
        let seconds = div_floor!(self.seconds.get(), Second::per_t::<i64>(Day))
            * Second::per_t::<i64>(Day)
            + seconds_since_midnight;
        // Safety: Seconds is constructed from an existing valid value, and nanoseconds are always
        // in range given the origin. Any time of day is valid for any date in range, as enforced by
        // const assertions.
        unsafe { Self::new_unchecked(seconds, time.nanosecond()) }
    }

    /// Replace the date, preserving the time.
    #[inline]
    #[must_use = "This method does not mutate the original `Timestamp`."]
    pub const fn replace_date(mut self, date: Date) -> Self {
        let seconds_after_midnight = self.seconds.get().rem_euclid(Second::per_t(Day));
        let seconds = (date.to_julian_day() as i64
            - UtcDateTime::UNIX_EPOCH.to_julian_day() as i64)
            * Second::per_t::<i64>(Day)
            + seconds_after_midnight;
        // Safety: The range of valid dates is identical to the range of valid timestamps, so any
        // date is necessarily valid.
        self.seconds = unsafe { Seconds::new_unchecked(seconds) };
        self
    }

    /// Replace the year, preserving the month and day. If the date is February 29 and the resulting
    /// year is not a leap year, an error is returned.
    #[inline]
    pub const fn replace_year(self, year: i32) -> Result<Self, ComponentRange> {
        let date = const_try!(self.date().replace_year(year));
        Ok(self.replace_date(date))
    }

    /// Replace the month of the year, preserving the year and day. If the day is invalid for the
    /// resulting month, an error is returned.
    #[inline]
    pub const fn replace_month(self, month: Month) -> Result<Self, ComponentRange> {
        let date = const_try!(self.date().replace_month(month));
        Ok(self.replace_date(date))
    }

    /// Replace the day of the month.
    #[inline]
    pub const fn replace_day(self, day: u8) -> Result<Self, ComponentRange> {
        let date = const_try!(self.date().replace_day(day));
        Ok(self.replace_date(date))
    }

    /// Replace the day of the year.
    #[inline]
    pub const fn replace_ordinal(self, ordinal: u16) -> Result<Self, ComponentRange> {
        let date = const_try!(self.date().replace_ordinal(ordinal));
        Ok(self.replace_date(date))
    }

    /// Replace the clock hour.
    #[inline]
    pub const fn replace_hour(mut self, hour: u8) -> Result<Self, ComponentRange> {
        ensure_ranged!(ru8<0, 23>: hour);
        let seconds = div_floor!(self.seconds.get(), Second::per_t::<i64>(Day))
            * Second::per_t::<i64>(Day)
            + hour as i64 * Second::per_t::<i64>(Hour)
            + self.minute() as i64 * Second::per_t::<i64>(Minute)
            + self.second() as i64;
        // Safety: Any value is valid so long as `hour` is in range.
        self.seconds = unsafe { Seconds::new_unchecked(seconds) };
        Ok(self)
    }

    /// Replace the minutes within the hour.
    #[inline]
    pub const fn replace_minute(mut self, minute: u8) -> Result<Self, ComponentRange> {
        ensure_ranged!(ru8<0, 59>: minute);
        let seconds = div_floor!(self.seconds.get(), Second::per_t::<i64>(Hour))
            * Second::per_t::<i64>(Hour)
            + minute as i64 * Second::per_t::<i64>(Minute)
            + self.second() as i64;
        // Safety: Any value is valid so long as `minute` is in range.
        self.seconds = unsafe { Seconds::new_unchecked(seconds) };
        Ok(self)
    }

    /// Replace the seconds within the minute.
    #[inline]
    pub const fn replace_second(mut self, second: u8) -> Result<Self, ComponentRange> {
        ensure_ranged!(ru8<0, 59>: second);
        let seconds = div_floor!(self.seconds.get(), Second::per_t::<i64>(Minute))
            * Second::per_t::<i64>(Minute)
            + second as i64;
        // Safety: Any value is valid so long as `second` is in range.
        self.seconds = unsafe { Seconds::new_unchecked(seconds) };
        Ok(self)
    }

    /// Replace the milliseconds within the second.
    #[inline]
    pub const fn replace_millisecond(
        self,
        millisecond: u16,
    ) -> Result<Self, ComponentRange> {
        let nanos = ensure_ranged!(Nanoseconds: millisecond as u32 * Nanosecond::per_t::<u32>(Millisecond));
        Ok(self.replace_nanosecond_ranged(nanos))
    }

    /// Replace the microseconds within the second.
    #[inline]
    pub const fn replace_microsecond(
        self,
        microsecond: u32,
    ) -> Result<Self, ComponentRange> {
        let nanos = ensure_ranged!(Nanoseconds: microsecond * Nanosecond::per_t::<u32>(Microsecond));
        Ok(self.replace_nanosecond_ranged(nanos))
    }

    /// Replace the nanoseconds within the second.
    #[inline]
    pub const fn replace_nanosecond(
        self,
        nanosecond: u32,
    ) -> Result<Self, ComponentRange> {
        let nanos = ensure_ranged!(Nanoseconds: nanosecond);
        Ok(self.replace_nanosecond_ranged(nanos))
    }

    /// Replace the nanoseconds within the second using a range-bounded integer to avoid range
    /// checks.
    #[inline]
    const fn replace_nanosecond_ranged(self, new_nanos: Nanoseconds) -> Self {
        let (seconds, nanoseconds) = self.as_parts_ranged();

        if seconds.get() >= 0 || nanoseconds.get() == 0 {
            Self::new_ranged(seconds, new_nanos)
        } else if new_nanos.get() == 0 {
            // Safety: The previous conditional guarantees that `seconds` is negative (if it were
            // non-negative, we wouldn't be in this branch). Given that the maximum value is
            // positive, we can always add one without exceeding the maximum.
            Self::new_ranged(unsafe { seconds.unchecked_add(1) }, new_nanos)
        } else {
            // Safety: Given the range of `new_nanos`, subtracting it from the maximum always
            // results in a value in range. Zero is excluded by a previous conditional.
            Self::new_ranged(seconds, unsafe {
                Nanoseconds::new_unchecked(
                    Nanosecond::per_t::<u32>(Second) - new_nanos.get(),
                )
            })
        }
    }
}

impl Timestamp {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// by the `Display` implementation.
    const DISPLAY_BUFFER_SIZE: usize = 25;

    /// Format the `Timestamp` into the provided buffer, returning the number of bytes written.
    pub(crate) fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        let mut idx = 0;

        let mut second = self.seconds.get();
        let mut nanosecond = self.nanoseconds;

        if second < 0 {
            buf[idx] = MaybeUninit::new(b'-');
            idx += 1;

            second = -second;

            if nanosecond != Nanoseconds::new_static::<0>() {
                second -= 1;
                // Safety: `nanosecond` is in the range 1..=999_999_999, so subtracting it from
                // 1_000_000_000 will always yield a value in the range 1..=999_999_999, which is a
                // subset of the valid range for `Nanoseconds`.
                nanosecond = unsafe {
                    Nanoseconds::new_unchecked(
                        Nanosecond::per_t::<u32>(Second) - nanosecond.get(),
                    )
                };
            }
        }
        let seconds_str = u64_pad_none(second.cast_unsigned());
        let seconds_len = seconds_str.len();
        // Safety: `buf` has sufficient capacity for the seconds digits.
        unsafe {
            seconds_str
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), seconds_len);
        }
        idx += seconds_len;

        if nanosecond != Nanoseconds::new_static::<0>() {
            buf[idx] = MaybeUninit::new(b'.');
            idx += 1;

            let subsecond = truncated_subsecond_from_nanos(nanosecond);
            // Safety: `buf` has sufficient capacity for the subsecond digits.
            unsafe {
                subsecond.as_ptr().copy_to_nonoverlapping(
                    buf.as_mut_ptr().add(idx).cast(),
                    subsecond.len(),
                );
            }
            idx += subsecond.len();
        }

        idx
    }
}

impl fmt::Display for Timestamp {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); Self::DISPLAY_BUFFER_SIZE];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts(buf.as_ptr().cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for Timestamp {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Add<SignedDuration> for Timestamp {
    type Output = Self;

    #[inline]
    fn add(self, rhs: SignedDuration) -> Self::Output {
        match self.checked_add(rhs) {
            Some(value) => value,
            None => self.saturating_add(rhs),
        }
    }
}

impl Add<StdDuration> for Timestamp {
    type Output = Self;

    #[inline]
    fn add(self, rhs: StdDuration) -> Self::Output {
        match self.add_std(rhs) {
            Ok(value) => value,
            Err(_) => self.saturating_add_std(rhs),
        }
    }
}

impl AddAssign<SignedDuration> for Timestamp {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        *self = *self + rhs;
    }
}

impl AddAssign<StdDuration> for Timestamp {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: StdDuration) {
        *self = *self + rhs;
    }
}

impl Sub<SignedDuration> for Timestamp {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SignedDuration) -> Self::Output {
        match self.checked_sub(rhs) {
            Some(value) => value,
            None => self.saturating_sub(rhs),
        }
    }
}

impl Sub<StdDuration> for Timestamp {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: StdDuration) -> Self::Output {
        match self.sub_std(rhs) {
            Ok(value) => value,
            Err(_) => self.saturating_sub_std(rhs),
        }
    }
}

impl SubAssign<SignedDuration> for Timestamp {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        *self = *self - rhs;
    }
}

impl SubAssign<StdDuration> for Timestamp {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: StdDuration) {
        *self = *self - rhs;
    }
}

impl Sub for Timestamp {
    type Output = SignedDuration;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let seconds = self.seconds.get() - rhs.seconds.get();
        let nanoseconds = self.nanoseconds.get() as i32 - rhs.nanoseconds.get() as i32;

        if nanoseconds < 0 {
            SignedDuration::new(
                seconds - 1,
                nanoseconds + Nanosecond::per_t::<i32>(Second),
            )
        } else {
            SignedDuration::new(seconds, nanoseconds)
        }
    }
}
