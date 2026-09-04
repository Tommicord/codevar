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

use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use std::hint;
use std::time::SystemTime;

use crate::timeutil::date::{Date, MAX_YEAR, MIN_YEAR};
use crate::timeutil::date_error::ComponentRange;
use crate::timeutil::date_internal_macro::{
    carry, cascade, const_try, const_try_opt, div_floor, ensure_ranged,
};
use crate::timeutil::date_month::Month;
use crate::timeutil::date_num_fmt::str_from_raw_parts;
use crate::timeutil::date_offset_time::OffsetDateTime;
use crate::timeutil::date_plain::PlainDateTime;
use crate::timeutil::date_signed_duration::SignedDuration;
use crate::timeutil::date_time::Time;
use crate::timeutil::date_timestamp::{Seconds, Timestamp};
use crate::timeutil::date_unit::{Day, Hour, Minute, Nanosecond, Second};
use crate::timeutil::date_utc_offset::UtcOffset;
use crate::timeutil::date_util::days_in_year;
use crate::timeutil::date_weekday::Weekday;
use deranged::ri64;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

/// The Julian day of the Unix epoch.
const UNIX_EPOCH_JULIAN_DAY: i32 = UtcDateTime::UNIX_EPOCH.to_julian_day();

/// A [`PlainDateTime`] that is known to be UTC.
///
/// `UtcDateTime` is guaranteed to be ABI-compatible with [`PlainDateTime`], meaning that
/// transmuting from one to the other will not result in undefined behavior.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UtcDateTime {
    inner: PlainDateTime,
}

impl From<SystemTime> for UtcDateTime {
    #[inline]
    fn from(value: SystemTime) -> Self {
        let timestamp = Timestamp::from(value);
        match timestamp.to_utc() {
            Some(datetime) => datetime,
            None if value >= SystemTime::UNIX_EPOCH => Self::MAX,
            None => Self::MIN,
        }
    }
}

impl UtcDateTime {
    /// Midnight, 1 January, 1970.
    pub const UNIX_EPOCH: Self = Self::new(Date::UNIX_EPOCH, Time::MIDNIGHT);

    /// The smallest value that can be represented by `UtcDateTime`.
    ///
    /// Depending on `large-dates` feature flag, value of this constant may vary.
    ///
    /// 1. With `large-dates` disabled it is equal to `-9999-01-01 00:00:00.0`
    /// 2. With `large-dates` enabled it is equal to `-999999-01-01 00:00:00.0`
    #[cfg_attr(
        feature = "large-dates",
        doc = "// Assuming `large-dates` feature is enabled."
    )]
    #[cfg_attr(
        feature = "large-dates",
        doc = "assert_eq!(UtcDateTime::MIN, utc_datetime!(-999999-01-01 0:00));"
    )]
    #[cfg_attr(
        not(feature = "large-dates"),
        doc = "// Assuming `large-dates` feature is disabled."
    )]
    #[cfg_attr(
        not(feature = "large-dates"),
        doc = "assert_eq!(UtcDateTime::MIN, utc_datetime!(-9999-01-01 0:00));"
    )]
    /// ```
    pub const MIN: Self = Self::new(Date::MIN, Time::MIDNIGHT);

    /// The largest value that can be represented by `UtcDateTime`.
    ///
    /// Depending on `large-dates` feature flag, value of this constant may vary.
    ///
    /// 1. With `large-dates` disabled it is equal to `9999-12-31 23:59:59.999_999_999`
    /// 2. With `large-dates` enabled it is equal to `999999-12-31 23:59:59.999_999_999`
    #[cfg_attr(
        feature = "large-dates",
        doc = "// Assuming `large-dates` feature is enabled."
    )]
    #[cfg_attr(
        feature = "large-dates",
        doc = "assert_eq!(UtcDateTime::MAX, utc_datetime!(+999999-12-31 23:59:59.999_999_999));"
    )]
    #[cfg_attr(
        not(feature = "large-dates"),
        doc = "// Assuming `large-dates` feature is disabled."
    )]
    #[cfg_attr(
        not(feature = "large-dates"),
        doc = "assert_eq!(UtcDateTime::MAX, utc_datetime!(+9999-12-31 23:59:59.999_999_999));"
    )]
    /// ```
    pub const MAX: Self = Self::new(Date::MAX, Time::MAX);

    /// Create a new `UtcDateTime` with the current date and time.
    #[cfg(feature = "std")]
    #[inline]
    pub fn now() -> Self {
        #[cfg(all(
            target_family = "wasm",
            not(any(target_os = "emscripten", target_os = "wasi")),
            feature = "wasm-bindgen"
        ))]
        {
            js_sys::Date::new_0().into()
        }

        #[cfg(not(all(
            target_family = "wasm",
            not(any(target_os = "emscripten", target_os = "wasi")),
            feature = "wasm-bindgen"
        )))]
        std::time::SystemTime::now().into()
    }

    /// Create a new `UtcDateTime` from the provided [`Date`] and [`Time`].
    #[inline]
    pub const fn new(date: Date, time: Time) -> Self {
        Self {
            inner: PlainDateTime::new(date, time),
        }
    }

    /// Create a new `UtcDateTime` from the [`PlainDateTime`], assuming that the latter is UTC.
    #[inline]
    pub(crate) const fn from_plain(date_time: PlainDateTime) -> Self {
        Self { inner: date_time }
    }

    /// Obtain the [`PlainDateTime`] that this `UtcDateTime` represents. The no-longer-attached
    /// [`UtcOffset`] is assumed to be UTC.
    #[inline]
    pub(crate) const fn as_plain(self) -> PlainDateTime {
        self.inner
    }

    /// Create a `UtcDateTime` from the provided Unix timestamp.
    #[inline]
    pub const fn from_unix_timestamp(timestamp: i64) -> Result<Self, ComponentRange> {
        type Timestamp = ri64<
            { UtcDateTime::MIN.unix_timestamp() },
            { UtcDateTime::MAX.unix_timestamp() },
        >;
        ensure_ranged!(Timestamp: timestamp);

        // Use the unchecked method here, as the input validity has already been verified.
        // Safety: The Julian day number is in range.
        let date = unsafe {
            Date::from_julian_day_unchecked(
                UNIX_EPOCH_JULIAN_DAY
                    + div_floor!(timestamp, Second::per_t::<i64>(Day)) as i32,
            )
        };

        let seconds_within_day = timestamp.rem_euclid(Second::per_t::<i64>(Day));
        // Safety: All values are in range.
        let time = unsafe {
            Time::from_hms_nanos_unchecked(
                (seconds_within_day / Second::per_t::<i64>(Hour)) as u8,
                ((seconds_within_day % Second::per_t::<i64>(Hour))
                    / Minute::per_t::<i64>(Hour)) as u8,
                (seconds_within_day % Second::per_t::<i64>(Minute)) as u8,
                0,
            )
        };

        Ok(Self::new(date, time))
    }

    /// Construct an `UtcDateTime` from the provided Unix timestamp (in nanoseconds).
    #[inline]
    pub const fn from_unix_timestamp_nanos(
        timestamp: i128,
    ) -> Result<Self, ComponentRange> {
        let seconds = div_floor!(timestamp, Nanosecond::per_t::<i128>(Second));
        if seconds < Seconds::MIN.get() as i128 || seconds > Seconds::MAX.get() as i128 {
            return Err(ComponentRange::unconditional("timestamp"));
        }

        let Ok(datetime) = Self::from_unix_timestamp(seconds as i64) else {
            // Safety: The range was just validated.
            unsafe { core::hint::unreachable_unchecked() };
        };

        Ok(Self::new(
            datetime.date(),
            // Safety: `nanosecond` is in range due to `rem_euclid`.
            unsafe {
                Time::from_hms_nanos_unchecked(
                    datetime.hour(),
                    datetime.minute(),
                    datetime.second(),
                    timestamp.rem_euclid(Nanosecond::per_t(Second)) as u32,
                )
            },
        ))
    }

    /// Convert the `UtcDateTime` from UTC to the provided [`UtcOffset`], returning an
    /// [`OffsetDateTime`].
    ///
    /// # Panics
    ///
    /// This method panics if the local date-time in the new offset is outside the supported range.
    #[inline]
    pub const fn to_offset(self, offset: UtcOffset) -> OffsetDateTime {
        match self.checked_to_offset(offset) {
            Some(value) => value,
            None => OffsetDateTime::new_in_offset(Date::MIN, Time::MIDNIGHT, offset),
        }
    }

    /// Convert the `UtcDateTime` from UTC to the provided [`UtcOffset`], returning an
    /// [`OffsetDateTime`]. `None` is returned if the date-time in the resulting offset is
    /// invalid.
    #[inline]
    pub const fn checked_to_offset(self, offset: UtcOffset) -> Option<OffsetDateTime> {
        // Fast path for when no conversion is necessary.
        if offset.is_utc() {
            return Some(self.inner.assume_utc());
        }
        let (year, ordinal, time) = self.to_offset_raw(offset);
        if year > MAX_YEAR || year < MIN_YEAR {
            return None;
        }
        Some(OffsetDateTime::new_in_offset(
            // Safety: `ordinal` is not zero.
            unsafe { Date::from_ordinal_date_unchecked(year, ordinal) },
            time,
            offset,
        ))
    }

    /// Equivalent to `.to_offset(offset)`, but returning the year, ordinal, and time. This avoids
    /// constructing an invalid [`Date`] if the new value is out of range.
    #[inline]
    pub(crate) const fn to_offset_raw(self, offset: UtcOffset) -> (i32, u16, Time) {
        let (second, carry) = carry!(@most_once
            self.second().cast_signed() + offset.seconds_past_minute(),
            0..Second::per_t(Minute)
        );
        let (minute, carry) = carry!(@most_once
            self.minute().cast_signed() + offset.minutes_past_hour() + carry,
            0..Minute::per_t(Hour)
        );
        let (hour, carry) = carry!(@most_twice
            self.hour().cast_signed() + offset.whole_hours() + carry,
            0..Hour::per_t(Day)
        );
        let (mut year, ordinal) = self.to_ordinal_date();
        let mut ordinal = ordinal.cast_signed() + carry;
        cascade!(ordinal => year);

        debug_assert!(ordinal > 0);
        debug_assert!(ordinal <= days_in_year(year).cast_signed());
        (
            year,
            ordinal.cast_unsigned(),
            // Safety: The cascades above ensure the values are in range.
            unsafe {
                Time::from_hms_nanos_unchecked(
                    hour.cast_unsigned(),
                    minute.cast_unsigned(),
                    second.cast_unsigned(),
                    self.nanosecond(),
                )
            },
        )
    }

    /// Get the [Unix timestamp](https://en.wikipedia.org/wiki/Unix_time).
    #[inline]
    pub const fn unix_timestamp(self) -> i64 {
        let days = (self.to_julian_day() as i64 - UNIX_EPOCH_JULIAN_DAY as i64)
            * Second::per_t::<i64>(Day);
        let hours = self.hour() as i64 * Second::per_t::<i64>(Hour);
        let minutes = self.minute() as i64 * Second::per_t::<i64>(Minute);
        let seconds = self.second() as i64;
        days + hours + minutes + seconds
    }

    /// Get the Unix timestamp in nanoseconds.
    #[inline]
    pub const fn unix_timestamp_nanos(self) -> i128 {
        self.unix_timestamp() as i128 * Nanosecond::per_t::<i128>(Second)
            + self.nanosecond() as i128
    }

    /// Get the [`Date`] component of the `UtcDateTime`.
    #[inline]
    pub const fn date(self) -> Date {
        self.inner.date()
    }

    /// Get the [`Time`] component of the `UtcDateTime`.
    #[inline]
    pub const fn time(self) -> Time {
        self.inner.time()
    }

    /// Get the year of the date.
    #[inline]
    pub const fn year(self) -> i32 {
        self.date().year()
    }

    /// Get the month of the date.
    #[inline]
    pub const fn month(self) -> Month {
        self.date().month()
    }

    /// Get the day of the date.
    ///
    /// The returned value will always be in the range `1..=31`.
    #[inline]
    pub const fn day(self) -> u8 {
        self.date().day()
    }

    /// Get the day of the year.
    ///
    /// The returned value will always be in the range `1..=366` (`1..=365` for common years).
    #[inline]
    pub const fn ordinal(self) -> u16 {
        self.date().ordinal()
    }

    /// Get the ISO week number.
    ///
    /// The returned value will always be in the range `1..=53`.
    #[inline]
    pub const fn iso_week(self) -> u8 {
        self.date().iso_week()
    }

    /// Get the week number where week 1 begins on the first Sunday.
    ///
    /// The returned value will always be in the range `0..=53`.
    #[inline]
    pub const fn sunday_based_week(self) -> u8 {
        self.date().sunday_based_week()
    }

    /// Get the week number where week 1 begins on the first Monday.
    ///
    /// The returned value will always be in the range `0..=53`.
    #[inline]
    pub const fn monday_based_week(self) -> u8 {
        self.date().monday_based_week()
    }

    /// Get the year, month, and day.
    #[inline]
    pub const fn to_calendar_date(self) -> (i32, Month, u8) {
        self.date().to_calendar_date()
    }

    /// Get the year and ordinal day number.
    #[inline]
    pub const fn to_ordinal_date(self) -> (i32, u16) {
        self.date().to_ordinal_date()
    }

    /// Get the ISO 8601 year, week number, and weekday.
    #[inline]
    pub const fn to_iso_week_date(self) -> (i32, u8, Weekday) {
        self.date().to_iso_week_date()
    }

    /// Get the weekday.
    #[inline]
    pub const fn weekday(self) -> Weekday {
        self.date().weekday()
    }

    /// Get the Julian day for the date. The time is not taken into account for this calculation.
    #[inline]
    pub const fn to_julian_day(self) -> i32 {
        self.date().to_julian_day()
    }

    /// Get the clock hour, minute, and second.
    #[inline]
    pub const fn as_hms(self) -> (u8, u8, u8) {
        self.time().as_hms()
    }

    /// Get the clock hour, minute, second, and millisecond.
    #[inline]
    pub const fn as_hms_milli(self) -> (u8, u8, u8, u16) {
        self.time().as_hms_milli()
    }

    /// Get the clock hour, minute, second, and microsecond.
    #[inline]
    pub const fn as_hms_micro(self) -> (u8, u8, u8, u32) {
        self.time().as_hms_micro()
    }

    /// Get the clock hour, minute, second, and nanosecond.
    #[inline]
    pub const fn as_hms_nano(self) -> (u8, u8, u8, u32) {
        self.time().as_hms_nano()
    }

    /// Get the clock hour.
    ///
    /// The returned value will always be in the range `0..24`.
    #[inline]
    pub const fn hour(self) -> u8 {
        self.time().hour()
    }

    /// Get the minute within the hour.
    ///
    /// The returned value will always be in the range `0..60`.
    #[inline]
    pub const fn minute(self) -> u8 {
        self.time().minute()
    }

    /// Get the second within the minute.
    ///
    /// The returned value will always be in the range `0..60`.
    #[inline]
    pub const fn second(self) -> u8 {
        self.time().second()
    }

    /// Get the milliseconds within the second.
    ///
    /// The returned value will always be in the range `0..1_000`.
    #[inline]
    pub const fn millisecond(self) -> u16 {
        self.time().millisecond()
    }

    /// Get the microseconds within the second.
    ///
    /// The returned value will always be in the range `0..1_000_000`.
    #[inline]
    pub const fn microsecond(self) -> u32 {
        self.time().microsecond()
    }

    /// Get the nanoseconds within the second.
    ///
    /// The returned value will always be in the range `0..1_000_000_000`.
    #[inline]
    pub const fn nanosecond(self) -> u32 {
        self.time().nanosecond()
    }

    /// Computes `self + duration`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_add(self, duration: SignedDuration) -> Option<Self> {
        Some(Self::from_plain(const_try_opt!(
            self.inner.checked_add(duration)
        )))
    }

    /// Computes `self - duration`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_sub(self, duration: SignedDuration) -> Option<Self> {
        Some(Self::from_plain(const_try_opt!(
            self.inner.checked_sub(duration)
        )))
    }

    /// Computes `self + duration`, saturating value on overflow.
    #[inline]
    pub const fn saturating_add(self, duration: SignedDuration) -> Self {
        Self::from_plain(self.inner.saturating_add(duration))
    }

    /// Computes `self - duration`, saturating value on overflow.
    #[inline]
    pub const fn saturating_sub(self, duration: SignedDuration) -> Self {
        Self::from_plain(self.inner.saturating_sub(duration))
    }
}

/// Methods that replace part of the `UtcDateTime`.
impl UtcDateTime {
    /// Replace the time, preserving the date.
    #[inline]
    pub const fn replace_time(self, time: Time) -> Self {
        Self::from_plain(self.inner.replace_time(time))
    }

    /// Replace the date, preserving the time.
    #[inline]
    pub const fn replace_date(self, date: Date) -> Self {
        Self::from_plain(self.inner.replace_date(date))
    }

    /// Replace the year. The month and day will be unchanged.
    #[inline]
    pub const fn replace_year(self, year: i32) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(self.inner.replace_year(year))))
    }

    /// Replace the month of the year.
    #[inline]
    pub const fn replace_month(self, month: Month) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_month(month)
        )))
    }

    /// Replace the day of the month.
    #[inline]
    pub const fn replace_day(self, day: u8) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(self.inner.replace_day(day))))
    }

    /// Replace the day of the year.
    #[inline]
    pub const fn replace_ordinal(self, ordinal: u16) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_ordinal(ordinal)
        )))
    }

    /// Truncate to the start of the day, setting the time to midnight.
    #[inline]
    pub const fn truncate_to_day(self) -> Self {
        Self::from_plain(self.inner.truncate_to_day())
    }

    /// Replace the clock hour.
    #[inline]
    pub const fn replace_hour(self, hour: u8) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(self.inner.replace_hour(hour))))
    }

    /// Truncate to the hour, setting the minute, second, and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_hour(self) -> Self {
        Self::from_plain(self.inner.truncate_to_hour())
    }

    /// Replace the minutes within the hour.
    #[inline]
    pub const fn replace_minute(self, minute: u8) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_minute(minute)
        )))
    }

    /// Truncate to the minute, setting the second and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_minute(self) -> Self {
        Self::from_plain(self.inner.truncate_to_minute())
    }

    /// Replace the seconds within the minute.
    #[inline]
    pub const fn replace_second(self, second: u8) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_second(second)
        )))
    }

    /// Truncate to the second, setting the subsecond components to zero.
    #[inline]
    pub const fn truncate_to_second(self) -> Self {
        Self::from_plain(self.inner.truncate_to_second())
    }

    /// Replace the milliseconds within the second.
    #[inline]
    pub const fn replace_millisecond(
        self,
        millisecond: u16,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_millisecond(millisecond)
        )))
    }

    /// Truncate to the millisecond, setting the microsecond and nanosecond components to zero.
    #[inline]
    pub const fn truncate_to_millisecond(self) -> Self {
        Self::from_plain(self.inner.truncate_to_millisecond())
    }

    /// Replace the microseconds within the second.
    #[inline]
    pub const fn replace_microsecond(
        self,
        microsecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_microsecond(microsecond)
        )))
    }

    /// Truncate to the microsecond, setting the nanosecond component to zero.
    #[must_use = "This method does not mutate the original `UtcDateTime`."]
    #[inline]
    pub const fn truncate_to_microsecond(self) -> Self {
        Self::from_plain(self.inner.truncate_to_microsecond())
    }

    /// Replace the nanoseconds within the second.
    #[inline]
    pub const fn replace_nanosecond(
        self,
        nanosecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_plain(const_try!(
            self.inner.replace_nanosecond(nanosecond)
        )))
    }
}

// This no longer needs special handling, as the format is fixed and doesn't require anything
// advanced. Trait impls can't be deprecated and the info is still useful for other types
// implementing `SmartDisplay`, so leave it as-is for now.
impl SmartDisplay for UtcDateTime {
    type Metadata = ();

    #[inline]
    fn metadata(&self, f: FormatterOptions) -> Metadata<'_, Self> {
        let width = self.as_plain().metadata(f).unpadded_width() + 4;
        Metadata::new(width, self, ())
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl UtcDateTime {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// for the `Display` implementation.
    pub(crate) const DISPLAY_BUFFER_SIZE: usize = PlainDateTime::DISPLAY_BUFFER_SIZE + 4;

    /// Format the `PlainDateTime` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let pdt_len = self
            .inner
            .fmt_into_buffer(unsafe { buf.first_chunk_mut().unwrap_unchecked() });
        // Safety: The buffer is large enough to hold the additional 4 bytes.
        unsafe {
            b" +00"
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(pdt_len).cast(), 4)
        };
        pdt_len + 4
    }
}

impl fmt::Display for UtcDateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); Self::DISPLAY_BUFFER_SIZE];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts(buf.as_ptr().cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for UtcDateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Add<SignedDuration> for UtcDateTime {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add(self, duration: SignedDuration) -> Self::Output {
        self.inner.add(duration).as_utc()
    }
}

impl Add<StdDuration> for UtcDateTime {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add(self, duration: StdDuration) -> Self::Output {
        self.inner.add(duration).as_utc()
    }
}

impl AddAssign<SignedDuration> for UtcDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        self.inner.add_assign(rhs);
    }
}

impl AddAssign<StdDuration> for UtcDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: StdDuration) {
        self.inner.add_assign(rhs);
    }
}

impl Sub<SignedDuration> for UtcDateTime {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SignedDuration) -> Self::Output {
        self.checked_sub(rhs)
            .unwrap_or_else(|| self.saturating_sub(rhs))
    }
}

impl Sub<StdDuration> for UtcDateTime {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub(self, duration: StdDuration) -> Self::Output {
        Self::from_plain(self.inner.sub(duration))
    }
}

impl SubAssign<SignedDuration> for UtcDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        self.inner.sub_assign(rhs);
    }
}

impl SubAssign<StdDuration> for UtcDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: StdDuration) {
        self.inner.sub_assign(rhs);
    }
}

impl Sub for UtcDateTime {
    type Output = SignedDuration;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.inner.sub(rhs.inner)
    }
}
