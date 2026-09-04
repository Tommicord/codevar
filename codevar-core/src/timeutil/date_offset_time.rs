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

//! The [`OffsetDateTime`] struct and its associated `impl`s.

use alloc::string::String;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use std::time::SystemTime;

use deranged::ri64;
use num_conv::prelude::*;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

use crate::timeutil::date::{Date, MAX_YEAR, MIN_YEAR};
use crate::timeutil::date_error::{ComponentRange, IndeterminateOffset};
use crate::timeutil::date_internal_macro::{
    carry, cascade, const_try, const_try_opt, div_floor, ensure_ranged,
};
use crate::timeutil::date_month::Month;
use crate::timeutil::date_num_fmt::str_from_raw_parts;
use crate::timeutil::date_plain::PlainDateTime;
use crate::timeutil::date_signed_duration::SignedDuration;
use crate::timeutil::date_time::{Seconds, Time};
use crate::timeutil::date_unit::{Day, Hour, Minute, Nanosecond, Second};
use crate::timeutil::date_utc_offset::UtcOffset;
use crate::timeutil::date_utc_time::UtcDateTime;
use crate::timeutil::date_util::days_in_year;
use crate::timeutil::date_weekday::Weekday;

/// The Julian day of the Unix epoch.
const UNIX_EPOCH_JULIAN_DAY: i32 = OffsetDateTime::UNIX_EPOCH.to_julian_day();

/// A [`PlainDateTime`] with a [`UtcOffset`].
#[derive(Clone, Copy, Eq)]
pub struct OffsetDateTime {
    local_date_time: PlainDateTime,
    offset: UtcOffset,
}

impl PartialEq for OffsetDateTime {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        raw_to_bits((self.year(), self.ordinal(), self.time()))
            == raw_to_bits(other.to_offset_raw(self.offset()))
    }
}

impl PartialOrd for OffsetDateTime {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OffsetDateTime {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        raw_to_bits((self.year(), self.ordinal(), self.time()))
            .cmp(&raw_to_bits(other.to_offset_raw(self.offset())))
    }
}

impl Hash for OffsetDateTime {
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        raw_to_bits(self.to_utc_raw()).hash(state);
    }
}

impl From<SystemTime> for OffsetDateTime {
    #[inline]
    fn from(value: SystemTime) -> Self {
        let utc = UtcDateTime::from(value);
        utc.to_offset(UtcOffset::UTC)
    }
}

/// **Note**: This value is explicitly signed, so do not cast this to or treat this as an
/// unsigned integer. Doing so will lead to incorrect results for values with differing
/// signs.
#[inline]
const fn raw_to_bits((year, ordinal, time): (i32, u16, Time)) -> i128 {
    ((year as i128) << 74) | ((ordinal as i128) << 64) | (time.as_u64() as i128)
}

impl OffsetDateTime {
    /// Midnight, 1 January, 1970 (UTC).
    pub const UNIX_EPOCH: Self =
        Self::new_in_offset(Date::UNIX_EPOCH, Time::MIDNIGHT, UtcOffset::UTC);

    /// Create a new `OffsetDateTime` with the current date and time in UTC.
    #[inline]
    pub fn now_utc() -> Self {
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
        SystemTime::now().into()
    }

    /// Attempt to create a new `OffsetDateTime` with the current date and time in the local offset.
    /// If the offset cannot be determined, an error is returned.
    #[inline]
    pub fn now_local() -> Result<Self, IndeterminateOffset> {
        let t = Self::now_utc();
        Ok(t.to_offset(UtcOffset::local_offset_at(t)?))
    }

    /// Create a new `OffsetDateTime` with the given [`Date`], [`Time`], and [`UtcOffset`].
    #[inline]
    pub const fn new_in_offset(date: Date, time: Time, offset: UtcOffset) -> Self {
        Self {
            local_date_time: date.with_time(time),
            offset,
        }
    }

    /// Create a new `OffsetDateTime` with the given [`Date`] and [`Time`] in the UTC timezone.
    #[inline]
    pub const fn new_utc(date: Date, time: Time) -> Self {
        PlainDateTime::new(date, time).assume_utc()
    }

    /// Convert the `OffsetDateTime` from the current [`UtcOffset`] to the provided [`UtcOffset`].
    ///
    /// # Panics
    ///
    /// This method panics if the local date-time in the new offset is outside the supported range.
    #[inline]
    pub const fn to_offset(self, offset: UtcOffset) -> Self {
        match self.checked_to_offset(offset) {
            Some(value) => value,
            None => Self::new_in_offset(Date::MIN, Time::MIDNIGHT, offset),
        }
    }

    /// Convert the `OffsetDateTime` from the current [`UtcOffset`] to the provided [`UtcOffset`],
    /// returning `None` if the date-time in the resulting offset is invalid.
    #[inline]
    pub const fn checked_to_offset(self, offset: UtcOffset) -> Option<Self> {
        if self.offset.as_u32_for_equality() == offset.as_u32_for_equality() {
            return Some(self);
        }

        let (year, ordinal, time) = self.to_offset_raw(offset);

        if year > MAX_YEAR || year < MIN_YEAR {
            return None;
        }

        Some(Self::new_in_offset(
            // Safety: `ordinal` is not zero.
            unsafe { Date::from_ordinal_date_unchecked(year, ordinal) },
            time,
            offset,
        ))
    }

    /// Convert the `OffsetDateTime` from the current [`UtcOffset`] to UTC, returning a
    /// [`UtcDateTime`].
    #[inline]
    pub const fn to_utc(self) -> UtcDateTime {
        match self.checked_to_utc() {
            Some(value) => value,
            None => {
                UtcDateTime::from_plain(PlainDateTime::new(Date::MIN, Time::MIDNIGHT))
            }
        }
    }

    /// Convert the `OffsetDateTime` from the current [`UtcOffset`] to UTC, returning `None` if the
    /// UTC date-time is invalid. Returns a [`UtcDateTime`].
    #[inline]
    pub const fn checked_to_utc(self) -> Option<UtcDateTime> {
        if self.offset.is_utc() {
            return Some(self.local_date_time.as_utc());
        }

        let (year, ordinal, time) = self.to_utc_raw();

        if year > MAX_YEAR || year < MIN_YEAR {
            return None;
        }
        Some(UtcDateTime::new(
            // Safety: `ordinal` is not zero.
            unsafe { Date::from_ordinal_date_unchecked(year, ordinal) },
            time,
        ))
    }

    /// Equivalent to `.to_utc()`, but returning the year, ordinal, and time. This avoids
    /// constructing an invalid [`Date`] if the new value is out of range.
    #[inline]
    pub(crate) const fn to_utc_raw(self) -> (i32, u16, Time) {
        let from = self.offset;

        // Fast path for when no conversion is necessary.
        if from.is_utc() {
            return (self.year(), self.ordinal(), self.time());
        }
        let (second, carry) = carry!(@most_once
            self.second().cast_signed() - from.seconds_past_minute(),
            0..Second::per_t(Minute)
        );
        let (minute, carry) = carry!(@most_once
            self.minute().cast_signed() - from.minutes_past_hour() + carry,
            0..Minute::per_t(Hour)
        );
        let (hour, carry) = carry!(@most_twice
            self.hour().cast_signed() - from.whole_hours() + carry,
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

    /// Equivalent to `.to_offset(offset)`, but returning the year, ordinal, and time. This avoids
    /// constructing an invalid [`Date`] if the new value is out of range.
    #[inline]
    pub(crate) const fn to_offset_raw(self, offset: UtcOffset) -> (i32, u16, Time) {
        let from = self.offset;
        let to = offset;

        if from.as_u32_for_equality() == to.as_u32_for_equality() {
            return (self.year(), self.ordinal(), self.time());
        }
        let (second, carry) = carry!(@most_twice
            self.second() as i16 - from.seconds_past_minute() as i16
                + to.seconds_past_minute() as i16,
            0..Second::per_t(Minute)
        );
        let (minute, carry) = carry!(@most_twice
            self.minute() as i16 - from.minutes_past_hour() as i16
                + to.minutes_past_hour() as i16
                + carry,
            0..Minute::per_t(Hour)
        );
        let (hour, carry) = carry!(@most_thrice
            self.hour().cast_signed() - from.whole_hours() + to.whole_hours() + carry,
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
                    minute as u8,
                    second as u8,
                    self.nanosecond(),
                )
            },
        )
    }

    /// Create an `OffsetDateTime` from the provided Unix timestamp. Calling `.offset()` on the
    /// resulting value is guaranteed to return UTC.
    #[inline]
    pub const fn from_unix_timestamp(timestamp: i64) -> Result<Self, ComponentRange> {
        type Timestamp = ri64<
            {
                OffsetDateTime::new_in_offset(Date::MIN, Time::MIDNIGHT, UtcOffset::UTC)
                    .unix_timestamp()
            },
            {
                OffsetDateTime::new_in_offset(Date::MAX, Time::MAX, UtcOffset::UTC)
                    .unix_timestamp()
            },
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

        let seconds_within_day = timestamp.rem_euclid(Second::per_t(Day));
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

        Ok(Self::new_in_offset(date, time, UtcOffset::UTC))
    }

    /// Construct an `OffsetDateTime` from the provided Unix timestamp (in nanoseconds). Calling
    /// `.offset()` on the resulting value is guaranteed to return UTC.
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

        Ok(Self::new_in_offset(
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
            UtcOffset::UTC,
        ))
    }

    /// Get the [`UtcOffset`].
    #[inline]
    pub const fn offset(self) -> UtcOffset {
        self.offset
    }

    /// Get the [Unix timestamp](https://en.wikipedia.org/wiki/Unix_time).
    #[inline]
    pub const fn unix_timestamp(self) -> i64 {
        let days = (self.to_julian_day() as i64 - UNIX_EPOCH_JULIAN_DAY as i64)
            * Second::per_t::<i64>(Day);
        let hours = self.hour() as i64 * Second::per_t::<i64>(Hour);
        let minutes = self.minute() as i64 * Second::per_t::<i64>(Minute);
        let seconds = self.second() as i64;
        let offset_seconds = self.offset.whole_seconds() as i64;
        days + hours + minutes + seconds - offset_seconds
    }

    /// Get the Unix timestamp in nanoseconds.
    #[inline]
    pub const fn unix_timestamp_nanos(self) -> i128 {
        self.unix_timestamp() as i128 * Nanosecond::per_t::<i128>(Second)
            + self.nanosecond() as i128
    }

    /// Get the [`PlainDateTime`] in the stored offset.
    #[inline]
    pub(crate) const fn date_time(self) -> PlainDateTime {
        self.local_date_time
    }

    /// Get the [`Date`] in the stored offset.
    #[inline]
    pub const fn date(self) -> Date {
        self.date_time().date()
    }

    /// Get the [`Time`] in the stored offset.
    #[inline]
    pub const fn time(self) -> Time {
        self.date_time().time()
    }

    /// Get the year of the date in the stored offset.
    #[inline]
    pub const fn year(self) -> i32 {
        self.date().year()
    }

    /// Get the month of the date in the stored offset.
    #[inline]
    pub const fn month(self) -> Month {
        self.date().month()
    }

    /// Get the day of the date in the stored offset.
    ///
    /// The returned value will always be in the range `1..=31`.
    #[inline]
    pub const fn day(self) -> u8 {
        self.date().day()
    }

    /// Get the day of the year of the date in the stored offset.
    ///
    /// The returned value will always be in the range `1..=366`.
    #[inline]
    pub const fn ordinal(self) -> u16 {
        self.date().ordinal()
    }

    /// Get the ISO week number of the date in the stored offset.
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

    /// Get the weekday of the date in the stored offset.
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
    pub const fn to_hms(self) -> (u8, u8, u8) {
        self.time().as_hms()
    }

    /// Get the clock hour, minute, second, and millisecond.
    #[inline]
    pub const fn to_hms_milli(self) -> (u8, u8, u8, u16) {
        self.time().as_hms_milli()
    }

    /// Get the clock hour, minute, second, and microsecond.
    #[inline]
    pub const fn to_hms_micro(self) -> (u8, u8, u8, u32) {
        self.time().as_hms_micro()
    }

    /// Get the clock hour, minute, second, and nanosecond.
    #[inline]
    pub const fn to_hms_nano(self) -> (u8, u8, u8, u32) {
        self.time().as_hms_nano()
    }

    /// Get the clock hour in the stored offset.
    ///
    /// The returned value will always be in the range `0..24`.
    #[inline]
    pub const fn hour(self) -> u8 {
        self.time().hour()
    }

    /// Get the minute within the hour in the stored offset.
    ///
    /// The returned value will always be in the range `0..60`.
    #[inline]
    pub const fn minute(self) -> u8 {
        self.time().minute()
    }

    /// Get the second within the minute in the stored offset.
    ///
    /// The returned value will always be in the range `0..60`.
    #[inline]
    pub const fn second(self) -> u8 {
        self.time().second()
    }

    // Because a `UtcOffset` is limited in resolution to one second, any subsecond value will not
    // change when adjusting for the offset.

    /// Get the milliseconds within the second in the stored offset.
    ///
    /// The returned value will always be in the range `0..1_000`.
    #[inline]
    pub const fn millisecond(self) -> u16 {
        self.time().millisecond()
    }

    /// Get the microseconds within the second in the stored offset.
    ///
    /// The returned value will always be in the range `0..1_000_000`.
    #[inline]
    pub const fn microsecond(self) -> u32 {
        self.time().microsecond()
    }

    /// Get the nanoseconds within the second in the stored offset.
    ///
    /// The returned value will always be in the range `0..1_000_000_000`.
    #[inline]
    pub const fn nanosecond(self) -> u32 {
        self.time().nanosecond()
    }

    /// Computes `self + duration`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_add(self, duration: SignedDuration) -> Option<Self> {
        Some(
            const_try_opt!(self.date_time().checked_add(duration))
                .assume_offset(self.offset()),
        )
    }

    /// Computes `self - duration`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_sub(self, duration: SignedDuration) -> Option<Self> {
        Some(
            const_try_opt!(self.date_time().checked_sub(duration))
                .assume_offset(self.offset()),
        )
    }

    /// Computes `self + duration`, saturating value on overflow.
    #[inline]
    pub const fn saturating_add(self, duration: SignedDuration) -> Self {
        if let Some(datetime) = self.checked_add(duration) {
            datetime
        } else if duration.is_negative() {
            PlainDateTime::MIN.assume_offset(self.offset())
        } else {
            PlainDateTime::MAX.assume_offset(self.offset())
        }
    }

    /// Computes `self - duration`, saturating value on overflow.
    pub const fn saturating_sub(self, duration: SignedDuration) -> Self {
        if let Some(datetime) = self.checked_sub(duration) {
            datetime
        } else if duration.is_negative() {
            PlainDateTime::MAX.assume_offset(self.offset())
        } else {
            PlainDateTime::MIN.assume_offset(self.offset())
        }
    }
}

/// Methods that replace part of the `OffsetDateTime`.
impl OffsetDateTime {
    /// Replace the time, which is assumed to be in the stored offset. The date and offset
    /// components are unchanged.
    #[inline]
    pub const fn replace_time(self, time: Time) -> Self {
        Self::new_in_offset(self.date(), time, self.offset())
    }

    /// Replace the date, which is assumed to be in the stored offset. The time and offset
    /// components are unchanged.
    #[inline]
    pub const fn replace_date(self, date: Date) -> Self {
        Self::new_in_offset(date, self.time(), self.offset())
    }

    /// Replace the date and time, which are assumed to be in the stored offset. The offset
    /// component remains unchanged.
    #[inline]
    pub const fn replace_date_time(self, date_time: PlainDateTime) -> Self {
        date_time.assume_offset(self.offset())
    }

    /// Replace the offset. The date and time components remain unchanged.
    #[inline]
    pub const fn replace_offset(self, offset: UtcOffset) -> Self {
        self.date_time().assume_offset(offset)
    }

    /// Replace the year. The month and day will be unchanged.
    #[inline]
    pub const fn replace_year(self, year: i32) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_year(year)).assume_offset(self.offset()))
    }

    /// Replace the month of the year.
    #[inline]
    pub const fn replace_month(self, month: Month) -> Result<Self, ComponentRange> {
        Ok(
            const_try!(self.date_time().replace_month(month))
                .assume_offset(self.offset()),
        )
    }

    /// Replace the day of the month.
    #[inline]
    pub const fn replace_day(self, day: u8) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_day(day)).assume_offset(self.offset()))
    }

    /// Replace the day of the year.
    #[inline]
    pub const fn replace_ordinal(self, ordinal: u16) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_ordinal(ordinal))
            .assume_offset(self.offset()))
    }

    /// Truncate to the start of the day, setting the time to midnight.
    #[inline]
    pub const fn truncate_to_day(mut self) -> Self {
        self.local_date_time = self.local_date_time.truncate_to_day();
        self
    }

    /// Replace the clock hour.
    #[inline]
    pub const fn replace_hour(self, hour: u8) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_hour(hour)).assume_offset(self.offset()))
    }

    /// Truncate to the hour, setting the minute, second, and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_hour(mut self) -> Self {
        self.local_date_time = self.local_date_time.truncate_to_hour();
        self
    }

    /// Replace the minutes within the hour.
    #[inline]
    pub const fn replace_minute(self, minute: u8) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_minute(minute))
            .assume_offset(self.offset()))
    }

    /// Truncate to the minute, setting the second and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_minute(mut self) -> Self {
        self.local_date_time = self.local_date_time.truncate_to_minute();
        self
    }

    /// Replace the seconds within the minute.
    #[inline]
    pub const fn replace_second(self, second: u8) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_second(second))
            .assume_offset(self.offset()))
    }

    /// Truncate to the second, setting the subsecond components to zero.
    #[inline]
    pub const fn truncate_to_second(mut self) -> Self {
        self.local_date_time = self.local_date_time.truncate_to_second();
        self
    }

    /// Replace the milliseconds within the second.
    #[inline]
    pub const fn replace_millisecond(
        self,
        millisecond: u16,
    ) -> Result<Self, ComponentRange> {
        Ok(
            const_try!(self.date_time().replace_millisecond(millisecond))
                .assume_offset(self.offset()),
        )
    }

    /// Truncate to the millisecond, setting the microsecond and nanosecond components to zero.
    #[inline]
    pub const fn truncate_to_millisecond(mut self) -> Self {
        self.local_date_time = self.local_date_time.truncate_to_millisecond();
        self
    }

    /// Replace the microseconds within the second.
    #[inline]
    pub const fn replace_microsecond(
        self,
        microsecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(
            const_try!(self.date_time().replace_microsecond(microsecond))
                .assume_offset(self.offset()),
        )
    }

    /// Truncate to the microsecond, setting the nanosecond component to zero.
    #[inline]
    pub const fn truncate_to_microsecond(mut self) -> Self {
        self.local_date_time = self.local_date_time.truncate_to_microsecond();
        self
    }

    /// Replace the nanoseconds within the second.
    #[inline]
    pub const fn replace_nanosecond(
        self,
        nanosecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(const_try!(self.date_time().replace_nanosecond(nanosecond))
            .assume_offset(self.offset()))
    }
}

// This no longer needs special handling, as the format is fixed and doesn't require anything
// advanced. Trait impls can't be deprecated and the info is still useful for other types
// implementing `SmartDisplay`, so leave it as-is for now.
impl SmartDisplay for OffsetDateTime {
    type Metadata = ();

    #[inline]
    fn metadata(&self, f: FormatterOptions) -> Metadata<'_, Self> {
        let width = self.date_time().metadata(f).unpadded_width()
            + UtcOffset::DISPLAY_BUFFER_SIZE
            + 1;
        Metadata::new(width, self, ())
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl OffsetDateTime {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// for the `Display` implementation.
    pub(crate) const DISPLAY_BUFFER_SIZE: usize =
        PlainDateTime::DISPLAY_BUFFER_SIZE + UtcOffset::DISPLAY_BUFFER_SIZE + 1;

    /// Format the `OffsetDateTime` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let date_time_len = self
            .date_time()
            .fmt_into_buffer(unsafe { buf.first_chunk_mut().unwrap_unchecked() });
        buf[date_time_len].write(b' ');
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let offset_len = self.offset().fmt_into_buffer(unsafe {
            buf[date_time_len + 1..]
                .first_chunk_mut()
                .unwrap_unchecked()
        });
        date_time_len + offset_len + 1
    }
}

impl fmt::Display for OffsetDateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); Self::DISPLAY_BUFFER_SIZE];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts(buf.as_ptr().cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for OffsetDateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Add<SignedDuration> for OffsetDateTime {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add(self, duration: SignedDuration) -> Self::Output {
        self.checked_add(duration)
            .unwrap_or_else(|| self.saturating_add(duration))
    }
}

impl Add<StdDuration> for OffsetDateTime {
    type Output = Self;

    #[inline]
    fn add(self, duration: StdDuration) -> Self::Output {
        let (is_next_day, time) = self.time().adjusting_add_std(duration);

        Self::new_in_offset(
            if is_next_day {
                (self.date() + duration).next_day().unwrap_or(Date::MAX)
            } else {
                self.date() + duration
            },
            time,
            self.offset,
        )
    }
}

impl AddAssign<SignedDuration> for OffsetDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        *self = *self + rhs;
    }
}

impl AddAssign<StdDuration> for OffsetDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: StdDuration) {
        *self = *self + rhs;
    }
}

impl Sub<SignedDuration> for OffsetDateTime {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SignedDuration) -> Self::Output {
        self.checked_sub(rhs)
            .unwrap_or_else(|| self.saturating_sub(rhs))
    }
}

impl Sub<StdDuration> for OffsetDateTime {
    type Output = Self;

    #[inline]
    fn sub(self, duration: StdDuration) -> Self::Output {
        let (is_previous_day, time) = self.time().adjusting_sub_std(duration);

        Self::new_in_offset(
            if is_previous_day {
                (self.date() - duration).previous_day().unwrap_or(Date::MIN)
            } else {
                self.date() - duration
            },
            time,
            self.offset,
        )
    }
}

impl SubAssign<SignedDuration> for OffsetDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        *self = *self - rhs;
    }
}

impl SubAssign<StdDuration> for OffsetDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: StdDuration) {
        *self = *self - rhs;
    }
}

impl Sub for OffsetDateTime {
    type Output = SignedDuration;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let base = self.date_time() - rhs.date_time();
        let adjustment = SignedDuration::seconds(
            (self.offset.whole_seconds() - rhs.offset.whole_seconds()).widen::<i64>(),
        );
        base - adjustment
    }
}
