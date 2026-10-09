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

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;

use crate::date::Date;
use crate::error::ComponentRange;
use crate::internal_macro::{const_try, const_try_opt};
use crate::month::Month;
use crate::num_fmt::str_from_raw_parts;
use crate::offset_time::OffsetDateTime;
use crate::signed_duration::SignedDuration;
use crate::time::Time;
use crate::utc_offset::UtcOffset;
use crate::utc_time::UtcDateTime;
use crate::util::DateAdjustment;
use crate::weekday::Weekday;
use powerfmt::smart_display::{self, FormatterOptions, Metadata, SmartDisplay};

/// Combined date and time.
#[derive(Clone, Copy, Eq)]
#[cfg_attr(not(docsrs), repr(C))]
pub struct PlainDateTime {
    #[cfg(target_endian = "little")]
    time: Time,
    #[cfg(target_endian = "little")]
    date: Date,

    #[cfg(target_endian = "big")]
    date: Date,
    #[cfg(target_endian = "big")]
    time: Time,
}

impl Hash for PlainDateTime {
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.as_i128().hash(state);
    }
}

impl PartialEq for PlainDateTime {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_i128().eq(&other.as_i128())
    }
}

impl PartialOrd for PlainDateTime {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PlainDateTime {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_i128().cmp(&other.as_i128())
    }
}

impl PlainDateTime {
    /// Provide a representation of `PlainDateTime` as a `i128`. This value can be used for
    /// equality, hashing, and ordering.
    ///
    /// **Note**: This value is explicitly signed, so do not cast this to or treat this as an
    /// unsigned integer. Doing so will lead to incorrect results for values with differing
    /// signs.
    #[inline]
    const fn as_i128(self) -> i128 {
        let time = self.time.as_u64() as i128;
        let date = self.date.as_i32() as i128;
        (date << 64) | time
    }

    /// The smallest value that can be represented by `PlainDateTime`.
    ///
    /// Depending on `large-dates` feature flag, value of this constant may vary.
    ///
    /// 1. With `large-dates` disabled it is equal to `-9999-01-01 00:00:00.0`
    /// 2. With `large-dates` enabled it is equal to `-999999-01-01 00:00:00.0`
    pub const MIN: Self = Self {
        date: Date::MIN,
        time: Time::MIDNIGHT,
    };

    /// The largest value that can be represented by `PlainDateTime`.
    ///
    /// Depending on `large-dates` feature flag, value of this constant may vary.
    ///
    /// 1. With `large-dates` disabled it is equal to `9999-12-31 23:59:59.999_999_999`
    /// 2. With `large-dates` enabled it is equal to `999999-12-31 23:59:59.999_999_999`
    pub const MAX: Self = Self {
        date: Date::MAX,
        time: Time::MAX,
    };

    /// Create a new `PlainDateTime` from the provided [`Date`] and [`Time`].
    #[inline]
    pub const fn new(date: Date, time: Time) -> Self {
        Self { date, time }
    }

    /// Get the [`Date`] component of the `PlainDateTime`.
    #[inline]
    pub const fn date(self) -> Date {
        self.date
    }

    /// Get the [`Time`] component of the `PlainDateTime`.
    #[inline]
    pub const fn time(self) -> Time {
        self.time
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
    #[inline]
    pub const fn hour(self) -> u8 {
        self.time().hour()
    }

    /// Get the minute within the hour.
    #[inline]
    pub const fn minute(self) -> u8 {
        self.time().minute()
    }

    /// Get the second within the minute.
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

    /// Assuming that the existing `PlainDateTime` represents a moment in the provided
    /// [`UtcOffset`], return an [`OffsetDateTime`].
    #[inline]
    pub const fn assume_offset(self, offset: UtcOffset) -> OffsetDateTime {
        OffsetDateTime::new_in_offset(self.date, self.time, offset)
    }

    /// Assuming that the existing `PlainDateTime` represents a moment in UTC, return an
    /// [`OffsetDateTime`].
    ///
    /// **Note**: You may want a [`UtcDateTime`] instead, which can be obtained with the
    /// [`PlainDateTime::as_utc`] method.
    #[inline]
    pub const fn assume_utc(self) -> OffsetDateTime {
        self.assume_offset(UtcOffset::UTC)
    }

    /// Assuming that the existing `PlainDateTime` represents a moment in UTC, return a
    /// [`UtcDateTime`].
    #[inline]
    pub const fn as_utc(self) -> UtcDateTime {
        UtcDateTime::from_plain(self)
    }

    /// Computes `self + duration`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_add(self, duration: SignedDuration) -> Option<Self> {
        let (date_adjustment, time) = self.time.adjusting_add(duration);
        let date = const_try_opt!(self.date.checked_add(duration));

        Some(Self {
            date: match date_adjustment {
                DateAdjustment::Previous => const_try_opt!(date.previous_day()),
                DateAdjustment::Next => const_try_opt!(date.next_day()),
                DateAdjustment::None => date,
            },
            time,
        })
    }

    /// Computes `self - duration`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_sub(self, duration: SignedDuration) -> Option<Self> {
        let (date_adjustment, time) = self.time.adjusting_sub(duration);
        let date = const_try_opt!(self.date.checked_sub(duration));

        Some(Self {
            date: match date_adjustment {
                DateAdjustment::Previous => const_try_opt!(date.previous_day()),
                DateAdjustment::Next => const_try_opt!(date.next_day()),
                DateAdjustment::None => date,
            },
            time,
        })
    }

    /// Computes `self + duration`, saturating value on overflow.
    #[inline]
    pub const fn saturating_add(self, duration: SignedDuration) -> Self {
        if let Some(datetime) = self.checked_add(duration) {
            datetime
        } else if duration.is_negative() {
            Self::MIN
        } else {
            Self::MAX
        }
    }

    /// Computes `self - duration`, saturating value on overflow.
    #[inline]
    pub const fn saturating_sub(self, duration: SignedDuration) -> Self {
        if let Some(datetime) = self.checked_sub(duration) {
            datetime
        } else if duration.is_negative() {
            Self::MAX
        } else {
            Self::MIN
        }
    }
}

/// Methods that replace part of the `PlainDateTime`.
impl PlainDateTime {
    /// Replace the time, preserving the date.
    #[must_use = "This method does not mutate the original `PlainDateTime`."]
    #[inline]
    pub const fn replace_time(self, time: Time) -> Self {
        Self {
            date: self.date,
            time,
        }
    }

    /// Replace the date, preserving the time.
    #[must_use = "This method does not mutate the original `PlainDateTime`."]
    #[inline]
    pub const fn replace_date(self, date: Date) -> Self {
        Self {
            date,
            time: self.time,
        }
    }

    /// Replace the year. The month and day will be unchanged.
    #[inline]
    pub const fn replace_year(self, year: i32) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: const_try!(self.date.replace_year(year)),
            time: self.time,
        })
    }

    /// Replace the month of the year.
    #[inline]
    pub const fn replace_month(self, month: Month) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: const_try!(self.date.replace_month(month)),
            time: self.time,
        })
    }

    /// Replace the day of the month.
    #[inline]
    pub const fn replace_day(self, day: u8) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: const_try!(self.date.replace_day(day)),
            time: self.time,
        })
    }

    /// Replace the day of the year.
    #[inline]
    pub const fn replace_ordinal(self, ordinal: u16) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: const_try!(self.date.replace_ordinal(ordinal)),
            time: self.time,
        })
    }

    /// Truncate to the start of the day, setting the time to midnight.
    #[must_use = "This method does not mutate the original `PlainDateTime`."]
    #[inline]
    pub const fn truncate_to_day(self) -> Self {
        self.replace_time(Time::MIDNIGHT)
    }

    /// Replace the clock hour.
    #[inline]
    pub const fn replace_hour(self, hour: u8) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: self.date,
            time: const_try!(self.time.replace_hour(hour)),
        })
    }

    /// Truncate to the hour, setting the minute, second, and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_hour(self) -> Self {
        self.replace_time(self.time.truncate_to_hour())
    }

    /// Replace the minutes within the hour.
    #[inline]
    pub const fn replace_minute(self, minute: u8) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: self.date,
            time: const_try!(self.time.replace_minute(minute)),
        })
    }

    /// Truncate to the minute, setting the second and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_minute(self) -> Self {
        self.replace_time(self.time.truncate_to_minute())
    }

    /// Replace the seconds within the minute.
    #[inline]
    pub const fn replace_second(self, second: u8) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: self.date,
            time: const_try!(self.time.replace_second(second)),
        })
    }

    /// Truncate to the second, setting the subsecond components to zero.
    #[inline]
    pub const fn truncate_to_second(self) -> Self {
        self.replace_time(self.time.truncate_to_second())
    }

    /// Replace the milliseconds within the second.
    #[inline]
    pub const fn replace_millisecond(self, millisecond: u16) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: self.date,
            time: const_try!(self.time.replace_millisecond(millisecond)),
        })
    }

    /// Truncate to the millisecond, setting the microsecond and nanosecond components to zero.
    #[inline]
    pub const fn truncate_to_millisecond(self) -> Self {
        self.replace_time(self.time.truncate_to_millisecond())
    }

    /// Replace the microseconds within the second.
    #[inline]
    pub const fn replace_microsecond(self, microsecond: u32) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: self.date,
            time: const_try!(self.time.replace_microsecond(microsecond)),
        })
    }

    /// Truncate to the microsecond, setting the nanosecond component to zero.
    #[inline]
    pub const fn truncate_to_microsecond(self) -> Self {
        self.replace_time(self.time.truncate_to_microsecond())
    }

    /// Replace the nanoseconds within the second.
    #[inline]
    pub const fn replace_nanosecond(self, nanosecond: u32) -> Result<Self, ComponentRange> {
        Ok(Self {
            date: self.date,
            time: const_try!(self.time.replace_nanosecond(nanosecond)),
        })
    }
}

// This no longer needs special handling, as the format is fixed and doesn't require anything
// advanced. Trait impls can't be deprecated and the info is still useful for other types
// implementing `SmartDisplay`, so leave it as-is for now.
impl SmartDisplay for PlainDateTime {
    type Metadata = ();

    #[inline]
    fn metadata(&self, _: FormatterOptions) -> Metadata<'_, Self> {
        let width = smart_display::padded_width_of!(self.date, " ", self.time);
        Metadata::new(width, self, ())
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl PlainDateTime {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// for the `Display` implementation.
    pub(crate) const DISPLAY_BUFFER_SIZE: usize = Date::DISPLAY_BUFFER_SIZE + Time::DISPLAY_BUFFER_SIZE + 1;

    /// Format the `PlainDateTime` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) fn fmt_into_buffer(self, buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE]) -> usize {
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let date_len = self
            .date
            .fmt_into_buffer(unsafe { buf.first_chunk_mut().unwrap_unchecked() });
        buf[date_len].write(b' ');
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let time_len = self.time.fmt_into_buffer(unsafe {
            buf[date_len + 1..]
                .first_chunk_mut()
                .unwrap_unchecked()
        });
        date_len + time_len + 1
    }
}

impl fmt::Display for PlainDateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); Self::DISPLAY_BUFFER_SIZE];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts(buf.as_ptr().cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for PlainDateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Add<SignedDuration> for PlainDateTime {
    type Output = Self;

    #[inline]
    fn add(self, duration: SignedDuration) -> Self::Output {
        self.checked_add(duration)
            .unwrap_or_else(|| self.saturating_add(duration))
    }
}

impl Add<StdDuration> for PlainDateTime {
    type Output = Self;

    #[inline]
    fn add(self, duration: StdDuration) -> Self::Output {
        let (is_next_day, time) = self.time.adjusting_add_std(duration);

        Self {
            date: if is_next_day {
                (self.date + duration)
                    .next_day()
                    .unwrap_or(Date::MAX)
            } else {
                self.date + duration
            },
            time,
        }
    }
}

impl AddAssign<SignedDuration> for PlainDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, duration: SignedDuration) {
        *self = *self + duration;
    }
}

impl AddAssign<StdDuration> for PlainDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, duration: StdDuration) {
        *self = *self + duration;
    }
}

impl Sub<SignedDuration> for PlainDateTime {
    type Output = Self;

    #[inline]
    fn sub(self, duration: SignedDuration) -> Self::Output {
        self.checked_sub(duration)
            .unwrap_or_else(|| self.saturating_sub(duration))
    }
}

impl Sub<StdDuration> for PlainDateTime {
    type Output = Self;

    #[inline]
    fn sub(self, duration: StdDuration) -> Self::Output {
        let (is_previous_day, time) = self.time.adjusting_sub_std(duration);

        Self {
            date: if is_previous_day {
                (self.date - duration)
                    .previous_day()
                    .unwrap_or(Date::MIN)
            } else {
                self.date - duration
            },
            time,
        }
    }
}

impl SubAssign<SignedDuration> for PlainDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, duration: SignedDuration) {
        *self = *self - duration;
    }
}

impl SubAssign<StdDuration> for PlainDateTime {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, duration: StdDuration) {
        *self = *self - duration;
    }
}

impl Sub for PlainDateTime {
    type Output = SignedDuration;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.date - rhs.date) + (self.time - rhs.time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn datetime(year: i32, month: Month, day: u8, hour: u8, minute: u8, second: u8) -> PlainDateTime {
        PlainDateTime::new(
            Date::from_calendar_date(year, month, day).expect("valid date"),
            Time::from_hms(hour, minute, second).expect("valid time"),
        )
    }

    #[test]
    fn new_exposes_date_and_time_components() {
        let value = datetime(2024, Month::February, 29, 1, 2, 3);
        assert_eq!(
            value.date(),
            Date::from_calendar_date(2024, Month::February, 29).expect("valid")
        );
        assert_eq!(value.time(), Time::from_hms(1, 2, 3).expect("valid"));

        // Date accessors delegate to the date component.
        assert_eq!(value.year(), 2024);
        assert_eq!(value.month(), Month::February);
        assert_eq!(value.day(), 29);
        assert_eq!(value.ordinal(), 60);
        assert_eq!(value.iso_week(), 9);
        assert_eq!(value.to_calendar_date(), (2024, Month::February, 29));
        assert_eq!(value.to_ordinal_date(), (2024, 60));
        assert_eq!(value.to_iso_week_date(), (2024, 9, Weekday::Thursday));
        assert_eq!(value.weekday(), Weekday::Thursday);
        assert_eq!(value.to_julian_day(), value.date().to_julian_day());
        assert_eq!(value.sunday_based_week(), value.date().sunday_based_week(),);
        assert_eq!(value.monday_based_week(), value.date().monday_based_week());

        // Time accessors delegate to the time component.
        assert_eq!(value.as_hms(), (1, 2, 3));
        assert_eq!(value.hour(), 1);
        assert_eq!(value.minute(), 2);
        assert_eq!(value.second(), 3);
        assert_eq!(value.millisecond(), 0);
        assert_eq!(value.nanosecond(), 0);
        let value = value.replace_nanosecond(1_234).expect("valid");
        assert_eq!(value.as_hms_nano(), (1, 2, 3, 1_234));
        // 1_234 ns is a microsecond fraction, not a whole millisecond.
        assert_eq!(value.as_hms_milli(), (1, 2, 3, 0));
        assert_eq!(value.as_hms_micro(), (1, 2, 3, 1));
        assert_eq!(value.microsecond(), 1);
        assert_eq!(value.millisecond(), 0);
    }

    #[test]
    fn min_and_max_bounds() {
        assert_eq!(PlainDateTime::MIN.to_string(), "-9999-01-01 0:00:00.0",);
        assert_eq!(PlainDateTime::MAX.to_string(), "9999-12-31 23:59:59.999999999",);
        assert_eq!(PlainDateTime::MIN.date(), Date::MIN);
        assert_eq!(PlainDateTime::MIN.time(), Time::MIDNIGHT);
        assert_eq!(PlainDateTime::MAX.date(), Date::MAX);
        assert_eq!(PlainDateTime::MAX.time(), Time::MAX);
        assert!(PlainDateTime::MIN < PlainDateTime::MAX);
    }

    #[test]
    fn display_formats_date_and_time() {
        let cases = [
            (
                datetime(2024, Month::February, 29, 1, 2, 3),
                "2024-02-29 1:02:03.0",
            ),
            (datetime(1970, Month::January, 1, 0, 0, 0), "1970-01-01 0:00:00.0"),
            (
                datetime(1, Month::January, 1, 23, 59, 59),
                "0001-01-01 23:59:59.0",
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(value.to_string(), expected, "Display of {value:?}");
            assert_eq!(format!("{value:?}"), expected, "Debug matches Display");
            assert_eq!(
                value
                    .metadata(FormatterOptions::default())
                    .unpadded_width(),
                expected.len(),
                "metadata width for {expected}",
            );
        }

        let value = datetime(1970, Month::January, 1, 1, 2, 3);
        assert_eq!(format!("{value:>22}"), "  1970-01-01 1:02:03.0");
        assert_eq!(format!("{value:<22}"), "1970-01-01 1:02:03.0  ");
        assert_eq!(format!("{value:.10}"), "1970-01-01");
    }

    #[test]
    fn checked_arithmetic_carries_across_midnight() {
        let value = datetime(2024, Month::March, 1, 0, 30, 0);

        // Borrowing an hour crosses back into February.
        assert_eq!(
            value
                .checked_sub(SignedDuration::hours(1))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-02-29 23:30:00.0"),
        );
        // Carrying an hour crosses into the next day.
        assert_eq!(
            value
                .checked_add(SignedDuration::hours(24 * 2))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-03-03 0:30:00.0"),
        );
        // A duration of whole days moves the date only.
        assert_eq!(
            value
                .checked_add(SignedDuration::days(30))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-03-31 0:30:00.0"),
        );
        // Sub-day durations that stay within the day never move the date.
        let noon = datetime(2024, Month::June, 15, 12, 0, 0);
        assert_eq!(
            noon.checked_add(SignedDuration::hours(7))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-06-15 19:00:00.0"),
        );
        // 23 hours from noon lands late the next day.
        assert_eq!(
            noon.checked_add(SignedDuration::hours(23))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-06-16 11:00:00.0"),
        );
        assert_eq!(
            noon.checked_sub(SignedDuration::hours(11))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-06-15 1:00:00.0"),
        );

        // Mixed date and time components: 1 day and 23 hours from just after midnight lands
        // in the previous day's late evening.
        let value = datetime(2024, Month::June, 15, 0, 0, 0);
        assert_eq!(
            value
                .checked_add(SignedDuration::days(1) + SignedDuration::hours(23))
                .map(|v| v.to_string())
                .as_deref(),
            Some("2024-06-16 23:00:00.0"),
        );

        // Round trip.
        let duration = SignedDuration::days(12) + SignedDuration::hours(5) + SignedDuration::seconds(30);
        let value = datetime(2024, Month::June, 15, 8, 45, 12);
        assert_eq!(
            value
                .checked_add(duration)
                .and_then(|v| v.checked_sub(duration)),
            Some(value),
        );
    }

    #[test]
    fn checked_arithmetic_reports_overflow_at_bounds() {
        assert_eq!(
            PlainDateTime::MAX.checked_add(SignedDuration::nanoseconds(1)),
            None
        );
        assert_eq!(
            PlainDateTime::MIN.checked_sub(SignedDuration::nanoseconds(1)),
            None
        );
        assert!(
            PlainDateTime::MAX
                .checked_add(SignedDuration::days(-1))
                .is_some()
        );
        assert!(
            PlainDateTime::MIN
                .checked_add(SignedDuration::days(1))
                .is_some()
        );

        // The very last representable instant cannot be advanced by even one hour.
        let max = PlainDateTime::MAX;
        assert_eq!(max.checked_add(SignedDuration::hours(1)), None);

        assert_eq!(
            PlainDateTime::MAX.saturating_add(SignedDuration::nanoseconds(1)),
            PlainDateTime::MAX,
        );
        assert_eq!(
            PlainDateTime::MIN.saturating_sub(SignedDuration::nanoseconds(1)),
            PlainDateTime::MIN,
        );
        assert_eq!(
            PlainDateTime::MIN
                .saturating_add(SignedDuration::nanoseconds(1))
                .to_string(),
            "-9999-01-01 0:00:00.000000001",
        );

        // Positive durations saturate to MAX and negative ones to MIN.
        let huge = SignedDuration::days(i64::from(i32::MAX) + 100);
        assert_eq!(PlainDateTime::MIN.saturating_add(huge), PlainDateTime::MAX);
        assert_eq!(PlainDateTime::MAX.saturating_sub(huge), PlainDateTime::MIN);
        assert_eq!(PlainDateTime::MAX.saturating_add(-huge), PlainDateTime::MIN);
        assert_eq!(PlainDateTime::MIN.saturating_sub(-huge), PlainDateTime::MAX);
    }

    #[test]
    fn operator_traits_add_and_subtract_durations() {
        let value = datetime(2024, Month::June, 15, 23, 0, 0);
        let one_hour = SignedDuration::hours(1);

        assert_eq!((value + one_hour).to_string(), "2024-06-16 0:00:00.0");
        assert_eq!((value - one_hour).to_string(), "2024-06-15 22:00:00.0");

        let mut mutated = value;
        mutated += one_hour;
        assert_eq!(mutated.to_string(), "2024-06-16 0:00:00.0");
        mutated -= one_hour;
        assert_eq!(mutated, value);

        // Standard durations carry into the date as well.
        let two_hours = StdDuration::from_secs(2 * 3_600);
        assert_eq!((value + two_hours).to_string(), "2024-06-16 1:00:00.0");
        assert_eq!((value - two_hours).to_string(), "2024-06-15 21:00:00.0");
        let mut mutated = value;
        mutated += two_hours;
        assert_eq!(mutated, value + two_hours);
        mutated -= two_hours;
        assert_eq!(mutated, value);

        // Subtraction of two datetimes yields the elapsed duration.
        let a = datetime(2024, Month::June, 15, 12, 0, 0);
        let b = datetime(2024, Month::June, 16, 6, 30, 0);
        assert_eq!(b - a, SignedDuration::hours(18) + SignedDuration::minutes(30));
        assert_eq!(a - b, -(SignedDuration::hours(18) + SignedDuration::minutes(30)));
        assert_eq!(a - a, SignedDuration::hours(0));
        // Crossing midnight in the difference.
        let a = datetime(2024, Month::June, 15, 23, 0, 0);
        let b = datetime(2024, Month::June, 16, 1, 0, 0);
        assert_eq!(b - a, SignedDuration::hours(2));

        // Saturation on overflow.
        assert_eq!(PlainDateTime::MAX + one_hour, PlainDateTime::MAX);
        assert_eq!(PlainDateTime::MIN - one_hour, PlainDateTime::MIN);
    }

    #[test]
    fn replace_and_truncate_preserve_the_other_component() {
        let value = datetime(2024, Month::June, 15, 13, 45, 59)
            .replace_nanosecond(987_654_321)
            .expect("valid");

        // Replacing the date preserves the time and vice versa.
        let other_date = Date::from_calendar_date(2000, Month::January, 2).expect("valid");
        assert_eq!(value.replace_date(other_date).date(), other_date);
        assert_eq!(value.replace_date(other_date).time(), value.time());
        let other_time = Time::from_hms(1, 2, 3).expect("valid");
        assert_eq!(value.replace_time(other_time).time(), other_time);
        assert_eq!(value.replace_time(other_time).date(), value.date());

        // Date replacements delegate to `Date` and keep the time.
        assert_eq!(
            value
                .replace_year(2025)
                .expect("valid")
                .to_string(),
            "2025-06-15 13:45:59.987654321",
        );
        assert_eq!(
            value
                .replace_month(Month::July)
                .expect("valid")
                .to_string(),
            "2024-07-15 13:45:59.987654321",
        );
        assert_eq!(value.replace_day(30).expect("valid").day(), 30,);
        assert_eq!(
            value
                .replace_ordinal(100)
                .expect("valid")
                .ordinal(),
            100,
        );
        assert!(value.replace_day(31).is_err());
        assert!(value.replace_year(10_000).is_err());

        // Time replacements keep the date.
        assert_eq!(value.replace_hour(20).expect("valid").hour(), 20);
        assert_eq!(value.replace_minute(1).expect("valid").minute(), 1);
        assert_eq!(value.replace_second(2).expect("valid").second(), 2);
        assert!(value.replace_hour(24).is_err());
        assert!(value.replace_minute(60).is_err());
        assert!(value.replace_second(60).is_err());
        assert!(value.replace_millisecond(1_000).is_err());
        assert!(value.replace_microsecond(1_000_000).is_err());
        assert!(value.replace_nanosecond(1_000_000_000).is_err());

        // Truncations only clear time components; `truncate_to_day` clears the whole time.
        assert_eq!(value.truncate_to_day().to_string(), "2024-06-15 0:00:00.0");
        assert_eq!(value.truncate_to_hour().to_string(), "2024-06-15 13:00:00.0");
        assert_eq!(value.truncate_to_minute().to_string(), "2024-06-15 13:45:00.0");
        assert_eq!(value.truncate_to_second().to_string(), "2024-06-15 13:45:59.0");
        assert_eq!(
            value.truncate_to_millisecond().to_string(),
            "2024-06-15 13:45:59.987",
        );
        assert_eq!(
            value.truncate_to_microsecond().to_string(),
            "2024-06-15 13:45:59.987654",
        );
        assert_eq!(value.truncate_to_day().date(), value.date());
        assert_eq!(PlainDateTime::MIN.truncate_to_day(), PlainDateTime::MIN);
    }

    #[test]
    fn assume_conversions_preserve_wall_clock_values() {
        let value = datetime(2024, Month::June, 15, 13, 45, 59);

        let utc = value.as_utc();
        assert_eq!(utc.date(), value.date());
        assert_eq!(utc.time(), value.time());

        let offset_datetime = value.assume_utc();
        assert_eq!(offset_datetime.date(), value.date());
        assert_eq!(offset_datetime.time(), value.time());
        assert_eq!(offset_datetime.offset(), UtcOffset::UTC);

        let offset = UtcOffset::from_whole_seconds(3_600).expect("valid offset");
        let offset_datetime = value.assume_offset(offset);
        assert_eq!(offset_datetime.date(), value.date());
        assert_eq!(offset_datetime.time(), value.time());
        assert_eq!(offset_datetime.offset(), offset);
    }

    #[test]
    fn datetimes_are_ordered_and_hashable() {
        use core::hash::{Hash, Hasher};

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let early = datetime(1999, Month::December, 31, 23, 59, 59);
        let epoch = datetime(1970, Month::January, 1, 0, 0, 0);
        let late = datetime(2100, Month::January, 1, 0, 0, 0);

        assert!(epoch < early);
        assert!(early < late);
        assert!(PlainDateTime::MIN < epoch);
        assert!(late < PlainDateTime::MAX);

        // Ordering differentiates on the time within the same date.
        let a = datetime(2024, Month::June, 15, 1, 0, 0);
        let b = datetime(2024, Month::June, 15, 2, 0, 0);
        assert!(a < b);
        assert_eq!(a, a);
        assert_ne!(a, b);

        let copy = PlainDateTime::new(a.date(), a.time());
        assert_eq!(hash_of(&a), hash_of(&copy));
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Less);
        assert_eq!(b.cmp(&a), core::cmp::Ordering::Greater);
        assert_eq!(a.cmp(&a), core::cmp::Ordering::Equal);
    }
}
