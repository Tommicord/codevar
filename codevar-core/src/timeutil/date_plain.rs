use alloc::string::String;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;

use crate::timeutil::date::Date;
use crate::timeutil::date_error::ComponentRange;
use crate::timeutil::date_internal_macro::{const_try, const_try_opt};
use crate::timeutil::date_month::Month;
use crate::timeutil::date_num_fmt::str_from_raw_parts;
use crate::timeutil::date_offset_time::OffsetDateTime;
use crate::timeutil::date_signed_duration::SignedDuration;
use crate::timeutil::date_time::Time;
use crate::timeutil::date_utc_offset::UtcOffset;
use crate::timeutil::date_utc_time::UtcDateTime;
use crate::timeutil::date_util::DateAdjustment;
use crate::timeutil::date_weekday::Weekday;
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
    pub const fn replace_millisecond(
        self,
        millisecond: u16,
    ) -> Result<Self, ComponentRange> {
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
    pub const fn replace_microsecond(
        self,
        microsecond: u32,
    ) -> Result<Self, ComponentRange> {
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
    pub const fn replace_nanosecond(
        self,
        nanosecond: u32,
    ) -> Result<Self, ComponentRange> {
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
    pub(crate) const DISPLAY_BUFFER_SIZE: usize =
        Date::DISPLAY_BUFFER_SIZE + Time::DISPLAY_BUFFER_SIZE + 1;

    /// Format the `PlainDateTime` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let date_len = self
            .date
            .fmt_into_buffer(unsafe { buf.first_chunk_mut().unwrap_unchecked() });
        buf[date_len].write(b' ');
        // Safety: The buffer is large enough that the first chunk is in bounds.
        let time_len = self.time.fmt_into_buffer(unsafe {
            buf[date_len + 1..].first_chunk_mut().unwrap_unchecked()
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
                (self.date + duration).next_day().unwrap_or(Date::MAX)
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
                (self.date - duration).previous_day().unwrap_or(Date::MIN)
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
