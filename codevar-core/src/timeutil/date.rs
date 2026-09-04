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

//! The [`Date`] struct and its associated `impl`s.

use alloc::string::String;
use core::fmt;
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use count_digits::CountDigits;
use std::hint;

use deranged::{ri32, ru8, ru32};
use num_conv::prelude::*;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

use crate::timeutil::date_error::ComponentRange;
use crate::timeutil::date_internal_macro::{
    const_try, const_try_opt, div_floor, ensure_ranged,
};
use crate::timeutil::date_month::Month;
use crate::timeutil::date_num_fmt::{
    four_to_six_digits, str_from_raw_parts, two_digits_zero_padded,
};
use crate::timeutil::date_plain::PlainDateTime;
use crate::timeutil::date_signed_duration::SignedDuration;
use crate::timeutil::date_time::Time;
use crate::timeutil::date_unit::{Day, Second};
use crate::timeutil::date_util::{
    days_in_month_leap, days_in_year, is_leap_year, weeks_in_year,
};
use crate::timeutil::date_weekday::Weekday;

type Year = ri32<MIN_YEAR, MAX_YEAR>;

/// The minimum valid year.
pub(crate) const MIN_YEAR: i32 = if cfg!(feature = "large-dates") {
    -999_999
} else {
    -9999
};
/// The maximum valid year.
pub(crate) const MAX_YEAR: i32 = if cfg!(feature = "large-dates") {
    999_999
} else {
    9999
};

/// Date in the proleptic Gregorian calendar.
///
/// By default, years between ±9999 inclusive are representable. This can be expanded to ±999,999
/// inclusive by enabling the `large-dates` crate feature. Doing so has performance implications
/// and introduces some ambiguities when parsing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Date {
    /// Bitpacked field containing the year, ordinal, and whether the year is a leap year.
    /// |     x      | xxxxxxxxxxxxxxxxxxxxx |       x       | xxxxxxxxx |
    /// |   1 bit    |        21 bits        |     1 bit     |  9 bits   |
    /// | unassigned |         year          | is leap year? |  ordinal  |
    /// The year is 15 bits when `large-dates` is not enabled.
    value: NonZero<i32>,
}

impl Date {
    /// Provide a representation of `Date` as a `i32`. This value can be used for equality, hashing,
    /// and ordering.
    ///
    /// **Note**: This value is explicitly signed, so do not cast this to or treat this as an
    /// unsigned integer. Doing so will lead to incorrect results for values with differing
    /// signs.
    #[inline]
    pub(crate) const fn as_i32(self) -> i32 {
        self.value.get()
    }

    /// The Unix epoch: 1970-01-01
    // Safety: `ordinal` is not zero.
    pub(crate) const UNIX_EPOCH: Self =
        unsafe { Self::from_ordinal_date_unchecked(1970, 1) };

    /// The minimum valid `Date`.
    ///
    /// The value of this may vary depending on the feature flags enabled.
    // Safety: `ordinal` is not zero.
    pub const MIN: Self = unsafe { Self::from_ordinal_date_unchecked(MIN_YEAR, 1) };

    /// The maximum valid `Date`.
    ///
    /// The value of this may vary depending on the feature flags enabled.
    // Safety: `ordinal` is not zero.
    pub const MAX: Self =
        unsafe { Self::from_ordinal_date_unchecked(MAX_YEAR, days_in_year(MAX_YEAR)) };

    /// Construct a `Date` from its internal representation, the validity of which must be
    /// guaranteed by the caller.
    ///
    /// # Safety
    ///
    /// - `ordinal` must be non-zero and at most the number of days in `year`
    /// - `is_leap_year` must be `true` if and only if `year` is a leap year
    #[inline]
    pub(crate) const unsafe fn from_parts(
        year: i32,
        leap_year: bool,
        ordinal: u16,
    ) -> Self {
        debug_assert!(year >= MIN_YEAR);
        debug_assert!(year <= MAX_YEAR);
        debug_assert!(ordinal != 0);
        debug_assert!(ordinal <= days_in_year(year));
        debug_assert!(is_leap_year(year) == leap_year);

        Self {
            // Safety: `ordinal` is not zero.
            value: unsafe {
                NonZero::new_unchecked(
                    (year << 10) | ((leap_year as i32) << 9) | ordinal as i32,
                )
            },
        }
    }

    /// Construct a `Date` from the year and ordinal values, the validity of which must be
    /// guaranteed by the caller.
    ///
    /// # Safety
    ///
    /// - `year` must be in the range `MIN_YEAR..=MAX_YEAR`.
    /// - `ordinal` must be non-zero and at most the number of days in `year`.
    #[doc(hidden)]
    #[inline]
    pub const unsafe fn from_ordinal_date_unchecked(year: i32, ordinal: u16) -> Self {
        // Safety: The caller must guarantee that `ordinal` is not zero and that the year is in
        // range.
        unsafe { Self::from_parts(year, is_leap_year(year), ordinal) }
    }

    /// Attempt to create a `Date` from the year, month, and day.
    #[inline]
    pub const fn from_calendar_date(
        year: i32,
        month: Month,
        day: u8,
    ) -> Result<Self, ComponentRange> {
        /// Cumulative days through the beginning of a month in both common and leap years.
        const DAYS_CUMULATIVE_COMMON_LEAP: [[u16; 12]; 2] = [
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
            [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        ];

        ensure_ranged!(Year: year);

        let is_leap_year = is_leap_year(year);
        match day {
            1..=28 => {}
            29..=31 if day <= days_in_month_leap(month as u8, is_leap_year) => {}
            _ => {
                return Err(ComponentRange::conditional("day"));
            }
        }

        // Safety: `ordinal` is not zero and `is_leap_year` is correct.
        Ok(unsafe {
            Self::from_parts(
                year,
                is_leap_year,
                DAYS_CUMULATIVE_COMMON_LEAP[is_leap_year as usize][month as usize - 1]
                    + day as u16,
            )
        })
    }

    /// Attempt to create a `Date` from the year and ordinal day number.
    #[inline]
    pub const fn from_ordinal_date(
        year: i32,
        ordinal: u16,
    ) -> Result<Self, ComponentRange> {
        ensure_ranged!(Year: year);

        let is_leap_year = is_leap_year(year);
        match ordinal {
            1..=365 => {}
            366 if is_leap_year => {}
            _ => {
                return Err(ComponentRange::conditional("ordinal"));
            }
        }

        // Safety: `ordinal` is not zero.
        Ok(unsafe { Self::from_parts(year, is_leap_year, ordinal) })
    }

    /// Attempt to create a `Date` from the ISO year, week, and weekday.
    pub const fn from_iso_week_date(
        year: i32,
        week: u8,
        weekday: Weekday,
    ) -> Result<Self, ComponentRange> {
        ensure_ranged!(Year: year);
        match week {
            1..=52 => {}
            53 if week <= weeks_in_year(year) => {}
            _ => {
                return Err(ComponentRange::conditional("week"));
            }
        }

        let adj_year = year - 1;
        let raw = 365 * adj_year + div_floor!(adj_year, 4) - div_floor!(adj_year, 100)
            + div_floor!(adj_year, 400);
        let jan_4 = match (raw % 7) as i8 {
            -6 | 1 => 8,
            -5 | 2 => 9,
            -4 | 3 => 10,
            -3 | 4 => 4,
            -2 | 5 => 5,
            -1 | 6 => 6,
            _ => 7,
        };
        let ordinal = week as i16 * 7 + weekday.number_from_monday() as i16 - jan_4;

        if ordinal <= 0 {
            // Safety: `ordinal` is not zero.
            return Ok(unsafe {
                Self::from_ordinal_date_unchecked(
                    year - 1,
                    ordinal.cast_unsigned().wrapping_add(days_in_year(year - 1)),
                )
            });
        }
        let is_leap_year = is_leap_year(year);
        let days_in_year = if is_leap_year { 366 } else { 365 };
        let ordinal = ordinal.cast_unsigned();
        Ok(if ordinal > days_in_year {
            if year == MAX_YEAR {
                return Err(ComponentRange::conditional("weekday"));
            }
            // Safety: the year is in range and `ordinal` is not zero.
            unsafe { Self::from_ordinal_date_unchecked(year + 1, ordinal - days_in_year) }
        } else {
            // Safety: `ordinal` is not zero and `is_leap_year` is correct.
            unsafe { Self::from_parts(year, is_leap_year, ordinal) }
        })
    }

    /// Create a `Date` from the Julian day.
    #[doc(alias = "from_julian_date")]
    #[inline]
    pub const fn from_julian_day(julian_day: i32) -> Result<Self, ComponentRange> {
        type JulianDay =
            ri32<{ Date::MIN.to_julian_day() }, { Date::MAX.to_julian_day() }>;
        ensure_ranged!(JulianDay: julian_day);
        // Safety: The Julian day number is in range.
        Ok(unsafe { Self::from_julian_day_unchecked(julian_day) })
    }

    /// Create a `Date` from the Julian day.
    ///
    /// # Safety
    ///
    /// The provided Julian day number must be between `Date::MIN.to_julian_day()` and
    /// `Date::MAX.to_julian_day()` inclusive.
    #[inline]
    pub(crate) const unsafe fn from_julian_day_unchecked(julian_day: i32) -> Self {
        debug_assert!(julian_day >= Self::MIN.to_julian_day());
        debug_assert!(julian_day <= Self::MAX.to_julian_day());

        const ERAS: u32 = 5_949;
        // Rata Die shift:
        const D_SHIFT: u32 = 146097 * ERAS - 1_721_060;
        // Year shift:
        const Y_SHIFT: u32 = 400 * ERAS;

        const CEN_MUL: u32 = ((4u64 << 47) / 146_097) as u32;
        const JUL_MUL: u32 = ((4u64 << 40) / 1_461 + 1) as u32;
        const CEN_CUT: u32 = ((365u64 << 32) / 36_525) as u32;

        let day = julian_day.cast_unsigned().wrapping_add(D_SHIFT);
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

        // Safety: `ordinal` is not zero and `is_leap_year` is correct, so long as the Julian day
        // number is in range, which is guaranteed by the caller.
        unsafe { Self::from_parts(year, leap, ordinal as u16) }
    }

    /// Whether `is_leap_year(self.year())` is `true`.
    ///
    /// This method is optimized to take advantage of the fact that the value is pre-computed upon
    /// construction and stored in the bitpacked struct.
    #[inline]
    pub(crate) const fn is_in_leap_year(self) -> bool {
        (self.value.get() >> 9) & 1 == 1
    }

    /// Get the year of the date.
    #[inline]
    pub const fn year(self) -> i32 {
        self.value.get() >> 10
    }

    /// Get the month.
    #[inline]
    pub const fn month(self) -> Month {
        let ordinal = self.ordinal() as u32;
        let jan_feb_len = 59 + self.is_in_leap_year() as u32;

        let (month_adj, ordinal_adj) = if ordinal <= jan_feb_len {
            (0, 0)
        } else {
            (2, jan_feb_len)
        };

        let ordinal = ordinal - ordinal_adj;
        let month = ((ordinal * 268 + 8031) >> 13) + month_adj;

        // Safety: `month` is guaranteed to be between 1 and 12 inclusive.
        unsafe {
            match Month::from_number(NonZero::new_unchecked(month as u8)) {
                Ok(month) => month,
                Err(_) => core::hint::unreachable_unchecked(),
            }
        }
    }

    /// Get the day of the month.
    ///
    /// The returned value will always be in the range `1..=31`.
    #[inline]
    pub const fn day(self) -> u8 {
        let ordinal = self.ordinal() as u32;
        let jan_feb_len = 59 + self.is_in_leap_year() as u32;

        let ordinal_adj = if ordinal <= jan_feb_len {
            0
        } else {
            jan_feb_len
        };

        let ordinal = ordinal - ordinal_adj;
        let month = (ordinal * 268 + 8031) >> 13;
        let days_in_preceding_months = (month * 3917 - 3866) >> 7;
        (ordinal - days_in_preceding_months) as u8
    }

    /// Get the day of the year.
    ///
    /// The returned value will always be in the range `1..=366` (`1..=365` for common years).
    #[inline]
    pub const fn ordinal(self) -> u16 {
        (self.value.get() & 0x1FF) as u16
    }

    /// Get the ISO 8601 year and week number.
    #[inline]
    pub(crate) const fn iso_year_week(self) -> (i32, u8) {
        let (year, ordinal) = self.to_ordinal_date();

        match ((ordinal + 10 - self.weekday().number_from_monday() as u16) / 7) as u8 {
            0 => (year - 1, weeks_in_year(year - 1)),
            53 if weeks_in_year(year) == 52 => (year + 1, 1),
            week => (year, week),
        }
    }

    /// Get the ISO week number.
    ///
    /// The returned value will always be in the range `1..=53`.
    #[inline]
    pub const fn iso_week(self) -> u8 {
        self.iso_year_week().1
    }

    /// Get the week number where week 1 begins on the first Sunday.
    #[inline]
    pub const fn sunday_based_week(self) -> u8 {
        ((self.ordinal().cast_signed() - self.weekday().number_days_from_sunday() as i16
            + 6)
            / 7) as u8
    }

    /// Get the week number where week 1 begins on the first Monday.
    #[inline]
    pub const fn monday_based_week(self) -> u8 {
        ((self.ordinal().cast_signed() - self.weekday().number_days_from_monday() as i16
            + 6)
            / 7) as u8
    }

    /// Get the year, month, and day.
    #[inline]
    pub const fn to_calendar_date(self) -> (i32, Month, u8) {
        let (year, ordinal) = self.to_ordinal_date();
        let ordinal = ordinal as u32;
        let jan_feb_len = 59 + self.is_in_leap_year() as u32;

        let (month_adj, ordinal_adj) = if ordinal <= jan_feb_len {
            (0, 0)
        } else {
            (2, jan_feb_len)
        };

        let ordinal = ordinal - ordinal_adj;
        let month = (ordinal * 268 + 8031) >> 13;
        let days_in_preceding_months = (month * 3917 - 3866) >> 7;
        let day = ordinal - days_in_preceding_months;
        let month = month + month_adj;

        (
            year,
            // Safety: `month` is guaranteed to be between 1 and 12 inclusive.
            unsafe {
                match Month::from_number(NonZero::new_unchecked(month as u8)) {
                    Ok(month) => month,
                    Err(_) => core::hint::unreachable_unchecked(),
                }
            },
            day as u8,
        )
    }

    /// Get the year and ordinal day number.
    #[inline]
    pub const fn to_ordinal_date(self) -> (i32, u16) {
        (self.year(), self.ordinal())
    }

    /// Get the ISO 8601 year, week number, and weekday.
    #[inline]
    pub const fn to_iso_week_date(self) -> (i32, u8, Weekday) {
        let (year, ordinal) = self.to_ordinal_date();
        let weekday = self.weekday();

        match ((ordinal + 10 - weekday.number_from_monday() as u16) / 7) as u8 {
            0 => (year - 1, weeks_in_year(year - 1), weekday),
            53 if weeks_in_year(year) == 52 => (year + 1, 1, weekday),
            week => (year, week, weekday),
        }
    }

    /// Get the weekday.
    #[inline]
    pub const fn weekday(self) -> Weekday {
        match self.to_julian_day() % 7 {
            -6 | 1 => Weekday::Tuesday,
            -5 | 2 => Weekday::Wednesday,
            -4 | 3 => Weekday::Thursday,
            -3 | 4 => Weekday::Friday,
            -2 | 5 => Weekday::Saturday,
            -1 | 6 => Weekday::Sunday,
            val => {
                debug_assert!(val == 0);
                Weekday::Monday
            }
        }
    }

    /// Get the next calendar date.
    #[inline]
    pub const fn next_day(self) -> Option<Self> {
        let is_last_day_of_year = matches!(self.value.get() & 0x3FF, 365 | 878);
        if is_last_day_of_year {
            if self.value.get() == Self::MAX.value.get() {
                None
            } else {
                // Safety: `ordinal` is not zero.
                unsafe { Some(Self::from_ordinal_date_unchecked(self.year() + 1, 1)) }
            }
        } else {
            // Safety: `self` is not the last day of the year.
            Some(unsafe { self.add_days_unchecked(1) })
        }
    }

    /// Get the previous calendar date.
    #[inline]
    pub const fn previous_day(self) -> Option<Self> {
        if self.ordinal() != 1 {
            // Safety: `self` is not the first day of the year.
            Some(unsafe { self.add_days_unchecked(-1) })
        } else if self.value.get() == Self::MIN.value.get() {
            None
        } else {
            let year = self.year() - 1;
            let is_leap_year = is_leap_year(year);
            let ordinal = if is_leap_year { 366 } else { 365 };
            // Safety: `ordinal` is not zero, `is_leap_year` is correct.
            Some(unsafe { Self::from_parts(year, is_leap_year, ordinal) })
        }
    }

    /// Calculates the first occurrence of a weekday that is strictly later than a given `Date`.
    ///
    /// # Panics
    /// Panics if an overflow occurred.
    #[inline]
    pub const fn next_occurrence(self, weekday: Weekday) -> Self {
        match self.checked_next_occurrence(weekday) {
            Some(date) => date,
            None => self.saturating_add(SignedDuration::days(7)),
        }
    }

    /// Calculates the first occurrence of a weekday that is strictly earlier than a given `Date`.
    ///
    /// # Panics
    /// Panics if an overflow occurred.
    #[inline]
    pub const fn prev_occurrence(self, weekday: Weekday) -> Self {
        match self.checked_prev_occurrence(weekday) {
            Some(date) => date,
            None => self.saturating_sub(SignedDuration::days(7)),
        }
    }

    /// Calculates the `n`th occurrence of a weekday that is strictly later than a given `Date`.
    ///
    /// # Panics
    /// Panics if an overflow occurred or if `n == 0`.
    #[inline]
    pub const fn nth_next_occurrence(self, weekday: Weekday, n: u8) -> Self {
        if n == 0 {
            return self;
        }
        match self.checked_nth_next_occurrence(weekday, n) {
            Some(date) => date,
            None => self.saturating_add(SignedDuration::weeks(n as i64)),
        }
    }

    /// Calculates the `n`th occurrence of a weekday that is strictly earlier than a given `Date`.
    ///
    /// # Panics
    /// Panics if an overflow occurred or if `n == 0`.
    #[inline]
    pub const fn nth_prev_occurrence(self, weekday: Weekday, n: u8) -> Self {
        if n == 0 {
            return self;
        }
        match self.checked_nth_prev_occurrence(weekday, n) {
            Some(date) => date,
            None => self.saturating_sub(SignedDuration::weeks(n as i64)),
        }
    }

    /// Get the Julian day for the date.
    #[inline]
    pub const fn to_julian_day(self) -> i32 {
        let (year, ordinal) = self.to_ordinal_date();
        // The algorithm requires a non-negative year. Add the lowest value to make it so. This is
        // adjusted for at the end with the final subtraction.
        let adj_year = year + 999_999;
        let century = adj_year / 100;

        let days_before_year =
            (1461 * adj_year as i64 / 4) as i32 - century + century / 4;
        days_before_year + ordinal as i32 - 363_521_075
    }

    /// Add a number of days to the date without checking for overflow.
    ///
    /// # Safety
    ///
    /// `self.ordinal() + days` must be in the range `1..=366` for leap years and `1..=365` for
    /// common years.
    #[inline]
    pub(crate) const unsafe fn add_days_unchecked(mut self, days: i32) -> Self {
        // Safety: asserted by caller
        self.value = unsafe { NonZero::new_unchecked(self.value.get() + days) };
        self
    }

    /// Computes `self + duration`, returning `None` if an overflow occurred.
    ///
    /// # Note
    ///
    /// This function only takes whole days into account.
    #[inline]
    pub const fn checked_add(self, duration: SignedDuration) -> Option<Self> {
        let whole_days = duration.whole_days();
        if whole_days < i32::MIN as i64 || whole_days > i32::MAX as i64 {
            return None;
        }

        let year = self.year();
        let is_leap_year = self.is_in_leap_year();
        let ordinal = self.ordinal() as i32;

        let days_in_year = if is_leap_year { 366 } else { 365 };
        let whole_days = whole_days as i32;

        // Fast path for when the result is in the same year.
        if let Some(new_ordinal) = ordinal.checked_add(whole_days)
            && new_ordinal >= 1
            && new_ordinal <= days_in_year
        {
            // Safety: `new_ordinal` is in range and `is_leap_year` is correct
            return Some(unsafe {
                Self::from_parts(year, is_leap_year, new_ordinal as u16)
            });
        }

        let julian_day = const_try_opt!(self.to_julian_day().checked_add(whole_days));
        if let Ok(date) = Self::from_julian_day(julian_day) {
            Some(date)
        } else {
            None
        }
    }

    /// Computes `self + duration`, returning `None` if an overflow occurred.
    ///
    /// # Note
    ///
    /// This function only takes whole days into account.
    #[inline]
    pub const fn checked_add_std(self, duration: StdDuration) -> Option<Self> {
        let whole_days = duration.as_secs() / Second::per_t::<u64>(Day);
        if whole_days > i32::MAX as u64 {
            return None;
        }

        let year = self.year();
        let is_leap_year = self.is_in_leap_year();
        let ordinal = self.ordinal() as i32;

        let days_in_year = if is_leap_year { 366 } else { 365 };
        let whole_days = whole_days as i32;

        // Fast path for when the result is in the same year.
        if let Some(new_ordinal) = ordinal.checked_add(whole_days)
            && new_ordinal >= 1
            && new_ordinal <= days_in_year
        {
            // Safety: `new_ordinal` is in range and `is_leap_year` is correct
            return Some(unsafe {
                Self::from_parts(year, is_leap_year, new_ordinal as u16)
            });
        }

        let julian_day = const_try_opt!(self.to_julian_day().checked_add(whole_days));
        if let Ok(date) = Self::from_julian_day(julian_day) {
            Some(date)
        } else {
            None
        }
    }

    /// Computes `self - duration`, returning `None` if an overflow occurred.
    ///
    /// # Note
    ///
    /// This function only takes whole days into account.
    #[inline]
    pub const fn checked_sub(self, duration: SignedDuration) -> Option<Self> {
        let whole_days = duration.whole_days();
        if whole_days < i32::MIN as i64 || whole_days > i32::MAX as i64 {
            return None;
        }

        let year = self.year();
        let is_leap_year = self.is_in_leap_year();
        let ordinal = self.ordinal() as i32;

        let days_in_year = if is_leap_year { 366 } else { 365 };
        let whole_days = whole_days as i32;

        // Fast path for when the result is in the same year.
        if let Some(new_ordinal) = ordinal.checked_sub(whole_days)
            && new_ordinal >= 1
            && new_ordinal <= days_in_year
        {
            // Safety: `new_ordinal` is in range and `is_leap_year` is correct
            return Some(unsafe {
                Self::from_parts(year, is_leap_year, new_ordinal as u16)
            });
        }

        let julian_day = const_try_opt!(self.to_julian_day().checked_sub(whole_days));
        if let Ok(date) = Self::from_julian_day(julian_day) {
            Some(date)
        } else {
            None
        }
    }

    /// Computes `self - duration`, returning `None` if an overflow occurred.
    ///
    /// # Note
    ///
    /// This function only takes whole days into account.
    #[inline]
    pub const fn checked_sub_std(self, duration: StdDuration) -> Option<Self> {
        let whole_days = duration.as_secs() / Second::per_t::<u64>(Day);
        if whole_days > i32::MAX as u64 {
            return None;
        }

        let year = self.year();
        let is_leap_year = self.is_in_leap_year();
        let ordinal = self.ordinal() as i32;

        let days_in_year = if is_leap_year { 366 } else { 365 };
        let whole_days = whole_days as i32;

        // Fast path for when the result is in the same year.
        if let Some(new_ordinal) = ordinal.checked_sub(whole_days)
            && new_ordinal >= 1
            && new_ordinal <= days_in_year
        {
            // Safety: `new_ordinal` is in range and `is_leap_year` is correct
            return Some(unsafe {
                Self::from_parts(year, is_leap_year, new_ordinal as u16)
            });
        }

        let julian_day = const_try_opt!(self.to_julian_day().checked_sub(whole_days));
        if let Ok(date) = Self::from_julian_day(julian_day) {
            Some(date)
        } else {
            None
        }
    }

    /// Calculates the first occurrence of a weekday that is strictly later than a given `Date`.
    /// Returns `None` if an overflow occurred.
    #[inline]
    pub(crate) const fn checked_next_occurrence(self, weekday: Weekday) -> Option<Self> {
        let day_diff = match weekday as i8 - self.weekday() as i8 {
            1 | -6 => 1,
            2 | -5 => 2,
            3 | -4 => 3,
            4 | -3 => 4,
            5 | -2 => 5,
            6 | -1 => 6,
            val => {
                debug_assert!(val == 0);
                7
            }
        };

        self.checked_add(SignedDuration::days(day_diff))
    }

    /// Calculates the first occurrence of a weekday that is strictly earlier than a given `Date`.
    /// Returns `None` if an overflow occurred.
    #[inline]
    pub(crate) const fn checked_prev_occurrence(self, weekday: Weekday) -> Option<Self> {
        let day_diff = match weekday as i8 - self.weekday() as i8 {
            1 | -6 => 6,
            2 | -5 => 5,
            3 | -4 => 4,
            4 | -3 => 3,
            5 | -2 => 2,
            6 | -1 => 1,
            val => {
                debug_assert!(val == 0);
                7
            }
        };

        self.checked_sub(SignedDuration::days(day_diff))
    }

    /// Calculates the `n`th occurrence of a weekday that is strictly later than a given `Date`.
    /// Returns `None` if an overflow occurred or if `n == 0`.
    #[inline]
    pub(crate) const fn checked_nth_next_occurrence(
        self,
        weekday: Weekday,
        n: u8,
    ) -> Option<Self> {
        if n == 0 {
            return None;
        }
        const_try_opt!(self.checked_next_occurrence(weekday))
            .checked_add(SignedDuration::weeks(n as i64 - 1))
    }

    /// Calculates the `n`th occurrence of a weekday that is strictly earlier than a given `Date`.
    /// Returns `None` if an overflow occurred or if `n == 0`.
    #[inline]
    pub(crate) const fn checked_nth_prev_occurrence(
        self,
        weekday: Weekday,
        n: u8,
    ) -> Option<Self> {
        if n == 0 {
            return None;
        }
        const_try_opt!(self.checked_prev_occurrence(weekday))
            .checked_sub(SignedDuration::weeks(n as i64 - 1))
    }

    /// Computes `self + duration`, saturating value on overflow.
    ///
    /// # Note
    ///
    /// This function only takes whole days into account.
    #[inline]
    pub const fn saturating_add(self, duration: SignedDuration) -> Self {
        if let Some(datetime) = self.checked_add(duration) {
            datetime
        } else if duration.is_negative() {
            Self::MIN
        } else {
            debug_assert!(duration.is_positive());
            Self::MAX
        }
    }

    /// Computes `self - duration`, saturating value on overflow.
    ///
    /// # Note
    ///
    /// This function only takes whole days into account.
    #[inline]
    pub const fn saturating_sub(self, duration: SignedDuration) -> Self {
        if let Some(datetime) = self.checked_sub(duration) {
            datetime
        } else if duration.is_negative() {
            Self::MAX
        } else {
            debug_assert!(duration.is_positive());
            Self::MIN
        }
    }

    /// Replace the year. The month and day will be unchanged.
    #[inline]
    pub const fn replace_year(self, year: i32) -> Result<Self, ComponentRange> {
        ensure_ranged!(Year: year);

        let new_is_leap_year = is_leap_year(year);
        let ordinal = self.ordinal();

        // Dates in January and February are unaffected by leap years.
        if ordinal <= 59 {
            // Safety: `ordinal` is not zero and `is_leap_year` is correct.
            return Ok(unsafe { Self::from_parts(year, new_is_leap_year, ordinal) });
        }

        match (self.is_in_leap_year(), new_is_leap_year) {
            (false, false) | (true, true) => {
                Ok(Self {
                    // Safety: Whether the year is leap or common, the ordinal are unchanged, with
                    // only the year being replaced.
                    value: unsafe {
                        NonZero::new_unchecked((year << 10) | (self.value.get() & 0x3FF))
                    },
                })
            }
            // February 29 does not exist in common years.
            (true, false) if ordinal == 60 => Err(ComponentRange::conditional("day")),
            // We're going from a common year to a leap year. Shift dates in March and later by
            // one day.
            // Safety: `ordinal` is not zero and `is_leap_year` is correct.
            (false, true) => Ok(unsafe { Self::from_parts(year, true, ordinal + 1) }),
            // We're going from a leap year to a common year. Shift dates in January and
            // February by one day.
            // Safety: `ordinal` is not zero and `is_leap_year` is correct.
            (true, false) => Ok(unsafe { Self::from_parts(year, false, ordinal - 1) }),
        }
    }

    /// Replace the month of the year.
    #[inline]
    pub const fn replace_month(self, month: Month) -> Result<Self, ComponentRange> {
        /// Cumulative days through the beginning of a month in both common and leap years.
        const DAYS_CUMULATIVE_COMMON_LEAP: [[u16; 12]; 2] = [
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
            [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        ];

        let (year, ordinal) = self.to_ordinal_date();
        let mut ordinal = ordinal as u32;
        let is_leap_year = self.is_in_leap_year();
        let jan_feb_len = 59 + is_leap_year as u32;

        if ordinal > jan_feb_len {
            ordinal -= jan_feb_len;
        }
        let current_month = (ordinal * 268 + 8031) >> 13;
        let days_in_preceding_months = (current_month * 3917 - 3866) >> 7;
        let day = (ordinal - days_in_preceding_months) as u8;

        match day {
            1..=28 => {}
            29..=31 if day <= days_in_month_leap(month as u8, is_leap_year) => {}
            _ => {
                return Err(ComponentRange::conditional("day"));
            }
        }

        // Safety: `ordinal` is not zero and `is_leap_year` is correct.
        Ok(unsafe {
            Self::from_parts(
                year,
                is_leap_year,
                DAYS_CUMULATIVE_COMMON_LEAP[is_leap_year as usize][month as usize - 1]
                    + day as u16,
            )
        })
    }

    /// Replace the day of the month.
    #[inline]
    pub const fn replace_day(self, day: u8) -> Result<Self, ComponentRange> {
        let is_leap_year = self.is_in_leap_year();
        match day {
            1..=28 => {}
            29..=31 if day <= days_in_month_leap(self.month() as u8, is_leap_year) => {}
            _ => {
                return Err(ComponentRange::conditional("day"));
            }
        }

        // Safety: `ordinal` is not zero and `is_leap_year` is correct.
        Ok(unsafe {
            Self::from_parts(
                self.year(),
                is_leap_year,
                (self.ordinal().cast_signed() - self.day() as i16 + day as i16)
                    .cast_unsigned(),
            )
        })
    }

    /// Replace the day of the year.
    #[inline]
    pub const fn replace_ordinal(self, ordinal: u16) -> Result<Self, ComponentRange> {
        let is_leap_year = self.is_in_leap_year();
        match ordinal {
            1..=365 => {}
            366 if is_leap_year => {}
            _ => {
                return Err(ComponentRange::conditional("ordinal"));
            }
        }
        // Safety: `ordinal` is in range and `is_leap_year` is correct.
        Ok(unsafe { Self::from_parts(self.year(), is_leap_year, ordinal) })
    }
}

/// Methods to add a [`Time`] component, resulting in a [`PlainDateTime`].
impl Date {
    /// Create a [`PlainDateTime`] using the existing date. The [`Time`] component will be set to
    /// midnight.
    #[inline]
    pub const fn midnight(self) -> PlainDateTime {
        PlainDateTime::new(self, Time::MIDNIGHT)
    }

    /// Create a [`PlainDateTime`] using the existing date and the provided [`Time`].
    #[inline]
    pub const fn with_time(self, time: Time) -> PlainDateTime {
        PlainDateTime::new(self, time)
    }

    /// Attempt to create a [`PlainDateTime`] using the existing date and the provided time.
    #[inline]
    pub const fn with_hms(
        self,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Result<PlainDateTime, ComponentRange> {
        Ok(PlainDateTime::new(
            self,
            const_try!(Time::from_hms(hour, minute, second)),
        ))
    }

    /// Attempt to create a [`PlainDateTime`] using the existing date and the provided time.
    #[inline]
    pub const fn with_hms_milli(
        self,
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u32,
    ) -> Result<PlainDateTime, ComponentRange> {
        Ok(PlainDateTime::new(
            self,
            const_try!(Time::from_hms_milli(hour, minute, second, millisecond)),
        ))
    }

    /// Attempt to create a [`PlainDateTime`] using the existing date and the provided time.
    #[inline]
    pub const fn with_hms_micro(
        self,
        hour: u8,
        minute: u8,
        second: u8,
        microsecond: u32,
    ) -> Result<PlainDateTime, ComponentRange> {
        Ok(PlainDateTime::new(
            self,
            const_try!(Time::from_hms_micro(hour, minute, second, microsecond)),
        ))
    }

    /// Attempt to create a [`PlainDateTime`] using the existing date and the provided time.
    #[inline]
    pub const fn with_hms_nano(
        self,
        hour: u8,
        minute: u8,
        second: u8,
        nanosecond: u32,
    ) -> Result<PlainDateTime, ComponentRange> {
        Ok(PlainDateTime::new(
            self,
            const_try!(Time::from_hms_nano(hour, minute, second, nanosecond)),
        ))
    }
}

// This no longer needs special handling, as the format is fixed and doesn't require anything
// advanced. Trait impls can't be deprecated and the info is still useful for other types
// implementing `SmartDisplay`, so leave it as-is for now.
impl SmartDisplay for Date {
    type Metadata = ();

    #[inline]
    fn metadata(&self, _: FormatterOptions) -> Metadata<'_, Self> {
        let year_sign_width = if self.year() < 0
            || (cfg!(feature = "large-dates") && self.year() >= 10_000)
        {
            1
        } else {
            0
        };
        let year_width = self.year().unsigned_abs().count_digits().clamp(4, 6);
        let formatted_width = year_sign_width + year_width + 6; // include two dashes and two digits each for month and day

        Metadata::new(formatted_width as usize, self, ())
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Date {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// for the `Display` implementation.
    pub(crate) const DISPLAY_BUFFER_SIZE: usize = 13;

    /// Format the `Date` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        let mut idx = 0;
        let (year, month, day) = self.to_calendar_date();

        // Compute the sign of the integer, if any. Doing this in a branchless manner gives a
        // significant performance improvement.
        let neg = year.is_negative() as u8;
        let pos = (cfg!(feature = "large-dates") && year - 10_000 >= 0) as u8;
        let sign = b'+' + 2 * neg; // b'-' if `neg` is true, b'+' otherwise
        // Always write the computed byte, even if it's later overwritten by the first digit of the
        // year.
        buf[idx] = MaybeUninit::new(sign);
        idx += (neg | pos) as usize;

        // Safety: `year.unsigned_abs()` is less than 1,000,000.
        let [first_two, second_two, third_two] =
            four_to_six_digits(unsafe { ru32::new_unchecked(year.unsigned_abs()) });
        // Safety:
        // - both `first_two` and `buf` are valid for reads and writes of up to 2 bytes.
        // - `u8` is 1-aligned, so that is not a concern.
        // - `first_two` points to static memory, while `buf` is a local variable, so they do not
        //   overlap.
        unsafe {
            first_two.as_ptr().copy_to_nonoverlapping(
                buf.as_mut_ptr().add(idx).cast(),
                first_two.len(),
            );
        }
        idx += first_two.len();
        // Safety: See above.
        unsafe {
            second_two
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), 2);
        }
        idx += 2;
        // Safety: See above.
        unsafe {
            third_two
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), 2);
        }
        idx += 2;

        buf[idx] = MaybeUninit::new(b'-');
        idx += 1;

        // Safety: See above for `copy_to_nonoverlapping`. `month` is in the range 1..=12.
        unsafe {
            two_digits_zero_padded(ru8::new_unchecked(u8::from(month)))
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), 2);
        }
        idx += 2;

        buf[idx] = MaybeUninit::new(b'-');
        idx += 1;

        // Safety: See above for `copy_to_nonoverlapping`. `day` is in the range 1..=31.
        unsafe {
            two_digits_zero_padded(ru8::new_unchecked(day))
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), 2);
        }
        idx += 2;

        idx
    }
}

impl fmt::Display for Date {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); 13];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts((&raw const buf).cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for Date {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self, f)
    }
}

impl Add<SignedDuration> for Date {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add(self, duration: SignedDuration) -> Self::Output {
        self.checked_add(duration).unwrap_or_else(|| {
            if duration.is_negative() {
                Self::MIN
            } else {
                Self::MAX
            }
        })
    }
}

impl Add<StdDuration> for Date {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add(self, duration: StdDuration) -> Self::Output {
        self.checked_add_std(duration).unwrap_or_else(|| {
            if duration.as_secs() > 0 || duration.subsec_nanos() > 0 {
                Self::MAX
            } else {
                Self::MIN
            }
        })
    }
}

impl AddAssign<SignedDuration> for Date {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        *self = *self + rhs;
    }
}

impl AddAssign<StdDuration> for Date {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: StdDuration) {
        *self = *self + rhs;
    }
}

impl Sub<SignedDuration> for Date {
    type Output = Self;

    #[inline]
    fn sub(self, duration: SignedDuration) -> Self::Output {
        self.checked_sub(duration).unwrap_or_else(|| {
            if duration.is_negative() {
                Self::MAX
            } else {
                Self::MIN
            }
        })
    }
}

impl Sub<StdDuration> for Date {
    type Output = Self;

    #[inline]
    fn sub(self, duration: StdDuration) -> Self::Output {
        self.checked_sub_std(duration).unwrap_or_else(|| {
            if duration.as_secs() > 0 || duration.subsec_nanos() > 0 {
                Self::MIN
            } else {
                Self::MAX
            }
        })
    }
}

impl SubAssign<SignedDuration> for Date {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        *self = *self - rhs;
    }
}

impl SubAssign<StdDuration> for Date {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: StdDuration) {
        *self = *self - rhs;
    }
}

impl Sub for Date {
    type Output = SignedDuration;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        SignedDuration::days((self.to_julian_day() - other.to_julian_day()).widen())
    }
}
