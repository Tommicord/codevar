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

//! The [`Date`] struct and its associated `impl`s.

use core::fmt;
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use count_digits::CountDigits;

use deranged::{ri32, ru8, ru32};
use num_conv::prelude::*;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

use crate::date_error::ComponentRange;
use crate::date_internal_macro::{const_try, const_try_opt, div_floor, ensure_ranged};
use crate::date_month::Month;
use crate::date_num_fmt::{four_to_six_digits, str_from_raw_parts, two_digits_zero_padded};
use crate::date_plain::PlainDateTime;
use crate::date_signed_duration::SignedDuration;
use crate::date_time::Time;
use crate::date_unit::{Day, Second};
use crate::date_util::{days_in_month_leap, days_in_year, is_leap_year, weeks_in_year};
use crate::date_weekday::Weekday;

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
    pub(crate) const UNIX_EPOCH: Self = unsafe { Self::from_ordinal_date_unchecked(1970, 1) };

    /// The minimum valid `Date`.
    ///
    /// The value of this may vary depending on the feature flags enabled.
    // Safety: `ordinal` is not zero.
    pub const MIN: Self = unsafe { Self::from_ordinal_date_unchecked(MIN_YEAR, 1) };

    /// The maximum valid `Date`.
    ///
    /// The value of this may vary depending on the feature flags enabled.
    // Safety: `ordinal` is not zero.
    pub const MAX: Self = unsafe { Self::from_ordinal_date_unchecked(MAX_YEAR, days_in_year(MAX_YEAR)) };

    /// Construct a `Date` from its internal representation, the validity of which must be
    /// guaranteed by the caller.
    ///
    /// # Safety
    ///
    /// - `ordinal` must be non-zero and at most the number of days in `year`
    /// - `is_leap_year` must be `true` if and only if `year` is a leap year
    #[inline]
    pub(crate) const unsafe fn from_parts(year: i32, leap_year: bool, ordinal: u16) -> Self {
        debug_assert!(year >= MIN_YEAR);
        debug_assert!(year <= MAX_YEAR);
        debug_assert!(ordinal != 0);
        debug_assert!(ordinal <= days_in_year(year));
        debug_assert!(is_leap_year(year) == leap_year);

        Self {
            // Safety: `ordinal` is not zero.
            value: unsafe {
                NonZero::new_unchecked((year << 10) | ((leap_year as i32) << 9) | ordinal as i32)
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
    pub const fn from_calendar_date(year: i32, month: Month, day: u8) -> Result<Self, ComponentRange> {
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
                DAYS_CUMULATIVE_COMMON_LEAP[is_leap_year as usize][month as usize - 1] + day as u16,
            )
        })
    }

    /// Attempt to create a `Date` from the year and ordinal day number.
    #[inline]
    pub const fn from_ordinal_date(year: i32, ordinal: u16) -> Result<Self, ComponentRange> {
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
    pub const fn from_iso_week_date(year: i32, week: u8, weekday: Weekday) -> Result<Self, ComponentRange> {
        ensure_ranged!(Year: year);
        match week {
            1..=52 => {}
            53 if week <= weeks_in_year(year) => {}
            _ => {
                return Err(ComponentRange::conditional("week"));
            }
        }

        let adj_year = year - 1;
        let raw =
            365 * adj_year + div_floor!(adj_year, 4) - div_floor!(adj_year, 100) + div_floor!(adj_year, 400);
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
                    ordinal
                        .cast_unsigned()
                        .wrapping_add(days_in_year(year - 1)),
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
        type JulianDay = ri32<{ Date::MIN.to_julian_day() }, { Date::MAX.to_julian_day() }>;
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

        let ordinal_adj = if ordinal <= jan_feb_len { 0 } else { jan_feb_len };

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
        ((self.ordinal().cast_signed() - self.weekday().number_days_from_sunday() as i16 + 6) / 7) as u8
    }

    /// Get the week number where week 1 begins on the first Monday.
    #[inline]
    pub const fn monday_based_week(self) -> u8 {
        ((self.ordinal().cast_signed() - self.weekday().number_days_from_monday() as i16 + 6) / 7) as u8
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

        let days_before_year = (1461 * adj_year as i64 / 4) as i32 - century + century / 4;
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
            return Some(unsafe { Self::from_parts(year, is_leap_year, new_ordinal as u16) });
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
            return Some(unsafe { Self::from_parts(year, is_leap_year, new_ordinal as u16) });
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
            return Some(unsafe { Self::from_parts(year, is_leap_year, new_ordinal as u16) });
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
            return Some(unsafe { Self::from_parts(year, is_leap_year, new_ordinal as u16) });
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
    pub(crate) const fn checked_nth_next_occurrence(self, weekday: Weekday, n: u8) -> Option<Self> {
        if n == 0 {
            return None;
        }
        const_try_opt!(self.checked_next_occurrence(weekday)).checked_add(SignedDuration::weeks(n as i64 - 1))
    }

    /// Calculates the `n`th occurrence of a weekday that is strictly earlier than a given `Date`.
    /// Returns `None` if an overflow occurred or if `n == 0`.
    #[inline]
    pub(crate) const fn checked_nth_prev_occurrence(self, weekday: Weekday, n: u8) -> Option<Self> {
        if n == 0 {
            return None;
        }
        const_try_opt!(self.checked_prev_occurrence(weekday)).checked_sub(SignedDuration::weeks(n as i64 - 1))
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
                    value: unsafe { NonZero::new_unchecked((year << 10) | (self.value.get() & 0x3FF)) },
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
                DAYS_CUMULATIVE_COMMON_LEAP[is_leap_year as usize][month as usize - 1] + day as u16,
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
                (self.ordinal().cast_signed() - self.day() as i16 + day as i16).cast_unsigned(),
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
    pub const fn with_hms(self, hour: u8, minute: u8, second: u8) -> Result<PlainDateTime, ComponentRange> {
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
        let year_sign_width = if self.year() < 0 || (cfg!(feature = "large-dates") && self.year() >= 10_000) {
            1
        } else {
            0
        };
        let year_width = self
            .year()
            .unsigned_abs()
            .count_digits()
            .clamp(4, 6);
        let formatted_width = year_sign_width + year_width + 6; // include two dashes and two digits each for month and day

        Metadata::new(formatted_width, self, ())
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
    pub(crate) fn fmt_into_buffer(self, buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE]) -> usize {
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
            first_two
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), first_two.len());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date_util::days_in_month_leap;

    /// Cumulative days before each month (1-based index) in a common year.
    const CUMULATIVE: [u16; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

    /// Independent civil date to days since the Unix epoch (Howard Hinnant's `days_from_civil`).
    fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
        let y = i64::from(year) - i64::from(month <= 2);
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let m = if month > 2 { month - 3 } else { month + 9 };
        let doy = (153 * i64::from(m) + 2) / 5 + i64::from(day) - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// Days since the Unix epoch for a `Date`, derived from the crate's Julian day.
    fn epoch_days(date: Date) -> i64 {
        i64::from(date.to_julian_day()) - 2_440_588
    }

    /// Independent weekday from days since the epoch (1970-01-01 was a Thursday).
    fn reference_weekday(epoch_days: i64) -> Weekday {
        match (epoch_days + 3).rem_euclid(7) {
            0 => Weekday::Monday,
            1 => Weekday::Tuesday,
            2 => Weekday::Wednesday,
            3 => Weekday::Thursday,
            4 => Weekday::Friday,
            5 => Weekday::Saturday,
            _ => Weekday::Sunday,
        }
    }

    /// Independent ISO week date: the ISO year is the calendar year containing the week's
    /// Thursday, and week 1 is the week containing the first Thursday of the ISO year.
    fn reference_iso_week(date: Date) -> (i32, u8) {
        let epoch = epoch_days(date);
        let thursday = epoch + i64::from(4 - i32::from(date.weekday().number_from_monday()));
        let jan_1 = days_from_civil(date.year(), 1, 1);
        let jan_1_next = days_from_civil(date.year() + 1, 1, 1);
        let iso_year = if thursday < jan_1 {
            date.year() - 1
        } else if thursday >= jan_1_next {
            date.year() + 1
        } else {
            date.year()
        };
        let week = ((thursday - days_from_civil(iso_year, 1, 1)) / 7 + 1) as u8;
        (iso_year, week)
    }

    /// Independent conversion of an ordinal day to its month and day of month.
    fn reference_month_day(year: i32, ordinal: u16) -> (Month, u8) {
        let leap = is_leap_year(year);
        for index in 0..12 {
            let start = CUMULATIVE[index] + u16::from(index >= 2 && leap);
            let end = if index == 11 {
                if leap { 366 } else { 365 }
            } else {
                CUMULATIVE[index + 1] + u16::from(index + 1 >= 2 && leap)
            };
            if ordinal > start && ordinal <= end {
                let month = Month::try_from((index as u8) + 1).expect("month is in range");
                return (month, (ordinal - start) as u8);
            }
        }
        panic!("ordinal {ordinal} is out of range for year {year}");
    }

    /// Independent week numbering where week 1 contains the first `weekday` of the year.
    fn reference_week(ordinal: u16, days_from_week_start: u8) -> u8 {
        ((i32::from(ordinal) - 1 - i32::from(days_from_week_start)).div_euclid(7) + 1) as u8
    }

    #[test]
    fn min_and_max_bounds() {
        assert_eq!(Date::MIN.to_calendar_date(), (MIN_YEAR, Month::January, 1));
        assert_eq!(Date::MAX.to_calendar_date(), (MAX_YEAR, Month::December, 31));
        assert_eq!(Date::MIN.ordinal(), 1);
        assert_eq!(
            Date::MAX.ordinal(),
            if is_leap_year(MAX_YEAR) { 366 } else { 365 }
        );
        assert_eq!(Date::MIN.year(), MIN_YEAR);
        assert_eq!(Date::MAX.year(), MAX_YEAR);
        assert_eq!(Date::UNIX_EPOCH.to_calendar_date(), (1970, Month::January, 1));
        assert!(Date::MIN < Date::UNIX_EPOCH);
        assert!(Date::UNIX_EPOCH < Date::MAX);
    }

    #[test]
    fn from_calendar_date_round_trip_for_sampled_days() {
        // Every year in the representable range, sampled days within each year.
        for year in (MIN_YEAR..=MAX_YEAR).step_by(7) {
            let leap = is_leap_year(year);
            let days_in_year = if leap { 366 } else { 365 };
            for ordinal in [1u16, 31, 32, 59, 60, 61, 100, 200, days_in_year - 1, days_in_year] {
                let date = Date::from_ordinal_date(year, ordinal).expect("ordinal is in range");
                let (month, day) = reference_month_day(year, ordinal);
                assert_eq!(date.year(), year, "year for {year}-{ordinal}");
                assert_eq!(date.ordinal(), ordinal, "ordinal for {year}-{ordinal}");
                assert_eq!(date.month(), month, "month for {year}-{ordinal}");
                assert_eq!(date.day(), day, "day for {year}-{ordinal}");

                let rebuilt = Date::from_calendar_date(year, month, day).expect("date is valid");
                assert_eq!(rebuilt, date, "calendar round trip for {year}-{ordinal}");
                assert_eq!(rebuilt.to_calendar_date(), (year, month, day));
                assert!(date >= Date::MIN && date <= Date::MAX);
            }
        }
    }

    #[test]
    fn from_calendar_date_rejects_invalid_components() {
        // Day out of range for the specific month/year combination (conditional).
        for (year, month, day) in [
            (2023, Month::February, 29),
            (2100, Month::February, 29),
            (2023, Month::April, 31),
            (2024, Month::June, 31),
            (2024, Month::September, 31),
            (2024, Month::November, 31),
            (2024, Month::January, 32),
            (2024, Month::December, 0),
        ] {
            let error = Date::from_calendar_date(year, month, day).expect_err("must be rejected");
            assert_eq!(error.name(), "day");
            assert!(error.is_conditional(), "{year}-{month:?}-{day}");
            assert_eq!(error.to_string(), "day was not in range");
        }

        // Valid leap day and month ends.
        assert!(Date::from_calendar_date(2024, Month::February, 29).is_ok());
        assert!(Date::from_calendar_date(2000, Month::February, 29).is_ok());
        assert!(Date::from_calendar_date(2024, Month::April, 31).is_err());
        assert!(Date::from_calendar_date(2024, Month::April, 30).is_ok());
        assert!(Date::from_calendar_date(2023, Month::December, 31).is_ok());

        // Year out of range (unconditional).
        for year in [MAX_YEAR + 1, MIN_YEAR - 1, i32::MAX, i32::MIN] {
            let error = Date::from_calendar_date(year, Month::January, 1).expect_err("rejected");
            assert_eq!(error.name(), "year");
            assert!(!error.is_conditional(), "year {year}");
        }
    }

    #[test]
    fn from_ordinal_date_validates_range() {
        assert_eq!(
            Date::from_ordinal_date(2024, 60).map(Date::to_calendar_date),
            Ok((2024, Month::February, 29)),
        );
        assert_eq!(
            Date::from_ordinal_date(2023, 59).map(Date::to_calendar_date),
            Ok((2023, Month::February, 28)),
        );

        for (year, ordinal) in [(2023, 366), (2024, 367), (2024, 0), (2100, 366), (1, 0)] {
            let error = Date::from_ordinal_date(year, ordinal).expect_err("must be rejected");
            assert_eq!(error.name(), "ordinal");
            assert!(error.is_conditional(), "{year}-{ordinal}");
        }

        let error = Date::from_ordinal_date(MAX_YEAR + 1, 1).expect_err("must be rejected");
        assert_eq!(error.name(), "year");
        assert!(!error.is_conditional());

        // Boundary ordinals are accepted.
        assert!(Date::from_ordinal_date(MIN_YEAR, 1).is_ok());
        assert!(Date::from_ordinal_date(MAX_YEAR, 365).is_ok());
        assert!(Date::from_ordinal_date(2024, 366).is_ok());
    }

    #[test]
    fn iso_week_date_known_anchors() {
        let cases: [(Date, (i32, u8, Weekday)); 8] = [
            (
                Date::from_calendar_date(2020, Month::December, 31).expect("valid"),
                (2020, 53, Weekday::Thursday),
            ),
            (
                Date::from_calendar_date(2021, Month::January, 1).expect("valid"),
                (2020, 53, Weekday::Friday),
            ),
            (
                Date::from_calendar_date(2019, Month::December, 30).expect("valid"),
                (2020, 1, Weekday::Monday),
            ),
            (
                Date::from_calendar_date(2021, Month::December, 31).expect("valid"),
                (2021, 52, Weekday::Friday),
            ),
            (
                Date::from_calendar_date(2015, Month::December, 31).expect("valid"),
                (2015, 53, Weekday::Thursday),
            ),
            (
                Date::from_calendar_date(2016, Month::January, 1).expect("valid"),
                (2015, 53, Weekday::Friday),
            ),
            (
                Date::from_calendar_date(2024, Month::February, 29).expect("valid"),
                (2024, 9, Weekday::Thursday),
            ),
            (Date::UNIX_EPOCH, (1970, 1, Weekday::Thursday)),
        ];
        for (date, expected) in cases {
            assert_eq!(date.to_iso_week_date(), expected, "{date}");
            assert_eq!(date.iso_week(), expected.1, "{date}");
            let (year, week, weekday) = expected;
            assert_eq!(
                Date::from_iso_week_date(year, week, weekday).map(|d| (d, d.weekday())),
                Ok((date, weekday)),
                "round trip for {year}-W{week:02}-{weekday}",
            );
        }
    }

    #[test]
    fn iso_week_date_round_trip_and_reference() {
        let mut date = Date::from_calendar_date(1999, Month::December, 1).expect("valid");
        let end = Date::from_calendar_date(2005, Month::January, 31).expect("valid");
        while date <= end {
            let (year, week, weekday) = date.to_iso_week_date();
            assert_eq!(weekday, date.weekday(), "{date}");
            assert!((1..=53).contains(&week), "week {week} for {date}");
            assert_eq!(reference_iso_week(date), (year, week), "{date}");
            let rebuilt = Date::from_iso_week_date(year, week, weekday).expect("valid ISO date");
            assert_eq!(rebuilt, date, "round trip for {date}");
            date = date.next_day().expect("not at MAX");
        }
    }

    #[test]
    fn iso_week_date_rejects_invalid_weeks() {
        // 2021 has 52 ISO weeks.
        let error = Date::from_iso_week_date(2021, 53, Weekday::Monday).expect_err("must be rejected");
        assert_eq!(error.name(), "week");
        assert!(error.is_conditional());

        for week in [0u8, 54, 255] {
            let error = Date::from_iso_week_date(2020, week, Weekday::Monday).expect_err("rejected");
            assert_eq!(error.name(), "week");
            assert!(error.is_conditional(), "week {week}");
        }

        // 2020 does have 53 weeks.
        assert!(Date::from_iso_week_date(2020, 53, Weekday::Monday).is_ok());

        let error = Date::from_iso_week_date(MAX_YEAR + 1, 1, Weekday::Monday).expect_err("rejected");
        assert_eq!(error.name(), "year");
        assert!(!error.is_conditional());

        // The final week of the maximum year extends past `Date::MAX`.
        let error = Date::from_iso_week_date(MAX_YEAR, 52, Weekday::Sunday).expect_err("rejected");
        assert_eq!(error.name(), "weekday");
        assert!(error.is_conditional());
    }

    #[test]
    fn julian_day_known_values_and_round_trip() {
        // 1970-01-01 (Unix epoch), 2000-01-01 (J2000.0), 2024-02-29.
        assert_eq!(Date::UNIX_EPOCH.to_julian_day(), 2_440_588);
        let j2000 = Date::from_calendar_date(2000, Month::January, 1).expect("valid");
        assert_eq!(j2000.to_julian_day(), 2_451_545);
        let leap_day = Date::from_calendar_date(2024, Month::February, 29).expect("valid");
        assert_eq!(leap_day.to_julian_day(), 2_460_370);

        for date in [
            Date::MIN,
            Date::MAX,
            Date::UNIX_EPOCH,
            j2000,
            leap_day,
            Date::from_calendar_date(1, Month::January, 1).expect("valid"),
            Date::from_calendar_date(1999, Month::December, 31).expect("valid"),
            Date::from_calendar_date(-9999, Month::December, 31).expect("valid"),
        ] {
            let jd = date.to_julian_day();
            assert_eq!(Date::from_julian_day(jd).expect("in range"), date, "JD {jd}");
            assert_eq!(
                i64::from(jd),
                days_from_civil(
                    date.year(),
                    u32::from(u8::from(date.month())),
                    u32::from(date.day())
                ) + 2_440_588,
                "JD for {date}",
            );
        }
    }

    #[test]
    fn julian_day_rejects_values_out_of_range() {
        let min_jd = Date::MIN.to_julian_day();
        let max_jd = Date::MAX.to_julian_day();
        assert!(Date::from_julian_day(min_jd).is_ok());
        assert!(Date::from_julian_day(max_jd).is_ok());
        for jd in [min_jd - 1, max_jd + 1, i32::MAX, i32::MIN] {
            let error = Date::from_julian_day(jd).expect_err("must be rejected");
            assert_eq!(error.name(), "julian_day");
            assert!(!error.is_conditional(), "JD {jd}");
        }
    }

    #[test]
    fn full_sweep_of_every_representable_date() {
        let mut date = Date::MIN;
        let mut expected_epoch = days_from_civil(MIN_YEAR, 1, 1);
        let mut visited = 0i64;

        loop {
            let year = date.year();
            let ordinal = date.ordinal();
            assert!(date >= Date::MIN && date <= Date::MAX);

            // Julian day and epoch day advance one per day, matching the reference calendar.
            assert_eq!(
                i64::from(date.to_julian_day()),
                expected_epoch + 2_440_588,
                "Julian day for {date}",
            );
            assert_eq!(
                date.weekday(),
                reference_weekday(expected_epoch),
                "weekday for {date}"
            );

            // Calendar components agree with an independent ordinal-to-date conversion.
            let (month, day) = reference_month_day(year, ordinal);
            assert_eq!((date.month(), date.day()), (month, day), "components for {date}");
            assert_eq!(date.to_calendar_date(), (year, month, day), "{date}");
            assert_eq!(date.to_ordinal_date(), (year, ordinal), "{date}");
            assert!((1..=366).contains(&ordinal), "ordinal {ordinal} for {date}");
            assert!(ordinal <= days_in_year(year), "ordinal {ordinal} for {date}");
            assert!(day >= 1 && day <= days_in_month_leap(u8::from(month), is_leap_year(year)));

            // Construction from the observed components yields the same date.
            assert_eq!(
                Date::from_calendar_date(year, month, day).expect("valid"),
                date,
                "calendar round trip for {date}",
            );
            assert_eq!(Date::from_ordinal_date(year, ordinal).expect("valid"), date);

            visited += 1;
            if date == Date::MAX {
                break;
            }
            let next = date
                .next_day()
                .expect("next_day succeeds before MAX");
            assert!(next > date);
            assert_eq!(
                next.previous_day()
                    .expect("previous_day succeeds"),
                date
            );
            expected_epoch += 1;
            date = next;
        }

        let expected_days = days_from_civil(MAX_YEAR, 12, 31) - days_from_civil(MIN_YEAR, 1, 1) + 1;
        assert_eq!(visited, expected_days, "total number of representable dates");
    }

    #[test]
    fn next_day_and_previous_day_bounds() {
        assert_eq!(Date::MAX.next_day(), None);
        assert_eq!(Date::MIN.previous_day(), None);
        assert_eq!(
            Date::MAX
                .previous_day()
                .map(Date::to_calendar_date),
            Some((MAX_YEAR, Month::December, 30)),
        );
        assert_eq!(
            Date::MIN.next_day().map(Date::to_calendar_date),
            Some((MIN_YEAR, Month::January, 2)),
        );
        // Year boundaries wrap correctly.
        let jan_1 = Date::from_calendar_date(2024, Month::January, 1).expect("valid");
        assert_eq!(
            jan_1.previous_day().map(Date::to_calendar_date),
            Some((2023, Month::December, 31)),
        );
        let dec_31 = Date::from_calendar_date(2023, Month::December, 31).expect("valid");
        assert_eq!(
            dec_31.next_day().map(Date::to_calendar_date),
            Some((2024, Month::January, 1)),
        );
        // Leap day handling.
        let feb_28 = Date::from_calendar_date(2024, Month::February, 28).expect("valid");
        assert_eq!(
            feb_28.next_day().map(Date::to_calendar_date),
            Some((2024, Month::February, 29)),
        );
        let feb_28_common = Date::from_calendar_date(2023, Month::February, 28).expect("valid");
        assert_eq!(
            feb_28_common
                .next_day()
                .map(Date::to_calendar_date),
            Some((2023, Month::March, 1)),
        );
    }

    #[test]
    fn week_numbers_match_reference() {
        // 2021-01-01 is a Friday: it belongs to week 0 for both numbering schemes.
        let date = Date::from_calendar_date(2021, Month::January, 1).expect("valid");
        assert_eq!(date.monday_based_week(), 0);
        assert_eq!(date.sunday_based_week(), 0);
        // The first Monday of 2021 (Jan 4) starts week 1.
        let date = Date::from_calendar_date(2021, Month::January, 4).expect("valid");
        assert_eq!(date.monday_based_week(), 1);
        assert_eq!(date.sunday_based_week(), 1);
        // 1970-01-01 is a Thursday; the first Sunday (Jan 4) and first Monday (Jan 5) both
        // fall later, so it belongs to week 0 of both numbering schemes.
        assert_eq!(Date::UNIX_EPOCH.sunday_based_week(), 0);
        assert_eq!(Date::UNIX_EPOCH.monday_based_week(), 0);

        for year in (MIN_YEAR..=MAX_YEAR).step_by(11) {
            for ordinal in [1u16, 2, 7, 8, 31, 100, 364, 365, 366] {
                let Ok(date) = Date::from_ordinal_date(year, ordinal) else {
                    continue;
                };
                assert_eq!(
                    date.sunday_based_week(),
                    reference_week(ordinal, date.weekday().number_days_from_sunday()),
                    "sunday-based week for {date}",
                );
                assert_eq!(
                    date.monday_based_week(),
                    reference_week(ordinal, date.weekday().number_days_from_monday()),
                    "monday-based week for {date}",
                );
                assert!((0..=53).contains(&date.sunday_based_week()), "{date}");
                assert!((0..=53).contains(&date.monday_based_week()), "{date}");
            }
        }
    }

    #[test]
    fn weekday_reference_agreement_across_years() {
        for year in (MIN_YEAR..=MAX_YEAR).step_by(3) {
            for (month, day) in [
                (Month::January, 1),
                (Month::February, 28),
                (Month::June, 15),
                (Month::December, 31),
            ] {
                let date = Date::from_calendar_date(year, month, day).expect("valid");
                assert_eq!(date.weekday(), reference_weekday(epoch_days(date)), "{date}",);
            }
        }
    }

    #[test]
    fn next_and_prev_occurrence_navigate_weekdays() {
        let date = Date::from_calendar_date(2024, Month::March, 6).expect("valid"); // Wednesday
        assert_eq!(date.weekday(), Weekday::Wednesday);

        assert_eq!(
            date.next_occurrence(Weekday::Friday)
                .to_calendar_date(),
            (2024, Month::March, 8),
        );
        // A matching weekday yields the following week, since occurrences are strictly later.
        assert_eq!(
            date.next_occurrence(Weekday::Wednesday)
                .to_calendar_date(),
            (2024, Month::March, 13),
        );
        assert_eq!(
            date.prev_occurrence(Weekday::Friday)
                .to_calendar_date(),
            (2024, Month::March, 1),
        );
        assert_eq!(
            date.prev_occurrence(Weekday::Wednesday)
                .to_calendar_date(),
            (2024, Month::February, 28),
        );
        assert_eq!(
            date.nth_next_occurrence(Weekday::Friday, 3)
                .to_calendar_date(),
            (2024, Month::March, 22),
        );
        // The first occurrence is strictly earlier (2024-03-01), so n = 3 lands two weeks
        // before it.
        assert_eq!(
            date.nth_prev_occurrence(Weekday::Friday, 3)
                .to_calendar_date(),
            (2024, Month::February, 16),
        );
        // n == 0 returns the date itself.
        assert_eq!(date.nth_next_occurrence(Weekday::Friday, 0), date);
        assert_eq!(date.nth_prev_occurrence(Weekday::Friday, 0), date);

        for weekday in [
            Weekday::Monday,
            Weekday::Tuesday,
            Weekday::Wednesday,
            Weekday::Thursday,
            Weekday::Friday,
            Weekday::Saturday,
            Weekday::Sunday,
        ] {
            let next = date.next_occurrence(weekday);
            assert_eq!(next.weekday(), weekday);
            assert!(next > date);
            assert!(next <= date + SignedDuration::days(7));

            let prev = date.prev_occurrence(weekday);
            assert_eq!(prev.weekday(), weekday);
            assert!(prev < date);
            assert!(prev >= date - SignedDuration::days(7));
        }
    }

    #[test]
    fn occurrence_overflow_saturates() {
        // Occurring on the same weekday is a full week away, which overflows at the bounds.
        assert_eq!(Date::MAX.next_occurrence(Weekday::Friday), Date::MAX);
        assert_eq!(Date::MIN.prev_occurrence(Weekday::Monday), Date::MIN);
        // Occurrences that stay within the range resolve normally.
        assert_eq!(
            Date::MAX
                .prev_occurrence(Weekday::Monday)
                .to_calendar_date(),
            (9999, Month::December, 27),
        );
        assert_eq!(
            Date::MIN
                .next_occurrence(Weekday::Monday)
                .to_calendar_date(),
            (-9999, Month::January, 8),
        );
        // Large distances saturate rather than overflowing.
        assert_eq!(Date::MAX.nth_next_occurrence(Weekday::Monday, 200), Date::MAX,);
        assert_eq!(Date::MIN.nth_prev_occurrence(Weekday::Monday, 200), Date::MIN,);
        // The checked variants report the overflow as `None`.
        assert_eq!(Date::MAX.checked_next_occurrence(Weekday::Monday), None);
        assert_eq!(Date::MAX.checked_next_occurrence(Weekday::Friday), None);
        assert_eq!(Date::MIN.checked_prev_occurrence(Weekday::Monday), None);
        assert_eq!(Date::MAX.checked_nth_next_occurrence(Weekday::Monday, 1), None,);
        assert_eq!(Date::MIN.checked_nth_prev_occurrence(Weekday::Monday, 1), None,);
        // n == 0 is rejected by the checked variants but handled by the panicking ones.
        assert_eq!(
            Date::UNIX_EPOCH.checked_nth_next_occurrence(Weekday::Monday, 0),
            None,
        );
        assert_eq!(
            Date::UNIX_EPOCH.checked_nth_prev_occurrence(Weekday::Monday, 0),
            None,
        );
        // A date just inside the range resolves when the target stays inside, but not when
        // the next occurrence would fall past `Date::MAX`.
        let almost_max = Date::MAX.previous_day().expect("not MIN"); // a Thursday
        assert_eq!(
            almost_max.checked_next_occurrence(Weekday::Friday),
            Some(Date::MAX),
        );
        assert_eq!(almost_max.checked_next_occurrence(Weekday::Monday), None);
    }

    #[test]
    fn checked_and_saturating_arithmetic() {
        let date = Date::from_calendar_date(2024, Month::January, 1).expect("valid");

        // Whole days only: sub-day durations are ignored.
        assert_eq!(date.checked_add(SignedDuration::hours(23)), Some(date));
        assert_eq!(
            date.checked_add(SignedDuration::hours(24))
                .map(Date::to_calendar_date),
            Some((2024, Month::January, 2)),
        );
        assert_eq!(
            date.checked_add(SignedDuration::days(31))
                .map(Date::to_calendar_date),
            Some((2024, Month::February, 1)),
        );
        assert_eq!(
            date.checked_sub(SignedDuration::days(1))
                .map(Date::to_calendar_date),
            Some((2023, Month::December, 31)),
        );
        // Adding and then subtracting returns the original date.
        let duration = SignedDuration::days(12_345);
        assert_eq!(
            date.checked_add(duration)
                .and_then(|d| d.checked_sub(duration)),
            Some(date)
        );

        // Overflow yields None; the saturating variants clamp.
        assert_eq!(Date::MAX.checked_add(SignedDuration::days(1)), None);
        assert_eq!(Date::MIN.checked_sub(SignedDuration::days(1)), None);
        // Moving away from the bounds always succeeds.
        assert!(
            Date::MAX
                .checked_add(SignedDuration::days(-1))
                .is_some()
        );
        assert!(
            Date::MIN
                .checked_sub(SignedDuration::days(-1))
                .is_some()
        );
        assert_eq!(Date::MAX.saturating_add(SignedDuration::days(1)), Date::MAX);
        assert_eq!(Date::MIN.saturating_sub(SignedDuration::days(1)), Date::MIN);
        // Saturating operations clamp to the bound only when moving past it.
        assert_eq!(
            Date::MIN.saturating_add(SignedDuration::days(1)),
            Date::MIN.next_day().expect("MIN is not MAX"),
        );
        assert_eq!(
            Date::MAX.saturating_sub(SignedDuration::days(1)),
            Date::MAX.previous_day().expect("MAX is not MIN"),
        );

        // Durations far outside the i32 day range.
        let huge = SignedDuration::days(i64::from(i32::MAX) + 1);
        assert_eq!(date.checked_add(huge), None);
        assert_eq!(date.checked_sub(huge), None);
        assert_eq!(date.checked_add(-huge), None);
        assert_eq!(date.saturating_add(huge), Date::MAX);
        assert_eq!(date.saturating_sub(huge), Date::MIN);

        // Standard durations.
        let two_days = StdDuration::from_secs(2 * 86_400);
        assert_eq!(
            date.checked_add_std(two_days)
                .map(Date::to_calendar_date),
            Some((2024, Month::January, 3)),
        );
        assert_eq!(
            date.checked_sub_std(two_days)
                .map(Date::to_calendar_date),
            Some((2023, Month::December, 30)),
        );
        // Standard durations also only count whole days.
        assert_eq!(date.checked_add_std(StdDuration::from_secs(86_399)), Some(date));
        assert_eq!(Date::MAX.checked_add_std(StdDuration::from_secs(u64::MAX)), None);
        assert_eq!(Date::MIN.checked_sub_std(StdDuration::from_secs(u64::MAX)), None);
        assert!(
            Date::MAX
                .checked_sub_std(StdDuration::from_secs(86_400))
                .is_some()
        );
    }

    #[test]
    fn arithmetic_operator_traits() {
        let date = Date::from_calendar_date(2024, Month::June, 15).expect("valid");
        let one_day = SignedDuration::days(1);
        let three_days = SignedDuration::days(3);

        assert_eq!((date + three_days).to_calendar_date(), (2024, Month::June, 18),);
        assert_eq!((date - three_days).to_calendar_date(), (2024, Month::June, 12));
        assert_eq!(date - (date - three_days), three_days);
        assert_eq!((date + three_days) - date, three_days);

        let mut mutated = date;
        mutated += three_days;
        assert_eq!(mutated.to_calendar_date(), (2024, Month::June, 18));
        mutated -= three_days;
        assert_eq!(mutated, date);
        assert_eq!(mutated, date + one_day - one_day);

        // Out-of-range operator additions saturate instead of panicking.
        assert_eq!(Date::MAX + one_day, Date::MAX);
        assert_eq!(Date::MIN - one_day, Date::MIN);
        assert_eq!(Date::MAX - one_day, Date::MAX.previous_day().expect("not MIN"));
        assert_eq!(Date::MIN + one_day, Date::MIN.next_day().expect("not MAX"));

        // Standard duration operators saturate as well.
        let std_one_day = StdDuration::from_secs(86_400);
        assert_eq!((Date::MAX + std_one_day), Date::MAX);
        assert_eq!((Date::MIN - std_one_day), Date::MIN);
        let mut mutated = date;
        mutated += std_one_day;
        assert_eq!(mutated, date + one_day);
        mutated -= std_one_day;
        assert_eq!(mutated, date);
    }

    #[test]
    fn replace_year_adjusts_for_leap_years() {
        // Dates in January and February are unaffected by leap status.
        let feb_28 = Date::from_calendar_date(2024, Month::February, 28).expect("valid");
        assert_eq!(
            feb_28
                .replace_year(2023)
                .map(Date::to_calendar_date),
            Ok((2023, Month::February, 28)),
        );
        assert_eq!(
            feb_28
                .replace_year(2024)
                .map(Date::to_calendar_date),
            Ok((2024, Month::February, 28)),
        );

        // February 29 cannot be replaced into a common year.
        let leap_day = Date::from_calendar_date(2024, Month::February, 29).expect("valid");
        let error = leap_day
            .replace_year(2023)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "day");
        assert!(error.is_conditional());
        assert!(leap_day.replace_year(2000).is_ok());

        // March and later keep their calendar date when leap status changes.
        let march_1_leap = Date::from_calendar_date(2024, Month::March, 1).expect("valid");
        assert_eq!(
            march_1_leap
                .replace_year(2023)
                .map(Date::to_calendar_date),
            Ok((2023, Month::March, 1)),
        );
        let march_1_common = Date::from_calendar_date(2023, Month::March, 1).expect("valid");
        assert_eq!(
            march_1_common
                .replace_year(2024)
                .map(Date::to_calendar_date),
            Ok((2024, Month::March, 1)),
        );
        let june_15_common = Date::from_calendar_date(2023, Month::June, 15).expect("valid");
        assert_eq!(
            june_15_common
                .replace_year(2024)
                .map(Date::to_calendar_date),
            Ok((2024, Month::June, 15)),
        );
        let june_15_leap = Date::from_calendar_date(2024, Month::June, 15).expect("valid");
        assert_eq!(
            june_15_leap
                .replace_year(2023)
                .map(Date::to_calendar_date),
            Ok((2023, Month::June, 15)),
        );

        for year in [MAX_YEAR + 1, MIN_YEAR - 1] {
            let error = june_15_common
                .replace_year(year)
                .expect_err("must be rejected");
            assert_eq!(error.name(), "year");
            assert!(!error.is_conditional());
        }
        assert!(june_15_common.replace_year(MAX_YEAR).is_ok());
    }

    #[test]
    fn replace_month_keeps_day_or_rejects_it() {
        let date = Date::from_calendar_date(2024, Month::January, 31).expect("valid");
        assert_eq!(
            date.replace_month(Month::March)
                .map(Date::to_calendar_date),
            Ok((2024, Month::March, 31)),
        );
        assert_eq!(
            date.replace_month(Month::January)
                .map(Date::to_calendar_date),
            Ok((2024, Month::January, 31)),
        );
        // 31 does not exist in February, even in a leap year.
        let error = date
            .replace_month(Month::February)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "day");
        assert!(error.is_conditional());

        let date = Date::from_calendar_date(2024, Month::March, 15).expect("valid");
        assert_eq!(
            date.replace_month(Month::February)
                .map(Date::to_calendar_date),
            Ok((2024, Month::February, 15)),
        );
        let date = Date::from_calendar_date(2023, Month::March, 31).expect("valid");
        let error = date
            .replace_month(Month::February)
            .expect_err("must be rejected");
        assert!(error.is_conditional());
        // May has 31 days, so the day is preserved.
        assert_eq!(
            date.replace_month(Month::May)
                .map(Date::to_calendar_date),
            Ok((2023, Month::May, 31)),
        );
    }

    #[test]
    fn replace_day_and_replace_ordinal_validate_ranges() {
        let date = Date::from_calendar_date(2023, Month::January, 15).expect("valid");
        assert_eq!(
            date.replace_day(29).map(Date::to_calendar_date),
            Ok((2023, Month::January, 29)),
        );
        let error = date.replace_day(0).expect_err("must be rejected");
        assert_eq!(error.name(), "day");
        assert!(error.is_conditional());
        let error = date
            .replace_day(32)
            .expect_err("must be rejected");
        assert!(error.is_conditional());

        // February 29 only exists in leap years.
        let feb = Date::from_calendar_date(2024, Month::February, 1).expect("valid");
        assert_eq!(
            feb.replace_day(29).map(Date::to_calendar_date),
            Ok((2024, Month::February, 29)),
        );
        let feb_common = Date::from_calendar_date(2023, Month::February, 1).expect("valid");
        let error = feb_common
            .replace_day(29)
            .expect_err("must be rejected");
        assert!(error.is_conditional());

        let date = Date::from_calendar_date(2024, Month::May, 1).expect("valid");
        assert_eq!(
            date.replace_ordinal(366)
                .map(Date::to_calendar_date),
            Ok((2024, Month::December, 31)),
        );
        assert_eq!(date.replace_ordinal(1).map(Date::ordinal), Ok(1));
        let error = date
            .replace_ordinal(0)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "ordinal");
        assert!(error.is_conditional());
        let error = date
            .replace_ordinal(367)
            .expect_err("must be rejected");
        assert!(error.is_conditional());

        let date_common = Date::from_calendar_date(2023, Month::May, 1).expect("valid");
        let error = date_common
            .replace_ordinal(366)
            .expect_err("must be rejected");
        assert!(error.is_conditional());
        assert!(date_common.replace_ordinal(365).is_ok());
    }

    #[test]
    fn display_formats_iso_dates() {
        let cases = [
            (Date::UNIX_EPOCH, "1970-01-01"),
            (Date::MIN, "-9999-01-01"),
            (Date::MAX, "9999-12-31"),
            (
                Date::from_calendar_date(2024, Month::February, 29).expect("valid"),
                "2024-02-29",
            ),
            (
                Date::from_calendar_date(1, Month::January, 1).expect("valid"),
                "0001-01-01",
            ),
            (
                Date::from_calendar_date(-9999, Month::December, 31).expect("valid"),
                "-9999-12-31",
            ),
            (
                Date::from_calendar_date(9999, Month::June, 7).expect("valid"),
                "9999-06-07",
            ),
        ];
        for (date, expected) in cases {
            assert_eq!(date.to_string(), expected);
            assert_eq!(format!("{date:?}"), expected, "Debug matches Display");
            assert_eq!(
                date.metadata(FormatterOptions::default())
                    .unpadded_width(),
                expected.len()
            );
            // Zero-padded components, including for years below 1000.
            assert_eq!(expected.len(), date.to_string().len());
        }

        // Formatting flags are honored by Display.
        let date = Date::UNIX_EPOCH;
        assert_eq!(format!("{date:>15}"), "     1970-01-01");
        assert_eq!(format!("{date:<15}"), "1970-01-01     ");
        assert_eq!(format!("{date:^15}"), "  1970-01-01   ");
        assert_eq!(format!("{date:.5}"), "1970-");
    }

    #[test]
    fn date_converts_to_plain_date_time() {
        let date = Date::from_calendar_date(2024, Month::February, 29).expect("valid");

        let midnight = date.midnight();
        assert_eq!(midnight.date(), date);
        assert_eq!(midnight.time(), Time::MIDNIGHT);

        let time = Time::from_hms_nano(13, 45, 59, 123_456_789).expect("valid");
        let with_time = date.with_time(time);
        assert_eq!(with_time.date(), date);
        assert_eq!(with_time.time(), time);

        let with_hms = date.with_hms(1, 2, 3).expect("valid time");
        assert_eq!(with_hms.date(), date);
        assert_eq!(with_hms.time(), Time::from_hms(1, 2, 3).expect("valid"));

        let with_milli = date
            .with_hms_milli(1, 2, 3, 999)
            .expect("valid time");
        assert_eq!(with_milli.time().millisecond(), 999);

        let with_micro = date
            .with_hms_micro(1, 2, 3, 999_999)
            .expect("valid time");
        assert_eq!(with_micro.time().microsecond(), 999_999);

        let with_nano = date
            .with_hms_nano(1, 2, 3, 999_999_999)
            .expect("valid time");
        assert_eq!(with_nano.time().nanosecond(), 999_999_999);

        // Invalid time components are rejected with the offending component named.
        let error = date
            .with_hms(24, 0, 0)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "hour");
        assert!(!error.is_conditional());
        let error = date
            .with_hms(0, 60, 0)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "minute");
        let error = date
            .with_hms(0, 0, 60)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "second");
        let error = date
            .with_hms_milli(0, 0, 0, 1_000)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "millisecond");
        let error = date
            .with_hms_micro(0, 0, 0, 1_000_000)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "microsecond");
        let error = date
            .with_hms_nano(0, 0, 0, 1_000_000_000)
            .expect_err("must be rejected");
        assert_eq!(error.name(), "nanosecond");
    }

    #[test]
    fn dates_are_ordered_and_hashable() {
        use core::hash::{Hash, Hasher};

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let early = Date::from_calendar_date(1999, Month::December, 31).expect("valid");
        let epoch = Date::UNIX_EPOCH;
        let late = Date::from_calendar_date(2100, Month::January, 1).expect("valid");

        assert!(epoch < early);
        assert!(early < late);
        assert!(Date::MIN < epoch);
        assert!(late < Date::MAX);

        // Equal values hash equally; distinct values are unlikely to collide.
        let early_copy = Date::from_ordinal_date(1999, 365).expect("valid");
        assert_eq!(early, early_copy);
        assert_eq!(hash_of(&early), hash_of(&early_copy));
        assert_ne!(hash_of(&early), hash_of(&late));
        assert_ne!(hash_of(&epoch), hash_of(&Date::MIN));

        // Chronological order agrees with Julian day order.
        for (a, b) in [
            (Date::MIN, epoch),
            (epoch, early),
            (early, late),
            (late, Date::MAX),
        ] {
            assert_eq!(a < b, a.to_julian_day() < b.to_julian_day());
        }

        // Ordinal ordering matches chronological ordering within a year.
        let mut ordinal = 2u16;
        while ordinal <= 365 {
            let date = Date::from_ordinal_date(2023, ordinal).expect("valid");
            let previous = Date::from_ordinal_date(2023, ordinal - 1).expect("valid");
            assert!(date > previous, "{date} should follow {previous}");
            ordinal += 1;
        }
    }
}
