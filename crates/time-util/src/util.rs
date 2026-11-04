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

use crate::month::Month;
use core::num::NonZero;

/// Which direction arithmetic overflow occurred in.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Overflow {
    /// The overflow was positive (i.e. towards positive infinity).
    Positive,
    /// The overflow was negative (i.e. towards negative infinity).
    Negative,
}

/// Whether to adjust the date, and in which direction. Useful when implementing arithmetic.
pub(crate) enum DateAdjustment {
    /// The previous day should be used.
    Previous,
    /// The next day should be used.
    Next,
    /// The date should be used as-is.
    None,
}

/// Returns if the provided year is a leap year in the proleptic Gregorian calendar, assuming
/// the year has already been range-validated.
///
/// Behavior is unspecified for years outside the valid range.
#[inline]
#[track_caller]
pub const fn is_leap_year(year: i32) -> bool {
    #[cfg(feature = "large-dates")]
    {
        super::is_leap_year(year)
    }
    #[cfg(not(feature = "large-dates"))]
    {
        debug_assert!(year >= -9999);
        debug_assert!(year <= 9999);
        year.unsigned_abs().wrapping_mul(0x20003D7) & 0x6007C0F <= 0x7C00
    }
}

/// Get the number of calendar days in a given year, assuming the year has already been
/// range-validated.
///
/// Behavior is unspecified for years outside the valid range.
#[inline]
#[track_caller]
pub const fn days_in_year(year: i32) -> u16 {
    #[cfg(feature = "large-dates")]
    {
        super::days_in_year(year)
    }
    #[cfg(not(feature = "large-dates"))]
    {
        if is_leap_year(year) { 366 } else { 365 }
    }
}

/// Get the number of days in the month of a given year, assuming the year has already been
/// range-validated.
#[inline]
#[track_caller]
pub const fn days_in_month(month: u8, year: i32) -> u8 {
    #[cfg(feature = "large-dates")]
    {
        super::days_in_month(month, year)
    }
    #[cfg(not(feature = "large-dates"))]
    {
        days_in_month_leap(month, is_leap_year(year))
    }
}

/// Get the number of weeks in the ISO year.
///
/// The returned value will always be either 52 or 53.
#[inline]
pub const fn weeks_in_year(year: i32) -> u8 {
    match year % 400 {
        -396 | -391 | -385 | -380 | -374 | -368 | -363 | -357 | -352 | -346 | -340 | -335 | -329 | -324
        | -318 | -312 | -307 | -301 | -295 | -289 | -284 | -278 | -272 | -267 | -261 | -256 | -250 | -244
        | -239 | -233 | -228 | -222 | -216 | -211 | -205 | -199 | -193 | -188 | -182 | -176 | -171 | -165
        | -160 | -154 | -148 | -143 | -137 | -132 | -126 | -120 | -115 | -109 | -104 | -97 | -92 | -86
        | -80 | -75 | -69 | -64 | -58 | -52 | -47 | -41 | -36 | -30 | -24 | -19 | -13 | -8 | -2 | 4 | 9
        | 15 | 20 | 26 | 32 | 37 | 43 | 48 | 54 | 60 | 65 | 71 | 76 | 82 | 88 | 93 | 99 | 105 | 111 | 116
        | 122 | 128 | 133 | 139 | 144 | 150 | 156 | 161 | 167 | 172 | 178 | 184 | 189 | 195 | 201 | 207
        | 212 | 218 | 224 | 229 | 235 | 240 | 246 | 252 | 257 | 263 | 268 | 274 | 280 | 285 | 291 | 296
        | 303 | 308 | 314 | 320 | 325 | 331 | 336 | 342 | 348 | 353 | 359 | 364 | 370 | 376 | 381 | 387
        | 392 | 398 => 53,
        _ => 52,
    }
}

/// Get the number of days in the month. The year does not need to be known, but whether the
/// year is a leap year does.
#[inline]
#[track_caller]
pub const fn days_in_month_leap(month: u8, is_leap_year: bool) -> u8 {
    debug_assert!(month >= 1);
    debug_assert!(month <= 12);

    if month == 2 {
        if is_leap_year { 29 } else { 28 }
    } else {
        30 | month ^ (month >> 3)
    }
}

/// Given whether a year is a leap year and the ordinal day, return the month and day of the month.
#[inline]
pub(crate) const fn leap_ordinal_to_month_day(leap: bool, ordinal: u16) -> (Month, u8) {
    let ordinal = ordinal as u32;
    let jan_feb_len = 59 + leap as u32;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent implementation of the proleptic Gregorian leap year rule, used as the
    /// reference for `is_leap_year`. Divisibility works identically for negative years.
    const fn reference_is_leap_year(year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
    }

    /// Number of days from 1970-01-01 to `year`-01-01, computed with an independent
    /// re-derivation of Howard Hinnant's `days_from_civil` (astronomical year numbering),
    /// anchored on 1970-01-01 being a Thursday.
    fn days_since_epoch_to_jan_1(year: i32) -> i64 {
        let y = i64::from(year) - 1;
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let doy = (153 * (1 + 9) + 2) / 5; // January 1 as days since March 1 of the prior year.
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// Zero-indexed weekday of `year`-01-01 where Monday = 0, derived independently from
    /// `days_since_epoch_to_jan_1` and the known anchor 1970-01-01 = Thursday (= 3).
    fn jan_1_weekday_monday_index(year: i32) -> i64 {
        (days_since_epoch_to_jan_1(year) + 3).rem_euclid(7)
    }

    /// Independent reference for the number of ISO weeks in a year: 53 iff 1 January is a
    /// Thursday, or the year is a leap year and 1 January is a Wednesday.
    fn reference_weeks_in_year(year: i32) -> u8 {
        let jan_1 = jan_1_weekday_monday_index(year);
        if jan_1 == 3 || (reference_is_leap_year(year) && jan_1 == 2) {
            53
        } else {
            52
        }
    }

    /// Common lengths of the twelve months, excluding leap day.
    const COMMON_LENGTHS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    #[test]
    fn is_leap_year_matches_gregorian_rule_for_all_representable_years() {
        for year in -9999..=9999 {
            assert_eq!(
                is_leap_year(year),
                reference_is_leap_year(year),
                "disagreement for year {year}",
            );
        }
    }

    #[test]
    fn is_leap_year_known_values() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(is_leap_year(1996));
        assert!(is_leap_year(0));
        assert!(is_leap_year(-4));
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2100));
        assert!(!is_leap_year(-9999));
        assert!(!is_leap_year(9999));
    }

    #[test]
    fn days_in_year_matches_leap_rule_for_all_representable_years() {
        for year in -9999..=9999 {
            let expected = if reference_is_leap_year(year) { 366 } else { 365 };
            assert_eq!(days_in_year(year), expected, "disagreement for year {year}");
        }
    }

    #[test]
    fn days_in_month_matches_reference_table() {
        // Representative leap years (including century leap year) and common years
        // (including non-leap century years).
        for year in [1900, 1999, 2000, 2023, 2024, 2100, -9999, -4, 4] {
            let leap = reference_is_leap_year(year);
            for (index, &expected) in COMMON_LENGTHS.iter().enumerate() {
                let month = index as u8 + 1;
                let expected = if month == 2 && leap { 29 } else { expected };
                assert_eq!(
                    days_in_month(month, year),
                    expected,
                    "disagreement for {year}-{month:02}",
                );
            }
        }
    }

    #[test]
    fn days_in_month_leap_depends_only_on_leap_flag() {
        for (index, &expected) in COMMON_LENGTHS.iter().enumerate() {
            let month = index as u8 + 1;
            assert_eq!(days_in_month_leap(month, false), expected);
            let expected = if month == 2 { 29 } else { expected };
            assert_eq!(days_in_month_leap(month, true), expected);
        }
    }

    #[test]
    fn weeks_in_year_matches_iso_reference_for_all_representable_years() {
        let mut years_with_53_weeks = 0;
        for year in -9999..=9999 {
            let expected = reference_weeks_in_year(year);
            let actual = weeks_in_year(year);
            assert_eq!(actual, expected, "disagreement for year {year}");
            assert!(actual == 52 || actual == 53, "unexpected value for {year}");
            if actual == 53 {
                years_with_53_weeks += 1;
            }
        }
        // 71 of every 400 years have 53 ISO weeks; guard against a trivially-constant
        // implementation.
        assert!((3_000..4_000).contains(&years_with_53_weeks));
    }

    #[test]
    fn weeks_in_year_known_values() {
        assert_eq!(weeks_in_year(2015), 53); // starts on Thursday
        assert_eq!(weeks_in_year(2020), 53); // leap year starting on Wednesday
        assert_eq!(weeks_in_year(2021), 52);
        assert_eq!(weeks_in_year(2004), 53);
        assert_eq!(weeks_in_year(2000), 52); // leap year starting on Saturday
        assert_eq!(weeks_in_year(9999), 52);
        assert_eq!(weeks_in_year(-9999), 52); // starts on Monday
        assert_eq!(weeks_in_year(1), 52); // proleptic Gregorian year 1 starts on Monday
    }

    #[test]
    fn leap_ordinal_to_month_day_boundaries() {
        assert_eq!(leap_ordinal_to_month_day(true, 1), (Month::January, 1));
        assert_eq!(leap_ordinal_to_month_day(true, 31), (Month::January, 31));
        assert_eq!(leap_ordinal_to_month_day(true, 32), (Month::February, 1));
        assert_eq!(leap_ordinal_to_month_day(true, 59), (Month::February, 28));
        assert_eq!(leap_ordinal_to_month_day(true, 60), (Month::February, 29));
        assert_eq!(leap_ordinal_to_month_day(true, 61), (Month::March, 1));
        assert_eq!(leap_ordinal_to_month_day(true, 365), (Month::December, 30));
        assert_eq!(leap_ordinal_to_month_day(true, 366), (Month::December, 31));

        assert_eq!(leap_ordinal_to_month_day(false, 1), (Month::January, 1));
        assert_eq!(leap_ordinal_to_month_day(false, 31), (Month::January, 31));
        assert_eq!(leap_ordinal_to_month_day(false, 32), (Month::February, 1));
        assert_eq!(leap_ordinal_to_month_day(false, 58), (Month::February, 27));
        assert_eq!(leap_ordinal_to_month_day(false, 59), (Month::February, 28));
        assert_eq!(leap_ordinal_to_month_day(false, 60), (Month::March, 1));
        assert_eq!(leap_ordinal_to_month_day(false, 365), (Month::December, 31));
    }

    #[test]
    fn leap_ordinal_to_month_day_sweeps_entire_years() {
        for leap in [false, true] {
            let last_ordinal = if leap { 366 } else { 365 };
            for ordinal in 1..=last_ordinal {
                let (month, day) = leap_ordinal_to_month_day(leap, ordinal);
                let month_index = u8::from(month) as usize - 1;
                let days_before = COMMON_LENGTHS[..month_index]
                    .iter()
                    .enumerate()
                    .map(|(index, &days)| u16::from(if index == 1 && leap { 29 } else { days }))
                    .sum::<u16>();
                let days_in_this_month = if month_index == 1 && leap {
                    29
                } else {
                    COMMON_LENGTHS[month_index]
                };
                assert!(
                    day >= 1 && day <= days_in_this_month,
                    "ordinal {ordinal} (leap={leap}) produced {month:?} {day}",
                );
                assert_eq!(
                    days_before + u16::from(day),
                    ordinal,
                    "ordinal {ordinal} (leap={leap}) produced {month:?} {day}",
                );
            }
        }
    }

    #[test]
    fn jan_1_reference_matches_known_weekdays() {
        // Validating the test's own reference machinery against well-known weekdays
        // (Monday = 0).
        assert_eq!(days_since_epoch_to_jan_1(1970), 0);
        assert_eq!(jan_1_weekday_monday_index(1970), 3); // Thursday
        assert_eq!(jan_1_weekday_monday_index(1999), 4); // Friday
        assert_eq!(jan_1_weekday_monday_index(2000), 5); // Saturday
        assert_eq!(jan_1_weekday_monday_index(2024), 0); // Monday
        assert_eq!(jan_1_weekday_monday_index(2021), 4); // Friday
        // 0001-01-01 is a Monday in the proleptic Gregorian calendar.
        assert_eq!(jan_1_weekday_monday_index(1), 0);
    }
}
