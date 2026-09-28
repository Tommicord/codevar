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

//! The `Month` enum and its associated `impl`s.

use core::fmt;
use core::num::NonZero;
use core::str::FromStr;

use self::Month::*;
use crate::date_error::{ComponentRange, InvalidVariant};
use crate::date_util::days_in_month;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

/// Months of the year.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Month {
    #[allow(missing_docs)]
    January = 1,
    #[allow(missing_docs)]
    February = 2,
    #[allow(missing_docs)]
    March = 3,
    #[allow(missing_docs)]
    April = 4,
    #[allow(missing_docs)]
    May = 5,
    #[allow(missing_docs)]
    June = 6,
    #[allow(missing_docs)]
    July = 7,
    #[allow(missing_docs)]
    August = 8,
    #[allow(missing_docs)]
    September = 9,
    #[allow(missing_docs)]
    October = 10,
    #[allow(missing_docs)]
    November = 11,
    #[allow(missing_docs)]
    December = 12,
}

impl Month {
    /// Create a `Month` from its numerical value.
    #[inline]
    pub(crate) const fn from_number(n: NonZero<u8>) -> Result<Self, ComponentRange> {
        match n.get() {
            1 => Ok(January),
            2 => Ok(February),
            3 => Ok(March),
            4 => Ok(April),
            5 => Ok(May),
            6 => Ok(June),
            7 => Ok(July),
            8 => Ok(August),
            9 => Ok(September),
            10 => Ok(October),
            11 => Ok(November),
            12 => Ok(December),
            _ => Err(ComponentRange::unconditional("month")),
        }
    }

    /// Get the number of days in the month of a given year.
    #[inline]
    pub const fn length(self, year: i32) -> u8 {
        days_in_month(self as u8, year)
    }

    /// Get the previous month.
    #[inline]
    pub const fn previous(self) -> Self {
        match self {
            January => December,
            February => January,
            March => February,
            April => March,
            May => April,
            June => May,
            July => June,
            August => July,
            September => August,
            October => September,
            November => October,
            December => November,
        }
    }

    /// Get the next month.
    #[inline]
    pub const fn next(self) -> Self {
        match self {
            January => February,
            February => March,
            March => April,
            April => May,
            May => June,
            June => July,
            July => August,
            August => September,
            September => October,
            October => November,
            November => December,
            December => January,
        }
    }

    /// Get n-th next month.
    #[inline]
    pub const fn nth_next(self, n: u8) -> Self {
        match (self as u8 - 1 + n % 12) % 12 {
            0 => January,
            1 => February,
            2 => March,
            3 => April,
            4 => May,
            5 => June,
            6 => July,
            7 => August,
            8 => September,
            9 => October,
            10 => November,
            val => {
                debug_assert!(val == 11);
                December
            }
        }
    }

    /// Get n-th previous month.
    #[inline]
    pub const fn nth_prev(self, n: u8) -> Self {
        match self as i8 - 1 - (n % 12).cast_signed() {
            1 | -11 => February,
            2 | -10 => March,
            3 | -9 => April,
            4 | -8 => May,
            5 | -7 => June,
            6 | -6 => July,
            7 | -5 => August,
            8 | -4 => September,
            9 | -3 => October,
            10 | -2 => November,
            11 | -1 => December,
            val => {
                debug_assert!(val == 0);
                January
            }
        }
    }
}

impl SmartDisplay for Month {
    type Metadata = ();

    #[inline]
    fn metadata(&self, _: FormatterOptions) -> Metadata<'_, Self> {
        match self {
            January => Metadata::new(7, self, ()),
            February => Metadata::new(8, self, ()),
            March => Metadata::new(5, self, ()),
            April => Metadata::new(5, self, ()),
            May => Metadata::new(3, self, ()),
            June => Metadata::new(4, self, ()),
            July => Metadata::new(4, self, ()),
            August => Metadata::new(6, self, ()),
            September => Metadata::new(9, self, ()),
            October => Metadata::new(7, self, ()),
            November => Metadata::new(8, self, ()),
            December => Metadata::new(8, self, ()),
        }
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            January => "January",
            February => "February",
            March => "March",
            April => "April",
            May => "May",
            June => "June",
            July => "July",
            August => "August",
            September => "September",
            October => "October",
            November => "November",
            December => "December",
        })
    }
}

impl fmt::Display for Month {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        SmartDisplay::fmt(self, f)
    }
}

impl FromStr for Month {
    type Err = InvalidVariant;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "January" => Ok(January),
            "February" => Ok(February),
            "March" => Ok(March),
            "April" => Ok(April),
            "May" => Ok(May),
            "June" => Ok(June),
            "July" => Ok(July),
            "August" => Ok(August),
            "September" => Ok(September),
            "October" => Ok(October),
            "November" => Ok(November),
            "December" => Ok(December),
            _ => Err(InvalidVariant),
        }
    }
}

impl From<Month> for u8 {
    #[inline]
    fn from(month: Month) -> Self {
        month as Self
    }
}

impl TryFrom<u8> for Month {
    type Error = ComponentRange;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match NonZero::new(value) {
            Some(value) => Self::from_number(value),
            None => Err(ComponentRange::unconditional("month")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_MONTHS: [Month; 12] = [
        January, February, March, April, May, June, July, August, September, October, November, December,
    ];

    const NAMES: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];

    #[test]
    fn from_number_round_trips_all_months() {
        for (index, &month) in ALL_MONTHS.iter().enumerate() {
            let number = NonZero::new(index as u8 + 1).expect("month numbers are non-zero");
            assert_eq!(Month::from_number(number), Ok(month));
            assert_eq!(u8::from(month), index as u8 + 1);
            assert_eq!(Month::try_from(index as u8 + 1), Ok(month));
        }
    }

    #[test]
    fn from_number_rejects_out_of_range_values() {
        for value in [13u8, 14, 100, 255] {
            let number = NonZero::new(value).expect("test inputs are non-zero");
            let error = Month::from_number(number).expect_err("should be rejected");
            assert_eq!(error, ComponentRange::unconditional("month"));
            assert_eq!(Month::try_from(value).expect_err("should be rejected"), error);
            assert!(!error.is_conditional());
        }
        assert_eq!(
            Month::try_from(0)
                .expect_err("zero is not a month")
                .name(),
            "month"
        );
    }

    #[test]
    fn parse_and_display_round_trip_all_months() {
        for (index, &month) in ALL_MONTHS.iter().enumerate() {
            assert_eq!(month.to_string(), NAMES[index]);
            assert_eq!(NAMES[index].parse::<Month>(), Ok(month));
            // Parsing is exact: no surrounding whitespace or case changes are accepted.
            assert!(
                format!(" {}", NAMES[index])
                    .parse::<Month>()
                    .is_err()
            );
            assert!(
                NAMES[index]
                    .to_lowercase()
                    .parse::<Month>()
                    .is_err()
            );
        }
    }

    #[test]
    fn parse_rejects_invalid_input() {
        for input in ["", "january", "Januaryy", "1", "Jan", "Sunday", "  "] {
            assert_eq!(
                input.parse::<Month>(),
                Err(InvalidVariant),
                "unexpectedly accepted {input:?}",
            );
        }
        assert_eq!(InvalidVariant.to_string(), "value was not a valid variant");
    }

    #[test]
    fn display_respects_formatting_flags() {
        assert_eq!(format!("{:<10}", January), "January   ");
        assert_eq!(format!("{:>10}", May), "       May");
        assert_eq!(format!("{:^10}", May), "   May    ");
    }

    #[test]
    fn previous_and_next_wrap_around_the_year() {
        for (index, &month) in ALL_MONTHS.iter().enumerate() {
            let expected_prev = ALL_MONTHS[(index + 11) % 12];
            let expected_next = ALL_MONTHS[(index + 1) % 12];
            assert_eq!(month.previous(), expected_prev, "previous of {month:?}");
            assert_eq!(month.next(), expected_next, "next of {month:?}");
        }
        assert_eq!(January.previous(), December);
        assert_eq!(December.next(), January);
        assert_eq!(January.previous().next(), January);
        assert_eq!(December.next().previous(), December);
    }

    #[test]
    fn nth_next_and_nth_prev_wrap_correctly() {
        for (index, &month) in ALL_MONTHS.iter().enumerate() {
            for n in 0u8..=30 {
                let expected_next = ALL_MONTHS[(index + n as usize) % 12];
                let expected_prev = ALL_MONTHS[(index + 12 - (n as usize % 12)) % 12];
                assert_eq!(month.nth_next(n), expected_next, "{month:?}.nth_next({n})");
                assert_eq!(month.nth_prev(n), expected_prev, "{month:?}.nth_prev({n})");
            }
            // A full rotation returns the same month, in both directions and for multiples
            // of the year length.
            assert_eq!(month.nth_next(12), month);
            assert_eq!(month.nth_prev(12), month);
            assert_eq!(month.nth_next(24), month);
            assert_eq!(month.nth_prev(0), month);
            assert_eq!(month.nth_next(0), month);
        }
    }

    #[test]
    fn length_reflects_leap_years_for_february() {
        assert_eq!(February.length(2024), 29);
        assert_eq!(February.length(2000), 29);
        assert_eq!(February.length(2023), 28);
        assert_eq!(February.length(1900), 28);
        for (index, &month) in ALL_MONTHS.iter().enumerate() {
            let expected = if index == 1 { 28 } else { COMMON_LENGTHS[index] };
            assert_eq!(month.length(2023), expected, "common year {month:?}");
            let expected = if index == 1 { 29 } else { COMMON_LENGTHS[index] };
            assert_eq!(month.length(2024), expected, "leap year {month:?}");
        }
    }

    const COMMON_LENGTHS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    #[test]
    fn months_are_ordered_january_to_december() {
        for window in ALL_MONTHS.windows(2) {
            assert!(window[0] < window[1]);
        }
        assert!(January < December);
        assert_eq!(January, January);
        assert_ne!(January, February);
    }

    #[test]
    fn from_str_and_u8_conversions_are_consistent() {
        for (index, &month) in ALL_MONTHS.iter().enumerate() {
            let number = index as u8 + 1;
            assert_eq!(month.to_string().parse::<Month>().map(u8::from), Ok(number));
        }
    }
}
