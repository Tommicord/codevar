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

use core::fmt;
use core::str::FromStr;

use self::Weekday::*;
use crate::date_error::InvalidVariant;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

/// Days of the week.
///
/// As order is dependent on context (Sunday could be either two days after or five days before
/// Friday), this type does not implement `PartialOrd` or `Ord`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weekday {
    #[allow(missing_docs)]
    Monday,
    #[allow(missing_docs)]
    Tuesday,
    #[allow(missing_docs)]
    Wednesday,
    #[allow(missing_docs)]
    Thursday,
    #[allow(missing_docs)]
    Friday,
    #[allow(missing_docs)]
    Saturday,
    #[allow(missing_docs)]
    Sunday,
}

impl Weekday {
    /// Get the previous weekday.
    #[inline]
    pub const fn previous(self) -> Self {
        match self {
            Monday => Sunday,
            Tuesday => Monday,
            Wednesday => Tuesday,
            Thursday => Wednesday,
            Friday => Thursday,
            Saturday => Friday,
            Sunday => Saturday,
        }
    }

    /// Get the next weekday.
    #[inline]
    pub const fn next(self) -> Self {
        match self {
            Monday => Tuesday,
            Tuesday => Wednesday,
            Wednesday => Thursday,
            Thursday => Friday,
            Friday => Saturday,
            Saturday => Sunday,
            Sunday => Monday,
        }
    }

    /// Get n-th next day.
    #[inline]
    pub const fn nth_next(self, n: u8) -> Self {
        match (self.number_days_from_monday() + n % 7) % 7 {
            0 => Monday,
            1 => Tuesday,
            2 => Wednesday,
            3 => Thursday,
            4 => Friday,
            5 => Saturday,
            val => {
                debug_assert!(val == 6);
                Sunday
            }
        }
    }

    /// Get n-th previous day.
    #[inline]
    pub const fn nth_prev(self, n: u8) -> Self {
        match self.number_days_from_monday().cast_signed() - (n % 7).cast_signed() {
            1 | -6 => Tuesday,
            2 | -5 => Wednesday,
            3 | -4 => Thursday,
            4 | -3 => Friday,
            5 | -2 => Saturday,
            6 | -1 => Sunday,
            val => {
                debug_assert!(val == 0);
                Monday
            }
        }
    }

    /// Get the one-indexed number of days from Monday.
    #[doc(alias = "iso_weekday_number")]
    #[inline]
    pub const fn number_from_monday(self) -> u8 {
        self.number_days_from_monday() + 1
    }

    /// Get the one-indexed number of days from Sunday.
    #[inline]
    pub const fn number_from_sunday(self) -> u8 {
        self.number_days_from_sunday() + 1
    }

    /// Get the zero-indexed number of days from Monday.
    #[inline]
    pub const fn number_days_from_monday(self) -> u8 {
        self as u8
    }

    /// Get the zero-indexed number of days from Sunday.
    #[inline]
    pub const fn number_days_from_sunday(self) -> u8 {
        match self {
            Monday => 1,
            Tuesday => 2,
            Wednesday => 3,
            Thursday => 4,
            Friday => 5,
            Saturday => 6,
            Sunday => 0,
        }
    }
}

impl SmartDisplay for Weekday {
    type Metadata = ();

    #[inline]
    fn metadata(&self, _: FormatterOptions) -> Metadata<'_, Self> {
        match self {
            Monday => Metadata::new(6, self, ()),
            Tuesday => Metadata::new(7, self, ()),
            Wednesday => Metadata::new(9, self, ()),
            Thursday => Metadata::new(8, self, ()),
            Friday => Metadata::new(6, self, ()),
            Saturday => Metadata::new(8, self, ()),
            Sunday => Metadata::new(6, self, ()),
        }
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Monday => "Monday",
            Tuesday => "Tuesday",
            Wednesday => "Wednesday",
            Thursday => "Thursday",
            Friday => "Friday",
            Saturday => "Saturday",
            Sunday => "Sunday",
        })
    }
}

impl fmt::Display for Weekday {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        SmartDisplay::fmt(self, f)
    }
}

impl FromStr for Weekday {
    type Err = InvalidVariant;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Monday" => Ok(Monday),
            "Tuesday" => Ok(Tuesday),
            "Wednesday" => Ok(Wednesday),
            "Thursday" => Ok(Thursday),
            "Friday" => Ok(Friday),
            "Saturday" => Ok(Saturday),
            "Sunday" => Ok(Sunday),
            _ => Err(InvalidVariant),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_DAYS: [Weekday; 7] = [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday];

    const NAMES: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];

    #[test]
    fn parse_and_display_round_trip_all_days() {
        for (index, &day) in ALL_DAYS.iter().enumerate() {
            assert_eq!(day.to_string(), NAMES[index]);
            assert_eq!(NAMES[index].parse::<Weekday>(), Ok(day));
            // Matching is exact.
            assert!(
                format!(" {}", NAMES[index])
                    .parse::<Weekday>()
                    .is_err()
            );
            assert!(
                NAMES[index]
                    .to_lowercase()
                    .parse::<Weekday>()
                    .is_err()
            );
        }
    }

    #[test]
    fn parse_rejects_invalid_input() {
        for input in ["", "monday", "Mondayy", "Mon", "1", "January", "  Monday"] {
            assert_eq!(
                input.parse::<Weekday>(),
                Err(InvalidVariant),
                "unexpectedly accepted {input:?}",
            );
        }
    }

    #[test]
    fn previous_and_next_wrap_around_the_week() {
        for (index, &day) in ALL_DAYS.iter().enumerate() {
            let expected_prev = ALL_DAYS[(index + 6) % 7];
            let expected_next = ALL_DAYS[(index + 1) % 7];
            assert_eq!(day.previous(), expected_prev, "previous of {day:?}");
            assert_eq!(day.next(), expected_next, "next of {day:?}");
        }
        assert_eq!(Monday.previous(), Sunday);
        assert_eq!(Sunday.next(), Monday);
        assert_eq!(Monday.previous().next(), Monday);
        assert_eq!(Sunday.next().previous(), Sunday);
    }

    #[test]
    fn nth_next_and_nth_prev_cover_all_offsets() {
        for (index, &day) in ALL_DAYS.iter().enumerate() {
            for n in 0u8..=100 {
                let expected_next = ALL_DAYS[(index + (n % 7) as usize) % 7];
                let expected_prev = ALL_DAYS[(index + 7 - (n % 7) as usize) % 7 % 7];
                assert_eq!(day.nth_next(n), expected_next, "{day:?}.nth_next({n})");
                assert_eq!(day.nth_prev(n), expected_prev, "{day:?}.nth_prev({n})");
            }
            assert_eq!(day.nth_next(7), day);
            assert_eq!(day.nth_prev(7), day);
            assert_eq!(day.nth_next(0), day);
            assert_eq!(day.nth_prev(0), day);
            // nth_next and nth_prev are inverse operations.
            assert_eq!(day.nth_next(5).nth_prev(5), day);
        }
    }

    #[test]
    fn weekday_numbers_are_consistent() {
        for (index, &day) in ALL_DAYS.iter().enumerate() {
            assert_eq!(day.number_days_from_monday(), index as u8);
            assert_eq!(day.number_from_monday(), index as u8 + 1);
            // Sunday is day 0 of the Sunday-based count, so Monday (= index 0) is 1 and
            // Saturday (= index 5) is 6.
            assert_eq!(day.number_days_from_sunday(), (index as u8 + 1) % 7);
            assert_eq!(day.number_from_sunday(), (index as u8 + 1) % 7 + 1);
            // The Sunday-based one-indexed number of a day is the Monday-based one-indexed
            // number of the following day.
            assert_eq!(day.number_from_sunday(), day.next().number_from_monday());
        }
        assert_eq!(Monday.number_from_monday(), 1);
        assert_eq!(Sunday.number_from_monday(), 7);
        assert_eq!(Sunday.number_from_sunday(), 1);
        assert_eq!(Monday.number_from_sunday(), 2);
        assert_eq!(Saturday.number_from_sunday(), 7);
        assert_eq!(Saturday.number_from_monday(), 6);
    }

    #[test]
    fn metadata_widths_match_displayed_names() {
        for (index, &day) in ALL_DAYS.iter().enumerate() {
            let metadata = day.metadata(FormatterOptions::default());
            assert_eq!(metadata.unpadded_width(), NAMES[index].len());
        }
    }

    #[test]
    fn display_respects_formatting_flags() {
        assert_eq!(format!("{:<10}", Wednesday), "Wednesday ");
        assert_eq!(format!("{:>10}", Monday), "    Monday");
        assert_eq!(format!("{:^9}", Friday), " Friday  ");
        assert_eq!(format!("{:.*}", 3, Wednesday), "Wed");
    }
}
