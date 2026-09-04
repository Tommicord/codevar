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

//! The `Month` enum and its associated `impl`s.

use core::fmt;
use core::num::NonZero;
use core::str::FromStr;

use self::Month::*;
use crate::timeutil::date_error::{ComponentRange, InvalidVariant};
use crate::timeutil::date_util::days_in_month;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

/// Months of the year.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Month {
    #[expect(missing_docs)]
    January = 1,
    #[expect(missing_docs)]
    February = 2,
    #[expect(missing_docs)]
    March = 3,
    #[expect(missing_docs)]
    April = 4,
    #[expect(missing_docs)]
    May = 5,
    #[expect(missing_docs)]
    June = 6,
    #[expect(missing_docs)]
    July = 7,
    #[expect(missing_docs)]
    August = 8,
    #[expect(missing_docs)]
    September = 9,
    #[expect(missing_docs)]
    October = 10,
    #[expect(missing_docs)]
    November = 11,
    #[expect(missing_docs)]
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
