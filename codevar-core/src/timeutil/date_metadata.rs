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

use crate::timeutil::date_adt_hack::EncodedConfig;
use crate::timeutil::date_format_description::{
    Component, FormatDescription, FormatDescriptionInner,
};
use crate::timeutil::date_format_description_modifier::SubsecondDigits;
use crate::timeutil::date_well_know_iso8601::{
    DateKind, Iso8601, OffsetPrecision, TimePrecision,
};
use crate::timeutil::date_well_know_rfc2822::Rfc2822;
use crate::timeutil::date_well_know_rfc3339::Rfc3339;
use core::iter::Sum;
use core::ops::{Add, Deref};

/// Metadata about a format description.
#[derive(Debug)]
pub(crate) struct Metadata {
    /// The maximum number of bytes needed for the provided format description.
    ///
    /// The number of bytes written should never exceed this value, but it may be less. This is
    /// used to pre-allocate a buffer of the appropriate size for formatting.
    pub(crate) max_bytes_needed: usize,
    /// Whether the output of the provided format description is guaranteed to be valid UTF-8.
    ///
    /// This is used to determine whether the output can be soundly converted to a `String` without
    /// checking for UTF-8 validity.
    pub(crate) guaranteed_utf8: bool,
}

impl Default for Metadata {
    #[inline]
    fn default() -> Self {
        Self {
            max_bytes_needed: 0,
            guaranteed_utf8: true,
        }
    }
}

impl Add for Metadata {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            max_bytes_needed: self.max_bytes_needed + rhs.max_bytes_needed,
            guaranteed_utf8: self.guaranteed_utf8 && rhs.guaranteed_utf8,
        }
    }
}

impl Sum for Metadata {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), Self::add)
    }
}

/// A trait for computing metadata about a format description.
pub(crate) trait ComputeMetadata {
    /// Compute the metadata for a format description.
    fn compute_metadata(&self) -> Metadata;
}

impl ComputeMetadata for Rfc2822 {
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        Metadata {
            max_bytes_needed: 31,
            guaranteed_utf8: true,
        }
    }
}

impl ComputeMetadata for Rfc3339 {
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        Metadata {
            max_bytes_needed: 35,
            guaranteed_utf8: true,
        }
    }
}

impl<const CONFIG: EncodedConfig> ComputeMetadata for Iso8601<CONFIG> {
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        const {
            let date_width = if Self::FORMAT_DATE {
                let year_width = if Self::YEAR_IS_SIX_DIGITS {
                    7 // sign + 6 digits
                } else {
                    4 // sign is not present when the year is four digits
                };
                let num_dashes = match Self::DATE_KIND {
                    DateKind::Calendar if Self::USE_SEPARATORS => 2,
                    DateKind::Week | DateKind::Ordinal if Self::USE_SEPARATORS => 1,
                    DateKind::Calendar | DateKind::Week | DateKind::Ordinal => 0,
                };
                let part_of_year_width = match Self::DATE_KIND {
                    DateKind::Calendar => 4,
                    DateKind::Week => 4,
                    DateKind::Ordinal => 3,
                };

                year_width + num_dashes + part_of_year_width
            } else {
                0
            };

            let time_width = if Self::FORMAT_TIME {
                let t_separator = (Self::USE_SEPARATORS || Self::FORMAT_DATE) as usize;
                let num_colons = match Self::TIME_PRECISION {
                    TimePrecision::Minute { .. } if Self::USE_SEPARATORS => 1,
                    TimePrecision::Second { .. } if Self::USE_SEPARATORS => 2,
                    TimePrecision::Hour { .. }
                    | TimePrecision::Minute { .. }
                    | TimePrecision::Second { .. } => 0,
                };
                let pre_decimal_digits = match Self::TIME_PRECISION {
                    TimePrecision::Hour { .. } => 2,
                    TimePrecision::Minute { .. } => 4,
                    TimePrecision::Second { .. } => 6,
                };
                let fractional_bytes = match Self::TIME_PRECISION {
                    TimePrecision::Hour { decimal_digits }
                    | TimePrecision::Minute { decimal_digits }
                    | TimePrecision::Second { decimal_digits } => {
                        if let Some(digits) = decimal_digits {
                            // add one for decimal point
                            1 + digits.get() as usize
                        } else {
                            0
                        }
                    }
                };

                t_separator + num_colons + pre_decimal_digits + fractional_bytes
            } else {
                0
            };

            let offset_width = if Self::FORMAT_OFFSET {
                match Self::OFFSET_PRECISION {
                    OffsetPrecision::Hour => 3,
                    OffsetPrecision::Minute if Self::USE_SEPARATORS => 6,
                    OffsetPrecision::Minute => 5,
                }
            } else {
                0
            };

            Metadata {
                max_bytes_needed: date_width + time_width + offset_width,
                guaranteed_utf8: true,
            }
        }
    }
}

impl ComputeMetadata for FormatDescription<'_> {
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        Metadata {
            max_bytes_needed: self.max_bytes_needed,
            guaranteed_utf8: true,
        }
    }
}

impl ComputeMetadata for Component {
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        let max_bytes_needed = match self {
            Self::Day(_) => 2,
            Self::MonthShort(_) => 3,
            Self::MonthLong(_) => 9,
            Self::MonthNumerical(_) => 2,
            Self::Ordinal(_) => 3,
            Self::WeekdayShort(_) => 3,
            Self::WeekdayLong(_) => 9,
            Self::WeekdaySunday(_) | Self::WeekdayMonday(_) => 1,
            Self::WeekNumberIso(_)
            | Self::WeekNumberSunday(_)
            | Self::WeekNumberMonday(_) => 2,
            Self::CalendarYearFullExtendedRange(_) => 7,
            Self::CalendarYearFullStandardRange(_) => 5,
            Self::IsoYearFullExtendedRange(_) => 7,
            Self::IsoYearFullStandardRange(_) => 5,
            Self::CalendarYearCenturyExtendedRange(_) => 5,
            Self::CalendarYearCenturyStandardRange(_) => 3,
            Self::IsoYearCenturyExtendedRange(_) => 5,
            Self::IsoYearCenturyStandardRange(_) => 3,
            Self::CalendarYearLastTwo(_) => 2,
            Self::IsoYearLastTwo(_) => 2,
            Self::Hour12(_) | Self::Hour24(_) => 2,
            Self::Minute(_) | Self::Period(_) | Self::Second(_) => 2,
            Self::Subsecond(modifier) => match modifier.digits {
                SubsecondDigits::One => 1,
                SubsecondDigits::Two => 2,
                SubsecondDigits::Three => 3,
                SubsecondDigits::Four => 4,
                SubsecondDigits::Five => 5,
                SubsecondDigits::Six => 6,
                SubsecondDigits::Seven => 7,
                SubsecondDigits::Eight => 8,
                SubsecondDigits::Nine => 9,
                SubsecondDigits::OneOrMore => 9,
            },
            Self::OffsetHour(_) => 3,
            Self::OffsetMinute(_) | Self::OffsetSecond(_) => 2,
            Self::UnixTimestampSecond(_) => 15,
            Self::UnixTimestampMillisecond(_) => 18,
            Self::UnixTimestampMicrosecond(_) => 21,
            Self::UnixTimestampNanosecond(_) => 24,
            Self::Ignore(_) | Self::End(_) => 0,
        };

        Metadata {
            max_bytes_needed,
            guaranteed_utf8: true,
        }
    }
}

impl<T> ComputeMetadata for [T]
where
    T: ComputeMetadata,
{
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        self.iter().map(|item| item.compute_metadata()).sum()
    }
}

impl<T> ComputeMetadata for T
where
    T: Deref<Target: ComputeMetadata>,
{
    #[inline]
    fn compute_metadata(&self) -> Metadata {
        self.deref().compute_metadata()
    }
}
