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

use crate::timeutil::date_format_description_modifier::*;
#[cfg(feature = "alloc")]
use alloc::boxed::Box;
use core::fmt;
use num_conv::Truncate;

/// A complete description of how to format and parse a type.
///
/// Both for forwards compatibility and to enable optimizations
#[derive(Clone)]
pub struct FormatDescription<'a> {
    /// The inner `enum` that controls all business logic.
    pub(crate) inner: FormatDescriptionInner<'a>,
    /// The maximum number of bytes that are needed to format any value using this format
    /// description.
    pub(crate) max_bytes_needed: usize,
}

impl fmt::Debug for FormatDescription<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl FormatDescription<'_> {
    /// Convert the format description to an owned version, enabling it to be stored without regard
    /// for lifetime.
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn to_owned(self) -> FormatDescription<'static> {
        FormatDescription {
            inner: self.inner.into_owned(),
            #[cfg(feature = "formatting")]
            max_bytes_needed: self.max_bytes_needed,
        }
    }
}

#[non_exhaustive]
#[derive(Clone)]
pub enum FormatDescriptionInner<'a> {
    /// Day of the month.
    Day(Day),
    /// Month of the year in the abbreviated form (e.g. "Jan").
    MonthShort(MonthShort),
    /// Month of the year in the full form (e.g. "January").
    MonthLong(MonthLong),
    /// Month of the year in the numerical form (e.g. "1" for January).
    MonthNumerical(MonthNumerical),
    /// Ordinal day of the year.
    Ordinal(Ordinal),
    /// Weekday in the abbreviated form (e.g. "Mon").
    WeekdayShort(WeekdayShort),
    /// Weekday in the full form (e.g. "Monday").
    WeekdayLong(WeekdayLong),
    /// Weekday number where Sunday is either 0 or 1 depending on the modifier.
    WeekdaySunday(WeekdaySunday),
    /// Weekday number where Monday is either 0 or 1 depending on the modifier.
    WeekdayMonday(WeekdayMonday),
    /// Week number of the year, where week 1 starts is the week beginning on Monday that contains
    /// January 4.
    WeekNumberIso(WeekNumberIso),
    /// Week number of the year, where week 1 starts on the first Sunday of the calendar year.
    WeekNumberSunday(WeekNumberSunday),
    /// Week number of the year, where week 1 starts on the first Monday of the calendar year.
    WeekNumberMonday(WeekNumberMonday),
    /// The calendar year. Supports the extended range.
    CalendarYearFullExtendedRange(CalendarYearFullExtendedRange),
    /// The calendar year. Does not support the extended range.
    CalendarYearFullStandardRange(CalendarYearFullStandardRange),
    /// The ISO week-based year. Supports the extended range.
    IsoYearFullExtendedRange(IsoYearFullExtendedRange),
    /// The ISO week-based year. Does not support the extended range.
    IsoYearFullStandardRange(IsoYearFullStandardRange),
    /// The century of the calendar year. Supports the extended range.
    CalendarYearCenturyExtendedRange(CalendarYearCenturyExtendedRange),
    /// The century of the calendar year. Does not support the extended range.
    CalendarYearCenturyStandardRange(CalendarYearCenturyStandardRange),
    /// The century of the ISO week-based year. Supports the extended range.
    IsoYearCenturyExtendedRange(IsoYearCenturyExtendedRange),
    /// The century of the ISO week-based year. Does not support the extended range.
    IsoYearCenturyStandardRange(IsoYearCenturyStandardRange),
    /// The last two digits of the calendar year.
    CalendarYearLastTwo(CalendarYearLastTwo),
    /// The last two digits of the ISO week-based year.
    IsoYearLastTwo(IsoYearLastTwo),
    /// Hour of the day using the 12-hour clock.
    Hour12(Hour12),
    /// Hour of the day using the 24-hour clock.
    Hour24(Hour24),
    /// Minute within the hour.
    Minute(Minute),
    /// AM/PM part of the time.
    Period(Period),
    /// Second within the minute.
    Second(Second),
    /// Subsecond within the second.
    Subsecond(Subsecond),
    /// Hour of the UTC offset.
    OffsetHour(OffsetHour),
    /// Minute within the hour of the UTC offset.
    OffsetMinute(OffsetMinute),
    /// Second within the minute of the UTC offset.
    OffsetSecond(OffsetSecond),
    /// A number of bytes to ignore when parsing. This has no effect on formatting.
    Ignore(Ignore),
    /// A Unix timestamp in seconds.
    UnixTimestampSecond(UnixTimestampSecond),
    /// A Unix timestamp in milliseconds.
    UnixTimestampMillisecond(UnixTimestampMillisecond),
    /// A Unix timestamp in microseconds.
    UnixTimestampMicrosecond(UnixTimestampMicrosecond),
    /// A Unix timestamp in nanoseconds.
    UnixTimestampNanosecond(UnixTimestampNanosecond),
    /// The end of input. Parsing this component will fail if there is any input remaining. This
    /// component neither affects formatting nor consumes any input when parsing.
    End(End),
    /// A string that is formatted as-is.
    BorrowedLiteral(&'a str),
    /// A series of literals or components that collectively form a partial or complete description.
    BorrowedCompound(&'a [Self]),
    /// An item that may or may not be present when parsing. If parsing fails, there will be no
    /// effect on the resulting `struct`.
    BorrowedOptional {
        /// Whether the item should be formatted.
        format: bool,
        /// The item in question.
        item: &'a Self,
    },
    /// A series of items where, when parsing, the first successful parse is used. When formatting,
    /// the first item is used. If no items are present, both formatting and parsing are no-ops.
    BorrowedFirst(&'a [Self]),
    /// A string that is formatted as-is.
    OwnedLiteral(Box<str>),
    /// A series of literals or components that collectively form a partial or complete description.
    OwnedCompound(Box<[Self]>),
    /// An item that may or may not be present when parsing. If parsing fails, there will be no
    /// effect on the resulting `struct`.
    OwnedOptional {
        /// Whether the item should be formatted.
        format: bool,
        /// The item in question.
        item: Box<Self>,
    },
    /// A series of items where, when parsing, the first successful parse is used. When formatting,
    /// the first item is used. If no items are present, both formatting and parsing are no-ops.
    OwnedFirst(Box<[Self]>),
}

impl fmt::Debug for FormatDescriptionInner<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Day(modifier) => modifier.fmt(f),
            Self::MonthShort(modifier) => modifier.fmt(f),
            Self::MonthLong(modifier) => modifier.fmt(f),
            Self::MonthNumerical(modifier) => modifier.fmt(f),
            Self::Ordinal(modifier) => modifier.fmt(f),
            Self::WeekdayShort(modifier) => modifier.fmt(f),
            Self::WeekdayLong(modifier) => modifier.fmt(f),
            Self::WeekdaySunday(modifier) => modifier.fmt(f),
            Self::WeekdayMonday(modifier) => modifier.fmt(f),
            Self::WeekNumberIso(modifier) => modifier.fmt(f),
            Self::WeekNumberSunday(modifier) => modifier.fmt(f),
            Self::WeekNumberMonday(modifier) => modifier.fmt(f),
            Self::CalendarYearFullExtendedRange(modifier) => modifier.fmt(f),
            Self::CalendarYearFullStandardRange(modifier) => modifier.fmt(f),
            Self::IsoYearFullExtendedRange(modifier) => modifier.fmt(f),
            Self::IsoYearFullStandardRange(modifier) => modifier.fmt(f),
            Self::CalendarYearCenturyExtendedRange(modifier) => modifier.fmt(f),
            Self::CalendarYearCenturyStandardRange(modifier) => modifier.fmt(f),
            Self::IsoYearCenturyExtendedRange(modifier) => modifier.fmt(f),
            Self::IsoYearCenturyStandardRange(modifier) => modifier.fmt(f),
            Self::CalendarYearLastTwo(modifier) => modifier.fmt(f),
            Self::IsoYearLastTwo(modifier) => modifier.fmt(f),
            Self::Hour12(modifier) => modifier.fmt(f),
            Self::Hour24(modifier) => modifier.fmt(f),
            Self::Minute(modifier) => modifier.fmt(f),
            Self::Period(modifier) => modifier.fmt(f),
            Self::Second(modifier) => modifier.fmt(f),
            Self::Subsecond(modifier) => modifier.fmt(f),
            Self::OffsetHour(modifier) => modifier.fmt(f),
            Self::OffsetMinute(modifier) => modifier.fmt(f),
            Self::OffsetSecond(modifier) => modifier.fmt(f),
            Self::Ignore(modifier) => modifier.fmt(f),
            Self::UnixTimestampSecond(modifier) => modifier.fmt(f),
            Self::UnixTimestampMillisecond(modifier) => modifier.fmt(f),
            Self::UnixTimestampMicrosecond(modifier) => modifier.fmt(f),
            Self::UnixTimestampNanosecond(modifier) => modifier.fmt(f),
            Self::End(modifier) => modifier.fmt(f),
            Self::BorrowedLiteral(literal) => {
                f.debug_tuple("Literal").field(literal).finish()
            }
            Self::BorrowedCompound(compound) => {
                f.debug_tuple("Compound").field(compound).finish()
            }
            Self::BorrowedOptional {
                format: should_format,
                item,
            } => f
                .debug_struct("Optional")
                .field("should_format", should_format)
                .field("item", item)
                .finish(),
            Self::BorrowedFirst(items) => f.debug_tuple("First").field(items).finish(),
            Self::OwnedLiteral(literal) => {
                f.debug_tuple("Literal").field(literal).finish()
            }
            Self::OwnedCompound(compound) => {
                f.debug_tuple("Compound").field(compound).finish()
            }
            Self::OwnedOptional {
                format: should_format,
                item,
            } => f
                .debug_struct("Optional")
                .field("should_format", should_format)
                .field("item", item)
                .finish(),
            Self::OwnedFirst(items) => f.debug_tuple("First").field(items).finish(),
        }
    }
}

impl<'a> FormatDescriptionInner<'a> {
    /// Recursively convert to an owned version, doing so in-place when possible.
    fn make_owned_in_place(&mut self) {
        use alloc::borrow::ToOwned as _;
        use alloc::boxed::Box;

        match self {
            Self::BorrowedLiteral(literal) => {
                *self = Self::OwnedLiteral(literal.to_owned().into_boxed_str());
            }
            Self::BorrowedCompound(compound) => {
                *self = Self::OwnedCompound(
                    compound
                        .iter()
                        .cloned()
                        .map(|item| item.into_owned())
                        .collect(),
                );
            }
            Self::BorrowedOptional { format, item } => {
                *self = Self::OwnedOptional {
                    format: *format,
                    item: Box::new(item.clone().into_owned()),
                };
            }
            Self::BorrowedFirst(items) => {
                *self = Self::OwnedFirst(
                    items
                        .iter()
                        .cloned()
                        .map(|item| item.into_owned())
                        .collect(),
                );
            }
            Self::OwnedCompound(compound) => {
                for item in compound {
                    item.make_owned_in_place();
                }
            }
            Self::OwnedOptional { format: _, item } => {
                item.make_owned_in_place();
            }
            Self::OwnedFirst(items) => {
                for item in items {
                    item.make_owned_in_place();
                }
            }
            FormatDescriptionInner::Day(_)
            | FormatDescriptionInner::MonthShort(_)
            | FormatDescriptionInner::MonthLong(_)
            | FormatDescriptionInner::MonthNumerical(_)
            | FormatDescriptionInner::Ordinal(_)
            | FormatDescriptionInner::WeekdayShort(_)
            | FormatDescriptionInner::WeekdayLong(_)
            | FormatDescriptionInner::WeekdaySunday(_)
            | FormatDescriptionInner::WeekdayMonday(_)
            | FormatDescriptionInner::WeekNumberIso(_)
            | FormatDescriptionInner::WeekNumberSunday(_)
            | FormatDescriptionInner::WeekNumberMonday(_)
            | FormatDescriptionInner::CalendarYearFullExtendedRange(_)
            | FormatDescriptionInner::CalendarYearFullStandardRange(_)
            | FormatDescriptionInner::IsoYearFullExtendedRange(_)
            | FormatDescriptionInner::IsoYearFullStandardRange(_)
            | FormatDescriptionInner::CalendarYearCenturyExtendedRange(_)
            | FormatDescriptionInner::CalendarYearCenturyStandardRange(_)
            | FormatDescriptionInner::IsoYearCenturyExtendedRange(_)
            | FormatDescriptionInner::IsoYearCenturyStandardRange(_)
            | FormatDescriptionInner::CalendarYearLastTwo(_)
            | FormatDescriptionInner::IsoYearLastTwo(_)
            | FormatDescriptionInner::Hour12(_)
            | FormatDescriptionInner::Hour24(_)
            | FormatDescriptionInner::Minute(_)
            | FormatDescriptionInner::Period(_)
            | FormatDescriptionInner::Second(_)
            | FormatDescriptionInner::Subsecond(_)
            | FormatDescriptionInner::OffsetHour(_)
            | FormatDescriptionInner::OffsetMinute(_)
            | FormatDescriptionInner::OffsetSecond(_)
            | FormatDescriptionInner::Ignore(_)
            | FormatDescriptionInner::UnixTimestampSecond(_)
            | FormatDescriptionInner::UnixTimestampMillisecond(_)
            | FormatDescriptionInner::UnixTimestampMicrosecond(_)
            | FormatDescriptionInner::UnixTimestampNanosecond(_)
            | FormatDescriptionInner::End(_)
            | FormatDescriptionInner::OwnedLiteral(_) => {
                // no-op, as these variants do not contain any references
            }
        }
    }

    /// Convert the format description to an owned version in place, replacing borrowed
    /// components with their owned equivalents.
    pub(super) fn into_owned(mut self) -> FormatDescriptionInner<'static> {
        self.make_owned_in_place();

        // Safety: `make_owned_in_place` recursively eliminates all variants that contain
        // references, so we can transmute between lifetimes freely. ADTs do not vary in layout when
        // only the lifetime differs.
        unsafe {
            core::mem::transmute::<
                FormatDescriptionInner<'a>,
                FormatDescriptionInner<'static>,
            >(self)
        }
    }

    /// Convert the inner `enum` to a `FormatDescription`.
    #[inline]
    pub const fn into_opaque(self) -> FormatDescription<'a> {
        FormatDescription {
            max_bytes_needed: self.max_bytes_needed(),
            inner: self,
        }
    }

    /// Obtain the maximum number of bytes that are needed to format any value using this format
    /// description.
    const fn max_bytes_needed(&self) -> usize {
        match self {
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
            #[cfg(feature = "large-dates")]
            Self::UnixTimestampSecond(_) => 15,
            #[cfg(not(feature = "large-dates"))]
            Self::UnixTimestampSecond(_) => 13,
            #[cfg(feature = "large-dates")]
            Self::UnixTimestampMillisecond(_) => 18,
            #[cfg(not(feature = "large-dates"))]
            Self::UnixTimestampMillisecond(_) => 16,
            #[cfg(feature = "large-dates")]
            Self::UnixTimestampMicrosecond(_) => 21,
            #[cfg(not(feature = "large-dates"))]
            Self::UnixTimestampMicrosecond(_) => 19,
            #[cfg(feature = "large-dates")]
            Self::UnixTimestampNanosecond(_) => 24,
            #[cfg(not(feature = "large-dates"))]
            Self::UnixTimestampNanosecond(_) => 22,
            Self::Ignore(_) | Self::End(_) => 0,
            FormatDescriptionInner::BorrowedLiteral(s) => s.len(),
            FormatDescriptionInner::BorrowedCompound(items) => {
                let mut max_bytes_needed = 0;
                let mut idx = 0;
                while idx < items.len() {
                    max_bytes_needed += items[idx].max_bytes_needed();
                    idx += 1;
                }
                max_bytes_needed
            }
            FormatDescriptionInner::BorrowedOptional { format, item } => {
                if *format {
                    item.max_bytes_needed()
                } else {
                    0
                }
            }
            FormatDescriptionInner::BorrowedFirst(items) => {
                if items.is_empty() {
                    0
                } else {
                    items[0].max_bytes_needed()
                }
            }
            FormatDescriptionInner::OwnedLiteral(s) => s.len(),
            FormatDescriptionInner::OwnedCompound(items) => {
                let mut max_bytes_needed = 0;
                let mut idx = 0;
                while idx < items.len() {
                    max_bytes_needed += items[idx].max_bytes_needed();
                    idx += 1;
                }
                max_bytes_needed
            }
            FormatDescriptionInner::OwnedOptional { format, item } => {
                if *format {
                    item.max_bytes_needed()
                } else {
                    0
                }
            }
            FormatDescriptionInner::OwnedFirst(items) => {
                if items.is_empty() {
                    0
                } else {
                    items[0].max_bytes_needed()
                }
            }
        }
    }
}

/// A component of a larger format description.
// public via `crate::format_description::__private` for macro use
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum Component {
    /// Day of the month.
    Day(Day),
    /// Month of the year in the abbreviated form (e.g. "Jan").
    MonthShort(MonthShort),
    /// Month of the year in the full form (e.g. "January").
    MonthLong(MonthLong),
    /// Month of the year in the numerical form (e.g. "1" for January).
    MonthNumerical(MonthNumerical),
    /// Ordinal day of the year.
    Ordinal(Ordinal),
    /// Weekday in the abbreviated form (e.g. "Mon").
    WeekdayShort(WeekdayShort),
    /// Weekday in the full form (e.g. "Monday").
    WeekdayLong(WeekdayLong),
    /// Weekday number where Sunday is either 0 or 1 depending on the modifier.
    WeekdaySunday(WeekdaySunday),
    /// Weekday number where Monday is either 0 or 1 depending on the modifier.
    WeekdayMonday(WeekdayMonday),
    /// Week number of the year, where week 1 starts is the week beginning on Monday that contains
    /// January 4.
    WeekNumberIso(WeekNumberIso),
    /// Week number of the year, where week 1 starts on the first Sunday of the calendar year.
    WeekNumberSunday(WeekNumberSunday),
    /// Week number of the year, where week 1 starts on the first Monday of the calendar year.
    WeekNumberMonday(WeekNumberMonday),
    /// The calendar year. Supports the extended range.
    CalendarYearFullExtendedRange(CalendarYearFullExtendedRange),
    /// The calendar year. Does not support the extended range.
    CalendarYearFullStandardRange(CalendarYearFullStandardRange),
    /// The ISO week-based year. Supports the extended range.
    IsoYearFullExtendedRange(IsoYearFullExtendedRange),
    /// The ISO week-based year. Does not support the extended range.
    IsoYearFullStandardRange(IsoYearFullStandardRange),
    /// The century of the calendar year. Supports the extended range.
    CalendarYearCenturyExtendedRange(CalendarYearCenturyExtendedRange),
    /// The century of the calendar year. Does not support the extended range.
    CalendarYearCenturyStandardRange(CalendarYearCenturyStandardRange),
    /// The century of the ISO week-based year. Supports the extended range.
    IsoYearCenturyExtendedRange(IsoYearCenturyExtendedRange),
    /// The century of the ISO week-based year. Does not support the extended range.
    IsoYearCenturyStandardRange(IsoYearCenturyStandardRange),
    /// The last two digits of the calendar year.
    CalendarYearLastTwo(CalendarYearLastTwo),
    /// The last two digits of the ISO week-based year.
    IsoYearLastTwo(IsoYearLastTwo),
    /// Hour of the day using the 12-hour clock.
    Hour12(Hour12),
    /// Hour of the day using the 24-hour clock.
    Hour24(Hour24),
    /// Minute within the hour.
    Minute(Minute),
    /// AM/PM part of the time.
    Period(Period),
    /// Second within the minute.
    Second(Second),
    /// Subsecond within the second.
    Subsecond(Subsecond),
    /// Hour of the UTC offset.
    OffsetHour(OffsetHour),
    /// Minute within the hour of the UTC offset.
    OffsetMinute(OffsetMinute),
    /// Second within the minute of the UTC offset.
    OffsetSecond(OffsetSecond),
    /// A number of bytes to ignore when parsing. This has no effect on formatting.
    Ignore(Ignore),
    /// A Unix timestamp in seconds.
    UnixTimestampSecond(UnixTimestampSecond),
    /// A Unix timestamp in milliseconds.
    UnixTimestampMillisecond(UnixTimestampMillisecond),
    /// A Unix timestamp in microseconds.
    UnixTimestampMicrosecond(UnixTimestampMicrosecond),
    /// A Unix timestamp in nanoseconds.
    UnixTimestampNanosecond(UnixTimestampNanosecond),
    /// The end of input. Parsing this component will fail if there is any input remaining. This
    /// component neither affects formatting nor consumes any input when parsing.
    End(End),
}

impl<'a> From<&'a Component> for FormatDescriptionInner<'a> {
    fn from(component: &'a Component) -> Self {
        match component {
            Component::Day(x) => Self::Day(*x),
            Component::MonthShort(x) => Self::MonthShort(*x),
            Component::MonthLong(x) => Self::MonthLong(*x),
            Component::MonthNumerical(x) => Self::MonthNumerical(*x),
            Component::Ordinal(x) => Self::Ordinal(*x),
            Component::WeekdayShort(x) => Self::WeekdayShort(*x),
            Component::WeekdayLong(x) => Self::WeekdayLong(*x),
            Component::WeekdaySunday(x) => Self::WeekdaySunday(*x),
            Component::WeekdayMonday(x) => Self::WeekdayMonday(*x),
            Component::WeekNumberIso(x) => Self::WeekNumberIso(*x),
            Component::WeekNumberSunday(x) => Self::WeekNumberSunday(*x),
            Component::WeekNumberMonday(x) => Self::WeekNumberMonday(*x),
            Component::CalendarYearFullExtendedRange(x) => {
                Self::CalendarYearFullExtendedRange(*x)
            }
            Component::CalendarYearFullStandardRange(x) => {
                Self::CalendarYearFullStandardRange(*x)
            }
            Component::IsoYearFullExtendedRange(x) => Self::IsoYearFullExtendedRange(*x),
            Component::IsoYearFullStandardRange(x) => Self::IsoYearFullStandardRange(*x),
            Component::CalendarYearCenturyExtendedRange(x) => {
                Self::CalendarYearCenturyExtendedRange(*x)
            }
            Component::CalendarYearCenturyStandardRange(x) => {
                Self::CalendarYearCenturyStandardRange(*x)
            }
            Component::IsoYearCenturyExtendedRange(x) => {
                Self::IsoYearCenturyExtendedRange(*x)
            }
            Component::IsoYearCenturyStandardRange(x) => {
                Self::IsoYearCenturyStandardRange(*x)
            }
            Component::CalendarYearLastTwo(x) => Self::CalendarYearLastTwo(*x),
            Component::IsoYearLastTwo(x) => Self::IsoYearLastTwo(*x),
            Component::Hour12(x) => Self::Hour12(*x),
            Component::Hour24(x) => Self::Hour24(*x),
            Component::Minute(x) => Self::Minute(*x),
            Component::Period(x) => Self::Period(*x),
            Component::Second(x) => Self::Second(*x),
            Component::Subsecond(x) => Self::Subsecond(*x),
            Component::OffsetHour(x) => Self::OffsetHour(*x),
            Component::OffsetMinute(x) => Self::OffsetMinute(*x),
            Component::OffsetSecond(x) => Self::OffsetSecond(*x),
            Component::Ignore(x) => Self::Ignore(*x),
            Component::UnixTimestampSecond(x) => Self::UnixTimestampSecond(*x),
            Component::UnixTimestampMillisecond(x) => Self::UnixTimestampMillisecond(*x),
            Component::UnixTimestampMicrosecond(x) => Self::UnixTimestampMicrosecond(*x),
            Component::UnixTimestampNanosecond(x) => Self::UnixTimestampNanosecond(*x),
            Component::End(x) => Self::End(*x),
        }
    }
}

impl crate::timeutil::date_formattable::Sealed for FormatDescriptionInner<'_> {
    fn format_into<V>(
        &self,
        output: &mut (impl std::io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, crate::timeutil::date_error::Error>
    where
        V: crate::timeutil::date_component_provider::ComponentProvider,
    {
        use crate::timeutil::date_formatting::*;

        match self {
            Self::Day(modifier) if V::SUPPLIES_DATE => {
                fmt_day(output, value.day(state), *modifier).map_err(Into::into)
            }
            Self::MonthShort(modifier) if V::SUPPLIES_DATE => {
                fmt_month_short(output, value.month(state), *modifier).map_err(Into::into)
            }
            Self::MonthLong(modifier) if V::SUPPLIES_DATE => {
                fmt_month_long(output, value.month(state), *modifier).map_err(Into::into)
            }
            Self::MonthNumerical(modifier) if V::SUPPLIES_DATE => {
                fmt_month_numerical(output, value.month(state), *modifier)
                    .map_err(Into::into)
            }
            Self::Ordinal(modifier) if V::SUPPLIES_DATE => {
                fmt_ordinal(output, value.ordinal(state), *modifier).map_err(Into::into)
            }
            Self::WeekdayShort(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_short(output, value.weekday(state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekdayLong(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_long(output, value.weekday(state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekdaySunday(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_sunday(output, value.weekday(state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekdayMonday(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_monday(output, value.weekday(state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekNumberIso(modifier) if V::SUPPLIES_DATE => {
                fmt_week_number_iso(output, value.iso_week_number(state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekNumberSunday(modifier) if V::SUPPLIES_DATE => {
                fmt_week_number_sunday(output, value.sunday_based_week(state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekNumberMonday(modifier) if V::SUPPLIES_DATE => {
                fmt_week_number_monday(output, value.monday_based_week(state), *modifier)
                    .map_err(Into::into)
            }
            Self::CalendarYearFullExtendedRange(modifier) if V::SUPPLIES_DATE => {
                fmt_calendar_year_full_extended_range(
                    output,
                    value.calendar_year(state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearFullStandardRange(modifier) if V::SUPPLIES_DATE => {
                fmt_calendar_year_full_standard_range(
                    output,
                    value
                        .calendar_year(state)
                        .narrow::<-9_999, 9_999>()
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "year",
                            )
                        })?
                        .try_into()
                        .map_err(|_| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "year",
                            )
                        })?,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::IsoYearFullExtendedRange(modifier) if V::SUPPLIES_DATE => {
                fmt_iso_year_full_extended_range(output, value.iso_year(state), *modifier)
                    .map_err(Into::into)
            }
            Self::IsoYearFullStandardRange(modifier) if V::SUPPLIES_DATE => {
                fmt_iso_year_full_standard_range(
                    output,
                    value
                        .iso_year(state)
                        .narrow::<-9_999, 9_999>()
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "year",
                            )
                        })?
                        .try_into()
                        .map_err(|_| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "year",
                            )
                        })?,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearCenturyExtendedRange(modifier) if V::SUPPLIES_DATE => {
                let year = value.calendar_year(state);
                let century =
                    deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?;
                fmt_calendar_year_century_extended_range(
                    output,
                    century,
                    year.is_negative(),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearCenturyStandardRange(modifier) if V::SUPPLIES_DATE => {
                let year = value.calendar_year(state);
                let is_negative = year.is_negative();
                let year =
                    deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?;
                fmt_calendar_year_century_standard_range(
                    output,
                    year.narrow::<-99, 99>()
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?
                        .try_into()
                        .map_err(|_| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?,
                    is_negative,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::IsoYearCenturyExtendedRange(modifier) if V::SUPPLIES_DATE => {
                let year = value.iso_year(state);
                let is_negative = year.is_negative();
                let century =
                    deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?;
                fmt_iso_year_century_extended_range(
                    output,
                    century,
                    is_negative,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::IsoYearCenturyStandardRange(modifier) if V::SUPPLIES_DATE => {
                let year = value.iso_year(state);
                let is_negative = year.is_negative();
                let year =
                    deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?;
                fmt_iso_year_century_standard_range(
                    output,
                    year.narrow::<-99, 99>()
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?
                        .try_into()
                        .map_err(|_| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "century",
                            )
                        })?,
                    is_negative,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearLastTwo(modifier) if V::SUPPLIES_DATE => {
                let year = value.calendar_year(state);
                let last_two =
                    deranged::RangedU8::new((year.get() % 100).unsigned_abs() as u8)
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "year",
                            )
                        })?;
                fmt_calendar_year_last_two(output, last_two, *modifier)
                    .map_err(Into::into)
            }
            Self::IsoYearLastTwo(modifier) if V::SUPPLIES_DATE => {
                let year = value.iso_year(state);
                let last_two =
                    deranged::RangedU8::new((year.get() % 100).unsigned_abs() as u8)
                        .ok_or_else(|| {
                            crate::timeutil::date_error::ComponentRange::conditional(
                                "year",
                            )
                        })?;
                fmt_iso_year_last_two(output, last_two, *modifier).map_err(Into::into)
            }
            Self::Hour12(modifier) if V::SUPPLIES_TIME => {
                fmt_hour_12(output, value.hour(state), *modifier).map_err(Into::into)
            }
            Self::Hour24(modifier) if V::SUPPLIES_TIME => {
                fmt_hour_24(output, value.hour(state), *modifier).map_err(Into::into)
            }
            Self::Minute(modifier) if V::SUPPLIES_TIME => {
                fmt_minute(output, value.minute(state), *modifier).map_err(Into::into)
            }
            Self::Period(modifier) if V::SUPPLIES_TIME => {
                fmt_period(output, value.period(state), *modifier).map_err(Into::into)
            }
            Self::Second(modifier) if V::SUPPLIES_TIME => {
                fmt_second(output, value.second(state), *modifier).map_err(Into::into)
            }
            Self::Subsecond(modifier) if V::SUPPLIES_TIME => {
                fmt_subsecond(output, value.subsecond(state), *modifier)
                    .map_err(Into::into)
            }
            Self::OffsetHour(modifier) if V::SUPPLIES_OFFSET => fmt_offset_hour(
                output,
                value.offset_is_negative(state),
                value.offset_hour(state),
                *modifier,
            )
            .map_err(Into::into),
            Self::OffsetMinute(modifier) if V::SUPPLIES_OFFSET => {
                fmt_offset_minute(output, value.offset_minute(state), *modifier)
                    .map_err(Into::into)
            }
            Self::OffsetSecond(modifier) if V::SUPPLIES_OFFSET => {
                fmt_offset_second(output, value.offset_second(state), *modifier)
                    .map_err(Into::into)
            }
            Self::Ignore(_) => Ok(0),
            Self::UnixTimestampSecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_second(
                    output,
                    value.unix_timestamp_seconds(state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::UnixTimestampMillisecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_millisecond(
                    output,
                    value.unix_timestamp_milliseconds(state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::UnixTimestampMicrosecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_microsecond(
                    output,
                    value.unix_timestamp_microseconds(state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::UnixTimestampNanosecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_nanosecond(
                    output,
                    value.unix_timestamp_nanoseconds(state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::End(_) => Ok(0),
            _ => Ok(0),
        }
    }
}

impl crate::timeutil::date_metadata::ComputeMetadata for FormatDescriptionInner<'_> {
    fn compute_metadata(&self) -> crate::timeutil::date_metadata::Metadata {
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
            Self::BorrowedLiteral(_) | Self::OwnedLiteral(_) => 0,
            Self::BorrowedCompound(_) | Self::OwnedCompound(_) => 0,
            Self::BorrowedOptional { .. } | Self::OwnedOptional { .. } => 0,
            Self::BorrowedFirst(_) | Self::OwnedFirst(_) => 0,
        };
        crate::timeutil::date_metadata::Metadata {
            max_bytes_needed,
            guaranteed_utf8: true,
        }
    }
}
