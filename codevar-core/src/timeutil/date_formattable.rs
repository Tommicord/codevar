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
use crate::timeutil::date_component_provider::ComponentProvider;
use crate::timeutil::date_error::{ComponentRange, Error};
use crate::timeutil::date_format_description::{
    Component, FormatDescription, FormatDescriptionInner,
};
use crate::timeutil::date_format_description_modifier::{End, Padding};
use crate::timeutil::date_formatting::{
    MONTH_NAMES, WEEKDAY_NAMES, fmt_calendar_year_century_extended_range,
    fmt_calendar_year_century_standard_range, fmt_calendar_year_full_extended_range,
    fmt_calendar_year_full_standard_range, fmt_calendar_year_last_two, fmt_day,
    fmt_hour_12, fmt_hour_24, fmt_iso_year_century_extended_range,
    fmt_iso_year_century_standard_range, fmt_iso_year_full_extended_range,
    fmt_iso_year_full_standard_range, fmt_iso_year_last_two, fmt_minute, fmt_month_long,
    fmt_month_numerical, fmt_month_short, fmt_offset_hour, fmt_offset_minute,
    fmt_offset_second, fmt_ordinal, fmt_period, fmt_second, fmt_subsecond,
    fmt_unix_timestamp_microsecond, fmt_unix_timestamp_millisecond,
    fmt_unix_timestamp_nanosecond, fmt_unix_timestamp_second, fmt_week_number_iso,
    fmt_week_number_monday, fmt_week_number_sunday, fmt_weekday_long, fmt_weekday_monday,
    fmt_weekday_short, fmt_weekday_sunday, format_four_digits_pad_zero,
    format_two_digits, write, write_if_else,
};
use crate::timeutil::date_internal_macro::try_err;
use crate::timeutil::date_iso8601::{format_date, format_offset, format_time};
use crate::timeutil::date_metadata::{ComputeMetadata, Metadata};
use crate::timeutil::date_num_fmt::truncated_subsecond_from_nanos;
use crate::timeutil::date_well_know_iso8601::Iso8601;
use crate::timeutil::date_well_know_rfc2822::Rfc2822;
use crate::timeutil::date_well_know_rfc3339::Rfc3339;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Deref;
use deranged::{ri16, ru8, ru16};
use num_conv::prelude::*;
use std::io;

macro_rules! fmt_component_match {
    ($self:expr, $output:ident, $value:ident, $state:ident, $($extra:tt)*) => {
        match $self {
            Self::Day(modifier) if V::SUPPLIES_DATE => {
                fmt_day($output, $value.day($state), *modifier).map_err(Into::into)
            }
            Self::MonthShort(modifier) if V::SUPPLIES_DATE => {
                fmt_month_short($output, $value.month($state), *modifier).map_err(Into::into)
            }
            Self::MonthLong(modifier) if V::SUPPLIES_DATE => {
                fmt_month_long($output, $value.month($state), *modifier).map_err(Into::into)
            }
            Self::MonthNumerical(modifier) if V::SUPPLIES_DATE => {
                fmt_month_numerical($output, $value.month($state), *modifier).map_err(Into::into)
            }
            Self::Ordinal(modifier) if V::SUPPLIES_DATE => {
                fmt_ordinal($output, $value.ordinal($state), *modifier).map_err(Into::into)
            }
            Self::WeekdayShort(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_short($output, $value.weekday($state), *modifier).map_err(Into::into)
            }
            Self::WeekdayLong(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_long($output, $value.weekday($state), *modifier).map_err(Into::into)
            }
            Self::WeekdaySunday(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_sunday($output, $value.weekday($state), *modifier).map_err(Into::into)
            }
            Self::WeekdayMonday(modifier) if V::SUPPLIES_DATE => {
                fmt_weekday_monday($output, $value.weekday($state), *modifier).map_err(Into::into)
            }
            Self::WeekNumberIso(modifier) if V::SUPPLIES_DATE => {
                fmt_week_number_iso($output, $value.iso_week_number($state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekNumberSunday(modifier) if V::SUPPLIES_DATE => {
                fmt_week_number_sunday($output, $value.sunday_based_week($state), *modifier)
                    .map_err(Into::into)
            }
            Self::WeekNumberMonday(modifier) if V::SUPPLIES_DATE => {
                fmt_week_number_monday($output, $value.monday_based_week($state), *modifier)
                    .map_err(Into::into)
            }
            Self::CalendarYearFullExtendedRange(modifier) if V::SUPPLIES_DATE => {
                fmt_calendar_year_full_extended_range(
                    $output,
                    $value.calendar_year($state),
                    *modifier
                ).map_err(Into::into)
            }
            Self::CalendarYearFullStandardRange(modifier) if V::SUPPLIES_DATE => {
                fmt_calendar_year_full_standard_range(
                    $output,
                    $value
                        .calendar_year($state)
                        .narrow::<-9_999, 9_999>()
                        .ok_or_else(|| ComponentRange::conditional("year"))?
                        .try_into()
                        .map_err(|_| ComponentRange::conditional("year"))?,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::IsoYearFullExtendedRange(modifier) if V::SUPPLIES_DATE => {
                fmt_iso_year_full_extended_range($output, $value.iso_year($state), *modifier)
                    .map_err(Into::into)
            }
            Self::IsoYearFullStandardRange(modifier) if V::SUPPLIES_DATE => {
                fmt_iso_year_full_standard_range(
                    $output,
                    $value.iso_year($state)
                          .narrow::<-9_999, 9_999>()
                          .ok_or_else(|| ComponentRange::conditional("year"))?
                          .try_into()
                          .map_err(|_| ComponentRange::conditional("year"))?,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearCenturyExtendedRange(modifier) if V::SUPPLIES_DATE => {
                let year = $value.calendar_year($state);
                // Safety: Given the range of `year`, the range of the century is `-9_999..=9_999`.
                let century = deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                    .ok_or_else(|| ComponentRange::conditional("century"))?;
                fmt_calendar_year_century_extended_range(
                    $output,
                    century,
                    year.is_negative(),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearCenturyStandardRange(modifier) if V::SUPPLIES_DATE => {
                let year = $value.calendar_year($state);
                let is_negative = year.is_negative();
                // Safety: Given the range of `year`, the range of the century is `-9_999..=9_999`.
                let year = deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                    .ok_or_else(|| ComponentRange::conditional("century"))?;
                fmt_calendar_year_century_standard_range(
                    $output,
                    year.narrow::<-99, 99>()
                        .ok_or_else(|| ComponentRange::conditional("century"))?
                        .try_into()
                        .map_err(|_| ComponentRange::conditional("century"))?,
                    is_negative,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::IsoYearCenturyExtendedRange(modifier) if V::SUPPLIES_DATE => {
                let year = $value.iso_year($state);
                let is_negative = year.is_negative();
                // Safety: Given the range of `year`, the range of the century is `-9_999..=9_999`.
                let century = deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                    .ok_or_else(|| ComponentRange::conditional("century"))?;
                fmt_iso_year_century_extended_range($output, century, is_negative, *modifier)
                    .map_err(Into::into)
            }
            Self::IsoYearCenturyStandardRange(modifier) if V::SUPPLIES_DATE => {
                let year = $value.iso_year($state);
                let is_negative = year.is_negative();
                // Safety: Given the range of `year`, the range of the century is `-9_999..=9_999`.
                let year = deranged::RangedI16::<-9_999, 9_999>::new((year.get() / 100) as i16)
                    .ok_or_else(|| ComponentRange::conditional("century"))?;
                fmt_iso_year_century_standard_range(
                    $output,
                    year.narrow::<-99, 99>()
                        .ok_or_else(|| ComponentRange::conditional("century"))?
                        .try_into()
                        .map_err(|_| ComponentRange::conditional("century"))?,
                    is_negative,
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::CalendarYearLastTwo(modifier) if V::SUPPLIES_DATE => {
                // Safety: Modulus of 100 followed by `.unsigned_abs()` guarantees that the $value
                // is in the range `0..=99`.
                let last_two = unsafe {
                    ru8::new_unchecked(
                        ($value.calendar_year($state).get().unsigned_abs() % 100).truncate(),
                    )
                };
                fmt_calendar_year_last_two($output, last_two, *modifier).map_err(Into::into)
            }
            Self::IsoYearLastTwo(modifier) if V::SUPPLIES_DATE => {
                // Safety: Modulus of 100 followed by `.unsigned_abs()` guarantees that the $value
                // is in the range `0..=99`.
                let last_two = unsafe {
                    ru8::new_unchecked(
                        ($value.iso_year($state).get().unsigned_abs() % 100).truncate(),
                    )
                };
                fmt_iso_year_last_two($output, last_two, *modifier).map_err(Into::into)
            }
            Self::Hour12(modifier) if V::SUPPLIES_TIME => {
                fmt_hour_12($output, $value.hour($state), *modifier).map_err(Into::into)
            }
            Self::Hour24(modifier) if V::SUPPLIES_TIME => {
                fmt_hour_24($output, $value.hour($state), *modifier).map_err(Into::into)
            }
            Self::Minute(modifier) if V::SUPPLIES_TIME => {
                fmt_minute($output, $value.minute($state), *modifier).map_err(Into::into)
            }
            Self::Period(modifier) if V::SUPPLIES_TIME => {
                fmt_period($output, *modifier, $value.period($state)).map_err(Into::into)
            }
            Self::Second(modifier) if V::SUPPLIES_TIME => {
                fmt_second($output, $value.second($state), *modifier).map_err(Into::into)
            }
            Self::Subsecond(modifier) if V::SUPPLIES_TIME => {
                fmt_subsecond($output, $value.nanosecond($state), *modifier).map_err(Into::into)
            }
            Self::OffsetHour(modifier) if V::SUPPLIES_OFFSET => fmt_offset_hour(
                $output,
                $value.offset_is_negative($state),
                $value.offset_hour($state),
                *modifier,
            )
            .map_err(Into::into),
            Self::OffsetMinute(modifier) if V::SUPPLIES_OFFSET => {
                fmt_offset_minute($output, $value.offset_minute($state), *modifier)
                    .map_err(Into::into)
            }
            Self::OffsetSecond(modifier) if V::SUPPLIES_OFFSET => {
                fmt_offset_second($output, $value.offset_second($state), *modifier)
                    .map_err(Into::into)
            }
            Self::Ignore(_) => Ok(0),
            Self::UnixTimestampSecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_second($output, $value.unix_timestamp_seconds($state), *modifier)
                    .map_err(Into::into)
            }
            Self::UnixTimestampMillisecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_millisecond(
                    $output,
                    $value.unix_timestamp_milliseconds($state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::UnixTimestampMicrosecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_microsecond(
                    $output,
                    $value.unix_timestamp_microseconds($state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::UnixTimestampNanosecond(modifier) if V::SUPPLIES_TIMESTAMP => {
                fmt_unix_timestamp_nanosecond(
                    $output,
                    $value.unix_timestamp_nanoseconds($state),
                    *modifier,
                )
                .map_err(Into::into)
            }
            Self::End(End { trailing_input: _ }) => Ok(0),
            $($extra)*
        }
    };
}

/// A type that describes a format.
///
/// Implementors of [`Formattable`] are [format descriptions](crate::format_description).
///
/// To format a value into a String, use the `format` method on the respective type.
#[cfg_attr(docsrs, doc(notable_trait))]
pub trait Formattable: Sealed {}
impl Formattable for Rfc3339 {}
impl Formattable for Rfc2822 {}
impl<const CONFIG: EncodedConfig> Formattable for Iso8601<CONFIG> {}
impl<T> Formattable for T where T: Deref<Target: Formattable> {}

/// Format the item using a format description, the intended output, and the various components.
#[expect(
    private_bounds,
    private_interfaces,
    reason = "irrelevant due to being a sealed trait"
)]
pub trait Sealed: ComputeMetadata {
    /// Format the item into the provided output, returning the number of bytes written.
    fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider;

    /// Format the item directly to a `String`.
    #[inline]
    fn format<V>(&self, value: &V, state: &mut V::State) -> Result<String, Error>
    where
        V: ComponentProvider,
    {
        let Metadata {
            max_bytes_needed,
            guaranteed_utf8,
        } = self.compute_metadata();

        let mut buf = Vec::with_capacity(max_bytes_needed);
        Ok(if guaranteed_utf8 {
            // Safety: The output is guaranteed to be UTF-8.
            unsafe { String::from_utf8_unchecked(buf) }
        } else {
            String::from_utf8_lossy(&buf).into_owned()
        })
    }
}

impl Sealed for FormatDescription<'_> {
    #[expect(
        private_bounds,
        private_interfaces,
        reason = "irrelevant due to being a sealed trait"
    )]
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        self.inner.format_into(output, value, state)
    }
}

impl Component {
    /// Format the component directly into the provided output.
    #[inline]
    pub(crate) fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        FormatDescriptionInner::<'_>::from(self).format_into(output, value, state)
    }
}

impl<T> Sealed for T
where
    T: Deref<Target: Sealed>,
{
    #[expect(
        private_bounds,
        private_interfaces,
        reason = "irrelevant due to being a sealed trait"
    )]
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        self.deref().format_into(output, value, state)
    }
}

#[expect(
    private_bounds,
    private_interfaces,
    reason = "irrelevant due to being a sealed trait"
)]
impl Sealed for Rfc2822 {
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        const {
            assert!(
                V::SUPPLIES_DATE && V::SUPPLIES_TIME && V::SUPPLIES_OFFSET,
                "Rfc2822 requires date, time, and offset components, but not all can be provided \
                 by this type"
            );
        }

        let mut bytes = 0;

        if value.calendar_year(state).get() < 1900
            && value.calendar_year(state).get() >= 10_000
        {
            return Err(Error::InvalidComponent("year"));
        }
        if value.offset_second(state).get() != 0 {
            return Err(Error::InvalidComponent("offset_second"));
        }
        // Safety: All weekday names are at least 3 bytes long.
        bytes += try_err!(
            write(output, unsafe {
                WEEKDAY_NAMES[value
                    .weekday(state)
                    .number_days_from_monday()
                    .widen::<usize>()]
                .get_unchecked(..3)
            }),
            Error
        );
        bytes += try_err!(write(output, ", "), Error);
        bytes += try_err!(
            format_two_digits(output, value.day(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, " "), Error);
        bytes += try_err!(
            write(output, unsafe {
                MONTH_NAMES[u8::from(value.month(state)).widen::<usize>() - 1]
                    .get_unchecked(..3)
            }),
            Error
        );
        bytes += try_err!(write(output, " "), Error);
        // Safety: Years with five or more digits were rejected above. Likewise with negative years.
        bytes += try_err!(
            format_four_digits_pad_zero(output, unsafe {
                ru16::new_unchecked(
                    value.calendar_year(state).get().cast_unsigned().truncate(),
                )
            }),
            Error
        );
        bytes += try_err!(write(output, " "), Error);
        bytes += try_err!(
            format_two_digits(output, value.hour(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, ":"), Error);
        bytes += try_err!(
            format_two_digits(output, value.minute(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, ":"), Error);
        bytes += try_err!(
            format_two_digits(output, value.second(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, " "), Error);
        bytes += try_err!(
            write_if_else(output, value.offset_is_negative(state), "-", "+"),
            Error
        );
        bytes += try_err!(
            format_two_digits(
                output,
                // Safety: `OffsetMinutes` is guaranteed to be in the range `-59..=59`, so the absolute
                // value is guaranteed to be in the range `0..=59`.
                unsafe {
                    ru8::new_unchecked(value.offset_minute(state).get().unsigned_abs())
                },
                Padding::Zero,
            ),
            Error
        );
        Ok(bytes)
    }
}

#[expect(
    private_bounds,
    private_interfaces,
    reason = "irrelevant due to being a sealed trait"
)]
impl Sealed for Rfc3339 {
    fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        const {
            assert!(
                V::SUPPLIES_DATE && V::SUPPLIES_TIME && V::SUPPLIES_OFFSET,
                "Rfc3339 requires date, time, and offset components, but not all can be provided \
                 by this type"
            );
        }
        let offset_hour = value.offset_hour(state);
        let mut bytes = 0;

        if !(0..10_000).contains(&value.calendar_year(state).get()) {
            return Err(Error::InvalidComponent("year"));
        }
        if offset_hour.get().unsigned_abs() > 23 {
            return Err(Error::InvalidComponent("offset_hour"));
        }
        if value.offset_second(state).get() != 0 {
            return Err(Error::InvalidComponent("offset_second"));
        }
        // Safety: Years outside this range were rejected above.
        bytes += try_err!(
            format_four_digits_pad_zero(output, unsafe {
                ru16::new_unchecked(
                    value.calendar_year(state).get().cast_unsigned().truncate(),
                )
            }),
            Error
        );
        bytes += try_err!(write(output, "-"), Error);
        bytes += try_err!(
            format_two_digits(
                output,
                // Safety: `month` is guaranteed to be in the range `1..=12`.
                unsafe { ru8::new_unchecked(u8::from(value.month(state))) },
                Padding::Zero,
            ),
            Error
        );
        bytes += try_err!(write(output, "-"), Error);
        bytes += try_err!(
            format_two_digits(output, value.day(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, "T"), Error);
        bytes += try_err!(
            format_two_digits(output, value.hour(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, ":"), Error);
        bytes += try_err!(
            format_two_digits(output, value.minute(state).expand(), Padding::Zero),
            Error
        );
        bytes += try_err!(write(output, ":"), Error);
        bytes += try_err!(
            format_two_digits(output, value.second(state).expand(), Padding::Zero),
            Error
        );

        let nanos = value.nanosecond(state);
        if nanos.get() != 0 {
            bytes += try_err!(write(output, "."), Error);
            try_err!(write(output, &truncated_subsecond_from_nanos(nanos)), Error);
        }

        if value.offset_is_utc(state) {
            bytes += try_err!(write(output, "Z"), Error);
            return Ok(bytes);
        }

        bytes += try_err!(
            write_if_else(output, value.offset_is_negative(state), "-", "+"),
            Error
        );
        bytes += try_err!(
            format_two_digits(
                output,
                // Safety: `OffsetHours` is guaranteed to be in the range `-23..=23`, so the absolute
                // value is guaranteed to be in the range `0..=23`.
                unsafe { ru8::new_unchecked(offset_hour.get().unsigned_abs()) },
                Padding::Zero,
            ),
            Error
        );
        bytes += try_err!(write(output, ":"), Error);
        bytes += try_err!(
            format_two_digits(
                output,
                // Safety: `OffsetMinutes` is guaranteed to be in the range `-59..=59`, so the absolute
                // value is guaranteed to be in the range `0..=59`.
                unsafe {
                    ru8::new_unchecked(value.offset_minute(state).get().unsigned_abs())
                },
                Padding::Zero,
            ),
            Error
        );
        Ok(bytes)
    }
}

impl<const CONFIG: EncodedConfig> Sealed for Iso8601<CONFIG> {
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl io::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        let mut bytes = 0;

        const {
            assert!(
                !Self::FORMAT_DATE || V::SUPPLIES_DATE,
                "this Iso8601 configuration formats date components, but this type cannot provide \
                 them"
            );
            assert!(
                !Self::FORMAT_TIME || V::SUPPLIES_TIME,
                "this Iso8601 configuration formats time components, but this type cannot provide \
                 them"
            );
            assert!(
                !Self::FORMAT_OFFSET || V::SUPPLIES_OFFSET,
                "this Iso8601 configuration formats offset components, but this type cannot \
                 provide them"
            );
            assert!(
                Self::FORMAT_DATE || Self::FORMAT_TIME || Self::FORMAT_OFFSET,
                "this Iso8601 configuration does not format any components"
            );
        }

        if Self::FORMAT_DATE {
            bytes += try_err!(format_date::<_, CONFIG>(output, value, state), Error);
        }
        if Self::FORMAT_TIME {
            bytes += try_err!(format_time::<_, CONFIG>(output, value, state), Error);
        }
        if Self::FORMAT_OFFSET {
            bytes += try_err!(format_offset::<_, CONFIG>(output, value, state), Error);
        }
        Ok(bytes)
    }
}
