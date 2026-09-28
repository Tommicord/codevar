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

use crate::date_adt_hack::EncodedConfig;
use crate::date_component_provider::ComponentProvider;
use crate::date_error::Error;
use crate::date_format_description::{Component, FormatDescription, FormatDescriptionInner};
use crate::date_format_description_modifier::Padding;
use crate::date_formatting::{
    MONTH_NAMES, WEEKDAY_NAMES, format_four_digits_pad_zero, format_two_digits, write, write_if_else,
};
use crate::date_internal_macro::try_err;
use crate::date_iso8601::{format_date, format_offset, format_time};
use crate::date_metadata::{ComputeMetadata, Metadata};
use crate::date_num_fmt::truncated_subsecond_from_nanos;
use crate::date_well_know_iso8601::Iso8601;
use crate::date_well_know_rfc2822::Rfc2822;
use crate::date_well_know_rfc3339::Rfc3339;
use alloc::string::String;
use core::ops::Deref;
use deranged::{ru8, ru16};
use num_conv::prelude::*;

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
#[allow(
    private_bounds,
    private_interfaces,
    reason = "irrelevant due to being a sealed trait"
)]
pub trait Sealed: ComputeMetadata {
    /// Format the item into the provided output, returning the number of bytes written.
    fn format_into<V>(
        &self,
        output: &mut (impl core::fmt::Write + ?Sized),
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
        let Metadata { max_bytes_needed, .. } = self.compute_metadata();

        let mut buf = String::with_capacity(max_bytes_needed);
        self.format_into(&mut buf, value, state)?;
        Ok(buf)
    }
}

impl Sealed for FormatDescription<'_> {
    #[allow(
        private_bounds,
        private_interfaces,
        reason = "irrelevant due to being a sealed trait"
    )]
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl core::fmt::Write + ?Sized),
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
        output: &mut (impl core::fmt::Write + ?Sized),
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
    #[allow(
        private_bounds,
        private_interfaces,
        reason = "irrelevant due to being a sealed trait"
    )]
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl core::fmt::Write + ?Sized),
        value: &V,
        state: &mut V::State,
    ) -> Result<usize, Error>
    where
        V: ComponentProvider,
    {
        self.deref().format_into(output, value, state)
    }
}

#[allow(
    private_bounds,
    private_interfaces,
    reason = "irrelevant due to being a sealed trait"
)]
impl Sealed for Rfc2822 {
    #[inline]
    fn format_into<V>(
        &self,
        output: &mut (impl core::fmt::Write + ?Sized),
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

        if value.calendar_year(state).get() < 1900 && value.calendar_year(state).get() >= 10_000 {
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
                MONTH_NAMES[u8::from(value.month(state)).widen::<usize>() - 1].get_unchecked(..3)
            }),
            Error
        );
        bytes += try_err!(write(output, " "), Error);
        // Safety: Years with five or more digits were rejected above. Likewise with negative years.
        bytes += try_err!(
            format_four_digits_pad_zero(output, unsafe {
                ru16::new_unchecked(
                    value
                        .calendar_year(state)
                        .get()
                        .cast_unsigned()
                        .truncate(),
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
                unsafe { ru8::new_unchecked(value.offset_minute(state).get().unsigned_abs()) },
                Padding::Zero,
            ),
            Error
        );
        Ok(bytes)
    }
}

#[allow(
    private_bounds,
    private_interfaces,
    reason = "irrelevant due to being a sealed trait"
)]
impl Sealed for Rfc3339 {
    fn format_into<V>(
        &self,
        output: &mut (impl core::fmt::Write + ?Sized),
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
                    value
                        .calendar_year(state)
                        .get()
                        .cast_unsigned()
                        .truncate(),
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
                unsafe { ru8::new_unchecked(value.offset_minute(state).get().unsigned_abs()) },
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
        output: &mut (impl core::fmt::Write + ?Sized),
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
