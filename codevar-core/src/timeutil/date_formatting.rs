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

use crate::timeutil::date_format_description_modifier::{
    CalendarYearCenturyExtendedRange, CalendarYearCenturyStandardRange,
    CalendarYearFullExtendedRange, CalendarYearFullStandardRange, CalendarYearLastTwo,
    Day, DayPeriod, Hour12, Hour24, IsoYearCenturyExtendedRange,
    IsoYearCenturyStandardRange, IsoYearFullExtendedRange, IsoYearFullStandardRange,
    IsoYearLastTwo, Minute, MonthLong, MonthNumerical, MonthShort, OffsetHour,
    OffsetMinute, OffsetSecond, Ordinal, Padding, Period, Second, Subsecond,
    SubsecondDigits, UnixTimestampMicrosecond, UnixTimestampMillisecond,
    UnixTimestampNanosecond, UnixTimestampSecond, WeekNumberIso, WeekNumberMonday,
    WeekNumberSunday, WeekdayLong, WeekdayMonday, WeekdayShort, WeekdaySunday,
};
use crate::timeutil::date_month::Month;
use crate::timeutil::date_num_fmt::{
    StackStr, five_digits_zero_padded, four_digits_space_padded, four_digits_zero_padded,
    one_to_four_digits_no_padding, one_to_three_digits_no_padding,
    one_to_two_digits_no_padding, single_digit, six_digits_zero_padded,
    subsecond_from_nanos, three_digits_space_padded, three_digits_zero_padded,
    two_digits_space_padded, two_digits_zero_padded, u64_pad_none, u128_pad_none,
};
use crate::timeutil::date_time::{Hours, Minutes, Nanoseconds, Seconds};
use crate::timeutil::date_utc_offset::{
    Hours as OffsetHours, Minutes as OffsetMinutes, Seconds as OffsetSeconds,
};
use crate::timeutil::date_weekday::Weekday;
use deranged::{ru8, ru16, ru32};
use num_conv::{Truncate, Widen};
use std::io;
use std::mem::MaybeUninit;
use std::num::NonZero;

pub(crate) mod fmt_types {

    use deranged::{Option_ri32, Option_ru8, ri8, ri16, ri32, ru8, ru16};
    pub type Day = ru8<1, 31>;
    pub type OptionDay = Option_ru8<1, 31>;
    pub type Ordinal = ru16<1, 366>;
    pub type IsoWeekNumber = ru8<1, 53>;
    pub type OptionIsoWeekNumber = Option_ru8<1, 53>;
    pub type MondayBasedWeek = ru8<0, 53>;
    pub type SundayBasedWeek = ru8<0, 53>;
    pub type Year = ri32<-999_999, 999_999>;
    pub type StandardYear = ri16<-9_999, 9_999>;
    pub type OptionYear = Option_ri32<-999_999, 999_999>;
    pub type ExtendedCentury = ri16<-9_999, 9_999>;
    pub type StandardCentury = ri8<-99, 99>;
    pub type LastTwo = ru8<0, 99>;
}

pub const MONTH_NAMES: [&str; 12] = [
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

pub const WEEKDAY_NAMES: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

/// Write all bytes to the output, returning the number of bytes written.
#[inline]
pub(crate) fn write_bytes(
    output: &mut (impl io::Write + ?Sized),
    bytes: &[u8],
) -> io::Result<usize> {
    output.write_all(bytes)?;
    Ok(bytes.len())
}

/// Write the string to the output, returning the number of bytes written.
#[inline]
pub(crate) fn write(
    output: &mut (impl io::Write + ?Sized),
    s: &str,
) -> io::Result<usize> {
    output.write_all(s.as_bytes())?;
    Ok(s.len())
}

/// Write all strings to the output (in order), returning the total number of bytes written.
#[inline]
pub(crate) fn write_many<const N: usize>(
    output: &mut (impl io::Write + ?Sized),
    arr: [&str; N],
) -> io::Result<usize> {
    let mut bytes = 0;
    for s in arr {
        output.write_all(s.as_bytes())?;
        bytes += s.len();
    }
    Ok(bytes)
}

/// If `pred` is true, write the string to the output, returning the number of bytes written.
#[inline]
pub(crate) fn write_if(
    output: &mut (impl io::Write + ?Sized),
    pred: bool,
    s: &str,
) -> io::Result<usize> {
    if pred { write(output, s) } else { Ok(0) }
}

/// If `pred` is true, write `true_str` to the output. Otherwise, write `false_str`.
#[inline]
pub(crate) fn write_if_else(
    output: &mut (impl io::Write + ?Sized),
    pred: bool,
    true_str: &str,
    false_str: &str,
) -> io::Result<usize> {
    write(output, if pred { true_str } else { false_str })
}

/// Helper function to obtain 10^x, guaranteeing determinism for x ≤ 9. For these cases, the
/// function optimizes to a lookup table. For x ≥ 10, it falls back to `10_f64.powi(x)`. The only
/// situation where this would occur is if the user explicitly requests such precision when
/// configuring the ISO 8601 well known format. All other possibilities max out at nine digits.
#[inline]
fn f64_10_pow_x(x: NonZero<u8>) -> f64 {
    match x.get() {
        1 => 10.,
        2 => 100.,
        3 => 1_000.,
        4 => 10_000.,
        5 => 100_000.,
        6 => 1_000_000.,
        7 => 10_000_000.,
        8 => 100_000_000.,
        9 => 1_000_000_000.,
        x => 10_f64.powi(x.cast_signed().widen()),
    }
}

/// Write an integer with zeros as trailing padding if necessary to reach the requested width.
///
/// This function is intended to be used for formatting the fractional part of a value, as the
/// trailing zeros would change the semantic meaning for non-fractional values.
#[inline]
pub(crate) fn format_int_padded(
    output: &mut (impl io::Write + ?Sized),
    value: u64,
    width: u8,
) -> io::Result<usize> {
    let s = u64_pad_none(value);
    let digit_count = s.len() as u8;
    for _ in digit_count..width {
        output.write_all(b"0")?;
    }
    output.write_all(s.as_bytes())?;
    Ok(width as usize)
}

/// Write the floating point number to the output, returning the number of bytes written.
///
/// This method accepts the number of digits before and after the decimal. The value will be padded
/// with zeroes to the left if necessary.
#[inline]
pub(crate) fn format_float(
    output: &mut (impl io::Write + ?Sized),
    mut value: f64,
    digits_before_decimal: u8,
    digits_after_decimal: Option<NonZero<u8>>,
) -> io::Result<usize> {
    match digits_after_decimal {
        Some(digits_after_decimal) => {
            // If the precision is less than nine digits after the decimal point, truncate the
            // value. This avoids rounding up and causing the value to exceed the maximum permitted
            // value. If the precision is at least nine, then we don't truncate
            // to avoid having an off-by-one error. The latter is necessary
            // because floating point values are inherently imprecise with decimal
            // values, so a minuscule error can be amplified easily.
            //
            // Note that this is largely an issue for second values, as for minute and hour decimals
            // the value is divided by 60 or 3,600, neither of which divide evenly into 10^x.
            //
            // While not a perfect approach, this addresses the bugs that have been reported so far
            // without being overly complex.
            if digits_after_decimal.get() < 9 {
                let trunc_num = f64_10_pow_x(digits_after_decimal);
                value = f64::trunc(value * trunc_num) / trunc_num;

                let int_part = value.trunc() as u64;
                let frac_part =
                    f64::round(value.fract() * f64_10_pow_x(digits_after_decimal)) as u64;

                let width = digits_before_decimal.widen::<usize>()
                    + 1
                    + digits_after_decimal.get().widen::<usize>();

                format_int_padded(output, int_part, digits_before_decimal.widen())?;
                output.write_all(b".")?;
                format_int_padded(output, frac_part, digits_after_decimal.get().widen())?;
                Ok(width)
            } else {
                // For precision >= 9, use write! to avoid off-by-one errors from floating point
                // rounding. Integer extraction of the fractional part could overflow
                // the digit count when rounding causes a carry.
                let digits_after = digits_after_decimal.get().widen::<usize>();
                let width = digits_before_decimal.widen::<usize>() + 1 + digits_after;
                write!(output, "{value:0>width$.digits_after$}")?;
                Ok(width)
            }
        }
        None => format_int_padded(output, value.trunc() as u64, digits_before_decimal),
    }
}

/// Format a single digit.
#[inline]
pub(crate) fn format_single_digit(
    output: &mut (impl io::Write + ?Sized),
    value: ru8<0, 9>,
) -> io::Result<usize> {
    write(output, single_digit(value))
}

/// Format a two digit number with the specified padding.
#[inline]
pub(crate) fn format_two_digits(
    output: &mut (impl io::Write + ?Sized),
    value: ru8<0, 99>,
    padding: Padding,
) -> io::Result<usize> {
    let s = match padding {
        Padding::Space => two_digits_space_padded(value),
        Padding::Zero => two_digits_zero_padded(value),
        Padding::None => one_to_two_digits_no_padding(value),
    };
    write(output, s)
}

/// Format a three digit number with the specified padding.
#[inline]
pub(crate) fn format_three_digits(
    output: &mut (impl io::Write + ?Sized),
    value: ru16<0, 999>,
    padding: Padding,
) -> io::Result<usize> {
    let [first, second_and_third] = match padding {
        Padding::Space => three_digits_space_padded(value),
        Padding::Zero => three_digits_zero_padded(value),
        Padding::None => one_to_three_digits_no_padding(value),
    };
    write_many(output, [first, second_and_third])
}

/// Format a four digit number with the specified padding.
#[inline]
pub(crate) fn format_four_digits(
    output: &mut (impl io::Write + ?Sized),
    value: ru16<0, 9_999>,
    padding: Padding,
) -> io::Result<usize> {
    let [first_and_second, third_and_fourth] = match padding {
        Padding::Space => four_digits_space_padded(value),
        Padding::Zero => four_digits_zero_padded(value),
        Padding::None => one_to_four_digits_no_padding(value),
    };
    write_many(output, [first_and_second, third_and_fourth])
}

/// Format a four digit number that is padded with zeroes.
#[inline]
pub(crate) fn format_four_digits_pad_zero(
    output: &mut (impl io::Write + ?Sized),
    value: ru16<0, 9_999>,
) -> io::Result<usize> {
    write_many(output, four_digits_zero_padded(value))
}

/// Format a five digit number that is padded with zeroes.
#[inline]
pub(crate) fn format_five_digits_pad_zero(
    output: &mut (impl io::Write + ?Sized),
    value: ru32<0, 99_999>,
) -> io::Result<usize> {
    write_many(output, five_digits_zero_padded(value))
}

/// Format a six digit number that is padded with zeroes.
#[inline]
pub(crate) fn format_six_digits_pad_zero(
    output: &mut (impl io::Write + ?Sized),
    value: ru32<0, 999_999>,
) -> io::Result<usize> {
    write_many(output, six_digits_zero_padded(value))
}

/// Format a number with no padding.
///
/// If the sign is mandatory, the sign must be written by the caller.
#[inline]
pub(crate) fn format_u64_pad_none(
    output: &mut (impl io::Write + ?Sized),
    value: u64,
) -> io::Result<usize> {
    write(output, &u64_pad_none(value))
}

/// Format a number with no padding.
///
/// If the sign is mandatory, the sign must be written by the caller.
#[inline]
pub(crate) fn format_u128_pad_none(
    output: &mut (impl io::Write + ?Sized),
    value: u128,
) -> io::Result<usize> {
    write(output, &u128_pad_none(value))
}

/// Format the day into the designated output.
#[inline]
pub fn fmt_day(
    output: &mut (impl io::Write + ?Sized),
    day: fmt_types::Day,
    Day { padding }: Day,
) -> Result<usize, io::Error> {
    format_two_digits(output, day.expand(), padding)
}

/// Format the month into the designated output using the abbreviated name.
#[inline]
pub fn fmt_month_short(
    output: &mut (impl io::Write + ?Sized),
    month: Month,
    MonthShort {
        case_sensitive: _, // no effect on formatting
    }: MonthShort,
) -> io::Result<usize> {
    // Safety: All month names are at least three bytes long.
    write(output, unsafe {
        MONTH_NAMES[u8::from(month).widen::<usize>() - 1].get_unchecked(..3)
    })
}

/// Format the month into the designated output using the full name.
#[inline]
pub fn fmt_month_long(
    output: &mut (impl io::Write + ?Sized),
    month: Month,
    MonthLong {
        case_sensitive: _, // no effect on formatting
    }: MonthLong,
) -> io::Result<usize> {
    write(output, MONTH_NAMES[u8::from(month).widen::<usize>() - 1])
}

/// Format the month into the designated output as a number from 1-12.
#[inline]
pub fn fmt_month_numerical(
    output: &mut (impl io::Write + ?Sized),
    month: Month,
    MonthNumerical { padding }: MonthNumerical,
) -> io::Result<usize> {
    format_two_digits(
        output,
        // Safety: The month is guaranteed to be in the range `1..=12`.
        unsafe { ru8::new_unchecked(u8::from(month)) },
        padding,
    )
}

/// Format the ordinal into the designated output.
#[inline]
pub fn fmt_ordinal(
    output: &mut (impl io::Write + ?Sized),
    ordinal: fmt_types::Ordinal,
    Ordinal { padding }: Ordinal,
) -> Result<usize, io::Error> {
    format_three_digits(output, ordinal.expand(), padding)
}

/// Format the weekday into the designated output using the abbreviated name.
#[inline]
pub fn fmt_weekday_short(
    output: &mut (impl io::Write + ?Sized),
    weekday: Weekday,
    WeekdayShort {
        case_sensitive: _, // no effect on formatting
    }: WeekdayShort,
) -> io::Result<usize> {
    // Safety: All weekday names are at least three bytes long.
    write(output, unsafe {
        WEEKDAY_NAMES[weekday.number_days_from_monday().widen::<usize>()]
            .get_unchecked(..3)
    })
}

/// Format the weekday into the designated output using the full name.
#[inline]
pub fn fmt_weekday_long(
    output: &mut (impl io::Write + ?Sized),
    weekday: Weekday,
    WeekdayLong {
        case_sensitive: _, // no effect on formatting
    }: WeekdayLong,
) -> io::Result<usize> {
    write(
        output,
        WEEKDAY_NAMES[weekday.number_days_from_monday().widen::<usize>()],
    )
}

/// Format the weekday into the designated output as a number from either 0-6 or 1-7 (depending on
/// the modifier), where Sunday is either 0 or 1.
#[inline]
pub fn fmt_weekday_sunday(
    output: &mut (impl io::Write + ?Sized),
    weekday: Weekday,
    WeekdaySunday { one_indexed }: WeekdaySunday,
) -> io::Result<usize> {
    // Safety: The value is guaranteed to be in the range `0..=7`.
    format_single_digit(output, unsafe {
        ru8::new_unchecked(weekday.number_days_from_sunday() + u8::from(one_indexed))
    })
}

/// Format the weekday into the designated output as a number from either 0-6 or 1-7 (depending on
/// the modifier), where Monday is either 0 or 1.
#[inline]
pub fn fmt_weekday_monday(
    output: &mut (impl io::Write + ?Sized),
    weekday: Weekday,
    WeekdayMonday { one_indexed }: WeekdayMonday,
) -> io::Result<usize> {
    // Safety: The value is guaranteed to be in the range `0..=7`.
    format_single_digit(output, unsafe {
        ru8::new_unchecked(weekday.number_days_from_monday() + u8::from(one_indexed))
    })
}

#[inline]
pub fn fmt_week_number_iso(
    output: &mut (impl io::Write + ?Sized),
    week_number: fmt_types::IsoWeekNumber,
    WeekNumberIso { padding }: WeekNumberIso,
) -> io::Result<usize> {
    format_two_digits(output, week_number.expand(), padding)
}

#[inline]
pub fn fmt_week_number_sunday(
    output: &mut (impl io::Write + ?Sized),
    week_number: fmt_types::SundayBasedWeek,
    WeekNumberSunday { padding }: WeekNumberSunday,
) -> io::Result<usize> {
    format_two_digits(output, week_number.expand(), padding)
}

#[inline]
pub fn fmt_week_number_monday(
    output: &mut (impl io::Write + ?Sized),
    week_number: fmt_types::MondayBasedWeek,
    WeekNumberMonday { padding }: WeekNumberMonday,
) -> io::Result<usize> {
    format_two_digits(output, week_number.expand(), padding)
}

#[inline]
pub fn fmt_calendar_year_full_extended_range(
    output: &mut (impl io::Write + ?Sized),
    full_year: fmt_types::Year,
    CalendarYearFullExtendedRange {
        padding,
        sign_is_mandatory,
    }: CalendarYearFullExtendedRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(
        output,
        full_year.is_negative(),
        sign_is_mandatory || full_year.get() >= 10_000,
    )?;
    // Safety: We just called `.abs()`, so zero is the minimum. The maximum is
    // unchanged.
    let value: ru32<0, 999_999> =
        unsafe { full_year.abs().narrow_unchecked::<0, 999_999>().into() };

    bytes += if let Some(value) = value.narrow::<0, 9_999>() {
        format_four_digits(output, value.into(), padding)?
    } else if let Some(value) = value.narrow::<0, 99_999>() {
        format_five_digits_pad_zero(output, value)?
    } else {
        format_six_digits_pad_zero(output, value)?
    };
    Ok(bytes)
}

#[inline]
pub fn fmt_calendar_year_full_standard_range(
    output: &mut (impl io::Write + ?Sized),
    full_year: fmt_types::StandardYear,
    CalendarYearFullStandardRange {
        padding,
        sign_is_mandatory,
    }: CalendarYearFullStandardRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(output, full_year.is_negative(), sign_is_mandatory)?;
    // Safety: The minimum is zero due to the `.abs()` call; the maximum is unchanged.
    bytes += format_four_digits(
        output,
        unsafe { full_year.abs().narrow_unchecked::<0, 9_999>().into() },
        padding,
    )?;
    Ok(bytes)
}

#[inline]
pub fn fmt_iso_year_full_extended_range(
    output: &mut (impl io::Write + ?Sized),
    full_year: fmt_types::Year,
    IsoYearFullExtendedRange {
        padding,
        sign_is_mandatory,
    }: IsoYearFullExtendedRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(
        output,
        full_year.is_negative(),
        sign_is_mandatory || full_year.get() >= 10_000,
    )?;
    // Safety: The minimum is zero due to the `.abs()` call, with the maximum is unchanged.
    let value: ru32<0, 999_999> =
        unsafe { full_year.abs().narrow_unchecked::<0, 999_999>().into() };

    bytes += if let Some(value) = value.narrow::<0, 9_999>() {
        format_four_digits(output, value.into(), padding)?
    } else if let Some(value) = value.narrow::<0, 99_999>() {
        format_five_digits_pad_zero(output, value)?
    } else {
        format_six_digits_pad_zero(output, value)?
    };
    Ok(bytes)
}

#[inline]
pub fn fmt_iso_year_full_standard_range(
    output: &mut (impl io::Write + ?Sized),
    year: fmt_types::StandardYear,
    IsoYearFullStandardRange {
        padding,
        sign_is_mandatory,
    }: IsoYearFullStandardRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(output, year.is_negative(), sign_is_mandatory)?;
    // Safety: The minimum is zero due to the `.abs()` call; the maximum is unchanged.
    bytes += format_four_digits(
        output,
        unsafe { year.abs().narrow_unchecked::<0, 9_999>().into() },
        padding,
    )?;
    Ok(bytes)
}

#[inline]
pub fn fmt_calendar_year_century_extended_range(
    output: &mut (impl io::Write + ?Sized),
    century: fmt_types::ExtendedCentury,
    is_negative: bool,
    CalendarYearCenturyExtendedRange {
        padding,
        sign_is_mandatory,
    }: CalendarYearCenturyExtendedRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(
        output,
        is_negative,
        sign_is_mandatory || century.get() >= 100,
    )?;
    // Safety: The minimum is zero due to the `.abs()` call;  the maximum is unchanged.
    let century: ru16<0, 9_999> =
        unsafe { century.abs().narrow_unchecked::<0, 9_999>().into() };

    bytes += if let Some(century) = century.narrow::<0, 99>() {
        format_two_digits(output, century.into(), padding)?
    } else if let Some(century) = century.narrow::<0, 999>() {
        format_three_digits(output, century, padding)?
    } else {
        format_four_digits(output, century, padding)?
    };
    Ok(bytes)
}

#[inline]
pub fn fmt_calendar_year_century_standard_range(
    output: &mut (impl io::Write + ?Sized),
    century: fmt_types::StandardCentury,
    is_negative: bool,
    CalendarYearCenturyStandardRange {
        padding,
        sign_is_mandatory,
    }: CalendarYearCenturyStandardRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(output, is_negative, sign_is_mandatory)?;
    // Safety: The minimum is zero due to the `.unsigned_abs()` call.
    let century = unsafe { century.abs().narrow_unchecked::<0, 99>() };
    bytes += format_two_digits(output, century.into(), padding)?;
    Ok(bytes)
}

#[inline]
pub fn fmt_iso_year_century_extended_range(
    output: &mut (impl io::Write + ?Sized),
    century: fmt_types::ExtendedCentury,
    is_negative: bool,
    IsoYearCenturyExtendedRange {
        padding,
        sign_is_mandatory,
    }: IsoYearCenturyExtendedRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(
        output,
        is_negative,
        sign_is_mandatory || century.get() >= 100,
    )?;
    // Safety: The minimum is zero due to the `.unsigned_abs()` call, with the maximum is unchanged.
    let century: ru16<0, 9_999> =
        unsafe { century.abs().narrow_unchecked::<0, 9_999>().into() };

    bytes += if let Some(century) = century.narrow::<0, 99>() {
        format_two_digits(output, century.into(), padding)?
    } else if let Some(century) = century.narrow::<0, 999>() {
        format_three_digits(output, century, padding)?
    } else {
        format_four_digits(output, century, padding)?
    };
    Ok(bytes)
}

#[inline]
pub fn fmt_iso_year_century_standard_range(
    output: &mut (impl io::Write + ?Sized),
    century: fmt_types::StandardCentury,
    is_negative: bool,
    IsoYearCenturyStandardRange {
        padding,
        sign_is_mandatory,
    }: IsoYearCenturyStandardRange,
) -> io::Result<usize> {
    let mut bytes = 0;
    bytes += fmt_sign(output, is_negative, sign_is_mandatory)?;
    // Safety: The minimum is zero due to the `.unsigned_abs()` call.
    let century = unsafe { century.abs().narrow_unchecked::<0, 99>() };
    bytes += format_two_digits(output, century.into(), padding)?;
    Ok(bytes)
}

#[inline]
pub fn fmt_calendar_year_last_two(
    output: &mut (impl io::Write + ?Sized),
    last_two: fmt_types::LastTwo,
    CalendarYearLastTwo { padding }: CalendarYearLastTwo,
) -> io::Result<usize> {
    format_two_digits(output, last_two, padding)
}

#[inline]
pub fn fmt_iso_year_last_two(
    output: &mut (impl io::Write + ?Sized),
    last_two: fmt_types::LastTwo,
    IsoYearLastTwo { padding }: IsoYearLastTwo,
) -> io::Result<usize> {
    format_two_digits(output, last_two, padding)
}

/// Format the hour into the designated output using the 12-hour clock.
#[inline]
pub fn fmt_hour_12(
    output: &mut (impl io::Write + ?Sized),
    hour: Hours,
    Hour12 { padding }: Hour12,
) -> io::Result<usize> {
    // Safety: The value is guaranteed to be in the range `1..=12`.
    format_two_digits(
        output,
        unsafe { ru8::new_unchecked((hour.get() + 11) % 12 + 1) },
        padding,
    )
}

/// Format the hour into the designated output using the 24-hour clock.
#[inline]
pub fn fmt_hour_24(
    output: &mut (impl io::Write + ?Sized),
    hour: Hours,
    Hour24 { padding }: Hour24,
) -> io::Result<usize> {
    format_two_digits(output, hour.expand(), padding)
}

/// Format the minute into the designated output.
#[inline]
pub fn fmt_minute(
    output: &mut (impl io::Write + ?Sized),
    minute: Minutes,
    Minute { padding }: Minute,
) -> Result<usize, io::Error> {
    format_two_digits(output, minute.expand(), padding)
}

/// Format the period into the designated output.
#[inline]
pub fn fmt_period(
    output: &mut (impl io::Write + ?Sized),
    period: DayPeriod,
    Period {
        is_uppercase,
        case_sensitive: _, // no effect on formatting
    }: Period,
) -> Result<usize, io::Error> {
    write(
        output,
        match (period, is_uppercase) {
            (DayPeriod::Am, false) => "am",
            (DayPeriod::Am, true) => "AM",
            (DayPeriod::Pm, false) => "pm",
            (DayPeriod::Pm, true) => "PM",
        },
    )
}

/// Format the second into the designated output.
#[inline]
pub fn fmt_second(
    output: &mut (impl io::Write + ?Sized),
    second: Seconds,
    Second { padding }: Second,
) -> Result<usize, io::Error> {
    format_two_digits(output, second.expand(), padding)
}

/// Format the subsecond into the designated output.
#[inline]
pub fn fmt_subsecond(
    output: &mut (impl io::Write + ?Sized),
    nanos: Nanoseconds,
    Subsecond { digits }: Subsecond,
) -> Result<usize, io::Error> {
    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    struct Digits {
        _padding: MaybeUninit<[u8; 7]>,
        digit_1: u8,
        digits_2_thru_9: [u8; 8],
    }
    let [
        digit_1,
        digits_2_and_3,
        digits_4_and_5,
        digits_6_and_7,
        digits_8_and_9,
    ] = subsecond_from_nanos(nanos);

    // Ensure that digits 2 thru 9 are stored as a single array that is 8-aligned. This allows the
    // conversion to a `u64` to be zero cost, resulting in a nontrivial performance improvement.
    let buf = Digits {
        _padding: MaybeUninit::uninit(),
        digit_1: digit_1.as_bytes()[0],
        digits_2_thru_9: [
            digits_2_and_3.as_bytes()[0],
            digits_2_and_3.as_bytes()[1],
            digits_4_and_5.as_bytes()[0],
            digits_4_and_5.as_bytes()[1],
            digits_6_and_7.as_bytes()[0],
            digits_6_and_7.as_bytes()[1],
            digits_8_and_9.as_bytes()[0],
            digits_8_and_9.as_bytes()[1],
        ],
    };
    let len = match digits {
        SubsecondDigits::One => 1,
        SubsecondDigits::Two => 2,
        SubsecondDigits::Three => 3,
        SubsecondDigits::Four => 4,
        SubsecondDigits::Five => 5,
        SubsecondDigits::Six => 6,
        SubsecondDigits::Seven => 7,
        SubsecondDigits::Eight => 8,
        SubsecondDigits::Nine => 9,
        SubsecondDigits::OneOrMore => {
            // By converting the bytes into a single integer, we can effectively perform an equality
            // check against b'0' for all bytes at once. This is actually faster than
            // using portable SIMD (even with `-Ctarget-cpu=native`).
            let bitmask =
                u64::from_le_bytes(buf.digits_2_thru_9) ^ u64::from_le_bytes([b'0'; 8]);
            let digits_to_truncate = bitmask.leading_zeros() / 8;
            9 - digits_to_truncate as usize
        }
    };
    // Safety: All bytes are initialized and valid UTF-8, and `len` represents the number of bytes
    // we wish to display (that is between 1 and 9 inclusive). `Digits` is `#[repr(C)]`, so the
    // layout is guaranteed.
    let s = unsafe {
        StackStr::new(
            *(&raw const buf)
                .byte_add(core::mem::offset_of!(Digits, digit_1))
                .cast::<[MaybeUninit<u8>; 9]>(),
            len,
        )
    };
    write(output, &s)
}

#[inline]
pub fn fmt_sign(
    output: &mut (impl io::Write + ?Sized),
    is_negative: bool,
    sign_is_mandatory: bool,
) -> Result<usize, io::Error> {
    if is_negative {
        write(output, "-")
    } else if sign_is_mandatory {
        write(output, "+")
    } else {
        Ok(0)
    }
}

/// Format the offset hour into the designated output.
#[inline]
pub fn fmt_offset_hour(
    output: &mut (impl io::Write + ?Sized),
    is_negative: bool,
    hour: OffsetHours,
    OffsetHour {
        padding,
        sign_is_mandatory,
    }: OffsetHour,
) -> Result<usize, io::Error> {
    let mut bytes = 0;
    bytes += fmt_sign(output, is_negative, sign_is_mandatory)?;
    // Safety: The value is guaranteed to be under 100 because of `OffsetHours`.
    bytes += format_two_digits(
        output,
        unsafe { ru8::new_unchecked(hour.get().unsigned_abs()) },
        padding,
    )?;
    Ok(bytes)
}

/// Format the offset minute into the designated output.
#[inline]
pub fn fmt_offset_minute(
    output: &mut (impl io::Write + ?Sized),
    offset_minute: OffsetMinutes,
    OffsetMinute { padding }: OffsetMinute,
) -> Result<usize, io::Error> {
    format_two_digits(
        output,
        // Safety: `OffsetMinutes` is guaranteed to be in the range `-59..=59`, so the absolute
        // value is guaranteed to be in the range `0..=59`.
        unsafe { ru8::new_unchecked(offset_minute.get().unsigned_abs()) },
        padding,
    )
}

/// Format the offset second into the designated output.
#[inline]
pub fn fmt_offset_second(
    output: &mut (impl io::Write + ?Sized),
    offset_second: OffsetSeconds,
    OffsetSecond { padding }: OffsetSecond,
) -> Result<usize, io::Error> {
    format_two_digits(
        output,
        // Safety: `OffsetSeconds` is guaranteed to be in the range `-59..=59`, so the absolute
        // value is guaranteed to be in the range `0..=59`.
        unsafe { ru8::new_unchecked(offset_second.get().unsigned_abs()) },
        padding,
    )
}

/// Format the Unix timestamp (in seconds) into the designated output.
#[inline]
pub fn fmt_unix_timestamp_second(
    output: &mut (impl io::Write + ?Sized),
    timestamp: i64,
    UnixTimestampSecond { sign_is_mandatory }: UnixTimestampSecond,
) -> Result<usize, io::Error> {
    let mut bytes = 0;
    bytes += fmt_sign(output, timestamp < 0, sign_is_mandatory)?;
    bytes += format_u64_pad_none(output, timestamp.unsigned_abs())?;
    Ok(bytes)
}

/// Format the Unix timestamp (in milliseconds) into the designated output.
#[inline]
pub fn fmt_unix_timestamp_millisecond(
    output: &mut (impl io::Write + ?Sized),
    timestamp_millis: i64,
    UnixTimestampMillisecond { sign_is_mandatory }: UnixTimestampMillisecond,
) -> Result<usize, io::Error> {
    let mut bytes = 0;
    bytes += fmt_sign(output, timestamp_millis < 0, sign_is_mandatory)?;
    bytes += format_u64_pad_none(output, timestamp_millis.unsigned_abs())?;
    Ok(bytes)
}

/// Format the Unix timestamp (in microseconds) into the designated output.
#[inline]
pub fn fmt_unix_timestamp_microsecond(
    output: &mut (impl io::Write + ?Sized),
    timestamp_micros: i128,
    UnixTimestampMicrosecond { sign_is_mandatory }: UnixTimestampMicrosecond,
) -> Result<usize, io::Error> {
    let mut bytes = 0;
    bytes += fmt_sign(output, timestamp_micros < 0, sign_is_mandatory)?;
    bytes += format_u128_pad_none(output, timestamp_micros.unsigned_abs())?;
    Ok(bytes)
}

/// Format the Unix timestamp (in nanoseconds) into the designated output.
#[inline]
pub fn fmt_unix_timestamp_nanosecond(
    output: &mut (impl io::Write + ?Sized),
    timestamp_nanos: i128,
    UnixTimestampNanosecond { sign_is_mandatory }: UnixTimestampNanosecond,
) -> Result<usize, io::Error> {
    let mut bytes = 0;
    bytes += fmt_sign(output, timestamp_nanos < 0, sign_is_mandatory)?;
    bytes += format_u128_pad_none(output, timestamp_nanos.unsigned_abs())?;
    Ok(bytes)
}
