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

use crate::timeutil::date::{MAX_YEAR, MIN_YEAR};
use crate::timeutil::date_error::ComponentRange;
use crate::timeutil::date_internal_macro::{
    cascade, const_try, const_try_opt, div_floor, ensure_ranged,
};
use crate::timeutil::date_num_fmt::{
    four_to_six_digits, one_to_two_digits_no_padding, str_from_raw_parts,
    truncated_subsecond_from_nanos, two_digits_zero_padded,
};
use crate::timeutil::date_signed_duration::SignedDuration;
use crate::timeutil::date_unit::{
    Day, Hour, Microsecond, Millisecond, Minute, Nanosecond, Second, Subsecond,
};
use crate::timeutil::date_util::{
    DateAdjustment, days_in_month_leap, days_in_year, is_leap_year, weeks_in_year,
};
use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use deranged::{ri32, ru8, ru32};
use num_conv::prelude::*;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

type Year = ri32<MIN_YEAR, MAX_YEAR>;

/// By explicitly inserting this enum where padding is expected, the compiler is able to better
/// perform niche value optimization.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Padding {
    #[allow(clippy::missing_docs_in_private_items)]
    Optimize,
}

/// The type of the `hour` field of `Time`.
pub(crate) type Hours = ru8<0, { Hour::per_t::<u8>(Day) - 1 }>;
/// The type of the `minute` field of `Time`.
pub(crate) type Minutes = ru8<0, { Minute::per_t::<u8>(Hour) - 1 }>;
/// The type of the `second` field of `Time`.
pub(crate) type Seconds = ru8<0, { Second::per_t::<u8>(Minute) - 1 }>;
/// The type of the `second` field of `Time`.
pub(crate) type Subseconds = ru32<0, { Subsecond::per_t::<u32>(Second) - 1 }>;
/// The type of the `nanosecond` field of `Time`.
pub(crate) type Nanoseconds = ru32<0, { Nanosecond::per_t::<u32>(Second) - 1 }>;

/// Date in the proleptic Gregorian calendar.
///
/// By default, years between ±9999 inclusive are representable. This can be expanded to ±999,999
/// inclusive by enabling the `large-dates` crate feature. Doing so has performance implications
/// and introduces some ambiguities when parsing.
#[derive(Clone, Copy, Eq)]
#[cfg_attr(not(docsrs), repr(C))]
pub struct Time {
    #[cfg(target_endian = "little")]
    nanosecond: Nanoseconds,
    #[cfg(target_endian = "little")]
    second: Seconds,
    #[cfg(target_endian = "little")]
    minute: Minutes,
    #[cfg(target_endian = "little")]
    hour: Hours,
    #[cfg(target_endian = "little")]
    padding: Padding,

    // Big endian version
    #[cfg(target_endian = "big")]
    padding: Padding,
    #[cfg(target_endian = "big")]
    hour: Hours,
    #[cfg(target_endian = "big")]
    minute: Minutes,
    #[cfg(target_endian = "big")]
    second: Seconds,
    #[cfg(target_endian = "big")]
    nanosecond: Nanoseconds,
}

impl Hash for Time {
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.as_u64().hash(state)
    }
}

impl PartialEq for Time {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_u64().eq(&other.as_u64())
    }
}

impl PartialOrd for Time {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Time {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_u64().cmp(&other.as_u64())
    }
}

impl Time {
    /// Provide a representation of `Time` as a `u64`. This value can be used for equality, hashing,
    /// and ordering.
    #[inline]
    pub(crate) const fn as_u64(self) -> u64 {
        // Safety: `self` is presumed valid because it exists, and any value of `u64` is valid. Size
        // and alignment are enforced by the compiler. There is no implicit padding in either `Time`
        // or `u64`.
        unsafe { core::mem::transmute(self) }
    }

    /// Create a `Time` from its components.
    ///
    /// # Safety
    ///
    /// - `hours` must be in the range `0..=23`.
    /// - `minutes` must be in the range `0..=59`.
    /// - `seconds` must be in the range `0..=59`.
    /// - `nanoseconds` must be in the range `0..=999_999_999`.
    #[doc(hidden)]
    #[inline]
    pub const unsafe fn from_hms_nanos_unchecked(
        hour: u8,
        minute: u8,
        second: u8,
        nanosecond: u32,
    ) -> Self {
        // Safety: The caller must uphold the safety invariants.
        unsafe {
            Self::from_hms_nanos_ranged(
                Hours::new_unchecked(hour),
                Minutes::new_unchecked(minute),
                Seconds::new_unchecked(second),
                Nanoseconds::new_unchecked(nanosecond),
            )
        }
    }

    /// A `Time` that is exactly midnight. This is the smallest possible value for a `Time`.
    #[doc(alias = "MIN")]
    pub const MIDNIGHT: Self = Self::from_hms_nanos_ranged(
        Hours::MIN,
        Minutes::MIN,
        Seconds::MIN,
        Nanoseconds::MIN,
    );

    /// A `Time` that is one nanosecond before midnight. This is the largest possible value for a
    /// `Time`.
    pub const MAX: Self = Self::from_hms_nanos_ranged(
        Hours::MAX,
        Minutes::MAX,
        Seconds::MAX,
        Nanoseconds::MAX,
    );

    /// Attempt to create a `Time` from the hour, minute, and second.
    #[inline]
    pub const fn from_hms(
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_hms_nanos_ranged(
            ensure_ranged!(Hours: hour),
            ensure_ranged!(Minutes: minute),
            ensure_ranged!(Seconds: second),
            Nanoseconds::MIN,
        ))
    }

    /// Create a `Time` from the hour, minute, second, and nanosecond.
    #[inline]
    pub(crate) const fn from_hms_nanos_ranged(
        hour: Hours,
        minute: Minutes,
        second: Seconds,
        nanosecond: Nanoseconds,
    ) -> Self {
        Self {
            hour,
            minute,
            second,
            nanosecond,
            padding: Padding::Optimize,
        }
    }

    /// Attempt to create a `Time` from the hour, minute, second, and millisecond.
    #[inline]
    pub const fn from_hms_milli(
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_hms_nanos_ranged(
            ensure_ranged!(Hours: hour),
            ensure_ranged!(Minutes: minute),
            ensure_ranged!(Seconds: second),
            ensure_ranged!(Nanoseconds: millisecond * Nanosecond::per_t::<u32>(Millisecond)),
        ))
    }

    /// Attempt to create a `Time` from the hour, minute, second, and microsecond.
    #[inline]
    pub const fn from_hms_micro(
        hour: u8,
        minute: u8,
        second: u8,
        microsecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_hms_nanos_ranged(
            ensure_ranged!(Hours: hour),
            ensure_ranged!(Minutes: minute),
            ensure_ranged!(Seconds: second),
            ensure_ranged!(Nanoseconds: microsecond * Nanosecond::per_t::<u32>(Microsecond)),
        ))
    }

    /// Attempt to create a `Time` from the hour, minute, second, and nanosecond.
    #[inline]
    pub const fn from_hms_nano(
        hour: u8,
        minute: u8,
        second: u8,
        nanosecond: u32,
    ) -> Result<Self, ComponentRange> {
        Ok(Self::from_hms_nanos_ranged(
            ensure_ranged!(Hours: hour),
            ensure_ranged!(Minutes: minute),
            ensure_ranged!(Seconds: second),
            ensure_ranged!(Nanoseconds: nanosecond),
        ))
    }

    /// Get the clock hour, minute, and second.
    #[inline]
    pub const fn as_hms(self) -> (u8, u8, u8) {
        (self.hour.get(), self.minute.get(), self.second.get())
    }

    /// Get the clock hour, minute, second, and millisecond.
    #[inline]
    pub const fn as_hms_milli(self) -> (u8, u8, u8, u16) {
        (
            self.hour.get(),
            self.minute.get(),
            self.second.get(),
            (self.nanosecond.get() / Nanosecond::per_t::<u32>(Millisecond)) as u16,
        )
    }

    /// Get the clock hour, minute, second, and microsecond.
    #[inline]
    pub const fn as_hms_micro(self) -> (u8, u8, u8, u32) {
        (
            self.hour.get(),
            self.minute.get(),
            self.second.get(),
            self.nanosecond.get() / Nanosecond::per_t::<u32>(Microsecond),
        )
    }

    /// Get the clock hour, minute, second, and nanosecond.
    #[inline]
    pub const fn as_hms_nano(self) -> (u8, u8, u8, u32) {
        (
            self.hour.get(),
            self.minute.get(),
            self.second.get(),
            self.nanosecond.get(),
        )
    }

    /// Get the clock hour, minute, second, and nanosecond.
    #[inline]
    pub(crate) const fn as_hms_nano_ranged(
        self,
    ) -> (Hours, Minutes, Seconds, Nanoseconds) {
        (self.hour, self.minute, self.second, self.nanosecond)
    }

    /// Get the clock hour.
    ///
    /// The returned value will always be in the range `0..24`.
    #[inline]
    pub const fn hour(self) -> u8 {
        self.hour.get()
    }

    /// Get the minute within the hour.
    ///
    /// The returned value will always be in the range `0..60`.
    #[inline]
    pub const fn minute(self) -> u8 {
        self.minute.get()
    }

    /// Get the second within the minute.
    ///
    /// The returned value will always be in the range `0..60`.
    #[inline]
    pub const fn second(self) -> u8 {
        self.second.get()
    }

    /// Get the milliseconds within the second.
    ///
    /// The returned value will always be in the range `0..1_000`.
    #[inline]
    pub const fn millisecond(self) -> u16 {
        (self.nanosecond.get() / Nanosecond::per_t::<u32>(Millisecond)) as u16
    }

    /// Get the microseconds within the second.
    ///
    /// The returned value will always be in the range `0..1_000_000`.
    #[inline]
    pub const fn microsecond(self) -> u32 {
        self.nanosecond.get() / Nanosecond::per_t::<u32>(Microsecond)
    }

    /// Get the nanoseconds within the second.
    ///
    /// The returned value will always be in the range `0..1_000_000_000`.
    #[inline]
    pub const fn nanosecond(self) -> u32 {
        self.nanosecond.get()
    }

    /// Determine the [`SignedDuration`] that, if added to `self`, would result in the parameter.
    #[inline]
    pub const fn duration_until(self, other: Self) -> SignedDuration {
        let mut nanoseconds =
            other.nanosecond.get().cast_signed() - self.nanosecond.get().cast_signed();
        let seconds = other.second.get().cast_signed() - self.second.get().cast_signed();
        let minutes = other.minute.get().cast_signed() - self.minute.get().cast_signed();
        let hours = other.hour.get().cast_signed() - self.hour.get().cast_signed();

        // Safety: For all four variables, the bounds are obviously true given the previous bounds
        // and nature of subtraction.
        unsafe {
            core::hint::assert_unchecked(
                nanoseconds
                    >= Nanoseconds::MIN.get().cast_signed()
                        - Nanoseconds::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                nanoseconds
                    <= Nanoseconds::MAX.get().cast_signed()
                        - Nanoseconds::MIN.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                seconds
                    >= Seconds::MIN.get().cast_signed()
                        - Seconds::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                seconds
                    <= Seconds::MAX.get().cast_signed()
                        - Seconds::MIN.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                minutes
                    >= Minutes::MIN.get().cast_signed()
                        - Minutes::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                minutes
                    <= Minutes::MAX.get().cast_signed()
                        - Minutes::MIN.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                hours >= Hours::MIN.get().cast_signed() - Hours::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                hours <= Hours::MAX.get().cast_signed() - Hours::MIN.get().cast_signed(),
            );
        }

        let mut total_seconds = hours as i32 * Second::per_t::<i32>(Hour)
            + minutes as i32 * Second::per_t::<i32>(Minute)
            + seconds as i32;

        cascade!(nanoseconds in 0..Nanosecond::per_t(Second) => total_seconds);

        if total_seconds < 0 {
            total_seconds += Second::per_t::<i32>(Day);
        }

        // Safety: The range of `nanoseconds` is guaranteed by the cascades above.
        unsafe { SignedDuration::new_unchecked(total_seconds as i64, nanoseconds) }
    }

    /// Determine the [`SignedDuration`] that, if added to the parameter, would result in `self`.
    #[inline]
    pub const fn duration_since(self, other: Self) -> SignedDuration {
        other.duration_until(self)
    }

    /// Add the sub-day time of the [`SignedDuration`] to the `Time`. Wraps on overflow, returning
    /// whether the date is different.
    #[inline]
    pub(crate) const fn adjusting_add(
        self,
        duration: SignedDuration,
    ) -> (DateAdjustment, Self) {
        let mut nanoseconds =
            self.nanosecond.get().cast_signed() + duration.subsec_nanoseconds();
        let mut seconds = self.second.get().cast_signed()
            + (duration.whole_seconds() % Second::per_t::<i64>(Minute)) as i8;
        let mut minutes = self.minute.get().cast_signed()
            + (duration.whole_minutes() % Minute::per_t::<i64>(Hour)) as i8;
        let mut hours = self.hour.get().cast_signed()
            + (duration.whole_hours() % Hour::per_t::<i64>(Day)) as i8;
        let mut date_adjustment = DateAdjustment::None;

        cascade!(nanoseconds in 0..Nanosecond::per_t(Second) => seconds);
        cascade!(seconds in 0..Second::per_t(Minute) => minutes);
        cascade!(minutes in 0..Minute::per_t(Hour) => hours);
        if hours >= Hour::per_t(Day) {
            hours -= Hour::per_t::<i8>(Day);
            date_adjustment = DateAdjustment::Next;
        } else if hours < 0 {
            hours += Hour::per_t::<i8>(Day);
            date_adjustment = DateAdjustment::Previous;
        }

        (
            date_adjustment,
            // Safety: The cascades above ensure the values are in range.
            unsafe {
                Self::from_hms_nanos_unchecked(
                    hours.cast_unsigned(),
                    minutes.cast_unsigned(),
                    seconds.cast_unsigned(),
                    nanoseconds.cast_unsigned(),
                )
            },
        )
    }

    /// Subtract the sub-day time of the [`SignedDuration`] to the `Time`. Wraps on overflow,
    /// returning whether the date is different.
    #[inline]
    pub(crate) const fn adjusting_sub(
        self,
        duration: SignedDuration,
    ) -> (DateAdjustment, Self) {
        let mut nanoseconds =
            self.nanosecond.get().cast_signed() - duration.subsec_nanoseconds();
        let mut seconds = self.second.get().cast_signed()
            - (duration.whole_seconds() % Second::per_t::<i64>(Minute)) as i8;
        let mut minutes = self.minute.get().cast_signed()
            - (duration.whole_minutes() % Minute::per_t::<i64>(Hour)) as i8;
        let mut hours = self.hour.get().cast_signed()
            - (duration.whole_hours() % Hour::per_t::<i64>(Day)) as i8;
        let mut date_adjustment = DateAdjustment::None;

        cascade!(nanoseconds in 0..Nanosecond::per_t(Second) => seconds);
        cascade!(seconds in 0..Second::per_t(Minute) => minutes);
        cascade!(minutes in 0..Minute::per_t(Hour) => hours);
        if hours >= Hour::per_t(Day) {
            hours -= Hour::per_t::<i8>(Day);
            date_adjustment = DateAdjustment::Next;
        } else if hours < 0 {
            hours += Hour::per_t::<i8>(Day);
            date_adjustment = DateAdjustment::Previous;
        }

        (
            date_adjustment,
            // Safety: The cascades above ensure the values are in range.
            unsafe {
                Self::from_hms_nanos_unchecked(
                    hours.cast_unsigned(),
                    minutes.cast_unsigned(),
                    seconds.cast_unsigned(),
                    nanoseconds.cast_unsigned(),
                )
            },
        )
    }

    /// Add the sub-day time of the [`std::time::Duration`] to the `Time`. Wraps on overflow,
    /// returning whether the date is the previous date as the first element of the tuple.
    #[inline]
    pub(crate) const fn adjusting_add_std(self, duration: StdDuration) -> (bool, Self) {
        let mut nanosecond = self.nanosecond.get() + duration.subsec_nanos();
        let mut second =
            self.second.get() + (duration.as_secs() % Second::per_t::<u64>(Minute)) as u8;
        let mut minute = self.minute.get()
            + ((duration.as_secs() / Second::per_t::<u64>(Minute))
                % Minute::per_t::<u64>(Hour)) as u8;
        let mut hour = self.hour.get()
            + ((duration.as_secs() / Second::per_t::<u64>(Hour))
                % Hour::per_t::<u64>(Day)) as u8;
        let mut is_next_day = false;

        cascade!(nanosecond in 0..Nanosecond::per_t(Second) => second);
        cascade!(second in 0..Second::per_t(Minute) => minute);
        cascade!(minute in 0..Minute::per_t(Hour) => hour);
        if hour >= Hour::per_t::<u8>(Day) {
            hour -= Hour::per_t::<u8>(Day);
            is_next_day = true;
        }

        (
            is_next_day,
            // Safety: The cascades above ensure the values are in range.
            unsafe { Self::from_hms_nanos_unchecked(hour, minute, second, nanosecond) },
        )
    }

    /// Subtract the sub-day time of the [`std::time::Duration`] to the `Time`. Wraps on overflow,
    /// returning whether the date is the previous date as the first element of the tuple.
    #[inline]
    pub(crate) const fn adjusting_sub_std(self, duration: StdDuration) -> (bool, Self) {
        let mut nanosecond =
            self.nanosecond.get().cast_signed() - duration.subsec_nanos().cast_signed();
        let mut second = self.second.get().cast_signed()
            - (duration.as_secs() % Second::per_t::<u64>(Minute)) as i8;
        let mut minute = self.minute.get().cast_signed()
            - ((duration.as_secs() / Second::per_t::<u64>(Minute))
                % Minute::per_t::<u64>(Hour)) as i8;
        let mut hour = self.hour.get().cast_signed()
            - ((duration.as_secs() / Second::per_t::<u64>(Hour))
                % Hour::per_t::<u64>(Day)) as i8;
        let mut is_previous_day = false;

        cascade!(nanosecond in 0..Nanosecond::per_t(Second) => second);
        cascade!(second in 0..Second::per_t(Minute) => minute);
        cascade!(minute in 0..Minute::per_t(Hour) => hour);
        if hour < 0 {
            hour += Hour::per_t::<i8>(Day);
            is_previous_day = true;
        }

        (
            is_previous_day,
            // Safety: The cascades above ensure the values are in range.
            unsafe {
                Self::from_hms_nanos_unchecked(
                    hour.cast_unsigned(),
                    minute.cast_unsigned(),
                    second.cast_unsigned(),
                    nanosecond.cast_unsigned(),
                )
            },
        )
    }

    /// Replace the clock hour.
    #[inline]
    pub const fn replace_hour(mut self, hour: u8) -> Result<Self, ComponentRange> {
        self.hour = ensure_ranged!(Hours: hour);
        Ok(self)
    }

    /// Truncate the time to the hour, setting the minute, second, and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_hour(mut self) -> Self {
        self.minute = Minutes::MIN;
        self.second = Seconds::MIN;
        self.nanosecond = Nanoseconds::MIN;
        self
    }

    /// Replace the minutes within the hour.
    #[inline]
    pub const fn replace_minute(mut self, minute: u8) -> Result<Self, ComponentRange> {
        self.minute = ensure_ranged!(Minutes: minute);
        Ok(self)
    }

    /// Truncate the time to the minute, setting the second and subsecond components to zero.
    #[inline]
    pub const fn truncate_to_minute(mut self) -> Self {
        self.second = Seconds::MIN;
        self.nanosecond = Nanoseconds::MIN;
        self
    }

    /// Replace the seconds within the minute.
    #[inline]
    pub const fn replace_second(mut self, second: u8) -> Result<Self, ComponentRange> {
        self.second = ensure_ranged!(Seconds: second);
        Ok(self)
    }

    /// Truncate the time to the second, setting the subsecond component to zero.
    #[inline]
    pub const fn truncate_to_second(mut self) -> Self {
        self.nanosecond = Nanoseconds::MIN;
        self
    }

    /// Replace the milliseconds within the second.
    #[inline]
    pub const fn replace_millisecond(
        mut self,
        millisecond: u16,
    ) -> Result<Self, ComponentRange> {
        self.nanosecond = ensure_ranged!(Nanoseconds: millisecond as u32 * Nanosecond::per_t::<u32>(Millisecond));
        Ok(self)
    }

    /// Truncate the time to the millisecond, setting the microsecond and nanosecond components to
    /// zero.
    #[inline]
    pub const fn truncate_to_millisecond(mut self) -> Self {
        // Safety: Truncating to the millisecond will always produce a valid nanosecond.
        self.nanosecond = unsafe {
            Nanoseconds::new_unchecked(
                self.nanosecond.get() - (self.nanosecond.get() % 1_000_000),
            )
        };
        self
    }

    /// Replace the microseconds within the second.
    #[inline]
    pub const fn replace_microsecond(
        mut self,
        microsecond: u32,
    ) -> Result<Self, ComponentRange> {
        self.nanosecond = ensure_ranged!(Nanoseconds: microsecond * Nanosecond::per_t::<u32>(Microsecond));
        Ok(self)
    }

    /// Truncate the time to the microsecond, setting the nanosecond component to zero.
    #[inline]
    pub const fn truncate_to_microsecond(mut self) -> Self {
        // Safety: Truncating to the microsecond will always produce a valid nanosecond.
        self.nanosecond = unsafe {
            Nanoseconds::new_unchecked(
                self.nanosecond.get() - (self.nanosecond.get() % 1_000),
            )
        };
        self
    }

    /// Replace the nanoseconds within the second.
    #[inline]
    pub const fn replace_nanosecond(
        mut self,
        nanosecond: u32,
    ) -> Result<Self, ComponentRange> {
        self.nanosecond = ensure_ranged!(Nanoseconds: nanosecond);
        Ok(self)
    }
}

// This no longer needs special handling, as the format is fixed and doesn't require anything
// advanced. Trait impls can't be deprecated and the info is still useful for other types
// implementing `SmartDisplay`, so leave it as-is for now.
impl SmartDisplay for Time {
    type Metadata = ();

    #[inline]
    fn metadata(&self, _: FormatterOptions) -> Metadata<'_, Self> {
        let hour_width = if self.hour() < 10 { 1 } else { 2 };
        let subsecond_width = match self.nanosecond() {
            nanos if nanos % 10 != 0 => 9,
            nanos if (nanos / 10) % 10 != 0 => 8,
            nanos if (nanos / 100) % 10 != 0 => 7,
            nanos if (nanos / 1_000) % 10 != 0 => 6,
            nanos if (nanos / 10_000) % 10 != 0 => 5,
            nanos if (nanos / 100_000) % 10 != 0 => 4,
            nanos if (nanos / 1_000_000) % 10 != 0 => 3,
            nanos if (nanos / 10_000_000) % 10 != 0 => 2,
            _ => 1,
        };
        let total_width = hour_width + subsecond_width + 7;

        Metadata::new(total_width, self, ())
    }

    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Time {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// for the `Display` implementation.
    pub(crate) const DISPLAY_BUFFER_SIZE: usize = 18;

    /// Format the `Time` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        let mut idx = 0;

        // Safety: `self.hour()` is in the range required by its type.
        let hour = one_to_two_digits_no_padding(
            unsafe { Hours::new_unchecked(self.hour()) }.expand(),
        );
        // Safety:
        // - both `hour` and `buf` are valid for reads and writes of up to 2 bytes.
        // - `u8` is 1-aligned, so that is not a concern.
        // - `hour` points to static memory, while `buf` is a local variable, so they do not
        //   overlap.
        unsafe {
            hour.as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), hour.len())
        };
        idx += hour.len();

        buf[idx] = MaybeUninit::new(b':');
        idx += 1;

        // Safety: See above.
        unsafe {
            two_digits_zero_padded(Minutes::new_unchecked(self.minute()).expand())
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), 2)
        };
        idx += 2;

        buf[idx] = MaybeUninit::new(b':');
        idx += 1;

        // Safety: See above.
        unsafe {
            two_digits_zero_padded(Seconds::new_unchecked(self.second()).expand())
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), 2)
        };
        idx += 2;

        buf[idx] = MaybeUninit::new(b'.');
        idx += 1;
        let subsecond = truncated_subsecond_from_nanos(unsafe {
            Nanoseconds::new_unchecked(self.nanosecond())
        });
        unsafe {
            subsecond
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(idx).cast(), subsecond.len())
        };
        idx += subsecond.len();

        idx
    }
}

impl fmt::Display for Time {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); Self::DISPLAY_BUFFER_SIZE];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts(buf.as_ptr().cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for Time {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Add<SignedDuration> for Time {
    type Output = Self;

    /// Add the sub-day time of the [`SignedDuration`] to the `Time`. Wraps on overflow.
    #[inline]
    fn add(self, duration: SignedDuration) -> Self::Output {
        self.adjusting_add(duration).1
    }
}

impl AddAssign<SignedDuration> for Time {
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        *self = *self + rhs;
    }
}

impl Add<StdDuration> for Time {
    type Output = Self;

    /// Add the sub-day time of the [`std::time::Duration`] to the `Time`. Wraps on overflow.
    #[inline]
    fn add(self, duration: StdDuration) -> Self::Output {
        self.adjusting_add_std(duration).1
    }
}

impl AddAssign<StdDuration> for Time {
    #[inline]
    fn add_assign(&mut self, rhs: StdDuration) {
        *self = *self + rhs;
    }
}

impl Sub<SignedDuration> for Time {
    type Output = Self;

    /// Subtract the sub-day time of the [`SignedDuration`] from the `Time`. Wraps on overflow.
    #[inline]
    fn sub(self, duration: SignedDuration) -> Self::Output {
        self.adjusting_sub(duration).1
    }
}

impl SubAssign<SignedDuration> for Time {
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        *self = *self - rhs;
    }
}

impl Sub<StdDuration> for Time {
    type Output = Self;

    /// Subtract the sub-day time of the [`std::time::Duration`] from the `Time`. Wraps on overflow.
    #[inline]
    fn sub(self, duration: StdDuration) -> Self::Output {
        self.adjusting_sub_std(duration).1
    }
}

impl SubAssign<StdDuration> for Time {
    #[inline]
    fn sub_assign(&mut self, rhs: StdDuration) {
        *self = *self - rhs;
    }
}

impl Sub for Time {
    type Output = SignedDuration;

    /// Subtract two `Time`s, returning the [`SignedDuration`] between. This assumes both `Time`s
    /// are in the same calendar day.
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let hour_diff = self.hour.get().cast_signed() - rhs.hour.get().cast_signed();
        let minute_diff =
            self.minute.get().cast_signed() - rhs.minute.get().cast_signed();
        let second_diff =
            self.second.get().cast_signed() - rhs.second.get().cast_signed();
        let nanosecond_diff =
            self.nanosecond.get().cast_signed() - rhs.nanosecond.get().cast_signed();

        let seconds = hour_diff.widen::<i32>() * Second::per_t::<i32>(Hour)
            + minute_diff.widen::<i32>() * Second::per_t::<i32>(Minute)
            + second_diff.widen::<i32>();

        let (seconds, nanoseconds) = if seconds > 0 && nanosecond_diff < 0 {
            (
                seconds - 1,
                nanosecond_diff + Nanosecond::per_t::<i32>(Second),
            )
        } else if seconds < 0 && nanosecond_diff > 0 {
            (
                seconds + 1,
                nanosecond_diff - Nanosecond::per_t::<i32>(Second),
            )
        } else {
            (seconds, nanosecond_diff)
        };

        // Safety: `nanoseconds` is in range due to the overflow handling.
        unsafe { SignedDuration::new_unchecked(seconds.widen(), nanoseconds) }
    }
}
