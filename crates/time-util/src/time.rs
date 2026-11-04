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

use crate::date::{MAX_YEAR, MIN_YEAR};
use crate::error::ComponentRange;
use crate::internal_macro::{cascade, ensure_ranged};
use crate::num_fmt::{
    one_to_two_digits_no_padding, str_from_raw_parts, truncated_subsecond_from_nanos, two_digits_zero_padded,
};
use crate::signed_duration::SignedDuration;
use crate::unit::{Day, Hour, Microsecond, Millisecond, Minute, Nanosecond, Second, Subsecond};
use crate::util::DateAdjustment;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as StdDuration;
use deranged::{ri32, ru8, ru32};
use num_conv::prelude::*;
use powerfmt::smart_display::{FormatterOptions, Metadata, SmartDisplay};

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
    pub const unsafe fn from_hms_nanos_unchecked(hour: u8, minute: u8, second: u8, nanosecond: u32) -> Self {
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
    pub const MIDNIGHT: Self =
        Self::from_hms_nanos_ranged(Hours::MIN, Minutes::MIN, Seconds::MIN, Nanoseconds::MIN);

    /// A `Time` that is one nanosecond before midnight. This is the largest possible value for a
    /// `Time`.
    pub const MAX: Self =
        Self::from_hms_nanos_ranged(Hours::MAX, Minutes::MAX, Seconds::MAX, Nanoseconds::MAX);

    /// Attempt to create a `Time` from the hour, minute, and second.
    #[inline]
    pub const fn from_hms(hour: u8, minute: u8, second: u8) -> Result<Self, ComponentRange> {
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
    pub(crate) const fn as_hms_nano_ranged(self) -> (Hours, Minutes, Seconds, Nanoseconds) {
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
        let mut nanoseconds = other.nanosecond.get().cast_signed() - self.nanosecond.get().cast_signed();
        let seconds = other.second.get().cast_signed() - self.second.get().cast_signed();
        let minutes = other.minute.get().cast_signed() - self.minute.get().cast_signed();
        let hours = other.hour.get().cast_signed() - self.hour.get().cast_signed();

        // Safety: For all four variables, the bounds are obviously true given the previous bounds
        // and nature of subtraction.
        unsafe {
            core::hint::assert_unchecked(
                nanoseconds >= Nanoseconds::MIN.get().cast_signed() - Nanoseconds::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                nanoseconds <= Nanoseconds::MAX.get().cast_signed() - Nanoseconds::MIN.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                seconds >= Seconds::MIN.get().cast_signed() - Seconds::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                seconds <= Seconds::MAX.get().cast_signed() - Seconds::MIN.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                minutes >= Minutes::MIN.get().cast_signed() - Minutes::MAX.get().cast_signed(),
            );
            core::hint::assert_unchecked(
                minutes <= Minutes::MAX.get().cast_signed() - Minutes::MIN.get().cast_signed(),
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
    pub(crate) const fn adjusting_add(self, duration: SignedDuration) -> (DateAdjustment, Self) {
        let mut nanoseconds = self.nanosecond.get().cast_signed() + duration.subsec_nanoseconds();
        let mut seconds =
            self.second.get().cast_signed() + (duration.whole_seconds() % Second::per_t::<i64>(Minute)) as i8;
        let mut minutes =
            self.minute.get().cast_signed() + (duration.whole_minutes() % Minute::per_t::<i64>(Hour)) as i8;
        let mut hours =
            self.hour.get().cast_signed() + (duration.whole_hours() % Hour::per_t::<i64>(Day)) as i8;
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
    pub(crate) const fn adjusting_sub(self, duration: SignedDuration) -> (DateAdjustment, Self) {
        let mut nanoseconds = self.nanosecond.get().cast_signed() - duration.subsec_nanoseconds();
        let mut seconds =
            self.second.get().cast_signed() - (duration.whole_seconds() % Second::per_t::<i64>(Minute)) as i8;
        let mut minutes =
            self.minute.get().cast_signed() - (duration.whole_minutes() % Minute::per_t::<i64>(Hour)) as i8;
        let mut hours =
            self.hour.get().cast_signed() - (duration.whole_hours() % Hour::per_t::<i64>(Day)) as i8;
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

    /// Add the sub-day time of the [`core::time::Duration`] to the `Time`. Wraps on overflow,
    /// returning whether the date is the previous date as the first element of the tuple.
    #[inline]
    pub(crate) const fn adjusting_add_std(self, duration: StdDuration) -> (bool, Self) {
        let mut nanosecond = self.nanosecond.get() + duration.subsec_nanos();
        let mut second = self.second.get() + (duration.as_secs() % Second::per_t::<u64>(Minute)) as u8;
        let mut minute = self.minute.get()
            + ((duration.as_secs() / Second::per_t::<u64>(Minute)) % Minute::per_t::<u64>(Hour)) as u8;
        let mut hour = self.hour.get()
            + ((duration.as_secs() / Second::per_t::<u64>(Hour)) % Hour::per_t::<u64>(Day)) as u8;
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

    /// Subtract the sub-day time of the [`core::time::Duration`] to the `Time`. Wraps on overflow,
    /// returning whether the date is the previous date as the first element of the tuple.
    #[inline]
    pub(crate) const fn adjusting_sub_std(self, duration: StdDuration) -> (bool, Self) {
        let mut nanosecond = self.nanosecond.get().cast_signed() - duration.subsec_nanos().cast_signed();
        let mut second =
            self.second.get().cast_signed() - (duration.as_secs() % Second::per_t::<u64>(Minute)) as i8;
        let mut minute = self.minute.get().cast_signed()
            - ((duration.as_secs() / Second::per_t::<u64>(Minute)) % Minute::per_t::<u64>(Hour)) as i8;
        let mut hour = self.hour.get().cast_signed()
            - ((duration.as_secs() / Second::per_t::<u64>(Hour)) % Hour::per_t::<u64>(Day)) as i8;
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
    pub const fn replace_millisecond(mut self, millisecond: u16) -> Result<Self, ComponentRange> {
        self.nanosecond =
            ensure_ranged!(Nanoseconds: millisecond as u32 * Nanosecond::per_t::<u32>(Millisecond));
        Ok(self)
    }

    /// Truncate the time to the millisecond, setting the microsecond and nanosecond components to
    /// zero.
    #[inline]
    pub const fn truncate_to_millisecond(mut self) -> Self {
        // Safety: Truncating to the millisecond will always produce a valid nanosecond.
        self.nanosecond = unsafe {
            Nanoseconds::new_unchecked(self.nanosecond.get() - (self.nanosecond.get() % 1_000_000))
        };
        self
    }

    /// Replace the microseconds within the second.
    #[inline]
    pub const fn replace_microsecond(mut self, microsecond: u32) -> Result<Self, ComponentRange> {
        self.nanosecond = ensure_ranged!(Nanoseconds: microsecond * Nanosecond::per_t::<u32>(Microsecond));
        Ok(self)
    }

    /// Truncate the time to the microsecond, setting the nanosecond component to zero.
    #[inline]
    pub const fn truncate_to_microsecond(mut self) -> Self {
        // Safety: Truncating to the microsecond will always produce a valid nanosecond.
        self.nanosecond =
            unsafe { Nanoseconds::new_unchecked(self.nanosecond.get() - (self.nanosecond.get() % 1_000)) };
        self
    }

    /// Replace the nanoseconds within the second.
    #[inline]
    pub const fn replace_nanosecond(mut self, nanosecond: u32) -> Result<Self, ComponentRange> {
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
    pub(crate) fn fmt_into_buffer(self, buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE]) -> usize {
        let mut idx = 0;

        // Safety: `self.hour()` is in the range required by its type.
        let hour = one_to_two_digits_no_padding(unsafe { Hours::new_unchecked(self.hour()) }.expand());
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
        let subsecond =
            truncated_subsecond_from_nanos(unsafe { Nanoseconds::new_unchecked(self.nanosecond()) });
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

    /// Add the sub-day time of the [`core::time::Duration`] to the `Time`. Wraps on overflow.
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

    /// Subtract the sub-day time of the [`core::time::Duration`] from the `Time`. Wraps on overflow.
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
        let minute_diff = self.minute.get().cast_signed() - rhs.minute.get().cast_signed();
        let second_diff = self.second.get().cast_signed() - rhs.second.get().cast_signed();
        let nanosecond_diff = self.nanosecond.get().cast_signed() - rhs.nanosecond.get().cast_signed();

        let seconds = hour_diff.widen::<i32>() * Second::per_t::<i32>(Hour)
            + minute_diff.widen::<i32>() * Second::per_t::<i32>(Minute)
            + second_diff.widen::<i32>();

        let (seconds, nanoseconds) = if seconds > 0 && nanosecond_diff < 0 {
            (seconds - 1, nanosecond_diff + Nanosecond::per_t::<i32>(Second))
        } else if seconds < 0 && nanosecond_diff > 0 {
            (seconds + 1, nanosecond_diff - Nanosecond::per_t::<i32>(Second))
        } else {
            (seconds, nanosecond_diff)
        };

        // Safety: `nanoseconds` is in range due to the overflow handling.
        unsafe { SignedDuration::new_unchecked(seconds.widen(), nanoseconds) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(hour: u8, minute: u8, second: u8) -> Time {
        Time::from_hms(hour, minute, second).expect("valid time")
    }

    #[test]
    fn constructors_build_expected_components() {
        let t = time(13, 45, 59);
        assert_eq!(t.as_hms(), (13, 45, 59));
        assert_eq!((t.hour(), t.minute(), t.second()), (13, 45, 59));
        assert_eq!(t.millisecond(), 0);
        assert_eq!(t.microsecond(), 0);
        assert_eq!(t.nanosecond(), 0);

        let t = Time::from_hms_milli(1, 2, 3, 456).expect("valid");
        assert_eq!(t.as_hms_milli(), (1, 2, 3, 456));
        assert_eq!(t.nanosecond(), 456_000_000);
        assert_eq!(t.as_hms_nano(), (1, 2, 3, 456_000_000));

        let t = Time::from_hms_micro(1, 2, 3, 456_789).expect("valid");
        assert_eq!(t.as_hms_micro(), (1, 2, 3, 456_789));
        assert_eq!(t.millisecond(), 456);
        assert_eq!(t.nanosecond(), 456_789_000);

        let t = Time::from_hms_nano(1, 2, 3, 456_789_012).expect("valid");
        assert_eq!(t.as_hms_nano(), (1, 2, 3, 456_789_012));
        assert_eq!(t.microsecond(), 456_789);
        assert_eq!(t.millisecond(), 456);

        assert_eq!(Time::MIDNIGHT.as_hms(), (0, 0, 0));
        assert_eq!(Time::MIDNIGHT.nanosecond(), 0);
        assert_eq!(Time::MAX.as_hms(), (23, 59, 59));
        assert_eq!(Time::MAX.nanosecond(), 999_999_999);
        assert_eq!(Time::MAX.millisecond(), 999);
    }

    #[test]
    fn constructors_reject_out_of_range_components() {
        let error = Time::from_hms(24, 0, 0).expect_err("must be rejected");
        assert_eq!(error.name(), "hour");
        assert!(!error.is_conditional());
        for hour in [25u8, 100, 255] {
            assert!(Time::from_hms(hour, 0, 0).is_err(), "hour {hour}");
        }
        for minute in [60u8, 61, 255] {
            let error = Time::from_hms(0, minute, 0).expect_err("must be rejected");
            assert_eq!(error.name(), "minute");
            assert!(!error.is_conditional());
        }
        for second in [60u8, 61, 255] {
            let error = Time::from_hms(0, 0, second).expect_err("must be rejected");
            assert_eq!(error.name(), "second");
            assert!(!error.is_conditional());
        }
        let error = Time::from_hms_milli(0, 0, 0, 1_000).expect_err("must be rejected");
        assert_eq!(error.name(), "millisecond");
        assert!(!error.is_conditional());
        let error = Time::from_hms_micro(0, 0, 0, 1_000_000).expect_err("must be rejected");
        assert_eq!(error.name(), "microsecond");
        let error = Time::from_hms_nano(0, 0, 0, 1_000_000_000).expect_err("must be rejected");
        assert_eq!(error.name(), "nanosecond");

        // Boundary values are accepted.
        assert!(Time::from_hms(23, 59, 59).is_ok());
        assert!(Time::from_hms_milli(0, 0, 0, 999).is_ok());
        assert!(Time::from_hms_micro(0, 0, 0, 999_999).is_ok());
        assert!(Time::from_hms_nano(0, 0, 0, 999_999_999).is_ok());
    }

    #[test]
    fn replace_components_updates_only_that_field() {
        let t = time(1, 2, 3)
            .replace_nanosecond(456_789_012)
            .expect("valid");
        assert_eq!(t.as_hms_nano(), (1, 2, 3, 456_789_012));

        assert!(t.replace_hour(24).is_err());
        let t = t.replace_hour(20).expect("valid");
        assert_eq!(t.hour(), 20);
        assert!(t.replace_minute(60).is_err());
        let t = t.replace_minute(30).expect("valid");
        assert_eq!(t.minute(), 30);
        assert!(t.replace_second(60).is_err());
        let t = t.replace_second(45).expect("valid");
        assert_eq!(t.second(), 45);

        let t = t.replace_millisecond(999).expect("valid");
        assert_eq!(t.nanosecond(), 999_000_000);
        assert!(t.replace_millisecond(1_000).is_err());
        let t = t.replace_microsecond(999_999).expect("valid");
        assert_eq!(t.nanosecond(), 999_999_000);
        assert!(t.replace_microsecond(1_000_000).is_err());
        let t = t.replace_nanosecond(999_999_999).expect("valid");
        assert_eq!(t.nanosecond(), 999_999_999);
        assert_eq!(t.as_hms_nano(), (20, 30, 45, 999_999_999));
        assert!(t.replace_nanosecond(1_000_000_000).is_err());
        assert!(t.replace_nanosecond(0).is_ok());
    }

    #[test]
    fn truncate_clears_lower_components() {
        let t = Time::from_hms_nano(13, 45, 59, 987_654_321).expect("valid");

        let truncated = t.truncate_to_hour();
        assert_eq!(truncated.as_hms_nano(), (13, 0, 0, 0));

        let truncated = t.truncate_to_minute();
        assert_eq!(truncated.as_hms_nano(), (13, 45, 0, 0));

        let truncated = t.truncate_to_second();
        assert_eq!(truncated.as_hms_nano(), (13, 45, 59, 0));

        let truncated = t.truncate_to_millisecond();
        assert_eq!(truncated.nanosecond(), 987_000_000);
        assert_eq!(truncated.as_hms(), (13, 45, 59));

        let truncated = t.truncate_to_microsecond();
        assert_eq!(truncated.nanosecond(), 987_654_000);

        // Truncating an already-truncated time is a no-op.
        assert_eq!(truncated.truncate_to_microsecond(), truncated);
        assert_eq!(Time::MIDNIGHT.truncate_to_hour(), Time::MIDNIGHT);
        assert_eq!(Time::MAX.truncate_to_second(), time(23, 59, 59));
    }

    #[test]
    fn display_formats_times_without_hour_padding() {
        let cases = [
            (Time::MIDNIGHT, "0:00:00.0"),
            (time(13, 45, 59), "13:45:59.0"),
            (time(1, 2, 3), "1:02:03.0"),
            (Time::MAX, "23:59:59.999999999"),
            (
                Time::from_hms_nano(0, 0, 0, 10).expect("valid"),
                "0:00:00.00000001",
            ),
            (
                Time::from_hms_nano(0, 0, 0, 500_000_000).expect("valid"),
                "0:00:00.5",
            ),
            (
                Time::from_hms_nano(9, 0, 0, 1_000_000).expect("valid"),
                "9:00:00.001",
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(value.to_string(), expected, "Display of {value:?}");
            assert_eq!(format!("{value:?}"), expected, "Debug matches Display");
            assert_eq!(
                value
                    .metadata(FormatterOptions::default())
                    .unpadded_width(),
                expected.len(),
                "metadata width for {expected}",
            );
        }

        let value = time(13, 45, 59);
        assert_eq!(format!("{value:>14}"), "    13:45:59.0");
        assert_eq!(format!("{value:<14}"), "13:45:59.0    ");
        assert_eq!(format!("{value:.5}"), "13:45");
    }

    #[test]
    fn times_are_ordered_and_hashable() {
        use core::hash::{Hash, Hasher};

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let midnight = Time::MIDNIGHT;
        let morning = time(1, 0, 0);
        let noon = time(12, 0, 0);
        let max = Time::MAX;

        assert!(midnight < morning);
        assert!(morning < noon);
        assert!(noon < max);
        assert!(max > midnight);
        assert_eq!(midnight, Time::from_hms(0, 0, 0).expect("valid"));
        assert_eq!(hash_of(&midnight), hash_of(&Time::MIDNIGHT));
        assert_ne!(hash_of(&noon), hash_of(&morning));
        assert_eq!(midnight.cmp(&noon), core::cmp::Ordering::Less);
        assert_eq!(max.cmp(&midnight), core::cmp::Ordering::Greater);
        assert_eq!(noon.cmp(&noon), core::cmp::Ordering::Equal);
        // `MIDNIGHT` is the minimum and `MAX` the maximum representable time.
        assert_eq!(
            Time::MIDNIGHT.as_u64(),
            Time::MIDNIGHT.as_u64().min(Time::MAX.as_u64())
        );
    }

    #[test]
    fn duration_until_wraps_within_a_day() {
        let midnight = Time::MIDNIGHT;
        let noon = time(12, 0, 0);
        let quarter = time(6, 0, 0);

        assert_eq!(midnight.duration_until(noon), SignedDuration::hours(12));
        assert_eq!(noon.duration_since(midnight), SignedDuration::hours(12));
        assert_eq!(midnight.duration_until(midnight), SignedDuration::hours(0));
        assert_eq!(midnight.duration_until(quarter), SignedDuration::hours(6));
        // Walking backwards wraps forward into the same day.
        assert_eq!(noon.duration_until(midnight), SignedDuration::hours(12));
        assert_eq!(quarter.duration_until(midnight), SignedDuration::hours(18));

        // For distinct times, the pair of durations in both directions covers exactly one
        // day; identical times yield zero in both directions.
        for a in [midnight, quarter, noon, time(23, 0, 0), Time::MAX] {
            for b in [midnight, quarter, noon, time(23, 59, 59), Time::MAX] {
                let forward = a.duration_until(b);
                let backward = b.duration_until(a);
                assert!(forward >= SignedDuration::hours(0));
                assert!(forward < SignedDuration::days(1));
                if a == b {
                    assert_eq!(forward, SignedDuration::hours(0));
                    assert_eq!(backward, SignedDuration::hours(0));
                } else {
                    assert_eq!(forward + backward, SignedDuration::days(1));
                }
            }
        }

        // Sub-second precision is preserved.
        let a = Time::from_hms_nano(0, 0, 0, 1).expect("valid");
        let b = Time::MIDNIGHT;
        assert_eq!(b.duration_until(a), SignedDuration::nanoseconds(1));
        assert_eq!(
            a.duration_until(b),
            SignedDuration::days(1) - SignedDuration::nanoseconds(1)
        );
    }

    #[test]
    fn subtraction_returns_signed_durations() {
        let noon = time(12, 0, 0);
        let midnight = Time::MIDNIGHT;
        let quarter = time(6, 0, 0);

        assert_eq!(noon - midnight, SignedDuration::hours(12));
        assert_eq!(midnight - noon, SignedDuration::hours(-12));
        assert_eq!(noon - noon, SignedDuration::hours(0));
        assert_eq!(quarter - midnight, SignedDuration::hours(6));
        assert_eq!(midnight - quarter, SignedDuration::hours(-6));

        // Mixing positive second and negative nanosecond components.
        let a = Time::from_hms_nano(12, 0, 0, 0).expect("valid");
        let b = Time::from_hms_nano(11, 59, 59, 500_000_000).expect("valid");
        assert_eq!(a - b, SignedDuration::nanoseconds(500_000_000));
        assert_eq!(b - a, SignedDuration::nanoseconds(-500_000_000));

        let a = Time::from_hms_nano(11, 59, 59, 0).expect("valid");
        let b = Time::from_hms_nano(12, 0, 0, 500_000_000).expect("valid");
        assert_eq!(b - a, SignedDuration::nanoseconds(1_500_000_000));
        assert_eq!(a - b, SignedDuration::nanoseconds(-1_500_000_000));

        // Consistency with duration_until for the non-wrapping direction.
        assert_eq!(noon - quarter, quarter.duration_until(noon));
    }

    #[test]
    fn addition_wraps_around_the_day() {
        let t = time(23, 30, 0);
        assert_eq!(
            t + SignedDuration::hours(1),
            Time::MIDNIGHT + SignedDuration::minutes(30)
        );
        assert_eq!(t + SignedDuration::hours(-1), time(22, 30, 0));
        assert_eq!(Time::MIDNIGHT - SignedDuration::hours(1), time(23, 0, 0),);

        // Whole-day components of a duration do not affect the time of day.
        assert_eq!(t + SignedDuration::days(1), t);
        assert_eq!(t - SignedDuration::days(3), t);
        assert_eq!(t + SignedDuration::days(-1), t);

        // Addition and subtraction are inverse operations within the day.
        let duration = SignedDuration::hours(5) + SignedDuration::minutes(30);
        assert_eq!((t + duration) - duration, t);
        assert_eq!((t - duration) + duration, t);

        // Standard durations wrap the same way.
        assert_eq!(
            t + StdDuration::from_secs(3_600),
            Time::MIDNIGHT + SignedDuration::minutes(30),
        );
        assert_eq!(t - StdDuration::from_secs(3_600), time(22, 30, 0),);
        assert_eq!(t + StdDuration::from_secs(86_400), t);

        // Assign operators delegate to the operator implementations.
        let mut value = t;
        value += SignedDuration::hours(1);
        assert_eq!(value, Time::MIDNIGHT + SignedDuration::minutes(30));
        value -= SignedDuration::hours(1);
        assert_eq!(value, t);
        value += StdDuration::from_secs(3_600);
        assert_eq!(value, Time::MIDNIGHT + SignedDuration::minutes(30));
        value -= StdDuration::from_secs(3_600);
        assert_eq!(value, t);
    }

    #[test]
    fn adjusting_add_reports_date_adjustment() {
        let t = time(23, 0, 0);
        let (adjustment, next) = t.adjusting_add(SignedDuration::hours(1));
        assert!(matches!(adjustment, DateAdjustment::Next));
        assert_eq!(next, Time::MIDNIGHT);

        let (adjustment, previous) = Time::MIDNIGHT.adjusting_add(SignedDuration::hours(-1));
        assert!(matches!(adjustment, DateAdjustment::Previous));
        assert_eq!(previous, time(23, 0, 0));

        let (adjustment, unchanged) = t.adjusting_add(SignedDuration::minutes(30));
        assert!(matches!(adjustment, DateAdjustment::None));
        assert_eq!(unchanged, time(23, 30, 0));

        // Days are ignored entirely: no date adjustment is reported.
        let (adjustment, unchanged) = t.adjusting_add(SignedDuration::days(100));
        assert!(matches!(adjustment, DateAdjustment::None));
        assert_eq!(unchanged, t);

        // Crossing midnight in the subtractive direction.
        let (adjustment, previous) = time(1, 0, 0).adjusting_sub(SignedDuration::hours(2));
        assert!(matches!(adjustment, DateAdjustment::Previous));
        assert_eq!(previous, time(23, 0, 0));

        let (adjustment, next) = time(23, 0, 0).adjusting_sub(SignedDuration::hours(-2));
        assert!(matches!(adjustment, DateAdjustment::Next));
        assert_eq!(next, time(1, 0, 0));

        // Standard durations only ever move forward.
        let (is_next_day, next) = time(23, 0, 0).adjusting_add_std(StdDuration::from_secs(3_600));
        assert!(is_next_day);
        assert_eq!(next, Time::MIDNIGHT);
        let (is_next_day, unchanged) = time(1, 0, 0).adjusting_add_std(StdDuration::from_secs(3_600));
        assert!(!is_next_day);
        assert_eq!(unchanged, time(2, 0, 0));
        let (is_previous_day, previous) = Time::MIDNIGHT.adjusting_sub_std(StdDuration::from_secs(3_600));
        assert!(is_previous_day);
        assert_eq!(previous, time(23, 0, 0));
        let (is_previous_day, unchanged) = time(1, 0, 0).adjusting_sub_std(StdDuration::from_secs(3_600));
        assert!(!is_previous_day);
        assert_eq!(unchanged, Time::MIDNIGHT);
    }

    #[test]
    fn as_u64_reflects_temporal_order() {
        let values = [
            Time::MIDNIGHT,
            time(0, 0, 1),
            time(0, 1, 0),
            time(1, 0, 0),
            time(12, 0, 0),
            time(23, 59, 59),
            Time::MAX,
        ];
        for window in values.windows(2) {
            assert!(window[0].as_u64() < window[1].as_u64());
            assert!(window[0] < window[1]);
        }
        // Equal times compare equally at the bit level too.
        assert_eq!(time(12, 30, 45).as_u64(), time(12, 30, 45).as_u64());
    }
}
