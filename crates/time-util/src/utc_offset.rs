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

use crate::error::{ComponentRange, Error, IndeterminateOffset};
use crate::formattable::Formattable;
use crate::internal_macro::ensure_ranged;
use crate::num_fmt::{str_from_raw_parts, two_digits_zero_padded};
use crate::unit::*;
use alloc::string::String;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::Neg;
use deranged::{ri8, ri32, ru8};

/// The type of the `hours` field of `UtcOffset`.
pub(crate) type Hours = ri8<-25, 25>;
/// The type of the `minutes` field of `UtcOffset`.
pub(crate) type Minutes = ri8<{ -(Minute::per_t::<i8>(Hour) - 1) }, { Minute::per_t::<i8>(Hour) - 1 }>;
/// The type of the `seconds` field of `UtcOffset`.
pub(crate) type Seconds = ri8<{ -(Second::per_t::<i8>(Minute) - 1) }, { Second::per_t::<i8>(Minute) - 1 }>;
/// The type capable of storing the range of whole seconds that a `UtcOffset` can encompass.
type WholeSeconds = ri32<
    {
        Hours::MIN.get() as i32 * Second::per_t::<i32>(Hour)
            + Minutes::MIN.get() as i32 * Second::per_t::<i32>(Minute)
            + Seconds::MIN.get() as i32
    },
    {
        Hours::MAX.get() as i32 * Second::per_t::<i32>(Hour)
            + Minutes::MAX.get() as i32 * Second::per_t::<i32>(Minute)
            + Seconds::MAX.get() as i32
    },
>;

/// An offset from UTC.
///
/// This struct can store values up to ±25:59:59. If you need support outside this range, please
/// file an issue with your use case.
// All three components _must_ have the same sign.
#[derive(Clone, Copy, Eq)]
#[cfg_attr(not(docsrs), repr(C))]
pub struct UtcOffset {
    #[cfg(target_endian = "little")]
    seconds: Seconds,
    #[cfg(target_endian = "little")]
    minutes: Minutes,
    #[cfg(target_endian = "little")]
    hours: Hours,

    #[cfg(target_endian = "big")]
    hours: Hours,
    #[cfg(target_endian = "big")]
    minutes: Minutes,
    #[cfg(target_endian = "big")]
    seconds: Seconds,
}

impl Hash for UtcOffset {
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_u32(self.as_u32_for_equality());
    }
}

impl PartialEq for UtcOffset {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_u32_for_equality()
            .eq(&other.as_u32_for_equality())
    }
}

impl PartialOrd for UtcOffset {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UtcOffset {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_i32_for_comparison()
            .cmp(&other.as_i32_for_comparison())
    }
}

impl UtcOffset {
    /// Provide a representation of the `UtcOffset` as a `i32`. This value can be used for equality,
    /// and hashing. This value is not suitable for ordering; use `as_i32_for_comparison` instead.
    #[inline]
    pub(crate) const fn as_u32_for_equality(self) -> u32 {
        // Safety: Size and alignment are handled by the compiler. Both the source and destination
        // types are plain old data (POD) types.
        unsafe {
            if const { cfg!(target_endian = "little") } {
                core::mem::transmute::<[i8; 4], u32>([
                    self.seconds.get(),
                    self.minutes.get(),
                    self.hours.get(),
                    0,
                ])
            } else {
                core::mem::transmute::<[i8; 4], u32>([
                    self.hours.get(),
                    self.minutes.get(),
                    self.seconds.get(),
                    0,
                ])
            }
        }
    }

    /// Provide a representation of the `UtcOffset` as a `i32`. This value can be used for ordering.
    /// While it is suitable for equality, `as_u32_for_equality` is preferred for performance
    /// reasons.
    #[inline]
    const fn as_i32_for_comparison(self) -> i32 {
        // The bitwise `or` of sign-extended components is not a valid ordering key for negative
        // offsets (every negative component sign-extends into the same high bits), so the offset is
        // reduced to its total in whole seconds instead.
        (self.hours.get() as i32) * Second::per_t::<i32>(Hour)
            + (self.minutes.get() as i32) * Second::per_t::<i32>(Minute)
            + self.seconds.get() as i32
    }

    /// A `UtcOffset` that is UTC.
    pub const UTC: Self = Self::from_whole_seconds_ranged(WholeSeconds::new_static::<0>());

    /// Create a `UtcOffset` representing an offset of the hours, minutes, and seconds provided, the
    /// validity of which must be guaranteed by the caller. All three parameters must have the same
    /// sign.
    ///
    /// # Safety
    ///
    /// - Hours must be in the range `-25..=25`.
    /// - Minutes must be in the range `-59..=59`.
    /// - Seconds must be in the range `-59..=59`.
    ///
    /// While the signs of the parameters are required to match to avoid bugs, this is not a safety
    /// invariant.
    #[doc(hidden)]
    #[inline]
    pub const unsafe fn from_hms_unchecked(hours: i8, minutes: i8, seconds: i8) -> Self {
        // Safety: The caller must uphold the safety invariants.
        unsafe {
            Self::from_hms_ranged_unchecked(
                Hours::new_unchecked(hours),
                Minutes::new_unchecked(minutes),
                Seconds::new_unchecked(seconds),
            )
        }
    }

    /// Create a `UtcOffset` representing an offset by the number of hours, minutes, and seconds
    /// provided.
    ///
    /// The sign of all three components should match. If they do not, all smaller components will
    /// have their signs flipped.
    #[inline]
    pub const fn from_hms(hours: i8, minutes: i8, seconds: i8) -> Result<Self, ComponentRange> {
        Ok(Self::from_hms_ranged(
            ensure_ranged!(Hours: hours("offset hour")),
            ensure_ranged!(Minutes: minutes("offset minute")),
            ensure_ranged!(Seconds: seconds("offset second")),
        ))
    }

    /// Create a `UtcOffset` representing an offset of the hours, minutes, and seconds provided. All
    /// three parameters must have the same sign.
    ///
    /// While the signs of the parameters are required to match, this is not a safety invariant.
    #[inline]
    pub(crate) const fn from_hms_ranged_unchecked(hours: Hours, minutes: Minutes, seconds: Seconds) -> Self {
        if hours.get() < 0 {
            debug_assert!(minutes.get() <= 0);
            debug_assert!(seconds.get() <= 0);
        } else if hours.get() > 0 {
            debug_assert!(minutes.get() >= 0);
            debug_assert!(seconds.get() >= 0);
        }
        if minutes.get() < 0 {
            debug_assert!(seconds.get() <= 0);
        } else if minutes.get() > 0 {
            debug_assert!(seconds.get() >= 0);
        }

        Self {
            hours,
            minutes,
            seconds,
        }
    }

    /// Create a `UtcOffset` representing an offset by the number of hours, minutes, and seconds
    /// provided.
    ///
    /// The sign of all three components should match. If they do not, all smaller components will
    /// have their signs flipped.
    #[inline]
    pub(crate) const fn from_hms_ranged(hours: Hours, mut minutes: Minutes, mut seconds: Seconds) -> Self {
        if (hours.get() > 0 && minutes.get() < 0) || (hours.get() < 0 && minutes.get() > 0) {
            minutes = minutes.neg();
        }
        if (hours.get() > 0 && seconds.get() < 0)
            || (hours.get() < 0 && seconds.get() > 0)
            || (minutes.get() > 0 && seconds.get() < 0)
            || (minutes.get() < 0 && seconds.get() > 0)
        {
            seconds = seconds.neg();
        }

        Self {
            hours,
            minutes,
            seconds,
        }
    }

    /// Create a `UtcOffset` representing an offset by the number of seconds provided.
    #[inline]
    pub const fn from_whole_seconds(seconds: i32) -> Result<Self, ComponentRange> {
        Ok(Self::from_whole_seconds_ranged(
            ensure_ranged!(WholeSeconds: seconds),
        ))
    }

    /// Create a `UtcOffset` representing an offset by the number of seconds provided.
    #[inline]
    pub(crate) const fn from_whole_seconds_ranged(seconds: WholeSeconds) -> Self {
        // Safety: The type of `seconds` guarantees that all values are in range.
        unsafe {
            Self::from_hms_unchecked(
                (seconds.get() / Second::per_t::<i32>(Hour)) as i8,
                ((seconds.get() % Second::per_t::<i32>(Hour)) / Minute::per_t::<i32>(Hour)) as i8,
                (seconds.get() % Second::per_t::<i32>(Minute)) as i8,
            )
        }
    }

    /// Obtain the UTC offset as its hours, minutes, and seconds. The sign of all three components
    /// will always match. A positive value indicates an offset to the east; a negative to the west.
    #[inline]
    pub const fn as_hms(self) -> (i8, i8, i8) {
        (self.hours.get(), self.minutes.get(), self.seconds.get())
    }

    /// Obtain the UTC offset as its hours, minutes, and seconds. The sign of all three components
    /// will always match. A positive value indicates an offset to the east; a negative to the west.
    #[inline]
    pub(crate) const fn as_hms_ranged(self) -> (Hours, Minutes, Seconds) {
        (self.hours, self.minutes, self.seconds)
    }

    /// Obtain the number of whole hours the offset is from UTC. A positive value indicates an
    /// offset to the east; a negative to the west.
    #[inline]
    pub const fn whole_hours(self) -> i8 {
        self.hours.get()
    }

    /// Obtain the number of whole minutes the offset is from UTC. A positive value indicates an
    /// offset to the east; a negative to the west.
    #[inline]
    pub const fn whole_minutes(self) -> i16 {
        self.hours.get() as i16 * Minute::per_t::<i16>(Hour) + self.minutes.get() as i16
    }

    /// Obtain the number of minutes past the hour the offset is from UTC. A positive value
    /// indicates an offset to the east; a negative to the west.
    #[inline]
    pub const fn minutes_past_hour(self) -> i8 {
        self.minutes.get()
    }

    /// Obtain the number of whole seconds the offset is from UTC. A positive value indicates an
    /// offset to the east; a negative to the west.
    #[inline]
    pub const fn whole_seconds(self) -> i32 {
        self.hours.get() as i32 * Second::per_t::<i32>(Hour)
            + self.minutes.get() as i32 * Second::per_t::<i32>(Minute)
            + self.seconds.get() as i32
    }

    /// Obtain the number of seconds past the minute the offset is from UTC. A positive value
    /// indicates an offset to the east; a negative to the west.
    #[inline]
    pub const fn seconds_past_minute(self) -> i8 {
        self.seconds.get()
    }

    /// Check if the offset is exactly UTC.
    #[inline]
    pub const fn is_utc(self) -> bool {
        self.as_u32_for_equality() == Self::UTC.as_u32_for_equality()
    }

    /// Check if the offset is positive, or east of UTC.
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.as_i32_for_comparison() > Self::UTC.as_i32_for_comparison()
    }

    /// Check if the offset is negative, or west of UTC.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.as_i32_for_comparison() < Self::UTC.as_i32_for_comparison()
    }

    /// Determine the local UTC offset for a supplied time value.
    #[inline]
    pub fn local_offset_at<T>(_time: T) -> Result<Self, IndeterminateOffset> {
        Ok(Self::UTC)
    }
}

impl UtcOffset {
    /// Format the `UtcOffset` using the provided [format description](crate::format_description).
    #[inline]
    pub fn format_into(
        self,
        output: &mut (impl fmt::Write + ?Sized),
        format: &(impl Formattable + ?Sized),
    ) -> Result<usize, Error> {
        format.format_into(output, &self, &mut Default::default())
    }

    /// Format the `UtcOffset` using the provided [format description](crate::format_description).
    #[inline]
    pub fn format(self, format: &(impl Formattable + ?Sized)) -> Result<String, Error> {
        format.format(&self, &mut Default::default())
    }
}

impl UtcOffset {
    /// The maximum number of bytes that the `fmt_into_buffer` method will write, which is also used
    /// for the `Display` implementation.
    pub(crate) const DISPLAY_BUFFER_SIZE: usize = 9;

    /// Format the `UtcOffset` into the provided buffer, returning the number of bytes written.
    #[inline]
    pub(crate) const fn fmt_into_buffer(
        self,
        buf: &mut [MaybeUninit<u8>; Self::DISPLAY_BUFFER_SIZE],
    ) -> usize {
        let hours = self.hours.get().unsigned_abs();
        let minutes = self.minutes.get().unsigned_abs();
        let seconds = self.seconds.get().unsigned_abs();

        let sign = if self.is_negative() { b'-' } else { b'+' };
        buf[0] = MaybeUninit::new(sign);
        buf[3] = MaybeUninit::new(b':');
        buf[6] = MaybeUninit::new(b':');

        // Safety: `hours`, `minutes` and `seconds` are all less than 100. Both the source and
        // destination are valid for two bytes, aligned, and do not overlap.
        unsafe {
            two_digits_zero_padded(ru8::new_unchecked(hours))
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(1).cast(), 2);
            two_digits_zero_padded(ru8::new_unchecked(minutes))
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(4).cast(), 2);
            two_digits_zero_padded(ru8::new_unchecked(seconds))
                .as_ptr()
                .copy_to_nonoverlapping(buf.as_mut_ptr().add(7).cast(), 2);
        }
        // The number of bytes written does not vary; it is always 9.
        9
    }
}

impl fmt::Display for UtcOffset {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::uninit(); Self::DISPLAY_BUFFER_SIZE];
        let len = self.fmt_into_buffer(&mut buf);
        // Safety: All bytes up to `len` have been initialized with ASCII characters.
        let s = unsafe { str_from_raw_parts(buf.as_ptr().cast(), len) };
        f.pad(s)
    }
}

impl fmt::Debug for UtcOffset {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Neg for UtcOffset {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_hms_ranged(self.hours.neg(), self.minutes.neg(), self.seconds.neg())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offset(hours: i8, minutes: i8, seconds: i8) -> UtcOffset {
        UtcOffset::from_hms(hours, minutes, seconds).expect("valid offset")
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn utc_is_the_zero_offset() {
        let utc = UtcOffset::UTC;
        assert!(utc.is_utc());
        assert!(!utc.is_positive());
        assert!(!utc.is_negative());
        assert_eq!(utc.as_hms(), (0, 0, 0));
        assert_eq!(utc.whole_hours(), 0);
        assert_eq!(utc.whole_minutes(), 0);
        assert_eq!(utc.minutes_past_hour(), 0);
        assert_eq!(utc.whole_seconds(), 0);
        assert_eq!(utc.seconds_past_minute(), 0);
        assert_eq!(utc.to_string(), "+00:00:00");
        assert_eq!(utc, offset(0, 0, 0));
        assert_eq!(utc, UtcOffset::from_whole_seconds(0).expect("valid offset"));
    }

    #[test]
    fn from_hms_validates_ranges() {
        // Hours are limited to ±25.
        assert!(UtcOffset::from_hms(25, 59, 59).is_ok());
        assert!(UtcOffset::from_hms(-25, -59, -59).is_ok());
        let err = UtcOffset::from_hms(26, 0, 0).expect_err("hour out of range");
        assert_eq!(err.name(), "offset hour");
        assert!(!err.is_conditional());
        assert_eq!(err.to_string(), "offset hour was not in range");
        assert_eq!(
            UtcOffset::from_hms(-26, 0, 0)
                .expect_err("hour out of range")
                .name(),
            "offset hour",
        );

        // Minutes and seconds are limited to ±59.
        let err = UtcOffset::from_hms(0, 60, 0).expect_err("minute out of range");
        assert_eq!(err.name(), "offset minute");
        assert!(!err.is_conditional());
        let err = UtcOffset::from_hms(0, -60, 0).expect_err("minute out of range");
        assert_eq!(err.name(), "offset minute");
        let err = UtcOffset::from_hms(0, 0, 60).expect_err("second out of range");
        assert_eq!(err.name(), "offset second");
        assert!(!err.is_conditional());
        assert_eq!(
            UtcOffset::from_hms(0, 0, -60)
                .expect_err("second out of range")
                .name(),
            "offset second",
        );
    }

    #[test]
    fn mismatched_signs_are_normalized_to_the_hour_sign() {
        // The hour component decides the sign of the whole offset.
        assert_eq!(offset(1, -2, -3), offset(1, 2, 3));
        assert_eq!(offset(-1, 2, 3), offset(-1, -2, -3));
        assert_eq!(offset(1, -2, 3), offset(1, 2, 3));
        assert_eq!(offset(-1, 2, -3), offset(-1, -2, -3));

        // Without an hour, the minute component decides.
        assert_eq!(offset(0, 2, -3), offset(0, 2, 3));
        assert_eq!(offset(0, -2, 3), offset(0, -2, -3));

        // Every component shares the sign of the overall offset afterwards.
        for value in [
            offset(1, -2, -3),
            offset(-1, 2, 3),
            offset(1, -2, 3),
            offset(-1, 2, -3),
            offset(0, 2, -3),
            offset(0, -2, 3),
            offset(0, 0, -1),
            offset(0, 0, 1),
        ] {
            let (hours, minutes, seconds) = value.as_hms();
            let sign = value.whole_seconds().signum();
            assert!(i32::from(hours) * sign >= 0, "hour sign of {value}");
            assert!(i32::from(minutes) * sign >= 0, "minute sign of {value}");
            assert!(i32::from(seconds) * sign >= 0, "second sign of {value}");
            assert!(sign != 0 || value.is_utc());
        }
    }

    #[test]
    fn from_whole_seconds_splits_into_hms() {
        let cases = [
            (0, (0, 0, 0), "+00:00:00"),
            (1, (0, 0, 1), "+00:00:01"),
            (59, (0, 0, 59), "+00:00:59"),
            (60, (0, 1, 0), "+00:01:00"),
            (3_599, (0, 59, 59), "+00:59:59"),
            (3_600, (1, 0, 0), "+01:00:00"),
            (3_661, (1, 1, 1), "+01:01:01"),
            (93_599, (25, 59, 59), "+25:59:59"),
            (-1, (0, 0, -1), "-00:00:01"),
            (-60, (0, -1, 0), "-00:01:00"),
            (-3_661, (-1, -1, -1), "-01:01:01"),
            (-93_599, (-25, -59, -59), "-25:59:59"),
        ];
        for (seconds, hms, display) in cases {
            let value = UtcOffset::from_whole_seconds(seconds).expect("valid offset");
            assert_eq!(value.as_hms(), hms, "hms for {seconds}");
            assert_eq!(value.whole_seconds(), seconds, "whole seconds for {seconds}");
            assert_eq!(value.to_string(), display, "display for {seconds}");
            assert_eq!(value.is_negative(), seconds < 0, "sign for {seconds}",);
        }

        // Out of range values are rejected.
        let err = UtcOffset::from_whole_seconds(93_600).expect_err("out of range");
        assert_eq!(err.name(), "seconds");
        assert!(!err.is_conditional());
        assert_eq!(
            UtcOffset::from_whole_seconds(-93_600)
                .expect_err("out of range")
                .name(),
            "seconds",
        );
    }

    #[test]
    fn component_accessors_accumulate_correctly() {
        let value = offset(1, 30, 45);
        assert_eq!(value.whole_hours(), 1);
        assert_eq!(value.minutes_past_hour(), 30);
        assert_eq!(value.seconds_past_minute(), 45);
        assert_eq!(value.whole_minutes(), 90);
        assert_eq!(value.whole_seconds(), 3600 + 1_800 + 45);

        let value = offset(-1, -30, -45);
        assert_eq!(value.whole_hours(), -1);
        assert_eq!(value.minutes_past_hour(), -30);
        assert_eq!(value.seconds_past_minute(), -45);
        assert_eq!(value.whole_minutes(), -90);
        assert_eq!(value.whole_seconds(), -(3600 + 1_800 + 45));

        // The largest magnitude offset.
        let value = offset(25, 59, 59);
        assert_eq!(value.whole_minutes(), 25 * 60 + 59);
        assert_eq!(value.whole_seconds(), 25 * 3600 + 59 * 60 + 59);
        assert!(value.is_positive());
        assert!(!value.is_negative());

        assert!(offset(0, 0, 1).is_positive());
        assert!(offset(0, 0, -1).is_negative());
        assert!(!offset(0, 0, 1).is_negative());
        assert!(!offset(0, 0, -1).is_positive());
    }

    #[test]
    fn display_formats_sign_and_zero_padded_components() {
        let cases = [
            (offset(0, 0, 0), "+00:00:00"),
            (offset(5, 0, 0), "+05:00:00"),
            (offset(-5, 0, 0), "-05:00:00"),
            (offset(1, 2, 3), "+01:02:03"),
            (offset(-1, -2, -3), "-01:02:03"),
            (offset(25, 59, 59), "+25:59:59"),
            (offset(-25, -59, -59), "-25:59:59"),
            (offset(0, 0, -1), "-00:00:01"),
        ];
        for (value, expected) in cases {
            assert_eq!(value.to_string(), expected, "Display of {value:?}");
            // Debug delegates to Display.
            assert_eq!(format!("{value:?}"), expected, "Debug of {value:?}");
            assert_eq!(format!("{value:>12}"), format!("   {expected}"));
            assert_eq!(format!("{value:<12}"), format!("{expected}   "));
            assert_eq!(format!("{value:^12}"), format!(" {expected}  "));
        }
    }

    #[test]
    fn negation_flips_all_components() {
        assert_eq!(-offset(1, 2, 3), offset(-1, -2, -3));
        assert_eq!(-offset(-1, -2, -3), offset(1, 2, 3));
        assert_eq!(-UtcOffset::UTC, UtcOffset::UTC);
        assert_eq!(-offset(0, 0, 1), offset(0, 0, -1));
        // Double negation is the identity.
        let value = offset(-5, -30, -15);
        assert_eq!(-(-value), value);
        assert_eq!((-value).to_string(), "+05:30:15");
    }

    #[test]
    fn offsets_are_ordered_consistently_with_whole_seconds() {
        let mut values = [
            offset(-25, -59, -59),
            offset(-5, 0, 0),
            offset(0, -30, 0),
            UtcOffset::UTC,
            offset(0, 0, 1),
            offset(1, 30, 45),
            offset(25, 59, 59),
        ];
        let expected_order = [-93_599, -18_000, -1_800, 0, 1, 5_445, 93_599];
        let mut indices: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
        indices.sort_by_key(|&i| values[i]);
        let sorted: Vec<i32> = indices
            .iter()
            .map(|&i| values[i].whole_seconds())
            .collect();
        assert_eq!(sorted, expected_order);

        // Every pair agrees with the whole-seconds comparison.
        for a in &values {
            for b in &values {
                assert_eq!(
                    a.cmp(b),
                    a.whole_seconds().cmp(&b.whole_seconds()),
                    "comparison of {a} and {b}",
                );
                assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
            }
            assert_eq!(a.cmp(a), Ordering::Equal);
        }

        values.sort();
        assert_eq!(values[0].whole_seconds(), -93_599);
        assert_eq!(values[6].whole_seconds(), 93_599);
        assert_eq!(values.iter().min(), Some(&offset(-25, -59, -59)));
        assert_eq!(values.iter().max(), Some(&offset(25, 59, 59)));
    }

    #[test]
    fn equality_and_hashing_treat_equivalent_offsets_as_identical() {
        // Constructed by different means, but represent the same instant offset.
        let a = offset(1, 0, 0);
        let b = UtcOffset::from_whole_seconds(3_600).expect("valid offset");
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));

        // Sign normalization also produces equal values.
        assert_eq!(offset(1, -2, -3), offset(1, 2, 3));
        assert_eq!(hash_of(&offset(1, -2, -3)), hash_of(&offset(1, 2, 3)));

        assert_eq!(
            UtcOffset::UTC,
            UtcOffset::from_whole_seconds(0).expect("valid offset")
        );
        assert_eq!(hash_of(&UtcOffset::UTC), hash_of(&offset(0, 0, 0)));

        // Different offsets compare and hash differently.
        assert_ne!(offset(1, 0, 0), UtcOffset::UTC);
        assert_ne!(offset(1, 0, 0), offset(0, 1, 0));
        assert_ne!(offset(0, 0, 1), offset(0, 0, -1));
        assert_ne!(hash_of(&UtcOffset::UTC), hash_of(&offset(0, 0, 1)));
        assert_ne!(hash_of(&offset(1, 0, 0)), hash_of(&offset(0, 1, 0)));

        // Copy semantics preserve equality.
        let copy = UtcOffset::from_hms(-5, -30, 0).expect("valid offset");
        let value = copy;
        assert_eq!(value, copy);
        assert_eq!(hash_of(&value), hash_of(&copy));
    }

    #[test]
    fn local_offset_is_always_utc() {
        let value = UtcOffset::local_offset_at(()).expect("offset determinable");
        assert!(value.is_utc());
        assert_eq!(value, UtcOffset::UTC);
    }
}
