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

use codevar_time_core::SystemTime;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::time::Duration as StdDuration;

use crate::error::ConversionRange;
use crate::internal_macro::const_try_opt;
use crate::unit::*;
use deranged::ri32;
use num_conv::prelude::*;

#[derive(Debug)]
enum FloatConstructorError {
    Nan,
    NegOverflow,
    PosOverflow,
}

/// By explicitly inserting this enum where padding is expected, the compiler is able to better
/// perform niche value optimization.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Padding {
    #[allow(clippy::missing_docs_in_private_items)]
    Optimize,
}

/// The type of the `nanosecond` field of `SignedDuration`.
type Nanoseconds = ri32<{ -Nanosecond::per_t::<i32>(Second) + 1 }, { Nanosecond::per_t::<i32>(Second) - 1 }>;

/// A span of time with nanosecond precision.
///
/// Each `SignedDuration` is composed of a whole number of seconds and a fractional part represented
/// in nanoseconds.
///
/// This implementation allows for negative durations, unlike [`core::time::Duration`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SignedDuration {
    /// Number of whole seconds.
    seconds: i64,
    /// Number of nanoseconds within the second. The sign always matches the `seconds` field.
    /// Sign must match that of `seconds` (though this is not a safety requirement).
    nanoseconds: Nanoseconds,
    _padding: Padding,
}

impl fmt::Debug for SignedDuration {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SignedDuration")
            .field("seconds", &self.seconds)
            .field("nanoseconds", &self.nanoseconds)
            .finish()
    }
}

impl PartialEq for SignedDuration {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_int_for_equality() == other.as_int_for_equality()
    }
}

impl Eq for SignedDuration {}

impl PartialOrd for SignedDuration {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SignedDuration {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.seconds
            .cmp(&other.seconds)
            .then_with(|| self.nanoseconds.cmp(&other.nanoseconds))
    }
}

impl Hash for SignedDuration {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.as_int_for_equality().hash(state);
    }
}

impl Default for SignedDuration {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

#[rustfmt::skip] // Skip `rustfmt` because it reformats the arguments of the macro weirdly.
macro_rules! try_from_secs {
    (
        secs = $secs: expr,
        mantissa_bits = $mant_bits: literal,
        exponent_bits = $exp_bits: literal,
        offset = $offset: literal,
        bits_ty = $bits_ty:ty,
        bits_ty_signed = $bits_ty_signed:ty,
        double_ty = $double_ty:ty,
        float_ty = $float_ty:ty,
    ) => {{
        'value: {
            const MIN_EXP: i16 = 1 - (1i16 << $exp_bits) / 2;
            const MANT_MASK: $bits_ty = (1 << $mant_bits) - 1;
            const EXP_MASK: $bits_ty = (1 << $exp_bits) - 1;

            let bits = $secs.to_bits();
            let mant = (bits & MANT_MASK) | (MANT_MASK + 1);
            let exp = ((bits >> $mant_bits) & EXP_MASK) as i16 + MIN_EXP;

            let (secs, nanos) = if exp < -31 {
                // the input represents less than 1ns and can not be rounded to it
                (0u64, 0u32)
            } else if exp < 0 {
                // the input is less than 1 second
                let t = (mant as $double_ty) << ($offset + exp);
                let nanos_offset = $mant_bits + $offset;
                #[allow(trivial_numeric_casts)]
                let nanos_tmp = Nanosecond::per_t::<u128>(Second) * t as u128;
                let nanos = (nanos_tmp >> nanos_offset) as u32;

                let rem_mask = (1 << nanos_offset) - 1;
                let rem_msb_mask = 1 << (nanos_offset - 1);
                let rem = nanos_tmp & rem_mask;
                let is_tie = rem == rem_msb_mask;
                let is_even = (nanos & 1) == 0;
                let rem_msb = nanos_tmp & rem_msb_mask == 0;
                let add_ns = !(rem_msb || (is_even && is_tie));

                // f32 does not have enough precision to trigger the second branch
                // since it can not represent numbers between 0.999_999_940_395 and 1.0.
                let nanos = nanos + add_ns as u32;
                if ($mant_bits == 23) || (nanos != Nanosecond::per_t::<u32>(Second)) {
                    (0, nanos)
                } else {
                    (1, 0)
                }
            } else if exp < $mant_bits {
                #[allow(trivial_numeric_casts)]
                let secs = (mant >> ($mant_bits - exp)) as u64;
                let t = ((mant << exp) & MANT_MASK) as $double_ty;
                let nanos_offset = $mant_bits;
                let nanos_tmp = Nanosecond::per_t::<$double_ty>(Second) * t;
                let nanos = (nanos_tmp >> nanos_offset) as u32;

                let rem_mask = (1 << nanos_offset) - 1;
                let rem_msb_mask = 1 << (nanos_offset - 1);
                let rem = nanos_tmp & rem_mask;
                let is_tie = rem == rem_msb_mask;
                let is_even = (nanos & 1) == 0;
                let rem_msb = nanos_tmp & rem_msb_mask == 0;
                let add_ns = !(rem_msb || (is_even && is_tie));

                let nanos = nanos + add_ns as u32;
                if ($mant_bits == 23) || (nanos != Nanosecond::per_t::<u32>(Second)) {
                    (secs, nanos)
                } else {
                    (secs + 1, 0)
                }
            } else if exp < 63 {
                #[allow(trivial_numeric_casts)]
                let secs = (mant as u64) << (exp - $mant_bits);
                (secs, 0)
            } else if bits == (i64::MIN as $float_ty).to_bits() {
                break 'value Ok(Self::new_ranged_unchecked(i64::MIN, Nanoseconds::new_static::<0>()));
            } else if $secs.is_nan() {
                break 'value Err(FloatConstructorError::Nan);
            } else if $secs.is_sign_negative() {
                break 'value Err(FloatConstructorError::NegOverflow);
            } else {
                break 'value Err(FloatConstructorError::PosOverflow);
            };
            let mask = (bits as $bits_ty_signed) >> ($mant_bits + $exp_bits);
            #[allow(trivial_numeric_casts)]
            let secs_signed = ((secs as i64) ^ (mask as i64)) - (mask as i64);
            #[allow(trivial_numeric_casts)]
            let nanos_signed = ((nanos as i32) ^ (mask as i32)) - (mask as i32);
            // Safety: `nanos_signed` is in range.
            Ok(unsafe { Self::new_unchecked(secs_signed, nanos_signed) })
        }
    }};
}

impl SignedDuration {
    /// Equivalent to `0.seconds()`.
    pub const ZERO: Self = Self::seconds(0);

    /// Equivalent to `1.nanoseconds()`.
    pub const NANOSECOND: Self = Self::nanoseconds(1);

    /// Equivalent to `1.microseconds()`.
    pub const MICROSECOND: Self = Self::microseconds(1);

    /// Equivalent to `1.milliseconds()`.
    pub const MILLISECOND: Self = Self::milliseconds(1);

    /// Equivalent to `1.seconds()`.
    pub const SECOND: Self = Self::seconds(1);

    /// Equivalent to `1.minutes()`.
    pub const MINUTE: Self = Self::minutes(1);

    /// Equivalent to `1.hours()`.
    pub const HOUR: Self = Self::hours(1);

    /// Equivalent to `1.days()`.
    pub const DAY: Self = Self::days(1);

    /// Equivalent to `1.weeks()`.
    pub const WEEK: Self = Self::weeks(1);

    /// The minimum possible duration. Adding any negative duration to this will cause an overflow.
    pub const MIN: Self = Self::new_ranged(i64::MIN, Nanoseconds::MIN);

    /// The maximum possible duration. Adding any positive duration to this will cause an overflow.
    pub const MAX: Self = Self::new_ranged(i64::MAX, Nanoseconds::MAX);

    #[inline]
    const fn as_int_for_equality(self) -> i128 {
        // Safety: There are no padding bytes that are not permitted to be read.
        unsafe { core::mem::transmute(self) }
    }

    /// Check if a duration is exactly zero.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.as_int_for_equality() == Self::ZERO.as_int_for_equality()
    }

    /// Check if a duration is negative.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.seconds < 0 || self.nanoseconds.get() < 0
    }

    /// Check if a duration is positive.
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.seconds > 0 || self.nanoseconds.get() > 0
    }

    /// Get the absolute value of the duration.
    ///
    /// This method saturates the returned value if it would otherwise overflow.
    #[inline]
    pub const fn abs(self) -> Self {
        match self.seconds.checked_abs() {
            Some(seconds) => Self::new_ranged_unchecked(seconds, self.nanoseconds.abs()),
            None => Self::MAX,
        }
    }

    /// Convert the existing `SignedDuration` to a `core::time::Duration` and its sign. This returns
    /// a [`core::time::Duration`] and does not saturate the returned value (unlike
    /// [`SignedDuration::abs`]).
    #[inline]
    pub const fn unsigned_abs(self) -> StdDuration {
        StdDuration::new(self.seconds.unsigned_abs(), self.nanoseconds.get().unsigned_abs())
    }

    /// Create a new `SignedDuration` without checking the validity of the components.
    ///
    /// # Safety
    ///
    /// - `nanoseconds` must be in the range `-999_999_999..=999_999_999`.
    ///
    /// While the sign of `nanoseconds` is required to be the same as the sign of `seconds`, this is
    /// not a safety invariant.
    #[inline]
    pub(crate) const unsafe fn new_unchecked(seconds: i64, nanoseconds: i32) -> Self {
        Self::new_ranged_unchecked(
            seconds,
            // Safety: The caller must uphold the safety invariants.
            unsafe { Nanoseconds::new_unchecked(nanoseconds) },
        )
    }

    /// Create a new `SignedDuration` without checking the validity of the components.
    #[inline]
    pub(crate) const fn new_ranged_unchecked(seconds: i64, nanoseconds: Nanoseconds) -> Self {
        if seconds < 0 {
            debug_assert!(nanoseconds.get() <= 0);
        } else if seconds > 0 {
            debug_assert!(nanoseconds.get() >= 0);
        }

        Self {
            seconds,
            nanoseconds,
            _padding: Padding::Optimize,
        }
    }

    /// Create a new `SignedDuration` with the provided seconds and nanoseconds. If nanoseconds is
    /// at least ±10<sup>9</sup>, it will wrap to the number of seconds.
    ///
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    pub const fn new(mut seconds: i64, mut nanoseconds: i32) -> Self {
        let second_adjustment = nanoseconds as i64 / Nanosecond::per_t::<i64>(Second);
        seconds = match seconds.checked_add(second_adjustment) {
            Some(value) => value,
            None if second_adjustment >= 0 => i64::MAX,
            None => i64::MIN,
        };
        nanoseconds %= Nanosecond::per_t::<i32>(Second);

        if seconds > 0 && nanoseconds < 0 {
            // `seconds` cannot overflow here because it is positive.
            seconds -= 1;
            nanoseconds += Nanosecond::per_t::<i32>(Second);
        } else if seconds < 0 && nanoseconds > 0 {
            // `seconds` cannot overflow here because it is negative.
            seconds += 1;
            nanoseconds -= Nanosecond::per_t::<i32>(Second);
        }

        // Safety: `nanoseconds` is in range due to the modulus above.
        unsafe { Self::new_unchecked(seconds, nanoseconds) }
    }

    /// Create a new `SignedDuration` with the provided seconds and nanoseconds.
    #[inline]
    pub(crate) const fn new_ranged(mut seconds: i64, mut nanoseconds: Nanoseconds) -> Self {
        if seconds > 0 && nanoseconds.get() < 0 {
            // `seconds` cannot overflow here because it is positive.
            seconds -= 1;
            // Safety: `nanoseconds` is negative with a maximum of 999,999,999, so adding a billion
            // to it is guaranteed to result in an in-range value.
            nanoseconds =
                unsafe { Nanoseconds::new_unchecked(nanoseconds.get() + Nanosecond::per_t::<i32>(Second)) };
        } else if seconds < 0 && nanoseconds.get() > 0 {
            // `seconds` cannot overflow here because it is negative.
            seconds += 1;
            // Safety: `nanoseconds` is positive with a minimum of -999,999,999, so subtracting a
            // billion from it is guaranteed to result in an in-range value.
            nanoseconds =
                unsafe { Nanoseconds::new_unchecked(nanoseconds.get() - Nanosecond::per_t::<i32>(Second)) };
        }

        Self::new_ranged_unchecked(seconds, nanoseconds)
    }

    /// Create a new `SignedDuration` with the given number of weeks. Equivalent to
    /// `SignedDuration::seconds(weeks * 604_800)`.
    ///
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    pub const fn weeks(weeks: i64) -> Self {
        Self::seconds(match weeks.checked_mul(Second::per_t(Week)) {
            Some(value) => value,
            None if weeks >= 0 => i64::MAX,
            None => i64::MIN,
        })
    }

    /// Create a new `SignedDuration` with the given number of days. Equivalent to
    /// `SignedDuration::seconds(days * 86_400)`.
    ///
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    pub const fn days(days: i64) -> Self {
        Self::seconds(match days.checked_mul(Second::per_t(Day)) {
            Some(value) => value,
            None if days >= 0 => i64::MAX,
            None => i64::MIN,
        })
    }

    /// Create a new `SignedDuration` with the given number of hours. Equivalent to
    /// `SignedDuration::seconds(hours * 3_600)`.
    ///
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    pub const fn hours(hours: i64) -> Self {
        Self::seconds(match hours.checked_mul(Second::per_t(Hour)) {
            Some(value) => value,
            None if hours >= 0 => i64::MAX,
            None => i64::MIN,
        })
    }

    /// Create a new `SignedDuration` with the given number of minutes. Equivalent to
    /// `SignedDuration::seconds(minutes * 60)`.
    ///
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    pub const fn minutes(minutes: i64) -> Self {
        Self::seconds(match minutes.checked_mul(Second::per_t(Minute)) {
            Some(value) => value,
            None if minutes >= 0 => i64::MAX,
            None => i64::MIN,
        })
    }

    /// Create a new `SignedDuration` with the given number of seconds.
    #[inline]
    pub const fn seconds(seconds: i64) -> Self {
        Self::new_ranged_unchecked(seconds, Nanoseconds::new_static::<0>())
    }

    /// Create a new `SignedDuration` from the specified number of seconds represented as `f64`.
    ///
    /// If the value is `NaN` or out of bounds, an error is returned that can be handled in the
    /// desired manner by the caller.
    #[inline]
    const fn try_seconds_f64(seconds: f64) -> Result<Self, FloatConstructorError> {
        try_from_secs!(
            secs = seconds,
            mantissa_bits = 52,
            exponent_bits = 11,
            offset = 44,
            bits_ty = u64,
            bits_ty_signed = i64,
            double_ty = u128,
            float_ty = f64,
        )
    }

    /// Create a new `SignedDuration` from the specified number of seconds represented as `f32`.
    ///
    /// If the value is `NaN` or out of bounds, an error is returned that can be handled in the
    /// desired manner by the caller.
    #[inline]
    const fn try_seconds_f32(seconds: f32) -> Result<Self, FloatConstructorError> {
        try_from_secs!(
            secs = seconds,
            mantissa_bits = 23,
            exponent_bits = 8,
            offset = 41,
            bits_ty = u32,
            bits_ty_signed = i32,
            double_ty = u64,
            float_ty = f32,
        )
    }

    /// Creates a new `SignedDuration` from the specified number of seconds represented as `f64`.
    ///
    /// # Panics
    ///
    /// This may panic if `seconds` is `NaN` or overflows the representable range of
    /// `SignedDuration`.
    #[inline]
    pub const fn seconds_f64(seconds: f64) -> Self {
        match Self::try_seconds_f64(seconds) {
            Ok(duration) => duration,
            Err(FloatConstructorError::Nan) => Self::ZERO,
            Err(FloatConstructorError::NegOverflow) => Self::MIN,
            Err(FloatConstructorError::PosOverflow) => Self::MAX,
        }
    }

    /// Creates a new `SignedDuration` from the specified number of seconds represented as `f32`.
    ///
    /// # Panics
    ///
    /// This may panic if `seconds` is `NaN` or overflows the representable range of
    /// `SignedDuration`.
    #[inline]
    pub const fn seconds_f32(seconds: f32) -> Self {
        match Self::try_seconds_f32(seconds) {
            Ok(duration) => duration,
            Err(FloatConstructorError::Nan) => Self::ZERO,
            Err(FloatConstructorError::NegOverflow) => Self::MIN,
            Err(FloatConstructorError::PosOverflow) => Self::MAX,
        }
    }

    /// Creates a new `SignedDuration` from the specified number of seconds represented as `f64`.
    /// Any values that are out of bounds are saturated at the minimum or maximum respectively.
    /// `NaN` gets turned into a `SignedDuration` of 0 seconds.
    #[inline]
    pub const fn saturating_seconds_f64(seconds: f64) -> Self {
        match Self::try_seconds_f64(seconds) {
            Ok(duration) => duration,
            Err(FloatConstructorError::Nan) => Self::ZERO,
            Err(FloatConstructorError::NegOverflow) => Self::MIN,
            Err(FloatConstructorError::PosOverflow) => Self::MAX,
        }
    }

    /// Creates a new `SignedDuration` from the specified number of seconds represented as `f32`.
    /// Any values that are out of bounds are saturated at the minimum or maximum respectively.
    /// `NaN` gets turned into a `SignedDuration` of 0 seconds.
    #[inline]
    pub const fn saturating_seconds_f32(seconds: f32) -> Self {
        match Self::try_seconds_f32(seconds) {
            Ok(duration) => duration,
            Err(FloatConstructorError::Nan) => Self::ZERO,
            Err(FloatConstructorError::NegOverflow) => Self::MIN,
            Err(FloatConstructorError::PosOverflow) => Self::MAX,
        }
    }

    /// Creates a new `SignedDuration` from the specified number of seconds represented as `f64`.
    /// Returns `None` if the `SignedDuration` can't be represented.
    #[inline]
    pub const fn checked_seconds_f64(seconds: f64) -> Option<Self> {
        match Self::try_seconds_f64(seconds) {
            Ok(duration) => Some(duration),
            Err(_) => None,
        }
    }

    /// Creates a new `SignedDuration` from the specified number of seconds represented as `f32`.
    /// Returns `None` if the `SignedDuration` can't be represented.
    #[inline]
    pub const fn checked_seconds_f32(seconds: f32) -> Option<Self> {
        match Self::try_seconds_f32(seconds) {
            Ok(duration) => Some(duration),
            Err(_) => None,
        }
    }

    /// Create a new `SignedDuration` with the given number of milliseconds.
    #[inline]
    pub const fn milliseconds(milliseconds: i64) -> Self {
        // Safety: `nanoseconds` is guaranteed to be in range because of the modulus.
        unsafe {
            Self::new_unchecked(
                milliseconds / Millisecond::per_t::<i64>(Second),
                (milliseconds % Millisecond::per_t::<i64>(Second) * Nanosecond::per_t::<i64>(Millisecond))
                    as i32,
            )
        }
    }

    /// Create a new `SignedDuration` with the given number of microseconds.
    #[inline]
    pub const fn microseconds(microseconds: i64) -> Self {
        // Safety: `nanoseconds` is guaranteed to be in range because of the modulus.
        unsafe {
            Self::new_unchecked(
                microseconds / Microsecond::per_t::<i64>(Second),
                (microseconds % Microsecond::per_t::<i64>(Second) * Nanosecond::per_t::<i64>(Microsecond))
                    as i32,
            )
        }
    }

    /// Create a new `SignedDuration` with the given number of nanoseconds.
    #[inline]
    pub const fn nanoseconds(nanoseconds: i64) -> Self {
        // Safety: `nanoseconds` is guaranteed to be in range because of the modulus.
        unsafe {
            Self::new_unchecked(
                nanoseconds / Nanosecond::per_t::<i64>(Second),
                (nanoseconds % Nanosecond::per_t::<i64>(Second)) as i32,
            )
        }
    }

    /// Create a new `SignedDuration` with the given number of nanoseconds.
    ///
    /// # Panics
    ///
    /// This may panic if an overflow occurs. This may happen because the input range cannot be
    /// fully mapped to the output.
    #[inline]
    pub const fn nanoseconds_i128(nanoseconds: i128) -> Self {
        let seconds = nanoseconds / Nanosecond::per_t::<i128>(Second);
        let nanoseconds = nanoseconds % Nanosecond::per_t::<i128>(Second);

        if seconds > i64::MAX as i128 {
            return Self::MAX;
        }
        if seconds < i64::MIN as i128 {
            return Self::MIN;
        }

        // Safety: `nanoseconds` is guaranteed to be in range because of the modulus above.
        unsafe { Self::new_unchecked(seconds as i64, nanoseconds as i32) }
    }

    /// Get the number of whole weeks in the duration.
    #[inline]
    pub const fn whole_weeks(self) -> i64 {
        self.whole_seconds() / Second::per_t::<i64>(Week)
    }

    /// Get the number of whole days in the duration.
    #[inline]
    pub const fn whole_days(self) -> i64 {
        self.whole_seconds() / Second::per_t::<i64>(Day)
    }

    /// Get the number of whole hours in the duration.
    #[inline]
    pub const fn whole_hours(self) -> i64 {
        self.whole_seconds() / Second::per_t::<i64>(Hour)
    }

    /// Get the number of whole minutes in the duration.
    #[inline]
    pub const fn whole_minutes(self) -> i64 {
        self.whole_seconds() / Second::per_t::<i64>(Minute)
    }

    /// Get the number of whole seconds in the duration.
    #[inline]
    pub const fn whole_seconds(self) -> i64 {
        self.seconds
    }

    /// Get the number of fractional seconds in the duration.
    #[inline]
    pub const fn as_seconds_f64(self) -> f64 {
        self.seconds as f64 + self.nanoseconds.get() as f64 / Nanosecond::per_t::<f64>(Second)
    }

    /// Get the number of fractional seconds in the duration.
    #[inline]
    pub const fn as_seconds_f32(self) -> f32 {
        self.seconds as f32 + self.nanoseconds.get() as f32 / Nanosecond::per_t::<f32>(Second)
    }

    /// Get the number of whole milliseconds in the duration.
    #[inline]
    pub const fn whole_milliseconds(self) -> i128 {
        self.seconds as i128 * Millisecond::per_t::<i128>(Second)
            + self.nanoseconds.get() as i128 / Nanosecond::per_t::<i128>(Millisecond)
    }

    /// Get the number of milliseconds past the number of whole seconds.
    ///
    /// Always in the range `-999..=999`.
    #[inline]
    pub const fn subsec_milliseconds(self) -> i16 {
        (self.nanoseconds.get() / Nanosecond::per_t::<i32>(Millisecond)) as i16
    }

    /// Get the number of whole microseconds in the duration.
    #[inline]
    pub const fn whole_microseconds(self) -> i128 {
        self.seconds as i128 * Microsecond::per_t::<i128>(Second)
            + self.nanoseconds.get() as i128 / Nanosecond::per_t::<i128>(Microsecond)
    }

    /// Get the number of microseconds past the number of whole seconds.
    ///
    /// Always in the range `-999_999..=999_999`.
    #[inline]
    pub const fn subsec_microseconds(self) -> i32 {
        self.nanoseconds.get() / Nanosecond::per_t::<i32>(Microsecond)
    }

    /// Get the number of nanoseconds in the duration.
    #[inline]
    pub const fn whole_nanoseconds(self) -> i128 {
        self.seconds as i128 * Nanosecond::per_t::<i128>(Second) + self.nanoseconds.get() as i128
    }

    /// Get the number of nanoseconds past the number of whole seconds.
    ///
    /// The returned value will always be in the range `-999_999_999..=999_999_999`.
    #[inline]
    pub const fn subsec_nanoseconds(self) -> i32 {
        self.nanoseconds.get()
    }

    /// Get the number of nanoseconds past the number of whole seconds.
    #[inline]
    pub(crate) const fn subsec_nanoseconds_ranged(self) -> Nanoseconds {
        self.nanoseconds
    }

    /// Computes `self + rhs`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        let mut seconds = const_try_opt!(self.seconds.checked_add(rhs.seconds));
        let mut nanoseconds = self.nanoseconds.get() + rhs.nanoseconds.get();

        if nanoseconds >= Nanosecond::per_t(Second) || seconds < 0 && nanoseconds > 0 {
            nanoseconds -= Nanosecond::per_t::<i32>(Second);
            seconds = const_try_opt!(seconds.checked_add(1));
        } else if nanoseconds <= -Nanosecond::per_t::<i32>(Second) || seconds > 0 && nanoseconds < 0 {
            nanoseconds += Nanosecond::per_t::<i32>(Second);
            seconds = const_try_opt!(seconds.checked_sub(1));
        }

        // Safety: `nanoseconds` is guaranteed to be in range because of the overflow handling.
        unsafe { Some(Self::new_unchecked(seconds, nanoseconds)) }
    }

    /// Computes `self - rhs`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        let mut seconds = const_try_opt!(self.seconds.checked_sub(rhs.seconds));
        let mut nanoseconds = self.nanoseconds.get() - rhs.nanoseconds.get();

        if nanoseconds >= Nanosecond::per_t(Second) || seconds < 0 && nanoseconds > 0 {
            nanoseconds -= Nanosecond::per_t::<i32>(Second);
            seconds = const_try_opt!(seconds.checked_add(1));
        } else if nanoseconds <= -Nanosecond::per_t::<i32>(Second) || seconds > 0 && nanoseconds < 0 {
            nanoseconds += Nanosecond::per_t::<i32>(Second);
            seconds = const_try_opt!(seconds.checked_sub(1));
        }

        // Safety: `nanoseconds` is guaranteed to be in range because of the overflow handling.
        unsafe { Some(Self::new_unchecked(seconds, nanoseconds)) }
    }

    /// Computes `self * rhs`, returning `None` if an overflow occurred.
    #[inline]
    pub const fn checked_mul(self, rhs: i32) -> Option<Self> {
        // Multiply nanoseconds as i64, because it cannot overflow that way.
        let total_nanos = self.nanoseconds.get() as i64 * rhs as i64;
        let extra_secs = total_nanos / Nanosecond::per_t::<i64>(Second);
        let nanoseconds = (total_nanos % Nanosecond::per_t::<i64>(Second)) as i32;
        let seconds =
            const_try_opt!(const_try_opt!(self.seconds.checked_mul(rhs as i64)).checked_add(extra_secs));

        // Safety: `nanoseconds` is guaranteed to be in range because of the modulus above.
        unsafe { Some(Self::new_unchecked(seconds, nanoseconds)) }
    }

    /// Computes `self / rhs`, returning `None` if `rhs == 0` or if the result would overflow.
    #[inline]
    pub const fn checked_div(self, rhs: i32) -> Option<Self> {
        let (secs, extra_secs) = (
            const_try_opt!(self.seconds.checked_div(rhs as i64)),
            self.seconds % (rhs as i64),
        );
        let (mut nanos, extra_nanos) = (self.nanoseconds.get() / rhs, self.nanoseconds.get() % rhs);
        nanos +=
            ((extra_secs * (Nanosecond::per_t::<i64>(Second)) + extra_nanos as i64) / (rhs as i64)) as i32;

        // Safety: `nanoseconds` is in range.
        unsafe { Some(Self::new_unchecked(secs, nanos)) }
    }

    /// Computes `-self`, returning `None` if the result would overflow.
    #[inline]
    pub const fn checked_neg(self) -> Option<Self> {
        if self.seconds == i64::MIN {
            None
        } else {
            Some(Self::new_ranged_unchecked(-self.seconds, self.nanoseconds.neg()))
        }
    }

    /// Computes `self + rhs`, saturating if an overflow occurred.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        let (mut seconds, overflow) = self.seconds.overflowing_add(rhs.seconds);
        if overflow {
            if self.seconds > 0 {
                return Self::MAX;
            }
            return Self::MIN;
        }
        let mut nanoseconds = self.nanoseconds.get() + rhs.nanoseconds.get();

        if nanoseconds >= Nanosecond::per_t(Second) || seconds < 0 && nanoseconds > 0 {
            nanoseconds -= Nanosecond::per_t::<i32>(Second);
            seconds = match seconds.checked_add(1) {
                Some(seconds) => seconds,
                None => return Self::MAX,
            };
        } else if nanoseconds <= -Nanosecond::per_t::<i32>(Second) || seconds > 0 && nanoseconds < 0 {
            nanoseconds += Nanosecond::per_t::<i32>(Second);
            seconds = match seconds.checked_sub(1) {
                Some(seconds) => seconds,
                None => return Self::MIN,
            };
        }

        // Safety: `nanoseconds` is guaranteed to be in range because of the overflow handling.
        unsafe { Self::new_unchecked(seconds, nanoseconds) }
    }

    /// Computes `self - rhs`, saturating if an overflow occurred.
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        let (mut seconds, overflow) = self.seconds.overflowing_sub(rhs.seconds);
        if overflow {
            if self.seconds > 0 {
                return Self::MAX;
            }
            return Self::MIN;
        }
        let mut nanoseconds = self.nanoseconds.get() - rhs.nanoseconds.get();

        if nanoseconds >= Nanosecond::per_t(Second) || seconds < 0 && nanoseconds > 0 {
            nanoseconds -= Nanosecond::per_t::<i32>(Second);
            seconds = match seconds.checked_add(1) {
                Some(seconds) => seconds,
                None => return Self::MAX,
            };
        } else if nanoseconds <= -Nanosecond::per_t::<i32>(Second) || seconds > 0 && nanoseconds < 0 {
            nanoseconds += Nanosecond::per_t::<i32>(Second);
            seconds = match seconds.checked_sub(1) {
                Some(seconds) => seconds,
                None => return Self::MIN,
            };
        }

        // Safety: `nanoseconds` is guaranteed to be in range because of the overflow handling.
        unsafe { Self::new_unchecked(seconds, nanoseconds) }
    }

    /// Computes `self * rhs`, saturating if an overflow occurred.
    #[inline]
    pub const fn saturating_mul(self, rhs: i32) -> Self {
        // Multiply nanoseconds as i64, because it cannot overflow that way.
        let total_nanos = self.nanoseconds.get() as i64 * rhs as i64;
        let extra_secs = total_nanos / Nanosecond::per_t::<i64>(Second);
        let nanoseconds = (total_nanos % Nanosecond::per_t::<i64>(Second)) as i32;
        let (seconds, overflow1) = self.seconds.overflowing_mul(rhs as i64);
        if overflow1 {
            if self.seconds > 0 && rhs > 0 || self.seconds < 0 && rhs < 0 {
                return Self::MAX;
            }
            return Self::MIN;
        }
        let (seconds, overflow2) = seconds.overflowing_add(extra_secs);
        if overflow2 {
            if self.seconds > 0 && rhs > 0 {
                return Self::MAX;
            }
            return Self::MIN;
        }

        // Safety: `nanoseconds` is guaranteed to be in range because of to the modulus above.
        unsafe { Self::new_unchecked(seconds, nanoseconds) }
    }
}

impl TryFrom<StdDuration> for SignedDuration {
    type Error = ConversionRange;

    #[inline]
    fn try_from(original: StdDuration) -> Result<Self, ConversionRange> {
        Ok(Self::new(
            original
                .as_secs()
                .try_into()
                .map_err(|_| ConversionRange)?,
            original.subsec_nanos().cast_signed(),
        ))
    }
}

impl TryFrom<SignedDuration> for StdDuration {
    type Error = ConversionRange;

    #[inline]
    fn try_from(duration: SignedDuration) -> Result<Self, ConversionRange> {
        Ok(Self::new(
            duration
                .seconds
                .try_into()
                .map_err(|_| ConversionRange)?,
            duration
                .nanoseconds
                .get()
                .try_into()
                .map_err(|_| ConversionRange)?,
        ))
    }
}

impl Add for SignedDuration {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs)
            .unwrap_or_else(|| self.saturating_add(rhs))
    }
}

impl Add<StdDuration> for SignedDuration {
    type Output = Self;

    #[inline]
    fn add(self, std_duration: StdDuration) -> Self::Output {
        match Self::try_from(std_duration) {
            Ok(rhs) => self + rhs,
            Err(_) => self.saturating_add(Self::MAX),
        }
    }
}

impl Add<SignedDuration> for StdDuration {
    type Output = SignedDuration;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add(self, rhs: SignedDuration) -> Self::Output {
        rhs + self
    }
}

impl AddAssign<Self> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<StdDuration> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn add_assign(&mut self, rhs: StdDuration) {
        *self = *self + rhs;
    }
}

impl AddAssign<SignedDuration> for StdDuration {
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        // Saturating on the magnitude keeps the result in range without round tripping through
        // `SignedDuration`, which cannot represent durations beyond `i64::MAX` seconds.
        if rhs.is_negative() {
            *self = self.saturating_sub(rhs.unsigned_abs());
        } else {
            *self = self.saturating_add(rhs.unsigned_abs());
        }
    }
}

impl Neg for SignedDuration {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        // The only value whose negation is not representable is `MIN`, which saturates to `MAX`.
        self.checked_neg().unwrap_or(Self::MAX)
    }
}

impl Sub for SignedDuration {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs)
            .unwrap_or_else(|| self.saturating_sub(rhs))
    }
}

impl Sub<StdDuration> for SignedDuration {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: StdDuration) -> Self::Output {
        match Self::try_from(rhs) {
            Ok(value) => self - value,
            Err(_) => self.saturating_sub(Self::MAX),
        }
    }
}

impl Sub<SignedDuration> for StdDuration {
    type Output = SignedDuration;

    #[inline]
    fn sub(self, rhs: SignedDuration) -> Self::Output {
        match SignedDuration::try_from(self) {
            Ok(value) => value - rhs,
            Err(_) => SignedDuration::MAX.saturating_sub(rhs),
        }
    }
}

impl SubAssign<Self> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<StdDuration> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn sub_assign(&mut self, rhs: StdDuration) {
        *self = *self - rhs;
    }
}

impl SubAssign<SignedDuration> for StdDuration {
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        // Saturating on the magnitude keeps the result in range: subtracting a duration larger
        // than the receiver yields zero rather than wrapping to `StdDuration::MAX`.
        if rhs.is_negative() {
            *self = self.saturating_add(rhs.unsigned_abs());
        } else {
            *self = self.saturating_sub(rhs.unsigned_abs());
        }
    }
}

/// Given a value and whether it is signed, cast it to the signed version.
macro_rules! cast_signed {
    (@signed $val:ident) => {
        $val
    };
    (@unsigned $val:ident) => {
        $val.cast_signed()
    };
}

/// Implement `Mul` (reflexively), `MulAssign`, `Div`, and `DivAssign` for `SignedDuration` for
/// various signed types.
macro_rules! duration_mul_div_int {
    ($(@$signedness:ident $type:ty),+ $(,)?) => {$(
        impl Mul<$type> for SignedDuration {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn mul(self, rhs: $type) -> Self::Output {
                let rhs_i128 = cast_signed!(@$signedness rhs).widen::<i128>();
                match self.whole_nanoseconds().checked_mul(rhs_i128) {
                    Some(total) => Self::nanoseconds_i128(total),
                    None if rhs_i128 >= 0 => Self::MAX,
                    None => Self::MIN,
                }
            }
        }

        impl Mul<SignedDuration> for $type {
            type Output = SignedDuration;

            /// # Panics
            ///
            /// This may panic if an overflow occurs.
            #[inline]
            #[track_caller]
            fn mul(self, rhs: SignedDuration) -> Self::Output {
                rhs * self
            }
        }

        impl MulAssign<$type> for SignedDuration {
            /// # Panics
            ///
            /// This may panic if an overflow occurs.
            #[inline]
            #[track_caller]
            fn mul_assign(&mut self, rhs: $type) {
                *self = *self * rhs;
            }
        }

        impl Div<$type> for SignedDuration {
            type Output = Self;

            /// # Panics
            ///
            /// This may panic if an overflow occurs or if `rhs == 0`.
            #[inline]
            #[track_caller]
            fn div(self, rhs: $type) -> Self::Output {
                Self::nanoseconds_i128(
                    self.whole_nanoseconds() / cast_signed!(@$signedness rhs).widen::<i128>()
                )
            }
        }

        impl DivAssign<$type> for SignedDuration {
            /// # Panics
            ///
            /// This may panic if an overflow occurs or if `rhs == 0`.
            #[inline]
            #[track_caller]
            fn div_assign(&mut self, rhs: $type) {
                *self = *self / rhs;
            }
        }
    )+};
}

duration_mul_div_int! {
    @signed i8,
    @signed i16,
    @signed i32,
    @unsigned u8,
    @unsigned u16,
    @unsigned u32,
}

impl Mul<f32> for SignedDuration {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::seconds_f32(self.as_seconds_f32() * rhs)
    }
}

impl Mul<SignedDuration> for f32 {
    type Output = SignedDuration;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn mul(self, rhs: SignedDuration) -> Self::Output {
        rhs * self
    }
}

impl Mul<f64> for SignedDuration {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::seconds_f64(self.as_seconds_f64() * rhs)
    }
}

impl Mul<SignedDuration> for f64 {
    type Output = SignedDuration;

    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn mul(self, rhs: SignedDuration) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<f32> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs.
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl Div<f32> for SignedDuration {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs or if `rhs == 0`.
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self::seconds_f32(self.as_seconds_f32() / rhs)
    }
}

impl Div<f64> for SignedDuration {
    type Output = Self;

    /// # Panics
    ///
    /// This may panic if an overflow occurs or if `rhs == 0`.
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self::seconds_f64(self.as_seconds_f64() / rhs)
    }
}

impl DivAssign<f32> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs or if `rhs == 0`.
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl DivAssign<f64> for SignedDuration {
    /// # Panics
    ///
    /// This may panic if an overflow occurs or if `rhs == 0`.
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl Div for SignedDuration {
    type Output = f64;

    /// # Panics
    ///
    /// This may panic if `rhs == SignedDuration::ZERO`.
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.as_seconds_f64() / rhs.as_seconds_f64()
    }
}

impl Div<StdDuration> for SignedDuration {
    type Output = f64;

    /// # Panics
    ///
    /// This may panic if `rhs == SignedDuration::ZERO`.
    #[inline]
    fn div(self, rhs: StdDuration) -> Self::Output {
        self.as_seconds_f64() / rhs.as_secs_f64()
    }
}

impl Div<SignedDuration> for StdDuration {
    type Output = f64;

    /// # Panics
    ///
    /// This may panic if `rhs == SignedDuration::ZERO`.
    #[inline]
    fn div(self, rhs: SignedDuration) -> Self::Output {
        self.as_secs_f64() / rhs.as_seconds_f64()
    }
}

impl PartialEq<StdDuration> for SignedDuration {
    #[inline]
    fn eq(&self, rhs: &StdDuration) -> bool {
        Ok(*self) == Self::try_from(*rhs)
    }
}

impl PartialEq<SignedDuration> for StdDuration {
    #[inline]
    fn eq(&self, rhs: &SignedDuration) -> bool {
        rhs == self
    }
}

impl PartialOrd<StdDuration> for SignedDuration {
    #[inline]
    fn partial_cmp(&self, rhs: &StdDuration) -> Option<Ordering> {
        if rhs.as_secs() > i64::MAX.cast_unsigned() {
            return Some(Ordering::Less);
        }

        Some(
            self.seconds
                .cmp(&rhs.as_secs().cast_signed())
                .then_with(|| {
                    self.nanoseconds
                        .get()
                        .cmp(&rhs.subsec_nanos().cast_signed())
                }),
        )
    }
}

impl PartialOrd<SignedDuration> for StdDuration {
    #[inline]
    fn partial_cmp(&self, rhs: &SignedDuration) -> Option<Ordering> {
        rhs.partial_cmp(self).map(Ordering::reverse)
    }
}

impl Sum for SignedDuration {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

impl<'a> Sum<&'a Self> for SignedDuration {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.copied().sum()
    }
}

impl Add<SignedDuration> for SystemTime {
    type Output = Self;

    #[inline]
    fn add(self, duration: SignedDuration) -> Self::Output {
        // `SystemTime` is a zero-sized marker type with no arithmetic state, so the result is
        // always the value itself. Re-entering `self + ...` here would recurse infinitely.
        let _ = duration;
        self
    }
}

impl AddAssign<SignedDuration> for SystemTime {
    #[inline]
    fn add_assign(&mut self, rhs: SignedDuration) {
        *self = *self + rhs;
    }
}

impl Sub<SignedDuration> for SystemTime {
    type Output = Self;

    #[inline]
    fn sub(self, duration: SignedDuration) -> Self::Output {
        // See the note on `Add<SignedDuration> for SystemTime`.
        let _ = duration;
        self
    }
}

impl SubAssign<SignedDuration> for SystemTime {
    #[inline]
    fn sub_assign(&mut self, rhs: SignedDuration) {
        *self = *self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn constants_match_unit_durations() {
        assert_eq!(SignedDuration::ZERO, SignedDuration::seconds(0));
        assert_eq!(SignedDuration::NANOSECOND, SignedDuration::nanoseconds(1));
        assert_eq!(SignedDuration::MICROSECOND, SignedDuration::microseconds(1));
        assert_eq!(SignedDuration::MILLISECOND, SignedDuration::milliseconds(1));
        assert_eq!(SignedDuration::SECOND, SignedDuration::seconds(1));
        assert_eq!(SignedDuration::MINUTE, SignedDuration::minutes(1));
        assert_eq!(SignedDuration::HOUR, SignedDuration::hours(1));
        assert_eq!(SignedDuration::DAY, SignedDuration::days(1));
        assert_eq!(SignedDuration::WEEK, SignedDuration::weeks(1));
        assert_eq!(SignedDuration::default(), SignedDuration::ZERO);
        assert!(SignedDuration::default().is_zero());

        assert!(SignedDuration::ZERO.is_zero());
        assert!(!SignedDuration::ZERO.is_positive());
        assert!(!SignedDuration::ZERO.is_negative());
        assert!(SignedDuration::SECOND.is_positive());
        assert!(!SignedDuration::SECOND.is_negative());
        assert!(SignedDuration::seconds(-1).is_negative());
        assert!(!SignedDuration::seconds(-1).is_positive());
        assert_eq!(-SignedDuration::SECOND, SignedDuration::seconds(-1));
        assert_eq!(SignedDuration::ZERO.neg(), SignedDuration::ZERO);
    }

    #[test]
    fn min_and_max_boundaries() {
        assert_eq!(SignedDuration::MIN.whole_seconds(), i64::MIN);
        assert_eq!(SignedDuration::MIN.subsec_nanoseconds(), -999_999_999);
        assert_eq!(SignedDuration::MAX.whole_seconds(), i64::MAX);
        assert_eq!(SignedDuration::MAX.subsec_nanoseconds(), 999_999_999);
        assert!(SignedDuration::MIN.is_negative());
        assert!(SignedDuration::MAX.is_positive());
        assert!(SignedDuration::MIN < SignedDuration::MAX);

        // `abs` saturates instead of overflowing.
        assert_eq!(SignedDuration::MIN.abs(), SignedDuration::MAX);
        assert_eq!(SignedDuration::MAX.abs(), SignedDuration::MAX);
        assert_eq!(SignedDuration::seconds(-5).abs(), SignedDuration::seconds(5));
        assert_eq!(SignedDuration::seconds(5).abs(), SignedDuration::seconds(5));

        // `unsigned_abs` never saturates.
        assert_eq!(
            SignedDuration::MIN.unsigned_abs(),
            StdDuration::new(9_223_372_036_854_775_808, 999_999_999),
        );
        assert_eq!(
            SignedDuration::seconds(-1).unsigned_abs(),
            StdDuration::from_secs(1)
        );
        assert_eq!(
            SignedDuration::milliseconds(-1_500).unsigned_abs(),
            StdDuration::from_millis(1_500)
        );

        // Overlarge components saturate rather than wrapping.
        assert_eq!(
            SignedDuration::new(i64::MAX, 1_000_000_000),
            SignedDuration::seconds(i64::MAX)
        );
        assert_eq!(
            SignedDuration::new(i64::MIN, -1_000_000_000),
            SignedDuration::seconds(i64::MIN)
        );
    }

    #[test]
    fn constructors_normalize_and_saturate() {
        // Sub-second overflows fold into the seconds component.
        assert_eq!(
            SignedDuration::new(1, 1_500_000_000),
            SignedDuration::new(2, 500_000_000),
        );
        assert_eq!(
            SignedDuration::new(-1, -1_500_000_000),
            SignedDuration::new(-2, -500_000_000),
        );
        // Mixed signs borrow from the seconds component.
        assert_eq!(
            SignedDuration::new(1, -1),
            SignedDuration::nanoseconds(999_999_999)
        );
        assert_eq!(
            SignedDuration::new(-1, 1),
            SignedDuration::nanoseconds(-999_999_999)
        );
        assert_eq!(SignedDuration::new(0, -1), SignedDuration::nanoseconds(-1));
        assert_eq!(SignedDuration::new(0, 1), SignedDuration::nanoseconds(1));

        // Each unit constructor is equivalent to scaling the next smaller one.
        assert_eq!(SignedDuration::weeks(2), SignedDuration::days(14));
        assert_eq!(SignedDuration::days(1), SignedDuration::hours(24));
        assert_eq!(SignedDuration::hours(1), SignedDuration::minutes(60));
        assert_eq!(SignedDuration::minutes(1), SignedDuration::seconds(60));
        assert_eq!(SignedDuration::seconds(1), SignedDuration::milliseconds(1_000));
        assert_eq!(
            SignedDuration::milliseconds(1),
            SignedDuration::microseconds(1_000)
        );
        assert_eq!(
            SignedDuration::microseconds(1),
            SignedDuration::nanoseconds(1_000)
        );
        assert_eq!(SignedDuration::hours(-1), SignedDuration::minutes(-60));
        assert_eq!(
            SignedDuration::nanoseconds(-1_500_000_000),
            SignedDuration::new(-1, -500_000_000),
        );

        // Multiplicative constructors saturate at the bounds instead of wrapping, truncating to
        // a whole number of seconds.
        assert_eq!(SignedDuration::hours(i64::MAX), SignedDuration::seconds(i64::MAX));
        assert_eq!(SignedDuration::hours(i64::MIN), SignedDuration::seconds(i64::MIN));
        assert_eq!(SignedDuration::weeks(i64::MAX), SignedDuration::seconds(i64::MAX));
        assert_eq!(SignedDuration::days(i64::MIN), SignedDuration::seconds(i64::MIN));
        assert_eq!(
            SignedDuration::minutes(i64::MAX),
            SignedDuration::seconds(i64::MAX)
        );
        assert!(SignedDuration::hours(i64::MAX) < SignedDuration::MAX);
        assert!(SignedDuration::minutes(i64::MIN) > SignedDuration::MIN);

        // `nanoseconds_i128` saturates outside of the representable range.
        assert_eq!(
            SignedDuration::nanoseconds_i128(1_500_000_000),
            SignedDuration::new(1, 500_000_000)
        );
        let exact = SignedDuration::nanoseconds_i128(i128::from(i64::MAX) * 2);
        assert_eq!(exact.whole_nanoseconds(), i128::from(i64::MAX) * 2);
        assert_eq!(SignedDuration::nanoseconds_i128(i128::MAX), SignedDuration::MAX);
        assert_eq!(SignedDuration::nanoseconds_i128(i128::MIN), SignedDuration::MIN);
    }

    #[test]
    fn whole_and_subsec_accessors() {
        let value = SignedDuration::new(172_803, 456_789_000); // 2 days, plus 3.456789 seconds
        assert_eq!(value.whole_weeks(), 0);
        assert_eq!(value.whole_days(), 2);
        assert_eq!(value.whole_hours(), 48);
        assert_eq!(value.whole_minutes(), 2_880);
        assert_eq!(value.whole_seconds(), 172_803);
        assert_eq!(value.whole_milliseconds(), 172_803_456);
        assert_eq!(value.subsec_milliseconds(), 456);
        assert_eq!(value.whole_microseconds(), 172_803_456_789);
        assert_eq!(value.subsec_microseconds(), 456_789);
        assert_eq!(value.whole_nanoseconds(), 172_803_456_789_000_i128);
        assert_eq!(value.subsec_nanoseconds(), 456_789_000);

        // Negative durations truncate towards zero.
        let value = SignedDuration::new(-172_803, -456_789_000);
        assert_eq!(value.whole_weeks(), 0);
        assert_eq!(value.whole_days(), -2);
        assert_eq!(value.whole_hours(), -48);
        assert_eq!(value.whole_minutes(), -2_880);
        assert_eq!(value.whole_seconds(), -172_803);
        assert_eq!(value.whole_milliseconds(), -172_803_456);
        assert_eq!(value.subsec_milliseconds(), -456);
        assert_eq!(value.whole_microseconds(), -172_803_456_789);
        assert_eq!(value.subsec_microseconds(), -456_789);
        assert_eq!(value.whole_nanoseconds(), -172_803_456_789_000_i128);
        assert_eq!(value.subsec_nanoseconds(), -456_789_000);

        // Sub-second values report zero for every coarser unit.
        let value = SignedDuration::milliseconds(-500);
        assert_eq!(value.whole_seconds(), 0);
        assert_eq!(value.whole_days(), 0);
        assert_eq!(value.whole_hours(), 0);
        assert_eq!(value.whole_milliseconds(), -500);
        assert_eq!(value.subsec_milliseconds(), -500);
        assert!(value.is_negative());

        // Whole-second values report zero subsecond remainders.
        let value = SignedDuration::seconds(3);
        assert_eq!(value.subsec_milliseconds(), 0);
        assert_eq!(value.subsec_microseconds(), 0);
        assert_eq!(value.subsec_nanoseconds(), 0);

        // Fractional seconds convert exactly for binary fractions.
        assert_eq!(SignedDuration::new(90, 500_000_000).as_seconds_f64(), 90.5);
        assert_eq!(SignedDuration::milliseconds(250).as_seconds_f64(), 0.25);
        assert_eq!(SignedDuration::milliseconds(-250).as_seconds_f32(), -0.25);
        assert_eq!(SignedDuration::seconds(-1).as_seconds_f64(), -1.0);
    }

    #[test]
    fn checked_and_saturating_arithmetic() {
        assert_eq!(
            SignedDuration::seconds(5).checked_add(SignedDuration::seconds(6)),
            Some(SignedDuration::seconds(11)),
        );
        assert_eq!(
            SignedDuration::seconds(5).checked_sub(SignedDuration::seconds(6)),
            Some(SignedDuration::seconds(-1)),
        );
        assert!(
            SignedDuration::MAX
                .checked_add(SignedDuration::SECOND)
                .is_none()
        );
        assert!(
            SignedDuration::MIN
                .checked_sub(SignedDuration::SECOND)
                .is_none()
        );
        assert!(
            SignedDuration::MIN
                .checked_add(SignedDuration::SECOND)
                .is_some()
        );
        assert!(
            SignedDuration::MAX
                .checked_sub(SignedDuration::SECOND)
                .is_some()
        );
        assert_eq!(
            SignedDuration::MAX.checked_add(SignedDuration::MIN),
            Some(SignedDuration::seconds(-1)),
        );

        // The operators saturate rather than overflowing.
        assert_eq!(SignedDuration::MAX + SignedDuration::SECOND, SignedDuration::MAX);
        assert_eq!(
            SignedDuration::MAX + SignedDuration::MILLISECOND,
            SignedDuration::MAX
        );
        assert_eq!(SignedDuration::MIN - SignedDuration::SECOND, SignedDuration::MIN);
        assert_eq!(
            SignedDuration::MIN - SignedDuration::MILLISECOND,
            SignedDuration::MIN
        );
        assert_eq!(
            SignedDuration::MAX.saturating_add(SignedDuration::SECOND),
            SignedDuration::MAX,
        );
        assert_eq!(
            SignedDuration::MIN.saturating_sub(SignedDuration::SECOND),
            SignedDuration::MIN,
        );
        assert_eq!(
            SignedDuration::seconds(5).saturating_add(SignedDuration::seconds(6)),
            SignedDuration::seconds(11),
        );

        // Multiplication.
        assert_eq!(
            SignedDuration::seconds(5).checked_mul(2),
            Some(SignedDuration::seconds(10))
        );
        assert_eq!(
            SignedDuration::seconds(5).checked_mul(0),
            Some(SignedDuration::ZERO)
        );
        assert!(SignedDuration::MIN.checked_mul(2).is_none());
        assert!(
            SignedDuration::seconds(5)
                .checked_div(0)
                .is_none()
        );
        assert_eq!(
            SignedDuration::hours(3).checked_div(2),
            Some(SignedDuration::minutes(90)),
        );
        assert_eq!(SignedDuration::MAX.saturating_mul(2), SignedDuration::MAX,);
        assert_eq!(SignedDuration::MIN.saturating_mul(2), SignedDuration::MIN,);
        assert_eq!(
            SignedDuration::seconds(3).saturating_mul(2),
            SignedDuration::seconds(6),
        );

        // Negation.
        assert_eq!(
            SignedDuration::seconds(5).checked_neg(),
            Some(SignedDuration::seconds(-5))
        );
        assert!(SignedDuration::MIN.checked_neg().is_none());
        assert_eq!(
            -SignedDuration::MAX,
            SignedDuration::seconds(-i64::MAX) - SignedDuration::nanoseconds(999_999_999)
        );
        // Negating the minimum value saturates to the maximum instead of returning itself.
        assert_eq!(-SignedDuration::MIN, SignedDuration::MAX);
    }

    #[test]
    fn operator_traits_add_and_subtract() {
        let hour = SignedDuration::HOUR;
        assert_eq!(hour + SignedDuration::MINUTE, SignedDuration::minutes(61));
        assert_eq!(hour - SignedDuration::minutes(90), SignedDuration::minutes(-30));
        assert_eq!(hour + SignedDuration::ZERO, hour);

        let mut value = hour;
        value += SignedDuration::MINUTE;
        assert_eq!(value, SignedDuration::minutes(61));
        value -= SignedDuration::MINUTE;
        assert_eq!(value, hour);

        // Arithmetic with standard durations.
        assert_eq!(hour + StdDuration::from_secs(60), SignedDuration::minutes(61));
        assert_eq!(hour - StdDuration::from_secs(120), SignedDuration::minutes(58));
        assert_eq!(StdDuration::from_secs(60) + hour, SignedDuration::minutes(61));
        assert_eq!(StdDuration::from_secs(120) - hour, SignedDuration::minutes(-58));
        let mut value = hour;
        value += StdDuration::from_secs(60);
        assert_eq!(value, SignedDuration::minutes(61));
        value -= StdDuration::from_secs(60);
        assert_eq!(value, hour);

        // A standard duration too large to convert saturates at the bounds.
        let huge = StdDuration::from_secs(u64::MAX);
        assert_eq!(hour + huge, SignedDuration::MAX);
        // Subtracting a std duration too large to convert saturates towards `-MAX`.
        assert_eq!(hour - huge, -SignedDuration::MAX + SignedDuration::HOUR);
        assert_eq!(SignedDuration::ZERO - huge, -SignedDuration::MAX);
        let mut value = huge;
        value += hour;
        assert_eq!(value, StdDuration::MAX);

        // Subtracting more than the receiver holds saturates at zero.
        let mut value = StdDuration::ZERO;
        value -= SignedDuration::seconds(1);
        assert_eq!(value, StdDuration::ZERO);
        let mut value = StdDuration::from_millis(500);
        value -= SignedDuration::seconds(1);
        assert_eq!(value, StdDuration::ZERO);
        // Adding a negative duration subtracts its magnitude.
        let mut value = StdDuration::from_secs(10);
        value += SignedDuration::seconds(-4);
        assert_eq!(value, StdDuration::from_secs(6));
        let mut value = StdDuration::from_secs(4);
        value -= SignedDuration::seconds(-6);
        assert_eq!(value, StdDuration::from_secs(10));
    }

    #[test]
    fn multiplication_and_division() {
        let value = SignedDuration::minutes(10);
        assert_eq!(value * 2i32, SignedDuration::minutes(20));
        assert_eq!(2i32 * value, SignedDuration::minutes(20));
        assert_eq!(value * 2u8, SignedDuration::minutes(20));
        let zero = 0i32;
        assert_eq!(value * zero, SignedDuration::ZERO);
        assert_eq!(value / 4i32, SignedDuration::seconds(150));
        let mut value = SignedDuration::minutes(10);
        value *= 3i32;
        assert_eq!(value, SignedDuration::minutes(30));
        value /= 3i32;
        assert_eq!(value, SignedDuration::minutes(10));

        // Scaling by floats goes through the float constructors.
        assert_eq!(SignedDuration::HOUR * 2.5, SignedDuration::minutes(150));
        assert_eq!(2.5 * SignedDuration::HOUR, SignedDuration::minutes(150));
        assert_eq!(SignedDuration::HOUR * 0.5f32, SignedDuration::minutes(30));
        assert_eq!(SignedDuration::HOUR * -1.0, SignedDuration::minutes(-60));
        let mut value = SignedDuration::HOUR;
        value *= 2.0;
        assert_eq!(value, SignedDuration::minutes(120));
        value /= 4.0;
        assert_eq!(value, SignedDuration::minutes(30));

        // Ratios produce floats.
        assert_eq!(SignedDuration::HOUR / SignedDuration::HOUR, 1.0);
        assert_eq!(SignedDuration::hours(1) / SignedDuration::minutes(1), 60.0);
        assert_eq!(SignedDuration::hours(1) / StdDuration::from_secs(1_800), 2.0);
        assert_eq!(StdDuration::from_secs(1_800) / SignedDuration::hours(1), 0.5);
        assert_eq!(SignedDuration::HOUR / 4i32, SignedDuration::minutes(15));
        assert_eq!(SignedDuration::HOUR / 2.0, SignedDuration::minutes(30));
        assert_eq!(SignedDuration::HOUR / 2.0f32, SignedDuration::minutes(30));
    }

    #[test]
    fn float_constructors() {
        assert_eq!(
            SignedDuration::seconds_f64(1.5),
            SignedDuration::new(1, 500_000_000)
        );
        assert_eq!(
            SignedDuration::seconds_f64(-1.5),
            SignedDuration::new(-1, -500_000_000)
        );
        assert_eq!(SignedDuration::seconds_f64(0.0), SignedDuration::ZERO);
        assert_eq!(SignedDuration::seconds_f64(f64::NAN), SignedDuration::ZERO);
        assert_eq!(SignedDuration::seconds_f64(1e30), SignedDuration::MAX);
        assert_eq!(SignedDuration::seconds_f64(-1e30), SignedDuration::MIN);
        assert_eq!(
            SignedDuration::saturating_seconds_f64(f64::NAN),
            SignedDuration::ZERO
        );
        assert_eq!(SignedDuration::saturating_seconds_f64(1e30), SignedDuration::MAX);
        assert_eq!(SignedDuration::saturating_seconds_f64(-1e30), SignedDuration::MIN);
        assert!(SignedDuration::checked_seconds_f64(1e30).is_none());
        assert!(SignedDuration::checked_seconds_f64(-1e30).is_none());
        assert!(SignedDuration::checked_seconds_f64(f64::NAN).is_none());
        assert_eq!(
            SignedDuration::checked_seconds_f64(2.0),
            Some(SignedDuration::seconds(2))
        );

        assert_eq!(
            SignedDuration::seconds_f32(1.5),
            SignedDuration::new(1, 500_000_000)
        );
        assert_eq!(SignedDuration::seconds_f32(f32::NAN), SignedDuration::ZERO);
        assert_eq!(SignedDuration::seconds_f32(1e30), SignedDuration::MAX);
        assert_eq!(SignedDuration::seconds_f32(-1e30), SignedDuration::MIN);
        assert!(SignedDuration::checked_seconds_f32(f32::NAN).is_none());
        assert_eq!(
            SignedDuration::checked_seconds_f32(0.5),
            Some(SignedDuration::milliseconds(500))
        );
        assert_eq!(SignedDuration::saturating_seconds_f32(-1e30), SignedDuration::MIN);
    }

    #[test]
    fn ordering_equality_and_hashing() {
        let values = [
            SignedDuration::days(-1),
            SignedDuration::minutes(-1),
            SignedDuration::nanoseconds(-1),
            SignedDuration::ZERO,
            SignedDuration::nanoseconds(1),
            SignedDuration::minutes(1),
            SignedDuration::days(1),
        ];
        for (index, value) in values.iter().enumerate() {
            for (other_index, other) in values.iter().enumerate() {
                let expected = index.cmp(&other_index);
                assert_eq!(value.cmp(other), expected, "{value:?} vs {other:?}");
                assert_eq!(value.partial_cmp(other), Some(expected));
                assert_eq!(value == other, index == other_index);
            }
        }
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));

        // Equivalent values built in different ways compare and hash identically.
        let a = SignedDuration::seconds(90);
        let b = SignedDuration::minutes(1) + SignedDuration::seconds(30);
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
        let a = SignedDuration::new(1, -1);
        let b = SignedDuration::nanoseconds(999_999_999);
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));

        assert_ne!(SignedDuration::SECOND, SignedDuration::seconds(2));
        assert_ne!(
            hash_of(&SignedDuration::SECOND),
            hash_of(&SignedDuration::seconds(2))
        );

        // Comparison against `Duration` agrees with the signed comparison.
        assert_eq!(SignedDuration::seconds(5), StdDuration::from_secs(5));
        assert_eq!(StdDuration::from_secs(5), SignedDuration::seconds(5));
        assert!(SignedDuration::seconds(5) > StdDuration::from_secs(4));
        assert!(SignedDuration::seconds(5) < StdDuration::from_secs(6));
        assert!(StdDuration::from_secs(6) > SignedDuration::seconds(5));
        assert!(SignedDuration::seconds(-1) < StdDuration::ZERO);
        // Durations beyond `i64::MAX` seconds always compare as greater.
        assert_eq!(
            SignedDuration::MAX.partial_cmp(&StdDuration::from_secs(u64::MAX)),
            Some(Ordering::Less),
        );
        assert_eq!(
            StdDuration::from_secs(u64::MAX).partial_cmp(&SignedDuration::MAX),
            Some(Ordering::Greater),
        );
    }

    #[test]
    fn try_from_conversions() {
        assert_eq!(
            SignedDuration::try_from(StdDuration::from_secs(5)),
            Ok(SignedDuration::seconds(5)),
        );
        assert_eq!(SignedDuration::try_from(StdDuration::MAX), Err(ConversionRange),);
        assert_eq!(
            StdDuration::try_from(SignedDuration::seconds(5)),
            Ok(StdDuration::from_secs(5)),
        );
        assert_eq!(
            StdDuration::try_from(SignedDuration::seconds(-1)),
            Err(ConversionRange),
        );
        assert_eq!(
            StdDuration::try_from(SignedDuration::new(-1, -1)),
            Err(ConversionRange),
        );
        // Positive durations with a subsecond component convert exactly.
        assert_eq!(
            StdDuration::try_from(SignedDuration::milliseconds(1_500)),
            Ok(StdDuration::from_millis(1_500)),
        );
        assert_eq!(
            StdDuration::try_from(SignedDuration::MAX),
            Ok(StdDuration::new(9_223_372_036_854_775_807, 999_999_999)),
        );
    }

    #[test]
    fn sum_combines_durations() {
        let parts = [
            SignedDuration::seconds(1),
            SignedDuration::seconds(2),
            SignedDuration::seconds(3),
            SignedDuration::seconds(-4),
        ];
        assert_eq!(
            parts.into_iter().sum::<SignedDuration>(),
            SignedDuration::seconds(2)
        );
        assert_eq!(parts.iter().sum::<SignedDuration>(), SignedDuration::seconds(2));
        assert_eq!(
            core::iter::empty::<SignedDuration>().sum::<SignedDuration>(),
            SignedDuration::ZERO,
        );
        assert_eq!(
            core::iter::empty::<&SignedDuration>().sum::<SignedDuration>(),
            SignedDuration::ZERO,
        );
    }

    #[test]
    fn system_time_operators_terminate() {
        // `SystemTime` is a zero-sized marker type, so the arithmetic can only return the
        // value itself; what matters is that it does not recurse.
        let added: SystemTime = SystemTime + SignedDuration::HOUR;
        assert!(matches!(added, SystemTime));
        let subbed: SystemTime = SystemTime - SignedDuration::HOUR;
        assert!(matches!(subbed, SystemTime));
        let added: SystemTime = SystemTime + SignedDuration::ZERO;
        assert!(matches!(added, SystemTime));
        let mut value = SystemTime;
        value += SignedDuration::HOUR;
        assert!(matches!(value, SystemTime));
        value -= SignedDuration::HOUR;
        assert!(matches!(value, SystemTime));
        let added: SystemTime = SystemTime + SignedDuration::MIN;
        assert!(matches!(added, SystemTime));
    }

    #[test]
    fn debug_formats_fields() {
        assert_eq!(
            format!("{:?}", SignedDuration::new(1, 500_000_000)),
            "SignedDuration { seconds: 1, nanoseconds: 500000000 }",
        );
        assert_eq!(
            format!("{:?}", SignedDuration::seconds(-2)),
            "SignedDuration { seconds: -2, nanoseconds: 0 }",
        );
        assert_eq!(
            format!("{:?}", SignedDuration::ZERO),
            "SignedDuration { seconds: 0, nanoseconds: 0 }"
        );
    }
}
