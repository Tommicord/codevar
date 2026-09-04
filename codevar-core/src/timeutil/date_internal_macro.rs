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

/// Division of integers, rounding the resulting value towards negative infinity.
macro_rules! div_floor {
    ($self:expr, $rhs:expr) => {
        match ($self, $rhs) {
            (this, rhs) => {
                let d = this / rhs;
                let r = this % rhs;

                // If the remainder is non-zero, we need to subtract one if the
                // signs of self and rhs differ, as this means we rounded upwards
                // instead of downwards. We do this branchlessly by creating a mask
                // which is all-ones iff the signs differ, and 0 otherwise. Then by
                // adding this mask (which corresponds to the signed value -1), we
                // get our correction.
                let correction = (this ^ rhs) >> (size_of_val(&this) * 8 - 1);
                if r != 0 { d + correction } else { d }
            }
        }
    };
}

/// Similar to `overflowing_add`, but returning the number of times that it overflowed. Contained to
/// a certain range and only overflows a maximum number of times.
macro_rules! carry {
    (@most_once $value:expr, $min:literal.. $max:expr) => {
        match ($value, $min, $max) {
            (value, min, max) => {
                if value >= min {
                    if value < max {
                        (value, 0)
                    } else {
                        (value - (max - min), 1)
                    }
                } else {
                    (value + (max - min), -1)
                }
            }
        }
    };
    (@most_twice $value:expr, $min:literal.. $max:expr) => {
        match ($value, $min, $max) {
            (value, min, max) => {
                if value >= min {
                    if value < max {
                        (value, 0)
                    } else if value < 2 * max - min {
                        (value - (max - min), 1)
                    } else {
                        (value - 2 * (max - min), 2)
                    }
                } else {
                    if value >= min - max {
                        (value + (max - min), -1)
                    } else {
                        (value + 2 * (max - min), -2)
                    }
                }
            }
        }
    };
    (@most_thrice $value:expr, $min:literal.. $max:expr) => {
        match ($value, $min, $max) {
            (value, min, max) => {
                if value >= min {
                    if value < max {
                        (value, 0)
                    } else if value < 2 * max - min {
                        (value - (max - min), 1)
                    } else if value < 3 * max - 2 * min {
                        (value - 2 * (max - min), 2)
                    } else {
                        (value - 3 * (max - min), 3)
                    }
                } else {
                    if value >= min - max {
                        (value + (max - min), -1)
                    } else if value >= 2 * (min - max) {
                        (value + 2 * (max - min), -2)
                    } else {
                        (value + 3 * (max - min), -3)
                    }
                }
            }
        }
    };
}

/// Cascade an out-of-bounds value.
macro_rules! cascade {
    (@ordinal ordinal) => {};
    (@year year) => {};

    // Cascade an out-of-bounds value from "from" to "to".
    ($from:ident in $min:literal.. $max:expr => $to:tt) => {
        #[allow(unused_comparisons, unused_assignments)]
        let min = $min;
        let max = $max;
        if $from >= max {
            $from -= max - min;
            $to += 1;
        } else if $from < min {
            $from += max - min;
            $to -= 1;
        }
    };

    // Special case the ordinal-to-year cascade, as it has different behavior.
    ($ordinal:ident => $year:ident) => {
        // We need to actually capture the idents. Without this, macro hygiene causes errors.
        cascade!(@ordinal $ordinal);
        cascade!(@year $year);

        let days_in_year_count = days_in_year($year).cast_signed();
        #[allow(unused_assignments)]
        if $ordinal > days_in_year_count {
            $ordinal -= days_in_year_count;
            $year += 1;
        } else if $ordinal < 1 {
            $year -= 1;
            // The year may now be out of range if it was previously `MIN_YEAR`. This macro arm is
            // only called in situations where this is possible, so branching on where this is a
            // possibility is not necessary.
            $ordinal += days_in_year($year).cast_signed();
        }
    };
}

/// Constructs a ranged integer, returning a `ComponentRange` error if the value is out of range.
macro_rules! ensure_ranged {
    ($type:ty : $value:ident) => {
        match <$type>::new($value) {
            Some(val) => val,
            None => {
                return Err(crate::timeutil::date_error::ComponentRange::unconditional(stringify!($value)));
            }
        }
    };

    ($type:ty : $value:ident ($name:literal)) => {
        match <$type>::new($value) {
            Some(val) => val,
            None => {
                return Err(crate::timeutil::date_error::ComponentRange::unconditional($name));
            }
        }
    };

    ($type:ty : $value:ident $(as $as_type:ident)? * $factor:expr) => {
        match ($value $(as $as_type)?).checked_mul($factor) {
            Some(val) => match <$type>::new(val) {
                Some(val) => val,
                None => {
                    return Err(crate::timeutil::date_error::ComponentRange::unconditional(stringify!($value)));
                }
            },
            None => {
                return Err(crate::timeutil::date_error::ComponentRange::unconditional(stringify!($value)));
            }
        }
    };
}

/// Try to unwrap an expression, returning if not possible.
///
/// This is similar to the `?` operator, but does not perform `.into()`. Because of this, it is
/// usable in `const` contexts.
macro_rules! const_try {
    ($e:expr) => {
        match $e {
            Ok(value) => value,
            Err(error) => {
                return Err(error);
            }
        }
    };
}

/// Try to unwrap an expression, returning if not possible.
///
/// This is similar to the `?` operator, but is usable in `const` contexts.
macro_rules! const_try_opt {
    ($e:expr) => {
        match $e {
            Some(value) => value,
            None => {
                return None;
            }
        }
    };
}

/// Macro to handle Result types with automatic error conversion.
/// This reduces code duplication by providing a consistent way to handle errors.
macro_rules! try_err {
    ($e:expr, $error_type:ty) => {
        match $e {
            Ok(value) => value,
            Err(err) => return Err(<$error_type>::from(err)),
        }
    };
}

/// Macro to handle Option types with automatic error conversion.
macro_rules! try_opt_err {
    ($e:expr, $error:expr) => {
        match $e {
            Some(value) => value,
            None => return Err($error),
        }
    };
}

pub(crate) use carry;
pub(crate) use cascade;
pub(crate) use const_try;
pub(crate) use const_try_opt;
pub(crate) use div_floor;
pub(crate) use ensure_ranged;
pub(crate) use try_err;
pub(crate) use try_opt_err;
