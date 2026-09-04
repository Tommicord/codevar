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

extern crate core;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use codevar_core::timeutil::date_month::Month;
use codevar_core::timeutil::date_plain::PlainDateTime;
use codevar_core::timeutil::date_signed_duration::SignedDuration;
use codevar_core::timeutil::date_time::Time;
use codevar_core::timeutil::date_unit::{
    Day, Hour, Microsecond, Millisecond, Minute, Nanosecond, Second,
};
use codevar_core::timeutil::date_utc_offset::UtcOffset;
use codevar_core::timeutil::date_weekday::Weekday;

#[test]
fn time_boundary_values() {
    // Test MIDNIGHT (smallest possible value)
    assert_eq!(Time::MIDNIGHT.hour(), 0);
    assert_eq!(Time::MIDNIGHT.minute(), 0);
    assert_eq!(Time::MIDNIGHT.second(), 0);
    assert_eq!(Time::MIDNIGHT.nanosecond(), 0);
    assert_eq!(Time::MIDNIGHT.as_hms(), (0, 0, 0));
    assert_eq!(Time::MIDNIGHT.as_hms_nano(), (0, 0, 0, 0));

    // Test MAX (largest possible value)
    assert_eq!(Time::MAX.hour(), 23);
    assert_eq!(Time::MAX.minute(), 59);
    assert_eq!(Time::MAX.second(), 59);
    assert_eq!(Time::MAX.nanosecond(), 999_999_999);
    assert_eq!(Time::MAX.as_hms(), (23, 59, 59));
    assert_eq!(Time::MAX.as_hms_nano(), (23, 59, 59, 999_999_999));
}

#[test]
fn time_component_boundaries() {
    // Test minimum valid values for each component
    let min_time = Time::from_hms(0, 0, 0).unwrap();
    assert_eq!(min_time, Time::MIDNIGHT);

    // Test maximum valid values for each component
    let max_time = Time::from_hms(23, 59, 59).unwrap();
    assert_eq!(max_time.hour(), 23);
    assert_eq!(max_time.minute(), 59);
    assert_eq!(max_time.second(), 59);

    // Test with maximum nanoseconds
    let max_nano = Time::from_hms_nano(23, 59, 59, 999_999_999).unwrap();
    assert_eq!(max_nano, Time::MAX);
}

#[test]
fn time_rejects_invalid_components() {
    // Invalid hours
    assert!(Time::from_hms(24, 0, 0).is_err());
    assert!(Time::from_hms(100, 0, 0).is_err());

    // Invalid minutes
    assert!(Time::from_hms(0, 60, 0).is_err());
    assert!(Time::from_hms(0, 100, 0).is_err());

    // Invalid seconds
    assert!(Time::from_hms(0, 0, 60).is_err());
    assert!(Time::from_hms(0, 0, 100).is_err());

    // Invalid milliseconds
    assert!(Time::from_hms_milli(0, 0, 0, 1000).is_err());
    assert!(Time::from_hms_milli(0, 0, 0, 5000).is_err());

    // Invalid microseconds
    assert!(Time::from_hms_micro(0, 0, 0, 1_000_000).is_err());
    assert!(Time::from_hms_micro(0, 0, 0, 5_000_000).is_err());

    // Invalid nanoseconds
    assert!(Time::from_hms_nano(0, 0, 0, 1_000_000_000).is_err());
    assert!(Time::from_hms_nano(0, 0, 0, 4_000_000_000).is_err());
}

#[test]
fn time_subsecond_conversions() {
    // Test millisecond conversion
    let time = Time::from_hms_milli(12, 30, 45, 500).unwrap();
    assert_eq!(time.millisecond(), 500);
    assert_eq!(time.microsecond(), 500_000);
    assert_eq!(time.nanosecond(), 500_000_000);

    // Test microsecond conversion
    let time = Time::from_hms_micro(12, 30, 45, 500_000).unwrap();
    assert_eq!(time.millisecond(), 500);
    assert_eq!(time.microsecond(), 500_000);
    assert_eq!(time.nanosecond(), 500_000_000);

    // Test nanosecond conversion
    let time = Time::from_hms_nano(12, 30, 45, 500_000_000).unwrap();
    assert_eq!(time.millisecond(), 500);
    assert_eq!(time.microsecond(), 500_000);
    assert_eq!(time.nanosecond(), 500_000_000);

    // Test zero subsecond
    let time = Time::from_hms(12, 30, 45).unwrap();
    assert_eq!(time.millisecond(), 0);
    assert_eq!(time.microsecond(), 0);
    assert_eq!(time.nanosecond(), 0);

    // Test maximum subsecond values
    let time = Time::from_hms_nano(12, 30, 45, 999_999_999).unwrap();
    assert_eq!(time.millisecond(), 999);
    assert_eq!(time.microsecond(), 999_999);
    assert_eq!(time.nanosecond(), 999_999_999);
}

#[test]
fn time_duration_until_crosses_midnight() {
    // Test duration calculation that crosses midnight
    let early = Time::from_hms(23, 59, 59).unwrap();
    let late = Time::from_hms(0, 0, 1).unwrap();
    let duration = early.duration_until(late);

    // Should be 2 seconds (wrapping around midnight)
    assert_eq!(duration.whole_seconds(), 2);
    assert_eq!(duration.subsec_nanoseconds(), 0);

    // Test reverse direction
    let duration_rev = late.duration_since(early);
    assert_eq!(duration_rev.whole_seconds(), 2);
}

#[test]
fn time_duration_with_subseconds() {
    let t1 = Time::from_hms_nano(12, 0, 0, 0).unwrap();
    let t2 = Time::from_hms_nano(12, 0, 1, 500_000_000).unwrap();
    let duration = t1.duration_until(t2);

    assert_eq!(duration.whole_seconds(), 1);
    assert_eq!(duration.subsec_nanoseconds(), 500_000_000);

    // Test with microsecond precision
    let t3 = Time::from_hms_micro(12, 0, 0, 500_000).unwrap();
    let t4 = Time::from_hms_micro(12, 0, 1, 0).unwrap();
    let duration_micro = t3.duration_until(t4);

    assert_eq!(duration_micro.whole_seconds(), 0);
    assert_eq!(duration_micro.subsec_nanoseconds(), 500_000_000);
}

#[test]
fn time_comparison_and_ordering() {
    let t1 = Time::from_hms(10, 30, 0).unwrap();
    let t2 = Time::from_hms(10, 30, 1).unwrap();
    let t3 = Time::from_hms(10, 30, 0).unwrap();

    assert!(t1 < t2);
    assert!(t2 > t1);
    assert_eq!(t1, t3);
    assert!(t1 <= t2);
    assert!(t2 >= t1);

    // Test with subseconds
    let t4 = Time::from_hms_nano(10, 30, 0, 1).unwrap();
    assert!(t1 < t4);
    assert!(t4 > t1);

    // Test boundary values
    assert!(Time::MIDNIGHT < Time::MAX);
    assert!(Time::MAX > Time::MIDNIGHT);
}

#[test]
fn time_hash_consistency() {
    let t1 = Time::from_hms(12, 30, 45).unwrap();
    let t2 = Time::from_hms(12, 30, 45).unwrap();
    let t3 = Time::from_hms(12, 30, 46).unwrap();

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    let mut h3 = DefaultHasher::new();

    t1.hash(&mut h1);
    t2.hash(&mut h2);
    t3.hash(&mut h3);

    assert_eq!(h1.finish(), h2.finish());
    assert_ne!(h1.finish(), h3.finish());
}

#[test]
fn time_as_hms_variants() {
    let time = Time::from_hms_nano(14, 25, 36, 123_456_789).unwrap();

    assert_eq!(time.as_hms(), (14, 25, 36));
    assert_eq!(time.as_hms_milli(), (14, 25, 36, 123));
    assert_eq!(time.as_hms_micro(), (14, 25, 36, 123_456));
    assert_eq!(time.as_hms_nano(), (14, 25, 36, 123_456_789));
}

#[test]
fn signed_duration_constants() {
    assert_eq!(SignedDuration::ZERO.whole_seconds(), 0);
    assert!(SignedDuration::ZERO.is_zero());
    assert!(!SignedDuration::ZERO.is_positive());
    assert!(!SignedDuration::ZERO.is_negative());

    assert_eq!(SignedDuration::NANOSECOND.whole_nanoseconds(), 1);
    assert_eq!(SignedDuration::MICROSECOND.whole_nanoseconds(), 1_000);
    assert_eq!(SignedDuration::MILLISECOND.whole_nanoseconds(), 1_000_000);
    assert_eq!(SignedDuration::SECOND.whole_nanoseconds(), 1_000_000_000);
    assert_eq!(SignedDuration::MINUTE.whole_nanoseconds(), 60_000_000_000);
    assert_eq!(SignedDuration::HOUR.whole_nanoseconds(), 3_600_000_000_000);
    assert_eq!(SignedDuration::DAY.whole_nanoseconds(), 86_400_000_000_000);
    assert_eq!(
        SignedDuration::WEEK.whole_nanoseconds(),
        604_800_000_000_000
    );
}

#[test]
fn signed_duration_boundary_values() {
    // Test MIN
    assert!(SignedDuration::MIN.is_negative());
    assert!(!SignedDuration::MIN.is_positive());
    assert!(!SignedDuration::MIN.is_zero());

    // Test MAX
    assert!(SignedDuration::MAX.is_positive());
    assert!(!SignedDuration::MAX.is_negative());
    assert!(!SignedDuration::MAX.is_zero());

    // Test that abs() saturates at MAX
    assert_eq!(SignedDuration::MIN.abs(), SignedDuration::MAX);
    assert_eq!(SignedDuration::MAX.abs(), SignedDuration::MAX);
}

#[test]
fn signed_duration_normalization() {
    // Test nanoseconds wrapping into seconds
    let d1 = SignedDuration::new(0, 1_500_000_000);
    assert_eq!(d1.whole_seconds(), 1);
    assert_eq!(d1.subsec_nanoseconds(), 500_000_000);

    // Test negative nanoseconds wrapping
    let d2 = SignedDuration::new(1, -500_000_000);
    assert_eq!(d2.whole_seconds(), 0);
    assert_eq!(d2.subsec_nanoseconds(), 500_000_000);

    // Test positive seconds with negative nanoseconds
    let d3 = SignedDuration::new(2, -100_000_000);
    assert_eq!(d3.whole_seconds(), 1);
    assert_eq!(d3.subsec_nanoseconds(), 900_000_000);

    // Test negative seconds with positive nanoseconds
    let d4 = SignedDuration::new(-2, 100_000_000);
    assert_eq!(d4.whole_seconds(), -1);
    assert_eq!(d4.subsec_nanoseconds(), -900_000_000);
}

#[test]
fn signed_duration_arithmetic_overflow() {
    // Test addition overflow
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

    // Test saturating arithmetic
    assert_eq!(
        SignedDuration::MAX.saturating_add(SignedDuration::SECOND),
        SignedDuration::MAX
    );
    assert_eq!(
        SignedDuration::MIN.saturating_sub(SignedDuration::SECOND),
        SignedDuration::MIN
    );

    // Test valid arithmetic
    let d1 = SignedDuration::days(1);
    let d2 = SignedDuration::hours(12);
    let sum = d1.checked_add(d2).unwrap();
    assert_eq!(sum.whole_seconds(), 36 * 3600); // 1.5 days
}

#[test]
fn signed_duration_comparison() {
    let d1 = SignedDuration::seconds(10);
    let d2 = SignedDuration::seconds(20);
    let d3 = SignedDuration::seconds(10);
    let d4 = SignedDuration::seconds(-10);

    assert!(d1 < d2);
    assert!(d2 > d1);
    assert_eq!(d1, d3);
    assert!(d1 > d4);
    assert!(d4 < d1);

    // Test with subseconds
    let d5 = SignedDuration::new(10, 500_000_000);
    assert!(d1 < d5);
    assert!(d5 > d1);
}

#[test]
fn signed_duration_division_and_multiplication() {
    let d = SignedDuration::seconds(100);

    // Test division
    assert_eq!(d / SignedDuration::seconds(10), 10.0);
    assert_eq!(d / 10, SignedDuration::seconds(10));

    // Test multiplication
    assert_eq!(SignedDuration::seconds(10) * 10, d);
    assert_eq!(10 * SignedDuration::seconds(10), d);

    // Test with subseconds
    let d_sub = SignedDuration::new(1, 500_000_000);
    assert_eq!(d_sub * 2, SignedDuration::seconds(3));
}

#[test]
fn signed_duration_float_constructors() {
    // Test f64 construction
    let d1 = SignedDuration::seconds_f64(1.5);
    assert_eq!(d1.whole_seconds(), 1);
    assert_eq!(d1.subsec_nanoseconds(), 500_000_000);

    // Test f32 construction
    let d2 = SignedDuration::seconds_f32(2.5);
    assert_eq!(d2.whole_seconds(), 2);
    assert_eq!(d2.subsec_nanoseconds(), 500_000_000);

    // Test negative values
    let d3 = SignedDuration::seconds_f64(-1.5);
    assert!(d3.is_negative());
    assert_eq!(d3.whole_seconds(), -1);
    assert_eq!(d3.subsec_nanoseconds(), -500_000_000);

    // Test very small values
    let d4 = SignedDuration::seconds_f64(0.000_000_001);
    assert_eq!(d4.whole_nanoseconds(), 1);

    // Test saturating behavior
    assert_eq!(
        SignedDuration::seconds_f64(f64::INFINITY),
        SignedDuration::MAX
    );
    assert_eq!(
        SignedDuration::seconds_f64(f64::NEG_INFINITY),
        SignedDuration::MIN
    );
    assert_eq!(SignedDuration::seconds_f64(f64::NAN), SignedDuration::ZERO);
}

#[test]
fn signed_duration_unsigned_abs() {
    let d1 = SignedDuration::seconds(10);
    let d2 = SignedDuration::seconds(-10);

    assert_eq!(d1.unsigned_abs().as_secs(), 10);
    assert_eq!(d2.unsigned_abs().as_secs(), 10);
    assert_eq!(d1.unsigned_abs(), d2.unsigned_abs());
}

#[test]
fn utc_offset_boundary_values() {
    // Test UTC
    assert_eq!(UtcOffset::UTC.whole_hours(), 0);
    assert_eq!(UtcOffset::UTC.whole_minutes(), 0);
    assert_eq!(UtcOffset::UTC.whole_seconds(), 0);
    assert!(UtcOffset::UTC.is_utc());
    assert!(!UtcOffset::UTC.is_positive());
    assert!(!UtcOffset::UTC.is_negative());

    // Test maximum positive offset
    let max_pos = UtcOffset::from_hms(25, 59, 59).unwrap();
    assert_eq!(max_pos.whole_hours(), 25);
    assert_eq!(max_pos.whole_minutes(), 25 * 60 + 59);
    assert!(max_pos.is_positive());

    // Test maximum negative offset
    let max_neg = UtcOffset::from_hms(-25, 59, 59).unwrap();
    assert_eq!(max_neg.whole_hours(), -25);
    assert!(max_neg.is_negative());
}

#[test]
fn utc_offset_rejects_invalid_values() {
    // Invalid hours
    assert!(UtcOffset::from_hms(26, 0, 0).is_err());
    assert!(UtcOffset::from_hms(-26, 0, 0).is_err());

    // Invalid minutes
    assert!(UtcOffset::from_hms(0, 60, 0).is_err());
    assert!(UtcOffset::from_hms(0, -60, 0).is_err());

    // Invalid seconds
    assert!(UtcOffset::from_hms(0, 0, 60).is_err());
    assert!(UtcOffset::from_hms(0, 0, -60).is_err());
}

#[test]
fn utc_offset_sign_normalization() {
    // Test that signs are normalized
    let offset1 = UtcOffset::from_hms(5, -30, 0).unwrap();
    assert_eq!(offset1.as_hms(), (5, 30, 0)); // minutes sign flipped

    let offset2 = UtcOffset::from_hms(-5, 30, 0).unwrap();
    assert_eq!(offset2.as_hms(), (-5, -30, 0)); // minutes sign flipped

    let offset3 = UtcOffset::from_hms(5, 30, -30).unwrap();
    assert_eq!(offset3.as_hms(), (5, 30, 30)); // seconds sign flipped
}

#[test]
fn utc_offset_from_whole_seconds() {
    // Test various second values
    let offset1 = UtcOffset::from_whole_seconds(3600).unwrap(); // 1 hour
    assert_eq!(offset1.as_hms(), (1, 0, 0));

    let offset2 = UtcOffset::from_whole_seconds(5400).unwrap(); // 1.5 hours
    assert_eq!(offset2.as_hms(), (1, 30, 0));

    let offset3 = UtcOffset::from_whole_seconds(3661).unwrap(); // 1h 1m 1s
    assert_eq!(offset3.as_hms(), (1, 1, 1));

    let offset4 = UtcOffset::from_whole_seconds(-3600).unwrap(); // -1 hour
    assert_eq!(offset4.as_hms(), (-1, 0, 0));
}

#[test]
fn utc_offset_comparison() {
    let offset1 = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset2 = UtcOffset::from_hms(6, 0, 0).unwrap();
    let offset3 = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset4 = UtcOffset::from_hms(-5, 30, 0).unwrap();

    assert!(offset1 < offset2);
    assert!(offset2 > offset1);
    assert_eq!(offset1, offset3);
    assert!(offset1 > offset4);
    assert!(offset4 < offset1);
}

#[test]
fn utc_offset_negation() {
    let offset1 = UtcOffset::from_hms(5, 30, 0).unwrap();
    let negated = -offset1;
    assert_eq!(negated.as_hms(), (-5, -30, 0));

    let offset2 = UtcOffset::from_hms(-8, 0, 0).unwrap();
    let negated2 = -offset2;
    assert_eq!(negated2.as_hms(), (8, 0, 0));

    // UTC should remain UTC
    assert_eq!(-UtcOffset::UTC, UtcOffset::UTC);
}

#[test]
fn utc_offset_hash_consistency() {
    let offset1 = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset2 = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset3 = UtcOffset::from_hms(5, 31, 0).unwrap();

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    let mut h3 = DefaultHasher::new();

    offset1.hash(&mut h1);
    offset2.hash(&mut h2);
    offset3.hash(&mut h3);

    assert_eq!(h1.finish(), h2.finish());
    assert_ne!(h1.finish(), h3.finish());
}

#[test]
fn weekday_navigation() {
    // Test previous/next for all weekdays
    assert_eq!(Weekday::Monday.previous(), Weekday::Sunday);
    assert_eq!(Weekday::Monday.next(), Weekday::Tuesday);

    assert_eq!(Weekday::Sunday.previous(), Weekday::Saturday);
    assert_eq!(Weekday::Sunday.next(), Weekday::Monday);

    // Test that navigation is cyclic
    assert_eq!(Weekday::Monday.previous().next(), Weekday::Monday);
    assert_eq!(Weekday::Sunday.next().previous(), Weekday::Sunday);
}

#[test]
fn weekday_nth_navigation() {
    // Test nth_next
    assert_eq!(Weekday::Monday.nth_next(0), Weekday::Monday);
    assert_eq!(Weekday::Monday.nth_next(1), Weekday::Tuesday);
    assert_eq!(Weekday::Monday.nth_next(7), Weekday::Monday);
    assert_eq!(Weekday::Monday.nth_next(14), Weekday::Monday);

    // Test nth_prev
    assert_eq!(Weekday::Monday.nth_prev(0), Weekday::Monday);
    assert_eq!(Weekday::Monday.nth_prev(1), Weekday::Sunday);
    assert_eq!(Weekday::Monday.nth_prev(7), Weekday::Monday);
    assert_eq!(Weekday::Monday.nth_prev(14), Weekday::Monday);

    // Test large values
    assert_eq!(
        Weekday::Friday.nth_next(100),
        Weekday::Monday.nth_next(100 + 4)
    );
}

#[test]
fn weekday_numbering() {
    // Test number_from_monday (1-indexed)
    assert_eq!(Weekday::Monday.number_from_monday(), 1);
    assert_eq!(Weekday::Tuesday.number_from_monday(), 2);
    assert_eq!(Weekday::Wednesday.number_from_monday(), 3);
    assert_eq!(Weekday::Thursday.number_from_monday(), 4);
    assert_eq!(Weekday::Friday.number_from_monday(), 5);
    assert_eq!(Weekday::Saturday.number_from_monday(), 6);
    assert_eq!(Weekday::Sunday.number_from_monday(), 7);

    // Test number_from_sunday (1-indexed)
    assert_eq!(Weekday::Sunday.number_from_sunday(), 1);
    assert_eq!(Weekday::Monday.number_from_sunday(), 2);
    assert_eq!(Weekday::Tuesday.number_from_sunday(), 3);
    assert_eq!(Weekday::Wednesday.number_from_sunday(), 4);
    assert_eq!(Weekday::Thursday.number_from_sunday(), 5);
    assert_eq!(Weekday::Friday.number_from_sunday(), 6);
    assert_eq!(Weekday::Saturday.number_from_sunday(), 7);

    // Test zero-indexed versions
    assert_eq!(Weekday::Monday.number_days_from_monday(), 0);
    assert_eq!(Weekday::Sunday.number_days_from_sunday(), 0);
}

#[test]
fn weekday_parsing() {
    // Test valid parsing
    assert_eq!("Monday".parse::<Weekday>().unwrap(), Weekday::Monday);
    assert_eq!("Tuesday".parse::<Weekday>().unwrap(), Weekday::Tuesday);
    assert_eq!("Wednesday".parse::<Weekday>().unwrap(), Weekday::Wednesday);
    assert_eq!("Thursday".parse::<Weekday>().unwrap(), Weekday::Thursday);
    assert_eq!("Friday".parse::<Weekday>().unwrap(), Weekday::Friday);
    assert_eq!("Saturday".parse::<Weekday>().unwrap(), Weekday::Saturday);
    assert_eq!("Sunday".parse::<Weekday>().unwrap(), Weekday::Sunday);

    // Test invalid parsing
    assert!("Mon".parse::<Weekday>().is_err());
    assert!("Funday".parse::<Weekday>().is_err());
    assert!("".parse::<Weekday>().is_err());
}

#[test]
fn unit_type_per_constants() {
    // Test Nanosecond conversions
    assert_eq!(Nanosecond::per(Microsecond), 1_000_u16);
    assert_eq!(Nanosecond::per(Millisecond), 1_000_000_u32);
    assert_eq!(Nanosecond::per(Second), 1_000_000_000_u32);

    // Test Second conversions
    assert_eq!(Second::per(Minute), 60_u8);
    assert_eq!(Second::per(Hour), 3_600_u16);
    assert_eq!(Second::per(Day), 86_400_u32);

    // Test Minute conversions
    assert_eq!(Minute::per(Hour), 60_u8);
    assert_eq!(Minute::per(Day), 1_440_u16);

    // Test Hour conversions
    assert_eq!(Hour::per(Day), 24_u8);
}

#[test]
fn unit_type_per_t_with_different_types() {
    // Test returning different types
    assert_eq!(Second::per_t::<u8>(Minute), 60_u8);
    assert_eq!(Second::per_t::<u16>(Hour), 3_600_u16);
    assert_eq!(Second::per_t::<u32>(Day), 86_400_u32);
    assert_eq!(Second::per_t::<i32>(Day), 86_400_i32);
    assert_eq!(Second::per_t::<f64>(Day), 86_400_f64);
}

#[test]
fn plain_datetime_boundary_values() {
    // Test MIN
    assert_eq!(PlainDateTime::MIN.time(), Time::MIDNIGHT);
    assert_eq!(
        PlainDateTime::MIN.date(),
        codevar_core::timeutil::date::Date::MIN
    );

    // Test MAX
    assert_eq!(PlainDateTime::MAX.time(), Time::MAX);
    assert_eq!(
        PlainDateTime::MAX.date(),
        codevar_core::timeutil::date::Date::MAX
    );
}

#[test]
fn plain_datetime_component_access() {
    let date =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::February, 29)
            .unwrap();
    let time = Time::from_hms_nano(14, 30, 45, 123_456_789).unwrap();
    let dt = PlainDateTime::new(date, time);

    assert_eq!(dt.date(), date);
    assert_eq!(dt.time(), time);
    assert_eq!(dt.year(), 2024);
    assert_eq!(dt.month(), Month::February);
    assert_eq!(dt.day(), 29);
    assert_eq!(dt.hour(), 14);
    assert_eq!(dt.minute(), 30);
    assert_eq!(dt.second(), 45);
    assert_eq!(dt.nanosecond(), 123_456_789);
}

#[test]
fn plain_datetime_arithmetic_overflow() {
    let dt = PlainDateTime::new(
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::January, 1)
            .unwrap(),
        Time::MIDNIGHT,
    );

    // Test overflow on add
    assert!(dt.checked_add(SignedDuration::days(10_000_000)).is_none());
    assert_eq!(
        dt.saturating_add(SignedDuration::days(10_000_000)),
        PlainDateTime::MAX
    );

    // Test overflow on sub
    assert!(dt.checked_sub(SignedDuration::days(10_000_000)).is_none());
    assert_eq!(
        dt.saturating_sub(SignedDuration::days(10_000_000)),
        PlainDateTime::MIN
    );
}

#[test]
fn plain_datetime_replacement() {
    let date =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::February, 29)
            .unwrap();
    let time = Time::from_hms(12, 30, 45).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test replace_time
    let new_time = Time::from_hms(15, 45, 30).unwrap();
    let dt_with_new_time = dt.replace_time(new_time);
    assert_eq!(dt_with_new_time.time(), new_time);
    assert_eq!(dt_with_new_time.date(), date);

    // Test replace_date
    let new_date =
        codevar_core::timeutil::date::Date::from_calendar_date(2025, Month::March, 1)
            .unwrap();
    let dt_with_new_date = dt.replace_date(new_date);
    assert_eq!(dt_with_new_date.date(), new_date);
    assert_eq!(dt_with_new_date.time(), time);
}

#[test]
fn plain_datetime_truncation() {
    let date =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::February, 29)
            .unwrap();
    let time = Time::from_hms_nano(14, 30, 45, 123_456_789).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test truncate_to_day
    let truncated_day = dt.truncate_to_day();
    assert_eq!(truncated_day.time(), Time::MIDNIGHT);
    assert_eq!(truncated_day.date(), date);

    // Test truncate_to_hour
    let truncated_hour = dt.truncate_to_hour();
    assert_eq!(truncated_hour.hour(), 14);
    assert_eq!(truncated_hour.minute(), 0);
    assert_eq!(truncated_hour.second(), 0);
    assert_eq!(truncated_hour.nanosecond(), 0);

    // Test truncate_to_minute
    let truncated_minute = dt.truncate_to_minute();
    assert_eq!(truncated_minute.hour(), 14);
    assert_eq!(truncated_minute.minute(), 30);
    assert_eq!(truncated_minute.second(), 0);
    assert_eq!(truncated_minute.nanosecond(), 0);

    // Test truncate_to_second
    let truncated_second = dt.truncate_to_second();
    assert_eq!(truncated_second.hour(), 14);
    assert_eq!(truncated_second.minute(), 30);
    assert_eq!(truncated_second.second(), 45);
    assert_eq!(truncated_second.nanosecond(), 0);
}

#[test]
fn plain_datetime_comparison() {
    let date1 =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::January, 1)
            .unwrap();
    let date2 =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::January, 2)
            .unwrap();
    let time1 = Time::from_hms(12, 0, 0).unwrap();
    let time2 = Time::from_hms(13, 0, 0).unwrap();

    let dt1 = PlainDateTime::new(date1, time1);
    let dt2 = PlainDateTime::new(date1, time2);
    let dt3 = PlainDateTime::new(date2, time1);
    let dt4 = PlainDateTime::new(date1, time1);

    assert!(dt1 < dt2); // Same date, later time
    assert!(dt1 < dt3); // Later date, same time
    assert_eq!(dt1, dt4); // Same date and time
}

#[test]
fn plain_datetime_hash_consistency() {
    let date =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::February, 29)
            .unwrap();
    let time = Time::from_hms(12, 30, 45).unwrap();
    let dt1 = PlainDateTime::new(date, time);
    let dt2 = PlainDateTime::new(date, time);
    let dt3 = PlainDateTime::new(date, Time::from_hms(12, 30, 46).unwrap());

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    let mut h3 = DefaultHasher::new();

    dt1.hash(&mut h1);
    dt2.hash(&mut h2);
    dt3.hash(&mut h3);

    assert_eq!(h1.finish(), h2.finish());
    assert_ne!(h1.finish(), h3.finish());
}

#[test]
fn plain_datetime_assume_offset() {
    let date =
        codevar_core::timeutil::date::Date::from_calendar_date(2024, Month::February, 29)
            .unwrap();
    let time = Time::from_hms(12, 30, 45).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test assume_utc
    let utc_dt = dt.assume_utc();
    assert_eq!(utc_dt.date(), date);
    assert_eq!(utc_dt.time(), time);

    // Test assume_offset
    let offset = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset_dt = dt.assume_offset(offset);
    assert_eq!(offset_dt.date(), date);
    assert_eq!(offset_dt.time(), time);
    assert_eq!(offset_dt.offset(), offset);
}
