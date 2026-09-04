use std::time::Duration as StdDuration;

use codevar_core::timeutil::date_signed_duration::SignedDuration;
use codevar_core::timeutil::date_time::Time;

#[test]
fn time_constructors_and_accessors() {
    let t = Time::from_hms(13, 14, 15).unwrap();
    assert_eq!(t.hour(), 13);
    assert_eq!(t.minute(), 14);
    assert_eq!(t.second(), 15);
    assert_eq!(t.millisecond(), 0);
    assert_eq!(t.microsecond(), 0);
    assert_eq!(t.nanosecond(), 0);
    assert_eq!(t.as_hms(), (13, 14, 15));

    let with_ms = Time::from_hms_milli(1, 2, 3, 456).unwrap();
    assert_eq!(with_ms.as_hms_milli(), (1, 2, 3, 456));

    let with_us = Time::from_hms_micro(1, 2, 3, 654_321).unwrap();
    assert_eq!(with_us.as_hms_micro(), (1, 2, 3, 654_321));

    let with_ns = Time::from_hms_nano(1, 2, 3, 987_654_321).unwrap();
    assert_eq!(with_ns.as_hms_nano(), (1, 2, 3, 987_654_321));

    let replaced = t
        .replace_hour(7)
        .unwrap()
        .replace_minute(8)
        .unwrap()
        .replace_second(9)
        .unwrap();
    assert_eq!(replaced.as_hms(), (7, 8, 9));

    assert_eq!(t.truncate_to_hour(), Time::from_hms(13, 0, 0).unwrap());
    assert_eq!(t.truncate_to_minute(), Time::from_hms(13, 14, 0).unwrap());
    assert_eq!(t.truncate_to_second(), t);
}

#[test]
fn signed_duration_creation_and_arithmetic() {
    let dur = SignedDuration::new(2, 500_000_000);
    assert_eq!(dur.whole_seconds(), 2);
    assert_eq!(dur.whole_nanoseconds(), 2_500_000_000_i128);
    assert!(!dur.is_zero());
    assert!(dur.is_positive());

    let neg = -dur;
    assert!(neg.is_negative());
    assert_eq!(neg.abs(), dur);

    let days = SignedDuration::days(3);
    let hours = SignedDuration::hours(2);
    assert_eq!(
        days.checked_add(hours).unwrap(),
        SignedDuration::days(3) + SignedDuration::hours(2)
    );
    assert_eq!(
        days.checked_sub(hours).unwrap(),
        SignedDuration::days(3) - SignedDuration::hours(2)
    );

    let overflow = SignedDuration::MAX.checked_add(SignedDuration::seconds(1));
    assert!(overflow.is_none());

    assert_eq!(
        SignedDuration::MAX.saturating_add(SignedDuration::seconds(1)),
        SignedDuration::MAX
    );
    assert_eq!(
        SignedDuration::MIN.saturating_sub(SignedDuration::seconds(1)),
        SignedDuration::MIN
    );
}

#[test]
fn signed_duration_std_duration_roundtrip() {
    let std_dur = StdDuration::new(4, 250_000_000);
    let sd = SignedDuration::try_from(std_dur).unwrap();
    assert_eq!(sd.whole_seconds(), 4);
    assert_eq!(sd.whole_nanoseconds(), 4_250_000_000_i128);

    let roundtrip: StdDuration = sd.try_into().unwrap();
    assert_eq!(roundtrip, std_dur);

    let negative = SignedDuration::new(-2, 500_000_000);
    let rr: StdDuration = negative.try_into().unwrap();
    assert_eq!(rr.as_nanos(), -1_500_000_000_i128 as u128);
}

#[test]
fn signed_duration_float_constructors() {
    assert_eq!(
        SignedDuration::seconds_f64(1.5).whole_nanoseconds(),
        1_500_000_000_i128
    );
    assert_eq!(
        SignedDuration::seconds_f32(-1.5).whole_nanoseconds(),
        -1_500_000_000_i128
    );
    assert!(SignedDuration::checked_seconds_f64(f64::NAN).is_none());
    assert_eq!(
        SignedDuration::saturating_seconds_f64(f64::INFINITY),
        SignedDuration::MAX
    );
    assert_eq!(
        SignedDuration::saturating_seconds_f64(f64::NEG_INFINITY),
        SignedDuration::MIN
    );
}

#[test]
fn time_rejects_bad_components_and_replacement_edges() {
    assert!(Time::from_hms(24, 0, 0).is_err());
    assert!(Time::from_hms(12, 60, 0).is_err());
    assert!(Time::from_hms_nano(0, 0, 0, 1_000_000_000).is_err());
    assert!(Time::from_hms_milli(0, 0, 0, 1_000).is_err());
    assert!(Time::from_hms_micro(0, 0, 0, 1_000_000).is_err());

    let t = Time::from_hms(1, 2, 3).unwrap();
    assert!(t.replace_hour(24).is_err());
    assert!(t.replace_minute(60).is_err());
    assert!(t.replace_second(60).is_err());
    assert!(t.replace_nanosecond(1_000_000_000).is_err());
}

#[test]
fn signed_duration_normalizes_and_overflow_edges() {
    let d = SignedDuration::new(1, 1_500_000_000);
    assert_eq!(d.whole_seconds(), 2);
    assert_eq!(d.whole_nanoseconds(), 2_500_000_000_i128);

    let d2 = SignedDuration::new(-1, -1_500_000_000);
    assert_eq!(d2.whole_seconds(), -2);
    assert_eq!(d2.whole_nanoseconds(), -2_500_000_000_i128);

    assert!(
        SignedDuration::MAX
            .checked_add(SignedDuration::seconds(1))
            .is_none()
    );
    assert!(
        SignedDuration::MIN
            .checked_sub(SignedDuration::seconds(1))
            .is_none()
    );
}
