use std::time::Duration as StdDuration;
use std::time::SystemTime;

use codevar_core::timeutil::date::Date;
use codevar_core::timeutil::date_month::Month;
use codevar_core::timeutil::date_offset_time::OffsetDateTime;
use codevar_core::timeutil::date_time::Time;
use codevar_core::timeutil::date_timestamp::Timestamp;
use codevar_core::timeutil::date_utc_offset::UtcOffset;
use codevar_core::timeutil::date_utc_time::UtcDateTime;

#[test]
fn utc_offset_constructs_and_formats() {
    let offset = UtcOffset::from_hms(5, 30, 0).unwrap();
    assert_eq!(offset.as_hms(), (5, 30, 0));
    assert_eq!(offset.whole_hours(), 5);
    assert_eq!(offset.whole_minutes(), 30);
    assert_eq!(offset.whole_seconds(), 19_800);

    let utc = UtcOffset::UTC;
    assert_eq!(utc.as_hms(), (0, 0, 0));
    assert!(utc.is_utc());

    let negative = UtcOffset::from_whole_seconds(-1800).unwrap();
    assert_eq!(negative.as_hms(), (-0, -30, 0));
}

#[test]
fn utc_datetime_and_timestamp_roundtrip() {
    let epoch = UtcDateTime::UNIX_EPOCH;
    assert_eq!(epoch.unix_timestamp(), 0);
    assert_eq!(epoch.unix_timestamp_nanos(), 0);

    let date = Date::from_calendar_date(
        2024,
        codevar_core::timeutil::date_month::Month::February,
        29,
    )
    .unwrap();
    let t = Time::from_hms(12, 34, 56).unwrap();
    let dt = UtcDateTime::new(date, t);

    assert_eq!(dt.date(), date);
    assert_eq!(dt.time(), t);
    assert_eq!(dt.year(), 2024);
    assert_eq!(
        dt.month(),
        codevar_core::timeutil::date_month::Month::February
    );
    assert_eq!(dt.day(), 29);

    let ts = Timestamp::from_nanoseconds(dt.unix_timestamp_nanos()).unwrap();
    let roundtrip = ts.to_utc().unwrap();
    assert_eq!(roundtrip, dt);

    let from_seconds = Timestamp::from_seconds(1_234_567).unwrap();
    assert_eq!(from_seconds.as_seconds(), 1_234_567);
}

#[test]
fn timestamp_from_system_time_and_unix_fields() {
    let sys = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_234_567);
    let ts = Timestamp::from(sys);
    assert_eq!(ts.as_seconds(), 1_234_567);

    let nanos = Timestamp::from_nanoseconds(1_500_000_000_i128).unwrap();
    assert_eq!(nanos.as_nanoseconds(), 1_500_000_000_i128);

    let ms = Timestamp::from_milliseconds(1_234).unwrap();
    assert_eq!(ms.as_milliseconds(), 1_234);

    let us = Timestamp::from_microseconds(1_234_567).unwrap();
    assert_eq!(us.as_microseconds(), 1_234_567);
}

#[test]
fn timestamp_rejects_out_of_range_values_and_handles_negative_system_times() {
    assert!(Timestamp::from_seconds(i64::MAX).is_err());
    assert!(Timestamp::from_nanoseconds(i128::MAX).is_err());
    assert!(Timestamp::from_nanoseconds(i128::MIN).is_err());

    let before_epoch = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(60);
    let ts = Timestamp::from(before_epoch);
    assert_eq!(ts.as_seconds(), -60);
    assert!(ts.as_nanoseconds() < 0);

    let zero = Timestamp::UNIX_EPOCH;
    assert_eq!(zero.to_offset(UtcOffset::UTC).unix_timestamp(), 0);
}

#[test]
fn timestamp_from_seconds_various_values() {
    // Test zero
    let ts0 = Timestamp::from_seconds(0).unwrap();
    assert_eq!(ts0.as_seconds(), 0);
    assert_eq!(ts0.as_nanoseconds(), 0);

    // Test positive value
    let ts1 = Timestamp::from_seconds(1_000_000).unwrap();
    assert_eq!(ts1.as_seconds(), 1_000_000);

    // Test negative value
    let ts2 = Timestamp::from_seconds(-1_000_000).unwrap();
    assert_eq!(ts2.as_seconds(), -1_000_000);

    // Test boundary values
    assert!(Timestamp::from_seconds(i64::MAX).is_err());
    assert!(Timestamp::from_seconds(i64::MIN).is_err());
}

#[test]
fn timestamp_from_milliseconds_various_values() {
    // Test zero
    let ts0 = Timestamp::from_milliseconds(0).unwrap();
    assert_eq!(ts0.as_milliseconds(), 0);

    // Test positive value
    let ts1 = Timestamp::from_milliseconds(1_234_567).unwrap();
    assert_eq!(ts1.as_milliseconds(), 1_234_567);

    // Test negative value
    let ts2 = Timestamp::from_milliseconds(-1_234_567).unwrap();
    assert_eq!(ts2.as_milliseconds(), -1_234_567);

    // Test conversion to seconds
    let ts3 = Timestamp::from_milliseconds(5000).unwrap();
    assert_eq!(ts3.as_seconds(), 5);
}

#[test]
fn timestamp_from_microseconds_various_values() {
    // Test zero
    let ts0 = Timestamp::from_microseconds(0).unwrap();
    assert_eq!(ts0.as_microseconds(), 0);

    // Test positive value
    let ts1 = Timestamp::from_microseconds(1_234_567_890).unwrap();
    assert_eq!(ts1.as_microseconds(), 1_234_567_890);

    // Test negative value
    let ts2 = Timestamp::from_microseconds(-1_234_567_890).unwrap();
    assert_eq!(ts2.as_microseconds(), -1_234_567_890);

    // Test conversion to milliseconds
    let ts3 = Timestamp::from_microseconds(5_000_000).unwrap();
    assert_eq!(ts3.as_milliseconds(), 5_000);
}

#[test]
fn timestamp_from_nanoseconds_various_values() {
    // Test zero
    let ts0 = Timestamp::from_nanoseconds(0).unwrap();
    assert_eq!(ts0.as_nanoseconds(), 0);

    // Test positive value
    let ts1 = Timestamp::from_nanoseconds(1_234_567_890_123).unwrap();
    assert_eq!(ts1.as_nanoseconds(), 1_234_567_890_123);

    // Test negative value
    let ts2 = Timestamp::from_nanoseconds(-1_234_567_890_123).unwrap();
    assert_eq!(ts2.as_nanoseconds(), -1_234_567_890_123);

    // Test boundary values
    assert!(Timestamp::from_nanoseconds(i128::MAX).is_err());
    assert!(Timestamp::from_nanoseconds(i128::MIN).is_err());
}

#[test]
fn timestamp_from_system_time_various_times() {
    // Test epoch
    let epoch = SystemTime::UNIX_EPOCH;
    let ts_epoch = Timestamp::from(epoch);
    assert_eq!(ts_epoch.as_seconds(), 0);

    // Test positive time
    let future = SystemTime::UNIX_EPOCH + StdDuration::from_secs(1_234_567);
    let ts_future = Timestamp::from(future);
    assert_eq!(ts_future.as_seconds(), 1_234_567);

    // Test negative time (before epoch)
    let past = SystemTime::UNIX_EPOCH - StdDuration::from_secs(60);
    let ts_past = Timestamp::from(past);
    assert_eq!(ts_past.as_seconds(), -60);

    // Test with subseconds
    let with_nanos = SystemTime::UNIX_EPOCH
        + StdDuration::from_secs(1)
        + StdDuration::from_nanos(500_000_000);
    let ts_nanos = Timestamp::from(with_nanos);
    assert_eq!(ts_nanos.as_seconds(), 1);
    assert_eq!(ts_nanos.as_nanoseconds(), 1_500_000_000);
}

#[test]
fn timestamp_to_utc_conversion() {
    // Test epoch
    let ts_epoch = Timestamp::UNIX_EPOCH;
    let utc_epoch = ts_epoch.to_utc().unwrap();
    assert_eq!(utc_epoch.unix_timestamp(), 0);

    // Test positive timestamp
    let ts_pos = Timestamp::from_seconds(1_000_000).unwrap();
    let utc_pos = ts_pos.to_utc().unwrap();
    assert_eq!(utc_pos.unix_timestamp(), 1_000_000);

    // Test negative timestamp
    let ts_neg = Timestamp::from_seconds(-1_000_000).unwrap();
    let utc_neg = ts_neg.to_utc().unwrap();
    assert_eq!(utc_neg.unix_timestamp(), -1_000_000);

    // Test with subseconds
    let ts_sub = Timestamp::from_nanoseconds(1_500_000_000).unwrap();
    let utc_sub = ts_sub.to_utc().unwrap();
    assert_eq!(utc_sub.unix_timestamp(), 1);
    assert_eq!(utc_sub.nanosecond(), 500_000_000);
}

#[test]
fn timestamp_to_offset_conversion() {
    let ts = Timestamp::from_seconds(1_000_000).unwrap();

    // Test conversion to UTC offset
    let utc_offset = UtcOffset::UTC;
    let offset_dt = ts.to_offset(utc_offset);
    assert_eq!(offset_dt.unix_timestamp(), 1_000_000);

    // Test conversion to positive offset
    let offset_pos = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset_dt_pos = ts.to_offset(offset_pos);
    assert_eq!(offset_dt_pos.offset(), offset_pos);

    // Test conversion to negative offset
    let offset_neg = UtcOffset::from_hms(-8, 0, 0).unwrap();
    let offset_dt_neg = ts.to_offset(offset_neg);
    assert_eq!(offset_dt_neg.offset(), offset_neg);
}

#[test]
fn utc_offset_from_hms_various_combinations() {
    // Test zero offset
    let utc = UtcOffset::from_hms(0, 0, 0).unwrap();
    assert_eq!(utc.as_hms(), (0, 0, 0));
    assert!(utc.is_utc());

    // Test positive offset
    let pos = UtcOffset::from_hms(5, 30, 0).unwrap();
    assert_eq!(pos.as_hms(), (5, 30, 0));
    assert!(pos.is_positive());

    // Test negative offset
    let neg = UtcOffset::from_hms(-8, 0, 0).unwrap();
    assert_eq!(neg.as_hms(), (-8, 0, 0));
    assert!(neg.is_negative());

    // Test with seconds
    let with_sec = UtcOffset::from_hms(5, 30, 45).unwrap();
    assert_eq!(with_sec.as_hms(), (5, 30, 45));
}

#[test]
fn utc_offset_from_whole_seconds_various_values() {
    // Test zero
    let utc = UtcOffset::from_whole_seconds(0).unwrap();
    assert_eq!(utc.as_hms(), (0, 0, 0));

    // Test positive hours
    let pos_hours = UtcOffset::from_whole_seconds(3600).unwrap();
    assert_eq!(pos_hours.as_hms(), (1, 0, 0));

    // Test positive hours and minutes
    let pos_hm = UtcOffset::from_whole_seconds(5400).unwrap();
    assert_eq!(pos_hm.as_hms(), (1, 30, 0));

    // Test negative values
    let neg = UtcOffset::from_whole_seconds(-3600).unwrap();
    assert_eq!(neg.as_hms(), (-1, 0, 0));

    // Test with seconds
    let with_sec = UtcOffset::from_whole_seconds(3661).unwrap();
    assert_eq!(with_sec.as_hms(), (1, 1, 1));
}

#[test]
fn utc_offset_whole_components_accessors() {
    let offset = UtcOffset::from_hms(5, 30, 45).unwrap();

    // Test whole_hours
    assert_eq!(offset.whole_hours(), 5);

    // Test whole_minutes
    assert_eq!(offset.whole_minutes(), 5 * 60 + 30);

    // Test whole_seconds
    assert_eq!(offset.whole_seconds(), 5 * 3600 + 30 * 60 + 45);

    // Test minutes_past_hour
    assert_eq!(offset.minutes_past_hour(), 30);

    // Test seconds_past_minute
    assert_eq!(offset.seconds_past_minute(), 45);
}

#[test]
fn utc_offset_comparison_operations() {
    let offset1 = UtcOffset::from_hms(5, 0, 0).unwrap();
    let offset2 = UtcOffset::from_hms(6, 0, 0).unwrap();
    let offset3 = UtcOffset::from_hms(5, 0, 0).unwrap();
    let offset4 = UtcOffset::from_hms(-5, 0, 0).unwrap();

    // Test comparison
    assert!(offset1 < offset2);
    assert!(offset2 > offset1);
    assert_eq!(offset1, offset3);
    assert!(offset1 > offset4);
    assert!(offset4 < offset1);

    // Test UTC comparison
    let utc = UtcOffset::UTC;
    assert!(utc < offset1);
    assert!(utc > offset4);
}

#[test]
fn utc_offset_negation_operation() {
    // Test positive to negative
    let pos = UtcOffset::from_hms(5, 30, 0).unwrap();
    let negated = -pos;
    assert_eq!(negated.as_hms(), (-5, -30, 0));

    // Test negative to positive
    let neg = UtcOffset::from_hms(-8, 0, 0).unwrap();
    let negated2 = -neg;
    assert_eq!(negated2.as_hms(), (8, 0, 0));

    // Test UTC remains UTC
    let utc = UtcOffset::UTC;
    assert_eq!(-utc, utc);
}

#[test]
fn utc_datetime_unix_timestamp_operations() {
    let date = Date::from_calendar_date(2024, Month::January, 1).unwrap();
    let time = Time::from_hms(0, 0, 0).unwrap();
    let utc = UtcDateTime::new(date, time);

    // Test unix_timestamp
    let timestamp = utc.unix_timestamp();
    assert!(timestamp > 0);

    // Test unix_timestamp_nanos
    let nanos = utc.unix_timestamp_nanos();
    assert!(nanos > 0);

    // Test with specific known timestamp
    let epoch = UtcDateTime::UNIX_EPOCH;
    assert_eq!(epoch.unix_timestamp(), 0);
    assert_eq!(epoch.unix_timestamp_nanos(), 0);
}

#[test]
fn offset_datetime_unix_timestamp_operations() {
    let date = Date::from_calendar_date(2024, Month::January, 1).unwrap();
    let time = Time::from_hms(0, 0, 0).unwrap();
    let offset = UtcOffset::from_hms(5, 30, 0).unwrap();
    let offset_dt = OffsetDateTime::new_in_offset(date, time, offset);

    // Test unix_timestamp
    let timestamp = offset_dt.unix_timestamp();
    assert!(timestamp > 0);

    // Test unix_timestamp_nanos
    let nanos = offset_dt.unix_timestamp_nanos();
    assert!(nanos > 0);

    // Test that offset affects the timestamp
    let utc = UtcDateTime::new(date, time);
    assert_ne!(offset_dt.unix_timestamp(), utc.unix_timestamp());
}
