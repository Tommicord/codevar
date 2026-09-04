use std::time::Duration as StdDuration;

use codevar_core::timeutil::date::Date;
use codevar_core::timeutil::date_month::Month;
use codevar_core::timeutil::date_offset_time::OffsetDateTime;
use codevar_core::timeutil::date_plain::PlainDateTime;
use codevar_core::timeutil::date_signed_duration::SignedDuration;
use codevar_core::timeutil::date_time::Time;
use codevar_core::timeutil::date_utc_offset::UtcOffset;
use codevar_core::timeutil::date_utc_time::UtcDateTime;

#[test]
fn plain_datetime_arithmetic_and_roundtrip() {
    let date = Date::from_calendar_date(2024, Month::February, 29).unwrap();
    let time = Time::from_hms(12, 0, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    let plus_day = dt.checked_add(SignedDuration::days(1)).unwrap();
    assert_eq!(
        plus_day.date(),
        Date::from_calendar_date(2024, Month::March, 1).unwrap()
    );
    assert_eq!(plus_day.time(), time);

    let minus_day = dt.checked_sub(SignedDuration::days(1)).unwrap();
    assert_eq!(
        minus_day.date(),
        Date::from_calendar_date(2024, Month::February, 28).unwrap()
    );

    let with_std = dt + StdDuration::from_secs(3600);
    assert_eq!(with_std.time(), Time::from_hms(13, 0, 0).unwrap());

    assert_eq!(
        dt.saturating_add(SignedDuration::days(10_000_000)),
        PlainDateTime::MAX
    );
    assert_eq!(
        dt.saturating_sub(SignedDuration::days(10_000_000)),
        PlainDateTime::MIN
    );
}

#[test]
fn utc_and_offset_datetime_conversion() {
    let utc = UtcDateTime::new(
        Date::from_calendar_date(2024, Month::February, 29).unwrap(),
        Time::from_hms(12, 30, 0).unwrap(),
    );
    let offset = UtcOffset::from_hms(2, 0, 0).unwrap();
    let converted = utc.to_offset(offset);
    assert_eq!(converted.offset(), offset);
    assert_eq!(converted.to_utc(), utc);

    let offset_dt = OffsetDateTime::new_in_offset(
        Date::from_calendar_date(2024, Month::February, 29).unwrap(),
        Time::from_hms(10, 30, 0).unwrap(),
        offset,
    );
    assert_eq!(
        offset_dt.date(),
        Date::from_calendar_date(2024, Month::February, 29).unwrap()
    );
    assert_eq!(offset_dt.time(), Time::from_hms(10, 30, 0).unwrap());
    assert_eq!(offset_dt.unix_timestamp(), 1709190600);
}

#[test]
fn offset_datetime_checked_and_saturating_math() {
    let offset = UtcOffset::UTC;
    let base = UtcDateTime::new(
        Date::from_calendar_date(2024, Month::March, 1).unwrap(),
        Time::MIDNIGHT,
    )
    .to_offset(offset);

    let plus = base.checked_add(SignedDuration::days(1)).unwrap();
    assert_eq!(
        plus.date(),
        Date::from_calendar_date(2024, Month::March, 2).unwrap()
    );

    let minus = base.checked_sub(SignedDuration::days(1)).unwrap();
    assert_eq!(
        minus.date(),
        Date::from_calendar_date(2024, Month::February, 29).unwrap()
    );

    assert_eq!(
        base.saturating_add(SignedDuration::days(1_000_000)).date(),
        Date::MAX
    );
    assert_eq!(
        base.saturating_sub(SignedDuration::days(1_000_000)).date(),
        Date::MIN
    );
}

#[test]
fn offset_datetime_crosses_date_boundaries_and_rejects_invalid_offsets() {
    let offset = UtcOffset::from_hms(2, 0, 0).unwrap();
    let dt = UtcDateTime::new(
        Date::from_calendar_date(2024, Month::January, 1).unwrap(),
        Time::MIDNIGHT,
    )
    .to_offset(offset);

    let converted = dt.to_offset(UtcOffset::from_hms(-2, 0, 0).unwrap());
    assert_eq!(
        converted.date(),
        Date::from_calendar_date(2023, Month::December, 31).unwrap()
    );

    let before_epoch = UtcDateTime::new(
        Date::from_calendar_date(1969, Month::December, 31).unwrap(),
        Time::MIDNIGHT,
    );
    assert!(before_epoch.checked_to_offset(UtcOffset::UTC).is_some());

    assert!(UtcOffset::from_hms(25, 0, 0).is_err());
    assert!(UtcOffset::from_hms(0, 60, 0).is_err());
}

#[test]
fn plain_datetime_checked_add_various_durations() {
    let date = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let time = Time::from_hms(12, 30, 45).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test adding days
    let plus_day = dt.checked_add(SignedDuration::days(1)).unwrap();
    assert_eq!(
        plus_day.date(),
        Date::from_calendar_date(2024, Month::January, 16).unwrap()
    );
    assert_eq!(plus_day.time(), time);

    // Test adding hours
    let plus_hour = dt.checked_add(SignedDuration::hours(1)).unwrap();
    assert_eq!(plus_hour.date(), date);
    assert_eq!(plus_hour.time(), Time::from_hms(13, 30, 45).unwrap());

    // Test adding minutes
    let plus_minute = dt.checked_add(SignedDuration::minutes(30)).unwrap();
    assert_eq!(plus_minute.date(), date);
    assert_eq!(plus_minute.time(), Time::from_hms(13, 0, 45).unwrap());

    // Test adding seconds
    let plus_second = dt.checked_add(SignedDuration::seconds(15)).unwrap();
    assert_eq!(plus_second.date(), date);
    assert_eq!(plus_second.time(), Time::from_hms(12, 31, 0).unwrap());
}

#[test]
fn plain_datetime_checked_add_crosses_day_boundary() {
    let date = Date::from_calendar_date(2024, Month::January, 31).unwrap();
    let time = Time::from_hms(23, 30, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test crossing day boundary with hours
    let result = dt.checked_add(SignedDuration::hours(2)).unwrap();
    assert_eq!(
        result.date(),
        Date::from_calendar_date(2024, Month::February, 1).unwrap()
    );
    assert_eq!(result.time(), Time::from_hms(1, 30, 0).unwrap());

    // Test crossing day boundary with seconds
    let result2 = dt.checked_add(SignedDuration::seconds(3600)).unwrap();
    assert_eq!(
        result2.date(),
        Date::from_calendar_date(2024, Month::February, 1).unwrap()
    );
    assert_eq!(result2.time(), Time::from_hms(0, 30, 0).unwrap());
}

#[test]
fn plain_datetime_checked_add_overflow_handling() {
    let max_dt = PlainDateTime::MAX;
    let min_dt = PlainDateTime::MIN;

    // Test overflow on MAX
    assert!(max_dt.checked_add(SignedDuration::seconds(1)).is_none());
    assert!(max_dt.checked_add(SignedDuration::days(1)).is_none());
    assert!(max_dt.checked_add(SignedDuration::nanoseconds(1)).is_none());

    // Test underflow on MIN
    assert!(min_dt.checked_sub(SignedDuration::seconds(1)).is_none());
    assert!(min_dt.checked_sub(SignedDuration::days(1)).is_none());
    assert!(min_dt.checked_sub(SignedDuration::nanoseconds(1)).is_none());
}

#[test]
fn plain_datetime_checked_sub_various_durations() {
    let date = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let time = Time::from_hms(12, 30, 45).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test subtracting days
    let minus_day = dt.checked_sub(SignedDuration::days(1)).unwrap();
    assert_eq!(
        minus_day.date(),
        Date::from_calendar_date(2024, Month::January, 14).unwrap()
    );
    assert_eq!(minus_day.time(), time);

    // Test subtracting hours
    let minus_hour = dt.checked_sub(SignedDuration::hours(1)).unwrap();
    assert_eq!(minus_hour.date(), date);
    assert_eq!(minus_hour.time(), Time::from_hms(11, 30, 45).unwrap());

    // Test subtracting minutes
    let minus_minute = dt.checked_sub(SignedDuration::minutes(30)).unwrap();
    assert_eq!(minus_minute.date(), date);
    assert_eq!(minus_minute.time(), Time::from_hms(12, 0, 45).unwrap());
}

#[test]
fn plain_datetime_checked_sub_crosses_day_boundary() {
    let date = Date::from_calendar_date(2024, Month::February, 1).unwrap();
    let time = Time::from_hms(0, 30, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test crossing day boundary backwards with hours
    let result = dt.checked_sub(SignedDuration::hours(2)).unwrap();
    assert_eq!(
        result.date(),
        Date::from_calendar_date(2024, Month::January, 31).unwrap()
    );
    assert_eq!(result.time(), Time::from_hms(22, 30, 0).unwrap());

    // Test crossing day boundary backwards with seconds
    let result2 = dt.checked_sub(SignedDuration::seconds(3600)).unwrap();
    assert_eq!(
        result2.date(),
        Date::from_calendar_date(2024, Month::January, 31).unwrap()
    );
    assert_eq!(result2.time(), Time::from_hms(23, 30, 0).unwrap());
}

#[test]
fn plain_datetime_saturating_add_boundary_values() {
    let date = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let time = Time::from_hms(12, 0, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test saturating to MAX
    let saturated_max = dt.saturating_add(SignedDuration::days(10_000_000));
    assert_eq!(saturated_max, PlainDateTime::MAX);

    // Test saturating to MIN
    let saturated_min = dt.saturating_sub(SignedDuration::days(10_000_000));
    assert_eq!(saturated_min, PlainDateTime::MIN);

    // Test normal saturating operations
    let normal_add = dt.saturating_add(SignedDuration::days(1));
    assert_eq!(
        normal_add.date(),
        Date::from_calendar_date(2024, Month::January, 16).unwrap()
    );

    let normal_sub = dt.saturating_sub(SignedDuration::days(1));
    assert_eq!(
        normal_sub.date(),
        Date::from_calendar_date(2024, Month::January, 14).unwrap()
    );
}

#[test]
fn plain_datetime_std_duration_addition() {
    let date = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let time = Time::from_hms(12, 0, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test adding seconds
    let result = dt + StdDuration::from_secs(3600);
    assert_eq!(result.date(), date);
    assert_eq!(result.time(), Time::from_hms(13, 0, 0).unwrap());

    // Test adding with subseconds
    let result2 = dt + StdDuration::from_millis(1500);
    assert_eq!(result2.date(), date);
    assert_eq!(result2.time(), Time::from_hms(12, 0, 1).unwrap());
    assert_eq!(result2.millisecond(), 500);
}

#[test]
fn plain_datetime_std_duration_subtraction() {
    let date = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let time = Time::from_hms(12, 0, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test subtracting seconds
    let result = dt - StdDuration::from_secs(3600);
    assert_eq!(result.date(), date);
    assert_eq!(result.time(), Time::from_hms(11, 0, 0).unwrap());

    // Test subtracting with subseconds
    let result2 = dt - StdDuration::from_millis(1500);
    assert_eq!(result2.date(), date);
    assert_eq!(result2.time(), Time::from_hms(11, 59, 58).unwrap());
    assert_eq!(result2.millisecond(), 500);
}

#[test]
fn plain_datetime_std_duration_crosses_boundary() {
    let date = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let time = Time::from_hms(23, 30, 0).unwrap();
    let dt = PlainDateTime::new(date, time);

    // Test adding across day boundary
    let result = dt + StdDuration::from_secs(3600);
    assert_eq!(
        result.date(),
        Date::from_calendar_date(2024, Month::January, 16).unwrap()
    );
    assert_eq!(result.time(), Time::from_hms(0, 30, 0).unwrap());

    // Test subtracting across day boundary
    let result2 = dt - StdDuration::from_secs(3600);
    assert_eq!(
        result2.date(),
        Date::from_calendar_date(2024, Month::January, 15).unwrap()
    );
    assert_eq!(result2.time(), Time::from_hms(22, 30, 0).unwrap());
}

#[test]
fn utc_datetime_checked_add_operations() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();
    let time = Time::from_hms(10, 15, 30).unwrap();
    let utc = UtcDateTime::new(date, time);

    // Test adding days
    let plus_day = utc.checked_add(SignedDuration::days(1)).unwrap();
    assert_eq!(
        plus_day.date(),
        Date::from_calendar_date(2024, Month::March, 16).unwrap()
    );
    assert_eq!(plus_day.time(), time);

    // Test adding hours
    let plus_hour = utc.checked_add(SignedDuration::hours(5)).unwrap();
    assert_eq!(plus_hour.date(), date);
    assert_eq!(plus_hour.time(), Time::from_hms(15, 15, 30).unwrap());

    // Test adding with subseconds
    let plus_nano = utc
        .checked_add(SignedDuration::nanoseconds(500_000_000))
        .unwrap();
    assert_eq!(plus_nano.date(), date);
    assert_eq!(plus_nano.time(), Time::from_hms(10, 15, 30).unwrap());
    assert_eq!(plus_nano.nanosecond(), 500_000_000);
}

#[test]
fn utc_datetime_checked_sub_operations() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();
    let time = Time::from_hms(10, 15, 30).unwrap();
    let utc = UtcDateTime::new(date, time);

    // Test subtracting days
    let minus_day = utc.checked_sub(SignedDuration::days(1)).unwrap();
    assert_eq!(
        minus_day.date(),
        Date::from_calendar_date(2024, Month::March, 14).unwrap()
    );
    assert_eq!(minus_day.time(), time);

    // Test subtracting hours
    let minus_hour = utc.checked_sub(SignedDuration::hours(5)).unwrap();
    assert_eq!(minus_hour.date(), date);
    assert_eq!(minus_hour.time(), Time::from_hms(5, 15, 30).unwrap());

    // Test subtracting with subseconds
    let minus_nano = utc
        .checked_sub(SignedDuration::nanoseconds(500_000_000))
        .unwrap();
    assert_eq!(minus_nano.date(), date);
    assert_eq!(minus_nano.time(), Time::from_hms(10, 15, 29).unwrap());
    assert_eq!(minus_nano.nanosecond(), 500_000_000);
}

#[test]
fn utc_datetime_saturating_operations() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();
    let time = Time::from_hms(10, 15, 30).unwrap();
    let utc = UtcDateTime::new(date, time);

    // Test saturating to boundaries
    let saturated_max = utc.saturating_add(SignedDuration::days(1_000_000));
    assert_eq!(saturated_max.date(), Date::MAX);

    let saturated_min = utc.saturating_sub(SignedDuration::days(1_000_000));
    assert_eq!(saturated_min.date(), Date::MIN);

    // Test normal saturating operations
    let normal_add = utc.saturating_add(SignedDuration::hours(2));
    assert_eq!(normal_add.time(), Time::from_hms(12, 15, 30).unwrap());

    let normal_sub = utc.saturating_sub(SignedDuration::hours(2));
    assert_eq!(normal_sub.time(), Time::from_hms(8, 15, 30).unwrap());
}

#[test]
fn utc_datetime_to_offset_conversion() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();
    let time = Time::from_hms(10, 0, 0).unwrap();
    let utc = UtcDateTime::new(date, time);

    // Test conversion to positive offset
    let offset_pos = UtcOffset::from_hms(5, 30, 0).unwrap();
    let converted_pos = utc.to_offset(offset_pos);
    assert_eq!(converted_pos.offset(), offset_pos);
    assert_eq!(converted_pos.time(), Time::from_hms(15, 30, 0).unwrap());

    // Test conversion to negative offset
    let offset_neg = UtcOffset::from_hms(-8, 0, 0).unwrap();
    let converted_neg = utc.to_offset(offset_neg);
    assert_eq!(converted_neg.offset(), offset_neg);
    assert_eq!(converted_neg.time(), Time::from_hms(2, 0, 0).unwrap());

    // Test conversion to UTC
    let converted_utc = utc.to_offset(UtcOffset::UTC);
    assert_eq!(converted_utc.to_utc(), utc);
}

#[test]
fn utc_datetime_to_offset_crosses_day_boundary() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();
    let time = Time::from_hms(22, 0, 0).unwrap();
    let utc = UtcDateTime::new(date, time);

    // Test crossing to next day with positive offset
    let offset_pos = UtcOffset::from_hms(5, 0, 0).unwrap();
    let converted_pos = utc.to_offset(offset_pos);
    assert_eq!(
        converted_pos.date(),
        Date::from_calendar_date(2024, Month::March, 16).unwrap()
    );
    assert_eq!(converted_pos.time(), Time::from_hms(3, 0, 0).unwrap());

    // Test crossing to previous day with negative offset
    let time2 = Time::from_hms(2, 0, 0).unwrap();
    let utc2 = UtcDateTime::new(date, time2);
    let offset_neg = UtcOffset::from_hms(-5, 0, 0).unwrap();
    let converted_neg = utc2.to_offset(offset_neg);
    assert_eq!(
        converted_neg.date(),
        Date::from_calendar_date(2024, Month::March, 14).unwrap()
    );
    assert_eq!(converted_neg.time(), Time::from_hms(21, 0, 0).unwrap());
}

#[test]
fn offset_datetime_checked_add_with_offset() {
    let date = Date::from_calendar_date(2024, Month::April, 20).unwrap();
    let time = Time::from_hms(14, 45, 0).unwrap();
    let offset = UtcOffset::from_hms(3, 0, 0).unwrap();
    let offset_dt = OffsetDateTime::new_in_offset(date, time, offset);

    // Test adding days
    let plus_day = offset_dt.checked_add(SignedDuration::days(1)).unwrap();
    assert_eq!(
        plus_day.date(),
        Date::from_calendar_date(2024, Month::April, 21).unwrap()
    );
    assert_eq!(plus_day.time(), time);
    assert_eq!(plus_day.offset(), offset);

    // Test adding hours
    let plus_hour = offset_dt.checked_add(SignedDuration::hours(2)).unwrap();
    assert_eq!(plus_hour.date(), date);
    assert_eq!(plus_hour.time(), Time::from_hms(16, 45, 0).unwrap());
    assert_eq!(plus_hour.offset(), offset);
}

#[test]
fn offset_datetime_checked_sub_with_offset() {
    let date = Date::from_calendar_date(2024, Month::April, 20).unwrap();
    let time = Time::from_hms(14, 45, 0).unwrap();
    let offset = UtcOffset::from_hms(-5, 0, 0).unwrap();
    let offset_dt = OffsetDateTime::new_in_offset(date, time, offset);

    // Test subtracting days
    let minus_day = offset_dt.checked_sub(SignedDuration::days(1)).unwrap();
    assert_eq!(
        minus_day.date(),
        Date::from_calendar_date(2024, Month::April, 19).unwrap()
    );
    assert_eq!(minus_day.time(), time);
    assert_eq!(minus_day.offset(), offset);

    // Test subtracting hours
    let minus_hour = offset_dt.checked_sub(SignedDuration::hours(2)).unwrap();
    assert_eq!(minus_hour.date(), date);
    assert_eq!(minus_hour.time(), Time::from_hms(12, 45, 0).unwrap());
    assert_eq!(minus_hour.offset(), offset);
}

#[test]
fn offset_datetime_saturating_operations_with_offset() {
    let date = Date::from_calendar_date(2024, Month::April, 20).unwrap();
    let time = Time::from_hms(14, 45, 0).unwrap();
    let offset = UtcOffset::from_hms(2, 30, 0).unwrap();
    let offset_dt = OffsetDateTime::new_in_offset(date, time, offset);

    // Test saturating to boundaries
    let saturated_max = offset_dt.saturating_add(SignedDuration::days(1_000_000));
    assert_eq!(saturated_max.date(), Date::MAX);

    let saturated_min = offset_dt.saturating_sub(SignedDuration::days(1_000_000));
    assert_eq!(saturated_min.date(), Date::MIN);

    // Test normal saturating operations
    let normal_add = offset_dt.saturating_add(SignedDuration::hours(3));
    assert_eq!(normal_add.time(), Time::from_hms(17, 45, 0).unwrap());
    assert_eq!(normal_add.offset(), offset);
}

#[test]
fn offset_datetime_to_utc_conversion() {
    let date = Date::from_calendar_date(2024, Month::April, 20).unwrap();
    let time = Time::from_hms(14, 45, 0).unwrap();
    let offset = UtcOffset::from_hms(3, 0, 0).unwrap();
    let offset_dt = OffsetDateTime::new_in_offset(date, time, offset);

    // Test conversion to UTC
    let utc = offset_dt.to_utc();
    assert_eq!(utc.date(), date);
    assert_eq!(utc.time(), Time::from_hms(11, 45, 0).unwrap());
}

#[test]
fn offset_datetime_to_offset_conversion() {
    let date = Date::from_calendar_date(2024, Month::April, 20).unwrap();
    let time = Time::from_hms(14, 45, 0).unwrap();
    let offset1 = UtcOffset::from_hms(3, 0, 0).unwrap();
    let offset_dt = OffsetDateTime::new_in_offset(date, time, offset1);

    // Test conversion to different offset
    let offset2 = UtcOffset::from_hms(-5, 0, 0).unwrap();
    let converted = offset_dt.to_offset(offset2);
    assert_eq!(converted.offset(), offset2);
    assert_eq!(converted.time(), Time::from_hms(6, 45, 0).unwrap());
}
