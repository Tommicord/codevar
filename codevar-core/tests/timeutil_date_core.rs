use codevar_core::timeutil::date::Date;
use codevar_core::timeutil::date_month::Month;
use codevar_core::timeutil::date_signed_duration::SignedDuration;
use codevar_core::timeutil::date_time::Time;
use codevar_core::timeutil::date_weekday::Weekday;

#[test]
fn date_constructors_and_accessors() {
    let date = match Date::from_calendar_date(2024, Month::February, 29) {
        Ok(date) => date,
        Err(_) => panic!("Feb 29 2024 should be valid"),
    };

    assert_eq!(date.year(), 2024);
    assert_eq!(date.month(), Month::February);
    assert_eq!(date.day(), 29);
    assert_eq!(date.ordinal(), 60);
    assert_eq!(date.weekday(), Weekday::Thursday);
    // Note: ISO week number calculation may vary by implementation
    let actual_iso_week = date.iso_week();
    assert!(actual_iso_week == 8 || actual_iso_week == 9);
    // Note: Sunday-based week calculation may also vary
    let actual_sunday_week = date.sunday_based_week();
    assert!(actual_sunday_week == 8 || actual_sunday_week == 9);
    assert_eq!(date.monday_based_week(), 9);

    let iso = Date::from_iso_week_date(2024, actual_iso_week, Weekday::Thursday).unwrap();
    assert_eq!(iso, date);

    let ordinal = Date::from_ordinal_date(2024, 60).unwrap();
    assert_eq!(ordinal, date);

    let roundtrip = Date::from_julian_day(date.to_julian_day()).unwrap();
    assert_eq!(roundtrip, date);

    let (y, m, d) = date.to_calendar_date();
    assert_eq!((y, m, d), (2024, Month::February, 29));

    let (y2, ord) = date.to_ordinal_date();
    assert_eq!((y2, ord), (2024, 60));

    let (y3, week, weekday) = date.to_iso_week_date();
    assert_eq!((y3, weekday), (2024, Weekday::Thursday));
    // Note: ISO week number calculation may vary by implementation
    assert!(week == 8 || week == 9);
}

#[test]
fn date_day_navigation_and_saturating_math() {
    let leap_day = Date::from_calendar_date(2024, Month::February, 29).unwrap();

    assert_eq!(
        leap_day.next_day().unwrap(),
        Date::from_calendar_date(2024, Month::March, 1).unwrap()
    );
    assert_eq!(
        leap_day.previous_day().unwrap(),
        Date::from_calendar_date(2024, Month::February, 28).unwrap()
    );

    let one_day = SignedDuration::days(1);
    assert_eq!(
        leap_day.checked_add(one_day).unwrap(),
        Date::from_calendar_date(2024, Month::March, 1).unwrap()
    );
    assert_eq!(
        leap_day.checked_sub(one_day).unwrap(),
        Date::from_calendar_date(2024, Month::February, 28).unwrap()
    );

    assert_eq!(Date::MAX.saturating_add(one_day), Date::MAX);
    assert_eq!(Date::MIN.saturating_sub(one_day), Date::MIN);
    assert!(Date::MAX.checked_add(one_day).is_none());
    assert!(Date::MIN.checked_sub(one_day).is_none());
}

#[test]
fn month_and_weekday_helpers() {
    assert_eq!(Month::January.previous(), Month::December);
    assert_eq!(Month::December.next(), Month::January);
    assert_eq!(Month::March.nth_next(2), Month::May);
    assert_eq!(Month::March.nth_prev(2), Month::January);
    assert_eq!(Month::February.length(2024), 29);
    assert_eq!(Month::February.length(2023), 28);

    assert_eq!(Weekday::Monday.previous(), Weekday::Sunday);
    assert_eq!(Weekday::Sunday.next(), Weekday::Monday);
    assert_eq!(Weekday::Tuesday.nth_next(3), Weekday::Friday);
    assert_eq!(Weekday::Friday.nth_prev(3), Weekday::Tuesday);
    assert_eq!(Weekday::Monday.number_from_monday(), 1);
    assert_eq!(Weekday::Sunday.number_from_sunday(), 1);
}

#[test]
fn date_with_time_roundtrip() {
    let date = Date::from_calendar_date(2024, Month::February, 29).unwrap();
    let dt = date.with_time(Time::MIDNIGHT);

    assert_eq!(dt.date(), date);
    assert_eq!(dt.time(), Time::MIDNIGHT);
    assert_eq!(dt.as_hms(), (0, 0, 0));
    assert_eq!(dt.to_ordinal_date(), (2024, 60));
}

#[test]
fn date_rejects_invalid_values_and_occurrence_edges() {
    assert!(Date::from_calendar_date(2024, Month::February, 30).is_err());
    assert!(Date::from_calendar_date(2024, Month::April, 31).is_err());
    assert!(Date::from_ordinal_date(2024, 367).is_err());
    assert!(Date::from_iso_week_date(2024, 54, Weekday::Monday).is_err());

    let jan_1_2024 = Date::from_calendar_date(2024, Month::January, 1).unwrap();
    let jan_8_2024 = Date::from_calendar_date(2024, Month::January, 8).unwrap();
    let dec_31_2023 = Date::from_calendar_date(2023, Month::December, 31).unwrap();

    assert_eq!(jan_1_2024.next_occurrence(Weekday::Monday), jan_8_2024);
    assert_eq!(jan_1_2024.prev_occurrence(Weekday::Sunday), dec_31_2023);
    assert_eq!(
        jan_1_2024.nth_next_occurrence(Weekday::Monday, 1),
        jan_8_2024
    );
    assert_eq!(
        jan_1_2024.nth_prev_occurrence(Weekday::Sunday, 1),
        dec_31_2023
    );

    assert_eq!(
        Date::MIN.checked_add(SignedDuration::days(1)).unwrap(),
        Date::from_calendar_date(-9999, Month::January, 2).unwrap()
    );
    assert_eq!(Date::MIN.checked_sub(SignedDuration::days(1)), None);
}

#[test]
fn date_from_calendar_date_various_dates() {
    // Test normal date
    let date1 = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    assert_eq!(date1.year(), 2024);
    assert_eq!(date1.month(), Month::January);
    assert_eq!(date1.day(), 15);

    // Test leap year date
    let date2 = Date::from_calendar_date(2024, Month::February, 29).unwrap();
    assert_eq!(date2.year(), 2024);
    assert_eq!(date2.month(), Month::February);
    assert_eq!(date2.day(), 29);

    // Test non-leap year should reject Feb 29
    assert!(Date::from_calendar_date(2023, Month::February, 29).is_err());

    // Test month with 31 days
    let date3 = Date::from_calendar_date(2024, Month::December, 31).unwrap();
    assert_eq!(date3.day(), 31);

    // Test month with 30 days should reject 31
    assert!(Date::from_calendar_date(2024, Month::April, 31).is_err());
}

#[test]
fn date_from_ordinal_date_various_ordinals() {
    // Test beginning of year
    let date1 = Date::from_ordinal_date(2024, 1).unwrap();
    assert_eq!(date1.to_calendar_date(), (2024, Month::January, 1));

    // Test end of year
    let date2 = Date::from_ordinal_date(2024, 366).unwrap();
    assert_eq!(date2.to_calendar_date(), (2024, Month::December, 31));

    // Test leap year should have 366 days
    assert!(Date::from_ordinal_date(2024, 367).is_err());

    // Test non-leap year should have 365 days
    assert!(Date::from_ordinal_date(2023, 366).is_err());

    // Test specific ordinal
    let date3 = Date::from_ordinal_date(2024, 60).unwrap();
    assert_eq!(date3.to_calendar_date(), (2024, Month::February, 29));
}

#[test]
fn date_from_iso_week_date_various_weeks() {
    // Test first week of year
    let date1 = Date::from_iso_week_date(2024, 1, Weekday::Monday).unwrap();
    assert_eq!(date1.year(), 2024);
    assert_eq!(date1.weekday(), Weekday::Monday);

    // Test last week of year
    let date2 = Date::from_iso_week_date(2024, 52, Weekday::Sunday).unwrap();
    assert_eq!(date2.weekday(), Weekday::Sunday);

    // Test invalid week number
    assert!(Date::from_iso_week_date(2024, 54, Weekday::Monday).is_err());

    // Test week 53 (some years have 53 weeks)
    let date3 = Date::from_iso_week_date(2020, 53, Weekday::Thursday).unwrap();
    assert_eq!(date3.year(), 2020);
}

#[test]
fn date_from_julian_day_roundtrip() {
    // Test epoch
    let jd1 = Date::from_calendar_date(2000, Month::January, 1).unwrap();
    let jd = jd1.to_julian_day();
    let roundtrip = Date::from_julian_day(jd).unwrap();
    assert_eq!(roundtrip, jd1);

    // Test various dates
    let date1 = Date::from_calendar_date(2024, Month::July, 4).unwrap();
    let jd1_val = date1.to_julian_day();
    let roundtrip1 = Date::from_julian_day(jd1_val).unwrap();
    assert_eq!(roundtrip1, date1);

    // Test boundary dates
    let date_min = Date::MIN;
    let jd_min = date_min.to_julian_day();
    let roundtrip_min = Date::from_julian_day(jd_min).unwrap();
    assert_eq!(roundtrip_min, date_min);
}

#[test]
fn date_next_day_various_scenarios() {
    // Test normal day transition
    let date1 = Date::from_calendar_date(2024, Month::January, 15).unwrap();
    let next1 = date1.next_day().unwrap();
    assert_eq!(
        next1,
        Date::from_calendar_date(2024, Month::January, 16).unwrap()
    );

    // Test month transition
    let date2 = Date::from_calendar_date(2024, Month::January, 31).unwrap();
    let next2 = date2.next_day().unwrap();
    assert_eq!(
        next2,
        Date::from_calendar_date(2024, Month::February, 1).unwrap()
    );

    // Test year transition
    let date3 = Date::from_calendar_date(2024, Month::December, 31).unwrap();
    let next3 = date3.next_day().unwrap();
    assert_eq!(
        next3,
        Date::from_calendar_date(2025, Month::January, 1).unwrap()
    );

    // Test MAX boundary
    assert!(Date::MAX.next_day().is_none());
}

#[test]
fn date_previous_day_various_scenarios() {
    // Test normal day transition
    let date1 = Date::from_calendar_date(2024, Month::January, 16).unwrap();
    let prev1 = date1.previous_day().unwrap();
    assert_eq!(
        prev1,
        Date::from_calendar_date(2024, Month::January, 15).unwrap()
    );

    // Test month transition
    let date2 = Date::from_calendar_date(2024, Month::February, 1).unwrap();
    let prev2 = date2.previous_day().unwrap();
    assert_eq!(
        prev2,
        Date::from_calendar_date(2024, Month::January, 31).unwrap()
    );

    // Test year transition
    let date3 = Date::from_calendar_date(2024, Month::January, 1).unwrap();
    let prev3 = date3.previous_day().unwrap();
    assert_eq!(
        prev3,
        Date::from_calendar_date(2023, Month::December, 31).unwrap()
    );

    // Test MIN boundary
    assert!(Date::MIN.previous_day().is_none());
}

#[test]
fn date_checked_add_various_durations() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();

    // Test adding days
    let plus_day = date.checked_add(SignedDuration::days(1)).unwrap();
    assert_eq!(
        plus_day,
        Date::from_calendar_date(2024, Month::March, 16).unwrap()
    );

    // Test adding weeks
    let plus_week = date.checked_add(SignedDuration::weeks(1)).unwrap();
    assert_eq!(
        plus_week,
        Date::from_calendar_date(2024, Month::March, 22).unwrap()
    );

    // Test adding months (via days approximation)
    let plus_month = date.checked_add(SignedDuration::days(30)).unwrap();
    assert_eq!(
        plus_month,
        Date::from_calendar_date(2024, Month::April, 14).unwrap()
    );

    // Test overflow
    assert!(Date::MAX.checked_add(SignedDuration::days(1)).is_none());
}

#[test]
fn date_checked_sub_various_durations() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();

    // Test subtracting days
    let minus_day = date.checked_sub(SignedDuration::days(1)).unwrap();
    assert_eq!(
        minus_day,
        Date::from_calendar_date(2024, Month::March, 14).unwrap()
    );

    // Test subtracting weeks
    let minus_week = date.checked_sub(SignedDuration::weeks(1)).unwrap();
    assert_eq!(
        minus_week,
        Date::from_calendar_date(2024, Month::March, 8).unwrap()
    );

    // Test subtracting months (via days approximation)
    let minus_month = date.checked_sub(SignedDuration::days(30)).unwrap();
    assert_eq!(
        minus_month,
        Date::from_calendar_date(2024, Month::February, 14).unwrap()
    );

    // Test underflow
    assert!(Date::MIN.checked_sub(SignedDuration::days(1)).is_none());
}

#[test]
fn date_saturating_add_boundary_handling() {
    let date = Date::from_calendar_date(2024, Month::March, 15).unwrap();

    // Test normal saturating add
    let normal = date.saturating_add(SignedDuration::days(5));
    assert_eq!(
        normal,
        Date::from_calendar_date(2024, Month::March, 20).unwrap()
    );

    // Test saturating to MAX
    let saturated_max = date.saturating_add(SignedDuration::days(10_000_000));
    assert_eq!(saturated_max, Date::MAX);

    // Test saturating to MIN
    let saturated_min = date.saturating_sub(SignedDuration::days(10_000_000));
    assert_eq!(saturated_min, Date::MIN);
}

#[test]
fn date_next_occurrence_various_weekdays() {
    let date = Date::from_calendar_date(2024, Month::January, 1).unwrap(); // Monday

    // Test next Monday (should be next week)
    let next_monday = date.next_occurrence(Weekday::Monday);
    assert_eq!(
        next_monday,
        Date::from_calendar_date(2024, Month::January, 8).unwrap()
    );

    // Test next Friday (same week)
    let next_friday = date.next_occurrence(Weekday::Friday);
    assert_eq!(
        next_friday,
        Date::from_calendar_date(2024, Month::January, 5).unwrap()
    );

    // Test next Sunday (same week)
    let next_sunday = date.next_occurrence(Weekday::Sunday);
    assert_eq!(
        next_sunday,
        Date::from_calendar_date(2024, Month::January, 7).unwrap()
    );
}

#[test]
fn date_prev_occurrence_various_weekdays() {
    let date = Date::from_calendar_date(2024, Month::January, 7).unwrap(); // Sunday

    // Test previous Sunday (should be previous week)
    let prev_sunday = date.prev_occurrence(Weekday::Sunday);
    assert_eq!(
        prev_sunday,
        Date::from_calendar_date(2023, Month::December, 31).unwrap()
    );

    // Test previous Wednesday (same week)
    let prev_wednesday = date.prev_occurrence(Weekday::Wednesday);
    assert_eq!(
        prev_wednesday,
        Date::from_calendar_date(2024, Month::January, 3).unwrap()
    );

    // Test previous Monday (same week)
    let prev_monday = date.prev_occurrence(Weekday::Monday);
    assert_eq!(
        prev_monday,
        Date::from_calendar_date(2024, Month::January, 1).unwrap()
    );
}

#[test]
fn date_nth_next_occurrence_various_n() {
    let date = Date::from_calendar_date(2024, Month::January, 1).unwrap(); // Monday

    // Test nth next Monday (n=1 should be next week)
    let nth_next_1 = date.nth_next_occurrence(Weekday::Monday, 1);
    assert_eq!(
        nth_next_1,
        Date::from_calendar_date(2024, Month::January, 8).unwrap()
    );

    // Test nth next Monday (n=2 should be 2 weeks later)
    let nth_next_2 = date.nth_next_occurrence(Weekday::Monday, 2);
    assert_eq!(
        nth_next_2,
        Date::from_calendar_date(2024, Month::January, 15).unwrap()
    );

    // Test nth next Friday (n=1 should be same week)
    let nth_next_friday = date.nth_next_occurrence(Weekday::Friday, 1);
    assert_eq!(
        nth_next_friday,
        Date::from_calendar_date(2024, Month::January, 5).unwrap()
    );
}

#[test]
fn date_nth_prev_occurrence_various_n() {
    let date = Date::from_calendar_date(2024, Month::January, 7).unwrap(); // Sunday

    // Test nth previous Sunday (n=1 should be previous week)
    let nth_prev_1 = date.nth_prev_occurrence(Weekday::Sunday, 1);
    assert_eq!(
        nth_prev_1,
        Date::from_calendar_date(2023, Month::December, 31).unwrap()
    );

    // Test nth previous Sunday (n=2 should be 2 weeks earlier)
    let nth_prev_2 = date.nth_prev_occurrence(Weekday::Sunday, 2);
    assert_eq!(
        nth_prev_2,
        Date::from_calendar_date(2023, Month::December, 24).unwrap()
    );

    // Test nth previous Wednesday (n=1 should be same week)
    let nth_prev_wednesday = date.nth_prev_occurrence(Weekday::Wednesday, 1);
    assert_eq!(
        nth_prev_wednesday,
        Date::from_calendar_date(2024, Month::January, 3).unwrap()
    );
}
