use codevar_core::timeutil::date::Date;
use codevar_core::timeutil::date_month::Month;
use codevar_core::timeutil::date_time::Time;
use codevar_core::timeutil::date_utc_offset::UtcOffset;

#[test]
fn date_and_time_display_roundtrip() {
    let date = Date::from_calendar_date(2024, Month::February, 29).unwrap();
    let text = format!("{date}");
    assert!(text.contains("2024"));
    assert!(text.contains("02"));
    assert!(text.contains("29"));

    let time = Time::from_hms(12, 34, 56).unwrap();
    let time_text = format!("{time}");
    assert!(time_text.contains("12"));
    assert!(time_text.contains("34"));
    assert!(time_text.contains("56"));

    let offset = UtcOffset::from_hms(2, 30, 0).unwrap();
    let offset_text = format!("{offset}");
    assert!(offset_text.contains("02") || offset_text.contains("2"));
}

#[test]
fn formatting_helpers_generate_expected_shapes() {
    let date = Date::from_calendar_date(2024, Month::January, 1).unwrap();
    assert_eq!(date.to_string(), "2024-01-01");

    let noon = Time::from_hms(12, 0, 0).unwrap();
    assert_eq!(noon.to_string(), "12:00:00");

    let offset = UtcOffset::from_hms(0, 0, 0).unwrap();
    assert_eq!(offset.to_string(), "+00:00:00");
}
