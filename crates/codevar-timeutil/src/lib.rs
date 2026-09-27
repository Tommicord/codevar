//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

#![cfg_attr(not(test), no_std)]
#![allow(dead_code)]
extern crate alloc;

mod date;
mod date_adt_hack;
mod date_component_provider;
mod date_error;
mod date_format_description;
mod date_format_description_modifier;
mod date_formattable;
mod date_formatting;
mod date_internal_macro;
mod date_iso8601;
mod date_metadata;
mod date_month;
mod date_num_fmt;
mod date_offset_time;
mod date_plain;
mod date_signed_duration;
mod date_time;
mod date_timestamp;
mod date_unit;
mod date_utc_offset;
mod date_utc_time;
mod date_util;
mod date_weekday;
mod date_well_know_iso8601;
mod date_well_know_rfc2822;
mod date_well_know_rfc3339;

pub use crate::date::Date;
pub use crate::date_signed_duration::SignedDuration;
pub use crate::date_time::Time;
pub use crate::date_timestamp::Timestamp;
pub use crate::date_unit::{Day, Hour, Microsecond, Millisecond, Minute, Nanosecond, Second};
pub use crate::date_utc_offset::UtcOffset;
pub use crate::date_utc_time::UtcDateTime;

/// Returns the current UTC timestamp using the system realtime clock.
/// This is a convenience function for logging and time-stamping operations.
#[inline]
pub fn utc_now() -> Timestamp {
    Timestamp::now()
}

/// Returns the current UTC date-time.
#[inline]
pub fn utc_datetime_now() -> Option<UtcDateTime> {
    Timestamp::now().to_utc()
}

/// Formats a UTC timestamp as an ISO 8601 string (YYYY-MM-DDTHH:MM:SS.sssssssssZ).
/// The buffer must be at least 32 bytes.
pub fn format_utc_iso8601(ts: Timestamp, buf: &mut [u8]) -> Result<usize, ()> {
    if buf.len() < 32 {
        return Err(());
    }
    let utc = ts.to_utc().ok_or(())?;
    let (year, month, day) = utc.to_calendar_date();
    let (hour, minute, second, nanosecond) = utc.time().as_hms_nano();

    let mut idx = 0;
    write_i32_fixed(&mut buf[idx..], year, 4)?;
    idx += 4;
    buf[idx] = b'-';
    idx += 1;
    write_u8_fixed(&mut buf[idx..], month as u8, 2)?;
    idx += 2;
    buf[idx] = b'-';
    idx += 1;
    write_u8_fixed(&mut buf[idx..], day, 2)?;
    idx += 2;
    buf[idx] = b'T';
    idx += 1;
    write_u8_fixed(&mut buf[idx..], hour, 2)?;
    idx += 2;
    buf[idx] = b':';
    idx += 1;
    write_u8_fixed(&mut buf[idx..], minute, 2)?;
    idx += 2;
    buf[idx] = b':';
    idx += 1;
    write_u8_fixed(&mut buf[idx..], second, 2)?;
    idx += 2;
    if nanosecond != 0 {
        buf[idx] = b'.';
        idx += 1;
        write_u32_fixed(&mut buf[idx..], nanosecond, 9)?;
        idx += 9;
    }
    buf[idx] = b'Z';
    idx += 1;
    Ok(idx)
}

#[inline]
fn write_i32_fixed(buf: &mut [u8], mut val: i32, width: usize) -> Result<(), ()> {
    if buf.len() < width {
        return Err(());
    }
    if val < 0 {
        buf[0] = b'-';
        val = -val;
        return write_u32_fixed(&mut buf[1..], val as u32, width - 1);
    }
    write_u32_fixed(buf, val as u32, width)
}

#[inline]
fn write_u32_fixed(buf: &mut [u8], mut val: u32, width: usize) -> Result<(), ()> {
    if buf.len() < width {
        return Err(());
    }
    for i in (0..width).rev() {
        buf[i] = b'0' + (val % 10) as u8;
        val /= 10;
    }
    Ok(())
}

#[inline]
fn write_u8_fixed(buf: &mut [u8], mut val: u8, width: usize) -> Result<(), ()> {
    if buf.len() < width {
        return Err(());
    }
    for i in (0..width).rev() {
        buf[i] = b'0' + (val % 10);
        val /= 10;
    }
    Ok(())
}
