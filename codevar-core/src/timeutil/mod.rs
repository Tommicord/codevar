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

pub mod date;
mod date_adt_hack;
pub mod date_component_provider;
pub mod date_error;
mod date_format_description;
mod date_format_description_modifier;
pub mod date_formattable;
mod date_formatting;
mod date_internal_macro;
pub mod date_iso8601;
mod date_metadata;
pub mod date_month;
mod date_num_fmt;
pub mod date_offset_time;
pub mod date_plain;
pub mod date_signed_duration;
pub mod date_time;
pub mod date_timestamp;
pub mod date_unit;
pub mod date_utc_offset;
pub mod date_utc_time;
pub mod date_util;
pub mod date_weekday;
mod date_well_know_iso8601;
mod date_well_know_rfc2822;
mod date_well_know_rfc3339;
