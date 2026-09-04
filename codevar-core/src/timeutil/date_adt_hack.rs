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

use crate::timeutil::date_well_know_iso8601::{
    Config, DateKind, FormattedComponents, Iso8601, OffsetPrecision, TimePrecision,
};
use core::num::NonZero;

// This provides a way to include `EncodedConfig` in documentation without displaying the type it is
// aliased to.
#[doc(hidden)]
pub type DoNotRelyOnWhatThisIs = u128;

/// An encoded [`Config`] that can be used as a const parameter to [`Iso8601`](super::Iso8601).
///
/// The type this is aliased to must not be relied upon. It can change in any release without
/// notice.
pub type EncodedConfig = DoNotRelyOnWhatThisIs;

impl<const CONFIG: EncodedConfig> Iso8601<CONFIG> {
    /// The user-provided configuration for the ISO 8601 format.
    const CONFIG: Config = Config::decode(CONFIG);
    /// Whether the date should be formatted.
    pub(crate) const FORMAT_DATE: bool = matches!(
        Self::CONFIG.formatted_components,
        FormattedComponents::Date
            | FormattedComponents::DateTime
            | FormattedComponents::DateTimeOffset
    );
    /// Whether the time should be formatted.
    pub(crate) const FORMAT_TIME: bool = matches!(
        Self::CONFIG.formatted_components,
        FormattedComponents::Time
            | FormattedComponents::DateTime
            | FormattedComponents::DateTimeOffset
            | FormattedComponents::TimeOffset
    );
    /// Whether the UTC offset should be formatted.
    pub(crate) const FORMAT_OFFSET: bool = matches!(
        Self::CONFIG.formatted_components,
        FormattedComponents::Offset
            | FormattedComponents::DateTimeOffset
            | FormattedComponents::TimeOffset
    );
    /// Whether the year is six digits.
    pub(crate) const YEAR_IS_SIX_DIGITS: bool = Self::CONFIG.year_is_six_digits;
    /// Whether the format contains separators (such as `-` or `:`).
    pub(crate) const USE_SEPARATORS: bool = Self::CONFIG.use_separators;
    /// Which format to use for the date.
    pub(crate) const DATE_KIND: DateKind = Self::CONFIG.date_kind;
    /// The precision and number of decimal digits to use for the time.
    pub(crate) const TIME_PRECISION: TimePrecision = Self::CONFIG.time_precision;
    /// The precision for the UTC offset.
    pub(crate) const OFFSET_PRECISION: OffsetPrecision = Self::CONFIG.offset_precision;
}

impl Config {
    /// Encode the configuration, permitting it to be used as a const parameter of [`Iso8601`].
    ///
    /// The value returned by this method must only be used as a const parameter to [`Iso8601`]. Any
    /// other usage is unspecified behavior.
    pub const fn encode(&self) -> EncodedConfig {
        let mut bytes = [0; EncodedConfig::BITS as usize / 8];

        bytes[0] = match self.formatted_components {
            FormattedComponents::None => 0,
            FormattedComponents::Date => 1,
            FormattedComponents::Time => 2,
            FormattedComponents::Offset => 3,
            FormattedComponents::DateTime => 4,
            FormattedComponents::DateTimeOffset => 5,
            FormattedComponents::TimeOffset => 6,
        };
        bytes[1] = self.use_separators as u8;
        bytes[2] = self.year_is_six_digits as u8;
        bytes[3] = match self.date_kind {
            DateKind::Calendar => 0,
            DateKind::Week => 1,
            DateKind::Ordinal => 2,
        };
        bytes[4] = match self.time_precision {
            TimePrecision::Hour { .. } => 0,
            TimePrecision::Minute { .. } => 1,
            TimePrecision::Second { .. } => 2,
        };
        bytes[5] = match self.time_precision {
            TimePrecision::Hour { decimal_digits }
            | TimePrecision::Minute { decimal_digits }
            | TimePrecision::Second { decimal_digits } => match decimal_digits {
                None => 0,
                Some(decimal_digits) => decimal_digits.get(),
            },
        };
        bytes[6] = match self.offset_precision {
            OffsetPrecision::Hour => 0,
            OffsetPrecision::Minute => 1,
        };

        EncodedConfig::from_be_bytes(bytes)
    }

    /// Decode the configuration. The configuration must have been generated from
    /// [`Config::encode`].
    pub(super) const fn decode(encoded: EncodedConfig) -> Self {
        let bytes = encoded.to_be_bytes();

        let formatted_components = match bytes[0] {
            0 => FormattedComponents::None,
            1 => FormattedComponents::Date,
            2 => FormattedComponents::Time,
            3 => FormattedComponents::Offset,
            4 => FormattedComponents::DateTime,
            5 => FormattedComponents::DateTimeOffset,
            6 => FormattedComponents::TimeOffset,
            _ => FormattedComponents::None,
        };
        let use_separators = matches!(bytes[1], 1);
        let year_is_six_digits = matches!(bytes[2], 1);
        let date_kind = match bytes[3] {
            0 => DateKind::Calendar,
            1 => DateKind::Week,
            2 => DateKind::Ordinal,
            _ => DateKind::Calendar,
        };
        let time_precision = match bytes[4] {
            0 => TimePrecision::Hour {
                decimal_digits: NonZero::new(bytes[5]),
            },
            1 => TimePrecision::Minute {
                decimal_digits: NonZero::new(bytes[5]),
            },
            2 => TimePrecision::Second {
                decimal_digits: NonZero::new(bytes[5]),
            },
            _ => TimePrecision::Hour {
                decimal_digits: None,
            },
        };
        let offset_precision = match bytes[6] {
            0 => OffsetPrecision::Hour,
            1 => OffsetPrecision::Minute,
            _ => OffsetPrecision::Hour,
        };
        let mut idx = 7; // first unused byte
        while idx < EncodedConfig::BITS as usize / 8 {
            if bytes[idx] != 0 {
                break;
            }
            idx += 1;
        }
        Self {
            formatted_components,
            use_separators,
            year_is_six_digits,
            date_kind,
            time_precision,
            offset_precision,
        }
    }
}
