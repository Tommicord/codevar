//! Copyright 2026 Codevar Project
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

use core::fmt;

/// An error type indicating that a component provided to a method was out of range, causing a
/// failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentRange {
    /// Name of the component.
    pub(crate) name: &'static str,
    /// Whether an input with the same value could have succeeded if the values of other components
    /// were different.
    pub(crate) is_conditional: bool,
}

impl ComponentRange {
    /// Create a new `ComponentRange` error that is not conditional.
    #[inline]
    pub(crate) const fn unconditional(name: &'static str) -> Self {
        Self {
            name,
            is_conditional: false,
        }
    }

    /// Create a new `ComponentRange` error that is conditional.
    #[inline]
    pub(crate) const fn conditional(name: &'static str) -> Self {
        Self {
            name,
            is_conditional: true,
        }
    }

    /// Obtain the name of the component whose value was out of range.
    #[inline]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Whether the value's permitted range is conditional, i.e. whether an input with this
    /// value could have succeeded if the values of other components were different.
    #[inline]
    pub const fn is_conditional(self) -> bool {
        self.is_conditional
    }
}

impl fmt::Display for ComponentRange {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} was not in range", self.name)
    }
}

impl core::error::Error for ComponentRange {}

/// An error type indicating that a conversion failed because the target type could not store the
/// initial value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionRange;

impl fmt::Display for ConversionRange {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Source value is out of range for the target type")
    }
}

impl core::error::Error for ConversionRange {}

/// An error occurred when formatting.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// The type being formatted does not contain sufficient information to format a component.
    InsufficientTypeInformation,
    /// The component named has a value that cannot be formatted into the requested format.
    /// This variant is only returned when using well-known formats.
    InvalidComponent(&'static str),
    /// A component provided was out of range.
    ComponentRange(ComponentRange),
    /// A conversion failed because the target type could not store the initial value.
    ConversionRange(ConversionRange),
    /// UTC offset cannot be determined.
    IndeterminateOffset(IndeterminateOffset),
    /// A value of `core::core::fmt::Error` was returned internally.
    StdIo(core::fmt::Error),
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientTypeInformation => write!(
                f,
                "The type being formatted does not contain sufficient information to format a \
                 component.",
            ),
            Self::InvalidComponent(component) => write!(
                f,
                "The {component} component cannot be formatted into the requested format."
            ),
            Self::ComponentRange(err) => err.fmt(f),
            Self::ConversionRange(err) => err.fmt(f),
            Self::IndeterminateOffset(err) => err.fmt(f),
            Self::StdIo(err) => err.fmt(f),
        }
    }
}

impl From<ComponentRange> for Error {
    #[inline]
    fn from(err: ComponentRange) -> Self {
        Self::ComponentRange(err)
    }
}

impl From<ConversionRange> for Error {
    #[inline]
    fn from(err: ConversionRange) -> Self {
        Self::ConversionRange(err)
    }
}

impl From<core::fmt::Error> for Error {
    #[inline]
    fn from(err: core::fmt::Error) -> Self {
        Self::StdIo(err)
    }
}

impl core::error::Error for Error {
    #[inline]
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InsufficientTypeInformation | Self::InvalidComponent(_) => None,
            Self::ComponentRange(err) => Some(err),
            Self::ConversionRange(err) => Some(err),
            Self::IndeterminateOffset(err) => Some(err),
            Self::StdIo(err) => Some(err),
        }
    }
}

/// An error type indicating that a [`FromStr`](core::str::FromStr) call failed because the value
/// was not a valid variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidVariant;

impl fmt::Display for InvalidVariant {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "value was not a valid variant")
    }
}

impl core::error::Error for InvalidVariant {}

/// An error returned when the local UTC offset cannot be determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndeterminateOffset;

impl fmt::Display for IndeterminateOffset {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("The UTC offset cannot be determined")
    }
}

impl core::error::Error for IndeterminateOffset {}

#[cfg(test)]
mod tests {
    use core::error::Error as _;

    use super::*;

    #[test]
    fn component_range_constructors_and_accessors() {
        let unconditional = ComponentRange::unconditional("day");
        assert_eq!(unconditional.name(), "day");
        assert!(!unconditional.is_conditional());

        let conditional = ComponentRange::conditional("year");
        assert_eq!(conditional.name(), "year");
        assert!(conditional.is_conditional());

        assert_ne!(unconditional, conditional);
        assert_eq!(unconditional, ComponentRange::unconditional("day"));
        let copy = unconditional;
        assert_eq!(copy, unconditional);
    }

    #[test]
    fn component_range_display_includes_component_name() {
        assert_eq!(
            ComponentRange::unconditional("day").to_string(),
            "day was not in range"
        );
        assert_eq!(
            ComponentRange::conditional("nanosecond").to_string(),
            "nanosecond was not in range",
        );
        assert_eq!(
            ComponentRange::unconditional("offset hour").to_string(),
            "offset hour was not in range",
        );
        // Padding flags are ignored: the implementation writes the message directly.
        assert_eq!(
            format!("{:>25}", ComponentRange::unconditional("day")),
            "day was not in range",
        );
    }

    #[test]
    fn simple_error_displays() {
        assert_eq!(
            ConversionRange.to_string(),
            "Source value is out of range for the target type",
        );
        assert_eq!(InvalidVariant.to_string(), "value was not a valid variant");
        assert_eq!(
            IndeterminateOffset.to_string(),
            "The UTC offset cannot be determined",
        );
    }

    #[test]
    fn error_displays_each_variant() {
        assert_eq!(
            Error::InsufficientTypeInformation.to_string(),
            "The type being formatted does not contain sufficient information to format a \
             component.",
        );
        assert_eq!(
            Error::InvalidComponent("offset hour").to_string(),
            "The offset hour component cannot be formatted into the requested format.",
        );
        assert_eq!(
            Error::InvalidComponent("seconds").to_string(),
            "The seconds component cannot be formatted into the requested format.",
        );
        let range = ComponentRange::unconditional("day");
        assert_eq!(Error::ComponentRange(range).to_string(), range.to_string());
        assert_eq!(
            Error::ConversionRange(ConversionRange).to_string(),
            ConversionRange.to_string(),
        );
        assert_eq!(
            Error::IndeterminateOffset(IndeterminateOffset).to_string(),
            IndeterminateOffset.to_string(),
        );
        assert_eq!(
            Error::StdIo(core::fmt::Error).to_string(),
            core::fmt::Error.to_string(),
        );
    }

    #[test]
    fn error_from_conversions_wrap_sources() {
        let range = ComponentRange::conditional("month");
        let from_range: Error = range.into();
        assert!(matches!(from_range, Error::ComponentRange(actual) if actual == range));
        assert_eq!(from_range.to_string(), "month was not in range");

        let conversion: Error = ConversionRange.into();
        assert!(matches!(conversion, Error::ConversionRange(ConversionRange)));
        let again: Error = ConversionRange.into();
        assert!(matches!(again, Error::ConversionRange(ConversionRange)));
        assert_eq!(again.to_string(), conversion.to_string());

        let fmt_error: Error = core::fmt::Error.into();
        assert!(matches!(fmt_error, Error::StdIo(_)));
        assert_eq!(fmt_error.to_string(), core::fmt::Error.to_string());
    }

    #[test]
    fn error_source_follows_variant() {
        assert!(
            Error::InsufficientTypeInformation
                .source()
                .is_none()
        );
        assert!(Error::InvalidComponent("hour").source().is_none());

        let error = Error::ComponentRange(ComponentRange::unconditional("day"));
        let source = error.source().expect("source should be present");
        assert_eq!(source.to_string(), "day was not in range");
        assert!(source.downcast_ref::<ComponentRange>().is_some());
        assert!(source.downcast_ref::<ConversionRange>().is_none());

        let error = Error::ConversionRange(ConversionRange);
        let source = error.source().expect("source should be present");
        assert!(source.downcast_ref::<ConversionRange>().is_some());

        let error = Error::IndeterminateOffset(IndeterminateOffset);
        let source = error.source().expect("source should be present");
        assert!(
            source
                .downcast_ref::<IndeterminateOffset>()
                .is_some()
        );

        let error = Error::StdIo(core::fmt::Error);
        let source = error.source().expect("source should be present");
        assert!(
            source
                .downcast_ref::<core::fmt::Error>()
                .is_some()
        );
    }

    #[test]
    fn leaf_errors_have_no_source() {
        assert!(
            ComponentRange::unconditional("day")
                .source()
                .is_none()
        );
        assert!(ConversionRange.source().is_none());
        assert!(InvalidVariant.source().is_none());
        assert!(IndeterminateOffset.source().is_none());
    }

    #[test]
    fn errors_can_be_used_as_dyn_error() {
        let errors: Vec<(Box<dyn core::error::Error>, &str)> = vec![
            (
                Box::new(ComponentRange::unconditional("day")),
                "day was not in range",
            ),
            (
                Box::new(ConversionRange),
                "Source value is out of range for the target type",
            ),
            (Box::new(InvalidVariant), "value was not a valid variant"),
            (
                Box::new(IndeterminateOffset),
                "The UTC offset cannot be determined",
            ),
            (Box::new(Error::InsufficientTypeInformation), ""),
        ];
        for (error, expected) in errors {
            if !expected.is_empty() {
                assert_eq!(error.to_string(), expected);
            }
            let _ = error.source();
        }
    }

    #[test]
    fn component_range_is_hashable() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        assert_eq!(
            hash_of(&ComponentRange::unconditional("day")),
            hash_of(&ComponentRange::unconditional("day")),
        );
        assert_ne!(
            hash_of(&ComponentRange::unconditional("day")),
            hash_of(&ComponentRange::unconditional("hour")),
        );
        assert_ne!(
            hash_of(&ComponentRange::unconditional("day")),
            hash_of(&ComponentRange::conditional("day")),
        );
    }
}
