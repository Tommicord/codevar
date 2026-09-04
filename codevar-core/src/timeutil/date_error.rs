use alloc::boxed::Box;
use core::fmt;
use std::io;

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
    /// A value of `std::io::Error` was returned internally.
    StdIo(io::Error),
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

impl From<io::Error> for Error {
    #[inline]
    fn from(err: io::Error) -> Self {
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
