//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Diagnostics reported by the semantic analyzer.
//!
//! A [`Diagnostic`] is a severity, an optional stable error [`codes`], a
//! primary message, zero or more spanned labels, and trailing notes or a
//! help line. [`crate::emit`] renders the structure; the analyzer never
//! formats its own output.
//!
//! Error codes follow rustc's convention of a letter plus four digits:
//! `E` prefixes errors, `W` prefixes warnings, and every constant lives in
//! [`codes`] so call sites never spell a code by hand.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use codevar_ocl_parse::Span;

/// Severity of a [`Diagnostic`], ordered from most to least severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The input cannot be compiled further.
    Error,
    /// The input compiles but is suspicious.
    Warning,
    /// Extra context attached to an earlier diagnostic.
    Note,
}

impl Severity {
    /// The word rendered before the message.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Note => "note",
        }
    }
}

/// One spanned line under (or over) the source excerpt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// Region of the source the label points at.
    pub span: Span,
    /// Text printed next to the marker.
    pub message: String,
}

/// Everything one analysis run reports about a single problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// How severe the problem is.
    pub severity: Severity,
    /// Stable code such as `E0308`, or `None` for pass-through errors.
    pub code: Option<&'static str>,
    /// The headline message.
    pub message: String,
    /// Primary span; also used to order diagnostics.
    pub span: Span,
    /// Spanned labels; the first one is rendered as the primary marker.
    pub labels: Vec<Label>,
    /// `= note:` lines rendered after the source excerpt.
    pub notes: Vec<String>,
    /// `= help:` line rendered after the notes.
    pub help: Option<String>,
}

impl Diagnostic {
    /// An error diagnostic with `span` as its primary label.
    #[must_use]
    pub fn error(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, span, message)
    }

    /// A warning diagnostic with `span` as its primary label.
    #[must_use]
    pub fn warning(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, span, message)
    }

    /// A note diagnostic with `span` as its primary label.
    #[must_use]
    pub fn note(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Note, span, message)
    }

    fn new(severity: Severity, span: Span, message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            severity,
            code: None,
            span,
            labels: alloc::vec![Label {
                span,
                message: String::new(),
            }],
            message,
            notes: Vec::new(),
            help: None,
        }
    }

    /// Attaches a stable error code, e.g. [`codes::MISMATCHED_TYPES`].
    #[must_use]
    pub const fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    /// Adds an extra spanned label under the source excerpt.
    #[must_use]
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
        });
        self
    }

    /// Adds a `= note:` line.
    #[must_use]
    pub fn with_note(mut self, message: impl Into<String>) -> Self {
        self.notes.push(message.into());
        self
    }

    /// Sets the `= help:` line (replacing any previous one).
    #[must_use]
    pub fn with_help(mut self, message: impl Into<String>) -> Self {
        self.help = Some(message.into());
        self
    }

    /// True when the diagnostic stops compilation.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self.severity, Severity::Error)
    }

    /// True when the diagnostic is a non-fatal warning.
    #[must_use]
    pub const fn is_warning(&self) -> bool {
        matches!(self.severity, Severity::Warning)
    }
}

/// Incrementally assembles a [`Diagnostic`] message from typed fragments.
///
/// Building messages through the helper keeps call sites readable when a
/// sentence mixes prose with displayed types or quoted names.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{Diagnostic, MessageBuilder, Span};
///
/// let message = MessageBuilder::new()
///     .text("expected `")
///     .text("int")
///     .text("`, found `")
///     .text("bool")
///     .text("`")
///     .finish();
/// let diagnostic = Diagnostic::error(Span::new(0, 0), message);
/// assert_eq!(diagnostic.message, "expected `int`, found `bool`");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MessageBuilder {
    buffer: String,
}

impl MessageBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Appends raw text.
    #[must_use]
    pub fn text(mut self, text: impl AsRef<str>) -> Self {
        self.buffer.push_str(text.as_ref());
        self
    }

    /// Appends `name` wrapped in backticks.
    #[must_use]
    pub fn quoted(mut self, name: impl AsRef<str>) -> Self {
        self.buffer.push('`');
        self.buffer.push_str(name.as_ref());
        self.buffer.push('`');
        self
    }

    /// Appends the display form of a type, backticked.
    #[must_use]
    pub fn ty(mut self, ty: &crate::types::Ty) -> Self {
        self.buffer.push('`');
        fmt::write(&mut self.buffer, format_args!("{ty}")).ok();
        self.buffer.push('`');
        self
    }

    /// Returns the assembled message.
    #[must_use]
    pub fn finish(self) -> String {
        self.buffer
    }
}

/// Stable diagnostic codes emitted by this crate.
///
/// Errors use `E` plus four digits; warnings use `W`. Numbers are grouped
/// by analysis stage: `E00xx` lexical, `E01xx` generics, `E03xx` types and
/// expressions, `E04xx` names, `E06xx` items, `E07xx` keywords, `W04xx`
/// lints, `W07xx` reserved words, `W08xx` confusable identifiers.
pub mod codes {
    // Lexical.
    /// A character that no token can start.
    pub const UNKNOWN_CHARACTER: &str = "E0001";
    /// A literal prefix the dialect does not define, e.g. `foo"bar"`.
    pub const UNKNOWN_LITERAL_PREFIX: &str = "E0002";
    /// A string literal with no closing quote.
    pub const UNTERMINATED_STRING: &str = "E0003";
    /// A character literal with no closing quote.
    pub const UNTERMINATED_CHAR: &str = "E0004";
    /// A block comment with no closing `*/`.
    pub const UNTERMINATED_BLOCK_COMMENT: &str = "E0005";
    /// A malformed raw string such as `r#"abc`.
    pub const INVALID_RAW_STRING: &str = "E0006";
    /// A literal with no digits, e.g. `0x`.
    pub const EMPTY_INTEGER_LITERAL: &str = "E0007";
    /// A float literal with an empty exponent, e.g. `1e`.
    pub const EMPTY_EXPONENT: &str = "E0008";
    /// A lifetime that starts with a digit, e.g. `'0`.
    pub const LIFETIME_STARTS_WITH_NUMBER: &str = "E0009";
    /// A literal suffix the dialect does not define, e.g. `1u128`.
    pub const INVALID_LITERAL_SUFFIX: &str = "E0010";
    /// A literal that does not fit the type its suffix names.
    pub const LITERAL_OUT_OF_RANGE: &str = "E0011";
    /// A character literal holding more than one character.
    pub const CHAR_LITERAL_TOO_LONG: &str = "E0012";
    /// An escape sequence that is not defined, e.g. `\q`.
    pub const UNKNOWN_ESCAPE_SEQUENCE: &str = "E0013";

    /// A control character outside the allowed whitespace set.
    pub const CONTROL_CHARACTER: &str = "E0100";
    /// A bidirectional control character that can reorder rendered text.
    pub const BIDI_CONTROL: &str = "E0101";
    /// An invisible, zero-width character.
    pub const INVISIBLE_CHARACTER: &str = "E0102";
    /// A code point Unicode reserves and never assigns.
    pub const NONCHARACTER: &str = "E0103";
    /// Input that is not valid UTF-8.
    pub const INVALID_UTF8: &str = "E0104";

    /// The wrong number of generic arguments, e.g. `Pair<int, bool>` on a
    /// one-parameter alias.
    pub const GENERIC_ARITY: &str = "E0107";

    /// A name with no definition in scope.
    pub const UNDEFINED_NAME: &str = "E0412";
    /// A function used where a value is expected.
    pub const UNEXPECTED_FUNCTION: &str = "E0419";
    /// A type used where a value is expected.
    pub const UNEXPECTED_TYPE: &str = "E0420";
    /// A multi-segment path such as `a::b`; the dialect has no modules.
    pub const MULTI_SEGMENT_PATH: &str = "E0433";

    /// A call with too few or too many arguments.
    pub const WRONG_NUMBER_OF_ARGUMENTS: &str = "E0061";
    /// Assignment to a place that cannot hold a value.
    pub const INVALID_ASSIGNMENT_TARGET: &str = "E0307";
    /// Assignment to a variable that is not `mut`.
    pub const ASSIGN_TO_IMMUTABLE: &str = "E0070";
    /// A `break` with no enclosing loop.
    pub const BREAK_OUTSIDE_LOOP: &str = "E0267";
    /// A `continue` with no enclosing loop.
    pub const CONTINUE_OUTSIDE_LOOP: &str = "E0268";
    /// A call of something that is not a function.
    pub const EXPECTED_FUNCTION: &str = "E0428";

    /// A type name that cannot be resolved.
    pub const UNKNOWN_TYPE: &str = "E0301";
    /// A function name used in type position.
    pub const NOT_A_TYPE: &str = "E0302";
    /// A name defined twice in the same namespace.
    pub const DUPLICATE_DEFINITION: &str = "E0425";
    /// A type alias that expands to itself.
    pub const ALIAS_CYCLE: &str = "E0426";
    /// A struct whose fields make it infinitely sized.
    pub const RECURSIVE_TYPE: &str = "E0427";
    /// An `as` cast between incompatible types.
    pub const INVALID_CAST: &str = "E0303";
    /// A condition that is not `bool`, or is not scalar.
    pub const INVALID_CONDITION: &str = "E0304";
    /// A binary operator applied to an unsupported type.
    pub const INVALID_BINARY_OPERAND: &str = "E0305";
    /// A unary operator applied to an unsupported type.
    pub const INVALID_UNARY_OPERAND: &str = "E0306";
    /// A type error between two expressions.
    pub const MISMATCHED_TYPES: &str = "E0308";
    /// Indexing something that is not indexable, or with a non-integer.
    pub const INVALID_INDEX: &str = "E0309";
    /// A field that the type does not have.
    pub const UNKNOWN_FIELD: &str = "E0310";
    /// A binding or array literal whose type cannot be inferred.
    pub const MISSING_TYPE_ANNOTATION: &str = "E0311";
    /// An array length that is not a constant integer.
    pub const INVALID_ARRAY_LENGTH: &str = "E0312";
    /// A vector type with a lane count OpenCL does not define.
    pub const INVALID_VECTOR_LANES: &str = "E0313";
    /// A string literal; the dialect has no string type.
    pub const UNSUPPORTED_STRING: &str = "E0314";
    /// A range expression outside a `for` header.
    pub const RANGE_NOT_ALLOWED: &str = "E0315";
    /// A `for` loop over something that cannot be iterated.
    pub const NOT_ITERABLE: &str = "E0316";
    /// The type `void` used where a value type is required.
    pub const INVALID_VOID: &str = "E0317";

    /// The `?` operator, which the dialect does not support.
    pub const TRY_NOT_SUPPORTED: &str = "E0600";
    /// A `#[kernel]` function that does not return `void`.
    pub const KERNEL_RETURN: &str = "E0601";
    /// A parameter type a `#[kernel]` function may not declare.
    pub const KERNEL_PARAM: &str = "E0602";
    /// A call cycle; the dialect forbids recursion.
    pub const RECURSION: &str = "E0603";
    /// A parameter of type `void`.
    pub const VOID_PARAM: &str = "E0604";
    /// An attribute the dialect does not define.
    pub const UNKNOWN_ATTRIBUTE: &str = "E0605";
    /// An attribute applied to something other than a function.
    pub const ATTRIBUTE_TARGET: &str = "E0606";
    /// A call of a `#[kernel]` function from another function.
    pub const KERNEL_CALL: &str = "E0607";
    /// The same attribute written twice on one item.
    pub const DUPLICATE_ATTRIBUTE: &str = "E0608";

    /// A declaration whose name is a dialect keyword.
    pub const KEYWORD_AS_NAME: &str = "E0701";

    /// A binding that is never read.
    pub const UNUSED_VARIABLE: &str = "W0421";
    /// An item that is never referenced.
    pub const UNUSED_ITEM: &str = "W0429";
    /// A name reserved by OpenCL, e.g. `kernel` or `uniform`.
    pub const RESERVED_IDENTIFIER: &str = "W0702";
    /// An identifier that visually resembles another.
    pub const CONFUSABLE_IDENTIFIER: &str = "W0801";
}
