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

//! Parsing and validation of D-Bus type signatures.
//!
//! A signature is a list of single complete types such as `sus` or
//! `a{sv}`. The rules follow the "Valid Signatures" section of the
//! D-Bus specification: at most 255 bytes, nesting depth of at most 32
//! arrays and 32 parentheses, and dict entries only as the element type
//! of an array.

use crate::dbus_error::{DbusError, DbusResult};

/// Maximum length of a signature in bytes.
pub const MAX_SIGNATURE_LEN: usize = 255;

/// Maximum nesting depth of arrays.
pub const MAX_ARRAY_DEPTH: usize = 32;

/// Maximum nesting depth of parentheses (structs and dict entries).
pub const MAX_STRUCT_DEPTH: usize = 32;

/// Returns the wire alignment of the type identified by `code`.
///
/// # Errors
///
/// Returns `None` when `code` is not a valid type code.
#[must_use]
pub const fn type_alignment(code: u8) -> Option<usize> {
    match code {
        b'y' | b'g' => Some(1),
        b'n' | b'q' => Some(2),
        b'b' | b'i' | b'u' | b'h' | b's' | b'o' => Some(4),
        b'x' | b't' | b'd' | b'a' | b'v' | b'(' | b'{' => Some(8),
        _ => None,
    }
}

/// Returns the size in bytes of a fixed-size type, `None` for
/// variable-size types.
#[must_use]
pub const fn type_fixed_size(code: u8) -> Option<usize> {
    match code {
        b'y' => Some(1),
        b'b' | b'i' | b'u' | b'h' => Some(4),
        b'n' | b'q' => Some(2),
        b'x' | b't' | b'd' => Some(8),
        _ => None,
    }
}

/// Returns `true` when `code` identifies a basic (non-container) type.
#[must_use]
pub const fn is_basic_type(code: u8) -> bool {
    matches!(
        code,
        b'y' | b'b'
            | b'n'
            | b'q'
            | b'i'
            | b'u'
            | b'x'
            | b't'
            | b'd'
            | b's'
            | b'o'
            | b'g'
            | b'h'
    )
}

/// Returns `true` when `code` identifies a container type.
///
/// Containers are arrays, structs, dict entries and variants.
#[must_use]
pub const fn is_container_type(code: u8) -> bool {
    matches!(code, b'a' | b'(' | b'{' | b'v')
}

enum ParseError {
    Truncated,
    Invalid,
    Depth,
}

/// Parses one single complete type starting at `pos`.
///
/// `allow_dict` must be set when parsing the element type of an array,
/// which is the only place a dict entry may appear.
fn parse_one(
    bytes: &[u8],
    pos: &mut usize,
    arrays: usize,
    structs: usize,
    allow_dict: bool,
) -> Result<(), ParseError> {
    let code = *bytes.get(*pos).ok_or(ParseError::Truncated)?;
    match code {
        b'y' | b'b' | b'n' | b'q' | b'i' | b'u' | b'x' | b't' | b'd' | b's' | b'o'
        | b'g' | b'h' => {
            *pos += 1;
            Ok(())
        }
        b'a' => {
            if arrays >= MAX_ARRAY_DEPTH {
                return Err(ParseError::Depth);
            }
            *pos += 1;
            parse_one(bytes, pos, arrays + 1, structs, true)
        }
        b'(' => {
            if structs >= MAX_STRUCT_DEPTH {
                return Err(ParseError::Depth);
            }
            *pos += 1;
            let mut fields = 0usize;
            loop {
                let next = *bytes.get(*pos).ok_or(ParseError::Truncated)?;
                if next == b')' {
                    break;
                }
                parse_one(bytes, pos, arrays, structs + 1, false)?;
                fields += 1;
            }
            if fields == 0 {
                return Err(ParseError::Invalid);
            }
            *pos += 1;
            Ok(())
        }
        b'{' => {
            if !allow_dict {
                return Err(ParseError::Invalid);
            }
            if structs >= MAX_STRUCT_DEPTH {
                return Err(ParseError::Depth);
            }
            *pos += 1;
            let key = *bytes.get(*pos).ok_or(ParseError::Truncated)?;
            if !is_basic_type(key) {
                return Err(ParseError::Invalid);
            }
            parse_one(bytes, pos, arrays, structs + 1, false)?;
            parse_one(bytes, pos, arrays, structs + 1, false)?;
            let close = *bytes.get(*pos).ok_or(ParseError::Truncated)?;
            if close != b'}' {
                return Err(ParseError::Invalid);
            }
            *pos += 1;
            Ok(())
        }
        b'v' => {
            *pos += 1;
            Ok(())
        }
        _ => Err(ParseError::Invalid),
    }
}

/// Returns the byte length of the first single complete type in `sig`.
///
/// Returns `None` when the signature is empty, truncated or invalid.
#[must_use]
pub fn single_complete_type_len(sig: &str) -> Option<usize> {
    let bytes = sig.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut pos = 0usize;
    parse_one(bytes, &mut pos, 0, 0, false).ok()?;
    Some(pos)
}

/// Iterates over the single complete types of a validated signature.
///
/// The iterator stops early when it reaches an invalid or truncated
/// type, so feeding it unvalidated input never loops forever.
#[derive(Debug, Clone)]
pub struct SignatureIter<'a> {
    rest: &'a str,
}

impl<'a> SignatureIter<'a> {
    /// Creates an iterator over the types of `sig`.
    #[must_use]
    pub fn new(sig: &'a str) -> Self {
        Self { rest: sig }
    }
}

impl<'a> Iterator for SignatureIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let length = single_complete_type_len(self.rest)?;
        let (item, rest) = self.rest.split_at(length);
        self.rest = rest;
        Some(item)
    }
}

/// Validates a complete signature.
///
/// # Errors
///
/// Returns [`DbusError::InvalidSignature`] when the signature exceeds
/// [`MAX_SIGNATURE_LEN`], contains an unknown type code, an empty
/// struct, a misplaced dict entry or nested containers deeper than the
/// limits of the specification.
pub fn validate_signature(sig: &str) -> DbusResult<()> {
    if sig.len() > MAX_SIGNATURE_LEN {
        return Err(DbusError::invalid_signature(alloc::format!(
            "signature of {} bytes exceeds the limit of {MAX_SIGNATURE_LEN}",
            sig.len()
        )));
    }
    let bytes = sig.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        parse_one(bytes, &mut pos, 0, 0, false).map_err(|kind| match kind {
            ParseError::Truncated => {
                DbusError::invalid_signature(alloc::format!("truncated signature: {sig}"))
            }
            ParseError::Depth => DbusError::invalid_signature(alloc::format!(
                "signature nests containers deeper than the specification allows: {sig}"
            )),
            ParseError::Invalid => {
                DbusError::invalid_signature(alloc::format!("invalid signature: {sig}"))
            }
        })?;
    }
    Ok(())
}

/// Validates a signature that must contain exactly one single complete
/// type, as required for variant values.
///
/// # Errors
///
/// Returns [`DbusError::InvalidSignature`] when `sig` does not describe
/// exactly one valid type.
pub fn validate_single_type(sig: &str) -> DbusResult<()> {
    if sig.len() > MAX_SIGNATURE_LEN {
        return Err(DbusError::invalid_signature(alloc::format!(
            "signature of {} bytes exceeds the limit of {MAX_SIGNATURE_LEN}",
            sig.len()
        )));
    }
    let length = single_complete_type_len(sig).ok_or_else(|| {
        DbusError::invalid_signature(alloc::format!("invalid variant signature: {sig}"))
    })?;
    if length != sig.len() {
        return Err(DbusError::invalid_signature(alloc::format!(
            "variant signature must hold exactly one type: {sig}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_and_sizes_match_the_specification() {
        assert_eq!(type_alignment(b'y'), Some(1));
        assert_eq!(type_alignment(b'g'), Some(1));
        assert_eq!(type_alignment(b'n'), Some(2));
        assert_eq!(type_alignment(b'i'), Some(4));
        assert_eq!(type_alignment(b's'), Some(4));
        assert_eq!(type_alignment(b'x'), Some(8));
        assert_eq!(type_alignment(b'a'), Some(8));
        assert_eq!(type_alignment(b'('), Some(8));
        assert_eq!(type_alignment(b'z'), None);

        assert_eq!(type_fixed_size(b'b'), Some(4));
        assert_eq!(type_fixed_size(b'd'), Some(8));
        assert_eq!(type_fixed_size(b's'), None);
        assert_eq!(type_fixed_size(b'a'), None);
    }

    #[test]
    fn basic_and_container_classification() {
        assert!(is_basic_type(b'u'));
        assert!(is_basic_type(b'o'));
        assert!(is_basic_type(b'h'));
        assert!(!is_basic_type(b'a'));
        assert!(!is_basic_type(b'v'));
        assert!(is_container_type(b'a'));
        assert!(is_container_type(b'('));
        assert!(is_container_type(b'{'));
        assert!(is_container_type(b'v'));
        assert!(!is_container_type(b'y'));
    }

    #[test]
    fn iterates_over_single_complete_types() {
        let types: Vec<&str> = SignatureIter::new("susa{sv}(ii)").collect();
        assert_eq!(types, alloc::vec!["s", "u", "s", "a{sv}", "(ii)"]);

        let types: Vec<&str> = SignatureIter::new("").collect();
        assert!(types.is_empty());

        let types: Vec<&str> = SignatureIter::new("aa(i)").collect();
        assert_eq!(types, alloc::vec!["aa(i)"]);
    }

    #[test]
    fn validates_full_signatures() {
        assert!(validate_signature("").is_ok());
        assert!(validate_signature("sus").is_ok());
        assert!(validate_signature("a{sv}").is_ok());
        assert!(validate_signature("a(sao)").is_ok());
        assert!(validate_signature("((ii)a{sd})").is_ok());
        assert!(validate_signature("v").is_ok());
        assert!(validate_signature("z").is_err());
        assert!(validate_signature("ii").is_ok());
        assert!(validate_signature("a").is_err());
        assert!(validate_signature("(i").is_err());
        assert!(validate_signature("()").is_err());
        assert!(validate_signature("(i)").is_ok());
        assert!(validate_signature("{sv}").is_err());
        assert!(validate_signature("a{is}").is_ok());
        assert!(validate_signature("a{(is)}").is_err());
    }

    #[test]
    fn rejects_excessive_nesting() {
        let deep = "a".repeat(MAX_ARRAY_DEPTH + 1);
        let deep = format!("{deep}i");
        assert!(validate_signature(&deep).is_err());

        let structs = "(".repeat(MAX_STRUCT_DEPTH + 1);
        let inner = format!("{structs}i{}", ")".repeat(MAX_STRUCT_DEPTH + 1));
        assert!(validate_signature(&inner).is_err());

        let ok = "a".repeat(MAX_ARRAY_DEPTH);
        let ok = format!("{ok}i");
        assert!(validate_signature(&ok).is_ok());
    }

    #[test]
    fn enforces_signature_length_limit() {
        let sig = "y".repeat(MAX_SIGNATURE_LEN);
        assert!(validate_signature(&sig).is_ok());
        let sig = "y".repeat(MAX_SIGNATURE_LEN + 1);
        assert!(validate_signature(&sig).is_err());
    }

    #[test]
    fn validates_variant_signatures() {
        assert!(validate_single_type("i").is_ok());
        assert!(validate_single_type("a{sv}").is_ok());
        assert!(validate_single_type("(ii)").is_ok());
        assert!(validate_single_type("").is_err());
        assert!(validate_single_type("ii").is_err());
        assert!(validate_single_type("z").is_err());
    }
}
