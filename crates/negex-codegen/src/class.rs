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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Compiling parsed character classes into byte sets.
//!
//! Negex matches a byte stream.  RFC 9485 character classes are expressed
//! over Unicode scalars, so this module is the boundary where that model is
//! mapped onto bytes:
//!
//! * A literal or class member at or below U+00FF is one byte.
//! * A class member above U+00FF is rejected with
//!   [`NegexError::ClassOutOfByteRange`] rather than silently truncated.
//! * A Unicode general category is compiled to its ASCII members, the only
//!   members a byte set can represent exactly.  Unknown categories are a
//!   hard error, never a wildcard.
//!
//! Rejecting what cannot be represented exactly keeps the engine from
//! accepting a pattern that would later match the wrong bytes.

use crate::{ByteClass, NegexError};
use codevar_negex_parse::{Class, ClassItem};

/// Compiles a parsed class expression into the set of bytes it matches.
///
/// # Errors
///
/// Returns [`NegexError::ClassOutOfByteRange`] for a scalar above U+00FF,
/// [`NegexError::InvalidClassRange`] for a backwards range,
/// [`NegexError::UnknownProperty`] for an unrecognised category, and
/// [`NegexError::InvalidClassAst`] for a parse-error placeholder.
pub fn byte_class(class: &Class) -> Result<ByteClass, NegexError> {
    match class {
        Class::Positive { items, negated } => {
            let mut bytes = ByteClass::empty();
            for item in items {
                bytes.union(class_item(item)?);
            }
            if *negated {
                bytes.invert();
            }
            Ok(bytes)
        }
        Class::Property { category, negated } => property(category, *negated),
        Class::Error { .. } => Err(NegexError::InvalidClassAst),
    }
}

/// Compiles one item inside a class expression.
fn class_item(item: &ClassItem) -> Result<ByteClass, NegexError> {
    match item {
        ClassItem::Char(scalar) => Ok(ByteClass::singleton(byte_of(*scalar)?)),
        ClassItem::Range { start, end } => {
            let start_byte = byte_of(*start)?;
            let end_byte = byte_of(*end)?;
            if start_byte > end_byte {
                return Err(NegexError::InvalidClassRange {
                    start: *start,
                    end: *end,
                });
            }
            Ok(ByteClass::range(start_byte, end_byte))
        }
        ClassItem::Property { category, negated } => property(category, *negated),
        ClassItem::Error { .. } => Err(NegexError::InvalidClassAst),
    }
}

/// The byte value of a scalar that fits in one byte.
fn byte_of(scalar: char) -> Result<u8, NegexError> {
    u8::try_from(u32::from(scalar)).map_err(|_| NegexError::ClassOutOfByteRange { scalar })
}

/// Compiles a Unicode general-category escape to its ASCII members.
fn property(category: &str, negated: bool) -> Result<ByteClass, NegexError> {
    let name = category.strip_prefix("Is").unwrap_or(category);
    let mut bytes = ascii_category(name).ok_or_else(|| NegexError::UnknownProperty {
        category: alloc::string::String::from(category),
    })?;
    if negated {
        bytes.invert();
    }
    Ok(bytes)
}

/// The ASCII members of a Unicode general category.
///
/// Returns `None` for a name that is not a general category.  ASCII has no
/// combining marks, so the `M*` categories are empty rather than unknown.
fn ascii_category(name: &str) -> Option<ByteClass> {
    let mut class = ByteClass::empty();
    match name {
        "L" | "Letter" => {
            class.insert_range(b'A', b'Z');
            class.insert_range(b'a', b'z');
        }
        "Lu" => class.insert_range(b'A', b'Z'),
        "Ll" => class.insert_range(b'a', b'z'),
        "Lt" | "Lm" | "Lo" => {}
        "M" | "Mn" | "Mc" | "Me" => {}
        "N" | "Nd" | "Number" => class.insert_range(b'0', b'9'),
        "Nl" | "No" => {}
        "P" | "Punctuation" => {
            class.insert_range(0x21, 0x2F);
            class.insert_range(0x3A, 0x40);
            class.insert_range(0x5B, 0x60);
            class.insert_range(0x7B, 0x7E);
        }
        "Pc" => class.insert(b'_'),
        "Pd" => class.insert(b'-'),
        "Ps" => {
            class.insert(b'(');
            class.insert(b'[');
            class.insert(b'{');
        }
        "Pe" => {
            class.insert(b')');
            class.insert(b']');
            class.insert(b'}');
        }
        "Pi" | "Pf" | "Po" => {
            class.insert_range(0x21, 0x2F);
            class.insert_range(0x3A, 0x40);
            class.insert_range(0x5B, 0x60);
            class.insert_range(0x7B, 0x7E);
            class.remove(b'_');
            class.remove(b'-');
            class.remove(b'(');
            class.remove(b'[');
            class.remove(b'{');
            class.remove(b')');
            class.remove(b']');
            class.remove(b'}');
        }
        "S" | "Symbol" => {
            class.insert(b'$');
            class.insert(b'+');
            class.insert(b'<');
            class.insert(b'=');
            class.insert(b'>');
            class.insert(b'^');
            class.insert(b'`');
            class.insert(b'|');
            class.insert(b'~');
        }
        "Sm" => {
            class.insert(b'+');
            class.insert(b'<');
            class.insert(b'=');
            class.insert(b'>');
            class.insert(b'|');
            class.insert(b'~');
        }
        "Sc" => class.insert(b'$'),
        "Sk" => {
            class.insert(b'^');
            class.insert(b'`');
        }
        "So" => {}
        "Z" | "Zs" | "Separator" => class.insert(b' '),
        "Zl" | "Zp" => {}
        "C" | "Cc" | "Control" => {
            class.insert_range(0x00, 0x1F);
            class.insert(0x7F);
        }
        "Cf" | "Co" | "Cs" | "Cn" => {}
        _ => return None,
    }
    Some(class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_negex_parse::parse;

    fn class_of(pattern: &str) -> ByteClass {
        let output = parse(pattern);
        assert!(output.errors.is_empty(), "{pattern} must parse");
        let atom = &output.ast.branches[0].pieces[0].atom;
        match atom {
            codevar_negex_parse::Atom::Class(class) => byte_class(class).expect("class compiles"),
            _ => panic!("expected a class atom for {pattern}"),
        }
    }

    #[test]
    fn digit_range_matches_digits_only() {
        let class = class_of("[0-9]");
        assert!(class.contains(b'0'));
        assert!(class.contains(b'9'));
        assert!(!class.contains(b'a'));
        assert!(!class.contains(b'/'));
    }

    #[test]
    fn negated_range_inverts() {
        let class = class_of("[^0-9]");
        assert!(!class.contains(b'5'));
        assert!(class.contains(b'a'));
        assert!(class.contains(0xFF));
    }

    #[test]
    fn single_char_class() {
        let class = class_of("[x]");
        assert!(class.contains(b'x'));
        assert!(!class.contains(b'y'));
    }

    #[test]
    fn ascii_property_letters() {
        let class = class_of(r"\p{L}");
        assert!(class.contains(b'A'));
        assert!(class.contains(b'z'));
        assert!(!class.contains(b'0'));
    }

    #[test]
    fn negated_ascii_property() {
        let class = class_of(r"\P{L}");
        assert!(!class.contains(b'A'));
        assert!(class.contains(b'0'));
    }

    #[test]
    fn unknown_property_is_an_error() {
        let output = parse(r"[\p{NotACategory}]");
        assert!(output.errors.is_empty());
        let atom = &output.ast.branches[0].pieces[0].atom;
        let class = match atom {
            codevar_negex_parse::Atom::Class(class) => class,
            _ => panic!("expected a class atom"),
        };
        assert!(matches!(
            byte_class(class),
            Err(NegexError::UnknownProperty { .. })
        ));
    }

    #[test]
    fn scalar_above_a_byte_is_rejected() {
        let output = parse("[\u{0100}]");
        assert!(output.errors.is_empty());
        let atom = &output.ast.branches[0].pieces[0].atom;
        let class = match atom {
            codevar_negex_parse::Atom::Class(class) => class,
            _ => panic!("expected a class atom"),
        };
        assert!(matches!(
            byte_class(class),
            Err(NegexError::ClassOutOfByteRange { .. })
        ));
    }

    #[test]
    fn backwards_range_is_rejected() {
        let class = Class::Positive {
            items: alloc::vec![ClassItem::Range { start: 'z', end: 'a' }],
            negated: false,
        };
        assert!(matches!(
            byte_class(&class),
            Err(NegexError::InvalidClassRange { .. })
        ));
    }
}
