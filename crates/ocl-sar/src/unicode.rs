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

//! Unicode validation of the source text.
//!
//! The lexer accepts every character it cannot classify as an unknown
//! token, which is the right recovery behavior but not a security policy:
//! bidirectional controls can reorder how a program renders, invisible
//! characters can hide code, and noncharacters have no business in source.
//! This pass rejects them before parsing, following rustc's
//! `text_direction_codepoint_in_literal` and Clang's `-Wunicode` families.

use crate::diagnostic::{Diagnostic, codes};
use alloc::format;
use alloc::vec::Vec;
use codevar_ocl_parse::Span;

/// Reports every disallowed code point in `source`, in source order.
pub(crate) fn check(source: &str, diags: &mut Vec<Diagnostic>) {
    for (offset, character) in source.char_indices() {
        let code_point = character as u32;
        let span = Span::new(offset as u32, character.len_utf8() as u32);
        let diagnostic = if is_bidi_control(code_point) {
            Diagnostic::error(span, format!("bidirectional control character {}", display(code_point)))
                .with_code(codes::BIDI_CONTROL)
                .with_label(span, "this character can reorder displayed source")
                .with_note("bidirectional controls let text render in a different order than it is written; remove the character")
        } else if is_invisible(code_point) {
            Diagnostic::error(span, format!("invisible character {}", display(code_point)))
                .with_code(codes::INVISIBLE_CHARACTER)
                .with_label(span, "this character has no visual representation")
        } else if is_noncharacter(code_point) {
            Diagnostic::error(span, format!("noncharacter code point {}", display(code_point)))
                .with_code(codes::NONCHARACTER)
                .with_label(
                    span,
                    "Unicode reserves this code point and it is never a character",
                )
        } else if is_disallowed_control(character) {
            Diagnostic::error(span, format!("control character {}", display(code_point)))
                .with_code(codes::CONTROL_CHARACTER)
                .with_label(
                    span,
                    "only tab, newline, carriage return, and vertical tab are allowed",
                )
        } else {
            continue;
        };
        diags.push(diagnostic);
    }
}

/// Formats a code point as `U+XXXX`.
fn display(code_point: u32) -> alloc::string::String {
    alloc::format!("U+{code_point:04X}")
}

/// True for the Unicode bidirectional control code points.
const fn is_bidi_control(code_point: u32) -> bool {
    matches!(code_point, 0x061C | 0x200E | 0x200F | 0x202A..=0x202E | 0x2066..=0x2069)
}

/// True for code points that render as nothing.
const fn is_invisible(code_point: u32) -> bool {
    matches!(code_point, 0x200B..=0x200D | 0x2060..=0x2064 | 0xFEFF)
}

/// True for the Unicode noncharacter code points: `U+FDD0..U+FDEF` and
/// the `U+xxFFFE`/`U+xxFFFF` pair of every plane.
fn is_noncharacter(code_point: u32) -> bool {
    (0xFDD0..=0xFDEF).contains(&code_point) || (code_point & 0xFFFE) == 0xFFFE
}

/// True for control characters outside the whitespace the dialect allows.
fn is_disallowed_control(character: char) -> bool {
    character.is_control()
        && !matches!(
            character,
            '\u{0009}' | '\u{000A}' | '\u{000B}' | '\u{000C}' | '\u{000D}' | '\u{0085}'
        )
}
