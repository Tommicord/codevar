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

//! Confusable-identifier detection.
//!
//! Two identifiers that differ only by a Cyrillic or Greek lookalike of an
//! ASCII letter are indistinguishable in review — the classic trick for
//! hiding a variable swap. Each declared name is folded through a small
//! homoglyph table; a name that changes is reported with its ASCII
//! skeleton, the same idea as Unicode's `confusables.txt` restricted to
//! the characters that actually appear in identifier attacks.

use alloc::string::String;

/// Homoglyph pairs: the source character and the ASCII letter it mimics.
const CONFUSABLES: &[(char, char)] = &[
    // Cyrillic lowercase.
    ('а', 'a'),
    ('в', 'b'),
    ('е', 'e'),
    ('ё', 'e'),
    ('і', 'i'),
    ('ї', 'i'),
    ('ј', 'j'),
    ('о', 'o'),
    ('р', 'p'),
    ('с', 'c'),
    ('т', 't'),
    ('у', 'y'),
    ('х', 'x'),
    ('ѕ', 's'),
    ('һ', 'h'),
    // Cyrillic uppercase.
    ('А', 'A'),
    ('В', 'B'),
    ('Е', 'E'),
    ('Ё', 'E'),
    ('І', 'I'),
    ('Ј', 'J'),
    ('К', 'K'),
    ('М', 'M'),
    ('Н', 'H'),
    ('О', 'O'),
    ('Р', 'P'),
    ('С', 'C'),
    ('Т', 'T'),
    ('У', 'Y'),
    ('Х', 'X'),
    ('Ѕ', 'S'),
    // Greek lowercase.
    ('α', 'a'),
    ('β', 'b'),
    ('ε', 'e'),
    ('η', 'n'),
    ('γ', 'y'),
    ('ι', 'i'),
    ('κ', 'k'),
    ('μ', 'm'),
    ('ν', 'v'),
    ('ο', 'o'),
    ('ρ', 'p'),
    ('τ', 't'),
    ('χ', 'x'),
    // Greek uppercase.
    ('Α', 'A'),
    ('Β', 'B'),
    ('Ε', 'E'),
    ('Ζ', 'Z'),
    ('Η', 'H'),
    ('Ι', 'I'),
    ('Κ', 'K'),
    ('Μ', 'M'),
    ('Ν', 'N'),
    ('Ο', 'O'),
    ('Ρ', 'P'),
    ('Τ', 'T'),
    ('Υ', 'Y'),
    ('Χ', 'X'),
    // Latin lookalikes.
    ('ı', 'i'),
    ('ɑ', 'a'),
    ('ɡ', 'g'),
    ('ɩ', 'i'),
    ('ϲ', 'c'),
    ('ⅰ', 'i'),
];

/// Folds `name` through the homoglyph table.
///
/// Returns `Some(ascii_skeleton)` when the name contains at least one
/// lookalike character, and `None` when it is already plain ASCII.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::confusable_skeleton;
///
/// assert_eq!(confusable_skeleton("count"), None);
/// // "а" is Cyrillic U+0430, not Latin "a".
/// assert_eq!(confusable_skeleton("аbc"), Some(String::from("abc")));
/// ```
#[must_use]
pub fn confusable_skeleton(name: &str) -> Option<String> {
    let mut skeleton = String::with_capacity(name.len());
    let mut folded = false;
    for character in name.chars() {
        match CONFUSABLES
            .iter()
            .find(|(source, _)| *source == character)
        {
            Some((_, ascii)) => {
                skeleton.push(*ascii);
                folded = true;
            }
            None => skeleton.push(character),
        }
    }
    if folded { Some(skeleton) } else { None }
}
