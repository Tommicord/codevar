//! Copyright 2026 Codevar
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

//! Tests for Pattern

use codevar_core::edit::regexp::Pattern;
use codevar_core::edit::regexp::regexp_pattern::{
    FLAG_CASE_INSENSITIVE, FLAG_COMMENTS, FLAG_DOTALL, FLAG_MULTILINE, FLAG_UNICODE_CASE,
};

#[test]
fn regexp_pattern_creation() {
    let pattern = Pattern::compile("\\d+").unwrap();
    assert_eq!(pattern.flags(), 0);
}

#[test]
fn regexp_pattern_with_flags_creation() {
    let pattern = Pattern::compile_with_flags(
        "[a-zA-Z0-9]+",
        FLAG_CASE_INSENSITIVE | FLAG_UNICODE_CASE,
    )
    .unwrap();
    assert_eq!(pattern.flags(), FLAG_CASE_INSENSITIVE | FLAG_UNICODE_CASE);
}

#[test]
fn regexp_pattern_set_matches() {
    let pattern = Pattern::compile_with_flags(
        "^[a-zA-Z0-9]+$",
        FLAG_CASE_INSENSITIVE | FLAG_UNICODE_CASE,
    )
    .unwrap();
    let matches = pattern.matches("AABBCC11");
    assert_eq!(matches, true);
}

#[test]
fn regexp_pattern_endswith_text() {
    let pattern = Pattern::compile_with_flags(" world$", FLAG_CASE_INSENSITIVE).unwrap();
    let matches = pattern.matches("Hello World");
    assert_eq!(matches, true);
}

#[test]
fn digit_class() {
    let pattern = Pattern::compile("\\d+").unwrap();
    assert!(pattern.matches("12345"));
    assert!(!pattern.matches("abc"));
    assert!(pattern.matches("0"));
}

#[test]
fn non_digit_class() {
    let pattern = Pattern::compile("\\D+").unwrap();
    assert!(!pattern.matches("12345"));
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("a"));
}

#[test]
fn whitespace_class() {
    let pattern = Pattern::compile("\\s+").unwrap();
    assert!(pattern.matches("   "));
    assert!(pattern.matches("\t\n"));
    assert!(!pattern.matches("abc"));
}

#[test]
fn non_whitespace_class() {
    let pattern = Pattern::compile("\\S+").unwrap();
    assert!(!pattern.matches("   "));
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("a"));
}

#[test]
fn word_class() {
    let pattern = Pattern::compile("\\w+").unwrap();
    assert!(pattern.matches("abc123"));
    assert!(pattern.matches("_test"));
    assert!(!pattern.matches("test!"));
}

#[test]
fn non_word_class() {
    let pattern = Pattern::compile("\\W+").unwrap();
    assert!(!pattern.matches("abc123"));
    assert!(pattern.matches("!@#"));
    assert!(pattern.matches(" "));
}

#[test]
fn star_quantifier() {
    let pattern = Pattern::compile("a*").unwrap();
    assert!(pattern.matches(""));
    assert!(pattern.matches("a"));
    assert!(pattern.matches("aaa"));
    assert!(!pattern.matches("b"));
}

#[test]
fn plus_quantifier() {
    let pattern = Pattern::compile("a+").unwrap();
    assert!(!pattern.matches(""));
    assert!(pattern.matches("a"));
    assert!(pattern.matches("aaa"));
    assert!(!pattern.matches("b"));
}

#[test]
fn question_quantifier() {
    let pattern = Pattern::compile("a?").unwrap();
    assert!(pattern.matches(""));
    assert!(pattern.matches("a"));
    assert!(!pattern.matches("aa"));
    assert!(!pattern.matches("b"));
}

#[test]
fn exact_count_quantifier() {
    let pattern = Pattern::compile("a{3}").unwrap();
    assert!(!pattern.matches(""));
    assert!(!pattern.matches("a"));
    assert!(!pattern.matches("aa"));
    assert!(pattern.matches("aaa"));
    assert!(!pattern.matches("aaaa"));
}

#[test]
fn range_quantifier() {
    let pattern = Pattern::compile("a{2,4}").unwrap();
    assert!(!pattern.matches("a"));
    assert!(pattern.matches("aa"));
    assert!(pattern.matches("aaa"));
    assert!(pattern.matches("aaaa"));
    assert!(!pattern.matches("aaaaa"));
}

#[test]
fn min_only_quantifier() {
    let pattern = Pattern::compile("a{2,}").unwrap();
    assert!(!pattern.matches("a"));
    assert!(pattern.matches("aa"));
    assert!(pattern.matches("aaa"));
    assert!(pattern.matches("aaaaa"));
}

#[test]
fn start_anchor() {
    let pattern = Pattern::compile("^abc").unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("abc123"));
    assert!(!pattern.matches("123abc"));
}

#[test]
fn end_anchor() {
    let pattern = Pattern::compile("abc$").unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("123abc"));
    assert!(!pattern.matches("abc123"));
}

#[test]
fn both_anchors() {
    let pattern = Pattern::compile("^abc$").unwrap();
    assert!(pattern.matches("abc"));
    assert!(!pattern.matches("abc123"));
    assert!(!pattern.matches("123abc"));
}

#[test]
fn multiline_anchor() {
    let pattern = Pattern::compile_with_flags("^abc", FLAG_MULTILINE).unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("abc\n123"));
}

#[test]
fn case_insensitive() {
    let pattern = Pattern::compile_with_flags("abc", FLAG_CASE_INSENSITIVE).unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("ABC"));
    assert!(pattern.matches("AbC"));
}

#[test]
fn dot_normal() {
    let pattern = Pattern::compile("a.c").unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("a$c"));
    assert!(!pattern.matches("a\nc"));
    assert!(!pattern.matches("ac"));
}

#[test]
fn dot_dotall() {
    let pattern = Pattern::compile_with_flags("a.c", FLAG_DOTALL).unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("a\nc"));
    assert!(pattern.matches("a\rc"));
    assert!(!pattern.matches("ac"));
}

#[test]
fn capturing_group() {
    let pattern = Pattern::compile("(abc)").unwrap();
    assert!(pattern.matches("abc"));
    assert_eq!(pattern.group_count(), 2); // Group 0 + Group 1
}

#[test]
fn non_capturing_group() {
    let pattern = Pattern::compile("(?:abc)").unwrap();
    assert!(pattern.matches("abc"));
    assert_eq!(pattern.group_count(), 1); // Only Group 0
}

#[test]
fn alternation() {
    let pattern = Pattern::compile("abc|def").unwrap();
    assert!(pattern.matches("abc"));
    assert!(pattern.matches("def"));
    assert!(!pattern.matches("ghi"));
}

#[test]
fn escape_tab() {
    let pattern = Pattern::compile("\\t").unwrap();
    assert!(pattern.matches("\t"));
}

#[test]
fn escape_newline() {
    let pattern = Pattern::compile("\\n").unwrap();
    assert!(pattern.matches("\n"));
}

#[test]
fn escape_carriage_return() {
    let pattern = Pattern::compile("\\r").unwrap();
    assert!(pattern.matches("\r"));
}

#[test]
fn escape_formfeed() {
    let pattern = Pattern::compile("\\f").unwrap();
    assert!(pattern.matches("\x0c"));
}

#[test]
fn escape_bell() {
    let pattern = Pattern::compile("\\a").unwrap();
    assert!(pattern.matches("\x07"));
}

#[test]
fn escape_escape() {
    let pattern = Pattern::compile("\\e").unwrap();
    assert!(pattern.matches("\x1b"));
}

// Hex escape tests
#[test]
fn hex_escape() {
    let pattern = Pattern::compile("\\x41").unwrap();
    assert!(pattern.matches("A"));
}

// Unicode escape tests
#[test]
fn unicode_escape() {
    let pattern = Pattern::compile("\\u0041").unwrap();
    assert!(pattern.matches("A"));
}

// Octal escape tests
#[test]
fn octal_escape() {
    let pattern = Pattern::compile("\\101").unwrap();
    assert!(pattern.matches("A"));
}

// Complex pattern tests
#[test]
fn email_pattern() {
    let pattern = Pattern::compile("\\w+@\\w+\\.\\w+").unwrap();
    assert!(pattern.matches("test@example.com"));
    assert!(!pattern.matches("invalid"));
}

#[test]
fn phone_pattern() {
    let pattern = Pattern::compile("\\d{3}-\\d{3}-\\d{4}").unwrap();
    assert!(pattern.matches("123-456-7890"));
    assert!(!pattern.matches("1234567890"));
}

#[test]
fn url_pattern() {
    let pattern = Pattern::compile("https?://\\w+\\.\\w+").unwrap();
    assert!(pattern.matches("http://example.com"));
    assert!(pattern.matches("https://example.com"));
    assert!(!pattern.matches("ftp://example.com"));
}

// Back reference tests
#[test]
fn back_reference() {
    let pattern = Pattern::compile("(\\w+)\\1").unwrap();
    assert!(pattern.matches("testtest"));
    assert!(!pattern.matches("test123"));
}

// Comments mode tests
#[test]
fn comments_mode() {
    let pattern = Pattern::compile_with_flags("a# comment\nb", FLAG_COMMENTS).unwrap();
    assert!(pattern.matches("ab"));
}

// Error handling tests
#[test]
fn unmatched_parenthesis() {
    let result = Pattern::compile("(abc");
    assert!(result.is_err());
}

#[test]
fn invalid_escape() {
    let result = Pattern::compile("\\");
    assert!(result.is_err());
}

#[test]
fn invalid_repeat() {
    let result = Pattern::compile("*");
    assert!(result.is_err());
}
