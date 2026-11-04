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

//! Equivalence tests for the streaming, chunked analysis API.
//!
//! Every test drives the streaming driver sequence — feed each item to
//! a [`DeclCollector`], `finish`, [`analyze_body`] per function item,
//! then [`file_checks`] — and asserts the result equals the whole-file
//! [`analyze`] run on the same source: same declarations, and the same
//! diagnostics once both sides are sorted by span.

use alloc::vec::Vec;

use codevar_ocl_parse::{Item, ItemStream};

use crate::{DeclCollector, Declaration, Diagnostic, analyze, analyze_body, codes, file_checks};

/// Runs the streaming pipeline over `source`, returning the
/// concatenated diagnostics (sorted) and the declaration list.
fn stream(source: &str) -> (Vec<Diagnostic>, Vec<Declaration>) {
    let mut collector = DeclCollector::new();
    let mut parser = ItemStream::new(source);
    while let Some(outcome) = parser.next_item() {
        assert!(
            outcome.errors.is_empty(),
            "unexpected parse errors for {source:?}: {:?}",
            outcome.errors
        );
        collector.feed(source, &outcome.item);
    }
    let (mut env, mut diagnostics) = collector.finish();
    let mut parser = ItemStream::new(source);
    while let Some(outcome) = parser.next_item() {
        let (body_diagnostics, _tables) = analyze_body(&mut env, source, &outcome.item);
        diagnostics.extend(body_diagnostics);
    }
    diagnostics.extend(file_checks(&env));
    crate::sort_diagnostics(&mut diagnostics);
    (diagnostics, env.declarations().to_vec())
}

/// Asserts the streaming pipeline matches [`analyze`] on `source`.
fn assert_streaming_equivalence(source: &str) {
    let expected = analyze(source);
    let (diagnostics, declarations) = stream(source);
    assert_eq!(
        declarations, expected.declarations,
        "declarations differ for {source:?}"
    );
    assert_eq!(
        diagnostics, expected.diagnostics,
        "diagnostics differ for {source:?}"
    );
}

#[test]
fn kernel_with_body_error_matches_analyze() {
    assert_streaming_equivalence("#[kernel] fn k(out: *mut int) { *out = true; }");
}

#[test]
fn call_cycle_between_two_functions_matches_analyze() {
    assert_streaming_equivalence("fn a() { b(); } fn b() { a(); }");
}

#[test]
fn unused_function_matches_analyze() {
    assert_streaming_equivalence("fn lonely() {}");
    assert_streaming_equivalence("#[kernel] fn k() {} fn lonely() {}");
}

#[test]
fn struct_and_alias_used_in_signature_match_analyze() {
    assert_streaming_equivalence(
        "struct Point { x: int, y: int } \
         type Coord = int; \
         fn first_x(p: Point) -> Coord { p.x }",
    );
    assert_streaming_equivalence(
        "struct Point { x: int, y: int } \
         type Coord = int; \
         #[kernel] fn scale(p: Point, factor: Coord) { let out = p.x * factor; }",
    );
}

#[test]
fn body_error_suppresses_unused_warnings_like_analyze() {
    assert_streaming_equivalence("#[kernel] fn k() { let x: int = true; } fn lonely() {}");
}

#[test]
fn duplicate_function_checks_only_the_last_body() {
    assert_streaming_equivalence("fn dup() { let a = 1; } fn dup() -> int { let b: int = true; b }");
}

#[test]
fn is_kernel_reports_kernel_declarations() {
    let source = "#[kernel] fn k() {} fn helper() {} fn main_() { helper(); }";
    let mut collector = DeclCollector::new();
    let mut parser = ItemStream::new(source);
    while let Some(outcome) = parser.next_item() {
        collector.feed(source, &outcome.item);
    }
    let (env, diagnostics) = collector.finish();
    assert!(diagnostics.is_empty(), "{:?}", diagnostics);
    assert!(env.is_kernel("k"));
    assert!(!env.is_kernel("helper"));
    assert!(!env.is_kernel("missing"));
}

#[test]
fn analyze_body_ignores_non_function_items() {
    let source = "struct S { x: int }";
    let mut collector = DeclCollector::new();
    let mut parser = ItemStream::new(source);
    let mut items = Vec::new();
    while let Some(outcome) = parser.next_item() {
        collector.feed(source, &outcome.item);
        items.push(outcome.item);
    }
    let (mut env, diagnostics) = collector.finish();
    assert!(diagnostics.is_empty(), "{:?}", diagnostics);
    for item in &items {
        let (body_diagnostics, tables) = analyze_body(&mut env, source, item);
        assert!(body_diagnostics.is_empty());
        assert!(tables.types.is_empty());
        assert!(tables.resolutions.is_empty());
    }
}

#[test]
fn streaming_tables_cover_the_checked_body() {
    let source = "fn f(x: int) -> int { x + 1 }";
    let mut collector = DeclCollector::new();
    let mut parser = ItemStream::new(source);
    let mut function_item: Option<Item> = None;
    while let Some(outcome) = parser.next_item() {
        collector.feed(source, &outcome.item);
        function_item = Some(outcome.item);
    }
    let (mut env, _) = collector.finish();
    let Some(item) = function_item else {
        panic!("expected one item");
    };
    let (diagnostics, tables) = analyze_body(&mut env, source, &item);
    assert!(diagnostics.is_empty(), "{:?}", diagnostics);
    assert!(!tables.types.is_empty(), "body expressions must be typed");
}

#[test]
fn kitchen_sink_kernel_matches_analyze() {
    let kernel = "struct Params { n: int, scale: float } \
                   type Count = int; \
                   fn helper(xs: *mut int, i: Count) -> int { xs[i] } \
                   #[kernel] \
                   fn k(p: *const Params, out: *mut float, buf: *mut int) { \
                       let mut idx: Count = 0; \
                       while idx < (*p).n { \
                           let v: float = (helper(buf, idx) as float) * (*p).scale; \
                           out[idx] = v; \
                           idx += 1; \
                       } \
                   }";
    assert!(
        !analyze(kernel).has_errors(),
        "kitchen-sink kernel must be well formed: {:#?}",
        analyze(kernel).diagnostics
    );
    assert_streaming_equivalence(kernel);
    assert_streaming_equivalence(
        "fn identity<T>(x: T) -> T { x } \
         #[kernel] fn k(v: int) { let _u = identity(v); }",
    );
}

#[test]
fn streaming_reports_the_same_codes_as_analyze() {
    let source = "#[kernel] fn k(out: *mut int) { *out = true; } \
                  fn a() { b(); } fn b() { a(); } \
                  struct Bad { next: Bad } \
                  fn lonely() {}";
    let expected = analyze(source);
    assert!(
        expected
            .find_code(codes::MISMATCHED_TYPES)
            .is_some()
    );
    assert!(expected.find_code(codes::RECURSION).is_some());
    assert!(
        expected
            .find_code(codes::RECURSIVE_TYPE)
            .is_some()
    );
    let (diagnostics, _) = stream(source);
    for code in [codes::MISMATCHED_TYPES, codes::RECURSION, codes::RECURSIVE_TYPE] {
        let found = diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == Some(code));
        assert!(found, "streaming run missed {code} in {source:?}");
    }
}
