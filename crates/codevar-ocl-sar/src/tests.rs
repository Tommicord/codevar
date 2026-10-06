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

//! Unit tests for the semantic analyzer and its gates.
//!
//! Every analysis test drives the public [`crate::analyze`] entry point so
//! the gating contract — Unicode, then lexical, then parse, then
//! semantics — is what is actually exercised. Helper assertions keep the
//! individual cases short: [`clean`] for programs that must not error,
//! [`error_in`] and [`warning_in`] for a specific diagnostic code.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::{
    AnalysisOutput, Builtin, BuiltinKind, BuiltinType, ColorChoice, DeclKind, MessageBuilder, Scalar, Ty,
    analyze, analyze_bytes, builtins, codes, coerce, confusable_skeleton, lookup_builtin, lookup_builtin_fn,
    render, substitute, unify,
};

/// Analyzes `source`, asserting no error was reported (warnings are fine).
fn clean(source: &str) -> AnalysisOutput {
    let output = analyze(source);
    assert!(
        !output.has_errors(),
        "expected no errors in {source:?}, found {:#?}",
        output.diagnostics
    );
    output
}

/// Analyzes `source`, asserting `code` was reported and is an error.
fn error_in(source: &str, code: &str) -> AnalysisOutput {
    let output = analyze(source);
    let found = output.find_code(code);
    assert!(
        found.is_some(),
        "expected error {code} in {source:?}, found {:#?}",
        output.diagnostics
    );
    assert!(
        found.is_some_and(crate::Diagnostic::is_error),
        "{code} should be an error severity"
    );
    output
}

/// Analyzes `source`, asserting `code` was reported and is a warning.
fn warning_in(source: &str, code: &str) -> AnalysisOutput {
    let output = analyze(source);
    let found = output.find_code(code);
    assert!(
        found.is_some(),
        "expected warning {code} in {source:?}, found {:#?}",
        output.diagnostics
    );
    assert!(
        found.is_some_and(crate::Diagnostic::is_warning),
        "{code} should be a warning severity"
    );
    output
}

// ---------------------------------------------------------------------------
// Unicode gate
// ---------------------------------------------------------------------------

#[test]
fn bidi_control_is_rejected() {
    let output = error_in("fn f() {}\u{202E}", codes::BIDI_CONTROL);
    assert!(output.declarations.is_empty());
}

#[test]
fn bidi_control_inside_comment_is_rejected() {
    error_in("// hide\u{202E}\nfn f() {}", codes::BIDI_CONTROL);
}

#[test]
fn invisible_character_is_rejected() {
    error_in("\u{200B}fn f() {}", codes::INVISIBLE_CHARACTER);
}

#[test]
fn noncharacter_is_rejected() {
    error_in("fn f() {}\u{FFFE}", codes::NONCHARACTER);
}

#[test]
fn disallowed_control_character_is_rejected() {
    error_in("fn f() {}\u{0007}", codes::CONTROL_CHARACTER);
}

#[test]
fn allowed_whitespace_is_accepted() {
    clean("fn f() {\n\tlet _x = 1;\r\n}");
}

// ---------------------------------------------------------------------------
// Lexical gate
// ---------------------------------------------------------------------------

#[test]
fn unknown_character_is_rejected() {
    let output = error_in("fn f() { № }", codes::UNKNOWN_CHARACTER);
    assert!(output.declarations.is_empty());
}

#[test]
fn unterminated_string_is_rejected() {
    error_in("fn f() { \"abc }", codes::UNTERMINATED_STRING);
}

#[test]
fn integer_out_of_range_for_suffix_is_rejected() {
    error_in("fn f() { let _x = 300u8; }", codes::LITERAL_OUT_OF_RANGE);
}

#[test]
fn integer_beyond_every_type_is_rejected() {
    error_in(
        "fn f() { let _x = 10000000000000000000000000000000000000000000000000000000; }",
        codes::LITERAL_OUT_OF_RANGE,
    );
}

#[test]
fn long_char_literal_is_rejected() {
    error_in("fn f() { let _c = 'ab'; }", codes::CHAR_LITERAL_TOO_LONG);
}

#[test]
fn bad_float_suffix_is_rejected() {
    error_in("fn f() { let _x = 1.0x; }", codes::INVALID_LITERAL_SUFFIX);
}

#[test]
fn unknown_escape_is_rejected() {
    error_in("fn f() { let _s = \"a\\q\"; }", codes::UNKNOWN_ESCAPE_SEQUENCE);
}

#[test]
fn lexical_errors_suppress_semantic_analysis() {
    let output = error_in("fn f() { № let x: int = true; }", codes::UNKNOWN_CHARACTER);
    assert!(
        output
            .find_code(codes::MISMATCHED_TYPES)
            .is_none()
    );
    assert!(output.declarations.is_empty());
}

// ---------------------------------------------------------------------------
// Parse gate
// ---------------------------------------------------------------------------

#[test]
fn syntax_errors_suppress_semantic_analysis() {
    let output = analyze("fn f( { }");
    assert!(output.has_errors());
    assert!(
        output
            .find_code(codes::MISMATCHED_TYPES)
            .is_none()
    );
    assert!(output.declarations.is_empty());
}

// ---------------------------------------------------------------------------
// Type and expression checking
// ---------------------------------------------------------------------------

#[test]
fn function_return_mismatch_is_reported() {
    let output = error_in("fn f() -> int { true }", codes::MISMATCHED_TYPES);
    assert_eq!(output.error_count(), 1);
    assert_eq!(output.warning_count(), 0);
}

#[test]
fn binding_annotation_mismatch_is_reported() {
    error_in("fn f() { let x: int = true; }", codes::MISMATCHED_TYPES);
}

#[test]
fn float_literal_in_int_position_is_reported() {
    error_in("fn f() { let x: int = 1.0; }", codes::MISMATCHED_TYPES);
}

#[test]
fn invalid_vector_lanes_are_reported() {
    error_in("fn f() { let v: float9 = 1.0; }", codes::INVALID_VECTOR_LANES);
}

#[test]
fn non_constant_array_length_is_reported() {
    error_in("fn f() { let xs: [int; true]; }", codes::INVALID_ARRAY_LENGTH);
}

#[test]
fn uninferable_array_literal_is_reported() {
    error_in("fn f() { let xs = []; }", codes::MISSING_TYPE_ANNOTATION);
}

#[test]
fn binding_without_type_or_value_is_reported() {
    error_in("fn f() { let x; }", codes::MISSING_TYPE_ANNOTATION);
}

#[test]
fn string_literal_is_reported() {
    error_in("fn f() { let s = \"hi\"; }", codes::UNSUPPORTED_STRING);
}

#[test]
fn bad_cast_is_reported() {
    error_in("fn f() { let x = 1 as bool; }", codes::INVALID_CAST);
}

#[test]
fn non_boolean_condition_is_reported() {
    let output = error_in("fn f() { if 1 {} }", codes::INVALID_CONDITION);
    let diagnostic = output.find_code(codes::INVALID_CONDITION);
    let help = diagnostic
        .and_then(|found| found.help.as_deref())
        .unwrap_or("");
    assert!(help.contains("x != 0"), "unexpected help: {help}");
}

#[test]
fn bad_binary_operands_are_reported() {
    error_in("fn f() { let x = true + 1; }", codes::INVALID_BINARY_OPERAND);
}

#[test]
fn bad_unary_operand_is_reported() {
    error_in("fn f() { let x = -true; }", codes::INVALID_UNARY_OPERAND);
}

#[test]
fn assignment_to_literal_is_reported() {
    error_in("fn f() { 1 = 2; }", codes::INVALID_ASSIGNMENT_TARGET);
}

#[test]
fn assignment_to_immutable_binding_is_reported() {
    error_in("fn f(x: int) { x = 1; }", codes::ASSIGN_TO_IMMUTABLE);
}

#[test]
fn mutable_parameter_can_be_assigned() {
    clean("fn f(mut x: int) -> int { x = 2; x }");
}

#[test]
fn non_integer_index_is_reported() {
    error_in(
        "fn f() { let a = [1, 2]; let b = a[true]; }",
        codes::INVALID_INDEX,
    );
}

#[test]
fn unknown_field_suggests_the_real_one() {
    let output = error_in(
        "struct S { val: int }\nfn f(s: S) -> int { s.vla }",
        codes::UNKNOWN_FIELD,
    );
    let help = output
        .find_code(codes::UNKNOWN_FIELD)
        .and_then(|found| found.help.as_deref())
        .unwrap_or("");
    assert!(help.contains("val"), "unexpected help: {help}");
}

#[test]
fn range_outside_for_is_reported() {
    error_in("fn f() { let r = 0..1; }", codes::RANGE_NOT_ALLOWED);
}

#[test]
fn for_over_non_iterable_is_reported() {
    error_in("fn f() { for x in 5 {} }", codes::NOT_ITERABLE);
}

#[test]
fn void_parameter_is_reported() {
    error_in("fn f(x: void) {}", codes::VOID_PARAM);
}

#[test]
fn void_inside_array_is_reported() {
    error_in("fn f() { let xs: [void; 4]; }", codes::INVALID_VOID);
}

#[test]
fn unknown_type_suggests_the_builtin() {
    let output = error_in("fn f() { let x: intr; }", codes::UNKNOWN_TYPE);
    let help = output
        .find_code(codes::UNKNOWN_TYPE)
        .and_then(|found| found.help.as_deref())
        .unwrap_or("");
    assert!(help.contains("int"), "unexpected help: {help}");
}

#[test]
fn function_name_in_type_position_is_reported() {
    error_in("fn g() {}\nfn f() { let x: g; }", codes::NOT_A_TYPE);
}

// ---------------------------------------------------------------------------
// Name resolution
// ---------------------------------------------------------------------------

#[test]
fn multi_segment_path_is_reported() {
    error_in("fn f() { let x = std::max; }", codes::MULTI_SEGMENT_PATH);
}

#[test]
fn function_used_as_value_is_reported() {
    error_in("fn g() {}\nfn f() { let x = g; }", codes::UNEXPECTED_FUNCTION);
}

#[test]
fn type_used_as_value_is_reported() {
    error_in("fn f() { let x = int; }", codes::UNEXPECTED_TYPE);
}

#[test]
fn undefined_name_suggests_a_binding() {
    let output = error_in("fn f() { let value = 1; let y = valu; }", codes::UNDEFINED_NAME);
    let help = output
        .find_code(codes::UNDEFINED_NAME)
        .and_then(|found| found.help.as_deref())
        .unwrap_or("");
    assert!(help.contains("value"), "unexpected help: {help}");
}

#[test]
fn wrong_argument_count_is_reported() {
    error_in(
        "fn g(a: int) {}\nfn f() { g(); }",
        codes::WRONG_NUMBER_OF_ARGUMENTS,
    );
}

#[test]
fn calling_a_non_function_is_reported() {
    error_in("fn f() { let x = 1; x(); }", codes::EXPECTED_FUNCTION);
}

#[test]
fn break_outside_loop_is_reported() {
    error_in("fn f() { break; }", codes::BREAK_OUTSIDE_LOOP);
}

#[test]
fn continue_outside_loop_is_reported() {
    error_in("fn f() { continue; }", codes::CONTINUE_OUTSIDE_LOOP);
}

#[test]
fn try_operator_is_reported() {
    error_in("fn f(a: int) -> int { a? }", codes::TRY_NOT_SUPPORTED);
}

// ---------------------------------------------------------------------------
// Items, attributes, cycles
// ---------------------------------------------------------------------------

#[test]
fn keyword_used_as_name_is_reported() {
    error_in("fn if() {}", codes::KEYWORD_AS_NAME);
}

#[test]
fn raw_identifier_escapes_the_keyword_check() {
    clean("fn r#if() {}");
}

#[test]
fn duplicate_function_is_reported() {
    error_in("fn f() {}\nfn f() {}", codes::DUPLICATE_DEFINITION);
}

#[test]
fn self_referential_alias_is_reported() {
    error_in("type A = A;", codes::ALIAS_CYCLE);
}

#[test]
fn mutual_alias_cycle_is_reported() {
    error_in("type A = B;\ntype B = A;", codes::ALIAS_CYCLE);
}

#[test]
fn self_referential_struct_is_reported() {
    error_in("struct S { next: S }", codes::RECURSIVE_TYPE);
}

#[test]
fn pointer_recursion_is_allowed() {
    clean("struct Node { next: *mut Node }\nfn f(n: Node) -> Node { n }");
}

#[test]
fn wrong_generic_arity_is_reported() {
    error_in(
        "struct Pair<T> {}\nfn f() { let x: Pair<int, int>; }",
        codes::GENERIC_ARITY,
    );
}

#[test]
fn unknown_attribute_is_reported() {
    error_in("#[gpu]\nfn f() {}", codes::UNKNOWN_ATTRIBUTE);
}

#[test]
fn attribute_on_struct_is_reported() {
    error_in("#[kernel]\nstruct S {}", codes::ATTRIBUTE_TARGET);
}

#[test]
fn duplicate_attribute_is_reported() {
    error_in("#[kernel]\n#[kernel]\nfn f() {}", codes::DUPLICATE_ATTRIBUTE);
}

// ---------------------------------------------------------------------------
// Kernel rules
// ---------------------------------------------------------------------------

#[test]
fn kernel_must_return_void() {
    error_in("#[kernel]\nfn k() -> int { 0 }", codes::KERNEL_RETURN);
}

#[test]
fn kernel_reference_parameter_is_rejected() {
    error_in("#[kernel]\nfn k(x: &int) {}", codes::KERNEL_PARAM);
}

#[test]
fn kernel_slice_parameter_is_rejected() {
    error_in("#[kernel]\nfn k(x: [int]) {}", codes::KERNEL_PARAM);
}

#[test]
fn kernel_pointer_parameter_is_accepted() {
    clean("#[kernel]\nfn k(p: *mut float, n: int) {}");
}

#[test]
fn kernel_cannot_be_called_from_another_function() {
    error_in("#[kernel]\nfn k() {}\nfn caller() { k(); }", codes::KERNEL_CALL);
}

#[test]
fn direct_recursion_is_reported() {
    error_in("fn a() { a(); }", codes::RECURSION);
}

#[test]
fn mutual_recursion_is_reported() {
    error_in("fn a() { b(); }\nfn b() { a(); }", codes::RECURSION);
}

// ---------------------------------------------------------------------------
// Lints
// ---------------------------------------------------------------------------

#[test]
fn unused_local_is_reported() {
    let output = warning_in("fn f() { let x = 1; }", codes::UNUSED_VARIABLE);
    assert!(!output.has_errors());
    assert_eq!(output.error_count(), 0);
    assert!(output.warning_count() >= 2);
}

#[test]
fn underscore_local_is_not_reported() {
    let output = analyze("fn f() { let _x = 1; }");
    assert!(output.find_code(codes::UNUSED_VARIABLE).is_none());
    assert!(!output.has_errors());
}

#[test]
fn read_local_is_not_reported() {
    let output = analyze("fn f() -> int { let x = 1; x }");
    assert!(output.find_code(codes::UNUSED_VARIABLE).is_none());
}

#[test]
fn parameters_are_never_reported_as_unused() {
    let output = analyze("fn f(a: int) {}");
    assert!(output.find_code(codes::UNUSED_VARIABLE).is_none());
}

#[test]
fn unused_function_is_reported() {
    let output = analyze("fn used() {}\nfn unused() {}\nfn caller() { used(); }");
    let first = output.find_code(codes::UNUSED_ITEM);
    assert!(first.is_some(), "expected {:#?}", output.diagnostics);
    let message = first
        .map(|found| found.message.as_str())
        .unwrap_or("");
    assert!(message.contains("unused"), "unexpected message: {message}");
    assert!(!output.has_errors());
}

#[test]
fn kernel_is_exempt_from_the_unused_item_lint() {
    let output = analyze("#[kernel]\nfn k() {}");
    assert!(output.find_code(codes::UNUSED_ITEM).is_none());
    assert!(!output.has_errors());
}

#[test]
fn reserved_opencl_name_is_reported() {
    let output = warning_in(
        "fn f() -> int { let kernel = 1; kernel }",
        codes::RESERVED_IDENTIFIER,
    );
    assert!(!output.has_errors(), "{:#?}", output.diagnostics);
}

#[test]
fn confusable_identifier_is_reported() {
    let output = warning_in("fn f() -> int { let аbc = 1; аbc }", codes::CONFUSABLE_IDENTIFIER);
    let help = output
        .find_code(codes::CONFUSABLE_IDENTIFIER)
        .and_then(|found| found.help.as_deref())
        .unwrap_or("");
    assert!(help.contains("abc"), "unexpected help: {help}");
    assert!(!output.has_errors(), "{:#?}", output.diagnostics);
}

#[test]
fn lints_are_suppressed_when_errors_exist() {
    let output = analyze("fn f() { let x = 1; break; }");
    assert!(output.find_code(codes::UNUSED_VARIABLE).is_none());
    assert!(output.find_code(codes::UNUSED_ITEM).is_none());
    assert!(output.has_errors());
}

// ---------------------------------------------------------------------------
// Diagnostics ordering and counting
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_are_sorted_by_source_position() {
    let output = analyze("fn a() { break; }\nfn b() { break; }");
    assert_eq!(output.error_count(), 2);
    let first = output.diagnostics.first();
    let second = output.diagnostics.get(1);
    let offsets = (first.map(|d| d.span.offset), second.map(|d| d.span.offset));
    if let (Some(first_offset), Some(second_offset)) = offsets {
        assert!(first_offset < second_offset);
    } else {
        panic!("expected two diagnostics");
    }
}

#[test]
fn warnings_do_not_set_has_errors() {
    let output = analyze("fn f() { let x = 1; }");
    assert!(!output.has_errors());
    assert_eq!(output.error_count(), 0);
    assert!(output.warning_count() > 0);
}

// ---------------------------------------------------------------------------
// Built-ins
// ---------------------------------------------------------------------------

#[test]
fn sqrt_checks_by_shape() {
    clean("fn f(x: float) -> float { sqrt(x) }");
}

#[test]
fn work_item_query_takes_one_dimension() {
    clean("fn f() -> size_t { get_global_id(0) }");
}

#[test]
fn work_item_query_wrong_arity_is_reported() {
    error_in(
        "fn f() -> size_t { get_global_id() }",
        codes::WRONG_NUMBER_OF_ARGUMENTS,
    );
}

#[test]
fn work_item_dimension_must_be_integer() {
    error_in(
        "fn f() -> size_t { get_global_id(true) }",
        codes::MISMATCHED_TYPES,
    );
}

#[test]
fn reduction_builtin_collapses_to_bool() {
    clean("fn f(v: bool4) -> bool { all(v) }");
}

#[test]
fn reduction_builtin_rejects_non_boolean() {
    error_in("fn f(v: float4) -> bool { all(v) }", codes::MISMATCHED_TYPES);
}

#[test]
fn barrier_arity_is_checked() {
    error_in("fn f() { barrier(); }", codes::WRONG_NUMBER_OF_ARGUMENTS);
}

#[test]
fn integer_builtin_keeps_its_shape() {
    clean("fn f() -> int { abs(-5) }");
}

// ---------------------------------------------------------------------------
// Well-formed programs
// ---------------------------------------------------------------------------

#[test]
fn simple_function_is_accepted() {
    clean("fn add(a: int, b: int) -> int { a + b }");
}

#[test]
fn branches_unify_to_one_type() {
    clean("fn f(a: bool) -> int { if a { 1 } else { 2 } }");
}

#[test]
fn range_loop_is_accepted() {
    clean("fn f() { for i in 0..4 { i; } }");
}

#[test]
fn while_loop_over_mutable_counter_is_accepted() {
    clean("fn f() { let mut i = 0; while i < 3 { i = i + 1; } }");
}

#[test]
fn declarations_are_reported_in_source_order() {
    let source = "\
struct Point { x: float, y: float }
type Coord = Point;
fn scale(p: Point) -> Point { p }
#[kernel]
fn k(c: *mut Coord) { }";
    let output = clean(source);
    assert_eq!(output.declarations.len(), 4);
    let names: Vec<&str> = output
        .declarations
        .iter()
        .map(|declaration| declaration.name.as_str())
        .collect();
    assert_eq!(names, ["Point", "Coord", "scale", "k"]);

    let point = Ty::Struct {
        name: String::from("Point"),
        args: Vec::new(),
    };
    match &output.declarations[0].kind {
        DeclKind::Struct { fields } => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0], (String::from("x"), Ty::Scalar(Scalar::F32)));
            assert_eq!(fields[1], (String::from("y"), Ty::Scalar(Scalar::F32)));
        }
        other => panic!("expected a struct, found {other:?}"),
    }
    match &output.declarations[1].kind {
        DeclKind::Alias { target } => assert_eq!(target, &point),
        other => panic!("expected an alias, found {other:?}"),
    }
    match &output.declarations[2].kind {
        DeclKind::Function { params, ret, kernel } => {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], (String::from("p"), point.clone()));
            assert_eq!(ret, &point);
            assert!(!kernel);
        }
        other => panic!("expected a function, found {other:?}"),
    }
    match &output.declarations[3].kind {
        DeclKind::Function { params, ret, kernel } => {
            assert_eq!(params.len(), 1);
            assert_eq!(
                params[0],
                (
                    String::from("c"),
                    Ty::Ptr {
                        mutable: true,
                        inner: Box::new(point),
                    },
                )
            );
            assert_eq!(ret, &Ty::void());
            assert!(kernel);
        }
        other => panic!("expected a kernel, found {other:?}"),
    }
}

#[test]
fn unit_struct_declares_no_fields() {
    let output = clean("struct Empty;\nfn f(e: Empty) {}");
    let structurally = output
        .declarations
        .iter()
        .find(|declaration| declaration.name == "Empty");
    match structurally.map(|declaration| &declaration.kind) {
        Some(DeclKind::Struct { fields }) => assert!(fields.is_empty()),
        other => panic!("expected a unit struct, found {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Byte input
// ---------------------------------------------------------------------------

#[test]
fn valid_utf8_bytes_are_analyzed() {
    let output = analyze_bytes(b"fn f() {}");
    assert!(!output.has_errors());
}

#[test]
fn invalid_utf8_bytes_are_rejected() {
    let mut bytes = Vec::from(&b"fn f() {}"[..]);
    bytes.push(0xFF);
    let output = analyze_bytes(&bytes);
    assert!(output.find_code(codes::INVALID_UTF8).is_some());
    assert!(output.declarations.is_empty());
}

// ---------------------------------------------------------------------------
// Type-system API
// ---------------------------------------------------------------------------

#[test]
fn literal_coercion_follows_the_rules() {
    let int = Ty::Scalar(Scalar::I32);
    let float = Ty::Scalar(Scalar::F32);
    assert!(coerce(&Ty::IntLit(1), &int));
    assert!(coerce(&Ty::IntLit(1), &float));
    assert!(!coerce(&Ty::FloatLit(Some(1.0)), &int));
    assert!(!coerce(&Ty::bool(), &int));
    assert!(coerce(&Ty::IntLit(7), &Ty::vector(Scalar::I32, 4)));
}

#[test]
fn pointer_coercion_is_one_way() {
    let int = Ty::Scalar(Scalar::I32);
    let mut_ptr = Ty::Ptr {
        mutable: true,
        inner: Box::new(int.clone()),
    };
    let const_ptr = Ty::Ptr {
        mutable: false,
        inner: Box::new(int),
    };
    assert!(coerce(&mut_ptr, &const_ptr));
    assert!(!coerce(&const_ptr, &mut_ptr));
}

#[test]
fn unify_records_generic_substitutions() {
    let mut substitutions = Vec::new();
    let generic = Ty::Generic(String::from("T"));
    let actual = Ty::Scalar(Scalar::I64);
    assert!(unify(&generic, &actual, &mut substitutions));
    assert_eq!(substitutions, alloc::vec![(String::from("T"), actual.clone())]);
    assert_eq!(
        substitute(&generic, &[String::from("T")], &[actual.clone()]),
        actual
    );
}

#[test]
fn builtin_type_table_resolves_lanes() {
    assert_eq!(
        lookup_builtin("int"),
        Some(BuiltinType::Type(Ty::Scalar(Scalar::I32)))
    );
    assert_eq!(
        lookup_builtin("float3"),
        Some(BuiltinType::Type(Ty::Vector {
            elem: Scalar::F32,
            lanes: 3,
        }))
    );
    assert!(matches!(lookup_builtin("int9"), Some(BuiltinType::BadLanes(_))));
    assert_eq!(lookup_builtin("Widget"), None);
    assert_eq!(lookup_builtin("void"), Some(BuiltinType::Type(Ty::void())));
}

#[test]
fn digit_suffixed_scalar_names_resolve_as_scalars() {
    assert_eq!(
        lookup_builtin("i32"),
        Some(BuiltinType::Type(Ty::Scalar(Scalar::I32)))
    );
    assert_eq!(
        lookup_builtin("u8"),
        Some(BuiltinType::Type(Ty::Scalar(Scalar::U8)))
    );
    assert_eq!(
        lookup_builtin("f64"),
        Some(BuiltinType::Type(Ty::Scalar(Scalar::F64)))
    );
    clean("fn f(x: i32) -> i32 { x }");
}

#[test]
fn builtin_function_table_is_consistent() {
    assert!(builtins().len() >= 10);
    assert!(matches!(
        lookup_builtin_fn("sqrt"),
        Some(Builtin {
            kind: BuiltinKind::Float { arity: 1 },
            ..
        })
    ));
    assert!(matches!(
        lookup_builtin_fn("all"),
        Some(Builtin {
            kind: BuiltinKind::Reduce { arity: 1 },
            ..
        })
    ));
    assert!(matches!(
        lookup_builtin_fn("get_global_id"),
        Some(Builtin {
            kind: BuiltinKind::WorkItem,
            ..
        })
    ));
    assert!(lookup_builtin_fn("printf").is_none());
}

#[test]
fn confusable_skeleton_folds_homoglyphs() {
    assert_eq!(confusable_skeleton("аbc"), Some(String::from("abc")));
    assert_eq!(confusable_skeleton("count"), None);
}

#[test]
fn message_builder_mixes_prose_and_types() {
    let message = MessageBuilder::new()
        .text("expected ")
        .ty(&Ty::Scalar(Scalar::I32))
        .text(", found ")
        .ty(&Ty::bool())
        .finish();
    assert_eq!(message, "expected `int`, found `bool`");
}

#[test]
fn types_render_in_canonical_spelling() {
    assert_eq!(Ty::vector(Scalar::F32, 4).to_string(), "float4");
    assert_eq!(Ty::Scalar(Scalar::I8).to_string(), "char");
    assert_eq!(
        Ty::Ptr {
            mutable: true,
            inner: Box::new(Ty::Scalar(Scalar::I32)),
        }
        .to_string(),
        "*mut int"
    );
    assert_eq!(
        Ty::Array {
            elem: Box::new(Ty::Scalar(Scalar::I32)),
            len: Some(4),
        }
        .to_string(),
        "[int; 4]"
    );
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

#[test]
fn render_without_color_is_rustc_shaped() {
    let source = "fn f() -> int { true }";
    let output = analyze(source);
    let text = render(&output.diagnostics, "kernel.cl", source, ColorChoice::Never);
    assert!(text.contains("error[E0308]"), "rendered: {text}");
    assert!(text.contains("--> kernel.cl:1:"), "rendered: {text}");
    assert!(text.contains("expected `int`, found `bool`"), "rendered: {text}");
    assert!(text.contains('=') && text.contains("help:"), "rendered: {text}");
    assert!(!text.contains('\u{1b}'), "rendered: {text}");
}

#[test]
fn render_carries_warnings_with_their_code() {
    let source = "fn f() { let x = 1; }";
    let output = analyze(source);
    let text = render(&output.diagnostics, "kernel.cl", source, ColorChoice::Never);
    assert!(text.contains("warning[W0421]"), "rendered: {text}");
    assert!(text.contains("unused variable"), "rendered: {text}");
    assert!(!text.contains('\u{1b}'));
}

#[test]
fn render_always_emits_ansi_color() {
    let source = "fn f() -> int { true }";
    let output = analyze(source);
    let text = render(&output.diagnostics, "kernel.cl", source, ColorChoice::Always);
    assert!(text.contains('\u{1b}'));
}

#[test]
fn render_of_nothing_is_empty() {
    let text = render(&[], "kernel.cl", "fn f() {}", ColorChoice::Never);
    assert!(text.is_empty());
}

#[test]
fn render_handles_syntax_errors_without_codes() {
    let source = "fn f( { }";
    let output = analyze(source);
    let text = render(&output.diagnostics, "kernel.cl", source, ColorChoice::Never);
    assert!(text.contains("error:"), "rendered: {text}");
    assert!(!text.contains("error[E"), "rendered: {text}");
}
