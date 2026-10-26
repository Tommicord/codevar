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

//! Grammar, precedence, recovery, and losslessness tests for the parser.
//!
//! `unwrap`/`expect`/`panic!` appear only on inputs this file constructs
//! itself (known-good AST shapes), which is the sanctioned unit-test
//! exception.

use crate::ast::{ExprKind, ItemKind, PatKind, TypeKind, UnaryOp};
use crate::token_tree::TokenTreeKind;
use crate::{TokenTree, build_token_trees, parse};
use alloc::format;
use alloc::vec::Vec;

/// Returns the first function item of a parsed program.
fn first_fn(program: &crate::ast::Program) -> &crate::ast::FnItem {
    match &program
        .items
        .first()
        .expect("at least one item")
        .kind
    {
        ItemKind::Fn(func) => func,
        other => panic!("expected function item, found {other:?}"),
    }
}

/// Returns a block's tail expression.
fn tail(block: &crate::ast::Block) -> &crate::ast::Expr {
    block.tail.as_deref().expect("tail expression")
}

/// Collects every byte-owning span depth-first in source order.
///
/// Group delimiters own their own spans, so the open delimiter is pushed
/// before the children and the close delimiter after them.
fn leaf_spans(trees: &[TokenTree], out: &mut Vec<crate::Span>) {
    for tree in trees {
        match &tree.kind {
            TokenTreeKind::Leaf { .. } => out.push(tree.span),
            TokenTreeKind::Group {
                open,
                children,
                close,
                ..
            } => {
                out.push(*open);
                leaf_spans(children, out);
                if let Some(close) = close {
                    out.push(*close);
                }
            }
        }
    }
}

#[test]
fn parses_minimal_function() {
    let output = parse("fn main() {}");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    assert_eq!(func.name, "main");
    assert!(func.params.is_empty());
    assert!(func.ret.is_none());
    assert!(func.body.stmts.is_empty());
    assert!(func.body.tail.is_none());
}

#[test]
fn parses_params_return_and_tail() {
    let output = parse("fn add(a: i32, b: i32) -> i32 { a + b }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    assert_eq!(func.params.len(), 2);
    assert!(matches!(
        func.params[0].pat.kind,
        PatKind::Ident { ref name, mutable: false } if name == "a"
    ));
    assert!(matches!(
        func.params[0].ty.kind,
        TypeKind::Path(ref path) if path.segments[0].name == "i32"
    ));
    let ret = func.ret.as_ref().expect("return type");
    assert!(matches!(
        ret.kind,
        TypeKind::Path(ref path) if path.segments[0].name == "i32"
    ));
    let tail = tail(&func.body);
    assert!(matches!(
        tail.kind,
        ExprKind::Binary {
            op: crate::BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn multiplication_binds_tighter_than_addition() {
    let output = parse("fn f() { let x = 1 + 2 * 3; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let stmt = func.body.stmts.first().expect("let statement");
    let crate::StmtKind::Let(binding) = &stmt.kind else {
        panic!("expected let statement");
    };
    let init = binding.init.as_ref().expect("initializer");
    match &init.kind {
        ExprKind::Binary { op, lhs, rhs } => {
            assert_eq!(*op, crate::BinaryOp::Add);
            assert!(matches!(lhs.kind, ExprKind::Literal { .. }));
            assert!(matches!(
                rhs.kind,
                ExprKind::Binary {
                    op: crate::BinaryOp::Mul,
                    ..
                }
            ));
        }
        other => panic!("expected binary expression, found {other:?}"),
    }
}

#[test]
fn unary_negation_binds_tighter_than_cast() {
    let output = parse("fn f() { let x = -a as i32; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Let(binding) = &func.body.stmts[0].kind else {
        panic!("expected let statement");
    };
    let init = binding.init.as_ref().expect("initializer");
    match &init.kind {
        ExprKind::Cast { expr, .. } => assert!(matches!(expr.kind, ExprKind::Unary { op: UnaryOp::Neg, .. })),
        other => panic!("expected cast, found {other:?}"),
    }
}

#[test]
fn assignment_is_right_associative() {
    let output = parse("fn f() { a = b = c; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Expr(expr) = &func.body.stmts[0].kind else {
        panic!("expected expression statement");
    };
    match &expr.kind {
        ExprKind::Assign { op, lhs, rhs } => {
            assert!(op.is_none());
            assert!(matches!(lhs.kind, ExprKind::Path(_)));
            assert!(matches!(rhs.kind, ExprKind::Assign { .. }));
        }
        other => panic!("expected assignment, found {other:?}"),
    }
}

#[test]
fn compound_assignment_records_operator() {
    let output = parse("fn f() { x += 1; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Expr(expr) = &func.body.stmts[0].kind else {
        panic!("expected expression statement");
    };
    assert!(matches!(
        expr.kind,
        ExprKind::Assign {
            op: Some(crate::BinaryOp::Add),
            ..
        }
    ));
}

#[test]
fn logical_operators_precede_correctly() {
    let output = parse("fn f() { a || b && c }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let expr = tail(&func.body);
    match &expr.kind {
        ExprKind::Binary { op, rhs, .. } => {
            assert_eq!(*op, crate::BinaryOp::Or);
            assert!(matches!(
                rhs.kind,
                ExprKind::Binary {
                    op: crate::BinaryOp::And,
                    ..
                }
            ));
        }
        other => panic!("expected logical or, found {other:?}"),
    }
}

#[test]
fn comparison_chains_report_non_associativity() {
    let output = parse("fn f() { if a < b < c {} }");
    assert!(
        output
            .errors
            .iter()
            .any(|error| error.message.contains("non-associative")),
        "{:?}",
        output.errors
    );
    assert_eq!(output.program.items.len(), 1);
}

#[test]
fn postfix_chain_composes_call_index_field_try() {
    let output = parse("fn f() { f(a)[0].b?; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Expr(expr) = &func.body.stmts[0].kind else {
        panic!("expected expression statement");
    };
    let mut kind = &expr.kind;
    for _ in 0..3 {
        kind = match kind {
            ExprKind::Try { expr } => &expr.kind,
            ExprKind::Field { expr, .. } => &expr.kind,
            ExprKind::Index { expr, .. } => &expr.kind,
            ExprKind::Call { .. } => break,
            other => panic!("unexpected postfix shape: {other:?}"),
        };
    }
    assert!(matches!(kind, ExprKind::Call { .. }));
}

#[test]
fn for_range_stops_before_loop_body() {
    let output = parse("fn f() { for i in 0..n { g(i); } }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let expr = tail(&func.body);
    let ExprKind::For { iter, body, .. } = &expr.kind else {
        panic!("expected for loop");
    };
    match &iter.kind {
        ExprKind::Range {
            start,
            end,
            inclusive,
        } => {
            assert!(!inclusive);
            assert!(matches!(
                start.as_deref().expect("range start").kind,
                ExprKind::Literal { .. }
            ));
            assert!(matches!(
                end.as_deref().expect("range end").kind,
                ExprKind::Path(_)
            ));
        }
        other => panic!("expected range, found {other:?}"),
    }
    assert_eq!(body.stmts.len(), 1);
    assert!(body.tail.is_none());
}

#[test]
fn if_else_chain_nests() {
    let output = parse("fn f() { if a { } else if b { } else { } }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let expr = tail(&func.body);
    let ExprKind::If { else_branch, .. } = &expr.kind else {
        panic!("expected if expression");
    };
    let else_if = else_branch.as_deref().expect("else branch");
    let ExprKind::If {
        else_branch: final_branch,
        ..
    } = &else_if.kind
    else {
        panic!("expected else-if branch");
    };
    assert!(matches!(
        final_branch
            .as_deref()
            .expect("final branch")
            .kind,
        ExprKind::Block(_)
    ));
}

#[test]
fn let_with_type_and_initializer() {
    let output = parse("fn f() { let x: i32 = 42; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Let(binding) = &func.body.stmts[0].kind else {
        panic!("expected let statement");
    };
    assert!(binding.ty.is_some());
    assert!(matches!(
        binding.init.as_ref().expect("initializer").kind,
        ExprKind::Literal { .. }
    ));
}

#[test]
fn let_without_semicolon_reports_error() {
    let output = parse("fn f() { let x = 1 }");
    assert!(
        output
            .errors
            .iter()
            .any(|error| error.message.contains(';')),
        "{:?}",
        output.errors
    );
    assert_eq!(output.program.items.len(), 1);
}

#[test]
fn nested_generic_closers_never_glue() {
    let output = parse("type A = Vec<Vec<i32>>;");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let ItemKind::TypeAlias(alias) = &output.program.items[0].kind else {
        panic!("expected type alias");
    };
    let TypeKind::Path(outer) = &alias.aliased.kind else {
        panic!("expected path type");
    };
    assert_eq!(outer.segments[0].name, "Vec");
    let crate::GenericArg::Type(inner) = &outer.args[0] else {
        panic!("expected type argument");
    };
    let TypeKind::Path(inner_path) = &inner.kind else {
        panic!("expected path type");
    };
    assert_eq!(inner_path.segments[0].name, "Vec");
    assert_eq!(inner_path.args.len(), 1);
}

#[test]
fn reference_types_carry_lifetime_and_mutability() {
    let output = parse("fn f(x: &'a mut [i32]) {}");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let TypeKind::Ref {
        lifetime,
        mutable,
        inner,
    } = &func.params[0].ty.kind
    else {
        panic!("expected reference type");
    };
    assert_eq!(lifetime.as_deref(), Some("a"));
    assert!(*mutable);
    assert!(matches!(inner.kind, TypeKind::Slice(_)));
}

#[test]
fn array_type_carries_length_expression() {
    let output = parse("type Buf = [f32; 4];");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let ItemKind::TypeAlias(alias) = &output.program.items[0].kind else {
        panic!("expected type alias");
    };
    let TypeKind::Array { elem, len } = &alias.aliased.kind else {
        panic!("expected array type");
    };
    assert!(matches!(elem.kind, TypeKind::Path(_)));
    assert!(matches!(len.kind, ExprKind::Literal { .. }));
}

#[test]
fn structs_with_fields_and_unit_form() {
    let output = parse("struct S { a: i32, } struct U;");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    assert_eq!(output.program.items.len(), 2);
    let ItemKind::Struct(first) = &output.program.items[0].kind else {
        panic!("expected struct");
    };
    assert_eq!(first.fields.len(), 1);
    assert_eq!(first.fields[0].name, "a");
    assert!(!first.unit);
    let ItemKind::Struct(second) = &output.program.items[1].kind else {
        panic!("expected struct");
    };
    assert!(second.unit);
    assert!(second.fields.is_empty());
}

#[test]
fn attributes_are_opaque_and_accepted() {
    let source = "#[kernel]\nfn f() {}";
    let output = parse(source);
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    assert_eq!(output.program.items[0].attrs.len(), 1);
    let attr = &output.program.items[0].attrs[0];
    let end = attr.span.end() as usize;
    assert_eq!(&source[attr.span.offset as usize..end], "#[kernel]");
}

#[test]
fn qualified_paths_keep_segments() {
    let output = parse("fn f() { a::b::c; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Expr(expr) = &func.body.stmts[0].kind else {
        panic!("expected expression statement");
    };
    let ExprKind::Path(path) = &expr.kind else {
        panic!("expected path expression");
    };
    let names: Vec<&str> = path
        .segments
        .iter()
        .map(|segment| segment.name.as_str())
        .collect();
    assert_eq!(names, ["a", "b", "c"]);
}

#[test]
fn return_break_continue_without_values() {
    let output = parse("fn f() { loop { break; } return; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    assert_eq!(func.body.stmts.len(), 2);
    let crate::StmtKind::Expr(loop_expr) = &func.body.stmts[0].kind else {
        panic!("expected loop statement");
    };
    assert!(matches!(loop_expr.kind, ExprKind::Loop { .. }));
    let crate::StmtKind::Expr(return_expr) = &func.body.stmts[1].kind else {
        panic!("expected return statement");
    };
    assert!(matches!(return_expr.kind, ExprKind::Return { expr: None }));
}

#[test]
fn tuple_field_access_parses() {
    let output = parse("fn f() { x.0; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Expr(expr) = &func.body.stmts[0].kind else {
        panic!("expected expression statement");
    };
    match &expr.kind {
        ExprKind::Field { name, .. } => assert_eq!(name, "0"),
        other => panic!("expected field access, found {other:?}"),
    }
}

#[test]
fn recovery_reaches_later_items() {
    let output = parse("fn a( { } fn b() {}");
    assert!(!output.errors.is_empty());
    assert_eq!(output.program.items.len(), 2);
    let ItemKind::Fn(second) = &output.program.items[1].kind else {
        panic!("expected second function");
    };
    assert_eq!(second.name, "b");
}

#[test]
fn recursion_limit_terminates_deep_nesting() {
    let source = format!("fn f() {{ {}x }}", "(".repeat(400));
    let output = parse(&source);
    assert!(
        output
            .errors
            .iter()
            .any(|error| error.message.contains("recursion limit")),
        "{:?}",
        output.errors
    );
    assert_eq!(output.program.items.len(), 1);
}

#[test]
fn empty_source_parses_cleanly() {
    let output = parse("");
    assert!(output.errors.is_empty());
    assert!(output.program.items.is_empty());
    assert!(output.trees.is_empty());
}

#[test]
fn token_trees_cover_every_byte() {
    let source = "a + /*c*/ (b)\n#[x]";
    let (trees, errors) = build_token_trees(source);
    assert!(errors.is_empty(), "{errors:?}");
    let mut spans = Vec::new();
    leaf_spans(&trees, &mut spans);
    assert!(!spans.is_empty());
    assert_eq!(spans[0].offset, 0);
    for pair in spans.windows(2) {
        assert_eq!(pair[0].end(), pair[1].offset, "gap between leaves");
    }
    assert_eq!(spans.last().expect("last leaf").end() as usize, source.len());
}

#[test]
fn unclosed_group_is_reported_and_recovered() {
    let (trees, errors) = build_token_trees("(a");
    assert!(!errors.is_empty());
    assert!(errors[0].message.contains("unclosed"), "{errors:?}");
    let TokenTreeKind::Group { close, children, .. } = &trees[0].kind else {
        panic!("expected group tree");
    };
    assert!(close.is_none());
    assert_eq!(children.len(), 1);
}

#[test]
fn mismatched_closer_is_reported() {
    let (_, errors) = build_token_trees("(]");
    assert!(!errors.is_empty());
    assert!(errors[0].message.contains("expected"), "{errors:?}");
}

#[test]
fn literal_text_is_preserved_verbatim() {
    let output = parse("fn f() { let x = 0xFFu8; }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let func = first_fn(&output.program);
    let crate::StmtKind::Let(binding) = &func.body.stmts[0].kind else {
        panic!("expected let statement");
    };
    let init = binding.init.as_ref().expect("initializer");
    let ExprKind::Literal { text, .. } = &init.kind else {
        panic!("expected literal");
    };
    assert_eq!(text, "0xFFu8");
}

#[derive(Default)]
struct IdCollector {
    ids: Vec<crate::NodeId>,
}

impl crate::Visitor for IdCollector {
    fn visit_expr(&mut self, expr: &crate::Expr) {
        self.ids.push(expr.id);
        crate::visit::walk_expr(self, expr);
    }
    fn visit_pat(&mut self, pat: &crate::Pat) {
        self.ids.push(pat.id);
        crate::visit::walk_pat(self, pat);
    }
}

#[test]
fn parser_mints_unique_non_dummy_node_ids() {
    let output = parse("fn f(a: int) -> int { let b = a + 1; b }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let mut collector = IdCollector::default();
    crate::walk_program(&mut collector, &output.program);
    assert!(collector.ids.len() >= 6, "collected {:?}", collector.ids);
    assert!(
        collector
            .ids
            .iter()
            .all(|id| *id != crate::NodeId::DUMMY)
    );
    let mut sorted = collector.ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), collector.ids.len(), "ids must be unique");
}

#[test]
fn expr_equality_ignores_node_ids() {
    let output = parse("fn f() -> int { 1 }");
    let func = first_fn(&output.program);
    let tail: &crate::Expr = func.body.tail.as_ref().expect("tail expression");
    let same_shape = crate::Expr {
        id: crate::NodeId::from_raw(9999),
        kind: tail.kind.clone(),
        span: tail.span,
    };
    assert_eq!(*tail, same_shape);
    let shifted = crate::Expr {
        id: tail.id,
        kind: tail.kind.clone(),
        span: crate::Span::new(0, 0),
    };
    assert_ne!(*tail, shifted);
}

#[test]
fn folding_reuses_replaced_node_ids() {
    let mut output = parse("fn f() -> int { (1 + 2) + 3 }");
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let mut before = IdCollector::default();
    crate::walk_program(&mut before, &output.program);
    let report = crate::optimize(&mut output.program);
    assert!(report.constant_folds >= 1, "{report:?}");
    let mut after = IdCollector::default();
    crate::walk_program(&mut after, &output.program);
    for id in &after.ids {
        assert!(before.ids.contains(id), "optimizer minted a new id: {id:?}");
    }
}
