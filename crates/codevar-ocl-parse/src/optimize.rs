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

//! AST optimization: constant folding, algebraic simplification, and
//! unreachable-code removal.
//!
//! The design mirrors the expression folders of mature C++ compilers:
//!
//! * GCC's [`fold-const.cc`](https://github.com/gcc-mirror/gcc/blob/master/gcc/fold-const.cc)
//!   entry point `fold` simplifies `x * 1 => x`, folds constant subtrees
//!   with `const_binop` / `int_const_binop`, and re-associates integer
//!   chains through `split_tree` + `associate_trees`. This pass reuses
//!   those ideas: [`Optimizer::fold_binary_kind`] walks constant folding,
//!   boolean short-circuit identities, arithmetic identities, and a
//!   bounded re-association of `+`/`-`/`*` chains.
//! * Clang's [`ExprConstant.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/AST/ExprConstant.cpp)
//!   evaluates expressions with an explicit success/failure flag, stops
//!   at side effects it cannot model, and bounds its work with step
//!   limits. Here `fold_const` returns `Option` (failure = "leave the
//!   node alone"), operand dropping is gated on [`is_pure`], and the
//!   whole pass runs for at most [`MAX_PASSES`] fixpoint rounds.
//! * Statement pruning follows what both compilers' dead-code passes do
//!   at the CFG level (`-Wunreachable-code`, tree-ssa-dce): statements
//!   after a direct `return`/`break`/`continue` never execute, so they
//!   are removed.
//!
//! Integer folding uses checked arithmetic: overflow and division by
//! zero abort the fold and leave the original expression in place, the
//! same "no silent wrong results" rule Clang applies when constant
//! evaluation hits undefined behavior. Re-association of integer chains
//! assumes two's-complement wrapping arithmetic, as GCC's `fold` does
//! under `-fwrapv`; revisit when the dialect pins overflow semantics.

use crate::ast::*;
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use codevar_ocl_lex::{Base, LiteralKind, TokenKind};

/// Maximum bottom-up fixpoint rounds over the whole program.
///
/// One round already folds nested constants innermost-first; extra
/// rounds only pick up shapes a fold exposed in a *sibling* statement.
/// The bound mirrors Clang's `ConstexprStepLimit` philosophy: work is
/// capped so pathological input cannot stall the compiler.
const MAX_PASSES: u32 = 0x24;

/// Maximum local re-folds of one expression node after a shape change.
const FOLD_ROUNDS: u32 = 0x16;

/// What one [`optimize`] run changed in the program.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OptReport {
    /// Constant subtrees replaced by literals (GCC `const_binop`).
    pub constant_folds: u32,
    /// Identity and re-association rewrites (`x * 1 => x`, `(x + 1) + 2`).
    pub algebraic_simplifications: u32,
    /// Statements and no-op blocks removed as dead or effect-free.
    pub dead_statements_removed: u32,
    /// `if`/`while` nodes resolved from constant conditions.
    pub branches_folded: u32,
}

impl OptReport {
    /// Total number of rewrites applied.
    #[must_use]
    pub const fn changes(self) -> u32 {
        self.constant_folds
            .saturating_add(self.algebraic_simplifications)
            .saturating_add(self.dead_statements_removed)
            .saturating_add(self.branches_folded)
    }

    /// Adds `other`'s counters into `self`.
    fn merge(&mut self, other: &Self) {
        self.constant_folds = self
            .constant_folds
            .saturating_add(other.constant_folds);
        self.algebraic_simplifications = self
            .algebraic_simplifications
            .saturating_add(other.algebraic_simplifications);
        self.dead_statements_removed = self
            .dead_statements_removed
            .saturating_add(other.dead_statements_removed);
        self.branches_folded = self
            .branches_folded
            .saturating_add(other.branches_folded);
    }
}

/// Optimizes `program` in place and reports how much changed.
///
/// The pass is semantics-preserving for side-effect-free input and never
/// removes code whose effects are unknown: operands are only dropped
/// when [`is_pure`] holds, and unreachable-code removal is limited to
/// statements directly after a terminator in the same block.
///
/// # Examples
///
/// ```
/// let mut output = codevar_ocl_parse::parse("fn f() { let x = 1 + 2 * 3; }");
/// let report = codevar_ocl_parse::optimize(&mut output.program);
/// assert!(report.constant_folds >= 1);
/// ```
#[must_use]
pub fn optimize(program: &mut Program) -> OptReport {
    let mut total = OptReport::default();
    for _ in 0..MAX_PASSES {
        let mut optimizer = Optimizer {
            report: OptReport::default(),
        };
        optimizer.optimize_program(program);
        if optimizer.report.changes() == 0 {
            break;
        }
        total.merge(&optimizer.report);
    }
    total
}

/// A compile-time value recovered from an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstVal {
    /// An integer literal value.
    Int(i64),
    /// A boolean literal value.
    Bool(bool),
}

/// The single optimization walker over one fixpoint round.
struct Optimizer {
    /// Rewrites performed in this round.
    report: OptReport,
}

impl Optimizer {
    /// Optimizes every item of the program.
    fn optimize_program(&mut self, program: &mut Program) {
        for item in &mut program.items {
            match &mut item.kind {
                ItemKind::Fn(function) => {
                    for param in &mut function.params {
                        self.optimize_type(&mut param.ty);
                    }
                    if let Some(ret) = &mut function.ret {
                        self.optimize_type(ret);
                    }
                    self.optimize_block(&mut function.body);
                }
                ItemKind::Struct(strukt) => {
                    for field in &mut strukt.fields {
                        self.optimize_type(&mut field.ty);
                    }
                }
                ItemKind::TypeAlias(alias) => self.optimize_type(&mut alias.aliased),
                ItemKind::Error => {}
            }
        }
    }

    /// Optimizes a type, recursing into nested types and array lengths.
    fn optimize_type(&mut self, ty: &mut Type) {
        match &mut ty.kind {
            TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } | TypeKind::Slice(inner) => {
                self.optimize_type(inner);
            }
            TypeKind::Tuple(items) => {
                for item in items {
                    self.optimize_type(item);
                }
            }
            TypeKind::Array { elem, len } => {
                self.optimize_type(elem);
                self.optimize_expr(len);
            }
            TypeKind::Path(path) => {
                for arg in &mut path.args {
                    if let GenericArg::Type(arg_ty) = arg {
                        self.optimize_type(arg_ty);
                    }
                }
            }
            TypeKind::Error => {}
        }
    }

    /// Optimizes a block's children, then simplifies the statement list.
    fn optimize_block(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.optimize_stmt(stmt);
        }
        if let Some(tail) = &mut block.tail {
            self.optimize_expr(tail);
        }
        self.simplify_block(block);
    }

    /// Optimizes one statement's children.
    fn optimize_stmt(&mut self, stmt: &mut Stmt) {
        match &mut stmt.kind {
            StmtKind::Let(binding) => {
                if let Some(ty) = &mut binding.ty {
                    self.optimize_type(ty);
                }
                if let Some(init) = &mut binding.init {
                    self.optimize_expr(init);
                }
            }
            StmtKind::Expr(expr) => self.optimize_expr(expr),
            StmtKind::Empty | StmtKind::Error => {}
        }
    }

    /// Optimizes an expression bottom-up: children first, then the node
    /// itself is folded until a fixpoint or [`FOLD_ROUNDS`] is reached.
    fn optimize_expr(&mut self, expr: &mut Expr) {
        match &mut expr.kind {
            ExprKind::Literal { .. }
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::Continue
            | ExprKind::Error => {}
            ExprKind::Unary { expr: inner, .. } => self.optimize_expr(inner),
            ExprKind::Binary { lhs, rhs, .. } | ExprKind::Assign { lhs, rhs, .. } => {
                self.optimize_expr(lhs);
                self.optimize_expr(rhs);
            }
            ExprKind::Call { callee, args } => {
                self.optimize_expr(callee);
                for arg in args {
                    self.optimize_expr(arg);
                }
            }
            ExprKind::Index { expr: inner, index } => {
                self.optimize_expr(inner);
                self.optimize_expr(index);
            }
            ExprKind::Field { expr: inner, .. } | ExprKind::Try { expr: inner } => {
                self.optimize_expr(inner);
            }
            ExprKind::Cast { expr: inner, ty } => {
                self.optimize_expr(inner);
                self.optimize_type(ty);
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(start) = start {
                    self.optimize_expr(start);
                }
                if let Some(end) = end {
                    self.optimize_expr(end);
                }
            }
            ExprKind::Block(block) => self.optimize_block(block),
            ExprKind::If {
                cond,
                then,
                else_branch,
            } => {
                self.optimize_expr(cond);
                self.optimize_block(then);
                if let Some(else_branch) = else_branch {
                    self.optimize_expr(else_branch);
                }
            }
            ExprKind::While { cond, body } => {
                self.optimize_expr(cond);
                self.optimize_block(body);
            }
            ExprKind::Loop { body } => self.optimize_block(body),
            ExprKind::For { iter, body, .. } => {
                self.optimize_expr(iter);
                self.optimize_block(body);
            }
            ExprKind::Return { expr: value } | ExprKind::Break { expr: value } => {
                if let Some(value) = value {
                    self.optimize_expr(value);
                }
            }
            ExprKind::Tuple(items) | ExprKind::Array(items) => {
                for item in items {
                    self.optimize_expr(item);
                }
            }
        }
        self.fold_expr(expr);
    }

    /// Applies the fold rules to `expr` until none reports a change.
    fn fold_expr(&mut self, expr: &mut Expr) {
        for _ in 0..FOLD_ROUNDS {
            if !self.fold_once(expr) {
                break;
            }
        }
    }

    /// One local fold attempt; returns whether the node changed.
    fn fold_once(&mut self, expr: &mut Expr) -> bool {
        if self.fold_unary(expr) {
            return true;
        }
        if self.fold_binary(expr) {
            return true;
        }
        if self.fold_cast(expr) {
            return true;
        }
        if self.fold_if(expr) {
            return true;
        }
        self.fold_while(expr)
    }

    /// Folds a unary node in place (`-(-x)`, `!!x`, `-5`, `!true`).
    fn fold_unary(&mut self, expr: &mut Expr) -> bool {
        if !matches!(expr.kind, ExprKind::Unary { .. }) {
            return false;
        }
        let kind = core::mem::replace(&mut expr.kind, ExprKind::Error);
        let (changed, kind) = match kind {
            ExprKind::Unary { op, expr: inner } => self.fold_unary_kind(op, inner),
            other => (false, other),
        };
        expr.kind = kind;
        changed
    }

    /// Fold rules for `op` applied to `inner`.
    fn fold_unary_kind(&mut self, op: UnaryOp, inner: Box<Expr>) -> (bool, ExprKind) {
        match op {
            UnaryOp::Neg => {
                if let Some(value) = int_value(&inner).and_then(i64::checked_neg) {
                    self.bump_constant_folds(1);
                    return (true, const_kind(ConstVal::Int(value)));
                }
                if let ExprKind::Unary {
                    op: UnaryOp::Neg,
                    expr: nested,
                } = inner.kind
                {
                    self.bump_algebraic(1);
                    return (true, into_kind(*nested));
                }
                (false, ExprKind::Unary { op, expr: inner })
            }
            UnaryOp::Not => {
                if let ExprKind::Bool(value) = inner.kind {
                    self.bump_constant_folds(1);
                    return (true, ExprKind::Bool(!value));
                }
                if let ExprKind::Unary {
                    op: UnaryOp::Not,
                    expr: nested,
                } = inner.kind
                {
                    self.bump_algebraic(1);
                    return (true, into_kind(*nested));
                }
                if let Some(value) = int_value(&inner) {
                    self.bump_constant_folds(1);
                    return (true, const_kind(ConstVal::Int(!value)));
                }
                (false, ExprKind::Unary { op, expr: inner })
            }
            UnaryOp::Deref | UnaryOp::AddrOf { .. } => (false, ExprKind::Unary { op, expr: inner }),
        }
    }

    /// Folds a binary node in place.
    fn fold_binary(&mut self, expr: &mut Expr) -> bool {
        if !matches!(expr.kind, ExprKind::Binary { .. }) {
            return false;
        }
        let kind = core::mem::replace(&mut expr.kind, ExprKind::Error);
        let (changed, kind) = match kind {
            ExprKind::Binary { op, lhs, rhs } => self.fold_binary_kind(op, lhs, rhs),
            other => (false, other),
        };
        expr.kind = kind;
        changed
    }

    /// Fold rules for `lhs op rhs`, in the order GCC's `fold` tries them:
    /// constant operands, boolean short-circuit identities, arithmetic
    /// identities, then bounded re-association.
    fn fold_binary_kind(&mut self, op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr>) -> (bool, ExprKind) {
        if let (Some(left), Some(right)) = (const_value(&lhs), const_value(&rhs)) {
            if let Some(value) = fold_const(op, left, right) {
                self.bump_constant_folds(1);
                return (true, const_kind(value));
            }
            return (false, ExprKind::Binary { op, lhs, rhs });
        }

        match op {
            BinaryOp::And => {
                if matches!(&lhs.kind, ExprKind::Bool(true)) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*rhs));
                }
                if matches!(&rhs.kind, ExprKind::Bool(true)) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*lhs));
                }
                if matches!(&lhs.kind, ExprKind::Bool(false)) && is_pure(&rhs) {
                    self.bump_constant_folds(1);
                    return (true, ExprKind::Bool(false));
                }
                if matches!(&rhs.kind, ExprKind::Bool(false)) && is_pure(&lhs) {
                    self.bump_constant_folds(1);
                    return (true, ExprKind::Bool(false));
                }
            }
            BinaryOp::Or => {
                if matches!(&lhs.kind, ExprKind::Bool(false)) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*rhs));
                }
                if matches!(&rhs.kind, ExprKind::Bool(false)) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*lhs));
                }
                if matches!(&lhs.kind, ExprKind::Bool(true)) && is_pure(&rhs) {
                    self.bump_constant_folds(1);
                    return (true, ExprKind::Bool(true));
                }
                if matches!(&rhs.kind, ExprKind::Bool(true)) && is_pure(&lhs) {
                    self.bump_constant_folds(1);
                    return (true, ExprKind::Bool(true));
                }
            }
            _ => {}
        }

        let lhs_int = int_value(&lhs);
        let rhs_int = int_value(&rhs);
        match op {
            BinaryOp::Add | BinaryOp::BitOr | BinaryOp::BitXor => {
                if rhs_int == Some(0) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*lhs));
                }
                if lhs_int == Some(0) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*rhs));
                }
            }
            BinaryOp::Mul => {
                if rhs_int == Some(1) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*lhs));
                }
                if lhs_int == Some(1) {
                    self.bump_algebraic(1);
                    return (true, into_kind(*rhs));
                }
            }
            BinaryOp::Sub if rhs_int == Some(0) => {
                self.bump_algebraic(1);
                return (true, into_kind(*lhs));
            }
            BinaryOp::Div if rhs_int == Some(1) => {
                self.bump_algebraic(1);
                return (true, into_kind(*lhs));
            }
            BinaryOp::Shl | BinaryOp::Shr if rhs_int == Some(0) => {
                self.bump_algebraic(1);
                return (true, into_kind(*lhs));
            }
            _ => {}
        }

        self.reassociate(op, lhs, rhs)
    }

    /// Folds constant parts out of `a (inner) c1 (outer) c2` chains, the
    /// bounded analogue of GCC's `split_tree` + `associate_trees`.
    fn reassociate(&mut self, op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr>) -> (bool, ExprKind) {
        let outer_const = int_value(&rhs);
        if let ExprKind::Binary {
            op: inner,
            lhs: a,
            rhs: b,
        } = lhs.kind
        {
            if let (Some(inner_const), Some(outer_const)) = (int_value(&b), outer_const)
                && let Some((new_op, value)) = assoc_right(inner, op, inner_const, outer_const)
            {
                self.bump_algebraic(1);
                let folded = Expr {
                    kind: const_kind(ConstVal::Int(value)),
                    span: b.span,
                };
                return (
                    true,
                    ExprKind::Binary {
                        op: new_op,
                        lhs: a,
                        rhs: Box::new(folded),
                    },
                );
            }
            return (
                false,
                ExprKind::Binary {
                    op: inner,
                    lhs: a,
                    rhs: b,
                },
            );
        }

        let left_const = int_value(&lhs);
        let rhs_span = rhs.span;
        if let ExprKind::Binary {
            op: inner,
            lhs: a,
            rhs: b,
        } = rhs.kind
        {
            if let (Some(outer_const), Some(inner_const)) = (left_const, int_value(&b))
                && let Some((new_op, value)) = assoc_comm(op, inner, outer_const, inner_const)
            {
                self.bump_algebraic(1);
                let folded = Expr {
                    kind: const_kind(ConstVal::Int(value)),
                    span: b.span,
                };
                return (
                    true,
                    ExprKind::Binary {
                        op: new_op,
                        lhs: a,
                        rhs: Box::new(folded),
                    },
                );
            }
            return (
                false,
                ExprKind::Binary {
                    op,
                    lhs,
                    rhs: Box::new(Expr {
                        kind: ExprKind::Binary {
                            op: inner,
                            lhs: a,
                            rhs: b,
                        },
                        span: rhs_span,
                    }),
                },
            );
        }

        (false, ExprKind::Binary { op, lhs, rhs })
    }

    /// Folds `(x as T) as T` down to `x as T`.
    fn fold_cast(&mut self, expr: &mut Expr) -> bool {
        if !matches!(expr.kind, ExprKind::Cast { .. }) {
            return false;
        }
        let kind = core::mem::replace(&mut expr.kind, ExprKind::Error);
        let (changed, kind) = match kind {
            ExprKind::Cast { expr: inner, ty } => {
                let inner_span = inner.span;
                match inner.kind {
                    ExprKind::Cast {
                        expr: nested,
                        ty: nested_ty,
                    } if same_type(ty.as_ref(), nested_ty.as_ref()) => {
                        self.bump_algebraic(1);
                        (true, ExprKind::Cast { expr: nested, ty })
                    }
                    other => (
                        false,
                        ExprKind::Cast {
                            expr: Box::new(Expr {
                                kind: other,
                                span: inner_span,
                            }),
                            ty,
                        },
                    ),
                }
            }
            other => (false, other),
        };
        expr.kind = kind;
        changed
    }

    /// Resolves `if` expressions with constant conditions.
    fn fold_if(&mut self, expr: &mut Expr) -> bool {
        if !matches!(expr.kind, ExprKind::If { .. }) {
            return false;
        }
        let kind = core::mem::replace(&mut expr.kind, ExprKind::Error);
        let (changed, kind) = match kind {
            ExprKind::If {
                cond,
                then,
                else_branch,
            } => {
                let cond_span = cond.span;
                match cond.kind {
                    ExprKind::Bool(true) => {
                        self.bump_branches(1);
                        (true, ExprKind::Block(then))
                    }
                    ExprKind::Bool(false) => {
                        self.bump_branches(1);
                        match else_branch {
                            Some(else_expr) => (true, into_kind(*else_expr)),
                            None => (
                                true,
                                ExprKind::Block(Block {
                                    stmts: Vec::new(),
                                    tail: None,
                                    span: expr.span,
                                }),
                            ),
                        }
                    }
                    other => (
                        false,
                        ExprKind::If {
                            cond: Box::new(Expr {
                                kind: other,
                                span: cond_span,
                            }),
                            then,
                            else_branch,
                        },
                    ),
                }
            }
            other => (false, other),
        };
        expr.kind = kind;
        changed
    }

    /// Resolves `while false { … }` (the body never runs) to unit.
    fn fold_while(&mut self, expr: &mut Expr) -> bool {
        if !matches!(expr.kind, ExprKind::While { .. }) {
            return false;
        }
        let kind = core::mem::replace(&mut expr.kind, ExprKind::Error);
        let (changed, kind) = match kind {
            ExprKind::While { cond, body } => {
                let cond_span = cond.span;
                match cond.kind {
                    ExprKind::Bool(false) => {
                        self.bump_branches(1);
                        (
                            true,
                            ExprKind::Block(Block {
                                stmts: Vec::new(),
                                tail: None,
                                span: expr.span,
                            }),
                        )
                    }
                    other => (
                        false,
                        ExprKind::While {
                            cond: Box::new(Expr {
                                kind: other,
                                span: cond_span,
                            }),
                            body,
                        },
                    ),
                }
            }
            other => (false, other),
        };
        expr.kind = kind;
        changed
    }

    /// Drops effect-free statements and truncates after a terminator.
    fn simplify_block(&mut self, block: &mut Block) {
        let before = block.stmts.len();
        block.stmts.retain(|stmt| !is_noop_stmt(stmt));
        self.bump_dead(before - block.stmts.len());

        if block.tail.as_deref().is_some_and(is_noop_expr) {
            block.tail = None;
            self.bump_dead(1);
        }

        if let Some(index) = block.stmts.iter().position(is_terminator_stmt) {
            let mut removed = block.stmts.len() - index - 1;
            if block.tail.is_some() {
                removed += 1;
                block.tail = None;
            }
            block.stmts.truncate(index + 1);
            self.bump_dead(removed);
        }
    }

    /// Adds `by` to the constant-fold counter without wrapping.
    fn bump_constant_folds(&mut self, by: u32) {
        self.report.constant_folds = self.report.constant_folds.saturating_add(by);
    }

    /// Adds `by` to the algebraic counter without wrapping.
    fn bump_algebraic(&mut self, by: u32) {
        self.report.algebraic_simplifications = self
            .report
            .algebraic_simplifications
            .saturating_add(by);
    }

    /// Adds `by` to the dead-statement counter without wrapping.
    fn bump_dead(&mut self, by: usize) {
        self.report.dead_statements_removed = self
            .report
            .dead_statements_removed
            .saturating_add(u32::try_from(by).unwrap_or(u32::MAX));
    }

    /// Adds `by` to the branch-fold counter without wrapping.
    fn bump_branches(&mut self, by: u32) {
        self.report.branches_folded = self.report.branches_folded.saturating_add(by);
    }
}

/// Removes an expression's kind, dropping the span.
fn into_kind(expr: Expr) -> ExprKind {
    expr.kind
}

/// Whether the expression can be dropped without changing behavior.
///
/// Mirrors the side-effect check Clang's `ExprConstant` performs before
/// folding through a subexpression: literals, paths, pure operators, and
/// aggregates of pure operands count; calls, assignments, indexing,
/// loops, and control flow (`return`, `break`, `continue`) do not.
fn is_pure(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Literal { .. } | ExprKind::Bool(_) | ExprKind::Path(_) => true,
        ExprKind::Unary { op, expr } => {
            matches!(op, UnaryOp::Neg | UnaryOp::Not | UnaryOp::AddrOf { .. }) && is_pure(expr)
        }
        ExprKind::Binary { lhs, rhs, .. } => is_pure(lhs) && is_pure(rhs),
        ExprKind::Cast { expr, .. } => is_pure(expr),
        ExprKind::Tuple(items) | ExprKind::Array(items) => items.iter().all(is_pure),
        ExprKind::Range { start, end, .. } => {
            start.as_deref().is_none_or(is_pure) && end.as_deref().is_none_or(is_pure)
        }
        _ => false,
    }
}

/// Whether the statement is a direct `return`, `break`, or `continue`.
fn is_terminator_stmt(stmt: &Stmt) -> bool {
    matches!(
        &stmt.kind,
        StmtKind::Expr(expr)
            if matches!(
                expr.kind,
                ExprKind::Return { .. } | ExprKind::Break { .. } | ExprKind::Continue
            )
    )
}

/// Whether the statement has no effect and can be dropped.
fn is_noop_stmt(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Empty => true,
        StmtKind::Expr(expr) => is_noop_expr(expr),
        StmtKind::Let(_) | StmtKind::Error => false,
    }
}

/// Whether the expression evaluates to unit with no side effects.
fn is_noop_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Block(block) => block.stmts.is_empty() && block.tail.is_none(),
        _ => false,
    }
}

/// Structural type equality that ignores spans.
///
/// The derived [`PartialEq`] for [`Type`] compares spans too, so the two
/// `i32`s written in `x as i32 as i32` never compare equal. Cast folding
/// needs value equality, so this walks the type shape instead. Array
/// lengths go through [`same_len`], which is deliberately conservative:
/// anything it cannot prove equal is reported unequal, so an unsupported
/// shape skips the fold rather than mis-folding.
fn same_type(left: &Type, right: &Type) -> bool {
    use TypeKind::*;
    match (&left.kind, &right.kind) {
        (
            Ref {
                lifetime: l_lifetime,
                mutable: l_mut,
                inner: l_inner,
            },
            Ref {
                lifetime: r_lifetime,
                mutable: r_mut,
                inner: r_inner,
            },
        ) => l_lifetime == r_lifetime && l_mut == r_mut && same_type(l_inner, r_inner),
        (
            Ptr {
                mutable: l_mut,
                inner: l_inner,
            },
            Ptr {
                mutable: r_mut,
                inner: r_inner,
            },
        ) => l_mut == r_mut && same_type(l_inner, r_inner),
        (Tuple(l_items), Tuple(r_items)) => {
            l_items.len() == r_items.len()
                && l_items
                    .iter()
                    .zip(r_items)
                    .all(|(l, r)| same_type(l, r))
        }
        (Slice(l_inner), Slice(r_inner)) => same_type(l_inner, r_inner),
        (
            Array {
                elem: l_elem,
                len: l_len,
            },
            Array {
                elem: r_elem,
                len: r_len,
            },
        ) => same_type(l_elem, r_elem) && same_len(l_len, r_len),
        (Path(l_path), Path(r_path)) => same_type_path(l_path, r_path),
        (Error, Error) => true,
        _ => false,
    }
}

/// Span-insensitive equality for qualified type paths.
fn same_type_path(left: &TypePath, right: &TypePath) -> bool {
    left.segments.len() == right.segments.len()
        && left
            .segments
            .iter()
            .zip(&right.segments)
            .all(|(l, r)| l.name == r.name)
        && left.args.len() == right.args.len()
        && left
            .args
            .iter()
            .zip(&right.args)
            .all(|(l, r)| match (l, r) {
                (GenericArg::Type(l_ty), GenericArg::Type(r_ty)) => same_type(l_ty, r_ty),
                (GenericArg::Lifetime { name: l_name, .. }, GenericArg::Lifetime { name: r_name, .. }) => {
                    l_name == r_name
                }
                _ => false,
            })
}

/// Span-insensitive equality for constant array-length expressions.
///
/// Conservative by design: only literal text, booleans, path segment
/// names, and structurally equal operators are compared; any other shape
/// reports `false`, so callers skip the fold instead of mis-folding.
fn same_len(left: &Expr, right: &Expr) -> bool {
    use ExprKind::*;
    match (&left.kind, &right.kind) {
        (Literal { text: l_text, .. }, Literal { text: r_text, .. }) => l_text == r_text,
        (Bool(l), Bool(r)) => l == r,
        (Path(l), Path(r)) => {
            l.segments.len() == r.segments.len()
                && l.segments
                    .iter()
                    .zip(&r.segments)
                    .all(|(l_seg, r_seg)| l_seg.name == r_seg.name)
        }
        (
            Unary {
                op: l_op,
                expr: l_expr,
            },
            Unary {
                op: r_op,
                expr: r_expr,
            },
        ) => l_op == r_op && same_len(l_expr, r_expr),
        (
            Binary {
                op: l_op,
                lhs: l_lhs,
                rhs: l_rhs,
            },
            Binary {
                op: r_op,
                lhs: r_lhs,
                rhs: r_rhs,
            },
        ) => l_op == r_op && same_len(l_lhs, r_lhs) && same_len(l_rhs, r_rhs),
        (
            Cast {
                expr: l_expr,
                ty: l_ty,
            },
            Cast {
                expr: r_expr,
                ty: r_ty,
            },
        ) => same_len(l_expr, r_expr) && same_type(l_ty, r_ty),
        (Tuple(l_items), Tuple(r_items)) | (Array(l_items), Array(r_items)) => {
            l_items.len() == r_items.len()
                && l_items
                    .iter()
                    .zip(r_items)
                    .all(|(l, r)| same_len(l, r))
        }
        _ => false,
    }
}

/// Reads an integer literal's value, honoring base prefixes and the
/// suffix boundary the lexer recorded (`42u8`, `0xFF`, `0b1010`).
fn int_value(expr: &Expr) -> Option<i64> {
    let ExprKind::Literal { text, kind } = &expr.kind else {
        return None;
    };
    let TokenKind::Literal {
        kind: LiteralKind::Int {
            base,
            empty_int: false,
        },
        suffix_start,
    } = kind
    else {
        return None;
    };
    let literal = text.get(..usize::try_from(*suffix_start).ok()?)?;
    let digits = match base {
        Base::Hexadecimal => strip_radix_prefix(literal, 'x', 'X'),
        Base::Binary => strip_radix_prefix(literal, 'b', 'B'),
        Base::Octal => strip_radix_prefix(literal, 'o', 'O'),
        Base::Decimal => literal,
    };
    if digits.is_empty() {
        return None;
    }
    if digits.contains('_') {
        let cleaned: alloc::string::String = digits.chars().filter(|c| *c != '_').collect();
        i64::from_str_radix(&cleaned, *base as u32).ok()
    } else {
        i64::from_str_radix(digits, *base as u32).ok()
    }
}

/// Strips a two-character radix prefix such as `0x`, if present.
fn strip_radix_prefix(literal: &str, lower: char, upper: char) -> &str {
    if literal.len() >= 2 && literal.starts_with('0') {
        let rest = &literal[1..];
        if rest.starts_with(lower) || rest.starts_with(upper) {
            return &rest[1..];
        }
    }
    literal
}

/// Reads any compile-time constant value (integer or boolean).
fn const_value(expr: &Expr) -> Option<ConstVal> {
    match &expr.kind {
        ExprKind::Literal { .. } => int_value(expr).map(ConstVal::Int),
        ExprKind::Bool(value) => Some(ConstVal::Bool(*value)),
        _ => None,
    }
}

/// Evaluates `left op right`, or `None` when the operation cannot be
/// folded safely (division by zero, overflow, mixed types, unsupported).
fn fold_const(op: BinaryOp, left: ConstVal, right: ConstVal) -> Option<ConstVal> {
    use BinaryOp::*;
    match (left, right) {
        (ConstVal::Int(a), ConstVal::Int(b)) => match op {
            Add => a.checked_add(b).map(ConstVal::Int),
            Sub => a.checked_sub(b).map(ConstVal::Int),
            Mul => a.checked_mul(b).map(ConstVal::Int),
            Div => a.checked_div(b).map(ConstVal::Int),
            Rem => a.checked_rem(b).map(ConstVal::Int),
            Shl => u32::try_from(b)
                .ok()
                .and_then(|n| a.checked_shl(n))
                .map(ConstVal::Int),
            Shr => u32::try_from(b)
                .ok()
                .and_then(|n| a.checked_shr(n))
                .map(ConstVal::Int),
            BitAnd => Some(ConstVal::Int(a & b)),
            BitOr => Some(ConstVal::Int(a | b)),
            BitXor => Some(ConstVal::Int(a ^ b)),
            Eq => Some(ConstVal::Bool(a == b)),
            Ne => Some(ConstVal::Bool(a != b)),
            Lt => Some(ConstVal::Bool(a < b)),
            Le => Some(ConstVal::Bool(a <= b)),
            Gt => Some(ConstVal::Bool(a > b)),
            Ge => Some(ConstVal::Bool(a >= b)),
            And | Or => None,
        },
        (ConstVal::Bool(a), ConstVal::Bool(b)) => match op {
            Eq => Some(ConstVal::Bool(a == b)),
            Ne => Some(ConstVal::Bool(a != b)),
            And => Some(ConstVal::Bool(a && b)),
            Or => Some(ConstVal::Bool(a || b)),
            _ => None,
        },
        _ => None,
    }
}

/// Combines `(a inner inner_c) outer outer_c` into `a new_op value`.
fn assoc_right(inner: BinaryOp, outer: BinaryOp, inner_c: i64, outer_c: i64) -> Option<(BinaryOp, i64)> {
    use BinaryOp::*;
    match (inner, outer) {
        (Add, Add) => Some((Add, inner_c.checked_add(outer_c)?)),
        (Add, Sub) => Some((Add, inner_c.checked_sub(outer_c)?)),
        (Sub, Sub) => Some((Sub, inner_c.checked_add(outer_c)?)),
        (Sub, Add) => Some((Add, outer_c.checked_sub(inner_c)?)),
        (Mul, Mul) => Some((Mul, inner_c.checked_mul(outer_c)?)),
        _ => None,
    }
}

/// Combines `outer_c outer (a inner inner_c)` for commutative operators.
fn assoc_comm(outer: BinaryOp, inner: BinaryOp, outer_c: i64, inner_c: i64) -> Option<(BinaryOp, i64)> {
    use BinaryOp::*;
    match (outer, inner) {
        (Add, Add) => Some((Add, outer_c.checked_add(inner_c)?)),
        (Mul, Mul) => Some((Mul, outer_c.checked_mul(inner_c)?)),
        _ => None,
    }
}

/// Renders a constant as an expression node.
fn const_kind(value: ConstVal) -> ExprKind {
    match value {
        ConstVal::Int(number) => {
            let text = format!("{number}");
            let suffix_start = text.len() as u32;
            ExprKind::Literal {
                text,
                kind: TokenKind::Literal {
                    kind: LiteralKind::Int {
                        base: Base::Decimal,
                        empty_int: false,
                    },
                    suffix_start,
                },
            }
        }
        ConstVal::Bool(value) => ExprKind::Bool(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    /// Parses and optimizes `source`, returning the program and report.
    fn optimized(source: &str) -> (Program, OptReport) {
        let mut output = parse(source);
        let report = optimize(&mut output.program);
        (output.program, report)
    }

    /// Returns the optional tail expression of the program's first function.
    fn first_tail_opt(program: &Program) -> Option<&Expr> {
        match &program
            .items
            .first()
            .expect("program has an item")
            .kind
        {
            ItemKind::Fn(function) => function.body.tail.as_deref(),
            other => panic!("expected a function, found {other:?}"),
        }
    }

    /// Returns the tail expression of the program's first function.
    fn first_tail(program: &Program) -> &Expr {
        first_tail_opt(program).expect("function has a tail expression")
    }

    /// Returns the statements of the program's first function.
    fn first_stmts(program: &Program) -> &[Stmt] {
        match &program
            .items
            .first()
            .expect("program has an item")
            .kind
        {
            ItemKind::Fn(function) => &function.body.stmts,
            other => panic!("expected a function, found {other:?}"),
        }
    }

    /// Asserts the tail expression is a literal with the given text.
    fn assert_tail_literal(program: &Program, expected: &str) {
        match &first_tail(program).kind {
            ExprKind::Literal { text, .. } => assert_eq!(text, expected),
            other => panic!("expected a literal tail, found {other:?}"),
        }
    }

    #[test]
    fn folds_arithmetic_constants() {
        let (program, report) = optimized("fn f() { 1 + 2 * 3 }");
        assert_tail_literal(&program, "7");
        assert!(report.constant_folds >= 2, "{report:?}");
    }

    #[test]
    fn folds_comparison_to_bool() {
        let (program, _) = optimized("fn f() { 1 < 2 }");
        assert!(matches!(first_tail(&program).kind, ExprKind::Bool(true)));
    }

    #[test]
    fn skips_division_by_zero() {
        let (program, report) = optimized("fn f() { 1 / 0 }");
        assert!(matches!(
            first_tail(&program).kind,
            ExprKind::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));
        assert_eq!(report.constant_folds, 0);
    }

    #[test]
    fn skips_overflowing_fold() {
        let (program, _) = optimized("fn f() { 9223372036854775807 + 1 }");
        assert!(matches!(
            first_tail(&program).kind,
            ExprKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn folds_prefixed_and_suffixed_literals() {
        let (program, _) = optimized("fn f() { 0xFFu8 + 1 }");
        assert_tail_literal(&program, "256");
    }

    #[test]
    fn eliminates_double_negation() {
        let (program, report) = optimized("fn f() { -(-x) }");
        assert!(matches!(first_tail(&program).kind, ExprKind::Path(_)));
        assert!(report.algebraic_simplifications >= 1);
    }

    #[test]
    fn eliminates_double_not() {
        let (program, _) = optimized("fn f() { !!x }");
        assert!(matches!(first_tail(&program).kind, ExprKind::Path(_)));
    }

    #[test]
    fn applies_arithmetic_identities() {
        let (program, _) = optimized("fn f() { x * 1 + 0 }");
        assert!(matches!(first_tail(&program).kind, ExprKind::Path(_)));
    }

    #[test]
    fn folds_true_and_to_operand() {
        let (program, _) = optimized("fn f() { true && x }");
        assert!(matches!(first_tail(&program).kind, ExprKind::Path(_)));
    }

    #[test]
    fn keeps_impure_operand_of_false_and() {
        let (program, _) = optimized("fn f() { false && g() }");
        assert!(matches!(
            first_tail(&program).kind,
            ExprKind::Binary {
                op: BinaryOp::And,
                ..
            }
        ));
    }

    #[test]
    fn reassociates_constant_terms() {
        let (program, _) = optimized("fn f() { (x + 1) + 2 }");
        match &first_tail(&program).kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                rhs,
                ..
            } => match &rhs.kind {
                ExprKind::Literal { text, .. } => assert_eq!(text, "3"),
                other => panic!("expected a folded constant, found {other:?}"),
            },
            other => panic!("expected an addition, found {other:?}"),
        }
    }

    #[test]
    fn folds_if_true_to_then_block() {
        let (program, report) = optimized("fn f() { if true { g(); } else { h(); } }");
        let tail = first_tail(&program);
        match &tail.kind {
            ExprKind::Block(block) => {
                assert_eq!(block.stmts.len(), 1);
                let StmtKind::Expr(kept) = &block.stmts[0].kind else {
                    panic!("expected an expression statement");
                };
                let ExprKind::Call { callee, .. } = &kept.kind else {
                    panic!("expected a call");
                };
                let ExprKind::Path(path) = &callee.kind else {
                    panic!("expected a path callee");
                };
                assert_eq!(path.segments[0].name, "g");
            }
            other => panic!("expected a block, found {other:?}"),
        }
        assert!(report.branches_folded >= 1);
    }

    #[test]
    fn folds_if_false_without_else_to_unit() {
        let (program, report) = optimized("fn f() { if false { g(); } }");
        assert!(first_tail_opt(&program).is_none());
        assert!(first_stmts(&program).is_empty());
        assert!(report.branches_folded >= 1);
    }

    #[test]
    fn removes_statements_after_terminator() {
        let (program, report) = optimized("fn f() { return; g(); h(); }");
        assert_eq!(first_stmts(&program).len(), 1);
        assert!(report.dead_statements_removed >= 2);
    }

    #[test]
    fn removes_unreachable_tail_after_return() {
        let (program, _) = optimized("fn f() { return 1; 2 }");
        assert!(first_tail_opt(&program).is_none());
    }

    #[test]
    fn removes_empty_statements() {
        let (program, _) = optimized("fn f() { ; ; x; }");
        assert_eq!(first_stmts(&program).len(), 1);
    }

    #[test]
    fn removes_while_false_loop() {
        let (program, _) = optimized("fn f() { while false { g(); } }");
        assert!(first_stmts(&program).is_empty());
        assert!(first_tail_opt(&program).is_none());
    }

    #[test]
    fn collapses_repeated_cast() {
        let (program, _) = optimized("fn f() { x as i32 as i32 }");
        match &first_tail(&program).kind {
            ExprKind::Cast { expr, .. } => {
                assert!(matches!(expr.kind, ExprKind::Path(_)));
            }
            other => panic!("expected a single cast, found {other:?}"),
        }
    }

    #[test]
    fn folds_array_length_expressions() {
        let (program, _) = optimized("type Buf = [u8; 2 * 3];");
        let ItemKind::TypeAlias(alias) = &program.items[0].kind else {
            panic!("expected a type alias");
        };
        let TypeKind::Array { len, .. } = &alias.aliased.kind else {
            panic!("expected an array type");
        };
        match &len.kind {
            ExprKind::Literal { text, .. } => assert_eq!(text, "6"),
            other => panic!("expected a folded length, found {other:?}"),
        }
    }

    #[test]
    fn report_counts_changes() {
        let (_, report) = optimized("fn f() { if true { ; 1 + 1 } }");
        assert!(report.changes() >= 3, "{report:?}");
    }

    #[test]
    fn pure_classification_is_conservative() {
        let pure = parse("fn f() { 1 + x * -y }").program;
        let ItemKind::Fn(function) = &pure.items[0].kind else {
            panic!("expected a function");
        };
        let tail = function.body.tail.as_deref().expect("tail");
        assert!(is_pure(tail));

        let mixed = parse("fn f() { f() + 1 }").program;
        let ItemKind::Fn(function) = &mixed.items[0].kind else {
            panic!("expected a function");
        };
        let tail = function.body.tail.as_deref().expect("tail");
        assert!(!is_pure(tail));
    }

    #[test]
    fn empty_program_optimizes_to_nothing() {
        let (program, report) = optimized("");
        assert!(program.items.is_empty());
        assert_eq!(report.changes(), 0);
    }

    #[test]
    fn const_helpers_cover_bools_and_failures() {
        assert_eq!(
            fold_const(BinaryOp::And, ConstVal::Bool(true), ConstVal::Bool(false)),
            Some(ConstVal::Bool(false))
        );
        assert_eq!(
            fold_const(BinaryOp::Add, ConstVal::Bool(true), ConstVal::Int(1)),
            None
        );
        assert_eq!(
            fold_const(BinaryOp::Rem, ConstVal::Int(5), ConstVal::Int(0)),
            None
        );
        assert_eq!(
            assoc_right(BinaryOp::Add, BinaryOp::Sub, 5, 2),
            Some((BinaryOp::Add, 3))
        );
        assert_eq!(assoc_right(BinaryOp::Mul, BinaryOp::Add, 5, 2), None);
        assert_eq!(
            assoc_comm(BinaryOp::Mul, BinaryOp::Mul, 3, 4),
            Some((BinaryOp::Mul, 12))
        );
        let literal = const_kind(ConstVal::Int(-7));
        let node = Expr {
            kind: literal,
            span: crate::Span::default(),
        };
        assert_eq!(int_value(&node), Some(-7));
        assert!(
            int_value(&Expr {
                kind: ExprKind::Bool(true),
                span: crate::Span::default(),
            })
            .is_none()
        );
    }
}
