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

//! Exhaustive AST traversal in the style of rustc's [`ast::visit`] and
//! clang's `RecursiveASTVisitor`.
//!
//! Implement [`Visitor`] and override only the hooks of interest; each
//! default hook descends into children via the matching `walk_*` free
//! function. Hooks fire in pre-order (parent before its children), and
//! every [`ExprKind`] and sibling enum is matched exhaustively — adding
//! a new node variant is a compile error here until the walker teaches
//! how to descend into it.
//!
//! ```
//! use codevar_ocl_parse::{Program, Visitor, walk_program};
//!
//! struct Counter(usize);
//!
//! impl Visitor for Counter {
//!     fn visit_expr(&mut self, expr: &codevar_ocl_parse::Expr) {
//!         self.0 += 1;
//!         codevar_ocl_parse::visit::walk_expr(self, expr);
//!     }
//! }
//!
//! let output = codevar_ocl_parse::parse("fn f() -> int { 1 + 2 }");
//! let mut counter = Counter(0);
//! walk_program(&mut counter, &output.program);
//! assert_eq!(counter.0, 3);
//! ```

use crate::ast::{
    Block, Expr, ExprKind, FnItem, GenericArg, Item, ItemKind, LetStmt, Param, Pat, Program, Stmt, StmtKind,
    StructItem, Type, TypeAliasItem, TypeKind,
};

/// A pre-order traversal over the AST.
///
/// Override a hook to observe that node; call the corresponding
/// `walk_*` function (or rely on the default body) to continue into
/// children. Types are visited too because [`TypeKind::Array`] embeds a
/// constant-length [`Expr`].
pub trait Visitor {
    /// Visits a top-level item before its children.
    fn visit_item(&mut self, item: &Item) {
        walk_item(self, item);
    }
    /// Visits a function item before its parameters and body.
    fn visit_fn(&mut self, func: &FnItem) {
        walk_fn(self, func);
    }
    /// Visits a struct item before its field types.
    fn visit_struct(&mut self, item: &StructItem) {
        walk_struct(self, item);
    }
    /// Visits a type alias before the aliased type.
    fn visit_type_alias(&mut self, item: &TypeAliasItem) {
        walk_type_alias(self, item);
    }
    /// Visits a parameter before its pattern and type.
    fn visit_param(&mut self, param: &Param) {
        walk_param(self, param);
    }
    /// Visits a pattern; patterns are leaves.
    fn visit_pat(&mut self, pat: &Pat) {
        walk_pat(self, pat);
    }
    /// Visits a block before its statements and tail expression.
    fn visit_block(&mut self, block: &Block) {
        walk_block(self, block);
    }
    /// Visits a statement before its children.
    fn visit_stmt(&mut self, stmt: &Stmt) {
        walk_stmt(self, stmt);
    }
    /// Visits a `let` statement before its pattern, type, and
    /// initializer.
    fn visit_let(&mut self, let_stmt: &LetStmt) {
        walk_let(self, let_stmt);
    }
    /// Visits an expression before its operands.
    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }
    /// Visits a type before its children.
    fn visit_type(&mut self, ty: &Type) {
        walk_type(self, ty);
    }
}

/// Entry point: visits every item in `program`.
pub fn walk_program<V: Visitor + ?Sized>(visitor: &mut V, program: &Program) {
    for item in &program.items {
        visitor.visit_item(item);
    }
}

/// Descends into an item's children.
pub fn walk_item<V: Visitor + ?Sized>(visitor: &mut V, item: &Item) {
    match &item.kind {
        ItemKind::Fn(func) => visitor.visit_fn(func),
        ItemKind::Struct(strukt) => visitor.visit_struct(strukt),
        ItemKind::TypeAlias(alias) => visitor.visit_type_alias(alias),
        ItemKind::Error => {}
    }
}

/// Descends into a function's parameters, return type, and body.
pub fn walk_fn<V: Visitor + ?Sized>(visitor: &mut V, func: &FnItem) {
    for param in &func.params {
        visitor.visit_param(param);
    }
    if let Some(ret) = &func.ret {
        visitor.visit_type(ret);
    }
    visitor.visit_block(&func.body);
}

/// Descends into a struct's field types.
pub fn walk_struct<V: Visitor + ?Sized>(visitor: &mut V, strukt: &StructItem) {
    for field in &strukt.fields {
        visitor.visit_type(&field.ty);
    }
}

/// Descends into a type alias's target type.
pub fn walk_type_alias<V: Visitor + ?Sized>(visitor: &mut V, alias: &TypeAliasItem) {
    visitor.visit_type(&alias.aliased);
}

/// Descends into a parameter's pattern and type.
pub fn walk_param<V: Visitor + ?Sized>(visitor: &mut V, param: &Param) {
    visitor.visit_pat(&param.pat);
    visitor.visit_type(&param.ty);
}

/// Patterns hold no child nodes; nothing to descend into.
pub fn walk_pat<V: Visitor + ?Sized>(_visitor: &mut V, _pat: &Pat) {}

/// Descends into a block's statements and tail expression.
pub fn walk_block<V: Visitor + ?Sized>(visitor: &mut V, block: &Block) {
    for stmt in &block.stmts {
        visitor.visit_stmt(stmt);
    }
    if let Some(tail) = &block.tail {
        visitor.visit_expr(tail);
    }
}

/// Descends into a statement's children.
pub fn walk_stmt<V: Visitor + ?Sized>(visitor: &mut V, stmt: &Stmt) {
    match &stmt.kind {
        StmtKind::Let(let_stmt) => visitor.visit_let(let_stmt),
        StmtKind::Expr(expr) => visitor.visit_expr(expr),
        StmtKind::Empty | StmtKind::Error => {}
    }
}

/// Descends into a `let` statement's pattern, annotation, and
/// initializer.
pub fn walk_let<V: Visitor + ?Sized>(visitor: &mut V, let_stmt: &LetStmt) {
    visitor.visit_pat(&let_stmt.pat);
    if let Some(ty) = &let_stmt.ty {
        visitor.visit_type(ty);
    }
    if let Some(init) = &let_stmt.init {
        visitor.visit_expr(init);
    }
}

/// Descends into an expression's operands.
///
/// The match is exhaustive over [`ExprKind`] with no wildcard: a new
/// variant fails to compile here until its children are wired in.
pub fn walk_expr<V: Visitor + ?Sized>(visitor: &mut V, expr: &Expr) {
    match &expr.kind {
        ExprKind::Literal { .. } | ExprKind::Bool(_) | ExprKind::Continue | ExprKind::Error => {}
        ExprKind::Path(_) => {}
        ExprKind::Unary { expr: inner, .. } => visitor.visit_expr(inner),
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::Assign { lhs, rhs, .. } => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(rhs);
        }
        ExprKind::Call { callee, args } => {
            visitor.visit_expr(callee);
            for arg in args {
                visitor.visit_expr(arg);
            }
        }
        ExprKind::Index { expr: inner, index } => {
            visitor.visit_expr(inner);
            visitor.visit_expr(index);
        }
        ExprKind::Field { expr: inner, .. } | ExprKind::Try { expr: inner } => {
            visitor.visit_expr(inner);
        }
        ExprKind::Cast { expr: inner, ty } => {
            visitor.visit_expr(inner);
            visitor.visit_type(ty);
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(start) = start {
                visitor.visit_expr(start);
            }
            if let Some(end) = end {
                visitor.visit_expr(end);
            }
        }
        ExprKind::Block(block) => visitor.visit_block(block),
        ExprKind::If {
            cond,
            then,
            else_branch,
        } => {
            visitor.visit_expr(cond);
            visitor.visit_block(then);
            if let Some(else_branch) = else_branch {
                visitor.visit_expr(else_branch);
            }
        }
        ExprKind::While { cond, body } => {
            visitor.visit_expr(cond);
            visitor.visit_block(body);
        }
        ExprKind::Loop { body } => visitor.visit_block(body),
        ExprKind::For { pat, iter, body } => {
            visitor.visit_pat(pat);
            visitor.visit_expr(iter);
            visitor.visit_block(body);
        }
        ExprKind::Return { expr: inner } | ExprKind::Break { expr: inner } => {
            if let Some(inner) = inner {
                visitor.visit_expr(inner);
            }
        }
        ExprKind::Tuple(items) | ExprKind::Array(items) => {
            for item in items {
                visitor.visit_expr(item);
            }
        }
    }
}

/// Descends into a type's children.
///
/// Exhaustive over [`TypeKind`]; [`TypeKind::Array`] carries a
/// constant-length expression that is part of the AST proper.
pub fn walk_type<V: Visitor + ?Sized>(visitor: &mut V, ty: &Type) {
    match &ty.kind {
        TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => visitor.visit_type(inner),
        TypeKind::Slice(inner) => visitor.visit_type(inner),
        TypeKind::Tuple(elems) => {
            for elem in elems {
                visitor.visit_type(elem);
            }
        }
        TypeKind::Array { elem, len } => {
            visitor.visit_type(elem);
            visitor.visit_expr(len);
        }
        TypeKind::Path(path) => {
            for arg in &path.args {
                if let GenericArg::Type(arg_ty) = arg {
                    visitor.visit_type(arg_ty);
                }
            }
        }
        TypeKind::Error => {}
    }
}
