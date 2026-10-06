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

//! Abstract syntax tree for the Codevar OpenCL dialect.
//!
//! The tree is error-tolerant: every construct has an `Error` fallback so
//! that [`parse`](crate::parse) always produces a tree, and every node
//! carries a [`Span`] into the original source for diagnostics and
//! tooling.

use crate::Span;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use codevar_ocl_lex::TokenKind;

/// A complete source file.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Items in source order.
    pub items: Vec<Item>,
    /// Span covering the whole file.
    pub span: Span,
}

/// An opaque `#[…]` attribute region, preserved for tooling.
#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    /// Span covering `#` through the matching `]`.
    pub span: Span,
}

/// A top-level item.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    /// Attributes preceding the item.
    pub attrs: Vec<Attr>,
    /// What the item declares.
    pub kind: ItemKind,
    /// Span covering attributes and the item.
    pub span: Span,
}

/// Kind of a top-level [`Item`].
#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    /// A function definition: `fn name(…) -> T { … }`.
    Fn(FnItem),
    /// A struct declaration with named fields, or a unit struct.
    Struct(StructItem),
    /// A type alias: `type Name = T;`.
    TypeAlias(TypeAliasItem),
    /// A region the parser could not recognize as an item.
    Error,
}

/// A function item.
#[derive(Debug, Clone, PartialEq)]
pub struct FnItem {
    /// Function name.
    pub name: String,
    /// Generic parameters: `<'a, T>`.
    pub generics: Vec<GenericParam>,
    /// Formal parameters.
    pub params: Vec<Param>,
    /// Return type after `->`, if any.
    pub ret: Option<Type>,
    /// Function body.
    pub body: Block,
    /// Span of `fn` through the body's closing `}`.
    pub span: Span,
}

/// A generic parameter of a function or struct.
#[derive(Debug, Clone, PartialEq)]
pub enum GenericParam {
    /// A lifetime parameter: `'a`.
    Lifetime {
        /// Lifetime name without the leading `'`.
        name: String,
        /// Span including the `'`.
        span: Span,
    },
    /// A type parameter: `T`.
    Type {
        /// Parameter name.
        name: String,
        /// Span of the identifier.
        span: Span,
    },
}

/// A function parameter: `pattern: Type`.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Parameter pattern.
    pub pat: Pat,
    /// Declared type.
    pub ty: Type,
    /// Span of the whole parameter.
    pub span: Span,
}

/// A struct item with named fields, or a unit struct (`unit == true`).
#[derive(Debug, Clone, PartialEq)]
pub struct StructItem {
    /// Struct name.
    pub name: String,
    /// Generic parameters: `<'a, T>`.
    pub generics: Vec<GenericParam>,
    /// Named fields; empty for a unit struct.
    pub fields: Vec<Field>,
    /// Whether this is a unit struct declared with `;`.
    pub unit: bool,
    /// Span of `struct` through `}` or `;`.
    pub span: Span,
}

/// A named struct field: `name: Type`.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Field name.
    pub name: String,
    /// Field type.
    pub ty: Type,
    /// Span of `name: Type`.
    pub span: Span,
}

/// A type alias item: `type Name = T;`.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeAliasItem {
    /// Alias name.
    pub name: String,
    /// Generic parameters on the alias.
    pub generics: Vec<GenericParam>,
    /// The aliased type.
    pub aliased: Type,
    /// Span of `type` through `;`.
    pub span: Span,
}

/// A brace-delimited statement block with an optional tail expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// Statements in order; the tail expression is not included here.
    pub stmts: Vec<Stmt>,
    /// Final expression without a trailing `;`, if any.
    pub tail: Option<Box<Expr>>,
    /// Span including both braces (or the recovery region).
    pub span: Span,
}

/// A statement inside a [`Block`].
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    /// What the statement does.
    pub kind: StmtKind,
    /// Span of the statement.
    pub span: Span,
}

/// Kind of a [`Stmt`].
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    /// A `let` binding, which requires a trailing `;`.
    Let(LetStmt),
    /// An expression statement (possibly a block-like expression).
    Expr(Expr),
    /// An empty statement (`;`).
    Empty,
    /// A region the parser could not recognize as a statement.
    Error,
}

/// A `let` binding: `let pat: T = expr;`.
#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt {
    /// Bound pattern.
    pub pat: Pat,
    /// Optional type annotation.
    pub ty: Option<Type>,
    /// Optional initializer.
    pub init: Option<Expr>,
    /// Span of `let` through `;` (or the recovery region).
    pub span: Span,
}

/// A binding pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct Pat {
    /// Shape of the pattern.
    pub kind: PatKind,
    /// Span of the pattern.
    pub span: Span,
}

/// Kind of a [`Pat`].
#[derive(Debug, Clone, PartialEq)]
pub enum PatKind {
    /// A name binding, optionally declared `mut`.
    Ident {
        /// Bound name.
        name: String,
        /// Whether the binding is `mut`.
        mutable: bool,
    },
    /// The wildcard pattern `_`.
    Wild,
    /// A region the parser could not recognize as a pattern.
    Error,
}

/// A type expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    /// Shape of the type.
    pub kind: TypeKind,
    /// Span of the type.
    pub span: Span,
}

/// Kind of a [`Type`].
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    /// A reference: `&'a T` or `&mut T`.
    Ref {
        /// Optional lifetime: `'a`.
        lifetime: Option<String>,
        /// Whether the reference is `mut`.
        mutable: bool,
        /// Referenced type.
        inner: Box<Type>,
    },
    /// A raw pointer: `*const T` or `*mut T`.
    Ptr {
        /// Whether the pointee is `mut` (vs `const`).
        mutable: bool,
        /// Pointed-to type.
        inner: Box<Type>,
    },
    /// A tuple type: `(A, B)`; the unit type is the empty tuple.
    Tuple(Vec<Type>),
    /// A slice type: `[T]`.
    Slice(Box<Type>),
    /// A fixed-size array type: `[T; N]`.
    Array {
        /// Element type.
        elem: Box<Type>,
        /// Constant length expression.
        len: Box<Expr>,
    },
    /// A named (possibly qualified and generic) type: `a::b<T>`.
    Path(TypePath),
    /// A region the parser could not recognize as a type.
    Error,
}

/// A possibly qualified type name with optional generic arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct TypePath {
    /// Path segments joined by `::`.
    pub segments: Vec<Segment>,
    /// Generic arguments inside `<…>`, if present.
    pub args: Vec<GenericArg>,
    /// Span covering segments and generic arguments.
    pub span: Span,
}

/// One `name` in a [`Path`] or [`TypePath`].
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Segment name.
    pub name: String,
    /// Span of the identifier.
    pub span: Span,
}

/// A generic argument inside `<…>`.
#[derive(Debug, Clone, PartialEq)]
pub enum GenericArg {
    /// A type argument: `T`.
    Type(Type),
    /// A lifetime argument: `'a`.
    Lifetime {
        /// Lifetime name without the leading `'`.
        name: String,
        /// Span including the `'`.
        span: Span,
    },
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    /// Shape of the expression.
    pub kind: ExprKind,
    /// Span of the expression.
    pub span: Span,
}

/// Kind of an [`Expr`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// A numeric, character, or string literal.
    Literal {
        /// Exact source text of the literal.
        text: String,
        /// Lexed classification, including base and suffix data.
        kind: TokenKind,
    },
    /// A boolean literal: `true` or `false`.
    Bool(bool),
    /// A path expression: `a::b`.
    Path(Path),
    /// A unary expression: `-x`, `!x`, `*x`, `&x`, `&mut x`.
    Unary {
        /// The operator.
        op: UnaryOp,
        /// The operand.
        expr: Box<Expr>,
    },
    /// A binary expression: `a + b`, `a && b`, …
    Binary {
        /// The operator.
        op: BinaryOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// An assignment; `op` is set for compound forms such as `+=`.
    Assign {
        /// The compound operator, or `None` for plain `=`.
        op: Option<BinaryOp>,
        /// Assignment target.
        lhs: Box<Expr>,
        /// Assigned value.
        rhs: Box<Expr>,
    },
    /// A call: `f(a, b)`.
    Call {
        /// Callee expression.
        callee: Box<Expr>,
        /// Positional arguments.
        args: Vec<Expr>,
    },
    /// An index: `a[i]`.
    Index {
        /// Indexed expression.
        expr: Box<Expr>,
        /// Index expression.
        index: Box<Expr>,
    },
    /// A field or tuple-index access: `a.b`, `a.0`.
    Field {
        /// Base expression.
        expr: Box<Expr>,
        /// Field name or tuple index text.
        name: String,
    },
    /// The try operator: `a?`.
    Try {
        /// Operanded expression.
        expr: Box<Expr>,
    },
    /// A cast: `a as T`.
    Cast {
        /// Cast source.
        expr: Box<Expr>,
        /// Target type.
        ty: Box<Type>,
    },
    /// A range: `a..b`, `a..=b`, `..b`, `a..`, `..`.
    Range {
        /// Inclusive start of the range.
        start: Option<Box<Expr>>,
        /// Optional end of the range.
        end: Option<Box<Expr>>,
        /// Whether the end is included (`..=` vs `..`).
        inclusive: bool,
    },
    /// A block used as an expression.
    Block(Block),
    /// An `if` expression, with an optional `else` branch (block or `if`).
    If {
        /// Condition expression.
        cond: Box<Expr>,
        /// Then branch.
        then: Block,
        /// Else branch: a block or another `if`.
        else_branch: Option<Box<Expr>>,
    },
    /// A `while` loop expression.
    While {
        /// Condition evaluated each iteration.
        cond: Box<Expr>,
        /// Loop body.
        body: Block,
    },
    /// A `loop` expression with no condition.
    Loop {
        /// Loop body.
        body: Block,
    },
    /// A `for` loop expression.
    For {
        /// Binding pattern for each element.
        pat: Pat,
        /// Iterated expression.
        iter: Box<Expr>,
        /// Loop body.
        body: Block,
    },
    /// A `return` with an optional value.
    Return {
        /// Returned expression, if any.
        expr: Option<Box<Expr>>,
    },
    /// A `break` with an optional value.
    Break {
        /// Break value, if any.
        expr: Option<Box<Expr>>,
    },
    /// A `continue` expression.
    Continue,
    /// A tuple expression: `(a, b)`; a parenthesized expression is not a
    /// tuple — its inner expression is returned directly.
    Tuple(Vec<Expr>),
    /// An array expression: `[a, b]`.
    Array(Vec<Expr>),
    /// A region the parser could not recognize as an expression.
    Error,
}

/// An expression path: `a::b::c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    /// Path segments joined by `::`.
    pub segments: Vec<Segment>,
    /// Span covering the whole path.
    pub span: Span,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `!x`
    Not,
    /// `*x`
    Deref,
    /// `&x` or `&mut x`
    AddrOf {
        /// Whether the address-of target is `mut`.
        mutable: bool,
    },
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `&`
    BitAnd,
    /// `^`
    BitXor,
    /// `|`
    BitOr,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `&&`
    And,
    /// `||`
    Or,
}
