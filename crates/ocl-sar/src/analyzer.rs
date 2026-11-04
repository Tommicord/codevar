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

//! The semantic analyzer: name resolution, type checking, and lints.
//!
//! Every pass is error-tolerant: a failed resolution yields [`Ty::Error`],
//! which coerces to anything, so one bad name never floods the output.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use codevar_ocl_lex::{Base, LiteralKind, TokenKind};
use codevar_ocl_parse::{
    Attr, BinaryOp, Block, Expr, ExprKind, FnItem, GenericArg, GenericParam, Item, ItemKind, NodeId, Param,
    Pat, PatKind, Path, Program, Span, Stmt, StmtKind, Type, TypeKind, TypePath, UnaryOp,
};

use crate::builtins::{Builtin, BuiltinKind, builtins, lookup_builtin_fn};
use crate::confusable::confusable_skeleton;
use crate::diagnostic::{Diagnostic, MessageBuilder, codes};
use crate::literal::{
    decode_escapes, float_scalar_for_suffix, float_value, int_digits_value, int_scalar_for_suffix,
    strip_raw_ident,
};
use crate::tables::{Res, ResolutionTable, TypeTable};
use crate::types::{BuiltinType, Scalar, Ty, coerce, lookup_builtin, substitute, unify};

/// Dialect keywords; these may not be declared as names.
const DIALECT_KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "else", "false", "fn", "for", "if", "in", "let", "loop", "mut",
    "return", "struct", "true", "type", "while",
];

/// Identifiers the OpenCL specification reserves for its own use.
const RESERVED_OPENCL: &[&str] = &[
    "kernel",
    "global",
    "local",
    "private",
    "constant",
    "generic",
    "uniform",
    "restrict",
    "volatile",
    "read_only",
    "write_only",
    "read_write",
    "pipe",
];

/// Attributes the dialect understands.
const KNOWN_ATTRIBUTES: &[&str] = &["kernel"];

/// A top-level declaration the analyzer resolved.
///
/// Declarations appear in source order and cover every item in the file,
/// including items the type checker had to recover from.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    /// Declared name with any `r#` raw-identifier marker stripped.
    pub name: String,
    /// Span covering the item, attributes included.
    pub span: Span,
    /// Position of the item in the program, used to keep source order.
    pub item_index: usize,
    /// What the item declares.
    pub kind: DeclKind,
}

/// The resolved shape of a [`Declaration`].
#[derive(Debug, Clone, PartialEq)]
pub enum DeclKind {
    /// A function with its resolved parameter types and return type.
    Function {
        /// Parameter names and types, in declaration order.
        params: Vec<(String, Ty)>,
        /// Resolved return type (`void` when the signature omits it).
        ret: Ty,
        /// Whether the function carries `#[kernel]`.
        kernel: bool,
    },
    /// A struct with its resolved fields; empty for a unit struct.
    Struct {
        /// Field names and types, in declaration order.
        fields: Vec<(String, Ty)>,
    },
    /// A type alias and the type it resolves to.
    Alias {
        /// The aliased type after generic substitution.
        target: Ty,
    },
}

/// An attribute reduced to the facts validation needs.
#[derive(Debug, Clone)]
struct AttrSketch {
    /// The name written inside `#[…]`.
    name: String,
    /// Span covering `#` through the matching `]`.
    span: Span,
}

/// One function parameter reduced to the facts signature resolution
/// needs; patterns and bodies are never retained.
#[derive(Debug, Clone)]
struct ParamSketch {
    /// Bound name with the raw-identifier marker stripped; `None` for
    /// `_` and non-ident patterns.
    bound: Option<String>,
    /// Cloned declared type.
    ty: Type,
    /// Span of the whole parameter.
    span: Span,
}

/// One struct field reduced to the facts field resolution needs.
#[derive(Debug, Clone)]
struct FieldSketch {
    /// Field name as written, raw-identifier marker included.
    raw_name: String,
    /// Cloned declared type.
    ty: Type,
    /// Span of the field declaration.
    span: Span,
}

/// An owned sketch of a function item: names, spans, and cloned
/// signature types — no body and no token state.
#[derive(Debug, Clone)]
struct FnSketch {
    /// Position of the item in the program.
    index: usize,
    /// Function name as written, raw-identifier marker included.
    raw_name: String,
    /// Function name with the raw-identifier marker stripped.
    name: String,
    /// Span of the declared name.
    name_span: Span,
    /// Span of `fn` through the body's closing `}`.
    span: Span,
    /// Whether the item carries `#[kernel]`.
    kernel: bool,
    /// Attributes preceding the item.
    attrs: Vec<AttrSketch>,
    /// Declared generic type-parameter names, lifetimes excluded.
    generics: Vec<String>,
    /// Parameter sketches, in declaration order.
    params: Vec<ParamSketch>,
    /// Cloned return type, if written.
    ret: Option<Type>,
}

/// An owned sketch of a struct item.
#[derive(Debug, Clone)]
struct StructSketch {
    /// Position of the item in the program.
    index: usize,
    /// Struct name as written, raw-identifier marker included.
    raw_name: String,
    /// Struct name with the raw-identifier marker stripped.
    name: String,
    /// Span of the declared name.
    name_span: Span,
    /// Span of `struct` through `}` or `;`.
    span: Span,
    /// Attributes preceding the item.
    attrs: Vec<AttrSketch>,
    /// Declared generic type-parameter names, lifetimes excluded.
    generics: Vec<String>,
    /// Field sketches, in declaration order.
    fields: Vec<FieldSketch>,
}

/// An owned sketch of a type-alias item.
#[derive(Debug, Clone)]
struct AliasSketch {
    /// Position of the item in the program.
    index: usize,
    /// Alias name as written, raw-identifier marker included.
    raw_name: String,
    /// Alias name with the raw-identifier marker stripped.
    name: String,
    /// Span of the declared name.
    name_span: Span,
    /// Span of `type` through `;`.
    span: Span,
    /// Attributes preceding the item.
    attrs: Vec<AttrSketch>,
    /// Declared generic type-parameter names, lifetimes excluded.
    generics: Vec<String>,
    /// Cloned aliased type, generics unsubstituted.
    ty: Type,
}

/// An owned sketch of one top-level item.
#[derive(Debug, Clone)]
enum ItemSketch {
    /// A function.
    Fn(FnSketch),
    /// A struct.
    Struct(StructSketch),
    /// A type alias.
    Alias(AliasSketch),
    /// A region the parser could not recover as an item.
    Error,
}

impl ItemSketch {
    /// Reduces one parsed item to an owned sketch.
    ///
    /// The collector only reads the AST during this call: names, spans,
    /// attribute names, and `ast::Ty` nodes are cloned, nothing is
    /// borrowed.
    fn from_item(source: &str, index: usize, item: &Item) -> Self {
        match &item.kind {
            ItemKind::Fn(function) => {
                let raw_name = function.name.clone();
                let name = strip_raw_ident(&raw_name).to_string();
                let name_span = decl_name_span(source, item.span, "fn", &name);
                let kernel = item
                    .attrs
                    .iter()
                    .any(|attr| attr_name(source, attr) == "kernel");
                let params = function
                    .params
                    .iter()
                    .map(|param| ParamSketch {
                        bound: match &param.pat.kind {
                            PatKind::Ident { name, .. } if strip_raw_ident(name) != "_" => {
                                Some(strip_raw_ident(name).to_string())
                            }
                            _ => None,
                        },
                        ty: param.ty.clone(),
                        span: param.span,
                    })
                    .collect();
                Self::Fn(FnSketch {
                    index,
                    raw_name,
                    name,
                    name_span,
                    span: function.span,
                    kernel,
                    attrs: attr_sketches(source, &item.attrs),
                    generics: generic_names(&function.generics),
                    params,
                    ret: function.ret.clone(),
                })
            }
            ItemKind::Struct(record) => {
                let raw_name = record.name.clone();
                let name = strip_raw_ident(&raw_name).to_string();
                let name_span = decl_name_span(source, item.span, "struct", &name);
                let fields = record
                    .fields
                    .iter()
                    .map(|field| FieldSketch {
                        raw_name: field.name.clone(),
                        ty: field.ty.clone(),
                        span: field.span,
                    })
                    .collect();
                Self::Struct(StructSketch {
                    index,
                    raw_name,
                    name,
                    name_span,
                    span: record.span,
                    attrs: attr_sketches(source, &item.attrs),
                    generics: generic_names(&record.generics),
                    fields,
                })
            }
            ItemKind::TypeAlias(alias) => {
                let raw_name = alias.name.clone();
                let name = strip_raw_ident(&raw_name).to_string();
                let name_span = decl_name_span(source, item.span, "type", &name);
                Self::Alias(AliasSketch {
                    index,
                    raw_name,
                    name,
                    name_span,
                    span: alias.span,
                    attrs: attr_sketches(source, &item.attrs),
                    generics: generic_names(&alias.generics),
                    ty: alias.aliased.clone(),
                })
            }
            ItemKind::Error => Self::Error,
        }
    }
}

/// Reduces an item's attributes to names and spans.
fn attr_sketches(source: &str, attrs: &[Attr]) -> Vec<AttrSketch> {
    attrs
        .iter()
        .map(|attr| AttrSketch {
            name: attr_name(source, attr).to_string(),
            span: attr.span,
        })
        .collect()
}

/// Every top-level definition in the program, keyed by resolved name.
///
/// The maps own small sketches — names, spans, and cloned signature
/// types — so the environment never retains an AST, a body, or a token
/// buffer.
#[derive(Debug)]
struct Defs {
    /// Sketches of every item, in source order (duplicates included).
    items: Vec<ItemSketch>,
    /// Functions by name; a later duplicate replaces the earlier entry.
    fns: BTreeMap<String, FnSketch>,
    /// Structs by name; a later duplicate replaces the earlier entry.
    structs: BTreeMap<String, StructSketch>,
    /// Type aliases by name; a later duplicate replaces the earlier entry.
    aliases: BTreeMap<String, AliasSketch>,
}

impl Defs {
    /// Indexes every sketch; later duplicates replace the earlier entry.
    fn from_sketches(items: Vec<ItemSketch>) -> Self {
        let mut defs = Self {
            items,
            fns: BTreeMap::new(),
            structs: BTreeMap::new(),
            aliases: BTreeMap::new(),
        };
        for sketch in &defs.items {
            match sketch {
                ItemSketch::Fn(function) => {
                    defs.fns
                        .insert(function.name.clone(), function.clone());
                }
                ItemSketch::Struct(record) => {
                    defs.structs
                        .insert(record.name.clone(), record.clone());
                }
                ItemSketch::Alias(alias) => {
                    defs.aliases
                        .insert(alias.name.clone(), alias.clone());
                }
                ItemSketch::Error => {}
            }
        }
        defs
    }
}

/// The resolved signature of a function.
#[derive(Debug, Clone)]
struct FnSig {
    /// Position of the item in the program.
    index: usize,
    /// Declared generic parameter names.
    generics: Vec<String>,
    /// Resolved parameter types, in order.
    params: Vec<ParamSig>,
    /// Resolved return type.
    ret: Ty,
    /// Whether the function carries `#[kernel]`.
    kernel: bool,
    /// Span covering the whole item.
    span: Span,
    /// Span of the function name.
    name_span: Span,
}

/// One resolved function parameter.
#[derive(Debug, Clone)]
struct ParamSig {
    /// Bound name, `None` for `_: T`.
    name: Option<String>,
    /// Resolved parameter type.
    ty: Ty,
}

/// The resolved signature of a struct.
#[derive(Debug, Clone)]
struct StructSig {
    /// Position of the item in the program.
    index: usize,
    /// Declared generic parameter names.
    generics: Vec<String>,
    /// Resolved fields, in declaration order.
    fields: Vec<FieldSig>,
    /// Span covering the whole item.
    span: Span,
    /// Span of the struct name.
    name_span: Span,
}

/// One resolved struct field.
#[derive(Debug, Clone)]
struct FieldSig {
    /// Field name with the raw-identifier marker stripped.
    name: String,
    /// Span of the field declaration.
    span: Span,
    /// Resolved field type.
    ty: Ty,
}

/// The resolved target of a type alias.
#[derive(Debug, Clone)]
struct AliasSig {
    /// Position of the item in the program.
    index: usize,
    /// The aliased type, generics unsubstituted.
    ty: Ty,
    /// Span covering the whole item.
    span: Span,
    /// Span of the alias name.
    name_span: Span,
}

/// A name bound by a parameter, `let`, or `for` pattern.
#[derive(Debug, Clone, PartialEq)]
struct Binding {
    /// Bound name with the raw-identifier marker stripped.
    name: String,
    /// Type of the binding.
    ty: Ty,
    /// Span of the pattern that introduced it.
    span: Span,
    /// Identity of the pattern that introduced it.
    pat: NodeId,
    /// Where the binding came from and whether it has been read.
    state: BindingState,
}

/// Where a binding was declared, folded with its mutability and use flag.
///
/// The dialect forbids structs holding more than one `bool`, so the three
/// facts live in one enum rather than as separate fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingState {
    /// A function parameter; never reported as unused.
    Parameter {
        /// Whether the pattern declared the parameter `mut`.
        mutable: bool,
    },
    /// A local binding introduced by `let` or `for`.
    Local {
        /// Whether the pattern declared the binding `mut`.
        mutable: bool,
        /// Whether the binding has been read or assigned to.
        used: bool,
    },
}

impl BindingState {
    /// True when the binding may be assigned to.
    const fn is_mutable(self) -> bool {
        match self {
            Self::Parameter { mutable } | Self::Local { mutable, .. } => mutable,
        }
    }

    /// True when a local binding has never been read.
    const fn is_unused(self) -> bool {
        matches!(self, Self::Local { used: false, .. })
    }

    /// Records a read of the binding.
    const fn mark_used(&mut self) {
        if let Self::Local { used, .. } = self {
            *used = true;
        }
    }
}

/// A lexical scope holding the bindings it introduced.
#[derive(Debug, Default)]
struct Scope {
    /// Bindings in declaration order; later ones shadow earlier ones.
    bindings: Vec<Binding>,
}

/// State of the function body currently being checked.
#[derive(Debug, Clone)]
struct FnCtx {
    /// Name of the function being checked.
    name: String,
    /// Its resolved return type.
    ret: Ty,
    /// How many enclosing loops accept `break` and `continue`.
    loop_depth: usize,
    /// Generic parameters in scope inside the body.
    generics: Vec<String>,
}

/// The mutable half of the analyzer: diagnostics and resolved state.
struct Sema<'a> {
    /// Collected definitions, borrowed for the whole run.
    defs: &'a Defs,
    /// Every diagnostic reported so far.
    diagnostics: Vec<Diagnostic>,
    /// Resolved function signatures.
    fn_sigs: BTreeMap<String, FnSig>,
    /// Resolved struct signatures.
    struct_sigs: BTreeMap<String, StructSig>,
    /// Resolved type aliases.
    alias_sigs: BTreeMap<String, AliasSig>,
    /// Names referenced as values or types, for the unused lint.
    references: BTreeSet<String>,
    /// Call graph edges: caller -> (callee, call span).
    calls: BTreeMap<String, Vec<(String, Span)>>,
    /// Alias names currently being resolved, innermost last.
    resolving_aliases: Vec<String>,
    /// Open scopes of the function body being checked.
    scopes: Vec<Scope>,
    /// The function whose body is being checked.
    ctx: Option<FnCtx>,
    /// Types recorded for every type-checked expression.
    types: TypeTable,
    /// Resolutions recorded for every resolved name use.
    resolutions: ResolutionTable,
    /// Whether a stage outside this run already reported an error.
    ///
    /// The streaming path sets it so per-body checks see the same
    /// error state the whole-file run accumulates in one diagnostics
    /// queue (unused-variable warnings stay quiet after any error).
    prior_errors: bool,
}

/// The analyzer's full result: diagnostics, declarations, and the side
/// tables keyed by [`NodeId`].
pub(crate) struct RunOutput {
    /// Diagnostics from every pass, unsorted.
    pub(crate) diagnostics: Vec<Diagnostic>,
    /// Declarations in source order.
    pub(crate) declarations: Vec<Declaration>,
    /// Type of every type-checked expression.
    pub(crate) types: TypeTable,
    /// Resolution of every resolved name use.
    pub(crate) resolutions: ResolutionTable,
}

/// Runs every semantic pass and returns diagnostics, declarations, and
/// the side tables.
pub(crate) fn run(source: &str, program: &Program) -> RunOutput {
    let sketches: Vec<ItemSketch> = program
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| ItemSketch::from_item(source, index, item))
        .collect();
    let defs = Defs::from_sketches(sketches);
    // Bodies stay borrowed from the program for this call only; the
    // sketch maps never retain them.
    let mut bodies: BTreeMap<String, &FnItem> = BTreeMap::new();
    for item in &program.items {
        if let ItemKind::Fn(function) = &item.kind {
            bodies.insert(strip_raw_ident(&function.name).to_string(), function);
        }
    }
    let mut sema = Sema::new(&defs);
    sema.validate_items();
    sema.resolve_structs();
    sema.resolve_aliases();
    sema.resolve_signatures();
    let names: Vec<String> = defs.fns.keys().cloned().collect();
    for name in names {
        let Some(sig) = sema.fn_sigs.get(&name).cloned() else {
            continue;
        };
        let Some(function) = bodies.get(&name).copied() else {
            continue;
        };
        sema.check_body(&name, &sig, &function.params, &function.body);
    }
    collect_call_cycles(&sema.calls, &mut sema.diagnostics);
    collect_struct_cycles(&sema.struct_sigs, &mut sema.diagnostics);
    let has_errors = sema.diagnostics.iter().any(Diagnostic::is_error);
    collect_unused(
        &sema.fn_sigs,
        &sema.struct_sigs,
        &sema.alias_sigs,
        &sema.references,
        has_errors,
        &mut sema.diagnostics,
    );
    let declarations = build_declarations(&sema.fn_sigs, &sema.struct_sigs, &sema.alias_sigs);
    RunOutput {
        diagnostics: sema.diagnostics,
        declarations,
        types: sema.types,
        resolutions: sema.resolutions,
    }
}

/// Which analysis stages have reported at least one error.
///
/// The phases fold into a single mask field (the crate avoids structs
/// holding several `bool`s) because the gates differ per stage:
/// unused-variable warnings inside a body respect only pre-body
/// failures, while the whole-file unused lint respects every failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ErrorPhases {
    /// Bit set per failed [`ErrorPhase`].
    failed: u8,
}

/// One analysis stage whose errors feed [`ErrorPhases`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorPhase {
    /// Item validation plus struct, alias, and signature resolution —
    /// the stages that run before any body.
    Declarations,
    /// Declaration-level cycle checks (struct cycles).
    Cycles,
    /// At least one type-checked function body.
    Bodies,
}

impl ErrorPhase {
    /// This phase's bit in an [`ErrorPhases`] mask.
    const fn bit(self) -> u8 {
        match self {
            Self::Declarations => 1,
            Self::Cycles => 2,
            Self::Bodies => 4,
        }
    }
}

impl ErrorPhases {
    /// True when any stage has reported an error.
    const fn any(self) -> bool {
        self.failed != 0
    }

    /// True when `phase` has reported an error.
    const fn contains(self, phase: ErrorPhase) -> bool {
        self.failed & phase.bit() != 0
    }

    /// Records an error from `phase`.
    fn record(&mut self, phase: ErrorPhase) {
        self.failed |= phase.bit();
    }
}

/// Whole-file declaration environment: resolved signatures for every
/// function, struct, and alias, plus the declaration list. Small: it
/// holds no ASTs and no body state.
///
/// A [`DeclCollector`] produces it; [`analyze_body`] then consumes
/// function items against it, and [`file_checks`](crate::file_checks)
/// closes the run.
#[derive(Debug)]
pub struct DeclEnv {
    /// Owned signature-level sketches of every item.
    defs: Defs,
    /// Resolved function signatures.
    fn_sigs: BTreeMap<String, FnSig>,
    /// Resolved struct signatures.
    struct_sigs: BTreeMap<String, StructSig>,
    /// Resolved type aliases.
    alias_sigs: BTreeMap<String, AliasSig>,
    /// Names referenced as values or types, for the unused lint.
    references: BTreeSet<String>,
    /// Call graph edges accumulated by [`analyze_body`].
    calls: BTreeMap<String, Vec<(String, Span)>>,
    /// Declarations in source order.
    declarations: Vec<Declaration>,
    /// Stages that have reported errors so far.
    errors: ErrorPhases,
}

impl DeclEnv {
    /// Resolved declarations in source order (as [`analyze`](crate::analyze)
    /// returns them).
    #[must_use]
    pub fn declarations(&self) -> &[Declaration] {
        &self.declarations
    }

    /// True when a `#[kernel]` function named `name` is declared.
    #[must_use]
    pub fn is_kernel(&self, name: &str) -> bool {
        self.fn_sigs
            .get(name)
            .is_some_and(|sig| sig.kernel)
    }
}

/// Incremental declaration-phase collector.
///
/// Feed every top-level item in source order, then call
/// [`finish`](Self::finish) once. The collector keeps only owned
/// sketches — names, generic names, and cloned `ast::Ty` nodes — so a
/// file's declaration phase never materializes its ASTs.
///
/// # Examples
///
/// ```
/// use codevar_ocl_parse::ItemStream;
/// use codevar_ocl_sar::DeclCollector;
///
/// let source = "#[kernel]\nfn zero(out: *mut int) { *out = 0; }";
/// let mut collector = DeclCollector::new();
/// let mut stream = ItemStream::new(source);
/// while let Some(outcome) = stream.next_item() {
///     collector.feed(source, &outcome.item);
/// }
/// let (env, diagnostics) = collector.finish();
/// assert!(diagnostics.is_empty());
/// assert!(env.is_kernel("zero"));
/// ```
#[derive(Debug, Default)]
pub struct DeclCollector {
    /// Sketches of every fed item, in feed order.
    items: Vec<ItemSketch>,
}

impl DeclCollector {
    /// Creates an empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumes one top-level item (already parsed; the collector only
    /// reads it during the call — it must NOT retain AST borrows; keep
    /// owned "sketches": names, generic names, cloned `ast::Ty` nodes).
    ///
    /// Items must be fed in source order; each call consumes one
    /// program index, error items included, so declaration
    /// `item_index` values match a whole-file [`analyze`](crate::analyze)
    /// run.
    pub fn feed(&mut self, source: &str, item: &Item) {
        let index = self.items.len();
        self.items
            .push(ItemSketch::from_item(source, index, item));
    }

    /// Ends the declaration phase: resolves structs, aliases, and
    /// signatures; runs declaration-level validation and alias/struct
    /// cycle checks; returns the environment and every diagnostic from
    /// those stages (sorted by span).
    #[must_use]
    pub fn finish(self) -> (DeclEnv, Vec<Diagnostic>) {
        let defs = Defs::from_sketches(self.items);
        let mut sema = Sema::new(&defs);
        sema.validate_items();
        sema.resolve_structs();
        sema.resolve_aliases();
        sema.resolve_signatures();
        let mut errors = ErrorPhases::default();
        if sema.has_errors() {
            errors.record(ErrorPhase::Declarations);
        }
        collect_struct_cycles(&sema.struct_sigs, &mut sema.diagnostics);
        if sema.diagnostics.iter().any(Diagnostic::is_error) {
            errors.record(ErrorPhase::Cycles);
        }
        let Sema {
            defs: _,
            diagnostics,
            fn_sigs,
            struct_sigs,
            alias_sigs,
            references,
            calls,
            ..
        } = sema;
        let declarations = build_declarations(&fn_sigs, &struct_sigs, &alias_sigs);
        let mut diagnostics = diagnostics;
        crate::sort_diagnostics(&mut diagnostics);
        let env = DeclEnv {
            defs,
            fn_sigs,
            struct_sigs,
            alias_sigs,
            references,
            calls,
            declarations,
            errors,
        };
        (env, diagnostics)
    }
}

/// Per-body analysis tables (NodeId spaces match the single item's AST).
#[derive(Debug)]
pub struct BodyTables {
    /// Type of every type-checked expression in the body.
    pub types: TypeTable,
    /// Resolution of every resolved name use in the body.
    pub resolutions: ResolutionTable,
}

impl BodyTables {
    /// Empty tables, returned for items with no body to check.
    fn empty() -> Self {
        Self {
            types: TypeTable::new(),
            resolutions: ResolutionTable::new(),
        }
    }
}

/// Type-checks the body of one function `item` against `env`.
///
/// Also records the body's name references and call edges into `env`
/// (so [`file_checks`](crate::file_checks) can see them). Returns the
/// body's diagnostics and its side tables. Non-function items are a
/// no-op returning empty tables.
///
/// The tables are keyed by the item's own [`NodeId`] space, matching
/// per-item parsing by [`ItemStream`](codevar_ocl_parse::ItemStream).
/// When a name was declared more than once, only the last declaration's
/// body is checked, exactly as [`analyze`](crate::analyze) does.
///
/// Full diagnostic equivalence with [`analyze`](crate::analyze) holds
/// when bodies are checked in the order [`analyze`] checks them
/// (function-name order): unused-variable warnings inside a body are
/// suppressed by any earlier error, so a different body order can
/// surface those warnings in a different set.
pub fn analyze_body(env: &mut DeclEnv, source: &str, item: &Item) -> (Vec<Diagnostic>, BodyTables) {
    // The signature-level text work (attribute names, declaration name
    // spans) already happened at feed time; `source` is accepted for
    // symmetry with the rest of the streaming driver's per-item calls.
    let _ = source;
    let ItemKind::Fn(function) = &item.kind else {
        return (Vec::new(), BodyTables::empty());
    };
    let name = strip_raw_ident(&function.name);
    let retained_last = env
        .defs
        .fns
        .get(name)
        .is_some_and(|sketch| sketch.span == function.span);
    if !retained_last {
        return (Vec::new(), BodyTables::empty());
    }
    let Some(sig) = env.fn_sigs.get(name).cloned() else {
        return (Vec::new(), BodyTables::empty());
    };
    let DeclEnv {
        defs,
        fn_sigs,
        struct_sigs,
        alias_sigs,
        references,
        calls,
        declarations: _,
        errors,
    } = env;
    let mut sema = Sema::new(defs);
    sema.prior_errors = errors.contains(ErrorPhase::Declarations) || errors.contains(ErrorPhase::Bodies);
    sema.fn_sigs = core::mem::take(fn_sigs);
    sema.struct_sigs = core::mem::take(struct_sigs);
    sema.alias_sigs = core::mem::take(alias_sigs);
    sema.references = core::mem::take(references);
    sema.calls = core::mem::take(calls);
    sema.check_body(name, &sig, &function.params, &function.body);
    let diagnostics = core::mem::take(&mut sema.diagnostics);
    let tables = BodyTables {
        types: core::mem::take(&mut sema.types),
        resolutions: core::mem::take(&mut sema.resolutions),
    };
    if diagnostics.iter().any(Diagnostic::is_error) {
        errors.record(ErrorPhase::Bodies);
    }
    *fn_sigs = core::mem::take(&mut sema.fn_sigs);
    *struct_sigs = core::mem::take(&mut sema.struct_sigs);
    *alias_sigs = core::mem::take(&mut sema.alias_sigs);
    *references = core::mem::take(&mut sema.references);
    *calls = core::mem::take(&mut sema.calls);
    (diagnostics, tables)
}

/// Whole-file checks over the accumulated environment: call cycles,
/// unused declarations, and any remaining global checks. Returns their
/// diagnostics (sorted by span).
///
/// Call it once, after every function body has been through
/// [`analyze_body`].
pub fn file_checks(env: &DeclEnv) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    collect_call_cycles(&env.calls, &mut diagnostics);
    let has_errors = env.errors.any() || diagnostics.iter().any(Diagnostic::is_error);
    collect_unused(
        &env.fn_sigs,
        &env.struct_sigs,
        &env.alias_sigs,
        &env.references,
        has_errors,
        &mut diagnostics,
    );
    crate::sort_diagnostics(&mut diagnostics);
    diagnostics
}

impl<'a> Sema<'a> {
    /// Creates an analyzer over already-collected definitions.
    fn new(defs: &'a Defs) -> Self {
        Self {
            defs,
            diagnostics: Vec::new(),
            fn_sigs: BTreeMap::new(),
            struct_sigs: BTreeMap::new(),
            alias_sigs: BTreeMap::new(),
            references: BTreeSet::new(),
            calls: BTreeMap::new(),
            resolving_aliases: Vec::new(),
            scopes: Vec::new(),
            ctx: None,
            types: TypeTable::new(),
            resolutions: ResolutionTable::new(),
            prior_errors: false,
        }
    }

    /// True once any stage has reported an error, this run included.
    fn has_errors(&self) -> bool {
        self.prior_errors || self.diagnostics.iter().any(Diagnostic::is_error)
    }

    /// Pushes a diagnostic onto the queue.
    fn report(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Validates attributes, names, and duplicate declarations.
    fn validate_items(&mut self) {
        let defs = self.defs;
        for (index, sketch) in defs.items.iter().enumerate() {
            match sketch {
                ItemSketch::Fn(function) => {
                    self.check_name(&function.raw_name, function.name_span);
                    if let Some(first) = defs.fns.get(&function.name)
                        && first.index != index
                    {
                        self.report_duplicate(&function.name, first.name_span, function.name_span);
                    }
                    self.check_attrs(&function.attrs, true);
                }
                ItemSketch::Struct(record) => {
                    self.check_name(&record.raw_name, record.name_span);
                    if let Some(first) = defs.structs.get(&record.name)
                        && first.index != index
                    {
                        self.report_duplicate(&record.name, first.name_span, record.name_span);
                    }
                    self.check_attrs(&record.attrs, false);
                }
                ItemSketch::Alias(alias) => {
                    self.check_name(&alias.raw_name, alias.name_span);
                    if let Some(first) = defs.aliases.get(&alias.name)
                        && first.index != index
                    {
                        self.report_duplicate(&alias.name, first.name_span, alias.name_span);
                    }
                    self.check_attrs(&alias.attrs, false);
                }
                ItemSketch::Error => {}
            }
        }
    }

    /// Checks a declared name against keywords, reservations, and
    /// homoglyph lookalikes.
    fn check_name(&mut self, raw: &str, span: Span) {
        let name = strip_raw_ident(raw);
        let escaped = raw.starts_with("r#");
        if !escaped && DIALECT_KEYWORDS.contains(&name) {
            self.report(
                Diagnostic::error(span, format!("expected identifier, found keyword `{name}`"))
                    .with_code(codes::KEYWORD_AS_NAME)
                    .with_help(format!("escape it as `r#{name}` if the keyword was intended")),
            );
        }
        if RESERVED_OPENCL.contains(&name) {
            self.report(
                Diagnostic::warning(span, format!("`{name}` is reserved by OpenCL"))
                    .with_code(codes::RESERVED_IDENTIFIER)
                    .with_help("the OpenCL standard reserves this name; choose another"),
            );
        }
        if let Some(skeleton) = confusable_skeleton(name) {
            self.report(
                Diagnostic::warning(span, format!("identifier `{name}` contains lookalike characters"))
                    .with_code(codes::CONFUSABLE_IDENTIFIER)
                    .with_help(format!("did you mean `{skeleton}`?")),
            );
        }
    }

    /// Validates one item's `#[…]` attributes.
    fn check_attrs(&mut self, attrs: &[AttrSketch], is_fn: bool) {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for attr in attrs {
            let name = attr.name.as_str();
            if !is_fn {
                self.report(
                    Diagnostic::error(attr.span, "attribute not allowed on this item")
                        .with_code(codes::ATTRIBUTE_TARGET)
                        .with_help("attributes may only be applied to functions"),
                );
                continue;
            }
            if !KNOWN_ATTRIBUTES.contains(&name) {
                self.report(
                    Diagnostic::error(attr.span, format!("unknown attribute `{name}`"))
                        .with_code(codes::UNKNOWN_ATTRIBUTE)
                        .with_help("the dialect defines `#[kernel]`"),
                );
                continue;
            }
            if !seen.insert(name.to_string()) {
                self.report(
                    Diagnostic::error(attr.span, format!("duplicate attribute `#[{name}]`"))
                        .with_code(codes::DUPLICATE_ATTRIBUTE)
                        .with_help("remove the repeated attribute"),
                );
            }
        }
    }

    /// Reports a name that appears more than once in the same namespace.
    fn report_duplicate(&mut self, name: &str, first: Span, again: Span) {
        self.report(
            Diagnostic::error(again, format!("the name `{name}` is defined multiple times"))
                .with_code(codes::DUPLICATE_DEFINITION)
                .with_label(first, "first defined here"),
        );
    }

    /// Resolves every struct's fields.
    fn resolve_structs(&mut self) {
        let defs = self.defs;
        let names: Vec<String> = defs.structs.keys().cloned().collect();
        for name in names {
            let Some(def) = defs.structs.get(&name) else {
                continue;
            };
            let generics = def.generics.clone();
            let mut fields = Vec::new();
            let mut seen: BTreeMap<String, Span> = BTreeMap::new();
            for field in &def.fields {
                let field_name = strip_raw_ident(&field.raw_name).to_string();
                self.check_name(&field.raw_name, field.span);
                let ty = self.resolve_type(&field.ty, &generics);
                if matches!(ty, Ty::Void) {
                    self.report(
                        Diagnostic::error(field.span, "fields cannot have type `void`")
                            .with_code(codes::INVALID_VOID)
                            .with_help("remove the field or give it a value type"),
                    );
                }
                if let Some(&first) = seen.get(&field_name) {
                    self.report_duplicate(&field_name, first, field.span);
                } else {
                    seen.insert(field_name.clone(), field.span);
                }
                fields.push(FieldSig {
                    name: field_name,
                    span: field.span,
                    ty,
                });
            }
            self.struct_sigs.insert(
                name,
                StructSig {
                    index: def.index,
                    generics,
                    fields,
                    span: def.span,
                    name_span: def.name_span,
                },
            );
        }
    }

    /// Resolves every type alias, reporting alias cycles.
    fn resolve_aliases(&mut self) {
        let defs = self.defs;
        let names: Vec<String> = defs.aliases.keys().cloned().collect();
        for name in names {
            let Some(def) = defs.aliases.get(&name) else {
                continue;
            };
            let generics = def.generics.clone();
            let ty = self.resolve_type(&def.ty, &generics);
            self.alias_sigs.insert(
                name,
                AliasSig {
                    index: def.index,
                    ty,
                    span: def.span,
                    name_span: def.name_span,
                },
            );
        }
    }

    /// Resolves every function signature and applies kernel restrictions.
    fn resolve_signatures(&mut self) {
        let defs = self.defs;
        let names: Vec<String> = defs.fns.keys().cloned().collect();
        for name in names {
            let Some(def) = defs.fns.get(&name) else {
                continue;
            };
            let generics = def.generics.clone();
            let mut params = Vec::new();
            let mut seen: BTreeMap<String, Span> = BTreeMap::new();
            for param in &def.params {
                let ty = self.resolve_type(&param.ty, &generics);
                if def.kernel {
                    self.check_kernel_param(&ty, param.span);
                }
                if matches!(ty, Ty::Void) {
                    self.report(
                        Diagnostic::error(param.span, "parameters cannot have type `void`")
                            .with_code(codes::VOID_PARAM)
                            .with_help("remove the parameter or give it a value type"),
                    );
                }
                if let Some(bound) = &param.bound {
                    if let Some(&first) = seen.get(bound) {
                        self.report_duplicate(bound, first, param.span);
                    } else {
                        seen.insert(bound.clone(), param.span);
                    }
                }
                params.push(ParamSig {
                    name: param.bound.clone(),
                    ty,
                });
            }
            let ret = def
                .ret
                .as_ref()
                .map_or(Ty::Void, |ty| self.resolve_type(ty, &generics));
            if def.kernel && !matches!(ret, Ty::Void) {
                let span = def.ret.as_ref().map_or(def.span, |ty| ty.span);
                self.report(
                    Diagnostic::error(span, format!("kernel function `{name}` must return `void`"))
                        .with_code(codes::KERNEL_RETURN)
                        .with_help("kernels return nothing; remove the return type"),
                );
            }
            self.fn_sigs.insert(
                name,
                FnSig {
                    index: def.index,
                    generics,
                    params,
                    ret,
                    kernel: def.kernel,
                    span: def.span,
                    name_span: def.name_span,
                },
            );
        }
    }

    /// Rejects parameter shapes `#[kernel]` functions may not declare.
    fn check_kernel_param(&mut self, ty: &Ty, span: Span) {
        match ty {
            Ty::Ref { .. } => self.report(
                Diagnostic::error(span, "kernel parameters cannot be references")
                    .with_code(codes::KERNEL_PARAM)
                    .with_help("pass buffers as `*mut` pointers instead"),
            ),
            Ty::Array { len: None, .. } => self.report(
                Diagnostic::error(span, "kernel parameters cannot be slices")
                    .with_code(codes::KERNEL_PARAM)
                    .with_help("use a pointer or a fixed-size array"),
            ),
            _ => {}
        }
    }
}

/// Names of the generic parameters, lifetimes excluded.
fn generic_names(generics: &[GenericParam]) -> Vec<String> {
    generics
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type { name, .. } => Some(name.clone()),
            GenericParam::Lifetime { .. } => None,
        })
        .collect()
}

/// Span of `name` found after `keyword` inside `span`.
fn decl_name_span(source: &str, span: Span, keyword: &str, name: &str) -> Span {
    let Some(text) = source.get(span.offset as usize..span.end() as usize) else {
        return span;
    };
    let found = text.find(keyword).and_then(|at| {
        let after = at + keyword.len();
        text.get(after..)
            .and_then(|rest| rest.find(name))
            .map(|offset| after + offset)
    });
    match found {
        Some(offset) => Span::new(span.offset + offset as u32, name.len() as u32),
        None => span,
    }
}

/// The name written inside `#[…]`, without the brackets or arguments.
fn attr_name<'s>(source: &'s str, attr: &Attr) -> &'s str {
    let text = source
        .get(attr.span.offset as usize..attr.span.end() as usize)
        .unwrap_or("");
    let inner = text
        .strip_prefix("#[")
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(text);
    inner
        .split(|character: char| character.is_whitespace() || character == '(' || character == '=')
        .next()
        .unwrap_or("")
        .trim()
}

/// Scalar and helper type names offered as spelling suggestions.
const BUILTIN_TYPE_NAMES: &[&str] = &[
    "bool",
    "char",
    "short",
    "int",
    "long",
    "uchar",
    "ushort",
    "uint",
    "ulong",
    "half",
    "float",
    "double",
    "void",
    "size_t",
    "ptrdiff_t",
];

/// Where a pattern binding is being introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingOrigin {
    /// A function parameter; never reported as unused.
    Parameter,
    /// A local introduced by `let` or `for`.
    Local,
}

/// Sets the message on a diagnostic's primary label.
///
/// [`Diagnostic::error`] creates the primary label with an empty message;
/// the renderer prints it next to the caret, so most diagnostics fill it
/// in with the expected-versus-found detail.
fn primary(mut diagnostic: Diagnostic, message: impl Into<String>) -> Diagnostic {
    if let Some(label) = diagnostic.labels.first_mut() {
        label.message = message.into();
    }
    diagnostic
}

/// Evaluates an expression as a constant integer, `None` when it is not
/// constant arithmetic over integer literals.
fn const_int(expr: &Expr) -> Option<i128> {
    match &expr.kind {
        ExprKind::Literal {
            text,
            kind:
                TokenKind::Literal {
                    kind:
                        LiteralKind::Int {
                            base,
                            empty_int: false,
                        },
                    suffix_start,
                },
        } => {
            let split = (*suffix_start as usize).min(text.len());
            let (body, _) = text.split_at(split);
            int_digits_value(body, *base)
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => const_int(inner)?.checked_neg(),
        ExprKind::Binary { op, lhs, rhs } => {
            let left = const_int(lhs)?;
            let right = const_int(rhs)?;
            match op {
                BinaryOp::Add => left.checked_add(right),
                BinaryOp::Sub => left.checked_sub(right),
                BinaryOp::Mul => left.checked_mul(right),
                BinaryOp::Div if right != 0 => left.checked_div(right),
                BinaryOp::Rem if right != 0 => left.checked_rem(right),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Replaces literal placeholders with their default types.
///
/// An unsuffixed integer literal defaults to `int`, a float literal to
/// `float`, and the rule applies inside arrays and tuples so that
/// `let xs = [1, 2];` becomes `[int; 2]`.
fn default_literals(ty: &Ty) -> Ty {
    match ty {
        Ty::IntLit(_) => Ty::Scalar(Scalar::I32),
        Ty::FloatLit(_) => Ty::Scalar(Scalar::F32),
        Ty::Array { elem, len } => Ty::Array {
            elem: Box::new(default_literals(elem)),
            len: *len,
        },
        Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(default_literals).collect()),
        Ty::Ptr { mutable, inner } => Ty::Ptr {
            mutable: *mutable,
            inner: Box::new(default_literals(inner)),
        },
        Ty::Ref { mutable, inner } => Ty::Ref {
            mutable: *mutable,
            inner: Box::new(default_literals(inner)),
        },
        other => other.clone(),
    }
}

impl<'a> Sema<'a> {
    /// Resolves a type expression into the dialect's type representation.
    fn resolve_type(&mut self, ty: &Type, generics: &[String]) -> Ty {
        match &ty.kind {
            TypeKind::Error => Ty::Error,
            TypeKind::Ref { mutable, inner, .. } => {
                let resolved = self.resolve_type(inner, generics);
                if matches!(resolved, Ty::Void) {
                    self.report(
                        Diagnostic::error(ty.span, "cannot take a reference to `void`")
                            .with_code(codes::INVALID_VOID)
                            .with_help("references must point at a value type"),
                    );
                }
                Ty::Ref {
                    mutable: *mutable,
                    inner: Box::new(resolved),
                }
            }
            TypeKind::Ptr { mutable, inner } => Ty::Ptr {
                mutable: *mutable,
                inner: Box::new(self.resolve_type(inner, generics)),
            },
            TypeKind::Tuple(elems) => {
                let resolved: Vec<Ty> = elems
                    .iter()
                    .map(|elem| self.resolve_type(elem, generics))
                    .collect();
                if resolved
                    .iter()
                    .any(|elem| matches!(elem, Ty::Void))
                {
                    self.report(
                        Diagnostic::error(ty.span, "tuples cannot contain `void`")
                            .with_code(codes::INVALID_VOID),
                    );
                }
                Ty::Tuple(resolved)
            }
            TypeKind::Slice(inner) => {
                let resolved = self.resolve_type(inner, generics);
                if matches!(resolved, Ty::Void) {
                    self.report(
                        Diagnostic::error(ty.span, "slices cannot contain `void`")
                            .with_code(codes::INVALID_VOID),
                    );
                }
                Ty::Array {
                    elem: Box::new(resolved),
                    len: None,
                }
            }
            TypeKind::Array { elem, len } => {
                let resolved = self.resolve_type(elem, generics);
                if matches!(resolved, Ty::Void) {
                    self.report(
                        Diagnostic::error(ty.span, "arrays cannot contain `void`")
                            .with_code(codes::INVALID_VOID),
                    );
                }
                let length = self.const_array_len(len);
                Ty::Array {
                    elem: Box::new(resolved),
                    len: length,
                }
            }
            TypeKind::Path(path) => self.resolve_type_path(path, generics),
        }
    }

    /// Evaluates an array length expression, reporting bad constants.
    fn const_array_len(&mut self, expr: &Expr) -> Option<u32> {
        match const_int(expr) {
            Some(value) if value < 0 => {
                self.report(
                    Diagnostic::error(expr.span, "array length must not be negative")
                        .with_code(codes::INVALID_ARRAY_LENGTH),
                );
                None
            }
            Some(value) => match u32::try_from(value) {
                Ok(len) => Some(len),
                Err(_) => {
                    self.report(
                        Diagnostic::error(expr.span, "array length is too large")
                            .with_code(codes::INVALID_ARRAY_LENGTH),
                    );
                    None
                }
            },
            None => {
                self.report(
                    Diagnostic::error(expr.span, "array length must be a constant integer")
                        .with_code(codes::INVALID_ARRAY_LENGTH)
                        .with_help("use a literal or constant arithmetic such as `2 * 4`"),
                );
                None
            }
        }
    }

    /// Resolves a named type path against generics, builtins, and items.
    fn resolve_type_path(&mut self, path: &TypePath, generics: &[String]) -> Ty {
        if path.segments.len() > 1 {
            self.report(
                Diagnostic::error(path.span, "paths with `::` are not supported")
                    .with_code(codes::MULTI_SEGMENT_PATH)
                    .with_note("the dialect has no modules; write the name directly"),
            );
        }
        let Some(segment) = path.segments.last() else {
            return Ty::Error;
        };
        let name = strip_raw_ident(&segment.name);
        let args = self.resolve_generic_args(&path.args, generics);

        if generics.iter().any(|param| param == name) {
            return Ty::Generic(name.to_string());
        }

        match lookup_builtin(name) {
            Some(BuiltinType::Type(ty)) => {
                if !path.args.is_empty() {
                    self.report_generic_arity(name, 0, args.len(), path.span, None);
                    return Ty::Error;
                }
                return ty;
            }
            Some(BuiltinType::BadLanes(scalar)) => {
                self.report(
                    Diagnostic::error(path.span, format!("`{name}` is not a valid vector type"))
                        .with_code(codes::INVALID_VECTOR_LANES)
                        .with_label(segment.span, format!("`{}` has no lane count", scalar.as_str()))
                        .with_help("OpenCL defines vector lengths 2, 3, 4, 8, and 16"),
                );
                return Ty::Error;
            }
            None => {}
        }

        let defs = self.defs;
        if let Some(def) = defs.structs.get(name) {
            if !path.args.is_empty() && path.args.len() != def.generics.len() {
                self.report_generic_arity(
                    name,
                    def.generics.len(),
                    args.len(),
                    path.span,
                    Some(def.name_span),
                );
                return Ty::Error;
            }
            self.references.insert(name.to_string());
            return Ty::Struct {
                name: name.to_string(),
                args,
            };
        }

        if let Some(def) = defs.aliases.get(name) {
            if !path.args.is_empty() && path.args.len() != def.generics.len() {
                self.report_generic_arity(
                    name,
                    def.generics.len(),
                    args.len(),
                    path.span,
                    Some(def.name_span),
                );
                return Ty::Error;
            }
            self.references.insert(name.to_string());
            let declared = def.generics.clone();
            return self.expand_alias(name, def, &declared, &args, segment.span);
        }

        if self.defs.fns.contains_key(name) || lookup_builtin_fn(name).is_some() {
            self.report(
                Diagnostic::error(path.span, format!("`{name}` is a function, not a type"))
                    .with_code(codes::NOT_A_TYPE)
                    .with_help("call the function instead of naming its type"),
            );
            return Ty::Error;
        }

        let candidates = self.type_candidates(generics);
        let mut diagnostic = Diagnostic::error(path.span, format!("cannot find type `{name}` in this scope"))
            .with_code(codes::UNKNOWN_TYPE);
        if let Some(suggestion) = suggest(&candidates, name, 3) {
            diagnostic = diagnostic.with_help(format!("did you mean `{suggestion}`?"));
        }
        self.report(diagnostic);
        Ty::Error
    }

    /// Resolves each generic argument, dropping lifetime arguments.
    fn resolve_generic_args(&mut self, args: &[GenericArg], generics: &[String]) -> Vec<Ty> {
        args.iter()
            .filter_map(|arg| match arg {
                GenericArg::Type(ty) => Some(self.resolve_type(ty, generics)),
                GenericArg::Lifetime { .. } => None,
            })
            .collect()
    }

    /// Expands a type alias to its target, detecting alias cycles.
    fn expand_alias(
        &mut self,
        name: &str,
        def: &'a AliasSketch,
        declared: &[String],
        args: &[Ty],
        span: Span,
    ) -> Ty {
        if let Some(open) = self
            .resolving_aliases
            .iter()
            .position(|busy| busy == name)
        {
            let mut chain: Vec<&str> = self.resolving_aliases[open..]
                .iter()
                .map(String::as_str)
                .collect();
            chain.push(name);
            self.report(
                Diagnostic::error(span, format!("cycle detected while resolving `{name}`"))
                    .with_code(codes::ALIAS_CYCLE)
                    .with_note(format!("cycle: {}", chain.join(" -> "))),
            );
            return Ty::Error;
        }
        self.resolving_aliases.push(name.to_string());
        let target = self.resolve_type(&def.ty, declared);
        self.resolving_aliases.pop();
        if args.is_empty() {
            target
        } else {
            substitute(&target, declared, args)
        }
    }

    /// Reports the wrong number of generic arguments on a type path.
    fn report_generic_arity(
        &mut self,
        name: &str,
        expected: usize,
        found: usize,
        span: Span,
        declared: Option<Span>,
    ) {
        let expectation = if expected == 1 {
            String::from("1 argument")
        } else {
            format!("{expected} arguments")
        };
        let mut diagnostic = Diagnostic::error(span, format!("wrong number of type arguments for `{name}`"))
            .with_code(codes::GENERIC_ARITY);
        if let Some(declared) = declared {
            diagnostic = diagnostic.with_label(declared, "declared here");
        }
        self.report(primary(
            diagnostic,
            format!("expected {expectation}, found {found}"),
        ));
    }

    /// Every type name that could be meant, for spelling suggestions.
    fn type_candidates(&self, generics: &[String]) -> Vec<String> {
        let mut candidates = generics.to_vec();
        candidates.extend(self.defs.structs.keys().cloned());
        candidates.extend(self.defs.aliases.keys().cloned());
        for scalar in BUILTIN_TYPE_NAMES {
            candidates.push((*scalar).to_string());
            for lanes in ["2", "3", "4", "8", "16"] {
                let mut vector = String::from(*scalar);
                vector.push_str(lanes);
                candidates.push(vector);
            }
        }
        candidates
    }

    /// Type-checks one function body against its resolved signature.
    ///
    /// This is the single-body core shared by the whole-file run and
    /// the streaming [`analyze_body`](crate::analyze_body) entry point.
    fn check_body(&mut self, name: &str, sig: &FnSig, params: &[Param], body: &Block) {
        self.ctx = Some(FnCtx {
            name: name.to_string(),
            ret: sig.ret.clone(),
            loop_depth: 0,
            generics: sig.generics.clone(),
        });
        self.scopes.clear();
        self.scopes.push(Scope::default());
        for (param, param_sig) in params.iter().zip(sig.params.iter()) {
            let ty = param_sig.ty.clone();
            self.bind_pattern(&param.pat, ty, BindingOrigin::Parameter);
        }
        let body_ty = self.check_block(body);
        let mismatch_span = body
            .tail
            .as_ref()
            .map_or(body.span, |tail| tail.span);
        self.expect_coerce(mismatch_span, &body_ty, &sig.ret);
        self.pop_scope();
        self.ctx = None;
    }

    /// Introduces a pattern's bindings into the innermost scope.
    fn bind_pattern(&mut self, pat: &Pat, ty: Ty, origin: BindingOrigin) {
        let PatKind::Ident { name, mutable } = &pat.kind else {
            return;
        };
        let stripped = strip_raw_ident(name);
        self.check_name(name, pat.span);
        let state = match origin {
            BindingOrigin::Parameter => BindingState::Parameter { mutable: *mutable },
            BindingOrigin::Local => BindingState::Local {
                mutable: *mutable,
                used: stripped.starts_with('_'),
            },
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.push(Binding {
                name: stripped.to_string(),
                ty,
                span: pat.span,
                pat: pat.id,
                state,
            });
        }
    }

    /// Pops a scope, reporting the locals it leaves unused.
    fn pop_scope(&mut self) {
        let Some(scope) = self.scopes.pop() else {
            return;
        };
        if self.has_errors() {
            return;
        }
        for binding in scope.bindings {
            if binding.state.is_unused() && !binding.name.starts_with('_') {
                self.report(
                    Diagnostic::warning(binding.span, format!("unused variable: `{}`", binding.name))
                        .with_code(codes::UNUSED_VARIABLE)
                        .with_help("consider prefixing the name with an underscore"),
                );
            }
        }
    }

    /// Type-checks a block, returning the type of its tail expression.
    fn check_block(&mut self, block: &Block) -> Ty {
        self.scopes.push(Scope::default());
        let mut last = Ty::Void;
        for stmt in &block.stmts {
            last = self.check_stmt(stmt);
        }
        let ty = match &block.tail {
            Some(expr) => self.check_expr(expr),
            None if last.is_never() => Ty::Never,
            None => Ty::Void,
        };
        self.pop_scope();
        ty
    }

    /// Type-checks one statement, returning the value it produces.
    fn check_stmt(&mut self, stmt: &Stmt) -> Ty {
        match &stmt.kind {
            StmtKind::Empty | StmtKind::Error => Ty::Void,
            StmtKind::Expr(expr) => self.check_expr(expr),
            StmtKind::Let(binding) => {
                let generics = self.current_generics();
                let annotated = binding
                    .ty
                    .as_ref()
                    .map(|ty| self.resolve_type(ty, &generics));
                let init = binding
                    .init
                    .as_ref()
                    .map(|expr| self.check_expr(expr));
                let ty = match (annotated, init) {
                    (Some(expected), Some(actual)) => {
                        let span = binding
                            .init
                            .as_ref()
                            .map_or(binding.span, |expr| expr.span);
                        self.expect_coerce(span, &actual, &expected);
                        expected
                    }
                    (Some(expected), None) => expected,
                    (None, Some(Ty::Error)) => Ty::Error,
                    (None, Some(actual)) => {
                        if matches!(actual, Ty::Void) {
                            self.report(
                                Diagnostic::error(
                                    binding.pat.span,
                                    "type annotations needed for this binding",
                                )
                                .with_code(codes::MISSING_TYPE_ANNOTATION)
                                .with_help("annotate the binding, for example `let x: int = …`"),
                            );
                            Ty::Error
                        } else {
                            default_literals(&actual)
                        }
                    }
                    (None, None) => {
                        self.report(
                            Diagnostic::error(binding.pat.span, "type annotations needed for this binding")
                                .with_code(codes::MISSING_TYPE_ANNOTATION)
                                .with_help("declare the type, for example `let x: int;`"),
                        );
                        Ty::Error
                    }
                };
                if matches!(ty, Ty::Void) {
                    self.report(
                        Diagnostic::error(binding.pat.span, "bindings cannot have type `void`")
                            .with_code(codes::INVALID_VOID),
                    );
                    self.bind_pattern(&binding.pat, Ty::Error, BindingOrigin::Local);
                } else {
                    self.bind_pattern(&binding.pat, ty, BindingOrigin::Local);
                }
                Ty::Void
            }
        }
    }

    /// Generic parameters in scope in the body being checked.
    fn current_generics(&self) -> Vec<String> {
        self.ctx
            .as_ref()
            .map(|ctx| ctx.generics.clone())
            .unwrap_or_default()
    }

    /// Type-checks an expression, reporting every problem it finds.
    ///
    /// The resulting type is recorded in the [`TypeTable`] under the
    /// expression's [`NodeId`]; children are checked through this
    /// wrapper too, so a checked subtree fills the table bottom-up.
    fn check_expr(&mut self, expr: &Expr) -> Ty {
        let ty = self.check_expr_inner(expr);
        self.types.insert(expr.id, ty.clone());
        ty
    }

    /// The type-checking body behind [`Self::check_expr`].
    fn check_expr_inner(&mut self, expr: &Expr) -> Ty {
        match &expr.kind {
            ExprKind::Error => Ty::Error,
            ExprKind::Literal { text, kind } => self.literal_type(expr.span, text, *kind),
            ExprKind::Bool(_) => Ty::bool(),
            ExprKind::Path(path) => self.path_value(path, expr),
            ExprKind::Unary { op, expr: inner } => self.unary_type(*op, expr, inner),
            ExprKind::Binary { op, lhs, rhs } => {
                let left = self.check_expr(lhs);
                let right = self.check_expr(rhs);
                self.binary_result(expr.span, *op, lhs.span, rhs.span, &left, &right)
            }
            ExprKind::Assign { op, lhs, rhs } => {
                let target = self.check_expr(lhs);
                let value = self.check_expr(rhs);
                match op {
                    Some(op) => {
                        self.binary_result(expr.span, *op, lhs.span, rhs.span, &target, &value);
                    }
                    None => {
                        self.check_assignable(lhs);
                        self.expect_coerce(rhs.span, &value, &target);
                    }
                }
                Ty::Void
            }
            ExprKind::Call { callee, args } => self.check_call(expr, callee, args),
            ExprKind::Index { expr: base, index } => self.index_type(base, index),
            ExprKind::Field { expr: base, name } => self.field_type(expr, base, name),
            ExprKind::Try { expr: inner } => {
                self.report(
                    Diagnostic::error(expr.span, "the `?` operator is not supported")
                        .with_code(codes::TRY_NOT_SUPPORTED)
                        .with_note("the dialect has no `Result` or `Option` type")
                        .with_help("handle the failure with an explicit `if`"),
                );
                let _ = self.check_expr(inner);
                Ty::Error
            }
            ExprKind::Cast { expr: inner, ty } => {
                let from = self.check_expr(inner);
                let generics = self.current_generics();
                let target = self.resolve_type(ty, &generics);
                if !castable(&from, &target) {
                    let detail = MessageBuilder::new()
                        .text("cannot cast `")
                        .ty(&from)
                        .text("` as `")
                        .ty(&target)
                        .text("`")
                        .finish();
                    let mut diagnostic = primary(
                        Diagnostic::error(expr.span, "invalid cast").with_code(codes::INVALID_CAST),
                        detail,
                    );
                    diagnostic = if matches!(target, Ty::Scalar(Scalar::Bool)) {
                        diagnostic.with_help("compare the value with `!= 0` instead")
                    } else {
                        diagnostic.with_help("only numeric and pointer casts are supported")
                    };
                    self.report(diagnostic);
                    return Ty::Error;
                }
                target
            }
            ExprKind::Range { .. } => {
                self.report(
                    Diagnostic::error(
                        expr.span,
                        "range expressions are only allowed in a `for` loop header",
                    )
                    .with_code(codes::RANGE_NOT_ALLOWED)
                    .with_help("iterate with `for x in a..b`"),
                );
                Ty::Error
            }
            ExprKind::Block(block) => self.check_block(block),
            ExprKind::If {
                cond,
                then,
                else_branch,
            } => self.if_type(expr, cond, then, else_branch.as_deref()),
            ExprKind::While { cond, body } => {
                self.check_condition(cond);
                self.enter_loop();
                let _ = self.check_block(body);
                self.exit_loop();
                Ty::Void
            }
            ExprKind::Loop { body } => {
                self.enter_loop();
                let _ = self.check_block(body);
                self.exit_loop();
                Ty::Void
            }
            ExprKind::For { pat, iter, body } => {
                let iter_ty = self.iter_type(iter);
                self.scopes.push(Scope::default());
                self.bind_pattern(pat, iter_ty, BindingOrigin::Local);
                self.enter_loop();
                let _ = self.check_block(body);
                self.exit_loop();
                self.pop_scope();
                Ty::Void
            }
            ExprKind::Return { expr: value } => self.return_type(expr, value.as_deref()),
            ExprKind::Break { expr: value } => {
                if let Some(value) = value {
                    let _ = self.check_expr(value);
                }
                if self.loop_depth() == 0 {
                    self.report(
                        Diagnostic::error(expr.span, "cannot `break` outside of a loop")
                            .with_code(codes::BREAK_OUTSIDE_LOOP)
                            .with_help("`break` is only valid inside `loop`, `while`, and `for`"),
                    );
                }
                Ty::Never
            }
            ExprKind::Continue => {
                if self.loop_depth() == 0 {
                    self.report(
                        Diagnostic::error(expr.span, "cannot `continue` outside of a loop")
                            .with_code(codes::CONTINUE_OUTSIDE_LOOP)
                            .with_help("`continue` is only valid inside `loop`, `while`, and `for`"),
                    );
                }
                Ty::Never
            }
            ExprKind::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|elem| self.check_expr(elem))
                    .collect(),
            ),
            ExprKind::Array(elems) => self.array_type(expr, elems),
        }
    }

    /// Assigns a type to a literal token, mirroring the lexical tables.
    fn literal_type(&mut self, span: Span, text: &str, kind: TokenKind) -> Ty {
        let TokenKind::Literal {
            kind: literal,
            suffix_start,
        } = kind
        else {
            return Ty::Error;
        };
        let split = (suffix_start as usize).min(text.len());
        let (body, suffix) = text.split_at(split);
        match literal {
            LiteralKind::Int {
                base,
                empty_int: true,
            } => {
                let _ = (base, span);
                Ty::Error
            }
            LiteralKind::Int { base, .. } => {
                if suffix.is_empty() {
                    return int_digits_value(body, base).map_or(Ty::Error, Ty::IntLit);
                }
                if let Some(scalar) = int_scalar_for_suffix(suffix) {
                    return Ty::Scalar(scalar);
                }
                if float_scalar_for_suffix(suffix).is_some() {
                    return Ty::Scalar(Scalar::F32);
                }
                Ty::Error
            }
            LiteralKind::Float {
                base,
                empty_exponent: true,
            } => {
                let _ = (base, span);
                Ty::Error
            }
            LiteralKind::Float { base, .. } => {
                if let Some(scalar) = float_scalar_for_suffix(suffix) {
                    return Ty::Scalar(scalar);
                }
                if !suffix.is_empty() {
                    return Ty::Error;
                }
                if base == Base::Hexadecimal {
                    return Ty::FloatLit(None);
                }
                Ty::FloatLit(float_value(body))
            }
            LiteralKind::Char { .. } => {
                if !suffix.is_empty() {
                    return Ty::Error;
                }
                let inner = strip_quotes(body);
                let Ok(decoded) = decode_escapes(inner) else {
                    return Ty::Error;
                };
                let mut chars = decoded.chars();
                match (chars.next(), chars.next()) {
                    (Some(character), None) => Ty::IntLit(i128::from(character as u32)),
                    _ => Ty::Error,
                }
            }
            LiteralKind::Str { .. } | LiteralKind::RawStr { .. } => {
                self.report(
                    Diagnostic::error(span, "string literals are not supported")
                        .with_code(codes::UNSUPPORTED_STRING)
                        .with_note("the dialect has no string type")
                        .with_help("use an array of `char` instead"),
                );
                Ty::Error
            }
        }
    }

    /// Types a path used as a value: bindings, then items, then error.
    ///
    /// A successful binding lookup is recorded in the
    /// [`ResolutionTable`] under `expr.id`.
    fn path_value(&mut self, path: &Path, expr: &Expr) -> Ty {
        if path.segments.len() > 1 {
            self.report(
                Diagnostic::error(path.span, "paths with `::` are not supported")
                    .with_code(codes::MULTI_SEGMENT_PATH)
                    .with_note("the dialect has no modules; write the name directly"),
            );
            return Ty::Error;
        }
        let Some(segment) = path.segments.last() else {
            return Ty::Error;
        };
        let name = strip_raw_ident(&segment.name);

        if let Some((_, ty, _, binding)) = self.binding_of(name) {
            self.mark_binding_used(name);
            self.resolutions
                .insert(expr.id, Res::Local { binding });
            return ty;
        }

        if self.defs.fns.contains_key(name) || lookup_builtin_fn(name).is_some() {
            self.references.insert(name.to_string());
            self.report(
                primary(
                    Diagnostic::error(path.span, format!("`{name}` is a function"))
                        .with_code(codes::UNEXPECTED_FUNCTION),
                    format!("`{name}` used as a value"),
                )
                .with_help(format!("call the function: `{name}(…)`")),
            );
            return Ty::Error;
        }

        if self.defs.structs.contains_key(name)
            || self.defs.aliases.contains_key(name)
            || lookup_builtin(name).is_some()
            || generics_contain(&self.current_generics(), name)
        {
            self.report(
                Diagnostic::error(path.span, format!("expected a value, found type `{name}`"))
                    .with_code(codes::UNEXPECTED_TYPE),
            );
            return Ty::Error;
        }

        let candidates = self.value_candidates();
        let mut diagnostic =
            Diagnostic::error(path.span, format!("cannot find value `{name}` in this scope"))
                .with_code(codes::UNDEFINED_NAME);
        if let Some(suggestion) = suggest(&candidates, name, 3) {
            diagnostic = diagnostic.with_help(format!("did you mean `{suggestion}`?"));
        }
        self.report(diagnostic);
        Ty::Error
    }

    /// Types a unary expression, reporting invalid operands.
    fn unary_type(&mut self, op: UnaryOp, whole: &Expr, inner: &Expr) -> Ty {
        let operand = self.check_expr(inner);
        match op {
            UnaryOp::Neg => {
                if !is_numeric(&operand) {
                    return self.report_bad_operand(
                        whole.span,
                        inner.span,
                        "-",
                        &operand,
                        "numeric types only",
                    );
                }
                operand
            }
            UnaryOp::Not => {
                if !is_bool(&operand) && !is_int_like(&operand) {
                    return self.report_bad_operand(
                        whole.span,
                        inner.span,
                        "!",
                        &operand,
                        "expected a `bool` or an integer",
                    );
                }
                operand
            }
            UnaryOp::Deref => match &operand {
                Ty::Ptr { inner: pointee, .. } | Ty::Ref { inner: pointee, .. } => (**pointee).clone(),
                Ty::Error | Ty::Never => Ty::Error,
                other => {
                    let other = other.clone();
                    self.report_bad_operand(
                        whole.span,
                        inner.span,
                        "*",
                        &other,
                        "only pointers and references can be dereferenced",
                    )
                }
            },
            UnaryOp::AddrOf { mutable } => {
                if mutable {
                    self.check_borrow_is_mutable(inner);
                }
                Ty::Ref {
                    mutable,
                    inner: Box::new(operand),
                }
            }
        }
    }

    /// Reports `&mut x` where `x` is not declared `mut`.
    fn check_borrow_is_mutable(&mut self, inner: &Expr) {
        let ExprKind::Path(path) = &inner.kind else {
            return;
        };
        if path.segments.len() != 1 {
            return;
        }
        let Some(segment) = path.segments.last() else {
            return;
        };
        let name = strip_raw_ident(&segment.name);
        let Some((mutable, _, span, _)) = self.binding_of(name) else {
            return;
        };
        if !mutable {
            self.report(
                primary(
                    Diagnostic::error(inner.span, format!("cannot borrow `{name}` as mutable"))
                        .with_code(codes::ASSIGN_TO_IMMUTABLE)
                        .with_label(span, "declared here without `mut`"),
                    format!("`{name}` is immutable"),
                )
                .with_help(format!("declare it as `let mut {name} = …`")),
            );
        }
    }

    /// Reports a unary operator applied to the wrong type.
    fn report_bad_operand(&mut self, span: Span, operand_span: Span, op: &str, ty: &Ty, help: &str) -> Ty {
        let detail = MessageBuilder::new()
            .text(format!("`{op}` cannot be applied to `"))
            .ty(ty)
            .text("`")
            .finish();
        self.report(
            primary(
                Diagnostic::error(span, "invalid unary operand")
                    .with_code(codes::INVALID_UNARY_OPERAND)
                    .with_label(operand_span, format!("this is `{ty}`")),
                detail,
            )
            .with_help(help),
        );
        Ty::Error
    }

    /// Types a binary or compound-assignment expression.
    fn binary_result(
        &mut self,
        span: Span,
        op: BinaryOp,
        lhs_span: Span,
        rhs_span: Span,
        left: &Ty,
        right: &Ty,
    ) -> Ty {
        use BinaryOp::*;
        if left.is_error() || right.is_error() || left.is_never() || right.is_never() {
            return Ty::Error;
        }
        match op {
            Add | Sub | Mul | Div | Rem => {
                if !is_numeric(left) || !is_numeric(right) {
                    let help = if is_bool(left) || is_bool(right) {
                        Some("arithmetic is not defined for `bool`")
                    } else {
                        None
                    };
                    return self.report_bad_operands(span, lhs_span, rhs_span, op, left, right, help);
                }
                self.unify_operands(span, lhs_span, rhs_span, op, left, right)
            }
            Shl | Shr => {
                if !is_int_like(left) || !is_int_like(right) {
                    let help = if is_float(left) || is_float(right) {
                        Some("shift operators require integer operands")
                    } else {
                        None
                    };
                    return self.report_bad_operands(span, lhs_span, rhs_span, op, left, right, help);
                }
                if matches!(left, Ty::IntLit(_)) && !matches!(right, Ty::IntLit(_)) {
                    right.clone()
                } else {
                    left.clone()
                }
            }
            BitAnd | BitXor | BitOr => {
                if is_bool(left) && is_bool(right) {
                    return Ty::bool();
                }
                if !is_int_like(left) || !is_int_like(right) {
                    let help = if is_float(left) || is_float(right) {
                        Some("bitwise operators require integer operands")
                    } else {
                        None
                    };
                    return self.report_bad_operands(span, lhs_span, rhs_span, op, left, right, help);
                }
                self.unify_operands(span, lhs_span, rhs_span, op, left, right)
            }
            Eq | Ne => {
                if is_bool(left) && is_bool(right) || is_numeric(left) && is_numeric(right) {
                    return Ty::bool();
                }
                if coerce(left, right) || coerce(right, left) {
                    return Ty::bool();
                }
                self.report_bad_operands(
                    span,
                    lhs_span,
                    rhs_span,
                    op,
                    left,
                    right,
                    Some("values of different types cannot be compared"),
                )
            }
            Lt | Le | Gt | Ge => {
                if is_numeric(left) && is_numeric(right) || is_bool(left) && is_bool(right) {
                    let unified = self.unify_operands(span, lhs_span, rhs_span, op, left, right);
                    return if unified.is_error() { unified } else { Ty::bool() };
                }
                if matches!(left, Ty::Ptr { .. }) && matches!(right, Ty::Ptr { .. }) {
                    return Ty::bool();
                }
                self.report_bad_operands(
                    span,
                    lhs_span,
                    rhs_span,
                    op,
                    left,
                    right,
                    Some("ordered comparison needs numeric operands"),
                )
            }
            And | Or => {
                if is_bool(left) && is_bool(right) {
                    return Ty::bool();
                }
                let help = if is_numeric(left) || is_numeric(right) {
                    Some("logical operators need `bool`; compare with `!= 0`")
                } else {
                    None
                };
                self.report_bad_operands(span, lhs_span, rhs_span, op, left, right, help)
            }
        }
    }

    /// Unifies two operands of a comparison or arithmetic operator.
    fn unify_operands(
        &mut self,
        span: Span,
        lhs_span: Span,
        rhs_span: Span,
        op: BinaryOp,
        left: &Ty,
        right: &Ty,
    ) -> Ty {
        if coerce(left, right) {
            right.clone()
        } else if coerce(right, left) {
            left.clone()
        } else {
            self.report_bad_operands(
                span,
                lhs_span,
                rhs_span,
                op,
                left,
                right,
                Some("operands must have the same or compatible types"),
            )
        }
    }

    /// Reports a binary operator that its operands do not support.
    #[allow(
        clippy::too_many_arguments,
        reason = "spans, operator, operands, and help read better as arguments"
    )]
    fn report_bad_operands(
        &mut self,
        span: Span,
        lhs_span: Span,
        rhs_span: Span,
        op: BinaryOp,
        left: &Ty,
        right: &Ty,
        help: Option<&str>,
    ) -> Ty {
        let symbol = op_symbol(op);
        let mut diagnostic =
            Diagnostic::error(span, format!("cannot apply `{symbol}` to `{left}` and `{right}`"))
                .with_code(codes::INVALID_BINARY_OPERAND)
                .with_label(lhs_span, format!("this is `{left}`"))
                .with_label(rhs_span, format!("this is `{right}`"));
        if let Some(help) = help {
            diagnostic = diagnostic.with_help(help);
        }
        self.report(primary(diagnostic, format!("`{symbol}` is not defined here")));
        Ty::Error
    }

    /// Types an `if` expression and unifies its branches.
    fn if_type(&mut self, whole: &Expr, cond: &Expr, then: &Block, else_branch: Option<&Expr>) -> Ty {
        self.check_condition(cond);
        let then_ty = self.check_block(then);
        let Some(else_expr) = else_branch else {
            return then_ty;
        };
        let else_ty = self.check_expr(else_expr);
        self.unify_branches(whole, then.span, else_expr.span, &then_ty, &else_ty)
    }

    /// Chooses the type of an `if`/`else`, reporting incompatible arms.
    fn unify_branches(
        &mut self,
        whole: &Expr,
        then_span: Span,
        else_span: Span,
        then_ty: &Ty,
        else_ty: &Ty,
    ) -> Ty {
        if then_ty.is_error() || else_ty.is_error() {
            return Ty::Error;
        }
        if then_ty.is_never() {
            return else_ty.clone();
        }
        if else_ty.is_never() {
            return then_ty.clone();
        }
        if then_ty.is_unit_like() && else_ty.is_unit_like() {
            return Ty::Void;
        }
        if coerce(then_ty, else_ty) {
            return else_ty.clone();
        }
        if coerce(else_ty, then_ty) {
            return then_ty.clone();
        }
        self.report(
            primary(
                Diagnostic::error(whole.span, "mismatched types in `if` and `else`")
                    .with_code(codes::MISMATCHED_TYPES)
                    .with_label(then_span, format!("`{then_ty}`"))
                    .with_label(else_span, format!("`{else_ty}`")),
                format!("the branches have incompatible types: `{then_ty}` and `{else_ty}`"),
            )
            .with_help("both branches must produce the same type"),
        );
        Ty::Error
    }

    /// Types a call: user functions first, then builtins, then errors.
    fn check_call(&mut self, whole: &Expr, callee: &Expr, args: &[Expr]) -> Ty {
        let path = match &callee.kind {
            ExprKind::Path(path) => path,
            ExprKind::Field { .. } => {
                self.report(
                    Diagnostic::error(callee.span, "method calls are not supported")
                        .with_code(codes::EXPECTED_FUNCTION)
                        .with_note("the dialect has no methods; call a free function"),
                );
                self.check_args_only(args);
                return Ty::Error;
            }
            _ => {
                self.report(
                    Diagnostic::error(callee.span, "expected a function").with_code(codes::EXPECTED_FUNCTION),
                );
                self.check_args_only(args);
                return Ty::Error;
            }
        };
        if path.segments.len() > 1 {
            self.report(
                Diagnostic::error(callee.span, "paths with `::` are not supported")
                    .with_code(codes::MULTI_SEGMENT_PATH)
                    .with_note("the dialect has no modules; write the name directly"),
            );
            self.check_args_only(args);
            return Ty::Error;
        }
        let Some(segment) = path.segments.last() else {
            self.check_args_only(args);
            return Ty::Error;
        };
        let name = strip_raw_ident(&segment.name);
        let args_span = call_args_span(whole, callee);

        if let Some((_, ty, span, _)) = self.binding_of(name) {
            self.mark_binding_used(name);
            self.report(
                primary(
                    Diagnostic::error(callee.span, format!("`{name}` is not a function"))
                        .with_code(codes::EXPECTED_FUNCTION)
                        .with_label(span, format!("`{name}` has type `{ty}`")),
                    format!("expected a function, found `{ty}`"),
                )
                .with_help("the dialect has no function values"),
            );
            self.check_args_only(args);
            return Ty::Error;
        }

        if let Some(sig) = self.fn_sigs.get(name).cloned() {
            self.resolutions.insert(
                callee.id,
                Res::Function {
                    name: name.to_string(),
                },
            );
            self.references.insert(name.to_string());
            if let Some(current) = self.ctx.as_ref().map(|ctx| ctx.name.clone()) {
                self.calls
                    .entry(current)
                    .or_default()
                    .push((name.to_string(), callee.span));
            }
            let is_self = self
                .ctx
                .as_ref()
                .is_some_and(|ctx| ctx.name == name);
            if sig.kernel && !is_self {
                self.report(
                    Diagnostic::error(
                        callee.span,
                        format!("cannot call kernel `{name}` from another function"),
                    )
                    .with_code(codes::KERNEL_CALL)
                    .with_note("kernels are entry points invoked by the host"),
                );
            }
            if args.len() != sig.params.len() {
                self.report(arity_diagnostic(args_span, sig.params.len(), args.len()));
            }
            let arg_types: Vec<Ty> = args
                .iter()
                .map(|arg| self.check_expr(arg))
                .collect();
            let mut substitutions: Vec<(String, Ty)> = Vec::new();
            for (index, param) in sig.params.iter().enumerate() {
                let (Some(actual), Some(arg)) = (arg_types.get(index), args.get(index)) else {
                    continue;
                };
                if !unify(&param.ty, actual, &mut substitutions) {
                    self.report_mismatch(arg.span, actual, &param.ty);
                }
            }
            if sig.generics.is_empty() {
                return sig.ret.clone();
            }
            let resolved: Vec<Ty> = sig
                .generics
                .iter()
                .map(|parameter| {
                    substitutions
                        .iter()
                        .find(|(found, _)| found == parameter)
                        .map_or_else(|| Ty::Generic(parameter.clone()), |(_, ty)| ty.clone())
                })
                .collect();
            return substitute(&sig.ret, &sig.generics, &resolved);
        }

        if let Some(builtin) = lookup_builtin_fn(name) {
            self.resolutions.insert(
                callee.id,
                Res::Builtin {
                    name: name.to_string(),
                },
            );
            let arg_types: Vec<Ty> = args
                .iter()
                .map(|arg| self.check_expr(arg))
                .collect();
            return self.check_builtin(builtin, args, &arg_types, args_span);
        }

        if self.defs.structs.contains_key(name)
            || self.defs.aliases.contains_key(name)
            || lookup_builtin(name).is_some()
        {
            self.report(
                Diagnostic::error(callee.span, format!("expected a function, found type `{name}`"))
                    .with_code(codes::EXPECTED_FUNCTION),
            );
            self.check_args_only(args);
            return Ty::Error;
        }

        let candidates = self.value_candidates();
        let mut diagnostic = Diagnostic::error(
            callee.span,
            format!("cannot find function `{name}` in this scope"),
        )
        .with_code(codes::UNDEFINED_NAME);
        if let Some(suggestion) = suggest(&candidates, name, 3) {
            diagnostic = diagnostic.with_help(format!("did you mean `{suggestion}`?"));
        }
        self.report(diagnostic);
        self.check_args_only(args);
        Ty::Error
    }

    /// Type-checks arguments that have no function to check against.
    fn check_args_only(&mut self, args: &[Expr]) {
        for arg in args {
            let _ = self.check_expr(arg);
        }
    }

    /// Checks a built-in call against its [`BuiltinKind`].
    fn check_builtin(&mut self, builtin: &Builtin, args: &[Expr], arg_types: &[Ty], args_span: Span) -> Ty {
        match &builtin.kind {
            BuiltinKind::WorkItem => {
                if arg_types.len() != 1 {
                    self.report(arity_diagnostic(args_span, 1, arg_types.len()));
                }
                let dimension = Ty::Scalar(Scalar::I32);
                if let (Some(arg), Some(actual)) = (args.first(), arg_types.first())
                    && !unify(&dimension, actual, &mut Vec::new())
                {
                    self.report_mismatch(arg.span, actual, &dimension);
                }
                Ty::Scalar(Scalar::U64)
            }
            BuiltinKind::Fixed { params, ret } => {
                if arg_types.len() != params.len() {
                    self.report(arity_diagnostic(args_span, params.len(), arg_types.len()));
                }
                for (index, expected) in params.iter().enumerate() {
                    if let (Some(arg), Some(actual)) = (args.get(index), arg_types.get(index)) {
                        self.expect_coerce(arg.span, actual, expected);
                    }
                }
                ret.clone()
            }
            BuiltinKind::Reduce { arity } => {
                if arg_types.len() != *arity as usize {
                    self.report(arity_diagnostic(args_span, *arity as usize, arg_types.len()));
                }
                if let (Some(arg), Some(actual)) = (args.first(), arg_types.first())
                    && !is_bool(actual)
                    && !actual.is_error()
                {
                    self.report(primary(
                        Diagnostic::error(
                            arg.span,
                            format!("`{}` expects a boolean value, found `{actual}`", builtin.name),
                        )
                        .with_code(codes::MISMATCHED_TYPES),
                        format!("expected `bool`, found `{actual}`"),
                    ));
                }
                Ty::bool()
            }
            BuiltinKind::Numeric { arity } | BuiltinKind::Float { arity } => {
                self.shape_builtin(builtin, *arity, args, arg_types, args_span)
            }
        }
    }

    /// Checks a shape-preserving builtin: `abs`, `min`, `sqrt`, `mix`, …
    ///
    /// The result keeps the shape of the widest argument; `Float` kinds
    /// additionally promote integer shapes to `float`.
    fn shape_builtin(
        &mut self,
        builtin: &Builtin,
        arity: u8,
        args: &[Expr],
        arg_types: &[Ty],
        args_span: Span,
    ) -> Ty {
        if arg_types.len() != arity as usize {
            self.report(arity_diagnostic(args_span, arity as usize, arg_types.len()));
        }
        let Some(first) = arg_types.first() else {
            return Ty::Error;
        };
        let mut shape = first.clone();
        for (index, actual) in arg_types.iter().enumerate().skip(1) {
            if coerce(actual, &shape) {
                continue;
            }
            if coerce(&shape, actual) {
                shape = actual.clone();
                continue;
            }
            if let Some(arg) = args.get(index) {
                self.report_mismatch(arg.span, actual, &shape);
            }
        }
        if !is_numeric(&shape) && !shape.is_error() {
            if let Some(arg) = args.first() {
                self.report(
                    Diagnostic::error(arg.span, format!("`{}` expects numeric arguments", builtin.name))
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_label(arg.span, format!("found `{shape}`")),
                );
            }
            return Ty::Error;
        }
        let input = default_literals(&shape);
        let promoted = if matches!(builtin.kind, BuiltinKind::Float { .. }) {
            promote_float(&input)
        } else {
            input.clone()
        };
        for (index, actual) in arg_types.iter().enumerate().skip(1) {
            if let Some(arg) = args.get(index) {
                self.expect_coerce(arg.span, actual, &input);
            }
        }
        promoted
    }

    /// Types a field access, tuple index, or vector swizzle.
    fn field_type(&mut self, whole: &Expr, base: &Expr, name: &str) -> Ty {
        let base_ty = self.check_expr(base);
        let field = strip_raw_ident(name);
        match strip_indirection(&base_ty) {
            Ty::Struct {
                name: struct_name,
                args,
            } => {
                let found = self.struct_sigs.get(struct_name).and_then(|sig| {
                    let generics = sig.generics.clone();
                    sig.fields
                        .iter()
                        .find(|candidate| candidate.name == field)
                        .map(|candidate| substitute(&candidate.ty, &generics, args))
                });
                match found {
                    Some(ty) => ty,
                    None => self.report_unknown_field(whole, base, struct_name, field),
                }
            }
            Ty::Vector { elem, lanes } => self.swizzle(whole.span, *elem, *lanes, field),
            Ty::Tuple(elems) => match field.parse::<usize>() {
                Ok(index) => match elems.get(index) {
                    Some(ty) => ty.clone(),
                    None => {
                        self.report(
                            Diagnostic::error(whole.span, format!("tuple index `{index}` is out of range"))
                                .with_code(codes::UNKNOWN_FIELD)
                                .with_label(whole.span, format!("the tuple has {} elements", elems.len())),
                        );
                        Ty::Error
                    }
                },
                Err(_) => {
                    self.report(
                        Diagnostic::error(whole.span, format!("tuple field `{field}` does not exist"))
                            .with_code(codes::UNKNOWN_FIELD)
                            .with_help("tuples are indexed numerically: `.0`, `.1`, …"),
                    );
                    Ty::Error
                }
            },
            Ty::Error | Ty::Never => Ty::Error,
            other => {
                let other = other.clone();
                self.report(
                    primary(
                        Diagnostic::error(whole.span, format!("type `{other}` has no field `{field}`"))
                            .with_code(codes::UNKNOWN_FIELD)
                            .with_label(base.span, format!("`{other}`")),
                        format!("`{other}` is not a struct"),
                    )
                    .with_help("only structs, tuples, and vectors have fields"),
                );
                Ty::Error
            }
        }
    }

    /// Reports a field the struct does not declare, with a suggestion.
    fn report_unknown_field(&mut self, whole: &Expr, base: &Expr, struct_name: &str, field: &str) -> Ty {
        let available: Vec<String> = self
            .struct_sigs
            .get(struct_name)
            .map(|sig| {
                sig.fields
                    .iter()
                    .map(|entry| entry.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        let mut diagnostic = primary(
            Diagnostic::error(
                whole.span,
                format!("struct `{struct_name}` has no field `{field}`"),
            )
            .with_code(codes::UNKNOWN_FIELD)
            .with_label(base.span, format!("`{struct_name}` has this type")),
            format!("struct `{struct_name}` has no field named `{field}`"),
        );
        if let Some(suggestion) = suggest(&available, field, 3) {
            diagnostic = diagnostic.with_help(format!("did you mean `{suggestion}`?"));
        } else if !available.is_empty() {
            let mut list = String::new();
            for (index, entry) in available.iter().take(4).enumerate() {
                if index > 0 {
                    list.push_str(", ");
                }
                list.push('`');
                list.push_str(entry);
                list.push('`');
            }
            diagnostic = diagnostic.with_help(format!("available fields: {list}"));
        }
        self.report(diagnostic);
        Ty::Error
    }

    /// Types a vector swizzle: `v.xyz`, `v.rgba`, or a lane index.
    fn swizzle(&mut self, span: Span, elem: Scalar, lanes: u8, field: &str) -> Ty {
        let vector = Ty::vector(elem, lanes);
        if let Ok(index) = field.parse::<usize>() {
            if index < lanes as usize {
                return Ty::Scalar(elem);
            }
            self.report(
                Diagnostic::error(span, format!("lane index `{index}` is out of range"))
                    .with_code(codes::INVALID_INDEX)
                    .with_label(span, format!("`{vector}` has {lanes} lanes")),
            );
            return Ty::Error;
        }
        let set = SWIZZLE_SETS.iter().find(|set| {
            !field.is_empty()
                && field
                    .chars()
                    .all(|character| set.contains(character))
        });
        let Some(set) = set else {
            self.report(
                Diagnostic::error(span, format!("invalid swizzle `{field}`"))
                    .with_code(codes::UNKNOWN_FIELD)
                    .with_label(span, format!("on `{vector}`"))
                    .with_help("swizzles use `xyzw`, `rgba`, or `stpq`"),
            );
            return Ty::Error;
        };
        let count = field.chars().count();
        if count > 4 {
            self.report(
                Diagnostic::error(span, "a swizzle selects at most four lanes")
                    .with_code(codes::UNKNOWN_FIELD)
                    .with_label(span, format!("`{field}` selects {count}")),
            );
            return Ty::Error;
        }
        for character in field.chars() {
            if let Some(position) = set.find(character)
                && position >= lanes as usize
            {
                self.report(
                    Diagnostic::error(span, format!("lane `{character}` is out of range for `{vector}`"))
                        .with_code(codes::UNKNOWN_FIELD)
                        .with_help(format!("`{vector}` only has lanes 0 to {}", lanes - 1)),
                );
                return Ty::Error;
            }
        }
        if count == 1 {
            Ty::Scalar(elem)
        } else {
            Ty::Vector {
                elem,
                lanes: count as u8,
            }
        }
    }

    /// Checks that a condition expression is a scalar `bool`.
    ///
    /// A `boolN` mask passes [`is_bool`] but cannot drive structured
    /// control flow, so it is rejected here with a reduction hint.
    fn check_condition(&mut self, cond: &Expr) {
        let ty = self.check_expr(cond);
        if ty.is_error() || ty.is_never() || matches!(ty, Ty::Scalar(Scalar::Bool)) {
            return;
        }
        let help = if matches!(
            ty,
            Ty::Vector {
                elem: Scalar::Bool,
                ..
            }
        ) {
            Some("reduce the vector with `all(…)` or `any(…)`")
        } else if is_numeric(&ty) {
            Some("compare the value, for example `x != 0`")
        } else {
            None
        };
        let mut diagnostic = primary(
            Diagnostic::error(cond.span, "condition must have type `bool`")
                .with_code(codes::INVALID_CONDITION),
            format!("expected `bool`, found `{ty}`"),
        );
        if let Some(help) = help {
            diagnostic = diagnostic.with_help(help);
        }
        self.report(diagnostic);
    }

    /// Types the iterable of a `for` loop: ranges, arrays, and vectors.
    fn iter_type(&mut self, iter: &Expr) -> Ty {
        if let ExprKind::Range { start, end, .. } = &iter.kind {
            let start_ty = start.as_deref().map(|expr| self.check_expr(expr));
            let end_ty = end.as_deref().map(|expr| self.check_expr(expr));
            for (expr, ty) in [(start.as_deref(), &start_ty), (end.as_deref(), &end_ty)] {
                if let (Some(expr), Some(ty)) = (expr, ty)
                    && !is_int_like(ty)
                    && !ty.is_error()
                    && !ty.is_never()
                {
                    self.report(
                        Diagnostic::error(expr.span, "range endpoints must be integers")
                            .with_code(codes::RANGE_NOT_ALLOWED)
                            .with_help("step through integer bounds"),
                    );
                    return Ty::Error;
                }
            }
            let elem = match (&start_ty, &end_ty) {
                (Some(low), Some(high)) => {
                    if coerce(low, high) {
                        high.clone()
                    } else {
                        low.clone()
                    }
                }
                (Some(bound), None) | (None, Some(bound)) => bound.clone(),
                (None, None) => {
                    self.report(
                        Diagnostic::error(iter.span, "range must have at least one endpoint")
                            .with_code(codes::RANGE_NOT_ALLOWED)
                            .with_help("write `0..n` or `0..=n`"),
                    );
                    return Ty::Error;
                }
            };
            return default_literals(&elem);
        }
        let ty = self.check_expr(iter);
        match strip_indirection(&ty) {
            Ty::Array { elem, .. } => (**elem).clone(),
            Ty::Vector { elem, .. } => Ty::Scalar(*elem),
            Ty::Error | Ty::Never => Ty::Error,
            Ty::Generic(name) => Ty::Generic(name.clone()),
            other => {
                let other = other.clone();
                self.report(
                    primary(
                        Diagnostic::error(iter.span, format!("`{other}` is not iterable"))
                            .with_code(codes::NOT_ITERABLE),
                        format!("cannot iterate over `{other}`"),
                    )
                    .with_help("iterate arrays, vectors, and ranges"),
                );
                Ty::Error
            }
        }
    }

    /// Checks a `return` against the enclosing function's signature.
    fn return_type(&mut self, whole: &Expr, value: Option<&Expr>) -> Ty {
        let ret = self
            .ctx
            .as_ref()
            .map_or(Ty::Error, |ctx| ctx.ret.clone());
        match value {
            Some(expr) => {
                let ty = self.check_expr(expr);
                self.expect_coerce(expr.span, &ty, &ret);
            }
            None => {
                if !matches!(ret, Ty::Void | Ty::Error) {
                    let detail = format!("expected `{ret}`, found no value");
                    self.report(
                        primary(
                            Diagnostic::error(whole.span, "missing return value")
                                .with_code(codes::MISMATCHED_TYPES),
                            detail,
                        )
                        .with_help("return a value or drop the return type"),
                    );
                }
            }
        }
        Ty::Never
    }

    /// Checks that an assignment target can hold a value.
    fn check_assignable(&mut self, target: &Expr) {
        match &target.kind {
            ExprKind::Error => {}
            ExprKind::Path(path) => {
                if path.segments.len() != 1 {
                    return;
                }
                let Some(segment) = path.segments.last() else {
                    return;
                };
                let name = strip_raw_ident(&segment.name);
                let Some((mutable, _, span, _)) = self.binding_of(name) else {
                    return;
                };
                if !mutable {
                    self.report(
                        primary(
                            Diagnostic::error(
                                target.span,
                                format!("cannot assign to immutable binding `{name}`"),
                            )
                            .with_code(codes::ASSIGN_TO_IMMUTABLE)
                            .with_label(span, "declared here"),
                            format!("cannot assign twice to `{name}`"),
                        )
                        .with_help(format!("declare it as `let mut {name}`")),
                    );
                }
            }
            ExprKind::Index { expr: base, .. } | ExprKind::Field { expr: base, .. } => {
                // Stores through a pointer or reference mutate the pointee,
                // never the binding, so `p[i] = v` needs no `let mut p` —
                // only a writable pointee.
                if self.binds_indirection(base) {
                    self.check_deref_assignable(base);
                } else {
                    self.check_assignable(base);
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: base,
            } => self.check_deref_assignable(base),
            _ => self.report(
                Diagnostic::error(target.span, "cannot assign to this expression")
                    .with_code(codes::INVALID_ASSIGNMENT_TARGET)
                    .with_help("only variables, fields, and array elements can be assigned"),
            ),
        }
    }

    /// Checks `*p = value` where `p` must point at writable memory.
    fn check_deref_assignable(&mut self, base: &Expr) {
        let ExprKind::Path(path) = &base.kind else {
            return;
        };
        if path.segments.len() != 1 {
            return;
        }
        let Some(segment) = path.segments.last() else {
            return;
        };
        let name = strip_raw_ident(&segment.name);
        let Some((_, ty, span, _)) = self.binding_of(name) else {
            return;
        };
        if matches!(
            ty,
            Ty::Ptr { mutable: false, .. } | Ty::Ref { mutable: false, .. }
        ) {
            self.report(
                primary(
                    Diagnostic::error(base.span, "cannot assign through an immutable pointer")
                        .with_code(codes::ASSIGN_TO_IMMUTABLE)
                        .with_label(span, "declared without `mut`"),
                    format!("`{name}` points at read-only memory"),
                )
                .with_help("declare it as `*mut …`"),
            );
        }
    }

    /// True when `expr` is a single-segment path bound to a pointer or
    /// reference — assignment through it mutates the pointee, not the binding.
    fn binds_indirection(&self, expr: &Expr) -> bool {
        let ExprKind::Path(path) = &expr.kind else {
            return false;
        };
        if path.segments.len() != 1 {
            return false;
        }
        let Some(segment) = path.segments.last() else {
            return false;
        };
        let name = strip_raw_ident(&segment.name);
        self.binding_of(name)
            .is_some_and(|(_, ty, _, _)| matches!(ty, Ty::Ptr { .. } | Ty::Ref { .. }))
    }

    /// Types an index expression: `a[i]`.
    fn index_type(&mut self, base: &Expr, index: &Expr) -> Ty {
        let base_ty = self.check_expr(base);
        let index_ty = self.check_expr(index);
        // A reference reads through to its referent; a pointer keeps its own
        // level — `p[i]` is `*(p + i)`, so `p: *mut [T; N]` yields the inner
        // array while `p: *mut T` yields `T`.
        let mut target = &base_ty;
        while let Ty::Ref { inner, .. } = target {
            target = inner;
        }
        let element = match target {
            Ty::Array { elem, .. } => (**elem).clone(),
            Ty::Ptr { inner, .. } => (**inner).clone(),
            Ty::Vector { elem, .. } => Ty::Scalar(*elem),
            Ty::Error | Ty::Never => Ty::Error,
            _ => {
                let shown = base_ty.clone();
                self.report(
                    primary(
                        Diagnostic::error(base.span, format!("cannot index `{shown}`"))
                            .with_code(codes::INVALID_INDEX),
                        format!("`{shown}` cannot be indexed"),
                    )
                    .with_help("only arrays, vectors, and pointers support `[]`"),
                );
                return Ty::Error;
            }
        };
        if !is_int_like(&index_ty) && !index_ty.is_error() && !index_ty.is_never() {
            self.report(primary(
                Diagnostic::error(index.span, "invalid index type").with_code(codes::INVALID_INDEX),
                format!("expected an integer index, found `{index_ty}`"),
            ));
            return Ty::Error;
        }
        element
    }

    /// Types an array literal, unifying its elements.
    fn array_type(&mut self, whole: &Expr, elems: &[Expr]) -> Ty {
        let Some(first) = elems.first() else {
            self.report(
                Diagnostic::error(whole.span, "cannot infer the element type of an empty array")
                    .with_code(codes::MISSING_TYPE_ANNOTATION)
                    .with_help("annotate the binding, for example `let xs: [int; 0] = [];`"),
            );
            return Ty::Error;
        };
        let mut elem = self.check_expr(first);
        for entry in elems.iter().skip(1) {
            let ty = self.check_expr(entry);
            if !unify(&elem, &ty, &mut Vec::new()) {
                self.report_mismatch(entry.span, &ty, &elem);
            }
            if !ty.is_error() && !elem.is_error() {
                elem = if coerce(&elem, &ty) { ty } else { elem };
            }
        }
        Ty::Array {
            elem: Box::new(elem),
            len: u32::try_from(elems.len()).ok(),
        }
    }

    /// Checks that `actual` may be used where `expected` is required.
    ///
    /// Returns whether the types matched; failures become an
    /// [`codes::MISMATCHED_TYPES`] diagnostic.
    fn expect_coerce(&mut self, span: Span, actual: &Ty, expected: &Ty) -> bool {
        if unify(expected, actual, &mut Vec::new()) {
            return true;
        }
        self.report_mismatch(span, actual, expected);
        false
    }

    /// Reports `expected X, found Y` with the most useful help line.
    fn report_mismatch(&mut self, span: Span, actual: &Ty, expected: &Ty) {
        let detail = MessageBuilder::new()
            .text("expected ")
            .ty(expected)
            .text(", found ")
            .ty(actual)
            .finish();
        let mut diagnostic = primary(
            Diagnostic::error(span, "mismatched types").with_code(codes::MISMATCHED_TYPES),
            detail,
        );
        if let Ty::IntLit(value) = actual {
            let fits = match expected {
                Ty::Scalar(scalar) => scalar.is_int() && scalar.fits_int(*value),
                Ty::Vector { elem, .. } => elem.is_int() && elem.fits_int(*value),
                _ => false,
            };
            if !fits {
                diagnostic =
                    diagnostic.with_note(format!("the literal `{value}` does not fit in `{expected}`"));
            }
        } else if actual.is_unit_like() && !expected.is_unit_like() {
            diagnostic = diagnostic.with_help("add a `return` expression");
        } else if expected.is_unit_like() {
            diagnostic = diagnostic.with_help("this function returns nothing; remove the value");
        } else if (is_float(actual) && is_int_like(expected)) || (is_bool(actual) && is_numeric(expected)) {
            diagnostic = diagnostic.with_help("cast the value with `as`");
        } else if is_bool(expected) && is_numeric(actual) {
            diagnostic = diagnostic.with_help("compare the value, for example `x != 0`");
        }
        self.report(diagnostic);
    }

    /// Records entry into a loop for `break` and `continue`.
    fn enter_loop(&mut self) {
        if let Some(ctx) = self.ctx.as_mut() {
            ctx.loop_depth += 1;
        }
    }

    /// Records exit from a loop.
    fn exit_loop(&mut self) {
        if let Some(ctx) = self.ctx.as_mut() {
            ctx.loop_depth = ctx.loop_depth.saturating_sub(1);
        }
    }

    /// How many loops enclose the statement being checked.
    fn loop_depth(&self) -> usize {
        self.ctx.as_ref().map_or(0, |ctx| ctx.loop_depth)
    }

    /// Looks a name up in the open scopes, innermost first.
    ///
    /// Returns the mutability, type, declaration span, and pattern id
    /// of the innermost binding.
    fn binding_of(&self, name: &str) -> Option<(bool, Ty, Span, NodeId)> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope
                .bindings
                .iter()
                .rev()
                .find(|b| b.name == name)
            {
                return Some((
                    binding.state.is_mutable(),
                    binding.ty.clone(),
                    binding.span,
                    binding.pat,
                ));
            }
        }
        None
    }

    /// Marks a binding as read so the unused lint stays quiet.
    fn mark_binding_used(&mut self, name: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope
                .bindings
                .iter_mut()
                .rev()
                .find(|b| b.name == name)
            {
                binding.state.mark_used();
                return;
            }
        }
    }

    /// Every value name that could be meant, for spelling suggestions.
    fn value_candidates(&self) -> Vec<String> {
        let mut candidates = Vec::new();
        for scope in &self.scopes {
            for binding in &scope.bindings {
                candidates.push(binding.name.clone());
            }
        }
        candidates.extend(self.defs.fns.keys().cloned());
        for builtin in builtins() {
            candidates.push(String::from(builtin.name));
        }
        candidates
    }
}

/// Reports call cycles; the OpenCL dialect forbids recursion.
///
/// The walk is an explicit-stack depth-first search so a deep call
/// chain cannot overflow the native stack.
fn collect_call_cycles(calls: &BTreeMap<String, Vec<(String, Span)>>, out: &mut Vec<Diagnostic>) {
    let mut state: BTreeMap<String, u8> = BTreeMap::new();
    let mut reported: BTreeSet<String> = BTreeSet::new();
    let roots: Vec<String> = calls.keys().cloned().collect();
    for root in roots {
        if state.get(&root).copied().unwrap_or(0) != 0 {
            continue;
        }
        let mut stack: Vec<(String, usize)> = Vec::new();
        state.insert(root.clone(), 1);
        stack.push((root, 0));
        while let Some(top) = stack.last() {
            let current = top.0.clone();
            let edge_index = top.1;
            let edges = calls.get(&current).cloned().unwrap_or_default();
            if edge_index >= edges.len() {
                state.insert(current, 2);
                stack.pop();
                continue;
            }
            if let Some(top) = stack.last_mut() {
                top.1 += 1;
            }
            let Some((callee, span)) = edges.get(edge_index) else {
                continue;
            };
            let callee = callee.clone();
            let span = *span;
            match state.get(&callee).copied().unwrap_or(0) {
                1 => {
                    let start = stack
                        .iter()
                        .position(|(name, _)| *name == callee)
                        .unwrap_or(0);
                    let mut chain: Vec<&str> = stack[start..]
                        .iter()
                        .map(|(name, _)| name.as_str())
                        .collect();
                    chain.push(callee.as_str());
                    if reported.insert(chain.join("->")) {
                        let cycle = chain.join(" -> ");
                        out.push(
                            primary(
                                Diagnostic::error(span, "recursive call cycle detected")
                                    .with_code(codes::RECURSION),
                                format!("cycle: {cycle}"),
                            )
                            .with_note("the OpenCL dialect does not support recursion"),
                        );
                    }
                }
                0 => {
                    state.insert(callee.clone(), 1);
                    stack.push((callee, 0));
                }
                _ => {}
            }
        }
    }
}

/// Reports struct definitions that depend on themselves.
///
/// Pointer indirection breaks the cycle, so `*mut T` edges are not
/// followed: a linked list is legal, an inline cycle is not.
fn collect_struct_cycles(struct_sigs: &BTreeMap<String, StructSig>, out: &mut Vec<Diagnostic>) {
    let mut edges: BTreeMap<String, Vec<(Span, String)>> = BTreeMap::new();
    for (name, sig) in struct_sigs {
        for field in &sig.fields {
            let mut targets = Vec::new();
            collect_struct_refs(&field.ty, &mut targets);
            for target in targets {
                edges
                    .entry(name.clone())
                    .or_default()
                    .push((field.span, target));
            }
        }
    }
    let mut state: BTreeMap<String, u8> = BTreeMap::new();
    let mut reported: BTreeSet<String> = BTreeSet::new();
    let roots: Vec<String> = edges.keys().cloned().collect();
    for root in roots {
        if state.get(&root).copied().unwrap_or(0) != 0 {
            continue;
        }
        let mut stack: Vec<String> = Vec::new();
        state.insert(root.clone(), 1);
        stack.push(root);
        while let Some(current) = stack.last().cloned() {
            let targets = edges.get(&current).cloned().unwrap_or_default();
            let mut pushed = None;
            for (span, target) in targets {
                match state.get(&target).copied().unwrap_or(0) {
                    1 => {
                        let start = stack
                            .iter()
                            .position(|name| *name == target)
                            .unwrap_or(0);
                        let mut chain: Vec<&str> = stack[start..]
                            .iter()
                            .map(String::as_str)
                            .collect();
                        chain.push(target.as_str());
                        if reported.insert(chain.join("->")) {
                            let cycle = chain.join(" -> ");
                            out.push(
                                primary(
                                    Diagnostic::error(span, "recursive struct definition")
                                        .with_code(codes::RECURSIVE_TYPE),
                                    format!("cycle: {cycle}"),
                                )
                                .with_note("use a `*mut` pointer to break the cycle"),
                            );
                        }
                    }
                    0 => {
                        state.insert(target.clone(), 1);
                        pushed = Some(target);
                        break;
                    }
                    _ => {}
                }
            }
            if let Some(child) = pushed {
                stack.push(child);
            } else {
                state.insert(current, 2);
                stack.pop();
            }
        }
    }
}

/// Reports declarations nothing references, quiet after any error.
#[allow(
    clippy::too_many_arguments,
    reason = "the three signature tables plus references read better as arguments"
)]
fn collect_unused(
    fn_sigs: &BTreeMap<String, FnSig>,
    struct_sigs: &BTreeMap<String, StructSig>,
    alias_sigs: &BTreeMap<String, AliasSig>,
    references: &BTreeSet<String>,
    has_errors: bool,
    out: &mut Vec<Diagnostic>,
) {
    if has_errors {
        return;
    }
    let mut entries: Vec<(Span, String)> = Vec::new();
    for (name, sig) in fn_sigs {
        if !sig.kernel && !references.contains(name) {
            entries.push((sig.name_span, format!("function `{name}` is never used")));
        }
    }
    for (name, sig) in struct_sigs {
        if !references.contains(name) {
            entries.push((sig.name_span, format!("struct `{name}` is never used")));
        }
    }
    for (name, sig) in alias_sigs {
        if !references.contains(name) {
            entries.push((sig.name_span, format!("type alias `{name}` is never used")));
        }
    }
    entries.sort_by_key(|(span, _)| span.offset);
    for (span, message) in entries {
        out.push(
            Diagnostic::warning(span, message)
                .with_code(codes::UNUSED_ITEM)
                .with_help("consider removing it if is not needed"),
        );
    }
}

/// Builds the public declaration list, in source order.
fn build_declarations(
    fn_sigs: &BTreeMap<String, FnSig>,
    struct_sigs: &BTreeMap<String, StructSig>,
    alias_sigs: &BTreeMap<String, AliasSig>,
) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    for (name, sig) in fn_sigs {
        let params = sig
            .params
            .iter()
            .map(|param| {
                let name = param
                    .name
                    .clone()
                    .unwrap_or_else(|| String::from("_"));
                (name, param.ty.clone())
            })
            .collect();
        declarations.push(Declaration {
            name: name.clone(),
            span: sig.span,
            item_index: sig.index,
            kind: DeclKind::Function {
                params,
                ret: sig.ret.clone(),
                kernel: sig.kernel,
            },
        });
    }
    for (name, sig) in struct_sigs {
        let fields = sig
            .fields
            .iter()
            .map(|field| (field.name.clone(), field.ty.clone()))
            .collect();
        declarations.push(Declaration {
            name: name.clone(),
            span: sig.span,
            item_index: sig.index,
            kind: DeclKind::Struct { fields },
        });
    }
    for (name, sig) in alias_sigs {
        declarations.push(Declaration {
            name: name.clone(),
            span: sig.span,
            item_index: sig.index,
            kind: DeclKind::Alias {
                target: sig.ty.clone(),
            },
        });
    }
    declarations.sort_by_key(|declaration| declaration.item_index);
    declarations
}

/// Swizzle alphabets the dialect accepts.
const SWIZZLE_SETS: [&str; 3] = ["xyzw", "rgba", "stpq"];

/// The source spelling of a binary operator, for messages.
const fn op_symbol(op: BinaryOp) -> &'static str {
    use BinaryOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
        Rem => "%",
        Shl => "<<",
        Shr => ">>",
        BitAnd => "&",
        BitXor => "^",
        BitOr => "|",
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        And => "&&",
        Or => "||",
    }
}

/// True when `from` may be cast to `to` with `as`.
///
/// Numeric kinds convert among themselves, pointers convert to pointers
/// of any mutability, `bool` converts to integers and back, and `void`
/// and structs never convert.
fn castable(from: &Ty, to: &Ty) -> bool {
    if from.is_error() || to.is_error() || from.is_never() {
        return true;
    }
    if matches!(to, Ty::Scalar(Scalar::Bool)) {
        return is_bool(from);
    }
    if matches!(from, Ty::Scalar(Scalar::Bool)) || matches!(from, Ty::Generic(_)) {
        return is_numeric(to);
    }
    let from_ok = is_numeric(from) || matches!(from, Ty::Ptr { .. });
    let to_ok = is_numeric(to) || matches!(to, Ty::Ptr { .. });
    from_ok && to_ok && matches!(from, Ty::Ptr { .. }) == matches!(to, Ty::Ptr { .. })
}

/// Removes the surrounding quotes from a literal body.
fn strip_quotes(body: &str) -> &str {
    if matches!(body.as_bytes().first(), Some(b'\'' | b'"')) && body.len() >= 2 {
        body.get(1..body.len() - 1).unwrap_or(body)
    } else {
        body
    }
}

/// Peels pointers and references off a type.
fn strip_indirection(ty: &Ty) -> &Ty {
    match ty {
        Ty::Ptr { inner, .. } | Ty::Ref { inner, .. } => strip_indirection(inner),
        other => other,
    }
}

/// True for scalar and vector `bool`.
fn is_bool(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::Scalar(Scalar::Bool)
            | Ty::Vector {
                elem: Scalar::Bool,
                ..
            }
    )
}

/// True for integers, floats, and their literals, vectors included.
fn is_numeric(ty: &Ty) -> bool {
    match ty {
        Ty::Vector { elem, .. } => elem.is_int() || elem.is_float(),
        other => other.is_numeric_like(),
    }
}

/// True for integer scalars and literals, vectors included.
fn is_int_like(ty: &Ty) -> bool {
    match ty {
        Ty::Vector { elem, .. } => elem.is_int(),
        other => other.is_int_like(),
    }
}

/// True for float scalars and literals, vectors included.
fn is_float(ty: &Ty) -> bool {
    match ty {
        Ty::FloatLit(_) => true,
        Ty::Scalar(scalar) => scalar.is_float(),
        Ty::Vector { elem, .. } => elem.is_float(),
        _ => false,
    }
}

/// Promotes an integer shape to `float`, leaving floats untouched.
fn promote_float(ty: &Ty) -> Ty {
    match ty {
        Ty::Scalar(scalar) if scalar.is_int() => Ty::Scalar(Scalar::F32),
        Ty::Vector { elem, lanes } if elem.is_int() => Ty::Vector {
            elem: Scalar::F32,
            lanes: *lanes,
        },
        Ty::IntLit(_) | Ty::FloatLit(_) => Ty::Scalar(Scalar::F32),
        other => other.clone(),
    }
}

/// Builds the `wrong number of arguments` diagnostic.
fn arity_diagnostic(span: Span, expected: usize, found: usize) -> Diagnostic {
    let takes = if expected == 1 {
        String::from("1 argument")
    } else {
        format!("{expected} arguments")
    };
    let supplied = if found == 1 {
        String::from("1 was supplied")
    } else {
        format!("{found} were supplied")
    };
    primary(
        Diagnostic::error(span, format!("this function takes {takes} but {supplied}"))
            .with_code(codes::WRONG_NUMBER_OF_ARGUMENTS),
        format!("expected {expected}, found {found}"),
    )
}

/// Span covering the argument list of a call: from `(` to `)`.
fn call_args_span(whole: &Expr, callee: &Expr) -> Span {
    let start = callee.span.end();
    let end = whole.span.end();
    if end > start {
        Span::new(start, end - start)
    } else {
        whole.span
    }
}

/// True when `name` appears in the generic parameter list.
fn generics_contain(generics: &[String], name: &str) -> bool {
    generics.iter().any(|generic| generic == name)
}

/// Collects the struct names a type refers to, stopping at pointers.
fn collect_struct_refs(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::Struct { name, args } => {
            out.push(name.clone());
            for arg in args {
                collect_struct_refs(arg, out);
            }
        }
        Ty::Ptr { .. } | Ty::Ref { .. } => {}
        Ty::Array { elem, .. } => collect_struct_refs(elem, out),
        Ty::Tuple(elems) => {
            for elem in elems {
                collect_struct_refs(elem, out);
            }
        }
        _ => {}
    }
}

/// The closest candidate to `name` within a Levenshtein distance of `max`.
fn suggest(candidates: &[String], name: &str, max: usize) -> Option<String> {
    let mut best: Option<(usize, &String)> = None;
    for candidate in candidates {
        if candidate == name {
            continue;
        }
        let cost = distance(candidate, name);
        if cost > max {
            continue;
        }
        if best.is_none_or(|(best_cost, _)| cost < best_cost) {
            best = Some((cost, candidate));
        }
    }
    best.map(|(_, candidate)| candidate.clone())
}

/// Levenshtein edit distance between two names, over their characters.
fn distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = alloc::vec![0; right.len() + 1];
    for (row, left_char) in left.iter().enumerate() {
        current[0] = row + 1;
        for (column, right_char) in right.iter().enumerate() {
            let substitution = usize::from(left_char != right_char);
            let deleted = previous[column + 1] + 1;
            let inserted = current[column] + 1;
            let matched = previous[column] + substitution;
            current[column + 1] = core::cmp::min(deleted, core::cmp::min(inserted, matched));
        }
        core::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}
