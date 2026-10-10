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

//! The compiler driver: arguments in, emission out.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use codevar_cli_arg_parse::{ArgParser, validate_choice};
use codevar_ocl_asm::SpirvStream;
use codevar_ocl_ir::lower::{ItemLowerer, lower};
use codevar_ocl_ir::verify::verify_function;
use codevar_ocl_lex::{Base, LiteralKind, Token, TokenKind, tokenize};
use codevar_ocl_parse::{GenericParam, ItemKind, ItemStream, optimize, parse};
use codevar_ocl_sar::{DeclCollector, analyze_body, file_checks};

use crate::fs;

/// Program name used in help, usage, and diagnostic prefixes.
const PROGRAM: &str = "codevar-oclc";

/// One-line description shown by `--help`.
const ABOUT: &str = "Compile the Codevar OpenCL dialect to SPIR-V, CUDA PTX, or Metal.";

/// Every stage accepted by `--emit` (future stages fail with a clear message).
const EMIT_STAGES: &[&str] = &["tokens", "ast", "analysis", "ir", "spirv", "ptx", "msl"];

/// Stages that currently run to completion.
const IMPLEMENTED_STAGES: &[&str] = &["tokens", "ast", "analysis", "ir", "spirv"];

/// Stage used when `--emit` is absent.
const DEFAULT_EMIT: &str = "tokens";

/// Process exit status produced by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Compilation (or `--help`/`--version`) completed successfully.
    Success,
    /// A compilation, I/O, or internal failure occurred.
    Failure,
    /// The command line was invalid (bad flags, missing `INPUT`).
    Usage,
}

impl Exit {
    /// The POSIX-style numeric exit code for this status.
    #[must_use]
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            Self::Success => 0,
            Self::Failure => 1,
            Self::Usage => 2,
        }
    }
}

/// The compiler driver: a full command line plus the pipeline that consumes it.
///
/// Constructed once from borrowed arguments, consumed by [`Driver::run`], which returns
/// the process exit status instead of exiting itself so embedders and tests can drive it.
///
/// # Examples
///
/// ```
/// use codevar_oclc::{Driver, Exit};
///
/// let args = vec![String::from("codevar-oclc"), String::from("--version")];
/// assert_eq!(Driver::new(&args).run(), Exit::Success);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Driver<'a> {
    args: &'a [String],
}

impl<'a> Driver<'a> {
    /// Creates a driver over the complete command line, `argv[0]` included.
    #[must_use]
    pub const fn new(args: &'a [String]) -> Self {
        Self { args }
    }

    /// Runs the compile pipeline and returns the exit status.
    ///
    /// Never panics on malformed input: every failure path reports a
    /// diagnostic to stderr and returns a non-[`Exit::Success`] status.
    pub fn run(self) -> Exit {
        let at_args = self.args.get(1..).unwrap_or_default();
        let borrowed: Vec<&str> = at_args.iter().map(String::as_str).collect();
        let expanded = match expand_response_files(&borrowed) {
            Ok(expanded) => expanded,
            Err(message) => return failure(&message),
        };
        let parser = build_parser();
        let matches = match parser.parse(expanded) {
            Ok(matches) => matches,
            Err(error) => return usage(&error.to_string()),
        };
        if matches.flag("help") {
            return write_stdout_text(&parser.help());
        }
        if matches.flag("version") {
            return write_stdout_text(&format!("{}\n", parser.version()));
        }
        let emit = match validate_choice("--emit", matches.option("emit"), EMIT_STAGES) {
            Ok(emit) => emit.unwrap_or(DEFAULT_EMIT),
            Err(error) => return usage(&error.to_string()),
        };
        let input = match matches.positional(0) {
            Some(input) => input,
            None => return usage("missing required argument '<INPUT>'"),
        };
        if !IMPLEMENTED_STAGES.contains(&emit) {
            return failure(&format!(
                "emission stage '{emit}' is not implemented yet (implemented stages: {})",
                IMPLEMENTED_STAGES.join(", ")
            ));
        }
        let source = match fs::read_source(input) {
            Ok(source) => source,
            Err(error) => return failure(&format!("cannot read '{input}': {error}")),
        };
        let display_path = if input == "-" { "<stdin>" } else { input };
        if emit == "spirv" {
            return run_spirv(&source, display_path, matches.option("output"));
        }
        let text = if emit == "analysis" {
            // The semantic analyzer owns its own Unicode, lexical, and
            // syntax gates, so it runs instead of (not after) the token
            // dump: one source of truth per diagnostic.
            let analyzed = codevar_ocl_sar::analyze(&source);
            codevar_ocl_sar::emit_stderr(
                &analyzed.diagnostics,
                display_path,
                &source,
                codevar_ocl_sar::ColorChoice::Auto,
            );
            if analyzed.has_errors() {
                return Exit::Failure;
            }
            format_declarations(&analyzed.declarations)
        } else if emit == "ir" {
            let analyzed = codevar_ocl_sar::analyze(&source);
            codevar_ocl_sar::emit_stderr(
                &analyzed.diagnostics,
                display_path,
                &source,
                codevar_ocl_sar::ColorChoice::Auto,
            );
            if analyzed.has_errors() {
                return Exit::Failure;
            }
            let parsed = parse(&source);
            if !parsed.errors.is_empty() {
                for error in &parsed.errors {
                    report_diagnostic(display_path, &source, error.span.offset, &error.message);
                }
                return Exit::Failure;
            }
            let module = match lower(&parsed.program, &analyzed) {
                Ok(module) => module,
                Err(error) => return failure(&format!("lowering failed: {error}")),
            };
            format!("{module}\n")
        } else {
            let output = lex_source(&source);
            if !output.diagnostics.is_empty() {
                for diagnostic in &output.diagnostics {
                    report_diagnostic(display_path, &source, diagnostic.offset, &diagnostic.message);
                }
                return Exit::Failure;
            }
            if emit == "ast" {
                let parsed = parse(&source);
                if !parsed.errors.is_empty() {
                    for error in &parsed.errors {
                        report_diagnostic(display_path, &source, error.span.offset, &error.message);
                    }
                    return Exit::Failure;
                }
                let mut program = parsed.program;
                let _ = optimize(&mut program);
                format!("{program:#?}\n")
            } else if output.lines.is_empty() {
                String::new()
            } else {
                format!("{}\n", output.lines.join("\n"))
            }
        };
        match matches.option("output") {
            Some(path) => match fs::write_file(path, &text) {
                Ok(()) => Exit::Success,
                Err(error) => failure(&format!("cannot write '{path}': {error}")),
            },
            None => write_stdout_text(&text),
        }
    }
}

/// Runs the streaming analysis → parse → lower → assemble pipeline for `--emit spirv`.
///
/// The source is walked one top-level item at a time instead of building
/// the whole-file AST, IR, and SPIR-V image at once: declarations are
/// collected in a first pass, then each function body is analyzed,
/// lowered, verified, emitted into the [`SpirvStream`], and reclaimed
/// before the next body starts, so peak memory tracks a single item.
///
/// Produces a binary SPIR-V module at the requested output (stdout when
/// `output` is `None`), with diagnostics rendered the same way as the
/// other emission stages.
fn run_spirv(source: &str, display_path: &str, output: Option<&str>) -> Exit {
    let mut collector = DeclCollector::new();
    let mut items = ItemStream::new(source);
    let mut parse_failed = false;
    while let Some(outcome) = items.next_item() {
        for error in &outcome.errors {
            report_diagnostic(display_path, source, error.span.offset, &error.message);
            parse_failed = true;
        }
        collector.feed(source, &outcome.item);
    }
    if parse_failed {
        return Exit::Failure;
    }
    let (mut env, diagnostics) = collector.finish();
    codevar_ocl_sar::emit_stderr(
        &diagnostics,
        display_path,
        source,
        codevar_ocl_sar::ColorChoice::Auto,
    );
    if diagnostics
        .iter()
        .any(codevar_ocl_sar::Diagnostic::is_error)
    {
        return Exit::Failure;
    }
    // The lowerer borrows the declaration slice for its whole life while
    // `analyze_body` needs `&mut env`, so the slice is copied out first.
    let declarations = env.declarations().to_vec();
    let mut lowerer = ItemLowerer::new(&declarations);
    for aliases in [true, false] {
        let mut items = ItemStream::new(source);
        while let Some(outcome) = items.next_item() {
            match (&outcome.item.kind, aliases) {
                (ItemKind::TypeAlias(alias), true) => {
                    if let Err(error) = lowerer.declare_alias(alias) {
                        return failure(&format!("lowering failed: {error}"));
                    }
                }
                (ItemKind::Fn(function), false) => {
                    let kernel = env.is_kernel(&function.name);
                    if let Err(error) = lowerer.declare_function(function, kernel) {
                        return failure(&format!("lowering failed: {error}"));
                    }
                }
                _ => {}
            }
        }
    }
    let mut stream = match SpirvStream::prelude(lowerer.module()) {
        Ok(stream) => stream,
        Err(error) => return failure(&format!("assembly failed: {error}")),
    };
    let mut items = ItemStream::new(source);
    while let Some(outcome) = items.next_item() {
        let ItemKind::Fn(function) = &outcome.item.kind else {
            continue;
        };
        if function
            .generics
            .iter()
            .any(|param| matches!(param, GenericParam::Type { .. }))
        {
            continue;
        }
        let mark = lowerer.module().watermark();
        let (diagnostics, tables) = analyze_body(&mut env, source, &outcome.item);
        codevar_ocl_sar::emit_stderr(
            &diagnostics,
            display_path,
            source,
            codevar_ocl_sar::ColorChoice::Auto,
        );
        if diagnostics
            .iter()
            .any(codevar_ocl_sar::Diagnostic::is_error)
        {
            return Exit::Failure;
        }
        if let Err(error) = lowerer.lower_function_body(function, &tables.types, &tables.resolutions) {
            return failure(&format!("lowering failed: {error}"));
        }
        let Some(kernel) = lowerer.function(&function.name) else {
            return failure(&format!(
                "lowering failed: function '{}' was never declared",
                function.name
            ));
        };
        if let Err(error) = verify_function(lowerer.module(), kernel) {
            return failure(&format!("verification failed: {error}"));
        }
        if let Err(error) = stream.emit_function_body(lowerer.module(), kernel) {
            return failure(&format!("assembly failed: {error}"));
        }
        drop(lowerer.module_mut().take_function_body(kernel));
        lowerer.module_mut().truncate_values(mark);
    }
    let checks = file_checks(&env);
    codevar_ocl_sar::emit_stderr(&checks, display_path, source, codevar_ocl_sar::ColorChoice::Auto);
    if checks
        .iter()
        .any(codevar_ocl_sar::Diagnostic::is_error)
    {
        return Exit::Failure;
    }
    let words = match stream.finish(lowerer.module()) {
        Ok(words) => words,
        Err(error) => return failure(&format!("assembly failed: {error}")),
    };
    let bytes = codevar_ocl_asm::spirv::to_bytes(&words);
    match output {
        Some(path) => match fs::write_file_bytes(path, &bytes) {
            Ok(()) => Exit::Success,
            Err(error) => failure(&format!("cannot write '{path}': {error}")),
        },
        None => write_stdout_bytes(&bytes),
    }
}

/// Renders the analyzer's resolved declarations, one line per item.
///
/// Mirrors the source shape so the dump can be diffed against the input:
///
/// ```text
/// fn add(a: int, b: int) -> int
/// struct Point { x: float, y: float }
/// type Coord = Point
/// #[kernel] fn vec_add(a: *mut float, b: *mut float, n: int) -> void
/// ```
fn format_declarations(declarations: &[codevar_ocl_sar::Declaration]) -> String {
    let mut text = String::new();
    for declaration in declarations {
        let name = declaration.name.as_str();
        match &declaration.kind {
            codevar_ocl_sar::DeclKind::Function { params, ret, kernel } => {
                if *kernel {
                    text.push_str("#[kernel] ");
                }
                text.push_str("fn ");
                text.push_str(name);
                text.push('(');
                push_typed_list(&mut text, params);
                text.push_str(") -> ");
                text.push_str(&format!("{ret}\n"));
            }
            codevar_ocl_sar::DeclKind::Struct { fields } => {
                text.push_str("struct ");
                text.push_str(name);
                if fields.is_empty() {
                    text.push_str(";\n");
                    continue;
                }
                text.push_str(" { ");
                push_typed_list(&mut text, fields);
                text.push_str(" }\n");
            }
            codevar_ocl_sar::DeclKind::Alias { target } => {
                text.push_str("type ");
                text.push_str(name);
                text.push_str(" = ");
                text.push_str(&format!("{target}\n"));
            }
        }
    }
    text
}

/// Appends `name: type` pairs, separated by `", "`, to `out`.
///
/// Shared by the function-parameter and struct-field arms of
/// [`format_declarations`], which differ only in their delimiters.
fn push_typed_list(out: &mut String, items: &[(String, codevar_ocl_sar::Ty)]) {
    for (index, (name, ty)) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(name);
        out.push_str(": ");
        out.push_str(&format!("{ty}"));
    }
}

/// Runs a full command line through [`Driver`] (the `run_compiler` free
/// function shape; `args` includes `argv[0]`).
///
/// # Examples
///
/// ```
/// let args = vec![String::from("codevar-oclc"), String::from("--version")];
/// assert_eq!(codevar_oclc::run(&args).code(), 0);
/// ```
#[must_use]
pub fn run(args: &[String]) -> Exit {
    Driver::new(args).run()
}

/// Builds the compiler's flag definition (help text comes from the same
/// definitions, so `--help` can never drift from what parses).
pub(crate) fn build_parser() -> ArgParser {
    ArgParser::new(PROGRAM, env!("CARGO_PKG_VERSION"), ABOUT)
        .flag("help", Some('h'), "Print help information")
        .flag("version", Some('V'), "Print version information")
        .option(
            "output",
            Some('o'),
            "FILE",
            "Write output to FILE (default: stdout)",
        )
        .option("emit", None, "STAGE", "Select emission stage (default: tokens)")
        .positional("INPUT", "Input source file, or - for stdin", false)
}

/// Expands `@response` file arguments using rustc's rules: a single level
/// (expanded lines are never re-scanned), one line = one argument, UTF-8,
/// `lines()` handling `\r\n`.
///
/// # Errors
///
/// Returns the diagnostic message when a referenced file cannot be read.
pub(crate) fn expand_response_files(args: &[&str]) -> Result<Vec<String>, String> {
    let mut expanded = Vec::with_capacity(args.len());
    for argument in args {
        if let Some(path) = argument.strip_prefix('@') {
            let contents = fs::read_file(path)
                .map_err(|error| format!("failed to load argument file '{path}': {error}"))?;
            for line in contents.lines() {
                expanded.push(String::from(line));
            }
        } else {
            expanded.push(String::from(*argument));
        }
    }
    Ok(expanded)
}

/// One lexical problem at a byte offset in the source.
pub(crate) struct Diagnostic {
    /// Byte offset of the offending token.
    pub(crate) offset: u32,
    /// Human-readable problem description.
    pub(crate) message: String,
}

/// The formatted token dump plus every lexical diagnostic found.
pub(crate) struct TokenOutput {
    /// One `offset len kind` line per token, in source order.
    pub(crate) lines: Vec<String>,
    /// Diagnostics collected while scanning; empty means a clean lex.
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// Formats every token of `source` as an `offset len kind` line and collects
/// lexical diagnostics (unterminated literals, empty numbers, stray
/// characters) without stopping at the first problem.
pub(crate) fn lex_source(source: &str) -> TokenOutput {
    let mut lines = Vec::new();
    let mut diagnostics = Vec::new();
    let mut offset: u32 = 0;
    for token in tokenize(source) {
        let start = offset as usize;
        let end = start + token.len as usize;
        let Some(text) = source.get(start..end) else {
            break;
        };
        lines.push(format!("{offset:8} {:4} {:?}", token.len, token.kind));
        if let Some(message) = token_diagnostic(text, token) {
            diagnostics.push(Diagnostic { offset, message });
        }
        offset = end as u32;
    }
    TokenOutput { lines, diagnostics }
}

/// Returns the diagnostic for a single token, if it carries a lexical error
/// flag (the lexer itself never fails; problems ride on token variants).
fn token_diagnostic(text: &str, token: Token) -> Option<String> {
    match token.kind {
        TokenKind::Literal { kind, .. } => literal_diagnostic(text, kind),
        TokenKind::BlockComment {
            terminated: false, ..
        } => Some(String::from("unterminated block comment")),
        TokenKind::Lifetime {
            starts_with_number: true,
        } => Some(String::from("lifetimes cannot start with a number")),
        TokenKind::Unknown => {
            let character = text.chars().next().unwrap_or('\u{fffd}');
            Some(format!("character not allowed in source code: {character:?}"))
        }
        TokenKind::UnknownPrefix => Some(format!("unknown literal prefix `{text}`")),
        _ => None,
    }
}

/// Returns the diagnostic for a literal token whose error flags are set.
fn literal_diagnostic(text: &str, kind: LiteralKind) -> Option<String> {
    match kind {
        LiteralKind::Int { empty_int: true, .. } => Some(String::from("integer literal has no digits")),
        LiteralKind::Float {
            base,
            empty_exponent: true,
        } => {
            if base == Base::Hexadecimal {
                Some(String::from("hexadecimal float literal requires an exponent"))
            } else {
                Some(String::from("exponent has no digits"))
            }
        }
        LiteralKind::Char { terminated: false } => Some(String::from("unterminated character literal")),
        LiteralKind::Str { terminated: false } => Some(String::from("unterminated string literal")),
        LiteralKind::RawStr { n_hashes: None } => match codevar_ocl_lex::validate_raw_str(text, 1) {
            Ok(()) => Some(String::from("invalid raw string literal")),
            Err(error) => Some(error.to_string()),
        },
        _ => None,
    }
}

/// Computes the 1-based line and character column of byte `offset`.
///
/// A `offset` that lands inside a multi-byte character is moved back to the
/// nearest character boundary so the slice below can never panic.
pub(crate) fn line_column(source: &str, offset: u32) -> (u32, u32) {
    let mut offset = (offset as usize).min(source.len());
    while offset > 0 && !source.is_char_boundary(offset) {
        offset -= 1;
    }
    let prefix = &source[..offset];
    let line = prefix
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1;
    let column = match prefix.rfind('\n') {
        Some(position) => prefix[position + 1..].chars().count() + 1,
        None => prefix.chars().count() + 1,
    };
    (line as u32, column as u32)
}

/// Writes a `program: error: …` line to stderr.
///
/// A failed stderr write is dropped: there is nowhere else to report it, and
/// the caller's exit status already conveys the failure.
pub(crate) fn report_error(message: &str) {
    let line = format!("{PROGRAM}: error: {message}\n");
    let _ = codevar_consoleutil::write_stderr(line.as_bytes());
}

/// Writes a usage error plus a `--help` hint to stderr.
///
/// A failed stderr write is dropped (see [`report_error`]).
pub(crate) fn report_usage(message: &str) {
    let line = format!("{PROGRAM}: error: {message}\ntry '{PROGRAM} --help' for more information\n");
    let _ = codevar_consoleutil::write_stderr(line.as_bytes());
}

/// Reports `message` through [`report_error`] and yields [`Exit::Failure`].
///
/// Keeps the error arms of [`Driver::run`]'s matches a single expression
/// instead of repeating the report-then-return pair in every branch.
#[inline]
fn failure(message: &str) -> Exit {
    report_error(message);
    Exit::Failure
}

/// Reports `message` through [`report_usage`] and yields [`Exit::Usage`].
///
/// The usage-side counterpart of [`failure`].
#[inline]
fn usage(message: &str) -> Exit {
    report_usage(message);
    Exit::Usage
}

/// Writes one `path:line:col: error: …` diagnostic to stderr.
///
/// A failed stderr write is dropped (see [`report_error`]).
pub(crate) fn report_diagnostic(path: &str, source: &str, offset: u32, message: &str) {
    let (line, column) = line_column(source, offset);
    let text = format!("{path}:{line}:{column}: error: {message}\n");
    let _ = codevar_consoleutil::write_stderr(text.as_bytes());
}

/// Writes compiler output to stdout, mapping a write failure to
/// [`Exit::Failure`].
fn write_stdout_text(text: &str) -> Exit {
    match codevar_consoleutil::write_stdout(text.as_bytes()) {
        Ok(()) => Exit::Success,
        Err(_) => failure("failed to write to stdout"),
    }
}

/// Writes raw compiler output (a SPIR-V module) to stdout, mapping a write
/// failure to [`Exit::Failure`].
fn write_stdout_bytes(bytes: &[u8]) -> Exit {
    match codevar_consoleutil::write_stdout(bytes) {
        Ok(()) => Exit::Success,
        Err(_) => failure("failed to write to stdout"),
    }
}
