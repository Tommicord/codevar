//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License.  You may obtain a copy of
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

//! Textual parser for [`Module`].
//!
//! [`parse`] reads the SPIR-V-assembly-shaped IR text that
//! [`crate::print`] produces and rebuilds a [`Module`].  Printing and
//! parsing are inverses: `print(parse(print(m))) == print(m)` for every
//! module `m`, which the tests here exercise over a module that uses the
//! whole instruction set.
//!
//! Identifiers live in one namespace, exactly as in the printer's
//! output, and a name may be defined only once.  The text is read in
//! three steps.  First the module scope imports, types, constants,
//! globals, function signatures is processed in order, so every
//! definition exists before any function body is read and a call may
//! name a function that appears later in the text.  Second, each body
//! registers its labels and instruction results before any instruction
//! is parsed, so branches and `OpPhi` may refer forward.  Third, the
//! entry points and decorations that the printer emits ahead of the
//! functions they name are resolved against the finished module.
//!
//! # Errors
//!
//! Every failure reports the 1-based line it was found on through
//! [`ParseError`].  The parser checks token shape, operand counts, and
//! agreement between storage-class words and pointer types; checks that
//! need to compare instructions against each other belong to the
//! verifier.

use crate::ir::{
    AddressingModel, BinOp, BlockId, BuildError, CmpOp, ConstValue, ConvOp, ExecutionModel, ExtSetId,
    FunctionControl, Inst, Linkage, MemoryModel, Module, Op, Storage, Target, Type, TypeId, UnOp, ValueId,
};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Parses `text` as a [`Module`].
///
/// # Errors
///
/// Returns a [`ParseError`] carrying the 1-based line of the first
/// problem: a missing `target` line, an unknown or duplicate
/// identifier, a malformed token, number, or string, an operand count
/// that does not match the instruction, or a construct the IR data
/// model rejects.
///
/// # Example
///
/// ```
/// use codevar_ocl_ir::parse::parse;
///
/// let module = parse(
///     "; Codevar IR 0.1\n\
///      target opencl address physical64 memory opencl\n",
/// )
/// .expect("parses");
/// assert!(module.functions().is_empty());
/// ```
pub fn parse(text: &str) -> Result<Module, ParseError> {
    let lines = tokenize_all(text)?;
    Parser::new().run(&lines)
}

/// A failure to parse textual IR, with the line it occurred on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based line number; 0 is never produced.
    pub line: usize,
    /// What went wrong.
    pub kind: ParseErrorKind,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.kind)
    }
}

impl core::error::Error for ParseError {}

/// The category of a [`ParseError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// The text has no `target` line.
    MissingTarget,
    /// The line ended before a required token appeared.
    UnexpectedEndOfLine,
    /// A token appeared where the syntax requires a specific shape.
    UnexpectedToken,
    /// The statement or instruction opcode is not part of the IR.
    UnknownStatement,
    /// An operand names something that was never defined.
    UnknownId,
    /// A name is defined twice in the shared identifier namespace.
    DuplicateId,
    /// A numeric literal cannot be parsed or does not fit its type.
    InvalidNumber,
    /// A quoted string is unterminated or carries a bad escape.
    InvalidString,
    /// A result disagrees with the instruction or its signature.
    ResultMismatch,
    /// An instruction or definition has the wrong operand count.
    WrongOperandCount,
    /// A function signature operand is not a function type.
    ExpectedFunctionType,
    /// A variable type operand is not a pointer type.
    ExpectedPointerType,
    /// A storage-class word disagrees with the pointer type.
    StorageMismatch,
    /// An entry point names something that is not a function.
    NotAFunction,
    /// A decoration targets something that cannot be decorated.
    NotDecoratable,
    /// An instruction appears before the first label of its body.
    MissingLabel,
    /// A function parameter appears after the first label.
    MisplacedParameter,
    /// The IR builder rejected the construct.
    Build(BuildError),
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTarget => write!(f, "missing `target` line"),
            Self::UnexpectedEndOfLine => write!(f, "unexpected end of line"),
            Self::UnexpectedToken => write!(f, "unexpected token"),
            Self::UnknownStatement => write!(f, "unknown statement"),
            Self::UnknownId => write!(f, "unknown identifier"),
            Self::DuplicateId => write!(f, "identifier already defined"),
            Self::InvalidNumber => write!(f, "invalid number"),
            Self::InvalidString => write!(f, "invalid string literal"),
            Self::ResultMismatch => write!(f, "result does not match the instruction"),
            Self::WrongOperandCount => write!(f, "wrong number of operands"),
            Self::ExpectedFunctionType => write!(f, "expected a function type"),
            Self::ExpectedPointerType => write!(f, "expected a pointer type"),
            Self::StorageMismatch => write!(f, "storage class does not match the pointer type"),
            Self::NotAFunction => write!(f, "entry point is not a function"),
            Self::NotDecoratable => write!(f, "target cannot carry this decoration"),
            Self::MissingLabel => write!(f, "instruction appears before the first label"),
            Self::MisplacedParameter => write!(f, "function parameter appears after a label"),
            Self::Build(error) => write!(f, "IR builder rejected the module: {error}"),
        }
    }
}

/// One lexical token: a bare word or a decoded quoted string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// A whitespace-delimited word such as `OpReturn` or `%entry`.
    Word(String),
    /// A double-quoted string with its escapes already decoded.
    Str(String),
}

impl Token {
    /// The token as a bare word, when it is one.
    fn as_word(&self) -> Option<&str> {
        match self {
            Self::Word(word) => Some(word),
            Self::Str(_) => None,
        }
    }

    /// True when the token is the bare word `word`.
    fn is_word(&self, word: &str) -> bool {
        matches!(self, Self::Word(text) if text == word)
    }
}

/// One source line with its 1-based number.
struct Line {
    /// 1-based line number in the source text.
    number: usize,
    /// The line's tokens; empty for blank and comment-only lines.
    tokens: Vec<Token>,
}

impl Line {
    fn err(&self, kind: ParseErrorKind) -> ParseError {
        ParseError {
            line: self.number,
            kind,
        }
    }
}

/// Splits `text` into numbered, tokenized lines.
fn tokenize_all(text: &str) -> Result<Vec<Line>, ParseError> {
    let mut lines = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        lines.push(Line {
            number,
            tokens: tokenize(raw, number)?,
        });
    }
    Ok(lines)
}

/// Tokenizes one line: words split on whitespace, quoted strings with
/// `\n`/`\r`/`\t`/`\"`/`\\` escapes, and `;` starting a comment outside
/// a string.
fn tokenize(line: &str, number: usize) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = line.char_indices().peekable();
    while let Some(&(index, ch)) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == ';' {
            break;
        }
        if ch == '"' {
            chars.next();
            let mut text = String::new();
            let mut closed = false;
            while let Some((_, ch)) = chars.next() {
                match ch {
                    '"' => {
                        closed = true;
                        break;
                    }
                    '\\' => match chars.next() {
                        Some((_, 'n')) => text.push('\n'),
                        Some((_, 'r')) => text.push('\r'),
                        Some((_, 't')) => text.push('\t'),
                        Some((_, '"')) => text.push('"'),
                        Some((_, '\\')) => text.push('\\'),
                        _ => {
                            return Err(ParseError {
                                line: number,
                                kind: ParseErrorKind::InvalidString,
                            });
                        }
                    },
                    other => text.push(other),
                }
            }
            if !closed {
                return Err(ParseError {
                    line: number,
                    kind: ParseErrorKind::InvalidString,
                });
            }
            tokens.push(Token::Str(text));
            continue;
        }
        let start = index;
        let mut end = index + ch.len_utf8();
        chars.next();
        while let Some(&(next_index, next)) = chars.peek() {
            if next.is_whitespace() || next == ';' {
                break;
            }
            chars.next();
            end = next_index + next.len_utf8();
        }
        tokens.push(Token::Word(line[start..end].to_string()));
    }
    Ok(tokens)
}

/// An entry point whose function is resolved after the whole module
/// scope has been read.
struct PendingEntry {
    /// Line the entry point appeared on.
    line: usize,
    /// Execution model.
    model: ExecutionModel,
    /// Printed identifier of the function.
    func: String,
    /// Entry-point name recorded in the binary.
    name: String,
}

/// A decoration whose target is resolved after the whole module scope
/// has been read.
struct PendingDecor {
    /// Line the decoration appeared on.
    line: usize,
    /// Printed identifier of the target.
    target: String,
    /// Linkage to apply to the target.
    linkage: Linkage,
}

/// The span of one function definition in the tokenized source.
struct FunctionSpan {
    /// The function value.
    func: ValueId,
    /// Line number of the `OpFunction` line, for diagnostics.
    header_line: usize,
    /// First body line, inclusive.
    start: usize,
    /// `OpFunctionEnd` line, exclusive.
    end: usize,
}

/// Statement shapes recognized at module scope.
enum Stmt {
    /// The `target` header line.
    Target,
    /// An `OpEntryPoint` declaration.
    EntryPoint,
    /// An `OpDecorate` declaration.
    Decorate,
    /// A `%name = …` definition.
    Definition,
    /// Anything else.
    Other,
}

fn classify(line: &Line) -> Stmt {
    match line.tokens.first().and_then(Token::as_word) {
        Some("target") => Stmt::Target,
        Some("OpEntryPoint") => Stmt::EntryPoint,
        Some("OpDecorate") => Stmt::Decorate,
        Some(word) if word.starts_with('%') => Stmt::Definition,
        _ => Stmt::Other,
    }
}

/// The parser state: the module under construction plus the symbol
/// tables the printer's single identifier namespace implies.
struct Parser {
    /// The module being built.
    module: Module,
    /// Every identifier already defined, `%`-prefixed.
    used: BTreeSet<String>,
    /// Module-scope values by printed identifier.
    values: BTreeMap<String, ValueId>,
    /// Types by printed identifier.
    types: BTreeMap<String, TypeId>,
    /// Extended-instruction sets by printed identifier.
    ext_sets: BTreeMap<String, ExtSetId>,
    /// Entry points awaiting their functions.
    pending_entry_points: Vec<PendingEntry>,
    /// Decorations awaiting their targets.
    pending_decorations: Vec<PendingDecor>,
    /// Function bodies awaiting parsing, in text order.
    functions: Vec<FunctionSpan>,
}

impl Parser {
    fn new() -> Self {
        Self {
            module: Module::new(Target::opencl()),
            used: BTreeSet::new(),
            values: BTreeMap::new(),
            types: BTreeMap::new(),
            ext_sets: BTreeMap::new(),
            pending_entry_points: Vec::new(),
            pending_decorations: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn run(mut self, lines: &[Line]) -> Result<Module, ParseError> {
        let first = lines
            .iter()
            .find(|line| !line.tokens.is_empty())
            .ok_or(ParseError {
                line: 1,
                kind: ParseErrorKind::MissingTarget,
            })?;
        if !first
            .tokens
            .first()
            .is_some_and(|token| token.is_word("target"))
        {
            return Err(first.err(ParseErrorKind::MissingTarget));
        }
        self.parse_target(first)?;

        let first_index = lines
            .iter()
            .position(|line| !line.tokens.is_empty())
            .unwrap_or(0);
        let mut index = first_index + 1;
        while index < lines.len() {
            let line = &lines[index];
            if line.tokens.is_empty() {
                index += 1;
                continue;
            }
            match classify(line) {
                Stmt::Target => return Err(line.err(ParseErrorKind::UnexpectedToken)),
                Stmt::EntryPoint => {
                    self.parse_entry_point(line)?;
                    index += 1;
                }
                Stmt::Decorate => {
                    self.parse_decorate(line)?;
                    index += 1;
                }
                Stmt::Definition => match self.parse_definition(line, lines, index)? {
                    Some(end) => index = end + 1,
                    None => index += 1,
                },
                Stmt::Other => return Err(line.err(ParseErrorKind::UnknownStatement)),
            }
        }

        let functions = core::mem::take(&mut self.functions);
        for span in functions {
            self.parse_body(span, lines)?;
        }
        self.resolve_pending()?;
        Ok(self.module)
    }

    fn parse_target(&mut self, line: &Line) -> Result<(), ParseError> {
        expect_len(line, line.tokens.len(), 6)?;
        let name = word_at(&line.tokens, 1, line)?.to_string();
        if !line.tokens[2].is_word("address") || !line.tokens[4].is_word("memory") {
            return Err(line.err(ParseErrorKind::UnexpectedToken));
        }
        let addressing = match word_at(&line.tokens, 3, line)? {
            "logical" => AddressingModel::Logical,
            "physical32" => AddressingModel::Physical32,
            "physical64" => AddressingModel::Physical64,
            _ => return Err(line.err(ParseErrorKind::UnexpectedToken)),
        };
        let memory = match word_at(&line.tokens, 5, line)? {
            "simple" => MemoryModel::Simple,
            "glsl450" => MemoryModel::GLSL450,
            "opencl" => MemoryModel::OpenCL,
            _ => return Err(line.err(ParseErrorKind::UnexpectedToken)),
        };
        self.module.target = Target {
            name,
            addressing,
            memory,
        };
        Ok(())
    }

    /// Reads one `%name = …` definition; returns the `OpFunctionEnd`
    /// index when the definition is a function whose body must be
    /// skipped here.
    fn parse_definition(
        &mut self,
        line: &Line,
        lines: &[Line],
        index: usize,
    ) -> Result<Option<usize>, ParseError> {
        expect_len_guard(line, 3)?;
        if !line.tokens[1].is_word("=") {
            return Err(line.err(ParseErrorKind::UnexpectedToken));
        }
        let opcode = word_at(&line.tokens, 2, line)?;
        if opcode == "OpFunction" {
            let func = self.parse_function_header(line)?;
            let end = find_function_end(lines, index + 1, line)?;
            self.functions.push(FunctionSpan {
                func,
                header_line: line.number,
                start: index + 1,
                end,
            });
            return Ok(Some(end));
        }
        match opcode {
            "OpExtInstImport" => self.parse_ext_inst_import(line)?,
            "OpTypeVoid" | "OpTypeBool" | "OpTypeInt" | "OpTypeFloat" | "OpTypeVector" | "OpTypeArray"
            | "OpTypeStruct" | "OpTypePointer" | "OpTypeFunction" => {
                self.parse_type(line, opcode)?;
            }
            "OpConstant" | "OpConstantTrue" | "OpConstantFalse" | "OpConstantNull" | "OpUndef" => {
                self.parse_constant(line, opcode)?;
            }
            "OpVariable" => self.parse_global(line)?,
            _ => return Err(line.err(ParseErrorKind::UnknownStatement)),
        }
        Ok(None)
    }

    fn parse_function_header(&mut self, line: &Line) -> Result<ValueId, ParseError> {
        expect_len_guard(line, 5)?;
        let name = id_token(&line.tokens[0], line)?;
        let ret = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
        let sig = self.resolve_type(tok_at(&line.tokens, line.tokens.len() - 1, line)?, line)?;
        let signature_ret = match self.module.ty(sig) {
            Type::Function { ret, .. } => *ret,
            _ => return Err(line.err(ParseErrorKind::ExpectedFunctionType)),
        };
        if signature_ret != ret {
            return Err(line.err(ParseErrorKind::ResultMismatch));
        }
        let control = parse_function_control(&line.tokens[4..line.tokens.len() - 1], line)?;
        self.claim(line, name)?;
        let func = self
            .module
            .add_function(&name[1..], sig, Linkage::External)
            .map_err(|error| line.err(ParseErrorKind::Build(error)))?;
        self.values.insert(name.to_string(), func);
        if let Some(function) = self.module.function_mut(func) {
            function.control = control;
        }
        Ok(func)
    }

    fn parse_ext_inst_import(&mut self, line: &Line) -> Result<(), ParseError> {
        expect_len(line, line.tokens.len(), 4)?;
        let name = id_token(&line.tokens[0], line)?;
        let set = string_at(&line.tokens, 3, line)?.to_string();
        self.claim(line, name)?;
        let id = self.module.add_ext_inst_set(&set);
        self.ext_sets.insert(name.to_string(), id);
        Ok(())
    }

    fn parse_type(&mut self, line: &Line, opcode: &str) -> Result<(), ParseError> {
        let ty = match opcode {
            "OpTypeVoid" => {
                expect_len(line, line.tokens.len(), 3)?;
                Type::Void
            }
            "OpTypeBool" => {
                expect_len(line, line.tokens.len(), 3)?;
                Type::Bool
            }
            "OpTypeInt" => {
                expect_len(line, line.tokens.len(), 5)?;
                let bits = parse_int::<u8>(tok_at(&line.tokens, 3, line)?, line)?;
                let signed = parse_int::<u8>(tok_at(&line.tokens, 4, line)?, line)?;
                if signed > 1 {
                    return Err(line.err(ParseErrorKind::UnexpectedToken));
                }
                Type::Int {
                    bits,
                    signed: signed == 1,
                }
            }
            "OpTypeFloat" => {
                expect_len(line, line.tokens.len(), 4)?;
                let bits = parse_int::<u16>(tok_at(&line.tokens, 3, line)?, line)?;
                Type::Float { bits }
            }
            "OpTypeVector" | "OpTypeArray" => {
                expect_len(line, line.tokens.len(), 5)?;
                let elem = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
                let len = parse_int::<u32>(tok_at(&line.tokens, 4, line)?, line)?;
                if opcode == "OpTypeVector" {
                    Type::Vector { elem, len }
                } else {
                    Type::Array { elem, len }
                }
            }
            "OpTypeStruct" => {
                expect_len_guard(line, 3)?;
                let mut fields = Vec::new();
                for index in 3..line.tokens.len() {
                    fields.push(self.resolve_type(tok_at(&line.tokens, index, line)?, line)?);
                }
                Type::Struct { fields }
            }
            "OpTypePointer" => {
                expect_len(line, line.tokens.len(), 5)?;
                let storage = parse_storage(tok_at(&line.tokens, 3, line)?, line)?;
                let pointee = self.resolve_type(tok_at(&line.tokens, 4, line)?, line)?;
                Type::Pointer { storage, pointee }
            }
            "OpTypeFunction" => {
                expect_len_guard(line, 4)?;
                let ret = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
                let mut params = Vec::new();
                for index in 4..line.tokens.len() {
                    params.push(self.resolve_type(tok_at(&line.tokens, index, line)?, line)?);
                }
                Type::Function { ret, params }
            }
            _ => return Err(line.err(ParseErrorKind::UnknownStatement)),
        };
        let name = id_token(&line.tokens[0], line)?;
        self.claim(line, name)?;
        let id = self.module.intern_named_type(ty, &name[1..]);
        self.types.insert(name.to_string(), id);
        Ok(())
    }

    fn parse_constant(&mut self, line: &Line, opcode: &str) -> Result<(), ParseError> {
        let (ty, payload) = match opcode {
            "OpConstant" => {
                expect_len(line, line.tokens.len(), 5)?;
                let ty = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
                let payload = self.int_literal(line, ty, tok_at(&line.tokens, 4, line)?)?;
                (ty, payload)
            }
            "OpConstantTrue" | "OpConstantFalse" | "OpConstantNull" | "OpUndef" => {
                expect_len(line, line.tokens.len(), 4)?;
                let ty = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
                let payload = match opcode {
                    "OpConstantTrue" => ConstValue::Bool(true),
                    "OpConstantFalse" => ConstValue::Bool(false),
                    "OpConstantNull" => ConstValue::Null,
                    _ => ConstValue::Undef,
                };
                (ty, payload)
            }
            _ => return Err(line.err(ParseErrorKind::UnknownStatement)),
        };
        let name = id_token(&line.tokens[0], line)?;
        self.claim(line, name)?;
        let id = self
            .module
            .intern_const(ty, payload)
            .map_err(|error| match error {
                BuildError::ConstantTypeMismatch if opcode == "OpConstant" => {
                    line.err(ParseErrorKind::InvalidNumber)
                }
                other => line.err(ParseErrorKind::Build(other)),
            })?;
        if !self.module.value(id).name.is_empty() {
            return Err(line.err(ParseErrorKind::DuplicateId));
        }
        self.module.set_value_name(id, &name[1..]);
        self.values.insert(name.to_string(), id);
        Ok(())
    }

    fn parse_global(&mut self, line: &Line) -> Result<(), ParseError> {
        if line.tokens.len() < 5 {
            return Err(line.err(ParseErrorKind::UnexpectedEndOfLine));
        }
        if line.tokens.len() > 6 {
            return Err(line.err(ParseErrorKind::WrongOperandCount));
        }
        let ty = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
        let storage = parse_storage(tok_at(&line.tokens, 4, line)?, line)?;
        let pointee_storage = self
            .module
            .pointer_parts(ty)
            .map(|(storage, _)| storage)
            .ok_or(line.err(ParseErrorKind::ExpectedPointerType))?;
        if storage != pointee_storage {
            return Err(line.err(ParseErrorKind::StorageMismatch));
        }
        let init = match line.tokens.get(5) {
            Some(token) => Some(self.resolve_module_value(token, line)?),
            None => None,
        };
        let name = id_token(&line.tokens[0], line)?;
        self.claim(line, name)?;
        let id = self
            .module
            .add_global(&name[1..], ty, Linkage::External, init)
            .map_err(|error| line.err(ParseErrorKind::Build(error)))?;
        self.values.insert(name.to_string(), id);
        Ok(())
    }

    fn parse_entry_point(&mut self, line: &Line) -> Result<(), ParseError> {
        expect_len(line, line.tokens.len(), 4)?;
        let model = match word_at(&line.tokens, 1, line)? {
            "Kernel" => ExecutionModel::Kernel,
            "GLCompute" => ExecutionModel::GLCompute,
            _ => return Err(line.err(ParseErrorKind::UnexpectedToken)),
        };
        let func = id_token(tok_at(&line.tokens, 2, line)?, line)?.to_string();
        let name = string_at(&line.tokens, 3, line)?.to_string();
        self.pending_entry_points.push(PendingEntry {
            line: line.number,
            model,
            func,
            name,
        });
        Ok(())
    }

    fn parse_decorate(&mut self, line: &Line) -> Result<(), ParseError> {
        expect_len(line, line.tokens.len(), 5)?;
        let target = id_token(tok_at(&line.tokens, 1, line)?, line)?.to_string();
        if !line.tokens[2].is_word("LinkageAttributes") {
            return Err(line.err(ParseErrorKind::UnexpectedToken));
        }
        string_at(&line.tokens, 3, line)?;
        let linkage = match word_at(&line.tokens, 4, line)? {
            "Internal" => Linkage::Internal,
            "Import" => Linkage::Import,
            "Export" => Linkage::Export,
            "External" => Linkage::External,
            _ => return Err(line.err(ParseErrorKind::UnexpectedToken)),
        };
        self.pending_decorations.push(PendingDecor {
            line: line.number,
            target,
            linkage,
        });
        Ok(())
    }

    fn resolve_pending(&mut self) -> Result<(), ParseError> {
        let entries = core::mem::take(&mut self.pending_entry_points);
        for entry in entries {
            let id = self
                .values
                .get(&entry.func)
                .copied()
                .ok_or(ParseError {
                    line: entry.line,
                    kind: ParseErrorKind::UnknownId,
                })?;
            if self.module.function(id).is_none() {
                return Err(ParseError {
                    line: entry.line,
                    kind: ParseErrorKind::NotAFunction,
                });
            }
            self.module
                .set_entry_point(entry.model, id, &entry.name)
                .map_err(|error| ParseError {
                    line: entry.line,
                    kind: ParseErrorKind::Build(error),
                })?;
        }
        let decorations = core::mem::take(&mut self.pending_decorations);
        for decor in decorations {
            let id = self
                .values
                .get(&decor.target)
                .copied()
                .ok_or(ParseError {
                    line: decor.line,
                    kind: ParseErrorKind::UnknownId,
                })?;
            if let Some(function) = self.module.function_mut(id) {
                function.linkage = decor.linkage;
            } else if let Some(global) = self.module.global_mut(id) {
                global.linkage = decor.linkage;
            } else {
                return Err(ParseError {
                    line: decor.line,
                    kind: ParseErrorKind::NotDecoratable,
                });
            }
        }
        Ok(())
    }

    fn claim(&mut self, line: &Line, name: &str) -> Result<(), ParseError> {
        if self.used.insert(name.to_string()) {
            Ok(())
        } else {
            Err(line.err(ParseErrorKind::DuplicateId))
        }
    }

    fn resolve_type(&self, token: &Token, line: &Line) -> Result<TypeId, ParseError> {
        let name = id_token(token, line)?;
        self.types
            .get(name)
            .copied()
            .ok_or_else(|| line.err(ParseErrorKind::UnknownId))
    }

    fn resolve_ext_set(&self, token: &Token, line: &Line) -> Result<ExtSetId, ParseError> {
        let name = id_token(token, line)?;
        self.ext_sets
            .get(name)
            .copied()
            .ok_or_else(|| line.err(ParseErrorKind::UnknownId))
    }

    fn resolve_module_value(&self, token: &Token, line: &Line) -> Result<ValueId, ParseError> {
        let name = id_token(token, line)?;
        self.values
            .get(name)
            .copied()
            .ok_or_else(|| line.err(ParseErrorKind::UnknownId))
    }

    fn resolve_value(
        &self,
        locals: &BTreeMap<String, ValueId>,
        token: &Token,
        line: &Line,
    ) -> Result<ValueId, ParseError> {
        let name = id_token(token, line)?;
        locals
            .get(name)
            .or_else(|| self.values.get(name))
            .copied()
            .ok_or_else(|| line.err(ParseErrorKind::UnknownId))
    }

    /// Reads the literal of an `OpConstant` at the shape of `ty`.
    fn int_literal(&self, line: &Line, ty: TypeId, token: &Token) -> Result<ConstValue, ParseError> {
        match self.module.ty(ty) {
            Type::Int { bits, signed } => {
                let word = word_of(token, line)?;
                let raw = if word.starts_with('-') {
                    if !signed {
                        return Err(line.err(ParseErrorKind::InvalidNumber));
                    }
                    let value = word
                        .parse::<i64>()
                        .map_err(|_| line.err(ParseErrorKind::InvalidNumber))?;
                    mask_bits(value as u64, *bits)
                } else {
                    word.parse::<u64>()
                        .map_err(|_| line.err(ParseErrorKind::InvalidNumber))?
                };
                Ok(ConstValue::Int(raw))
            }
            Type::Float { bits: 32 } => {
                let value = word_of(token, line)?
                    .parse::<f32>()
                    .map_err(|_| line.err(ParseErrorKind::InvalidNumber))?;
                Ok(ConstValue::from_f32_bits(value.to_bits()))
            }
            Type::Float { bits: 64 } => {
                let value = word_of(token, line)?
                    .parse::<f64>()
                    .map_err(|_| line.err(ParseErrorKind::InvalidNumber))?;
                Ok(ConstValue::from_f64_bits(value.to_bits()))
            }
            _ => Err(line.err(ParseErrorKind::Build(BuildError::ConstantTypeMismatch))),
        }
    }

    /// Parses one function body: parameters, then a pre-scan that
    /// registers labels and instruction results, then the
    /// instructions themselves.
    fn parse_body(&mut self, span: FunctionSpan, lines: &[Line]) -> Result<(), ParseError> {
        let FunctionSpan {
            func,
            header_line,
            start,
            end,
        } = span;
        let args = match self.module.function(func) {
            Some(function) => function.args.clone(),
            None => {
                return Err(ParseError {
                    line: header_line,
                    kind: ParseErrorKind::NotAFunction,
                });
            }
        };
        let mut locals: BTreeMap<String, ValueId> = BTreeMap::new();
        let mut labels: BTreeMap<String, BlockId> = BTreeMap::new();

        let mut index = start;
        let mut seen_params = 0usize;
        loop {
            while index < end
                && lines
                    .get(index)
                    .is_some_and(|line| line.tokens.is_empty())
            {
                index += 1;
            }
            let Some(line) = lines.get(index) else {
                break;
            };
            if !is_param_line(line) {
                break;
            }
            self.parse_param(line, &args, seen_params, &mut locals)?;
            seen_params += 1;
            index += 1;
        }
        if seen_params != args.len() {
            return Err(ParseError {
                line: header_line,
                kind: ParseErrorKind::WrongOperandCount,
            });
        }

        let body = lines.get(start..end).unwrap_or(&[]);
        let mut label_names: Vec<String> = Vec::new();
        for line in body {
            if line.tokens.is_empty() {
                continue;
            }
            let Ok((Some(name), rest)) = split_form(line) else {
                continue;
            };
            let Some(opcode) = rest.first().and_then(Token::as_word) else {
                continue;
            };
            match opcode {
                "OpLabel" => {
                    self.claim(line, name)?;
                    label_names.push(name.to_string());
                }
                op if is_result_op(op) => {
                    let ty = self.resolve_type(tok_at(rest, 1, line)?, line)?;
                    self.claim(line, name)?;
                    let id = self.module.new_inst_value(ty);
                    self.module.set_value_name(id, &name[1..]);
                    locals.insert(name.to_string(), id);
                }
                _ => {}
            }
        }

        if !label_names.is_empty() {
            self.module
                .begin_body(func)
                .map_err(|error| ParseError {
                    line: header_line,
                    kind: ParseErrorKind::Build(error),
                })?;
            for name in &label_names {
                let block = self
                    .module
                    .push_block(func, &name[1..])
                    .map_err(|error| ParseError {
                        line: header_line,
                        kind: ParseErrorKind::Build(error),
                    })?;
                labels.insert(name.clone(), block);
            }
        }

        let ctx = Body {
            locals: &locals,
            labels: &labels,
        };
        let mut current: Option<BlockId> = None;
        for line in lines.get(index..end).unwrap_or(&[]) {
            if line.tokens.is_empty() {
                continue;
            }
            let (result, rest) = split_form(line)?;
            let opcode = word_at(rest, 0, line)?;
            if opcode == "OpLabel" {
                let name = result.ok_or_else(|| line.err(ParseErrorKind::UnexpectedToken))?;
                expect_len(line, rest.len(), 1)?;
                let block = labels
                    .get(name)
                    .copied()
                    .ok_or_else(|| line.err(ParseErrorKind::UnknownId))?;
                current = Some(block);
                continue;
            }
            let block = current.ok_or_else(|| line.err(ParseErrorKind::MissingLabel))?;
            let inst = self.parse_inst(&ctx, line, result, rest)?;
            self.module
                .emit(func, block, inst)
                .map_err(|error| line.err(ParseErrorKind::Build(error)))?;
        }
        Ok(())
    }

    fn parse_param(
        &mut self,
        line: &Line,
        args: &[ValueId],
        position: usize,
        locals: &mut BTreeMap<String, ValueId>,
    ) -> Result<(), ParseError> {
        expect_len(line, line.tokens.len(), 4)?;
        let name = id_token(&line.tokens[0], line)?;
        let ty = self.resolve_type(tok_at(&line.tokens, 3, line)?, line)?;
        let arg = *args
            .get(position)
            .ok_or_else(|| line.err(ParseErrorKind::WrongOperandCount))?;
        if self.module.type_of(arg) != ty {
            return Err(line.err(ParseErrorKind::ResultMismatch));
        }
        self.claim(line, name)?;
        self.module.set_value_name(arg, &name[1..]);
        locals.insert(name.to_string(), arg);
        Ok(())
    }
}

/// The per-body symbol tables instruction parsing resolves against.
struct Body<'a> {
    /// Parameter and instruction-result values of this function.
    locals: &'a BTreeMap<String, ValueId>,
    /// Labels of this function by printed identifier.
    labels: &'a BTreeMap<String, BlockId>,
}

/// True when the line is a leading `OpFunctionParameter` definition.
fn is_param_line(line: &Line) -> bool {
    line.tokens.len() >= 3
        && line.tokens[1].is_word("=")
        && matches!(line.tokens.get(2), Some(token) if token.is_word("OpFunctionParameter"))
}

/// Finds the `OpFunctionEnd` line at or after `from`.
fn find_function_end(lines: &[Line], from: usize, header: &Line) -> Result<usize, ParseError> {
    for (index, line) in lines.iter().enumerate().skip(from) {
        if line.tokens.len() == 1 && line.tokens[0].is_word("OpFunctionEnd") {
            return Ok(index);
        }
    }
    Err(header.err(ParseErrorKind::UnexpectedEndOfLine))
}

impl Parser {
    /// Parses one instruction line into an [`Inst`].
    fn parse_inst(
        &self,
        ctx: &Body<'_>,
        line: &Line,
        result: Option<&str>,
        rest: &[Token],
    ) -> Result<Inst, ParseError> {
        let opcode = word_at(rest, 0, line)?;
        if opcode == "OpFunctionParameter" {
            return Err(line.err(ParseErrorKind::MisplacedParameter));
        }
        if is_result_op(opcode) {
            let name = result.ok_or_else(|| line.err(ParseErrorKind::ResultMismatch))?;
            let id = ctx
                .locals
                .get(name)
                .copied()
                .ok_or_else(|| line.err(ParseErrorKind::UnknownId))?;
            if rest.len() < 2 {
                return Err(line.err(ParseErrorKind::UnexpectedEndOfLine));
            }
            let ty = self.module.type_of(id);
            let op = self.parse_result_op(ctx, opcode, ty, &rest[2..], line)?;
            return Ok(Inst::def(id, ty, op));
        }
        if !is_statement_op(opcode) {
            return Err(line.err(ParseErrorKind::UnknownStatement));
        }
        if result.is_some() {
            return Err(line.err(ParseErrorKind::UnexpectedToken));
        }
        let op = self.parse_statement_op(ctx, opcode, &rest[1..], line)?;
        Ok(Inst::none(op))
    }

    fn parse_result_op(
        &self,
        ctx: &Body<'_>,
        opcode: &str,
        ty: TypeId,
        ops: &[Token],
        line: &Line,
    ) -> Result<Op, ParseError> {
        match opcode {
            "OpVariable" => {
                let (storage_tok, init_tok) = match ops {
                    [storage] => (storage, None),
                    [storage, init] => (storage, Some(init)),
                    _ => {
                        return Err(if ops.is_empty() {
                            line.err(ParseErrorKind::UnexpectedEndOfLine)
                        } else {
                            line.err(ParseErrorKind::WrongOperandCount)
                        });
                    }
                };
                let storage = parse_storage(storage_tok, line)?;
                let pointer_storage = self
                    .module
                    .pointer_parts(ty)
                    .map(|(storage, _)| storage)
                    .ok_or(line.err(ParseErrorKind::ExpectedPointerType))?;
                if storage != pointer_storage {
                    return Err(line.err(ParseErrorKind::StorageMismatch));
                }
                let init = match init_tok {
                    Some(token) => Some(self.resolve_value(ctx.locals, token, line)?),
                    None => None,
                };
                Ok(Op::Variable { init })
            }
            "OpLoad" => {
                let [ptr] = ops else {
                    return Err(arity_err(line, ops.len(), 1));
                };
                Ok(Op::Load {
                    ptr: self.resolve_value(ctx.locals, ptr, line)?,
                })
            }
            "OpAccessChain" | "OpPtrAccessChain" => {
                let (base, indices) = ops
                    .split_first()
                    .ok_or_else(|| line.err(ParseErrorKind::UnexpectedEndOfLine))?;
                let mut resolved = Vec::new();
                for token in indices {
                    resolved.push(self.resolve_value(ctx.locals, token, line)?);
                }
                let base = self.resolve_value(ctx.locals, base, line)?;
                if opcode == "OpPtrAccessChain" {
                    Ok(Op::PtrAccessChain {
                        base,
                        indices: resolved,
                    })
                } else {
                    Ok(Op::AccessChain {
                        base,
                        indices: resolved,
                    })
                }
            }
            "OpCopyObject" => {
                let [operand] = ops else {
                    return Err(arity_err(line, ops.len(), 1));
                };
                Ok(Op::CopyObject {
                    operand: self.resolve_value(ctx.locals, operand, line)?,
                })
            }
            "OpSelect" => {
                let [cond, a, b] = ops else {
                    return Err(arity_err(line, ops.len(), 3));
                };
                Ok(Op::Select {
                    cond: self.resolve_value(ctx.locals, cond, line)?,
                    a: self.resolve_value(ctx.locals, a, line)?,
                    b: self.resolve_value(ctx.locals, b, line)?,
                })
            }
            "OpFunctionCall" => {
                let (callee, args) = ops
                    .split_first()
                    .ok_or_else(|| line.err(ParseErrorKind::UnexpectedEndOfLine))?;
                let mut resolved = Vec::new();
                for token in args {
                    resolved.push(self.resolve_value(ctx.locals, token, line)?);
                }
                Ok(Op::Call {
                    callee: self.resolve_value(ctx.locals, callee, line)?,
                    args: resolved,
                })
            }
            "OpExtInst" => {
                if ops.len() < 2 {
                    return Err(line.err(ParseErrorKind::UnexpectedEndOfLine));
                }
                let set = self.resolve_ext_set(&ops[0], line)?;
                let inst = parse_int::<u32>(&ops[1], line)?;
                let mut resolved = Vec::new();
                for token in &ops[2..] {
                    resolved.push(self.resolve_value(ctx.locals, token, line)?);
                }
                Ok(Op::ExtInst {
                    set,
                    inst,
                    args: resolved,
                })
            }
            "OpCompositeExtract" => {
                let (composite, indices) = ops
                    .split_first()
                    .ok_or_else(|| line.err(ParseErrorKind::UnexpectedEndOfLine))?;
                let mut resolved = Vec::new();
                for token in indices {
                    resolved.push(parse_int::<u32>(token, line)?);
                }
                Ok(Op::CompositeExtract {
                    composite: self.resolve_value(ctx.locals, composite, line)?,
                    indices: resolved,
                })
            }
            "OpCompositeConstruct" => {
                let mut constituents = Vec::new();
                for token in ops {
                    constituents.push(self.resolve_value(ctx.locals, token, line)?);
                }
                Ok(Op::CompositeConstruct { constituents })
            }
            "OpPhi" => {
                if ops.len() < 2 || !ops.len().is_multiple_of(2) {
                    return Err(line.err(ParseErrorKind::WrongOperandCount));
                }
                let mut incomings = Vec::with_capacity(ops.len() / 2);
                for start in (0..ops.len()).step_by(2) {
                    let value = self.resolve_value(ctx.locals, &ops[start], line)?;
                    let block = resolve_label(ctx.labels, &ops[start + 1], line)?;
                    incomings.push((value, block));
                }
                Ok(Op::Phi { incomings })
            }
            _ => {
                if let Some(op) = bin_from_opcode(opcode) {
                    let [lhs, rhs] = ops else {
                        return Err(arity_err(line, ops.len(), 2));
                    };
                    return Ok(Op::Binary {
                        op,
                        lhs: self.resolve_value(ctx.locals, lhs, line)?,
                        rhs: self.resolve_value(ctx.locals, rhs, line)?,
                    });
                }
                if let Some(op) = un_from_opcode(opcode) {
                    let [operand] = ops else {
                        return Err(arity_err(line, ops.len(), 1));
                    };
                    return Ok(Op::Unary {
                        op,
                        operand: self.resolve_value(ctx.locals, operand, line)?,
                    });
                }
                if let Some(op) = cmp_from_opcode(opcode) {
                    let [lhs, rhs] = ops else {
                        return Err(arity_err(line, ops.len(), 2));
                    };
                    return Ok(Op::Compare {
                        op,
                        lhs: self.resolve_value(ctx.locals, lhs, line)?,
                        rhs: self.resolve_value(ctx.locals, rhs, line)?,
                    });
                }
                if let Some(op) = conv_from_opcode(opcode) {
                    let [operand] = ops else {
                        return Err(arity_err(line, ops.len(), 1));
                    };
                    return Ok(Op::Convert {
                        op,
                        operand: self.resolve_value(ctx.locals, operand, line)?,
                    });
                }
                Err(line.err(ParseErrorKind::UnknownStatement))
            }
        }
    }

    fn parse_statement_op(
        &self,
        ctx: &Body<'_>,
        opcode: &str,
        ops: &[Token],
        line: &Line,
    ) -> Result<Op, ParseError> {
        match opcode {
            "OpStore" => {
                let [ptr, value] = ops else {
                    return Err(arity_err(line, ops.len(), 2));
                };
                Ok(Op::Store {
                    ptr: self.resolve_value(ctx.locals, ptr, line)?,
                    value: self.resolve_value(ctx.locals, value, line)?,
                })
            }
            "OpControlBarrier" => {
                let [exec, mem, semantics] = ops else {
                    return Err(arity_err(line, ops.len(), 3));
                };
                Ok(Op::ControlBarrier {
                    exec: self.resolve_value(ctx.locals, exec, line)?,
                    mem: self.resolve_value(ctx.locals, mem, line)?,
                    semantics: self.resolve_value(ctx.locals, semantics, line)?,
                })
            }
            "OpMemoryBarrier" => {
                let [mem, semantics] = ops else {
                    return Err(arity_err(line, ops.len(), 2));
                };
                Ok(Op::MemoryBarrier {
                    mem: self.resolve_value(ctx.locals, mem, line)?,
                    semantics: self.resolve_value(ctx.locals, semantics, line)?,
                })
            }
            "OpSelectionMerge" => {
                let [target, control] = ops else {
                    return Err(arity_err(line, ops.len(), 2));
                };
                Ok(Op::SelectionMerge {
                    target: resolve_label(ctx.labels, target, line)?,
                    control: parse_mask(control, line)?,
                })
            }
            "OpLoopMerge" => {
                let [merge, cont, control] = ops else {
                    return Err(arity_err(line, ops.len(), 3));
                };
                Ok(Op::LoopMerge {
                    merge: resolve_label(ctx.labels, merge, line)?,
                    cont: resolve_label(ctx.labels, cont, line)?,
                    control: parse_mask(control, line)?,
                })
            }
            "OpBranch" => {
                let [target] = ops else {
                    return Err(arity_err(line, ops.len(), 1));
                };
                Ok(Op::Branch {
                    target: resolve_label(ctx.labels, target, line)?,
                })
            }
            "OpBranchConditional" => {
                let [cond, then, other] = ops else {
                    return Err(arity_err(line, ops.len(), 3));
                };
                Ok(Op::BranchConditional {
                    cond: self.resolve_value(ctx.locals, cond, line)?,
                    then: resolve_label(ctx.labels, then, line)?,
                    other: resolve_label(ctx.labels, other, line)?,
                })
            }
            "OpReturn" => {
                let [] = ops else {
                    return Err(arity_err(line, ops.len(), 0));
                };
                Ok(Op::Return)
            }
            "OpReturnValue" => {
                let [value] = ops else {
                    return Err(arity_err(line, ops.len(), 1));
                };
                Ok(Op::ReturnValue {
                    value: self.resolve_value(ctx.locals, value, line)?,
                })
            }
            "OpUnreachable" => {
                let [] = ops else {
                    return Err(arity_err(line, ops.len(), 0));
                };
                Ok(Op::Unreachable)
            }
            _ => Err(line.err(ParseErrorKind::UnknownStatement)),
        }
    }
}

/// True when the opcode's assembly form carries a result id and type.
fn is_result_op(opcode: &str) -> bool {
    matches!(
        opcode,
        "OpVariable"
            | "OpLoad"
            | "OpAccessChain"
            | "OpPtrAccessChain"
            | "OpCopyObject"
            | "OpSelect"
            | "OpFunctionCall"
            | "OpExtInst"
            | "OpCompositeExtract"
            | "OpCompositeConstruct"
            | "OpPhi"
    ) || bin_from_opcode(opcode).is_some()
        || un_from_opcode(opcode).is_some()
        || cmp_from_opcode(opcode).is_some()
        || conv_from_opcode(opcode).is_some()
}

/// True when the opcode's assembly form is a bare statement.
fn is_statement_op(opcode: &str) -> bool {
    matches!(
        opcode,
        "OpStore"
            | "OpControlBarrier"
            | "OpMemoryBarrier"
            | "OpSelectionMerge"
            | "OpLoopMerge"
            | "OpBranch"
            | "OpBranchConditional"
            | "OpReturn"
            | "OpReturnValue"
            | "OpUnreachable"
    )
}

fn bin_from_opcode(opcode: &str) -> Option<BinOp> {
    let mnemonic = opcode.strip_prefix("Op")?;
    Some(match mnemonic {
        "IAdd" => BinOp::IAdd,
        "ISub" => BinOp::ISub,
        "IMul" => BinOp::IMul,
        "SDiv" => BinOp::SDiv,
        "UDiv" => BinOp::UDiv,
        "SRem" => BinOp::SRem,
        "UMod" => BinOp::UMod,
        "FAdd" => BinOp::FAdd,
        "FSub" => BinOp::FSub,
        "FMul" => BinOp::FMul,
        "FDiv" => BinOp::FDiv,
        "FRem" => BinOp::FRem,
        "BitwiseAnd" => BinOp::BitwiseAnd,
        "BitwiseOr" => BinOp::BitwiseOr,
        "BitwiseXor" => BinOp::BitwiseXor,
        "ShiftLeftLogical" => BinOp::ShiftLeftLogical,
        "ShiftRightArithmetic" => BinOp::ShiftRightArithmetic,
        "ShiftRightLogical" => BinOp::ShiftRightLogical,
        "LogicalAnd" => BinOp::LogicalAnd,
        "LogicalOr" => BinOp::LogicalOr,
        _ => return None,
    })
}

fn un_from_opcode(opcode: &str) -> Option<UnOp> {
    let mnemonic = opcode.strip_prefix("Op")?;
    Some(match mnemonic {
        "Not" => UnOp::Not,
        "SNegate" => UnOp::SNegate,
        "FNegate" => UnOp::FNegate,
        "LogicalNot" => UnOp::LogicalNot,
        _ => return None,
    })
}

fn cmp_from_opcode(opcode: &str) -> Option<CmpOp> {
    let mnemonic = opcode.strip_prefix("Op")?;
    Some(match mnemonic {
        "IEqual" => CmpOp::IEqual,
        "INotEqual" => CmpOp::INotEqual,
        "SLessThan" => CmpOp::SLessThan,
        "SLessThanEqual" => CmpOp::SLessThanEqual,
        "SGreaterThan" => CmpOp::SGreaterThan,
        "SGreaterThanEqual" => CmpOp::SGreaterThanEqual,
        "ULessThan" => CmpOp::ULessThan,
        "ULessThanEqual" => CmpOp::ULessThanEqual,
        "UGreaterThan" => CmpOp::UGreaterThan,
        "UGreaterThanEqual" => CmpOp::UGreaterThanEqual,
        "FOrdEqual" => CmpOp::FOrdEqual,
        "FOrdNotEqual" => CmpOp::FOrdNotEqual,
        "FOrdLessThan" => CmpOp::FOrdLessThan,
        "FOrdLessThanEqual" => CmpOp::FOrdLessThanEqual,
        "FOrdGreaterThan" => CmpOp::FOrdGreaterThan,
        "FOrdGreaterThanEqual" => CmpOp::FOrdGreaterThanEqual,
        _ => return None,
    })
}

fn conv_from_opcode(opcode: &str) -> Option<ConvOp> {
    let mnemonic = opcode.strip_prefix("Op")?;
    Some(match mnemonic {
        "SConvert" => ConvOp::SConvert,
        "UConvert" => ConvOp::UConvert,
        "FConvert" => ConvOp::FConvert,
        "ConvertFToS" => ConvOp::ConvertFToS,
        "ConvertFToU" => ConvOp::ConvertFToU,
        "ConvertSToF" => ConvOp::ConvertSToF,
        "ConvertUToF" => ConvOp::ConvertUToF,
        "Bitcast" => ConvOp::Bitcast,
        _ => return None,
    })
}

/// Requires exactly `want` tokens on the line.
fn expect_len(line: &Line, len: usize, want: usize) -> Result<(), ParseError> {
    if len == want {
        Ok(())
    } else {
        Err(arity_err(line, len, want))
    }
}

/// Requires at least `want` tokens on the line.
fn expect_len_guard(line: &Line, want: usize) -> Result<(), ParseError> {
    if line.tokens.len() < want {
        Err(line.err(ParseErrorKind::UnexpectedEndOfLine))
    } else {
        Ok(())
    }
}

/// Builds the arity error for `len` tokens where `want` were required.
fn arity_err(line: &Line, len: usize, want: usize) -> ParseError {
    if len < want {
        line.err(ParseErrorKind::UnexpectedEndOfLine)
    } else {
        line.err(ParseErrorKind::WrongOperandCount)
    }
}

/// The token at `index`, or the end-of-line error.
fn tok_at<'t>(tokens: &'t [Token], index: usize, line: &Line) -> Result<&'t Token, ParseError> {
    tokens
        .get(index)
        .ok_or_else(|| line.err(ParseErrorKind::UnexpectedEndOfLine))
}

/// The bare word at `index`.
fn word_at<'t>(tokens: &'t [Token], index: usize, line: &Line) -> Result<&'t str, ParseError> {
    tok_at(tokens, index, line)?
        .as_word()
        .ok_or_else(|| line.err(ParseErrorKind::UnexpectedToken))
}

/// The quoted string at `index`.
fn string_at<'t>(tokens: &'t [Token], index: usize, line: &Line) -> Result<&'t str, ParseError> {
    match tok_at(tokens, index, line)? {
        Token::Str(text) => Ok(text),
        _ => Err(line.err(ParseErrorKind::UnexpectedToken)),
    }
}

/// The `%`-prefixed identifier carried by `token`.
fn id_token<'t>(token: &'t Token, line: &Line) -> Result<&'t str, ParseError> {
    match token {
        Token::Word(word) if word.starts_with('%') && word.len() > 1 => Ok(word),
        _ => Err(line.err(ParseErrorKind::UnexpectedToken)),
    }
}

/// The bare word carried by `token`.
fn word_of<'t>(token: &'t Token, line: &Line) -> Result<&'t str, ParseError> {
    token
        .as_word()
        .ok_or_else(|| line.err(ParseErrorKind::UnexpectedToken))
}

/// Parses a numeric literal token.
fn parse_int<T: core::str::FromStr>(token: &Token, line: &Line) -> Result<T, ParseError> {
    let word = word_of(token, line)?;
    word.parse::<T>()
        .map_err(|_| line.err(ParseErrorKind::InvalidNumber))
}

/// Reads a storage-class word.
fn parse_storage(token: &Token, line: &Line) -> Result<Storage, ParseError> {
    match word_of(token, line)? {
        "Function" => Ok(Storage::Function),
        "Workgroup" => Ok(Storage::Workgroup),
        "CrossWorkgroup" => Ok(Storage::CrossWorkgroup),
        "UniformConstant" => Ok(Storage::UniformConstant),
        "Input" => Ok(Storage::Input),
        "Output" => Ok(Storage::Output),
        "Private" => Ok(Storage::Private),
        _ => Err(line.err(ParseErrorKind::UnexpectedToken)),
    }
}

/// Reads a merge-control mask word (`None` or a literal).
fn parse_mask(token: &Token, line: &Line) -> Result<u32, ParseError> {
    let word = word_of(token, line)?;
    if word == "None" {
        return Ok(0);
    }
    if let Some(hex) = word.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).map_err(|_| line.err(ParseErrorKind::InvalidNumber));
    }
    word.parse::<u32>()
        .map_err(|_| line.err(ParseErrorKind::InvalidNumber))
}

/// Reads a function-control mask (`None`, flag names, or literals).
fn parse_function_control(tokens: &[Token], line: &Line) -> Result<FunctionControl, ParseError> {
    if tokens.is_empty() {
        return Err(line.err(ParseErrorKind::UnexpectedEndOfLine));
    }
    if tokens.len() == 1 && tokens[0].is_word("None") {
        return Ok(FunctionControl::NONE);
    }
    let mut bits = 0u32;
    for token in tokens {
        let word = word_of(token, line)?;
        if word == "None" {
            return Err(line.err(ParseErrorKind::UnexpectedToken));
        }
        let flag = match word {
            "Inline" => 1u32,
            "DontInline" => 2,
            "Pure" => 4,
            "Const" => 8,
            literal => {
                let hex = literal
                    .strip_prefix("0x")
                    .ok_or_else(|| line.err(ParseErrorKind::UnexpectedToken))?;
                u32::from_str_radix(hex, 16).map_err(|_| line.err(ParseErrorKind::InvalidNumber))?
            }
        };
        bits |= flag;
    }
    Ok(FunctionControl(bits))
}

/// Zero-extends (or sign-extends) `raw` to the type's width.
fn mask_bits(raw: u64, bits: u8) -> u64 {
    if bits == 0 || bits >= 64 {
        raw
    } else {
        raw & ((1u64 << bits) - 1)
    }
}

/// Resolves a label operand inside one function.
fn resolve_label(
    labels: &BTreeMap<String, BlockId>,
    token: &Token,
    line: &Line,
) -> Result<BlockId, ParseError> {
    let name = id_token(token, line)?;
    labels
        .get(name)
        .copied()
        .ok_or_else(|| line.err(ParseErrorKind::UnknownId))
}

/// Splits a body line into its optional result identifier and the rest
/// of the line starting at the opcode.
fn split_form(line: &Line) -> Result<(Option<&str>, &[Token]), ParseError> {
    let tokens = &line.tokens;
    if tokens.is_empty() {
        return Err(line.err(ParseErrorKind::UnexpectedEndOfLine));
    }
    if tokens.len() >= 2 && tokens[1].is_word("=") {
        let name = id_token(&tokens[0], line)?;
        return Ok((Some(name), &tokens[2..]));
    }
    if tokens
        .first()
        .and_then(Token::as_word)
        .is_some_and(|word| word.starts_with("Op"))
    {
        return Ok((None, tokens));
    }
    Err(line.err(ParseErrorKind::UnexpectedToken))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::print::print;

    fn emit(module: &mut Module, function: ValueId, block: BlockId, inst: Inst) {
        module.emit(function, block, inst).expect("block");
    }

    fn def(module: &mut Module, ty: TypeId, name: &str) -> ValueId {
        let value = module.new_inst_value(ty);
        module.set_value_name(value, name);
        value
    }

    fn value_by_name(module: &Module, name: &str) -> ValueId {
        module
            .values()
            .iter()
            .enumerate()
            .find(|(_, value)| value.name == name)
            .map(|(index, _)| ValueId(u32::try_from(index).unwrap_or(u32::MAX)))
            .expect("named value")
    }

    /// Builds a module that uses every [`Op`] variant, every linkage
    /// shape, every constant shape, and a structured control-flow graph.
    fn rich_module() -> Module {
        let mut module = Module::new(Target::opencl());
        let set = module.add_ext_inst_set("OpenCL.std");
        let void = module.intern_named_type(Type::Void, "void");
        let boolean = module.intern_named_type(Type::Bool, "bool");
        let int = module.intern_named_type(
            Type::Int {
                bits: 32,
                signed: true,
            },
            "int",
        );
        let uint = module.intern_named_type(
            Type::Int {
                bits: 32,
                signed: false,
            },
            "uint",
        );
        let ulong = module.intern_named_type(
            Type::Int {
                bits: 64,
                signed: false,
            },
            "ulong",
        );
        let float = module.intern_named_type(Type::Float { bits: 32 }, "float");
        let double = module.intern_named_type(Type::Float { bits: 64 }, "double");
        let float4 = module.intern_named_type(Type::Vector { elem: float, len: 4 }, "float4");
        let _arr8 = module.intern_named_type(Type::Array { elem: int, len: 8 }, "arr8");
        let _pair = module.intern_named_type(
            Type::Struct {
                fields: vec![int, float],
            },
            "pair",
        );
        let ptr_cw_float = module.intern_named_type(
            Type::Pointer {
                storage: Storage::CrossWorkgroup,
                pointee: float,
            },
            "ptr_cw_float",
        );
        let ptr_uc_int = module.intern_named_type(
            Type::Pointer {
                storage: Storage::UniformConstant,
                pointee: int,
            },
            "ptr_uc_int",
        );
        let ptr_fn_int = module.intern_named_type(
            Type::Pointer {
                storage: Storage::Function,
                pointee: int,
            },
            "ptr_fn_int",
        );
        let ptr_cw_int = module.ptr_ty(Storage::CrossWorkgroup, int);
        let fn_void = module.intern_named_type(
            Type::Function {
                ret: void,
                params: vec![ptr_cw_float, ptr_cw_float],
            },
            "fn_void",
        );
        let fn_gid = module.intern_named_type(
            Type::Function {
                ret: ulong,
                params: vec![uint],
            },
            "fn_ulong_uint",
        );
        let fn_void0 = module.intern_named_type(
            Type::Function {
                ret: void,
                params: Vec::new(),
            },
            "fn_void0",
        );
        let fn_int0 = module.intern_named_type(
            Type::Function {
                ret: int,
                params: Vec::new(),
            },
            "fn_int0",
        );

        let zero = module
            .intern_const(uint, ConstValue::Int(0))
            .expect("fits");
        let one = module
            .intern_const(uint, ConstValue::Int(1))
            .expect("fits");
        let minus_one = module
            .intern_const(int, ConstValue::Int(0xFFFF_FFFF))
            .expect("fits");
        let half = module
            .intern_const(float, ConstValue::from_f32_bits(0xBFC0_0000))
            .expect("fits");
        let _infinite = module
            .intern_const(float, ConstValue::from_f32_bits(0x7F80_0000))
            .expect("fits");
        let _not_a_number = module
            .intern_const(float, ConstValue::from_f32_bits(0x7FC0_0001))
            .expect("fits");
        let _negative_zero = module
            .intern_const(float, ConstValue::from_f32_bits(0x8000_0000))
            .expect("fits");
        let _pi = module
            .intern_const(double, ConstValue::from_f64_bits(0x4009_21FB_5444_2D18))
            .expect("fits");
        let _truth = module
            .intern_const(boolean, ConstValue::Bool(true))
            .expect("fits");
        let _nothing = module
            .intern_const(void, ConstValue::Null)
            .expect("fits");
        let _garbage = module
            .intern_const(float, ConstValue::Undef)
            .expect("fits");
        let scope = module
            .intern_const(uint, ConstValue::Int(2))
            .expect("fits");
        let semantics = module
            .intern_const(uint, ConstValue::Int(0x110))
            .expect("fits");

        module
            .add_global("counter", ptr_cw_int, Linkage::External, None)
            .expect("fresh name");
        module
            .add_global("table", ptr_uc_int, Linkage::External, Some(minus_one))
            .expect("fresh name");
        module
            .add_global("hidden", ptr_cw_int, Linkage::Internal, None)
            .expect("fresh name");

        let gid = module
            .add_function("get_global_id", fn_gid, Linkage::Import)
            .expect("fresh name");
        let gid_args = module
            .function(gid)
            .expect("declaration")
            .args
            .clone();
        module.set_value_name(gid_args[0], "dim");
        let helper = module
            .add_function("helper", fn_void0, Linkage::Export)
            .expect("fresh name");
        let int_helper = module
            .add_function("int_helper", fn_int0, Linkage::Export)
            .expect("fresh name");
        let kernel = module
            .add_function("kernel", fn_void, Linkage::External)
            .expect("fresh name");
        module
            .set_entry_point(ExecutionModel::Kernel, kernel, "ker\"nel")
            .expect("function");
        let kernel_args = module
            .function(kernel)
            .expect("kernel")
            .args
            .clone();
        module.set_value_name(kernel_args[0], "a");
        module.set_value_name(kernel_args[1], "b");

        module.begin_body(helper).expect("declaration");
        let helper_entry = module
            .push_block(helper, "helper_entry")
            .expect("block");
        emit(&mut module, helper, helper_entry, Inst::none(Op::Return));

        module
            .begin_body(int_helper)
            .expect("declaration");
        let int_helper_entry = module
            .push_block(int_helper, "int_entry")
            .expect("block");
        emit(
            &mut module,
            int_helper,
            int_helper_entry,
            Inst::none(Op::ReturnValue { value: minus_one }),
        );

        module.begin_body(kernel).expect("declaration");
        let entry = module.push_block(kernel, "entry").expect("block");
        let then_block = module.push_block(kernel, "then").expect("block");
        let else_block = module.push_block(kernel, "else").expect("block");
        let merge_block = module.push_block(kernel, "merge").expect("block");
        let header = module
            .push_block(kernel, "header")
            .expect("block");
        let loop_body = module.push_block(kernel, "body").expect("block");
        let latch = module.push_block(kernel, "latch").expect("block");
        let exit = module.push_block(kernel, "exit").expect("block");
        let dead = module.push_block(kernel, "dead").expect("block");

        let gid_value = def(&mut module, ulong, "gid");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                gid_value,
                ulong,
                Op::Call {
                    callee: gid,
                    args: vec![zero],
                },
            ),
        );
        let gid32 = def(&mut module, uint, "gid32");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                gid32,
                uint,
                Op::Convert {
                    op: ConvOp::UConvert,
                    operand: gid_value,
                },
            ),
        );
        let loaded = def(&mut module, float, "loaded");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(loaded, float, Op::Load { ptr: kernel_args[0] }),
        );
        let square_root = def(&mut module, float, "sqrt");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                square_root,
                float,
                Op::ExtInst {
                    set,
                    inst: 61,
                    args: vec![loaded],
                },
            ),
        );
        let slot_ty = module.ptr_ty(Storage::Function, float);
        let slot = def(&mut module, slot_ty, "slot");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(slot, slot_ty, Op::Variable { init: None }),
        );
        let slot2 = def(&mut module, ptr_fn_int, "slot2");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                slot2,
                ptr_fn_int,
                Op::Variable {
                    init: Some(minus_one),
                },
            ),
        );
        emit(
            &mut module,
            kernel,
            entry,
            Inst::none(Op::Store {
                ptr: slot,
                value: square_root,
            }),
        );
        let sum = def(&mut module, float, "sum");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                sum,
                float,
                Op::Binary {
                    op: BinOp::FAdd,
                    lhs: loaded,
                    rhs: square_root,
                },
            ),
        );
        let negated = def(&mut module, float, "neg");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                negated,
                float,
                Op::Unary {
                    op: UnOp::FNegate,
                    operand: sum,
                },
            ),
        );
        let comparison = def(&mut module, boolean, "cmp");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                comparison,
                boolean,
                Op::Compare {
                    op: CmpOp::FOrdLessThan,
                    lhs: negated,
                    rhs: half,
                },
            ),
        );
        let picked = def(&mut module, float, "pick");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                picked,
                float,
                Op::Select {
                    cond: comparison,
                    a: negated,
                    b: half,
                },
            ),
        );
        let copied = def(&mut module, float, "copied");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(copied, float, Op::CopyObject { operand: picked }),
        );
        let lanes = def(&mut module, float4, "v4");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                lanes,
                float4,
                Op::CompositeConstruct {
                    constituents: vec![loaded, square_root, negated, picked],
                },
            ),
        );
        let lane = def(&mut module, float, "lane");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                lane,
                float,
                Op::CompositeExtract {
                    composite: lanes,
                    indices: vec![2],
                },
            ),
        );
        let element = def(&mut module, ptr_cw_float, "elem");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                element,
                ptr_cw_float,
                Op::AccessChain {
                    base: kernel_args[0],
                    indices: vec![gid32],
                },
            ),
        );
        let ptr_off = def(&mut module, ptr_cw_float, "off");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                ptr_off,
                ptr_cw_float,
                Op::PtrAccessChain {
                    base: kernel_args[0],
                    indices: vec![gid32],
                },
            ),
        );
        let void_call = def(&mut module, void, "hc");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                void_call,
                void,
                Op::Call {
                    callee: helper,
                    args: Vec::new(),
                },
            ),
        );
        let logical_not = def(&mut module, boolean, "ln");
        emit(
            &mut module,
            kernel,
            entry,
            Inst::def(
                logical_not,
                boolean,
                Op::Unary {
                    op: UnOp::LogicalNot,
                    operand: comparison,
                },
            ),
        );

        let binaries = vec![
            (BinOp::IAdd, int, minus_one, minus_one),
            (BinOp::ISub, int, minus_one, minus_one),
            (BinOp::IMul, int, minus_one, minus_one),
            (BinOp::SDiv, int, minus_one, minus_one),
            (BinOp::UDiv, uint, gid32, one),
            (BinOp::SRem, int, minus_one, minus_one),
            (BinOp::UMod, uint, gid32, one),
            (BinOp::FAdd, float, loaded, square_root),
            (BinOp::FSub, float, loaded, square_root),
            (BinOp::FMul, float, loaded, square_root),
            (BinOp::FDiv, float, loaded, square_root),
            (BinOp::FRem, float, loaded, square_root),
            (BinOp::BitwiseAnd, uint, gid32, one),
            (BinOp::BitwiseOr, uint, gid32, one),
            (BinOp::BitwiseXor, uint, gid32, one),
            (BinOp::ShiftLeftLogical, uint, gid32, one),
            (BinOp::ShiftRightArithmetic, uint, gid32, one),
            (BinOp::ShiftRightLogical, uint, gid32, one),
            (BinOp::LogicalAnd, boolean, comparison, logical_not),
            (BinOp::LogicalOr, boolean, comparison, logical_not),
        ];
        for (op, ty, lhs, rhs) in binaries {
            let value = def(&mut module, ty, op.as_str());
            emit(
                &mut module,
                kernel,
                entry,
                Inst::def(value, ty, Op::Binary { op, lhs, rhs }),
            );
        }

        let compares = vec![
            (CmpOp::IEqual, minus_one, minus_one),
            (CmpOp::INotEqual, minus_one, minus_one),
            (CmpOp::SLessThan, minus_one, minus_one),
            (CmpOp::SLessThanEqual, minus_one, minus_one),
            (CmpOp::SGreaterThan, minus_one, minus_one),
            (CmpOp::SGreaterThanEqual, minus_one, minus_one),
            (CmpOp::ULessThan, gid32, one),
            (CmpOp::ULessThanEqual, gid32, one),
            (CmpOp::UGreaterThan, gid32, one),
            (CmpOp::UGreaterThanEqual, gid32, one),
            (CmpOp::FOrdEqual, loaded, square_root),
            (CmpOp::FOrdNotEqual, loaded, square_root),
            (CmpOp::FOrdLessThan, loaded, square_root),
            (CmpOp::FOrdLessThanEqual, loaded, square_root),
            (CmpOp::FOrdGreaterThan, loaded, square_root),
            (CmpOp::FOrdGreaterThanEqual, loaded, square_root),
        ];
        for (op, lhs, rhs) in compares {
            let value = def(&mut module, boolean, op.as_str());
            emit(
                &mut module,
                kernel,
                entry,
                Inst::def(value, boolean, Op::Compare { op, lhs, rhs }),
            );
        }

        let conversions = vec![
            (ConvOp::SConvert, int, gid32),
            (ConvOp::UConvert, ulong, gid32),
            (ConvOp::FConvert, double, loaded),
            (ConvOp::ConvertFToS, int, loaded),
            (ConvOp::ConvertFToU, uint, loaded),
            (ConvOp::ConvertSToF, float, minus_one),
            (ConvOp::ConvertUToF, float, gid32),
            (ConvOp::Bitcast, uint, loaded),
        ];
        for (op, ty, operand) in conversions {
            let value = def(&mut module, ty, op.as_str());
            emit(
                &mut module,
                kernel,
                entry,
                Inst::def(value, ty, Op::Convert { op, operand }),
            );
        }

        let unaries = vec![
            (UnOp::Not, uint, gid32),
            (UnOp::SNegate, int, minus_one),
            (UnOp::FNegate, float, sum),
            (UnOp::LogicalNot, boolean, comparison),
        ];
        for (op, ty, operand) in unaries {
            let value = def(&mut module, ty, op.as_str());
            emit(
                &mut module,
                kernel,
                entry,
                Inst::def(value, ty, Op::Unary { op, operand }),
            );
        }

        emit(
            &mut module,
            kernel,
            entry,
            Inst::none(Op::SelectionMerge {
                target: merge_block,
                control: 0,
            }),
        );
        emit(
            &mut module,
            kernel,
            entry,
            Inst::none(Op::BranchConditional {
                cond: comparison,
                then: then_block,
                other: else_block,
            }),
        );

        let multiplied = def(&mut module, int, "tmul");
        emit(
            &mut module,
            kernel,
            then_block,
            Inst::def(
                multiplied,
                int,
                Op::Binary {
                    op: BinOp::IMul,
                    lhs: minus_one,
                    rhs: minus_one,
                },
            ),
        );
        emit(
            &mut module,
            kernel,
            then_block,
            Inst::none(Op::Branch { target: merge_block }),
        );

        let subtracted = def(&mut module, int, "tsub");
        emit(
            &mut module,
            kernel,
            else_block,
            Inst::def(
                subtracted,
                int,
                Op::Binary {
                    op: BinOp::ISub,
                    lhs: minus_one,
                    rhs: minus_one,
                },
            ),
        );
        emit(
            &mut module,
            kernel,
            else_block,
            Inst::none(Op::Branch { target: merge_block }),
        );

        let merged = def(&mut module, int, "phi");
        emit(
            &mut module,
            kernel,
            merge_block,
            Inst::def(
                merged,
                int,
                Op::Phi {
                    incomings: vec![(multiplied, then_block), (subtracted, else_block)],
                },
            ),
        );
        emit(
            &mut module,
            kernel,
            merge_block,
            Inst::none(Op::Branch { target: header }),
        );

        emit(
            &mut module,
            kernel,
            header,
            Inst::none(Op::LoopMerge {
                merge: exit,
                cont: latch,
                control: 0,
            }),
        );
        emit(
            &mut module,
            kernel,
            header,
            Inst::none(Op::BranchConditional {
                cond: comparison,
                then: loop_body,
                other: exit,
            }),
        );

        let incremented = def(&mut module, uint, "badd");
        emit(
            &mut module,
            kernel,
            loop_body,
            Inst::def(
                incremented,
                uint,
                Op::Binary {
                    op: BinOp::IAdd,
                    lhs: gid32,
                    rhs: one,
                },
            ),
        );
        emit(
            &mut module,
            kernel,
            loop_body,
            Inst::none(Op::Branch { target: latch }),
        );

        let incremented_again = def(&mut module, uint, "badd2");
        emit(
            &mut module,
            kernel,
            latch,
            Inst::def(
                incremented_again,
                uint,
                Op::Binary {
                    op: BinOp::IAdd,
                    lhs: incremented,
                    rhs: one,
                },
            ),
        );
        emit(
            &mut module,
            kernel,
            latch,
            Inst::none(Op::BranchConditional {
                cond: comparison,
                then: header,
                other: dead,
            }),
        );

        emit(
            &mut module,
            kernel,
            exit,
            Inst::none(Op::ControlBarrier {
                exec: scope,
                mem: scope,
                semantics,
            }),
        );
        emit(
            &mut module,
            kernel,
            exit,
            Inst::none(Op::MemoryBarrier {
                mem: scope,
                semantics,
            }),
        );
        emit(&mut module, kernel, exit, Inst::none(Op::Return));
        emit(&mut module, kernel, dead, Inst::none(Op::Unreachable));
        module
    }

    #[test]
    fn round_trips_a_module_using_every_instruction() {
        let module = rich_module();
        let text = print(&module);
        let parsed = parse(&text).expect("parses");
        assert_eq!(print(&parsed), text);

        assert_eq!(parsed.entry_points.len(), 1);
        assert_eq!(parsed.entry_points[0].name, "ker\"nel");
        assert_eq!(parsed.globals().len(), 3);

        let gid = parsed
            .find_function("get_global_id")
            .expect("declared");
        let gid_function = parsed.function(gid).expect("function");
        assert!(gid_function.is_declaration());
        assert_eq!(gid_function.linkage, Linkage::Import);

        let helper = parsed.find_function("helper").expect("helper");
        assert_eq!(
            parsed.function(helper).expect("function").linkage,
            Linkage::Export
        );
        let hidden = parsed
            .globals()
            .iter()
            .copied()
            .find(|&id| parsed.value(id).name == "hidden")
            .expect("global");
        assert_eq!(parsed.global(hidden).expect("global").linkage, Linkage::Internal);

        let kernel = parsed.find_function("kernel").expect("kernel");
        let body = parsed
            .function(kernel)
            .expect("function")
            .body
            .as_ref()
            .expect("body");
        assert_eq!(body.len(), 9);
        assert_eq!(body[0].name, "entry");
        assert_eq!(body[8].name, "dead");
        assert!(body[8].insts[0].op.is_terminator());
    }

    #[test]
    fn parses_entry_points_linkage_and_declarations() {
        let text = "\
; Codevar IR 0.1
target opencl address physical64 memory opencl

OpEntryPoint Kernel %main \"main\"

OpDecorate %get_global_id LinkageAttributes \"get_global_id\" Import

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%fn_ulong_uint = OpTypeFunction %ulong %uint
%fn_void = OpTypeFunction %void

%get_global_id = OpFunction %ulong None %fn_ulong_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    OpReturn
OpFunctionEnd
";
        let module = parse(text).expect("parses");
        assert_eq!(print(&module), text);

        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].name, "main");
        assert_eq!(module.entry_points[0].model, ExecutionModel::Kernel);
        let main = module.find_function("main").expect("main");
        assert_eq!(module.entry_points[0].func, main);
        assert!(
            module
                .function(main)
                .expect("function")
                .body
                .is_some()
        );

        let gid = module
            .find_function("get_global_id")
            .expect("declared");
        let function = module.function(gid).expect("function");
        assert!(function.is_declaration());
        assert_eq!(function.linkage, Linkage::Import);
        let dim = *function.args.first().expect("argument");
        assert_eq!(module.value(dim).name, "dim");
        assert_eq!(
            module.type_of(dim),
            module
                .find_type(&Type::Int {
                    bits: 32,
                    signed: false
                })
                .expect("uint")
        );
    }

    #[test]
    fn parses_signed_and_special_float_constants() {
        let text = "\
; Codevar IR 0.1
target opencl address physical64 memory opencl

%bool = OpTypeBool
%void = OpTypeVoid
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%double = OpTypeFloat 64

%v0 = OpConstant %int -1
%v1 = OpConstant %uint 4294967295
%v2 = OpConstant %float -1.5
%v3 = OpConstant %float inf
%v4 = OpConstant %float NaN
%v5 = OpConstant %float -0
%v6 = OpConstant %double 3.141592653589793
%v7 = OpConstantTrue %bool
%v8 = OpConstantNull %void
%v9 = OpUndef %float
";
        let module = parse(text).expect("parses");
        assert_eq!(print(&module), text);

        let constant = |name: &str| module.constant_value(value_by_name(&module, name));
        assert_eq!(constant("v0"), Some(ConstValue::Int(0xFFFF_FFFF)));
        assert_eq!(constant("v1"), Some(ConstValue::Int(0xFFFF_FFFF)));
        assert_eq!(constant("v2"), Some(ConstValue::Float32(0xBFC0_0000)));
        assert_eq!(constant("v3"), Some(ConstValue::Float32(0x7F80_0000)));
        assert_eq!(constant("v4"), Some(ConstValue::Float32(0x7FC0_0000)));
        assert_eq!(constant("v5"), Some(ConstValue::Float32(0x8000_0000)));
        assert_eq!(constant("v6"), Some(ConstValue::Float64(0x4009_21FB_5444_2D18)));
        assert_eq!(constant("v7"), Some(ConstValue::Bool(true)));
        assert_eq!(constant("v8"), Some(ConstValue::Null));
        assert_eq!(constant("v9"), Some(ConstValue::Undef));
    }

    #[test]
    fn rejects_a_missing_target_line() {
        let error = parse("%void = OpTypeVoid\n").expect_err("no target");
        assert_eq!(error.kind, ParseErrorKind::MissingTarget);
        assert_eq!(error.line, 1);

        let error = parse("; only a comment\n").expect_err("no target");
        assert_eq!(error.kind, ParseErrorKind::MissingTarget);
        assert_eq!(error.line, 1);
    }

    #[test]
    fn rejects_unknown_and_duplicate_identifiers() {
        let error = parse(
            "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn = OpTypeFunction %void
%f = OpFunction %void None %fn
    %entry = OpLabel
    %r = OpLoad %void %missing
OpFunctionEnd
",
        )
        .expect_err("unknown id");
        assert_eq!(error.kind, ParseErrorKind::UnknownId);
        assert_eq!(error.line, 7);

        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %void = OpTypeVoid\n\
             %void = OpTypeBool\n",
        )
        .expect_err("duplicate id");
        assert_eq!(error.kind, ParseErrorKind::DuplicateId);
        assert_eq!(error.line, 3);
    }

    #[test]
    fn rejects_malformed_definitions() {
        let error = parse("target opencl address physical64 memory opencl\n%i = OpTypeInt 32 x\n")
            .expect_err("bad number");
        assert_eq!(error.kind, ParseErrorKind::InvalidNumber);
        assert_eq!(error.line, 2);

        let error = parse("target opencl address physical64 memory opencl\n%i = OpTypeInt 32\n")
            .expect_err("truncated");
        assert_eq!(error.kind, ParseErrorKind::UnexpectedEndOfLine);
        assert_eq!(error.line, 2);

        let error = parse("target opencl address physical64 memory opencl\n%i = OpTypeInt 32 0 0\n")
            .expect_err("extra");
        assert_eq!(error.kind, ParseErrorKind::WrongOperandCount);
        assert_eq!(error.line, 2);
    }

    #[test]
    fn rejects_structural_instruction_errors() {
        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %float = OpTypeFloat 32\n\
             %ptr = OpTypePointer CrossWorkgroup %float\n\
             %g = OpVariable %ptr Function\n",
        )
        .expect_err("storage mismatch");
        assert_eq!(error.kind, ParseErrorKind::StorageMismatch);
        assert_eq!(error.line, 4);

        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %void = OpTypeVoid\n\
             %int = OpTypeInt 32 1\n\
             %f = OpFunction %void None %int\n",
        )
        .expect_err("not a function type");
        assert_eq!(error.kind, ParseErrorKind::ExpectedFunctionType);
        assert_eq!(error.line, 4);

        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %void = OpTypeVoid\n\
             %fn = OpTypeFunction %void\n\
             %f = OpFunction %void None %fn\n\
             %entry = OpLabel\n\
             OpIAdd %void %x %y\n\
             OpFunctionEnd\n",
        )
        .expect_err("result missing");
        assert_eq!(error.kind, ParseErrorKind::ResultMismatch);
        assert_eq!(error.line, 6);

        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %void = OpTypeVoid\n\
             %fn = OpTypeFunction %void\n\
             %f = OpFunction %void None %fn\n\
             OpReturn\n\
             OpFunctionEnd\n",
        )
        .expect_err("no label");
        assert_eq!(error.kind, ParseErrorKind::MissingLabel);
        assert_eq!(error.line, 5);

        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %void = OpTypeVoid\n\
             %uint = OpTypeInt 32 0\n\
             %fn = OpTypeFunction %void\n\
             %f = OpFunction %void None %fn\n\
             %entry = OpLabel\n\
             %p = OpFunctionParameter %uint\n\
             OpFunctionEnd\n",
        )
        .expect_err("late parameter");
        assert_eq!(error.kind, ParseErrorKind::MisplacedParameter);
        assert_eq!(error.line, 7);
    }

    #[test]
    fn rejects_unresolvable_entry_points_and_decorations() {
        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %int = OpTypeInt 32 1\n\
             %c = OpConstant %int -1\n\
             OpEntryPoint Kernel %c \"k\"\n",
        )
        .expect_err("not a function");
        assert_eq!(error.kind, ParseErrorKind::NotAFunction);
        assert_eq!(error.line, 4);

        let error = parse(
            "target opencl address physical64 memory opencl\n\
             %int = OpTypeInt 32 1\n\
             %c = OpConstant %int -1\n\
             OpDecorate %c LinkageAttributes \"c\" Import\n",
        )
        .expect_err("not decoratable");
        assert_eq!(error.kind, ParseErrorKind::NotDecoratable);
        assert_eq!(error.line, 4);
    }

    #[test]
    fn rejects_unterminated_strings() {
        let error = parse(
            "target opencl address physical64 memory opencl\n\
             OpEntryPoint Kernel %f \"unterminated\n",
        )
        .expect_err("unterminated string");
        assert_eq!(error.kind, ParseErrorKind::InvalidString);
        assert_eq!(error.line, 2);
    }
}
