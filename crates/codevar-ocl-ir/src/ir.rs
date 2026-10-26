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

//! Data model of the Codevar intermediate representation.
//!
//! The model follows LLVM IR's shape rather than SPIR-V's flat id soup,
//! because the planned optimization pipeline (DCE, CFG folding, `mem2reg`,
//! GVN, inlining, LICM) is defined over LLVM's structure:
//!
//! * **Three handle spaces.** [`TypeId`] indexes the module's type arena,
//!   [`ValueId`] its value arena, and [`BlockId`] the blocks of one
//!   function.  Mixing them is a compile error, and lookups are `O(1)`.
//! * **Interning.** [`Module::intern_type`] and [`Module::intern_const`]
//!   uniquify structurally equal types and constants the way LLVM's
//!   context does, so structural comparison and folding are cheap.
//! * **Every value knows its type.** [`Value::ty`] gives the type of any
//!   constant, argument, instruction result, global, or function without
//!   consulting its definition — LLVM's `Value::getType()`.
//! * **Globals and functions are values.** [`ValueKind::Global`] and
//!   [`ValueKind::Function`] carry their payloads inline, exactly like
//!   LLVM's `GlobalVariable` and `Function` subclasses, so `Call` operands
//!   and initializers are ordinary [`ValueId`]s.
//! * **Uniform instructions.** [`Inst`] pairs an optional result with an
//!   [`Op`]; operands are typed fields instead of magic indices, and
//!   block-local control flow uses [`BlockId`] so a branch can never name
//!   a constant.
//! * **Alloca-based generation.** AST lowering emits a stack slot
//!   (`OpVariable`) for every local and reads and writes it with
//!   `load`/`store` — what Clang's code generator produces before
//!   `mem2reg`.  [`Op::Phi`] exists in the model so `mem2reg` can
//!   rewrite slots into SSA form later.
//!
//! Instruction mnemonics and operand order match SPIR-V assembly
//! (`OpIAdd`, `OpLoad`, `OpBranchConditional`, …), so the textual form of
//! the IR reads as disassembled SPIR-V and the binary builder can map the
//! two mechanically.  The SPIR-V bound and numeric identifiers are decided
//! at emission time by the builder, not stored in the IR.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_ir::ir::{Linkage, Module, Storage, Target};
//!
//! let mut module = Module::new(Target::opencl());
//! let float = module.float_ty(32);
//! let ptr = module.ptr_ty(Storage::CrossWorkgroup, float);
//! let void = module.void_ty();
//! let sig = module.fn_ty(void, vec![ptr, ptr]);
//! let kernel = module
//!     .add_function("vector_add", sig, Linkage::External)
//!     .expect("fresh name");
//! let _ = kernel;
//! ```

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Index into a module's type arena.
///
/// Types are interned: structurally equal types share one index, so
/// `TypeId` can be compared and ordered directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub u32);

impl TypeId {
    /// Arena index of this handle.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Index into a module's value arena.
///
/// Values are constants, function arguments, instruction results, global
/// variables, and functions.  Like LLVM, every value carries its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Arena index of this handle.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Index of a basic block inside one function.
///
/// Block handles are function-local, so a terminator operand can never
/// accidentally reference another function's label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Index of this handle within its function.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Index into a module's extended-instruction-import arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExtSetId(pub u32);

impl ExtSetId {
    /// Arena index of this handle.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ExtSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{}", self.0)
    }
}

/// The addressing model a module selects through `OpMemoryModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressingModel {
    /// Abstract pointers; pointers never have a numeric value.
    Logical,
    /// 32-bit physical addresses.
    Physical32,
    /// 64-bit physical addresses, required by the OpenCL environment.
    Physical64,
}

/// The memory model a module selects through `OpMemoryModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryModel {
    /// The legacy `Simple` model.
    Simple,
    /// The GLSL 4.50 model, required by the Shader environment.
    GLSL450,
    /// The OpenCL memory model, required by the OpenCL environment.
    OpenCL,
}

/// The execution model of an entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionModel {
    /// OpenCL kernels (`OpEntryPoint Kernel`).
    Kernel,
    /// Compute kernel (`OpEntryPoint GLCompute`), reserved for the
    /// Vulkan backend.
    GLCompute,
}

/// Address space / storage class of a pointer or variable.
///
/// Maps 1:1 onto SPIR-V storage classes and onto the OpenCL address
/// spaces: `Function` is private memory, `Workgroup` is `local`,
/// `CrossWorkgroup` is `global`, and `UniformConstant` is `constant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Storage {
    /// Function-local storage; every stack slot lives here.
    Function,
    /// Memory shared by one work-group (OpenCL `local`).
    Workgroup,
    /// Memory visible to all work-items (OpenCL `global`).
    CrossWorkgroup,
    /// Uniform constants such as sampled images.
    UniformConstant,
    /// Pipeline inputs.
    Input,
    /// Pipeline outputs.
    Output,
    /// Device-private storage.
    Private,
}

/// Linkage of a global or function.
///
/// Maps onto LLVM linkage for the front end and onto SPIR-V
/// `OpDecorate LinkageAttributes` for the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Visible outside the module with no linkage decoration (the
    /// default; entry points are external).
    External,
    /// Visible only inside the module.
    Internal,
    /// Imported from another module; function declarations must use this
    /// because SPIR-V requires declarations to carry a Linkage
    /// Attributes Import decoration.
    Import,
    /// Explicitly exported to other modules.
    Export,
}

/// Environment a module compiles for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// Textual environment name, currently always `opencl`.
    pub name: String,
    /// Addressing model emitted through `OpMemoryModel`.
    pub addressing: AddressingModel,
    /// Memory model emitted through `OpMemoryModel`.
    pub memory: MemoryModel,
}

impl Target {
    /// The OpenCL target: 64-bit physical addressing with the OpenCL
    /// memory model, which the SPIR-V OpenCL environment specification
    /// requires.
    #[must_use]
    pub fn opencl() -> Self {
        Self {
            name: String::from("opencl"),
            addressing: AddressingModel::Physical64,
            memory: MemoryModel::OpenCL,
        }
    }
}

/// A type in the module's interned type arena.
///
/// Structural: equality is shape equality, which is what interning and
/// the verifier rely on.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    /// The empty type.
    Void,
    /// One bit of truth (`OpTypeBool`).
    Bool,
    /// An integer of `bits` width; `signed` selects printing and the
    /// signed arithmetic family (`SDiv` versus `UDiv`).
    Int {
        /// Width in bits: 8, 16, 32, or 64.
        bits: u8,
        /// Whether the type is signed.
        signed: bool,
    },
    /// A floating-point value of `bits` width: 16, 32, or 64.
    Float {
        /// Width in bits.
        bits: u16,
    },
    /// A fixed-length vector of scalars.
    Vector {
        /// Element type; must be a scalar.
        elem: TypeId,
        /// Lane count.
        len: u32,
    },
    /// A fixed-length array.
    Array {
        /// Element type.
        elem: TypeId,
        /// Element count.  Stored as a plain integer like LLVM's
        /// `[N x T]`; the SPIR-V builder materializes the length constant
        /// the binary format requires.
        len: u32,
    },
    /// A structure with named-by-position fields (`OpTypeStruct`).
    Struct {
        /// Field types.
        fields: Vec<TypeId>,
    },
    /// A pointer in `storage` to `pointee`.
    Pointer {
        /// Address space of the pointer.
        storage: Storage,
        /// Pointed-to type.
        pointee: TypeId,
    },
    /// A function signature: return type plus parameter types.
    Function {
        /// Return type.
        ret: TypeId,
        /// Parameter types.
        params: Vec<TypeId>,
    },
}

impl Type {
    /// Short description used in diagnostics.
    #[must_use]
    pub fn describe(&self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Bool => "bool",
            Self::Int { .. } => "int",
            Self::Float { .. } => "float",
            Self::Vector { .. } => "vector",
            Self::Array { .. } => "array",
            Self::Struct { .. } => "struct",
            Self::Pointer { .. } => "pointer",
            Self::Function { .. } => "function",
        }
    }
}

/// A type-arena entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDef {
    /// Display name; empty means "print as a number".
    pub name: String,
    /// The type shape.
    pub ty: Type,
}

/// The payload of a constant value.
///
/// Integers store raw zero-extended bits sized by the value's type;
/// floats store their IEEE-754 bit pattern so printing and re-parsing is
/// exact.  [`ConstValue`] values are canonicalized at construction:
/// every NaN payload collapses to the quiet NaN that textual
/// round-tripping preserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstValue {
    /// Raw integer bits, zero-extended to the type's width.
    Int(u64),
    /// IEEE-754 binary32 bits.
    Float32(u32),
    /// IEEE-754 binary64 bits.
    Float64(u64),
    /// A boolean.
    Bool(bool),
    /// The undefined value (`OpUndef`); valid at any type.
    Undef,
    /// The null value (`OpConstantNull`); valid at any type.
    Null,
}

impl ConstValue {
    /// Builds a float constant from raw bits, canonicalizing NaN payloads.
    #[must_use]
    pub fn from_f32_bits(bits: u32) -> Self {
        if bits & 0x7F80_0000 == 0x7F80_0000 && bits & 0x007F_FFFF != 0 {
            Self::Float32(0x7FC0_0000)
        } else {
            Self::Float32(bits)
        }
    }

    /// Builds a double constant from raw bits with the same NaN
    /// canonicalization as [`ConstValue::from_f32_bits`].
    #[must_use]
    pub fn from_f64_bits(bits: u64) -> Self {
        if bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && bits & 0x000F_FFFF_FFFF_FFFF != 0 {
            Self::Float64(0x7FF8_0000_0000_0000)
        } else {
            Self::Float64(bits)
        }
    }

    /// True when this payload is valid at `ty`.
    ///
    /// `Undef` and `Null` are valid at every type; the others must match
    /// the type shape, and integer payloads must fit the type's width in
    /// zero-extended form (the canonical representation).
    #[must_use]
    pub fn fits(self, ty: &Type) -> bool {
        match (self, ty) {
            (Self::Undef | Self::Null, _) => true,
            (Self::Int(raw), Type::Int { bits, .. }) => *bits >= 64 || raw < (1u64 << *bits),
            (Self::Float32(_), Type::Float { bits: 32 }) => true,
            (Self::Float64(_), Type::Float { bits: 64 }) => true,
            (Self::Bool(_), Type::Bool) => true,
            _ => false,
        }
    }
}

/// Payload of a module-scope variable value.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalVar {
    /// Linkage of the variable.
    pub linkage: Linkage,
    /// Optional initializer; must be a constant of the pointee type.
    pub init: Option<ValueId>,
}

/// One basic block: a label name plus instructions, terminator last.
///
/// The label is represented by the block itself rather than an explicit
/// `OpLabel` instruction; printers and builders materialize it from
/// [`BasicBlock::name`].
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    /// Label display name, unique within the function.
    pub name: String,
    /// Instructions in order; the last one terminates the block.
    pub insts: Vec<Inst>,
}

/// Payload of a function value: its signature and body.
///
/// `body: None` marks a declaration (an `OpFunction` with no basic
/// blocks), which SPIR-V requires to carry a Linkage Attributes Import
/// decoration — that is how work-item built-ins such as
/// `get_global_id` are referenced.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    /// The [`Type::Function`] signature; parameters and return type
    /// derive from it.
    pub sig: TypeId,
    /// Linkage; declarations use [`Linkage::Import`].
    pub linkage: Linkage,
    /// Function-control mask (`None`, `Inline`, …).
    pub control: FunctionControl,
    /// Argument values in signature order.
    pub args: Vec<ValueId>,
    /// Basic blocks when defined, `None` for a declaration.
    pub body: Option<Vec<BasicBlock>>,
}

impl FunctionDef {
    /// True when the function has no body (SPIR-V declaration).
    #[must_use]
    pub const fn is_declaration(&self) -> bool {
        self.body.is_none()
    }
}

/// What a value is.
///
/// Mirrors LLVM's `Value` subclass hierarchy: constants, arguments, and
/// instruction results are leaves; globals and functions carry their
/// payloads inline.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueKind {
    /// A uniqued constant of the value's type.
    Constant(ConstValue),
    /// A formal parameter; identified by its position in
    /// [`FunctionDef::args`].
    Argument,
    /// An instruction result; the defining instruction lives in some
    /// function's block list.
    Instruction,
    /// A module-scope variable.
    Global(GlobalVar),
    /// A function, defined or declared.
    Function(FunctionDef),
}

/// A value in the module's value arena.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    /// Display name; empty means "print as a number".
    pub name: String,
    /// Type of this value (LLVM's `Value::getType`).
    pub ty: TypeId,
    /// What the value is.
    pub kind: ValueKind,
}

/// The SPIR-V function-control mask (a bit set of `FunctionControl`
/// enumerants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionControl(pub u32);

impl FunctionControl {
    /// No control flags.
    pub const NONE: Self = Self(0);

    /// Bits of the raw mask.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// An `OpExtInstImport` set such as `OpenCL.std`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtInstSet {
    /// Set name, which doubles as its textual identifier.
    pub name: String,
}

/// An entry point declaration (`OpEntryPoint`).
#[derive(Debug, Clone, PartialEq)]
pub struct EntryPoint {
    /// Execution model of the entry point.
    pub model: ExecutionModel,
    /// The entry function.
    pub func: ValueId,
    /// Name recorded in the binary; for kernels this equals the
    /// function's name.
    pub name: String,
}

/// A decoration applied to a module-scope value.
#[derive(Debug, Clone, PartialEq)]
pub enum Decor {
    /// Linkage attributes; declarations require this with `import` set,
    /// as the SPIR-V specification mandates for functions without a
    /// body.
    LinkageAttributes {
        /// Exported or imported linkage name.
        name: String,
        /// Whether the linkage is an import.
        import: bool,
    },
    /// A built-in variable decoration (OpenCL SPIR-V Environment
    /// Specification §2.9): the target must be an `Input`-storage
    /// module-scope variable carrying a SPIR-V built-in name such as
    /// `GlobalInvocationId`.
    BuiltIn {
        /// The built-in name as it appears in SPIR-V assembly.
        name: String,
    },
}

/// A decoration attached to a target value.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoration {
    /// Value being decorated.
    pub target: ValueId,
    /// The decoration.
    pub kind: Decor,
}

/// One machine instruction with its optional result.
///
/// Invariant: `result.is_some()` if and only if `ty.is_some()`.  For
/// every opcode except [`Op::Call`] this coincides with
/// `op.has_result()`; a call's result presence follows the callee's
/// return type, so [`Inst::def`] (value-returning) and [`Inst::none`]
/// (void) both accept it.  Use these constructors rather than the
/// struct literal.
#[derive(Debug, Clone, PartialEq)]
pub struct Inst {
    /// Result value; `None` for stores, branches, returns, barriers.
    pub result: Option<ValueId>,
    /// Result type; `Some` exactly when `result` is `Some`.
    pub ty: Option<TypeId>,
    /// The operation.
    pub op: Op,
}

impl Inst {
    /// Builds an instruction that defines `result` at type `ty`.
    ///
    /// # Panics
    ///
    /// Debug builds only: `op` must produce a result, except for
    /// [`Op::Call`], whose result presence depends on the callee.
    #[must_use]
    pub fn def(result: ValueId, ty: TypeId, op: Op) -> Self {
        debug_assert!(
            op.has_result() || matches!(op, Op::Call { .. }),
            "instruction must produce a result"
        );
        Self {
            result: Some(result),
            ty: Some(ty),
            op,
        }
    }

    /// Builds an instruction without a result.
    ///
    /// # Panics
    ///
    /// Debug builds only: `op` must not produce a result, except for
    /// [`Op::Call`] to a void function, whose result presence depends
    /// on the callee.
    #[must_use]
    pub fn none(op: Op) -> Self {
        debug_assert!(
            !op.has_result() || matches!(op, Op::Call { .. }),
            "instruction must not produce a result"
        );
        Self {
            result: None,
            ty: None,
            op,
        }
    }
}

/// Integer and floating-point binary arithmetic and logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Integer addition (bit-pattern identical for either signedness).
    IAdd,
    /// Integer subtraction.
    ISub,
    /// Integer multiplication.
    IMul,
    /// Signed integer division.
    SDiv,
    /// Unsigned integer division.
    UDiv,
    /// Signed integer remainder.
    SRem,
    /// Unsigned integer remainder (`OpUMod`).
    UMod,
    /// Float addition.
    FAdd,
    /// Float subtraction.
    FSub,
    /// Float multiplication.
    FMul,
    /// Float division.
    FDiv,
    /// Float remainder.
    FRem,
    /// Bitwise AND.
    BitwiseAnd,
    /// Bitwise OR.
    BitwiseOr,
    /// Bitwise XOR.
    BitwiseXor,
    /// Left shift.
    ShiftLeftLogical,
    /// Arithmetic (sign-extending) right shift.
    ShiftRightArithmetic,
    /// Logical (zero-filling) right shift.
    ShiftRightLogical,
    /// Boolean AND, eager (see `lower` for the short-circuit note).
    LogicalAnd,
    /// Boolean OR, eager.
    LogicalOr,
}

impl BinOp {
    /// SPIR-V mnemonic without the `Op` prefix.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IAdd => "IAdd",
            Self::ISub => "ISub",
            Self::IMul => "IMul",
            Self::SDiv => "SDiv",
            Self::UDiv => "UDiv",
            Self::SRem => "SRem",
            Self::UMod => "UMod",
            Self::FAdd => "FAdd",
            Self::FSub => "FSub",
            Self::FMul => "FMul",
            Self::FDiv => "FDiv",
            Self::FRem => "FRem",
            Self::BitwiseAnd => "BitwiseAnd",
            Self::BitwiseOr => "BitwiseOr",
            Self::BitwiseXor => "BitwiseXor",
            Self::ShiftLeftLogical => "ShiftLeftLogical",
            Self::ShiftRightArithmetic => "ShiftRightArithmetic",
            Self::ShiftRightLogical => "ShiftRightLogical",
            Self::LogicalAnd => "LogicalAnd",
            Self::LogicalOr => "LogicalOr",
        }
    }
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// Bitwise NOT (integers).
    Not,
    /// Integer negation.
    SNegate,
    /// Float negation.
    FNegate,
    /// Boolean NOT.
    LogicalNot,
}

impl UnOp {
    /// SPIR-V mnemonic without the `Op` prefix.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Not => "Not",
            Self::SNegate => "SNegate",
            Self::FNegate => "FNegate",
            Self::LogicalNot => "LogicalNot",
        }
    }
}

/// Comparisons; every result has type `Bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Integer equality (any signedness).
    IEqual,
    /// Integer inequality (any signedness).
    INotEqual,
    /// Signed less-than.
    SLessThan,
    /// Signed less-or-equal.
    SLessThanEqual,
    /// Signed greater-than.
    SGreaterThan,
    /// Signed greater-or-equal.
    SGreaterThanEqual,
    /// Unsigned less-than.
    ULessThan,
    /// Unsigned less-or-equal.
    ULessThanEqual,
    /// Unsigned greater-than.
    UGreaterThan,
    /// Unsigned greater-or-equal.
    UGreaterThanEqual,
    /// Ordered float equality.
    FOrdEqual,
    /// Ordered float inequality.
    FOrdNotEqual,
    /// Ordered float less-than.
    FOrdLessThan,
    /// Ordered float less-or-equal.
    FOrdLessThanEqual,
    /// Ordered float greater-than.
    FOrdGreaterThan,
    /// Ordered float greater-or-equal.
    FOrdGreaterThanEqual,
}

impl CmpOp {
    /// SPIR-V mnemonic without the `Op` prefix.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IEqual => "IEqual",
            Self::INotEqual => "INotEqual",
            Self::SLessThan => "SLessThan",
            Self::SLessThanEqual => "SLessThanEqual",
            Self::SGreaterThan => "SGreaterThan",
            Self::SGreaterThanEqual => "SGreaterThanEqual",
            Self::ULessThan => "ULessThan",
            Self::ULessThanEqual => "ULessThanEqual",
            Self::UGreaterThan => "UGreaterThan",
            Self::UGreaterThanEqual => "UGreaterThanEqual",
            Self::FOrdEqual => "FOrdEqual",
            Self::FOrdNotEqual => "FOrdNotEqual",
            Self::FOrdLessThan => "FOrdLessThan",
            Self::FOrdLessThanEqual => "FOrdLessThanEqual",
            Self::FOrdGreaterThan => "FOrdGreaterThan",
            Self::FOrdGreaterThanEqual => "FOrdGreaterThanEqual",
        }
    }
}

/// Scalar conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvOp {
    /// Sign-extend or truncate an integer.
    SConvert,
    /// Zero-extend or truncate an integer.
    UConvert,
    /// Float width conversion.
    FConvert,
    /// Float to signed integer.
    ConvertFToS,
    /// Float to unsigned integer.
    ConvertFToU,
    /// Signed integer to float.
    ConvertSToF,
    /// Unsigned integer to float.
    ConvertUToF,
    /// Bit-level reinterpretation.
    Bitcast,
}

impl ConvOp {
    /// SPIR-V mnemonic without the `Op` prefix.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SConvert => "SConvert",
            Self::UConvert => "UConvert",
            Self::FConvert => "FConvert",
            Self::ConvertFToS => "ConvertFToS",
            Self::ConvertFToU => "ConvertFToU",
            Self::ConvertSToF => "ConvertSToF",
            Self::ConvertUToF => "ConvertUToF",
            Self::Bitcast => "Bitcast",
        }
    }
}

/// A machine instruction.
///
/// Variant shapes follow SPIR-V assembly operand order so printing,
/// parsing, and binary emission are mechanical.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// A function-local stack slot.  The storage class follows the
    /// result pointer type, so it is not repeated here.
    Variable {
        /// Optional initializer.
        init: Option<ValueId>,
    },
    /// Load through a pointer.
    Load {
        /// Pointer to read.
        ptr: ValueId,
    },
    /// Store through a pointer.
    Store {
        /// Pointer to write.
        ptr: ValueId,
        /// Value to write.
        value: ValueId,
    },
    /// Compute a pointer to a nested element (`base[i][j]…`).
    AccessChain {
        /// Base pointer.
        base: ValueId,
        /// Indices into the pointee.
        indices: Vec<ValueId>,
    },
    /// Offset a pointer to a non-aggregate target (`p[i]`).
    ///
    /// The first index is an element offset from `base`; any further
    /// indices walk into the pointee when it is itself an aggregate.
    PtrAccessChain {
        /// Base pointer.
        base: ValueId,
        /// Element offset, then indices into the pointee.
        indices: Vec<ValueId>,
    },
    /// Bitwise copy.
    CopyObject {
        /// Source value.
        operand: ValueId,
    },
    /// A unary operation.
    Unary {
        /// Which operation.
        op: UnOp,
        /// Operand.
        operand: ValueId,
    },
    /// A binary operation.
    Binary {
        /// Which operation.
        op: BinOp,
        /// Left operand.
        lhs: ValueId,
        /// Right operand.
        rhs: ValueId,
    },
    /// A comparison; the result type is `Bool`.
    Compare {
        /// Which comparison.
        op: CmpOp,
        /// Left operand.
        lhs: ValueId,
        /// Right operand.
        rhs: ValueId,
    },
    /// Ternary select.
    Select {
        /// Condition.
        cond: ValueId,
        /// Value selected when `cond` is true.
        a: ValueId,
        /// Value selected when `cond` is false.
        b: ValueId,
    },
    /// A scalar conversion.
    Convert {
        /// Which conversion.
        op: ConvOp,
        /// Source value.
        operand: ValueId,
    },
    /// A direct call to a function in this module.
    Call {
        /// Callee; must be a function value.
        callee: ValueId,
        /// Actual arguments.
        args: Vec<ValueId>,
    },
    /// A call into an extended-instruction set such as `OpenCL.std`.
    ExtInst {
        /// Imported set.
        set: ExtSetId,
        /// Instruction number within the set.
        inst: u32,
        /// Arguments.
        args: Vec<ValueId>,
    },
    /// Extract scalars from a vector using literal indices.
    CompositeExtract {
        /// Source vector.
        composite: ValueId,
        /// Literal component indices.
        indices: Vec<u32>,
    },
    /// Build a vector or array from values.
    CompositeConstruct {
        /// Constituent values.
        constituents: Vec<ValueId>,
    },
    /// SSA merge of values flowing from predecessor blocks; produced by
    /// `mem2reg`, never by initial AST lowering.
    Phi {
        /// (value, predecessor block) pairs, at least two.
        incomings: Vec<(ValueId, BlockId)>,
    },
    /// Synchronise work-items.
    ControlBarrier {
        /// Execution scope constant.
        exec: ValueId,
        /// Memory scope constant.
        mem: ValueId,
        /// Memory-semantics constant.
        semantics: ValueId,
    },
    /// Order memory operations across work-items.
    MemoryBarrier {
        /// Memory scope constant.
        mem: ValueId,
        /// Memory-semantics constant.
        semantics: ValueId,
    },
    /// Marks the merge block of the branch that follows.
    SelectionMerge {
        /// Merge-block label.
        target: BlockId,
        /// Raw `SelectionControl` mask; normally 0.
        control: u32,
    },
    /// Marks the merge and continue blocks of the branch that follows.
    LoopMerge {
        /// Merge-block label.
        merge: BlockId,
        /// Continue-block label.
        cont: BlockId,
        /// Raw `LoopControl` mask; normally 0.
        control: u32,
    },
    /// Unconditional branch.
    Branch {
        /// Target label.
        target: BlockId,
    },
    /// Conditional branch.
    BranchConditional {
        /// Boolean condition.
        cond: ValueId,
        /// Label when `cond` is true.
        then: BlockId,
        /// Label when `cond` is false.
        other: BlockId,
    },
    /// Return from a void function.
    Return,
    /// Return a value.
    ReturnValue {
        /// Returned value.
        value: ValueId,
    },
    /// Control flow never reaches here.
    Unreachable,
}

impl Op {
    /// True when the instruction produces a result value.
    ///
    /// [`Op::Call`] is the exception: its answer depends on the callee's
    /// return type, so it reports `true` and the [`Inst`] constructors
    /// accept either shape.
    #[must_use]
    pub const fn has_result(&self) -> bool {
        !matches!(
            self,
            Self::Store { .. }
                | Self::ControlBarrier { .. }
                | Self::MemoryBarrier { .. }
                | Self::SelectionMerge { .. }
                | Self::LoopMerge { .. }
                | Self::Branch { .. }
                | Self::BranchConditional { .. }
                | Self::Return
                | Self::ReturnValue { .. }
                | Self::Unreachable
        )
    }

    /// True when the instruction terminates its block.
    #[must_use]
    pub const fn is_terminator(&self) -> bool {
        matches!(
            self,
            Self::Branch { .. }
                | Self::BranchConditional { .. }
                | Self::Return
                | Self::ReturnValue { .. }
                | Self::Unreachable
        )
    }
}

/// A failure while constructing IR through [`Module`]'s builder methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A definition reuses a name that is already taken.
    DuplicateName {
        /// The offending name.
        name: String,
    },
    /// A function was created with a signature that is not a
    /// [`Type::Function`].
    ExpectedFunctionType,
    /// A global was created at a type that is not a [`Type::Pointer`].
    NotPointerType,
    /// A constant payload does not match its type or does not fit.
    ConstantTypeMismatch,
    /// Blocks or instructions were added before [`Module::begin_body`].
    MissingBody,
    /// [`Module::begin_body`] was called on a function that already has
    /// a body.
    BodyAlreadyExists,
    /// The value is not a function.
    NotAFunction {
        /// The offending value.
        id: ValueId,
    },
    /// A block handle is out of range for its function.
    BlockOutOfRange {
        /// The offending block.
        block: BlockId,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName { name } => write!(f, "name `{name}` is already defined"),
            Self::ExpectedFunctionType => {
                write!(f, "function signature must be a function type")
            }
            Self::NotPointerType => {
                write!(f, "global variable type must be a pointer")
            }
            Self::ConstantTypeMismatch => {
                write!(f, "constant payload does not match its type")
            }
            Self::MissingBody => {
                write!(f, "function body has not been started")
            }
            Self::BodyAlreadyExists => {
                write!(f, "function already has a body")
            }
            Self::NotAFunction { id } => write!(f, "{id} is not a function"),
            Self::BlockOutOfRange { block } => write!(f, "block {block} does not exist"),
        }
    }
}

impl core::error::Error for BuildError {}

/// Arena checkpoint for streaming truncation.
///
/// Produced by [`Module::watermark`] and handed back to
/// [`Module::truncate_values`].  It records only the length of the value
/// arena at the checkpoint, which is all truncation needs: everything
/// created from that point on sits at or after the recorded index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleWatermark {
    /// Value-arena length when the watermark was taken.
    values: usize,
}

/// A whole IR module.
///
/// Construction goes through the builder methods on this type
/// ([`Module::intern_type`], [`Module::intern_const`],
/// [`Module::add_function`], [`Module::emit`], …) so that interning,
/// name uniquing, and arena bookkeeping stay consistent.  Validation of
/// the resulting structure is the verifier's job, not the builders'.
#[derive(Debug, Clone)]
pub struct Module {
    /// Environment the module compiles for.
    pub target: Target,
    /// Entry points in declaration order.
    pub entry_points: Vec<EntryPoint>,
    /// Decorations in declaration order.
    pub decorations: Vec<Decoration>,
    /// Imported extended-institution sets in declaration order.
    pub ext_inst_sets: Vec<ExtInstSet>,
    /// Type arena in intern order; the printer emits these first, which
    /// satisfies SPIR-V's declare-before-use rule.
    types: Vec<TypeDef>,
    /// Interning table: structural type to arena index.
    type_intern: BTreeMap<Type, TypeId>,
    /// Value arena in creation order.
    values: Vec<Value>,
    /// Interning table: (type, payload) of constants to arena index.
    const_intern: BTreeMap<(TypeId, ConstValue), ValueId>,
    /// Function values in definition order.
    functions: Vec<ValueId>,
    /// Global values in definition order.
    globals: Vec<ValueId>,
    /// Names already handed out by [`Module::uniquify_name`];
    /// deliberately excluded from equality because it is a construction
    /// aid, not module content.
    taken: BTreeSet<String>,
}

impl Module {
    /// Creates an empty module for `target`.
    #[must_use]
    pub fn new(target: Target) -> Self {
        Self {
            target,
            entry_points: Vec::new(),
            decorations: Vec::new(),
            ext_inst_sets: Vec::new(),
            types: Vec::new(),
            type_intern: BTreeMap::new(),
            values: Vec::new(),
            const_intern: BTreeMap::new(),
            functions: Vec::new(),
            globals: Vec::new(),
            taken: BTreeSet::new(),
        }
    }

    /// Returns a variant of `base` that no earlier call has taken,
    /// appending `.1`, `.2`, … on collision, and reserves it.
    ///
    /// Lowering uses this for source-derived names (`entry`, `x`, …)
    /// that may repeat across functions.  The parser does not need it:
    /// its symbol tables enforce uniqueness over the text it reads.
    pub fn uniquify_name(&mut self, base: &str) -> String {
        if base.is_empty() {
            return String::new();
        }
        let mut candidate = String::from(base);
        let mut suffix = 1u32;
        while self.taken.contains(&candidate) {
            candidate = alloc::format!("{base}.{suffix}");
            suffix += 1;
        }
        self.taken.insert(candidate.clone());
        candidate
    }

    /// Reserves `name` for a definition, failing when it is taken.
    ///
    /// Used by the parser at definition sites, where a repeated name in
    /// the text is ambiguous and must be rejected.
    pub fn claim_name(&mut self, name: &str) -> Result<(), BuildError> {
        if name.is_empty() {
            return Ok(());
        }
        if !self.taken.insert(name.to_string()) {
            return Err(BuildError::DuplicateName {
                name: name.to_string(),
            });
        }
        Ok(())
    }

    /// Interns `ty`, returning the existing handle when an equal type is
    /// already present (LLVM's context uniquing).
    pub fn intern_type(&mut self, ty: Type) -> TypeId {
        if let Some(&id) = self.type_intern.get(&ty) {
            return id;
        }
        let id = TypeId(u32::try_from(self.types.len()).unwrap_or(u32::MAX));
        self.types.push(TypeDef {
            name: String::new(),
            ty: ty.clone(),
        });
        self.type_intern.insert(ty, id);
        id
    }

    /// Interns `ty` and gives the result the display name `base`,
    /// uniquified; an already-interned type keeps its existing name.
    pub fn intern_named_type(&mut self, ty: Type, base: &str) -> TypeId {
        let empty = base.is_empty();
        let id = self.intern_type(ty);
        if empty {
            return id;
        }
        if self.types[id.index()].name.is_empty() {
            let name = self.uniquify_name(base);
            self.types[id.index()].name = name;
        }
        id
    }

    /// The interned handle for `ty`, if present.
    #[must_use]
    pub fn find_type(&self, ty: &Type) -> Option<TypeId> {
        self.type_intern.get(ty).copied()
    }

    /// The type-arena entry for `id`.
    #[inline]
    #[must_use]
    pub fn type_def(&self, id: TypeId) -> &TypeDef {
        &self.types[id.index()]
    }

    /// The type shape for `id`.
    ///
    /// # Panics
    ///
    /// Debug builds only: `id` must come from this module.
    #[inline]
    #[must_use]
    pub fn ty(&self, id: TypeId) -> &Type {
        debug_assert!(id.index() < self.types.len(), "unknown type handle");
        &self.types[id.index()].ty
    }

    /// The display name of type `id`, or `""`.
    #[must_use]
    pub fn type_name(&self, id: TypeId) -> &str {
        &self.types[id.index()].name
    }

    /// The type arena, in intern order.
    #[must_use]
    pub fn types(&self) -> &[TypeDef] {
        &self.types
    }

    /// The `Void` type.
    pub fn void_ty(&mut self) -> TypeId {
        self.intern_type(Type::Void)
    }

    /// The `Bool` type.
    pub fn bool_ty(&mut self) -> TypeId {
        self.intern_type(Type::Bool)
    }

    /// An integer type of `bits` width.
    pub fn int_ty(&mut self, bits: u8, signed: bool) -> TypeId {
        self.intern_type(Type::Int { bits, signed })
    }

    /// A floating-point type of `bits` width.
    pub fn float_ty(&mut self, bits: u16) -> TypeId {
        self.intern_type(Type::Float { bits })
    }

    /// A vector type of `len` lanes over `elem`.
    pub fn vector_ty(&mut self, elem: TypeId, len: u32) -> TypeId {
        self.intern_type(Type::Vector { elem, len })
    }

    /// An array type of `len` elements.
    pub fn array_ty(&mut self, elem: TypeId, len: u32) -> TypeId {
        self.intern_type(Type::Array { elem, len })
    }

    /// A structure type over `fields`.
    pub fn struct_ty(&mut self, fields: Vec<TypeId>) -> TypeId {
        self.intern_type(Type::Struct { fields })
    }

    /// A pointer type in `storage` to `pointee`.
    pub fn ptr_ty(&mut self, storage: Storage, pointee: TypeId) -> TypeId {
        self.intern_type(Type::Pointer { storage, pointee })
    }

    /// A function signature type.
    pub fn fn_ty(&mut self, ret: TypeId, params: Vec<TypeId>) -> TypeId {
        self.intern_type(Type::Function { ret, params })
    }

    /// The `(storage, pointee)` pair when `id` is a pointer type.
    #[must_use]
    pub fn pointer_parts(&self, id: TypeId) -> Option<(Storage, TypeId)> {
        match self.ty(id) {
            Type::Pointer { storage, pointee } => Some((*storage, *pointee)),
            _ => None,
        }
    }

    /// Interns a constant of type `ty`, returning the existing handle
    /// when an equal constant is already present.
    ///
    /// Fails when the payload does not match the type or does not fit
    /// the type's width.
    pub fn intern_const(&mut self, ty: TypeId, value: ConstValue) -> Result<ValueId, BuildError> {
        if !value.fits(self.ty(ty)) {
            return Err(BuildError::ConstantTypeMismatch);
        }
        let key = (ty, value);
        if let Some(&id) = self.const_intern.get(&key) {
            return Ok(id);
        }
        let id = self.push_value(String::new(), ty, ValueKind::Constant(value));
        self.const_intern.insert(key, id);
        Ok(id)
    }

    /// The constant payload of `id`, when it is a constant.
    #[must_use]
    pub fn constant_value(&self, id: ValueId) -> Option<ConstValue> {
        match &self.values[id.index()].kind {
            ValueKind::Constant(value) => Some(*value),
            _ => None,
        }
    }

    /// Declares a module-scope variable at pointer type `ptr_ty`.
    ///
    /// The storage class follows the pointer type, as `OpVariable`
    /// requires the operand to match the result type's class.
    pub fn add_global(
        &mut self,
        name: &str,
        ptr_ty: TypeId,
        linkage: Linkage,
        init: Option<ValueId>,
    ) -> Result<ValueId, BuildError> {
        if self.pointer_parts(ptr_ty).is_none() {
            return Err(BuildError::NotPointerType);
        }
        self.claim_name(name)?;
        let id = self.push_value(
            name.to_string(),
            ptr_ty,
            ValueKind::Global(GlobalVar { linkage, init }),
        );
        self.globals.push(id);
        Ok(id)
    }

    /// Declares a function at signature type `sig`, creating one
    /// unnamed argument value per parameter.
    ///
    /// The result has no body; call [`Module::begin_body`] to define it,
    /// or leave it a declaration with [`Linkage::Import`].
    pub fn add_function(&mut self, name: &str, sig: TypeId, linkage: Linkage) -> Result<ValueId, BuildError> {
        let (ret, params) = match self.ty(sig) {
            Type::Function { ret, params } => (*ret, params.clone()),
            _ => return Err(BuildError::ExpectedFunctionType),
        };
        let _ = ret;
        self.claim_name(name)?;
        let mut args = Vec::with_capacity(params.len());
        for param in params {
            let id = self.push_value(String::new(), param, ValueKind::Argument);
            args.push(id);
        }
        let id = self.push_value(
            name.to_string(),
            sig,
            ValueKind::Function(FunctionDef {
                sig,
                linkage,
                control: FunctionControl::NONE,
                args,
                body: None,
            }),
        );
        self.functions.push(id);
        Ok(id)
    }

    /// The declared or defined functions in definition order.
    #[must_use]
    pub fn functions(&self) -> &[ValueId] {
        &self.functions
    }

    /// The module-scope variables in declaration order.
    #[must_use]
    pub fn globals(&self) -> &[ValueId] {
        &self.globals
    }

    /// The value arena.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// The value `id`.
    ///
    /// # Panics
    ///
    /// Debug builds only: `id` must come from this module.
    #[inline]
    #[must_use]
    pub fn value(&self, id: ValueId) -> &Value {
        debug_assert!(id.index() < self.values.len(), "unknown value handle");
        &self.values[id.index()]
    }

    /// The type of any value — LLVM's `Value::getType`.
    #[inline]
    #[must_use]
    pub fn type_of(&self, id: ValueId) -> TypeId {
        self.value(id).ty
    }

    /// The kind of value `id`.
    #[inline]
    #[must_use]
    pub fn value_kind(&self, id: ValueId) -> &ValueKind {
        &self.value(id).kind
    }

    /// The function payload of `id`, when it is a function.
    #[must_use]
    pub fn function(&self, id: ValueId) -> Option<&FunctionDef> {
        match self.value_kind(id) {
            ValueKind::Function(function) => Some(function),
            _ => None,
        }
    }

    /// The function payload of `id`, mutably; `None` when `id` is not
    /// a function.
    #[must_use]
    pub fn function_mut(&mut self, id: ValueId) -> Option<&mut FunctionDef> {
        match &mut self.values[id.index()].kind {
            ValueKind::Function(function) => Some(function),
            _ => None,
        }
    }

    /// The value of `name`, when it names a function.
    #[must_use]
    pub fn find_function(&self, name: &str) -> Option<ValueId> {
        self.functions
            .iter()
            .copied()
            .find(|&id| self.value(id).name == name)
    }

    /// The global payload of `id`, when it is a variable.
    #[must_use]
    pub fn global(&self, id: ValueId) -> Option<&GlobalVar> {
        match self.value_kind(id) {
            ValueKind::Global(global) => Some(global),
            _ => None,
        }
    }

    /// The global payload of `id`, mutably; `None` when `id` is not a
    /// module-scope variable.
    #[must_use]
    pub fn global_mut(&mut self, id: ValueId) -> Option<&mut GlobalVar> {
        match &mut self.values[id.index()].kind {
            ValueKind::Global(global) => Some(global),
            _ => None,
        }
    }

    /// Gives the unnamed value `id` the display name `base`, uniquified.
    /// Returns the stored name.
    pub fn name_value(&mut self, id: ValueId, base: &str) -> String {
        let name = self.uniquify_name(base);
        self.values[id.index()].name.clone_from(&name);
        name
    }

    /// Sets the display name of `id` verbatim; the caller guarantees
    /// uniqueness (the parser resolves names through its own tables).
    pub fn set_value_name(&mut self, id: ValueId, name: &str) {
        self.values[id.index()].name = name.to_string();
    }

    /// Creates an unnamed instruction-result value of type `ty`.
    pub fn new_inst_value(&mut self, ty: TypeId) -> ValueId {
        self.push_value(String::new(), ty, ValueKind::Instruction)
    }

    /// Starts an empty body on the declared function `func`.
    pub fn begin_body(&mut self, func: ValueId) -> Result<(), BuildError> {
        let function = self
            .function_mut(func)
            .ok_or(BuildError::NotAFunction { id: func })?;
        if function.body.is_some() {
            return Err(BuildError::BodyAlreadyExists);
        }
        function.body = Some(Vec::new());
        Ok(())
    }

    /// Appends a block with label `name` to `func`'s body and returns
    /// its handle.
    ///
    /// `name` must already be unique (the parser takes it from its
    /// symbol table; lowering passes the result of
    /// [`Module::uniquify_name`]).
    pub fn push_block(&mut self, func: ValueId, name: &str) -> Result<BlockId, BuildError> {
        let function = self
            .function_mut(func)
            .ok_or(BuildError::NotAFunction { id: func })?;
        let body = function
            .body
            .as_mut()
            .ok_or(BuildError::MissingBody)?;
        let block = BlockId(u32::try_from(body.len()).unwrap_or(u32::MAX));
        body.push(BasicBlock {
            name: name.to_string(),
            insts: Vec::new(),
        });
        Ok(block)
    }

    /// Appends `inst` to block `block` of function `func`.
    pub fn emit(&mut self, func: ValueId, block: BlockId, inst: Inst) -> Result<(), BuildError> {
        debug_assert_eq!(
            inst.result.is_some(),
            inst.ty.is_some(),
            "result and result type must appear together"
        );
        if !matches!(inst.op, Op::Call { .. }) {
            debug_assert_eq!(
                inst.result.is_some(),
                inst.op.has_result(),
                "result presence must match the opcode"
            );
        }
        let function = self
            .function_mut(func)
            .ok_or(BuildError::NotAFunction { id: func })?;
        let body = function
            .body
            .as_mut()
            .ok_or(BuildError::MissingBody)?;
        let slot = body
            .get_mut(block.index())
            .ok_or(BuildError::BlockOutOfRange { block })?;
        slot.insts.push(inst);
        Ok(())
    }

    /// Records an entry point for `func`.
    pub fn set_entry_point(
        &mut self,
        model: ExecutionModel,
        func: ValueId,
        name: &str,
    ) -> Result<(), BuildError> {
        if self.function(func).is_none() {
            return Err(BuildError::NotAFunction { id: func });
        }
        self.entry_points.push(EntryPoint {
            model,
            func,
            name: name.to_string(),
        });
        Ok(())
    }

    /// Attaches a decoration to `target`.
    pub fn decorate(&mut self, target: ValueId, kind: Decor) {
        self.decorations.push(Decoration { target, kind });
    }

    /// Imports an extended-instruction set, reusing the handle when the
    /// set is already imported.
    pub fn add_ext_inst_set(&mut self, name: &str) -> ExtSetId {
        if let Some(index) = self
            .ext_inst_sets
            .iter()
            .position(|set| set.name == name)
        {
            return ExtSetId(u32::try_from(index).unwrap_or(u32::MAX));
        }
        self.ext_inst_sets.push(ExtInstSet {
            name: name.to_string(),
        });
        ExtSetId(u32::try_from(self.ext_inst_sets.len().saturating_sub(1)).unwrap_or(u32::MAX))
    }

    /// The extended-instruction set `id`.
    ///
    /// # Panics
    ///
    /// Debug builds only: `id` must come from this module.
    #[inline]
    #[must_use]
    pub fn ext_inst_set(&self, id: ExtSetId) -> &ExtInstSet {
        debug_assert!(
            id.index() < self.ext_inst_sets.len(),
            "unknown extended-instruction set handle"
        );
        &self.ext_inst_sets[id.index()]
    }

    /// Records the current value-arena length.
    ///
    /// The watermark brackets one streaming item: take it before
    /// lowering an item's body, then hand it back to
    /// [`Module::truncate_values`] once that body has been emitted and
    /// detached with [`Module::take_function_body`], so the memory the
    /// body allocated is reclaimed before the next item is lowered.
    #[must_use]
    pub fn watermark(&self) -> ModuleWatermark {
        ModuleWatermark {
            values: self.values.len(),
        }
    }

    /// Removes every value created after `mark`: instruction results
    /// and constants interned in that window, whose `const_intern`
    /// entries are dropped alongside them.
    ///
    /// Types, globals, functions, the taken-name set, entry points,
    /// decorations, and extended-instruction sets are untouched, as is
    /// everything at or before the mark.
    ///
    /// # Invariant
    ///
    /// No surviving value may reference a truncated value.  The caller
    /// preserves it by detaching the truncated function's body first
    /// with [`Module::take_function_body`], so every operand that
    /// survives points at a value at or before the mark.  The one
    /// exception is module-scope state created inside the window — a
    /// work-item built-in variable is created lazily by the first body
    /// that queries it — which truncation keeps by stopping just after
    /// the last such value: `globals`, `decorations`, and later bodies
    /// reference it, and arena indices cannot be renumbered.
    pub fn truncate_values(&mut self, mark: ModuleWatermark) {
        let mut keep = mark.values;
        for (index, value) in self.values.iter().enumerate().skip(keep) {
            if matches!(value.kind, ValueKind::Global(_) | ValueKind::Function(_)) {
                keep = index + 1;
            }
        }
        self.values.truncate(keep);
        self.const_intern
            .retain(|_, id| id.index() < keep);
    }

    /// Detaches the body of `function`, returning its blocks and
    /// leaving the function as a declaration (`body: None`).
    ///
    /// Returns [`None`] when `function` is not a function or is already
    /// a declaration.  Dropping the returned blocks frees their memory.
    /// Call this only after the body's SPIR-V has been emitted, and
    /// follow it with [`Module::truncate_values`] to reclaim the
    /// values the body created.
    pub fn take_function_body(&mut self, function: ValueId) -> Option<Vec<BasicBlock>> {
        self.function_mut(function)?.body.take()
    }

    fn push_value(&mut self, name: String, ty: TypeId, kind: ValueKind) -> ValueId {
        let id = ValueId(u32::try_from(self.values.len()).unwrap_or(u32::MAX));
        self.values.push(Value { name, ty, kind });
        id
    }
}

impl PartialEq for Module {
    /// Structural equality of module content: target, entry points,
    /// decorations, imports, and the four arenas.  Interning tables and
    /// the taken-name set are derived construction state and are not
    /// compared.
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
            && self.entry_points == other.entry_points
            && self.decorations == other.decorations
            && self.ext_inst_sets == other.ext_inst_sets
            && self.types == other.types
            && self.values == other.values
            && self.functions == other.functions
            && self.globals == other.globals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn type_interning_deduplicates_structural_types() {
        let mut module = Module::new(Target::opencl());
        let int_a = module.int_ty(32, true);
        let int_b = module.int_ty(32, true);
        assert_eq!(int_a, int_b);

        let unsigned = module.int_ty(32, false);
        assert_ne!(int_a, unsigned);

        let ptr_a = module.ptr_ty(Storage::CrossWorkgroup, int_a);
        let ptr_b = module.ptr_ty(Storage::CrossWorkgroup, int_b);
        assert_eq!(ptr_a, ptr_b);

        let local = module.ptr_ty(Storage::Function, int_a);
        assert_ne!(ptr_a, local);
        assert_eq!(module.types().len(), 4);
    }

    #[test]
    fn constants_intern_by_type_and_value() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let one = module
            .intern_const(int, ConstValue::Int(1))
            .expect("fits");
        let one_again = module
            .intern_const(int, ConstValue::Int(1))
            .expect("fits");
        let two = module
            .intern_const(int, ConstValue::Int(2))
            .expect("fits");
        assert_eq!(one, one_again);
        assert_ne!(one, two);
        assert_eq!(module.constant_value(one), Some(ConstValue::Int(1)));
    }

    #[test]
    fn constants_reject_payload_type_mismatches() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let float = module.float_ty(32);
        assert_eq!(
            module.intern_const(float, ConstValue::Int(1)),
            Err(BuildError::ConstantTypeMismatch)
        );
        assert_eq!(
            module.intern_const(int, ConstValue::Float32(0)),
            Err(BuildError::ConstantTypeMismatch)
        );
        let tiny = module.int_ty(8, true);
        assert_eq!(
            module.intern_const(tiny, ConstValue::Int(0x1FF)),
            Err(BuildError::ConstantTypeMismatch)
        );
    }

    #[test]
    fn float_nan_payloads_are_canonicalized() {
        assert_eq!(
            ConstValue::from_f32_bits(0x7FC0_0001),
            ConstValue::Float32(0x7FC0_0000)
        );
        assert_eq!(
            ConstValue::from_f64_bits(0x7FF8_0000_0000_0001),
            ConstValue::Float64(0x7FF8_0000_0000_0000)
        );
        assert_eq!(
            ConstValue::from_f32_bits(0x3F80_0000),
            ConstValue::Float32(0x3F80_0000)
        );
    }

    #[test]
    fn names_are_uniquified_and_claims_reject_duplicates() {
        let mut module = Module::new(Target::opencl());
        assert_eq!(module.uniquify_name("entry"), "entry");
        assert_eq!(module.uniquify_name("entry"), "entry.1");
        assert_eq!(module.uniquify_name("entry"), "entry.2");
        assert_eq!(module.uniquify_name(""), "");

        assert_eq!(module.claim_name("kernel"), Ok(()));
        assert_eq!(
            module.claim_name("kernel"),
            Err(BuildError::DuplicateName {
                name: "kernel".to_string()
            })
        );
    }

    #[test]
    fn add_function_requires_a_function_type_and_creates_arguments() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        assert_eq!(
            module.add_function("bad", int, Linkage::External),
            Err(BuildError::ExpectedFunctionType)
        );

        let ret = module.void_ty();
        let sig = module.fn_ty(ret, vec![int, int]);
        let function = module
            .add_function("kernel", sig, Linkage::External)
            .expect("fresh name");
        let payload = module.function(function).expect("is a function");
        assert_eq!(payload.args.len(), 2);
        assert!(payload.is_declaration());
        assert_eq!(module.type_of(payload.args[0]), int);
        assert_eq!(module.find_function("kernel"), Some(function));
    }

    #[test]
    fn globals_require_pointer_types() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        assert_eq!(
            module.add_global("g", int, Linkage::External, None),
            Err(BuildError::NotPointerType)
        );
        let ptr = module.ptr_ty(Storage::CrossWorkgroup, int);
        let global = module
            .add_global("g", ptr, Linkage::External, None)
            .expect("pointer type");
        assert_eq!(module.globals(), &[global]);
        assert_eq!(module.type_of(global), ptr);
    }

    #[test]
    fn body_blocks_and_instructions_flow_through_builder() {
        let mut module = Module::new(Target::opencl());
        let ret = module.void_ty();
        let sig = module.fn_ty(ret, Vec::new());
        let function = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh name");

        assert_eq!(module.push_block(function, "entry"), Err(BuildError::MissingBody));
        module.begin_body(function).expect("declaration");
        assert_eq!(module.begin_body(function), Err(BuildError::BodyAlreadyExists));

        let entry = module
            .push_block(function, "entry")
            .expect("body");
        module
            .emit(function, entry, Inst::none(Op::Return))
            .expect("block");
        assert_eq!(
            module.emit(function, BlockId(9), Inst::none(Op::Return)),
            Err(BuildError::BlockOutOfRange { block: BlockId(9) })
        );

        let body = module
            .function(function)
            .and_then(|f| f.body.as_ref());
        let body = body.expect("body");
        assert_eq!(body[0].insts.len(), 1);
        assert!(body[0].insts[0].op.is_terminator());
    }

    #[test]
    fn take_function_body_detaches_blocks_exactly_once() {
        let mut module = Module::new(Target::opencl());
        let ret = module.void_ty();
        let sig = module.fn_ty(ret, Vec::new());
        let function = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh name");
        assert_eq!(module.take_function_body(function), None);
        module.begin_body(function).expect("declaration");
        let entry = module
            .push_block(function, "entry")
            .expect("body");
        module
            .emit(function, entry, Inst::none(Op::Return))
            .expect("block");

        let blocks = module
            .take_function_body(function)
            .expect("one body");
        assert_eq!(blocks.len(), 1);
        assert!(
            module
                .function(function)
                .expect("function")
                .is_declaration()
        );
        assert_eq!(module.take_function_body(function), None);
        let int = module.int_ty(32, true);
        let constant = module
            .intern_const(int, ConstValue::Int(1))
            .expect("fits");
        assert_eq!(module.take_function_body(constant), None);
    }

    #[test]
    fn truncate_values_frees_the_window_but_keeps_declarations() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let ret = module.void_ty();
        let sig = module.fn_ty(ret, Vec::new());
        let function = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh name");
        let before = module.watermark();

        let kept = module
            .intern_const(int, ConstValue::Int(7))
            .expect("fits");
        let dropped = module
            .intern_const(int, ConstValue::Int(8))
            .expect("fits");
        let result = module.new_inst_value(int);
        assert!(module.values().len() > before.values);

        module.truncate_values(before);
        assert_eq!(module.values().len(), before.values);
        assert!(
            kept.index() >= before.values && dropped.index() >= before.values,
            "window constants were removed from the arena"
        );
        assert!(result.index() >= before.values);
        assert_eq!(module.functions(), &[function]);
        assert!(
            module
                .function(function)
                .expect("function")
                .is_declaration()
        );
        assert_eq!(module.find_function("f"), Some(function));

        // Re-interning after truncation proves the stale `const_intern`
        // entry is gone: the old handle would have been handed back.
        let again = module
            .intern_const(int, ConstValue::Int(7))
            .expect("fits");
        assert_eq!(again.index(), before.values);
        assert_eq!(module.constant_value(again), Some(ConstValue::Int(7)));
    }

    #[test]
    fn truncate_values_keeps_module_scope_values_created_in_the_window() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let ptr = module.ptr_ty(Storage::Input, int);
        let mark = module.watermark();

        let global = module
            .add_global("GlobalInvocationId", ptr, Linkage::External, None)
            .expect("fresh name");
        module.decorate(
            global,
            Decor::BuiltIn {
                name: "GlobalInvocationId".to_string(),
            },
        );
        let window_result = module.new_inst_value(int);
        assert!(window_result.index() > global.index());

        module.truncate_values(mark);
        // The global is referenced by `globals` and `decorations`,
        // which truncation does not touch, so it must stay addressable.
        assert_eq!(module.globals(), &[global]);
        assert_eq!(module.type_of(global), ptr);
        assert!(window_result.index() >= module.values().len());
        assert_eq!(module.decorations.len(), 1);
        assert_eq!(module.decorations[0].target, global);
    }
}
