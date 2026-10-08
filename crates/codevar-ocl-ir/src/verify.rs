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

//! Structural and type verification of the Codevar IR.
//!
//! [`verify`] checks everything the SPIR-V backend and the staged
//! optimizers rely on but neither the builders nor the parser enforce:
//!
//! * **Module shape** — entry points name functions with bodies,
//!   declarations carry `Linkage::Import`, globals have pointer types
//!   with constant initializers of the pointee type, and decorations
//!   target values that can carry them.
//! * **Block structure** — every block ends with exactly one terminator,
//!   merge instructions immediately precede a branch, `OpPhi` starts a
//!   block, and every block is reachable from the entry block.
//! * **Types** — each instruction's result type and operand types agree
//!   (`OpIAdd` over integers, `OpLoad` through a matching pointer,
//!   comparisons yielding booleans, calls matching their signature, …).
//! * **Definite assignment** — every use of an instruction result is
//!   dominated by its definition, including `OpPhi`'s per-predecessor
//!   rule, so `mem2reg` and the optimizers can assume valid SSA dataflow.
//!
//! The verifier is total: it inspects every handle defensively rather
//! than indexing arenas, so a malformed module yields a diagnostic
//! instead of a panic.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_ir::parse::parse;
//! use codevar_ocl_ir::verify::verify;
//!
//! let module = parse(
//!     "target opencl address physical64 memory opencl\n\
//!      %void = OpTypeVoid\n\
//!      %fn = OpTypeFunction %void\n\
//!      %main = OpFunction %void None %fn\n\
//!      %entry = OpLabel\n\
//!      OpReturn\n\
//!      OpFunctionEnd\n",
//! )
//! .expect("parses");
//! verify(&module).expect("a well-formed module verifies");
//! ```

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use crate::ir::{
    BasicBlock, BinOp, BlockId, CmpOp, ConstValue, ConvOp, Decor, ExtSetId, Inst, Linkage, Module, Op,
    Storage, Type, TypeId, UnOp, ValueId, ValueKind,
};
use crate::spirv::ops;

/// Why a module failed [`verify`].
///
/// Each variant records the entities involved so callers can map the
/// failure back to IR text or to a lowering pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// A type handle does not point into the type arena.
    UnknownType {
        /// The offending handle.
        id: TypeId,
    },
    /// A value handle does not point into the value arena.
    UnknownValue {
        /// The offending handle.
        id: ValueId,
    },
    /// An extended-instruction set handle does not exist.
    UnknownExtSet {
        /// The offending handle.
        id: ExtSetId,
    },
    /// A block handle does not exist in its function.
    UnknownBlock {
        /// The offending handle.
        block: BlockId,
    },
    /// An extended instruction number is not part of the referenced set.
    UnknownExtendedInstruction {
        /// The set that was queried.
        set: ExtSetId,
        /// The unknown instruction number.
        inst: u32,
    },
    /// A name that must be present in the binary is empty.
    EmptyName {
        /// What kind of name is empty.
        what: &'static str,
        /// Position of the offending entity.
        index: usize,
    },
    /// An entity that must be a function is not one.
    NotAFunction {
        /// The offending value.
        func: ValueId,
    },
    /// An entity that must be a module-scope variable is not one.
    NotAGlobal {
        /// The offending value.
        id: ValueId,
    },
    /// A function's signature handle is not a function type.
    SignatureNotFunctionType {
        /// The type found instead.
        found: TypeId,
    },
    /// A function value's type differs from its signature handle.
    SignatureMismatch {
        /// The offending function.
        func: ValueId,
    },
    /// A function has a different number of arguments than its
    /// signature declares.
    ArgumentCountMismatch {
        /// The offending function.
        func: ValueId,
        /// Arguments required by the signature.
        expected: usize,
        /// Arguments present on the function.
        found: usize,
    },
    /// An argument's type differs from the signature parameter.
    ArgumentTypeMismatch {
        /// The offending argument.
        arg: ValueId,
        /// Type required by the signature.
        expected: TypeId,
        /// Type of the argument.
        found: TypeId,
    },
    /// A function has no body but does not use `Linkage::Import`, which
    /// SPIR-V requires of every declaration.
    DeclarationNotImported {
        /// The offending function.
        func: ValueId,
    },
    /// A function is imported but also defines a body.
    ImportedWithBody {
        /// The offending function.
        func: ValueId,
    },
    /// A defined function has no basic blocks.
    EmptyBody {
        /// The offending function.
        func: ValueId,
    },
    /// An entry point does not name a function.
    EntryPointNotAFunction {
        /// Position of the entry point.
        index: usize,
    },
    /// An entry point names a function without a body.
    EntryPointWithoutBody {
        /// Position of the entry point.
        index: usize,
    },
    /// A module-scope variable does not have a pointer type.
    GlobalNotPointer {
        /// The offending variable.
        id: ValueId,
    },
    /// A module-scope variable's initializer is not a constant.
    GlobalInitNotConstant {
        /// The offending variable.
        id: ValueId,
    },
    /// A module-scope variable's initializer has the wrong type.
    GlobalInitTypeMismatch {
        /// The offending variable.
        id: ValueId,
        /// The pointee type of the variable.
        expected: TypeId,
        /// The type of the initializer.
        found: TypeId,
    },
    /// A decoration targets a value that cannot carry decorations.
    DecorTargetNotDecoratable {
        /// Position of the decoration.
        index: usize,
    },
    /// A `BuiltIn` decoration targets something other than an
    /// `Input`-storage module-scope variable (OpenCL SPIR-V
    /// Environment §2.9).
    BuiltInNotInputVariable {
        /// Position of the decoration.
        index: usize,
    },
    /// A block has an empty label.
    EmptyBlockName {
        /// The offending block.
        block: BlockId,
    },
    /// Two blocks in one function share a label.
    DuplicateBlockName {
        /// The second block with the repeated label.
        block: BlockId,
    },
    /// A block's last instruction does not terminate it.
    MissingTerminator {
        /// The offending block.
        block: BlockId,
    },
    /// A terminator appears before the end of its block.
    TerminatorNotLast {
        /// The offending block.
        block: BlockId,
    },
    /// A merge instruction is not immediately followed by a branch.
    MergeNotFollowedByBranch {
        /// The offending block.
        block: BlockId,
    },
    /// An `OpPhi` appears after a non-phi instruction.
    PhiNotAtBlockStart {
        /// The offending block.
        block: BlockId,
    },
    /// A block cannot be reached from the entry block and ends in a
    /// terminator that may kill the flow (`OpReturn`, `OpReturnValue`,
    /// `OpUnreachable`). Branch-terminated dead blocks are allowed so
    /// structurally required headers and continue targets stay legal.
    UnreachableBlock {
        /// The offending block.
        block: BlockId,
    },
    /// An instruction's result, result type, and opcode disagree about
    /// whether a result exists.
    ResultShapeMismatch {
        /// The block holding the instruction.
        block: BlockId,
    },
    /// A result id names a value that is not an instruction result.
    NotAnInstructionResult {
        /// The offending value.
        id: ValueId,
    },
    /// More than one instruction defines the same result id.
    DuplicateDefinition {
        /// The multiply defined value.
        id: ValueId,
    },
    /// A value is never defined by any instruction or function
    /// parameter.
    OrphanValue {
        /// The undefined value.
        id: ValueId,
    },
    /// An argument value belongs to a different function.
    ArgumentNotInFunction {
        /// The offending argument.
        id: ValueId,
    },
    /// A use of an instruction result is not dominated by its
    /// definition (or the definition is in another function).
    UndefinedUse {
        /// The unavailable value.
        id: ValueId,
        /// The block containing the use.
        block: BlockId,
    },
    /// An operand or result has the wrong kind of type.
    WrongKind {
        /// What was checked.
        what: &'static str,
        /// The kind of type required.
        expected: &'static str,
        /// The type found.
        found: TypeId,
    },
    /// Two types that must be identical are not.
    TypeMismatch {
        /// What was checked.
        what: &'static str,
        /// The required type.
        expected: TypeId,
        /// The type found.
        found: TypeId,
    },
    /// An operand that must be a constant is not.
    NotConstant {
        /// What was checked.
        what: &'static str,
        /// The offending value.
        id: ValueId,
    },
    /// An instruction has the wrong number of operands.
    WrongOperandCount {
        /// What was checked.
        what: &'static str,
        /// The required count.
        expected: usize,
        /// The count found.
        found: usize,
    },
    /// A literal composite index lies outside the component count.
    IndexOutOfRange {
        /// What was indexed.
        what: &'static str,
        /// The offending index.
        index: u32,
        /// The number of components available.
        len: u32,
    },
    /// An `OpPhi` has fewer than two value/label pairs.
    MalformedPhi {
        /// Number of pairs present.
        found: usize,
    },
    /// An instruction that must produce a result does not.
    MissingResult {
        /// What was checked.
        what: &'static str,
    },
    /// An instruction that must not produce a result does.
    UnexpectedResult {
        /// What was checked.
        what: &'static str,
    },
    /// `OpReturn` appears in a function that returns a value.
    ReturnFromNonVoid {
        /// The block holding the instruction.
        block: BlockId,
    },
    /// `OpReturnValue` appears in a function that returns nothing.
    ReturnValueFromVoid {
        /// The block holding the instruction.
        block: BlockId,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownType { id } => write!(f, "type {id} does not exist"),
            Self::UnknownValue { id } => write!(f, "value {id} does not exist"),
            Self::UnknownExtSet { id } => write!(f, "extended instruction set {id} does not exist"),
            Self::UnknownBlock { block } => write!(f, "block {block} does not exist"),
            Self::UnknownExtendedInstruction { set, inst } => {
                write!(f, "extended instruction set {set} has no instruction #{inst}")
            }
            Self::EmptyName { what, index } => write!(f, "{what} #{index} is empty"),
            Self::NotAFunction { func } => write!(f, "{func} is not a function"),
            Self::NotAGlobal { id } => write!(f, "{id} is not a module-scope variable"),
            Self::SignatureNotFunctionType { found } => {
                write!(f, "function signature must be a function type, found {found}")
            }
            Self::SignatureMismatch { func } => {
                write!(f, "{func}'s type does not match its signature")
            }
            Self::ArgumentCountMismatch {
                func,
                expected,
                found,
            } => write!(f, "{func}: expected {expected} arguments, found {found}"),
            Self::ArgumentTypeMismatch { arg, expected, found } => {
                write!(f, "argument {arg}: expected {expected}, found {found}")
            }
            Self::DeclarationNotImported { func } => {
                write!(f, "{func} has no body but is not imported")
            }
            Self::ImportedWithBody { func } => {
                write!(f, "{func} is imported but defines a body")
            }
            Self::EmptyBody { func } => write!(f, "{func} defines an empty body"),
            Self::EntryPointNotAFunction { index } => {
                write!(f, "entry point #{index} does not name a function")
            }
            Self::EntryPointWithoutBody { index } => {
                write!(f, "entry point #{index} names a function without a body")
            }
            Self::GlobalNotPointer { id } => {
                write!(f, "module-scope variable {id} does not have a pointer type")
            }
            Self::GlobalInitNotConstant { id } => {
                write!(f, "initializer of {id} is not a constant")
            }
            Self::GlobalInitTypeMismatch { id, expected, found } => {
                write!(f, "initializer of {id}: expected {expected}, found {found}")
            }
            Self::DecorTargetNotDecoratable { index } => {
                write!(f, "decoration #{index} targets a value that cannot be decorated")
            }
            Self::BuiltInNotInputVariable { index } => {
                write!(
                    f,
                    "decoration #{index}: BuiltIn decorations require an Input-storage variable"
                )
            }
            Self::EmptyBlockName { block } => write!(f, "{block} has no label"),
            Self::DuplicateBlockName { block } => {
                write!(f, "{block} repeats a label used earlier in the function")
            }
            Self::MissingTerminator { block } => {
                write!(f, "{block} does not end with a terminator")
            }
            Self::TerminatorNotLast { block } => {
                write!(f, "{block} has a terminator that is not its last instruction")
            }
            Self::MergeNotFollowedByBranch { block } => {
                write!(f, "{block} has a merge instruction not followed by a branch")
            }
            Self::PhiNotAtBlockStart { block } => {
                write!(f, "{block} has an OpPhi after a non-phi instruction")
            }
            Self::UnreachableBlock { block } => {
                write!(
                    f,
                    "{block} is unreachable and must end in OpUnreachable \
                     or a branch instruction"
                )
            }
            Self::ResultShapeMismatch { block } => {
                write!(f, "{block} has an instruction whose result and opcode disagree")
            }
            Self::NotAnInstructionResult { id } => {
                write!(f, "{id} is used as an instruction result but is not one")
            }
            Self::DuplicateDefinition { id } => {
                write!(f, "{id} is defined by more than one instruction")
            }
            Self::OrphanValue { id } => write!(f, "{id} is never defined"),
            Self::ArgumentNotInFunction { id } => {
                write!(f, "argument {id} belongs to a different function")
            }
            Self::UndefinedUse { id, block } => {
                write!(f, "{id} is not available in {block}")
            }
            Self::WrongKind {
                what,
                expected,
                found,
            } => write!(f, "{what}: expected {expected}, found {found}"),
            Self::TypeMismatch {
                what,
                expected,
                found,
            } => write!(f, "{what}: expected {expected}, found {found}"),
            Self::NotConstant { what, id } => write!(f, "{what} ({id}) is not a constant"),
            Self::WrongOperandCount {
                what,
                expected,
                found,
            } => write!(f, "{what}: expected {expected}, found {found}"),
            Self::IndexOutOfRange { what, index, len } => {
                write!(f, "{what}: index {index} out of range for {len} elements")
            }
            Self::MalformedPhi { found } => {
                write!(f, "OpPhi: expected at least 2 value/label pairs, found {found}")
            }
            Self::MissingResult { what } => write!(f, "{what} must produce a result"),
            Self::UnexpectedResult { what } => write!(f, "{what} must not produce a result"),
            Self::ReturnFromNonVoid { block } => {
                write!(f, "OpReturn in {block} leaves a value-returning function")
            }
            Self::ReturnValueFromVoid { block } => {
                write!(f, "OpReturnValue in {block} belongs to a void function")
            }
        }
    }
}

impl core::error::Error for VerifyError {}

/// Verifies `module`, returning the first violation found.
///
/// The check order is module shape (types, entry points, decorations,
/// globals), then per function: signatures and linkage, block
/// structure, control-flow graph, definitions, and finally the types
/// and definite assignment of every instruction.
///
/// # Errors
///
/// Returns a [`VerifyError`] describing the first problem; `Ok(())`
/// means the module satisfies every rule the SPIR-V backend assumes.
///
/// # Examples
///
/// ```
/// use codevar_ocl_ir::parse::parse;
/// use codevar_ocl_ir::verify::{verify, VerifyError};
///
/// let module = parse(
///     "target opencl address physical64 memory opencl\n\
///      %void = OpTypeVoid\n\
///      %fn = OpTypeFunction %void\n\
///      %main = OpFunction %void None %fn\n\
///      %entry = OpLabel\n\
///      OpReturn\n\
///      %dead = OpLabel\n\
///      OpReturn\n\
///      OpFunctionEnd\n",
/// )
/// .expect("parses");
///
/// // The `%dead` block is never branched to.
/// let error = verify(&module).expect_err("invalid");
/// assert!(matches!(error, VerifyError::UnreachableBlock { .. }));
/// ```
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    Checker::new(module).run()
}

/// Function-local facts used while checking instructions.
struct FnView<'a> {
    /// Index of the function in the module's function list.
    func: usize,
    /// Basic blocks of the function.
    body: &'a [BasicBlock],
    /// Argument values belonging to this function.
    args: &'a BTreeSet<ValueId>,
    /// Dominator bitsets: `dom[b][a]` when `a` dominates `b`.
    dom: &'a [Vec<bool>],
    /// Reachability per block; uses in unreachable blocks only need
    /// their definition to exist.
    reachable: &'a [bool],
    /// Return type from the signature.
    ret: TypeId,
}

/// Where a result id was defined: (function, block, instruction).
type DefSite = (usize, usize, usize);

/// Verification state over one module.
struct Checker<'a> {
    /// The module under verification.
    module: &'a Module,
    /// Definition site of every instruction result seen so far.
    def: BTreeMap<ValueId, DefSite>,
    /// Argument values claimed by some function.
    argued: BTreeSet<ValueId>,
}

impl<'a> Checker<'a> {
    fn new(module: &'a Module) -> Self {
        Self {
            module,
            def: BTreeMap::new(),
            argued: BTreeSet::new(),
        }
    }

    fn run(mut self) -> Result<(), VerifyError> {
        self.check_types()?;
        self.check_ext_inst_sets()?;
        self.check_entry_points()?;
        self.check_decorations()?;
        self.check_globals()?;
        let functions: Vec<ValueId> = self.module.functions().to_vec();
        for (index, &func) in functions.iter().enumerate() {
            self.check_function(index, func)?;
        }
        self.check_no_orphans()?;
        Ok(())
    }

    fn ty(&self, id: TypeId) -> Result<&Type, VerifyError> {
        self.module
            .types()
            .get(id.index())
            .map(|def| &def.ty)
            .ok_or(VerifyError::UnknownType { id })
    }

    fn type_of(&self, id: ValueId) -> Result<TypeId, VerifyError> {
        self.module
            .values()
            .get(id.index())
            .map(|value| value.ty)
            .ok_or(VerifyError::UnknownValue { id })
    }

    fn kind_of(&self, id: ValueId) -> Result<&ValueKind, VerifyError> {
        self.module
            .values()
            .get(id.index())
            .map(|value| &value.kind)
            .ok_or(VerifyError::UnknownValue { id })
    }

    fn check_block(&self, body: &[BasicBlock], block: BlockId) -> Result<usize, VerifyError> {
        if block.index() >= body.len() {
            Err(VerifyError::UnknownBlock { block })
        } else {
            Ok(block.index())
        }
    }

    fn elem_of(&self, id: TypeId) -> Result<Option<TypeId>, VerifyError> {
        Ok(match self.ty(id)? {
            Type::Vector { elem, .. } => Some(*elem),
            _ => None,
        })
    }

    fn scalar_of(&self, id: TypeId) -> Result<TypeId, VerifyError> {
        Ok(self.elem_of(id)?.unwrap_or(id))
    }

    fn is_bool(&self, id: TypeId) -> Result<bool, VerifyError> {
        Ok(matches!(self.ty(self.scalar_of(id)?)?, Type::Bool))
    }

    fn is_int(&self, id: TypeId) -> Result<bool, VerifyError> {
        Ok(matches!(self.ty(self.scalar_of(id)?)?, Type::Int { .. }))
    }

    fn is_float(&self, id: TypeId) -> Result<bool, VerifyError> {
        Ok(matches!(self.ty(self.scalar_of(id)?)?, Type::Float { .. }))
    }

    fn int_signedness(&self, id: TypeId) -> Result<Option<bool>, VerifyError> {
        Ok(match self.ty(self.scalar_of(id)?)? {
            Type::Int { signed, .. } => Some(*signed),
            _ => None,
        })
    }

    fn lanes(&self, id: TypeId) -> Result<Option<u32>, VerifyError> {
        Ok(match self.ty(id)? {
            Type::Vector { len, .. } => Some(*len),
            _ => None,
        })
    }

    fn bit_width(&self, id: TypeId) -> Result<Option<u32>, VerifyError> {
        Ok(match self.ty(id)? {
            Type::Int { bits, .. } => Some(u32::from(*bits)),
            Type::Float { bits } => Some(u32::from(*bits)),
            Type::Vector { elem, len } => {
                let base = match self.ty(*elem)? {
                    Type::Int { bits, .. } => Some(u32::from(*bits)),
                    Type::Float { bits } => Some(u32::from(*bits)),
                    _ => None,
                };
                base.map(|width| width * *len)
            }
            _ => None,
        })
    }

    fn require(
        &self,
        what: &'static str,
        expected: &'static str,
        found: TypeId,
        ok: bool,
    ) -> Result<(), VerifyError> {
        if ok {
            Ok(())
        } else {
            Err(VerifyError::WrongKind {
                what,
                expected,
                found,
            })
        }
    }

    fn expect_constant_int(&self, what: &'static str, id: ValueId) -> Result<(), VerifyError> {
        match self.kind_of(id)? {
            ValueKind::Constant(ConstValue::Int(_)) => Ok(()),
            _ => Err(VerifyError::NotConstant { what, id }),
        }
    }

    fn check_types(&self) -> Result<(), VerifyError> {
        for def in self.module.types() {
            match &def.ty {
                Type::Vector { elem, .. } | Type::Array { elem, .. } => {
                    self.ty(*elem)?;
                }
                Type::Struct { fields } => {
                    for &field in fields {
                        self.ty(field)?;
                    }
                }
                Type::Pointer { pointee, .. } => {
                    self.ty(*pointee)?;
                }
                Type::Function { ret, params } => {
                    self.ty(*ret)?;
                    for &param in params {
                        self.ty(param)?;
                    }
                }
                Type::Void | Type::Bool | Type::Int { .. } | Type::Float { .. } => {}
            }
        }
        Ok(())
    }

    fn check_ext_inst_sets(&self) -> Result<(), VerifyError> {
        for (index, set) in self.module.ext_inst_sets.iter().enumerate() {
            if set.name.is_empty() {
                return Err(VerifyError::EmptyName {
                    what: "extended instruction set name",
                    index,
                });
            }
        }
        Ok(())
    }

    fn check_entry_points(&self) -> Result<(), VerifyError> {
        for (index, entry) in self.module.entry_points.iter().enumerate() {
            let function = self
                .module
                .function(entry.func)
                .ok_or(VerifyError::EntryPointNotAFunction { index })?;
            if function.body.is_none() {
                return Err(VerifyError::EntryPointWithoutBody { index });
            }
            if entry.name.is_empty() {
                return Err(VerifyError::EmptyName {
                    what: "entry point name",
                    index,
                });
            }
        }
        Ok(())
    }

    fn check_decorations(&self) -> Result<(), VerifyError> {
        for (index, decor) in self.module.decorations.iter().enumerate() {
            let value = self
                .module
                .values()
                .get(decor.target.index())
                .ok_or(VerifyError::UnknownValue { id: decor.target })?;
            if !matches!(value.kind, ValueKind::Function(_) | ValueKind::Global(_)) {
                return Err(VerifyError::DecorTargetNotDecoratable { index });
            }
            // OpenCL SPIR-V Environment §2.9: every built-in variable
            // lives in the Input storage class.
            if matches!(decor.kind, Decor::BuiltIn { .. }) {
                let input_variable = matches!(value.kind, ValueKind::Global(_))
                    && matches!(self.module.pointer_parts(value.ty), Some((Storage::Input, _)));
                if !input_variable {
                    return Err(VerifyError::BuiltInNotInputVariable { index });
                }
            }
        }
        Ok(())
    }

    fn check_globals(&self) -> Result<(), VerifyError> {
        for &id in self.module.globals() {
            let global = self
                .module
                .global(id)
                .ok_or(VerifyError::NotAGlobal { id })?;
            let pointee = match self.ty(self.type_of(id)?)? {
                Type::Pointer { pointee, .. } => *pointee,
                _ => return Err(VerifyError::GlobalNotPointer { id }),
            };
            if let Some(init) = global.init {
                let value = self
                    .module
                    .values()
                    .get(init.index())
                    .ok_or(VerifyError::UnknownValue { id: init })?;
                if !matches!(value.kind, ValueKind::Constant(_)) {
                    return Err(VerifyError::GlobalInitNotConstant { id });
                }
                if value.ty != pointee {
                    return Err(VerifyError::GlobalInitTypeMismatch {
                        id,
                        expected: pointee,
                        found: value.ty,
                    });
                }
            }
        }
        Ok(())
    }

    fn check_function(&mut self, func_index: usize, func: ValueId) -> Result<(), VerifyError> {
        let module = self.module;
        let function = module
            .function(func)
            .ok_or(VerifyError::NotAFunction { func })?;
        let (ret, params) = match lookup_ty(module, function.sig)? {
            Type::Function { ret, params } => (*ret, params),
            _ => return Err(VerifyError::SignatureNotFunctionType { found: function.sig }),
        };
        if module.type_of(func) != function.sig {
            return Err(VerifyError::SignatureMismatch { func });
        }
        if function.args.len() != params.len() {
            return Err(VerifyError::ArgumentCountMismatch {
                func,
                expected: params.len(),
                found: function.args.len(),
            });
        }
        for (&arg, &expected) in function.args.iter().zip(params.iter()) {
            let found = self.type_of(arg)?;
            if found != expected {
                return Err(VerifyError::ArgumentTypeMismatch { arg, expected, found });
            }
        }
        if function.is_declaration() && function.linkage != Linkage::Import {
            return Err(VerifyError::DeclarationNotImported { func });
        }
        if !function.is_declaration() && function.linkage == Linkage::Import {
            return Err(VerifyError::ImportedWithBody { func });
        }
        for &arg in &function.args {
            self.argued.insert(arg);
        }
        let Some(body) = function.body.as_deref() else {
            return Ok(());
        };
        if body.is_empty() {
            return Err(VerifyError::EmptyBody { func });
        }

        let mut labels = BTreeSet::new();
        for (block_index, block_def) in body.iter().enumerate() {
            let block = block_id(block_index);
            if block_def.name.is_empty() {
                return Err(VerifyError::EmptyBlockName { block });
            }
            if !labels.insert(block_def.name.as_str()) {
                return Err(VerifyError::DuplicateBlockName { block });
            }
            self.check_block_structure(body, block_index)?;
        }

        let succs = self.successors(body)?;
        let reachable = self.check_reachable(body, &succs)?;
        let dom = dominators(&succs, &reachable);

        for (block_index, block) in body.iter().enumerate() {
            for (inst_index, inst) in block.insts.iter().enumerate() {
                if let Some(result) = inst.result {
                    let value = self
                        .module
                        .values()
                        .get(result.index())
                        .ok_or(VerifyError::UnknownValue { id: result })?;
                    if !matches!(value.kind, ValueKind::Instruction) {
                        return Err(VerifyError::NotAnInstructionResult { id: result });
                    }
                    let site = (func_index, block_index, inst_index);
                    if self.def.insert(result, site).is_some() {
                        return Err(VerifyError::DuplicateDefinition { id: result });
                    }
                }
            }
        }

        let args: BTreeSet<ValueId> = function.args.iter().copied().collect();
        let view = FnView {
            func: func_index,
            body,
            args: &args,
            dom: &dom,
            reachable: &reachable,
            ret,
        };
        for (block_index, block) in body.iter().enumerate() {
            for (inst_index, inst) in block.insts.iter().enumerate() {
                self.check_inst(&view, block_index, inst_index, inst)?;
            }
        }
        Ok(())
    }

    fn check_block_structure(&self, body: &[BasicBlock], index: usize) -> Result<(), VerifyError> {
        let block = block_id(index);
        let Some(block_def) = body.get(index) else {
            return Err(VerifyError::UnknownBlock { block });
        };
        if block_def.insts.is_empty() {
            return Err(VerifyError::MissingTerminator { block });
        }
        let mut after_non_phi = false;
        for (inst_index, inst) in block_def.insts.iter().enumerate() {
            // `Op::Call`'s result presence follows the callee's return
            // type, which `check_inst` validates against the signature;
            // the static shape check does not apply to it.
            if inst.result.is_some() != inst.ty.is_some()
                || (inst.result.is_some() != inst.op.has_result() && !matches!(inst.op, Op::Call { .. }))
            {
                return Err(VerifyError::ResultShapeMismatch { block });
            }
            if let Some(ty) = inst.ty {
                self.ty(ty)?;
            }
            if inst.op.is_terminator() && inst_index + 1 != block_def.insts.len() {
                return Err(VerifyError::TerminatorNotLast { block });
            }
            match &inst.op {
                Op::SelectionMerge { target, .. } => {
                    self.check_block(body, *target)?;
                    if !followed_by_branch(block_def, inst_index) {
                        return Err(VerifyError::MergeNotFollowedByBranch { block });
                    }
                }
                Op::LoopMerge { merge, cont, .. } => {
                    self.check_block(body, *merge)?;
                    self.check_block(body, *cont)?;
                    if !followed_by_branch(block_def, inst_index) {
                        return Err(VerifyError::MergeNotFollowedByBranch { block });
                    }
                }
                Op::Phi { .. } => {
                    if after_non_phi {
                        return Err(VerifyError::PhiNotAtBlockStart { block });
                    }
                }
                _ => after_non_phi = true,
            }
        }
        if !block_def
            .insts
            .last()
            .is_some_and(|inst| inst.op.is_terminator())
        {
            return Err(VerifyError::MissingTerminator { block });
        }
        Ok(())
    }

    fn successors(&self, body: &[BasicBlock]) -> Result<Vec<Vec<usize>>, VerifyError> {
        let mut succs = Vec::with_capacity(body.len());
        for block in body {
            let mut next = Vec::new();
            if let Some(last) = block.insts.last() {
                match &last.op {
                    Op::Branch { target } => {
                        next.push(self.check_block(body, *target)?);
                    }
                    Op::BranchConditional { then, other, .. } => {
                        next.push(self.check_block(body, *then)?);
                        next.push(self.check_block(body, *other)?);
                    }
                    _ => {}
                }
            }
            succs.push(next);
        }
        Ok(succs)
    }

    fn check_reachable(&self, body: &[BasicBlock], succs: &[Vec<usize>]) -> Result<Vec<bool>, VerifyError> {
        let mut seen = Vec::with_capacity(body.len());
        seen.resize(body.len(), false);
        let mut queue = Vec::new();
        if let Some(first) = seen.first_mut() {
            *first = true;
            queue.push(0usize);
        }
        while let Some(block) = queue.pop() {
            for &next in &succs[block] {
                let Some(flag) = seen.get_mut(next) else {
                    continue;
                };
                if *flag {
                    continue;
                }
                *flag = true;
                queue.push(next);
            }
        }
        for (index, block_def) in body.iter().enumerate() {
            if seen.get(index).is_some_and(|reached| *reached) {
                continue;
            }
            let allows_flow = block_def.insts.last().is_some_and(|inst| {
                matches!(
                    inst.op,
                    Op::Unreachable | Op::Branch { .. } | Op::BranchConditional { .. }
                )
            });
            if !allows_flow {
                return Err(VerifyError::UnreachableBlock {
                    block: block_id(index),
                });
            }
        }
        Ok(seen)
    }

    fn undefined_use(&self, id: ValueId, block: usize) -> VerifyError {
        VerifyError::UndefinedUse {
            id,
            block: block_id(block),
        }
    }

    fn use_value(
        &self,
        view: &FnView<'_>,
        id: ValueId,
        block: usize,
        index: usize,
        phi_pred: Option<usize>,
    ) -> Result<(), VerifyError> {
        let value = self
            .module
            .values()
            .get(id.index())
            .ok_or(VerifyError::UnknownValue { id })?;
        match value.kind {
            ValueKind::Instruction => {
                let (func, def_block, def_index) = self
                    .def
                    .get(&id)
                    .copied()
                    .ok_or_else(|| self.undefined_use(id, block))?;
                if func != view.func {
                    return Err(self.undefined_use(id, block));
                }
                let available = if view
                    .reachable
                    .get(block)
                    .is_some_and(|reached| *reached)
                {
                    match phi_pred {
                        Some(pred) => def_block == pred || dominates(view.dom, def_block, pred),
                        None if def_block == block => def_index < index,
                        None => dominates(view.dom, def_block, block),
                    }
                } else {
                    true
                };
                if available {
                    Ok(())
                } else {
                    Err(self.undefined_use(id, block))
                }
            }
            ValueKind::Argument => {
                if view.args.contains(&id) {
                    Ok(())
                } else {
                    Err(VerifyError::ArgumentNotInFunction { id })
                }
            }
            ValueKind::Constant(_) | ValueKind::Global(_) | ValueKind::Function(_) => Ok(()),
        }
    }

    fn check_inst(
        &self,
        view: &FnView<'_>,
        block: usize,
        index: usize,
        inst: &Inst,
    ) -> Result<(), VerifyError> {
        let block = block_id(block);
        let use_value =
            |id: ValueId, phi_pred: Option<usize>| self.use_value(view, id, block.index(), index, phi_pred);
        let shape_mismatch = VerifyError::ResultShapeMismatch { block };

        match &inst.op {
            Op::Variable { init } => {
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let (storage, pointee) = match self.ty(result)? {
                    Type::Pointer { storage, pointee } => (*storage, *pointee),
                    _ => {
                        return Err(VerifyError::WrongKind {
                            what: "variable result",
                            expected: "a pointer",
                            found: result,
                        });
                    }
                };
                if storage != Storage::Function {
                    return Err(VerifyError::WrongKind {
                        what: "variable storage",
                        expected: "the Function storage class",
                        found: result,
                    });
                }
                if let Some(init) = init {
                    use_value(*init, None)?;
                    if !matches!(self.kind_of(*init)?, ValueKind::Constant(_)) {
                        return Err(VerifyError::NotConstant {
                            what: "variable initializer",
                            id: *init,
                        });
                    }
                    let found = self.type_of(*init)?;
                    if found != pointee {
                        return Err(VerifyError::TypeMismatch {
                            what: "variable initializer",
                            expected: pointee,
                            found,
                        });
                    }
                }
            }
            Op::Load { ptr } => {
                use_value(*ptr, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let pointee = match self.ty(self.type_of(*ptr)?)? {
                    Type::Pointer { pointee, .. } => *pointee,
                    _ => {
                        return Err(VerifyError::WrongKind {
                            what: "load pointer",
                            expected: "a pointer",
                            found: self.type_of(*ptr)?,
                        });
                    }
                };
                if result != pointee {
                    return Err(VerifyError::TypeMismatch {
                        what: "load result",
                        expected: pointee,
                        found: result,
                    });
                }
            }
            Op::Store { ptr, value } => {
                use_value(*ptr, None)?;
                use_value(*value, None)?;
                let ptr_ty = self.type_of(*ptr)?;
                let pointee = match self.ty(ptr_ty)? {
                    Type::Pointer { pointee, .. } => *pointee,
                    _ => {
                        return Err(VerifyError::WrongKind {
                            what: "store pointer",
                            expected: "a pointer",
                            found: ptr_ty,
                        });
                    }
                };
                let found = self.type_of(*value)?;
                if found != pointee {
                    return Err(VerifyError::TypeMismatch {
                        what: "stored value",
                        expected: pointee,
                        found,
                    });
                }
            }
            Op::AccessChain { base, indices } => {
                use_value(*base, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                if indices.is_empty() {
                    return Err(VerifyError::WrongOperandCount {
                        what: "OpAccessChain",
                        expected: 1,
                        found: 0,
                    });
                }
                let base_ty = self.type_of(*base)?;
                let (base_storage, mut current) = match self.ty(base_ty)? {
                    Type::Pointer { storage, pointee } => (*storage, *pointee),
                    _ => {
                        return Err(VerifyError::WrongKind {
                            what: "access chain base",
                            expected: "a pointer",
                            found: base_ty,
                        });
                    }
                };
                for &index in indices {
                    use_value(index, None)?;
                    let index_ty = self.type_of(index)?;
                    if !self.is_int(index_ty)? {
                        return Err(VerifyError::WrongKind {
                            what: "access chain index",
                            expected: "an integer",
                            found: index_ty,
                        });
                    }
                    current = match self.ty(current)? {
                        Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
                        Type::Struct { fields } => {
                            let member = self.struct_member(index, fields.len())?;
                            fields[member]
                        }
                        _ => {
                            return Err(VerifyError::WrongKind {
                                what: "access chain target",
                                expected: "a vector, array, or structure",
                                found: current,
                            });
                        }
                    };
                }
                let expected = Type::Pointer {
                    storage: base_storage,
                    pointee: current,
                };
                if self.ty(result)? != &expected {
                    return match self.module.find_type(&expected) {
                        Some(expected_id) => Err(VerifyError::TypeMismatch {
                            what: "OpAccessChain result",
                            expected: expected_id,
                            found: result,
                        }),
                        None => Err(VerifyError::WrongKind {
                            what: "OpAccessChain result",
                            expected: "a pointer to the indexed type",
                            found: result,
                        }),
                    };
                }
            }
            Op::PtrAccessChain { base, indices } => {
                use_value(*base, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let Some((offset, rest)) = indices.split_first() else {
                    return Err(VerifyError::WrongOperandCount {
                        what: "OpPtrAccessChain",
                        expected: 1,
                        found: indices.len(),
                    });
                };
                use_value(*offset, None)?;
                let offset_ty = self.type_of(*offset)?;
                if !self.is_int(offset_ty)? {
                    return Err(VerifyError::WrongKind {
                        what: "pointer offset",
                        expected: "an integer",
                        found: offset_ty,
                    });
                }
                let base_ty = self.type_of(*base)?;
                let (base_storage, mut current) = match self.ty(base_ty)? {
                    Type::Pointer { storage, pointee } => (*storage, *pointee),
                    _ => {
                        return Err(VerifyError::WrongKind {
                            what: "pointer access chain base",
                            expected: "a pointer",
                            found: base_ty,
                        });
                    }
                };
                for &index in rest {
                    use_value(index, None)?;
                    let index_ty = self.type_of(index)?;
                    if !self.is_int(index_ty)? {
                        return Err(VerifyError::WrongKind {
                            what: "pointer access chain index",
                            expected: "an integer",
                            found: index_ty,
                        });
                    }
                    current = match self.ty(current)? {
                        Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
                        Type::Struct { fields } => {
                            let member = self.struct_member(index, fields.len())?;
                            fields[member]
                        }
                        _ => {
                            return Err(VerifyError::WrongKind {
                                what: "pointer access chain target",
                                expected: "a vector, array, or structure",
                                found: current,
                            });
                        }
                    };
                }
                let expected = Type::Pointer {
                    storage: base_storage,
                    pointee: current,
                };
                if self.ty(result)? != &expected {
                    return match self.module.find_type(&expected) {
                        Some(expected_id) => Err(VerifyError::TypeMismatch {
                            what: "OpPtrAccessChain result",
                            expected: expected_id,
                            found: result,
                        }),
                        None => Err(VerifyError::WrongKind {
                            what: "OpPtrAccessChain result",
                            expected: "a pointer to the indexed type",
                            found: result,
                        }),
                    };
                }
            }
            Op::CopyObject { operand } => {
                use_value(*operand, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let found = self.type_of(*operand)?;
                if result != found {
                    return Err(VerifyError::TypeMismatch {
                        what: "OpCopyObject result",
                        expected: found,
                        found: result,
                    });
                }
            }
            Op::Unary { op, operand } => {
                use_value(*operand, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let operand_ty = self.type_of(*operand)?;
                let (expected, ok) = match op {
                    UnOp::Not => ("an integer", self.is_int(operand_ty)?),
                    UnOp::SNegate => ("a signed integer", self.int_signedness(operand_ty)? == Some(true)),
                    UnOp::FNegate => ("a float", self.is_float(operand_ty)?),
                    UnOp::LogicalNot => ("a boolean", self.is_bool(operand_ty)?),
                };
                self.require("unary operand", expected, operand_ty, ok)?;
                if result != operand_ty {
                    return Err(VerifyError::TypeMismatch {
                        what: "unary result",
                        expected: operand_ty,
                        found: result,
                    });
                }
            }
            Op::Binary { op, lhs, rhs } => {
                use_value(*lhs, None)?;
                use_value(*rhs, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let lhs_ty = self.type_of(*lhs)?;
                let rhs_ty = self.type_of(*rhs)?;
                if lhs_ty != rhs_ty {
                    return Err(VerifyError::TypeMismatch {
                        what: "binary operand",
                        expected: lhs_ty,
                        found: rhs_ty,
                    });
                }
                let signed = self.int_signedness(lhs_ty)?;
                let (expected, ok) = match op {
                    BinOp::IAdd
                    | BinOp::ISub
                    | BinOp::IMul
                    | BinOp::BitwiseAnd
                    | BinOp::BitwiseOr
                    | BinOp::BitwiseXor
                    | BinOp::ShiftLeftLogical
                    | BinOp::ShiftRightArithmetic
                    | BinOp::ShiftRightLogical => ("an integer", signed.is_some()),
                    BinOp::SDiv | BinOp::SRem => ("a signed integer", signed == Some(true)),
                    BinOp::UDiv | BinOp::UMod => ("an unsigned integer", signed == Some(false)),
                    BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                        ("a float", self.is_float(lhs_ty)?)
                    }
                    BinOp::LogicalAnd | BinOp::LogicalOr => ("a boolean", self.is_bool(lhs_ty)?),
                };
                self.require("binary operand", expected, lhs_ty, ok)?;
                if result != lhs_ty {
                    return Err(VerifyError::TypeMismatch {
                        what: "binary result",
                        expected: lhs_ty,
                        found: result,
                    });
                }
            }
            Op::Compare { op, lhs, rhs } => {
                use_value(*lhs, None)?;
                use_value(*rhs, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let lhs_ty = self.type_of(*lhs)?;
                let rhs_ty = self.type_of(*rhs)?;
                if lhs_ty != rhs_ty {
                    return Err(VerifyError::TypeMismatch {
                        what: "comparison operand",
                        expected: lhs_ty,
                        found: rhs_ty,
                    });
                }
                let signed = self.int_signedness(lhs_ty)?;
                let (expected, ok) = match op {
                    CmpOp::IEqual | CmpOp::INotEqual => ("an integer", signed.is_some()),
                    CmpOp::SLessThan
                    | CmpOp::SLessThanEqual
                    | CmpOp::SGreaterThan
                    | CmpOp::SGreaterThanEqual => ("a signed integer", signed == Some(true)),
                    CmpOp::ULessThan
                    | CmpOp::ULessThanEqual
                    | CmpOp::UGreaterThan
                    | CmpOp::UGreaterThanEqual => ("an unsigned integer", signed == Some(false)),
                    CmpOp::FOrdEqual
                    | CmpOp::FOrdNotEqual
                    | CmpOp::FOrdLessThan
                    | CmpOp::FOrdLessThanEqual
                    | CmpOp::FOrdGreaterThan
                    | CmpOp::FOrdGreaterThanEqual => ("a float", self.is_float(lhs_ty)?),
                };
                self.require("comparison operand", expected, lhs_ty, ok)?;
                self.require("comparison result", "a boolean", result, self.is_bool(result)?)?;
                if let Some(lanes) = self.lanes(lhs_ty)? {
                    let ok = self.is_bool(result)? && self.lanes(result)? == Some(lanes);
                    self.require(
                        "comparison result",
                        "a vector of booleans matching the operand lanes",
                        result,
                        ok,
                    )?;
                }
            }
            Op::Select { cond, a, b } => {
                use_value(*cond, None)?;
                use_value(*a, None)?;
                use_value(*b, None)?;
                let a_ty = self.type_of(*a)?;
                let b_ty = self.type_of(*b)?;
                let cond_ty = self.type_of(*cond)?;
                if !self.is_bool(cond_ty)? {
                    return Err(VerifyError::WrongKind {
                        what: "select condition",
                        expected: "a boolean",
                        found: cond_ty,
                    });
                }
                if let Some(lanes) = self.lanes(cond_ty)? {
                    for &arm in [a, b] {
                        let arm_ty = self.type_of(arm)?;
                        if self.lanes(arm_ty)? != Some(lanes) {
                            return Err(VerifyError::WrongKind {
                                what: "select arm",
                                expected: "a vector matching the condition lanes",
                                found: arm_ty,
                            });
                        }
                    }
                }
                if a_ty != b_ty {
                    return Err(VerifyError::TypeMismatch {
                        what: "select arm",
                        expected: a_ty,
                        found: b_ty,
                    });
                }
                if let Some(result) = inst.ty
                    && result != a_ty
                {
                    return Err(VerifyError::TypeMismatch {
                        what: "select result",
                        expected: a_ty,
                        found: result,
                    });
                }
            }
            Op::Convert { op, operand } => {
                use_value(*operand, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                let operand_ty = self.type_of(*operand)?;
                match op {
                    ConvOp::SConvert | ConvOp::UConvert => {
                        self.require(
                            "conversion operand",
                            "an integer",
                            operand_ty,
                            self.is_int(operand_ty)?,
                        )?;
                        self.require("conversion result", "an integer", result, self.is_int(result)?)?;
                    }
                    ConvOp::FConvert => {
                        self.require(
                            "conversion operand",
                            "a float",
                            operand_ty,
                            self.is_float(operand_ty)?,
                        )?;
                        self.require("conversion result", "a float", result, self.is_float(result)?)?;
                    }
                    ConvOp::ConvertFToS | ConvOp::ConvertFToU => {
                        self.require(
                            "conversion operand",
                            "a float",
                            operand_ty,
                            self.is_float(operand_ty)?,
                        )?;
                        self.require("conversion result", "an integer", result, self.is_int(result)?)?;
                    }
                    ConvOp::ConvertSToF | ConvOp::ConvertUToF => {
                        self.require(
                            "conversion operand",
                            "an integer",
                            operand_ty,
                            self.is_int(operand_ty)?,
                        )?;
                        self.require("conversion result", "a float", result, self.is_float(result)?)?;
                    }
                    ConvOp::Bitcast => match (self.bit_width(operand_ty)?, self.bit_width(result)?) {
                        (Some(from), Some(to)) if from == to => {}
                        (Some(_), Some(_)) => {
                            return Err(VerifyError::WrongKind {
                                what: "bitcast result",
                                expected: "a type of the same bit width",
                                found: result,
                            });
                        }
                        (None, _) => {
                            return Err(VerifyError::WrongKind {
                                what: "bitcast operand",
                                expected: "a scalar or vector",
                                found: operand_ty,
                            });
                        }
                        (_, None) => {
                            return Err(VerifyError::WrongKind {
                                what: "bitcast result",
                                expected: "a scalar or vector",
                                found: result,
                            });
                        }
                    },
                }
            }
            Op::Call { callee, args } => {
                use_value(*callee, None)?;
                for &arg in args {
                    use_value(arg, None)?;
                }
                let ValueKind::Function(def) = self.kind_of(*callee)? else {
                    return Err(VerifyError::WrongKind {
                        what: "call target",
                        expected: "a function",
                        found: self.type_of(*callee)?,
                    });
                };
                let (expected_ret, params) = match self.ty(def.sig)? {
                    Type::Function { ret, params } => (*ret, params),
                    _ => return Err(VerifyError::SignatureNotFunctionType { found: def.sig }),
                };
                if args.len() != params.len() {
                    return Err(VerifyError::WrongOperandCount {
                        what: "call arguments",
                        expected: params.len(),
                        found: args.len(),
                    });
                }
                for (&arg, &param) in args.iter().zip(params.iter()) {
                    let found = self.type_of(arg)?;
                    if found != param {
                        return Err(VerifyError::TypeMismatch {
                            what: "call argument",
                            expected: param,
                            found,
                        });
                    }
                }
                let returns_void = matches!(self.ty(expected_ret)?, Type::Void);
                match (returns_void, inst.ty) {
                    (true, Some(_)) => {
                        return Err(VerifyError::UnexpectedResult {
                            what: "a call to a void function",
                        });
                    }
                    (false, None) => {
                        return Err(VerifyError::MissingResult {
                            what: "a call that returns a value",
                        });
                    }
                    (false, Some(result)) => {
                        if result != expected_ret {
                            return Err(VerifyError::TypeMismatch {
                                what: "call result",
                                expected: expected_ret,
                                found: result,
                            });
                        }
                    }
                    (true, None) => {}
                }
            }
            Op::ExtInst {
                set,
                inst: number,
                args,
            } => {
                for &arg in args {
                    use_value(arg, None)?;
                }
                if args.is_empty() {
                    return Err(VerifyError::WrongOperandCount {
                        what: "OpExtInst",
                        expected: 1,
                        found: 0,
                    });
                }
                let ext_set = self
                    .module
                    .ext_inst_sets
                    .get(set.index())
                    .ok_or(VerifyError::UnknownExtSet { id: *set })?;
                if ext_set.name == "OpenCL.std" && ops::ocl_name(*number).is_empty() {
                    return Err(VerifyError::UnknownExtendedInstruction {
                        set: *set,
                        inst: *number,
                    });
                }
            }
            Op::CompositeExtract { composite, indices } => {
                use_value(*composite, None)?;
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                if indices.is_empty() {
                    return Err(VerifyError::WrongOperandCount {
                        what: "OpCompositeExtract",
                        expected: 1,
                        found: 0,
                    });
                }
                let mut current = self.type_of(*composite)?;
                for &literal in indices {
                    current = match self.ty(current)? {
                        Type::Array { elem, len } | Type::Vector { elem, len } => {
                            if literal >= *len {
                                return Err(VerifyError::IndexOutOfRange {
                                    what: "composite component",
                                    index: literal,
                                    len: *len,
                                });
                            }
                            *elem
                        }
                        Type::Struct { fields } => {
                            if usize::try_from(literal).unwrap_or(usize::MAX) >= fields.len() {
                                return Err(VerifyError::IndexOutOfRange {
                                    what: "structure member",
                                    index: literal,
                                    len: u32::try_from(fields.len()).unwrap_or(u32::MAX),
                                });
                            }
                            fields[usize::try_from(literal).unwrap_or(usize::MAX)]
                        }
                        _ => {
                            return Err(VerifyError::WrongKind {
                                what: "extracted component",
                                expected: "a vector, array, or structure",
                                found: current,
                            });
                        }
                    };
                }
                if result != current {
                    return Err(VerifyError::TypeMismatch {
                        what: "OpCompositeExtract result",
                        expected: current,
                        found: result,
                    });
                }
            }
            Op::CompositeConstruct { constituents } => {
                for &value in constituents {
                    use_value(value, None)?;
                }
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                match self.ty(result)? {
                    Type::Vector { elem, len } | Type::Array { elem, len } => {
                        if constituents.len() != *len as usize {
                            return Err(VerifyError::WrongOperandCount {
                                what: "OpCompositeConstruct",
                                expected: *len as usize,
                                found: constituents.len(),
                            });
                        }
                        for &value in constituents {
                            let found = self.type_of(value)?;
                            if found != *elem {
                                return Err(VerifyError::TypeMismatch {
                                    what: "OpCompositeConstruct component",
                                    expected: *elem,
                                    found,
                                });
                            }
                        }
                    }
                    Type::Struct { fields } => {
                        if constituents.len() != fields.len() {
                            return Err(VerifyError::WrongOperandCount {
                                what: "OpCompositeConstruct",
                                expected: fields.len(),
                                found: constituents.len(),
                            });
                        }
                        for (position, &value) in constituents.iter().enumerate() {
                            let expected = fields[position];
                            let found = self.type_of(value)?;
                            if found != expected {
                                return Err(VerifyError::TypeMismatch {
                                    what: "OpCompositeConstruct component",
                                    expected,
                                    found,
                                });
                            }
                        }
                    }
                    _ => {
                        return Err(VerifyError::WrongKind {
                            what: "OpCompositeConstruct result",
                            expected: "a vector, array, or structure",
                            found: result,
                        });
                    }
                }
            }
            Op::Phi { incomings } => {
                let Some(result) = inst.ty else {
                    return Err(shape_mismatch);
                };
                if incomings.len() < 2 {
                    return Err(VerifyError::MalformedPhi {
                        found: incomings.len(),
                    });
                }
                for &(value, pred) in incomings {
                    self.check_block(view.body, pred)?;
                    use_value(value, Some(pred.index()))?;
                    let found = self.type_of(value)?;
                    if found != result {
                        return Err(VerifyError::TypeMismatch {
                            what: "phi value",
                            expected: result,
                            found,
                        });
                    }
                }
            }
            Op::ControlBarrier { exec, mem, semantics } => {
                use_value(*exec, None)?;
                use_value(*mem, None)?;
                use_value(*semantics, None)?;
                self.expect_constant_int("barrier execution scope", *exec)?;
                self.expect_constant_int("barrier memory scope", *mem)?;
                self.expect_constant_int("barrier memory semantics", *semantics)?;
            }
            Op::MemoryBarrier { mem, semantics } => {
                use_value(*mem, None)?;
                use_value(*semantics, None)?;
                self.expect_constant_int("barrier memory scope", *mem)?;
                self.expect_constant_int("barrier memory semantics", *semantics)?;
            }
            Op::SelectionMerge { .. } | Op::LoopMerge { .. } => {}
            Op::Branch { .. } => {}
            Op::BranchConditional { cond, .. } => {
                use_value(*cond, None)?;
                let cond_ty = self.type_of(*cond)?;
                self.require("branch condition", "a boolean", cond_ty, self.is_bool(cond_ty)?)?;
            }
            Op::Return => {
                if !matches!(self.ty(view.ret)?, Type::Void) {
                    return Err(VerifyError::ReturnFromNonVoid { block });
                }
            }
            Op::ReturnValue { value } => {
                use_value(*value, None)?;
                if matches!(self.ty(view.ret)?, Type::Void) {
                    return Err(VerifyError::ReturnValueFromVoid { block });
                }
                let found = self.type_of(*value)?;
                if found != view.ret {
                    return Err(VerifyError::TypeMismatch {
                        what: "return value",
                        expected: view.ret,
                        found,
                    });
                }
            }
            Op::Unreachable => {}
        }
        Ok(())
    }

    fn struct_member(&self, index: ValueId, fields: usize) -> Result<usize, VerifyError> {
        let raw = match self.kind_of(index)? {
            ValueKind::Constant(ConstValue::Int(raw)) => *raw,
            _ => {
                return Err(VerifyError::NotConstant {
                    what: "structure member index",
                    id: index,
                });
            }
        };
        let member = usize::try_from(raw).unwrap_or(usize::MAX);
        if member >= fields {
            return Err(VerifyError::IndexOutOfRange {
                what: "structure member",
                index: u32::try_from(raw).unwrap_or(u32::MAX),
                len: u32::try_from(fields).unwrap_or(u32::MAX),
            });
        }
        Ok(member)
    }

    fn check_no_orphans(&self) -> Result<(), VerifyError> {
        for (index, value) in self.module.values().iter().enumerate() {
            let id = ValueId(u32::try_from(index).unwrap_or(u32::MAX));
            self.ty(value.ty)?;
            match value.kind {
                ValueKind::Instruction if !self.def.contains_key(&id) => {
                    return Err(VerifyError::OrphanValue { id });
                }
                ValueKind::Argument if !self.argued.contains(&id) => {
                    return Err(VerifyError::OrphanValue { id });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn lookup_ty(module: &Module, id: TypeId) -> Result<&Type, VerifyError> {
    module
        .types()
        .get(id.index())
        .map(|def| &def.ty)
        .ok_or(VerifyError::UnknownType { id })
}

fn block_id(index: usize) -> BlockId {
    BlockId(u32::try_from(index).unwrap_or(u32::MAX))
}

fn followed_by_branch(block: &BasicBlock, merge_index: usize) -> bool {
    matches!(
        block.insts.get(merge_index + 1),
        Some(inst) if matches!(inst.op, Op::Branch { .. } | Op::BranchConditional { .. })
    )
}

fn dominates(dom: &[Vec<bool>], ancestor: usize, block: usize) -> bool {
    dom.get(block)
        .is_some_and(|set| set.get(ancestor).copied().unwrap_or(false))
}

fn dominators(succs: &[Vec<usize>], reachable: &[bool]) -> Vec<Vec<bool>> {
    let len = succs.len();
    let mut preds: Vec<Vec<usize>> = Vec::with_capacity(len);
    preds.resize_with(len, Vec::new);
    for (block, next) in succs.iter().enumerate() {
        for &target in next {
            if let Some(list) = preds.get_mut(target) {
                list.push(block);
            }
        }
    }
    let mut dom: Vec<Vec<bool>> = Vec::with_capacity(len);
    for block in 0..len {
        let mut row = Vec::with_capacity(len);
        if reachable
            .get(block)
            .is_some_and(|reached| *reached)
        {
            row.resize(len, true);
        } else {
            row.resize(len, false);
            if let Some(flag) = row.get_mut(block) {
                *flag = true;
            }
        }
        dom.push(row);
    }
    if let Some(entry) = dom.first_mut() {
        for (block, flag) in entry.iter_mut().enumerate() {
            *flag = block == 0;
        }
    }
    loop {
        let mut changed = false;
        for block in 1..len {
            if !reachable
                .get(block)
                .is_some_and(|reached| *reached)
            {
                continue;
            }
            let mut next = Vec::with_capacity(len);
            next.resize(len, true);
            if let Some(list) = preds.get(block) {
                for &pred in list {
                    if let Some(row) = dom.get(pred) {
                        for (flag, &known) in next.iter_mut().zip(row.iter()) {
                            *flag &= known;
                        }
                    }
                }
            }
            if let Some(flag) = next.get_mut(block) {
                *flag = true;
            }
            if dom.get(block) != Some(&next) {
                if let Some(slot) = dom.get_mut(block) {
                    *slot = next;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Target;
    use crate::parse::parse;

    fn parse_ok(text: &str) -> Module {
        parse(text).expect("test module parses")
    }

    fn verify_err(text: &str) -> VerifyError {
        verify(&parse_ok(text)).expect_err("test module must fail verification")
    }

    fn type_id(module: &Module, ty: Type) -> TypeId {
        module.find_type(&ty).expect("type is interned")
    }

    const PREAMBLE: &str = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%fn_void = OpTypeFunction %void
%ptr_fn_int = OpTypePointer Function %int

%v1 = OpConstant %int 1
%vf = OpConstant %float 1.5

OpEntryPoint Kernel %main \"main\"
";

    const WELL_FORMED: &str = "\
; Codevar IR 0.1
target opencl address physical64 memory opencl

%OpenCL.std = OpExtInstImport \"OpenCL.std\"

OpEntryPoint Kernel %vector_add \"vector_add\"

OpDecorate %get_global_id LinkageAttributes \"get_global_id\" Import

%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%ptr_fn_int = OpTypePointer Function %int
%fn_void = OpTypeFunction %void %ptr_cw_float %ptr_cw_float
%fn_uint_uint = OpTypeFunction %uint %uint

%v0 = OpConstant %uint 0
%v1 = OpConstant %uint 1
%v_sem = OpConstant %uint 64
%v_int_1 = OpConstant %int 1

%get_global_id = OpFunction %uint None %fn_uint_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd

%vector_add = OpFunction %void None %fn_void
    %a = OpFunctionParameter %ptr_cw_float
    %b = OpFunctionParameter %ptr_cw_float
    %entry = OpLabel
    %idx = OpFunctionCall %uint %get_global_id %v0
    %slot = OpVariable %ptr_fn_int Function
    OpStore %slot %v_int_1
    %sum = OpLoad %int %slot
    %aval = OpLoad %float %a
    %scaled = OpExtInst %float %OpenCL.std 61 %aval
    OpStore %b %scaled
    %cmp = OpUGreaterThan %bool %idx %v0
    OpSelectionMerge %merge None
    OpBranchConditional %cmp %then %else
    %then = OpLabel
    OpBranch %merge
    %else = OpLabel
    OpBranch %merge
    %merge = OpLabel
    OpControlBarrier %v1 %v1 %v_sem
    OpReturn
OpFunctionEnd
";

    #[test]
    fn accepts_a_well_formed_kernel() {
        let module = parse_ok(WELL_FORMED);
        verify(&module).expect("kernel verifies");
    }

    #[test]
    fn accepts_a_self_referential_loop_phi() {
        let module = parse_ok(
            "target opencl address physical64 memory opencl\n\
             \n\
             %bool = OpTypeBool\n\
             %void = OpTypeVoid\n\
             %int = OpTypeInt 32 1\n\
             %fn_void = OpTypeFunction %void\n\
             \n\
             %v0 = OpConstant %int 0\n\
             %v1 = OpConstant %int 1\n\
             \n\
             OpEntryPoint Kernel %main \"main\"\n\
             \n\
             %main = OpFunction %void None %fn_void\n\
             %preheader = OpLabel\n\
             OpBranch %loop\n\
             %loop = OpLabel\n\
             %i = OpPhi %int %v0 %preheader %next %loop\n\
             %next = OpIAdd %int %i %v1\n\
             %done = OpSGreaterThanEqual %bool %next %v1\n\
             OpBranchConditional %done %exit %loop\n\
             %exit = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n",
        );
        verify(&module).expect("loop verifies");
    }

    #[test]
    fn accepts_an_unreachable_merge_ending_in_op_unreachable() {
        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %vb = OpConstantTrue %bool\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpSelectionMerge %merge None\n\
             OpBranchConditional %vb %then %else\n\
             %then = OpLabel\n\
             OpReturn\n\
             %else = OpLabel\n\
             OpReturn\n\
             %merge = OpLabel\n\
             OpUnreachable\n\
             OpFunctionEnd\n"
        ));
        verify(&module).expect("unreachable structural merge verifies");
    }

    #[test]
    fn accepts_an_unreachable_block_ending_in_a_branch() {
        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpReturn\n\
             %dead = OpLabel\n\
             OpBranch %entry\n\
             OpFunctionEnd\n"
        ));
        verify(&module).expect("branch-terminated dead block verifies");
    }

    #[test]
    fn rejects_instruction_type_errors() {
        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %sum = OpIAdd %int %vf %vf\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let float = type_id(&module, Type::Float { bits: 32 });
        let error = verify(&module).expect_err("float addition");
        assert!(
            matches!(
                &error,
                VerifyError::WrongKind {
                    what: "binary operand",
                    expected: "an integer",
                    found,
                } if *found == float
            ),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %slot = OpVariable %ptr_fn_int Function\n\
             OpStore %slot %vf\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let int = type_id(
            &module,
            Type::Int {
                bits: 32,
                signed: true,
            },
        );
        let float = type_id(&module, Type::Float { bits: 32 });
        let error = verify(&module).expect_err("store mismatch");
        assert!(
            matches!(
                &error,
                VerifyError::TypeMismatch {
                    what: "stored value",
                    expected,
                    found,
                } if *expected == int && *found == float
            ),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpBranchConditional %v1 %exit %exit\n\
             %exit = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let int = type_id(
            &module,
            Type::Int {
                bits: 32,
                signed: true,
            },
        );
        let error = verify(&module).expect_err("non-boolean condition");
        assert!(
            matches!(
                &error,
                VerifyError::WrongKind {
                    what: "branch condition",
                    expected: "a boolean",
                    found,
                } if *found == int
            ),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %slot = OpVariable %ptr_fn_int Function\n\
             %loaded = OpLoad %float %slot\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let int = type_id(
            &module,
            Type::Int {
                bits: 32,
                signed: true,
            },
        );
        let float = type_id(&module, Type::Float { bits: 32 });
        let error = verify(&module).expect_err("load mismatch");
        assert!(
            matches!(
                &error,
                VerifyError::TypeMismatch {
                    what: "load result",
                    expected,
                    found,
                } if *expected == int && *found == float
            ),
            "unexpected error: {error}"
        );
    }

    const ACCESS_CHAIN: &str = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%arr4 = OpTypeArray %float 4
%ptr_fn_float = OpTypePointer Function %float
%ptr_fn_arr4 = OpTypePointer Function %arr4
%fn_void = OpTypeFunction %void %ptr_fn_arr4

%idx = OpConstant %uint 1

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
%buf = OpFunctionParameter %ptr_fn_arr4
%entry = OpLabel
";

    #[test]
    fn accepts_access_chains_with_pointer_results() {
        let module = parse_ok(&format!(
            "{ACCESS_CHAIN}\n\
             %elem = OpAccessChain %ptr_fn_float %buf %idx\n\
             %off = OpPtrAccessChain %ptr_fn_float %buf %idx %idx\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        verify(&module).expect("access chains with pointer results verify");
    }

    #[test]
    fn accepts_a_void_call_without_a_result() {
        let module = parse_ok(
            "target opencl address physical64 memory opencl\n\
             \n\
             %void = OpTypeVoid\n\
             %fn_void = OpTypeFunction %void\n\
             \n\
             OpEntryPoint Kernel %main \"main\"\n\
             \n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpFunctionCall %helper\n\
             OpReturn\n\
             OpFunctionEnd\n\
             \n\
             %helper = OpFunction %void None %fn_void\n\
             %hentry = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n",
        );
        verify(&module).expect("void call without a result verifies");
    }

    #[test]
    fn rejects_access_chains_with_non_pointer_results() {
        let module = parse_ok(&format!(
            "{ACCESS_CHAIN}\n\
             %elem = OpAccessChain %float %buf %idx\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let float = type_id(&module, Type::Float { bits: 32 });
        let ptr = type_id(
            &module,
            Type::Pointer {
                storage: Storage::Function,
                pointee: float,
            },
        );
        let error = verify(&module).expect_err("non-pointer access chain result");
        assert!(
            matches!(
                &error,
                VerifyError::TypeMismatch {
                    what: "OpAccessChain result",
                    expected,
                    found,
                } if *expected == ptr && *found == float
            ),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{ACCESS_CHAIN}\n\
             %elem = OpPtrAccessChain %float %buf %idx %idx\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("non-pointer ptr access chain result");
        assert!(
            matches!(
                &error,
                VerifyError::TypeMismatch {
                    what: "OpPtrAccessChain result",
                    expected,
                    found,
                } if *expected == ptr && *found == float
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_structural_control_flow_errors() {
        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %sum = OpIAdd %int %v1 %v1\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("missing terminator");
        assert!(
            matches!(error, VerifyError::MissingTerminator { block } if block == BlockId(0)),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpReturn\n\
             OpMemoryBarrier %v1 %v1\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("terminator not last");
        assert!(
            matches!(error, VerifyError::TerminatorNotLast { block } if block == BlockId(0)),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpSelectionMerge %merge None\n\
             OpReturn\n\
             %merge = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("merge without branch");
        assert!(
            matches!(
                error,
                VerifyError::MergeNotFollowedByBranch { block } if block == BlockId(0)
            ),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpReturn\n\
             %dead = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("unreachable block");
        assert!(
            matches!(error, VerifyError::UnreachableBlock { block } if block == BlockId(1)),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %sum = OpIAdd %int %v1 %v1\n\
             %p = OpPhi %int %v1 %entry %v1 %entry\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("phi after instructions");
        assert!(
            matches!(error, VerifyError::PhiNotAtBlockStart { block } if block == BlockId(0)),
            "unexpected error: {error}"
        );

        let mut module = Module::new(Target::opencl());
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::new());
        let func = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh");
        module.begin_body(func).expect("body");
        let first = module.push_block(func, "entry").expect("block");
        let second = module.push_block(func, "entry").expect("block");
        module
            .emit(func, first, Inst::none(Op::Return))
            .expect("emit");
        module
            .emit(func, second, Inst::none(Op::Return))
            .expect("emit");
        let error = verify(&module).expect_err("duplicate block name");
        assert!(
            matches!(error, VerifyError::DuplicateBlockName { block } if block == second),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_uses_without_dominating_definitions() {
        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %early = OpIAdd %int %late %late\n\
             %late = OpIAdd %int %v1 %v1\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("use before definition");
        assert!(
            matches!(
                error,
                VerifyError::UndefinedUse { block, .. } if block == BlockId(0)
            ),
            "unexpected error: {error}"
        );

        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %vb = OpConstantTrue %bool\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             OpSelectionMerge %merge None\n\
             OpBranchConditional %vb %then %merge\n\
             %then = OpLabel\n\
             %hidden = OpIAdd %int %v1 %v1\n\
             OpBranch %merge\n\
             %merge = OpLabel\n\
             %used = OpIAdd %int %hidden %hidden\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("value does not dominate its use");
        assert!(
            matches!(
                error,
                VerifyError::UndefinedUse { block, .. } if block == BlockId(2)
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_entry_point_and_linkage_errors() {
        let declaration = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn_void = OpTypeFunction %void

%helper = OpFunction %void None %fn_void
OpFunctionEnd
";
        let error = verify_err(declaration);
        assert!(
            matches!(error, VerifyError::DeclarationNotImported { .. }),
            "unexpected error: {error}"
        );

        let with_entry = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn_void = OpTypeFunction %void

OpEntryPoint Kernel %helper \"helper\"

%helper = OpFunction %void None %fn_void
OpFunctionEnd
";
        let error = verify_err(with_entry);
        assert!(
            matches!(error, VerifyError::EntryPointWithoutBody { index: 0 }),
            "unexpected error: {error}"
        );

        let empty_name = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn_void = OpTypeFunction %void

OpEntryPoint Kernel %main \"\"

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    OpReturn
OpFunctionEnd
";
        let error = verify_err(empty_name);
        assert!(
            matches!(
                error,
                VerifyError::EmptyName {
                    what: "entry point name",
                    index: 0,
                }
            ),
            "unexpected error: {error}"
        );

        let imported_body = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn_void = OpTypeFunction %void

OpDecorate %main LinkageAttributes \"main\" Import

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    OpReturn
OpFunctionEnd
";
        let error = verify_err(imported_body);
        assert!(
            matches!(error, VerifyError::ImportedWithBody { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_bad_globals_and_barriers() {
        let globals = "\
target opencl address physical64 memory opencl

%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ptr_cw_int = OpTypePointer CrossWorkgroup %int
%vf = OpConstant %float 1.5

%g = OpVariable %ptr_cw_int CrossWorkgroup %vf
";
        let module = parse_ok(globals);
        let int = type_id(
            &module,
            Type::Int {
                bits: 32,
                signed: true,
            },
        );
        let float = type_id(&module, Type::Float { bits: 32 });
        let error = verify(&module).expect_err("global initializer type");
        assert!(
            matches!(
                &error,
                VerifyError::GlobalInitTypeMismatch { expected, found, .. }
                    if *expected == int && *found == float
            ),
            "unexpected error: {error}"
        );

        let non_constant = "\
target opencl address physical64 memory opencl

%int = OpTypeInt 32 1
%ptr_cw_int = OpTypePointer CrossWorkgroup %int

%g1 = OpVariable %ptr_cw_int CrossWorkgroup
%g2 = OpVariable %ptr_cw_int CrossWorkgroup %g1
";
        let error = verify_err(non_constant);
        assert!(
            matches!(error, VerifyError::GlobalInitNotConstant { .. }),
            "unexpected error: {error}"
        );

        let module = parse_ok(
            "target opencl address physical64 memory opencl\n\
             \n\
             %void = OpTypeVoid\n\
             %int = OpTypeInt 32 1\n\
             %fn_void = OpTypeFunction %void\n\
             %ptr_cw_int = OpTypePointer CrossWorkgroup %int\n\
             \n\
             %g1 = OpVariable %ptr_cw_int CrossWorkgroup\n\
             \n\
             OpEntryPoint Kernel %main \"main\"\n\
             \n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %s = OpLoad %int %g1\n\
             OpControlBarrier %s %s %s\n\
             OpReturn\n\
             OpFunctionEnd\n",
        );
        let error = verify(&module).expect_err("non-constant barrier operand");
        assert!(
            matches!(
                error,
                VerifyError::NotConstant {
                    what: "barrier execution scope",
                    ..
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_malformed_phi_and_unknown_extended_instructions() {
        let module = parse_ok(&format!(
            "{PREAMBLE}\n\
             %main = OpFunction %void None %fn_void\n\
             %entry = OpLabel\n\
             %p = OpPhi %int %v1 %entry\n\
             OpReturn\n\
             OpFunctionEnd\n"
        ));
        let error = verify(&module).expect_err("single-pair phi");
        assert!(
            matches!(error, VerifyError::MalformedPhi { found: 1 }),
            "unexpected error: {error}"
        );

        let text = "\
target opencl address physical64 memory opencl

%OpenCL.std = OpExtInstImport \"OpenCL.std\"

%void = OpTypeVoid
%float = OpTypeFloat 32
%fn_void = OpTypeFunction %void
%vf = OpConstant %float 1.5

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    %r = OpExtInst %float %OpenCL.std 999 %vf
    OpReturn
OpFunctionEnd
";
        let error = verify_err(text);
        assert!(
            matches!(error, VerifyError::UnknownExtendedInstruction { inst: 999, .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_duplicate_definition_and_orphan_values() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::new());
        let func = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh");
        module.begin_body(func).expect("body");
        let block = module.push_block(func, "entry").expect("block");
        let c0 = module
            .intern_const(int, ConstValue::Int(0))
            .expect("fits");
        let shared = module.new_inst_value(int);
        let add = Op::Binary {
            op: BinOp::IAdd,
            lhs: c0,
            rhs: c0,
        };
        module
            .emit(func, block, Inst::def(shared, int, add.clone()))
            .expect("emit");
        module
            .emit(func, block, Inst::def(shared, int, add))
            .expect("emit");
        module
            .emit(func, block, Inst::none(Op::Return))
            .expect("emit");
        let error = verify(&module).expect_err("shared result id");
        assert!(
            matches!(error, VerifyError::DuplicateDefinition { id } if id == shared),
            "unexpected error: {error}"
        );

        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let orphan = module.new_inst_value(int);
        let error = verify(&module).expect_err("never defined value");
        assert!(
            matches!(error, VerifyError::OrphanValue { id } if id == orphan),
            "unexpected error: {error}"
        );

        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::from([int]));
        let func = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh");
        let arg = module.function(func).expect("f").args[0];
        let c0 = module
            .intern_const(int, ConstValue::Int(0))
            .expect("fits");
        module.begin_body(func).expect("body");
        let block = module.push_block(func, "entry").expect("block");
        let copy = Op::CopyObject { operand: c0 };
        module
            .emit(func, block, Inst::def(arg, int, copy))
            .expect("emit");
        module
            .emit(func, block, Inst::none(Op::Return))
            .expect("emit");
        let error = verify(&module).expect_err("argument as result");
        assert!(
            matches!(error, VerifyError::NotAnInstructionResult { id } if id == arg),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_signature_and_argument_mismatches() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::from([int, int]));
        let func = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh");
        module.function_mut(func).expect("f").sig = int;
        let error = verify(&module).expect_err("signature is not a function type");
        assert!(
            matches!(
                error,
                VerifyError::SignatureNotFunctionType { found } if found == int
            ),
            "unexpected error: {error}"
        );

        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::from([int, int]));
        let func = module
            .add_function("f", sig, Linkage::External)
            .expect("fresh");
        let removed = module.function_mut(func).expect("f").args.pop();
        assert!(removed.is_some());
        let error = verify(&module).expect_err("one argument short");
        assert!(
            matches!(
                error,
                VerifyError::ArgumentCountMismatch {
                    expected: 2,
                    found: 1,
                    ..
                }
            ),
            "unexpected error: {error}"
        );

        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let float = module.float_ty(32);
        let void = module.void_ty();
        let int_sig = module.fn_ty(void, Vec::from([int, int]));
        let float_sig = module.fn_ty(void, Vec::from([float]));
        let func = module
            .add_function("f", int_sig, Linkage::External)
            .expect("fresh");
        let other = module
            .add_function("g", float_sig, Linkage::Import)
            .expect("fresh");
        let other_arg = module.function(other).expect("g").args[0];
        module.function_mut(func).expect("f").args[0] = other_arg;
        let error = verify(&module).expect_err("argument type mismatch");
        assert!(
            matches!(
                error,
                VerifyError::ArgumentTypeMismatch { arg, expected, found }
                    if arg == other_arg && expected == int && found == float
            ),
            "unexpected error: {error}"
        );
    }
}
