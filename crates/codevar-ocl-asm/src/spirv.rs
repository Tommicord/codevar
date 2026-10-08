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

//! SPIR-V binary backend of the assembler stage.
//!
//! Encodes a Codevar [`Module`] as a SPIR-V 1.0 module:
//!
//! * physical layout — a 5-word header, then instructions whose first
//!   word packs `WordCount << 16 | opcode` (SPIR-V §2.3);
//! * logical layout — capabilities, extended-instruction imports, the
//!   memory model, entry points, debug names, annotations,
//!   types/constants/globals, function declarations, and function
//!   definitions (SPIR-V §2.4).
//!
//! [`assemble`] does not run the IR verifier; call [`crate::assemble`]
//! for the checked entry point.
//!
//! [`Module`]: codevar_ocl_ir::ir::Module

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{
    AddressingModel, BinOp, BlockId, CmpOp, ConstValue, ConvOp, ExecutionModel, ExtSetId, Inst, Linkage,
    MemoryModel, Module, Op, Storage, Type, TypeId, UnOp, ValueId, ValueKind,
};
use codevar_ocl_ir::spirv::ops::{
    self, OP_ACCESSCHAIN, OP_BITCAST, OP_BITWISEAND, OP_BITWISEOR, OP_BITWISEXOR, OP_BRANCH,
    OP_BRANCHCONDITIONAL, OP_CAPABILITY, OP_COMPOSITECONSTRUCT, OP_COMPOSITEEXTRACT, OP_CONSTANT,
    OP_CONSTANTFALSE, OP_CONSTANTNULL, OP_CONSTANTTRUE, OP_CONTROLBARRIER, OP_CONVERTFTOS, OP_CONVERTFTOU,
    OP_CONVERTSTOF, OP_CONVERTUTOF, OP_COPYOBJECT, OP_DECORATE, OP_ENTRYPOINT, OP_EXTINST, OP_EXTINSTIMPORT,
    OP_FADD, OP_FCONVERT, OP_FDIV, OP_FMUL, OP_FNEGATE, OP_FORDEQUAL, OP_FORDGREATERTHAN,
    OP_FORDGREATERTHANEQUAL, OP_FORDLESSTHAN, OP_FORDLESSTHANEQUAL, OP_FORDNOTEQUAL, OP_FREM, OP_FSUB,
    OP_FUNCTION, OP_FUNCTIONCALL, OP_FUNCTIONEND, OP_FUNCTIONPARAMETER, OP_IADD, OP_IEQUAL, OP_IMUL,
    OP_INOTEQUAL, OP_ISUB, OP_LABEL, OP_LOAD, OP_LOGICALAND, OP_LOGICALNOT, OP_LOGICALOR, OP_LOOPMERGE,
    OP_MEMORYBARRIER, OP_MEMORYMODEL, OP_NAME, OP_NOT, OP_PHI, OP_PTRACCESSCHAIN, OP_RETURN, OP_RETURNVALUE,
    OP_SCONVERT, OP_SDIV, OP_SELECT, OP_SELECTIONMERGE, OP_SGREATERTHAN, OP_SGREATERTHANEQUAL,
    OP_SHIFTLEFTLOGICAL, OP_SHIFTRIGHTARITHMETIC, OP_SHIFTRIGHTLOGICAL, OP_SLESSTHAN, OP_SLESSTHANEQUAL,
    OP_SNEGATE, OP_SREM, OP_STORE, OP_TYPEARRAY, OP_TYPEBOOL, OP_TYPEFLOAT, OP_TYPEFUNCTION, OP_TYPEINT,
    OP_TYPEPOINTER, OP_TYPESTRUCT, OP_TYPEVECTOR, OP_TYPEVOID, OP_UCONVERT, OP_UDIV, OP_UGREATERTHAN,
    OP_UGREATERTHANEQUAL, OP_ULESSTHAN, OP_ULESSTHANEQUAL, OP_UMOD, OP_UNDEF, OP_UNREACHABLE, OP_VARIABLE,
};

use crate::AssembleError;

/// The SPIR-V magic number that starts every module.
const MAGIC: u32 = 0x0723_0203;

/// The SPIR-V 1.0 version word: major 1, minor 0.
const VERSION_1_0: u32 = 0x0001_0000;

/// The generator id written into the header (`'C''v'` plus a build
/// number); other tools use it to detect which assembler produced a
/// module.
const GENERATOR: u32 = 0x4376_0001;

/// Largest legal SPIR-V string length in bytes including the
/// terminating NUL (SPIR-V §2.3: at most 65535 characters).
const MAX_STRING_BYTES: usize = 65536;

/// Rejects module features this backend does not implement.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when an entry point uses an
/// execution model other than [`ExecutionModel::Kernel`].
fn check_supported(module: &Module) -> Result<(), AssembleError> {
    for entry in &module.entry_points {
        if entry.model != ExecutionModel::Kernel {
            return Err(AssembleError::Unsupported {
                what: "a non-Kernel execution model",
            });
        }
    }
    Ok(())
}

/// The word stream being assembled: a header placeholder followed by
/// instructions.
struct Encoder {
    words: Vec<u32>,
}

impl Encoder {
    /// Starts a module with the five header words; the bound is
    /// patched by [`Encoder::finish`] once every id is known.
    fn new() -> Self {
        Self {
            words: Vec::from([MAGIC, VERSION_1_0, GENERATOR, 0, 0]),
        }
    }

    /// Appends one literal word.
    fn word(&mut self, word: u32) {
        self.words.push(word);
    }

    /// Appends a NUL-terminated, 4-byte-padded UTF-8 string.
    ///
    /// # Errors
    ///
    /// [`AssembleError::StringTooLong`] when the string (including its
    /// NUL) would exceed [`MAX_STRING_BYTES`].
    fn string(&mut self, text: &str) -> Result<(), AssembleError> {
        let bytes = text.as_bytes();
        if bytes.len() + 1 > MAX_STRING_BYTES {
            return Err(AssembleError::StringTooLong { len: bytes.len() });
        }
        let mut padded = Vec::with_capacity(bytes.len() + 4);
        padded.extend_from_slice(bytes);
        padded.push(0);
        while padded.len() % 4 != 0 {
            padded.push(0);
        }
        for word in padded.chunks_exact(4) {
            let [a, b, c, d] = [word[0], word[1], word[2], word[3]];
            self.words.push(u32::from_le_bytes([a, b, c, d]));
        }
        Ok(())
    }

    /// Writes one instruction: `operands` pushes the operand words, and
    /// the leading word is patched to `(WordCount << 16) | opcode`.
    ///
    /// # Errors
    ///
    /// [`AssembleError::InstructionTooLarge`] when the instruction
    /// exceeds the 16-bit word count, or whatever `operands` returns.
    fn instruction(
        &mut self,
        opcode: u16,
        operands: impl FnOnce(&mut Self) -> Result<(), AssembleError>,
    ) -> Result<(), AssembleError> {
        let start = self.words.len();
        self.words.push(0);
        operands(self)?;
        let count = self.words.len() - start;
        let count = u16::try_from(count).map_err(|_| AssembleError::InstructionTooLarge { opcode })?;
        self.words[start] = (u32::from(count) << 16) | u32::from(opcode);
        Ok(())
    }

    /// Patches the header bound and returns the finished module.
    fn finish(mut self, bound: u32) -> Vec<u32> {
        self.words[3] = bound;
        self.words
    }
}

/// Serializes SPIR-V words as little-endian bytes.
#[must_use]
pub fn to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// The `<id>` assignment of one module.
///
/// Ids are handed out from 1 upward: every type, value, extended
/// instruction set, and basic block receives exactly one id, and the
/// largest assigned id plus one becomes the header bound.
struct Layout {
    /// Type ids, parallel to [`Module::types`]; structurally equal
    /// canonical types share an id.
    types: Vec<u32>,
    /// Value ids, parallel to [`Module::values`].
    values: Vec<u32>,
    /// Extended-instruction-set ids, parallel to
    /// [`Module::ext_inst_sets`].
    ext_sets: Vec<u32>,
    /// Label ids, parallel to [`Module::functions`] and within a
    /// function to its block list (declarations have none).
    labels: Vec<Vec<u32>>,
    /// The next unallocated id; also the module bound once assignment
    /// and emission finish.
    next: u32,
}

impl Layout {
    /// Assigns ids to every module-scope handle.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when the arena breaks the
    /// declare-before-use order or contains a non-function entry, and
    /// [`AssembleError::IdExhausted`] when ids run out.
    fn assign(module: &Module) -> Result<Self, AssembleError> {
        let mut layout = Self {
            types: Vec::new(),
            values: Vec::new(),
            ext_sets: Vec::new(),
            labels: Vec::new(),
            next: 1,
        };
        let mut canonical: BTreeMap<Type, u32> = BTreeMap::new();
        for def in module.types() {
            let key = layout.canonical(&def.ty)?;
            let id = match canonical.get(&key) {
                Some(&id) => id,
                None => {
                    let id = layout.alloc()?;
                    canonical.insert(key, id);
                    id
                }
            };
            layout.types.push(id);
        }
        for _value in module.values() {
            let id = layout.alloc()?;
            layout.values.push(id);
        }
        for _set in &module.ext_inst_sets {
            let id = layout.alloc()?;
            layout.ext_sets.push(id);
        }
        for &function in module.functions() {
            let def = module
                .function(function)
                .ok_or(AssembleError::Unsupported {
                    what: "a non-function value in the function list",
                })?;
            let mut labels = Vec::new();
            if let Some(body) = &def.body {
                for _block in body {
                    let id = layout.alloc()?;
                    labels.push(id);
                }
            }
            layout.labels.push(labels);
        }
        Ok(layout)
    }

    /// The structural key used to deduplicate types: child ids are
    /// translated to their assigned ids, and integer signedness is
    /// dropped because SPIR-V Kernel rules force `OpTypeInt`
    /// signedness to 0 (int and uint of a width share one id).
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when a child type is declared
    /// after its parent.
    fn canonical(&self, ty: &Type) -> Result<Type, AssembleError> {
        Ok(match ty {
            Type::Int { bits, signed: _ } => Type::Int {
                bits: *bits,
                signed: false,
            },
            Type::Vector { elem, len } => Type::Vector {
                elem: self.assigned(*elem)?,
                len: *len,
            },
            Type::Array { elem, len } => Type::Array {
                elem: self.assigned(*elem)?,
                len: *len,
            },
            Type::Struct { fields } => Type::Struct {
                fields: fields
                    .iter()
                    .map(|field| self.assigned(*field))
                    .collect::<Result<Vec<_>, AssembleError>>()?,
            },
            Type::Pointer { storage, pointee } => Type::Pointer {
                storage: *storage,
                pointee: self.assigned(*pointee)?,
            },
            Type::Function { ret, params } => Type::Function {
                ret: self.assigned(*ret)?,
                params: params
                    .iter()
                    .map(|param| self.assigned(*param))
                    .collect::<Result<Vec<_>, AssembleError>>()?,
            },
            other => other.clone(),
        })
    }

    /// The assigned id of an already-processed type.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when `id` has not been assigned,
    /// i.e. the type was used before it was declared.
    fn assigned(&self, id: TypeId) -> Result<TypeId, AssembleError> {
        self.types
            .get(id.index())
            .map(|&assigned| TypeId(assigned))
            .ok_or(AssembleError::Unsupported {
                what: "a type used before it is declared",
            })
    }

    /// Allocates the next id.
    ///
    /// # Errors
    ///
    /// [`AssembleError::IdExhausted`] when the id space wraps.
    fn alloc(&mut self) -> Result<u32, AssembleError> {
        let id = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(AssembleError::IdExhausted)?;
        Ok(id)
    }

    /// The SPIR-V id of an IR type.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when the id is out of range.
    fn ty(&self, id: TypeId) -> Result<u32, AssembleError> {
        self.types
            .get(id.index())
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range type id",
            })
    }

    /// The SPIR-V id of an IR value.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when the id is out of range.
    fn value(&self, id: ValueId) -> Result<u32, AssembleError> {
        self.values
            .get(id.index())
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range value id",
            })
    }

    /// The SPIR-V id of an imported extended-instruction set.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when the id is out of range.
    fn ext_set(&self, id: ExtSetId) -> Result<u32, AssembleError> {
        self.ext_sets
            .get(id.index())
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range extended instruction set id",
            })
    }

    /// The label id of a block inside the function at `function`,
    /// which indexes [`Module::functions`].
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when the block is out of range.
    fn label(&self, function: usize, block: BlockId) -> Result<u32, AssembleError> {
        self.labels
            .get(function)
            .and_then(|labels| labels.get(block.index()))
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range basic block",
            })
    }
}

/// The SPIR-V storage class of an IR pointer type.
const fn storageclass_of(storage: Storage) -> u32 {
    match storage {
        Storage::UniformConstant => ops::STORAGECLASS_UNIFORMCONSTANT,
        Storage::Input => ops::STORAGECLASS_INPUT,
        Storage::Output => ops::STORAGECLASS_OUTPUT,
        Storage::Workgroup => ops::STORAGECLASS_WORKGROUP,
        Storage::CrossWorkgroup => ops::STORAGECLASS_CROSSWORKGROUP,
        Storage::Private => ops::STORAGECLASS_PRIVATE,
        Storage::Function => ops::STORAGECLASS_FUNCTION,
    }
}

/// The type of value without trusting arena bounds.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when the id is out of range.
fn value_ty(module: &Module, id: ValueId) -> Result<TypeId, AssembleError> {
    module
        .values()
        .get(id.index())
        .map(|value| value.ty)
        .ok_or(AssembleError::Unsupported {
            what: "an out-of-range value id",
        })
}

/// Emits one `OpCapability`.
fn emit_capability(enc: &mut Encoder, capability: u32) -> Result<(), AssembleError> {
    enc.instruction(OP_CAPABILITY, |e| {
        e.word(capability);
        Ok(())
    })
}

/// Emits the capability list: the OpenCL Kernel baseline plus one
/// capability per optional feature the type arena actually uses.
fn emit_capabilities(module: &Module, enc: &mut Encoder) -> Result<(), AssembleError> {
    emit_capability(enc, ops::CAPABILITY_KERNEL)?;
    emit_capability(enc, ops::CAPABILITY_ADDRESSES)?;
    emit_capability(enc, ops::CAPABILITY_LINKAGE)?;
    let (mut int8, mut int16, mut int64) = (false, false, false);
    let (mut float16, mut float64, mut vector16) = (false, false, false);
    for def in module.types() {
        match &def.ty {
            Type::Int { bits: 8, .. } => int8 = true,
            Type::Int { bits: 16, .. } => int16 = true,
            Type::Int { bits: 64, .. } => int64 = true,
            Type::Float { bits: 16 } => float16 = true,
            Type::Float { bits: 64 } => float64 = true,
            Type::Vector { len, .. } if *len > 4 => vector16 = true,
            _ => {}
        }
    }
    if int8 {
        emit_capability(enc, ops::CAPABILITY_INT8)?;
    }
    if int16 {
        emit_capability(enc, ops::CAPABILITY_INT16)?;
    }
    if int64 {
        emit_capability(enc, ops::CAPABILITY_INT64)?;
    }
    if float16 {
        emit_capability(enc, ops::CAPABILITY_FLOAT16)?;
    }
    if float64 {
        emit_capability(enc, ops::CAPABILITY_FLOAT64)?;
    }
    if vector16 {
        emit_capability(enc, ops::CAPABILITY_VECTOR16)?;
    }
    Ok(())
}

/// Emits one `OpExtInstImport` per imported extended-instruction set.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] on an out-of-range set id, or a
/// string error from [`Encoder::string`].
fn emit_ext_inst_imports(module: &Module, layout: &Layout, enc: &mut Encoder) -> Result<(), AssembleError> {
    for (index, set) in module.ext_inst_sets.iter().enumerate() {
        let id = layout
            .ext_sets
            .get(index)
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range extended instruction set id",
            })?;
        enc.instruction(OP_EXTINSTIMPORT, |e| {
            e.word(id);
            e.string(&set.name)?;
            Ok(())
        })?;
    }
    Ok(())
}

/// Emits `OpMemoryModel` from the module's target environment.
///
/// # Errors
///
/// Never in practice; the signature matches the other emitters.
fn emit_memory_model(module: &Module, enc: &mut Encoder) -> Result<(), AssembleError> {
    let addressing = match module.target.addressing {
        AddressingModel::Logical => ops::ADDRESSINGMODEL_LOGICAL,
        AddressingModel::Physical32 => ops::ADDRESSINGMODEL_PHYSICAL32,
        AddressingModel::Physical64 => ops::ADDRESSINGMODEL_PHYSICAL64,
    };
    let memory = match module.target.memory {
        MemoryModel::Simple => ops::MEMORYMODEL_SIMPLE,
        MemoryModel::GLSL450 => ops::MEMORYMODEL_GLSL450,
        MemoryModel::OpenCL => ops::MEMORYMODEL_OPENCL,
    };
    enc.instruction(OP_MEMORYMODEL, |e| {
        e.word(addressing);
        e.word(memory);
        Ok(())
    })
}

/// Emits one `OpEntryPoint` per kernel, with `Input`/`Output` globals
/// in the interface (empty for OpenCL).
///
/// # Errors
///
/// [`AssembleError::Unsupported`] for a non-Kernel model or an
/// out-of-range id.
fn emit_entry_points(module: &Module, layout: &Layout, enc: &mut Encoder) -> Result<(), AssembleError> {
    for entry in &module.entry_points {
        let model = match entry.model {
            ExecutionModel::Kernel => ops::EXECUTIONMODEL_KERNEL,
            ExecutionModel::GLCompute => {
                return Err(AssembleError::Unsupported {
                    what: "a non-Kernel execution model",
                });
            }
        };
        let function = layout.value(entry.func)?;
        enc.instruction(OP_ENTRYPOINT, |e| {
            e.word(model);
            e.word(function);
            e.string(&entry.name)?;
            for &global in module.globals() {
                let storage = module.pointer_parts(value_ty(module, global)?);
                if matches!(storage, Some((Storage::Input | Storage::Output, _))) {
                    e.word(layout.value(global)?);
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// Emits `OpName` for every named value, block, and named type.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] on an out-of-range id, or a string
/// error from [`Encoder::string`].
fn emit_names(module: &Module, layout: &Layout, enc: &mut Encoder) -> Result<(), AssembleError> {
    for (index, value) in module.values().iter().enumerate() {
        if value.name.is_empty() {
            continue;
        }
        let id = layout
            .values
            .get(index)
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range value id",
            })?;
        enc.instruction(OP_NAME, |e| {
            e.word(id);
            e.string(&value.name)?;
            Ok(())
        })?;
    }
    for (function, &id) in module.functions().iter().enumerate() {
        let Some(def) = module.function(id) else {
            continue;
        };
        let Some(body) = &def.body else {
            continue;
        };
        for (block, basic) in body.iter().enumerate() {
            if basic.name.is_empty() {
                continue;
            }
            let index = block_id(block)?;
            let label = layout.label(function, index)?;
            enc.instruction(OP_NAME, |e| {
                e.word(label);
                e.string(&basic.name)?;
                Ok(())
            })?;
        }
    }
    for (index, def) in module.types().iter().enumerate() {
        if def.name.is_empty() {
            continue;
        }
        let id = layout
            .types
            .get(index)
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range type id",
            })?;
        enc.instruction(OP_NAME, |e| {
            e.word(id);
            e.string(&def.name)?;
            Ok(())
        })?;
    }
    Ok(())
}

/// Emits `OpDecorate … LinkageAttributes` for every function and
/// global, mirroring `print`: the [`Linkage`] payload field is the
/// single source of truth, and [`Module::decorations`] is not
/// consulted (the parser never fills it).
///
/// SPIR-V has no `Internal` enumerant and treats entities without
/// linkage attributes as module-local, so [`Linkage::External`] and
/// [`Linkage::Internal`] emit no decoration.
///
/// [`Module::decorations`]: codevar_ocl_ir::ir::Module#structfield.decorations
///
/// # Errors
///
/// [`AssembleError::Unsupported`] on an out-of-range target, or a
/// string error from [`Encoder::string`].
fn emit_decorations(module: &Module, layout: &Layout, enc: &mut Encoder) -> Result<(), AssembleError> {
    for id in module
        .globals()
        .iter()
        .copied()
        .chain(module.functions().iter().copied())
    {
        let linkage = match module.value_kind(id) {
            ValueKind::Global(global) => global.linkage,
            ValueKind::Function(function) => function.linkage,
            _ => continue,
        };
        let linkage_type = match linkage {
            Linkage::External | Linkage::Internal => continue,
            Linkage::Import => ops::LINKAGETYPE_IMPORT,
            Linkage::Export => ops::LINKAGETYPE_EXPORT,
        };
        let value = module.value(id);
        let name = if value.name.is_empty() {
            format!("{}", id.index())
        } else {
            value.name.clone()
        };
        let target = layout.value(id)?;
        enc.instruction(OP_DECORATE, |e| {
            e.word(target);
            e.word(ops::DECORATION_LINKAGEATTRIBUTES);
            e.string(&name)?;
            e.word(linkage_type);
            Ok(())
        })?;
    }
    Ok(())
}

/// A block handle from a block index.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when the index does not fit.
fn block_id(index: usize) -> Result<BlockId, AssembleError> {
    u32::try_from(index)
        .map(BlockId)
        .map_err(|_| AssembleError::Unsupported {
            what: "too many basic blocks in one function",
        })
}

/// The SPIR-V opcode of a unary operation.
const fn unary_opcode(op: UnOp) -> u16 {
    match op {
        UnOp::Not => OP_NOT,
        UnOp::SNegate => OP_SNEGATE,
        UnOp::FNegate => OP_FNEGATE,
        UnOp::LogicalNot => OP_LOGICALNOT,
    }
}

/// The SPIR-V opcode of a binary operation.
const fn binary_opcode(op: BinOp) -> u16 {
    match op {
        BinOp::IAdd => OP_IADD,
        BinOp::ISub => OP_ISUB,
        BinOp::IMul => OP_IMUL,
        BinOp::SDiv => OP_SDIV,
        BinOp::UDiv => OP_UDIV,
        BinOp::SRem => OP_SREM,
        BinOp::UMod => OP_UMOD,
        BinOp::FAdd => OP_FADD,
        BinOp::FSub => OP_FSUB,
        BinOp::FMul => OP_FMUL,
        BinOp::FDiv => OP_FDIV,
        BinOp::FRem => OP_FREM,
        BinOp::BitwiseAnd => OP_BITWISEAND,
        BinOp::BitwiseOr => OP_BITWISEOR,
        BinOp::BitwiseXor => OP_BITWISEXOR,
        BinOp::ShiftLeftLogical => OP_SHIFTLEFTLOGICAL,
        BinOp::ShiftRightArithmetic => OP_SHIFTRIGHTARITHMETIC,
        BinOp::ShiftRightLogical => OP_SHIFTRIGHTLOGICAL,
        BinOp::LogicalAnd => OP_LOGICALAND,
        BinOp::LogicalOr => OP_LOGICALOR,
    }
}

/// The SPIR-V opcode of a comparison.
const fn compare_opcode(op: CmpOp) -> u16 {
    match op {
        CmpOp::IEqual => OP_IEQUAL,
        CmpOp::INotEqual => OP_INOTEQUAL,
        CmpOp::SLessThan => OP_SLESSTHAN,
        CmpOp::SLessThanEqual => OP_SLESSTHANEQUAL,
        CmpOp::SGreaterThan => OP_SGREATERTHAN,
        CmpOp::SGreaterThanEqual => OP_SGREATERTHANEQUAL,
        CmpOp::ULessThan => OP_ULESSTHAN,
        CmpOp::ULessThanEqual => OP_ULESSTHANEQUAL,
        CmpOp::UGreaterThan => OP_UGREATERTHAN,
        CmpOp::UGreaterThanEqual => OP_UGREATERTHANEQUAL,
        CmpOp::FOrdEqual => OP_FORDEQUAL,
        CmpOp::FOrdNotEqual => OP_FORDNOTEQUAL,
        CmpOp::FOrdLessThan => OP_FORDLESSTHAN,
        CmpOp::FOrdLessThanEqual => OP_FORDLESSTHANEQUAL,
        CmpOp::FOrdGreaterThan => OP_FORDGREATERTHAN,
        CmpOp::FOrdGreaterThanEqual => OP_FORDGREATERTHANEQUAL,
    }
}

/// The SPIR-V opcode of a scalar conversion.
const fn convert_opcode(op: ConvOp) -> u16 {
    match op {
        ConvOp::SConvert => OP_SCONVERT,
        ConvOp::UConvert => OP_UCONVERT,
        ConvOp::FConvert => OP_FCONVERT,
        ConvOp::ConvertFToS => OP_CONVERTFTOS,
        ConvOp::ConvertFToU => OP_CONVERTFTOU,
        ConvOp::ConvertSToF => OP_CONVERTSTOF,
        ConvOp::ConvertUToF => OP_CONVERTUTOF,
        ConvOp::Bitcast => OP_BITCAST,
    }
}

/// Resolves a list of value ids to SPIR-V ids.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when any id is out of range.
fn value_ids(layout: &Layout, ids: &[ValueId]) -> Result<Vec<u32>, AssembleError> {
    ids.iter().map(|&id| layout.value(id)).collect()
}

/// Resolves a list of type ids to SPIR-V ids.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when any id is out of range.
fn type_ids(layout: &Layout, ids: &[TypeId]) -> Result<Vec<u32>, AssembleError> {
    ids.iter().map(|&id| layout.ty(id)).collect()
}

/// Emits one type instruction, skipping ids already emitted (the
/// deduplicated integers and the array-length `uint32` share ids).
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when a referenced id is out of
/// range or an array lacks its length constant.
fn emit_type(
    id: u32,
    ty: &Type,
    layout: &Layout,
    lengths: &BTreeMap<u32, u32>,
    enc: &mut Encoder,
    emitted: &mut BTreeSet<u32>,
) -> Result<(), AssembleError> {
    if !emitted.insert(id) {
        return Ok(());
    }
    match ty {
        Type::Void => enc.instruction(OP_TYPEVOID, |e| {
            e.word(id);
            Ok(())
        })?,
        Type::Bool => enc.instruction(OP_TYPEBOOL, |e| {
            e.word(id);
            Ok(())
        })?,
        Type::Int { bits, signed: _ } => enc.instruction(OP_TYPEINT, |e| {
            e.word(id);
            e.word(u32::from(*bits));
            // Kernel rules: signedness is always 0.
            e.word(0);
            Ok(())
        })?,
        Type::Float { bits } => enc.instruction(OP_TYPEFLOAT, |e| {
            e.word(id);
            e.word(u32::from(*bits));
            Ok(())
        })?,
        Type::Vector { elem, len } => {
            let elem = layout.ty(*elem)?;
            enc.instruction(OP_TYPEVECTOR, |e| {
                e.word(id);
                e.word(elem);
                e.word(*len);
                Ok(())
            })?;
        }
        Type::Array { elem, len } => {
            let elem = layout.ty(*elem)?;
            let length = lengths
                .get(len)
                .copied()
                .ok_or(AssembleError::Unsupported {
                    what: "a missing array length constant",
                })?;
            enc.instruction(OP_TYPEARRAY, |e| {
                e.word(id);
                e.word(elem);
                e.word(length);
                Ok(())
            })?;
        }
        Type::Struct { fields } => {
            let fields = type_ids(layout, fields)?;
            enc.instruction(OP_TYPESTRUCT, |e| {
                e.word(id);
                for field in fields {
                    e.word(field);
                }
                Ok(())
            })?;
        }
        Type::Pointer { storage, pointee } => {
            let pointee = layout.ty(*pointee)?;
            enc.instruction(OP_TYPEPOINTER, |e| {
                e.word(id);
                e.word(storageclass_of(*storage));
                e.word(pointee);
                Ok(())
            })?;
        }
        Type::Function { ret, params } => {
            let ret = layout.ty(*ret)?;
            let params = type_ids(layout, params)?;
            enc.instruction(OP_TYPEFUNCTION, |e| {
                e.word(id);
                e.word(ret);
                for param in params {
                    e.word(param);
                }
                Ok(())
            })?;
        }
    }
    Ok(())
}

/// Emits the type, constant, and global-variable region: array-length
/// constants first (they precede the array types that reference them),
/// then every type, constant, and module-scope variable.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when the arena or a payload cannot
/// be encoded, [`AssembleError::IdExhausted`] when ids run out.
fn emit_types_constants_globals(
    module: &Module,
    layout: &mut Layout,
    enc: &mut Encoder,
) -> Result<(), AssembleError> {
    let mut emitted: BTreeSet<u32> = BTreeSet::new();
    let mut lengths: BTreeMap<u32, u32> = BTreeMap::new();
    if module
        .types()
        .iter()
        .any(|def| matches!(def.ty, Type::Array { .. }))
    {
        // OpTypeArray counts elements with a uint32 constant: reuse an
        // interned Int{32} (all signedness variants share one id), or
        // synthesize one.  Either way its instruction goes first so the
        // constants below can reference it.
        let uint32 = match module
            .types()
            .iter()
            .position(|def| matches!(def.ty, Type::Int { bits: 32, .. }))
        {
            Some(index) => layout
                .types
                .get(index)
                .copied()
                .ok_or(AssembleError::Unsupported {
                    what: "an out-of-range type id",
                })?,
            None => layout.alloc()?,
        };
        enc.instruction(OP_TYPEINT, |e| {
            e.word(uint32);
            e.word(32);
            e.word(0);
            Ok(())
        })?;
        emitted.insert(uint32);
        let array_lengths: BTreeSet<u32> = module
            .types()
            .iter()
            .filter_map(|def| match def.ty {
                Type::Array { len, .. } => Some(len),
                _ => None,
            })
            .collect();
        for len in array_lengths {
            let id = layout.alloc()?;
            enc.instruction(OP_CONSTANT, |e| {
                e.word(uint32);
                e.word(id);
                e.word(len);
                Ok(())
            })?;
            lengths.insert(len, id);
        }
    }
    for (index, def) in module.types().iter().enumerate() {
        let id = layout
            .types
            .get(index)
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range type id",
            })?;
        emit_type(id, &def.ty, layout, &lengths, enc, &mut emitted)?;
    }
    for (index, value) in module.values().iter().enumerate() {
        let ValueKind::Constant(payload) = &value.kind else {
            continue;
        };
        let id = layout
            .values
            .get(index)
            .copied()
            .ok_or(AssembleError::Unsupported {
                what: "an out-of-range value id",
            })?;
        let ty = layout.ty(value.ty)?;
        emit_constant(ty, id, value.ty, payload, module, enc)?;
    }
    for &global in module.globals() {
        let id = layout.value(global)?;
        let ty = value_ty(module, global)?;
        let (storage, _) = module
            .pointer_parts(ty)
            .ok_or(AssembleError::Unsupported {
                what: "a global whose type is not a pointer",
            })?;
        let pointer = layout.ty(ty)?;
        let init = module
            .global(global)
            .ok_or(AssembleError::Unsupported {
                what: "a global without a payload",
            })?
            .init
            .map(|init| layout.value(init))
            .transpose()?;
        enc.instruction(OP_VARIABLE, |e| {
            e.word(pointer);
            e.word(id);
            e.word(storageclass_of(storage));
            if let Some(init) = init {
                e.word(init);
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// Emits the instruction for one interned constant.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] when the payload does not fit the
/// value's type.
fn emit_constant(
    ty: u32,
    id: u32,
    type_id: TypeId,
    payload: &ConstValue,
    module: &Module,
    enc: &mut Encoder,
) -> Result<(), AssembleError> {
    let payload_too_large = || AssembleError::Unsupported {
        what: "an integer constant that does not fit its type",
    };
    match payload {
        ConstValue::Int(raw) => {
            let width = match module.ty(type_id) {
                Type::Int { bits, .. } => *bits,
                _ => {
                    return Err(AssembleError::Unsupported {
                        what: "an integer constant of a non-integer type",
                    });
                }
            };
            enc.instruction(OP_CONSTANT, |e| {
                e.word(ty);
                e.word(id);
                if width <= 32 {
                    e.word(u32::try_from(*raw).map_err(|_| payload_too_large())?);
                } else {
                    e.word(u32::try_from(*raw & 0xFFFF_FFFF).map_err(|_| payload_too_large())?);
                    e.word(u32::try_from(*raw >> 32).map_err(|_| payload_too_large())?);
                }
                Ok(())
            })?;
        }
        ConstValue::Float32(bits) => enc.instruction(OP_CONSTANT, |e| {
            e.word(ty);
            e.word(id);
            e.word(*bits);
            Ok(())
        })?,
        ConstValue::Float64(bits) => enc.instruction(OP_CONSTANT, |e| {
            e.word(ty);
            e.word(id);
            e.word(u32::try_from(*bits & 0xFFFF_FFFF).map_err(|_| payload_too_large())?);
            e.word(u32::try_from(*bits >> 32).map_err(|_| payload_too_large())?);
            Ok(())
        })?,
        ConstValue::Bool(value) => {
            let opcode = if *value { OP_CONSTANTTRUE } else { OP_CONSTANTFALSE };
            enc.instruction(opcode, |e| {
                e.word(ty);
                e.word(id);
                Ok(())
            })?;
        }
        ConstValue::Undef => enc.instruction(OP_UNDEF, |e| {
            e.word(ty);
            e.word(id);
            Ok(())
        })?,
        ConstValue::Null => enc.instruction(OP_CONSTANTNULL, |e| {
            e.word(ty);
            e.word(id);
            Ok(())
        })?,
    }
    Ok(())
}

/// Emits every function: SPIR-V requires all declarations (no basic
/// blocks) before all definitions, so `Module::functions` is stably
/// partitioned while label ids keep their original order.
///
/// # Errors
///
/// Propagates the errors of [`emit_function`].
fn emit_functions(module: &Module, layout: &Layout, enc: &mut Encoder) -> Result<(), AssembleError> {
    let mut order = (0..module.functions().len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| {
        match module
            .functions()
            .get(index)
            .and_then(|&id| module.function(id))
        {
            Some(def) if def.is_declaration() => 0,
            _ => 1,
        }
    });
    for index in order {
        emit_function(module, index, layout, enc)?;
    }
    Ok(())
}

/// Emits one function: `OpFunction`, parameters, blocks, `OpFunctionEnd`.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] for a malformed function or
/// out-of-range id, plus whatever instruction emission returns.
fn emit_function(
    module: &Module,
    function: usize,
    layout: &Layout,
    enc: &mut Encoder,
) -> Result<(), AssembleError> {
    let &id = module
        .functions()
        .get(function)
        .ok_or(AssembleError::Unsupported {
            what: "an out-of-range function id",
        })?;
    let def = module
        .function(id)
        .ok_or(AssembleError::Unsupported {
            what: "a non-function value in the function list",
        })?;
    let function_id = layout.value(id)?;
    let ret = match module.ty(def.sig) {
        Type::Function { ret, .. } => *ret,
        _ => {
            return Err(AssembleError::Unsupported {
                what: "a function whose signature is not a function type",
            });
        }
    };
    let ret = layout.ty(ret)?;
    let sig = layout.ty(def.sig)?;
    enc.instruction(OP_FUNCTION, |e| {
        e.word(ret);
        e.word(function_id);
        e.word(def.control.bits());
        e.word(sig);
        Ok(())
    })?;
    for &arg in &def.args {
        let arg_id = layout.value(arg)?;
        let arg_ty = layout.ty(value_ty(module, arg)?)?;
        enc.instruction(OP_FUNCTIONPARAMETER, |e| {
            e.word(arg_ty);
            e.word(arg_id);
            Ok(())
        })?;
    }
    if let Some(body) = &def.body {
        for (index, block) in body.iter().enumerate() {
            let label = layout.label(function, block_id(index)?)?;
            enc.instruction(OP_LABEL, |e| {
                e.word(label);
                Ok(())
            })?;
            for inst in &block.insts {
                emit_inst(module, function, layout, enc, inst)?;
            }
        }
    }
    enc.instruction(OP_FUNCTIONEND, |_| Ok(()))?;
    Ok(())
}

/// Emits one body instruction: opcode and operand words first, then
/// the optional type/result head.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] for an out-of-range id, a malformed
/// result/type pair, or a non-pointer variable; other errors come from
/// the encoder.
fn emit_inst(
    module: &Module,
    function: usize,
    layout: &Layout,
    enc: &mut Encoder,
    inst: &Inst,
) -> Result<(), AssembleError> {
    let mut operands: Vec<u32> = Vec::new();
    let opcode = match &inst.op {
        Op::Variable { init } => {
            let result = inst.result.ok_or(AssembleError::Unsupported {
                what: "a variable without a result",
            })?;
            let (storage, _) = module
                .pointer_parts(value_ty(module, result)?)
                .ok_or(AssembleError::Unsupported {
                    what: "a variable whose result is not a pointer",
                })?;
            operands.push(storageclass_of(storage));
            if let Some(init) = init {
                operands.push(layout.value(*init)?);
            }
            OP_VARIABLE
        }
        Op::Load { ptr } => {
            operands.push(layout.value(*ptr)?);
            OP_LOAD
        }
        Op::Store { ptr, value } => {
            operands.push(layout.value(*ptr)?);
            operands.push(layout.value(*value)?);
            OP_STORE
        }
        Op::AccessChain { base, indices } => {
            operands.push(layout.value(*base)?);
            operands.extend(value_ids(layout, indices)?);
            OP_ACCESSCHAIN
        }
        Op::PtrAccessChain { base, indices } => {
            operands.push(layout.value(*base)?);
            operands.extend(value_ids(layout, indices)?);
            OP_PTRACCESSCHAIN
        }
        Op::CopyObject { operand } => {
            operands.push(layout.value(*operand)?);
            OP_COPYOBJECT
        }
        Op::Unary { op, operand } => {
            operands.push(layout.value(*operand)?);
            unary_opcode(*op)
        }
        Op::Binary { op, lhs, rhs } => {
            operands.push(layout.value(*lhs)?);
            operands.push(layout.value(*rhs)?);
            binary_opcode(*op)
        }
        Op::Compare { op, lhs, rhs } => {
            operands.push(layout.value(*lhs)?);
            operands.push(layout.value(*rhs)?);
            compare_opcode(*op)
        }
        Op::Select { cond, a, b } => {
            operands.push(layout.value(*cond)?);
            operands.push(layout.value(*a)?);
            operands.push(layout.value(*b)?);
            OP_SELECT
        }
        Op::Convert { op, operand } => {
            operands.push(layout.value(*operand)?);
            convert_opcode(*op)
        }
        Op::Call { callee, args } => {
            operands.push(layout.value(*callee)?);
            operands.extend(value_ids(layout, args)?);
            OP_FUNCTIONCALL
        }
        Op::ExtInst {
            set,
            inst: number,
            args,
        } => {
            operands.push(layout.ext_set(*set)?);
            operands.push(*number);
            operands.extend(value_ids(layout, args)?);
            OP_EXTINST
        }
        Op::CompositeExtract { composite, indices } => {
            operands.push(layout.value(*composite)?);
            operands.extend(indices.iter().copied());
            OP_COMPOSITEEXTRACT
        }
        Op::CompositeConstruct { constituents } => {
            operands.extend(value_ids(layout, constituents)?);
            OP_COMPOSITECONSTRUCT
        }
        Op::Phi { incomings } => {
            for &(value, block) in incomings {
                operands.push(layout.value(value)?);
                operands.push(layout.label(function, block)?);
            }
            OP_PHI
        }
        Op::ControlBarrier { exec, mem, semantics } => {
            operands.push(layout.value(*exec)?);
            operands.push(layout.value(*mem)?);
            operands.push(layout.value(*semantics)?);
            OP_CONTROLBARRIER
        }
        Op::MemoryBarrier { mem, semantics } => {
            operands.push(layout.value(*mem)?);
            operands.push(layout.value(*semantics)?);
            OP_MEMORYBARRIER
        }
        Op::SelectionMerge { target, control } => {
            operands.push(layout.label(function, *target)?);
            operands.push(*control);
            OP_SELECTIONMERGE
        }
        Op::LoopMerge { merge, cont, control } => {
            operands.push(layout.label(function, *merge)?);
            operands.push(layout.label(function, *cont)?);
            operands.push(*control);
            OP_LOOPMERGE
        }
        Op::Branch { target } => {
            operands.push(layout.label(function, *target)?);
            OP_BRANCH
        }
        Op::BranchConditional { cond, then, other } => {
            operands.push(layout.value(*cond)?);
            operands.push(layout.label(function, *then)?);
            operands.push(layout.label(function, *other)?);
            OP_BRANCHCONDITIONAL
        }
        Op::Return => OP_RETURN,
        Op::ReturnValue { value } => {
            operands.push(layout.value(*value)?);
            OP_RETURNVALUE
        }
        Op::Unreachable => OP_UNREACHABLE,
    };
    enc.instruction(opcode, |e| {
        match (inst.ty, inst.result) {
            (Some(ty), Some(result)) => {
                e.word(layout.ty(ty)?);
                e.word(layout.value(result)?);
            }
            (None, None) => {}
            _ => {
                return Err(AssembleError::Unsupported {
                    what: "an instruction with mismatched result and type",
                });
            }
        }
        for word in &operands {
            e.word(*word);
        }
        Ok(())
    })
}

/// Assembles `module` into SPIR-V words.
///
/// The IR verifier is not run here; [`crate::assemble`] verifies
/// first.  Instruction order follows the SPIR-V logical layout.
///
/// # Errors
///
/// [`AssembleError::Unsupported`] for a non-Kernel entry point or a
/// construct the backend cannot encode; [`AssembleError::IdExhausted`]
/// when ids run out; encoder errors for oversized instructions or
/// strings.
pub fn assemble(module: &Module) -> Result<Vec<u32>, AssembleError> {
    check_supported(module)?;
    let mut layout = Layout::assign(module)?;
    let mut enc = Encoder::new();
    emit_capabilities(module, &mut enc)?;
    emit_ext_inst_imports(module, &layout, &mut enc)?;
    emit_memory_model(module, &mut enc)?;
    emit_entry_points(module, &layout, &mut enc)?;
    emit_names(module, &layout, &mut enc)?;
    emit_decorations(module, &layout, &mut enc)?;
    emit_types_constants_globals(module, &mut layout, &mut enc)?;
    emit_functions(module, &layout, &mut enc)?;
    Ok(enc.finish(layout.next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use codevar_ocl_ir::ir::{EntryPoint, Target};
    use codevar_ocl_ir::parse::parse;

    /// One decoded instruction: opcode plus operand words.
    struct Instr {
        opcode: u16,
        operands: Vec<u32>,
    }

    /// Splits a finished module into instructions after the 5-word
    /// header; panics on a word count of zero (the encoder never
    /// emits one).
    fn decode(words: &[u32]) -> Vec<Instr> {
        let mut instructions = Vec::new();
        let mut index = 5;
        while index < words.len() {
            let word = words[index];
            let count = usize::try_from(word >> 16).expect("word count fits usize");
            assert!(count >= 1, "instruction must occupy at least one word");
            let opcode = u16::try_from(word & 0xFFFF).expect("opcode fits u16");
            assert!(index + count <= words.len(), "instruction must fit in the module");
            instructions.push(Instr {
                opcode,
                operands: words[index + 1..index + count].to_vec(),
            });
            index += count;
        }
        assert_eq!(index, words.len(), "instructions must consume the module");
        instructions
    }

    /// Decodes the NUL-terminated string starting at operand `start`
    /// and returns it with the index just past its padded words.
    fn string_operand(words: &[u32], start: usize) -> (String, usize) {
        let mut text = String::new();
        let mut index = start;
        loop {
            let word = words.get(index).copied().unwrap_or(0);
            for byte in word.to_le_bytes() {
                if byte == 0 {
                    return (text, index + 1);
                }
                text.push(char::from(byte));
            }
            index += 1;
        }
    }

    /// The logical-layout section rank of an opcode: instructions may
    /// only move to a higher-or-equal rank as the module progresses.
    fn section(opcode: u16) -> u8 {
        match opcode {
            OP_CAPABILITY => 0,
            OP_EXTINSTIMPORT => 1,
            OP_MEMORYMODEL => 2,
            OP_ENTRYPOINT => 3,
            OP_NAME => 4,
            OP_DECORATE => 5,
            OP_TYPEVOID | OP_TYPEBOOL | OP_TYPEINT | OP_TYPEFLOAT | OP_TYPEVECTOR | OP_TYPEARRAY
            | OP_TYPESTRUCT | OP_TYPEPOINTER | OP_TYPEFUNCTION | OP_CONSTANT | OP_CONSTANTTRUE
            | OP_CONSTANTFALSE | OP_CONSTANTNULL | OP_UNDEF | OP_VARIABLE => 6,
            _ => 7,
        }
    }

    /// A well-formed kernel whose function *definition* appears before
    /// the imported declaration it calls, so the backend must reorder
    /// them for the logical layout.
    const KERNEL: &str = "\
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

%get_global_id = OpFunction %uint None %fn_uint_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd
";

    /// Parses [`KERNEL`].
    fn kernel_module() -> Module {
        parse(KERNEL).expect("kernel parses")
    }

    #[test]
    fn empty_module_encodes_exact_layout() {
        let module = Module::new(Target::opencl());
        let words = assemble(&module).expect("empty module assembles");
        assert_eq!(
            words,
            vec![
                MAGIC,
                VERSION_1_0,
                GENERATOR,
                1,
                0,
                (2 << 16) | u32::from(OP_CAPABILITY),
                ops::CAPABILITY_KERNEL,
                (2 << 16) | u32::from(OP_CAPABILITY),
                ops::CAPABILITY_ADDRESSES,
                (2 << 16) | u32::from(OP_CAPABILITY),
                ops::CAPABILITY_LINKAGE,
                (3 << 16) | u32::from(OP_MEMORYMODEL),
                ops::ADDRESSINGMODEL_PHYSICAL64,
                ops::MEMORYMODEL_OPENCL,
            ]
        );
    }

    #[test]
    fn bound_covers_every_assigned_id() {
        let module = kernel_module();
        let words = assemble(&module).expect("kernel assembles");
        let layout = Layout::assign(&module).expect("ids assign");
        // No arrays means emission allocates nothing beyond `assign`.
        assert_eq!(words[3], layout.next);
        assert!(words[3] >= 1, "bound must exceed every id");
    }

    #[test]
    fn sections_follow_the_logical_layout() {
        let words = assemble(&kernel_module()).expect("kernel assembles");
        let instructions = decode(&words);
        let mut rank = 0;
        let mut functions_started = false;
        for instruction in &instructions {
            // Everything from the first OpFunction onward is the
            // function region, including function-scope OpVariable.
            let next = if functions_started {
                7
            } else {
                section(instruction.opcode)
            };
            if instruction.opcode == OP_FUNCTION {
                functions_started = true;
            }
            assert!(
                next >= rank,
                "opcode {} appeared out of section order",
                instruction.opcode
            );
            rank = next;
        }
        assert!(functions_started, "fixture must contain a function");
    }

    #[test]
    fn type_int_signedness_is_always_zero() {
        let words = assemble(&kernel_module()).expect("kernel assembles");
        let instructions = decode(&words);
        let mut seen = 0;
        for instruction in &instructions {
            if instruction.opcode == OP_TYPEINT {
                assert_eq!(instruction.operands[2], 0, "Kernel signedness must be 0");
                seen += 1;
            }
        }
        // The fixture's uint and int share one signless id.
        assert_eq!(seen, 1, "one canonical 32-bit integer type");
    }

    #[test]
    fn declarations_precede_definitions() {
        let words = assemble(&kernel_module()).expect("kernel assembles");
        let instructions = decode(&words);
        let mut seen_definition = false;
        let mut in_function = false;
        let mut has_label = false;
        let mut functions = 0;
        for instruction in &instructions {
            match instruction.opcode {
                OP_FUNCTION => {
                    functions += 1;
                    in_function = true;
                    has_label = false;
                }
                OP_LABEL => has_label = true,
                OP_FUNCTIONEND => {
                    assert!(in_function, "OpFunctionEnd must follow OpFunction");
                    in_function = false;
                    if has_label {
                        seen_definition = true;
                    } else {
                        assert!(!seen_definition, "a declaration must precede every definition");
                    }
                }
                _ => {}
            }
        }
        assert!(!in_function, "every function must end");
        assert_eq!(functions, 2, "fixture has one definition and one declaration");
        assert!(seen_definition, "fixture must contain a definition");
    }

    #[test]
    fn entry_point_is_kernel_with_the_right_name() {
        let words = assemble(&kernel_module()).expect("kernel assembles");
        let instructions = decode(&words);
        let entry = instructions
            .iter()
            .find(|instruction| instruction.opcode == OP_ENTRYPOINT)
            .expect("entry point present");
        assert_eq!(entry.operands[0], ops::EXECUTIONMODEL_KERNEL);
        assert!(entry.operands[1] >= 1, "function id must be a valid id");
        let (name, _) = string_operand(&entry.operands, 2);
        assert_eq!(name, "vector_add");
    }

    #[test]
    fn declarations_carry_linkage_attributes_import() {
        let words = assemble(&kernel_module()).expect("kernel assembles");
        let instructions = decode(&words);
        let decorate = instructions
            .iter()
            .find(|instruction| instruction.opcode == OP_DECORATE)
            .expect("linkage decoration present");
        assert_eq!(decorate.operands[1], ops::DECORATION_LINKAGEATTRIBUTES);
        let (name, linkage_at) = string_operand(&decorate.operands, 2);
        assert_eq!(name, "get_global_id");
        assert_eq!(decorate.operands[linkage_at], ops::LINKAGETYPE_IMPORT);
    }

    #[test]
    fn every_block_ends_in_a_terminator() {
        let words = assemble(&kernel_module()).expect("kernel assembles");
        let instructions = decode(&words);
        for (index, instruction) in instructions.iter().enumerate() {
            if instruction.opcode != OP_LABEL {
                continue;
            }
            let end = instructions[index + 1..]
                .iter()
                .position(|next| matches!(next.opcode, OP_LABEL | OP_FUNCTIONEND))
                .map_or(instructions.len(), |offset| index + 1 + offset);
            let last = instructions[end - 1].opcode;
            assert!(
                matches!(
                    last,
                    OP_BRANCH | OP_BRANCHCONDITIONAL | OP_RETURN | OP_RETURNVALUE | OP_UNREACHABLE
                ),
                "block must end in a terminator, found opcode {last}"
            );
        }
    }

    #[test]
    fn wide_integer_constant_splits_into_two_words() {
        let mut module = Module::new(Target::opencl());
        let long = module.int_ty(64, true);
        module
            .intern_const(long, ConstValue::Int(0xDEAD_BEEF_CAFE_BABE))
            .expect("constant interns");
        let words = assemble(&module).expect("module assembles");
        let constant = decode(&words)
            .iter()
            .find(|instruction| instruction.opcode == OP_CONSTANT && instruction.operands.len() == 4)
            .expect("i64 constant present")
            .operands
            .clone();
        assert_eq!(constant[2], 0xCAFE_BABE, "low word first");
        assert_eq!(constant[3], 0xDEAD_BEEF, "high word second");
    }

    #[test]
    fn float_constant_keeps_its_bit_pattern() {
        let mut module = Module::new(Target::opencl());
        let float = module.float_ty(32);
        module
            .intern_const(float, ConstValue::Float32(0x4140_0000))
            .expect("constant interns");
        let words = assemble(&module).expect("module assembles");
        let constant = decode(&words)
            .iter()
            .find(|instruction| instruction.opcode == OP_CONSTANT && instruction.operands.len() == 3)
            .expect("f32 constant present")
            .operands
            .clone();
        assert_eq!(constant[2], 0x4140_0000, "12.0f bit pattern");
    }

    #[test]
    fn array_type_shares_an_interned_length_constant() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        module.intern_type(Type::Array { elem: int, len: 4 });
        let words = assemble(&module).expect("module assembles");
        let instructions = decode(&words);
        let typeint = instructions
            .iter()
            .position(|instruction| instruction.opcode == OP_TYPEINT && instruction.operands[1] == 32)
            .expect("uint32 type present");
        let uint32 = instructions[typeint].operands[0];
        let constant = instructions
            .iter()
            .position(|instruction| {
                instruction.opcode == OP_CONSTANT
                    && instruction.operands[0] == uint32
                    && instruction.operands[2] == 4
            })
            .expect("length constant present");
        let array = instructions
            .iter()
            .position(|instruction| instruction.opcode == OP_TYPEARRAY)
            .expect("array type present");
        assert_eq!(instructions[constant].operands[0], uint32, "constant typed");
        assert_eq!(instructions[array].operands[1], uint32, "element is uint32");
        assert_eq!(
            instructions[array].operands[2], instructions[constant].operands[1],
            "array counts with the constant"
        );
        assert!(typeint < constant && constant < array, "declare before use");
    }

    #[test]
    fn array_without_int32_synthesizes_the_uint32_type() {
        let mut module = Module::new(Target::opencl());
        let half = module.int_ty(16, false);
        module.intern_type(Type::Array { elem: half, len: 2 });
        let words = assemble(&module).expect("module assembles");
        let instructions = decode(&words);
        assert!(
            instructions
                .iter()
                .any(|instruction| { instruction.opcode == OP_TYPEINT && instruction.operands[1] == 32 }),
            "a uint32 type must exist for the length constant"
        );
    }

    #[test]
    fn non_kernel_entry_point_is_rejected() {
        let mut module = Module::new(Target::opencl());
        module.entry_points.push(EntryPoint {
            model: ExecutionModel::GLCompute,
            func: ValueId(0),
            name: String::from("main"),
        });
        let error = assemble(&module).expect_err("GLCompute must be rejected");
        assert!(
            matches!(
                error,
                AssembleError::Unsupported {
                    what: "a non-Kernel execution model"
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn string_over_the_limit_is_rejected() {
        let mut encoder = Encoder::new();
        let text = String::from("x").repeat(65536);
        let error = encoder
            .string(&text)
            .expect_err("oversized string");
        assert_eq!(error, AssembleError::StringTooLong { len: 65536 });
    }

    #[test]
    fn oversized_instruction_is_rejected() {
        let mut encoder = Encoder::new();
        let error = encoder
            .instruction(0, |e| {
                for _ in 0..70_000 {
                    e.word(0);
                }
                Ok(())
            })
            .expect_err("oversized instruction");
        assert_eq!(error, AssembleError::InstructionTooLarge { opcode: 0 });
    }

    #[test]
    fn to_bytes_is_little_endian() {
        assert_eq!(to_bytes(&[MAGIC]), vec![0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn wide_types_request_capabilities() {
        let mut module = Module::new(Target::opencl());
        module.int_ty(64, true);
        let float = module.float_ty(32);
        module.intern_type(Type::Vector { elem: float, len: 8 });
        let words = assemble(&module).expect("module assembles");
        let capabilities: Vec<u32> = decode(&words)
            .iter()
            .filter(|instruction| instruction.opcode == OP_CAPABILITY)
            .map(|instruction| instruction.operands[0])
            .collect();
        assert!(capabilities.contains(&ops::CAPABILITY_INT64));
        assert!(capabilities.contains(&ops::CAPABILITY_VECTOR16));
    }

    #[test]
    fn verified_kernel_assembles_end_to_end() {
        let module = kernel_module();
        let words = crate::assemble(&module).expect("verify and assemble");
        assert!(words.len() > 5, "module must contain instructions");
    }
}
