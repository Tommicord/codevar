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

//! Selection: the intermediate representation becomes machine code's
//! own language.
//!
//! The selection stage walks the verified module and produces a
//! [`MirModule`](crate::mir::MirModule): virtual registers instead of
//! values, semantic operations instead of dialect operations, an
//! explicit frame layout, and the kernel ABI already folded into the
//! instruction stream.
//!
//! # What the stage decides
//!
//! * **The work-item queries** become
//!   [`MOp::WorkItemLoad`](crate::mir::MOp::WorkItemLoad) operations,
//!   and the kernel's parameters become
//!   [`MOp::ArgLoad`](crate::mir::MOp::ArgLoad) operations, so the
//!   entry function reads its arguments the way the driver wrote them.
//! * **The stack slots** become frame slots, addressed by
//!   [`MOp::FrameAddr`](crate::mir::MOp::FrameAddr).
//! * **The calls** become [`MOp::Call`](crate::mir::MOp::Call)
//!   operations naming a symbol of the module.
//! * **The types** become machine types: the dialect's types are
//!   mapped onto the machine's own, with the narrow ones widened and
//!   an explicit conversion left behind.
//!
//! # The kernel ABI
//!
//! The compiled kernel is an `unsafe extern "C" fn(*mut WorkItem)`:
//! the entry function has exactly one machine parameter, the work-item
//! pointer, and it reads the kernel's arguments from the block that
//! pointer names.  The stage therefore emits
//! [`MOp::ArgBlockBase`](crate::mir::MOp::ArgBlockBase) once and one
//! [`MOp::ArgLoad`](crate::mir::MOp::ArgLoad) per kernel argument.
//!
//! # The work-item queries
//!
//! The front end lowers a work-item query to a load of a module-scope
//! variable carrying a built-in decoration.  The stage recognizes
//! those loads and replaces them with the reads of the per-work-item
//! state:
//!
//! * A query of a constant component becomes a
//!   [`MOp::WorkItemLoad`](crate::mir::MOp::WorkItemLoad) of that
//!   component.
//! * A query of a computed component becomes the same read, selected
//!   by the component index.
//! * A query of the global size becomes the product of the group
//!   count and the local size.
//! * A query of a linear identifier becomes the mixed-radix sum of
//!   the multi-dimensional ones.
//!
//! The state holds thirty-two-bit components, so a read of a
//! sixty-four-bit query is widened with an explicit zero extension.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_abi::WorkItemField;
use codevar_ocl_ir::ir::{
    BasicBlock, BinOp, CmpOp, ConstValue, ConvOp, Decor, FunctionDef, Inst, Module, Op, Type, TypeId, UnOp,
    ValueId, ValueKind,
};

use crate::abi::{WorkItemSource, argument_kind, extended_instruction_name, math_op, work_item_source};
use crate::error::{CodegenError, Result};
use crate::mir::{
    Destination, FPred, IPred, MInst, MOp, MType, MirBlock, MirFunction, MirModule, Operand, Operands,
    SymId, SymbolKind, VReg,
};

/// The alignment every frame is padded to.
///
/// The frame holds the function's own slots and the slots the
/// allocator spilled into; padding the frame to a multiple of this
/// keeps every slot inside it naturally aligned.
const FRAME_ALIGNMENT: u32 = 16;

/// The largest alignment a slot may ask for.
const MAX_ALIGNMENT: u32 = 16;

/// The result of selecting a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The machine code of the module.
    pub module: MirModule,
    /// The symbol of the entry function.
    pub entry: SymId,
}

/// Selects the machine code of `module`, whose entry function is
/// `entry`.
///
/// The entry function comes first in the module's function list, and
/// every function it can reach follows it.  The kernel's parameters
/// are the entry function's arguments: they are read from the
/// argument block the work item points at, not passed in registers.
///
/// # Errors
///
/// Returns [`CodegenError::Selection`](crate::error::CodegenError::Selection)
/// when the module uses something the machine code cannot express, and
/// [`CodegenError::Unsupported`](crate::error::CodegenError::Unsupported)
/// when the construct has no meaning for a native back end at all.
pub fn select_module(module: &Module, entry: ValueId) -> Result<Selection> {
    Selector::new(module).run(entry)
}

/// Builds a selection error.
fn selection_error(what: &'static str, detail: impl Into<String>) -> CodegenError {
    CodegenError::Selection {
        what,
        detail: detail.into(),
    }
}

/// Counts how often every value of the module is used.
fn count_uses(module: &Module) -> BTreeMap<ValueId, u32> {
    let mut uses: BTreeMap<ValueId, u32> = BTreeMap::new();
    let mut operands: Vec<ValueId> = Vec::new();
    for &function in module.functions() {
        let Some(definition) = module.function(function) else {
            continue;
        };
        let Some(body) = &definition.body else {
            continue;
        };
        for block in body {
            for inst in &block.insts {
                operands.clear();
                crate::opt::util::collect_operands(&inst.op, &mut operands);
                for operand in &operands {
                    *uses.entry(*operand).or_insert(0) += 1;
                }
            }
        }
    }
    uses
}

/// True when an operation only computes a value.
///
/// Such an operation can be left out when nobody reads its result;
/// the ones that touch the memory, call a function or wait at a
/// barrier cannot.
fn is_pure(op: &Op) -> bool {
    matches!(
        op,
        Op::CopyObject { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Compare { .. }
            | Op::Select { .. }
            | Op::Convert { .. }
            | Op::CompositeExtract { .. }
            | Op::CompositeConstruct { .. }
    )
}

/// Builds an unsupported error.
fn unsupported(what: &'static str, detail: impl Into<String>) -> CodegenError {
    CodegenError::Unsupported {
        what,
        detail: detail.into(),
    }
}

/// Rounds `value` up to a multiple of `align`.
#[inline]
const fn align_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

/// The machine type of an intermediate-representation type.
///
/// A pointer names the size of its pointee, and a vector names its
/// element type; the aggregates that have no machine counterpart are
/// rejected.
fn machine_type(module: &Module, ty: TypeId) -> Result<MType> {
    let shape = module.ty(ty);
    Ok(match shape {
        Type::Void => MType::Void,
        Type::Bool => MType::I1,
        Type::Int { bits, .. } => match bits {
            8 => MType::I8,
            16 => MType::I16,
            32 => MType::I32,
            64 => MType::I64,
            other => {
                return Err(unsupported(
                    "an integer of an unusual width",
                    format!("the intermediate representation has {other}-bit integers"),
                ));
            }
        },
        Type::Float { bits } => match bits {
            16 => MType::F16,
            32 => MType::F32,
            64 => MType::F64,
            other => {
                return Err(unsupported(
                    "a float of an unusual width",
                    format!("the intermediate representation has {other}-bit floats"),
                ));
            }
        },
        Type::Vector { elem, len } => MType::Vec {
            lanes: *len,
            elem: Box::new(machine_type(module, *elem)?),
        },
        Type::Pointer { pointee, .. } => MType::Ptr {
            size: type_bytes(module, *pointee)?,
        },
        Type::Array { .. } | Type::Struct { .. } | Type::Function { .. } => {
            return Err(unsupported(
                "an aggregate value",
                format!("the intermediate representation has a {} type", shape.describe()),
            ));
        }
    })
}

/// The size of a type in bytes; zero when the type has no known size.
fn type_bytes(module: &Module, ty: TypeId) -> Result<u32> {
    let shape = module.ty(ty);
    Ok(match shape {
        Type::Void => 0,
        Type::Bool => 1,
        Type::Int { bits, .. } => u32::from(*bits) / 8,
        Type::Float { bits } => u32::from(*bits) / 8,
        Type::Vector { elem, len } => type_bytes(module, *elem)?.saturating_mul(*len),
        Type::Array { elem, len } => type_bytes(module, *elem)?.saturating_mul(*len),
        Type::Pointer { .. } => 8,
        Type::Struct { .. } | Type::Function { .. } => 0,
    })
}

/// The natural alignment of a type in bytes.
fn type_alignment(module: &Module, ty: TypeId) -> u32 {
    let shape = module.ty(ty);
    let align = match shape {
        Type::Void | Type::Bool => 1,
        Type::Int { bits, .. } => u32::from(*bits).div_ceil(8),
        Type::Float { bits } => u32::from(*bits).div_ceil(8),
        Type::Vector { elem, .. } | Type::Array { elem, .. } => type_alignment(module, *elem),
        Type::Pointer { .. } => 8,
        Type::Struct { .. } | Type::Function { .. } => 8,
    };
    align.clamp(1, MAX_ALIGNMENT)
}

/// The width of a scalar type in bits, when it has one.
fn scalar_bits(module: &Module, ty: TypeId) -> Option<u32> {
    match module.ty(ty) {
        Type::Bool => Some(1),
        Type::Int { bits, .. } => Some(u32::from(*bits)),
        Type::Float { bits } => Some(u32::from(*bits)),
        Type::Vector { elem, .. } => scalar_bits(module, *elem),
        _ => None,
    }
}

/// The signedness of an integer type, when it is one.
fn integer_signedness(module: &Module, ty: TypeId) -> Option<bool> {
    match module.ty(ty) {
        Type::Int { signed, .. } => Some(*signed),
        _ => None,
    }
}

/// The machine operation that turns a value of `from` into one of `to`.
fn conversion(from: &MType, to: &MType) -> MOp {
    match (from, to) {
        (MType::I32, MType::I64) => MOp::ZExt,
        (MType::I32, MType::I16) | (MType::I32, MType::I8) => MOp::Trunc,
        (MType::I64, MType::I32) | (MType::I64, MType::I16) | (MType::I64, MType::I8) => MOp::Trunc,
        _ => MOp::Copy,
    }
}

/// True when the source is a scalar query rather than a vector one.
fn is_scalar_source(source: WorkItemSource) -> bool {
    matches!(
        source,
        WorkItemSource::GlobalLinearId | WorkItemSource::LocalLinearId
    )
}

/// The type the read of a work-item source has before it is
/// converted to the type the query asked for.
fn natural_type(source: WorkItemSource) -> MType {
    match source {
        WorkItemSource::GlobalSize => MType::I64,
        _ => MType::I32,
    }
}

/// The whole selection: one machine module out of one intermediate
/// representation.
struct Selector<'a> {
    /// The module being selected.
    module: &'a Module,
    /// The machine module being built.
    out: MirModule,
    /// The symbol of every function the entry can reach.
    syms: BTreeMap<ValueId, SymId>,
    /// The work-item built-ins of the module, by the variable that
    /// carries them.
    builtins: BTreeMap<ValueId, WorkItemSource>,
    /// How often every value of the module is used.
    uses: BTreeMap<ValueId, u32>,
}

impl<'a> Selector<'a> {
    /// Creates a selector over `module`.
    fn new(module: &'a Module) -> Self {
        let mut builtins = BTreeMap::new();
        for decoration in &module.decorations {
            if let Decor::BuiltIn { name } = &decoration.kind
                && let Some(source) = work_item_source(name)
            {
                builtins.insert(decoration.target, source);
            }
        }
        Self {
            module,
            out: MirModule::new(),
            syms: BTreeMap::new(),
            builtins,
            uses: count_uses(module),
        }
    }

    /// Selects the whole module.
    fn run(mut self, entry: ValueId) -> Result<Selection> {
        let definition = self.module.function(entry).ok_or_else(|| {
            selection_error(
                "choosing the entry function",
                "the entry point does not name a function",
            )
        })?;
        if definition.body.is_none() {
            return Err(selection_error(
                "choosing the entry function",
                "the entry function has no body",
            ));
        }
        let order = self.reachable(entry)?;
        self.declare(&order)?;
        let mut entry_symbol = SymId::new(0);
        for (index, &func) in order.iter().enumerate() {
            let function = self.select_function(func, func == entry)?;
            if index == 0 {
                entry_symbol = function.symbol;
            }
            self.out.functions.push(function);
        }
        Ok(Selection {
            module: self.out,
            entry: entry_symbol,
        })
    }

    /// The functions the entry can reach, the entry first.
    ///
    /// The walk follows the calls in program order, so the order of
    /// the result depends only on the module's contents.
    fn reachable(&self, entry: ValueId) -> Result<Vec<ValueId>> {
        let mut order: Vec<ValueId> = Vec::new();
        let mut seen: BTreeSet<ValueId> = BTreeSet::new();
        let mut stack: Vec<ValueId> = vec![entry];
        while let Some(func) = stack.pop() {
            if !seen.insert(func) {
                continue;
            }
            order.push(func);
            let Some(definition) = self.module.function(func) else {
                continue;
            };
            let mut callees: Vec<ValueId> = Vec::new();
            for block in definition.body.iter().flatten() {
                for inst in &block.insts {
                    if let Op::Call { callee, .. } = &inst.op {
                        callees.push(*callee);
                    }
                }
            }
            for callee in callees.into_iter().rev() {
                if !seen.contains(&callee) {
                    stack.push(callee);
                }
            }
        }
        Ok(order)
    }

    /// Gives every reachable function a symbol.
    ///
    /// A function without a body is a declaration: it gets an external
    /// symbol, so the calls that name it stay resolvable.
    fn declare(&mut self, order: &[ValueId]) -> Result<()> {
        for (index, &func) in order.iter().enumerate() {
            let definition = self.module.function(func).ok_or_else(|| {
                selection_error("declaring the functions", format!("{func} is not a function"))
            })?;
            let name = self.module.value(func).name.clone();
            let external = definition.body.is_none();
            let symbol = self
                .out
                .add_symbol(&name, SymbolKind::Function { index }, external);
            self.syms.insert(func, symbol);
        }
        Ok(())
    }

    /// Selects one function.
    fn select_function(&mut self, func: ValueId, is_entry: bool) -> Result<MirFunction> {
        let definition = self
            .module
            .function(func)
            .ok_or_else(|| selection_error("selecting a function", format!("{func} is not a function")))?;
        let body = definition.body.as_deref().unwrap_or(&[]);
        let name = self.module.value(func).name.clone();
        let symbol = self
            .syms
            .get(&func)
            .copied()
            .unwrap_or(SymId::new(0));
        let mut selector = FnSel {
            module: self.module,
            syms: &self.syms,
            def: definition,
            body,
            is_entry,
            out: MirFunction::new(&name),
            vregs: BTreeMap::new(),
            slots: BTreeMap::new(),
            builtins: &self.builtins,
            fused: BTreeMap::new(),
            dyn_elem: BTreeMap::new(),
            copies: BTreeMap::new(),
            phi_defs: BTreeMap::new(),
            work_item: None,
            uses: &self.uses,
        };
        selector.out.symbol = symbol;
        selector.out.is_entry = is_entry;
        selector.run()
    }
}

/// The selection of one function.
struct FnSel<'a> {
    /// The module being selected.
    module: &'a Module,
    /// The symbol of every function of the module.
    syms: &'a BTreeMap<ValueId, SymId>,
    /// The function's definition.
    def: &'a FunctionDef,
    /// The function's blocks.
    body: &'a [BasicBlock],
    /// Whether the function is the module's entry point.
    is_entry: bool,
    /// The machine function being built.
    out: MirFunction,
    /// The virtual register of every value of the function.
    vregs: BTreeMap<ValueId, VReg>,
    /// The frame offset of every stack slot of the function.
    slots: BTreeMap<ValueId, u32>,
    /// The work-item built-ins of the module.
    builtins: &'a BTreeMap<ValueId, WorkItemSource>,
    /// The work-item vector loads whose components are read
    /// separately.
    fused: BTreeMap<ValueId, WorkItemSource>,
    /// The work-item element pointers, by the value that holds them.
    dyn_elem: BTreeMap<ValueId, (WorkItemSource, ValueId)>,
    /// The copies a predecessor block owes to the phis of its
    /// successors.
    copies: BTreeMap<usize, Vec<(VReg, ValueId)>>,
    /// The definitions the merge blocks owe to their phis.
    phi_defs: BTreeMap<usize, Vec<(VReg, ValueId, usize)>>,
    /// The work-item pointer of the entry function.
    work_item: Option<VReg>,
    /// How often every value of the module is used.
    uses: &'a BTreeMap<ValueId, u32>,
}

impl FnSel<'_> {
    /// Selects the whole function.
    fn run(mut self) -> Result<MirFunction> {
        self.allocate_vregs()?;
        self.lay_out_frame();
        self.analyze_phis();
        self.analyze_work_items();
        let preamble = self.emit_preamble()?;
        for index in 0..self.body.len() {
            let block = self.emit_block(index)?;
            self.out.push_block(block);
        }
        if let Some(block) = self.out.blocks.first_mut() {
            let mut merged = preamble;
            merged.append(&mut block.insts);
            block.insts = merged;
        } else {
            let mut block = MirBlock::new("entry");
            block.insts = preamble;
            self.out.push_block(block);
        }
        self.out.frame_size = self.out.fixed_frame_size;
        Ok(self.out)
    }

    /// Gives every value of the function a virtual register.
    fn allocate_vregs(&mut self) -> Result<()> {
        for &arg in &self.def.args {
            let ty = machine_type(self.module, self.module.type_of(arg))?;
            let reg = self.out.new_vreg(ty);
            self.vregs.insert(arg, reg);
        }
        for block in self.body {
            for inst in &block.insts {
                let Some(result) = inst.result else {
                    continue;
                };
                let ty = machine_type(self.module, self.module.type_of(result))?;
                let reg = self.out.new_vreg(ty);
                self.vregs.insert(result, reg);
            }
        }
        Ok(())
    }

    /// Gives every stack slot of the function a frame offset.
    ///
    /// The offsets grow from zero and every slot is aligned to its
    /// natural alignment; the size of the frame is the end of the last
    /// slot, padded to the frame's alignment.
    fn lay_out_frame(&mut self) {
        let mut offset = 0u32;
        for block in self.body {
            for inst in &block.insts {
                let Op::Variable { .. } = &inst.op else {
                    continue;
                };
                let Some(result) = inst.result else {
                    continue;
                };
                let Some((_, pointee)) = self
                    .module
                    .pointer_parts(self.module.type_of(result))
                else {
                    continue;
                };
                let size = type_bytes(self.module, pointee).unwrap_or(0);
                let align = type_alignment(self.module, pointee);
                let start = align_up(offset, align);
                self.slots.insert(result, start);
                offset = start.saturating_add(size);
            }
        }
        self.out.fixed_frame_size = align_up(offset, FRAME_ALIGNMENT);
    }

    /// Collects the copies the phis owe to their blocks.
    ///
    /// A phi has no counterpart in the machine code: every predecessor
    /// copies its incoming value into the phi's register just before
    /// its terminator, and the merge block defines the register once
    /// more.
    fn analyze_phis(&mut self) {
        for (index, block) in self.body.iter().enumerate() {
            for inst in &block.insts {
                let Op::Phi { incomings } = &inst.op else {
                    continue;
                };
                let Some(result) = inst.result else {
                    continue;
                };
                let Some(reg) = self.vregs.get(&result).copied() else {
                    continue;
                };
                if let Some(&(first, _)) = incomings.first() {
                    self.phi_defs
                        .entry(index)
                        .or_default()
                        .push((reg, first, incomings.len()));
                }
                for &(value, pred) in incomings {
                    self.copies
                        .entry(pred.index())
                        .or_default()
                        .push((reg, value));
                }
            }
        }
    }

    /// Recognizes the work-item queries of the function.
    ///
    /// A query of a whole vector is read component by component, so
    /// the load of the built-in variable is folded into the extracts
    /// that read it.  A query of a computed component indexes the
    /// vector, and the index is folded into the load that follows it.
    fn analyze_work_items(&mut self) {
        for block in self.body {
            for inst in &block.insts {
                match &inst.op {
                    Op::Load { ptr } => {
                        let Some(source) = self.builtins.get(ptr).copied() else {
                            continue;
                        };
                        if is_scalar_source(source) {
                            continue;
                        }
                        if let Some(result) = inst.result {
                            self.fused.insert(result, source);
                        }
                    }
                    Op::AccessChain { base, indices } => {
                        let Some(source) = self.builtins.get(base).copied() else {
                            continue;
                        };
                        if is_scalar_source(source) {
                            continue;
                        }
                        let Some(index) = indices.first().copied() else {
                            continue;
                        };
                        if let Some(result) = inst.result {
                            self.dyn_elem.insert(result, (source, index));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// The instructions every block of the function shares.
    ///
    /// The parameters come first, then the kernel's arguments, then
    /// the addresses of the stack slots.
    fn emit_preamble(&mut self) -> Result<Vec<MInst>> {
        let mut insts: Vec<MInst> = Vec::new();
        if self.is_entry {
            let pointer = MType::Ptr { size: 0 };
            let work_item = self.out.new_vreg(pointer.clone());
            self.work_item = Some(work_item);
            self.out.params.push(work_item);
            insts.push(MInst::def(
                work_item,
                pointer.clone(),
                MOp::Param { index: 0 },
                Operands::none(),
            ));
            if !self.def.args.is_empty() {
                let base = self.out.new_vreg(pointer.clone());
                insts.push(MInst::def(
                    base,
                    pointer,
                    MOp::ArgBlockBase,
                    Operands::one(Operand::Virtual(work_item)),
                ));
                for (index, &arg) in self.def.args.iter().enumerate() {
                    let Some(dest) = self.vregs.get(&arg).copied() else {
                        continue;
                    };
                    let ty = self
                        .out
                        .vreg_type(dest)
                        .cloned()
                        .unwrap_or(MType::Void);
                    let kind = argument_kind(&ty);
                    insts.push(MInst::def(
                        dest,
                        ty,
                        MOp::ArgLoad {
                            index: u32::try_from(index).unwrap_or(u32::MAX),
                            kind,
                        },
                        Operands::one(Operand::Virtual(base)),
                    ));
                }
            }
        } else {
            for (index, &arg) in self.def.args.iter().enumerate() {
                let Some(dest) = self.vregs.get(&arg).copied() else {
                    continue;
                };
                let ty = self
                    .out
                    .vreg_type(dest)
                    .cloned()
                    .unwrap_or(MType::Void);
                self.out.params.push(dest);
                insts.push(MInst::def(
                    dest,
                    ty,
                    MOp::Param {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                    },
                    Operands::none(),
                ));
            }
        }
        for block in self.body {
            for inst in &block.insts {
                let Op::Variable { init } = &inst.op else {
                    continue;
                };
                let Some(result) = inst.result else {
                    continue;
                };
                let Some(offset) = self.slots.get(&result).copied() else {
                    continue;
                };
                let Some(dest) = self.vregs.get(&result).copied() else {
                    continue;
                };
                let ty = self
                    .out
                    .vreg_type(dest)
                    .cloned()
                    .unwrap_or(MType::Void);
                insts.push(MInst::def(
                    dest,
                    ty.clone(),
                    MOp::FrameAddr { offset },
                    Operands::none(),
                ));
                if let Some(init) = init {
                    let size = self.pointee_size(result);
                    let value = self.operand(*init)?;
                    insts.push(MInst::none(
                        ty,
                        MOp::Store { size },
                        Operands::two(Operand::Virtual(dest), value),
                    ));
                }
            }
        }
        Ok(insts)
    }

    /// The size of the value a pointer points at.
    fn pointee_size(&self, ptr: ValueId) -> u32 {
        self.module
            .pointer_parts(self.module.type_of(ptr))
            .map(|(_, pointee)| type_bytes(self.module, pointee).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Selects one block.
    fn emit_block(&mut self, index: usize) -> Result<MirBlock> {
        let ir_block = &self.body[index];
        let mut block = MirBlock::new(&ir_block.name);
        if let Some(defs) = self.phi_defs.get(&index).cloned() {
            for (reg, first, count) in defs {
                let ty = self
                    .out
                    .vreg_type(reg)
                    .cloned()
                    .unwrap_or(MType::Void);
                let source = if count <= 1 {
                    match self.vregs.get(&first).copied() {
                        Some(reg) => Operand::Virtual(reg),
                        None => self.operand(first)?,
                    }
                } else {
                    // The predecessors have already written the
                    // value; the copy keeps the register defined in
                    // the merge block.
                    Operand::Virtual(reg)
                };
                block
                    .insts
                    .push(MInst::def(reg, ty, MOp::Copy, Operands::one(source)));
            }
        }
        let last = ir_block.insts.len().saturating_sub(1);
        for (position, inst) in ir_block.insts.iter().enumerate() {
            if position == last
                && let Some(copies) = self.copies.get(&index).cloned()
            {
                for (reg, value) in copies {
                    let ty = self
                        .out
                        .vreg_type(reg)
                        .cloned()
                        .unwrap_or(MType::Void);
                    let source = match self.vregs.get(&value).copied() {
                        Some(source) => Operand::Virtual(source),
                        None => self.operand(value)?,
                    };
                    block
                        .insts
                        .push(MInst::def(reg, ty, MOp::Copy, Operands::one(source)));
                }
            }
            self.emit_inst(&mut block, inst)?;
        }
        Ok(block)
    }

    /// Selects one instruction.
    fn emit_inst(&mut self, block: &mut MirBlock, inst: &Inst) -> Result<()> {
        // A computation nobody reads costs a register and a move, so
        // it is left out of the machine code.
        if let Some(result) = inst.result
            && self.uses.get(&result).copied().unwrap_or(0) == 0
            && is_pure(&inst.op)
        {
            return Ok(());
        }
        match &inst.op {
            Op::Variable { .. } => {
                // The slot's address and its initializer are part of
                // the function's preamble.
            }
            Op::Load { ptr } => self.emit_load(block, inst, *ptr)?,
            Op::Store { ptr, value } => {
                let size = self.pointee_size(*ptr);
                let pointer = self.operand(*ptr)?;
                let value = self.operand(*value)?;
                block.insts.push(MInst::none(
                    MType::Void,
                    MOp::Store { size },
                    Operands::two(pointer, value),
                ));
            }
            Op::AccessChain { base, indices } => {
                self.emit_access_chain(block, inst, *base, indices)?;
            }
            Op::PtrAccessChain { base, indices } => {
                self.emit_access_chain(block, inst, *base, indices)?;
            }
            Op::CopyObject { operand } => {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                let source = self.operand(*operand)?;
                block
                    .insts
                    .push(MInst::def(dest, ty, MOp::Copy, Operands::one(source)));
            }
            Op::Unary { op, operand } => {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                let op = match op {
                    UnOp::Not | UnOp::LogicalNot => MOp::Not,
                    UnOp::SNegate => MOp::Neg,
                    UnOp::FNegate => MOp::FNeg,
                };
                let source = self.operand(*operand)?;
                block
                    .insts
                    .push(MInst::def(dest, ty, op, Operands::one(source)));
            }
            Op::Binary { op, lhs, rhs } => {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                let op = match op {
                    BinOp::IAdd => MOp::Add,
                    BinOp::ISub => MOp::Sub,
                    BinOp::IMul => MOp::Mul,
                    BinOp::SDiv => MOp::SDiv,
                    BinOp::UDiv => MOp::UDiv,
                    BinOp::SRem => MOp::SRem,
                    BinOp::UMod => MOp::URem,
                    BinOp::FAdd => MOp::FAdd,
                    BinOp::FSub => MOp::FSub,
                    BinOp::FMul => MOp::FMul,
                    BinOp::FDiv => MOp::FDiv,
                    BinOp::FRem => MOp::FRem,
                    BinOp::BitwiseAnd | BinOp::LogicalAnd => MOp::And,
                    BinOp::BitwiseOr | BinOp::LogicalOr => MOp::Or,
                    BinOp::BitwiseXor => MOp::Xor,
                    BinOp::ShiftLeftLogical => MOp::Shl,
                    BinOp::ShiftRightArithmetic => MOp::AShr,
                    BinOp::ShiftRightLogical => MOp::LShr,
                };
                let lhs = self.operand(*lhs)?;
                let rhs = self.operand(*rhs)?;
                block
                    .insts
                    .push(MInst::def(dest, ty, op, Operands::two(lhs, rhs)));
            }
            Op::Compare { op, lhs, rhs } => {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                let op = match op {
                    CmpOp::IEqual => MOp::ICmp { pred: IPred::Eq },
                    CmpOp::INotEqual => MOp::ICmp { pred: IPred::Ne },
                    CmpOp::SLessThan => MOp::ICmp { pred: IPred::Slt },
                    CmpOp::SLessThanEqual => MOp::ICmp { pred: IPred::Sle },
                    CmpOp::SGreaterThan => MOp::ICmp { pred: IPred::Sgt },
                    CmpOp::SGreaterThanEqual => MOp::ICmp { pred: IPred::Sge },
                    CmpOp::ULessThan => MOp::ICmp { pred: IPred::Ult },
                    CmpOp::ULessThanEqual => MOp::ICmp { pred: IPred::Ule },
                    CmpOp::UGreaterThan => MOp::ICmp { pred: IPred::Ugt },
                    CmpOp::UGreaterThanEqual => MOp::ICmp { pred: IPred::Uge },
                    CmpOp::FOrdEqual => MOp::FCmp { pred: FPred::Oeq },
                    CmpOp::FOrdNotEqual => MOp::FCmp { pred: FPred::One },
                    CmpOp::FOrdLessThan => MOp::FCmp { pred: FPred::Olt },
                    CmpOp::FOrdLessThanEqual => MOp::FCmp { pred: FPred::Ole },
                    CmpOp::FOrdGreaterThan => MOp::FCmp { pred: FPred::Ogt },
                    CmpOp::FOrdGreaterThanEqual => MOp::FCmp { pred: FPred::Oge },
                };
                let lhs = self.operand(*lhs)?;
                let rhs = self.operand(*rhs)?;
                block
                    .insts
                    .push(MInst::def(dest, ty, op, Operands::two(lhs, rhs)));
            }
            Op::Select { cond, a, b } => {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                let cond = self.operand(*cond)?;
                let a = self.operand(*a)?;
                let b = self.operand(*b)?;
                block
                    .insts
                    .push(MInst::def(dest, ty, MOp::Select, Operands::three(cond, a, b)));
            }
            Op::Convert { op, operand } => {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                let source_ty = self.module.type_of(*operand);
                let op = self.convert_op(*op, source_ty, &ty);
                let source = self.operand(*operand)?;
                block
                    .insts
                    .push(MInst::def(dest, ty, op, Operands::one(source)));
            }
            Op::Call { callee, args } => {
                let symbol = self.syms.get(callee).copied().ok_or_else(|| {
                    selection_error("selecting a call", format!("the callee {callee} has no symbol"))
                })?;
                let mut operands: Vec<Operand> = Vec::with_capacity(args.len());
                for &arg in args {
                    operands.push(self.operand(arg)?);
                }
                let op = MOp::Call { callee: symbol };
                match inst.result {
                    Some(result) => {
                        let dest = self.vreg(result)?;
                        let ty = self.type_of_dest(inst)?;
                        block
                            .insts
                            .push(MInst::def(dest, ty, op, Operands::many(operands)));
                    }
                    None => {
                        block
                            .insts
                            .push(MInst::none(MType::Void, op, Operands::many(operands)));
                    }
                }
            }
            Op::ExtInst {
                set, inst: number, ..
            } => {
                let name = extended_instruction_name(*number);
                let set_name = self.module.ext_inst_set(*set).name.clone();
                let detail = match math_op(*number) {
                    Some(operation) => format!(
                        "the extended instruction {name} of {set_name} is the mathematical operation {operation:?}, which the machine code has no operation for"
                    ),
                    None => {
                        format!("the extended instruction {name} of {set_name} has no machine counterpart")
                    }
                };
                return Err(unsupported("an extended instruction", detail));
            }
            Op::CompositeExtract { composite, indices } => {
                self.emit_extract(block, inst, *composite, indices)?;
            }
            Op::CompositeConstruct { constituents } => {
                self.emit_construct(block, inst, constituents)?;
            }
            Op::Phi { .. } => {
                // The copies live in the predecessors; the merge block
                // defines the result once.
            }
            Op::ControlBarrier { .. } => {
                return Err(unsupported(
                    "a control barrier",
                    "a native work-item has no barrier to wait at",
                ));
            }
            Op::MemoryBarrier { .. } => {
                return Err(unsupported(
                    "a memory barrier",
                    "a native work-item orders its own memory accesses",
                ));
            }
            Op::SelectionMerge { .. } | Op::LoopMerge { .. } => {
                // The merge instructions only mark the branch that
                // follows; the machine code has no counterpart.
            }
            Op::Branch { target } => {
                block.insts.push(MInst::none(
                    MType::Void,
                    MOp::Br {
                        target: u32::try_from(target.index()).unwrap_or(u32::MAX),
                    },
                    Operands::none(),
                ));
            }
            Op::BranchConditional { cond, then, other } => {
                let cond = self.operand(*cond)?;
                block.insts.push(MInst::none(
                    MType::Void,
                    MOp::CondBr,
                    Operands::three(
                        cond,
                        Operand::Immediate(i64::try_from(then.index()).unwrap_or(i64::MAX)),
                        Operand::Immediate(i64::try_from(other.index()).unwrap_or(i64::MAX)),
                    ),
                ));
            }
            Op::Return => {
                block
                    .insts
                    .push(MInst::none(MType::Void, MOp::RetVoid, Operands::none()));
            }
            Op::ReturnValue { value } => {
                let value = self.operand(*value)?;
                block
                    .insts
                    .push(MInst::none(MType::Void, MOp::Ret, Operands::one(value)));
            }
            Op::Unreachable => {
                block
                    .insts
                    .push(MInst::none(MType::Void, MOp::Unreachable, Operands::none()));
            }
        }
        Ok(())
    }

    /// Selects a load.
    fn emit_load(&mut self, block: &mut MirBlock, inst: &Inst, ptr: ValueId) -> Result<()> {
        if let Some((source, index)) = self.dyn_elem.get(&ptr).copied() {
            let dest = self.dest_of(inst)?;
            let ty = self.type_of_dest(inst)?;
            let index = self.operand(index)?;
            return self.emit_work_item_dynamic(block, source, index, dest, ty);
        }
        if let Some(source) = self.builtins.get(&ptr).copied() {
            if is_scalar_source(source) {
                let dest = self.dest_of(inst)?;
                let ty = self.type_of_dest(inst)?;
                return self.emit_work_item_linear(block, source, dest, ty);
            }
            // The components of the vector are read one by one, by
            // the extracts that follow.
            return Ok(());
        }
        let size = self.pointee_size(ptr);
        let pointer = self.operand(ptr)?;
        let Some(result) = inst.result else {
            return Ok(());
        };
        let dest = self.vreg(result)?;
        let ty = self.type_of_dest(inst)?;
        block
            .insts
            .push(MInst::def(dest, ty, MOp::Load { size }, Operands::one(pointer)));
        Ok(())
    }

    /// Selects an address computation.
    fn emit_access_chain(
        &mut self,
        block: &mut MirBlock,
        inst: &Inst,
        base: ValueId,
        indices: &[ValueId],
    ) -> Result<()> {
        if self.builtins.contains_key(&base) {
            // The work-item element is read through the load that
            // follows; the chain itself has no counterpart.
            return Ok(());
        }
        let Some(result) = inst.result else {
            return Ok(());
        };
        let dest = self.vreg(result)?;
        let ty = self.type_of_dest(inst)?;
        let mut current = self
            .module
            .pointer_parts(self.module.type_of(base))
            .map(|(_, pointee)| pointee);
        let mut pointer = self.operand(base)?;
        for (step, &index) in indices.iter().enumerate() {
            let Some(pointee) = current else {
                break;
            };
            let size = self.element_size(pointee)?;
            let signed = integer_signedness(self.module, self.module.type_of(index)).unwrap_or(false);
            let offset = self.operand(index)?;
            let next = if step + 1 == indices.len() {
                dest
            } else {
                self.out.new_vreg(ty.clone())
            };
            block.insts.push(MInst::def(
                next,
                ty.clone(),
                MOp::Gep { size, signed },
                Operands::two(pointer, offset),
            ));
            pointer = Operand::Virtual(next);
            current = self.element_type(pointee);
        }
        Ok(())
    }

    /// The size of one element of the type an address computation
    /// walks into.
    fn element_size(&self, pointee: TypeId) -> Result<u32> {
        match self.module.ty(pointee) {
            Type::Array { elem, .. } | Type::Vector { elem, .. } => type_bytes(self.module, *elem),
            Type::Struct { .. } => Err(unsupported(
                "a member of a structure",
                "the intermediate representation does not lay structures out",
            )),
            _ => type_bytes(self.module, pointee),
        }
    }

    /// The type an address computation walks into.
    fn element_type(&self, pointee: TypeId) -> Option<TypeId> {
        match self.module.ty(pointee) {
            Type::Array { elem, .. } | Type::Vector { elem, .. } => Some(*elem),
            _ => None,
        }
    }

    /// Selects a component read.
    fn emit_extract(
        &mut self,
        block: &mut MirBlock,
        inst: &Inst,
        composite: ValueId,
        indices: &[u32],
    ) -> Result<()> {
        let Some(&index) = indices.first() else {
            return Err(selection_error(
                "selecting a component read",
                "the read names no component",
            ));
        };
        if indices.len() > 1 {
            return Err(unsupported(
                "a nested component read",
                "the machine code reads one component at a time",
            ));
        }
        let dest = self.dest_of(inst)?;
        let ty = self.type_of_dest(inst)?;
        if let Some(source) = self.fused.get(&composite).copied() {
            return self.emit_work_item_lane(block, source, index, dest, ty);
        }
        let composite_type = self.module.type_of(composite);
        if !matches!(self.module.ty(composite_type), Type::Vector { .. }) {
            return Err(unsupported(
                "a component of an aggregate",
                "the machine code reads the components of vectors only",
            ));
        }
        let source = self.operand(composite)?;
        block.insts.push(MInst::def(
            dest,
            ty,
            MOp::VecExtract { index },
            Operands::one(source),
        ));
        Ok(())
    }

    /// Selects a composite construction.
    fn emit_construct(&mut self, block: &mut MirBlock, inst: &Inst, constituents: &[ValueId]) -> Result<()> {
        let dest = self.dest_of(inst)?;
        let ty = self.type_of_dest(inst)?;
        let Some(&first) = constituents.first() else {
            return Err(selection_error(
                "selecting a composite construction",
                "the construction has no constituent",
            ));
        };
        if constituents.iter().all(|&value| value == first) {
            let source = self.operand(first)?;
            block
                .insts
                .push(MInst::def(dest, ty, MOp::VecSplat, Operands::one(source)));
            return Ok(());
        }
        // A vector of distinct values: the first lane is broadcast and
        // the others are written into their lane, one at a time.
        let lanes = u32::try_from(constituents.len()).unwrap_or(u32::MAX);
        if ty.lane_count() != lanes {
            return Err(selection_error(
                "selecting a composite construction",
                "the construction does not fill the vector",
            ));
        }
        let mut current = self.out.new_vreg(ty.clone());
        let source = self.operand(first)?;
        block
            .insts
            .push(MInst::def(current, ty.clone(), MOp::VecSplat, Operands::one(source)));
        for (lane, &value) in constituents.iter().enumerate().skip(1) {
            let last = lane + 1 == constituents.len();
            let target = if last {
                dest
            } else {
                self.out.new_vreg(ty.clone())
            };
            let scalar = self.operand(value)?;
            let index = u32::try_from(lane).unwrap_or(u32::MAX);
            block.insts.push(MInst::def(
                target,
                ty.clone(),
                MOp::VecInsert { index },
                Operands::two(Operand::Virtual(current), scalar),
            ));
            current = target;
        }
        Ok(())
    }

    /// The machine operation a conversion becomes.
    fn convert_op(&self, op: ConvOp, from: TypeId, to: &MType) -> MOp {
        let from_bits = scalar_bits(self.module, from);
        let to_bits = match to {
            MType::I1 => Some(1),
            MType::I8 => Some(8),
            MType::I16 => Some(16),
            MType::I32 => Some(32),
            MType::I64 => Some(64),
            _ => None,
        };
        match op {
            ConvOp::SConvert => match (from_bits, to_bits) {
                (Some(from), Some(to)) if to > from => MOp::SExt,
                (Some(from), Some(to)) if to < from => MOp::Trunc,
                _ => MOp::Copy,
            },
            ConvOp::UConvert => match (from_bits, to_bits) {
                (Some(from), Some(to)) if to > from => MOp::ZExt,
                (Some(from), Some(to)) if to < from => MOp::Trunc,
                _ => MOp::Copy,
            },
            ConvOp::FConvert => match (from_bits, to_bits) {
                (Some(from), Some(to)) if to > from => MOp::FPExt,
                (Some(from), Some(to)) if to < from => MOp::FPTrunc,
                _ => MOp::Copy,
            },
            ConvOp::ConvertFToS => MOp::FPToSI,
            ConvOp::ConvertFToU => MOp::FPToUI,
            ConvOp::ConvertSToF => MOp::SIToFP,
            ConvOp::ConvertUToF => MOp::UIToFP,
            ConvOp::Bitcast => MOp::Bitcast,
        }
    }

    /// The virtual register an instruction writes.
    fn dest_of(&self, inst: &Inst) -> Result<VReg> {
        let result = inst.result.ok_or_else(|| {
            selection_error(
                "selecting an instruction",
                "the instruction has no result to write",
            )
        })?;
        self.vreg(result)
    }

    /// The machine type an instruction writes.
    fn type_of_dest(&self, inst: &Inst) -> Result<MType> {
        let result = inst.result.ok_or_else(|| {
            selection_error(
                "selecting an instruction",
                "the instruction has no result to write",
            )
        })?;
        machine_type(self.module, self.module.type_of(result))
    }

    /// The virtual register of a value.
    fn vreg(&self, value: ValueId) -> Result<VReg> {
        self.vregs.get(&value).copied().ok_or_else(|| {
            selection_error(
                "selecting a value",
                format!("the value {value} has no virtual register"),
            )
        })
    }

    /// The operand a value becomes.
    fn operand(&self, value: ValueId) -> Result<Operand> {
        if let Some(reg) = self.vregs.get(&value).copied() {
            if self.fused.contains_key(&value) {
                return Err(unsupported(
                    "a work-item identifier as a whole",
                    "the machine code reads the components of the work-item state one by one",
                ));
            }
            return Ok(Operand::Virtual(reg));
        }
        match self.module.value_kind(value) {
            ValueKind::Constant(constant) => Ok(constant_operand(*constant, self.module, value)),
            ValueKind::Global(_) => {
                if self.dyn_elem.contains_key(&value) {
                    return Err(unsupported(
                        "a work-item element outside a load",
                        "the element of a work-item field has no address",
                    ));
                }
                Err(unsupported(
                    "a module-scope variable",
                    "the machine code reads the work-item state and the argument block only",
                ))
            }
            ValueKind::Function(_) => Err(selection_error("selecting a value", "a function is not a value")),
            ValueKind::Argument | ValueKind::Instruction => Err(selection_error(
                "selecting a value",
                format!("the value {value} has no virtual register"),
            )),
        }
    }

    /// Reads one component of a work-item field.
    fn emit_work_item_lane(
        &mut self,
        block: &mut MirBlock,
        source: WorkItemSource,
        lane: u32,
        dest: VReg,
        ty: MType,
    ) -> Result<()> {
        let Some(work_item) = self.work_item else {
            return Err(selection_error(
                "selecting a work-item query",
                "the work-item state is only available in the entry function",
            ));
        };
        if lane > 2 {
            return Err(unsupported(
                "a work-item component above the third",
                "the work-item state holds three components",
            ));
        }
        let natural = natural_type(source);
        match source {
            WorkItemSource::Field(_) | WorkItemSource::EnqueuedLocalSize => {
                let field = match source {
                    WorkItemSource::Field(field) => field,
                    _ => WorkItemField::LocalSize,
                };
                if natural == ty {
                    block.insts.push(MInst::def(
                        dest,
                        ty,
                        MOp::WorkItemLoad { field, lane },
                        Operands::one(Operand::Virtual(work_item)),
                    ));
                    return Ok(());
                }
                let raw = self.out.new_vreg(MType::I32);
                block.insts.push(MInst::def(
                    raw,
                    MType::I32,
                    MOp::WorkItemLoad { field, lane },
                    Operands::one(Operand::Virtual(work_item)),
                ));
                self.convert_into(block, raw, MType::I32, ty, dest)?;
            }
            WorkItemSource::GlobalSize => {
                let groups = self.out.new_vreg(MType::I32);
                block.insts.push(MInst::def(
                    groups,
                    MType::I32,
                    MOp::WorkItemLoad {
                        field: WorkItemField::NumGroups,
                        lane,
                    },
                    Operands::one(Operand::Virtual(work_item)),
                ));
                let local = self.out.new_vreg(MType::I32);
                block.insts.push(MInst::def(
                    local,
                    MType::I32,
                    MOp::WorkItemLoad {
                        field: WorkItemField::LocalSize,
                        lane,
                    },
                    Operands::one(Operand::Virtual(work_item)),
                ));
                let groups = self.widen_to(block, groups, MType::I32)?;
                let local = self.widen_to(block, local, MType::I32)?;
                let product = if natural == ty {
                    dest
                } else {
                    self.out.new_vreg(MType::I64)
                };
                block.insts.push(MInst::def(
                    product,
                    MType::I64,
                    MOp::Mul,
                    Operands::two(Operand::Virtual(groups), Operand::Virtual(local)),
                ));
                if natural != ty {
                    self.convert_into(block, product, MType::I64, ty, dest)?;
                }
            }
            WorkItemSource::GlobalLinearId | WorkItemSource::LocalLinearId => {
                return Err(selection_error(
                    "selecting a work-item query",
                    "a linear identifier has no single component",
                ));
            }
        }
        Ok(())
    }

    /// Reads the component of a work-item field the index names.
    fn emit_work_item_dynamic(
        &mut self,
        block: &mut MirBlock,
        source: WorkItemSource,
        index: Operand,
        dest: VReg,
        ty: MType,
    ) -> Result<()> {
        let natural = natural_type(source);
        let Some(work_item) = self.work_item else {
            return Err(selection_error(
                "selecting a work-item query",
                "the work-item state is only available in the entry function",
            ));
        };
        let mut lanes: Vec<VReg> = Vec::with_capacity(3);
        for lane in 0..3u32 {
            let value = match source {
                WorkItemSource::Field(field) => {
                    let raw = self.out.new_vreg(MType::I32);
                    block.insts.push(MInst::def(
                        raw,
                        MType::I32,
                        MOp::WorkItemLoad { field, lane },
                        Operands::one(Operand::Virtual(work_item)),
                    ));
                    if natural == MType::I32 {
                        raw
                    } else {
                        self.widen_to(block, raw, MType::I32)?
                    }
                }
                WorkItemSource::EnqueuedLocalSize => {
                    let raw = self.out.new_vreg(MType::I32);
                    block.insts.push(MInst::def(
                        raw,
                        MType::I32,
                        MOp::WorkItemLoad {
                            field: WorkItemField::LocalSize,
                            lane,
                        },
                        Operands::one(Operand::Virtual(work_item)),
                    ));
                    if natural == MType::I32 {
                        raw
                    } else {
                        self.widen_to(block, raw, MType::I32)?
                    }
                }
                WorkItemSource::GlobalSize => {
                    let groups = self.out.new_vreg(MType::I32);
                    block.insts.push(MInst::def(
                        groups,
                        MType::I32,
                        MOp::WorkItemLoad {
                            field: WorkItemField::NumGroups,
                            lane,
                        },
                        Operands::one(Operand::Virtual(work_item)),
                    ));
                    let local = self.out.new_vreg(MType::I32);
                    block.insts.push(MInst::def(
                        local,
                        MType::I32,
                        MOp::WorkItemLoad {
                            field: WorkItemField::LocalSize,
                            lane,
                        },
                        Operands::one(Operand::Virtual(work_item)),
                    ));
                    let groups = self.widen_to(block, groups, MType::I32)?;
                    let local = self.widen_to(block, local, MType::I32)?;
                    let product = self.out.new_vreg(MType::I64);
                    block.insts.push(MInst::def(
                        product,
                        MType::I64,
                        MOp::Mul,
                        Operands::two(Operand::Virtual(groups), Operand::Virtual(local)),
                    ));
                    product
                }
                WorkItemSource::GlobalLinearId | WorkItemSource::LocalLinearId => {
                    return Err(selection_error(
                        "selecting a work-item query",
                        "a linear identifier has no single component",
                    ));
                }
            };
            lanes.push(value);
        }
        let first = self.compare_lane(block, index, 0)?;
        let second = self.compare_lane(block, index, 1)?;
        let lower = self.out.new_vreg(natural.clone());
        block.insts.push(MInst::def(
            lower,
            natural.clone(),
            MOp::Select,
            Operands::three(
                Operand::Virtual(first),
                Operand::Virtual(lanes[0]),
                Operand::Virtual(lanes[1]),
            ),
        ));
        let upper = if natural == ty {
            dest
        } else {
            self.out.new_vreg(natural.clone())
        };
        block.insts.push(MInst::def(
            upper,
            natural.clone(),
            MOp::Select,
            Operands::three(
                Operand::Virtual(second),
                Operand::Virtual(lower),
                Operand::Virtual(lanes[2]),
            ),
        ));
        if natural != ty {
            self.convert_into(block, upper, natural, ty, dest)?;
        }
        Ok(())
    }

    /// Compares a component index with a constant.
    fn compare_lane(&mut self, block: &mut MirBlock, index: Operand, lane: u32) -> Result<VReg> {
        let flag = self.out.new_vreg(MType::I1);
        block.insts.push(MInst::def(
            flag,
            MType::I1,
            MOp::ICmp { pred: IPred::Eq },
            Operands::two(index, Operand::Immediate(i64::from(lane))),
        ));
        Ok(flag)
    }

    /// Reads a linear identifier of the work item.
    fn emit_work_item_linear(
        &mut self,
        block: &mut MirBlock,
        source: WorkItemSource,
        dest: VReg,
        ty: MType,
    ) -> Result<()> {
        // The local identifier is the mixed-radix sum of the
        // in-group identifier.
        let local = self.linear_local(block)?;
        match source {
            WorkItemSource::LocalLinearId => {
                self.convert_into(block, local, MType::I64, ty, dest)?;
            }
            WorkItemSource::GlobalLinearId => {
                let group = self.linear_group(block)?;
                let groups = self.load_component(block, WorkItemField::NumGroups, 0)?;
                let size = self.load_component(block, WorkItemField::LocalSize, 0)?;
                let width = self.mul(block, groups, size)?;
                let offset = self.mul(block, group, width)?;
                let total = self.add(block, offset, local)?;
                self.convert_into(block, total, MType::I64, ty, dest)?;
            }
            _ => {
                return Err(selection_error(
                    "selecting a work-item query",
                    "the query is not a linear identifier",
                ));
            }
        }
        Ok(())
    }

    /// The mixed-radix sum of the in-group identifier.
    fn linear_local(&mut self, block: &mut MirBlock) -> Result<VReg> {
        let first = self.load_component(block, WorkItemField::LocalId, 0)?;
        let second = self.load_component(block, WorkItemField::LocalId, 1)?;
        let third = self.load_component(block, WorkItemField::LocalId, 2)?;
        let size_first = self.load_component(block, WorkItemField::LocalSize, 0)?;
        let size_second = self.load_component(block, WorkItemField::LocalSize, 1)?;
        let inner = self.mul(block, size_second, third)?;
        let inner = self.add(block, second, inner)?;
        let middle = self.mul(block, size_first, inner)?;
        let outer = self.add(block, first, middle)?;
        Ok(outer)
    }

    /// The mixed-radix sum of the work-group identifier.
    fn linear_group(&mut self, block: &mut MirBlock) -> Result<VReg> {
        let first = self.load_component(block, WorkItemField::GroupId, 0)?;
        let second = self.load_component(block, WorkItemField::GroupId, 1)?;
        let third = self.load_component(block, WorkItemField::GroupId, 2)?;
        let size_first = self.load_component(block, WorkItemField::LocalSize, 0)?;
        let size_second = self.load_component(block, WorkItemField::LocalSize, 1)?;
        let inner = self.mul(block, size_second, third)?;
        let inner = self.add(block, second, inner)?;
        let middle = self.mul(block, size_first, inner)?;
        let outer = self.add(block, first, middle)?;
        Ok(outer)
    }

    /// Reads one component of a work-item field as a wide value.
    fn load_component(&mut self, block: &mut MirBlock, field: WorkItemField, lane: u32) -> Result<VReg> {
        let Some(work_item) = self.work_item else {
            return Err(selection_error(
                "selecting a work-item query",
                "the work-item state is only available in the entry function",
            ));
        };
        let raw = self.out.new_vreg(MType::I32);
        block.insts.push(MInst::def(
            raw,
            MType::I32,
            MOp::WorkItemLoad { field, lane },
            Operands::one(Operand::Virtual(work_item)),
        ));
        self.widen_to(block, raw, MType::I32)
    }

    /// Adds two values.
    fn add(&mut self, block: &mut MirBlock, lhs: VReg, rhs: VReg) -> Result<VReg> {
        let dest = self.out.new_vreg(MType::I64);
        block.insts.push(MInst::def(
            dest,
            MType::I64,
            MOp::Add,
            Operands::two(Operand::Virtual(lhs), Operand::Virtual(rhs)),
        ));
        Ok(dest)
    }

    /// Multiplies two values.
    fn mul(&mut self, block: &mut MirBlock, lhs: VReg, rhs: VReg) -> Result<VReg> {
        let dest = self.out.new_vreg(MType::I64);
        block.insts.push(MInst::def(
            dest,
            MType::I64,
            MOp::Mul,
            Operands::two(Operand::Virtual(lhs), Operand::Virtual(rhs)),
        ));
        Ok(dest)
    }

    /// Widens a value, returning the new register.
    fn widen_to(&mut self, block: &mut MirBlock, source: VReg, from: MType) -> Result<VReg> {
        let dest = self.out.new_vreg(MType::I64);
        block.insts.push(MInst::def(
            dest,
            MType::I64,
            conversion(&from, &MType::I64),
            Operands::one(Operand::Virtual(source)),
        ));
        Ok(dest)
    }

    /// Converts a value, writing the result into `dest`.
    fn convert_into(
        &mut self,
        block: &mut MirBlock,
        source: VReg,
        from: MType,
        to: MType,
        dest: VReg,
    ) -> Result<()> {
        let op = conversion(&from, &to);
        block
            .insts
            .push(MInst::def(dest, to, op, Operands::one(Operand::Virtual(source))));
        Ok(())
    }
}
/// The operand a constant becomes.
fn constant_operand(constant: ConstValue, module: &Module, value: ValueId) -> Operand {
    match constant {
        ConstValue::Int(raw) => {
            let ty = module.type_of(value);
            let bits = scalar_bits(module, ty).unwrap_or(64).min(64);
            let signed = matches!(module.ty(ty), Type::Int { signed: true, .. });
            if signed && bits < 64 {
                let shift = 64 - bits;
                Operand::Immediate(((raw << shift) as i64) >> shift)
            } else {
                Operand::Immediate(raw as i64)
            }
        }
        ConstValue::Bool(value) => Operand::Immediate(i64::from(value)),
        ConstValue::Float32(bits) => Operand::FloatImmediate(u64::from(bits)),
        ConstValue::Float64(bits) => Operand::FloatImmediate(bits),
        ConstValue::Null | ConstValue::Undef => Operand::Immediate(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::ArgKind;
    use codevar_ocl_ir::ir::{Linkage, Target};
    use codevar_ocl_ir::lower::lower_source;

    /// Lowers `source`, failing the test when the source is rejected.
    fn lower(source: &str) -> Module {
        lower_source(source).expect("the source lowers")
    }

    /// Lowers and selects `source`, whose entry is its first entry point.
    fn select(source: &str) -> Selection {
        let module = lower(source);
        let entry = module
            .entry_points
            .first()
            .expect("the module has an entry point")
            .func;
        select_module(&module, entry).expect("the module selects")
    }

    /// Every instruction of the entry function.
    fn insts(selection: &Selection) -> Vec<&MInst> {
        selection
            .module
            .functions
            .first()
            .expect("the module has a function")
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .collect()
    }

    #[test]
    fn the_entry_function_takes_the_work_item_pointer() {
        let selection = select("#[kernel]\nfn k(out: *mut int) {\n    *out = 1;\n}\n");
        let entry = &selection.module.functions[0];
        assert!(entry.is_entry, "the first function is the entry point");
        assert_eq!(entry.params.len(), 1, "the entry has one machine parameter");
        let first = &entry.blocks[0].insts[0];
        assert!(
            matches!(first.op, MOp::Param { index: 0 }),
            "the first instruction reads the parameter"
        );
        assert_eq!(first.dest, Some(Destination::Virtual(entry.params[0])));
        assert_eq!(
            selection
                .module
                .symbol(entry.symbol)
                .map(|symbol| symbol.name.as_str()),
            Some("k")
        );
    }

    #[test]
    fn a_kernel_argument_becomes_an_argument_load() {
        let selection = select("#[kernel]\nfn k(out: *mut int) {\n    *out = 1;\n}\n");
        let entry = &selection.module.functions[0];
        let loads: Vec<&MInst> = entry.blocks[0]
            .insts
            .iter()
            .filter(|inst| matches!(inst.op, MOp::ArgLoad { .. }))
            .collect();
        assert_eq!(loads.len(), 1, "the kernel has one argument");
        let MOp::ArgLoad { index, kind } = loads[0].op else {
            unreachable!("the instruction is an argument load");
        };
        assert_eq!(index, 0);
        assert_eq!(kind, ArgKind::Pointer);
        let base = entry.blocks[0]
            .insts
            .iter()
            .find(|inst| matches!(inst.op, MOp::ArgBlockBase))
            .expect("the argument block has a base");
        assert_eq!(
            loads[0].args.get(0),
            base.dest.and_then(Destination::virtual_reg).map(Operand::Virtual)
        );
    }

    #[test]
    fn a_scalar_argument_carries_its_size() {
        let selection = select("#[kernel]\nfn k(n: int) {\n    let x = n;\n}\n");
        let loads: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::ArgLoad { .. }))
            .collect();
        assert_eq!(loads.len(), 1);
        let MOp::ArgLoad { index, kind } = loads[0].op else {
            unreachable!("the instruction is an argument load");
        };
        assert_eq!(index, 0);
        assert_eq!(kind, ArgKind::Scalar { size: 4 });
    }

    #[test]
    fn a_work_item_query_becomes_a_work_item_load() {
        let selection =
            select("#[kernel]\nfn k(out: *mut int) {\n    let i = get_global_id(0);\n    out[0] = i;\n}\n");
        let loads: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::WorkItemLoad { .. }))
            .collect();
        assert_eq!(loads.len(), 1, "the query reads one component");
        let MOp::WorkItemLoad { field, lane } = loads[0].op else {
            unreachable!("the instruction reads the work item");
        };
        assert_eq!(field, WorkItemField::GlobalId);
        assert_eq!(lane, 0);
        // The read takes the work-item pointer.
        let entry = &selection.module.functions[0];
        assert_eq!(loads[0].args.get(0), Some(Operand::Virtual(entry.params[0])));
    }

    #[test]
    fn a_computed_work_item_query_selects_the_component() {
        let selection = select(
            "#[kernel]\nfn k(n: int, out: *mut int) {\n    let i = get_global_id(n);\n    *out = 1;\n}\n",
        );
        let reads: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::WorkItemLoad { .. }))
            .collect();
        assert_eq!(reads.len(), 3, "every component is read");
        let selects: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::Select))
            .collect();
        assert_eq!(selects.len(), 2, "the component is selected twice");
    }

    #[test]
    fn the_global_size_is_the_group_count_times_the_local_size() {
        let selection =
            select("#[kernel]\nfn k(out: *mut long) {\n    let s = get_global_size(0);\n    out[0] = s;\n}\n");
        let reads: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::WorkItemLoad { .. }))
            .collect();
        assert_eq!(reads.len(), 2, "the group count and the local size are read");
        let fields: Vec<WorkItemField> = reads
            .iter()
            .map(|inst| match inst.op {
                MOp::WorkItemLoad { field, .. } => field,
                _ => unreachable!("the instruction reads the work item"),
            })
            .collect();
        assert!(fields.contains(&WorkItemField::NumGroups));
        assert!(fields.contains(&WorkItemField::LocalSize));
        assert!(
            insts(&selection)
                .into_iter()
                .any(|inst| matches!(inst.op, MOp::Mul)),
            "the two sizes are multiplied"
        );
    }

    #[test]
    fn a_linear_identifier_is_the_mixed_radix_sum() {
        let selection = select(
            "#[kernel]\nfn k(out: *mut int) {\n    let s = get_global_linear_id(0);\n    *out = 1;\n}\n",
        );
        let reads: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::WorkItemLoad { .. }))
            .collect();
        assert!(reads.len() >= 6, "the identifiers and the sizes are read");
        let fields: Vec<WorkItemField> = reads
            .iter()
            .map(|inst| match inst.op {
                MOp::WorkItemLoad { field, .. } => field,
                _ => unreachable!("the instruction reads the work item"),
            })
            .collect();
        assert!(!fields.contains(&WorkItemField::GlobalId));
        assert!(fields.contains(&WorkItemField::LocalId));
        assert!(fields.contains(&WorkItemField::GroupId));
        assert!(fields.contains(&WorkItemField::LocalSize));
        assert!(
            insts(&selection)
                .into_iter()
                .any(|inst| matches!(inst.op, MOp::Mul)),
            "the components are multiplied"
        );
        assert!(
            insts(&selection)
                .into_iter()
                .any(|inst| matches!(inst.op, MOp::Add)),
            "the components are added"
        );
    }

    #[test]
    fn a_call_names_the_callee_symbol() {
        let selection = select(
            "#[kernel]\nfn k(out: *mut int) {\n    *out = helper(1);\n}\nfn helper(x: int) -> int {\n    x + 1\n}\n",
        );
        assert_eq!(selection.module.functions.len(), 2, "the callee is selected too");
        let calls: Vec<&MInst> = insts(&selection)
            .into_iter()
            .filter(|inst| matches!(inst.op, MOp::Call { .. }))
            .collect();
        assert_eq!(calls.len(), 1);
        let MOp::Call { callee } = calls[0].op else {
            unreachable!("the instruction is a call");
        };
        let symbol = selection
            .module
            .symbol(callee)
            .expect("the callee has a symbol");
        assert_eq!(symbol.name, "helper");
        assert_eq!(
            selection
                .module
                .function(callee)
                .map(|function| function.name.as_str()),
            Some("helper")
        );
        // The helper keeps its own parameters.
        let helper = &selection.module.functions[1];
        assert!(!helper.is_entry);
        assert_eq!(helper.params.len(), 1);
        assert!(matches!(helper.blocks[0].insts[0].op, MOp::Param { index: 0 }));
    }

    #[test]
    fn a_barrier_is_refused() {
        let module = lower("#[kernel]\nfn k(out: *mut int) {\n    barrier(0);\n    *out = 1;\n}\n");
        let entry = module
            .entry_points
            .first()
            .expect("an entry point")
            .func;
        let error = select_module(&module, entry).expect_err("the barrier is refused");
        assert!(
            matches!(&error, CodegenError::Unsupported { what, .. } if *what == "a control barrier"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_memory_barrier_is_refused() {
        let module = lower("#[kernel]\nfn k(out: *mut int) {\n    mem_fence(0);\n    *out = 1;\n}\n");
        let entry = module
            .entry_points
            .first()
            .expect("an entry point")
            .func;
        let error = select_module(&module, entry).expect_err("the barrier is refused");
        assert!(
            matches!(&error, CodegenError::Unsupported { what, .. } if *what == "a memory barrier"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn an_extended_instruction_is_refused() {
        let module = lower("#[kernel]\nfn k(out: *mut float) {\n    let x = sqrt(2.0);\n    *out = x;\n}\n");
        let entry = module
            .entry_points
            .first()
            .expect("an entry point")
            .func;
        let error = select_module(&module, entry).expect_err("the built-in is refused");
        match &error {
            CodegenError::Unsupported { what, detail } => {
                assert_eq!(*what, "an extended instruction");
                assert!(detail.contains("sqrt"), "unexpected detail: {detail}");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn a_stack_slot_becomes_a_frame_address() {
        let selection = select(
            "#[kernel]\nfn k(n: int, out: *mut int) {\n    let mut x = n;\n    x = x + 1;\n    *out = x;\n}\n",
        );
        let entry = &selection.module.functions[0];
        let addresses: Vec<&MInst> = entry.blocks[0]
            .insts
            .iter()
            .filter(|inst| matches!(inst.op, MOp::FrameAddr { .. }))
            .collect();
        assert_eq!(addresses.len(), 1, "the function has one slot");
        let MOp::FrameAddr { offset } = addresses[0].op else {
            unreachable!("the instruction is a frame address");
        };
        assert_eq!(offset, 0, "the first slot sits at the bottom of the frame");
        assert!(entry.fixed_frame_size >= 4, "the frame holds the slot");
        assert_eq!(entry.frame_size, entry.fixed_frame_size);
    }

    #[test]
    fn the_types_of_the_values_are_recorded() {
        let selection = select(
            "#[kernel]\nfn k(n: int, out: *mut int) {\n    let mut x = n;\n    x = x + 1;\n    *out = x;\n}\n",
        );
        let entry = &selection.module.functions[0];
        assert!(
            entry.vreg_types.contains(&MType::I32),
            "the machine module records the type of every value"
        );
        assert!(
            entry
                .vreg_types
                .iter()
                .any(|ty| matches!(ty, MType::Ptr { .. })),
            "the argument is a pointer"
        );
    }

    /// Builds a module with a merge block and a phi in it.
    fn phi_module() -> (Module, ValueId) {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let boolean = module.bool_ty();
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::new());
        let func = module
            .add_function("k", sig, Linkage::External)
            .expect("a fresh name");
        module.begin_body(func).expect("a body");
        let entry = module.push_block(func, "entry").expect("a block");
        let left = module.push_block(func, "left").expect("a block");
        let right = module.push_block(func, "right").expect("a block");
        let merge = module.push_block(func, "merge").expect("a block");

        let condition = module
            .intern_const(boolean, ConstValue::Bool(true))
            .expect("a constant");
        let one = module
            .intern_const(int, ConstValue::Int(1))
            .expect("a constant");
        let two = module
            .intern_const(int, ConstValue::Int(2))
            .expect("a constant");

        let first = module.new_inst_value(int);
        let second = module.new_inst_value(int);
        let result = module.new_inst_value(int);

        module
            .emit(
                func,
                entry,
                Inst::none(Op::BranchConditional {
                    cond: condition,
                    then: left,
                    other: right,
                }),
            )
            .expect("the branch");
        module
            .emit(func, left, Inst::def(first, int, Op::CopyObject { operand: one }))
            .expect("the copy");
        module
            .emit(func, left, Inst::none(Op::Branch { target: merge }))
            .expect("the branch");
        module
            .emit(
                func,
                right,
                Inst::def(second, int, Op::CopyObject { operand: two }),
            )
            .expect("the copy");
        module
            .emit(func, right, Inst::none(Op::Branch { target: merge }))
            .expect("the branch");
        module
            .emit(
                func,
                merge,
                Inst::def(
                    result,
                    int,
                    Op::Phi {
                        incomings: vec![(first, left), (second, right)],
                    },
                ),
            )
            .expect("the phi");
        module
            .emit(func, merge, Inst::none(Op::Return))
            .expect("the return");
        (module, func)
    }

    #[test]
    fn a_phi_is_lowered_to_copies_in_the_predecessors() {
        let (module, func) = phi_module();
        codevar_ocl_ir::verify::verify(&module).expect("the module verifies");
        let selection = select_module(&module, func).expect("the module selects");
        let function = &selection.module.functions[0];
        assert_eq!(function.blocks.len(), 4);
        // The merge block defines the result once.
        let merge = &function.blocks[3];
        let definition = merge
            .insts
            .iter()
            .find(|inst| matches!(inst.op, MOp::Copy))
            .expect("the merge block defines the result");
        // The predecessors copy their incoming value into it.
        for index in [1usize, 2] {
            let copies: Vec<&MInst> = function.blocks[index]
                .insts
                .iter()
                .filter(|inst| matches!(inst.op, MOp::Copy))
                .collect();
            assert!(
                copies
                    .iter()
                    .any(|copy| copy.dest == definition.dest),
                "the block copies the incoming value into the result"
            );
        }
    }
}
