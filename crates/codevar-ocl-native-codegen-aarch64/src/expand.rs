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

//! The expansion of the machine code into AArch64 instructions.
//!
//! Every operation of the machine code names what it computes, not
//! how it is encoded, so the back end has to pick the instructions
//! that compute it.  This module does that: it turns the
//! instructions of a block into the words of the architecture, and
//! it leaves behind the places where a branch or a call has to be
//! patched once the blocks are laid out.
//!
//! # The frame
//!
//! The prologue of a function saves the callee-saved registers it
//! uses and then reserves the frame of the function with a single
//! subtraction.  The frame base is the value of the stack pointer
//! right after that subtraction, and the saved registers sit above
//! the frame, at `frame_base + frame_size + 8 * index`.
//!
//! # The calls
//!
//! The arguments of a call go into the argument registers, the
//! integer ones counting from `x0` and the floating-point ones from
//! `v0`.  The result comes back in `x0`, or in `v0` when the result
//! is a floating-point value.
//!
//! # The registers
//!
//! The machine code that reaches the back end has already been
//! allocated: every operand and every destination names a physical
//! register of the target's file.  The back end reads those
//! registers as they are, and refuses a machine code that still
//! names a virtual register, because such a machine code never went
//! through the allocator.
//!
//! The machine code names the type of the values every instruction
//! reads, so the back end never has to guess the width of a value
//! from the register that holds it.  The types are what tell a
//! thirty-two-bit comparison from a sixty-four-bit one and a
//! single-precision value from a double-precision one.

use alloc::vec::Vec;

use codevar_ocl_abi::{ARG_BASE_OFFSET, ARG_SLOT, WorkItemField};
use codevar_ocl_native_codegen::error::{CodegenError, Result};
use codevar_ocl_native_codegen::mir::{
    ArgKind, FPred, IPred, MInst, MOp, MType, MirFunction, MirModule, Operand, PReg, SymId,
};

use crate::encode::{Cond, Direction, Encoder, Width};
use crate::reg;

/// A place an operand of the machine code lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Loc {
    /// A general-purpose register.
    Gpr(u8),
    /// A SIMD-and-floating-point register.
    Simd(u8),
    /// An integer constant.
    Imm(i64),
    /// The bits of a floating-point constant.
    FloatImm(u64),
}

/// A place in the code that has to be patched once the blocks are
/// laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fixup {
    /// A branch to a block of the same function.
    Branch {
        /// The offset of the instruction inside the function.
        at: u32,
        /// The block the branch goes to.
        block: u32,
        /// Whether the branch has the wide displacement.
        wide: bool,
    },
    /// A call to another function of the module.
    Call {
        /// The offset of the instruction inside the function.
        at: u32,
        /// The index of the function that is called.
        function: usize,
    },
}

/// The code of one function, before the branches are patched.
#[derive(Debug, Clone, Default)]
pub struct FunctionCode {
    /// The words of the function.
    pub bytes: Vec<u8>,
    /// The places that have to be patched.
    pub fixups: Vec<Fixup>,
    /// The offset of every block inside the function.
    pub blocks: Vec<u32>,
    /// The number of instructions of the function.
    pub instructions: u32,
}

/// The expansion of one function.
struct Expander<'a> {
    /// The module the function belongs to.
    module: &'a MirModule,
    /// The function itself.
    function: &'a MirFunction,
    /// The words of the function.
    bytes: Vec<u8>,
    /// The places that have to be patched.
    fixups: Vec<Fixup>,
    /// The offset of every block inside the function.
    blocks: Vec<u32>,
    /// The registers the prologue saves, in the order they are
    /// saved.
    saved: Vec<u8>,
    /// The size of the frame of the function.
    frame_size: u32,
    /// The size of the area the saved registers live in.
    save_area: u32,
    /// Whether the frame has been released in the current block.
    released: bool,
    /// Whether the machine code asked for the frame to be set up.
    frame_setup: bool,
}

/// Turns the machine code of a function into the words of the
/// architecture.
///
/// # Errors
///
/// Returns an error when the function uses something the
/// architecture cannot express.
pub fn expand(module: &MirModule, function: &MirFunction) -> Result<FunctionCode> {
    let mut expander = Expander {
        module,
        function,
        bytes: Vec::new(),
        fixups: Vec::new(),
        blocks: Vec::new(),
        saved: Vec::new(),
        frame_size: 0,
        save_area: 0,
        released: false,
        frame_setup: false,
    };
    expander.run()?;
    let instructions = (expander.bytes.len() / 4) as u32;
    Ok(FunctionCode {
        bytes: expander.bytes,
        fixups: expander.fixups,
        blocks: expander.blocks,
        instructions,
    })
}

impl<'a> Expander<'a> {
    /// Expands the whole function.
    fn run(&mut self) -> Result<()> {
        self.collect_saved();
        self.frame_size = align_up(self.function.frame_size, 16);
        self.save_area = align_up(8 * self.saved.len() as u32, 16);

        self.prologue()?;

        for (index, block) in self.function.blocks.iter().enumerate() {
            self.blocks.push(self.bytes.len() as u32);
            self.released = false;
            for (offset, inst) in block.insts.iter().enumerate() {
                if offset + 1 == block.insts.len() {
                    self.expand_terminator(inst)?;
                } else {
                    self.expand_inst(inst)?;
                }
            }
            let _ = index;
        }
        Ok(())
    }

    /// Finds the registers the prologue has to save.
    ///
    /// A register that survives a call has to be saved, so the
    /// prologue saves every one of them that the function reads or
    /// writes.  The link register is saved as well when the function
    /// calls another one.
    fn collect_saved(&mut self) {
        let mut saved: Vec<u8> = Vec::new();
        let keep = |index: u8, saved: &mut Vec<u8>| {
            if (19..=28).contains(&index) && !saved.contains(&index) {
                saved.push(index);
            }
        };
        for block in &self.function.blocks {
            for inst in &block.insts {
                if let Some(dest) = inst.dest_reg()
                    && let Some(index) = reg::gpr_index(dest)
                {
                    keep(index, &mut saved);
                }
                for operand in inst.args.as_slice() {
                    if let Operand::Physical(reg) = operand
                        && let Some(index) = reg::gpr_index(*reg)
                    {
                        keep(index, &mut saved);
                    }
                }
            }
        }
        saved.sort_unstable();
        if self.has_call() {
            saved.push(30);
        }
        self.saved = saved;
    }

    /// Whether the function calls another function.
    fn has_call(&self) -> bool {
        self.function.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(inst.op, MOp::Call { .. }))
        })
    }

    /// The encoder of the function.
    fn enc(&mut self) -> Encoder<'_> {
        Encoder::new(&mut self.bytes)
    }

    /// Emits the prologue of the function.
    fn prologue(&mut self) -> Result<()> {
        if !self.saved.is_empty() {
            self.sub_stack(self.save_area)?;
            let mut index = 0usize;
            while index + 1 < self.saved.len() {
                let first = self.saved[index];
                let second = self.saved[index + 1];
                let offset = (index * 8) as i64;
                self.enc()
                    .pair(Direction::Store, first, second, reg::SP, offset, 8)?;
                index += 2;
            }
            if index < self.saved.len() {
                let register = self.saved[index];
                let offset = (index * 8) as i64;
                self.enc().store(register, reg::SP, offset, 8)?;
            }
        }
        self.sub_stack(self.frame_size)?;
        Ok(())
    }

    /// Emits the epilogue of the function.
    fn epilogue(&mut self) -> Result<()> {
        if !self.saved.is_empty() {
            let last = (self.saved.len() - 1) * 8;
            if self.frame_size as usize + last <= 248 {
                let mut index = 0usize;
                while index + 1 < self.saved.len() {
                    let first = self.saved[index];
                    let second = self.saved[index + 1];
                    let offset = (self.frame_size as usize + index * 8) as i64;
                    self.enc()
                        .pair(Direction::Load, first, second, reg::SP, offset, 8)?;
                    index += 2;
                }
                if index < self.saved.len() {
                    let register = self.saved[index];
                    let offset = (self.frame_size as usize + index * 8) as i64;
                    self.enc().load(register, reg::SP, offset, 8)?;
                }
            } else {
                self.add_stack(reg::SCRATCH_A, self.frame_size)?;
                let mut index = 0usize;
                while index + 1 < self.saved.len() {
                    let first = self.saved[index];
                    let second = self.saved[index + 1];
                    let offset = (index * 8) as i64;
                    self.enc()
                        .pair(Direction::Load, first, second, reg::SCRATCH_A, offset, 8)?;
                    index += 2;
                }
                if index < self.saved.len() {
                    let register = self.saved[index];
                    let offset = (index * 8) as i64;
                    self.enc()
                        .load(register, reg::SCRATCH_A, offset, 8)?;
                }
            }
        }
        self.add_stack(reg::SP, self.frame_size + self.save_area)?;
        Ok(())
    }

    /// Subtracts an amount from the stack pointer.
    fn sub_stack(&mut self, amount: u32) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        if amount < 4096 {
            self.enc()
                .sub_imm(reg::SP, reg::SP, amount as u16, Width::Bits64);
        } else if amount.is_multiple_of(4096) {
            self.enc()
                .sub_imm_scaled(reg::SP, reg::SP, (amount / 4096) as u16, Width::Bits64);
        } else {
            return Err(CodegenError::Target {
                what: "reserving the frame",
                detail: alloc::format!("the frame of {amount} bytes has no encoding"),
            });
        }
        Ok(())
    }

    /// Adds an amount to a register that holds the stack pointer.
    fn add_stack(&mut self, target: u8, amount: u32) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        if amount < 4096 {
            self.enc()
                .add_imm(target, reg::SP, amount as u16, Width::Bits64);
        } else if amount.is_multiple_of(4096) {
            self.enc()
                .add_imm_scaled(target, reg::SP, (amount / 4096) as u16, Width::Bits64);
        } else {
            return Err(CodegenError::Target {
                what: "releasing the frame",
                detail: alloc::format!("the frame of {amount} bytes has no encoding"),
            });
        }
        Ok(())
    }

    /// The place an operand lives in.
    ///
    /// The machine code has already been allocated, so an operand
    /// names a physical register; a virtual one means the machine
    /// code never went through the allocator.
    fn loc(&self, operand: Operand) -> Result<Loc> {
        match operand {
            Operand::Physical(preg) => Ok(locate(preg)),
            Operand::Virtual(vreg) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: alloc::format!(
                    "the machine code was not allocated: the operand names the virtual register {}",
                    vreg.index()
                ),
            }),
            Operand::Immediate(value) => Ok(Loc::Imm(value)),
            Operand::FloatImmediate(bits) => Ok(Loc::FloatImm(bits)),
            Operand::Symbol(symbol) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: alloc::format!("the symbol {} is not a value", symbol.index()),
            }),
        }
    }

    /// The type of the values an instruction reads.
    ///
    /// The machine code names the type of the operands of every
    /// instruction, so the back end never has to guess it from the
    /// register that happens to hold the value.
    fn operand_type(inst: &MInst) -> Option<&MType> {
        inst.operand_ty.as_ref()
    }

    /// The type of the argument at `index` of a call to `callee`.
    ///
    /// The arguments of a call have types of their own, so the
    /// instruction does not name them; the callee's own parameters
    /// do.
    fn argument_type(&self, callee: SymId, index: usize) -> Option<&MType> {
        let function = self.module.function(callee)?;
        let param = function.params.get(index)?;
        function.vreg_types.get(param.index())
    }

    /// The SIMD register a floating-point operand holds.
    ///
    /// A constant is moved into one of the scratch registers first, so
    /// that the operation has a register to read.
    fn float_operand(&mut self, inst: &MInst, index: usize) -> Result<u8> {
        let operand = self.arg(inst, index)?;
        match self.loc(operand)? {
            Loc::Simd(reg) => Ok(reg),
            Loc::FloatImm(bits) => {
                let single = matches!(inst.ty, MType::F32);
                let scratch = if index == 0 {
                    reg::SIMD_SCRATCH
                } else {
                    reg::SIMD_SCRATCH_B
                };
                self.materialize_float(scratch, bits, single);
                Ok(scratch)
            }
            _ => Err(CodegenError::Target {
                what: "reading a floating-point operand",
                detail: alloc::string::String::from("the value is not a floating-point value"),
            }),
        }
    }

    /// The general-purpose register an integer operand holds.
    ///
    /// A constant is moved into the scratch register first, so that
    /// the operation has a register to read.
    fn int_operand(&mut self, inst: &MInst, index: usize) -> Result<u8> {
        let operand = self.arg(inst, index)?;
        match self.loc(operand)? {
            Loc::Gpr(reg) => Ok(reg),
            Loc::Imm(number) => {
                self.materialize(reg::SCRATCH_A, number as u64, Width::Bits64);
                Ok(reg::SCRATCH_A)
            }
            _ => Err(CodegenError::Target {
                what: "reading an integer operand",
                detail: alloc::string::String::from("the value is not an integer"),
            }),
        }
    }

    /// The register an instruction writes.
    fn dest(&self, inst: &MInst) -> Result<Option<Loc>> {
        match inst.dest_reg() {
            Some(preg) => Ok(Some(locate(preg))),
            None if inst.dest_vreg().is_some() => Err(CodegenError::Target {
                what: "writing a value",
                detail: alloc::string::String::from(
                    "the machine code was not allocated: the destination is still a virtual register",
                ),
            }),
            None => Ok(None),
        }
    }

    /// The general-purpose register an instruction writes.
    fn dest_gpr(&self, inst: &MInst) -> Result<u8> {
        match self.dest(inst)? {
            Some(Loc::Gpr(reg)) => Ok(reg),
            _ => Err(CodegenError::Target {
                what: "writing a value",
                detail: alloc::string::String::from("the value is not an integer"),
            }),
        }
    }

    /// The SIMD register an instruction writes.
    fn dest_simd(&self, inst: &MInst) -> Result<u8> {
        match self.dest(inst)? {
            Some(Loc::Simd(reg)) => Ok(reg),
            _ => Err(CodegenError::Target {
                what: "writing a value",
                detail: alloc::string::String::from("the value is not a vector or a floating-point value"),
            }),
        }
    }

    /// The width of an integer operation.
    fn int_width(&self, inst: &MInst) -> Width {
        if let Some(size) = inst.ty.size_in_bytes() {
            return Width::of_type(Some(size));
        }
        Width::Bits32
    }

    /// The width of the values an operation compares.
    fn compare_width(&self, inst: &MInst) -> Width {
        match Self::operand_type(inst).and_then(MType::size_in_bytes) {
            Some(size) => Width::of_type(Some(size)),
            None => Width::Bits32,
        }
    }

    /// Puts a constant into a register.
    fn materialize(&mut self, target: u8, value: u64, width: Width) {
        let bits = width.bits();
        let mut shift = 0u32;
        loop {
            let piece = ((value >> shift) & 0xFFFF) as u16;
            if shift == 0 {
                self.enc().movz(target, piece, 0, width);
            } else {
                self.enc().movk(target, piece, shift / 16, width);
            }
            shift += 16;
            if shift >= bits {
                break;
            }
        }
    }

    /// Puts a constant into the scratch register and returns it.
    fn scratch_imm(&mut self, value: u64, width: Width) -> u8 {
        self.materialize(reg::SCRATCH_A, value, width);
        reg::SCRATCH_A
    }

    /// Puts the bits of a floating-point constant into a register.
    fn materialize_float(&mut self, target: u8, bits: u64, single: bool) {
        self.materialize(reg::SCRATCH_A, bits, Width::Bits64);
        self.enc()
            .fmov_from_gp(target, reg::SCRATCH_A, single);
    }

    /// Expands an instruction that is not a terminator.
    fn expand_inst(&mut self, inst: &MInst) -> Result<()> {
        match &inst.op {
            MOp::Copy => self.expand_copy(inst),
            MOp::Load { size } => self.expand_load(inst, *size),
            MOp::Store { size } => self.expand_store(inst, *size),
            MOp::Gep { size, signed } => self.expand_gep(inst, *size, *signed),
            MOp::VecSplat => self.expand_splat(inst),
            MOp::VecExtract { index } => self.expand_extract(inst, *index),
            MOp::VecInsert { index } => self.expand_insert(inst, *index),
            MOp::VecShuffle { mask } => self.expand_shuffle(inst, mask),
            MOp::Add => self.expand_add(inst),
            MOp::Sub => self.expand_sub(inst),
            MOp::Mul => self.expand_mul(inst),
            MOp::UDiv => self.expand_div(inst, false),
            MOp::SDiv => self.expand_div(inst, true),
            MOp::URem => self.expand_rem(inst, false),
            MOp::SRem => self.expand_rem(inst, true),
            MOp::And => self.expand_logic(inst, Logic::And),
            MOp::Or => self.expand_logic(inst, Logic::Or),
            MOp::Xor => self.expand_logic(inst, Logic::Xor),
            MOp::Shl => self.expand_shift(inst, Shift::Left),
            MOp::LShr => self.expand_shift(inst, Shift::Logical),
            MOp::AShr => self.expand_shift(inst, Shift::Arithmetic),
            MOp::Neg => self.expand_neg(inst),
            MOp::Not => self.expand_not(inst),
            MOp::FNeg => self.expand_fneg(inst),
            MOp::FAdd => self.expand_float(inst, Float::Add),
            MOp::FSub => self.expand_float(inst, Float::Sub),
            MOp::FMul => self.expand_float(inst, Float::Mul),
            MOp::FDiv => self.expand_float(inst, Float::Div),
            MOp::FRem => self.expand_frem(inst),
            MOp::FSqrt => self.expand_fsqrt(inst),
            MOp::FAbs => self.expand_fabs(inst),
            MOp::FMin => self.expand_fmin(inst, true),
            MOp::FMax => self.expand_fmin(inst, false),
            MOp::FDim => self.expand_fdim(inst),
            MOp::FCopySign => self.expand_fcopysign(inst),
            MOp::ICmp { pred } => self.expand_icmp(inst, *pred),
            MOp::FCmp { pred } => self.expand_fcmp(inst, *pred),
            MOp::Select => self.expand_select(inst),
            MOp::ZExt => self.expand_zext(inst),
            MOp::SExt => self.expand_sext(inst),
            MOp::Trunc => self.expand_trunc(inst),
            MOp::FPToUI => self.expand_fp_to_int(inst, false),
            MOp::FPToSI => self.expand_fp_to_int(inst, true),
            MOp::UIToFP => self.expand_int_to_fp(inst, false),
            MOp::SIToFP => self.expand_int_to_fp(inst, true),
            MOp::FPTrunc => self.expand_fp_trunc(inst),
            MOp::FPExt => self.expand_fp_ext(inst),
            MOp::Bitcast => self.expand_bitcast(inst),
            MOp::Call { callee } => self.expand_call(inst, *callee),
            MOp::WorkItemLoad { field, lane } => self.expand_work_item(inst, *field, *lane),
            MOp::ArgBlockBase => self.expand_arg_base(inst),
            MOp::ArgLoad { index, kind } => self.expand_arg_load(inst, *index, *kind),
            MOp::Param { index } => self.expand_param(inst, *index),
            MOp::FrameAddr { offset } => self.expand_frame_addr(inst, *offset),
            MOp::SpillStore { offset, size } => self.expand_spill_store(inst, *offset, *size),
            MOp::SpillLoad { offset, size } => self.expand_spill_load(inst, *offset, *size),
            MOp::FrameSetup => {
                // The prologue of the function reserves the frame
                // already, so the operation that asks for it is
                // only a marker.
                if self.frame_setup {
                    return Err(CodegenError::Target {
                        what: "reserving the frame",
                        detail: alloc::string::String::from("the frame is set up twice"),
                    });
                }
                self.frame_setup = true;
                Ok(())
            }
            MOp::FrameTeardown => {
                self.epilogue()?;
                self.released = true;
                Ok(())
            }
            MOp::Br { .. }
            | MOp::CondBr
            | MOp::CondBrCmp { .. }
            | MOp::Ret
            | MOp::RetVoid
            | MOp::Unreachable => Err(CodegenError::Target {
                what: "expanding an instruction",
                detail: alloc::string::String::from("the operation belongs to the end of a block"),
            }),
        }
    }

    /// Expands the terminator of a block.
    fn expand_terminator(&mut self, inst: &MInst) -> Result<()> {
        match &inst.op {
            MOp::Br { target } => {
                self.branch(*target);
                Ok(())
            }
            MOp::CondBr => {
                let condition = self.loc(self.arg(inst, 0)?)?;
                let Loc::Gpr(cond) = condition else {
                    return Err(CodegenError::Target {
                        what: "branching on a condition",
                        detail: alloc::string::String::from("the condition is not an integer"),
                    });
                };
                let taken = self.arg(inst, 1)?;
                let missed = self.arg(inst, 2)?;
                let (Loc::Imm(taken), Loc::Imm(missed)) = (self.loc(taken)?, self.loc(missed)?) else {
                    return Err(CodegenError::Target {
                        what: "branching on a condition",
                        detail: alloc::string::String::from("the targets of the branch are not blocks"),
                    });
                };
                let at = self.bytes.len() as u32;
                self.enc().cbnz(cond, Width::Bits32);
                self.fixups.push(Fixup::Branch {
                    at,
                    block: taken as u32,
                    wide: false,
                });
                let at = self.bytes.len() as u32;
                self.enc().b();
                self.fixups.push(Fixup::Branch {
                    at,
                    block: missed as u32,
                    wide: true,
                });
                Ok(())
            }
            MOp::CondBrCmp { pred } => {
                // The comparison sets the condition flags; `cset`
                // turns them into the one-bit value the branch reads,
                // so that the branch keeps the single encoding the
                // separate comparison would feed.
                self.compare_values(inst)?;
                self.enc()
                    .cset(reg::SCRATCH_A, Self::condition(*pred), Width::Bits32);
                let taken = self.arg(inst, 2)?;
                let missed = self.arg(inst, 3)?;
                let (Loc::Imm(taken), Loc::Imm(missed)) = (self.loc(taken)?, self.loc(missed)?) else {
                    return Err(CodegenError::Target {
                        what: "branching on a comparison",
                        detail: alloc::string::String::from("the targets of the branch are not blocks"),
                    });
                };
                let at = self.bytes.len() as u32;
                self.enc().cbnz(reg::SCRATCH_A, Width::Bits32);
                self.fixups.push(Fixup::Branch {
                    at,
                    block: taken as u32,
                    wide: false,
                });
                let at = self.bytes.len() as u32;
                self.enc().b();
                self.fixups.push(Fixup::Branch {
                    at,
                    block: missed as u32,
                    wide: true,
                });
                Ok(())
            }
            MOp::Ret => {
                if let Some(operand) = inst.arg0() {
                    let value = self.loc(operand)?;
                    match value {
                        Loc::Gpr(reg) => {
                            let width = self.int_width(inst);
                            self.enc().mov_reg(0, reg, width);
                        }
                        Loc::Simd(reg) => {
                            let single = matches!(inst.ty, MType::F32);
                            self.enc().fmov_reg(0, reg, single);
                        }
                        Loc::Imm(value) => {
                            self.materialize(0, value as u64, Width::Bits64);
                        }
                        Loc::FloatImm(bits) => {
                            self.materialize(0, bits, Width::Bits64);
                        }
                    }
                }
                if !self.released {
                    self.epilogue()?;
                }
                self.enc().ret();
                Ok(())
            }
            MOp::RetVoid => {
                if !self.released {
                    self.epilogue()?;
                }
                self.enc().ret();
                Ok(())
            }
            MOp::Unreachable => {
                self.enc().brk(0);
                Ok(())
            }
            other => {
                let _ = other;
                Err(CodegenError::Target {
                    what: "expanding a block",
                    detail: alloc::string::String::from("the block does not end in a branch"),
                })
            }
        }
    }

    /// The operand at the given place.
    fn arg(&self, inst: &MInst, index: usize) -> Result<Operand> {
        inst.args
            .get(index)
            .ok_or_else(|| CodegenError::Target {
                what: "reading an operand",
                detail: alloc::format!("the operation needs operand {index}"),
            })
    }

    /// Emits a branch to a block of the function.
    fn branch(&mut self, target: u32) {
        let at = self.bytes.len() as u32;
        self.enc().b();
        self.fixups.push(Fixup::Branch {
            at,
            block: target,
            wide: true,
        });
    }

    /// Expands a copy.
    fn expand_copy(&mut self, inst: &MInst) -> Result<()> {
        let operand = self.arg(inst, 0)?;
        let source = self.loc(operand)?;
        let dest = self.dest(inst)?;
        match (dest, source) {
            (Some(Loc::Gpr(target)), Loc::Gpr(reg)) => {
                if target != reg {
                    let width = self.int_width(inst);
                    self.enc().mov_reg(target, reg, width);
                }
            }
            (Some(Loc::Gpr(target)), Loc::Imm(value)) => {
                let width = self.int_width(inst);
                self.materialize(target, value as u64, width);
            }
            (Some(Loc::Gpr(target)), Loc::FloatImm(bits)) => {
                let width = self.int_width(inst);
                self.materialize(target, bits, width);
            }
            (Some(Loc::Simd(target)), Loc::Simd(reg)) => {
                if target != reg {
                    self.enc().mov_vec(target, reg);
                }
            }
            (Some(Loc::Simd(target)), Loc::Gpr(reg)) => {
                let single = matches!(inst.ty, MType::F32);
                self.enc().fmov_from_gp(target, reg, single);
            }
            (Some(Loc::Simd(target)), Loc::Imm(value)) => {
                let single = matches!(inst.ty, MType::F32);
                self.materialize_float(target, value as u64, single);
            }
            (Some(Loc::Simd(target)), Loc::FloatImm(bits)) => {
                let single = matches!(inst.ty, MType::F32);
                self.materialize_float(target, bits, single);
            }
            (None, _) => {}
            _ => {
                return Err(CodegenError::Target {
                    what: "copying a value",
                    detail: alloc::string::String::from("the destination of the copy is not a register"),
                });
            }
        }
        Ok(())
    }

    /// Expands a load.
    fn expand_load(&mut self, inst: &MInst, size: u32) -> Result<()> {
        let pointer = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(base) = pointer else {
            return Err(CodegenError::Target {
                what: "loading a value",
                detail: alloc::string::String::from("the address is not an integer"),
            });
        };
        let target = self.dest_gpr(inst)?;
        self.enc().load(target, base, 0, size)?;
        Ok(())
    }

    /// Expands a store.
    fn expand_store(&mut self, inst: &MInst, size: u32) -> Result<()> {
        let pointer = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(base) = pointer else {
            return Err(CodegenError::Target {
                what: "storing a value",
                detail: alloc::string::String::from("the address is not an integer"),
            });
        };
        let value_operand = self.arg(inst, 1)?;
        let value = self.loc(value_operand)?;
        match value {
            Loc::Gpr(reg) => self.enc().store(reg, base, 0, size)?,
            Loc::Simd(reg) => {
                // The stored value keeps its precision, which the
                // instruction names.
                let single = matches!(Self::operand_type(inst), Some(MType::F32));
                self.enc().fmov_to_gp(reg::SCRATCH_A, reg, single);
                self.enc()
                    .store(reg::SCRATCH_A, base, 0, if single { 4 } else { 8 })?;
            }
            Loc::Imm(number) => {
                self.materialize(reg::SCRATCH_A, number as u64, Width::Bits64);
                self.enc().store(reg::SCRATCH_A, base, 0, size)?;
            }
            Loc::FloatImm(bits) => {
                self.materialize(reg::SCRATCH_A, bits, Width::Bits64);
                self.enc().store(reg::SCRATCH_A, base, 0, size)?;
            }
        }
        Ok(())
    }

    /// Expands an address computation.
    fn expand_gep(&mut self, inst: &MInst, size: u32, signed: bool) -> Result<()> {
        let base = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(base) = base else {
            return Err(CodegenError::Target {
                what: "computing an address",
                detail: alloc::string::String::from("the base is not an integer"),
            });
        };
        let target = self.dest_gpr(inst)?;
        if target != base {
            self.enc().mov_reg(target, base, Width::Bits64);
        }

        let mut offset: i64 = 0;
        let scale = match size {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => {
                return Err(CodegenError::Target {
                    what: "computing an address",
                    detail: alloc::format!("an element of {size} bytes has no scaling"),
                });
            }
        };

        let operands = inst.args.as_slice();
        for operand in operands.iter().skip(1) {
            let value = self.loc(*operand)?;
            match value {
                Loc::Imm(number) => {
                    offset = offset.saturating_add(number.saturating_mul(i64::from(size)));
                }
                Loc::Gpr(index) => {
                    // The index says how wide it is, so a narrow one
                    // is widened before it is scaled.
                    let narrow = matches!(
                        Self::operand_type(inst),
                        Some(ty) if ty.size_in_bytes() == Some(4)
                    );
                    let index = if narrow {
                        if signed {
                            self.enc().sxtw(reg::SCRATCH_A, index);
                        } else {
                            self.enc().uxtw(reg::SCRATCH_A, index);
                        }
                        reg::SCRATCH_A
                    } else {
                        index
                    };
                    if scale == 0 {
                        self.enc()
                            .add_reg(target, target, index, Width::Bits64);
                    } else {
                        self.enc()
                            .add_reg_lsl(target, target, index, scale, Width::Bits64);
                    }
                }
                Loc::Simd(_) | Loc::FloatImm(_) => {
                    return Err(CodegenError::Target {
                        what: "computing an address",
                        detail: alloc::string::String::from("an index is not an integer"),
                    });
                }
            }
        }

        if offset != 0 {
            if (1..=4095).contains(&offset) {
                self.enc()
                    .add_imm(target, target, offset as u16, Width::Bits64);
            } else if (-4095..0).contains(&offset) {
                self.enc()
                    .sub_imm(target, target, (-offset) as u16, Width::Bits64);
            } else {
                let scratch = self.scratch_imm(offset as u64, Width::Bits64);
                self.enc()
                    .add_reg(target, target, scratch, Width::Bits64);
            }
        }
        Ok(())
    }

    /// Expands the broadcast of a scalar over the lanes of a vector.
    fn expand_splat(&mut self, inst: &MInst) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let operand = self.arg(inst, 0)?;
        let source = self.loc(operand)?;
        let lanes = self.lane_count(&inst.ty);
        let element = self.element_size(&inst.ty);
        match source {
            Loc::Gpr(reg) => {
                if element == 8 {
                    self.enc().dup_2d(target, reg);
                } else {
                    self.enc().dup_4s(target, reg);
                }
            }
            Loc::Simd(reg) => {
                let single = element == 4;
                self.enc().fmov_to_gp(reg::SCRATCH_A, reg, single);
                if element == 8 {
                    self.enc().dup_2d(target, reg::SCRATCH_A);
                } else {
                    self.enc().dup_4s(target, reg::SCRATCH_A);
                }
            }
            Loc::Imm(value) => {
                self.materialize(reg::SCRATCH_A, value as u64, Width::Bits64);
                if element == 8 {
                    self.enc().dup_2d(target, reg::SCRATCH_A);
                } else {
                    self.enc().dup_4s(target, reg::SCRATCH_A);
                }
            }
            Loc::FloatImm(bits) => {
                self.materialize(reg::SCRATCH_A, bits, Width::Bits64);
                if element == 8 {
                    self.enc().dup_2d(target, reg::SCRATCH_A);
                } else {
                    self.enc().dup_4s(target, reg::SCRATCH_A);
                }
            }
        }
        let _ = lanes;
        Ok(())
    }

    /// Expands the read of a lane of a vector.
    fn expand_extract(&mut self, inst: &MInst, index: u32) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        let Loc::Simd(reg) = source else {
            return Err(CodegenError::Target {
                what: "reading a lane",
                detail: alloc::string::String::from("the value is not a vector"),
            });
        };
        let target = self.dest_gpr(inst)?;
        let element = self.element_size(&inst.ty);
        if element == 8 {
            self.enc().umov_64(target, reg, index);
        } else {
            self.enc().umov_32(target, reg, index);
        }
        Ok(())
    }

    /// Expands the write of a lane of a vector.
    fn expand_insert(&mut self, inst: &MInst, index: u32) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let first = self.loc(self.arg(inst, 0)?)?;
        let Loc::Simd(vector) = first else {
            return Err(CodegenError::Target {
                what: "writing a lane",
                detail: alloc::string::String::from("the value is not a vector"),
            });
        };
        let second = self.loc(self.arg(inst, 1)?)?;
        let element = self.element_size(&inst.ty);
        if target != vector {
            self.enc().mov_vec(target, vector);
        }
        match second {
            Loc::Gpr(reg) => {
                if element == 8 {
                    self.enc().ins_64(target, reg, index);
                } else {
                    self.enc().ins_32(target, reg, index);
                }
            }
            Loc::Simd(reg) => {
                let single = element == 4;
                self.enc().fmov_to_gp(reg::SCRATCH_A, reg, single);
                if element == 8 {
                    self.enc().ins_64(target, reg::SCRATCH_A, index);
                } else {
                    self.enc().ins_32(target, reg::SCRATCH_A, index);
                }
            }
            Loc::Imm(value) => {
                self.materialize(reg::SCRATCH_A, value as u64, Width::Bits64);
                if element == 8 {
                    self.enc().ins_64(target, reg::SCRATCH_A, index);
                } else {
                    self.enc().ins_32(target, reg::SCRATCH_A, index);
                }
            }
            Loc::FloatImm(bits) => {
                self.materialize(reg::SCRATCH_A, bits, Width::Bits64);
                if element == 8 {
                    self.enc().ins_64(target, reg::SCRATCH_A, index);
                } else {
                    self.enc().ins_32(target, reg::SCRATCH_A, index);
                }
            }
        }
        Ok(())
    }

    /// Expands a shuffle of the lanes of two vectors.
    ///
    /// The lane map names a lane of the first vector, or a lane of
    /// the second one when its number is past the end of the first.
    fn expand_shuffle(&mut self, inst: &MInst, mask: &[u32]) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        let (Loc::Simd(first), Loc::Simd(mut second)) = (first, second) else {
            return Err(CodegenError::Target {
                what: "shuffling lanes",
                detail: alloc::string::String::from("the values are not vectors"),
            });
        };
        let lanes = self.lane_count(&inst.ty);
        let element = self.element_size(&inst.ty);
        if mask.len() != lanes as usize {
            return Err(CodegenError::Target {
                what: "shuffling lanes",
                detail: alloc::format!("the lane map has {} entries for {lanes} lanes", mask.len()),
            });
        }
        let first_lanes = self.lanes_of(inst, lanes);

        if target == second {
            self.enc().mov_vec(reg::SIMD_SCRATCH, second);
            second = reg::SIMD_SCRATCH;
        }
        if target != first {
            self.enc().mov_vec(target, first);
        }
        for (index, entry) in mask.iter().enumerate() {
            let (source, lane) = if *entry < first_lanes {
                (first, *entry)
            } else {
                (second, entry - first_lanes)
            };
            if lane >= first_lanes && source == second {
                return Err(CodegenError::Target {
                    what: "shuffling lanes",
                    detail: alloc::format!("the lane map names the lane {entry}, which is out of range"),
                });
            }
            let index = index as u32;
            if element == 8 {
                self.enc()
                    .ins_64_lane(target, source, index, lane);
            } else {
                self.enc()
                    .ins_32_lane(target, source, index, lane);
            }
        }
        Ok(())
    }

    /// The number of lanes of a vector type.
    fn lane_count(&self, ty: &MType) -> u32 {
        ty.lane_count()
    }

    /// The number of lanes of the vector an instruction reads.
    fn lanes_of(&self, inst: &MInst, fallback: u32) -> u32 {
        match Self::operand_type(inst) {
            Some(ty) if ty.is_vector() => ty.lane_count(),
            _ => fallback,
        }
    }

    /// The size of an element of a vector type.
    fn element_size(&self, ty: &MType) -> u32 {
        match ty {
            MType::Vec { elem, .. } => elem.size_in_bytes().unwrap_or(4),
            _ => ty.size_in_bytes().unwrap_or(4),
        }
    }

    /// Expands an addition.
    fn expand_add(&mut self, inst: &MInst) -> Result<()> {
        if let Some(MType::Vec { lanes, elem }) = vector_type(inst) {
            let target = self.dest_simd(inst)?;
            let first = self.loc(self.arg(inst, 0)?)?;
            let second = self.loc(self.arg(inst, 1)?)?;
            let (Loc::Simd(first), Loc::Simd(second)) = (first, second) else {
                return Err(CodegenError::Target {
                    what: "adding vectors",
                    detail: alloc::string::String::from("the values are not vectors"),
                });
            };
            if elem.size_in_bytes() == Some(8) {
                self.enc().add_vec_64(target, first, second);
            } else {
                let _ = lanes;
                self.enc().add_vec_32(target, first, second);
            }
            return Ok(());
        }
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => self.enc().add_reg(target, a, b, width),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                self.add_imm(target, a, value, width);
            }
            (Loc::Imm(value), Loc::Gpr(b)) => {
                let value = narrow(value, width);
                if (0..=4095).contains(&value) {
                    self.enc().add_imm(target, b, value as u16, width);
                } else {
                    let scratch = self.scratch_imm(value as u64, width);
                    self.enc().add_reg(target, scratch, b, width);
                }
            }
            (Loc::Imm(a), Loc::Imm(b)) => {
                let value = narrow(a.wrapping_add(b), width);
                self.materialize(target, value as u64, width);
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "adding values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Adds a constant to a register.
    fn add_imm(&mut self, target: u8, source: u8, value: i64, width: Width) {
        if (0..=4095).contains(&value) {
            self.enc()
                .add_imm(target, source, value as u16, width);
        } else if (-4095..0).contains(&value) {
            self.enc()
                .sub_imm(target, source, (-value) as u16, width);
        } else {
            let scratch = self.scratch_imm(value as u64, width);
            self.enc().add_reg(target, source, scratch, width);
        }
    }

    /// Expands a subtraction.
    fn expand_sub(&mut self, inst: &MInst) -> Result<()> {
        if let Some(MType::Vec { elem, .. }) = vector_type(inst) {
            let target = self.dest_simd(inst)?;
            let first = self.loc(self.arg(inst, 0)?)?;
            let second = self.loc(self.arg(inst, 1)?)?;
            let (Loc::Simd(first), Loc::Simd(second)) = (first, second) else {
                return Err(CodegenError::Target {
                    what: "subtracting vectors",
                    detail: alloc::string::String::from("the values are not vectors"),
                });
            };
            if elem.size_in_bytes() == Some(8) {
                self.enc().sub_vec_64(target, first, second);
            } else {
                self.enc().sub_vec_32(target, first, second);
            }
            return Ok(());
        }
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => self.enc().sub_reg(target, a, b, width),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                if (0..=4095).contains(&value) {
                    self.enc().sub_imm(target, a, value as u16, width);
                } else if value < 0 {
                    self.enc()
                        .add_imm(target, a, (-value) as u16, width);
                } else {
                    let scratch = self.scratch_imm(value as u64, width);
                    self.enc().sub_reg(target, a, scratch, width);
                }
            }
            (Loc::Imm(value), Loc::Gpr(b)) => {
                let value = narrow(value, width);
                let scratch = self.scratch_imm(value as u64, width);
                self.enc().sub_reg(target, scratch, b, width);
            }
            (Loc::Imm(a), Loc::Imm(b)) => {
                let value = narrow(a.wrapping_sub(b), width);
                self.materialize(target, value as u64, width);
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "subtracting values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Expands a multiplication.
    fn expand_mul(&mut self, inst: &MInst) -> Result<()> {
        if let Some(MType::Vec { elem, .. }) = vector_type(inst) {
            let target = self.dest_simd(inst)?;
            let first = self.loc(self.arg(inst, 0)?)?;
            let second = self.loc(self.arg(inst, 1)?)?;
            let (Loc::Simd(first), Loc::Simd(second)) = (first, second) else {
                return Err(CodegenError::Target {
                    what: "multiplying vectors",
                    detail: alloc::string::String::from("the values are not vectors"),
                });
            };
            if elem.size_in_bytes() == Some(8) {
                self.enc().mul_vec_64(target, first, second);
            } else {
                self.enc().mul_vec_32(target, first, second);
            }
            return Ok(());
        }
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        let (a, b) = match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => (a, b),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                (a, self.scratch_imm(value as u64, width))
            }
            (Loc::Imm(value), Loc::Gpr(b)) => {
                let value = narrow(value, width);
                (self.scratch_imm(value as u64, width), b)
            }
            (Loc::Imm(a), Loc::Imm(b)) => {
                let value = narrow(a.wrapping_mul(b), width);
                self.materialize(target, value as u64, width);
                return Ok(());
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "multiplying values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        };
        self.enc().mul(target, a, b, width);
        Ok(())
    }

    /// Expands a division.
    fn expand_div(&mut self, inst: &MInst, signed: bool) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        let (a, b) = match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => (a, b),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                (a, self.scratch_imm(value as u64, width))
            }
            (Loc::Imm(value), Loc::Gpr(b)) => {
                let value = narrow(value, width);
                (self.scratch_imm(value as u64, width), b)
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "dividing values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        };
        if signed {
            self.enc().sdiv(target, a, b, width);
        } else {
            self.enc().udiv(target, a, b, width);
        }
        Ok(())
    }

    /// Expands a remainder.
    ///
    /// The architecture has no remainder instruction, so the
    /// quotient is computed first and the remainder follows from it.
    fn expand_rem(&mut self, inst: &MInst, signed: bool) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        let (a, b) = match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => (a, b),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                (a, self.scratch_imm(value as u64, width))
            }
            (Loc::Imm(value), Loc::Gpr(b)) => {
                let value = narrow(value, width);
                (self.scratch_imm(value as u64, width), b)
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "dividing values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        };
        if signed {
            self.enc().sdiv(reg::SCRATCH_A, a, b, width);
        } else {
            self.enc().udiv(reg::SCRATCH_A, a, b, width);
        }
        self.enc()
            .msub(target, a, b, reg::SCRATCH_A, width);
        Ok(())
    }

    /// Expands a bitwise operation.
    fn expand_logic(&mut self, inst: &MInst, op: Logic) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        let (a, b) = match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => (a, b),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                (a, self.scratch_imm(value as u64, width))
            }
            (Loc::Imm(value), Loc::Gpr(b)) => {
                let value = narrow(value, width);
                (self.scratch_imm(value as u64, width), b)
            }
            (Loc::Imm(a), Loc::Imm(b)) => {
                let value = match op {
                    Logic::And => a & b,
                    Logic::Or => a | b,
                    Logic::Xor => a ^ b,
                };
                self.materialize(target, value as u64, width);
                return Ok(());
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "combining values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        };
        match op {
            Logic::And => self.enc().and_reg(target, a, b, width),
            Logic::Or => self.enc().orr_reg(target, a, b, width),
            Logic::Xor => self.enc().eor_reg(target, a, b, width),
        }
        Ok(())
    }

    /// Expands a shift.
    fn expand_shift(&mut self, inst: &MInst, op: Shift) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => {
                self.emit_shift(target, a, b, op, width);
            }
            (Loc::Gpr(a), Loc::Imm(amount)) => {
                let amount = narrow(amount, width) as u32;
                if amount < width.bits() {
                    match op {
                        Shift::Left => self.enc().lsl_imm(target, a, amount, width),
                        Shift::Logical => self.enc().lsr_imm(target, a, amount, width),
                        Shift::Arithmetic => self.enc().asr_imm(target, a, amount, width),
                    }
                } else {
                    let scratch = self.scratch_imm(u64::from(amount), width);
                    self.emit_shift(target, a, scratch, op, width);
                }
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "shifting values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Emits a shift by a register.
    fn emit_shift(&mut self, target: u8, source: u8, amount: u8, op: Shift, width: Width) {
        match op {
            Shift::Left => self.enc().lsl_reg(target, source, amount, width),
            Shift::Logical => self.enc().lsr_reg(target, source, amount, width),
            Shift::Arithmetic => self.enc().asr_reg(target, source, amount, width),
        }
    }

    /// Expands a negation.
    fn expand_neg(&mut self, inst: &MInst) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        match source {
            Loc::Gpr(reg) => {
                let target = self.dest_gpr(inst)?;
                let width = self.int_width(inst);
                self.enc().neg_reg(target, reg, width);
            }
            Loc::Simd(reg) => {
                let target = self.dest_simd(inst)?;
                let single = matches!(inst.ty, MType::F32);
                self.enc().fneg(target, reg, single);
            }
            Loc::Imm(value) => {
                let target = self.dest_gpr(inst)?;
                let width = self.int_width(inst);
                self.materialize(target, (value as u64).wrapping_neg(), width);
            }
            Loc::FloatImm(bits) => {
                let target = self.dest_simd(inst)?;
                let single = matches!(inst.ty, MType::F32);
                self.materialize_float(target, !bits, single);
            }
        }
        Ok(())
    }

    /// Expands a bitwise complement.
    fn expand_not(&mut self, inst: &MInst) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(reg) = source else {
            return Err(CodegenError::Target {
                what: "complementing a value",
                detail: alloc::string::String::from("the value is not an integer"),
            });
        };
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        self.enc().mvn_reg(target, reg, width);
        Ok(())
    }

    /// Expands the negation of a floating-point value.
    fn expand_fneg(&mut self, inst: &MInst) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        let Loc::Simd(reg) = source else {
            return Err(CodegenError::Target {
                what: "negating a value",
                detail: alloc::string::String::from("the value is not a floating-point value"),
            });
        };
        let target = self.dest_simd(inst)?;
        let single = matches!(inst.ty, MType::F32);
        self.enc().fneg(target, reg, single);
        Ok(())
    }

    /// Expands a floating-point operation.
    fn expand_float(&mut self, inst: &MInst, op: Float) -> Result<()> {
        let first = self.float_operand(inst, 0)?;
        let second = self.float_operand(inst, 1)?;
        let target = self.dest_simd(inst)?;
        let single = matches!(inst.ty, MType::F32);
        match op {
            Float::Add => self.enc().fadd(target, first, second, single),
            Float::Sub => self.enc().fsub(target, first, second, single),
            Float::Mul => self.enc().fmul(target, first, second, single),
            Float::Div => self.enc().fdiv(target, first, second, single),
        }
        Ok(())
    }

    /// Expands the square root of a floating-point value.
    fn expand_fsqrt(&mut self, inst: &MInst) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let reg = self.float_operand(inst, 0)?;
        let single = matches!(inst.ty, MType::F32);
        self.enc().fsqrt(target, reg, single);
        Ok(())
    }

    /// Expands the absolute value of a floating-point value.
    fn expand_fabs(&mut self, inst: &MInst) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let reg = self.float_operand(inst, 0)?;
        let single = matches!(inst.ty, MType::F32);
        self.enc().fabs(target, reg, single);
        Ok(())
    }

    /// Expands the smaller or the larger of two floating-point
    /// values.
    ///
    /// The instruction of the architecture returns a value that is not
    /// a number when one of the two is not one.
    fn expand_fmin(&mut self, inst: &MInst, smallest: bool) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let first = self.float_operand(inst, 0)?;
        let second = self.float_operand(inst, 1)?;
        let single = matches!(inst.ty, MType::F32);
        if smallest {
            self.enc().fmin(target, first, second, single);
        } else {
            self.enc().fmax(target, first, second, single);
        }
        Ok(())
    }

    /// Expands the positive difference of two floating-point values.
    ///
    /// The difference of the values is clamped at zero, which is what
    /// the difference of a value that is not the larger one comes to.
    fn expand_fdim(&mut self, inst: &MInst) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let first = self.float_operand(inst, 0)?;
        let second = self.float_operand(inst, 1)?;
        let single = matches!(inst.ty, MType::F32);
        self.enc().fsub(target, first, second, single);
        // The zero the difference is clamped at.
        self.materialize(reg::SCRATCH_A, 0, Width::Bits64);
        self.enc()
            .fmov_from_gp(reg::SIMD_SCRATCH, reg::SCRATCH_A, single);
        self.enc()
            .fmax(target, target, reg::SIMD_SCRATCH, single);
        Ok(())
    }

    /// Expands the magnitude of one floating-point value with the
    /// sign of another.
    ///
    /// The sign of the second value is copied into the result with a
    /// mask that holds the sign bit alone.
    fn expand_fcopysign(&mut self, inst: &MInst) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let magnitude = self.float_operand(inst, 0)?;
        let sign = self.float_operand(inst, 1)?;
        let single = matches!(inst.ty, MType::F32);
        // The magnitude comes to the destination first: the mask and
        // the sign take the scratches afterwards.
        self.enc().mov_vec(target, magnitude);
        // The mask of the sign bit.
        let mask = if single {
            0x0000_0000_8000_0000u64
        } else {
            0x8000_0000_0000_0000u64
        };
        self.materialize(reg::SCRATCH_A, mask, Width::Bits64);
        self.enc()
            .fmov_from_gp(reg::SIMD_SCRATCH, reg::SCRATCH_A, false);
        // The sign of the second value, and the magnitude with that
        // sign.
        self.enc()
            .and_vec(reg::SIMD_SCRATCH_B, sign, reg::SIMD_SCRATCH);
        self.enc()
            .eor_vec(target, target, reg::SIMD_SCRATCH_B);
        Ok(())
    }

    /// Expands the remainder of a division of two floating-point
    /// values.
    ///
    /// The architecture has no remainder instruction, so the
    /// quotient of a division that is truncated towards zero is
    /// computed first, and the remainder follows from it.
    fn expand_frem(&mut self, inst: &MInst) -> Result<()> {
        let target = self.dest_simd(inst)?;
        let first = self.float_operand(inst, 0)?;
        let second = self.float_operand(inst, 1)?;
        let single = matches!(inst.ty, MType::F32);
        // The dividend and the divisor move to registers of their
        // own, so that the quotient can be computed without losing
        // them.
        self.enc().mov_vec(target, first);
        self.enc().mov_vec(reg::SIMD_SCRATCH, second);
        // The quotient of the division.
        self.enc()
            .fdiv(reg::SIMD_SCRATCH_B, target, reg::SIMD_SCRATCH, single);
        // The quotient of a division that is truncated towards zero.
        self.enc()
            .fcvtzs(reg::SCRATCH_A, reg::SIMD_SCRATCH_B, single, true);
        self.enc()
            .scvtf(reg::SIMD_SCRATCH_B, reg::SCRATCH_A, single);
        // The dividend less the quotient times the divisor.
        self.enc()
            .fmls(reg::SIMD_SCRATCH_B, target, reg::SIMD_SCRATCH, single);
        self.enc().mov_vec(target, reg::SIMD_SCRATCH_B);
        Ok(())
    }

    /// Expands an integer comparison.
    fn expand_icmp(&mut self, inst: &MInst, pred: IPred) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        self.compare_values(inst)?;
        self.enc()
            .cset(target, Self::condition(pred), Width::Bits32);
        Ok(())
    }

    /// Compares the two integer values an instruction names, leaving
    /// the condition flags set for them.
    ///
    /// A comparison that produces a value turns those flags into the
    /// one-bit result with `cset`; a fused compare-and-branch reads
    /// them directly.
    fn compare_values(&mut self, inst: &MInst) -> Result<()> {
        let width = self.compare_width(inst);
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => self.enc().cmp_reg(a, b, width),
            (Loc::Gpr(a), Loc::Imm(value)) => {
                let value = narrow(value, width);
                if (0..=4095).contains(&value) {
                    self.enc().cmp_imm(a, value as u16, width);
                } else {
                    let scratch = self.scratch_imm(value as u64, width);
                    self.enc().cmp_reg(a, scratch, width);
                }
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "comparing values",
                    detail: alloc::string::String::from("an operand is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// The condition a comparison predicate stands for.
    fn condition(pred: IPred) -> Cond {
        match pred {
            IPred::Eq => Cond::Eq,
            IPred::Ne => Cond::Ne,
            IPred::Ugt => Cond::Hi,
            IPred::Uge => Cond::Cs,
            IPred::Ult => Cond::Cc,
            IPred::Ule => Cond::Ls,
            IPred::Sgt => Cond::Gt,
            IPred::Sge => Cond::Ge,
            IPred::Slt => Cond::Lt,
            IPred::Sle => Cond::Le,
        }
    }

    /// Expands a floating-point comparison.
    ///
    /// The architecture compares the two values and sets the
    /// condition flags: the flags tell whether the comparison was
    /// ordered, and the condition the comparison stands for is
    /// picked so that it holds only for an ordered result.
    fn expand_fcmp(&mut self, inst: &MInst, pred: FPred) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        let first = self.loc(self.arg(inst, 0)?)?;
        let second = self.loc(self.arg(inst, 1)?)?;
        let (Loc::Simd(a), Loc::Simd(b)) = (first, second) else {
            return Err(CodegenError::Target {
                what: "comparing floating-point values",
                detail: alloc::string::String::from("a value is not a floating-point value"),
            });
        };
        let single = !matches!(Self::operand_type(inst), Some(MType::F64));
        self.enc().fcmp(a, b, single);
        match pred {
            FPred::Oeq => self.enc().cset(target, Cond::Eq, Width::Bits32),
            FPred::One => {
                self.enc().cset(target, Cond::Ne, Width::Bits32);
                self.enc()
                    .csel(target, target, 31, Cond::Vc, Width::Bits32);
            }
            FPred::Olt => self.enc().cset(target, Cond::Mi, Width::Bits32),
            FPred::Ole => self.enc().cset(target, Cond::Ls, Width::Bits32),
            FPred::Ogt => self.enc().cset(target, Cond::Gt, Width::Bits32),
            FPred::Oge => self.enc().cset(target, Cond::Ge, Width::Bits32),
            FPred::Ueq | FPred::Une => self.enc().cset(target, Cond::Vs, Width::Bits32),
            FPred::False => self.enc().mov_reg(target, 31, Width::Bits32),
        }
        Ok(())
    }

    /// Expands a conditional selection.
    fn expand_select(&mut self, inst: &MInst) -> Result<()> {
        let condition = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(cond) = condition else {
            return Err(CodegenError::Target {
                what: "selecting a value",
                detail: alloc::string::String::from("the condition is not an integer"),
            });
        };
        let first = self.loc(self.arg(inst, 1)?)?;
        let second = self.loc(self.arg(inst, 2)?)?;
        self.enc().cmp_imm(cond, 0, Width::Bits32);
        match (first, second) {
            (Loc::Gpr(a), Loc::Gpr(b)) => {
                let target = self.dest_gpr(inst)?;
                let width = self.int_width(inst);
                self.enc().csel(target, a, b, Cond::Ne, width);
            }
            (Loc::Simd(a), Loc::Simd(b)) => {
                let target = self.dest_simd(inst)?;
                if matches!(inst.ty, MType::Vec { .. }) {
                    self.select_vector(target, a, b);
                } else {
                    let single = matches!(inst.ty, MType::F32);
                    self.enc().fcsel(target, a, b, Cond::Ne, single);
                }
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "selecting a value",
                    detail: alloc::string::String::from("the values do not share a class"),
                });
            }
        }
        Ok(())
    }

    /// Selects between two vectors with the condition flags.
    ///
    /// The architecture has no conditional select for vectors, so
    /// the choice is made with a mask: the mask holds every bit for
    /// the value that is taken and no bit for the other one.
    fn select_vector(&mut self, target: u8, first: u8, second: u8) {
        self.enc()
            .cset(reg::SCRATCH_A, Cond::Ne, Width::Bits32);
        self.enc()
            .neg_reg(reg::SCRATCH_B, reg::SCRATCH_A, Width::Bits64);
        self.enc()
            .dup_2d(reg::SIMD_SCRATCH, reg::SCRATCH_B);
        self.enc().eor_vec(reg::SCRATCH_B, first, second);
        self.enc()
            .and_vec(reg::SCRATCH_B, reg::SCRATCH_B, reg::SIMD_SCRATCH);
        self.enc().eor_vec(target, first, reg::SCRATCH_B);
    }

    /// Expands a zero extension.
    fn expand_zext(&mut self, inst: &MInst) -> Result<()> {
        let operand = self.arg(inst, 0)?;
        let source = self.loc(operand)?;
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let source_size = Self::operand_type(inst).and_then(MType::size_in_bytes);
        match (source, source_size) {
            (Loc::Gpr(reg), Some(1)) => self.enc().uxtb(target, reg),
            (Loc::Gpr(reg), Some(2)) => self.enc().uxth(target, reg),
            (Loc::Gpr(reg), _) if width.is_64() => self.enc().uxtw(target, reg),
            (Loc::Gpr(reg), _) => {
                if target != reg {
                    self.enc().mov_reg(target, reg, Width::Bits32);
                }
            }
            (Loc::Imm(value), _) => {
                self.materialize(target, value as u64, width);
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "extending a value",
                    detail: alloc::string::String::from("the value is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Expands a sign extension.
    fn expand_sext(&mut self, inst: &MInst) -> Result<()> {
        let operand = self.arg(inst, 0)?;
        let source = self.loc(operand)?;
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let source_size = Self::operand_type(inst).and_then(MType::size_in_bytes);
        match (source, source_size) {
            (Loc::Gpr(reg), Some(1)) => {
                if width.is_64() {
                    self.enc().sxtb(target, reg);
                } else {
                    self.enc().sxtb32(target, reg);
                }
            }
            (Loc::Gpr(reg), Some(2)) => {
                if width.is_64() {
                    self.enc().sxth(target, reg);
                } else {
                    self.enc().sxth32(target, reg);
                }
            }
            (Loc::Gpr(reg), _) if width.is_64() => self.enc().sxtw(target, reg),
            (Loc::Gpr(reg), _) => {
                if target != reg {
                    self.enc().mov_reg(target, reg, Width::Bits32);
                }
            }
            (Loc::Imm(value), _) => {
                self.materialize(target, value as u64, width);
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "extending a value",
                    detail: alloc::string::String::from("the value is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Expands a truncation.
    fn expand_trunc(&mut self, inst: &MInst) -> Result<()> {
        let operand = self.arg(inst, 0)?;
        let source = self.loc(operand)?;
        let target = self.dest_gpr(inst)?;
        let width = self.int_width(inst);
        let source_size = Self::operand_type(inst).and_then(MType::size_in_bytes);
        match (source, source_size) {
            (Loc::Gpr(reg), Some(1)) => self.enc().uxtb(target, reg),
            (Loc::Gpr(reg), Some(2)) => self.enc().uxth(target, reg),
            (Loc::Gpr(reg), _) => {
                if target != reg {
                    self.enc().mov_reg(target, reg, width);
                }
            }
            (Loc::Imm(value), _) => {
                self.materialize(target, value as u64, width);
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "truncating a value",
                    detail: alloc::string::String::from("the value is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Expands the conversion of a floating-point value to an
    /// integer.
    fn expand_fp_to_int(&mut self, inst: &MInst, signed: bool) -> Result<()> {
        let reg = self.float_operand(inst, 0)?;
        let target = self.dest_gpr(inst)?;
        let single = !matches!(Self::operand_type(inst), Some(MType::F64));
        let wide = self.int_width(inst).is_64();
        if signed {
            self.enc().fcvtzs(target, reg, single, wide);
        } else if wide {
            self.enc().fcvtzu(target, reg, single, true);
        } else {
            // The architecture converts to an unsigned sixty-four-bit
            // value only, so the signed conversion is used and the
            // low half of the result is kept.
            self.enc()
                .fcvtzs(reg::SCRATCH_A, reg, single, true);
            self.enc()
                .mov_reg(target, reg::SCRATCH_A, Width::Bits32);
        }
        Ok(())
    }

    /// Expands the conversion of an integer to a floating-point
    /// value.
    fn expand_int_to_fp(&mut self, inst: &MInst, signed: bool) -> Result<()> {
        let reg = self.int_operand(inst, 0)?;
        let target = self.dest_simd(inst)?;
        let single = matches!(inst.ty, MType::F32);
        // The source is read at the width the instruction names: a
        // thirty-two-bit value keeps its bits in the low half of the
        // register, so it is widened before it is converted.
        let wide = Self::operand_type(inst).and_then(MType::size_in_bytes) == Some(8);
        let reg = if wide {
            reg
        } else {
            if signed {
                self.enc().sxtw(reg::SCRATCH_A, reg);
            } else {
                self.enc().uxtw(reg::SCRATCH_A, reg);
            }
            reg::SCRATCH_A
        };
        if signed {
            self.enc().scvtf(target, reg, single);
        } else {
            self.enc().ucvtf(target, reg, single);
        }
        Ok(())
    }

    /// Expands the narrowing of a floating-point value.
    fn expand_fp_trunc(&mut self, inst: &MInst) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        let Loc::Simd(reg) = source else {
            return Err(CodegenError::Target {
                what: "converting a value",
                detail: alloc::string::String::from("the value is not a floating-point value"),
            });
        };
        let target = self.dest_simd(inst)?;
        if matches!(inst.ty, MType::F32) {
            self.enc().fcvt_narrow(target, reg);
        } else {
            return Err(CodegenError::Unsupported {
                what: "a sixteen-bit floating-point value",
                detail: alloc::string::String::from(
                    "the architecture has no sixteen-bit floating-point form",
                ),
            });
        }
        Ok(())
    }

    /// Expands the widening of a floating-point value.
    fn expand_fp_ext(&mut self, inst: &MInst) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        let Loc::Simd(reg) = source else {
            return Err(CodegenError::Target {
                what: "converting a value",
                detail: alloc::string::String::from("the value is not a floating-point value"),
            });
        };
        let target = self.dest_simd(inst)?;
        if matches!(inst.ty, MType::F64) {
            self.enc().fcvt_widen(target, reg);
        } else {
            return Err(CodegenError::Unsupported {
                what: "a sixteen-bit floating-point value",
                detail: alloc::string::String::from(
                    "the architecture has no sixteen-bit floating-point form",
                ),
            });
        }
        Ok(())
    }

    /// Expands a change of the interpretation of the bits of a
    /// value.
    fn expand_bitcast(&mut self, inst: &MInst) -> Result<()> {
        let operand = self.arg(inst, 0)?;
        let source = self.loc(operand)?;
        match (self.dest(inst)?, source) {
            (Some(Loc::Gpr(target)), Loc::Gpr(reg)) => {
                let width = self.int_width(inst);
                self.enc().mov_reg(target, reg, width);
            }
            (Some(Loc::Gpr(target)), Loc::Simd(reg)) => {
                let single = !matches!(Self::operand_type(inst), Some(MType::F64));
                self.enc().fmov_to_gp(target, reg, single);
            }
            (Some(Loc::Simd(target)), Loc::Gpr(reg)) => {
                let single = matches!(inst.ty, MType::F32);
                self.enc().fmov_from_gp(target, reg, single);
            }
            (Some(Loc::Simd(target)), Loc::Simd(reg)) => {
                if target != reg {
                    self.enc().mov_vec(target, reg);
                }
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "reinterpreting a value",
                    detail: alloc::string::String::from(
                        "the classes of the value and its destination differ",
                    ),
                });
            }
        }
        Ok(())
    }

    /// Expands a call.
    fn expand_call(&mut self, inst: &MInst, callee: SymId) -> Result<()> {
        let function = self
            .module
            .function_index(callee)
            .ok_or_else(|| CodegenError::Unsupported {
                what: "a call to a function outside the module",
                detail: alloc::string::String::from(
                    "the code is position-independent and has no way to name an outside function",
                ),
            })?;

        let mut integers = 0u32;
        let mut floats = 0u32;
        for (index, operand) in inst.args.as_slice().iter().enumerate() {
            let value = self.loc(*operand)?;
            // The arguments of a call have types of their own, which
            // the callee's own parameters name.
            let argument_ty = self.argument_type(callee, index);
            // The wide move is right for a value of either
            // precision: a single-precision one keeps its bits in the
            // low half of the register.
            let single = matches!(argument_ty, Some(MType::F32));
            match value {
                Loc::Gpr(reg) => {
                    if integers >= 8 {
                        return Err(CodegenError::Unsupported {
                            what: "a call with more than eight integer arguments",
                            detail: alloc::string::String::from(
                                "the architecture passes the rest on the stack",
                            ),
                        });
                    }
                    if reg != integers as u8 {
                        let width = argument_ty
                            .and_then(MType::size_in_bytes)
                            .map_or(Width::Bits64, |size| Width::of_type(Some(size)));
                        self.enc().mov_reg(integers as u8, reg, width);
                    }
                    integers += 1;
                }
                Loc::Simd(reg) => {
                    if floats >= 8 {
                        return Err(CodegenError::Unsupported {
                            what: "a call with more than eight floating-point arguments",
                            detail: alloc::string::String::from(
                                "the architecture passes the rest on the stack",
                            ),
                        });
                    }
                    if reg != floats as u8 {
                        self.enc().fmov_reg(floats as u8, reg, single);
                    }
                    floats += 1;
                }
                Loc::Imm(number) => {
                    if integers >= 8 {
                        return Err(CodegenError::Unsupported {
                            what: "a call with more than eight integer arguments",
                            detail: alloc::string::String::from(
                                "the architecture passes the rest on the stack",
                            ),
                        });
                    }
                    self.materialize(integers as u8, number as u64, Width::Bits64);
                    integers += 1;
                }
                Loc::FloatImm(bits) => {
                    if floats >= 8 {
                        return Err(CodegenError::Unsupported {
                            what: "a call with more than eight floating-point arguments",
                            detail: alloc::string::String::from(
                                "the architecture passes the rest on the stack",
                            ),
                        });
                    }
                    self.materialize_float(floats as u8, bits, single);
                    floats += 1;
                }
            }
        }

        let at = self.bytes.len() as u32;
        self.enc().bl();
        self.fixups.push(Fixup::Call { at, function });

        if inst.dest.is_some() {
            match inst.ty {
                MType::F32 | MType::F64 => {
                    let target = self.dest_simd(inst)?;
                    let single = matches!(inst.ty, MType::F32);
                    self.enc().fmov_reg(target, 0, single);
                }
                MType::Vec { .. } => {
                    return Err(CodegenError::Unsupported {
                        what: "a call that returns a vector",
                        detail: alloc::string::String::from(
                            "the architecture returns a vector in no register",
                        ),
                    });
                }
                _ => {
                    let target = self.dest_gpr(inst)?;
                    let width = self.int_width(inst);
                    self.enc().mov_reg(target, 0, width);
                }
            }
        }
        Ok(())
    }

    /// Expands the read of a field of the work item.
    fn expand_work_item(&mut self, inst: &MInst, field: WorkItemField, lane: u32) -> Result<()> {
        let pointer = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(base) = pointer else {
            return Err(CodegenError::Target {
                what: "reading the work item",
                detail: alloc::string::String::from("the address is not an integer"),
            });
        };
        let target = self.dest_gpr(inst)?;
        let offset = i64::from(field.offset()) + i64::from(lane) * 4;
        self.enc().load(target, base, offset, 4)?;
        Ok(())
    }

    /// Expands the read of the base of the argument block.
    fn expand_arg_base(&mut self, inst: &MInst) -> Result<()> {
        let pointer = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(base) = pointer else {
            return Err(CodegenError::Target {
                what: "reading the argument block",
                detail: alloc::string::String::from("the address is not an integer"),
            });
        };
        let target = self.dest_gpr(inst)?;
        self.enc()
            .load(target, base, i64::from(ARG_BASE_OFFSET), 8)?;
        Ok(())
    }

    /// Expands the read of an argument of the kernel.
    fn expand_arg_load(&mut self, inst: &MInst, index: u32, kind: ArgKind) -> Result<()> {
        let pointer = self.loc(self.arg(inst, 0)?)?;
        let Loc::Gpr(base) = pointer else {
            return Err(CodegenError::Target {
                what: "reading an argument",
                detail: alloc::string::String::from("the address is not an integer"),
            });
        };
        let offset = i64::from(index) * ARG_SLOT as i64;
        let target = self.dest_gpr(inst)?;
        match kind {
            ArgKind::Pointer => {
                self.enc().load(target, base, offset, 8)?;
            }
            ArgKind::Scalar { size } => match size {
                1 => self.enc().load(target, base, offset, 1)?,
                2 => self.enc().load(target, base, offset, 2)?,
                4 => {
                    if matches!(inst.ty, MType::F32) {
                        self.enc().load(reg::SCRATCH_A, base, offset, 4)?;
                        self.enc()
                            .fmov_from_gp(target, reg::SCRATCH_A, true);
                    } else {
                        self.enc().load(target, base, offset, 4)?;
                    }
                }
                8 => {
                    if matches!(inst.ty, MType::F64) {
                        self.enc().load(reg::SCRATCH_A, base, offset, 8)?;
                        self.enc()
                            .fmov_from_gp(target, reg::SCRATCH_A, false);
                    } else {
                        self.enc().load(target, base, offset, 8)?;
                    }
                }
                other => {
                    return Err(CodegenError::Unsupported {
                        what: "a kernel argument of an unusual size",
                        detail: alloc::format!("an argument of {other} bytes has no form"),
                    });
                }
            },
        }
        Ok(())
    }

    /// Expands the read of a parameter of the function.
    ///
    /// The integer parameters and the floating-point ones are
    /// counted apart, because the architecture hands them to two
    /// sets of registers.
    fn expand_param(&mut self, inst: &MInst, index: u32) -> Result<()> {
        if index >= 8 {
            return Err(CodegenError::Unsupported {
                what: "a function with more than eight parameters",
                detail: alloc::string::String::from("the architecture passes the rest on the stack"),
            });
        }
        let mut integers = 0u32;
        let mut floats = 0u32;
        for (position, floating) in self.parameters() {
            if position >= index {
                break;
            }
            if floating {
                floats += 1;
            } else {
                integers += 1;
            }
        }
        let target = self.dest(inst)?;
        match target {
            Some(Loc::Gpr(reg)) => {
                let width = self.int_width(inst);
                self.enc().mov_reg(reg, integers as u8, width);
            }
            Some(Loc::Simd(reg)) => {
                let single = matches!(inst.ty, MType::F32);
                self.enc().fmov_reg(reg, floats as u8, single);
            }
            Some(_) => {
                return Err(CodegenError::Target {
                    what: "reading a parameter",
                    detail: alloc::string::String::from("the parameter has no register"),
                });
            }
            None => {}
        }
        Ok(())
    }

    /// The parameters of the function, in order, and whether each of
    /// them is a floating-point value.
    fn parameters(&self) -> Vec<(u32, bool)> {
        let mut parameters: Vec<(u32, bool)> = Vec::new();
        for block in &self.function.blocks {
            for inst in &block.insts {
                let MOp::Param { index } = inst.op else {
                    continue;
                };
                parameters.push((index, inst.ty.is_floating()));
            }
        }
        parameters.sort_by_key(|(index, _)| *index);
        parameters
    }

    /// Expands the address of a slot of the frame.
    fn expand_frame_addr(&mut self, inst: &MInst, offset: u32) -> Result<()> {
        let target = self.dest_gpr(inst)?;
        if offset == 0 {
            self.enc().mov_reg(target, reg::SP, Width::Bits64);
        } else if offset < 4096 {
            self.enc()
                .add_imm(target, reg::SP, offset as u16, Width::Bits64);
        } else if offset.is_multiple_of(4096) {
            self.enc()
                .add_imm_scaled(target, reg::SP, (offset / 4096) as u16, Width::Bits64);
        } else {
            let scratch = self.scratch_imm(u64::from(offset), Width::Bits64);
            self.enc()
                .add_reg(target, reg::SP, scratch, Width::Bits64);
        }
        Ok(())
    }

    /// Expands the store of a value into a slot of the frame.
    ///
    /// The slot holds the bits of the value, so a floating-point
    /// value is moved into a general-purpose register first.
    fn expand_spill_store(&mut self, inst: &MInst, offset: u32, size: u32) -> Result<()> {
        let source = self.loc(self.arg(inst, 0)?)?;
        match source {
            Loc::Gpr(reg) => self
                .enc()
                .store(reg, reg::SP, i64::from(offset), size)?,
            Loc::Simd(reg) => {
                // A floating-point value keeps its bits in the
                // frame, so they are moved into a general-purpose
                // register first.
                let single = size == 4;
                self.enc().fmov_to_gp(reg::SCRATCH_A, reg, single);
                self.enc()
                    .store(reg::SCRATCH_A, reg::SP, i64::from(offset), size)?;
            }
            _ => {
                return Err(CodegenError::Target {
                    what: "storing a value",
                    detail: alloc::string::String::from("the value is not an integer"),
                });
            }
        }
        Ok(())
    }

    /// Expands the load of a value from a slot of the frame.
    ///
    /// The slot holds the bits of the value, so a floating-point
    /// value is read into a general-purpose register first and moved
    /// into the floating-point register afterwards.
    fn expand_spill_load(&mut self, inst: &MInst, offset: u32, size: u32) -> Result<()> {
        match self.dest(inst)? {
            Some(Loc::Gpr(target)) => {
                self.enc()
                    .load(target, reg::SP, i64::from(offset), size)?;
            }
            Some(Loc::Simd(target)) => {
                // A floating-point value keeps its bits in the
                // frame, so they are read into a general-purpose
                // register first and moved into the floating-point
                // register afterwards.
                self.enc()
                    .load(reg::SCRATCH_A, reg::SP, i64::from(offset), size)?;
                let single = size == 4;
                self.enc()
                    .fmov_from_gp(target, reg::SCRATCH_A, single);
            }
            Some(_) | None => {
                return Err(CodegenError::Target {
                    what: "reading a value",
                    detail: alloc::string::String::from("the value has no register"),
                });
            }
        }
        Ok(())
    }
}

/// The place a physical register lives in.
fn locate(preg: PReg) -> Loc {
    if reg::is_simd(preg) {
        Loc::Simd(reg::simd_index(preg).unwrap_or(0))
    } else {
        Loc::Gpr(reg::gpr_index(preg).unwrap_or(0))
    }
}

/// The value of an immediate, narrowed to the width of the
/// operation.
fn narrow(value: i64, width: Width) -> i64 {
    if width.is_64() {
        value
    } else {
        i64::from(value as u32)
    }
}

/// The vector type of an instruction, when it has one.
fn vector_type(inst: &MInst) -> Option<&MType> {
    match &inst.ty {
        MType::Vec { .. } => Some(&inst.ty),
        _ => None,
    }
}

/// Rounds a number up to a multiple of sixteen.
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

/// The bitwise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Logic {
    /// A bitwise and.
    And,
    /// A bitwise or.
    Or,
    /// A bitwise exclusive or.
    Xor,
}
/// The shifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shift {
    /// A shift to the left.
    Left,
    /// A shift to the right, filling with zeros.
    Logical,
    /// A shift to the right, filling with the sign.
    Arithmetic,
}

/// The floating-point operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Float {
    /// An addition.
    Add,
    /// A subtraction.
    Sub,
    /// A multiplication.
    Mul,
    /// A division.
    Div,
}
