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

//! The expansion of the machine operations into target instructions.
//!
//! The machine IR is semantic: an operation says what it computes,
//! not how.  This module turns every operation into the instructions
//! of the architecture, and it is where the decisions the instruction
//! set forces upon the back end are taken:
//!
//! * A division has no instruction, so it becomes a loop.
//! * A comparison produces a one-bit value, so it becomes a compare
//!   and a conditional execution block.
//! * A selection between two values becomes a compare and a
//!   conditional execution block with a "then" and an "else" part.
//! * A narrow load is a single load, because the architecture
//!   already extends the value it loads.
//!
//! # The registers
//!
//! The expansion works on physical registers.  The module that
//! reaches the back end has been through the register allocator: every
//! operand and every destination names a register of the target's
//! file.  A module that still names a virtual register never went
//! through the allocator, and the back end refuses it.
//!
//! # The scratch
//!
//! The expansion needs registers of its own: a constant has to live
//! in a register before an instruction can use it, and the division
//! loop needs four of them.  The scratch registers are the four
//! argument registers, which the allocator never hands out because
//! they belong to the calling convention.  A live value therefore
//! never sits in them, and the expansion can use them freely.
//!
//! # The frame
//!
//! The prologue of a function saves the callee-saved registers it
//! uses and then reserves the frame of the function.  The frame base
//! is the value of the stack pointer right after that subtraction, so
//! every slot of the frame is at a known distance from the stack
//! pointer, and the saved registers sit above the frame.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_abi::{ARG_BASE_OFFSET, ARG_SLOT};
use codevar_ocl_native_codegen::error::{CodegenError, Result};
use codevar_ocl_native_codegen::mir::{ArgKind, Destination, IPred, MInst, MOp, MType, MirFunction, Operand};

use crate::encode;
use crate::reg;

/// The register the expansion materializes a constant in.
///
/// The scratch is one of the four argument registers, and the
/// allocator never hands those out, so no live value can be
/// clobbered by it.
pub const SCRATCH: u8 = 3;

/// The second register the expansion materializes a constant in.
///
/// A store needs the address and the value at the same time, so the
/// value gets a register of its own.
const SCRATCH_VALUE: u8 = 2;

/// The size of the argument slot of the kernel arguments.
const ARG_SLOT_SIZE: u32 = ARG_SLOT as u32;

/// The first callee-saved register of the architecture.
const SAVED_FIRST: u8 = 4;

/// The last callee-saved register of the architecture.
const SAVED_LAST: u8 = 11;

/// Where a branch of the expansion points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupTarget {
    /// A block of the function.
    Block(u32),
    /// A function of the module.
    Function(usize),
    /// A place inside the block that holds the branch.
    Local(usize),
}

/// The kind of a branch that waits for its displacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupKind {
    /// An unconditional branch.
    Branch,
    /// A call.
    BranchLink,
    /// A conditional branch.
    BranchCond(u8),
}

/// A branch whose displacement is not known yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixup {
    /// The offset of the instruction inside its block.
    pub at: usize,
    /// The kind of the branch.
    pub kind: FixupKind,
    /// Where the branch points.
    pub target: FixupTarget,
}

/// The code of one block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockCode {
    /// The bytes of the block.
    pub bytes: Vec<u8>,
    /// The branches that still need their displacement.
    pub fixups: Vec<Fixup>,
    /// The number of instructions of the block.
    pub instructions: u32,
}

/// The code of one function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCode {
    /// The blocks of the function, in layout order.
    pub blocks: Vec<BlockCode>,
    /// The prologue of the function.
    pub prologue: Vec<u8>,
    /// The number of instructions of the function.
    pub instructions: u32,
}

/// A value an operand of the machine code names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value {
    /// A general-purpose register.
    Reg(u8),
    /// A constant.
    Imm(u32),
}

/// The comparison a comparison result needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compare {
    /// A register and a register.
    Regs(u8, u8),
    /// A register and a constant.
    RegImm(u8, u32),
    /// A constant and a register.
    ImmReg(u32, u8),
    /// Two constants.
    Imms(u32, u32),
}

/// The registers of the function that survive a call, and whether it
/// calls at all.
///
/// The prologue saves the registers, so the scan has to know them
/// before the frame is reserved: the frame and the saved registers
/// together have to keep the stack pointer aligned.
fn scan_registers(function: &MirFunction) -> (u16, bool) {
    let mut saved = 0u16;
    let mut uses_link = false;
    for block in &function.blocks {
        for inst in &block.insts {
            if matches!(inst.op, MOp::Call { .. }) {
                uses_link = true;
            }
            if let Some(dest) = inst.dest_reg()
                && let Some(number) = reg::gpr_number(dest)
            {
                saved |= saved_bit(number);
            }
            for operand in inst.args.as_slice() {
                if let Operand::Physical(preg) = operand
                    && let Some(number) = reg::gpr_number(*preg)
                {
                    saved |= saved_bit(number);
                }
            }
        }
    }
    (saved, uses_link)
}

/// The bit of the callee-saved register in the mask of the saved
/// registers.
///
/// The mask counts the registers by their architectural number, which
/// is the way the register lists of the architecture are written.
fn saved_bit(number: u8) -> u16 {
    if (SAVED_FIRST..=SAVED_LAST).contains(&number) {
        1 << number
    } else {
        0
    }
}

/// The expansion of one function.
struct Expander<'a> {
    callees: &'a [Option<usize>],
    has_fp: bool,
    bytes: Vec<u8>,
    fixups: Vec<Fixup>,
    instructions: u32,
    saved: u16,
    uses_link: bool,
    frame: u32,
}

impl<'a> Expander<'a> {
    /// Creates the expansion of the function.
    fn new(
        function: &'a MirFunction,
        callees: &'a [Option<usize>],
        has_fp: bool,
        saved: u16,
        uses_link: bool,
    ) -> Self {
        let pushed = saved.count_ones() + u32::from(uses_link);
        let frame = align_frame(function.frame_size, pushed);
        Self {
            callees,
            has_fp,
            bytes: Vec::new(),
            fixups: Vec::new(),
            instructions: 0,
            saved,
            uses_link,
            frame,
        }
    }

    /// The value an operand names.
    ///
    /// The machine code has already been allocated, so an operand
    /// names a physical register or a constant; a virtual one means
    /// the machine code never went through the allocator.
    fn value(&self, operand: Operand) -> Result<Value> {
        match operand {
            Operand::Physical(preg) => match reg::gpr_number(preg) {
                Some(number) => Ok(Value::Reg(number)),
                None => Err(self.foreign_register_error(preg)),
            },
            Operand::Immediate(value) => Ok(Value::Imm(value as u32)),
            Operand::Virtual(value) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: format!(
                    "the machine code was not allocated: the operand names the virtual register {}",
                    value.index()
                ),
            }),
            Operand::FloatImmediate(_) => Err(CodegenError::Unsupported {
                what: "a floating-point constant",
                detail: String::from("the back end does not implement the floating-point unit"),
            }),
            Operand::Symbol(_) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: String::from("the operand is a symbol"),
            }),
        }
    }

    /// The error the back end returns for a register that is not an
    /// integer one.
    ///
    /// A value of the floating-point unit or of the vector unit has
    /// no home in the integer registers, and the back end refuses it.
    fn foreign_register_error(&self, preg: codevar_ocl_native_codegen::mir::PReg) -> CodegenError {
        if reg::is_q(preg) {
            CodegenError::Unsupported {
                what: "a vector value",
                detail: String::from("the architecture has no vector unit"),
            }
        } else {
            self.floating_point_error("a floating-point value")
        }
    }

    /// The general-purpose register the instruction writes.
    fn dest(&self, inst: &MInst) -> Result<u8> {
        let dest = inst.dest.ok_or_else(|| CodegenError::Target {
            what: "expanding an operation",
            detail: String::from("the operation writes no register"),
        })?;
        match dest {
            Destination::Physical(preg) => match reg::gpr_number(preg) {
                Some(number) => Ok(number),
                None => Err(self.foreign_register_error(preg)),
            },
            Destination::Virtual(value) => Err(CodegenError::Target {
                what: "expanding an operation",
                detail: format!(
                    "the machine code was not allocated: the destination names the virtual register {}",
                    value.index()
                ),
            }),
        }
    }

    /// The operand of the instruction at the given position.
    fn arg(&self, inst: &MInst, index: usize) -> Result<Operand> {
        inst.args
            .get(index)
            .ok_or_else(|| CodegenError::Target {
                what: "expanding an operation",
                detail: String::from("the operation has too few operands"),
            })
    }

    /// The type of the values an instruction reads.
    fn operand_type(&self, inst: &MInst) -> MType {
        inst.operand_ty.clone().unwrap_or(MType::I32)
    }

    /// Emits the move of one register into another.
    fn emit_move(&mut self, rd: u8, rm: u8) {
        if rd == rm {
            return;
        }
        if rd < 8 && rm < 8 {
            encode::mov_reg(&mut self.bytes, rd, rm);
        } else {
            encode::mov_reg_wide(&mut self.bytes, rd, rm);
        }
        self.instructions += 1;
    }

    /// Emits the move of one register into another, even when the two
    /// registers are the same.
    ///
    /// A conditional block needs its shape, so the move of a register
    /// into itself is emitted as a move of the register into itself.
    fn emit_move_always(&mut self, rd: u8, rm: u8) {
        if rd != rm {
            self.emit_move(rd, rm);
            return;
        }
        if rd < 8 {
            encode::mov_reg(&mut self.bytes, rd, rm);
        } else {
            encode::mov_reg_wide(&mut self.bytes, rd, rm);
        }
        self.instructions += 1;
    }

    /// Emits the move of a constant into a register.
    fn emit_mov_imm(&mut self, rd: u8, value: u32) {
        if value <= 0xff && rd < 8 {
            encode::movs_imm(&mut self.bytes, rd, u8::try_from(value).unwrap_or(0));
            self.instructions += 1;
        } else if value <= 0xffff {
            encode::movw(&mut self.bytes, rd, u16::try_from(value).unwrap_or(0));
            self.instructions += 1;
        } else {
            encode::movw(&mut self.bytes, rd, u16::try_from(value & 0xffff).unwrap_or(0));
            encode::movt(
                &mut self.bytes,
                rd,
                u16::try_from((value >> 16) & 0xffff).unwrap_or(0),
            );
            self.instructions += 2;
        }
    }

    /// Emits the move of a value into a register.
    ///
    /// A constant is materialized in the scratch register, which the
    /// allocator never hands out.
    fn emit_value(&mut self, value: Value, rd: u8) {
        match value {
            Value::Reg(rm) => self.emit_move(rd, rm),
            Value::Imm(imm) => self.emit_mov_imm(rd, imm),
        }
    }

    /// Emits the addition of two registers.
    fn emit_add(&mut self, rd: u8, rn: u8, rm: u8) {
        if rd < 8 && rn < 8 && rm < 8 {
            encode::adds_reg3(&mut self.bytes, rd, rn, rm);
        } else {
            encode::adds_reg_wide(&mut self.bytes, rd, rn, rm);
        }
        self.instructions += 1;
    }

    /// Emits the addition of a register and a value.
    fn emit_add_value(&mut self, rd: u8, rn: u8, value: Value) {
        match value {
            Value::Reg(rm) => self.emit_add(rd, rn, rm),
            Value::Imm(imm) => {
                if rd == rn && rd < 8 && imm <= 0xff {
                    encode::adds_imm8(&mut self.bytes, rd, u8::try_from(imm).unwrap_or(0));
                    self.instructions += 1;
                } else {
                    self.emit_mov_imm(SCRATCH, imm);
                    self.emit_add(rd, rn, SCRATCH);
                }
            }
        }
    }

    /// Emits the addition of two values.
    fn emit_add_values(&mut self, rd: u8, left: Value, right: Value) {
        match (left, right) {
            (Value::Reg(rn), Value::Reg(rm)) => self.emit_add(rd, rn, rm),
            (Value::Reg(rn), Value::Imm(imm)) => self.emit_add_value(rd, rn, Value::Imm(imm)),
            (Value::Imm(imm), Value::Reg(rm)) => {
                self.emit_mov_imm(rd, imm);
                self.emit_add(rd, rd, rm);
            }
            (Value::Imm(left), Value::Imm(right)) => {
                self.emit_mov_imm(rd, left);
                self.emit_add_value(rd, rd, Value::Imm(right));
            }
        }
    }

    /// Emits the subtraction of two values.
    fn emit_sub_values(&mut self, rd: u8, left: Value, right: Value) {
        match (left, right) {
            (Value::Reg(rn), Value::Reg(rm)) => self.emit_sub(rd, rn, rm),
            (Value::Reg(rn), Value::Imm(imm)) => self.emit_sub_value(rd, rn, Value::Imm(imm)),
            (Value::Imm(imm), Value::Reg(rm)) => {
                self.emit_mov_imm(rd, imm);
                self.emit_sub(rd, rd, rm);
            }
            (Value::Imm(left), Value::Imm(right)) => {
                self.emit_mov_imm(rd, left);
                self.emit_sub_value(rd, rd, Value::Imm(right));
            }
        }
    }

    /// Emits the addition of a constant to a register.
    fn emit_add_offset(&mut self, rd: u8, offset: u32) {
        if offset == 0 {
            return;
        }
        if rd < 8 && offset <= 0xff {
            encode::adds_imm8(&mut self.bytes, rd, u8::try_from(offset).unwrap_or(0));
            self.instructions += 1;
            return;
        }
        self.emit_mov_imm(SCRATCH, offset);
        self.emit_add(rd, rd, SCRATCH);
    }

    /// Emits the subtraction of two registers.
    fn emit_sub(&mut self, rd: u8, rn: u8, rm: u8) {
        if rd < 8 && rn < 8 && rm < 8 {
            encode::subs_reg3(&mut self.bytes, rd, rn, rm);
        } else {
            encode::subs_reg_wide(&mut self.bytes, rd, rn, rm);
        }
        self.instructions += 1;
    }

    /// Emits the subtraction of a value from a register.
    fn emit_sub_value(&mut self, rd: u8, rn: u8, value: Value) {
        match value {
            Value::Reg(rm) => self.emit_sub(rd, rn, rm),
            Value::Imm(imm) => {
                if rd == rn && rd < 8 && imm <= 0xff {
                    encode::subs_imm8(&mut self.bytes, rd, u8::try_from(imm).unwrap_or(0));
                    self.instructions += 1;
                } else {
                    self.emit_mov_imm(SCRATCH, imm);
                    self.emit_sub(rd, rn, SCRATCH);
                }
            }
        }
    }

    /// Emits the comparison of two registers.
    fn emit_cmp(&mut self, rn: u8, rm: u8) {
        if rn < 8 && rm < 8 {
            encode::cmp_reg(&mut self.bytes, rn, rm);
        } else {
            encode::cmp_reg_wide(&mut self.bytes, rn, rm);
        }
        self.instructions += 1;
    }

    /// Emits the comparison of a register with a constant.
    fn emit_cmp_imm(&mut self, rn: u8, value: u32) {
        if rn < 8 && value <= 0xff {
            encode::cmp_imm(&mut self.bytes, rn, u8::try_from(value).unwrap_or(0));
            self.instructions += 1;
            return;
        }
        self.emit_mov_imm(SCRATCH, value);
        self.emit_cmp(rn, SCRATCH);
    }

    /// Emits the extension of a register.
    fn emit_extend(&mut self, rd: u8, rm: u8, signed: bool, halfword: bool) {
        if rd < 8 && rm < 8 {
            match (signed, halfword) {
                (false, false) => encode::uxtb_reg(&mut self.bytes, rd, rm),
                (false, true) => encode::uxth_reg(&mut self.bytes, rd, rm),
                (true, false) => encode::sxtb_reg(&mut self.bytes, rd, rm),
                (true, true) => encode::sxth_reg(&mut self.bytes, rd, rm),
            }
        } else {
            match (signed, halfword) {
                (false, false) => encode::uxtb_reg_wide(&mut self.bytes, rd, rm),
                (false, true) => encode::uxth_reg_wide(&mut self.bytes, rd, rm),
                (true, false) => encode::sxtb_reg_wide(&mut self.bytes, rd, rm),
                (true, true) => encode::sxth_reg_wide(&mut self.bytes, rd, rm),
            }
        }
        self.instructions += 1;
    }

    /// Emits a branch that waits for its displacement.
    fn emit_branch(&mut self, kind: FixupKind, target: FixupTarget) {
        let at = self.emit_placeholder();
        self.fixups.push(Fixup { at, kind, target });
        self.instructions += 1;
    }

    /// Emits the placeholder of a branch whose displacement is not
    /// known yet, and returns the place it occupies.
    fn emit_placeholder(&mut self) -> usize {
        let at = self.bytes.len();
        self.bytes.extend_from_slice(&[0, 0, 0, 0]);
        at
    }

    /// Emits the epilogue of the function.
    fn emit_epilogue(&mut self) {
        if self.frame != 0 {
            emit_frame_release(&mut self.bytes, self.frame);
            self.instructions += 1;
        }
        if self.uses_link {
            // The link register was saved with the other registers,
            // so the pop returns to the caller.
            encode::pop(&mut self.bytes, self.saved & 0x0fff, true);
            self.instructions += 1;
        } else {
            if self.saved != 0 {
                encode::pop(&mut self.bytes, self.saved & 0x0fff, false);
                self.instructions += 1;
            }
            encode::bx(&mut self.bytes, reg::LR);
            self.instructions += 1;
        }
    }

    /// Expands one operation.
    fn expand(&mut self, inst: &MInst) -> Result<()> {
        match &inst.op {
            MOp::Copy => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                self.emit_value(value, rd);
            }
            MOp::Load { size } => {
                let rd = self.dest(inst)?;
                let address = self.value(self.arg(inst, 0)?)?;
                self.emit_load(rd, address, *size)?;
            }
            MOp::Store { size } => {
                let address = self.value(self.arg(inst, 0)?)?;
                let value = self.value(self.arg(inst, 1)?)?;
                self.emit_store(address, value, *size)?;
            }
            MOp::Gep { size, signed } => {
                let rd = self.dest(inst)?;
                let base = self.value(self.arg(inst, 0)?)?;
                self.emit_gep(inst, rd, base, *size, *signed)?;
            }
            MOp::VecSplat | MOp::VecExtract { .. } | MOp::VecInsert { .. } | MOp::VecShuffle { .. } => {
                return Err(CodegenError::Unsupported {
                    what: "a vector operation",
                    detail: String::from("the architecture has no vector unit"),
                });
            }
            MOp::Add => {
                let rd = self.dest(inst)?;
                let left = self.value(self.arg(inst, 0)?)?;
                let right = self.value(self.arg(inst, 1)?)?;
                self.emit_add_values(rd, left, right);
            }
            MOp::Sub => {
                let rd = self.dest(inst)?;
                let left = self.value(self.arg(inst, 0)?)?;
                let right = self.value(self.arg(inst, 1)?)?;
                self.emit_sub_values(rd, left, right);
            }
            MOp::Mul => {
                let rd = self.dest(inst)?;
                let left = self.value(self.arg(inst, 0)?)?;
                let right = self.value(self.arg(inst, 1)?)?;
                let rn = self.materialize(left, rd);
                let rm = self.materialize(right, SCRATCH);
                encode::muls_reg_wide(&mut self.bytes, rd, rn, rm);
                self.instructions += 1;
            }
            MOp::And | MOp::Or | MOp::Xor => {
                let rd = self.dest(inst)?;
                let left = self.value(self.arg(inst, 0)?)?;
                let right = self.value(self.arg(inst, 1)?)?;
                let kind = match inst.op {
                    MOp::And => Logic::And,
                    MOp::Or => Logic::Or,
                    _ => Logic::Xor,
                };
                let rn = self.materialize(left, rd);
                let rm = self.materialize(right, SCRATCH);
                self.emit_logic(rd, rn, rm, kind);
            }
            MOp::Shl | MOp::LShr | MOp::AShr => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                let amount = self.value(self.arg(inst, 1)?)?;
                let rm = self.materialize(value, rd);
                let rs = self.materialize(amount, SCRATCH);
                match inst.op {
                    MOp::Shl => encode::lsls_reg(&mut self.bytes, rd, rm, rs),
                    MOp::LShr => encode::lsrs_reg(&mut self.bytes, rd, rm, rs),
                    _ => encode::asrs_reg(&mut self.bytes, rd, rm, rs),
                }
                self.instructions += 1;
            }
            MOp::Neg => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                let rm = self.materialize(value, rd);
                if rd < 8 && rm < 8 {
                    encode::negs_reg(&mut self.bytes, rd, rm);
                } else {
                    encode::negs_reg_wide(&mut self.bytes, rd, rm);
                }
                self.instructions += 1;
            }
            MOp::Not => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                let rm = self.materialize(value, rd);
                if rd < 8 && rm < 8 {
                    encode::mvns_reg(&mut self.bytes, rd, rm);
                } else {
                    encode::mvn_reg_wide(&mut self.bytes, rd, rm);
                }
                self.instructions += 1;
            }
            MOp::UDiv | MOp::URem | MOp::SDiv | MOp::SRem => {
                let rd = self.dest(inst)?;
                let dividend = self.value(self.arg(inst, 0)?)?;
                let divisor = self.value(self.arg(inst, 1)?)?;
                self.emit_division(inst, rd, dividend, divisor);
            }
            MOp::ICmp { pred } => {
                let rd = self.dest(inst)?;
                let left = self.value(self.arg(inst, 0)?)?;
                let right = self.value(self.arg(inst, 1)?)?;
                self.emit_compare(rd, left, right, *pred);
            }
            MOp::FCmp { .. } => {
                return Err(self.floating_point_error("a floating-point comparison"));
            }
            MOp::FAdd
            | MOp::FSub
            | MOp::FMul
            | MOp::FDiv
            | MOp::FRem
            | MOp::FNeg
            | MOp::FSqrt
            | MOp::FAbs
            | MOp::FMin
            | MOp::FMax
            | MOp::FDim
            | MOp::FCopySign => {
                return Err(self.floating_point_error("a floating-point operation"));
            }
            MOp::Select => {
                let rd = self.dest(inst)?;
                let condition = self.value(self.arg(inst, 0)?)?;
                let yes = self.value(self.arg(inst, 1)?)?;
                let no = self.value(self.arg(inst, 2)?)?;
                self.emit_select(rd, condition, yes, no);
            }
            MOp::ZExt | MOp::SExt => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                let rm = self.materialize(value, rd);
                let ty = self.operand_type(inst);
                let signed = matches!(inst.op, MOp::SExt);
                self.emit_extend(rd, rm, signed, ty == MType::I16);
            }
            MOp::Trunc => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                let rm = self.materialize(value, rd);
                match inst.ty {
                    MType::I8 | MType::I1 => self.emit_extend(rd, rm, false, false),
                    MType::I16 => self.emit_extend(rd, rm, false, true),
                    _ => self.emit_move(rd, rm),
                }
            }
            MOp::FPToUI | MOp::FPToSI | MOp::UIToFP | MOp::SIToFP | MOp::FPTrunc | MOp::FPExt => {
                return Err(self.floating_point_error("a floating-point conversion"));
            }
            MOp::Bitcast => {
                let rd = self.dest(inst)?;
                let value = self.value(self.arg(inst, 0)?)?;
                if inst.ty.is_floating() || self.operand_type(inst).is_floating() {
                    return Err(self.floating_point_error("a bit-level reinterpretation"));
                }
                self.emit_value(value, rd);
            }
            MOp::Br { target } => {
                self.emit_branch(FixupKind::Branch, FixupTarget::Block(*target));
            }
            MOp::CondBr => {
                let condition = self.value(self.arg(inst, 0)?)?;
                let yes = self.block_target(self.arg(inst, 1)?)?;
                let no = self.block_target(self.arg(inst, 2)?)?;
                self.emit_conditional_branch(condition, yes, no);
            }
            MOp::CondBrCmp { pred } => {
                // The comparison writes its one-bit result into the
                // scratch register, which the branch then tests: the
                // architecture branches on a value, not on the flags.
                let first = self.value(self.arg(inst, 0)?)?;
                let second = self.value(self.arg(inst, 1)?)?;
                let yes = self.block_target(self.arg(inst, 2)?)?;
                let no = self.block_target(self.arg(inst, 3)?)?;
                self.emit_compare(SCRATCH_VALUE, first, second, *pred);
                self.emit_conditional_branch(Value::Reg(SCRATCH_VALUE), yes, no);
            }
            MOp::Ret => {
                let value = self.value(self.arg(inst, 0)?)?;
                self.emit_value(value, 0);
                self.emit_epilogue();
            }
            MOp::RetVoid => {
                self.emit_epilogue();
            }
            MOp::Unreachable => {
                encode::undefined(&mut self.bytes);
                self.instructions += 1;
            }
            MOp::Call { callee } => {
                let index = self
                    .callees
                    .get(callee.index())
                    .copied()
                    .flatten()
                    .ok_or_else(|| CodegenError::Unsupported {
                        what: "a call",
                        detail: String::from(
                            "the callee is not a function of the module, so it has no relative address",
                        ),
                    })?;
                let count = inst.args.len();
                if count > 4 {
                    return Err(CodegenError::Unsupported {
                        what: "a call",
                        detail: String::from("the architecture passes four arguments in registers"),
                    });
                }
                for position in 0..count {
                    let value = self.value(self.arg(inst, position)?)?;
                    let target = u8::try_from(position).unwrap_or(0);
                    self.emit_value(value, target);
                }
                self.uses_link = true;
                self.emit_branch(FixupKind::BranchLink, FixupTarget::Function(index));
                if let Some(rd) = inst.dest_reg()
                    && let Some(number) = reg::gpr_number(rd)
                {
                    self.emit_move(number, 0);
                }
            }
            MOp::WorkItemLoad { field, lane } => {
                let rd = self.dest(inst)?;
                let pointer = self.value(self.arg(inst, 0)?)?;
                let offset = field.offset() + lane * 4;
                self.emit_load_offset(rd, pointer, offset, 4);
            }
            MOp::ArgBlockBase => {
                let rd = self.dest(inst)?;
                let pointer = self.value(self.arg(inst, 0)?)?;
                self.emit_load_offset(rd, pointer, ARG_BASE_OFFSET, 4);
            }
            MOp::ArgLoad { index, kind } => {
                let rd = self.dest(inst)?;
                let base = self.value(self.arg(inst, 0)?)?;
                let offset = index * ARG_SLOT_SIZE;
                match kind {
                    ArgKind::Pointer => self.emit_load_offset(rd, base, offset, 4),
                    ArgKind::Scalar { size } => self.emit_load_offset(rd, base, offset, *size),
                }
            }
            MOp::Param { index } => {
                let rd = self.dest(inst)?;
                if *index >= 4 {
                    return Err(CodegenError::Unsupported {
                        what: "a parameter",
                        detail: String::from("the architecture passes four parameters in registers"),
                    });
                }
                self.emit_move(rd, u8::try_from(*index).unwrap_or(0));
            }
            MOp::FrameAddr { offset } => {
                let rd = self.dest(inst)?;
                self.emit_frame_address(rd, *offset);
            }
            MOp::SpillStore { offset, size } => {
                let value = self.value(self.arg(inst, 0)?)?;
                self.emit_store_stack(value, *offset, *size);
            }
            MOp::SpillLoad { offset, size } => {
                let rd = self.dest(inst)?;
                self.emit_load_stack(rd, *offset, *size);
            }
            MOp::FrameSetup | MOp::FrameTeardown => {
                // The prologue and the epilogue of the function take
                // care of the frame.
            }
        }
        Ok(())
    }

    /// Puts a value in a register.
    ///
    /// A constant is materialized in the register the caller names,
    /// which is either the destination of the instruction or the
    /// scratch register.
    fn materialize(&mut self, value: Value, rd: u8) -> u8 {
        match value {
            Value::Reg(rm) => rm,
            Value::Imm(imm) => {
                self.emit_mov_imm(rd, imm);
                rd
            }
        }
    }

    /// The block an operand names.
    fn block_target(&self, operand: Operand) -> Result<u32> {
        match operand {
            Operand::Immediate(value) => u32::try_from(value).map_err(|_| CodegenError::Target {
                what: "expanding a conditional branch",
                detail: String::from("the target block does not fit in a number"),
            }),
            _ => Err(CodegenError::Target {
                what: "expanding a conditional branch",
                detail: String::from("the target block is not a number"),
            }),
        }
    }

    /// The error the back end returns for a floating-point operation.
    fn floating_point_error(&self, what: &'static str) -> CodegenError {
        if self.has_fp {
            CodegenError::Unsupported {
                what,
                detail: String::from(
                    "the back end does not implement the floating-point instructions of the unit",
                ),
            }
        } else {
            CodegenError::Unsupported {
                what,
                detail: String::from("the core of the target has no floating-point unit"),
            }
        }
    }

    /// Emits the logical operation of two registers.
    fn emit_logic(&mut self, rd: u8, rn: u8, rm: u8, kind: Logic) {
        if rd < 8 && rn < 8 && rm < 8 && rd == rn {
            // The short form names the destination and one source
            // only, so it fits when the destination is the first
            // operand.
            match kind {
                Logic::And => encode::ands_reg(&mut self.bytes, rd, rm),
                Logic::Or => encode::orrs_reg(&mut self.bytes, rd, rm),
                Logic::Xor => encode::eors_reg(&mut self.bytes, rd, rm),
            }
        } else {
            match kind {
                Logic::And => encode::ands_reg_wide(&mut self.bytes, rd, rn, rm),
                Logic::Or => encode::orrs_reg_wide(&mut self.bytes, rd, rn, rm),
                Logic::Xor => encode::eors_reg_wide(&mut self.bytes, rd, rn, rm),
            }
        }
        self.instructions += 1;
    }

    /// Emits the load of a value.
    fn emit_load(&mut self, rd: u8, address: Value, size: u32) -> Result<()> {
        match size {
            4 | 1 | 2 => {
                self.emit_load_offset(rd, address, 0, size);
                Ok(())
            }
            _ => Err(CodegenError::Unsupported {
                what: "a load",
                detail: String::from(
                    "the machine register holds one word, so a load of eight bytes has no home",
                ),
            }),
        }
    }

    /// Emits the load of a value at an offset.
    fn emit_load_offset(&mut self, rd: u8, address: Value, offset: u32, size: u32) {
        let rn = self.materialize(address, SCRATCH);
        match size {
            1 => {
                if offset <= 31 {
                    encode::ldrb_imm(&mut self.bytes, rd, rn, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::ldrb_imm_wide(&mut self.bytes, rd, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset.is_multiple_of(2) {
                    encode::ldrh_imm(&mut self.bytes, rd, rn, u8::try_from(offset / 2).unwrap_or(0));
                } else {
                    encode::ldrh_imm_wide(&mut self.bytes, rd, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 124 && offset.is_multiple_of(4) {
                    encode::ldr_imm(&mut self.bytes, rd, rn, u8::try_from(offset / 4).unwrap_or(0));
                } else {
                    encode::ldr_imm_wide(&mut self.bytes, rd, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
        }
        self.instructions += 1;
    }

    /// Emits the store of a value.
    fn emit_store(&mut self, address: Value, value: Value, size: u32) -> Result<()> {
        match size {
            4 | 1 | 2 => {
                self.emit_store_offset(address, value, 0, size);
                Ok(())
            }
            _ => Err(CodegenError::Unsupported {
                what: "a store",
                detail: String::from("the machine register holds one word"),
            }),
        }
    }

    /// Emits the store of a value at an offset.
    fn emit_store_offset(&mut self, address: Value, value: Value, offset: u32, size: u32) {
        let rn = self.materialize(address, SCRATCH);
        let rm = self.materialize(value, SCRATCH_VALUE);
        match size {
            1 => {
                if offset <= 31 {
                    encode::strb_imm(&mut self.bytes, rm, rn, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::strb_imm_wide(&mut self.bytes, rm, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset.is_multiple_of(2) {
                    encode::strh_imm(&mut self.bytes, rm, rn, u8::try_from(offset / 2).unwrap_or(0));
                } else {
                    encode::strh_imm_wide(&mut self.bytes, rm, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 124 && offset.is_multiple_of(4) {
                    encode::str_imm(&mut self.bytes, rm, rn, u8::try_from(offset / 4).unwrap_or(0));
                } else {
                    encode::str_imm_wide(&mut self.bytes, rm, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
        }
        self.instructions += 1;
    }

    /// Emits the load of a value from the frame.
    fn emit_load_stack(&mut self, rd: u8, offset: u32, size: u32) {
        match size {
            1 => {
                if offset <= 31 {
                    encode::ldrb_imm(&mut self.bytes, rd, reg::SP, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::ldrb_imm_wide(&mut self.bytes, rd, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset.is_multiple_of(2) {
                    encode::ldrh_imm(
                        &mut self.bytes,
                        rd,
                        reg::SP,
                        u8::try_from(offset / 2).unwrap_or(0),
                    );
                } else {
                    encode::ldrh_imm_wide(&mut self.bytes, rd, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 1020 && offset.is_multiple_of(4) {
                    encode::ldr_sp(&mut self.bytes, rd, u8::try_from(offset / 4).unwrap_or(0));
                } else {
                    encode::ldr_sp_wide(&mut self.bytes, rd, u16::try_from(offset).unwrap_or(0));
                }
            }
        }
        self.instructions += 1;
    }

    /// Emits the store of a value into the frame.
    fn emit_store_stack(&mut self, value: Value, offset: u32, size: u32) {
        let rm = self.materialize(value, SCRATCH);
        match size {
            1 => {
                if offset <= 31 {
                    encode::strb_imm(&mut self.bytes, rm, reg::SP, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::strb_imm_wide(&mut self.bytes, rm, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset.is_multiple_of(2) {
                    encode::strh_imm(
                        &mut self.bytes,
                        rm,
                        reg::SP,
                        u8::try_from(offset / 2).unwrap_or(0),
                    );
                } else {
                    encode::strh_imm_wide(&mut self.bytes, rm, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 1020 && offset.is_multiple_of(4) {
                    encode::str_sp(&mut self.bytes, rm, u8::try_from(offset / 4).unwrap_or(0));
                } else {
                    encode::str_sp_wide(&mut self.bytes, rm, u16::try_from(offset).unwrap_or(0));
                }
            }
        }
        self.instructions += 1;
    }

    /// Emits the address of a frame slot.
    fn emit_frame_address(&mut self, rd: u8, offset: u32) {
        if offset <= 1020 && offset.is_multiple_of(4) {
            encode::add_sp_imm(&mut self.bytes, rd, u16::try_from(offset).unwrap_or(0));
        } else {
            self.emit_mov_imm(SCRATCH, offset);
            encode::adds_reg_wide(&mut self.bytes, rd, reg::SP, SCRATCH);
            self.instructions += 1;
        }
        self.instructions += 1;
    }

    /// Emits the computation of an address from a base and indices.
    fn emit_gep(&mut self, inst: &MInst, rd: u8, base: Value, size: u32, signed: bool) -> Result<()> {
        let count = inst.args.len();
        if count == 0 {
            self.emit_value(base, rd);
            return Ok(());
        }
        let shift = match size {
            1 => Some(0),
            2 => Some(1),
            4 => Some(2),
            8 => Some(3),
            _ => None,
        };
        let index_type = self.operand_type(inst);
        self.emit_value(base, rd);
        for position in 1..count {
            let operand = self.value(self.arg(inst, position)?)?;
            match operand {
                Value::Imm(imm) => {
                    // The constant is scaled by the size of the
                    // element and added to the address.
                    self.emit_add_offset(rd, imm.wrapping_mul(size));
                }
                Value::Reg(index) => {
                    let index = if signed && matches!(index_type, MType::I8 | MType::I16) {
                        // The index is signed, so it has to be widened
                        // before it is scaled.
                        self.emit_move(SCRATCH, index);
                        self.emit_extend(SCRATCH, SCRATCH, true, index_type == MType::I16);
                        SCRATCH
                    } else {
                        index
                    };
                    match shift {
                        Some(0) => self.emit_add(rd, rd, index),
                        Some(shift) => {
                            encode::adds_reg_shifted(&mut self.bytes, rd, rd, index, shift);
                            self.instructions += 1;
                        }
                        None => {
                            // The element size is not a power of two,
                            // so the index is multiplied by it.
                            self.emit_mov_imm(SCRATCH, size);
                            encode::muls_reg_wide(&mut self.bytes, SCRATCH, SCRATCH, index);
                            self.instructions += 1;
                            self.emit_add(rd, rd, SCRATCH);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Emits the comparison of two values into a one-bit value.
    fn emit_compare(&mut self, rd: u8, left: Value, right: Value, pred: IPred) {
        let compare = match (left, right) {
            (Value::Reg(rn), Value::Reg(rm)) => Compare::Regs(rn, rm),
            (Value::Reg(rn), Value::Imm(imm)) => Compare::RegImm(rn, imm),
            (Value::Imm(imm), Value::Reg(rm)) => Compare::ImmReg(imm, rm),
            (Value::Imm(left), Value::Imm(right)) => Compare::Imms(left, right),
        };
        let condition = condition_code(pred);
        // Two constants are compared here, so the result is known
        // before the code is written.
        if let Compare::Imms(left, right) = compare {
            let holds = evaluate(pred, left, right);
            self.emit_mov_imm(rd, u32::from(holds));
            return;
        }
        // The value is cleared before the comparison, because the
        // comparison sets the flags the conditional block reads.
        self.emit_mov_imm(rd, 0);
        match compare {
            Compare::Regs(rn, rm) => self.emit_cmp(rn, rm),
            Compare::RegImm(rn, imm) => self.emit_cmp_imm(rn, imm),
            Compare::ImmReg(imm, rm) => {
                self.emit_mov_imm(SCRATCH, imm);
                self.emit_cmp(SCRATCH, rm);
            }
            Compare::Imms(_, _) => {}
        }
        encode::it(&mut self.bytes, condition, encode::IT_ONE);
        self.instructions += 1;
        self.emit_mov_imm_in_conditional(rd, 1);
    }

    /// Emits the move of an immediate inside a conditional block.
    ///
    /// The move of an immediate in the short form sets the flags, so
    /// the wide form, which does not, is the one the conditional
    /// block uses.
    fn emit_mov_imm_in_conditional(&mut self, rd: u8, value: u32) {
        encode::movw(&mut self.bytes, rd, u16::try_from(value).unwrap_or(0));
        self.instructions += 1;
    }

    /// Emits the selection between two values.
    fn emit_select(&mut self, rd: u8, condition: Value, yes: Value, no: Value) {
        // A constant condition is known here, so only the value that
        // the condition picks is written.
        if let Value::Imm(value) = condition {
            if value != 0 {
                self.emit_value(yes, rd);
            } else {
                self.emit_value(no, rd);
            }
            return;
        }
        let Value::Reg(test) = condition else {
            return;
        };
        // A conditional block holds one instruction per part, so the
        // values of the parts have to live in registers already.
        let yes = self.block_value(yes, SCRATCH);
        let no = self.block_value(no, SCRATCH_VALUE);
        self.emit_cmp_imm(test, 0);
        encode::it(&mut self.bytes, encode::COND_NE, encode::IT_THEN_ELSE);
        self.instructions += 1;
        self.emit_move_always(rd, yes);
        self.emit_move_always(rd, no);
    }

    /// Puts the value of a part of a conditional block in a register.
    fn block_value(&mut self, value: Value, scratch: u8) -> u8 {
        match value {
            Value::Reg(rm) => rm,
            Value::Imm(imm) => {
                self.emit_mov_imm(scratch, imm);
                scratch
            }
        }
    }

    /// Emits a conditional branch on a value.
    fn emit_conditional_branch(&mut self, condition: Value, yes: u32, no: u32) {
        match condition {
            Value::Imm(value) => {
                if value != 0 {
                    self.emit_branch(FixupKind::Branch, FixupTarget::Block(yes));
                } else {
                    self.emit_branch(FixupKind::Branch, FixupTarget::Block(no));
                }
            }
            Value::Reg(rm) => {
                self.emit_cmp_imm(rm, 0);
                self.emit_branch(FixupKind::BranchCond(encode::COND_NE), FixupTarget::Block(yes));
                self.emit_branch(FixupKind::Branch, FixupTarget::Block(no));
            }
        }
    }

    /// Emits the division of two values.
    ///
    /// The architecture has no division instruction, so the division
    /// becomes a loop that shifts the dividend into the remainder.
    /// The loop works in the four argument registers, which the
    /// allocator never hands out.
    fn emit_division(&mut self, inst: &MInst, rd: u8, dividend: Value, divisor: Value) {
        let signed = matches!(inst.op, MOp::SDiv | MOp::SRem);
        let remainder = matches!(inst.op, MOp::URem | MOp::SRem);
        // The dividend goes into the first register and the divisor
        // into the second one, because the division loop works on
        // them.
        self.emit_value(dividend, 0);
        self.emit_value(divisor, 1);
        if signed {
            self.emit_signed_division();
        } else {
            self.emit_unsigned_division();
        }
        let value = if remainder { 2 } else { 0 };
        self.emit_move(rd, value);
    }

    /// Emits the signed division of the first two registers.
    ///
    /// The signs of the operands, the magnitudes and the sign of the
    /// quotient are kept in the saved registers.
    fn emit_signed_division(&mut self) {
        // The signs of the two operands.
        encode::asrs_imm(&mut self.bytes, 2, 0, 31);
        encode::asrs_imm(&mut self.bytes, 3, 1, 31);
        self.instructions += 2;
        // The magnitudes of the two operands.
        encode::eors_reg(&mut self.bytes, 0, 2);
        encode::subs_reg3(&mut self.bytes, 0, 0, 2);
        encode::eors_reg(&mut self.bytes, 1, 3);
        encode::subs_reg3(&mut self.bytes, 1, 1, 3);
        self.instructions += 4;
        // The division loop needs the two registers the signs sit in,
        // so the signs go on the stack.
        encode::push(&mut self.bytes, 0x0c, false);
        self.instructions += 1;
        self.emit_unsigned_division_body();
        // The remainder moves out of the way of the signs.
        encode::mov_reg(&mut self.bytes, 1, 2);
        self.instructions += 1;
        encode::pop(&mut self.bytes, 0x0c, false);
        self.instructions += 1;
        // The quotient takes the sign of the operands, the remainder
        // the sign of the dividend.  A value with the sign of a mask
        // is the value exclusive-or the mask, less the mask.
        encode::eors_reg(&mut self.bytes, 3, 2);
        encode::eors_reg(&mut self.bytes, 0, 3);
        encode::subs_reg3(&mut self.bytes, 0, 0, 3);
        self.instructions += 3;
        encode::eors_reg(&mut self.bytes, 1, 2);
        encode::subs_reg3(&mut self.bytes, 1, 1, 2);
        encode::mov_reg(&mut self.bytes, 2, 1);
        self.instructions += 3;
    }

    /// Emits the unsigned division of the first two registers.
    fn emit_unsigned_division(&mut self) {
        self.emit_unsigned_division_body();
    }

    /// Emits the body of the unsigned division.
    ///
    /// The first register holds the dividend, the second the divisor;
    /// the quotient ends up in the first and the remainder in the
    /// third.
    fn emit_unsigned_division_body(&mut self) {
        // The remainder and the counter of the loop.
        encode::movs_imm(&mut self.bytes, 2, 0);
        encode::movs_imm(&mut self.bytes, 3, 32);
        self.instructions += 2;
        let top = self.bytes.len();
        // The dividend is shifted into the remainder, and the bit
        // that falls out of the shift is added to it.
        encode::adds_reg3(&mut self.bytes, 0, 0, 0);
        encode::adcs_reg(&mut self.bytes, 2, 2);
        encode::cmp_reg(&mut self.bytes, 2, 1);
        self.instructions += 3;
        // The remainder is at least the divisor: take one back.
        let skip_at = self.emit_placeholder();
        encode::subs_reg3(&mut self.bytes, 2, 2, 1);
        encode::adds_imm8(&mut self.bytes, 0, 1);
        self.instructions += 2;
        // The counter of the loop.
        let end = self.bytes.len();
        encode::subs_imm8(&mut self.bytes, 3, 1);
        self.instructions += 1;
        // The branch that closes the loop is a placeholder, because
        // its displacement is only known once the blocks are laid
        // out.
        let branch_at = self.emit_placeholder();
        self.fixups.push(Fixup {
            at: branch_at,
            kind: FixupKind::BranchCond(encode::COND_NE),
            target: FixupTarget::Local(top),
        });
        self.fixups.push(Fixup {
            at: skip_at,
            kind: FixupKind::BranchCond(encode::COND_CC),
            target: FixupTarget::Local(end),
        });
        self.instructions += 1;
    }
}

/// The logical operations the expansion knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Logic {
    /// The bitwise and.
    And,
    /// The bitwise or.
    Or,
    /// The bitwise exclusive or.
    Xor,
}

/// The condition code of an integer comparison.
fn condition_code(pred: IPred) -> u8 {
    match pred {
        IPred::Eq => encode::COND_EQ,
        IPred::Ne => encode::COND_NE,
        IPred::Ugt => encode::COND_HI,
        IPred::Uge => encode::COND_CS,
        IPred::Ult => encode::COND_CC,
        IPred::Ule => encode::COND_LS,
        IPred::Sgt => encode::COND_GT,
        IPred::Sge => encode::COND_GE,
        IPred::Slt => encode::COND_LT,
        IPred::Sle => encode::COND_LE,
    }
}

/// The result of a comparison of two constants.
fn evaluate(pred: IPred, left: u32, right: u32) -> bool {
    match pred {
        IPred::Eq => left == right,
        IPred::Ne => left != right,
        IPred::Ugt => left > right,
        IPred::Uge => left >= right,
        IPred::Ult => left < right,
        IPred::Ule => left <= right,
        IPred::Sgt => (left as i32) > (right as i32),
        IPred::Sge => (left as i32) >= (right as i32),
        IPred::Slt => (left as i32) < (right as i32),
        IPred::Sle => (left as i32) <= (right as i32),
    }
}

/// Rounds the size of the frame up so that the stack pointer keeps
/// its alignment.
///
/// The frame is a whole number of words, and the saved registers and
/// the frame together are a multiple of eight bytes.
fn align_frame(size: u32, pushed: u32) -> u32 {
    let mut frame = size;
    while !frame.is_multiple_of(4) {
        frame += 4;
    }
    while !(frame + 4 * pushed).is_multiple_of(8) {
        frame += 4;
    }
    frame
}

/// The expansion of a whole function.
pub fn expand_function(
    function: &MirFunction,
    callees: &[Option<usize>],
    has_fp: bool,
) -> Result<FunctionCode> {
    let (saved, uses_link) = scan_registers(function);
    let mut expander = Expander::new(function, callees, has_fp, saved, uses_link);
    let mut blocks = Vec::with_capacity(function.blocks.len());

    for block in &function.blocks {
        expander.bytes.clear();
        expander.fixups.clear();
        expander.instructions = 0;
        for inst in &block.insts {
            expander.expand(inst)?;
        }
        blocks.push(BlockCode {
            bytes: expander.bytes.clone(),
            fixups: expander.fixups.clone(),
            instructions: expander.instructions,
        });
    }

    let prologue = build_prologue(&expander);
    let instructions = blocks
        .iter()
        .map(|block| block.instructions)
        .sum::<u32>()
        + u32::try_from(prologue.len() / 2).unwrap_or(0);

    Ok(FunctionCode {
        blocks,
        prologue,
        instructions,
    })
}

/// Builds the prologue of the function.
///
/// The prologue saves the callee-saved registers the function uses,
/// and then reserves the frame.  The frame base is the stack pointer
/// right after that subtraction, so every slot of the frame is at a
/// known distance from the stack pointer, and the saved registers sit
/// above the frame.
fn build_prologue(expander: &Expander<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    let saved = expander.saved & 0x0fff;
    if saved != 0 || expander.uses_link {
        encode::push(&mut out, saved, expander.uses_link);
    }
    if expander.frame != 0 {
        emit_frame_reserve(&mut out, expander.frame);
    }
    out
}

/// Emits the reservation of the frame.
///
/// The frame is subtracted from the stack pointer in pieces of at
/// most a kilobyte, because the immediate of the short form reaches
/// that far.  The frame base is the stack pointer after the last
/// subtraction, so the frame has the size the machine code asked for.
fn emit_frame_reserve(out: &mut Vec<u8>, frame: u32) {
    let mut rest = frame;
    while rest > 1020 {
        encode::sub_sp_imm(out, 1020);
        rest -= 1020;
    }
    if rest != 0 {
        encode::sub_sp_imm(out, rest);
    }
}

/// Emits the release of the frame.
fn emit_frame_release(out: &mut Vec<u8>, frame: u32) {
    let mut rest = frame;
    while rest > 1020 {
        encode::add_sp_frame(out, 1020);
        rest -= 1020;
    }
    if rest != 0 {
        encode::add_sp_frame(out, rest);
    }
}
