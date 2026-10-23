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
//! Every operation of the machine IR names what it computes; this
//! module decides how the architecture computes it.  An operation the
//! architecture has a single instruction for becomes that
//! instruction; an operation it does not becomes a short sequence:
//! a division loads a register pair, a selection compares and moves
//! conditionally, a narrow load loads and widens.
//!
//! # The registers of the operands
//!
//! The machine IR names the values of an operation with virtual
//! registers while the machine code is being built.  The allocator
//! gives every value a home before the back end sees the module, so
//! the destination of an instruction and the operands of an
//! instruction name the physical registers of the register model.  The
//! class of a register — below [`reg::XMM_BASE`] or above it — tells
//! whether the value it holds is an integer or a floating-point one.
//! The register model of the target never offers the argument
//! registers, the accumulator, or the two scratch vector registers to
//! the allocator, so the expander may use them freely.
//!
//! # The width of an operand
//!
//! The machine IR gives the type of the value an operation writes and
//! the type of the values it reads, so the expander never has to guess
//! the width of an operand from the register that holds it: a
//! comparison of two values compares them at the width the
//! instruction names, and a conversion reads its operand at the width
//! it was produced at.
//!
//! # The integers
//!
//! An integer value lives in a general-purpose register.  A value of
//! thirty-two bits or fewer keeps its bits in the low half of the
//! register and the bits above them are unspecified, so every
//! operation that reads a narrow value reads only the bits that hold
//! it: a comparison of two thirty-two-bit values compares their low
//! halves, and a sign extension reads the low half and keeps its
//! sign.
//!
//! # The frame
//!
//! The expander owns the frame of the function: it pushes the
//! callee-saved registers the function uses, reserves the frame with
//! a single subtraction, and undoes both in the epilogue.  The frame
//! base is the stack pointer right after the subtraction, so every
//! slot of the frame is at a known distance from the stack pointer.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_abi::{ARG_BASE_OFFSET, ARG_SLOT};
use codevar_ocl_native_codegen::error::CodegenError;
use codevar_ocl_native_codegen::mir::{
    ArgKind, Destination, FPred, IPred, MInst, MOp, MType, MirFunction, MirModule, Operand, SymId,
};

use crate::encode::{Arith, Cc, Encoder, Size};
use crate::layout::Fixup;
use crate::reg;

/// The result of the expansion of a function.
pub(crate) struct Expanded {
    /// The bytes of the function.
    pub bytes: Vec<u8>,
    /// The places whose displacement the layout has to fill in.
    pub fixups: Vec<Fixup>,
    /// The offset of every block inside the function.
    pub block_offsets: Vec<u32>,
    /// The number of instructions the function holds.
    pub instruction_count: u32,
}

/// The class of a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// An integer or a pointer.
    Int,
    /// A floating-point value or a vector.
    Float,
}

/// The expander of one function.
struct Expander<'a> {
    /// The module the function belongs to.
    module: &'a MirModule,
    /// The function being expanded.
    function: &'a MirFunction,
    /// The callee-saved registers the function saves.
    saved: Vec<u16>,
    /// The size of the reserved frame.
    frame_size: u32,
}

/// Builds the error of a rejected machine operation.
fn target_error(what: &'static str, detail: &str) -> CodegenError {
    CodegenError::Target {
        what,
        detail: String::from(detail),
    }
}

/// Builds the error of a machine operation the target cannot express.
fn unsupported_error(what: &'static str, detail: &str) -> CodegenError {
    CodegenError::Unsupported {
        what,
        detail: String::from(detail),
    }
}

/// The size of the general-purpose operations on a machine type.
fn int_size(ty: &MType) -> Size {
    match ty {
        MType::I64 | MType::Ptr { .. } => Size::Qword,
        _ => Size::Dword,
    }
}

/// The size of the floating-point operations on a machine type.
fn float_size(ty: &MType) -> Option<Size> {
    match ty {
        MType::F32 => Some(Size::Dword),
        MType::F64 => Some(Size::Qword),
        _ => None,
    }
}

/// The size of one lane of a vector type.
fn lane_size(ty: &MType) -> Option<Size> {
    match ty {
        MType::Vec { elem, .. } => match elem.as_ref() {
            MType::F32 | MType::I32 => Some(Size::Dword),
            MType::F64 | MType::I64 => Some(Size::Qword),
            MType::F16 | MType::I16 => Some(Size::Word),
            MType::I8 => Some(Size::Byte),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> Expander<'a> {
    /// Creates the expander of a function.
    fn new(module: &'a MirModule, function: &'a MirFunction) -> Self {
        let saved = saved_registers(function);
        let frame_size = frame_size(function);
        Self {
            module,
            function,
            saved,
            frame_size,
        }
    }

    /// The register the instruction writes.
    ///
    /// The allocator gives every value a home before the back end
    /// sees it, so the destination of an instruction is a physical
    /// register.  A destination that is still virtual means the
    /// module never went through the allocator.
    fn dest(&self, inst: &MInst) -> Result<u16, CodegenError> {
        match inst.dest {
            Some(Destination::Physical(reg)) => {
                let number = reg.number();
                if !reg::is_valid(number) {
                    return Err(target_error(
                        "expanding the machine operation",
                        &format!(
                            "the destination names register {number}, which the register file does not hold"
                        ),
                    ));
                }
                Ok(number)
            }
            Some(Destination::Virtual(_)) => Err(target_error(
                "expanding the machine operation",
                "the destination of the operation is still a virtual register",
            )),
            None => Err(target_error(
                "expanding the machine operation",
                "the operation writes no register",
            )),
        }
    }

    /// The register an operand names.
    ///
    /// An immediate that the operation needs in a register is moved
    /// into a scratch register first.  The scratch registers are the
    /// ones the allocator never hands out, so they hold nothing the
    /// operation needs; the registers in `reserved` are the ones the
    /// expansion itself uses and are left alone.
    fn operand_reg(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        index: usize,
        reserved: &[u16],
    ) -> Result<u16, CodegenError> {
        match inst.args.get(index) {
            Some(Operand::Physical(reg)) => Ok(reg.number()),
            Some(Operand::Virtual(_)) => Err(target_error(
                "expanding the machine operation",
                &format!("operand {index} of the operation is still a virtual register"),
            )),
            Some(Operand::Immediate(value)) => {
                let scratch = self.scratch_register(inst, reserved)?;
                let size = int_size(Self::operand_type(inst).unwrap_or(&inst.ty));
                enc.mov_ri(size, scratch, value);
                Ok(scratch)
            }
            Some(_) => Err(target_error(
                "expanding the machine operation",
                &format!("operand {index} of the operation is not a register"),
            )),
            None => Err(target_error(
                "expanding the machine operation",
                &format!("the operation has no operand {index}"),
            )),
        }
    }

    /// A scratch register the expansion may use.
    ///
    /// The destination of the instruction and the registers the
    /// expansion reserved are left alone, so a value that is already
    /// there is never overwritten.
    fn scratch_register(&self, inst: &MInst, reserved: &[u16]) -> Result<u16, CodegenError> {
        let dest = self.dest(inst).unwrap_or(u16::MAX);
        for candidate in reg::SCRATCH_GPRS {
            if candidate != dest && !reserved.contains(&candidate) {
                return Ok(candidate);
            }
        }
        Err(target_error(
            "expanding the machine operation",
            "the operation needs a scratch register and none is free",
        ))
    }

    /// The types of the parameters of the function at `index`.
    ///
    /// The parameter of a function is read with an instruction that
    /// names its type, so the signature of the callee is read from
    /// its own body.
    fn parameter_types(&self, index: usize) -> Vec<MType> {
        let mut types: Vec<MType> = Vec::new();
        let Some(function) = self.module.functions.get(index) else {
            return types;
        };
        for block in &function.blocks {
            for inst in &block.insts {
                let MOp::Param { index: position } = inst.op else {
                    continue;
                };
                let slot = position as usize;
                if types.len() <= slot {
                    types.resize(slot + 1, MType::Void);
                }
                types[slot] = inst.operand_ty.clone().unwrap_or(MType::Void);
            }
        }
        types
    }

    /// The type of the values an instruction reads.
    ///
    /// The machine IR names the type of the operands of every
    /// instruction, so the expander never has to guess it from the
    /// register that happens to hold the value.
    fn operand_type(inst: &MInst) -> Option<&MType> {
        inst.operand_ty.as_ref()
    }

    /// The shape of the vector an instruction reads.
    ///
    /// The lane size and the number of lanes follow the type of the
    /// vector, which the instruction names.
    fn vector_shape(&self, inst: &MInst, index: usize) -> Result<(Size, u32), CodegenError> {
        let ty = Self::operand_type(inst).ok_or_else(|| {
            target_error(
                "expanding the machine operation",
                "the operation does not name the type of the vector it reads",
            )
        })?;
        let size = lane_size(ty).ok_or_else(|| {
            target_error(
                "expanding the machine operation",
                &format!("operand {index} does not hold a vector"),
            )
        })?;
        Ok((size, ty.lane_count().max(1)))
    }

    /// The class of a register.
    fn class_of(reg: u16) -> Class {
        if reg::is_xmm(reg) {
            Class::Float
        } else {
            Class::Int
        }
    }

    /// Checks that an operand is an integer.
    fn expect_int(&self, reg: u16, index: usize) -> Result<(), CodegenError> {
        if Self::class_of(reg) == Class::Float {
            return Err(target_error(
                "expanding the machine operation",
                &format!(
                    "operand {index} of the operation is held in the vector register {}",
                    reg::name(reg)
                ),
            ));
        }
        Ok(())
    }

    /// Checks that an operand is a floating-point value.
    fn expect_float(&self, reg: u16, index: usize) -> Result<(), CodegenError> {
        if Self::class_of(reg) == Class::Int {
            return Err(target_error(
                "expanding the machine operation",
                &format!(
                    "operand {index} of the operation is held in the general-purpose register {}",
                    reg::name(reg)
                ),
            ));
        }
        Ok(())
    }

    /// Checks that the destination of an operation has the class its
    /// type asks for.
    fn expect_dest(&self, inst: &MInst, dest: u16) -> Result<(), CodegenError> {
        let wants_float = inst.ty.is_floating() || inst.ty.is_vector();
        let has_float = Self::class_of(dest) == Class::Float;
        if wants_float != has_float {
            return Err(target_error(
                "expanding the machine operation",
                &format!(
                    "the destination register {} does not hold a value of type {:?}",
                    reg::name(dest),
                    inst.ty
                ),
            ));
        }
        Ok(())
    }

    /// Checks that the function holds the block a branch names.
    fn check_block(&self, block: u32) -> Result<(), CodegenError> {
        if block as usize >= self.function.blocks.len() {
            return Err(target_error(
                "expanding the machine operation",
                &format!("the branch names the block {block}, which the function does not hold"),
            ));
        }
        Ok(())
    }

    /// Emits the prologue of the function.
    fn emit_prologue(&self, enc: &mut Encoder<'_>) {
        for reg in &self.saved {
            enc.push(*reg);
        }
        if self.frame_size > 0 {
            enc.sub_ri(Size::Qword, reg::RSP, i64::from(self.frame_size));
        }
    }

    /// Emits the epilogue of the function.
    ///
    /// The frame is released first: the saved registers sit above it,
    /// at the frame base plus the size of the frame.
    fn emit_epilogue(&self, enc: &mut Encoder<'_>) {
        if self.frame_size > 0 {
            enc.add_ri(Size::Qword, reg::RSP, i64::from(self.frame_size));
        }
        for reg in self.saved.iter().rev() {
            enc.pop(*reg);
        }
    }

    /// Expands one instruction.
    fn expand_inst(
        &mut self,
        enc: &mut Encoder<'_>,
        fixups: &mut Vec<Fixup>,
        inst: &MInst,
    ) -> Result<(), CodegenError> {
        self.expand_op(enc, fixups, inst)
    }

    /// Expands one operation.
    fn expand_op(
        &mut self,
        enc: &mut Encoder<'_>,
        fixups: &mut Vec<Fixup>,
        inst: &MInst,
    ) -> Result<(), CodegenError> {
        match &inst.op {
            MOp::Copy => self.expand_copy(enc, inst),
            MOp::Load { size } => self.expand_load(enc, inst, *size),
            MOp::Store { size } => self.expand_store(enc, inst, *size),
            MOp::Gep { size, signed } => self.expand_gep(enc, inst, *size, *signed),
            MOp::VecSplat => self.expand_vec_splat(enc, inst),
            MOp::VecExtract { index } => self.expand_vec_extract(enc, inst, *index),
            MOp::VecInsert { index } => self.expand_vec_insert(enc, inst, *index),
            MOp::VecShuffle { mask } => self.expand_vec_shuffle(enc, inst, mask),
            MOp::Add => self.expand_binary(enc, inst, Binary::Add),
            MOp::Sub => self.expand_binary(enc, inst, Binary::Sub),
            MOp::Mul => self.expand_binary(enc, inst, Binary::Mul),
            MOp::And => self.expand_binary(enc, inst, Binary::And),
            MOp::Or => self.expand_binary(enc, inst, Binary::Or),
            MOp::Xor => self.expand_binary(enc, inst, Binary::Xor),
            MOp::UDiv => self.expand_divide(enc, inst, true, false),
            MOp::SDiv => self.expand_divide(enc, inst, true, true),
            MOp::URem => self.expand_divide(enc, inst, false, false),
            MOp::SRem => self.expand_divide(enc, inst, false, true),
            MOp::Shl => self.expand_shift(enc, inst, Shift::Left),
            MOp::LShr => self.expand_shift(enc, inst, Shift::LogicalRight),
            MOp::AShr => self.expand_shift(enc, inst, Shift::ArithmeticRight),
            MOp::Neg => self.expand_unary(enc, inst, Unary::Neg),
            MOp::Not => self.expand_unary(enc, inst, Unary::Not),
            MOp::FNeg => self.expand_fneg(enc, inst),
            MOp::FAdd => self.expand_float(enc, inst, Arith::Add),
            MOp::FSub => self.expand_float(enc, inst, Arith::Sub),
            MOp::FMul => self.expand_float(enc, inst, Arith::Mul),
            MOp::FDiv => self.expand_float(enc, inst, Arith::Div),
            MOp::FRem => self.expand_frem(enc, inst),
            MOp::FSqrt => self.expand_fsqrt(enc, inst),
            MOp::FAbs => self.expand_fabs(enc, inst),
            MOp::FMin => self.expand_fmin(enc, inst, true),
            MOp::FMax => self.expand_fmin(enc, inst, false),
            MOp::FDim => self.expand_fdim(enc, inst),
            MOp::FCopySign => self.expand_fcopysign(enc, inst),
            MOp::ICmp { pred } => self.expand_icmp(enc, inst, *pred),
            MOp::FCmp { pred } => self.expand_fcmp(enc, inst, *pred),
            MOp::Select => self.expand_select(enc, inst),
            MOp::ZExt => self.expand_zext(enc, inst),
            MOp::SExt => self.expand_sext(enc, inst),
            MOp::Trunc => self.expand_trunc(enc, inst),
            MOp::FPToUI => self.expand_fp_to_ui(enc, inst),
            MOp::FPToSI => self.expand_fp_to_si(enc, inst),
            MOp::UIToFP => self.expand_ui_to_fp(enc, inst),
            MOp::SIToFP => self.expand_si_to_fp(enc, inst),
            MOp::FPTrunc => self.expand_fp_width(enc, inst, false),
            MOp::FPExt => self.expand_fp_width(enc, inst, true),
            MOp::Bitcast => self.expand_bitcast(enc, inst),
            MOp::Br { target } => {
                let block = *target;
                self.check_block(block)?;
                let site = enc.jmp_rel32();
                fixups.push(Fixup::block(site, block));
                Ok(())
            }
            // The blocks the branch names are the numbers of the
            // blocks of the function, carried by the operands as
            // immediates.
            #[allow(clippy::unneeded_struct_pattern)]
            MOp::CondBr { .. } => {
                let cond = self.operand_reg(enc, inst, 0, &[])?;
                self.expect_int(cond, 0)?;
                let (when_true, when_false) = inst.cond_br_targets().ok_or_else(|| {
                    target_error(
                        "expanding the machine operation",
                        "the conditional branch does not name the two blocks it branches to",
                    )
                })?;
                self.check_block(when_true)?;
                self.check_block(when_false)?;
                enc.test_rr(Size::Dword, cond, cond);
                let site = enc.jcc_rel32(Cc::Ne);
                fixups.push(Fixup::block(site, when_true));
                let site = enc.jmp_rel32();
                fixups.push(Fixup::block(site, when_false));
                Ok(())
            }
            MOp::Ret => {
                if inst.ty != MType::Void {
                    let value = self.operand_reg(enc, inst, 0, &[])?;
                    if inst.ty.is_floating() {
                        enc.mov_xmm_rr(Size::Qword, reg::XMM0, value);
                    } else if inst.ty.is_vector() {
                        enc.mov_xmm_rr(Size::Xmm, reg::XMM0, value);
                    } else {
                        enc.mov_rr(Size::Qword, reg::RAX, value);
                    }
                }
                self.emit_epilogue(enc);
                enc.ret();
                Ok(())
            }
            MOp::RetVoid => {
                self.emit_epilogue(enc);
                enc.ret();
                Ok(())
            }
            MOp::Unreachable => {
                enc.ud2();
                Ok(())
            }
            MOp::Call { callee } => self.expand_call(enc, fixups, inst, *callee),
            MOp::WorkItemLoad { field, lane } => {
                let base = self.operand_reg(enc, inst, 0, &[])?;
                self.expect_int(base, 0)?;
                let dest = self.dest(inst)?;
                self.expect_dest(inst, dest)?;
                let offset = i32::try_from(field.offset() + lane * 4).unwrap_or(i32::MAX);
                enc.mov_rm(Size::Dword, dest, base, offset);
                Ok(())
            }
            MOp::ArgBlockBase => {
                let base = self.operand_reg(enc, inst, 0, &[])?;
                self.expect_int(base, 0)?;
                let dest = self.dest(inst)?;
                self.expect_dest(inst, dest)?;
                enc.mov_rm(
                    Size::Qword,
                    dest,
                    base,
                    i32::try_from(ARG_BASE_OFFSET).unwrap_or(i32::MAX),
                );
                Ok(())
            }
            MOp::ArgLoad { index, kind } => {
                let base = self.operand_reg(enc, inst, 0, &[])?;
                self.expect_int(base, 0)?;
                let dest = self.dest(inst)?;
                self.expect_dest(inst, dest)?;
                let offset = i32::try_from(index * ARG_SLOT as u32).unwrap_or(i32::MAX);
                match kind {
                    ArgKind::Pointer => enc.mov_rm(Size::Qword, dest, base, offset),
                    ArgKind::Scalar { size } => match size {
                        1 => enc.movzx_rm(Size::Byte, Size::Dword, dest, base, offset),
                        2 => enc.movzx_rm(Size::Word, Size::Dword, dest, base, offset),
                        4 => enc.mov_rm(Size::Dword, dest, base, offset),
                        8 => enc.mov_rm(Size::Qword, dest, base, offset),
                        _ => {
                            return Err(unsupported_error(
                                "the size of a kernel argument",
                                "the target reads arguments of one, two, four, or eight bytes",
                            ));
                        }
                    },
                }
                Ok(())
            }
            MOp::Param { index } => {
                let dest = self.dest(inst)?;
                self.expect_dest(inst, dest)?;
                let position = *index as usize;
                if Self::class_of(dest) == Class::Float {
                    if position >= reg::FLOAT_ARG_REGS.len() {
                        return Err(target_error(
                            "expanding the machine operation",
                            "the parameter is the ninth floating-point argument",
                        ));
                    }
                    enc.mov_xmm_rr(Size::Qword, dest, reg::FLOAT_ARG_REGS[position]);
                } else {
                    if position >= reg::ARG_REGS.len() {
                        return Err(target_error(
                            "expanding the machine operation",
                            "the parameter is the seventh integer argument",
                        ));
                    }
                    enc.mov_rr(Size::Qword, dest, reg::ARG_REGS[position]);
                }
                Ok(())
            }
            MOp::FrameAddr { offset } => {
                let dest = self.dest(inst)?;
                self.expect_dest(inst, dest)?;
                enc.lea(dest, reg::RSP, i32::try_from(*offset).unwrap_or(i32::MAX));
                Ok(())
            }
            MOp::SpillStore { offset, size } => {
                let value = self.operand_reg(enc, inst, 0, &[])?;
                let disp = i32::try_from(*offset).unwrap_or(i32::MAX);
                match Self::class_of(value) {
                    Class::Float => {
                        let size = match size {
                            4 => Size::Dword,
                            8 => Size::Qword,
                            _ => {
                                return Err(unsupported_error(
                                    "the size of a spilled value",
                                    "the target spills floating-point values of four or eight bytes",
                                ));
                            }
                        };
                        enc.mov_xmm_mr(size, reg::RSP, disp, value);
                    }
                    Class::Int => match size {
                        1 => enc.mov_mr(Size::Byte, reg::RSP, disp, value),
                        2 => enc.mov_mr(Size::Word, reg::RSP, disp, value),
                        4 => enc.mov_mr(Size::Dword, reg::RSP, disp, value),
                        8 => enc.mov_mr(Size::Qword, reg::RSP, disp, value),
                        _ => {
                            return Err(unsupported_error(
                                "the size of a spilled value",
                                "the target spills values of one, two, four, or eight bytes",
                            ));
                        }
                    },
                }
                Ok(())
            }
            MOp::SpillLoad { offset, size } => {
                let dest = self.dest(inst)?;
                self.expect_dest(inst, dest)?;
                let disp = i32::try_from(*offset).unwrap_or(i32::MAX);
                match size {
                    1 => enc.movzx_rm(Size::Byte, Size::Dword, dest, reg::RSP, disp),
                    2 => enc.movzx_rm(Size::Word, Size::Dword, dest, reg::RSP, disp),
                    4 => enc.mov_rm(Size::Dword, dest, reg::RSP, disp),
                    8 => enc.mov_rm(Size::Qword, dest, reg::RSP, disp),
                    _ => {
                        return Err(unsupported_error(
                            "the size of a spilled value",
                            "the target loads values of one, two, four, or eight bytes",
                        ));
                    }
                }
                Ok(())
            }
            MOp::FrameSetup | MOp::FrameTeardown => {
                // The back end owns the frame: it reserves the frame
                // in the prologue and releases it in the epilogue.
                Ok(())
            }
        }
    }

    /// Expands a copy.
    fn expand_copy(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        if let Some(Operand::FloatImmediate(bits)) = inst.args.get(0) {
            if !inst.ty.is_floating() && !inst.ty.is_vector() {
                return Err(target_error(
                    "expanding the machine operation",
                    "a floating-point immediate is copied into an integer register",
                ));
            }
            // The immediate holds the bits of the value, so it moves
            // into the register as they are.
            enc.mov_ri(Size::Qword, reg::RAX, bits as i64);
            enc.movq_xmm_rm(dest, reg::RAX);
            return Ok(());
        }
        if let Some(Operand::Immediate(value)) = inst.args.get(0) {
            if inst.ty.is_floating() || inst.ty.is_vector() {
                return Err(target_error(
                    "expanding the machine operation",
                    "a floating-point value is not copied from an immediate",
                ));
            }
            enc.mov_ri(int_size(&inst.ty), dest, value);
            return Ok(());
        }
        let src = self.operand_reg(enc, inst, 0, &[])?;
        match (Self::class_of(dest), Self::class_of(src)) {
            (Class::Int, Class::Int) => {
                enc.mov_rr(int_size(&inst.ty), dest, src);
            }
            (Class::Float, Class::Float) => {
                let size = if inst.ty.is_vector() {
                    Size::Xmm
                } else {
                    float_size(&inst.ty).unwrap_or(Size::Xmm)
                };
                enc.mov_xmm_rr(size, dest, src);
            }
            (Class::Int, Class::Float) => {
                if int_size(&inst.ty) == Size::Qword {
                    enc.movq_rm_xmm(dest, src);
                } else {
                    enc.movd_rm_xmm(dest, src);
                }
            }
            (Class::Float, Class::Int) => {
                let size = float_size(&inst.ty).unwrap_or(Size::Dword);
                if size == Size::Qword {
                    enc.movq_xmm_rm(dest, src);
                } else {
                    enc.movd_xmm_rm(dest, src);
                }
            }
        }
        Ok(())
    }

    /// Expands a load.
    fn expand_load(&self, enc: &mut Encoder<'_>, inst: &MInst, size: u32) -> Result<(), CodegenError> {
        let base = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(base, 0)?;
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        if Self::class_of(dest) == Class::Float {
            let size = match size {
                4 => Size::Dword,
                8 => Size::Qword,
                _ => {
                    return Err(unsupported_error(
                        "the size of a load",
                        "the target loads floating-point values of four or eight bytes",
                    ));
                }
            };
            enc.mov_xmm_rm(size, dest, base, 0);
            return Ok(());
        }
        match size {
            1 => enc.movzx_rm(Size::Byte, Size::Dword, dest, base, 0),
            2 => enc.movzx_rm(Size::Word, Size::Dword, dest, base, 0),
            4 => enc.mov_rm(Size::Dword, dest, base, 0),
            8 => enc.mov_rm(Size::Qword, dest, base, 0),
            _ => {
                return Err(unsupported_error(
                    "the size of a load",
                    "the target loads values of one, two, four, or eight bytes",
                ));
            }
        }
        Ok(())
    }

    /// Expands a store.
    fn expand_store(&self, enc: &mut Encoder<'_>, inst: &MInst, size: u32) -> Result<(), CodegenError> {
        let base = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(base, 0)?;
        // A constant is moved into a register first, so that the
        // store has a value to write.
        let (value, floating) = match inst.args.get(1) {
            Some(Operand::Immediate(number)) => {
                let width = match size {
                    1 => Size::Byte,
                    2 => Size::Word,
                    4 => Size::Dword,
                    _ => Size::Qword,
                };
                enc.mov_ri(width, reg::RAX, number);
                (reg::RAX, false)
            }
            Some(Operand::FloatImmediate(bits)) => {
                enc.mov_ri(Size::Qword, reg::RAX, bits as i64);
                enc.movq_xmm_rm(reg::SCRATCH_XMMS[0], reg::RAX);
                (reg::SCRATCH_XMMS[0], true)
            }
            Some(Operand::Physical(reg)) => {
                let number = reg.number();
                (number, Self::class_of(number) == Class::Float)
            }
            Some(_) => {
                return Err(target_error(
                    "expanding the machine operation",
                    "the value the store writes is not a register",
                ));
            }
            None => {
                return Err(target_error(
                    "expanding the machine operation",
                    "the operation has no second operand",
                ));
            }
        };
        match floating {
            true => {
                let size = match size {
                    4 => Size::Dword,
                    8 => Size::Qword,
                    _ => {
                        return Err(unsupported_error(
                            "the size of a store",
                            "the target stores floating-point values of four or eight bytes",
                        ));
                    }
                };
                enc.mov_xmm_mr(size, base, 0, value);
            }
            false => match size {
                1 => enc.mov_mr(Size::Byte, base, 0, value),
                2 => enc.mov_mr(Size::Word, base, 0, value),
                4 => enc.mov_mr(Size::Dword, base, 0, value),
                8 => enc.mov_mr(Size::Qword, base, 0, value),
                _ => {
                    return Err(unsupported_error(
                        "the size of a store",
                        "the target stores values of one, two, four, or eight bytes",
                    ));
                }
            },
        }
        Ok(())
    }

    /// Expands an address computation.
    ///
    /// The base is copied into the destination and every index is
    /// scaled by the size of the pointee and added.  A single index
    /// becomes one address computation when the architecture can scale
    /// it: the addressing mode multiplies the index by the size of the
    /// pointee and adds the base, so the index never leaves its
    /// register.
    fn expand_gep(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        size: u32,
        signed: bool,
    ) -> Result<(), CodegenError> {
        let base = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(base, 0)?;
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        // The addressing mode scales the index by one, two, four, or
        // eight and adds the base.  An index of `rsp` has no encoding
        // in the index field of the byte that carries the scale, and a
        // constant index is scaled when its product fits the
        // displacement.
        if inst.args.len() == 2 {
            match inst.args.get(1) {
                Some(Operand::Immediate(value)) => {
                    let disp = value
                        .checked_mul(i64::from(size))
                        .and_then(|disp| i32::try_from(disp).ok());
                    if let Some(disp) = disp {
                        enc.lea(dest, base, disp);
                        return Ok(());
                    }
                }
                Some(Operand::Physical(_)) | Some(Operand::Virtual(_)) => {
                    let index = self.operand_reg(enc, inst, 1, &[])?;
                    self.expect_int(index, 1)?;
                    if matches!(size, 1 | 2 | 4 | 8) && index != reg::RSP {
                        enc.lea_indexed(dest, base, index, size as u8, 0);
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        enc.mov_rr(Size::Qword, dest, base);
        let mut index = 1;
        while index < inst.args.len() {
            match inst.args.get(index) {
                Some(Operand::Immediate(value)) => {
                    if signed {
                        enc.mov_ri(Size::Qword, reg::RAX, value);
                    } else {
                        enc.mov_ri(Size::Dword, reg::RAX, value);
                    }
                }
                Some(Operand::Physical(_)) | Some(Operand::Virtual(_)) => {
                    let source = self.operand_reg(enc, inst, index, &[reg::RAX])?;
                    self.expect_int(source, index)?;
                    enc.mov_rr(Size::Qword, reg::RAX, source);
                }
                Some(_) => {
                    return Err(target_error(
                        "expanding the machine operation",
                        "an index of the address computation is neither a register nor an immediate",
                    ));
                }
                None => break,
            }
            if size != 1 {
                if size.is_power_of_two() {
                    let amount = size.trailing_zeros();
                    if amount > 0 {
                        enc.shift_ri(4, Size::Qword, reg::RAX, amount as u8);
                    }
                } else {
                    match i32::try_from(size) {
                        Ok(factor) => enc.imul_ri(Size::Qword, reg::RAX, reg::RAX, factor),
                        Err(_) => {
                            return Err(unsupported_error(
                                "the size of the pointee",
                                "the target scales an index by a size of at most four gigabytes",
                            ));
                        }
                    }
                }
            }
            enc.add_rr(Size::Qword, dest, reg::RAX);
            index += 1;
        }
        Ok(())
    }

    /// Expands the broadcast of a scalar over the lanes of a vector.
    fn expand_vec_splat(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let source = self.operand_reg(enc, inst, 0, &[])?;
        // The broadcast fills the vector the destination names, so the
        // shape follows the destination.
        let size = lane_size(&inst.ty).unwrap_or(Size::Dword);
        match size {
            Size::Byte | Size::Word => {
                if Self::class_of(source) == Class::Float {
                    enc.movd_rm_xmm(reg::RAX, source);
                } else {
                    enc.mov_rr(Size::Dword, reg::RAX, source);
                }
                enc.movd_xmm_rm(dest, reg::RAX);
                enc.pshuflw(dest, dest, 0);
                enc.pshufd(dest, dest, 0x04);
            }
            Size::Dword => {
                if Self::class_of(source) == Class::Float {
                    enc.movd_rm_xmm(dest, source);
                } else {
                    enc.movd_xmm_rm(dest, source);
                }
                enc.pshufd(dest, dest, 0);
            }
            Size::Qword => {
                if Self::class_of(source) == Class::Float {
                    enc.mov_xmm_rr(Size::Qword, dest, source);
                } else {
                    enc.movq_xmm_rm(dest, source);
                }
                enc.punpcklqdq(dest, dest);
            }
            Size::Xmm => {
                return Err(unsupported_error(
                    "the broadcast of a scalar",
                    "the target broadcasts scalars over the lanes of a vector",
                ));
            }
        }
        Ok(())
    }

    /// Expands the read of one lane of a vector.
    fn expand_vec_extract(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        index: u32,
    ) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        let source = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_float(source, 0)?;
        let (size, lanes) = self.vector_shape(inst, 0)?;
        if index >= lanes {
            return Err(target_error(
                "expanding the machine operation",
                &format!("the lane {index} lies outside the vector"),
            ));
        }
        match size {
            Size::Byte | Size::Word => {
                if Self::class_of(dest) == Class::Float {
                    return Err(target_error(
                        "expanding the machine operation",
                        "the lane of the vector is read into a vector register",
                    ));
                }
                enc.pextrw(dest, source, index as u8);
            }
            Size::Dword => {
                if Self::class_of(dest) == Class::Float {
                    enc.pshufd(reg::SCRATCH_XMMS[1], source, index as u8);
                    enc.mov_xmm_rr(Size::Dword, dest, reg::SCRATCH_XMMS[1]);
                } else {
                    enc.pextrd_rm_xmm(dest, source, index as u8);
                }
            }
            Size::Qword => {
                let selector = if index == 0 { 0x00 } else { 0xEE };
                enc.pshufd(reg::SCRATCH_XMMS[1], source, selector);
                if Self::class_of(dest) == Class::Float {
                    enc.mov_xmm_rr(Size::Qword, dest, reg::SCRATCH_XMMS[1]);
                } else {
                    enc.movq_rm_xmm(dest, reg::SCRATCH_XMMS[1]);
                }
            }
            Size::Xmm => {
                return Err(unsupported_error(
                    "the read of a lane",
                    "the target reads the lanes of a vector of bytes, halves, or words",
                ));
            }
        }
        Ok(())
    }

    /// Expands the write of one lane of a vector.
    fn expand_vec_insert(&self, enc: &mut Encoder<'_>, inst: &MInst, index: u32) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        let vector = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_float(vector, 0)?;
        let scalar = self.operand_reg(enc, inst, 1, &[])?;
        let (size, lanes) = self.vector_shape(inst, 0)?;
        if index >= lanes {
            return Err(target_error(
                "expanding the machine operation",
                &format!("the lane {index} lies outside the vector"),
            ));
        }
        match size {
            Size::Byte | Size::Word => {
                if Self::class_of(scalar) == Class::Float {
                    enc.movd_rm_xmm(reg::RAX, scalar);
                } else {
                    enc.mov_rr(Size::Dword, reg::RAX, scalar);
                }
                enc.mov_xmm_rr(Size::Xmm, dest, vector);
                enc.pinsrw(dest, reg::RAX, index as u8);
            }
            Size::Dword => {
                enc.mov_xmm_rr(Size::Xmm, dest, vector);
                if Self::class_of(scalar) == Class::Float {
                    enc.insertps_xmm_rr(dest, scalar, index as u8);
                } else {
                    enc.pinsrd_xmm_rr(dest, scalar, index as u8);
                }
            }
            Size::Qword => {
                if Self::class_of(scalar) == Class::Float {
                    enc.movq_rm_xmm(reg::RAX, scalar);
                } else {
                    enc.mov_rr(Size::Qword, reg::RAX, scalar);
                }
                enc.movq_xmm_rm(reg::SCRATCH_XMMS[1], reg::RAX);
                if index == 0 {
                    // The low quad-word is replaced, so the high one
                    // moves down into a register of its own and the
                    // two are interleaved again.
                    enc.punpckhqdq(reg::SCRATCH_XMMS[0], dest);
                    enc.punpcklqdq(reg::SCRATCH_XMMS[1], reg::SCRATCH_XMMS[0]);
                    enc.mov_xmm_rr(Size::Xmm, dest, reg::SCRATCH_XMMS[1]);
                } else {
                    // The high quad-word is replaced, so the low one
                    // stays where it is.
                    enc.punpcklqdq(dest, reg::SCRATCH_XMMS[1]);
                }
            }
            Size::Xmm => {
                return Err(unsupported_error(
                    "the write of a lane",
                    "the target writes the lanes of a vector of bytes, halves, or words",
                ));
            }
        }
        Ok(())
    }

    /// Expands the shuffle of two vectors.
    ///
    /// The baseline has no instruction that picks lanes from two
    /// registers at once, so every lane is taken from its source and
    /// written into the destination one lane at a time.
    fn expand_vec_shuffle(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        mask: &[u32],
    ) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        let first = self.operand_reg(enc, inst, 0, &[])?;
        let second = self.operand_reg(enc, inst, 1, &[])?;
        self.expect_float(first, 0)?;
        self.expect_float(second, 1)?;
        let (size, lane_count) = self.vector_shape(inst, 0)?;
        let lanes = lane_count as usize;
        if mask.len() != lanes {
            return Err(target_error(
                "expanding the machine operation",
                &format!(
                    "the shuffle names {} lanes, but the vector holds {lanes}",
                    mask.len()
                ),
            ));
        }
        for (lane, entry) in mask.iter().enumerate() {
            if *entry >= lanes as u32 * 2 {
                return Err(target_error(
                    "expanding the machine operation",
                    &format!("the shuffle names the lane {entry}, which is outside the sources"),
                ));
            }
            let source = if *entry < lanes as u32 { first } else { second };
            let index = *entry % lanes as u32;
            match size {
                Size::Byte | Size::Word => {
                    enc.pextrw(reg::RAX, source, index as u8);
                    enc.pinsrw(dest, reg::RAX, lane as u8);
                }
                Size::Dword => {
                    enc.pshufd(reg::SCRATCH_XMMS[1], source, index as u8);
                    enc.movd_rm_xmm(reg::RAX, reg::SCRATCH_XMMS[1]);
                    if lane == 0 {
                        enc.movd_xmm_rm(dest, reg::RAX);
                    } else {
                        enc.pinsrw(dest, reg::RAX, (lane * 2) as u8);
                        enc.shift_ri(5, Size::Dword, reg::RAX, 16);
                        enc.pinsrw(dest, reg::RAX, (lane * 2 + 1) as u8);
                    }
                }
                Size::Qword => {
                    let selector = if index == 0 { 0x00 } else { 0xEE };
                    enc.pshufd(reg::SCRATCH_XMMS[1], source, selector);
                    enc.movq_rm_xmm(reg::RAX, reg::SCRATCH_XMMS[1]);
                    if lane == 0 {
                        enc.movq_xmm_rm(dest, reg::RAX);
                    } else {
                        enc.movq_xmm_rm(reg::SCRATCH_XMMS[1], reg::RAX);
                        enc.punpcklqdq(dest, reg::SCRATCH_XMMS[1]);
                    }
                }
                Size::Xmm => {
                    return Err(unsupported_error(
                        "the shuffle of two vectors",
                        "the target shuffles vectors of bytes, halves, or words",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Expands an integer operation of two operands.
    fn expand_binary(&self, enc: &mut Encoder<'_>, inst: &MInst, op: Binary) -> Result<(), CodegenError> {
        // A vector operation uses the packed forms of the
        // architecture.
        if let Some(size) = self.vector_lane_size(inst) {
            return self.expand_vector_binary(enc, inst, op, size);
        }
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = int_size(&inst.ty);
        let left = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(left, 0)?;
        match inst.args.get(1) {
            Some(Operand::Immediate(value)) => {
                enc.mov_rr(size, dest, left);
                match op {
                    Binary::Add => enc.add_ri(size, dest, value),
                    Binary::Sub => enc.sub_ri(size, dest, value),
                    Binary::And => enc.and_ri(size, dest, value),
                    Binary::Or => enc.or_ri(size, dest, value),
                    Binary::Xor => enc.xor_ri(size, dest, value),
                    Binary::Mul => match i32::try_from(value) {
                        Ok(factor) => enc.imul_ri(size, dest, dest, factor),
                        Err(_) => {
                            // The wide form of the multiplication
                            // takes a thirty-two-bit factor, so a
                            // wider one moves into a register first.
                            let scratch = self.scratch_register(inst, &[dest])?;
                            enc.mov_ri(size, scratch, value);
                            enc.imul_rr(size, dest, scratch);
                        }
                    },
                }
            }
            Some(_) => {
                let right = self.operand_reg(enc, inst, 1, &[])?;
                self.expect_int(right, 1)?;
                enc.mov_rr(size, dest, left);
                match op {
                    Binary::Add => enc.add_rr(size, dest, right),
                    Binary::Sub => enc.sub_rr(size, dest, right),
                    Binary::And => enc.and_rr(size, dest, right),
                    Binary::Or => enc.or_rr(size, dest, right),
                    Binary::Xor => enc.xor_rr(size, dest, right),
                    Binary::Mul => enc.imul_rr(size, dest, right),
                }
            }
            None => {
                return Err(target_error(
                    "expanding the machine operation",
                    "the operation has no second operand",
                ));
            }
        }
        Ok(())
    }

    /// The size of a lane of the vector an operation reads, when the
    /// operation reads a vector.
    fn vector_lane_size(&self, inst: &MInst) -> Option<Size> {
        let ty = Self::operand_type(inst)?;
        if !ty.is_vector() {
            return None;
        }
        match ty {
            MType::Vec { elem, .. } => match elem.as_ref() {
                MType::F64 => Some(Size::Qword),
                _ => Some(Size::Dword),
            },
            _ => None,
        }
    }

    /// True when the lanes of a vector hold floating-point values.
    fn vector_is_floating(&self, inst: &MInst) -> bool {
        let Some(ty) = Self::operand_type(inst) else {
            return false;
        };
        match ty {
            MType::Vec { elem, .. } => elem.is_floating(),
            _ => false,
        }
    }

    /// Expands an operation over the lanes of a vector.
    ///
    /// The architecture has a packed form for the operations it can
    /// do on several lanes at once; the others are refused.
    fn expand_vector_binary(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        op: Binary,
        size: Size,
    ) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        let left = self.operand_reg(enc, inst, 0, &[])?;
        let right = self.operand_reg(enc, inst, 1, &[])?;
        if Self::class_of(left) != Class::Float || Self::class_of(right) != Class::Float {
            return Err(target_error(
                "expanding the machine operation",
                "the operation reads a value that is not in a vector register",
            ));
        }
        // The whole vector moves, not one lane of it.
        enc.mov_xmm_rr(Size::Xmm, dest, left);
        let floating = self.vector_is_floating(inst);
        if floating {
            let arith = match op {
                Binary::Add => Arith::Add,
                Binary::Sub => Arith::Sub,
                Binary::Mul => Arith::Mul,
                Binary::And | Binary::Or | Binary::Xor => {
                    return Err(target_error(
                        "expanding the machine operation",
                        "the operation has no packed form over floating-point lanes",
                    ));
                }
            };
            enc.packed_xmm_rr(arith, size, dest, right);
            return Ok(());
        }
        // The packed integer forms: the addition and the subtraction
        // of thirty-two and sixty-four-bit lanes, and the three
        // bitwise operations.
        let opcode = match (op, size) {
            (Binary::Add, Size::Dword) => 0xFE,
            (Binary::Add, Size::Qword) => 0xD4,
            (Binary::Sub, Size::Dword) => 0xF8,
            (Binary::Sub, Size::Qword) => 0xFB,
            (Binary::And, _) => 0xDB,
            (Binary::Or, _) => 0xEB,
            (Binary::Xor, _) => 0xEF,
            _ => {
                // The baseline has no packed multiplication of
                // thirty-two-bit lanes, so the lanes are multiplied
                // one at a time.
                if op == Binary::Mul && size == Size::Dword {
                    return self.expand_vector_multiply(enc, inst, dest, left, right);
                }
                return Err(target_error(
                    "expanding the machine operation",
                    &format!(
                        "the architecture has no packed form for the {} over the lanes of {:?}",
                        op.name(),
                        Self::operand_type(inst)
                    ),
                ));
            }
        };
        enc.packed_int_xmm_rr(opcode, dest, right);
        Ok(())
    }

    /// Multiplies the lanes of two vectors of thirty-two-bit integers.
    ///
    /// The packed multiplication of thirty-two-bit lanes computes
    /// every product at once, so the lanes never leave their vectors.
    /// The instruction is the packed form the architecture gained
    /// with SSE4.1.
    fn expand_vector_multiply(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        dest: u16,
        left: u16,
        right: u16,
    ) -> Result<(), CodegenError> {
        // The operation reads a vector, which the shape confirms, but
        // the instruction reads the whole register at once.
        let _ = self.vector_shape(inst, 0)?;
        enc.mov_xmm_rr(Size::Xmm, dest, left);
        enc.pmulld_xmm_rr(dest, right);
        Ok(())
    }

    /// Expands a division.
    ///
    /// The architecture divides a register pair, so the dividend is
    /// moved into the accumulator, widened into the data register, and
    /// divided by the divisor; the quotient or the remainder is then
    /// moved into the destination.
    fn expand_divide(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        quotient: bool,
        signed: bool,
    ) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = int_size(&inst.ty);
        let dividend = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(dividend, 0)?;
        if dest == reg::RCX {
            return Err(target_error(
                "expanding the machine operation",
                "the destination of the division is the register the expander uses",
            ));
        }
        // The divisor moves first: the accumulator holds the dividend
        // afterwards.
        match inst.args.get(1) {
            Some(Operand::Immediate(value)) => enc.mov_ri(size, reg::RCX, value),
            Some(_) => {
                let divisor = self.operand_reg(enc, inst, 1, &[])?;
                self.expect_int(divisor, 1)?;
                enc.mov_rr(size, reg::RCX, divisor);
            }
            None => {
                return Err(target_error(
                    "expanding the machine operation",
                    "the operation has no second operand",
                ));
            }
        }
        enc.mov_rr(size, reg::RAX, dividend);
        if signed {
            enc.cdq(size);
        } else {
            enc.xor_rr(size, reg::RDX, reg::RDX);
        }
        enc.div_rr(size, reg::RCX, signed);
        let result = if quotient { reg::RAX } else { reg::RDX };
        if dest != result {
            enc.mov_rr(size, dest, result);
        }
        Ok(())
    }

    /// Expands a shift.
    fn expand_shift(&self, enc: &mut Encoder<'_>, inst: &MInst, op: Shift) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = int_size(&inst.ty);
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(value, 0)?;
        let digit = match op {
            Shift::Left => 4,
            Shift::LogicalRight => 5,
            Shift::ArithmeticRight => 7,
        };
        match inst.args.get(1) {
            Some(Operand::Immediate(amount)) => {
                if !(0..=255).contains(&amount) {
                    return Err(target_error(
                        "expanding the machine operation",
                        "the shift amount lies outside the range of the architecture",
                    ));
                }
                enc.mov_rr(size, dest, value);
                enc.shift_ri(digit, size, dest, amount as u8);
            }
            Some(_) => {
                let amount = self.operand_reg(enc, inst, 1, &[])?;
                self.expect_int(amount, 1)?;
                enc.mov_rr(size, dest, value);
                enc.mov_rr(size, reg::RCX, amount);
                enc.shift_cl(digit, size, dest);
            }
            None => {
                return Err(target_error(
                    "expanding the machine operation",
                    "the operation has no second operand",
                ));
            }
        }
        Ok(())
    }

    /// Expands a negation or a complement.
    fn expand_unary(&self, enc: &mut Encoder<'_>, inst: &MInst, op: Unary) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = int_size(&inst.ty);
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(value, 0)?;
        enc.mov_rr(size, dest, value);
        match op {
            Unary::Neg => enc.neg_rr(size, dest),
            Unary::Not => enc.not_rr(size, dest),
        }
        Ok(())
    }

    /// Expands the negation of a floating-point value.
    ///
    /// The sign is flipped with an exclusive or against a mask whose
    /// sign bits are set.
    fn expand_fneg(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_float(value, 0)?;
        let size = float_size(&inst.ty).unwrap_or(Size::Qword);
        let vector = inst.ty.is_vector();
        let mask = match (size, vector) {
            // A single-precision scalar only needs its own sign bit
            // set; a vector of them needs every lane.
            (Size::Dword, false) => 0x0000_0000_8000_0000u64,
            (Size::Dword, true) => 0x8000_0000_8000_0000u64,
            _ => 0x8000_0000_0000_0000u64,
        };
        enc.mov_ri(Size::Qword, reg::RAX, mask as i64);
        enc.movq_xmm_rm(reg::SCRATCH_XMMS[1], reg::RAX);
        if vector {
            enc.punpcklqdq(reg::SCRATCH_XMMS[1], reg::SCRATCH_XMMS[1]);
        }
        enc.mov_xmm_rr(Size::Xmm, dest, value);
        enc.xorps_rr(dest, reg::SCRATCH_XMMS[1]);
        Ok(())
    }

    /// The register an operand of a floating-point operation holds.
    ///
    /// A constant is moved into one of the scratch registers first,
    /// so that the operation has a register to read.
    fn float_operand(&self, enc: &mut Encoder<'_>, inst: &MInst, index: usize) -> Result<u16, CodegenError> {
        match inst.args.get(index) {
            Some(Operand::FloatImmediate(bits)) => {
                let scratch = if index == 0 {
                    reg::SCRATCH_XMMS[0]
                } else {
                    reg::SCRATCH_XMMS[1]
                };
                enc.mov_ri(Size::Qword, reg::RAX, bits as i64);
                enc.movq_xmm_rm(scratch, reg::RAX);
                Ok(scratch)
            }
            Some(Operand::Physical(reg)) => {
                let number = reg.number();
                if !reg::is_valid(number) {
                    return Err(target_error(
                        "expanding the machine operation",
                        &format!("operand {index} names the register {number}, which the file does not hold"),
                    ));
                }
                Ok(number)
            }
            Some(_) => Err(target_error(
                "expanding the machine operation",
                &format!("operand {index} of the operation is not a floating-point value"),
            )),
            None => Err(target_error(
                "expanding the machine operation",
                &format!("the operation has no operand {index}"),
            )),
        }
    }

    /// The width of the floating-point operation.
    fn float_width(&self, inst: &MInst) -> Result<Size, CodegenError> {
        float_size(&inst.ty).ok_or_else(|| {
            target_error(
                "expanding the machine operation",
                "the operation has no floating-point type",
            )
        })
    }

    /// The register an operand of a conversion holds.
    ///
    /// A constant is moved into the accumulator first, so that the
    /// conversion has a register to read.
    fn int_operand(&self, enc: &mut Encoder<'_>, inst: &MInst, index: usize) -> Result<u16, CodegenError> {
        match inst.args.get(index) {
            Some(Operand::Immediate(value)) => {
                enc.mov_ri(Size::Qword, reg::RAX, value);
                Ok(reg::RAX)
            }
            Some(Operand::Physical(reg)) => {
                let number = reg.number();
                if !reg::is_valid(number) {
                    return Err(target_error(
                        "expanding the machine operation",
                        &format!("operand {index} names the register {number}, which the file does not hold"),
                    ));
                }
                Ok(number)
            }
            Some(_) => Err(target_error(
                "expanding the machine operation",
                &format!("operand {index} of the operation is not an integer"),
            )),
            None => Err(target_error(
                "expanding the machine operation",
                &format!("the operation has no operand {index}"),
            )),
        }
    }

    /// Expands the square root of a floating-point value.
    fn expand_fsqrt(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = self.float_width(inst)?;
        let value = self.float_operand(enc, inst, 0)?;
        enc.mov_xmm_rr(size, dest, value);
        enc.sqrt_xmm_rr(size, dest, dest);
        Ok(())
    }

    /// Expands the absolute value of a floating-point value.
    ///
    /// The sign bit is cleared with a mask that holds no other bit.
    fn expand_fabs(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = self.float_width(inst)?;
        let value = self.float_operand(enc, inst, 0)?;
        let mask = match size {
            Size::Dword => 0x0000_0000_7FFF_FFFFu64,
            _ => 0x7FFF_FFFF_FFFF_FFFFu64,
        };
        enc.mov_ri(Size::Qword, reg::RAX, mask as i64);
        enc.movq_xmm_rm(reg::SCRATCH_XMMS[1], reg::RAX);
        enc.mov_xmm_rr(size, dest, value);
        enc.andps_rr(dest, reg::SCRATCH_XMMS[1]);
        Ok(())
    }

    /// Expands the smaller or the larger of two floating-point
    /// values.
    ///
    /// The instruction of the architecture returns its second operand
    /// when one of the two is not a number.
    fn expand_fmin(&self, enc: &mut Encoder<'_>, inst: &MInst, smallest: bool) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = self.float_width(inst)?;
        let left = self.float_operand(enc, inst, 0)?;
        let right = self.float_operand(enc, inst, 1)?;
        enc.mov_xmm_rr(size, dest, left);
        if smallest {
            enc.min_xmm_rr(size, dest, right);
        } else {
            enc.max_xmm_rr(size, dest, right);
        }
        Ok(())
    }

    /// Expands the positive difference of two floating-point values.
    ///
    /// The difference of the values is clamped at zero, which is what
    /// the difference of a value that is not the larger one comes to.
    fn expand_fdim(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = self.float_width(inst)?;
        let left = self.float_operand(enc, inst, 0)?;
        let right = self.float_operand(enc, inst, 1)?;
        enc.mov_xmm_rr(size, dest, left);
        enc.arith_xmm_rr(Arith::Sub, size, dest, right);
        // The zero the difference is clamped at.
        enc.mov_ri(Size::Qword, reg::RAX, 0);
        enc.movq_xmm_rm(reg::SCRATCH_XMMS[0], reg::RAX);
        enc.max_xmm_rr(size, dest, reg::SCRATCH_XMMS[0]);
        Ok(())
    }

    /// Expands the magnitude of one floating-point value with the
    /// sign of another.
    ///
    /// The sign of the second value is copied into the result with a
    /// mask that holds the sign bit alone.
    fn expand_fcopysign(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = self.float_width(inst)?;
        let magnitude = self.float_operand(enc, inst, 0)?;
        let sign = self.float_operand(enc, inst, 1)?;
        enc.mov_xmm_rr(size, dest, magnitude);
        // The mask of the sign bit.
        let mask = match size {
            Size::Dword => 0x0000_0000_8000_0000u64,
            _ => 0x8000_0000_0000_0000u64,
        };
        enc.mov_ri(Size::Qword, reg::RAX, mask as i64);
        enc.movq_xmm_rm(reg::SCRATCH_XMMS[0], reg::RAX);
        // The sign of the second value.
        enc.mov_xmm_rr(size, reg::SCRATCH_XMMS[1], sign);
        enc.andps_rr(reg::SCRATCH_XMMS[1], reg::SCRATCH_XMMS[0]);
        // The magnitude without its sign, with the sign of the
        // second value.
        enc.andnps_rr(reg::SCRATCH_XMMS[0], dest);
        enc.orps_rr(dest, reg::SCRATCH_XMMS[1]);
        Ok(())
    }

    /// Expands the remainder of a division of two floating-point
    /// values.
    ///
    /// The architecture has no remainder instruction in its scalar
    /// floating-point unit, but the stack of the unit has one, and it
    /// computes the remainder exactly.  The two values are therefore
    /// pushed on the machine stack, the remainder is computed, and
    /// the result is read back.
    fn expand_frem(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let size = self.float_width(inst)?;
        // The bits of the dividend, then the bits of the divisor.
        let dividend = self.float_operand(enc, inst, 0)?;
        self.push_double(enc, size, dividend);
        let divisor = self.float_operand(enc, inst, 1)?;
        self.push_double(enc, size, divisor);
        // The divisor goes on the stack first, so that the dividend
        // is the one the remainder is taken of.
        enc.fld_m64(reg::RSP, 0);
        enc.fld_m64(reg::RSP, 8);
        enc.fprem();
        enc.fstp_m64(reg::RSP, 0);
        // The stack of the floating-point unit holds double-precision
        // values, so the remainder is narrowed when the operation is a
        // single-precision one.
        enc.movq_xmm_mr(reg::SCRATCH_XMMS[0], reg::RSP, 0);
        if size == Size::Dword {
            enc.cvt_xmm2xmm(false, dest, reg::SCRATCH_XMMS[0]);
        } else {
            enc.mov_xmm_rr(Size::Qword, dest, reg::SCRATCH_XMMS[0]);
        }
        // The stack of the floating-point unit keeps the dividend.
        enc.fstp_st0();
        enc.add_ri(Size::Qword, reg::RSP, 16);
        Ok(())
    }

    /// Pushes the bits of a floating-point value on the machine
    /// stack.
    ///
    /// The stack of the floating-point unit holds double-precision
    /// values, so a single-precision one is widened first.
    fn push_double(&self, enc: &mut Encoder<'_>, size: Size, value: u16) {
        if size == Size::Dword {
            enc.cvt_xmm2xmm(true, reg::SCRATCH_XMMS[1], value);
            enc.movq_rm_xmm(reg::RAX, reg::SCRATCH_XMMS[1]);
        } else {
            enc.movq_rm_xmm(reg::RAX, value);
        }
        enc.push(reg::RAX);
    }

    /// Expands a floating-point operation of two operands.
    fn expand_float(&self, enc: &mut Encoder<'_>, inst: &MInst, op: Arith) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        // A vector of floating-point values runs the packed form; a
        // single value runs the scalar one.
        let vector = match &inst.ty {
            MType::Vec { elem, .. } => Some(float_size(elem).ok_or_else(|| {
                target_error(
                    "expanding the machine operation",
                    "the lanes of the vector are not floating-point values",
                )
            })?),
            _ => None,
        };
        let size = match vector {
            Some(size) => size,
            None => float_size(&inst.ty).ok_or_else(|| {
                target_error(
                    "expanding the machine operation",
                    "the operation has no floating-point type",
                )
            })?,
        };
        let left = self.float_operand(enc, inst, 0)?;
        self.expect_float(left, 0)?;
        let right = self.float_operand(enc, inst, 1)?;
        self.expect_float(right, 1)?;
        match vector {
            Some(size) => {
                enc.mov_xmm_rr(Size::Xmm, dest, left);
                enc.packed_xmm_rr(op, size, dest, right);
            }
            None => {
                enc.mov_xmm_rr(size, dest, left);
                enc.arith_xmm_rr(op, size, dest, right);
            }
        }
        Ok(())
    }

    /// Expands an integer comparison.
    fn expand_icmp(&self, enc: &mut Encoder<'_>, inst: &MInst, pred: IPred) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let left = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(left, 0)?;
        // The operands are compared at the width the instruction
        // names, which is the width of the values themselves.
        let size = match Self::operand_type(inst) {
            Some(MType::I64) | Some(MType::Ptr { .. }) => Size::Qword,
            _ => Size::Dword,
        };
        match inst.args.get(1) {
            Some(Operand::Immediate(value)) => enc.cmp_ri(size, left, value),
            Some(_) => {
                let right = self.operand_reg(enc, inst, 1, &[])?;
                self.expect_int(right, 1)?;
                enc.cmp_rr(size, left, right);
            }
            None => {
                return Err(target_error(
                    "expanding the machine operation",
                    "the operation has no second operand",
                ));
            }
        }
        enc.setcc(prediction(pred), reg::RDX);
        let dst_size = if matches!(inst.ty, MType::I64 | MType::Ptr { .. }) {
            Size::Qword
        } else {
            Size::Dword
        };
        enc.movzx_rr(Size::Byte, dst_size, dest, reg::RDX);
        Ok(())
    }

    /// Expands a floating-point comparison.
    fn expand_fcmp(&self, enc: &mut Encoder<'_>, inst: &MInst, pred: FPred) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let left = self.float_operand(enc, inst, 0)?;
        self.expect_float(left, 0)?;
        let right = self.float_operand(enc, inst, 1)?;
        self.expect_float(right, 1)?;
        // The comparison runs at the precision the instruction names.
        let size = match Self::operand_type(inst) {
            Some(MType::F64) => Size::Qword,
            _ => Size::Dword,
        };
        enc.ucomis_rr(size, left, right);
        match pred {
            FPred::Oeq => {
                enc.setcc(Cc::E, reg::RDX);
                enc.setcc(Cc::Np, reg::RCX);
                enc.and_rr(Size::Byte, reg::RDX, reg::RCX);
            }
            FPred::One => {
                enc.setcc(Cc::Ne, reg::RDX);
                enc.setcc(Cc::Np, reg::RCX);
                enc.and_rr(Size::Byte, reg::RDX, reg::RCX);
            }
            FPred::Olt => {
                enc.setcc(Cc::B, reg::RDX);
                enc.setcc(Cc::Np, reg::RCX);
                enc.and_rr(Size::Byte, reg::RDX, reg::RCX);
            }
            FPred::Ole => {
                enc.setcc(Cc::Be, reg::RDX);
                enc.setcc(Cc::Np, reg::RCX);
                enc.and_rr(Size::Byte, reg::RDX, reg::RCX);
            }
            FPred::Ogt => {
                enc.setcc(Cc::A, reg::RDX);
                enc.setcc(Cc::Np, reg::RCX);
                enc.and_rr(Size::Byte, reg::RDX, reg::RCX);
            }
            FPred::Oge => {
                enc.setcc(Cc::Ae, reg::RDX);
                enc.setcc(Cc::Np, reg::RCX);
                enc.and_rr(Size::Byte, reg::RDX, reg::RCX);
            }
            FPred::Ueq => enc.setcc(Cc::Ne, reg::RDX),
            FPred::Une => enc.setcc(Cc::Ne, reg::RDX),
            FPred::False => {
                enc.xor_rr(Size::Dword, dest, dest);
                return Ok(());
            }
        }
        let dst_size = if matches!(inst.ty, MType::I64 | MType::Ptr { .. }) {
            Size::Qword
        } else {
            Size::Dword
        };
        enc.movzx_rr(Size::Byte, dst_size, dest, reg::RDX);
        Ok(())
    }

    /// Expands the selection of one of two values.
    fn expand_select(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let condition = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(condition, 0)?;
        match Self::class_of(dest) {
            Class::Int => {
                let size = int_size(&inst.ty);
                // The value of the branch that is not taken is the
                // one the destination holds first.
                match inst.args.get(2) {
                    Some(Operand::Immediate(value)) => enc.mov_ri(size, dest, value),
                    Some(_) => {
                        let when_false = self.operand_reg(enc, inst, 2, &[])?;
                        self.expect_int(when_false, 2)?;
                        enc.mov_rr(size, dest, when_false);
                    }
                    None => {
                        return Err(target_error(
                            "expanding the machine operation",
                            "the operation has no third operand",
                        ));
                    }
                }
                // The value of the branch that is taken moves into
                // the destination when the condition holds.
                let source = match inst.args.get(1) {
                    Some(Operand::Immediate(value)) => {
                        let scratch = if dest == reg::RAX { reg::RCX } else { reg::RAX };
                        enc.mov_ri(size, scratch, value);
                        scratch
                    }
                    Some(_) => {
                        let when_true = self.operand_reg(enc, inst, 1, &[])?;
                        self.expect_int(when_true, 1)?;
                        when_true
                    }
                    None => {
                        return Err(target_error(
                            "expanding the machine operation",
                            "the operation has no second operand",
                        ));
                    }
                };
                enc.test_rr(Size::Dword, condition, condition);
                enc.cmovcc(Cc::Ne, size, dest, source);
            }
            Class::Float => {
                let when_true = self.operand_reg(enc, inst, 1, &[])?;
                let when_false = self.operand_reg(enc, inst, 2, &[])?;
                self.expect_float(when_true, 1)?;
                self.expect_float(when_false, 2)?;
                // The mask is all ones when the condition holds and
                // all zeroes otherwise.  The mask covers a whole
                // quad-word, so that a double-precision value is
                // selected as well as a single-precision one.
                enc.xor_rr(Size::Dword, reg::RAX, reg::RAX);
                enc.test_rr(Size::Dword, condition, condition);
                enc.setcc(Cc::Ne, reg::RAX);
                enc.neg_rr(Size::Qword, reg::RAX);
                enc.movq_xmm_rm(reg::SCRATCH_XMMS[1], reg::RAX);
                enc.punpcklqdq(reg::SCRATCH_XMMS[1], reg::SCRATCH_XMMS[1]);
                enc.mov_xmm_rr(Size::Xmm, reg::SCRATCH_XMMS[0], reg::SCRATCH_XMMS[1]);
                enc.andnps_rr(reg::SCRATCH_XMMS[0], when_false);
                enc.mov_xmm_rr(Size::Xmm, dest, when_true);
                enc.andps_rr(dest, reg::SCRATCH_XMMS[1]);
                enc.orps_rr(dest, reg::SCRATCH_XMMS[0]);
            }
        }
        Ok(())
    }

    /// The width of the integer an operand holds.
    ///
    /// The width follows the type of the values the instruction
    /// reads, which the instruction names.
    fn integer_width(&self, inst: &MInst) -> Size {
        match Self::operand_type(inst) {
            Some(MType::I8) => Size::Byte,
            Some(MType::I16) => Size::Word,
            Some(MType::I32) => Size::Dword,
            Some(MType::I64) | Some(MType::Ptr { .. }) => Size::Qword,
            _ => Size::Dword,
        }
    }

    /// Expands a zero extension.
    fn expand_zext(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(value, 0)?;
        let dst_size = int_size(&inst.ty);
        let src_size = self.integer_width(inst);
        match (dst_size, src_size) {
            (Size::Qword, Size::Byte) => enc.movzx_rr(Size::Byte, Size::Qword, dest, value),
            (Size::Qword, Size::Word) => enc.movzx_rr(Size::Word, Size::Qword, dest, value),
            (Size::Dword, Size::Byte) => enc.movzx_rr(Size::Byte, Size::Dword, dest, value),
            (Size::Dword, Size::Word) => enc.movzx_rr(Size::Word, Size::Dword, dest, value),
            // The thirty-two-bit form of the move clears the upper
            // half of the register, which is what the extension asks
            // for.
            _ => enc.mov_rr(Size::Dword, dest, value),
        }
        Ok(())
    }

    /// Expands a sign extension.
    fn expand_sext(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(value, 0)?;
        let dst_size = int_size(&inst.ty);
        let src_size = self.integer_width(inst);
        match (dst_size, src_size) {
            (Size::Qword, Size::Byte) => enc.movsx_rr(Size::Byte, Size::Qword, dest, value),
            (Size::Qword, Size::Word) => enc.movsx_rr(Size::Word, Size::Qword, dest, value),
            // The thirty-two-bit form of the move keeps the sign of
            // the value in the whole register.
            (Size::Qword, Size::Dword) => enc.movsxd_rr(dest, value),
            (Size::Dword, Size::Byte) => enc.movsx_rr(Size::Byte, Size::Dword, dest, value),
            (Size::Dword, Size::Word) => enc.movsx_rr(Size::Word, Size::Dword, dest, value),
            _ => enc.mov_rr(Size::Dword, dest, value),
        }
        Ok(())
    }

    /// Expands a truncation.
    fn expand_trunc(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_int(value, 0)?;
        match inst.ty {
            MType::I1 => enc.and_ri(Size::Dword, dest, 1),
            MType::I8 => enc.movzx_rr(Size::Byte, Size::Dword, dest, value),
            MType::I16 => enc.movzx_rr(Size::Word, Size::Dword, dest, value),
            _ => enc.mov_rr(Size::Dword, dest, value),
        }
        Ok(())
    }

    /// Expands the conversion of a floating-point value into an
    /// unsigned integer.
    ///
    /// The baseline converts to a signed integer only, so a
    /// sixty-four-bit unsigned value is converted in two steps: the
    /// difference of the value and two to the sixty-third is a signed
    /// integer, and the high bit is added back to it.
    fn expand_fp_to_ui(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.float_operand(enc, inst, 0)?;
        let to_double = Self::source_is_double(inst);
        match inst.ty {
            MType::I64 => {
                // The bits of two to the sixty-third.
                const HIGH: i64 = 0x43E0_0000_0000_0000;
                // The value is widened to double precision first: the
                // difference of the value and two to the sixty-third
                // is then exact.
                let wide = reg::SCRATCH_XMMS[1];
                let value = self.float_operand(enc, inst, 0)?;
                if to_double {
                    enc.mov_xmm_rr(Size::Qword, wide, value);
                } else {
                    enc.cvt_xmm2xmm(true, wide, value);
                }
                enc.mov_ri(Size::Qword, reg::RAX, HIGH);
                enc.movq_xmm_rm(reg::SCRATCH_XMMS[0], reg::RAX);
                // The value below two to the sixty-third fits in a
                // signed integer; the one above it does not.
                enc.cvtts_xmm2si(true, Size::Qword, reg::RAX, wide);
                // The comparison tells the two apart, and it comes
                // before the instructions that change the flags.
                enc.ucomis_rr(Size::Qword, wide, reg::SCRATCH_XMMS[0]);
                // The high bit is taken away, and the value that is
                // left fits in a signed integer.
                enc.arith_xmm_rr(Arith::Sub, Size::Qword, wide, reg::SCRATCH_XMMS[0]);
                enc.cvtts_xmm2si(true, Size::Qword, reg::RCX, wide);
                // The high bit is put back: the value that is left
                // holds no bit of its own above the sixty-third, so
                // setting the bit is the same as adding it.
                enc.mov_ri(Size::Qword, reg::RDX, i64::MIN);
                enc.xor_rr(Size::Qword, reg::RCX, reg::RDX);
                enc.cmovcc(Cc::Ae, Size::Qword, reg::RAX, reg::RCX);
                enc.mov_rr(Size::Qword, dest, reg::RAX);
            }
            MType::I32 => {
                // The conversion is exact for every value a
                // thirty-two-bit unsigned integer holds.
                enc.cvtts_xmm2si(to_double, Size::Qword, reg::RAX, value);
                enc.mov_rr(Size::Dword, dest, reg::RAX);
            }
            _ => {
                return Err(unsupported_error(
                    "the conversion of a floating-point value into an unsigned integer",
                    "the target converts into a thirty-two-bit or a sixty-four-bit integer",
                ));
            }
        }
        Ok(())
    }

    /// Whether the value an instruction reads is a double-precision
    /// one.
    fn source_is_double(inst: &MInst) -> bool {
        matches!(Self::operand_type(inst), Some(MType::F64))
    }

    /// Expands the conversion of a floating-point value into a signed
    /// integer.
    fn expand_fp_to_si(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.float_operand(enc, inst, 0)?;
        let to_double = Self::source_is_double(inst);
        match inst.ty {
            MType::I64 => {
                enc.cvtts_xmm2si(to_double, Size::Qword, dest, value);
            }
            MType::I32 => {
                enc.cvtts_xmm2si(to_double, Size::Dword, dest, value);
            }
            _ => {
                return Err(unsupported_error(
                    "the conversion of a floating-point value into a signed integer",
                    "the target converts into a thirty-two-bit or a sixty-four-bit integer",
                ));
            }
        }
        Ok(())
    }

    /// Expands the conversion of an unsigned integer into a
    /// floating-point value.
    ///
    /// The baseline converts a signed integer only, so a
    /// sixty-four-bit unsigned value is halved first: the quotient is
    /// exact, and the odd bit is added back afterwards.
    fn expand_ui_to_fp(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.int_operand(enc, inst, 0)?;
        let to_double = float_size(&inst.ty).unwrap_or(Size::Dword) == Size::Qword;
        // The width of the source decides the conversion: a
        // thirty-two-bit unsigned integer is converted exactly, a
        // sixty-four-bit one is not.
        let src_size = match Self::operand_type(inst) {
            Some(MType::I64) | Some(MType::Ptr { .. }) => Size::Qword,
            _ => Size::Dword,
        };
        if src_size == Size::Qword {
            // The quotient of the value and two is exact, and the bit
            // that was shifted out is added back as a floating-point
            // value.
            enc.mov_rr(Size::Qword, reg::RAX, value);
            enc.shift_ri(5, Size::Qword, reg::RAX, 1);
            enc.mov_rr(Size::Qword, reg::RCX, value);
            enc.and_ri(Size::Qword, reg::RCX, 1);
            enc.cvtsi2_xmm(true, Size::Qword, reg::SCRATCH_XMMS[0], reg::RAX);
            enc.arith_xmm_rr(
                Arith::Add,
                Size::Qword,
                reg::SCRATCH_XMMS[0],
                reg::SCRATCH_XMMS[0],
            );
            enc.cvtsi2_xmm(true, Size::Dword, reg::SCRATCH_XMMS[1], reg::RCX);
            enc.arith_xmm_rr(
                Arith::Add,
                Size::Qword,
                reg::SCRATCH_XMMS[0],
                reg::SCRATCH_XMMS[1],
            );
            if to_double {
                enc.mov_xmm_rr(Size::Qword, dest, reg::SCRATCH_XMMS[0]);
            } else {
                enc.cvt_xmm2xmm(false, dest, reg::SCRATCH_XMMS[0]);
            }
            return Ok(());
        }
        // The thirty-two-bit form of the move clears the upper half
        // of the register, so the value is exact.
        enc.mov_rr(Size::Dword, reg::RAX, value);
        enc.cvtsi2_xmm(to_double, Size::Qword, dest, reg::RAX);
        Ok(())
    }

    /// Expands the conversion of a signed integer into a floating-point
    /// value.
    fn expand_si_to_fp(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.int_operand(enc, inst, 0)?;
        let size = float_size(&inst.ty).unwrap_or(Size::Dword);
        // The source is read at the width the instruction names.
        let src_size = match Self::operand_type(inst) {
            Some(MType::I64) | Some(MType::Ptr { .. }) => Size::Qword,
            _ => Size::Dword,
        };
        enc.cvtsi2_xmm(size == Size::Qword, src_size, dest, value);
        Ok(())
    }

    /// Expands the conversion of a floating-point value between the
    /// two widths.
    fn expand_fp_width(
        &self,
        enc: &mut Encoder<'_>,
        inst: &MInst,
        to_double: bool,
    ) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.operand_reg(enc, inst, 0, &[])?;
        self.expect_float(value, 0)?;
        enc.cvt_xmm2xmm(to_double, dest, value);
        Ok(())
    }

    /// Expands the reinterpretation of the bits of a value.
    fn expand_bitcast(&self, enc: &mut Encoder<'_>, inst: &MInst) -> Result<(), CodegenError> {
        let dest = self.dest(inst)?;
        self.expect_dest(inst, dest)?;
        let value = self.operand_reg(enc, inst, 0, &[])?;
        match (Self::class_of(dest), Self::class_of(value)) {
            (Class::Int, Class::Int) => enc.mov_rr(int_size(&inst.ty), dest, value),
            (Class::Float, Class::Float) => {
                let size = float_size(&inst.ty).unwrap_or(Size::Xmm);
                enc.mov_xmm_rr(size, dest, value);
            }
            (Class::Int, Class::Float) => {
                if int_size(&inst.ty) == Size::Qword {
                    enc.movq_rm_xmm(dest, value);
                } else {
                    enc.movd_rm_xmm(dest, value);
                }
            }
            (Class::Float, Class::Int) => {
                let size = float_size(&inst.ty).unwrap_or(Size::Dword);
                if size == Size::Qword {
                    enc.movq_xmm_rm(dest, value);
                } else {
                    enc.movd_xmm_rm(dest, value);
                }
            }
        }
        Ok(())
    }

    /// Expands a call.
    ///
    /// The arguments move into the argument registers, the call
    /// reaches the callee with a relative displacement, and the result
    /// moves out of the register the calling convention returns it in.
    fn expand_call(
        &self,
        enc: &mut Encoder<'_>,
        fixups: &mut Vec<Fixup>,
        inst: &MInst,
        callee: SymId,
    ) -> Result<(), CodegenError> {
        let target = self
            .module
            .function_index(callee)
            .ok_or_else(|| {
                target_error(
                    "expanding the machine operation",
                    "the call names a symbol that is not a function of the module",
                )
            })?;
        let mut integer_args = 0usize;
        let mut float_args = 0usize;
        // The arguments that are already in a register are read
        // before anything is moved, so a scratch register must not be
        // one of them.
        let mut held: Vec<u16> = Vec::with_capacity(inst.args.len());
        for operand in inst.args.as_slice() {
            if let Operand::Physical(reg) = operand {
                held.push(reg.number());
            }
        }
        let parameters = self.parameter_types(target);
        for index in 0..inst.args.len() {
            let value = match inst.args.get(index) {
                Some(Operand::Immediate(value)) => {
                    // An immediate is an integer argument unless the
                    // callee takes a floating-point value there.
                    let floating = parameters
                        .get(index)
                        .is_some_and(|ty| ty.is_floating());
                    if floating {
                        return Err(target_error(
                            "expanding the machine operation",
                            "a floating-point argument must be a value, not an immediate",
                        ));
                    }
                    let scratch = self.scratch_register(inst, &held)?;
                    enc.mov_ri(Size::Qword, scratch, value);
                    scratch
                }
                _ => self.operand_reg(enc, inst, index, &held)?,
            };
            match Self::class_of(value) {
                Class::Int => {
                    if integer_args >= reg::ARG_REGS.len() {
                        return Err(target_error(
                            "expanding the machine operation",
                            "the call has more than six integer arguments",
                        ));
                    }
                    let arg = reg::ARG_REGS[integer_args];
                    integer_args += 1;
                    if arg != value {
                        enc.mov_rr(Size::Qword, arg, value);
                    }
                }
                Class::Float => {
                    if float_args >= reg::FLOAT_ARG_REGS.len() {
                        return Err(target_error(
                            "expanding the machine operation",
                            "the call has more than eight floating-point arguments",
                        ));
                    }
                    let arg = reg::FLOAT_ARG_REGS[float_args];
                    float_args += 1;
                    if arg != value {
                        enc.mov_xmm_rr(Size::Qword, arg, value);
                    }
                }
            }
        }
        let site = enc.call_rel32();
        fixups.push(Fixup::function(site, target));
        if let Some(Destination::Physical(dest)) = inst.dest {
            let number = dest.number();
            if !reg::is_valid(number) {
                return Err(target_error(
                    "expanding the machine operation",
                    "the destination of the call names a register the file does not hold",
                ));
            }
            if inst.ty.is_floating() {
                enc.mov_xmm_rr(Size::Qword, number, reg::XMM0);
            } else if inst.ty.is_vector() {
                enc.mov_xmm_rr(Size::Xmm, number, reg::XMM0);
            } else if inst.ty != MType::Void {
                enc.mov_rr(Size::Qword, number, reg::RAX);
            }
        }
        Ok(())
    }
}

/// The integer operations of two operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Binary {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
}

impl Binary {
    /// The name of the operation, for the diagnostics.
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "addition",
            Self::Sub => "subtraction",
            Self::Mul => "multiplication",
            Self::And => "bitwise and",
            Self::Or => "bitwise or",
            Self::Xor => "bitwise exclusive or",
        }
    }
}

/// The operations of one operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Unary {
    /// Negation.
    Neg,
    /// Bitwise complement.
    Not,
}

/// The kinds of shift.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shift {
    /// Left shift.
    Left,
    /// Logical right shift.
    LogicalRight,
    /// Arithmetic right shift.
    ArithmeticRight,
}

/// The condition code of an integer comparison.
fn prediction(pred: IPred) -> Cc {
    match pred {
        IPred::Eq => Cc::E,
        IPred::Ne => Cc::Ne,
        IPred::Ugt => Cc::A,
        IPred::Uge => Cc::Ae,
        IPred::Ult => Cc::B,
        IPred::Ule => Cc::Be,
        IPred::Sgt => Cc::G,
        IPred::Sge => Cc::Ge,
        IPred::Slt => Cc::L,
        IPred::Sle => Cc::Le,
    }
}

/// The callee-saved registers a function uses.
///
/// The registers the expander reserves for itself — the argument
/// registers and the accumulator — are never live, so they never
/// appear here.
fn saved_registers(function: &MirFunction) -> Vec<u16> {
    let mut used = [false; reg::NUM_GPR as usize];
    for block in &function.blocks {
        for inst in &block.insts {
            if let Some(Destination::Physical(dest)) = inst.dest {
                let number = dest.number();
                if !reg::is_xmm(number) && number < reg::NUM_GPR {
                    used[number as usize] = true;
                }
            }
            for operand in inst.args.as_slice() {
                let number = match operand {
                    Operand::Physical(reg) => reg.number(),
                    _ => continue,
                };
                if !reg::is_xmm(number) && number < reg::NUM_GPR {
                    used[number as usize] = true;
                }
            }
        }
    }
    let mut saved: Vec<u16> = Vec::new();
    for reg in reg::CALLEE_SAVED {
        if used[reg as usize] {
            saved.push(reg);
        }
    }
    // The stack has to stay aligned for the calls the function makes,
    // and the number of saved registers decides the alignment.  When
    // the count is even one more register is saved, so that the
    // stack pointer is aligned when the function calls.  A register
    // the function does not use is saved for that purpose: saving it
    // and restoring it changes nothing.
    if saved.len().is_multiple_of(2) {
        for reg in reg::CALLEE_SAVED
            .iter()
            .chain(reg::PADDING_REGS.iter())
        {
            if !saved.contains(reg) {
                saved.push(*reg);
                break;
            }
        }
    }
    saved
}

/// The size of the frame a function reserves.
///
/// The frame is a whole number of sixteen-byte units, so that the
/// stack stays aligned.
fn frame_size(function: &MirFunction) -> u32 {
    function.frame_size.div_ceil(16) * 16
}

/// Expands a function into the bytes of its instructions.
pub(crate) fn expand_function(module: &MirModule, function: &MirFunction) -> Result<Expanded, CodegenError> {
    let mut expander = Expander::new(module, function);
    let mut bytes: Vec<u8> = Vec::with_capacity(64);
    let mut fixups: Vec<Fixup> = Vec::new();
    let mut block_offsets: Vec<u32> = Vec::with_capacity(function.blocks.len());
    let instruction_count = {
        let mut encoder = Encoder::new(&mut bytes);
        expander.emit_prologue(&mut encoder);
        for block in &function.blocks {
            block_offsets.push(encoder.position());
            for inst in &block.insts {
                expander.expand_inst(&mut encoder, &mut fixups, inst)?;
            }
        }
        encoder.count()
    };
    Ok(Expanded {
        bytes,
        fixups,
        block_offsets,
        instruction_count,
    })
}
