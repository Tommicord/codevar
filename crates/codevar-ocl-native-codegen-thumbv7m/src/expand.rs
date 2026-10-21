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
//! The expansion works on physical registers.  A module that has been
//! through the register allocator names them directly; a module that
//! has not is given a home here, by a scan that keeps the values of
//! a function in the callee-saved registers.  The scan refuses a
//! function whose values do not fit, and asks for the module to be
//! allocated before it can be compiled.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_abi::{ARG_BASE_OFFSET, ARG_SLOT};
use codevar_ocl_native_codegen::error::{CodegenError, Result};
use codevar_ocl_native_codegen::mir::{
    ArgKind, Destination, IPred, MInst, MOp, MType, MirFunction, Operand, VReg,
};

use crate::encode;
use crate::reg;

/// The register the expansion uses for the values it computes on the
/// way, when it needs one of its own.
pub const SCRATCH: u8 = 12;

/// The first callee-saved register the allocation uses.
const ALLOC_FIRST: u8 = 4;

/// The last callee-saved register the allocation uses.
const ALLOC_LAST: u8 = 11;

/// The size of the argument slot of the kernel arguments.
const ARG_SLOT_SIZE: u32 = ARG_SLOT as u32;

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

/// The home of every virtual register of a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegAlloc {
    /// The register of every virtual register; `None` when the
    /// virtual register does not appear in the function.
    homes: Vec<Option<u8>>,
}

impl RegAlloc {
    /// The register of the virtual register, when it has one.
    #[must_use]
    #[inline]
    pub fn home(&self, reg: VReg) -> Option<u8> {
        self.homes.get(reg.index()).copied().flatten()
    }
}

/// Gives every virtual register of the function a home.
///
/// The values live in the callee-saved registers, so a call inside
/// the function does not disturb them.  The scan walks the
/// instructions in order and keeps the values whose range is still
/// open; when no register is free the function is refused, because
/// the module has to be allocated before it can be compiled.
fn allocate(function: &MirFunction) -> Result<RegAlloc> {
    let count = function.vreg_types.len();
    let mut start = vec![usize::MAX; count];
    let mut end = vec![0usize; count];
    let mut index = 0usize;

    for block in &function.blocks {
        for inst in &block.insts {
            if let Some(Destination::Virtual(dest)) = inst.dest {
                let slot = dest.index();
                if slot < count {
                    start[slot] = start[slot].min(index);
                    end[slot] = end[slot].max(index);
                }
            }
            for operand in inst.args.as_slice() {
                if let Operand::Virtual(value) = operand {
                    let slot = value.index();
                    if slot < count {
                        start[slot] = start[slot].min(index);
                        end[slot] = end[slot].max(index);
                    }
                }
            }
            index += 1;
        }
    }

    let mut order: Vec<usize> = (0..count)
        .filter(|&slot| start[slot] != usize::MAX)
        .collect();
    order.sort_by_key(|&slot| start[slot]);

    let mut homes: Vec<Option<u8>> = vec![None; count];
    let mut active: Vec<(u8, usize)> = Vec::new();
    let mut free: Vec<u8> = (ALLOC_FIRST..=ALLOC_LAST).rev().collect();

    for &slot in &order {
        let first = start[slot];
        let mut expired = Vec::new();
        active.retain(|&(number, last)| {
            if last < first {
                expired.push(number);
                false
            } else {
                true
            }
        });
        free.extend(expired);
        free.sort_unstable_by(|a, b| b.cmp(a));

        let Some(&number) = free.last() else {
            return Err(CodegenError::Target {
                what: "allocating the registers of the function",
                detail: String::from(
                    "the function keeps more values live than the register file holds",
                ),
            });
        };
        free.pop();
        homes[slot] = Some(number);
        active.push((number, end[slot]));
    }

    Ok(RegAlloc { homes })
}

/// The registers of the function that survive a call, and whether it
/// calls at all.
///
/// The prologue saves the registers the function uses, so the scan
/// has to know them before the frame is reserved: the frame and the
/// saved registers together have to keep the stack pointer aligned.
fn scan_registers(function: &MirFunction, alloc: &RegAlloc) -> (u16, bool) {
    let mut saved = 0u16;
    let mut uses_link = false;
    for block in &function.blocks {
        for inst in &block.insts {
            if matches!(inst.op, MOp::Call { .. }) {
                uses_link = true;
            }
            if let Some(dest) = inst.dest {
                match dest {
                    Destination::Virtual(value) => {
                        if let Some(number) = alloc.home(value) {
                            saved |= saved_bit(number);
                        }
                    }
                    Destination::Physical(reg) => {
                        if let Some(number) = reg::gpr_number(reg) {
                            saved |= saved_bit(number);
                        }
                    }
                }
            }
            for operand in inst.args.as_slice() {
                let number = match operand {
                    Operand::Physical(reg) => reg::gpr_number(*reg),
                    Operand::Virtual(value) => alloc.home(*value),
                    _ => None,
                };
                if let Some(number) = number {
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
    if (ALLOC_FIRST..=ALLOC_LAST).contains(&number) {
        1 << number
    } else {
        0
    }
}

/// The expansion of one function.
struct Expander<'a> {
    function: &'a MirFunction,
    alloc: &'a RegAlloc,
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
        alloc: &'a RegAlloc,
        callees: &'a [Option<usize>],
        has_fp: bool,
        saved: u16,
        uses_link: bool,
    ) -> Self {
        let pushed = u32::from(saved.count_ones()) + u32::from(uses_link);
        let frame = align_frame(function.frame_size, pushed);
        Self {
            function,
            alloc,
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

    /// The general-purpose register of the operand.
    fn gpr(&self, operand: Operand) -> Result<u8> {
        match operand {
            Operand::Physical(reg) => {
                reg::gpr_number(reg).ok_or_else(|| CodegenError::Target {
                    what: "reading an operand",
                    detail: String::from("the operand is not in a general-purpose register"),
                })
            }
            Operand::Virtual(value) => {
                self.alloc.home(value).ok_or_else(|| CodegenError::Target {
                    what: "reading an operand",
                    detail: String::from("the operand is a virtual register without a home"),
                })
            }
            Operand::Immediate(_) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: String::from("the operand is an immediate"),
            }),
            Operand::FloatImmediate(_) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: String::from("the operand is a floating-point immediate"),
            }),
            Operand::Symbol(_) => Err(CodegenError::Target {
                what: "reading an operand",
                detail: String::from("the operand is a symbol"),
            }),
        }
    }

    /// The operand of the instruction at the given position.
    fn arg(&self, inst: &MInst, index: usize) -> Result<Operand> {
        inst.args.get(index).ok_or_else(|| CodegenError::Target {
            what: "expanding an operation",
            detail: String::from("the operation has too few operands"),
        })
    }

    /// The general-purpose register the instruction writes.
    fn dest(&self, inst: &MInst) -> Result<u8> {
        let dest = inst.dest.ok_or_else(|| CodegenError::Target {
            what: "expanding an operation",
            detail: String::from("the operation writes no register"),
        })?;
        match dest {
            Destination::Virtual(value) => self.gpr(Operand::Virtual(value)),
            Destination::Physical(reg) => {
                reg::gpr_number(reg).ok_or_else(|| CodegenError::Target {
                    what: "expanding an operation",
                    detail: String::from("the destination is not in a general-purpose register"),
                })
            }
        }
    }

    /// The type of the virtual register of an operand.
    fn operand_type(&self, operand: Operand) -> MType {
        match operand {
            Operand::Virtual(value) => self
                .function
                .vreg_type(value)
                .cloned()
                .unwrap_or(MType::I32),
            _ => MType::I32,
        }
    }

    /// Records that the expansion used a callee-saved register.
    fn use_saved(&mut self, number: u8) {
        self.saved |= saved_bit(number);
    }

    /// Emits the move of one register into another.
    fn emit_move(&mut self, rd: u8, rm: u8) {
        self.use_saved(rd);
        self.use_saved(rm);
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

    /// Emits the move of an immediate into a register.
    fn emit_mov_imm(&mut self, rd: u8, value: u32) {
        self.use_saved(rd);
        if value <= 0xff && rd < 8 {
            encode::movs_imm(&mut self.bytes, rd, u8::try_from(value).unwrap_or(0));
        } else if value <= 0xffff {
            encode::movw(&mut self.bytes, rd, u16::try_from(value).unwrap_or(0));
        } else {
            encode::movw(&mut self.bytes, rd, u16::try_from(value & 0xffff).unwrap_or(0));
            encode::movt(&mut self.bytes, rd, u16::try_from((value >> 16) & 0xffff).unwrap_or(0));
            self.instructions += 1;
        }
        self.instructions += 1;
    }

    /// Emits the addition of two registers.
    fn emit_add(&mut self, rd: u8, rn: u8, rm: u8) {
        self.use_saved(rd);
        self.use_saved(rn);
        self.use_saved(rm);
        if rd < 8 && rn < 8 && rm < 8 {
            encode::adds_reg3(&mut self.bytes, rd, rn, rm);
        } else {
            encode::adds_reg_wide(&mut self.bytes, rd, rn, rm);
        }
        self.instructions += 1;
    }

    /// Emits the addition of a register and a shifted register.
    fn emit_add_shifted(&mut self, rd: u8, rn: u8, rm: u8, shift: u8) {
        self.use_saved(rd);
        self.use_saved(rn);
        self.use_saved(rm);
        encode::adds_reg_shifted(&mut self.bytes, rd, rn, rm, shift);
        self.instructions += 1;
    }

    /// Emits the subtraction of two registers.
    fn emit_sub(&mut self, rd: u8, rn: u8, rm: u8) {
        self.use_saved(rd);
        self.use_saved(rn);
        self.use_saved(rm);
        if rd < 8 && rn < 8 && rm < 8 {
            encode::subs_reg3(&mut self.bytes, rd, rn, rm);
        } else {
            encode::subs_reg_wide(&mut self.bytes, rd, rn, rm);
        }
        self.instructions += 1;
    }

    /// Emits the comparison of two registers.
    fn emit_cmp(&mut self, rn: u8, rm: u8) {
        self.use_saved(rn);
        self.use_saved(rm);
        if rn < 8 && rm < 8 {
            encode::cmp_reg(&mut self.bytes, rn, rm);
        } else {
            encode::cmp_reg_wide(&mut self.bytes, rn, rm);
        }
        self.instructions += 1;
    }

    /// Emits the comparison of a register with an immediate.
    fn emit_cmp_imm(&mut self, rn: u8, value: u32) {
        self.use_saved(rn);
        if value <= 0xff {
            encode::cmp_imm(&mut self.bytes, rn, u8::try_from(value).unwrap_or(0));
        } else {
            self.emit_mov_imm(SCRATCH, value);
            encode::cmp_reg_wide(&mut self.bytes, rn, SCRATCH);
            self.instructions += 1;
        }
        self.instructions += 1;
    }

    /// Emits the extension of a register.
    fn emit_extend(&mut self, rd: u8, rm: u8, signed: bool, halfword: bool) {
        self.use_saved(rd);
        self.use_saved(rm);
        if rd < 8 && rm < 8 {
            match (signed, halfword) {
                (false, false) => encode::uxtb_reg(&mut self.bytes, rd, rm),
                (false, true) => encode::uxth_reg(&mut self.bytes, rd, rm),
                (true, false) => encode::sxtb_reg(&mut self.bytes, rd, rm),
                (true, true) => encode::sxth_reg(&mut self.bytes, rd, rm),
            }
        } else {
            let first = match (signed, halfword) {
                (false, false) => 0xfa5fu16,
                (false, true) => 0xfa1fu16,
                (true, false) => 0xfa4fu16,
                (true, true) => 0xfa0fu16,
            };
            let second = 0xf000u16 | u16::from(rd) << 8 | 0x80 | u16::from(rm);
            self.bytes
                .push(u8::try_from(first & 0xff).unwrap_or(0));
            self.bytes
                .push(u8::try_from((first >> 8) & 0xff).unwrap_or(0));
            self.bytes
                .push(u8::try_from(second & 0xff).unwrap_or(0));
            self.bytes
                .push(u8::try_from((second >> 8) & 0xff).unwrap_or(0));
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
            let mut registers = self.saved;
            registers |= 1 << 14;
            encode::pop(&mut self.bytes, registers & 0x0fff, true);
        } else {
            if self.saved != 0 {
                encode::pop(&mut self.bytes, self.saved & 0x0fff, false);
                self.instructions += 1;
            }
            encode::bx(&mut self.bytes, reg::LR);
        }
        self.instructions += 1;
    }

    /// Emits the unsigned division of two registers.
    ///
    /// The quotient ends up in the first register and the remainder
    /// in the fourth.  The division has no instruction, so it becomes
    /// a loop that shifts the dividend into the remainder.
    fn emit_unsigned_division(&mut self) {
        // The registers the loop uses are saved and restored.
        encode::push(&mut self.bytes, 0x3c, false);
        self.instructions += 1;
        self.emit_unsigned_division_body();
        encode::pop(&mut self.bytes, 0x3c, false);
        self.instructions += 1;
    }

    /// Expands one operation.
    fn expand(&mut self, inst: &MInst) -> Result<()> {
        match &inst.op {
            MOp::Copy => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                self.emit_move(rd, rm);
            }
            MOp::Load { size } => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                self.emit_load(rd, rn, *size)?;
            }
            MOp::Store { size } => {
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_store(rm, rn, *size)?;
            }
            MOp::Gep { size, signed } => {
                let rd = self.dest(inst)?;
                let base = self.gpr(self.arg(inst, 0)?)?;
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
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_add(rd, rn, rm);
            }
            MOp::Sub => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_sub(rd, rn, rm);
            }
            MOp::Mul => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.use_saved(rd);
                encode::muls_reg_wide(&mut self.bytes, rd, rn, rm);
                self.instructions += 1;
            }
            MOp::And => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_logic(rd, rn, rm, Logic::And);
            }
            MOp::Or => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_logic(rd, rn, rm, Logic::Or);
            }
            MOp::Xor => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_logic(rd, rn, rm, Logic::Xor);
            }
            MOp::Shl => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                let rs = self.gpr(self.arg(inst, 1)?)?;
                self.use_saved(rd);
                encode::lsls_reg(&mut self.bytes, rd, rm, rs);
                self.instructions += 1;
            }
            MOp::LShr => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                let rs = self.gpr(self.arg(inst, 1)?)?;
                self.use_saved(rd);
                encode::lsrs_reg(&mut self.bytes, rd, rm, rs);
                self.instructions += 1;
            }
            MOp::AShr => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                let rs = self.gpr(self.arg(inst, 1)?)?;
                self.use_saved(rd);
                encode::asrs_reg(&mut self.bytes, rd, rm, rs);
                self.instructions += 1;
            }
            MOp::Neg => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                self.use_saved(rd);
                if rd < 8 && rm < 8 {
                    encode::negs_reg(&mut self.bytes, rd, rm);
                } else {
                    encode::negs_reg_wide(&mut self.bytes, rd, rm);
                }
                self.instructions += 1;
            }
            MOp::Not => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                self.use_saved(rd);
                if rd < 8 && rm < 8 {
                    encode::mvns_reg(&mut self.bytes, rd, rm);
                } else {
                    encode::mvn_reg_wide(&mut self.bytes, rd, rm);
                }
                self.instructions += 1;
            }
            MOp::UDiv | MOp::URem | MOp::SDiv | MOp::SRem => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_division(inst, rd, rn, rm)?;
            }
            MOp::ICmp { pred } => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let rm = self.gpr(self.arg(inst, 1)?)?;
                self.emit_compare(rd, rn, rm, *pred)?;
            }
            MOp::FCmp { .. } => {
                return Err(self.floating_point_error("a floating-point comparison"));
            }
            MOp::FAdd | MOp::FSub | MOp::FMul | MOp::FDiv | MOp::FRem | MOp::FNeg => {
                return Err(self.floating_point_error("a floating-point operation"));
            }
            MOp::Select => {
                let rd = self.dest(inst)?;
                let cond = self.gpr(self.arg(inst, 0)?)?;
                let yes = self.gpr(self.arg(inst, 1)?)?;
                let no = self.gpr(self.arg(inst, 2)?)?;
                self.emit_select(rd, cond, yes, no);
            }
            MOp::ZExt => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                let ty = self.operand_type(self.arg(inst, 0)?);
                self.emit_extend(rd, rm, false, ty == MType::I16);
            }
            MOp::SExt => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                let ty = self.operand_type(self.arg(inst, 0)?);
                self.emit_extend(rd, rm, true, ty == MType::I16);
            }
            MOp::Trunc => {
                let rd = self.dest(inst)?;
                let rm = self.gpr(self.arg(inst, 0)?)?;
                let ty = inst.ty.clone();
                match ty {
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
                let rm = self.gpr(self.arg(inst, 0)?)?;
                if inst.ty.is_floating() || self.operand_type(self.arg(inst, 0)?).is_floating() {
                    return Err(self.floating_point_error("a bit-level reinterpretation"));
                }
                self.emit_move(rd, rm);
            }
            MOp::Br { target } => {
                self.emit_branch(FixupKind::Branch, FixupTarget::Block(*target));
            }
            MOp::CondBr => {
                let cond = self.gpr(self.arg(inst, 0)?)?;
                let yes = self.block_target(self.arg(inst, 1)?)?;
                let no = self.block_target(self.arg(inst, 2)?)?;
                self.emit_cmp_imm(cond, 0);
                self.emit_branch(FixupKind::BranchCond(encode::COND_NE), FixupTarget::Block(yes));
                self.emit_branch(FixupKind::Branch, FixupTarget::Block(no));
            }
            MOp::Ret => {
                let rm = self.gpr(self.arg(inst, 0)?)?;
                self.emit_move(0, rm);
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
                let index = self.callees.get(callee.index()).copied().flatten().ok_or_else(|| {
                    CodegenError::Unsupported {
                        what: "a call",
                        detail: String::from(
                            "the callee is not a function of the module, so it has no relative address",
                        ),
                    }
                })?;
                let count = inst.args.len();
                if count > 4 {
                    return Err(CodegenError::Unsupported {
                        what: "a call",
                        detail: String::from("the architecture passes four arguments in registers"),
                    });
                }
                for position in 0..count {
                    let rm = self.gpr(self.arg(inst, position)?)?;
                    self.emit_move(u8::try_from(position).unwrap_or(0), rm);
                }
                self.uses_link = true;
                self.emit_branch(FixupKind::BranchLink, FixupTarget::Function(index));
                if let Some(dest) = inst.dest {
                    let rd = match dest {
                        Destination::Virtual(value) => self.gpr(Operand::Virtual(value))?,
                        Destination::Physical(reg) => {
                            reg::gpr_number(reg).ok_or_else(|| CodegenError::Target {
                                what: "expanding a call",
                                detail: String::from(
                                    "the result is not in a general-purpose register",
                                ),
                            })?
                        }
                    };
                    self.emit_move(rd, 0);
                }
            }
            MOp::WorkItemLoad { field, lane } => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let offset = field.offset() + lane * 4;
                self.emit_load_imm(rd, rn, offset, 4);
            }
            MOp::ArgBlockBase => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                self.emit_load_imm(rd, rn, ARG_BASE_OFFSET, 4);
            }
            MOp::ArgLoad { index, kind } => {
                let rd = self.dest(inst)?;
                let rn = self.gpr(self.arg(inst, 0)?)?;
                let offset = index * ARG_SLOT_SIZE;
                match kind {
                    ArgKind::Pointer => self.emit_load_imm(rd, rn, offset, 4),
                    ArgKind::Scalar { size } => self.emit_load_imm(rd, rn, offset, *size),
                }
            }
            MOp::Param { index } => {
                let rd = self.dest(inst)?;
                let source = u8::try_from(*index).unwrap_or(0);
                if *index >= 4 {
                    return Err(CodegenError::Unsupported {
                        what: "a parameter",
                        detail: String::from(
                            "the architecture passes four parameters in registers",
                        ),
                    });
                }
                self.emit_move(rd, source);
            }
            MOp::FrameAddr { offset } => {
                let rd = self.dest(inst)?;
                self.emit_frame_address(rd, *offset);
            }
            MOp::SpillStore { offset, size } => {
                let rm = self.gpr(self.arg(inst, 0)?)?;
                self.emit_store_stack(rm, *offset, *size);
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
        self.use_saved(rd);
        self.use_saved(rn);
        self.use_saved(rm);
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
    fn emit_load(&mut self, rd: u8, rn: u8, size: u32) -> Result<()> {
        match size {
            4 => {
                self.emit_load_imm(rd, rn, 0, 4);
                Ok(())
            }
            1 => {
                self.emit_load_imm(rd, rn, 0, 1);
                Ok(())
            }
            2 => {
                self.emit_load_imm(rd, rn, 0, 2);
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
    fn emit_load_imm(&mut self, rd: u8, rn: u8, offset: u32, size: u32) {
        self.use_saved(rd);
        self.use_saved(rn);
        match size {
            1 => {
                if offset <= 31 {
                    encode::ldrb_imm(&mut self.bytes, rd, rn, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::ldrb_imm_wide(&mut self.bytes, rd, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset % 2 == 0 {
                    encode::ldrh_imm(&mut self.bytes, rd, rn, u8::try_from(offset / 2).unwrap_or(0));
                } else {
                    encode::ldrh_imm_wide(&mut self.bytes, rd, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 124 && offset % 4 == 0 {
                    encode::ldr_imm(&mut self.bytes, rd, rn, u8::try_from(offset / 4).unwrap_or(0));
                } else {
                    encode::ldr_imm_wide(&mut self.bytes, rd, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
        }
        self.instructions += 1;
    }

    /// Emits the store of a value.
    fn emit_store(&mut self, rm: u8, rn: u8, size: u32) -> Result<()> {
        match size {
            4 => {
                self.emit_store_imm(rm, rn, 0, 4);
                Ok(())
            }
            1 => {
                self.emit_store_imm(rm, rn, 0, 1);
                Ok(())
            }
            2 => {
                self.emit_store_imm(rm, rn, 0, 2);
                Ok(())
            }
            _ => Err(CodegenError::Unsupported {
                what: "a store",
                detail: String::from("the machine register holds one word"),
            }),
        }
    }

    /// Emits the store of a value at an offset.
    fn emit_store_imm(&mut self, rm: u8, rn: u8, offset: u32, size: u32) {
        self.use_saved(rm);
        self.use_saved(rn);
        match size {
            1 => {
                if offset <= 31 {
                    encode::strb_imm(&mut self.bytes, rm, rn, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::strb_imm_wide(&mut self.bytes, rm, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset % 2 == 0 {
                    encode::strh_imm(&mut self.bytes, rm, rn, u8::try_from(offset / 2).unwrap_or(0));
                } else {
                    encode::strh_imm_wide(&mut self.bytes, rm, rn, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 124 && offset % 4 == 0 {
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
        self.use_saved(rd);
        match size {
            1 => {
                if offset <= 31 {
                    encode::ldrb_imm(&mut self.bytes, rd, reg::SP, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::ldrb_imm_wide(&mut self.bytes, rd, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset % 2 == 0 {
                    encode::ldrh_imm(&mut self.bytes, rd, reg::SP, u8::try_from(offset / 2).unwrap_or(0));
                } else {
                    encode::ldrh_imm_wide(&mut self.bytes, rd, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 1020 && offset % 4 == 0 {
                    encode::ldr_sp(&mut self.bytes, rd, u8::try_from(offset / 4).unwrap_or(0));
                } else {
                    encode::ldr_sp_wide(&mut self.bytes, rd, u16::try_from(offset).unwrap_or(0));
                }
            }
        }
        self.instructions += 1;
    }

    /// Emits the store of a value into the frame.
    fn emit_store_stack(&mut self, rm: u8, offset: u32, size: u32) {
        self.use_saved(rm);
        match size {
            1 => {
                if offset <= 31 {
                    encode::strb_imm(&mut self.bytes, rm, reg::SP, u8::try_from(offset).unwrap_or(0));
                } else {
                    encode::strb_imm_wide(&mut self.bytes, rm, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            2 => {
                if offset <= 62 && offset % 2 == 0 {
                    encode::strh_imm(&mut self.bytes, rm, reg::SP, u8::try_from(offset / 2).unwrap_or(0));
                } else {
                    encode::strh_imm_wide(&mut self.bytes, rm, reg::SP, u16::try_from(offset).unwrap_or(0));
                }
            }
            _ => {
                if offset <= 1020 && offset % 4 == 0 {
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
        self.use_saved(rd);
        if offset <= 1020 {
            encode::add_sp_imm(&mut self.bytes, rd, u16::try_from(offset).unwrap_or(0));
        } else {
            self.emit_mov_imm(SCRATCH, offset);
            encode::adds_reg_wide(&mut self.bytes, rd, reg::SP, SCRATCH);
            self.instructions += 1;
        }
        self.instructions += 1;
    }

    /// Emits the computation of an address from a base and indices.
    fn emit_gep(
        &mut self,
        inst: &MInst,
        rd: u8,
        base: u8,
        size: u32,
        signed: bool,
    ) -> Result<()> {
        let count = inst.args.len();
        if count == 0 {
            self.emit_move(rd, base);
            return Ok(());
        }
        let shift = match size {
            1 => Some(0),
            2 => Some(1),
            4 => Some(2),
            8 => Some(3),
            _ => None,
        };
        self.emit_move(rd, base);
        for position in 1..count {
            let operand = self.arg(inst, position)?;
            let mut index = self.gpr(operand)?;
            let ty = self.operand_type(operand);
            if signed && matches!(ty, MType::I8 | MType::I16) {
                // The index is signed, so it has to be widened before
                // it is scaled.
                self.emit_move(SCRATCH, index);
                self.emit_extend(SCRATCH, SCRATCH, true, ty == MType::I16);
                index = SCRATCH;
            }
            match shift {
                Some(shift) => {
                    if shift == 0 {
                        self.emit_add(rd, rd, index);
                    } else {
                        self.emit_add_shifted(rd, rd, index, shift);
                    }
                }
                None => {
                    // The element size is not a power of two, so the
                    // index is multiplied by it.
                    self.emit_mov_imm(SCRATCH, size);
                    encode::muls_reg_wide(&mut self.bytes, SCRATCH, SCRATCH, index);
                    self.use_saved(SCRATCH);
                    self.instructions += 1;
                    self.emit_add(rd, rd, SCRATCH);
                }
            }
        }
        Ok(())
    }

    /// Emits the comparison of two registers into a one-bit value.
    fn emit_compare(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        pred: IPred,
    ) -> Result<()> {
        let cond = match pred {
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
        };
        // The value is cleared before the comparison, because the
        // comparison sets the flags the conditional block reads.
        self.emit_mov_imm(rd, 0);
        self.emit_cmp(rn, rm);
        encode::it(&mut self.bytes, cond, encode::IT_ONE);
        self.instructions += 1;
        self.emit_mov_imm_in_conditional(rd, 1);
        Ok(())
    }

    /// Emits the move of an immediate inside a conditional block.
    ///
    /// The move of an immediate in the short form sets the flags, so
    /// the wide form, which does not, is the one the conditional
    /// block uses.
    fn emit_mov_imm_in_conditional(&mut self, rd: u8, value: u32) {
        self.use_saved(rd);
        encode::movw(&mut self.bytes, rd, u16::try_from(value).unwrap_or(0));
        self.instructions += 1;
    }

    /// Emits the selection between two values.
    fn emit_select(&mut self, rd: u8, cond: u8, yes: u8, no: u8) {
        let test = if rd == cond { SCRATCH } else { cond };
        if test != cond {
            self.emit_move(test, cond);
        }
        self.emit_cmp_imm(test, 0);
        encode::it(&mut self.bytes, encode::COND_NE, encode::IT_THEN_ELSE);
        self.instructions += 1;
        self.emit_move(rd, yes);
        self.emit_move(rd, no);
    }

    /// Emits the division of two registers.
    fn emit_division(
        &mut self,
        inst: &MInst,
        rd: u8,
        rn: u8,
        rm: u8,
    ) -> Result<()> {
        let signed = matches!(inst.op, MOp::SDiv | MOp::SRem);
        let remainder = matches!(inst.op, MOp::URem | MOp::SRem);
        // The dividend goes into the first register and the divisor
        // into the second one, because the division loop works on
        // them.  The dividend travels through the scratch register, so
        // that the two moves cannot overwrite each other.
        self.emit_move(SCRATCH, rn);
        self.emit_move(1, rm);
        self.emit_move(0, SCRATCH);
        if signed {
            self.emit_signed_division();
        } else {
            self.emit_unsigned_division();
        }
        let value = if remainder { 4 } else { 0 };
        self.emit_move(rd, value);
        Ok(())
    }

    /// Emits the signed division of the first two registers.
    fn emit_signed_division(&mut self) {
        // The signs of the operands, the magnitudes and the sign of
        // the quotient are kept in the saved registers.
        encode::push(&mut self.bytes, 0xfc, false);
        self.instructions += 1;
        encode::asrs_imm(&mut self.bytes, 5, 0, 31);
        encode::asrs_imm(&mut self.bytes, 6, 1, 31);
        self.instructions += 2;
        // The magnitudes of the two operands.
        encode::eors_reg_wide(&mut self.bytes, 2, 0, 5);
        encode::subs_reg3(&mut self.bytes, 2, 2, 5);
        encode::eors_reg_wide(&mut self.bytes, 3, 1, 6);
        encode::subs_reg3(&mut self.bytes, 3, 3, 6);
        // The sign of the quotient is the sign of the operands.
        encode::eors_reg_wide(&mut self.bytes, 7, 5, 6);
        self.instructions += 5;
        // The unsigned division of the magnitudes.
        encode::mov_reg(&mut self.bytes, 1, 3);
        encode::mov_reg(&mut self.bytes, 0, 2);
        self.instructions += 2;
        self.emit_unsigned_division_body();
        // The quotient takes the sign of the operands, the remainder
        // the sign of the dividend.
        encode::eors_reg_wide(&mut self.bytes, 0, 0, 7);
        encode::subs_reg3(&mut self.bytes, 0, 0, 7);
        encode::eors_reg_wide(&mut self.bytes, 4, 4, 5);
        encode::subs_reg3(&mut self.bytes, 4, 4, 5);
        self.instructions += 4;
        encode::pop(&mut self.bytes, 0xfc, false);
        self.instructions += 1;
    }

    /// Emits the body of the unsigned division.
    ///
    /// The first register holds the dividend, the second the divisor;
    /// the quotient ends up in the first and the remainder in the
    /// fourth.
    fn emit_unsigned_division_body(&mut self) {
        // The counter, the shifted dividend and the remainder.
        encode::mov_reg(&mut self.bytes, 3, 0);
        encode::movs_imm(&mut self.bytes, 2, 32);
        encode::movs_imm(&mut self.bytes, 0, 0);
        encode::movs_imm(&mut self.bytes, 4, 0);
        self.instructions += 4;
        let top = self.bytes.len();
        // The dividend is shifted into the remainder, and the bit
        // that falls out of the shift is added to it.
        encode::adds_reg3(&mut self.bytes, 3, 3, 3);
        encode::adcs_reg(&mut self.bytes, 4, 4);
        encode::lsls_imm(&mut self.bytes, 0, 0, 1);
        encode::cmp_reg(&mut self.bytes, 4, 1);
        self.instructions += 4;
        // The remainder is at least the divisor: take one back.
        let skip_at = self.emit_placeholder();
        encode::subs_reg3(&mut self.bytes, 4, 4, 1);
        encode::adds_imm8(&mut self.bytes, 0, 1);
        self.instructions += 2;
        // The counter of the loop.
        let end = self.bytes.len();
        encode::subs_imm8(&mut self.bytes, 2, 1);
        self.instructions += 1;
        let branch_at = self.bytes.len();
        // The loop branches back to its head, and the branch over the
        // subtraction goes to the counter.
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

/// Rounds the size of the frame up so that the stack pointer keeps
/// its alignment.
///
/// The frame is a whole number of words, and the saved registers and
/// the frame together are a multiple of eight bytes.
fn align_frame(size: u32, pushed: u32) -> u32 {
    let mut frame = size;
    while frame % 4 != 0 {
        frame += 4;
    }
    while (frame + 4 * pushed) % 8 != 0 {
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
    let alloc = allocate(function)?;
    let (saved, uses_link) = scan_registers(function, &alloc);
    let mut expander = Expander::new(function, &alloc, callees, has_fp, saved, uses_link);
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
/// and then reserves the frame with one subtraction.  The frame base
/// is the stack pointer right after that subtraction, so every slot
/// of the frame is at a known distance from the stack pointer.
fn build_prologue(expander: &Expander<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut saved = expander.saved;
    if expander.uses_link {
        saved |= 1 << 14;
    }
    if saved != 0 {
        // The mask of the encoder counts the low registers, and the
        // saved registers start above them.
        encode::push(&mut out, saved & 0x0fff, expander.uses_link);
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
/// that far.
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
