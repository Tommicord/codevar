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

//! Machine-code decoding and basic-block construction.
//!
//! [`schedule_block`] schedules one
//! straight-line block; this module produces those blocks from bytes. It is
//! the two-pass shape disassembler-based schedulers use — the same shape as
//! QEMU's translation-block builder and LLVM's `MachineBasicBlock` formation:
//!
//! 1. **Decode.** A target [`Decoder`] turns every instruction in a byte range
//!    into a [`Decoded`]: the scheduler's [`Instruction`] facts plus the
//!    address, length and [`ControlFlow`] of the instruction.
//! 2. **Split.** [`decode_function`] marks leaders — the entry, every in-range
//!    direct target, and the instruction after every transfer — and cuts the
//!    stream at them into [`BasicBlock`]s with their successors.
//!
//! I/O paths are the case that makes the split mandatory: an event loop loads
//! a descriptor, branches back to retry, and traps into the kernel. Keeping
//! the transfers as block boundaries is what stops the list scheduler from
//! moving a load across a branch, a fence or a syscall. [`Rv32i`] is the
//! built-in decoder for the RV32I/M instruction set that
//! [`riscv_embedded`](crate::bare::riscv_embedded) models.
//!
//! # Example
//!
//! ```rust
//! use codevar_inssched::decode::{ControlFlow, Rv32i, decode_function};
//!
//! // `addi x0, x0, 0` (nop) followed by `ret` (`jalr x0, 0(x1)`).
//! let mut code = [0u8; 8];
//! code[..4].copy_from_slice(&0x0000_0013u32.to_le_bytes());
//! code[4..].copy_from_slice(&0x0000_8067u32.to_le_bytes());
//!
//! let function = decode_function(&code, 0x1000, &Rv32i::new())?;
//! let block = &function.blocks()[0];
//! assert_eq!(block.instructions().len(), 2);
//! assert_eq!(block.instructions()[1].control(), ControlFlow::Return);
//! assert!(block.successors().is_empty());
//! # Ok::<(), codevar_inssched::DecodeError>(())
//! ```

use alloc::vec::Vec;

use crate::asm::{InstFlags, Instruction, Reg, Unit};
use crate::error::{DecodeError, ScheduleError};
use crate::machine::Machine;
use crate::sched::{BlockSchedule, SchedStrategy, schedule_block};

/// Where control goes after an instruction.
///
/// The variants mirror the flow-control classes a disassembler reports (see
/// iced-x86's `FlowControl`): plain fallthrough, conditional branch, direct
/// and indirect transfers, calls, returns and traps. A block ends at every
/// variant except [`Next`](Self::Next).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlFlow {
    /// Execution continues at the next instruction.
    Next,
    /// A conditional branch that reaches `target` or falls through.
    Conditional {
        /// The absolute branch target.
        target: u64,
    },
    /// An unconditional direct transfer to `target`.
    Jump {
        /// The absolute jump target.
        target: u64,
    },
    /// A computed transfer whose target is not known statically.
    IndirectJump,
    /// A direct call that returns to the next instruction.
    Call {
        /// The absolute call target.
        target: u64,
    },
    /// A computed call that returns to the next instruction.
    IndirectCall,
    /// A return to the caller.
    Return,
    /// A trap or system call; execution resumes at the next instruction if
    /// the handler returns.
    Trap,
}

impl ControlFlow {
    /// Whether the instruction ends its basic block.
    #[inline]
    pub const fn ends_block(self) -> bool {
        !matches!(self, Self::Next)
    }

    /// Whether control may continue at the next instruction.
    ///
    /// A call and a trap both return, so their fallthrough is the return site.
    #[inline]
    pub const fn falls_through(self) -> bool {
        matches!(
            self,
            Self::Next | Self::Conditional { .. } | Self::Call { .. } | Self::IndirectCall | Self::Trap
        )
    }

    /// The statically known target of a direct transfer.
    #[inline]
    pub const fn target(self) -> Option<u64> {
        match self {
            Self::Conditional { target } | Self::Jump { target } | Self::Call { target } => Some(target),
            _ => None,
        }
    }
}

/// One decoded instruction: the scheduler facts plus where it leads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Decoded {
    address: u64,
    length: u8,
    instruction: Instruction,
    control: ControlFlow,
}

impl Decoded {
    /// Records one decoded instruction.
    #[inline]
    pub const fn new(address: u64, length: u8, instruction: Instruction, control: ControlFlow) -> Self {
        Self {
            address,
            length,
            instruction,
            control,
        }
    }

    /// The address of the first byte.
    #[inline]
    pub const fn address(self) -> u64 {
        self.address
    }

    /// The encoded length in bytes.
    #[inline]
    pub const fn len_bytes(self) -> u8 {
        self.length
    }

    /// The scheduler's view of the instruction.
    #[inline]
    pub const fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Where control goes after this instruction.
    #[inline]
    pub const fn control(self) -> ControlFlow {
        self.control
    }
}

/// A target-specific instruction decoder.
///
/// Implementations are stateless value types, so one decoder can be shared by
/// every byte range a caller analyses.
pub trait Decoder {
    /// The longest instruction the decoder can produce, in bytes.
    fn max_instruction_len(&self) -> usize;

    /// Decodes the instruction at the start of `code`, which begins at
    /// `address`.
    ///
    /// The returned length must be non-zero, no longer than `code`, and no
    /// longer than [`max_instruction_len`](Self::max_instruction_len); the
    /// splitter rejects a decoder that violates any of these.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] when the bytes do not encode an instruction the
    /// decoder supports.
    fn decode(&self, code: &[u8], address: u64) -> Result<Decoded, DecodeError>;
}

/// A maximal straight-line run of instructions.
///
/// The block includes its terminator, if it has one. `successors` holds the
/// start addresses of the blocks this one can reach; transfers outside the
/// decoded range (an external callee, a mapped-in stub) are visible through
/// the last instruction's [`ControlFlow`] but are not listed here.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BasicBlock {
    start: u64,
    end: u64,
    instructions: Vec<Decoded>,
    successors: Vec<u64>,
}

impl BasicBlock {
    /// The address of the first instruction.
    #[inline]
    pub const fn start(&self) -> u64 {
        self.start
    }

    /// One past the last byte of the block.
    #[inline]
    pub const fn end(&self) -> u64 {
        self.end
    }

    /// The decoded instructions, in program order.
    #[inline]
    pub fn instructions(&self) -> &[Decoded] {
        &self.instructions
    }

    /// The start addresses of the successor blocks.
    #[inline]
    pub fn successors(&self) -> &[u64] {
        &self.successors
    }

    /// How the block ends: the last instruction's control flow.
    #[inline]
    pub fn control(&self) -> ControlFlow {
        self.instructions
            .last()
            .map_or(ControlFlow::Next, |decoded| decoded.control())
    }
}

/// The basic blocks decoded from one contiguous byte range.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Function {
    entry: u64,
    blocks: Vec<BasicBlock>,
}

impl Function {
    /// The address the range was decoded from.
    #[inline]
    pub const fn entry(&self) -> u64 {
        self.entry
    }

    /// The blocks, in ascending address order.
    #[inline]
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    /// The block starting at `address`, if one does.
    #[inline]
    pub fn block_at(&self, address: u64) -> Option<&BasicBlock> {
        self.blocks
            .binary_search_by_key(&address, BasicBlock::start)
            .ok()
            .and_then(|index| self.blocks.get(index))
    }

    /// The number of decoded instructions over every block.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.instructions.len())
            .sum()
    }
}

/// The list schedule of every block of a [`Function`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FunctionSchedule {
    blocks: Vec<BlockSchedule>,
    makespan: u32,
}

impl FunctionSchedule {
    /// The per-block schedules, parallel to [`Function::blocks`].
    #[inline]
    pub fn blocks(&self) -> &[BlockSchedule] {
        &self.blocks
    }

    /// The sum of the per-block makespans: the cycles one pass over the
    /// function's static blocks needs.
    #[inline]
    pub const fn makespan(&self) -> u32 {
        self.makespan
    }

    /// The schedule of block `index`, if it exists.
    #[inline]
    pub fn block(&self, index: usize) -> Option<&BlockSchedule> {
        self.blocks.get(index)
    }
}

/// Decodes `code` into basic blocks.
///
/// `base` is the address of `code[0]`; every address in the returned function
/// is absolute. Leaders are the entry, every direct target that falls inside
/// the range, and the instruction after every transfer. A target outside the
/// range is an external block and does not split anything.
///
/// # Errors
///
/// Returns [`DecodeError`] when the decoder rejects an encoding, the range
/// ends inside an instruction, a decoder reports an impossible length, or an
/// in-range branch target lands in the middle of an instruction.
pub fn decode_function<D: Decoder>(code: &[u8], base: u64, decoder: &D) -> Result<Function, DecodeError> {
    let end = base.wrapping_add(code.len() as u64);
    let mut flat: Vec<Decoded> = Vec::new();
    let mut offset = 0usize;
    while offset < code.len() {
        let address = base.wrapping_add(offset as u64);
        let decoded = decoder.decode(&code[offset..], address)?;
        let length = usize::from(decoded.len_bytes());
        if length == 0 || length > code.len() - offset || length > decoder.max_instruction_len() {
            return Err(DecodeError::InvalidLength { address });
        }
        offset += length;
        flat.push(decoded);
    }

    let mut leaders: Vec<u64> = Vec::with_capacity(flat.len() + 1);
    leaders.push(base);
    for decoded in &flat {
        if decoded.control().ends_block() {
            let after = decoded
                .address()
                .wrapping_add(u64::from(decoded.len_bytes()));
            if after >= base && after < end {
                leaders.push(after);
            }
        }
        if let Some(target) = decoded.control().target()
            && target >= base
            && target < end
        {
            leaders.push(target);
        }
    }
    leaders.sort_unstable();
    leaders.dedup();

    // A leader inside the range must start an instruction. A target outside
    // it belongs to another range, so there is nothing here to validate.
    for &leader in &leaders {
        if leader < base || leader >= end {
            continue;
        }
        if flat
            .binary_search_by_key(&leader, |decoded| decoded.address())
            .is_err()
        {
            return Err(DecodeError::MisalignedTarget { target: leader });
        }
    }

    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut current: Vec<Decoded> = Vec::new();
    let mut start = base;
    for (index, decoded) in flat.iter().enumerate() {
        let is_leader = leaders.binary_search(&decoded.address()).is_ok();
        if is_leader && !current.is_empty() {
            let next = decoded.address();
            blocks.push(close_block(
                base,
                end,
                start,
                core::mem::take(&mut current),
                Some(next),
            ));
            start = next;
        }
        current.push(*decoded);
        if decoded.control().ends_block() {
            let next = flat
                .get(index + 1)
                .map(|decoded| decoded.address());
            blocks.push(close_block(base, end, start, core::mem::take(&mut current), next));
            start = next.unwrap_or_else(|| {
                decoded
                    .address()
                    .wrapping_add(u64::from(decoded.len_bytes()))
            });
        }
    }
    if !current.is_empty() {
        blocks.push(close_block(base, end, start, current, None));
    }

    Ok(Function { entry: base, blocks })
}

/// Cuts one block and records its in-range successors.
fn close_block(base: u64, end: u64, start: u64, instructions: Vec<Decoded>, next: Option<u64>) -> BasicBlock {
    let block_end = instructions.last().map_or(start, |decoded| {
        decoded
            .address()
            .wrapping_add(u64::from(decoded.len_bytes()))
    });
    let mut successors = Vec::new();
    if let Some(last) = instructions.last() {
        let control = last.control();
        if let Some(target) = control.target()
            && target >= base
            && target < end
        {
            successors.push(target);
        }
        if control.falls_through()
            && let Some(next) = next
            && Some(next) != control.target()
        {
            successors.push(next);
        }
    }
    BasicBlock {
        start,
        end: block_end,
        instructions,
        successors,
    }
}

/// Schedules every block of `function` with `strategy`.
///
/// Each block is scheduled independently, so control flow is preserved: the
/// scheduler may reorder instructions inside a block, never across one of the
/// decoders' block boundaries. The makespan is the sum over the blocks.
///
/// # Errors
///
/// Propagates [`ScheduleError`] from the block scheduler.
pub fn schedule_function<S: SchedStrategy>(
    function: &Function,
    machine: &Machine,
    strategy: &S,
) -> Result<FunctionSchedule, ScheduleError> {
    let mut blocks = Vec::with_capacity(function.blocks.len());
    let mut makespan = 0u32;
    for block in &function.blocks {
        let instructions: Vec<Instruction> = block
            .instructions
            .iter()
            .map(|decoded| *decoded.instruction())
            .collect();
        let schedule = schedule_block(&instructions, machine, strategy)?;
        makespan = makespan.saturating_add(u32::from(schedule.makespan));
        blocks.push(schedule);
    }
    Ok(FunctionSchedule { blocks, makespan })
}

/// Decoder for the RV32I base instruction set plus the M extension.
///
/// The operand shape of every instruction is mapped onto an
/// [`Instruction`]: loads read memory, stores write
/// memory, `fence` and the trap instructions are barriers, and `jalr`
/// distinguishes a return from an indirect call. Register `x0` is omitted
/// from the read and write sets, because it is hardwired to zero.
///
/// Programs that use compressed instructions (the C extension), floating
/// point, atomics or privileged control transfers are rejected rather than
/// misdecoded.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32i;

impl Rv32i {
    /// Creates the decoder.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Decoder for Rv32i {
    #[inline]
    fn max_instruction_len(&self) -> usize {
        4
    }

    fn decode(&self, code: &[u8], address: u64) -> Result<Decoded, DecodeError> {
        if address & 3 != 0 {
            return Err(DecodeError::Misaligned { address });
        }
        let bytes: [u8; 4] = code
            .get(..4)
            .and_then(|slice| <[u8; 4]>::try_from(slice).ok())
            .ok_or(DecodeError::Truncated {
                address,
                available: code.len(),
            })?;
        decode_word(u32::from_le_bytes(bytes), address)
    }
}

const OP_LOAD: u32 = 0b000_0011;
const OP_MISC_MEM: u32 = 0b000_1111;
const OP_IMM: u32 = 0b001_0011;
const OP_AUIPC: u32 = 0b001_0111;
const OP_STORE: u32 = 0b010_0011;
const OP_OP: u32 = 0b011_0011;
const OP_LUI: u32 = 0b011_0111;
const OP_BRANCH: u32 = 0b110_0011;
const OP_JALR: u32 = 0b110_0111;
const OP_JAL: u32 = 0b110_1111;
const OP_SYSTEM: u32 = 0b111_0011;

/// Decodes one little-endian RV32 word.
fn decode_word(word: u32, address: u64) -> Result<Decoded, DecodeError> {
    let opcode = word & 0x7f;
    let rd = (word >> 7) & 0x1f;
    let funct3 = (word >> 12) & 0x7;
    let rs1 = (word >> 15) & 0x1f;
    let rs2 = (word >> 20) & 0x1f;
    let funct7 = word >> 25;
    let invalid = || DecodeError::InvalidEncoding { address, word };

    match opcode {
        OP_LUI | OP_AUIPC => shape(
            address,
            word,
            Unit::Alu,
            InstFlags::NONE,
            &[],
            &[rd],
            ControlFlow::Next,
        ),
        OP_JAL => {
            let target = relative(address, imm_j(word));
            if rd == 0 {
                shape(
                    address,
                    word,
                    Unit::Branch,
                    InstFlags::BRANCH | InstFlags::TERMINATOR,
                    &[],
                    &[],
                    ControlFlow::Jump { target },
                )
            } else {
                shape(
                    address,
                    word,
                    Unit::Branch,
                    InstFlags::BRANCH | InstFlags::TERMINATOR | InstFlags::CALL,
                    &[],
                    &[rd],
                    ControlFlow::Call { target },
                )
            }
        }
        OP_JALR => {
            if funct3 != 0 {
                return Err(invalid());
            }
            let (control, flags) = if rd == 0 && rs1 == 1 && imm_i(word) == 0 {
                (ControlFlow::Return, InstFlags::BRANCH | InstFlags::TERMINATOR)
            } else if rd == 0 {
                (
                    ControlFlow::IndirectJump,
                    InstFlags::BRANCH | InstFlags::TERMINATOR,
                )
            } else {
                (
                    ControlFlow::IndirectCall,
                    InstFlags::BRANCH | InstFlags::TERMINATOR | InstFlags::CALL,
                )
            };
            shape(address, word, Unit::Branch, flags, &[rs1], &[rd], control)
        }
        OP_BRANCH => {
            if funct3 == 2 || funct3 == 3 {
                return Err(invalid());
            }
            let target = relative(address, imm_b(word));
            shape(
                address,
                word,
                Unit::Branch,
                InstFlags::BRANCH | InstFlags::TERMINATOR | InstFlags::CONDITIONAL,
                &[rs1, rs2],
                &[],
                ControlFlow::Conditional { target },
            )
        }
        OP_LOAD => {
            if !matches!(funct3, 0 | 1 | 2 | 4 | 5) {
                return Err(invalid());
            }
            shape(
                address,
                word,
                Unit::Load,
                InstFlags::READS_MEMORY,
                &[rs1],
                &[rd],
                ControlFlow::Next,
            )
        }
        OP_STORE => {
            if !matches!(funct3, 0..=2) {
                return Err(invalid());
            }
            shape(
                address,
                word,
                Unit::Store,
                InstFlags::WRITES_MEMORY | InstFlags::SIDE_EFFECT,
                &[rs1, rs2],
                &[],
                ControlFlow::Next,
            )
        }
        OP_IMM => {
            let supported = match funct3 {
                1 => funct7 == 0,
                5 => funct7 == 0 || funct7 == 0x20,
                _ => true,
            };
            if !supported {
                return Err(invalid());
            }
            shape(
                address,
                word,
                Unit::Alu,
                InstFlags::NONE,
                &[rs1],
                &[rd],
                ControlFlow::Next,
            )
        }
        OP_OP => {
            let unit = match funct7 {
                0 => Unit::Alu,
                0x20 if funct3 == 0 || funct3 == 5 => Unit::Alu,
                1 => {
                    if funct3 < 4 {
                        Unit::Mul
                    } else {
                        Unit::Div
                    }
                }
                _ => return Err(invalid()),
            };
            shape(
                address,
                word,
                unit,
                InstFlags::NONE,
                &[rs1, rs2],
                &[rd],
                ControlFlow::Next,
            )
        }
        OP_MISC_MEM => {
            if funct3 > 1 {
                return Err(invalid());
            }
            shape(
                address,
                word,
                Unit::System,
                InstFlags::BARRIER | InstFlags::SIDE_EFFECT,
                &[],
                &[],
                ControlFlow::Next,
            )
        }
        OP_SYSTEM => decode_system(word, address),
        _ => Err(invalid()),
    }
}

/// Decodes the SYSTEM opcode: traps, fences from the CSR space, and CSR
/// accesses.
fn decode_system(word: u32, address: u64) -> Result<Decoded, DecodeError> {
    let funct3 = (word >> 12) & 0x7;
    let rd = (word >> 7) & 0x1f;
    let rs1 = (word >> 15) & 0x1f;
    let immediate = word >> 20;
    let invalid = || DecodeError::InvalidEncoding { address, word };

    match funct3 {
        0 => {
            if rd != 0 || rs1 != 0 {
                return Err(invalid());
            }
            // ECALL, EBREAK and WFI are privileged traps. WFI resumes at the
            // next instruction once an interrupt arrives, so all three fall
            // through; all three also observe and change memory, so the
            // scheduler must not move traffic across them.
            if matches!(immediate, 0 | 1 | 0x105) {
                shape(
                    address,
                    word,
                    Unit::System,
                    InstFlags::BARRIER | InstFlags::SIDE_EFFECT,
                    &[],
                    &[],
                    ControlFlow::Trap,
                )
            } else {
                Err(invalid())
            }
        }
        4 => Err(invalid()),
        1..=3 => shape(
            address,
            word,
            Unit::System,
            InstFlags::SIDE_EFFECT,
            &[rs1],
            &[rd],
            ControlFlow::Next,
        ),
        _ => shape(
            address,
            word,
            Unit::System,
            InstFlags::SIDE_EFFECT,
            &[],
            &[rd],
            ControlFlow::Next,
        ),
    }
}

/// Builds the scheduler facts from RV32 register indices.
///
/// `x0` is elided from both sets: it is hardwired to zero, so reading it
/// creates no dependence and writing it has no effect.
fn shape(
    address: u64,
    word: u32,
    unit: Unit,
    flags: InstFlags,
    reads: &[u32],
    writes: &[u32],
    control: ControlFlow,
) -> Result<Decoded, DecodeError> {
    let (read_registers, read_count) = registers(reads);
    let (write_registers, write_count) = registers(writes);
    let instruction = Instruction::new(
        unit,
        flags,
        0,
        &read_registers[..read_count],
        &write_registers[..write_count],
    )
    .map_err(|_| DecodeError::InvalidEncoding { address, word })?;
    Ok(Decoded::new(address, 4, instruction, control))
}

/// Maps register indices to interned [`Reg`]s, dropping `x0`.
fn registers(indices: &[u32]) -> ([Reg; 2], usize) {
    let mut registers = [Reg::NONE; 2];
    let mut count = 0;
    for &index in indices {
        if index == 0 || count == registers.len() {
            continue;
        }
        registers[count] = Reg::new(index as u16);
        count += 1;
    }
    (registers, count)
}

/// Adds a signed instruction-relative offset to `address`.
#[inline]
fn relative(address: u64, offset: i64) -> u64 {
    address.wrapping_add(offset as u64)
}

/// Sign-extends the low `bits` bits of `value`.
#[inline]
const fn sign_extend(value: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value as i64) << shift) >> shift
}

#[inline]
const fn imm_i(word: u32) -> i64 {
    sign_extend(word >> 20, 12)
}

#[inline]
const fn imm_b(word: u32) -> i64 {
    let imm = (((word >> 31) & 1) << 12)
        | (((word >> 7) & 1) << 11)
        | (((word >> 25) & 0x3f) << 5)
        | (((word >> 8) & 0xf) << 1);
    sign_extend(imm, 13)
}

#[inline]
const fn imm_j(word: u32) -> i64 {
    let imm = (((word >> 31) & 1) << 20)
        | (((word >> 12) & 0xff) << 12)
        | (((word >> 20) & 1) << 11)
        | (((word >> 21) & 0x3ff) << 1);
    sign_extend(imm, 21)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bare;
    use crate::sched::Latency;
    use crate::task::Task;

    fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn i_type(imm: i32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
        (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn s_type(imm: i32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
        let imm = imm as u32;
        (((imm >> 5) & 0x7f) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | ((imm & 0x1f) << 7)
            | opcode
    }

    fn b_type(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
        let imm = imm as u32;
        (((imm >> 12) & 1) << 31)
            | (((imm >> 5) & 0x3f) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (((imm >> 1) & 0xf) << 8)
            | (((imm >> 11) & 1) << 7)
            | OP_BRANCH
    }

    fn j_type(imm: i32, rd: u32) -> u32 {
        let imm = imm as u32;
        (((imm >> 20) & 1) << 31)
            | (((imm >> 1) & 0x3ff) << 21)
            | (((imm >> 11) & 1) << 20)
            | (((imm >> 12) & 0xff) << 12)
            | (rd << 7)
            | OP_JAL
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn decode_one(word: u32) -> Decoded {
        Rv32i::new()
            .decode(&word.to_le_bytes(), 0x1000)
            .expect("valid encoding")
    }

    #[test]
    fn decodes_memory_and_alu_instructions() {
        // lw a0, 0(a1)
        let lw = decode_one(i_type(0, 11, 2, 10, OP_LOAD));
        assert_eq!(lw.instruction().unit(), Unit::Load);
        assert!(
            lw.instruction()
                .flags()
                .contains(InstFlags::READS_MEMORY)
        );
        assert_eq!(lw.instruction().reads(), &[Reg::new(11)]);
        assert_eq!(lw.instruction().writes(), &[Reg::new(10)]);
        assert_eq!(lw.control(), ControlFlow::Next);

        // sw a0, 0(a1)
        let sw = decode_one(s_type(0, 10, 11, 2, OP_STORE));
        assert_eq!(sw.instruction().unit(), Unit::Store);
        assert!(
            sw.instruction()
                .flags()
                .contains(InstFlags::WRITES_MEMORY)
        );
        assert!(
            sw.instruction()
                .flags()
                .contains(InstFlags::SIDE_EFFECT)
        );
        assert_eq!(sw.instruction().reads(), &[Reg::new(11), Reg::new(10)]);

        // mul a0, a1, a2 and div a0, a1, a2
        assert_eq!(
            decode_one(r_type(1, 12, 11, 0, 10, OP_OP))
                .instruction()
                .unit(),
            Unit::Mul
        );
        assert_eq!(
            decode_one(r_type(1, 12, 11, 4, 10, OP_OP))
                .instruction()
                .unit(),
            Unit::Div
        );
    }

    #[test]
    fn decodes_control_transfers() {
        // beq a0, x0, +8
        let beq = decode_one(b_type(8, 0, 10, 0));
        assert_eq!(beq.control(), ControlFlow::Conditional { target: 0x1008 });
        assert!(
            beq.instruction()
                .flags()
                .contains(InstFlags::CONDITIONAL)
        );
        assert!(beq.instruction().is_terminator());

        // jal ra, +8 is a call; jal x0, +8 is a plain jump.
        let call = decode_one(j_type(8, 1));
        assert_eq!(call.control(), ControlFlow::Call { target: 0x1008 });
        assert!(
            call.instruction()
                .flags()
                .contains(InstFlags::CALL)
        );
        let jump = decode_one(j_type(8, 0));
        assert_eq!(jump.control(), ControlFlow::Jump { target: 0x1008 });
        assert!(
            !jump
                .instruction()
                .flags()
                .contains(InstFlags::CALL)
        );

        // ret is jalr x0, 0(x1); jalr says target is not statically known.
        let ret = decode_one(i_type(0, 1, 0, 0, OP_JALR));
        assert_eq!(ret.control(), ControlFlow::Return);
        let indirect = decode_one(i_type(0, 2, 0, 1, OP_JALR));
        assert_eq!(indirect.control(), ControlFlow::IndirectCall);
    }

    #[test]
    fn decodes_barriers_and_traps() {
        // fence
        let fence = decode_one(0x0ff0_000f);
        assert_eq!(fence.instruction().unit(), Unit::System);
        assert!(
            fence
                .instruction()
                .flags()
                .contains(InstFlags::BARRIER)
        );
        assert_eq!(fence.control(), ControlFlow::Next);

        // ecall
        let ecall = decode_one(0x0000_0073);
        assert_eq!(ecall.instruction().unit(), Unit::System);
        assert!(
            ecall
                .instruction()
                .flags()
                .contains(InstFlags::BARRIER)
        );
        assert_eq!(ecall.control(), ControlFlow::Trap);

        // csrrw x0, mstatus, x1: a CSR write is a side effect, not a fence.
        let csr = decode_one(i_type(0x300, 1, 1, 0, OP_SYSTEM));
        assert!(
            csr.instruction()
                .flags()
                .contains(InstFlags::SIDE_EFFECT)
        );
        assert!(
            !csr.instruction()
                .flags()
                .contains(InstFlags::BARRIER)
        );
        assert_eq!(csr.instruction().reads(), &[Reg::new(1)]);
    }

    #[test]
    fn splits_a_polling_loop() {
        let base = 0x8000_0000u64;
        // lw a0, 0(a1); beq a0, x0, -4; ecall
        let code = bytes(&[i_type(0, 11, 2, 10, OP_LOAD), b_type(-4, 0, 10, 0), 0x73]);
        let function = decode_function(&code, base, &Rv32i::new()).expect("decodes");

        assert_eq!(function.entry(), base);
        assert_eq!(function.instruction_count(), 3);
        assert_eq!(function.blocks().len(), 2);

        let header = &function.blocks()[0];
        assert_eq!(header.start(), base);
        assert_eq!(header.end(), base + 8);
        assert_eq!(header.successors(), &[base, base + 8]);
        assert_eq!(header.control(), ControlFlow::Conditional { target: base });

        let exit = &function.blocks()[1];
        assert_eq!(exit.control(), ControlFlow::Trap);
        assert!(exit.successors().is_empty());
        assert!(function.block_at(base + 8).is_some());
    }

    #[test]
    fn keeps_call_targets_and_return_sites_apart() {
        let base = 0x1000u64;
        // jal ra, +8; addi t0, x0, 1; ret
        let code = bytes(&[
            j_type(8, 1),
            i_type(1, 0, 0, 5, OP_IMM),
            i_type(0, 1, 0, 0, OP_JALR),
        ]);
        let function = decode_function(&code, base, &Rv32i::new()).expect("decodes");

        assert_eq!(function.blocks().len(), 3);
        assert_eq!(function.blocks()[0].successors(), &[base + 8, base + 4]);
        assert_eq!(function.blocks()[2].control(), ControlFlow::Return);
    }

    #[test]
    fn schedules_an_io_path_block_by_block() {
        let base = 0x8000_0000u64;
        let code = bytes(&[i_type(0, 11, 2, 10, OP_LOAD), b_type(-4, 0, 10, 0), 0x73]);
        let function = decode_function(&code, base, &Rv32i::new()).expect("decodes");

        let schedule = schedule_function(&function, &bare::riscv_embedded(), &Latency).expect("schedules");
        assert_eq!(schedule.blocks().len(), 2);
        // The branch reads the load result, which resolves after two cycles.
        assert_eq!(schedule.blocks()[0].cycles, vec![0, 2]);
        assert_eq!(schedule.blocks()[0].makespan, 3);
        assert_eq!(schedule.blocks()[1].makespan, 1);
        assert_eq!(schedule.makespan(), 4);
        assert_eq!(
            Task::cost_from_function(&function, &bare::riscv_embedded(), &Latency),
            Ok(4)
        );
    }

    #[test]
    fn rejects_misaligned_targets() {
        let base = 0x1000u64;
        // A branch at +4 with a +2 offset lands at +6, inside the lw at +4.
        let code = bytes(&[
            i_type(0, 0, 2, 10, OP_LOAD),
            b_type(2, 0, 10, 0),
            i_type(0, 0, 0, 0, OP_IMM),
        ]);
        assert_eq!(
            decode_function(&code, base, &Rv32i::new()),
            Err(DecodeError::MisalignedTarget { target: base + 6 })
        );
    }

    #[test]
    fn rejects_truncated_misaligned_and_invalid_input() {
        assert_eq!(
            Rv32i::new().decode(&[0x13, 0x00, 0x00], 0x1000),
            Err(DecodeError::Truncated {
                address: 0x1000,
                available: 3
            })
        );
        assert_eq!(
            Rv32i::new().decode(&0u32.to_le_bytes(), 0x1002),
            Err(DecodeError::Misaligned { address: 0x1002 })
        );
        assert_eq!(
            Rv32i::new().decode(&0u32.to_le_bytes(), 0x1000),
            Err(DecodeError::InvalidEncoding {
                address: 0x1000,
                word: 0
            })
        );
    }
}
