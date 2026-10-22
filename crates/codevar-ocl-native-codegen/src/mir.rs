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

//! The machine intermediate representation.
//!
//! The machine IR is the contract between the target-independent half of
//! the pipeline (optimization, selection, register allocation) and the
//! architecture back ends that turn it into bytes.  It is deliberately
//! close to the instruction set every target has, and deliberately
//! ignorant of every target's encoding:
//!
//! * **Values are virtual registers.**  [`VReg`] numbers a value the
//!   selection stage produced; [`PReg`] numbers a register of the
//!   target's file.  The allocator rewrites one into the other, so the
//!   back ends never see a virtual register.
//! * **Operations are semantic.**  [`MOp::Add`] is an addition, not
//!   `add eax, ebx`; the back end picks the encoding, the flags, and
//!   the width.
//! * **Layout is explicit.**  Every function carries the size of its
//!   stack frame and the offset of every slot inside it, so the back
//!   end never has to invent a frame.
//! * **The kernel ABI is part of the model.**  Reading a work-item
//!   field ([`MOp::WorkItemLoad`]) and reading a kernel argument
//!   ([`MOp::ArgLoad`]) are ordinary operations, which is what lets the
//!   back end implement them with plain loads.
//!
//! # Register classes
//!
//! A value's register class follows from its type: integers and
//! pointers live in integer registers, floating-point scalars in
//! floating-point registers, and vectors in vector registers.  The
//! allocator asks the target's [`RegModel`](crate::target::RegModel)
//! for the register list of each class, so a target that keeps its
//! vectors in the same file as its floats simply lists the same
//! registers twice.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_native_codegen::mir::{MInst, MOp, MirBlock, MirFunction, MType, Operand, Operands, VReg};
//!
//! let sum = VReg::new(0);
//! let mut function = MirFunction::new("scale");
//! let mut block = MirBlock::new("entry");
//! block.insts.push(MInst::def(
//!     sum,
//!     MType::I32,
//!     MOp::Add,
//!     Operands::two(Operand::Immediate(1), Operand::Immediate(2)),
//! ));
//! function.push_block(block);
//! assert_eq!(function.blocks.len(), 1);
//! ```

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_abi::WorkItemField;

/// The type of a machine value.
///
/// The set is deliberately small: the widest scalar of each family,
/// plus pointers and vectors.  Narrower integers are represented by
/// the wider type with an explicit conversion, which keeps the back
/// ends' job simple and matches how the hardware actually works.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MType {
    /// The empty type; only functions without a result have it.
    Void,
    /// A one-bit boolean.
    I1,
    /// An eight-bit integer.
    I8,
    /// A sixteen-bit integer.
    I16,
    /// A thirty-two-bit integer.
    I32,
    /// A sixty-four-bit integer.
    I64,
    /// A sixteen-bit floating-point value.
    F16,
    /// A thirty-two-bit floating-point value.
    F32,
    /// A sixty-four-bit floating-point value.
    F64,
    /// A pointer to memory.  `size` is the size of the pointee in
    /// bytes; zero means the pointee's size is unknown.
    Ptr {
        /// Size of the pointee in bytes.
        size: u32,
    },
    /// A vector of `lanes` elements of type `elem`.
    Vec {
        /// Number of lanes.
        lanes: u32,
        /// Element type.
        elem: Box<MType>,
    },
}

impl MType {
    /// The size of the type in bytes, or `None` for the void type and
    /// for pointers whose pointee size is unknown.
    #[must_use]
    pub fn size_in_bytes(&self) -> Option<u32> {
        match self {
            Self::Void => None,
            Self::I1 => Some(1),
            Self::I8 => Some(1),
            Self::I16 => Some(2),
            Self::I32 | Self::F32 => Some(4),
            Self::I64 | Self::F64 => Some(8),
            Self::F16 => Some(2),
            Self::Ptr { size } => {
                if *size == 0 {
                    Some(0)
                } else {
                    Some(*size)
                }
            }
            Self::Vec { lanes, elem } => elem.size_in_bytes().map(|size| size * lanes),
        }
    }

    /// True when the type is an integer or a pointer.
    #[must_use]
    pub const fn is_integral(&self) -> bool {
        matches!(
            self,
            Self::I1 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Ptr { .. }
        )
    }

    /// True when the type is a floating-point value.
    #[must_use]
    pub const fn is_floating(&self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// True when the type is a pointer.
    #[must_use]
    pub const fn is_pointer(&self) -> bool {
        matches!(self, Self::Ptr { .. })
    }

    /// True when the type is a vector.
    #[must_use]
    pub const fn is_vector(&self) -> bool {
        matches!(self, Self::Vec { .. })
    }

    /// The number of lanes of a vector type; one for a scalar.
    #[must_use]
    pub const fn lane_count(&self) -> u32 {
        match self {
            Self::Vec { lanes, .. } => *lanes,
            _ => 1,
        }
    }

    /// The element type of a vector type; the type itself for a
    /// scalar.
    #[must_use]
    pub fn scalar(&self) -> &MType {
        match self {
            Self::Vec { elem, .. } => elem,
            other => other,
        }
    }

    /// The register class a value of this type needs.
    #[must_use]
    pub const fn reg_class(&self) -> RegClass {
        match self {
            Self::Vec { .. } => RegClass::Vector,
            Self::F16 | Self::F32 | Self::F64 => RegClass::Float,
            _ => RegClass::Integer,
        }
    }
}

/// The class of register a value occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// Integer and pointer registers.
    Integer,
    /// Floating-point registers.
    Float,
    /// Vector registers.
    Vector,
}

/// A virtual register: a value the selection stage produced.
///
/// Virtual registers are dense: the index of a register is its slot in
/// the function's type table, so a back end can look the type up in
/// constant time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(u32);

impl VReg {
    /// Creates a virtual register with the given index.
    #[must_use]
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The index of the register.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A physical register: a number in the target's register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u16);

impl PReg {
    /// Creates a physical register with the given number.
    #[must_use]
    #[inline]
    pub const fn new(number: u16) -> Self {
        Self(number)
    }

    /// The number of the register.
    #[must_use]
    #[inline]
    pub const fn number(self) -> u16 {
        self.0
    }
}

/// An index into a module's symbol table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymId(u32);

impl SymId {
    /// Creates a symbol handle with the given index.
    #[must_use]
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The index of the symbol.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// An operand of a machine instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    /// A virtual register.
    Virtual(VReg),
    /// A physical register.
    Physical(PReg),
    /// An integer immediate, sign-extended to sixty-four bits.
    Immediate(i64),
    /// A floating-point immediate, as its raw bit pattern.
    FloatImmediate(u64),
    /// A reference to a module symbol.
    Symbol(SymId),
}

impl Operand {
    /// The virtual register the operand names, when it names one.
    #[must_use]
    #[inline]
    pub const fn virtual_reg(self) -> Option<VReg> {
        match self {
            Self::Virtual(reg) => Some(reg),
            _ => None,
        }
    }

    /// The physical register the operand names, when it names one.
    #[must_use]
    #[inline]
    pub const fn physical_reg(self) -> Option<PReg> {
        match self {
            Self::Physical(reg) => Some(reg),
            _ => None,
        }
    }

    /// The integer the operand carries, when it carries one.
    #[must_use]
    #[inline]
    pub const fn immediate(self) -> Option<i64> {
        match self {
            Self::Immediate(value) => Some(value),
            _ => None,
        }
    }
}

/// The operands of a machine instruction.
///
/// The common shapes are stored inline, so an instruction costs no
/// allocation; only a call with many arguments reaches the heap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operands {
    /// No operands.
    None,
    /// One operand.
    One([Operand; 1]),
    /// Two operands.
    Two([Operand; 2]),
    /// Three operands.
    Three([Operand; 3]),
    /// More than three operands.
    Many(Vec<Operand>),
}

impl Operands {
    /// Builds an empty operand list.
    #[must_use]
    pub const fn none() -> Self {
        Self::None
    }

    /// Builds a one-operand list.
    #[must_use]
    pub const fn one(operand: Operand) -> Self {
        Self::One([operand])
    }

    /// Builds a two-operand list.
    #[must_use]
    pub const fn two(first: Operand, second: Operand) -> Self {
        Self::Two([first, second])
    }

    /// Builds a three-operand list.
    #[must_use]
    pub const fn three(first: Operand, second: Operand, third: Operand) -> Self {
        Self::Three([first, second, third])
    }

    /// Builds an operand list of any length.
    #[must_use]
    pub fn many(operands: Vec<Operand>) -> Self {
        match operands.len() {
            0 => Self::None,
            1 => Self::One([operands[0]]),
            2 => Self::Two([operands[0], operands[1]]),
            3 => Self::Three([operands[0], operands[1], operands[2]]),
            _ => Self::Many(operands),
        }
    }

    /// The operands as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[Operand] {
        match self {
            Self::None => &[],
            Self::One(operands) => operands,
            Self::Two(operands) => operands,
            Self::Three(operands) => operands,
            Self::Many(operands) => operands,
        }
    }

    /// The number of operands.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(operands) => operands.len(),
            Self::Two(operands) => operands.len(),
            Self::Three(operands) => operands.len(),
            Self::Many(operands) => operands.len(),
        }
    }

    /// Whether the list holds no operand.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The operand at `index`, when it exists.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Operand> {
        self.as_slice().get(index).copied()
    }
}

/// The predicate of an integer comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IPred {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater or equal.
    Uge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less or equal.
    Ule,
    /// Signed greater than.
    Sgt,
    /// Signed greater or equal.
    Sge,
    /// Signed less than.
    Slt,
    /// Signed less or equal.
    Sle,
}

/// The predicate of a floating-point comparison.
///
/// The `O` prefix marks an ordered comparison, which is false when
/// either operand is a NaN; the `U` prefix marks an unordered one,
/// which is true when either operand is a NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FPred {
    /// Ordered equality.
    Oeq,
    /// Ordered inequality.
    One,
    /// Ordered less than.
    Olt,
    /// Ordered less or equal.
    Ole,
    /// Ordered greater than.
    Ogt,
    /// Ordered greater or equal.
    Oge,
    /// Unordered equality.
    Ueq,
    /// Unordered inequality.
    Une,
    /// Always false; the ordered-less form of a NaN test.
    False,
}

/// The kind of a kernel argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgKind {
    /// A buffer argument: the slot holds a native pointer.
    Pointer,
    /// A scalar argument: the slot holds the value itself.
    Scalar {
        /// Size of the value in bytes.
        size: u32,
    },
}

/// A machine operation.
///
/// The variants name what the operation computes; how it is encoded is
/// the back end's business.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MOp {
    /// `dest = src`.
    Copy,
    /// `dest = load(src)` of `size` bytes.
    Load {
        /// Size of the access in bytes.
        size: u32,
    },
    /// `store(dst, src)` of `size` bytes.
    Store {
        /// Size of the access in bytes.
        size: u32,
    },
    /// `dest = src + sum(index[i] * size)`.
    ///
    /// `size` is the size of the element in bytes; `signed` selects
    /// whether the indices are sign-extended.  The selection stage
    /// walks an access chain one level at a time and therefore emits
    /// one operation per level, each with a single index; several
    /// indices are accepted for the case where the elements have the
    /// same size.
    Gep {
        /// Size of the element in bytes.
        size: u32,
        /// Whether the indices are signed.
        signed: bool,
    },
    /// `dest = broadcast(src)` over the lanes of `dest`.
    VecSplat,
    /// `dest = src[index]`.
    VecExtract {
        /// The lane to read.
        index: u32,
    },
    /// `dest = src` with lane `index` replaced by the scalar operand.
    ///
    /// The operands are the source vector and the scalar to write.
    VecInsert {
        /// The lane to write.
        index: u32,
    },
    /// `dest = shuffle(src, src, mask)`.
    VecShuffle {
        /// The lane map; each entry selects a lane of the source.
        mask: Vec<u32>,
    },
    /// Integer addition.
    Add,
    /// Integer subtraction.
    Sub,
    /// Integer multiplication.
    Mul,
    /// Unsigned integer division.
    UDiv,
    /// Signed integer division.
    SDiv,
    /// Unsigned integer remainder.
    URem,
    /// Signed integer remainder.
    SRem,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Left shift.
    Shl,
    /// Logical right shift.
    LShr,
    /// Arithmetic right shift.
    AShr,
    /// Integer negation.
    Neg,
    /// Bitwise complement.
    Not,
    /// Floating-point addition.
    FAdd,
    /// Floating-point subtraction.
    FSub,
    /// Floating-point multiplication.
    FMul,
    /// Floating-point division.
    FDiv,
    /// Floating-point remainder.
    ///
    /// The remainder is the one of a division that is truncated
    /// towards zero, and it has the sign of the dividend.
    FRem,
    /// Floating-point negation.
    FNeg,
    /// Floating-point square root.
    FSqrt,
    /// Floating-point absolute value.
    FAbs,
    /// The smaller of two floating-point values.
    FMin,
    /// The larger of two floating-point values.
    FMax,
    /// The positive difference of two floating-point values.
    ///
    /// The difference of the two values, or zero when the first is
    /// not the larger one.
    FDim,
    /// The magnitude of one floating-point value with the sign of
    /// another.
    FCopySign,
    /// Integer comparison; the result is one bit wide.
    ICmp {
        /// The predicate.
        pred: IPred,
    },
    /// Floating-point comparison; the result is one bit wide.
    FCmp {
        /// The predicate.
        pred: FPred,
    },
    /// `dest = cond ? a : b`.
    Select,
    /// Zero extension.
    ZExt,
    /// Sign extension.
    SExt,
    /// Truncation to a narrower type.
    Trunc,
    /// Floating-point to unsigned integer.
    FPToUI,
    /// Floating-point to signed integer.
    FPToSI,
    /// Unsigned integer to floating point.
    UIToFP,
    /// Signed integer to floating point.
    SIToFP,
    /// Floating-point truncation to a narrower type.
    FPTrunc,
    /// Floating-point extension to a wider type.
    FPExt,
    /// Bit-level reinterpretation.
    Bitcast,
    /// Unconditional branch to a block.
    Br {
        /// The target block.
        target: u32,
    },
    /// Conditional branch.
    ///
    /// The operands are, in order: the condition, the block taken when
    /// the condition holds, and the block taken when it does not.  The
    /// two blocks are the numbers of the blocks in the function's
    /// block list, carried as immediates — they name a place in the
    /// code, not a value.  Use [`MInst::cond_br_targets`] to read
    /// them.
    CondBr,
    /// Return the first operand.
    Ret,
    /// Return nothing.
    RetVoid,
    /// Control flow never reaches this point.
    Unreachable,
    /// Call a function of this module; the callee is a symbol.
    Call {
        /// The symbol of the callee.
        callee: SymId,
    },
    /// Read a work-item field; the operand is the work-item pointer.
    WorkItemLoad {
        /// The field to read.
        field: WorkItemField,
        /// The component to read.
        lane: u32,
    },
    /// The base of the argument block the work item points at; the
    /// operand is the work-item pointer.
    ArgBlockBase,
    /// Read a kernel argument from the argument block; the operand is
    /// the argument block's base pointer.
    ArgLoad {
        /// The argument's index.
        index: u32,
        /// The kind of argument.
        kind: ArgKind,
    },
    /// The value of the function's `index`-th parameter; the
    /// instruction takes no operand.
    ///
    /// The parameter arrives in the architecture's argument register,
    /// which the back end reads.  The allocator gives the parameter a
    /// register of its own, so a back end emits a move from the
    /// argument register to the assigned register.
    Param {
        /// The parameter's position.
        index: u32,
    },
    /// The address of a stack slot, relative to the frame base.
    FrameAddr {
        /// The slot's offset from the frame base.
        offset: u32,
    },
    /// Store an operand into a stack slot, relative to the frame base.
    SpillStore {
        /// The slot's offset from the frame base.
        offset: u32,
        /// Size of the stored value in bytes.
        size: u32,
    },
    /// Load from a stack slot, relative to the frame base.
    SpillLoad {
        /// The slot's offset from the frame base.
        offset: u32,
        /// Size of the loaded value in bytes.
        size: u32,
    },
    /// Reserve the stack frame.
    FrameSetup,
    /// Release the stack frame.
    FrameTeardown,
}

/// Where an instruction writes its result.
///
/// A value is virtual while the machine code is being built and
/// physical once the allocator has given it a home, so the two are
/// the same field with two shapes: the selection stage writes
/// [`Destination::Virtual`], the allocator rewrites it to
/// [`Destination::Physical`], and a back end only ever sees the
/// latter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Destination {
    /// A virtual register, before the allocation.
    Virtual(VReg),
    /// A physical register, after the allocation.
    Physical(PReg),
}

impl Destination {
    /// The virtual register the destination names, when it names one.
    #[must_use]
    #[inline]
    pub const fn virtual_reg(self) -> Option<VReg> {
        match self {
            Self::Virtual(reg) => Some(reg),
            Self::Physical(_) => None,
        }
    }

    /// The physical register the destination names, when it names one.
    #[must_use]
    #[inline]
    pub const fn physical_reg(self) -> Option<PReg> {
        match self {
            Self::Physical(reg) => Some(reg),
            Self::Virtual(_) => None,
        }
    }
}

/// One machine instruction.
///
/// The instruction carries the type of the value it writes
/// ([`MInst::ty`]) and the type of the values it reads
/// ([`MInst::operand_ty`]).  The second type is what tells a back end
/// how wide a comparison is, how precise a floating-point operation
/// is, how large a load or a store is, and how large the elements of a
/// vector are; without it a back end would have to guess the width of
/// an operand from the register that happens to hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MInst {
    /// The register the instruction writes, when it writes one.
    pub dest: Option<Destination>,
    /// The type of the written value.
    pub ty: MType,
    /// The type of the values the instruction reads.
    ///
    /// The meaning depends on the operation:
    ///
    /// * A comparison reads two values of the type of the compared
    ///   values, which is the width the comparison runs at.
    /// * A load reads the value the pointer points at, so the type is
    ///   the pointee type.
    /// * A store writes the value its second operand holds, so the
    ///   type is the type of the stored value.
    /// * A conversion reads a value of the source type, which is the
    ///   width the conversion reads its operand at.
    /// * A vector operation reads or writes a vector of this type; the
    ///   type of a lane — the element type — is the element type of
    ///   the vector.
    /// * An address computation reads an index of this type, which
    ///   says how wide the index is.
    ///
    /// `None` means that the operation reads no value, or that the
    /// type of the values it reads is of no interest: a branch, a
    /// return of nothing, a frame operation.
    pub operand_ty: Option<MType>,
    /// The operation.
    pub op: MOp,
    /// The operands.
    pub args: Operands,
}

impl MInst {
    /// Builds an instruction that writes the virtual register `dest`.
    ///
    /// The type of the values the instruction reads is left unknown;
    /// [`MInst::with_operand_type`] names it.
    #[must_use]
    pub fn def(dest: VReg, ty: MType, op: MOp, args: Operands) -> Self {
        Self {
            dest: Some(Destination::Virtual(dest)),
            ty,
            operand_ty: None,
            op,
            args,
        }
    }

    /// Builds an instruction that writes the physical register `dest`.
    ///
    /// The type of the values the instruction reads is left unknown;
    /// [`MInst::with_operand_type`] names it.
    #[must_use]
    pub fn def_physical(dest: PReg, ty: MType, op: MOp, args: Operands) -> Self {
        Self {
            dest: Some(Destination::Physical(dest)),
            ty,
            operand_ty: None,
            op,
            args,
        }
    }

    /// Builds an instruction that writes no register.
    ///
    /// The type of the values the instruction reads is left unknown;
    /// [`MInst::with_operand_type`] names it.
    #[must_use]
    pub fn none(ty: MType, op: MOp, args: Operands) -> Self {
        Self {
            dest: None,
            ty,
            operand_ty: None,
            op,
            args,
        }
    }

    /// Names the type of the values the instruction reads.
    ///
    /// The instruction is returned unchanged apart from the type, so
    /// the call reads as one expression:
    ///
    /// ```
    /// use codevar_ocl_native_codegen::mir::{MInst, MOp, MType, Operand, Operands, VReg};
    ///
    /// let instruction = MInst::def(
    ///     VReg::new(0),
    ///     MType::I1,
    ///     MOp::ICmp { pred: codevar_ocl_native_codegen::mir::IPred::Eq },
    ///     Operands::two(Operand::Immediate(1), Operand::Immediate(2)),
    /// )
    /// .with_operand_type(MType::I64);
    /// assert_eq!(instruction.operand_ty, Some(MType::I64));
    /// ```
    #[must_use]
    pub fn with_operand_type(mut self, operand_ty: MType) -> Self {
        self.operand_ty = Some(operand_ty);
        self
    }

    /// The type of the values the instruction reads, when it is known.
    #[must_use]
    #[inline]
    pub fn operand_type(&self) -> Option<&MType> {
        self.operand_ty.as_ref()
    }

    /// The register the instruction writes, when it writes one that
    /// has already been given a home.
    #[must_use]
    #[inline]
    pub fn dest_reg(&self) -> Option<PReg> {
        self.dest.and_then(Destination::physical_reg)
    }

    /// The virtual register the instruction writes, when it writes one
    /// that has not been given a home yet.
    #[must_use]
    #[inline]
    pub fn dest_vreg(&self) -> Option<VReg> {
        self.dest.and_then(Destination::virtual_reg)
    }

    /// The first operand, when the instruction has one.
    #[must_use]
    #[inline]
    pub fn arg0(&self) -> Option<Operand> {
        self.args.get(0)
    }

    /// The second operand, when the instruction has one.
    #[must_use]
    #[inline]
    pub fn arg1(&self) -> Option<Operand> {
        self.args.get(1)
    }

    /// The third operand, when the instruction has one.
    #[must_use]
    #[inline]
    pub fn arg2(&self) -> Option<Operand> {
        self.args.get(2)
    }

    /// The two blocks a conditional branch names.
    ///
    /// Returns `None` for every instruction that is not a conditional
    /// branch, and for a conditional branch whose block operands are
    /// not immediates.
    #[must_use]
    pub fn cond_br_targets(&self) -> Option<(u32, u32)> {
        if !matches!(self.op, MOp::CondBr) {
            return None;
        }
        let then = self.arg1()?.immediate()?;
        let otherwise = self.arg2()?.immediate()?;
        Some((then as u32, otherwise as u32))
    }
}

/// One basic block of machine code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirBlock {
    /// The block's label.
    pub label: String,
    /// The instructions, in order.  The last one is the terminator.
    pub insts: Vec<MInst>,
}

impl MirBlock {
    /// Creates an empty block with the given label.
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self {
            label: String::from(label),
            insts: Vec::new(),
        }
    }

    /// The terminator of the block, when it has one.
    #[must_use]
    pub fn terminator(&self) -> Option<&MInst> {
        self.insts.last()
    }
}

/// What a symbol names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolKind {
    /// A function of the module.
    Function {
        /// The function's index.
        index: usize,
    },
    /// A module-scope object.
    Object {
        /// The object's index.
        index: usize,
    },
}

/// A named entity of a machine module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// The symbol's name.
    pub name: String,
    /// What the symbol names.
    pub kind: SymbolKind,
    /// Whether the symbol is visible outside the module.
    pub external: bool,
}

/// A module-scope object with a fixed address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirGlobal {
    /// The object's name.
    pub name: String,
    /// Size of the object in bytes.
    pub size: u32,
    /// Alignment of the object in bytes.
    pub align: u32,
}

/// One function of a machine module.
///
/// # The parameters
///
/// A parameter is not a register of its own: the function reads it
/// with an [`MOp::Param`] instruction in its entry block, and the
/// allocator gives the result of that instruction a register like any
/// other value.  A back end therefore reads the parameter from the
/// architecture's argument register and moves it to the assigned
/// register, and the argument registers themselves are never handed
/// out to the allocator.
///
/// # The frame
///
/// The frame holds the function's own slots — the stack slots the
/// front end created — and the slots the allocator spilled into.  The
/// selection stage reserves the first part of the frame and records
/// its size in [`MirFunction::fixed_frame_size`]; the allocator places
/// its spill slots above it and sets [`MirFunction::frame_size`] to
/// the total.  Every slot is addressed by its byte offset from the
/// frame base, which is the value of the stack pointer right after the
/// frame has been reserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirFunction {
    /// The function's name.
    pub name: String,
    /// The symbol the function is known by.
    pub symbol: SymId,
    /// Whether the function is the module's entry point.
    pub is_entry: bool,
    /// The blocks of the function; the first one is the entry block.
    pub blocks: Vec<MirBlock>,
    /// The type of every virtual register, indexed by register number.
    pub vreg_types: Vec<MType>,
    /// The virtual registers holding the function's parameters.
    pub params: Vec<VReg>,
    /// The size of the stack frame in bytes, excluding the saved
    /// registers the target pushes around it.
    pub frame_size: u32,
    /// The part of the frame the selection stage reserved for the
    /// function's own slots; the allocator places its spill slots
    /// above it.
    pub fixed_frame_size: u32,
}

impl MirFunction {
    /// Creates an empty function with the given name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            symbol: SymId::new(0),
            is_entry: false,
            blocks: Vec::new(),
            vreg_types: Vec::new(),
            params: Vec::new(),
            frame_size: 0,
            fixed_frame_size: 0,
        }
    }

    /// Appends a block.
    pub fn push_block(&mut self, block: MirBlock) {
        self.blocks.push(block);
    }

    /// Creates a virtual register of the given type and returns it.
    pub fn new_vreg(&mut self, ty: MType) -> VReg {
        let reg = VReg::new(u32::try_from(self.vreg_types.len()).unwrap_or(u32::MAX));
        self.vreg_types.push(ty);
        reg
    }

    /// The type of a virtual register.
    #[must_use]
    pub fn vreg_type(&self, reg: VReg) -> Option<&MType> {
        self.vreg_types.get(reg.index())
    }
}

/// A whole machine module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirModule {
    /// The functions of the module, in layout order.
    pub functions: Vec<MirFunction>,
    /// The symbols of the module, in layout order.
    pub symbols: Vec<Symbol>,
    /// The module-scope objects.
    pub globals: Vec<MirGlobal>,
}

impl MirModule {
    /// Creates an empty module.
    #[must_use]
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            symbols: Vec::new(),
            globals: Vec::new(),
        }
    }

    /// Adds a symbol and returns its handle.
    pub fn add_symbol(&mut self, name: &str, kind: SymbolKind, external: bool) -> SymId {
        let id = SymId::new(u32::try_from(self.symbols.len()).unwrap_or(u32::MAX));
        self.symbols.push(Symbol {
            name: String::from(name),
            kind,
            external,
        });
        id
    }

    /// The symbol with the given handle.
    #[must_use]
    pub fn symbol(&self, id: SymId) -> Option<&Symbol> {
        self.symbols.get(id.index())
    }

    /// The function with the given handle.
    #[must_use]
    pub fn function(&self, id: SymId) -> Option<&MirFunction> {
        match self.symbol(id).map(|symbol| &symbol.kind) {
            Some(SymbolKind::Function { index }) => self.functions.get(*index),
            _ => None,
        }
    }

    /// The index of the function with the given handle.
    #[must_use]
    pub fn function_index(&self, id: SymId) -> Option<usize> {
        match self.symbol(id).map(|symbol| &symbol.kind) {
            Some(SymbolKind::Function { index }) => Some(*index),
            _ => None,
        }
    }
}

impl Default for MirModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn the_size_of_a_type_follows_its_shape() {
        assert_eq!(MType::I32.size_in_bytes(), Some(4));
        assert_eq!(MType::F64.size_in_bytes(), Some(8));
        assert_eq!(MType::Void.size_in_bytes(), None);
        assert_eq!(MType::Ptr { size: 4 }.size_in_bytes(), Some(4));
        let vector = MType::Vec {
            lanes: 4,
            elem: Box::new(MType::I32),
        };
        assert_eq!(vector.size_in_bytes(), Some(16));
        assert_eq!(vector.lane_count(), 4);
        assert_eq!(vector.scalar(), &MType::I32);
    }

    #[test]
    fn the_register_class_follows_the_type() {
        assert_eq!(MType::I32.reg_class(), RegClass::Integer);
        assert_eq!(MType::Ptr { size: 8 }.reg_class(), RegClass::Integer);
        assert_eq!(MType::F32.reg_class(), RegClass::Float);
        let vector = MType::Vec {
            lanes: 2,
            elem: Box::new(MType::F64),
        };
        assert_eq!(vector.reg_class(), RegClass::Vector);
    }

    #[test]
    fn the_operand_list_holds_its_operands() {
        let one = Operands::one(Operand::Immediate(1));
        assert_eq!(one.len(), 1);
        assert_eq!(one.get(0), Some(Operand::Immediate(1)));
        let two = Operands::two(Operand::Immediate(1), Operand::Immediate(2));
        assert_eq!(two.as_slice().len(), 2);
        let many = Operands::many(vec![
            Operand::Immediate(1),
            Operand::Immediate(2),
            Operand::Immediate(3),
            Operand::Immediate(4),
        ]);
        assert_eq!(many.len(), 4);
        assert!(Operands::none().is_empty());
    }

    #[test]
    fn a_conditional_branch_names_its_two_blocks() {
        let branch = MInst::none(
            MType::Void,
            MOp::CondBr,
            Operands::three(
                Operand::Immediate(0),
                Operand::Immediate(3),
                Operand::Immediate(7),
            ),
        );
        assert_eq!(branch.cond_br_targets(), Some((3, 7)));
        let other = MInst::none(MType::Void, MOp::RetVoid, Operands::none());
        assert_eq!(other.cond_br_targets(), None);
    }

    #[test]
    fn a_function_numbers_its_virtual_registers() {
        let mut function = MirFunction::new("scale");
        let first = function.new_vreg(MType::I32);
        let second = function.new_vreg(MType::F32);
        assert_eq!(first.index(), 0);
        assert_eq!(second.index(), 1);
        assert_eq!(function.vreg_type(first), Some(&MType::I32));
        assert_eq!(function.vreg_type(second), Some(&MType::F32));
    }

    #[test]
    fn an_instruction_names_the_type_of_its_operands() {
        let instruction = MInst::def(
            VReg::new(0),
            MType::I1,
            MOp::ICmp { pred: IPred::Eq },
            Operands::two(Operand::Immediate(1), Operand::Immediate(2)),
        );
        assert_eq!(instruction.operand_type(), None, "the type is unknown");
        let instruction = instruction.with_operand_type(MType::I64);
        assert_eq!(instruction.operand_type(), Some(&MType::I64));
        assert_eq!(instruction.ty, MType::I1, "the result keeps its own type");
    }

    #[test]
    fn a_module_finds_its_functions_by_symbol() {
        let mut module = MirModule::new();
        let mut function = MirFunction::new("kernel");
        function.symbol = module.add_symbol("kernel", SymbolKind::Function { index: 0 }, true);
        module.functions.push(function);
        let symbol = SymId::new(0);
        assert_eq!(module.function_index(symbol), Some(0));
        assert!(module.function(symbol).is_some());
    }
}
