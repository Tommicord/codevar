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

//! Execution of one work-group of a loaded SPIR-V kernel.
//!
//! The interpreter is deliberately single-threaded and allocation-light:
//! a work-group is a flat list of work-items, each with a fixed register
//! file (`slots_len` lanes), a private byte arena and a call stack. Every
//! instruction is executed against those tables; nothing is re-decoded.
//!
//! # Scheduling
//!
//! Modules without barriers or group operations run one work-item to
//! completion before the next starts ([`run_sequential`]); a module the
//! loader marked [`SpirvModule::needs_collective`] runs every work-item
//! in lock-step rounds instead ([`run_collective`]): each round takes
//! every work-item to its next rendezvous, then the round's rendezvous is
//! resolved for the whole group at once. That is what makes
//! `OpControlBarrier` and the `OpGroup*` reductions correct without
//! threads.
//!
//! # Memory
//!
//! Addresses are absolute: the loader requires a physical address space
//! whose width matches the host, so a register holding a pointer is a
//! real address. Three regions exist — the launch's argument buffers (the
//! caller's memory), the per-work-item arena and the shared group arena —
//! and an access into an arena region is range-checked against that
//! region before it is used.

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::exec::{ARG_SLOT, ArgBlock, Launch};

use super::builtin::{ImportFn, ext_inst_arity, import_fn};
use super::module::{SpirvEntry, SpirvModule};
use super::ops;
use super::types::{self, MAX_VALUE_LANES, Type};

/// Instruction steps one work-item may take before the interpreter gives
/// up with [`Error::OutOfResources`]. Bounds runaway loops in a module
/// whose back edges the loader cannot disprove.
const MAX_STEPS: u64 = 1 << 28;

/// Bytes of registers, arena and call stack one work-item may occupy.
const MAX_ITEM_BUDGET: u64 = 1 << 20;

/// Whole-work-group bytes when the module needs rendezvous scheduling.
const MAX_COLLECTIVE_BUDGET: u64 = 1 << 20;

/// Maximum nesting of a composite walk inside an access chain.
const MAX_WALK_DEPTH: u32 = 64;

/// Lane capacity of one scratch window (mirrors [`MAX_VALUE_LANES`]).
const LANES: usize = MAX_VALUE_LANES as usize;

/// Scheduling state of a work-item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Executing until the next rendezvous.
    Running,
    /// Waiting at a [`Park`] for the rest of the work-group.
    Parked,
    /// Returned from the entry point.
    Done,
}

/// Why a work-item stopped running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Park {
    /// `OpControlBarrier`: the whole group must arrive first.
    Barrier,
    /// A group operation: its opcode, result id, operand id, and an
    /// instruction-specific extra — the `GroupOperation` literal for
    /// reductions, the index value id for broadcasts, `0` otherwise.
    Group {
        /// The `OpGroup*` opcode.
        opcode: u16,
        /// Result value id.
        result: u32,
        /// Contributing value id.
        operand: u32,
        /// Instruction-specific extra (see above).
        aux: u32,
    },
}

/// A pending call frame of a work-item's call stack.
#[derive(Debug, Clone, Copy)]
struct Frame {
    /// Instruction to resume at when the callee returns.
    ret_ip: u32,
    /// Callee result id (`0` for a void call).
    result: u32,
    /// Caller's current block label id.
    label: u32,
    /// Caller's predecessor block label id (restored for `OpPhi`).
    pred: u32,
}

/// The mutable state of one work-item.
#[derive(Debug)]
struct Item {
    /// Next instruction index.
    ip: u32,
    /// Label id of the block being executed.
    label: u32,
    /// Label id the current block was entered from (`0` = entry).
    pred: u32,
    /// Local id of this work-item.
    local: [u32; 3],
    /// Global id of this work-item.
    global: [u32; 3],
    /// Call stack, innermost last.
    frames: Vec<Frame>,
    /// Register file: one `u64` per scalar lane, laid out by
    /// `ValueSlot::off` in the loader's tables.
    regs: Vec<u64>,
    /// Private arena (function/private/input storage).
    arena: Vec<u8>,
    /// Address of `arena`'s buffer, or `0` when the arena is empty.
    arena_addr: usize,
    /// Rendezvous the work-item is waiting at, if [`Status::Parked`].
    park: Option<Park>,
    /// Scheduler state.
    status: Status,
    /// Instructions executed since the last re-initialization.
    steps: u64,
}

impl Item {
    /// Allocates a work-item's registers, arena and call stack up front
    /// so that running it can no longer fail on memory.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when a reservation fails.
    fn new(slots: usize, arena_bytes: usize, frame_cap: usize) -> Result<Self> {
        let mut regs: Vec<u64> = Vec::new();
        regs.try_reserve_exact(slots)
            .map_err(|_| Error::OutOfMemory)?;
        regs.resize(slots, 0);
        let mut arena: Vec<u8> = Vec::new();
        arena
            .try_reserve_exact(arena_bytes)
            .map_err(|_| Error::OutOfMemory)?;
        arena.resize(arena_bytes, 0);
        let mut frames: Vec<Frame> = Vec::new();
        frames
            .try_reserve_exact(frame_cap)
            .map_err(|_| Error::OutOfMemory)?;
        let arena_addr = if arena.is_empty() {
            0
        } else {
            arena.as_mut_ptr() as usize
        };
        Ok(Self {
            ip: 0,
            label: 0,
            pred: 0,
            local: [0; 3],
            global: [0; 3],
            frames,
            regs,
            arena,
            arena_addr,
            park: None,
            status: Status::Done,
            steps: 0,
        })
    }

    /// Resets the work-item for a new local/global id pair.
    fn reset(&mut self, local: [u32; 3], global: [u32; 3]) {
        self.ip = 0;
        self.label = 0;
        self.pred = 0;
        self.local = local;
        self.global = global;
        self.frames.clear();
        self.park = None;
        self.status = Status::Running;
        self.steps = 0;
        for lane in &mut self.regs {
            *lane = 0;
        }
        for byte in &mut self.arena {
            *byte = 0;
        }
    }
}

/// Four scratch windows of [`LANES`] lanes used to hold instruction
/// operands and results without allocating: 0, 1 and 2 are the operands
/// `a`, `b` and `c`, and results are computed in window 0 in place.
///
/// `[u64; LANES]` is larger than the largest array `Default` covers, so
/// the zeroed default is written out here instead.
#[derive(Debug)]
struct Scratch {
    /// Operand/result windows.
    windows: [[u64; LANES]; 4],
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            windows: [[0; LANES]; 4],
        }
    }
}

/// Immutable launch context shared by every work-item of one group.
struct Ctx<'a> {
    /// The loaded module.
    m: &'a SpirvModule,
    /// Marshalled kernel arguments.
    args: &'a ArgBlock,
    /// Group coordinates of the work-group being executed.
    group: [u32; 3],
    /// Number of work-groups per axis.
    groups: [u32; 3],
    /// Local size per axis.
    local: [u32; 3],
    /// Global size per axis (axes `>= dims` are 1).
    global: [u32; 3],
    /// Global offset per axis.
    offset: [u32; 3],
    /// Work dimension, `1..=3`.
    dims: u32,
    /// Address of the shared group arena, or `0` when there is none.
    group_addr: usize,
    /// Bytes of the shared group arena.
    group_len: usize,
}

/// Bit mask of the low `width` bits (`width >= 64` masks nothing).
#[must_use]
#[inline]
const fn mask(width: u16) -> u64 {
    if width == 0 {
        0
    } else if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Sign-extends the low `width` bits of `lane` to 64 bits.
#[must_use]
#[inline]
fn sext(lane: u64, width: u16) -> u64 {
    if width == 0 {
        return 0;
    }
    if width >= 64 {
        return lane;
    }
    let shift = 64 - width;
    ((lane << shift) as i64 >> shift) as u64
}

/// The scalar a value's type reduces to for arithmetic: one vector level
/// of indirection is unwrapped, exactly like the loader's own kind
/// classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Elem {
    /// A boolean (`width` one bit).
    Bool,
    /// An integer of `width` bits.
    Int {
        /// Bit width.
        width: u16,
        /// Whether the type is signed.
        signed: bool,
    },
    /// A floating-point number of `width` bits.
    Float {
        /// Bit width.
        width: u16,
    },
    /// A pointer (one native-sized lane).
    Ptr,
}

impl Elem {
    /// Bit width of one lane of this element.
    #[must_use]
    const fn width(self) -> u16 {
        match self {
            Self::Bool => 1,
            Self::Int { width, .. } | Self::Float { width } => width,
            Self::Ptr => (core::mem::size_of::<usize>() * 8) as u16,
        }
    }

    /// Whether lanes of this element compare and shift as signed values.
    #[must_use]
    const fn signed(self) -> bool {
        matches!(self, Self::Int { signed: true, .. })
    }

    /// Whether lanes of this element hold floating-point bits.
    #[must_use]
    const fn is_float(self) -> bool {
        matches!(self, Self::Float { .. })
    }
}

/// Reduces `ty` to its scalar element, unwrapping at most one vector
/// level.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id or a type that has no
/// scalar element (void, composites, function types).
fn elem_of(m: &SpirvModule, ty: u32) -> Result<Elem> {
    let entry = m
        .types
        .get(ty as usize)
        .ok_or(Error::InvalidBinary)?;
    Ok(match entry {
        Type::Bool => Elem::Bool,
        Type::Int { width, signed } => Elem::Int {
            width: *width,
            signed: *signed,
        },
        Type::Float { width } => Elem::Float { width: *width },
        Type::Pointer { .. } => Elem::Ptr,
        Type::Vector { elem, .. } => match m
            .types
            .get(*elem as usize)
            .ok_or(Error::InvalidBinary)?
        {
            Type::Bool => Elem::Bool,
            Type::Int { width, signed } => Elem::Int {
                width: *width,
                signed: *signed,
            },
            Type::Float { width } => Elem::Float { width: *width },
            _ => return Err(Error::InvalidBinary),
        },
        _ => return Err(Error::InvalidBinary),
    })
}

/// The declared type of a defined value.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown or never-defined id.
fn value_ty(m: &SpirvModule, id: u32) -> Result<u32> {
    let slot = m
        .values
        .get(id as usize)
        .ok_or(Error::InvalidBinary)?;
    if slot.ty == 0 {
        return Err(Error::InvalidBinary);
    }
    Ok(slot.ty)
}

/// Whether `id` names a declared type (the loader's own test).
#[must_use]
fn is_type(m: &SpirvModule, id: u32) -> bool {
    matches!(
        m.types.get(id as usize),
        Some(ty) if !matches!(ty, Type::Unsupported)
    )
}

/// Reads one value into `out`, returning the lane count written.
///
/// Constants are read from the constant pool; everything else from
/// `regs`. A void result writes nothing and returns `0`.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an undefined id, a value wider than
/// `out`, or a register/pool range outside its table.
fn read_value(m: &SpirvModule, regs: &[u64], id: u32, out: &mut [u64]) -> Result<u32> {
    let slot = m
        .values
        .get(id as usize)
        .ok_or(Error::InvalidBinary)?;
    if slot.ty == 0 {
        return Err(Error::InvalidBinary);
    }
    let count = slot.lanes;
    if count == 0 {
        return Ok(0);
    }
    if count as usize > out.len() || count as usize > LANES {
        return Err(Error::InvalidBinary);
    }
    let consts = m
        .consts
        .get(id as usize)
        .ok_or(Error::InvalidBinary)?;
    if consts.count > 0 {
        if consts.count != count {
            return Err(Error::InvalidBinary);
        }
        let start = consts.off as usize;
        let end = start
            .checked_add(count as usize)
            .ok_or(Error::InvalidBinary)?;
        let src = m
            .const_pool
            .get(start..end)
            .ok_or(Error::InvalidBinary)?;
        out[..count as usize].copy_from_slice(src);
        return Ok(count);
    }
    let start = slot.off as usize;
    let end = start
        .checked_add(count as usize)
        .ok_or(Error::InvalidBinary)?;
    let src = regs.get(start..end).ok_or(Error::InvalidBinary)?;
    out[..count as usize].copy_from_slice(src);
    Ok(count)
}

/// Reads the single lane of a scalar value.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an undefined id or a non-scalar value.
fn read_lane(m: &SpirvModule, regs: &[u64], id: u32) -> Result<u64> {
    let slot = m
        .values
        .get(id as usize)
        .ok_or(Error::InvalidBinary)?;
    if slot.ty == 0 {
        return Err(Error::InvalidBinary);
    }
    if slot.lanes != 1 {
        return Err(Error::InvalidBinary);
    }
    let consts = m
        .consts
        .get(id as usize)
        .ok_or(Error::InvalidBinary)?;
    if consts.count > 0 {
        return m
            .const_pool
            .get(consts.off as usize)
            .copied()
            .ok_or(Error::InvalidBinary);
    }
    regs.get(slot.off as usize)
        .copied()
        .ok_or(Error::InvalidBinary)
}

/// Writes `src` into a value's registers.
///
/// Void results (`lanes == 0`) are a no-op.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an undefined id or a source shorter than
/// the value.
fn write_value(m: &SpirvModule, regs: &mut [u64], id: u32, src: &[u64]) -> Result<()> {
    let slot = m
        .values
        .get(id as usize)
        .ok_or(Error::InvalidBinary)?;
    if slot.ty == 0 {
        return Err(Error::InvalidBinary);
    }
    let count = slot.lanes as usize;
    if count == 0 {
        return Ok(());
    }
    if count > src.len() || count > LANES {
        return Err(Error::InvalidBinary);
    }
    let start = slot.off as usize;
    let end = start
        .checked_add(count)
        .ok_or(Error::InvalidBinary)?;
    let dst = regs
        .get_mut(start..end)
        .ok_or(Error::InvalidBinary)?;
    dst.copy_from_slice(&src[..count]);
    Ok(())
}

/// Instruction index of a label id, minus one — the index of the label
/// itself so the interpreter lands *on* it and records the block.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when `label` is not a label of this module.
fn label_ip(m: &SpirvModule, label: u32) -> Result<u32> {
    let slot = m
        .label_ip
        .get(label as usize)
        .copied()
        .ok_or(Error::InvalidBinary)?;
    slot.checked_sub(1).ok_or(Error::InvalidBinary)
}

/// Byte range of the arena backing `storage`, when the storage class is
/// arena-resident (`Function`, `Private`, `Input` and `Workgroup`);
/// `None` marks caller-owned buffer storage, which has no range here.
fn arena_bounds(ctx: &Ctx<'_>, item: &Item, storage: u32) -> Option<(usize, usize)> {
    match storage {
        ops::STORAGE_FUNCTION | ops::STORAGE_PRIVATE | ops::STORAGE_INPUT => {
            Some((item.arena_addr, item.arena_addr.saturating_add(item.arena.len())))
        }
        ops::STORAGE_WORKGROUP => Some((ctx.group_addr, ctx.group_addr.saturating_add(ctx.group_len))),
        _ => None,
    }
}

/// Range-checks a `[addr, addr + size)` access of `storage` against its
/// arena; caller-owned buffer storage is unchecked here (the launch
/// contract covers it, exactly as for a native kernel).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the access leaves its arena.
fn check_region(ctx: &Ctx<'_>, item: &Item, storage: u32, addr: usize, size: u32) -> Result<()> {
    let Some((start, end)) = arena_bounds(ctx, item, storage) else {
        return Ok(());
    };
    let after = addr
        .checked_add(size as usize)
        .ok_or(Error::InvalidArgument)?;
    if addr < start || after > end {
        return Err(Error::InvalidArgument);
    }
    Ok(())
}

/// Byte offset of one `OpCompositeExtract`-style literal walk through
/// `start_ty`, with the leaf's lane count.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id, an index outside its
/// component, nesting past [`MAX_WALK_DEPTH`] or a non-composite step.
fn walk_lanes(m: &SpirvModule, start_ty: u32, literals: &[u32]) -> Result<(u32, u32)> {
    let mut ty = start_ty;
    let mut offset = 0u32;
    let mut depth = 0u32;
    for &literal in literals {
        if depth >= MAX_WALK_DEPTH {
            return Err(Error::InvalidBinary);
        }
        depth += 1;
        match m
            .types
            .get(ty as usize)
            .ok_or(Error::InvalidBinary)?
        {
            Type::Vector { elem, len } => {
                if literal >= u32::from(*len) {
                    return Err(Error::InvalidBinary);
                }
                let per = types::lanes(&m.types, *elem, &m.decor)?;
                offset = offset
                    .checked_add(
                        per.checked_mul(literal)
                            .ok_or(Error::InvalidBinary)?,
                    )
                    .ok_or(Error::InvalidBinary)?;
                ty = *elem;
            }
            Type::Array { elem, len } => {
                if literal >= *len {
                    return Err(Error::InvalidBinary);
                }
                let per = types::lanes(&m.types, *elem, &m.decor)?;
                offset = offset
                    .checked_add(
                        per.checked_mul(literal)
                            .ok_or(Error::InvalidBinary)?,
                    )
                    .ok_or(Error::InvalidBinary)?;
                ty = *elem;
            }
            Type::Struct { fields } => {
                let field = *fields
                    .get(literal as usize)
                    .ok_or(Error::InvalidBinary)?;
                for prior in fields.iter().take(literal as usize) {
                    offset = offset
                        .checked_add(types::lanes(&m.types, *prior, &m.decor)?)
                        .ok_or(Error::InvalidBinary)?;
                }
                ty = field;
            }
            _ => return Err(Error::InvalidBinary),
        }
    }
    let lanes = types::lanes(&m.types, ty, &m.decor)?;
    Ok((offset, lanes))
}

/// Value of a `BuiltIn` work-item query: three components, read in the
/// order the pointee declares them.
fn builtin_components(number: u32, ctx: &Ctx<'_>, item: &Item) -> [u64; 3] {
    let local_linear = u64::from(item.local[0])
        + u64::from(item.local[1]) * u64::from(ctx.local[0])
        + u64::from(item.local[2]) * u64::from(ctx.local[0]) * u64::from(ctx.local[1]);
    let local_product = u64::from(ctx.local[0]) * u64::from(ctx.local[1]) * u64::from(ctx.local[2]);
    let mut global_linear = 0u64;
    let mut stride = 1u64;
    for axis in 0..ctx.dims as usize {
        let id = i64::from(item.global[axis]) - i64::from(ctx.offset[axis]);
        global_linear += id.max(0) as u64 * stride;
        stride *= u64::from(ctx.global[axis]);
    }
    match number {
        24 => [
            u64::from(ctx.groups[0]),
            u64::from(ctx.groups[1]),
            u64::from(ctx.groups[2]),
        ],
        25 | 32 => [
            u64::from(ctx.local[0]),
            u64::from(ctx.local[1]),
            u64::from(ctx.local[2]),
        ],
        26 => [
            u64::from(ctx.group[0]),
            u64::from(ctx.group[1]),
            u64::from(ctx.group[2]),
        ],
        27 => [
            u64::from(item.local[0]),
            u64::from(item.local[1]),
            u64::from(item.local[2]),
        ],
        28 => [
            u64::from(item.global[0]),
            u64::from(item.global[1]),
            u64::from(item.global[2]),
        ],
        29 | 40 => [local_linear, 0, 0],
        30 => [u64::from(ctx.dims), 0, 0],
        31 => [
            u64::from(ctx.global[0]),
            u64::from(ctx.global[1]),
            u64::from(ctx.global[2]),
        ],
        33 => [
            u64::from(ctx.offset[0]),
            u64::from(ctx.offset[1]),
            u64::from(ctx.offset[2]),
        ],
        34 => [global_linear, 0, 0],
        // A subgroup is one work-item in this profile, so its size is
        // one and every work-item is its own subgroup id.
        36 | 37 => [1, 0, 0],
        38 | 39 => [local_product, 0, 0],
        _ => [0, 0, 0],
    }
}

/// Saturated add of two sign-extended `width`-bit values.
#[must_use]
fn s_add_sat(a: u64, b: u64, width: u16) -> u64 {
    let sum = sext(a, width) as i64;
    let sum = sum.saturating_add(sext(b, width) as i64);
    clamp_signed(sum, width)
}

/// Saturated subtract of two sign-extended `width`-bit values.
#[must_use]
fn s_sub_sat(a: u64, b: u64, width: u16) -> u64 {
    let diff = (sext(a, width) as i64).saturating_sub(sext(b, width) as i64);
    clamp_signed(diff, width)
}

/// Clamps a 64-bit value into the signed range of `width` bits and
/// masks it back to `width` bits.
#[must_use]
fn clamp_signed(value: i64, width: u16) -> u64 {
    let (min, max) = if width >= 64 {
        (i64::MIN, i64::MAX)
    } else {
        let half = 1i64 << (width - 1);
        (-(half), half - 1)
    };
    value.clamp(min, max) as u64 & mask(width)
}

/// Saturated add of two zero-extended `width`-bit values.
#[must_use]
fn u_add_sat(a: u64, b: u64, width: u16) -> u64 {
    let sum = u128::from(a & mask(width)) + u128::from(b & mask(width));
    (sum.min(u128::from(mask(width)))) as u64
}

/// Saturated subtract of two zero-extended `width`-bit values.
#[must_use]
fn u_sub_sat(a: u64, b: u64, width: u16) -> u64 {
    (a & mask(width)).saturating_sub(b & mask(width))
}

/// High half of the signed product of two `width`-bit values.
#[must_use]
fn s_mul_hi(a: u64, b: u64, width: u16) -> u64 {
    let product = sext(a, width) as i64 as i128 * sext(b, width) as i64 as i128;
    ((product >> width) as u64) & mask(width)
}

/// High half of the unsigned product of two `width`-bit values.
#[must_use]
fn u_mul_hi(a: u64, b: u64, width: u16) -> u64 {
    let product = u128::from(a & mask(width)) * u128::from(b & mask(width));
    ((product >> width) as u64) & mask(width)
}

/// `mul_hi` for the 24-bit variants: only the low 24 bits take part.
#[must_use]
fn mul24(a: u64, b: u64, width: u16, signed: bool) -> u64 {
    const LOW_24: u64 = 0x00FF_FFFF;
    if signed {
        let product = sext(a & LOW_24, 24) as i64 as i128 * sext(b & LOW_24, 24) as i64 as i128;
        product as u64 & mask(width)
    } else {
        (a & LOW_24).wrapping_mul(b & LOW_24) & mask(width)
    }
}

/// One-argument `OpenCL.std` integer instructions.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an instruction outside the integer set.
fn eval_int1(inst: u32, a: u64, width: u16) -> Result<u64> {
    let m = mask(width);
    let v = a & m;
    Ok(match inst {
        ops::OCL_S_ABS => (sext(a, width) as i64).unsigned_abs() & m,
        // `|x|` with `x` read as unsigned is `x` itself.
        ops::OCL_U_ABS => v,
        ops::OCL_CLZ => u64::from(v.leading_zeros() - (64 - u32::from(width))),
        ops::OCL_CTZ => {
            if v == 0 {
                u64::from(width)
            } else {
                u64::from(v.trailing_zeros())
            }
        }
        ops::OCL_POPCOUNT => u64::from(v.count_ones()),
        _ => return Err(Error::InvalidBinary),
    })
}

/// Two-argument `OpenCL.std` integer instructions. Signedness is chosen
/// by the instruction's own name (`s_*` vs `u_*`), not by the type.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an instruction outside the integer set
/// or an upsample whose result would not fit in a lane.
fn eval_int2(inst: u32, a: u64, b: u64, width: u16) -> Result<u64> {
    let m = mask(width);
    let x = a & m;
    let y = b & m;
    Ok(match inst {
        ops::OCL_S_ABS_DIFF => (sext(a, width) as i64).abs_diff(sext(b, width) as i64) as u64 & m,
        ops::OCL_U_ABS_DIFF => x.abs_diff(y),
        ops::OCL_S_ADD_SAT => s_add_sat(a, b, width),
        ops::OCL_U_ADD_SAT => u_add_sat(a, b, width),
        ops::OCL_S_SUB_SAT => s_sub_sat(a, b, width),
        ops::OCL_U_SUB_SAT => u_sub_sat(a, b, width),
        ops::OCL_S_HADD => {
            let sum = sext(a, width) as i128 + sext(b, width) as i128;
            ((sum >> 1) as u64) & m
        }
        ops::OCL_U_HADD => (((u128::from(x) + u128::from(y)) >> 1) as u64) & m,
        ops::OCL_S_RHADD => {
            let sum = sext(a, width) as i128 + sext(b, width) as i128 + 1;
            ((sum >> 1) as u64) & m
        }
        ops::OCL_U_RHADD => (((u128::from(x) + u128::from(y) + 1) >> 1) as u64) & m,
        ops::OCL_S_MAX => (sext(a, width) as i64).max(sext(b, width) as i64) as u64 & m,
        ops::OCL_U_MAX => x.max(y),
        ops::OCL_S_MIN => (sext(a, width) as i64).min(sext(b, width) as i64) as u64 & m,
        ops::OCL_U_MIN => x.min(y),
        ops::OCL_S_MUL_HI => s_mul_hi(a, b, width),
        ops::OCL_U_MUL_HI => u_mul_hi(a, b, width),
        ops::OCL_ROTATE => {
            let bits = u32::from(width);
            if bits == 0 {
                return Err(Error::InvalidBinary);
            }
            let shift = (y % u64::from(bits)) as u32;
            if shift == 0 {
                x
            } else {
                ((x << shift) | (x >> (bits - shift))) & m
            }
        }
        ops::OCL_U_UPSAMPLE | ops::OCL_S_UPSAMPLE => {
            // `hi` in the high half, `lo` in the low half; both are read
            // as raw bits of `width`, which is what the signed form
            // amounts to once the result is masked back.
            if width == 0 || width >= 64 {
                return Err(Error::InvalidBinary);
            }
            let wide = mask(width.wrapping_mul(2));
            ((x << width) | y) & wide
        }
        ops::OCL_S_MUL24 => mul24(a, b, width, true),
        ops::OCL_U_MUL24 => mul24(a, b, width, false),
        _ => return Err(Error::InvalidBinary),
    })
}

/// Three-argument `OpenCL.std` integer instructions: clamps and the
/// fused multiply-adds.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an instruction outside the integer set.
fn eval_int3(inst: u32, a: u64, b: u64, c: u64, width: u16) -> Result<u64> {
    let m = mask(width);
    Ok(match inst {
        ops::OCL_S_CLAMP => {
            let x = sext(a, width) as i64;
            let low = sext(b, width) as i64;
            let high = sext(c, width) as i64;
            x.max(low).min(high) as u64 & m
        }
        ops::OCL_U_CLAMP => (a & m).max(b & m).min(c & m),
        ops::OCL_S_MAD_HI => s_mul_hi(a, b, width).wrapping_add(c & m) & m,
        ops::OCL_U_MAD_HI => u_mul_hi(a, b, width).wrapping_add(c & m) & m,
        ops::OCL_S_MAD_SAT => s_add_sat(s_mul_hi(a, b, width), c, width),
        ops::OCL_U_MAD_SAT => u_add_sat(u_mul_hi(a, b, width), c, width),
        ops::OCL_S_MAD24 => mul24(a, b, width, true).wrapping_add(c & m) & m,
        ops::OCL_U_MAD24 => mul24(a, b, width, false).wrapping_add(c & m) & m,
        _ => return Err(Error::InvalidBinary),
    })
}

// Local `OpenCL.std` helpers libm does not carry. Each has an `f32` and
// an `f64` form so the dispatch macros below can pick by result width.
#[inline]
fn recip32(x: f32) -> f32 {
    1.0 / x
}
#[inline]
fn recip64(x: f64) -> f64 {
    1.0 / x
}
#[inline]
fn rsqrt32(x: f32) -> f32 {
    1.0 / libm::sqrtf(x)
}
#[inline]
fn rsqrt64(x: f64) -> f64 {
    1.0 / libm::sqrt(x)
}
#[inline]
fn degrees32(x: f32) -> f32 {
    x * (180.0 / core::f32::consts::PI)
}
#[inline]
fn degrees64(x: f64) -> f64 {
    x * (180.0 / core::f64::consts::PI)
}
#[inline]
fn radians32(x: f32) -> f32 {
    x * (core::f32::consts::PI / 180.0)
}
#[inline]
fn radians64(x: f64) -> f64 {
    x * (core::f64::consts::PI / 180.0)
}
/// C99 `logb`: the exponent of `x`, with the three special cases
/// `+-0 -> -inf`, `+-inf -> +inf`, `NaN -> NaN` that `ilogb` answers
/// with integer sentinels instead.
#[inline]
fn logb32(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x == 0.0 {
        f32::NEG_INFINITY
    } else if x.is_infinite() {
        f32::INFINITY
    } else {
        libm::ilogbf(x) as f32
    }
}
/// See [`logb32`].
#[inline]
fn logb64(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        f64::NEG_INFINITY
    } else if x.is_infinite() {
        f64::INFINITY
    } else {
        f64::from(libm::ilogb(x))
    }
}
#[inline]
fn acospif32(x: f32) -> f32 {
    libm::acosf(x) / core::f32::consts::PI
}
#[inline]
fn acospi64(x: f64) -> f64 {
    libm::acos(x) / core::f64::consts::PI
}
#[inline]
fn asinpif32(x: f32) -> f32 {
    libm::asinf(x) / core::f32::consts::PI
}
#[inline]
fn asinpi64(x: f64) -> f64 {
    libm::asin(x) / core::f64::consts::PI
}
#[inline]
fn atanpif32(x: f32) -> f32 {
    libm::atanf(x) / core::f32::consts::PI
}
#[inline]
fn atanpi64(x: f64) -> f64 {
    libm::atan(x) / core::f64::consts::PI
}
#[inline]
fn cospif32(x: f32) -> f32 {
    libm::cosf(x * core::f32::consts::PI)
}
#[inline]
fn cospi64(x: f64) -> f64 {
    libm::cos(x * core::f64::consts::PI)
}
#[inline]
fn sinpif32(x: f32) -> f32 {
    libm::sinf(x * core::f32::consts::PI)
}
#[inline]
fn sinpi64(x: f64) -> f64 {
    libm::sin(x * core::f64::consts::PI)
}
#[inline]
fn tanpif32(x: f32) -> f32 {
    libm::tanf(x * core::f32::consts::PI)
}
#[inline]
fn tanpi64(x: f64) -> f64 {
    libm::tan(x * core::f64::consts::PI)
}
#[inline]
fn step32(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}
#[inline]
fn step64(edge: f64, x: f64) -> f64 {
    if x < edge { 0.0 } else { 1.0 }
}
#[inline]
fn divide32(x: f32, y: f32) -> f32 {
    x / y
}
#[inline]
fn divide64(x: f64, y: f64) -> f64 {
    x / y
}
#[inline]
fn powr32(x: f32, y: f32) -> f32 {
    if x < 0.0 { f32::NAN } else { libm::powf(x, y) }
}
#[inline]
fn powr64(x: f64, y: f64) -> f64 {
    if x < 0.0 { f64::NAN } else { libm::pow(x, y) }
}
#[inline]
fn maxmag32(x: f32, y: f32) -> f32 {
    if x.abs() > y.abs() {
        x
    } else if y.abs() > x.abs() {
        y
    } else {
        f32::max(x, y)
    }
}
#[inline]
fn maxmag64(x: f64, y: f64) -> f64 {
    if x.abs() > y.abs() {
        x
    } else if y.abs() > x.abs() {
        y
    } else {
        f64::max(x, y)
    }
}
#[inline]
fn minmag32(x: f32, y: f32) -> f32 {
    if x.abs() < y.abs() {
        x
    } else if y.abs() < x.abs() {
        y
    } else {
        f32::min(x, y)
    }
}
#[inline]
fn minmag64(x: f64, y: f64) -> f64 {
    if x.abs() < y.abs() {
        x
    } else if y.abs() < x.abs() {
        y
    } else {
        f64::min(x, y)
    }
}
#[inline]
fn fmax_common32(x: f32, y: f32) -> f32 {
    if x < y { y } else { x }
}
#[inline]
fn fmax_common64(x: f64, y: f64) -> f64 {
    if x < y { y } else { x }
}
#[inline]
fn fmin_common32(x: f32, y: f32) -> f32 {
    if y < x { y } else { x }
}
#[inline]
fn fmin_common64(x: f64, y: f64) -> f64 {
    if y < x { y } else { x }
}

/// Generates the single-argument float dispatch: one arm per
/// instruction and result width, so `f32` kernels stay in `f32`.
macro_rules! float1_impl {
    ($($inst:ident => $f32fn:expr, $f64fn:expr;)*) => {
        /// Evaluates one single-argument float instruction on one lane.
        ///
        /// # Errors
        ///
        /// [`Error::InvalidBinary`] for an instruction outside the set or
        /// a result width other than 32 or 64 bits.
        fn eval_float1(inst: u32, lane: u64, width: u16) -> Result<u64> {
            Ok(match (inst, width) {
                $(
                    (ops::$inst, 32) => u64::from($f32fn(f32::from_bits(lane as u32)).to_bits()),
                    (ops::$inst, 64) => $f64fn(f64::from_bits(lane)).to_bits(),
                )*
                _ => return Err(Error::InvalidBinary),
            })
        }
    };
}

/// Generates the two-argument float dispatch (both operands share the
/// result's float type, which the caller has already checked).
macro_rules! float2_impl {
    ($($inst:ident => $f32fn:expr, $f64fn:expr;)*) => {
        /// Evaluates one two-argument float instruction on one lane.
        ///
        /// # Errors
        ///
        /// [`Error::InvalidBinary`] for an instruction outside the set or
        /// a result width other than 32 or 64 bits.
        fn eval_float2(inst: u32, a: u64, b: u64, width: u16) -> Result<u64> {
            Ok(match (inst, width) {
                $(
                    (ops::$inst, 32) => u64::from(
                        $f32fn(f32::from_bits(a as u32), f32::from_bits(b as u32)).to_bits(),
                    ),
                    (ops::$inst, 64) => $f64fn(f64::from_bits(a), f64::from_bits(b)).to_bits(),
                )*
                _ => return Err(Error::InvalidBinary),
            })
        }
    };
}

float1_impl! {
    OCL_ACOS => libm::acosf, libm::acos;
    OCL_ACOSH => libm::acoshf, libm::acosh;
    OCL_ACOSPI => acospif32, acospi64;
    OCL_ASIN => libm::asinf, libm::asin;
    OCL_ASINH => libm::asinhf, libm::asinh;
    OCL_ASINPI => asinpif32, asinpi64;
    OCL_ATAN => libm::atanf, libm::atan;
    OCL_ATANH => libm::atanhf, libm::atanh;
    OCL_ATANPI => atanpif32, atanpi64;
    OCL_CBRT => libm::cbrtf, libm::cbrt;
    OCL_CEIL => libm::ceilf, libm::ceil;
    OCL_COS => libm::cosf, libm::cos;
    OCL_COSH => libm::coshf, libm::cosh;
    OCL_COSPI => cospif32, cospi64;
    OCL_ERFC => libm::erfcf, libm::erfc;
    OCL_ERF => libm::erff, libm::erf;
    OCL_EXP => libm::expf, libm::exp;
    OCL_EXP2 => libm::exp2f, libm::exp2;
    OCL_EXP10 => libm::exp10f, libm::exp10;
    OCL_EXPM1 => libm::expm1f, libm::expm1;
    OCL_FABS => libm::fabsf, libm::fabs;
    OCL_FLOOR => libm::floorf, libm::floor;
    OCL_LGAMMA => libm::lgammaf, libm::lgamma;
    OCL_LOG => libm::logf, libm::log;
    OCL_LOG2 => libm::log2f, libm::log2;
    OCL_LOG10 => libm::log10f, libm::log10;
    OCL_LOG1P => libm::log1pf, libm::log1p;
    OCL_LOGB => logb32, logb64;
    OCL_RINT => libm::rintf, libm::rint;
    OCL_ROUND => libm::roundf, libm::round;
    OCL_RSQRT => rsqrt32, rsqrt64;
    OCL_SIN => libm::sinf, libm::sin;
    OCL_SINH => libm::sinhf, libm::sinh;
    OCL_SINPI => sinpif32, sinpi64;
    OCL_SQRT => libm::sqrtf, libm::sqrt;
    OCL_TAN => libm::tanf, libm::tan;
    OCL_TANH => libm::tanhf, libm::tanh;
    OCL_TANPI => tanpif32, tanpi64;
    OCL_TGAMMA => libm::tgammaf, libm::tgamma;
    OCL_TRUNC => libm::truncf, libm::trunc;
    OCL_DEGREES => degrees32, degrees64;
    OCL_RADIANS => radians32, radians64;
    OCL_HALF_COS => libm::cosf, libm::cos;
    OCL_HALF_EXP => libm::expf, libm::exp;
    OCL_HALF_EXP2 => libm::exp2f, libm::exp2;
    OCL_HALF_EXP10 => libm::exp10f, libm::exp10;
    OCL_HALF_LOG => libm::logf, libm::log;
    OCL_HALF_LOG2 => libm::log2f, libm::log2;
    OCL_HALF_LOG10 => libm::log10f, libm::log10;
    OCL_HALF_RECIP => recip32, recip64;
    OCL_HALF_RSQRT => rsqrt32, rsqrt64;
    OCL_HALF_SIN => libm::sinf, libm::sin;
    OCL_HALF_SQRT => libm::sqrtf, libm::sqrt;
    OCL_HALF_TAN => libm::tanf, libm::tan;
    OCL_NATIVE_COS => libm::cosf, libm::cos;
    OCL_NATIVE_EXP => libm::expf, libm::exp;
    OCL_NATIVE_EXP2 => libm::exp2f, libm::exp2;
    OCL_NATIVE_EXP10 => libm::exp10f, libm::exp10;
    OCL_NATIVE_LOG => libm::logf, libm::log;
    OCL_NATIVE_LOG2 => libm::log2f, libm::log2;
    OCL_NATIVE_LOG10 => libm::log10f, libm::log10;
    OCL_NATIVE_RECIP => recip32, recip64;
    OCL_NATIVE_RSQRT => rsqrt32, rsqrt64;
    OCL_NATIVE_SIN => libm::sinf, libm::sin;
    OCL_NATIVE_SQRT => libm::sqrtf, libm::sqrt;
    OCL_NATIVE_TAN => libm::tanf, libm::tan;
}

float2_impl! {
    OCL_ATAN2 => libm::atan2f, libm::atan2;
    OCL_ATAN2PI => atan2pif32, atan2pi64;
    OCL_COPYSIGN => libm::copysignf, libm::copysign;
    OCL_FDIM => libm::fdimf, libm::fdim;
    OCL_FMAX => libm::fmaxf, libm::fmax;
    OCL_FMIN => libm::fminf, libm::fmin;
    OCL_FMOD => libm::fmodf, libm::fmod;
    OCL_HYPOT => libm::hypotf, libm::hypot;
    OCL_NEXTAFTER => libm::nextafterf, libm::nextafter;
    OCL_POW => libm::powf, libm::pow;
    OCL_REMAINDER => libm::remainderf, libm::remainder;
    OCL_HALF_DIVIDE => divide32, divide64;
    OCL_NATIVE_DIVIDE => divide32, divide64;
    OCL_HALF_POWR => powr32, powr64;
    OCL_NATIVE_POWR => powr32, powr64;
    OCL_POWR => powr32, powr64;
    OCL_MAXMAG => maxmag32, maxmag64;
    OCL_MINMAG => minmag32, minmag64;
    OCL_F_MAX_COMMON => fmax_common32, fmax_common64;
    OCL_F_MIN_COMMON => fmin_common32, fmin_common64;
    OCL_STEP => step32, step64;
}

#[inline]
fn atan2pif32(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x) / core::f32::consts::PI
}
#[inline]
fn atan2pi64(y: f64, x: f64) -> f64 {
    libm::atan2(y, x) / core::f64::consts::PI
}

/// Three-argument float instructions: the fused multiply-adds, clamps
/// and the interpolation helpers. Written out by hand because each has
/// its own operand meaning rather than a plain function call.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an instruction outside the set or a
/// result width other than 32 or 64 bits.
fn eval_float3(inst: u32, a: u64, b: u64, c: u64, width: u16) -> Result<u64> {
    Ok(match (inst, width) {
        (ops::OCL_FMA, 32) => u64::from(
            libm::fmaf(
                f32::from_bits(a as u32),
                f32::from_bits(b as u32),
                f32::from_bits(c as u32),
            )
            .to_bits(),
        ),
        (ops::OCL_FMA, 64) => libm::fma(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c)).to_bits(),
        (ops::OCL_MAD, 32) => u64::from(
            (f32::from_bits(a as u32) * f32::from_bits(b as u32) + f32::from_bits(c as u32)).to_bits(),
        ),
        (ops::OCL_MAD, 64) => (f64::from_bits(a) * f64::from_bits(b) + f64::from_bits(c)).to_bits(),
        // `fclamp(x, minval, maxval)` = `fmin(fmax(x, minval), maxval)`,
        // written as `max`/`min` so a reversed pair can never panic.
        (ops::OCL_F_CLAMP, 32) => u64::from(
            f32::from_bits(a as u32)
                .max(f32::from_bits(b as u32))
                .min(f32::from_bits(c as u32))
                .to_bits(),
        ),
        (ops::OCL_F_CLAMP, 64) => f64::from_bits(a)
            .max(f64::from_bits(b))
            .min(f64::from_bits(c))
            .to_bits(),
        (ops::OCL_MIX, 32) => {
            let (x, y, t) = (
                f32::from_bits(a as u32),
                f32::from_bits(b as u32),
                f32::from_bits(c as u32),
            );
            u64::from((x + (y - x) * t).to_bits())
        }
        (ops::OCL_MIX, 64) => {
            let (x, y, t) = (f64::from_bits(a), f64::from_bits(b), f64::from_bits(c));
            (x + (y - x) * t).to_bits()
        }
        (ops::OCL_SMOOTHSTEP, 32) => {
            let (e0, e1, x) = (
                f32::from_bits(a as u32),
                f32::from_bits(b as u32),
                f32::from_bits(c as u32),
            );
            u64::from(smooth_step32(e0, e1, x).to_bits())
        }
        (ops::OCL_SMOOTHSTEP, 64) => {
            smooth_step64(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c)).to_bits()
        }
        _ => return Err(Error::InvalidBinary),
    })
}

/// `t = clamp((x - e0) / (e1 - e0), 0, 1); t * t * (3 - 2t)`, with a
/// degenerate edge range folding to zero instead of dividing by zero.
#[inline]
fn smooth_step32(e0: f32, e1: f32, x: f32) -> f32 {
    if !(e0 < e1) {
        return 0.0;
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// See [`smooth_step32`].
#[inline]
fn smooth_step64(e0: f64, e1: f64, x: f64) -> f64 {
    if !(e0 < e1) {
        return 0.0;
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `rootn(x, n)`: the `n`-th root of `x`, defined for negative `x`
/// only when `|n|` is odd.
#[inline]
fn rootn64(x: f64, n: i32) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    if x < 0.0 {
        let odd = (i64::from(n)).unsigned_abs() % 2 == 1;
        if !odd {
            return f64::NAN;
        }
        let root = -libm::pow(-x, 1.0 / (i64::from(n)).unsigned_abs() as f64);
        if n < 0 { 1.0 / root } else { root }
    } else {
        libm::pow(x, 1.0 / f64::from(n))
    }
}

/// See [`rootn64`].
#[inline]
fn rootn32(x: f32, n: i32) -> f32 {
    rootn64(f64::from(x), n) as f32
}

/// Executes one `OpExtInst`, leaving its result in scratch window 0.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unsupported instruction, an arity or
/// type mismatch, or a value that cannot be produced from its operands.
fn ext_inst(ctx: &Ctx<'_>, item: &mut Item, scratch: &mut Scratch, words: &[u32]) -> Result<()> {
    let m = ctx.m;
    if words.len() < 4 || m.opencl_set == 0 || words[2] != m.opencl_set {
        return Err(Error::InvalidBinary);
    }
    let ty = words[0];
    let id = words[1];
    let inst = words[3];
    let arity = ext_inst_arity(inst).ok_or(Error::InvalidBinary)? as usize;
    if words.len() != 4 + arity {
        return Err(Error::InvalidBinary);
    }
    let lanes = m
        .values
        .get(id as usize)
        .map_or(0, |value| value.lanes);
    // A void result (`prefetch`) computes nothing; the loader already
    // accepted the operand list.
    if lanes == 0 || inst == ops::OCL_PREFETCH || !(1..=3).contains(&arity) {
        return Ok(());
    }
    let n = lanes as usize;
    if n > LANES {
        return Err(Error::InvalidBinary);
    }
    let args = &words[4..];
    let mut arg_ty = [0u32; 3];
    for (index, &raw) in args.iter().enumerate() {
        arg_ty[index] = value_ty(m, raw)?;
        read_value(m, &item.regs, raw, &mut scratch.windows[index])?;
    }
    match inst {
        ops::OCL_NAN => {
            let width = float_width(elem_of(m, ty)?)?;
            let bits = nan_bits(width);
            scratch.windows[0][..n].fill(bits);
        }
        ops::OCL_ILOGB => {
            let width = match elem_of(m, ty)? {
                Elem::Int { width, .. } => width,
                _ => return Err(Error::InvalidBinary),
            };
            let arg_width = float_width(elem_of(m, arg_ty[0])?)?;
            let out = &mut scratch.windows[0];
            for lane in 0..n {
                let value = match arg_width {
                    32 => i64::from(libm::ilogbf(f32::from_bits(out[lane] as u32))),
                    64 => i64::from(libm::ilogb(f64::from_bits(out[lane]))),
                    _ => return Err(Error::InvalidBinary),
                };
                out[lane] = value as u64 & mask(width);
            }
        }
        ops::OCL_SIGN => {
            require_same(arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            let out = &mut scratch.windows[0];
            for lane in 0..n {
                out[lane] = match width {
                    32 => u64::from(sign32(f32::from_bits(out[lane] as u32)).to_bits()),
                    64 => sign64(f64::from_bits(out[lane])).to_bits(),
                    _ => return Err(Error::InvalidBinary),
                };
            }
        }
        ops::OCL_LENGTH | ops::OCL_FAST_LENGTH => {
            let width = float_width(elem_of(m, ty)?)?;
            if n != 1 {
                return Err(Error::InvalidBinary);
            }
            let arg_width = float_width(elem_of(m, arg_ty[0])?)?;
            let sum = norm_sum(&scratch.windows[0], value_lanes(m, arg_ty[0])?, arg_width)?;
            scratch.windows[0][0] = float_bits(width, libm::sqrt(sum));
        }
        ops::OCL_DISTANCE | ops::OCL_FAST_DISTANCE => {
            let width = float_width(elem_of(m, ty)?)?;
            if n != 1 {
                return Err(Error::InvalidBinary);
            }
            let count = value_lanes(m, arg_ty[0])?;
            if count != value_lanes(m, arg_ty[1])? {
                return Err(Error::InvalidBinary);
            }
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut sum = 0.0f64;
            for lane in 0..count as usize {
                let d = to_f64(a[lane], width) - to_f64(b[lane], width);
                sum += d * d;
            }
            scratch.windows[0][0] = float_bits(width, libm::sqrt(sum));
        }
        ops::OCL_NORMALIZE | ops::OCL_FAST_NORMALIZE => {
            require_same(arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            let count = value_lanes(m, arg_ty[0])?;
            if count != n as u32 {
                return Err(Error::InvalidBinary);
            }
            let sum = norm_sum(&scratch.windows[0], count, width)?;
            let len = libm::sqrt(sum);
            let out = &mut scratch.windows[0];
            for lane in 0..n {
                out[lane] = float_bits(width, to_f64(out[lane], width) / len);
            }
        }
        ops::OCL_CROSS => {
            require_same(arg_ty[0], ty)?;
            require_same(arg_ty[1], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            let count = value_lanes(m, arg_ty[0])?;
            if count < 3 || n < 3 {
                return Err(Error::InvalidBinary);
            }
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut cross = [0.0f64; 3];
            cross[0] = to_f64(a[1], width) * to_f64(b[2], width) - to_f64(a[2], width) * to_f64(b[1], width);
            cross[1] = to_f64(a[2], width) * to_f64(b[0], width) - to_f64(a[0], width) * to_f64(b[2], width);
            cross[2] = to_f64(a[0], width) * to_f64(b[1], width) - to_f64(a[1], width) * to_f64(b[0], width);
            let out = &mut scratch.windows[0];
            for lane in 0..n {
                out[lane] = if lane < 3 {
                    float_bits(width, cross[lane])
                } else {
                    float_bits(width, 0.0)
                };
            }
        }
        ops::OCL_SELECT => {
            require_same(arg_ty[0], ty)?;
            require_same(arg_ty[1], ty)?;
            let cond_ty = arg_ty[2];
            let cond_elem = elem_of(m, cond_ty)?;
            let cond_lanes = value_lanes(m, cond_ty)?;
            if value_lanes(m, ty)? != n as u32 {
                return Err(Error::InvalidBinary);
            }
            let bits = match cond_elem {
                Elem::Int { .. } | Elem::Bool => mask(cond_elem.width()),
                Elem::Float { .. } | Elem::Ptr => return Err(Error::InvalidBinary),
            };
            let high = if bits == u64::MAX {
                1u64 << 63
            } else {
                1u64 << (64 - bits.leading_zeros() - 1)
            };
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let c = &scratch.windows[2];
            let mut out = [0u64; LANES];
            for lane in 0..n {
                let pick = if cond_lanes == 1 {
                    c[0] != 0
                } else if cond_lanes == n as u32 {
                    c[lane] & high != 0
                } else {
                    return Err(Error::InvalidBinary);
                };
                out[lane] = if pick { b[lane] } else { a[lane] };
            }
            scratch.windows[0][..n].copy_from_slice(&out[..n]);
        }
        ops::OCL_BITSELECT => {
            require_same(arg_ty[0], ty)?;
            require_same(arg_ty[1], ty)?;
            require_same(arg_ty[2], ty)?;
            let bits = mask(elem_of(m, ty)?.width());
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let c = &scratch.windows[2];
            let mut out = [0u64; LANES];
            for lane in 0..n {
                out[lane] = (a[lane] & !c[lane] | (b[lane] & c[lane])) & bits;
            }
            scratch.windows[0][..n].copy_from_slice(&out[..n]);
        }
        ops::OCL_LDEXP => {
            require_same(arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            require_int(elem_of(m, arg_ty[1])?)?;
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut out = [0u64; LANES];
            for lane in 0..n {
                let k = b[lane] as i32;
                out[lane] = match width {
                    32 => u64::from(libm::scalbnf(f32::from_bits(a[lane] as u32), k).to_bits()),
                    64 => libm::scalbn(f64::from_bits(a[lane]), k).to_bits(),
                    _ => return Err(Error::InvalidBinary),
                };
            }
            scratch.windows[0][..n].copy_from_slice(&out[..n]);
        }
        ops::OCL_POWN => {
            require_same(arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            require_int(elem_of(m, arg_ty[1])?)?;
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut out = [0u64; LANES];
            for lane in 0..n {
                let y = b[lane] as i32;
                out[lane] = match width {
                    32 => u64::from(libm::powf(f32::from_bits(a[lane] as u32), y as f32).to_bits()),
                    64 => libm::pow(f64::from_bits(a[lane]), f64::from(y)).to_bits(),
                    _ => return Err(Error::InvalidBinary),
                };
            }
            scratch.windows[0][..n].copy_from_slice(&out[..n]);
        }
        ops::OCL_ROOTN => {
            require_same(arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            require_int(elem_of(m, arg_ty[1])?)?;
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut out = [0u64; LANES];
            for lane in 0..n {
                let y = b[lane] as i32;
                out[lane] = match width {
                    32 => u64::from(rootn32(f32::from_bits(a[lane] as u32), y).to_bits()),
                    64 => rootn64(f64::from_bits(a[lane]), y).to_bits(),
                    _ => return Err(Error::InvalidBinary),
                };
            }
            scratch.windows[0][..n].copy_from_slice(&out[..n]);
        }
        _ => {
            // Generic dispatch: every operand shares the result type, so
            // one element kind describes the whole instruction.
            for &given in &arg_ty[..arity] {
                if given != ty {
                    return Err(Error::InvalidBinary);
                }
            }
            let elem = elem_of(m, ty)?;
            if value_lanes(m, arg_ty[0])? != lanes {
                return Err(Error::InvalidBinary);
            }
            let width = elem.width();
            let mut results = [0u64; LANES];
            match elem {
                Elem::Float { .. } => {
                    for lane in 0..n {
                        let a = scratch.windows[0][lane];
                        let b = scratch.windows[1][lane];
                        let c = scratch.windows[2][lane];
                        results[lane] = match arity {
                            1 => eval_float1(inst, a, width)?,
                            2 => eval_float2(inst, a, b, width)?,
                            3 => eval_float3(inst, a, b, c, width)?,
                            _ => return Err(Error::InvalidBinary),
                        };
                    }
                }
                Elem::Int { width, .. } => {
                    for lane in 0..n {
                        let a = scratch.windows[0][lane];
                        let b = scratch.windows[1][lane];
                        let c = scratch.windows[2][lane];
                        results[lane] = match arity {
                            1 => eval_int1(inst, a, width)?,
                            2 => eval_int2(inst, a, b, width)?,
                            3 => eval_int3(inst, a, b, c, width)?,
                            _ => return Err(Error::InvalidBinary),
                        };
                    }
                }
                Elem::Bool | Elem::Ptr => return Err(Error::InvalidBinary),
            }
            scratch.windows[0][..n].copy_from_slice(&results[..n]);
        }
    }
    write_value(m, &mut item.regs, id, &scratch.windows[0])
}

/// Bit width of a float element, rejecting anything else.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when `elem` is not a float.
fn float_width(elem: Elem) -> Result<u16> {
    match elem {
        Elem::Float { width } => Ok(width),
        _ => Err(Error::InvalidBinary),
    }
}

/// Rejects a non-integer element.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when `elem` is not an integer.
fn require_int(elem: Elem) -> Result<()> {
    match elem {
        Elem::Int { .. } => Ok(()),
        _ => Err(Error::InvalidBinary),
    }
}

/// Rejects a type that differs from `expected`.
///
/// # Errors
///
/// [`Error::InvalidBinary`] on a mismatch.
fn require_same(given: u32, expected: u32) -> Result<()> {
    if given != expected {
        return Err(Error::InvalidBinary);
    }
    Ok(())
}

/// Number of register lanes a value of `ty` holds.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown or uncapped type.
fn value_lanes(m: &SpirvModule, ty: u32) -> Result<u32> {
    types::lanes(&m.types, ty, &m.decor)
}

/// Quiet-NaN payload bits of `width`.
#[must_use]
fn nan_bits(width: u16) -> u64 {
    match width {
        32 => u64::from(f32::NAN.to_bits()),
        _ => f64::NAN.to_bits(),
    }
}

/// Packs a `f64` value into a `width`-bit lane.
#[must_use]
fn float_bits(width: u16, value: f64) -> u64 {
    match width {
        32 => u64::from((value as f32).to_bits()),
        _ => value.to_bits(),
    }
}

/// Reads a `width`-bit lane as `f64`.
#[must_use]
fn to_f64(lane: u64, width: u16) -> f64 {
    match width {
        32 => f64::from(f32::from_bits(lane as u32)),
        _ => f64::from_bits(lane),
    }
}

/// Sum of squares of `count` lanes, read as `width`-bit floats.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when `count` exceeds the window.
fn norm_sum(lanes: &[u64; LANES], count: u32, width: u16) -> Result<f64> {
    let count = count as usize;
    if count == 0 || count > LANES {
        return Err(Error::InvalidBinary);
    }
    let mut sum = 0.0f64;
    for lane in lanes[..count].iter() {
        let value = to_f64(*lane, width);
        sum += value * value;
    }
    Ok(sum)
}

/// `sign` for one `f32` lane: `NaN -> 0.0`, zero keeps its sign.
#[must_use]
fn sign32(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// See [`sign32`].
#[must_use]
fn sign64(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// Storage class and pointee of a pointer type.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id or a type that is not a
/// pointer.
fn pointer_type(m: &SpirvModule, ty: u32) -> Result<(u32, u32)> {
    match m
        .types
        .get(ty as usize)
        .ok_or(Error::InvalidBinary)?
    {
        Type::Pointer { storage, pointee } => Ok((*storage, *pointee)),
        _ => Err(Error::InvalidBinary),
    }
}

/// Walks one `AccessChain` index list from `ty` at `addr`, stepping each
/// index by its component's stride in bytes.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown type, an index outside its
/// component, nesting past [`MAX_WALK_DEPTH`] or a step that leaves the
/// addressable range.
fn walk_indices(
    m: &SpirvModule,
    regs: &[u64],
    ty: u32,
    mut addr: usize,
    indices: &[u32],
) -> Result<(usize, u32)> {
    let mut current = ty;
    let mut depth = 0u32;
    for &index in indices {
        if depth >= MAX_WALK_DEPTH {
            return Err(Error::InvalidBinary);
        }
        depth += 1;
        let step = match m
            .types
            .get(current as usize)
            .ok_or(Error::InvalidBinary)?
        {
            Type::Vector { elem, len } => {
                let position = read_lane(m, regs, index)?;
                if position >= u64::from(*len) {
                    return Err(Error::InvalidBinary);
                }
                current = *elem;
                types::size_of(&m.types, *elem, &m.decor)? as u64
            }
            Type::Array { elem, len } => {
                let position = read_lane(m, regs, index)?;
                if position >= u64::from(*len) {
                    return Err(Error::InvalidBinary);
                }
                let stride = m
                    .ty_stride
                    .get(current as usize)
                    .copied()
                    .filter(|stride| *stride > 0)
                    .ok_or(Error::InvalidBinary)?;
                current = *elem;
                u64::from(stride)
            }
            Type::Struct { fields } => {
                let position = read_lane(m, regs, index)?;
                let field = *fields
                    .get(position as usize)
                    .ok_or(Error::InvalidBinary)?;
                let key = (u64::from(current) << 32) | u64::from(position);
                let found = m
                    .struct_off
                    .binary_search_by_key(&key, |entry| entry.0)
                    .map_err(|_| Error::InvalidBinary)?;
                current = field;
                u64::from(m.struct_off[found].1)
            }
            _ => return Err(Error::InvalidBinary),
        };
        let step = i128::from(step);
        addr = apply_offset(addr, step)?;
    }
    Ok((addr, current))
}

/// Adds a signed byte displacement to an address, rejecting anything
/// that leaves the `usize` range.
///
/// # Errors
///
/// [`Error::InvalidBinary`] on an overflow or a negative result.
fn apply_offset(addr: usize, displacement: i128) -> Result<usize> {
    let next = addr as i128 + displacement;
    if next < 0 {
        return Err(Error::InvalidBinary);
    }
    usize::try_from(next).map_err(|_| Error::InvalidBinary)
}

/// One work-item's arithmetic binary instruction on one lane.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an opcode outside the arithmetic set or
/// a float opcode on a non-float element.
fn eval_arith(opcode: u16, elem: Elem, a: u64, b: u64) -> Result<u64> {
    let width = elem.width();
    let bits = mask(width);
    if elem.is_float() {
        return Ok(match (opcode, width) {
            (ops::OP_F_ADD, 32) => u64::from((f32::from_bits(a as u32) + f32::from_bits(b as u32)).to_bits()),
            (ops::OP_F_ADD, 64) => (f64::from_bits(a) + f64::from_bits(b)).to_bits(),
            (ops::OP_F_SUB, 32) => u64::from((f32::from_bits(a as u32) - f32::from_bits(b as u32)).to_bits()),
            (ops::OP_F_SUB, 64) => (f64::from_bits(a) - f64::from_bits(b)).to_bits(),
            (ops::OP_F_MUL, 32) => u64::from((f32::from_bits(a as u32) * f32::from_bits(b as u32)).to_bits()),
            (ops::OP_F_MUL, 64) => (f64::from_bits(a) * f64::from_bits(b)).to_bits(),
            (ops::OP_F_DIV, 32) => u64::from((f32::from_bits(a as u32) / f32::from_bits(b as u32)).to_bits()),
            (ops::OP_F_DIV, 64) => (f64::from_bits(a) / f64::from_bits(b)).to_bits(),
            (ops::OP_F_REM, 32) => {
                let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                if y == 0.0 {
                    0
                } else {
                    u64::from(libm::fmodf(x, y).to_bits())
                }
            }
            (ops::OP_F_REM, 64) => {
                let (x, y) = (f64::from_bits(a), f64::from_bits(b));
                if y == 0.0 { 0 } else { libm::fmod(x, y).to_bits() }
            }
            (ops::OP_F_MOD, 32) => {
                let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                if y == 0.0 {
                    0
                } else {
                    u64::from((x - y * libm::floorf(x / y)).to_bits())
                }
            }
            (ops::OP_F_MOD, 64) => {
                let (x, y) = (f64::from_bits(a), f64::from_bits(b));
                if y == 0.0 {
                    0
                } else {
                    (x - y * libm::floor(x / y)).to_bits()
                }
            }
            _ => return Err(Error::InvalidBinary),
        });
    }
    let (sa, sb) = (sext(a, width) as i64, sext(b, width) as i64);
    Ok(match opcode {
        ops::OP_I_ADD => a.wrapping_add(b) & bits,
        ops::OP_I_SUB => a.wrapping_sub(b) & bits,
        ops::OP_I_MUL => a.wrapping_mul(b) & bits,
        ops::OP_U_DIV => {
            if b & bits == 0 {
                0
            } else {
                (a & bits) / (b & bits)
            }
        }
        ops::OP_S_DIV => {
            if sb == 0 {
                0
            } else {
                sa.checked_div(sb).unwrap_or(i64::MIN) as u64 & bits
            }
        }
        ops::OP_U_MOD => {
            if b & bits == 0 {
                0
            } else {
                (a & bits) % (b & bits)
            }
        }
        ops::OP_S_REM => {
            if sb == 0 {
                0
            } else {
                sa.checked_rem(sb).unwrap_or(0) as u64 & bits
            }
        }
        ops::OP_LOGICAL_OR => u64::from(a != 0 || b != 0),
        ops::OP_LOGICAL_AND => u64::from(a != 0 && b != 0),
        ops::OP_BITWISE_OR => (a | b) & bits,
        ops::OP_BITWISE_XOR => (a ^ b) & bits,
        ops::OP_BITWISE_AND => (a & b) & bits,
        ops::OP_SHIFT_RIGHT_LOGICAL | ops::OP_SHIFT_RIGHT_ARITHMETIC | ops::OP_SHIFT_LEFT_LOGICAL => {
            shift_lane(opcode, a, b, elem)
        }
        _ => return Err(Error::InvalidBinary),
    })
}

/// One shift lane: the shift amount is taken modulo the value width only
/// as far as the SPIR-V rule allows — an amount at or above the width
/// produces all zeros (logical) or all sign bits (arithmetic).
#[must_use]
fn shift_lane(opcode: u16, a: u64, b: u64, elem: Elem) -> u64 {
    let width = elem.width();
    let bits = mask(width);
    let value = a & bits;
    let amount = b & bits;
    if amount >= u64::from(width) {
        if opcode == ops::OP_SHIFT_RIGHT_ARITHMETIC && elem.signed() {
            return bits;
        }
        return 0;
    }
    let shift = amount as u32;
    match opcode {
        ops::OP_SHIFT_RIGHT_LOGICAL => (value >> shift) & bits,
        ops::OP_SHIFT_RIGHT_ARITHMETIC => ((sext(value, width) as i64) >> shift) as u64 & bits,
        _ => (value << shift) & bits,
    }
}

/// One work-item's comparison instruction on one lane: `0` or `1`.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an opcode outside the comparison set.
fn eval_compare(opcode: u16, elem: Elem, a: u64, b: u64) -> Result<u64> {
    let width = elem.width();
    let bits = mask(width);
    Ok(if elem.is_float() {
        let (x, y) = match width {
            32 => (
                f64::from(f32::from_bits(a as u32)),
                f64::from(f32::from_bits(b as u32)),
            ),
            64 => (f64::from_bits(a), f64::from_bits(b)),
            _ => return Err(Error::InvalidBinary),
        };
        match opcode {
            ops::OP_F_ORD_EQUAL => u64::from(x == y),
            ops::OP_F_ORD_NOT_EQUAL => u64::from(x != y),
            ops::OP_F_ORD_LESS_THAN => u64::from(x < y),
            ops::OP_F_ORD_GREATER_THAN => u64::from(x > y),
            ops::OP_F_ORD_LESS_THAN_EQUAL => u64::from(x <= y),
            ops::OP_F_ORD_GREATER_THAN_EQUAL => u64::from(x >= y),
            _ => return Err(Error::InvalidBinary),
        }
    } else {
        let (x, y) = if elem.signed() {
            (sext(a, width) as i64, sext(b, width) as i64)
        } else {
            ((a & bits) as i64, (b & bits) as i64)
        };
        match opcode {
            ops::OP_I_EQUAL => u64::from(x == y),
            ops::OP_I_NOT_EQUAL => u64::from(x != y),
            ops::OP_U_GREATER_THAN | ops::OP_S_GREATER_THAN => u64::from(x > y),
            ops::OP_U_GREATER_THAN_EQUAL | ops::OP_S_GREATER_THAN_EQUAL => u64::from(x >= y),
            ops::OP_U_LESS_THAN | ops::OP_S_LESS_THAN => u64::from(x < y),
            ops::OP_U_LESS_THAN_EQUAL | ops::OP_S_LESS_THAN_EQUAL => u64::from(x <= y),
            _ => return Err(Error::InvalidBinary),
        }
    })
}

/// Saturating float-to-unsigned conversion: `NaN` becomes zero and every
/// value outside `[0, 2^width - 1]` clamps to its nearer bound.
#[must_use]
fn f_to_u(value: f64, width: u16) -> u64 {
    if value.is_nan() {
        return 0;
    }
    // `mask(width) as f64` rounds `2^64 - 1` up to `2^64` for a 64-bit
    // result, which the saturating `as` cast turns back into `u64::MAX`.
    let clamped = value.clamp(0.0, mask(width) as f64);
    clamped as u64 & mask(width)
}

/// Saturating float-to-signed conversion, as [`f_to_u`].
#[must_use]
fn f_to_s(value: f64, width: u16) -> u64 {
    if value.is_nan() {
        return 0;
    }
    let shift = if width >= 64 { 0 } else { 64 - width };
    let min = (i64::MIN >> shift) as f64;
    let max = (i64::MAX >> shift) as f64;
    let clamped = value.clamp(min, max);
    clamped as i64 as u64 & mask(width)
}

/// Executes one instruction of `item`, then advances it (or parks it).
///
/// # Errors
///
/// * [`Error::InvalidBinary`] — unsupported opcode, operand shape or
///   value/type mismatch.
/// * [`Error::InvalidArgument`] — a memory access outside its arena.
/// * [`Error::OutOfResources`] — the step budget of [`MAX_STEPS`] is
///   spent.
fn step(ctx: &Ctx<'_>, item: &mut Item, scratch: &mut Scratch) -> Result<()> {
    let m = ctx.m;
    item.steps = item
        .steps
        .checked_add(1)
        .ok_or(Error::OutOfResources)?;
    if item.steps > MAX_STEPS {
        return Err(Error::OutOfResources);
    }
    let instr = *m
        .instrs
        .get(item.ip as usize)
        .ok_or(Error::InvalidBinary)?;
    let opcode = instr.opcode;
    let words = instr.words(&m.words);
    let mut next = item
        .ip
        .checked_add(1)
        .ok_or(Error::InvalidBinary)?;
    match opcode {
        ops::OP_NOP
        | ops::OP_LINE
        | ops::OP_NO_LINE
        | ops::OP_SELECTION_MERGE
        | ops::OP_LOOP_MERGE
        | ops::OP_MEMORY_BARRIER => {}
        ops::OP_LABEL => {
            item.label = *words.first().ok_or(Error::InvalidBinary)?;
        }
        ops::OP_BRANCH => {
            let target = *words.first().ok_or(Error::InvalidBinary)?;
            item.pred = item.label;
            next = label_ip(m, target)?;
        }
        ops::OP_BRANCH_CONDITIONAL => {
            let cond = *words.first().ok_or(Error::InvalidBinary)?;
            let taken_label = *words.get(1).ok_or(Error::InvalidBinary)?;
            let other_label = *words.get(2).ok_or(Error::InvalidBinary)?;
            let slot = m
                .values
                .get(cond as usize)
                .ok_or(Error::InvalidBinary)?;
            if slot.lanes != 1 || !matches!(m.types.get(slot.ty as usize), Some(Type::Bool)) {
                return Err(Error::InvalidBinary);
            }
            let taken = read_lane(m, &item.regs, cond)? != 0;
            item.pred = item.label;
            next = label_ip(m, if taken { taken_label } else { other_label })?;
        }
        ops::OP_SWITCH => {
            let selector = *words.first().ok_or(Error::InvalidBinary)?;
            let mut target = *words.get(1).ok_or(Error::InvalidBinary)?;
            let width = elem_of(m, value_ty(m, selector)?)?.width();
            if width > 32 {
                return Err(Error::InvalidBinary);
            }
            let value = read_lane(m, &item.regs, selector)? & mask(width);
            let mut pair = 2usize;
            while pair < words.len() {
                let literal = u64::from(words[pair]) & mask(width);
                let label = *words.get(pair + 1).ok_or(Error::InvalidBinary)?;
                if literal == value {
                    target = label;
                    break;
                }
                pair += 2;
            }
            item.pred = item.label;
            next = label_ip(m, target)?;
        }
        ops::OP_RETURN | ops::OP_FUNCTION_END => {
            if let Some(frame) = item.frames.pop() {
                item.label = frame.label;
                item.pred = frame.pred;
                next = frame.ret_ip;
            } else {
                item.status = Status::Done;
                return Ok(());
            }
        }
        ops::OP_RETURN_VALUE => {
            let value = *words.first().ok_or(Error::InvalidBinary)?;
            read_value(m, &item.regs, value, &mut scratch.windows[0])?;
            if let Some(frame) = item.frames.pop() {
                if frame.result != 0 {
                    write_value(m, &mut item.regs, frame.result, &scratch.windows[0])?;
                }
                item.label = frame.label;
                item.pred = frame.pred;
                next = frame.ret_ip;
            } else {
                item.status = Status::Done;
                return Ok(());
            }
        }
        ops::OP_UNREACHABLE => return Err(Error::InvalidBinary),
        ops::OP_PHI => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            if words.len() < 4 || (words.len() - 2) % 2 != 0 {
                return Err(Error::InvalidBinary);
            }
            let mut index = 2usize;
            let mut found = false;
            while index + 1 < words.len() {
                let value = words[index];
                let parent = words[index + 1];
                if parent == item.pred {
                    read_value(m, &item.regs, value, &mut scratch.windows[0])?;
                    write_value(m, &mut item.regs, id, &scratch.windows[0])?;
                    found = true;
                    break;
                }
                index += 2;
            }
            if !found {
                return Err(Error::InvalidBinary);
            }
        }
        ops::OP_LOAD => {
            // `OpLoad ResultType ResultId Pointer` — the result type is
            // operand 0, exactly as `module::load` validated it.
            if words.len() < 3 {
                return Err(Error::InvalidBinary);
            }
            let id = words[1];
            let pointer = words[2];
            let ty = value_ty(m, id)?;
            let addr = read_lane(m, &item.regs, pointer)? as usize;
            let size = types::size_of(&m.types, ty, &m.decor)?;
            let (storage, _) = pointer_type(m, value_ty(m, pointer)?)?;
            check_region(ctx, item, storage, addr, size)?;
            let lanes = types::lanes(&m.types, ty, &m.decor)?;
            if lanes as usize > LANES {
                return Err(Error::InvalidBinary);
            }
            // SAFETY: `check_region` proved `[addr, addr + size)` sits
            // inside an arena (or a caller-owned buffer, whose lifetime
            // the launch contract covers), and `size` is the byte size
            // of `ty` from the same layout tables the loader validated.
            let used = unsafe {
                types::load_lanes(&m.types, ty, &m.decor, addr as *const u8, &mut scratch.windows[0])?
            };
            if used != lanes {
                return Err(Error::InvalidBinary);
            }
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_STORE => {
            let pointer = *words.first().ok_or(Error::InvalidBinary)?;
            let value = *words.get(1).ok_or(Error::InvalidBinary)?;
            let ty = value_ty(m, value)?;
            let addr = read_lane(m, &item.regs, pointer)? as usize;
            let size = types::size_of(&m.types, ty, &m.decor)?;
            let (storage, _) = pointer_type(m, value_ty(m, pointer)?)?;
            check_region(ctx, item, storage, addr, size)?;
            read_value(m, &item.regs, value, &mut scratch.windows[0])?;
            // SAFETY: as for `OpLoad`, with `store_lanes` writing at most
            // the lanes `ty` occupies inside that same checked range.
            unsafe {
                types::store_lanes(&m.types, ty, &m.decor, addr as *mut u8, &scratch.windows[0])?;
            }
        }
        ops::OP_ACCESS_CHAIN | ops::OP_IN_BOUNDS_ACCESS_CHAIN => {
            if words.len() < 4 {
                return Err(Error::InvalidBinary);
            }
            let id = words[1];
            let base = words[2];
            let (_, pointee) = pointer_type(m, value_ty(m, base)?)?;
            let base_addr = read_lane(m, &item.regs, base)? as usize;
            let (addr, _) = walk_indices(m, &item.regs, pointee, base_addr, &words[3..])?;
            scratch.windows[0][0] = addr as u64;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_PTR_ACCESS_CHAIN => {
            // `OpPtrAccessChain ResultType ResultId Base Element [Indexes…]`
            // — four operands minimum, exactly what `module::access_chain`
            // requires; the trailing index list may be empty.
            if words.len() < 4 {
                return Err(Error::InvalidBinary);
            }
            let id = words[1];
            let base = words[2];
            let element = words[3];
            let (_, pointee) = pointer_type(m, value_ty(m, base)?)?;
            let base_addr = read_lane(m, &item.regs, base)? as usize;
            let element_ty = value_ty(m, element)?;
            let element_elem = elem_of(m, element_ty)?;
            let position = sext(read_lane(m, &item.regs, element)?, element_elem.width());
            let step_bytes = i128::from(types::size_of(&m.types, pointee, &m.decor)?);
            let addr = apply_offset(base_addr, i128::from(position) * step_bytes)?;
            let (addr, _) = walk_indices(m, &item.regs, pointee, addr, &words[4..])?;
            scratch.windows[0][0] = addr as u64;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_VARIABLE => {
            if words.len() < 3 {
                return Err(Error::InvalidBinary);
            }
            let id = words[1];
            let storage = words[2];
            let var = m
                .arena_vars
                .iter()
                .find(|var| var.ptr == id)
                .ok_or(Error::InvalidBinary)?;
            let base = match storage {
                ops::STORAGE_WORKGROUP => ctx.group_addr,
                ops::STORAGE_FUNCTION | ops::STORAGE_PRIVATE => item.arena_addr,
                _ => return Err(Error::InvalidBinary),
            };
            if base == 0 {
                return Err(Error::InvalidArgument);
            }
            let addr = base
                .checked_add(var.off as usize)
                .ok_or(Error::InvalidArgument)?;
            scratch.windows[0][0] = addr as u64;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
            if let Some(init) = var.init {
                let size = types::size_of(&m.types, var.pointee, &m.decor)?;
                check_region(ctx, item, storage, addr, size)?;
                read_value(m, &item.regs, init, &mut scratch.windows[1])?;
                // SAFETY: the same checked range as the pointer store
                // above, sized to the pointee.
                unsafe {
                    types::store_lanes(
                        &m.types,
                        var.pointee,
                        &m.decor,
                        addr as *mut u8,
                        &scratch.windows[1],
                    )?;
                }
            }
        }
        ops::OP_COPY_OBJECT => {
            let id = words
                .get(1)
                .copied()
                .ok_or(Error::InvalidBinary)?;
            let operand = words
                .get(2)
                .copied()
                .ok_or(Error::InvalidBinary)?;
            read_value(m, &item.regs, operand, &mut scratch.windows[0])?;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_COMPOSITE_CONSTRUCT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let ty = value_ty(m, id)?;
            let total = types::lanes(&m.types, ty, &m.decor)? as usize;
            if total == 0 || total > LANES {
                return Err(Error::InvalidBinary);
            }
            let mut cursor = 0usize;
            for &part in words.get(2..).ok_or(Error::InvalidBinary)? {
                let used = read_value(m, &item.regs, part, &mut scratch.windows[0][cursor..])? as usize;
                cursor = cursor
                    .checked_add(used)
                    .ok_or(Error::InvalidBinary)?;
                if cursor > total {
                    return Err(Error::InvalidBinary);
                }
            }
            if cursor != total {
                return Err(Error::InvalidBinary);
            }
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_COMPOSITE_EXTRACT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let composite = *words.get(2).ok_or(Error::InvalidBinary)?;
            let literals = words.get(3..).ok_or(Error::InvalidBinary)?;
            let (offset, leaf) = walk_lanes(m, value_ty(m, composite)?, literals)?;
            if leaf as usize > LANES {
                return Err(Error::InvalidBinary);
            }
            read_value(m, &item.regs, composite, &mut scratch.windows[1])?;
            copy_lanes(scratch, 1, offset as usize, 0, 0, leaf as usize)?;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_COMPOSITE_INSERT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let object = *words.get(2).ok_or(Error::InvalidBinary)?;
            let constituent = *words.get(3).ok_or(Error::InvalidBinary)?;
            let literals = words.get(4..).ok_or(Error::InvalidBinary)?;
            let (offset, leaf) = walk_lanes(m, value_ty(m, object)?, literals)?;
            if leaf as usize > LANES {
                return Err(Error::InvalidBinary);
            }
            read_value(m, &item.regs, object, &mut scratch.windows[0])?;
            read_value(m, &item.regs, constituent, &mut scratch.windows[1])?;
            copy_lanes(scratch, 1, 0, 0, offset as usize, leaf as usize)?;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_VECTOR_EXTRACT_DYNAMIC => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let vector = *words.get(2).ok_or(Error::InvalidBinary)?;
            let index = *words.get(3).ok_or(Error::InvalidBinary)?;
            let (elem_ty, len) = vector_shape(m, value_ty(m, vector)?)?;
            let per = types::lanes(&m.types, elem_ty, &m.decor)? as usize;
            let position = read_lane(m, &item.regs, index)?;
            if position >= u64::from(len) || per == 0 || per > LANES {
                return Err(Error::InvalidBinary);
            }
            read_value(m, &item.regs, vector, &mut scratch.windows[1])?;
            copy_lanes(scratch, 1, position as usize * per, 0, 0, per)?;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_VECTOR_INSERT_DYNAMIC => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let vector = *words.get(2).ok_or(Error::InvalidBinary)?;
            let component = *words.get(3).ok_or(Error::InvalidBinary)?;
            let index = *words.get(4).ok_or(Error::InvalidBinary)?;
            let (elem_ty, len) = vector_shape(m, value_ty(m, vector)?)?;
            let per = types::lanes(&m.types, elem_ty, &m.decor)? as usize;
            let position = read_lane(m, &item.regs, index)?;
            if position >= u64::from(len) || per == 0 || per > LANES {
                return Err(Error::InvalidBinary);
            }
            read_value(m, &item.regs, vector, &mut scratch.windows[0])?;
            read_value(m, &item.regs, component, &mut scratch.windows[1])?;
            copy_lanes(scratch, 1, 0, 0, position as usize * per, per)?;
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_VECTOR_SHUFFLE => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let first = *words.get(2).ok_or(Error::InvalidBinary)?;
            let second = *words.get(3).ok_or(Error::InvalidBinary)?;
            let (elem_ty, len) = vector_shape(m, value_ty(m, first)?)?;
            let per = types::lanes(&m.types, elem_ty, &m.decor)? as usize;
            let components = words.get(4..).ok_or(Error::InvalidBinary)?;
            let total = types::lanes(&m.types, value_ty(m, id)?, &m.decor)? as usize;
            if components.len() != total || per == 0 || per > LANES {
                return Err(Error::InvalidBinary);
            }
            read_value(m, &item.regs, first, &mut scratch.windows[1])?;
            read_value(m, &item.regs, second, &mut scratch.windows[2])?;
            let (_, second_len) = vector_shape(m, value_ty(m, second)?)?;
            if second_len != len {
                return Err(Error::InvalidBinary);
            }
            for (slot, &component) in components.iter().enumerate() {
                if component == u32::MAX {
                    scratch.windows[0][slot * per..slot * per + per].fill(0);
                    continue;
                }
                let (window, base) = if component < u32::from(len) {
                    (1usize, component as usize)
                } else {
                    let other = component
                        .checked_sub(u32::from(len))
                        .ok_or(Error::InvalidBinary)?;
                    if other >= u32::from(second_len) {
                        return Err(Error::InvalidBinary);
                    }
                    (2usize, other as usize)
                };
                let start = base
                    .checked_mul(per)
                    .ok_or(Error::InvalidBinary)?;
                copy_lanes(scratch, window, start, 0, slot * per, per)?;
            }
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_SELECT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let cond = *words.get(2).ok_or(Error::InvalidBinary)?;
            let first = *words.get(3).ok_or(Error::InvalidBinary)?;
            let second = *words.get(4).ok_or(Error::InvalidBinary)?;
            let result_ty = value_ty(m, id)?;
            let total = types::lanes(&m.types, result_ty, &m.decor)? as usize;
            if total == 0 || total > LANES {
                return Err(Error::InvalidBinary);
            }
            read_value(m, &item.regs, first, &mut scratch.windows[0])?;
            read_value(m, &item.regs, second, &mut scratch.windows[1])?;
            read_value(m, &item.regs, cond, &mut scratch.windows[2])?;
            let cond_elem = elem_of(m, value_ty(m, cond)?)?;
            let cond_lanes = m
                .values
                .get(cond as usize)
                .ok_or(Error::InvalidBinary)?
                .lanes;
            let high = high_bit(mask(cond_elem.width()));
            for lane in 0..total {
                let pick = if cond_lanes == 1 {
                    scratch.windows[2][0] != 0
                } else if cond_lanes as usize == total {
                    scratch.windows[2][lane] & high != 0
                } else {
                    return Err(Error::InvalidBinary);
                };
                scratch.windows[0][lane] = if pick {
                    scratch.windows[1][lane]
                } else {
                    scratch.windows[0][lane]
                };
            }
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_EXT_INST => {
            ext_inst(ctx, item, scratch, words)?;
        }
        ops::OP_FUNCTION_CALL => {
            next = function_call(ctx, item, scratch, words)?;
        }
        ops::OP_CONTROL_BARRIER => {
            item.park = Some(Park::Barrier);
            item.status = Status::Parked;
        }
        ops::OP_GROUP_ALL | ops::OP_GROUP_ANY => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let predicate = *words.get(3).ok_or(Error::InvalidBinary)?;
            if m.values
                .get(predicate as usize)
                .ok_or(Error::InvalidBinary)?
                .lanes
                != 1
            {
                return Err(Error::InvalidBinary);
            }
            item.park = Some(Park::Group {
                opcode,
                result: id,
                operand: predicate,
                aux: 0,
            });
            item.status = Status::Parked;
        }
        ops::OP_GROUP_BROADCAST => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(3).ok_or(Error::InvalidBinary)?;
            let index = *words.get(4).ok_or(Error::InvalidBinary)?;
            if m.values
                .get(value as usize)
                .ok_or(Error::InvalidBinary)?
                .lanes
                != 1
            {
                return Err(Error::InvalidBinary);
            }
            item.park = Some(Park::Group {
                opcode,
                result: id,
                operand: value,
                aux: index,
            });
            item.status = Status::Parked;
        }
        ops::OP_GROUP_I_ADD
        | ops::OP_GROUP_F_ADD
        | ops::OP_GROUP_F_MIN
        | ops::OP_GROUP_U_MIN
        | ops::OP_GROUP_S_MIN
        | ops::OP_GROUP_F_MAX
        | ops::OP_GROUP_U_MAX
        | ops::OP_GROUP_S_MAX => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let operation = *words.get(3).ok_or(Error::InvalidBinary)?;
            let value = *words.get(4).ok_or(Error::InvalidBinary)?;
            item.park = Some(Park::Group {
                opcode,
                result: id,
                operand: value,
                aux: operation,
            });
            item.status = Status::Parked;
        }
        ops::OP_S_NEGATE => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            unary_int(id, value, m, item, scratch, |lane, width| {
                lane.wrapping_neg() & mask(width)
            })?;
        }
        ops::OP_NOT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            unary_int(id, value, m, item, scratch, |lane, width| !lane & mask(width))?;
        }
        ops::OP_LOGICAL_NOT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            unary_int(id, value, m, item, scratch, |lane, _| u64::from(lane == 0))?;
        }
        ops::OP_F_NEGATE => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            let elem = elem_of(m, value_ty(m, id)?)?;
            let width = float_width(elem)?;
            read_value(m, &item.regs, value, &mut scratch.windows[0])?;
            let sign = 1u64 << (width - 1);
            let total = types::lanes(&m.types, value_ty(m, id)?, &m.decor)?;
            for lane in scratch.windows[0][..total as usize].iter_mut() {
                *lane ^= sign;
            }
            write_value(m, &mut item.regs, id, &scratch.windows[0])?;
        }
        ops::OP_I_ADD..=ops::OP_F_MOD | ops::OP_F_REM | ops::OP_LOGICAL_OR | ops::OP_LOGICAL_AND => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let a = *words.get(2).ok_or(Error::InvalidBinary)?;
            let b = *words.get(3).ok_or(Error::InvalidBinary)?;
            binary_lanes(opcode, id, a, b, m, item, scratch, eval_arith)?;
        }
        ops::OP_I_EQUAL..=ops::OP_F_ORD_GREATER_THAN_EQUAL => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let a = *words.get(2).ok_or(Error::InvalidBinary)?;
            let b = *words.get(3).ok_or(Error::InvalidBinary)?;
            binary_lanes(opcode, id, a, b, m, item, scratch, eval_compare)?;
        }
        ops::OP_SHIFT_RIGHT_LOGICAL
        | ops::OP_SHIFT_RIGHT_ARITHMETIC
        | ops::OP_SHIFT_LEFT_LOGICAL
        | ops::OP_BITWISE_OR
        | ops::OP_BITWISE_XOR
        | ops::OP_BITWISE_AND => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let a = *words.get(2).ok_or(Error::InvalidBinary)?;
            let b = *words.get(3).ok_or(Error::InvalidBinary)?;
            binary_lanes(opcode, id, a, b, m, item, scratch, eval_arith)?;
        }
        ops::OP_CONVERT_F_TO_U => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            convert_float(id, value, m, item, scratch, f_to_u)?;
        }
        ops::OP_CONVERT_F_TO_S => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            convert_float(id, value, m, item, scratch, f_to_s)?;
        }
        ops::OP_CONVERT_S_TO_F | ops::OP_CONVERT_U_TO_F => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            convert_to_float(opcode, id, value, m, item, scratch)?;
        }
        ops::OP_U_CONVERT | ops::OP_S_CONVERT | ops::OP_F_CONVERT => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            convert_numeric(opcode, id, value, m, item, scratch)?;
        }
        ops::OP_BITCAST => {
            let id = *words.get(1).ok_or(Error::InvalidBinary)?;
            let value = *words.get(2).ok_or(Error::InvalidBinary)?;
            bitcast(id, value, m, item, scratch)?;
        }
        _ => return Err(Error::InvalidBinary),
    }
    item.ip = next;
    Ok(())
}

/// Element type and length of a vector type.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when the id is not a vector.
fn vector_shape(m: &SpirvModule, ty: u32) -> Result<(u32, u16)> {
    match m
        .types
        .get(ty as usize)
        .ok_or(Error::InvalidBinary)?
    {
        Type::Vector { elem, len } => Ok((*elem, *len)),
        _ => Err(Error::InvalidBinary),
    }
}

/// The most significant bit of a `bits`-wide value.
#[must_use]
fn high_bit(bits: u64) -> u64 {
    if bits == u64::MAX {
        1u64 << 63
    } else {
        1u64 << (64 - bits.leading_zeros() - 1)
    }
}

/// Applies `f` to every lane of one integer unary instruction.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown result id or a lane count
/// above [`LANES`].
fn unary_int<F>(
    id: u32,
    value: u32,
    m: &SpirvModule,
    item: &mut Item,
    scratch: &mut Scratch,
    f: F,
) -> Result<()>
where
    F: Fn(u64, u16) -> u64,
{
    let ty = value_ty(m, id)?;
    let width = elem_of(m, ty)?.width();
    let total = types::lanes(&m.types, ty, &m.decor)? as usize;
    if total > LANES {
        return Err(Error::InvalidBinary);
    }
    read_value(m, &item.regs, value, &mut scratch.windows[0])?;
    for lane in scratch.windows[0][..total].iter_mut() {
        *lane = f(*lane, width);
    }
    write_value(m, &mut item.regs, id, &scratch.windows[0])
}

/// Applies `f` lane by lane to one binary instruction.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown result id, a lane count above
/// [`LANES`] or a rejection from `f`.
fn binary_lanes<F>(
    opcode: u16,
    id: u32,
    first: u32,
    second: u32,
    m: &SpirvModule,
    item: &mut Item,
    scratch: &mut Scratch,
    f: F,
) -> Result<()>
where
    F: Fn(u16, Elem, u64, u64) -> Result<u64>,
{
    // The element comes from the *operand*, not the result: arithmetic
    // shares one type across both, but a comparison's result is `%bool`
    // while its width, signedness and floatness live on the operands.
    let elem = elem_of(m, value_ty(m, first)?)?;
    let total = types::lanes(&m.types, value_ty(m, id)?, &m.decor)? as usize;
    if total > LANES {
        return Err(Error::InvalidBinary);
    }
    read_value(m, &item.regs, first, &mut scratch.windows[0])?;
    read_value(m, &item.regs, second, &mut scratch.windows[1])?;
    for lane in 0..total {
        let a = scratch.windows[0][lane];
        let b = scratch.windows[1][lane];
        scratch.windows[0][lane] = f(opcode, elem, a, b)?;
    }
    write_value(m, &mut item.regs, id, &scratch.windows[0])
}

/// Saturating float-to-integer conversion, one lane at a time.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id, a non-float operand or a
/// non-integer result.
fn convert_float<F>(
    id: u32,
    value: u32,
    m: &SpirvModule,
    item: &mut Item,
    scratch: &mut Scratch,
    f: F,
) -> Result<()>
where
    F: Fn(f64, u16) -> u64,
{
    let ty = value_ty(m, id)?;
    let width = match elem_of(m, ty)? {
        Elem::Int { width, .. } => width,
        _ => return Err(Error::InvalidBinary),
    };
    let src_width = float_width(elem_of(m, value_ty(m, value)?)?)?;
    let total = types::lanes(&m.types, ty, &m.decor)? as usize;
    if total > LANES {
        return Err(Error::InvalidBinary);
    }
    read_value(m, &item.regs, value, &mut scratch.windows[0])?;
    for lane in scratch.windows[0][..total].iter_mut() {
        *lane = f(to_f64(*lane, src_width), width);
    }
    write_value(m, &mut item.regs, id, &scratch.windows[0])
}

/// Integer-to-float conversion of one instruction.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id, a non-integer operand or
/// a non-float result.
fn convert_to_float(
    opcode: u16,
    id: u32,
    value: u32,
    m: &SpirvModule,
    item: &mut Item,
    scratch: &mut Scratch,
) -> Result<()> {
    let ty = value_ty(m, id)?;
    let width = float_width(elem_of(m, ty)?)?;
    let (src_width, signed) = match elem_of(m, value_ty(m, value)?)? {
        Elem::Int { width, signed } => (width, signed),
        _ => return Err(Error::InvalidBinary),
    };
    let total = types::lanes(&m.types, ty, &m.decor)? as usize;
    if total > LANES {
        return Err(Error::InvalidBinary);
    }
    read_value(m, &item.regs, value, &mut scratch.windows[0])?;
    for lane in scratch.windows[0][..total].iter_mut() {
        let number = if opcode == ops::OP_CONVERT_S_TO_F || signed {
            sext(*lane, src_width) as i64 as f64
        } else {
            (*lane & mask(src_width)) as f64
        };
        *lane = float_bits(width, number);
    }
    write_value(m, &mut item.regs, id, &scratch.windows[0])
}

/// Numeric conversion between integers and between floats.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id or a source/result pair
/// outside the three conversion opcodes.
fn convert_numeric(
    opcode: u16,
    id: u32,
    value: u32,
    m: &SpirvModule,
    item: &mut Item,
    scratch: &mut Scratch,
) -> Result<()> {
    let ty = value_ty(m, id)?;
    let result = elem_of(m, ty)?;
    let source = elem_of(m, value_ty(m, value)?)?;
    let total = types::lanes(&m.types, ty, &m.decor)? as usize;
    if total > LANES {
        return Err(Error::InvalidBinary);
    }
    read_value(m, &item.regs, value, &mut scratch.windows[0])?;
    match opcode {
        ops::OP_F_CONVERT => {
            let (from, to) = (float_width(source)?, float_width(result)?);
            for lane in scratch.windows[0][..total].iter_mut() {
                *lane = float_bits(to, to_f64(*lane, from));
            }
        }
        ops::OP_U_CONVERT => {
            let width = result.width();
            for lane in scratch.windows[0][..total].iter_mut() {
                *lane &= mask(source.width().min(width));
            }
        }
        ops::OP_S_CONVERT => {
            let width = result.width();
            for lane in scratch.windows[0][..total].iter_mut() {
                *lane = sext(*lane, source.width()) & mask(width);
            }
        }
        _ => return Err(Error::InvalidBinary),
    }
    write_value(m, &mut item.regs, id, &scratch.windows[0])
}

/// `OpBitcast` between two types of equal byte size: the same lane
/// pattern when both pack identically, otherwise a through-memory
/// reinterpretation.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for a size mismatch, an unknown id or a
/// lane count above [`LANES`]; [`Error::OutOfMemory`] when the staging
/// buffer cannot be reserved.
fn bitcast(id: u32, value: u32, m: &SpirvModule, item: &mut Item, scratch: &mut Scratch) -> Result<()> {
    let ty = value_ty(m, id)?;
    let source_ty = value_ty(m, value)?;
    let source_size = types::size_of(&m.types, source_ty, &m.decor)?;
    let target_size = types::size_of(&m.types, ty, &m.decor)?;
    if source_size != target_size {
        return Err(Error::InvalidBinary);
    }
    let source_lanes = types::lanes(&m.types, source_ty, &m.decor)? as usize;
    let target_lanes = types::lanes(&m.types, ty, &m.decor)? as usize;
    if source_lanes > LANES || target_lanes > LANES {
        return Err(Error::InvalidBinary);
    }
    read_value(m, &item.regs, value, &mut scratch.windows[0])?;
    if source_lanes == target_lanes {
        return write_value(m, &mut item.regs, id, &scratch.windows[0]);
    }
    let mut bytes: Vec<u8> = Vec::new();
    bytes
        .try_reserve_exact(source_size as usize)
        .map_err(|_| Error::OutOfMemory)?;
    bytes.resize(source_size as usize, 0);
    // SAFETY: `source_size` is the layout size of `source_ty`, which is
    // the number of bytes `store_lanes` writes from the lanes the load
    // above just produced.
    unsafe {
        types::store_lanes(
            &m.types,
            source_ty,
            &m.decor,
            bytes.as_mut_ptr(),
            &scratch.windows[0],
        )?;
    }
    // SAFETY: the same buffer, sized exactly to the destination's equal
    // layout size, read as the destination lanes.
    unsafe {
        types::load_lanes(&m.types, ty, &m.decor, bytes.as_ptr(), &mut scratch.windows[1])?;
    }
    write_value(m, &mut item.regs, id, &scratch.windows[1])
}

/// Executes one `OpFunctionCall`, returning the instruction index the
/// work-item continues at.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown callee, an argument-count or
/// type mismatch, an unbounded import result or a function without a
/// body; [`Error::OutOfResources`] when the call stack is exhausted.
fn function_call(ctx: &Ctx<'_>, item: &mut Item, scratch: &mut Scratch, words: &[u32]) -> Result<u32> {
    let m = ctx.m;
    let (result, callee, args) = if words.len() >= 2 && is_type(m, words[0]) {
        (
            *words.get(1).ok_or(Error::InvalidBinary)?,
            *words.get(2).ok_or(Error::InvalidBinary)?,
            words.get(3..).ok_or(Error::InvalidBinary)?,
        )
    } else {
        (
            0,
            *words.first().ok_or(Error::InvalidBinary)?,
            words.get(1..).ok_or(Error::InvalidBinary)?,
        )
    };
    let raw = m
        .func_index
        .get(callee as usize)
        .copied()
        .unwrap_or(0);
    if raw == 0 {
        return Err(Error::InvalidBinary);
    }
    let func = m
        .funcs
        .get((raw - 1) as usize)
        .ok_or(Error::InvalidBinary)?;
    if func.params.len() != args.len() {
        return Err(Error::InvalidBinary);
    }
    if let Some(name) = &func.link_name {
        let which = import_fn(name).ok_or(Error::InvalidBinary)?;
        let value = if matches!(which, ImportFn::WorkDim) {
            u64::from(ctx.dims)
        } else {
            let dim = read_lane(m, &item.regs, *args.first().ok_or(Error::InvalidBinary)?)?;
            let comps = builtin_components(import_number(which), ctx, item);
            if dim < 3 { comps[dim as usize] } else { 0 }
        };
        if result != 0 {
            let slot = m
                .values
                .get(result as usize)
                .ok_or(Error::InvalidBinary)?;
            if slot.lanes != 1 {
                return Err(Error::InvalidBinary);
            }
            scratch.windows[0][0] = value;
            write_value(m, &mut item.regs, result, &scratch.windows[0])?;
        }
        return item.ip.checked_add(1).ok_or(Error::InvalidBinary);
    }
    if !func.has_body || func.entry == 0 {
        return Err(Error::InvalidBinary);
    }
    if item.frames.len() >= m.funcs.len() {
        return Err(Error::OutOfResources);
    }
    for (index, &param) in func.params.iter().enumerate() {
        let argument = *args.get(index).ok_or(Error::InvalidBinary)?;
        let used = read_value(m, &item.regs, argument, &mut scratch.windows[0])?;
        let slot = m
            .values
            .get(param as usize)
            .ok_or(Error::InvalidBinary)?;
        if slot.lanes != used {
            return Err(Error::InvalidBinary);
        }
        write_value(m, &mut item.regs, param, &scratch.windows[0])?;
    }
    item.frames.push(Frame {
        ret_ip: item
            .ip
            .checked_add(1)
            .ok_or(Error::InvalidBinary)?,
        result,
        label: item.label,
        pred: item.pred,
    });
    item.label = 0;
    item.pred = 0;
    Ok(func.entry)
}

/// `BuiltIn` number of an `OpenCL.std` work-item query import.
#[must_use]
const fn import_number(which: ImportFn) -> u32 {
    match which {
        ImportFn::WorkDim => ops::BUILTIN_WORK_DIM,
        ImportFn::GlobalId => ops::BUILTIN_GLOBAL_INVOCATION_ID,
        ImportFn::LocalId => ops::BUILTIN_LOCAL_INVOCATION_ID,
        ImportFn::GroupId => ops::BUILTIN_WORKGROUP_ID,
        ImportFn::GlobalOffset => ops::BUILTIN_GLOBAL_OFFSET,
        ImportFn::GlobalSize => ops::BUILTIN_GLOBAL_SIZE,
        ImportFn::LocalSize => ops::BUILTIN_WORKGROUP_SIZE,
        ImportFn::NumGroups => ops::BUILTIN_NUM_WORKGROUPS,
        ImportFn::GlobalLinearId => ops::BUILTIN_GLOBAL_LINEAR_ID,
        ImportFn::EnqueuedLocalSize => ops::BUILTIN_ENQUEUED_WORKGROUP_SIZE,
    }
}

/// Folds one work-item's contribution into a group reduction accumulator.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an operation outside the group set.
fn group_fold(opcode: u16, acc: u64, value: u64, elem: Elem) -> Result<u64> {
    let width = elem.width();
    let bits = mask(width);
    Ok(if elem.is_float() {
        match (opcode, width) {
            (ops::OP_GROUP_F_ADD, 32) => {
                u64::from((f32::from_bits(acc as u32) + f32::from_bits(value as u32)).to_bits())
            }
            (ops::OP_GROUP_F_ADD, 64) => (f64::from_bits(acc) + f64::from_bits(value)).to_bits(),
            (ops::OP_GROUP_F_MIN, 32) => u64::from(
                f32::from_bits(acc as u32)
                    .min(f32::from_bits(value as u32))
                    .to_bits(),
            ),
            (ops::OP_GROUP_F_MIN, 64) => f64::from_bits(acc)
                .min(f64::from_bits(value))
                .to_bits(),
            (ops::OP_GROUP_F_MAX, 32) => u64::from(
                f32::from_bits(acc as u32)
                    .max(f32::from_bits(value as u32))
                    .to_bits(),
            ),
            (ops::OP_GROUP_F_MAX, 64) => f64::from_bits(acc)
                .max(f64::from_bits(value))
                .to_bits(),
            _ => return Err(Error::InvalidBinary),
        }
    } else {
        let (a, b) = if elem.signed() {
            (sext(acc, width) as i64, sext(value, width) as i64)
        } else {
            ((acc & bits) as i64, (value & bits) as i64)
        };
        match opcode {
            ops::OP_GROUP_I_ADD => acc.wrapping_add(value) & bits,
            ops::OP_GROUP_U_MIN | ops::OP_GROUP_S_MIN => {
                if b < a {
                    value & bits
                } else {
                    acc & bits
                }
            }
            ops::OP_GROUP_U_MAX | ops::OP_GROUP_S_MAX => {
                if b > a {
                    value & bits
                } else {
                    acc & bits
                }
            }
            ops::OP_GROUP_ALL => acc & u64::from(value != 0),
            ops::OP_GROUP_ANY => acc | u64::from(value != 0),
            _ => return Err(Error::InvalidBinary),
        }
    })
}

/// Starting accumulator of a group reduction or scan.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an operation outside the group set.
fn group_identity(opcode: u16, elem: Elem) -> Result<u64> {
    let width = elem.width();
    Ok(if elem.is_float() {
        match (opcode, width) {
            (ops::OP_GROUP_F_ADD, _) => 0,
            (ops::OP_GROUP_F_MIN, 32) => u64::from(f32::INFINITY.to_bits()),
            (ops::OP_GROUP_F_MIN, _) => f64::INFINITY.to_bits(),
            (ops::OP_GROUP_F_MAX, 32) => u64::from(f32::NEG_INFINITY.to_bits()),
            (ops::OP_GROUP_F_MAX, _) => f64::NEG_INFINITY.to_bits(),
            _ => return Err(Error::InvalidBinary),
        }
    } else {
        match opcode {
            ops::OP_GROUP_I_ADD | ops::OP_GROUP_ALL | ops::OP_GROUP_ANY => 0,
            ops::OP_GROUP_U_MIN => mask(width),
            ops::OP_GROUP_S_MIN => mask(width) >> 1,
            ops::OP_GROUP_U_MAX => 0,
            ops::OP_GROUP_S_MAX => 1u64 << 63,
            _ => return Err(Error::InvalidBinary),
        }
    })
}

/// Applies the rendezvous every parked work-item of a group is waiting
/// at, writing each reduction, scan or broadcast result back into the
/// waiting registers.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an operation outside the group set, a
/// result or operand id that is not a value, a lane count above
/// [`LANES`], or a broadcast index outside the group.
fn resolve_group(ctx: &Ctx<'_>, items: &mut [Item], park: Park) -> Result<()> {
    let m = ctx.m;
    let Park::Group {
        opcode,
        result,
        operand,
        aux,
    } = park
    else {
        return Ok(());
    };
    let slot = m
        .values
        .get(result as usize)
        .ok_or(Error::InvalidBinary)?;
    let n = slot.lanes as usize;
    if n == 0 || n > LANES {
        return Err(Error::InvalidBinary);
    }
    let result_off = slot.off as usize;
    let elem_ty = value_ty(m, operand)?;
    let elem = elem_of(m, elem_ty)?;
    match opcode {
        ops::OP_GROUP_BROADCAST => {
            let index_elem = elem_of(m, value_ty(m, aux)?)?;
            let first = items.first().ok_or(Error::InvalidBinary)?;
            let index = read_lane(m, &first.regs, aux)? & mask(index_elem.width());
            if index >= items.len() as u64 {
                return Err(Error::InvalidBinary);
            }
            let source = index as usize;
            let mut value = [0u64; LANES];
            read_value(m, &items[source].regs, operand, &mut value)?;
            for item in items.iter_mut() {
                let lanes = item
                    .regs
                    .get_mut(result_off..result_off + n)
                    .ok_or(Error::InvalidBinary)?;
                lanes.copy_from_slice(&value[..n]);
            }
            Ok(())
        }
        ops::OP_GROUP_ALL | ops::OP_GROUP_ANY => {
            let mut acc = group_identity(opcode, Elem::Bool)?;
            for item in items.iter() {
                let value = read_lane(m, &item.regs, operand)?;
                acc = group_fold(opcode, acc, value, Elem::Bool)?;
            }
            for item in items.iter_mut() {
                let lanes = item
                    .regs
                    .get_mut(result_off..result_off + n)
                    .ok_or(Error::InvalidBinary)?;
                lanes[0] = acc & 1;
            }
            Ok(())
        }
        _ => {
            let mut contributions: Vec<u64> = Vec::new();
            let count = items
                .len()
                .checked_mul(n)
                .ok_or(Error::OutOfResources)?;
            contributions
                .try_reserve_exact(count)
                .map_err(|_| Error::OutOfMemory)?;
            contributions.resize(count, 0);
            for (index, item) in items.iter().enumerate() {
                let row = contributions
                    .get_mut(index * n..index * n + n)
                    .ok_or(Error::InvalidBinary)?;
                read_value(m, &item.regs, operand, row)?;
            }
            let mut column = [0u64; LANES];
            match aux {
                ops::GROUP_OP_REDUCE => {
                    for lane in 0..n {
                        let mut acc = group_identity(opcode, elem)?;
                        for item in 0..items.len() {
                            acc = group_fold(opcode, acc, contributions[item * n + lane], elem)?;
                        }
                        column[lane] = acc;
                    }
                    for item in items.iter_mut() {
                        let lanes = item
                            .regs
                            .get_mut(result_off..result_off + n)
                            .ok_or(Error::InvalidBinary)?;
                        lanes.copy_from_slice(&column[..n]);
                    }
                }
                ops::GROUP_OP_INCLUSIVE_SCAN => {
                    let mut acc = [0u64; LANES];
                    for lane in 0..n {
                        acc[lane] = group_identity(opcode, elem)?;
                    }
                    for item in 0..items.len() {
                        for lane in 0..n {
                            acc[lane] = group_fold(opcode, acc[lane], contributions[item * n + lane], elem)?;
                            column[lane] = acc[lane];
                        }
                        let row = items
                            .get_mut(item)
                            .ok_or(Error::InvalidBinary)?
                            .regs
                            .get_mut(result_off..result_off + n)
                            .ok_or(Error::InvalidBinary)?;
                        row.copy_from_slice(&column[..n]);
                    }
                }
                ops::GROUP_OP_EXCLUSIVE_SCAN => {
                    let mut acc = [0u64; LANES];
                    for lane in 0..n {
                        acc[lane] = group_identity(opcode, elem)?;
                        column[lane] = acc[lane];
                    }
                    for item in 0..items.len() {
                        let row = items
                            .get_mut(item)
                            .ok_or(Error::InvalidBinary)?
                            .regs
                            .get_mut(result_off..result_off + n)
                            .ok_or(Error::InvalidBinary)?;
                        row.copy_from_slice(&column[..n]);
                        for lane in 0..n {
                            acc[lane] = group_fold(opcode, acc[lane], contributions[item * n + lane], elem)?;
                            column[lane] = acc[lane];
                        }
                    }
                }
                _ => return Err(Error::InvalidBinary),
            }
            Ok(())
        }
    }
}

/// Prepares `item` to run as work-item `local` / `global` of the entry
/// function: registers and arena zeroed, arena pointers published,
/// built-in inputs stored and entry parameters bound from the launch.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an entry function without a body or a
/// value id without a register slot; [`Error::InvalidArgument`] for an
/// arena access outside its region or an entry parameter wider than an
/// argument slot.
fn init_item(
    ctx: &Ctx<'_>,
    item: &mut Item,
    scratch: &mut Scratch,
    local: [u32; 3],
    global: [u32; 3],
    entry_idx: u32,
) -> Result<()> {
    let m = ctx.m;
    item.reset(local, global);
    let func = m
        .funcs
        .get(entry_idx as usize)
        .ok_or(Error::InvalidBinary)?;
    if !func.has_body || func.entry == 0 {
        return Err(Error::InvalidBinary);
    }
    item.ip = func.entry;
    for var in m.arena_vars.iter() {
        let base = match var.storage {
            ops::STORAGE_WORKGROUP => ctx.group_addr,
            ops::STORAGE_FUNCTION | ops::STORAGE_PRIVATE => item.arena_addr,
            _ => return Err(Error::InvalidBinary),
        };
        if base == 0 {
            return Err(Error::InvalidArgument);
        }
        let addr = base
            .checked_add(var.off as usize)
            .ok_or(Error::InvalidArgument)?;
        let slot = m
            .values
            .get(var.ptr as usize)
            .ok_or(Error::InvalidBinary)?;
        if slot.ty == 0 || slot.lanes != 1 {
            return Err(Error::InvalidBinary);
        }
        let lane = item
            .regs
            .get_mut(slot.off as usize)
            .ok_or(Error::InvalidBinary)?;
        *lane = addr as u64;
    }
    for var in m.input_vars.iter() {
        if item.arena_addr == 0 {
            return Err(Error::InvalidArgument);
        }
        let addr = item
            .arena_addr
            .checked_add(var.off as usize)
            .ok_or(Error::InvalidArgument)?;
        let lanes = var.lanes as usize;
        if lanes == 0 || lanes > 3 {
            return Err(Error::InvalidBinary);
        }
        let comps = builtin_components(var.number, ctx, item);
        scratch.windows[0][..lanes].copy_from_slice(&comps[..lanes]);
        // SAFETY: `var.off` was placed by the loader inside the item
        // arena, and the pointee's byte size is what the same layout
        // tables report for `lanes` components.
        unsafe {
            types::store_lanes(
                &m.types,
                var.pointee,
                &m.decor,
                addr as *mut u8,
                &scratch.windows[0][..lanes],
            )?;
        }
        let slot = m
            .values
            .get(var.ptr as usize)
            .ok_or(Error::InvalidBinary)?;
        if slot.ty == 0 || slot.lanes != 1 {
            return Err(Error::InvalidBinary);
        }
        let lane = item
            .regs
            .get_mut(slot.off as usize)
            .ok_or(Error::InvalidBinary)?;
        *lane = addr as u64;
    }
    for (index, &param) in func.params.iter().enumerate() {
        let slot = m
            .values
            .get(param as usize)
            .ok_or(Error::InvalidBinary)?;
        if slot.ty == 0 {
            return Err(Error::InvalidBinary);
        }
        let size = types::size_of(&m.types, slot.ty, &m.decor)?;
        if size == 0 || size as usize > ARG_SLOT {
            return Err(Error::InvalidArgument);
        }
        let bytes = ctx.args.slot(index);
        // SAFETY: `bytes` is a live `ARG_SLOT`-byte slice of the launch's
        // argument block and `size` is at most `ARG_SLOT`.
        unsafe {
            types::load_lanes(
                &m.types,
                slot.ty,
                &m.decor,
                bytes.as_ptr(),
                &mut scratch.windows[0],
            )?;
        }
        let count = slot.lanes as usize;
        let target = item
            .regs
            .get_mut(slot.off as usize..slot.off as usize + count)
            .ok_or(Error::InvalidBinary)?;
        target.copy_from_slice(&scratch.windows[0][..count]);
    }
    Ok(())
}

/// Runs one work-item until it parks or returns.
///
/// # Errors
///
/// Propagates the first failure of [`step`].
fn run_item(ctx: &Ctx<'_>, item: &mut Item, scratch: &mut Scratch) -> Result<()> {
    while item.status == Status::Running {
        step(ctx, item, scratch)?;
    }
    Ok(())
}

/// Global id of the work-item with local id `local` of the group `ctx`
/// describes.
///
/// # Errors
///
/// [`Error::InvalidLaunch`] when the coordinate leaves the `u32` range.
fn global_id(ctx: &Ctx<'_>, local: [u32; 3]) -> Result<[u32; 3]> {
    let mut global = [0u32; 3];
    for axis in 0..3 {
        let within = ctx.group[axis]
            .checked_mul(ctx.local[axis])
            .and_then(|base| base.checked_add(local[axis]))
            .ok_or(Error::InvalidLaunch)?;
        global[axis] = ctx.offset[axis]
            .checked_add(within)
            .ok_or(Error::InvalidLaunch)?;
    }
    Ok(global)
}

/// Runs a work-group of a module that never rendezvouses: one work-item
/// record is reused for every local id, local id `x` varying fastest.
///
/// # Errors
///
/// Propagates a work-item failure, or [`Error::InvalidBinary`] when a
/// work-item stops without returning.
fn run_sequential(ctx: &Ctx<'_>, entry_idx: u32) -> Result<()> {
    let m = ctx.m;
    let mut item = Item::new(m.slots_len as usize, m.item_arena as usize, m.funcs.len())?;
    let mut scratch = Scratch::default();
    let local = ctx.local;
    let mut z = 0u32;
    while z < local[2] {
        let mut y = 0u32;
        while y < local[1] {
            let mut x = 0u32;
            while x < local[0] {
                let local_id = [x, y, z];
                let global = global_id(ctx, local_id)?;
                init_item(ctx, &mut item, &mut scratch, local_id, global, entry_idx)?;
                run_item(ctx, &mut item, &mut scratch)?;
                if item.status != Status::Done {
                    return Err(Error::InvalidBinary);
                }
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    Ok(())
}

/// Runs a work-group of a module that uses barriers or group operations:
/// every work-item runs to its next rendezvous, the group resolves
/// together, and the pass repeats until all of them have returned.
///
/// # Errors
///
/// Propagates a work-item failure, [`Error::InvalidBinary`] when the
/// work-items park at different rendezvous (or stop without returning),
/// or [`Error::OutOfMemory`] when the work-item records cannot be
/// reserved.
fn run_collective(ctx: &Ctx<'_>, entry_idx: u32) -> Result<()> {
    let m = ctx.m;
    let local = ctx.local;
    let count = (local[0] as usize)
        .checked_mul(local[1] as usize)
        .and_then(|rows| rows.checked_mul(local[2] as usize))
        .ok_or(Error::InvalidLaunch)?;
    if count == 0 {
        return Err(Error::InvalidLaunch);
    }
    let mut items: Vec<Item> = Vec::new();
    items
        .try_reserve_exact(count)
        .map_err(|_| Error::OutOfMemory)?;
    let mut scratch = Scratch::default();
    let mut z = 0u32;
    while z < local[2] {
        let mut y = 0u32;
        while y < local[1] {
            let mut x = 0u32;
            while x < local[0] {
                let local_id = [x, y, z];
                let global = global_id(ctx, local_id)?;
                let mut item = Item::new(m.slots_len as usize, m.item_arena as usize, m.funcs.len())?;
                init_item(ctx, &mut item, &mut scratch, local_id, global, entry_idx)?;
                items.push(item);
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    loop {
        let mut pending = 0usize;
        for item in items.iter_mut() {
            if item.status == Status::Running {
                run_item(ctx, item, &mut scratch)?;
            }
            if item.status != Status::Done {
                pending += 1;
            }
        }
        if pending == 0 {
            return Ok(());
        }
        let mut rendezvous: Option<Park> = None;
        for item in items.iter() {
            match item.status {
                Status::Running => return Err(Error::InvalidBinary),
                Status::Parked => {
                    let park = item.park.ok_or(Error::InvalidBinary)?;
                    if let Some(want) = rendezvous {
                        if want != park {
                            return Err(Error::InvalidBinary);
                        }
                    } else {
                        rendezvous = Some(park);
                    }
                }
                Status::Done => return Err(Error::InvalidBinary),
            }
        }
        let park = rendezvous.ok_or(Error::InvalidBinary)?;
        resolve_group(ctx, &mut items, park)?;
        for item in items.iter_mut() {
            item.status = Status::Running;
            item.park = None;
        }
    }
}

/// Executes one work-group of a kernel launch through the SPIR-V
/// interpreter.
///
/// # Errors
///
/// * [`Error::InvalidBinary`] — a binding that is not an entry point, a
///   work-item that parks at a different rendezvous than its peers, or
///   an unsupported instruction.
/// * [`Error::InvalidLaunch`] — a dimension or group coordinate outside
///   the launch.
/// * [`Error::InvalidArgument`] — an arena access outside its region.
/// * [`Error::OutOfResources`] / [`Error::OutOfMemory`] — a step or
///   memory budget spent.
pub(crate) fn execute_group(launch: &Launch<'_>, group: [u32; 3]) -> Result<()> {
    let pointer = launch.binding.as_ptr();
    if pointer.is_null() {
        return Err(Error::InvalidBinary);
    }
    // SAFETY: `SpirvBinding` is only ever built from a program-owned
    // `SpirvEntry`, and the program outlives every kernel that can be
    // enqueued from it.
    let entry: &SpirvEntry = unsafe { &*pointer.cast::<SpirvEntry>() };
    let module = entry.module;
    if module.is_null() {
        return Err(Error::InvalidBinary);
    }
    // SAFETY: `SpirvModule::build_defs` links every entry point to its
    // owning boxed module, and the program keeps that box alive for as
    // long as any of its kernels exists.
    let m: &SpirvModule = unsafe { &*module };
    let raw = m
        .func_index
        .get(entry.func as usize)
        .copied()
        .unwrap_or(0);
    if raw == 0 {
        return Err(Error::InvalidBinary);
    }
    let entry_idx = raw - 1;
    let dims = launch.dims;
    if !(1..=3).contains(&dims) {
        return Err(Error::InvalidLaunch);
    }
    let local = launch.local_full();
    let mut global = launch.global;
    let offset = launch.offset;
    for axis in 0..3 {
        if axis >= dims as usize {
            global[axis] = 1;
        }
        if local[axis] == 0 || global[axis] == 0 || global[axis] % local[axis] != 0 {
            return Err(Error::InvalidLaunch);
        }
        let end = u64::from(offset[axis]) + u64::from(global[axis]);
        if end > u64::from(u32::MAX) {
            return Err(Error::InvalidLaunch);
        }
        if group[axis] >= global[axis] / local[axis] {
            return Err(Error::InvalidLaunch);
        }
    }
    let item_count = u64::from(local[0]) * u64::from(local[1]) * u64::from(local[2]);
    if item_count == 0 {
        return Err(Error::InvalidLaunch);
    }
    let per_item = (u64::from(m.slots_len) * 8)
        .checked_add(u64::from(m.item_arena))
        .and_then(|bytes| bytes.checked_add(core::mem::size_of::<Item>() as u64))
        .and_then(|bytes| {
            (m.funcs.len() as u64)
                .checked_mul(core::mem::size_of::<Frame>() as u64)
                .and_then(|frames| bytes.checked_add(frames))
        })
        .ok_or(Error::OutOfResources)?;
    if per_item > MAX_ITEM_BUDGET
        || item_count
            .checked_mul(per_item)
            .map_or(true, |total| total > MAX_COLLECTIVE_BUDGET)
    {
        return Err(Error::OutOfResources);
    }
    let mut group_mem: Vec<u8> = Vec::new();
    if m.group_arena > 0 {
        group_mem
            .try_reserve_exact(m.group_arena as usize)
            .map_err(|_| Error::OutOfMemory)?;
        group_mem.resize(m.group_arena as usize, 0);
    }
    let group_addr = if group_mem.is_empty() {
        0
    } else {
        group_mem.as_mut_ptr() as usize
    };
    let ctx = Ctx {
        m,
        args: launch.args,
        group,
        groups: [global[0] / local[0], global[1] / local[1], global[2] / local[2]],
        local,
        global,
        offset,
        dims,
        group_addr,
        group_len: group_mem.len(),
    };
    let outcome = if m.needs_collective {
        run_collective(&ctx, entry_idx)
    } else {
        run_sequential(&ctx, entry_idx)
    };
    // `group_mem` must outlive every pointer into it the items held.
    drop(group_mem);
    outcome
}

/// Moves `count` lanes from one scratch window to another through a
/// stack buffer, so the two windows never overlap in a borrow.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an out-of-range count, source or target.
fn copy_lanes(
    scratch: &mut Scratch,
    from: usize,
    source: usize,
    to: usize,
    target: usize,
    count: usize,
) -> Result<()> {
    if count > LANES {
        return Err(Error::InvalidBinary);
    }
    let source_end = source
        .checked_add(count)
        .ok_or(Error::InvalidBinary)?;
    let src = scratch
        .windows
        .get(from)
        .and_then(|window| window.get(source..source_end))
        .ok_or(Error::InvalidBinary)?;
    let mut buffer = [0u64; LANES];
    buffer[..count].copy_from_slice(src);
    let target_end = target
        .checked_add(count)
        .ok_or(Error::InvalidBinary)?;
    let dst = scratch
        .windows
        .get_mut(to)
        .and_then(|window| window.get_mut(target..target_end))
        .ok_or(Error::InvalidBinary)?;
    dst.copy_from_slice(&buffer[..count]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::module::load;
    use super::*;
    use crate::exec::{ArgBlock, Launch, WorkItem};

    fn bytes_of(words: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn instruction(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let mut words = Vec::with_capacity(operands.len() + 1);
        words.push(((operands.len() as u32 + 1) << 16) | u32::from(opcode));
        words.extend_from_slice(operands);
        words
    }

    fn module_words(bound: u32, instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut words = vec![super::super::decode::MAGIC, 0x0001_0000, 0, bound, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
    }

    unsafe extern "C" fn noop(_item: *mut WorkItem) {}

    /// Assembles, loads and runs one work-group of `source` with one
    /// buffer argument.
    fn run_group_of(source: &str, args: &mut ArgBlock, global: u32, local: u32) -> Result<()> {
        let parsed = codevar_ocl_ir::parse::parse(source).expect("kernel parses");
        let words = codevar_ocl_asm::assemble(&parsed).expect("kernel assembles");
        let mut module = load(&bytes_of(&words)).expect("kernel loads");
        let defs = module.build_defs().expect("defs build");
        let def = defs.first().expect("one kernel");
        let launch = Launch {
            kernel: noop,
            name: def.name.as_ref(),
            binding: def.binding,
            args,
            dims: 1,
            offset: [0; 3],
            global: [global, 1, 1],
            local: [local, 1, 1],
        };
        execute_group(&launch, [0, 0, 0]).unwrap_or_else(|e| panic!("group runs: {e:?}"));
        Ok(())
    }

    /// Loads and runs a hand-built module (one that the textual
    /// assembler cannot express) with a single buffer argument.
    fn run_raw_group(words: &[u32], args: &mut ArgBlock, global: u32, local: u32) -> Result<()> {
        let mut module = load(&bytes_of(words)).expect("module loads");
        let defs = module.build_defs().expect("defs build");
        let def = defs.first().expect("one kernel");
        let launch = Launch {
            kernel: noop,
            name: def.name.as_ref(),
            binding: def.binding,
            args,
            dims: 1,
            offset: [0; 3],
            global: [global, 1, 1],
            local: [local, 1, 1],
        };
        execute_group(&launch, [0, 0, 0])
    }

    const STORE_CONST: &str = "\
target opencl address physical64 memory opencl

OpEntryPoint Kernel %store_const \"store_const\"

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%fn_void = OpTypeFunction %void %ptr_cw_float

%seven = OpConstant %float 7.0

%store_const = OpFunction %void None %fn_void
    %out = OpFunctionParameter %ptr_cw_float
    %entry = OpLabel
    OpStore %out %seven
    OpReturn
OpFunctionEnd
";

    #[test]
    fn stores_a_constant_through_a_pointer_argument() {
        let mut buffer = [0u32; 4];
        let mut args = ArgBlock::new();
        args.set_ptr(0, buffer.as_mut_ptr().cast());
        run_group_of(STORE_CONST, &mut args, 1, 1).expect("group runs");
        assert_eq!(buffer[0], f32::to_bits(7.0));
        assert_eq!(buffer[1], 0);
    }

    const FILL_BY_ID: &str = "\
target opencl address physical64 memory opencl

OpEntryPoint Kernel %fill \"fill\"

OpDecorate %get_global_id LinkageAttributes \"get_global_id\" Import

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ptr_cw_uint = OpTypePointer CrossWorkgroup %uint
%fn_void = OpTypeFunction %void %ptr_cw_uint
%fn_uint_uint = OpTypeFunction %uint %uint

%v0 = OpConstant %uint 0

%fill = OpFunction %void None %fn_void
    %out = OpFunctionParameter %ptr_cw_uint
    %entry = OpLabel
    %gid = OpFunctionCall %uint %get_global_id %v0
    %slot = OpPtrAccessChain %ptr_cw_uint %out %gid
    OpStore %slot %gid
    OpReturn
OpFunctionEnd

%get_global_id = OpFunction %uint None %fn_uint_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd
";

    #[test]
    fn fills_an_output_buffer_with_global_ids() {
        let mut buffer = [0u32; 4];
        let mut args = ArgBlock::new();
        args.set_ptr(0, buffer.as_mut_ptr().cast());
        run_group_of(FILL_BY_ID, &mut args, 4, 4).expect("group runs");
        assert_eq!(buffer, [0, 1, 2, 3]);
    }

    const SHARED_COUNTER: &str = "\
target opencl address physical64 memory opencl

OpEntryPoint Kernel %share \"share\"

OpDecorate %get_local_id LinkageAttributes \"get_local_id\" Import

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%ptr_wg_uint = OpTypePointer Workgroup %uint
%ptr_cw_uint = OpTypePointer CrossWorkgroup %uint
%fn_void = OpTypeFunction %void %ptr_cw_uint
%fn_uint_uint = OpTypeFunction %uint %uint

%v0 = OpConstant %uint 0
%v1 = OpConstant %uint 1
%v2 = OpConstant %uint 2
%v3 = OpConstant %uint 3
%v64 = OpConstant %uint 64

%counter = OpVariable %ptr_wg_uint Workgroup

%share = OpFunction %void None %fn_void
    %out = OpFunctionParameter %ptr_cw_uint
    %entry = OpLabel
    %lid = OpFunctionCall %uint %get_local_id %v0
    %nz = OpUGreaterThan %bool %lid %v0
    OpSelectionMerge %merge None
    OpBranchConditional %nz %skip %write
    %write = OpLabel
    OpStore %counter %v3
    OpBranch %merge
    %skip = OpLabel
    OpBranch %merge
    %merge = OpLabel
    OpControlBarrier %v2 %v1 %v64
    %value = OpLoad %uint %counter
    %slot = OpPtrAccessChain %ptr_cw_uint %out %lid
    OpStore %slot %value
    OpReturn
OpFunctionEnd

%get_local_id = OpFunction %uint None %fn_uint_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd
";

    #[test]
    fn shares_a_workgroup_variable_across_a_barrier() {
        let mut buffer = [0u32; 4];
        let mut args = ArgBlock::new();
        args.set_ptr(0, buffer.as_mut_ptr().cast());
        run_group_of(SHARED_COUNTER, &mut args, 4, 4).expect("group runs");
        assert_eq!(buffer, [3, 3, 3, 3]);
    }

    const ROOTS_DIAMOND: &str = "\
target opencl address physical64 memory opencl

%OpenCL.std = OpExtInstImport \"OpenCL.std\"

OpEntryPoint Kernel %roots \"roots\"

OpDecorate %get_local_id LinkageAttributes \"get_local_id\" Import

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%float = OpTypeFloat 32
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%ptr_fn_float = OpTypePointer Function %float
%fn_void = OpTypeFunction %void %ptr_cw_float %ptr_cw_float
%fn_uint_uint = OpTypeFunction %uint %uint

%v0 = OpConstant %uint 0
%v1 = OpConstant %uint 1
%v2 = OpConstant %uint 2
%v3 = OpConstant %uint 3
%v64 = OpConstant %uint 64
%f1 = OpConstant %float 1.0
%f3 = OpConstant %float 3.0

%roots = OpFunction %void None %fn_void
    %in = OpFunctionParameter %ptr_cw_float
    %out = OpFunctionParameter %ptr_cw_float
    %entry = OpLabel
    %slot = OpVariable %ptr_fn_float Function
    %lid = OpFunctionCall %uint %get_local_id %v0
    %islot = OpPtrAccessChain %ptr_cw_float %in %lid
    %value = OpLoad %float %islot
    %root = OpExtInst %float %OpenCL.std 61 %value
    %hi = OpUGreaterThan %bool %lid %v1
    OpSelectionMerge %merge None
    OpBranchConditional %hi %high %low
    %low = OpLabel
    %lo_sum = OpFAdd %float %root %f1
    OpBranch %merge
    %high = OpLabel
    %hi_sum = OpFAdd %float %root %f3
    OpBranch %merge
    %merge = OpLabel
    %sum = OpPhi %float %lo_sum %low %hi_sum %high
    OpStore %slot %sum
    OpControlBarrier %v2 %v1 %v64
    %again = OpLoad %float %slot
    %oslot = OpPtrAccessChain %ptr_cw_float %out %lid
    OpStore %oslot %again
    OpReturn
OpFunctionEnd

%get_local_id = OpFunction %uint None %fn_uint_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd
";

    #[test]
    fn merges_a_diamond_through_a_phi_and_a_local_variable() {
        let mut input = [4.0f32; 4];
        let mut output = [0.0f32; 4];
        let mut args = ArgBlock::new();
        args.set_ptr(0, input.as_mut_ptr().cast());
        args.set_ptr(1, output.as_mut_ptr().cast());
        let parsed = codevar_ocl_ir::parse::parse(ROOTS_DIAMOND).expect("kernel parses");
        let words = codevar_ocl_asm::assemble(&parsed).expect("kernel assembles");
        let mut module = load(&bytes_of(&words)).expect("kernel loads");
        let defs = module.build_defs().expect("defs build");
        let def = defs.first().expect("one kernel");
        let launch = Launch {
            kernel: noop,
            name: def.name.as_ref(),
            binding: def.binding,
            args: &args,
            dims: 1,
            offset: [0; 3],
            global: [4, 1, 1],
            local: [4, 1, 1],
        };
        execute_group(&launch, [0, 0, 0]).expect("group runs");
        assert_eq!(input, [4.0; 4]);
        assert_eq!(output, [3.0, 3.0, 5.0, 5.0]);
    }

    /// A kernel the textual assembler cannot express: an `OpGroupIAdd`
    /// reduction over the work-group's global ids, summed back into the
    /// output buffer. Ids: 1 void, 2 uint, 3 pointer, 4/5 function
    /// types, 6/7 constants, 8 the imported `get_global_id`, 9 the
    /// kernel, 10 its parameter, 11 the entry label, 12 the id, 13 the
    /// sum, 14 the slot, 15 the import parameter.
    fn group_add_words() -> Vec<u32> {
        let instructions = vec![
            instruction(
                super::super::ops::OP_CAPABILITY,
                &[super::super::ops::CAPABILITY_KERNEL],
            ),
            instruction(
                super::super::ops::OP_MEMORY_MODEL,
                &[
                    super::super::ops::ADDRESSING_MODEL_PHYSICAL64,
                    super::super::ops::MEMORY_MODEL_OPENCL,
                ],
            ),
            instruction(
                super::super::ops::OP_ENTRY_POINT,
                &[
                    super::super::ops::EXECUTION_MODEL_KERNEL,
                    9,
                    0x7564_6572,
                    0x0000_6563,
                ],
            ),
            instruction(
                super::super::ops::OP_DECORATE,
                &[
                    8,
                    super::super::ops::DECORATION_LINKAGE_ATTRIBUTES,
                    0x5f74_6567,
                    0x626f_6c67,
                    0x695f_6c61,
                    0x0000_0064,
                    super::super::ops::LINKAGE_IMPORT,
                ],
            ),
            instruction(super::super::ops::OP_TYPE_VOID, &[1]),
            instruction(super::super::ops::OP_TYPE_INT, &[2, 32, 0]),
            instruction(super::super::ops::OP_TYPE_POINTER, &[3, 5, 2]),
            instruction(super::super::ops::OP_TYPE_FUNCTION, &[4, 1, 3]),
            instruction(super::super::ops::OP_TYPE_FUNCTION, &[5, 2, 2]),
            instruction(super::super::ops::OP_CONSTANT, &[2, 6, 0]),
            instruction(super::super::ops::OP_CONSTANT, &[2, 7, 2]),
            instruction(super::super::ops::OP_FUNCTION, &[2, 8, 0, 5]),
            instruction(super::super::ops::OP_FUNCTION_PARAMETER, &[2, 15]),
            instruction(super::super::ops::OP_FUNCTION_END, &[]),
            instruction(super::super::ops::OP_FUNCTION, &[1, 9, 0, 4]),
            instruction(super::super::ops::OP_FUNCTION_PARAMETER, &[3, 10]),
            instruction(super::super::ops::OP_LABEL, &[11]),
            instruction(super::super::ops::OP_FUNCTION_CALL, &[2, 12, 8, 6]),
            instruction(super::super::ops::OP_GROUP_I_ADD, &[2, 13, 7, 0, 12]),
            instruction(super::super::ops::OP_PTR_ACCESS_CHAIN, &[3, 14, 10, 12]),
            instruction(super::super::ops::OP_STORE, &[14, 13]),
            instruction(super::super::ops::OP_RETURN, &[]),
            instruction(super::super::ops::OP_FUNCTION_END, &[]),
        ];
        module_words(16, &instructions)
    }

    #[test]
    fn reduces_a_group_sum() {
        let mut buffer = [0u32; 4];
        let mut args = ArgBlock::new();
        args.set_ptr(0, buffer.as_mut_ptr().cast());
        run_raw_group(&group_add_words(), &mut args, 4, 4).expect("group runs");
        assert_eq!(buffer, [6, 6, 6, 6]);
    }
}
