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
use crate::exec::{ArgBlock, Launch, ARG_SLOT};

use super::builtin::{ext_inst_arity, import_fn, ImportFn};
use super::module::{SpirvEntry, SpirvModule};
use super::ops;
use super::types::{self, Type, MAX_VALUE_LANES};

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
        regs.try_reserve_exact(slots).map_err(|_| Error::OutOfMemory)?;
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
    let entry = m.types.get(ty as usize).ok_or(Error::InvalidBinary)?;
    Ok(match entry {
        Type::Bool => Elem::Bool,
        Type::Int { width, signed } => Elem::Int {
            width: *width,
            signed: *signed,
        },
        Type::Float { width } => Elem::Float { width: *width },
        Type::Pointer { .. } => Elem::Ptr,
        Type::Vector { elem, .. } => match m.types.get(*elem as usize).ok_or(Error::InvalidBinary)? {
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
    let slot = m.values.get(id as usize).ok_or(Error::InvalidBinary)?;
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
    let slot = m.values.get(id as usize).ok_or(Error::InvalidBinary)?;
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
    let consts = m.consts.get(id as usize).ok_or(Error::InvalidBinary)?;
    if consts.count > 0 {
        if consts.count != count {
            return Err(Error::InvalidBinary);
        }
        let start = consts.off as usize;
        let end = start.checked_add(count as usize).ok_or(Error::InvalidBinary)?;
        let src = m.const_pool.get(start..end).ok_or(Error::InvalidBinary)?;
        out[..count as usize].copy_from_slice(src);
        return Ok(count);
    }
    let start = slot.off as usize;
    let end = start.checked_add(count as usize).ok_or(Error::InvalidBinary)?;
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
    let slot = m.values.get(id as usize).ok_or(Error::InvalidBinary)?;
    if slot.ty == 0 {
        return Err(Error::InvalidBinary);
    }
    if slot.lanes != 1 {
        return Err(Error::InvalidBinary);
    }
    let consts = m.consts.get(id as usize).ok_or(Error::InvalidBinary)?;
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
    let slot = m.values.get(id as usize).ok_or(Error::InvalidBinary)?;
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
    let end = start.checked_add(count).ok_or(Error::InvalidBinary)?;
    let dst = regs.get_mut(start..end).ok_or(Error::InvalidBinary)?;
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
        ops::STORAGE_FUNCTION | ops::STORAGE_PRIVATE | ops::STORAGE_INPUT => Some((
            item.arena_addr,
            item.arena_addr.saturating_add(item.arena.len()),
        )),
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
fn check_region(
    ctx: &Ctx<'_>,
    item: &Item,
    storage: u32,
    addr: usize,
    size: u32,
) -> Result<()> {
    let Some((start, end)) = arena_bounds(ctx, item, storage) else {
        return Ok(());
    };
    let after = addr.checked_add(size as usize).ok_or(Error::InvalidArgument)?;
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
        match m.types.get(ty as usize).ok_or(Error::InvalidBinary)? {
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
        let product =
            sext(a & LOW_24, 24) as i64 as i128 * sext(b & LOW_24, 24) as i64 as i128;
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
            (((x << width) | y) & wide)
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
    if x < edge {
        0.0
    } else {
        1.0
    }
}
#[inline]
fn step64(edge: f64, x: f64) -> f64 {
    if x < edge {
        0.0
    } else {
        1.0
    }
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
    if x < 0.0 {
        f32::NAN
    } else {
        libm::powf(x, y)
    }
}
#[inline]
fn powr64(x: f64, y: f64) -> f64 {
    if x < 0.0 {
        f64::NAN
    } else {
        libm::pow(x, y)
    }
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
    if x < y {
        y
    } else {
        x
    }
}
#[inline]
fn fmax_common64(x: f64, y: f64) -> f64 {
    if x < y {
        y
    } else {
        x
    }
}
#[inline]
fn fmin_common32(x: f32, y: f32) -> f32 {
    if y < x {
        y
    } else {
        x
    }
}
#[inline]
fn fmin_common64(x: f64, y: f64) -> f64 {
    if y < x {
        y
    } else {
        x
    }
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
    OCL_LOGB => libm::logbf, libm::logb;
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
        (ops::OCL_FMA, 64) => libm::fma(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c))
            .to_bits(),
        (ops::OCL_MAD, 32) => u64::from(
            (f32::from_bits(a as u32) * f32::from_bits(b as u32)
                + f32::from_bits(c as u32))
            .to_bits(),
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
            smooth_step64(
                f64::from_bits(a),
                f64::from_bits(b),
                f64::from_bits(c),
            )
            .to_bits()
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
        if n < 0 {
            1.0 / root
        } else {
            root
        }
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
    let lanes = m.values.get(id as usize).map_or(0, |value| value.lanes);
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
            require_same(m, arg_ty[0], ty)?;
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
            let sum = norm_sum(&scratch.windows[0], value_lanes(m, arg_ty[0])?)?;
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
            require_same(m, arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            let count = value_lanes(m, arg_ty[0])?;
            if count != n {
                return Err(Error::InvalidBinary);
            }
            let sum = norm_sum(&scratch.windows[0], count)?;
            let len = libm::sqrt(sum);
            let out = &mut scratch.windows[0];
            for lane in 0..n {
                out[lane] = float_bits(width, to_f64(out[lane], width) / len);
            }
        }
        ops::OCL_CROSS => {
            require_same(m, arg_ty[0], ty)?;
            require_same(m, arg_ty[1], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            let count = value_lanes(m, arg_ty[0])?;
            if count < 3 || n < 3 {
                return Err(Error::InvalidBinary);
            }
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut cross = [0.0f64; 3];
            cross[0] = to_f64(a[1], width) * to_f64(b[2], width)
                - to_f64(a[2], width) * to_f64(b[1], width);
            cross[1] = to_f64(a[2], width) * to_f64(b[0], width)
                - to_f64(a[0], width) * to_f64(b[2], width);
            cross[2] = to_f64(a[0], width) * to_f64(b[1], width)
                - to_f64(a[1], width) * to_f64(b[0], width);
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
            require_same(m, arg_ty[0], ty)?;
            require_same(m, arg_ty[1], ty)?;
            let cond_ty = arg_ty[2];
            let cond_elem = elem_of(m, cond_ty)?;
            let cond_lanes = value_lanes(m, cond_ty)?;
            let result_elem = elem_of(m, ty)?;
            let result_lanes = value_lanes(m, ty)?;
            if result_lanes != n {
                return Err(Error::InvalidBinary);
            }
            let mask = match cond_elem {
                Elem::Int { width, .. } | Elem::Bool => mask(cond_elem.width()),
                Elem::Float { .. } | Elem::Ptr => return Err(Error::InvalidBinary),
            };
            let high = if mask == u64::MAX {
                1u64 << 63
            } else {
                1u64 << (64 - mask.leading_zeros() - 1)
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
            let _ = result_elem;
        }
        ops::OCL_BITSELECT => {
            require_same(m, arg_ty[0], ty)?;
            require_same(m, arg_ty[1], ty)?;
            require_same(m, arg_ty[2], ty)?;
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
            require_same(m, arg_ty[0], ty)?;
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
            require_same(m, arg_ty[0], ty)?;
            let width = float_width(elem_of(m, ty)?)?;
            require_int(elem_of(m, arg_ty[1])?)?;
            let a = &scratch.windows[0];
            let b = &scratch.windows[1];
            let mut out = [0u64; LANES];
            for lane in 0..n {
                let y = b[lane] as i32;
                out[lane] = match width {
                    32 => u64::from(
                        libm::powf(f32::from_bits(a[lane] as u32), y as f32).to_bits(),
                    ),
                    64 => libm::pow(f64::from_bits(a[lane]), f64::from(y)).to_bits(),
                    _ => return Err(Error::InvalidBinary),
                };
            }
            scratch.windows[0][..n].copy_from_slice(&out[..n]);
        }
        ops::OCL_ROOTN => {
            require_same(m, arg_ty[0], ty)?;
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
            let out = &mut scratch.windows[0];
            match elem {
                Elem::Float { .. } => {
                    for lane in 0..n {
                        let a = out[lane];
                        let b = scratch.windows[1][lane];
                        let c = scratch.windows[2][lane];
                        out[lane] = match arity {
                            1 => eval_float1(inst, a, width)?,
                            2 => eval_float2(inst, a, b, width)?,
                            3 => eval_float3(inst, a, b, c, width)?,
                            _ => return Err(Error::InvalidBinary),
                        };
                    }
                }
                Elem::Int { width, .. } => {
                    for lane in 0..n {
                        let a = out[lane];
                        let b = scratch.windows[1][lane];
                        let c = scratch.windows[2][lane];
                        out[lane] = match arity {
                            1 => eval_int1(inst, a, width)?,
                            2 => eval_int2(inst, a, b, width)?,
                            3 => eval_int3(inst, a, b, c, width)?,
                            _ => return Err(Error::InvalidBinary),
                        };
                    }
                }
                Elem::Bool | Elem::Ptr => return Err(Error::InvalidBinary),
            }
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
fn require_same(m: &SpirvModule, given: u32, expected: u32) -> Result<()> {
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
fn norm_sum(lanes: &[u64; LANES], count: u32) -> Result<f64> {
    let count = count as usize;
    if count == 0 || count > LANES {
        return Err(Error::InvalidBinary);
    }
    let mut sum = 0.0f64;
    for lane in lanes[..count].iter() {
        let value = f64::from(f32::from_bits(*lane as u32));
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
