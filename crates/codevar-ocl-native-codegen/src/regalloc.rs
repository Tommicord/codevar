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

//! Register allocation.
//!
//! The allocator gives every virtual register of the machine code a
//! home: a register of the target's file, or a slot in the function's
//! frame when the file is too small to hold everything that is live at
//! once.
//!
//! The allocation is computed from the live intervals of the virtual
//! registers: an interval runs from the definition of the register to
//! its last use.  Two intervals that overlap cannot share a register,
//! so the allocator walks the intervals in order of their start and
//! gives each one the first register of the right class that is still
//! free, spilling the interval that is cheapest to spill when no
//! register is left.
//!
//! A spilled value is saved before its register is reused and read
//! back where it is used: the definition stays where it is, a
//! [`MOp::SpillStore`](crate::mir::MOp::SpillStore) saves the value
//! in the frame, and every later use becomes a
//! [`MOp::SpillLoad`](crate::mir::MOp::SpillLoad) into a register of
//! its own.  The back ends therefore never see a value that lives in
//! memory only.
//!
//! # The destination of an instruction
//!
//! The allocator rewrites every operand into a physical register.  The
//! destination of an instruction is a virtual register, and the
//! machine IR has no field that carries the register the allocator
//! chose for it; a back end needs that field to know where an
//! instruction writes its result.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{CodegenError, Result};
use crate::mir::{
    Destination, MInst, MOp, MType, MirFunction, MirModule, Operand, Operands, PReg, RegClass, VReg,
};
use crate::target::RegModel;

/// The alignment every spill slot is given.
const SPILL_ALIGNMENT: u32 = 8;

/// The alignment the frame is padded to.
const FRAME_ALIGNMENT: u32 = 16;

/// The position that means "nowhere".
const NOWHERE: u32 = u32::MAX;

/// What the allocator did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AllocStats {
    /// The number of registers the allocator assigned.
    pub registers: u32,
    /// The number of registers the allocator spilled.
    pub spills: u32,
    /// The number of instructions the allocator added.
    pub rewrites: u32,
}

/// Assigns a register to every virtual register of `module`.
///
/// The registers of `registers` are the ones the allocator may use.
/// The architecture's argument registers are never handed out: they
/// belong to the calling convention, and a back end moves a value
/// into one of them only right before a call.
///
/// # Errors
///
/// Returns [`CodegenError::Allocation`](crate::error::CodegenError::Allocation)
/// when the machine code is malformed, for instance when a virtual
/// register is used but never defined, or when the target has no
/// register of the class a value needs.
pub fn allocate(module: &mut MirModule, registers: &RegModel) -> Result<AllocStats> {
    let mut allocator = Allocator::new(registers);
    let mut stats = AllocStats::default();
    for function in &mut module.functions {
        let one = allocator.allocate_function(function)?;
        stats.registers = stats.registers.saturating_add(one.registers);
        stats.spills = stats.spills.saturating_add(one.spills);
        stats.rewrites = stats.rewrites.saturating_add(one.rewrites);
    }
    Ok(stats)
}

/// Builds an allocation error.
fn allocation_error(what: &'static str, detail: impl Into<alloc::string::String>) -> CodegenError {
    CodegenError::Allocation {
        what,
        detail: detail.into(),
    }
}

/// The index of a register class in the allocator's tables.
const fn class_index(class: RegClass) -> usize {
    match class {
        RegClass::Integer => 0,
        RegClass::Float => 1,
        RegClass::Vector => 2,
    }
}

/// An interval that holds a register.
#[derive(Debug, Clone, Copy)]
struct Active {
    /// The value that holds the register.
    value: VReg,
    /// The register.
    register: PReg,
    /// The position the value is live until.
    end: u32,
    /// Whether the interval belongs to a reloaded value, which is
    /// live for one position only and is never spilled.
    transient: bool,
}

/// The reload of a spilled value.
#[derive(Debug, Clone)]
struct Reload {
    /// The type of the value.
    ty: MType,
    /// The class of the value.
    class: RegClass,
    /// The slot the value was saved in.
    offset: u32,
    /// The size of the value.
    size: u32,
    /// The register the reloaded value gets.
    register: Option<PReg>,
}

/// An item that needs a register at the current position.
#[derive(Debug, Clone, Copy)]
struct Item {
    /// The class of the register.
    class: RegClass,
    /// The reload that needs the register, when the item is one.
    reload: Option<usize>,
    /// The value that needs the register, when the item is one.
    value: Option<VReg>,
}

/// The allocator, with the buffers it reuses between functions.
struct Allocator<'a> {
    /// The register file of the target.
    registers: &'a RegModel,
    /// The registers of every class, without the argument registers.
    ///
    /// A target whose vectors live in the same file as its floats
    /// lists the same registers in two classes, so the pools overlap.
    pools: [Vec<PReg>; 3],
    /// The free registers of every class.
    ///
    /// A register that belongs to two classes is in the free list of
    /// both until it is taken: taking it removes it from every list
    /// it belongs to, and freeing it returns it to every list, so a
    /// register is never handed out to two classes at the same time.
    free: [Vec<PReg>; 3],
    /// The intervals that hold a register.
    active: Vec<Active>,
    /// The position every virtual register is defined at.
    def_pos: Vec<u32>,
    /// The position every virtual register is live until.
    last_use: Vec<u32>,
    /// The positions every virtual register is used at.
    uses: Vec<Vec<u32>>,
    /// The register of every virtual register.
    assigned: Vec<Option<PReg>>,
    /// The position every virtual register was spilled at.
    spill_point: Vec<u32>,
    /// Whether every virtual register was spilled.
    spilled: Vec<bool>,
    /// The slot every spilled virtual register was saved in.
    slot: Vec<u32>,
    /// The reloads of the spilled values.
    reloads: Vec<Reload>,
    /// The reloads waiting at every position.
    pending: BTreeMap<u32, Vec<usize>>,
    /// The values saved before the instruction of every position.
    stored_before: BTreeMap<u32, Vec<VReg>>,
    /// The values saved after the instruction of every position.
    stored_after: BTreeMap<u32, Vec<VReg>>,
    /// The reload of a use, by the position and the value.
    reload_of: BTreeMap<(u32, usize), usize>,
    /// The values that start their interval at every position.
    starting: BTreeMap<u32, Vec<VReg>>,
    /// The type of every virtual register.
    types: Vec<MType>,
    /// The next free offset of the frame.
    next_slot: u32,
}

impl<'a> Allocator<'a> {
    /// Creates an allocator over the register file of `registers`.
    fn new(registers: &'a RegModel) -> Self {
        let mut pools: [Vec<PReg>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for class in [RegClass::Integer, RegClass::Float, RegClass::Vector] {
            let index = class_index(class);
            for &reg in registers.regs_for(class) {
                if registers.arg_regs.contains(&reg) {
                    continue;
                }
                if !pools[index].contains(&reg) {
                    pools[index].push(reg);
                }
            }
        }
        Self {
            registers,
            pools,
            free: [Vec::new(), Vec::new(), Vec::new()],
            active: Vec::new(),
            def_pos: Vec::new(),
            last_use: Vec::new(),
            uses: Vec::new(),
            assigned: Vec::new(),
            spill_point: Vec::new(),
            spilled: Vec::new(),
            slot: Vec::new(),
            reloads: Vec::new(),
            pending: BTreeMap::new(),
            stored_before: BTreeMap::new(),
            stored_after: BTreeMap::new(),
            reload_of: BTreeMap::new(),
            starting: BTreeMap::new(),
            types: Vec::new(),
            next_slot: 0,
        }
    }

    /// Allocates the registers of one function.
    fn allocate_function(&mut self, function: &mut MirFunction) -> Result<AllocStats> {
        self.reset(function);
        fuse_compare_branches(function);
        // Coalescing runs before the two-address reuse: the reuse writes
        // an operation into its first operand, which gives that operand a
        // second definition, and a copy can no longer be forwarded into a
        // register with several definitions.  Forwarding the copies first
        // lets the reuse see the operation's real operand -- otherwise the
        // first operand is often the destination of an unforwarded copy
        // and the copy survives as a move.
        self.coalesce_copies(function);
        reuse_two_address_destinations(function);
        // The reuse rewrites the destination of the operation, which can
        // leave copies that are now single-use, so one more round of
        // coalescing cleans them up.
        self.coalesce_copies(function);
        self.gather(function)?;
        self.plan(function)?;
        let added = self.rewrite(function)?;
        function.frame_size = align_up(function.fixed_frame_size.max(self.next_slot), FRAME_ALIGNMENT);
        let registers = self
            .assigned
            .iter()
            .filter(|reg| reg.is_some())
            .count();
        let spills = self.spilled.iter().filter(|flag| **flag).count();
        Ok(AllocStats {
            registers: u32::try_from(registers).unwrap_or(u32::MAX),
            spills: u32::try_from(spills).unwrap_or(u32::MAX),
            rewrites: added,
        })
    }

    /// Coalesces the copies the selection stage left behind.
    ///
    /// The machine IR is in SSA form, so a virtual register has exactly
    /// one definition, and a copy `dest = copy src` is often a move the
    /// selection stage made for its own convenience rather than for the
    /// hardware.  The pass forwards such a copy, replacing every use of
    /// its destination with `src` and dropping the copy, when both ends
    /// have the same type and one of two conditions holds:
    ///
    /// * The destination is used exactly once.  The one use reads `src`
    ///   directly, whether or not the forwarding extends the live range
    ///   of `src`.
    /// * The source already outlives the destination, that is
    ///   `last_use(src) >= last_use(dest)`.  Every use of `dest` sits
    ///   between the copy and its last use, and `src` is already live
    ///   over that whole range, so rewriting the uses of `dest` to read
    ///   `src` does not extend the live interval of `src` at all: the
    ///   coalescing is free and cannot increase register pressure.
    ///
    /// A destination that is never used drops the copy as well, since a
    /// copy has no side effect to preserve.  A copy whose source is an
    /// immediate or a physical register is left alone, because there is
    /// no register to forward to, and so is a copy whose destination is
    /// used more than once while its source is not already live long
    /// enough, because forwarding it would extend the live range of
    /// `src` across uses that may not be able to share a register.
    ///
    /// Only a destination with a single definition is forwarded.  The
    /// machine code is not in static single assignment form — a phi is
    /// lowered to one copy in every predecessor block — so a register
    /// written by several copies cannot be renamed without changing the
    /// value its uses read.
    ///
    /// The pass runs to a fixpoint and recomputes the use counts and the
    /// last uses on every round, so deleting a copy can expose the copy
    /// that defined the forwarded value as single-use in turn: a chain
    /// of copies collapses in one call.  Every round removes exactly one
    /// copy, so the loop terminates.
    fn coalesce_copies(&mut self, function: &mut MirFunction) {
        let count = function.vreg_types.len();
        loop {
            let mut uses: Vec<u32> = vec![0; count];
            let mut defs: Vec<u32> = vec![0; count];
            let mut last_use: Vec<u32> = vec![NOWHERE; count];
            let mut position = 0u32;
            for block in &function.blocks {
                for inst in &block.insts {
                    if let Some(dest) = inst.dest_vreg() {
                        if let Some(slot) = defs.get_mut(dest.index()) {
                            *slot = slot.saturating_add(1);
                        }
                        if let Some(entry) = last_use.get_mut(dest.index())
                            && *entry == NOWHERE
                        {
                            *entry = position;
                        }
                    }
                    for operand in inst.args.as_slice() {
                        let Some(value) = operand.virtual_reg() else {
                            continue;
                        };
                        if let Some(slot) = uses.get_mut(value.index()) {
                            *slot = slot.saturating_add(1);
                        }
                        if let Some(entry) = last_use.get_mut(value.index()) {
                            *entry = position;
                        }
                    }
                    position = position.saturating_add(1);
                }
            }
            let mut chosen: Option<(usize, usize, VReg, VReg)> = None;
            for (block_index, block) in function.blocks.iter().enumerate() {
                for (inst_index, inst) in block.insts.iter().enumerate() {
                    if !matches!(inst.op, MOp::Copy) {
                        continue;
                    }
                    let Some(dest) = inst.dest_vreg() else {
                        continue;
                    };
                    let dest_uses = uses.get(dest.index()).copied().unwrap_or(0);
                    if dest_uses == 0 {
                        chosen = Some((block_index, inst_index, dest, dest));
                        break;
                    }
                    // The machine code is not in static single assignment
                    // form: a phi is lowered to a copy in every
                    // predecessor block, so one register can be written
                    // by several instructions.  Forwarding a copy into
                    // the uses of such a register would let a use read a
                    // value written by a different definition, so only a
                    // register with a single definition may be
                    // forwarded.
                    if defs.get(dest.index()).copied().unwrap_or(0) != 1 {
                        continue;
                    }
                    let Some(Operand::Virtual(src)) = inst.arg0() else {
                        continue;
                    };
                    if self.type_of(dest) != self.type_of(src) {
                        continue;
                    }
                    if dest_uses != 1 {
                        // Forwarding a multiply-used destination only
                        // stays free when `src` is already live at
                        // least as long as `dest`.
                        let Some(&src_last) = last_use.get(src.index()) else {
                            continue;
                        };
                        let Some(&dest_last) = last_use.get(dest.index()) else {
                            continue;
                        };
                        if src_last == NOWHERE || dest_last == NOWHERE || src_last < dest_last {
                            continue;
                        }
                    }
                    chosen = Some((block_index, inst_index, dest, src));
                    break;
                }
                if chosen.is_some() {
                    break;
                }
            }
            let Some((block_index, inst_index, dest, src)) = chosen else {
                return;
            };
            if dest != src {
                rewrite_uses(function, dest, src);
            }
            function.blocks[block_index]
                .insts
                .remove(inst_index);
        }
    }

    /// Empties the allocator's buffers and prepares them for a
    /// function with `count` virtual registers.
    fn reset(&mut self, function: &MirFunction) {
        let count = function.vreg_types.len();
        self.types.clear();
        self.types
            .extend(function.vreg_types.iter().cloned());
        self.free[0].clear();
        self.free[1].clear();
        self.free[2].clear();
        for index in 0..3 {
            self.free[index].extend(self.pools[index].iter().copied());
        }
        self.active.clear();
        self.def_pos.clear();
        self.def_pos.resize(count, NOWHERE);
        self.last_use.clear();
        self.last_use.resize(count, NOWHERE);
        self.uses.clear();
        self.uses.resize_with(count, Vec::new);
        for uses in &mut self.uses {
            uses.clear();
        }
        self.assigned.clear();
        self.assigned.resize(count, None);
        self.spill_point.clear();
        self.spill_point.resize(count, NOWHERE);
        self.spilled.clear();
        self.spilled.resize(count, false);
        self.slot.clear();
        self.slot.resize(count, 0);
        self.reloads.clear();
        self.pending.clear();
        self.stored_before.clear();
        self.stored_after.clear();
        self.reload_of.clear();
        self.starting.clear();
        self.next_slot = function.fixed_frame_size;
    }

    /// Collects the definitions and the uses of every virtual register.
    fn gather(&mut self, function: &MirFunction) -> Result<()> {
        let mut position = 0u32;
        for block in &function.blocks {
            for inst in &block.insts {
                if let Some(dest) = inst.dest_vreg() {
                    let index = dest.index();
                    if index >= self.def_pos.len() {
                        return Err(allocation_error(
                            "collecting the definitions",
                            "an instruction writes a register the function does not declare",
                        ));
                    }
                    if self.def_pos[index] > position {
                        self.def_pos[index] = position;
                    }
                    if self.last_use[index] == NOWHERE {
                        self.last_use[index] = position;
                    }
                }
                for operand in inst.args.as_slice() {
                    let Some(value) = operand.virtual_reg() else {
                        continue;
                    };
                    let index = value.index();
                    if index >= self.uses.len() {
                        return Err(allocation_error(
                            "collecting the uses",
                            "an instruction reads a register the function does not declare",
                        ));
                    }
                    self.uses[index].push(position);
                    if self.last_use[index] < position {
                        self.last_use[index] = position;
                    }
                }
                position = position.saturating_add(1);
            }
        }
        for index in 0..self.uses.len() {
            if self.def_pos[index] == NOWHERE && !self.uses[index].is_empty() {
                return Err(allocation_error(
                    "collecting the definitions",
                    "a virtual register is used but never defined",
                ));
            }
        }
        Ok(())
    }

    /// Decides the register of every virtual register.
    fn plan(&mut self, function: &MirFunction) -> Result<()> {
        for index in 0..self.uses.len() {
            if self.def_pos[index] == NOWHERE {
                continue;
            }
            let mut start = self.def_pos[index];
            for &use_pos in &self.uses[index] {
                if use_pos < start {
                    start = use_pos;
                }
            }
            self.starting
                .entry(start)
                .or_default()
                .push(VReg::new(u32::try_from(index).unwrap_or(u32::MAX)));
        }
        let mut position = 0u32;
        for block in &function.blocks {
            for _ in &block.insts {
                self.plan_position(position)?;
                position = position.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Decides the registers one position needs.
    fn plan_position(&mut self, position: u32) -> Result<()> {
        self.expire(position);
        let mut queue: Vec<Item> = Vec::new();
        if let Some(values) = self.starting.get(&position) {
            for &value in values {
                queue.push(Item {
                    class: self.class_of(value),
                    reload: None,
                    value: Some(value),
                });
            }
        }
        if let Some(ids) = self.pending.get(&position) {
            for &id in ids {
                let reload = &self.reloads[id];
                queue.push(Item {
                    class: reload.class,
                    reload: Some(id),
                    value: None,
                });
            }
        }
        let mut cursor = 0usize;
        let mut spills = 0usize;
        let limit = self.uses.len().saturating_add(1);
        while cursor < queue.len() {
            let item = queue[cursor];
            cursor += 1;
            let Some(register) = self.take_register(item.class) else {
                let Some(victim) = self.pick_victim(item.class) else {
                    return Err(allocation_error(
                        "assigning a register",
                        "the target has no register of the class the value needs",
                    ));
                };
                spills += 1;
                if spills > limit {
                    return Err(allocation_error(
                        "assigning a register",
                        "the register file is too small to hold the values that are live",
                    ));
                }
                self.spill(victim, position, &mut queue)?;
                let Some(register) = self.take_register(item.class) else {
                    continue;
                };
                self.assign(item, register, position);
                continue;
            };
            self.assign(item, register, position);
        }
        Ok(())
    }

    /// Records the register of an item.
    fn assign(&mut self, item: Item, register: PReg, position: u32) {
        if let Some(id) = item.reload {
            if let Some(reload) = self.reloads.get_mut(id) {
                reload.register = Some(register);
            }
            self.active.push(Active {
                value: VReg::new(0),
                register,
                end: position,
                transient: true,
            });
            return;
        }
        let Some(value) = item.value else {
            return;
        };
        let index = value.index();
        self.assigned[index] = Some(register);
        let end = self.last_use[index].max(self.def_pos[index]);
        self.active.push(Active {
            value,
            register,
            end,
            transient: false,
        });
    }

    /// Frees the registers of the intervals that are over.
    fn expire(&mut self, position: u32) {
        let mut index = 0usize;
        while index < self.active.len() {
            if self.active[index].end < position {
                let entry = self.active.remove(index);
                self.release_register(entry.register);
            } else {
                index += 1;
            }
        }
    }

    /// Takes a free register of the given class.
    ///
    /// A register that belongs to more than one class leaves every
    /// list it is in, so that no other class is handed the same
    /// register while this one holds it.
    fn take_register(&mut self, class: RegClass) -> Option<PReg> {
        let index = class_index(class);
        let register = *self.free[index].first()?;
        for other in 0..3 {
            if other == index || !self.pools[other].contains(&register) {
                continue;
            }
            if let Some(position) = self.free[other]
                .iter()
                .position(|held| *held == register)
            {
                self.free[other].remove(position);
            }
        }
        self.free[index].remove(0);
        Some(register)
    }

    /// Returns a register to every class whose pool holds it.
    fn release_register(&mut self, register: PReg) {
        for index in 0..3 {
            if self.pools[index].contains(&register) {
                self.free[index].push(register);
            }
        }
    }

    /// The interval to spill to make room.
    ///
    /// Only an interval whose register belongs to the pool of the
    /// class can be spilled: freeing a register of another pool would
    /// not make room for the value that needs one.
    fn pick_victim(&self, class: RegClass) -> Option<VReg> {
        let mut best: Option<&Active> = None;
        for entry in &self.active {
            if entry.transient || !self.pools[class_index(class)].contains(&entry.register) {
                continue;
            }
            match best {
                Some(current) if current.end >= entry.end => {}
                _ => best = Some(entry),
            }
        }
        best.map(|entry| entry.value)
    }

    /// Spills a value: its register is saved in the frame and freed.
    fn spill(&mut self, victim: VReg, position: u32, queue: &mut Vec<Item>) -> Result<()> {
        let index = victim.index();
        if self.spilled[index] {
            return Ok(());
        }
        self.spilled[index] = true;
        let ty = self.type_of(victim);
        let size = self.value_size(&ty);
        let offset = self.allocate_slot(size);
        self.slot[index] = offset;
        // A value that is defined at this very position still holds
        // its register while the instruction runs, so it is saved
        // afterwards and its uses read the register.
        let defined_here = self.def_pos[index] == position;
        if defined_here {
            self.stored_after
                .entry(position)
                .or_default()
                .push(victim);
            self.spill_point[index] = position.saturating_add(1);
        } else {
            self.stored_before
                .entry(position)
                .or_default()
                .push(victim);
            self.spill_point[index] = position;
        }
        let class = ty.reg_class();
        let uses = self.uses[index].clone();
        for use_pos in uses {
            if use_pos < self.spill_point[index] {
                continue;
            }
            let id = self.reloads.len();
            self.reloads.push(Reload {
                ty: ty.clone(),
                class,
                offset,
                size,
                register: None,
            });
            self.reload_of.insert((use_pos, index), id);
            if use_pos == position {
                queue.push(Item {
                    class,
                    reload: Some(id),
                    value: None,
                });
            } else {
                self.pending.entry(use_pos).or_default().push(id);
            }
        }
        self.release(victim);
        Ok(())
    }

    /// Frees the register of a value.
    fn release(&mut self, value: VReg) {
        let mut index = 0usize;
        while index < self.active.len() {
            if self.active[index].value == value && !self.active[index].transient {
                let entry = self.active.remove(index);
                self.release_register(entry.register);
                return;
            }
            index += 1;
        }
    }

    /// Reserves a slot of the frame for a spilled value.
    fn allocate_slot(&mut self, size: u32) -> u32 {
        let offset = align_up(self.next_slot, SPILL_ALIGNMENT);
        self.next_slot = offset.saturating_add(size.max(1));
        offset
    }

    /// Rewrites the function: every operand becomes a physical
    /// register, and the spilled values are saved and read back.
    fn rewrite(&mut self, function: &mut MirFunction) -> Result<u32> {
        let mut added = 0u32;
        let mut position = 0u32;
        for block in &mut function.blocks {
            let old = core::mem::take(&mut block.insts);
            let mut insts: Vec<MInst> = Vec::with_capacity(old.len() + 4);
            for mut inst in old {
                if let Some(values) = self.stored_before.get(&position) {
                    for value in values.clone() {
                        insts.push(self.save(value)?);
                        added += 1;
                    }
                }
                if let Some(ids) = self.pending.get(&position) {
                    for id in ids.clone() {
                        let reload = &self.reloads[id];
                        let ty = reload.ty.clone();
                        let register = reload.register.ok_or_else(|| {
                            allocation_error(
                                "reading a spilled value back",
                                "the reloaded value has no register",
                            )
                        })?;
                        insts.push(
                            MInst::def_physical(
                                register,
                                ty.clone(),
                                MOp::SpillLoad {
                                    offset: reload.offset,
                                    size: reload.size,
                                },
                                Operands::none(),
                            )
                            .with_operand_type(ty),
                        );
                        added += 1;
                    }
                }
                self.rewrite_operands(&mut inst, position)?;
                self.rewrite_destination(&mut inst)?;
                insts.push(inst);
                if let Some(values) = self.stored_after.get(&position) {
                    for value in values.clone() {
                        insts.push(self.save(value)?);
                        added += 1;
                    }
                }
                position = position.saturating_add(1);
            }
            block.insts = insts;
        }
        Ok(added)
    }

    /// The instruction that saves a spilled value in the frame.
    fn save(&self, value: VReg) -> Result<MInst> {
        let index = value.index();
        let register = self.assigned[index]
            .ok_or_else(|| allocation_error("saving a spilled value", "the value has no register to save"))?;
        let ty = self.type_of(value);
        let size = self.value_size(&ty);
        Ok(MInst::none(
            ty.clone(),
            MOp::SpillStore {
                offset: self.slot[index],
                size,
            },
            Operands::one(Operand::Physical(register)),
        )
        .with_operand_type(ty))
    }

    /// Rewrites the destination of one instruction.
    ///
    /// The destination is virtual while the machine code is built and
    /// physical once the allocator has given the value a home, so the
    /// rewriting replaces the virtual register with the physical one.
    fn rewrite_destination(&self, inst: &mut MInst) -> Result<()> {
        let Some(dest) = inst.dest_vreg() else {
            return Ok(());
        };
        let register = self.assigned[dest.index()]
            .ok_or_else(|| allocation_error("assigning a register", "a virtual register has no register"))?;
        inst.dest = Some(Destination::Physical(register));
        Ok(())
    }

    /// Rewrites the operands of one instruction.
    fn rewrite_operands(&self, inst: &mut MInst, position: u32) -> Result<()> {
        let operands = inst.args.as_slice();
        if operands.is_empty() {
            return Ok(());
        }
        let mut rewritten: Vec<Operand> = Vec::with_capacity(operands.len());
        for operand in operands {
            rewritten.push(self.rewrite_operand(*operand, position)?);
        }
        inst.args = Operands::many(rewritten);
        Ok(())
    }

    /// Rewrites one operand.
    fn rewrite_operand(&self, operand: Operand, position: u32) -> Result<Operand> {
        let Some(value) = operand.virtual_reg() else {
            return Ok(operand);
        };
        let index = value.index();
        if self.spilled[index] && position >= self.spill_point[index] {
            let id = self
                .reload_of
                .get(&(position, index))
                .ok_or_else(|| {
                    allocation_error(
                        "reading a spilled value back",
                        "the use of the value has no reload",
                    )
                })?;
            let register = self.reloads[*id].register.ok_or_else(|| {
                allocation_error(
                    "reading a spilled value back",
                    "the reloaded value has no register",
                )
            })?;
            return Ok(Operand::Physical(register));
        }
        let register = self.assigned[index]
            .ok_or_else(|| allocation_error("assigning a register", "a virtual register has no register"))?;
        Ok(Operand::Physical(register))
    }

    /// The type of a virtual register.
    fn type_of(&self, value: VReg) -> MType {
        self.types
            .get(value.index())
            .cloned()
            .unwrap_or(MType::Void)
    }

    /// The class of a virtual register.
    fn class_of(&self, value: VReg) -> RegClass {
        self.type_of(value).reg_class()
    }

    /// The size of a value in the frame.
    fn value_size(&self, ty: &MType) -> u32 {
        match ty {
            MType::Void => 0,
            MType::Ptr { .. } => self.registers.pointer_size,
            other => other
                .size_in_bytes()
                .unwrap_or(self.registers.pointer_size),
        }
    }
}

/// Fuses an integer comparison with the conditional branch that
/// consumes it.
///
/// The selection stage leaves a comparison and the branch that tests
/// its one-bit result as two instructions, so a back end materialises
/// the result and then tests it again.  The pass replaces the pair
/// with a single [`MOp::CondBrCmp`] whose operands are the compared
/// values followed by the two target blocks, which a back end lowers
/// to one comparison and one conditional branch instead of the five
/// instructions the separate pair takes.
///
/// The rewrite is sound only when the comparison has a single use and
/// a single definition.  The machine code is not in static single
/// assignment form — a phi is lowered to a copy in every predecessor —
/// so a register written by several instructions, or read more than
/// once, cannot be folded into the branch without changing the values
/// the branch sees.  The comparison may sit in any block: the single
/// use is the branch, and a value is available at every use.
///
/// Only an integer comparison is fused.  A floating-point comparison
/// keeps its own lowering, which handles the ordered and unordered
/// results a [`MOp::FCmp`] can produce.
fn fuse_compare_branches(function: &mut MirFunction) {
    let count = function.vreg_types.len();
    let mut defs: Vec<u32> = vec![0; count];
    let mut uses: Vec<u32> = vec![0; count];
    let mut def_at: Vec<Option<(usize, usize)>> = vec![None; count];
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            if let Some(dest) = inst.dest_vreg()
                && let Some(slot) = defs.get_mut(dest.index())
            {
                *slot = slot.saturating_add(1);
                if let Some(location) = def_at.get_mut(dest.index()) {
                    *location = Some((block_index, inst_index));
                }
            }
            for operand in inst.args.as_slice() {
                if let Some(value) = operand.virtual_reg()
                    && let Some(slot) = uses.get_mut(value.index())
                {
                    *slot = slot.saturating_add(1);
                }
            }
        }
    }

    // Collect the rewrites before applying them, so that deleting a
    // comparison in one block and replacing a branch in another cannot
    // disturb the indices of the scan.
    let mut deleted: BTreeMap<(usize, usize), ()> = BTreeMap::new();
    let mut replaced: BTreeMap<(usize, usize), MInst> = BTreeMap::new();
    let mut touched: BTreeMap<usize, ()> = BTreeMap::new();
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            if !matches!(inst.op, MOp::CondBr) {
                continue;
            }
            let Some(Operand::Virtual(condition)) = inst.arg0() else {
                continue;
            };
            if defs.get(condition.index()).copied().unwrap_or(0) != 1
                || uses.get(condition.index()).copied().unwrap_or(0) != 1
            {
                continue;
            }
            let Some(Some((def_block, def_index))) = def_at.get(condition.index()).copied() else {
                continue;
            };
            if def_block == block_index && def_index == inst_index {
                continue;
            }
            let definition = &function.blocks[def_block].insts[def_index];
            let MOp::ICmp { pred } = definition.op else {
                continue;
            };
            let (Some(left), Some(right)) = (definition.arg0(), definition.arg1()) else {
                continue;
            };
            let Some((then, otherwise)) = inst.cond_br_targets() else {
                continue;
            };
            let operand_ty = definition.operand_ty.clone();
            let mut fused = MInst::none(
                MType::Void,
                MOp::CondBrCmp { pred },
                Operands::many(vec![
                    left,
                    right,
                    Operand::Immediate(i64::from(then)),
                    Operand::Immediate(i64::from(otherwise)),
                ]),
            );
            if let Some(ty) = operand_ty {
                fused = fused.with_operand_type(ty);
            }
            deleted.insert((def_block, def_index), ());
            replaced.insert((block_index, inst_index), fused);
            touched.insert(def_block, ());
            touched.insert(block_index, ());
        }
    }
    if replaced.is_empty() {
        return;
    }
    for (block_index, block) in function.blocks.iter_mut().enumerate() {
        if !touched.contains_key(&block_index) {
            continue;
        }
        let mut kept: Vec<MInst> = Vec::with_capacity(block.insts.len());
        for (inst_index, inst) in core::mem::take(&mut block.insts)
            .into_iter()
            .enumerate()
        {
            if deleted.contains_key(&(block_index, inst_index)) {
                continue;
            }
            if let Some(fused) = replaced.get(&(block_index, inst_index)) {
                kept.push(fused.clone());
                continue;
            }
            kept.push(inst);
        }
        block.insts = kept;
    }
}

/// Reuses a two-address operation's first operand as its destination.
///
/// The x86 form of [`MOp::Add`], [`MOp::Sub`], [`MOp::Mul`],
/// [`MOp::And`], [`MOp::Or`], [`MOp::Xor`], [`MOp::Neg`] and
/// [`MOp::Not`] is two-address: the destination is also the first
/// operand, so a back end emits `mov dest, left` before the operation.
/// When the first operand is dead after the operation, that copy is
/// unnecessary: the operation can write its result into the register the
/// operand already holds.
///
/// The pass therefore rewrites `dest = op(left, ...)` into
/// `left = op(left, ...)` and replaces every use of `dest` with `left`.
/// The allocator then gives the old value of `left` and the result of
/// the operation the same register, and the back end drops the copy.
///
/// Only an operation whose first operand has a single definition and is
/// read for the last time by the operation, and whose destination has a
/// single definition, is rewritten.  The machine code is not in static
/// single assignment form — a phi is lowered to one copy per predecessor
/// block — so a register written by several instructions cannot be
/// renamed without changing the values its uses read, and the old value
/// of the first operand is dead after the operation only when the
/// operation is its last use.
///
/// Only the two-address operations are rewritten.  A shift or a division
/// reads a fixed register in its back end and is not two-address in the
/// same way.
///
/// The pass runs to a fixpoint and recomputes the definition and the last
/// use of every register on each round, so a rewrite can expose the
/// operation that produced the first operand as reusable in turn.  Every
/// round leaves one fewer register defined, so the loop terminates.
fn reuse_two_address_destinations(function: &mut MirFunction) {
    let count = function.vreg_types.len();
    loop {
        let mut defs: Vec<u32> = vec![0; count];
        let mut last_use: Vec<u32> = vec![NOWHERE; count];
        let mut position = 0u32;
        for block in &function.blocks {
            for inst in &block.insts {
                if let Some(dest) = inst.dest_vreg()
                    && let Some(slot) = defs.get_mut(dest.index())
                {
                    *slot = slot.saturating_add(1);
                }
                for operand in inst.args.as_slice() {
                    let Some(value) = operand.virtual_reg() else {
                        continue;
                    };
                    if let Some(entry) = last_use.get_mut(value.index()) {
                        *entry = position;
                    }
                }
                position = position.saturating_add(1);
            }
        }
        let mut chosen: Option<(usize, usize, VReg, VReg)> = None;
        let mut position = 0u32;
        for (block_index, block) in function.blocks.iter().enumerate() {
            for (inst_index, inst) in block.insts.iter().enumerate() {
                if chosen.is_none()
                    && matches!(
                        inst.op,
                        MOp::Add | MOp::Sub | MOp::Mul | MOp::And | MOp::Or | MOp::Xor | MOp::Neg | MOp::Not
                    )
                    && let Some(dest) = inst.dest_vreg()
                    && let Some(Operand::Virtual(left)) = inst.arg0()
                    && dest != left
                    && defs.get(left.index()).copied().unwrap_or(0) == 1
                    && defs.get(dest.index()).copied().unwrap_or(0) == 1
                    && last_use.get(left.index()).copied() == Some(position)
                {
                    chosen = Some((block_index, inst_index, dest, left));
                }
                position = position.saturating_add(1);
            }
        }
        let Some((block_index, inst_index, dest, left)) = chosen else {
            return;
        };
        function.blocks[block_index].insts[inst_index].dest = Some(Destination::Virtual(left));
        rewrite_uses(function, dest, left);
    }
}

/// Replaces every use of `dest` with `src`.
///
/// The caller forwards a copy whose destination has one use, or several
/// uses that the source already outlives, so the walk rewrites every
/// operand that reads `dest`; it scans the whole function to stay
/// independent of where those uses sit.
fn rewrite_uses(function: &mut MirFunction, dest: VReg, src: VReg) {
    for block in &mut function.blocks {
        for inst in &mut block.insts {
            if !inst
                .args
                .as_slice()
                .contains(&Operand::Virtual(dest))
            {
                continue;
            }
            let mut operands: Vec<Operand> = Vec::with_capacity(inst.args.len());
            for operand in inst.args.as_slice() {
                if *operand == Operand::Virtual(dest) {
                    operands.push(Operand::Virtual(src));
                } else {
                    operands.push(*operand);
                }
            }
            inst.args = Operands::many(operands);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{FPred, IPred, MirBlock};
    use alloc::vec;
    use alloc::vec::Vec;

    /// A register model with `integers` integer registers, four
    /// floating-point registers, and four vector registers.
    fn model(integers: u16) -> RegModel {
        let mut int_regs: Vec<PReg> = Vec::new();
        for number in 0..integers {
            int_regs.push(PReg::new(number));
        }
        let mut float_regs: Vec<PReg> = Vec::new();
        for number in 0..4 {
            float_regs.push(PReg::new(100 + number));
        }
        let mut vec_regs: Vec<PReg> = Vec::new();
        for number in 0..4 {
            vec_regs.push(PReg::new(200 + number));
        }
        RegModel {
            int_regs,
            float_regs,
            vec_regs,
            callee_saved: Vec::new(),
            stack_pointer: PReg::new(30),
            frame_pointer: None,
            arg_regs: vec![PReg::new(30)],
            return_regs: vec![PReg::new(0)],
            pointer_size: 8,
        }
    }

    /// A function that keeps three values alive at once.
    ///
    /// The three values are live at the same time, so a file with
    /// three registers has to spill one of them.
    fn three_live_function() -> MirFunction {
        let mut function = MirFunction::new("sum");
        let first = function.new_vreg(MType::I32);
        let second = function.new_vreg(MType::I32);
        let third = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let total = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            first,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            second,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(MInst::def(
            third,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(3)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(first), Operand::Virtual(second)),
        ));
        block.insts.push(MInst::def(
            total,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(third), Operand::Virtual(sum)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(total)),
        ));
        function.push_block(block);
        function
    }

    /// A function that keeps three values alive at once, built from
    /// shifts.
    ///
    /// A shift is not two-address, so the register-reuse pass leaves the
    /// function alone and the three values stay live at the same time: a
    /// file with three registers still has to spill one of them.
    fn three_live_shift_function() -> MirFunction {
        let mut function = MirFunction::new("shift");
        let first = function.new_vreg(MType::I32);
        let second = function.new_vreg(MType::I32);
        let third = function.new_vreg(MType::I32);
        let shifted = function.new_vreg(MType::I32);
        let total = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            first,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            second,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(MInst::def(
            third,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(3)),
        ));
        block.insts.push(MInst::def(
            shifted,
            MType::I32,
            MOp::Shl,
            Operands::two(Operand::Virtual(first), Operand::Virtual(second)),
        ));
        block.insts.push(MInst::def(
            total,
            MType::I32,
            MOp::Shl,
            Operands::two(Operand::Virtual(third), Operand::Virtual(shifted)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(total)),
        ));
        function.push_block(block);
        function
    }

    /// A module with one function.
    fn module(function: MirFunction) -> MirModule {
        let mut module = MirModule::new();
        module.functions.push(function);
        module
    }

    #[test]
    fn every_operand_becomes_a_physical_register() {
        let mut machine = module(three_live_function());
        let stats = allocate(&mut machine, &model(4)).expect("the allocation succeeds");
        assert!(stats.registers > 0, "the allocator assigned registers");
        assert_eq!(stats.spills, 0, "the file is large enough");
        for function in &machine.functions {
            for block in &function.blocks {
                for inst in &block.insts {
                    for operand in inst.args.as_slice() {
                        assert!(
                            operand.virtual_reg().is_none(),
                            "a virtual register survived the allocation"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_allocation_is_deterministic() {
        let mut first = module(three_live_function());
        let mut second = module(three_live_function());
        let one = allocate(&mut first, &model(4)).expect("the allocation succeeds");
        let two = allocate(&mut second, &model(4)).expect("the allocation succeeds");
        assert_eq!(one, two, "the statistics match");
        assert_eq!(first, second, "the machine code matches");
    }

    #[test]
    fn a_small_file_spills_the_values_that_do_not_fit() {
        // The reuse pass folds the additions of `three_live_function`,
        // which removes the pressure; the shifts here are not
        // two-address, so the three values stay live.
        let mut machine = module(three_live_shift_function());
        let stats = allocate(&mut machine, &model(3)).expect("the allocation succeeds");
        assert_eq!(stats.spills, 1, "one value did not fit");
        assert!(stats.rewrites > 0, "the rewriting added instructions");
        let function = &machine.functions[0];
        let stores: Vec<&MInst> = function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| matches!(inst.op, MOp::SpillStore { .. }))
            .collect();
        let loads: Vec<&MInst> = function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| matches!(inst.op, MOp::SpillLoad { .. }))
            .collect();
        assert_eq!(stores.len(), 1, "the spilled value is saved once");
        assert_eq!(loads.len(), 1, "the spilled value is read back once");
        let MOp::SpillStore { offset, size } = stores[0].op else {
            unreachable!("the instruction saves a value");
        };
        assert_eq!(size, 4, "the slot holds a thirty-two-bit value");
        assert!(offset >= function.fixed_frame_size, "the slot is above the frame");
        assert!(function.frame_size > function.fixed_frame_size);
        let MOp::SpillLoad { offset: read, .. } = loads[0].op else {
            unreachable!("the instruction reads a value back");
        };
        assert_eq!(read, offset, "the value is read from the slot it was saved in");
    }

    #[test]
    fn the_argument_registers_are_never_handed_out() {
        let mut machine = module(three_live_function());
        allocate(&mut machine, &model(4)).expect("the allocation succeeds");
        for function in &machine.functions {
            for block in &function.blocks {
                for inst in &block.insts {
                    for operand in inst.args.as_slice() {
                        let Some(register) = operand.physical_reg() else {
                            continue;
                        };
                        assert_ne!(register, PReg::new(30), "the argument register was handed out");
                    }
                }
            }
        }
    }

    #[test]
    fn a_selected_kernel_can_be_allocated() {
        let source = "#[kernel]\nfn k(out: *mut int) {\n    let i = get_global_id(0);\n    *out = 1;\n}\n";
        let module = codevar_ocl_ir::lower::lower_source(source).expect("the source lowers");
        let entry = module
            .entry_points
            .first()
            .expect("the module has an entry point")
            .func;
        let selection = crate::isel::select_module(&module, entry).expect("the module selects");
        let mut machine = selection.module;
        let stats = allocate(&mut machine, &model(8)).expect("the allocation succeeds");
        assert!(stats.registers > 0, "the allocator assigned registers");
        for function in &machine.functions {
            for block in &function.blocks {
                for inst in &block.insts {
                    for operand in inst.args.as_slice() {
                        assert!(
                            operand.virtual_reg().is_none(),
                            "a virtual register survived the allocation"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_value_without_a_register_of_its_class_is_refused() {
        let mut machine = module(three_live_function());
        let error = allocate(&mut machine, &model(0)).expect_err("the allocation fails");
        assert!(
            matches!(&error, CodegenError::Allocation { .. }),
            "unexpected error: {error}"
        );
    }

    /// A register model whose vectors live in the same file as its
    /// floats, the way the targets that keep them together publish
    /// it.
    fn shared_model() -> RegModel {
        let mut model = model(8);
        model.vec_regs = model.float_regs.clone();
        model
    }

    /// A function with a floating-point value and a vector value that
    /// are live at the same time.
    ///
    /// The target keeps its vectors in the same file as its floats, so
    /// the two values must not be given the same register.
    fn float_and_vector_function() -> MirFunction {
        let mut function = MirFunction::new("mixed");
        let scalar = function.new_vreg(MType::F32);
        let vector = function.new_vreg(MType::Vec {
            lanes: 4,
            elem: Box::new(MType::F32),
        });
        let sum = function.new_vreg(MType::F32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            scalar,
            MType::F32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            vector,
            MType::Vec {
                lanes: 4,
                elem: Box::new(MType::F32),
            },
            MOp::VecSplat,
            Operands::one(Operand::Immediate(2)),
        ));
        // The two values are live at the same time: the sum reads
        // the scalar and the vector is read afterwards.
        block.insts.push(MInst::def(
            sum,
            MType::F32,
            MOp::FAdd,
            Operands::two(Operand::Virtual(scalar), Operand::Virtual(scalar)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(vector)),
        ));
        function.push_block(block);
        function
    }

    #[test]
    fn a_float_and_a_vector_never_share_a_register() {
        let mut machine = module(float_and_vector_function());
        allocate(&mut machine, &shared_model()).expect("the allocation succeeds");
        let function = &machine.functions[0];
        let mut registers: Vec<u16> = Vec::new();
        for block in &function.blocks {
            for inst in &block.insts {
                if let Some(dest) = inst.dest_reg() {
                    registers.push(dest.number());
                }
            }
        }
        assert!(
            registers
                .windows(2)
                .all(|pair| pair[0] != pair[1]),
            "two values share a register: {registers:?}"
        );
        // The two classes draw from the same file, so the two values
        // are in two different registers of it.
        let mut floats: Vec<u16> = Vec::new();
        for block in &function.blocks {
            for inst in &block.insts {
                if let Some(dest) = inst.dest_reg()
                    && dest.number() >= 100
                {
                    floats.push(dest.number());
                }
            }
        }
        assert!(
            floats.len() >= 2,
            "the two values are in the floating-point file: {floats:?}"
        );
    }

    #[test]
    fn a_value_that_is_never_defined_is_refused() {
        let mut function = MirFunction::new("broken");
        let value = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(value)),
        ));
        function.push_block(block);
        let mut machine = module(function);
        let error = allocate(&mut machine, &model(4)).expect_err("the allocation fails");
        assert!(
            matches!(&error, CodegenError::Allocation { .. }),
            "unexpected error: {error}"
        );
    }

    /// Runs only the copy-coalescing pre-pass over `function`.
    fn coalesce(function: &mut MirFunction) {
        let registers = model(4);
        let mut allocator = Allocator::new(&registers);
        allocator.reset(function);
        allocator.coalesce_copies(function);
    }

    /// The number of copy instructions of a function.
    fn copies(function: &MirFunction) -> usize {
        function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| matches!(inst.op, MOp::Copy))
            .count()
    }

    /// The values a function defines.
    fn defined(function: &MirFunction) -> Vec<VReg> {
        function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter_map(MInst::dest_vreg)
            .collect()
    }

    /// A function whose source is defined from an immediate and then
    /// copied through a chain that ends in a return.
    fn copy_chain() -> MirFunction {
        let mut function = MirFunction::new("chain");
        let source = function.new_vreg(MType::I32);
        let middle = function.new_vreg(MType::I32);
        let last = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            source,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(7)),
        ));
        block.insts.push(MInst::def(
            middle,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(source)),
        ));
        block.insts.push(MInst::def(
            last,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(middle)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(last)),
        ));
        function.push_block(block);
        function
    }

    #[test]
    fn a_chain_of_copies_collapses_to_the_original_source() {
        let mut function = copy_chain();
        let source = VReg::new(0);
        coalesce(&mut function);
        assert_eq!(copies(&function), 1, "only the copy from the immediate survives");
        let return_inst = function.blocks[0]
            .insts
            .last()
            .expect("the block ends with a return");
        assert_eq!(
            return_inst.arg0(),
            Some(Operand::Virtual(source)),
            "the return reads the original source"
        );
    }

    #[test]
    fn a_copy_with_two_uses_is_kept() {
        let mut function = MirFunction::new("twice");
        let source = function.new_vreg(MType::I32);
        let copied = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            source,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            copied,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(source)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(copied), Operand::Virtual(copied)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(sum)),
        ));
        function.push_block(block);
        coalesce(&mut function);
        assert_eq!(copies(&function), 2, "the two-use copy is kept");
        assert!(
            defined(&function).contains(&copied),
            "the two-use copy still defines its value"
        );
    }

    /// A two-use copy whose source is read after the last use of the
    /// destination is folded: the source is already live across both
    /// uses, so forwarding it is free.
    #[test]
    fn a_two_use_copy_is_folded_when_the_source_outlives_it() {
        let mut function = MirFunction::new("outlives");
        let source = function.new_vreg(MType::I32);
        let copied = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let total = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            source,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            copied,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(source)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(copied), Operand::Virtual(copied)),
        ));
        // The source is read after the two uses of the destination.
        block.insts.push(MInst::def(
            total,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(source), Operand::Virtual(sum)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(total)),
        ));
        function.push_block(block);
        coalesce(&mut function);
        assert_eq!(copies(&function), 1, "only the copy from the immediate survives");
        assert!(
            !defined(&function).contains(&copied),
            "the folded copy no longer defines its value"
        );
        let sum_inst = function.blocks[0]
            .insts
            .iter()
            .find(|inst| inst.dest_vreg() == Some(sum))
            .expect("the sum instruction survives");
        assert_eq!(
            sum_inst.arg0(),
            Some(Operand::Virtual(source)),
            "the first use reads the source"
        );
        assert_eq!(
            sum_inst.args.as_slice().get(1),
            Some(&Operand::Virtual(source)),
            "the second use reads the source"
        );
    }

    /// A two-use copy whose source dies before the destination is kept:
    /// forwarding it would extend the source's live range.
    #[test]
    fn a_two_use_copy_is_kept_when_the_source_dies_earlier() {
        let mut function = MirFunction::new("earlier");
        let source = function.new_vreg(MType::I32);
        let copied = function.new_vreg(MType::I32);
        let before = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            source,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            copied,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(source)),
        ));
        // The source's last use is here, before the destination's uses.
        block.insts.push(MInst::def(
            before,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(source), Operand::Virtual(source)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(copied), Operand::Virtual(copied)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(sum)),
        ));
        function.push_block(block);
        coalesce(&mut function);
        assert_eq!(copies(&function), 2, "the two-use copy is kept");
        assert!(
            defined(&function).contains(&copied),
            "the two-use copy still defines its value"
        );
    }

    /// A register written by a copy in two predecessors carries the
    /// phi's value there: neither copy may be forwarded, because a use
    /// would then read the value of one predecessor only.
    #[test]
    fn a_copy_into_a_merge_block_is_kept() {
        let mut function = MirFunction::new("merge");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let merged = function.new_vreg(MType::I32);

        let mut first = MirBlock::new("first");
        first.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        first.insts.push(MInst::def(
            merged,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(left)),
        ));
        first
            .insts
            .push(MInst::none(MType::Void, MOp::Br { target: 2 }, Operands::none()));

        let mut second = MirBlock::new("second");
        second.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        second.insts.push(MInst::def(
            merged,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(right)),
        ));
        second
            .insts
            .push(MInst::none(MType::Void, MOp::Br { target: 2 }, Operands::none()));

        let mut join = MirBlock::new("join");
        join.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(merged)),
        ));

        function.push_block(first);
        function.push_block(second);
        function.push_block(join);
        coalesce(&mut function);
        assert_eq!(copies(&function), 4, "the two merge copies are kept");
        assert!(
            defined(&function).contains(&merged),
            "the merge value is still defined"
        );
        let return_inst = function.blocks[2]
            .insts
            .last()
            .expect("the block ends with a return");
        assert_eq!(
            return_inst.arg0(),
            Some(Operand::Virtual(merged)),
            "the return still reads the merge value"
        );
    }

    #[test]
    fn a_copy_whose_value_is_never_used_is_deleted() {
        let mut function = MirFunction::new("dead");
        let source = function.new_vreg(MType::I32);
        let dead = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            source,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(3)),
        ));
        block.insts.push(MInst::def(
            dead,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Virtual(source)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(source)),
        ));
        function.push_block(block);
        coalesce(&mut function);
        assert_eq!(copies(&function), 1, "the dead copy is deleted");
        assert!(
            !defined(&function).contains(&dead),
            "the dead value is no longer defined"
        );
    }

    #[test]
    fn a_copy_from_an_immediate_is_kept() {
        let mut function = MirFunction::new("immediate");
        let value = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            value,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(9)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(value)),
        ));
        function.push_block(block);
        coalesce(&mut function);
        assert_eq!(copies(&function), 1, "there is no register to forward to");
    }

    #[test]
    fn the_copy_coalescing_is_idempotent() {
        let mut function = copy_chain();
        coalesce(&mut function);
        let collapsed = function.clone();
        coalesce(&mut function);
        assert_eq!(collapsed, function, "the pass makes no further change");
    }

    /// A comparison whose result feeds a branch in the same block.
    fn compared_function() -> MirFunction {
        let mut function = MirFunction::new("compared");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let condition = function.new_vreg(MType::I1);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(
            MInst::def(
                condition,
                MType::I1,
                MOp::ICmp { pred: IPred::Slt },
                Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
            )
            .with_operand_type(MType::I32),
        );
        block.insts.push(MInst::none(
            MType::Void,
            MOp::CondBr,
            Operands::three(
                Operand::Virtual(condition),
                Operand::Immediate(4),
                Operand::Immediate(5),
            ),
        ));
        function.push_block(block);
        function
    }

    /// The fused branch of a function, when it has one.
    fn fused_branch(function: &MirFunction) -> Option<&MInst> {
        function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find(|inst| matches!(inst.op, MOp::CondBrCmp { .. }))
    }

    /// The number of separate comparisons and conditional branches of
    /// a function, as a pair.
    fn comparison_branches(function: &MirFunction) -> (usize, usize) {
        let mut comparisons = 0usize;
        let mut branches = 0usize;
        for block in &function.blocks {
            for inst in &block.insts {
                if matches!(inst.op, MOp::ICmp { .. }) {
                    comparisons += 1;
                }
                if matches!(inst.op, MOp::CondBr) {
                    branches += 1;
                }
            }
        }
        (comparisons, branches)
    }

    #[test]
    fn a_comparison_feeding_a_branch_is_fused() {
        let mut function = compared_function();
        fuse_compare_branches(&mut function);
        assert_eq!(
            comparison_branches(&function),
            (0, 0),
            "the pair is replaced by one fused branch"
        );
        let branch = fused_branch(&function).expect("the branch is fused");
        assert_eq!(
            branch.args.as_slice(),
            &[
                Operand::Virtual(VReg::new(0)),
                Operand::Virtual(VReg::new(1)),
                Operand::Immediate(4),
                Operand::Immediate(5),
            ],
            "the branch carries the compared values and the two blocks"
        );
        assert_eq!(
            branch.operand_type(),
            Some(&MType::I32),
            "the compare width is kept"
        );
        assert_eq!(branch.cond_br_targets(), Some((4, 5)));
    }

    #[test]
    fn a_comparison_in_another_block_is_fused() {
        let mut function = MirFunction::new("split");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let condition = function.new_vreg(MType::I1);
        let mut first = MirBlock::new("first");
        first.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        first.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        first.insts.push(
            MInst::def(
                condition,
                MType::I1,
                MOp::ICmp { pred: IPred::Slt },
                Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
            )
            .with_operand_type(MType::I32),
        );
        first
            .insts
            .push(MInst::none(MType::Void, MOp::Br { target: 1 }, Operands::none()));
        let mut second = MirBlock::new("second");
        second.insts.push(MInst::none(
            MType::Void,
            MOp::CondBr,
            Operands::three(
                Operand::Virtual(condition),
                Operand::Immediate(0),
                Operand::Immediate(2),
            ),
        ));
        function.push_block(first);
        function.push_block(second);
        fuse_compare_branches(&mut function);
        assert_eq!(
            comparison_branches(&function),
            (0, 0),
            "the pair is fused across blocks"
        );
        let branch = fused_branch(&function).expect("the branch is fused");
        assert_eq!(branch.cond_br_targets(), Some((0, 2)));
        assert!(
            !function.blocks[0]
                .insts
                .iter()
                .any(|inst| matches!(inst.op, MOp::ICmp { .. })),
            "the comparison block no longer holds the comparison"
        );
    }

    #[test]
    fn a_comparison_with_two_uses_is_not_fused() {
        let mut function = compared_function();
        let extra = function.new_vreg(MType::I32);
        let last = function.blocks[0].insts.len();
        // A second use of the condition keeps the separate comparison
        // alive: folding it would have the branch and this use read the
        // same one-bit value.
        function.blocks[0].insts.insert(
            last - 1,
            MInst::def(
                extra,
                MType::I32,
                MOp::Add,
                Operands::two(Operand::Virtual(VReg::new(2)), Operand::Virtual(VReg::new(2))),
            ),
        );
        fuse_compare_branches(&mut function);
        assert_eq!(
            comparison_branches(&function),
            (1, 1),
            "the comparison and the branch are kept"
        );
        assert!(fused_branch(&function).is_none(), "nothing was fused");
    }

    #[test]
    fn a_comparison_written_twice_is_not_fused() {
        let mut function = compared_function();
        // The condition register is defined a second time, so folding
        // the first comparison would change the value the branch reads.
        function.blocks[0].insts.insert(
            2,
            MInst::def(
                VReg::new(2),
                MType::I1,
                MOp::ICmp { pred: IPred::Slt },
                Operands::two(Operand::Immediate(0), Operand::Immediate(1)),
            ),
        );
        fuse_compare_branches(&mut function);
        assert_eq!(
            comparison_branches(&function),
            (2, 1),
            "both comparisons and the branch are kept"
        );
        assert!(fused_branch(&function).is_none(), "nothing was fused");
    }

    #[test]
    fn a_condition_that_is_not_an_integer_comparison_is_not_fused() {
        let mut floating = MirFunction::new("floating");
        let left = floating.new_vreg(MType::F32);
        let right = floating.new_vreg(MType::F32);
        let condition = floating.new_vreg(MType::I1);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::F32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::F32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(
            MInst::def(
                condition,
                MType::I1,
                MOp::FCmp { pred: FPred::Olt },
                Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
            )
            .with_operand_type(MType::F32),
        );
        block.insts.push(MInst::none(
            MType::Void,
            MOp::CondBr,
            Operands::three(
                Operand::Virtual(condition),
                Operand::Immediate(0),
                Operand::Immediate(1),
            ),
        ));
        floating.push_block(block);
        fuse_compare_branches(&mut floating);
        assert!(
            fused_branch(&floating).is_none(),
            "a floating-point comparison is kept"
        );

        let mut copied = MirFunction::new("copied");
        let condition = copied.new_vreg(MType::I1);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            condition,
            MType::I1,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::CondBr,
            Operands::three(
                Operand::Virtual(condition),
                Operand::Immediate(0),
                Operand::Immediate(1),
            ),
        ));
        copied.push_block(block);
        fuse_compare_branches(&mut copied);
        assert!(
            fused_branch(&copied).is_none(),
            "a condition that is a plain copy is kept"
        );
    }

    /// A function whose first operand is read for the last time by a
    /// two-address addition, so the addition may write into it.
    fn two_address_function() -> MirFunction {
        let mut function = MirFunction::new("two_address");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(sum)),
        ));
        function.push_block(block);
        function
    }

    /// The operation with the given predicate, when the function has one.
    fn operation(function: &MirFunction, wanted: fn(&MOp) -> bool) -> Option<&MInst> {
        function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find(|inst| wanted(&inst.op))
    }

    #[test]
    fn a_two_address_operation_writes_into_its_dead_operand() {
        let mut function = two_address_function();
        let left = VReg::new(0);
        let sum = VReg::new(2);
        reuse_two_address_destinations(&mut function);
        assert!(
            !defined(&function).contains(&sum),
            "the old destination is no longer defined"
        );
        let add = operation(&function, |op| matches!(op, MOp::Add)).expect("the addition survives");
        assert_eq!(
            add.dest_vreg(),
            Some(left),
            "the addition writes into its first operand"
        );
        let return_inst = function.blocks[0]
            .insts
            .last()
            .expect("the block ends with a return");
        assert_eq!(
            return_inst.arg0(),
            Some(Operand::Virtual(left)),
            "the return reads the first operand"
        );
    }

    #[test]
    fn a_negation_writes_into_its_dead_operand() {
        let mut function = MirFunction::new("negation");
        let value = function.new_vreg(MType::I32);
        let negated = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            value,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            negated,
            MType::I32,
            MOp::Neg,
            Operands::one(Operand::Virtual(value)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(negated)),
        ));
        function.push_block(block);
        reuse_two_address_destinations(&mut function);
        assert!(
            !defined(&function).contains(&negated),
            "the old destination is no longer defined"
        );
        let neg = operation(&function, |op| matches!(op, MOp::Neg)).expect("the negation survives");
        assert_eq!(
            neg.dest_vreg(),
            Some(value),
            "the negation writes into its operand"
        );
    }

    #[test]
    fn a_two_address_operation_keeps_an_operand_that_is_used_later() {
        let mut function = MirFunction::new("live");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
        ));
        // The first operand is read once more, after the addition.
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(left)),
        ));
        function.push_block(block);
        reuse_two_address_destinations(&mut function);
        let add = operation(&function, |op| matches!(op, MOp::Add)).expect("the addition survives");
        assert_eq!(add.dest_vreg(), Some(sum), "the addition keeps its destination");
        assert!(
            defined(&function).contains(&sum),
            "the destination is still defined"
        );
    }

    #[test]
    fn a_two_address_operation_keeps_an_operand_written_twice() {
        let mut function = MirFunction::new("redefined_operand");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        // The first operand is defined a second time.
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(3)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(sum)),
        ));
        function.push_block(block);
        reuse_two_address_destinations(&mut function);
        let add = operation(&function, |op| matches!(op, MOp::Add)).expect("the addition survives");
        assert_eq!(add.dest_vreg(), Some(sum), "the addition keeps its destination");
    }

    #[test]
    fn a_two_address_operation_keeps_a_destination_written_twice() {
        let mut function = MirFunction::new("redefined_destination");
        let left = function.new_vreg(MType::I32);
        let right = function.new_vreg(MType::I32);
        let sum = function.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        // The destination is defined a first time by a copy.
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(3)),
        ));
        block.insts.push(MInst::def(
            sum,
            MType::I32,
            MOp::Add,
            Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(sum)),
        ));
        function.push_block(block);
        reuse_two_address_destinations(&mut function);
        let add = operation(&function, |op| matches!(op, MOp::Add)).expect("the addition survives");
        assert_eq!(add.dest_vreg(), Some(sum), "the addition keeps its destination");
    }

    #[test]
    fn an_operation_that_is_not_two_address_is_kept() {
        let mut shifted = MirFunction::new("shifted");
        let value = shifted.new_vreg(MType::I32);
        let amount = shifted.new_vreg(MType::I32);
        let result = shifted.new_vreg(MType::I32);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            value,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            amount,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(MInst::def(
            result,
            MType::I32,
            MOp::Shl,
            Operands::two(Operand::Virtual(value), Operand::Virtual(amount)),
        ));
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(result)),
        ));
        shifted.push_block(block);
        reuse_two_address_destinations(&mut shifted);
        let shift = operation(&shifted, |op| matches!(op, MOp::Shl)).expect("the shift survives");
        assert_eq!(shift.dest_vreg(), Some(result), "a shift keeps its destination");

        let mut compared = MirFunction::new("compared");
        let left = compared.new_vreg(MType::I32);
        let right = compared.new_vreg(MType::I32);
        let condition = compared.new_vreg(MType::I1);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def(
            left,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(1)),
        ));
        block.insts.push(MInst::def(
            right,
            MType::I32,
            MOp::Copy,
            Operands::one(Operand::Immediate(2)),
        ));
        block.insts.push(
            MInst::def(
                condition,
                MType::I1,
                MOp::ICmp { pred: IPred::Slt },
                Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
            )
            .with_operand_type(MType::I32),
        );
        block.insts.push(MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(condition)),
        ));
        compared.push_block(block);
        reuse_two_address_destinations(&mut compared);
        let compare =
            operation(&compared, |op| matches!(op, MOp::ICmp { .. })).expect("the comparison survives");
        assert_eq!(
            compare.dest_vreg(),
            Some(condition),
            "a comparison keeps its destination"
        );
    }

    #[test]
    fn the_allocation_folds_a_two_address_operation() {
        let mut machine = module(two_address_function());
        allocate(&mut machine, &model(4)).expect("the allocation succeeds");
        let function = &machine.functions[0];
        let add = operation(function, |op| matches!(op, MOp::Add)).expect("the addition survives");
        let destination = add
            .dest_reg()
            .expect("the addition writes a register");
        // The addition writes the register its first operand already
        // holds, so the back end has nothing to move.
        let first = function.blocks[0]
            .insts
            .first()
            .expect("the block starts with a copy");
        assert_eq!(
            first.dest_reg(),
            Some(destination),
            "the operand and the result share a register"
        );
        let return_inst = function.blocks[0]
            .insts
            .last()
            .expect("the block ends with a return");
        assert_eq!(
            return_inst.arg0().and_then(Operand::physical_reg),
            Some(destination),
            "the return reads the shared register"
        );
    }
}
