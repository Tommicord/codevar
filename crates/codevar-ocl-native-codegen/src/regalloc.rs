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
    /// The class of the register.
    class: RegClass,
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
    /// The free registers of every class.
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
        Self {
            registers,
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
        for class in [RegClass::Integer, RegClass::Float, RegClass::Vector] {
            let list = self.registers.regs_for(class);
            for &reg in list {
                if self.registers.arg_regs.contains(&reg) {
                    continue;
                }
                self.free[class_index(class)].push(reg);
            }
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
                class: item.class,
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
            class: item.class,
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
                self.free[class_index(entry.class)].push(entry.register);
            } else {
                index += 1;
            }
        }
    }

    /// Takes a free register of the given class.
    fn take_register(&mut self, class: RegClass) -> Option<PReg> {
        let list = &mut self.free[class_index(class)];
        if list.is_empty() {
            return None;
        }
        Some(list.remove(0))
    }

    /// The interval to spill to make room.
    fn pick_victim(&self, class: RegClass) -> Option<VReg> {
        let mut best: Option<&Active> = None;
        for entry in &self.active {
            if entry.transient || entry.class != class {
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
                self.free[class_index(entry.class)].push(entry.register);
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
                        insts.push(MInst::def_physical(
                            register,
                            ty,
                            MOp::SpillLoad {
                                offset: reload.offset,
                                size: reload.size,
                            },
                            Operands::none(),
                        ));
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
            ty,
            MOp::SpillStore {
                offset: self.slot[index],
                size,
            },
            Operands::one(Operand::Physical(register)),
        ))
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
        let register = self.assigned[dest.index()].ok_or_else(|| {
            allocation_error("assigning a register", "a virtual register has no register")
        })?;
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
    use crate::mir::MirBlock;
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
        let mut machine = module(three_live_function());
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
}
