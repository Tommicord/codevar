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

//! The contract between the pipeline and an architecture back end.
//!
//! A back end receives a finished [`MirModule`] and returns the bytes
//! of the machine code.  Everything the back end needs to know about
//! the machine — the register file, the calling convention, the frame
//! layout — it answers for itself through [`NativeTarget`]; the
//! pipeline never asks the target what an instruction looks like.
//!
//! # What a back end does
//!
//! 1. **Expand.**  Turn the semantic operations of the machine IR into
//!    the target's own instructions: a compare-and-select becomes two
//!    instructions, an address computation becomes a chain of adds or a
//!    single address instruction, a narrow load becomes a load and an
//!    extension.
//! 2. **Lay out.**  Place the functions of the module, resolve the
//!    calls between them, and give every function its prologue and
//!    epilogue.
//! 3. **Encode.**  Turn the instruction stream into bytes.
//!
//! # The frame
//!
//! The frame layout is fixed by the machine IR, not by the back end:
//! the function carries the total frame size, and every slot inside
//! the frame is addressed by its offset from the frame base.  The frame
//! base is the value of the stack pointer right after the frame has
//! been reserved, so a back end that reserves the frame with a single
//! subtraction can place every slot at a known distance from the stack
//! pointer.
//!
//! Registers the target must preserve across a call are listed in
//! [`RegModel::callee_saved`]; the back end saves them in its prologue
//! and restores them in its epilogue.  A back end saves them *before*
//! it reserves the frame, so the saved registers sit above the frame
//! and the frame base stays at the bottom of the reserved area.
//!
//! # The argument registers
//!
//! The registers of [`RegModel::arg_regs`] belong to the calling
//! convention: the allocator never hands them out, and a back end
//! moves a value into one of them only right before a call.  A
//! parameter of a function arrives in the argument register of its
//! position, and the back end moves it to the register the allocator
//! chose.

use alloc::vec::Vec;

use crate::error::Result;
use crate::mir::{MirModule, PReg, RegClass, SymId};

/// The register file and calling convention of a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegModel {
    /// The integer registers the allocator may use, in the order it
    /// should try them.
    pub int_regs: Vec<PReg>,
    /// The floating-point registers the allocator may use.
    pub float_regs: Vec<PReg>,
    /// The vector registers the allocator may use.  A target whose
    /// vectors live in the same file as its floats lists the same
    /// registers here.
    pub vec_regs: Vec<PReg>,
    /// The registers whose value survives a call.
    pub callee_saved: Vec<PReg>,
    /// The stack pointer.
    pub stack_pointer: PReg,
    /// The frame pointer, when the target dedicates one.
    pub frame_pointer: Option<PReg>,
    /// The registers the arguments arrive in, in argument order.
    pub arg_regs: Vec<PReg>,
    /// The registers a result is returned in.
    pub return_regs: Vec<PReg>,
    /// The size of a pointer in bytes.
    pub pointer_size: u32,
}

impl RegModel {
    /// The registers a value of the given class may occupy.
    #[must_use]
    pub fn regs_for(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Integer => &self.int_regs,
            RegClass::Float => &self.float_regs,
            RegClass::Vector => &self.vec_regs,
        }
    }

    /// Whether the register must survive a call.
    #[must_use]
    pub fn is_callee_saved(&self, reg: PReg) -> bool {
        self.callee_saved.contains(&reg)
    }
}

/// The machine code of a whole module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeImage {
    /// The bytes of the machine code.
    pub bytes: Vec<u8>,
    /// The offset of the entry point inside [`CodeImage::bytes`].
    pub entry_offset: u32,
    /// The number of machine instructions the image holds.
    pub instruction_count: u32,
}

impl CodeImage {
    /// The bytes of the machine code.
    #[must_use]
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The offset of the entry point inside the image.
    #[must_use]
    #[inline]
    pub fn entry_offset(&self) -> u32 {
        self.entry_offset
    }

    /// The number of machine instructions the image holds.
    #[must_use]
    #[inline]
    pub fn instruction_count(&self) -> u32 {
        self.instruction_count
    }
}

/// An architecture back end.
///
/// The trait is object safe on purpose: the pipeline takes a back end
/// by reference, so a program can choose its target at run time.
pub trait NativeTarget {
    /// The target's name, used in diagnostics.
    fn name(&self) -> &'static str;

    /// The register file and calling convention of the target.
    fn registers(&self) -> &RegModel;

    /// Lowers a whole module to machine code.
    ///
    /// The entry symbol must name a function of the module.  The
    /// returned image holds the code of every function of the module;
    /// the entry offset points at the entry function.
    ///
    /// # Errors
    ///
    /// Returns [`CodegenError::Target`](crate::error::CodegenError::Target)
    /// when the module uses something the target cannot express, and
    /// [`CodegenError::Unsupported`](crate::error::CodegenError::Unsupported)
    /// when the machine IR itself is outside the target's reach.
    fn lower_module(&self, module: &MirModule, entry: SymId) -> Result<CodeImage>;
}
