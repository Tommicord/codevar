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

//! The execution of the control flow and of the frame.
//!
//! The kernel of the tests branches on the global identifier, and
//! keeps a value in a slot of the frame across a branch.  The frame
//! of a function is reserved with a single subtraction, and every
//! slot of the frame is at a known distance from the stack pointer.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]
mod common;

use std::ptr;

use codevar_ocl_abi::{MAX_PARAMS, WorkItem, WorkItemField};
use codevar_ocl_native_codegen::mir::{
    ArgKind, IPred, MOp, MType, MirFunction, MirModule, Operands, SymbolKind,
};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The size of the page the code is copied into.
const PAGE: usize = 4096;

/// Builds a kernel that branches on the global identifier and keeps a
/// value in the frame.
///
/// When `fused` is set the branch is a single
/// [`MOp::CondBrCmp`] that compares and branches without
/// materialising the one-bit condition; otherwise the separate
/// [`MOp::ICmp`] and [`MOp::CondBr`] pair is emitted.
fn kernel_module(fused: bool) -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut function = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let value = vreg(&mut function, R10, MType::I32);
    let condition = vreg(&mut function, R13, MType::I1);
    let base = vreg(&mut function, R15, MType::Ptr { size: 4 });
    let output = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let chosen = vreg(&mut function, R11, MType::I32);

    entry.insts.push(def(
        item,
        MType::Ptr { size: 4 },
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry.insts.push(def(
        value,
        MType::I32,
        MOp::WorkItemLoad {
            field: WorkItemField::GlobalId,
            lane: 0,
        },
        Operands::one(reg(RBX)),
    ));
    entry.insts.push(def(
        base,
        MType::Ptr { size: 4 },
        MOp::ArgBlockBase,
        Operands::one(reg(RBX)),
    ));
    entry.insts.push(def(
        output,
        MType::Ptr { size: 4 },
        MOp::ArgLoad {
            index: 0,
            kind: ArgKind::Pointer,
        },
        Operands::one(reg(R15)),
    ));
    // The value is kept in a slot of the frame, so that the branch
    // that follows finds it there.
    entry.insts.push(none(
        MType::Void,
        MOp::SpillStore { offset: 8, size: 4 },
        Operands::one(reg(R10)),
    ));
    if fused {
        entry.insts.push(none(
            MType::Void,
            MOp::CondBrCmp { pred: IPred::Slt },
            Operands::many(vec![reg(R10), imm(10), imm(1), imm(2)]),
        ));
    } else {
        entry.insts.push(def_with_operands(
            condition,
            MType::I1,
            MType::I32,
            MOp::ICmp { pred: IPred::Slt },
            Operands::two(reg(R10), imm(10)),
        ));
        entry.insts.push(none(
            MType::Void,
            MOp::CondBr,
            Operands::three(reg(R13), imm(1), imm(2)),
        ));
    }
    function.push_block(entry);

    let mut then_block = block("then");
    then_block
        .insts
        .push(def(chosen, MType::I32, MOp::Copy, Operands::one(imm(1))));
    then_block
        .insts
        .push(none(MType::Void, MOp::Br { target: 3 }, Operands::none()));
    function.push_block(then_block);

    let mut else_block = block("else");
    else_block
        .insts
        .push(def(chosen, MType::I32, MOp::Copy, Operands::one(imm(2))));
    else_block
        .insts
        .push(none(MType::Void, MOp::Br { target: 3 }, Operands::none()));
    function.push_block(else_block);

    let mut merge = block("merge");
    let pointer = vreg(&mut function, R15, MType::Ptr { size: 4 });
    merge.insts.push(def(
        value,
        MType::I32,
        MOp::SpillLoad { offset: 8, size: 4 },
        Operands::none(),
    ));
    merge.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(RBX), reg(R11)),
    ));
    merge.insts.push(def(
        pointer,
        MType::Ptr { size: 4 },
        MOp::Gep {
            size: 4,
            signed: false,
        },
        Operands::two(reg(RBX), imm(1)),
    ));
    merge.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(R15), reg(R10)),
    ));
    merge
        .insts
        .push(none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(merge);

    function.frame_size = 32;

    let mut module = MirModule::new();
    let symbol = module.add_symbol("kernel", SymbolKind::Function { index: 0 }, false);
    function.symbol = symbol;
    module.functions.push(function);
    (module, symbol)
}

/// A page of memory the code can run in.
fn executable_page() -> *mut u8 {
    let page = unsafe {
        libc::mmap(
            ptr::null_mut(),
            PAGE,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(page, libc::MAP_FAILED, "the page is mapped");
    page as *mut u8
}

/// Runs the kernel with the given global identifier and returns the
/// two words it wrote.
fn run(global_id: u32, fused: bool) -> [u32; 2] {
    let (module, entry) = kernel_module(fused);
    let image = target()
        .lower_module(&module, entry)
        .expect("the module compiles");
    let page = executable_page();
    // SAFETY: the page holds a whole number of bytes of executable
    // memory, and the code is shorter than the page.
    unsafe {
        ptr::copy_nonoverlapping(image.bytes().as_ptr(), page, image.bytes().len());
    }
    let address = page as usize + image.entry_offset() as usize;
    // SAFETY: the page holds the code of the entry function, which
    // follows the calling convention of a kernel.
    let kernel: unsafe extern "C" fn(*mut WorkItem) = unsafe { std::mem::transmute(address) };

    let mut output = [0u32; 4];
    let mut block = [0u8; MAX_PARAMS];
    let pointer = output.as_mut_ptr() as usize;
    block[0..8].copy_from_slice(&pointer.to_ne_bytes());
    let mut item = WorkItem {
        global_id: [global_id, 0, 0],
        local_id: [0; 3],
        group_id: [0; 3],
        global_offset: [0; 3],
        num_groups: [0; 3],
        local_size: [0; 3],
        arg_base: block.as_mut_ptr(),
    };
    // SAFETY: the work item is fully initialized and points at a
    // live argument block.
    unsafe {
        kernel(&mut item);
    }
    // SAFETY: the page was mapped by this test.
    unsafe {
        libc::munmap(page as *mut libc::c_void, PAGE);
    }
    [output[0], output[1]]
}

#[test]
fn the_branch_that_the_condition_selects_is_taken() {
    let result = run(5, false);
    assert_eq!(result[0], 1, "the wrong branch was taken");
    assert_eq!(result[1], 5, "the frame does not hold the value");
}

#[test]
fn the_other_branch_is_taken_when_the_condition_fails() {
    let result = run(50, false);
    assert_eq!(result[0], 2, "the wrong branch was taken");
    assert_eq!(result[1], 50, "the frame does not hold the value");
}

#[test]
fn the_fused_comparison_branches_to_the_selected_block() {
    let result = run(5, true);
    assert_eq!(result[0], 1, "the wrong branch was taken");
    assert_eq!(result[1], 5, "the frame does not hold the value");
}

#[test]
fn the_fused_comparison_falls_through_when_the_condition_fails() {
    let result = run(50, true);
    assert_eq!(result[0], 2, "the wrong branch was taken");
    assert_eq!(result[1], 50, "the frame does not hold the value");
}
