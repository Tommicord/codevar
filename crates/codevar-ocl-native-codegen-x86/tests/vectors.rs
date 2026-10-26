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

//! The execution of the vector operations.
//!
//! The kernel of the tests broadcasts the global identifier over the
//! lanes of a vector, writes one lane of the vector, shuffles the
//! lanes of the result, and reads two lanes of it back.  The
//! baseline has no instruction that picks lanes from two registers at
//! once, so the shuffle is a sequence of lane moves.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::ptr;

use codevar_ocl_abi::{MAX_PARAMS, WorkItem, WorkItemField};
use codevar_ocl_native_codegen::mir::{ArgKind, MOp, MType, MirFunction, MirModule, Operands, SymbolKind};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The size of the page the code is copied into.
const PAGE: usize = 4096;

/// The type of a vector of four thirty-two-bit lanes.
fn vector_of_four() -> MType {
    MType::Vec {
        lanes: 4,
        elem: Box::new(MType::I32),
    }
}

/// The type of a vector of two sixty-four-bit lanes.
fn vector_of_two() -> MType {
    MType::Vec {
        lanes: 2,
        elem: Box::new(MType::I64),
    }
}

/// Builds the kernel of the tests.
fn kernel_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut function = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let value = vreg(&mut function, R10, MType::I32);
    let base = vreg(&mut function, R15, MType::Ptr { size: 4 });
    let output = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let splat = vreg(&mut function, XMM0, vector_of_four());
    let written = vreg(&mut function, XMM1, vector_of_four());
    let shuffled = vreg(&mut function, XMM2, vector_of_four());
    let wide = vreg(&mut function, XMM3, vector_of_two());
    let first = vreg(&mut function, R11, MType::I32);
    let second = vreg(&mut function, R12, MType::I32);
    let third = vreg(&mut function, R13, MType::I32);
    let fourth = vreg(&mut function, R14, MType::I32);
    let pointer = vreg(&mut function, R15, MType::Ptr { size: 4 });

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
    entry.insts.push(def(
        splat,
        vector_of_four(),
        MOp::VecSplat,
        Operands::one(reg(R10)),
    ));
    entry.insts.push(def(
        written,
        vector_of_four(),
        MOp::VecInsert { index: 0 },
        Operands::two(reg(XMM0), reg(R10)),
    ));
    entry.insts.push(def(
        shuffled,
        vector_of_four(),
        MOp::VecShuffle {
            mask: vec![3, 2, 1, 0],
        },
        Operands::two(reg(XMM1), reg(XMM1)),
    ));
    entry.insts.push(def_with_operands(
        first,
        MType::I32,
        vector_of_four(),
        MOp::VecExtract { index: 0 },
        Operands::one(reg(XMM2)),
    ));
    entry.insts.push(def_with_operands(
        second,
        MType::I32,
        vector_of_four(),
        MOp::VecExtract { index: 3 },
        Operands::one(reg(XMM2)),
    ));
    entry
        .insts
        .push(def(wide, vector_of_two(), MOp::VecSplat, Operands::one(reg(R10))));
    entry.insts.push(def_with_operands(
        third,
        MType::I64,
        vector_of_two(),
        MOp::VecExtract { index: 1 },
        Operands::one(reg(XMM3)),
    ));
    entry.insts.push(def_with_operands(
        fourth,
        MType::I64,
        vector_of_two(),
        MOp::VecExtract { index: 0 },
        Operands::one(reg(XMM3)),
    ));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(RBX), reg(R11)),
    ));
    entry.insts.push(def(
        pointer,
        MType::Ptr { size: 4 },
        MOp::Gep {
            size: 4,
            signed: false,
        },
        Operands::two(reg(RBX), imm(1)),
    ));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(R15), reg(R12)),
    ));
    entry.insts.push(def(
        pointer,
        MType::Ptr { size: 4 },
        MOp::Gep {
            size: 4,
            signed: false,
        },
        Operands::two(reg(RBX), imm(2)),
    ));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(R15), reg(R13)),
    ));
    entry.insts.push(def(
        pointer,
        MType::Ptr { size: 4 },
        MOp::Gep {
            size: 4,
            signed: false,
        },
        Operands::two(reg(RBX), imm(3)),
    ));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(R15), reg(R14)),
    ));
    entry
        .insts
        .push(none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(entry);
    function.frame_size = 0;

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
/// four words it wrote.
fn run(global_id: u32) -> [u32; 4] {
    let (module, entry) = kernel_module();
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
    output
}

#[test]
fn the_lanes_of_the_vector_are_read_and_written() {
    let output = run(5);
    // The broadcast fills every lane with the identifier, the write
    // puts it in the first lane, and the shuffle turns the lanes
    // around.
    assert_eq!(output[0], 5, "the first lane is wrong");
    assert_eq!(output[1], 5, "the last lane is wrong");
    // The sixty-four-bit lanes hold the identifier as well.
    assert_eq!(output[2], 5, "the wide lane is wrong");
    assert_eq!(output[3], 5, "the wide lane is wrong");
}
