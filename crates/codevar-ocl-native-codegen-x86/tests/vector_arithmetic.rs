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

//! The arithmetic over the lanes of a vector.
//!
//! The architecture has a packed form for the operations it can do on
//! several lanes at once, and the back end uses it.  The operations it
//! has no packed form for run one lane at a time.  The kernel of the
//! test adds two vectors of four thirty-two-bit lanes, multiplies them
//! by a third, and writes the four results out.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::ptr;

use codevar_ocl_abi::{MAX_PARAMS, WorkItem, WorkItemField};
use codevar_ocl_native_codegen::mir::{ArgKind, MOp, MType, MirFunction, MirModule, Operands, SymbolKind};
use codevar_ocl_native_codegen::target::{NativeTarget, TargetCode};
use codevar_ocl_native_codegen_x86::X86Target;
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

/// Builds the kernel of the test.
///
/// The vector holds the identifier of the work item in its first lane
/// and the identifier plus one, two and three in the others.  The
/// second vector holds twice that, and the third one the value three.
/// The result is the sum of the first two times the third.
fn kernel_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut function = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let value = vreg(&mut function, R10, MType::I32);
    let base = vreg(&mut function, R15, MType::Ptr { size: 4 });
    let output = vreg(&mut function, RDI, MType::Ptr { size: 4 });
    let one = vreg(&mut function, R11, MType::I32);
    let two = vreg(&mut function, R12, MType::I32);
    let three = vreg(&mut function, R13, MType::I32);
    let first = vreg(&mut function, RSI, MType::I32);
    let second = vreg(&mut function, R14, MType::I32);
    let third = vreg(&mut function, RCX, MType::I32);
    let fourth = vreg(&mut function, RDX, MType::I32);
    let lanes = vreg(&mut function, XMM0, vector_of_four());
    let doubled = vreg(&mut function, XMM1, vector_of_four());
    let tripled = vreg(&mut function, XMM2, vector_of_four());
    let sum = vreg(&mut function, XMM3, vector_of_four());
    let product = vreg(&mut function, XMM4, vector_of_four());
    let result = vreg(&mut function, XMM5, vector_of_four());
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
    // The first vector: the identifier in every lane, then the
    // identifier plus one, two and three in the other lanes.
    entry.insts.push(def(
        lanes,
        vector_of_four(),
        MOp::VecSplat,
        Operands::one(reg(R10)),
    ));
    entry
        .insts
        .push(def(one, MType::I32, MOp::Add, Operands::two(reg(R10), imm(1))));
    entry.insts.push(def(
        doubled,
        vector_of_four(),
        MOp::VecInsert { index: 1 },
        Operands::two(reg(XMM0), reg(R11)),
    ));
    entry
        .insts
        .push(def(two, MType::I32, MOp::Add, Operands::two(reg(R10), imm(2))));
    entry.insts.push(def(
        doubled,
        vector_of_four(),
        MOp::VecInsert { index: 2 },
        Operands::two(reg(XMM1), reg(R12)),
    ));
    entry
        .insts
        .push(def(three, MType::I32, MOp::Add, Operands::two(reg(R10), imm(3))));
    entry.insts.push(def(
        doubled,
        vector_of_four(),
        MOp::VecInsert { index: 3 },
        Operands::two(reg(XMM1), reg(R13)),
    ));
    // The second vector: twice the first one.
    entry.insts.push(def(
        tripled,
        vector_of_four(),
        MOp::VecSplat,
        Operands::one(imm(2)),
    ));
    entry.insts.push(def_with_operands(
        sum,
        vector_of_four(),
        vector_of_four(),
        MOp::Add,
        Operands::two(reg(XMM1), reg(XMM2)),
    ));
    // The third vector: three in every lane.
    entry.insts.push(def(
        product,
        vector_of_four(),
        MOp::VecSplat,
        Operands::one(imm(3)),
    ));
    entry.insts.push(def_with_operands(
        result,
        vector_of_four(),
        vector_of_four(),
        MOp::Mul,
        Operands::two(reg(XMM3), reg(XMM4)),
    ));
    // The four results, read out of the vector.
    for (index, lane) in [first, second, third, fourth].iter().enumerate() {
        entry.insts.push(def_with_operands(
            *lane,
            MType::I32,
            vector_of_four(),
            MOp::VecExtract { index: index as u32 },
            Operands::one(reg(XMM5)),
        ));
        entry.insts.push(def(
            pointer,
            MType::Ptr { size: 4 },
            MOp::Gep {
                size: 4,
                signed: false,
            },
            Operands::two(reg(RDI), imm(index as i64)),
        ));
        entry.insts.push(none(
            MType::Void,
            MOp::Store { size: 4 },
            Operands::two(reg(R15), reg(*lane)),
        ));
    }
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

/// Runs the kernel with the given global identifier and returns the
/// four words it wrote.
fn run(global_id: u32) -> [u32; 4] {
    run_with(TargetCode::Baseline, global_id)
}

/// Runs the kernel under the selected instruction set.
fn run_with(code: TargetCode, global_id: u32) -> [u32; 4] {
    let (module, entry) = kernel_module();
    let image = X86Target::with_code(code)
        .lower_module(&module, entry)
        .expect("the module compiles");
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
    // SAFETY: the page holds a whole number of bytes of executable
    // memory, and the code is shorter than the page.
    unsafe {
        ptr::copy_nonoverlapping(image.bytes().as_ptr(), page.cast::<u8>(), image.bytes().len());
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
        libc::munmap(page, PAGE);
    }
    output
}

#[test]
fn the_lanes_are_added_and_multiplied() {
    let output = run(5);
    // The lanes hold five, six, seven and eight; their sum with the
    // doubled vector is twelve, fourteen, sixteen and eighteen, and
    // the product with three is thirty-six, forty-two, forty-eight and
    // fifty-four.
    assert_eq!(output, [21, 24, 27, 30], "the lanes are wrong");
}

#[test]
fn the_lanes_are_added_and_multiplied_with_avx2() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    // The three-register AVX2 forms compute the same result as the
    // baseline, which they replace when the code allows them.
    let output = run_with(TargetCode::Avx2, 5);
    assert_eq!(output, [21, 24, 27, 30], "the lanes are wrong");
}
