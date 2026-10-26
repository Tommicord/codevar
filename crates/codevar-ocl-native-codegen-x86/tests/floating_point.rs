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

//! The execution of the floating-point operations.
//!
//! The kernel of the tests turns the global identifier into a
//! floating-point value, adds a constant to it, and writes the bits
//! of the result into the buffer its argument points at.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]
mod common;

use std::ptr;

use codevar_ocl_abi::{MAX_PARAMS, WorkItem, WorkItemField};
use codevar_ocl_native_codegen::mir::{
    ArgKind, MOp, MType, MirFunction, MirModule, Operand, Operands, SymbolKind,
};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The size of the page the code is copied into.
const PAGE: usize = 4096;

/// The bits of one and a half, in single precision.
const ONE_AND_A_HALF: u64 = 0x3FC0_0000;

/// Builds the kernel of the tests.
fn kernel_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut function = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let value = vreg(&mut function, R10, MType::I32);
    let base = vreg(&mut function, R15, MType::Ptr { size: 4 });
    let output = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let converted = vreg(&mut function, XMM0, MType::F32);
    let constant = vreg(&mut function, XMM1, MType::F32);
    let sum = vreg(&mut function, XMM2, MType::F32);
    let bits = vreg(&mut function, R11, MType::I32);

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
    entry
        .insts
        .push(def(converted, MType::F32, MOp::UIToFP, Operands::one(reg(R10))));
    entry.insts.push(def(
        constant,
        MType::F32,
        MOp::Copy,
        Operands::one(Operand::FloatImmediate(ONE_AND_A_HALF)),
    ));
    entry.insts.push(def(
        sum,
        MType::F32,
        MOp::FAdd,
        Operands::two(reg(XMM0), reg(XMM1)),
    ));
    entry
        .insts
        .push(def(bits, MType::I32, MOp::Bitcast, Operands::one(reg(XMM2))));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(RBX), reg(R11)),
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
/// bits it wrote.
fn run(global_id: u32) -> u32 {
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
    output[0]
}

#[test]
fn the_floating_point_sum_is_computed() {
    let bits = run(5);
    let value = f32::from_bits(bits);
    assert_eq!(value, 6.5, "the sum is wrong");
}

#[test]
fn the_sum_of_a_large_identifier_is_computed() {
    let bits = run(1000);
    let value = f32::from_bits(bits);
    assert_eq!(value, 1001.5, "the sum is wrong");
}
