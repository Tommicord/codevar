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

//! The execution of a call between two functions of a module.
//!
//! The module of the test holds two functions: the first one adds
//! two numbers, the second one calls it and writes the result into
//! the buffer its argument points at.  The call reaches the callee
//! with a relative displacement, so the code holds no address at all.

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

/// Builds the module of the test: a function that adds two numbers and
/// a kernel that calls it.
fn module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut adder = MirFunction::new("adder");
    let mut entry = block("entry");
    let left = vreg(&mut adder, RBX, MType::I64);
    let right = vreg(&mut adder, R10, MType::I64);
    let sum = vreg(&mut adder, R11, MType::I64);
    entry
        .insts
        .push(def(left, MType::I64, MOp::Param { index: 0 }, Operands::none()));
    entry
        .insts
        .push(def(right, MType::I64, MOp::Param { index: 1 }, Operands::none()));
    entry
        .insts
        .push(def(sum, MType::I64, MOp::Add, Operands::two(reg(RBX), reg(R10))));
    entry
        .insts
        .push(def(sum, MType::I64, MOp::Ret, Operands::one(reg(R11))));
    adder.push_block(entry);

    let mut kernel = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut kernel, RBX, MType::Ptr { size: 4 });
    let value = vreg(&mut kernel, R10, MType::I32);
    let base = vreg(&mut kernel, R15, MType::Ptr { size: 4 });
    let output = vreg(&mut kernel, RBX, MType::Ptr { size: 4 });
    let wide = vreg(&mut kernel, R12, MType::I64);
    let result = vreg(&mut kernel, R13, MType::I64);
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
        .push(def(wide, MType::I64, MOp::SExt, Operands::one(reg(R10))));
    entry.insts.push(def(
        result,
        MType::I64,
        MOp::Call {
            callee: codevar_ocl_native_codegen::mir::SymId::new(0),
        },
        Operands::two(reg(R12), reg(R10)),
    ));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 8 },
        Operands::two(reg(RBX), reg(R13)),
    ));
    entry
        .insts
        .push(none(MType::Void, MOp::RetVoid, Operands::none()));
    kernel.push_block(entry);

    let mut module = MirModule::new();
    let adder_symbol = module.add_symbol("adder", SymbolKind::Function { index: 0 }, false);
    let kernel_symbol = module.add_symbol("kernel", SymbolKind::Function { index: 1 }, false);
    adder.symbol = adder_symbol;
    kernel.symbol = kernel_symbol;
    module.functions.push(adder);
    module.functions.push(kernel);
    (module, kernel_symbol)
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
/// word it wrote.
fn run(global_id: u32) -> u64 {
    let (module, entry) = module();
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

    let mut output = [0u64; 4];
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
fn the_call_reaches_the_callee_and_its_result_comes_back() {
    assert_eq!(run(5), 10, "the call computed the wrong sum");
    assert_eq!(run(40), 80, "the call computed the wrong sum");
}
