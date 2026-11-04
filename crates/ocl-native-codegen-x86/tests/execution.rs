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

//! The execution of the compiled code.
//!
//! The host the tests run on is an x86-64 system, so the code the
//! back end produces can be copied into a page of executable memory
//! and entered there.  The kernel of the test follows the calling
//! convention of a real kernel: it takes the work item, reads the
//! global identifier out of it, reaches its argument through the
//! argument block, and writes the result into the buffer the
//! argument points at.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]
mod common;

use std::ptr;

use codevar_ocl_abi::{MAX_PARAMS, WorkItem};
use codevar_ocl_native_codegen::mir::{MOp, MType, MirFunction, MirModule, Operands, SymbolKind};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The size of the page the code is copied into.
const PAGE: usize = 4096;

/// Builds a kernel that reads the first component of the global
/// identifier, doubles it, adds one, and writes the result into the
/// buffer its first argument points at.
fn kernel_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut function = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let value = vreg(&mut function, R10, MType::I32);
    let base = vreg(&mut function, R11, MType::Ptr { size: 4 });
    let output = vreg(&mut function, R12, MType::Ptr { size: 4 });
    let doubled = vreg(&mut function, R13, MType::I32);
    let result = vreg(&mut function, R14, MType::I32);
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
            field: codevar_ocl_abi::WorkItemField::GlobalId,
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
            kind: codevar_ocl_native_codegen::mir::ArgKind::Pointer,
        },
        Operands::one(reg(R11)),
    ));
    entry.insts.push(def(
        doubled,
        MType::I32,
        MOp::Mul,
        Operands::two(reg(R10), imm(2)),
    ));
    entry
        .insts
        .push(def(result, MType::I32, MOp::Add, Operands::two(reg(R13), imm(1))));
    entry.insts.push(none(
        MType::Void,
        MOp::Store { size: 4 },
        Operands::two(reg(R12), reg(R14)),
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

#[test]
fn the_kernel_computes_the_expected_value() {
    let (module, entry) = kernel_module();
    let image = target()
        .lower_module(&module, entry)
        .expect("the module compiles");
    assert!(!image.bytes().is_empty());

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
        global_id: [5, 0, 0],
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
    assert_eq!(output[0], 11, "the kernel computed the wrong value");

    // SAFETY: the page was mapped by this test.
    unsafe {
        libc::munmap(page as *mut libc::c_void, PAGE);
    }
}

#[test]
fn the_kernel_reads_every_component_of_the_global_identifier() {
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
    // SAFETY: the page holds the code of the entry function.
    let kernel: unsafe extern "C" fn(*mut WorkItem) = unsafe { std::mem::transmute(address) };

    let mut output = [0u32; 4];
    let mut block = [0u8; MAX_PARAMS];
    let pointer = output.as_mut_ptr() as usize;
    block[0..8].copy_from_slice(&pointer.to_ne_bytes());
    let mut item = WorkItem {
        global_id: [0, 0, 0],
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
    assert_eq!(output[0], 1, "the kernel computed the wrong value");

    // SAFETY: the page was mapped by this test.
    unsafe {
        libc::munmap(page as *mut libc::c_void, PAGE);
    }
}
