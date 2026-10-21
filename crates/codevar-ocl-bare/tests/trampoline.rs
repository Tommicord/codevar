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

//! The trampoline, end to end: machine code in, a kernel call out.
//!
//! The test only makes sense where the trampoline can allocate
//! executable memory — a hosted Unix host — and where the hand-written
//! machine code is valid: x86-64.

#![cfg(all(feature = "std", unix, target_arch = "x86_64"))]

use codevar_ocl_bare::exec::{ArgBlock, Launch, SpirvBinding, run_all};
use codevar_ocl_bare::{Trampoline, WorkItem};

/// `mov dword ptr [rdi], 42; ret` — the kernel ABI passes the work item
/// in `rdi`, so this writes 42 into the first word of the work item,
/// which is `global_id[0]`.
const WRITE_42: &[u8] = &[
    0xC7, 0x07, 0x2A, 0x00, 0x00, 0x00, // mov dword ptr [rdi], 42
    0xC3, // ret
];

/// `mov eax, 42; ret` — returns 42 in the return register.
const RETURN_42: &[u8] = &[
    0xB8, 0x2A, 0x00, 0x00, 0x00, // mov eax, 42
    0xC3, // ret
];

/// A work item with no arguments.
fn work_item() -> WorkItem {
    WorkItem {
        global_id: [0; 3],
        local_id: [0; 3],
        group_id: [0; 3],
        global_offset: [0; 3],
        num_groups: [0; 3],
        local_size: [0; 3],
        arg_base: core::ptr::null_mut(),
    }
}

#[test]
fn a_trampoline_calls_the_machine_code() {
    let trampoline = Trampoline::new(WRITE_42, 0).expect("the code is mapped");
    let mut item = work_item();
    trampoline.call(&mut item);
    assert_eq!(item.global_id[0], 42);
    assert_eq!(item.global_id[1], 0);
}

#[test]
fn the_entry_offset_selects_the_entry_point() {
    // A `nop` sled in front of the kernel: entering at offset 4 must
    // reach the same code.
    let mut code = vec![0x90; 4];
    code.extend_from_slice(WRITE_42);
    let trampoline = Trampoline::new(&code, 4).expect("the code is mapped");
    let mut item = work_item();
    trampoline.call(&mut item);
    assert_eq!(item.global_id[0], 42);
}

#[test]
fn the_entry_point_can_be_taken_as_a_kernel_function() {
    let trampoline = Trampoline::new(WRITE_42, 0).expect("the code is mapped");
    let kernel = trampoline.kernel_fn();
    let mut item = work_item();
    // SAFETY: the trampoline holds the code of a kernel that follows
    // the ABI of `codevar-ocl-abi`; the work item is live and exclusive.
    unsafe { kernel(&mut item) };
    assert_eq!(item.global_id[0], 42);
}

#[test]
fn the_driver_runs_a_trampoline_as_a_native_kernel() {
    let trampoline = Trampoline::new(WRITE_42, 0).expect("the code is mapped");
    let args = ArgBlock::new();
    let launch = Launch {
        kernel: trampoline.kernel_fn(),
        name: "write_42",
        binding: SpirvBinding::NONE,
        args: &args,
        dims: 1,
        offset: [0, 0, 0],
        global: [4, 1, 1],
        local: [2, 1, 1],
    };
    // SAFETY: the kernel pointer is the trampoline's entry point and the
    // argument block matches the (argument-less) kernel.
    unsafe { run_all(&launch) }.expect("the launch runs");
}

#[test]
fn a_return_value_comes_back_in_the_return_register() {
    let trampoline = Trampoline::new(RETURN_42, 0).expect("the code is mapped");
    let kernel = trampoline.kernel_fn();
    let mut item = work_item();
    // SAFETY: the code is a plain `mov eax, 42; ret`; it touches neither
    // the work item nor the stack beyond the call.
    let value = unsafe {
        let typed: extern "C" fn(*mut WorkItem) -> u32 = core::mem::transmute(kernel);
        typed(&mut item)
    };
    assert_eq!(value, 42);
}

#[test]
fn a_bad_entry_offset_is_refused() {
    let error = Trampoline::new(WRITE_42, 64).expect_err("the entry is outside the code");
    assert!(matches!(
        error,
        codevar_ocl_bare::TrampolineError::BadEntryOffset { .. }
    ));
}
