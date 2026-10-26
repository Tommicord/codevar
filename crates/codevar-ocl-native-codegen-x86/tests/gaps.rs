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

//! The three gaps that kept the kernels of the editor from compiling.
//!
//! * A multiplication whose factor is wider than thirty-two bits has
//!   no immediate form: the factor moves into a register first.
//! * An operation that reads a constant has no register to read: the
//!   constant moves into a scratch register first.
//! * An operation that receives a register of the wrong class is
//!   refused with a message that names the register.
//!
//! The first two are checked against the reference interpreter, which
//! computes the expected value for the same module.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::ptr;

use codevar_ocl_abi::{ARG_SLOT, MAX_PARAMS, WorkItem};
use codevar_ocl_ir::ir::{Module, ValueId};
use codevar_ocl_ir::parse::parse;
use codevar_ocl_native_codegen::interp::{Interp, Pointer, Value};
use codevar_ocl_native_codegen::mir::{MOp, MType, MirFunction, MirModule, Operands, SymId, SymbolKind};
use codevar_ocl_native_codegen::pipeline::{CodegenOptions, compile_entry};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The size of the page the code is copied into.
const PAGE: usize = 4096;

/// The value the kernel reads.
const INPUT: u32 = 5;

/// The wide value the kernel reads.
const WIDE_INPUT: u64 = 3;

/// The factor of the wide multiplication: the largest thirty-two-bit
/// value, which does not fit in the signed thirty-two-bit immediate
/// the multiplication has.
const WIDE_FACTOR: u64 = 0xFFFF_FFFF;

/// The module of the test.
///
/// The kernel multiplies a wide value by a wide factor, compares a
/// narrow value with a constant, selects between the value and a
/// constant, adds a constant, and stores a constant.
const MODULE: &str = r#"
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%ulong = OpTypeInt 64 0
%bool = OpTypeBool
%ptr_int = OpTypePointer CrossWorkgroup %int
%ptr_ulong = OpTypePointer CrossWorkgroup %ulong
%fn_void = OpTypeFunction %void %ptr_int %ptr_ulong %ptr_int %ptr_ulong

OpEntryPoint Kernel %main "main"

%c0 = OpConstant %int 0
%c1 = OpConstant %int 1
%c2 = OpConstant %int 2
%c7 = OpConstant %int 7
%c42 = OpConstant %int 42
%factor = OpConstant %ulong 4294967295

%main = OpFunction %void None %fn_void
%in32 = OpFunctionParameter %ptr_int
%in64 = OpFunctionParameter %ptr_ulong
%out32 = OpFunctionParameter %ptr_int
%out64 = OpFunctionParameter %ptr_ulong
%entry = OpLabel
%narrow = OpLoad %int %in32
%wide = OpLoad %ulong %in64
%product = OpIMul %ulong %wide %factor
%pwide = OpPtrAccessChain %ptr_ulong %out64 %c0
OpStore %pwide %product
%same = OpIEqual %bool %narrow %c7
%selected = OpSelect %int %same %narrow %c0
%p0 = OpPtrAccessChain %ptr_int %out32 %c0
OpStore %p0 %selected
%sum = OpIAdd %int %narrow %c7
%p1 = OpPtrAccessChain %ptr_int %out32 %c1
OpStore %p1 %sum
%p2 = OpPtrAccessChain %ptr_int %out32 %c2
OpStore %p2 %c42
OpReturn
OpFunctionEnd
"#;

/// The module of the test.
fn module() -> Module {
    parse(MODULE).expect("the module parses")
}

/// The entry point of the module.
fn entry(module: &Module) -> ValueId {
    module
        .entry_points
        .first()
        .expect("the module has an entry point")
        .func
}

/// The values the interpreter computes.
fn interpreted() -> (Vec<u32>, Vec<u64>) {
    let module = module();
    let mut interp = Interp::new(&module);
    let narrow_input = interp.memory().allocate(4);
    let wide_input = interp.memory().allocate(8);
    let narrow_output = interp.memory().allocate(3 * 4);
    let wide_output = interp.memory().allocate(8);
    interp
        .memory()
        .write(narrow_input, 0, &INPUT.to_le_bytes())
        .expect("the input holds the value");
    interp
        .memory()
        .write(wide_input, 0, &WIDE_INPUT.to_le_bytes())
        .expect("the input holds the value");
    let values = [
        Value::pointer(Pointer::new(narrow_input, 0)),
        Value::pointer(Pointer::new(wide_input, 0)),
        Value::pointer(Pointer::new(narrow_output, 0)),
        Value::pointer(Pointer::new(wide_output, 0)),
    ];
    interp
        .run(entry(&module), &values)
        .expect("the kernel runs");
    let bytes = interp
        .memory()
        .bytes(narrow_output)
        .expect("the buffer holds the values")
        .to_vec();
    let narrow: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let bytes = interp
        .memory()
        .bytes(wide_output)
        .expect("the buffer holds the values")
        .to_vec();
    let wide: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    (narrow, wide)
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
    page.cast::<u8>()
}

/// The values the code of the back end computes.
fn compiled() -> (Vec<u32>, Vec<u64>) {
    let module = module();
    let options = CodegenOptions::default();
    let code = compile_entry(&module, &target(), &options, entry(&module)).expect("the kernel compiles");
    let page = executable_page();
    // SAFETY: the page holds a whole number of bytes of executable
    // memory, and the code is shorter than the page.
    unsafe {
        ptr::copy_nonoverlapping(code.bytes().as_ptr(), page, code.bytes().len());
    }
    let address = page as usize + code.entry_offset() as usize;
    // SAFETY: the page holds the code of the entry function, which
    // follows the calling convention of a kernel.
    let kernel: unsafe extern "C" fn(*mut WorkItem) = unsafe { std::mem::transmute(address) };

    let mut narrow_input = [0u32; 1];
    let mut wide_input = [0u64; 1];
    let mut narrow_output = [0u32; 3];
    let mut wide_output = [0u64; 1];
    narrow_input[0] = INPUT;
    wide_input[0] = WIDE_INPUT;
    let mut block = [0u8; MAX_PARAMS];
    let pointers = [
        narrow_input.as_mut_ptr() as usize,
        wide_input.as_mut_ptr() as usize,
        narrow_output.as_mut_ptr() as usize,
        wide_output.as_mut_ptr() as usize,
    ];
    for (slot, pointer) in pointers.iter().enumerate() {
        let start = slot * ARG_SLOT;
        block[start..start + 8].copy_from_slice(&pointer.to_ne_bytes());
    }
    let mut item = WorkItem {
        global_id: [0; 3],
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
        libc::munmap(page.cast::<libc::c_void>(), PAGE);
    }
    (narrow_output.to_vec(), wide_output.to_vec())
}

#[test]
fn the_operations_that_read_a_constant_compute_the_values_the_interpreter_computes() {
    let (expected_narrow, expected_wide) = interpreted();
    let (narrow, wide) = compiled();
    assert_eq!(narrow, expected_narrow, "the narrow values differ");
    assert_eq!(wide, expected_wide, "the wide values differ");
}

#[test]
fn the_operations_compute_the_values_they_should() {
    let (narrow, wide) = compiled();
    // The selection takes the second value, because the comparison is
    // false.
    assert_eq!(narrow[0], 0, "the selection is wrong");
    // The sum of the value and seven.
    assert_eq!(narrow[1], INPUT + 7, "the sum is wrong");
    // The constant that is stored.
    assert_eq!(narrow[2], 42, "the stored constant is wrong");
    // The product of the wide value and the wide factor.
    assert_eq!(
        wide[0],
        WIDE_INPUT * WIDE_FACTOR,
        "the product of the wide factor is wrong"
    );
}

/// The type of a vector of four thirty-two-bit lanes.
fn vector_of_four() -> MType {
    MType::Vec {
        lanes: 4,
        elem: Box::new(MType::I32),
    }
}

/// Builds a kernel that broadcasts a constant over the lanes of a
/// vector and reads one lane back.
fn broadcast_module() -> (MirModule, SymId) {
    let mut function = MirFunction::new("kernel");
    let mut entry = block("entry");
    let item = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let base = vreg(&mut function, R15, MType::Ptr { size: 4 });
    let output = vreg(&mut function, RBX, MType::Ptr { size: 4 });
    let splat = vreg(&mut function, XMM0, vector_of_four());
    let lane = vreg(&mut function, R11, MType::I32);
    entry.insts.push(def(
        item,
        MType::Ptr { size: 4 },
        MOp::Param { index: 0 },
        Operands::none(),
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
        Operands::one(reg(R15)),
    ));
    entry
        .insts
        .push(def(splat, vector_of_four(), MOp::VecSplat, Operands::one(imm(7))));
    entry.insts.push(def_with_operands(
        lane,
        MType::I32,
        vector_of_four(),
        MOp::VecExtract { index: 2 },
        Operands::one(reg(XMM0)),
    ));
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

#[test]
fn a_broadcast_of_a_constant_fills_every_lane() {
    let (module, entry) = broadcast_module();
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
        global_id: [0; 3],
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
        libc::munmap(page.cast::<libc::c_void>(), PAGE);
    }
    assert_eq!(output[0], 7, "the lane does not hold the constant");
}

#[test]
fn an_operation_with_a_register_of_the_wrong_class_is_refused() {
    let mut function = MirFunction::new("broken");
    let mut entry = block("entry");
    vreg(&mut function, XMM0, MType::I32);
    vreg(&mut function, R11, MType::I32);
    let sum = vreg(&mut function, R12, MType::I32);
    // An integer addition that reads a value held in a vector
    // register.
    entry.insts.push(def_with_operands(
        sum,
        MType::I32,
        MType::I32,
        MOp::Add,
        Operands::two(reg(XMM0), reg(R11)),
    ));
    entry
        .insts
        .push(none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(entry);
    let mut module = MirModule::new();
    let symbol = module.add_symbol("broken", SymbolKind::Function { index: 0 }, false);
    function.symbol = symbol;
    module.functions.push(function);

    let error = target()
        .lower_module(&module, symbol)
        .expect_err("the operation is refused");
    let message = format!("{error}");
    assert!(
        message.contains("xmm0"),
        "the message does not name the register: {message}"
    );
}
