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

//! The execution of the mathematical operations.
//!
//! The kernel of the test computes a square root, an absolute value,
//! a minimum, a maximum, a remainder, a positive difference, a sign
//! copy and the two conversions between a floating-point value and an
//! unsigned sixty-four-bit integer, and writes them into the buffers
//! its arguments point at.
//!
//! The interpreter is the oracle: the very same module is run by the
//! interpreter and by the code the back end produces, and the two
//! runs must agree bit for bit.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::ptr;

use codevar_ocl_abi::{ARG_SLOT, MAX_PARAMS, WorkItem};
use codevar_ocl_ir::ir::{Module, ValueId};
use codevar_ocl_ir::parse::parse;
use codevar_ocl_native_codegen::interp::{Interp, Pointer, Value};
use codevar_ocl_native_codegen::opt::{OptConfig, Optimizations};
use codevar_ocl_native_codegen::pipeline::{CodegenOptions, compile_entry};
use codevar_ocl_native_codegen_x86::target;

/// The size of the page the code is copied into.
const PAGE: usize = 4096;

/// The number of floating-point values the kernel writes.
const FLOATS: usize = 8;

/// The number of sixty-four-bit integers the kernel writes.
const WORDS: usize = 2;

/// The module of the test.
///
/// The kernel writes eight floating-point values and two
/// sixty-four-bit integers, one after the other.
const MODULE: &str = r#"
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ulong = OpTypeInt 64 0
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%ptr_cw_ulong = OpTypePointer CrossWorkgroup %ulong
%fn_void = OpTypeFunction %void %ptr_cw_float %ptr_cw_ulong

%OpenCL.std = OpExtInstImport "OpenCL.std"

%i0 = OpConstant %int 0
%i1 = OpConstant %int 1
%i2 = OpConstant %int 2
%i3 = OpConstant %int 3
%i4 = OpConstant %int 4
%i5 = OpConstant %int 5
%i6 = OpConstant %int 6
%i7 = OpConstant %int 7
%ftwo = OpConstant %float 2.0
%fneg = OpConstant %float -2.5
%fbig = OpConstant %float 1.0e19
%uval = OpConstant %ulong 12345678901234567890

OpEntryPoint Kernel %main "main"

%main = OpFunction %void None %fn_void
%out = OpFunctionParameter %ptr_cw_float
%uout = OpFunctionParameter %ptr_cw_ulong
%entry = OpLabel
%root = OpExtInst %float %OpenCL.std 61 %ftwo
%absolute = OpExtInst %float %OpenCL.std 23 %fneg
%smaller = OpExtInst %float %OpenCL.std 28 %ftwo %fneg
%larger = OpExtInst %float %OpenCL.std 27 %ftwo %fneg
%remainder = OpFRem %float %ftwo %fneg
%positive = OpExtInst %float %OpenCL.std 24 %ftwo %fneg
%sign = OpExtInst %float %OpenCL.std 13 %ftwo %fneg
%wide = OpConvertFToU %ulong %fbig
%back = OpConvertUToF %float %uval
%p0 = OpPtrAccessChain %ptr_cw_float %out %i0
OpStore %p0 %root
%p1 = OpPtrAccessChain %ptr_cw_float %out %i1
OpStore %p1 %absolute
%p2 = OpPtrAccessChain %ptr_cw_float %out %i2
OpStore %p2 %smaller
%p3 = OpPtrAccessChain %ptr_cw_float %out %i3
OpStore %p3 %larger
%p4 = OpPtrAccessChain %ptr_cw_float %out %i4
OpStore %p4 %remainder
%p5 = OpPtrAccessChain %ptr_cw_float %out %i5
OpStore %p5 %positive
%p6 = OpPtrAccessChain %ptr_cw_float %out %i6
OpStore %p6 %sign
%p7 = OpPtrAccessChain %ptr_cw_float %out %i7
OpStore %p7 %back
%q0 = OpPtrAccessChain %ptr_cw_ulong %uout %i0
OpStore %q0 %wide
%q1 = OpPtrAccessChain %ptr_cw_ulong %uout %i1
OpStore %q1 %uval
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
fn interpreted() -> (Vec<f32>, Vec<u64>) {
    let module = module();
    let mut interp = Interp::new(&module);
    let floats = interp.memory().allocate(FLOATS * 4);
    let words = interp.memory().allocate(WORDS * 8);
    let values = [
        Value::pointer(Pointer::new(floats, 0)),
        Value::pointer(Pointer::new(words, 0)),
    ];
    interp
        .run(entry(&module), &values)
        .expect("the kernel runs");
    let bytes = interp
        .memory()
        .bytes(floats)
        .expect("the buffer holds the values")
        .to_vec();
    let mut floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let bytes = interp
        .memory()
        .bytes(words)
        .expect("the buffer holds the values")
        .to_vec();
    let mut words: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    // The values are read in the order the kernel writes them.
    floats.rotate_left(0);
    words.rotate_left(0);
    (floats, words)
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

/// The values the code of the back end computes.
fn compiled() -> (Vec<f32>, Vec<u64>) {
    let module = module();
    let options = CodegenOptions {
        opt: OptConfig {
            optimizations: Optimizations::NONE,
            ..OptConfig::default()
        },
        ..CodegenOptions::default()
    };
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

    let mut floats = [0f32; FLOATS];
    let mut words = [0u64; WORDS];
    let mut block = [0u8; MAX_PARAMS];
    let float_pointer = floats.as_mut_ptr() as usize;
    let word_pointer = words.as_mut_ptr() as usize;
    block[0..8].copy_from_slice(&float_pointer.to_ne_bytes());
    block[ARG_SLOT..ARG_SLOT + 8].copy_from_slice(&word_pointer.to_ne_bytes());
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
        libc::munmap(page as *mut libc::c_void, PAGE);
    }
    (floats.to_vec(), words.to_vec())
}

#[test]
fn the_mathematical_operations_compute_the_values_the_interpreter_computes() {
    let (expected_floats, expected_words) = interpreted();
    let (floats, words) = compiled();
    assert_eq!(floats, expected_floats, "the floating-point values differ");
    assert_eq!(words, expected_words, "the integers differ");
}

#[test]
fn the_operations_compute_the_values_they_should() {
    let (floats, words) = compiled();
    // The square root of two.
    assert_eq!(floats[0], 2.0f32.sqrt(), "the square root is wrong");
    // The absolute value of minus two and a half.
    assert_eq!(floats[1], 2.5, "the absolute value is wrong");
    // The smaller of two and minus two and a half.
    assert_eq!(floats[2], -2.5, "the minimum is wrong");
    // The larger of the two.
    assert_eq!(floats[3], 2.0, "the maximum is wrong");
    // The remainder of a division that is truncated towards zero.
    assert_eq!(floats[4], 2.0, "the remainder is wrong");
    // The positive difference of the two.
    assert_eq!(floats[5], 4.5, "the positive difference is wrong");
    // The magnitude of two with the sign of minus two and a half.
    assert_eq!(floats[6], -2.0, "the sign copy is wrong");
    // The value of a large unsigned integer as a float.
    assert_eq!(
        floats[7], 12345678901234567890u64 as f32,
        "the conversion of an unsigned integer is wrong"
    );
    // The value of a large float as an unsigned integer.
    assert_eq!(
        words[0], 1.0e19f32 as u64,
        "the conversion into an unsigned integer is wrong"
    );
    // The unsigned integer itself.
    assert_eq!(words[1], 12345678901234567890, "the integer is wrong");
}
