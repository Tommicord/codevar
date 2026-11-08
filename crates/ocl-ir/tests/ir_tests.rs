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

//! Integration tests for the `codevar-ocl-ir` crate.
//!
//! Covers module construction, parse/print round-trip fidelity, verify
//! accepting valid IR and rejecting invalid IR, lower producing correct
//! IR, and type/value correctness.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use codevar_ocl_ir::ir::{
    BinOp, BlockId, BuildError, ConstValue, ExecutionModel, Inst, Linkage, Module, Op, Storage, Target, Type,
};
use codevar_ocl_ir::lower::lower_source;
use codevar_ocl_ir::parse::{ParseErrorKind, parse};
use codevar_ocl_ir::print::print;
use codevar_ocl_ir::spirv::ops;
use codevar_ocl_ir::verify::{VerifyError, verify, verify_function};

fn unwrap<T, E: core::fmt::Debug>(result: Result<T, E>, what: &'static str) -> T {
    result.unwrap_or_else(|e| panic!("{what} failed: {e:?}"))
}

// ---------------------------------------------------------------------------
// Module construction
// ---------------------------------------------------------------------------

#[test]
fn module_construction_builds_types_values_and_functions() {
    let mut module = Module::new(Target::opencl());
    let void = module.void_ty();
    let int = module.int_ty(32, true);
    let float = module.float_ty(32);
    let ptr = module.ptr_ty(Storage::CrossWorkgroup, float);
    let sig = module.fn_ty(void, vec![ptr, ptr]);

    // Types intern structurally equal types.
    let int2 = module.int_ty(32, true);
    assert_eq!(int, int2);
    let ptr2 = module.ptr_ty(Storage::CrossWorkgroup, float);
    assert_eq!(ptr, ptr2);

    // A function signature must be a function type.
    assert_eq!(
        module.add_function("bad", int, Linkage::External),
        Err(BuildError::ExpectedFunctionType)
    );

    // A global needs a pointer type.
    assert_eq!(
        module.add_global("g", int, Linkage::External, None),
        Err(BuildError::NotPointerType)
    );
    let global = unwrap(module.add_global("g", ptr, Linkage::External, None), "add_global");
    assert_eq!(module.type_of(global), ptr);

    let function = unwrap(
        module.add_function("main", sig, Linkage::External),
        "add_function",
    );
    assert!(
        module
            .function(function)
            .unwrap_or_else(|| panic!("missing function"))
            .is_declaration()
    );
    assert_eq!(module.find_function("main"), Some(function));
}

#[test]
fn module_construction_interns_constants_and_rejects_mismatches() {
    let mut module = Module::new(Target::opencl());
    let int = module.int_ty(32, true);
    let float = module.float_ty(32);

    let one = unwrap(module.intern_const(int, ConstValue::Int(1)), "intern_const");
    let one_again = unwrap(module.intern_const(int, ConstValue::Int(1)), "intern_const");
    assert_eq!(one, one_again);
    assert_eq!(module.constant_value(one), Some(ConstValue::Int(1)));

    // Payload must match the type.
    assert_eq!(
        module.intern_const(float, ConstValue::Int(1)),
        Err(BuildError::ConstantTypeMismatch)
    );
    // Payload must fit the width.
    let tiny = module.int_ty(8, true);
    assert_eq!(
        module.intern_const(tiny, ConstValue::Int(0x1FF)),
        Err(BuildError::ConstantTypeMismatch)
    );
}

#[test]
fn module_construction_emits_blocks_and_instructions() {
    let mut module = Module::new(Target::opencl());
    let void = module.void_ty();
    let sig = module.fn_ty(void, Vec::new());
    let function = unwrap(module.add_function("f", sig, Linkage::External), "add_function");

    // Blocks cannot be pushed before a body is started.
    assert_eq!(module.push_block(function, "entry"), Err(BuildError::MissingBody));
    unwrap(module.begin_body(function), "begin_body");
    assert_eq!(module.begin_body(function), Err(BuildError::BodyAlreadyExists));

    let entry = unwrap(module.push_block(function, "entry"), "push_block");
    unwrap(
        module.emit(function, entry, Inst::none(Op::Return)),
        "emit return",
    );

    // Out-of-range block is rejected.
    assert_eq!(
        module.emit(function, BlockId(9), Inst::none(Op::Return)),
        Err(BuildError::BlockOutOfRange { block: BlockId(9) })
    );

    let body = module
        .function(function)
        .unwrap_or_else(|| panic!("missing function"))
        .body
        .as_ref()
        .unwrap_or_else(|| panic!("missing body"));
    assert_eq!(body[0].insts.len(), 1);
    assert!(body[0].insts[0].op.is_terminator());
}

// ---------------------------------------------------------------------------
// ConstValue
// ---------------------------------------------------------------------------

#[test]
fn const_value_fits_matches_type_shape_and_width() {
    let int8 = Type::Int {
        bits: 8,
        signed: true,
    };
    let int32 = Type::Int {
        bits: 32,
        signed: true,
    };
    let float32 = Type::Float { bits: 32 };

    assert!(ConstValue::Int(0).fits(&int8));
    assert!(ConstValue::Int(127).fits(&int8));
    // Zero-extended: 128 fits in 8 bits as raw bits; only 256 overflows.
    assert!(ConstValue::Int(128).fits(&int8));
    assert!(!ConstValue::Int(256).fits(&int8));
    assert!(!ConstValue::Int(-1i64 as u64).fits(&int8));
    assert!(ConstValue::Int(0xFFFF_FFFF).fits(&int32));
    assert!(!ConstValue::Int(0x1_0000_0000).fits(&int32));
    assert!(ConstValue::Float32(0).fits(&float32));
    assert!(!ConstValue::Float32(0).fits(&int32));
    assert!(ConstValue::Undef.fits(&float32));
    assert!(ConstValue::Null.fits(&float32));
    assert!(ConstValue::Bool(true).fits(&Type::Bool));
    assert!(!ConstValue::Bool(true).fits(&int32));
}

#[test]
fn float_nan_payloads_are_canonicalized() {
    assert_eq!(
        ConstValue::from_f32_bits(0x7FC0_0001),
        ConstValue::Float32(0x7FC0_0000)
    );
    assert_eq!(
        ConstValue::from_f64_bits(0x7FF8_0000_0000_0001),
        ConstValue::Float64(0x7FF8_0000_0000_0000)
    );
    assert_eq!(
        ConstValue::from_f32_bits(0x3F80_0000),
        ConstValue::Float32(0x3F80_0000)
    );
}

// ---------------------------------------------------------------------------
// parse -> print round-trip fidelity
// ---------------------------------------------------------------------------

fn roundtrip_module() -> Module {
    let mut module = Module::new(Target::opencl());
    let _set = module.add_ext_inst_set("OpenCL.std");

    let void = module.intern_named_type(Type::Void, "void");
    let int = module.intern_named_type(
        Type::Int {
            bits: 32,
            signed: true,
        },
        "int",
    );
    let float = module.intern_named_type(Type::Float { bits: 32 }, "float");
    let ptr = module.intern_named_type(
        Type::Pointer {
            storage: Storage::CrossWorkgroup,
            pointee: float,
        },
        "ptr_cw_float",
    );
    let sig = module.intern_named_type(
        Type::Function {
            ret: void,
            params: vec![ptr, ptr],
        },
        "fn_void",
    );

    let c0 = unwrap(module.intern_const(int, ConstValue::Int(0)), "intern_const");
    module.set_value_name(c0, "c0");

    let gptr = module.ptr_ty(Storage::CrossWorkgroup, int);
    let g = unwrap(
        module.add_global("g", gptr, Linkage::External, None),
        "add_global",
    );
    let _ = g;

    let function = unwrap(
        module.add_function("main", sig, Linkage::External),
        "add_function",
    );
    unwrap(
        module.set_entry_point(ExecutionModel::Kernel, function, "main"),
        "set_entry_point",
    );
    let args = module
        .function(function)
        .unwrap_or_else(|| panic!("missing function"))
        .args
        .clone();

    unwrap(module.begin_body(function), "begin_body");
    let entry = unwrap(module.push_block(function, "entry"), "push_block");

    let loaded = module.new_inst_value(float);
    module.name_value(loaded, "loaded");
    unwrap(
        module.emit(
            function,
            entry,
            Inst::def(loaded, float, Op::Load { ptr: args[0] }),
        ),
        "emit load",
    );
    unwrap(
        module.emit(
            function,
            entry,
            Inst::none(Op::Store {
                ptr: args[1],
                value: loaded,
            }),
        ),
        "emit store",
    );
    unwrap(
        module.emit(function, entry, Inst::none(Op::Return)),
        "emit return",
    );

    module
}

#[test]
fn parse_print_round_trip_is_faithful() {
    let module = roundtrip_module();
    let text = print(&module);
    let parsed = unwrap(parse(&text), "parse");
    let again = print(&parsed);
    assert_eq!(text, again, "round-trip is not idempotent:\n{text}\n---\n{again}");
}

#[test]
fn parse_print_round_trip_preserves_structure() {
    let module = roundtrip_module();
    let parsed = unwrap(parse(&print(&module)), "parse");

    assert_eq!(parsed.target, module.target);
    assert_eq!(parsed.functions().len(), module.functions().len());
    assert_eq!(parsed.globals().len(), module.globals().len());
    assert_eq!(parsed.types().len(), module.types().len());
    assert_eq!(parsed.values().len(), module.values().len());
    assert_eq!(parsed.entry_points.len(), module.entry_points.len());
    assert_eq!(parsed.ext_inst_sets.len(), module.ext_inst_sets.len());
}

#[test]
fn parse_rejects_text_without_target_line() {
    let error = parse("%void = OpTypeVoid\n").unwrap_err();
    assert_eq!(error.kind, ParseErrorKind::MissingTarget);
}

#[test]
fn parse_rejects_duplicate_ids() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%void = OpTypeVoid
";
    let error = parse(text).unwrap_err();
    assert_eq!(error.kind, ParseErrorKind::DuplicateId);
}

// ---------------------------------------------------------------------------
// verify accepts valid IR and rejects invalid IR
// ---------------------------------------------------------------------------

#[test]
fn verify_accepts_well_formed_kernel() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%ptr_fn_int = OpTypePointer Function %int
%fn_void = OpTypeFunction %void %ptr_cw_float %ptr_cw_float

%c0 = OpConstant %int 0

OpEntryPoint Kernel %vector_add \"vector_add\"

%vector_add = OpFunction %void None %fn_void
    %a = OpFunctionParameter %ptr_cw_float
    %b = OpFunctionParameter %ptr_cw_float
    %entry = OpLabel
    %slot = OpVariable %ptr_fn_int Function
    OpStore %slot %c0
    %loaded = OpLoad %int %slot
    OpStore %slot %loaded
    OpReturn
OpFunctionEnd
";
    let module = unwrap(parse(text), "parse");
    unwrap(verify(&module), "verify");
}

#[test]
fn verify_rejects_type_mismatches() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ptr_fn_int = OpTypePointer Function %int
%fn_void = OpTypeFunction %void

%vf = OpConstant %float 1.5

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    %bad = OpIAdd %int %vf %vf
    OpReturn
OpFunctionEnd
";
    let module = unwrap(parse(text), "parse");
    let error = verify(&module).unwrap_err();
    assert!(
        matches!(
            error,
            VerifyError::WrongKind {
                what: "binary operand",
                expected: "an integer",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn verify_rejects_missing_terminator() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%ptr_fn_int = OpTypePointer Function %int
%fn_void = OpTypeFunction %void %ptr_fn_int

%v1 = OpConstant %int 1

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
    %a = OpFunctionParameter %ptr_fn_int
    %entry = OpLabel
    %sum = OpIAdd %int %v1 %v1
OpFunctionEnd
";
    let module = unwrap(parse(text), "parse");
    let error = verify(&module).unwrap_err();
    assert!(
        matches!(error, VerifyError::MissingTerminator { block } if block == BlockId(0)),
        "unexpected error: {error}"
    );
}

#[test]
fn verify_rejects_unreachable_block() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%fn_void = OpTypeFunction %void

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    OpReturn
    %dead = OpLabel
    OpReturn
OpFunctionEnd
";
    let module = unwrap(parse(text), "parse");
    let error = verify(&module).unwrap_err();
    assert!(
        matches!(error, VerifyError::UnreachableBlock { block } if block == BlockId(1)),
        "unexpected error: {error}"
    );
}

#[test]
fn verify_rejects_declaration_without_import_linkage() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn_void = OpTypeFunction %void

%helper = OpFunction %void None %fn_void
OpFunctionEnd
";
    let module = unwrap(parse(text), "parse");
    let error = verify(&module).unwrap_err();
    assert!(
        matches!(error, VerifyError::DeclarationNotImported { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn verify_function_accepts_taken_body_declaration() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%fn_void = OpTypeFunction %void

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
    %entry = OpLabel
    OpReturn
OpFunctionEnd
";
    let mut module = unwrap(parse(text), "parse");
    let main = module
        .find_function("main")
        .unwrap_or_else(|| panic!("missing main"));
    assert!(module.take_function_body(main).is_some());
    assert!(
        verify(&module).is_err(),
        "whole-module verify wants the body back"
    );
    unwrap(verify_function(&module, main), "verify_function");
}

// ---------------------------------------------------------------------------
// lower produces correct IR
// ---------------------------------------------------------------------------

#[test]
fn lower_source_produces_verifiable_ir() {
    let source = "#[kernel]\nfn zero(out: *mut int) {\n    *out = 0;\n}";
    let module = unwrap(lower_source(source), "lower_source");
    unwrap(verify(&module), "verify");

    // The module should have exactly one entry point / kernel.
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].model, ExecutionModel::Kernel);
    assert_eq!(module.entry_points[0].name, "zero");

    // The kernel function returns void and takes one pointer parameter.
    let kernel = module
        .find_function("zero")
        .unwrap_or_else(|| panic!("missing kernel"));
    let function = module
        .function(kernel)
        .unwrap_or_else(|| panic!("missing kernel payload"));
    assert!(matches!(
        module.ty(function.sig),
        Type::Function { ret, .. } if matches!(module.ty(*ret), Type::Void)
    ));
    assert_eq!(function.args.len(), 1);
}

#[test]
fn lower_source_reports_errors_for_bad_source() {
    // `half` has no IR representation (cl_khr_fp16 not enabled).
    let source = "#[kernel]\nfn half_val(x: half) -> half { x }";
    let error = lower_source(source).unwrap_err();
    assert!(
        matches!(error, codevar_ocl_ir::lower::LowerError::AnalysisFailed { .. }),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// Instruction result shape matches opcode
// ---------------------------------------------------------------------------

#[test]
fn instruction_result_shape_matches_opcode() {
    let mut module = Module::new(Target::opencl());
    let void = module.void_ty();
    let int = module.int_ty(32, true);
    let sig = module.fn_ty(void, Vec::new());
    let function = unwrap(module.add_function("f", sig, Linkage::External), "add_function");
    unwrap(module.begin_body(function), "begin_body");
    let block = unwrap(module.push_block(function, "entry"), "push_block");

    // A result-producing op needs a result value and type.
    let c0 = unwrap(module.intern_const(int, ConstValue::Int(0)), "intern_const");
    let add = Op::Binary {
        op: BinOp::IAdd,
        lhs: c0,
        rhs: c0,
    };
    let sum = module.new_inst_value(int);
    unwrap(
        module.emit(function, block, Inst::def(sum, int, add)),
        "emit binary op",
    );

    // A result-less op carries neither.
    unwrap(
        module.emit(function, block, Inst::none(Op::Return)),
        "emit return",
    );

    // The emitted instructions keep their result/ty shapes.
    let body = module
        .function(function)
        .unwrap_or_else(|| panic!("missing function"))
        .body
        .as_ref()
        .unwrap_or_else(|| panic!("missing body"));
    assert_eq!(body[0].insts[0].result, Some(sum));
    assert_eq!(body[0].insts[0].ty, Some(int));
    assert_eq!(body[0].insts[1].result, None);
    assert_eq!(body[0].insts[1].ty, None);
}

// ---------------------------------------------------------------------------
// SPIR-V opcode tables exist (documented surface)
// ---------------------------------------------------------------------------

#[test]
fn spirv_module_exists_with_opcode_tables() {
    // The `spirv` module exposes only enumerant tables (`ops`). There is
    // no binary emitter function (`to_words`/`emit`) in this crate yet,
    // so SPIR-V emission of a module cannot be exercised directly. This
    // test documents the available surface: core opcode and extended
    // instruction lookups both exist.
    assert_eq!(ops::op_opcode("OpIAdd"), Some(128));
    assert_eq!(ops::ocl_opcode("fabs"), Some(23));
    assert_eq!(ops::op_name(128), "OpIAdd");
    assert_eq!(ops::ocl_name(23), "fabs");
}
