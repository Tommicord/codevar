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

//! Equivalence tests for the optimization and the vectorization pass.
//!
//! The interpreter in [`codevar_ocl_native_codegen::interp`] is the
//! oracle: a kernel is lowered once, and the module is then run as it
//! was lowered, as the optimizer rewrote it, and as the vectorizer
//! rewrote it.  All three runs must agree, byte for byte.
//!
//! Every test spells out the values the kernel must produce as well, so
//! the suite keeps its meaning even when a pass leaves the module alone:
//! a pass that is still a stub changes nothing and the comparison is
//! trivially true.
//!
//! A failure means the pass changed the meaning of the program, not that
//! the interpreter is wrong: the run of the original module is checked
//! against the expected values in the same test, so the two sides of the
//! comparison can be told apart.

use codevar_ocl_ir::ir::Module;
use codevar_ocl_ir::lower::lower_source;
use codevar_ocl_ir::parse::parse;
use codevar_ocl_ir::verify::verify;
use codevar_ocl_native_codegen::interp::{Interp, Pointer, Value, WorkItem};
use codevar_ocl_native_codegen::opt::{OptConfig, optimize};
use codevar_ocl_native_codegen::vecz::{VectorConfig, vectorize};

/// Unwraps a result, reporting the failure as a panic.
///
/// The workspace denies `unwrap` and `expect`, so the tests spell the
/// failure out instead.
fn unwrap<T, E: std::fmt::Display>(result: Result<T, E>, what: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{what}: {error}"),
    }
}

/// Unwraps an option, reporting the failure as a panic.
fn some<T>(option: Option<T>, what: &str) -> T {
    match option {
        Some(value) => value,
        None => panic!("{what}"),
    }
}

/// One argument of a kernel run.
#[derive(Debug, Clone)]
enum Arg {
    /// A scalar argument.
    Scalar(i64),
    /// A buffer argument, with the bytes it starts with.
    Buffer(Vec<u8>),
}

/// What a kernel run produced.
#[derive(Debug, Clone, PartialEq)]
struct Outcome {
    /// The value the kernel returned.
    value: Value,
    /// The bytes of every buffer the kernel was given, in order.
    buffers: Vec<Vec<u8>>,
}

/// Runs `kernel` of `module` and collects what it produced.
///
/// The buffers are allocated inside the interpreter's memory and copied
/// out when the run is over, so the caller sees the final bytes.
fn run(module: &Module, kernel: &str, args: &[Arg], work_item: Option<WorkItem>, label: &str) -> Outcome {
    let entry = some(
        module.find_function(kernel),
        "the module has the kernel the test names",
    );
    let mut interp = Interp::new(module);
    // A runaway loop must not hold the suite up for minutes.
    interp.set_step_limit(1_000_000);
    if let Some(state) = work_item {
        unwrap(
            interp.set_work_item(state),
            "the work-item state fits the built-ins",
        );
    }
    let mut values = Vec::with_capacity(args.len());
    let mut buffers = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            Arg::Scalar(number) => values.push(Value::int(*number)),
            Arg::Buffer(bytes) => {
                let alloc = interp.memory().allocate(bytes.len());
                unwrap(
                    interp.memory().write(alloc, 0, bytes),
                    "the buffer holds its own bytes",
                );
                buffers.push(alloc);
                values.push(Value::pointer(Pointer::new(alloc, 0)));
            }
        }
    }
    let value = interp
        .run(entry, &values)
        .unwrap_or_else(|error| panic!("the {label} module runs the kernel: {error}"));
    let contents = buffers
        .iter()
        .map(|alloc| unwrap(interp.memory().bytes(*alloc), "the buffer is there").to_vec())
        .collect();
    Outcome {
        value,
        buffers: contents,
    }
}

/// Lowers `source`, runs the kernel on the original module and on the
/// rewritten ones, and checks that the three runs agree.
///
/// The expected bytes are returned so the caller can spell out what the
/// kernel must compute.
fn check_equivalence(source: &str, kernel: &str, args: &[Arg], work_item: Option<WorkItem>) -> Outcome {
    let original = lower_source(source).unwrap_or_else(|error| panic!("the source lowers: {error:?}"));
    let mut optimized = original.clone();
    unwrap(
        optimize(&mut optimized, &OptConfig::default()),
        "the optimizer runs",
    );
    // The pipeline runs the vectorizer on the optimized module, so the
    // suite does the same: the vectorizer sees the shape the optimizer
    // left behind.
    let mut vectorized = optimized.clone();
    unwrap(
        vectorize(&mut vectorized, &VectorConfig::default()),
        "the vectorizer runs",
    );

    // A pass that leaves a module the verifier rejects has not produced a
    // valid module at all; the interpreter cannot judge it, so the
    // failure is reported where it happens.
    if let Err(error) = verify(&optimized) {
        panic!("the optimizer produced a module the verifier rejects: {error}");
    }
    if let Err(error) = verify(&vectorized) {
        panic!("the vectorizer produced a module the verifier rejects: {error}");
    }

    let base = run(&original, kernel, args, work_item, "original");
    let optimized_run = run(&optimized, kernel, args, work_item, "optimized");
    let vectorized_run = run(&vectorized, kernel, args, work_item, "vectorized");

    // The passes are stubs until they land; the comparison below is
    // trivially true then, and it is the real check once they rewrite
    // the module.
    let optimizer_changed = optimized != original;
    let vectorizer_changed = vectorized != original;
    assert_eq!(
        base.buffers, optimized_run.buffers,
        "the optimizer changed the result (the module changed: {optimizer_changed})"
    );
    assert_eq!(
        base.buffers, vectorized_run.buffers,
        "the vectorizer changed the result (the module changed: {vectorizer_changed})"
    );
    assert_eq!(
        base.value, optimized_run.value,
        "the optimizer changed the result of the kernel"
    );
    assert_eq!(
        base.value, vectorized_run.value,
        "the vectorizer changed the result of the kernel"
    );
    base
}

/// The bytes of one `int` in little-endian order.
fn int_bytes(value: i32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// The bytes of one `float` in little-endian order.
fn float_bytes(value: f32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[test]
fn a_loop_with_an_accumulator_computes_the_same_sum() {
    let source = "\
#[kernel]
fn accumulate(out: *mut int, n: int) {
    let mut total = 0;
    let mut i = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    out[0] = total;
}
";
    let outcome = check_equivalence(
        source,
        "accumulate",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(5)],
        None,
    );
    // The loop sums 0 + 1 + 2 + 3 + 4.
    assert_eq!(outcome.buffers, vec![int_bytes(10)]);
}

#[test]
fn a_loop_over_an_array_writes_the_same_bytes() {
    let source = "\
#[kernel]
fn double_values(out: *mut int) {
    let mut values = [1, 2, 3, 4];
    let mut i = 0;
    while i < 4 {
        values[i] = values[i] * 2;
        i = i + 1;
    }
    out[0] = values[0];
    out[1] = values[1];
    out[2] = values[2];
    out[3] = values[3];
}
";
    let outcome = check_equivalence(source, "double_values", &[Arg::Buffer(vec![0; 16])], None);
    assert_eq!(
        outcome.buffers,
        vec![[int_bytes(2), int_bytes(4), int_bytes(6), int_bytes(8)].concat()]
    );
}

#[test]
fn an_if_else_takes_the_same_branch() {
    let source = "\
#[kernel]
fn classify(out: *mut int, x: int) {
    if x > 0 {
        out[0] = 1;
    } else {
        out[0] = -1;
    }
}
";
    let positive = check_equivalence(
        source,
        "classify",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(7)],
        None,
    );
    assert_eq!(positive.buffers, vec![int_bytes(1)]);

    let negative = check_equivalence(
        source,
        "classify",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(-7)],
        None,
    );
    assert_eq!(negative.buffers, vec![int_bytes(-1)]);
}

#[test]
fn a_call_to_a_helper_function_gives_the_same_answer() {
    let source = "\
fn square_plus_one(x: int) -> int {
    x * x + 1
}

#[kernel]
fn call_helper(out: *mut int, x: int) {
    out[0] = square_plus_one(x);
}
";
    let outcome = check_equivalence(
        source,
        "call_helper",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(6)],
        None,
    );
    assert_eq!(outcome.buffers, vec![int_bytes(6 * 6 + 1)]);
}

#[test]
fn a_kernel_that_uses_the_global_id_scales_its_own_slot() {
    let source = "\
#[kernel]
fn scale(out: *mut int, factor: int) {
    let i = get_global_id(0) as int;
    out[i] = out[i] * factor;
}
";
    let mut current = [1i32, 2, 3, 4]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    for lane in 0..4u64 {
        let state = WorkItem {
            global_id: [lane, 0, 0],
            local_id: [lane, 0, 0],
            global_size: [4, 1, 1],
            local_size: [4, 1, 1],
            num_groups: [1, 1, 1],
            ..WorkItem::default()
        };
        let outcome = check_equivalence(
            source,
            "scale",
            &[Arg::Buffer(current.clone()), Arg::Scalar(3)],
            Some(state),
        );
        assert_eq!(outcome.buffers.len(), 1);
        current = outcome.buffers[0].clone();
    }
    // Every work item scaled its own slot by three.
    assert_eq!(
        current,
        [3i32, 6, 9, 12]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    );
}

#[test]
fn a_floating_point_kernel_computes_the_same_value() {
    let source = "\
#[kernel]
fn compute(out: *mut float, x: float) {
    out[0] = sqrt(x) + sin(x) * 2.0;
}
";
    let outcome = check_equivalence(
        source,
        "compute",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(0)],
        None,
    );
    // The argument is zero, so the result is zero as well; the point of
    // the test is that the three runs agree bit for bit.
    assert_eq!(outcome.buffers, vec![float_bytes(0.0)]);
}

#[test]
fn a_floating_point_kernel_computes_the_expected_value() {
    let source = "\
#[kernel]
fn compute(out: *mut float, x: float) {
    out[0] = sqrt(x) + sin(x) * 2.0;
}
";
    let outcome = check_equivalence(
        source,
        "compute",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(2)],
        None,
    );
    let computed = f32::from_le_bytes(unwrap(
        outcome.buffers[0][0..4].try_into(),
        "the float takes four bytes",
    ));
    let expected = 2.0f32.sqrt() + 2.0f32.sin() * 2.0;
    assert!(
        (computed - expected).abs() < 1e-5,
        "the kernel computed {computed}, expected {expected}"
    );
}

#[test]
fn a_caller_buffer_is_written_in_place() {
    let source = "\
#[kernel]
fn fill(out: *mut int) {
    out[0] = 42;
}
";
    let module = unwrap(lower_source(source), "the source lowers");
    let kernel = some(module.find_function("fill"), "the kernel exists");
    let mut buffer = [0u8; 4];
    {
        let mut interp = Interp::new(&module);
        let alloc = interp.memory().external(&mut buffer);
        unwrap(
            interp.run(kernel, &[Value::pointer(Pointer::new(alloc, 0))]),
            "the kernel runs",
        );
        assert_eq!(
            unwrap(interp.memory().bytes(alloc), "the buffer is there"),
            &[42, 0, 0, 0]
        );
    }
    assert_eq!(buffer, [42, 0, 0, 0]);
}

#[test]
fn a_hand_written_module_runs_to_the_expected_value() {
    // A module with a constant-folded expression, a branch on a known
    // condition, and a loop with a known trip count.
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%bool = OpTypeBool
%ptr_cw_int = OpTypePointer CrossWorkgroup %int
%ptr_fn_int = OpTypePointer Function %int
%fn_void = OpTypeFunction %void %ptr_cw_int %int

%c0 = OpConstant %int 0
%c1 = OpConstant %int 1
%c2 = OpConstant %int 2
%c10 = OpConstant %int 10

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
%out = OpFunctionParameter %ptr_cw_int
%n = OpFunctionParameter %int
%entry = OpLabel
%folded = OpIMul %int %c2 %c10
%total = OpIAdd %int %folded %c1
%cond = OpSGreaterThan %bool %n %c0
OpSelectionMerge %merge None
OpBranchConditional %cond %then %else
%then = OpLabel
%addr0 = OpPtrAccessChain %ptr_cw_int %out %c0
OpStore %addr0 %total
OpBranch %merge
%else = OpLabel
%addr1 = OpPtrAccessChain %ptr_cw_int %out %c0
OpStore %addr1 %c0
OpBranch %merge
%merge = OpLabel
%acc = OpVariable %ptr_fn_int Function
%index = OpVariable %ptr_fn_int Function
OpStore %acc %c0
OpStore %index %c0
OpBranch %loop.header
%loop.header = OpLabel
OpLoopMerge %loop.exit %loop.cont None
OpBranch %loop.cond
%loop.cond = OpLabel
%loaded = OpLoad %int %index
%test = OpSLessThan %bool %loaded %n
OpBranchConditional %test %loop.body %loop.exit
%loop.body = OpLabel
%sum = OpLoad %int %acc
%counter = OpLoad %int %index
%next = OpIAdd %int %sum %counter
OpStore %acc %next
%moved = OpLoad %int %index
%incremented = OpIAdd %int %moved %c1
OpStore %index %incremented
OpBranch %loop.cont
%loop.cont = OpLabel
OpBranch %loop.header
%loop.exit = OpLabel
%answer = OpLoad %int %acc
%addr2 = OpPtrAccessChain %ptr_cw_int %out %c1
OpStore %addr2 %answer
OpReturn
OpFunctionEnd
";
    let module = unwrap(parse(text), "the module parses");
    let outcome = run(
        &module,
        "main",
        &[Arg::Buffer(vec![0; 8]), Arg::Scalar(5)],
        None,
        "hand written",
    );
    // `2 * 10 + 1` is 21, and the loop sums 0 + 1 + 2 + 3 + 4.
    assert_eq!(outcome.buffers, vec![[int_bytes(21), int_bytes(10)].concat()]);
    assert_eq!(outcome.value, Value::Void);
}

#[test]
fn a_hand_written_module_takes_the_other_branch() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%bool = OpTypeBool
%ptr_cw_int = OpTypePointer CrossWorkgroup %int
%fn_void = OpTypeFunction %void %ptr_cw_int %int

%c0 = OpConstant %int 0
%c1 = OpConstant %int 1

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
%out = OpFunctionParameter %ptr_cw_int
%n = OpFunctionParameter %int
%entry = OpLabel
%cond = OpSGreaterThan %bool %n %c0
OpSelectionMerge %merge None
OpBranchConditional %cond %then %else
%then = OpLabel
%addr0 = OpPtrAccessChain %ptr_cw_int %out %c0
OpStore %addr0 %c1
OpBranch %merge
%else = OpLabel
%addr1 = OpPtrAccessChain %ptr_cw_int %out %c0
OpStore %addr1 %c0
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
";
    let module = unwrap(parse(text), "the module parses");
    let outcome = run(
        &module,
        "main",
        &[Arg::Buffer(vec![0; 4]), Arg::Scalar(0)],
        None,
        "hand written",
    );
    assert_eq!(outcome.buffers, vec![int_bytes(0)]);
}

#[test]
fn a_phi_takes_the_value_of_the_predecessor_it_came_from() {
    // The loop header merges two paths, so the value the loop counts
    // with is a phi.
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%bool = OpTypeBool
%fn_int = OpTypeFunction %int %int

%c1 = OpConstant %int 1
%c10 = OpConstant %int 10

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %int None %fn_int
%x = OpFunctionParameter %int
%entry = OpLabel
%first = OpIAdd %int %x %c1
OpBranch %loop
%loop = OpLabel
%count = OpPhi %int %first %entry %next %body
%test = OpSLessThan %bool %count %c10
OpBranchConditional %test %body %exit
%body = OpLabel
%next = OpIAdd %int %count %c1
OpBranch %loop
%exit = OpLabel
OpReturnValue %count
OpFunctionEnd
";
    let module = unwrap(parse(text), "the module parses");
    let entry = some(module.find_function("main"), "the kernel exists");
    let mut interp = Interp::new(&module);
    let value = unwrap(interp.run(entry, &[Value::int(0)]), "the kernel runs");
    // The loop counts from 1 up to 10.
    assert_eq!(value, Value::int(10));
}

#[test]
fn a_vector_operation_is_done_lane_by_lane() {
    // The vectorizer writes vector operations, so the interpreter has to
    // run them the way the machine would: one lane at a time.
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%float = OpTypeFloat 32
%float4 = OpTypeVector %float 4
%ptr_cw_float4 = OpTypePointer CrossWorkgroup %float4
%fn_void = OpTypeFunction %void %ptr_cw_float4

%c1 = OpConstant %float 1
%c2 = OpConstant %float 2
%c3 = OpConstant %float 3
%c4 = OpConstant %float 4

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
%out = OpFunctionParameter %ptr_cw_float4
%entry = OpLabel
%forward = OpCompositeConstruct %float4 %c1 %c2 %c3 %c4
%backward = OpCompositeConstruct %float4 %c4 %c3 %c2 %c1
%sum = OpFAdd %float4 %forward %backward
OpStore %out %sum
OpReturn
OpFunctionEnd
";
    let module = unwrap(parse(text), "the module parses");
    let outcome = run(&module, "main", &[Arg::Buffer(vec![0; 16])], None, "hand written");
    let expected = [5.0f32; 4]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    assert_eq!(outcome.buffers, vec![expected]);
}

#[test]
fn a_division_by_zero_is_reported_and_not_trapped() {
    let text = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%int = OpTypeInt 32 1
%fn_int = OpTypeFunction %int %int

%c0 = OpConstant %int 0
%c7 = OpConstant %int 7

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %int None %fn_int
%x = OpFunctionParameter %int
%entry = OpLabel
%divided = OpSDiv %int %x %c0
OpReturnValue %divided
OpFunctionEnd
";
    let module = unwrap(parse(text), "the module parses");
    let entry = some(module.find_function("main"), "the kernel exists");
    let mut interp = Interp::new(&module);
    let error = match interp.run(entry, &[Value::int(7)]) {
        Ok(_) => panic!("the division by zero is reported"),
        Err(error) => error,
    };
    assert_eq!(error.to_string(), "division by zero");
}
