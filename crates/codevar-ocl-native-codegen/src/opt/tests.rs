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

//! Tests of the optimization passes.
//!
//! Every test builds a small module by hand, checks that the module
//! verifies before the pass, runs the pass, and checks that the module
//! still verifies afterwards.

use codevar_ocl_ir::ir::{
    BasicBlock, BinOp, BlockId, CmpOp, ConstValue, Inst, Linkage, Module, Op, Storage, Target, TypeId,
    ValueId,
};
use codevar_ocl_ir::verify::verify;

use crate::opt::passes;

/// A module under construction.
struct Builder {
    /// The module.
    module: Module,
    /// A signed 32-bit integer type.
    int: TypeId,
    /// A boolean type.
    boolean: TypeId,
    /// The void type.
    void: TypeId,
    /// The function under construction.
    func: ValueId,
}

impl Builder {
    /// Creates a builder over a function that returns nothing.
    fn new() -> Self {
        Self::with_return(false)
    }

    /// Creates a builder over a function that returns an integer.
    fn with_int_result() -> Self {
        Self::with_return(true)
    }

    /// Creates a builder over a function with the given result.
    fn with_return(result: bool) -> Self {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        let boolean = module.bool_ty();
        let void = module.void_ty();
        let ret = if result { int } else { void };
        let sig = module.fn_ty(ret, Vec::new());
        let func = module
            .add_function("f", sig, Linkage::External)
            .expect("a fresh name");
        module.begin_body(func).expect("a body");
        Self {
            module,
            int,
            boolean,
            void,
            func,
        }
    }

    /// The function under construction.
    fn func(&self) -> ValueId {
        self.func
    }

    /// The signed integer type.
    fn int(&self) -> TypeId {
        self.int
    }

    /// Adds a block to the function.
    fn block(&mut self, name: &str) -> BlockId {
        let name = self.module.uniquify_name(name);
        self.module
            .push_block(self.func, &name)
            .expect("a block")
    }

    /// Interns an integer constant.
    fn int_const(&mut self, value: i64) -> ValueId {
        let ty = self.int;
        self.module
            .intern_const(ty, ConstValue::Int(value as u64))
            .expect("a constant")
    }

    /// Interns a boolean constant.
    fn bool_const(&mut self, value: bool) -> ValueId {
        let ty = self.boolean;
        self.module
            .intern_const(ty, ConstValue::Bool(value))
            .expect("a constant")
    }

    /// The type of a stack slot.
    fn slot_type(&mut self) -> TypeId {
        let int = self.int;
        self.module.ptr_ty(Storage::Function, int)
    }

    /// Emits an instruction that produces a result.
    fn def(&mut self, block: BlockId, ty: TypeId, op: Op) -> ValueId {
        let result = self.module.new_inst_value(ty);
        self.module
            .emit(self.func, block, Inst::def(result, ty, op))
            .expect("an instruction");
        result
    }

    /// Reserves a value without defining it yet.
    ///
    /// A merge needs the value its loop computes, and the loop needs
    /// the value the merge defines, so one of the two has to be
    /// reserved first.
    fn reserve(&mut self, ty: TypeId) -> ValueId {
        self.module.new_inst_value(ty)
    }

    /// Emits an instruction that defines a reserved value.
    fn define(&mut self, block: BlockId, result: ValueId, ty: TypeId, op: Op) {
        self.module
            .emit(self.func, block, Inst::def(result, ty, op))
            .expect("an instruction");
    }

    /// Emits an instruction without a result.
    fn emit(&mut self, block: BlockId, op: Op) {
        self.module
            .emit(self.func, block, Inst::none(op))
            .expect("an instruction");
    }

    /// Hands the module over.
    fn finish(self) -> Module {
        self.module
    }
}

/// The blocks of a function.
fn blocks_of(module: &Module, func: ValueId) -> &[BasicBlock] {
    module
        .function(func)
        .and_then(|function| function.body.as_ref())
        .expect("a body")
}

/// The operation that defines `value`.
fn defining_op(module: &Module, func: ValueId, value: ValueId) -> Option<&Op> {
    for block in blocks_of(module, func) {
        for inst in &block.insts {
            if inst.result == Some(value) {
                return Some(&inst.op);
            }
        }
    }
    None
}

/// The value a copy of a constant points at.
fn copied_constant(module: &Module, op: &Op) -> Option<ConstValue> {
    let Op::CopyObject { operand } = op else {
        return None;
    };
    module.constant_value(*operand)
}

/// The number of the operations of a kind in a function.
fn count_ops(module: &Module, func: ValueId, wanted: fn(&Op) -> bool) -> usize {
    let mut count = 0;
    for block in blocks_of(module, func) {
        for inst in &block.insts {
            if wanted(&inst.op) {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn mem2reg_promotes_the_stack_slots() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let slot_type = builder.slot_type();
    let slot = builder.def(entry, slot_type, Op::Variable { init: None });
    let one = builder.int_const(1);
    builder.emit(
        entry,
        Op::Store {
            ptr: slot,
            value: one,
        },
    );
    let loaded = builder.def(entry, builder.int(), Op::Load { ptr: slot });
    builder.emit(entry, Op::ReturnValue { value: loaded });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::mem2reg(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let blocks = blocks_of(&module, func);
    assert!(
        !blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| matches!(inst.op, Op::Variable { .. } | Op::Load { .. } | Op::Store { .. })),
        "the slot, its load, and its store are gone"
    );
}

#[test]
fn mem2reg_leaves_a_slot_whose_address_escapes_alone() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let slot_type = builder.slot_type();
    let slot = builder.def(entry, slot_type, Op::Variable { init: None });
    // A function that takes the slot's address.
    let void = builder.void;
    let sig = builder.module.fn_ty(void, vec![slot_type]);
    let escape = builder
        .module
        .add_function("escape", sig, Linkage::External)
        .expect("a fresh name");
    builder.module.begin_body(escape).expect("a body");
    let escape_entry = builder
        .module
        .push_block(escape, "escape.entry")
        .expect("a block");
    builder
        .module
        .emit(escape, escape_entry, Inst::none(Op::Return))
        .expect("an instruction");

    builder.emit(
        entry,
        Op::Call {
            callee: escape,
            args: vec![slot],
        },
    );
    builder.emit(entry, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::mem2reg(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let blocks = blocks_of(&module, func);
    assert!(
        blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| matches!(inst.op, Op::Variable { .. })),
        "the slot whose address escapes is still there"
    );
}

#[test]
fn constant_folding_evaluates_the_arithmetic() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let one = builder.int_const(1);
    let two = builder.int_const(2);
    let sum = builder.def(
        entry,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: one,
            rhs: two,
        },
    );
    builder.emit(entry, Op::ReturnValue { value: sum });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::constant_folding(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let op = defining_op(&module, func, sum).expect("the definition");
    assert_eq!(
        copied_constant(&module, op),
        Some(ConstValue::Int(3)),
        "the addition folded to three"
    );
}

#[test]
fn constant_folding_leaves_a_division_by_zero_alone() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let one = builder.int_const(1);
    let zero = builder.int_const(0);
    let quotient = builder.def(
        entry,
        builder.int(),
        Op::Binary {
            op: BinOp::SDiv,
            lhs: one,
            rhs: zero,
        },
    );
    builder.emit(entry, Op::ReturnValue { value: quotient });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    passes::constant_folding(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let op = defining_op(&module, func, quotient).expect("the definition");
    assert!(
        matches!(op, Op::Binary { .. }),
        "the division by zero was not folded"
    );
}

#[test]
fn sccp_marks_the_values_of_the_impossible_branch_undefined() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let then = builder.block("then");
    let other = builder.block("other");
    let merge = builder.block("merge");
    let truth = builder.bool_const(true);
    let one = builder.int_const(1);
    builder.emit(
        entry,
        Op::BranchConditional {
            cond: truth,
            then,
            other,
        },
    );
    builder.emit(then, Op::Branch { target: merge });
    let hidden = builder.def(
        other,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: one,
            rhs: one,
        },
    );
    builder.emit(other, Op::Branch { target: merge });
    builder.emit(merge, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::sccp(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let op = defining_op(&module, func, hidden).expect("the definition");
    assert_eq!(
        copied_constant(&module, op),
        Some(ConstValue::Undef),
        "the value of the branch that cannot happen is undefined"
    );
}

#[test]
fn instcombine_collapses_the_identities() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let zero = builder.int_const(0);
    let argument = builder
        .module
        .function(builder.func())
        .expect("the function")
        .args
        .first()
        .copied();
    let value = match argument {
        Some(argument) => argument,
        None => {
            // The function has no argument: use a constant instead.
            zero
        }
    };
    let sum = builder.def(
        entry,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: value,
            rhs: zero,
        },
    );
    builder.emit(entry, Op::ReturnValue { value: sum });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::instcombine(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let op = defining_op(&module, func, sum).expect("the definition");
    assert!(
        matches!(op, Op::CopyObject { .. }),
        "the addition of zero collapsed"
    );
}

#[test]
fn gvn_shares_the_computations() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let one = builder.int_const(1);
    let two = builder.int_const(2);
    let first = builder.def(
        entry,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: one,
            rhs: two,
        },
    );
    let second = builder.def(
        entry,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: one,
            rhs: two,
        },
    );
    builder.emit(entry, Op::ReturnValue { value: second });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::gvn(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let op = defining_op(&module, func, second).expect("the definition");
    let Op::CopyObject { operand } = op else {
        panic!("the second addition became a copy");
    };
    assert_eq!(*operand, first, "the second addition reuses the first");
    let _ = first;
}

#[test]
fn simplify_cfg_folds_the_branch_and_removes_the_dead_block() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let then = builder.block("then");
    let other = builder.block("other");
    let merge = builder.block("merge");
    let truth = builder.bool_const(true);
    builder.emit(
        entry,
        Op::BranchConditional {
            cond: truth,
            then,
            other,
        },
    );
    builder.emit(then, Op::Branch { target: merge });
    builder.emit(other, Op::Branch { target: merge });
    builder.emit(merge, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::simplify_cfg(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let blocks = blocks_of(&module, func);
    assert!(
        !blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| matches!(inst.op, Op::BranchConditional { .. })),
        "the branch on a known condition is gone"
    );
    assert!(
        !blocks.iter().any(|block| block.name == "other"),
        "the block the folded branch could not reach is gone"
    );
}

#[test]
fn dce_removes_the_computation_nobody_uses() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let one = builder.int_const(1);
    builder.def(
        entry,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: one,
            rhs: one,
        },
    );
    builder.emit(entry, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::dce(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    assert_eq!(
        count_ops(&module, func, |op| matches!(op, Op::Binary { .. })),
        0,
        "the unused addition is gone"
    );
}

#[test]
fn dce_keeps_the_store() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let slot_type = builder.slot_type();
    let slot = builder.def(entry, slot_type, Op::Variable { init: None });
    let one = builder.int_const(1);
    builder.emit(
        entry,
        Op::Store {
            ptr: slot,
            value: one,
        },
    );
    builder.emit(entry, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    passes::dce(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    assert_eq!(
        count_ops(&module, func, |op| matches!(op, Op::Store { .. })),
        1,
        "the store survived"
    );
}

#[test]
fn inline_brings_the_callee_into_the_caller() {
    let mut builder = Builder::with_int_result();
    // The callee.
    let int = builder.int();
    let sig = builder.module.fn_ty(int, Vec::new());
    let callee = builder
        .module
        .add_function("g", sig, Linkage::External)
        .expect("a fresh name");
    builder.module.begin_body(callee).expect("a body");
    let callee_entry = builder
        .module
        .push_block(callee, "g.entry")
        .expect("a block");
    let one = builder.int_const(1);
    let two = builder.int_const(2);
    let result = builder.module.new_inst_value(int);
    builder
        .module
        .emit(
            callee,
            callee_entry,
            Inst::def(
                result,
                int,
                Op::Binary {
                    op: BinOp::IAdd,
                    lhs: one,
                    rhs: two,
                },
            ),
        )
        .expect("an instruction");
    builder
        .module
        .emit(
            callee,
            callee_entry,
            Inst::none(Op::ReturnValue { value: result }),
        )
        .expect("an instruction");

    // The caller.
    let entry = builder.block("entry");
    let call_result = builder.def(
        entry,
        int,
        Op::Call {
            callee,
            args: Vec::new(),
        },
    );
    builder.emit(entry, Op::ReturnValue { value: call_result });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::inline(&mut module, 16).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let blocks = blocks_of(&module, func);
    assert!(
        !blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .any(|inst| matches!(inst.op, Op::Call { .. })),
        "the call is gone"
    );
    assert_eq!(
        count_ops(&module, func, |op| matches!(op, Op::Binary { .. })),
        1,
        "the body of the callee is in the caller"
    );
}

#[test]
fn licm_moves_the_invariant_computation_out_of_the_loop() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let header = builder.block("header");
    let body = builder.block("body");
    let merge = builder.block("merge");
    let zero = builder.int_const(0);
    let one = builder.int_const(1);
    let two = builder.int_const(2);
    let bound = builder.int_const(4);
    let induction = builder.reserve(builder.int());
    let next = builder.def(
        body,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: induction,
            rhs: one,
        },
    );
    builder.def(
        body,
        builder.int(),
        Op::Binary {
            op: BinOp::IMul,
            lhs: two,
            rhs: two,
        },
    );
    builder.emit(body, Op::Branch { target: header });
    builder.define(
        header,
        induction,
        builder.int(),
        Op::Phi {
            incomings: vec![(zero, entry), (next, body)],
        },
    );
    let condition = builder.def(
        header,
        builder.boolean,
        Op::Compare {
            op: CmpOp::SLessThan,
            lhs: induction,
            rhs: bound,
        },
    );
    builder.emit(
        header,
        Op::LoopMerge {
            merge,
            cont: body,
            control: 0,
        },
    );
    builder.emit(
        header,
        Op::BranchConditional {
            cond: condition,
            then: body,
            other: merge,
        },
    );
    builder.emit(entry, Op::Branch { target: header });
    builder.emit(merge, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::licm(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    let blocks = blocks_of(&module, func);
    let entry_ops: Vec<&Op> = blocks[0]
        .insts
        .iter()
        .map(|inst| &inst.op)
        .collect();
    assert!(
        entry_ops
            .iter()
            .any(|op| matches!(op, Op::Binary { op: BinOp::IMul, .. })),
        "the invariant computation moved into the block that enters the loop"
    );
    let body_ops: Vec<&Op> = blocks[2]
        .insts
        .iter()
        .map(|inst| &inst.op)
        .collect();
    assert!(
        !body_ops
            .iter()
            .any(|op| matches!(op, Op::Binary { op: BinOp::IMul, .. })),
        "the loop no longer computes the invariant value"
    );
}

#[test]
fn unroll_writes_the_iterations_of_a_counted_loop() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let header = builder.block("header");
    let body = builder.block("body");
    let merge = builder.block("merge");
    let zero = builder.int_const(0);
    let one = builder.int_const(1);
    let bound = builder.int_const(2);
    let induction = builder.reserve(builder.int());
    let next = builder.def(
        body,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: induction,
            rhs: one,
        },
    );
    builder.emit(body, Op::Branch { target: header });
    builder.define(
        header,
        induction,
        builder.int(),
        Op::Phi {
            incomings: vec![(zero, entry), (next, body)],
        },
    );
    let condition = builder.def(
        header,
        builder.boolean,
        Op::Compare {
            op: CmpOp::SLessThan,
            lhs: induction,
            rhs: bound,
        },
    );
    builder.emit(
        header,
        Op::LoopMerge {
            merge,
            cont: body,
            control: 0,
        },
    );
    builder.emit(
        header,
        Op::BranchConditional {
            cond: condition,
            then: body,
            other: merge,
        },
    );
    builder.emit(entry, Op::Branch { target: header });
    builder.emit(merge, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::unroll(&mut module, 8).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    assert_eq!(
        count_ops(&module, func, |op| matches!(
            op,
            Op::Binary { op: BinOp::IAdd, .. }
        )),
        2,
        "the body of the loop appears twice"
    );
}

#[test]
fn a_pass_leaves_a_module_it_cannot_rewrite_alone() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    builder.emit(entry, Op::Return);
    let module = builder.finish();
    let before = module.clone();

    let mut module = module;
    passes::mem2reg(&mut module).expect("the pass runs");
    passes::constant_folding(&mut module).expect("the pass runs");
    passes::sccp(&mut module).expect("the pass runs");
    passes::gvn(&mut module).expect("the pass runs");
    passes::instcombine(&mut module).expect("the pass runs");
    passes::simplify_cfg(&mut module).expect("the pass runs");
    passes::dce(&mut module).expect("the pass runs");
    passes::licm(&mut module).expect("the pass runs");
    passes::unroll(&mut module, 4).expect("the pass runs");

    verify(&module).expect("the module still verifies");
    assert_eq!(
        module, before,
        "the passes left the module exactly as they found it"
    );
}

#[test]
fn the_whole_pipeline_leaves_the_module_verifiable() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let slot_type = builder.slot_type();
    let slot = builder.def(entry, slot_type, Op::Variable { init: None });
    let one = builder.int_const(1);
    builder.emit(
        entry,
        Op::Store {
            ptr: slot,
            value: one,
        },
    );
    let loaded = builder.def(entry, builder.int(), Op::Load { ptr: slot });
    let then = builder.block("then");
    let merge = builder.block("merge");
    let truth = builder.bool_const(true);
    builder.emit(
        entry,
        Op::BranchConditional {
            cond: truth,
            then,
            other: merge,
        },
    );
    builder.emit(then, Op::Branch { target: merge });
    builder.emit(merge, Op::ReturnValue { value: loaded });
    let mut module = builder.finish();

    verify(&module).expect("the module verifies before the pipeline");
    crate::opt::optimize(&mut module, &crate::opt::OptConfig::default()).expect("the pipeline runs");
    verify(&module).expect("the module verifies after the pipeline");
}

/// The sources the front end lowers into the shape the optimizer
/// expects.
const SOURCES: [&str; 7] = [
    "#[kernel]\nfn zero(out: *mut int) {\n    *out = 0;\n}",
    "#[kernel]\nfn sum(out: *mut int, a: int, b: int) {\n    *out = a + b;\n}",
    "#[kernel]\nfn scale(out: *mut float, a: float, b: float) {\n    *out = a * b + 1.0;\n}",
    "#[kernel]\nfn pick(out: *mut int, a: int, b: int) {\n    if a > b { *out = a; } else { *out = b; }\n}",
    "#[kernel]\nfn vector_add(a: *const int, b: *const int, out: *mut int, n: int) -> void {\n    let i = get_global_id(0);\n    if i < n {\n        out[i] = a[i] + b[i];\n    }\n}",
    "#[kernel]\nfn looped(out: *mut int, n: int) {\n    let mut i = 0;\n    while i < n {\n        *out = i;\n        i = i + 1;\n    }\n}",
    "#[kernel]\nfn counted(out: *mut int) {\n    let mut i = 0;\n    while i < 4 {\n        *out = i;\n        i = i + 1;\n    }\n}",
];

#[test]
fn the_optimizer_handles_the_modules_the_front_end_produces() {
    for source in SOURCES {
        let mut module = codevar_ocl_ir::lower::lower_source(source).expect("the source lowers");
        verify(&module).expect("the lowered module verifies");
        crate::opt::optimize(&mut module, &crate::opt::OptConfig::default()).expect("the pipeline runs");
        verify(&module).expect("the optimized module verifies");
    }
}

#[test]
fn the_optimization_sets_combine_with_bit_or() {
    use crate::opt::{Optimization, Optimizations};

    let set = Optimizations::O1 | Optimization::Inlineable;
    assert!(set.has(Optimization::Mem2Reg));
    assert!(set.has(Optimization::Inlineable));
    assert!(!set.has(Optimization::Vectorizable));
    assert!(!Optimizations::O0.has(Optimization::Mem2Reg));
    assert_eq!(Optimizations::from_level(3), Some(Optimizations::O3));
    assert_eq!(Optimizations::from_level(4), None);
}

#[test]
fn every_optimization_level_leaves_the_module_verifiable() {
    use crate::opt::{OptConfig, Optimizations, optimize};

    for level in [
        Optimizations::O0,
        Optimizations::O1,
        Optimizations::O2,
        Optimizations::O3,
    ] {
        let mut module = codevar_ocl_ir::lower::lower_source(SOURCES[5]).expect("the source lowers");
        verify(&module).expect("the lowered module verifies");
        let config = OptConfig {
            optimizations: level,
            ..OptConfig::default()
        };
        optimize(&mut module, &config).expect("the pipeline runs");
        verify(&module).expect("the optimized module verifies");
    }
}

#[test]
fn the_optimizer_promotes_the_slots_the_front_end_creates() {
    let source = SOURCES[5];
    let mut module = codevar_ocl_ir::lower::lower_source(source).expect("the source lowers");
    let func = module
        .entry_points
        .first()
        .map(|entry| entry.func)
        .expect("an entry point");
    let slots = count_ops(&module, func, |op| matches!(op, Op::Variable { .. }));
    assert!(slots > 0, "the front end creates stack slots");

    crate::opt::optimize(&mut module, &crate::opt::OptConfig::default()).expect("the pipeline runs");
    verify(&module).expect("the optimized module verifies");

    let left = count_ops(&module, func, |op| matches!(op, Op::Variable { .. }));
    assert!(left < slots, "the slots became values");
}

/// The optimizer must not change what the kernel computes.
///
/// The interpreter runs the kernel before and after the pipeline and
/// compares the memory the kernel wrote.
#[test]
fn the_optimizer_keeps_the_meaning_of_the_kernel() {
    use crate::interp::Interp;
    for source in SOURCES {
        let module = codevar_ocl_ir::lower::lower_source(source).expect("the source lowers");
        let kernel = module.entry_points[0].func;
        let mut optimized = module.clone();
        crate::opt::optimize(&mut optimized, &crate::opt::OptConfig::default()).expect("the pipeline runs");

        let mut before = [0u8; 512];
        {
            let mut interp = Interp::new(&module);
            let alloc = interp.memory().external(&mut before);
            let args = kernel_args(&module, kernel, alloc);
            interp
                .run(kernel, &args)
                .expect("the kernel runs");
        }
        let mut after = [0u8; 512];
        {
            let mut interp = Interp::new(&optimized);
            let alloc = interp.memory().external(&mut after);
            let args = kernel_args(&optimized, kernel, alloc);
            interp
                .run(kernel, &args)
                .expect("the kernel runs");
        }
        assert_eq!(before, after, "the kernel changed its meaning: {source}");
    }
}

/// The arguments of a kernel: a pointer for every pointer parameter and
/// a zero for every other one.
fn kernel_args(
    module: &codevar_ocl_ir::ir::Module,
    kernel: codevar_ocl_ir::ir::ValueId,
    alloc: crate::interp::AllocId,
) -> Vec<crate::interp::Value> {
    use crate::interp::{Pointer, Value};
    let function = module.function(kernel).expect("the kernel");
    function
        .args
        .iter()
        .enumerate()
        .map(|(index, &arg)| {
            if matches!(
                module.ty(module.type_of(arg)),
                codevar_ocl_ir::ir::Type::Pointer { .. }
            ) {
                Value::pointer(Pointer::new(alloc, index * 64))
            } else {
                Value::int(3)
            }
        })
        .collect()
}

#[test]
fn mem2reg_merges_the_values_of_the_two_arms() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let then = builder.block("then");
    let other = builder.block("other");
    let merge = builder.block("merge");
    let slot_type = builder.slot_type();
    let slot = builder.def(entry, slot_type, Op::Variable { init: None });
    let one = builder.int_const(1);
    let two = builder.int_const(2);
    let truth = builder.bool_const(true);
    builder.emit(
        entry,
        Op::Store {
            ptr: slot,
            value: one,
        },
    );
    builder.emit(
        entry,
        Op::BranchConditional {
            cond: truth,
            then,
            other,
        },
    );
    builder.emit(
        then,
        Op::Store {
            ptr: slot,
            value: two,
        },
    );
    builder.emit(then, Op::Branch { target: merge });
    builder.emit(other, Op::Branch { target: merge });
    let loaded = builder.def(merge, builder.int(), Op::Load { ptr: slot });
    builder.emit(merge, Op::ReturnValue { value: loaded });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::mem2reg(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    // The merge holds a merge of the two values.
    let mut merges = 0;
    let mut incomings = 0;
    for block in blocks_of(&module, func) {
        for inst in &block.insts {
            if let Op::Phi { incomings: pairs } = &inst.op {
                merges += 1;
                incomings += pairs.len();
            }
        }
    }
    assert_eq!(merges, 1, "the merge block received a merge");
    assert_eq!(incomings, 2, "the merge has one incoming per predecessor");
}

#[test]
fn dce_keeps_the_value_a_merge_reads() {
    let mut builder = Builder::with_int_result();
    let entry = builder.block("entry");
    let then = builder.block("then");
    let other = builder.block("other");
    let merge = builder.block("merge");
    let one = builder.int_const(1);
    let two = builder.int_const(2);
    let truth = builder.bool_const(true);
    let first = builder.def(
        then,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: one,
            rhs: one,
        },
    );
    let second = builder.def(
        other,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: two,
            rhs: two,
        },
    );
    builder.emit(
        entry,
        Op::BranchConditional {
            cond: truth,
            then,
            other,
        },
    );
    builder.emit(then, Op::Branch { target: merge });
    builder.emit(other, Op::Branch { target: merge });
    let merged = builder.def(
        merge,
        builder.int(),
        Op::Phi {
            incomings: vec![(first, then), (second, other)],
        },
    );
    builder.emit(merge, Op::ReturnValue { value: merged });
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::dce(&mut module).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    assert_eq!(
        count_ops(&module, func, |op| matches!(op, Op::Binary { .. })),
        2,
        "the two computations the merge reads stayed"
    );
}

#[test]
fn unroll_leaves_the_loop_when_the_count_is_large() {
    let mut builder = Builder::new();
    let entry = builder.block("entry");
    let header = builder.block("header");
    let body = builder.block("body");
    let merge = builder.block("merge");
    let zero = builder.int_const(0);
    let one = builder.int_const(1);
    let bound = builder.int_const(10);
    let induction = builder.reserve(builder.int());
    let next = builder.def(
        body,
        builder.int(),
        Op::Binary {
            op: BinOp::IAdd,
            lhs: induction,
            rhs: one,
        },
    );
    builder.emit(body, Op::Branch { target: header });
    builder.define(
        header,
        induction,
        builder.int(),
        Op::Phi {
            incomings: vec![(zero, entry), (next, body)],
        },
    );
    let condition = builder.def(
        header,
        builder.boolean,
        Op::Compare {
            op: CmpOp::SLessThan,
            lhs: induction,
            rhs: bound,
        },
    );
    builder.emit(
        header,
        Op::LoopMerge {
            merge,
            cont: body,
            control: 0,
        },
    );
    builder.emit(
        header,
        Op::BranchConditional {
            cond: condition,
            then: body,
            other: merge,
        },
    );
    builder.emit(entry, Op::Branch { target: header });
    builder.emit(merge, Op::Return);
    let mut module = builder.finish();
    let func = module.find_function("f").expect("the function");

    verify(&module).expect("the module verifies before the pass");
    passes::unroll(&mut module, 4).expect("the pass runs");
    verify(&module).expect("the module verifies after the pass");

    assert_eq!(
        count_ops(&module, func, |op| matches!(op, Op::LoopMerge { .. })),
        1,
        "the loop is still there"
    );
    assert!(
        count_ops(&module, func, |op| matches!(
            op,
            Op::Binary { op: BinOp::IAdd, .. }
        )) >= 4,
        "the first iterations were written out"
    );
}

#[test]
fn unroll_writes_out_the_counted_loop_of_the_front_end() {
    let source = SOURCES[6];
    let mut module = codevar_ocl_ir::lower::lower_source(source).expect("the source lowers");
    let func = module
        .entry_points
        .first()
        .map(|entry| entry.func)
        .expect("an entry point");
    let before = count_ops(&module, func, |op| {
        matches!(op, Op::Binary { op: BinOp::IAdd, .. })
    });

    passes::mem2reg(&mut module).expect("the pass runs");
    passes::unroll(&mut module, 8).expect("the pass runs");
    verify(&module).expect("the module verifies after the passes");

    let after = count_ops(&module, func, |op| {
        matches!(op, Op::Binary { op: BinOp::IAdd, .. })
    });
    assert!(
        after > before,
        "the counted loop was written out: {before} became {after}"
    );
}
