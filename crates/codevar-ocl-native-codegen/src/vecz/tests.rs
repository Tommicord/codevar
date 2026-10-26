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

//! Tests of the vectorizer.
//!
//! The modules are built by hand, block by block, so that every test
//! states the shape it exercises and nothing else.

use alloc::vec::Vec;

use codevar_ocl_ir::ir::{
    BinOp, BlockId, CmpOp, ConstValue, Inst, Linkage, Module, Op, Storage, Target, Type, TypeId, ValueId,
};
use codevar_ocl_ir::print::print;
use codevar_ocl_ir::verify::verify;

use super::{VectorConfig, vectorize};

/// The number of lanes the tests vectorize with.
const LANES: u32 = 4;

/// A module with one function, assembled block by block.
struct Builder {
    /// The module under construction.
    module: Module,
    /// The function the blocks belong to.
    func: ValueId,
}

impl Builder {
    /// Starts a module with a void function of the given signature.
    fn new(params: Vec<TypeId>) -> Self {
        let mut module = Module::new(Target::opencl());
        let void = module.void_ty();
        let sig = module.fn_ty(void, params);
        let func = module
            .add_function("kernel", sig, Linkage::External)
            .expect("fresh name");
        module.begin_body(func).expect("declaration");
        Self { module, func }
    }

    /// Appends a block and returns its handle.
    fn block(&mut self, name: &str) -> BlockId {
        self.module
            .push_block(self.func, name)
            .expect("the body has started")
    }

    /// Appends an instruction to a block.
    fn emit(&mut self, block: BlockId, inst: Inst) {
        self.module
            .emit(self.func, block, inst)
            .expect("the block exists");
    }

    /// Creates an unnamed value of the given type.
    fn value(&mut self, ty: TypeId) -> ValueId {
        self.module.new_inst_value(ty)
    }

    /// Interns a constant.
    fn constant(&mut self, ty: TypeId, payload: ConstValue) -> ValueId {
        self.module
            .intern_const(ty, payload)
            .expect("the payload fits")
    }

    /// Finishes the module.
    fn finish(self) -> Module {
        self.module
    }
}

/// The pieces of a counted loop over an array of floats.
struct LoopParts {
    /// The induction variable's type.
    int: TypeId,
    /// The element type of the array.
    float: TypeId,
    /// The start of the induction variable.
    zero: ValueId,
    /// The header of the loop.
    header: BlockId,
    /// The body of the loop.
    body: BlockId,
}

/// Builds a counted loop over an array of floats.
///
/// The loop adds a constant to every element of the array, which is
/// the shape a vectorized loop must handle: the body reads an
/// element, computes on it, and writes it back.
fn counted_loop() -> (Builder, LoopParts) {
    let mut builder = Builder::new(Vec::new());
    let float = builder.module.float_ty(32);
    let int = builder.module.int_ty(32, true);
    let bool_ty = builder.module.bool_ty();
    let array = builder.module.array_ty(float, 64);
    let array_pointer = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, array);
    let arr = builder
        .module
        .add_global("arr", array_pointer, Linkage::External, None)
        .expect("fresh name");
    let element = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, float);
    let zero = builder.constant(int, ConstValue::Int(0));
    let one = builder.constant(int, ConstValue::Int(1));
    let bound = builder.constant(int, ConstValue::Int(64));
    let delta = builder.constant(float, ConstValue::Float32(0));

    let entry = builder.block("entry");
    let header = builder.block("loop.header");
    let body = builder.block("loop.body");
    let cont = builder.block("loop.cont");
    let merge = builder.block("loop.merge");

    // The induction variable and the comparison.
    let iv = builder.value(int);
    let cmp = builder.value(bool_ty);
    builder.emit(
        header,
        Inst::def(
            iv,
            int,
            Op::Phi {
                incomings: alloc::vec![(zero, entry), (ValueId(0), cont)],
            },
        ),
    );
    builder.emit(
        header,
        Inst::def(
            cmp,
            bool_ty,
            Op::Compare {
                op: CmpOp::SLessThan,
                lhs: iv,
                rhs: bound,
            },
        ),
    );
    builder.emit(
        header,
        Inst::none(Op::LoopMerge {
            merge,
            cont,
            control: 0,
        }),
    );
    builder.emit(
        header,
        Inst::none(Op::BranchConditional {
            cond: cmp,
            then: body,
            other: merge,
        }),
    );

    // The body: arr[i] = arr[i] + delta.
    let access = builder.value(element);
    let loaded = builder.value(float);
    let added = builder.value(float);
    builder.emit(
        body,
        Inst::def(
            access,
            element,
            Op::AccessChain {
                base: arr,
                indices: alloc::vec![iv],
            },
        ),
    );
    builder.emit(body, Inst::def(loaded, float, Op::Load { ptr: access }));
    builder.emit(
        body,
        Inst::def(
            added,
            float,
            Op::Binary {
                op: BinOp::FAdd,
                lhs: loaded,
                rhs: delta,
            },
        ),
    );
    builder.emit(
        body,
        Inst::none(Op::Store {
            ptr: access,
            value: added,
        }),
    );
    builder.emit(body, Inst::none(Op::Branch { target: cont }));

    // The increment.
    let next = builder.value(int);
    builder.emit(
        cont,
        Inst::def(
            next,
            int,
            Op::Binary {
                op: BinOp::IAdd,
                lhs: iv,
                rhs: one,
            },
        ),
    );
    builder.emit(cont, Inst::none(Op::Branch { target: header }));

    builder.emit(entry, Inst::none(Op::Branch { target: header }));
    builder.emit(merge, Inst::none(Op::Return));

    // The induction variable takes the value the increment produces
    // when the loop continues.
    let blocks = builder
        .module
        .function_mut(builder.func)
        .expect("a function");
    let body_blocks = blocks.body.as_mut().expect("a body");
    if let Op::Phi { incomings } = &mut body_blocks[header.index()].insts[0].op {
        incomings[1].0 = next;
    }

    let parts = LoopParts {
        int,
        float,
        zero,
        header,
        body,
    };
    (builder, parts)
}

/// The number of blocks of a function.
fn block_count(module: &Module, func: ValueId) -> usize {
    module
        .function(func)
        .and_then(|f| f.body.as_ref())
        .map(|body| body.len())
        .unwrap_or(0)
}

/// The instructions of every block of a function.
fn instructions(module: &Module, func: ValueId) -> Vec<Inst> {
    module
        .function(func)
        .and_then(|f| f.body.as_ref())
        .map(|body| {
            body.iter()
                .flat_map(|block| block.insts.iter().cloned())
                .collect()
        })
        .unwrap_or_default()
}

/// True when the module holds an operation whose result is a vector.
fn holds_vector_operation(module: &Module, func: ValueId) -> bool {
    instructions(module, func).iter().any(|inst| {
        inst.ty
            .is_some_and(|ty| matches!(module.ty(ty), Type::Vector { .. }))
    })
}

/// True when the module holds a vector whose lanes are the given type.
fn holds_vector_of(module: &Module, func: ValueId, element: TypeId) -> bool {
    instructions(module, func).iter().any(|inst| {
        inst.ty
            .is_some_and(|ty| matches!(module.ty(ty), Type::Vector { elem, .. } if *elem == element))
    })
}

/// The number of loads of the module.
fn load_count(module: &Module, func: ValueId) -> usize {
    instructions(module, func)
        .iter()
        .filter(|inst| matches!(inst.op, Op::Load { .. }))
        .count()
}

/// The number of stores of the module.
fn store_count(module: &Module, func: ValueId) -> usize {
    instructions(module, func)
        .iter()
        .filter(|inst| matches!(inst.op, Op::Store { .. }))
        .count()
}

/// The configuration the tests vectorize with.
fn config() -> VectorConfig {
    VectorConfig {
        max_lanes: LANES,
        ..VectorConfig::default()
    }
}

/// Builds a counted loop whose header holds the merge instruction and
/// whose comparison sits in the block the header branches to.
///
/// This is the shape the front end produces: the header carries the
/// merge, and the comparison is evaluated in the block that follows.
fn split_condition_loop() -> (Builder, LoopParts) {
    let mut builder = Builder::new(Vec::new());
    let float = builder.module.float_ty(32);
    let int = builder.module.int_ty(32, true);
    let bool_ty = builder.module.bool_ty();
    let array = builder.module.array_ty(float, 64);
    let array_pointer = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, array);
    let arr = builder
        .module
        .add_global("arr", array_pointer, Linkage::External, None)
        .expect("fresh name");
    let element = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, float);
    let zero = builder.constant(int, ConstValue::Int(0));
    let one = builder.constant(int, ConstValue::Int(1));
    let bound = builder.constant(int, ConstValue::Int(64));
    let delta = builder.constant(float, ConstValue::Float32(0));

    let entry = builder.block("entry");
    let header = builder.block("loop.header");
    let cond = builder.block("loop.cond");
    let body = builder.block("loop.body");
    let cont = builder.block("loop.cont");
    let merge = builder.block("loop.merge");

    let iv = builder.value(int);
    let cmp = builder.value(bool_ty);
    builder.emit(
        header,
        Inst::def(
            iv,
            int,
            Op::Phi {
                incomings: alloc::vec![(zero, entry), (ValueId(0), cont)],
            },
        ),
    );
    builder.emit(
        header,
        Inst::none(Op::LoopMerge {
            merge,
            cont,
            control: 0,
        }),
    );
    builder.emit(header, Inst::none(Op::Branch { target: cond }));
    builder.emit(
        cond,
        Inst::def(
            cmp,
            bool_ty,
            Op::Compare {
                op: CmpOp::SLessThan,
                lhs: iv,
                rhs: bound,
            },
        ),
    );
    builder.emit(
        cond,
        Inst::none(Op::BranchConditional {
            cond: cmp,
            then: body,
            other: merge,
        }),
    );

    let access = builder.value(element);
    let loaded = builder.value(float);
    let added = builder.value(float);
    builder.emit(
        body,
        Inst::def(
            access,
            element,
            Op::AccessChain {
                base: arr,
                indices: alloc::vec![iv],
            },
        ),
    );
    builder.emit(body, Inst::def(loaded, float, Op::Load { ptr: access }));
    builder.emit(
        body,
        Inst::def(
            added,
            float,
            Op::Binary {
                op: BinOp::FAdd,
                lhs: loaded,
                rhs: delta,
            },
        ),
    );
    builder.emit(
        body,
        Inst::none(Op::Store {
            ptr: access,
            value: added,
        }),
    );
    builder.emit(body, Inst::none(Op::Branch { target: cont }));

    let next = builder.value(int);
    builder.emit(
        cont,
        Inst::def(
            next,
            int,
            Op::Binary {
                op: BinOp::IAdd,
                lhs: iv,
                rhs: one,
            },
        ),
    );
    builder.emit(cont, Inst::none(Op::Branch { target: header }));
    builder.emit(entry, Inst::none(Op::Branch { target: header }));
    builder.emit(merge, Inst::none(Op::Return));

    let blocks = {
        let function = builder
            .module
            .function_mut(builder.func)
            .expect("a function");
        function.body.as_mut().expect("a body")
    };
    if let Op::Phi { incomings } = &mut blocks[header.index()].insts[0].op {
        incomings[1].0 = next;
    }

    let parts = LoopParts {
        int,
        float,
        zero,
        header,
        body,
    };
    (builder, parts)
}

#[test]
fn a_loop_with_a_split_condition_is_vectorized() {
    let (builder, parts) = split_condition_loop();
    let func = builder.func;
    let mut module = builder.finish();
    verify(&module).expect("the module verifies before the rewrite");
    let blocks_before = block_count(&module, func);

    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies after the rewrite");

    assert_eq!(block_count(&module, func), blocks_before + 4);
    assert!(holds_vector_of(&module, func, parts.float));
    assert_eq!(load_count(&module, func), LANES as usize + 1);
}

#[test]
fn counted_loop_becomes_a_vector_loop_with_a_remainder() {
    let (builder, parts) = counted_loop();
    let func = builder.func;
    let mut module = builder.finish();
    verify(&module).expect("the module verifies before the rewrite");

    let blocks_before = block_count(&module, func);
    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies after the rewrite");

    // The vector loop and the remainder loop added four blocks, and
    // the block count did not explode.
    assert_eq!(block_count(&module, func), blocks_before + 4);

    // The body was rewritten with vector-typed values: the addition
    // is now a vector addition over the lanes.
    assert!(holds_vector_operation(&module, func));
    assert!(holds_vector_of(&module, func, parts.float));

    // The vector loop gathers one element per lane and scatters one
    // element per lane, and the remainder loop keeps the original
    // load and store.
    assert_eq!(load_count(&module, func), LANES as usize + 1);
    assert_eq!(store_count(&module, func), LANES as usize + 1);

    // The vector loop's induction variable steps by the number of
    // lanes, and the remainder loop still steps by one.
    let steps_by_lanes = instructions(&module, func).iter().any(|inst| {
        matches!(
            &inst.op,
            Op::Binary { op: BinOp::IAdd, rhs, .. }
                if module.constant_value(*rhs) == Some(ConstValue::Int(u64::from(LANES)))
        )
    });
    assert!(steps_by_lanes, "the vector loop steps by the number of lanes");
    let steps_by_one = instructions(&module, func).iter().any(|inst| {
        matches!(
            &inst.op,
            Op::Binary { op: BinOp::IAdd, rhs, .. }
                if module.constant_value(*rhs) == Some(ConstValue::Int(1))
        )
    });
    assert!(steps_by_one, "the remainder loop steps by one");
}

#[test]
fn independent_additions_become_one_vector_addition() {
    let mut builder = Builder::new(Vec::new());
    let float = builder.module.float_ty(32);
    let int = builder.module.int_ty(32, true);
    let array = builder.module.array_ty(float, 16);
    let array_pointer = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, array);
    let arr = builder
        .module
        .add_global("arr", array_pointer, Linkage::External, None)
        .expect("fresh name");
    let element = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, float);

    let entry = builder.block("entry");
    let mut results = Vec::new();
    for lane in 0..LANES {
        let index = builder.constant(int, ConstValue::Int(u64::from(lane)));
        let access = builder.value(element);
        let loaded = builder.value(float);
        let added = builder.value(float);
        builder.emit(
            entry,
            Inst::def(
                access,
                element,
                Op::AccessChain {
                    base: arr,
                    indices: alloc::vec![index],
                },
            ),
        );
        builder.emit(entry, Inst::def(loaded, float, Op::Load { ptr: access }));
        builder.emit(
            entry,
            Inst::def(
                added,
                float,
                Op::Binary {
                    op: BinOp::FAdd,
                    lhs: loaded,
                    rhs: loaded,
                },
            ),
        );
        results.push(added);
    }
    // The results are read once the group is complete, so that the
    // group is worth a vector.
    for result in results {
        let copy = builder.value(float);
        builder.emit(entry, Inst::def(copy, float, Op::CopyObject { operand: result }));
    }
    builder.emit(entry, Inst::none(Op::Return));
    let func = builder.func;
    let mut module = builder.finish();
    verify(&module).expect("the module verifies before the rewrite");

    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies after the rewrite");

    // The four additions became one vector addition.
    let additions = instructions(&module, func)
        .iter()
        .filter(|inst| matches!(inst.op, Op::Binary { op: BinOp::FAdd, .. }))
        .count();
    assert_eq!(additions, 1, "the group became one vector addition");
    assert!(holds_vector_of(&module, func, float));

    // The original results still exist, and each of them reads one
    // lane of the vector.
    let reads = instructions(&module, func)
        .iter()
        .filter(|inst| matches!(inst.op, Op::CompositeExtract { .. }))
        .count();
    assert_eq!(reads, LANES as usize);
}

#[test]
fn a_loop_with_a_call_is_left_alone() {
    let (mut builder, parts) = counted_loop();
    let func = builder.func;
    let void = builder.module.void_ty();
    let sig = builder.module.fn_ty(void, Vec::new());
    let callee = builder
        .module
        .add_function("helper", sig, Linkage::Import)
        .expect("fresh name");
    // The body calls a function, which the vectorizer does not
    // vectorize.
    {
        let function = builder
            .module
            .function_mut(func)
            .expect("a function");
        let blocks = function.body.as_mut().expect("a body");
        blocks[parts.body.index()].insts.insert(
            1,
            Inst::none(Op::Call {
                callee,
                args: Vec::new(),
            }),
        );
    }
    let mut module = builder.finish();
    verify(&module).expect("the module still verifies");
    let before = print(&module);

    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies");
    assert_eq!(print(&module), before, "the loop was left alone");
}

#[test]
fn a_loop_with_a_stride_of_two_is_left_alone() {
    let (mut builder, parts) = counted_loop();
    let func = builder.func;
    // The body walks the array with a stride of two.
    let two = builder.constant(parts.int, ConstValue::Int(2));
    let step = builder.value(parts.int);
    {
        let function = builder
            .module
            .function_mut(func)
            .expect("a function");
        let blocks = function.body.as_mut().expect("a body");
        let body = &mut blocks[parts.body.index()];
        let iv = match &body.insts[0].op {
            Op::AccessChain { indices, .. } => indices[0],
            _ => ValueId(0),
        };
        body.insts.insert(
            0,
            Inst::def(
                step,
                parts.int,
                Op::Binary {
                    op: BinOp::IMul,
                    lhs: iv,
                    rhs: two,
                },
            ),
        );
        if let Op::AccessChain { indices, .. } = &mut body.insts[1].op {
            indices[0] = step;
        }
    }
    let mut module = builder.finish();
    verify(&module).expect("the module verifies before the rewrite");
    let before = print(&module);

    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies");
    assert_eq!(print(&module), before, "the loop was left alone");
}

#[test]
fn a_loop_without_a_comparison_is_left_alone() {
    let (mut builder, parts) = counted_loop();
    let func = builder.func;
    // The exit no longer depends on a comparison of the induction
    // variable against the bound.
    {
        let function = builder
            .module
            .function_mut(func)
            .expect("a function");
        let blocks = function.body.as_mut().expect("a body");
        if let Op::Compare { lhs, .. } = &mut blocks[parts.header.index()].insts[1].op {
            *lhs = parts.zero;
        }
    }
    let mut module = builder.finish();
    verify(&module).expect("the module still verifies");
    let before = print(&module);

    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies");
    assert_eq!(print(&module), before, "the loop was left alone");
}

#[test]
fn a_loop_over_vectors_is_left_alone() {
    let mut builder = Builder::new(Vec::new());
    let float = builder.module.float_ty(32);
    let vector = builder.module.vector_ty(float, 4);
    let int = builder.module.int_ty(32, true);
    let bool_ty = builder.module.bool_ty();
    let array = builder.module.array_ty(vector, 64);
    let array_pointer = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, array);
    let arr = builder
        .module
        .add_global("arr", array_pointer, Linkage::External, None)
        .expect("fresh name");
    let element = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, vector);
    let zero = builder.constant(int, ConstValue::Int(0));
    let one = builder.constant(int, ConstValue::Int(1));
    let bound = builder.constant(int, ConstValue::Int(64));
    let delta = builder
        .module
        .intern_const(vector, ConstValue::Undef)
        .expect("a constant");

    let entry = builder.block("entry");
    let header = builder.block("loop.header");
    let body = builder.block("loop.body");
    let cont = builder.block("loop.cont");
    let merge = builder.block("loop.merge");

    let iv = builder.value(int);
    let cmp = builder.value(bool_ty);
    builder.emit(
        header,
        Inst::def(
            iv,
            int,
            Op::Phi {
                incomings: alloc::vec![(zero, entry), (ValueId(0), cont)],
            },
        ),
    );
    builder.emit(
        header,
        Inst::def(
            cmp,
            bool_ty,
            Op::Compare {
                op: CmpOp::SLessThan,
                lhs: iv,
                rhs: bound,
            },
        ),
    );
    builder.emit(
        header,
        Inst::none(Op::LoopMerge {
            merge,
            cont,
            control: 0,
        }),
    );
    builder.emit(
        header,
        Inst::none(Op::BranchConditional {
            cond: cmp,
            then: body,
            other: merge,
        }),
    );

    let access = builder.value(element);
    let loaded = builder.value(vector);
    let added = builder.value(vector);
    builder.emit(
        body,
        Inst::def(
            access,
            element,
            Op::AccessChain {
                base: arr,
                indices: alloc::vec![iv],
            },
        ),
    );
    builder.emit(body, Inst::def(loaded, vector, Op::Load { ptr: access }));
    builder.emit(
        body,
        Inst::def(
            added,
            vector,
            Op::Binary {
                op: BinOp::FAdd,
                lhs: loaded,
                rhs: delta,
            },
        ),
    );
    builder.emit(
        body,
        Inst::none(Op::Store {
            ptr: access,
            value: added,
        }),
    );
    builder.emit(body, Inst::none(Op::Branch { target: cont }));

    let next = builder.value(int);
    builder.emit(
        cont,
        Inst::def(
            next,
            int,
            Op::Binary {
                op: BinOp::IAdd,
                lhs: iv,
                rhs: one,
            },
        ),
    );
    builder.emit(cont, Inst::none(Op::Branch { target: header }));
    builder.emit(entry, Inst::none(Op::Branch { target: header }));
    builder.emit(merge, Inst::none(Op::Return));
    let blocks = {
        let function = builder
            .module
            .function_mut(builder.func)
            .expect("a function");
        function.body.as_mut().expect("a body")
    };
    if let Op::Phi { incomings } = &mut blocks[header.index()].insts[0].op {
        incomings[1].0 = next;
    }
    let mut module = builder.finish();
    verify(&module).expect("the module verifies before the rewrite");
    let before = print(&module);

    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies");
    assert_eq!(
        print(&module),
        before,
        "a loop over vectors is already a vector wide"
    );
}

#[test]
fn vectorizing_twice_changes_nothing() {
    let (builder, _parts) = counted_loop();
    let mut module = builder.finish();
    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies");
    let once = print(&module);

    vectorize(&mut module, &config()).expect("the module is rewritten again");
    verify(&module).expect("the module verifies");
    assert_eq!(print(&module), once, "the second run changed the module");
}

#[test]
fn vectorizing_a_group_twice_changes_nothing() {
    let mut builder = Builder::new(Vec::new());
    let float = builder.module.float_ty(32);
    let int = builder.module.int_ty(32, true);
    let array = builder.module.array_ty(float, 16);
    let array_pointer = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, array);
    let arr = builder
        .module
        .add_global("arr", array_pointer, Linkage::External, None)
        .expect("fresh name");
    let element = builder
        .module
        .ptr_ty(Storage::CrossWorkgroup, float);
    let entry = builder.block("entry");
    let mut results = Vec::new();
    for lane in 0..LANES {
        let index = builder.constant(int, ConstValue::Int(u64::from(lane)));
        let access = builder.value(element);
        let loaded = builder.value(float);
        let added = builder.value(float);
        builder.emit(
            entry,
            Inst::def(
                access,
                element,
                Op::AccessChain {
                    base: arr,
                    indices: alloc::vec![index],
                },
            ),
        );
        builder.emit(entry, Inst::def(loaded, float, Op::Load { ptr: access }));
        builder.emit(
            entry,
            Inst::def(
                added,
                float,
                Op::Binary {
                    op: BinOp::FAdd,
                    lhs: loaded,
                    rhs: loaded,
                },
            ),
        );
        results.push(added);
    }
    for result in results {
        let copy = builder.value(float);
        builder.emit(entry, Inst::def(copy, float, Op::CopyObject { operand: result }));
    }
    builder.emit(entry, Inst::none(Op::Return));
    let mut module = builder.finish();
    verify(&module).expect("the module verifies before the rewrite");
    vectorize(&mut module, &config()).expect("the module is rewritten");
    verify(&module).expect("the module verifies");
    let once = print(&module);

    vectorize(&mut module, &config()).expect("the module is rewritten again");
    verify(&module).expect("the module verifies");
    assert_eq!(print(&module), once, "the second run changed the module");
}

#[test]
fn a_disabled_vectorizer_changes_nothing() {
    let (builder, _parts) = counted_loop();
    let mut module = builder.finish();
    let before = print(&module);
    let config = VectorConfig {
        max_lanes: LANES,
        loop_vectorization: false,
        slp_vectorization: false,
        min_group_size: 2,
    };
    vectorize(&mut module, &config).expect("the module is untouched");
    verify(&module).expect("the module verifies");
    assert_eq!(print(&module), before);
}
