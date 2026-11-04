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

//! The pipeline, from a dialect source to an allocated machine module.
//!
//! The test drives the whole chain with a target that records what it
//! is given instead of encoding it: the machine module the pipeline
//! hands to a back end must already be allocated — every operand and
//! every destination must name a physical register — and it must carry
//! the kernel ABI the driver expects.

use std::vec::Vec;

use codevar_ocl_native_codegen::mir::{MOp, MirModule, Operand, SymId};
use codevar_ocl_native_codegen::pipeline::{CodegenOptions, compile_module};
use codevar_ocl_native_codegen::target::{CodeImage, NativeTarget, RegModel};

/// The kernel the test compiles: it reads a work-item identifier and
/// multiplies the element it points at.
const KERNEL: &str = "\
#[kernel]
fn scale(out: *mut int, factor: int) -> void {
    let i = get_global_id(0);
    out[i] = out[i] * factor;
}
";

/// A target that records the machine module instead of encoding it.
struct RecordingTarget {
    /// The register file the allocator draws from.
    registers: RegModel,
    /// The module the pipeline handed over.
    seen: std::cell::RefCell<Option<MirModule>>,
}

impl RecordingTarget {
    /// Creates the target with a small register file.
    fn new() -> Self {
        let int_regs = (0..8)
            .map(codevar_ocl_native_codegen::mir::PReg::new)
            .collect();
        Self {
            registers: RegModel {
                int_regs,
                float_regs: Vec::new(),
                vec_regs: Vec::new(),
                callee_saved: Vec::new(),
                stack_pointer: codevar_ocl_native_codegen::mir::PReg::new(8),
                frame_pointer: None,
                arg_regs: Vec::new(),
                return_regs: Vec::new(),
                pointer_size: 8,
            },
            seen: std::cell::RefCell::new(None),
        }
    }
}

impl NativeTarget for RecordingTarget {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn registers(&self) -> &RegModel {
        &self.registers
    }

    fn lower_module(
        &self,
        module: &MirModule,
        entry: SymId,
    ) -> Result<CodeImage, codevar_ocl_native_codegen::error::CodegenError> {
        *self.seen.borrow_mut() = Some(module.clone());
        let _ = entry;
        Ok(CodeImage {
            bytes: Vec::new(),
            entry_offset: 0,
            instruction_count: 0,
        })
    }
}

/// Compiles the kernel and returns the machine module the target saw.
fn compile() -> MirModule {
    let module = match codevar_ocl_ir::lower::lower_source(KERNEL) {
        Ok(module) => module,
        Err(error) => panic!("the kernel lowers: {error}"),
    };
    let target = RecordingTarget::new();
    let options = CodegenOptions::default();
    if let Err(error) = compile_module(&module, &target, &options) {
        panic!("the kernel compiles: {error}");
    }
    target
        .seen
        .into_inner()
        .unwrap_or_else(|| panic!("the target saw a module"))
}

#[test]
fn the_entry_function_takes_the_work_item_pointer() {
    let module = compile();
    let entry = module
        .functions
        .first()
        .unwrap_or_else(|| panic!("the module has a function"));
    assert!(entry.is_entry);
    let work_item = entry
        .blocks
        .first()
        .and_then(|block| block.insts.first())
        .unwrap_or_else(|| panic!("the entry block has an instruction"));
    assert!(
        matches!(work_item.op, MOp::Param { index: 0 }),
        "the entry block does not read the work-item pointer"
    );
}

#[test]
fn the_kernel_arguments_are_read_from_the_argument_block() {
    let module = compile();
    let entry = module
        .functions
        .first()
        .unwrap_or_else(|| panic!("the module has a function"));
    let loads: Vec<_> = entry
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter(|inst| matches!(inst.op, MOp::ArgLoad { .. }))
        .collect();
    assert_eq!(loads.len(), 2, "the kernel has two arguments");
    let indices: Vec<u32> = loads
        .iter()
        .filter_map(|inst| match inst.op {
            MOp::ArgLoad { index, .. } => Some(index),
            _ => None,
        })
        .collect();
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn the_work_item_query_becomes_a_read_of_the_work_item() {
    let module = compile();
    let entry = module
        .functions
        .first()
        .unwrap_or_else(|| panic!("the module has a function"));
    let reads: Vec<_> = entry
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter(|inst| matches!(inst.op, MOp::WorkItemLoad { .. }))
        .collect();
    assert!(!reads.is_empty(), "the kernel reads no work-item field");
}

#[test]
fn every_operand_and_destination_is_physical() {
    let module = compile();
    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.insts {
                assert!(
                    inst.dest_reg().is_some() || inst.dest.is_none(),
                    "an instruction writes a register that was never allocated"
                );
                for operand in inst.args.as_slice() {
                    assert!(
                        !matches!(operand, Operand::Virtual(_)),
                        "an operand still names a virtual register"
                    );
                }
            }
        }
    }
}

#[test]
fn the_frame_is_reserved_for_the_stack_slots() {
    let module = compile();
    let entry = module
        .functions
        .first()
        .unwrap_or_else(|| panic!("the module has a function"));
    assert!(
        entry.frame_size >= entry.fixed_frame_size,
        "the frame is smaller than the slots the function reserved"
    );
}
