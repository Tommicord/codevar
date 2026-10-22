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

//! Native code generation for the Codevar OpenCL pipeline.
//!
//! This crate turns the intermediate representation of
//! [`codevar_ocl_ir`] into machine code.  The pipeline is a straight
//! line, and every stage is a module that can be used on its own:
//!
//! 1. **[`opt`] — optimization.**  The module is rewritten into a
//!    simpler but equivalent form: stack slots become registers, dead
//!    code disappears, common subexpressions are shared, control flow
//!    is folded, and calls are inlined.
//! 2. **[`vecz`] — vectorization.**  Straight-line groups of
//!    independent operations become vector operations, and counted
//!    loops with a unit stride are executed a vector at a time.
//! 3. **[`isel`] — selection.**  The optimized module becomes
//!    [`mir`], the machine IR: virtual registers, semantic operations,
//!    an explicit frame layout.
//! 4. **[`regalloc`] — register allocation.**  Every virtual register
//!    is given a home in the target's register file, or a slot in the
//!    frame when the file is too small.
//! 5. **[`target`] — code emission.**  An architecture back end
//!    expands the machine IR into the target's own instructions and
//!    encodes them.
//!
//! # The kernel ABI
//!
//! A compiled kernel is an `unsafe extern "C" fn(*mut WorkItem)`: it
//! takes the per-work-item state, reads the work-item identifiers from
//! it, and reaches its arguments through the argument block it points
//! at.  The layout of both structures is published by
//! [`codevar_ocl_abi`], which the driver and the generated code share.
//! The selection stage turns the work-item queries of the source into
//! [`mir::MOp::WorkItemLoad`] operations and the kernel's parameters
//! into [`mir::MOp::ArgLoad`] operations, so the back ends need to know
//! nothing about the dialect.
//!
//! # Example
//!
//! ```no_run
//! use codevar_ocl_native_codegen::pipeline::{CodegenOptions, compile_module};
//! use codevar_ocl_native_codegen::target::NativeTarget;
//!
//! fn emit(target: &dyn NativeTarget, module: &codevar_ocl_ir::ir::Module) {
//!     let options = CodegenOptions::default();
//!     let code = compile_module(module, target, &options).expect("the module compiles");
//!     let _bytes = code.bytes();
//! }
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod abi;
pub mod error;
pub mod interp;
pub mod isel;
pub mod mir;
pub mod opt;
pub mod pipeline;
pub mod regalloc;
pub mod target;
pub mod vecz;

pub use error::{CodegenError, Result};
pub use mir::{MInst, MOp, MType, MirBlock, MirFunction, MirModule, Operand, Operands, VReg};
pub use pipeline::{CodegenOptions, CompiledCode, compile_module};
pub use target::{CodeImage, NativeTarget, RegModel};
