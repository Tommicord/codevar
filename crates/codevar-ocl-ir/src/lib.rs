//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Codevar intermediate representation (IR).
//!
//! The IR sits between the semantic analyzer ([`codevar_ocl_sar`]) and the
//! binary backends (SPIR-V, and later MSL and PTX).  Its textual form is
//! shaped like SPIR-V assembly so that IR, SPIR-V text and binary stay
//! one-to-one; its expression lowering is shaped like LLVM IR and Clang's
//! code generator: every local variable lives in a stack slot and reads and
//! writes go through `load`/`store`, so staged scalar optimizations
//! (DCE, CFG folding, `mem2reg`, GVN, inlining, LICM) have exactly the
//! structure they expect.
//!
//! # Syntax
//!
//! A module is a sequence of lines: an optional `target` header, type and
//! constant declarations, global variables, entry-point declarations, and
//! function definitions.  Identifiers are `%`-prefixed; `;` starts a
//! comment.
//!
//! ```text
//! ; Codevar IR 0.1
//! target opencl address physical64 memory opencl
//!
//! %void = OpTypeVoid
//! %int = OpTypeInt 32 1
//! %uint = OpTypeInt 32 0
//! %float = OpTypeFloat 32
//! %ptr_cw_float = OpTypePointer CrossWorkgroup %float
//! %fn_uint_int = OpTypeFunction %uint %int
//! %c_0 = OpConstant %int 0
//!
//! OpEntryPoint Kernel %vector_add "vector_add"
//!
//! %vector_add = OpFunction %void None %fn_void
//!     %a = OpFunctionParameter %ptr_cw_float
//!     %entry = OpLabel
//!     %gid = OpFunctionCall %uint %get_global_id %c_0
//!     %v = OpLoad %float %ptr
//!     OpStore %ptr %v
//!     OpReturn
//! OpFunctionEnd
//! ```
//!
//! Instructions carry their result type exactly as SPIR-V assembly does
//! (`%r = OpIAdd %int %a %b`), result-less instructions stand alone
//! (`OpReturn`), and structured control flow is explicit
//! (`OpSelectionMerge`, `OpLoopMerge`, `OpBranch`, `OpBranchConditional`).
//!
//! # Stages
//!
//! 1. [`lower`] — AST plus semantic tables to [`ir::Module`].
//! 2. [`print`] / [`parse`] — textual form, round-trip stable.
//! 3. [`verify`] — type and structural checks on the IR itself.
//! 4. [`spirv`] — module to validated SPIR-V binary words and `.spv` bytes.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod ir;
pub mod lower;
pub mod parse;
pub mod print;
pub mod spirv;
pub mod verify;
