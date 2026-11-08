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

//! Additional target-code API tests beyond [`compile.rs`].
//!
//! These probe control-flow kernels, explicit target selection, and the
//! host-target description.

use codevar_ocl_native_codegen_x86::X86Target;
use codevar_ocl_tc::{
    Error, OptLevel, compile_source_at, compile_source_for, find_entry, host_target_name, kernel_names, lower,
};

/// A kernel with a conditional and a loop.
const BRANCH: &str = "\
#[kernel]
fn branch(out: *mut int, n: int) -> void {
    let i = get_global_id(0);
    if i < n {
        out[i] = 1;
    } else {
        out[i] = 2;
    }
    let k = 0;
    while k < 4 {
        out[k] = k;
    }
}
";

/// A kernel that does not exist in `BRANCH`.
const MISSING: &str = "no_such_kernel";

#[test]
fn a_branch_and_loop_kernel_compiles_at_every_level() {
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let code = match compile_source_at(BRANCH, "branch", level) {
            Ok(code) => code,
            Err(error) => panic!("branch compiles at {level:?}: {error}"),
        };
        assert!(!code.bytes().is_empty(), "no code at {level:?}");
        assert!(code.instruction_count() > 0, "no instructions at {level:?}");
    }
}

#[test]
fn an_explicit_target_compiles_the_kernel() {
    let target = X86Target::new();
    let code = match compile_source_for(&target, BRANCH, "branch") {
        Ok(code) => code,
        Err(error) => panic!("compiles with an explicit target: {error}"),
    };
    assert!(!code.bytes().is_empty());
}

#[test]
fn host_target_name_is_non_empty() {
    let name = host_target_name();
    assert!(!name.is_empty(), "host target has a name: {name}");
}

#[test]
fn an_unknown_kernel_reports_unknown_entry() {
    let module = lower(BRANCH).expect("branch lowers");
    match find_entry(&module, MISSING) {
        Err(Error::UnknownEntry { name }) => assert_eq!(name, MISSING),
        other => panic!("expected UnknownEntry, got {other:?}"),
    }
}

#[test]
fn kernel_names_list_all_entry_points() {
    let source = format!("{BRANCH}\n#[kernel]\nfn other(out: *mut int) -> void {{ out[0] = 0; }}");
    let names = kernel_names(&source).expect("source lowers");
    assert_eq!(names, vec![String::from("branch"), String::from("other")]);
}
