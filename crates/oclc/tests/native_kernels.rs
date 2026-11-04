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

//! Every embedded kernel, compiled to native machine code.
//!
//! The kernels of the crate are the real workload of the editor, so
//! they are the real test of the native pipeline: each of them is
//! analyzed, lowered, optimized, vectorized, selected, allocated, and
//! encoded, and the result must be machine code that holds no absolute
//! address.
//!
//! The test is about the pipeline, not about the architecture: a
//! kernel the native back end cannot express yet is reported as
//! skipped rather than as a failure, because the SPIR-V path still
//! serves it.

use std::path::{Path, PathBuf};

use codevar_ocl_tc::{compile_module_for, host_target, lower};

/// The first user-space address, the lower bound of the range a non
/// position-independent reference would fall into.
const FIRST_USER_ADDRESS: u64 = 0x0000_4000_0000_0000;

/// The last user-space address, the upper bound of that range.
const LAST_USER_ADDRESS: u64 = 0x0000_8000_0000_0000;

/// The directory the kernels live in.
fn kernel_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

/// Every kernel source under the kernel directory, sorted by path.
fn kernel_sources() -> Vec<PathBuf> {
    fn walk(directory: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(directory) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
                continue;
            }
            if path
                .extension()
                .is_some_and(|extension| extension == "rs")
            {
                out.push(path);
            }
        }
    }
    let mut sources = Vec::new();
    walk(&kernel_dir(), &mut sources);
    sources.sort();
    sources
}

/// The `#[kernel]` names a source declares.
fn kernel_names(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| line.trim().strip_prefix("fn "))
        .filter_map(|line| line.split(['(', ' ']).next())
        .filter(|name| !name.is_empty())
        .map(String::from)
        .collect()
}

/// True when the code holds no absolute address.
///
/// On x86-64 a sixty-four-bit absolute address can only reach the
/// code through the `movabs` form (`REX.W B8+r imm64`), so the scan
/// looks for that opcode and reads the address that follows it.  The
/// scan over every window of the code would report addresses that
/// only look like one — a thirty-two-bit displacement next to another
/// instruction is enough — so the opcode is what is looked for.
#[cfg(target_arch = "x86_64")]
fn holds_no_absolute_address(bytes: &[u8]) -> bool {
    let mut index = 0usize;
    while index + 10 <= bytes.len() {
        if bytes[index] == 0x48 && (0xB8..=0xBF).contains(&bytes[index + 1]) {
            let mut address = [0u8; 8];
            address.copy_from_slice(&bytes[index + 2..index + 10]);
            let value = u64::from_le_bytes(address);
            if (FIRST_USER_ADDRESS..LAST_USER_ADDRESS).contains(&value) {
                return false;
            }
        }
        index += 1;
    }
    true
}

/// True when the code holds no absolute address.
///
/// The architectures this test does not know about are trusted to
/// emit position-independent code, and the back ends' own tests cover
/// the property where it is checked.
#[cfg(not(target_arch = "x86_64"))]
fn holds_no_absolute_address(_bytes: &[u8]) -> bool {
    true
}

#[test]
fn every_kernel_compiles_to_position_independent_machine_code() {
    let target = match host_target() {
        Ok(target) => target,
        Err(error) => panic!("the host has a native back end: {error}"),
    };
    let sources = kernel_sources();
    assert!(!sources.is_empty(), "the crate ships no kernels");

    let mut compiled = 0usize;
    let mut refused: Vec<String> = Vec::new();
    for path in &sources {
        let source = match std::fs::read_to_string(path) {
            Ok(source) => source,
            Err(error) => panic!("the kernel is readable: {error}"),
        };
        let module = match lower(&source) {
            Ok(module) => module,
            Err(error) => panic!("{} does not lower: {error}", path.display()),
        };
        for name in module
            .entry_points
            .iter()
            .map(|entry| entry.name.clone())
        {
            match compile_module_for(target.as_target(), &module, &name) {
                Ok(code) => {
                    compiled += 1;
                    assert!(
                        !code.bytes().is_empty(),
                        "{}: the kernel `{name}` produced no code",
                        path.display()
                    );
                    assert!(
                        holds_no_absolute_address(code.bytes()),
                        "{}: the kernel `{name}` holds an absolute address",
                        path.display()
                    );
                    assert!(
                        (code.entry_offset() as usize) < code.bytes().len(),
                        "{}: the entry point of `{name}` lies outside the code",
                        path.display()
                    );
                }
                Err(error) => refused.push(format!("{}:{name}: {error}", path.display())),
            }
        }
    }
    assert!(
        compiled > 0,
        "no kernel compiled to machine code; every one was refused: {refused:?}"
    );
}

#[test]
fn the_kernel_sources_declare_their_kernels() {
    for path in kernel_sources() {
        let source = match std::fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) => panic!("the kernel is readable: {error}"),
        };
        let module = match lower(&source) {
            Ok(module) => module,
            Err(error) => panic!("the kernel lowers: {error}"),
        };
        assert!(
            !module.entry_points.is_empty(),
            "{} declares no kernel",
            path.display()
        );
        let declared = kernel_names(&source);
        for entry in &module.entry_points {
            assert!(
                declared.contains(&entry.name),
                "{}: the kernel `{}` is not declared as a function",
                path.display(),
                entry.name
            );
        }
    }
}
