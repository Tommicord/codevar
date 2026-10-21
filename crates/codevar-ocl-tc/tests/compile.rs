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

//! Tests of the target code API.
//!
//! The assertions are structural: the back ends are free to choose
//! their own prologue and to order their blocks, so the tests check
//! that the output *is* machine code — it holds instructions, its
//! entry point lies inside it, and it carries no absolute address.

use codevar_ocl_tc::{Error, compile_source, find_entry, kernel_names, lower};

/// A kernel with a loop and a work-item query.
const KERNEL: &str = "\
#[kernel]
fn scale(out: *mut int, factor: int) -> void {
    let i = get_global_id(0);
    out[i] = out[i] * factor;
}
";

/// A second kernel, to tell the entry points apart.
const OTHER: &str = "\
#[kernel]
fn fill(out: *mut int) -> void {
    out[0] = 7;
}
";

/// The first user-space address, the lower bound of the range a
/// non position-independent reference would fall into.
const FIRST_USER_ADDRESS: u64 = 0x0000_4000_0000_0000;

/// The last user-space address, the upper bound of that range.
const LAST_USER_ADDRESS: u64 = 0x0000_8000_0000_0000;

/// True when no eight-byte window of `bytes` holds a canonical
/// user-space address.
fn holds_no_absolute_address(bytes: &[u8]) -> bool {
    bytes.windows(8).all(|window| {
        let value = u64::from_le_bytes([
            window[0], window[1], window[2], window[3], window[4], window[5], window[6], window[7],
        ]);
        !(FIRST_USER_ADDRESS..LAST_USER_ADDRESS).contains(&value)
    })
}

#[test]
fn the_source_lowers_to_the_intermediate_representation() {
    let module = lower(KERNEL).expect("the kernel lowers");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "scale");
}

#[test]
fn the_kernel_names_are_listed_in_source_order() {
    let source = format!("{KERNEL}{OTHER}");
    let names = kernel_names(&source).expect("the source lowers");
    assert_eq!(names, vec![String::from("scale"), String::from("fill")]);
}

#[test]
fn an_unknown_kernel_is_reported() {
    let module = lower(KERNEL).expect("the kernel lowers");
    match find_entry(&module, "missing") {
        Err(Error::UnknownEntry { name }) => assert_eq!(name, "missing"),
        other => panic!("expected an unknown entry point, got {other:?}"),
    }
}

#[test]
fn a_broken_source_is_reported() {
    assert!(lower("#[kernel]\nfn broken(n: noway) { }\n").is_err());
}

#[test]
fn the_compiled_code_is_machine_code() {
    let code = match compile_source(KERNEL, "scale") {
        Ok(code) => code,
        Err(error) => panic!("the kernel compiles: {error}"),
    };
    assert!(!code.bytes().is_empty(), "the kernel produced no code");
    assert!(
        (code.entry_offset() as usize) < code.bytes().len(),
        "the entry offset {} is outside the {} bytes of code",
        code.entry_offset(),
        code.bytes().len()
    );
    assert!(code.instruction_count() > 0);
    assert!(
        holds_no_absolute_address(code.bytes()),
        "the code holds an absolute address: {:02X?}",
        code.bytes()
    );
}
