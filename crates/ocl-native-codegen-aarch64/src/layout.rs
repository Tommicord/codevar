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

//! The layout of the module.
//!
//! The functions of the module are laid out one after the other, in
//! the order the module holds them, and the blocks of a function are
//! laid out one after the other, in the order the function holds
//! them.  A branch inside a function and a call to another function
//! of the module are relative: the displacement is counted in words
//! from the instruction itself, so the code holds no address and can
//! be moved anywhere.
//!
//! # The reach of a branch
//!
//! The unconditional branches carry a twenty-six-bit displacement,
//! which reaches about a hundred and twenty-eight mebibytes in each
//! direction; the conditional ones carry a nineteen-bit
//! displacement, which reaches about one mebibyte.  A branch that
//! reaches further is reported as an error instead of being
//! truncated.

use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::{CodegenError, Result};
use codevar_ocl_native_codegen::mir::{MirModule, SymId};
use codevar_ocl_native_codegen::target::CodeImage;

use crate::expand::{self, Fixup, FunctionCode};

/// The number of bits of the displacement of an unconditional
/// branch.
const WIDE_SHIFT: u32 = 26;

/// The number of bits of the displacement of a conditional branch.
const NARROW_SHIFT: u32 = 19;

/// Lays the module out and returns the machine code.
///
/// # Errors
///
/// Returns an error when a function cannot be expanded, when a
/// branch reaches further than the architecture allows, or when the
/// entry symbol does not name a function of the module.
pub fn lower(module: &MirModule, entry: SymId) -> Result<CodeImage> {
    let entry_index = module
        .function_index(entry)
        .ok_or_else(|| CodegenError::Target {
            what: "laying the module out",
            detail: alloc::string::String::from("the entry symbol does not name a function of the module"),
        })?;

    let mut codes: Vec<FunctionCode> = Vec::with_capacity(module.functions.len());
    for function in &module.functions {
        codes.push(expand::expand(module, function)?);
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(codes.len());
    let mut total = 0u32;
    for code in &codes {
        offsets.push(total);
        total += code.bytes.len() as u32;
    }

    let mut bytes: Vec<u8> = Vec::with_capacity(total as usize);
    for (index, code) in codes.iter().enumerate() {
        let base = offsets[index];
        bytes.extend_from_slice(&code.bytes);
        patch(code, base, &offsets, &mut bytes)?;
    }
    let instructions: u32 = codes.iter().map(|code| code.instructions).sum();

    Ok(CodeImage {
        bytes,
        entry_offset: offsets[entry_index],
        instruction_count: instructions,
    })
}

/// Patches the branches and the calls of one function.
fn patch(code: &FunctionCode, base: u32, function_offsets: &[u32], bytes: &mut [u8]) -> Result<()> {
    for fixup in &code.fixups {
        match *fixup {
            Fixup::Branch { at, block, wide } => {
                let at = base + at;
                let start = *code
                    .blocks
                    .get(block as usize)
                    .ok_or_else(|| CodegenError::Target {
                        what: "laying the module out",
                        detail: alloc::format!("the branch goes to the block {block}, which does not exist"),
                    })?;
                let target = base + start;
                let words = displacement(target, at, wide)?;
                if wide {
                    let word = 0x1400_0000 | (words as u32 & 0x03FF_FFFF);
                    write_word(bytes, at, word);
                } else {
                    let existing = read_word(bytes, at);
                    let word = existing | ((words as u32 & 0x0007_FFFF) << 5);
                    write_word(bytes, at, word);
                }
            }
            Fixup::Call { at, function } => {
                let at = base + at;
                let target = *function_offsets
                    .get(function)
                    .ok_or_else(|| CodegenError::Target {
                        what: "laying the module out",
                        detail: alloc::format!(
                            "the call goes to the function {function}, which does not exist"
                        ),
                    })?;
                let words = displacement(target, at, true)?;
                let word = 0x9400_0000 | (words as u32 & 0x03FF_FFFF);
                write_word(bytes, at, word);
            }
        }
    }
    Ok(())
}

/// The displacement of a branch, counted in words.
fn displacement(target: u32, at: u32, wide: bool) -> Result<i64> {
    let bytes = i64::from(target) - i64::from(at);
    if bytes % 4 != 0 {
        return Err(CodegenError::Target {
            what: "laying the module out",
            detail: alloc::string::String::from("a branch does not land on an instruction"),
        });
    }
    let words = bytes / 4;
    let shift = if wide { WIDE_SHIFT } else { NARROW_SHIFT };
    let limit = 1i64 << (shift - 1);
    if words < -limit || words >= limit {
        return Err(CodegenError::Target {
            what: "laying the module out",
            detail: alloc::format!("the branch reaches {bytes} bytes, which is out of range"),
        });
    }
    Ok(words)
}

/// Reads a word from the image.
fn read_word(bytes: &[u8], at: u32) -> u32 {
    let start = at as usize;
    u32::from_le_bytes([bytes[start], bytes[start + 1], bytes[start + 2], bytes[start + 3]])
}

/// Writes a word into the image.
fn write_word(bytes: &mut [u8], at: u32, word: u32) {
    let start = at as usize;
    bytes[start..start + 4].copy_from_slice(&word.to_le_bytes());
}
