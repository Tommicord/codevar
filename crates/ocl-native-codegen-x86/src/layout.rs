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
//! The functions of the module are placed one after another, each of
//! them starting at a multiple of sixteen bytes.  Once every place is
//! known, the branches between the blocks of a function and the calls
//! between the functions of the module are given their
//! displacements.
//!
//! The displacements are relative, so the code holds no address at
//! all: it can be copied anywhere and entered there.

use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::CodegenError;
use codevar_ocl_native_codegen::target::CodeImage;

use crate::expand::Expanded;

/// The target of a displacement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FixupKind {
    /// A branch to a block of the same function.
    Block(u32),
    /// A call to a function of the module.
    Function(usize),
}

/// A place whose displacement the layout fills in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Fixup {
    /// The offset of the displacement field inside the function.
    pub site: u32,
    /// What the displacement reaches.
    pub kind: FixupKind,
}

impl Fixup {
    /// A branch to a block of the same function.
    pub(crate) fn block(site: u32, block: u32) -> Self {
        Self {
            site,
            kind: FixupKind::Block(block),
        }
    }

    /// A call to a function of the module.
    pub(crate) fn function(site: u32, function: usize) -> Self {
        Self {
            site,
            kind: FixupKind::Function(function),
        }
    }
}

/// Places the functions of the module and fills in the displacements.
///
/// `entry` is the index of the entry function.
pub(crate) fn lay_out(expanded: &[Expanded], entry: usize) -> Result<CodeImage, CodegenError> {
    // Every function starts at a multiple of sixteen bytes, so the
    // code stays aligned when it is entered.
    let mut offsets: Vec<u32> = Vec::with_capacity(expanded.len());
    let mut cursor = 0u32;
    for function in expanded {
        cursor = cursor.div_ceil(16) * 16;
        offsets.push(cursor);
        cursor += u32::try_from(function.bytes.len()).unwrap_or(u32::MAX);
    }
    let total = usize::try_from(cursor).unwrap_or(usize::MAX);
    let mut bytes: Vec<u8> = Vec::with_capacity(total);
    for (index, function) in expanded.iter().enumerate() {
        while bytes.len() < offsets[index] as usize {
            // The padding between two functions holds no operation.
            bytes.push(0x90);
        }
        bytes.extend_from_slice(&function.bytes);
    }
    let mut instruction_count = 0u32;
    for function in expanded {
        instruction_count += function.instruction_count;
    }
    // The displacements are relative: the target lies at the end of
    // the displacement field.
    for (index, function) in expanded.iter().enumerate() {
        for fixup in &function.fixups {
            let site = offsets[index] + fixup.site;
            let target = match fixup.kind {
                FixupKind::Block(block) => {
                    let block = block as usize;
                    match function.block_offsets.get(block) {
                        Some(offset) => offsets[index] + offset,
                        None => {
                            return Err(CodegenError::Target {
                                what: "laying out the module",
                                detail: alloc::string::String::from(
                                    "a branch names a block the function does not hold",
                                ),
                            });
                        }
                    }
                }
                FixupKind::Function(function) => match offsets.get(function) {
                    Some(offset) => *offset,
                    None => {
                        return Err(CodegenError::Target {
                            what: "laying out the module",
                            detail: alloc::string::String::from(
                                "a call names a function the module does not hold",
                            ),
                        });
                    }
                },
            };
            let displacement = i64::from(target) - i64::from(site) - 4;
            let displacement = match i32::try_from(displacement) {
                Ok(value) => value,
                Err(_) => {
                    return Err(CodegenError::Target {
                        what: "laying out the module",
                        detail: alloc::string::String::from(
                            "the distance to the target of a branch or a call does not fit in thirty-two bits",
                        ),
                    });
                }
            };
            let start = site as usize;
            let end = start + 4;
            match bytes.get_mut(start..end) {
                Some(field) => field.copy_from_slice(&displacement.to_le_bytes()),
                None => {
                    return Err(CodegenError::Target {
                        what: "laying out the module",
                        detail: alloc::string::String::from(
                            "a displacement lies outside the code of the module",
                        ),
                    });
                }
            }
        }
    }
    let entry_offset = match offsets.get(entry) {
        Some(offset) => *offset,
        None => {
            return Err(CodegenError::Target {
                what: "laying out the module",
                detail: alloc::string::String::from("the entry point does not name a function of the module"),
            });
        }
    };
    Ok(CodeImage {
        bytes,
        entry_offset,
        instruction_count,
    })
}
