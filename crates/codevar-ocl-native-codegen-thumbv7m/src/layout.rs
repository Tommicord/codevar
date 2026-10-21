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
//! The layout gives every function and every block of the module an
//! address, and then writes the displacement of every branch.  The
//! branches of the expansion carry a placeholder until this point,
//! because the distance between two blocks is not known before the
//! blocks are placed.
//!
//! The code is position-independent: a branch is the only thing that
//! names an address, and it names it as a distance.  No absolute
//! address ever reaches the image, so the image can be copied into
//! executable memory and run there.

use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::{CodegenError, Result};

use crate::encode;
use crate::expand::{FixupKind, FixupTarget, FunctionCode};

/// The largest distance an unconditional branch reaches.
const BRANCH_RANGE: i64 = 1 << 23;

/// The largest distance a conditional branch reaches.
const BRANCH_COND_RANGE: i64 = 1 << 20;

/// The result of the layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// The bytes of the module.
    pub image: Vec<u8>,
    /// The offset of the entry function.
    pub entry_offset: u32,
    /// The number of instructions of the module.
    pub instructions: u32,
}

/// Places the functions of the module and writes the branches.
pub fn layout(functions: &[FunctionCode], entry: usize) -> Result<Layout> {
    let mut function_addresses: Vec<u32> = Vec::with_capacity(functions.len());
    let mut block_addresses: Vec<Vec<u32>> = Vec::with_capacity(functions.len());
    let mut address = 0u32;

    for function in functions {
        function_addresses.push(address);
        let mut blocks = Vec::with_capacity(function.blocks.len());
        let mut inner = address + u32::try_from(function.prologue.len()).unwrap_or(0);
        for block in &function.blocks {
            blocks.push(inner);
            inner += u32::try_from(block.bytes.len()).unwrap_or(0);
        }
        block_addresses.push(blocks);
        address = inner;
    }

    let mut image: Vec<u8> = Vec::with_capacity(address as usize);
    let mut instructions = 0u32;

    for (index, function) in functions.iter().enumerate() {
        image.extend_from_slice(&function.prologue);
        instructions += u32::try_from(function.prologue.len() / 2).unwrap_or(0);
        for (block_index, block) in function.blocks.iter().enumerate() {
            let base = block_addresses[index][block_index];
            let mut bytes = block.bytes.clone();
            for fixup in &block.fixups {
                let target = match fixup.target {
                    FixupTarget::Block(block) => {
                        let block = block as usize;
                        *block_addresses[index].get(block).ok_or_else(|| {
                            CodegenError::Target {
                                what: "laying out the module",
                                detail: String::from("a branch points at a block that does not exist"),
                            }
                        })?
                    }
                    FixupTarget::Function(function) => {
                        *function_addresses.get(function).ok_or_else(|| {
                            CodegenError::Target {
                                what: "laying out the module",
                                detail: String::from("a call points at a function that does not exist"),
                            }
                        })?
                    }
                    FixupTarget::Local(offset) => base + u32::try_from(offset).unwrap_or(0),
                };
                let at = base + u32::try_from(fixup.at).unwrap_or(0);
                let distance = i64::from(target) - (i64::from(at) + 4);
                patch(&mut bytes, fixup.at, fixup.kind, distance)?;
            }
            image.extend_from_slice(&bytes);
            instructions += block.instructions;
        }
    }

    let entry_offset = *function_addresses.get(entry).ok_or_else(|| CodegenError::Target {
        what: "laying out the module",
        detail: String::from("the entry function is not a function of the module"),
    })?;

    Ok(Layout {
        image,
        entry_offset,
        instructions,
    })
}

/// Writes the displacement of a branch.
fn patch(
    bytes: &mut [u8],
    at: usize,
    kind: FixupKind,
    distance: i64,
) -> Result<()> {
    let range = match kind {
        FixupKind::Branch | FixupKind::BranchLink => BRANCH_RANGE,
        FixupKind::BranchCond(_) => BRANCH_COND_RANGE,
    };
    if distance < -range || distance >= range {
        return Err(CodegenError::Target {
            what: "laying out the module",
            detail: String::from("a branch reaches too far to be encoded"),
        });
    }
    let distance = i32::try_from(distance).unwrap_or(0);
    let encoded = match kind {
        FixupKind::Branch => encode::branch_bytes(distance),
        FixupKind::BranchLink => encode::branch_link_bytes(distance),
        FixupKind::BranchCond(cond) => encode::branch_cond_bytes(cond, distance),
    };
    let end = at + 4;
    if end > bytes.len() {
        return Err(CodegenError::Target {
            what: "laying out the module",
            detail: String::from("a branch reaches past the end of its block"),
        });
    }
    bytes[at..end].copy_from_slice(&encoded);
    Ok(())
}
