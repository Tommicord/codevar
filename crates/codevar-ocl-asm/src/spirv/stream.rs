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

//! Streaming SPIR-V assembly for the bounded-memory pipeline.
//!
//! [`SpirvStream`] is the counterpart of [`assemble`]: instead of
//! holding the whole module's words and id assignment at once, it
//! grows the module in three phases whose peak memory is proportional
//! to one function body:
//!
//! * [`prelude`](SpirvStream::prelude) assigns ids for everything the
//!   module already declares — types, values, extended-instruction
//!   sets, labels of pre-existing bodies — and writes the sections
//!   that depend only on that state (scaffold, memory model, built-in
//!   annotations);
//! * [`emit_function_body`](SpirvStream::emit_function_body) pulls in
//!   the types, values, and annotations one body introduced, then
//!   appends its `OpFunction` definition;
//! * [`finish`](SpirvStream::finish) closes the module with the
//!   capability list (computed from the final type arena), the entry
//!   points, and every function that never received a body.
//!
//! Section buffers decouple write time from layout order: the final
//! words concatenate the buffers in the SPIR-V logical layout (SPIR-V
//! §2.4) no matter when each buffer was written, so a body emitted
//! after the prelude still lands after every declaration.
//!
//! Ids are never renumbered.  Values reclaimed with
//! [`Module::truncate_values`] keep their ids reserved — only the
//! parallel id vector shrinks — so bodies written earlier stay valid.
//!
//! Relative to [`assemble`], a streamed module differs only where the
//! whole-module view is unavailable:
//!
//! * `OpName` covers module-scope values, blocks, and types, but not
//!   per-instruction temporaries and parameters (whole-module names
//!   for them are dropped by name-filtering comparisons);
//! * an array type first seen after the prelude carries its length
//!   constant (and the `uint32` type it needs) inline in the type
//!   region instead of in the scaffold;
//! * ids match [`assemble`] exactly for any module fully present at
//!   [`prelude`]; when bodies arrive incrementally, label and
//!   body-value ids are allocated in arrival order instead.
//!
//! Like [`assemble`], the stream does not run the IR verifier; callers
//! should [`verify_function`] each body before emitting it.
//!
//! [`assemble`]: super::assemble
//! [`Module::truncate_values`]: codevar_ocl_ir::ir::Module::truncate_values
//! [`verify_function`]: codevar_ocl_ir::verify::verify_function

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{Decor, ExecutionModel, Linkage, Module, Storage, Type, ValueId, ValueKind};
use codevar_ocl_ir::spirv::ops::{
    self, OP_CONSTANT, OP_DECORATE, OP_ENTRYPOINT, OP_EXTINSTIMPORT, OP_NAME, OP_VARIABLE,
};

use super::{
    Encoder, Layout, block_id, builtin_number, check_supported, emit_capabilities, emit_constant,
    emit_function, emit_memory_model, emit_type, storageclass_of,
};
use crate::AssembleError;

/// A SPIR-V module under construction, assembled section by section.
///
/// One instance assembles exactly one module; the section buffers it
/// owns are concatenated by [`finish`](Self::finish) in logical-layout
/// order.  See the [module documentation](self) for how the three
/// phases map onto the whole-module assembler and where the results
/// diverge.
pub struct SpirvStream {
    /// Header; the capability list is appended by [`finish`](Self::finish).
    head: Encoder,
    /// `OpExtInstImport` list: between capabilities and the memory model.
    imports: Encoder,
    /// `OpMemoryModel` and the `OpEntryPoint` list.
    tail: Encoder,
    /// `OpName` debug instructions.
    names: Encoder,
    /// `OpDecorate` built-in annotations.
    builtins: Encoder,
    /// `OpDecorate` linkage attributes for module-scope variables.
    link_globals: Encoder,
    /// `OpDecorate` linkage attributes for functions.
    link_functions: Encoder,
    /// Array-length `uint32` type and `OpConstant`s for declared arrays.
    scaffold: Encoder,
    /// `OpType*` instructions.
    types: Encoder,
    /// `OpConstant`/`OpUndef` instructions for interned values.
    consts: Encoder,
    /// `OpVariable` module-scope variables.
    globals: Encoder,
    /// `OpFunction` declarations (bodies that never arrived).
    func_decls: Encoder,
    /// `OpFunction` definitions, in emission order.
    func_defs: Encoder,
    /// The `<id>` assignment, shared with the whole-module assembler.
    layout: Layout,
    /// Canonical structural key → assigned id (the dedup table).
    canonical: BTreeMap<Type, u32>,
    /// Type ids already written; deduplicates shared canonical ids.
    emitted: BTreeSet<u32>,
    /// Array length → id of its `OpConstant`.
    lengths: BTreeMap<u32, u32>,
    /// Types written up to this index of [`Module::types`].
    types_emitted: usize,
    /// Values assigned up to this index of [`Module::values`].
    values_synced: usize,
    /// Annotations copied from [`Module::decorations`].
    decor_synced: usize,
    /// `Input`/`Output` global ids for the entry-point interface.
    interface: Vec<u32>,
    /// Functions whose definitions went into `func_defs`.
    emitted_functions: BTreeSet<ValueId>,
}

impl SpirvStream {
    /// Starts a module from the state `module` already holds.
    ///
    /// Assigns ids in the whole-module order — types, values,
    /// extended-instruction sets, labels of pre-existing bodies, then
    /// the array-length scaffold — writes the declared region
    /// (constants, globals, type names, built-in annotations) and the
    /// memory model, and leaves everything body-dependent for
    /// [`emit_function_body`](Self::emit_function_body).  Capabilities
    /// are deferred to [`finish`](Self::finish) because bodies may
    /// still add wide types.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] for a non-Kernel entry point, a
    /// malformed declared region (a global that is not a pointer, a
    /// type used before it is declared, an unknown `BuiltIn` name),
    /// or a string the encoder rejects; [`AssembleError::IdExhausted`]
    /// when ids run out.
    pub fn prelude(module: &Module) -> Result<Self, AssembleError> {
        check_supported(module)?;
        let mut stream = Self {
            head: Encoder::new(),
            imports: Encoder { words: Vec::new() },
            tail: Encoder { words: Vec::new() },
            names: Encoder { words: Vec::new() },
            builtins: Encoder { words: Vec::new() },
            link_globals: Encoder { words: Vec::new() },
            link_functions: Encoder { words: Vec::new() },
            scaffold: Encoder { words: Vec::new() },
            types: Encoder { words: Vec::new() },
            consts: Encoder { words: Vec::new() },
            globals: Encoder { words: Vec::new() },
            func_decls: Encoder { words: Vec::new() },
            func_defs: Encoder { words: Vec::new() },
            layout: Layout {
                types: Vec::new(),
                values: Vec::new(),
                ext_sets: Vec::new(),
                labels: Vec::new(),
                next: 1,
            },
            canonical: BTreeMap::new(),
            emitted: BTreeSet::new(),
            lengths: BTreeMap::new(),
            types_emitted: 0,
            values_synced: 0,
            decor_synced: 0,
            interface: Vec::new(),
            emitted_functions: BTreeSet::new(),
        };
        stream.assign_types(module)?;
        stream.sync_values(module)?;
        stream.sync_ext_sets(module)?;
        for &function in module.functions() {
            let def = module
                .function(function)
                .ok_or(AssembleError::Unsupported {
                    what: "a non-function value in the function list",
                })?;
            let mut labels = Vec::new();
            if let Some(body) = &def.body {
                for _block in body {
                    labels.push(stream.layout.alloc()?);
                }
            }
            stream.layout.labels.push(labels);
        }
        stream.ensure_scaffold(module)?;
        stream.sync_types(module)?;
        emit_memory_model(module, &mut stream.tail)?;
        stream.sync_decorations(module)?;
        Ok(stream)
    }

    /// Emits one function's definition into the definition region.
    ///
    /// First pulls in every type, value, extended-instruction set, and
    /// annotation introduced since the previous call, allocates label
    /// ids for the body's blocks when they were not assigned at
    /// [`prelude`](Self::prelude), records the block names, then
    /// appends `OpFunction` … `OpFunctionEnd`.  The body must be
    /// attached to `module` (see [`Module::begin_body`]); detaching it
    /// afterward with [`Module::take_function_body`] is the caller's
    /// job.
    ///
    /// [`Module::begin_body`]: codevar_ocl_ir::ir::Module::begin_body
    /// [`Module::take_function_body`]: codevar_ocl_ir::ir::Module::take_function_body
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when `function` is not in
    /// [`Module::functions`], has no body, was already emitted, or
    /// uses a construct the backend cannot encode;
    /// [`AssembleError::IdExhausted`] when ids run out; encoder
    /// errors for oversized instructions or strings.
    pub fn emit_function_body(&mut self, module: &Module, function: ValueId) -> Result<(), AssembleError> {
        let index = module
            .functions()
            .iter()
            .position(|&id| id == function)
            .ok_or(AssembleError::Unsupported {
                what: "a function outside the function list",
            })?;
        if self.emitted_functions.contains(&function) {
            return Err(AssembleError::Unsupported {
                what: "a function emitted twice",
            });
        }
        let def = module
            .function(function)
            .ok_or(AssembleError::Unsupported {
                what: "a non-function value in the function list",
            })?;
        let Some(body) = &def.body else {
            return Err(AssembleError::Unsupported {
                what: "a function without a body",
            });
        };
        self.sync(module)?;
        while self.layout.labels.len() <= index {
            self.layout.labels.push(Vec::new());
        }
        if self.layout.labels[index].is_empty() {
            for _block in body {
                let id = self.layout.alloc()?;
                self.layout.labels[index].push(id);
            }
        }
        for (block, basic) in body.iter().enumerate() {
            if basic.name.is_empty() {
                continue;
            }
            let label = self.layout.label(index, block_id(block)?)?;
            self.names.instruction(OP_NAME, |e| {
                e.word(label);
                e.string(&basic.name)?;
                Ok(())
            })?;
        }
        emit_function(module, index, &self.layout, &mut self.func_defs)?;
        self.emitted_functions.insert(function);
        Ok(())
    }

    /// Closes the module and returns its SPIR-V words.
    ///
    /// Synchronizes the remaining module state, emits the capability
    /// list computed from the final type arena, one `OpEntryPoint`
    /// per kernel (with the collected `Input`/`Output` interface),
    /// and every function still without a body as a declaration, then
    /// concatenates the section buffers in logical-layout order:
    /// capabilities, imports, memory model, entry points, names,
    /// annotations, scaffold, types, constants, globals, function
    /// declarations, function definitions.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when an entry point uses a
    /// non-Kernel execution model, a function with a body was never
    /// emitted through [`emit_function_body`](Self::emit_function_body),
    /// or an id is out of range; [`AssembleError::IdExhausted`] when
    /// ids run out; encoder errors for oversized instructions or
    /// strings.
    pub fn finish(mut self, module: &Module) -> Result<Vec<u32>, AssembleError> {
        check_supported(module)?;
        self.sync(module)?;
        emit_capabilities(module, &mut self.head)?;
        for entry in &module.entry_points {
            let model = match entry.model {
                ExecutionModel::Kernel => ops::EXECUTIONMODEL_KERNEL,
                ExecutionModel::GLCompute => {
                    return Err(AssembleError::Unsupported {
                        what: "a non-Kernel execution model",
                    });
                }
            };
            let function = self.layout.value(entry.func)?;
            let interface = &self.interface;
            self.tail.instruction(OP_ENTRYPOINT, |e| {
                e.word(model);
                e.word(function);
                e.string(&entry.name)?;
                for &global in interface {
                    e.word(global);
                }
                Ok(())
            })?;
        }
        for (index, &function) in module.functions().iter().enumerate() {
            if self.emitted_functions.contains(&function) {
                continue;
            }
            let def = module
                .function(function)
                .ok_or(AssembleError::Unsupported {
                    what: "a non-function value in the function list",
                })?;
            if !def.is_declaration() {
                return Err(AssembleError::Unsupported {
                    what: "a function body that was never emitted",
                });
            }
            emit_function(module, index, &self.layout, &mut self.func_decls)?;
        }
        let bound = self.layout.next;
        let Self {
            head,
            imports,
            tail,
            names,
            builtins,
            link_globals,
            link_functions,
            scaffold,
            types,
            consts,
            globals,
            func_decls,
            func_defs,
            ..
        } = self;
        let mut words = head.finish(bound);
        for mut section in [
            imports,
            tail,
            names,
            builtins,
            link_globals,
            link_functions,
            scaffold,
            types,
            consts,
            globals,
            func_decls,
            func_defs,
        ] {
            words.append(&mut section.words);
        }
        Ok(words)
    }

    /// Assigns ids for every type declared so far that has none,
    /// mirroring [`Layout::assign`]'s canonical-dedup pass.
    ///
    /// [`Layout::assign`]: super::Layout::assign
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when a child type is declared
    /// after its parent; [`AssembleError::IdExhausted`] when ids run
    /// out.
    fn assign_types(&mut self, module: &Module) -> Result<(), AssembleError> {
        while self.layout.types.len() < module.types().len() {
            let index = self.layout.types.len();
            let key = self.layout.canonical(&module.types()[index].ty)?;
            let id = match self.canonical.get(&key) {
                Some(&id) => id,
                None => {
                    let id = self.layout.alloc()?;
                    self.canonical.insert(key, id);
                    id
                }
            };
            self.layout.types.push(id);
        }
        Ok(())
    }

    /// Writes the array-length scaffold — the `uint32` type and one
    /// `OpConstant` per declared array length — into the buffer that
    /// precedes the type region, mirroring
    /// `emit_types_constants_globals` so declared arrays keep the
    /// whole-module id assignment.
    ///
    /// # Errors
    ///
    /// [`AssembleError::IdExhausted`] when ids run out; encoder
    /// errors for oversized instructions.
    fn ensure_scaffold(&mut self, module: &Module) -> Result<(), AssembleError> {
        if !module
            .types()
            .iter()
            .any(|def| matches!(def.ty, Type::Array { .. }))
        {
            return Ok(());
        }
        let uint32 = match self.canonical.get(&Type::Int {
            bits: 32,
            signed: false,
        }) {
            Some(&id) => id,
            None => self.layout.alloc()?,
        };
        emit_type(
            uint32,
            &Type::Int {
                bits: 32,
                signed: false,
            },
            &self.layout,
            &self.lengths,
            &mut self.scaffold,
            &mut self.emitted,
        )?;
        let array_lengths = module
            .types()
            .iter()
            .filter_map(|def| match def.ty {
                Type::Array { len, .. } => Some(len),
                _ => None,
            })
            .collect::<BTreeSet<u32>>();
        for len in array_lengths {
            let id = self.layout.alloc()?;
            self.scaffold.instruction(OP_CONSTANT, |e| {
                e.word(uint32);
                e.word(id);
                e.word(len);
                Ok(())
            })?;
            self.lengths.insert(len, id);
        }
        Ok(())
    }

    /// Provides the `uint32` type and length constant for an array
    /// type first seen after the prelude (a body-created array).
    ///
    /// Both go into the type region immediately before the array
    /// type, which is where the whole-module assembler's scaffold
    /// would have placed them relative to that type's references.
    ///
    /// # Errors
    ///
    /// [`AssembleError::IdExhausted`] when ids run out; encoder
    /// errors for oversized instructions.
    fn ensure_length(&mut self, len: u32) -> Result<(), AssembleError> {
        if self.lengths.contains_key(&len) {
            return Ok(());
        }
        let uint32 = match self.canonical.get(&Type::Int {
            bits: 32,
            signed: false,
        }) {
            Some(&id) => id,
            None => self.layout.alloc()?,
        };
        if !self.emitted.contains(&uint32) {
            emit_type(
                uint32,
                &Type::Int {
                    bits: 32,
                    signed: false,
                },
                &self.layout,
                &self.lengths,
                &mut self.types,
                &mut self.emitted,
            )?;
        }
        let id = self.layout.alloc()?;
        self.types.instruction(OP_CONSTANT, |e| {
            e.word(uint32);
            e.word(id);
            e.word(len);
            Ok(())
        })?;
        self.lengths.insert(len, id);
        Ok(())
    }

    /// Assigns and writes every type declared since the last sync.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when the arena breaks
    /// declare-before-use order, a type cannot be encoded, or a name
    /// is too long; [`AssembleError::IdExhausted`] when ids run out.
    fn sync_types(&mut self, module: &Module) -> Result<(), AssembleError> {
        self.assign_types(module)?;
        while self.types_emitted < module.types().len() {
            let index = self.types_emitted;
            let def = &module.types()[index];
            let id = self
                .layout
                .types
                .get(index)
                .copied()
                .ok_or(AssembleError::Unsupported {
                    what: "an out-of-range type id",
                })?;
            if let Type::Array { len, .. } = def.ty {
                self.ensure_length(len)?;
            }
            emit_type(
                id,
                &def.ty,
                &self.layout,
                &self.lengths,
                &mut self.types,
                &mut self.emitted,
            )?;
            if !def.name.is_empty() {
                self.names.instruction(OP_NAME, |e| {
                    e.word(id);
                    e.string(&def.name)?;
                    Ok(())
                })?;
            }
            self.types_emitted = index + 1;
        }
        Ok(())
    }

    /// Assigns and writes every value created since the last sync.
    ///
    /// When the arena shrank through
    /// [`Module::truncate_values`], only the parallel id vector is
    /// truncated — ids already handed out stay reserved, so bodies
    /// written earlier keep valid references.
    ///
    /// [`Module::truncate_values`]: codevar_ocl_ir::ir::Module::truncate_values
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] when a payload cannot be
    /// encoded (a global whose type is not a pointer, a constant that
    /// does not fit its type) or an id is out of range;
    /// [`AssembleError::IdExhausted`] when ids run out; encoder
    /// errors for oversized instructions or strings.
    fn sync_values(&mut self, module: &Module) -> Result<(), AssembleError> {
        if module.values().len() < self.values_synced {
            self.layout.values.truncate(module.values().len());
            self.values_synced = module.values().len();
        }
        while self.values_synced < module.values().len() {
            let index = self.values_synced;
            let value = &module.values()[index];
            let id = self.layout.alloc()?;
            self.layout.values.push(id);
            self.values_synced += 1;
            match &value.kind {
                ValueKind::Constant(payload) => {
                    let ty = self.layout.ty(value.ty)?;
                    emit_constant(ty, id, value.ty, payload, module, &mut self.consts)?;
                }
                ValueKind::Global(global) => {
                    let (storage, _) = module
                        .pointer_parts(value.ty)
                        .ok_or(AssembleError::Unsupported {
                            what: "a global whose type is not a pointer",
                        })?;
                    let pointer = self.layout.ty(value.ty)?;
                    let init = global
                        .init
                        .map(|init| self.layout.value(init))
                        .transpose()?;
                    self.globals.instruction(OP_VARIABLE, |e| {
                        e.word(pointer);
                        e.word(id);
                        e.word(storageclass_of(storage));
                        if let Some(init) = init {
                            e.word(init);
                        }
                        Ok(())
                    })?;
                    if matches!(storage, Storage::Input | Storage::Output) {
                        self.interface.push(id);
                    }
                    self.value_name(&value.name, id)?;
                    write_linkage(&mut self.link_globals, id, global.linkage, &value.name, index)?;
                }
                ValueKind::Function(function) => {
                    self.value_name(&value.name, id)?;
                    write_linkage(&mut self.link_functions, id, function.linkage, &value.name, index)?;
                }
                ValueKind::Argument | ValueKind::Instruction => {}
            }
        }
        Ok(())
    }

    /// Assigns and writes every extended-instruction set created
    /// since the last sync, keeping them in the import region
    /// between the capabilities and the memory model.
    ///
    /// # Errors
    ///
    /// [`AssembleError::IdExhausted`] when ids run out; encoder
    /// errors for oversized names.
    fn sync_ext_sets(&mut self, module: &Module) -> Result<(), AssembleError> {
        while self.layout.ext_sets.len() < module.ext_inst_sets.len() {
            let set = &module.ext_inst_sets[self.layout.ext_sets.len()];
            let id = self.layout.alloc()?;
            self.layout.ext_sets.push(id);
            self.imports.instruction(OP_EXTINSTIMPORT, |e| {
                e.word(id);
                e.string(&set.name)?;
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Copies `BuiltIn` annotations added since the last sync into
    /// the annotation region; linkage decorations come from the
    /// value payloads instead, mirroring `emit_decorations`.
    ///
    /// # Errors
    ///
    /// [`AssembleError::Unsupported`] for an unknown `BuiltIn` name
    /// or an out-of-range target id.
    fn sync_decorations(&mut self, module: &Module) -> Result<(), AssembleError> {
        while self.decor_synced < module.decorations.len() {
            let decor = &module.decorations[self.decor_synced];
            self.decor_synced += 1;
            let Decor::BuiltIn { name } = &decor.kind else {
                continue;
            };
            let target = self.layout.value(decor.target)?;
            let builtin = builtin_number(name)?;
            self.builtins.instruction(OP_DECORATE, |e| {
                e.word(target);
                e.word(ops::DECORATION_BUILTIN);
                e.word(builtin);
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Synchronizes every module-scope region; called before each
    /// body emission and once more from [`finish`](Self::finish).
    ///
    /// # Errors
    ///
    /// Propagates the errors of the individual sync steps.
    fn sync(&mut self, module: &Module) -> Result<(), AssembleError> {
        self.sync_types(module)?;
        self.sync_values(module)?;
        self.sync_ext_sets(module)?;
        self.sync_decorations(module)?;
        Ok(())
    }

    /// Emits `OpName` for a named module-scope value.
    ///
    /// # Errors
    ///
    /// Encoder errors for oversized strings.
    fn value_name(&mut self, name: &str, id: u32) -> Result<(), AssembleError> {
        if name.is_empty() {
            return Ok(());
        }
        self.names.instruction(OP_NAME, |e| {
            e.word(id);
            e.string(name)?;
            Ok(())
        })
    }
}

/// Emits one `OpDecorate` linkage-attributes instruction into `enc`.
///
/// [`Linkage::External`] and [`Linkage::Internal`] carry no
/// decoration (SPIR-V has no `Internal` enumerant and treats
/// undecorated entities as module-local); a value with no name falls
/// back to its arena index, mirroring `emit_decorations`.
///
/// # Errors
///
/// [`AssembleError::IdExhausted`] is not produced here; encoder
/// errors for oversized strings or instructions are propagated.
fn write_linkage(
    enc: &mut Encoder,
    target: u32,
    linkage: Linkage,
    name: &str,
    index: usize,
) -> Result<(), AssembleError> {
    let linkage_type = match linkage {
        Linkage::External | Linkage::Internal => return Ok(()),
        Linkage::Import => ops::LINKAGETYPE_IMPORT,
        Linkage::Export => ops::LINKAGETYPE_EXPORT,
    };
    let name = if name.is_empty() {
        format!("{index}")
    } else {
        String::from(name)
    };
    enc.instruction(OP_DECORATE, |e| {
        e.word(target);
        e.word(ops::DECORATION_LINKAGEATTRIBUTES);
        e.string(&name)?;
        e.word(linkage_type);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::{decode, kernel_module, section, string_operand};
    use super::*;
    use crate::spirv::assemble as whole_assemble;
    use codevar_ocl_ir::ir::{EntryPoint, Target};
    use codevar_ocl_ir::lower::ItemLowerer;
    use codevar_ocl_ir::spirv::ops::{
        OP_BRANCH, OP_BRANCHCONDITIONAL, OP_FUNCTION, OP_FUNCTIONCALL, OP_FUNCTIONEND, OP_LABEL, OP_RETURN,
        OP_RETURNVALUE, OP_UNREACHABLE,
    };
    use codevar_ocl_ir::verify::verify_function;
    use codevar_ocl_parse::{GenericParam, ItemKind, ItemStream};
    use codevar_ocl_sar::{DeclCollector, Diagnostic, analyze_body, file_checks};

    /// The word pairs compared between the streamed and whole-module
    /// assemblies: every instruction but `OpName` (the stream only
    /// names module-scope values, blocks, and types).
    fn comparable(words: &[u32]) -> Vec<(u16, Vec<u32>)> {
        decode(words)
            .into_iter()
            .filter(|instruction| instruction.opcode != OP_NAME)
            .map(|instruction| (instruction.opcode, instruction.operands))
            .collect()
    }

    /// Streams a module that has no function bodies: prelude, finish.
    fn stream_module(module: &Module) -> Vec<u32> {
        let stream = SpirvStream::prelude(module).expect("prelude");
        stream.finish(module).expect("finish")
    }

    #[test]
    fn empty_module_matches_the_whole_module_assembly() {
        let module = Module::new(Target::opencl());
        let whole = whole_assemble(&module).expect("whole module assembles");
        let words = stream_module(&module);
        assert_eq!(words, whole, "a body-less module assembles identically");
    }

    #[test]
    fn kernel_matches_the_whole_module_assembly_modulo_names() {
        let module = kernel_module();
        let whole = whole_assemble(&module).expect("whole module assembles");
        let mut stream = SpirvStream::prelude(&module).expect("prelude");
        for &function in module.functions() {
            let defined = module
                .function(function)
                .is_some_and(|def| def.body.is_some());
            if defined {
                stream
                    .emit_function_body(&module, function)
                    .expect("body emits");
            }
        }
        let words = stream.finish(&module).expect("stream finishes");
        assert_eq!(words[3], whole[3], "both assemblies share the bound");
        assert_eq!(
            comparable(&words),
            comparable(&whole),
            "instructions must match modulo OpName"
        );
        // The stream still names the module-scope functions.
        assert!(
            decode(&words).iter().any(|instruction| {
                instruction.opcode == OP_NAME && string_operand(&instruction.operands, 1).0 == "vector_add"
            }),
            "OpName must cover the kernel function"
        );
    }

    #[test]
    fn declared_array_matches_the_whole_module_assembly() {
        let mut module = Module::new(Target::opencl());
        let int = module.int_ty(32, true);
        module.intern_type(Type::Array { elem: int, len: 4 });
        let whole = whole_assemble(&module).expect("whole module assembles");
        let words = stream_module(&module);
        assert_eq!(words, whole, "the array scaffold keeps the whole-module ids");
    }

    #[test]
    fn array_without_int32_matches_the_whole_module_assembly() {
        let mut module = Module::new(Target::opencl());
        let half = module.int_ty(16, false);
        module.intern_type(Type::Array { elem: half, len: 2 });
        let whole = whole_assemble(&module).expect("whole module assembles");
        let words = stream_module(&module);
        assert_eq!(words, whole, "the synthesized uint32 keeps the whole-module ids");
    }

    #[test]
    fn emitting_a_function_twice_is_rejected() {
        let module = kernel_module();
        let defined = module
            .functions()
            .iter()
            .copied()
            .find(|&id| {
                module
                    .function(id)
                    .is_some_and(|def| def.body.is_some())
            })
            .expect("the fixture defines a function");
        let mut stream = SpirvStream::prelude(&module).expect("prelude");
        stream
            .emit_function_body(&module, defined)
            .expect("first emit");
        let error = stream
            .emit_function_body(&module, defined)
            .expect_err("second emit");
        assert!(
            matches!(
                error,
                AssembleError::Unsupported {
                    what: "a function emitted twice"
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn emitting_a_declaration_is_rejected() {
        let module = kernel_module();
        let declared = module
            .functions()
            .iter()
            .copied()
            .find(|&id| {
                module
                    .function(id)
                    .is_some_and(|def| def.body.is_none())
            })
            .expect("the fixture declares a function");
        let mut stream = SpirvStream::prelude(&module).expect("prelude");
        let error = stream
            .emit_function_body(&module, declared)
            .expect_err("a body-less function");
        assert!(
            matches!(
                error,
                AssembleError::Unsupported {
                    what: "a function without a body"
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn finish_without_emitting_bodies_is_rejected() {
        let module = kernel_module();
        let stream = SpirvStream::prelude(&module).expect("prelude");
        let error = stream
            .finish(&module)
            .expect_err("unemitted body");
        assert!(
            matches!(
                error,
                AssembleError::Unsupported {
                    what: "a function body that was never emitted"
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn non_kernel_entry_point_is_rejected_at_the_prelude() {
        let mut module = Module::new(Target::opencl());
        module.entry_points.push(EntryPoint {
            model: ExecutionModel::GLCompute,
            func: ValueId(0),
            name: alloc::string::String::from("main"),
        });
        let error = match SpirvStream::prelude(&module) {
            Err(error) => error,
            Ok(_) => panic!("GLCompute must be rejected"),
        };
        assert!(
            matches!(
                error,
                AssembleError::Unsupported {
                    what: "a non-Kernel execution model"
                }
            ),
            "unexpected error: {error}"
        );
    }

    /// A two-function kernel: a helper called from the kernel body,
    /// with a work-item built-in and a conditional, so the streaming
    /// driver sees declarations, bodies, truncation, globals, and
    /// annotations in one file.
    const SOURCE: &str = "\
fn double(x: int) -> int {
    x + x
}

#[kernel]
fn vector_add(a: *mut float, n: int) {
    let i = get_global_id(0) as int;
    if i < n {
        a[i] = double(i) as float;
    }
}
";

    /// Runs the full streaming protocol over `SOURCE`: declaration
    /// analysis, signature declarations, prelude, then one
    /// analyze → lower → verify → emit → reclaim cycle per body.
    fn stream_source() -> Vec<u32> {
        let mut collector = DeclCollector::new();
        let mut items = ItemStream::new(SOURCE);
        while let Some(outcome) = items.next_item() {
            assert!(outcome.errors.is_empty(), "parse errors: {:?}", outcome.errors);
            collector.feed(SOURCE, &outcome.item);
        }
        let (mut env, diagnostics) = collector.finish();
        assert!(
            !diagnostics.iter().any(Diagnostic::is_error),
            "declaration diagnostics: {diagnostics:?}"
        );
        let declarations = env.declarations().to_vec();
        let mut lowerer = ItemLowerer::new(&declarations);
        // Declare aliases first, then functions, mirroring `Lowerer::run`.
        for aliases in [true, false] {
            let mut items = ItemStream::new(SOURCE);
            while let Some(outcome) = items.next_item() {
                match (&outcome.item.kind, aliases) {
                    (ItemKind::TypeAlias(alias), true) => {
                        lowerer
                            .declare_alias(alias)
                            .expect("alias declares");
                    }
                    (ItemKind::Fn(function), false) => {
                        let kernel = env.is_kernel(&function.name);
                        lowerer
                            .declare_function(function, kernel)
                            .expect("function declares");
                    }
                    _ => {}
                }
            }
        }
        let mut stream = SpirvStream::prelude(lowerer.module()).expect("prelude");
        let mut items = ItemStream::new(SOURCE);
        while let Some(outcome) = items.next_item() {
            let ItemKind::Fn(function) = &outcome.item.kind else {
                continue;
            };
            if function
                .generics
                .iter()
                .any(|param| matches!(param, GenericParam::Type { .. }))
            {
                continue;
            }
            let mark = lowerer.module().watermark();
            let (diagnostics, tables) = analyze_body(&mut env, SOURCE, &outcome.item);
            assert!(
                !diagnostics.iter().any(Diagnostic::is_error),
                "body diagnostics: {diagnostics:?}"
            );
            lowerer
                .lower_function_body(function, &tables.types, &tables.resolutions)
                .expect("body lowers");
            let kernel = lowerer
                .function(&function.name)
                .expect("declared");
            verify_function(lowerer.module(), kernel).expect("body verifies");
            stream
                .emit_function_body(lowerer.module(), kernel)
                .expect("body emits");
            drop(lowerer.module_mut().take_function_body(kernel));
            lowerer.module_mut().truncate_values(mark);
        }
        let checks = file_checks(&env);
        assert!(
            !checks.iter().any(Diagnostic::is_error),
            "file diagnostics: {checks:?}"
        );
        stream.finish(lowerer.module()).expect("finish")
    }

    #[test]
    fn work_item_kernel_streams_end_to_end() {
        let words = stream_source();
        assert_eq!(words[0], crate::spirv::MAGIC, "SPIR-V magic");
        assert!(words[3] >= 1, "bound must exceed every id");
        let instructions = decode(&words);

        // The logical-layout sections never move backwards.
        let mut rank = 0;
        let mut functions_started = false;
        for instruction in &instructions {
            let next = if functions_started {
                7
            } else {
                section(instruction.opcode)
            };
            if instruction.opcode == OP_FUNCTION {
                functions_started = true;
            }
            assert!(
                next >= rank,
                "opcode {} appeared out of section order",
                instruction.opcode
            );
            rank = next;
        }
        assert!(functions_started, "the kernel must be defined");

        // Both functions carry bodies: the helper and the kernel.
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.opcode == OP_FUNCTION)
                .count(),
            2,
            "helper and kernel are both defined"
        );
        assert!(
            instructions
                .iter()
                .any(|instruction| instruction.opcode == OP_FUNCTIONCALL),
            "the kernel calls the helper"
        );

        // The entry point names the kernel.
        let entry = instructions
            .iter()
            .find(|instruction| instruction.opcode == OP_ENTRYPOINT)
            .expect("entry point present");
        assert_eq!(entry.operands[0], ops::EXECUTIONMODEL_KERNEL);
        let (name, name_at) = string_operand(&entry.operands, 2);
        assert_eq!(name, "vector_add");

        // The work-item variable is an Input variable decorated
        // `BuiltIn GlobalInvocationId` (enumerant 28) and listed in
        // the entry-point interface.
        let builtin = instructions
            .iter()
            .find(|instruction| {
                instruction.opcode == OP_DECORATE
                    && instruction.operands.get(1) == Some(&ops::DECORATION_BUILTIN)
            })
            .expect("BuiltIn decoration present");
        assert_eq!(builtin.operands[2], 28, "GlobalInvocationId enumerant");
        let variable = builtin.operands[0];
        assert!(
            instructions.iter().any(|instruction| {
                instruction.opcode == OP_VARIABLE && instruction.operands.get(1) == Some(&variable)
            }),
            "the decorated id must be an OpVariable"
        );
        assert!(
            entry.operands[name_at..].contains(&variable),
            "built-in variable must appear in the entry point interface"
        );

        // No import linkage survives lowering.
        assert!(
            !instructions.iter().any(|instruction| {
                instruction.opcode == OP_DECORATE
                    && instruction.operands.get(1) == Some(&ops::DECORATION_LINKAGEATTRIBUTES)
            }),
            "no import linkage remains"
        );

        // Every block ends in a terminator.
        for (index, instruction) in instructions.iter().enumerate() {
            if instruction.opcode != OP_LABEL {
                continue;
            }
            let end = instructions[index + 1..]
                .iter()
                .position(|next| matches!(next.opcode, OP_LABEL | OP_FUNCTIONEND))
                .map_or(instructions.len(), |offset| index + 1 + offset);
            let last = instructions[end - 1].opcode;
            assert!(
                matches!(
                    last,
                    OP_BRANCH | OP_BRANCHCONDITIONAL | OP_RETURN | OP_RETURNVALUE | OP_UNREACHABLE
                ),
                "block must end in a terminator, found opcode {last}"
            );
        }
    }
}
