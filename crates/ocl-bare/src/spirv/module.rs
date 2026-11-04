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

//! Validation of a SPIR-V program binary into interpreter tables.
//!
//! [`load`] accepts the bytes of an OpenCL SPIR-V module and produces a
//! boxed [`SpirvModule`]: the instruction stream, the per-id type, value
//! and constant tables, the label and function indexes, precomputed
//! layouts, and the entry points with their argument kinds. Parsing runs
//! in two passes — a linear scan that records every instruction, then a
//! finalization pass that resolves cross-references (calls, linkage
//! names, built-in variables, arena placement, recursion) — so a
//! malformed module fails here, before any kernel can execute.
//!
//! Rejection is always [`Error::InvalidBinary`]; allocation failures are
//! [`Error::OutOfMemory`]. Nothing in this module panics.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_void;

use super::builtin;
use super::decode::{self, Instr};
use super::ops;
use super::types::{self, Layout, LayoutDecorations, Type};
use crate::error::{Error, Result};
use crate::exec::{self, SpirvBinding, WorkItem};
use crate::registry::{ArgKind, KernelDef};

/// Highest `<id>` bound a module may declare. Larger tables are rejected
/// outright, which bounds every per-id vector the loader allocates.
const MAX_IDS: u32 = 8192;

/// Largest module, in words, including the five-word header.
const MAX_WORDS: usize = 131_072;

/// Largest operand count, in words, of a single instruction. Handlers
/// copy operands (calls, phis, switches) or read strings out of them, so
/// this bounds those allocations.
const MAX_ITEM_WORDS: u32 = 512;

/// Byte budget of one work-item's private arena (function/private
/// variables and the built-in input values).
const MAX_ITEM_ARENA: u32 = 4096;

/// Byte budget of the work-group shared arena (`local` variables).
const MAX_GROUP_ARENA: u32 = 65_536;

/// The interpreter's type-shape tables: the byte size of every declared
/// type, the stride of every array, and the sorted `(type, member)`
/// offsets an access chain binary-searches.
type TypeIndexes = (Vec<u32>, Vec<u32>, Vec<(u64, u32)>);

/// Lane budget of the module-wide constant pool.
const MAX_CONST_WORDS: usize = 16_384;

/// The magic number of a big-endian-written module: a host that read the
/// header little-endian sees the bytes of [`decode::MAGIC`] reversed.
const SWAPPED_MAGIC: u32 = decode::MAGIC.swap_bytes();

/// One defined value: its type, its first lane in the per-item register
/// file, and its scalar-leaf count.
///
/// `ty == 0` marks the slot as never defined (id 0 is never valid).
/// A void function-call result keeps a real type with `lanes == 0`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ValueSlot {
    /// The value's type id.
    pub ty: u32,
    /// First lane of the value in the register file.
    pub off: u32,
    /// Number of scalar-leaf lanes (`0` only for void results).
    pub lanes: u32,
}

/// A constant: where its lanes live in the pool, and how many there are.
///
/// `count == 0` marks the id as not a constant.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConstSlot {
    /// First lane of the constant in the pool.
    pub off: u32,
    /// Number of pooled lanes (`0` when the id is not a constant).
    pub count: u32,
}

/// One function of the module: signature, body shape and call graph.
#[derive(Debug)]
pub(crate) struct Func {
    /// Return type id (`OpTypeFunction`'s return).
    pub ret_ty: u32,
    /// The `OpTypeFunction` id.
    pub fn_ty: u32,
    /// Parameter value ids, in declaration order.
    pub params: Vec<u32>,
    /// Callee function ids of every `OpFunctionCall` inside the body.
    pub calls: Vec<u32>,
    /// Whether the function has a body (vs. a linkage declaration).
    pub has_body: bool,
    /// Instruction index of the first label (`0` when there is none).
    pub entry: u32,
    /// Linkage import name, set for imported declarations.
    pub link_name: Option<String>,
}

/// A module-scope `Input` variable materialized per work-item.
#[derive(Debug)]
pub(crate) struct InputVar {
    /// The pointer value id of the variable.
    pub ptr: u32,
    /// Pointed-to type id.
    pub pointee: u32,
    /// The `BuiltIn` decoration number.
    pub number: u32,
    /// Byte offset of the value inside the item arena.
    pub off: u32,
    /// Scalar-leaf count of the pointee (`1` or `3`).
    pub lanes: u32,
}

/// A variable that lives in an arena: function/private (item) or
/// workgroup (group) storage.
#[derive(Debug)]
pub(crate) struct ArenaVar {
    /// The pointer value id of the variable.
    pub ptr: u32,
    /// Pointed-to type id.
    pub pointee: u32,
    /// Storage class of the pointer.
    pub storage: u32,
    /// Constant initializer, present only for body function/private
    /// variables.
    pub init: Option<u32>,
    /// Byte offset inside its arena.
    pub off: u32,
}

/// One `OpEntryPoint` of the module, resolved to its function.
#[derive(Debug)]
pub(crate) struct SpirvEntry {
    /// The OpenCL kernel name (the entry point name).
    pub name: String,
    /// The entry function's id.
    pub func: u32,
    /// Owning module, linked by [`SpirvModule::build_defs`] after the
    /// module is boxed (null until then).
    pub module: *const SpirvModule,
}

/// The fully validated module the interpreter runs from.
///
/// Fields are `pub(crate)` so the sibling interpreter can read them
/// directly; every table is sized by the module's `<id>` bound or by its
/// instruction count.
#[derive(Debug)]
pub(crate) struct SpirvModule {
    /// The module words, little-endian decoded.
    pub words: Vec<u32>,
    /// Instruction stream (offsets into `words`).
    pub instrs: Vec<Instr>,
    /// The `<id>` bound. The loader sizes every per-id table by it; the
    /// module keeps it so the loaded shape can be inspected.
    #[allow(dead_code)]
    pub bound: u32,
    /// Type table by id (`Unsupported` = never declared).
    pub types: Vec<Type>,
    /// Value table by id.
    pub values: Vec<ValueSlot>,
    /// Constant marks by id.
    pub consts: Vec<ConstSlot>,
    /// Constant-pool lanes.
    pub const_pool: Vec<u64>,
    /// Label id to instruction index plus one (`0` = not a label).
    pub label_ip: Vec<u32>,
    /// Function id to function index plus one (`0` = not a function).
    pub func_index: Vec<u32>,
    /// Function records, indexed by `func_index`.
    pub funcs: Vec<Func>,
    /// Entry points, in module order.
    pub entries: Vec<SpirvEntry>,
    /// Module-scope `Input` variables.
    pub input_vars: Vec<InputVar>,
    /// Arena-backed variables, in declaration order.
    pub arena_vars: Vec<ArenaVar>,
    /// `Offset`/`ArrayStride` decorations for the layout walkers.
    pub decor: LayoutDecorations,
    /// Byte size of each declared type (`0` = layout failed or absent).
    pub ty_size: Vec<u32>,
    /// Element stride of each array type (`0` = not an array).
    pub ty_stride: Vec<u32>,
    /// `(type id << 32 | member)` to byte offset, sorted for searching.
    pub struct_off: Vec<(u64, u32)>,
    /// The `OpenCL.std` extended-instruction set id (`0` = absent).
    pub opencl_set: u32,
    /// Whether the module uses barriers or group operations and needs
    /// work-group rendezvous scheduling.
    pub needs_collective: bool,
    /// Total lanes of the per-item register file.
    pub slots_len: u32,
    /// Bytes the item arena occupies.
    pub item_arena: u32,
    /// Bytes the group arena occupies.
    pub group_arena: u32,
}

impl SpirvModule {
    /// Builds the [`KernelDef`]s of the module's entry points.
    ///
    /// Must run on the final, boxed module: each entry records the
    /// module's own address so a kernel binding can reach the tables at
    /// execution time.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBinary`] when an entry parameter is not a
    /// supported kernel argument (a pointer to an unsupported address
    /// space, or a type without a byte size).
    pub(crate) fn build_defs(&mut self) -> Result<Vec<KernelDef>> {
        let module = self as *const Self;
        let Self {
            entries,
            funcs,
            values,
            types,
            func_index,
            ty_size,
            ..
        } = self;
        let mut defs: Vec<KernelDef> = Vec::new();
        defs.try_reserve(entries.len())
            .map_err(|_| Error::OutOfMemory)?;
        for entry in entries.iter_mut() {
            let name = entry.name.clone();
            let raw = func_index
                .get(entry.func as usize)
                .copied()
                .unwrap_or(0);
            if raw == 0 {
                return Err(Error::InvalidBinary);
            }
            let func = funcs
                .get((raw - 1) as usize)
                .ok_or(Error::InvalidBinary)?;
            let mut kinds: Vec<ArgKind> = Vec::new();
            kinds
                .try_reserve(func.params.len())
                .map_err(|_| Error::OutOfMemory)?;
            for param in &func.params {
                let ty = values
                    .get(*param as usize)
                    .map(|slot| slot.ty)
                    .filter(|&ty| ty != 0)
                    .ok_or(Error::InvalidBinary)?;
                let kind = match types
                    .get(ty as usize)
                    .ok_or(Error::InvalidBinary)?
                {
                    Type::Pointer { storage, .. }
                        if *storage == ops::STORAGE_CROSS_WORKGROUP
                            || *storage == ops::STORAGE_UNIFORM_CONSTANT =>
                    {
                        ArgKind::Buffer
                    }
                    Type::Pointer { .. } => return Err(Error::InvalidBinary),
                    Type::Void | Type::Function { .. } | Type::Unsupported => {
                        return Err(Error::InvalidBinary);
                    }
                    _ => {
                        let size = ty_size.get(ty as usize).copied().unwrap_or(0);
                        if size == 0 {
                            return Err(Error::InvalidBinary);
                        }
                        ArgKind::Scalar(size as usize)
                    }
                };
                kinds.push(kind);
            }
            entry.module = module;
            let binding = SpirvBinding::new(entry as *const SpirvEntry as *const c_void);
            defs.push(KernelDef {
                name: Cow::Owned(name),
                func: spirv_stub,
                kinds: Cow::Owned(kinds),
                binding,
            });
        }
        Ok(defs)
    }
}

/// The kernel entry of a SPIR-V-backed [`KernelDef`].
///
/// The claim loop never calls it: a launch whose binding is set runs the
/// interpreter on the module instead.
unsafe extern "C" fn spirv_stub(_item: *mut WorkItem) {}

/// Parses the bytes of a SPIR-V module into interpreter tables.
///
/// Accepts little-endian modules (the format SPIR-V defines) and
/// byte-swapped ones written big-endian. The magic number, header,
/// instruction framing, the embedded OpenCL subset and every
/// cross-reference are validated before this returns.
///
/// # Errors
///
/// * [`Error::InvalidBinary`] — the module is malformed or uses a
///   construct outside this driver's subset.
/// * [`Error::OutOfMemory`] — a table reservation failed.
pub(crate) fn load(bytes: &[u8]) -> Result<Box<SpirvModule>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::InvalidBinary);
    }
    let count = bytes.len() / 4;
    if count < decode::HEADER_WORDS {
        return Err(Error::InvalidBinary);
    }
    if count > MAX_WORDS {
        return Err(Error::InvalidBinary);
    }
    let mut words: Vec<u32> = Vec::new();
    words
        .try_reserve_exact(count)
        .map_err(|_| Error::OutOfMemory)?;
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if words.first() == Some(&SWAPPED_MAGIC) {
        for word in &mut words {
            *word = word.swap_bytes();
        }
    }
    let (header, instrs) = decode::decode(&words)?;
    if header.bound > MAX_IDS {
        return Err(Error::InvalidBinary);
    }
    let module = Loader::new(words, header.bound, instrs)?.scan()?;
    Ok(Box::new(module))
}

/// Allocates a `len`-element table filled with `fill` without ever
/// aborting on allocation failure: the reservation is checked first, and
/// the fill never allocates again.
fn table<T: Clone>(len: usize, fill: T) -> Result<Vec<T>> {
    let mut slots: Vec<T> = Vec::new();
    slots
        .try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory)?;
    slots.resize(len, fill);
    Ok(slots)
}

/// Bumps `cursor` by `size` (aligned to `align`), failing when the
/// placement would exceed `cap`.
fn place(cursor: &mut u32, size: u32, align: u32, cap: u32) -> Result<u32> {
    let off = Layout::align_up(*cursor, align);
    let end = off
        .checked_add(size)
        .ok_or(Error::InvalidBinary)?;
    if end > cap {
        return Err(Error::InvalidBinary);
    }
    *cursor = end;
    Ok(off)
}

/// The scalar category an instruction's result and operands must agree
/// on: float arithmetic produces floats, comparisons booleans, and so
/// on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A boolean (or vector of booleans).
    Bool,
    /// An integer (or vector of integers).
    Int,
    /// A floating-point number (or vector of floats).
    Float,
    /// A pointer.
    Ptr,
    /// Anything else: void, composites, function types.
    Other,
}

/// The parser's state for the function currently being scanned.
#[derive(Debug)]
struct CurFunc {
    /// Index into `funcs`.
    index: u32,
    /// Until the first label: only parameters may appear.
    phase_header: bool,
    /// After a block terminator: only a label may follow.
    expect_label: bool,
    /// Label ids of this function.
    labels: Vec<u32>,
    /// Branch/merge/phi-parent targets validated at `OpFunctionEnd`.
    pending: Vec<u32>,
    /// `(values, parents)` of every `OpPhi`, validated at
    /// `OpFunctionEnd` because dominance may forward-reference values.
    phi_sites: Vec<(Vec<u32>, Vec<u32>)>,
}

/// The two-pass parser. Pass A ( [`Loader::scan`]) walks the instruction
/// stream linearly; pass B ([`Loader::finish`]) resolves what only the
/// whole module can answer.
#[derive(Debug)]
struct Loader {
    /// Module words (taken out during the scan so operand slices never
    /// alias `&mut self`).
    words: Vec<u32>,
    /// The instruction stream.
    instrs: Vec<Instr>,
    /// Instruction count, remembered while `instrs` is taken out.
    total_instrs: usize,
    /// The `<id>` bound.
    bound: u32,
    /// Ids already given a meaning (duplicate detection).
    seen: Vec<bool>,
    /// Type table by id.
    types: Vec<Type>,
    /// Value table by id.
    values: Vec<ValueSlot>,
    /// Constant marks by id.
    consts: Vec<ConstSlot>,
    /// Constant-pool lanes.
    const_pool: Vec<u64>,
    /// Label id to instruction index plus one.
    label_ip: Vec<u32>,
    /// Function id to function index plus one.
    func_index: Vec<u32>,
    /// Layout decorations.
    decor: LayoutDecorations,
    /// `(target, BuiltIn number)` decoration records.
    builtin_decos: Vec<(u32, u32)>,
    /// `(target, kind, name)` linkage records.
    linkages: Vec<(u32, u32, String)>,
    /// `(name, function id)` raw entry points.
    entry_raw: Vec<(String, u32)>,
    /// `(caller, callee, arguments)` call records.
    call_sites: Vec<(u32, u32, Vec<u32>)>,
    /// Function records.
    funcs: Vec<Func>,
    /// Module-scope `Input` variables.
    input_vars: Vec<InputVar>,
    /// Arena variables, in declaration order.
    arena_vars: Vec<ArenaVar>,
    /// The `OpenCL.std` set id (`0` = none seen).
    opencl_set: u32,
    /// Whether `OpCapability Kernel` was declared.
    caps_kernel: bool,
    /// Whether `OpMemoryModel` was declared.
    has_memory_model: bool,
    /// Whether the module uses barriers or group operations.
    needs_collective: bool,
    /// Running lane count of the register file.
    slots_len: u32,
    /// The function being scanned, if any.
    current: Option<CurFunc>,
}

impl Loader {
    /// Prepares every table up front so the scan itself cannot abort.
    fn new(words: Vec<u32>, bound: u32, instrs: Vec<Instr>) -> Result<Self> {
        let len = bound as usize;
        let cap = instrs.len();
        let mut linkages: Vec<(u32, u32, String)> = Vec::new();
        linkages
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        let mut entry_raw: Vec<(String, u32)> = Vec::new();
        entry_raw
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        let mut builtin_decos: Vec<(u32, u32)> = Vec::new();
        builtin_decos
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        let mut call_sites: Vec<(u32, u32, Vec<u32>)> = Vec::new();
        call_sites
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        let mut funcs: Vec<Func> = Vec::new();
        funcs
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        let mut input_vars: Vec<InputVar> = Vec::new();
        input_vars
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        let mut arena_vars: Vec<ArenaVar> = Vec::new();
        arena_vars
            .try_reserve_exact(cap)
            .map_err(|_| Error::OutOfMemory)?;
        Ok(Self {
            words,
            total_instrs: instrs.len(),
            instrs,
            bound,
            seen: table(len, false)?,
            types: table(len, Type::Unsupported)?,
            values: table(len, ValueSlot::default())?,
            consts: table(len, ConstSlot::default())?,
            const_pool: Vec::new(),
            label_ip: table(len, 0u32)?,
            func_index: table(len, 0u32)?,
            decor: LayoutDecorations::default(),
            builtin_decos,
            linkages,
            entry_raw,
            call_sites,
            funcs,
            input_vars,
            arena_vars,
            opencl_set: 0,
            caps_kernel: false,
            has_memory_model: false,
            needs_collective: false,
            slots_len: 0,
            current: None,
        })
    }

    /// Pass A: walks every instruction, then returns the finished module.
    fn scan(mut self) -> Result<SpirvModule> {
        let words = core::mem::take(&mut self.words);
        let mut outcome = Ok(());
        for index in 0..self.instrs.len() {
            let instr = self.instrs[index];
            if instr.len as u32 > MAX_ITEM_WORDS {
                outcome = Err(Error::InvalidBinary);
                break;
            }
            let step = if self.current.is_some() {
                self.body(&words, instr, index)
            } else {
                self.module_instr(&words, instr)
            };
            if let Err(error) = step {
                outcome = Err(error);
                break;
            }
        }
        self.words = words;
        outcome?;
        if self.current.is_some() {
            return Err(Error::InvalidBinary);
        }
        self.finish()
    }

    /// Bounds-checks a raw operand word as an `<id>`.
    fn ref_word(&self, raw: u32) -> Result<u32> {
        if raw == 0 || raw >= self.bound {
            return Err(Error::InvalidBinary);
        }
        Ok(raw)
    }

    /// Bounds-checks operand `index` as an `<id>`.
    fn ref_at(&self, ops: &[u32], index: usize) -> Result<u32> {
        let raw = ops
            .get(index)
            .copied()
            .ok_or(Error::InvalidBinary)?;
        self.ref_word(raw)
    }

    /// Claims an id for a definition, rejecting duplicates.
    fn check_id(&mut self, id: u32) -> Result<()> {
        let slot = self
            .seen
            .get_mut(id as usize)
            .ok_or(Error::InvalidBinary)?;
        if *slot {
            return Err(Error::InvalidBinary);
        }
        *slot = true;
        Ok(())
    }

    /// The declared type of a value id (`ty != 0` = defined).
    fn val_ty(&self, id: u32) -> Result<u32> {
        let slot = self
            .values
            .get(id as usize)
            .ok_or(Error::InvalidBinary)?;
        if slot.ty == 0 {
            return Err(Error::InvalidBinary);
        }
        Ok(slot.ty)
    }

    /// Whether `id` names a declared type (never `Unsupported`).
    fn is_type(&self, id: u32) -> bool {
        matches!(
            self.types.get(id as usize),
            Some(ty) if !matches!(ty, Type::Unsupported)
        )
    }

    /// The declared type table entry of `ty`.
    fn declared(&self, ty: u32) -> Result<&Type> {
        let entry = self
            .types
            .get(ty as usize)
            .ok_or(Error::InvalidBinary)?;
        if matches!(entry, Type::Unsupported) {
            return Err(Error::InvalidBinary);
        }
        Ok(entry)
    }

    /// The lane value of a constant id, or an error when `id` is not a
    /// constant.
    fn const_lane(&self, id: u32) -> Result<u64> {
        let slot = self
            .consts
            .get(id as usize)
            .ok_or(Error::InvalidBinary)?;
        if slot.count == 0 {
            return Err(Error::InvalidBinary);
        }
        self.const_pool
            .get(slot.off as usize)
            .copied()
            .ok_or(Error::InvalidBinary)
    }

    /// The scalar kind of `ty`, unwrapping one vector level so vector
    /// arithmetic type-checks against its element kind.
    fn kind_of(&self, ty: u32) -> Result<Kind> {
        let entry = self.declared(ty)?;
        Ok(match entry {
            Type::Bool => Kind::Bool,
            Type::Int { .. } => Kind::Int,
            Type::Float { .. } => Kind::Float,
            Type::Pointer { .. } => Kind::Ptr,
            Type::Vector { elem, .. } => match self.declared(*elem)? {
                Type::Bool => Kind::Bool,
                Type::Int { .. } => Kind::Int,
                Type::Float { .. } => Kind::Float,
                _ => Kind::Other,
            },
            Type::Void
            | Type::Array { .. }
            | Type::Struct { .. }
            | Type::Function { .. }
            | Type::Unsupported => Kind::Other,
        })
    }

    /// How many constituents a composite of `ty` holds.
    fn constituent_count(&self, ty: u32) -> Result<u32> {
        Ok(match self.declared(ty)? {
            Type::Vector { len, .. } => u32::from(*len),
            Type::Array { len, .. } => *len,
            Type::Struct { fields } => fields.len() as u32,
            _ => return Err(Error::InvalidBinary),
        })
    }

    /// The type of constituent `index` of composite `ty`.
    fn constituent_ty(&self, ty: u32, index: u32) -> Result<u32> {
        Ok(match self.declared(ty)? {
            Type::Vector { elem, .. } | Type::Array { elem, .. } => *elem,
            Type::Struct { fields } => fields
                .get(index as usize)
                .copied()
                .ok_or(Error::InvalidBinary)?,
            _ => return Err(Error::InvalidBinary),
        })
    }

    /// Defines `id` as a value of `ty`, validating the type and charging
    /// its lanes against the register file.
    fn def_value(&mut self, ty: u32, id: u32) -> Result<()> {
        self.check_id(id)?;
        self.declared(ty)?;
        if matches!(self.types[ty as usize], Type::Void) {
            return Err(Error::InvalidBinary);
        }
        let lanes = types::lanes(&self.types, ty, &self.decor)?;
        types::layout(&self.types, ty, &self.decor)?;
        let off = self.slots_len;
        self.slots_len = off.checked_add(lanes).ok_or(Error::OutOfMemory)?;
        self.values[id as usize] = ValueSlot { ty, off, lanes };
        Ok(())
    }

    /// Defines `id` as a void call result (a typed slot of no lanes).
    fn def_void(&mut self, ty: u32, id: u32) -> Result<()> {
        self.check_id(id)?;
        if !matches!(self.types.get(ty as usize), Some(Type::Void)) {
            return Err(Error::InvalidBinary);
        }
        self.values[id as usize] = ValueSlot { ty, off: 0, lanes: 0 };
        Ok(())
    }

    /// Reserves `count` constant lanes for `id` and marks it as a
    /// constant; the caller writes the lanes afterward.
    fn def_const(&mut self, ty: u32, id: u32, count: u32) -> Result<u32> {
        self.check_id(id)?;
        self.declared(ty)?;
        let count = count as usize;
        if count == 0 {
            return Err(Error::InvalidBinary);
        }
        if self.const_pool.len().saturating_add(count) > MAX_CONST_WORDS {
            return Err(Error::InvalidBinary);
        }
        let off = self.const_pool.len() as u32;
        self.const_pool
            .try_reserve(count)
            .map_err(|_| Error::OutOfMemory)?;
        let end = (off as usize)
            .checked_add(count)
            .ok_or(Error::InvalidBinary)?;
        self.const_pool.resize(end, 0);
        let lanes = count as u32;
        self.values[id as usize] = ValueSlot { ty, off: 0, lanes };
        self.consts[id as usize] = ConstSlot { off, count: lanes };
        Ok(off)
    }

    /// Pass A handler for instructions outside any function body.
    fn module_instr(&mut self, words: &[u32], instr: Instr) -> Result<()> {
        let ops = instr.words(words);
        match instr.opcode {
            ops::OP_CAPABILITY => {
                if ops.len() != 1 {
                    return Err(Error::InvalidBinary);
                }
                if ops[0] == ops::CAPABILITY_KERNEL {
                    self.caps_kernel = true;
                }
                Ok(())
            }
            ops::OP_EXT_INST_IMPORT => self.ext_inst_import(ops),
            ops::OP_MEMORY_MODEL => self.memory_model(ops),
            ops::OP_ENTRY_POINT => self.entry_point(ops),
            ops::OP_DECORATE => self.decorate(ops),
            ops::OP_MEMBER_DECORATE => self.member_decorate(ops),
            ops::OP_TYPE_VOID => self.type_scalar(ops, Type::Void),
            ops::OP_TYPE_BOOL => self.type_scalar(ops, Type::Bool),
            ops::OP_TYPE_INT => self.type_int(ops),
            ops::OP_TYPE_FLOAT => self.type_float(ops),
            ops::OP_TYPE_VECTOR => self.type_vector(ops),
            ops::OP_TYPE_ARRAY => self.type_array(ops),
            ops::OP_TYPE_STRUCT => self.type_struct(ops),
            ops::OP_TYPE_POINTER => self.type_pointer(ops),
            ops::OP_TYPE_FUNCTION => self.type_function(ops),
            ops::OP_TYPE_MATRIX
            | ops::OP_TYPE_IMAGE
            | ops::OP_TYPE_SAMPLER
            | ops::OP_TYPE_SAMPLED_IMAGE
            | ops::OP_TYPE_RUNTIME_ARRAY
            | ops::OP_TYPE_OPAQUE
            | ops::OP_TYPE_EVENT
            | ops::OP_TYPE_DEVICE_EVENT
            | ops::OP_TYPE_RESERVE_ID
            | ops::OP_TYPE_QUEUE
            | ops::OP_TYPE_PIPE
            | ops::OP_TYPE_FORWARD_POINTER => Err(Error::InvalidBinary),
            ops::OP_CONSTANT_TRUE | ops::OP_CONSTANT_FALSE => self.constant_bool(instr.opcode, ops),
            ops::OP_CONSTANT => self.constant_scalar(ops),
            ops::OP_CONSTANT_COMPOSITE => self.constant_composite(ops),
            ops::OP_CONSTANT_NULL => self.constant_null(ops),
            ops::OP_UNDEF => self.constant_undef(ops),
            ops::OP_VARIABLE => self.module_variable(ops),
            ops::OP_FUNCTION => self.start_function(ops),
            ops::OP_EXECUTION_MODE
            | ops::OP_SOURCE
            | ops::OP_SOURCE_EXTENSION
            | ops::OP_SOURCE_CONTINUED
            | ops::OP_EXTENSION
            | ops::OP_NAME
            | ops::OP_MEMBER_NAME
            | ops::OP_STRING
            | ops::OP_LINE
            | ops::OP_NO_LINE
            | ops::OP_NOP => Ok(()),
            _ => Err(Error::InvalidBinary),
        }
    }

    /// `OpExtInstImport`: only a single `OpenCL.std` set is accepted.
    fn ext_inst_import(&mut self, ops: &[u32]) -> Result<()> {
        if self.opencl_set != 0 || ops.len() < 2 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        let (name, used) = decode::read_string(&ops[1..], 0)?;
        if used != ops.len() - 1 || name != "OpenCL.std" {
            return Err(Error::InvalidBinary);
        }
        self.check_id(id)?;
        self.opencl_set = id;
        Ok(())
    }

    /// `OpMemoryModel`: one OpenCL model whose addressing width matches
    /// the host pointer width.
    fn memory_model(&mut self, ops: &[u32]) -> Result<()> {
        if self.has_memory_model || ops.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let width = match ops[0] {
            ops::ADDRESSING_MODEL_LOGICAL => core::mem::size_of::<usize>(),
            ops::ADDRESSING_MODEL_PHYSICAL32 => 4,
            ops::ADDRESSING_MODEL_PHYSICAL64 => 8,
            _ => return Err(Error::InvalidBinary),
        };
        if ops[1] != ops::MEMORY_MODEL_OPENCL || width != core::mem::size_of::<usize>() {
            return Err(Error::InvalidBinary);
        }
        self.has_memory_model = true;
        Ok(())
    }

    /// `OpEntryPoint`: kernel execution model only; the function is
    /// resolved in pass B (entry points precede function definitions).
    fn entry_point(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 3 {
            return Err(Error::InvalidBinary);
        }
        if ops[0] != ops::EXECUTION_MODEL_KERNEL {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_word(ops[1])?;
        let (name, used) = decode::read_string(&ops[2..], 0)?;
        if name.is_empty() {
            return Err(Error::InvalidBinary);
        }
        for &interface in &ops[2 + used..] {
            self.ref_word(interface)?;
        }
        self.entry_raw.push((name, id));
        Ok(())
    }

    /// `OpDecorate`: records built-ins, linkage attributes and array
    /// strides; unknown decorations are ignored.
    fn decorate(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 2 {
            return Err(Error::InvalidBinary);
        }
        let target = self.ref_at(ops, 0)?;
        match ops[1] {
            ops::DECORATION_BUILTIN => {
                if ops.len() != 3 {
                    return Err(Error::InvalidBinary);
                }
                if self
                    .builtin_decos
                    .iter()
                    .any(|&(t, _)| t == target)
                {
                    return Err(Error::InvalidBinary);
                }
                self.builtin_decos.push((target, ops[2]));
                Ok(())
            }
            ops::DECORATION_LINKAGE_ATTRIBUTES => {
                if ops.len() < 3 {
                    return Err(Error::InvalidBinary);
                }
                let kind = ops[ops.len() - 1];
                if kind > ops::LINKAGE_IMPORT {
                    return Err(Error::InvalidBinary);
                }
                let (name, used) = decode::read_string(&ops[2..ops.len() - 1], 0)?;
                if used != ops.len() - 3 {
                    return Err(Error::InvalidBinary);
                }
                if self.linkages.iter().any(|&(t, _, _)| t == target) {
                    return Err(Error::InvalidBinary);
                }
                self.linkages.push((target, kind, name));
                Ok(())
            }
            ops::DECORATION_ARRAY_STRIDE => {
                if ops.len() != 3 {
                    return Err(Error::InvalidBinary);
                }
                self.decor.array_strides.push((target, ops[2]));
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// `OpMemberDecorate`: records explicit member offsets.
    fn member_decorate(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 3 {
            return Err(Error::InvalidBinary);
        }
        let target = self.ref_at(ops, 0)?;
        if ops[2] == ops::DECORATION_OFFSET {
            if ops.len() != 4 {
                return Err(Error::InvalidBinary);
            }
            self.decor
                .member_offsets
                .push((target, ops[1], ops[3]));
        }
        Ok(())
    }

    /// Declares a `void`/`bool` type with only a result id.
    fn type_scalar(&mut self, ops: &[u32], kind: Type) -> Result<()> {
        if ops.len() != 1 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        self.check_id(id)?;
        self.types[id as usize] = kind;
        Ok(())
    }

    /// `OpTypeInt`: widths 8/16/32/64, signedness 0/1.
    fn type_int(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        if !matches!(ops[1], 8 | 16 | 32 | 64) || ops[2] > 1 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        self.check_id(id)?;
        self.types[id as usize] = Type::Int {
            width: ops[1] as u16,
            signed: ops[2] == 1,
        };
        Ok(())
    }

    /// `OpTypeFloat`: widths 32/64 (the interpreter has no `half`
    /// arithmetic), optional encoding must be `0`.
    fn type_float(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 && !(ops.len() == 3 && ops[2] == 0) {
            return Err(Error::InvalidBinary);
        }
        if !matches!(ops[1], 32 | 64) {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        self.check_id(id)?;
        self.types[id as usize] = Type::Float { width: ops[1] as u16 };
        Ok(())
    }

    /// `OpTypeVector`: `1..=16` elements; the element may be declared
    /// later and is validated at use.
    fn type_vector(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        if ops[2] == 0 || ops[2] > 16 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        let elem = self.ref_at(ops, 1)?;
        self.check_id(id)?;
        self.types[id as usize] = Type::Vector {
            elem,
            len: ops[2] as u16,
        };
        Ok(())
    }

    /// `OpTypeArray`: the length must already be a positive 32-bit
    /// integer constant.
    fn type_array(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        let elem = self.ref_at(ops, 1)?;
        let length = self.ref_at(ops, 2)?;
        let slot = self.consts[length as usize];
        if slot.count == 0 {
            return Err(Error::InvalidBinary);
        }
        let length_ty = self.values[length as usize].ty;
        if !matches!(
            self.types.get(length_ty as usize),
            Some(Type::Int { width: 32, .. })
        ) {
            return Err(Error::InvalidBinary);
        }
        let lane = self
            .const_pool
            .get(slot.off as usize)
            .copied()
            .ok_or(Error::InvalidBinary)?;
        if lane == 0 || lane > u64::from(u32::MAX) {
            return Err(Error::InvalidBinary);
        }
        self.check_id(id)?;
        self.types[id as usize] = Type::Array {
            elem,
            len: lane as u32,
        };
        Ok(())
    }

    /// `OpTypeStruct`: at least one field, all ids in range.
    fn type_struct(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 2 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        let mut fields: Vec<u32> = Vec::new();
        fields
            .try_reserve_exact(ops.len() - 1)
            .map_err(|_| Error::OutOfMemory)?;
        for &raw in &ops[1..] {
            fields.push(self.ref_word(raw)?);
        }
        self.check_id(id)?;
        self.types[id as usize] = Type::Struct { fields };
        Ok(())
    }

    /// `OpTypePointer`: storage classes `0..=7`; the pointee may be
    /// declared later.
    fn type_pointer(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 || ops[1] > ops::STORAGE_FUNCTION {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        let pointee = self.ref_at(ops, 2)?;
        self.check_id(id)?;
        self.types[id as usize] = Type::Pointer {
            storage: ops[1],
            pointee,
        };
        Ok(())
    }

    /// `OpTypeFunction`: return type and parameter types in range.
    fn type_function(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 2 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        let ret = self.ref_at(ops, 1)?;
        let mut params: Vec<u32> = Vec::new();
        params
            .try_reserve_exact(ops.len() - 2)
            .map_err(|_| Error::OutOfMemory)?;
        for &raw in &ops[2..] {
            params.push(self.ref_word(raw)?);
        }
        self.check_id(id)?;
        self.types[id as usize] = Type::Function { ret, params };
        Ok(())
    }

    /// `OpConstantTrue`/`OpConstantFalse`.
    fn constant_bool(&mut self, opcode: u16, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        if !matches!(self.types.get(ty as usize), Some(Type::Bool)) {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 1)?;
        let off = self.def_const(ty, id, 1)?;
        let lane = u64::from(opcode == ops::OP_CONSTANT_TRUE);
        self.const_pool[off as usize] = lane;
        Ok(())
    }

    /// `OpConstant`: one value word for widths up to 32, two for 64;
    /// integer values are masked to their width.
    fn constant_scalar(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 && ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let width = match self.declared(ty)? {
            Type::Int { width, .. } | Type::Float { width } => *width,
            _ => return Err(Error::InvalidBinary),
        };
        let words = &ops[2..];
        if width <= 32 && words.len() != 1 {
            return Err(Error::InvalidBinary);
        }
        if width == 64 && words.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let mut lane = u64::from(words[0]) & mask;
        if width == 64 {
            lane |= u64::from(words[1]) << 32;
        }
        let off = self.def_const(ty, id, 1)?;
        self.const_pool[off as usize] = lane;
        Ok(())
    }

    /// `OpConstantComposite`: exact constituent types whose pooled lanes
    /// sum to the composite's own lane count.
    fn constant_composite(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let parts = &ops[2..];
        let expected = self.constituent_count(ty)?;
        if expected as usize != parts.len() {
            return Err(Error::InvalidBinary);
        }
        let mut total = 0u64;
        for (index, &part) in parts.iter().enumerate() {
            let part = self.ref_word(part)?;
            if self.consts[part as usize].count == 0 {
                return Err(Error::InvalidBinary);
            }
            let wanted = self.constituent_ty(ty, index as u32)?;
            if self.values[part as usize].ty != wanted {
                return Err(Error::InvalidBinary);
            }
            total = total
                .checked_add(u64::from(self.consts[part as usize].count))
                .ok_or(Error::InvalidBinary)?;
        }
        let want = u64::from(types::lanes_any(&self.types, ty, &self.decor)?);
        if total != want || total > u64::from(u32::MAX) {
            return Err(Error::InvalidBinary);
        }
        let off = self.def_const(ty, id, total as u32)?;
        let mut cursor = off as usize;
        for &part in parts {
            let slot = self.consts[part as usize];
            let start = slot.off as usize;
            let end = start
                .checked_add(slot.count as usize)
                .ok_or(Error::InvalidBinary)?;
            // Every constituent predates this composite, so its lanes
            // sit strictly before the freshly appended target range —
            // `copy_within` never has to shift live data.
            let len = end - start;
            let stop = cursor
                .checked_add(len)
                .ok_or(Error::InvalidBinary)?;
            if end > cursor || stop > self.const_pool.len() {
                return Err(Error::InvalidBinary);
            }
            self.const_pool.copy_within(start..end, cursor);
            cursor = stop;
        }
        Ok(())
    }

    /// `OpConstantNull`: one zeroed pool entry of the type's lane count.
    fn constant_null(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let lanes = types::lanes_any(&self.types, ty, &self.decor)?;
        self.def_const(ty, id, lanes)?;
        Ok(())
    }

    /// `OpUndef`: a zeroed pool entry, so reads are deterministic.
    fn constant_undef(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        if matches!(self.types.get(ty as usize), Some(Type::Void)) {
            return Err(Error::InvalidBinary);
        }
        let lanes = types::lanes_any(&self.types, ty, &self.decor)?;
        self.def_const(ty, id, lanes)?;
        Ok(())
    }

    /// Module-scope `OpVariable`: `Input` (built-in materialization) or
    /// `Workgroup` (shared arena), never initialized.
    fn module_variable(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let (storage, pointee) = match self.declared(ty)? {
            Type::Pointer { storage, pointee } => (*storage, *pointee),
            _ => return Err(Error::InvalidBinary),
        };
        self.def_value(ty, id)?;
        match storage {
            ops::STORAGE_INPUT => self.input_vars.push(InputVar {
                ptr: id,
                pointee,
                number: 0,
                off: 0,
                lanes: 0,
            }),
            ops::STORAGE_WORKGROUP => self.arena_vars.push(ArenaVar {
                ptr: id,
                pointee,
                storage,
                init: None,
                off: 0,
            }),
            _ => return Err(Error::InvalidBinary),
        }
        Ok(())
    }

    /// `OpFunction`: opens a new function record and scan state.
    fn start_function(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        let ret = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let fn_ty = self.ref_at(ops, 3)?;
        if ops[2] != 0 || self.current.is_some() {
            return Err(Error::InvalidBinary);
        }
        self.declared(ret)?;
        match self.declared(fn_ty)? {
            Type::Function { .. } => {}
            _ => return Err(Error::InvalidBinary),
        }
        self.check_id(id)?;
        let index = self.funcs.len() as u32;
        let reserve = self.total_instrs;
        self.funcs.push(Func {
            ret_ty: ret,
            fn_ty,
            params: Vec::new(),
            calls: Vec::new(),
            has_body: false,
            entry: 0,
            link_name: None,
        });
        self.func_index[id as usize] = index + 1;
        let mut labels = Vec::new();
        labels
            .try_reserve_exact(reserve)
            .map_err(|_| Error::OutOfMemory)?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(reserve)
            .map_err(|_| Error::OutOfMemory)?;
        let mut phi_sites = Vec::new();
        phi_sites
            .try_reserve_exact(reserve)
            .map_err(|_| Error::OutOfMemory)?;
        self.current = Some(CurFunc {
            index,
            phase_header: true,
            expect_label: false,
            labels,
            pending,
            phi_sites,
        });
        Ok(())
    }

    /// Pass A handler for instructions inside a function.
    fn body(&mut self, words: &[u32], instr: Instr, index: usize) -> Result<()> {
        let ops = instr.words(words);
        let opcode = instr.opcode;
        {
            let cur = self
                .current
                .as_ref()
                .ok_or(Error::InvalidBinary)?;
            if cur.expect_label
                && !matches!(
                    opcode,
                    ops::OP_LABEL | ops::OP_NOP | ops::OP_LINE | ops::OP_NO_LINE | ops::OP_FUNCTION_END
                )
            {
                return Err(Error::InvalidBinary);
            }
            if cur.phase_header
                && !matches!(
                    opcode,
                    ops::OP_FUNCTION_PARAMETER
                        | ops::OP_LABEL
                        | ops::OP_FUNCTION_END
                        | ops::OP_NOP
                        | ops::OP_LINE
                        | ops::OP_NO_LINE
                )
            {
                return Err(Error::InvalidBinary);
            }
        }
        match opcode {
            ops::OP_LABEL => self.label(ops, index),
            ops::OP_FUNCTION_PARAMETER => self.function_parameter(ops),
            ops::OP_FUNCTION_END => self.end_function(),
            ops::OP_SELECTION_MERGE => self.merge(ops, 2),
            ops::OP_LOOP_MERGE => self.merge(ops, 3),
            ops::OP_BRANCH => self.branch(ops),
            ops::OP_BRANCH_CONDITIONAL => self.branch_conditional(ops),
            ops::OP_SWITCH => self.switch(ops),
            ops::OP_RETURN => self.return_void(),
            ops::OP_RETURN_VALUE => self.return_value(ops),
            ops::OP_UNREACHABLE => self.unreachable(),
            ops::OP_PHI => self.phi(ops),
            ops::OP_UNDEF => Err(Error::InvalidBinary),
            ops::OP_LOAD => self.load(ops),
            ops::OP_STORE => self.store(ops),
            ops::OP_ACCESS_CHAIN | ops::OP_IN_BOUNDS_ACCESS_CHAIN => self.access_chain(ops, false),
            ops::OP_PTR_ACCESS_CHAIN => self.access_chain(ops, true),
            ops::OP_VARIABLE => self.body_variable(ops),
            ops::OP_COPY_OBJECT => self.copy_object(ops),
            ops::OP_SELECT => self.select(ops),
            ops::OP_COMPOSITE_CONSTRUCT => self.composite_construct(ops),
            ops::OP_COMPOSITE_EXTRACT => self.composite_extract(ops),
            ops::OP_COMPOSITE_INSERT => self.composite_insert(ops),
            ops::OP_VECTOR_EXTRACT_DYNAMIC => self.vector_extract_dynamic(ops),
            ops::OP_VECTOR_INSERT_DYNAMIC => self.vector_insert_dynamic(ops),
            ops::OP_VECTOR_SHUFFLE => self.vector_shuffle(ops),
            ops::OP_EXT_INST => self.ext_inst(ops),
            ops::OP_FUNCTION_CALL => self.function_call(ops),
            ops::OP_CONTROL_BARRIER => self.control_barrier(ops),
            ops::OP_MEMORY_BARRIER => self.memory_barrier(ops),
            ops::OP_GROUP_ALL | ops::OP_GROUP_ANY => self.group_bool(ops),
            ops::OP_GROUP_BROADCAST => self.group_broadcast(ops),
            ops::OP_GROUP_I_ADD
            | ops::OP_GROUP_F_ADD
            | ops::OP_GROUP_F_MIN
            | ops::OP_GROUP_U_MIN
            | ops::OP_GROUP_S_MIN
            | ops::OP_GROUP_F_MAX
            | ops::OP_GROUP_U_MAX
            | ops::OP_GROUP_S_MAX => self.group_reduce(opcode, ops),
            ops::OP_CONVERT_F_TO_U
            | ops::OP_CONVERT_F_TO_S
            | ops::OP_CONVERT_S_TO_F
            | ops::OP_CONVERT_U_TO_F
            | ops::OP_U_CONVERT
            | ops::OP_S_CONVERT
            | ops::OP_F_CONVERT
            | ops::OP_BITCAST => self.convert(opcode, ops),
            ops::OP_S_NEGATE | ops::OP_F_NEGATE | ops::OP_LOGICAL_NOT | ops::OP_NOT => {
                self.unary(opcode, ops)
            }
            ops::OP_I_ADD..=ops::OP_F_REM
            | ops::OP_LOGICAL_OR
            | ops::OP_LOGICAL_AND
            | ops::OP_SHIFT_RIGHT_LOGICAL
            | ops::OP_SHIFT_RIGHT_ARITHMETIC
            | ops::OP_SHIFT_LEFT_LOGICAL
            | ops::OP_BITWISE_OR
            | ops::OP_BITWISE_XOR
            | ops::OP_BITWISE_AND => self.binary(opcode, ops),
            ops::OP_I_EQUAL
            | ops::OP_I_NOT_EQUAL
            | ops::OP_U_GREATER_THAN
            | ops::OP_S_GREATER_THAN
            | ops::OP_U_GREATER_THAN_EQUAL
            | ops::OP_S_GREATER_THAN_EQUAL
            | ops::OP_U_LESS_THAN
            | ops::OP_S_LESS_THAN
            | ops::OP_U_LESS_THAN_EQUAL
            | ops::OP_S_LESS_THAN_EQUAL
            | ops::OP_F_ORD_EQUAL
            | ops::OP_F_ORD_NOT_EQUAL
            | ops::OP_F_ORD_LESS_THAN
            | ops::OP_F_ORD_GREATER_THAN
            | ops::OP_F_ORD_LESS_THAN_EQUAL
            | ops::OP_F_ORD_GREATER_THAN_EQUAL => self.comparison(opcode, ops),
            _ => Err(Error::InvalidBinary),
        }
    }

    /// The function currently being scanned.
    fn current_index(&self) -> Result<u32> {
        self.current
            .as_ref()
            .map(|cur| cur.index)
            .ok_or(Error::InvalidBinary)
    }

    /// `OpLabel`: records the block and (first label) the entry ip.
    fn label(&mut self, ops: &[u32], index: usize) -> Result<()> {
        if ops.len() != 1 {
            return Err(Error::InvalidBinary);
        }
        let id = self.ref_at(ops, 0)?;
        self.check_id(id)?;
        self.label_ip[id as usize] = index as u32 + 1;
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        if cur.phase_header {
            cur.phase_header = false;
            self.funcs[cur.index as usize].entry = index as u32;
        }
        cur.labels.push(id);
        cur.expect_label = false;
        Ok(())
    }

    /// `OpFunctionParameter`: one per signature slot, header phase only.
    fn function_parameter(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let cur = self
            .current
            .as_ref()
            .ok_or(Error::InvalidBinary)?;
        if !cur.phase_header {
            return Err(Error::InvalidBinary);
        }
        let index = cur.index;
        let fn_ty = self.funcs[index as usize].fn_ty;
        let position = self.funcs[index as usize].params.len();
        let expected = match self.declared(fn_ty)? {
            Type::Function { params, .. } => params
                .get(position)
                .copied()
                .ok_or(Error::InvalidBinary)?,
            _ => return Err(Error::InvalidBinary),
        };
        if ty != expected {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)?;
        self.funcs[index as usize].params.push(id);
        Ok(())
    }

    /// `OpFunctionEnd`: closes the function and validates the targets
    /// whose membership only the finished label set can answer.
    fn end_function(&mut self) -> Result<()> {
        let cur = self.current.take().ok_or(Error::InvalidBinary)?;
        let fn_ty = self.funcs[cur.index as usize].fn_ty;
        let expected = match self.declared(fn_ty)? {
            Type::Function { params, .. } => params.len(),
            _ => return Err(Error::InvalidBinary),
        };
        if self.funcs[cur.index as usize].params.len() != expected {
            return Err(Error::InvalidBinary);
        }
        if !cur.labels.is_empty() {
            self.funcs[cur.index as usize].has_body = true;
        }
        for &target in &cur.pending {
            if !cur.labels.contains(&target) {
                return Err(Error::InvalidBinary);
            }
        }
        for (values, parents) in &cur.phi_sites {
            for &value in values {
                let slot = self
                    .values
                    .get(value as usize)
                    .ok_or(Error::InvalidBinary)?;
                if slot.ty == 0 {
                    return Err(Error::InvalidBinary);
                }
            }
            for &parent in parents {
                if !cur.labels.contains(&parent) {
                    return Err(Error::InvalidBinary);
                }
            }
        }
        Ok(())
    }

    /// `OpSelectionMerge`/`OpLoopMerge`: defer the merge (and loop
    /// continue) targets to the label check.
    fn merge(&mut self, ops: &[u32], words: usize) -> Result<()> {
        if ops.len() != words {
            return Err(Error::InvalidBinary);
        }
        let target = self.ref_at(ops, 0)?;
        if words == 3 {
            let continue_target = self.ref_at(ops, 1)?;
            let cur = self
                .current
                .as_mut()
                .ok_or(Error::InvalidBinary)?;
            cur.pending.push(continue_target);
        }
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.pending.push(target);
        cur.expect_label = false;
        Ok(())
    }

    /// `OpBranch`: the target joins the label check.
    fn branch(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 1 {
            return Err(Error::InvalidBinary);
        }
        let target = self.ref_at(ops, 0)?;
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.pending.push(target);
        cur.expect_label = true;
        Ok(())
    }

    /// `OpBranchConditional`: a boolean condition and two label targets.
    fn branch_conditional(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 && ops.len() != 5 {
            return Err(Error::InvalidBinary);
        }
        let cond = self.ref_at(ops, 0)?;
        if self.kind_of(self.val_ty(cond)?)? != Kind::Bool {
            return Err(Error::InvalidBinary);
        }
        let then_target = self.ref_at(ops, 1)?;
        let else_target = self.ref_at(ops, 2)?;
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.pending.push(then_target);
        cur.pending.push(else_target);
        cur.expect_label = true;
        Ok(())
    }

    /// `OpSwitch`: an integer selector plus `(literal, label)` pairs.
    fn switch(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 2 || !ops.len().is_multiple_of(2) {
            return Err(Error::InvalidBinary);
        }
        let selector = self.ref_at(ops, 0)?;
        if !matches!(self.declared(self.val_ty(selector)?)?, Type::Int { .. }) {
            return Err(Error::InvalidBinary);
        }
        let default = self.ref_at(ops, 1)?;
        let mut targets: Vec<u32> = Vec::new();
        targets
            .try_reserve_exact(1 + ops[2..].len() / 2)
            .map_err(|_| Error::OutOfMemory)?;
        targets.push(default);
        for pair in ops[2..].chunks(2) {
            targets.push(self.ref_word(pair[1])?);
        }
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.pending
            .try_reserve(targets.len())
            .map_err(|_| Error::OutOfMemory)?;
        cur.pending.extend(targets);
        cur.expect_label = true;
        Ok(())
    }

    /// `OpReturn`: only from a void function.
    fn return_void(&mut self) -> Result<()> {
        let index = self.current_index()?;
        let ret = self.funcs[index as usize].ret_ty;
        if !matches!(self.declared(ret)?, Type::Void) {
            return Err(Error::InvalidBinary);
        }
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.expect_label = true;
        Ok(())
    }

    /// `OpReturnValue`: the operand must have the declared return type.
    fn return_value(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 1 {
            return Err(Error::InvalidBinary);
        }
        let value = self.ref_at(ops, 0)?;
        let index = self.current_index()?;
        let ret = self.funcs[index as usize].ret_ty;
        if matches!(self.declared(ret)?, Type::Void) || self.val_ty(value)? != ret {
            return Err(Error::InvalidBinary);
        }
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.expect_label = true;
        Ok(())
    }

    /// `OpUnreachable`.
    fn unreachable(&mut self) -> Result<()> {
        if self.current.is_none() {
            return Err(Error::InvalidBinary);
        }
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.expect_label = true;
        Ok(())
    }

    /// `OpPhi`: records operands/parents for the deferred check (a phi
    /// may textually forward-reference the latch value it merges).
    fn phi(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 4 || !ops.len().is_multiple_of(2) {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let pairs = (ops.len() - 2) / 2;
        let mut values: Vec<u32> = Vec::new();
        values
            .try_reserve_exact(pairs)
            .map_err(|_| Error::OutOfMemory)?;
        let mut parents: Vec<u32> = Vec::new();
        parents
            .try_reserve_exact(pairs)
            .map_err(|_| Error::OutOfMemory)?;
        for pair in ops[2..].chunks(2) {
            values.push(self.ref_word(pair[0])?);
            parents.push(self.ref_word(pair[1])?);
        }
        self.def_value(ty, id)?;
        let cur = self
            .current
            .as_mut()
            .ok_or(Error::InvalidBinary)?;
        cur.phi_sites.push((values, parents));
        Ok(())
    }

    /// `OpLoad`: the result must have the pointer's pointee type.
    fn load(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 && ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let ptr = self.ref_at(ops, 2)?;
        let pointee = self.pointer_pointee(self.val_ty(ptr)?)?;
        if pointee != ty {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// `OpStore`: the object must have the pointer's pointee type.
    fn store(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 && ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let ptr = self.ref_at(ops, 0)?;
        let object = self.ref_at(ops, 1)?;
        let pointee = self.pointer_pointee(self.val_ty(ptr)?)?;
        if self.val_ty(object)? != pointee {
            return Err(Error::InvalidBinary);
        }
        Ok(())
    }

    /// The pointee of a declared pointer type.
    fn pointer_pointee(&self, ty: u32) -> Result<u32> {
        match self.declared(ty)? {
            Type::Pointer { pointee, .. } => Ok(*pointee),
            _ => Err(Error::InvalidBinary),
        }
    }

    /// Steps one index into `current`, returning the next type: vectors
    /// and arrays yield their element, structures their (constant)
    /// member.
    fn walk_index(&self, current: u32, index: u32) -> Result<u32> {
        if !matches!(self.declared(self.val_ty(index)?)?, Type::Int { .. }) {
            return Err(Error::InvalidBinary);
        }
        match self.declared(current)? {
            Type::Vector { elem, .. } | Type::Array { elem, .. } => Ok(*elem),
            Type::Struct { fields } => {
                let lane = self.const_lane(index)?;
                let member = u32::try_from(lane).map_err(|_| Error::InvalidBinary)?;
                fields
                    .get(member as usize)
                    .copied()
                    .ok_or(Error::InvalidBinary)
            }
            _ => Err(Error::InvalidBinary),
        }
    }

    /// `OpAccessChain`/`OpInBoundsAccessChain` and the workspace's
    /// element-offset `OpPtrAccessChain` (no `Element` operand).
    fn access_chain(&mut self, ops: &[u32], ptr: bool) -> Result<()> {
        let minimum = if ptr { 4 } else { 3 };
        if ops.len() < minimum {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let base = self.ref_at(ops, 2)?;
        let (result_storage, result_pointee) = match self.declared(ty)? {
            Type::Pointer { storage, pointee } => (*storage, *pointee),
            _ => return Err(Error::InvalidBinary),
        };
        let base_ty = self.val_ty(base)?;
        let (base_storage, mut current) = match self.declared(base_ty)? {
            Type::Pointer { storage, pointee } => (*storage, *pointee),
            _ => return Err(Error::InvalidBinary),
        };
        if base_storage != result_storage {
            return Err(Error::InvalidBinary);
        }
        let walk: &[u32] = if ptr {
            if self.is_type(ops[3]) {
                // A standard `Element` operand is outside this subset.
                return Err(Error::InvalidBinary);
            }
            // The element offset only has to be an integer: it scales
            // the base without stepping into a different type. The
            // remaining indices walk into the pointee.
            let element = self
                .ref_word(ops[3])
                .and_then(|id| self.val_ty(id))?;
            if !matches!(self.declared(element)?, Type::Int { .. }) {
                return Err(Error::InvalidBinary);
            }
            &ops[4..]
        } else {
            &ops[3..]
        };
        for &raw in walk {
            let index = self.ref_word(raw)?;
            current = self.walk_index(current, index)?;
        }
        if current != result_pointee {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// Body `OpVariable`: function/private (item arena, constant
    /// initializer) or workgroup (group arena, never initialized).
    fn body_variable(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 && ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let (storage, pointee) = match self.declared(ty)? {
            Type::Pointer { storage, pointee } => (*storage, *pointee),
            _ => return Err(Error::InvalidBinary),
        };
        if !matches!(
            storage,
            ops::STORAGE_FUNCTION | ops::STORAGE_PRIVATE | ops::STORAGE_WORKGROUP
        ) {
            return Err(Error::InvalidBinary);
        }
        let init = if ops.len() == 4 {
            if storage == ops::STORAGE_WORKGROUP {
                return Err(Error::InvalidBinary);
            }
            let init = self.ref_at(ops, 2)?;
            let slot = self.consts[init as usize];
            if slot.count == 0 {
                return Err(Error::InvalidBinary);
            }
            let expected = types::lanes_any(&self.types, pointee, &self.decor)?;
            if slot.count != expected {
                return Err(Error::InvalidBinary);
            }
            Some(init)
        } else {
            None
        };
        self.def_value(ty, id)?;
        self.arena_vars
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.arena_vars.push(ArenaVar {
            ptr: id,
            pointee,
            storage,
            init,
            off: 0,
        });
        Ok(())
    }

    /// `OpCopyObject`: a bitwise copy at the same type.
    fn copy_object(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let operand = self.ref_at(ops, 2)?;
        if self.val_ty(operand)? != ty {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// `OpSelect`: boolean condition, matching arms, result type.
    fn select(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 5 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let cond = self.ref_at(ops, 2)?;
        let yes = self.ref_at(ops, 3)?;
        let no = self.ref_at(ops, 4)?;
        if self.kind_of(self.val_ty(cond)?)? != Kind::Bool {
            return Err(Error::InvalidBinary);
        }
        if self.val_ty(yes)? != ty || self.val_ty(no)? != ty {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// `OpCompositeConstruct`: one constituent per component, each with
    /// exactly the component's declared type.
    fn composite_construct(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let parts = &ops[2..];
        if self.constituent_count(ty)? as usize != parts.len() {
            return Err(Error::InvalidBinary);
        }
        for (index, &raw) in parts.iter().enumerate() {
            let part = self.ref_word(raw)?;
            let wanted = self.constituent_ty(ty, index as u32)?;
            if self.val_ty(part)? != wanted {
                return Err(Error::InvalidBinary);
            }
        }
        self.def_value(ty, id)
    }

    /// Walks literal component indexes through a composite type.
    fn walk_literals(&self, mut current: u32, literals: &[u32]) -> Result<u32> {
        for &literal in literals {
            current = match self.declared(current)? {
                Type::Vector { elem, len } if literal < u32::from(*len) => *elem,
                Type::Array { elem, len } if literal < *len => *elem,
                Type::Struct { fields } => fields
                    .get(literal as usize)
                    .copied()
                    .ok_or(Error::InvalidBinary)?,
                _ => return Err(Error::InvalidBinary),
            };
        }
        Ok(current)
    }

    /// `OpCompositeExtract`: a static literal walk that lands on `ty`.
    fn composite_extract(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let composite = self.ref_at(ops, 2)?;
        let start = self.val_ty(composite)?;
        if self.walk_literals(start, &ops[3..])? != ty {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// `OpCompositeInsert`: the walk from the object type lands on the
    /// constituent's type, and the result is the object type.
    fn composite_insert(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 5 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let object = self.ref_at(ops, 2)?;
        let constituent = self.ref_at(ops, 3)?;
        if self.val_ty(object)? != ty {
            return Err(Error::InvalidBinary);
        }
        if self.walk_literals(ty, &ops[4..])? != self.val_ty(constituent)? {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// `OpVectorExtractDynamic`: a dynamic lane of a vector.
    fn vector_extract_dynamic(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let vector = self.ref_at(ops, 2)?;
        let index = self.ref_at(ops, 3)?;
        if !matches!(self.declared(self.val_ty(index)?)?, Type::Int { .. }) {
            return Err(Error::InvalidBinary);
        }
        match self.declared(self.val_ty(vector)?)? {
            Type::Vector { elem, .. } if *elem == ty => {}
            _ => return Err(Error::InvalidBinary),
        }
        self.def_value(ty, id)
    }

    /// `OpVectorInsertDynamic`: a vector with one lane replaced.
    fn vector_insert_dynamic(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 5 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let vector = self.ref_at(ops, 2)?;
        let component = self.ref_at(ops, 3)?;
        let index = self.ref_at(ops, 4)?;
        if !matches!(self.declared(self.val_ty(index)?)?, Type::Int { .. }) {
            return Err(Error::InvalidBinary);
        }
        match self.declared(ty)? {
            Type::Vector { elem, .. } => {
                if self.val_ty(vector)? != ty || self.val_ty(component)? != *elem {
                    return Err(Error::InvalidBinary);
                }
            }
            _ => return Err(Error::InvalidBinary),
        }
        self.def_value(ty, id)
    }

    /// `OpVectorShuffle`: the result picks lanes from two vectors of
    /// the same element type (`0xffffffff` selects an undefined lane).
    fn vector_shuffle(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 5 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let first = self.ref_at(ops, 2)?;
        let second = self.ref_at(ops, 3)?;
        let (elem, count) = match self.declared(ty)? {
            Type::Vector { elem, len } => (*elem, u32::from(*len)),
            _ => return Err(Error::InvalidBinary),
        };
        if count as usize != ops.len() - 4 {
            return Err(Error::InvalidBinary);
        }
        let lanes_first = match self.declared(self.val_ty(first)?)? {
            Type::Vector { elem: other, len } if *other == elem => u32::from(*len),
            _ => return Err(Error::InvalidBinary),
        };
        let lanes_second = match self.declared(self.val_ty(second)?)? {
            Type::Vector { elem: other, len } if *other == elem => u32::from(*len),
            _ => return Err(Error::InvalidBinary),
        };
        for &mask in &ops[4..] {
            if mask != u32::MAX && mask >= lanes_first + lanes_second {
                return Err(Error::InvalidBinary);
            }
        }
        self.def_value(ty, id)
    }

    /// `OpExtInst`: an `OpenCL.std` instruction with its exact arity.
    fn ext_inst(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() < 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let set = self.ref_at(ops, 2)?;
        if self.opencl_set == 0 || set != self.opencl_set {
            return Err(Error::InvalidBinary);
        }
        let args = &ops[4..];
        let arity = builtin::ext_inst_arity(ops[3]).ok_or(Error::InvalidBinary)?;
        if arity as usize != args.len() {
            return Err(Error::InvalidBinary);
        }
        for &raw in args {
            let arg = self.ref_word(raw)?;
            self.val_ty(arg)?;
        }
        // `prefetch` is the one instruction with no result; every other
        // supported instruction defines a value.
        if matches!(self.types.get(ty as usize), Some(Type::Void)) {
            self.def_void(ty, id)
        } else {
            self.def_value(ty, id)
        }
    }

    /// `OpFunctionCall`: the head form when word 0 is a declared type,
    /// the workspace's result-less void form otherwise; cross-references
    /// resolve in pass B (a call may precede its declaration).
    fn function_call(&mut self, ops: &[u32]) -> Result<()> {
        if ops.is_empty() {
            return Err(Error::InvalidBinary);
        }
        let caller = self.current_index()?;
        let (args, callee) = if self.is_type(ops[0]) {
            if ops.len() < 3 {
                return Err(Error::InvalidBinary);
            }
            let ty = self.ref_at(ops, 0)?;
            let id = self.ref_at(ops, 1)?;
            if matches!(self.declared(ty)?, Type::Void) {
                self.def_void(ty, id)?;
            } else {
                self.def_value(ty, id)?;
            }
            (&ops[3..], self.ref_at(ops, 2)?)
        } else {
            (&ops[1..], self.ref_at(ops, 0)?)
        };
        let mut arguments: Vec<u32> = Vec::new();
        arguments
            .try_reserve_exact(args.len())
            .map_err(|_| Error::OutOfMemory)?;
        for &raw in args {
            let arg = self.ref_word(raw)?;
            self.val_ty(arg)?;
            arguments.push(arg);
        }
        self.call_sites
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.call_sites.push((caller, callee, arguments));
        self.funcs[caller as usize]
            .calls
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.funcs[caller as usize].calls.push(callee);
        Ok(())
    }

    /// `OpControlBarrier`: execution scope workgroup/subgroup, memory
    /// scope and semantics must at least be constants.
    fn control_barrier(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let exec_scope = self.ref_at(ops, 0)?;
        let scope = self.const_lane(exec_scope)?;
        if scope != 2 && scope != 3 {
            return Err(Error::InvalidBinary);
        }
        let memory_scope = self.ref_at(ops, 1)?;
        self.const_lane(memory_scope)?;
        let semantics = self.ref_at(ops, 2)?;
        self.const_lane(semantics)?;
        self.needs_collective = true;
        Ok(())
    }

    /// `OpMemoryBarrier`: constant operands; a pure ordering hint this
    /// single-core interpreter needs no rendezvous for.
    fn memory_barrier(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 2 {
            return Err(Error::InvalidBinary);
        }
        let memory_scope = self.ref_at(ops, 0)?;
        self.const_lane(memory_scope)?;
        let semantics = self.ref_at(ops, 1)?;
        self.const_lane(semantics)?;
        Ok(())
    }

    /// The workgroup-only scope operand of a group operation: a group
    /// op's operands are `type, result, scope, …`, so the scope sits at
    /// index two.
    fn group_scope(&self, ops: &[u32]) -> Result<()> {
        let scope_id = self.ref_at(ops, 2)?;
        let scope = self.const_lane(scope_id)?;
        if scope != 2 {
            return Err(Error::InvalidBinary);
        }
        Ok(())
    }

    /// `OpGroupAll`/`OpGroupAny`: boolean reduction over the group.
    fn group_bool(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        self.group_scope(ops)?;
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let value = self.ref_at(ops, 3)?;
        if !matches!(self.declared(ty)?, Type::Bool)
            || !matches!(self.declared(self.val_ty(value)?)?, Type::Bool)
        {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)?;
        self.needs_collective = true;
        Ok(())
    }

    /// `OpGroupBroadcast`: one lane's value shared with the group.
    fn group_broadcast(&mut self, ops: &[u32]) -> Result<()> {
        if ops.len() != 5 {
            return Err(Error::InvalidBinary);
        }
        self.group_scope(ops)?;
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let value = self.ref_at(ops, 3)?;
        let local = self.ref_at(ops, 4)?;
        if self.val_ty(value)? != ty || !matches!(self.declared(self.val_ty(local)?)?, Type::Int { .. }) {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)?;
        self.needs_collective = true;
        Ok(())
    }

    /// `OpGroupIAdd`..`OpGroupSMax`: reductions with a group operation
    /// (`0` reduce, `1` inclusive scan, `2` exclusive scan). The core
    /// grammar orders the operands `Execution, Operation, X` — the
    /// `GroupOperation` literal comes before the value.
    fn group_reduce(&mut self, opcode: u16, ops: &[u32]) -> Result<()> {
        if ops.len() != 5 {
            return Err(Error::InvalidBinary);
        }
        self.group_scope(ops)?;
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let operation = ops[3];
        let value = self.ref_at(ops, 4)?;
        if operation > ops::GROUP_OP_EXCLUSIVE_SCAN || self.val_ty(value)? != ty {
            return Err(Error::InvalidBinary);
        }
        let float = matches!(
            opcode,
            ops::OP_GROUP_F_ADD | ops::OP_GROUP_F_MIN | ops::OP_GROUP_F_MAX
        );
        let want = if float { Kind::Float } else { Kind::Int };
        if self.kind_of(ty)? != want {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)?;
        self.needs_collective = true;
        Ok(())
    }

    /// Conversions and `OpBitcast`.
    fn convert(&mut self, opcode: u16, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let operand = self.ref_at(ops, 2)?;
        let from = self.kind_of(self.val_ty(operand)?)?;
        let to = self.kind_of(ty)?;
        let ok = match opcode {
            ops::OP_CONVERT_F_TO_U | ops::OP_CONVERT_F_TO_S => from == Kind::Float && to == Kind::Int,
            ops::OP_CONVERT_S_TO_F | ops::OP_CONVERT_U_TO_F => from == Kind::Int && to == Kind::Float,
            ops::OP_U_CONVERT | ops::OP_S_CONVERT => from == Kind::Int && to == Kind::Int,
            ops::OP_F_CONVERT => from == Kind::Float && to == Kind::Float,
            ops::OP_BITCAST => {
                types::size_of(&self.types, ty, &self.decor)?
                    == types::size_of(&self.types, self.val_ty(operand)?, &self.decor)?
            }
            _ => false,
        };
        if !ok {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// Unary operations.
    fn unary(&mut self, opcode: u16, ops: &[u32]) -> Result<()> {
        if ops.len() != 3 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let operand = self.ref_at(ops, 2)?;
        if self.val_ty(operand)? != ty {
            return Err(Error::InvalidBinary);
        }
        let want = match opcode {
            ops::OP_S_NEGATE | ops::OP_NOT => Kind::Int,
            ops::OP_F_NEGATE => Kind::Float,
            ops::OP_LOGICAL_NOT => Kind::Bool,
            _ => return Err(Error::InvalidBinary),
        };
        if self.kind_of(ty)? != want {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// Binary arithmetic, logical, shift and bit operations.
    fn binary(&mut self, opcode: u16, ops: &[u32]) -> Result<()> {
        if ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let left = self.ref_at(ops, 2)?;
        let right = self.ref_at(ops, 3)?;
        if self.val_ty(left)? != ty || self.val_ty(right)? != ty {
            return Err(Error::InvalidBinary);
        }
        let want = match opcode {
            ops::OP_F_ADD | ops::OP_F_SUB | ops::OP_F_MUL | ops::OP_F_DIV | ops::OP_F_MOD | ops::OP_F_REM => {
                Kind::Float
            }
            ops::OP_LOGICAL_OR | ops::OP_LOGICAL_AND => Kind::Bool,
            _ => Kind::Int,
        };
        if self.kind_of(ty)? != want {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// Integer and floating comparisons: a scalar boolean result.
    fn comparison(&mut self, opcode: u16, ops: &[u32]) -> Result<()> {
        if ops.len() != 4 {
            return Err(Error::InvalidBinary);
        }
        let ty = self.ref_at(ops, 0)?;
        let id = self.ref_at(ops, 1)?;
        let left = self.ref_at(ops, 2)?;
        let right = self.ref_at(ops, 3)?;
        if !matches!(self.declared(ty)?, Type::Bool) {
            return Err(Error::InvalidBinary);
        }
        let operand = self.val_ty(left)?;
        if operand != self.val_ty(right)? {
            return Err(Error::InvalidBinary);
        }
        let want = match opcode {
            ops::OP_F_ORD_EQUAL
            | ops::OP_F_ORD_NOT_EQUAL
            | ops::OP_F_ORD_LESS_THAN
            | ops::OP_F_ORD_GREATER_THAN
            | ops::OP_F_ORD_LESS_THAN_EQUAL
            | ops::OP_F_ORD_GREATER_THAN_EQUAL => Kind::Float,
            _ => Kind::Int,
        };
        if self.kind_of(operand)? != want {
            return Err(Error::InvalidBinary);
        }
        self.def_value(ty, id)
    }

    /// Pass B: resolves what only the finished module can answer and
    /// materializes the tables the interpreter indexes — entries, calls,
    /// linkage, recursion, built-in variables, arena placement and the
    /// type-shape tables.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBinary`] — a cross-reference fails: a missing
    ///   capability or memory model, a bad entry point, a call-signature
    ///   mismatch, an unresolvable import, recursion, an
    ///   un-materializable built-in, or an arena overflow.
    /// * [`Error::OutOfMemory`] — a table reservation failed.
    fn finish(mut self) -> Result<SpirvModule> {
        if !self.caps_kernel || !self.has_memory_model {
            return Err(Error::InvalidBinary);
        }
        let entries = self.finish_entries()?;
        self.finish_calls()?;
        self.finish_linkage()?;
        self.check_recursion()?;
        let (item, group) = self.place_arenas()?;
        let (ty_size, ty_stride, struct_off) = self.build_indexes()?;
        let Self {
            words,
            instrs,
            bound,
            types,
            values,
            consts,
            const_pool,
            label_ip,
            func_index,
            funcs,
            input_vars,
            arena_vars,
            decor,
            opencl_set,
            needs_collective,
            slots_len,
            ..
        } = self;
        Ok(SpirvModule {
            words,
            instrs,
            bound,
            types,
            values,
            consts,
            const_pool,
            label_ip,
            func_index,
            funcs,
            entries,
            input_vars,
            arena_vars,
            decor,
            ty_size,
            ty_stride,
            struct_off,
            opencl_set,
            needs_collective,
            slots_len,
            item_arena: item,
            group_arena: group,
        })
    }

    /// Resolves every `OpEntryPoint` into an [`SpirvEntry`]: the target
    /// must be a bodied function returning void whose parameter count a
    /// kernel launch can bind, and neither names nor functions repeat.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBinary`] when an entry point fails those checks.
    fn finish_entries(&mut self) -> Result<Vec<SpirvEntry>> {
        let raw_entries = core::mem::take(&mut self.entry_raw);
        let mut entries: Vec<SpirvEntry> = Vec::new();
        entries
            .try_reserve_exact(raw_entries.len())
            .map_err(|_| Error::OutOfMemory)?;
        for (name, func_id) in raw_entries {
            let raw = self
                .func_index
                .get(func_id as usize)
                .copied()
                .unwrap_or(0);
            if raw == 0 {
                return Err(Error::InvalidBinary);
            }
            let func = self
                .funcs
                .get((raw - 1) as usize)
                .ok_or(Error::InvalidBinary)?;
            if !func.has_body
                || func.params.len() > exec::MAX_ARGS
                || !matches!(self.declared(func.ret_ty)?, Type::Void)
            {
                return Err(Error::InvalidBinary);
            }
            if entries
                .iter()
                .any(|other| other.name == name || other.func == func_id)
            {
                return Err(Error::InvalidBinary);
            }
            entries.push(SpirvEntry {
                name,
                func: func_id,
                module: core::ptr::null(),
            });
        }
        Ok(entries)
    }

    /// Checks every recorded call site against its callee's
    /// `OpTypeFunction`: a call may precede the declaration it names,
    /// so arity and argument types only settle here.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBinary`] when a callee is not a function or the
    /// arguments do not match its signature.
    fn finish_calls(&mut self) -> Result<()> {
        for (_, callee, arguments) in core::mem::take(&mut self.call_sites) {
            let raw = self
                .func_index
                .get(callee as usize)
                .copied()
                .unwrap_or(0);
            if raw == 0 {
                return Err(Error::InvalidBinary);
            }
            let func = self
                .funcs
                .get((raw - 1) as usize)
                .ok_or(Error::InvalidBinary)?;
            let expected = match self.declared(func.fn_ty)? {
                Type::Function { params, .. } => params,
                _ => return Err(Error::InvalidBinary),
            };
            if expected.len() != arguments.len() {
                return Err(Error::InvalidBinary);
            }
            for (argument, want) in arguments.iter().zip(expected) {
                if self.val_ty(*argument)? != *want {
                    return Err(Error::InvalidBinary);
                }
            }
        }
        Ok(())
    }

    /// Resolves `LinkageAttributes` imports: an import must name a
    /// body-less function the interpreter implements with a matching
    /// argument count, and every body-less function needs one.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBinary`] for an unknown import name, a signature
    /// mismatch, an import that has a body, or a declaration that never
    /// binds to an implementation.
    fn finish_linkage(&mut self) -> Result<()> {
        for (target, kind, name) in core::mem::take(&mut self.linkages) {
            if kind != ops::LINKAGE_IMPORT {
                continue;
            }
            let raw = self
                .func_index
                .get(target as usize)
                .copied()
                .unwrap_or(0);
            if raw == 0 {
                return Err(Error::InvalidBinary);
            }
            let index = (raw - 1) as usize;
            let func = self
                .funcs
                .get(index)
                .ok_or(Error::InvalidBinary)?;
            if func.has_body {
                return Err(Error::InvalidBinary);
            }
            let argc = builtin::import_fn(&name)
                .ok_or(Error::InvalidBinary)?
                .argc();
            if argc as usize != func.params.len() {
                return Err(Error::InvalidBinary);
            }
            self.funcs[index].link_name = Some(name);
        }
        for func in &self.funcs {
            if !func.has_body && func.link_name.is_none() {
                return Err(Error::InvalidBinary);
            }
        }
        Ok(())
    }

    /// Depth-first search over the call graph with an explicit stack
    /// (never recursion of the loader's own), marking functions white,
    /// gray or black: a gray callee closes a cycle.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBinary`] when the module is recursive or a
    /// recorded callee id names no function.
    fn check_recursion(&self) -> Result<()> {
        let total = self.funcs.len();
        let mut color: Vec<u8> = Vec::new();
        color
            .try_reserve_exact(total)
            .map_err(|_| Error::OutOfMemory)?;
        color.resize(total, 0);
        let mut stack: Vec<(u32, usize)> = Vec::new();
        stack
            .try_reserve_exact(total)
            .map_err(|_| Error::OutOfMemory)?;
        for start in 0..total as u32 {
            if color[start as usize] != 0 {
                continue;
            }
            color[start as usize] = 1;
            stack.clear();
            stack.push((start, 0));
            while let Some(&(func, position)) = stack.last() {
                let calls = &self.funcs[func as usize].calls;
                if position >= calls.len() {
                    color[func as usize] = 2;
                    stack.pop();
                    continue;
                }
                stack.last_mut().ok_or(Error::InvalidBinary)?.1 += 1;
                let callee = calls[position];
                let raw = self
                    .func_index
                    .get(callee as usize)
                    .copied()
                    .unwrap_or(0);
                if raw == 0 {
                    return Err(Error::InvalidBinary);
                }
                let callee_index = (raw - 1) as usize;
                match color[callee_index] {
                    1 => return Err(Error::InvalidBinary),
                    0 => {
                        color[callee_index] = 1;
                        stack.push((callee_index as u32, 0));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Places the built-in input variables and every arena variable:
    /// inputs and function/private storage share the per-item arena,
    /// workgroup storage the group one.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBinary`] for a variable without a materializable
    /// `BuiltIn` number, a pointee whose lane count is neither one nor
    /// three, a pointee without a layout, or an arena budget overrun.
    fn place_arenas(&mut self) -> Result<(u32, u32)> {
        let mut item: u32 = 0;
        let mut group: u32 = 0;
        for var in &mut self.input_vars {
            let number = self
                .builtin_decos
                .iter()
                .find(|&&(target, _)| target == var.ptr)
                .map(|&(_, number)| number)
                .ok_or(Error::InvalidBinary)?;
            if !matches!(number, 24..=34 | 36..=40) {
                return Err(Error::InvalidBinary);
            }
            let lanes = types::lanes(&self.types, var.pointee, &self.decor)?;
            if lanes != 1 && lanes != 3 {
                return Err(Error::InvalidBinary);
            }
            let shape = types::layout(&self.types, var.pointee, &self.decor)?;
            let off = place(&mut item, shape.size, shape.align, MAX_ITEM_ARENA)?;
            var.number = number;
            var.off = off;
            var.lanes = lanes;
        }
        for var in &mut self.arena_vars {
            let shape = types::layout(&self.types, var.pointee, &self.decor)?;
            if var.storage == ops::STORAGE_WORKGROUP {
                var.off = place(&mut group, shape.size, shape.align, MAX_GROUP_ARENA)?;
            } else {
                var.off = place(&mut item, shape.size, shape.align, MAX_ITEM_ARENA)?;
            }
        }
        Ok((item, group))
    }

    /// Fills the interpreter's type-shape tables: the byte size of every
    /// declared type, the stride of every array, and the sorted
    /// `(type, member)` offsets an access chain binary-searches. A type
    /// whose layout fails records zeroes — using it as a value or a
    /// variable was already rejected during the scan.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] for a table reservation.
    fn build_indexes(&self) -> Result<TypeIndexes> {
        let mut ty_size = table(self.bound as usize, 0u32)?;
        let mut ty_stride = table(self.bound as usize, 0u32)?;
        let mut struct_off: Vec<(u64, u32)> = Vec::new();
        for id in 1..self.bound {
            let entry = self
                .types
                .get(id as usize)
                .ok_or(Error::InvalidBinary)?;
            if matches!(entry, Type::Unsupported) {
                continue;
            }
            ty_size[id as usize] = types::size_of(&self.types, id, &self.decor).unwrap_or(0);
            if let Type::Array { len, .. } = entry
                && *len != 0
            {
                ty_stride[id as usize] = ty_size[id as usize] / *len;
            }
            if matches!(entry, Type::Struct { .. })
                && let Ok(offsets) = types::member_offsets(&self.types, id, &self.decor)
            {
                struct_off
                    .try_reserve(offsets.len())
                    .map_err(|_| Error::OutOfMemory)?;
                for (member, offset) in offsets {
                    struct_off.push(((u64::from(id) << 32) | u64::from(member), offset));
                }
            }
        }
        struct_off.sort_unstable_by_key(|&(key, _)| key);
        Ok((ty_size, ty_stride, struct_off))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The workspace's reference kernel (the asm crate's own fixture):
    /// an imported `get_global_id`, a function-local variable, an
    /// `OpenCL.std` square root, and a work-group barrier at execution
    /// scope 2 (the only change from the asm fixture, whose scope-1
    /// barrier this loader rejects).
    const KERNEL: &str = "\
target opencl address physical64 memory opencl

%OpenCL.std = OpExtInstImport \"OpenCL.std\"

OpEntryPoint Kernel %vector_add \"vector_add\"

OpDecorate %get_global_id LinkageAttributes \"get_global_id\" Import

%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%ptr_fn_int = OpTypePointer Function %int
%fn_void = OpTypeFunction %void %ptr_cw_float %ptr_cw_float
%fn_uint_uint = OpTypeFunction %uint %uint

%v0 = OpConstant %uint 0
%v1 = OpConstant %uint 1
%v2 = OpConstant %uint 2
%v_sem = OpConstant %uint 64
%v_int_1 = OpConstant %int 1

%vector_add = OpFunction %void None %fn_void
    %a = OpFunctionParameter %ptr_cw_float
    %b = OpFunctionParameter %ptr_cw_float
    %entry = OpLabel
    %idx = OpFunctionCall %uint %get_global_id %v0
    %slot = OpVariable %ptr_fn_int Function
    OpStore %slot %v_int_1
    %sum = OpLoad %int %slot
    %aval = OpLoad %float %a
    %scaled = OpExtInst %float %OpenCL.std 61 %aval
    OpStore %b %scaled
    %cmp = OpUGreaterThan %bool %idx %v0
    OpSelectionMerge %merge None
    OpBranchConditional %cmp %then %else
    %then = OpLabel
    OpBranch %merge
    %else = OpLabel
    OpBranch %merge
    %merge = OpLabel
    OpControlBarrier %v2 %v1 %v_sem
    OpReturn
OpFunctionEnd

%get_global_id = OpFunction %uint None %fn_uint_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd
";

    /// The same kernel with a barrier at execution scope 1, which the
    /// loader rejects (only scopes 2 and 3 are supported).
    const SCOPE_ONE: &str = "\
target opencl address physical64 memory opencl

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn_void = OpTypeFunction %void
%v1 = OpConstant %uint 1

OpEntryPoint Kernel %main \"main\"

%main = OpFunction %void None %fn_void
%entry = OpLabel
OpControlBarrier %v1 %v1 %v1
OpReturn
OpFunctionEnd
";

    fn bytes_of(words: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn instruction(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let mut words = Vec::with_capacity(operands.len() + 1);
        words.push(((operands.len() as u32 + 1) << 16) | u32::from(opcode));
        words.extend_from_slice(operands);
        words
    }

    fn module_words(bound: u32, instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut words = vec![decode::MAGIC, 0x0001_0000, 0, bound, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
    }

    fn capability_and_model(addressing: u32) -> Vec<Vec<u32>> {
        vec![
            instruction(ops::OP_CAPABILITY, &[ops::CAPABILITY_KERNEL]),
            instruction(ops::OP_MEMORY_MODEL, &[addressing, ops::MEMORY_MODEL_OPENCL]),
        ]
    }

    #[test]
    fn loads_the_workspace_kernel() {
        let source = codevar_ocl_ir::parse::parse(KERNEL).expect("kernel parses");
        let words = codevar_ocl_asm::assemble(&source).expect("kernel assembles");
        let mut module = load(&bytes_of(&words)).expect("kernel loads");

        // Header, instruction stream and id bound.
        assert_eq!(module.words.first(), Some(&decode::MAGIC));
        assert!(module.words.len() >= decode::HEADER_WORDS);
        assert!(module.instrs.len() > 16);
        assert!(module.bound > 16);

        // Type, value, constant and label tables.
        assert!(
            module
                .types
                .iter()
                .any(|ty| matches!(ty, Type::Void))
        );
        assert!(module.values.iter().any(|slot| slot.ty != 0));
        assert!(module.consts.iter().any(|slot| slot.count != 0));
        assert!(!module.const_pool.is_empty());
        assert!(module.label_ip.iter().any(|&ip| ip != 0));
        assert!(module.func_index.iter().any(|raw| *raw != 0));

        // Functions: one bodied kernel plus the imported declaration.
        assert_eq!(module.funcs.len(), 2);
        let kernel_func = module
            .funcs
            .iter()
            .find(|func| func.has_body)
            .expect("kernel has a body");
        assert!(matches!(module.types[kernel_func.ret_ty as usize], Type::Void),);
        assert_eq!(kernel_func.params.len(), 2);
        assert_ne!(kernel_func.entry, 0);
        assert!(!kernel_func.calls.is_empty());
        let import = module
            .funcs
            .iter()
            .find(|func| func.link_name.is_some())
            .expect("import resolves");
        assert_eq!(import.link_name.as_deref(), Some("get_global_id"));
        assert!(!import.has_body);

        // Entry point (unbuilt modules point at nothing).
        assert_eq!(module.entries.len(), 1);
        assert_eq!(module.entries[0].name, "vector_add");
        assert!(module.entries[0].module.is_null());

        // Variables and arenas: one function-local `int`, no builtins.
        assert!(module.input_vars.is_empty());
        assert_eq!(module.arena_vars.len(), 1);
        let slot = &module.arena_vars[0];
        assert_eq!(slot.storage, ops::STORAGE_FUNCTION);
        assert!(slot.init.is_none());
        assert!(module.item_arena > 0);
        assert_eq!(module.group_arena, 0);

        // Shape tables and module flags.
        assert!(module.ty_size.contains(&4));
        assert!(module.ty_stride.iter().all(|&stride| stride == 0));
        assert!(module.struct_off.is_empty());
        assert!(module.decor.member_offsets.is_empty());
        assert!(module.decor.array_strides.is_empty());
        assert_ne!(module.opencl_set, 0);
        assert!(module.needs_collective);
        assert!(module.slots_len > 0);

        // Kernel definitions bind through the boxed module.
        let defs = module.build_defs().expect("defs build");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "vector_add");
        let kinds: &[ArgKind] = &defs[0].kinds;
        assert_eq!(kinds, &[ArgKind::Buffer, ArgKind::Buffer]);
        assert!(defs[0].binding.is_some());
        let stub: exec::KernelFn = spirv_stub;
        assert!(core::ptr::fn_addr_eq(defs[0].func, stub));
        let entry_module = module.entries[0].module;
        assert_eq!(entry_module, core::ptr::from_ref(&*module));
    }

    #[test]
    fn rejects_a_barrier_outside_the_supported_scopes() {
        let source = codevar_ocl_ir::parse::parse(SCOPE_ONE).expect("kernel parses");
        let words = codevar_ocl_asm::assemble(&source).expect("kernel assembles");
        assert!(matches!(load(&bytes_of(&words)), Err(Error::InvalidBinary)));
    }

    #[test]
    fn a_minimal_module_without_kernels_loads() {
        let words = module_words(1, &capability_and_model(ops::ADDRESSING_MODEL_PHYSICAL64));
        let module = load(&bytes_of(&words)).expect("minimal module loads");
        assert!(module.entries.is_empty());
        assert!(module.funcs.is_empty());
        assert_eq!(module.bound, 1);
        assert!(!module.needs_collective);
        assert_eq!(module.opencl_set, 0);
        assert_eq!(module.slots_len, 0);
        assert_eq!(module.item_arena, 0);
        assert_eq!(module.group_arena, 0);
        assert!(module.struct_off.is_empty());
        assert_eq!(module.ty_size, vec![0]);
    }

    #[test]
    fn an_unknown_opcode_is_rejected() {
        let mut instructions = capability_and_model(ops::ADDRESSING_MODEL_PHYSICAL64);
        instructions.push(instruction(0xF0F0, &[]));
        let words = module_words(1, &instructions);
        assert!(matches!(load(&bytes_of(&words)), Err(Error::InvalidBinary)));
    }

    #[test]
    fn a_missing_kernel_capability_is_rejected() {
        let words = module_words(
            1,
            &[instruction(
                ops::OP_MEMORY_MODEL,
                &[ops::ADDRESSING_MODEL_PHYSICAL64, ops::MEMORY_MODEL_OPENCL],
            )],
        );
        assert!(matches!(load(&bytes_of(&words)), Err(Error::InvalidBinary)));
    }

    #[test]
    fn a_non_kernel_entry_point_is_rejected() {
        let mut instructions = capability_and_model(ops::ADDRESSING_MODEL_PHYSICAL64);
        // Execution model 4 (GLCompute), name "k" in one little-endian word.
        instructions.push(instruction(ops::OP_ENTRY_POINT, &[4, 3, 0x0000_006B]));
        let words = module_words(4, &instructions);
        assert!(matches!(load(&bytes_of(&words)), Err(Error::InvalidBinary)));
    }

    #[test]
    fn recursion_is_rejected() {
        let instructions = vec![
            instruction(ops::OP_CAPABILITY, &[ops::CAPABILITY_KERNEL]),
            instruction(
                ops::OP_MEMORY_MODEL,
                &[ops::ADDRESSING_MODEL_PHYSICAL64, ops::MEMORY_MODEL_OPENCL],
            ),
            instruction(ops::OP_TYPE_VOID, &[1]),
            instruction(ops::OP_TYPE_FUNCTION, &[2, 1]),
            instruction(ops::OP_FUNCTION, &[1, 3, 0, 2]),
            instruction(ops::OP_LABEL, &[4]),
            // A result-less void call to the function itself.
            instruction(ops::OP_FUNCTION_CALL, &[3]),
            instruction(ops::OP_FUNCTION_END, &[]),
        ];
        let words = module_words(5, &instructions);
        assert!(matches!(load(&bytes_of(&words)), Err(Error::InvalidBinary)));
    }

    #[test]
    fn physical32_matches_the_host_pointer_width() {
        let words = module_words(1, &capability_and_model(ops::ADDRESSING_MODEL_PHYSICAL32));
        let result = load(&bytes_of(&words));
        if core::mem::size_of::<usize>() == 8 {
            assert!(matches!(result, Err(Error::InvalidBinary)));
        } else {
            assert!(result.is_ok());
        }
    }
}
