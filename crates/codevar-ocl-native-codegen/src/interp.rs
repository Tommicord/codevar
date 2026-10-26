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

//! A reference interpreter for the Codevar intermediate representation.
//!
//! The interpreter runs a [`Module`] the way the machine would, one
//! instruction at a time, and it is the oracle the optimization and
//! vectorization passes are checked against: a pass may rewrite a module
//! as long as the interpreter still produces the same answers.
//!
//! # What it models
//!
//! * **Values.**  [`Value`] carries a scalar (integer bits, a float, a
//!   boolean), a pointer, or an aggregate of values.  Integers keep their
//!   raw bits; the width and the signedness come from the IR type, so the
//!   same runtime value serves `int` and `uint`.
//! * **Memory.**  [`Memory`] holds the allocations: fixed buffers for
//!   module-scope variables, arenas for function-local stack slots, and
//!   buffers the caller lends the interpreter.  A [`Pointer`] is an
//!   allocation plus a byte offset.
//! * **Control flow.**  Blocks are followed one instruction at a time;
//!   `OpPhi` picks the incoming value that matches the predecessor the
//!   block was entered from.
//! * **Work items.**  The front end lowers a work-item query to a load of
//!   a module-scope variable decorated with a built-in name, so the
//!   interpreter fills those variables from a [`WorkItem`] state the
//!   caller sets.
//!
//! # What it refuses
//!
//! Every failure is an [`InterpError`]; the interpreter never panics.
//! A division by zero, a signed division that overflows, a memory access
//! outside an allocation, an unknown extended instruction, and an
//! unsupported operation are all reported, not trapped.
//!
//! # The semantics it fixes
//!
//! The intermediate representation leaves a few corners open; the
//! interpreter settles them, and the passes have to agree with it:
//!
//! * **Integers** are truncated to the width of their type after every
//!   operation, and a signed operation reads the sign from the type, not
//!   from the value.
//! * **Shifts** take the count modulo the width of the type.
//! * **A signed division** truncates toward zero, a division by zero is
//!   an error, and `i64::MIN / -1` is an error as well.
//! * **A floating-point operation** is computed in double precision and
//!   rounded to the width of the result type, which is what a `float`
//!   operation has to produce.
//! * **A float to integer conversion** truncates toward zero and wraps
//!   to the width of the destination; the range outside the destination
//!   is undefined in the dialect, so the interpreter picks a rule and
//!   keeps it.
//! * **A structure** is laid out packed: its fields follow one another
//!   without padding.
//! * **`OpAccessChain`** walks one level into the aggregate the pointer
//!   points at: every index selects an element of an array or a vector,
//!   or a field of a structure.  **`Op::PtrAccessChain`** strides over
//!   the pointee with its first index, so `p[i]` on a pointer to `T`
//!   addresses the `i`-th `T`, and the indices that follow walk into
//!   the pointee.
//! * **A vector operation** is done lane by lane, and a scalar operand
//!   of a vector operation is spread over the lanes.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_ir::lower::lower_source;
//! use codevar_ocl_native_codegen::interp::{Interp, Pointer, Value};
//!
//! let module = lower_source("#[kernel]\nfn zero(out: *mut int) { *out = 0; }")
//!     .expect("the source lowers");
//! let kernel = module.find_function("zero").expect("the kernel exists");
//!
//! let mut interp = Interp::new(&module);
//! let buffer = interp.memory().allocate(4);
//! interp
//!     .run(kernel, &[Value::Pointer(Pointer::new(buffer, 0))])
//!     .expect("the kernel runs");
//! assert_eq!(interp.memory().bytes(buffer).expect("the buffer"), &[0, 0, 0, 0]);
//! ```

use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{
    BasicBlock, BinOp, BlockId, CmpOp, ConstValue, ConvOp, Decor, Inst, Module, Op, Type as IrType, TypeId,
    UnOp, ValueId, ValueKind,
};
use codevar_ocl_ir::spirv::ops::ocl_name;

/// The size a pointer occupies in memory.
const POINTER_SIZE: usize = 8;

/// The mark that says a block has not been entered yet.
const UNVISITED: u64 = u64::MAX;

/// How many instructions one run may execute before it is stopped.
const DEFAULT_STEP_LIMIT: u64 = 64 * 1024 * 1024;

/// How deep the call stack may grow before a call is refused.
const DEFAULT_DEPTH_LIMIT: usize = 256;

/// A runtime value.
///
/// Integers keep their raw bits, zero-extended to 64 bits; the width and
/// the signedness of an integer come from the IR type of the value, so
/// `int` and `uint` share one representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// The result of a function that returns nothing.
    Void,
    /// A boolean.
    Bool(bool),
    /// An integer's raw bits.
    Int(u64),
    /// A floating-point value; the IR type gives the width.
    Float(f64),
    /// A pointer.
    Pointer(Pointer),
    /// A vector, an array, or a structure.
    Aggregate(Vec<Value>),
    /// An undefined value.
    Undef,
}

impl Value {
    /// Builds an integer value from a signed number.
    #[must_use]
    pub fn int(value: i64) -> Self {
        Self::Int(value as u64)
    }

    /// Builds an integer value from an unsigned number.
    #[must_use]
    pub fn uint(value: u64) -> Self {
        Self::Int(value)
    }

    /// Builds a floating-point value.
    #[must_use]
    pub fn float(value: f64) -> Self {
        Self::Float(value)
    }

    /// Builds a boolean value.
    #[must_use]
    pub fn bool(value: bool) -> Self {
        Self::Bool(value)
    }

    /// Builds a pointer value.
    #[must_use]
    pub fn pointer(pointer: Pointer) -> Self {
        Self::Pointer(pointer)
    }
}

/// A handle to one allocation of a [`Memory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocId(pub u32);

impl core::fmt::Display for AllocId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "allocation {}", self.0)
    }
}

/// The address of a byte in an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pointer {
    /// The allocation the address belongs to.
    alloc: AllocId,
    /// The byte offset inside the allocation.
    offset: usize,
}

impl Pointer {
    /// Builds the address of the byte at `offset` in `alloc`.
    #[must_use]
    pub const fn new(alloc: AllocId, offset: usize) -> Self {
        Self { alloc, offset }
    }

    /// The allocation the address belongs to.
    #[must_use]
    pub const fn alloc(self) -> AllocId {
        self.alloc
    }

    /// The byte offset inside the allocation.
    #[must_use]
    pub const fn offset(self) -> usize {
        self.offset
    }
}

/// One allocation of a [`Memory`].
#[derive(Debug)]
enum Allocation<'b> {
    /// A fixed-size buffer.
    Fixed {
        /// The bytes of the buffer.
        bytes: Vec<u8>,
    },
    /// A stack arena: the bytes are handed out by a bump pointer.
    Arena {
        /// The bytes of the arena.
        bytes: Vec<u8>,
        /// How much of the arena is live.
        used: usize,
    },
    /// A buffer the caller lent the interpreter.
    External {
        /// The caller's bytes.
        bytes: &'b mut [u8],
    },
}

/// The interpreter's memory.
///
/// Every allocation is a byte buffer.  Module-scope variables get fixed
/// buffers, function-local stack slots come from arenas, and the caller
/// can lend a buffer of its own with [`Memory::external`].
#[derive(Debug)]
pub struct Memory<'b> {
    /// The allocations, in creation order.
    allocs: Vec<Allocation<'b>>,
}

impl<'b> Default for Memory<'b> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'b> Memory<'b> {
    /// Creates an empty memory.
    #[must_use]
    pub fn new() -> Self {
        Self { allocs: Vec::new() }
    }

    /// Lends a buffer of the caller's to the interpreter.
    ///
    /// The interpreter reads and writes the bytes in place; the caller
    /// sees the result through [`Memory::bytes`] or, once the
    /// interpreter is gone, through the buffer itself.
    pub fn external(&mut self, bytes: &'b mut [u8]) -> AllocId {
        let id = AllocId(u32::try_from(self.allocs.len()).unwrap_or(u32::MAX));
        self.allocs.push(Allocation::External { bytes });
        id
    }

    /// Creates a fixed-size buffer of `size` bytes.
    pub fn allocate(&mut self, size: usize) -> AllocId {
        let id = AllocId(u32::try_from(self.allocs.len()).unwrap_or(u32::MAX));
        self.allocs.push(Allocation::Fixed {
            bytes: vec![0u8; size],
        });
        id
    }

    /// Creates a stack arena: a buffer the interpreter hands out with a
    /// bump pointer.
    pub fn arena(&mut self) -> AllocId {
        let id = AllocId(u32::try_from(self.allocs.len()).unwrap_or(u32::MAX));
        self.allocs.push(Allocation::Arena {
            bytes: Vec::new(),
            used: 0,
        });
        id
    }

    /// The number of allocations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.allocs.len()
    }

    /// True when the memory holds no allocation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allocs.is_empty()
    }

    /// The live bytes of an allocation.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation.
    pub fn bytes(&self, alloc: AllocId) -> Result<&[u8], InterpError> {
        match self.allocs.get(alloc.0 as usize) {
            Some(Allocation::Fixed { bytes }) => Ok(bytes),
            Some(Allocation::Arena { bytes, used }) => Ok(&bytes[..*used]),
            Some(Allocation::External { bytes }) => Ok(bytes),
            None => Err(InterpError::UnknownAllocation { id: alloc }),
        }
    }

    /// The live bytes of an allocation, mutably.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation.
    pub fn bytes_mut(&mut self, alloc: AllocId) -> Result<&mut [u8], InterpError> {
        match self.allocs.get_mut(alloc.0 as usize) {
            Some(Allocation::Fixed { bytes }) => Ok(bytes),
            Some(Allocation::Arena { bytes, used }) => Ok(&mut bytes[..*used]),
            Some(Allocation::External { bytes }) => Ok(bytes),
            None => Err(InterpError::UnknownAllocation { id: alloc }),
        }
    }

    /// How much of an arena is live.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation.
    pub fn used(&self, alloc: AllocId) -> Result<usize, InterpError> {
        match self.allocs.get(alloc.0 as usize) {
            Some(Allocation::Arena { used, .. }) => Ok(*used),
            Some(_) => Ok(0),
            None => Err(InterpError::UnknownAllocation { id: alloc }),
        }
    }

    /// Sets how much of an arena is live.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation, and [`InterpError::OutOfBounds`] when the
    /// arena is not large enough.
    pub fn set_used(&mut self, alloc: AllocId, used: usize) -> Result<(), InterpError> {
        match self.allocs.get_mut(alloc.0 as usize) {
            Some(Allocation::Arena { bytes, used: live }) => {
                if used > bytes.len() {
                    return Err(InterpError::OutOfBounds {
                        alloc,
                        offset: used,
                        len: bytes.len(),
                    });
                }
                *live = used;
                Ok(())
            }
            Some(_) => Ok(()),
            None => Err(InterpError::UnknownAllocation { id: alloc }),
        }
    }

    /// Reserves `size` bytes in an arena and returns their offset.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation, and [`InterpError::OutOfBounds`] when the
    /// allocation is not an arena.
    pub fn bump(&mut self, alloc: AllocId, size: usize) -> Result<usize, InterpError> {
        match self.allocs.get_mut(alloc.0 as usize) {
            Some(Allocation::Arena { bytes, used }) => {
                let end = used
                    .checked_add(size)
                    .ok_or(InterpError::OutOfBounds {
                        alloc,
                        offset: *used,
                        len: bytes.len(),
                    })?;
                if end > bytes.len() {
                    bytes.resize(end, 0);
                }
                let offset = *used;
                *used = end;
                Ok(offset)
            }
            Some(_) => Err(InterpError::NotAnArena { id: alloc }),
            None => Err(InterpError::UnknownAllocation { id: alloc }),
        }
    }

    /// Reads `len` bytes at `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation, and [`InterpError::OutOfBounds`] when the
    /// range leaves the allocation.
    pub fn read(&self, alloc: AllocId, offset: usize, len: usize) -> Result<&[u8], InterpError> {
        let bytes = self.bytes(alloc)?;
        let end = offset
            .checked_add(len)
            .ok_or(InterpError::OutOfBounds {
                alloc,
                offset,
                len: bytes.len(),
            })?;
        if end > bytes.len() {
            return Err(InterpError::OutOfBounds {
                alloc,
                offset,
                len: bytes.len(),
            });
        }
        Ok(&bytes[offset..end])
    }

    /// Writes `data` at `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::UnknownAllocation`] when the handle does
    /// not name an allocation, and [`InterpError::OutOfBounds`] when the
    /// range leaves the allocation.
    pub fn write(&mut self, alloc: AllocId, offset: usize, data: &[u8]) -> Result<(), InterpError> {
        let len = self.bytes(alloc)?.len();
        let end = offset
            .checked_add(data.len())
            .ok_or(InterpError::OutOfBounds { alloc, offset, len })?;
        if end > len {
            return Err(InterpError::OutOfBounds { alloc, offset, len });
        }
        let target = self.bytes_mut(alloc)?;
        target[offset..end].copy_from_slice(data);
        Ok(())
    }
}

/// The state of one work item.
///
/// The front end lowers a work-item query to a load of a module-scope
/// variable decorated with a built-in name, so the interpreter fills
/// those variables from this state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkItem {
    /// The global identifier of the work item, per dimension.
    pub global_id: [u64; 3],
    /// The identifier inside the work group, per dimension.
    pub local_id: [u64; 3],
    /// The identifier of the work group, per dimension.
    pub group_id: [u64; 3],
    /// The global offset of the work group, per dimension.
    pub global_offset: [u64; 3],
    /// The global size of the launch, per dimension.
    pub global_size: [u64; 3],
    /// The number of work groups in the launch, per dimension.
    pub num_groups: [u64; 3],
    /// The size of one work group, per dimension.
    pub local_size: [u64; 3],
    /// The linear global identifier of the work item.
    pub global_linear_id: u64,
    /// The linear identifier of the work item inside its group.
    pub local_linear_id: u64,
}

/// Why the interpreter stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpError {
    /// The value does not name a function.
    NotAFunction {
        /// The offending value.
        id: ValueId,
    },
    /// The function has no body to run.
    NoBody {
        /// The offending function.
        id: ValueId,
    },
    /// The number of arguments does not match the signature.
    ArgumentCount {
        /// How many arguments the signature takes.
        expected: usize,
        /// How many arguments were given.
        found: usize,
    },
    /// An argument has a type the parameter cannot take.
    ArgumentType {
        /// The position of the offending argument.
        index: usize,
    },
    /// A value is not defined where it is used.
    UndefinedValue {
        /// The offending value.
        id: ValueId,
    },
    /// An operation was applied to a value of the wrong shape.
    TypeMismatch {
        /// What the operation expected.
        what: &'static str,
    },
    /// An integer division or remainder by zero.
    DivisionByZero,
    /// A signed division whose result does not fit.
    DivisionOverflow,
    /// A memory access that leaves its allocation.
    OutOfBounds {
        /// The allocation.
        alloc: AllocId,
        /// The offset the access started at.
        offset: usize,
        /// The size of the allocation.
        len: usize,
    },
    /// The handle does not name an allocation.
    UnknownAllocation {
        /// The offending handle.
        id: AllocId,
    },
    /// The allocation is not an arena.
    NotAnArena {
        /// The offending handle.
        id: AllocId,
    },
    /// The handle does not name a block.
    UnknownBlock {
        /// The offending block.
        block: BlockId,
    },
    /// A `OpPhi` has no incoming value for the predecessor the block was
    /// entered from.
    MissingPhiIncoming {
        /// The block the phi lives in.
        block: BlockId,
    },
    /// The extended instruction number is not one the interpreter knows.
    UnknownExtInst {
        /// The offending instruction number.
        inst: u32,
    },
    /// The operation has no interpretation.
    Unsupported {
        /// What was rejected.
        what: &'static str,
    },
    /// The call stack grew past its limit.
    CallDepthExceeded,
    /// The run executed more instructions than its limit allows.
    StepLimitExceeded,
}

impl core::fmt::Display for InterpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotAFunction { id } => write!(f, "{id} is not a function"),
            Self::NoBody { id } => write!(f, "{id} has no body"),
            Self::ArgumentCount { expected, found } => {
                write!(f, "the function takes {expected} argument(s), {found} given")
            }
            Self::ArgumentType { index } => write!(f, "argument {index} has the wrong type"),
            Self::UndefinedValue { id } => write!(f, "{id} is not defined here"),
            Self::TypeMismatch { what } => write!(f, "expected {what}"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::DivisionOverflow => write!(f, "signed division overflows"),
            Self::OutOfBounds { alloc, offset, len } => {
                write!(
                    f,
                    "the access at {offset} leaves {alloc}, which holds {len} byte(s)"
                )
            }
            Self::UnknownAllocation { id } => write!(f, "{id} does not name an allocation"),
            Self::NotAnArena { id } => write!(f, "{id} is not an arena"),
            Self::UnknownBlock { block } => write!(f, "{block} does not exist"),
            Self::MissingPhiIncoming { block } => {
                write!(f, "{block} has no phi value for this predecessor")
            }
            Self::UnknownExtInst { inst } => write!(f, "the extended instruction {inst} is unknown"),
            Self::Unsupported { what } => write!(f, "{what} is not supported"),
            Self::CallDepthExceeded => write!(f, "the call stack is too deep"),
            Self::StepLimitExceeded => write!(f, "the run executed too many instructions"),
        }
    }
}

impl core::error::Error for InterpError {}

/// One slot of a frame: the value a value handle holds.
#[derive(Debug)]
struct Slot {
    /// The generation the value was written in.
    stamp: u64,
    /// The value itself.
    value: Value,
}

/// The state of one call.
#[derive(Debug)]
struct Frame {
    /// The generation of the current call.
    stamp: u64,
    /// The values of the call, indexed by value handle.
    slots: Vec<Slot>,
    /// The arena position each block was entered at.
    marks: Vec<u64>,
    /// The arena the function's stack slots come from.
    arena: AllocId,
}

/// A reference interpreter for a [`Module`].
///
/// The interpreter borrows the module it runs and, when the caller lends
/// it a buffer, the caller's bytes as well.  It is single-threaded: a
/// barrier has no effect, and the work-item state is one work item's.
///
/// # Example
///
/// ```
/// use codevar_ocl_ir::lower::lower_source;
/// use codevar_ocl_native_codegen::interp::{Interp, Pointer, Value};
///
/// let module = lower_source("#[kernel]\nfn zero(out: *mut int) { *out = 7; }")
///     .expect("the source lowers");
/// let kernel = module.find_function("zero").expect("the kernel exists");
///
/// let mut interp = Interp::new(&module);
/// let buffer = interp.memory().allocate(4);
/// interp
///     .run(kernel, &[Value::Pointer(Pointer::new(buffer, 0))])
///     .expect("the kernel runs");
/// assert_eq!(interp.memory().bytes(buffer).expect("the buffer"), &[7, 0, 0, 0]);
/// ```
#[derive(Debug)]
pub struct Interp<'m, 'b> {
    /// The module the interpreter runs.
    module: &'m Module,
    /// The memory the program works on.
    memory: Memory<'b>,
    /// The call frames, one per nesting level.
    frames: Vec<Frame>,
    /// The address of every module-scope variable.
    globals: Vec<Option<Pointer>>,
    /// The work-item state the built-in variables are filled from.
    work_item: WorkItem,
    /// Whether the module-scope variables have been allocated.
    prepared: bool,
    /// How many instructions the current run has executed.
    steps: u64,
    /// How many instructions one run may execute.
    step_limit: u64,
    /// How deep the call stack is.
    depth: usize,
    /// How deep the call stack may grow.
    depth_limit: usize,
}

impl<'m, 'b> Interp<'m, 'b> {
    /// Creates an interpreter for `module`.
    ///
    /// The module-scope variables are allocated when the first run
    /// starts, so the work-item state may still be set afterwards.
    #[must_use]
    pub fn new(module: &'m Module) -> Self {
        let values = module.values().len();
        Self {
            module,
            memory: Memory::new(),
            frames: Vec::new(),
            globals: vec![None; values],
            work_item: WorkItem::default(),
            prepared: false,
            steps: 0,
            step_limit: DEFAULT_STEP_LIMIT,
            depth: 0,
            depth_limit: DEFAULT_DEPTH_LIMIT,
        }
    }

    /// The module the interpreter runs.
    #[must_use]
    pub const fn module(&self) -> &'m Module {
        self.module
    }

    /// The memory the program works on.
    pub fn memory(&mut self) -> &mut Memory<'b> {
        &mut self.memory
    }

    /// The memory the program works on, read-only.
    #[must_use]
    pub const fn memory_ref(&self) -> &Memory<'b> {
        &self.memory
    }

    /// The work-item state the built-in variables are filled from.
    #[must_use]
    pub const fn work_item(&self) -> &WorkItem {
        &self.work_item
    }

    /// Sets the work-item state.
    ///
    /// The built-in variables are refreshed at once when the module has
    /// already been prepared, so a caller may run one work item after
    /// another.  The names the interpreter knows are the ones the OpenCL
    /// environment defines: `GlobalInvocationId`, `LocalInvocationId`,
    /// `WorkgroupId`, `GlobalOffset`, `GlobalSize`, `NumWorkgroups`,
    /// `WorkgroupSize`, `EnqueuedWorkgroupSize`, `GlobalLinearId`, and
    /// `LocalInvocationIndex`.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::OutOfBounds`] when a built-in variable does
    /// not hold the values the state carries.
    pub fn set_work_item(&mut self, work_item: WorkItem) -> Result<(), InterpError> {
        self.work_item = work_item;
        if self.prepared {
            self.refresh_builtins()?;
        }
        Ok(())
    }

    /// Sets how many instructions one run may execute.
    pub fn set_step_limit(&mut self, limit: u64) {
        self.step_limit = limit;
    }

    /// Sets how deep the call stack may grow.
    pub fn set_depth_limit(&mut self, limit: usize) {
        self.depth_limit = limit;
    }

    /// Runs `function` to completion and returns its result.
    ///
    /// The arguments are bound to the function's parameters in order.  A
    /// function that returns nothing yields [`Value::Void`].
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::NotAFunction`] when the value is not a
    /// function, [`InterpError::NoBody`] when it has no body,
    /// [`InterpError::ArgumentCount`] when the arguments do not match the
    /// signature, and [`InterpError::ArgumentType`] when an argument has
    /// a type the parameter cannot take.  Everything the program does
    /// wrong is reported as well.
    pub fn run(&mut self, function: ValueId, args: &[Value]) -> Result<Value, InterpError> {
        self.prepare()?;
        self.steps = 0;
        self.depth = 0;
        self.call(function, args)
    }

    /// Allocates the module-scope variables and fills the built-ins.
    fn prepare(&mut self) -> Result<(), InterpError> {
        if self.prepared {
            return Ok(());
        }
        self.prepared = true;
        let module = self.module;
        for &global in module.globals() {
            let ty = module.type_of(global);
            let (_, pointee) = module
                .pointer_parts(ty)
                .ok_or(InterpError::TypeMismatch {
                    what: "a pointer-typed global variable",
                })?;
            let size = type_size(module, pointee).ok_or(InterpError::TypeMismatch {
                what: "a global variable of a sized type",
            })?;
            let alloc = self.memory.allocate(size);
            let pointer = Pointer::new(alloc, 0);
            if let Some(init) = module.global(global).and_then(|entry| entry.init) {
                let value = self.value_of(0, init)?;
                self.write_value(pointer, &value, module.ty(pointee))?;
            }
            self.globals[global.index()] = Some(pointer);
        }
        self.refresh_builtins()?;
        Ok(())
    }

    /// Fills the built-in variables from the work-item state.
    fn refresh_builtins(&mut self) -> Result<(), InterpError> {
        let module = self.module;
        for decoration in &module.decorations {
            let Decor::BuiltIn { name } = &decoration.kind else {
                continue;
            };
            let Some(pointer) = self.globals[decoration.target.index()] else {
                continue;
            };
            let (values, count): ([u64; 3], usize) = match name.as_str() {
                "GlobalInvocationId" => (self.work_item.global_id, 3),
                "LocalInvocationId" => (self.work_item.local_id, 3),
                "WorkgroupId" => (self.work_item.group_id, 3),
                "GlobalOffset" => (self.work_item.global_offset, 3),
                "GlobalSize" => (self.work_item.global_size, 3),
                "NumWorkgroups" => (self.work_item.num_groups, 3),
                "WorkgroupSize" | "EnqueuedWorkgroupSize" => (self.work_item.local_size, 3),
                "GlobalLinearId" => ([self.work_item.global_linear_id, 0, 0], 1),
                "LocalInvocationIndex" => ([self.work_item.local_linear_id, 0, 0], 1),
                _ => continue,
            };
            let mut bytes = [0u8; POINTER_SIZE * 3];
            for (index, value) in values.iter().take(count).enumerate() {
                let start = index * POINTER_SIZE;
                bytes[start..start + POINTER_SIZE].copy_from_slice(&value.to_le_bytes());
            }
            self.memory
                .write(pointer.alloc, pointer.offset, &bytes[..count * POINTER_SIZE])?;
        }
        Ok(())
    }

    /// Runs a function with the given arguments.
    fn call(&mut self, function: ValueId, args: &[Value]) -> Result<Value, InterpError> {
        let module = self.module;
        let definition = module
            .function(function)
            .ok_or(InterpError::NotAFunction { id: function })?;
        let body = definition
            .body
            .as_deref()
            .ok_or(InterpError::NoBody { id: function })?;
        let params = definition.args.as_slice();
        if params.len() != args.len() {
            return Err(InterpError::ArgumentCount {
                expected: params.len(),
                found: args.len(),
            });
        }
        if self.depth >= self.depth_limit {
            return Err(InterpError::CallDepthExceeded);
        }
        let depth = self.depth;
        self.enter_frame(depth, body.len())?;
        for (index, (param, value)) in params.iter().zip(args.iter()).enumerate() {
            let bound = self
                .coerce(value, module.type_of(*param))
                .map_err(|error| match error {
                    InterpError::TypeMismatch { .. } => InterpError::ArgumentType { index },
                    other => other,
                })?;
            self.set_slot(depth, *param, bound);
        }
        self.depth += 1;
        let result = self.execute(depth, body);
        self.depth -= 1;
        result
    }

    /// Prepares the frame of a call.
    fn enter_frame(&mut self, depth: usize, blocks: usize) -> Result<(), InterpError> {
        while self.frames.len() <= depth {
            let values = self.module.values().len();
            let arena = self.memory.arena();
            let slots = (0..values)
                .map(|_| Slot {
                    stamp: 0,
                    value: Value::Void,
                })
                .collect();
            self.frames.push(Frame {
                stamp: 1,
                slots,
                marks: vec![UNVISITED; blocks],
                arena,
            });
        }
        let frame = &mut self.frames[depth];
        frame.stamp += 1;
        if frame.marks.len() < blocks {
            frame.marks.resize(blocks, UNVISITED);
        }
        frame.marks.fill(UNVISITED);
        let arena = frame.arena;
        self.memory.set_used(arena, 0)?;
        Ok(())
    }

    /// Runs the blocks of a function.
    fn execute(&mut self, depth: usize, body: &[BasicBlock]) -> Result<Value, InterpError> {
        let mut block = BlockId(0);
        let mut predecessor: Option<BlockId> = None;
        loop {
            self.steps += 1;
            if self.steps > self.step_limit {
                return Err(InterpError::StepLimitExceeded);
            }
            let insts = body
                .get(block.index())
                .ok_or(InterpError::UnknownBlock { block })?
                .insts
                .as_slice();
            self.enter_block(depth, block)?;
            for inst in insts {
                if let Op::Phi { incomings } = &inst.op {
                    self.exec_phi(depth, block, inst, incomings, predecessor)?;
                }
            }
            for inst in insts {
                if inst.op.is_terminator() {
                    break;
                }
                self.exec_inst(depth, inst)?;
            }
            let terminator = insts
                .last()
                .ok_or(InterpError::UnknownBlock { block })?;
            match &terminator.op {
                Op::Branch { target } => {
                    predecessor = Some(block);
                    block = *target;
                }
                Op::BranchConditional { cond, then, other } => {
                    let taken = self.boolean(depth, *cond)?;
                    predecessor = Some(block);
                    block = if taken { *then } else { *other };
                }
                Op::Return => return Ok(Value::Void),
                Op::ReturnValue { value } => return self.value_of(depth, *value),
                Op::Unreachable => {
                    return Err(InterpError::Unsupported {
                        what: "an unreachable instruction",
                    });
                }
                _ => {
                    return Err(InterpError::Unsupported {
                        what: "a block without a terminator",
                    });
                }
            }
        }
    }

    /// Enters a block, releasing the stack slots the block's own scope
    /// owns when the block is reached a second time.
    fn enter_block(&mut self, depth: usize, block: BlockId) -> Result<(), InterpError> {
        let arena = self.frames[depth].arena;
        let mark = self.frames[depth].marks[block.index()];
        if mark == UNVISITED {
            let used = self.memory.used(arena)?;
            self.frames[depth].marks[block.index()] = used as u64;
        } else {
            self.memory.set_used(arena, mark as usize)?;
        }
        Ok(())
    }

    /// Evaluates a `OpPhi` against the predecessor the block was
    /// entered from.
    fn exec_phi(
        &mut self,
        depth: usize,
        block: BlockId,
        inst: &Inst,
        incomings: &[(ValueId, BlockId)],
        predecessor: Option<BlockId>,
    ) -> Result<(), InterpError> {
        let result = inst.result.ok_or(InterpError::TypeMismatch {
            what: "a phi with a result",
        })?;
        let incoming = incomings
            .iter()
            .find(|(_, label)| Some(*label) == predecessor)
            .ok_or(InterpError::MissingPhiIncoming { block })?;
        let value = self.value_of(depth, incoming.0)?;
        self.set_slot(depth, result, value);
        Ok(())
    }

    /// Executes one instruction.
    fn exec_inst(&mut self, depth: usize, inst: &Inst) -> Result<(), InterpError> {
        let module = self.module;
        match &inst.op {
            Op::Variable { init } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a variable with a result",
                })?;
                let ty = inst.ty.ok_or(InterpError::TypeMismatch {
                    what: "a variable with a type",
                })?;
                let (_, pointee) = module
                    .pointer_parts(ty)
                    .ok_or(InterpError::TypeMismatch {
                        what: "a pointer-typed variable",
                    })?;
                let size = type_size(module, pointee).ok_or(InterpError::TypeMismatch {
                    what: "a variable of a sized type",
                })?;
                let arena = self.frames[depth].arena;
                let offset = self.memory.bump(arena, size)?;
                let pointer = Pointer::new(arena, offset);
                if let Some(init) = init {
                    let value = self.value_of(depth, *init)?;
                    self.write_value(pointer, &value, module.ty(pointee))?;
                }
                self.set_slot(depth, result, Value::Pointer(pointer));
            }
            Op::Load { ptr } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a load with a result",
                })?;
                let pointer = self.pointer(depth, *ptr)?;
                let pointee = module
                    .pointer_parts(module.type_of(*ptr))
                    .map(|(_, pointee)| pointee)
                    .ok_or(InterpError::TypeMismatch {
                        what: "a pointer-typed load",
                    })?;
                let value = self.read_value(pointer, module.ty(pointee))?;
                self.set_slot(depth, result, value);
            }
            Op::Store { ptr, value } => {
                let pointer = self.pointer(depth, *ptr)?;
                let pointee = module
                    .pointer_parts(module.type_of(*ptr))
                    .map(|(_, pointee)| pointee)
                    .ok_or(InterpError::TypeMismatch {
                        what: "a pointer-typed store",
                    })?;
                let stored = self.value_of(depth, *value)?;
                self.write_value(pointer, &stored, module.ty(pointee))?;
            }
            Op::AccessChain { base, indices } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "an access chain with a result",
                })?;
                let pointer = self.access_chain(depth, *base, indices)?;
                self.set_slot(depth, result, Value::Pointer(pointer));
            }
            Op::PtrAccessChain { base, indices } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a pointer access chain with a result",
                })?;
                let pointer = self.ptr_access_chain(depth, *base, indices)?;
                self.set_slot(depth, result, Value::Pointer(pointer));
            }
            Op::CopyObject { operand } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a copy with a result",
                })?;
                let value = self.value_of(depth, *operand)?;
                self.set_slot(depth, result, value);
            }
            Op::Unary { op, operand } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "an operation with a result",
                })?;
                let operand = self.value_of(depth, *operand)?;
                let value = self.unary(*op, &operand, inst.ty)?;
                self.set_slot(depth, result, value);
            }
            Op::Binary { op, lhs, rhs } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "an operation with a result",
                })?;
                let left = self.value_of(depth, *lhs)?;
                let right = self.value_of(depth, *rhs)?;
                let value = self.binary(*op, &left, &right, inst.ty)?;
                self.set_slot(depth, result, value);
            }
            Op::Compare { op, lhs, rhs } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a comparison with a result",
                })?;
                let left = self.value_of(depth, *lhs)?;
                let right = self.value_of(depth, *rhs)?;
                let operand_ty = module.type_of(*lhs);
                let value = self.compare(*op, &left, &right, Some(operand_ty))?;
                self.set_slot(depth, result, value);
            }
            Op::Select { cond, a, b } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a select with a result",
                })?;
                let value = self.select(depth, *cond, *a, *b, inst.ty)?;
                self.set_slot(depth, result, value);
            }
            Op::Convert { op, operand } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a conversion with a result",
                })?;
                let source_ty = module.type_of(*operand);
                let operand = self.value_of(depth, *operand)?;
                let value = self.convert(*op, &operand, source_ty, inst.ty)?;
                self.set_slot(depth, result, value);
            }
            Op::Call { callee, args } => {
                let mut actuals = Vec::with_capacity(args.len());
                for arg in args {
                    actuals.push(self.value_of(depth, *arg)?);
                }
                let value = self.call(*callee, &actuals)?;
                if let Some(result) = inst.result {
                    self.set_slot(depth, result, value);
                }
            }
            Op::ExtInst {
                inst: number, args, ..
            } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "an extended instruction with a result",
                })?;
                let mut actuals = Vec::with_capacity(args.len());
                for arg in args {
                    actuals.push(self.value_of(depth, *arg)?);
                }
                let value = self.ext_inst(*number, &actuals, inst.ty)?;
                self.set_slot(depth, result, value);
            }
            Op::CompositeExtract { composite, indices } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a composite extract with a result",
                })?;
                let value = self.composite_extract(depth, *composite, indices, inst.ty)?;
                self.set_slot(depth, result, value);
            }
            Op::CompositeConstruct { constituents } => {
                let result = inst.result.ok_or(InterpError::TypeMismatch {
                    what: "a composite construct with a result",
                })?;
                let mut parts = Vec::with_capacity(constituents.len());
                for constituent in constituents {
                    parts.push(self.value_of(depth, *constituent)?);
                }
                self.set_slot(depth, result, Value::Aggregate(parts));
            }
            Op::Phi { .. } => {
                // Evaluated when the block is entered.
            }
            Op::ControlBarrier { .. } | Op::MemoryBarrier { .. } => {
                // A single-threaded interpreter has nothing to order.
            }
            Op::SelectionMerge { .. } | Op::LoopMerge { .. } => {
                // Structured control flow carries no value.
            }
            Op::Branch { .. }
            | Op::BranchConditional { .. }
            | Op::Return
            | Op::ReturnValue { .. }
            | Op::Unreachable => {
                return Err(InterpError::Unsupported {
                    what: "a terminator outside a block",
                });
            }
        }
        Ok(())
    }

    /// Reads the value a handle holds.
    fn value_of(&self, depth: usize, id: ValueId) -> Result<Value, InterpError> {
        match self.module.value_kind(id) {
            ValueKind::Constant(payload) => {
                Ok(constant_value(self.module, *payload, self.module.type_of(id)))
            }
            ValueKind::Global(_) => self
                .globals
                .get(id.index())
                .and_then(|pointer| *pointer)
                .map(Value::Pointer)
                .ok_or(InterpError::UndefinedValue { id }),
            ValueKind::Argument | ValueKind::Instruction => self
                .slot(depth, id)
                .cloned()
                .ok_or(InterpError::UndefinedValue { id }),
            ValueKind::Function(_) => Err(InterpError::TypeMismatch {
                what: "a value that is not a function",
            }),
        }
    }

    /// The value a handle holds in the current frame.
    fn slot(&self, depth: usize, id: ValueId) -> Option<&Value> {
        let frame = self.frames.get(depth)?;
        let slot = frame.slots.get(id.index())?;
        if slot.stamp == frame.stamp {
            Some(&slot.value)
        } else {
            None
        }
    }

    /// Writes a value into the current frame.
    fn set_slot(&mut self, depth: usize, id: ValueId, value: Value) {
        let Some(frame) = self.frames.get_mut(depth) else {
            return;
        };
        let stamp = frame.stamp;
        if let Some(slot) = frame.slots.get_mut(id.index()) {
            slot.stamp = stamp;
            slot.value = value;
        }
    }

    /// The raw bits of an integer operand.
    fn integer(&self, depth: usize, id: ValueId) -> Result<u64, InterpError> {
        match self.module.value_kind(id) {
            ValueKind::Constant(ConstValue::Int(raw)) => Ok(*raw),
            ValueKind::Constant(ConstValue::Bool(value)) => Ok(u64::from(*value)),
            ValueKind::Argument | ValueKind::Instruction => {
                let slot = self
                    .slot(depth, id)
                    .ok_or(InterpError::UndefinedValue { id })?;
                match slot {
                    Value::Int(raw) => Ok(*raw),
                    _ => Err(InterpError::TypeMismatch {
                        what: "an integer operand",
                    }),
                }
            }
            _ => Err(InterpError::TypeMismatch {
                what: "an integer operand",
            }),
        }
    }

    /// A boolean operand.
    fn boolean(&self, depth: usize, id: ValueId) -> Result<bool, InterpError> {
        match self.module.value_kind(id) {
            ValueKind::Constant(ConstValue::Bool(value)) => Ok(*value),
            ValueKind::Argument | ValueKind::Instruction => {
                let slot = self
                    .slot(depth, id)
                    .ok_or(InterpError::UndefinedValue { id })?;
                match slot {
                    Value::Bool(value) => Ok(*value),
                    _ => Err(InterpError::TypeMismatch {
                        what: "a boolean operand",
                    }),
                }
            }
            _ => Err(InterpError::TypeMismatch {
                what: "a boolean operand",
            }),
        }
    }

    /// A pointer operand.
    fn pointer(&self, depth: usize, id: ValueId) -> Result<Pointer, InterpError> {
        match self.module.value_kind(id) {
            ValueKind::Global(_) => self
                .globals
                .get(id.index())
                .and_then(|pointer| *pointer)
                .ok_or(InterpError::UndefinedValue { id }),
            ValueKind::Argument | ValueKind::Instruction => {
                let slot = self
                    .slot(depth, id)
                    .ok_or(InterpError::UndefinedValue { id })?;
                match slot {
                    Value::Pointer(pointer) => Ok(*pointer),
                    _ => Err(InterpError::TypeMismatch {
                        what: "a pointer operand",
                    }),
                }
            }
            _ => Err(InterpError::TypeMismatch {
                what: "a pointer operand",
            }),
        }
    }

    /// The index an access chain walks with.
    fn index(&self, depth: usize, id: ValueId) -> Result<u64, InterpError> {
        let raw = self.integer(depth, id)?;
        let bits = int_bits(self.module.ty(self.module.type_of(id))).unwrap_or(64);
        let signed = sign_extend(raw, bits);
        if signed < 0 {
            return Err(InterpError::TypeMismatch {
                what: "a non-negative index",
            });
        }
        Ok(signed as u64)
    }

    /// The address an access chain computes.
    ///
    /// Every index walks one level into the aggregate the pointer points
    /// at: an array or a vector selects an element, a structure selects a
    /// field.  The stride of an element is the size of the element type.
    fn access_chain(
        &mut self,
        depth: usize,
        base: ValueId,
        indices: &[ValueId],
    ) -> Result<Pointer, InterpError> {
        let module = self.module;
        let pointer = self.pointer(depth, base)?;
        let (_, mut current) =
            module
                .pointer_parts(module.type_of(base))
                .ok_or(InterpError::TypeMismatch {
                    what: "a pointer-typed access chain base",
                })?;
        let mut offset = pointer.offset;
        for index in indices {
            let position = self.index(depth, *index)?;
            offset += self.walk(module, current, position)?;
            current = self.walked_type(module, current, position)?;
        }
        Ok(Pointer::new(pointer.alloc, offset))
    }

    /// The address a pointer access chain computes.
    ///
    /// The first index strides over the pointee: `p[i]` on a pointer to
    /// `T` addresses the `i`-th `T`.  The indices that follow walk into
    /// the pointee the way an access chain does.
    fn ptr_access_chain(
        &mut self,
        depth: usize,
        base: ValueId,
        indices: &[ValueId],
    ) -> Result<Pointer, InterpError> {
        let module = self.module;
        let pointer = self.pointer(depth, base)?;
        let (_, mut current) =
            module
                .pointer_parts(module.type_of(base))
                .ok_or(InterpError::TypeMismatch {
                    what: "a pointer-typed access chain base",
                })?;
        let (first, rest) = indices
            .split_first()
            .ok_or(InterpError::TypeMismatch {
                what: "a pointer access chain with an index",
            })?;
        let position = self.index(depth, *first)?;
        let stride = type_size(module, current).ok_or(InterpError::TypeMismatch {
            what: "a pointer to a sized type",
        })?;
        let mut offset = pointer.offset + stride * (position as usize);
        for index in rest {
            let position = self.index(depth, *index)?;
            offset += self.walk(module, current, position)?;
            current = self.walked_type(module, current, position)?;
        }
        Ok(Pointer::new(pointer.alloc, offset))
    }

    /// The byte offset an index adds inside an aggregate.
    fn walk(&self, module: &Module, current: TypeId, position: u64) -> Result<usize, InterpError> {
        match module.ty(current) {
            IrType::Array { elem, .. } | IrType::Vector { elem, .. } => {
                let size = type_size(module, *elem).ok_or(InterpError::TypeMismatch {
                    what: "an aggregate of sized elements",
                })?;
                Ok(size * (position as usize))
            }
            IrType::Struct { fields } => {
                let mut offset = 0usize;
                for (index, field) in fields.iter().enumerate() {
                    if index as u64 == position {
                        return Ok(offset);
                    }
                    offset += type_size(module, *field).ok_or(InterpError::TypeMismatch {
                        what: "a structure of sized fields",
                    })?;
                }
                Err(InterpError::TypeMismatch {
                    what: "a field index inside the structure",
                })
            }
            _ => Err(InterpError::TypeMismatch {
                what: "an aggregate to walk into",
            }),
        }
    }

    /// The type an index selects inside an aggregate.
    fn walked_type(&self, module: &Module, current: TypeId, position: u64) -> Result<TypeId, InterpError> {
        match module.ty(current) {
            IrType::Array { elem, .. } | IrType::Vector { elem, .. } => Ok(*elem),
            IrType::Struct { fields } => {
                fields
                    .get(position as usize)
                    .copied()
                    .ok_or(InterpError::TypeMismatch {
                        what: "a field index inside the structure",
                    })
            }
            _ => Err(InterpError::TypeMismatch {
                what: "an aggregate to walk into",
            }),
        }
    }

    /// A unary operation.
    ///
    /// A vector operand is handled lane by lane.
    fn unary(&self, op: UnOp, operand: &Value, result_ty: Option<TypeId>) -> Result<Value, InterpError> {
        let module = self.module;
        if let Some((element, lanes)) = result_ty.and_then(|ty| vector_parts(module, ty)) {
            let parts = self.lanes(operand, lanes)?;
            let mut out = Vec::with_capacity(lanes);
            for part in &parts {
                out.push(self.unary_scalar(op, part, Some(element))?);
            }
            return Ok(Value::Aggregate(out));
        }
        self.unary_scalar(op, operand, result_ty)
    }

    /// A unary operation over one lane.
    fn unary_scalar(
        &self,
        op: UnOp,
        operand: &Value,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        match op {
            UnOp::LogicalNot => Ok(Value::Bool(!bool_value(operand)?)),
            UnOp::FNegate => Ok(Value::Float(-float_value(operand)?)),
            UnOp::SNegate => {
                let raw = int_value(operand)?;
                let bits = self.width(module, result_ty);
                Ok(Value::Int(mask_bits(raw.wrapping_neg(), bits)))
            }
            UnOp::Not => {
                let raw = int_value(operand)?;
                let bits = self.width(module, result_ty);
                Ok(Value::Int(mask_bits(!raw, bits)))
            }
        }
    }

    /// A binary operation.
    ///
    /// A vector operand is handled lane by lane.
    fn binary(
        &self,
        op: BinOp,
        left: &Value,
        right: &Value,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        if let Some((element, lanes)) = result_ty.and_then(|ty| vector_parts(module, ty)) {
            let left = self.lanes(left, lanes)?;
            let right = self.lanes(right, lanes)?;
            let mut out = Vec::with_capacity(lanes);
            for lane in 0..lanes {
                out.push(self.binary_scalar(op, &left[lane], &right[lane], Some(element))?);
            }
            return Ok(Value::Aggregate(out));
        }
        self.binary_scalar(op, left, right, result_ty)
    }

    /// A binary operation over one lane.
    fn binary_scalar(
        &self,
        op: BinOp,
        left: &Value,
        right: &Value,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        let bits = self.width(module, result_ty);
        match op {
            BinOp::LogicalAnd => Ok(Value::Bool(bool_value(left)? && bool_value(right)?)),
            BinOp::LogicalOr => Ok(Value::Bool(bool_value(left)? || bool_value(right)?)),
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                let left = float_value(left)?;
                let right = float_value(right)?;
                let value = match op {
                    BinOp::FAdd => left + right,
                    BinOp::FSub => left - right,
                    BinOp::FMul => left * right,
                    BinOp::FDiv => left / right,
                    _ => libm::fmod(left, right),
                };
                Ok(Value::Float(self.round(module, result_ty, value)))
            }
            BinOp::SDiv | BinOp::SRem => {
                let left = sign_extend(int_value(left)?, bits);
                let right = sign_extend(int_value(right)?, bits);
                if right == 0 {
                    return Err(InterpError::DivisionByZero);
                }
                if bits >= 64 && left == i64::MIN && right == -1 {
                    return Err(InterpError::DivisionOverflow);
                }
                let value = if op == BinOp::SDiv {
                    left / right
                } else {
                    left % right
                };
                Ok(Value::Int(mask_bits(value as u64, bits)))
            }
            BinOp::UDiv | BinOp::UMod => {
                let left = mask_bits(int_value(left)?, bits);
                let right = mask_bits(int_value(right)?, bits);
                if right == 0 {
                    return Err(InterpError::DivisionByZero);
                }
                let value = if op == BinOp::UDiv {
                    left / right
                } else {
                    left % right
                };
                Ok(Value::Int(mask_bits(value, bits)))
            }
            BinOp::ShiftLeftLogical => {
                let left = mask_bits(int_value(left)?, bits);
                let count = shift_count(right, bits);
                Ok(Value::Int(mask_bits(left << count, bits)))
            }
            BinOp::ShiftRightLogical => {
                let left = mask_bits(int_value(left)?, bits);
                let count = shift_count(right, bits);
                Ok(Value::Int(mask_bits(left >> count, bits)))
            }
            BinOp::ShiftRightArithmetic => {
                let left = sign_extend(int_value(left)?, bits);
                let count = shift_count(right, bits);
                Ok(Value::Int(mask_bits((left >> count) as u64, bits)))
            }
            _ => {
                let left = mask_bits(int_value(left)?, bits);
                let right = mask_bits(int_value(right)?, bits);
                let value = match op {
                    BinOp::IAdd => left.wrapping_add(right),
                    BinOp::ISub => left.wrapping_sub(right),
                    BinOp::IMul => left.wrapping_mul(right),
                    BinOp::BitwiseAnd => left & right,
                    BinOp::BitwiseOr => left | right,
                    BinOp::BitwiseXor => left ^ right,
                    _ => {
                        return Err(InterpError::Unsupported {
                            what: "this binary operation",
                        });
                    }
                };
                Ok(Value::Int(mask_bits(value, bits)))
            }
        }
    }

    /// A comparison.
    ///
    /// A vector operand is handled lane by lane.
    fn compare(
        &self,
        op: CmpOp,
        left: &Value,
        right: &Value,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        if let Some((element, lanes)) = result_ty.and_then(|ty| vector_parts(module, ty)) {
            let left = self.lanes(left, lanes)?;
            let right = self.lanes(right, lanes)?;
            let mut out = Vec::with_capacity(lanes);
            for lane in 0..lanes {
                out.push(self.compare_scalar(op, &left[lane], &right[lane], Some(element))?);
            }
            return Ok(Value::Aggregate(out));
        }
        self.compare_scalar(op, left, right, result_ty)
    }

    /// A comparison over one lane.
    fn compare_scalar(
        &self,
        op: CmpOp,
        left: &Value,
        right: &Value,
        operand_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        let value = match op {
            CmpOp::FOrdEqual
            | CmpOp::FOrdNotEqual
            | CmpOp::FOrdLessThan
            | CmpOp::FOrdLessThanEqual
            | CmpOp::FOrdGreaterThan
            | CmpOp::FOrdGreaterThanEqual => {
                let left = float_value(left)?;
                let right = float_value(right)?;
                match op {
                    CmpOp::FOrdEqual => left == right,
                    CmpOp::FOrdNotEqual => left != right,
                    CmpOp::FOrdLessThan => left < right,
                    CmpOp::FOrdLessThanEqual => left <= right,
                    CmpOp::FOrdGreaterThan => left > right,
                    _ => left >= right,
                }
            }
            _ => {
                let bits = operand_ty
                    .and_then(|ty| int_bits(module.ty(ty)))
                    .unwrap_or(64);
                let left = mask_bits(int_value(left)?, bits);
                let right = mask_bits(int_value(right)?, bits);
                match op {
                    CmpOp::IEqual => left == right,
                    CmpOp::INotEqual => left != right,
                    CmpOp::SLessThan
                    | CmpOp::SLessThanEqual
                    | CmpOp::SGreaterThan
                    | CmpOp::SGreaterThanEqual => {
                        let left = sign_extend(left, bits);
                        let right = sign_extend(right, bits);
                        match op {
                            CmpOp::SLessThan => left < right,
                            CmpOp::SLessThanEqual => left <= right,
                            CmpOp::SGreaterThan => left > right,
                            _ => left >= right,
                        }
                    }
                    _ => match op {
                        CmpOp::ULessThan => left < right,
                        CmpOp::ULessThanEqual => left <= right,
                        CmpOp::UGreaterThan => left > right,
                        _ => left >= right,
                    },
                }
            }
        };
        Ok(Value::Bool(value))
    }

    /// A conversion.
    ///
    /// A vector operand is handled lane by lane.
    fn convert(
        &self,
        op: ConvOp,
        operand: &Value,
        source_ty: TypeId,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        if let Some((element, lanes)) = result_ty.and_then(|ty| vector_parts(module, ty)) {
            let parts = self.lanes(operand, lanes)?;
            let source_element = vector_parts(module, source_ty)
                .map(|(element, _)| element)
                .unwrap_or(source_ty);
            let mut out = Vec::with_capacity(lanes);
            for part in &parts {
                out.push(self.convert_scalar(op, part, source_element, Some(element))?);
            }
            return Ok(Value::Aggregate(out));
        }
        self.convert_scalar(op, operand, source_ty, result_ty)
    }

    /// A conversion over one lane.
    fn convert_scalar(
        &self,
        op: ConvOp,
        operand: &Value,
        source_ty: TypeId,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        let source_bits = int_bits(module.ty(source_ty)).unwrap_or(64);
        let target_bits = self.width(module, result_ty);
        match op {
            ConvOp::SConvert => {
                let raw = sign_extend(int_value(operand)?, source_bits);
                Ok(Value::Int(mask_bits(raw as u64, target_bits)))
            }
            ConvOp::UConvert => {
                let raw = mask_bits(int_value(operand)?, source_bits);
                Ok(Value::Int(mask_bits(raw, target_bits)))
            }
            ConvOp::FConvert => {
                let value = float_value(operand)?;
                Ok(Value::Float(self.round(module, result_ty, value)))
            }
            ConvOp::ConvertFToS => {
                let value = libm::trunc(float_value(operand)?);
                Ok(Value::Int(mask_bits(value as i64 as u64, target_bits)))
            }
            ConvOp::ConvertFToU => {
                let value = libm::trunc(float_value(operand)?);
                Ok(Value::Int(mask_bits(value as u64, target_bits)))
            }
            ConvOp::ConvertSToF => {
                let raw = sign_extend(int_value(operand)?, source_bits);
                Ok(Value::Float(self.round(module, result_ty, raw as f64)))
            }
            ConvOp::ConvertUToF => {
                let raw = mask_bits(int_value(operand)?, source_bits);
                Ok(Value::Float(self.round(module, result_ty, raw as f64)))
            }
            ConvOp::Bitcast => {
                if is_float(module.ty(result_ty.unwrap_or(source_ty))) {
                    let raw = mask_bits(int_value(operand)?, target_bits);
                    return Ok(Value::Float(match target_bits {
                        32 => f64::from(f32::from_bits(raw as u32)),
                        64 => f64::from_bits(raw),
                        _ => {
                            return Err(InterpError::Unsupported {
                                what: "a bitcast to this float width",
                            });
                        }
                    }));
                }
                if is_float(module.ty(source_ty)) {
                    let value = float_value(operand)?;
                    let raw = match source_bits {
                        32 => u64::from((value as f32).to_bits()),
                        64 => value.to_bits(),
                        _ => {
                            return Err(InterpError::Unsupported {
                                what: "a bitcast from this float width",
                            });
                        }
                    };
                    return Ok(Value::Int(mask_bits(raw, target_bits)));
                }
                let raw = int_value(operand)?;
                Ok(Value::Int(mask_bits(raw, target_bits)))
            }
        }
    }

    /// The lanes of a value: a vector keeps its lanes, a scalar is
    /// spread over all of them.
    fn lanes(&self, value: &Value, lanes: usize) -> Result<Vec<Value>, InterpError> {
        match value {
            Value::Aggregate(parts) => {
                if parts.len() != lanes {
                    return Err(InterpError::TypeMismatch {
                        what: "a vector with the right number of lanes",
                    });
                }
                Ok(parts.clone())
            }
            other => Ok(vec![other.clone(); lanes]),
        }
    }

    /// A select.
    ///
    /// A vector result is chosen lane by lane; a scalar condition then
    /// decides every lane.
    fn select(
        &self,
        depth: usize,
        cond: ValueId,
        a: ValueId,
        b: ValueId,
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        let left = self.value_of(depth, a)?;
        let right = self.value_of(depth, b)?;
        if let Some((element, lanes)) = result_ty.and_then(|ty| vector_parts(module, ty)) {
            let left = self.lanes(&left, lanes)?;
            let right = self.lanes(&right, lanes)?;
            let condition = self.value_of(depth, cond)?;
            let conditions = match &condition {
                Value::Aggregate(parts) => {
                    if parts.len() != lanes {
                        return Err(InterpError::TypeMismatch {
                            what: "a condition with the right number of lanes",
                        });
                    }
                    Some(parts.clone())
                }
                _ => None,
            };
            let mut out = Vec::with_capacity(lanes);
            for lane in 0..lanes {
                let taken = match &conditions {
                    Some(parts) => bool_value(&parts[lane])?,
                    None => bool_value(&condition)?,
                };
                let value = if taken { &left[lane] } else { &right[lane] };
                out.push(match value {
                    Value::Aggregate(parts) => {
                        let _ = element;
                        Value::Aggregate(parts.clone())
                    }
                    other => other.clone(),
                });
            }
            return Ok(Value::Aggregate(out));
        }
        let taken = self.boolean(depth, cond)?;
        Ok(if taken { left } else { right })
    }

    /// Extracts a component of a composite.
    fn composite_extract(
        &self,
        depth: usize,
        composite: ValueId,
        indices: &[u32],
        result_ty: Option<TypeId>,
    ) -> Result<Value, InterpError> {
        let module = self.module;
        let mut current = self.value_of(depth, composite)?;
        let mut current_ty = module.type_of(composite);
        for index in indices {
            let parts = match current {
                Value::Aggregate(parts) => parts,
                _ => {
                    return Err(InterpError::TypeMismatch {
                        what: "a composite value",
                    });
                }
            };
            let part = parts
                .get(*index as usize)
                .ok_or(InterpError::TypeMismatch {
                    what: "a component inside the composite",
                })?
                .clone();
            current = part;
            current_ty = self.component_type(module, current_ty, *index)?;
        }
        let _ = result_ty;
        Ok(current)
    }

    /// The type a component of a composite has.
    fn component_type(&self, module: &Module, ty: TypeId, index: u32) -> Result<TypeId, InterpError> {
        match module.ty(ty) {
            IrType::Array { elem, .. } | IrType::Vector { elem, .. } => Ok(*elem),
            IrType::Struct { fields } => {
                fields
                    .get(index as usize)
                    .copied()
                    .ok_or(InterpError::TypeMismatch {
                        what: "a field index inside the structure",
                    })
            }
            _ => Err(InterpError::TypeMismatch {
                what: "a composite type",
            }),
        }
    }

    /// An extended instruction.
    fn ext_inst(&self, number: u32, args: &[Value], result_ty: Option<TypeId>) -> Result<Value, InterpError> {
        let module = self.module;
        let result_ty = result_ty.ok_or(InterpError::TypeMismatch {
            what: "an extended instruction with a type",
        })?;
        let name = ocl_name(number);
        if name.is_empty() || name == "unknown" {
            return Err(InterpError::UnknownExtInst { inst: number });
        }
        let (element, lanes) = match module.ty(result_ty) {
            IrType::Vector { elem, len } => (*elem, *len as usize),
            _ => (result_ty, 1),
        };
        let mut parts = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let mut scalars = Vec::with_capacity(args.len());
            for arg in args {
                scalars.push(self.lane(arg, lane, lanes)?);
            }
            parts.push(self.ext_scalar(name, element, &scalars)?);
        }
        if lanes == 1 {
            Ok(parts.pop().unwrap_or(Value::Undef))
        } else {
            Ok(Value::Aggregate(parts))
        }
    }

    /// One lane of an extended instruction's argument.
    fn lane(&self, value: &Value, lane: usize, lanes: usize) -> Result<Value, InterpError> {
        match value {
            Value::Aggregate(parts) => parts
                .get(lane)
                .cloned()
                .ok_or(InterpError::TypeMismatch {
                    what: "an argument with enough lanes",
                }),
            other => {
                let _ = lanes;
                Ok(other.clone())
            }
        }
    }

    /// One lane of an extended instruction.
    fn ext_scalar(&self, name: &str, element: TypeId, args: &[Value]) -> Result<Value, InterpError> {
        let module = self.module;
        let name = name.strip_prefix("native_").unwrap_or(name);
        match module.ty(element) {
            IrType::Float { bits } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(match arg {
                        Value::Float(value) => *value,
                        _ => {
                            return Err(InterpError::TypeMismatch {
                                what: "a floating-point argument",
                            });
                        }
                    });
                }
                let value = self.ext_float(name, &values)?;
                Ok(Value::Float(round_float(value, *bits)))
            }
            IrType::Int { bits, signed } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(match arg {
                        Value::Int(raw) => mask_bits(*raw, *bits),
                        _ => {
                            return Err(InterpError::TypeMismatch {
                                what: "an integer argument",
                            });
                        }
                    });
                }
                let value = self.ext_int(name, *bits, *signed, &values)?;
                Ok(Value::Int(mask_bits(value, *bits)))
            }
            _ => Err(InterpError::Unsupported {
                what: "an extended instruction on this type",
            }),
        }
    }

    /// A floating-point extended instruction.
    fn ext_float(&self, name: &str, args: &[f64]) -> Result<f64, InterpError> {
        let arity = |want: usize| -> Result<(), InterpError> {
            if args.len() == want {
                Ok(())
            } else {
                Err(InterpError::TypeMismatch {
                    what: "an extended instruction with this many arguments",
                })
            }
        };
        let value = match name {
            "sqrt" => {
                arity(1)?;
                libm::sqrt(args[0])
            }
            "rsqrt" => {
                arity(1)?;
                1.0 / libm::sqrt(args[0])
            }
            "sin" => {
                arity(1)?;
                libm::sin(args[0])
            }
            "cos" => {
                arity(1)?;
                libm::cos(args[0])
            }
            "tan" => {
                arity(1)?;
                libm::tan(args[0])
            }
            "asin" => {
                arity(1)?;
                libm::asin(args[0])
            }
            "acos" => {
                arity(1)?;
                libm::acos(args[0])
            }
            "atan" => {
                arity(1)?;
                libm::atan(args[0])
            }
            "sinh" => {
                arity(1)?;
                libm::sinh(args[0])
            }
            "cosh" => {
                arity(1)?;
                libm::cosh(args[0])
            }
            "tanh" => {
                arity(1)?;
                libm::tanh(args[0])
            }
            "exp" => {
                arity(1)?;
                libm::exp(args[0])
            }
            "exp2" => {
                arity(1)?;
                libm::exp2(args[0])
            }
            "expm1" => {
                arity(1)?;
                libm::expm1(args[0])
            }
            "log" => {
                arity(1)?;
                libm::log(args[0])
            }
            "log2" => {
                arity(1)?;
                libm::log2(args[0])
            }
            "log10" => {
                arity(1)?;
                libm::log10(args[0])
            }
            "log1p" => {
                arity(1)?;
                libm::log1p(args[0])
            }
            "floor" => {
                arity(1)?;
                libm::floor(args[0])
            }
            "ceil" => {
                arity(1)?;
                libm::ceil(args[0])
            }
            "round" => {
                arity(1)?;
                libm::round(args[0])
            }
            "rint" => {
                arity(1)?;
                libm::rint(args[0])
            }
            "trunc" => {
                arity(1)?;
                libm::trunc(args[0])
            }
            "fabs" => {
                arity(1)?;
                libm::fabs(args[0])
            }
            "cbrt" => {
                arity(1)?;
                libm::cbrt(args[0])
            }
            "fract" => {
                arity(1)?;
                args[0] - libm::floor(args[0])
            }
            "sign" => {
                arity(1)?;
                if args[0] > 0.0 {
                    1.0
                } else if args[0] < 0.0 {
                    -1.0
                } else {
                    args[0]
                }
            }
            "degrees" => {
                arity(1)?;
                args[0] * (180.0 / core::f64::consts::PI)
            }
            "radians" => {
                arity(1)?;
                args[0] * (core::f64::consts::PI / 180.0)
            }
            "pow" => {
                arity(2)?;
                libm::pow(args[0], args[1])
            }
            "powr" => {
                arity(2)?;
                if args[0] < 0.0 {
                    f64::NAN
                } else {
                    libm::pow(args[0], args[1])
                }
            }
            "atan2" => {
                arity(2)?;
                libm::atan2(args[0], args[1])
            }
            "fmin" => {
                arity(2)?;
                libm::fmin(args[0], args[1])
            }
            "fmax" => {
                arity(2)?;
                libm::fmax(args[0], args[1])
            }
            "fmod" => {
                arity(2)?;
                libm::fmod(args[0], args[1])
            }
            "copysign" => {
                arity(2)?;
                libm::copysign(args[0], args[1])
            }
            "fdim" => {
                arity(2)?;
                libm::fdim(args[0], args[1])
            }
            "hypot" => {
                arity(2)?;
                libm::hypot(args[0], args[1])
            }
            "nextafter" => {
                arity(2)?;
                libm::nextafter(args[0], args[1])
            }
            "remainder" => {
                arity(2)?;
                libm::remainder(args[0], args[1])
            }
            "fma" => {
                arity(3)?;
                libm::fma(args[0], args[1], args[2])
            }
            "mad" => {
                arity(3)?;
                args[0] * args[1] + args[2]
            }
            "mix" => {
                arity(3)?;
                args[0] + (args[1] - args[0]) * args[2]
            }
            "fclamp" => {
                arity(3)?;
                libm::fmin(libm::fmax(args[0], args[1]), args[2])
            }
            "step" => {
                arity(2)?;
                if args[1] < args[0] { 0.0 } else { 1.0 }
            }
            "smoothstep" => {
                arity(3)?;
                let edge0 = args[0];
                let edge1 = args[1];
                let x = args[2];
                if edge0 == edge1 {
                    f64::NAN
                } else {
                    let t = libm::fmin(libm::fmax((x - edge0) / (edge1 - edge0), 0.0), 1.0);
                    t * t * (3.0 - 2.0 * t)
                }
            }
            _ => {
                return Err(InterpError::Unsupported {
                    what: "this extended instruction",
                });
            }
        };
        Ok(value)
    }

    /// An integer extended instruction.
    fn ext_int(&self, name: &str, bits: u8, signed: bool, args: &[u64]) -> Result<u64, InterpError> {
        let arity = |want: usize| -> Result<(), InterpError> {
            if args.len() == want {
                Ok(())
            } else {
                Err(InterpError::TypeMismatch {
                    what: "an extended instruction with this many arguments",
                })
            }
        };
        let value = match name {
            "s_abs" => {
                arity(1)?;
                let raw = sign_extend(args[0], bits);
                mask_bits(raw.unsigned_abs(), bits)
            }
            "u_abs" => {
                arity(1)?;
                args[0]
            }
            "s_min" => {
                arity(2)?;
                let left = sign_extend(args[0], bits);
                let right = sign_extend(args[1], bits);
                mask_bits(left.min(right) as u64, bits)
            }
            "s_max" => {
                arity(2)?;
                let left = sign_extend(args[0], bits);
                let right = sign_extend(args[1], bits);
                mask_bits(left.max(right) as u64, bits)
            }
            "u_min" => {
                arity(2)?;
                args[0].min(args[1])
            }
            "u_max" => {
                arity(2)?;
                args[0].max(args[1])
            }
            "s_clamp" => {
                arity(3)?;
                let value = sign_extend(args[0], bits);
                let low = sign_extend(args[1], bits);
                let high = sign_extend(args[2], bits);
                mask_bits(value.max(low).min(high) as u64, bits)
            }
            "u_clamp" => {
                arity(3)?;
                args[0].max(args[1]).min(args[2])
            }
            "popcount" => {
                arity(1)?;
                args[0].count_ones() as u64
            }
            "clz" => {
                arity(1)?;
                let width = u32::from(bits);
                if args[0] == 0 {
                    u64::from(width)
                } else {
                    u64::from(args[0].leading_zeros() - (64 - width))
                }
            }
            "ctz" => {
                arity(1)?;
                if args[0] == 0 {
                    u64::from(bits)
                } else {
                    u64::from(args[0].trailing_zeros())
                }
            }
            "rotate" => {
                arity(2)?;
                let count = (args[1] % u64::from(bits)) as u32;
                let width = u32::from(bits);
                let value = args[0] & mask_bits(u64::MAX, bits);
                let rotated = if count == 0 {
                    value
                } else {
                    (value << count) | (value >> (width - count))
                };
                mask_bits(rotated, bits)
            }
            _ => {
                let _ = signed;
                return Err(InterpError::Unsupported {
                    what: "this extended instruction",
                });
            }
        };
        Ok(value)
    }

    /// Reads a value of a type from memory.
    fn read_value(&self, pointer: Pointer, ty: &IrType) -> Result<Value, InterpError> {
        match ty {
            IrType::Int { bits, .. } => {
                let bytes = self
                    .memory
                    .read(pointer.alloc, pointer.offset, usize::from(*bits) / 8)?;
                Ok(Value::Int(mask_bits(little_endian(bytes), *bits)))
            }
            IrType::Float { bits } => {
                let bytes = self
                    .memory
                    .read(pointer.alloc, pointer.offset, usize::from(*bits) / 8)?;
                let raw = little_endian(bytes);
                Ok(Value::Float(match *bits {
                    32 => f64::from(f32::from_bits(raw as u32)),
                    64 => f64::from_bits(raw),
                    _ => {
                        return Err(InterpError::Unsupported {
                            what: "a float of this width",
                        });
                    }
                }))
            }
            IrType::Bool => {
                let bytes = self
                    .memory
                    .read(pointer.alloc, pointer.offset, 1)?;
                Ok(Value::Bool(bytes.first().copied().unwrap_or(0) != 0))
            }
            IrType::Vector { elem, len } | IrType::Array { elem, len } => {
                let element = self.module.ty(*elem);
                let size = type_size(self.module, *elem).ok_or(InterpError::TypeMismatch {
                    what: "an aggregate of sized elements",
                })?;
                let mut parts = Vec::with_capacity(*len as usize);
                for lane in 0..*len as usize {
                    let element_pointer = Pointer::new(pointer.alloc, pointer.offset + lane * size);
                    parts.push(self.read_value(element_pointer, element)?);
                }
                Ok(Value::Aggregate(parts))
            }
            IrType::Struct { fields } => {
                let mut parts = Vec::with_capacity(fields.len());
                let mut offset = pointer.offset;
                for field in fields {
                    let field_ty = self.module.ty(*field);
                    let size = type_size(self.module, *field).ok_or(InterpError::TypeMismatch {
                        what: "a structure of sized fields",
                    })?;
                    parts.push(self.read_value(Pointer::new(pointer.alloc, offset), field_ty)?);
                    offset += size;
                }
                Ok(Value::Aggregate(parts))
            }
            IrType::Pointer { .. } => {
                let bytes = self
                    .memory
                    .read(pointer.alloc, pointer.offset, POINTER_SIZE)?;
                Ok(Value::Pointer(Pointer::new(
                    AllocId(little_endian(bytes) as u32),
                    0,
                )))
            }
            IrType::Void | IrType::Function { .. } => Err(InterpError::TypeMismatch {
                what: "a value with a size",
            }),
        }
    }

    /// Writes a value of a type into memory.
    fn write_value(&mut self, pointer: Pointer, value: &Value, ty: &IrType) -> Result<(), InterpError> {
        match ty {
            IrType::Int { bits, .. } => {
                let raw = match value {
                    Value::Int(raw) => mask_bits(*raw, *bits),
                    Value::Bool(flag) => u64::from(*flag),
                    Value::Undef => 0,
                    _ => {
                        return Err(InterpError::TypeMismatch {
                            what: "an integer value",
                        });
                    }
                };
                let bytes = raw.to_le_bytes();
                let size = usize::from(*bits) / 8;
                self.memory
                    .write(pointer.alloc, pointer.offset, &bytes[..size])?;
            }
            IrType::Float { bits } => {
                let number = match value {
                    Value::Float(number) => *number,
                    Value::Undef => 0.0,
                    _ => {
                        return Err(InterpError::TypeMismatch {
                            what: "a floating-point value",
                        });
                    }
                };
                let mut bytes = [0u8; POINTER_SIZE];
                match *bits {
                    32 => bytes[..4].copy_from_slice(&(number as f32).to_bits().to_le_bytes()),
                    64 => bytes.copy_from_slice(&number.to_bits().to_le_bytes()),
                    _ => {
                        return Err(InterpError::Unsupported {
                            what: "a float of this width",
                        });
                    }
                }
                let size = usize::from(*bits) / 8;
                self.memory
                    .write(pointer.alloc, pointer.offset, &bytes[..size])?;
            }
            IrType::Bool => {
                let flag = match value {
                    Value::Bool(flag) => *flag,
                    Value::Int(raw) => *raw != 0,
                    Value::Undef => false,
                    _ => {
                        return Err(InterpError::TypeMismatch {
                            what: "a boolean value",
                        });
                    }
                };
                self.memory
                    .write(pointer.alloc, pointer.offset, &[u8::from(flag)])?;
            }
            IrType::Vector { elem, len } | IrType::Array { elem, len } => {
                let parts = match value {
                    Value::Aggregate(parts) => parts,
                    _ => {
                        return Err(InterpError::TypeMismatch {
                            what: "an aggregate value",
                        });
                    }
                };
                let element = self.module.ty(*elem);
                let size = type_size(self.module, *elem).ok_or(InterpError::TypeMismatch {
                    what: "an aggregate of sized elements",
                })?;
                if parts.len() != *len as usize {
                    return Err(InterpError::TypeMismatch {
                        what: "an aggregate with the right number of elements",
                    });
                }
                for (lane, part) in parts.iter().enumerate() {
                    let element_pointer = Pointer::new(pointer.alloc, pointer.offset + lane * size);
                    self.write_value(element_pointer, part, element)?;
                }
            }
            IrType::Struct { fields } => {
                let parts = match value {
                    Value::Aggregate(parts) => parts,
                    _ => {
                        return Err(InterpError::TypeMismatch {
                            what: "a structure value",
                        });
                    }
                };
                if parts.len() != fields.len() {
                    return Err(InterpError::TypeMismatch {
                        what: "a structure with the right number of fields",
                    });
                }
                let mut offset = pointer.offset;
                for (field, part) in fields.iter().zip(parts.iter()) {
                    let field_ty = self.module.ty(*field);
                    let size = type_size(self.module, *field).ok_or(InterpError::TypeMismatch {
                        what: "a structure of sized fields",
                    })?;
                    self.write_value(Pointer::new(pointer.alloc, offset), part, field_ty)?;
                    offset += size;
                }
            }
            IrType::Pointer { .. } => {
                let address = match value {
                    Value::Pointer(pointer) => pointer.alloc.0 as u64,
                    _ => {
                        return Err(InterpError::TypeMismatch {
                            what: "a pointer value",
                        });
                    }
                };
                self.memory
                    .write(pointer.alloc, pointer.offset, &address.to_le_bytes())?;
            }
            IrType::Void | IrType::Function { .. } => {
                return Err(InterpError::TypeMismatch {
                    what: "a value with a size",
                });
            }
        }
        Ok(())
    }

    /// The width of an integer type, in bits.
    fn width(&self, module: &Module, ty: Option<TypeId>) -> u8 {
        ty.and_then(|ty| int_bits(module.ty(ty)))
            .unwrap_or(64)
    }

    /// Rounds a floating-point value to the width of a type.
    fn round(&self, module: &Module, ty: Option<TypeId>, value: f64) -> f64 {
        match ty.map(|ty| module.ty(ty)) {
            Some(IrType::Float { bits }) => round_float(value, *bits),
            _ => value,
        }
    }

    /// Converts an argument to the type of its parameter.
    fn coerce(&self, value: &Value, ty: TypeId) -> Result<Value, InterpError> {
        let module = self.module;
        match module.ty(ty) {
            IrType::Bool => match value {
                Value::Bool(flag) => Ok(Value::Bool(*flag)),
                _ => Err(InterpError::TypeMismatch {
                    what: "a boolean argument",
                }),
            },
            IrType::Int { bits, .. } => match value {
                Value::Int(raw) => Ok(Value::Int(mask_bits(*raw, *bits))),
                Value::Bool(flag) => Ok(Value::Int(u64::from(*flag))),
                _ => Err(InterpError::TypeMismatch {
                    what: "an integer argument",
                }),
            },
            IrType::Float { bits } => match value {
                Value::Float(number) => Ok(Value::Float(round_float(*number, *bits))),
                Value::Int(raw) => Ok(Value::Float(round_float(*raw as f64, *bits))),
                _ => Err(InterpError::TypeMismatch {
                    what: "a floating-point argument",
                }),
            },
            IrType::Pointer { .. } => match value {
                Value::Pointer(pointer) => Ok(Value::Pointer(*pointer)),
                _ => Err(InterpError::TypeMismatch {
                    what: "a pointer argument",
                }),
            },
            IrType::Vector { .. } | IrType::Array { .. } | IrType::Struct { .. } => match value {
                Value::Aggregate(parts) => Ok(Value::Aggregate(parts.clone())),
                _ => Err(InterpError::TypeMismatch {
                    what: "an aggregate argument",
                }),
            },
            IrType::Void | IrType::Function { .. } => Err(InterpError::TypeMismatch {
                what: "an argument of this type",
            }),
        }
    }
}

/// The raw bits of an integer value.
fn int_value(value: &Value) -> Result<u64, InterpError> {
    match value {
        Value::Int(raw) => Ok(*raw),
        Value::Bool(flag) => Ok(u64::from(*flag)),
        _ => Err(InterpError::TypeMismatch {
            what: "an integer operand",
        }),
    }
}

/// The number of a floating-point value.
fn float_value(value: &Value) -> Result<f64, InterpError> {
    match value {
        Value::Float(number) => Ok(*number),
        _ => Err(InterpError::TypeMismatch {
            what: "a floating-point operand",
        }),
    }
}

/// The truth of a boolean value.
fn bool_value(value: &Value) -> Result<bool, InterpError> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        _ => Err(InterpError::TypeMismatch {
            what: "a boolean operand",
        }),
    }
}

/// The element type and the lane count of a vector type.
fn vector_parts(module: &Module, ty: TypeId) -> Option<(TypeId, usize)> {
    match module.ty(ty) {
        IrType::Vector { elem, len } => Some((*elem, *len as usize)),
        _ => None,
    }
}

/// The number of places a shift moves, taken modulo the width.
fn shift_count(value: &Value, bits: u8) -> u32 {
    let raw = int_value(value).unwrap_or(0);
    let count = u32::try_from(raw).unwrap_or(u32::MAX);
    let modulus = u32::from(bits.saturating_sub(1).max(1));
    count & modulus
}

/// The size of a type in bytes, `None` when the type has none.
fn type_size(module: &Module, ty: TypeId) -> Option<usize> {
    match module.ty(ty) {
        IrType::Void => Some(0),
        IrType::Bool => Some(1),
        IrType::Int { bits, .. } => Some(usize::from(*bits) / 8),
        IrType::Float { bits } => Some(usize::from(*bits) / 8),
        IrType::Vector { elem, len } => type_size(module, *elem)?.checked_mul(*len as usize),
        IrType::Array { elem, len } => type_size(module, *elem)?.checked_mul(*len as usize),
        IrType::Struct { fields } => {
            let mut total = 0usize;
            for field in fields {
                total = total.checked_add(type_size(module, *field)?)?;
            }
            Some(total)
        }
        IrType::Pointer { .. } => Some(POINTER_SIZE),
        IrType::Function { .. } => None,
    }
}

/// The raw bits of a little-endian byte string.
fn little_endian(bytes: &[u8]) -> u64 {
    let mut raw = 0u64;
    for (index, byte) in bytes.iter().enumerate() {
        raw |= u64::from(*byte) << (8 * index);
    }
    raw
}

/// Keeps the low `bits` bits of a value.
#[inline]
fn mask_bits(value: u64, bits: u8) -> u64 {
    if bits >= 64 {
        value
    } else {
        value & ((1u64 << bits) - 1)
    }
}

/// Sign-extends the low `bits` bits of a value.
#[inline]
fn sign_extend(value: u64, bits: u8) -> i64 {
    if bits >= 64 {
        value as i64
    } else {
        let shift = 64 - u32::from(bits);
        ((value << shift) as i64) >> shift
    }
}

/// Rounds a floating-point value to the width of a float type.
#[inline]
fn round_float(value: f64, bits: u16) -> f64 {
    match bits {
        32 => f64::from(value as f32),
        64 => value,
        _ => value,
    }
}

/// The value a constant carries.
fn constant_value(module: &Module, payload: ConstValue, ty: TypeId) -> Value {
    match payload {
        ConstValue::Int(raw) => Value::Int(mask_bits(raw, int_bits(module.ty(ty)).unwrap_or(64))),
        ConstValue::Float32(bits) => Value::Float(f64::from(f32::from_bits(bits))),
        ConstValue::Float64(bits) => Value::Float(f64::from_bits(bits)),
        ConstValue::Bool(flag) => Value::Bool(flag),
        ConstValue::Undef => Value::Undef,
        ConstValue::Null => match module.ty(ty) {
            IrType::Pointer { .. } => Value::Pointer(Pointer::new(AllocId(u32::MAX), 0)),
            _ => Value::Undef,
        },
    }
}

/// The width of an integer type, in bits, `None` for every other type.
fn int_bits(ty: &IrType) -> Option<u8> {
    match ty {
        IrType::Int { bits, .. } => Some(*bits),
        _ => None,
    }
}

/// True for a floating-point type.
fn is_float(ty: &IrType) -> bool {
    matches!(ty, IrType::Float { .. })
}
