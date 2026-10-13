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

//! Reference-counted objects behind the `cl_context`/`cl_command_queue`/
//! `cl_mem`/`cl_program`/`cl_kernel`/`cl_event` handles.
//!
//! # Handle discipline
//!
//! Every object starts with an [`ObjectHeader`]: a magic tag plus an atomic
//! reference count. A C handle is the object's address; validation re-reads
//! the tag, which turns "freed object handed back" and "context handle
//! passed where a queue belongs" into `CL_INVALID_*` returns instead of
//! type confusion (while the memory still maps). `clRetain*` bumps the
//! count, `clRelease*` drops it, and the object that reaches zero frees
//! itself through [`release_ref`], running the `Drop` impl that releases
//! whatever it owns — a queue its context, a kernel its program and its
//! buffer arguments, an event its context and queue.
//!
//! The C contract applies: handles must come from this driver (or be
//! null). No handle table exists — an OS-less image has no process to
//! protect and a table lookup on every enqueue would cost more than the
//! tag check.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void};
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::exec::{ArgBlock, ARG_SLOT, MAX_ARGS};
use crate::registry::{ArgKind, KernelDef};
use crate::sys::{self, MemHandle};

/// Magic tag of a [`Context`].
const MAGIC_CONTEXT: u32 = 0x4354_5831;
/// Magic tag of a [`Queue`].
const MAGIC_QUEUE: u32 = 0x5155_4531;
/// Magic tag of a [`Mem`] buffer.
const MAGIC_MEM: u32 = 0x4D45_4D31;
/// Magic tag of a [`Program`].
const MAGIC_PROGRAM: u32 = 0x5052_4F31;
/// Magic tag of a [`Kernel`].
const MAGIC_KERNEL: u32 = 0x4B52_4E31;
/// Magic tag of an [`Event`].
const MAGIC_EVENT: u32 = 0x4556_5431;
/// Magic tag of a [`DeviceObject`].
const MAGIC_DEVICE: u32 = 0x4445_5631;

/// The shared prefix of every object a handle can point at.
pub(crate) struct ObjectHeader {
    magic: u32,
    refs: AtomicU32,
}

impl ObjectHeader {
    /// A fresh header with the reference the creating call owns.
    pub(crate) const fn new(magic: u32) -> Self {
        Self {
            magic,
            refs: AtomicU32::new(1),
        }
    }

    /// Adds one reference (`clRetain*`).
    pub(crate) fn retain(&self) {
        self.refs.fetch_add(1, Ordering::Relaxed);
    }

    /// Drops one reference (`clRelease*`); returns `true` when the object
    /// reached zero and the caller must free it.
    pub(crate) fn release(&self) -> bool {
        self.refs.fetch_sub(1, Ordering::AcqRel) == 1
    }

    /// Current count (`CL_*_REFERENCE_COUNT` queries).
    pub(crate) fn refs(&self) -> u32 {
        self.refs.load(Ordering::Relaxed)
    }

    /// Magic tag validation for the accessors below.
    fn matches(&self, magic: u32) -> bool {
        self.magic == magic
    }
}

/// Objects that participate in reference counting — everything a
/// `clRetain*`/`clRelease*` pair can target.
pub(crate) trait RefCounted {
    /// This object's header.
    fn header(&self) -> &ObjectHeader;
}

/// Releases one reference, freeing the object when the count reaches zero.
///
/// # Safety
///
/// `object` must be a `'static` object created by this crate (a leaked
/// box), and every other reference to it must have been obtained through
/// the handle API.
pub(crate) fn release_ref<T: RefCounted + 'static>(object: &'static T) {
    if object.header().release() {
        // SAFETY: the count reached zero, so no handle can validate to this
        // object any more and this was the last live reference.
        unsafe { drop(Box::from_raw(core::ptr::from_ref(object).cast_mut())) };
    }
}

/// Validates a typed handle and borrows the object behind it.
///
/// The null check comes first so a well-behaved caller's "no object" path
/// costs one compare; the tag check then catches cross-type reuse of a
/// stale address.
macro_rules! accessor {
    ($name:ident, $handle:ident, $object:ty, $magic:expr, $error:path) => {
        pub(crate) fn $name(handle: crate::sys::$handle) -> Result<&'static $object> {
            let pointer = handle.as_raw().cast::<$object>();
            if pointer.is_null() {
                return Err($error);
            }
            // SAFETY: the C API contract — `handle` was returned by this
            // driver and has not been forged; null is excluded above. The
            // magic tag guards against a freed-then-reused address still
            // being readable.
            let object = unsafe { &*pointer };
            if !object.header.matches($magic) {
                return Err($error);
            }
            Ok(object)
        }
    };
}

accessor!(context, ContextHandle, Context, MAGIC_CONTEXT, Error::InvalidContext);
accessor!(queue, CommandQueueHandle, Queue, MAGIC_QUEUE, Error::InvalidQueue);
accessor!(mem, MemHandle, Mem, MAGIC_MEM, Error::InvalidMemObject);
accessor!(
    program,
    ProgramHandle,
    Program,
    MAGIC_PROGRAM,
    Error::InvalidProgram
);
accessor!(kernel, KernelHandle, Kernel, MAGIC_KERNEL, Error::InvalidKernel);
accessor!(event, EventHandle, Event, MAGIC_EVENT, Error::InvalidEvent);
accessor!(
    device_object,
    DeviceHandle,
    DeviceObject,
    MAGIC_DEVICE,
    Error::InvalidDevice
);

/// A device of the platform, boxed so its address round-trips through a
/// `cl_device_id` (a `&dyn Device` is a fat pointer and would not).
pub struct DeviceObject {
    header: ObjectHeader,
    device: &'static dyn Device,
}

impl RefCounted for DeviceObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl DeviceObject {
    /// Leaks a box around `device`, giving it a stable handle address.
    pub(crate) fn leak(device: &'static dyn Device) -> &'static Self {
        Box::leak(Box::new(Self {
            header: ObjectHeader::new(MAGIC_DEVICE),
            device,
        }))
    }

    /// The back end this handle selects.
    pub(crate) fn device(&self) -> &'static dyn Device {
        self.device
    }
}

/// An OpenCL context: the set of devices its objects were created for.
pub struct Context {
    header: ObjectHeader,
    devices: Vec<&'static DeviceObject>,
}

impl RefCounted for Context {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl Context {
    /// Creates a context over `devices` (the reference count starts at the
    /// one the creating call owns).
    pub(crate) fn new(devices: Vec<&'static DeviceObject>) -> Box<Self> {
        Box::new(Self {
            header: ObjectHeader::new(MAGIC_CONTEXT),
            devices,
        })
    }

    /// The devices of this context (`CL_CONTEXT_DEVICES`).
    pub fn devices(&self) -> &[&'static DeviceObject] {
        &self.devices
    }

    /// Validates that `device` belongs to this context — `clCreateQueue`
    /// and friends require it.
    pub(crate) fn has_device(&self, device: &DeviceObject) -> bool {
        self.devices.iter().any(|entry| core::ptr::eq(*entry, device))
    }

    /// Number of devices (`CL_CONTEXT_NUM_DEVICES`).
    pub(crate) fn num_devices(&self) -> usize {
        self.devices.len()
    }
}

/// An in-order command queue bound to one device of its context.
///
/// The embedded profile has one execution stream per queue; every enqueue
/// validates, executes (or dispatches) and retires its command before
/// returning, which is what lets [`crate::arch`] wait loops be so short.
pub struct Queue {
    header: ObjectHeader,
    context: *mut Context,
    device: &'static dyn Device,
}

impl RefCounted for Queue {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl Queue {
    /// Creates a queue, retaining `context` for the queue's lifetime.
    pub(crate) fn new(context: &'static Context, device: &'static dyn Device) -> Box<Self> {
        context.header.retain();
        Box::new(Self {
            header: ObjectHeader::new(MAGIC_QUEUE),
            context: core::ptr::from_ref(context).cast_mut(),
            device,
        })
    }

    /// The context of this queue (`CL_QUEUE_CONTEXT`).
    pub(crate) fn context(&self) -> &'static Context {
        // SAFETY: the queue retained its context at creation and releases
        // it in `Drop`, so the pointer is live for every borrow of `self`.
        unsafe { &*self.context }
    }

    /// The device this queue dispatches to (`CL_QUEUE_DEVICE`).
    pub fn device(&self) -> &'static dyn Device {
        self.device
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // SAFETY: `context` was retained in `new` exactly once.
        let context = unsafe { &*self.context };
        release_ref(context);
    }
}

/// Backing storage of a buffer: driver-allocated, or the application's own
/// pointer when `CL_MEM_USE_HOST_PTR` asked for it.
pub(crate) enum MemData {
    /// Storage the buffer owns; freed with the buffer.
    Owned(Box<[u8]>),
    /// Application memory the buffer borrows (single address space, so
    /// "device memory" and "host memory" are the same bytes).
    Host(*mut u8),
}

/// A buffer object (`cl_mem`) — the only memory type the embedded profile
/// needs (no images, no sub-buffers, no pipes).
///
/// On a machine without an MMU the storage is directly addressable by
/// every device, so reads, writes and copies are `memcpy`s and kernels
/// receive the storage pointer itself as an argument.
pub struct Mem {
    header: ObjectHeader,
    context: *mut Context,
    flags: u64,
    size: usize,
    data: MemData,
}

// SAFETY: buffer storage is shared between host code and device kernels by
// design (single address space); the in-order queue serializes command
// execution, and interior access goes through raw pointers whose bounds are
// checked at every enqueue.
unsafe impl Sync for Mem {}

impl RefCounted for Mem {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl Mem {
    /// Creates a buffer per the `clCreateBuffer` flag rules, retaining
    /// `context`.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBufferSize`] — `size` is zero.
    /// * [`Error::InvalidArgument`] — unknown flag bits, contradictory
    ///   access flags, or `USE` together with `COPY`.
    /// * [`Error::InvalidHostPtr`] — `USE`/`COPY` without a pointer.
    /// * [`Error::OutOfMemory`] — allocation failure.
    pub(crate) fn create(
        context: &'static Context,
        flags: u64,
        size: usize,
        host_ptr: *mut c_void,
    ) -> Result<Box<Self>> {
        if size == 0 {
            return Err(Error::InvalidBufferSize);
        }
        const ACCESS: u64 = sys::MEM_READ_ONLY | sys::MEM_WRITE_ONLY;
        const KNOWN: u64 =
            sys::MEM_READ_WRITE | ACCESS | sys::MEM_USE_HOST_PTR | sys::MEM_ALLOC_HOST_PTR | sys::MEM_COPY_HOST_PTR;
        if flags & !KNOWN != 0 {
            return Err(Error::InvalidArgument);
        }
        if flags & ACCESS == ACCESS {
            return Err(Error::InvalidArgument);
        }
        let wants_use = flags & sys::MEM_USE_HOST_PTR != 0;
        let wants_copy = flags & sys::MEM_COPY_HOST_PTR != 0;
        if wants_use && wants_copy {
            return Err(Error::InvalidArgument);
        }
        if (wants_use || wants_copy) && host_ptr.is_null() {
            return Err(Error::InvalidHostPtr);
        }
        let data = if wants_use {
            MemData::Host(host_ptr.cast::<u8>())
        } else {
            let mut storage = Vec::new();
            storage
                .try_reserve_exact(size)
                .map_err(|_| Error::OutOfMemory)?;
            storage.resize(size, 0);
            let mut storage = storage.into_boxed_slice();
            if wants_copy {
                // SAFETY: the caller supplied a non-null pointer per the
                // flag contract, readable for `size` bytes.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        host_ptr.cast::<u8>(),
                        storage.as_mut_ptr(),
                        size,
                    );
                }
            }
            MemData::Owned(storage)
        };
        context.header.retain();
        Ok(Box::new(Self {
            header: ObjectHeader::new(MAGIC_MEM),
            context: core::ptr::from_ref(context).cast_mut(),
            flags,
            size,
            data,
        }))
    }

    /// Base address of the storage — what kernels receive as the argument.
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        match &self.data {
            MemData::Owned(storage) => storage.as_ptr().cast_mut(),
            MemData::Host(pointer) => *pointer,
        }
    }

    /// Size in bytes (`CL_MEM_SIZE`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Creation flags (`CL_MEM_FLAGS`).
    #[must_use]
    pub fn flags(&self) -> u64 {
        self.flags
    }

    /// The context of this buffer (`CL_MEM_CONTEXT`).
    pub(crate) fn context(&self) -> &'static Context {
        // SAFETY: retained at creation, released in `Drop`.
        unsafe { &*self.context }
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        // SAFETY: retained exactly once in `create`.
        let context = unsafe { &*self.context };
        release_ref(context);
    }
}

/// A program bound to its kernel definitions — the static registry for
/// `clCreateProgramWithBuiltInKernels`/registry-key binaries, or the entry
/// points parsed out of a SPIR-V binary.
///
/// Built-in (and "binary") programs resolve their names at creation, so a
/// program is always in the built state — `clBuildProgram` only re-validates
/// and records the options the caller passed.
pub struct Program {
    header: ObjectHeader,
    context: *mut Context,
    defs: Vec<KernelDef>,
    options: spin::Mutex<String>,
}

impl RefCounted for Program {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl Program {
    /// Creates a program over already-resolved definitions, retaining
    /// `context`.
    pub(crate) fn new(context: &'static Context, defs: Vec<KernelDef>) -> Box<Self> {
        context.header.retain();
        Box::new(Self {
            header: ObjectHeader::new(MAGIC_PROGRAM),
            context: core::ptr::from_ref(context).cast_mut(),
            defs,
            options: spin::Mutex::new(String::new()),
        })
    }

    /// Resolves a kernel name within this program (`clCreateKernel`).
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&KernelDef> {
        self.defs.iter().find(|def| def.name == name)
    }

    /// The definitions of this program (`CL_PROGRAM_KERNEL_NAMES` order).
    pub fn defs(&self) -> &[KernelDef] {
        &self.defs
    }

    /// Records the options of the last `clBuildProgram`.
    pub(crate) fn set_options(&self, options: String) {
        *self.options.lock() = options;
    }

    /// The recorded build options (`CL_PROGRAM_BUILD_OPTIONS`).
    pub(crate) fn options(&self) -> String {
        self.options.lock().clone()
    }

    /// The context of this program (`CL_PROGRAM_CONTEXT`).
    pub(crate) fn context(&self) -> &'static Context {
        // SAFETY: retained at creation, released in `Drop`.
        unsafe { &*self.context }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: retained exactly once in `new`.
        let context = unsafe { &*self.context };
        release_ref(context);
    }
}

/// One bound argument of a [`Kernel`].
pub(crate) enum ArgValue {
    /// Not set since the last reset — enqueue reports
    /// `CL_INVALID_KERNEL_ARGS`.
    Unset,
    /// A buffer: the kernel retains the object so the storage cannot die
    /// between `clSetKernelArg` and the enqueue that uses it.
    Buffer(*mut Mem),
    /// A scalar copied verbatim out of the caller's `arg_value`.
    Scalar { bytes: [u8; ARG_SLOT], len: usize },
}

/// A kernel instance: its program's definition (cloned, so a SPIR-V
/// entry's owned name/kinds survive independently of the definition list)
/// plus its bound arguments.
pub struct Kernel {
    header: ObjectHeader,
    program: *mut Program,
    def: KernelDef,
    args: [ArgValue; MAX_ARGS],
}

impl RefCounted for Kernel {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl Kernel {
    /// Creates a kernel for `def`, retaining `program`.
    pub(crate) fn new(program: &'static Program, def: &KernelDef) -> Box<Self> {
        program.header.retain();
        Box::new(Self {
            header: ObjectHeader::new(MAGIC_KERNEL),
            program: core::ptr::from_ref(program).cast_mut(),
            def: def.clone(),
            args: core::array::from_fn(|_| ArgValue::Unset),
        })
    }

    /// The definition this kernel runs.
    pub fn def(&self) -> &KernelDef {
        &self.def
    }

    /// The program of this kernel (`CL_KERNEL_PROGRAM`).
    pub(crate) fn program(&self) -> &'static Program {
        // SAFETY: retained at creation, released in `Drop`.
        unsafe { &*self.program }
    }

    /// Binds argument `index` per `clSetKernelArg`.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgIndex`] — past the declared arguments.
    /// * [`Error::InvalidArgSize`] — `size` mismatches the declaration.
    /// * [`Error::InvalidArgValue`] — null value (local-memory arguments
    ///   are not supported).
    /// * [`Error::InvalidMemObject`] — buffer value is not a live object.
    pub(crate) fn set_arg(
        &mut self,
        index: usize,
        size: usize,
        value: *const c_void,
    ) -> Result<()> {
        let expected = self.def.arg_size(index).ok_or(Error::InvalidArgIndex)?;
        if size != expected {
            return Err(Error::InvalidArgSize);
        }
        if size > ARG_SLOT {
            // A by-value argument wider than one marshalling slot (a large
            // struct passed by value, typically from a SPIR-V module) has
            // nowhere to live in the argument block.
            return Err(Error::InvalidArgSize);
        }
        if value.is_null() {
            return Err(Error::InvalidArgValue);
        }
        match self.def.kinds[index] {
            ArgKind::Buffer => {
                // SAFETY: `size` equals the handle size and the caller
                // contractually points at that many readable bytes.
                let handle = unsafe { *value.cast::<MemHandle>() };
                let buffer = crate::objects::mem(handle)?;
                buffer.header.retain();
                self.replace_arg(index, ArgValue::Buffer(core::ptr::from_ref(buffer).cast_mut()));
            }
            ArgKind::Scalar(_) => {
                let mut bytes = [0u8; ARG_SLOT];
                // SAFETY: `size <= ARG_SLOT` (checked against the
                // declaration) and the caller provides `size` readable bytes.
                unsafe {
                    core::ptr::copy_nonoverlapping(value.cast::<u8>(), bytes.as_mut_ptr(), size);
                }
                self.replace_arg(index, ArgValue::Scalar { bytes, len: size });
            }
        }
        Ok(())
    }

    /// Swaps in a new argument value, releasing what the slot held.
    fn replace_arg(&mut self, index: usize, value: ArgValue) {
        let old = core::mem::replace(&mut self.args[index], value);
        if let ArgValue::Buffer(buffer) = old {
            // SAFETY: the buffer pointer was retained when it entered the
            // slot; this is that reference leaving.
            release_ref(unsafe { &*buffer });
        }
    }

    /// Marshals the bound arguments into the launch block.
    ///
    /// # Errors
    ///
    /// [`Error::MissingKernelArg`] when any declared argument is unset.
    pub(crate) fn marshal(&self) -> Result<ArgBlock> {
        let mut block = ArgBlock::new();
        for index in 0..self.def.kinds.len() {
            match &self.args[index] {
                ArgValue::Unset => return Err(Error::MissingKernelArg),
                ArgValue::Buffer(buffer) => {
                    // SAFETY: retained while bound (see `set_arg`), so the
                    // object is live for the duration of this borrow.
                    let buffer = unsafe { &**buffer };
                    block.set_ptr(index, buffer.as_ptr().cast());
                }
                ArgValue::Scalar { bytes, len } => {
                    block.set_scalar(index, &bytes[..*len])?;
                }
            }
        }
        Ok(block)
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        for slot in &mut self.args {
            if let ArgValue::Buffer(buffer) = slot {
                // SAFETY: each buffer slot owns one retained reference.
                release_ref(unsafe { &**buffer });
            }
        }
        // SAFETY: retained exactly once in `new`.
        let program = unsafe { &*self.program };
        release_ref(program);
    }
}

/// Execution state of an [`Event`], the `CL_COMPLETE..CL_QUEUED` values
/// plus negative error codes as `CL_*` defines them.
pub struct Event {
    header: ObjectHeader,
    context: *mut Context,
    queue: *mut Queue,
    device: Option<&'static dyn Device>,
    command_type: u32,
    status: AtomicI32,
}

// SAFETY: the two raw pointers are only ever read to reach reference-counted
// objects retained for the event's lifetime (drops release them exactly
// once), all reference-count updates go through `AtomicU32`, and the status
// word is atomic — so sharing `&Event` across cores (the device in-flight
// lists do this) cannot race on any field.
unsafe impl Sync for Event {}

impl RefCounted for Event {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }
}

impl Event {
    /// Creates an event in `CL_QUEUED` state, retaining `context` and
    /// `queue`.
    pub(crate) fn new(
        command_type: u32,
        context: &'static Context,
        queue: &'static Queue,
        device: Option<&'static dyn Device>,
    ) -> Box<Self> {
        context.header.retain();
        queue.header.retain();
        Box::new(Self {
            header: ObjectHeader::new(MAGIC_EVENT),
            context: core::ptr::from_ref(context).cast_mut(),
            queue: core::ptr::from_ref(queue).cast_mut(),
            device,
            command_type,
            status: AtomicI32::new(sys::QUEUED),
        })
    }

    /// The tracked command type (`CL_EVENT_COMMAND_TYPE`).
    pub fn command_type(&self) -> u32 {
        self.command_type
    }

    /// Current execution status (`CL_EVENT_COMMAND_EXECUTION_STATUS`).
    #[must_use]
    pub fn status(&self) -> i32 {
        self.status.load(Ordering::Acquire)
    }

    /// Whether the command retired — completed successfully (0) or with an
    /// error (negative).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.status() <= sys::COMPLETE
    }

    /// Moves the event to `status` (`CL_QUEUED` → `CL_SUBMITTED` →
    /// `CL_RUNNING` → `CL_COMPLETE` or an error).
    pub(crate) fn set_status(&self, status: i32) {
        self.status.store(status, Ordering::Release);
    }

    /// Completes the event with `errcode` (0 or a negative `CL_*` code).
    pub(crate) fn finish(&self, errcode: i32) {
        self.set_status(errcode);
    }

    /// The context of the event's command (`CL_EVENT_CONTEXT`).
    pub(crate) fn context(&self) -> &'static Context {
        // SAFETY: retained at creation, released in `Drop`.
        unsafe { &*self.context }
    }

    /// The queue of the event's command (`CL_EVENT_COMMAND_QUEUE`).
    pub(crate) fn queue(&self) -> &'static Queue {
        // SAFETY: retained at creation, released in `Drop`.
        unsafe { &*self.queue }
    }

    /// Waits until the command retires, polling the device's progress
    /// (which drives AMP/accelerator descriptors to completion) and
    /// sleeping between polls.
    ///
    /// # Errors
    ///
    /// Propagates a device progress failure; a command that completed with
    /// an error is *not* an error here — callers read [`Event::status`]
    /// afterwards (as `clWaitForEvents` specifies).
    pub fn wait(&self) -> Result<()> {
        while !self.is_complete() {
            if let Some(device) = self.device {
                device.progress()?;
            }
            codevar_target::wfe();
        }
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: one retain each, taken in `new`.
        release_ref(unsafe { &*self.context });
        release_ref(unsafe { &*self.queue });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceInfo;
    use crate::exec::{vector_add, SpirvBinding};
    use crate::registry::{KernelModule, Registry};

    use alloc::borrow::Cow;

    struct TestDevice;
    impl Device for TestDevice {
        fn info(&self) -> &DeviceInfo {
            static INFO: DeviceInfo = DeviceInfo::new(
                "test",
                "codevar",
                sys::DEVICE_TYPE_CPU,
                1,
                64,
                1 << 20,
                4096,
                256,
            );
            &INFO
        }
        fn submit(&self, _launch: &crate::exec::Launch<'_>, event: &'static Event) -> Result<()> {
            event.finish(sys::COMPLETE);
            release_ref(event);
            Ok(())
        }
        fn progress(&self) -> Result<()> {
            Ok(())
        }
        fn inflight(&self) -> usize {
            0
        }
    }

    #[test]
    fn refcount_lifecycle_frees_at_zero() {
        const KERNEL: KernelDef = KernelDef {
            name: Cow::Borrowed("vadd"),
            func: vector_add,
            kinds: Cow::Borrowed(&[ArgKind::Scalar(4)]),
            binding: SpirvBinding::NONE,
        };
        static MODULE: KernelModule = KernelModule {
            name: "t",
            kernels: &[KERNEL],
        };
        static MODULES: &[&KernelModule] = &[&MODULE];
        let registry = Registry::new(MODULES);
        let device = DeviceObject::leak(&TestDevice);
        let context = Context::new(vec![device]);
        let context_ref: &'static Context = Box::leak(context);

        assert_eq!(context_ref.header.refs(), 1);
        let program = Box::leak(Program::new(
            context_ref,
            vec![registry.find("vadd").expect("reg").clone()],
        ));
        assert_eq!(context_ref.header.refs(), 2);

        let kernel = Box::leak(Kernel::new(program, program.find("vadd").expect("def")));
        assert_eq!(program.header.refs(), 2);
        kernel.header.retain();
        assert_eq!(kernel.header.refs(), 2);
        release_ref(kernel);
        assert_eq!(kernel.header.refs(), 1);
        release_ref(kernel); // frees the kernel, dropping its program ref
        assert_eq!(program.header.refs(), 1);
        assert_eq!(context_ref.header.refs(), 2);
        release_ref(program); // frees the program, dropping its context ref
        assert_eq!(context_ref.header.refs(), 1);
        release_ref(context_ref);
    }

    #[test]
    fn buffer_create_and_bounds() {
        let device = DeviceObject::leak(&TestDevice);
        let context: &'static Context = Box::leak(Context::new(vec![device]));
        assert!(Mem::create(context, sys::MEM_READ_WRITE, 0, core::ptr::null_mut()).is_err());
        let buffer: &'static Mem =
            Box::leak(Mem::create(context, sys::MEM_READ_WRITE, 64, core::ptr::null_mut())
                .expect("alloc"));
        assert_eq!(buffer.size(), 64);
        assert!(!buffer.as_ptr().is_null());
        assert!(Mem::create(
            context,
            sys::MEM_USE_HOST_PTR,
            64,
            core::ptr::null_mut()
        )
        .is_err());
        assert!(Mem::create(
            context,
            sys::MEM_USE_HOST_PTR | sys::MEM_COPY_HOST_PTR,
            64,
            buffer.as_ptr().cast()
        )
        .is_err());
        release_ref(buffer);
        release_ref(context);
    }

    #[test]
    fn set_arg_validates_index_size_and_kind() {
        const KERNEL: KernelDef = KernelDef {
            name: Cow::Borrowed("one"),
            func: vector_add,
            kinds: Cow::Borrowed(&[ArgKind::Scalar(4), ArgKind::Buffer]),
            binding: SpirvBinding::NONE,
        };
        static MODULE: KernelModule = KernelModule {
            name: "t",
            kernels: &[KERNEL],
        };
        static MODULES: &[&KernelModule] = &[&MODULE];
        let device = DeviceObject::leak(&TestDevice);
        let context: &'static Context = Box::leak(Context::new(vec![device]));
        let program: &'static Program = Box::leak(Program::new(
            context,
            vec![Registry::new(MODULES).find("one").expect("reg").clone()],
        ));
        let mut kernel_box = Kernel::new(program, program.find("one").expect("def"));
        let kernel: &mut Kernel = &mut kernel_box;

        let four = 7u32;
        assert!(kernel
            .set_arg(1, core::mem::size_of::<MemHandle>(), core::ptr::null())
            .is_err());
        assert!(kernel.set_arg(2, 4, (&four as *const u32).cast()).is_err());
        assert!(kernel
            .set_arg(0, 8, (&four as *const u32).cast())
            .is_err());
        kernel
            .set_arg(0, 4, (&four as *const u32).cast())
            .expect("scalar fits");
        assert!(kernel.marshal().is_err()); // buffer arg still unset

        let buffer: &'static Mem =
            Box::leak(Mem::create(context, sys::MEM_READ_WRITE, 32, core::ptr::null_mut())
                .expect("alloc"));
        let buffer_raw = buffer.as_ptr() as usize;
        let handle = MemHandle::from_raw(core::ptr::from_ref(buffer).cast_mut().cast());
        kernel
            .set_arg(
                1,
                core::mem::size_of::<MemHandle>(),
                (&handle as *const MemHandle).cast(),
            )
            .expect("buffer binds");
        release_ref(buffer); // kernel retained its own reference
        let block = kernel.marshal().expect("all set");
        let mut bytes = [0u8; core::mem::size_of::<usize>()];
        bytes.copy_from_slice(&block.slot(1)[..core::mem::size_of::<usize>()]);
        assert_eq!(usize::from_ne_bytes(bytes), buffer_raw);

        drop(kernel_box);
        release_ref(program);
        release_ref(context);
    }
}
