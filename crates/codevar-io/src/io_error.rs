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

use alloc::boxed::Box;
use core::error;
use core::marker::PhantomData;
use core::num::NonZeroUsize;
use core::ptr::NonNull;
use core::{fmt, result};

/// Enumeration of possible methods to seek within an I/O object.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum SeekFrom {
    /// Sets the offset to the provided number of bytes.
    Start(u64),
    /// Sets the offset to the size of this object plus the specified number of bytes.
    End(i64),
    /// Sets the offset to the current position plus the specified number of bytes.
    Current(i64),
}

/// Possible kinds of errors.
///
/// This list is intended to grow over time, and it is not recommended to
/// exhaustively match against it. In application code, use `match` for the `ErrorKind`
/// values you are expecting; use `_` to match "all other errors".
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum ErrorKind {
    /// An entity was not found, often a file.
    NotFound,
    /// The operation lacked the necessary privileges to complete.
    PermissionDenied,
    /// The connection was refused by the remote server.
    ConnectionRefused,
    /// The connection was reset by the remote server.
    ConnectionReset,
    /// The remote host is not reachable.
    HostUnreachable,
    /// The network containing the remote host is not reachable.
    NetworkUnreachable,
    /// The connection was aborted (terminated) by the remote server.
    ConnectionAborted,
    /// The network operation failed because it was not connected yet.
    NotConnected,
    /// A socket address could not be bound because the address is already in
    /// use elsewhere.
    AddrInUse,
    /// A nonexistent interface was requested or the requested address was not
    /// local.
    AddrNotAvailable,
    /// The system's networking is down.
    NetworkDown,
    /// The operation failed because a pipe was closed.
    BrokenPipe,
    /// An entity already exists, often a file.
    AlreadyExists,
    /// The operation needs to block to complete, but the blocking operation was
    /// requested to not occur.
    WouldBlock,
    /// A filesystem object is, unexpectedly, not a directory.
    ///
    /// For example, a filesystem path was specified where one of the intermediate directory
    /// components was, in fact, a plain file.
    NotADirectory,
    /// The filesystem object is, unexpectedly, a directory.
    ///
    /// A directory was specified when a non-directory was expected.
    IsADirectory,
    /// A non-empty directory was specified where an empty directory was expected.
    DirectoryNotEmpty,
    /// The filesystem or storage medium is read-only, but a write operation was attempted.
    ReadOnlyFilesystem,
    /// Loop in the filesystem or IO subsystem; often, too many levels of symbolic links.
    ///
    /// There was a loop (or excessively long chain) resolving a filesystem object
    /// or file IO object.
    ///
    /// On Unix this is usually the result of a symbolic link loop; or, of exceeding the
    /// system-specific limit on the depth of symlink traversal.
    FilesystemLoop,
    /// Stale network file handle.
    ///
    /// With some network filesystems, notably NFS, an open file (or directory) can be invalidated
    /// by problems with the network or server.
    StaleNetworkFileHandle,
    /// A parameter was incorrect.
    InvalidInput,
    /// Data not valid for the operation were encountered.
    ///
    /// Unlike [`InvalidInput`], this typically means that the operation
    /// parameters were valid, however the error was caused by malformed
    /// input data.
    ///
    /// For example, a function that reads a file into a string will error with
    /// `InvalidData` if the file's contents are not valid UTF-8.
    ///
    /// [`InvalidInput`]: ErrorKind::InvalidInput
    InvalidData,
    /// The I/O operation's timeout expired, causing it to be canceled.
    TimedOut,
    /// An error returned when an operation could not be completed because a
    /// call to [`write`] returned [`Ok(0)`].
    ///
    /// This typically means that an operation could only succeed if it wrote a
    /// particular number of bytes but only a smaller number of bytes could be
    /// written.
    ///
    /// [`write`]: crate::io::Write::write
    /// [`Ok(0)`]: Ok
    WriteZero,
    /// The underlying storage (typically, a filesystem) is full.
    ///
    /// This does not include out of quota errors.
    StorageFull,
    /// Seek on unseekable file.
    ///
    /// Seeking was attempted on an open file handle which is not suitable for seeking - for
    /// example, on Unix, a named pipe opened with `File::open`.
    NotSeekable,
    /// File larger than allowed or supported.
    ///
    /// This might arise from a hard limit of the underlying filesystem or file access API, or from
    /// an administratively imposed resource limitation.  Simple disk full, and out of quota, have
    /// their own errors.
    FileTooLarge,
    /// Resource is busy.
    ResourceBusy,
    /// Executable file is busy.
    ///
    /// An attempt was made to write to a file which is also in use as a running program.  (Not all
    /// operating systems detect this situation.)
    ExecutableFileBusy,
    /// Cross-device or cross-filesystem (hard) link or rename.
    CrossesDevices,
    /// Too many (hard) links to the same filesystem object.
    ///
    /// The filesystem does not support making so many hardlinks to the same file.
    TooManyLinks,
    /// A filename was invalid.
    ///
    /// This error can also occur if a length limit for a name was exceeded.
    InvalidFilename,
    /// Program argument list too long.
    ///
    /// When trying to run an external program, a system or process limit on the size of the
    /// arguments would have been exceeded.
    ArgumentListTooLong,
    /// This operation was interrupted.
    ///
    /// Interrupted operations can typically be retried.
    Interrupted,

    /// This operation is unsupported on this platform.
    ///
    /// This means that the operation can never succeed.
    Unsupported,

    /// An error returned when an operation could not be completed because an
    /// "end of file" was reached prematurely.
    ///
    /// This typically means that an operation could only succeed if it read a
    /// particular number of bytes but only a smaller number of bytes could be
    /// read.
    UnexpectedEof,

    /// An operation could not be completed, because it failed
    /// to allocate enough memory.
    OutOfMemory,

    /// The operation was partially successful and needs to be checked
    /// later on due to not blocking.
    InProgress,

    /// A custom error that does not fall under any other I/O error kind.
    ///
    /// This can be used to construct your own [`Error`]s that do not match any
    /// [`ErrorKind`].
    ///
    /// This [`ErrorKind`] is not used by the standard library.
    ///
    /// Errors from the standard library that do not fall under any of the I/O
    /// error kinds cannot be `match`ed on, and will only match a wildcard (`_`) pattern.
    /// New [`ErrorKind`]s might be added in the future for some of those.
    Other,

    /// Any I/O error from the standard library that's not part of this list.
    ///
    /// Errors that are `Uncategorized` now may move to a different or a new
    /// [`ErrorKind`] variant in the future. It is not recommended to match
    /// an error against `Uncategorized`; use a wildcard match (`_`) instead.
    #[doc(hidden)]
    Uncategorized,
}

impl ErrorKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match *self {
            // tidy-alphabetical-start
            ErrorKind::AddrInUse => "address in use",
            ErrorKind::AddrNotAvailable => "address not available",
            ErrorKind::AlreadyExists => "entity already exists",
            ErrorKind::ArgumentListTooLong => "argument list too long",
            ErrorKind::BrokenPipe => "broken pipe",
            ErrorKind::ConnectionAborted => "connection aborted",
            ErrorKind::ConnectionRefused => "connection refused",
            ErrorKind::ConnectionReset => "connection reset",
            ErrorKind::CrossesDevices => "cross-device link or rename",
            ErrorKind::DirectoryNotEmpty => "directory not empty",
            ErrorKind::ExecutableFileBusy => "executable file busy",
            ErrorKind::FileTooLarge => "file too large",
            ErrorKind::FilesystemLoop => "filesystem loop or indirection limit (e.g. symlink loop)",
            ErrorKind::HostUnreachable => "host unreachable",
            ErrorKind::InProgress => "in progress",
            ErrorKind::Interrupted => "operation interrupted",
            ErrorKind::InvalidData => "invalid data",
            ErrorKind::InvalidFilename => "invalid filename",
            ErrorKind::InvalidInput => "invalid input parameter",
            ErrorKind::IsADirectory => "is a directory",
            ErrorKind::NetworkDown => "network down",
            ErrorKind::NetworkUnreachable => "network unreachable",
            ErrorKind::NotADirectory => "not a directory",
            ErrorKind::NotConnected => "not connected",
            ErrorKind::NotFound => "entity not found",
            ErrorKind::NotSeekable => "seek on unseekable file",
            ErrorKind::Other => "other error",
            ErrorKind::OutOfMemory => "out of memory",
            ErrorKind::PermissionDenied => "permission denied",
            ErrorKind::ReadOnlyFilesystem => "read-only filesystem or storage medium",
            ErrorKind::ResourceBusy => "resource busy",
            ErrorKind::StaleNetworkFileHandle => "stale network file handle",
            ErrorKind::StorageFull => "no storage space",
            ErrorKind::TimedOut => "timed out",
            ErrorKind::TooManyLinks => "too many links",
            ErrorKind::Uncategorized => "uncategorized error",
            ErrorKind::UnexpectedEof => "unexpected end of file",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::WouldBlock => "operation would block",
            ErrorKind::WriteZero => "write zero",
        }
    }
}
impl core::error::Error for ErrorKind {}

#[doc(hidden)]
#[repr(align(4))]
#[derive(Debug)]
pub struct SimpleMessage {
    pub kind: ErrorKind,
    pub message: &'static str,
}

/// The type of raw OS error codes returned by [`Error::raw_os_error`].
///
/// This is an [`i32`] on all currently supported platforms, but platforms
/// added in the future (such as UEFI) may use a different primitive type like
/// [`usize`]. Use `as`or [`into`] conversions where applicable to ensure maximum
/// portability.
///
/// [`into`]: Into::into
type RawOsError = i32;
// Only derive debug in tests, to make sure it
// doesn't accidentally get printed.
#[cfg_attr(test, derive(Debug))]
enum ErrorData<C> {
    Simple(ErrorKind),
    SimpleMessage(&'static SimpleMessage),
    Custom(C),
}
// The 2 least-significant bits are used as tag.
const TAG_MASK: usize = 0b11;
const TAG_SIMPLE_MESSAGE: usize = 0b00;
const TAG_CUSTOM: usize = 0b01;
const TAG_SIMPLE: usize = 0b11;

/// The internal representation.
///
/// This is just a way to hack in a check that we indeed are not unwind-safe.
#[repr(transparent)]
pub(super) struct Repr(NonNull<()>, PhantomData<ErrorData<Box<Custom>>>);

// All the types `Repr` stores internally are Send + Sync, and so is it.
unsafe impl Send for Repr {}
unsafe impl Sync for Repr {}

impl Repr {
    pub(super) fn new_custom(b: Box<Custom>) -> Self {
        let p = Box::into_raw(b).cast::<u8>();
        // Should only be possible if an allocator handed out a pointer with
        // wrong alignment.
        debug_assert_eq!(p.addr() & TAG_MASK, 0);
        // Note: We know `TAG_CUSTOM <= size_of::<Custom>()` (static_assert at
        // end of file), and both the start and end of the expression must be
        // valid without address space wraparound due to `Box`'s semantics.
        //
        // This means it would be correct to implement this using `ptr::add`
        // (rather than `ptr::wrapping_add`), but it's unclear this would give
        // any benefit, so we just use `wrapping_add` instead.
        let tagged = p.wrapping_add(TAG_CUSTOM).cast::<()>();
        // Safety: `TAG_CUSTOM + p` is the same as `TAG_CUSTOM | p`,
        // because `p`'s alignment means it isn't allowed to have any of the
        // `TAG_BITS` set (you can verify that addition and bitwise-or are the
        // same when the operands have no bits in common using a truth table).
        //
        // Then, `TAG_CUSTOM | p` is not zero, as that would require
        // `TAG_CUSTOM` and `p` both be zero, and neither is (as `p` came from a
        // box, and `TAG_CUSTOM` just... isn't zero -- it's `0b01`). Therefore,
        // `TAG_CUSTOM + p` isn't zero and so `tagged` can't be, and the
        // `new_unchecked` is safe.
        let res = Self(unsafe { NonNull::new_unchecked(tagged) }, PhantomData);
        // quickly smoke-check we encoded the right thing (This generally will
        // only run in std's tests, unless the user uses -Zbuild-std)
        debug_assert!(matches!(res.data(), ErrorData::Custom(_)), "repr(custom) encoding failed");
        res
    }

    #[inline]
    pub(super) fn new_simple(kind: ErrorKind) -> Self {
        let utagged = ((kind as usize) << 32) | TAG_SIMPLE;
        // Safety: `TAG_SIMPLE` is not zero, so the result of the `|` is not 0.
        let res = Self(
            NonNull::without_provenance(unsafe { NonZeroUsize::new_unchecked(utagged) }),
            PhantomData,
        );
        // quickly smoke-check we encoded the right thing (This generally will
        // only run in std's tests, unless the user uses -Zbuild-std)
        debug_assert!(
            matches!(res.data(), ErrorData::Simple(k) if k == kind),
            "repr(simple) encoding failed {:?}",
            kind,
        );
        res
    }

    #[inline]
    pub(super) const fn new_simple_message(m: &'static SimpleMessage) -> Self {
        // Safety: References are never null.
        Self(unsafe { NonNull::new_unchecked(m as *const _ as *mut ()) }, PhantomData)
    }

    #[inline]
    pub(super) fn data(&self) -> ErrorData<&Custom> {
        // Safety: We're a Repr, decode_repr is fine.
        unsafe { decode_repr(self.0, |c| &*c) }
    }

    #[inline]
    pub(super) fn data_mut(&mut self) -> ErrorData<&mut Custom> {
        // Safety: We're a Repr, decode_repr is fine.
        unsafe { decode_repr(self.0, |c| &mut *c) }
    }

    #[inline]
    pub(super) fn into_data(self) -> ErrorData<Box<Custom>> {
        let this = core::mem::ManuallyDrop::new(self);
        // Safety: We're a Repr, decode_repr is fine. The `Box::from_raw` is
        // safe because we prevent double-drop using `ManuallyDrop`.
        unsafe { decode_repr(this.0, |p| Box::from_raw(p)) }
    }
}

impl Drop for Repr {
    #[inline]
    fn drop(&mut self) {
        // Safety: We're a Repr, decode_repr is fine. The `Box::from_raw` is
        // safe because we're being dropped.
        unsafe {
            let _ = decode_repr(self.0, |p| Box::<Custom>::from_raw(p));
        }
    }
}

#[inline]
unsafe fn decode_repr<C, F>(ptr: NonNull<()>, make_custom: F) -> ErrorData<C>
where
    F: FnOnce(*mut Custom) -> C,
{
    let bits = ptr.as_ptr().addr();
    match bits & TAG_MASK {
        TAG_SIMPLE => {
            let kind_bits = (bits >> 32) as u32;
            let kind = kind_from_prim(kind_bits).unwrap_or_else(|| {
                debug_assert!(false, "Invalid io::error::Repr bits: `Repr({:#018x})`", bits);
                // This means the `ptr` passed in was not valid, which violates
                // the unsafe contract of `decode_repr`.
                //
                // Using this rather than unwrap meaningfully improves the code
                // for callers which only care about one variant (usually
                // `Custom`)
                unsafe { core::hint::unreachable_unchecked() };
            });
            ErrorData::Simple(kind)
        }
        TAG_SIMPLE_MESSAGE => {
            // SAFETY: per tag
            unsafe { ErrorData::SimpleMessage(&*ptr.cast::<SimpleMessage>().as_ptr()) }
        }
        TAG_CUSTOM => {
            // It would be correct for us to use `ptr::byte_sub` here (see the
            // comment above the `wrapping_add` call in `new_custom` for why),
            // but it isn't clear that it makes a difference, so we don't.
            let custom = ptr.as_ptr().wrapping_byte_sub(TAG_CUSTOM).cast::<Custom>();
            ErrorData::Custom(make_custom(custom))
        }
        _ => {
            // Can't happen, and compiler can tell
            unreachable!();
        }
    }
}

#[inline]
fn kind_from_prim(ek: u32) -> Option<ErrorKind> {
    macro_rules! from_prim {
        ($prim:expr => $Enum:ident { $($Variant:ident),* $(,)? }) => {{
            // Force a compile error if the list gets out of date.
            const _: fn(e: $Enum) = |e: $Enum| match e {
                $($Enum::$Variant => ()),*
            };
            match $prim {
                $(v if v == ($Enum::$Variant as _) => Some($Enum::$Variant),)*
                _ => None,
            }
        }}
    }
    from_prim!(ek => ErrorKind {
        NotFound,
        PermissionDenied,
        ConnectionRefused,
        ConnectionReset,
        HostUnreachable,
        NetworkUnreachable,
        ConnectionAborted,
        NotConnected,
        AddrInUse,
        AddrNotAvailable,
        NetworkDown,
        BrokenPipe,
        AlreadyExists,
        WouldBlock,
        NotADirectory,
        IsADirectory,
        DirectoryNotEmpty,
        ReadOnlyFilesystem,
        FilesystemLoop,
        StaleNetworkFileHandle,
        InvalidInput,
        InvalidData,
        TimedOut,
        WriteZero,
        StorageFull,
        NotSeekable,
        FileTooLarge,
        ResourceBusy,
        ExecutableFileBusy,
        CrossesDevices,
        TooManyLinks,
        InvalidFilename,
        ArgumentListTooLong,
        Interrupted,
        Other,
        UnexpectedEof,
        Unsupported,
        OutOfMemory,
        InProgress,
        Uncategorized,
    })
}

// Some static checking to alert us if a change breaks any of the assumptions
// that our encoding relies on for correctness and soundness. (Some of these are
// a bit overly thorough/cautious, admittedly)
//
// If any of these are hit on a platform that std supports, we should likely
// just use `repr_unpacked.rs` there instead (unless the fix is easy).
macro_rules! static_assert {
    ($condition:expr) => {
        const _: () = assert!($condition);
    };
    (@usize_eq: $lhs:expr, $rhs:expr) => {
        const _: [(); $lhs] = [(); $rhs];
    };
}

// The bitpacking we use requires pointers be exactly 64 bits.
static_assert!(@usize_eq: size_of::<NonNull<()>>(), 8);

// We also require pointers and usize be the same size.
static_assert!(@usize_eq: size_of::<NonNull<()>>(), size_of::<usize>());

// `Custom` and `SimpleMessage` need to be thin pointers.
static_assert!(@usize_eq: size_of::<&'static SimpleMessage>(), 8);
static_assert!(@usize_eq: size_of::<Box<Custom>>(), 8);

static_assert!((TAG_MASK + 1).is_power_of_two());
// And they must have sufficient alignment.
static_assert!(align_of::<SimpleMessage>() >= TAG_MASK + 1);
static_assert!(align_of::<Custom>() >= TAG_MASK + 1);

static_assert!(@usize_eq: TAG_MASK & TAG_SIMPLE_MESSAGE, TAG_SIMPLE_MESSAGE);
static_assert!(@usize_eq: TAG_MASK & TAG_CUSTOM, TAG_CUSTOM);
static_assert!(@usize_eq: TAG_MASK & TAG_SIMPLE, TAG_SIMPLE);

// This is obviously true (`TAG_CUSTOM` is `0b01`), but in `Repr::new_custom` we
// offset a pointer by this value, and expect it to both be within the same
// object, and to not wrap around the address space. See the comment in that
// function for further details.
//
// Actually, at the moment we use `ptr::wrapping_add`, not `ptr::add`, so this
// check isn't needed for that one, although the assertion that we don't
// actually wrap around in that wrapping_add does simplify the safety reasoning
// elsewhere considerably.
static_assert!(size_of::<Custom>() >= TAG_CUSTOM);

// These two store a payload which is allowed to be zero, so they must be
// non-zero to preserve the `NonNull`'s range invariant.
static_assert!(TAG_SIMPLE != 0);
// We can't tag `SimpleMessage`s, the tag must be 0.
static_assert!(@usize_eq: TAG_SIMPLE_MESSAGE, 0);
static_assert!(@usize_eq: size_of::<Repr>(), 8);
static_assert!(@usize_eq: size_of::<Option<Repr>>(), 8);
static_assert!(@usize_eq: size_of::<Result<(), Repr>>(), 8);
static_assert!(@usize_eq: size_of::<Result<usize, Repr>>(), 16);

#[derive(Debug)]
pub struct Error {
    repr: Repr,
}

#[derive(Debug)]
#[repr(align(4))]
struct Custom {
    kind: ErrorKind,
    error: Box<dyn error::Error + Send + Sync>,
}
impl alloc::fmt::Display for ErrorKind {
    /// Shows a human-readable description of the `ErrorKind`.
    ///
    /// This is similar to `impl Display for Error`, but doesn't require first converting to Error.
    ///
    /// # Examples
    /// ```
    /// use codevar_io::ErrorKind;
    /// assert_eq!("entity not found", ErrorKind::NotFound.to_string());
    /// ```
    fn fmt(&self, fmt: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        fmt.write_str(self.as_str())
    }
}

/// Intended for use for errors not exposed to the user, where allocating onto
/// the heap (for normal construction via Error::new) is too costly.
impl From<ErrorKind> for Error {
    /// Converts an [`ErrorKind`] into an [`Error`].
    ///
    /// This conversion creates a new error with a simple representation of error kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_io::{Error, ErrorKind};
    ///
    /// let not_found = ErrorKind::NotFound;
    /// let error = Error::from(not_found);
    /// assert_eq!("entity not found", format!("{error}"));
    /// ```
    #[inline]
    fn from(kind: ErrorKind) -> Error {
        Error {repr: Repr::new_simple(kind)}
    }
}

impl Error {
    /// Creates a new I/O error from a known kind of error as well as an
    /// arbitrary error payload.
    ///
    /// This function is used to generically create I/O errors which do not
    /// originate from the OS itself. The `error` argument is an arbitrary
    /// payload which will be contained in this [`Error`].
    ///
    /// Note that this function allocates memory on the heap.
    /// If no extra payload is required, use the `From` conversion from
    /// `ErrorKind`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_io::{Error, ErrorKind};
    ///
    /// // errors can be created from strings
    /// let custom_error = Error::new(ErrorKind::Other, "oh no!");
    ///
    /// // errors can also be created from other errors
    /// let custom_error2 = Error::new(ErrorKind::Interrupted, custom_error);
    ///
    /// // creating an error without payload (and without memory allocation)
    /// let eof_error = Error::from(ErrorKind::UnexpectedEof);
    /// ```
    #[inline(never)]
    pub fn new<E>(kind: ErrorKind, error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        Self::_new(kind, error.into())
    }
    fn _new(kind: ErrorKind, error: Box<dyn error::Error + Send + Sync>) -> Error {
        Error { repr: Repr::new_custom(Box::new(Custom { kind, error })) }
    }

    /// Creates a new I/O error from an arbitrary error payload.
    ///
    /// This function is used to generically create I/O errors which do not
    /// originate from the OS itself. It is a shortcut for [`Error::new`]
    /// with [`ErrorKind::Other`].
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_io::Error;
    ///
    /// // errors can be created from strings
    /// let custom_error = Error::other("oh no!");
    ///
    /// // errors can also be created from other errors
    /// let custom_error2 = Error::other(custom_error);
    /// ```
    pub fn other<E>(error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        Self::new(ErrorKind::Other, error.into())
    }

    /// Creates a new I/O error from a known kind of error as well as a constant
    /// message.
    ///
    /// This function does not allocate.
    ///
    /// You should not use this directly, and instead use the `const_error!`
    /// macro: `io::const_error!(ErrorKind::Something, "some_message")`.
    ///
    /// This function should maybe change to `from_static_message<const MSG: &'static
    /// str>(kind: ErrorKind)` in the future, when const generics allow that.
    #[inline]
    #[doc(hidden)]
    pub const fn from_static_message(msg: &'static SimpleMessage) -> Error {
        Self { repr: Repr::new_simple_message(msg) }
    }

    /// Returns a reference to the inner error wrapped by this error (if any).
    #[must_use]
    #[inline]
    pub fn get_ref(&self) -> Option<&(dyn error::Error + Send + Sync + 'static)> {
        match self.repr.data() {
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(&*c.error),
        }
    }

    /// Returns a mutable reference to the inner error wrapped by this error
    /// (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: Error::new
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut (dyn error::Error + Send + Sync + 'static)> {
        match self.repr.data_mut() {
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(&mut *c.error),
        }
    }

    /// Consumes the `Error`, returning its inner error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] or [`other`],
    /// then this function will return [`Some`],
    /// otherwise it will return [`None`].
    ///
    /// [`new`]: Error::new
    /// [`other`]: Error::other
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub fn into_inner(self) -> Option<Box<dyn error::Error + Send + Sync>> {
        match self.repr.into_data() {
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(c.error),
        }
    }

    /// Attempts to downcast the custom boxed error to `E`.
    ///
    /// If this [`Error`] contains a custom boxed error,
    /// then it would attempt downcasting on the boxed error,
    /// otherwise it will return [`Err`].
    ///
    /// If the custom boxed error has the same type as `E`, it will return [`Ok`],
    /// otherwise it will also return [`Err`].
    ///
    /// This method is meant to be a convenience routine for calling
    /// `Box<dyn Error + Sync + Send>::downcast` on the custom boxed error, returned by
    /// [`Error::into_inner`].
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    /// use std::error::Error;
    ///
    /// #[derive(Debug)]
    /// enum E {
    ///     Io(codevar_io::Error),
    ///     SomeOtherVariant,
    /// }
    ///
    /// impl fmt::Display for E {
    ///    // ...
    /// #    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #        todo!()
    /// #    }
    /// }
    /// impl Error for E {}
    ///
    /// impl From<codevar_io::Error> for E {
    ///     fn from(err: codevar_io::Error) -> E {
    ///         err.downcast::<E>()
    ///             .unwrap_or_else(E::Io)
    ///     }
    /// }
    ///
    /// impl From<E> for codevar_io::Error {
    ///     fn from(err: E) -> codevar_io::Error {
    ///         match err {
    ///             E::Io(io_error) => io_error,
    ///             e => codevar_io::Error::new(codevar_io::ErrorKind::Other, e),
    ///         }
    ///     }
    /// }
    ///
    /// # fn main() {
    /// let e = E::SomeOtherVariant;
    /// // Convert it to an io::Error
    /// let io_error = codevar_io::Error::from(e);
    /// // Cast it back to the original variant
    /// let e = E::from(io_error);
    /// assert!(matches!(e, E::SomeOtherVariant));
    ///
    /// let io_error = codevar_io::Error::from(codevar_io::ErrorKind::AlreadyExists);
    /// // Convert it to E
    /// let e = E::from(io_error);
    /// // Cast it back to the original variant
    /// let io_error = codevar_io::Error::from(e);
    /// assert_eq!(io_error.kind(), codevar_io::ErrorKind::AlreadyExists);
    /// assert!(io_error.get_ref().is_none());
    /// # }
    /// ```
    pub fn downcast<E>(self) -> result::Result<E, Self>
    where
        E: error::Error + Send + Sync + 'static,
    {
        if let ErrorData::Custom(c) = self.repr.data()
            && c.error.is::<E>()
        {
            if let ErrorData::Custom(b) = self.repr.into_data()
                && let Ok(err) = b.error.downcast::<E>()
            {
                Ok(*err)
            } else {
                unsafe { core::hint::unreachable_unchecked() }
            }
        } else {
            Err(self)
        }
    }

    /// Returns the corresponding [`ErrorKind`] for this error.
    #[must_use]
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        match self.repr.data() {
            ErrorData::Custom(c) => c.kind,
            ErrorData::Simple(kind) => kind,
            ErrorData::SimpleMessage(m) => m.kind,
        }
    }
}

impl alloc::fmt::Debug for Repr {
    fn fmt(&self, fmt: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        match self.data() {
            ErrorData::Custom(c) => alloc::fmt::Debug::fmt(&c, fmt),
            ErrorData::Simple(kind) => fmt.debug_tuple("Kind").field(&kind).finish(),
            ErrorData::SimpleMessage(msg) => fmt
                .debug_struct("Error")
                .field("kind", &msg.kind)
                .field("message", &msg.message)
                .finish(),
        }
    }
}

impl alloc::fmt::Display for Error {
    fn fmt(&self, fmt: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        match self.repr.data() {
            ErrorData::Custom(ref c) => c.error.fmt(fmt),
            ErrorData::Simple(kind) => write!(fmt, "{}", kind.as_str()),
            ErrorData::SimpleMessage(msg) => msg.message.fmt(fmt),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.repr.data() {
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error.source(),
        }
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn error::Error> {
        match self.repr.data() {
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error.cause(),
        }
    }
}

/// Base trait for all IO traits, defining the error type.
///
/// All IO operations of all traits return the error defined in this trait.
///
/// Having a shared trait instead of having every trait define its own
/// `Error` associated type enforces all impls on the same type use the same error.
/// This is very convenient when writing generic code, it means you have to
/// handle a single error type `T::Error`, instead of `<T as Read>::Error` and `<T as Write>::Error`
/// which might be different types.
pub trait ErrorType {
    /// Error type of all the IO operations on this type.
    type Error: core::error::Error;
}

impl<T: ?Sized + ErrorType> ErrorType for &mut T {
    type Error = T::Error;
}

/// Error returned by [`Read::read_exact`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum ReadExactError<E> {
    /// An EOF error was encountered before reading the exact amount of requested bytes.
    UnexpectedEof,
    /// Error returned by the inner Read.
    Other(E),
}

impl<E> From<E> for ReadExactError<E> {
    fn from(err: E) -> Self {
        Self::Other(err)
    }
}

impl<E: fmt::Debug> fmt::Display for ReadExactError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<E: core::error::Error + 'static> core::error::Error for ReadExactError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::UnexpectedEof => None,
            Self::Other(error) => Some(error),
        }
    }
}

/// Errors that could be returned by `Write` on `&mut [u8]`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum SliceWriteError {
    /// The target slice was full and so could not receive any new data.
    Full,
}

impl From<ErrorKind> for SliceWriteError {
    fn from(_kind: ErrorKind) -> Self {
        SliceWriteError::Full
    }
}

impl fmt::Display for SliceWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::error::Error for SliceWriteError {}

/// Error returned by [`Write::write_fmt`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum WriteFmtError<E> {
    /// An error was encountered while formatting.
    FmtError,
    /// Error returned by the inner Write.
    Other(E),
}

impl<E> From<E> for WriteFmtError<E> {
    fn from(err: E) -> Self {
        Self::Other(err)
    }
}

impl<E: fmt::Debug> fmt::Display for WriteFmtError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<E: core::error::Error + 'static> core::error::Error for WriteFmtError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::FmtError => None,
            Self::Other(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Read};
    use std::error::Error as StdError;

    #[test]
    fn test_seek_from_variants() {
        let start = SeekFrom::Start(100);
        assert_eq!(start, SeekFrom::Start(100));

        let end = SeekFrom::End(-50);
        assert_eq!(end, SeekFrom::End(-50));

        let current = SeekFrom::Current(10);
        assert_eq!(current, SeekFrom::Current(10));
    }

    #[test]
    fn test_error_kind_equality() {
        assert_eq!(ErrorKind::Other, ErrorKind::Other);
        assert_eq!(ErrorKind::NotFound, ErrorKind::NotFound);
        assert_ne!(ErrorKind::Other, ErrorKind::NotFound);
    }

    #[test]
    fn test_error_kind_debug() {
        let kind = ErrorKind::NotFound;
        let debug_str = format!("{:?}", kind);
        assert_eq!(debug_str, "NotFound");
    }

    #[test]
    fn test_error_kind_display() {
        let kind = ErrorKind::PermissionDenied;
        let display_str = format!("{}", kind);
        assert_eq!(display_str, "permission denied");
    }

    #[test]
    fn test_error_kind_error_trait() {
        let kind = ErrorKind::OutOfMemory;
        let _error: &dyn core::error::Error = &kind;
    }

    #[test]
    fn test_read_exact_error_from() {
        let err = ReadExactError::from(ErrorKind::TimedOut);
        assert!(matches!(err, ReadExactError::Other(ErrorKind::TimedOut)));
    }

    #[test]
    fn test_read_exact_error_display() {
        let err: ReadExactError<ErrorKind> = ReadExactError::UnexpectedEof;
        let display_str = format!("{}", err);
        assert!(display_str.contains("UnexpectedEof"));
    }

    #[test]
    fn test_read_exact_error_source() {
        let err: ReadExactError<ErrorKind> = ReadExactError::UnexpectedEof;
        assert!(err.source().is_none());

        let err: ReadExactError<ErrorKind> = ReadExactError::Other(ErrorKind::Other);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_slice_write_error_display() {
        let err = SliceWriteError::Full;
        let display_str = format!("{}", err);
        assert!(display_str.contains("Full"));
    }

    #[test]
    fn test_slice_write_error_source() {
        let err = SliceWriteError::Full;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_write_fmt_error_from() {
        let err = WriteFmtError::from(ErrorKind::Other);
        assert!(matches!(err, WriteFmtError::Other(ErrorKind::Other)));
    }

    #[test]
    fn test_write_fmt_error_display() {
        let err: WriteFmtError<ErrorKind> = WriteFmtError::FmtError;
        let display_str = format!("{}", err);
        assert!(display_str.contains("FmtError"));
    }

    #[test]
    fn test_write_fmt_error_source() {
        let err: WriteFmtError<ErrorKind> = WriteFmtError::FmtError;
        assert!(err.source().is_none());

        let err: WriteFmtError<ErrorKind> = WriteFmtError::Other(ErrorKind::Other);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_error_type_for_mut() {
        struct MyReader;
        impl ErrorType for MyReader {
            type Error = ErrorKind;
        }
        impl Read for MyReader {
            fn read(&mut self, _buf: &mut [u8]) -> Result<usize, Self::Error> {
                Ok(0)
            }
        }

        let mut reader = MyReader;
        let _mut_ref = &mut reader;
        // Verify ErrorType is implemented for &mut T
        let _: <&mut MyReader as ErrorType>::Error = ErrorKind::Other;
    }
}
