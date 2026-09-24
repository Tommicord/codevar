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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Async-signal-safe crash/signal handling with stack unwinding.
//!
//! Installs POSIX `sigaction` handlers (or Windows vectored/console handlers)
//! that dump a backtrace via [`crate::basic_unwind`] and
//! [`crate::basic_pretty_unwind`], then reset the disposition to `SIG_DFL`
//! and re-raise so the default action (terminate/core/stop) still occurs.
//!
//! Notes:
//! - **No heap allocation, no locks, no `std`** — safe to run from a signal
//!   handler on an alternate signal stack.
//! - **Reentrancy guard** — a single `AtomicU32` CAS; a recursive or
//!   concurrent fault writes one short line and immediately re-raises with
//!   `SIG_DFL` instead of deadlocking.
//! - **Never catches `SIGKILL`/`SIGSTOP`**; job-control stop signals are
//!   opt-in via [`Options::catch_job_control`].
//!
//! **Known limitation**: on ELF targets the unwind walker may deadlock if the
//! crash occurs while the dynamic loader lock is held (`dl_iterate_phdr`);
//! the reentrancy guard bounds the damage to one extra line of output.

use core::fmt::{self, Write as _};
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::basic_pretty_unwind::write_frame;
use crate::basic_unwind::{Frame, capture_frames};

/// Maximum number of frames captured by [`dump_backtrace`] and the signal
/// handler. Bounded so the handler stays well within a 64 KiB alt stack.
const DUMP_FRAMES: usize = 64;

/// Reentrancy guard: `0` = idle, `1` = inside handler/dump.
static ENTERED: AtomicU32 = AtomicU32::new(0);

/// Whether [`install`] / [`install_with`] has succeeded and [`uninstall`]
/// has not yet been called.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Which categories of signals are currently being caught (mirrors the
/// [`Options`] used at install time).
static OPT_FAULTS: AtomicBool = AtomicBool::new(false);
static OPT_TERMINATION: AtomicBool = AtomicBool::new(false);
static OPT_JOB_CONTROL: AtomicBool = AtomicBool::new(false);
static OPT_ALT_STACK: AtomicBool = AtomicBool::new(false);

/// Which categories were active at install time (used by [`uninstall`]).
static INSTALLED_FAULTS: AtomicBool = AtomicBool::new(false);
static INSTALLED_TERMINATION: AtomicBool = AtomicBool::new(false);
static INSTALLED_JOB_CONTROL: AtomicBool = AtomicBool::new(false);

/// Configuration for [`install_with`].
///
/// The default enables fault signals, termination signals, and the
/// alternate signal stack, and disables job-control stop signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Catch synchronous hardware faults (`SIGSEGV`, `SIGFPE`, `SIGILL`,
    /// …) and dump a backtrace before re-raising.
    pub catch_faults: bool,
    /// Catch process-termination signals (`SIGINT`, `SIGTERM`, `SIGHUP`, …)
    /// and dump a backtrace before re-raising with `SIG_DFL`.
    pub catch_termination: bool,
    /// Catch job-control stop signals (`SIGTSTP`, `SIGTTIN`, `SIGTTOU`).
    /// Dump runs, then the signal is re-raised with `SIG_DFL`, which stops
    /// the process as usual. Disabled by default.
    pub catch_job_control: bool,
    /// Install a static 64 KiB 16-byte-aligned alternate signal stack so
    /// handlers run even when the faulting thread's stack is exhausted.
    pub install_alt_stack: bool,
}

impl Default for Options {
    #[inline]
    fn default() -> Self {
        Self {
            catch_faults: true,
            catch_termination: true,
            catch_job_control: false,
            install_alt_stack: true,
        }
    }
}

/// Errors returned by [`install`], [`install_with`], [`uninstall`], and
/// [`install_alt_stack`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallError {
    /// The current target has no supported signal-handler backend
    /// (e.g. bare metal, or a platform outside Unix/Windows).
    Unsupported,
    /// A platform syscall failed. `op` names the operation; `errno` is the
    /// raw OS error code (zero when the platform does not expose one).
    Syscall {
        /// Name of the failed operation (e.g. `"sigaction"`).
        op: &'static str,
        /// Raw OS error code.
        errno: i32,
    },
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => {
                f.write_str("signal handling not supported on this target")
            }
            Self::Syscall { op, errno } => {
                write!(f, "`{op}` failed with errno {errno}")
            }
        }
    }
}

impl core::error::Error for InstallError {}

/// Installs signal handlers with default [`Options`].
///
/// Equivalent to `install_with(Options::default())`. Calling this when
/// handlers are already installed is a no-op (`Ok`).
///
/// # Errors
///
/// Returns [`InstallError::Unsupported`] on targets without a handler
/// backend, or [`InstallError::Syscall`] when the OS rejects the request.
#[inline]
pub fn install() -> Result<(), InstallError> {
    install_with(Options::default())
}

/// Installs signal handlers according to `opts`.
///
/// On Unix this registers a `SA_SIGINFO | SA_ONSTACK` handler for every
/// enabled signal class, optionally installs a static alternate signal
/// stack, and saves previous dispositions for [`uninstall`]. On Windows it
/// registers a vectored exception handler, an unhandled-exception filter,
/// and (when `catch_termination` is set) a console control handler.
///
/// Calling this when handlers are already installed is a no-op (`Ok`).
///
/// # Errors
///
/// Returns [`InstallError::Unsupported`] on targets without a handler
/// backend, or [`InstallError::Syscall`] when the OS rejects the request
/// (any partially installed handlers are rolled back first).
#[inline]
pub fn install_with(opts: Options) -> Result<(), InstallError> {
    if INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }

    OPT_FAULTS.store(opts.catch_faults, Ordering::Relaxed);
    OPT_TERMINATION.store(opts.catch_termination, Ordering::Relaxed);
    OPT_JOB_CONTROL.store(opts.catch_job_control, Ordering::Relaxed);
    OPT_ALT_STACK.store(opts.install_alt_stack, Ordering::Relaxed);

    let result = install_impl(opts);
    if result.is_ok() {
        INSTALLED_FAULTS.store(opts.catch_faults, Ordering::Release);
        INSTALLED_TERMINATION.store(opts.catch_termination, Ordering::Release);
        INSTALLED_JOB_CONTROL.store(opts.catch_job_control, Ordering::Release);
        INSTALLED.store(true, Ordering::Release);
    } else {
        OPT_FAULTS.store(false, Ordering::Relaxed);
        OPT_TERMINATION.store(false, Ordering::Relaxed);
        OPT_JOB_CONTROL.store(false, Ordering::Relaxed);
        OPT_ALT_STACK.store(false, Ordering::Relaxed);
    }
    result
}

/// Removes handlers installed by [`install`] / [`install_with`], restoring
/// the previous dispositions on Unix.
///
/// Safe to call when nothing is installed (no-op). Does not disable an
/// alternate signal stack installed via [`install_alt_stack`].
///
/// # Errors
///
/// Currently infallible except on unsupported targets.
#[inline]
pub fn uninstall() -> Result<(), InstallError> {
    if !INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }
    let result = uninstall_impl();
    if result.is_ok() {
        INSTALLED.store(false, Ordering::Release);
        INSTALLED_FAULTS.store(false, Ordering::Release);
        INSTALLED_TERMINATION.store(false, Ordering::Release);
        INSTALLED_JOB_CONTROL.store(false, Ordering::Release);
        OPT_FAULTS.store(false, Ordering::Relaxed);
        OPT_TERMINATION.store(false, Ordering::Relaxed);
        OPT_JOB_CONTROL.store(false, Ordering::Relaxed);
        OPT_ALT_STACK.store(false, Ordering::Relaxed);
    }
    result
}

/// Returns `true` between a successful [`install`] / [`install_with`] and
/// the matching [`uninstall`].
#[inline]
pub fn is_installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// Installs `buf` as the alternate signal stack for the current thread.
///
/// `buf` must remain alive and unused for as long as it should serve as the
/// handler stack (typically a `'static` or leaked buffer). It is rejected if
/// shorter than 8 KiB.
///
/// # Errors
///
/// Returns [`InstallError::Unsupported`] off Unix, or
/// [`InstallError::Syscall`] when the buffer is too small or
/// `sigaltstack(2)` fails.
#[inline]
pub fn install_alt_stack(buf: &mut [u8]) -> Result<(), InstallError> {
    install_alt_stack_impl(buf)
}

/// Captures the current thread's backtrace and writes it to stderr.
///
/// Never panics; unwinds up to [`DUMP_FRAMES`] frames. Performs no heap
/// allocation. Concurrent calls (including from a signal handler) are
/// serialized by the reentrancy guard: if a dump is already in progress the
/// call returns immediately without output.
pub fn dump_backtrace() {
    // Only one dumper at a time; a failed CAS means either another thread
    // or the signal handler is already inside — do not interleave.
    if ENTERED
        .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    dump_frames();
    ENTERED.store(0, Ordering::Release);
}

/// Formats and writes captured frames. Caller must own the reentrancy
/// guard (or accept its use).
fn dump_frames() {
    let mut frames = [Frame::new(0, 0, None); DUMP_FRAMES];
    let n = capture_frames(&mut frames);
    for (i, frame) in frames[..n].iter().enumerate() {
        let mut line = LineWriter::new();
        let _ = line.write_char('#');
        let _ = write_frame(&mut line, frame, i);
        let _ = line.write_char('\n');
        line.flush();
    }
}

/// Fixed stack buffer that accumulates one formatted line, then flushes it
/// to stderr. Truncates instead of overflowing (never panics).
struct LineWriter {
    buf: [u8; 512],
    len: usize,
}

impl LineWriter {
    #[inline]
    fn new() -> Self {
        Self {
            buf: [0; 512],
            len: 0,
        }
    }

    /// Writes any pending bytes to stderr and resets the buffer.
    #[inline]
    fn flush(&mut self) {
        if self.len > 0 {
            write_console(&self.buf[..self.len]);
            self.len = 0;
        }
    }
}

impl fmt::Write for LineWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let src = s.as_bytes();
        let space = self.buf.len() - self.len;
        let take = src.len().min(space);
        self.buf[self.len..self.len + take].copy_from_slice(&src[..take]);
        self.len += take;
        // Silent truncation is fine: this is a crash-dump path.
        Ok(())
    }
}

/// Formats `s` into `buf` (truncating) and writes the result to stderr as a
/// single line including the trailing newline.
fn write_line(s: fmt::Arguments<'_>) {
    let mut line = LineWriter::new();
    let _ = line.write_fmt(s);
    let _ = line.write_char('\n');
    line.flush();
}

/// Returns a short symbolic name for `sig`, or `"SIGUNKNOWN"`.
///
/// Only defined on Unix, where the POSIX signal constants exist (Windows
/// maps exception codes to the same names in its own backend).
#[cfg(unix)]
fn signal_name(sig: core::ffi::c_int) -> &'static str {
    // Constant values differ per OS, so match on the libc constants.
    match sig {
        libc::SIGHUP => "SIGHUP",
        libc::SIGINT => "SIGINT",
        libc::SIGQUIT => "SIGQUIT",
        libc::SIGILL => "SIGILL",
        libc::SIGTRAP => "SIGTRAP",
        libc::SIGABRT => "SIGABRT",
        libc::SIGBUS => "SIGBUS",
        libc::SIGFPE => "SIGFPE",
        libc::SIGKILL => "SIGKILL",
        libc::SIGUSR1 => "SIGUSR1",
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGUSR2 => "SIGUSR2",
        libc::SIGPIPE => "SIGPIPE",
        libc::SIGALRM => "SIGALRM",
        libc::SIGTERM => "SIGTERM",
        libc::SIGCHLD => "SIGCHLD",
        libc::SIGCONT => "SIGCONT",
        libc::SIGSTOP => "SIGSTOP",
        libc::SIGTSTP => "SIGTSTP",
        libc::SIGTTIN => "SIGTTIN",
        libc::SIGTTOU => "SIGTTOU",
        libc::SIGURG => "SIGURG",
        libc::SIGXCPU => "SIGXCPU",
        libc::SIGXFSZ => "SIGXFSZ",
        libc::SIGVTALRM => "SIGVTALRM",
        libc::SIGPROF => "SIGPROF",
        libc::SIGWINCH => "SIGWINCH",
        libc::SIGSYS => "SIGSYS",
        #[cfg(any(target_os = "linux", target_os = "android"))]
        libc::SIGSTKFLT => "SIGSTKFLT",
        #[cfg(any(target_os = "linux", target_os = "android"))]
        libc::SIGPWR => "SIGPWR",
        #[cfg(any(target_os = "linux", target_os = "android"))]
        libc::SIGIO => "SIGIO",
        _ => "SIGUNKNOWN",
    }
}

/// Returns `true` for synchronous fault signals (where `si_addr` is
/// meaningful and the faulting instruction pointer is worth reporting).
#[cfg(unix)]
fn is_fault_signal(sig: core::ffi::c_int) -> bool {
    matches!(
        sig,
        libc::SIGILL
            | libc::SIGTRAP
            | libc::SIGABRT
            | libc::SIGBUS
            | libc::SIGFPE
            | libc::SIGSEGV
            | libc::SIGSYS
            | libc::SIGXCPU
            | libc::SIGXFSZ
    ) || {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            sig == libc::SIGSTKFLT
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Backend dispatch
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn install_impl(opts: Options) -> Result<(), InstallError> {
    unix::install(opts)
}

#[cfg(unix)]
fn uninstall_impl() -> Result<(), InstallError> {
    unix::uninstall();
    Ok(())
}

#[cfg(unix)]
fn install_alt_stack_impl(buf: &mut [u8]) -> Result<(), InstallError> {
    unix::install_alt_stack(buf)
}

#[cfg(all(windows, not(target_vendor = "uwp")))]
fn install_impl(opts: Options) -> Result<(), InstallError> {
    windows::install(opts)
}

#[cfg(all(windows, not(target_vendor = "uwp")))]
fn uninstall_impl() -> Result<(), InstallError> {
    windows::uninstall();
    Ok(())
}

#[cfg(all(windows, not(target_vendor = "uwp")))]
fn install_alt_stack_impl(_buf: &mut [u8]) -> Result<(), InstallError> {
    // Windows has no direct equivalent wired up here; the OS provides
    // stack-guarantee mechanisms for stack-overflow recovery instead.
    Err(InstallError::Unsupported)
}

#[cfg(not(any(unix, all(windows, not(target_vendor = "uwp")))))]
fn install_impl(_opts: Options) -> Result<(), InstallError> {
    Err(InstallError::Unsupported)
}

#[cfg(not(any(unix, all(windows, not(target_vendor = "uwp")))))]
fn uninstall_impl() -> Result<(), InstallError> {
    Err(InstallError::Unsupported)
}

#[cfg(not(any(unix, all(windows, not(target_vendor = "uwp")))))]
fn install_alt_stack_impl(_buf: &mut [u8]) -> Result<(), InstallError> {
    Err(InstallError::Unsupported)
}

// ---------------------------------------------------------------------------
// Console output
// ---------------------------------------------------------------------------

/// Writes `bytes` to stderr, retrying on `EINTR`. Best-effort: silently
/// stops on hard errors (the crash path must not block or panic).
#[cfg(unix)]
fn write_console(bytes: &[u8]) {
    let mut off = 0usize;
    while off < bytes.len() {
        // SAFETY: `bytes[off..]` is a valid read-only slice for the
        // duration of the call; `write` does not retain the pointer.
        let n = unsafe {
            libc::write(
                libc::STDERR_FILENO,
                bytes[off..].as_ptr().cast(),
                bytes.len() - off,
            )
        };
        if n < 0 {
            // SAFETY: `errno` is thread-local and valid immediately after
            // a failed syscall on all supported Unix targets.
            if unsafe { errno() } == libc::EINTR {
                continue;
            }
            return;
        }
        off += n as usize;
    }
}

/// Writes `bytes` to stderr via `WriteFile` on the cached standard-error
/// handle. Best-effort: ignores failures.
#[cfg(all(windows, not(target_vendor = "uwp")))]
fn write_console(bytes: &[u8]) {
    windows::write_stderr(bytes);
}

/// Writes `bytes` to stderr. No-op on targets without console support
/// (e.g. `wasm32`).
#[cfg(not(any(unix, all(windows, not(target_vendor = "uwp")))))]
fn write_console(_bytes: &[u8]) {}

/// Reads the calling thread's `errno` immediately after a failed syscall.
///
/// # Safety
///
/// Must be called on the same thread, immediately after the syscall that
/// failed, before any other libc call can clobber the value.
#[cfg(unix)]
unsafe fn errno() -> i32 {
    cfg_if::cfg_if! {
        if #[cfg(target_os = "android")] {
            // SAFETY: `__errno` (bionic) returns a valid pointer to the
            // calling thread's errno slot; bionic does not export
            // `__errno_location`.
            unsafe { *libc::__errno() }
        } else if #[cfg(any(target_os = "linux", target_os = "emscripten"))] {
            // SAFETY: `__errno_location` returns a valid pointer to the
            // calling thread's errno slot (glibc/musl contract).
            unsafe { *libc::__errno_location() }
        } else if #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos",
            target_os = "visionos",
            target_os = "freebsd",
            target_os = "dragonfly",
        ))] {
            // SAFETY: `__error` returns a valid pointer to the calling
            // thread's errno slot (BSD/libSystem contract).
            unsafe { *libc::__error() }
        } else {
            // Unknown Unix: no portable errno accessor in libc; report 0.
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Unix backend
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix {
    use super::{
        ENTERED, InstallError, Options, dump_frames, is_fault_signal, signal_name,
        write_console, write_line,
    };
    use core::ffi::c_void;
    use core::mem::{self, MaybeUninit};
    use core::ptr;
    use core::sync::atomic::Ordering;

    /// Maximum signals we track (fault + termination + job-control sets
    /// with Linux extras fits comfortably below this).
    const MAX_SAVED: usize = 32;

    /// Alternate stack size installed by default (matches `SIGSTKSZ` on
    /// mainstream Linux, well above `MINSIGSTKSZ` everywhere else).
    const DEFAULT_ALT_STACK: usize = 64 * 1024;

    /// Minimum accepted size for a user-supplied alternate stack.
    const MIN_ALT_STACK: usize = 8 * 1024;

    /// Saved previous dispositions for [`uninstall`].
    ///
    /// Only touched by [`install`] / [`uninstall`] (normal context), never
    /// from the signal handler; callers must not race these entry points.
    struct Saved {
        sigs: [libc::c_int; MAX_SAVED],
        acts: [MaybeUninit<libc::sigaction>; MAX_SAVED],
        count: usize,
    }

    static mut SAVED: Saved = Saved {
        sigs: [0; MAX_SAVED],
        acts: [const { MaybeUninit::uninit() }; MAX_SAVED],
        count: 0,
    };

    /// Static default alternate signal stack (16-byte aligned).
    #[repr(C, align(16))]
    struct AltStack {
        storage: [u8; DEFAULT_ALT_STACK],
    }

    static mut DEFAULT_STACK: AltStack = AltStack {
        storage: [0; DEFAULT_ALT_STACK],
    };

    /// Collects the signal numbers enabled by `opts` into `out`.
    /// Returns how many were written (truncates at `out.len()`).
    fn collect_signals(opts: Options, out: &mut [libc::c_int]) -> usize {
        let mut n = 0usize;
        macro_rules! push {
            ($($sig:expr),+ $(,)?) => {
                $(
                    if n < out.len() {
                        out[n] = $sig;
                        n += 1;
                    }
                )+
            };
        }

        if opts.catch_faults {
            push!(
                libc::SIGILL,
                libc::SIGTRAP,
                libc::SIGABRT,
                libc::SIGBUS,
                libc::SIGFPE,
                libc::SIGSEGV,
                libc::SIGSYS,
                libc::SIGXCPU,
                libc::SIGXFSZ,
            );
            #[cfg(any(target_os = "linux", target_os = "android"))]
            push!(libc::SIGSTKFLT);
        }
        if opts.catch_termination {
            push!(
                libc::SIGHUP,
                libc::SIGINT,
                libc::SIGQUIT,
                libc::SIGTERM,
                libc::SIGUSR1,
                libc::SIGUSR2,
                libc::SIGALRM,
                libc::SIGVTALRM,
                libc::SIGPROF,
            );
            #[cfg(any(target_os = "linux", target_os = "android"))]
            push!(libc::SIGPWR, libc::SIGIO);
        }
        if opts.catch_job_control {
            push!(libc::SIGTSTP, libc::SIGTTIN, libc::SIGTTOU);
        }
        n
    }

    /// Builds an empty `sigaction` whose mask blocks every signal in
    /// `sigs` (so a handler is never interrupted by a sibling caught
    /// signal).
    ///
    /// # Safety
    ///
    /// `sigs[..n]` must contain valid signal numbers.
    unsafe fn make_action(
        handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut c_void),
        sigs: &[libc::c_int],
    ) -> libc::sigaction {
        // SAFETY: zeroed `sigaction` has all fields at their zero value,
        // which is a valid starting point before we assign each field.
        let mut sa: libc::sigaction = unsafe { mem::zeroed() };
        sa.sa_sigaction = handler as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        // SAFETY: zeroed `sigset_t` is a valid empty set on all supported
        // libcs; `sigemptyset` normalizes any padding.
        unsafe {
            libc::sigemptyset(&mut sa.sa_mask);
            for &s in sigs {
                libc::sigaddset(&mut sa.sa_mask, s);
            }
        }
        sa
    }

    /// Saves `act` as the previous disposition of `sig` for [`uninstall`].
    ///
    /// # Safety
    ///
    /// Must only be called from `install`/`uninstall` context (not from a
    /// signal handler); the caller must not race another install/uninstall.
    unsafe fn save_prev(sig: libc::c_int, act: libc::sigaction) {
        // SAFETY: raw pointer to `static mut` without creating an
        // intermediate reference (edition 2024 `static mut` rules); only
        // touched under the documented install/uninstall discipline.
        let saved = unsafe { &mut *ptr::addr_of_mut!(SAVED) };
        if saved.count < MAX_SAVED {
            saved.sigs[saved.count] = sig;
            saved.acts[saved.count].write(act);
            saved.count += 1;
        }
    }

    /// Restores every saved disposition and clears the table.
    ///
    /// # Safety
    ///
    /// Same discipline as [`save_prev`].
    unsafe fn restore_prev() {
        // SAFETY: see `save_prev`.
        let saved = unsafe { &mut *ptr::addr_of_mut!(SAVED) };
        for i in (0..saved.count).rev() {
            let sig = saved.sigs[i];
            // SAFETY: entries are written only via `save_prev`, which
            // initializes them before `count` advances past the slot.
            let act = unsafe { saved.acts[i].assume_init_read() };
            // SAFETY: `sig` was successfully passed to `sigaction` during
            // install, so it is a valid signal number; `act` is the value
            // the kernel previously accepted for it.
            unsafe {
                libc::sigaction(sig, &act, ptr::null_mut());
            }
        }
        saved.count = 0;
    }

    /// Rolls back the first `count` entries of the current install attempt
    /// (which have not yet been recorded in `SAVED`) by resetting them to
    /// `SIG_DFL`.
    fn rollback_defaults(sigs: &[libc::c_int]) {
        for &sig in sigs {
            // SAFETY: zeroed action with flags 0 and `sa_sigaction =
            // SIG_DFL` (0) is the documented default disposition.
            let dfl: libc::sigaction = unsafe { mem::zeroed() };
            // SAFETY: signal numbers come from `collect_signals`.
            unsafe {
                libc::sigaction(sig, &dfl, ptr::null_mut());
            }
        }
    }

    pub(super) fn install(opts: Options) -> Result<(), InstallError> {
        let mut sigs = [0 as libc::c_int; MAX_SAVED];
        let n = collect_signals(opts, &mut sigs);

        if opts.install_alt_stack {
            // SAFETY: `DEFAULT_STACK` is a static with the required size
            // and alignment; it outlives the process. Taking a raw pointer
            // to a field of a `static mut` requires `unsafe` in edition
            // 2024; no reference is materialized.
            let storage = unsafe { ptr::addr_of_mut!(DEFAULT_STACK.storage) };
            let ss = libc::stack_t {
                ss_sp: storage.cast::<c_void>(),
                ss_size: DEFAULT_ALT_STACK,
                ss_flags: 0,
            };
            // SAFETY: `ss` points at a valid, 16-byte-aligned, process-
            // lifetime buffer of `ss_size` bytes.
            if unsafe { libc::sigaltstack(&ss, ptr::null_mut()) } != 0 {
                // SAFETY: called immediately after the failed syscall.
                let e = unsafe { super::errno() };
                return Err(InstallError::Syscall {
                    op: "sigaltstack",
                    errno: e,
                });
            }
        }

        if n == 0 {
            return Ok(());
        }

        // SAFETY: valid signal numbers from `collect_signals`.
        let action = unsafe { make_action(handler, &sigs[..n]) };

        // Start a fresh save table for this install attempt.
        // SAFETY: install discipline (no concurrent install/uninstall).
        unsafe {
            (*ptr::addr_of_mut!(SAVED)).count = 0;
        }

        for (installed, &sig) in sigs[..n].iter().enumerate() {
            let mut old: libc::sigaction = unsafe { mem::zeroed() };
            // SAFETY: `sig` is a valid signal number; `action` is fully
            // initialized; `old` receives the previous disposition.
            let rc = unsafe { libc::sigaction(sig, &action, &mut old) };
            if rc != 0 {
                // SAFETY: immediately after the failed syscall.
                let e = unsafe { super::errno() };
                rollback_defaults(&sigs[..installed]);
                return Err(InstallError::Syscall {
                    op: "sigaction",
                    errno: e,
                });
            }
            // SAFETY: same install discipline as above.
            unsafe { save_prev(sig, old) };
        }
        Ok(())
    }

    pub(super) fn uninstall() {
        // SAFETY: install discipline (no concurrent install/uninstall).
        unsafe { restore_prev() };
    }

    pub(super) fn install_alt_stack(buf: &mut [u8]) -> Result<(), InstallError> {
        if buf.len() < MIN_ALT_STACK {
            return Err(InstallError::Syscall {
                op: "sigaltstack:size",
                errno: 22, // EINVAL
            });
        }
        let ss = libc::stack_t {
            ss_sp: buf.as_mut_ptr().cast::<c_void>(),
            ss_size: buf.len(),
            ss_flags: 0,
        };
        // SAFETY: `buf` is a live mutable slice for the duration of this
        // call; the caller is documented to keep it alive while in use.
        if unsafe { libc::sigaltstack(&ss, ptr::null_mut()) } != 0 {
            // SAFETY: immediately after the failed syscall.
            let e = unsafe { super::errno() };
            return Err(InstallError::Syscall {
                op: "sigaltstack",
                errno: e,
            });
        }
        Ok(())
    }

    /// Resets `sig` to `SIG_DFL`, unblocks it on the current thread, and
    /// re-raises so the default action (terminate / core / stop) runs.
    ///
    /// May return for job-control stop signals after the process is
    /// resumed with `SIGCONT`; fatal signals do not return.
    fn reset_and_raise(sig: libc::c_int) {
        // SAFETY: zeroed action = `SIG_DFL`, flags 0.
        let dfl: libc::sigaction = unsafe { mem::zeroed() };
        // SAFETY: `sig` is the number of the signal currently being
        // delivered, hence valid.
        unsafe {
            libc::sigaction(sig, &dfl, ptr::null_mut());
        }

        // SAFETY: zeroed set + `sigemptyset` yields a valid empty set;
        // `sigaddset` on a valid signal number succeeds.
        let mut set: libc::sigset_t = unsafe { mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, sig);
            libc::sigprocmask(libc::SIG_UNBLOCK, &set, ptr::null_mut());
            libc::raise(sig);
        }
    }

    /// Extracts the faulting instruction pointer from a `ucontext_t`.
    ///
    /// # Safety
    ///
    /// `uc` must be null or a valid `ucontext_t*` provided by the kernel
    /// to a `SA_SIGINFO` handler.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    unsafe fn fault_ip(uc: *mut c_void) -> Option<usize> {
        if uc.is_null() {
            return None;
        }
        // SAFETY: caller guarantees `uc` is a kernel-supplied ucontext.
        let uc = unsafe { &*(uc.cast::<libc::ucontext_t>()) };
        let rip = uc.uc_mcontext.gregs[libc::REG_RIP as usize] as usize;
        (rip != 0).then_some(rip)
    }

    /// Extracts the faulting program counter from a `ucontext_t`.
    ///
    /// # Safety
    ///
    /// `uc` must be null or a valid `ucontext_t*` provided by the kernel.
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    unsafe fn fault_ip(uc: *mut c_void) -> Option<usize> {
        if uc.is_null() {
            return None;
        }
        // SAFETY: caller guarantees `uc` is a kernel-supplied ucontext.
        let uc = unsafe { &*(uc.cast::<libc::ucontext_t>()) };
        let pc = uc.uc_mcontext.pc as usize;
        (pc != 0).then_some(pc)
    }

    /// Extracts the faulting RIP from a Darwin `ucontext_t`.
    ///
    /// # Safety
    ///
    /// `uc` must be null or a valid `ucontext_t*` provided by the kernel.
    #[cfg(all(target_vendor = "apple", target_arch = "x86_64"))]
    unsafe fn fault_ip(uc: *mut c_void) -> Option<usize> {
        if uc.is_null() {
            return None;
        }
        // SAFETY: caller guarantees `uc` is a kernel-supplied ucontext;
        // `uc_mcontext` is a pointer that Darwin fills in before the
        // handler runs.
        let uc = unsafe { &*(uc.cast::<libc::ucontext_t>()) };
        let mc = uc.uc_mcontext;
        if mc.is_null() {
            return None;
        }
        // SAFETY: non-null mcontext pointer from the kernel.
        let rip = unsafe { (*mc).__ss.__rip as usize };
        (rip != 0).then_some(rip)
    }

    /// Extracts the faulting PC from a Darwin `ucontext_t`.
    ///
    /// # Safety
    ///
    /// `uc` must be null or a valid `ucontext_t*` provided by the kernel.
    #[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
    unsafe fn fault_ip(uc: *mut c_void) -> Option<usize> {
        if uc.is_null() {
            return None;
        }
        // SAFETY: caller guarantees `uc` is a kernel-supplied ucontext.
        let uc = unsafe { &*(uc.cast::<libc::ucontext_t>()) };
        let mc = uc.uc_mcontext;
        if mc.is_null() {
            return None;
        }
        // SAFETY: non-null mcontext pointer from the kernel.
        let pc = unsafe { (*mc).__ss.__pc as usize };
        (pc != 0).then_some(pc)
    }

    /// Fallback: no portable ucontext access on this architecture.
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_vendor = "apple", target_arch = "x86_64"),
        all(target_vendor = "apple", target_arch = "aarch64"),
    )))]
    unsafe fn fault_ip(_uc: *mut c_void) -> Option<usize> {
        None
    }

    /// Reads `si_addr` from a `siginfo_t` (fault address).
    ///
    /// # Safety
    ///
    /// `info` must be a valid kernel-supplied `siginfo_t*`; only meaningful
    /// for fault signals.
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "freebsd",
    ))]
    unsafe fn fault_addr(info: *const libc::siginfo_t) -> usize {
        if info.is_null() {
            return 0;
        }
        // SAFETY: `info` is kernel-provided; `si_addr` is defined for the
        // fault codes we query it on.
        unsafe { (*info).si_addr() as usize }
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "freebsd",
    )))]
    unsafe fn fault_addr(_info: *const libc::siginfo_t) -> usize {
        0
    }

    /// `SA_SIGINFO` handler: dump a report, then re-raise with `SIG_DFL`.
    extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, uc: *mut c_void) {
        // Reentrancy guard: never wait, never nest.
        if ENTERED
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            write_console(b"codevar: recursive signal during handling\n");
            reset_and_raise(sig);
            return;
        }

        let addr = if is_fault_signal(sig) {
            // SAFETY: kernel-supplied siginfo for a fault signal.
            unsafe { fault_addr(info) }
        } else {
            0
        };
        // SAFETY: kernel-supplied ucontext (maybe null in odd cases).
        let ip = unsafe { fault_ip(uc) };
        write_line(format_args!("Caught signal {} ({sig})", signal_name(sig)));
        if addr != 0 {
            write_line(format_args!("fault address: {addr:#018x}"));
        }
        if let Some(ip) = ip {
            write_line(format_args!("fault IP:      {ip:#018x}"));
        }
        write_line(format_args!("Backtrace: "));
        dump_frames();

        reset_and_raise(sig);
        // Only reachable for job-control stops resumed with SIGCONT.
        ENTERED.store(0, Ordering::Release);
    }
}

#[cfg(all(windows, not(target_vendor = "uwp")))]
mod windows {
    use super::{ENTERED, InstallError, Options};
    use super::{capture_frames, dump_frames, write_console, write_frame, write_line};
    use core::ffi::c_void;
    use core::fmt::{self, Write as _};
    use core::sync::atomic::{AtomicUsize, Ordering};

    // Parameter names are snake_case for the lints; they do not affect the
    // ABI of the imported symbols.
    windows_link::link!(
        "kernel32.dll" "system"
        fn AddVectoredExceptionHandler(
            first: u32,
            handler: *const c_void,
        ) -> *mut c_void
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn RemoveVectoredExceptionHandler(handler: *mut c_void) -> i32
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn SetUnhandledExceptionFilter(
            filter: *const c_void,
        ) -> *mut c_void
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn SetConsoleCtrlHandler(
            handler: *const c_void,
            add: i32,
        ) -> i32
    );
    windows_link::link!("kernel32.dll" "system" fn GetStdHandle(std_handle: i32) -> *mut c_void);
    windows_link::link!(
        "kernel32.dll" "system"
        fn WriteFile(
            file: *mut c_void,
            buffer: *const u8,
            number_of_bytes_to_write: u32,
            number_of_bytes_written: *mut u32,
            overlapped: *mut c_void,
        ) -> i32
    );
    windows_link::link!("kernel32.dll" "system" fn IsDebuggerPresent() -> i32);
    windows_link::link!(
        "kernel32.dll" "system"
        fn TerminateProcess(handle: *mut c_void, exit_code: u32) -> i32
    );
    windows_link::link!("kernel32.dll" "system" fn GetCurrentProcess() -> *mut c_void);
    windows_link::link!("kernel32.dll" "system" fn GetLastError() -> u32);

    const STD_ERROR_HANDLE: i32 = -12;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 1;

    // Exception codes we treat as fatal faults.
    const EXCEPTION_DATATYPE_MISALIGNMENT: u32 = 0x8000_0002;
    const EXCEPTION_BREAKPOINT: u32 = 0x8000_0003;
    const EXCEPTION_SINGLE_STEP: u32 = 0x8000_0004;
    const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;
    const EXCEPTION_IN_PAGE_ERROR: u32 = 0xC000_0006;
    const EXCEPTION_ARRAY_BOUNDS_EXCEEDED: u32 = 0xC000_008C;
    const EXCEPTION_FLT_DENORMAL_OPERAND: u32 = 0xC000_008D;
    const EXCEPTION_FLT_DIVIDE_BY_ZERO: u32 = 0xC000_008E;
    const EXCEPTION_FLT_OVERFLOW: u32 = 0xC000_008F;
    const EXCEPTION_FLT_UNDERFLOW: u32 = 0xC000_0090;
    const EXCEPTION_FLT_INVALID_OPERATION: u32 = 0xC000_0091;
    const EXCEPTION_FLT_STACK_CHECK: u32 = 0xC000_0092;
    const EXCEPTION_FLT_INEXACT_RESULT: u32 = 0xC000_0093;
    const EXCEPTION_INT_DIVIDE_BY_ZERO: u32 = 0xC000_0094;
    const EXCEPTION_INT_OVERFLOW: u32 = 0xC000_0095;
    const EXCEPTION_PRIV_INSTRUCTION: u32 = 0xC000_0096;
    const EXCEPTION_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
    const EXCEPTION_STACK_OVERFLOW: u32 = 0xC000_00FD;

    /// Cached `STD_ERROR_HANDLE` (zero = not yet fetched).
    static STDERR: AtomicUsize = AtomicUsize::new(0);
    /// Handle returned by `AddVectoredExceptionHandler` (zero = none).
    static VEH: AtomicUsize = AtomicUsize::new(0);
    /// Previous unhandled-exception filter (may be null).
    static PREV_FILTER: AtomicUsize = AtomicUsize::new(0);

    /// Minimal `EXCEPTION_RECORD` prefix (only the fields we read).
    #[repr(C)]
    struct ExceptionRecord {
        exception_code: u32,
        exception_flags: u32,
        exception_record: *mut ExceptionRecord,
        exception_address: *mut c_void,
        number_parameters: u32,
        _alignment: u32,
        exception_information: [usize; 15],
    }

    /// Minimal `EXCEPTION_POINTERS`.
    #[repr(C)]
    struct ExceptionPointers {
        exception_record: *mut ExceptionRecord,
        context_record: *mut c_void,
    }

    fn stderr_handle() -> *mut c_void {
        let cached = STDERR.load(Ordering::Acquire);
        if cached != 0 {
            return cached as *mut c_void;
        }
        // SAFETY: `GetStdHandle` accepts any `STD_*_HANDLE` constant.
        let h = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
        if !h.is_null() && h as isize != -1 {
            STDERR.store(h as usize, Ordering::Release);
        }
        h
    }

    pub(super) fn write_stderr(bytes: &[u8]) {
        let h = stderr_handle();
        if h.is_null() || h as isize == -1 {
            return;
        }
        let mut written = 0u32;
        // SAFETY: `h` is a valid std handle; `bytes` is a live slice of
        // the given length; `written` is a valid out-pointer.
        unsafe {
            WriteFile(
                h,
                bytes.as_ptr(),
                bytes.len() as u32,
                &mut written,
                core::ptr::null_mut(),
            );
        }
    }

    /// Maps an exception code to a short name for the report header.
    fn exception_name(code: u32) -> &'static str {
        match code {
            EXCEPTION_ACCESS_VIOLATION | EXCEPTION_IN_PAGE_ERROR => "SIGSEGV",
            EXCEPTION_ILLEGAL_INSTRUCTION | EXCEPTION_PRIV_INSTRUCTION => "SIGILL",
            EXCEPTION_INT_DIVIDE_BY_ZERO
            | EXCEPTION_FLT_DIVIDE_BY_ZERO
            | EXCEPTION_FLT_OVERFLOW
            | EXCEPTION_FLT_UNDERFLOW
            | EXCEPTION_FLT_INVALID_OPERATION
            | EXCEPTION_FLT_DENORMAL_OPERAND
            | EXCEPTION_FLT_STACK_CHECK
            | EXCEPTION_FLT_INEXACT_RESULT
            | EXCEPTION_INT_OVERFLOW => "SIGFPE",
            EXCEPTION_BREAKPOINT | EXCEPTION_SINGLE_STEP => "SIGTRAP",
            EXCEPTION_STACK_OVERFLOW => "SIGSEGV",
            EXCEPTION_ARRAY_BOUNDS_EXCEEDED | EXCEPTION_DATATYPE_MISALIGNMENT => "SIGBUS",
            _ => "SIGUNKNOWN",
        }
    }

    /// Returns `true` when `code` is a fault we should report.
    fn is_fault_code(code: u32) -> bool {
        matches!(
            code,
            EXCEPTION_DATATYPE_MISALIGNMENT
                | EXCEPTION_BREAKPOINT
                | EXCEPTION_SINGLE_STEP
                | EXCEPTION_ACCESS_VIOLATION
                | EXCEPTION_IN_PAGE_ERROR
                | EXCEPTION_ARRAY_BOUNDS_EXCEEDED
                | EXCEPTION_FLT_DENORMAL_OPERAND
                | EXCEPTION_FLT_DIVIDE_BY_ZERO
                | EXCEPTION_FLT_OVERFLOW
                | EXCEPTION_FLT_UNDERFLOW
                | EXCEPTION_FLT_INVALID_OPERATION
                | EXCEPTION_FLT_STACK_CHECK
                | EXCEPTION_FLT_INEXACT_RESULT
                | EXCEPTION_INT_DIVIDE_BY_ZERO
                | EXCEPTION_INT_OVERFLOW
                | EXCEPTION_PRIV_INSTRUCTION
                | EXCEPTION_ILLEGAL_INSTRUCTION
                | EXCEPTION_STACK_OVERFLOW
        )
    }

    /// Shared dump body: header + frames. Caller owns `ENTERED`.
    fn dump_exception(code: u32, fault_addr: usize, ip: usize) {
        write_line(format_args!(
            "=== codevar: {} ({code:#010x}) ===",
            exception_name(code)
        ));
        if fault_addr != 0 {
            write_line(format_args!("fault address: {fault_addr:#018x}"));
        }
        if ip != 0 {
            write_line(format_args!("fault IP:      {ip:#018x}"));
        }
        write_line(format_args!("--- backtrace ---"));
        dump_frames();
    }

    /// Stack-overflow-safe dump: static scratch only, minimal stack use.
    ///
    /// # Safety
    ///
    /// Caller must hold `ENTERED` (exclusive access to the statics).
    unsafe fn dump_stack_overflow(ip: usize) {
        write_console(b"codevar: stack overflow\n");
        if ip != 0 {
            write_line(format_args!("fault IP:      {ip:#018x}"));
        }

        static mut SCRATCH_FRAMES: [Frame; 16] = [Frame::new(0, 0, None); 16];
        static mut SCRATCH_LINE: [u8; 256] = [0; 256];

        // SAFETY: `ENTERED` is held, so no other thread or handler can
        // touch these statics concurrently.
        let frames = unsafe { &mut *core::ptr::addr_of_mut!(SCRATCH_FRAMES) };
        let n = capture_frames(&mut frames[..16]);
        for (i, frame) in frames[..n].iter().enumerate() {
            // SAFETY: exclusive access as above.
            let buf = unsafe { &mut *core::ptr::addr_of_mut!(SCRATCH_LINE) };
            let mut w = SliceWriter { buf, len: 0 };
            let _ = w.write_char('#');
            let _ = write_frame(&mut w, frame, i);
            let _ = w.write_char('\n');
            let out_len = w.len;
            // SAFETY: `out_len <= buf.len()` by construction of
            // `SliceWriter`; the bytes are UTF-8 (formatted output). The
            // slice is built from `addr_of!` to avoid an implicit autoref
            // of the raw pointer's dereference.
            let out = unsafe {
                core::slice::from_raw_parts(
                    core::ptr::addr_of!(SCRATCH_LINE).cast::<u8>(),
                    out_len,
                )
            };
            write_console(out);
        }
    }

    /// Bounded `fmt::Write` over a caller-provided byte slice.
    struct SliceWriter<'a> {
        buf: &'a mut [u8],
        len: usize,
    }

    impl fmt::Write for SliceWriter<'_> {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            let src = s.as_bytes();
            let space = self.buf.len() - self.len;
            let take = src.len().min(space);
            self.buf[self.len..self.len + take].copy_from_slice(&src[..take]);
            self.len += take;
            Ok(())
        }
    }

    use super::super::basic_unwind::Frame;

    /// `EXCEPTION_POINTERS` vectored handler.
    ///
    /// # Safety
    ///
    /// Called by the OS with a valid `EXCEPTION_POINTERS*`.
    unsafe extern "system" fn veh_handler(raw: *mut ExceptionPointers) -> i32 {
        if raw.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: OS guarantees a valid `ExceptionPointers` here.
        let rec = unsafe { (*raw).exception_record };
        if rec.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `rec` points at the active exception record.
        let code = unsafe { (*rec).exception_code };
        if !is_fault_code(code) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // Let a attached debugger handle breakpoints / single-steps.
        if matches!(code, EXCEPTION_BREAKPOINT | EXCEPTION_SINGLE_STEP) {
            // SAFETY: plain query, no arguments.
            if unsafe { IsDebuggerPresent() } != 0 {
                return EXCEPTION_CONTINUE_SEARCH;
            }
        }

        if ENTERED
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            write_console(b"codevar: recursive exception during handling\n");
            // SAFETY: current process handle is always valid.
            unsafe {
                TerminateProcess(GetCurrentProcess(), code);
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }

        // SAFETY: `rec` is live for the duration of this handler.
        let fault_addr = unsafe { (*rec).exception_address as usize };
        if code == EXCEPTION_STACK_OVERFLOW {
            // SAFETY: we hold `ENTERED`.
            unsafe { dump_stack_overflow(fault_addr) };
        } else {
            dump_exception(code, 0, fault_addr);
        }

        // SAFETY: current process handle is always valid.
        unsafe {
            TerminateProcess(GetCurrentProcess(), code);
        }
        // Unreachable if TerminateProcess succeeds.
        ENTERED.store(0, Ordering::Release);
        EXCEPTION_CONTINUE_SEARCH
    }

    /// Top-level unhandled-exception filter (fallback when the VEH is
    /// bypassed). Same report, then terminate with the exception code.
    ///
    /// # Safety
    ///
    /// Called by the OS with a valid `EXCEPTION_POINTERS*`.
    unsafe extern "system" fn unhandled_filter(raw: *mut ExceptionPointers) -> i32 {
        if raw.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: OS guarantees a valid `ExceptionPointers` here.
        let rec = unsafe { (*raw).exception_record };
        if rec.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `rec` points at the active exception record.
        let code = unsafe { (*rec).exception_code };
        if !is_fault_code(code) {
            return EXCEPTION_CONTINUE_SEARCH;
        }

        if ENTERED
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            write_console(b"codevar: recursive exception during handling\n");
            // SAFETY: current process handle is always valid.
            unsafe {
                TerminateProcess(GetCurrentProcess(), code);
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }

        // SAFETY: `rec` is live for the duration of this filter.
        let fault_addr = unsafe { (*rec).exception_address as usize };
        dump_exception(code, 0, fault_addr);

        // SAFETY: current process handle is always valid.
        unsafe {
            TerminateProcess(GetCurrentProcess(), code);
        }
        ENTERED.store(0, Ordering::Release);
        EXCEPTION_CONTINUE_SEARCH
    }

    /// Console control handler for Ctrl-C / Ctrl-Break / close events.
    ///
    /// # Safety
    ///
    /// Invoked by `SetConsoleCtrlHandler` with a well-known event code.
    unsafe extern "system" fn ctrl_handler(ctrl_type: u32) -> i32 {
        // 128 + signo: conventional shell-style exit codes.
        let (name, exit_code) = match ctrl_type {
            0 => (b"CTRL_C_EVENT".as_slice(), 128 + 2), // SIGINT
            1 => (b"CTRL_BREAK_EVENT".as_slice(), 128 + 3), // SIGQUIT
            2 => (b"CTRL_CLOSE_EVENT".as_slice(), 128 + 1), // SIGHUP
            5 => (b"CTRL_LOGOFF_EVENT".as_slice(), 128 + 1),
            6 => (b"CTRL_SHUTDOWN_EVENT".as_slice(), 128 + 1),
            _ => return 0,
        };

        if ENTERED
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            write_console(b"codevar: recursive signal during handling\n");
            // SAFETY: current process handle is always valid.
            unsafe {
                TerminateProcess(GetCurrentProcess(), exit_code as u32);
            }
            return 1;
        }

        write_console(b"=== codevar: ");
        write_console(name);
        write_console(b" ===\n");
        write_line(format_args!("--- backtrace ---"));
        dump_frames();

        // SAFETY: current process handle is always valid.
        unsafe {
            TerminateProcess(GetCurrentProcess(), exit_code as u32);
        }
        ENTERED.store(0, Ordering::Release);
        1
    }

    pub(super) fn install(opts: Options) -> Result<(), InstallError> {
        if opts.catch_faults || opts.catch_termination {
            // SAFETY: `veh_handler` has the required ABI and stays loaded
            // for the process lifetime (uninstall removes it explicitly).
            let veh =
                unsafe { AddVectoredExceptionHandler(1, veh_handler as *const c_void) };
            if veh.is_null() {
                // SAFETY: immediately after the failed call.
                let e = unsafe { GetLastError() } as i32;
                return Err(InstallError::Syscall {
                    op: "AddVectoredExceptionHandler",
                    errno: e,
                });
            }
            VEH.store(veh as usize, Ordering::Release);

            // SAFETY: `unhandled_filter` has the required ABI; the return
            // value is the previous filter pointer (may be null).
            let prev =
                unsafe { SetUnhandledExceptionFilter(unhandled_filter as *const c_void) };
            PREV_FILTER.store(prev as usize, Ordering::Release);
        }

        if opts.catch_termination {
            // SAFETY: `ctrl_handler` has the required ABI; `Add = 1`
            // registers it.
            if unsafe { SetConsoleCtrlHandler(ctrl_handler as *const c_void, 1) } == 0 {
                // SAFETY: immediately after the failed call.
                let e = unsafe { GetLastError() } as i32;
                return Err(InstallError::Syscall {
                    op: "SetConsoleCtrlHandler",
                    errno: e,
                });
            }
        }
        Ok(())
    }

    pub(super) fn uninstall() {
        let veh = VEH.swap(0, Ordering::AcqRel);
        if veh != 0 {
            // SAFETY: the handle was returned by
            // `AddVectoredExceptionHandler` and not yet removed.
            unsafe {
                RemoveVectoredExceptionHandler(veh as *mut c_void);
            }
        }
        let prev = PREV_FILTER.swap(0, Ordering::AcqRel);
        // SAFETY: restoring the exact pointer previously returned by
        // `SetUnhandledExceptionFilter` (may be null).
        unsafe {
            SetUnhandledExceptionFilter(prev as *const c_void);
        }
        // SAFETY: `ctrl_handler` has the required ABI; `Add = 0` removes
        // a previously registered handler.
        unsafe {
            SetConsoleCtrlHandler(ctrl_handler as *const c_void, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the signal-handler module.
    //!
    //! Tests that mutate process-wide handler state are serialized by
    //! [`LOCK`] because `cargo test` runs tests in parallel threads.
    //! `.unwrap()` is permitted here: tests only.

    use super::*;
    #[cfg(unix)]
    use core::ffi::c_int;
    #[cfg(unix)]
    use std::sync::{Mutex, PoisonError};

    /// Serializes tests that install/uninstall process-wide Unix handlers.
    #[cfg(unix)]
    static LOCK: Mutex<()> = Mutex::new(());

    #[cfg(unix)]
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[test]
    fn options_default_enables_faults_and_termination() {
        let opts = Options::default();
        assert!(opts.catch_faults);
        assert!(opts.catch_termination);
        assert!(!opts.catch_job_control);
        assert!(opts.install_alt_stack);
    }

    #[test]
    fn install_error_display_formats() {
        let unsupported = InstallError::Unsupported;
        assert!(!unsupported.to_string().is_empty());
        let syscall = InstallError::Syscall {
            op: "sigaction",
            errno: 22,
        };
        assert_eq!(syscall.to_string(), "`sigaction` failed with errno 22");
    }

    #[cfg(unix)]
    #[test]
    fn signal_name_reports_known_and_unknown() {
        assert_eq!(signal_name(libc::SIGTERM), "SIGTERM");
        assert_eq!(signal_name(libc::SIGSEGV), "SIGSEGV");
        assert_eq!(signal_name(libc::SIGINT), "SIGINT");
        // A reserved/unused number: portable "not a real signal" case.
        assert_eq!(signal_name(c_int::MAX), "SIGUNKNOWN");
    }

    #[cfg(unix)]
    #[test]
    fn is_fault_signal_classifies_signals() {
        assert!(is_fault_signal(libc::SIGSEGV));
        assert!(is_fault_signal(libc::SIGILL));
        assert!(is_fault_signal(libc::SIGFPE));
        assert!(is_fault_signal(libc::SIGBUS));
        assert!(is_fault_signal(libc::SIGABRT));
        assert!(!is_fault_signal(libc::SIGTERM));
        assert!(!is_fault_signal(libc::SIGINT));
        assert!(!is_fault_signal(libc::SIGUSR1));
        assert!(!is_fault_signal(c_int::MAX));
    }

    #[test]
    fn line_writer_truncates_instead_of_overflowing() {
        let mut line = LineWriter::new();
        let long = "x".repeat(600);
        let _ = line.write_str(&long);
        assert_eq!(line.len, line.buf.len());
        // Flushing twice must not re-emit stale bytes.
        line.flush();
        line.flush();
        assert_eq!(line.len, 0);
    }

    #[cfg(unix)]
    #[test]
    fn install_uninstall_roundtrip_restores_dispositions() {
        let _g = lock();

        // SAFETY: querying the current disposition of `SIGUSR1`, a signal
        // the test harness does not consume; `old` receives the result.
        let before: libc::sigaction = unsafe {
            let mut old: libc::sigaction = std::mem::zeroed();
            let rc = libc::sigaction(libc::SIGUSR1, std::ptr::null(), &mut old);
            assert_eq!(rc, 0, "sigaction query failed");
            old
        };

        assert!(!is_installed());
        install().expect("install must succeed");
        assert!(is_installed());

        // Installing again is a documented no-op.
        install().expect("second install must be a no-op");

        // SAFETY: querying the disposition installed by `install`.
        let during: libc::sigaction = unsafe {
            let mut old: libc::sigaction = std::mem::zeroed();
            let rc = libc::sigaction(libc::SIGUSR1, std::ptr::null(), &mut old);
            assert_eq!(rc, 0, "sigaction query failed");
            old
        };
        assert_ne!(
            before.sa_sigaction, during.sa_sigaction,
            "install must have replaced the disposition"
        );
        assert!(during.sa_flags & libc::SA_SIGINFO != 0);

        uninstall().expect("uninstall must succeed");
        assert!(!is_installed());
        // Uninstalling with nothing installed is a no-op.
        uninstall().expect("second uninstall must be a no-op");

        // SAFETY: querying the disposition after uninstall to compare with
        // the snapshot taken before install.
        let after: libc::sigaction = unsafe {
            let mut old: libc::sigaction = std::mem::zeroed();
            let rc = libc::sigaction(libc::SIGUSR1, std::ptr::null(), &mut old);
            assert_eq!(rc, 0, "sigaction query failed");
            old
        };
        assert_eq!(
            before.sa_sigaction, after.sa_sigaction,
            "uninstall must restore the previous disposition"
        );
    }

    #[cfg(unix)]
    #[test]
    fn install_with_empty_options_succeeds() {
        let _g = lock();
        let opts = Options {
            catch_faults: false,
            catch_termination: false,
            catch_job_control: false,
            install_alt_stack: false,
        };
        install_with(opts).expect("no-signal install must succeed");
        assert!(is_installed());
        uninstall().expect("uninstall must succeed");
        assert!(!is_installed());
    }

    #[cfg(unix)]
    #[test]
    fn install_with_job_control_opt_in() {
        let _g = lock();
        let opts = Options {
            catch_job_control: true,
            ..Options::default()
        };
        install_with(opts).expect("job-control install must succeed");
        assert!(is_installed());
        uninstall().expect("uninstall must succeed");
        assert!(!is_installed());
    }

    #[cfg(unix)]
    #[test]
    fn alt_stack_rejects_small_buffer_and_accepts_large() {
        let _g = lock();

        let mut small = [0u8; 1024];
        let err = install_alt_stack(&mut small).expect_err("tiny buffer must fail");
        assert!(
            matches!(err, InstallError::Syscall { op, .. } if op == "sigaltstack:size"),
            "unexpected error: {err:?}"
        );

        let mut big = vec![0u8; 16 * 1024];
        install_alt_stack(&mut big).expect("16 KiB buffer must be accepted");

        // SAFETY: disabling the alternate stack for this (test) thread so
        // no dangling pointer to `big` remains after the buffer drops.
        unsafe {
            let disable = libc::stack_t {
                ss_sp: std::ptr::null_mut(),
                ss_size: 0,
                ss_flags: libc::SS_DISABLE,
            };
            let rc = libc::sigaltstack(&disable, std::ptr::null_mut());
            assert_eq!(rc, 0, "sigaltstack disable failed");
        }
    }

    #[cfg(unix)]
    #[test]
    fn dump_backtrace_smoke() {
        let _g = lock();
        // Writes to stderr; the assertion is that it neither panics nor
        // leaves the reentrancy guard latched.
        dump_backtrace();
        assert_eq!(
            ENTERED.load(Ordering::Acquire),
            0,
            "reentrancy guard must be released after a dump"
        );
        // A second call must also run (guard not stuck).
        dump_backtrace();
        assert_eq!(ENTERED.load(Ordering::Acquire), 0);
    }
}
