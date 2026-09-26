//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Unix domain socket transport and `poll(2)` poller.
//!
//! [`WlUnixTransport`] implements [`WlTransport`] over a connected
//! `AF_UNIX` stream socket with `SCM_RIGHTS` file descriptor passing,
//! which is the wire a Wayland client uses to talk to a compositor
//! listening at `$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY`. [`WlUnixPoller`]
//! is the matching [`WlPoller`] that drives a [`crate::WlEventLoop`]
//! from `poll(2)`.
//!
//! The module is only compiled on Unix targets.
//!
//! # Example
//!
//! ```no_run
//! use codevar_wl_protocol::{WlClientDisplay, WlUnixTransport};
//!
//! # fn demo() -> codevar_wl_protocol::WlResult<()> {
//! let transport = WlUnixTransport::connect_session()?;
//! let mut display = WlClientDisplay::connect(transport)?;
//! let _registry = display.get_registry()?;
//! # Ok(())
//! # }
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

use crate::wl_conn::{WlHandle, WlTransport};
use crate::wl_error::{WlError, WlResult};
use crate::wl_evloop::{WlPollEntry, WlPoller};
use crate::wl_handle::{WlFd, WlPollEvents};

/// Maximum number of file descriptors moved with a single message in
/// either direction.
///
/// The control buffer of a read reserves space for this many
/// descriptors; reads that would carry more are rejected instead of
/// silently dropping descriptors, and writes of larger lists are
/// refused up front.
const MAX_FDS_PER_MESSAGE: usize = 16;

/// Control buffer space reserved by each `recvmsg`/`sendmsg` call,
/// large enough for [`MAX_FDS_PER_MESSAGE`] descriptors.
const CONTROL_SPACE: usize = {
    // SAFETY: `CMSG_SPACE` only aligns and adds its argument and the
    // header size; no memory is dereferenced.
    unsafe { libc::CMSG_SPACE((MAX_FDS_PER_MESSAGE * core::mem::size_of::<libc::c_int>()) as u32) as usize }
};

/// Returns the thread-local `errno` value of the last failed call.
#[inline]
fn errno() -> libc::c_int {
    // SAFETY: `__errno_location` returns a pointer to the calling
    // thread's error number, which is always valid to read.
    unsafe { *libc::__errno_location() }
}

/// Builds a transport error carrying the failed operation and `errno`.
#[inline]
fn io_error(operation: &str, code: libc::c_int) -> WlError {
    WlError::io(format!("{operation}: errno {code}"))
}

/// Closes `fd`, ignoring errors from descriptors that are already gone.
#[inline]
fn close_fd(fd: libc::c_int) {
    // SAFETY: `fd` is a descriptor the caller owns; `close` releases it.
    unsafe { libc::close(fd) };
}

/// Closes every descriptor in `fds`.
fn close_fds(fds: &[WlFd]) {
    for &fd in fds {
        close_fd(fd);
    }
}

/// Puts `fd` into non-blocking mode and marks it close-on-exec.
///
/// # Errors
///
/// Returns [`WlError::Io`] when a `fcntl` query or update fails. The
/// descriptor is left untouched and owned by the caller on failure.
fn set_nonblocking_cloexec(fd: libc::c_int) -> WlResult<()> {
    // SAFETY: `fd` is an open descriptor; `fcntl` only reads or
    // writes flag bits and never dereferences a pointer.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io_error("fcntl", errno()));
    }
    // SAFETY: as above, the value argument is a plain flag word.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io_error("fcntl", errno()));
    }
    // SAFETY: as above.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return Err(io_error("fcntl", errno()));
    }
    // SAFETY: as above.
    if unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(io_error("fcntl", errno()));
    }
    Ok(())
}

/// Builds a `sockaddr_un` for the filesystem socket `path`.
///
/// # Errors
///
/// Returns [`WlError::InvalidArgument`] when `path` does not fit into
/// `sun_path`.
fn unix_address(path: &str) -> WlResult<(libc::sockaddr_un, libc::socklen_t)> {
    // SAFETY: a zeroed `sockaddr_un` has a valid NUL-terminated path.
    let mut sun: libc::sockaddr_un = unsafe { core::mem::zeroed() };
    sun.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_bytes();
    if bytes.is_empty() {
        return Err(WlError::invalid_argument("empty socket path"));
    }
    if bytes.len() >= sun.sun_path.len() {
        return Err(WlError::invalid_argument(format!("socket path too long: {path}")));
    }
    // SAFETY: `sun_path` is zero-initialized and `bytes.len()` was
    // bounds-checked above, so the copy stays inside the array and
    // the trailing NUL written by `zeroed` remains in place.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            sun.sun_path.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    let offset = core::mem::size_of::<libc::sockaddr_un>() - sun.sun_path.len();
    let len = (offset + bytes.len() + 1) as libc::socklen_t;
    Ok((sun, len))
}

/// Waits until `fd` becomes writable after a interrupted connect.
///
/// # Errors
///
/// Returns [`WlError::Io`] when the poll fails or the socket reports
/// a pending connection error.
fn finish_connect(fd: libc::c_int) -> WlResult<()> {
    loop {
        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        // SAFETY: `pollfd` is a valid single-element array and `poll`
        // only writes into `revents`.
        let ready = unsafe { libc::poll(&mut pollfd, 1, -1) };
        if ready < 0 {
            let code = errno();
            if code == libc::EINTR {
                continue;
            }
            return Err(io_error("poll", code));
        }
        let mut error: libc::c_int = 0;
        let mut length = core::mem::size_of_val(&error) as libc::socklen_t;
        // SAFETY: `error` and `length` describe a writable buffer of
        // the size the kernel expects for `SO_ERROR`.
        if unsafe {
            libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_ERROR,
                &mut error as *mut _ as *mut libc::c_void,
                &mut length,
            )
        } < 0
        {
            return Err(io_error("getsockopt", errno()));
        }
        if error != 0 {
            return Err(io_error("connect", error));
        }
        return Ok(());
    }
}

/// Reads the `SCM_RIGHTS` descriptors carried by `message`.
///
/// # Safety
///
/// `message` must describe a completed `recvmsg` call whose control
/// buffer is still valid and initialized. The returned descriptors
/// were installed by the kernel for this process; the caller must
/// close or hand each of them over.
unsafe fn collect_rights(message: &libc::msghdr) -> Vec<WlFd> {
    let mut fds = Vec::new();
    // SAFETY: the caller guarantees `message` wraps a live control
    // buffer, so `CMSG_FIRSTHDR` and `CMSG_NXTHDR` return pointers
    // inside that buffer or null, and the header and data reads
    // below stay inside headers the kernel initialized.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(message);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let total = (*header).cmsg_len as usize;
                let header_len = libc::CMSG_LEN(0) as usize;
                if total >= header_len {
                    let count = (total - header_len) / core::mem::size_of::<libc::c_int>();
                    let data = libc::CMSG_DATA(header).cast::<libc::c_int>();
                    for index in 0..count {
                        fds.push(*data.add(index));
                    }
                }
            }
            header = libc::CMSG_NXTHDR(message, header);
        }
    }
    fds
}

/// Converts `timeout` into the millisecond argument of `poll`.
///
/// `None` maps to an infinite wait and zero-length durations to a
/// non-blocking poll.
#[inline]
fn poll_timeout(timeout: Option<Duration>) -> libc::c_int {
    match timeout {
        None => -1,
        Some(limit) => {
            let millis = limit.as_millis();
            if millis > libc::c_int::MAX as u128 {
                libc::c_int::MAX
            } else {
                millis as libc::c_int
            }
        }
    }
}

/// Maps `revents` from `poll(2)` onto [`WlPollEvents`].
#[inline]
fn map_revents(revents: libc::c_short) -> WlPollEvents {
    let mut events = WlPollEvents::EMPTY;
    if revents & libc::POLLIN != 0 {
        events.insert(WlPollEvents::READABLE);
    }
    if revents & libc::POLLOUT != 0 {
        events.insert(WlPollEvents::WRITABLE);
    }
    if revents & libc::POLLHUP != 0 {
        events.insert(WlPollEvents::HANGUP);
    }
    if revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
        events.insert(WlPollEvents::ERROR);
    }
    events
}

/// Reads the environment variable `name`, if set.
fn get_env(name: &str) -> Option<String> {
    // SAFETY: `getenv` returns a pointer to an internal buffer valid
    // until the next call to `getenv`. The string is copied out
    // immediately and the pointer is never retained.
    let ptr = unsafe { libc::getenv(name.as_ptr() as *const libc::c_char) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the pointer is a NUL-terminated C string (or null,
    // which was ruled out above).
    let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
    cstr.to_str().ok().map(String::from)
}

/// Builds the compositor socket path for `wayland_display`.
///
/// A display name of `None` selects the default `wayland-0`. An
/// absolute display path is used verbatim, otherwise the name is
/// resolved inside `runtime_dir`, matching `libwayland`'s rules.
///
/// # Errors
///
/// Returns [`WlError::InvalidArgument`] for an empty display name and
/// [`WlError::InvalidState`] when the display is relative but
/// `runtime_dir` is missing.
fn display_socket_path(runtime_dir: Option<&str>, wayland_display: Option<&str>) -> WlResult<String> {
    let display = wayland_display.unwrap_or("wayland-0");
    if display.is_empty() {
        return Err(WlError::invalid_argument("WAYLAND_DISPLAY is empty"));
    }
    if display.starts_with('/') {
        return Ok(String::from(display));
    }
    let runtime_dir = runtime_dir.ok_or_else(|| WlError::invalid_state("XDG_RUNTIME_DIR is not set"))?;
    Ok(format!("{runtime_dir}/{display}"))
}

/// A [`WlTransport`] backed by a Unix domain stream socket.
///
/// The socket is put into non-blocking mode and close-on-exec by
/// every constructor. [`WlTransport::recv`] collects `SCM_RIGHTS`
/// descriptors into the caller's vector in arrival order, matching
/// how [`crate::WlConnection`] queues them for message arguments, and
/// [`WlTransport::send`] attaches descriptors to the first written
/// byte. Dropping the transport closes the socket; descriptors
/// delivered to `recv` belong to the caller and are released with
/// [`WlTransport::release_fd`].
#[derive(Debug)]
pub struct WlUnixTransport {
    fd: libc::c_int,
}

impl WlUnixTransport {
    /// Connects to the filesystem socket at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when `path` is empty or
    /// too long for `sockaddr_un`, and [`WlError::Io`] when the
    /// socket cannot be created or connected.
    pub fn connect(path: &str) -> WlResult<Self> {
        let (sun, length) = unix_address(path)?;
        let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
        if fd < 0 {
            return Err(io_error("socket", errno()));
        }
        if let Err(error) = set_nonblocking_cloexec(fd) {
            close_fd(fd);
            return Err(error);
        }
        loop {
            // SAFETY: `sun` describes a live address of `length`
            // bytes and `fd` is an open socket descriptor.
            let rc = unsafe { libc::connect(fd, &sun as *const _ as *const libc::sockaddr, length) };
            if rc == 0 {
                return Ok(Self { fd });
            }
            let code = errno();
            if code == libc::EINTR {
                continue;
            }
            if code == libc::EINPROGRESS || code == libc::EWOULDBLOCK {
                return finish_connect(fd).map(|()| Self { fd });
            }
            close_fd(fd);
            return Err(io_error("connect", code));
        }
    }

    /// Connects to the compositor of the current session.
    ///
    /// The socket path is resolved from `WAYLAND_DISPLAY` (defaulting
    /// to `wayland-0`) against `XDG_RUNTIME_DIR`, using the same
    /// rules as `libwayland`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] when `XDG_RUNTIME_DIR` is
    /// unset for a relative display name, [`WlError::InvalidArgument`]
    /// when `WAYLAND_DISPLAY` is empty, and [`WlError::Io`] when the
    /// socket cannot be opened.
    pub fn connect_session() -> WlResult<Self> {
        let runtime_dir = get_env("XDG_RUNTIME_DIR");
        let display = get_env("WAYLAND_DISPLAY");
        let path = display_socket_path(runtime_dir.as_deref(), display.as_deref())?;
        Self::connect(&path)
    }

    /// Adopts an already connected socket `fd`.
    ///
    /// The descriptor is switched to non-blocking, close-on-exec
    /// mode and ownership moves to the transport.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when `fd` is negative and
    /// [`WlError::Io`] when a `fcntl` update fails. `fd` stays owned
    /// by the caller on failure.
    pub fn from_fd(fd: libc::c_int) -> WlResult<Self> {
        if fd < 0 {
            return Err(WlError::invalid_argument("negative file descriptor"));
        }
        set_nonblocking_cloexec(fd)?;
        Ok(Self { fd })
    }

    /// Creates a connected socket pair.
    ///
    /// The two transports are peers of each other, which is the
    /// cheapest way to wire a client to a server inside one process.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Io`] when `socketpair` or a flag update
    /// fails.
    pub fn pair() -> WlResult<(Self, Self)> {
        let mut fds = [-1 as libc::c_int; 2];
        // SAFETY: `fds` is a valid two element array; on success the
        // kernel fills both entries with connected descriptors.
        if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) } < 0 {
            return Err(io_error("socketpair", errno()));
        }
        for fd in fds {
            if let Err(error) = set_nonblocking_cloexec(fd) {
                close_fds(&fds);
                return Err(error);
            }
        }
        Ok((Self { fd: fds[0] }, Self { fd: fds[1] }))
    }

    /// Returns the raw file descriptor.
    #[must_use]
    #[inline]
    pub const fn fd(&self) -> libc::c_int {
        self.fd
    }

    /// Detaches the raw file descriptor without closing it.
    ///
    /// The caller takes over ownership and is responsible for
    /// releasing the descriptor.
    #[must_use]
    pub fn into_fd(self) -> libc::c_int {
        let this = core::mem::ManuallyDrop::new(self);
        this.fd
    }
}

impl WlTransport for WlUnixTransport {
    fn recv(&mut self, buf: &mut [u8], fds: &mut Vec<WlFd>) -> WlResult<usize> {
        loop {
            let mut iov = libc::iovec {
                iov_base: buf.as_mut_ptr().cast(),
                iov_len: buf.len(),
            };
            let mut control = [0u8; CONTROL_SPACE];
            let mut message: libc::msghdr = unsafe { core::mem::zeroed() };
            message.msg_iov = &mut iov;
            message.msg_iovlen = 1;
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = control.len();
            // SAFETY: `iov` points into `buf`, `control` is a valid
            // writable array and `message` references both with their
            // correct lengths; `self.fd` is an open socket, so
            // `recvmsg` may write into all three.
            let n = unsafe { libc::recvmsg(self.fd, &mut message, 0) };
            if n < 0 {
                let code = errno();
                if code == libc::EINTR {
                    continue;
                }
                if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
                    return Err(WlError::WouldBlock);
                }
                return Err(io_error("recvmsg", code));
            }
            // SAFETY: `message` wraps a completed `recvmsg` call whose
            // control buffer is the local `control` array.
            let collected = unsafe { collect_rights(&message) };
            if message.msg_flags & libc::MSG_CTRUNC != 0 {
                // The control buffer overflowed, so part of the
                // descriptor list was lost. Release what arrived and
                // fail loudly instead of losing descriptors silently.
                close_fds(&collected);
                return Err(WlError::io(
                    "control message truncated: too many file descriptors",
                ));
            }
            // Queue descriptors in arrival order so they line up with
            // the message arguments parsed from the byte stream.
            fds.extend(collected);
            if n == 0 {
                return Ok(0);
            }
            return Ok(n as usize);
        }
    }

    fn send(&mut self, data: &[u8], fds: &[WlFd]) -> WlResult<usize> {
        if data.is_empty() {
            return Ok(0);
        }
        if fds.is_empty() {
            loop {
                // SAFETY: `data` is a readable slice and `self.fd`
                // is an open socket; `MSG_NOSIGNAL` suppresses
                // `SIGPIPE` when the peer is gone.
                let n = unsafe { libc::send(self.fd, data.as_ptr().cast(), data.len(), libc::MSG_NOSIGNAL) };
                if n >= 0 {
                    return Ok(n as usize);
                }
                let code = errno();
                if code == libc::EINTR {
                    continue;
                }
                if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
                    return Err(WlError::WouldBlock);
                }
                return Err(io_error("send", code));
            }
        }
        if fds.len() > MAX_FDS_PER_MESSAGE {
            return Err(WlError::unsupported(format!(
                "cannot send {} file descriptors in one message (limit {})",
                fds.len(),
                MAX_FDS_PER_MESSAGE
            )));
        }
        loop {
            let mut iov = libc::iovec {
                iov_base: data.as_ptr().cast_mut().cast(),
                iov_len: data.len(),
            };
            let mut control = [0u8; CONTROL_SPACE];
            let mut message: libc::msghdr = unsafe { core::mem::zeroed() };
            message.msg_iov = &mut iov;
            message.msg_iovlen = 1;
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = control.len();
            // SAFETY: `control` is at least `CMSG_SPACE` bytes long,
            // which always fits one `cmsghdr`, so `CMSG_FIRSTHDR`
            // returns a pointer inside `control`.
            let header = unsafe { libc::CMSG_FIRSTHDR(&message) };
            if header.is_null() {
                return Err(WlError::io("control buffer too small for file descriptors"));
            }
            let bytes = core::mem::size_of_val(fds);
            // SAFETY: `header` points into `control`, which has room
            // for `MAX_FDS_PER_MESSAGE` descriptors and `fds.len()`
            // was bounded above; `CMSG_LEN` computes the declared
            // payload length for that many descriptors.
            unsafe {
                (*header).cmsg_level = libc::SOL_SOCKET;
                (*header).cmsg_type = libc::SCM_RIGHTS;
                (*header).cmsg_len = libc::CMSG_LEN(bytes as u32) as _;
                core::ptr::copy_nonoverlapping(
                    fds.as_ptr(),
                    libc::CMSG_DATA(header).cast::<libc::c_int>(),
                    fds.len(),
                );
            }
            // Only the bytes of the header filled in above belong to
            // the kernel; trailing zero padding would be parsed as a
            // broken control message.
            message.msg_controllen = unsafe { libc::CMSG_SPACE(bytes as u32) } as _;
            // SAFETY: `self.fd` is an open socket, `iov` and
            // `control` reference valid buffers for the duration of
            // the call, and the control message was fully initialized
            // above. `MSG_NOSIGNAL` suppresses `SIGPIPE`.
            let n = unsafe { libc::sendmsg(self.fd, &message, libc::MSG_NOSIGNAL) };
            if n < 0 {
                let code = errno();
                if code == libc::EINTR {
                    continue;
                }
                if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
                    // Nothing was queued, so the caller still owns
                    // `fds` and retries with them.
                    return Err(WlError::WouldBlock);
                }
                return Err(io_error("sendmsg", code));
            }
            // On success the kernel keeps its own references to the
            // descriptors; the caller closes its copies. At least one
            // byte was accepted, so all of them are consumed.
            return Ok(n as usize);
        }
    }

    fn wait(&mut self, timeout: Option<Duration>, mask: WlPollEvents) -> WlResult<WlPollEvents> {
        let mut events: libc::c_short = 0;
        if mask.contains(WlPollEvents::READABLE) {
            events |= libc::POLLIN;
        }
        if mask.contains(WlPollEvents::WRITABLE) {
            events |= libc::POLLOUT;
        }
        let mut pollfd = libc::pollfd {
            fd: self.fd,
            events,
            revents: 0,
        };
        // SAFETY: `pollfd` is a valid single-element array and
        // `poll` writes only into `revents`.
        let ready = unsafe { libc::poll(&mut pollfd, 1, poll_timeout(timeout)) };
        if ready < 0 {
            let code = errno();
            if code == libc::EINTR {
                return Ok(WlPollEvents::EMPTY);
            }
            return Err(io_error("poll", code));
        }
        if ready == 0 {
            return Ok(WlPollEvents::EMPTY);
        }
        // Hangups and errors are reported even when they were not
        // requested, matching `epoll` semantics, so a blocking wait
        // with a read-only mask can never miss a closed peer.
        Ok(map_revents(pollfd.revents))
    }

    fn handle(&self) -> WlHandle {
        self.fd as WlHandle
    }

    fn release_fd(&mut self, fd: WlFd) {
        close_fd(fd);
    }
}

impl Drop for WlUnixTransport {
    fn drop(&mut self) {
        close_fd(self.fd);
    }
}

/// A [`WlPoller`] backed by `poll(2)`.
///
/// The poller waits on the raw file descriptors in
/// [`WlPollEntry::handle`], which is how [`WlUnixTransport::handle`]
/// presents its socket, and reports readiness back to a
/// [`crate::WlEventLoop`]. Hangups and errors are always reported,
/// even when only one direction was requested.
#[derive(Debug, Default, Clone, Copy)]
pub struct WlUnixPoller;

impl WlPoller for WlUnixPoller {
    fn poll(&mut self, entries: &mut [WlPollEntry], timeout: Option<Duration>) -> WlResult<usize> {
        let mut pollfds = Vec::with_capacity(entries.len());
        for entry in entries.iter() {
            let mut events: libc::c_short = 0;
            if entry.interest.contains(WlPollEvents::READABLE) {
                events |= libc::POLLIN;
            }
            if entry.interest.contains(WlPollEvents::WRITABLE) {
                events |= libc::POLLOUT;
            }
            pollfds.push(libc::pollfd {
                fd: entry.handle as libc::c_int,
                events,
                revents: 0,
            });
        }
        // SAFETY: `pollfds` holds `entries.len()` initialized
        // elements; `poll` writes only into the `revents` fields and
        // accepts a null pointer when the count is zero.
        let ready = unsafe {
            libc::poll(
                pollfds.as_mut_ptr(),
                pollfds.len() as libc::nfds_t,
                poll_timeout(timeout),
            )
        };
        if ready < 0 {
            let code = errno();
            if code == libc::EINTR {
                return Ok(0);
            }
            return Err(io_error("poll", code));
        }
        let mut count = 0usize;
        for (entry, pollfd) in entries.iter_mut().zip(pollfds.iter()) {
            entry.revents = map_revents(pollfd.revents);
            if !entry.revents.is_empty() {
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_display_name_against_runtime_dir() {
        let path = display_socket_path(Some("/run/user/1000"), Some("wayland-0")).unwrap();
        assert_eq!(path, "/run/user/1000/wayland-0");
    }

    #[test]
    fn defaults_to_wayland_zero_without_display_name() {
        let path = display_socket_path(Some("/run/user/1000"), None).unwrap();
        assert_eq!(path, "/run/user/1000/wayland-0");
    }

    #[test]
    fn absolute_display_path_is_used_verbatim() {
        let path = display_socket_path(Some("/run/user/1000"), Some("/tmp/wayland-custom")).unwrap();
        assert_eq!(path, "/tmp/wayland-custom");
    }

    #[test]
    fn relative_display_without_runtime_dir_is_rejected() {
        let error = display_socket_path(None, Some("wayland-0")).unwrap_err();
        assert!(matches!(error, WlError::InvalidState(_)));
    }

    #[test]
    fn empty_display_name_is_rejected() {
        let error = display_socket_path(Some("/run/user/1000"), Some("")).unwrap_err();
        assert!(matches!(error, WlError::InvalidArgument(_)));
    }

    #[test]
    fn overlong_socket_path_is_rejected() {
        let long = "x".repeat(200);
        let error = unix_address(&long).unwrap_err();
        assert!(matches!(error, WlError::InvalidArgument(_)));
    }
}
