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

//! Unix domain socket transport using [`libc`].
//!
//! This module is only compiled for native Unix targets. It
//! connects to well-known socket paths and exposes a
//! [`DbusTransport`] implementation backed by an `AF_UNIX`
//! socket.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use core::time::Duration;
use libc::{c_int, sockaddr_un, socklen_t};

use crate::dbus_addr::{DbusAddress, SESSION_BUS_FILE, SYSTEM_BUS_SOCKET, SYSTEM_BUS_SOCKET_LEGACY};
use crate::dbus_error::{DbusError, DbusResult};
use crate::dbus_transport::{DbusPollEvents, DbusTransport, close_fds};

/// Default timeout for a method call, matching the D-Bus
/// specification default of 25 seconds.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_millis(25_000);

/// Maximum number of file descriptors moved with a single message in
/// either direction.
///
/// The control buffer of a read reserves space for this many
/// descriptors; reads that would carry more are rejected instead of
/// silently dropping descriptors, and writes of larger lists are
/// refused up front.
pub const MAX_FDS_PER_MESSAGE: usize = 16;

/// Control buffer space reserved by each `recvmsg`/`sendmsg` call,
/// large enough for [`MAX_FDS_PER_MESSAGE`] descriptors.
const CONTROL_SPACE: usize = {
    // SAFETY: `CMSG_SPACE` only aligns and adds its argument and the
    // header size; no memory is dereferenced.
    unsafe { libc::CMSG_SPACE((MAX_FDS_PER_MESSAGE * core::mem::size_of::<libc::c_int>()) as u32) as usize }
};

/// A [`DbusTransport`] backed by a Unix domain socket.
///
/// Reads are performed with `recvmsg`, so descriptors sent with
/// `SCM_RIGHTS` are collected into an internal queue returned by
/// [`DbusTransport::take_fds`]. The queue is drained automatically
/// when the transport is dropped.
pub struct UnixTransport {
    fd: c_int,
    pending_fds: VecDeque<i32>,
}

impl UnixTransport {
    /// Connects to a `unix:path=...` address.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidAddress`] when the address
    /// carries a non-path transport or an empty path, and
    /// [`DbusError::Io`] when the socket cannot be opened or
    /// connected.
    pub fn connect_to(address: &DbusAddress) -> DbusResult<Self> {
        let kind = match address.get("path") {
            Some(_) => "path",
            None => match address.get("abstract") {
                Some(_) => "abstract",
                None => {
                    return Err(DbusError::invalid_address("unix address has no path or abstract"));
                }
            },
        };
        let raw = address.get(kind).unwrap();
        if raw.is_empty() {
            return Err(DbusError::invalid_address("empty unix socket name"));
        }
        if !address.is_unix() {
            return Err(DbusError::invalid_address("not a unix address"));
        }
        let mut sun = sockaddr_un {
            sun_family: libc::AF_UNIX as libc::sa_family_t,
            sun_path: [0; 108],
        };
        let bytes = raw.as_bytes();
        let copy_len = if kind == "abstract" && !bytes.starts_with(&[0]) {
            if bytes.len() + 1 >= sun.sun_path.len() {
                return Err(DbusError::invalid_address(alloc::format!(
                    "abstract socket name too long: {raw}"
                )));
            }
            // SAFETY: `sun_path` is zero-initialized; writing one
            // leading NUL plus the bytes is valid for
            // trivially-copyable types.
            unsafe {
                core::ptr::write_bytes(sun.sun_path.as_mut_ptr(), 0u8, 1);
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr() as *const libc::c_char,
                    sun.sun_path.as_mut_ptr().add(1),
                    bytes.len(),
                );
            }
            bytes.len() + 1
        } else {
            if bytes.len() >= sun.sun_path.len() {
                return Err(DbusError::invalid_address(alloc::format!(
                    "socket path too long: {raw}"
                )));
            }
            // SAFETY: `sun_path` is zero-initialized; `bytes`
            // are valid UTF-8 bytes and `copy_nonoverlapping`
            // is safe for trivially-copyable types.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr() as *const libc::c_char,
                    sun.sun_path.as_mut_ptr(),
                    bytes.len(),
                );
            }
            bytes.len()
        };
        let fd = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                0,
            )
        };
        if fd < 0 {
            return Err(DbusError::io("socket", unsafe { *libc::__errno_location() }));
        }
        let len = core::mem::size_of_val(&sun) as socklen_t - (sun.sun_path.len() - copy_len) as socklen_t;
        let err = unsafe { libc::connect(fd, &sun as *const _ as *const libc::sockaddr, len) };
        if err < 0 {
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EINPROGRESS || errno == libc::EWOULDBLOCK {
                let mut pollfd = libc::pollfd {
                    fd,
                    events: libc::POLLOUT,
                    revents: 0,
                };
                let n = unsafe { libc::poll(&mut pollfd, 1, -1) };
                if n < 0 {
                    let saved = errno;
                    let _ = unsafe { libc::close(fd) };
                    return Err(DbusError::io("poll", saved));
                }
                let mut error: c_int = 0;
                let mut len = core::mem::size_of_val(&error) as socklen_t;
                unsafe {
                    libc::getsockopt(
                        fd,
                        libc::SOL_SOCKET,
                        libc::SO_ERROR,
                        &mut error as *mut _ as *mut libc::c_void,
                        &mut len,
                    )
                };
                if error != 0 {
                    let saved = errno;
                    let _ = unsafe { libc::close(fd) };
                    return Err(DbusError::io("connect", saved));
                }
            } else {
                let saved = errno;
                let _ = unsafe { libc::close(fd) };
                return Err(DbusError::io("connect", saved));
            }
        }
        Ok(Self {
            fd,
            pending_fds: VecDeque::new(),
        })
    }

    /// Returns the raw file descriptor.
    #[must_use]
    pub fn fd(&self) -> c_int {
        self.fd
    }
}

/// Extracts the `SCM_RIGHTS` descriptors carried by `message`.
///
/// # Safety
///
/// `message` must describe a completed `recvmsg` call whose
/// `msg_control` buffer is still valid and initialized, as produced
/// by [`UnixTransport::read`]. The returned descriptors were
/// installed by the kernel for this process and are owned by the
/// caller, which must close or hand them over.
unsafe fn collect_rights(message: &libc::msghdr) -> Vec<i32> {
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

impl DbusTransport for UnixTransport {
    fn read(&mut self, buf: &mut [u8]) -> DbusResult<usize> {
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
            // correct lengths; `fd` is an open socket descriptor, so
            // `recvmsg` may write into all three.
            let n = unsafe { libc::recvmsg(self.fd, &mut message, 0) };
            if n < 0 {
                let errno = unsafe { *libc::__errno_location() };
                if errno == libc::EAGAIN || errno == libc::EWOULDBLOCK {
                    return Err(DbusError::WouldBlock);
                }
                if errno == libc::EINTR {
                    continue;
                }
                return Err(DbusError::io("recvmsg", errno));
            }
            let collected = unsafe { collect_rights(&message) };
            if message.msg_flags & libc::MSG_CTRUNC != 0 {
                // The control buffer overflowed, so part of the
                // descriptor list was lost. Nothing is recoverable:
                // release what arrived and fail loudly.
                close_fds(collected);
                return Err(DbusError::invalid_message(
                    "control message truncated: too many file descriptors",
                ));
            }
            // Queue the descriptors BEFORE the bytes that accompany
            // them are handed to the caller. On Linux `SCM_RIGHTS`
            // descriptors attach to a byte position and are returned
            // by the read that reaches it, so queueing them here (the
            // connection feeds them to the stream only after `read`
            // returns) keeps descriptor order aligned with message
            // order.
            self.pending_fds.extend(collected);
            return Ok(n as usize);
        }
    }

    fn write(&mut self, buf: &[u8]) -> DbusResult<usize> {
        loop {
            let n = unsafe { libc::write(self.fd, buf.as_ptr() as *const libc::c_void, buf.len()) };
            if n >= 0 {
                return Ok(n as usize);
            }
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EAGAIN || errno == libc::EWOULDBLOCK {
                return Err(DbusError::WouldBlock);
            }
            if errno == libc::EINTR {
                continue;
            }
            return Err(DbusError::io("write", errno));
        }
    }

    fn take_fds(&mut self) -> Vec<i32> {
        Vec::from(core::mem::take(&mut self.pending_fds))
    }

    fn write_with_fds(&mut self, buf: &[u8], fds: &[i32]) -> DbusResult<usize> {
        if fds.is_empty() {
            return self.write(buf);
        }
        if fds.len() > MAX_FDS_PER_MESSAGE {
            return Err(DbusError::unsupported(alloc::format!(
                "cannot send {} file descriptors in one message (limit {})",
                fds.len(),
                MAX_FDS_PER_MESSAGE
            )));
        }
        loop {
            let mut iov = libc::iovec {
                iov_base: buf.as_ptr().cast_mut().cast(),
                iov_len: buf.len(),
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
                return Err(DbusError::invalid_message(
                    "control buffer too small for file descriptors",
                ));
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
            // Only the bytes of the header we filled in belong to the
            // kernel; trailing zero padding would be parsed as a
            // broken control message.
            message.msg_controllen = unsafe { libc::CMSG_SPACE(bytes as u32) } as _;
            // SAFETY: `fd` is an open socket, `iov` and `control`
            // reference valid buffers for the duration of the call,
            // and the control message was fully initialized above.
            // `MSG_NOSIGNAL` suppresses `SIGPIPE` on a dead peer.
            let n = unsafe { libc::sendmsg(self.fd, &message, libc::MSG_NOSIGNAL) };
            if n < 0 {
                let errno = unsafe { *libc::__errno_location() };
                if errno == libc::EAGAIN || errno == libc::EWOULDBLOCK {
                    // Nothing was queued, so the caller still owns
                    // `fds` and must retry with them.
                    return Err(DbusError::WouldBlock);
                }
                if errno == libc::EINTR {
                    continue;
                }
                return Err(DbusError::io("sendmsg", errno));
            }
            // On success the kernel keeps its own references to the
            // descriptors; the caller closes its copies.
            return Ok(n as usize);
        }
    }

    fn wait(&mut self, timeout: Option<Duration>, interest: DbusPollEvents) -> DbusResult<DbusPollEvents> {
        // SAFETY: `fd` is a valid Unix file descriptor opened by
        // `UnixTransport::connect_to`; `pollfd` is a trivial struct.
        let mut fds = [libc::pollfd {
            fd: self.fd,
            events: if interest.contains(DbusPollEvents::READABLE) {
                libc::POLLIN
            } else {
                0
            } | if interest.contains(DbusPollEvents::WRITABLE) {
                libc::POLLOUT
            } else {
                0
            },
            revents: 0,
        }];
        let millis = timeout.map(|d| d.as_millis() as i64).unwrap_or(-1) as libc::c_int;
        // SAFETY: `fds` points to a valid single-element array and
        // `poll` writes only into `revents`.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), 1, millis) };
        if n < 0 {
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EINTR {
                return Ok(DbusPollEvents::EMPTY);
            }
            return Err(DbusError::io("poll", errno));
        }
        if n == 0 {
            return Err(DbusError::Timeout);
        }
        let revents = fds[0].revents as u32;
        let mut out = DbusPollEvents::EMPTY;
        if revents & (libc::POLLIN as u32) != 0 {
            out |= DbusPollEvents::READABLE;
        }
        if revents & (libc::POLLOUT as u32) != 0 {
            out |= DbusPollEvents::WRITABLE;
        }
        if revents & (libc::POLLHUP as u32 | libc::POLLERR as u32) != 0 {
            out |= DbusPollEvents::HANGUP;
        }
        Ok(out)
    }

    fn now_ms(&self) -> u64 {
        // SAFETY: `clock_gettime` is safe because `CLOCK_MONOTONIC`
        // is a valid clock ID and `ts` is a properly aligned
        // writable struct.
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        (ts.tv_sec as u64) * 1_000 + (ts.tv_nsec as u64) / 1_000_000
    }
}

impl Drop for UnixTransport {
    fn drop(&mut self) {
        // Descriptors nobody retrieved with `take_fds` are still
        // owned by the transport; release them with the socket.
        close_fds(core::mem::take(&mut self.pending_fds));
        if self.fd >= 0 {
            // SAFETY: `fd` was produced by `libc::socket` and is
            // therefore a valid descriptor; `close` releases it.
            unsafe { libc::close(self.fd) };
        }
    }
}

/// Returns the `uid` of the current process for `EXTERNAL`
/// authentication.
pub fn current_uid() -> u32 {
    // SAFETY: `getuid` reads the real user ID of the calling
    // process, which is safe and has no side effects.
    unsafe { libc::getuid() as u32 }
}

/// Returns environment variable `name`, if set.
fn get_env(name: &str) -> Option<String> {
    // SAFETY: `getenv` returns a pointer to an internal static
    // buffer valid until the next call to `getenv`. We copy the
    // string immediately and never call `getenv` again while
    // holding the reference, so the borrow is sound.
    let ptr = unsafe { libc::getenv(name.as_ptr() as *const libc::c_char) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `getenv` returns a NUL-terminated C string or a
    // null pointer; we checked for null above.
    let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
    cstr.to_str().ok().map(String::from)
}

/// Resolves default addresses for the system bus.
///
/// Checks `DBUS_SYSTEM_BUS_ADDRESS` first, then falls back to
/// the well-known socket paths.
pub fn resolve_system_addresses() -> Vec<DbusAddress> {
    if let Some(env) = get_env("DBUS_SYSTEM_BUS_ADDRESS")
        && let Ok(addresses) = DbusAddress::parse_all(&env)
    {
        return addresses;
    }
    [SYSTEM_BUS_SOCKET, SYSTEM_BUS_SOCKET_LEGACY]
        .iter()
        .filter_map(|path| DbusAddress::parse(&alloc::format!("unix:path={path}")).ok())
        .collect()
}

/// Resolves default addresses for the session bus.
///
/// Checks `DBUS_SESSION_BUS_ADDRESS` first, then falls back to
/// `$XDG_RUNTIME_DIR/bus`.
pub fn resolve_session_addresses() -> DbusResult<Vec<DbusAddress>> {
    if let Some(env) = get_env("DBUS_SESSION_BUS_ADDRESS")
        && let Ok(addresses) = DbusAddress::parse_all(&env)
    {
        return Ok(addresses);
    }
    if let Some(runtime) = get_env("XDG_RUNTIME_DIR") {
        let path = alloc::format!("{runtime}/{SESSION_BUS_FILE}");
        return Ok(vec![DbusAddress::parse(&path)?]);
    }
    Err(DbusError::invalid_address("no session bus address configured"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dbus_conn::Connection;
    use crate::dbus_message::DbusMessage;

    /// Creates a non-blocking `AF_UNIX` stream socket pair.
    fn socket_pair() -> [i32; 2] {
        let mut pair = [-1i32; 2];
        // SAFETY: `pair` is a valid two element array; on success the
        // kernel fills both entries with connected descriptors.
        let rc = unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_NONBLOCK,
                0,
                pair.as_mut_ptr(),
            )
        };
        // Justified: a socketpair must not fail in the test runner.
        assert_eq!(rc, 0);
        pair
    }

    /// Creates a pipe and returns `[read_end, write_end]`.
    fn pipe_pair() -> [i32; 2] {
        let mut pair = [-1i32; 2];
        // SAFETY: `pair` is a valid two element array; on success the
        // kernel fills both entries with open descriptors.
        // Justified: a pipe must not fail in the test runner.
        assert_eq!(unsafe { libc::pipe(pair.as_mut_ptr()) }, 0);
        pair
    }

    #[test]
    fn now_ms_is_increasing() {
        let transport = UnixTransport {
            fd: -1,
            pending_fds: VecDeque::new(),
        };
        let a = transport.now_ms();
        let b = transport.now_ms();
        assert!(b >= a);
    }

    #[test]
    fn poll_events_flags() {
        let events = DbusPollEvents::READABLE | DbusPollEvents::WRITABLE;
        assert!(events.contains(DbusPollEvents::READABLE));
        assert!(events.contains(DbusPollEvents::WRITABLE));
    }

    #[test]
    fn sendmsg_recvmsg_roundtrip_passes_file_descriptors() {
        let pair = socket_pair();
        let mut sender = UnixTransport {
            fd: pair[0],
            pending_fds: VecDeque::new(),
        };
        let mut receiver = UnixTransport {
            fd: pair[1],
            pending_fds: VecDeque::new(),
        };

        let pipe_a = pipe_pair();
        let pipe_b = pipe_pair();
        let fds = [pipe_a[0], pipe_b[0]];

        let payload = b"a method call carrying descriptors";
        // Justified: the socket buffer is empty, so this must succeed.
        let written = sender.write_with_fds(payload, &fds).unwrap();
        assert_eq!(written, payload.len());
        // SAFETY: the kernel took its own references during
        // `sendmsg`, so the sender copies can be released now.
        unsafe {
            libc::close(pipe_a[0]);
            libc::close(pipe_b[0]);
        }

        let mut buf = [0u8; 64];
        // Justified: the payload is already queued in the socket.
        let n = receiver.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], payload);
        // Descriptors arrive in the order they were sent. Their
        // numbers are chosen by the kernel, so only identity and
        // count are meaningful here, never equality with `fds`.
        // Justified: the sender attached exactly these two.
        let received = receiver.take_fds();
        assert_eq!(received.len(), fds.len());
        for &fd in &received {
            // SAFETY: `fcntl` only inspects the descriptor.
            assert!(unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0);
        }

        // Each received descriptor is a working duplicate: fill both
        // pipes with distinguishable bytes and verify them in send
        // order, which `SCM_RIGHTS` preserves.
        // SAFETY: the write ends are live, so each pipe accepts one
        // byte without blocking.
        assert_eq!(unsafe { libc::write(pipe_a[1], [42u8].as_ptr().cast(), 1) }, 1);
        assert_eq!(unsafe { libc::write(pipe_b[1], [43u8].as_ptr().cast(), 1) }, 1);
        for (index, &fd) in received.iter().enumerate() {
            let mut got = [0u8; 1];
            // SAFETY: `fd` is a live read end duplicated by the
            // kernel during `sendmsg` and its pipe already holds a
            // byte, so this read completes immediately.
            assert_eq!(unsafe { libc::read(fd, got.as_mut_ptr().cast(), got.len()) }, 1);
            // The first descriptor travels with the first payload.
            if index == 0 {
                assert_eq!(got, [42u8]);
            } else {
                assert_eq!(got, [43u8]);
            }
        }

        for fd in received {
            // SAFETY: the descriptors were received for this test and
            // are not used anymore.
            unsafe { libc::close(fd) };
        }
        // SAFETY: closing the write ends this test still owns.
        unsafe {
            libc::close(pipe_a[1]);
            libc::close(pipe_b[1]);
        }
    }

    #[test]
    fn connection_delivers_fds_with_a_method_call_end_to_end() {
        let pair = socket_pair();
        let mut sender = Connection::new(UnixTransport {
            fd: pair[0],
            pending_fds: VecDeque::new(),
        });
        let mut receiver = Connection::new(UnixTransport {
            fd: pair[1],
            pending_fds: VecDeque::new(),
        });
        let pipe = pipe_pair();

        let mut call = DbusMessage::method_call("a.b.Service", "/a/b", "a.b.Iface", "Open").unwrap();
        call.build_body(|body| {
            body.write_fd(0)?;
            body.write_str("/tmp/file")
        })
        // Justified: the closure writes the declared `hs` body.
        .unwrap();
        call.set_fds(vec![pipe[0]]);
        // Justified: the message encodes and the socket is empty.
        sender.send_message(call).unwrap();
        sender.flush().unwrap();

        // Justified: the bytes are already queued on the socket.
        // SAFETY: `fcntl` only inspects the descriptor.
        assert_eq!(unsafe { libc::fcntl(pipe[0], libc::F_GETFD) }, -1);

        // Justified: the complete message is queued on the socket.
        let message = receiver.try_recv().unwrap().unwrap();
        assert_eq!(message.member(), Some("Open"));
        assert_eq!(message.unix_fds(), 1);
        assert_eq!(message.fds().len(), 1);
        let received = message.fds()[0];
        let mut reader = message.body_reader();
        assert_eq!(reader.read_fd().unwrap(), 0);
        assert_eq!(reader.read_str().unwrap(), "/tmp/file");

        // Data flows through the descriptor that crossed the socket.
        let byte = [9u8];
        // SAFETY: `pipe[1]` is the write end of a live pipe.
        assert_eq!(
            unsafe { libc::write(pipe[1], byte.as_ptr().cast(), byte.len()) },
            1
        );
        let mut got = [0u8; 1];
        // SAFETY: `received` is a live read end duplicated by the
        // kernel when the message was sent.
        assert_eq!(
            unsafe { libc::read(received, got.as_mut_ptr().cast(), got.len()) },
            1
        );
        assert_eq!(got, byte);

        // Dropping the message releases the descriptor it owns.
        drop(message);
        // SAFETY: `fcntl` only inspects the descriptor.
        assert_eq!(unsafe { libc::fcntl(received, libc::F_GETFD) }, -1);
        // SAFETY: closing the write end this test still owns.
        unsafe { libc::close(pipe[1]) };
    }
}
