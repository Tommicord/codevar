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

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use core::time::Duration;
use libc::{c_int, sockaddr_un, socklen_t};

use crate::dbus_addr::{
    DbusAddress, SESSION_BUS_FILE, SYSTEM_BUS_SOCKET, SYSTEM_BUS_SOCKET_LEGACY,
};
use crate::dbus_error::{DbusError, DbusResult};
use crate::dbus_transport::{DbusPollEvents, DbusTransport};

/// Default timeout for a method call, matching the D-Bus
/// specification default of 25 seconds.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_millis(25_000);

/// A [`DbusTransport`] backed by a Unix domain socket.
pub struct UnixTransport {
    fd: c_int,
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
                    return Err(DbusError::invalid_address(
                        "unix address has no path or abstract",
                    ));
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
            sun_path: [0i8; 108],
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
                    bytes.as_ptr() as *const i8,
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
                    bytes.as_ptr() as *const i8,
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
            return Err(DbusError::io("socket", unsafe {
                *libc::__errno_location()
            }));
        }
        let len = core::mem::size_of_val(&sun) as socklen_t
            - (sun.sun_path.len() - copy_len) as socklen_t;
        let err =
            unsafe { libc::connect(fd, &sun as *const _ as *const libc::sockaddr, len) };
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
        Ok(Self { fd })
    }

    /// Returns the raw file descriptor.
    #[must_use]
    pub fn fd(&self) -> c_int {
        self.fd
    }
}

impl DbusTransport for UnixTransport {
    fn read(&mut self, buf: &mut [u8]) -> DbusResult<usize> {
        loop {
            let n = unsafe {
                libc::read(self.fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len())
            };
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
            return Err(DbusError::io("read", errno));
        }
    }

    fn write(&mut self, buf: &[u8]) -> DbusResult<usize> {
        loop {
            let n = unsafe {
                libc::write(self.fd, buf.as_ptr() as *const libc::c_void, buf.len())
            };
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

    fn wait(
        &mut self,
        timeout: Option<Duration>,
        interest: DbusPollEvents,
    ) -> DbusResult<DbusPollEvents> {
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
    Err(DbusError::invalid_address(
        "no session bus address configured",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_ms_is_increasing() {
        let transport = UnixTransport { fd: -1 };
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
}
