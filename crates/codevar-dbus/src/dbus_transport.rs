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

//! Transport abstraction used by [`Connection`].
//!
//! The trait abstracts the underlying socket or in-memory channel
//! so that the protocol stack can be tested without a real bus
//! daemon. Implementations provide [`wait`] so that callers can
//! block until the transport is ready without spinning.

use alloc::vec::Vec;
use core::time::Duration;

use crate::dbus_error::{DbusError, DbusResult};

/// Closes every descriptor in `fds`, ignoring errors.
///
/// Descriptors that are already closed or were never valid (`fd < 0`
/// is skipped) do not cause failures, so this is safe to call from
/// cleanup paths.
#[cfg(all(unix, not(target_arch = "wasm32")))]
pub(crate) fn close_fds(fds: impl IntoIterator<Item = i32>) {
    for fd in fds {
        if fd >= 0 {
            // SAFETY: `fd` comes from the kernel (`recvmsg`) or from
            // the caller that created it; `close` releases that
            // descriptor exactly once because ownership was
            // transferred to this function. Errors are ignored, the
            // descriptor is unusable afterwards either way.
            unsafe { libc::close(fd) };
        }
    }
}

/// Closes every descriptor in `fds`, ignoring errors.
///
/// Platforms without Unix file descriptors never produce descriptors,
/// so this is a no-op there.
#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
pub(crate) fn close_fds(fds: impl IntoIterator<Item = i32>) {
    let _ = fds;
}

/// Readiness bits reported by a transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DbusPollEvents(u32);

impl DbusPollEvents {
    /// No events ready.
    pub const EMPTY: Self = Self(0);
    /// The transport is readable.
    pub const READABLE: Self = Self(1);
    /// The transport is writable.
    pub const WRITABLE: Self = Self(2);
    /// The peer closed the connection.
    pub const HANGUP: Self = Self(4);
    /// An error condition is pending.
    pub const ERROR: Self = Self(8);
    /// All event bits.
    pub const ALL: Self = Self(0x0f);

    /// Creates an event set from raw bits.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bits.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when every bit of `other` is set.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns `true` when no bit is set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for DbusPollEvents {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for DbusPollEvents {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A byte transport used by [`Connection`].
///
/// Implementations may wrap a Unix domain socket, an in-memory
/// channel, or any other reliable byte stream. The trait supplies
/// [`wait`] so that a caller can block until the transport is ready
/// without busy polling.
///
/// [`wait`] supplies a monotonic clock in milliseconds through
/// [`now_ms`] so that callers can enforce round-trip deadlines
/// without depending on the operating system.
pub trait DbusTransport {
    /// Reads up to `buf.len()` bytes from the transport.
    ///
    /// Returns `Ok(0)` when the peer closed the connection,
    /// [`DbusError::WouldBlock`] when no data is available, or the
    /// number of bytes read otherwise.
    fn read(&mut self, buf: &mut [u8]) -> DbusResult<usize>;

    /// Writes `buf` to the transport.
    ///
    /// Returns the number of bytes accepted, which may be less
    /// than `buf.len()` for non-blocking transports.
    fn write(&mut self, buf: &[u8]) -> DbusResult<usize>;

    /// Waits until one of `interest` is ready or `timeout` elapses.
    ///
    /// A `timeout` of `None` blocks indefinitely, `Some(Duration::ZERO)`
    /// polls without blocking. Returns the events that became ready.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Timeout`] when the deadline elapses
    /// before any requested event is ready.
    fn wait(
        &mut self,
        timeout: Option<Duration>,
        interest: DbusPollEvents,
    ) -> DbusResult<DbusPollEvents>;

    /// Returns the current monotonic time in milliseconds.
    ///
    /// The value only increases and is used to enforce
    /// round-trip deadlines in [`Connection`].
    fn now_ms(&self) -> u64;

    /// Takes the file descriptors received by the transport, if any.
    ///
    /// Implementations that support `SCM_RIGHTS` collect the
    /// descriptors reported by the most recent [`read`](Self::read)
    /// calls and return them here in arrival order. The returned
    /// descriptors are owned by the caller, which is responsible for
    /// closing them. The default implementation returns an empty
    /// list.
    fn take_fds(&mut self) -> Vec<i32> {
        Vec::new()
    }

    /// Writes `buf` to the transport, attaching `fds` to the byte
    /// position where `buf` starts in the stream.
    ///
    /// When `fds` is empty this is equivalent to [`write`](Self::write).
    /// Otherwise, on `Ok(n)` with `n > 0` the kernel has taken its own
    /// reference to every descriptor in `fds`, so the caller must
    /// close its copies; on [`DbusError::WouldBlock`], `Ok(0)` or an
    /// error the caller keeps ownership of `fds` and must retry with
    /// them. Transports never close the descriptors themselves.
    ///
    /// The default implementation forwards to [`write`](Self::write)
    /// when `fds` is empty and reports [`DbusError::Unsupported`]
    /// otherwise, so transports without descriptor passing keep
    /// compiling.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Unsupported`] when `fds` is not empty and
    /// the transport cannot pass descriptors, plus whatever
    /// [`write`](Self::write) returns.
    fn write_with_fds(&mut self, buf: &[u8], fds: &[i32]) -> DbusResult<usize> {
        if fds.is_empty() {
            self.write(buf)
        } else {
            Err(DbusError::unsupported(
                "this transport does not support file descriptor passing",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    #[allow(dead_code)]
    struct FakeTransport {
        events: DbusPollEvents,
        now: u64,
    }

    impl DbusTransport for FakeTransport {
        fn read(&mut self, _buf: &mut [u8]) -> DbusResult<usize> {
            Ok(0)
        }
        fn write(&mut self, _buf: &[u8]) -> DbusResult<usize> {
            Ok(0)
        }
        fn wait(
            &mut self,
            timeout: Option<Duration>,
            _interest: DbusPollEvents,
        ) -> DbusResult<DbusPollEvents> {
            assert!(timeout.is_none());
            Ok(self.events)
        }
        fn now_ms(&self) -> u64 {
            self.now
        }
    }

    #[test]
    fn poll_events_empty_and_bits() {
        assert!(DbusPollEvents::EMPTY.is_empty());
        assert_eq!(DbusPollEvents::READABLE.bits(), 1);
        assert!(DbusPollEvents::ALL.contains(DbusPollEvents::ERROR));
        assert_eq!(
            DbusPollEvents::READABLE | DbusPollEvents::WRITABLE,
            DbusPollEvents::from_bits(3)
        );
    }

    #[test]
    fn default_fd_methods_keep_plain_transports_working() {
        let mut transport = FakeTransport {
            events: DbusPollEvents::EMPTY,
            now: 0,
        };
        // Writing without descriptors forwards to `write`.
        assert_eq!(transport.write_with_fds(b"abc", &[]), Ok(0));
        // A transport without descriptor passing refuses fd writes
        // but keeps the descriptors (it never closes them).
        let result = transport.write_with_fds(b"abc", &[3]);
        assert!(matches!(result, Err(DbusError::Unsupported(_))));
        // Plain transports never queue descriptors.
        assert!(transport.take_fds().is_empty());
    }

    #[test]
    fn close_fds_ignores_negative_placeholders() {
        // Negative values are placeholders, never real descriptors.
        close_fds([-1, -2]);
        close_fds(core::iter::empty::<i32>());
    }
}
