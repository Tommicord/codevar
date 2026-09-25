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

//! SASL authentication handshake for the D-Bus protocol.
//!
//! The handshake begins with a nul byte and ends with `BEGIN`.
//! The client authenticates using [`EXTERNAL`](AuthSession::with_external)
//! and falls back to [`ANONYMOUS`](AuthSession::with_anonymous) if the
//! server rejects the requested mechanism.

use alloc::string::String;
use alloc::vec::Vec;

use crate::dbus_error::{DbusError, DbusResult};

const HEX_TABLE: &[u8; 16] = b"0123456789abcdef";

/// The outcome of a call to [`AuthSession::poll`].
#[derive(Debug, PartialEq, Eq)]
pub enum AuthPoll {
    /// Write the returned bytes to the socket.
    Write(Vec<u8>),
    /// Read more bytes from the socket before calling [`poll`] again.
    Read,
    /// Authentication completed successfully; start sending messages.
    Done,
}

/// States of the client-side authentication handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthState {
    /// Awaiting the caller's first [`poll`] to emit the initial
    /// `AUTH` command.
    Start,
    /// Waiting for the server's response to an `AUTH EXTERNAL`
    /// command.
    AwaitExternal,
    /// Waiting for the server's response to an `AUTH ANONYMOUS`
    /// command.
    AwaitAnonymous,
    /// The server accepted; the `BEGIN` bytes are queued.
    SendBegin,
    /// Handshake finished.
    Finished,
}

/// Client-side SASL authentication session.
///
/// The caller drives the handshake through [`poll`] and [`feed`]:
///
/// ```text
/// loop {
///     match session.poll() {
///         AuthPoll::Write(bytes) => transport.write_all(bytes),
///         AuthPoll::Read         => transport.read(&mut buf),
///         AuthPoll::Done         => break,
///     }
/// }
/// ```
pub struct AuthSession {
    state: AuthState,
    pending: Vec<u8>,
    rx: Vec<u8>,
    uid: u32,
}

impl AuthSession {
    /// Creates a session that authenticates via `EXTERNAL` for
    /// `uid`.
    ///
    /// The first [`poll`] call emits a nul byte followed by
    /// `AUTH EXTERNAL <hex uid>\r\n`.
    pub fn with_external(uid: u32) -> Self {
        Self {
            state: AuthState::Start,
            pending: Vec::new(),
            rx: Vec::new(),
            uid,
        }
    }

    /// Creates a session that first attempts `EXTERNAL`, then
    /// falls back to `ANONYMOUS`.
    ///
    /// The first [`poll`] call emits the `EXTERNAL` request. If the
    /// server rejects it, the session automatically switches to an
    /// `AUTH ANONYMOUS\r\n` request.
    pub fn with_external_then_anonymous(uid: u32) -> Self {
        Self::with_external(uid)
    }

    /// Returns the bytes that must be written to the transport,
    /// if any are pending.
    pub fn poll(&mut self) -> DbusResult<AuthPoll> {
        if !self.pending.is_empty() {
            if matches!(self.state, AuthState::SendBegin) {
                self.state = AuthState::Finished;
            }
            return Ok(AuthPoll::Write(core::mem::take(&mut self.pending)));
        }
        match self.state {
            AuthState::Start => {
                let mut request = Vec::with_capacity(32);
                request.push(0);
                request.extend_from_slice(b"AUTH EXTERNAL ");
                push_ascii_uid_hex(&mut request, self.uid);
                request.extend_from_slice(b"\r\n");
                self.state = AuthState::AwaitExternal;
                Ok(AuthPoll::Write(request))
            }
            AuthState::AwaitExternal | AuthState::AwaitAnonymous => Ok(AuthPoll::Read),
            AuthState::SendBegin => {
                self.state = AuthState::Finished;
                Ok(AuthPoll::Write(b"BEGIN\r\n".to_vec()))
            }
            AuthState::Finished => Ok(AuthPoll::Done),
        }
    }

    /// Feeds bytes received from the transport into the session.
    ///
    /// Complete lines terminated by `\n` are processed. Returns
    /// `Ok(())` while the handshake is in progress.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Auth`] on protocol violations,
    /// malformed responses, or exhausted mechanisms.
    pub fn feed(&mut self, data: &[u8]) -> DbusResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let state = self.state;
        match state {
            AuthState::Start | AuthState::Finished => {
                return Err(DbusError::auth(
                    "authentication data received in unexpected state",
                ));
            }
            AuthState::AwaitExternal | AuthState::AwaitAnonymous => {}
            AuthState::SendBegin => {
                return Err(DbusError::auth("authentication data received after OK"));
            }
        }
        let mut rx = core::mem::take(&mut self.rx);
        rx.extend_from_slice(data);
        let mut lines = Vec::new();
        let mut pos = 0;
        while pos < rx.len() {
            if let Some(end) = rx[pos..].iter().position(|&b| b == b'\n') {
                let line_end = pos + end;
                let line = core::str::from_utf8(&rx[pos..=line_end])
                    .map_err(|_| DbusError::auth("non-UTF-8 auth line"))?;
                let trimmed = line
                    .strip_suffix('\n')
                    .and_then(|s| s.strip_suffix('\r'))
                    .unwrap_or(line);
                let owned = String::from(trimmed);
                rx.drain(pos..=line_end);
                pos = 0;
                if !owned.is_empty() {
                    lines.push(owned);
                }
            } else {
                break;
            }
        }
        for line in &lines {
            self.handle_line(line)?;
        }
        self.rx = rx;
        Ok(())
    }

    /// Returns `true` when the handshake is finished.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self.state, AuthState::Finished)
    }

    fn handle_line(&mut self, line: &str) -> DbusResult<()> {
        if line.starts_with("OK ") || line == "OK" {
            self.state = AuthState::SendBegin;
            self.pending = b"BEGIN\r\n".to_vec();
            return Ok(());
        }
        if line.starts_with("REJECTED") {
            let rest = line.strip_prefix("REJECTED").unwrap_or("");
            let mechanisms = rest.trim();
            match self.state {
                AuthState::AwaitExternal => {
                    self.state = AuthState::AwaitAnonymous;
                    self.pending = b"AUTH ANONYMOUS\r\n".to_vec();
                    Ok(())
                }
                AuthState::AwaitAnonymous => Err(DbusError::auth(alloc::format!(
                    "server rejected all mechanisms ({mechanisms})"
                ))),
                _ => Err(DbusError::auth("REJECTED received out of state")),
            }
        } else if line.starts_with("ERROR") {
            Err(DbusError::auth(alloc::format!("server reported error: {line}")))
        } else {
            Err(DbusError::auth(alloc::format!(
                "unexpected server response: {line}"
            )))
        }
    }
}

fn push_ascii_uid_hex(out: &mut Vec<u8>, uid: u32) {
    let decimal = alloc::format!("{uid}");
    for byte in decimal.as_bytes() {
        out.push(HEX_TABLE[(byte >> 4) as usize]);
        out.push(HEX_TABLE[(byte & 0x0f) as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(session: &mut AuthSession, steps: &[&[u8]]) -> DbusResult<()> {
        for data in steps {
            session.feed(data)?;
        }
        Ok(())
    }

    #[test]
    fn external_success() -> DbusResult<()> {
        let mut session = AuthSession::with_external(1000);
        let first = session.poll().unwrap();
        match first {
            AuthPoll::Write(bytes) => {
                assert_eq!(bytes[0], 0);
                assert!(bytes.starts_with(b"\0AUTH EXTERNAL "));
                assert!(bytes.ends_with(b"\r\n"));
            }
            other => panic!("expected Write, got {other:?}"),
        }
        fake(&mut session, &[&b"OK 1234deadbeef\r\n"[..], &b""[..]])?;
        let begin = session.poll().unwrap();
        assert_eq!(begin, AuthPoll::Write(b"BEGIN\r\n".to_vec()));
        assert_eq!(session.poll().unwrap(), AuthPoll::Done);
        Ok(())
    }

    #[test]
    fn external_falls_back_to_anonymous() -> DbusResult<()> {
        let mut session = AuthSession::with_external(1000);
        session.poll().unwrap();
        fake(
            &mut session,
            &[&b"REJECTED KERBEROS_V4\r\n"[..], &b"OK deadbeef\r\n"[..]],
        )?;
        let begin = session.poll().unwrap();
        assert_eq!(begin, AuthPoll::Write(b"BEGIN\r\n".to_vec()));
        Ok(())
    }

    #[test]
    fn exhausted_mechanisms_fails() {
        let mut session = AuthSession::with_external(1000);
        session.poll().unwrap();
        // The first REJECTED switches to the anonymous mechanism.
        session.feed(b"REJECTED KERBEROS_V4\r\n").unwrap();
        let err = session
            .feed(b"REJECTED ANONYMOUS\r\n")
            .unwrap_err();
        assert!(err.is_auth());
    }

    #[test]
    fn ok_without_space_is_accepted() -> DbusResult<()> {
        let mut session = AuthSession::with_external(1000);
        session.poll().unwrap();
        fake(&mut session, &[&b"OK\r\n"[..]])?;
        Ok(())
    }

    #[test]
    fn rejects_garbage_lines() {
        let mut session = AuthSession::with_external(1000);
        session.poll().unwrap();
        let err = session.feed(b"NONSENSE\r\n").unwrap_err();
        assert!(err.is_auth());
    }

    #[test]
    fn split_line_across_feeds() {
        let mut session = AuthSession::with_external(1000);
        session.poll().unwrap();
        session.feed(b"OK ").unwrap();
        assert!(!session.is_finished());
        session.feed(b"deadbeef\r\n").unwrap();
        assert!(!session.is_finished());
        let begin = session.poll().unwrap();
        assert_eq!(begin, AuthPoll::Write(b"BEGIN\r\n".to_vec()));
        assert!(session.is_finished());
    }

    #[test]
    fn feed_after_finished_is_an_error() {
        let mut session = AuthSession::with_external(1000);
        session.poll().unwrap();
        session.feed(b"OK deadbeef\r\n").unwrap();
        assert!(session.feed(b"DATA x\r\n").is_err());
    }
}
