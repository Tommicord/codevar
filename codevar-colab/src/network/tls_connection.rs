//! Copyright 2026 Codevar
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

//! Shared connection state for client and server roles.

use crate::network::tls_alert::{Alert, AlertDescription, AlertLevel};
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_handshake::HandshakeReassembly;
use crate::network::tls_ids::{CipherSuite, ContentType, ProtocolVersion};
use crate::network::tls_record::{RecordLayer, RecordProtection};
use crate::network::tls_transcript::Transcript;
use std::collections::VecDeque;

/// High-level connection lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Handshake in progress.
    Handshaking,
    /// Application data may be exchanged.
    Connected,
    /// Local close_notify sent; waiting for peer or already draining.
    Closing,
    /// Connection fully closed.
    Closed,
}

/// Shared buffers and record-layer state.
pub struct CommonState {
    /// Record layer.
    pub record: RecordLayer,
    /// Handshake message reassembly buffer.
    pub hs_rx: HandshakeReassembly,
    /// Decrypted application data waiting for the user.
    pub app_rx: VecDeque<u8>,
    /// Lifecycle state.
    pub state: ConnectionState,
    /// Negotiated protocol version.
    pub version: Option<ProtocolVersion>,
    /// Negotiated cipher suite.
    pub suite: Option<CipherSuite>,
    /// Selected ALPN protocol.
    pub alpn: Option<Vec<u8>>,
    /// Transcript hash (rebound after suite selection).
    pub transcript: Transcript,
    /// True after peer `close_notify`.
    pub peer_closed: bool,
    /// True after local `close_notify` queued.
    pub local_closed: bool,
    /// Buffered fatal error to surface after alert is queued.
    pub error: Option<TlsError>,
}

impl CommonState {
    /// Creates a new common state with a provisional SHA-256 transcript.
    #[must_use]
    pub fn new() -> Self {
        Self {
            record: RecordLayer::new(),
            hs_rx: HandshakeReassembly::default(),
            app_rx: VecDeque::new(),
            state: ConnectionState::Handshaking,
            version: None,
            suite: None,
            alpn: None,
            transcript: Transcript::new(crate::network::tls_ids::HashAlgorithm::Sha256),
            peer_closed: false,
            local_closed: false,
            error: None,
        }
    }

    /// Returns true while the handshake has not completed.
    #[must_use]
    pub fn is_handshaking(&self) -> bool {
        self.state == ConnectionState::Handshaking
    }

    /// Returns true when outbound ciphertext is pending.
    #[must_use]
    pub fn wants_write(&self) -> bool {
        self.record.wants_write()
    }

    /// Returns true when the connection expects more network input.
    #[must_use]
    pub fn wants_read(&self) -> bool {
        !self.peer_closed && self.state != ConnectionState::Closed
    }

    /// Queues a cleartext or protected alert record.
    pub fn send_alert(&mut self, alert: Alert) -> TlsResult<()> {
        let payload = alert.encode();
        self.record.write_raw(ContentType::Alert, &payload)?;
        if alert.level == AlertLevel::Fatal
            || alert.description == AlertDescription::CloseNotify
        {
            self.local_closed = true;
            if alert.description == AlertDescription::CloseNotify {
                self.state = ConnectionState::Closing;
            } else {
                self.state = ConnectionState::Closed;
            }
        }
        Ok(())
    }

    /// Queues `close_notify`.
    pub fn send_close_notify(&mut self) -> TlsResult<()> {
        if self.local_closed {
            return Ok(());
        }
        self.send_alert(Alert::warning(AlertDescription::CloseNotify))
    }

    /// Handles a decrypted alert payload.
    pub fn handle_alert(&mut self, payload: &[u8]) -> TlsResult<()> {
        let alert = Alert::decode(payload)?;
        if alert.description == AlertDescription::CloseNotify {
            self.peer_closed = true;
            self.state = ConnectionState::Closed;
            return Err(TlsError::Closed);
        }
        self.state = ConnectionState::Closed;
        Err(TlsError::PeerAlert {
            level: alert.level,
            description: alert.description,
        })
    }

    /// Appends application plaintext to the user-facing RX queue.
    pub fn queue_appdata(&mut self, data: &[u8]) {
        self.app_rx.extend(data.iter().copied());
    }

    /// Reads buffered application data into `buf`.
    pub fn read_app(&mut self, buf: &mut [u8]) -> TlsResult<usize> {
        if self.app_rx.is_empty() {
            if self.peer_closed || self.state == ConnectionState::Closed {
                return Ok(0);
            }
            if self.is_handshaking() {
                return Err(TlsError::HandshakeNotComplete);
            }
            return Err(TlsError::WouldBlock);
        }
        let n = buf.len().min(self.app_rx.len());
        for slot in buf.iter_mut().take(n) {
            *slot = self.app_rx.pop_front().unwrap_or(0);
        }
        Ok(n)
    }

    /// Encrypts and queues application data.
    pub fn write_app(&mut self, data: &[u8]) -> TlsResult<()> {
        if self.is_handshaking() {
            return Err(TlsError::HandshakeNotComplete);
        }
        if self.local_closed || self.state == ConnectionState::Closed {
            return Err(TlsError::Closed);
        }
        // Fragment to MAX_FRAGMENT_LENGTH.
        let mut offset = 0;
        while offset < data.len() {
            let end =
                (offset + crate::network::tls_ids::MAX_FRAGMENT_LENGTH).min(data.len());
            self.record
                .write_raw(ContentType::ApplicationData, &data[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    /// Sends a ChangeCipherSpec record (TLS 1.2 / TLS 1.3 middlebox CCS).
    pub fn send_ccs(&mut self) -> TlsResult<()> {
        self.record.write_raw(ContentType::ChangeCipherSpec, &[1])
    }

    /// Installs write keys for the negotiated version / suite.
    pub fn install_write_keys(
        &mut self,
        keys: crate::network::tls_record::TrafficKeys,
        version: ProtocolVersion,
        suite: CipherSuite,
    ) {
        let mode = RecordProtection::from_suite(version, suite.aead_algorithm());
        self.record.set_write_keys(keys, mode);
    }

    /// Installs read keys for the negotiated version / suite.
    pub fn install_read_keys(
        &mut self,
        keys: crate::network::tls_record::TrafficKeys,
        version: ProtocolVersion,
        suite: CipherSuite,
    ) {
        let mode = RecordProtection::from_suite(version, suite.aead_algorithm());
        self.record.set_read_keys(keys, mode);
    }

    /// Queues a handshake message ciphertext / cleartext and adds it to the transcript.
    pub fn send_handshake_raw(&mut self, message: &[u8]) -> TlsResult<()> {
        self.transcript.add_message(message);
        // Fragment handshake across records if needed.
        let mut offset = 0;
        while offset < message.len() {
            let end = (offset + crate::network::tls_ids::MAX_FRAGMENT_LENGTH)
                .min(message.len());
            self.record
                .write_raw(ContentType::Handshake, &message[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    /// On fatal local error, queue the matching alert when possible.
    pub fn fail(&mut self, err: TlsError) -> TlsError {
        if let Some(desc) = err.alert_description() {
            if desc != AlertDescription::CloseNotify {
                let _ = self.send_alert(Alert::fatal(desc));
            }
        }
        self.state = ConnectionState::Closed;
        self.error = Some(err.clone());
        err
    }
}

impl Default for CommonState {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of I/O interest after processing.
#[derive(Debug, Clone, Copy)]
pub struct IoState {
    /// Plaintext application bytes available to read.
    pub plaintext_bytes_to_read: usize,
    /// Ciphertext bytes waiting to be written to the transport.
    pub tls_bytes_to_write: usize,
    /// Handshake still in progress.
    pub handshaking: bool,
}
