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

//! Shared connection state for client and server roles.

use crate::tls_alert::{Alert, AlertDescription, AlertLevel};
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_handshake::HandshakeReassembly;
use crate::tls_ids::{CipherSuite, ContentType, ProtocolVersion};
use crate::tls_record::{RecordLayer, RecordProtection};
use crate::tls_transcript::Transcript;
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
            transcript: Transcript::new(crate::tls_ids::HashAlgorithm::Sha256),
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
        self.record
            .write_raw(ContentType::Alert, &payload)?;
        if alert.level == AlertLevel::Fatal || alert.description == AlertDescription::CloseNotify {
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
            let end = (offset + crate::tls_ids::MAX_FRAGMENT_LENGTH).min(data.len());
            self.record
                .write_raw(ContentType::ApplicationData, &data[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    /// Sends a ChangeCipherSpec record (TLS 1.2 / TLS 1.3 middlebox CCS).
    pub fn send_ccs(&mut self) -> TlsResult<()> {
        self.record
            .write_raw(ContentType::ChangeCipherSpec, &[1])
    }

    /// Installs write keys for the negotiated version / suite.
    pub fn install_write_keys(
        &mut self,
        keys: crate::tls_record::TrafficKeys,
        version: ProtocolVersion,
        suite: CipherSuite,
    ) {
        let mode = RecordProtection::from_suite(version, suite.aead_algorithm());
        self.record.set_write_keys(keys, mode);
    }

    /// Installs read keys for the negotiated version / suite.
    pub fn install_read_keys(
        &mut self,
        keys: crate::tls_record::TrafficKeys,
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
            let end = (offset + crate::tls_ids::MAX_FRAGMENT_LENGTH).min(message.len());
            self.record
                .write_raw(ContentType::Handshake, &message[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    /// On fatal local error, queue the matching alert when possible.
    pub fn fail(&mut self, err: TlsError) -> TlsError {
        if let Some(desc) = err.alert_description()
            && desc != AlertDescription::CloseNotify
        {
            let _ = self.send_alert(Alert::fatal(desc));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_alert::Alert;
    use crate::tls_ids::MAX_FRAGMENT_LENGTH;

    fn cleartext_record(ty: ContentType, payload: &[u8]) -> Vec<u8> {
        crate::tls_record::encode_cleartext_record(ty, ProtocolVersion::Tls12, payload)
    }

    #[test]
    fn default_common_state_is_handshaking_and_idle() {
        let mut state = CommonState::default();
        assert!(state.is_handshaking());
        assert_eq!(state.state, ConnectionState::Handshaking);
        assert!(state.wants_read());
        assert!(!state.wants_write());
        assert!(state.version.is_none());
        assert!(state.suite.is_none());
        assert!(!state.peer_closed);
        assert!(!state.local_closed);
        assert!(state.error.is_none());
        assert!(state.read_app(&mut [0u8; 1]).is_err());
    }

    #[test]
    fn close_notify_is_idempotent_and_moves_to_closing() {
        let mut state = CommonState::new();
        state.send_close_notify().unwrap();
        assert!(state.local_closed);
        assert_eq!(state.state, ConnectionState::Closing);
        assert!(!state.is_handshaking());
        let pending = state.record.pending_tx_len();
        assert_eq!(pending, 5 + 2);
        // Second call must not queue another record.
        state.send_close_notify().unwrap();
        assert_eq!(state.record.pending_tx_len(), pending);
        // After local close, writing application data fails.
        let err = state.write_app(b"x").unwrap_err();
        assert_eq!(err, TlsError::Closed);
    }

    #[test]
    fn fatal_alert_queues_and_closes() {
        let mut state = CommonState::new();
        state
            .send_alert(Alert::fatal(AlertDescription::HandshakeFailure))
            .unwrap();
        assert_eq!(state.state, ConnectionState::Closed);
        assert!(state.local_closed);
        assert!(state.wants_write());
        assert!(!state.wants_read());
        let bytes = state.record.take_ciphertext();
        assert_eq!(bytes[0], ContentType::Alert as u8);
        assert_eq!(&bytes[5..], &[2, 40]);
    }

    #[test]
    fn warning_alert_does_not_close() {
        let mut state = CommonState::new();
        state
            .send_alert(Alert::warning(AlertDescription::UserCanceled))
            .unwrap();
        assert_eq!(state.state, ConnectionState::Handshaking);
        assert!(!state.local_closed);
    }

    #[test]
    fn handle_alert_close_notify_vs_error_alert() {
        let mut state = CommonState::new();
        let err = state.handle_alert(&[1, 0]).unwrap_err();
        assert_eq!(err, TlsError::Closed);
        assert!(state.peer_closed);
        assert_eq!(state.state, ConnectionState::Closed);
        assert!(!state.wants_read());
        // Reading after peer close yields EOF.
        assert_eq!(state.read_app(&mut [0u8; 4]).unwrap(), 0);

        let mut other = CommonState::new();
        let err = other.handle_alert(&[2, 47]).unwrap_err();
        assert_eq!(
            err,
            TlsError::PeerAlert {
                level: crate::tls_alert::AlertLevel::Fatal,
                description: AlertDescription::IllegalParameter,
            }
        );
        assert!(!other.peer_closed);
        assert_eq!(other.state, ConnectionState::Closed);
    }

    #[test]
    fn handle_alert_malformed_payloads_rejected() {
        let mut state = CommonState::new();
        // Wrong length.
        assert!(state.handle_alert(&[2]).is_err());
        assert!(state.handle_alert(&[]).is_err());
        // Unknown level.
        assert!(state.handle_alert(&[9, 0]).is_err());
        // Unknown description.
        assert!(state.handle_alert(&[2, 200]).is_err());
        // Connection remains usable after bad alerts.
        assert_eq!(state.state, ConnectionState::Handshaking);
    }

    #[test]
    fn read_app_states_and_partial_reads() {
        let mut state = CommonState::new();
        // Empty + handshaking.
        let err = state.read_app(&mut [0u8; 4]).unwrap_err();
        assert_eq!(err, TlsError::HandshakeNotComplete);
        // Connected with no data.
        state.state = ConnectionState::Connected;
        let err = state.read_app(&mut [0u8; 4]).unwrap_err();
        assert_eq!(err, TlsError::WouldBlock);
        // Partial reads drain incrementally.
        state.queue_appdata(b"abcdef");
        let mut buf = [0u8; 3];
        assert_eq!(state.read_app(&mut buf).unwrap(), 3);
        assert_eq!(&buf, b"abc");
        assert_eq!(state.read_app(&mut buf).unwrap(), 3);
        assert_eq!(&buf, b"def");
        let err = state.read_app(&mut buf).unwrap_err();
        assert_eq!(err, TlsError::WouldBlock);
        // Zero-length buffer with queued data.
        state.queue_appdata(b"z");
        assert_eq!(state.read_app(&mut []).unwrap(), 0);
    }

    #[test]
    fn write_app_guards_handshake_and_closed_states() {
        let mut state = CommonState::new();
        let err = state.write_app(b"hello").unwrap_err();
        assert_eq!(err, TlsError::HandshakeNotComplete);
        state.state = ConnectionState::Connected;
        state.write_app(b"hello").unwrap();
        assert!(state.wants_write());
        let bytes = state.record.take_ciphertext();
        assert_eq!(bytes[0], ContentType::ApplicationData as u8);
        assert_eq!(&bytes[5..], b"hello");
        state.send_close_notify().unwrap();
        let err = state.write_app(b"more").unwrap_err();
        assert_eq!(err, TlsError::Closed);
    }

    #[test]
    fn write_app_fragments_at_max_fragment_length() {
        let mut state = CommonState::new();
        state.state = ConnectionState::Connected;
        let payload = vec![0x5Au8; MAX_FRAGMENT_LENGTH + 100];
        state.write_app(&payload).unwrap();
        let bytes = state.record.take_ciphertext();
        let expected = (5 + MAX_FRAGMENT_LENGTH) + (5 + 100);
        assert_eq!(bytes.len(), expected);
        // Feed back and reassemble.
        state.record.feed_ciphertext(&bytes);
        let mut first = state.record.read_raw().unwrap().unwrap();
        assert_eq!(first.payload.len(), MAX_FRAGMENT_LENGTH);
        let second = state.record.read_raw().unwrap().unwrap();
        assert_eq!(second.payload.len(), 100);
        first.payload.extend_from_slice(&second.payload);
        assert_eq!(first.payload, payload);
        assert!(state.record.read_raw().unwrap().is_none());
    }

    #[test]
    fn application_data_round_trip_through_ten_thousand_messages() {
        let mut tx = CommonState::new();
        tx.state = ConnectionState::Connected;
        let mut wire = Vec::new();
        for i in 0..10_000u32 {
            tx.write_app(&i.to_be_bytes()).unwrap();
            if i % 500 == 499 {
                wire.extend_from_slice(&tx.record.take_ciphertext());
            }
        }
        wire.extend_from_slice(&tx.record.take_ciphertext());

        let mut rx = CommonState::new();
        rx.state = ConnectionState::Connected;
        rx.record.feed_ciphertext(&wire);
        let mut count = 0usize;
        while let Some(rec) = rx.record.read_raw().unwrap() {
            assert_eq!(rec.content_type, ContentType::ApplicationData);
            assert_eq!(rec.payload.len(), 4);
            let val = u32::from_be_bytes(rec.payload.as_slice().try_into().unwrap());
            assert_eq!(val, count as u32);
            count += 1;
        }
        assert_eq!(count, 10_000);
    }

    #[test]
    fn one_mebibyte_payload_survives_fragmented_round_trip() {
        let mut tx = CommonState::new();
        tx.state = ConnectionState::Connected;
        let payload: Vec<u8> = (0..(1 << 20)).map(|i| (i & 0xFF) as u8).collect();
        tx.write_app(&payload).unwrap();
        let wire = tx.record.take_ciphertext();
        // 1 MiB / 16 KiB = 64 fragments exactly.
        assert_eq!(wire.len(), 64 * (5 + MAX_FRAGMENT_LENGTH));

        let mut rx = CommonState::new();
        rx.state = ConnectionState::Connected;
        rx.record.feed_ciphertext(&wire);
        let mut out = Vec::with_capacity(payload.len());
        let mut records = 0;
        while let Some(rec) = rx.record.read_raw().unwrap() {
            out.extend_from_slice(&rec.payload);
            records += 1;
        }
        assert_eq!(records, 64);
        assert_eq!(out, payload);
    }

    #[test]
    fn send_handshake_raw_fragments_and_updates_transcript() {
        let mut state = CommonState::new();
        let big = crate::tls_handshake::encode_handshake(
            crate::tls_ids::HandshakeType::Certificate,
            &vec![0xAB; MAX_FRAGMENT_LENGTH + 10],
        );
        let transcript_before = state.transcript.bytes().len();
        state.send_handshake_raw(&big).unwrap();
        assert_eq!(state.transcript.bytes().len(), transcript_before + big.len());
        let bytes = state.record.take_ciphertext();
        assert_eq!(bytes.len(), (5 + MAX_FRAGMENT_LENGTH) + (5 + 14));
        // Records parse back as handshake content.
        state.record.feed_ciphertext(&bytes);
        let r1 = state.record.read_raw().unwrap().unwrap();
        assert_eq!(r1.content_type, ContentType::Handshake);
        assert_eq!(r1.payload.len(), MAX_FRAGMENT_LENGTH);
        let r2 = state.record.read_raw().unwrap().unwrap();
        assert_eq!(r2.payload.len(), 14);
    }

    #[test]
    fn send_ccs_writes_single_byte_record() {
        let mut state = CommonState::new();
        state.send_ccs().unwrap();
        let bytes = state.record.take_ciphertext();
        assert_eq!(bytes[0], ContentType::ChangeCipherSpec as u8);
        assert_eq!(&bytes[3..5], &[0, 1]);
        assert_eq!(&bytes[5..], &[1]);
    }

    #[test]
    fn fail_sends_alert_stores_error_and_closes() {
        let mut state = CommonState::new();
        let err = state.fail(TlsError::Alert(AlertDescription::DecodeError));
        assert_eq!(err, TlsError::Alert(AlertDescription::DecodeError));
        assert_eq!(state.state, ConnectionState::Closed);
        assert_eq!(state.error.as_ref(), Some(&err));
        let bytes = state.record.take_ciphertext();
        assert_eq!(bytes[0], ContentType::Alert as u8);
        assert_eq!(&bytes[5..], &[2, 50]); // fatal decode_error
    }

    #[test]
    fn fail_without_alert_description_queues_no_alert() {
        let mut state = CommonState::new();
        let err = state.fail(TlsError::Internal("boom".into()));
        assert!(matches!(err, TlsError::Internal(_)));
        assert_eq!(state.state, ConnectionState::Closed);
        assert!(!state.wants_write());
        assert!(state.error.is_some());
    }

    #[test]
    fn fail_with_close_notify_does_not_echo_alert() {
        let mut state = CommonState::new();
        let _ = state.fail(TlsError::Closed);
        assert_eq!(state.state, ConnectionState::Closed);
        assert!(!state.wants_write());
    }

    #[test]
    fn oversized_record_declared_length_rejected_before_payload() {
        let mut state = CommonState::new();
        // 0xFFFF length exceeds MAX_CIPHERTEXT_LENGTH: rejected with only the
        // 5-byte header buffered; no payload allocation occurs.
        let header = [ContentType::Handshake as u8, 3, 3, 0xFF, 0xFF];
        state.record.feed_ciphertext(&header);
        let err = state.record.read_raw().unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::RecordOverflow));
        assert_eq!(state.record.pending_rx(), 5);

        // 2^14 + 1 passes the ciphertext ceiling but violates the fragment
        // limit once the full payload is present.
        let mut rec = vec![ContentType::Handshake as u8, 3, 3];
        rec.extend_from_slice(&((MAX_FRAGMENT_LENGTH + 1) as u16).to_be_bytes());
        rec.extend_from_slice(&[0u8; MAX_FRAGMENT_LENGTH + 1]);
        state.record.feed_ciphertext(&rec);
        let err = state.record.read_raw().unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::RecordOverflow));
    }

    #[test]
    fn partial_record_returns_none_until_complete() {
        let mut state = CommonState::new();
        let rec = cleartext_record(ContentType::Handshake, &[1, 2, 3, 4]);
        state.record.feed_ciphertext(&rec[..4]);
        assert!(state.record.read_raw().unwrap().is_none());
        assert_eq!(state.record.pending_rx(), 4);
        state.record.feed_ciphertext(&rec[4..]);
        let out = state.record.read_raw().unwrap().unwrap();
        assert_eq!(out.payload, vec![1, 2, 3, 4]);
        assert!(state.record.read_raw().unwrap().is_none());
    }

    #[test]
    fn wrong_record_length_splits_into_garbage_without_panic() {
        let mut state = CommonState::new();
        // Declared length 2 while 4 payload bytes follow; the trailing bytes
        // are misparsed as the next record header.
        let mut rec = vec![ContentType::Handshake as u8, 3, 3, 0, 2, 9, 9];
        rec.extend_from_slice(&[1, 1]);
        state.record.feed_ciphertext(&rec);
        let first = state.record.read_raw().unwrap().unwrap();
        assert_eq!(first.payload, vec![9, 9]);
        // Remaining [1, 1] is shorter than a header.
        assert!(state.record.read_raw().unwrap().is_none());
        // Supplying the rest of a bogus header surfaces a content-type error.
        state.record.feed_ciphertext(&[1, 3, 3, 0, 1, 7]);
        let err = state.record.read_raw().unwrap_err();
        assert!(matches!(err, TlsError::Decode(_)));
    }

    #[test]
    fn sequence_number_overflow_is_rejected() {
        use crate::tls_aead::AeadKey;
        use crate::tls_ids::AeadAlgorithm;
        use crate::tls_record::TrafficKeys;

        let aead = AeadKey::new(AeadAlgorithm::Aes128Gcm, vec![7u8; 16], vec![9u8; 12]).unwrap();
        let mut keys = TrafficKeys::new(aead.clone());
        assert_eq!(keys.next_seq().unwrap(), 0);
        assert_eq!(keys.next_seq().unwrap(), 1);

        let mut wrapped = TrafficKeys::new(aead);
        wrapped.seq = u64::MAX;
        let err = wrapped.next_seq().unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::InternalError));

        // A connection whose write keys have exhausted the sequence number
        // fails the next protected write.
        let mut state = CommonState::new();
        state.state = ConnectionState::Connected;
        let exhausted = TrafficKeys {
            aead: AeadKey::new(AeadAlgorithm::Aes128Gcm, vec![7u8; 16], vec![9u8; 12]).unwrap(),
            seq: u64::MAX,
        };
        state.install_write_keys(exhausted, ProtocolVersion::Tls13, CipherSuite::TlsAes128GcmSha256);
        let err = state.write_app(b"x").unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::InternalError));
    }

    #[test]
    fn corrupted_aead_tag_is_rejected_on_read() {
        use crate::tls_aead::AeadKey;
        use crate::tls_ids::AeadAlgorithm;
        use crate::tls_record::TrafficKeys;

        for (version, suite) in [
            (ProtocolVersion::Tls13, CipherSuite::TlsAes128GcmSha256),
            (
                ProtocolVersion::Tls12,
                CipherSuite::TlsEcdheEcdsaWithAes128GcmSha256,
            ),
        ] {
            let key = || AeadKey::new(AeadAlgorithm::Aes128Gcm, vec![0x11u8; 16], vec![0x22u8; 12]).unwrap();
            let mut tx = CommonState::new();
            tx.state = ConnectionState::Connected;
            tx.install_write_keys(TrafficKeys::new(key()), version, suite);
            tx.write_app(b"integrity").unwrap();
            let mut wire = tx.record.take_ciphertext();
            assert!(!wire.is_empty());
            // Flip the final byte (authentication tag region).
            let last = wire.len() - 1;
            wire[last] ^= 0x01;

            let mut rx = CommonState::new();
            rx.state = ConnectionState::Connected;
            rx.install_read_keys(TrafficKeys::new(key()), version, suite);
            rx.record.feed_ciphertext(&wire);
            let err = rx.record.read_raw().unwrap_err();
            assert_eq!(err, TlsError::Alert(AlertDescription::BadRecordMac));
        }
    }

    #[test]
    fn wants_read_false_after_peer_or_local_close() {
        let mut state = CommonState::new();
        assert!(state.wants_read());
        let _ = state.handle_alert(&[1, 0]);
        assert!(!state.wants_read());
        let mut other = CommonState::new();
        other.send_close_notify().unwrap();
        // Local close_notify: still reading until peer answers (state Closing).
        assert!(other.wants_read());
        let _ = other.handle_alert(&[1, 0]);
        assert!(!other.wants_read());
    }

    #[test]
    fn io_state_reports_pending_interest() {
        let mut state = CommonState::new();
        let io = IoState {
            plaintext_bytes_to_read: state.app_rx.len(),
            tls_bytes_to_write: state.record.pending_tx_len(),
            handshaking: state.is_handshaking(),
        };
        assert!(io.handshaking);
        assert_eq!(io.plaintext_bytes_to_read, 0);
        assert_eq!(io.tls_bytes_to_write, 0);
        state.queue_appdata(b"xyz");
        state.send_ccs().unwrap();
        assert_eq!(state.app_rx.len(), 3);
        assert!(state.record.pending_tx_len() > 0);
    }
}
