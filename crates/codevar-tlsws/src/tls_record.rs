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

use crate::tls_aead::{AeadKey, TlsAead};
use crate::tls_alert::AlertDescription;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::{
    AeadAlgorithm, ContentType, MAX_CIPHERTEXT_LENGTH, MAX_FRAGMENT_LENGTH,
    ProtocolVersion,
};

/// Five-byte TLS record header: `type (1) || legacy_record_version (2) || length (2)`.
pub const RECORD_HEADER_LEN: usize = 5;

/// Decrypted / plaintext record delivered to the handshake or application layer.
#[derive(Debug, Clone)]
pub struct PlainRecord {
    /// Inner content type (after TLS 1.3 deprotection, or the outer type for cleartext).
    pub content_type: ContentType,
    /// Plaintext payload (handshake bytes, alert, or application data).
    pub payload: Vec<u8>,
}

/// Traffic secrets currently installed on one direction of the connection.
#[derive(Clone)]
pub struct TrafficKeys {
    /// AEAD key material.
    pub aead: AeadKey,
    /// Sequence number for this direction (starts at 0 after each key installation).
    pub seq: u64,
}

impl TrafficKeys {
    /// Creates traffic keys with sequence number zero.
    #[must_use]
    pub fn new(aead: AeadKey) -> Self {
        Self { aead, seq: 0 }
    }

    /// Increments the sequence number, failing on wrap (RFC 8446 §5.3).
    pub fn next_seq(&mut self) -> TlsResult<u64> {
        let seq = self.seq;
        self.seq = self
            .seq
            .checked_add(1)
            .ok_or_else(|| TlsError::Alert(AlertDescription::InternalError))?;
        Ok(seq)
    }
}

/// Negotiated record-protection parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordProtection {
    /// Cleartext records (handshake before ChangeCipherSpec / TLS 1.3 handshake keys).
    Cleartext,
    /// TLS 1.3 AEAD with opaque `application_data` outer type.
    Tls13,
    /// TLS 1.2 AES-GCM (RFC 5288).
    Tls12Gcm,
    /// TLS 1.2 ChaCha20-Poly1305 (RFC 7905).
    Tls12ChaCha,
}

impl RecordProtection {
    /// Derives the protection mode from an AEAD algorithm and protocol version.
    #[must_use]
    pub const fn from_suite(version: ProtocolVersion, alg: AeadAlgorithm) -> Self {
        match version {
            ProtocolVersion::Tls13 => Self::Tls13,
            ProtocolVersion::Tls12 => match alg {
                AeadAlgorithm::Aes128Gcm | AeadAlgorithm::Aes256Gcm => Self::Tls12Gcm,
                AeadAlgorithm::ChaCha20Poly1305 => Self::Tls12ChaCha,
            },
            ProtocolVersion::Tls10 | ProtocolVersion::Tls11 => Self::Cleartext,
        }
    }
}

/// Bidirectional record layer state.
pub struct RecordLayer {
    /// Inbound receive buffer (may hold a partial record).
    rx_buf: Vec<u8>,
    /// Outbound ciphertext ready for the transport.
    tx_buf: Vec<u8>,
    /// Current read-side keys (`None` = cleartext).
    read_keys: Option<TrafficKeys>,
    /// Current write-side keys (`None` = cleartext).
    write_keys: Option<TrafficKeys>,
    /// Active read protection mode.
    read_mode: RecordProtection,
    /// Active write protection mode.
    write_mode: RecordProtection,
    /// Legacy record version written in cleartext headers (usually TLS 1.2 / 0x0303).
    legacy_version: ProtocolVersion,
}

impl Default for RecordLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordLayer {
    /// Creates a cleartext record layer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rx_buf: Vec::new(),
            tx_buf: Vec::new(),
            read_keys: None,
            write_keys: None,
            read_mode: RecordProtection::Cleartext,
            write_mode: RecordProtection::Cleartext,
            legacy_version: ProtocolVersion::Tls12,
        }
    }

    /// Sets the legacy_record_version used for outbound cleartext / TLS 1.2 headers.
    pub fn set_legacy_version(&mut self, version: ProtocolVersion) {
        self.legacy_version = version;
    }

    /// Installs write traffic keys and switches the write protection mode.
    pub fn set_write_keys(&mut self, keys: TrafficKeys, mode: RecordProtection) {
        self.write_keys = Some(keys);
        self.write_mode = mode;
    }

    /// Installs read traffic keys and switches the read protection mode.
    pub fn set_read_keys(&mut self, keys: TrafficKeys, mode: RecordProtection) {
        self.read_keys = Some(keys);
        self.read_mode = mode;
    }

    /// Returns a mutable reference to write keys (for KeyUpdate).
    pub fn write_keys_mut(&mut self) -> Option<&mut TrafficKeys> {
        self.write_keys.as_mut()
    }

    /// Returns a mutable reference to read keys (for KeyUpdate).
    pub fn read_keys_mut(&mut self) -> Option<&mut TrafficKeys> {
        self.read_keys.as_mut()
    }

    /// Appends ciphertext received from the network.
    pub fn feed_ciphertext(&mut self, data: &[u8]) {
        self.rx_buf.extend_from_slice(data);
    }

    /// Drains all buffered outbound ciphertext.
    pub fn take_ciphertext(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.tx_buf)
    }

    /// Returns true when outbound ciphertext is pending.
    #[must_use]
    pub fn wants_write(&self) -> bool {
        !self.tx_buf.is_empty()
    }

    /// Number of outbound ciphertext bytes pending.
    #[must_use]
    pub fn pending_tx_len(&self) -> usize {
        self.tx_buf.len()
    }

    /// Returns the number of buffered inbound ciphertext bytes.
    #[must_use]
    pub fn pending_rx(&self) -> usize {
        self.rx_buf.len()
    }

    /// Encodes and (if keyed) protects a plaintext record, appending to the TX buffer.
    pub fn write_raw(
        &mut self,
        content_type: ContentType,
        plaintext: &[u8],
    ) -> TlsResult<()> {
        if plaintext.len() > MAX_FRAGMENT_LENGTH {
            return Err(TlsError::Alert(AlertDescription::RecordOverflow));
        }
        match self.write_mode {
            RecordProtection::Cleartext => {
                self.encode_cleartext(content_type, plaintext);
            }
            RecordProtection::Tls13 => {
                self.encrypt_tls13(content_type, plaintext)?;
            }
            RecordProtection::Tls12Gcm => {
                self.encrypt_tls12_gcm(content_type, plaintext)?;
            }
            RecordProtection::Tls12ChaCha => {
                self.encrypt_tls12_chacha(content_type, plaintext)?;
            }
        }
        Ok(())
    }

    /// Parses and decrypts the next complete record from the RX buffer, if available.
    pub fn read_raw(&mut self) -> TlsResult<Option<PlainRecord>> {
        if self.rx_buf.len() < RECORD_HEADER_LEN {
            return Ok(None);
        }
        let content_type = ContentType::from_u8(self.rx_buf[0])?;
        let length = u16::from_be_bytes([self.rx_buf[3], self.rx_buf[4]]) as usize;
        if length > MAX_CIPHERTEXT_LENGTH {
            return Err(TlsError::Alert(AlertDescription::RecordOverflow));
        }
        if self.rx_buf.len() < RECORD_HEADER_LEN + length {
            return Ok(None);
        }
        let header: [u8; 5] = self.rx_buf[..RECORD_HEADER_LEN]
            .try_into()
            .map_err(|_| TlsError::Internal("record header".into()))?;
        let payload = self.rx_buf[RECORD_HEADER_LEN..RECORD_HEADER_LEN + length].to_vec();
        self.rx_buf.drain(..RECORD_HEADER_LEN + length);

        match self.read_mode {
            RecordProtection::Cleartext => {
                if payload.len() > MAX_FRAGMENT_LENGTH {
                    return Err(TlsError::Alert(AlertDescription::RecordOverflow));
                }
                Ok(Some(PlainRecord {
                    content_type,
                    payload,
                }))
            }
            RecordProtection::Tls13 => {
                // TLS 1.3 ciphertext always uses outer type application_data (or handshake
                // before the first encrypted flight uses the cleartext path). CCS is ignored.
                if content_type == ContentType::ChangeCipherSpec {
                    // RFC 8446 §5: CCS may appear for middlebox compatibility; ignore.
                    return Ok(None);
                }
                if content_type != ContentType::ApplicationData {
                    return Err(TlsError::Alert(AlertDescription::UnexpectedMessage));
                }
                self.decrypt_tls13(&header, &payload)
            }
            RecordProtection::Tls12Gcm => {
                self.decrypt_tls12_gcm(content_type, &header, &payload)
            }
            RecordProtection::Tls12ChaCha => {
                self.decrypt_tls12_chacha(content_type, &header, &payload)
            }
        }
    }

    fn encode_cleartext(&mut self, content_type: ContentType, plaintext: &[u8]) {
        self.tx_buf.push(content_type.as_u8());
        self.tx_buf
            .extend_from_slice(&self.legacy_version.to_be_bytes());
        let len = plaintext.len() as u16;
        self.tx_buf.extend_from_slice(&len.to_be_bytes());
        self.tx_buf.extend_from_slice(plaintext);
    }

    fn encrypt_tls13(
        &mut self,
        content_type: ContentType,
        plaintext: &[u8],
    ) -> TlsResult<()> {
        let keys = self
            .write_keys
            .as_mut()
            .ok_or_else(|| TlsError::Internal("TLS 1.3 write keys missing".into()))?;
        let seq = keys.next_seq()?;
        let mut inner = Vec::with_capacity(plaintext.len() + 1);
        inner.extend_from_slice(plaintext);
        inner.push(content_type.as_u8());

        let ciphertext_len = inner.len() + keys.aead.algorithm().tag_len();
        if ciphertext_len > MAX_CIPHERTEXT_LENGTH {
            return Err(TlsError::Alert(AlertDescription::RecordOverflow));
        }
        // Outer header used as AAD (RFC 8446 §5.2).
        let mut header = [0u8; 5];
        header[0] = ContentType::ApplicationData.as_u8();
        header[1..3].copy_from_slice(&ProtocolVersion::Tls12.to_be_bytes());
        header[3..5].copy_from_slice(&(ciphertext_len as u16).to_be_bytes());

        let sealed = TlsAead::encrypt_tls13(&keys.aead, seq, &header, &inner)?;
        self.tx_buf.extend_from_slice(&header);
        self.tx_buf.extend_from_slice(&sealed);
        Ok(())
    }

    fn decrypt_tls13(
        &mut self,
        header: &[u8; 5],
        ciphertext: &[u8],
    ) -> TlsResult<Option<PlainRecord>> {
        let keys = self
            .read_keys
            .as_mut()
            .ok_or_else(|| TlsError::Internal("TLS 1.3 read keys missing".into()))?;
        let seq = keys.next_seq()?;
        let inner = TlsAead::decrypt_tls13(&keys.aead, seq, header, ciphertext)?;
        // Strip zero padding from the end, then read content type.
        let mut end = inner.len();
        while end > 0 && inner[end - 1] == 0 {
            end -= 1;
        }
        if end == 0 {
            return Err(TlsError::Alert(AlertDescription::UnexpectedMessage));
        }
        let content_type = ContentType::from_u8(inner[end - 1])?;
        let payload = inner[..end - 1].to_vec();
        if payload.len() > MAX_FRAGMENT_LENGTH {
            return Err(TlsError::Alert(AlertDescription::RecordOverflow));
        }
        Ok(Some(PlainRecord {
            content_type,
            payload,
        }))
    }

    fn encrypt_tls12_gcm(
        &mut self,
        content_type: ContentType,
        plaintext: &[u8],
    ) -> TlsResult<()> {
        let keys = self
            .write_keys
            .as_mut()
            .ok_or_else(|| TlsError::Internal("TLS 1.2 write keys missing".into()))?;
        let seq = keys.next_seq()?;
        let explicit = seq.to_be_bytes();
        let ciphertext_len = 8 + plaintext.len() + keys.aead.algorithm().tag_len();
        let mut header = [0u8; 5];
        header[0] = content_type.as_u8();
        header[1..3].copy_from_slice(&self.legacy_version.to_be_bytes());
        header[3..5].copy_from_slice(&(ciphertext_len as u16).to_be_bytes());
        let aad = tls12_aad(seq, content_type, self.legacy_version, plaintext.len());
        let sealed = TlsAead::encrypt_tls12_gcm(&keys.aead, explicit, &aad, plaintext)?;
        self.tx_buf.extend_from_slice(&header);
        self.tx_buf.extend_from_slice(&sealed);
        Ok(())
    }

    fn decrypt_tls12_gcm(
        &mut self,
        content_type: ContentType,
        header: &[u8; 5],
        ciphertext: &[u8],
    ) -> TlsResult<Option<PlainRecord>> {
        let _ = header;
        let keys = self
            .read_keys
            .as_mut()
            .ok_or_else(|| TlsError::Internal("TLS 1.2 read keys missing".into()))?;
        let seq = keys.next_seq()?;
        if ciphertext.len() < 8 + 16 {
            return Err(TlsError::Alert(AlertDescription::BadRecordMac));
        }
        let plain_len = ciphertext.len() - 8 - 16;
        let aad = tls12_aad(seq, content_type, self.legacy_version, plain_len);
        let payload = TlsAead::decrypt_tls12_gcm(&keys.aead, ciphertext, &aad)?;
        if payload.len() > MAX_FRAGMENT_LENGTH {
            return Err(TlsError::Alert(AlertDescription::RecordOverflow));
        }
        Ok(Some(PlainRecord {
            content_type,
            payload,
        }))
    }

    fn encrypt_tls12_chacha(
        &mut self,
        content_type: ContentType,
        plaintext: &[u8],
    ) -> TlsResult<()> {
        let keys = self
            .write_keys
            .as_mut()
            .ok_or_else(|| TlsError::Internal("TLS 1.2 write keys missing".into()))?;
        let seq = keys.next_seq()?;
        let ciphertext_len = plaintext.len() + keys.aead.algorithm().tag_len();
        let mut header = [0u8; 5];
        header[0] = content_type.as_u8();
        header[1..3].copy_from_slice(&self.legacy_version.to_be_bytes());
        header[3..5].copy_from_slice(&(ciphertext_len as u16).to_be_bytes());
        let aad = tls12_aad(seq, content_type, self.legacy_version, plaintext.len());
        let sealed = TlsAead::encrypt_tls12_chacha(&keys.aead, seq, &aad, plaintext)?;
        self.tx_buf.extend_from_slice(&header);
        self.tx_buf.extend_from_slice(&sealed);
        Ok(())
    }

    fn decrypt_tls12_chacha(
        &mut self,
        content_type: ContentType,
        header: &[u8; 5],
        ciphertext: &[u8],
    ) -> TlsResult<Option<PlainRecord>> {
        let _ = header;
        let keys = self
            .read_keys
            .as_mut()
            .ok_or_else(|| TlsError::Internal("TLS 1.2 read keys missing".into()))?;
        let seq = keys.next_seq()?;
        if ciphertext.len() < 16 {
            return Err(TlsError::Alert(AlertDescription::BadRecordMac));
        }
        let plain_len = ciphertext.len() - 16;
        let aad = tls12_aad(seq, content_type, self.legacy_version, plain_len);
        let payload = TlsAead::decrypt_tls12_chacha(&keys.aead, seq, &aad, ciphertext)?;
        if payload.len() > MAX_FRAGMENT_LENGTH {
            return Err(TlsError::Alert(AlertDescription::RecordOverflow));
        }
        Ok(Some(PlainRecord {
            content_type,
            payload,
        }))
    }
}

/// TLS 1.2 additional data: `seq_num (8) || type (1) || version (2) || length (2)`.
#[must_use]
pub fn tls12_aad(
    seq: u64,
    content_type: ContentType,
    version: ProtocolVersion,
    plaintext_len: usize,
) -> [u8; 13] {
    let mut aad = [0u8; 13];
    aad[0..8].copy_from_slice(&seq.to_be_bytes());
    aad[8] = content_type.as_u8();
    aad[9..11].copy_from_slice(&version.to_be_bytes());
    aad[11..13].copy_from_slice(&(plaintext_len as u16).to_be_bytes());
    aad
}

/// Encodes a standalone cleartext record (used by tests / alert helpers).
#[must_use]
pub fn encode_cleartext_record(
    content_type: ContentType,
    version: ProtocolVersion,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(RECORD_HEADER_LEN + payload.len());
    out.push(content_type.as_u8());
    out.extend_from_slice(&version.to_be_bytes());
    out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aead_key() -> AeadKey {
        AeadKey::new(AeadAlgorithm::Aes128Gcm, vec![7u8; 16], vec![9u8; 12]).unwrap()
    }

    #[test]
    fn cleartext_write_produces_header_and_round_trips() {
        let mut tx = RecordLayer::new();
        tx.write_raw(ContentType::Handshake, &[1, 2, 3]).unwrap();
        let wire = tx.take_ciphertext();
        assert_eq!(wire.len(), 8);
        assert_eq!(wire[0], ContentType::Handshake.as_u8());
        assert_eq!(&wire[1..3], &[3, 3]);
        assert_eq!(&wire[3..5], &[0, 3]);

        let mut rx = RecordLayer::new();
        rx.feed_ciphertext(&wire);
        let rec = rx.read_raw().unwrap().unwrap();
        assert_eq!(rec.content_type, ContentType::Handshake);
        assert_eq!(rec.payload, vec![1, 2, 3]);
        assert_eq!(rx.pending_rx(), 0);
    }

    #[test]
    fn zero_length_record_round_trips() {
        let mut tx = RecordLayer::new();
        tx.write_raw(ContentType::Alert, &[]).unwrap();
        let wire = tx.take_ciphertext();
        assert_eq!(wire, vec![21, 3, 3, 0, 0]);

        let mut rx = RecordLayer::new();
        rx.feed_ciphertext(&wire);
        let rec = rx.read_raw().unwrap().unwrap();
        assert_eq!(rec.content_type, ContentType::Alert);
        assert!(rec.payload.is_empty());
    }

    #[test]
    fn partial_records_return_none_until_complete() {
        let wire =
            encode_cleartext_record(ContentType::Alert, ProtocolVersion::Tls12, &[1, 0]);
        let mut layer = RecordLayer::new();
        let last = wire.len() - 1;
        for (i, b) in wire.iter().enumerate() {
            layer.feed_ciphertext(std::slice::from_ref(b));
            let got = layer.read_raw().unwrap();
            if i < last {
                assert!(got.is_none(), "premature record at byte {i}");
            } else {
                let rec = got.unwrap();
                assert_eq!(rec.content_type, ContentType::Alert);
                assert_eq!(rec.payload, vec![1, 0]);
            }
        }
        assert_eq!(layer.pending_rx(), 0);
    }

    #[test]
    fn corrupted_content_type_rejected_once_header_present() {
        let mut layer = RecordLayer::new();
        layer.feed_ciphertext(&[0x07, 3, 3, 0, 1]);
        let err = layer.read_raw().unwrap_err();
        assert!(matches!(err, TlsError::Decode(_)), "err={err:?}");
        assert_eq!(layer.pending_rx(), 5);
    }

    #[test]
    fn declared_length_above_max_rejected_before_payload() {
        let mut layer = RecordLayer::new();
        layer.feed_ciphertext(&[23, 3, 3, 0xff, 0xff]);
        let err = layer.read_raw().unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::RecordOverflow)),
            "err={err:?}"
        );
        assert_eq!(layer.pending_rx(), 5);

        let mut layer2 = RecordLayer::new();
        layer2.feed_ciphertext(&[23, 3, 3, 0x41, 0x01]);
        assert!(layer2.read_raw().is_err());
        assert_eq!(layer2.pending_rx(), 5);
    }

    #[test]
    fn ciphertext_bounds_match_documented_constants() {
        assert_eq!(MAX_FRAGMENT_LENGTH, 16384);
        assert_eq!(MAX_CIPHERTEXT_LENGTH, 16640);
        assert_eq!(RECORD_HEADER_LEN, 5);
        assert!(MAX_CIPHERTEXT_LENGTH > MAX_FRAGMENT_LENGTH);
    }

    #[test]
    fn cleartext_payload_above_fragment_max_rejected() {
        let payload = vec![0u8; MAX_CIPHERTEXT_LENGTH];
        let wire = encode_cleartext_record(
            ContentType::ApplicationData,
            ProtocolVersion::Tls12,
            &payload,
        );
        let mut layer = RecordLayer::new();
        layer.feed_ciphertext(&wire);
        let err = layer.read_raw().unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::RecordOverflow)),
            "err={err:?}"
        );
    }

    #[test]
    fn write_raw_bounds_check_does_not_mutate_tx_buffer() {
        let mut layer = RecordLayer::new();
        let oversized = vec![0u8; MAX_FRAGMENT_LENGTH + 1];
        let err = layer
            .write_raw(ContentType::Handshake, &oversized)
            .unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::RecordOverflow)),
            "err={err:?}"
        );
        assert_eq!(layer.pending_tx_len(), 0);

        let exact = vec![0u8; MAX_FRAGMENT_LENGTH];
        layer.write_raw(ContentType::Handshake, &exact).unwrap();
        assert_eq!(
            layer.pending_tx_len(),
            RECORD_HEADER_LEN + MAX_FRAGMENT_LENGTH
        );
    }

    #[test]
    fn legacy_version_written_and_read_is_not_validated() {
        let mut tx = RecordLayer::new();
        tx.set_legacy_version(ProtocolVersion::Tls10);
        tx.write_raw(ContentType::Alert, &[0]).unwrap();
        let wire = tx.take_ciphertext();
        assert_eq!(&wire[1..3], &[0x03, 0x01]);

        let foreign =
            encode_cleartext_record(ContentType::Handshake, ProtocolVersion::Tls11, &[9]);
        let mut rx = RecordLayer::new();
        rx.feed_ciphertext(&foreign);
        let rec = rx.read_raw().unwrap().unwrap();
        assert_eq!(rec.content_type, ContentType::Handshake);
        assert_eq!(rec.payload, vec![9]);
    }

    #[test]
    fn multiple_back_to_back_records_read_in_order() {
        let mut tx = RecordLayer::new();
        tx.write_raw(ContentType::Handshake, &[1]).unwrap();
        tx.write_raw(ContentType::ApplicationData, &[2, 3]).unwrap();
        let wire = tx.take_ciphertext();

        let mut rx = RecordLayer::new();
        rx.feed_ciphertext(&wire);
        let first = rx.read_raw().unwrap().unwrap();
        let second = rx.read_raw().unwrap().unwrap();
        assert_eq!(first.content_type, ContentType::Handshake);
        assert_eq!(first.payload, vec![1]);
        assert_eq!(second.content_type, ContentType::ApplicationData);
        assert_eq!(second.payload, vec![2, 3]);
        assert!(rx.read_raw().unwrap().is_none());
    }

    #[test]
    fn take_ciphertext_drains_pending_output() {
        let mut layer = RecordLayer::new();
        assert!(!layer.wants_write());
        layer.write_raw(ContentType::Handshake, &[1, 2]).unwrap();
        assert!(layer.wants_write());
        assert_eq!(layer.pending_tx_len(), 7);
        let first = layer.take_ciphertext();
        assert_eq!(first.len(), 7);
        assert!(!layer.wants_write());
        assert_eq!(layer.pending_tx_len(), 0);
        assert!(layer.take_ciphertext().is_empty());
    }

    #[test]
    fn default_layer_is_idle_cleartext() {
        let mut layer = RecordLayer::default();
        assert_eq!(layer.pending_rx(), 0);
        assert!(!layer.wants_write());
        assert!(layer.write_keys_mut().is_none());
        assert!(layer.read_keys_mut().is_none());
    }

    #[test]
    fn traffic_keys_sequence_increments_until_wrap() {
        let mut keys = TrafficKeys::new(aead_key());
        assert_eq!(keys.seq, 0);
        assert_eq!(keys.next_seq().unwrap(), 0);
        assert_eq!(keys.next_seq().unwrap(), 1);
        assert_eq!(keys.seq, 2);
        keys.seq = u64::MAX;
        let err = keys.next_seq().unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::InternalError)),
            "err={err:?}"
        );
    }

    #[test]
    fn write_fails_when_sequence_would_wrap() {
        let mut keys = TrafficKeys::new(aead_key());
        keys.seq = u64::MAX;
        let mut layer = RecordLayer::new();
        layer.set_write_keys(keys, RecordProtection::Tls13);
        let err = layer
            .write_raw(ContentType::ApplicationData, b"x")
            .unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::InternalError)),
            "err={err:?}"
        );
        assert_eq!(layer.pending_tx_len(), 0);
    }

    #[test]
    fn record_protection_selected_from_version_and_aead() {
        use AeadAlgorithm::{Aes128Gcm, Aes256Gcm, ChaCha20Poly1305};
        use ProtocolVersion::{Tls10, Tls11, Tls12, Tls13};

        assert_eq!(
            RecordProtection::from_suite(Tls13, Aes128Gcm),
            RecordProtection::Tls13
        );
        assert_eq!(
            RecordProtection::from_suite(Tls13, ChaCha20Poly1305),
            RecordProtection::Tls13
        );
        assert_eq!(
            RecordProtection::from_suite(Tls12, Aes128Gcm),
            RecordProtection::Tls12Gcm
        );
        assert_eq!(
            RecordProtection::from_suite(Tls12, Aes256Gcm),
            RecordProtection::Tls12Gcm
        );
        assert_eq!(
            RecordProtection::from_suite(Tls12, ChaCha20Poly1305),
            RecordProtection::Tls12ChaCha
        );
        assert_eq!(
            RecordProtection::from_suite(Tls10, Aes128Gcm),
            RecordProtection::Cleartext
        );
        assert_eq!(
            RecordProtection::from_suite(Tls11, ChaCha20Poly1305),
            RecordProtection::Cleartext
        );
    }

    #[test]
    fn tls12_aad_layout_matches_rfc5288() {
        let seq = 0x0102_0304_0506_0708u64;
        let aad = tls12_aad(seq, ContentType::Handshake, ProtocolVersion::Tls12, 0x00ab);
        assert_eq!(&aad[0..8], &seq.to_be_bytes());
        assert_eq!(aad[8], 22);
        assert_eq!(&aad[9..11], &[3, 3]);
        assert_eq!(&aad[11..13], &[0x00, 0xab]);
    }

    #[test]
    fn encode_cleartext_record_layout() {
        let wire = encode_cleartext_record(
            ContentType::ApplicationData,
            ProtocolVersion::Tls12,
            &[0xaa],
        );
        assert_eq!(wire, vec![23, 3, 3, 0, 1, 0xaa]);
        let empty =
            encode_cleartext_record(ContentType::Alert, ProtocolVersion::Tls10, &[]);
        assert_eq!(empty, vec![21, 3, 1, 0, 0]);
    }

    #[test]
    fn tls13_record_round_trips_with_inner_content_type() {
        let key = aead_key();
        let mut tx = RecordLayer::new();
        tx.set_write_keys(TrafficKeys::new(key.clone()), RecordProtection::Tls13);
        tx.write_raw(ContentType::Handshake, &[0xAA; 10]).unwrap();
        let wire = tx.take_ciphertext();
        assert_eq!(wire[0], ContentType::ApplicationData.as_u8());
        assert_eq!(&wire[1..3], &[3, 3]);
        let header_len = usize::from(u16::from_be_bytes([wire[3], wire[4]]));
        assert_eq!(header_len, 10 + 1 + 16);

        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls13);
        rx.feed_ciphertext(&wire);
        let rec = rx.read_raw().unwrap().unwrap();
        assert_eq!(rec.content_type, ContentType::Handshake);
        assert_eq!(rec.payload, vec![0xAA; 10]);
        assert_eq!(rx.pending_rx(), 0);
        assert_eq!(rx.read_keys_mut().unwrap().seq, 1);
        assert_eq!(tx.write_keys_mut().unwrap().seq, 1);
    }

    #[test]
    fn tls13_middlebox_ccs_is_ignored_and_drained() {
        let key = aead_key();
        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls13);
        rx.feed_ciphertext(&[20, 3, 3, 0, 1, 1]);
        assert!(rx.read_raw().unwrap().is_none());
        assert_eq!(rx.pending_rx(), 0);
    }

    #[test]
    fn tls13_non_application_data_outer_type_rejected() {
        let key = aead_key();
        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls13);
        rx.feed_ciphertext(&[22, 3, 3, 0, 0]);
        let err = rx.read_raw().unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::UnexpectedMessage)),
            "err={err:?}"
        );
    }

    #[test]
    fn tls13_tampered_ciphertext_fails_mac_check() {
        let key = aead_key();
        let mut tx = RecordLayer::new();
        tx.set_write_keys(TrafficKeys::new(key.clone()), RecordProtection::Tls13);
        tx.write_raw(ContentType::ApplicationData, b"hello")
            .unwrap();
        let mut wire = tx.take_ciphertext();
        let n = wire.len();
        wire[n - 1] ^= 0x01;

        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls13);
        rx.feed_ciphertext(&wire);
        let err = rx.read_raw().unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::BadRecordMac)),
            "err={err:?}"
        );
    }

    #[test]
    fn tls12_gcm_record_round_trips() {
        let key = aead_key();
        let mut tx = RecordLayer::new();
        tx.set_write_keys(TrafficKeys::new(key.clone()), RecordProtection::Tls12Gcm);
        tx.write_raw(ContentType::ApplicationData, &[1, 2, 3, 4])
            .unwrap();
        let wire = tx.take_ciphertext();
        assert_eq!(wire[0], ContentType::ApplicationData.as_u8());
        let header_len = usize::from(u16::from_be_bytes([wire[3], wire[4]]));
        assert_eq!(header_len, 8 + 4 + 16);

        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls12Gcm);
        rx.feed_ciphertext(&wire);
        let rec = rx.read_raw().unwrap().unwrap();
        assert_eq!(rec.content_type, ContentType::ApplicationData);
        assert_eq!(rec.payload, vec![1, 2, 3, 4]);
    }

    #[test]
    fn tls12_chacha_record_round_trips() {
        let key = aead_key();
        let mut tx = RecordLayer::new();
        tx.set_write_keys(TrafficKeys::new(key.clone()), RecordProtection::Tls12ChaCha);
        tx.write_raw(ContentType::Handshake, b"payload").unwrap();
        let wire = tx.take_ciphertext();
        let header_len = usize::from(u16::from_be_bytes([wire[3], wire[4]]));
        assert_eq!(header_len, 7 + 16);

        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls12ChaCha);
        rx.feed_ciphertext(&wire);
        let rec = rx.read_raw().unwrap().unwrap();
        assert_eq!(rec.content_type, ContentType::Handshake);
        assert_eq!(rec.payload.as_slice(), &b"payload"[..]);
    }

    #[test]
    fn tls12_short_ciphertext_rejected_before_decrypt() {
        let key = aead_key();
        let mut rx = RecordLayer::new();
        rx.set_read_keys(TrafficKeys::new(key), RecordProtection::Tls12Gcm);
        rx.feed_ciphertext(&[23, 3, 3, 0, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
        let err = rx.read_raw().unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::BadRecordMac)),
            "err={err:?}"
        );
    }
}
