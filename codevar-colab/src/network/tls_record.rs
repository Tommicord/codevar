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

use crate::network::tls_aead::{AeadKey, TlsAead};
use crate::network::tls_alert::AlertDescription;
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_ids::{
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
