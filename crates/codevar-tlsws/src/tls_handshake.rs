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

//! Handshake message framing and common structures.

use crate::tls_alert::AlertDescription;
use crate::tls_codec::{
    Reader, fill_u24_len, put_u16, put_vec_u8, put_vec_u16, put_vec_u24, start_u24_vec,
};
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_extensions::ParsedExtensions;
use crate::tls_ids::{
    CipherSuite, HELLO_RETRY_REQUEST_RANDOM, HandshakeType, ProtocolVersion,
    SignatureScheme,
};

/// A framed handshake message.
#[derive(Debug, Clone)]
pub struct HandshakeMessage {
    /// Message type.
    pub msg_type: HandshakeType,
    /// Message body (excluding type and length).
    pub body: Vec<u8>,
}

impl HandshakeMessage {
    /// Encodes `type || uint24(length) || body`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.body.len());
        out.push(self.msg_type as u8);
        put_u24_len_body(&mut out, &self.body);
        out
    }

    /// Builds a message from type and body.
    #[must_use]
    pub fn new(msg_type: HandshakeType, body: Vec<u8>) -> Self {
        Self { msg_type, body }
    }
}

fn put_u24_len_body(out: &mut Vec<u8>, body: &[u8]) {
    let len = body.len();
    out.push(((len >> 16) & 0xff) as u8);
    out.push(((len >> 8) & 0xff) as u8);
    out.push((len & 0xff) as u8);
    out.extend_from_slice(body);
}

/// Reassembles handshake messages from a byte stream (record boundaries may split messages).
#[derive(Debug, Default)]
pub struct HandshakeReassembly {
    buf: Vec<u8>,
}

impl HandshakeReassembly {
    /// Appends decrypted handshake bytes.
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Clears buffered bytes.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Pops the next complete handshake message, if available.
    pub fn pop_message(&mut self) -> TlsResult<Option<HandshakeMessage>> {
        if self.buf.len() < 4 {
            return Ok(None);
        }
        let msg_type = HandshakeType::from_u8(self.buf[0])?;
        let len = u32::from_be_bytes([0, self.buf[1], self.buf[2], self.buf[3]]) as usize;
        if len > 256 * 1024 {
            return Err(TlsError::Alert(AlertDescription::DecodeError));
        }
        if self.buf.len() < 4 + len {
            return Ok(None);
        }
        let body = self.buf[4..4 + len].to_vec();
        self.buf.drain(..4 + len);
        Ok(Some(HandshakeMessage { msg_type, body }))
    }

    /// Returns true if buffered data remains.
    #[must_use]
    pub fn has_buffered(&self) -> bool {
        !self.buf.is_empty()
    }
}

/// Parsed ClientHello.
#[derive(Debug, Clone)]
pub struct ClientHello {
    /// Legacy version field.
    pub legacy_version: ProtocolVersion,
    /// Client random.
    pub random: [u8; 32],
    /// Legacy session id.
    pub session_id: Vec<u8>,
    /// Offered cipher suites.
    pub cipher_suites: Vec<u16>,
    /// Parsed extensions.
    pub extensions: ParsedExtensions,
    /// Full encoded handshake message (for transcript).
    pub raw_message: Vec<u8>,
}

impl ClientHello {
    /// Parses a ClientHello body (without handshake header).
    pub fn parse(body: &[u8], raw_message: Vec<u8>) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let legacy_raw = r.u16()?;
        let legacy_version =
            ProtocolVersion::from_u16(legacy_raw).unwrap_or(ProtocolVersion::Tls12);
        let random: [u8; 32] = r
            .bytes(32)?
            .try_into()
            .map_err(|_| TlsError::decode("client random"))?;
        let session_id = r.vec_u8()?.to_vec();
        if session_id.len() > 32 {
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        let suites = r.vec_u16()?;
        if suites.len() < 2 || !suites.len().is_multiple_of(2) {
            return Err(TlsError::Alert(AlertDescription::DecodeError));
        }
        let mut cipher_suites = Vec::new();
        let mut sr = Reader::new(suites);
        while !sr.is_empty() {
            cipher_suites.push(sr.u16()?);
        }
        let compression = r.vec_u8()?;
        if compression.is_empty() || !compression.contains(&0) {
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        let extensions = if r.is_empty() {
            ParsedExtensions::default()
        } else {
            let ext = r.vec_u16()?;
            r.expect_empty("client_hello")?;
            ParsedExtensions::parse(ext)?
        };
        Ok(Self {
            legacy_version,
            random,
            session_id,
            cipher_suites,
            extensions,
            raw_message,
        })
    }

    /// Negotiated version preference from extensions or legacy field.
    #[must_use]
    pub fn offered_versions(&self) -> Vec<ProtocolVersion> {
        if self.extensions.supported_versions.is_empty() {
            vec![self.legacy_version]
        } else {
            self.extensions.supported_versions.clone()
        }
    }
}

/// Parsed ServerHello / HelloRetryRequest.
#[derive(Debug, Clone)]
pub struct ServerHello {
    /// Legacy version.
    pub legacy_version: ProtocolVersion,
    /// Server random.
    pub random: [u8; 32],
    /// Echoed session id.
    pub session_id_echo: Vec<u8>,
    /// Selected cipher suite.
    pub cipher_suite: CipherSuite,
    /// Parsed extensions.
    pub extensions: ParsedExtensions,
    /// Full encoded handshake message.
    pub raw_message: Vec<u8>,
}

impl ServerHello {
    /// Parses a ServerHello body.
    pub fn parse(body: &[u8], raw_message: Vec<u8>) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let legacy_version =
            ProtocolVersion::from_u16(r.u16()?).unwrap_or(ProtocolVersion::Tls12);
        let random: [u8; 32] = r
            .bytes(32)?
            .try_into()
            .map_err(|_| TlsError::decode("server random"))?;
        let session_id_echo = r.vec_u8()?.to_vec();
        let suite_code = r.u16()?;
        let cipher_suite = CipherSuite::from_u16(suite_code)
            .ok_or_else(|| TlsError::Alert(AlertDescription::HandshakeFailure))?;
        let compression = r.u8()?;
        if compression != 0 {
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        let extensions = if r.is_empty() {
            ParsedExtensions::default()
        } else {
            let ext = r.vec_u16()?;
            r.expect_empty("server_hello")?;
            ParsedExtensions::parse(ext)?
        };
        Ok(Self {
            legacy_version,
            random,
            session_id_echo,
            cipher_suite,
            extensions,
            raw_message,
        })
    }

    /// Returns true if this is a HelloRetryRequest.
    #[must_use]
    pub fn is_hello_retry_request(&self) -> bool {
        self.random == HELLO_RETRY_REQUEST_RANDOM
    }

    /// Negotiated protocol version.
    pub fn negotiated_version(&self) -> TlsResult<ProtocolVersion> {
        if let Some(v) = self.extensions.supported_versions.first() {
            return Ok(*v);
        }
        // No supported_versions => TLS 1.2 ServerHello
        if self.legacy_version == ProtocolVersion::Tls12 {
            Ok(ProtocolVersion::Tls12)
        } else {
            Err(TlsError::Alert(AlertDescription::ProtocolVersion))
        }
    }
}

/// Encodes a handshake header around `body`.
pub fn encode_handshake(msg_type: HandshakeType, body: &[u8]) -> Vec<u8> {
    HandshakeMessage::new(msg_type, body.to_vec()).encode()
}

/// Builds a ClientHello body (extensions already encoded).
pub fn build_client_hello_body(
    legacy_version: ProtocolVersion,
    random: &[u8; 32],
    session_id: &[u8],
    cipher_suites: &[u16],
    extensions: &[u8],
) -> TlsResult<Vec<u8>> {
    let mut body = Vec::new();
    put_u16(&mut body, legacy_version.as_u16());
    body.extend_from_slice(random);
    put_vec_u8(&mut body, session_id)?;
    let mut suites = Vec::new();
    for s in cipher_suites {
        put_u16(&mut suites, *s);
    }
    put_vec_u16(&mut body, &suites)?;
    put_vec_u8(&mut body, &[0])?; // null compression
    put_vec_u16(&mut body, extensions)?;
    Ok(body)
}

/// Builds a ServerHello body.
pub fn build_server_hello_body(
    legacy_version: ProtocolVersion,
    random: &[u8; 32],
    session_id_echo: &[u8],
    cipher_suite: CipherSuite,
    extensions: &[u8],
) -> TlsResult<Vec<u8>> {
    let mut body = Vec::new();
    put_u16(&mut body, legacy_version.as_u16());
    body.extend_from_slice(random);
    put_vec_u8(&mut body, session_id_echo)?;
    put_u16(&mut body, cipher_suite.as_u16());
    body.push(0); // compression
    put_vec_u16(&mut body, extensions)?;
    Ok(body)
}

/// TLS 1.3 Certificate message (RFC 8446 §4.4.2).
#[derive(Debug, Clone)]
pub struct CertificateTls13 {
    /// Certificate request context.
    pub request_context: Vec<u8>,
    /// DER certificates (leaf first).
    pub cert_chain: Vec<Vec<u8>>,
}

impl CertificateTls13 {
    /// Parses a TLS 1.3 Certificate body.
    pub fn parse(body: &[u8]) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let request_context = r.vec_u8()?.to_vec();
        let list = r.vec_u24()?;
        r.expect_empty("certificate")?;
        let mut lr = Reader::new(list);
        let mut cert_chain = Vec::new();
        while !lr.is_empty() {
            let cert = lr.vec_u24()?.to_vec();
            let _exts = lr.vec_u16()?; // certificate extensions
            cert_chain.push(cert);
        }
        Ok(Self {
            request_context,
            cert_chain,
        })
    }

    /// Encodes a TLS 1.3 Certificate body.
    pub fn encode(request_context: &[u8], cert_chain: &[Vec<u8>]) -> TlsResult<Vec<u8>> {
        let mut body = Vec::new();
        put_vec_u8(&mut body, request_context)?;
        let list_idx = start_u24_vec(&mut body);
        for cert in cert_chain {
            put_vec_u24(&mut body, cert)?;
            put_vec_u16(&mut body, &[])?; // no per-cert extensions
        }
        fill_u24_len(&mut body, list_idx)?;
        Ok(body)
    }
}

/// TLS 1.2 Certificate message (RFC 5246 §7.4.2).
#[derive(Debug, Clone)]
pub struct CertificateTls12 {
    /// DER certificates (leaf first).
    pub cert_chain: Vec<Vec<u8>>,
}

impl CertificateTls12 {
    /// Parses a TLS 1.2 Certificate body.
    pub fn parse(body: &[u8]) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let list = r.vec_u24()?;
        r.expect_empty("certificate")?;
        let mut lr = Reader::new(list);
        let mut cert_chain = Vec::new();
        while !lr.is_empty() {
            cert_chain.push(lr.vec_u24()?.to_vec());
        }
        Ok(Self { cert_chain })
    }

    /// Encodes a TLS 1.2 Certificate body.
    pub fn encode(cert_chain: &[Vec<u8>]) -> TlsResult<Vec<u8>> {
        let mut body = Vec::new();
        let idx = start_u24_vec(&mut body);
        for cert in cert_chain {
            put_vec_u24(&mut body, cert)?;
        }
        fill_u24_len(&mut body, idx)?;
        Ok(body)
    }
}

/// CertificateVerify message.
#[derive(Debug, Clone)]
pub struct CertificateVerify {
    /// Signature scheme.
    pub scheme: SignatureScheme,
    /// Signature bytes.
    pub signature: Vec<u8>,
}

impl CertificateVerify {
    /// Parses CertificateVerify.
    pub fn parse(body: &[u8]) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let scheme = SignatureScheme::from_u16(r.u16()?)
            .ok_or_else(|| TlsError::Alert(AlertDescription::IllegalParameter))?;
        let signature = r.vec_u16()?.to_vec();
        r.expect_empty("certificate_verify")?;
        Ok(Self { scheme, signature })
    }

    /// Encodes CertificateVerify.
    pub fn encode(scheme: SignatureScheme, signature: &[u8]) -> TlsResult<Vec<u8>> {
        let mut body = Vec::new();
        put_u16(&mut body, scheme.as_u16());
        put_vec_u16(&mut body, signature)?;
        Ok(body)
    }
}

/// Finished message (`verify_data`).
#[derive(Debug, Clone)]
pub struct Finished {
    /// Verify data.
    pub verify_data: Vec<u8>,
}

impl Finished {
    /// Parses Finished (body is verify_data).
    #[must_use]
    pub fn parse(body: &[u8]) -> Self {
        Self {
            verify_data: body.to_vec(),
        }
    }

    /// Encodes Finished.
    #[must_use]
    pub fn encode(verify_data: &[u8]) -> Vec<u8> {
        verify_data.to_vec()
    }
}

/// EncryptedExtensions body is just an extensions block.
pub fn parse_encrypted_extensions(body: &[u8]) -> TlsResult<ParsedExtensions> {
    let mut r = Reader::new(body);
    let ext = if r.is_empty() { &[][..] } else { r.vec_u16()? };
    r.expect_empty("encrypted_extensions")?;
    ParsedExtensions::parse(ext)
}

/// Encodes EncryptedExtensions from a raw extensions payload (already a vector body).
pub fn encode_encrypted_extensions(extensions_payload: &[u8]) -> TlsResult<Vec<u8>> {
    let mut body = Vec::new();
    put_vec_u16(&mut body, extensions_payload)?;
    Ok(body)
}

/// TLS 1.2 ECDHE ServerKeyExchange (RFC 8422).
#[derive(Debug, Clone)]
pub struct ServerKeyExchangeEcdhe {
    /// Named curve.
    pub curve: crate::tls_ids::NamedGroup,
    /// Uncompressed public key bytes.
    pub public_key: Vec<u8>,
    /// Signature scheme (TLS 1.2 SignatureAndHashAlgorithm mapped to scheme).
    pub scheme: SignatureScheme,
    /// Signature over `client_random || server_random || curve_params || public`.
    pub signature: Vec<u8>,
}

impl ServerKeyExchangeEcdhe {
    /// Parses an ECDHE ServerKeyExchange body.
    pub fn parse(body: &[u8]) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let curve_type = r.u8()?;
        if curve_type != 3 {
            // named_curve
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        let curve = crate::tls_ids::NamedGroup::from_u16(r.u16()?)
            .ok_or_else(|| TlsError::Alert(AlertDescription::IllegalParameter))?;
        let public_key = r.vec_u8()?.to_vec();
        let scheme = SignatureScheme::from_u16(r.u16()?)
            .ok_or_else(|| TlsError::Alert(AlertDescription::IllegalParameter))?;
        let signature = r.vec_u16()?.to_vec();
        r.expect_empty("server_key_exchange")?;
        Ok(Self {
            curve,
            public_key,
            scheme,
            signature,
        })
    }

    /// Encodes an ECDHE ServerKeyExchange body.
    pub fn encode(
        curve: crate::tls_ids::NamedGroup,
        public_key: &[u8],
        scheme: SignatureScheme,
        signature: &[u8],
    ) -> TlsResult<Vec<u8>> {
        let mut body = Vec::new();
        body.push(3); // named_curve
        put_u16(&mut body, curve.as_u16());
        put_vec_u8(&mut body, public_key)?;
        put_u16(&mut body, scheme.as_u16());
        put_vec_u16(&mut body, signature)?;
        Ok(body)
    }

    /// Builds the signed payload for ServerKeyExchange.
    #[must_use]
    pub fn signed_content(
        client_random: &[u8; 32],
        server_random: &[u8; 32],
        curve: crate::tls_ids::NamedGroup,
        public_key: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + 3 + 1 + public_key.len());
        out.extend_from_slice(client_random);
        out.extend_from_slice(server_random);
        out.push(3);
        out.extend_from_slice(&curve.as_u16().to_be_bytes());
        out.push(public_key.len() as u8);
        out.extend_from_slice(public_key);
        out
    }
}

/// TLS 1.2 ECDHE ClientKeyExchange: `opaque public<1..2^8-1>`.
#[derive(Debug, Clone)]
pub struct ClientKeyExchangeEcdhe {
    /// Client ephemeral public key.
    pub public_key: Vec<u8>,
}

impl ClientKeyExchangeEcdhe {
    /// Parses ClientKeyExchange.
    pub fn parse(body: &[u8]) -> TlsResult<Self> {
        let mut r = Reader::new(body);
        let public_key = r.vec_u8()?.to_vec();
        r.expect_empty("client_key_exchange")?;
        Ok(Self { public_key })
    }

    /// Encodes ClientKeyExchange.
    pub fn encode(public_key: &[u8]) -> TlsResult<Vec<u8>> {
        let mut body = Vec::new();
        put_vec_u8(&mut body, public_key)?;
        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_extensions::{encode_server_name, encode_supported_versions_client};
    use crate::tls_ids::{ExtensionType, NamedGroup};

    fn ext_block(entries: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (ty, data) in entries {
            out.extend_from_slice(&ty.to_be_bytes());
            out.extend_from_slice(&(data.len() as u16).to_be_bytes());
            out.extend_from_slice(data);
        }
        out
    }

    fn manual_ch_body(
        legacy: u16,
        random: &[u8; 32],
        session_id: &[u8],
        suites: &[u8],
        compression: &[u8],
        extensions: Option<&[u8]>,
    ) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&legacy.to_be_bytes());
        b.extend_from_slice(random);
        b.push(session_id.len() as u8);
        b.extend_from_slice(session_id);
        b.extend_from_slice(&(suites.len() as u16).to_be_bytes());
        b.extend_from_slice(suites);
        b.push(compression.len() as u8);
        b.extend_from_slice(compression);
        if let Some(ext) = extensions {
            b.extend_from_slice(&(ext.len() as u16).to_be_bytes());
            b.extend_from_slice(ext);
        }
        b
    }

    fn manual_sh_body(
        legacy: u16,
        random: &[u8; 32],
        session_id_echo: &[u8],
        suite: u16,
        compression: u8,
        extensions: Option<&[u8]>,
    ) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&legacy.to_be_bytes());
        b.extend_from_slice(random);
        b.push(session_id_echo.len() as u8);
        b.extend_from_slice(session_id_echo);
        b.extend_from_slice(&suite.to_be_bytes());
        b.push(compression);
        if let Some(ext) = extensions {
            b.extend_from_slice(&(ext.len() as u16).to_be_bytes());
            b.extend_from_slice(ext);
        }
        b
    }

    #[test]
    fn handshake_message_encode_frames_type_len_body() {
        let msg = HandshakeMessage::new(HandshakeType::ServerHelloDone, vec![1, 2, 3]);
        let enc = msg.encode();
        assert_eq!(enc, vec![14, 0, 0, 3, 1, 2, 3]);
        let framed = encode_handshake(HandshakeType::Finished, &[9; 40]);
        assert_eq!(framed.len(), 44);
        assert_eq!(framed[0], HandshakeType::Finished as u8);
        assert_eq!(&framed[1..4], &[0, 0, 40]);
        assert_eq!(&framed[4..], &[9; 40][..]);
    }

    #[test]
    fn reassembly_rejects_huge_declared_length_without_buffering_body() {
        let mut ra = HandshakeReassembly::default();
        // u24 length 0xFFFFFF with only the 4-byte header present.
        ra.push(&[1, 0xFF, 0xFF, 0xFF]);
        let err = ra.pop_message().unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::DecodeError));
        // Header is retained; no body bytes were ever supplied.
        assert!(ra.has_buffered());
    }

    #[test]
    fn reassembly_rejects_unknown_handshake_type() {
        let mut ra = HandshakeReassembly::default();
        ra.push(&[99, 0, 0, 0]);
        let err = ra.pop_message().unwrap_err();
        assert!(matches!(err, TlsError::Decode(_)));
        // Type 3 is also unassigned.
        let mut ra2 = HandshakeReassembly::default();
        ra2.push(&[3, 0, 0, 1, 0xFF]);
        assert!(ra2.pop_message().is_err());
    }

    #[test]
    fn reassembly_zero_length_message() {
        let mut ra = HandshakeReassembly::default();
        ra.push(&[20, 0, 0, 0]);
        let msg = ra.pop_message().unwrap().unwrap();
        assert_eq!(msg.msg_type, HandshakeType::Finished);
        assert!(msg.body.is_empty());
        assert!(!ra.has_buffered());
    }

    #[test]
    fn reassembly_waits_while_declared_length_exceeds_buffer() {
        let mut ra = HandshakeReassembly::default();
        ra.push(&[11, 0, 0, 10, 1, 2, 3]);
        assert!(ra.pop_message().unwrap().is_none());
        assert!(ra.has_buffered());
        ra.push(&[4, 5, 6, 7, 8, 9, 10]);
        let msg = ra.pop_message().unwrap().unwrap();
        assert_eq!(msg.msg_type, HandshakeType::Certificate);
        assert_eq!(msg.body.len(), 10);
        assert!(!ra.has_buffered());
    }

    #[test]
    fn reassembly_fragmented_message_fed_byte_by_byte() {
        let full = encode_handshake(HandshakeType::EncryptedExtensions, &[0x0A; 20]);
        let mut ra = HandshakeReassembly::default();
        for (i, byte) in full.iter().enumerate() {
            ra.push(&[*byte]);
            if i + 1 < full.len() {
                assert!(ra.pop_message().unwrap().is_none(), "early pop at {i}");
            }
        }
        let msg = ra.pop_message().unwrap().unwrap();
        assert_eq!(msg.msg_type, HandshakeType::EncryptedExtensions);
        assert_eq!(msg.body, vec![0x0A; 20]);
        assert!(!ra.has_buffered());
    }

    #[test]
    fn reassembly_pops_multiple_messages_and_clear_resets() {
        let mut ra = HandshakeReassembly::default();
        let a = HandshakeMessage::new(HandshakeType::NewSessionTicket, vec![1]).encode();
        let b = HandshakeMessage::new(HandshakeType::KeyUpdate, vec![0]).encode();
        let mut combined = a.clone();
        combined.extend_from_slice(&b);
        combined.extend_from_slice(&[20, 0, 0]); // partial trailer
        ra.push(&combined);
        let m1 = ra.pop_message().unwrap().unwrap();
        assert_eq!(m1.msg_type, HandshakeType::NewSessionTicket);
        let m2 = ra.pop_message().unwrap().unwrap();
        assert_eq!(m2.msg_type, HandshakeType::KeyUpdate);
        assert_eq!(m2.body, vec![0]);
        assert!(ra.pop_message().unwrap().is_none());
        assert!(ra.has_buffered());
        ra.clear();
        assert!(!ra.has_buffered());
    }

    #[test]
    fn reassembly_streams_ten_thousand_small_messages() {
        let mut ra = HandshakeReassembly::default();
        for i in 0..10_000u32 {
            let msg = HandshakeMessage::new(
                HandshakeType::NewSessionTicket,
                i.to_be_bytes().to_vec(),
            );
            ra.push(&msg.encode());
        }
        let mut count = 0usize;
        while let Some(msg) = ra.pop_message().unwrap() {
            assert_eq!(msg.msg_type, HandshakeType::NewSessionTicket);
            assert_eq!(msg.body.len(), 4);
            count += 1;
        }
        assert_eq!(count, 10_000);
    }

    #[test]
    fn client_hello_round_trip_and_offered_versions() {
        let random = [0x11u8; 32];
        let session_id = [0x22u8; 32];
        let sni = encode_server_name("example.com").unwrap();
        let sv = encode_supported_versions_client(&[
            ProtocolVersion::Tls13,
            ProtocolVersion::Tls12,
        ])
        .unwrap();
        let extensions = ext_block(&[
            (ExtensionType::ServerName as u16, sni),
            (ExtensionType::SupportedVersions as u16, sv),
        ]);
        let body = build_client_hello_body(
            ProtocolVersion::Tls12,
            &random,
            &session_id,
            &[0x1301, 0xC02F],
            &extensions,
        )
        .unwrap();
        let raw = encode_handshake(HandshakeType::ClientHello, &body);
        let ch = ClientHello::parse(&body, raw).unwrap();
        assert_eq!(ch.legacy_version, ProtocolVersion::Tls12);
        assert_eq!(ch.random, random);
        assert_eq!(ch.session_id, session_id);
        assert_eq!(ch.cipher_suites, vec![0x1301, 0xC02F]);
        assert_eq!(ch.extensions.server_name.as_deref(), Some("example.com"));
        assert_eq!(
            ch.offered_versions(),
            vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]
        );
    }

    #[test]
    fn client_hello_truncated_at_fixed_field_boundaries() {
        let random = [7u8; 32];
        let session_id = [8u8; 32];
        let body = build_client_hello_body(
            ProtocolVersion::Tls12,
            &random,
            &session_id,
            &[0x1301],
            &[],
        )
        .unwrap();
        // Cut points inside the fixed prefix (legacy + random + session id)
        // must always fail; the prefix ends at offset 69.
        for cut in 0..69 {
            assert!(
                ClientHello::parse(&body[..cut], Vec::new()).is_err(),
                "truncation at {cut} accepted"
            );
        }
        assert!(ClientHello::parse(&body, Vec::new()).is_ok());
    }

    #[test]
    fn client_hello_legacy_version_fallback_quirks() {
        let random = [1u8; 32];
        // Unknown legacy version falls back to TLS 1.2.
        let body = manual_ch_body(0x9999, &random, &[], &[0x13, 0x01], &[0], None);
        let ch = ClientHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(ch.legacy_version, ProtocolVersion::Tls12);
        // TLS 1.0 legacy version is preserved.
        let body = manual_ch_body(0x0301, &random, &[], &[0x13, 0x01], &[0], None);
        let ch = ClientHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(ch.legacy_version, ProtocolVersion::Tls10);
        assert_eq!(ch.offered_versions(), vec![ProtocolVersion::Tls10]);
        // Zero legacy version also falls back to TLS 1.2.
        let body = manual_ch_body(0x0000, &random, &[], &[0x13, 0x01], &[0], None);
        let ch = ClientHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(ch.legacy_version, ProtocolVersion::Tls12);
    }

    #[test]
    fn client_hello_random_all_zero_and_all_ff_accepted() {
        let zero = [0u8; 32];
        let ff = [0xFFu8; 32];
        for random in [zero, ff] {
            let body = manual_ch_body(0x0303, &random, &[], &[0x13, 0x01], &[0], None);
            let ch = ClientHello::parse(&body, Vec::new()).unwrap();
            assert_eq!(ch.random, random);
        }
    }

    #[test]
    fn client_hello_session_id_overlong_rejected() {
        let random = [3u8; 32];
        let body =
            manual_ch_body(0x0303, &random, &[0x44; 33], &[0x13, 0x01], &[0], None);
        let err = ClientHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
        // Exactly 32 bytes is accepted.
        let body =
            manual_ch_body(0x0303, &random, &[0x44; 32], &[0x13, 0x01], &[0], None);
        assert!(ClientHello::parse(&body, Vec::new()).is_ok());
    }

    #[test]
    fn client_hello_cipher_suite_list_edges() {
        let random = [5u8; 32];
        // Empty list.
        let body = manual_ch_body(0x0303, &random, &[], &[], &[0], None);
        let err = ClientHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::DecodeError));
        // Odd-length list.
        let body = manual_ch_body(0x0303, &random, &[], &[0x13], &[0], None);
        let err = ClientHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::DecodeError));
        // Declared length 0xFFFF with a tiny buffer: rejected, not allocated.
        let mut body = Vec::new();
        body.extend_from_slice(&0x0303u16.to_be_bytes());
        body.extend_from_slice(&random);
        body.push(0); // empty session id
        body.extend_from_slice(&0xFFFFu16.to_be_bytes());
        body.extend_from_slice(&[0x13, 0x01]);
        body.push(1);
        body.push(0);
        assert!(ClientHello::parse(&body, Vec::new()).is_err());
        // Single suite accepted.
        let body = manual_ch_body(0x0303, &random, &[], &[0x13, 0x01], &[0], None);
        let ch = ClientHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(ch.cipher_suites, vec![0x1301]);
        // Multiple suites accepted.
        let body =
            manual_ch_body(0x0303, &random, &[], &[0x13, 0x01, 0xC0, 0x2F], &[0], None);
        let ch = ClientHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(ch.cipher_suites, vec![0x1301, 0xC02F]);
    }

    #[test]
    fn client_hello_compression_method_rules() {
        let random = [6u8; 32];
        // Empty compression list rejected.
        let body = manual_ch_body(0x0303, &random, &[0], &[0x13, 0x01], &[], None);
        let err = ClientHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
        // No null method rejected.
        let body = manual_ch_body(0x0303, &random, &[0], &[0x13, 0x01], &[1], None);
        let err = ClientHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
        // Null present among extra methods accepted (quirk: only `contains(0)`).
        let body = manual_ch_body(0x0303, &random, &[0], &[0x13, 0x01], &[1, 0], None);
        assert!(ClientHello::parse(&body, Vec::new()).is_ok());
    }

    #[test]
    fn client_hello_trailing_bytes_rejected() {
        let random = [9u8; 32];
        let mut body = manual_ch_body(0x0303, &random, &[0], &[0x13, 0x01], &[0], None);
        body.push(0xFF);
        assert!(ClientHello::parse(&body, Vec::new()).is_err());
    }

    #[test]
    fn client_hello_sni_empty_overlong_and_embedded_nul() {
        let random = [4u8; 32];
        let long = "a".repeat(300);
        for host in ["", long.as_str(), "ex\0ample.com"] {
            let sni = encode_server_name(host).unwrap();
            let extensions = ext_block(&[(ExtensionType::ServerName as u16, sni)]);
            let body = manual_ch_body(
                0x0303,
                &random,
                &[0],
                &[0x13, 0x01],
                &[0],
                Some(&extensions),
            );
            let ch = ClientHello::parse(&body, Vec::new()).unwrap();
            assert_eq!(ch.extensions.server_name.as_deref(), Some(host));
        }
    }

    #[test]
    fn supported_versions_empty_list_and_unknown_only() {
        let random = [2u8; 32];
        // Empty list rejected.
        let extensions = ext_block(&[(ExtensionType::SupportedVersions as u16, vec![0])]);
        let body = manual_ch_body(
            0x0303,
            &random,
            &[0],
            &[0x13, 0x01],
            &[0],
            Some(&extensions),
        );
        let err = ClientHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::DecodeError));
        // Unknown-only list parses to empty; offered_versions falls back.
        let extensions =
            ext_block(&[(ExtensionType::SupportedVersions as u16, vec![2, 0x99, 0x99])]);
        let body = manual_ch_body(
            0x0303,
            &random,
            &[0],
            &[0x13, 0x01],
            &[0],
            Some(&extensions),
        );
        let ch = ClientHello::parse(&body, Vec::new()).unwrap();
        assert!(ch.extensions.supported_versions.is_empty());
        assert_eq!(ch.offered_versions(), vec![ProtocolVersion::Tls12]);
        // Odd-length list rejected.
        let extensions =
            ext_block(&[(ExtensionType::SupportedVersions as u16, vec![1, 3])]);
        let body = manual_ch_body(
            0x0303,
            &random,
            &[0],
            &[0x13, 0x01],
            &[0],
            Some(&extensions),
        );
        assert!(ClientHello::parse(&body, Vec::new()).is_err());
    }

    #[test]
    fn server_hello_parse_negotiated_version_edges() {
        let random = [0x33u8; 32];
        // TLS 1.3 via supported_versions extension.
        let exts = ext_block(&[(ExtensionType::SupportedVersions as u16, vec![3, 4])]);
        let body = manual_sh_body(0x0303, &random, &[0; 32], 0x1301, 0, Some(&exts));
        let sh = ServerHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(sh.negotiated_version().unwrap(), ProtocolVersion::Tls13);
        // No extension + legacy TLS 1.2.
        let body = manual_sh_body(0x0303, &random, &[0; 32], 0xC02F, 0, None);
        let sh = ServerHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(sh.negotiated_version().unwrap(), ProtocolVersion::Tls12);
        // Unknown legacy version falls back to TLS 1.2 (quirk).
        let body = manual_sh_body(0xABCD, &random, &[0; 32], 0xC02F, 0, None);
        let sh = ServerHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(sh.legacy_version, ProtocolVersion::Tls12);
        assert_eq!(sh.negotiated_version().unwrap(), ProtocolVersion::Tls12);
        // Legacy TLS 1.3 without supported_versions is a protocol error.
        let body = manual_sh_body(0x0304, &random, &[0; 32], 0x1301, 0, None);
        let sh = ServerHello::parse(&body, Vec::new()).unwrap();
        let err = sh.negotiated_version().unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::ProtocolVersion));
    }

    #[test]
    fn server_hello_hrr_random_detection() {
        let sid = [0u8; 32];
        let body = build_server_hello_body(
            ProtocolVersion::Tls12,
            &HELLO_RETRY_REQUEST_RANDOM,
            &sid,
            CipherSuite::TlsAes128GcmSha256,
            &[],
        )
        .unwrap();
        let sh = ServerHello::parse(&body, Vec::new()).unwrap();
        assert!(sh.is_hello_retry_request());
        for random in [[0u8; 32], [0xFFu8; 32], [1u8; 32]] {
            let body = build_server_hello_body(
                ProtocolVersion::Tls12,
                &random,
                &sid,
                CipherSuite::TlsAes128GcmSha256,
                &[],
            )
            .unwrap();
            let sh = ServerHello::parse(&body, Vec::new()).unwrap();
            assert!(!sh.is_hello_retry_request());
        }
    }

    #[test]
    fn server_hello_unknown_suite_and_bad_compression() {
        let random = [8u8; 32];
        // Unknown cipher suite.
        let body = manual_sh_body(0x0303, &random, &[], 0xFFFF, 0, None);
        let err = ServerHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::HandshakeFailure));
        // Non-null compression method.
        let body = manual_sh_body(0x0303, &random, &[], 0x1301, 1, None);
        let err = ServerHello::parse(&body, Vec::new()).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
        // Truncated random.
        assert!(ServerHello::parse(&random[..31], Vec::new()).is_err());
        // Trailing byte after extensions.
        let mut body = manual_sh_body(0x0303, &random, &[], 0x1301, 0, Some(&[]));
        body.push(0);
        assert!(ServerHello::parse(&body, Vec::new()).is_err());
    }

    #[test]
    fn certificate_tls13_round_trip_and_edges() {
        let chain = vec![vec![1u8, 2, 3], vec![4, 5, 6, 7]];
        let body = CertificateTls13::encode(&[], &chain).unwrap();
        let parsed = CertificateTls13::parse(&body).unwrap();
        assert!(parsed.request_context.is_empty());
        assert_eq!(parsed.cert_chain, chain);

        // Request context round trip.
        let body = CertificateTls13::encode(&[9, 8], &chain).unwrap();
        let parsed = CertificateTls13::parse(&body).unwrap();
        assert_eq!(parsed.request_context, vec![9, 8]);

        // Empty chain.
        let body = CertificateTls13::encode(&[], &[]).unwrap();
        let parsed = CertificateTls13::parse(&body).unwrap();
        assert!(parsed.cert_chain.is_empty());

        // Truncation.
        let full = CertificateTls13::encode(&[], &chain).unwrap();
        assert!(CertificateTls13::parse(&full[..full.len() - 1]).is_err());

        // Declared u24 list length 0xFFFFFF with a tiny buffer.
        assert!(CertificateTls13::parse(&[0, 0xFF, 0xFF, 0xFF]).is_err());
    }

    #[test]
    fn certificate_tls12_round_trip_and_declared_overflow() {
        let chain = vec![vec![0xAAu8; 3], vec![0xBB; 5]];
        let body = CertificateTls12::encode(&chain).unwrap();
        let parsed = CertificateTls12::parse(&body).unwrap();
        assert_eq!(parsed.cert_chain, chain);

        let empty = CertificateTls12::encode(&[]).unwrap();
        let parsed = CertificateTls12::parse(&empty).unwrap();
        assert!(parsed.cert_chain.is_empty());

        // Trailing garbage after the list.
        let mut bad = empty.clone();
        bad.push(1);
        assert!(CertificateTls12::parse(&bad).is_err());

        // Declared list length exceeds buffer (0xFFFFFF) — rejected without alloc.
        let err = CertificateTls12::parse(&[0xFF, 0xFF, 0xFF]).unwrap_err();
        assert!(matches!(err, TlsError::Decode(_)));

        // Inner cert length exceeds the outer list.
        let bad = vec![0, 0, 6, 0xFF, 0xFF, 0xFF];
        assert!(CertificateTls12::parse(&bad).is_err());
    }

    #[test]
    fn certificate_verify_round_trip_and_errors() {
        let body =
            CertificateVerify::encode(SignatureScheme::EcdsaSecp256r1Sha256, &[1, 2])
                .unwrap();
        let cv = CertificateVerify::parse(&body).unwrap();
        assert_eq!(cv.scheme, SignatureScheme::EcdsaSecp256r1Sha256);
        assert_eq!(cv.signature, vec![1, 2]);

        // Unknown scheme.
        assert!(CertificateVerify::parse(&[0xFF, 0xFF, 0, 0]).is_err());
        // Trailing bytes.
        let mut trailing = body.clone();
        trailing.push(0);
        assert!(CertificateVerify::parse(&trailing).is_err());
        // Declared signature length exceeds buffer.
        assert!(CertificateVerify::parse(&[0x04, 0x03, 0xFF, 0xFF, 1]).is_err());
    }

    #[test]
    fn finished_is_passthrough_for_verify_data() {
        // Finished framing here is raw verify_data; mismatched data is kept
        // verbatim for the key-schedule comparison in client/server.
        let vd = vec![0xCDu8; 32];
        assert_eq!(Finished::encode(&vd), vd);
        let parsed = Finished::parse(&vd);
        assert_eq!(parsed.verify_data, vd);
        let wrong = Finished::parse(&[0u8; 32]);
        assert_ne!(wrong.verify_data, vd);
        let empty = Finished::parse(&[]);
        assert!(empty.verify_data.is_empty());
    }

    #[test]
    fn encrypted_extensions_round_trip_and_trailing() {
        let default = parse_encrypted_extensions(&[]).unwrap();
        assert!(default.raw.is_empty());

        let entries = ext_block(&[(0x1234, vec![7, 7])]);
        let body = encode_encrypted_extensions(&entries).unwrap();
        let parsed = parse_encrypted_extensions(&body).unwrap();
        assert_eq!(parsed.raw.len(), 1);
        assert_eq!(parsed.raw[0].ext_type, 0x1234);

        // Trailing byte after the u16 extension vector.
        let mut bad = body;
        bad.push(0);
        assert!(parse_encrypted_extensions(&bad).is_err());
    }

    #[test]
    fn server_key_exchange_ecdhe_round_trip_and_edges() {
        let pk = vec![0x04; 65];
        let sig = vec![0x99; 70];
        let body = ServerKeyExchangeEcdhe::encode(
            NamedGroup::Secp256r1,
            &pk,
            SignatureScheme::EcdsaSecp256r1Sha256,
            &sig,
        )
        .unwrap();
        let ske = ServerKeyExchangeEcdhe::parse(&body).unwrap();
        assert_eq!(ske.curve, NamedGroup::Secp256r1);
        assert_eq!(ske.public_key, pk);
        assert_eq!(ske.scheme, SignatureScheme::EcdsaSecp256r1Sha256);
        assert_eq!(ske.signature, sig);

        // Wrong curve type.
        let mut bad = body.clone();
        bad[0] = 2;
        let err = ServerKeyExchangeEcdhe::parse(&bad).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
        // Unknown named group.
        let mut bad = body.clone();
        bad[1] = 0xFF;
        bad[2] = 0xFF;
        assert!(ServerKeyExchangeEcdhe::parse(&bad).is_err());
        // Trailing bytes.
        let mut bad = body.clone();
        bad.push(0);
        assert!(ServerKeyExchangeEcdhe::parse(&bad).is_err());
        // Truncation.
        assert!(ServerKeyExchangeEcdhe::parse(&body[..body.len() - 1]).is_err());

        // signed_content layout: client_random || server_random || curve_params
        let cr = [1u8; 32];
        let sr = [2u8; 32];
        let signed =
            ServerKeyExchangeEcdhe::signed_content(&cr, &sr, NamedGroup::X25519, &pk);
        assert_eq!(signed.len(), 32 + 32 + 1 + 2 + 1 + pk.len());
        assert_eq!(&signed[..32], &cr);
        assert_eq!(&signed[32..64], &sr);
        assert_eq!(signed[64], 3);
        assert_eq!(&signed[65..67], &0x001du16.to_be_bytes());
        assert_eq!(signed[67], pk.len() as u8);
    }

    #[test]
    fn client_key_exchange_edges() {
        let pk = vec![0x04; 65];
        let body = ClientKeyExchangeEcdhe::encode(&pk).unwrap();
        let cke = ClientKeyExchangeEcdhe::parse(&body).unwrap();
        assert_eq!(cke.public_key, pk);

        // Empty public key accepted by the parser (RFC says 1..255; quirk).
        let empty = ClientKeyExchangeEcdhe::encode(&[]).unwrap();
        let cke = ClientKeyExchangeEcdhe::parse(&empty).unwrap();
        assert!(cke.public_key.is_empty());

        // Declared u8 length exceeds buffer.
        assert!(ClientKeyExchangeEcdhe::parse(&[5, 1, 2]).is_err());
        // Trailing bytes.
        let mut trailing = body.clone();
        trailing.push(0);
        assert!(ClientKeyExchangeEcdhe::parse(&trailing).is_err());
    }

    #[test]
    fn build_client_hello_body_rejects_oversized_session_id() {
        let random = [0u8; 32];
        let err = build_client_hello_body(
            ProtocolVersion::Tls12,
            &random,
            &[0u8; 256],
            &[0x1301],
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, TlsError::Internal(_)));
        // Empty cipher suite list builds fine (rejected later by the parser).
        let body =
            build_client_hello_body(ProtocolVersion::Tls12, &random, &[], &[], &[])
                .unwrap();
        assert!(ClientHello::parse(&body, Vec::new()).is_err());
    }

    #[test]
    fn build_server_hello_body_round_trip_with_session_echo() {
        let random = [0x55u8; 32];
        let sid = [0x66u8; 32];
        let body = build_server_hello_body(
            ProtocolVersion::Tls12,
            &random,
            &sid,
            CipherSuite::TlsChacha20Poly1305Sha256,
            &[],
        )
        .unwrap();
        let sh = ServerHello::parse(&body, Vec::new()).unwrap();
        assert_eq!(sh.session_id_echo, sid);
        assert_eq!(sh.cipher_suite, CipherSuite::TlsChacha20Poly1305Sha256);
        assert_eq!(sh.random, random);
    }
}
