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

//! Handshake message framing and common structures.

use crate::network::tls_alert::AlertDescription;
use crate::network::tls_codec::{
    Reader, fill_u24_len, put_u16, put_vec_u8, put_vec_u16, put_vec_u24, start_u24_vec,
};
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_extensions::ParsedExtensions;
use crate::network::tls_ids::{
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
    pub curve: crate::network::tls_ids::NamedGroup,
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
        let curve = crate::network::tls_ids::NamedGroup::from_u16(r.u16()?)
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
        curve: crate::network::tls_ids::NamedGroup,
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
        curve: crate::network::tls_ids::NamedGroup,
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
