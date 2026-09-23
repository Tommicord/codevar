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

//! TLS client connection and handshake state machine.
//!
//! Supports:
//! - TLS 1.3 full handshake with (EC)DHE, optional HelloRetryRequest
//! - TLS 1.2 ECDHE_RSA / ECDHE_ECDSA with AEAD suites and Extended Master Secret

use crate::tls_alert::AlertDescription;
use crate::tls_cert::ServerName;
use crate::tls_connection::{CommonState, ConnectionState, IoState};
use crate::tls_connection_conf::ClientConfig;
use crate::tls_crypto_random::random_array;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_extensions::{
    encode_alpn, encode_ec_point_formats, encode_key_share_client,
    encode_renegotiation_info_empty, encode_server_name, encode_signature_algorithms,
    encode_supported_groups, encode_supported_versions_client, finish_extensions,
    put_extension, start_extensions,
};
use crate::tls_handshake::{
    CertificateTls12, CertificateTls13, CertificateVerify, ClientKeyExchangeEcdhe,
    Finished, HandshakeMessage, ServerHello, ServerKeyExchangeEcdhe,
    build_client_hello_body, encode_handshake, parse_encrypted_extensions,
};
use crate::tls_ids::{
    CipherSuite, ContentType, DOWNGRADE_TLS11_SENTINEL, DOWNGRADE_TLS12_SENTINEL,
    ExtensionType, HandshakeType, NamedGroup, ProtocolVersion,
};
use crate::tls_key_schedule::{Tls12Keys, Tls13KeySchedule, finished_label};
use crate::tls_kx::{KeySharePrivate, KeySharePublic, generate_key_share, shared_secret};
use crate::tls_prf::ct_eq;
use crate::tls_sign::{
    tls13_cert_verify_content, verify_handshake_signature, verify_raw_signature,
};
use std::sync::Arc;

/// Client handshake progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClientHs {
    /// Waiting for ServerHello / HelloRetryRequest.
    ExpectServerHello,
    /// TLS 1.3: waiting for EncryptedExtensions.
    Tls13ExpectEe,
    /// TLS 1.3: waiting for Certificate.
    Tls13ExpectCertificate,
    /// TLS 1.3: waiting for CertificateVerify.
    Tls13ExpectCertVerify,
    /// TLS 1.3: waiting for Finished.
    Tls13ExpectFinished,
    /// TLS 1.2: waiting for Certificate.
    Tls12ExpectCertificate,
    /// TLS 1.2: waiting for ServerKeyExchange.
    Tls12ExpectSke,
    /// TLS 1.2: waiting for ServerHelloDone.
    Tls12ExpectShd,
    /// TLS 1.2: waiting for CCS.
    Tls12ExpectCcs,
    /// TLS 1.2: waiting for Finished.
    Tls12ExpectFinished,
    /// Handshake complete.
    Complete,
}

/// Ephemeral client key-exchange state.
struct ClientKx {
    group: NamedGroup,
    private: KeySharePrivate,
    public: KeySharePublic,
}

/// TLS client connection (I/O-agnostic).
pub struct TlsClientConnection {
    config: Arc<ClientConfig>,
    server_name: ServerName,
    common: CommonState,
    hs: ClientHs,
    client_random: [u8; 32],
    session_id: Vec<u8>,
    kx: Option<ClientKx>,
    /// Retained first ClientHello for HRR transcript replacement.
    first_client_hello: Option<Vec<u8>>,
    hrr_seen: bool,
    cookie: Option<Vec<u8>>,
    /// TLS 1.3 schedule after ServerHello.
    tls13: Option<Tls13KeySchedule>,
    /// TLS 1.2 keys after ClientKeyExchange.
    tls12: Option<Tls12Keys>,
    server_random: Option<[u8; 32]>,
    leaf_spki: Option<Vec<u8>>,
    /// Peer ECDHE public (TLS 1.2).
    tls12_peer_kx: Option<KeySharePublic>,
    ems: bool,
}

impl TlsClientConnection {
    /// Starts a client handshake toward `server_name`.
    pub fn new(config: Arc<ClientConfig>, server_name: ServerName) -> TlsResult<Self> {
        let client_random = random_array()?;
        // Middlebox compatibility: non-empty legacy session_id for TLS 1.3.
        let session_id = random_array::<32>()?.to_vec();
        let group = config
            .named_groups
            .first()
            .copied()
            .ok_or_else(|| TlsError::Internal("no named groups configured".into()))?;
        let (private, public) = generate_key_share(group)?;
        let mut client = Self {
            config,
            server_name,
            common: CommonState::new(),
            hs: ClientHs::ExpectServerHello,
            client_random,
            session_id,
            kx: Some(ClientKx {
                group,
                private,
                public,
            }),
            first_client_hello: None,
            hrr_seen: false,
            cookie: None,
            tls13: None,
            tls12: None,
            server_random: None,
            leaf_spki: None,
            tls12_peer_kx: None,
            ems: false,
        };
        client.send_client_hello()?;
        Ok(client)
    }

    /// Negotiated protocol version after the handshake completes.
    #[must_use]
    pub fn protocol_version(&self) -> Option<ProtocolVersion> {
        self.common.version
    }

    /// Negotiated cipher suite.
    #[must_use]
    pub fn negotiated_cipher_suite(&self) -> Option<CipherSuite> {
        self.common.suite
    }

    /// Selected ALPN protocol.
    #[must_use]
    pub fn alpn_protocol(&self) -> Option<&[u8]> {
        self.common.alpn.as_deref()
    }

    /// Whether the handshake is still running.
    #[must_use]
    pub fn is_handshaking(&self) -> bool {
        self.common.is_handshaking()
    }

    /// Whether outbound TLS bytes are pending.
    #[must_use]
    pub fn wants_write(&self) -> bool {
        self.common.wants_write()
    }

    /// Whether more network input is needed.
    #[must_use]
    pub fn wants_read(&self) -> bool {
        self.common.wants_read()
    }

    /// Feeds ciphertext from the transport.
    pub fn read_tls(&mut self, data: &[u8]) -> TlsResult<()> {
        self.common.record.feed_ciphertext(data);
        Ok(())
    }

    /// Drains ciphertext destined for the transport.
    pub fn write_tls(&mut self) -> Vec<u8> {
        self.common.record.take_ciphertext()
    }

    /// Processes buffered records and advances the handshake / decrypts app data.
    pub fn process_new_packets(&mut self) -> TlsResult<IoState> {
        loop {
            let record = match self.common.record.read_raw() {
                Ok(Some(r)) => r,
                Ok(None) => break,
                Err(e) => return Err(self.common.fail(e)),
            };
            if let Err(e) = self.dispatch_record(record.content_type, &record.payload) {
                return Err(self.common.fail(e));
            }
        }
        Ok(self.io_state())
    }

    /// Reads decrypted application data.
    pub fn reader_read(&mut self, buf: &mut [u8]) -> TlsResult<usize> {
        self.common.read_app(buf)
    }

    /// Encrypts application data.
    pub fn writer_write(&mut self, data: &[u8]) -> TlsResult<()> {
        self.common.write_app(data)
    }

    /// Sends `close_notify`.
    pub fn send_close_notify(&mut self) -> TlsResult<()> {
        self.common.send_close_notify()
    }

    fn dispatch_record(&mut self, ty: ContentType, payload: &[u8]) -> TlsResult<()> {
        match ty {
            ContentType::Alert => self.common.handle_alert(payload),
            ContentType::ChangeCipherSpec => {
                if payload != [1] {
                    return Err(TlsError::Alert(AlertDescription::UnexpectedMessage));
                }
                if self.hs == ClientHs::Tls12ExpectCcs {
                    self.arm_tls12_read_after_ccs()?;
                    self.hs = ClientHs::Tls12ExpectFinished;
                }
                // TLS 1.3: ignore CCS for middlebox compatibility.
                Ok(())
            }
            ContentType::Handshake => {
                self.common.hs_rx.push(payload);
                while let Some(msg) = self.common.hs_rx.pop_message()? {
                    self.process_handshake(msg)?;
                }
                Ok(())
            }
            ContentType::ApplicationData => {
                if self.common.is_handshaking() {
                    return Err(TlsError::Alert(AlertDescription::UnexpectedMessage));
                }
                self.common.queue_appdata(payload);
                Ok(())
            }
        }
    }

    fn process_handshake(&mut self, msg: HandshakeMessage) -> TlsResult<()> {
        let raw = msg.encode();
        match (self.hs, msg.msg_type) {
            (ClientHs::ExpectServerHello, HandshakeType::ServerHello) => {
                self.on_server_hello(&msg.body, raw)
            }
            (ClientHs::Tls13ExpectEe, HandshakeType::EncryptedExtensions) => {
                self.common.transcript.add_message(&raw);
                let exts = parse_encrypted_extensions(&msg.body)?;
                if let Some(alpn) =
                    exts.selected_alpn.or_else(|| exts.alpn.first().cloned())
                {
                    self.common.alpn = Some(alpn);
                }
                self.hs = ClientHs::Tls13ExpectCertificate;
                Ok(())
            }
            (ClientHs::Tls13ExpectCertificate, HandshakeType::Certificate) => {
                self.common.transcript.add_message(&raw);
                let cert = CertificateTls13::parse(&msg.body)?;
                let leaf = self.config.verifier.verify_server_cert(
                    &cert.cert_chain,
                    Some(self.server_name.as_str()),
                )?;
                self.leaf_spki = Some(leaf.spki_der);
                self.hs = ClientHs::Tls13ExpectCertVerify;
                Ok(())
            }
            (ClientHs::Tls13ExpectCertVerify, HandshakeType::CertificateVerify) => {
                self.on_tls13_cert_verify(&msg.body, raw)
            }
            (ClientHs::Tls13ExpectFinished, HandshakeType::Finished) => {
                self.on_tls13_finished(&msg.body, raw)
            }
            (ClientHs::Tls12ExpectCertificate, HandshakeType::Certificate) => {
                self.common.transcript.add_message(&raw);
                let cert = CertificateTls12::parse(&msg.body)?;
                let leaf = self.config.verifier.verify_server_cert(
                    &cert.cert_chain,
                    Some(self.server_name.as_str()),
                )?;
                self.leaf_spki = Some(leaf.spki_der);
                self.hs = ClientHs::Tls12ExpectSke;
                Ok(())
            }
            (ClientHs::Tls12ExpectSke, HandshakeType::ServerKeyExchange) => {
                self.on_tls12_ske(&msg.body, raw)
            }
            (ClientHs::Tls12ExpectShd, HandshakeType::ServerHelloDone) => {
                self.common.transcript.add_message(&raw);
                self.send_tls12_client_flight()
            }
            (ClientHs::Tls12ExpectFinished, HandshakeType::Finished) => {
                self.on_tls12_finished(&msg.body, raw)
            }
            _ => Err(TlsError::Alert(AlertDescription::UnexpectedMessage)),
        }
    }

    fn send_client_hello(&mut self) -> TlsResult<()> {
        let kx = self
            .kx
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing key share".into()))?;
        let mut ext_payload = Vec::new();
        let ext_idx = start_extensions(&mut ext_payload);

        put_extension(
            &mut ext_payload,
            ExtensionType::ServerName,
            &encode_server_name(self.server_name.as_str())?,
        )?;
        put_extension(
            &mut ext_payload,
            ExtensionType::SupportedVersions,
            &encode_supported_versions_client(&self.config.versions)?,
        )?;
        put_extension(
            &mut ext_payload,
            ExtensionType::SupportedGroups,
            &encode_supported_groups(&self.config.named_groups)?,
        )?;
        put_extension(
            &mut ext_payload,
            ExtensionType::KeyShare,
            &encode_key_share_client(&[KeySharePublic {
                group: kx.public.group,
                key_exchange: kx.public.key_exchange.clone(),
            }])?,
        )?;
        put_extension(
            &mut ext_payload,
            ExtensionType::SignatureAlgorithms,
            &encode_signature_algorithms(&self.config.signature_schemes)?,
        )?;
        if !self.config.alpn_protocols.is_empty() {
            put_extension(
                &mut ext_payload,
                ExtensionType::ApplicationLayerProtocolNegotiation,
                &encode_alpn(&self.config.alpn_protocols)?,
            )?;
        }
        // TLS 1.2 helpers
        put_extension(&mut ext_payload, ExtensionType::ExtendedMasterSecret, &[])?;
        put_extension(
            &mut ext_payload,
            ExtensionType::RenegotiationInfo,
            &encode_renegotiation_info_empty()?,
        )?;
        put_extension(
            &mut ext_payload,
            ExtensionType::EcPointFormats,
            &encode_ec_point_formats()?,
        )?;
        if let Some(cookie) = &self.cookie {
            let mut c = Vec::new();
            crate::tls_codec::put_vec_u16(&mut c, cookie)?;
            put_extension(&mut ext_payload, ExtensionType::Cookie, &c)?;
        }
        finish_extensions(&mut ext_payload, ext_idx)?;

        // The extensions block in ClientHello is the content after the u16 length
        // which build_client_hello_body will add, so pass only the extension entries.
        // start_extensions already wrote a length prefix into ext_payload; strip it.
        let extensions = ext_payload[2..].to_vec();

        let suites: Vec<u16> = self
            .config
            .cipher_suites
            .iter()
            .map(|s| s.as_u16())
            .collect();
        let body = build_client_hello_body(
            ProtocolVersion::Tls12,
            &self.client_random,
            &self.session_id,
            &suites,
            &extensions,
        )?;
        let message = encode_handshake(HandshakeType::ClientHello, &body);
        if !self.hrr_seen {
            self.first_client_hello = Some(message.clone());
        }
        self.common.send_handshake_raw(&message)?;
        Ok(())
    }

    fn on_server_hello(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let sh = ServerHello::parse(body, raw.clone())?;
        if sh.is_hello_retry_request() {
            return self.on_hello_retry_request(&sh, raw);
        }

        let version = sh.negotiated_version()?;
        if !self.config.versions.contains(&version) {
            return Err(TlsError::Alert(AlertDescription::ProtocolVersion));
        }
        // Downgrade sentinels (RFC 8446 §4.1.3)
        if self.config.versions.contains(&ProtocolVersion::Tls13)
            && version == ProtocolVersion::Tls12
        {
            if sh.random[24..] == DOWNGRADE_TLS12_SENTINEL {
                return Err(TlsError::Alert(AlertDescription::IllegalParameter));
            }
            if sh.random[24..] == DOWNGRADE_TLS11_SENTINEL {
                return Err(TlsError::Alert(AlertDescription::IllegalParameter));
            }
        }

        self.common.version = Some(version);
        self.common.suite = Some(sh.cipher_suite);
        self.server_random = Some(sh.random);
        self.common
            .transcript
            .rebind_hash(sh.cipher_suite.hash_algorithm());
        self.common.transcript.add_message(&raw);
        self.ems = sh.extensions.extended_master_secret;

        match version {
            ProtocolVersion::Tls13 => self.enter_tls13(&sh),
            ProtocolVersion::Tls12 => {
                if self.config.require_ems && !self.ems {
                    return Err(TlsError::Alert(AlertDescription::HandshakeFailure));
                }
                self.hs = ClientHs::Tls12ExpectCertificate;
                Ok(())
            }
            _ => Err(TlsError::Alert(AlertDescription::ProtocolVersion)),
        }
    }

    fn on_hello_retry_request(
        &mut self,
        sh: &ServerHello,
        raw: Vec<u8>,
    ) -> TlsResult<()> {
        if self.hrr_seen {
            return Err(TlsError::Alert(AlertDescription::UnexpectedMessage));
        }
        self.hrr_seen = true;
        let selected = sh
            .extensions
            .selected_group
            .ok_or_else(|| TlsError::Alert(AlertDescription::MissingExtension))?;
        if !self.config.named_groups.contains(&selected) {
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        self.cookie = sh.extensions.cookie.clone();

        // Transcript: replace ClientHello with message_hash, then add HRR.
        let first = self
            .first_client_hello
            .take()
            .ok_or_else(|| TlsError::Internal("missing first ClientHello".into()))?;
        self.common
            .transcript
            .rebind_hash(sh.cipher_suite.hash_algorithm());
        self.common
            .transcript
            .replace_client_hello_with_hash(&first);
        self.common.transcript.add_message(&raw);

        let (private, public) = generate_key_share(selected)?;
        self.kx = Some(ClientKx {
            group: selected,
            private,
            public,
        });
        // Middlebox CCS before second ClientHello
        self.common.send_ccs()?;
        self.send_client_hello()?;
        self.hs = ClientHs::ExpectServerHello;
        Ok(())
    }

    fn enter_tls13(&mut self, sh: &ServerHello) -> TlsResult<()> {
        let peer = sh
            .extensions
            .key_shares
            .first()
            .ok_or_else(|| TlsError::Alert(AlertDescription::MissingExtension))?;
        let kx = self
            .kx
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing kx".into()))?;
        if peer.group != kx.group {
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        let secret = shared_secret(&kx.private, peer)?;
        let hello_hash = self.common.transcript.hash();
        let schedule =
            Tls13KeySchedule::from_handshake(sh.cipher_suite, &secret, &hello_hash)?;

        // Read keys = server handshake; write keys installed after we send Finished.
        self.common.install_read_keys(
            schedule.server_handshake.traffic_keys(),
            ProtocolVersion::Tls13,
            sh.cipher_suite,
        );
        self.tls13 = Some(schedule);
        self.hs = ClientHs::Tls13ExpectEe;
        // Optional middlebox CCS
        Ok(())
    }

    fn on_tls13_cert_verify(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let cv = CertificateVerify::parse(body)?;
        if !cv.scheme.allowed_in_tls13_cert_verify() {
            return Err(TlsError::Alert(AlertDescription::IllegalParameter));
        }
        let spki = self
            .leaf_spki
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing leaf SPKI".into()))?;
        let th = self.common.transcript.hash();
        let content = tls13_cert_verify_content(true, &th);
        verify_handshake_signature(cv.scheme, spki, &content, &cv.signature)?;
        self.common.transcript.add_message(&raw);
        self.hs = ClientHs::Tls13ExpectFinished;
        Ok(())
    }

    fn on_tls13_finished(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let finished = Finished::parse(body);
        let schedule = self
            .tls13
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing TLS1.3 schedule".into()))?;
        let expected = schedule.server_finished_verify(&self.common.transcript.hash())?;
        if !ct_eq(&expected, &finished.verify_data) {
            return Err(TlsError::Alert(AlertDescription::DecryptError));
        }
        self.common.transcript.add_message(&raw);

        // Derive application secrets with hash including server Finished.
        let hash_after_sf = self.common.transcript.hash();
        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("missing suite".into()))?;
        {
            let schedule = self
                .tls13
                .as_mut()
                .ok_or_else(|| TlsError::Internal("missing schedule".into()))?;
            schedule.derive_application_secrets(&hash_after_sf)?;
        }

        // Send CCS (middlebox) + client Finished under handshake keys.
        self.common.send_ccs()?;
        {
            let schedule = self
                .tls13
                .as_ref()
                .ok_or_else(|| TlsError::Internal("missing schedule".into()))?;
            self.common.install_write_keys(
                schedule.client_handshake.traffic_keys(),
                ProtocolVersion::Tls13,
                suite,
            );
            let vd = schedule.client_finished_verify(&self.common.transcript.hash())?;
            let fin = encode_handshake(HandshakeType::Finished, &Finished::encode(&vd));
            self.common.send_handshake_raw(&fin)?;
        }

        // Switch both directions to application traffic keys.
        let schedule = self
            .tls13
            .as_mut()
            .ok_or_else(|| TlsError::Internal("missing schedule".into()))?;
        let hash_after_cf = self.common.transcript.hash();
        schedule.derive_resumption_master(&hash_after_cf)?;
        let client_app = schedule
            .client_application
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing client app secrets".into()))?
            .traffic_keys();
        let server_app = schedule
            .server_application
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing server app secrets".into()))?
            .traffic_keys();
        self.common
            .install_write_keys(client_app, ProtocolVersion::Tls13, suite);
        self.common
            .install_read_keys(server_app, ProtocolVersion::Tls13, suite);

        self.hs = ClientHs::Complete;
        self.common.state = ConnectionState::Connected;
        Ok(())
    }

    fn on_tls12_ske(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let ske = ServerKeyExchangeEcdhe::parse(body)?;
        let spki = self
            .leaf_spki
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing SPKI".into()))?;
        let server_random = self
            .server_random
            .ok_or_else(|| TlsError::Internal("missing server random".into()))?;
        let signed = ServerKeyExchangeEcdhe::signed_content(
            &self.client_random,
            &server_random,
            ske.curve,
            &ske.public_key,
        );
        verify_raw_signature(ske.scheme, spki, &signed, &ske.signature)?;
        self.tls12_peer_kx = Some(KeySharePublic {
            group: ske.curve,
            key_exchange: ske.public_key,
        });
        // Regenerate client share for the negotiated curve if needed.
        if self.kx.as_ref().map(|k| k.group) != Some(ske.curve) {
            let (private, public) = generate_key_share(ske.curve)?;
            self.kx = Some(ClientKx {
                group: ske.curve,
                private,
                public,
            });
        }
        self.common.transcript.add_message(&raw);
        self.hs = ClientHs::Tls12ExpectShd;
        Ok(())
    }

    fn send_tls12_client_flight(&mut self) -> TlsResult<()> {
        let kx = self
            .kx
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing kx".into()))?;
        let peer = self
            .tls12_peer_kx
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing peer kx".into()))?;
        let pms = shared_secret(&kx.private, peer)?;
        let cke = encode_handshake(
            HandshakeType::ClientKeyExchange,
            &ClientKeyExchangeEcdhe::encode(&kx.public.key_exchange)?,
        );
        self.common.send_handshake_raw(&cke)?;

        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("missing suite".into()))?;
        let server_random = self
            .server_random
            .ok_or_else(|| TlsError::Internal("missing server random".into()))?;
        let session_hash = if self.ems {
            Some(self.common.transcript.hash())
        } else {
            None
        };
        let keys = Tls12Keys::derive(
            suite,
            &pms,
            &self.client_random,
            &server_random,
            session_hash.as_deref(),
        )?;

        self.common.send_ccs()?;
        self.common.install_write_keys(
            keys.client_traffic(),
            ProtocolVersion::Tls12,
            suite,
        );

        let vd =
            keys.finished_verify(finished_label::CLIENT, &self.common.transcript.hash())?;
        let fin = encode_handshake(HandshakeType::Finished, &Finished::encode(&vd));
        self.common.send_handshake_raw(&fin)?;

        // Keep reading cleartext until the server CCS arrives, then arm AEAD.
        self.tls12 = Some(keys);
        self.hs = ClientHs::Tls12ExpectCcs;
        Ok(())
    }

    fn on_tls12_finished(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let finished = Finished::parse(body);
        let keys = self
            .tls12
            .as_ref()
            .ok_or_else(|| TlsError::Internal("missing TLS1.2 keys".into()))?;
        let expected =
            keys.finished_verify(finished_label::SERVER, &self.common.transcript.hash())?;
        if !ct_eq(&expected, &finished.verify_data) {
            return Err(TlsError::Alert(AlertDescription::DecryptError));
        }
        self.common.transcript.add_message(&raw);
        self.hs = ClientHs::Complete;
        self.common.state = ConnectionState::Connected;
        Ok(())
    }
}

impl TlsClientConnection {
    /// Accurate I/O interest snapshot.
    #[must_use]
    pub fn io_state(&self) -> IoState {
        IoState {
            plaintext_bytes_to_read: self.common.app_rx.len(),
            tls_bytes_to_write: self.common.record.pending_tx_len(),
            handshaking: self.common.is_handshaking(),
        }
    }

    /// Arms TLS 1.2 read keys after CCS.
    fn arm_tls12_read_after_ccs(&mut self) -> TlsResult<()> {
        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;
        let keys = self
            .tls12
            .as_ref()
            .ok_or_else(|| TlsError::Internal("tls12 keys".into()))?
            .server_traffic();
        self.common
            .install_read_keys(keys, ProtocolVersion::Tls12, suite);
        Ok(())
    }
}
