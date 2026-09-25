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

//! TLS server connection and handshake state machine.

use crate::tls_alert::AlertDescription;
use crate::tls_connection::{CommonState, ConnectionState, IoState};
use crate::tls_connection_conf::{
    ServerConfig, select_alpn, select_cipher_suite, select_group, select_version,
};
use crate::tls_crypto_random::random_array;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_extensions::{
    encode_alpn_selected, encode_ec_point_formats, encode_key_share_hrr, encode_key_share_server,
    encode_renegotiation_info_empty, encode_supported_versions_server, finish_extensions, put_extension,
    start_extensions,
};
use crate::tls_handshake::{
    CertificateTls12, CertificateTls13, CertificateVerify, ClientHello, ClientKeyExchangeEcdhe, Finished,
    HandshakeMessage, ServerKeyExchangeEcdhe, build_server_hello_body, encode_encrypted_extensions,
    encode_handshake,
};
use crate::tls_ids::{
    CipherSuite, ContentType, DOWNGRADE_TLS12_SENTINEL, ExtensionType, HELLO_RETRY_REQUEST_RANDOM,
    HandshakeType, NamedGroup, ProtocolVersion, SignatureScheme,
};
use crate::tls_key_schedule::{Tls12Keys, Tls13KeySchedule, finished_label};
use crate::tls_kx::{KeySharePrivate, KeySharePublic, generate_key_share, shared_secret};
use crate::tls_prf::ct_eq;
use crate::tls_sign::tls13_cert_verify_content;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerHs {
    ExpectClientHello,
    /// After HRR, expect the second ClientHello.
    ExpectSecondClientHello,
    /// TLS 1.3: expect client Finished.
    Tls13ExpectFinished,
    /// TLS 1.2: expect ClientKeyExchange.
    Tls12ExpectCke,
    /// TLS 1.2: expect CCS.
    Tls12ExpectCcs,
    /// TLS 1.2: expect Finished.
    Tls12ExpectFinished,
    Complete,
}

struct ServerKx {
    group: NamedGroup,
    private: KeySharePrivate,
    #[allow(dead_code)]
    public: KeySharePublic,
}

/// TLS server connection (I/O-agnostic).
pub struct TlsServerConnection {
    config: Arc<ServerConfig>,
    common: CommonState,
    hs: ServerHs,
    server_random: [u8; 32],
    client_random: Option<[u8; 32]>,
    session_id_echo: Vec<u8>,
    kx: Option<ServerKx>,
    hrr_sent: bool,
    tls13: Option<Tls13KeySchedule>,
    tls12: Option<Tls12Keys>,
    ems: bool,
    /// Selected signature scheme for CertificateVerify / SKE.
    sig_scheme: Option<SignatureScheme>,
}

impl TlsServerConnection {
    /// Creates a server connection waiting for ClientHello.
    #[must_use]
    pub fn new(config: Arc<ServerConfig>) -> Self {
        Self {
            config,
            common: CommonState::new(),
            hs: ServerHs::ExpectClientHello,
            server_random: [0u8; 32],
            client_random: None,
            session_id_echo: Vec::new(),
            kx: None,
            hrr_sent: false,
            tls13: None,
            tls12: None,
            ems: false,
            sig_scheme: None,
        }
    }

    /// Negotiated protocol version.
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

    /// Processes buffered records.
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

    /// I/O interest snapshot.
    #[must_use]
    pub fn io_state(&self) -> IoState {
        IoState {
            plaintext_bytes_to_read: self.common.app_rx.len(),
            tls_bytes_to_write: self.common.record.pending_tx_len(),
            handshaking: self.common.is_handshaking(),
        }
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
                if self.hs == ServerHs::Tls12ExpectCcs {
                    let suite = self
                        .common
                        .suite
                        .ok_or_else(|| TlsError::Internal("suite".into()))?;
                    let keys = self
                        .tls12
                        .as_ref()
                        .ok_or_else(|| TlsError::Internal("tls12".into()))?
                        .client_traffic();
                    self.common
                        .install_read_keys(keys, ProtocolVersion::Tls12, suite);
                    self.hs = ServerHs::Tls12ExpectFinished;
                }
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
            (ServerHs::ExpectClientHello | ServerHs::ExpectSecondClientHello, HandshakeType::ClientHello) => {
                self.on_client_hello(&msg.body, raw)
            }
            (ServerHs::Tls13ExpectFinished, HandshakeType::Finished) => {
                self.on_tls13_finished(&msg.body, raw)
            }
            (ServerHs::Tls12ExpectCke, HandshakeType::ClientKeyExchange) => self.on_tls12_cke(&msg.body, raw),
            (ServerHs::Tls12ExpectFinished, HandshakeType::Finished) => {
                self.on_tls12_finished(&msg.body, raw)
            }
            _ => Err(TlsError::Alert(AlertDescription::UnexpectedMessage)),
        }
    }

    fn on_client_hello(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let ch = ClientHello::parse(body, raw.clone())?;
        let offered = ch.offered_versions();
        let version = select_version(&offered, &self.config.versions)?;
        let suite = select_cipher_suite(&self.config.cipher_suites, &ch.cipher_suites, version)?;

        self.client_random = Some(ch.random);
        self.session_id_echo = ch.session_id.clone();
        self.ems = ch.extensions.extended_master_secret;
        self.common.version = Some(version);
        self.common.suite = Some(suite);
        self.common.transcript.rebind_hash(suite.hash_algorithm());

        if version == ProtocolVersion::Tls12 && self.config.require_ems && !self.ems {
            return Err(TlsError::Alert(AlertDescription::HandshakeFailure));
        }

        // ALPN
        if let Some(selected) = select_alpn(&self.config.alpn_protocols, &ch.extensions.alpn) {
            self.common.alpn = Some(selected);
        } else if !self.config.alpn_protocols.is_empty() && !ch.extensions.alpn.is_empty() {
            return Err(TlsError::Alert(AlertDescription::NoApplicationProtocol));
        }

        match version {
            ProtocolVersion::Tls13 => self.handshake_tls13(ch, raw),
            ProtocolVersion::Tls12 => self.handshake_tls12(ch, raw),
            _ => Err(TlsError::Alert(AlertDescription::ProtocolVersion)),
        }
    }

    fn handshake_tls13(&mut self, ch: ClientHello, raw: Vec<u8>) -> TlsResult<()> {
        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;

        // Pick group: prefer one present in key_shares; otherwise HRR.
        let offered_groups = if ch.extensions.supported_groups.is_empty() {
            ch.extensions
                .key_shares
                .iter()
                .map(|k| k.group)
                .collect::<Vec<_>>()
        } else {
            ch.extensions.supported_groups.clone()
        };
        let group = select_group(&self.config.named_groups, &offered_groups)?;
        let client_share = ch.extensions.key_shares.iter().find(|s| s.group == group);

        if client_share.is_none() {
            // HelloRetryRequest
            if self.hrr_sent {
                return Err(TlsError::Alert(AlertDescription::UnexpectedMessage));
            }
            self.hrr_sent = true;
            self.common.transcript.replace_client_hello_with_hash(&raw);
            self.send_hello_retry_request(suite, group)?;
            self.hs = ServerHs::ExpectSecondClientHello;
            return Ok(());
        }

        if self.hrr_sent {
            // Transcript already has message_hash || HRR; add second ClientHello.
            self.common.transcript.add_message(&raw);
        } else {
            self.common.transcript.add_message(&raw);
        }

        let peer = client_share.ok_or(TlsError::Alert(AlertDescription::MissingExtension))?;
        let (private, public) = generate_key_share(group)?;
        let secret = shared_secret(&private, peer)?;
        self.kx = Some(ServerKx {
            group,
            private,
            public: public.clone(),
        });

        // Signature scheme
        let scheme = self
            .config
            .certified_key
            .key
            .select_scheme(&ch.extensions.signature_algorithms)?;
        if !scheme.allowed_in_tls13_cert_verify() {
            // Prefer TLS 1.3-allowed schemes; try again filtering.
            let filtered: Vec<SignatureScheme> = ch
                .extensions
                .signature_algorithms
                .iter()
                .copied()
                .filter(|s| s.allowed_in_tls13_cert_verify())
                .collect();
            self.sig_scheme = Some(self.config.certified_key.key.select_scheme(&filtered)?);
        } else {
            self.sig_scheme = Some(scheme);
        }

        self.server_random = random_array()?;
        self.send_tls13_server_flight(&public, secret)?;
        self.hs = ServerHs::Tls13ExpectFinished;
        Ok(())
    }

    fn send_hello_retry_request(&mut self, suite: CipherSuite, group: NamedGroup) -> TlsResult<()> {
        let mut ext = Vec::new();
        let idx = start_extensions(&mut ext);
        put_extension(
            &mut ext,
            ExtensionType::SupportedVersions,
            &encode_supported_versions_server(ProtocolVersion::Tls13),
        )?;
        put_extension(&mut ext, ExtensionType::KeyShare, &encode_key_share_hrr(group))?;
        finish_extensions(&mut ext, idx)?;
        let extensions = &ext[2..];
        let body = build_server_hello_body(
            ProtocolVersion::Tls12,
            &HELLO_RETRY_REQUEST_RANDOM,
            &self.session_id_echo,
            suite,
            extensions,
        )?;
        let msg = encode_handshake(HandshakeType::ServerHello, &body);
        self.common.send_handshake_raw(&msg)?;
        self.common.send_ccs()?;
        Ok(())
    }

    fn send_tls13_server_flight(
        &mut self,
        server_share: &KeySharePublic,
        ecdhe_secret: Vec<u8>,
    ) -> TlsResult<()> {
        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;

        // ServerHello
        let mut ext = Vec::new();
        let idx = start_extensions(&mut ext);
        put_extension(
            &mut ext,
            ExtensionType::SupportedVersions,
            &encode_supported_versions_server(ProtocolVersion::Tls13),
        )?;
        put_extension(
            &mut ext,
            ExtensionType::KeyShare,
            &encode_key_share_server(server_share)?,
        )?;
        finish_extensions(&mut ext, idx)?;
        let body = build_server_hello_body(
            ProtocolVersion::Tls12,
            &self.server_random,
            &self.session_id_echo,
            suite,
            &ext[2..],
        )?;
        let sh = encode_handshake(HandshakeType::ServerHello, &body);
        self.common.send_handshake_raw(&sh)?;

        let hello_hash = self.common.transcript.hash();
        let schedule = Tls13KeySchedule::from_handshake(suite, &ecdhe_secret, &hello_hash)?;

        // Middlebox CCS then encrypted flight under server handshake keys.
        self.common.send_ccs()?;
        self.common.install_write_keys(
            schedule.server_handshake.traffic_keys(),
            ProtocolVersion::Tls13,
            suite,
        );
        // Client handshake traffic for reading Finished later.
        self.common.install_read_keys(
            schedule.client_handshake.traffic_keys(),
            ProtocolVersion::Tls13,
            suite,
        );

        // EncryptedExtensions
        let mut ee_ext = Vec::new();
        let ee_idx = start_extensions(&mut ee_ext);
        if let Some(alpn) = &self.common.alpn {
            put_extension(
                &mut ee_ext,
                ExtensionType::ApplicationLayerProtocolNegotiation,
                &encode_alpn_selected(alpn)?,
            )?;
        }
        finish_extensions(&mut ee_ext, ee_idx)?;
        let ee_body = encode_encrypted_extensions(&ee_ext[2..])?;
        let ee = encode_handshake(HandshakeType::EncryptedExtensions, &ee_body);
        self.common.send_handshake_raw(&ee)?;

        // Certificate
        let cert_body = CertificateTls13::encode(&[], &self.config.certified_key.cert_chain)?;
        let cert = encode_handshake(HandshakeType::Certificate, &cert_body);
        self.common.send_handshake_raw(&cert)?;

        // CertificateVerify
        let scheme = self
            .sig_scheme
            .ok_or_else(|| TlsError::Internal("sig scheme".into()))?;
        let th = self.common.transcript.hash();
        let content = tls13_cert_verify_content(true, &th);
        let signature = self.config.certified_key.key.sign(scheme, &content)?;
        let cv_body = CertificateVerify::encode(scheme, &signature)?;
        let cv = encode_handshake(HandshakeType::CertificateVerify, &cv_body);
        self.common.send_handshake_raw(&cv)?;

        // Finished
        let vd = schedule.server_finished_verify(&self.common.transcript.hash())?;
        let fin = encode_handshake(HandshakeType::Finished, &Finished::encode(&vd));
        self.common.send_handshake_raw(&fin)?;

        // Application secrets (hash includes server Finished).
        let mut schedule = schedule;
        let hash_after_sf = self.common.transcript.hash();
        schedule.derive_application_secrets(&hash_after_sf)?;
        // Write keys switch to app after Finished sent; read stays handshake until client Finished.
        let server_app = schedule
            .server_application
            .as_ref()
            .ok_or_else(|| TlsError::Internal("server app".into()))?
            .traffic_keys();
        self.common
            .install_write_keys(server_app, ProtocolVersion::Tls13, suite);

        self.tls13 = Some(schedule);
        Ok(())
    }

    fn on_tls13_finished(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let finished = Finished::parse(body);
        let schedule = self
            .tls13
            .as_ref()
            .ok_or_else(|| TlsError::Internal("tls13".into()))?;
        let expected = schedule.client_finished_verify(&self.common.transcript.hash())?;
        if !ct_eq(&expected, &finished.verify_data) {
            return Err(TlsError::Alert(AlertDescription::DecryptError));
        }
        self.common.transcript.add_message(&raw);

        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;
        let schedule = self
            .tls13
            .as_mut()
            .ok_or_else(|| TlsError::Internal("tls13".into()))?;
        let hash_after_cf = self.common.transcript.hash();
        schedule.derive_resumption_master(&hash_after_cf)?;
        let client_app = schedule
            .client_application
            .as_ref()
            .ok_or_else(|| TlsError::Internal("client app".into()))?
            .traffic_keys();
        self.common
            .install_read_keys(client_app, ProtocolVersion::Tls13, suite);

        self.hs = ServerHs::Complete;
        self.common.state = ConnectionState::Connected;
        Ok(())
    }

    fn handshake_tls12(&mut self, ch: ClientHello, raw: Vec<u8>) -> TlsResult<()> {
        self.common.transcript.add_message(&raw);
        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;

        let group = select_group(
            &self.config.named_groups,
            if ch.extensions.supported_groups.is_empty() {
                NamedGroup::default_offered()
            } else {
                &ch.extensions.supported_groups
            },
        )?;
        let (private, public) = generate_key_share(group)?;
        self.kx = Some(ServerKx {
            group,
            private,
            public: public.clone(),
        });

        let offered_sigs = if ch.extensions.signature_algorithms.is_empty() {
            self.config.signature_schemes.clone()
        } else {
            ch.extensions.signature_algorithms.clone()
        };
        let scheme = self.config.certified_key.key.select_scheme(&offered_sigs)?;
        self.sig_scheme = Some(scheme);

        // Server random with downgrade sentinel if TLS 1.3 is also supported.
        let mut random = random_array::<32>()?;
        if self.config.versions.contains(&ProtocolVersion::Tls13) {
            random[24..].copy_from_slice(&DOWNGRADE_TLS12_SENTINEL);
        }
        self.server_random = random;

        // ServerHello
        let mut ext = Vec::new();
        let idx = start_extensions(&mut ext);
        if self.ems {
            put_extension(&mut ext, ExtensionType::ExtendedMasterSecret, &[])?;
        }
        put_extension(
            &mut ext,
            ExtensionType::RenegotiationInfo,
            &encode_renegotiation_info_empty()?,
        )?;
        put_extension(
            &mut ext,
            ExtensionType::EcPointFormats,
            &encode_ec_point_formats()?,
        )?;
        if let Some(alpn) = &self.common.alpn {
            put_extension(
                &mut ext,
                ExtensionType::ApplicationLayerProtocolNegotiation,
                &encode_alpn_selected(alpn)?,
            )?;
        }
        finish_extensions(&mut ext, idx)?;
        let sh_body = build_server_hello_body(
            ProtocolVersion::Tls12,
            &self.server_random,
            &self.session_id_echo,
            suite,
            &ext[2..],
        )?;
        let sh = encode_handshake(HandshakeType::ServerHello, &sh_body);
        self.common.send_handshake_raw(&sh)?;

        // Certificate
        let cert_body = CertificateTls12::encode(&self.config.certified_key.cert_chain)?;
        let cert = encode_handshake(HandshakeType::Certificate, &cert_body);
        self.common.send_handshake_raw(&cert)?;

        // ServerKeyExchange
        let client_random = self
            .client_random
            .ok_or_else(|| TlsError::Internal("client random".into()))?;
        let signed = ServerKeyExchangeEcdhe::signed_content(
            &client_random,
            &self.server_random,
            group,
            &public.key_exchange,
        );
        let signature = self.config.certified_key.key.sign(scheme, &signed)?;
        let ske_body = ServerKeyExchangeEcdhe::encode(group, &public.key_exchange, scheme, &signature)?;
        let ske = encode_handshake(HandshakeType::ServerKeyExchange, &ske_body);
        self.common.send_handshake_raw(&ske)?;

        // ServerHelloDone
        let shd = encode_handshake(HandshakeType::ServerHelloDone, &[]);
        self.common.send_handshake_raw(&shd)?;

        self.hs = ServerHs::Tls12ExpectCke;
        Ok(())
    }

    fn on_tls12_cke(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let cke = ClientKeyExchangeEcdhe::parse(body)?;
        self.common.transcript.add_message(&raw);
        let kx = self
            .kx
            .as_ref()
            .ok_or_else(|| TlsError::Internal("kx".into()))?;
        let peer = KeySharePublic {
            group: kx.group,
            key_exchange: cke.public_key,
        };
        let pms = shared_secret(&kx.private, &peer)?;
        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;
        let client_random = self
            .client_random
            .ok_or_else(|| TlsError::Internal("client random".into()))?;
        let session_hash = if self.ems {
            Some(self.common.transcript.hash())
        } else {
            None
        };
        let keys = Tls12Keys::derive(
            suite,
            &pms,
            &client_random,
            &self.server_random,
            session_hash.as_deref(),
        )?;
        self.tls12 = Some(keys);
        self.hs = ServerHs::Tls12ExpectCcs;
        Ok(())
    }

    fn on_tls12_finished(&mut self, body: &[u8], raw: Vec<u8>) -> TlsResult<()> {
        let finished = Finished::parse(body);
        let keys = self
            .tls12
            .as_ref()
            .ok_or_else(|| TlsError::Internal("tls12".into()))?;
        let expected = keys.finished_verify(finished_label::CLIENT, &self.common.transcript.hash())?;
        if !ct_eq(&expected, &finished.verify_data) {
            return Err(TlsError::Alert(AlertDescription::DecryptError));
        }
        self.common.transcript.add_message(&raw);

        let suite = self
            .common
            .suite
            .ok_or_else(|| TlsError::Internal("suite".into()))?;
        // Server CCS + Finished
        self.common.send_ccs()?;
        self.common
            .install_write_keys(keys.server_traffic(), ProtocolVersion::Tls12, suite);
        let vd = keys.finished_verify(finished_label::SERVER, &self.common.transcript.hash())?;
        let fin = encode_handshake(HandshakeType::Finished, &Finished::encode(&vd));
        self.common.send_handshake_raw(&fin)?;

        self.hs = ServerHs::Complete;
        self.common.state = ConnectionState::Connected;
        Ok(())
    }
}
