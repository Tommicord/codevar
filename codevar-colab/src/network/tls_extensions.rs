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

use crate::network::tls_alert::AlertDescription;
use crate::network::tls_codec::{
    Reader, fill_u8_len, fill_u16_len, put_u16, put_vec_u8, put_vec_u16, start_u8_vec,
    start_u16_vec,
};
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_ids::{
    ExtensionType, NamedGroup, ProtocolVersion, SignatureScheme,
};
use crate::network::tls_kx::KeySharePublic;
use std::collections::HashSet;

/// A raw extension as it appears on the wire.
#[derive(Debug, Clone)]
pub struct RawExtension {
    /// Extension type code point.
    pub ext_type: u16,
    /// Extension data.
    pub data: Vec<u8>,
}

/// Parsed ClientHello / ServerHello extensions of interest.
#[derive(Debug, Clone, Default)]
pub struct ParsedExtensions {
    /// SNI host name (DNS).
    pub server_name: Option<String>,
    /// Supported versions.
    pub supported_versions: Vec<ProtocolVersion>,
    /// Supported groups.
    pub supported_groups: Vec<NamedGroup>,
    /// Key shares.
    pub key_shares: Vec<KeySharePublic>,
    /// Selected key share group (HRR / ServerHello).
    pub selected_group: Option<NamedGroup>,
    /// Signature algorithms.
    pub signature_algorithms: Vec<SignatureScheme>,
    /// ALPN protocols.
    pub alpn: Vec<Vec<u8>>,
    /// Cookie from HRR.
    pub cookie: Option<Vec<u8>>,
    /// Extended master secret offered (TLS 1.2).
    pub extended_master_secret: bool,
    /// Empty renegotiation_info present.
    pub renegotiation_info: bool,
    /// PSK key exchange modes.
    pub psk_modes: Vec<u8>,
    /// Selected ALPN protocol (EncryptedExtensions).
    pub selected_alpn: Option<Vec<u8>>,
    /// All raw extensions (for echo / diagnostics).
    pub raw: Vec<RawExtension>,
}

impl ParsedExtensions {
    /// Parses an extensions block (`vector<Extension>` with u16 length).
    pub fn parse(data: &[u8]) -> TlsResult<Self> {
        let mut r = Reader::new(data);
        let mut out = Self::default();
        let mut seen = HashSet::new();
        while !r.is_empty() {
            let ext_type = r.u16()?;
            let ext_data = r.vec_u16()?.to_vec();
            if !seen.insert(ext_type) {
                return Err(TlsError::Alert(AlertDescription::IllegalParameter));
            }
            out.raw.push(RawExtension {
                ext_type,
                data: ext_data.clone(),
            });
            match ExtensionType::from_u16(ext_type) {
                Some(ExtensionType::ServerName) => {
                    out.server_name = parse_server_name(&ext_data)?;
                }
                Some(ExtensionType::SupportedVersions) => {
                    out.supported_versions = parse_supported_versions(&ext_data)?;
                }
                Some(ExtensionType::SupportedGroups) => {
                    out.supported_groups = parse_named_groups(&ext_data)?;
                }
                Some(ExtensionType::KeyShare) => {
                    parse_key_share(&ext_data, &mut out)?;
                }
                Some(ExtensionType::SignatureAlgorithms) => {
                    out.signature_algorithms = parse_signature_schemes(&ext_data)?;
                }
                Some(ExtensionType::ApplicationLayerProtocolNegotiation) => {
                    out.alpn = parse_alpn(&ext_data)?;
                    if out.alpn.len() == 1 {
                        out.selected_alpn = out.alpn.first().cloned();
                    }
                }
                Some(ExtensionType::Cookie) => {
                    let mut cr = Reader::new(&ext_data);
                    out.cookie = Some(cr.vec_u16()?.to_vec());
                    cr.expect_empty("cookie")?;
                }
                Some(ExtensionType::ExtendedMasterSecret) => {
                    if !ext_data.is_empty() {
                        return Err(TlsError::Alert(AlertDescription::IllegalParameter));
                    }
                    out.extended_master_secret = true;
                }
                Some(ExtensionType::RenegotiationInfo) => {
                    let mut cr = Reader::new(&ext_data);
                    let _ = cr.vec_u8()?;
                    cr.expect_empty("renegotiation_info")?;
                    out.renegotiation_info = true;
                }
                Some(ExtensionType::PskKeyExchangeModes) => {
                    let mut cr = Reader::new(&ext_data);
                    out.psk_modes = cr.vec_u8()?.to_vec();
                    cr.expect_empty("psk_modes")?;
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

fn parse_server_name(data: &[u8]) -> TlsResult<Option<String>> {
    let mut r = Reader::new(data);
    let list = r.vec_u16()?;
    r.expect_empty("server_name")?;
    let mut lr = Reader::new(list);
    let mut name = None;
    while !lr.is_empty() {
        let name_type = lr.u8()?;
        let host = lr.vec_u16()?;
        if name_type == 0 {
            let s = std::str::from_utf8(host)
                .map_err(|_| TlsError::Alert(AlertDescription::IllegalParameter))?
                .to_string();
            if name.is_some() {
                return Err(TlsError::Alert(AlertDescription::IllegalParameter));
            }
            name = Some(s);
        }
    }
    Ok(name)
}

fn parse_supported_versions(data: &[u8]) -> TlsResult<Vec<ProtocolVersion>> {
    // ClientHello: versions<2..254>, ServerHello: single version (2 bytes).
    if data.len() == 2 {
        let v = u16::from_be_bytes([data[0], data[1]]);
        return Ok(vec![ProtocolVersion::from_u16(v)?]);
    }
    let mut r = Reader::new(data);
    let list = r.vec_u8()?;
    r.expect_empty("supported_versions")?;
    if list.len() < 2 || !list.len().is_multiple_of(2) {
        return Err(TlsError::Alert(AlertDescription::DecodeError));
    }
    let mut versions = Vec::new();
    let mut lr = Reader::new(list);
    while !lr.is_empty() {
        let v = lr.u16()?;
        if let Ok(pv) = ProtocolVersion::from_u16(v) {
            versions.push(pv);
        }
    }
    Ok(versions)
}

fn parse_named_groups(data: &[u8]) -> TlsResult<Vec<NamedGroup>> {
    let mut r = Reader::new(data);
    let list = r.vec_u16()?;
    r.expect_empty("supported_groups")?;
    let mut groups = Vec::new();
    let mut lr = Reader::new(list);
    while !lr.is_empty() {
        let g = lr.u16()?;
        if let Some(ng) = NamedGroup::from_u16(g) {
            groups.push(ng);
        }
    }
    Ok(groups)
}

fn parse_signature_schemes(data: &[u8]) -> TlsResult<Vec<SignatureScheme>> {
    let mut r = Reader::new(data);
    let list = r.vec_u16()?;
    r.expect_empty("signature_algorithms")?;
    let mut schemes = Vec::new();
    let mut lr = Reader::new(list);
    while !lr.is_empty() {
        let s = lr.u16()?;
        if let Some(ss) = SignatureScheme::from_u16(s) {
            schemes.push(ss);
        }
    }
    Ok(schemes)
}

fn parse_alpn(data: &[u8]) -> TlsResult<Vec<Vec<u8>>> {
    let mut r = Reader::new(data);
    let list = r.vec_u16()?;
    r.expect_empty("alpn")?;
    let mut protocols = Vec::new();
    let mut lr = Reader::new(list);
    while !lr.is_empty() {
        protocols.push(lr.vec_u8()?.to_vec());
    }
    if protocols.is_empty() {
        return Err(TlsError::Alert(AlertDescription::DecodeError));
    }
    Ok(protocols)
}

fn parse_key_share(data: &[u8], out: &mut ParsedExtensions) -> TlsResult<()> {
    // ClientHello: client_shares vector; ServerHello: single KeyShareEntry;
    // HRR: selected_group (2 bytes).
    if data.len() == 2 {
        let g = u16::from_be_bytes([data[0], data[1]]);
        out.selected_group = NamedGroup::from_u16(g);
        return Ok(());
    }
    let mut r = Reader::new(data);
    // Try ServerHello form first: group(2) + key_exchange<1..2^16-1>
    if data.len() >= 4 {
        let maybe_group = u16::from_be_bytes([data[0], data[1]]);
        let maybe_len = u16::from_be_bytes([data[2], data[3]]) as usize;
        if 4 + maybe_len == data.len() {
            if let Some(group) = NamedGroup::from_u16(maybe_group) {
                out.key_shares.push(KeySharePublic {
                    group,
                    key_exchange: data[4..].to_vec(),
                });
                out.selected_group = Some(group);
                return Ok(());
            }
        }
    }
    // ClientHello form
    let list = r.vec_u16()?;
    r.expect_empty("key_share")?;
    let mut lr = Reader::new(list);
    while !lr.is_empty() {
        let g = lr.u16()?;
        let kx = lr.vec_u16()?.to_vec();
        if let Some(group) = NamedGroup::from_u16(g) {
            out.key_shares.push(KeySharePublic {
                group,
                key_exchange: kx,
            });
        }
    }
    Ok(())
}

/// Encodes the SNI extension data for a DNS host name.
pub fn encode_server_name(hostname: &str) -> TlsResult<Vec<u8>> {
    let mut host_entry = Vec::new();
    host_entry.push(0); // host_name
    put_vec_u16(&mut host_entry, hostname.as_bytes())?;
    let mut out = Vec::new();
    put_vec_u16(&mut out, &host_entry)?;
    Ok(out)
}

/// Encodes ClientHello `supported_versions`.
pub fn encode_supported_versions_client(
    versions: &[ProtocolVersion],
) -> TlsResult<Vec<u8>> {
    let mut list = Vec::new();
    for v in versions {
        put_u16(&mut list, v.as_u16());
    }
    let mut out = Vec::new();
    put_vec_u8(&mut out, &list)?;
    Ok(out)
}

/// Encodes ServerHello `supported_versions`.
#[must_use]
pub fn encode_supported_versions_server(version: ProtocolVersion) -> Vec<u8> {
    version.to_be_bytes().to_vec()
}

/// Encodes `supported_groups`.
pub fn encode_supported_groups(groups: &[NamedGroup]) -> TlsResult<Vec<u8>> {
    let mut list = Vec::new();
    for g in groups {
        put_u16(&mut list, g.as_u16());
    }
    let mut out = Vec::new();
    put_vec_u16(&mut out, &list)?;
    Ok(out)
}

/// Encodes `signature_algorithms`.
pub fn encode_signature_algorithms(schemes: &[SignatureScheme]) -> TlsResult<Vec<u8>> {
    let mut list = Vec::new();
    for s in schemes {
        put_u16(&mut list, s.as_u16());
    }
    let mut out = Vec::new();
    put_vec_u16(&mut out, &list)?;
    Ok(out)
}

/// Encodes ClientHello key_share list.
pub fn encode_key_share_client(shares: &[KeySharePublic]) -> TlsResult<Vec<u8>> {
    let mut list = Vec::new();
    for s in shares {
        put_u16(&mut list, s.group.as_u16());
        put_vec_u16(&mut list, &s.key_exchange)?;
    }
    let mut out = Vec::new();
    put_vec_u16(&mut out, &list)?;
    Ok(out)
}

/// Encodes ServerHello key_share entry.
pub fn encode_key_share_server(share: &KeySharePublic) -> TlsResult<Vec<u8>> {
    let mut out = Vec::new();
    put_u16(&mut out, share.group.as_u16());
    put_vec_u16(&mut out, &share.key_exchange)?;
    Ok(out)
}

/// Encodes HRR selected group.
#[must_use]
pub fn encode_key_share_hrr(group: NamedGroup) -> Vec<u8> {
    group.as_u16().to_be_bytes().to_vec()
}

/// Encodes ALPN extension data.
pub fn encode_alpn(protocols: &[Vec<u8>]) -> TlsResult<Vec<u8>> {
    let mut list = Vec::new();
    for p in protocols {
        put_vec_u8(&mut list, p)?;
    }
    let mut out = Vec::new();
    put_vec_u16(&mut out, &list)?;
    Ok(out)
}

/// Encodes a single selected ALPN protocol for EncryptedExtensions.
pub fn encode_alpn_selected(protocol: &[u8]) -> TlsResult<Vec<u8>> {
    encode_alpn(&[protocol.to_vec()])
}

/// Appends one extension to `out`.
pub fn put_extension(
    out: &mut Vec<u8>,
    ext_type: ExtensionType,
    data: &[u8],
) -> TlsResult<()> {
    put_u16(out, ext_type.as_u16());
    put_vec_u16(out, data)?;
    Ok(())
}

/// Starts an extensions block and returns the length index; finish with [`finish_extensions`].
#[must_use]
pub fn start_extensions(out: &mut Vec<u8>) -> usize {
    start_u16_vec(out)
}

/// Finishes an extensions block.
pub fn finish_extensions(out: &mut Vec<u8>, idx: usize) -> TlsResult<()> {
    fill_u16_len(out, idx)
}

/// Encodes empty renegotiation_info (secure renegotiation sentinel for initial handshake).
pub fn encode_renegotiation_info_empty() -> TlsResult<Vec<u8>> {
    let mut out = Vec::new();
    put_vec_u8(&mut out, &[])?;
    Ok(out)
}

/// Encodes `psk_key_exchange_modes` with `psk_dhe_ke`.
pub fn encode_psk_modes_dhe() -> TlsResult<Vec<u8>> {
    let mut out = Vec::new();
    put_vec_u8(&mut out, &[1])?; // psk_dhe_ke
    Ok(out)
}

/// Encodes EC point formats: uncompressed only.
pub fn encode_ec_point_formats() -> TlsResult<Vec<u8>> {
    let mut out = Vec::new();
    let idx = start_u8_vec(&mut out);
    out.push(0); // uncompressed
    fill_u8_len(&mut out, idx)?;
    Ok(out)
}
