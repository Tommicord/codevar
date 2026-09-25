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

use crate::tls_alert::AlertDescription;
use crate::tls_codec::{
    Reader, fill_u8_len, fill_u16_len, put_u16, put_vec_u8, put_vec_u16, start_u8_vec, start_u16_vec,
};
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::{ExtensionType, NamedGroup, ProtocolVersion, SignatureScheme};
use crate::tls_kx::KeySharePublic;
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
            let ext_data = r
                .vec_u16()?
                .to_vec();
            if !seen.insert(ext_type) {
                return Err(TlsError::Alert(AlertDescription::IllegalParameter));
            }
            out.raw
                .push(RawExtension {
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
                    if out
                        .alpn
                        .len()
                        == 1
                    {
                        out.selected_alpn = out
                            .alpn
                            .first()
                            .cloned();
                    }
                }
                Some(ExtensionType::Cookie) => {
                    let mut cr = Reader::new(&ext_data);
                    out.cookie = Some(
                        cr.vec_u16()?
                            .to_vec(),
                    );
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
                    out.psk_modes = cr
                        .vec_u8()?
                        .to_vec();
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
    if list.len() < 2
        || !list
            .len()
            .is_multiple_of(2)
    {
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
        protocols.push(
            lr.vec_u8()?
                .to_vec(),
        );
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
        if 4 + maybe_len == data.len()
            && let Some(group) = NamedGroup::from_u16(maybe_group)
        {
            out.key_shares
                .push(KeySharePublic {
                    group,
                    key_exchange: data[4..].to_vec(),
                });
            out.selected_group = Some(group);
            return Ok(());
        }
    }
    // ClientHello form
    let list = r.vec_u16()?;
    r.expect_empty("key_share")?;
    let mut lr = Reader::new(list);
    while !lr.is_empty() {
        let g = lr.u16()?;
        let kx = lr
            .vec_u16()?
            .to_vec();
        if let Some(group) = NamedGroup::from_u16(g) {
            out.key_shares
                .push(KeySharePublic {
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
pub fn encode_supported_versions_client(versions: &[ProtocolVersion]) -> TlsResult<Vec<u8>> {
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
    version
        .to_be_bytes()
        .to_vec()
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
        put_u16(
            &mut list,
            s.group
                .as_u16(),
        );
        put_vec_u16(&mut list, &s.key_exchange)?;
    }
    let mut out = Vec::new();
    put_vec_u16(&mut out, &list)?;
    Ok(out)
}

/// Encodes ServerHello key_share entry.
pub fn encode_key_share_server(share: &KeySharePublic) -> TlsResult<Vec<u8>> {
    let mut out = Vec::new();
    put_u16(
        &mut out,
        share
            .group
            .as_u16(),
    );
    put_vec_u16(&mut out, &share.key_exchange)?;
    Ok(out)
}

/// Encodes HRR selected group.
#[must_use]
pub fn encode_key_share_hrr(group: NamedGroup) -> Vec<u8> {
    group
        .as_u16()
        .to_be_bytes()
        .to_vec()
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
pub fn put_extension(out: &mut Vec<u8>, ext_type: ExtensionType, data: &[u8]) -> TlsResult<()> {
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
#[allow(clippy::ptr_arg)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_kx::KeySharePublic;

    fn entries(items: &[(u16, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        for &(ext_type, data) in items {
            put_u16(&mut buf, ext_type);
            put_vec_u16(&mut buf, data).unwrap();
        }
        buf
    }

    #[test]
    fn parse_empty_list_yields_default() {
        let parsed = ParsedExtensions::parse(&[]).unwrap();
        assert!(
            parsed
                .raw
                .is_empty()
        );
        assert!(
            parsed
                .server_name
                .is_none()
        );
        assert!(
            parsed
                .supported_versions
                .is_empty()
        );
        assert!(
            parsed
                .supported_groups
                .is_empty()
        );
        assert!(
            parsed
                .key_shares
                .is_empty()
        );
        assert!(
            parsed
                .selected_group
                .is_none()
        );
        assert!(
            parsed
                .signature_algorithms
                .is_empty()
        );
        assert!(
            parsed
                .alpn
                .is_empty()
        );
        assert!(
            parsed
                .cookie
                .is_none()
        );
        assert!(!parsed.extended_master_secret);
        assert!(!parsed.renegotiation_info);
        assert!(
            parsed
                .psk_modes
                .is_empty()
        );
        assert!(
            parsed
                .selected_alpn
                .is_none()
        );
    }

    #[test]
    fn parse_single_extension() {
        let data = encode_server_name("example.com").unwrap();
        let wire = entries(&[(ExtensionType::ServerName.as_u16(), &data)]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .server_name
                .as_deref(),
            Some("example.com")
        );
        assert_eq!(
            parsed
                .raw
                .len(),
            1
        );
        assert_eq!(parsed.raw[0].ext_type, ExtensionType::ServerName.as_u16());
        assert_eq!(parsed.raw[0].data, data);
    }

    #[test]
    fn parse_multiple_mixed_known_and_unknown_extensions() {
        let sni = encode_server_name("a.test").unwrap();
        let groups = encode_supported_groups(&[NamedGroup::X25519, NamedGroup::Secp256r1]).unwrap();
        let versions =
            encode_supported_versions_client(&[ProtocolVersion::Tls12, ProtocolVersion::Tls13]).unwrap();
        let wire = entries(&[
            (ExtensionType::ServerName.as_u16(), &sni),
            (0x9999, &[1, 2, 3]),
            (ExtensionType::SupportedGroups.as_u16(), &groups),
            (ExtensionType::SupportedVersions.as_u16(), &versions),
        ]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .server_name
                .as_deref(),
            Some("a.test")
        );
        assert_eq!(
            parsed.supported_groups,
            vec![NamedGroup::X25519, NamedGroup::Secp256r1]
        );
        assert_eq!(
            parsed.supported_versions,
            vec![ProtocolVersion::Tls12, ProtocolVersion::Tls13]
        );
        assert_eq!(
            parsed
                .raw
                .len(),
            4
        );
        assert_eq!(parsed.raw[1].ext_type, 0x9999);
        assert_eq!(parsed.raw[1].data, vec![1, 2, 3]);
    }

    #[test]
    fn declared_extension_length_exceeding_buffer_rejected() {
        let mut wire = Vec::new();
        put_u16(&mut wire, 0x1234);
        put_u16(&mut wire, 0xffff);
        wire.extend_from_slice(&[0xaa, 0xbb]);
        assert_eq!(wire.len(), 6);
        let err = ParsedExtensions::parse(&wire).unwrap_err();
        assert!(matches!(err, TlsError::Decode(_)), "err={err:?}");
    }

    #[test]
    fn oversized_declared_extension_rejected_without_large_allocation() {
        let mut wire = Vec::new();
        put_u16(&mut wire, 0x9999);
        put_u16(&mut wire, 0xffff);
        assert_eq!(wire.len(), 4);
        assert!(ParsedExtensions::parse(&wire).is_err());
    }

    #[test]
    fn extension_list_with_trailing_partial_header_rejected() {
        let sni = encode_server_name("x.test").unwrap();
        let mut wire = entries(&[(ExtensionType::ServerName.as_u16(), &sni)]);
        wire.extend_from_slice(&[0x00, 0x05, 0x01]);
        assert!(ParsedExtensions::parse(&wire).is_err());

        let mut odd = entries(&[(ExtensionType::ServerName.as_u16(), &sni)]);
        odd.push(0x42);
        assert!(ParsedExtensions::parse(&odd).is_err());
    }

    #[test]
    fn duplicate_extensions_rejected_even_when_unknown() {
        let sni = encode_server_name("dup.test").unwrap();
        let wire = entries(&[
            (ExtensionType::ServerName.as_u16(), &sni),
            (ExtensionType::ServerName.as_u16(), &sni),
        ]);
        let err = ParsedExtensions::parse(&wire).unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::IllegalParameter)),
            "err={err:?}"
        );

        let dup_unknown = entries(&[(0x9999, &[1]), (0x9999, &[2])]);
        let err2 = ParsedExtensions::parse(&dup_unknown).unwrap_err();
        assert!(
            matches!(err2, TlsError::Alert(AlertDescription::IllegalParameter)),
            "err={err2:?}"
        );
    }

    #[test]
    fn zero_length_extension_data_is_tolerated() {
        let wire = entries(&[(ExtensionType::SessionTicket.as_u16(), &[]), (0x9999, &[])]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .raw
                .len(),
            2
        );
        assert!(
            parsed.raw[0]
                .data
                .is_empty()
        );
        assert!(
            parsed.raw[1]
                .data
                .is_empty()
        );
        assert!(
            parsed
                .server_name
                .is_none()
        );
    }

    #[test]
    fn unknown_extension_type_is_tolerated_and_kept_raw() {
        let wire = entries(&[(0xabcd, &[9, 8, 7, 6])]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .raw
                .len(),
            1
        );
        assert_eq!(parsed.raw[0].ext_type, 0xabcd);
        assert!(
            parsed
                .server_name
                .is_none()
        );
        assert!(
            parsed
                .supported_versions
                .is_empty()
        );
        assert!(
            parsed
                .selected_alpn
                .is_none()
        );
    }

    #[test]
    fn extension_data_truncated_mid_field_rejected() {
        // supported_groups declares a 4-byte list but only 2 bytes follow.
        let truncated_groups = [0x00, 0x04, 0x00, 0x17];
        let wire = entries(&[(ExtensionType::SupportedGroups.as_u16(), &truncated_groups)]);
        assert!(ParsedExtensions::parse(&wire).is_err());

        // Cookie declares more bytes than the extension carries.
        let truncated_cookie = [0x00, 0x10, 0xaa];
        let wire2 = entries(&[(ExtensionType::Cookie.as_u16(), &truncated_cookie)]);
        assert!(ParsedExtensions::parse(&wire2).is_err());
    }

    #[test]
    fn server_name_encode_parse_round_trip_and_malformed_forms() {
        let data = encode_server_name("localhost").unwrap();
        let wire = entries(&[(ExtensionType::ServerName.as_u16(), &data)]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .server_name
                .as_deref(),
            Some("localhost")
        );

        // Two host_name entries in one list are rejected.
        let mut list = Vec::new();
        list.push(0);
        put_vec_u16(&mut list, b"one").unwrap();
        list.push(0);
        put_vec_u16(&mut list, b"two").unwrap();
        let mut outer = Vec::new();
        put_vec_u16(&mut outer, &list).unwrap();
        let dup_wire = entries(&[(ExtensionType::ServerName.as_u16(), &outer)]);
        let err = ParsedExtensions::parse(&dup_wire).unwrap_err();
        assert!(
            matches!(err, TlsError::Alert(AlertDescription::IllegalParameter)),
            "err={err:?}"
        );

        // Non-UTF-8 host bytes are rejected.
        let mut bad_list = Vec::new();
        bad_list.push(0);
        put_vec_u16(&mut bad_list, &[0xff, 0xfe]).unwrap();
        let mut bad_outer = Vec::new();
        put_vec_u16(&mut bad_outer, &bad_list).unwrap();
        let bad_wire = entries(&[(ExtensionType::ServerName.as_u16(), &bad_outer)]);
        assert!(matches!(
            ParsedExtensions::parse(&bad_wire),
            Err(TlsError::Alert(AlertDescription::IllegalParameter))
        ));
    }

    #[test]
    fn supported_versions_client_and_server_forms() {
        let client =
            encode_supported_versions_client(&[ProtocolVersion::Tls12, ProtocolVersion::Tls13]).unwrap();
        assert_eq!(client, vec![4, 3, 3, 3, 4]);
        let wire = entries(&[(ExtensionType::SupportedVersions.as_u16(), &client)]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed.supported_versions,
            vec![ProtocolVersion::Tls12, ProtocolVersion::Tls13]
        );

        let server = encode_supported_versions_server(ProtocolVersion::Tls13);
        assert_eq!(server, vec![3, 4]);
        let sh_wire = entries(&[(ExtensionType::SupportedVersions.as_u16(), &server)]);
        let sh = ParsedExtensions::parse(&sh_wire).unwrap();
        assert_eq!(sh.supported_versions, vec![ProtocolVersion::Tls13]);

        // ServerHello form with an unknown version is a hard error.
        let unknown = [0x03, 0x05];
        let unk_wire = entries(&[(ExtensionType::SupportedVersions.as_u16(), &unknown)]);
        assert!(ParsedExtensions::parse(&unk_wire).is_err());

        // ClientHello form with an odd-length list is a decode error.
        let odd = [3, 0x03, 0x03, 0x03];
        let odd_wire = entries(&[(ExtensionType::SupportedVersions.as_u16(), &odd)]);
        assert!(matches!(
            ParsedExtensions::parse(&odd_wire),
            Err(TlsError::Alert(AlertDescription::DecodeError))
        ));

        // Unknown versions inside a ClientHello list are skipped.
        let mixed = [4, 0x99, 0x99, 0x03, 0x04];
        let mixed_wire = entries(&[(ExtensionType::SupportedVersions.as_u16(), &mixed)]);
        let mixed_parsed = ParsedExtensions::parse(&mixed_wire).unwrap();
        assert_eq!(mixed_parsed.supported_versions, vec![ProtocolVersion::Tls13]);
    }

    #[test]
    fn supported_groups_and_signature_algorithms_round_trip() {
        let groups =
            encode_supported_groups(&[NamedGroup::X25519, NamedGroup::Secp256r1, NamedGroup::Secp384r1])
                .unwrap();
        let wire = entries(&[(ExtensionType::SupportedGroups.as_u16(), &groups)]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed.supported_groups,
            vec![NamedGroup::X25519, NamedGroup::Secp256r1, NamedGroup::Secp384r1]
        );

        // Unknown group codes inside the list are skipped.
        let mut list = Vec::new();
        put_u16(&mut list, 0x9999);
        put_u16(&mut list, NamedGroup::X25519.as_u16());
        let mut with_unknown = Vec::new();
        put_vec_u16(&mut with_unknown, &list).unwrap();
        let unk_wire = entries(&[(ExtensionType::SupportedGroups.as_u16(), &with_unknown)]);
        let unk_parsed = ParsedExtensions::parse(&unk_wire).unwrap();
        assert_eq!(unk_parsed.supported_groups, vec![NamedGroup::X25519]);

        let schemes =
            encode_signature_algorithms(&[SignatureScheme::EcdsaSecp256r1Sha256, SignatureScheme::Ed25519])
                .unwrap();
        let sig_wire = entries(&[(ExtensionType::SignatureAlgorithms.as_u16(), &schemes)]);
        let sig_parsed = ParsedExtensions::parse(&sig_wire).unwrap();
        assert_eq!(
            sig_parsed.signature_algorithms,
            vec![SignatureScheme::EcdsaSecp256r1Sha256, SignatureScheme::Ed25519]
        );
    }

    #[test]
    fn alpn_round_trip_selection_and_errors() {
        let multi = encode_alpn(&[b"h2".to_vec(), b"http/1.1".to_vec()]).unwrap();
        let wire = entries(&[(
            ExtensionType::ApplicationLayerProtocolNegotiation.as_u16(),
            &multi,
        )]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .alpn
                .len(),
            2
        );
        assert_eq!(parsed.alpn[0].as_slice(), b"h2");
        assert_eq!(parsed.alpn[1].as_slice(), b"http/1.1");
        assert!(
            parsed
                .selected_alpn
                .is_none()
        );

        let single = encode_alpn_selected(b"h2").unwrap();
        let single_wire = entries(&[(
            ExtensionType::ApplicationLayerProtocolNegotiation.as_u16(),
            &single,
        )]);
        let single_parsed = ParsedExtensions::parse(&single_wire).unwrap();
        assert_eq!(
            single_parsed
                .alpn
                .len(),
            1
        );
        assert_eq!(
            single_parsed
                .selected_alpn
                .as_deref(),
            Some(&b"h2"[..])
        );

        // Empty protocol list is a decode error.
        let empty = encode_alpn(&[]).unwrap();
        let empty_wire = entries(&[(
            ExtensionType::ApplicationLayerProtocolNegotiation.as_u16(),
            &empty,
        )]);
        assert!(matches!(
            ParsedExtensions::parse(&empty_wire),
            Err(TlsError::Alert(AlertDescription::DecodeError))
        ));

        // A protocol longer than 255 bytes cannot be encoded.
        let oversized_proto = vec![b'x'; 256];
        assert!(encode_alpn(&[oversized_proto]).is_err());
    }

    #[test]
    fn key_share_client_server_and_hrr_forms() {
        let shares = vec![
            KeySharePublic {
                group: NamedGroup::X25519,
                key_exchange: vec![0x42; 32],
            },
            KeySharePublic {
                group: NamedGroup::Secp256r1,
                key_exchange: vec![0x24; 65],
            },
        ];
        let client = encode_key_share_client(&shares).unwrap();
        let wire = entries(&[(ExtensionType::KeyShare.as_u16(), &client)]);
        let parsed = ParsedExtensions::parse(&wire).unwrap();
        assert_eq!(
            parsed
                .key_shares
                .len(),
            2
        );
        assert_eq!(parsed.key_shares[0].group, NamedGroup::X25519);
        assert_eq!(parsed.key_shares[0].key_exchange, vec![0x42; 32]);
        assert_eq!(parsed.key_shares[1].group, NamedGroup::Secp256r1);
        assert_eq!(parsed.key_shares[1].key_exchange, vec![0x24; 65]);

        let server_share = KeySharePublic {
            group: NamedGroup::X25519,
            key_exchange: vec![0x43; 32],
        };
        let server = encode_key_share_server(&server_share).unwrap();
        let sh_wire = entries(&[(ExtensionType::KeyShare.as_u16(), &server)]);
        let sh_parsed = ParsedExtensions::parse(&sh_wire).unwrap();
        assert_eq!(
            sh_parsed
                .key_shares
                .len(),
            1
        );
        assert_eq!(sh_parsed.key_shares[0].key_exchange, vec![0x43; 32]);
        assert_eq!(sh_parsed.selected_group, Some(NamedGroup::X25519));

        let hrr = encode_key_share_hrr(NamedGroup::Secp384r1);
        assert_eq!(hrr, vec![0x00, 0x18]);
        let hrr_wire = entries(&[(ExtensionType::KeyShare.as_u16(), &hrr)]);
        let hrr_parsed = ParsedExtensions::parse(&hrr_wire).unwrap();
        assert_eq!(hrr_parsed.selected_group, Some(NamedGroup::Secp384r1));
        assert!(
            hrr_parsed
                .key_shares
                .is_empty()
        );

        let unknown_hrr = [0x99, 0x99];
        let unk_wire = entries(&[(ExtensionType::KeyShare.as_u16(), &unknown_hrr)]);
        let unk_parsed = ParsedExtensions::parse(&unk_wire).unwrap();
        assert!(
            unk_parsed
                .selected_group
                .is_none()
        );
    }

    #[test]
    fn cookie_psk_modes_renegotiation_and_master_secret() {
        let mut cookie_data = Vec::new();
        put_vec_u16(&mut cookie_data, &[7, 7, 7]).unwrap();
        let cookie_wire = entries(&[(ExtensionType::Cookie.as_u16(), &cookie_data)]);
        let cookie_parsed = ParsedExtensions::parse(&cookie_wire).unwrap();
        assert_eq!(cookie_parsed.cookie, Some(vec![7, 7, 7]));

        let psk = encode_psk_modes_dhe().unwrap();
        assert_eq!(psk, vec![1, 1]);
        let psk_wire = entries(&[(ExtensionType::PskKeyExchangeModes.as_u16(), &psk)]);
        let psk_parsed = ParsedExtensions::parse(&psk_wire).unwrap();
        assert_eq!(psk_parsed.psk_modes, vec![1]);

        // Trailing bytes after the psk_modes vector are rejected.
        let mut psk_trailing = psk.clone();
        psk_trailing.push(0xff);
        let psk_bad_wire = entries(&[(ExtensionType::PskKeyExchangeModes.as_u16(), &psk_trailing)]);
        assert!(ParsedExtensions::parse(&psk_bad_wire).is_err());

        let reneg = encode_renegotiation_info_empty().unwrap();
        assert_eq!(reneg, vec![0]);
        let reneg_wire = entries(&[(ExtensionType::RenegotiationInfo.as_u16(), &reneg)]);
        let reneg_parsed = ParsedExtensions::parse(&reneg_wire).unwrap();
        assert!(reneg_parsed.renegotiation_info);

        let ems_wire = entries(&[(ExtensionType::ExtendedMasterSecret.as_u16(), &[])]);
        let ems_parsed = ParsedExtensions::parse(&ems_wire).unwrap();
        assert!(ems_parsed.extended_master_secret);

        let ems_bad = [0x01];
        let ems_bad_wire = entries(&[(ExtensionType::ExtendedMasterSecret.as_u16(), &ems_bad)]);
        assert!(matches!(
            ParsedExtensions::parse(&ems_bad_wire),
            Err(TlsError::Alert(AlertDescription::IllegalParameter))
        ));
    }

    #[test]
    fn extension_block_builders_round_trip_through_parse() {
        let mut block = Vec::new();
        let idx = start_extensions(&mut block);
        put_extension(
            &mut block,
            ExtensionType::ServerName,
            &encode_server_name("build.test").unwrap(),
        )
        .unwrap();
        put_extension(&mut block, ExtensionType::ExtendedMasterSecret, &[]).unwrap();
        put_extension(
            &mut block,
            ExtensionType::RenegotiationInfo,
            &encode_renegotiation_info_empty().unwrap(),
        )
        .unwrap();
        finish_extensions(&mut block, idx).unwrap();

        let total = u16::from_be_bytes([block[0], block[1]]);
        assert_eq!(usize::from(total), block.len() - 2);

        let parsed = ParsedExtensions::parse(&block[2..]).unwrap();
        assert_eq!(
            parsed
                .server_name
                .as_deref(),
            Some("build.test")
        );
        assert!(parsed.extended_master_secret);
        assert!(parsed.renegotiation_info);
        assert_eq!(
            parsed
                .raw
                .len(),
            3
        );
    }

    #[test]
    fn extension_block_total_length_overflow_rejected() {
        let mut block = Vec::new();
        let idx = start_extensions(&mut block);
        block.resize(block.len() + 65536, 0);
        let err = finish_extensions(&mut block, idx).unwrap_err();
        assert!(matches!(err, TlsError::Internal(_)), "err={err:?}");
        assert_eq!(&block[..2], &[0, 0]);
    }

    #[test]
    fn put_extension_rejects_data_above_u16_but_leaves_partial_writes_documented() {
        let mut out = Vec::new();
        let oversized = vec![0u8; 65536];
        assert!(put_extension(&mut out, ExtensionType::Cookie, &oversized).is_err());
        // The type word is appended before the length check fails.
        assert_eq!(out.len(), 2);

        let mut exact = Vec::new();
        let max_data = vec![0u8; 65535];
        put_extension(&mut exact, ExtensionType::Cookie, &max_data).unwrap();
        assert_eq!(exact.len(), 2 + 2 + 65535);
    }

    #[test]
    fn ec_point_formats_encoding() {
        let formats = encode_ec_point_formats().unwrap();
        assert_eq!(formats, vec![1, 0]);
        assert!(encode_ec_point_formats().is_ok());
    }

    #[test]
    fn encoder_reuses_large_buffer_across_extension_blocks() {
        let mut out = Vec::with_capacity(1024 * 1024);
        let payload = vec![0x5Au8; 32 * 1024];
        for _ in 0..16 {
            out.clear();
            let idx = start_extensions(&mut out);
            put_extension(&mut out, ExtensionType::Cookie, &payload).unwrap();
            finish_extensions(&mut out, idx).unwrap();
            let total = u16::from_be_bytes([out[0], out[1]]);
            assert_eq!(usize::from(total), out.len() - 2);
        }
        assert!(out.capacity() >= 1024 * 1024);
    }
}
