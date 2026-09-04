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

//! WebSocket framing unit tests (RFC 6455 §5).

use codevar_colab::network::ws_frame::{
    apply_mask, decode_close_payload, encode_close_payload, random_mask_key,
    try_parse_frame,
};
use codevar_colab::network::{Role, WsCloseCode, WsFrame, WsFrameHeader, WsOpcode};

#[test]
fn rfc6455_unmasked_hello() {
    let wire = [0x81, 0x05, 0x48, 0x65, 0x6c, 0x6c, 0x6f];
    let (frame, n) = try_parse_frame(&wire, Role::Client, 1024)
        .expect("parse")
        .expect("complete");
    assert_eq!(n, 7);
    assert!(frame.header.fin);
    assert_eq!(frame.header.opcode, WsOpcode::Text);
    assert!(!frame.header.masked);
    assert_eq!(frame.payload, b"Hello");
}

#[test]
fn rfc6455_masked_hello() {
    let wire = [
        0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58,
    ];
    let (frame, _) = try_parse_frame(&wire, Role::Server, 1024)
        .expect("parse")
        .expect("complete");
    assert_eq!(frame.payload, b"Hello");
    assert!(frame.header.masked);
}

#[test]
fn rfc6455_fragmented_unmasked_text() {
    let part1 = [0x01, 0x03, 0x48, 0x65, 0x6c];
    let part2 = [0x80, 0x02, 0x6c, 0x6f];
    let (f1, _) = try_parse_frame(&part1, Role::Client, 1024)
        .unwrap()
        .unwrap();
    let (f2, _) = try_parse_frame(&part2, Role::Client, 1024)
        .unwrap()
        .unwrap();
    assert!(!f1.header.fin);
    assert_eq!(f1.header.opcode, WsOpcode::Text);
    assert_eq!(f1.payload, b"Hel");
    assert!(f2.header.fin);
    assert_eq!(f2.header.opcode, WsOpcode::Continuation);
    assert_eq!(f2.payload, b"lo");
}

#[test]
fn rfc6455_256_byte_binary_length_encoding() {
    let payload = vec![0xabu8; 256];
    let frame = WsFrame::binary(payload.clone());
    let mut out = Vec::new();
    frame.encode(&mut out, None).expect("encode");
    assert_eq!(&out[..4], &[0x82, 0x7e, 0x01, 0x00]);
    let (parsed, n) = try_parse_frame(&out, Role::Client, 1024).unwrap().unwrap();
    assert_eq!(n, out.len());
    assert_eq!(parsed.payload, payload);
}

#[test]
fn rfc6455_64kib_binary_length_encoding() {
    let payload = vec![0x5au8; 65536];
    let frame = WsFrame::binary(payload.clone());
    let mut out = Vec::new();
    frame.encode(&mut out, None).expect("encode");
    assert_eq!(&out[..2], &[0x82, 0x7f]);
    assert_eq!(&out[2..10], &[0, 0, 0, 0, 0, 1, 0, 0]);
    let (parsed, _) = try_parse_frame(&out, Role::Client, 100_000)
        .unwrap()
        .unwrap();
    assert_eq!(parsed.payload, payload);
}

#[test]
fn encode_decode_roundtrip_masked_and_unmasked() {
    for &mask in &[false, true] {
        let frame = WsFrame::text(b"roundtrip".to_vec());
        let mut out = Vec::new();
        let key = if mask { Some([1, 2, 3, 4]) } else { None };
        frame.encode(&mut out, key).unwrap();
        let role = if mask { Role::Server } else { Role::Client };
        let (parsed, _) = try_parse_frame(&out, role, 1024).unwrap().unwrap();
        assert_eq!(parsed.payload, b"roundtrip");
        assert_eq!(parsed.header.opcode, WsOpcode::Text);
    }
}

#[test]
fn apply_mask_is_involutive() {
    let key = [0x11, 0x22, 0x33, 0x44];
    let mut data = b"abcdef".to_vec();
    let original = data.clone();
    apply_mask(&mut data, key);
    assert_ne!(data, original);
    apply_mask(&mut data, key);
    assert_eq!(data, original);
}

#[test]
fn empty_payload_frames() {
    for opcode in [
        WsOpcode::Text,
        WsOpcode::Binary,
        WsOpcode::Ping,
        WsOpcode::Pong,
    ] {
        let frame = WsFrame::new(true, opcode, Vec::new());
        let mut out = Vec::new();
        frame.encode(&mut out, None).unwrap();
        let (parsed, _) = try_parse_frame(&out, Role::Client, 1024).unwrap().unwrap();
        assert!(parsed.payload.is_empty());
        assert_eq!(parsed.header.opcode, opcode);
    }
}

#[test]
fn control_frame_must_not_be_fragmented() {
    // FIN=0, opcode=Ping
    let wire = [0x09, 0x00];
    let err = try_parse_frame(&wire, Role::Client, 1024).unwrap_err();
    assert!(err.to_string().contains("fragment") || err.close_code().is_some());
}

#[test]
fn control_payload_over_125_rejected() {
    let mut wire = vec![0x89, 126, 0x00, 126];
    wire.extend(std::iter::repeat_n(0u8, 126));
    // Actually length 126 with control is invalid even before reading payload.
    // Simpler: header claiming len=126 for ping.
    let wire = [0x89, 126, 0x00, 0x7e];
    let err = WsFrameHeader::parse(&wire, 1024).unwrap_err();
    assert!(err.to_string().contains("125") || err.close_code().is_some());
}

#[test]
fn non_minimal_16bit_length_rejected() {
    // len7=126 but extended length is 10 (<=125)
    let wire = [0x82, 126, 0x00, 0x0a];
    let err = WsFrameHeader::parse(&wire, 1024).unwrap_err();
    assert!(err.to_string().contains("non-minimal"));
}

#[test]
fn non_minimal_64bit_length_rejected() {
    let wire = [
        0x82, 127, 0, 0, 0, 0, 0, 0, 0x01,
        0x00, // length 256 — should use 16-bit form
    ];
    let err = WsFrameHeader::parse(&wire, 1024).unwrap_err();
    assert!(err.to_string().contains("non-minimal"));
}

#[test]
fn msb_set_on_64bit_length_rejected() {
    let wire = [0x82, 127, 0x80, 0, 0, 0, 0, 0, 0, 1];
    let err = WsFrameHeader::parse(&wire, 1024).unwrap_err();
    assert!(err.to_string().contains("MSB") || err.close_code().is_some());
}

#[test]
fn rsv_bits_rejected() {
    // RSV1 set on otherwise valid Hello
    let wire = [0xc1, 0x05, 0x48, 0x65, 0x6c, 0x6c, 0x6f];
    let err = try_parse_frame(&wire, Role::Client, 1024).unwrap_err();
    assert!(err.to_string().contains("RSV") || err.close_code().is_some());
}

#[test]
fn reserved_opcodes_rejected() {
    for op in [0x3u8, 0x7, 0xB, 0xF] {
        let wire = [0x80 | op, 0x00];
        assert!(
            try_parse_frame(&wire, Role::Client, 1024).is_err(),
            "opcode 0x{op:x} must be rejected"
        );
    }
}

#[test]
fn server_rejects_unmasked_client_accepts_unmasked() {
    let wire = [0x81, 0x05, 0x48, 0x65, 0x6c, 0x6c, 0x6f];
    assert!(try_parse_frame(&wire, Role::Server, 1024).is_err());
    assert!(try_parse_frame(&wire, Role::Client, 1024).is_ok());
}

#[test]
fn client_rejects_masked_server_accepts_masked() {
    let wire = [
        0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58,
    ];
    assert!(try_parse_frame(&wire, Role::Client, 1024).is_err());
    assert!(try_parse_frame(&wire, Role::Server, 1024).is_ok());
}

#[test]
fn incomplete_header_returns_none() {
    assert!(WsFrameHeader::parse(&[0x81], 1024).unwrap().is_none());
    assert!(
        WsFrameHeader::parse(&[0x82, 126, 0x01], 1024)
            .unwrap()
            .is_none()
    );
    assert!(
        try_parse_frame(&[0x81, 0x05, 0x48, 0x65], Role::Client, 1024)
            .unwrap()
            .is_none()
    );
}

#[test]
fn max_frame_size_enforced() {
    let payload = vec![0u8; 200];
    let frame = WsFrame::binary(payload);
    let mut out = Vec::new();
    frame.encode(&mut out, None).unwrap();
    let err = try_parse_frame(&out, Role::Client, 100).unwrap_err();
    match err {
        codevar_colab::network::WsError::MessageTooBig { size, limit } => {
            assert_eq!(size, 200);
            assert_eq!(limit, 100);
        }
        other => panic!("expected MessageTooBig, got {other:?}"),
    }
}

#[test]
fn close_payload_empty_single_byte_and_with_reason() {
    let (code, reason) = decode_close_payload(&[]).unwrap();
    assert_eq!(code, WsCloseCode::NoStatusReceived);
    assert!(reason.is_empty());

    assert!(decode_close_payload(&[0x03]).is_err());

    let payload = encode_close_payload(Some(WsCloseCode::GoingAway), "leaving").unwrap();
    let (code, reason) = decode_close_payload(&payload).unwrap();
    assert_eq!(code, WsCloseCode::GoingAway);
    assert_eq!(reason, "leaving");
}

#[test]
fn close_payload_rejects_forbidden_codes_and_non_utf8_reason() {
    assert!(encode_close_payload(Some(WsCloseCode::NoStatusReceived), "").is_err());
    assert!(encode_close_payload(Some(WsCloseCode::Abnormal), "").is_err());
    assert!(encode_close_payload(None, "needs code").is_err());

    // code 1000 + invalid UTF-8 reason
    let bad = [0x03, 0xe8, 0xff, 0xfe];
    assert!(decode_close_payload(&bad).is_err());
}

#[test]
fn ping_pong_payload_limit() {
    let ok = vec![0u8; 125];
    assert!(WsFrame::ping(ok.clone()).is_ok());
    assert!(WsFrame::pong(ok).is_ok());
    let too_big = vec![0u8; 126];
    assert!(WsFrame::ping(too_big.clone()).is_err());
    assert!(WsFrame::pong(too_big).is_err());
}

#[test]
fn random_mask_key_is_four_bytes() {
    let a = random_mask_key().unwrap();
    let b = random_mask_key().unwrap();
    assert_eq!(a.len(), 4);
    // Extremely unlikely to collide if CSPRNG works.
    assert_ne!(a, b);
}

#[test]
fn role_masking_helpers() {
    assert!(Role::Client.must_mask_outbound());
    assert!(!Role::Server.must_mask_outbound());
    assert!(Role::Server.expects_inbound_masked());
    assert!(!Role::Client.expects_inbound_masked());
}
