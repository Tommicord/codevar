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

//! WebSocket Base64 and UTF-8 validator unit tests.

use codevar_colab::network::ws_base64::{decode, encode};
use codevar_colab::network::ws_utf8::{Utf8Validator, validate_utf8};

#[test]
fn base64_rfc6455_sample_nonce() {
    assert_eq!(encode(b"the sample nonce"), "dGhlIHNhbXBsZSBub25jZQ==");
    assert_eq!(
        decode("dGhlIHNhbXBsZSBub25jZQ==").unwrap(),
        b"the sample nonce"
    );
}

#[test]
fn base64_padding_variants() {
    assert_eq!(encode(b""), "");
    assert_eq!(encode(b"f"), "Zg==");
    assert_eq!(encode(b"fo"), "Zm8=");
    assert_eq!(encode(b"foo"), "Zm9v");
    assert_eq!(encode(b"foob"), "Zm9vYg==");
    assert_eq!(encode(b"fooba"), "Zm9vYmE=");
    assert_eq!(encode(b"foobar"), "Zm9vYmFy");

    assert_eq!(decode("Zg==").unwrap(), b"f");
    assert_eq!(decode("Zm8=").unwrap(), b"fo");
    assert_eq!(decode("Zm9v").unwrap(), b"foo");
}

#[test]
fn base64_rejects_invalid_input() {
    assert!(decode("abc").is_err()); // not multiple of 4
    assert!(decode("@@@@").is_err());
    assert!(decode("Zm8===").is_err());
    assert!(decode("Zg=A").is_err()); // bad padding layout
}

#[test]
fn base64_ignores_whitespace() {
    assert_eq!(decode("Zm9v\nYmFy").unwrap(), b"foobar");
    assert_eq!(decode(" Zm 9v ").unwrap(), b"foo");
}

#[test]
fn utf8_valid_strings() {
    validate_utf8(b"").unwrap();
    validate_utf8(b"ascii").unwrap();
    validate_utf8("café".as_bytes()).unwrap();
    validate_utf8("日本語🚀".as_bytes()).unwrap();
}

#[test]
fn utf8_rejects_overlong_and_invalid_leads() {
    assert!(validate_utf8(&[0xC0, 0x80]).is_err()); // overlong NUL
    assert!(validate_utf8(&[0xC1, 0x81]).is_err());
    assert!(validate_utf8(&[0xF5, 0x80, 0x80, 0x80]).is_err());
    assert!(validate_utf8(&[0xFF]).is_err());
    assert!(validate_utf8(&[0x80]).is_err()); // lone continuation
}

#[test]
fn utf8_rejects_surrogates() {
    // U+D800 encoded as ED A0 80
    assert!(validate_utf8(&[0xED, 0xA0, 0x80]).is_err());
    // U+DFFF
    assert!(validate_utf8(&[0xED, 0xBF, 0xBF]).is_err());
}

#[test]
fn utf8_fragmented_across_feeds() {
    let mut v = Utf8Validator::new();
    // é = C3 A9
    v.feed(&[0xC3]).unwrap();
    assert!(v.finish().is_err());
    v.feed(&[0xA9]).unwrap();
    v.finish().unwrap();

    let mut v = Utf8Validator::new();
    // 日 = E6 97 A5
    v.feed(&[0xE6]).unwrap();
    v.feed(&[0x97, 0xA5]).unwrap();
    v.finish().unwrap();
}

#[test]
fn utf8_reset_clears_pending() {
    let mut v = Utf8Validator::new();
    v.feed(&[0xE6]).unwrap();
    v.reset();
    v.finish().unwrap();
    validate_utf8(b"ok").unwrap();
}

#[test]
fn utf8_four_byte_emoji_split() {
    let bytes = "😀".as_bytes(); // F0 9F 98 80
    assert_eq!(bytes.len(), 4);
    let mut v = Utf8Validator::new();
    v.feed(&bytes[..1]).unwrap();
    v.feed(&bytes[1..3]).unwrap();
    v.feed(&bytes[3..]).unwrap();
    v.finish().unwrap();
}
