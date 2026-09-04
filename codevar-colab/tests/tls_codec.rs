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

//! TLS presentation-language codec tests.

use codevar_colab::network::tls_codec::{
    Reader, fill_u8_len, fill_u16_len, fill_u24_len, put_u16, put_u24, put_u32,
    put_vec_u8, put_vec_u16, put_vec_u24, start_u8_vec, start_u16_vec, start_u24_vec,
};

#[test]
fn reader_integers_and_vectors() {
    let mut buf = Vec::new();
    put_u16(&mut buf, 0x1234);
    put_u24(&mut buf, 0xABCDEF);
    put_u32(&mut buf, 0xDEAD_BEEF);
    put_vec_u8(&mut buf, b"ab").unwrap();
    put_vec_u16(&mut buf, b"cdef").unwrap();
    put_vec_u24(&mut buf, b"gh").unwrap();

    let mut r = Reader::new(&buf);
    assert_eq!(r.u16().unwrap(), 0x1234);
    assert_eq!(r.u24().unwrap(), 0xABCDEF);
    assert_eq!(r.u32().unwrap(), 0xDEAD_BEEF);
    assert_eq!(r.vec_u8().unwrap(), b"ab");
    assert_eq!(r.vec_u16().unwrap(), b"cdef");
    assert_eq!(r.vec_u24().unwrap(), b"gh");
    r.expect_empty("end").unwrap();
}

#[test]
fn reader_underflow_and_trailing() {
    let mut r = Reader::new(&[0x01]);
    assert!(r.u16().is_err());

    let mut r = Reader::new(&[0x00, 0x01, 0xff]);
    assert_eq!(r.u16().unwrap(), 1);
    assert!(r.expect_empty("x").is_err());
    assert_eq!(r.remaining(), 1);
    assert!(!r.is_empty());
}

#[test]
fn length_prefix_helpers() {
    let mut out = Vec::new();
    let i8 = start_u8_vec(&mut out);
    out.extend_from_slice(b"xy");
    fill_u8_len(&mut out, i8).unwrap();
    assert_eq!(out[0], 2);

    let mut out = Vec::new();
    let i16 = start_u16_vec(&mut out);
    out.extend_from_slice(&[1, 2, 3]);
    fill_u16_len(&mut out, i16).unwrap();
    assert_eq!(&out[..2], &[0, 3]);

    let mut out = Vec::new();
    let i24 = start_u24_vec(&mut out);
    out.push(9);
    fill_u24_len(&mut out, i24).unwrap();
    assert_eq!(&out[..3], &[0, 0, 1]);
}

#[test]
fn vector_length_limits() {
    let big = vec![0u8; 256];
    let mut out = Vec::new();
    assert!(put_vec_u8(&mut out, &big).is_err());

    // u16 limit is fine for 256
    put_vec_u16(&mut out, &big).unwrap();
}

#[test]
fn reader_bytes_split() {
    let data = b"hello";
    let mut r = Reader::new(data);
    assert_eq!(r.bytes(2).unwrap(), b"he");
    assert_eq!(r.rest(), b"llo");
    assert_eq!(r.u8().unwrap(), b'l');
}
