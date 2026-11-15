//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! End-to-end checks at the size the codec exists for.
//!
//! The unit tests pin the stages against the Python prototype; these drive
//! the public API over a document big enough to exercise what a small buffer
//! cannot: a byte history longer than a page, a match model that has to find
//! recurrences tens of kilobytes apart, and a `u32` history position that a
//! `u16` would wrap. They also pin the ratio, because a codec that stops
//! beating the raw element array has stopped earning its model.

use codevar_wredit_werc::codec::{decode, encode};
use codevar_wredit_werc::element::{COMMON_MASK, pack};
use codevar_wredit_werc::frame::{Header, Params};

/// A document of `lines` source lines, each with a distinct identifier so the
/// match model has to earn its keep on the repeated shape rather than on an
/// identical copy. Every eighth line carries one wide codepoint, which is
/// what makes this a three-plane document rather than a two-plane one.
fn document(lines: usize) -> Vec<u32> {
    let mut text = String::with_capacity(lines * 44);
    use std::fmt::Write;
    for line in 0..lines {
        let _ = writeln!(text, "    let value_{line} = compute(a, b);");
        if line % 8 == 7 {
            let _ = writeln!(text, "    // 注 line {line}");
        }
    }
    text.chars()
        .map(|c| pack(c as u32, COMMON_MASK))
        .collect()
}

/// The parameters these tests code at: 1.2 MiB of tables, no cache cliff.
const PARAMS: Params = Params {
    mem_bits: 16,
    sse_bits: 12,
    lr_shift: 11,
};

#[test]
fn a_sixteen_kib_document_round_trips_exactly() {
    let source = document(520);
    let blob = encode(&PARAMS, &[], &source).expect("a valid document codes");
    let header = Header::parse(&blob).expect("the frame it wrote parses");
    assert_eq!(header.count as usize, source.len());
    // The wide comment lines make this a three-plane document: plane 0, the
    // codepoint bits 8..16 and the type byte. Plane 2 stays dropped.
    assert_eq!(header.flags, codevar_wredit_werc::plane::presence(&source));
    assert_eq!(header.body_len(), Some(source.len() * 3));

    let mut back = vec![0u32; source.len()];
    let recovered = decode(&blob, &[], &mut back).expect("the frame decodes");
    assert_eq!(recovered, source.len());
    assert_eq!(&back[..recovered], source.as_slice());
}

#[test]
fn a_document_of_repeated_shape_ships_a_fraction_of_its_elements() {
    // 2080 elements of packed u32 is 8 KB of RAM; the codec is only worth
    // calling if it ships materially less than that.
    let source = document(520);
    let blob = encode(&PARAMS, &[], &source).expect("a valid document codes");
    let coded = blob.len() - codevar_wredit_werc::frame::HEADER_SIZE;
    assert!(
        coded * 6 < source.len() * 4,
        "{} element bytes shipped as {coded} coded bytes",
        source.len() * 4
    );
}

#[test]
fn a_primed_document_needs_its_own_dictionary() {
    let dictionary = document(400);
    let source = document(260);
    let blob = encode(&PARAMS, &dictionary, &source).expect("a primed document codes");
    let cold = encode(&PARAMS, &[], &source).expect("the same document codes unprimed");
    assert!(
        blob.len() < cold.len(),
        "primed {} bytes, unprimed {}",
        blob.len(),
        cold.len()
    );

    let mut back = vec![0u32; source.len()];
    assert_eq!(decode(&blob, &dictionary, &mut back).ok(), Some(source.len()));
    assert_eq!(&back[..source.len()], source.as_slice());
}

#[test]
fn a_buffer_larger_than_the_document_is_left_alone() {
    let source = document(200);
    let blob = encode(&PARAMS, &[], &source).expect("a valid document codes");
    let mut back = vec![0xDEAD_BEEF; source.len() + 16];
    assert_eq!(decode(&blob, &[], &mut back).ok(), Some(source.len()));
    assert_eq!(&back[..source.len()], source.as_slice());
    assert_eq!(back[source.len()], 0xDEAD_BEEF, "past the count is the caller's");
}
