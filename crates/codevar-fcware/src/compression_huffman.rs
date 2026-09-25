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

use crate::compression_bits::BitLaneReader;
use crate::compression_error::{CompressorError, CompressorResult};
use crate::compression_frame::FrameKind;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Debug)]
enum HuffmanNode {
    Leaf(u8),
    Branch(Box<HuffmanNode>, Box<HuffmanNode>),
}

fn huffman_tree(frequencies: &[u32; 256]) -> Option<HuffmanNode> {
    let mut nodes: Vec<(u32, u8, HuffmanNode)> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency != 0)
        .map(|(value, frequency)| (*frequency, value as u8, HuffmanNode::Leaf(value as u8)))
        .collect();
    if nodes.is_empty() {
        return None;
    }
    while nodes.len() > 1 {
        nodes.sort_by_key(|(frequency, minimum, node)| {
            (*frequency, *minimum, matches!(node, HuffmanNode::Branch(_, _)))
        });
        let (left_frequency, left_minimum, left) = nodes.remove(0);
        let (right_frequency, right_minimum, right) = nodes.remove(0);
        nodes.push((
            left_frequency.saturating_add(right_frequency),
            left_minimum.min(right_minimum),
            HuffmanNode::Branch(Box::new(left), Box::new(right)),
        ));
    }
    nodes.pop().map(|(_, _, node)| node)
}

fn huffman_codes(node: &HuffmanNode, prefix: u32, length: u8, codes: &mut [(u32, u8); 256]) {
    match node {
        HuffmanNode::Leaf(value) => codes[usize::from(*value)] = (prefix, length.max(1)),
        HuffmanNode::Branch(left, right) => {
            huffman_codes(left, prefix << 1, length.saturating_add(1), codes);
            huffman_codes(right, (prefix << 1) | 1, length.saturating_add(1), codes);
        }
    }
}

/// Encodes `input` as a Huffman frame.
pub fn huffman_encode(input: &[u8]) -> CompressorResult<Vec<u8>> {
    let length = u32::try_from(input.len()).map_err(|_| CompressorError::InputTooLarge)?;
    let mut frequencies = [0u32; 256];
    // SAFETY: each `value` indexes `frequencies[0..256]`.
    for &value in input {
        let slot = unsafe { frequencies.get_unchecked_mut(usize::from(value)) };
        *slot = slot
            .checked_add(1)
            .ok_or(CompressorError::InputTooLarge)?;
    }
    let mut output = Vec::with_capacity(8 + 1024 + input.len());
    output.extend_from_slice(FrameKind::Huffman.magic());
    output.extend_from_slice(&length.to_be_bytes());
    for frequency in frequencies {
        output.extend_from_slice(&frequency.to_be_bytes());
    }
    let Some(tree) = huffman_tree(&frequencies) else {
        output.push(0);
        return Ok(output);
    };
    let mut codes = [(0u32, 0u8); 256];
    huffman_codes(&tree, 0, 0, &mut codes);
    let mut buffer = 0u64;
    let mut bits = 0u8;
    let mut payload = Vec::with_capacity(input.len());
    for &value in input {
        let (code, code_length) = unsafe { *codes.get_unchecked(usize::from(value)) };
        buffer = (buffer << code_length) | u64::from(code);
        bits = bits.saturating_add(code_length);
        while bits >= 8 {
            bits -= 8;
            payload.push((buffer >> bits) as u8);
            buffer &= if bits == 0 { 0 } else { (1u64 << bits) - 1 };
        }
    }
    let padding = if bits == 0 { 0 } else { 8 - bits };
    if bits != 0 {
        payload.push((buffer << padding) as u8);
    }
    output.push(padding);
    output.extend_from_slice(&payload);
    Ok(output)
}

/// Decodes a Huffman frame using SIMD lane-backed bit reads.
pub fn huffman_decode(frame: &[u8]) -> CompressorResult<Vec<u8>> {
    if frame.len() < 1032 {
        return Err(CompressorError::TruncatedFrame);
    }
    if FrameKind::from_magic(frame) != Some(FrameKind::Huffman) {
        return Err(CompressorError::InvalidFrame);
    }
    let expected = unsafe {
        let p = frame.as_ptr().add(3);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as usize
    };
    let mut frequencies = [0u32; 256];
    for (index, frequency) in frequencies.iter_mut().enumerate() {
        let start = 7 + index * 4;
        // SAFETY: `7 + 256*4 = 1031`, and frame.len() >= 1032.
        unsafe {
            let p = frame.as_ptr().add(start);
            *frequency = u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]);
        }
    }
    let padding = frame[1031];
    if padding > 7 {
        return Err(CompressorError::invalid_control(padding));
    }
    let payload = &frame[1032..];
    if expected == 0 {
        return if payload.is_empty() {
            Ok(Vec::new())
        } else {
            Err(CompressorError::length_mismatch(0, payload.len()))
        };
    }
    let tree = huffman_tree(&frequencies).ok_or(CompressorError::InvalidFrame)?;
    if matches!(tree, HuffmanNode::Leaf(_)) {
        let value = match tree {
            HuffmanNode::Leaf(value) => value,
            HuffmanNode::Branch(_, _) => return Err(CompressorError::InvalidFrame),
        };
        return Ok(vec![value; expected]);
    }
    let available_bits = payload
        .len()
        .checked_mul(8)
        .and_then(|bits| bits.checked_sub(usize::from(padding)))
        .ok_or(CompressorError::TruncatedFrame)?;
    let mut output = Vec::with_capacity(expected);
    let mut bits = BitLaneReader::new(payload, available_bits);
    while output.len() < expected {
        let mut node = &tree;
        loop {
            match node {
                HuffmanNode::Leaf(value) => {
                    output.push(*value);
                    break;
                }
                HuffmanNode::Branch(left, right) => {
                    let bit = bits.next_bit()?;
                    node = if bit == 0 { left } else { right };
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{huffman_decode, huffman_encode};
    use crate::compression_error::CompressorError;

    #[test]
    fn roundtrip_empty_uniform_and_mixed() {
        for sample in [
            &b""[..],
            &b"aaaaaaaa"[..],
            &b"the quick brown fox jumps over the lazy dog"[..],
            &[0u8, 1, 2, 3, 4, 5, 255, 128][..],
        ] {
            let frame = huffman_encode(sample).expect("encode");
            assert_eq!(huffman_decode(&frame).expect("decode"), sample);
        }
    }

    #[test]
    fn rejects_bad_magic_padding_and_short() {
        assert_eq!(huffman_decode(b"HF"), Err(CompressorError::TruncatedFrame));
        let mut frame = huffman_encode(b"abc").expect("encode");
        frame[0] = b'X';
        assert_eq!(huffman_decode(&frame), Err(CompressorError::InvalidFrame));
        let mut frame = huffman_encode(b"abc").expect("encode");
        frame[1031] = 8;
        assert_eq!(huffman_decode(&frame), Err(CompressorError::invalid_control(8)));
    }
}
