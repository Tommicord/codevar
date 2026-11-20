//! Integration tests for the Wredit editing engine.
//! Tests cover encoding round-trips, BOM handling, Unicode segmentation,
//! and various editing scenarios.

use codevar_wredit::TextEditableEncoding;
use codevar_wredit::encoding::DocumentEncoding;
use codevar_wredit::encoding_dispatch::{DecodePolicy, EncodePolicy, decode, encode};
use codevar_wredit::fs::DiskFs;
use codevar_wredit::fs_block::MemoryDisk as MemoryDiskType;
use codevar_wredit::storage_bind::BoundDocument;

type TestBound<'a> = BoundDocument<'a, MemoryDiskType, 256, 1024>;

#[test]
fn encoding_utf8_round_trip() {
    // UTF-8 encode "Hello" then decode back should round-trip.
    let scalars: Vec<u32> = [0x0048, 0x0065, 0x006C, 0x006C, 0x006F]
        .iter()
        .copied()
        .collect();
    let bytes = encode(&scalars, DocumentEncoding::Utf8, EncodePolicy::Strict).unwrap();
    assert_eq!(&bytes, b"Hello");
    let decoded = decode(&bytes, DocumentEncoding::Utf8, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf16le_round_trip() {
    // UTF-16LE encode "Hello" then decode back should round-trip.
    let scalars: Vec<u32> = [0x0048, 0x0065, 0x006C, 0x006C, 0x006F]
        .iter()
        .copied()
        .collect();
    let bytes = encode(&scalars, DocumentEncoding::Utf16Le, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf16Le, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf32le_round_trip() {
    // UTF-32LE encode "Hello" then decode back should round-trip.
    let scalars: Vec<u32> = [0x0048, 0x0065, 0x006C, 0x006C, 0x006F]
        .iter()
        .copied()
        .collect();
    let bytes = encode(&scalars, DocumentEncoding::Utf32Le, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf32Le, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf8_4byte_char_round_trip() {
    let scalars: Vec<u32> = vec![0x10348];
    let bytes = encode(&scalars, DocumentEncoding::Utf8, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf8, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf16le_surrogate_pair() {
    let scalars: Vec<u32> = vec![0x1F600];
    let bytes = encode(&scalars, DocumentEncoding::Utf16Le, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf16Le, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_invalid_utf8_replace_policy() {
    let invalid = vec![0xC0, 0xFF, 0xC1];
    let decoded = decode(&invalid, DocumentEncoding::Utf8, DecodePolicy::Replace).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn unicode_segmentation_grapheme_clusters() {
    use codevar_wredit::unicode_segmentation::{
        is_grapheme_boundary, next_grapheme_position, prev_grapheme_position,
    };

    // Combining character: e + combining acute accent.
    let scalars: Vec<u32> = vec![0x0065, 0x0301];
    // The cluster occupies positions 0..2; boundaries at 0 and 2.
    // next_grapheme_position starting at 0 should land past the cluster.
    let gb_next = next_grapheme_position(&scalars, 0);
    let gb_prev = prev_grapheme_position(&scalars, scalars.len());
    assert_eq!(
        gb_next, 2,
        "next grapheme position after combining cluster: got {}",
        gb_next
    );
    assert_eq!(
        gb_prev, 0,
        "prev grapheme position before combining cluster: got {}",
        gb_prev
    );
    // In a combining cluster, index 1 is NOT a grapheme boundary (GB9).
    assert!(
        !is_grapheme_boundary(&scalars, 1),
        "combining mark should not be a grapheme boundary"
    );
    // Boundaries at start (0) and end (2) ARE valid.
    assert!(
        is_grapheme_boundary(&scalars, 0),
        "start of text is a grapheme boundary"
    );
    assert!(
        is_grapheme_boundary(&scalars, 2),
        "end of text is a grapheme boundary"
    );
}

#[test]
fn unicode_segmentation_word_boundaries() {
    use codevar_wredit::unicode_segmentation::{is_word_boundary, next_word_position};

    // Latin word: "hello"
    let scalars: Vec<u32> = vec![0x0068, 0x0065, 0x006C, 0x006C, 0x006F];
    // WB1/WB2: start and end of text are word boundaries.
    assert!(is_word_boundary(&scalars, 0), "start of text is a word boundary");
    assert!(
        is_word_boundary(&scalars, scalars.len()),
        "end of text is a word boundary"
    );
    // Within the word there are no other boundaries.
    for i in 1..scalars.len() {
        assert!(!is_word_boundary(&scalars, i), "unexpected word boundary at {i}");
    }
    let next = next_word_position(&scalars, 0);
    assert_eq!(next, 5, "next word boundary after 'hello': got {}", next);
}

#[test]
fn encoding_utf16be_round_trip() {
    // UTF-16BE encode "Hello" then decode back should round-trip.
    let scalars: Vec<u32> = [0x0048, 0x0065, 0x006C, 0x006C, 0x006F]
        .iter()
        .copied()
        .collect();
    let bytes = encode(&scalars, DocumentEncoding::Utf16Be, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf16Be, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf32be_round_trip() {
    // UTF-32BE encode "Hello" then decode back should round-trip.
    let scalars: Vec<u32> = [0x0048, 0x0065, 0x006C, 0x006C, 0x006F]
        .iter()
        .copied()
        .collect();
    let bytes = encode(&scalars, DocumentEncoding::Utf32Be, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf32Be, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf16be_4byte_char_round_trip() {
    let scalars: Vec<u32> = vec![0x10348];
    let bytes = encode(&scalars, DocumentEncoding::Utf16Be, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf16Be, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_utf32be_4byte_char_round_trip() {
    let scalars: Vec<u32> = vec![0x10348];
    let bytes = encode(&scalars, DocumentEncoding::Utf32Be, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, DocumentEncoding::Utf32Be, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_single_byte_round_trip() {
    // windows-1252 covers ASCII + é and the euro sign.
    let scalars: Vec<u32> = vec![0x0063, 0x0061, 0x0066, 0x00E9, 0x20AC];
    let encoding = DocumentEncoding::for_label("windows-1252").unwrap();
    let bytes = encode(&scalars, encoding, EncodePolicy::Strict).unwrap();
    let decoded = decode(&bytes, encoding, DecodePolicy::Strict).unwrap();
    assert_eq!(decoded, scalars);
}

#[test]
fn encoding_strict_policy_rejects_invalid_utf8() {
    let invalid = vec![0xC0, 0xFF, 0xC1];
    let decoded = decode(&invalid, DocumentEncoding::Utf8, DecodePolicy::Strict);
    assert!(decoded.is_err());
}

#[test]
fn encoding_strict_policy_rejects_lone_surrogate() {
    // U+D800 is a lone surrogate (not a valid scalar).
    let scalars: Vec<u32> = vec![0xD800];
    let bytes = encode(&scalars, DocumentEncoding::Utf8, EncodePolicy::Strict);
    assert!(bytes.is_err());
}

#[test]
fn encoding_strict_policy_rejects_invalid_utf32_scalar() {
    // U+110000 is past the maximum scalar value.
    let scalars: Vec<u32> = vec![0x110000];
    let bytes = encode(&scalars, DocumentEncoding::Utf32Le, EncodePolicy::Strict);
    assert!(bytes.is_err());
}

#[test]
fn encoding_utf16_strict_rejects_lone_surrogate_units() {
    // A lone high surrogate without a trailing low surrogate, UTF-16LE:
    // high surrogate U+D800 -> [0x00, 0xD8], then a terminator.
    let bytes = vec![0x00, 0xD8, 0x00, 0x00];
    let decoded = decode(&bytes, DocumentEncoding::Utf16Le, DecodePolicy::Strict);
    assert!(decoded.is_err());
}

#[test]
fn storage_crlf_round_trip() {
    let disk = MemoryDiskType::new(256).unwrap();
    let mut fs = DiskFs::format(disk, 256, 128).unwrap();
    let inode = fs.create(2, "crlf.txt", 0x8000).unwrap();
    // Write "A\r\nB" as UTF-8 bytes.
    fs.write_at(inode, 0, b"A\r\nB").unwrap();
    let mut bound: TestBound = BoundDocument::open(&mut fs, inode).unwrap();
    // A, CR, LF, B = 4 scalars.
    assert_eq!(bound.raw_length(), 4);
    let chars: Vec<u32> = bound
        .raw_slice(0, bound.raw_length())
        .unwrap()
        .iter()
        .map(|&e| codevar_wredit::base::packed_char(e))
        .collect();
    assert_eq!(chars, vec![0x0041, 0x000D, 0x000A, 0x0042]);
    bound.save().unwrap();
    drop(bound);
    let mut buf = vec![0u8; 512];
    fs.read_at(inode, 0, &mut buf).unwrap();
    let data = buf
        .into_iter()
        .take_while(|&b| b != 0)
        .collect::<Vec<_>>();
    assert_eq!(&data[..], b"A\r\nB");
}

#[test]
fn storage_bomless_utf16be_needs_explicit_encoding() {
    let disk = MemoryDiskType::new(256).unwrap();
    let mut fs = DiskFs::format(disk, 256, 128).unwrap();
    let inode = fs.create(2, "bomless_be.txt", 0x8000).unwrap();
    let bytes = encode(
        &[u32::from('o'), u32::from('k')],
        DocumentEncoding::Utf16Be,
        EncodePolicy::Strict,
    )
    .unwrap();
    fs.write_at(inode, 0, &bytes).unwrap();

    let bound: TestBound =
        BoundDocument::open_with_encoding(&mut fs, inode, DocumentEncoding::Utf16Be).unwrap();
    assert_eq!(bound.encoding(), DocumentEncoding::Utf16Be);
    assert!(!bound.emits_bom());
    assert_eq!(bound.stream().read_string(0, 2).unwrap(), "ok");
}

#[test]
fn segmentation_grapheme_cluster_emoji_zwj() {
    use codevar_wredit::unicode_segmentation::next_grapheme_position;

    // 👨‍👩‍👧‍👦 family emoji: ZWJ sequence, one grapheme cluster.
    let scalars: Vec<u32> = vec![0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466];
    assert_eq!(next_grapheme_position(&scalars, 0), 7);

    // Regional indicator pair: one grapheme cluster (flag).
    let scalars: Vec<u32> = vec![0x1F1E6, 0x1F1E6];
    assert_eq!(next_grapheme_position(&scalars, 0), 2);
}

#[test]
fn segmentation_word_position_across_whitespace() {
    use codevar_wredit::unicode_segmentation::{next_word_position, prev_word_position};

    // "hello world" – word boundaries: [h,e,l,l,o,' ',w,o,r,l,d]
    // (1..5 = "ello", 5 = space boundary, 6 = 'w' start)
    let scalars: Vec<u32> = vec![
        0x0068, 0x0065, 0x006C, 0x006C, 0x006F, 0x0020, 0x0077, 0x006F, 0x0072, 0x006C, 0x0064,
    ];
    // next_word_position from 0 -> lands on the next boundary (space, index 5)
    assert_eq!(next_word_position(&scalars, 0), 5);
    // prev_word_position from 5 -> returns the previous boundary (start, index 0)
    assert_eq!(prev_word_position(&scalars, 5), 0);
    // next_word_position from 5 -> next word starts at index 6
    assert_eq!(next_word_position(&scalars, 5), 6);
    // prev_word_position from 6 -> back to the space (index 5)
    assert_eq!(prev_word_position(&scalars, 6), 5);
    // at end of document, next_word_position returns len
    assert_eq!(next_word_position(&scalars, scalars.len()), scalars.len());
}

#[test]
fn storage_bom_stripped_on_load_and_reemitted_on_save() {
    let disk = MemoryDiskType::new(256).unwrap();
    let mut fs = DiskFs::format(disk, 256, 128).unwrap();
    let inode = fs.create(2, "bom.txt", 0x8000).unwrap();
    fs.write_at(inode, 0, b"\xEF\xBB\xBFHello")
        .unwrap();
    let mut bound: TestBound = BoundDocument::open(&mut fs, inode).unwrap();
    assert_eq!(bound.raw_length(), 5);
    let chars: Vec<u32> = bound
        .raw_slice(0, bound.raw_length())
        .unwrap()
        .iter()
        .map(|&e| codevar_wredit::base::packed_char(e))
        .collect();
    assert_eq!(chars, vec![0x0048, 0x0065, 0x006C, 0x006C, 0x006F]);
    bound.save().unwrap();
    drop(bound);
    let mut buf = vec![0u8; 512];
    fs.read_at(inode, 0, &mut buf).unwrap();
    let data = buf
        .into_iter()
        .take_while(|&b| b != 0)
        .collect::<Vec<_>>();
    assert_eq!(&data[0..3], b"\xEF\xBB\xBF");
    assert_eq!(&data[3..], b"Hello");
}
