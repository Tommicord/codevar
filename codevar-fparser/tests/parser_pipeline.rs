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

//! Integration tests for per-format parsers.

use codevar_fparser::{
    FileType, FileTypeDetector, FormatParser, JpegParser, PNG_SIG, PngParser, Utf8Parser,
    Windows12xxParser,
};

#[test]
fn jpeg_pipeline() {
    let mut data = vec![0xFF, 0xD8];
    data.extend_from_slice(&[0xFF, 0xFE]);
    data.extend_from_slice(&4u16.to_be_bytes());
    data.extend_from_slice(b"hi");
    data.extend_from_slice(&[0xFF, 0xD9]);

    assert_eq!(FileTypeDetector::new().detect(&data), FileType::Jpeg);
    let v = JpegParser::validate(&data);
    assert!(v.as_ref().is_ok_and(|r| r.valid));
    let segs = JpegParser::segments(&data);
    assert!(segs.is_ok());
    assert!(segs.unwrap_or_default().iter().any(|s| s.tag == 0xD8));
}

#[test]
fn png_and_utf8_pipelines() {
    let mut png = PNG_SIG.to_vec();
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&8u32.to_be_bytes());
    png.extend_from_slice(&8u32.to_be_bytes());
    while png.len() < 24 {
        png.push(0);
    }
    assert_eq!(FileTypeDetector::new().detect(&png), FileType::Png);
    assert!(PngParser::validate(&png).is_ok_and(|v| v.valid));

    let text = b"fn main() {}";
    assert_eq!(FileTypeDetector::new().detect(text), FileType::Utf8);
    assert!(Utf8Parser::validate(text).is_ok_and(|v| v.valid));
}

#[test]
fn windows1252_euro() {
    let chars = Windows12xxParser::cp1252().decode(&[0x80]);
    assert_eq!(chars.unwrap_or_default(), vec!['€']);
}
