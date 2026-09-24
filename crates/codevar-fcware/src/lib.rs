#![allow(dead_code)]

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

//! The FcWare frame-oriented multi-codec compression.
//!
//! The stable entry points live in [`compression`]:
//! [`compression::compress`], [`compression::decompress`],
//! [`compression::compress_values`], and [`compression::decompress_values`].

extern crate alloc;

mod compression_bits;
mod compression_bitward;
mod compression_delta;
mod compression_dysu;
mod compression_frame;
mod compression_huffman;
mod compression_lzmatch;
mod compression_valmap;

pub mod compression;
pub mod compression_error;
pub mod compression_stream;

pub use compression::{
    BIT_LANES, BitReader, Codec, DEFAULT_DYSU_BLOCK, Frame, StreamWorkspace,
    StreamingEncoder, ValueCodec, bit_reader, compress, compress_values, decompress,
    decompress_values, detect_frame, lz_match_bound, lz_match_encode,
};
pub use compression_error::{CompressorError, CompressorResult};
