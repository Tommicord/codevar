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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Zero-branch-misprediction dynamic dispatch table for text encoders.
//!
//! The table maps each `ENC_*` tag (from [`super::base`]) directly to a
//! [`DispatchEntry`] holding encoder/decoder function pointers plus the
//! [`codevar_textlike_encode::encoding::Encoding`] metadata. Lookup is a
//! single indexed load — no `match`, no `if`, zero data-dependent branches
//! on the hot path.
//!
//! Each `DispatchEntry` is `#[repr(C, align(64))]` and exactly one cache
//! line (64 bytes), so the hot encode/decode fields load in a single
//! cache-line fetch. The table itself is read-only `.rodata`, indexed by
//! tag, so the hot entry stays resident and prefetch-friendly.

use codevar_textlike_encode::encoding::VariantEncoding;
use codevar_textlike_encode::encoding::{
    Decoder, DecoderResult, Encoder, EncoderResult, Encoding, UTF_8, UTF_16BE, UTF_16LE,
};
use codevar_textlike_encode::single_byte_tables::INDEX_WINDOWS_1252;
use codevar_wredit_asm::{CACHE_LINE_SIZE, PREFETCH_T0, prefetch};
use core::sync::atomic::{AtomicPtr, Ordering};

use codevar_logger::{log_debug, log_info};

use spin::Mutex;

pub use super::base::{ENC_SINGLE_BYTE, ENC_UNKNOWN, ENC_UTF8, ENC_UTF16BE, ENC_UTF16LE};

/// Number of entries in the dispatch table. Dense indices `0..=ENC_SINGLE_BYTE`.
pub const DISPATCH_TABLE_LEN: usize = (ENC_SINGLE_BYTE as usize) + 1;

/// Out-of-range tags clamp to entry `ENC_SINGLE_BYTE`, the fallback/custom row.
/// Encode UTF-8 text to bytes.
type EncodeUtf8Fn =
    unsafe fn(encoder: &mut Encoder, src: &str, dst: &mut [u8], last: bool) -> (EncoderResult, usize, usize);

/// Encode UTF-16 text to bytes.
type EncodeUtf16Fn = unsafe fn(
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize);

/// Decode bytes to UTF-8 text.
type DecodeUtf8Fn =
    unsafe fn(decoder: &mut Decoder, src: &[u8], dst: &mut [u8], last: bool) -> (DecoderResult, usize, usize);

/// Decode bytes to UTF-16 text.
type DecodeUtf16Fn = unsafe fn(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize);

/// One dispatch-table row: exactly one cache line (64 bytes).
///
/// Field order keeps the hot encode/decode pointers at the front so a single
/// indexed load brings in everything the hot path touches.
///
/// # Safety
///
/// Each thunk must only be invoked for the encoding whose entry owns it; the
/// invariant is upheld because entries are constructed statically and never
/// mutated after table initialization. The table is read-only for the life of
/// the program, so it is safe to share across threads.
#[repr(C, align(64))]
pub struct DispatchEntry {
    /// The `ENC_*` tag; implicit = array index, not stored for lookup.
    pub tag: u16,
    /// Padding so function pointers start at an 8-byte aligned offset.
    _pad0: [u8; 6],
    /// Encode UTF-8 → bytes.
    pub encode_utf8: EncodeUtf8Fn,
    /// Encode UTF-16 → bytes.
    pub encode_utf16: EncodeUtf16Fn,
    /// Decode bytes → UTF-8.
    pub decode_utf8: DecodeUtf8Fn,
    /// Decode bytes → UTF-16.
    pub decode_utf16: DecodeUtf16Fn,
    /// Metadata (`name`, `variant`) for `new_encoder`/`new_decoder` (cold).
    pub encoding: &'static Encoding,
    /// Fallback/custom coder-instance pointer (`ENC_UNKNOWN` or user
    /// registered). `NULL` ⇒ this entry self-handles via its own thunks.
    ///
    /// Stored as an `AtomicPtr` so user coders can be registered and
    /// unregistered at runtime from any thread without mutating the read-only
    /// dispatch table. A non-null value must point to a live coder instance
    /// whose vtable is registered alongside it (see [`register_custom_coder`]).
    pub custom: AtomicPtr<()>,
    /// Pointer into variant static data (e.g. `SingleByte` `[u16; 128]`),
    /// `NULL` for unit variants. Prefetched for cache warmth.
    pub variant_data: *const (),
    /// Pad to exactly `CACHE_LINE_SIZE` so no two rows share a line.
    _pad1: [u8; _PAD1_LEN],
}

// Layout (bytes): tag(2) + pad(6) + 4*fn_ptr(32) + encoding_ref(8) + custom(8) + variant_data(8) = 64.
// _pad1 must be 0.
const _PAD1_LEN: usize = {
    let used = core::mem::size_of::<u16>()
        + 6
        + core::mem::size_of::<EncodeUtf8Fn>()
        + core::mem::size_of::<EncodeUtf16Fn>()
        + core::mem::size_of::<DecodeUtf8Fn>()
        + core::mem::size_of::<DecodeUtf16Fn>()
        + core::mem::size_of::<&'static Encoding>()
        + core::mem::size_of::<*mut ()>()
        + core::mem::size_of::<*const ()>();
    assert!(used <= CACHE_LINE_SIZE, "DispatchEntry exceeds one cache line");
    CACHE_LINE_SIZE - used
};

// `DispatchEntry` contains raw pointers, which are not `Sync` by default.
// The dispatch table is immutable after construction, so it is safe to share.
unsafe impl Sync for DispatchEntry {}

/// The static dispatch table, indexed directly by `ENC_*` tag.
///
/// Indexing `TABLE[tag as usize]` is a single load — zero branches. Out-of-
/// range tags clamp to the fallback row so an invalid tag never falls
/// through to garbage.
pub static DISPATCH_TABLE: [DispatchEntry; DISPATCH_TABLE_LEN] = [
    // ENC_UNKNOWN — fallback/custom row. `custom` holds the user-registered
    // coder instance (or NULL → no-op thunks). tag field kept for introspection.
    DispatchEntry {
        tag: ENC_UNKNOWN,
        _pad0: [0; 6],
        encode_utf8: encode_unknown_utf8,
        encode_utf16: encode_unknown_utf16,
        decode_utf8: decode_unknown_utf8,
        decode_utf16: decode_unknown_utf16,
        encoding: &codevar_textlike_encode::encoding::Encoding {
            name: "UNKNOWN",
            variant: VariantEncoding::Utf8,
        },
        custom: AtomicPtr::new(core::ptr::null_mut()),
        variant_data: core::ptr::null(),
        _pad1: [0; _PAD1_LEN],
    },
    // ENC_UTF8
    DispatchEntry {
        tag: ENC_UTF8,
        _pad0: [0; 6],
        encode_utf8: encode_utf8_utf8,
        encode_utf16: encode_utf8_utf16,
        decode_utf8: decode_utf8_utf8,
        decode_utf16: decode_utf8_utf16,
        encoding: &UTF_8,
        custom: AtomicPtr::new(core::ptr::null_mut()),
        variant_data: core::ptr::null(),
        _pad1: [0; _PAD1_LEN],
    },
    // ENC_UTF16LE
    DispatchEntry {
        tag: ENC_UTF16LE,
        _pad0: [0; 6],
        encode_utf8: encode_utf16le_utf8,
        encode_utf16: encode_utf16le_utf16,
        decode_utf8: decode_utf16le_utf8,
        decode_utf16: decode_utf16le_utf16,
        encoding: &UTF_16LE,
        custom: AtomicPtr::new(core::ptr::null_mut()),
        variant_data: core::ptr::null(),
        _pad1: [0; _PAD1_LEN],
    },
    // ENC_UTF16BE
    DispatchEntry {
        tag: ENC_UTF16BE,
        _pad0: [0; 6],
        encode_utf8: encode_utf16be_utf8,
        encode_utf16: encode_utf16be_utf16,
        decode_utf8: decode_utf16be_utf8,
        decode_utf16: decode_utf16be_utf16,
        encoding: &UTF_16BE,
        custom: AtomicPtr::new(core::ptr::null_mut()),
        variant_data: core::ptr::null(),
        _pad1: [0; _PAD1_LEN],
    },
    // ENC_SINGLE_BYTE — fallback/custom row (windows-1252 default).
    DispatchEntry {
        tag: ENC_SINGLE_BYTE,
        _pad0: [0; 6],
        encode_utf8: encode_single_byte_utf8,
        encode_utf16: encode_single_byte_utf16,
        decode_utf8: decode_single_byte_utf8,
        decode_utf16: decode_single_byte_utf16,
        encoding: &codevar_textlike_encode::encoding::Encoding {
            name: "windows-1252",
            variant: VariantEncoding::SingleByte(&INDEX_WINDOWS_1252, 0x00, 0x00, 0),
        },
        custom: AtomicPtr::new(core::ptr::null_mut()),
        variant_data: INDEX_WINDOWS_1252.as_ptr() as *const (),
        _pad1: [0; _PAD1_LEN],
    },
];

/// Resolve the dispatch entry for `tag`, clamping out-of-range tags to the
/// fallback row. Zero branches — direct index + `min` clamp.
#[inline]
pub fn entry_for_tag(tag: u16) -> &'static DispatchEntry {
    let index = tag.min(ENC_SINGLE_BYTE) as usize;
    // Safety: `index` is always `< DISPATCH_TABLE_LEN` due to the clamp.
    unsafe { &*DISPATCH_TABLE.as_ptr().add(index) }
}

/// Prefetch the hot dispatch entry for `tag` plus its metadata, hiding
/// load-to-use latency before the indirect call. `prefetch` is a no-op on
/// WASM (via `codevar_wredit_asm`), so this needs no target `#[cfg]` gates.
#[inline]
pub fn prefetch_entry(tag: u16) {
    prefetch_encoder_entry(tag);
    prefetch_decoder_entry(tag);
}

/// Prefetch the encoder I-cache target for `tag`: the dispatch entry's encode
/// thunks (`encode_utf8` + `encode_utf16`) and `variant_data`, but NOT the
/// decode fields. Warms the encode path before the indirect call.
///
/// # Safety
///
/// `entry_for_tag` clamps out-of-range tags to the fallback row, so the
/// index is always in bounds. The pointer reads are non-dereferencing prefetches
/// (never fault on invalid addresses), so null `variant_data` is skipped.
#[inline]
pub fn prefetch_encoder_entry(tag: u16) {
    let entry = entry_for_tag(tag);
    unsafe {
        prefetch(entry as *const DispatchEntry as *const u8, PREFETCH_T0);
        prefetch(
            &entry.encode_utf8 as *const EncodeUtf8Fn as *const u8,
            PREFETCH_T0,
        );
        prefetch(
            &entry.encode_utf16 as *const EncodeUtf16Fn as *const u8,
            PREFETCH_T0,
        );
        if !entry.variant_data.is_null() {
            prefetch(entry.variant_data as *const u8, PREFETCH_T0);
        }
    }
}

/// Prefetch the decoder I-cache target for `tag`: the dispatch entry's decode
/// thunks (`decode_utf8` + `decode_utf16`) and `variant_data`, but NOT the
/// encode fields. Warms the decode path before the indirect call.
///
/// # Safety
///
/// `entry_for_tag` clamps out-of-range tags to the fallback row, so the
/// index is always in bounds. The pointer reads are non-dereferencing prefetches
/// (never fault on invalid addresses), so null `variant_data` is skipped.
#[inline]
pub fn prefetch_decoder_entry(tag: u16) {
    let entry = entry_for_tag(tag);
    unsafe {
        prefetch(entry as *const DispatchEntry as *const u8, PREFETCH_T0);
        prefetch(
            &entry.decode_utf8 as *const DecodeUtf8Fn as *const u8,
            PREFETCH_T0,
        );
        prefetch(
            &entry.decode_utf16 as *const DecodeUtf16Fn as *const u8,
            PREFETCH_T0,
        );
        if !entry.variant_data.is_null() {
            prefetch(entry.variant_data as *const u8, PREFETCH_T0);
        }
    }
}

/// ---- custom coder dispatch -----------------------------------------------
/// A user-registered coder instance stores its pointer in `DispatchEntry.custom`.
/// To keep the read-only dispatch table immutable, the vtable (the four thunks
/// that operate on the instance) is registered separately in a thread-safe
/// registry keyed by tag, while `custom` holds the instance pointer itself.
///
/// `NULL` custom still means self-handling: the dispatcher thunks fall through
/// to the built-in no-op thunks when `custom` is null.

/// Vtable of thunks for a user-registered custom coder instance.
///
/// Each thunk receives the coder instance pointer (the `custom` field) as
/// `instance`, followed by the same arguments as the built-in thunks.
///
/// # Safety
///
/// The caller must ensure `instance` remains valid and correctly typed for the
/// lifetime of the registration, and that the thunk signatures match the
/// actual coder implementation.
#[repr(C)]
pub struct CustomCoderVtable {
    /// Encode UTF-8 → bytes through the custom instance.
    pub encode_utf8: unsafe fn(
        instance: *mut (),
        encoder: &mut Encoder,
        src: &str,
        dst: &mut [u8],
        last: bool,
    ) -> (EncoderResult, usize, usize),
    /// Encode UTF-16 → bytes through the custom instance.
    pub encode_utf16: unsafe fn(
        instance: *mut (),
        encoder: &mut Encoder,
        src: &[u16],
        dst: &mut [u8],
        last: bool,
    ) -> (EncoderResult, usize, usize),
    /// Decode bytes → UTF-8 through the custom instance.
    pub decode_utf8: unsafe fn(
        instance: *mut (),
        decoder: &mut Decoder,
        src: &[u8],
        dst: &mut [u8],
        last: bool,
    ) -> (DecoderResult, usize, usize),
    /// Decode bytes → UTF-16 through the custom instance.
    pub decode_utf16: unsafe fn(
        instance: *mut (),
        decoder: &mut Decoder,
        src: &[u8],
        dst: &mut [u16],
        last: bool,
    ) -> (DecoderResult, usize, usize),
}

/// Errors returned by custom-coder registration.
#[derive(Debug, PartialEq, Eq)]
pub enum CustomCoderError {
    /// A custom coder is already registered for this tag.
    AlreadyRegistered,
    /// The tag is not one of the dispatch-table rows that may carry a custom
    /// coder (`ENC_UNKNOWN` or `ENC_SINGLE_BYTE`).
    InvalidTag,
}

/// Thread-safe registry mapping a tag to its registered custom coder vtable.
///
/// Only `ENC_UNKNOWN` and `ENC_SINGLE_BYTE` may carry a custom coder, so a
/// fixed two-slot array indexed by tag is used instead of a hash map — this
/// avoids `alloc::collections` (unavailable under `no_std`) and keeps the hot
/// path lookup to a single array index plus a brief lock.
static CUSTOM_REGISTRY: Mutex<[Option<&'static CustomCoderVtable>; 2]> = Mutex::new([None, None]);

fn registry() -> &'static Mutex<[Option<&'static CustomCoderVtable>; 2]> {
    &CUSTOM_REGISTRY
}

/// Tags that may carry a custom coder: the documented fallback/custom rows.
const CUSTOM_TAGS: [u16; 2] = [ENC_UNKNOWN, ENC_SINGLE_BYTE];

/// Map a custom-registrable tag to its registry slot index.
const fn tag_to_slot(tag: u16) -> Option<usize> {
    match tag {
        ENC_UNKNOWN => Some(0),
        ENC_SINGLE_BYTE => Some(1),
        _ => None,
    }
}

/// Register a custom encoder/decoder instance for `tag`.
///
/// The `instance` pointer is stored in `DispatchEntry.custom` for `tag` and
/// `vtable` supplies the thunks that dispatch through it. `NULL` custom still
/// means self-handling; registering overwrites the built-in no-op thunks for
/// this tag's row so subsequent encode/decode calls dispatch through `custom`.
///
/// Returns [`CustomCoderError::AlreadyRegistered`] if a custom coder is already
/// registered for `tag`, or [`CustomCoderError::InvalidTag`] if `tag` is not a
/// custom-registrable row.
///
/// # Safety
///
/// The caller must ensure `instance` points to a valid, correctly-typed coder
/// instance that outlives the registration (e.g. a `Box::leak`ed value), and
/// that `vtable`'s thunks are safe to call with `instance` as their first
/// argument.
pub unsafe fn register_custom_coder(
    tag: u16,
    instance: *mut (),
    vtable: &'static CustomCoderVtable,
) -> Result<(), CustomCoderError> {
    if !CUSTOM_TAGS.contains(&tag) {
        log_debug!(
            "wredit: register_custom_coder rejected tag 0x{:04x}: not a custom row",
            tag
        );
        return Err(CustomCoderError::InvalidTag);
    }
    let entry = entry_for_tag(tag);
    if !entry.custom.load(Ordering::Acquire).is_null() {
        log_debug!(
            "wredit: register_custom_coder tag 0x{:04x}: already registered",
            tag
        );
        return Err(CustomCoderError::AlreadyRegistered);
    }
    let Some(slot) = tag_to_slot(tag) else {
        log_debug!(
            "wredit: register_custom_coder tag 0x{:04x}: not a custom row",
            tag
        );
        return Err(CustomCoderError::InvalidTag);
    };
    let mut guard = registry().lock();
    if guard[slot].is_some() {
        log_debug!(
            "wredit: register_custom_coder tag 0x{:04x}: already registered",
            tag
        );
        return Err(CustomCoderError::AlreadyRegistered);
    }
    guard[slot] = Some(vtable);
    drop(guard);
    // Publish the instance pointer last (release) so the vtable is visible
    // before any reader observes a non-null `custom`.
    entry.custom.store(instance, Ordering::Release);
    log_info!("wredit: registered custom coder for tag 0x{:04x}", tag);
    Ok(())
}

/// Unregister the custom coder for `tag`, restoring the built-in no-op thunks.
///
/// Returns [`CustomCoderError::InvalidTag`] if `tag` is not a custom row, or
/// if no custom coder is currently registered for `tag`.
///
/// # Safety
///
/// After this returns, `DispatchEntry.custom` for `tag` is null. The caller
/// must ensure no thread is concurrently dispatching through the old instance
/// and is responsible for reclaiming the leaked instance memory.
pub unsafe fn unregister_custom_coder(tag: u16) -> Result<(), CustomCoderError> {
    if !CUSTOM_TAGS.contains(&tag) {
        return Err(CustomCoderError::InvalidTag);
    }
    let entry = entry_for_tag(tag);
    if entry.custom.load(Ordering::Acquire).is_null() {
        return Err(CustomCoderError::InvalidTag);
    }
    let Some(slot) = tag_to_slot(tag) else {
        return Err(CustomCoderError::InvalidTag);
    };
    registry().lock()[slot] = None;
    entry
        .custom
        .store(core::ptr::null_mut(), Ordering::Release);
    log_debug!("wredit: unregistered custom coder for tag 0x{:04x}", tag);
    Ok(())
}

/// Look up the registered vtable for `tag`, if any.
fn custom_vtable_for(tag: u16) -> Option<&'static CustomCoderVtable> {
    let slot = tag_to_slot(tag)?;
    registry().lock()[slot]
}

/// Dispatch through the custom instance registered for `tag`, falling back to
/// the built-in no-op when `custom` is null. Called by the fallback-row thunks
/// for the matching operation.
///
/// # Safety
///
/// The caller must ensure `instance` (the `custom` pointer) is valid for the
/// duration of the call and matches the vtable registered for `tag`.
#[inline]
unsafe fn custom_encode_utf8(
    tag: u16,
    instance: *mut (),
    encoder: &mut Encoder,
    src: &str,
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    let Some(vtable) = custom_vtable_for(tag) else {
        return (EncoderResult::InputEmpty, 0, 0);
    };
    unsafe { (vtable.encode_utf8)(instance, encoder, src, dst, last) }
}

#[inline]
unsafe fn custom_encode_utf16(
    tag: u16,
    instance: *mut (),
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    let Some(vtable) = custom_vtable_for(tag) else {
        return (EncoderResult::InputEmpty, 0, 0);
    };
    unsafe { (vtable.encode_utf16)(instance, encoder, src, dst, last) }
}

#[inline]
unsafe fn custom_decode_utf8(
    tag: u16,
    instance: *mut (),
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u8],
    last: bool,
) -> (DecoderResult, usize, usize) {
    let Some(vtable) = custom_vtable_for(tag) else {
        return (DecoderResult::InputEmpty, 0, 0);
    };
    unsafe { (vtable.decode_utf8)(instance, decoder, src, dst, last) }
}

#[inline]
unsafe fn custom_decode_utf16(
    tag: u16,
    instance: *mut (),
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize) {
    let Some(vtable) = custom_vtable_for(tag) else {
        return (DecoderResult::InputEmpty, 0, 0);
    };
    unsafe { (vtable.decode_utf16)(instance, decoder, src, dst, last) }
}

/// UTF-8 thunks (delegate to `Encoder`/`Decoder` public methods).
#[inline]
unsafe fn encode_utf8_utf8(
    encoder: &mut Encoder,
    src: &str,
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    encoder.encode_from_utf8_without_replacement(src, dst, last)
}
#[inline]
unsafe fn encode_utf8_utf16(
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    encoder.encode_from_utf16_without_replacement(src, dst, last)
}
#[inline]
unsafe fn decode_utf8_utf8(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u8],
    last: bool,
) -> (DecoderResult, usize, usize) {
    decoder.decode_to_utf8_without_replacement(src, dst, last)
}
#[inline]
unsafe fn decode_utf8_utf16(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize) {
    decoder.decode_to_utf16_without_replacement(src, dst, last)
}

/// UTF-16LE thunks.
#[inline]
unsafe fn encode_utf16le_utf8(
    encoder: &mut Encoder,
    src: &str,
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    encoder.encode_from_utf8_without_replacement(src, dst, last)
}
#[inline]
unsafe fn encode_utf16le_utf16(
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    encoder.encode_from_utf16_without_replacement(src, dst, last)
}
#[inline]
unsafe fn decode_utf16le_utf8(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u8],
    last: bool,
) -> (DecoderResult, usize, usize) {
    decoder.decode_to_utf8_without_replacement(src, dst, last)
}
#[inline]
unsafe fn decode_utf16le_utf16(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize) {
    decoder.decode_to_utf16_without_replacement(src, dst, last)
}

/// UTF-16BE thunks.
#[inline]
unsafe fn encode_utf16be_utf8(
    encoder: &mut Encoder,
    src: &str,
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    encoder.encode_from_utf8_without_replacement(src, dst, last)
}
#[inline]
unsafe fn encode_utf16be_utf16(
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    encoder.encode_from_utf16_without_replacement(src, dst, last)
}
#[inline]
unsafe fn decode_utf16be_utf8(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u8],
    last: bool,
) -> (DecoderResult, usize, usize) {
    decoder.decode_to_utf8_without_replacement(src, dst, last)
}
#[inline]
unsafe fn decode_utf16be_utf16(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize) {
    decoder.decode_to_utf16_without_replacement(src, dst, last)
}

/// Unknown / fallback thunks. When `custom` is non-null they dispatch through
/// the registered custom coder for `ENC_UNKNOWN`; otherwise they return
/// `InputEmpty`, preserving the previous no-op behavior.
#[inline]
unsafe fn encode_unknown_utf8(
    encoder: &mut Encoder,
    src: &str,
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_UNKNOWN);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (EncoderResult::InputEmpty, 0, 0);
        }
        custom_encode_utf8(ENC_UNKNOWN, instance, encoder, src, dst, last)
    }
}
#[inline]
unsafe fn encode_unknown_utf16(
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_UNKNOWN);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (EncoderResult::InputEmpty, 0, 0);
        }
        custom_encode_utf16(ENC_UNKNOWN, instance, encoder, src, dst, last)
    }
}
#[inline]
unsafe fn decode_unknown_utf8(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u8],
    last: bool,
) -> (DecoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_UNKNOWN);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (DecoderResult::InputEmpty, 0, 0);
        }
        custom_decode_utf8(ENC_UNKNOWN, instance, decoder, src, dst, last)
    }
}
#[inline]
unsafe fn decode_unknown_utf16(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_UNKNOWN);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (DecoderResult::InputEmpty, 0, 0);
        }
        custom_decode_utf16(ENC_UNKNOWN, instance, decoder, src, dst, last)
    }
}

/// Single-byte (windows-1252 default) thunks — placeholder no-op.
#[inline]
unsafe fn encode_single_byte_utf8(
    encoder: &mut Encoder,
    src: &str,
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_SINGLE_BYTE);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (EncoderResult::InputEmpty, 0, 0);
        }
        custom_encode_utf8(ENC_SINGLE_BYTE, instance, encoder, src, dst, last)
    }
}
#[inline]
unsafe fn encode_single_byte_utf16(
    encoder: &mut Encoder,
    src: &[u16],
    dst: &mut [u8],
    last: bool,
) -> (EncoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_SINGLE_BYTE);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (EncoderResult::InputEmpty, 0, 0);
        }
        custom_encode_utf16(ENC_SINGLE_BYTE, instance, encoder, src, dst, last)
    }
}
#[inline]
unsafe fn decode_single_byte_utf8(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u8],
    last: bool,
) -> (DecoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_SINGLE_BYTE);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (DecoderResult::InputEmpty, 0, 0);
        }
        custom_decode_utf8(ENC_SINGLE_BYTE, instance, decoder, src, dst, last)
    }
}
#[inline]
unsafe fn decode_single_byte_utf16(
    decoder: &mut Decoder,
    src: &[u8],
    dst: &mut [u16],
    last: bool,
) -> (DecoderResult, usize, usize) {
    unsafe {
        let entry = entry_for_tag(ENC_SINGLE_BYTE);
        let instance = entry.custom.load(Ordering::Acquire);
        if instance.is_null() {
            return (DecoderResult::InputEmpty, 0, 0);
        }
        custom_decode_utf16(ENC_SINGLE_BYTE, instance, decoder, src, dst, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_textlike_encode::encoding::VariantEncoder;
    use codevar_textlike_encode::utf8::Utf8Encoder;

    /// Serializes tests that share the mutable static `ENC_UNKNOWN` custom
    /// dispatch slot, since `cargo test` runs tests in parallel by default.
    static CUSTOM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn entry_for_tag_clamps_out_of_range() {
        // In-range tags resolve to their own row.
        assert_eq!(entry_for_tag(ENC_UTF8).tag, ENC_UTF8);
        assert_eq!(entry_for_tag(ENC_UTF16LE).tag, ENC_UTF16LE);
        assert_eq!(entry_for_tag(ENC_UTF16BE).tag, ENC_UTF16BE);
        assert_eq!(entry_for_tag(ENC_SINGLE_BYTE).tag, ENC_SINGLE_BYTE);
        // Out-of-range clamps to the fallback row.
        assert_eq!(entry_for_tag(0xFFFF).tag, ENC_SINGLE_BYTE);
        assert_eq!(entry_for_tag(5).tag, ENC_SINGLE_BYTE);
    }

    #[test]
    fn prefetch_encoder_entry_does_not_panic_for_all_tags() {
        // Every tag in range plus out-of-range tags must not panic.
        for tag in 0..=ENC_SINGLE_BYTE {
            prefetch_encoder_entry(tag);
        }
        prefetch_encoder_entry(0xFFFF);
        prefetch_encoder_entry(5);
    }

    #[test]
    fn prefetch_decoder_entry_does_not_panic_for_all_tags() {
        // Every tag in range plus out-of-range tags must not panic.
        for tag in 0..=ENC_SINGLE_BYTE {
            prefetch_decoder_entry(tag);
        }
        prefetch_decoder_entry(0xFFFF);
        prefetch_decoder_entry(5);
    }

    #[test]
    fn prefetch_entry_calls_both_directions_without_panic() {
        for tag in 0..=ENC_SINGLE_BYTE {
            prefetch_entry(tag);
        }
        prefetch_entry(0xFFFF);
        prefetch_entry(5);
    }

    #[test]
    fn dispatch_entry_is_one_cache_line() {
        assert_eq!(core::mem::size_of::<DispatchEntry>(), CACHE_LINE_SIZE);
        assert_eq!(core::mem::align_of::<DispatchEntry>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn utf8_entry_points_to_utf8_static() {
        assert!(core::ptr::eq(entry_for_tag(ENC_UTF8).encoding, &UTF_8));
    }

    #[test]
    fn unknown_entry_has_null_custom() {
        assert!(
            entry_for_tag(ENC_UNKNOWN)
                .custom
                .load(Ordering::Acquire)
                .is_null()
        );
    }

    /// A tiny custom coder instance used by the tests. It records how many
    /// times each thunk was invoked and echoes input length as output.
    struct TestCoder {
        encode_calls: usize,
        decode_calls: usize,
    }

    impl TestCoder {
        fn new() -> Self {
            Self {
                encode_calls: 0,
                decode_calls: 0,
            }
        }
    }

    unsafe fn test_encode_utf8(
        instance: *mut (),
        _encoder: &mut Encoder,
        src: &str,
        _dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let coder = unsafe { &mut *(instance as *mut TestCoder) };
        coder.encode_calls += 1;
        (EncoderResult::InputEmpty, src.len(), src.len())
    }

    unsafe fn test_encode_utf16(
        instance: *mut (),
        _encoder: &mut Encoder,
        src: &[u16],
        _dst: &mut [u8],
        _last: bool,
    ) -> (EncoderResult, usize, usize) {
        let coder = unsafe { &mut *(instance as *mut TestCoder) };
        coder.encode_calls += 1;
        (EncoderResult::InputEmpty, src.len() * 2, src.len() * 2)
    }

    unsafe fn test_decode_utf8(
        instance: *mut (),
        _decoder: &mut Decoder,
        src: &[u8],
        _dst: &mut [u8],
        _last: bool,
    ) -> (DecoderResult, usize, usize) {
        let coder = unsafe { &mut *(instance as *mut TestCoder) };
        coder.decode_calls += 1;
        (DecoderResult::InputEmpty, src.len(), src.len())
    }

    unsafe fn test_decode_utf16(
        instance: *mut (),
        _decoder: &mut Decoder,
        src: &[u8],
        _dst: &mut [u16],
        _last: bool,
    ) -> (DecoderResult, usize, usize) {
        let coder = unsafe { &mut *(instance as *mut TestCoder) };
        coder.decode_calls += 1;
        (DecoderResult::InputEmpty, src.len() / 2, src.len() / 2)
    }

    static TEST_VTABLE: CustomCoderVtable = CustomCoderVtable {
        encode_utf8: test_encode_utf8,
        encode_utf16: test_encode_utf16,
        decode_utf8: test_decode_utf8,
        decode_utf16: test_decode_utf16,
    };

    #[test]
    fn custom_coder_register_and_dispatch_unknown() {
        // Serialize on the shared ENC_UNKNOWN custom slot (tests run in parallel).
        let _lock = CUSTOM_TEST_LOCK.lock().unwrap();
        let instance = Box::leak(Box::new(TestCoder::new())) as *mut TestCoder as *mut ();
        // Clear any leftover registration from another test so this test is
        // order-independent (tests share the static custom dispatch state).
        let _ = unsafe { unregister_custom_coder(ENC_UNKNOWN) };
        assert!(unsafe { register_custom_coder(ENC_UNKNOWN, instance, &TEST_VTABLE) }.is_ok());

        let mut enc = Encoder::new(&UTF_8, VariantEncoder::Utf8(Utf8Encoder));
        let mut dec = UTF_8.new_decoder_without_bom_handling();

        let (r, rd, wr) =
            unsafe { (entry_for_tag(ENC_UNKNOWN).encode_utf8)(&mut enc, "hi", &mut [0u8; 4], true) };
        assert_eq!(r, EncoderResult::InputEmpty);
        assert_eq!((rd, wr), (2, 2));

        let (r, rd, wr) =
            unsafe { (entry_for_tag(ENC_UNKNOWN).decode_utf8)(&mut dec, b"hi", &mut [0u8; 4], true) };
        assert_eq!(r, DecoderResult::InputEmpty);
        assert_eq!((rd, wr), (2, 2));

        let coder = unsafe { &*(instance as *const TestCoder) };
        assert_eq!(coder.encode_calls, 1);
        assert_eq!(coder.decode_calls, 1);

        assert!(unsafe { unregister_custom_coder(ENC_UNKNOWN) }.is_ok());
        assert!(
            entry_for_tag(ENC_UNKNOWN)
                .custom
                .load(Ordering::Acquire)
                .is_null()
        );
        // After unregister, falls back to no-op.
        let (r, _, _) =
            unsafe { (entry_for_tag(ENC_UNKNOWN).encode_utf8)(&mut enc, "hi", &mut [0u8; 4], true) };
        assert_eq!(r, EncoderResult::InputEmpty);
    }

    #[test]
    fn custom_coder_rejects_duplicate_and_invalid_tag() {
        // Serialize on the shared ENC_UNKNOWN custom slot (tests run in parallel).
        let _lock = CUSTOM_TEST_LOCK.lock().unwrap();
        // Clear any leftover registration from another test so this test is
        // order-independent (tests share the static custom dispatch state).
        let _ = unsafe { unregister_custom_coder(ENC_UNKNOWN) };
        let instance = Box::leak(Box::new(TestCoder::new())) as *mut TestCoder as *mut ();
        assert!(unsafe { register_custom_coder(ENC_UNKNOWN, instance, &TEST_VTABLE) }.is_ok());
        assert_eq!(
            unsafe { register_custom_coder(ENC_UNKNOWN, instance, &TEST_VTABLE) },
            Err(CustomCoderError::AlreadyRegistered)
        );
        assert_eq!(
            unsafe { register_custom_coder(ENC_UTF8, instance, &TEST_VTABLE) },
            Err(CustomCoderError::InvalidTag)
        );
        let _ = unsafe { unregister_custom_coder(ENC_UNKNOWN) };
    }

    #[test]
    fn custom_coder_register_single_byte_row() {
        let instance = Box::leak(Box::new(TestCoder::new())) as *mut TestCoder as *mut ();
        assert!(unsafe { register_custom_coder(ENC_SINGLE_BYTE, instance, &TEST_VTABLE) }.is_ok());

        let mut enc = Encoder::new(&UTF_8, VariantEncoder::Utf8(Utf8Encoder));
        let (r, _, _) =
            unsafe { (entry_for_tag(ENC_SINGLE_BYTE).encode_utf8)(&mut enc, "hi", &mut [0u8; 4], true) };
        assert_eq!(r, EncoderResult::InputEmpty);

        let coder = unsafe { &*(instance as *const TestCoder) };
        assert_eq!(coder.encode_calls, 1);

        let _ = unsafe { unregister_custom_coder(ENC_SINGLE_BYTE) };
    }
}
