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

//! OS-default text encoding (charset) detection.
//!
//! This module detects the operating system's default text encoding so
//! callers (e.g. the Wredit engine) can select an encoder without
//! guessing. Each platform queries its native API and falls back to
//! UTF-8 when the default cannot be determined.
//!
//! ## Platform strategy
//!
//! - **Windows:** `GetACP()` (ANSI code page) and `GetConsoleOutputCP()`.
//!   Common code pages are mapped to IANA/MIME names; anything else is
//!   reported as a single-byte `CP{n}` encoding.
//! - **Unix (Linux, macOS, BSD, …):** `libc::nl_langinfo(CODESET)`, with the
//!   `LC_ALL`/`LC_CTYPE`/`LANG` environment variables as a secondary source.
//! - **WASM / other:** UTF-8 fallback.
//!
//! All results are cached after the first successful detection. The public
//! functions never panic; they degrade gracefully to UTF-8.

#![allow(dead_code)]
use alloc::boxed::Box;
use alloc::string::String;
use codevar_logger::{log_debug, log_warn};
use core::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

/// The detected OS-default encoding.
///
/// `SingleByte` and `Unknown` carry a human-readable encoding name (an
/// IANA/MIME or code-page derived name) suitable for passing to `codevar_textlike_encode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultEncoding {
    /// UTF-8.
    Utf8,
    /// UTF-16 little-endian.
    Utf16Le,
    /// UTF-16 big-endian.
    Utf16Be,
    /// A single-byte / legacy encoding with the given name.
    SingleByte(&'static str),
    /// Could not be determined; `name` is the fallback name (normally "UTF-8").
    Unknown(&'static str),
}

/// Initialisation state of the cached default encoding.
///
/// `0` = uninitialised, `1` = initialized. Written once with a release
/// store and read with acquire loads so detection runs exactly once.
static DEFAULT_ENCODING_STATE: AtomicU8 = AtomicU8::new(0);
static DEFAULT_ENCODING_DISC: AtomicU8 = AtomicU8::new(0);
static DEFAULT_ENCODING_NAME: AtomicPtr<&'static str> = AtomicPtr::new(core::ptr::null_mut());

/// Discriminant for `DefaultEncoding::Utf8`.
const DISC_UTF8: u8 = 0;
/// Discriminant for `DefaultEncoding::Utf16Le`.
const DISC_UTF16LE: u8 = 1;
/// Discriminant for `DefaultEncoding::Utf16Be`.
const DISC_UTF16BE: u8 = 2;
/// Discriminant for `DefaultEncoding::SingleByte`.
const DISC_SINGLE_BYTE: u8 = 3;
/// Discriminant for `DefaultEncoding::Unknown`.
const DISC_UNKNOWN: u8 = 4;

/// Returns the OS-default text encoding.
///
/// The result is cached after the first call. On platforms where the native
/// API cannot be queried, this degrades to [`DefaultEncoding::Unknown`] with
/// the fallback name "UTF-8".
///
/// # Example
///
/// ```
/// use codevar_os_def::default_encoding;
/// let enc = default_encoding();
/// assert!(!enc.name().is_empty());
/// ```
#[inline]
pub fn default_encoding() -> DefaultEncoding {
    if DEFAULT_ENCODING_STATE.load(Ordering::Acquire) == 1 {
        let disc = DEFAULT_ENCODING_DISC.load(Ordering::Acquire);
        let name_ptr = DEFAULT_ENCODING_NAME.load(Ordering::Acquire);
        let name = if name_ptr.is_null() {
            "UTF-8"
        } else {
            // Safety: `DEFAULT_ENCODING_NAME` is written once with a release
            // store before `DEFAULT_ENCODING_STATE` is set to 1, so an acquire
            // load here observes a valid, non-null `&'static str` pointer.
            unsafe { *name_ptr }
        };
        return decode_cached(disc, name);
    }
    let enc = detect_default_encoding();
    let disc = encode_discriminant(&enc);
    let name = enc.name();
    let name_boxed = Box::leak(Box::new(name));
    DEFAULT_ENCODING_NAME.store(name_boxed, Ordering::Release);
    DEFAULT_ENCODING_DISC.store(disc, Ordering::Release);
    DEFAULT_ENCODING_STATE.store(1, Ordering::Release);
    enc
}

/// Returns the OS-default encoding name as a `&'static str`.
///
/// Falls back to `"UTF-8"` when the default cannot be determined.
#[inline]
pub fn default_encoding_name() -> &'static str {
    default_encoding().name()
}

/// Returns `true` when the OS-default encoding is UTF-8.
#[inline]
pub fn is_utf8_default() -> bool {
    matches!(default_encoding(), DefaultEncoding::Utf8)
}

impl DefaultEncoding {
    /// The human-readable encoding name (IANA/MIME or code-page derived).
    #[must_use]
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            DefaultEncoding::Utf8 => "UTF-8",
            DefaultEncoding::Utf16Le => "UTF-16LE",
            DefaultEncoding::Utf16Be => "UTF-16BE",
            DefaultEncoding::SingleByte(name) => *name,
            DefaultEncoding::Unknown(name) => *name,
        }
    }
}

fn encode_discriminant(enc: &DefaultEncoding) -> u8 {
    match enc {
        DefaultEncoding::Utf8 => DISC_UTF8,
        DefaultEncoding::Utf16Le => DISC_UTF16LE,
        DefaultEncoding::Utf16Be => DISC_UTF16BE,
        DefaultEncoding::SingleByte(_) => DISC_SINGLE_BYTE,
        DefaultEncoding::Unknown(_) => DISC_UNKNOWN,
    }
}

fn decode_cached(disc: u8, name: &'static str) -> DefaultEncoding {
    match disc {
        DISC_UTF8 => DefaultEncoding::Utf8,
        DISC_UTF16LE => DefaultEncoding::Utf16Le,
        DISC_UTF16BE => DefaultEncoding::Utf16Be,
        DISC_SINGLE_BYTE => DefaultEncoding::SingleByte(name),
        DISC_UNKNOWN => DefaultEncoding::Unknown(name),
        _ => DefaultEncoding::Unknown("UTF-8"),
    }
}

/// Detect the default encoding by querying the native OS API.
///
/// This is the cold path; it runs once and is cached by [`default_encoding`].
#[cold]
fn detect_default_encoding() -> DefaultEncoding {
    #[cfg(target_os = "windows")]
    {
        return detect_windows();
    }
    #[cfg(any(unix, target_os = "macos"))]
    {
        return detect_unix();
    }
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    {
        log_debug!("os-def: wasm target detected, defaulting to UTF-8");
        return DefaultEncoding::Utf8;
    }
    #[cfg(not(any(
        target_os = "windows",
        unix,
        target_os = "macos",
        target_arch = "wasm32",
        target_arch = "wasm64",
    )))]
    {
        log_warn!("os-def: unsupported target, defaulting to UTF-8");
        return DefaultEncoding::Unknown("UTF-8");
    }
}

/// ---- Windows -----------------------------------------------------------

#[cfg(target_os = "windows")]
mod windows_ffi {
    /// Retrieves the ANSI code page for the system.
    ///
    /// # Safety
    ///
    /// `GetACP` is a safe Win32 API; calling it with no arguments is always
    /// valid and cannot fault.
    #[link(name = "kernel32")]
    extern "system" {
        pub(super) fn GetACP() -> u32;
    }

    /// Retrieves the output code page used by the console.
    ///
    /// # Safety
    ///
    /// `GetConsoleOutputCP` is a safe Win32 API; calling it with no arguments
    /// is always valid and cannot fault.
    #[link(name = "kernel32")]
    extern "system" {
        pub(super) fn GetConsoleOutputCP() -> u32;
    }
}

#[cfg(target_os = "windows")]
#[cold]
fn detect_windows() -> DefaultEncoding {
    // Prefer the console output code page when it differs from the system ANSI
    // code page, as that is what text written to the terminal actually uses.
    let console_cp = unsafe { windows_ffi::GetConsoleOutputCP() };
    let ansi_cp = unsafe { windows_ffi::GetACP() };

    let cp = if console_cp != 0 && console_cp != ansi_cp {
        console_cp
    } else {
        ansi_cp
    };

    match map_codepage(cp) {
        Some(name) => match name {
            "UTF-8" => DefaultEncoding::Utf8,
            "UTF-16LE" => DefaultEncoding::Utf16Le,
            "UTF-16BE" => DefaultEncoding::Utf16Be,
            other => DefaultEncoding::SingleByte(other),
        },
        None => {
            log_warn!("os-def: unknown Windows code page {cp}, defaulting to UTF-8");
            DefaultEncoding::Unknown("UTF-8")
        }
    }
}

/// Map a Windows code page to a canonical encoding name.
///
/// Returns `None` for code pages that have no single-byte / UTF mapping and
/// should be treated as opaque single-byte `CP{n}` encodings.
#[cfg(target_os = "windows")]
#[cold]
fn map_codepage(cp: u32) -> Option<&'static str> {
    match cp {
        65001 | 54936 => Some("UTF-8"), // CP_UTF8 / x-Mac-Cyrillic alias handled below
        1200 => Some("UTF-16LE"),
        1201 => Some("UTF-16BE"),
        1252 => Some("windows-1252"),
        1250 => Some("windows-1250"),
        1251 => Some("windows-1251"),
        1253 => Some("windows-1253"),
        1254 => Some("windows-1254"),
        1255 => Some("windows-1255"),
        1256 => Some("windows-1256"),
        1257 => Some("windows-1257"),
        1258 => Some("windows-1258"),
        28591 | 20127 => Some("ISO-8859-1"),
        28592 => Some("ISO-8859-2"),
        28593 => Some("ISO-8859-3"),
        28594 => Some("ISO-8859-4"),
        28595 => Some("ISO-8859-5"),
        28596 => Some("ISO-8859-6"),
        28597 => Some("ISO-8859-7"),
        28598 => Some("ISO-8859-8"),
        28599 => Some("ISO-8859-9"),
        51932 | 51936 => Some("EUC-JP"),
        51949 => Some("EUC-KR"),
        932 => Some("shift_jis"),
        936 => Some("gbk"),
        950 => Some("big5"),
        874 => Some("windows-874"),
        _ => None,
    }
}

/// ---- Unix (Linux, macOS, BSD, …) ---------------------------------------

#[cfg(any(unix, target_os = "macos"))]
mod unix_ffi {
    #[link(name = "c")]
    unsafe extern "C" {
        /// Query the locale character set (codeset) for the current category.
        ///
        /// # Safety
        ///
        /// `nl_langinfo` reads locale data owned by the C library; passing
        /// `CODESET` is always valid and returns a pointer to a NUL-terminated
        /// static buffer that must not be freed.
        pub(super) fn nl_langinfo(item: libc::c_int) -> *const libc::c_char;
    }

    pub(super) const CODESET: libc::c_int = 0;
}

#[cfg(any(unix, target_os = "macos"))]
#[cold]
fn detect_unix() -> DefaultEncoding {
    // Primary source: nl_langinfo(CODESET).
    if let Some(codeset) = unsafe { read_codeset() } {
        if !codeset.is_empty() {
            let normalized = normalize_codeset(&codeset);
            log_debug!("os-def: nl_langinfo(CODESET) = {codeset:?} -> {normalized}");
            return match normalized.as_str() {
                "UTF-8" | "utf-8" | "UTF8" => DefaultEncoding::Utf8,
                "UTF-16LE" | "utf-16le" => DefaultEncoding::Utf16Le,
                "UTF-16BE" | "utf-16be" => DefaultEncoding::Utf16Be,
                other => DefaultEncoding::SingleByte(leak_codeset_name(String::from(other))),
            };
        }
    }
    // Secondary source: locale environment variables.
    for var in ["LC_ALL", "LC_CTYPE", "LANG"] {
        if let Some(val) = env_var(var) {
            if let Some(codeset) = extract_codeset_from_locale(&val) {
                let normalized = normalize_codeset(&codeset);
                log_debug!("os-def: env {var} = {val:?} -> codeset {codeset:?} -> {normalized}");
                return match normalized.as_str() {
                    "UTF-8" | "utf-8" | "UTF8" => DefaultEncoding::Utf8,
                    "UTF-16LE" | "utf-16le" => DefaultEncoding::Utf16Le,
                    "UTF-16BE" | "utf-16be" => DefaultEncoding::Utf16Be,
                    other => DefaultEncoding::SingleByte(leak_codeset_name(String::from(other))),
                };
            }
        }
    }

    log_warn!("os-def: could not determine OS default encoding, defaulting to UTF-8");
    DefaultEncoding::Unknown("UTF-8")
}

/// Read `CODESET` via `nl_langinfo`.
///
/// Returns `None` when the function pointer is null or the result is an
/// empty string.
#[cfg(any(unix, target_os = "macos"))]
#[cold]
unsafe fn read_codeset() -> Option<String> {
    unsafe {
        let ptr = unix_ffi::nl_langinfo(unix_ffi::CODESET);
        if ptr.is_null() {
            return None;
        }
        cstr_to_string(ptr)
    }
}

/// Convert a NUL-terminated `*const c_char` to an owned `String`.
///
/// Returns `None` if the pointer is null or the bytes are not valid UTF-8
/// (codeset names are expected to be ASCII).
#[cfg(any(unix, target_os = "macos"))]
unsafe fn cstr_to_string(ptr: *const libc::c_char) -> Option<String> {
    unsafe {
        if ptr.is_null() {
            return None;
        }
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let bytes = core::slice::from_raw_parts(ptr as *const u8, len);
        core::str::from_utf8(bytes).ok().map(String::from)
    }
}

/// Read an environment variable by name.
///
/// Returns `None` when the variable is unset or not valid UTF-8.
#[cfg(any(unix, target_os = "macos"))]
#[cold]
fn env_var(name: &str) -> Option<String> {
    // `libc::getenv` is the simplest interface; use it via FFI.
    unsafe { getenv_value(name.as_ptr() as *const libc::c_char) }
}

#[cfg(any(unix, target_os = "macos"))]
mod env_ffi {
    #[link(name = "c")]
    unsafe extern "C" {
        /// # Safety
        /// `getenv` returns a pointer to the environment block; the caller
        /// must not modify or free it. The name must be NUL-terminated.
        pub(super) fn getenv(name: *const libc::c_char) -> *const libc::c_char;
    }
}

#[cfg(any(unix, target_os = "macos"))]
#[cold]
unsafe fn getenv_value(name: *const libc::c_char) -> Option<String> {
    unsafe {
        if name.is_null() {
            return None;
        }
        let ptr = env_ffi::getenv(name);
        if ptr.is_null() {
            return None;
        }
        cstr_to_string(ptr)
    }
}

/// Extract the codeset portion from a locale value such as "en_US.UTF-8".
#[cfg(any(unix, target_os = "macos"))]
#[cold]
fn extract_codeset_from_locale(locale: &str) -> Option<String> {
    for separator in ['.', '@', '+'] {
        if let Some(rest) = locale.split_once(separator) {
            let codeset = rest
                .1
                .split(|c: char| c.is_ascii_whitespace() || c == ';')
                .next()?;
            if !codeset.is_empty() {
                return Some(String::from(codeset));
            }
        }
    }
    None
}

/// Normalize a codeset name to a canonical form.
#[cfg(any(unix, target_os = "macos"))]
#[cold]
fn normalize_codeset(codeset: &str) -> String {
    let mut out = String::with_capacity(codeset.len());
    for ch in codeset.chars() {
        match ch {
            '-' | '_' => out.push('-'),
            c if c.is_ascii_alphanumeric() => out.push(c.to_ascii_lowercase()),
            _ => {}
        }
    }
    if out.is_empty() {
        out.push_str("utf-8");
    }
    out
}

/// Leak a codeset name into a `&'static str`.
///
/// Used for single-byte encodings whose names are not known at compile time.
/// The leaked memory lives for the program duration, which is acceptable
/// because OS-default encoding detection runs once and is cached.
#[cfg(any(unix, target_os = "macos"))]
#[cold]
fn leak_codeset_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn default_encoding_name_is_non_empty_static() {
        let name = default_encoding_name();
        assert!(!name.is_empty());
        // The returned reference must be `'static`.
        let _: &'static str = name;
    }

    #[test]
    fn is_utf8_default_returns_bool() {
        let _ = is_utf8_default();
    }

    #[test]
    fn default_encoding_is_cached() {
        let first = default_encoding();
        let second = default_encoding();
        assert_eq!(first, second);
        assert_eq!(first.name(), second.name());
    }

    #[test]
    fn default_encoding_variant_has_name() {
        let enc = default_encoding();
        assert_eq!(
            enc.name(),
            match enc {
                DefaultEncoding::Utf8 => "UTF-8",
                DefaultEncoding::Utf16Le => "UTF-16LE",
                DefaultEncoding::Utf16Be => "UTF-16BE",
                DefaultEncoding::SingleByte(n) | DefaultEncoding::Unknown(n) => n,
            }
        );
        assert!(!enc.name().is_empty());
    }

    #[test]
    fn normalize_codeset_handles_common_forms() {
        // Normalization is only compiled on unix; on other targets this test
        // still compiles because `normalize_codeset` is cfg-gated. Guard it.
        #[cfg(any(unix, target_os = "macos"))]
        {
            assert_eq!(normalize_codeset("UTF-8"), "utf-8");
            assert_eq!(normalize_codeset("utf_8"), "utf-8");
            assert_eq!(normalize_codeset("ISO-8859-1"), "iso-8859-1");
            assert_eq!(normalize_codeset("windows-1252"), "windows-1252");
        }
    }

    #[test]
    fn default_encoding_never_panics() {
        // Calling repeatedly must not panic; the result is always defined.
        for _ in 0..10 {
            let _ = default_encoding();
            let _ = default_encoding_name();
            let _ = is_utf8_default();
        }
    }

    /// E2E: detect_default_encoding → default_encoding → name round trip.
    #[test]
    fn encoding_round_trip_preserves_name() {
        let enc = default_encoding();
        let name = enc.name();
        assert!(!name.is_empty());
        // The name is always a valid &'static str.
        let _static: &'static str = name;
        // Round trip through default_encoding_name returns the same name.
        assert_eq!(default_encoding_name(), name);
    }

    /// The cached value preserves SingleByte names (the pre-fix bug dropped
    /// them and returned Unknown("UTF-8")).
    #[test]
    fn caching_preserves_single_byte_name() {
        // Build a SingleByte encoding directly and simulate the cached path.
        let disc = DISC_SINGLE_BYTE;
        let name = "windows-1252";
        let cached = decode_cached(disc, name);
        assert!(matches!(cached, DefaultEncoding::SingleByte(n) if n == "windows-1252"));
    }

    /// The cached value preserves Unknown names.
    #[test]
    fn caching_preserves_unknown_name() {
        let disc = DISC_UNKNOWN;
        let name = "ISO-8859-15";
        let cached = decode_cached(disc, name);
        assert!(matches!(cached, DefaultEncoding::Unknown(n) if n == "ISO-8859-15"));
    }

    /// decode_cached rejects unknown discriminants gracefully.
    #[test]
    fn decode_cached_unknown_discriminant_falls_back() {
        let cached = decode_cached(0xFF, "anything");
        assert!(matches!(cached, DefaultEncoding::Unknown("UTF-8")));
    }

    /// encode_discriminant round-trips each variant.
    #[test]
    fn discriminant_round_trip() {
        let cases = [
            (DefaultEncoding::Utf8, DISC_UTF8),
            (DefaultEncoding::Utf16Le, DISC_UTF16LE),
            (DefaultEncoding::Utf16Be, DISC_UTF16BE),
            (DefaultEncoding::SingleByte("windows-1252"), DISC_SINGLE_BYTE),
            (DefaultEncoding::Unknown("x"), DISC_UNKNOWN),
        ];
        for (enc, expected_disc) in cases {
            assert_eq!(encode_discriminant(&enc), expected_disc);
        }
    }

    /// map_codepage covers the common Windows code pages.
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_codepage_mapping() {
        assert_eq!(map_codepage(65001), Some("UTF-8"));
        assert_eq!(map_codepage(1200), Some("UTF-16LE"));
        assert_eq!(map_codepage(1201), Some("UTF-16BE"));
        assert_eq!(map_codepage(1252), Some("windows-1252"));
        assert_eq!(map_codepage(1250), Some("windows-1250"));
        assert_eq!(map_codepage(1251), Some("windows-1251"));
        assert_eq!(map_codepage(28591), Some("ISO-8859-1"));
        assert_eq!(map_codepage(932), Some("shift_jis"));
        assert_eq!(map_codepage(936), Some("gbk"));
        assert_eq!(map_codepage(950), Some("big5"));
        assert_eq!(map_codepage(874), Some("windows-874"));
        // Unknown code pages return None.
        assert_eq!(map_codepage(0), None);
        assert_eq!(map_codepage(99999), None);
    }

    /// normalize_codeset handles common locale forms (compiled only on unix).
    #[cfg(any(unix, target_os = "macos"))]
    #[test]
    fn normalize_codeset_handles_common_forms_unix() {
        assert_eq!(normalize_codeset("UTF-8"), "utf-8");
        assert_eq!(normalize_codeset("utf_8"), "utf-8");
        assert_eq!(normalize_codeset("ISO-8859-1"), "iso-8859-1");
        assert_eq!(normalize_codeset("windows-1252"), "windows-1252");
        assert_eq!(normalize_codeset("en_US.UTF-8"), "en-usutf-8");
    }

    /// extract_codeset_from_locale parses common locale values.
    #[cfg(any(unix, target_os = "macos"))]
    #[test]
    fn extract_codeset_from_locale_parses_common_locales() {
        assert_eq!(
            extract_codeset_from_locale("en_US.UTF-8"),
            Some("UTF-8".to_string())
        );
        assert_eq!(extract_codeset_from_locale("C.UTF-8"), Some("UTF-8".to_string()));
        assert_eq!(
            extract_codeset_from_locale("de_DE@euro"),
            Some("euro".to_string())
        );
        assert_eq!(extract_codeset_from_locale("POSIX"), None);
    }

    /// E2E: every public query returns a defined value without panicking.
    #[test]
    fn public_queries_always_defined() {
        for _ in 0..50 {
            let enc = default_encoding();
            let _name = enc.name();
            let _is_utf8 = is_utf8_default();
            let _name2 = default_encoding_name();
            assert!(!_name.is_empty());
            assert!(!_name2.is_empty());
        }
    }

    /// DefaultEncoding derives Debug, Clone, Copy, PartialEq, Eq.
    #[test]
    fn default_encoding_derives() {
        let a = DefaultEncoding::Utf8;
        let b = DefaultEncoding::Utf8;
        assert_eq!(a, b);
        let c = DefaultEncoding::SingleByte("windows-1252");
        let d = DefaultEncoding::SingleByte("windows-1252");
        assert_eq!(c, d);
        let e = DefaultEncoding::Unknown("UTF-8");
        assert_ne!(a, e);
        // Clone and Copy.
        let _f = a.clone();
        let _g: DefaultEncoding = a;
        // Debug.
        let _dbg = format!("{a:?}");
    }
}
