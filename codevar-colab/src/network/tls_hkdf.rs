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

//! TLS 1.3 HKDF helpers (RFC 8446 §7.1, RFC 5869).

use crate::network::tls_codec::put_u16;
use crate::network::tls_crypto_hash::hash_empty;
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_ids::HashAlgorithm;
use hkdf::Hkdf;
use sha2::{Sha256, Sha384};

/// HKDF algorithm selected by the cipher suite hash.
#[derive(Clone, Copy)]
pub enum HkdfAlg {
    /// HKDF-SHA256.
    Sha256,
    /// HKDF-SHA384.
    Sha384,
}

impl From<HashAlgorithm> for HkdfAlg {
    fn from(h: HashAlgorithm) -> Self {
        match h {
            HashAlgorithm::Sha256 => Self::Sha256,
            HashAlgorithm::Sha384 => Self::Sha384,
        }
    }
}

/// HKDF-Extract(`salt`, `ikm`).
pub fn hkdf_extract(alg: HashAlgorithm, salt: &[u8], ikm: &[u8]) -> Vec<u8> {
    match alg {
        HashAlgorithm::Sha256 => {
            let (prk, _) = Hkdf::<Sha256>::extract(Some(salt), ikm);
            prk.to_vec()
        }
        HashAlgorithm::Sha384 => {
            let (prk, _) = Hkdf::<Sha384>::extract(Some(salt), ikm);
            prk.to_vec()
        }
    }
}

/// HKDF-Expand-Label as defined in RFC 8446 §7.1:
///
/// `HkdfLabel = length (u16) || "tls13 " + Label || Context`
pub fn hkdf_expand_label(
    alg: HashAlgorithm,
    secret: &[u8],
    label: &[u8],
    context: &[u8],
    length: usize,
) -> TlsResult<Vec<u8>> {
    if length > 65535 {
        return Err(TlsError::Internal("HKDF expand length too large".into()));
    }
    if context.len() > 255 {
        return Err(TlsError::Internal("HKDF context too long".into()));
    }
    let mut hkdf_label = Vec::new();
    put_u16(&mut hkdf_label, length as u16);
    let mut full_label = Vec::with_capacity(6 + label.len());
    full_label.extend_from_slice(b"tls13 ");
    full_label.extend_from_slice(label);
    if full_label.len() > 255 {
        return Err(TlsError::Internal("HKDF label too long".into()));
    }
    hkdf_label.push(full_label.len() as u8);
    hkdf_label.extend_from_slice(&full_label);
    hkdf_label.push(context.len() as u8);
    hkdf_label.extend_from_slice(context);

    let mut out = vec![0u8; length];
    match alg {
        HashAlgorithm::Sha256 => {
            let hk = Hkdf::<Sha256>::from_prk(secret)
                .map_err(|_| TlsError::crypto("HKDF-SHA256 invalid PRK"))?;
            hk.expand(&hkdf_label, &mut out)
                .map_err(|_| TlsError::crypto("HKDF-SHA256 expand failed"))?;
        }
        HashAlgorithm::Sha384 => {
            let hk = Hkdf::<Sha384>::from_prk(secret)
                .map_err(|_| TlsError::crypto("HKDF-SHA384 invalid PRK"))?;
            hk.expand(&hkdf_label, &mut out)
                .map_err(|_| TlsError::crypto("HKDF-SHA384 expand failed"))?;
        }
    }
    Ok(out)
}

/// `Derive-Secret(Secret, Label, Messages)` — when `messages_hash` is `None`,
/// uses `Transcript-Hash("")`.
pub fn derive_secret(
    alg: HashAlgorithm,
    secret: &[u8],
    label: &[u8],
    messages_hash: Option<&[u8]>,
) -> TlsResult<Vec<u8>> {
    let empty = hash_empty(alg);
    let ctx = messages_hash.unwrap_or(&empty);
    hkdf_expand_label(alg, secret, label, ctx, alg.output_len())
}
