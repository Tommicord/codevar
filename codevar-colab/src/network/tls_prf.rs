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

//! TLS 1.2 PRF (RFC 5246 §5).

use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_ids::HashAlgorithm;
use hmac::{Hmac, Mac};
use sha2::{Sha256, Sha384};

type HmacSha256 = Hmac<Sha256>;
type HmacSha384 = Hmac<Sha384>;

/// TLS 1.2 `PRF(secret, label, seed)` producing `out_len` bytes.
///
/// SHA-256 suites use P_SHA256; SHA-384 suites use P_SHA384 (RFC 5246 §5,
/// RFC 5288 / RFC 5289).
pub fn tls12_prf(
    alg: HashAlgorithm,
    secret: &[u8],
    label: &[u8],
    seed: &[u8],
    out_len: usize,
) -> TlsResult<Vec<u8>> {
    let mut label_seed = Vec::with_capacity(label.len() + seed.len());
    label_seed.extend_from_slice(label);
    label_seed.extend_from_slice(seed);
    match alg {
        HashAlgorithm::Sha256 => p_hash_sha256(secret, &label_seed, out_len),
        HashAlgorithm::Sha384 => p_hash_sha384(secret, &label_seed, out_len),
    }
}

fn p_hash_sha256(secret: &[u8], seed: &[u8], out_len: usize) -> TlsResult<Vec<u8>> {
    let mut a = hmac_sha256(secret, seed)?;
    let mut out = Vec::with_capacity(out_len);
    while out.len() < out_len {
        let mut a_seed = Vec::with_capacity(a.len() + seed.len());
        a_seed.extend_from_slice(&a);
        a_seed.extend_from_slice(seed);
        let block = hmac_sha256(secret, &a_seed)?;
        let need = (out_len - out.len()).min(block.len());
        out.extend_from_slice(&block[..need]);
        a = hmac_sha256(secret, &a)?;
    }
    Ok(out)
}

fn p_hash_sha384(secret: &[u8], seed: &[u8], out_len: usize) -> TlsResult<Vec<u8>> {
    let mut a = hmac_sha384(secret, seed)?;
    let mut out = Vec::with_capacity(out_len);
    while out.len() < out_len {
        let mut a_seed = Vec::with_capacity(a.len() + seed.len());
        a_seed.extend_from_slice(&a);
        a_seed.extend_from_slice(seed);
        let block = hmac_sha384(secret, &a_seed)?;
        let need = (out_len - out.len()).min(block.len());
        out.extend_from_slice(&block[..need]);
        a = hmac_sha384(secret, &a)?;
    }
    Ok(out)
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> TlsResult<Vec<u8>> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|_| TlsError::crypto("HMAC-SHA256 key"))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn hmac_sha384(key: &[u8], data: &[u8]) -> TlsResult<Vec<u8>> {
    let mut mac = HmacSha384::new_from_slice(key)
        .map_err(|_| TlsError::crypto("HMAC-SHA384 key"))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

/// HMAC used for TLS 1.3 Finished / PSK binders.
pub fn hmac_hash(alg: HashAlgorithm, key: &[u8], data: &[u8]) -> TlsResult<Vec<u8>> {
    match alg {
        HashAlgorithm::Sha256 => hmac_sha256(key, data),
        HashAlgorithm::Sha384 => hmac_sha384(key, data),
    }
}

/// Constant-time equality for MAC / Finished verify_data.
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}
