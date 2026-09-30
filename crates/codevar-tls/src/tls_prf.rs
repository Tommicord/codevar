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

//! TLS 1.2 PRF (RFC 5246 §5).

use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::HashAlgorithm;
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
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| TlsError::crypto("HMAC-SHA256 key"))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn hmac_sha384(key: &[u8], data: &[u8]) -> TlsResult<Vec<u8>> {
    let mut mac = HmacSha384::new_from_slice(key).map_err(|_| TlsError::crypto("HMAC-SHA384 key"))?;
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
///
/// Re-exported from [`crate::tls_crypto_ct`] so every comparison in the
/// handshake goes through the same branch-free implementation.
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    crate::tls_crypto_ct::ct_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::{Hmac, Mac};
    use sha2::{Digest, Sha256, Sha384};

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    /// Independent RFC 5246 §5 P_hash: P_hash = HMAC(secret, A(1)+seed) || ...
    fn reference_p_hash(alg: HashAlgorithm, secret: &[u8], seed: &[u8], out_len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        match alg {
            HashAlgorithm::Sha256 => {
                let mut a = seed.to_vec();
                while out.len() < out_len {
                    let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
                    mac.update(&a);
                    a = mac.finalize().into_bytes().to_vec();
                    let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
                    mac.update(&a);
                    mac.update(seed);
                    out.extend_from_slice(&mac.finalize().into_bytes());
                }
            }
            HashAlgorithm::Sha384 => {
                let mut a = seed.to_vec();
                while out.len() < out_len {
                    let mut mac = Hmac::<Sha384>::new_from_slice(secret).unwrap();
                    mac.update(&a);
                    a = mac.finalize().into_bytes().to_vec();
                    let mut mac = Hmac::<Sha384>::new_from_slice(secret).unwrap();
                    mac.update(&a);
                    mac.update(seed);
                    out.extend_from_slice(&mac.finalize().into_bytes());
                }
            }
        }
        out.truncate(out_len);
        out
    }

    #[test]
    fn prf_matches_reference_p_hash_for_both_algorithms() {
        let secret = [0x01u8; 48];
        let seed = [0x02u8; 64];
        for alg in [HashAlgorithm::Sha256, HashAlgorithm::Sha384] {
            for out_len in [0usize, 1, 16, 31, 32, 33, 48, 64, 70, 128] {
                let got = tls12_prf(alg, &secret, b"master secret", &seed, out_len).unwrap();
                let want = reference_p_hash(
                    alg,
                    &secret,
                    &[b"master secret".as_slice(), &seed].concat(),
                    out_len,
                );
                assert_eq!(got, want, "alg={alg:?} out_len={out_len}");
                assert_eq!(got.len(), out_len);
            }
        }
    }

    #[test]
    fn prf_known_answers_computed_independently() {
        let secret = [0x01u8; 48];
        let seed = [0x02u8; 64];
        // Independent Python hmac/sha256 computation.
        let sha256 = tls12_prf(HashAlgorithm::Sha256, &secret, b"master secret", &seed, 48).unwrap();
        assert_eq!(
            sha256,
            unhex(
                "6e5a2a97a87e1151943a72f710f2138989a9fb19a09d98bfdcb0e6c806f72e6c\
                 316d9f33a671c715819de25940439770"
            )
        );
        let sha384 = tls12_prf(HashAlgorithm::Sha384, &secret, b"master secret", &seed, 48).unwrap();
        assert_eq!(
            sha384,
            unhex(
                "768f5bc12d610ac749f224789050224a156515982b08b9c663e03a9f642c5da9\
                 838bc54a7e2c9bc7b9e1aa2fee684155"
            )
        );
        // 70 bytes exercises two SHA-256 blocks plus a partial third.
        let long = tls12_prf(HashAlgorithm::Sha256, &secret, b"key expansion", &seed, 70).unwrap();
        assert_eq!(
            long,
            unhex(
                "b4850e178e8d2b084cfa49505cf1ef89387f1f6bc1aac4eaa9a3bef61a63b627\
                 1ee35f6bfef79ee220e0c4dd09b8fd0decbb184164b4b4e15f8eb3abbe6c6cc0\
                 8b1199e12f65"
            )
        );
    }

    #[test]
    fn prf_is_deterministic_and_input_sensitive() {
        let secret = [0x07u8; 32];
        let seed = [0x09u8; 32];
        let a = tls12_prf(HashAlgorithm::Sha256, &secret, b"lbl", &seed, 32).unwrap();
        let b = tls12_prf(HashAlgorithm::Sha256, &secret, b"lbl", &seed, 32).unwrap();
        assert_eq!(a, b);
        let other_label = tls12_prf(HashAlgorithm::Sha256, &secret, b"lbm", &seed, 32).unwrap();
        assert_ne!(a, other_label);
        let mut other_seed = seed;
        other_seed[0] ^= 1;
        assert_ne!(
            a,
            tls12_prf(HashAlgorithm::Sha256, &secret, b"lbl", &other_seed, 32).unwrap()
        );
        let mut other_secret = secret;
        other_secret[0] ^= 1;
        assert_ne!(
            a,
            tls12_prf(HashAlgorithm::Sha256, &other_secret, b"lbl", &seed, 32).unwrap()
        );
        // Prefix property: a longer PRF starts with the shorter one.
        let long = tls12_prf(HashAlgorithm::Sha256, &secret, b"lbl", &seed, 96).unwrap();
        assert_eq!(&long[..32], &a[..]);
    }

    #[test]
    fn hmac_hash_matches_rfc4231_vectors() {
        // RFC 4231 test case 1 (HMAC-SHA-256 / HMAC-SHA-384).
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        assert_eq!(
            hmac_hash(HashAlgorithm::Sha256, &key, data).unwrap(),
            unhex("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7")
        );
        assert_eq!(
            hmac_hash(HashAlgorithm::Sha384, &key, data).unwrap(),
            unhex(
                "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59c\
                 faea9ea9076ede7f4af152e8b2fa9cb6"
            )
        );
        // RFC 4231 test case 2.
        assert_eq!(
            hmac_hash(HashAlgorithm::Sha256, b"Jefe", b"what do ya want for nothing?").unwrap(),
            unhex("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
        );
    }

    #[test]
    fn hmac_hash_output_lengths() {
        assert_eq!(
            hmac_hash(HashAlgorithm::Sha256, b"k", b"m")
                .unwrap()
                .len(),
            32
        );
        assert_eq!(
            hmac_hash(HashAlgorithm::Sha384, b"k", b"m")
                .unwrap()
                .len(),
            48
        );
    }

    #[test]
    fn ct_eq_handles_equal_different_and_length_mismatch() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(ct_eq(b"", b""));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
        assert!(!ct_eq(b"", b"x"));
        assert!(!ct_eq(&[0u8; 32], &[0u8; 48]));
    }

    #[test]
    fn sha_digest_length_matches_hash_algorithm() {
        // Sanity: the PRF really is parameterized by the requested algorithm.
        let out = tls12_prf(HashAlgorithm::Sha256, b"s", b"l", b"seed", 32).unwrap();
        assert_ne!(
            out,
            tls12_prf(HashAlgorithm::Sha384, b"s", b"l", b"seed", 32).unwrap()
        );
        assert_eq!(Sha256::digest(b"").len(), 32);
        assert_eq!(Sha384::digest(b"").len(), 48);
    }
}
