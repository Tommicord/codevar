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

//! TLS 1.3 HKDF helpers (RFC 8446 §7.1, RFC 5869).

use crate::tls_codec::put_u16;
use crate::tls_crypto_hash::hash_empty;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::HashAlgorithm;
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
            let hk =
                Hkdf::<Sha256>::from_prk(secret).map_err(|_| TlsError::crypto("HKDF-SHA256 invalid PRK"))?;
            hk.expand(&hkdf_label, &mut out)
                .map_err(|_| TlsError::crypto("HKDF-SHA256 expand failed"))?;
        }
        HashAlgorithm::Sha384 => {
            let hk =
                Hkdf::<Sha384>::from_prk(secret).map_err(|_| TlsError::crypto("HKDF-SHA384 invalid PRK"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn rfc5869_test_case_1_extract() {
        // IKM 0x0b * 22, salt 0x00..0x0c, info 0xf0..0xf9, L = 42.
        let ikm = [0x0bu8; 22];
        let salt: Vec<u8> = (0..=12u8).collect();
        let prk = hkdf_extract(HashAlgorithm::Sha256, &salt, &ikm);
        assert_eq!(
            prk,
            unhex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")
        );
        // RFC 5869 A.1: expand the PRK with the test info and compare OKM.
        let info: Vec<u8> = (0xf0..0xfa).collect();
        let okm = hkdf_expand_label_manual(HashAlgorithm::Sha256, &prk, &info, 42);
        assert_eq!(
            okm,
            unhex(
                "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf\
                 34007208d5b887185865"
            )
        );
    }

    #[test]
    fn rfc5869_test_case_3_zero_salt_and_info() {
        let ikm = [0x0bu8; 22];
        let prk = hkdf_extract(HashAlgorithm::Sha256, b"", &ikm);
        assert_eq!(
            prk,
            unhex("19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04")
        );
        let okm = hkdf_expand_label_manual(HashAlgorithm::Sha256, &prk, b"", 42);
        assert_eq!(
            okm,
            unhex(
                "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d\
                 9d201395faa4b61a96c8"
            )
        );
    }

    #[test]
    fn extract_sha384_known_answer() {
        let ikm = [0x50u8; 20];
        let salt = [0x60u8; 22];
        let prk = hkdf_extract(HashAlgorithm::Sha384, &salt, &ikm);
        assert_eq!(
            prk,
            unhex(
                "99f647617e35b1313623173feb9198885f6ce40cead677c25a1ff1aa9f9f9015\
                 b78cd737a59985fa83221b57e73883bd"
            )
        );
        assert_eq!(prk.len(), 48);
    }

    #[test]
    fn extract_differs_by_salt_and_ikm() {
        let a = hkdf_extract(HashAlgorithm::Sha256, b"salt", b"ikm");
        let b = hkdf_extract(HashAlgorithm::Sha256, b"salt", b"ikm");
        assert_eq!(a, b);
        assert_ne!(a, hkdf_extract(HashAlgorithm::Sha256, b"salt2", b"ikm"));
        assert_ne!(a, hkdf_extract(HashAlgorithm::Sha256, b"salt", b"ikm2"));
        assert_eq!(a.len(), 32);
        assert_eq!(
            hkdf_extract(HashAlgorithm::Sha384, b"salt", b"ikm").len(),
            48
        );
    }

    #[test]
    fn expand_label_known_answer_and_prefix_property() {
        let secret = [0x77u8; 32];
        let ctx = [0xABu8; 32];
        let out = hkdf_expand_label(
            HashAlgorithm::Sha256,
            &secret,
            b"c hs traffic",
            &ctx,
            42,
        )
        .unwrap();
        assert_eq!(
            out,
            unhex(
                "f25c2329800c85f3e7740bcf756076683d60949ae47533981dea59d0e8abb7f0\
                 907d0773a4e7c731b5df"
            )
        );
        assert_eq!(out.len(), 42);

        // The requested length is bound into HkdfLabel, so outputs of different
        // lengths are unrelated (unlike raw HKDF-Expand over a fixed info).
        let long = hkdf_expand_label(HashAlgorithm::Sha256, &secret, b"c hs traffic", &ctx, 96)
            .unwrap();
        assert_eq!(long.len(), 96);
        assert_ne!(&long[..42], &out[..]);

        // The "tls13 " prefix makes labels domain-separated.
        let other = hkdf_expand_label(HashAlgorithm::Sha256, &secret, b"c hs traffi", &ctx, 42)
            .unwrap();
        assert_ne!(other, out);

        // Raw HKDF-Expand (RFC 5869) does have the prefix property.
        let prk = [0x55u8; 32];
        let short = hkdf_expand_label_manual(HashAlgorithm::Sha256, &prk, b"info", 32);
        let longer = hkdf_expand_label_manual(HashAlgorithm::Sha256, &prk, b"info", 96);
        assert_eq!(&longer[..32], &short[..]);
    }

    #[test]
    fn expand_label_sha384_known_answer() {
        let out = hkdf_expand_label(
            HashAlgorithm::Sha384,
            &[0x88u8; 48],
            b"s hs traffic",
            &[0xABu8; 32],
            48,
        )
        .unwrap();
        assert_eq!(
            out,
            unhex(
                "2d1e06c491612b22f22c96bfbf78a28e0921a32c175b0873d6c2e8de7a89ea38\
                 cb92955ceade6466cf21b67fc4395152"
            )
        );
    }

    #[test]
    fn expand_label_rejects_oversized_inputs() {
        // Length > 65535.
        let err = hkdf_expand_label(HashAlgorithm::Sha256, &[0u8; 32], b"key", &[], 65_536)
            .unwrap_err();
        assert!(matches!(err, TlsError::Internal(_)));
        // Context > 255 bytes.
        let err = hkdf_expand_label(
            HashAlgorithm::Sha256,
            &[0u8; 32],
            b"key",
            &[0u8; 256],
            32,
        )
        .unwrap_err();
        assert!(matches!(err, TlsError::Internal(_)));
        // Full label ("tls13 " + label) > 255 bytes.
        let long_label = [b'a'; 250];
        let err = hkdf_expand_label(HashAlgorithm::Sha256, &[0u8; 32], &long_label, &[], 32)
            .unwrap_err();
        assert!(matches!(err, TlsError::Internal(_)));
    }

    #[test]
    fn expand_label_rejects_short_prk() {
        // HKDF-Expand requires PRK >= hash length.
        let err = hkdf_expand_label(HashAlgorithm::Sha256, &[0u8; 16], b"key", &[], 32)
            .unwrap_err();
        assert!(matches!(err, TlsError::Crypto(_)));
        let err = hkdf_expand_label(HashAlgorithm::Sha384, &[0u8; 32], b"key", &[], 48)
            .unwrap_err();
        assert!(matches!(err, TlsError::Crypto(_)));
    }

    #[test]
    fn derive_secret_uses_empty_transcript_hash_by_default() {
        let secret = [0x77u8; 32];
        let explicit = hkdf_expand_label(
            HashAlgorithm::Sha256,
            &secret,
            b"derived",
            &crate::tls_crypto_hash::hash_empty(HashAlgorithm::Sha256),
            32,
        )
        .unwrap();
        let automatic = derive_secret(HashAlgorithm::Sha256, &secret, b"derived", None).unwrap();
        assert_eq!(automatic, explicit);
        assert_eq!(
            automatic,
            unhex("2ae86f6d5e72da644e094ee8359d423dd718897c3c23b616b1fdcc228968b96f")
        );

        // With an explicit message hash, that hash is used as the context.
        let ctx = [0xAAu8; 32];
        let with_ctx =
            derive_secret(HashAlgorithm::Sha256, &secret, b"derived", Some(&ctx)).unwrap();
        assert_ne!(with_ctx, automatic);
        assert_eq!(
            with_ctx,
            hkdf_expand_label(HashAlgorithm::Sha256, &secret, b"derived", &ctx, 32).unwrap()
        );

        // SHA-384 derives 48 bytes.
        let sha384_secret = [0x99u8; 48];
        let sha384 = derive_secret(HashAlgorithm::Sha384, &sha384_secret, b"derived", None)
            .unwrap();
        assert_eq!(sha384.len(), 48);
        // A PRK shorter than the hash output is invalid (RFC 5869 §2.2).
        let err = derive_secret(HashAlgorithm::Sha384, &secret, b"derived", None).unwrap_err();
        assert!(matches!(err, TlsError::Crypto(_)));
    }

    #[test]
    fn hkdf_alg_from_hash_algorithm() {
        let alg: HkdfAlg = HashAlgorithm::Sha384.into();
        match alg {
            HkdfAlg::Sha256 | HkdfAlg::Sha384 => {}
        }
        let alg: HkdfAlg = HashAlgorithm::Sha256.into();
        assert!(matches!(alg, HkdfAlg::Sha256));
    }

    /// Raw HKDF-Expand (no "tls13" label wrapping) for RFC 5869 vectors.
    fn hkdf_expand_label_manual(alg: HashAlgorithm, prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
        use hmac::{Hmac, Mac};
        fn expand<M: Mac + hmac::digest::KeyInit>(prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(length);
            let mut prev = Vec::new();
            let mut counter = 1u8;
            while out.len() < length {
                let mut mac = <M as Mac>::new_from_slice(prk).unwrap();
                mac.update(&prev);
                mac.update(info);
                mac.update(&[counter]);
                let block = mac.finalize().into_bytes();
                out.extend_from_slice(&block);
                prev = block.to_vec();
                counter += 1;
            }
            out.truncate(length);
            out
        }
        match alg {
            HashAlgorithm::Sha256 => expand::<Hmac<Sha256>>(prk, info, length),
            HashAlgorithm::Sha384 => expand::<Hmac<Sha384>>(prk, info, length),
        }
    }
}
