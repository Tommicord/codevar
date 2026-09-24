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

//! Handshake and certificate signatures.

use crate::tls_alert::AlertDescription;
use crate::tls_crypto_random::SysRng;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::SignatureScheme;
use ecdsa::signature::Verifier as EcdsaVerifier;
use ed25519_dalek::{Signature as Ed25519Signature, Signer as Ed25519Signer};
use p256::ecdsa::{
    Signature as P256Signature, SigningKey as P256SigningKey,
    VerifyingKey as P256VerifyingKey,
};
use p384::ecdsa::{
    Signature as P384Signature, SigningKey as P384SigningKey,
    VerifyingKey as P384VerifyingKey,
};
use rsa::pkcs1v15::{
    Signature as RsaPkcs1Signature, SigningKey as RsaPkcs1SigningKey,
    VerifyingKey as RsaPkcs1VerifyingKey,
};
use rsa::pss::{
    Signature as RsaPssSignature, SigningKey as RsaPssSigningKey,
    VerifyingKey as RsaPssVerifyingKey,
};
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::{RsaPrivateKey, RsaPublicKey};
use sha2::{Digest, Sha256, Sha384};

/// Parsed private key used for CertificateVerify / ServerKeyExchange.
pub enum PrivateKey {
    /// RSA private key.
    Rsa(RsaPrivateKey),
    /// ECDSA P-256.
    EcdsaP256(P256SigningKey),
    /// ECDSA P-384.
    EcdsaP384(P384SigningKey),
    /// Ed25519.
    Ed25519(ed25519_dalek::SigningKey),
}

/// Classification of a signature scheme for algorithm matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureKind {
    /// RSA key.
    Rsa,
    /// ECDSA P-256.
    EcdsaP256,
    /// ECDSA P-384.
    EcdsaP384,
    /// Ed25519.
    Ed25519,
}

impl PrivateKey {
    /// Loads a private key from PEM (`PRIVATE KEY`, `RSA PRIVATE KEY`, or `EC PRIVATE KEY`).
    pub fn from_pem(pem_bytes: &[u8]) -> TlsResult<Self> {
        let pem_str = std::str::from_utf8(pem_bytes)
            .map_err(|_| TlsError::certificate("private key PEM is not UTF-8"))?;
        let parsed = pem::parse(pem_str)
            .map_err(|e| TlsError::certificate(format!("PEM parse: {e}")))?;
        match parsed.tag() {
            "PRIVATE KEY" => Self::from_pkcs8_der(parsed.contents()),
            "RSA PRIVATE KEY" => {
                let key =
                    RsaPrivateKey::from_pkcs1_der(parsed.contents()).map_err(|e| {
                        TlsError::certificate(format!("RSA PKCS#1 parse: {e}"))
                    })?;
                Ok(Self::Rsa(key))
            }
            "EC PRIVATE KEY" => Self::from_sec1_der(parsed.contents()),
            other => Err(TlsError::certificate(format!(
                "unsupported private key PEM tag {other}"
            ))),
        }
    }

    fn from_pkcs8_der(der: &[u8]) -> TlsResult<Self> {
        if let Ok(key) = P256SigningKey::from_pkcs8_der(der) {
            return Ok(Self::EcdsaP256(key));
        }
        if let Ok(key) = P384SigningKey::from_pkcs8_der(der) {
            return Ok(Self::EcdsaP384(key));
        }
        if let Ok(key) = ed25519_dalek::SigningKey::from_pkcs8_der(der) {
            return Ok(Self::Ed25519(key));
        }
        if let Ok(key) = RsaPrivateKey::from_pkcs8_der(der) {
            return Ok(Self::Rsa(key));
        }
        Err(TlsError::certificate(String::from(
            "unable to parse PKCS#8 private key",
        )))
    }

    fn from_sec1_der(der: &[u8]) -> TlsResult<Self> {
        if let Ok(sk) = p256::SecretKey::from_sec1_der(der) {
            return Ok(Self::EcdsaP256(P256SigningKey::from(sk)));
        }
        if let Ok(sk) = p384::SecretKey::from_sec1_der(der) {
            return Ok(Self::EcdsaP384(P384SigningKey::from(sk)));
        }
        Err(TlsError::certificate(String::from(
            "unable to parse SEC1 EC private key",
        )))
    }

    /// Returns the key kind.
    #[must_use]
    pub fn kind(&self) -> SignatureKind {
        match self {
            Self::Rsa(_) => SignatureKind::Rsa,
            Self::EcdsaP256(_) => SignatureKind::EcdsaP256,
            Self::EcdsaP384(_) => SignatureKind::EcdsaP384,
            Self::Ed25519(_) => SignatureKind::Ed25519,
        }
    }

    /// Selects a signature scheme compatible with this key from the peer's list.
    pub fn select_scheme(
        &self,
        offered: &[SignatureScheme],
    ) -> TlsResult<SignatureScheme> {
        let candidates: &[SignatureScheme] = match self.kind() {
            SignatureKind::Rsa => &[
                SignatureScheme::RsaPssRsaeSha256,
                SignatureScheme::RsaPssRsaeSha384,
                SignatureScheme::RsaPkcs1Sha256,
                SignatureScheme::RsaPkcs1Sha384,
            ],
            SignatureKind::EcdsaP256 => &[SignatureScheme::EcdsaSecp256r1Sha256],
            SignatureKind::EcdsaP384 => &[SignatureScheme::EcdsaSecp384r1Sha384],
            SignatureKind::Ed25519 => &[SignatureScheme::Ed25519],
        };
        for scheme in offered {
            if candidates.contains(scheme) {
                return Ok(*scheme);
            }
        }
        Err(TlsError::Alert(AlertDescription::HandshakeFailure))
    }

    /// Signs `message` with `scheme`.
    pub fn sign(&self, scheme: SignatureScheme, message: &[u8]) -> TlsResult<Vec<u8>> {
        let mut rng = SysRng;
        match (self, scheme) {
            (Self::Rsa(key), SignatureScheme::RsaPssRsaeSha256) => {
                let signing_key = RsaPssSigningKey::<Sha256>::new(key.clone());
                let sig = signing_key.sign_with_rng(&mut rng, message);
                Ok(sig.to_bytes().into())
            }
            (Self::Rsa(key), SignatureScheme::RsaPssRsaeSha384) => {
                let signing_key = RsaPssSigningKey::<Sha384>::new(key.clone());
                let sig = signing_key.sign_with_rng(&mut rng, message);
                Ok(sig.to_bytes().into())
            }
            (Self::Rsa(key), SignatureScheme::RsaPkcs1Sha256) => {
                let signing_key = RsaPkcs1SigningKey::<Sha256>::new(key.clone());
                let sig = signing_key.sign_with_rng(&mut rng, message);
                Ok(sig.to_bytes().into())
            }
            (Self::Rsa(key), SignatureScheme::RsaPkcs1Sha384) => {
                let signing_key = RsaPkcs1SigningKey::<Sha384>::new(key.clone());
                let sig = signing_key.sign_with_rng(&mut rng, message);
                Ok(sig.to_bytes().into())
            }
            (Self::EcdsaP256(key), SignatureScheme::EcdsaSecp256r1Sha256) => {
                use ecdsa::signature::RandomizedSigner;
                let sig: P256Signature = key.sign_with_rng(&mut rng, message);
                Ok(sig.to_der().as_bytes().to_vec())
            }
            (Self::EcdsaP384(key), SignatureScheme::EcdsaSecp384r1Sha384) => {
                use ecdsa::signature::RandomizedSigner;
                let sig: P384Signature = key.sign_with_rng(&mut rng, message);
                Ok(sig.to_der().as_bytes().to_vec())
            }
            (Self::Ed25519(key), SignatureScheme::Ed25519) => {
                let sig = key.sign(message);
                Ok(sig.to_bytes().to_vec())
            }
            _ => Err(TlsError::Unsupported(
                "signature scheme does not match private key".into(),
            )),
        }
    }
}

use pkcs8::DecodePrivateKey;
use rsa::pkcs1::DecodeRsaPrivateKey;

/// Trait for types that can be signed/verified as raw public keys from SPKI.
pub trait Signable {}

/// Builds the TLS 1.3 CertificateVerify content (RFC 8446 §4.4.3).
#[must_use]
pub fn tls13_cert_verify_content(is_server: bool, transcript_hash: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + 33 + 1 + transcript_hash.len());
    out.extend(std::iter::repeat_n(0x20u8, 64));
    if is_server {
        out.extend_from_slice(b"TLS 1.3, server CertificateVerify");
    } else {
        out.extend_from_slice(b"TLS 1.3, client CertificateVerify");
    }
    out.push(0x00);
    out.extend_from_slice(transcript_hash);
    out
}

/// Verifies a signature over `message` using SPKI DER and `scheme`.
pub fn verify_handshake_signature(
    scheme: SignatureScheme,
    spki_der: &[u8],
    message: &[u8],
    signature: &[u8],
) -> TlsResult<()> {
    verify_raw_signature(scheme, spki_der, message, signature)
}

/// Verifies `signature` over `message` with the public key in `spki_der`.
pub fn verify_raw_signature(
    scheme: SignatureScheme,
    spki_der: &[u8],
    message: &[u8],
    signature: &[u8],
) -> TlsResult<()> {
    match scheme {
        SignatureScheme::EcdsaSecp256r1Sha256 => {
            let vk = P256VerifyingKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid P-256 SPKI"))?;
            let sig = P256Signature::from_der(signature)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
        SignatureScheme::EcdsaSecp384r1Sha384 => {
            let vk = P384VerifyingKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid P-384 SPKI"))?;
            let sig = P384Signature::from_der(signature)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
        SignatureScheme::Ed25519 => {
            let vk = ed25519_dalek::VerifyingKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid Ed25519 SPKI"))?;
            let sig_arr: [u8; 64] = signature
                .try_into()
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            let sig = Ed25519Signature::from_bytes(&sig_arr);
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
        SignatureScheme::RsaPssRsaeSha256 => {
            let pk = RsaPublicKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid RSA SPKI"))?;
            let vk = RsaPssVerifyingKey::<Sha256>::new(pk);
            let sig = RsaPssSignature::try_from(signature)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
        SignatureScheme::RsaPssRsaeSha384 => {
            let pk = RsaPublicKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid RSA SPKI"))?;
            let vk = RsaPssVerifyingKey::<Sha384>::new(pk);
            let sig = RsaPssSignature::try_from(signature)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
        SignatureScheme::RsaPkcs1Sha256 => {
            let pk = RsaPublicKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid RSA SPKI"))?;
            let vk = RsaPkcs1VerifyingKey::<Sha256>::new(pk);
            let sig = RsaPkcs1Signature::try_from(signature)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
        SignatureScheme::RsaPkcs1Sha384 => {
            let pk = RsaPublicKey::from_public_key_der(spki_der)
                .map_err(|_| TlsError::certificate("invalid RSA SPKI"))?;
            let vk = RsaPkcs1VerifyingKey::<Sha384>::new(pk);
            let sig = RsaPkcs1Signature::try_from(signature)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))?;
            vk.verify(message, &sig)
                .map_err(|_| TlsError::Alert(AlertDescription::DecryptError))
        }
    }
}

/// Verifies an X.509 certificate signature using the issuer SPKI and algorithm OID.
pub fn verify_cert_signature(
    tbs: &[u8],
    signature_oid: &str,
    signature: &[u8],
    issuer_spki: &[u8],
) -> TlsResult<()> {
    match signature_oid {
        "1.2.840.113549.1.1.11" => {
            // sha256WithRSAEncryption — verify DigestInfo via PKCS#1
            verify_rsa_pkcs1_digest(issuer_spki, tbs, signature, HashChoice::Sha256)
        }
        "1.2.840.113549.1.1.12" => {
            verify_rsa_pkcs1_digest(issuer_spki, tbs, signature, HashChoice::Sha384)
        }
        "1.2.840.10045.4.3.2" => verify_raw_signature(
            SignatureScheme::EcdsaSecp256r1Sha256,
            issuer_spki,
            tbs,
            signature,
        ),
        "1.2.840.10045.4.3.3" => verify_raw_signature(
            SignatureScheme::EcdsaSecp384r1Sha384,
            issuer_spki,
            tbs,
            signature,
        ),
        "1.3.101.112" => {
            verify_raw_signature(SignatureScheme::Ed25519, issuer_spki, tbs, signature)
        }
        "1.2.840.113549.1.1.10" => {
            // RSASSA-PSS — try SHA-256 then SHA-384
            if verify_raw_signature(
                SignatureScheme::RsaPssRsaeSha256,
                issuer_spki,
                tbs,
                signature,
            )
            .is_ok()
            {
                Ok(())
            } else {
                verify_raw_signature(
                    SignatureScheme::RsaPssRsaeSha384,
                    issuer_spki,
                    tbs,
                    signature,
                )
            }
        }
        other => Err(TlsError::certificate(format!(
            "unsupported certificate signature OID {other}"
        ))),
    }
}

enum HashChoice {
    Sha256,
    Sha384,
}

fn verify_rsa_pkcs1_digest(
    spki: &[u8],
    tbs: &[u8],
    signature: &[u8],
    hash: HashChoice,
) -> TlsResult<()> {
    match hash {
        HashChoice::Sha256 => {
            // rsa crate VerifyingKey hashes the message itself
            verify_raw_signature(SignatureScheme::RsaPkcs1Sha256, spki, tbs, signature)
        }
        HashChoice::Sha384 => {
            verify_raw_signature(SignatureScheme::RsaPkcs1Sha384, spki, tbs, signature)
        }
    }
}

use pkcs8::DecodePublicKey;

/// Hashes data with SHA-256.
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}
