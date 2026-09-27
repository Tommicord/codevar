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
use ed25519_dalek::{Signature as Ed25519Signature, Signer as Ed25519Signer, VerifyingKey};
use p256::ecdsa::{
    Signature as P256Signature, SigningKey as P256SigningKey, VerifyingKey as P256VerifyingKey,
};
use p384::ecdsa::{
    Signature as P384Signature, SigningKey as P384SigningKey, VerifyingKey as P384VerifyingKey,
};
use rsa::pkcs1v15::{
    Signature as RsaPkcs1Signature, SigningKey as RsaPkcs1SigningKey, VerifyingKey as RsaPkcs1VerifyingKey,
};
use rsa::pss::{
    Signature as RsaPssSignature, SigningKey as RsaPssSigningKey, VerifyingKey as RsaPssVerifyingKey,
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
        let parsed = pem::parse(pem_str).map_err(|e| TlsError::certificate(format!("PEM parse: {e}")))?;
        match parsed.tag() {
            "PRIVATE KEY" => Self::from_pkcs8_der(parsed.contents()),
            "RSA PRIVATE KEY" => {
                let key = RsaPrivateKey::from_pkcs1_der(parsed.contents())
                    .map_err(|e| TlsError::certificate(format!("RSA PKCS#1 parse: {e}")))?;
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
    pub fn select_scheme(&self, offered: &[SignatureScheme]) -> TlsResult<SignatureScheme> {
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
            let vk = VerifyingKey::from_public_key_der(spki_der)
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
        "1.2.840.113549.1.1.12" => verify_rsa_pkcs1_digest(issuer_spki, tbs, signature, HashChoice::Sha384),
        "1.2.840.10045.4.3.2" => {
            verify_raw_signature(SignatureScheme::EcdsaSecp256r1Sha256, issuer_spki, tbs, signature)
        }
        "1.2.840.10045.4.3.3" => {
            verify_raw_signature(SignatureScheme::EcdsaSecp384r1Sha384, issuer_spki, tbs, signature)
        }
        "1.3.101.112" => verify_raw_signature(SignatureScheme::Ed25519, issuer_spki, tbs, signature),
        "1.2.840.113549.1.1.10" => {
            // RSASSA-PSS — try SHA-256 then SHA-384
            if verify_raw_signature(SignatureScheme::RsaPssRsaeSha256, issuer_spki, tbs, signature).is_ok() {
                Ok(())
            } else {
                verify_raw_signature(SignatureScheme::RsaPssRsaeSha384, issuer_spki, tbs, signature)
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

fn verify_rsa_pkcs1_digest(spki: &[u8], tbs: &[u8], signature: &[u8], hash: HashChoice) -> TlsResult<()> {
    match hash {
        HashChoice::Sha256 => {
            // rsa crate VerifyingKey hashes the message itself
            verify_raw_signature(SignatureScheme::RsaPkcs1Sha256, spki, tbs, signature)
        }
        HashChoice::Sha384 => verify_raw_signature(SignatureScheme::RsaPkcs1Sha384, spki, tbs, signature),
    }
}

use pkcs8::DecodePublicKey;

/// Hashes data with SHA-256.
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    const ED25519_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIGHJ7vDVIPs7azrXADYOKhREcL5Wvccy7u3MGBD58his
-----END PRIVATE KEY-----
";
    const ED25519_SPKI: &str = "302a300506032b6570032100eb1a56d88d3d6918a298c05f413853e70a22f3a22f0f1e20494d51682579cc96";

    const P256_PKCS8_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgm/WN/mvVOsU5YIKS
AXB+dAb1FL7IZUuBjXRQ4Mp6WtGhRANCAAT7O5vBjUJz74gj+iMMdHAff7WXr6Bi
BG8wFg0fX8jfhi2NV6VFKNWUtO26TqrAeTvvoSC6uPPLH7PMRudXv6aW
-----END PRIVATE KEY-----
";
    const P256_SEC1_PEM: &[u8] = b"-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIJv1jf5r1TrFOWCCkgFwfnQG9RS+yGVLgY10UODKelrRoAoGCCqGSM49
AwEHoUQDQgAE+zubwY1Cc++II/ojDHRwH3+1l6+gYgRvMBYNH1/I34YtjVelRSjV
lLTtuk6qwHk776Egurjzyx+zzEbnV7+mlg==
-----END EC PRIVATE KEY-----
";
    const P256_SPKI: &str = "3059301306072a8648ce3d020106082a8648ce3d03010703420004fb3b9bc18d4273ef8823fa230c74701f7fb597afa062046f30160d1f5fc8df862d8d57a54528d594b4edba4eaac0793befa120bab8f3cb1fb3cc46e757bfa696";

    const P384_SEC1_PEM: &[u8] = b"-----BEGIN EC PRIVATE KEY-----
MIGkAgEBBDDR4TEPtXmEuXV8B/X1XKJNu/LXKxuUvTODUQdeRO9BJjgjApFo8Kd0
mshUpjBAMhugBwYFK4EEACKhZANiAARWQEk/6b/Bx16X1+pKs9d9OWfYjuYPfWJE
9shV+ZdgoRjXzMhhbQf4077SV/Kol9DR/Vrbrr1VxpfuMqRKBwGQC+rvjh8fQVip
tB6kwAbOPWirWoQxtqRpnhAaSEsM61E=
-----END EC PRIVATE KEY-----
";
    const P384_PKCS8_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIG2AgEAMBAGByqGSM49AgEGBSuBBAAiBIGeMIGbAgEBBDDR4TEPtXmEuXV8B/X1
XKJNu/LXKxuUvTODUQdeRO9BJjgjApFo8Kd0mshUpjBAMhuhZANiAARWQEk/6b/B
x16X1+pKs9d9OWfYjuYPfWJE9shV+ZdgoRjXzMhhbQf4077SV/Kol9DR/Vrbrr1V
xpfuMqRKBwGQC+rvjh8fQViptB6kwAbOPWirWoQxtqRpnhAaSEsM61E=
-----END PRIVATE KEY-----
";
    const P384_SPKI: &str = "3076301006072a8648ce3d020106052b81040022036200045640493fe9bfc1c75e97d7ea4ab3d77d3967d88ee60f7d6244f6c855f99760a118d7ccc8616d07f8d3bed257f2a897d0d1fd5adbaebd55c697ee32a44a0701900beaef8e1f1f4158a9b41ea4c006ce3d68ab5a8431b6a4699e101a484b0ceb51";

    const RSA_PKCS1_PEM: &[u8] = b"-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEArSHGICofnzgByhA3WS0hsj2g+yv4HYgbsnJR1dpVpxgbRVmU
y4fEgZmSunq8/XJU3bXsTBTqIg8C118t1bo9s4kZj11JpBeILt0oUTy0nEAnlgkV
H5IOpTqpSG60eSw97FDvLEIFNRC013ZTFnsYKvxH2n49GAk4oByLetfyNVeT/fuq
CMNH3PlrPYrj0KabzTCjavihGu1LHVtKevXx4sQCLH0VTDAPqHQfABRaXUnjXmYS
7tK0A0QU3APKCU7cUCYQBrTT32HXgqzN/2PoMmP0B7Q9msdhbsirhUG1eyG2zIJ+
aNb/Iw2VJs02ZNeF3noZ9FYJ/0SrVgwW7IC3pwIDAQABAoIBAAIhV8j+eEwyfGMH
ccckA5Sv4L3hlHtXcxmcrHSGc5bkk6GwIhOLNY1j8wi7qemb4LkMnfKftAsRsXT/
w7Nx5tmx1qDPVizDQv9TKl0O3CynVdo0u1ey56/8A1qy8DG/+ZAbrVAfO2mQezIf
6qIi88CNtYivHEKiIgyM9KLRkYuYTl/W3tA8Zzw1OM/rRTVWPEZoX7X7Bh2pXNmL
47ags69LsrM9fF1f6XpBeEO+trwFHowOlRN/e8firXVAgLjiKUg3OONAavoKzaDv
E8+Biv1ZypR8G9xVPSIiwmlJwqCCkRSvCDGEaqrBpB0T5IWT2aMq8MhirA321Vh9
SzSMUp0CgYEA16McIyc9WQ8xNnHfGm/EUrz0+tZR8R81oBgygMYUBpN/5YY6LCsj
XX3tLRNaIkaI0YHA16Wx0w1q3WljeduvWxO8VbERqk5alIQqRponyEDTMu4lVh5t
7DAya9DFz9KYzk2q3JqiCu+wAt1SwRH4HS6zn66aQuuIda6fV0PjdC0CgYEAzYnm
rKF6yYw5VYqgVO37oM2IIrjyLQa61tD+lAtI8IeGKnvYJTVqFpnv1oBWhDPLy1ei
EUqxKB6m0VAD7L7gv/N4hlECWsIKkxWky616BKL3HKA9Wq7M2WvusTWCZz56LL5v
VtzI+uijdeto0RsNpJ18itm2WMkeFrprRc5BG6MCgYEAgLeMhC9YH2FCLw2p703o
ErtYQKLOJvBKQlYvT07thT9miMumzdayNYyFlvKGSw9vfB/HzPHLRAAuWhwelQ7d
jbAqK7yWnuJGj4CU+6/pL1ACwdQF1XI0i1f5wQHOEk/ThtI1u5lSQNTya8NRMo5X
XJQUBK1kx61wt06VLrIHg8ECgYACbjoLGpHAJmdbKpuAxZuvKB8PAg7jF2xINAp+
p6+CIKgmgnZo4jvR7poWeEgNoisryWqCN9bmeY5RdVfGie9QdcmcFQMpHZhQavYa
7n895Mt1TDPNibmkbJFTCSX0oJmDQ/E/HrdFuFCOiW0g4+gCZVZesfplDGPNswCn
v6unbwKBgAZYdEgxVI9cvwki/ndRL0qOu3rdwMAvDhwMeCf180xcVYSqlf8t+iV9
7ni4zfEoNkmU9qOEIN8q7EBZzuUKHK7827QtChh89Vr9z0FrSaiiwroKj+917CxO
Uap664VuCyehCd2ifEdytjvQUbTakC4dZdOuooge+1cxadVbGQy0
-----END RSA PRIVATE KEY-----
";
    const RSA_SPKI: &str = "30820122300d06092a864886f70d01010105000382010f003082010a0282010100ad21c6202a1f9f3801ca1037592d21b23da0fb2bf81d881bb27251d5da55a7181b455994cb87c4819992ba7abcfd7254ddb5ec4c14ea220f02d75f2dd5ba3db389198f5d49a417882edd28513cb49c40279609151f920ea53aa9486eb4792c3dec50ef2c42053510b4d77653167b182afc47da7e3d180938a01c8b7ad7f2355793fdfbaa08c347dcf96b3d8ae3d0a69bcd30a36af8a11aed4b1d5b4a7af5f1e2c4022c7d154c300fa8741f00145a5d49e35e6612eed2b4034414dc03ca094edc50261006b4d3df61d782accdff63e83263f407b43d9ac7616ec8ab8541b57b21b6cc827e68d6ff230d9526cd3664d785de7a19f45609ff44ab560c16ec80b7a70203010001";

    #[test]
    fn from_pem_classifies_supported_formats() {
        let k = PrivateKey::from_pem(ED25519_PEM).unwrap();
        assert_eq!(k.kind(), SignatureKind::Ed25519);

        let k = PrivateKey::from_pem(P256_PKCS8_PEM).unwrap();
        assert_eq!(k.kind(), SignatureKind::EcdsaP256);
        let k = PrivateKey::from_pem(P256_SEC1_PEM).unwrap();
        assert_eq!(k.kind(), SignatureKind::EcdsaP256);

        let k = PrivateKey::from_pem(P384_SEC1_PEM).unwrap();
        assert_eq!(k.kind(), SignatureKind::EcdsaP384);
        let k = PrivateKey::from_pem(P384_PKCS8_PEM).unwrap();
        assert_eq!(k.kind(), SignatureKind::EcdsaP384);

        let k = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        assert_eq!(k.kind(), SignatureKind::Rsa);
    }

    #[test]
    fn from_pem_rejects_invalid_input() {
        // Not UTF-8.
        assert!(PrivateKey::from_pem(&[0xFF, 0xFE, 0x00, 0x80]).is_err());
        // No PEM block at all.
        assert!(PrivateKey::from_pem(b"not a pem").is_err());
        // Valid PEM framing but unsupported tag.
        let cert = b"-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n";
        let err = match PrivateKey::from_pem(cert) {
            Err(e) => e,
            Ok(_) => TlsError::certificate("unexpected success"),
        };
        assert!(matches!(err, TlsError::Certificate(_)));
        // Private-key tag with unparseable DER body.
        let junk = b"-----BEGIN PRIVATE KEY-----\nAAAAAAAA\n-----END PRIVATE KEY-----\n";
        assert!(PrivateKey::from_pem(junk).is_err());
    }

    #[test]
    fn select_scheme_respects_offered_preference_order() {
        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        // Offered order, not candidate order, decides.
        let chosen = rsa
            .select_scheme(&[
                SignatureScheme::EcdsaSecp256r1Sha256,
                SignatureScheme::RsaPkcs1Sha384,
                SignatureScheme::RsaPssRsaeSha256,
            ])
            .unwrap();
        assert_eq!(chosen, SignatureScheme::RsaPkcs1Sha384);
        let chosen = rsa
            .select_scheme(&[
                SignatureScheme::RsaPssRsaeSha256,
                SignatureScheme::RsaPkcs1Sha256,
            ])
            .unwrap();
        assert_eq!(chosen, SignatureScheme::RsaPssRsaeSha256);

        let p256 = PrivateKey::from_pem(P256_SEC1_PEM).unwrap();
        let chosen = p256
            .select_scheme(&[
                SignatureScheme::Ed25519,
                SignatureScheme::EcdsaSecp256r1Sha256,
            ])
            .unwrap();
        assert_eq!(chosen, SignatureScheme::EcdsaSecp256r1Sha256);

        let ed = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let chosen = ed.select_scheme(&[SignatureScheme::Ed25519]).unwrap();
        assert_eq!(chosen, SignatureScheme::Ed25519);

        let p384 = PrivateKey::from_pem(P384_SEC1_PEM).unwrap();
        let chosen = p384
            .select_scheme(&[SignatureScheme::EcdsaSecp384r1Sha384])
            .unwrap();
        assert_eq!(chosen, SignatureScheme::EcdsaSecp384r1Sha384);
    }

    #[test]
    fn select_scheme_rejects_when_no_overlap() {
        let p256 = PrivateKey::from_pem(P256_SEC1_PEM).unwrap();
        let err = p256
            .select_scheme(&[
                SignatureScheme::RsaPssRsaeSha256,
                SignatureScheme::RsaPkcs1Sha256,
                SignatureScheme::Ed25519,
            ])
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::HandshakeFailure));

        let ed = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let err = ed
            .select_scheme(&[SignatureScheme::EcdsaSecp256r1Sha256])
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::HandshakeFailure));

        // Empty list never matches.
        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        let err = rsa.select_scheme(&[]).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::HandshakeFailure));
    }

    /// Signs `message` with every (key, scheme) pair and verifies the result.
    fn sign_and_verify(key: &PrivateKey, scheme: SignatureScheme, spki_hex: &str, message: &[u8]) {
        let signature = key.sign(scheme, message).unwrap();
        assert!(!signature.is_empty());
        let spki = unhex(spki_hex);
        verify_handshake_signature(scheme, &spki, message, &signature).unwrap();
        // The wrapper and the underlying verifier agree.
        verify_raw_signature(scheme, &spki, message, &signature).unwrap();
        // A different message must fail to verify.
        let mut other = message.to_vec();
        other.push(0xFF);
        assert!(verify_handshake_signature(scheme, &spki, &other, &signature).is_err());
    }

    #[test]
    fn ed25519_sign_verify_round_trip() {
        let key = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let sig = key.sign(SignatureScheme::Ed25519, b"handshake transcript").unwrap();
        // Ed25519 signatures are fixed 64 bytes and deterministic.
        assert_eq!(sig.len(), 64);
        let sig2 = key.sign(SignatureScheme::Ed25519, b"handshake transcript").unwrap();
        assert_eq!(sig, sig2);

        sign_and_verify(
            &key,
            SignatureScheme::Ed25519,
            ED25519_SPKI,
            b"handshake transcript",
        );
    }

    #[test]
    fn ecdsa_sign_verify_round_trip() {
        let p256 = PrivateKey::from_pem(P256_PKCS8_PEM).unwrap();
        sign_and_verify(
            &p256,
            SignatureScheme::EcdsaSecp256r1Sha256,
            P256_SPKI,
            b"certificate verify 256",
        );
        // ECDSA signatures are DER-encoded SEQUENCEs.
        let sig = p256
            .sign(SignatureScheme::EcdsaSecp256r1Sha256, b"x")
            .unwrap();
        assert_eq!(sig[0], 0x30);

        let p384 = PrivateKey::from_pem(P384_SEC1_PEM).unwrap();
        sign_and_verify(
            &p384,
            SignatureScheme::EcdsaSecp384r1Sha384,
            P384_SPKI,
            b"certificate verify 384",
        );
    }

    #[test]
    fn rsa_sign_verify_round_trip_all_four_schemes() {
        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        let msg = b"server key exchange";
        for scheme in [
            SignatureScheme::RsaPssRsaeSha256,
            SignatureScheme::RsaPssRsaeSha384,
            SignatureScheme::RsaPkcs1Sha256,
            SignatureScheme::RsaPkcs1Sha384,
        ] {
            let sig = rsa.sign(scheme, msg).unwrap();
            // RSA-2048 signatures are 256 bytes regardless of scheme.
            assert_eq!(sig.len(), 256, "{scheme:?}");
            let spki = unhex(RSA_SPKI);
            verify_handshake_signature(scheme, &spki, msg, &sig).unwrap();
            assert!(verify_handshake_signature(scheme, &spki, b"other", &sig).is_err());
        }
    }

    #[test]
    fn pss_signatures_are_randomized() {
        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        let a = rsa
            .sign(SignatureScheme::RsaPssRsaeSha256, b"m")
            .unwrap();
        let b = rsa
            .sign(SignatureScheme::RsaPssRsaeSha256, b"m")
            .unwrap();
        // PSS padding includes a random salt: same input, different output.
        assert_ne!(a, b);
    }

    #[test]
    fn sign_rejects_scheme_key_mismatch() {
        let p256 = PrivateKey::from_pem(P256_SEC1_PEM).unwrap();
        for scheme in [
            SignatureScheme::Ed25519,
            SignatureScheme::RsaPkcs1Sha256,
            SignatureScheme::RsaPssRsaeSha256,
            SignatureScheme::EcdsaSecp384r1Sha384,
        ] {
            let err = p256.sign(scheme, b"m").unwrap_err();
            assert!(matches!(err, TlsError::Unsupported(_)), "{scheme:?}");
        }

        let ed = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let err = ed.sign(SignatureScheme::EcdsaSecp256r1Sha256, b"m").unwrap_err();
        assert!(matches!(err, TlsError::Unsupported(_)));

        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        let err = rsa.sign(SignatureScheme::Ed25519, b"m").unwrap_err();
        assert!(matches!(err, TlsError::Unsupported(_)));
    }

    #[test]
    fn tampered_signature_fails_verification() {
        let key = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let message = b"transcript hash";
        let mut sig = key.sign(SignatureScheme::Ed25519, message).unwrap();
        let spki = unhex(ED25519_SPKI);
        verify_handshake_signature(SignatureScheme::Ed25519, &spki, message, &sig).unwrap();
        sig[0] ^= 0x01;
        assert!(verify_handshake_signature(SignatureScheme::Ed25519, &spki, message, &sig).is_err());

        // Wrong-length Ed25519 signature is rejected before verification.
        let short = &sig[..63];
        assert!(verify_handshake_signature(SignatureScheme::Ed25519, &spki, message, short).is_err());
    }

    #[test]
    fn verify_rejects_garbage_spki_or_scheme_key_mismatch() {
        let p256 = PrivateKey::from_pem(P256_SEC1_PEM).unwrap();
        let sig = p256
            .sign(SignatureScheme::EcdsaSecp256r1Sha256, b"m")
            .unwrap();
        let garbage = [0u8; 16];
        // Not a valid SPKI at all.
        let err =
            verify_handshake_signature(SignatureScheme::EcdsaSecp256r1Sha256, &garbage, b"m", &sig)
                .unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // Valid SPKI but of the wrong key type for the scheme.
        let err =
            verify_handshake_signature(SignatureScheme::Ed25519, &unhex(P256_SPKI), b"m", &sig)
                .unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // RSA scheme against an EC SPKI.
        let err = verify_handshake_signature(
            SignatureScheme::RsaPkcs1Sha256,
            &unhex(P256_SPKI),
            b"m",
            &sig,
        )
        .unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // Malformed signature bytes (not DER, wrong length).
        let err = verify_handshake_signature(
            SignatureScheme::EcdsaSecp256r1Sha256,
            &unhex(P256_SPKI),
            b"m",
            &[0x30, 0x00],
        )
        .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::DecryptError));
    }

    #[test]
    fn tls13_cert_verify_content_layout() {
        let hash = [0x42u8; 32];
        let server = tls13_cert_verify_content(true, &hash);
        let client = tls13_cert_verify_content(false, &hash);

        // 64 spaces || context string || 0x00 || transcript hash.
        let server_ctx = b"TLS 1.3, server CertificateVerify";
        let client_ctx = b"TLS 1.3, client CertificateVerify";
        assert_eq!(server.len(), 64 + server_ctx.len() + 1 + 32);
        assert_eq!(client.len(), 64 + client_ctx.len() + 1 + 32);
        assert!(server[..64].iter().all(|&b| b == 0x20));
        assert_eq!(&server[64..64 + server_ctx.len()], server_ctx);
        assert_eq!(server[64 + server_ctx.len()], 0x00);
        assert_eq!(&server[server.len() - 32..], &hash);
        // Server and client context strings differ.
        assert_ne!(&server[64..97], &client[64..97]);
        // Deterministic.
        assert_eq!(server, tls13_cert_verify_content(true, &hash));
        // Different transcript hash changes only the tail.
        let other = tls13_cert_verify_content(true, &[0x43u8; 32]);
        assert_eq!(&server[..98], &other[..98]);
        assert_ne!(server, other);

        // An empty transcript hash still renders the 0x00 separator.
        let empty = tls13_cert_verify_content(true, &[]);
        assert_eq!(empty.len(), 64 + server_ctx.len() + 1);
        assert_eq!(*empty.last().unwrap(), 0x00);
    }

    #[test]
    fn tls13_cert_verify_sign_verify_round_trip() {
        let key = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let spki = unhex(ED25519_SPKI);
        let transcript_hash = [0x5Cu8; 32];

        let content = tls13_cert_verify_content(true, &transcript_hash);
        let sig = key.sign(SignatureScheme::Ed25519, &content).unwrap();
        verify_handshake_signature(SignatureScheme::Ed25519, &spki, &content, &sig).unwrap();

        // Signing the client variant with the client content verifies too,
        // but the server content must not verify as the client one.
        let client_content = tls13_cert_verify_content(false, &transcript_hash);
        assert!(verify_handshake_signature(
            SignatureScheme::Ed25519,
            &spki,
            &client_content,
            &sig
        )
        .is_err());
    }

    #[test]
    fn verify_cert_signature_dispatches_on_oid() {
        let hash = sha256(b"tbs certificate bytes");
        let tbs = b"tbs certificate bytes";

        // Ed25519 (1.3.101.112): pure signature, no digest.
        let ed = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let ed_sig = ed.sign(SignatureScheme::Ed25519, tbs).unwrap();
        verify_cert_signature(tbs, "1.3.101.112", &ed_sig, &unhex(ED25519_SPKI)).unwrap();
        assert!(
            verify_cert_signature(b"other", "1.3.101.112", &ed_sig, &unhex(ED25519_SPKI)).is_err()
        );

        // sha256WithRSAEncryption (1.2.840.113549.1.1.11).
        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        let rsa_sig = rsa.sign(SignatureScheme::RsaPkcs1Sha256, tbs).unwrap();
        verify_cert_signature(tbs, "1.2.840.113549.1.1.11", &rsa_sig, &unhex(RSA_SPKI)).unwrap();
        assert!(verify_cert_signature(
            b"tampered",
            "1.2.840.113549.1.1.11",
            &rsa_sig,
            &unhex(RSA_SPKI)
        )
        .is_err());

        // sha384WithRSAEncryption (1.2.840.113549.1.1.12).
        let rsa_sig384 = rsa.sign(SignatureScheme::RsaPkcs1Sha384, tbs).unwrap();
        verify_cert_signature(tbs, "1.2.840.113549.1.1.12", &rsa_sig384, &unhex(RSA_SPKI)).unwrap();

        // RSASSA-PSS (1.2.840.113549.1.1.10): tries SHA-256 then SHA-384.
        let pss_sig = rsa.sign(SignatureScheme::RsaPssRsaeSha256, tbs).unwrap();
        verify_cert_signature(tbs, "1.2.840.113549.1.1.10", &pss_sig, &unhex(RSA_SPKI)).unwrap();
        let pss_sig384 = rsa.sign(SignatureScheme::RsaPssRsaeSha384, tbs).unwrap();
        verify_cert_signature(tbs, "1.2.840.113549.1.1.10", &pss_sig384, &unhex(RSA_SPKI)).unwrap();

        // ecdsa-with-SHA256 (1.2.840.10045.4.3.2).
        let p256 = PrivateKey::from_pem(P256_PKCS8_PEM).unwrap();
        let ec_sig = p256
            .sign(SignatureScheme::EcdsaSecp256r1Sha256, tbs)
            .unwrap();
        verify_cert_signature(tbs, "1.2.840.10045.4.3.2", &ec_sig, &unhex(P256_SPKI)).unwrap();

        // ecdsa-with-SHA384 (1.2.840.10045.4.3.3).
        let p384 = PrivateKey::from_pem(P384_PKCS8_PEM).unwrap();
        let ec_sig384 = p384
            .sign(SignatureScheme::EcdsaSecp384r1Sha384, tbs)
            .unwrap();
        verify_cert_signature(tbs, "1.2.840.10045.4.3.3", &ec_sig384, &unhex(P384_SPKI)).unwrap();

        // Unknown OID.
        let err = verify_cert_signature(tbs, "1.2.3.4.5", &ed_sig, &unhex(ED25519_SPKI)).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        let _ = hash;
    }

    #[test]
    fn verify_cert_signature_rejects_issuer_key_mismatch() {
        let tbs = b"tbs certificate bytes";
        let ed = PrivateKey::from_pem(ED25519_PEM).unwrap();
        let ed_sig = ed.sign(SignatureScheme::Ed25519, tbs).unwrap();
        // Ed25519 signature against a P-256 issuer SPKI.
        let err = verify_cert_signature(tbs, "1.3.101.112", &ed_sig, &unhex(P256_SPKI)).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // RSA signature against an RSA SPKI but with a corrupted signature.
        let rsa = PrivateKey::from_pem(RSA_PKCS1_PEM).unwrap();
        let mut rsa_sig = rsa.sign(SignatureScheme::RsaPkcs1Sha256, tbs).unwrap();
        let last = rsa_sig.len() - 1;
        rsa_sig[last] ^= 0xFF;
        let err =
            verify_cert_signature(tbs, "1.2.840.113549.1.1.11", &rsa_sig, &unhex(RSA_SPKI))
                .unwrap_err();
        assert!(matches!(err, TlsError::Alert(AlertDescription::DecryptError)));
    }

    #[test]
    fn sha256_helper_matches_reference_digest() {
        let digest = sha256(b"abc");
        let expected = unhex(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        assert_eq!(digest.to_vec(), expected);
        // Length is always 32 bytes.
        assert_eq!(sha256(b"").len(), 32);
        assert_ne!(sha256(b""), sha256(b"a"));
    }
}
