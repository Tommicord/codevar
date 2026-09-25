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

//! (EC)DHE key exchange (RFC 8446 §4.2.8, RFC 8422).

use crate::tls_crypto_random::{SysRng, fill_random};
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::NamedGroup;
use elliptic_curve::sec1::ToEncodedPoint;
use p256::ecdh::diffie_hellman as p256_dh;
use p384::ecdh::diffie_hellman as p384_dh;
use x25519_dalek::{PublicKey as X25519Public, StaticSecret};
use zeroize::Zeroize;

/// Ephemeral private key share.
pub enum KeySharePrivate {
    /// X25519 scalar.
    X25519(StaticSecret),
    /// P-256 secret key.
    P256(p256::SecretKey),
    /// P-384 secret key.
    P384(p384::SecretKey),
}

/// Serialized public share (wire format).
#[derive(Clone, Debug)]
pub struct KeySharePublic {
    /// Named group.
    pub group: NamedGroup,
    /// Key exchange bytes (uncompressed SEC1 for NIST curves, raw for X25519).
    pub key_exchange: Vec<u8>,
}

/// Generates an ephemeral key share for `group`.
pub fn generate_key_share(group: NamedGroup) -> TlsResult<(KeySharePrivate, KeySharePublic)> {
    let mut rng = SysRng;
    match group {
        NamedGroup::X25519 => {
            let mut seed = [0u8; 32];
            fill_random(&mut seed)?;
            let secret = StaticSecret::from(seed);
            seed.zeroize();
            let public = X25519Public::from(&secret);
            let key_exchange = public.as_bytes().to_vec();
            Ok((
                KeySharePrivate::X25519(secret),
                KeySharePublic { group, key_exchange },
            ))
        }
        NamedGroup::Secp256r1 => {
            let secret = p256::SecretKey::random(&mut rng);
            let public = secret.public_key();
            let encoded = public.to_encoded_point(false);
            Ok((
                KeySharePrivate::P256(secret),
                KeySharePublic {
                    group,
                    key_exchange: encoded.as_bytes().to_vec(),
                },
            ))
        }
        NamedGroup::Secp384r1 => {
            let secret = p384::SecretKey::random(&mut rng);
            let public = secret.public_key();
            let encoded = public.to_encoded_point(false);
            Ok((
                KeySharePrivate::P384(secret),
                KeySharePublic {
                    group,
                    key_exchange: encoded.as_bytes().to_vec(),
                },
            ))
        }
    }
}

/// Computes the shared secret with a peer public share.
pub fn shared_secret(private: &KeySharePrivate, peer: &KeySharePublic) -> TlsResult<Vec<u8>> {
    match (private, peer.group) {
        (KeySharePrivate::X25519(secret), NamedGroup::X25519) => {
            if peer.key_exchange.len() != 32 {
                return Err(TlsError::Alert(
                    crate::tls_alert::AlertDescription::IllegalParameter,
                ));
            }
            let mut pk_bytes = [0u8; 32];
            pk_bytes.copy_from_slice(&peer.key_exchange);
            let peer_pk = X25519Public::from(pk_bytes);
            let shared = secret.diffie_hellman(&peer_pk);
            Ok(shared.as_bytes().to_vec())
        }
        (KeySharePrivate::P256(secret), NamedGroup::Secp256r1) => {
            let peer_pk = p256::PublicKey::from_sec1_bytes(&peer.key_exchange)
                .map_err(|_| TlsError::Alert(crate::tls_alert::AlertDescription::IllegalParameter))?;
            let shared = p256_dh(secret.to_nonzero_scalar(), peer_pk.as_affine());
            Ok(shared.raw_secret_bytes().to_vec())
        }
        (KeySharePrivate::P384(secret), NamedGroup::Secp384r1) => {
            let peer_pk = p384::PublicKey::from_sec1_bytes(&peer.key_exchange)
                .map_err(|_| TlsError::Alert(crate::tls_alert::AlertDescription::IllegalParameter))?;
            let shared = p384_dh(secret.to_nonzero_scalar(), peer_pk.as_affine());
            Ok(shared.raw_secret_bytes().to_vec())
        }
        _ => Err(TlsError::Alert(
            crate::tls_alert::AlertDescription::IllegalParameter,
        )),
    }
}
