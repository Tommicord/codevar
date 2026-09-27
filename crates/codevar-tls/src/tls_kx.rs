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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_alert::AlertDescription;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn rfc7748_x25519_diffie_hellman_vector() {
        // RFC 7748 §5.2 test vector 1.
        let alice = unhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let alice_pub = unhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
        let bob = unhex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        let bob_pub = unhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f");
        let shared = unhex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");

        let alice_seed: [u8; 32] = alice.as_slice().try_into().unwrap();
        let bob_seed: [u8; 32] = bob.as_slice().try_into().unwrap();
        let alice_static = StaticSecret::from(alice_seed);
        let bob_static = StaticSecret::from(bob_seed);

        // Public keys derived from the private scalars match the RFC.
        assert_eq!(
            X25519Public::from(&alice_static)
                .as_bytes()
                .to_vec(),
            alice_pub
        );
        assert_eq!(
            X25519Public::from(&bob_static)
                .as_bytes()
                .to_vec(),
            bob_pub
        );

        let alice_secret = KeySharePrivate::X25519(alice_static);
        let bob_secret = KeySharePrivate::X25519(bob_static);

        // Shared secret, both directions.
        let k1 = shared_secret(
            &alice_secret,
            &KeySharePublic {
                group: NamedGroup::X25519,
                key_exchange: bob_pub.clone(),
            },
        )
        .unwrap();
        let k2 = shared_secret(
            &bob_secret,
            &KeySharePublic {
                group: NamedGroup::X25519,
                key_exchange: alice_pub,
            },
        )
        .unwrap();
        assert_eq!(k1, shared);
        assert_eq!(k2, shared);
        assert_eq!(k1.len(), 32);
    }

    #[test]
    fn generate_key_share_lengths_and_group_tags() {
        let (priv_key, pub_key) = generate_key_share(NamedGroup::X25519).unwrap();
        assert!(matches!(priv_key, KeySharePrivate::X25519(_)));
        assert_eq!(pub_key.group, NamedGroup::X25519);
        assert_eq!(pub_key.key_exchange.len(), 32);

        let (priv_key, pub_key) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        assert!(matches!(priv_key, KeySharePrivate::P256(_)));
        assert_eq!(pub_key.group, NamedGroup::Secp256r1);
        // Uncompressed SEC1 point: 0x04 || X || Y.
        assert_eq!(pub_key.key_exchange.len(), 65);
        assert_eq!(pub_key.key_exchange[0], 0x04);

        let (priv_key, pub_key) = generate_key_share(NamedGroup::Secp384r1).unwrap();
        assert!(matches!(priv_key, KeySharePrivate::P384(_)));
        assert_eq!(pub_key.group, NamedGroup::Secp384r1);
        assert_eq!(pub_key.key_exchange.len(), 97);
        assert_eq!(pub_key.key_exchange[0], 0x04);
    }

    #[test]
    fn generated_shares_agree_in_both_directions() {
        for group in [NamedGroup::X25519, NamedGroup::Secp256r1, NamedGroup::Secp384r1] {
            let (a_priv, a_pub) = generate_key_share(group).unwrap();
            let (b_priv, b_pub) = generate_key_share(group).unwrap();
            let s1 = shared_secret(&a_priv, &b_pub).unwrap();
            let s2 = shared_secret(&b_priv, &a_pub).unwrap();
            assert_eq!(s1, s2, "DH mismatch for {group:?}");
            let expected_len = match group {
                NamedGroup::X25519 => 32,
                NamedGroup::Secp256r1 => 32,
                NamedGroup::Secp384r1 => 48,
            };
            assert_eq!(s1.len(), expected_len);
            // Self-agreement differs from peer agreement (different key pairs).
            let s3 = shared_secret(&a_priv, &a_pub).unwrap();
            assert_ne!(s1, s3);
        }
    }

    #[test]
    fn fresh_key_shares_are_randomized() {
        let (_, p1) = generate_key_share(NamedGroup::X25519).unwrap();
        let (_, p2) = generate_key_share(NamedGroup::X25519).unwrap();
        assert_ne!(p1.key_exchange, p2.key_exchange);

        let (_, e1) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        let (_, e2) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        assert_ne!(e1.key_exchange, e2.key_exchange);
    }

    #[test]
    fn mismatched_group_is_rejected() {
        let (a_priv, _) = generate_key_share(NamedGroup::X25519).unwrap();
        let (_, p256_pub) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        let err = shared_secret(&a_priv, &p256_pub).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));

        let (b_priv, _) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        let (_, x_pub) = generate_key_share(NamedGroup::X25519).unwrap();
        let err = shared_secret(&b_priv, &x_pub).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));

        let (c_priv, _) = generate_key_share(NamedGroup::Secp384r1).unwrap();
        let (_, p256_pub) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        let err = shared_secret(&c_priv, &p256_pub).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
    }

    #[test]
    fn x25519_peer_key_must_be_32_bytes() {
        let (priv_key, _) = generate_key_share(NamedGroup::X25519).unwrap();
        for len in [0usize, 1, 31, 33, 64] {
            let peer = KeySharePublic {
                group: NamedGroup::X25519,
                key_exchange: vec![0x09; len],
            };
            let err = shared_secret(&priv_key, &peer).unwrap_err();
            assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
        }
    }

    #[test]
    fn nist_peer_key_must_be_valid_sec1_point() {
        let (priv_key, _) = generate_key_share(NamedGroup::Secp256r1).unwrap();
        let peer = KeySharePublic {
            group: NamedGroup::Secp256r1,
            key_exchange: vec![0xFF; 65],
        };
        let err = shared_secret(&priv_key, &peer).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));

        // Truncated SEC1 point.
        let peer = KeySharePublic {
            group: NamedGroup::Secp256r1,
            key_exchange: vec![0x04; 33],
        };
        let err = shared_secret(&priv_key, &peer).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));

        let (priv384, _) = generate_key_share(NamedGroup::Secp384r1).unwrap();
        let peer = KeySharePublic {
            group: NamedGroup::Secp384r1,
            key_exchange: vec![0x00; 97],
        };
        let err = shared_secret(&priv384, &peer).unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::IllegalParameter));
    }
}
