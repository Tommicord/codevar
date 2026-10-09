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

//! CSPRNG helpers and device-aware AEAD preference.

use crate::crypto_device::{
    CryptoCapabilities, DeviceClass, capabilities, preferred_aead, preferred_cipher_suites,
};
use crate::error::{TlsError, TlsResult};
use crate::ids::{AeadAlgorithm, CipherSuite};

/// Fills `buf` with cryptographically secure random bytes.
pub fn fill_random(buf: &mut [u8]) -> TlsResult<()> {
    getrandom::getrandom(buf).map_err(|_| TlsError::RandomFailed)
}

/// Returns a random array of `N` bytes.
pub fn random_array<const N: usize>() -> TlsResult<[u8; N]> {
    let mut buf = [0u8; N];
    fill_random(&mut buf)?;
    Ok(buf)
}

/// Detects the local device class and crypto capabilities used for AEAD selection.
#[must_use]
pub fn detect_device() -> CryptoCapabilities {
    capabilities()
}

/// Returns the preferred AEAD for this device (AES-256-GCM or ChaCha20-Poly1305).
#[must_use]
pub fn best_aead_for_device() -> AeadAlgorithm {
    preferred_aead()
}

/// Returns cipher suites ordered for the current device.
#[must_use]
pub fn device_cipher_suites() -> &'static [CipherSuite] {
    preferred_cipher_suites()
}

/// Human-readable summary of the selected crypto profile.
#[must_use]
pub fn device_crypto_profile() -> &'static str {
    let caps = capabilities();
    match (caps.device, caps.preferred_aead()) {
        (DeviceClass::Mobile, AeadAlgorithm::ChaCha20Poly1305) => "mobile / ChaCha20-Poly1305",
        (DeviceClass::Mobile, AeadAlgorithm::Aes256Gcm) => "mobile / AES-256-GCM (hw)",
        (DeviceClass::Web, AeadAlgorithm::ChaCha20Poly1305) => "web / ChaCha20-Poly1305",
        (DeviceClass::Web, AeadAlgorithm::Aes256Gcm) => "web / AES-256-GCM",
        (DeviceClass::DesktopHw, _) => "desktop / AES-256-GCM (AES-NI)",
        (DeviceClass::DesktopSw, AeadAlgorithm::ChaCha20Poly1305) => {
            "desktop / ChaCha20-Poly1305 (no AES-NI)"
        }
        (DeviceClass::DesktopSw, _) => "desktop / AES-256-GCM",
        (_, AeadAlgorithm::Aes128Gcm) => "AES-128-GCM",
    }
}

/// `rand_core::RngCore` backed by `getrandom`, used for PSS and key generation.
pub struct SysRng;

impl rand_core::RngCore for SysRng {
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        let _ = getrandom::getrandom(&mut b);
        u32::from_le_bytes(b)
    }

    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        let _ = getrandom::getrandom(&mut b);
        u64::from_le_bytes(b)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let _ = getrandom::getrandom(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        getrandom::getrandom(dest).map_err(rand_core::Error::from)
    }
}

impl rand_core::CryptoRng for SysRng {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_device::DeviceClass;

    #[test]
    fn fill_random_writes_distinct_bytes() {
        let mut a = [0xAAu8; 64];
        let mut b = [0xAAu8; 64];
        fill_random(&mut a).unwrap();
        fill_random(&mut b).unwrap();
        // The prefill must be overwritten.
        assert_ne!(a, [0xAAu8; 64]);
        assert_ne!(b, [0xAAu8; 64]);
        // Two independent draws must differ (2^-512 collision probability).
        assert_ne!(a, b);
        // Not all zero.
        assert!(a.iter().any(|&x| x != 0));
    }

    #[test]
    fn fill_random_accepts_empty_and_single_byte_buffers() {
        fill_random(&mut []).unwrap();
        let mut one = [0u8; 1];
        fill_random(&mut one).unwrap();
        let mut two = [0u8; 2];
        fill_random(&mut two).unwrap();
        // 1-in-256 chance of flaking; two draws make it negligible.
        let mut again = [0u8; 2];
        fill_random(&mut again).unwrap();
        assert_ne!(two, again);
    }

    #[test]
    fn random_array_returns_requested_length_and_varies() {
        let empty: [u8; 0] = random_array().unwrap();
        assert!(empty.is_empty());
        let a: [u8; 32] = random_array().unwrap();
        let b: [u8; 32] = random_array().unwrap();
        assert_ne!(a, b);
        let big: [u8; 256] = random_array().unwrap();
        assert!(big.iter().any(|&x| x != 0));
        let v: [u8; 1] = random_array().unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn device_preferences_are_internally_consistent() {
        let caps = detect_device();
        let best = best_aead_for_device();
        assert_eq!(best, caps.preferred_aead());

        let suites = device_cipher_suites();
        assert!(!suites.is_empty());
        // The top-offered suite uses the device's preferred AEAD.
        assert_eq!(suites[0].aead_algorithm(), best);
        // Suite preference is exactly what the config layer offers.
        assert_eq!(suites, crate::ids::CipherSuite::default_offered());

        // Preference is driven purely by AES/CLMUL hardware flags.
        let expected = if caps.aes_hw && caps.clmul_hw {
            AeadAlgorithm::Aes256Gcm
        } else {
            AeadAlgorithm::ChaCha20Poly1305
        };
        assert_eq!(best, expected);
        assert_eq!(caps.prefer_aes_gcm(), best == AeadAlgorithm::Aes256Gcm);
    }

    #[test]
    fn device_crypto_profile_describes_capabilities() {
        let caps = detect_device();
        let profile = device_crypto_profile();
        assert!(!profile.is_empty());
        // Stable across calls (derived from the cached snapshot).
        assert_eq!(profile, device_crypto_profile());

        match caps.device {
            DeviceClass::Mobile => assert!(profile.starts_with("mobile"), "{profile}"),
            DeviceClass::Web => assert!(profile.starts_with("web"), "{profile}"),
            DeviceClass::DesktopHw | DeviceClass::DesktopSw => {
                assert!(profile.starts_with("desktop"), "{profile}");
            }
        }
        // The AEAD half of the description matches the selection.
        match caps.preferred_aead() {
            AeadAlgorithm::ChaCha20Poly1305 => assert!(profile.contains("ChaCha"), "{profile}"),
            AeadAlgorithm::Aes256Gcm => assert!(profile.contains("AES-256"), "{profile}"),
            AeadAlgorithm::Aes128Gcm => assert!(profile.contains("AES-128"), "{profile}"),
        }
        // AES hardware acceleration is reflected in the wording.
        if caps.device == DeviceClass::DesktopHw {
            assert!(profile.contains("AES-NI"), "{profile}");
        }
    }

    #[test]
    fn sys_rng_satisfies_rand_core_interface() {
        use rand_core::RngCore;

        let mut rng = SysRng;
        let a = rng.next_u32();
        let b = rng.next_u32();
        assert_ne!(a, b);

        let x = rng.next_u64();
        let y = rng.next_u64();
        assert_ne!(x, y);

        let mut buf1 = [0u8; 32];
        let mut buf2 = [0u8; 32];
        rng.fill_bytes(&mut buf1);
        rng.fill_bytes(&mut buf2);
        assert_ne!(buf1, buf2);
        assert!(buf1.iter().any(|&v| v != 0));

        let mut dest = [0u8; 16];
        rng.try_fill_bytes(&mut dest).unwrap();
        let mut other = [0u8; 16];
        rng.try_fill_bytes(&mut other).unwrap();
        assert_ne!(dest, other);
    }

    #[test]
    fn sys_rng_is_marked_as_crypto_rng() {
        use rand_core::{CryptoRng, RngCore};
        fn assert_crypto_rng<T: CryptoRng>() {}
        assert_crypto_rng::<SysRng>();
        // Used as the RNG for ECDSA / RSA-PSS signing in this crate.
        let mut rng = SysRng;
        let sig_rng_bytes: [u8; 48] = random_array().unwrap();
        assert_ne!(sig_rng_bytes, [0u8; 48]);
        let mut via_rng = [0u8; 48];
        rng.fill_bytes(&mut via_rng);
        assert_ne!(via_rng, sig_rng_bytes);
    }
}
