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

//! CSPRNG helpers and device-aware AEAD preference.

use crate::network::tls_crypto_device::{
    CryptoCapabilities, DeviceClass, capabilities, preferred_aead,
    preferred_cipher_suites,
};
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_ids::{AeadAlgorithm, CipherSuite};

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
        (DeviceClass::Mobile, AeadAlgorithm::ChaCha20Poly1305) => {
            "mobile / ChaCha20-Poly1305"
        }
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
