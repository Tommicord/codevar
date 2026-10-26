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

use crate::ids::{AeadAlgorithm, CipherSuite};
use spin::once::Once;

/// Coarse device class used for cipher-suite preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// Phones / tablets (Android, iOS).
    Mobile,
    /// Browser / WASM runtime (often mobile-class CPUs, no AES-NI).
    Web,
    /// Desktop / server with AES hardware acceleration.
    DesktopHw,
    /// Desktop / server without AES hardware (software AES).
    DesktopSw,
}

/// Cached capability snapshot.
#[derive(Debug, Clone, Copy)]
pub struct CryptoCapabilities {
    /// Coarse device class.
    pub device: DeviceClass,
    /// AES block encryption is hardware-accelerated.
    pub aes_hw: bool,
    /// Carry-less multiply for GHASH is hardware-accelerated.
    pub clmul_hw: bool,
}

impl CryptoCapabilities {
    /// AEAD that should be offered first for this device.
    #[must_use]
    pub const fn preferred_aead(self) -> AeadAlgorithm {
        if self.aes_hw && self.clmul_hw {
            AeadAlgorithm::Aes256Gcm
        } else {
            AeadAlgorithm::ChaCha20Poly1305
        }
    }

    /// True when AES-GCM is expected to outperform ChaCha20-Poly1305.
    #[must_use]
    pub const fn prefer_aes_gcm(self) -> bool {
        matches!(self.preferred_aead(), AeadAlgorithm::Aes256Gcm)
    }
}

static CAPS: Once<CryptoCapabilities> = Once::new();

/// Returns the process-wide crypto capability snapshot.
#[must_use]
pub fn capabilities() -> CryptoCapabilities {
    *CAPS.call_once(detect_capabilities)
}

/// Detects device class and AES / CLMUL hardware support.
#[must_use]
pub fn detect_capabilities() -> CryptoCapabilities {
    let aes_hw = detect_aes_hw();
    let clmul_hw = detect_clmul_hw();
    let device = detect_device_class(aes_hw && clmul_hw);
    CryptoCapabilities {
        device,
        aes_hw,
        clmul_hw,
    }
}

fn detect_device_class(aes_gcm_hw: bool) -> DeviceClass {
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let _ = aes_gcm_hw;
        DeviceClass::Mobile
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = aes_gcm_hw;
        DeviceClass::Web
    }
    #[cfg(not(any(target_os = "android", target_os = "ios", target_arch = "wasm32")))]
    {
        if aes_gcm_hw {
            DeviceClass::DesktopHw
        } else {
            DeviceClass::DesktopSw
        }
    }
}

fn detect_aes_hw() -> bool {
    codevar_base::cpu_feature!(aes)
}

fn detect_clmul_hw() -> bool {
    // PMULL (the AArch64 carry-less multiply extension) is reported as the
    // AArch64 equivalent of `pclmulqdq` by the shared CPU feature probe.
    codevar_base::cpu_feature!(pclmulqdq)
}

/// Cipher suites ordered for the current device (TLS 1.3 first, then TLS 1.2).
#[must_use]
pub fn preferred_cipher_suites() -> &'static [CipherSuite] {
    if capabilities().prefer_aes_gcm() {
        &AES_FIRST_SUITES
    } else {
        &CHACHA_FIRST_SUITES
    }
}

/// Preferred AEAD for traffic on this device.
#[must_use]
pub fn preferred_aead() -> AeadAlgorithm {
    capabilities().preferred_aead()
}

const AES_FIRST_SUITES: [CipherSuite; 9] = [
    CipherSuite::TlsAes256GcmSha384,
    CipherSuite::TlsAes128GcmSha256,
    CipherSuite::TlsChacha20Poly1305Sha256,
    CipherSuite::TlsEcdheEcdsaWithAes256GcmSha384,
    CipherSuite::TlsEcdheRsaWithAes256GcmSha384,
    CipherSuite::TlsEcdheEcdsaWithAes128GcmSha256,
    CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
    CipherSuite::TlsEcdheEcdsaWithChacha20Poly1305Sha256,
    CipherSuite::TlsEcdheRsaWithChacha20Poly1305Sha256,
];

const CHACHA_FIRST_SUITES: [CipherSuite; 9] = [
    CipherSuite::TlsChacha20Poly1305Sha256,
    CipherSuite::TlsAes256GcmSha384,
    CipherSuite::TlsAes128GcmSha256,
    CipherSuite::TlsEcdheEcdsaWithChacha20Poly1305Sha256,
    CipherSuite::TlsEcdheRsaWithChacha20Poly1305Sha256,
    CipherSuite::TlsEcdheEcdsaWithAes256GcmSha384,
    CipherSuite::TlsEcdheRsaWithAes256GcmSha384,
    CipherSuite::TlsEcdheEcdsaWithAes128GcmSha256,
    CipherSuite::TlsEcdheRsaWithAes128GcmSha256,
];
