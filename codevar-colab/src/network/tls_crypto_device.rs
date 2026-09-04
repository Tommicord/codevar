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

use crate::network::tls_ids::{AeadAlgorithm, CipherSuite};
use std::sync::OnceLock;

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

static CAPS: OnceLock<CryptoCapabilities> = OnceLock::new();

/// Returns the process-wide crypto capability snapshot.
#[must_use]
pub fn capabilities() -> CryptoCapabilities {
    *CAPS.get_or_init(detect_capabilities)
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
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        std::is_x86_feature_detected!("aes")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("aes")
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64"
    )))]
    {
        false
    }
}

fn detect_clmul_hw() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        std::is_x86_feature_detected!("pclmulqdq")
    }
    #[cfg(target_arch = "aarch64")]
    {
        // PMULL ships with the ARM crypto extension set used for AES.
        std::arch::is_aarch64_feature_detected!("aes")
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64"
    )))]
    {
        false
    }
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
