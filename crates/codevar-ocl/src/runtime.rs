//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Entry point of the OpenCL runtime.
//!
//! [`Runtime::load`] is the normal way in: it opens the system ICD loader,
//! and when that yields no platforms it falls back to scanning vendor
//! libraries. [`Runtime::load_direct`] skips the loader and forces the
//! vendor scan, which is useful for diagnosing a broken installation (and
//! for shipping a driver next to the application without registering it).
//!
//! A successful `Runtime` always owns at least one [`Platform`].

use alloc::ffi::CString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use codevar_logger::{log_debug, log_warn};

use crate::api::Api;
use crate::error::{Error, Result};
use crate::icd;
use crate::loader::{self, Library};
use crate::platform::Platform;

/// An opened OpenCL installation: every platform discovered on the machine
/// plus the resolved entry points of the driver that exposes them.
///
/// The runtime keeps the driver libraries open for the process lifetime
/// (see [`crate::loader`]); dropping a [`Runtime`] only releases the
/// platform handles' wrappers, not the underlying driver.
pub struct Runtime {
    platforms: Vec<Platform>,
}

impl Runtime {
    /// Loads OpenCL from the system installation.
    ///
    /// Order: the Khronos/client ICD loader (`libOpenCL.so.1`,
    /// `OpenCL.dll`, the macOS framework) first, and — when it is absent
    /// or reports zero platforms — a scan of the vendor libraries listed
    /// in the module documentation of [`crate::icd`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoVendorLibraries`] when no candidate library
    /// could be opened at all, and [`Error::NoPlatforms`] when libraries
    /// opened but none exposed a usable platform.
    pub fn load() -> Result<Self> {
        match loader::open_first(loader::client_candidates()) {
            Ok((library, origin)) => {
                log_debug!("opened the OpenCL client loader {origin}");
                match Api::from_symbols(&library) {
                    Ok(api) => {
                        let api = Arc::new(api);
                        match icd::query_platforms(&api) {
                            Ok(platforms) if !platforms.is_empty() => {
                                return Ok(Self { platforms });
                            }
                            Ok(_) => {
                                log_debug!(
                                    "the client loader {origin} reported no platforms; \
                                     scanning vendor libraries"
                                );
                            }
                            Err(error) => {
                                log_warn!("platform enumeration through {origin} failed: {error}");
                            }
                        }
                    }
                    Err(error) => {
                        log_warn!("the client loader {origin} lacks the core API: {error}");
                    }
                }
            }
            Err(error) => {
                log_debug!("no OpenCL client loader found: {error}");
            }
        }
        Self::load_direct()
    }

    /// Loads OpenCL by scanning vendor driver libraries directly.
    ///
    /// The client loader is never consulted; every candidate from
    /// [`crate::icd::candidates`] is opened and probed first through the
    /// full `cl*` API (client loaders and legacy vendors) and then through
    /// the vendor ICD protocol (dispatch table per platform).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoVendorLibraries`] when no candidate could be
    /// opened and [`Error::NoPlatforms`] when every opened library yielded
    /// zero platforms.
    pub fn load_direct() -> Result<Self> {
        let candidates = icd::candidates();
        if candidates.is_empty() {
            return Err(Error::NoVendorLibraries);
        }
        let mut opened_any = false;
        for candidate in &candidates {
            let Ok(c_path) = CString::new(candidate.path.as_str()) else {
                log_warn!(
                    "skipping vendor candidate {}: the path contains an interior NUL",
                    candidate.path
                );
                continue;
            };
            let library = match Library::open(&c_path) {
                Ok(library) => library,
                Err(error) => {
                    log_debug!(
                        "skipping vendor candidate {} ({}): {error}",
                        candidate.path,
                        candidate.origin
                    );
                    continue;
                }
            };
            opened_any = true;
            match Self::platforms_from_library(&library) {
                Ok(platforms) if !platforms.is_empty() => {
                    log_debug!(
                        "using vendor candidate {} ({}), {} platform(s)",
                        candidate.path,
                        candidate.origin,
                        platforms.len()
                    );
                    return Ok(Self { platforms });
                }
                Ok(_) => {
                    log_debug!("vendor candidate {} reported no platforms", candidate.path);
                }
                Err(error) => {
                    log_debug!("vendor candidate {} is unusable: {error}", candidate.path);
                }
            }
        }
        if opened_any {
            Err(Error::NoPlatforms)
        } else {
            Err(Error::NoVendorLibraries)
        }
    }

    /// Probes one opened library through both supported protocols,
    /// preferring the full `cl*` API and falling back to the vendor ICD
    /// dispatch table. Returns an empty vector when the library exposes no
    /// platforms (an error is returned only when it exposes no usable
    /// protocol at all).
    fn platforms_from_library(library: &Library) -> Result<Vec<Platform>> {
        match Api::from_symbols(library) {
            Ok(api) => icd::query_platforms(&Arc::new(api)),
            // Expected for modern vendor ICDs: they export only
            // `clIcdGetPlatformIDsKHR` plus the dispatch table.
            Err(Error::MissingSymbol { .. }) => icd::vendor_platforms(library),
            // The library is not an OpenCL implementation at all.
            Err(error) => Err(error),
        }
    }

    /// Returns the platforms discovered by this runtime.
    #[must_use]
    pub fn platforms(&self) -> &[Platform] {
        &self.platforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    /// `Ok` runtimes must always expose platforms; machines without a
    /// working OpenCL stack must fail with an error instead of producing
    /// an empty runtime.
    #[test]
    fn an_ok_runtime_always_reports_at_least_one_platform() {
        if let Ok(runtime) = Runtime::load() {
            assert!(!runtime.platforms().is_empty());
        }
        if let Ok(runtime) = Runtime::load_direct() {
            assert!(!runtime.platforms().is_empty());
        }
    }

    /// `load` and `load_direct` must agree on machines with a single
    /// installation path (whichever succeeds reports the same platform
    /// names when both succeed).
    #[test]
    fn load_and_load_direct_agree_when_both_succeed() {
        let (Ok(direct), Ok(via_loader)) = (Runtime::load_direct(), Runtime::load()) else {
            return;
        };
        let loader_names: Vec<String> = via_loader
            .platforms()
            .iter()
            .filter_map(|platform| platform.name().ok())
            .collect();
        let direct_names: Vec<String> = direct
            .platforms()
            .iter()
            .filter_map(|platform| platform.name().ok())
            .collect();
        assert_eq!(loader_names, direct_names);
    }
}
