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

//! System metrics for the Win32 decoration backend.
//!
//! Queries `GetSystemMetrics` and `SystemParametersInfoW` for the values a
//! decoration reports: title bar height, border thickness, icon sizes and
//! the double-click distance/time.

use codevar_kuitui_decoration::DecorationMetrics;

/// Reads the system metrics the decoration reports.
///
/// On non-Windows targets every metric falls back to its default.
#[must_use]
pub fn decoration_metrics() -> DecorationMetrics {
    #[cfg(windows)]
    return windows_metrics();
    #[cfg(not(windows))]
    return DecorationMetrics::default();
}

#[cfg(windows)]
fn windows_metrics() -> DecorationMetrics {
    use windows::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, SM_CXBORDER, SM_CXDOUBLECLK, SM_CXICON, SM_CYCAPTION,
    };

    let title_bar_height = metric_or_default(unsafe { GetSystemMetrics(SM_CYCAPTION) }, 32);
    let border_thickness = metric_or_default(unsafe { GetSystemMetrics(SM_CXBORDER) }, 6);
    let small_icon_size = metric_or_default(unsafe { GetSystemMetrics(SM_CXICON) }, 16);
    let large_icon_size = metric_or_default(unsafe { GetSystemMetrics(SM_CXICON) }, 32);
    let double_click_distance = metric_or_default(unsafe { GetSystemMetrics(SM_CXDOUBLECLK) }, 4);
    let double_click_time = metric_or_default(unsafe { GetSystemMetrics(SM_CXDOUBLECLK) }, 500);

    DecorationMetrics {
        title_bar_height: title_bar_height as u32,
        border_thickness: border_thickness as u32,
        small_icon_size: small_icon_size as u32,
        large_icon_size: large_icon_size as u32,
        double_click_distance: double_click_distance as u32,
        double_click_time: double_click_time as u32,
    }
}

/// Returns `value` when it is positive, otherwise `default`.
///
/// `GetSystemMetrics` returns `0` when a metric is not available, so a
/// sensible default keeps the decoration usable.
#[must_use]
fn metric_or_default(value: i32, default: i32) -> i32 {
    if value > 0 { value } else { default }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_have_sensible_defaults() {
        let metrics = decoration_metrics();
        assert!(metrics.title_bar_height > 0);
        assert!(metrics.border_thickness > 0);
        assert!(metrics.small_icon_size > 0);
        assert!(metrics.large_icon_size > 0);
    }
}
