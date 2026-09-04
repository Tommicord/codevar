//! Copyright 2026 Codevar
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

//! Telemetry configuration using integer thresholds with dynamic hardware detection.

/// Basis points for percentage calculations (10000 = 100%)
const BASIS_POINTS: u32 = 10000;

/// Scale for millisecond calculations
const MILLI_SCALE: u64 = 1000;

/// Nanoseconds per second
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Nanoseconds per millisecond
const NANOS_PER_MILLI: u64 = 1_000_000;

/// CPU power score thresholds
const CPU_POWER_SCORE_LOW: u64 = 30;
const CPU_POWER_SCORE_MEDIUM: u64 = 60;
const CPU_POWER_SCORE_HIGH_KEY_CLASSIFICATION: u64 = 40;
const CPU_POWER_SCORE_HIGH_ADAPTIVE: u64 = 30;

/// Window duration configurations (nanoseconds)
const WINDOW_DURATION_LOW_POWER: u64 = 7_000_000_000;
const WINDOW_DURATION_MEDIUM_POWER: u64 = 5_000_000_000;
const WINDOW_DURATION_HIGH_POWER: u64 = 3_000_000_000;

/// Sub-window configuration limits
const SUB_WINDOWS_MIN: u32 = 4;
const SUB_WINDOWS_MAX: u32 = 32;

/// Burst threshold configuration
const BURST_THRESHOLD_BASE: u64 = 15_000;
const BURST_THRESHOLD_MULTIPLIER: u64 = 200;

/// Cache size configuration (bytes)
const CACHE_SIZE_BASE_MULTIPLIER: u64 = 64;
const CACHE_SIZE_MIN: usize = 256;
const CACHE_SIZE_MAX: usize = 16384;
const CACHE_SIZE_DIVISOR: u64 = 4;

/// Memory ratio threshold for adaptive sizing (basis points)
const MEMORY_RATIO_THRESHOLD_ADAPTIVE: u64 = 7000;

/// Core count scoring
const CORE_SCORE_MULTIPLIER: u64 = 10;
const CORE_SCORE_MAX: u64 = 40;

/// SMT bonus
const SMT_BONUS: u64 = 10;

/// CPU frequency thresholds (MHz)
const CPU_FREQ_HIGH: u32 = 3000;
const CPU_FREQ_MEDIUM: u32 = 2000;
const CPU_FREQ_HIGH_BONUS: u64 = 20;
const CPU_FREQ_MEDIUM_BONUS: u64 = 10;

/// Vendor bonuses
const VENDOR_BONUS_INTEL_AMD: u64 = 10;
const VENDOR_BONUS_APPLE: u64 = 15;
const VENDOR_BONUS_ARM: u64 = 5;

/// Memory thresholds (GB)
const MEMORY_HIGH_GB: u64 = 16;
const MEMORY_MEDIUM_GB: u64 = 8;
const MEMORY_HIGH_BONUS: u64 = 10;
const MEMORY_MEDIUM_BONUS: u64 = 5;

/// GPU bonuses
const GPU_DISCRETE_BONUS: u64 = 15;
const GPU_INTEGRATED_BONUS: u64 = 5;

/// Maximum power score
const MAX_POWER_SCORE: u64 = 100;

/// Bytes per megabyte
const BYTES_PER_MB: u64 = 1024 * 1024;

/// Bytes per gigabyte
const BYTES_PER_GB: u64 = 1024 * 1024 * 1024;

/// CPU vendor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    AMD,
    ARM,
    Apple,
    Unknown,
}

/// CPU architecture information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// CPU vendor
    pub vendor: CpuVendor,
    /// Number of physical cores
    pub physical_cores: u32,
    /// Number of logical cores (threads)
    pub logical_cores: u32,
    /// CPU frequency in MHz (if available)
    pub frequency_mhz: Option<u32>,
    /// CPU model name
    pub model: String,
    /// Has hyperthreading/SMT
    pub has_smt: bool,
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU vendor
    pub vendor: String,
    /// GPU model
    pub model: String,
    /// VRAM in MB (if available)
    pub vram_mb: Option<u32>,
    /// GPU is integrated
    pub is_integrated: bool,
}

/// Hardware capabilities detected from the system
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// CPU information
    pub cpu: CpuInfo,
    /// GPU information (if available)
    pub gpu: Option<GpuInfo>,
    /// Total system memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
}

/// Configuration for telemetry collection.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Sliding window duration in nanoseconds.
    pub window_duration_ns: u64,
    /// Number of sub-windows for velocity tracking.
    pub sub_windows: usize,
    /// Milli-actions per second considered active.
    pub active_threshold_milli_aps: u64,
    /// Milli-actions per second considered a burst.
    pub burst_threshold_milli_aps: u64,
    /// Navigation ratio threshold in basis points.
    pub navigation_ratio_bps: u64,
    /// Visible-key ratio threshold in basis points.
    pub visible_key_ratio_bps: u64,
    /// Enable key classification.
    pub enable_key_classification: bool,
    /// Enable navigation tracking.
    pub enable_navigation_tracking: bool,
    /// Enable cache size adaptation.
    pub enable_adaptive_sizing: bool,
    /// Minimum cache size.
    pub min_cache_size: usize,
    /// Maximum cache size.
    pub max_cache_size: usize,
}

impl TelemetryConfig {
    /// Creates a default telemetry configuration.
    pub fn new() -> Self {
        Self {
            window_duration_ns: WINDOW_DURATION_MEDIUM_POWER,
            sub_windows: 8,
            active_threshold_milli_aps: 2 * MILLI_SCALE,
            burst_threshold_milli_aps: 20_000,
            navigation_ratio_bps: 6_000,
            visible_key_ratio_bps: 3_000,
            enable_key_classification: true,
            enable_navigation_tracking: true,
            enable_adaptive_sizing: true,
            min_cache_size: CACHE_SIZE_MIN,
            max_cache_size: 8192,
        }
    }

    /// Creates telemetry configuration optimized for detected hardware.
    pub fn from_hardware(hardware: &HardwareCapabilities) -> Self {
        let cpu_power_score = cpu_power_score_approp(hardware);

        // Calculate memory ratio in basis points
        let memory_ratio_bps = (hardware.available_memory_bytes * BASIS_POINTS as u64
            / hardware.total_memory_bytes) as u64;

        // Adjust window duration based on CPU power
        let window_duration_ns = if cpu_power_score < CPU_POWER_SCORE_LOW {
            WINDOW_DURATION_LOW_POWER
        } else if cpu_power_score < CPU_POWER_SCORE_MEDIUM {
            WINDOW_DURATION_MEDIUM_POWER
        } else {
            WINDOW_DURATION_HIGH_POWER
        };

        // Adjust sub-windows based on core count
        let sub_windows = hardware
            .cpu
            .logical_cores
            .clamp(SUB_WINDOWS_MIN, SUB_WINDOWS_MAX) as usize;

        // Adjust burst threshold based on CPU performance
        let burst_threshold_milli_aps =
            BURST_THRESHOLD_BASE + (cpu_power_score * BURST_THRESHOLD_MULTIPLIER);

        // Adjust cache sizes based on available memory
        let available_mb = hardware.available_memory_bytes / BYTES_PER_MB;
        let base_cache_size = (available_mb * CACHE_SIZE_BASE_MULTIPLIER) as usize;
        let base_cache_size = base_cache_size.clamp(CACHE_SIZE_MIN, CACHE_SIZE_MAX);
        let min_cache_size = base_cache_size / CACHE_SIZE_DIVISOR as usize;
        let max_cache_size = base_cache_size * CACHE_SIZE_DIVISOR as usize;

        // Enable/disable features based on hardware
        let enable_key_classification =
            cpu_power_score > CPU_POWER_SCORE_HIGH_KEY_CLASSIFICATION;
        let enable_adaptive_sizing = memory_ratio_bps > MEMORY_RATIO_THRESHOLD_ADAPTIVE
            && cpu_power_score > CPU_POWER_SCORE_HIGH_ADAPTIVE;

        Self {
            window_duration_ns,
            sub_windows,
            active_threshold_milli_aps: 2 * MILLI_SCALE,
            burst_threshold_milli_aps,
            navigation_ratio_bps: 6_000,
            visible_key_ratio_bps: 3_000,
            enable_key_classification,
            enable_navigation_tracking: true,
            enable_adaptive_sizing,
            min_cache_size,
            max_cache_size,
        }
    }

    /// Creates telemetry configuration by detecting hardware automatically.
    pub fn auto_detect() -> Self {
        let hardware = detect_hardware_capabilities();
        Self::from_hardware(&hardware)
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Calculates a CPU power score (0-100) based on detected capabilities.
fn cpu_power_score_approp(hardware: &HardwareCapabilities) -> u64 {
    let mut score = 0u64;

    // Base score from core count
    let core_score = hardware.cpu.physical_cores as u64 * CORE_SCORE_MULTIPLIER;
    score += core_score.min(CORE_SCORE_MAX);

    // Bonus for SMT/hyperthreading
    if hardware.cpu.has_smt {
        score += SMT_BONUS;
    }

    // Bonus for high frequency
    if let Some(freq) = hardware.cpu.frequency_mhz {
        if freq > CPU_FREQ_HIGH {
            score += CPU_FREQ_HIGH_BONUS;
        } else if freq > CPU_FREQ_MEDIUM {
            score += CPU_FREQ_MEDIUM_BONUS;
        }
    }

    // Vendor-specific adjustments
    match hardware.cpu.vendor {
        CpuVendor::Intel | CpuVendor::AMD => score += VENDOR_BONUS_INTEL_AMD,
        CpuVendor::Apple => score += VENDOR_BONUS_APPLE,
        CpuVendor::ARM => score += VENDOR_BONUS_ARM,
        CpuVendor::Unknown => {}
    }

    // Memory bonus
    let memory_gb = hardware.total_memory_bytes / BYTES_PER_GB;
    if memory_gb >= MEMORY_HIGH_GB {
        score += MEMORY_HIGH_BONUS;
    } else if memory_gb >= MEMORY_MEDIUM_GB {
        score += MEMORY_MEDIUM_BONUS;
    }

    // GPU bonus
    if let Some(ref gpu) = hardware.gpu {
        if !gpu.is_integrated {
            score += GPU_DISCRETE_BONUS;
        } else {
            score += GPU_INTEGRATED_BONUS;
        }
    }

    score.min(MAX_POWER_SCORE)
}

/// Detects hardware capabilities from the system.
pub fn detect_hardware_capabilities() -> HardwareCapabilities {
    let cpu_info = detect_cpu_info();
    let gpu_info = detect_gpu_info();
    let memory_info = detect_memory_info();

    HardwareCapabilities {
        cpu: cpu_info,
        gpu: gpu_info,
        total_memory_bytes: memory_info.total,
        available_memory_bytes: memory_info.available,
    }
}

/// Detects CPU information.
fn detect_cpu_info() -> CpuInfo {
    let physical_cores = num_cpus::get_physical() as u32;
    let logical_cores = num_cpus::get() as u32;
    let has_smt = logical_cores > physical_cores;

    let vendor = detect_cpu_vendor();
    let model = detect_cpu_model();
    let frequency_mhz = detect_cpu_frequency();

    CpuInfo {
        vendor,
        physical_cores,
        logical_cores,
        frequency_mhz,
        model,
        has_smt,
    }
}

/// Detects CPU vendor.
fn detect_cpu_vendor() -> CpuVendor {
    #[cfg(target_arch = "x86_64")]
    {
        let model = detect_cpu_model();
        if model.contains("Intel") || model.contains("Xeon") || model.contains("Core") {
            return CpuVendor::Intel;
        }
        if model.contains("AMD") || model.contains("Ryzen") || model.contains("EPYC") {
            return CpuVendor::AMD;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let model = detect_cpu_model();
        if model.contains("Apple") {
            return CpuVendor::Apple;
        }
        return CpuVendor::ARM;
    }

    #[cfg(target_arch = "arm")]
    {
        return CpuVendor::ARM;
    }

    CpuVendor::Unknown
}

/// Detects CPU model string.
fn detect_cpu_model() -> String {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name") {
                    if let Some(model) = line.split(':').nth(1) {
                        return model.trim().to_string();
                    }
                }
                if line.starts_with("Hardware") {
                    if let Some(hardware) = line.split(':').nth(1) {
                        return hardware.trim().to_string();
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(&["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let model = String::from_utf8_lossy(&output.stdout);
            return model.trim().to_string();
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["cpu", "get", "name"])
            .output()
        {
            let model = String::from_utf8_lossy(&output.stdout);
            for line in model.lines() {
                if !line.trim().is_empty() && !line.contains("Name") {
                    return line.trim().to_string();
                }
            }
        }
    }

    "Unknown CPU".to_string()
}

/// Detects CPU frequency in MHz.
fn detect_cpu_frequency() -> Option<u32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("cpu MHz") {
                    if let Some(freq_str) = line.split(':').nth(1) {
                        if let Ok(freq) = freq_str.trim().parse::<f64>() {
                            return Some(freq as u32);
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(&["-n", "hw.cpufrequency"])
            .output()
        {
            let freq_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                return Some((freq_hz / 1_000_000) as u32);
            }
        }
    }

    None
}

/// Detects GPU information.
fn detect_gpu_info() -> Option<GpuInfo> {
    #[cfg(target_os = "linux")]
    {
        // Try to read from /proc/driver/nvidia/gpus if available
        if let Ok(_) = std::fs::metadata("/proc/driver/nvidia/gpus") {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=name,memory.total", "--format=csv,noheader"])
                .output()
            {
                let gpu_info = String::from_utf8_lossy(&output.stdout);
                for line in gpu_info.lines() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        let model = parts[0].trim().to_string();
                        let vram_str = parts[1].trim();
                        let vram_mb = vram_str.parse::<u32>().ok();
                        return Some(GpuInfo {
                            vendor: "NVIDIA".to_string(),
                            model,
                            vram_mb,
                            is_integrated: false,
                        });
                    }
                }
            }
        }

        // Try to detect AMD GPU
        if let Ok(_) = std::fs::metadata("/sys/class/drm") {
            if let Ok(output) = std::process::Command::new("lspci")
                .args(&["-nn", "-d", "::1002"])
                .output()
            {
                let gpu_info = String::from_utf8_lossy(&output.stdout);
                if !gpu_info.is_empty() {
                    return Some(GpuInfo {
                        vendor: "AMD".to_string(),
                        model: "AMD GPU".to_string(),
                        vram_mb: None,
                        is_integrated: false,
                    });
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output()
        {
            let display_info = String::from_utf8_lossy(&output.stdout);
            if display_info.contains("Chipset Model") {
                return Some(GpuInfo {
                    vendor: "Apple".to_string(),
                    model: "Apple Silicon GPU".to_string(),
                    vram_mb: None,
                    is_integrated: true,
                });
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "name,AdapterRAM"])
            .output()
        {
            let gpu_info = String::from_utf8_lossy(&output.stdout);
            for line in gpu_info.lines() {
                if !line.trim().is_empty()
                    && !line.contains("Name")
                    && !line.contains("AdapterRAM")
                {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if !parts.is_empty() {
                        let model = parts[0].to_string();
                        let is_integrated =
                            model.contains("Intel") || model.contains("Integrated");
                        return Some(GpuInfo {
                            vendor: if model.contains("NVIDIA") {
                                "NVIDIA".to_string()
                            } else if model.contains("AMD") {
                                "AMD".to_string()
                            } else if model.contains("Intel") {
                                "Intel".to_string()
                            } else {
                                "Unknown".to_string()
                            },
                            model,
                            vram_mb: None,
                            is_integrated,
                        });
                    }
                }
            }
        }
    }

    None
}

/// Memory information
struct MemoryInfo {
    total: u64,
    available: u64,
}

/// Detects memory information.
fn detect_memory_info() -> MemoryInfo {
    let sample = crate::base::base_memory::sample_system_memory();
    MemoryInfo {
        total: sample.total_bytes,
        available: sample.available_bytes,
    }
}
