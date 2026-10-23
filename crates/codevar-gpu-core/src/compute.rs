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

//! Backend-neutral compute-offload cost model.
//!
//! A graphics backend does not only render: it also owns a compute
//! device that a caller may offload bulk work to. Deciding *whether* to
//! offload is a cost problem, and this module is the shared vocabulary
//! for it:
//!
//! * [`ComputeProfile`] describes one device's potency — its peak
//!   arithmetic throughput, its device-memory bandwidth, the cost of the
//!   host↔device link and its fixed per-dispatch overhead. Every number
//!   comes from a probe (Vulkan in `codevar-gpu-vk`, OpenCL elsewhere),
//!   so nothing is hardcoded per device.
//! * [`Workload`] describes the work an operation represents.
//! * [`CostModel`] predicts the seconds each profile needs for a
//!   workload and picks the cheaper [`Tier`].
//!
//! The model is a roofline over four terms: the fixed dispatch cost, the
//! arithmetic time `operations / peak_flops`, the device-memory time
//! `device_bytes / memory_bandwidth` and the host-link time
//! `latency + host_bytes / bandwidth`. A discrete accelerator only wins
//! once the workload is large enough that its compute advantage repays
//! the transfer, which is exactly the crossover the caller cares about.
//!
//! The only physical constant encoded here is PCIe framing overhead, in
//! [`pcie_link_bandwidth`]: it is a property of the protocol revision,
//! not of any device, and is derived from the negotiated signalling
//! rate instead of a per-device table.

/// Which of the two tiers should service a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// The host's own processors.
    Cpu,
    /// The (possibly discrete) compute device.
    Gpu,
}

/// The sustained bandwidth and start-up cost of a host↔device link.
///
/// A device that shares the host's memory — a CPU fallback, or an
/// integrated accelerator — has no link cost and uses [`HostLink::LOCAL`];
/// a discrete device reached over PCIe uses a probe's numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostLink {
    /// Sustained one-directional bandwidth, in bytes per second.
    pub bandwidth: f64,
    /// One-directional transfer start-up cost, in seconds.
    pub latency: f64,
}

impl HostLink {
    /// A link that costs nothing to cross: the device shares the host's
    /// memory, so a transfer is a move in place.
    pub const LOCAL: Self = Self {
        bandwidth: f64::INFINITY,
        latency: 0.0,
    };

    /// Seconds to move `bytes` in one direction over this link.
    ///
    /// A link with a zero or non-finite bandwidth is unusable and reports
    /// an infinite cost, so a profile built from a failed probe declines
    /// instead of claiming to be fast.
    #[must_use]
    pub fn seconds(&self, bytes: u64) -> f64 {
        // A positive infinity is the local link and costs nothing; a NaN,
        // a negative or a zero bandwidth is an unusable link.
        if self.bandwidth.is_nan() || self.bandwidth <= 0.0 {
            return f64::INFINITY;
        }
        self.latency + bytes as f64 / self.bandwidth
    }
}

/// The potency of one compute device.
///
/// Every field is a measurement or a derivation from device queries; a
/// profile with a non-positive throughput or bandwidth is unusable and
/// [`CostModel::select`] keeps such a device on the CPU tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputeProfile {
    /// Peak arithmetic throughput, in floating-point operations per
    /// second.
    pub peak_flops: f64,
    /// Peak device-memory bandwidth, in bytes per second.
    pub memory_bandwidth: f64,
    /// The host↔device link this device is reached over.
    pub host_link: HostLink,
    /// Fixed cost of one dispatch (kernel launch plus queue submit), in
    /// seconds.
    pub launch_overhead: f64,
}

impl ComputeProfile {
    /// A device with no host link: the host's processors, or an
    /// integrated accelerator whose memory is the host's.
    #[must_use]
    pub const fn host_only(peak_flops: f64, memory_bandwidth: f64) -> Self {
        Self {
            peak_flops,
            memory_bandwidth,
            host_link: HostLink::LOCAL,
            launch_overhead: 0.0,
        }
    }

    /// A device reached over `host_link`, paying `launch_overhead` per
    /// dispatch.
    #[must_use]
    pub const fn discrete(
        peak_flops: f64,
        memory_bandwidth: f64,
        host_link: HostLink,
        launch_overhead: f64,
    ) -> Self {
        Self {
            peak_flops,
            memory_bandwidth,
            host_link,
            launch_overhead,
        }
    }

    /// Whether the profile carries enough information to be compared.
    ///
    /// A probe that could not read the core count, the clock or the link
    /// produces zeros; such a profile must never win the selection.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.peak_flops > 0.0 && self.memory_bandwidth > 0.0
    }

    /// Seconds of arithmetic for `operations` operations.
    #[must_use]
    pub fn arithmetic_seconds(&self, operations: u64) -> f64 {
        if !self.peak_flops.is_finite() || self.peak_flops <= 0.0 {
            return f64::INFINITY;
        }
        operations as f64 / self.peak_flops
    }

    /// Seconds to stream `bytes` through device memory.
    #[must_use]
    pub fn memory_seconds(&self, bytes: u64) -> f64 {
        if !self.memory_bandwidth.is_finite() || self.memory_bandwidth <= 0.0 {
            return f64::INFINITY;
        }
        bytes as f64 / self.memory_bandwidth
    }

    /// Predicted seconds for `workload` on this device: dispatch plus
    /// arithmetic, memory and host-link time.
    #[must_use]
    pub fn seconds(&self, workload: Workload) -> f64 {
        self.launch_overhead
            + self.arithmetic_seconds(workload.operations)
            + self.memory_seconds(workload.device_bytes)
            + self.host_link.seconds(workload.host_bytes)
    }
}

/// The work one operation represents, in the units the model needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workload {
    /// Operations the device executes.
    pub operations: u64,
    /// Bytes the device touches in its own memory.
    pub device_bytes: u64,
    /// Bytes crossing the host↔device link, counting both directions.
    pub host_bytes: u64,
}

impl Workload {
    /// A workload with explicit operation, device-byte and host-byte
    /// counts.
    #[must_use]
    pub const fn new(operations: u64, device_bytes: u64, host_bytes: u64) -> Self {
        Self {
            operations,
            device_bytes,
            host_bytes,
        }
    }

    /// A one-pass scan of `bytes` bytes.
    ///
    /// The engine's bulk operations read every byte once and perform one
    /// elementary operation per byte, so `operations == device_bytes`.
    /// The buffer crosses the host link once on upload and once on the
    /// way back, which is the `2 * bytes` term.
    #[must_use]
    pub const fn byte_scan(bytes: u64) -> Self {
        Self::new(bytes, bytes, bytes.saturating_mul(2))
    }

    /// A one-pass scan of `words` 32-bit words.
    #[must_use]
    pub const fn word_scan(words: u64) -> Self {
        Self::byte_scan(words.saturating_mul(4))
    }
}

/// The decision policy: pick the tier predicted to be cheaper.
///
/// `margin` is the speed-up the GPU must reach before it is chosen, so a
/// value above `1.0` keeps the CPU unless the accelerator is clearly
/// better. This damps the noise around the crossover size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    margin: f64,
}

impl CostModel {
    /// The neutral policy: the GPU is chosen when it is faster at all.
    pub const DEFAULT: Self = Self { margin: 1.0 };

    /// A policy that requires the GPU to be `margin` times faster than
    /// the CPU before it is chosen. Values below `1.0` are clamped.
    #[must_use]
    pub fn new(margin: f64) -> Self {
        Self {
            margin: if margin >= 1.0 { margin } else { 1.0 },
        }
    }

    /// The speed-up the GPU must reach.
    #[must_use]
    pub const fn margin(&self) -> f64 {
        self.margin
    }

    /// Chooses the tier for `workload`.
    ///
    /// The GPU is chosen only when its profile is usable and its
    /// predicted time beats the CPU's by `margin`.
    #[must_use]
    pub fn select(&self, cpu: &ComputeProfile, gpu: &ComputeProfile, workload: Workload) -> Tier {
        if !gpu.is_usable() {
            return Tier::Cpu;
        }
        let cpu_seconds = cpu.seconds(workload);
        let gpu_seconds = gpu.seconds(workload);
        if gpu_seconds * self.margin < cpu_seconds {
            Tier::Gpu
        } else {
            Tier::Cpu
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Effective one-directional bandwidth of a PCIe link, in bytes/s.
///
/// `gigatransfers_per_second` is the negotiated per-lane signaling rate
/// (Linux exposes it as `current_link_speed`) and `lanes` the negotiated
/// width (`current_link_width`). The framing overhead is derived from the
/// rate itself rather than a per-generation table: PCIe 1.x and 2.x frame
/// with 8b/10b, PCIe 3.0 and later with 128b/130b. A zero rate or width
/// reports `0.0`, which [`HostLink::seconds`] treats as unusable.
#[must_use]
pub fn pcie_link_bandwidth(gigatransfers_per_second: f64, lanes: u32) -> f64 {
    if !gigatransfers_per_second.is_finite() || gigatransfers_per_second <= 0.0 || lanes == 0 {
        return 0.0;
    }
    let framing = if gigatransfers_per_second <= 5.0 {
        8.0 / 10.0
    } else {
        128.0 / 130.0
    };
    gigatransfers_per_second * 1.0e9 * f64::from(lanes) * framing / 8.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slow scalar CPU: a gigaflop per second at ten gigabytes per
    /// second.
    fn cpu_profile() -> ComputeProfile {
        ComputeProfile::host_only(1.0e9, 1.0e10)
    }

    /// A discrete accelerator of a few teraflops reached over a twelve
    /// gigabyte per second link, with a fixed start-up cost.
    fn gpu_profile() -> ComputeProfile {
        ComputeProfile::discrete(
            4.0e12,
            5.0e11,
            HostLink {
                bandwidth: 1.2e10,
                latency: 5.0e-6,
            },
            1.0e-5,
        )
    }

    #[test]
    fn pcie_bandwidth_uses_the_protocol_framing() {
        // 16 GT/s is PCIe 4.0/5.0 signalling: 128b/130b framing.
        let gen4_x16 = pcie_link_bandwidth(16.0, 16);
        assert!((gen4_x16 - 31_507_692_307.69).abs() < 1.0);
        // 5 GT/s is PCIe 2.0 signalling: 8b/10b framing.
        let gen2_x16 = pcie_link_bandwidth(5.0, 16);
        assert!((gen2_x16 - 8.0e9).abs() < 1.0);
    }

    #[test]
    fn pcie_bandwidth_rejects_empty_links() {
        assert_eq!(pcie_link_bandwidth(0.0, 16), 0.0);
        assert_eq!(pcie_link_bandwidth(16.0, 0), 0.0);
        assert_eq!(pcie_link_bandwidth(-1.0, 16), 0.0);
    }

    #[test]
    fn unusable_profiles_decline() {
        let zero = ComputeProfile::host_only(0.0, 0.0);
        assert!(!zero.is_usable());
        assert!(
            zero.seconds(Workload::byte_scan(1 << 20))
                .is_infinite()
        );
    }

    #[test]
    fn a_host_only_profile_ignores_the_link() {
        let profile = cpu_profile();
        let small = profile.seconds(Workload::byte_scan(1024));
        let large = profile.seconds(Workload::byte_scan(1 << 20));
        assert!(large > small);
    }

    #[test]
    fn the_gpu_wins_only_past_the_crossover() {
        let model = CostModel::DEFAULT;
        let (cpu, gpu) = (cpu_profile(), gpu_profile());
        let small = model.select(&cpu, &gpu, Workload::byte_scan(64));
        let large = model.select(&cpu, &gpu, Workload::byte_scan(1 << 24));
        assert_eq!(small, Tier::Cpu, "a tiny scan must stay on the CPU");
        assert_eq!(large, Tier::Gpu, "a large scan must take the GPU");
    }

    #[test]
    fn a_margin_biases_the_crossover_towards_the_cpu() {
        let workload = Workload::byte_scan(1 << 20);
        let (cpu, gpu) = (cpu_profile(), gpu_profile());
        let neutral = CostModel::DEFAULT.select(&cpu, &gpu, workload);
        let biased = CostModel::new(1.0e6).select(&cpu, &gpu, workload);
        assert_eq!(neutral, Tier::Gpu);
        assert_eq!(biased, Tier::Cpu);
        // A sub-unit margin is clamped, never accelerating CPU preference.
        assert_eq!(CostModel::new(0.1).margin(), 1.0);
    }

    #[test]
    fn an_unusable_gpu_always_falls_back_to_the_cpu() {
        let model = CostModel::DEFAULT;
        let cpu = cpu_profile();
        let dead = ComputeProfile::discrete(0.0, 0.0, HostLink::LOCAL, 0.0);
        assert_eq!(model.select(&cpu, &dead, Workload::byte_scan(1 << 30)), Tier::Cpu);
    }

    #[test]
    fn workload_shapes_scale_with_the_element_size() {
        assert_eq!(Workload::byte_scan(1024).device_bytes, 1024);
        assert_eq!(Workload::word_scan(256).device_bytes, 1024);
        assert_eq!(Workload::byte_scan(1024).host_bytes, 2048);
    }
}
