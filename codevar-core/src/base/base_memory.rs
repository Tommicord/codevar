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

//! Centralized system and editor memory pressure detection.
//!
//! Samples host memory when available and combines it with
//! editor-reported usage (opened files, writable buffers, history)
//! so callers can reclaim memory under pressure.
//!
//! Periodic sampling is driven by a background [`crate::base::Daemon`]
//! via [`PeriodicMemoryChecker`].

use crate::base::base_task::Daemon;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Basis points for percentage calculations (`10_000` = 100%).
const BASIS_POINTS: u64 = 10_000;

/// Available-memory ratio (bps) at or below which pressure is critical.
const CRITICAL_AVAILABLE_BPS: u64 = 1_024;
/// Available-memory ratio (bps) at or below which pressure is high.
const HIGH_AVAILABLE_BPS: u64 = 2_048;
/// Available-memory ratio (bps) at or below which pressure is elevated.
const ELEVATED_AVAILABLE_BPS: u64 = 4_096;

/// Default soft budget for editor-tracked allocations (64 MiB).
pub const DEFAULT_SOFT_LIMIT_BYTES: u64 = 67108864;
/// Default hard budget for editor-tracked allocations (128 MiB).
pub const DEFAULT_HARD_LIMIT_BYTES: u64 = 134217728;
/// Default minimum interval between periodic samples.
pub const DEFAULT_SAMPLE_INTERVAL: Duration = Duration::from_secs(16);
/// Sleep slice used while waiting for the next periodic sample.
const PERIODIC_SLEEP_SLICE: Duration = Duration::from_millis(128);

/// Fallback totals used when the host cannot report memory (e.g. WASM).
const FALLBACK_TOTAL_BYTES: u64 = 8589934592;
const FALLBACK_AVAILABLE_BYTES: u64 = 4294967296;

/// Snapshot of host physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemory {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Currently available memory in bytes.
    pub available_bytes: u64,
}

impl SystemMemory {
    /// Creates a snapshot from total and available byte counts.
    #[inline]
    #[must_use]
    pub const fn new(total_bytes: u64, available_bytes: u64) -> Self {
        Self {
            total_bytes,
            available_bytes,
        }
    }

    /// Available memory as a fraction of total, in basis points.
    #[inline]
    #[must_use]
    pub fn available_ratio_bps(self) -> u64 {
        if self.total_bytes == 0 {
            return BASIS_POINTS;
        }
        self.available_bytes
            .saturating_mul(BASIS_POINTS)
            .saturating_div(self.total_bytes)
    }

    /// Classifies host pressure from the available/total ratio alone.
    #[inline]
    #[must_use]
    pub fn host_pressure(self) -> MemoryPressure {
        let ratio = self.available_ratio_bps();
        if ratio <= CRITICAL_AVAILABLE_BPS {
            MemoryPressure::Critical
        } else if ratio <= HIGH_AVAILABLE_BPS {
            MemoryPressure::High
        } else if ratio <= ELEVATED_AVAILABLE_BPS {
            MemoryPressure::Elevated
        } else {
            MemoryPressure::Normal
        }
    }
}

/// Ordered memory pressure levels used to trigger reclaim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// Plenty of headroom; reclaim only for very large histories.
    Normal = 0,
    /// Mild pressure; compress large cold TXUs.
    Elevated = 1,
    /// Significant pressure; compress most cold history.
    High = 2,
    /// Severe pressure; aggressive reclaim.
    Critical = 3,
}

impl MemoryPressure {
    /// Returns the more severe of two pressure levels.
    #[inline]
    #[must_use]
    pub const fn max(self, other: Self) -> Self {
        match (self, other) {
            (Self::Critical, _) | (_, Self::Critical) => Self::Critical,
            (Self::High, _) | (_, Self::High) => Self::High,
            (Self::Elevated, _) | (_, Self::Elevated) => Self::Elevated,
            _ => Self::Normal,
        }
    }

    /// Whether reclaim actions should run for this level.
    #[inline]
    #[must_use]
    pub const fn should_reclaim(self) -> bool {
        !matches!(self, Self::Normal)
    }
}

/// Editor-reported allocation totals used with host sampling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditorMemoryStats {
    /// Number of opened files.
    pub opened_files: usize,
    /// Sum of writable raw buffer lengths in bytes.
    pub writable_bytes: u64,
    /// Estimated resident history (TXU) bytes.
    pub history_bytes: u64,
    /// History construct buffer bytes.
    pub construct_bytes: u64,
}

impl EditorMemoryStats {
    /// Total editor-tracked bytes.
    #[inline]
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.writable_bytes
            .saturating_add(self.history_bytes)
            .saturating_add(self.construct_bytes)
    }
}

/// Soft/hard budgets for editor-tracked memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Soft limit: crossing it raises pressure to at least Elevated.
    pub soft_limit_bytes: u64,
    /// Hard limit: crossing it raises pressure to at least High.
    pub hard_limit_bytes: u64,
}

impl MemoryBudget {
    /// Creates a budget with explicit soft and hard limits.
    #[inline]
    #[must_use]
    pub const fn new(soft_limit_bytes: u64, hard_limit_bytes: u64) -> Self {
        Self {
            soft_limit_bytes,
            hard_limit_bytes,
        }
    }

    /// Default soft/hard limits suitable for WASM and desktop.
    #[inline]
    #[must_use]
    pub const fn default_limits() -> Self {
        Self::new(DEFAULT_SOFT_LIMIT_BYTES, DEFAULT_HARD_LIMIT_BYTES)
    }

    /// Classifies pressure from editor stats alone.
    #[inline]
    #[must_use]
    pub fn editor_pressure(self, stats: &EditorMemoryStats) -> MemoryPressure {
        let total = stats.total_bytes();
        if total >= self.hard_limit_bytes.saturating_mul(2) {
            MemoryPressure::Critical
        } else if total >= self.hard_limit_bytes {
            MemoryPressure::High
        } else if total >= self.soft_limit_bytes {
            MemoryPressure::Elevated
        } else {
            MemoryPressure::Normal
        }
    }

    /// Combines host and editor pressure into a single level.
    #[inline]
    #[must_use]
    pub fn evaluate(
        self,
        stats: &EditorMemoryStats,
        system: &SystemMemory,
    ) -> MemoryPressure {
        system.host_pressure().max(self.editor_pressure(stats))
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::default_limits()
    }
}

/// Rate-limited memory sampler for periodic reclaim checks.
pub struct MemoryMonitor {
    last_sample: Mutex<Option<Instant>>,
    interval: Duration,
    budget: MemoryBudget,
}

impl MemoryMonitor {
    /// Creates a monitor with the given sample interval and budget.
    #[inline]
    #[must_use]
    pub fn new(interval: Duration, budget: MemoryBudget) -> Self {
        Self {
            last_sample: Mutex::new(None),
            interval,
            budget,
        }
    }

    /// Creates a monitor with default interval and budget.
    #[inline]
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_SAMPLE_INTERVAL, MemoryBudget::default_limits())
    }

    /// Returns the configured budget.
    #[inline]
    #[must_use]
    pub const fn budget(&self) -> MemoryBudget {
        self.budget
    }

    /// Samples immediately (ignores the interval) and returns pressure.
    #[must_use]
    pub fn sample_now(
        &self,
        stats: &EditorMemoryStats,
    ) -> (SystemMemory, MemoryPressure) {
        let system = sample_system_memory();
        let pressure = self.budget.evaluate(stats, &system);
        if let Ok(mut guard) = self.last_sample.lock() {
            *guard = Some(Instant::now());
        }
        (system, pressure)
    }

    /// Samples when the interval has elapsed; returns `None` if still cooling down.
    pub fn maybe_sample(
        &self,
        stats: &EditorMemoryStats,
    ) -> Option<(SystemMemory, MemoryPressure)> {
        let now = Instant::now();
        let mut guard = self.last_sample.lock().ok()?;
        if let Some(last) = *guard
            && now.duration_since(last) < self.interval
        {
            return None;
        }
        let system = sample_system_memory();
        let pressure = self.budget.evaluate(stats, &system);
        *guard = Some(now);
        Some((system, pressure))
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// One sample produced by [`PeriodicMemoryChecker`] or [`MemoryMonitor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCheckSnapshot {
    /// Host memory snapshot.
    pub system: SystemMemory,
    /// Combined host + editor pressure.
    pub pressure: MemoryPressure,
    /// Editor-reported usage at sample time.
    pub stats: EditorMemoryStats,
}

/// Provider of live editor memory stats for the background checker.
pub type EditStatsProvider = Arc<dyn Fn() -> EditorMemoryStats + Send + Sync>;

/// Optional callback invoked after each periodic sample.
pub type MemorySampleCallback = Arc<dyn Fn(MemoryCheckSnapshot) + Send + Sync>;

/// Background periodic memory pressure checker driven by a [`Daemon`].
///
/// Spawns a daemon thread that samples host + editor memory on `interval`,
/// stores the latest snapshot, and optionally invokes a callback.
pub struct PeriodicMemoryChecker {
    stop: Arc<AtomicBool>,
    latest: Arc<Mutex<Option<MemoryCheckSnapshot>>>,
    sample_count: Arc<std::sync::atomic::AtomicU64>,
    daemon: Option<Daemon<u64>>,
    interval: Duration,
}

impl PeriodicMemoryChecker {
    /// Starts a background daemon that samples every `interval`.
    ///
    /// # Arguments
    ///
    /// * `interval` - Time between samples (clamped to at least 1 ms).
    /// * `budget` - Soft/hard limits used when classifying pressure.
    /// * `stats_provider` - Closure returning current editor memory stats.
    /// * `on_sample` - Optional callback invoked after each sample.
    #[must_use]
    pub fn start(
        interval: Duration,
        budget: MemoryBudget,
        stats_provider: EditStatsProvider,
        on_sample: Option<MemorySampleCallback>,
    ) -> Self {
        let interval = interval.max(Duration::from_millis(8));
        let stop = Arc::new(AtomicBool::new(false));
        let latest = Arc::new(Mutex::new(None));
        let sample_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let stop_flag = Arc::clone(&stop);
        let latest_slot = Arc::clone(&latest);
        let count_slot = Arc::clone(&sample_count);

        let daemon = Daemon::new(async move {
            run_periodic_sample_loop(
                interval,
                budget,
                stats_provider,
                on_sample,
                stop_flag,
                latest_slot,
                count_slot,
            )
        });

        Self {
            stop,
            latest,
            sample_count,
            daemon: Some(daemon),
            interval,
        }
    }

    /// Starts a checker with default soft/hard budgets.
    #[must_use]
    pub fn start_with_defaults(
        interval: Duration,
        stats_provider: EditStatsProvider,
        on_sample: Option<MemorySampleCallback>,
    ) -> Self {
        Self::start(
            interval,
            MemoryBudget::default_limits(),
            stats_provider,
            on_sample,
        )
    }

    /// Configured sample interval.
    #[inline]
    #[must_use]
    pub const fn interval(&self) -> Duration {
        self.interval
    }

    /// Whether the background daemon is still running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.stop.load(Ordering::Acquire)
            && self.daemon.as_ref().is_some_and(Daemon::is_running)
    }

    /// Number of samples completed so far.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Acquire)
    }

    /// Latest snapshot, if at least one sample has completed.
    #[must_use]
    pub fn latest(&self) -> Option<MemoryCheckSnapshot> {
        self.latest.lock().ok().and_then(|guard| *guard)
    }

    /// Latest pressure level, if a sample exists.
    #[must_use]
    pub fn latest_pressure(&self) -> Option<MemoryPressure> {
        self.latest().map(|snap| snap.pressure)
    }

    /// Signals the daemon to stop and waits for it to finish.
    pub fn stop(&mut self) -> u64 {
        self.stop.store(true, Ordering::Release);
        let Some(daemon) = self.daemon.take() else {
            return self.sample_count();
        };
        // Prefer a clean loop exit via `stop` over `cancel`, so the future
        // can still complete with the sample count.
        match daemon.get() {
            Ok(count) => count,
            Err(_) => self.sample_count(),
        }
    }
}

impl Drop for PeriodicMemoryChecker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(daemon) = self.daemon.take() {
            let deadline = Instant::now() + Duration::from_millis(200);
            while Instant::now() < deadline {
                match daemon.try_get() {
                    Ok(_) => return,
                    Err(crate::base::base_task::DaemonError::StillRunning) => {
                        std::thread::yield_now();
                    }
                    Err(_) => return,
                }
            }
            daemon.cancel();
        }
    }
}

impl MemoryMonitor {
    /// Spawns a [`PeriodicMemoryChecker`] using this monitor's interval and budget.
    #[must_use]
    pub fn spawn_periodic(
        &self,
        stats_provider: EditStatsProvider,
        on_sample: Option<MemorySampleCallback>,
    ) -> PeriodicMemoryChecker {
        PeriodicMemoryChecker::start(
            self.interval,
            self.budget,
            stats_provider,
            on_sample,
        )
    }
}

fn run_periodic_sample_loop(
    interval: Duration,
    budget: MemoryBudget,
    stats_provider: EditStatsProvider,
    on_sample: Option<MemorySampleCallback>,
    stop: Arc<AtomicBool>,
    latest: Arc<Mutex<Option<MemoryCheckSnapshot>>>,
    sample_count: Arc<std::sync::atomic::AtomicU64>,
) -> u64 {
    let mut completed = 0u64;
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let stats = stats_provider();
        let system = sample_system_memory();
        let pressure = budget.evaluate(&stats, &system);
        let snapshot = MemoryCheckSnapshot {
            system,
            pressure,
            stats,
        };
        if let Ok(mut guard) = latest.lock() {
            *guard = Some(snapshot);
        }
        completed = completed.saturating_add(1);
        sample_count.store(completed, Ordering::Release);

        crate::debug!(
            "Periodic memory sample #{}: pressure={:?} writable={} history={}",
            completed,
            pressure,
            stats.writable_bytes,
            stats.history_bytes
        );
        if let Some(callback) = on_sample.as_ref() {
            callback(snapshot);
        }
        if !sleep_interruptible(interval, &stop) {
            break;
        }
    }
    completed
}

/// Sleeps up to `duration`, returning `false` if `stop` was set.
fn sleep_interruptible(duration: Duration, stop: &AtomicBool) -> bool {
    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        std::thread::sleep(remaining.min(PERIODIC_SLEEP_SLICE));
    }
    !stop.load(Ordering::Acquire)
}

/// Samples host memory; falls back to conservative defaults when unavailable.
#[must_use]
pub fn sample_system_memory() -> SystemMemory {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if let Some(info) = sample_linux_meminfo() {
            return info;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(info) = sample_macos_memory() {
            return info;
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(info) = sample_windows_memory() {
            return info;
        }
    }
    SystemMemory::new(FALLBACK_TOTAL_BYTES, FALLBACK_AVAILABLE_BYTES)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn sample_linux_meminfo() -> Option<SystemMemory> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = 0u64;
    let mut available = 0u64;
    for line in content.lines() {
        if line.starts_with("MemTotal:")
            && let Some(mem_str) = line.split_whitespace().nth(1)
            && let Ok(mem_kb) = mem_str.parse::<u64>()
        {
            total = mem_kb.saturating_mul(1024);
        } else if line.starts_with("MemAvailable:")
            && let Some(mem_str) = line.split_whitespace().nth(1)
            && let Ok(mem_kb) = mem_str.parse::<u64>()
        {
            available = mem_kb.saturating_mul(1024);
        }
    }
    if total > 0 {
        Some(SystemMemory::new(total, available.min(total)))
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn sample_macos_memory() -> Option<SystemMemory> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let mem_str = String::from_utf8_lossy(&output.stdout);
    let total = mem_str.trim().parse::<u64>().ok()?;
    Some(SystemMemory::new(total, total / 2))
}

#[cfg(target_os = "windows")]
fn sample_windows_memory() -> Option<SystemMemory> {
    let output = std::process::Command::new("wmic")
        .args(["OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory"])
        .output()
        .ok()?;
    let mem_info = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = mem_info.lines().collect();
    if lines.len() < 3 {
        return None;
    }
    let total_kb = lines[1].trim().parse::<u64>().ok()?;
    let free_kb = lines[2].trim().parse::<u64>().ok()?;
    Some(SystemMemory::new(
        total_kb.saturating_mul(1024),
        free_kb.saturating_mul(1024),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_HARD_LIMIT_BYTES, DEFAULT_SOFT_LIMIT_BYTES, EditorMemoryStats,
        MemoryBudget, MemoryCheckSnapshot, MemoryMonitor, MemoryPressure,
        PeriodicMemoryChecker, SystemMemory, sample_system_memory,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        predicate()
    }

    #[test]
    fn host_pressure_thresholds() {
        let normal = SystemMemory::new(10_000, 5_000);
        assert_eq!(normal.host_pressure(), MemoryPressure::Normal);

        let elevated = SystemMemory::new(10_000, 3_000);
        assert_eq!(elevated.host_pressure(), MemoryPressure::Elevated);

        let high = SystemMemory::new(10_000, 1_500);
        assert_eq!(high.host_pressure(), MemoryPressure::High);

        let critical = SystemMemory::new(10_000, 500);
        assert_eq!(critical.host_pressure(), MemoryPressure::Critical);
    }

    #[test]
    fn editor_budget_pressure() {
        let budget = MemoryBudget::default_limits();
        let soft = EditorMemoryStats {
            opened_files: 1,
            writable_bytes: DEFAULT_SOFT_LIMIT_BYTES,
            history_bytes: 0,
            construct_bytes: 0,
        };
        assert_eq!(budget.editor_pressure(&soft), MemoryPressure::Elevated);

        let hard = EditorMemoryStats {
            writable_bytes: DEFAULT_HARD_LIMIT_BYTES,
            ..soft
        };
        assert_eq!(budget.editor_pressure(&hard), MemoryPressure::High);

        let critical = EditorMemoryStats {
            writable_bytes: DEFAULT_HARD_LIMIT_BYTES.saturating_mul(2),
            ..soft
        };
        assert_eq!(budget.editor_pressure(&critical), MemoryPressure::Critical);
    }

    #[test]
    fn pressure_max_picks_worse() {
        assert_eq!(
            MemoryPressure::Normal.max(MemoryPressure::High),
            MemoryPressure::High
        );
        assert_eq!(
            MemoryPressure::Critical.max(MemoryPressure::Elevated),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn sample_system_memory_returns_positive_totals() {
        let sample = sample_system_memory();
        assert!(sample.total_bytes > 0);
        assert!(sample.available_bytes <= sample.total_bytes);
    }

    #[test]
    fn memory_monitor_maybe_sample_respects_interval() {
        let monitor =
            MemoryMonitor::new(Duration::from_millis(80), MemoryBudget::default_limits());
        let stats = EditorMemoryStats::default();
        assert!(monitor.maybe_sample(&stats).is_some());
        assert!(monitor.maybe_sample(&stats).is_none());
        assert!(wait_until(Duration::from_millis(200), || {
            monitor.maybe_sample(&stats).is_some()
        }));
    }

    #[test]
    fn memory_monitor_sample_now_always_returns() {
        let monitor = MemoryMonitor::with_defaults();
        let stats = EditorMemoryStats {
            opened_files: 2,
            writable_bytes: 1024,
            history_bytes: 2048,
            construct_bytes: 16,
        };
        let (system, pressure) = monitor.sample_now(&stats);
        assert!(system.total_bytes > 0);
        assert_eq!(pressure, monitor.budget().evaluate(&stats, &system));
    }

    #[test]
    fn periodic_checker_produces_samples() {
        let stats = Arc::new(Mutex::new(EditorMemoryStats {
            opened_files: 1,
            writable_bytes: 100,
            history_bytes: 50,
            construct_bytes: 10,
        }));
        let stats_for_provider = Arc::clone(&stats);
        let provider: super::EditStatsProvider = Arc::new(move || {
            *stats_for_provider
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });

        let mut checker = PeriodicMemoryChecker::start(
            Duration::from_millis(15),
            MemoryBudget::default_limits(),
            provider,
            None,
        );

        assert!(wait_until(Duration::from_millis(300), || {
            checker.sample_count() >= 2
        }));
        let latest = checker.latest().expect("snapshot");
        assert_eq!(latest.stats.opened_files, 1);
        assert_eq!(latest.stats.writable_bytes, 100);
        assert!(latest.system.total_bytes > 0);
        assert!(checker.latest_pressure().is_some());

        let count = checker.stop();
        assert!(count >= 2);
        assert!(!checker.is_running());
    }

    #[test]
    fn periodic_checker_invokes_callback() {
        let callback_hits = Arc::new(AtomicU64::new(0));
        let hits = Arc::clone(&callback_hits);
        let snapshots = Arc::new(Mutex::new(Vec::new()));
        let snaps = Arc::clone(&snapshots);

        let provider: super::EditStatsProvider = Arc::new(|| EditorMemoryStats {
            opened_files: 3,
            writable_bytes: DEFAULT_SOFT_LIMIT_BYTES,
            history_bytes: 0,
            construct_bytes: 0,
        });

        let on_sample: super::MemorySampleCallback =
            Arc::new(move |snap: MemoryCheckSnapshot| {
                hits.fetch_add(1, Ordering::AcqRel);
                if let Ok(mut guard) = snaps.lock() {
                    guard.push(snap);
                }
            });

        let mut checker = PeriodicMemoryChecker::start(
            Duration::from_millis(10),
            MemoryBudget::default_limits(),
            provider,
            Some(on_sample),
        );

        assert!(wait_until(Duration::from_millis(250), || {
            callback_hits.load(Ordering::Acquire) >= 2
        }));
        let recorded = snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(recorded.len() >= 2);
        assert_eq!(recorded[0].pressure, MemoryPressure::Elevated);
        assert_eq!(recorded[0].stats.opened_files, 3);
        drop(recorded);
        let _ = checker.stop();
    }

    #[test]
    fn periodic_checker_stop_is_idempotent() {
        let provider: super::EditStatsProvider =
            Arc::new(|| EditorMemoryStats::default());
        let mut checker = PeriodicMemoryChecker::start_with_defaults(
            Duration::from_millis(20),
            provider,
            None,
        );
        assert!(wait_until(Duration::from_millis(200), || {
            checker.sample_count() >= 1
        }));
        let first = checker.stop();
        let second = checker.stop();
        assert!(first >= 1);
        assert_eq!(second, first);
    }

    #[test]
    fn periodic_checker_sees_updated_editor_stats() {
        let stats = Arc::new(Mutex::new(EditorMemoryStats::default()));
        let stats_for_provider = Arc::clone(&stats);
        let provider: super::EditStatsProvider = Arc::new(move || {
            *stats_for_provider
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });

        let mut checker = PeriodicMemoryChecker::start(
            Duration::from_millis(10),
            MemoryBudget::new(1_000, 2_000),
            provider,
            None,
        );

        assert!(wait_until(Duration::from_millis(200), || {
            checker.sample_count() >= 1
        }));
        let before = checker.latest_pressure().expect("pressure");

        {
            let mut guard = stats
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Far above hard limit => editor Critical; max() must surface it.
            guard.writable_bytes = 5_000;
        }

        assert!(wait_until(Duration::from_millis(250), || {
            checker.latest_pressure() == Some(MemoryPressure::Critical)
        }));
        assert!(before <= MemoryPressure::Critical);
        let _ = checker.stop();
    }

    #[test]
    fn memory_monitor_spawn_periodic() {
        let monitor =
            MemoryMonitor::new(Duration::from_millis(12), MemoryBudget::default_limits());
        let provider: super::EditStatsProvider = Arc::new(|| EditorMemoryStats {
            opened_files: 1,
            writable_bytes: 64,
            history_bytes: 32,
            construct_bytes: 8,
        });
        let mut checker = monitor.spawn_periodic(provider, None);
        assert_eq!(checker.interval(), Duration::from_millis(12));
        assert!(wait_until(Duration::from_millis(200), || {
            checker.latest().is_some()
        }));
        let _ = checker.stop();
    }

    #[test]
    fn evaluate_combines_host_and_editor_pressure() {
        let budget = MemoryBudget::new(1_000, 2_000);
        let stats = EditorMemoryStats {
            opened_files: 0,
            writable_bytes: 1_500,
            history_bytes: 0,
            construct_bytes: 0,
        };
        // Editor alone => Elevated; critical host should win.
        let critical_host = SystemMemory::new(10_000, 500);
        assert_eq!(
            budget.evaluate(&stats, &critical_host),
            MemoryPressure::Critical
        );
        let healthy_host = SystemMemory::new(10_000, 8_000);
        assert_eq!(
            budget.evaluate(&stats, &healthy_host),
            MemoryPressure::Elevated
        );
    }

    #[test]
    fn should_reclaim_only_when_not_normal() {
        assert!(!MemoryPressure::Normal.should_reclaim());
        assert!(MemoryPressure::Elevated.should_reclaim());
        assert!(MemoryPressure::High.should_reclaim());
        assert!(MemoryPressure::Critical.should_reclaim());
    }
}
