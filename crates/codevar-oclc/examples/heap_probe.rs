//! Temporary heap-measurement probe (hosted): runs the `--emit spirv`
//! pipeline under a fixed-size TLSF heap and reports peak usage plus a
//! backtrace-attributed allocation profile.

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::ptr;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use codevar_ocl_asm::SpirvStream;
use codevar_ocl_ir::lower::ItemLowerer;
use codevar_ocl_ir::verify::verify_function;
use codevar_ocl_parse::{GenericParam, ItemKind, ItemStream};
use codevar_ocl_sar::{DeclCollector, analyze_body, file_checks};
use codevar_tlsf_alloc::LockedTlsf;

static HEAP: LockedTlsf = LockedTlsf::new();

const REGION_BYTES: usize = 0x100_0000;

static mut REGION: [u8; REGION_BYTES] = [0; REGION_BYTES];

static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Cumulative allocated bytes per site (first non-probe frame).
static PROFILE: LazyLock<Mutex<HashMap<String, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// Live bytes per site.
static LIVE_PROFILE: LazyLock<Mutex<HashMap<String, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// Live sizes keyed by allocation pointer, for dealloc attribution.
static SIZES: LazyLock<Mutex<HashMap<usize, (usize, String)>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

static SAMPLE: AtomicUsize = AtomicUsize::new(0);
const SAMPLE_RATE: usize = 32;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_default();
    let source = std::fs::read_to_string(&path).unwrap_or_default();
    let label = std::env::args().nth(2);
    let filter = label.as_deref().unwrap_or("");
    mark("boot");
    run_pipeline(&source, filter);
    dump_totals();
}

fn site() -> Option<String> {
    let tick = SAMPLE.fetch_add(1, Ordering::Relaxed);
    if !tick.is_multiple_of(SAMPLE_RATE) {
        return None;
    }
    let backtrace = std::backtrace::Backtrace::force_capture();
    let text = backtrace.to_string();
    let mut result = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Skip the probe's own frames and the allocator shim.
        if trimmed.is_empty()
            || trimmed.starts_with("std::alloc")
            || trimmed.contains("heap_probe")
            || trimmed.contains("core::alloc")
        {
            continue;
        }
        // Keep the function name only (drop address and file/line noise).
        let name = trimmed
            .split(" at ")
            .next()
            .unwrap_or(trimmed)
            .trim()
            .to_string();
        result.push_str(&name);
        result.push('\n');
        if result.lines().count() >= 4 {
            break;
        }
    }
    Some(result)
}

struct Probe;

static INIT: std::sync::Once = std::sync::Once::new();

fn ensure_heap() {
    INIT.call_once(|| {
        let start = ptr::addr_of_mut!(REGION).cast::<u8>();
        // SAFETY: REGION backs only this heap and outlives the process.
        if unsafe { HEAP.lock().add_region(start, REGION_BYTES) }.is_err() {
            eprintln!("region rejected");
            std::process::abort();
        }
    });
}

unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ensure_heap();
        let mut heap = HEAP.lock();
        match heap.allocate(layout.size(), layout.align()) {
            Some(block) => {
                let used = heap.stats().used_bytes;
                PEAK.fetch_max(used, Ordering::Relaxed);
                drop(heap);
                if let Some(key) = site() {
                    let scaled = layout.size() * SAMPLE_RATE;
                    *PROFILE
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .entry(key.clone())
                        .or_insert(0) += scaled;
                    *LIVE_PROFILE
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .entry(key.clone())
                        .or_insert(0) += scaled;
                    SIZES
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(block.as_ptr() as usize, (scaled, key));
                }
                block.as_ptr()
            }
            None => {
                let stats = heap.stats();
                eprintln!(
                    "OOM: request {} bytes (align {}); used={} total={} peak={}",
                    layout.size(),
                    layout.align(),
                    stats.used_bytes,
                    stats.total_bytes,
                    PEAK.load(Ordering::Relaxed)
                );
                std::process::abort();
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        let mut heap = HEAP.lock();
        if let Some(block) = ptr::NonNull::new(ptr) {
            unsafe { heap.deallocate(block) };
        }
        let used = heap.stats().used_bytes;
        PEAK.fetch_max(used, Ordering::Relaxed);
        drop(heap);
        if let Some((size, key)) = SIZES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(ptr as usize))
        {
            LIVE_PROFILE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(key)
                .and_modify(|value| *value = value.saturating_sub(size));
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { self.alloc(layout) };
        if !result.is_null() {
            unsafe { ptr::write_bytes(result, 0, layout.size()) };
        }
        result
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { self.alloc(Layout::from_size_align_unchecked(new_size, layout.align())) };
        if !ptr.is_null() && !new_ptr.is_null() {
            let copy = if layout.size() < new_size {
                layout.size()
            } else {
                new_size
            };
            unsafe { ptr::copy_nonoverlapping(ptr, new_ptr, copy) };
        }
        if !ptr.is_null() {
            unsafe { self.dealloc(ptr, layout) };
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: Probe = Probe;

fn mark(label: &str) {
    let used = HEAP.lock().stats().used_bytes;
    let peak = PEAK.load(Ordering::Relaxed);
    println!("--- {label}: used={used} peak={peak}");
    let live = LIVE_PROFILE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut entries: Vec<(&String, &usize)> = live.iter().filter(|(_, v)| **v > 0).collect();
    entries.sort_by(|left, right| right.1.cmp(left.1));
    for (key, bytes) in entries.into_iter().take(12) {
        println!("    {bytes:>8}  {}", key.lines().next().unwrap_or("?"));
    }
}

fn dump_totals() {
    let profile = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let mut entries: Vec<(&String, &usize)> = profile.iter().collect();
    entries.sort_by(|left, right| right.1.cmp(left.1));
    println!("=== cumulative allocated bytes by site ===");
    for (key, bytes) in entries.into_iter().take(40) {
        println!("{bytes:>10}");
        for line in key.lines() {
            println!("           {line}");
        }
    }
}

fn run_pipeline(source: &str, filter: &str) {
    let mut collector = DeclCollector::new();
    let mut items = ItemStream::new(source);
    while let Some(outcome) = items.next_item() {
        collector.feed(source, &outcome.item);
    }
    mark("decl-scan");
    let (mut env, _diagnostics) = collector.finish();
    mark("decl-env");
    let declarations = env.declarations().to_vec();
    mark("decl-vec");
    let mut lowerer = ItemLowerer::new(&declarations);
    for aliases in [true, false] {
        let mut items = ItemStream::new(source);
        while let Some(outcome) = items.next_item() {
            match (&outcome.item.kind, aliases) {
                (ItemKind::TypeAlias(alias), true) => {
                    let _ = lowerer.declare_alias(alias);
                }
                (ItemKind::Fn(function), false) => {
                    let kernel = env.is_kernel(&function.name);
                    let _ = lowerer.declare_function(function, kernel);
                }
                _ => {}
            }
        }
    }
    mark("declared");
    let mut stream = match SpirvStream::prelude(lowerer.module()) {
        Ok(stream) => stream,
        Err(error) => {
            eprintln!("prelude failed: {error}");
            return;
        }
    };
    mark("prelude");
    let mut items = ItemStream::new(source);
    let mut bodies = 0usize;
    while let Some(outcome) = items.next_item() {
        let ItemKind::Fn(function) = &outcome.item.kind else {
            continue;
        };
        if function
            .generics
            .iter()
            .any(|param| matches!(param, GenericParam::Type { .. }))
        {
            continue;
        }
        if !filter.is_empty() && function.name != filter {
            continue;
        }
        let mark_point = lowerer.module().watermark();
        let (_diagnostics, tables) = analyze_body(&mut env, source, &outcome.item);
        mark("analyze_body");
        if lowerer
            .lower_function_body(function, &tables.types, &tables.resolutions)
            .is_err()
        {
            eprintln!("lower failed for {}", function.name);
            return;
        }
        mark("lower_body");
        let Some(kernel) = lowerer.function(&function.name) else {
            return;
        };
        if let Err(error) = verify_function(lowerer.module(), kernel) {
            eprintln!("verify failed: {error}");
            return;
        }
        mark("verify");
        if let Err(error) = stream.emit_function_body(lowerer.module(), kernel) {
            eprintln!("emit failed: {error}");
            return;
        }
        mark("emit");
        drop(lowerer.module_mut().take_function_body(kernel));
        lowerer.module_mut().truncate_values(mark_point);
        bodies += 1;
        mark("reclaim");
    }
    let _checks = file_checks(&env);
    mark("file-checks");
    let Ok(words) = stream.finish(lowerer.module()) else {
        return;
    };
    mark("finish");
    let bytes = codevar_ocl_asm::spirv::to_bytes(&words);
    println!(
        "functions={bodies} spirv={}B final-peak={}",
        bytes.len(),
        PEAK.load(Ordering::Relaxed)
    );
}
