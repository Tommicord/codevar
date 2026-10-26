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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Build script for `codevar-wredit`.
//!
//! Assembles the hand-written `.S` kernels under `src/arch/` for the
//! target architecture via the `cc` crate and links them into the
//! crate. The selection is driven by Cargo's `CARGO_CFG_TARGET_ARCH` /
//! `CARGO_CFG_TARGET_FEATURE` environment (the *target* triple, so this
//! is correct while cross-compiling) rather than `#[cfg(target_arch)]`,
//! which would only describe the build host.
//!
//! On x86_64 and aarch64 a single `.S` source is assembled once per
//! instruction-set level with the preprocessor selecting the vector
//! width; the Rust side picks the widest level the running CPU supports.

use std::env;

fn main() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let arch_dir = format!("{}/src/arch", env!("CARGO_MANIFEST_DIR"));

    let has = |name: &str| features.split(',').any(|f| f == name);

    match arch.as_str() {
        "x86_64" => build_x86_64(&arch_dir),
        "x86" => build_i686(&arch_dir, has("sse2")),
        "aarch64" => build_aarch64(&arch_dir),
        "arm" => {
            // Hard-float triples imply NEON even when Cargo does not list
            // it among the target features.
            let neon = has("neon") || target.contains("eabihf");
            build_arm(&arch_dir, &target, neon)
        }
        "riscv64" => build_riscv(&arch_dir, "riscv64", has("v") || has("zve32x")),
        "riscv32" => build_riscv(&arch_dir, "riscv32", has("v") || has("zve32x")),
        // Every other target (wasm, etc.) uses the pure-Rust fallbacks.
        _ => {}
    }

    println!("cargo:rerun-if-changed={arch_dir}");
}

/// Assembles one `memmove.S`/`memcpy.S`/`memchr.S` set into a static
/// library. `suffix` is the symbol suffix (e.g. `"_avx2"`), `feature`
/// is the preprocessor feature macro to define (or `None`), and `flags`
/// are extra compiler flags for the variant.
fn compile_x86_set(dir: &str, lib: &str, suffix: &str, feature: Option<&str>, flags: &[&str]) {
    let mut build = cc::Build::new();
    build
        .include(dir)
        .file(format!("{dir}/memmove.S"))
        .file(format!("{dir}/memcpy.S"))
        .file(format!("{dir}/memchr.S"))
        .file(format!("{dir}/wordscan.S"))
        .define("CODEVAR_SUFFIX", Some(suffix));
    if let Some(name) = feature {
        build.define(name, None);
    }
    // POPCNT accompanies every AVX2-class part in practice; keeping the
    // SSE2 baseline free of it preserves the oldest x86_64 targets.
    if suffix != "_sse2" {
        build.define("CODEVAR_HAVE_POPCNT", None);
        build.flag_if_supported("-mpopcnt");
    }
    for flag in flags {
        build.flag(flag);
    }
    build.flag_if_supported("-fno-asynchronous-unwind-tables");
    build.compile(lib);
}

/// x86_64: three ISA levels selected at run time via cpuid.
fn build_x86_64(arch_dir: &str) {
    let dir = format!("{arch_dir}/x86_64");

    // SSE2 baseline: always available on x86_64, no extra flags.
    compile_x86_set(&dir, "codevar_wredit_x86_64_sse2", "_sse2", None, &[]);
    // AVX2.
    compile_x86_set(
        &dir,
        "codevar_wredit_x86_64_avx2",
        "_avx2",
        Some("CODEVAR_HAVE_AVX2"),
        &["-mavx2"],
    );
    // AVX-512 (F + BW + VL) with AVX2 as a convenient superset base.
    compile_x86_set(
        &dir,
        "codevar_wredit_x86_64_avx512",
        "_avx512",
        Some("CODEVAR_HAVE_AVX512"),
        &["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx2"],
    );

    // Prefetch needs no ISA level: one build, no suffix. The extra
    // integer/byte kernels are also baseline-only, so they live in a
    // separate archive with unsuffixed symbols.
    let mut build = cc::Build::new();
    build
        .include(&dir)
        .file(format!("{dir}/prefetch.S"))
        .compile("codevar_wredit_x86_64_prefetch");
}

/// i686: SSE2 when the target baseline provides it, scalar otherwise.
fn build_i686(arch_dir: &str, has_sse2: bool) {
    let dir = format!("{arch_dir}/i686");
    let mut build = cc::Build::new();
    build
        .include(&dir)
        .file(format!("{dir}/memmove.S"))
        .file(format!("{dir}/memcpy.S"))
        .file(format!("{dir}/memchr.S"))
        .file(format!("{dir}/prefetch.S"));
    if has_sse2 {
        build.define("CODEVAR_HAVE_SSE2", None);
        build.flag("-msse2");
    }
    build.compile("codevar_wredit_i686");
}

/// aarch64: baseline NEON always, plus an SVE variant of the copy
/// kernels selected at run time via cpuid. Both are assembled into the
/// binary. `memchr` has no SVE path, so it only appears in the baseline.
fn build_aarch64(arch_dir: &str) {
    let dir = format!("{arch_dir}/aarch64");

    let mut baseline = cc::Build::new();
    baseline
        .include(&dir)
        .file(format!("{dir}/memmove.S"))
        .file(format!("{dir}/memcpy.S"))
        .file(format!("{dir}/memchr.S"))
        .file(format!("{dir}/prefetch.S"))
        .compile("codevar_wredit_aarch64");

    let mut sve = cc::Build::new();
    sve.include(&dir)
        .file(format!("{dir}/memmove.S"))
        .file(format!("{dir}/memcpy.S"))
        .define("CODEVAR_SUFFIX", Some("_sve"))
        .define("CODEVAR_HAVE_SVE", None)
        .flag_if_supported("-march=armv8-a+sve")
        .compile("codevar_wredit_aarch64_sve");
}

/// arm: NEON for A-profile (armv7/r), scalar word copies for the
/// Thumb M-profile cores (thumbv6m/thumbv7m/thumbv7em/thumbv8m).
fn build_arm(arch_dir: &str, target: &str, has_neon: bool) {
    let m_profile = target.starts_with("thumbv6m")
        || target.starts_with("thumbv7m")
        || target.starts_with("thumbv7em")
        || target.starts_with("thumbv8m");

    if m_profile {
        let dir = format!("{arch_dir}/thumbv7m");
        let mut build = cc::Build::new();
        build
            .include(&dir)
            .file(format!("{dir}/memmove.S"))
            .file(format!("{dir}/memcpy.S"))
            .file(format!("{dir}/memchr.S"))
            .file(format!("{dir}/prefetch.S"));
        // A word-only kernel works from armv6-m upwards; pick the exact
        // ISA when we recognise the triple, otherwise let the target
        // default stand.
        if target.starts_with("thumbv6m") {
            build.flag_if_supported("-march=armv6-m");
        } else if target.starts_with("thumbv8m") {
            build.flag_if_supported("-march=armv8-m.main");
        } else if target.starts_with("thumbv7em") {
            build.flag_if_supported("-march=armv7e-m");
        } else {
            build.flag_if_supported("-march=armv7-m");
        }
        build.compile("codevar_wredit_thumbv7m");
        return;
    }

    let dir = format!("{arch_dir}/armv7");
    let mut build = cc::Build::new();
    build
        .include(&dir)
        .file(format!("{dir}/memmove.S"))
        .file(format!("{dir}/memcpy.S"))
        .file(format!("{dir}/memchr.S"))
        .file(format!("{dir}/prefetch.S"));
    if has_neon {
        build.define("CODEVAR_HAVE_NEON", None);
        build.flag_if_supported("-mfpu=neon");
    }
    build.compile("codevar_wredit_armv7");
}

/// riscv: scalar 8-byte (RV64) / 4-byte (RV32) copies, with an optional
/// RVV path when the target enables the vector extension.
fn build_riscv(arch_dir: &str, sub: &str, has_rvv: bool) {
    let dir = format!("{arch_dir}/{sub}");
    let mut build = cc::Build::new();
    build
        .include(&dir)
        .file(format!("{dir}/memmove.S"))
        .file(format!("{dir}/memcpy.S"))
        .file(format!("{dir}/memchr.S"))
        .file(format!("{dir}/prefetch.S"));
    if has_rvv {
        build.define("CODEVAR_HAVE_RVV", None);
        let march = if sub == "riscv32" { "rv32gcv" } else { "rv64gcv" };
        build.flag_if_supported(format!("-march={march}"));
    }
    build.compile(&format!("codevar_wredit_{sub}"));
}
