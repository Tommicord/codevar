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

//! Portable CPU feature detection at compile time and at run time.
//!
//! this module drives the identification facilities of each
//! architecture directly through [`core::arch::asm!`]:
//!
//! | Architecture  | Run-time probe |
//! |---------------|----------------|
//! | `x86`/`x86_64` | `CPUID` leaves `0`, `1`, `7`, `0x8000_0000/0x8000_0001` plus `XGETBV` for OS-enabled AVX/AVX-512 state |
//! | `aarch64`    | `AT_HWCAP`/`AT_HWCAP2` via `getauxval` on Linux/Android, `MRS` of the `ID_AA64*` ID registers elsewhere |
//! | `arm`        | `AT_HWCAP`/`AT_HWCAP2` via `getauxval` on Linux/Android |
//! | everything else (`riscv64`, `powerpc64`, `wasm32`, …) | compile time only: user mode has no portable feature probe, so `cfg(target_feature)` is used |
//!
//! Detection is layered:
//!
//! 1. **Compile time** — [`compile_time_features`] and
//!    `cpu_feature_at_time!` report what was compiled into the binary
//!    (`-C target-feature=…`, `#[target_feature]`).
//! 2. **Run time** — [`detect`] executes the probes above; its result is
//!    cached in a process-wide lock so the `CPUID`/`MRS`/`getauxval` work
//!    happens only once. [`features`] returns the union of both layers and
//!    [`has`] answers a single question.
//!
//! ## Examples
//!
//! Runtime check (any architecture, unknown names fail to compile):
//!
//! ```rust
//! let runtime = codevar_base::cpu_feature!(avx2);
//! let also_runtime = codevar_base::cpu_feature!("sse4.1");
//! assert_eq!(
//!     runtime,
//!     codevar_base::basic_cpuid::has(codevar_base::basic_cpuid::Feature::Avx2)
//! );
//! assert_eq!(also_runtime, codevar_base::cpu_feature!("sse4_1"));
//! ```
//!
//! Const check, usable in `const` items and generic bounds:
//!
//! ```rust
//! const SSE2_AT_BUILD_TIME: bool = codevar_base::cpu_feature_at_time!(sse2);
//! let _ = SSE2_AT_BUILD_TIME;
//! ```
//!
//! ## Performance
//!
//! [`compile_time_features`] is a pure `const fn`. The first call to
//! [`features`] probes the CPU and fills an atomic cache; every later call is
//! a single atomic load plus a `u64` union. [`has`] short-circuits on
//! the compile-time answer, so features compiled into the binary cost nothing.

use core::fmt;
use spin::once::Once;

/// Compares two byte slices without calling trait methods, so it can be used
/// inside `const fn` (where `PartialEq::eq` is not yet callable).
const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Builds [`Feature`], its lookup tables, [`compile_time_features`] and the
/// `cpu_feature!` / `cpu_feature_at_time!` macros from one feature
/// table, so the compile-time names, the run-time lookup and the macros can
/// never drift apart.
///
/// Each entry is `name => Variant = mask, cfg!, "doc", aliases [..];`.
///
/// Aliases are captured as `tt` rather than `literal` because only `tt`
/// fragments are transparent when interpolated into the matcher of the
/// generated `cpu_feature!` macros (an opaque `literal` fragment would
/// never match a literal written by the caller).
macro_rules! define_features {
    (
        $(
            $name:ident => $variant:ident = $mask:expr, $cfg:expr, $doc:expr,
                aliases [ $($alias:tt),* $(,)? ] ;
        )*
    ) => {
        /// A single CPU capability that can be queried portably.
        ///
        /// The same variant is meaningful on every architecture: on an
        /// architecture that cannot provide it (for example [`Feature::Neon`]
        /// on `x86_64`) it is simply never reported as available.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Feature {
            $(
                #[doc = $doc]
                $variant,
            )*
        }

        impl Feature {
            /// Every feature known to this module, in table order.
            pub const ALL: &'static [Feature] = &[$(Self::$variant,)*];

            /// Bit mask backing `self` inside a [`Features`] set.
            ///
            /// Umbrella features (such as [`Feature::Avx512`]) expand to the
            /// union of the bits they require, which makes both
            /// [`Features::insert`] and [`Features::contains`] work
            /// structurally: inserting an umbrella stores all of its
            /// constituents and `contains` succeeds only when all of them
            /// are set.
            #[inline]
            pub const fn mask(self) -> u64 {
                match self {
                    $(Self::$variant => $mask,)*
                }
            }

            /// Canonical spelling of this feature, as accepted by
            /// [`Feature::from_name`] and by the `cpu_feature!` macros.
            #[inline]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($name),)*
                }
            }

            /// Whether the feature is compiled into this binary.
            ///
            /// This is a `const fn`: the result is a compile-time constant
            /// derived from `cfg(target_feature)`, so callers can use it in
            /// `const` items without probing the CPU.
            #[inline]
            pub const fn compile_time_available(self) -> bool {
                match self {
                    $(Self::$variant => $cfg,)*
                }
            }

            /// Resolves a spelling (canonical name or alias) to a feature.
            ///
            /// Accepts every canonical name such as `"sse4_1"` plus the
            /// aliases listed in the feature table, for example `"sse4.1"`,
            /// `"crc"` or `"atomics"`.
            pub const fn from_name(name: &str) -> Option<Feature> {
                let bytes = name.as_bytes();
                $(
                    if bytes_eq(bytes, stringify!($name).as_bytes()) {
                        return Some(Self::$variant);
                    }
                    $(
                        if bytes_eq(bytes, $alias.as_bytes()) {
                            return Some(Self::$variant);
                        }
                    )*
                )*
                None
            }
        }

        /// Features that are known at compile time from `cfg(target_feature)`
        /// (that is, what `-C target-feature=…` or `#[target_feature]`
        /// enabled), without executing any probe instruction.
        ///
        /// # Performance
        ///
        /// `const fn`; evaluating it costs nothing at run time.
        pub const fn compile_time_features() -> Features {
            let mut bits = 0u64;
            $(
                if $cfg {
                    bits |= Feature::$variant.mask();
                }
            )*
            Features(bits)
        }

        /// Reports whether a CPU feature is available on this machine,
        /// combining the compile-time and the run-time answers.
        ///
        /// Accepts a canonical feature name as an identifier:
        ///
        /// ```rust
        /// if codevar_base::cpu_feature!(sse2) {
        ///     // Safe to run SSE2 code.
        /// }
        /// ```
        ///
        /// String literals are also accepted — every canonical name
        /// (`"sse4_1"`) and every alias (`"sse4.1"`, `"crc"`, `"atomics"`).
        /// Unknown names are a compile error.
        #[macro_export]
        macro_rules! cpu_feature {
            $(
                ($name) => {
                    $crate::basic_cpuid::has(
                        $crate::basic_cpuid::Feature::$variant
                    )
                };
                $( ($alias) => {
                    $crate::basic_cpuid::has(
                        $crate::basic_cpuid::Feature::$variant
                    )
                }; )*
            )*
            ($feature:literal) => {
                ::core::compile_error!(::core::concat!(
                    "unknown CPU feature: ", $feature
                ))
            };
            ($other:ident) => {
                ::core::compile_error!(::core::concat!(
                    "unknown CPU feature: ", ::core::stringify!($other)
                ))
            };
        }

        /// Reports whether a CPU feature is guaranteed by the way this binary
        /// was compiled, without consulting the CPU at run time.
        ///
        /// The result is a `const bool`, usable in `const` items:
        ///
        /// ```rust
        /// const HAS_SSE2: bool = codevar_base::cpu_feature_at_time!(sse2);
        /// let _ = HAS_SSE2;
        /// ```
        ///
        /// Accepts the same identifiers and string literals as
        /// `cpu_feature!`. Unknown names are a compile error.
        #[macro_export]
        macro_rules! cpu_feature_at_time {
            $(
                ($name) => {
                    $crate::basic_cpuid::Feature::$variant
                        .compile_time_available()
                };
                $( ($alias) => {
                    $crate::basic_cpuid::Feature::$variant
                        .compile_time_available()
                }; )*
            )*
            ($feature:literal) => {
                ::core::compile_error!(::core::concat!(
                    "unknown CPU feature: ", $feature
                ))
            };
            ($other:ident) => {
                ::core::compile_error!(::core::concat!(
                    "unknown CPU feature: ", ::core::stringify!($other)
                ))
            };
        }
    };
}

define_features! {
    // x86 / x86_64 (CPUID).
    sse => Sse = 1 << 0, cfg!(target_feature = "sse"),
        "SSE — Streaming SIMD Extensions (x86)", aliases ["sse"];
    sse2 => Sse2 = 1 << 1, cfg!(target_feature = "sse2"),
        "SSE2 — Streaming SIMD Extensions 2 (x86)", aliases ["sse2"];
    sse3 => Sse3 = 1 << 2, cfg!(target_feature = "sse3"),
        "SSE3 — Streaming SIMD Extensions 3 (x86)", aliases ["sse3"];
    ssse3 => Ssse3 = 1 << 3, cfg!(target_feature = "ssse3"),
        "SSSE3 — Supplementary Streaming SIMD Extensions 3 (x86)", aliases ["ssse3"];
    sse4_1 => Sse41 = 1 << 4, cfg!(target_feature = "sse4.1"),
        "SSE4.1 — Streaming SIMD Extensions 4.1 (x86)", aliases ["sse4_1", "sse4.1"];
    sse4_2 => Sse42 = 1 << 5, cfg!(target_feature = "sse4.2"),
        "SSE4.2 — Streaming SIMD Extensions 4.2 (x86)", aliases ["sse4_2", "sse4.2"];
    sse4a => Sse4a = 1 << 6, cfg!(target_feature = "sse4a"),
        "SSE4a — Streaming SIMD Extensions 4a (AMD x86)", aliases ["sse4a"];
    avx => Avx = 1 << 7, cfg!(target_feature = "avx"),
        "AVX — Advanced Vector Extensions, OS-enabled state included",
        aliases ["avx"];
    avx2 => Avx2 = 1 << 8, cfg!(target_feature = "avx2"),
        "AVX2 — Advanced Vector Extensions 2, OS-enabled state included",
        aliases ["avx2"];
    avx512f => Avx512F = 1 << 9, cfg!(target_feature = "avx512f"),
        "AVX-512 Foundation, OS-enabled state included", aliases ["avx512f"];
    avx512cd => Avx512Cd = 1 << 10, cfg!(target_feature = "avx512cd"),
        "AVX-512 Conflict Detection, OS-enabled state included", aliases ["avx512cd"];
    avx512bw => Avx512Bw = 1 << 11, cfg!(target_feature = "avx512bw"),
        "AVX-512 Byte and Word, OS-enabled state included", aliases ["avx512bw"];
    avx512dq => Avx512Dq = 1 << 12, cfg!(target_feature = "avx512dq"),
        "AVX-512 Doubleword and Quadword, OS-enabled state included",
        aliases ["avx512dq"];
    avx512vl => Avx512Vl = 1 << 13, cfg!(target_feature = "avx512vl"),
        "AVX-512 Vector Length Extensions, OS-enabled state included",
        aliases ["avx512vl"];
    avx512vnni => Avx512Vnni = 1 << 14, cfg!(target_feature = "avx512vnni"),
        "AVX-512 Vector Neural Network Instructions, OS-enabled state",
        aliases ["avx512vnni"];
    fma => Fma = 1 << 15, cfg!(target_feature = "fma"),
        "FMA3 — fused multiply-add (x86)", aliases ["fma"];
    f16c => F16c = 1 << 16, cfg!(target_feature = "f16c"),
        "F16C — half-precision conversion instructions (x86)", aliases ["f16c"];
    bmi1 => Bmi1 = 1 << 17, cfg!(target_feature = "bmi1"),
        "BMI1 — Bit Manipulation Instruction Set 1 (x86)", aliases ["bmi1"];
    bmi2 => Bmi2 = 1 << 18, cfg!(target_feature = "bmi2"),
        "BMI2 — Bit Manipulation Instruction Set 2 (x86)", aliases ["bmi2"];
    popcnt => Popcnt = 1 << 19, cfg!(target_feature = "popcnt"),
        "POPCNT — population count instruction (x86)", aliases ["popcnt"];
    lzcnt => Lzcnt = 1 << 20, cfg!(target_feature = "lzcnt"),
        "LZCNT — leading zero count instruction (x86)", aliases ["lzcnt"];
    aes => Aes = 1 << 21, cfg!(target_feature = "aes"),
        "AES — AES block cipher instructions (x86, ARM, AArch64)",
        aliases ["aes"];
    pclmulqdq => Pclmulqdq = 1 << 22,
        cfg!(target_feature = "pclmulqdq"),
        "PCLMULQDQ — 64×64 carry-less multiply (x86)", aliases ["pclmulqdq"];
    sha => Sha = 1 << 23,
        cfg!(any(target_feature = "sha", target_feature = "sha2")),
        "SHA — SHA-1 and SHA-256 hash instructions (x86 SHA-NI)",
        aliases ["sha"];
    rdrand => Rdrand = 1 << 24, cfg!(target_feature = "rdrand"),
        "RDRAND — hardware random number generator (x86)", aliases ["rdrand"];
    rdseed => Rdseed = 1 << 25, cfg!(target_feature = "rdseed"),
        "RDSEED — hardware seed random number generator (x86)", aliases ["rdseed"];
    adx => Adx = 1 << 26, cfg!(target_feature = "adx"),
        "ADX — multi-precision multiply add (x86)", aliases ["adx"];
    movbe => Movbe = 1 << 27, cfg!(target_feature = "movbe"),
        "MOVBE — byte-swapping load/store (x86)", aliases ["movbe"];
    // `fsgsbase` has no stable rustc `target_feature` name yet, so the
    // compile-time answer is always false; the run-time probe detects it.
    fsgsbase => Fsgsbase = 1 << 28, false,
        "FSGSBASE — FS/GS base instructions (x86)", aliases ["fsgsbase"];
    tbm => Tbm = 1 << 29, cfg!(target_feature = "tbm"),
        "TBM — Trailing Bit Manipulation (AMD x86)", aliases ["tbm"];
    xsave => Xsave = 1 << 30, cfg!(target_feature = "xsave"),
        "XSAVE — XSAVE/XRSTOR save area support (x86)", aliases ["xsave"];

    // ARM / AArch64 (AT_HWCAP or ID registers).
    neon => Neon = 1 << 31, cfg!(target_feature = "neon"),
        "NEON — Advanced SIMD (arm, aarch64)", aliases ["neon", "asimd"];
    fp => Fp = 1 << 32,
        cfg!(any(
            target_feature = "neon",
            target_feature = "vfp2",
            target_feature = "vfp3",
            target_feature = "vfp4",
        )),
        "Floating point (arm, aarch64)", aliases ["fp", "vfp"];
    pmull => Pmull = 1 << 33,
        cfg!(target_feature = "pclmulqdq"),
        "PMULL — 64×64 polynomial multiply (arm, aarch64)", aliases ["pmull"];
    sha1 => Sha1 = 1 << 34,
        cfg!(any(target_feature = "sha", target_feature = "sha2")),
        "SHA-1 instructions (arm, aarch64)", aliases ["sha1"];
    sha2 => Sha2 = 1 << 35,
        cfg!(any(target_feature = "sha2", target_feature = "sha")),
        "SHA-256 instructions (arm, aarch64)", aliases ["sha2"];
    crc32 => Crc32 = 1 << 36,
        cfg!(any(target_feature = "crc", target_feature = "sse4.2")),
        "CRC32 checksum instructions (arm, aarch64, x86 SSE4.2)",
        aliases ["crc32", "crc"];
    dotprod => DotProd = 1 << 37, cfg!(target_feature = "dotprod"),
        "Dot product instructions (arm, aarch64)", aliases ["dotprod", "asimddp"];
    sve => Sve = 1 << 38, cfg!(target_feature = "sve"),
        "SVE — Scalable Vector Extension (aarch64)", aliases ["sve"];
    sve2 => Sve2 = 1 << 39, cfg!(target_feature = "sve2"),
        "SVE2 — Scalable Vector Extension 2 (aarch64)", aliases ["sve2"];
    fp16 => Fp16 = 1 << 40, cfg!(target_feature = "fp16"),
        "Half-precision floating point arithmetic (arm, aarch64)",
        aliases ["fp16"];
    lse => Lse = 1 << 41, cfg!(target_feature = "lse"),
        "LSE — Large System Extensions atomics (arm, aarch64)",
        aliases ["lse", "atomics"];

    // RISC-V (compile time only; user mode has no portable probe).
    rvv => Rvv = 1 << 42, cfg!(target_feature = "v"),
        "V — vector extension (riscv)", aliases ["rvv", "v"];
    zba => Zba = 1 << 43, cfg!(target_feature = "zba"),
        "Zba — address generation bit-manipulation (riscv)", aliases ["zba"];
    zbb => Zbb = 1 << 44, cfg!(target_feature = "zbb"),
        "Zbb — basic bit-manipulation (riscv)", aliases ["zbb"];

    // WebAssembly.
    simd128 => Simd128 = 1 << 45, cfg!(target_feature = "simd128"),
        "simd128 — 128-bit SIMD (wasm32)", aliases ["simd128"];

    // Umbrella feature: requires F + BW + DQ + VL (the usual AVX-512
    // baseline for portable kernels).
    avx512 => Avx512 = Feature::Avx512F.mask() | Feature::Avx512Bw.mask()
            | Feature::Avx512Dq.mask() | Feature::Avx512Vl.mask(),
        cfg!(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
        )),
        "AVX-512 — the F+BW+DQ+VL baseline (x86_64)", aliases ["avx512"];
}

/// A set of available CPU features, stored as a compact bit set.
///
/// Feature sets are cheap to copy (`u64`) and support the usual set
/// operations. [`Features::contains`] also understands umbrella features such
/// as [`Feature::Avx512`], which succeed only when every constituent bit is
/// set.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Features(u64);

impl Features {
    /// The empty feature set.
    pub const EMPTY: Self = Self(0);

    /// Returns whether every bit required by `feature` is set.
    ///
    /// # Performance
    ///
    /// `const fn`; one mask, one compare.
    #[inline]
    pub const fn contains(self, feature: Feature) -> bool {
        (self.0 & feature.mask()) == feature.mask()
    }

    /// Adds `feature` to the set. Inserting an umbrella feature such as
    /// [`Feature::Avx512`] stores all of its constituent bits.
    #[inline]
    pub const fn insert(&mut self, feature: Feature) {
        self.0 |= feature.mask();
    }

    /// Returns the union of two feature sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns whether no feature is set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the raw bit set, mainly useful for diagnostics and tests.
    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Iterates over the set features in [`Feature::ALL`] order.
    ///
    /// # Performance
    ///
    /// Walks the full feature table; intended for diagnostics, not hot paths.
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = Feature> {
        Feature::ALL
            .iter()
            .copied()
            .filter(move |feature| self.contains(*feature))
    }
}

impl fmt::Debug for Features {
    /// Prints the set as `Features(sse2, avx2, …)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Features(")?;
        let mut first = true;
        for feature in self.iter() {
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            f.write_str(feature.name())?;
        }
        f.write_str(")")
    }
}

/// Process-wide cache of [`features`].
static FEATURES: Once<Features> = Once::new();

/// Returns the union of the compile-time and run-time feature sets for this
/// machine, probing the CPU on the first call only.
///
/// # Performance
///
/// The first call runs [`detect`] and stores the result; later calls are a
/// single atomic load and a bitwise union.
#[inline]
pub fn features() -> Features {
    *FEATURES.call_once(|| compile_time_features().union(detect()))
}

/// Returns whether `feature` is available here, combining the compile-time
/// and run-time answers.
///
/// Features compiled into the binary short-circuit before the cached run-time
/// set is even loaded.
///
/// # Examples
///
/// ```rust
/// if codevar_base::basic_cpuid::has(codevar_base::basic_cpuid::Feature::Sse2) {
///     // Safe to run SSE2 code on this machine.
/// }
/// ```
#[inline]
pub fn has(feature: Feature) -> bool {
    if feature.compile_time_available() {
        return true;
    }
    features().contains(feature)
}

/// Probes the CPU at run time and returns every feature the probe found.
///
/// The result does not include [`compile_time_features`]; use [`features`] to
/// obtain the union of both layers. The probe runs on every call — prefer
/// [`features`] unless you specifically want an uncached probe (for example
/// in tests).
///
/// Architectures without a portable user-space probe (for example `riscv64`
/// or `wasm32`) return an empty set; [`compile_time_features`] is their only
/// source.
///
/// # Performance
///
/// A handful of `CPUID`/`MRS`/`getauxval` calls; no allocation, no locks.
pub fn detect() -> Features {
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            let mut detected = Features::EMPTY;
            x86::detect(&mut detected);
            detected
        } else if #[cfg(target_arch = "aarch64")] {
            let mut detected = Features::EMPTY;
            aarch64::detect(&mut detected);
            detected
        } else if #[cfg(target_arch = "arm")] {
            let mut detected = Features::EMPTY;
            arm::detect(&mut detected);
            detected
        } else {
            Features::EMPTY
        }
    }
}

/// Run-time detection for `x86` and `x86_64` via `CPUID` and `XGETBV`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::{Feature, Features};

    /// Output registers of one `CPUID` leaf.
    #[derive(Clone, Copy)]
    struct Registers {
        eax: u32,
        ebx: u32,
        ecx: u32,
        edx: u32,
    }

    impl Registers {
        /// Returns whether bit `bit` of `EDX` is set.
        #[inline]
        const fn bit_edx(self, bit: u32) -> bool {
            (self.edx >> bit) & 1 != 0
        }

        /// Returns whether bit `bit` of `EBX` is set.
        #[inline]
        const fn bit_ebx(self, bit: u32) -> bool {
            (self.ebx >> bit) & 1 != 0
        }

        /// Returns whether bit `bit` of `ECX` is set.
        #[inline]
        const fn bit_ecx(self, bit: u32) -> bool {
            (self.ecx >> bit) & 1 != 0
        }
    }

    /// Executes `CPUID` with the given leaf and sub-leaf.
    ///
    /// # Safety
    ///
    /// The caller must run on an x86 or x86_64 CPU. `CPUID` is supported by
    /// every such CPU that Rust can target, and the instruction itself has no
    /// preconditions, so in practice this is safe to call unconditionally.
    #[inline]
    unsafe fn cpuid(leaf: u32, sub_leaf: u32) -> Registers {
        let eax: u32;
        let ebx: u32;
        let ecx: u32;
        let edx: u32;
        // SAFETY: every operand is a register the block fully defines; `EBX`
        // is saved into the temporary output register before `CPUID`
        // clobbers it and swapped back afterwards, so the value the compiler
        // placed in `EBX` is preserved across the block (same scheme as
        // `core::arch::x86::__cpuid_count`). `CPUID` reads no memory and
        // writes only registers, hence `nostack`; `preserves_flags` holds
        // because `CPUID` does not touch the flags register.
        #[cfg(target_arch = "x86")]
        unsafe {
            core::arch::asm!(
                "mov {tmp}, ebx",
                "cpuid",
                "xchg {tmp}, ebx",
                inout("eax") leaf => eax,
                inout("ecx") sub_leaf => ecx,
                out("edx") edx,
                tmp = out(reg) ebx,
                options(nostack, preserves_flags),
            );
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "mov {tmp:r}, rbx",
                "cpuid",
                "xchg {tmp:r}, rbx",
                inout("eax") leaf => eax,
                inout("ecx") sub_leaf => ecx,
                out("edx") edx,
                tmp = out(reg) ebx,
                options(nostack, preserves_flags),
            );
        }
        Registers { eax, ebx, ecx, edx }
    }

    /// Reads `XCR0`, the extended control register selected by `ECX = 0`.
    ///
    /// # Safety
    ///
    /// The caller must have verified via `CPUID.1:ECX.OSXSAVE` that the OS
    /// enabled `XSAVE` management of the processor state; otherwise `XGETBV`
    /// raises `#UD`.
    #[inline]
    unsafe fn xcr0() -> u64 {
        let low: u32;
        let high: u32;
        // SAFETY: the caller guarantees `OSXSAVE` (see this function's docs).
        // `XGETBV` reads `ECX` as input and writes `EAX`/`EDX`; it neither
        // touches the stack nor memory nor the flags.
        unsafe {
            core::arch::asm!(
                "xgetbv",
                in("ecx") 0u32,
                out("eax") low,
                out("edx") high,
                options(nostack, preserves_flags),
            );
        }
        ((u64::from(high)) << 32) | u64::from(low)
    }

    /// Fills `features` from `CPUID`, including the OS-enabled AVX and
    /// AVX-512 state checks (`XCR0`) that gate every VEX/EVEX feature.
    pub(super) fn detect(features: &mut Features) {
        // SAFETY: `CPUID` is valid on every x86/x86_64 CPU (see `cpuid`).
        // `XGETBV` is only executed when `CPUID.1:ECX.OSXSAVE` reports that
        // the OS manages `XSAVE` state, which is exactly the architectural
        // precondition for `XGETBV` not raising `#UD`.
        unsafe {
            let max_leaf = cpuid(0, 0).eax;
            let mut os_ymm = false;
            let mut os_zmm = false;

            if max_leaf >= 1 {
                let r = cpuid(1, 0);

                if r.bit_edx(25) {
                    features.insert(Feature::Sse);
                }
                if r.bit_edx(26) {
                    features.insert(Feature::Sse2);
                }
                if r.bit_ecx(0) {
                    features.insert(Feature::Sse3);
                }
                if r.bit_ecx(9) {
                    features.insert(Feature::Ssse3);
                }
                if r.bit_ecx(19) {
                    features.insert(Feature::Sse41);
                }
                if r.bit_ecx(20) {
                    features.insert(Feature::Sse42);
                    // SSE4.2 carries the `CRC32` instruction, so the portable
                    // `crc32` feature is satisfied by it on x86.
                    features.insert(Feature::Crc32);
                }
                if r.bit_ecx(1) {
                    features.insert(Feature::Pclmulqdq);
                    // PCLMULQDQ is the x86 equivalent of ARM's PMULL.
                    features.insert(Feature::Pmull);
                }
                if r.bit_ecx(25) {
                    features.insert(Feature::Aes);
                }
                if r.bit_ecx(22) {
                    features.insert(Feature::Movbe);
                }
                if r.bit_ecx(23) {
                    features.insert(Feature::Popcnt);
                }
                if r.bit_ecx(26) {
                    features.insert(Feature::Xsave);
                }
                if r.bit_ecx(27) {
                    let xcr0 = xcr0();
                    os_ymm = xcr0 & 0x6 == 0x6;
                    os_zmm = os_ymm && xcr0 & 0xE6 == 0xE6;
                }
                if r.bit_ecx(28) && os_ymm {
                    features.insert(Feature::Avx);
                }
                if r.bit_ecx(12) && os_ymm {
                    features.insert(Feature::Fma);
                }
                if r.bit_ecx(29) {
                    features.insert(Feature::F16c);
                }
                if r.bit_ecx(30) {
                    features.insert(Feature::Rdrand);
                }
            }

            if max_leaf >= 7 {
                let r = cpuid(7, 0);

                if r.bit_ebx(0) {
                    features.insert(Feature::Fsgsbase);
                }
                if r.bit_ebx(3) {
                    features.insert(Feature::Bmi1);
                }
                if r.bit_ebx(5) && os_ymm {
                    features.insert(Feature::Avx2);
                }
                if r.bit_ebx(8) {
                    features.insert(Feature::Bmi2);
                }
                if r.bit_ecx(11) && os_zmm {
                    features.insert(Feature::Avx512Vnni);
                }
                if r.bit_ebx(16) && os_zmm {
                    features.insert(Feature::Avx512F);
                }
                if r.bit_ebx(17) && os_zmm {
                    features.insert(Feature::Avx512Dq);
                }
                if r.bit_ebx(18) {
                    features.insert(Feature::Rdseed);
                }
                if r.bit_ebx(19) {
                    features.insert(Feature::Adx);
                }
                if r.bit_ebx(28) && os_zmm {
                    features.insert(Feature::Avx512Cd);
                }
                if r.bit_ebx(29) {
                    // SHA-NI implements both SHA-1 and SHA-256.
                    features.insert(Feature::Sha);
                    features.insert(Feature::Sha1);
                    features.insert(Feature::Sha2);
                }
                if r.bit_ebx(30) && os_zmm {
                    features.insert(Feature::Avx512Bw);
                }
                if r.bit_ebx(31) && os_zmm {
                    features.insert(Feature::Avx512Vl);
                }
            }

            let extended_max = cpuid(0x8000_0000, 0).eax;
            if extended_max >= 0x8000_0001 {
                let r = cpuid(0x8000_0001, 0);
                if r.bit_ecx(5) {
                    features.insert(Feature::Lzcnt);
                }
                if r.bit_ecx(6) {
                    features.insert(Feature::Sse4a);
                }
                if r.bit_ecx(11) {
                    features.insert(Feature::Tbm);
                }
            }
        }
    }
}

/// Run-time detection for `aarch64`.
///
/// On Linux and Android the kernel's `AT_HWCAP`/`AT_HWCAP2` auxiliary vector
/// is authoritative (it already accounts for what the OS lets user space use)
/// and is read through `getauxval`. On every other OS the architectural ID
/// registers are read with `MRS`.
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{Feature, Features};

    /// Fills `features` from the OS or the ID registers.
    pub(super) fn detect(features: &mut Features) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            hwcap(features);
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            // SAFETY: `ID_AA64*` registers are readable from EL0 whenever
            // `SCTLR_EL1.TID0` is clear, which is how all mainstream aarch64
            // OSes (macOS, Windows, BSDs, bare metal) configure the system;
            // reading an ID register has no side effects.
            unsafe { id_registers(features) };
        }
    }

    /// `AT_HWCAP`/`AT_HWCAP2` values from the Linux/Android uapi headers
    /// (stable kernel ABI), defined locally so the constants are identical on
    /// glibc, musl and Android.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn hwcap(features: &mut Features) {
        const AT_HWCAP: libc::c_ulong = 16;
        const AT_HWCAP2: libc::c_ulong = 26;

        const HWCAP_FP: u64 = 1 << 0;
        const HWCAP_ASIMD: u64 = 1 << 1;
        const HWCAP_AES: u64 = 1 << 3;
        const HWCAP_PMULL: u64 = 1 << 4;
        const HWCAP_SHA1: u64 = 1 << 5;
        const HWCAP_SHA2: u64 = 1 << 6;
        const HWCAP_CRC32: u64 = 1 << 7;
        const HWCAP_ATOMICS: u64 = 1 << 8;
        const HWCAP_FPHP: u64 = 1 << 9;
        const HWCAP_ASIMDHP: u64 = 1 << 10;
        const HWCAP_ASIMDDP: u64 = 1 << 20;
        const HWCAP_SVE: u64 = 1 << 22;
        const HWCAP2_SVE2: u64 = 1 << 1;

        // SAFETY: `getauxval` takes no pointers and never fails catastrophically;
        // an unknown key yields zero, which simply leaves features unset.
        let hw = unsafe { libc::getauxval(AT_HWCAP) } as u64;
        let hw2 = unsafe { libc::getauxval(AT_HWCAP2) } as u64;

        if hw & HWCAP_FP != 0 {
            features.insert(Feature::Fp);
        }
        if hw & HWCAP_ASIMD != 0 {
            features.insert(Feature::Neon);
        }
        if hw & HWCAP_AES != 0 {
            features.insert(Feature::Aes);
        }
        if hw & HWCAP_PMULL != 0 {
            features.insert(Feature::Pmull);
            features.insert(Feature::Pclmulqdq);
        }
        if hw & HWCAP_SHA1 != 0 {
            features.insert(Feature::Sha1);
        }
        if hw & HWCAP_SHA2 != 0 {
            features.insert(Feature::Sha2);
        }
        if hw & HWCAP_SHA1 != 0 && hw & HWCAP_SHA2 != 0 {
            // Both halves present: equivalent to x86 SHA-NI for our purposes.
            features.insert(Feature::Sha);
        }
        if hw & HWCAP_CRC32 != 0 {
            features.insert(Feature::Crc32);
        }
        if hw & HWCAP_ATOMICS != 0 {
            features.insert(Feature::Lse);
        }
        if hw & HWCAP_FPHP != 0 || hw & HWCAP_ASIMDHP != 0 {
            features.insert(Feature::Fp16);
        }
        if hw & HWCAP_ASIMDDP != 0 {
            features.insert(Feature::DotProd);
        }
        if hw & HWCAP_SVE != 0 {
            features.insert(Feature::Sve);
        }
        if hw2 & HWCAP2_SVE2 != 0 {
            features.insert(Feature::Sve2);
        }
    }

    /// Reads the architectural ID registers with `MRS`.
    ///
    /// # Safety
    ///
    /// The caller must run on aarch64 in a context where the `ID_AA64*`
    /// registers are readable from EL0 (see the call site).
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    unsafe fn id_registers(features: &mut Features) {
        /// Reads a four-bit ID field of `value` at `shift`.
        const fn field(value: u64, shift: u32) -> u64 {
            (value >> shift) & 0xF
        }

        macro_rules! mrs {
            ($register:literal) => {{
                let value: u64;
                // SAFETY: `MRS` of an ID register only returns the register;
                // it writes no memory, adjusts no stack and leaves the flags
                // alone. The register name is a literal chosen by the macro
                // caller, so the instruction assembles to a valid encoding.
                unsafe {
                    core::arch::asm!(
                        concat!("mrs {0}, ", $register),
                        out(reg) value,
                        options(nostack, nomem, preserves_flags),
                    );
                }
                value
            }};
        }

        let pfr0 = mrs!("ID_AA64PFR0_EL1");
        let fp = field(pfr0, 16);
        let advsimd = field(pfr0, 20);
        let sve = field(pfr0, 32);
        // The FP/AdvSIMD fields are signed: 0 = implemented, 1 = FP16,
        // 0b1111 (-1) = not implemented.
        if fp != 0xF {
            features.insert(Feature::Fp);
        }
        if advsimd != 0xF {
            features.insert(Feature::Neon);
        }
        if (1..=7).contains(&fp) || (1..=7).contains(&advsimd) {
            features.insert(Feature::Fp16);
        }
        let sve_present = sve >= 1;
        if sve_present {
            features.insert(Feature::Sve);
        }

        let isar0 = mrs!("ID_AA64ISAR0_EL1");
        let aes = field(isar0, 4);
        if aes >= 1 {
            features.insert(Feature::Aes);
        }
        if aes >= 2 {
            features.insert(Feature::Pmull);
            features.insert(Feature::Pclmulqdq);
        }
        if field(isar0, 8) >= 1 {
            features.insert(Feature::Sha1);
        }
        if field(isar0, 12) >= 1 {
            features.insert(Feature::Sha2);
        }
        if field(isar0, 8) >= 1 && field(isar0, 12) >= 1 {
            features.insert(Feature::Sha);
        }
        if field(isar0, 16) >= 1 {
            features.insert(Feature::Crc32);
        }
        if field(isar0, 20) >= 2 {
            features.insert(Feature::Lse);
        }
        if field(isar0, 44) >= 1 {
            features.insert(Feature::DotProd);
        }

        if sve_present {
            let zfr0 = mrs!("ID_AA64ZFR0_EL1");
            if field(zfr0, 0) >= 1 {
                features.insert(Feature::Sve2);
            }
        }
    }
}

/// Run-time detection for 32-bit `arm` through `AT_HWCAP`/`AT_HWCAP2`.
///
/// Only Linux and Android are probed: they are the OSes that ship a
/// user-readable auxiliary vector on this architecture. Elsewhere, the
/// compile-time feature set is used.
#[cfg(target_arch = "arm")]
mod arm {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    use super::Feature;
    use super::Features;

    /// Fills `features` from the auxiliary vector, when available.
    ///
    /// On operating systems without an auxiliary vector nothing is probed;
    /// [`super::compile_time_features`] already covers such targets.
    pub(super) fn detect(features: &mut Features) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            hwcap(features);
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = features;
        }
    }

    /// `AT_HWCAP`/`AT_HWCAP2` values from the Linux/Android uapi headers
    /// (stable kernel ABI), defined locally so the constants are identical on
    /// glibc, musl and Android.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn hwcap(features: &mut Features) {
        const AT_HWCAP: libc::c_ulong = 16;
        const AT_HWCAP2: libc::c_ulong = 26;

        const HWCAP_VFP: u64 = 1 << 6;
        const HWCAP_NEON: u64 = 1 << 12;
        const HWCAP_FPHP: u64 = 1 << 22;
        const HWCAP_ASIMDHP: u64 = 1 << 23;
        const HWCAP_ASIMDDP: u64 = 1 << 24;

        const HWCAP2_AES: u64 = 1 << 0;
        const HWCAP2_PMULL: u64 = 1 << 1;
        const HWCAP2_SHA1: u64 = 1 << 2;
        const HWCAP2_SHA2: u64 = 1 << 3;
        const HWCAP2_CRC32: u64 = 1 << 4;

        // SAFETY: `getauxval` takes no pointers and never fails catastrophically;
        // an unknown key yields zero, which simply leaves features unset.
        let hw = unsafe { libc::getauxval(AT_HWCAP) } as u64;
        let hw2 = unsafe { libc::getauxval(AT_HWCAP2) } as u64;

        if hw & HWCAP_VFP != 0 {
            features.insert(Feature::Fp);
        }
        if hw & HWCAP_NEON != 0 {
            features.insert(Feature::Neon);
        }
        if hw & HWCAP_FPHP != 0 || hw & HWCAP_ASIMDHP != 0 {
            features.insert(Feature::Fp16);
        }
        if hw & HWCAP_ASIMDDP != 0 {
            features.insert(Feature::DotProd);
        }
        if hw2 & HWCAP2_AES != 0 {
            features.insert(Feature::Aes);
        }
        if hw2 & HWCAP2_PMULL != 0 {
            features.insert(Feature::Pmull);
            features.insert(Feature::Pclmulqdq);
        }
        if hw2 & HWCAP2_SHA1 != 0 {
            features.insert(Feature::Sha1);
        }
        if hw2 & HWCAP2_SHA2 != 0 {
            features.insert(Feature::Sha2);
        }
        if hw2 & HWCAP2_SHA1 != 0 && hw2 & HWCAP2_SHA2 != 0 {
            features.insert(Feature::Sha);
        }
        if hw2 & HWCAP2_CRC32 != 0 {
            features.insert(Feature::Crc32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn canonical_names_roundtrip() {
        for feature in Feature::ALL {
            assert_eq!(Feature::from_name(feature.name()), Some(*feature));
        }
    }

    #[test]
    fn aliases_resolve_to_canonical_features() {
        assert_eq!(Feature::from_name("sse4.1"), Some(Feature::Sse41));
        assert_eq!(Feature::from_name("sse4.2"), Some(Feature::Sse42));
        assert_eq!(Feature::from_name("crc"), Some(Feature::Crc32));
        assert_eq!(Feature::from_name("atomics"), Some(Feature::Lse));
        assert_eq!(Feature::from_name("asimd"), Some(Feature::Neon));
        assert_eq!(Feature::from_name("vfp"), Some(Feature::Fp));
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert_eq!(Feature::from_name("sse4_3"), None);
        assert_eq!(Feature::from_name(""), None);
        assert_eq!(Feature::from_name("AVX2"), None);
    }

    #[test]
    fn compile_time_features_are_a_subset_of_detected_features() {
        let compile_time = compile_time_features();
        let available = features();
        assert_eq!(
            compile_time.bits() & !available.bits(),
            0,
            "compile-time feature {compile_time:?} not covered by {available:?}"
        );
    }

    #[test]
    fn umbrella_avx512_requires_all_constituents() {
        let mut set = Features::EMPTY;
        assert!(!set.contains(Feature::Avx512));

        set.insert(Feature::Avx512F);
        set.insert(Feature::Avx512Bw);
        set.insert(Feature::Avx512Dq);
        assert!(!set.contains(Feature::Avx512));

        set.insert(Feature::Avx512Vl);
        assert!(set.contains(Feature::Avx512));

        // Inserting the umbrella itself stores every constituent.
        let mut via_umbrella = Features::EMPTY;
        via_umbrella.insert(Feature::Avx512);
        assert!(via_umbrella.contains(Feature::Avx512F));
        assert!(via_umbrella.contains(Feature::Avx512Bw));
        assert!(via_umbrella.contains(Feature::Avx512Dq));
        assert!(via_umbrella.contains(Feature::Avx512Vl));
    }

    #[test]
    fn macros_match_the_api() {
        assert_eq!(cpu_feature!("sse2"), has(Feature::Sse2));
        assert_eq!(cpu_feature!("avx"), has(Feature::Avx));
        assert_eq!(cpu_feature!("sse4.1"), has(Feature::Sse41));
        assert_eq!(cpu_feature!("avx512"), has(Feature::Avx512));
        assert_eq!(cpu_feature_at_time!(sse2), Feature::Sse2.compile_time_available());
        assert_eq!(
            cpu_feature_at_time!("crc32"),
            Feature::Crc32.compile_time_available()
        );
    }

    const COMPILE_TIME_SSE2: bool = cpu_feature_at_time!(sse2);

    #[test]
    fn compile_time_macro_is_const_usable() {
        assert_eq!(COMPILE_TIME_SSE2, cfg!(target_feature = "sse2"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_64_baseline_features_are_detected() {
        // Every x86_64 CPU and OS provides SSE2, and Rust enables it in the
        // target baseline, so both layers must agree it is available.
        assert!(cpu_feature!(sse2));
        assert!(has(Feature::Sse2));
        assert!(compile_time_features().contains(Feature::Sse2));
        // The raw probe must find something on any x86_64 machine.
        assert!(!detect().is_empty());
        assert!(detect().contains(Feature::Sse));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_baseline_features_are_detected() {
        let detected = detect();
        assert!(!detected.is_empty());
        assert!(detected.contains(Feature::Fp));
        assert!(detected.contains(Feature::Neon));
    }

    #[test]
    fn debug_output_lists_set_features() {
        let mut set = Features::EMPTY;
        set.insert(Feature::Sse2);
        set.insert(Feature::Aes);
        assert_eq!(format!("{set:?}"), "Features(sse2, aes)");
    }
}
