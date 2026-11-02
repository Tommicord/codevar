# Codevar

Codevar is a code editor built from the ground up in Rust, meant to run anywhere a
program can run: a browser tab, a Wayland or Win32 desktop, a phone, or a Cortex-M
microcontroller. It owns its stack instead of binding to the platform: the Wayland
client, the TrueType rasterizer, the OpenCL implementation, the input layer and the
single-core scheduler all live in this repository.

[![CI](https://github.com/Tommicord/codevar/actions/workflows/rust.yml/badge.svg)](https://github.com/Tommicord/codevar/actions/workflows/rust.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![MSRV](https://img.shields.io/badge/MSRV-1.93.0-orange.svg)

## Goals

- **Run everywhere.** The same crates target wasm32, Linux/Wayland, Windows, macOS,
  bare-metal Cortex-M and accelerator-backed devices. Portable code goes through
  `codevar-target`, `codevar-io` and `codevar-base` instead of `std`.
- **Own the stack.** A self-hosted Wayland client and server, a TrueType rasterizer
  with autofit and pshinter hinting plus color emoji, an OpenCL ICD runtime next to
  a statically linked bare-metal driver, and a scheduler for both instruction order
  and single-core async tasks.
- **Stay fast on modest hardware.** `no_std`-first crates, TLSF allocation, SIMD
  kernels and a display-list compositor: the editor should be usable on hardware
  that could not run a browser-based editor.
- **Be agent-ready.** The architecture is being prepared for AI agents as
  first-class clients of the editor, not as a bolt-on panel.
- **No install.** The web target is meant to run from a URL, with no installer.

## Status

63 crates. Working pieces, partial implementations and scaffolds are labelled so
the table is not mistaken for a feature list.

| Area                           | State                                                                                                                                                                                                                                                                                                                                                        |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Vulkan compositor              | **Working.** Offscreen dma-buf targets, multi-pipe compositing and a display-list renderer, presented through `zwp_linux_dmabuf_v1`. Three runnable examples (below).                                                                                                                                                                                        |
| WebGPU backend                 | Mirrors the Vulkan backend primitive for primitive on `wgpu-hal`. wasm32 only; other targets compile a stub.                                                                                                                                                                                                                                                 |
| Kuitui UI framework            | Retained-mode, Compose-inspired widget tree with layout and a backend-neutral display list. Canvas and text widgets exist.                                                                                                                                                                                                                                   |
| Text stack                     | Own TrueType implementation (`sfnt`, `glyf`, `smooth`, `autofit`, `pshinter`, `psnames`, color emoji). Glyph coverage is produced by a `font_raster` kernel embedded as SPIR-V and run on the in-tree OpenCL runtime.                                                                                                                                        |
| OpenCL                         | Lexer, parser, semantic analysis, IR, SPIR-V assembler, cubin/ELF reader, and `codevar-oclc`, a compiler with its own machine-code back ends for x86-64, AArch64 and ARMv7-M. Host runtime loads vendor ICDs at run time; `codevar-ocl-bare` is a statically linked embedded-profile driver. `--emit ptx` and `--emit msl` are accepted but not implemented. |
| Editor core (`codevar-wredit`) | Rope storage, edit/change model and SIMD helpers, `no_std`. Not yet connected to a window.                                                                                                                                                                                                                                                                   |
| Input                          | Linux evdev backend enumerates and classifies devices and streams events through cooperative tasks. Windows Raw Input, macOS event taps and browser DOM backends are scaffolds.                                                                                                                                                                              |
| Platform                       | Wayland client (no libwayland), D-Bus, XDG desktop portals, Cortex-M runtime and semihosting, TLS, WebSocket, TLSF allocator, unwinding, logging, portable I/O.                                                                                                                                                                                              |
| Launcher (`codevar`)           | Opens and pumps a native window (Wayland on Linux; a Win32 backend exists) and installs process hooks. No editor UI inside it yet.                                                                                                                                                                                                                           |

### What's next

1. Wire `codevar-wredit` and the Kuitui widgets into a launcher window so the
   project becomes an editor you can type in.
2. Replace the scaffolded input backends with real implementations.
3. Get the WebGPU backend building and running in a browser.
4. Add AI agents as first-class clients of the editor.

## Try it

Prerequisites:

- Rust from `rust-toolchain.toml` (currently 1.93.0).
- Linux with a Wayland compositor and a Vulkan driver exposing
  `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_semaphore_fd` and
  `VK_EXT_image_drm_format_modifier` (Mesa drivers do).
- For the text example: an OpenCL ICD loader (`libOpenCL.so.1`), such as Mesa's
  rusticl or clover.

```bash
# Vulkan compositor demos: each opens a 640x480 window, renders a fixed number
# of frames, prints a summary and exits on its own.
cargo run -p codevar-gpu-vk --example render_hit   # a triangle presented through dma-buf
cargo run -p codevar-gpu-vk --example render_mix   # three compositor pipes mixed
cargo run -p codevar-gpu-vk --example hello_text   # "Hello World!" with color emoji

# The launcher window (Wayland on Linux; a Win32 backend exists)
cargo run -p codevar-launcher --bin codevar

# The OpenCL compiler
cargo run -p codevar-oclc --bin codevar-oclc -- --help
```

The OpenCL dialect is Rust-shaped. Save this as `vector_add.cl`:

```rust
#[kernel]
fn vector_add(a: *const float, b: *const float, c: *mut float, n: int) -> void {
    let i = get_global_id(0);
    if i < n {
        c[i] = a[i] + b[i];
    }
}
```

```bash
# SPIR-V module
cargo run -p codevar-oclc --bin codevar-oclc -- --emit spirv -o vector_add.spv vector_add.cl

# Native machine code for the host, with the optimizer at -O3
cargo run -p codevar-oclc --bin codevar-oclc -- --emit native -O3 -o vector_add.bin vector_add.cl
```

## Repository layout

| Layer                      | Crates                                                                                                                                                                                                                                                                                                                                |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Editor                     | `codevar-wredit` (rope storage, edits, SIMD), `codevar-textlike-encode`                                                                                                                                                                                                                                                               |
| UI                         | `codevar-kuitui`, `codevar-kuitui-text`, `codevar-kuitui-widget`, `codevar-kuitui-control` plus the `-linux`, `-windows`, `-macos` and `-wasm` input backends                                                                                                                                                                         |
| Graphics                   | `codevar-gpu-core` (backend-neutral traits), `codevar-gpu-vk`, `codevar-gpu-wg`                                                                                                                                                                                                                                                       |
| Fonts                      | `codevar-truetype-{core,sfnt,glyf,smooth,autofit,pshinter,psnames,gzip,emoji-color}`                                                                                                                                                                                                                                                  |
| OpenCL                     | `codevar-ocl-{lex,parse,sar,ir,asm,cubin,abi,tc,native-codegen,native-codegen-x86,native-codegen-aarch64,native-codegen-thumbv7m}`, `codevar-ocl` (ICD runtime), `codevar-ocl-bare` (embedded driver), `codevar-oclc` (compiler CLI)                                                                                                  |
| Platform                   | `codevar-wayland`, `codevar-dbus`, `codevar-xdg-portal`, `codevar-target`, `codevar-rt`, `codevar-env`, `codevar-pathbuf`                                                                                                                                                                                                             |
| Runtime and infrastructure | `codevar-inssched`, `codevar-tlsf-alloc`, `codevar-logger`, `codevar-io`, `codevar-console-util`, `codevar-time-core`, `codevar-time-util`, `codevar-cli-parse`, `codevar-base`, `codevar-packlow`, `codevar-mergen`, `codevar-shake`, `codevar-sig-handler`, `codevar-sig-module-base`, `codevar-unwinding`, `codevar-pretty-unwind` |
| Networking                 | `codevar-tls`, `codevar-wsocket`                                                                                                                                                                                                                                                                                                      |
| Launcher                   | `codevar-launcher` (the `codevar` binary)                                                                                                                                                                                                                                                                                             |

The flow between layers is deliberate: Kuitui computes a widget tree, layout and a
flat display list; `codevar-gpu-core` defines the traits a backend implements; the
Vulkan and WebGPU backends consume the same display list. Input travels the other
way through `codevar-kuitui-control` and a platform backend.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

CI (`.github/workflows/rust.yml`) runs these on every push and pull request to
`main`. While working, scope commands to the crates you touched; a full-workspace
check costs minutes while a per-crate check costs seconds:

```bash
cargo check  -p codevar-tls -p codevar-io
cargo clippy -p codevar-tls -p codevar-io --all-targets -- -D warnings
cargo test   -p codevar-tls -p codevar-io
```

The engineering standards this project holds itself to — error handling without
panics, `no_std`-first modules, unsafe-code rules, commit size limits — are in
[CODE_QUALITY.md](CODE_QUALITY.md), and the operating rules for AI agents are in
[AGENTS.md](AGENTS.md).

## Contributing

Contributions are welcome! Please read the [Contributing Guide](CONTRIBUTING.md)
and [Code of Conduct](CODE_OF_CONDUCT.md) before getting started. To report a
security vulnerability, please see the [Security Policy](SECURITY.md) rather than
opening a public issue.

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
for the full license text.
