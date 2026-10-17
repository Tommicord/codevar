# syntax=docker/dockerfile:1

# Codevar deployment image (Linux, glibc).
#
# Builds the `codevar` and `codevar-oclc` executables and ships only those
# plus the bundled assets in a slim runtime image. The build stage compiles
# for the container's own platform, so every architecture Docker can run is
# a one-flag build:
#
#   docker buildx build --platform linux/arm64 -t codevar:arm64 .
#
# Platforms that are not Linux, or that need a cross toolchain (Windows,
# macOS, iOS, Android, FreeBSD, NetBSD, Hermit, bare metal, wasm, static
# musl) are built with the Dockerfiles in `docker/`; `docker/build.sh` is
# the one-command driver for all of them.
#
# Stages:
#   builder    — compile the executables
#   artifacts  — export-only scratch stage (used by docker/build.sh)
#   runtime    — slim deployable image (default target)

ARG RUST_VERSION=1.93
ARG DEBIAN_VERSION=bookworm

FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS builder
WORKDIR /usr/src/codevar

# Sources first and in full; there is no C tooling in the dependency graph,
# so no apt packages are needed (the rust image ships the host linker).
# BuildKit cache mounts keep the registry and build cache across runs.
COPY . .

RUN --mount=type=cache,id=cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=cargo-target-linux,target=/usr/src/codevar/target \
    cargo build --locked --release -p codevar-launcher -p codevar-oclc \
 && mkdir -p /out \
 && cp target/release/codevar /out/codevar \
 && cp target/release/codevar-oclc /out/codevar-oclc

# Export-only stage:
#   docker build --target artifacts --output type=local,dest=dist .
FROM scratch AS artifacts
COPY --from=builder /out/ /

# Runtime: slim Debian, both executables on PATH, assets where
# `codevar_env::ENV_ASSETS_DIR` resolves them, running unprivileged.
FROM debian:${DEBIAN_VERSION}-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --create-home --home-dir /data/codevar \
      --shell /usr/sbin/nologin codevar

COPY --from=builder /out/ /usr/local/bin/
COPY assets /data/codevar/assets

USER codevar
WORKDIR /data/codevar

ENTRYPOINT ["codevar"]
