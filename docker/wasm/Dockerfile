# Copyright 2026 Codevar
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Dockerfile for WebAssembly builds
FROM rust:1.93.0-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./
COPY codevar-core ./codevar-core
COPY codevar-colab ./codevar-colab
COPY codevar-fparser ./codevar-fparser

# Install wasm-pack
RUN cargo install wasm-pack

# Add WASM target
RUN rustup target add wasm32-unknown-unknown

# Build WASM packages
RUN cd codevar-core && wasm-pack build --release --out-dir ../../pkg/codevar-core
RUN cd codevar-colab && wasm-pack build --release --out-dir ../../pkg/codevar-colab
RUN cd codevar-fparser && wasm-pack build --release --out-dir ../../pkg/codevar-fparser

# Runtime stage
FROM node:20-slim

WORKDIR /app

# Copy WASM packages
COPY --from=builder /app/pkg ./pkg

CMD ["node"]